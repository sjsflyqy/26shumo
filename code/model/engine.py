from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SpanMaskConfig
from .masking import corrupt_aligned_batch, teacher_emotion_importance
from .metrics import classification_regression_metrics
from .model import M2Model, update_ema_teacher


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def inverse_frequency_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: M2Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    objective: torch.nn.Module,
    mask_config: SpanMaskConfig,
    device: torch.device,
    generator: torch.Generator,
    clip_grad_norm: float = 1.0,
    teacher: M2Model | None = None,
    teacher_ema_decay: float = 0.996,
    emotion_mask_probability: float = 0.0,
    emotion_mask_temperature: float = 0.2,
    emotion_intensity_weight: float = 0.25,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()
    totals: dict[str, float] = defaultdict(float)
    samples = 0
    for cpu_batch in loader:
        clean = move_to_device(cpu_batch, device)
        importance = None
        teacher_output = None
        if teacher is not None:
            if emotion_mask_probability > 0:
                with torch.enable_grad():
                    guided_output = teacher(clean, capture_saliency=True)
                    importance = teacher_emotion_importance(
                        guided_output, clean, intensity_weight=emotion_intensity_weight
                    )
                teacher_output = {
                    key: tuple(value.detach() for value in item)
                    if isinstance(item, tuple) else item.detach()
                    for key, item in guided_output.items()
                    if key in {"class_logits", "intensity", "fusion_feature", "latent_targets"}
                }
                del guided_output
            else:
                with torch.no_grad():
                    teacher_output = teacher(clean)
        masked_cpu = corrupt_aligned_batch(
            cpu_batch,
            mask_config,
            generator,
            importance=importance,
            emotion_probability=emotion_mask_probability,
            emotion_temperature=emotion_mask_temperature,
        )
        masked = move_to_device(masked_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        clean_output = model(clean)
        masked_output = model(masked)
        losses = objective(
            clean_output,
            masked_output,
            clean["class_label"],
            clean["regression_label"],
            teacher_output=teacher_output,
            clean_batch=clean,
            masked_batch=masked,
        )
        losses["loss"].backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        if teacher is not None:
            update_ema_teacher(model, teacher, teacher_ema_decay)
        batch_size = int(clean["class_label"].shape[0])
        samples += batch_size
        for key, value in losses.items():
            totals[key] += float(value.detach()) * (1 if key == "reconstructed_positions" else batch_size)
    return {key: value / max(samples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mask_config: SpanMaskConfig | None = None,
    seed: int = 0,
    mask_rate: float | None = None,
    mask_modalities: str | None = None,
    mask_location: str | None = None,
    mask_spans: int | None = None,
    mask_sync: bool | None = None,
) -> dict[str, float]:
    model.eval()
    class_truth: list[np.ndarray] = []
    class_prediction: list[np.ndarray] = []
    regression_truth: list[np.ndarray] = []
    regression_prediction: list[np.ndarray] = []
    generator = torch.Generator().manual_seed(seed)
    for cpu_batch in loader:
        if mask_config is not None:
            cpu_batch = corrupt_aligned_batch(
                cpu_batch,
                mask_config,
                generator,
                force_rate=mask_rate,
                force_modalities=mask_modalities,
                force_location=mask_location,
                force_spans=mask_spans,
                force_sync=mask_sync,
            )
        batch = move_to_device(cpu_batch, device)
        output = model(batch)
        class_truth.append(batch["class_label"].cpu().numpy())
        class_prediction.append(output["class_logits"].argmax(dim=-1).cpu().numpy())
        regression_truth.append(batch["regression_label"].cpu().numpy())
        regression_prediction.append(output["intensity"].cpu().numpy())
    return classification_regression_metrics(
        np.concatenate(class_truth),
        np.concatenate(class_prediction),
        np.concatenate(regression_truth),
        np.concatenate(regression_prediction),
    )
