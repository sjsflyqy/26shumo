from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
import torch


SYMBOLS = ("T", "A", "V")
NAMES = {"T": "text", "A": "audio", "V": "vision"}


@dataclass(frozen=True)
class EvidenceWindow:
    modality: str
    start: int
    end: int
    internal_score: float = 0.0
    occlusion_score: float = 0.0
    probability_drop: float = 0.0

    @property
    def symbol(self) -> str:
        return {value: key for key, value in NAMES.items()}[self.modality]


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def is_unaligned(batch: dict[str, Any]) -> bool:
    return "audio_reliability_mask" in batch


def validity_mask(batch: dict[str, Any], symbol: str) -> torch.Tensor:
    if symbol == "T":
        return batch["content_mask"].bool()
    if is_unaligned(batch):
        return batch[f"{NAMES[symbol]}_structural_mask"].bool()
    return batch["content_mask"].bool()


def _selection(batch: dict[str, Any], symbol: str, start: int, end: int) -> torch.Tensor:
    valid = validity_mask(batch, symbol)
    selected = torch.zeros_like(valid)
    selected[:, max(0, start) : max(0, end)] = True
    return selected & valid


def _remove_selection(
    output: dict[str, Any], symbol: str, selected: torch.Tensor
) -> None:
    if symbol == "T":
        output["input_ids"][selected] = 0
        output["token_type_ids"][selected] = 0
        output["text_attention_mask"] = output["text_attention_mask"].bool() & ~selected
        if "reliability_mask" in output:
            output["reliability_mask"][..., 0] &= ~selected
        return
    name = NAMES[symbol]
    output[name][selected] = 0.0
    if is_unaligned(output):
        key = f"{name}_reliability_mask"
        output[key] = output[key].bool() & ~selected
    else:
        modality_index = 1 if symbol == "A" else 2
        output["reliability_mask"][..., modality_index] &= ~selected


def mask_window(
    batch: dict[str, Any], symbol: str, start: int, end: int
) -> dict[str, Any]:
    output = clone_batch(batch)
    _remove_selection(output, symbol, _selection(output, symbol, start, end))
    return output


def mask_modality(batch: dict[str, Any], symbol: str) -> dict[str, Any]:
    output = clone_batch(batch)
    _remove_selection(output, symbol, validity_mask(output, symbol))
    return output


def delete_windows(
    batch: dict[str, Any], windows: Iterable[EvidenceWindow]
) -> dict[str, Any]:
    output = clone_batch(batch)
    for window in windows:
        selected = _selection(output, window.symbol, window.start, window.end)
        _remove_selection(output, window.symbol, selected)
    return output


def keep_only_windows(
    batch: dict[str, Any], windows: Iterable[EvidenceWindow]
) -> dict[str, Any]:
    original = clone_batch(batch)
    output = clone_batch(batch)
    for symbol in SYMBOLS:
        _remove_selection(output, symbol, validity_mask(output, symbol))
    for window in windows:
        symbol = window.symbol
        selected = _selection(original, symbol, window.start, window.end)
        if symbol == "T":
            output["input_ids"][selected] = original["input_ids"][selected]
            output["token_type_ids"][selected] = original["token_type_ids"][selected]
            output["text_attention_mask"][selected] = original["text_attention_mask"][selected]
            if "reliability_mask" in output:
                output["reliability_mask"][..., 0][selected] = (
                    original["reliability_mask"][..., 0][selected]
                )
        else:
            name = NAMES[symbol]
            output[name][selected] = original[name][selected]
            if is_unaligned(output):
                key = f"{name}_reliability_mask"
                output[key][selected] = original[key][selected]
            else:
                index = 1 if symbol == "A" else 2
                output["reliability_mask"][..., index][selected] = (
                    original["reliability_mask"][..., index][selected]
                )
    return output


@torch.no_grad()
def predict_variants(
    model: torch.nn.Module,
    variants: list[dict[str, Any]],
    batch_size: int = 64,
) -> list[dict[str, torch.Tensor]]:
    results: list[dict[str, torch.Tensor]] = []
    for begin in range(0, len(variants), batch_size):
        chunk = variants[begin : begin + batch_size]
        keys = [key for key, value in chunk[0].items() if torch.is_tensor(value)]
        combined = {key: torch.cat([item[key] for item in chunk], dim=0) for key in keys}
        output = model(combined)
        probabilities = torch.softmax(output["class_logits"], dim=-1)
        for index in range(len(chunk)):
            results.append(
                {
                    "class_logits": output["class_logits"][index],
                    "probabilities": probabilities[index],
                    "intensity": output["intensity"][index],
                }
            )
    return results


def internal_importance(
    output: dict[str, torch.Tensor], batch: dict[str, Any]
) -> dict[str, np.ndarray]:
    temporal = output["temporal_weights"][0]
    gates = output["modality_weights"][0]
    text = temporal * gates[:, 0]
    audio_query = temporal * gates[:, 1]
    vision_query = temporal * gates[:, 2]
    audio = torch.einsum("t,tj->j", audio_query, output["audio_attention"][0])
    vision = torch.einsum("t,tj->j", vision_query, output["vision_attention"][0])
    curves = {"text": text, "audio": audio, "vision": vision}
    for symbol, name in NAMES.items():
        valid = validity_mask(batch, symbol)[0]
        curves[name] = curves[name] * valid.to(curves[name].dtype)
    return {key: value.detach().float().cpu().numpy() for key, value in curves.items()}


def candidate_windows(
    batch: dict[str, Any],
    curves: dict[str, np.ndarray],
    window_rate: float = 0.15,
    stride_rate: float = 0.5,
) -> list[EvidenceWindow]:
    candidates = []
    for symbol, name in NAMES.items():
        positions = torch.nonzero(validity_mask(batch, symbol)[0], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        begin, stop = int(positions[0]), int(positions[-1]) + 1
        length = max(1, stop - begin)
        window = max(1, min(length, int(round(length * window_rate))))
        stride = max(1, int(round(window * stride_rate)))
        maximum = stop - window
        starts = list(range(begin, maximum + 1, stride))
        if not starts or starts[-1] != maximum:
            starts.append(maximum)
        for start in starts:
            end = start + window
            candidates.append(
                EvidenceWindow(
                    modality=name,
                    start=start,
                    end=end,
                    internal_score=float(np.asarray(curves[name])[start:end].sum()),
                )
            )
    return candidates


@torch.no_grad()
def score_occlusion_windows(
    model: torch.nn.Module,
    batch: dict[str, Any],
    base_output: dict[str, torch.Tensor],
    candidates: list[EvidenceWindow],
    batch_size: int = 64,
) -> list[EvidenceWindow]:
    predicted_class = int(base_output["class_logits"][0].argmax())
    base_logit = float(base_output["class_logits"][0, predicted_class])
    base_probability = float(torch.softmax(base_output["class_logits"], dim=-1)[0, predicted_class])
    variants = [mask_window(batch, item.symbol, item.start, item.end) for item in candidates]
    predictions = predict_variants(model, variants, batch_size=batch_size)
    scored = []
    for item, prediction in zip(candidates, predictions):
        scored.append(
            replace(
                item,
                occlusion_score=base_logit - float(prediction["class_logits"][predicted_class]),
                probability_drop=(
                    base_probability - float(prediction["probabilities"][predicted_class])
                ),
            )
        )
    return scored


def select_non_overlapping(
    windows: Iterable[EvidenceWindow],
    top_k: int,
    score: str = "occlusion_score",
) -> list[EvidenceWindow]:
    selected: list[EvidenceWindow] = []
    ordered = sorted(windows, key=lambda item: getattr(item, score), reverse=True)
    for candidate in ordered:
        overlap = any(
            candidate.modality == item.modality
            and candidate.start < item.end
            and item.start < candidate.end
            for item in selected
        )
        if not overlap:
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


@torch.no_grad()
def modality_counterfactuals(
    model: torch.nn.Module,
    batch: dict[str, Any],
    base_output: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    predicted_class = int(base_output["class_logits"][0].argmax())
    base_logit = float(base_output["class_logits"][0, predicted_class])
    predictions = predict_variants(model, [mask_modality(batch, symbol) for symbol in SYMBOLS])
    deltas = torch.tensor(
        [base_logit - float(item["class_logits"][predicted_class]) for item in predictions]
    )
    contributions = torch.softmax(deltas, dim=0)
    delta_map = {NAMES[symbol]: float(value) for symbol, value in zip(SYMBOLS, deltas)}
    contribution_map = {
        NAMES[symbol]: float(value) for symbol, value in zip(SYMBOLS, contributions)
    }
    return delta_map, contribution_map


def random_matched_windows(
    batch: dict[str, Any],
    reference: list[EvidenceWindow],
    rng: np.random.Generator,
) -> list[EvidenceWindow]:
    output = []
    for item in reference:
        valid = torch.nonzero(validity_mask(batch, item.symbol)[0], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        first, stop = int(valid[0]), int(valid[-1]) + 1
        length = min(item.end - item.start, stop - first)
        start = int(rng.integers(first, max(first + 1, stop - length + 1)))
        output.append(EvidenceWindow(item.modality, start, start + length))
    return output


@torch.no_grad()
def faithfulness_scores(
    model: torch.nn.Module,
    batch: dict[str, Any],
    base_output: dict[str, torch.Tensor],
    selected: list[EvidenceWindow],
    random_windows: list[EvidenceWindow],
) -> dict[str, float]:
    predicted_class = int(base_output["class_logits"][0].argmax())
    base_probability = float(torch.softmax(base_output["class_logits"], dim=-1)[0, predicted_class])
    variants = [delete_windows(batch, selected), keep_only_windows(batch, selected)]
    variants.extend(delete_windows(batch, selected[:index]) for index in range(1, len(selected) + 1))
    variants.append(delete_windows(batch, random_windows))
    variants.extend(
        delete_windows(batch, random_windows[:index])
        for index in range(1, len(random_windows) + 1)
    )
    predictions = predict_variants(model, variants)
    probabilities = [float(item["probabilities"][predicted_class]) for item in predictions]
    selected_steps = probabilities[2 : 2 + len(selected)]
    random_full_index = 2 + len(selected)
    random_steps = probabilities[random_full_index + 1 :]
    return {
        "comprehensiveness": base_probability - probabilities[0],
        "sufficiency": base_probability - probabilities[1],
        "aopc": float(np.mean([base_probability - value for value in selected_steps])),
        "random_comprehensiveness": base_probability - probabilities[random_full_index],
        "random_aopc": float(np.mean([base_probability - value for value in random_steps])),
    }
