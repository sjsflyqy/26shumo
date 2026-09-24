from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(obj).__name__}")
    return obj


def _envelope_mask(observed: np.ndarray) -> np.ndarray:
    """Return positions up to the final observed text token.

    Internal zeros remain structurally valid and can therefore be represented as
    genuine local missing positions instead of being confused with tail padding.
    """

    observed = np.asarray(observed, dtype=bool)
    structural = np.zeros_like(observed)
    nonzero = np.flatnonzero(observed)
    if nonzero.size:
        structural[: nonzero[-1] + 1] = True
    return structural


def _nonzero_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.any(np.abs(values) > eps, axis=-1)


@dataclass
class AVStandardizer:
    audio_mean: np.ndarray
    audio_scale: np.ndarray
    vision_mean: np.ndarray
    vision_scale: np.ndarray

    @staticmethod
    def _fit_modality(values: np.ndarray, structural_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_dim = values.shape[-1]
        total = np.zeros(feature_dim, dtype=np.float64)
        total_sq = np.zeros(feature_dim, dtype=np.float64)
        count = 0
        for begin in range(0, len(values), 256):
            block = np.asarray(values[begin : begin + 256], dtype=np.float64)
            structural_block = np.asarray(structural_mask[begin : begin + 256], dtype=bool)
            observed = _nonzero_rows(block) & structural_block
            selected = block[observed]
            if selected.size:
                total += selected.sum(axis=0)
                total_sq += np.square(selected).sum(axis=0)
                count += selected.shape[0]
        if count == 0:
            raise ValueError("Cannot fit a standardizer without observed rows")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    @classmethod
    def fit(cls, train_split: Mapping[str, Any]) -> "AVStandardizer":
        text_attention = np.asarray(train_split["text_bert"])[:, 1, :] > 0
        audio = np.asarray(train_split["audio"])
        vision = np.asarray(train_split["vision"])
        if audio.shape[1] == text_attention.shape[1]:
            audio_structural = text_attention
        else:
            audio_structural = _split_structural_mask(train_split, "audio")
        if vision.shape[1] == text_attention.shape[1]:
            vision_structural = text_attention
        else:
            vision_structural = _split_structural_mask(train_split, "vision")
        audio_mean, audio_scale = cls._fit_modality(
            audio, audio_structural
        )
        vision_mean, vision_scale = cls._fit_modality(
            vision, vision_structural
        )
        return cls(audio_mean, audio_scale, vision_mean, vision_scale)

    def transform(self, values: np.ndarray, modality: str, observed: np.ndarray) -> np.ndarray:
        if modality == "audio":
            mean, scale = self.audio_mean, self.audio_scale
        elif modality == "vision":
            mean, scale = self.vision_mean, self.vision_scale
        else:
            raise ValueError(f"Unknown modality: {modality}")
        output = (np.asarray(values, dtype=np.float32) - mean) / scale
        output[~observed] = 0.0
        return output

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "audio_mean": self.audio_mean.tolist(),
            "audio_scale": self.audio_scale.tolist(),
            "vision_mean": self.vision_mean.tolist(),
            "vision_scale": self.vision_scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "AVStandardizer":
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in state.items()})


def _split_structural_mask(split: Mapping[str, Any], modality: str) -> np.ndarray:
    """Return the valid temporal envelope for an unaligned A/V split.

    Some vision sequences contain real internal all-zero frames, so the explicit
    length alone is not always the last occupied array index.  Taking the larger
    of the declared length and final observed index preserves those frames while
    still excluding right padding.
    """

    values = np.asarray(split[modality])
    observed = _nonzero_rows(values)
    any_observed = observed.any(axis=1)
    reverse_first = np.argmax(observed[:, ::-1], axis=1)
    extents = np.where(any_observed, observed.shape[1] - reverse_first, 0)
    length_key = f"{modality}_lengths"
    if length_key in split:
        declared = np.asarray(split[length_key], dtype=np.int64).reshape(-1)
        extents = np.maximum(extents, declared)
    extents = np.clip(extents, 0, values.shape[1])
    return np.arange(values.shape[1])[None, :] < extents[:, None]


def infer_alignment(split: Mapping[str, Any]) -> str:
    text_length = int(np.asarray(split["text_bert"]).shape[-1])
    audio_length = int(np.asarray(split["audio"]).shape[1])
    vision_length = int(np.asarray(split["vision"]).shape[1])
    return "aligned" if audio_length == vision_length == text_length else "unaligned"


def _prepare_text(split: Mapping[str, Any], index: int) -> dict[str, np.ndarray]:
    text_bert = np.asarray(split["text_bert"][index])
    if text_bert.ndim != 2 or text_bert.shape[0] != 3:
        raise ValueError(f"Expected text_bert shape (3, L), got {text_bert.shape}")
    input_ids = np.rint(text_bert[0]).astype(np.int64)
    text_observed = text_bert[1] > 0
    token_type_ids = np.rint(text_bert[2]).astype(np.int64)
    text_structural = _envelope_mask(text_observed)
    content_mask = text_structural.copy()
    structural_positions = np.flatnonzero(text_structural)
    if structural_positions.size:
        content_mask[structural_positions[0]] = False
        content_mask[structural_positions[-1]] = False
    return {
        "input_ids": input_ids,
        "text_observed": text_observed,
        "token_type_ids": token_type_ids,
        "text_structural": text_structural,
        "content_mask": content_mask,
    }


def _add_labels_and_id(
    sample: dict[str, Any],
    split: Mapping[str, Any],
    index: int,
    fallback_id: str | None,
) -> dict[str, Any]:
    ids = split.get("id")
    sample["sample_id"] = fallback_id or (str(ids[index]) if ids is not None else str(index))
    if "classification_labels" in split:
        sample["class_label"] = torch.tensor(
            int(np.asarray(split["classification_labels"])[index]), dtype=torch.long
        )
    if "regression_labels" in split:
        sample["regression_label"] = torch.tensor(
            float(np.asarray(split["regression_labels"])[index]), dtype=torch.float32
        )
    return sample


def prepare_aligned_sample(
    split: Mapping[str, Any],
    index: int,
    standardizer: AVStandardizer,
    *,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    text = _prepare_text(split, index)
    input_ids = text["input_ids"]
    text_observed = text["text_observed"]
    token_type_ids = text["token_type_ids"]
    text_structural = text["text_structural"]

    audio_raw = np.asarray(split["audio"][index], dtype=np.float32)
    vision_raw = np.asarray(split["vision"][index], dtype=np.float32)
    if audio_raw.shape != (50, 74) or vision_raw.shape != (50, 35):
        raise ValueError(
            f"Aligned sample must contain audio (50,74) and vision (50,35), "
            f"got {audio_raw.shape} and {vision_raw.shape}"
        )
    audio_observed = text_structural & _nonzero_rows(audio_raw)
    vision_observed = text_structural & _nonzero_rows(vision_raw)
    audio = standardizer.transform(audio_raw, "audio", audio_observed)
    vision = standardizer.transform(vision_raw, "vision", vision_observed)

    # Exclude [CLS] and final [SEP] from artificial corruption and missing-rate
    # denominators. They are legitimate text tokens but structural A/V zero rows.
    content_mask = text["content_mask"]

    reliability = np.stack((text_observed, audio_observed, vision_observed), axis=-1)
    sample: dict[str, Any] = {
        "input_ids": torch.from_numpy(input_ids),
        "text_attention_mask": torch.from_numpy(text_observed.astype(np.bool_)),
        "token_type_ids": torch.from_numpy(token_type_ids),
        "text_structural_mask": torch.from_numpy(text_structural),
        "content_mask": torch.from_numpy(content_mask),
        "audio": torch.from_numpy(audio),
        "vision": torch.from_numpy(vision),
        "reliability_mask": torch.from_numpy(reliability),
    }
    return _add_labels_and_id(sample, split, index, fallback_id)


def prepare_unaligned_sample(
    split: Mapping[str, Any],
    index: int,
    standardizer: AVStandardizer,
    *,
    fallback_id: str | None = None,
) -> dict[str, Any]:
    """Prepare one sample while preserving independent T/A/V time axes."""

    text = _prepare_text(split, index)
    audio_raw = np.asarray(split["audio"][index], dtype=np.float32)
    vision_raw = np.asarray(split["vision"][index], dtype=np.float32)
    if audio_raw.ndim != 2 or audio_raw.shape[1] != 74:
        raise ValueError(f"Expected unaligned audio shape (T, 74), got {audio_raw.shape}")
    if vision_raw.ndim != 2 or vision_raw.shape[1] != 35:
        raise ValueError(f"Expected unaligned vision shape (T, 35), got {vision_raw.shape}")

    def modality_masks(values: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
        observed = _nonzero_rows(values)
        occupied = np.flatnonzero(observed)
        extent = int(occupied[-1] + 1) if occupied.size else 0
        length_key = f"{modality}_lengths"
        if length_key in split:
            declared = int(np.asarray(split[length_key]).reshape(-1)[index])
            extent = max(extent, declared)
        extent = min(max(extent, 0), values.shape[0])
        structural = np.arange(values.shape[0]) < extent
        return structural, observed & structural

    audio_structural, audio_observed = modality_masks(audio_raw, "audio")
    vision_structural, vision_observed = modality_masks(vision_raw, "vision")
    audio = standardizer.transform(audio_raw, "audio", audio_observed)
    vision = standardizer.transform(vision_raw, "vision", vision_observed)
    sample: dict[str, Any] = {
        "input_ids": torch.from_numpy(text["input_ids"]),
        "text_attention_mask": torch.from_numpy(text["text_observed"].astype(np.bool_)),
        "token_type_ids": torch.from_numpy(text["token_type_ids"]),
        "text_structural_mask": torch.from_numpy(text["text_structural"]),
        "content_mask": torch.from_numpy(text["content_mask"]),
        "audio": torch.from_numpy(audio),
        "vision": torch.from_numpy(vision),
        "audio_structural_mask": torch.from_numpy(audio_structural),
        "vision_structural_mask": torch.from_numpy(vision_structural),
        "audio_reliability_mask": torch.from_numpy(audio_observed),
        "vision_reliability_mask": torch.from_numpy(vision_observed),
    }
    return _add_labels_and_id(sample, split, index, fallback_id)


class AlignedMoseiDataset(Dataset):
    def __init__(
        self,
        split: Mapping[str, Any],
        standardizer: AVStandardizer,
        max_samples: int | None = None,
    ) -> None:
        self.split = split
        self.standardizer = standardizer
        self.size = len(split["audio"])
        if max_samples is not None:
            self.size = min(self.size, max_samples)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return prepare_aligned_sample(self.split, index, self.standardizer)


class UnalignedMoseiDataset(Dataset):
    def __init__(
        self,
        split: Mapping[str, Any],
        standardizer: AVStandardizer,
        max_samples: int | None = None,
    ) -> None:
        self.split = split
        self.standardizer = standardizer
        self.size = len(split["audio"])
        if max_samples is not None:
            self.size = min(self.size, max_samples)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return prepare_unaligned_sample(self.split, index, self.standardizer)


def make_mosei_dataset(
    split: Mapping[str, Any],
    standardizer: AVStandardizer,
    max_samples: int | None = None,
    alignment: str = "auto",
) -> Dataset:
    detected = infer_alignment(split)
    selected = detected if alignment == "auto" else alignment
    if selected not in {"aligned", "unaligned"}:
        raise ValueError("alignment must be 'auto', 'aligned', or 'unaligned'")
    if selected != detected:
        raise ValueError(f"Requested {selected} data, but array shapes indicate {detected} data")
    dataset_type = AlignedMoseiDataset if selected == "aligned" else UnalignedMoseiDataset
    return dataset_type(split, standardizer, max_samples)
