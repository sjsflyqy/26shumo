from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BERT_PATH = str(
    Path(__file__).resolve().parents[2] / "pretrained/bert-base-uncased"
)


@dataclass
class M2Config:
    """Network configuration shared by training and inference."""

    audio_dim: int = 74
    vision_dim: int = 35
    hidden_dim: int = 128
    num_heads: int = 4
    modality_layers: int = 2
    text_layers: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.25
    num_classes: int = 3
    max_text_len: int = 50
    max_av_len: int = 500
    vocab_size: int = 30522
    type_vocab_size: int = 2
    text_backend: str = "hf"
    bert_model_name: str = DEFAULT_BERT_PATH
    freeze_bert: bool = True
    fusion_type: str = "cross_gated"
    fusion_levels: int = 2
    av_encoder_type: str = "transformer"
    use_latent_generator: bool = False
    generator_window: int = 7

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.text_backend not in {"lightweight", "hf"}:
            raise ValueError("text_backend must be 'lightweight' or 'hf'")
        if self.fusion_type not in {"cross_gated", "vertfound_multilevel"}:
            raise ValueError(
                "fusion_type must be 'cross_gated' or 'vertfound_multilevel'"
            )
        if self.av_encoder_type not in {"transformer", "mlp"}:
            raise ValueError("av_encoder_type must be 'transformer' or 'mlp'")
        if self.generator_window < 1 or self.generator_window % 2 != 1:
            raise ValueError("generator_window must be a positive odd integer")
        if self.fusion_levels < 1:
            raise ValueError("fusion_levels must be at least 1")
        if self.fusion_type == "vertfound_multilevel":
            if self.fusion_levels > self.modality_layers:
                raise ValueError("fusion_levels cannot exceed modality_layers")
            if self.text_backend == "lightweight" and self.fusion_levels > self.text_layers:
                raise ValueError("fusion_levels cannot exceed text_layers for lightweight text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "M2Config":
        return cls(**values)


@dataclass
class SpanMaskConfig:
    """Distribution used to generate local contiguous missing spans."""

    clean_probability: float = 0.25
    min_rate: float = 0.05
    max_rate: float = 0.60
    min_spans: int = 1
    max_spans: int = 3
    sync_probability: float = 0.50
    modality_combinations: tuple[str, ...] = (
        "T",
        "A",
        "V",
        "TA",
        "TV",
        "AV",
        "TAV",
    )
    locations: tuple[str, ...] = ("random", "begin", "middle", "end")

    def __post_init__(self) -> None:
        if not 0.0 <= self.clean_probability < 1.0:
            raise ValueError("clean_probability must be in [0, 1)")
        if not 0.0 < self.min_rate <= self.max_rate <= 1.0:
            raise ValueError("mask rates must satisfy 0 < min_rate <= max_rate <= 1")
        if not 1 <= self.min_spans <= self.max_spans:
            raise ValueError("invalid span count range")
        if not 0.0 <= self.sync_probability <= 1.0:
            raise ValueError("sync_probability must be in [0, 1]")
        valid_combinations = {"T", "A", "V", "TA", "TV", "AV", "TAV"}
        if not self.modality_combinations or any(
            value not in valid_combinations for value in self.modality_combinations
        ):
            raise ValueError("modality_combinations must contain valid nonempty T/A/V combinations")
        valid_locations = {"random", "begin", "middle", "end"}
        if not self.locations or any(value not in valid_locations for value in self.locations):
            raise ValueError("locations must contain valid nonempty locations")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["modality_combinations"] = list(self.modality_combinations)
        values["locations"] = list(self.locations)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SpanMaskConfig":
        values = dict(values)
        if "modality_combinations" in values:
            values["modality_combinations"] = tuple(values["modality_combinations"])
        if "locations" in values:
            values["locations"] = tuple(values["locations"])
        return cls(**values)
