from __future__ import annotations

import torch

from .config import M2Config, SpanMaskConfig
from .losses import M2Objective
from .masking import corrupt_aligned_batch
from .model import M2Model


def synthetic_batch(batch_size: int = 3, length: int = 10) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    structural = torch.zeros(batch_size, length, dtype=torch.bool)
    structural[:, :8] = True
    content = structural.clone()
    content[:, 0] = False
    content[:, 7] = False
    reliability = structural[..., None].expand(-1, -1, 3).clone()
    reliability[:, 0, 1:] = False
    reliability[:, 7, 1:] = False
    audio = torch.randn(batch_size, length, 4, generator=generator)
    vision = torch.randn(batch_size, length, 3, generator=generator)
    audio[~reliability[..., 1]] = 0
    vision[~reliability[..., 2]] = 0
    return {
        "input_ids": torch.randint(1, 99, (batch_size, length), generator=generator),
        "text_attention_mask": structural.clone(),
        "token_type_ids": torch.zeros(batch_size, length, dtype=torch.long),
        "text_structural_mask": structural,
        "content_mask": content,
        "audio": audio,
        "vision": vision,
        "reliability_mask": reliability,
        "class_label": torch.tensor([0, 1, 2]),
        "regression_label": torch.tensor([-1.0, 0.0, 1.0]),
    }


def main() -> None:
    config = M2Config(
        audio_dim=4,
        vision_dim=3,
        hidden_dim=16,
        num_heads=4,
        modality_layers=1,
        text_layers=1,
        feedforward_dim=32,
        dropout=0.0,
        max_text_len=10,
        max_av_len=10,
        vocab_size=100,
        text_backend="lightweight",
    )
    clean = synthetic_batch()
    masked = corrupt_aligned_batch(
        clean,
        SpanMaskConfig(
            clean_probability=0.0,
            min_rate=0.3,
            max_rate=0.3,
            min_spans=1,
            max_spans=1,
            sync_probability=1.0,
        ),
        torch.Generator().manual_seed(11),
        force_rate=0.3,
        force_modalities="TAV",
        force_location="middle",
    )
    assert masked["corruption_mask"].any()
    assert torch.all(masked["audio"][masked["corruption_mask"][..., 1]] == 0)
    assert torch.all(masked["vision"][masked["corruption_mask"][..., 2]] == 0)

    model = M2Model(config)
    clean_output = model(clean)
    masked_output = model(masked)
    assert clean_output["class_logits"].shape == (3, 3)
    assert clean_output["intensity"].shape == (3,)
    assert clean_output["modality_weights"].shape == (3, 10, 3)
    assert torch.isfinite(masked_output["class_logits"]).all()
    assert torch.allclose(
        masked_output["modality_weights"][~masked["reliability_mask"]],
        torch.zeros_like(masked_output["modality_weights"][~masked["reliability_mask"]]),
    )
    losses = M2Objective()(clean_output, masked_output, clean["class_label"], clean["regression_label"])
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    print("M2 synthetic smoke test passed")


if __name__ == "__main__":
    main()
