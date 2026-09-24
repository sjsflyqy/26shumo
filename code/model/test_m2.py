from __future__ import annotations

import torch

from .config import M2Config, SpanMaskConfig
from .engine import train_one_epoch
from .losses import M2Objective
from .masking import corrupt_aligned_batch, teacher_emotion_importance
from .model import M2Model, create_ema_teacher, update_ema_teacher


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


def synthetic_unaligned_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    batch = synthetic_batch(batch_size=batch_size, length=10)
    batch.pop("reliability_mask")
    generator = torch.Generator().manual_seed(13)
    audio = torch.randn(batch_size, 18, 4, generator=generator)
    vision = torch.randn(batch_size, 14, 3, generator=generator)
    audio_structural = torch.arange(18)[None, :] < torch.tensor([18, 13, 9])[:, None]
    vision_structural = torch.arange(14)[None, :] < torch.tensor([14, 10, 7])[:, None]
    audio_reliability = audio_structural.clone()
    vision_reliability = vision_structural.clone()
    vision_reliability[1, 4:6] = False
    audio[~audio_reliability] = 0
    vision[~vision_reliability] = 0
    batch.update(
        {
            "audio": audio,
            "vision": vision,
            "audio_structural_mask": audio_structural,
            "vision_structural_mask": vision_structural,
            "audio_reliability_mask": audio_reliability,
            "vision_reliability_mask": vision_reliability,
        }
    )
    return batch


def main() -> None:
    config = M2Config(
        audio_dim=4,
        vision_dim=3,
        hidden_dim=16,
        num_heads=4,
        modality_layers=2,
        text_layers=2,
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

    unaligned_config = M2Config.from_dict(config.to_dict() | {"max_av_len": 20})
    unaligned = synthetic_unaligned_batch()
    unaligned_masked = corrupt_aligned_batch(
        unaligned,
        SpanMaskConfig(clean_probability=0.0),
        torch.Generator().manual_seed(17),
        force_rate=0.3,
        force_modalities="TAV",
        force_location="middle",
        force_spans=1,
    )
    assert unaligned_masked["text_corruption_mask"].any()
    assert unaligned_masked["audio_corruption_mask"].any()
    assert unaligned_masked["vision_corruption_mask"].any()
    unaligned_model = M2Model(unaligned_config)
    unaligned_output = unaligned_model(unaligned_masked)
    assert unaligned_output["class_logits"].shape == (3, 3)
    assert unaligned_output["audio_attention"].shape == (3, 10, 18)
    assert unaligned_output["vision_attention"].shape == (3, 10, 14)
    assert torch.isfinite(unaligned_output["class_logits"]).all()
    unaligned_multilevel_config = M2Config.from_dict(
        unaligned_config.to_dict()
        | {"fusion_type": "vertfound_multilevel", "fusion_levels": 2}
    )
    unaligned_multilevel_output = M2Model(unaligned_multilevel_config)(unaligned_masked)
    assert unaligned_multilevel_output["audio_attention"].shape == (3, 10, 18)
    assert unaligned_multilevel_output["vision_attention"].shape == (3, 10, 14)
    assert torch.isfinite(unaligned_multilevel_output["class_logits"]).all()
    mlp_config = M2Config.from_dict(
        unaligned_config.to_dict()
        | {
            "av_encoder_type": "mlp",
            "fusion_type": "vertfound_multilevel",
            "fusion_levels": 2,
        }
    )
    mlp_output = M2Model(mlp_config)(unaligned_masked)
    assert mlp_output["class_logits"].shape == (3, 3)
    assert mlp_output["audio_attention"].shape == (3, 10, 18)
    assert torch.isfinite(mlp_output["class_logits"]).all()

    multilevel_config = M2Config.from_dict(
        config.to_dict() | {"fusion_type": "vertfound_multilevel", "fusion_levels": 2}
    )
    multilevel_model = M2Model(multilevel_config)
    multilevel_clean = multilevel_model(clean)
    multilevel_masked = multilevel_model(masked)
    assert multilevel_clean["class_logits"].shape == (3, 3)
    assert multilevel_clean["fusion_level_weights"].shape == (3, 10, 2)
    assert torch.allclose(
        multilevel_clean["fusion_level_weights"].sum(dim=-1),
        torch.ones(3, 10),
    )
    assert torch.isfinite(multilevel_masked["class_logits"]).all()
    multilevel_losses = M2Objective()(
        multilevel_clean,
        multilevel_masked,
        clean["class_label"],
        clean["regression_label"],
    )
    multilevel_losses["loss"].backward()
    assert torch.isfinite(multilevel_losses["loss"])

    teacher = create_ema_teacher(multilevel_model)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    with torch.no_grad():
        teacher_clean = teacher(clean)
    distilled_losses = M2Objective()(
        multilevel_clean,
        multilevel_masked,
        clean["class_label"],
        clean["regression_label"],
        teacher_output=teacher_clean,
    )
    assert {
        "distill_logits", "distill_regression", "distill_feature"
    }.issubset(distilled_losses)
    assert torch.isfinite(distilled_losses["loss"])
    teacher_before = teacher.classifier.weight.detach().clone()
    with torch.no_grad():
        multilevel_model.classifier.weight.add_(1.0)
    update_ema_teacher(multilevel_model, teacher, decay=0.5)
    assert torch.allclose(teacher.classifier.weight, teacher_before + 0.5)

    train_student = M2Model(config)
    train_teacher = create_ema_teacher(train_student)
    optimizer = torch.optim.AdamW(train_student.parameters(), lr=1e-3)
    epoch_losses = train_one_epoch(
        train_student,
        [clean],
        optimizer,
        M2Objective(),
        SpanMaskConfig(clean_probability=0.0),
        torch.device("cpu"),
        torch.Generator().manual_seed(19),
        teacher=train_teacher,
    )
    assert "distill_logits" in epoch_losses
    assert all(torch.isfinite(torch.tensor(value)) for value in epoch_losses.values())

    # An EMA teacher proposes training-only emotion-aware masks. Padding and
    # already missing positions never receive saliency or reconstruction targets.
    generator_config = M2Config.from_dict(
        config.to_dict() | {"use_latent_generator": True, "generator_window": 3}
    )
    generator_model = M2Model(generator_config)
    generator_teacher = create_ema_teacher(generator_model)
    with torch.enable_grad():
        guided_teacher_output = generator_teacher(clean, capture_saliency=True)
        saliency = teacher_emotion_importance(guided_teacher_output, clean)
    assert all(saliency[symbol].shape == (3, 10) for symbol in "TAV")
    assert torch.all(saliency["T"][:, 8:] == 0)
    assert torch.all(saliency["A"][:, 0] == 0)
    peaked = {symbol: torch.zeros_like(saliency[symbol]) for symbol in "TAV"}
    peaked["T"][:, 3] = 1.0
    peaked_mask = corrupt_aligned_batch(
        clean,
        SpanMaskConfig(clean_probability=0.0),
        torch.Generator().manual_seed(23),
        force_rate=1 / 6,
        force_modalities="T",
        force_location="random",
        force_spans=1,
        importance=peaked,
        emotion_probability=1.0,
        emotion_temperature=0.01,
    )
    assert torch.all(peaked_mask["corruption_mask"][:, 3, 0])
    assert torch.all(peaked_mask["corruption_mask"][..., 0].sum(dim=1) == 1)
    guided_masked = corrupt_aligned_batch(
        clean,
        SpanMaskConfig(clean_probability=0.0),
        torch.Generator().manual_seed(29),
        force_rate=0.3,
        force_modalities="TAV",
        force_location="random",
        importance=saliency,
        emotion_probability=1.0,
    )
    generated = generator_model(guided_masked)
    assert torch.isfinite(generated["class_logits"]).all()
    for index in range(3):
        observed = clean["reliability_mask"][..., index]
        missing = guided_masked["corruption_mask"][..., index] & observed
        assert torch.all(generated["latent_confidence"][index][missing] > 0)
        assert torch.equal(
            generated["latent_reconstruction"][index][observed & ~guided_masked["corruption_mask"][..., index]],
            generated["latent_targets"][index][observed & ~guided_masked["corruption_mask"][..., index]],
        )
    generator_objective = M2Objective(
        distill_enabled=False, reconstruction_weight=0.1, confidence_weight=0.02
    )
    generator_losses = generator_objective(
        generator_model(clean),
        generated,
        clean["class_label"],
        clean["regression_label"],
        teacher_output=guided_teacher_output,
        clean_batch=clean,
        masked_batch=guided_masked,
    )
    assert float(generator_losses["reconstructed_positions"]) > 0
    assert torch.isfinite(generator_losses["loss"])
    generator_losses["loss"].backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in generator_model.latent_generator.parameters()
    )

    unaligned_generator_config = M2Config.from_dict(
        unaligned_multilevel_config.to_dict()
        | {"use_latent_generator": True, "generator_window": 3}
    )
    unaligned_generator = M2Model(unaligned_generator_config)
    unaligned_teacher = create_ema_teacher(unaligned_generator)
    with torch.enable_grad():
        unaligned_teacher_output = unaligned_teacher(unaligned, capture_saliency=True)
        unaligned_importance = teacher_emotion_importance(unaligned_teacher_output, unaligned)
    unaligned_guided = corrupt_aligned_batch(
        unaligned,
        SpanMaskConfig(clean_probability=0.0),
        torch.Generator().manual_seed(31),
        force_rate=0.3,
        force_modalities="TAV",
        force_location="random",
        importance=unaligned_importance,
        emotion_probability=1.0,
    )
    unaligned_generated = unaligned_generator(unaligned_guided)
    assert unaligned_generated["audio_attention"].shape == (3, 10, 18)
    assert torch.isfinite(unaligned_generated["class_logits"]).all()
    unaligned_generator_losses = generator_objective(
        unaligned_generator(unaligned),
        unaligned_generated,
        unaligned["class_label"],
        unaligned["regression_label"],
        teacher_output=unaligned_teacher_output,
        clean_batch=unaligned,
        masked_batch=unaligned_guided,
    )
    assert float(unaligned_generator_losses["reconstructed_positions"]) > 0
    assert torch.isfinite(unaligned_generator_losses["loss"])

    one_epoch_generator = M2Model(generator_config)
    one_epoch_teacher = create_ema_teacher(one_epoch_generator)
    one_epoch_losses = train_one_epoch(
        one_epoch_generator,
        [clean],
        torch.optim.AdamW(one_epoch_generator.parameters(), lr=1e-3),
        generator_objective,
        SpanMaskConfig(clean_probability=0.0),
        torch.device("cpu"),
        torch.Generator().manual_seed(37),
        teacher=one_epoch_teacher,
        emotion_mask_probability=1.0,
    )
    assert "reconstruction" in one_epoch_losses
    assert all(torch.isfinite(torch.tensor(value)) for value in one_epoch_losses.values())
    emotion_only_student = M2Model(config)
    emotion_only_losses = train_one_epoch(
        emotion_only_student,
        [clean],
        torch.optim.AdamW(emotion_only_student.parameters(), lr=1e-3),
        M2Objective(distill_enabled=False),
        SpanMaskConfig(clean_probability=0.0),
        torch.device("cpu"),
        torch.Generator().manual_seed(41),
        teacher=create_ema_teacher(emotion_only_student),
        emotion_mask_probability=1.0,
    )
    assert "reconstruction" not in emotion_only_losses
    assert all(torch.isfinite(torch.tensor(value)) for value in emotion_only_losses.values())
    print("M2 synthetic smoke test passed")


if __name__ == "__main__":
    main()
