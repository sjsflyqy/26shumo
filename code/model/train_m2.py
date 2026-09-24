from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_BERT_PATH, M2Config, SpanMaskConfig
from .data import AVStandardizer, infer_alignment, load_pickle, make_mosei_dataset
from .engine import evaluate_model, inverse_frequency_class_weights, train_one_epoch
from .losses import M2Objective
from .metrics import composite_score
from .model import M2Model, checkpoint_state, create_ema_teacher, load_checkpoint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M2 on aligned or unaligned MOSEI features")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/attachment2/aligned_50.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs/m2_aligned",
    )
    parser.add_argument(
        "--alignment",
        choices=("auto", "aligned", "unaligned"),
        default="auto",
        help="Data layout; auto infers it from the T/A/V sequence lengths",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument(
        "--av-encoder",
        choices=("transformer", "mlp"),
        default="transformer",
        help="Temporal Transformer or lightweight point-wise residual MLP for audio/vision",
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--text-backend", choices=("lightweight", "hf"), default="hf")
    parser.add_argument("--bert-model-name", default=DEFAULT_BERT_PATH)
    parser.add_argument("--finetune-bert", action="store_true")
    parser.add_argument(
        "--fusion-type",
        choices=("cross_gated", "vertfound_multilevel"),
        default="cross_gated",
        help="Baseline single-level fusion or VertFound-style multi-level bidirectional fusion",
    )
    parser.add_argument("--fusion-levels", type=int, default=2)
    parser.add_argument("--clean-probability", type=float, default=0.25)
    parser.add_argument("--min-mask-rate", type=float, default=0.05)
    parser.add_argument("--max-mask-rate", type=float, default=0.60)
    parser.add_argument("--valid-mask-rate", type=float, default=0.30)
    parser.add_argument(
        "--emotion-mask-probability", type=float, default=0.0,
        help="Maximum fraction of training samples using EMA-teacher emotion saliency",
    )
    parser.add_argument("--emotion-mask-warmup-epochs", type=int, default=3)
    parser.add_argument("--emotion-mask-ramp-epochs", type=int, default=5)
    parser.add_argument("--emotion-mask-temperature", type=float, default=0.2)
    parser.add_argument("--emotion-intensity-weight", type=float, default=0.25)
    parser.add_argument("--latent-generator", action="store_true")
    parser.add_argument("--generator-window", type=int, default=7)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--confidence-weight", type=float, default=0.02)
    parser.add_argument("--consistency-weight", type=float, default=0.10)
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Enable clean-view EMA teacher to masked-view student distillation",
    )
    parser.add_argument("--teacher-ema-decay", type=float, default=0.996)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-logit-weight", type=float, default=0.5)
    parser.add_argument("--distill-regression-weight", type=float, default=0.25)
    parser.add_argument("--distill-feature-weight", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.teacher_ema_decay < 1.0:
        raise ValueError("--teacher-ema-decay must satisfy 0 <= decay < 1")
    if min(
        args.distill_logit_weight,
        args.distill_regression_weight,
        args.distill_feature_weight,
    ) < 0:
        raise ValueError("distillation loss weights must be non-negative")
    if not 0.0 <= args.emotion_mask_probability <= 1.0:
        raise ValueError("--emotion-mask-probability must be in [0, 1]")
    if args.emotion_mask_warmup_epochs < 0 or args.emotion_mask_ramp_epochs < 1:
        raise ValueError("emotion mask warmup must be non-negative and ramp must be positive")
    if args.emotion_mask_temperature <= 0 or args.emotion_intensity_weight < 0:
        raise ValueError("emotion mask temperature must be positive and intensity weight non-negative")
    if args.reconstruction_weight < 0 or args.confidence_weight < 0:
        raise ValueError("generator loss weights must be non-negative")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_pickle(args.data)
    required = {"train", "valid", "test"}
    if not required.issubset(data):
        raise KeyError(f"Expected splits {sorted(required)}, found {sorted(data)}")
    standardizer = AVStandardizer.fit(data["train"])
    alignment = infer_alignment(data["train"]) if args.alignment == "auto" else args.alignment
    datasets = {
        "train": make_mosei_dataset(
            data["train"], standardizer, args.max_train_samples, alignment
        ),
        "valid": make_mosei_dataset(
            data["valid"], standardizer, args.max_valid_samples, alignment
        ),
        "test": make_mosei_dataset(
            data["test"], standardizer, args.max_test_samples, alignment
        ),
    }
    train_generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            generator=train_generator,
            pin_memory=args.device.startswith("cuda"),
        ),
        "valid": DataLoader(
            datasets["valid"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ),
    }
    model_config = M2Config(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        modality_layers=args.encoder_layers,
        text_layers=args.encoder_layers,
        feedforward_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        text_backend=args.text_backend,
        bert_model_name=args.bert_model_name,
        freeze_bert=not args.finetune_bert,
        fusion_type=args.fusion_type,
        fusion_levels=args.fusion_levels,
        av_encoder_type=args.av_encoder,
        use_latent_generator=args.latent_generator,
        generator_window=args.generator_window,
        max_text_len=int(np.asarray(data["train"]["text_bert"]).shape[-1]),
        max_av_len=max(
            int(np.asarray(data["train"]["audio"]).shape[1]),
            int(np.asarray(data["train"]["vision"]).shape[1]),
        ),
    )
    mask_config = SpanMaskConfig(
        clean_probability=args.clean_probability,
        min_rate=args.min_mask_rate,
        max_rate=args.max_mask_rate,
    )
    device = torch.device(args.device)
    model = M2Model(model_config)
    teacher = create_ema_teacher(model) if (
        args.distill or args.emotion_mask_probability > 0 or args.latent_generator
    ) else None
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device)
    labels = np.asarray(data["train"]["classification_labels"])[: len(datasets["train"])]
    class_weights = inverse_frequency_class_weights(labels).to(device)
    objective = M2Objective(
        class_weights=class_weights,
        consistency_weight=args.consistency_weight,
        distill_temperature=args.distill_temperature,
        distill_logit_weight=args.distill_logit_weight,
        distill_regression_weight=args.distill_regression_weight,
        distill_feature_weight=args.distill_feature_weight,
        distill_enabled=args.distill,
        reconstruction_weight=args.reconstruction_weight if args.latent_generator else 0.0,
        confidence_weight=args.confidence_weight if args.latent_generator else 0.0,
    ).to(device)
    bert_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_encoder.backbone."):
            bert_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    parameter_groups = [{"params": other_parameters, "lr": args.learning_rate}]
    if bert_parameters:
        parameter_groups.append({"params": bert_parameters, "lr": args.bert_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mask_generator = torch.Generator().manual_seed(args.seed + 17)
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_m2.pt"

    for epoch in range(1, args.epochs + 1):
        emotion_probability = args.emotion_mask_probability * min(
            1.0,
            max(0.0, (epoch - args.emotion_mask_warmup_epochs) / args.emotion_mask_ramp_epochs),
        )
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            objective,
            mask_config,
            device,
            mask_generator,
            teacher=teacher,
            teacher_ema_decay=args.teacher_ema_decay,
            emotion_mask_probability=emotion_probability,
            emotion_mask_temperature=args.emotion_mask_temperature,
            emotion_intensity_weight=args.emotion_intensity_weight,
        )
        scheduler.step()
        valid_clean = evaluate_model(model, loaders["valid"], device)
        valid_masked = evaluate_model(
            model,
            loaders["valid"],
            device,
            mask_config=mask_config,
            seed=args.seed + 1000,
            mask_rate=args.valid_mask_rate,
        )
        score = 0.5 * (composite_score(valid_clean) + composite_score(valid_masked))
        record: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_losses,
            "emotion_mask_probability": emotion_probability,
            "valid_clean": valid_clean,
            "valid_masked": valid_masked,
            "selection_score": score,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            saved_state, omitted_backbone = checkpoint_state(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": saved_state,
                    "omitted_frozen_text_backbone": omitted_backbone,
                    "model_config": model_config.to_dict(),
                    "mask_config": mask_config.to_dict(),
                    "standardizer": standardizer.state_dict(),
                    "alignment": alignment,
                    "valid_clean": valid_clean,
                    "valid_masked": valid_masked,
                    "selection_score": score,
                    "distillation": {
                        "enabled": args.distill,
                        "teacher_ema_decay": args.teacher_ema_decay,
                        "temperature": args.distill_temperature,
                        "logit_weight": args.distill_logit_weight,
                        "regression_weight": args.distill_regression_weight,
                        "feature_weight": args.distill_feature_weight,
                    },
                    "emotion_masking": {
                        "maximum_probability": args.emotion_mask_probability,
                        "warmup_epochs": args.emotion_mask_warmup_epochs,
                        "ramp_epochs": args.emotion_mask_ramp_epochs,
                        "temperature": args.emotion_mask_temperature,
                        "intensity_weight": args.emotion_intensity_weight,
                    },
                    "latent_generation": {
                        "enabled": args.latent_generator,
                        "window": args.generator_window,
                        "reconstruction_weight": args.reconstruction_weight,
                        "confidence_weight": args.confidence_weight,
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_checkpoint_state(model, checkpoint)
    test_clean = evaluate_model(model, loaders["test"], device)
    test_masked = evaluate_model(
        model,
        loaders["test"],
        device,
        mask_config=mask_config,
        seed=args.seed + 2000,
        mask_rate=args.valid_mask_rate,
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "valid_clean": checkpoint["valid_clean"],
        "valid_masked": checkpoint["valid_masked"],
        "test_clean": test_clean,
        "test_masked": test_masked,
        "arguments": vars(args) | {"data": str(args.data), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
