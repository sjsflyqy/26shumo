from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import M2Config, SpanMaskConfig
from .data import AVStandardizer, load_pickle, make_mosei_dataset
from .engine import move_to_device
from .masking import corrupt_aligned_batch
from .model import M2Model, load_checkpoint_state


def _spans(mask: torch.Tensor) -> str:
    positions = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not positions:
        return ""
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append(f"{start}:{previous + 1}")
            start = position
        previous = position
    runs.append(f"{start}:{previous + 1}")
    return ";".join(runs)


def _masks(batch: dict[str, object], masked: dict[str, object], index: int) -> dict[str, object]:
    unaligned = "audio_reliability_mask" in batch
    row: dict[str, object] = {}
    for symbol, name, position in (("T", "text", 0), ("A", "audio", 1), ("V", "vision", 2)):
        if unaligned:
            valid_key = "content_mask" if symbol == "T" else f"{name}_structural_mask"
            observed_key = "text_attention_mask" if symbol == "T" else f"{name}_reliability_mask"
            corruption_key = f"{name}_corruption_mask"
            valid = batch[valid_key][index].bool()
            observed = batch[observed_key][index].bool()
            corrupted = masked[corruption_key][index].bool() if corruption_key in masked else torch.zeros_like(valid)
        else:
            valid = batch["content_mask"][index].bool()
            observed = batch["reliability_mask"][index, :, position].bool()
            corrupted = (
                masked["corruption_mask"][index, :, position].bool()
                if "corruption_mask" in masked else torch.zeros_like(valid)
            )
        selected = corrupted & observed & valid
        count = max(1, int(valid.sum()))
        row[f"{name}_valid_length"] = int(valid.sum())
        row[f"{name}_natural_missing_rate"] = 1.0 - float((observed & valid).sum()) / count
        row[f"{name}_artificial_mask_count"] = int(selected.sum())
        row[f"{name}_artificial_mask_rate"] = float(selected.sum()) / count
        row[f"{name}_artificial_mask_spans"] = _spans(selected)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labeled M2 predictions and mask positions")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--alignment", choices=("auto", "aligned", "unaligned"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--modalities", choices=("T", "A", "V", "TA", "TV", "AV", "TAV"), default="TAV")
    parser.add_argument("--location", choices=("random", "begin", "middle", "end"), default="random")
    parser.add_argument("--spans", type=int, default=1)
    parser.add_argument("--sync-mode", choices=("mixed", "sync", "async"), default="mixed")
    args = parser.parse_args()
    if args.rate is not None and not 0 < args.rate <= 1:
        raise ValueError("--rate must be in (0, 1]")
    if args.spans < 1:
        raise ValueError("--spans must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = M2Model(M2Config.from_dict(checkpoint["model_config"])).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    alignment = checkpoint.get("alignment", "aligned")
    if args.alignment != "auto" and args.alignment != alignment:
        raise ValueError(f"Checkpoint uses {alignment}, not {args.alignment}")
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    split = load_pickle(args.data)[args.split]
    loader = DataLoader(
        make_mosei_dataset(split, standardizer, args.max_samples, alignment),
        batch_size=args.batch_size, shuffle=False,
    )
    mask_config = SpanMaskConfig.from_dict(checkpoint["mask_config"])
    generator = torch.Generator().manual_seed(args.seed)
    force_sync = {"sync": True, "async": False}.get(args.sync_mode)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for clean_cpu in loader:
            masked_cpu = (
                corrupt_aligned_batch(
                    clean_cpu, mask_config, generator,
                    force_rate=args.rate,
                    force_modalities=args.modalities,
                    force_location=args.location,
                    force_spans=args.spans,
                    force_sync=force_sync,
                ) if args.rate is not None else clean_cpu
            )
            output = model(move_to_device(masked_cpu, device))
            probabilities = torch.softmax(output["class_logits"], dim=-1).cpu()
            intensity = output["intensity"].cpu()
            for index, sample_id in enumerate(clean_cpu["sample_id"]):
                true_class = int(clean_cpu["class_label"][index])
                predicted_class = int(probabilities[index].argmax())
                true_intensity = float(clean_cpu["regression_label"][index])
                predicted_intensity = float(intensity[index])
                row: dict[str, object] = {
                    "sample_id": sample_id,
                    "split": args.split,
                    "alignment": alignment,
                    "requested_rate": args.rate if args.rate is not None else 0.0,
                    "modalities": args.modalities if args.rate is not None else "none",
                    "location": args.location if args.rate is not None else "none",
                    "span_count": args.spans if args.rate is not None else 0,
                    "sync_mode": args.sync_mode if args.rate is not None else "none",
                    "seed": args.seed,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "class_correct": int(true_class == predicted_class),
                    "prob_negative": float(probabilities[index, 0]),
                    "prob_neutral": float(probabilities[index, 1]),
                    "prob_positive": float(probabilities[index, 2]),
                    "true_intensity": true_intensity,
                    "predicted_intensity": predicted_intensity,
                    "absolute_error": abs(predicted_intensity - true_intensity),
                }
                row.update(_masks(clean_cpu, masked_cpu, index))
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
