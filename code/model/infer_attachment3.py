from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .config import M2Config
from .data import AVStandardizer, load_pickle, prepare_aligned_sample
from .engine import move_to_device
from .model import M2Model, load_checkpoint_state


ANNOTATIONS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2 on aligned attachment 3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/attachment3/对齐版本",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _batch_sample(sample: dict[str, object]) -> dict[str, object]:
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = M2Model(M2Config.from_dict(checkpoint["model_config"])).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    output_path = args.output or args.checkpoint.parent / "attachment3_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for path in sorted(args.input_dir.glob("*.pkl")):
            obj = load_pickle(path)
            if "test" not in obj:
                raise KeyError(f"{path} does not contain top-level key 'test'")
            split = obj["test"]
            sample_id = path.stem.split("_")[-1]
            sample = prepare_aligned_sample(split, 0, standardizer, fallback_id=sample_id)
            batch = move_to_device(_batch_sample(sample), device)
            prediction = model(batch)
            probabilities = torch.softmax(prediction["class_logits"], dim=-1)[0].cpu()
            predicted_class = int(probabilities.argmax())
            content = batch["content_mask"][0].bool()
            denominator = max(1, int(content.sum()))
            reliability = batch["reliability_mask"][0].bool()
            ratios = 1.0 - (reliability & content[:, None]).sum(dim=0).float() / denominator
            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_class": predicted_class,
                    "predicted_annotation": ANNOTATIONS[predicted_class],
                    "predicted_intensity": float(prediction["intensity"][0].cpu()),
                    "prob_negative": float(probabilities[0]),
                    "prob_neutral": float(probabilities[1]),
                    "prob_positive": float(probabilities[2]),
                    "missing_text_ratio": float(ratios[0].cpu()),
                    "missing_audio_ratio": float(ratios[1].cpu()),
                    "missing_vision_ratio": float(ratios[2].cpu()),
                }
            )
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
