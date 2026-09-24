from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .config import M2Config
from .data import (
    AVStandardizer,
    infer_alignment,
    load_pickle,
    prepare_aligned_sample,
    prepare_unaligned_sample,
)
from .engine import move_to_device
from .model import M2Model, load_checkpoint_state


ANNOTATIONS = {0: "Negative", 1: "Neutral", 2: "Positive"}


@lru_cache(maxsize=2)
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to tokenize unaligned attachment 3") from exc
    return AutoTokenizer.from_pretrained(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M2 on aligned or unaligned attachment 3")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--alignment", choices=("auto", "aligned", "unaligned"), default="auto"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _batch_sample(sample: dict[str, object]) -> dict[str, object]:
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }


def _ensure_text_bert(
    split: dict[str, object], model_config: M2Config
) -> dict[str, object]:
    if "text_bert" in split:
        return split
    if "raw_text" not in split:
        raise KeyError("Attachment-3 sample contains neither 'text_bert' nor 'raw_text'")
    tokenizer = _load_tokenizer(model_config.bert_model_name)
    encoded = tokenizer(
        np.asarray(split["raw_text"]).reshape(-1).astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=model_config.max_text_len,
        return_tensors="np",
    )
    token_types = encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"]))
    prepared = dict(split)
    prepared["text_bert"] = np.stack(
        (encoded["input_ids"], encoded["attention_mask"], token_types), axis=1
    )
    return prepared


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = M2Config.from_dict(checkpoint["model_config"])
    model = M2Model(model_config).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    checkpoint_alignment = checkpoint.get("alignment", "aligned")
    alignment = checkpoint_alignment if args.alignment == "auto" else args.alignment
    if alignment != checkpoint_alignment:
        raise ValueError(
            f"Checkpoint was trained on {checkpoint_alignment} data, but --alignment={alignment}"
        )
    input_dir = args.input_dir
    if input_dir is None:
        version = "对齐版本" if alignment == "aligned" else "未对齐版本"
        input_dir = Path(__file__).resolve().parents[2] / "data/attachment3" / version
    output_path = args.output or args.checkpoint.parent / "attachment3_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for path in sorted(input_dir.glob("*.pkl")):
            obj = load_pickle(path)
            if "test" not in obj:
                raise KeyError(f"{path} does not contain top-level key 'test'")
            split = _ensure_text_bert(obj["test"], model_config)
            detected = infer_alignment(split)
            if detected != alignment:
                raise ValueError(f"{path} is {detected}, but the checkpoint is {alignment}")
            sample_id = path.stem.split("_")[-1]
            prepare = prepare_aligned_sample if alignment == "aligned" else prepare_unaligned_sample
            sample = prepare(split, 0, standardizer, fallback_id=sample_id)
            batch = move_to_device(_batch_sample(sample), device)
            prediction = model(batch)
            probabilities = torch.softmax(prediction["class_logits"], dim=-1)[0].cpu()
            predicted_class = int(probabilities.argmax())
            content = batch["content_mask"][0].bool()
            text_denominator = max(1, int(content.sum()))
            text_ratio = 1.0 - float(
                (batch["text_attention_mask"][0].bool() & content).sum()
            ) / text_denominator
            if alignment == "unaligned":
                audio_denominator = batch["audio_structural_mask"][0].sum().clamp_min(1)
                vision_denominator = batch["vision_structural_mask"][0].sum().clamp_min(1)
                audio_ratio = 1.0 - batch["audio_reliability_mask"][0].sum().float() / audio_denominator
                vision_ratio = 1.0 - batch["vision_reliability_mask"][0].sum().float() / vision_denominator
            else:
                reliability = batch["reliability_mask"][0].bool()
                audio_ratio = 1.0 - (reliability[:, 1] & content).sum().float() / text_denominator
                vision_ratio = 1.0 - (reliability[:, 2] & content).sum().float() / text_denominator
            rows.append(
                {
                    "sample_id": sample_id,
                    "predicted_class": predicted_class,
                    "predicted_annotation": ANNOTATIONS[predicted_class],
                    "predicted_intensity": float(prediction["intensity"][0].cpu()),
                    "prob_negative": float(probabilities[0]),
                    "prob_neutral": float(probabilities[1]),
                    "prob_positive": float(probabilities[2]),
                    "missing_text_ratio": text_ratio,
                    "missing_audio_ratio": float(audio_ratio),
                    "missing_vision_ratio": float(vision_ratio),
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
