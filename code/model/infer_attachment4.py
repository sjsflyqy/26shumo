from __future__ import annotations

import argparse
import csv
import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import M2Config
from .data import AVStandardizer, prepare_aligned_sample, prepare_unaligned_sample
from .engine import move_to_device
from .explain import (
    NAMES,
    candidate_windows,
    internal_importance,
    modality_counterfactuals,
    score_occlusion_windows,
    select_non_overlapping,
    validity_mask,
)
from .model import M2Model, load_checkpoint_state
from .plot_explanation_card import (
    create_explanation_card,
    extract_keyframe,
    video_metadata,
)


ANNOTATIONS = {0: "Negative", 1: "Neutral", 2: "Positive"}
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/attachment4/附件4-可解释专项视频样本与特征文件"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict and explain every attachment-4 sample")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--alignment", choices=("auto", "aligned", "unaligned"), default="auto"
    )
    parser.add_argument("--window-rate", type=float, default=0.15)
    parser.add_argument("--stride-rate", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--occlusion-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-keyframes", action="store_true")
    parser.add_argument("--no-cards", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a dictionary")
    return value


def _alignment_of(sample: dict[str, Any]) -> str:
    text_length = int(np.asarray(sample["text_bert"]).shape[-1])
    return (
        "aligned"
        if np.asarray(sample["audio"]).shape[-2]
        == np.asarray(sample["vision"]).shape[-2]
        == text_length
        else "unaligned"
    )


def _as_split(sample: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    sequence_keys = {"audio", "vision", "text", "text_bert"}
    scalar_keys = {"raw_text", "id", "audio_lengths", "vision_lengths"}
    for key, value in sample.items():
        array = np.asarray(value)
        if key in sequence_keys:
            output[key] = array[None, ...]
        elif key in scalar_keys:
            output[key] = array.reshape(1)
        else:
            output[key] = value
    return output


def _batch_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _decode_text(model_name: str, input_ids: torch.Tensor, start: int, end: int) -> str:
    ids = input_ids[0, start:end].detach().cpu().tolist()
    tokenizer = _tokenizer(model_name)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _extent(batch: dict[str, Any], symbol: str) -> int:
    positions = torch.nonzero(validity_mask(batch, symbol)[0], as_tuple=False).flatten()
    return int(positions[-1]) + 1 if positions.numel() else 1


def _time_bounds(start: int, end: int, extent: int, duration: float) -> tuple[float, float]:
    return duration * start / max(1, extent), duration * end / max(1, extent)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.window_rate <= 1.0:
        raise ValueError("--window-rate must be in (0, 1]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_alignment = checkpoint.get("alignment")
    alignment = args.alignment
    if alignment == "auto" and checkpoint_alignment in {"aligned", "unaligned"}:
        alignment = checkpoint_alignment
    if alignment == "auto" and args.input_dir is None:
        alignment = "aligned"
    input_dir = args.input_dir
    if input_dir is None:
        version = "对齐版本" if alignment == "aligned" else "未对齐版本"
        input_dir = DEFAULT_ROOT / version
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No PKL files found in {input_dir}")
    first_alignment = _alignment_of(_load_pickle(files[0]))
    if alignment == "auto":
        alignment = first_alignment
    if alignment != first_alignment:
        raise ValueError(f"Input is {first_alignment}, but --alignment={alignment}")
    if checkpoint_alignment in {"aligned", "unaligned"} and checkpoint_alignment != alignment:
        raise ValueError(
            f"Checkpoint is {checkpoint_alignment}, but attachment-4 input is {alignment}"
        )

    device = torch.device(args.device)
    model_config = M2Config.from_dict(checkpoint["model_config"])
    model = M2Model(model_config).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    prepare = prepare_aligned_sample if alignment == "aligned" else prepare_unaligned_sample
    output_dir = args.output_dir or args.checkpoint.parent / f"attachment4_{alignment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.video_dir or input_dir / "videos"
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for file_index, path in enumerate(files):
        if args.max_samples is not None and file_index >= args.max_samples:
            break
        raw_sample = _load_pickle(path)
        sample_id = str(np.asarray(raw_sample.get("id", path.stem)).reshape(-1)[0])
        sample = prepare(_as_split(raw_sample), 0, standardizer, fallback_id=sample_id)
        batch = move_to_device(_batch_sample(sample), device)
        with torch.no_grad():
            base_output = model(batch)
        probabilities_tensor = torch.softmax(base_output["class_logits"], dim=-1)[0]
        probabilities = probabilities_tensor.detach().cpu().tolist()
        predicted_class = int(probabilities_tensor.argmax())
        curves = internal_importance(base_output, batch)
        candidates = candidate_windows(batch, curves, args.window_rate, args.stride_rate)
        scored = score_occlusion_windows(
            model, batch, base_output, candidates, batch_size=args.occlusion_batch_size
        )
        deltas, contributions = modality_counterfactuals(model, batch, base_output)
        main_modality = max(contributions, key=contributions.get)
        per_modality = {
            name: select_non_overlapping(
                [item for item in scored if item.modality == name],
                args.top_k,
                score="occlusion_score",
            )
            for name in NAMES.values()
        }

        video_path = video_dir / f"{sample_id}.mp4"
        duration, _, _ = video_metadata(video_path)
        best_text = per_modality["text"][0] if per_modality["text"] else None
        best_audio = per_modality["audio"][0] if per_modality["audio"] else None
        best_vision = per_modality["vision"][0] if per_modality["vision"] else None
        audio_times = (
            _time_bounds(best_audio.start, best_audio.end, _extent(batch, "A"), duration)
            if best_audio
            else (0.0, 0.0)
        )
        vision_times = (
            _time_bounds(best_vision.start, best_vision.end, _extent(batch, "V"), duration)
            if best_vision
            else (0.0, 0.0)
        )
        text_evidence = (
            _decode_text(model_config.bert_model_name, batch["input_ids"], best_text.start, best_text.end)
            if best_text
            else ""
        )
        keyframe_path = None
        if best_vision and not args.no_keyframes and video_path.exists():
            keyframe_path = extract_keyframe(
                video_path,
                0.5 * (vision_times[0] + vision_times[1]),
                output_dir / "keyframes" / f"{sample_id}.jpg",
            )

        evidence_summary = []
        for modality, windows in per_modality.items():
            for rank, item in enumerate(windows, start=1):
                evidence_text = ""
                start_time = end_time = 0.0
                if modality == "text":
                    evidence_text = _decode_text(
                        model_config.bert_model_name,
                        batch["input_ids"],
                        item.start,
                        item.end,
                    )
                else:
                    symbol = "A" if modality == "audio" else "V"
                    start_time, end_time = _time_bounds(
                        item.start, item.end, _extent(batch, symbol), duration
                    )
                detail_rows.append(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "evidence_rank": rank,
                        "start_index": item.start,
                        "end_index": item.end,
                        "start_time": start_time,
                        "end_time": end_time,
                        "attention_score": item.internal_score,
                        "occlusion_score": item.occlusion_score,
                        "probability_drop": item.probability_drop,
                        "evidence_text": evidence_text,
                    }
                )
                evidence_summary.append(
                    f"{modality.title()} #{rank}: [{item.start},{item.end}) "
                    f"logit drop={item.occlusion_score:.3f} {evidence_text}"
                )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "predicted_class": predicted_class,
                "predicted_annotation": ANNOTATIONS[predicted_class],
                "predicted_intensity": float(base_output["intensity"][0]),
                "prob_negative": probabilities[0],
                "prob_neutral": probabilities[1],
                "prob_positive": probabilities[2],
                "main_modality": main_modality,
                "text_contribution": contributions["text"],
                "audio_contribution": contributions["audio"],
                "vision_contribution": contributions["vision"],
                "text_logit_drop": deltas["text"],
                "audio_logit_drop": deltas["audio"],
                "vision_logit_drop": deltas["vision"],
                "text_evidence": text_evidence,
                "audio_start_time": audio_times[0],
                "audio_end_time": audio_times[1],
                "vision_start_time": vision_times[0],
                "vision_end_time": vision_times[1],
                "keyframe_path": str(keyframe_path or ""),
            }
        )
        curve_path = output_dir / "curves" / f"{sample_id}.npz"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(curve_path, **curves)
        if not args.no_cards:
            create_explanation_card(
                output_dir / "cards" / f"{sample_id}.png",
                sample_id=sample_id,
                predicted_annotation=ANNOTATIONS[predicted_class],
                predicted_intensity=float(base_output["intensity"][0]),
                probabilities=probabilities,
                main_modality=main_modality,
                contributions=contributions,
                curves=curves,
                raw_text=str(raw_sample.get("raw_text", "")),
                evidence_summary=evidence_summary,
                keyframe_path=keyframe_path,
            )
        print(f"[{file_index + 1}/{min(len(files), args.max_samples or len(files))}] {sample_id}")

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "attachment4_predictions_explanations.csv", summary_rows)
    write_csv(output_dir / "attachment4_evidence_details.csv", detail_rows)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "alignment": alignment,
                "input_dir": str(input_dir),
                "video_dir": str(video_dir),
                "window_rate": args.window_rate,
                "stride_rate": args.stride_rate,
                "top_k": args.top_k,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
