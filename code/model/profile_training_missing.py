from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import infer_alignment, load_pickle


def _observed_rows(values: np.ndarray) -> np.ndarray:
    return np.any(np.abs(values) > 1e-12, axis=-1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(indices)]
    return [(int(indices[start]), int(indices[end - 1] + 1)) for start, end in zip(starts, ends)]


def _write(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def profile_training_missing(data_path: Path, output_dir: Path, max_samples: int | None = None) -> tuple[Path, Path]:
    data = load_pickle(data_path)
    train = data["train"]  # Never inspect attachment 3 or validation/test to design training masks.
    alignment = infer_alignment(train)
    text_attention = np.asarray(train["text_bert"])[:, 1, :] > 0
    audio = np.asarray(train["audio"])
    vision = np.asarray(train["vision"])
    count = len(text_attention) if max_samples is None else min(max_samples, len(text_attention))
    ids = train.get("id")
    rows: list[dict[str, object]] = []
    valid_totals: dict[str, int] = defaultdict(int)
    for index in range(count):
        text_observed = text_attention[index]
        text_valid = np.zeros_like(text_observed)
        text_positions = np.flatnonzero(text_observed)
        if text_positions.size:
            text_valid[: text_positions[-1] + 1] = True
            text_valid[text_positions[0]] = False  # [CLS]
            text_valid[text_positions[-1]] = False  # [SEP]
        modalities = {"T": (text_observed, text_valid)}
        for symbol, key, values in (("A", "audio", audio), ("V", "vision", vision)):
            observed = _observed_rows(values[index])
            if alignment == "aligned":
                valid = text_valid.copy()
            else:
                nonzero = np.flatnonzero(observed)
                extent = int(nonzero[-1] + 1) if nonzero.size else 0
                length_key = f"{key}_lengths"
                if length_key in train:
                    extent = max(extent, int(np.asarray(train[length_key]).reshape(-1)[index]))
                valid = np.arange(len(observed)) < min(extent, len(observed))
            modalities[symbol] = observed, valid
        for symbol, (observed, valid) in modalities.items():
            valid_length = int(valid.sum())
            valid_totals[symbol] += valid_length
            seen = np.flatnonzero(observed & valid)
            if seen.size < 2:
                continue
            # Only bounded internal zero runs are identifiable as local gaps.
            # Leading/trailing zeros may instead be padding or extraction artifacts.
            missing = valid & ~observed
            missing[: seen[0] + 1] = False
            missing[seen[-1] :] = False
            for start, end in _runs(missing):
                length = end - start
                center = (start + end) / (2 * max(valid_length, 1))
                location = "begin" if center < 1 / 3 else "end" if center > 2 / 3 else "middle"
                rows.append({
                    "sample_id": str(ids[index]) if ids is not None else str(index),
                    "alignment": alignment,
                    "modality": symbol,
                    "start": start,
                    "end_exclusive": end,
                    "length": length,
                    "valid_length": valid_length,
                    "relative_length": length / max(valid_length, 1),
                    "relative_center": center,
                    "location": location,
                })
    summary: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        summary[(str(row["modality"]), str(row["location"]))].append(row)
    summary_rows = []
    for (symbol, location), group in sorted(summary.items()):
        lengths = np.asarray([float(row["relative_length"]) for row in group])
        summary_rows.append({
            "alignment": alignment,
            "modality": symbol,
            "location": location,
            "span_count": len(group),
            "sample_count": count,
            "valid_position_count": valid_totals[symbol],
            "mean_relative_length": float(lengths.mean()),
            "median_relative_length": float(np.median(lengths)),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    detail = output_dir / "train_missing_spans.csv"
    aggregate = output_dir / "train_missing_profile.csv"
    _write(detail, rows, ["sample_id", "alignment", "modality", "start", "end_exclusive", "length", "valid_length", "relative_length", "relative_center", "location"])
    _write(aggregate, summary_rows, ["alignment", "modality", "location", "span_count", "sample_count", "valid_position_count", "mean_relative_length", "median_relative_length"])
    return detail, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile identifiable natural gaps in attachment-2 training split only")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    for path in profile_training_missing(args.data, args.output_dir, args.max_samples):
        print(path)


if __name__ == "__main__":
    main()
