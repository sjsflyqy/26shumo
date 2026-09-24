from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("accuracy", "macro_f1", "mae", "pearson")
VIEWS = ("valid_clean", "valid_masked", "test_clean", "test_masked")
ARGUMENTS = (
    "alignment", "seed", "epochs", "batch_size", "av_encoder", "encoder_layers",
    "fusion_type", "fusion_levels", "distill", "min_mask_rate", "max_mask_rate",
    "min_mask_spans", "max_mask_spans", "sync_probability", "mask_combinations",
    "mask_locations", "valid_mask_rate", "emotion_mask_probability", "emotion_mask_warmup_epochs",
    "emotion_mask_ramp_epochs", "latent_generator", "generator_window",
    "reconstruction_weight", "confidence_weight",
)


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(inputs: list[Path], output_dir: Path) -> tuple[Path, Path]:
    if not inputs:
        raise ValueError("At least one experiment directory is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for directory in inputs:
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        arguments = metrics.get("arguments", {})
        row: dict[str, Any] = {"experiment": directory.name, "best_epoch": metrics.get("best_epoch")}
        row.update({
            key: ";".join(value) if isinstance(value := arguments.get(key, ""), list) else value
            for key in ARGUMENTS
        })
        for view in VIEWS:
            row.update({f"{view}_{metric}": metrics.get(view, {}).get(metric, "") for metric in METRICS})
        metric_rows.append(row)

        history_path = directory / "history.json"
        if not history_path.is_file():
            continue
        for record in json.loads(history_path.read_text(encoding="utf-8")):
            history_row: dict[str, Any] = {
                "experiment": directory.name,
                "alignment": arguments.get("alignment", ""),
                "epoch": record.get("epoch", ""),
                "best_epoch": metrics.get("best_epoch", ""),
                "learning_rate": record.get("learning_rate", ""),
                "selection_score": record.get("selection_score", ""),
                "emotion_mask_probability": record.get("emotion_mask_probability", ""),
            }
            for view in ("train", "valid_clean", "valid_masked"):
                history_row.update({f"{view}_{key}": value for key, value in record.get(view, {}).items()})
            history_rows.append(history_row)

    metric_columns = ["experiment", "best_epoch", *ARGUMENTS]
    metric_columns += [f"{view}_{metric}" for view in VIEWS for metric in METRICS]
    history_columns = list(dict.fromkeys(key for row in history_rows for key in row))
    metric_output = output_dir / "model_metrics.csv"
    history_output = output_dir / "training_history.csv"
    _write_rows(metric_output, metric_rows, metric_columns)
    _write_rows(history_output, history_rows, history_columns)
    return metric_output, history_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export M2 model metrics and epoch histories to CSV")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in export_tables(args.inputs, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
