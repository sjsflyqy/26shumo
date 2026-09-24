from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "macro_f1", "mae", "pearson")
GROUP_FIELDS = ("modalities", "rate", "location", "span_count", "sync_mode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated missing-factor evaluations")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same length as --inputs")
    labels = args.labels or [path.parent.name for path in args.inputs]
    output_rows = []
    for label, path in zip(labels, args.inputs):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        clean_rows = [row for row in rows if row["modalities"] == "none"]
        if not clean_rows:
            raise ValueError(f"{path} does not contain a clean row")
        clean = {metric: float(clean_rows[0][metric]) for metric in METRICS}
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row["modalities"] != "none":
                groups[tuple(row.get(field, "") for field in GROUP_FIELDS)].append(row)
        for key, group in groups.items():
            summary = {"experiment": label}
            summary.update(dict(zip(GROUP_FIELDS, key)))
            summary["repeats"] = len(group)
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in group])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary["accuracy_drop"] = clean["accuracy"] - summary["accuracy_mean"]
            summary["macro_f1_drop"] = clean["macro_f1"] - summary["macro_f1_mean"]
            summary["mae_increase"] = summary["mae_mean"] - clean["mae"]
            summary["pearson_drop"] = clean["pearson"] - summary["pearson_mean"]
            output_rows.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]) if output_rows else []
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(args.output)


if __name__ == "__main__":
    main()
