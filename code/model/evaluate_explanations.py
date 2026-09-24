from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import M2Config
from .data import AVStandardizer, infer_alignment, load_pickle, make_mosei_dataset
from .engine import move_to_device
from .explain import (
    candidate_windows,
    faithfulness_scores,
    internal_importance,
    random_matched_windows,
    score_occlusion_windows,
    select_non_overlapping,
)
from .metrics import classification_regression_metrics
from .model import M2Model, load_checkpoint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate explanation faithfulness on attachment 2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/attachment2/aligned_50.pkl",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--alignment", choices=("auto", "aligned", "unaligned"), default="auto"
    )
    parser.add_argument("--ranking", choices=("internal", "occlusion"), default="internal")
    parser.add_argument("--window-rate", type=float, default=0.15)
    parser.add_argument("--stride-rate", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--occlusion-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data = load_pickle(args.data)
    split = data[args.split]
    detected = infer_alignment(split)
    alignment = detected if args.alignment == "auto" else args.alignment
    checkpoint_alignment = checkpoint.get("alignment")
    if alignment != detected:
        raise ValueError(f"Data is {detected}, but --alignment={alignment}")
    if checkpoint_alignment in {"aligned", "unaligned"} and checkpoint_alignment != alignment:
        raise ValueError(f"Checkpoint is {checkpoint_alignment}, but data is {alignment}")
    model = M2Model(M2Config.from_dict(checkpoint["model_config"])).to(device)
    load_checkpoint_state(model, checkpoint)
    model.eval()
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    dataset = make_mosei_dataset(
        split, standardizer, max_samples=args.max_samples, alignment=alignment
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_dir = args.output_dir or args.checkpoint.parent / f"explanation_eval_{alignment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    class_truth = []
    class_prediction = []
    regression_truth = []
    regression_prediction = []

    for index, cpu_batch in enumerate(loader):
        batch = move_to_device(cpu_batch, device)
        with torch.no_grad():
            base_output = model(batch)
        predicted_class = int(base_output["class_logits"][0].argmax())
        class_truth.append(int(batch["class_label"][0]))
        class_prediction.append(predicted_class)
        regression_truth.append(float(batch["regression_label"][0]))
        regression_prediction.append(float(base_output["intensity"][0]))
        curves = internal_importance(base_output, batch)
        candidates = candidate_windows(batch, curves, args.window_rate, args.stride_rate)
        if args.ranking == "occlusion":
            candidates = score_occlusion_windows(
                model,
                batch,
                base_output,
                candidates,
                batch_size=args.occlusion_batch_size,
            )
            score_name = "occlusion_score"
        else:
            score_name = "internal_score"
        selected = select_non_overlapping(candidates, args.top_k, score=score_name)
        if args.ranking == "internal":
            selected = score_occlusion_windows(
                model,
                batch,
                base_output,
                selected,
                batch_size=args.occlusion_batch_size,
            )
        random_metrics = []
        faithfulness = None
        for _ in range(args.random_repeats):
            random_windows = random_matched_windows(batch, selected, rng)
            result = faithfulness_scores(
                model, batch, base_output, selected, random_windows
            )
            faithfulness = result if faithfulness is None else faithfulness
            random_metrics.append(
                (result["random_comprehensiveness"], result["random_aopc"])
            )
        if faithfulness is None:
            raise RuntimeError("No explanation windows were generated")
        random_comprehensiveness = float(np.mean([item[0] for item in random_metrics]))
        random_aopc = float(np.mean([item[1] for item in random_metrics]))
        rows.append(
            {
                "sample_id": str(cpu_batch["sample_id"][0]),
                "true_class": class_truth[-1],
                "predicted_class": predicted_class,
                "true_intensity": regression_truth[-1],
                "predicted_intensity": regression_prediction[-1],
                "comprehensiveness": faithfulness["comprehensiveness"],
                "sufficiency": faithfulness["sufficiency"],
                "aopc": faithfulness["aopc"],
                "random_comprehensiveness": random_comprehensiveness,
                "random_aopc": random_aopc,
                "comprehensiveness_gain": (
                    faithfulness["comprehensiveness"] - random_comprehensiveness
                ),
                "aopc_gain": faithfulness["aopc"] - random_aopc,
                "selected_windows": ";".join(
                    f"{item.modality}:{item.start}-{item.end}"
                    for item in selected
                ),
                "selected_internal_scores": ";".join(
                    f"{item.internal_score:.8f}" for item in selected
                ),
                "selected_occlusion_scores": ";".join(
                    f"{item.occlusion_score:.8f}" for item in selected
                ),
            }
        )
        print(f"[{index + 1}/{len(dataset)}] {rows[-1]['sample_id']}")

    predictive = classification_regression_metrics(
        np.asarray(class_truth),
        np.asarray(class_prediction),
        np.asarray(regression_truth),
        np.asarray(regression_prediction),
    )
    faithfulness_keys = (
        "comprehensiveness",
        "sufficiency",
        "aopc",
        "random_comprehensiveness",
        "random_aopc",
        "comprehensiveness_gain",
        "aopc_gain",
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "split": args.split,
        "alignment": alignment,
        "ranking": args.ranking,
        "samples": len(rows),
        "predictive_metrics": predictive,
        "faithfulness_metrics": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in faithfulness_keys
        },
        "settings": {
            "window_rate": args.window_rate,
            "stride_rate": args.stride_rate,
            "top_k": args.top_k,
            "random_repeats": args.random_repeats,
            "seed": args.seed,
        },
    }
    with (output_dir / "explanation_samples.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "explanation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
