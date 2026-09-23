from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import M2Config, SpanMaskConfig
from .data import AVStandardizer, AlignedMoseiDataset, load_pickle
from .engine import evaluate_model
from .model import M2Model, load_checkpoint_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled missing-factor evaluation for M2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/attachment2/aligned_50.pkl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--modalities", nargs="+", default=["T", "A", "V", "TA", "TV", "AV", "TAV"])
    parser.add_argument("--locations", nargs="+", default=["begin", "middle", "end"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = M2Model(M2Config.from_dict(checkpoint["model_config"])).to(device)
    load_checkpoint_state(model, checkpoint)
    standardizer = AVStandardizer.from_state_dict(checkpoint["standardizer"])
    split = load_pickle(args.data)[args.split]
    loader = DataLoader(
        AlignedMoseiDataset(split, standardizer),
        batch_size=args.batch_size,
        shuffle=False,
    )
    mask_config = SpanMaskConfig.from_dict(checkpoint["mask_config"])
    output = args.output or args.checkpoint.parent / f"{args.split}_missing_grid.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split", "modalities", "rate", "location", "repeat",
        "accuracy", "macro_f1", "mae", "pearson",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        clean = evaluate_model(model, loader, device)
        writer.writerow(
            {"split": args.split, "modalities": "none", "rate": 0.0, "location": "none", "repeat": 0} | clean
        )
        for modalities in args.modalities:
            for rate in args.rates:
                if rate == 0:
                    continue
                for location in args.locations:
                    for repeat in range(args.repeats):
                        metrics = evaluate_model(
                            model,
                            loader,
                            device,
                            mask_config=mask_config,
                            seed=args.seed + repeat,
                            mask_rate=rate,
                            mask_modalities=modalities,
                            mask_location=location,
                            mask_spans=1,
                        )
                        writer.writerow(
                            {
                                "split": args.split,
                                "modalities": modalities,
                                "rate": rate,
                                "location": location,
                                "repeat": repeat,
                            }
                            | metrics
                        )
                        handle.flush()
    print(output)


if __name__ == "__main__":
    main()
