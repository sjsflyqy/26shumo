#!/usr/bin/env python3
"""Pure NumPy baseline for the unaligned attachment-2 features.

This script intentionally has no pandas/scikit-learn/PyTorch dependency.  It is
the first-stage sanity baseline, not the final temporal deep model:

* text: stable hashed TF-IDF from raw_text (compatible with attachment 3);
* audio/vision: non-zero-frame masked mean/std plus coverage statistics;
* classification: weighted ridge one-vs-rest scores;
* regression: ridge regression, clipped to [-3, 3];
* optional controlled contiguous-span masking on valid/test;
* optional blind prediction for attachment 3 (unaligned version).

Never unpickle untrusted files. The competition files are assumed trusted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CLASS_NAMES = np.asarray(["Negative", "Neutral", "Positive"])
TOKEN_RE = re.compile(r"[a-z0-9']+")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "data" / "attachment2" / "unaligned_50.pkl",
    )
    parser.add_argument(
        "--attachment3",
        type=Path,
        default=here / "data" / "attachment3" / "未对齐版本",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "outputs" / "numpy_baseline")
    parser.add_argument("--text-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mask-eval", action="store_true")
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument(
        "--mask-types",
        nargs="+",
        default=["T", "A", "V", "TA", "TV", "AV", "TAV"],
        choices=["T", "A", "V", "TA", "TV", "AV", "TAV"],
    )
    parser.add_argument("--predict-attachment3", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def text_terms(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    bigrams = [f"{a}__{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


def stable_bin(term: str, dim: int) -> int:
    return zlib.crc32(term.encode("utf-8")) % dim


def fit_hashed_idf(texts: Sequence[str], dim: int) -> np.ndarray:
    df = np.zeros(dim, dtype=np.int64)
    for text in texts:
        bins = {stable_bin(term, dim) for term in text_terms(str(text))}
        if bins:
            df[np.fromiter(bins, dtype=np.int64)] += 1
    return (np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0).astype(np.float64)


def transform_text(texts: Sequence[str], idf: np.ndarray) -> np.ndarray:
    dim = len(idf)
    out = np.zeros((len(texts), dim), dtype=np.float64)
    for i, text in enumerate(texts):
        bins = np.fromiter(
            (stable_bin(term, dim) for term in text_terms(str(text))),
            dtype=np.int64,
        )
        if bins.size == 0:
            continue
        counts = np.bincount(bins, minlength=dim)
        nz = np.flatnonzero(counts)
        out[i, nz] = (1.0 + np.log(counts[nz])) * idf[nz]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out /= np.maximum(norms, 1e-12)
    return out


def modality_statistics(values: np.ndarray) -> np.ndarray:
    """Pool N x T x D sequences while ignoring all-zero missing/padded rows."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected N x T x D array, got {values.shape}")
    observed = np.any(values != 0.0, axis=2)
    counts = observed.sum(axis=1).astype(np.float64)
    denom = np.maximum(counts[:, None], 1.0)

    sums = values.sum(axis=1)
    sum_squares = np.einsum("ntd,ntd->nd", values, values, optimize=True)
    means = sums / denom
    variances = np.maximum(sum_squares / denom - means * means, 0.0)
    stds = np.sqrt(variances)

    # Last observed frame defines a length estimate that is also available for
    # attachment 3, where explicit *_lengths fields are absent.
    any_observed = observed.any(axis=1)
    reverse_first = np.argmax(observed[:, ::-1], axis=1)
    extents = np.where(any_observed, observed.shape[1] - reverse_first, 0).astype(np.float64)
    coverage_total = counts / observed.shape[1]
    extent_ratio = extents / observed.shape[1]
    internal_gap_ratio = 1.0 - counts / np.maximum(extents, 1.0)
    extras = np.column_stack([coverage_total, extent_ratio, internal_gap_ratio])
    return np.concatenate([means, stds, extras], axis=1)


def extract_features(split: dict, idf: np.ndarray) -> np.ndarray:
    texts = np.asarray(split["raw_text"]).reshape(-1).astype(str)
    audio = np.asarray(split["audio"])
    vision = np.asarray(split["vision"])
    if audio.ndim == 2:
        audio = audio[None, ...]
    if vision.ndim == 2:
        vision = vision[None, ...]
    return np.concatenate(
        [transform_text(texts, idf), modality_statistics(audio), modality_statistics(vision)],
        axis=1,
    )


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    xb = add_intercept(x)
    target = np.asarray(y, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if sample_weight is not None:
        root_weight = np.sqrt(np.asarray(sample_weight, dtype=np.float64))[:, None]
        xb_fit = xb * root_weight
        target_fit = target * root_weight
    else:
        xb_fit, target_fit = xb, target
    gram = xb_fit.T @ xb_fit
    penalty = np.eye(gram.shape[0], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0  # Do not regularize the intercept.
    coef = np.linalg.solve(gram + penalty, xb_fit.T @ target_fit)
    return coef


def predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coef


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float(np.mean(y_true == y_pred))
    f1s = []
    for cls in range(3):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": float(np.mean(f1s))}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = math.sqrt(float(yt @ yt) * float(yp @ yp))
    corr = float((yt @ yp) / denom) if denom > 0 else 0.0
    return {"mae": mae, "corr": corr}


def evaluate(
    x: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    cls_coef: np.ndarray,
    reg_coef: np.ndarray,
) -> dict[str, float]:
    scores = predict_ridge(x, cls_coef)
    pred_cls = scores.argmax(axis=1)
    pred_reg = np.clip(predict_ridge(x, reg_coef).reshape(-1), -3.0, 3.0)
    return classification_metrics(y_cls, pred_cls) | regression_metrics(y_reg, pred_reg)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def select_models(
    x_train: np.ndarray,
    train: dict,
    x_valid: np.ndarray,
    valid: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_train_cls = np.asarray(train["classification_labels"], dtype=np.int64)
    y_valid_cls = np.asarray(valid["classification_labels"], dtype=np.int64)
    y_train_reg = np.asarray(train["regression_labels"], dtype=np.float64)
    y_valid_reg = np.asarray(valid["regression_labels"], dtype=np.float64)
    y_onehot = np.eye(3, dtype=np.float64)[y_train_cls]
    counts = np.bincount(y_train_cls, minlength=3).astype(np.float64)
    balanced_weights = len(y_train_cls) / (3.0 * counts[y_train_cls])

    best_cls = None
    best_cls_key = None
    cls_trials = []
    for weighted in (False, True):
        for alpha in (0.1, 1.0, 10.0, 100.0):
            coef = fit_ridge(
                x_train,
                y_onehot,
                alpha,
                balanced_weights if weighted else None,
            )
            pred = predict_ridge(x_valid, coef).argmax(axis=1)
            metrics = classification_metrics(y_valid_cls, pred)
            trial = {"alpha": alpha, "weighted": weighted} | metrics
            cls_trials.append(trial)
            key = (0.5 * (metrics["accuracy"] + metrics["macro_f1"]), metrics["macro_f1"])
            if best_cls_key is None or key > best_cls_key:
                best_cls_key, best_cls = key, coef

    best_reg = None
    best_reg_key = None
    reg_trials = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        coef = fit_ridge(x_train, y_train_reg, alpha)
        pred = np.clip(predict_ridge(x_valid, coef).reshape(-1), -3.0, 3.0)
        metrics = regression_metrics(y_valid_reg, pred)
        reg_trials.append({"alpha": alpha} | metrics)
        key = (-metrics["mae"], metrics["corr"])
        if best_reg_key is None or key > best_reg_key:
            best_reg_key, best_reg = key, coef

    return best_cls, best_reg, {"classification_trials": cls_trials, "regression_trials": reg_trials}


def estimated_lengths(values: np.ndarray) -> np.ndarray:
    observed = np.any(values != 0.0, axis=2)
    any_observed = observed.any(axis=1)
    reverse_first = np.argmax(observed[:, ::-1], axis=1)
    return np.where(any_observed, observed.shape[1] - reverse_first, 0).astype(np.int64)


def span_bounds(lengths: np.ndarray, rate: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    locations = rng.random(len(lengths))
    result = []
    for length, location in zip(lengths, locations):
        span = max(1, int(round(float(length) * rate))) if length else 0
        max_start = max(int(length) - span, 0)
        start = int(round(location * max_start)) if span else 0
        result.append((start, start + span))
    return result


def mask_texts(texts: Sequence[str], rate: float, locations: np.ndarray) -> np.ndarray:
    output = []
    for text, location in zip(texts, locations):
        tokens = str(text).split()
        span = max(1, int(round(len(tokens) * rate))) if tokens else 0
        max_start = max(len(tokens) - span, 0)
        start = int(round(float(location) * max_start)) if span else 0
        output.append(" ".join(tokens[:start] + tokens[start + span :]))
    return np.asarray(output)


def make_masked_split(split: dict, mask_type: str, rate: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(np.asarray(split["raw_text"]).reshape(-1))
    locations = rng.random(n)
    output = {
        "raw_text": np.asarray(split["raw_text"]).reshape(-1).copy(),
        "audio": np.asarray(split["audio"]),
        "vision": np.asarray(split["vision"]),
    }
    if "T" in mask_type:
        output["raw_text"] = mask_texts(output["raw_text"], rate, locations)
    for symbol, key in (("A", "audio"), ("V", "vision")):
        if symbol not in mask_type:
            continue
        values = np.asarray(split[key]).copy()
        lengths = estimated_lengths(values)
        for i, (length, location) in enumerate(zip(lengths, locations)):
            span = max(1, int(round(float(length) * rate))) if length else 0
            max_start = max(int(length) - span, 0)
            start = int(round(float(location) * max_start)) if span else 0
            values[i, start : start + span] = 0.0
        output[key] = values
    return output


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_attachment3(
    directory: Path,
    idf: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    cls_coef: np.ndarray,
    reg_coef: np.ndarray,
    output_path: Path,
) -> None:
    rows = []
    for path in sorted(directory.glob("*.pkl")):
        obj = load_pickle(path)
        sample = obj.get("test", obj)
        x = extract_features(sample, idf)
        x = standardize(x, mean, scale)
        scores = predict_ridge(x, cls_coef)
        probs = softmax(scores)[0]
        pred_cls = int(np.argmax(probs))
        pred_reg = float(np.clip(predict_ridge(x, reg_coef).reshape(-1)[0], -3.0, 3.0))
        match = re.search(r"(\d+)(?=\.pkl$)", path.name)
        sample_id = match.group(1) if match else path.stem
        rows.append(
            {
                "sample_id": sample_id,
                "file": path.name,
                "predicted_class": pred_cls,
                "predicted_annotation": CLASS_NAMES[pred_cls],
                "predicted_intensity": f"{pred_reg:.6f}",
                "prob_negative": f"{probs[0]:.6f}",
                "prob_neutral": f"{probs[1]:.6f}",
                "prob_positive": f"{probs[2]:.6f}",
            }
        )
    write_csv(
        output_path,
        rows,
        [
            "sample_id",
            "file",
            "predicted_class",
            "predicted_annotation",
            "predicted_intensity",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
        ],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.data} ...", flush=True)
    data = load_pickle(args.data)
    required_splits = {"train", "valid", "test"}
    if set(data) != required_splits:
        raise ValueError(f"Expected splits {required_splits}, got {set(data)}")
    train, valid, test = data["train"], data["valid"], data["test"]
    print("Fitting hashed TF-IDF and extracting pooled features ...", flush=True)
    idf = fit_hashed_idf(np.asarray(train["raw_text"]).astype(str), args.text_dim)
    x_train_raw = extract_features(train, idf)
    x_valid_raw = extract_features(valid, idf)
    x_test_raw = extract_features(test, idf)
    mean, scale = fit_standardizer(x_train_raw)
    x_train = standardize(x_train_raw, mean, scale)
    x_valid = standardize(x_valid_raw, mean, scale)
    x_test = standardize(x_test_raw, mean, scale)

    print(f"Feature dimensions: {x_train.shape}", flush=True)
    cls_coef, reg_coef, trials = select_models(x_train, train, x_valid, valid)
    results = {
        "data": str(args.data),
        "feature_dimension": int(x_train.shape[1]),
        "splits": {key: len(value["raw_text"]) for key, value in data.items()},
        "clean": {
            "valid": evaluate(
                x_valid,
                valid["classification_labels"],
                valid["regression_labels"],
                cls_coef,
                reg_coef,
            ),
            "test": evaluate(
                x_test,
                test["classification_labels"],
                test["regression_labels"],
                cls_coef,
                reg_coef,
            ),
        },
        "selection": trials,
    }
    print(json.dumps(results["clean"], ensure_ascii=False, indent=2), flush=True)

    np.savez_compressed(
        args.output_dir / "model.npz",
        idf=idf,
        mean=mean,
        scale=scale,
        cls_coef=cls_coef,
        reg_coef=reg_coef,
        text_dim=np.asarray([args.text_dim], dtype=np.int64),
    )

    if args.mask_eval:
        mask_rows = []
        for mask_type in args.mask_types:
            for rate in args.mask_rates:
                masked = make_masked_split(
                    valid,
                    mask_type,
                    rate,
                    args.seed + int(rate * 1000) + sum(map(ord, mask_type)),
                )
                x_masked = standardize(extract_features(masked, idf), mean, scale)
                metrics = evaluate(
                    x_masked,
                    valid["classification_labels"],
                    valid["regression_labels"],
                    cls_coef,
                    reg_coef,
                )
                row = {"split": "valid", "mask_type": mask_type, "mask_rate": rate} | metrics
                mask_rows.append(row)
                print(row, flush=True)
        results["masked_valid"] = mask_rows
        write_csv(
            args.output_dir / "mask_evaluation.csv",
            mask_rows,
            ["split", "mask_type", "mask_rate", "accuracy", "macro_f1", "mae", "corr"],
        )

    if args.predict_attachment3:
        predict_attachment3(
            args.attachment3,
            idf,
            mean,
            scale,
            cls_coef,
            reg_coef,
            args.output_dir / "attachment3_predictions_baseline.csv",
        )

    save_json(args.output_dir / "metrics.json", results)
    print(f"Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
