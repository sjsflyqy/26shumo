from __future__ import annotations

import numpy as np


def classification_regression_metrics(
    class_truth: np.ndarray,
    class_prediction: np.ndarray,
    regression_truth: np.ndarray,
    regression_prediction: np.ndarray,
    num_classes: int = 3,
) -> dict[str, float]:
    class_truth = np.asarray(class_truth, dtype=np.int64)
    class_prediction = np.asarray(class_prediction, dtype=np.int64)
    regression_truth = np.asarray(regression_truth, dtype=np.float64)
    regression_prediction = np.asarray(regression_prediction, dtype=np.float64)
    accuracy = float(np.mean(class_truth == class_prediction))
    f1_scores = []
    for label in range(num_classes):
        true_positive = np.sum((class_truth == label) & (class_prediction == label))
        false_positive = np.sum((class_truth != label) & (class_prediction == label))
        false_negative = np.sum((class_truth == label) & (class_prediction != label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    mae = float(np.mean(np.abs(regression_truth - regression_prediction)))
    if regression_truth.size < 2 or np.std(regression_truth) == 0 or np.std(regression_prediction) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(regression_truth, regression_prediction)[0, 1])
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "mae": mae,
        "pearson": correlation,
    }


def composite_score(metrics: dict[str, float]) -> float:
    """Map the four official metrics to a transparent [roughly 0,1] score."""

    correlation = (max(-1.0, min(1.0, metrics["pearson"])) + 1.0) / 2.0
    mae_score = 1.0 - max(0.0, min(6.0, metrics["mae"])) / 6.0
    return float(
        0.25 * (metrics["accuracy"] + metrics["macro_f1"] + correlation + mae_score)
    )
