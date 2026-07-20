"""Metrics used by the Peptide-OmniPanel training stages."""

from __future__ import annotations

import math
from typing import Any


def _require_numeric_stack() -> tuple[Any, Any]:
    try:
        import numpy as np
        from sklearn import metrics
    except ImportError as exc:  # pragma: no cover - exercised on the training host.
        raise RuntimeError("NumPy and scikit-learn are required for model evaluation") from exc
    return np, metrics


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired_finite_arrays(
    y_true: Any,
    y_pred: Any,
    *,
    prediction_name: str,
) -> tuple[Any, Any, Any]:
    np, metric_stack = _require_numeric_stack()
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or prediction.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional")
    if len(truth) != len(prediction):
        raise ValueError(f"y_true and {prediction_name} must have the same length")
    mask = np.isfinite(truth) & np.isfinite(prediction)
    return truth[mask], prediction[mask], metric_stack


def _average_ranks(values: Any, np: Any) -> Any:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _binary_arrays(y_true: Any, probabilities: Any) -> tuple[Any, Any, Any]:
    truth, probability, metric_stack = _paired_finite_arrays(
        y_true,
        probabilities,
        prediction_name="probabilities",
    )
    np, _ = _require_numeric_stack()
    if len(truth) and not np.all(np.isin(truth, (0.0, 1.0))):
        raise ValueError("Binary labels must be 0 or 1")
    if len(probability) and not np.all((probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("Probabilities must be in [0, 1]")
    return truth.astype(int), probability, metric_stack


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | None]:
    truth, prediction, metrics = _paired_finite_arrays(
        y_true,
        y_pred,
        prediction_name="y_pred",
    )
    np, _ = _require_numeric_stack()
    if not len(truth):
        return {name: None for name in ("n", "mae", "rmse", "median_ae", "r2", "spearman")}
    residual = np.abs(truth - prediction)
    r2 = metrics.r2_score(truth, prediction) if len(truth) >= 2 else None
    if len(truth) >= 2:
        truth_rank = _average_ranks(truth, np)
        pred_rank = _average_ranks(prediction, np)
        if np.ptp(truth_rank) == 0.0 or np.ptp(pred_rank) == 0.0:
            spearman = None
        else:
            spearman = float(np.corrcoef(truth_rank, pred_rank)[0, 1])
    else:
        spearman = None
    return {
        "n": float(len(truth)),
        "mae": float(metrics.mean_absolute_error(truth, prediction)),
        "rmse": float(metrics.root_mean_squared_error(truth, prediction)),
        "median_ae": float(np.median(residual)),
        "r2": finite_or_none(r2),
        "spearman": finite_or_none(spearman),
    }


def choose_binary_threshold(y_true: Any, probabilities: Any) -> float:
    truth, probability, metrics = _binary_arrays(y_true, probabilities)
    np, _ = _require_numeric_stack()
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate(([0.5], probability)))
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        predicted = (probability >= threshold).astype(int)
        score = metrics.matthews_corrcoef(truth, predicted)
        numeric_threshold = float(threshold)
        key = (float(score), -abs(numeric_threshold - 0.5), -numeric_threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = numeric_threshold
    return best_threshold


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    numeric_threshold = finite_or_none(threshold)
    if numeric_threshold is None or not 0.0 <= numeric_threshold <= 1.0:
        raise ValueError("Classification threshold must be in [0, 1]")
    truth, probability, metrics = _binary_arrays(y_true, probabilities)
    np, _ = _require_numeric_stack()
    if not len(truth):
        return {
            name: None
            for name in (
                "n",
                "positive_fraction",
                "roc_auc",
                "pr_auc",
                "mcc",
                "balanced_accuracy",
                "brier",
            )
        }
    predicted = (probability >= numeric_threshold).astype(int)
    has_both_classes = len(np.unique(truth)) == 2
    return {
        "n": float(len(truth)),
        "positive_fraction": float(np.mean(truth)),
        "roc_auc": float(metrics.roc_auc_score(truth, probability)) if has_both_classes else None,
        "pr_auc": float(metrics.average_precision_score(truth, probability))
        if has_both_classes
        else None,
        "mcc": float(metrics.matthews_corrcoef(truth, predicted)) if has_both_classes else None,
        "balanced_accuracy": float(metrics.balanced_accuracy_score(truth, predicted))
        if has_both_classes
        else None,
        "brier": float(metrics.brier_score_loss(truth, probability)),
    }
