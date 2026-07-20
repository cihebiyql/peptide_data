"""Frozen external-regression metrics and gates for Peptide-OmniPanel V2.6."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionReference:
    component_mae: float
    rmse: float
    max_mae_ratio: float = 1.25
    max_rmse_ratio: float = 1.25
    max_bootstrap_upper_ratio: float = 1.50

    def __post_init__(self) -> None:
        values = (
            self.component_mae,
            self.rmse,
            self.max_mae_ratio,
            self.max_rmse_ratio,
            self.max_bootstrap_upper_ratio,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("reference metrics and ratios must be finite and positive")


def _finite_vector(values: list[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        ranks[order[position:end]] = rank
        position = end
    return ranks


def spearman_correlation(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    truth_rank = _average_ranks(truth)
    prediction_rank = _average_ranks(prediction)
    if np.ptp(truth_rank) == 0 or np.ptp(prediction_rank) == 0:
        return None
    value = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
    return value if math.isfinite(value) else None


def component_errors(
    component_ids: list[str],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    if len(component_ids) != len(truth):
        raise ValueError("component IDs and predictions are not row aligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    for component_id, observed, predicted in zip(
        component_ids, truth, prediction, strict=True
    ):
        normalized = str(component_id).strip()
        if not normalized:
            raise ValueError("component IDs must be non-empty")
        grouped[normalized].append(abs(float(observed) - float(predicted)))
    return {
        component_id: float(np.mean(np.asarray(errors, dtype=np.float64)))
        for component_id, errors in sorted(grouped.items())
    }


def bootstrap_component_mae(
    errors: dict[str, float],
    *,
    replicates: int = 2000,
    seed: int = 20260716,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    values = _finite_vector(list(errors.values()), name="component errors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    sampled = np.mean(values[indices], axis=1, dtype=np.float64)
    interval = np.quantile(sampled, [0.025, 0.975], method="linear")
    return {
        "replicates": replicates,
        "seed": seed,
        "sampling_unit": "component_with_replacement",
        "valid_replicates": int(len(sampled)),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
    }


def evaluate_regression(
    *,
    observation_ids: list[str],
    component_ids: list[str],
    truth: list[float],
    prediction: list[float],
    reference: RegressionReference,
    expected_observations: int,
    crossings: int,
    bundle_verified: bool,
    representation_failures: int,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260716,
) -> dict[str, Any]:
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("external observations must be unique")
    if not (
        len(observation_ids)
        == len(component_ids)
        == len(truth)
        == len(prediction)
    ):
        raise ValueError("external regression inputs are not row aligned")
    truth_array = _finite_vector(truth, name="truth")
    prediction_array = _finite_vector(prediction, name="prediction")
    errors = component_errors(component_ids, truth_array, prediction_array)
    absolute = np.abs(truth_array - prediction_array)
    component_mae = float(np.mean(np.asarray(list(errors.values()), dtype=np.float64)))
    rmse = float(np.sqrt(np.mean(np.square(truth_array - prediction_array))))
    p95 = float(np.quantile(absolute, 0.95, method="linear"))
    bootstrap = bootstrap_component_mae(
        errors,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    thresholds = {
        "component_mae": reference.component_mae * reference.max_mae_ratio,
        "rmse": reference.rmse * reference.max_rmse_ratio,
        "bootstrap_ci_high": (
            reference.component_mae * reference.max_bootstrap_upper_ratio
        ),
    }
    gates = {
        "complete_cohort": (
            len(observation_ids) == expected_observations
            and representation_failures == 0
        ),
        "zero_crossings": crossings == 0,
        "bundle_verified": bool(bundle_verified),
        "mae": component_mae <= thresholds["component_mae"],
        "rmse": rmse <= thresholds["rmse"],
        "bootstrap_upper": bootstrap["ci_high"] <= thresholds["bootstrap_ci_high"],
    }
    return {
        "observations": len(observation_ids),
        "components": len(errors),
        "component_mae": component_mae,
        "pooled_rmse": rmse,
        "pooled_p95_absolute_error": p95,
        "spearman": spearman_correlation(truth_array, prediction_array),
        "reference": {
            "component_mae": reference.component_mae,
            "rmse": reference.rmse,
        },
        "thresholds": thresholds,
        "bootstrap": bootstrap,
        "gates": gates,
        "single_source_point_validation_pass": all(gates.values()),
        "claim_scope": "retrospective_external_descriptive_only",
        "deployment_promotion_pass": False,
        "active_service_unchanged": True,
    }
