"""Constrained, fail-closed blending for nested per-task diversity experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BlendDecision:
    weights: tuple[float, ...]
    weight_id: str
    component_mae: float | None
    p95_absolute_error: float | None
    relative_improvement: float
    fallback_used: bool
    failure_reason: str | None
    selection_scores: tuple[dict[str, Any], ...]


def component_equal_mae(
    truth: Sequence[Any],
    prediction: Sequence[Any],
    components: Sequence[Any],
) -> float:
    """Average observation MAE within components, then weight components equally."""

    import numpy as np

    target = _finite_vector(truth, name="truth")
    estimate = _finite_vector(prediction, name="prediction")
    component = np.asarray([str(value).strip() for value in components], dtype=object)
    if target.shape != estimate.shape or target.shape != component.shape:
        raise ValueError("truth, prediction, and components must have the same shape")
    if any(not value for value in component):
        raise ValueError("components must be non-empty")
    values = [
        float(np.mean(np.abs(target[component == value] - estimate[component == value])))
        for value in np.unique(component)
    ]
    if not values:
        raise ValueError("component-equal MAE requires at least one row")
    result = float(np.mean(values))
    if not math.isfinite(result):
        raise RuntimeError("component-equal MAE is non-finite")
    return result


def p95_absolute_error(truth: Sequence[Any], prediction: Sequence[Any]) -> float:
    import numpy as np

    target = _finite_vector(truth, name="truth")
    estimate = _finite_vector(prediction, name="prediction")
    if target.shape != estimate.shape or target.size == 0:
        raise ValueError("truth and prediction must be non-empty with the same shape")
    result = float(np.quantile(np.abs(target - estimate), 0.95))
    if not math.isfinite(result):
        raise RuntimeError("P95 absolute error is non-finite")
    return result


def mean_seed_predictions(seed_predictions: Sequence[Sequence[Any]]) -> Any:
    import numpy as np

    if len(seed_predictions) < 3:
        raise ValueError("NPDB requires at least three seed prediction vectors")
    arrays = [_finite_vector(values, name="seed prediction") for values in seed_predictions]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("seed prediction vectors must have the same shape")
    result = np.mean(np.stack(arrays, axis=0), axis=0)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("seed-mean predictions are non-finite")
    return result


def validate_weight_grid(
    weights: Sequence[Sequence[Any]],
    *,
    predictor_count: int,
    minimum_reference_weight: float = 0.5,
) -> tuple[tuple[float, ...], ...]:
    if predictor_count < 2:
        raise ValueError("NPDB requires at least two predictors")
    minimum = float(minimum_reference_weight)
    if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_reference_weight must be in [0, 1]")
    normalized: list[tuple[float, ...]] = []
    for raw in weights:
        row = tuple(float(value) for value in raw)
        if len(row) != predictor_count:
            raise ValueError("weight vector length does not match predictor count")
        if not all(math.isfinite(value) and value >= 0.0 for value in row):
            raise ValueError("weights must be finite and non-negative")
        if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("weights must sum to one")
        if row[0] < minimum:
            raise ValueError("reference weight is below the registered minimum")
        if row in normalized:
            raise ValueError("weight grid contains duplicates")
        normalized.append(row)
    reference = (1.0,) + (0.0,) * (predictor_count - 1)
    if reference not in normalized:
        raise ValueError("weight grid must include the exact reference fallback")
    return tuple(normalized)


def weight_id(weights: Sequence[Any]) -> str:
    return "w_" + "_".join(f"{float(value):.2f}" for value in weights)


def reference_fallback_decision(
    predictor_count: int,
    *,
    failure_reason: str,
) -> BlendDecision:
    if predictor_count < 2:
        raise ValueError("NPDB fallback requires at least two predictors")
    reason = str(failure_reason).strip()
    if not reason:
        raise ValueError("NPDB fallback requires a failure reason")
    weights = (1.0,) + (0.0,) * (predictor_count - 1)
    score = {
        "weights": weights,
        "weight_id": weight_id(weights),
        "component_mae": None,
        "p95_absolute_error": None,
        "relative_improvement": 0.0,
        "selection_gate_pass": True,
        "is_reference": True,
        "failure_reason": reason,
    }
    return BlendDecision(
        weights=weights,
        weight_id=weight_id(weights),
        component_mae=None,
        p95_absolute_error=None,
        relative_improvement=0.0,
        fallback_used=True,
        failure_reason=reason,
        selection_scores=(score,),
    )


def blend_predictions(
    predictions: Sequence[Sequence[Any] | None],
    weights: Sequence[Any],
) -> Any:
    """Blend predictors while preserving a bit-exact reference-only fallback."""

    import numpy as np

    row = tuple(float(value) for value in weights)
    if len(predictions) != len(row) or not predictions:
        raise ValueError("prediction and weight counts must match")
    if row[0] == 1.0 and all(value == 0.0 for value in row[1:]):
        return np.asarray(predictions[0]).copy()
    if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("blend weights must sum to one")
    active: list[tuple[Any, float]] = []
    expected_shape = None
    for values, value in zip(predictions, row, strict=True):
        if value == 0.0:
            continue
        if values is None:
            raise ValueError("active blend predictor is unavailable")
        array = _finite_vector(values, name="active blend prediction")
        expected_shape = array.shape if expected_shape is None else expected_shape
        if array.shape != expected_shape:
            raise ValueError("active prediction vectors must have the same shape")
        active.append((array, value))
    result = np.zeros(expected_shape, dtype=float)
    for array, value in active:
        result += array * value
    if not np.all(np.isfinite(result)):
        raise RuntimeError("blended predictions are non-finite")
    return result


def select_blend_weights(
    truth: Sequence[Any],
    predictions: Mapping[str, Sequence[Any]],
    components: Sequence[Any],
    *,
    predictor_order: Sequence[str],
    weight_grid: Sequence[Sequence[Any]],
    minimum_relative_improvement: float = 0.01,
    maximum_p95_ratio: float = 1.05,
) -> BlendDecision:
    if not predictor_order or predictor_order[0] != "reference":
        raise ValueError("predictor_order must start with reference")
    if set(predictions) != set(predictor_order):
        raise ValueError("prediction roles do not match predictor_order")
    minimum = float(minimum_relative_improvement)
    p95_ratio = float(maximum_p95_ratio)
    if not math.isfinite(minimum) or not 0.0 <= minimum < 1.0:
        raise ValueError("minimum_relative_improvement must be in [0, 1)")
    if not math.isfinite(p95_ratio) or p95_ratio < 1.0:
        raise ValueError("maximum_p95_ratio must be finite and at least one")
    grid = validate_weight_grid(weight_grid, predictor_count=len(predictor_order))
    ordered = [predictions[name] for name in predictor_order]
    reference_weights = (1.0,) + (0.0,) * (len(predictor_order) - 1)
    reference = blend_predictions(ordered, reference_weights)
    reference_mae = component_equal_mae(truth, reference, components)
    reference_p95 = p95_absolute_error(truth, reference)
    scores: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for weights in grid:
        prediction = blend_predictions(ordered, weights)
        mae = component_equal_mae(truth, prediction, components)
        tail = p95_absolute_error(truth, prediction)
        relative = (reference_mae - mae) / reference_mae if reference_mae > 0.0 else 0.0
        is_reference = weights == reference_weights
        passes = is_reference or (
            relative >= minimum and tail <= reference_p95 * p95_ratio
        )
        score = {
            "weights": weights,
            "weight_id": weight_id(weights),
            "component_mae": mae,
            "p95_absolute_error": tail,
            "relative_improvement": relative,
            "selection_gate_pass": passes,
            "is_reference": is_reference,
        }
        scores.append(score)
        if passes:
            eligible.append(score)
    selected = min(
        eligible,
        key=lambda item: (
            item["component_mae"],
            item["p95_absolute_error"],
            -item["weights"][0],
            item["weight_id"],
        ),
    )
    return BlendDecision(
        weights=tuple(selected["weights"]),
        weight_id=str(selected["weight_id"]),
        component_mae=float(selected["component_mae"]),
        p95_absolute_error=float(selected["p95_absolute_error"]),
        relative_improvement=float(selected["relative_improvement"]),
        fallback_used=bool(selected["is_reference"]),
        failure_reason=None,
        selection_scores=tuple(scores),
    )


def _finite_vector(values: Sequence[Any], *, name: str) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array
