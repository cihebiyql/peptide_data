"""Fail-safe residual corrections and per-head promotion decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class HeadPromotionDecision:
    promoted: bool
    effective_lambda: float
    fallback: str
    failure_reasons: tuple[str, ...]


def apply_residual_correction(
    base_prediction: Any,
    residual: Any,
    *,
    lambda_value: float,
    task_kind: str,
    probability_epsilon: float = 1e-7,
) -> Any:
    """Apply a residual while preserving a bit-exact zero-lambda fallback."""

    import numpy as np

    base = np.asarray(base_prediction)
    numeric_lambda = float(lambda_value)
    if numeric_lambda == 0.0:
        return base.copy()
    if not math.isfinite(numeric_lambda):
        raise ValueError("lambda_value must be finite")

    correction = np.asarray(residual, dtype=float)
    if base.shape != correction.shape:
        raise ValueError("base prediction and residual must have the same shape")
    if not np.all(np.isfinite(correction)):
        raise ValueError("non-zero residual correction must be finite")
    if task_kind == "regression":
        return np.asarray(base, dtype=float) + numeric_lambda * correction
    if task_kind != "binary_classification":
        raise ValueError(f"unsupported task kind: {task_kind!r}")
    epsilon = float(probability_epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be finite and in (0, 0.5)")
    probability = np.asarray(base, dtype=float)
    if not np.all(np.isfinite(probability)) or not np.all(
        (probability >= 0.0) & (probability <= 1.0)
    ):
        raise ValueError("binary base predictions must be finite probabilities")
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)) + numeric_lambda * correction
    return 1.0 / (1.0 + np.exp(-logits))


def build_neighbor_feature_block(
    neighbors: Sequence[Mapping[str, Any] | None],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Represent missing and out-of-domain neighbor predictions explicitly."""

    values: list[float] = []
    masks: list[float] = []
    for neighbor in neighbors:
        available = bool(neighbor is not None and neighbor.get("available") is True)
        in_domain = bool(neighbor is not None and neighbor.get("in_domain") is True)
        prediction = None if neighbor is None else neighbor.get("prediction")
        try:
            numeric_prediction = float(prediction)
        except (TypeError, ValueError):
            numeric_prediction = math.nan
        usable = available and in_domain and math.isfinite(numeric_prediction)
        values.append(numeric_prediction if usable else 0.0)
        masks.append(1.0 if usable else 0.0)
    return tuple(values), tuple(masks)


def validate_component_fold_map(
    rows: Iterable[Mapping[str, Any]],
    *,
    component_field: str = "component_id",
    fold_field: str = "fold_id",
    expected_fold_count: int | None = None,
) -> dict[str, int]:
    """Require every release-global component to have exactly one family fold."""

    component_folds: dict[str, set[int]] = {}
    for row in rows:
        component = str(row.get(component_field, "")).strip()
        if not component:
            raise RuntimeError("component fold map contains a missing component_id")
        try:
            fold = int(row[fold_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("component fold map contains an invalid fold_id") from exc
        if fold < 0:
            raise RuntimeError("component fold map contains a negative fold_id")
        component_folds.setdefault(component, set()).add(fold)
    crossings = sorted(
        component for component, folds in component_folds.items() if len(folds) != 1
    )
    if crossings:
        raise RuntimeError(f"component crosses family-global folds: {crossings[:5]}")
    output = {component: next(iter(folds)) for component, folds in component_folds.items()}
    if expected_fold_count is not None:
        if expected_fold_count < 2:
            raise ValueError("expected_fold_count must be at least two")
        observed = set(output.values())
        expected = set(range(expected_fold_count))
        if observed != expected:
            raise RuntimeError(
                f"component fold map is not contiguous: observed={sorted(observed)} "
                f"expected={sorted(expected)}"
            )
    return output


def component_sample_weights(components: Sequence[Any]) -> tuple[float, ...]:
    """Give each task-component equal total weight regardless of row multiplicity."""

    counts: dict[str, int] = {}
    normalized = [str(component) for component in components]
    for component in normalized:
        counts[component] = counts.get(component, 0) + 1
    raw = [1.0 / counts[component] for component in normalized]
    mean = sum(raw) / len(raw) if raw else 1.0
    return tuple(weight / mean for weight in raw)


def evaluate_head_promotion(
    *,
    task_kind: str,
    primary_improvement: float,
    ci_low: float,
    reference_primary: float,
    challenger_secondary: float,
    reference_secondary: float,
    learned_lambda: float,
    p95_ratio: float | None = None,
    seed_improvements: Sequence[float] = (),
) -> HeadPromotionDecision:
    """Apply the V2.2 gate independently to one endpoint head."""

    reasons: list[str] = []
    if not math.isfinite(primary_improvement):
        reasons.append("primary_nonfinite")
    if not math.isfinite(ci_low) or ci_low <= 0.0:
        reasons.append("ci")
    if seed_improvements and any(
        not math.isfinite(value) or value <= 0.0 for value in seed_improvements
    ):
        reasons.append("seed_direction")
    if not math.isfinite(reference_primary) or reference_primary <= 0.0:
        reasons.append("reference_primary")
    if not math.isfinite(challenger_secondary) or not math.isfinite(reference_secondary):
        reasons.append("secondary_nonfinite")
    if not math.isfinite(learned_lambda):
        reasons.append("lambda_nonfinite")
    if p95_ratio is not None and not math.isfinite(p95_ratio):
        reasons.append("p95_nonfinite")
    if task_kind == "regression":
        relative = (
            primary_improvement / reference_primary
            if math.isfinite(reference_primary) and reference_primary > 0.0
            else -math.inf
        )
        if not math.isfinite(relative) or relative < 0.05:
            reasons.append("mcid")
        if (
            math.isfinite(challenger_secondary)
            and math.isfinite(reference_secondary)
            and challenger_secondary > reference_secondary * 1.05
        ):
            reasons.append("rmse_noninferiority")
        if p95_ratio is not None and math.isfinite(p95_ratio) and p95_ratio > 1.10:
            reasons.append("p95_noninferiority")
    elif task_kind == "binary_classification":
        if primary_improvement < 0.02:
            reasons.append("mcid")
        if (
            math.isfinite(challenger_secondary)
            and math.isfinite(reference_secondary)
            and challenger_secondary > reference_secondary + 0.01
        ):
            reasons.append("brier_noninferiority")
    else:
        raise ValueError(f"unsupported task kind: {task_kind!r}")
    promoted = not reasons
    return HeadPromotionDecision(
        promoted=promoted,
        effective_lambda=float(learned_lambda) if promoted else 0.0,
        fallback="challenger" if promoted else "base_exact",
        failure_reasons=tuple(reasons),
    )
