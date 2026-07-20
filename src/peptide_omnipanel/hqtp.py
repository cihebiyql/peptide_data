"""Small hierarchical query-typed regression heads for sparse endpoint families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def _numeric_stack() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - training host contract.
        raise RuntimeError("NumPy is required for HQT-P models") from exc
    return np


def equal_task_component_weights(
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
) -> Any:
    """Give equal mass to tasks, components within task, and rows within component."""

    np = _numeric_stack()
    tasks = np.asarray([str(value) for value in task_ids], dtype=object)
    components = np.asarray([str(value) for value in component_ids], dtype=object)
    if len(tasks) == 0 or len(tasks) != len(components):
        raise ValueError("task/component weights require aligned non-empty inputs")
    unique_tasks = sorted(set(tasks))
    weights = np.zeros(len(tasks), dtype=float)
    for task_id in unique_tasks:
        task_mask = tasks == task_id
        task_components = sorted(set(components[task_mask]))
        if not task_components:
            raise ValueError(f"task has no components: {task_id}")
        component_mass = 1.0 / (len(unique_tasks) * len(task_components))
        for component_id in task_components:
            row_mask = task_mask & (components == component_id)
            weights[row_mask] = component_mass / int(np.sum(row_mask))
    if not np.all(np.isfinite(weights)) or not np.isclose(float(np.sum(weights)), 1.0):
        raise RuntimeError("HQT-P sample weights are invalid")
    return weights


def unique_representation_weights(representation_keys: Iterable[Any]) -> Any:
    """Give each unique molecular representation equal scaler influence."""

    np = _numeric_stack()
    keys = np.asarray([str(value) for value in representation_keys], dtype=object)
    if len(keys) == 0:
        raise ValueError("representation weights require non-empty keys")
    weights = np.zeros(len(keys), dtype=float)
    unique_keys = sorted(set(keys))
    for key in unique_keys:
        mask = keys == key
        weights[mask] = 1.0 / (len(unique_keys) * int(np.sum(mask)))
    if not np.isclose(float(np.sum(weights)), 1.0):
        raise RuntimeError("representation weights are invalid")
    return weights


def _weighted_mean_scale(matrix: Any, weights: Any) -> tuple[Any, Any]:
    np = _numeric_stack()
    values = np.asarray(matrix, dtype=float)
    mass = np.asarray(weights, dtype=float)
    mass = mass / float(np.sum(mass))
    mean = np.sum(values * mass[:, None], axis=0)
    variance = np.sum(((values - mean) ** 2) * mass[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    return mean, scale


@dataclass(frozen=True)
class WeightedRandomProjection:
    raw_mean: Any
    raw_scale: Any
    projection: Any
    projected_mean: Any
    projected_scale: Any

    @property
    def output_dim(self) -> int:
        return int(self.projection.shape[1])

    def transform(self, matrix: Any) -> Any:
        np = _numeric_stack()
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.raw_mean):
            raise ValueError("projection input feature dimension mismatch")
        projected = ((values - self.raw_mean) / self.raw_scale) @ self.projection
        transformed = (projected - self.projected_mean) / self.projected_scale
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("projection produced non-finite features")
        return transformed


def fit_weighted_random_projection(
    matrix: Any,
    sample_weight: Any,
    *,
    output_dim: int,
    seed: int,
) -> WeightedRandomProjection:
    """Fit fold-local scaling around a data-independent Rademacher projection."""

    np = _numeric_stack()
    values = np.asarray(matrix, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
        raise ValueError("projection requires a non-empty two-dimensional matrix")
    if len(weights) != len(values) or np.any(weights < 0) or not np.all(np.isfinite(values)):
        raise ValueError("projection received invalid matrix or weights")
    if output_dim < 1:
        raise ValueError("projection output_dim must be positive")
    raw_mean, raw_scale = _weighted_mean_scale(values, weights)
    standardized = (values - raw_mean) / raw_scale
    dimension = min(int(output_dim), values.shape[1])
    if dimension == values.shape[1]:
        projection = np.eye(values.shape[1], dtype=float)
    else:
        generator = np.random.default_rng(int(seed))
        projection = generator.choice(
            np.asarray([-1.0, 1.0]),
            size=(values.shape[1], dimension),
        ) / np.sqrt(dimension)
    projected = standardized @ projection
    projected_mean, projected_scale = _weighted_mean_scale(projected, weights)
    return WeightedRandomProjection(
        raw_mean=raw_mean,
        raw_scale=raw_scale,
        projection=projection,
        projected_mean=projected_mean,
        projected_scale=projected_scale,
    )


@dataclass(frozen=True)
class HQTPRegressor:
    mode: str
    tasks: tuple[str, ...]
    target_mean: dict[str, float]
    target_scale: dict[str, float]
    family_coefficients: Any
    task_coefficients: dict[str, Any]

    def predict(self, matrix: Any, task_ids: Iterable[Any]) -> Any:
        np = _numeric_stack()
        values = np.asarray(matrix, dtype=float)
        tasks = np.asarray([str(value) for value in task_ids], dtype=object)
        if values.ndim != 2 or len(values) != len(tasks):
            raise ValueError("HQT-P prediction inputs are not aligned")
        augmented = np.column_stack([np.ones(len(values)), values])
        output = np.empty(len(values), dtype=float)
        for index, task_id in enumerate(tasks):
            if task_id not in self.target_mean:
                raise ValueError(f"HQT-P received an unseen task: {task_id}")
            standardized = float(augmented[index] @ self.family_coefficients)
            task_coefficients = self.task_coefficients.get(task_id)
            if task_coefficients is not None:
                standardized += float(augmented[index] @ task_coefficients)
            output[index] = (
                standardized * self.target_scale[task_id] + self.target_mean[task_id]
            )
        if not np.all(np.isfinite(output)):
            raise RuntimeError("HQT-P produced non-finite predictions")
        return output


def _target_standardization(
    targets: Any,
    tasks: Any,
    weights: Any,
) -> tuple[dict[str, float], dict[str, float], Any]:
    np = _numeric_stack()
    target_mean: dict[str, float] = {}
    target_scale: dict[str, float] = {}
    standardized = np.empty(len(targets), dtype=float)
    for task_id in sorted(set(tasks)):
        mask = tasks == task_id
        task_weights = weights[mask]
        task_weights = task_weights / float(np.sum(task_weights))
        task_targets = targets[mask]
        mean = float(np.sum(task_targets * task_weights))
        variance = float(np.sum(((task_targets - mean) ** 2) * task_weights))
        scale = variance**0.5 if variance > 1e-12 else 1.0
        target_mean[str(task_id)] = mean
        target_scale[str(task_id)] = scale
        standardized[mask] = (task_targets - mean) / scale
    return target_mean, target_scale, standardized


def _ridge_solution(design: Any, targets: Any, weights: Any, penalty: Any) -> Any:
    np = _numeric_stack()
    mass = np.asarray(weights, dtype=float)
    mass = mass / float(np.sum(mass))
    weighted_design = design * np.sqrt(mass)[:, None]
    weighted_targets = targets * np.sqrt(mass)
    system = weighted_design.T @ weighted_design + np.diag(penalty)
    right_hand_side = weighted_design.T @ weighted_targets
    try:
        coefficients = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("HQT-P fit produced non-finite coefficients")
    return coefficients


def fit_hqtp_regressor(
    matrix: Any,
    targets: Any,
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
    *,
    mode: str,
    family_l2: float,
    task_l2: float,
) -> HQTPRegressor:
    """Fit independent, family-only, or hierarchical MAP linear task heads."""

    np = _numeric_stack()
    values = np.asarray(matrix, dtype=float)
    truth = np.asarray(targets, dtype=float)
    tasks = np.asarray([str(value) for value in task_ids], dtype=object)
    components = np.asarray([str(value) for value in component_ids], dtype=object)
    if mode not in {"independent", "family", "hierarchical"}:
        raise ValueError(f"unsupported HQT-P mode: {mode}")
    if (
        values.ndim != 2
        or len(values) == 0
        or len(values) != len(truth)
        or len(values) != len(tasks)
        or len(values) != len(components)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(truth))
    ):
        raise ValueError("HQT-P training inputs are invalid")
    if family_l2 < 0 or task_l2 < 0:
        raise ValueError("HQT-P penalties must be non-negative")
    task_names = tuple(sorted(set(tasks)))
    weights = equal_task_component_weights(tasks, components)
    target_mean, target_scale, standardized = _target_standardization(
        truth,
        tasks,
        weights,
    )
    augmented = np.column_stack([np.ones(len(values)), values])
    width = augmented.shape[1]
    family = np.zeros(width, dtype=float)
    deviations: dict[str, Any] = {}

    if mode == "family":
        penalty = np.full(width, float(family_l2), dtype=float)
        penalty[0] = 0.0
        family = _ridge_solution(augmented, standardized, weights, penalty)
    elif mode == "independent":
        for task_id in task_names:
            mask = tasks == task_id
            task_weights = weights[mask]
            task_weights = task_weights / float(np.sum(task_weights))
            penalty = np.full(width, float(task_l2), dtype=float)
            penalty[0] = 0.0
            deviations[task_id] = _ridge_solution(
                augmented[mask],
                standardized[mask],
                task_weights,
                penalty,
            )
    else:
        design = np.zeros((len(values), width * (1 + len(task_names))), dtype=float)
        design[:, :width] = augmented
        for task_index, task_id in enumerate(task_names):
            block = slice(width * (task_index + 1), width * (task_index + 2))
            design[tasks == task_id, block] = augmented[tasks == task_id]
        penalty = np.concatenate(
            [
                np.asarray([0.0, *([float(family_l2)] * (width - 1))]),
                *[
                    np.full(width, float(task_l2), dtype=float)
                    for _ in task_names
                ],
            ]
        )
        coefficients = _ridge_solution(design, standardized, weights, penalty)
        family = coefficients[:width]
        for task_index, task_id in enumerate(task_names):
            block = slice(width * (task_index + 1), width * (task_index + 2))
            deviations[task_id] = coefficients[block]

    return HQTPRegressor(
        mode=mode,
        tasks=task_names,
        target_mean=target_mean,
        target_scale=target_scale,
        family_coefficients=family,
        task_coefficients=deviations,
    )


def task_component_mae(
    truth: Any,
    prediction: Any,
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
) -> dict[str, float]:
    np = _numeric_stack()
    actual = np.asarray(truth, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    tasks = np.asarray([str(value) for value in task_ids], dtype=object)
    components = np.asarray([str(value) for value in component_ids], dtype=object)
    if not (
        len(actual) == len(predicted) == len(tasks) == len(components)
        and len(actual) > 0
    ):
        raise ValueError("task component MAE inputs are not aligned")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("task component MAE requires finite truth and predictions")
    output: dict[str, float] = {}
    for task_id in sorted(set(tasks)):
        task_mask = tasks == task_id
        component_errors = []
        for component_id in sorted(set(components[task_mask])):
            mask = task_mask & (components == component_id)
            with np.errstate(over="ignore", invalid="ignore"):
                error = float(np.mean(np.abs(actual[mask] - predicted[mask])))
            if not math.isfinite(error):
                raise ValueError("task component MAE overflowed to a non-finite value")
            component_errors.append(error)
        score = float(np.mean(component_errors))
        if not math.isfinite(score):
            raise ValueError("task component MAE is non-finite")
        output[str(task_id)] = score
    return output
