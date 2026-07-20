"""Nested component CV for the linear HQT-P permeability challenger."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .hqtp import (
    equal_task_component_weights,
    fit_hqtp_regressor,
    fit_weighted_random_projection,
    task_component_mae,
    unique_representation_weights,
)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - training host contract.
        raise RuntimeError("NumPy is required for HQT-P nested CV") from exc
    return np


@dataclass(frozen=True)
class HQTPConfig:
    name: str
    mode: str
    projection_dim: int
    family_l2: float
    task_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "projection_dim": self.projection_dim,
            "family_l2": self.family_l2,
            "task_l2": self.task_l2,
        }


def hqtp_candidate_registry() -> tuple[HQTPConfig, ...]:
    return (
        HQTPConfig("independent_p64_a0.1", "independent", 64, 1.0, 0.1),
        HQTPConfig("independent_p64_a1", "independent", 64, 1.0, 1.0),
        HQTPConfig("independent_p64_a10", "independent", 64, 1.0, 10.0),
        HQTPConfig("family_only_p64_a1", "family", 64, 1.0, 1.0),
        HQTPConfig("hierarchical_p64_d0.1", "hierarchical", 64, 1.0, 0.1),
        HQTPConfig("hierarchical_p64_d1", "hierarchical", 64, 1.0, 1.0),
        HQTPConfig("hierarchical_p64_d10", "hierarchical", 64, 1.0, 10.0),
        HQTPConfig("hierarchical_p128_d1", "hierarchical", 128, 1.0, 1.0),
    )


def make_component_fold_map(
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
    *,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Build deterministic group folds and require every task in every fold."""

    np = _numpy()
    tasks = np.asarray([str(value) for value in task_ids], dtype=object)
    components = np.asarray([str(value) for value in component_ids], dtype=object)
    unique_components = np.asarray(sorted(set(components)), dtype=object)
    effective = min(int(n_splits), len(unique_components))
    if effective < 2:
        raise ValueError("HQT-P CV requires at least two independent components")
    expected = set(range(effective))
    for attempt in range(256):
        shuffled = unique_components.copy()
        np.random.default_rng(int(seed) + attempt).shuffle(shuffled)
        mapping = {
            str(component): index % effective
            for index, component in enumerate(shuffled)
        }
        if all(
            {mapping[str(component)] for component in components[tasks == task]} == expected
            for task in sorted(set(tasks))
        ):
            return mapping
    raise RuntimeError("unable to build HQT-P folds covering every task")


def validate_fold_map(
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
    component_to_fold: dict[str, int],
) -> int:
    tasks = [str(value) for value in task_ids]
    components = [str(value) for value in component_ids]
    missing = sorted(set(components) - set(component_to_fold))
    if missing:
        raise RuntimeError(f"HQT-P fold map misses components: {missing[:5]}")
    folds = sorted(set(component_to_fold.values()))
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise RuntimeError("HQT-P fold IDs must be contiguous from zero")
    expected = set(folds)
    for task_id in sorted(set(tasks)):
        task_folds = {
            component_to_fold[component]
            for task, component in zip(tasks, components, strict=True)
            if task == task_id
        }
        if task_folds != expected:
            raise RuntimeError(f"HQT-P task does not cover every fold: {task_id}")
    return len(folds)


def _fit_predict(
    config: HQTPConfig,
    raw_train: Any,
    train_truth: Any,
    train_tasks: Any,
    train_components: Any,
    train_representation_keys: Any,
    raw_validation: Any,
    validation_tasks: Any,
    *,
    projection_seed: int,
) -> tuple[Any, dict[str, Any]]:
    projection = fit_weighted_random_projection(
        raw_train,
        unique_representation_weights(train_representation_keys),
        output_dim=config.projection_dim,
        seed=projection_seed,
    )
    transformed_train = projection.transform(raw_train)
    transformed_validation = projection.transform(raw_validation)
    model = fit_hqtp_regressor(
        transformed_train,
        train_truth,
        train_tasks,
        train_components,
        mode=config.mode,
        family_l2=config.family_l2,
        task_l2=config.task_l2,
    )
    prediction = model.predict(transformed_validation, validation_tasks)
    weight = equal_task_component_weights(train_tasks, train_components)
    audit = {
        "projection_dim": projection.output_dim,
        "projection_seed": projection_seed,
        "train_weight_sum": float(_numpy().sum(weight)),
        "train_tasks": len(set(train_tasks)),
        "train_components": len(set(train_components)),
        "unique_train_representations": len(set(train_representation_keys)),
    }
    return prediction, audit


def _inner_candidate_predictions(
    config: HQTPConfig,
    raw_matrix: Any,
    truth: Any,
    tasks: Any,
    components: Any,
    representation_keys: Any,
    inner_map: dict[str, int],
    *,
    projection_seed: int,
) -> Any:
    np = _numpy()
    predictions = np.full(len(truth), np.nan, dtype=float)
    for fold_id in sorted(set(inner_map.values())):
        train = np.asarray(
            [inner_map[str(component)] != fold_id for component in components],
            dtype=bool,
        )
        validation = ~train
        prediction, _ = _fit_predict(
            config,
            raw_matrix[train],
            truth[train],
            tasks[train],
            components[train],
            representation_keys[train],
            raw_matrix[validation],
            tasks[validation],
            projection_seed=projection_seed,
        )
        predictions[validation] = prediction
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("HQT-P inner OOF predictions are incomplete")
    return predictions


def _select_configuration(
    configs: tuple[HQTPConfig, ...],
    scores: dict[str, dict[str, float]],
    *,
    minimum_relative_improvement: float,
    per_task_noninferiority: float,
) -> tuple[HQTPConfig | None, dict[str, str], list[dict[str, Any]]]:
    task_ids = sorted(next(iter(scores.values())))
    independent = [config for config in configs if config.mode == "independent"]
    independent_by_task = {
        task_id: min(
            independent,
            key=lambda config: (scores[config.name][task_id], config.name),
        ).name
        for task_id in task_ids
    }
    reference = {
        task_id: scores[independent_by_task[task_id]][task_id]
        for task_id in task_ids
    }
    candidates: list[dict[str, Any]] = []
    for config in configs:
        if config.mode == "independent":
            continue
        ratios = {}
        for task_id in task_ids:
            candidate_score = float(scores[config.name][task_id])
            reference_score = float(reference[task_id])
            if not math.isfinite(candidate_score) or candidate_score < 0:
                ratio = math.inf
            elif reference_score <= 1e-12:
                ratio = 1.0 if candidate_score <= 1e-12 else math.inf
            else:
                ratio = max(candidate_score / reference_score, 1e-12)
            ratios[task_id] = ratio
        valid = all(value <= 1.0 + per_task_noninferiority for value in ratios.values())
        log_ratio = float(sum(math.log(value) for value in ratios.values()) / len(ratios))
        relative = 1.0 - math.exp(log_ratio)
        candidates.append(
            {
                "config": config.name,
                "valid": valid,
                "macro_log_ratio": log_ratio,
                "macro_relative_improvement": relative,
                "task_ratios": ratios,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["valid"]
        and row["macro_relative_improvement"] >= minimum_relative_improvement
    ]
    if not eligible:
        return None, independent_by_task, candidates
    selected_name = min(
        eligible,
        key=lambda row: (row["macro_log_ratio"], row["config"]),
    )["config"]
    return next(config for config in configs if config.name == selected_name), independent_by_task, candidates


def nested_hqtp_predictions(
    raw_matrix: Any,
    truth: Any,
    task_ids: Iterable[Any],
    component_ids: Iterable[Any],
    representation_keys: Iterable[Any],
    component_to_outer_fold: dict[str, int],
    *,
    configs: tuple[HQTPConfig, ...] | None = None,
    inner_splits: int = 3,
    fold_seed: int = 20260716,
    projection_seed: int = 20260716,
    minimum_relative_improvement: float = 0.01,
    per_task_noninferiority: float = 0.05,
) -> tuple[Any, Any, Any, list[dict[str, Any]]]:
    """Return nested independent-reference and HQT-P selected OOF predictions."""

    np = _numpy()
    matrix = np.asarray(raw_matrix, dtype=float)
    actual = np.asarray(truth, dtype=float)
    tasks = np.asarray([str(value) for value in task_ids], dtype=object)
    components = np.asarray([str(value) for value in component_ids], dtype=object)
    keys = np.asarray([str(value) for value in representation_keys], dtype=object)
    if not (len(matrix) == len(actual) == len(tasks) == len(components) == len(keys)):
        raise ValueError("HQT-P nested inputs are not aligned")
    fold_count = validate_fold_map(tasks, components, component_to_outer_fold)
    outer_folds = np.asarray(
        [component_to_outer_fold[str(component)] for component in components],
        dtype=int,
    )
    candidate_configs = configs or hqtp_candidate_registry()
    if len({config.name for config in candidate_configs}) != len(candidate_configs):
        raise ValueError("HQT-P candidate names must be unique")
    if not any(config.mode == "independent" for config in candidate_configs):
        raise ValueError("HQT-P nested selection requires an independent reference")
    if not any(config.mode != "independent" for config in candidate_configs):
        raise ValueError("HQT-P nested selection requires a family challenger")
    independent_oof = np.full(len(actual), np.nan, dtype=float)
    selected_oof = np.full(len(actual), np.nan, dtype=float)
    reports: list[dict[str, Any]] = []

    for outer_fold in range(fold_count):
        train = outer_folds != outer_fold
        test = ~train
        inner_map = make_component_fold_map(
            tasks[train],
            components[train],
            n_splits=inner_splits,
            seed=fold_seed + outer_fold * 100,
        )
        scores: dict[str, dict[str, float]] = {}
        inner_predictions: dict[str, Any] = {}
        for config in candidate_configs:
            predictions = _inner_candidate_predictions(
                config,
                matrix[train],
                actual[train],
                tasks[train],
                components[train],
                keys[train],
                inner_map,
                projection_seed=projection_seed + config.projection_dim,
            )
            inner_predictions[config.name] = predictions
            scores[config.name] = task_component_mae(
                actual[train],
                predictions,
                tasks[train],
                components[train],
            )
        selected, independent_by_task, selection_audit = _select_configuration(
            candidate_configs,
            scores,
            minimum_relative_improvement=minimum_relative_improvement,
            per_task_noninferiority=per_task_noninferiority,
        )

        reference_predictions = np.full(int(np.sum(test)), np.nan, dtype=float)
        test_tasks = tasks[test]
        for config_name in sorted(set(independent_by_task.values())):
            config = next(item for item in candidate_configs if item.name == config_name)
            prediction, _ = _fit_predict(
                config,
                matrix[train],
                actual[train],
                tasks[train],
                components[train],
                keys[train],
                matrix[test],
                test_tasks,
                projection_seed=projection_seed + config.projection_dim,
            )
            for task_id, selected_name in independent_by_task.items():
                if selected_name == config_name:
                    reference_predictions[test_tasks == task_id] = prediction[
                        test_tasks == task_id
                    ]
        if not np.all(np.isfinite(reference_predictions)):
            raise RuntimeError("HQT-P independent reference did not cover outer test")

        if selected is None:
            selected_predictions = reference_predictions.copy()
            selected_name = "independent_fallback"
            selected_audit = {"fallback": True}
        else:
            selected_predictions, selected_audit = _fit_predict(
                selected,
                matrix[train],
                actual[train],
                tasks[train],
                components[train],
                keys[train],
                matrix[test],
                test_tasks,
                projection_seed=projection_seed + selected.projection_dim,
            )
            selected_name = selected.name
            selected_audit = {"fallback": False, **selected_audit}
        independent_oof[test] = reference_predictions
        selected_oof[test] = selected_predictions
        reports.append(
            {
                "outer_fold_id": outer_fold,
                "train_rows": int(np.sum(train)),
                "test_rows": int(np.sum(test)),
                "train_components": len(set(components[train])),
                "test_components": len(set(components[test])),
                "independent_by_task": independent_by_task,
                "selected_config": selected_name,
                "selection_candidates": selection_audit,
                "selected_fit_audit": selected_audit,
                "inner_task_component_mae": scores,
            }
        )

    if not np.all(np.isfinite(independent_oof)) or not np.all(np.isfinite(selected_oof)):
        raise RuntimeError("HQT-P outer OOF predictions are incomplete")
    return independent_oof, selected_oof, outer_folds, reports
