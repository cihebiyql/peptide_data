"""Leakage-aware representation lanes and classical V2.1 model candidates."""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

from .contracts import InputValidationError, PeptideInput
from .datasets import Observation
from .features import (
    FEATURE_SCHEMA_VERSION,
    combine_feature_blocks,
    hashed_character_ngrams,
    rdkit_morgan_descriptor_features,
    sequence_physicochemical_features,
)
from .metrics import classification_metrics, regression_metrics
from .representations import is_sentinel


REPRESENTATION_PROFILES = frozenset({"sequence", "helm", "structure"})


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    framework: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "framework": self.framework,
            "parameters": dict(self.parameters),
        }


def representation_text(observation: Observation, profile: str) -> str | None:
    """Return only representations explicitly eligible for the requested lane."""

    if profile not in REPRESENTATION_PROFILES:
        raise ValueError(f"unsupported representation profile: {profile!r}")
    if profile == "sequence":
        if observation.sequence_model_eligible is not True or is_sentinel(observation.sequence):
            return None
        try:
            validated = PeptideInput(sequence=str(observation.sequence), format="plain").validate()
        except InputValidationError:
            return None
        return validated.sequence
    if profile == "helm":
        if is_sentinel(observation.helm):
            return None
        return str(observation.helm).strip()
    if observation.structure_model_eligible is not True or is_sentinel(observation.smiles):
        return None
    return str(observation.smiles).strip()


def feature_schema(profile: str) -> str:
    if profile == "sequence":
        return f"{FEATURE_SCHEMA_VERSION}:sequence_char256_physchem_v21"
    if profile == "helm":
        return f"{FEATURE_SCHEMA_VERSION}:helm_char512_v21"
    if profile == "structure":
        return f"{FEATURE_SCHEMA_VERSION}:smiles_char256_morgan512_descriptors_v21"
    raise ValueError(f"unsupported representation profile: {profile!r}")


def representation_feature_vector(text: str, profile: str) -> tuple[float, ...]:
    if profile == "sequence":
        return combine_feature_blocks(
            [
                hashed_character_ngrams(
                    text,
                    modality="sequence",
                    n_features=256,
                    ngram_range=(1, 3),
                ),
                sequence_physicochemical_features(text),
            ]
        ).values
    if profile == "helm":
        return combine_feature_blocks(
            [
                hashed_character_ngrams(
                    text,
                    modality="helm",
                    n_features=512,
                    ngram_range=(1, 4),
                )
            ]
        ).values
    if profile == "structure":
        morgan, descriptors = rdkit_morgan_descriptor_features(
            text,
            radius=2,
            n_bits=512,
            use_rdkit=True,
        )
        return combine_feature_blocks(
            [
                hashed_character_ngrams(
                    text,
                    modality="smiles",
                    n_features=256,
                    ngram_range=(1, 3),
                ),
                morgan,
                descriptors,
            ]
        ).values
    raise ValueError(f"unsupported representation profile: {profile!r}")


def available_frameworks() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("sklearn", "xgboost", "lightgbm", "catboost")
    }


def model_candidates(task_kind: str) -> list[ModelCandidate]:
    if task_kind not in {"regression", "binary_classification"}:
        raise ValueError(f"unsupported task kind: {task_kind}")
    candidates = [
        ModelCandidate("dummy", "sklearn", {}),
        ModelCandidate("linear_l2_light", "sklearn", {"regularization": 0.1}),
        ModelCandidate("linear_l2_medium", "sklearn", {"regularization": 10.0}),
        ModelCandidate(
            "analog_knn5_cosine",
            "sklearn",
            {"n_neighbors": 5, "metric": "cosine", "weights": "distance"},
        ),
        ModelCandidate(
            "analog_knn15_cosine",
            "sklearn",
            {"n_neighbors": 15, "metric": "cosine", "weights": "distance"},
        ),
        ModelCandidate(
            "extra_trees_leaf1_sqrt",
            "sklearn",
            {"n_estimators": 512, "min_samples_leaf": 1, "max_features": "sqrt"},
        ),
        ModelCandidate(
            "extra_trees_leaf2_half",
            "sklearn",
            {"n_estimators": 512, "min_samples_leaf": 2, "max_features": 0.5},
        ),
        ModelCandidate(
            "extra_trees_leaf5_sqrt",
            "sklearn",
            {"n_estimators": 512, "min_samples_leaf": 5, "max_features": "sqrt"},
        ),
        ModelCandidate(
            "extra_trees_leaf3_half_1024",
            "sklearn",
            {"n_estimators": 1024, "min_samples_leaf": 3, "max_features": 0.5},
        ),
        ModelCandidate(
            "random_forest_leaf2_half",
            "sklearn",
            {"n_estimators": 384, "min_samples_leaf": 2, "max_features": 0.5},
        ),
        ModelCandidate(
            "hist_gradient_leaf15",
            "sklearn",
            {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2": 3.0},
        ),
        ModelCandidate(
            "hist_gradient_leaf31",
            "sklearn",
            {"max_iter": 300, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2": 10.0},
        ),
    ]
    optional = available_frameworks()
    if optional["xgboost"]:
        candidates.extend(
            [
                ModelCandidate(
                    "xgboost_depth3",
                    "xgboost",
                    {
                        "n_estimators": 500,
                        "max_depth": 3,
                        "learning_rate": 0.04,
                        "subsample": 0.85,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 3,
                        "reg_lambda": 5.0,
                    },
                ),
                ModelCandidate(
                    "xgboost_depth6",
                    "xgboost",
                    {
                        "n_estimators": 500,
                        "max_depth": 6,
                        "learning_rate": 0.025,
                        "subsample": 0.8,
                        "colsample_bytree": 0.7,
                        "min_child_weight": 5,
                        "reg_lambda": 10.0,
                    },
                ),
            ]
        )
        if task_kind == "regression":
            candidates.extend(
                [
                    ModelCandidate(
                        "xgboost_depth4_mae",
                        "xgboost",
                        {
                            "n_estimators": 750,
                            "max_depth": 4,
                            "learning_rate": 0.03,
                            "subsample": 0.85,
                            "colsample_bytree": 0.8,
                            "min_child_weight": 4,
                            "reg_lambda": 8.0,
                            "objective": "reg:absoluteerror",
                        },
                    ),
                    ModelCandidate(
                        "xgboost_depth4_huber",
                        "xgboost",
                        {
                            "n_estimators": 750,
                            "max_depth": 4,
                            "learning_rate": 0.03,
                            "subsample": 0.85,
                            "colsample_bytree": 0.8,
                            "min_child_weight": 4,
                            "reg_lambda": 8.0,
                            "objective": "reg:pseudohubererror",
                        },
                    ),
                ]
            )
        else:
            candidates.extend(
                [
                    ModelCandidate(
                        "xgboost_depth3_auto_balanced",
                        "xgboost",
                        {
                            "n_estimators": 750,
                            "max_depth": 3,
                            "learning_rate": 0.03,
                            "subsample": 0.85,
                            "colsample_bytree": 0.8,
                            "min_child_weight": 3,
                            "reg_lambda": 8.0,
                            "auto_class_weight": True,
                        },
                    ),
                    ModelCandidate(
                        "xgboost_depth5_auto_balanced",
                        "xgboost",
                        {
                            "n_estimators": 750,
                            "max_depth": 5,
                            "learning_rate": 0.025,
                            "subsample": 0.8,
                            "colsample_bytree": 0.75,
                            "min_child_weight": 5,
                            "reg_lambda": 10.0,
                            "auto_class_weight": True,
                        },
                    ),
                ]
            )
    if optional["lightgbm"]:
        candidates.extend(
            [
                ModelCandidate(
                    "lightgbm_leaf15",
                    "lightgbm",
                    {
                        "n_estimators": 500,
                        "num_leaves": 15,
                        "learning_rate": 0.03,
                        "min_child_samples": 15,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.85,
                    },
                ),
                ModelCandidate(
                    "lightgbm_leaf31",
                    "lightgbm",
                    {
                        "n_estimators": 500,
                        "num_leaves": 31,
                        "learning_rate": 0.025,
                        "min_child_samples": 25,
                        "feature_fraction": 0.7,
                        "bagging_fraction": 0.8,
                    },
                ),
            ]
        )
        if task_kind == "binary_classification":
            candidates.extend(
                [
                    ModelCandidate(
                        "lightgbm_leaf7_child8",
                        "lightgbm",
                        {
                            "n_estimators": 750,
                            "num_leaves": 7,
                            "learning_rate": 0.025,
                            "min_child_samples": 8,
                            "feature_fraction": 0.9,
                            "bagging_fraction": 0.9,
                        },
                    ),
                    ModelCandidate(
                        "lightgbm_leaf31_child10",
                        "lightgbm",
                        {
                            "n_estimators": 750,
                            "num_leaves": 31,
                            "learning_rate": 0.02,
                            "min_child_samples": 10,
                            "feature_fraction": 0.8,
                            "bagging_fraction": 0.85,
                        },
                    ),
                ]
            )
    if optional["catboost"]:
        candidates.append(
            ModelCandidate(
                "catboost_depth6",
                "catboost",
                {
                    "iterations": 500,
                    "depth": 6,
                    "learning_rate": 0.04,
                    "l2_leaf_reg": 5.0,
                    "random_strength": 0.5,
                },
            )
        )
        if task_kind == "regression":
            candidates.extend(
                [
                    ModelCandidate(
                        "catboost_depth4_mae",
                        "catboost",
                        {
                            "iterations": 1000,
                            "depth": 4,
                            "learning_rate": 0.03,
                            "l2_leaf_reg": 8.0,
                            "random_strength": 0.25,
                            "loss_function": "MAE",
                        },
                    ),
                    ModelCandidate(
                        "catboost_depth8_mae",
                        "catboost",
                        {
                            "iterations": 750,
                            "depth": 8,
                            "learning_rate": 0.025,
                            "l2_leaf_reg": 10.0,
                            "random_strength": 0.5,
                            "loss_function": "MAE",
                        },
                    ),
                ]
            )
        else:
            candidates.append(
                ModelCandidate(
                    "catboost_depth4_balanced",
                    "catboost",
                    {
                        "iterations": 750,
                        "depth": 4,
                        "learning_rate": 0.03,
                        "l2_leaf_reg": 8.0,
                        "random_strength": 0.25,
                    },
                )
            )
    return candidates


def build_estimator(
    candidate: ModelCandidate,
    task_kind: str,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    if candidate.framework == "sklearn":
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if candidate.name == "dummy":
            return (
                DummyRegressor(strategy="median")
                if task_kind == "regression"
                else DummyClassifier(strategy="prior")
            )
        if candidate.name.startswith("linear_l2"):
            regularization = float(candidate.parameters["regularization"])
            estimator = (
                Ridge(alpha=regularization)
                if task_kind == "regression"
                else LogisticRegression(
                    C=1.0 / regularization,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                )
            )
            return make_pipeline(StandardScaler(), estimator)
        if candidate.name.startswith("analog_knn"):
            cls = (
                KNeighborsRegressor
                if task_kind == "regression"
                else KNeighborsClassifier
            )
            estimator = cls(n_jobs=n_jobs, **candidate.parameters)
            return make_pipeline(StandardScaler(), estimator)
        if candidate.name.startswith("extra_trees"):
            cls = ExtraTreesRegressor if task_kind == "regression" else ExtraTreesClassifier
            kwargs = dict(candidate.parameters)
            if task_kind == "binary_classification":
                kwargs["class_weight"] = "balanced"
            return cls(random_state=seed, n_jobs=n_jobs, **kwargs)
        if candidate.name.startswith("random_forest"):
            cls = RandomForestRegressor if task_kind == "regression" else RandomForestClassifier
            kwargs = dict(candidate.parameters)
            if task_kind == "binary_classification":
                kwargs["class_weight"] = "balanced_subsample"
            return cls(random_state=seed, n_jobs=n_jobs, **kwargs)
        if candidate.name.startswith("hist_gradient"):
            cls = (
                HistGradientBoostingRegressor
                if task_kind == "regression"
                else HistGradientBoostingClassifier
            )
            params = candidate.parameters
            return cls(
                max_iter=int(params["max_iter"]),
                learning_rate=float(params["learning_rate"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
                l2_regularization=float(params["l2"]),
                early_stopping=True,
                random_state=seed,
            )
    if candidate.framework == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if task_kind == "regression" else XGBClassifier
        kwargs = dict(candidate.parameters)
        kwargs.pop("auto_class_weight", None)
        kwargs.update(
            {
                "random_state": seed,
                "n_jobs": n_jobs,
                "tree_method": "hist",
                "verbosity": 0,
            }
        )
        if task_kind == "regression":
            kwargs.setdefault("objective", "reg:squarederror")
        else:
            kwargs["eval_metric"] = "logloss"
        return cls(**kwargs)
    if candidate.framework == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if task_kind == "regression" else LGBMClassifier
        kwargs = dict(candidate.parameters)
        kwargs.update(
            {
                "random_state": seed,
                "n_jobs": n_jobs,
                "verbosity": -1,
                "bagging_freq": 1,
                "reg_lambda": 5.0,
            }
        )
        if task_kind == "binary_classification":
            kwargs["class_weight"] = "balanced"
        return cls(**kwargs)
    if candidate.framework == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if task_kind == "regression" else CatBoostClassifier
        kwargs = dict(candidate.parameters)
        kwargs.update(
            {
                "random_seed": seed,
                "thread_count": n_jobs,
                "verbose": False,
                "allow_writing_files": False,
            }
        )
        kwargs.setdefault(
            "loss_function",
            "RMSE" if task_kind == "regression" else "Logloss",
        )
        if task_kind == "binary_classification":
            kwargs["auto_class_weights"] = "Balanced"
        return cls(**kwargs)
    raise ValueError(f"unsupported candidate: {candidate}")


def predict_estimator(estimator: Any, matrix: Any, task_kind: str) -> Any:
    if task_kind == "regression":
        return estimator.predict(matrix)
    probabilities = estimator.predict_proba(matrix)
    classes = list(estimator.classes_)
    if 1 not in classes:
        return probabilities[:, 0] * 0.0
    return probabilities[:, classes.index(1)]


def component_folds(
    matrix: Any,
    targets: Any,
    groups: Any,
    task_kind: str,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[Any, Any]]:
    try:
        import numpy as np
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover - training host contract.
        raise RuntimeError("NumPy and scikit-learn are required for grouped CV") from exc
    unique_groups = np.unique(groups)
    effective_splits = min(int(n_splits), len(unique_groups))
    if effective_splits < 2:
        raise ValueError("component-aware CV needs at least two independent components")
    if task_kind == "binary_classification":
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=seed,
        )
        folds = list(splitter.split(matrix, targets, groups))
    else:
        splitter = GroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(matrix, targets, groups))
    for train_indices, validation_indices in folds:
        if set(groups[train_indices]) & set(groups[validation_indices]):
            raise RuntimeError("component leakage detected inside development CV")
    return folds


def evaluate_candidate_cv(
    candidate: ModelCandidate,
    matrix: Any,
    targets: Any,
    groups: Any,
    task_kind: str,
    *,
    n_splits: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    import numpy as np

    predictions = np.full(len(targets), np.nan, dtype=float)
    fold_ids = np.full(len(targets), -1, dtype=int)
    started = time.perf_counter()
    folds = component_folds(
        matrix,
        targets,
        groups,
        task_kind,
        n_splits=n_splits,
        seed=seed,
    )
    for fold_index, (train_indices, validation_indices) in enumerate(folds):
        estimator = build_estimator(candidate, task_kind, seed=seed + fold_index, n_jobs=n_jobs)
        fit_kwargs: dict[str, Any] = {}
        if (
            task_kind == "binary_classification"
            and candidate.parameters.get("auto_class_weight") is True
        ):
            train_targets = targets[train_indices].astype(int)
            counts = np.bincount(train_targets, minlength=2)
            if np.all(counts > 0):
                class_weights = len(train_targets) / (2.0 * counts)
                fit_kwargs["sample_weight"] = class_weights[train_targets]
        estimator.fit(matrix[train_indices], targets[train_indices], **fit_kwargs)
        predictions[validation_indices] = predict_estimator(
            estimator,
            matrix[validation_indices],
            task_kind,
        )
        fold_ids[validation_indices] = fold_index
    if not np.all(np.isfinite(predictions)) or np.any(fold_ids < 0):
        raise RuntimeError("OOF prediction coverage is incomplete")
    metrics = (
        regression_metrics(targets, predictions)
        if task_kind == "regression"
        else classification_metrics(targets, predictions)
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "fold_ids": fold_ids,
        "fit_seconds": time.perf_counter() - started,
        "fold_count": len(folds),
    }
