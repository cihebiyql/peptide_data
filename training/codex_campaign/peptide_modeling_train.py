"""Grouped, checkpointed baselines on frozen and comparable feature populations.

The outer group assignment is created from the complete task before any feature
mask is applied. Holdout metrics belong to the saved outer-train estimator. An
optional all-data refit is a separate artifact and never replaces that estimator.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    ParameterSampler,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits

DEFAULTS = {
    "routes": ["sequence", "structure", "fusion"],
    "models": ["linear", "lightgbm"],
    "n_trials": 20,
    "inner_folds": 5,
    "test_size": 0.2,
    "seed": 42,
    "n_jobs": 4,
    "min_rows": 40,
    "min_groups": 10,
    "min_class_count": 4,
    "min_train_rows": 6,
    "scope_label": "common",
    "refit_full": False,
    "svm_max_rows": 2000,
    "svm_max_features": 8192,
    "svm_pca_components": 64,
    "histgb_max_rows": 50000,
    "histgb_max_features": 8192,
    "elasticnet_max_rows": 50000,
    "elasticnet_max_features": 20000,
    "search_backend": "random_per_pair",
    "search_budget": 120,
}

MODEL_FAMILIES = (
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
)


class DeferredTraining(ValueError):
    """The supplied task cannot support the declared validation protocol."""


def _validated_target_bounds(bounds):
    if bounds is None:
        return None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("target_bounds must contain two finite endpoints")
    lower, upper = (float(v) for v in bounds)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("target_bounds must be finite and strictly increasing")
    return lower, upper


class BoundedRegressor(RegressorMixin, BaseEstimator):
    """Persist a task-declared prediction range as part of the fitted estimator."""

    def __init__(self, estimator, bounds):
        self.estimator = estimator
        self.bounds = bounds

    def fit(self, X, y):
        self.bounds_ = _validated_target_bounds(self.bounds)
        if self.bounds_ is None:
            raise ValueError("BoundedRegressor requires explicitly declared bounds")
        target = np.asarray(y, dtype=float)
        lower, upper = self.bounds_
        if (
            not np.isfinite(target).all()
            or np.any(target < lower - 1e-12)
            or np.any(target > upper + 1e-12)
        ):
            raise ValueError("observed_targets_violate_declared_target_bounds")
        self.estimator_ = clone(self.estimator).fit(X, y)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        if not np.isfinite(prediction).all():
            raise ValueError("Underlying bounded estimator produced nonfinite predictions")
        return np.clip(prediction, *self.bounds_)


class AdaptivePCA(TransformerMixin, BaseEstimator):
    """Bound PCA inside each training fold, including folds smaller than 64 rows."""

    def __init__(self, n_components=64, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.n_components_ = min(int(self.n_components), X.shape[1], X.shape[0] - 1)
        if self.n_components_ < 1:
            raise ValueError("PCA needs at least two training samples")
        self.pca_ = PCA(
            n_components=self.n_components_, svd_solver="randomized", random_state=self.random_state
        ).fit(X)
        # Randomized PCA can leave a strided view; joblib restores it as C order.
        # Fix the layout before training/prediction so float32 BLAS follows the
        # same reduction path before and after estimator serialization.
        self.pca_.components_ = np.ascontiguousarray(self.pca_.components_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def model_capability(model, kind, n_rows, n_features, config=None):
    """Deterministic compute/dependency admission, independent of targets or scores."""
    settings = _settings(config)
    reason = ""
    if kind not in {"regression", "classification"}:
        reason = "unsupported_task_kind"
    elif model not in MODEL_FAMILIES:
        reason = "unsupported_model_family"
    elif n_rows < 2 or n_features < 1:
        reason = "empty_model_input"
    elif model in {"xgboost", "lightgbm"} and importlib.util.find_spec(model) is None:
        reason = "optional_dependency_not_installed:" + model
    elif model == "svm" and n_rows > settings["svm_max_rows"]:
        reason = "kernel_row_budget_exceeded"
    elif model == "svm" and n_features > settings["svm_max_features"]:
        reason = "kernel_feature_budget_exceeded"
    elif model == "histgb" and (
        n_rows > settings["histgb_max_rows"] or n_features > settings["histgb_max_features"]
    ):
        reason = "histogram_compute_budget_exceeded"
    elif model == "elasticnet" and (
        n_rows > settings["elasticnet_max_rows"] or n_features > settings["elasticnet_max_features"]
    ):
        reason = "elasticnet_compute_budget_exceeded"
    return {
        "eligible": not reason,
        "reason": reason,
        "model": model,
        "task_kind": kind,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
    }


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_table(path, fields, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_name(value):
    value = str(value)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:80] or "task"
    return name + "_" + _digest(value)[:8]


def _settings(config):
    settings = {**DEFAULTS, **(config or {})}
    if not 0 < float(settings["test_size"]) < 1:
        raise ValueError("test_size must be between zero and one")
    for key in ("n_trials", "inner_folds", "n_jobs", "min_rows", "min_groups", "min_train_rows"):
        settings[key] = int(settings[key])
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if settings["inner_folds"] < 2:
        raise ValueError("inner_folds must be at least two")
    settings["seed"] = int(settings["seed"])
    return settings


def _sequence_is_groupable(row):
    # Partial or unresolved projections can collapse different nonstandard
    # monomers to X. They cannot establish a shared biological sequence.
    if str(row.get("sequence_kind", "")).lower() in {
        "ambiguous_sequences",
        "multichain",
        "unparsed_notation",
        "unparsed_or_missing",
        "partial_parent_projection",
        "partial_projection",
        "missing",
    }:
        return False
    if "sequence_eligible" not in row:
        return True  # Backwards-compatible fixtures and callers with explicit valid sequences.
    return str(row["sequence_eligible"]).lower() in {"true", "1"}


def validate_rows(rows, kind):
    if kind not in {"regression", "classification"}:
        raise DeferredTraining("unsupported_task_kind")
    if not rows:
        raise DeferredTraining("empty_task")
    ids = [str(r.get("record_id", "")) for r in rows]
    if any(not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Task record IDs must be nonempty and unique")
    try:
        y = np.asarray([float(r["target"]) for r in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Task target is not numeric") from exc
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite targets are not dropped silently")
    groups = np.asarray([str(r.get("split_group", "")) for r in rows])
    if any(not g for g in groups):
        raise ValueError("Each task record needs a nonempty split_group")
    for field in ("canonical_identity_group", "measurement_group", "sequence"):
        seen = {}
        for r, group in zip(rows, groups, strict=True):
            if field == "sequence" and not _sequence_is_groupable(r):
                continue
            key = str(r.get(field, ""))
            if key:
                previous = seen.setdefault(key, group)
                if previous != group:
                    raise ValueError(f"Shared {field} crosses split groups")
    values = np.unique(y)
    if kind == "classification":
        if len(values) != 2 or set(values) != {0.0, 1.0}:
            raise DeferredTraining("classification_requires_both_binary_classes")
    elif len(values) < 2:
        raise DeferredTraining("regression_target_has_no_variation")
    return y, groups


def _class_counts(y):
    return {str(int(k)): int(v) for k, v in Counter(y).items()}


def freeze_outer_split(rows, kind, path, config):
    settings = _settings(config)
    y, groups = validate_rows(rows, kind)
    split_contract = {
        "rows": [(r["record_id"], r["target"], r["split_group"]) for r in rows],
        "seed": settings["seed"],
        "test_size": settings["test_size"],
    }
    # Preserve the legacy default split contract. Explicit tiny-sample protocols
    # receive their own signature and cannot silently reuse a stricter split.
    if settings["min_train_rows"] != DEFAULTS["min_train_rows"]:
        split_contract["min_train_rows"] = settings["min_train_rows"]
    signature = _digest(split_contract)
    path = Path(path)
    if path.exists():
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("input_hash") != signature:
            raise ValueError("Frozen outer split does not match task rows or split configuration")
        return receipt
    if len(np.unique(groups)) < settings["min_groups"]:
        raise DeferredTraining("insufficient_full_task_groups")
    for offset in range(64):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=settings["test_size"], random_state=settings["seed"] + offset
        )
        train, test = next(splitter.split(np.zeros((len(rows), 1)), y, groups))
        if (
            len(train) < settings["min_train_rows"]
            or len(test) < 4
            or len(np.unique(groups[train])) < 3
        ):
            continue
        if kind == "classification":
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                continue
            if min(Counter(y[train]).values()) < int(settings["min_class_count"]):
                continue
            if min(Counter(y[test]).values()) < 2:
                continue
        receipt = {
            "input_hash": signature,
            "task_kind": kind,
            "protocol": "full_task_group_assignment_before_feature_mask",
            "seed": settings["seed"] + offset,
            "train_groups": sorted(set(groups[train].tolist())),
            "test_groups": sorted(set(groups[test].tolist())),
            "record_groups": {r["record_id"]: r["split_group"] for r in rows},
            "train_records": [rows[i]["record_id"] for i in train],
            "test_records": [rows[i]["record_id"] for i in test],
        }
        _write_json(path, receipt)
        return receipt
    raise DeferredTraining("no_feasible_frozen_group_holdout")


def _load_feature_route(task_rows, cache, route):
    from peptide_modeling_features import load_feature_route

    return load_feature_route(task_rows, cache, route)


def prepare_routes(rows, cache, config):
    settings = _settings(config)
    loaded, coverage, excluded = {}, {}, {}
    expected_ids = [r["record_id"] for r in rows]
    for route in settings["routes"]:
        try:
            payload = _load_feature_route(rows, cache, route)
            X = np.asarray(payload["X"], dtype=np.float32)
            valid = np.asarray(payload["valid"], dtype=bool)
            if payload["record_ids"] != expected_ids:
                raise ValueError("Feature record order does not match task rows")
            if X.ndim != 2 or X.shape[0] != len(rows) or valid.shape != (len(rows),):
                raise ValueError("Feature matrix or validity mask has the wrong shape")
            if X.shape[1] != len(payload["feature_names"]):
                raise ValueError("Feature schema dimension does not match matrix")
            valid = valid & ~np.isinf(X).any(axis=1) & ~np.isnan(X).all(axis=1)
            coverage[route] = {
                "valid_rows": int(valid.sum()),
                "total_rows": len(rows),
                "feature_count": X.shape[1],
                "error_count": len(payload.get("errors", [])),
            }
            if X.shape[1] == 0 or valid.sum() < settings["min_rows"]:
                excluded[route] = "insufficient_route_coverage"
                continue
            loaded[route] = {**payload, "X": X, "valid": valid}
        except KeyError as exc:
            excluded[route] = f"feature_route_unavailable:{type(exc).__name__}:{exc}"
    common = np.ones(len(rows), dtype=bool)
    if not loaded:
        return loaded, np.zeros(len(rows), dtype=bool), coverage, excluded
    for payload in loaded.values():
        common &= payload["valid"]
    return loaded, common, coverage, excluded


def _inner_splits(y, groups, kind, config):
    folds = int(config["inner_folds"])
    if folds < 2:
        raise ValueError("inner_folds must be at least two")
    if len(np.unique(groups)) < folds:
        raise DeferredTraining(f"requested_{folds}_fold_insufficient_independent_groups")
    if kind == "classification":
        if any(len(set(groups[y == label])) < folds for label in (0, 1)):
            raise DeferredTraining(f"requested_{folds}_fold_insufficient_groups_per_class")
    for offset in range(16):
        splitter = (
            StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=config["seed"] + offset)
            if kind == "classification"
            else GroupKFold(n_splits=folds)
        )
        splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
        if all(
            not (set(groups[a]) & set(groups[b]))
            and (kind != "classification" or (len(np.unique(y[a])) == len(np.unique(y[b])) == 2))
            for a, b in splits
        ):
            return splits
        if kind != "classification":
            break
    raise DeferredTraining(f"requested_{folds}_fold_class_group_constraints_not_satisfied")


def _parameter_candidates(model, config):
    if model == "linear":
        space = {"regularization": np.logspace(-3, 3, 31).tolist()}
    elif model == "lightgbm":
        space = {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.02, 0.05, 0.1],
            "num_leaves": [7, 15, 31],
            "min_child_samples": [5, 10, 20, 40],
            "reg_lambda": [0.0, 0.1, 1.0, 10.0],
            "colsample_bytree": [0.7, 1.0],
        }
    elif model == "elasticnet":
        space = {"regularization": np.logspace(-3, 1, 20).tolist(), "l1_ratio": [0.1, 0.5, 0.9]}
    elif model in {"random_forest", "extra_trees"}:
        space = {
            "n_estimators": [100, 200],
            "max_depth": [6, 12],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.1],
        }
    elif model == "histgb":
        space = {
            "max_iter": [100, 200],
            "max_leaf_nodes": [7, 15, 31],
            "learning_rate": [0.03, 0.1],
            "l2_regularization": [0.0, 1.0, 10.0],
        }
    elif model == "xgboost":
        space = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "reg_lambda": [1.0, 10.0],
        }
    elif model == "svm":
        space = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
        if config.get("task_kind") != "classification":
            space["epsilon"] = [0.01, 0.1, 0.3]
    else:
        raise ValueError(f"Unsupported model family: {model}")
    return [
        _plain(p)
        for p in ParameterSampler(space, n_iter=config["n_trials"], random_state=config["seed"])
    ]


def _linear_policy(kind, feature_count):
    high_dimensional = kind == "classification" and int(feature_count) >= 256
    return {
        "version": "dimension_based_linear_policy_v2",
        "selection_basis": "raw_route_feature_count_only",
        "feature_count": int(feature_count),
        "dense_classification_threshold": 256,
        "solver": "lbfgs"
        if high_dimensional
        else "liblinear"
        if kind == "classification"
        else "lsqr",
        "scaler_with_mean": high_dimensional,
        "max_iter": 500 if high_dimensional else 1500 if kind == "classification" else None,
        "tol": 1e-3 if high_dimensional else 1e-4,
        "convergence_warning": "candidate_failure",
    }


def _base_estimator(model, kind, params, config, feature_count=0):
    if model == "dummy":
        return (
            DummyClassifier(strategy="prior")
            if kind == "classification"
            else DummyRegressor(strategy="median")
        )
    if model == "linear":
        policy = _linear_policy(kind, feature_count)
        learner = (
            LogisticRegression(
                C=params["regularization"],
                solver=policy["solver"],
                max_iter=policy["max_iter"],
                tol=policy["tol"],
                random_state=config["seed"],
            )
            if kind == "classification"
            else Ridge(alpha=params["regularization"], solver="lsqr")
        )
        return Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler(with_mean=policy["scaler_with_mean"])),
                ("model", learner),
            ]
        )
    if model == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if kind == "classification" else lgb.LGBMRegressor
        return cls(
            **params,
            n_jobs=config["n_jobs"],
            random_state=config["seed"],
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            zero_as_missing=False,
        )
    if model == "elasticnet":
        learner = (
            LogisticRegression(
                C=params["regularization"],
                solver="saga",
                penalty="elasticnet",
                l1_ratio=params["l1_ratio"],
                max_iter=1500,
                tol=1e-3,
                random_state=config["seed"],
            )
            if kind == "classification"
            else ElasticNet(
                alpha=params["regularization"],
                l1_ratio=params["l1_ratio"],
                max_iter=3000,
                tol=1e-3,
                selection="cyclic",
            )
        )
        return Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler(with_mean=True)),
                ("model", learner),
            ]
        )
    if model in {"random_forest", "extra_trees"}:
        depth = params.get("max_depth")
        if depth is None or not 1 <= int(depth) <= 16:
            raise ValueError("forest_requires_bounded_max_depth_at_most_16")
        if int(params.get("n_estimators", 0)) > 300:
            raise ValueError("forest_estimator_budget_exceeded")
        fraction = params.get("max_features")
        if feature_count > 256 and isinstance(fraction, float) and fraction > 0.3:
            raise ValueError("wide_forest_feature_fraction_budget_exceeded")
        cls = (
            {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}
            if kind == "classification"
            else {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor}
        )[model]
        return Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("model", cls(**params, random_state=config["seed"], n_jobs=config["n_jobs"])),
            ]
        )
    if model == "histgb":
        cls = (
            HistGradientBoostingClassifier
            if kind == "classification"
            else HistGradientBoostingRegressor
        )
        return cls(**params, max_bins=63, early_stopping=False, random_state=config["seed"])
    if model == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if kind == "classification" else xgb.XGBRegressor
        return cls(
            **params,
            tree_method="hist",
            device="cpu",
            n_jobs=config["n_jobs"],
            random_state=config["seed"],
            verbosity=0,
            objective="binary:logistic" if kind == "classification" else "reg:squarederror",
            eval_metric="logloss" if kind == "classification" else "rmse",
        )
    if model == "svm":
        common = {k: params[k] for k in ("C", "gamma")}
        learner = (
            SVC(
                **common,
                probability=True,
                cache_size=256,
                max_iter=20000,
                random_state=config["seed"],
            )
            if kind == "classification"
            else SVR(**common, epsilon=params.get("epsilon", 0.1), cache_size=256, max_iter=20000)
        )
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler(with_mean=True)),
                (
                    "pca",
                    AdaptivePCA(
                        n_components=int(config.get("svm_pca_components", 64)),
                        random_state=config["seed"],
                    ),
                ),
                ("model", learner),
            ]
        )
    raise ValueError(f"Unsupported model family: {model}")


def _estimator(model, kind, params, config, feature_count=0):
    estimator = _base_estimator(model, kind, params, config, feature_count)
    bounds = _validated_target_bounds(config.get("target_bounds")) if kind == "regression" else None
    return BoundedRegressor(estimator, bounds) if bounds is not None else estimator


def _predict(model, X, kind):
    if kind == "classification":
        classes = list(model.classes_)
        return np.asarray(model.predict_proba(X)[:, classes.index(1)], dtype=np.float64)
    return np.asarray(model.predict(X), dtype=np.float64)


def _metrics(y, prediction, kind):
    if not np.isfinite(prediction).all():
        raise ValueError("Estimator produced nonfinite predictions")
    if kind == "classification":
        return {
            "pr_auc": float(average_precision_score(y, prediction)),
            "roc_auc": float(roc_auc_score(y, prediction)),
            "mcc": float(matthews_corrcoef(y, prediction >= 0.5)),
            "brier": float(brier_score_loss(y, prediction)),
            "threshold": 0.5,
            "calibration": "uncalibrated_model_probabilities",
        }
    rho = (
        spearmanr(y, prediction).statistic
        if len(np.unique(y)) > 1 and len(np.unique(prediction)) > 1
        else None
    )
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "spearman": float(rho) if rho is not None and math.isfinite(rho) else None,
    }


def _score(metrics, kind):
    return metrics["pr_auc"] if kind == "classification" else -metrics["mae"]


def _fit_model(estimator, X, y, n_jobs):
    with threadpool_limits(limits=n_jobs), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return clone(estimator).fit(X, y)


def _cv_predictions(estimator, X, y, splits, kind, n_jobs=1):
    prediction = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train, valid) in enumerate(splits):
        fitted = _fit_model(estimator, X[train], y[train], n_jobs)
        with threadpool_limits(limits=n_jobs):
            prediction[valid] = _predict(fitted, X[valid], kind)
        fold_ids[valid] = fold
    if not np.isfinite(prediction).all() or np.any(fold_ids < 0):
        raise ValueError("Inner OOF predictions do not cover every development sample")
    return prediction, fold_ids


def _save_checked_model(fitted, path, probe, kind, n_jobs):
    """Verify persistence using training features, without consulting a holdout."""
    with threadpool_limits(limits=n_jobs):
        before = _predict(fitted, probe, kind)
        joblib.dump(fitted, path)
        after = _predict(joblib.load(path), probe, kind)
    if not np.isfinite(before).all() or not np.allclose(after, before, rtol=1e-7, atol=1e-8):
        raise RuntimeError("Reloaded model predictions do not match the evaluated estimator")


def _fit_cv_ranked_candidate(candidates, loaded, train, y_train, splits, kind, settings, directory):
    """Admit the highest CV-ranked candidate that can fit the outer training set.

    Only development targets and development feature rows enter this gate. A
    final full-data refit is deliberately excluded: its convergence cannot be
    allowed to alter model selection after held-out labels have been observed.
    """
    validation = {
        "policy": "descending_inner_cv_score_then_outer_train_fit_and_persistence",
        "validation_labels": "outer_train_only",
        "holdout_used": False,
        "rejected_candidates": [],
    }
    seen = set()
    folder = directory / "candidate_validation"
    folder.mkdir(exist_ok=True)
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        key = _digest([candidate["route"], candidate["model"], candidate["params"]])
        if key in seen:
            continue
        seen.add(key)
        route, name = candidate["route"], candidate["model"]
        X_train = loaded[route]["X"][train]
        stage = "estimator_construction"
        try:
            estimator = _estimator(name, kind, candidate["params"], settings, X_train.shape[1])
            stage = "outer_train_fit"
            fitted = _fit_model(estimator, X_train, y_train, settings["n_jobs"])
            stage = "training_probe_persistence"
            candidate_path = folder / (key + ".joblib")
            _save_checked_model(fitted, candidate_path, X_train, kind, settings["n_jobs"])
            stage = "development_oof_reconstruction"
            oof, folds = _cv_predictions(
                estimator, X_train, y_train, splits, kind, settings["n_jobs"]
            )
        except Exception as exc:
            validation["rejected_candidates"].append(
                {
                    "route": route,
                    "model": name,
                    "params": candidate["params"],
                    "inner_cv_score": candidate["score"],
                    "stage": stage,
                    "reason": f"{type(exc).__name__}:{str(exc)[:1600]}",
                }
            )
            _write_json(directory / "candidate_fit_admission.json", validation)
            continue
        model_path = directory / "model.joblib"
        candidate_path.replace(model_path)
        validation.update(selected_cv_rank=len(seen), selected_candidate_key=key)
        _write_json(directory / "candidate_fit_admission.json", validation)
        return candidate, estimator, fitted, oof, folds, model_path, validation
    _write_json(directory / "candidate_fit_admission.json", validation)
    raise RuntimeError("No CV candidate passed outer-training fit and persistence admission")


def _solver_repair(estimator):
    """Increase only solver iteration limits, never statistical hyperparameters."""
    changes = {}
    parameters = estimator.get_params(deep=True)
    components = [("", estimator), *parameters.items()]
    for prefix, component in components:
        if isinstance(component, (ElasticNet, LogisticRegression, SVC, SVR)):
            current = int(component.max_iter)
            if 0 < current < 80000:
                changes[(prefix + "__" if prefix else "") + "max_iter"] = min(current * 4, 80000)
    if not changes:
        return None, {}
    repaired = clone(estimator).set_params(**changes)
    return repaired, changes


def _refit_selected(estimator, X, y, path, kind, settings):
    """Attempt optional deployment refit without revisiting the chosen model."""
    attempts = []
    candidate, limits = estimator, {}
    for number in range(2):
        try:
            fitted = _fit_model(candidate, X, y, settings["n_jobs"])
            _save_checked_model(fitted, path, X[:32], kind, settings["n_jobs"])
            attempts.append({"status": "completed", "solver_iteration_overrides": limits})
            return {"status": "completed", "attempts": attempts}
        except Exception as exc:
            attempts.append(
                {
                    "status": "failed",
                    "solver_iteration_overrides": limits,
                    "reason": f"{type(exc).__name__}:{str(exc)[:1600]}",
                }
            )
            if number == 0 and isinstance(exc, ConvergenceWarning):
                candidate, limits = _solver_repair(estimator)
                if candidate is not None:
                    continue
            break
    return {"status": "failed", "attempts": attempts, "reason": attempts[-1]["reason"]}


def _source_receipt(rows):
    sources = {str(r.get("source_group", "")) for r in rows}
    trusted = {s for s in sources if s.startswith(("DOI:", "PMID:", "PAPER:"))}
    return {
        "reported_source_groups": len(sources - {"", "not_reported"}),
        "recognizable_paper_groups": len(trusted),
        "status": "not_evaluated",
        "reason": "source_holdout_requires_its_own_training_and_nested_selection_protocol",
    }


def train_task(task, task_rows, features_cache, run_dir, config=None):
    """Train and validate a task; return completed, deferred, or failed with artifacts."""
    settings = _settings(config)
    rows = list(task_rows)
    task_id = str(task.get("task_id", "unnamed_task"))
    scope = _safe_name(settings["scope_label"])
    root = Path(run_dir).resolve()
    directory = root / "training" / _safe_name(task_id) / scope
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    kind = task.get("task_kind", "")
    result = {
        "task_id": task_id,
        "endpoint": task.get("endpoint", ""),
        "task_kind": kind,
        "scope_label": settings["scope_label"],
        "status": "failed",
        "reason": "",
        "selected_route": None,
        "selected_model": None,
        "inner_cv": {},
        "holdout": {},
        "artifacts": {},
        "n_task_rows": len(rows),
        "requested_inner_folds": settings["inner_folds"],
        "actual_inner_folds": 0,
    }
    try:
        declared_bounds = task.get(
            "target_bounds", task.get("task_definition", {}).get("target_bounds")
        )
        bounds = _validated_target_bounds(declared_bounds) if kind == "regression" else None
        transform = task.get(
            "target_transform", rows[0].get("target_transform", "identity") if rows else "identity"
        )
        if bounds is not None and transform != "identity":
            raise ValueError("declared_target_bounds_require_explicit_identity_target_space")
        settings["target_bounds"] = list(bounds) if bounds is not None else None
        result["target_bounds"] = settings["target_bounds"]
        y, groups = validate_rows(rows, kind)
        if bounds is not None and (np.any(y < bounds[0] - 1e-12) or np.any(y > bounds[1] + 1e-12)):
            raise ValueError("observed_targets_violate_declared_target_bounds")
        if len(rows) < settings["min_rows"]:
            raise DeferredTraining("insufficient_full_task_rows")
        split_path = Path(
            settings.get("outer_split_path") or root / "splits" / (_safe_name(task_id) + ".json")
        )
        split = freeze_outer_split(rows, kind, split_path, settings)
        result["outer_split_path"] = str(split_path)
        loaded, common, coverage, excluded = prepare_routes(rows, features_cache, settings)
        result.update(coverage=coverage, excluded_routes=excluded, n_common_rows=int(common.sum()))
        if int(common.sum()) < settings["min_rows"]:
            raise DeferredTraining("insufficient_common_feature_rows")
        train_groups, test_groups = set(split["train_groups"]), set(split["test_groups"])
        train = np.flatnonzero(common & np.isin(groups, list(train_groups)))
        test = np.flatnonzero(common & np.isin(groups, list(test_groups)))
        if len(train) < settings["min_train_rows"] or len(test) < 4:
            raise DeferredTraining("feature_mask_leaves_insufficient_frozen_holdout")
        if kind == "classification" and (
            len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2
        ):
            raise DeferredTraining("feature_mask_removes_class_from_frozen_split")
        splits = _inner_splits(y[train], groups[train], kind, settings)
        result["actual_inner_folds"] = len(splits)
        versions = {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        }
        for optional in ("lightgbm", "xgboost"):
            if optional in settings["models"]:
                try:
                    versions[optional] = importlib.metadata.version(optional)
                except importlib.metadata.PackageNotFoundError:
                    versions[optional] = "not_installed"
        search_implementation = {"backend": settings["search_backend"]}
        if settings["search_backend"] == "optuna":
            search_module = importlib.import_module("peptide_modeling_search")
            source_file = getattr(search_module, "__file__", None)
            search_implementation["source_sha256"] = (
                _file_hash(source_file) if source_file else None
            )
            try:
                versions["optuna"] = importlib.metadata.version("optuna")
            except importlib.metadata.PackageNotFoundError:
                versions["optuna"] = "not_installed"
        feature_hashes = {
            route: {
                "names": payload["feature_names"],
                "matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(payload["X"]).tobytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(payload["valid"].tobytes()).hexdigest(),
            }
            for route, payload in loaded.items()
        }
        linear_policies = {
            route: _linear_policy(kind, payload["X"].shape[1]) for route, payload in loaded.items()
        }
        signature = _digest(
            {
                "implementation_sha256": _file_hash(__file__),
                "search_implementation": search_implementation,
                "task": task,
                "rows": [(r["record_id"], r["target"], r["split_group"]) for r in rows],
                "config": settings,
                "features": feature_hashes,
                "linear_solver_policies": linear_policies,
                "versions": versions,
                "outer_split": split["input_hash"],
            }
        )
        result["input_hash"] = signature
        contract_path = directory / "input_contract.json"
        if contract_path.exists():
            prior = json.loads(contract_path.read_text(encoding="utf-8"))
            if prior["input_hash"] != signature:
                raise ValueError(
                    "Existing task/scope output has different inputs; use a new run or scope"
                )
        _write_json(
            contract_path,
            {
                "input_hash": signature,
                "implementation_sha256": _file_hash(__file__),
                "search_implementation": search_implementation,
                "task": task,
                "config": settings,
                "versions": versions,
                "feature_hashes": feature_hashes,
                "linear_solver_policies": linear_policies,
            },
        )
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("status") == "completed" and prior.get("input_hash") == signature:
                if all(
                    Path(path).is_file() and _file_hash(path) == receipt["sha256"]
                    for path, receipt in prior.get("artifact_hashes", {}).items()
                ):
                    return {**prior, "resumed": True}
        candidates = []
        trials_dir = directory / "trials"
        trials_dir.mkdir(exist_ok=True)
        pairs, capability = [], {}
        for route, payload in loaded.items():
            capability[route] = {}
            for model in settings["models"]:
                admission = model_capability(
                    model, kind, int(common.sum()), payload["X"].shape[1], settings
                )
                capability[route][model] = admission
                if admission["eligible"]:
                    pairs.append((route, model))
        result["model_capabilities"] = capability
        if not pairs:
            raise DeferredTraining("no_model_family_satisfies_compute_dependency_guards")

        def evaluate(route, model, params):
            """One complete grouped-CV candidate; holdout labels are never provided."""
            key = _digest([signature, route, model, params])
            checkpoint = trials_dir / (key + ".json")
            if checkpoint.exists():
                return json.loads(checkpoint.read_text(encoding="utf-8"))
            payload = loaded[route]
            trial = {
                "route": route,
                "model": model,
                "params": params,
                "input_hash": signature,
                "requested_folds": settings["inner_folds"],
                "actual_folds": len(splits),
            }
            try:
                estimator = _estimator(
                    model, kind, params, settings, feature_count=payload["X"].shape[1]
                )
                if model == "linear":
                    trial["linear_solver_policy"] = linear_policies[route]
                pred, _ = _cv_predictions(
                    estimator, payload["X"][train], y[train], splits, kind, settings["n_jobs"]
                )
                metrics = _metrics(y[train], pred, kind)
                if kind == "classification" and model == "svm":
                    metrics["calibration"] = "internal_training_platt_not_group_validated"
                trial.update(status="completed", metrics=metrics, score=_score(metrics, kind))
            except Exception as exc:
                trial.update(status="failed", reason=f"{type(exc).__name__}:{str(exc)[:1000]}")
            _write_json(checkpoint, trial)
            return trial

        if settings["search_backend"] == "optuna":
            from peptide_modeling_search import search_candidates

            search = search_candidates(
                pairs, evaluate, settings, directory / "search", signature, kind
            )
            candidates = search["trials"]
            result["search"] = {k: v for k, v in search.items() if k != "trials"}
        elif settings["search_backend"] == "random_per_pair":
            for route, model in pairs:
                for params in _parameter_candidates(model, {**settings, "task_kind": kind}):
                    candidates.append(evaluate(route, model, params))
            result["search"] = {
                "backend": "random_per_pair",
                "budget_per_model_route": settings["n_trials"],
            }
        else:
            raise ValueError("Unknown search backend: " + settings["search_backend"])
        valid_trials = [t for t in candidates if t["status"] == "completed"]
        result["search"].update(
            {
                "completed": len(valid_trials),
                "failed": len(candidates) - len(valid_trials),
                "eligible_model_route_pairs": len(pairs),
            }
        )
        if not valid_trials:
            raise RuntimeError(
                "All model candidates failed; no mean-prediction fallback was substituted"
            )
        winner, estimator, fitted, oof, folds, model_path, fit_admission = _fit_cv_ranked_candidate(
            valid_trials, loaded, train, y[train], splits, kind, settings, directory
        )
        result["search"]["candidate_fit_admission"] = fit_admission
        route, model_name = winner["route"], winner["model"]
        X = loaded[route]["X"]
        dummy = _estimator("dummy", kind, {}, settings)
        baseline_oof, _ = _cv_predictions(
            dummy, X[train], y[train], splits, kind, settings["n_jobs"]
        )
        # Model selection and persistence admission have finished. The outer
        # holdout is evaluated exactly once and cannot change the chosen model.
        with threadpool_limits(limits=settings["n_jobs"]):
            holdout_prediction = _predict(fitted, X[test], kind)
        dummy_fitted = _fit_model(dummy, X[train], y[train], settings["n_jobs"])
        baseline_holdout = _predict(dummy_fitted, X[test], kind)
        inner_metrics, holdout_metrics = (
            _metrics(y[train], oof, kind),
            _metrics(y[test], holdout_prediction, kind),
        )
        if kind == "classification" and model_name == "svm":
            for metrics in (inner_metrics, holdout_metrics):
                metrics["calibration"] = "internal_training_platt_not_group_validated"
        oof_path, holdout_path = directory / "oof.tsv", directory / "holdout.tsv"
        prediction_fields = [
            "record_id",
            "split_group",
            "fold",
            "target",
            "prediction",
            "dummy_prediction",
        ]
        for path, indices, pred, baseline, fold_values in (
            (oof_path, train, oof, baseline_oof, folds),
            (holdout_path, test, holdout_prediction, baseline_holdout, np.full(len(test), -1)),
        ):
            _write_table(
                path,
                prediction_fields,
                (
                    {
                        "record_id": rows[i]["record_id"],
                        "split_group": groups[i],
                        "fold": int(fold),
                        "target": float(y[i]),
                        "prediction": float(value),
                        "dummy_prediction": float(prior),
                    }
                    for i, value, prior, fold in zip(
                        indices, pred, baseline, fold_values, strict=True
                    )
                ),
            )
        artifacts = {
            "model": str(model_path),
            "oof": str(oof_path),
            "holdout": str(holdout_path),
            "outer_split": str(split_path),
            "input_contract": str(contract_path),
        }
        refit = {"status": "not_requested", "attempts": []}
        if settings["refit_full"]:
            full = np.flatnonzero(common)
            refit_path = directory / "refit_model.joblib"
            refit = _refit_selected(estimator, X[full], y[full], refit_path, kind, settings)
            if refit["status"] == "completed":
                artifacts["refit_model"] = str(refit_path)
        refit_fields = {
            "refit_status": refit["status"],
            "refit_attempts": refit["attempts"],
            "refit_failure_reason": refit.get("reason", ""),
            "deployment_ready": refit["status"] == "completed",
        }
        model_contract = {
            "task": task,
            "selected_route": route,
            "selected_model": model_name,
            "selected_params": winner["params"],
            "target_bounds": settings["target_bounds"],
            "prediction_bounds_policy": "estimator_predict_clip_declared_target_space"
            if settings["target_bounds"] is not None
            else "unbounded_no_clipping",
            "validation_protocol": {
                "requested_inner_folds": settings["inner_folds"],
                "actual_inner_folds": len(splits),
                "strict_fold_count": True,
                "outer_holdout_frozen_before_feature_mask": True,
            },
            "model_capability": capability[route][model_name],
            "probability_calibration": (
                "svm_internal_platt_on_training_data_not_group_validated"
                if model_name == "svm" and kind == "classification"
                else "not_independently_calibrated"
                if kind == "classification"
                else "not_applicable"
            ),
            "linear_solver_policy": linear_policies[route] if model_name == "linear" else None,
            "linear_solver_policies_by_route": linear_policies,
            "feature_names": loaded[route]["feature_names"],
            "target_transform": task.get(
                "target_transform", rows[0].get("target_transform", "identity")
            ),
            "inverse_transform": task.get(
                "inverse_transform", rows[0].get("inverse_transform", "identity")
            ),
            "target_unit": task.get("unit", rows[0].get("unit", "")),
            "model_fit_scope": "outer_train_only",
            "holdout_not_used_for_selection": True,
            "selection_fit_admission": fit_admission,
            **refit_fields,
            "optional_refit_fit_scope": (
                "all_common_after_evaluation"
                if refit["status"] == "completed"
                else "refit_failed"
                if refit["status"] == "failed"
                else "not_run"
            ),
            "input_hash": signature,
            "versions": versions,
            "common_record_ids": [rows[i]["record_id"] for i in np.flatnonzero(common)],
        }
        contract = directory / "model_contract.json"
        _write_json(contract, model_contract)
        artifacts["model_contract"] = str(contract)
        result.update(
            status="completed",
            reason="",
            selected_route=route,
            selected_model=model_name,
            inner_cv=inner_metrics,
            holdout=holdout_metrics,
            holdout_metrics=holdout_metrics,
            baseline_inner_cv=_metrics(y[train], baseline_oof, kind),
            baseline_holdout=_metrics(y[test], baseline_holdout, kind),
            selected_params=winner["params"],
            **refit_fields,
            model_path=str(model_path),
            artifacts=artifacts,
            n_train=len(train),
            n_test=len(test),
            n_inner_folds=len(splits),
            train_groups=len(set(groups[train])),
            test_groups=len(set(groups[test])),
            source_holdout=_source_receipt(rows),
            versions=versions,
            comparison_scope="common_subset" if len(loaded) > 1 else "single_available_route",
            class_counts=_class_counts(y[common]) if kind == "classification" else {},
        )
        result["inner_improves_dummy"] = (
            _score(inner_metrics, kind) > _score(result["baseline_inner_cv"], kind) + 1e-12
        )
        result["performance_status"] = (
            "exploratory_inner_cv_gain_over_dummy"
            if result["inner_improves_dummy"]
            else "no_inner_cv_gain_over_dummy"
        )
        result["artifact_hashes"] = {
            path: {"sha256": _file_hash(path)} for path in artifacts.values()
        }
    except DeferredTraining as exc:
        result.update(status="deferred", reason=str(exc))
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}:{str(exc)[:1600]}")
    if result_path.exists() and result["status"] != "completed":
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            result["previous_completed_result"] = str(result_path)
            _write_json(directory / "last_failed_attempt.json", result)
            return result
    _write_json(result_path, result)
    return result
