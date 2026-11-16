"""Nested grouped evaluation of a source-reported cell-injury weak benchmark.

No label, assay ID, study ID, exposure or free text is an input feature. Context
vocabularies are learned inside each training fold. Outer predictions evaluate
the entire selection procedure; they never select an arm or a final model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import socket
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import peptide_modeling_train as training
from peptide_modeling_campaign import process_identity, representation_registry
from peptide_modeling_features import _verify_artifacts
from peptide_modeling_followup_seeds import parent_ready
from peptide_modeling_pipeline import load_rows, portable_features
from peptide_modeling_search import search_candidates
from peptide_modeling_supervise import atomic_json, digest, read_json
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

PROTOCOL = "source_qualitative_cell_injury_nested_grouped_5x5_v1"
CONTEXT = ("cell", "species", "readout_family")
ARMS = ("molecule_only", "context_only", "joint", "prevalence")
WARNING = (
    "Source-reported qualitative cell-injury weak benchmark. Thresholds and protocols "
    "are heterogeneous and primary tables are not independently validated. Predictions "
    "are uncalibrated source-report scores, not clinical safety or tolerated doses. "
    "Three seeds reuse the same observations; they are not independent experiments."
)


class ContextClassifier(ClassifierMixin, BaseEstimator):
    """Fold-local categorical encoding around the existing model implementation."""

    def __init__(self, model, params, settings, chemical_width, arm):
        self.model = model
        self.params = params
        self.settings = settings
        self.chemical_width = chemical_width
        self.arm = arm

    def _transform(self, X):
        chemical = np.asarray(X[:, : self.chemical_width], dtype=np.float32)
        if self.arm == "molecule_only":
            return chemical
        context = self.encoder_.transform(X[:, self.chemical_width :])
        return context if self.arm == "context_only" else np.column_stack((chemical, context))

    def fit(self, X, y):
        if self.arm not in ARMS[:3] or X.shape[1] != self.chemical_width + len(CONTEXT):
            raise ValueError("Unexpected contextual input schema")
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        self.encoder_.fit(X[:, self.chemical_width :])
        transformed = self._transform(X)
        estimator = training._estimator(
            self.model, "classification", self.params, self.settings, transformed.shape[1]
        )
        self.estimator_ = training._fit_model(estimator, transformed, y, self.settings["n_jobs"])
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def context_coverage(model, X):
    values = X[:, model.chemical_width :]
    known = np.column_stack(
        [
            np.isin(values[:, i], categories)
            for i, categories in enumerate(model.encoder_.categories_)
        ]
    )
    return {
        "supported_rows": int(known.all(axis=1).sum()),
        "total_rows": len(X),
        "unseen_by_field": {field: int((~known[:, i]).sum()) for i, field in enumerate(CONTEXT)},
        "training_categories": {
            field: categories.tolist()
            for field, categories in zip(CONTEXT, model.encoder_.categories_, strict=True)
        },
    }


def predict_guarded(model, chemical_X, chemical_names, contexts, contract):
    """Require the exact chemical schema and supported biological context."""
    if list(chemical_names) != contract["chemical_feature_names"]:
        raise ValueError("Chemical feature schema differs from the fitted contract")
    chemical_X = np.asarray(chemical_X, dtype=np.float32)
    if chemical_X.ndim != 2 or chemical_X.shape[1] != len(chemical_names):
        raise ValueError("Chemical feature array does not match the fitted schema")
    if np.isinf(chemical_X).any() or (
        chemical_X.shape[1] and np.isnan(chemical_X).all(axis=1).any()
    ):
        raise ValueError("Invalid chemical features")
    if len(chemical_X) != len(contexts) or not contexts:
        raise ValueError("One explicit context is required for each complete source graph")
    if any(set(context) != set(CONTEXT) for context in contexts):
        raise ValueError("Exactly cell, species and readout_family are required")
    values = np.asarray(
        [[context[field] for field in CONTEXT] for context in contexts], dtype=object
    )
    if any(not isinstance(value, str) or not value.strip() for value in values.ravel()):
        raise ValueError("Missing required inference context")
    X = np.column_stack((np.asarray(chemical_X, dtype=np.float32), values))
    if context_coverage(model, X)["supported_rows"] != len(X):
        raise ValueError("Unseen biological context: abstain outside training support")
    return training._predict(model, X, "classification")


def normalize_rows(rows):
    result = [
        {
            **row,
            "target": row.get("target", row.get("y")),
            "split_group": row.get("split_group", row.get("group_id")),
        }
        for row in rows
    ]
    training.validate_rows(result, "classification")
    for field in ("activity_id", "structure_key", "study_id", *CONTEXT):
        if any(not str(row.get(field, "")).strip() for row in result):
            raise ValueError("Missing source/group/context field: " + field)
    if len({row["activity_id"] for row in result}) != len(result):
        raise ValueError("Source activity IDs must be unique")
    for field in ("structure_key", "study_id"):
        memberships = {}
        for row in result:
            group = memberships.setdefault(row[field], row["split_group"])
            if group != row["split_group"]:
                raise ValueError("Source graph/study crosses connected-component groups")
    return result


def _assert_split(rows, left, right):
    if not len(left) or not len(right) or set(left) & set(right):
        raise ValueError("Empty or overlapping fold membership")
    for field in ("split_group", "structure_key", "study_id"):
        if {rows[i][field] for i in left} & {rows[i][field] for i in right}:
            raise ValueError("Graph/study/component leakage in " + field)
    if any(len({float(rows[i]["target"]) for i in indices}) != 2 for indices in (left, right)):
        raise ValueError("Both classes are required in every training and validation fold")


def nested_plan(rows, split_evidence, seed):
    """Recheck reviewed exact memberships instead of searching outcome-friendly splits."""
    probes = split_evidence.get(
        "probes", split_evidence.get("validation_contract", {}).get("probes", [])
    )
    probe = next((value for value in probes if value["seed"] == seed), None)
    if probe is None or len(probe["outer_folds"]) != 5:
        raise ValueError("Reviewed exact outer fivefold plan is required for each seed")
    universe = {row["activity_id"] for row in rows}
    seen, plan = [], []
    for fold in probe["outer_folds"]:
        test_ids = set(fold["test_activity_ids"])
        if not test_ids <= universe:
            raise ValueError("Reviewed split contains unknown source activities")
        train = np.asarray([i for i, row in enumerate(rows) if row["activity_id"] not in test_ids])
        test = np.asarray([i for i, row in enumerate(rows) if row["activity_id"] in test_ids])
        _assert_split(rows, train, test)
        inner, inner_seen = [], []
        if len(fold["inner_folds"]) != 5:
            raise ValueError("Every outer training set requires five reviewed inner folds")
        train_rows = [rows[i] for i in train]
        train_ids = {row["activity_id"] for row in train_rows}
        for value in fold["inner_folds"]:
            valid_ids = set(value["test_activity_ids"])
            if not valid_ids <= train_ids:
                raise ValueError("An outer test activity entered the inner selection plan")
            a = np.asarray(
                [i for i, row in enumerate(train_rows) if row["activity_id"] not in valid_ids]
            )
            b = np.asarray(
                [i for i, row in enumerate(train_rows) if row["activity_id"] in valid_ids]
            )
            _assert_split(train_rows, a, b)
            inner.append((a, b))
            inner_seen.extend(b.tolist())
        if sorted(inner_seen) != list(range(len(train))):
            raise ValueError("Inner OOF membership must cover every training row exactly once")
        plan.append((train, test, inner))
        seen.extend(test.tolist())
    if sorted(seen) != list(range(len(rows))):
        raise ValueError("Outer OOF membership must cover every row exactly once")
    return plan


def _weighted_metrics(y, prediction, weights):
    bins = np.minimum((prediction * 10).astype(int), 9)
    calibration = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            calibration += (
                weights[mask].sum()
                / weights.sum()
                * abs(
                    np.average(y[mask], weights=weights[mask])
                    - np.average(prediction[mask], weights=weights[mask])
                )
            )
    return {
        "pr_auc": float(average_precision_score(y, prediction, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y, prediction, sample_weight=weights)),
        "brier": float(brier_score_loss(y, prediction, sample_weight=weights)),
        "ece_fixed_10_bins": float(calibration),
    }


def performance(rows, prediction, seed, bootstrap=300):
    y = np.asarray([float(row["target"]) for row in rows])
    counts = Counter(row["structure_key"] for row in rows)
    weights = np.asarray([1 / counts[row["structure_key"]] for row in rows])
    groups = np.asarray([row["split_group"] for row in rows])
    unique = sorted(set(groups))
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        sampled = Counter(rng.choice(unique, len(unique), replace=True))
        weighted = weights * np.asarray([sampled[group] for group in groups])
        if len(set(y[weighted > 0])) == 2:
            draws.append(_weighted_metrics(y, prediction, weighted))
    return {
        "record_metrics": training._metrics(y, prediction, "classification"),
        "molecule_equal_weight_metrics": _weighted_metrics(y, prediction, weights),
        "component_bootstrap_95_percent": {
            key: np.quantile([draw[key] for draw in draws], [0.025, 0.975]).tolist()
            for key in draws[0]
        }
        if draws
        else {},
        "bootstrap_valid_draws": len(draws),
        "calibration": "diagnostic_only_no_independent_probability_calibration",
    }


def matrix_fingerprint(X):
    """Hash typed cell values, preserving NaN missingness without object-pointer bytes."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Feature fingerprints require a two-dimensional matrix")
    signature = hashlib.sha256()
    for value in X.flat:
        if isinstance(value, (str, np.str_)):
            tag, encoded = b"s", str(value).encode("utf-8")
        elif isinstance(value, (bool, np.bool_)):
            tag, encoded = b"b", b"1" if value else b"0"
        elif isinstance(value, (int, np.integer)):
            tag, encoded = b"i", str(int(value)).encode("ascii")
        elif isinstance(value, (float, np.floating)):
            number = float(value)
            if np.isinf(number):
                raise ValueError("Infinite feature values cannot be fingerprinted")
            tag, encoded = b"f", number.hex().encode("ascii")
        elif value is None:
            tag, encoded = b"n", b""
        else:
            raise ValueError("Unsupported feature fingerprint cell type")
        signature.update(tag + len(encoded).to_bytes(8, "big") + encoded)
    return {
        "schema": "typed_feature_matrix_fingerprint_v1",
        "shape": list(X.shape),
        "dtype": X.dtype.str,
        "data_sha256": signature.hexdigest(),
    }


def _fit_arm(loaded, rows, train, inner, arm, directory, settings):
    """Only outer-training rows and labels enter Optuna or persistence admission."""
    if arm == "context_only":
        settings = {
            **settings,
            "search_budget": min(
                int(settings.get("context_search_budget", 64)), int(settings["search_budget"])
            ),
        }
    directory.mkdir(parents=True, exist_ok=True)
    y = np.asarray([float(rows[i]["target"]) for i in train])
    local = {route: {**value, "X": value["X"][train]} for route, value in loaded.items()}
    pairs = []
    for route, payload in local.items():
        for model in settings["models"]:
            width = payload["chemical_width"] + len(train) * len(CONTEXT)
            if training.model_capability(model, "classification", len(train), width, settings)[
                "eligible"
            ]:
                pairs.append((route, model))
    signature = training._digest(
        {
            "protocol": PROTOCOL,
            "arm": arm,
            "settings": settings,
            "training_rows": [rows[i] for i in train],
            "features": {
                route: {
                    "chemical_names": payload["feature_names"],
                    "matrix": matrix_fingerprint(payload["X"]),
                }
                for route, payload in local.items()
            },
            "inner": [[a.tolist(), b.tolist()] for a, b in inner],
            "implementation": digest(__file__),
            "training_implementation": digest(training.__file__),
        }
    )

    def estimator(route, model, params):
        return ContextClassifier(model, params, settings, local[route]["chemical_width"], arm)

    def evaluate(route, model, params):
        key = training._digest([signature, route, model, params])
        path = directory / "candidates" / (key + ".json")
        if path.exists():
            return read_json(path)
        candidate = {"route": route, "model": model, "params": params, "requested_folds": 5}
        try:
            prediction, _ = training._cv_predictions(
                estimator(route, model, params),
                local[route]["X"],
                y,
                inner,
                "classification",
                settings["n_jobs"],
            )
            metrics = training._metrics(y, prediction, "classification")
            candidate.update(
                status="completed",
                actual_folds=5,
                score=training._score(metrics, "classification"),
                metrics=metrics,
            )
        except Exception as exc:
            candidate.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(path, candidate)
        return candidate

    search = search_candidates(
        pairs, evaluate, settings, directory / "search", signature, "classification"
    )
    admission, seen = [], set()
    for winner in sorted(
        (value for value in search["trials"] if value["status"] == "completed"),
        key=lambda value: value["score"],
        reverse=True,
    ):
        key = training._digest([winner["route"], winner["model"], winner["params"]])
        if key in seen:
            continue
        seen.add(key)
        try:
            model = training._fit_model(
                estimator(winner["route"], winner["model"], winner["params"]),
                local[winner["route"]]["X"],
                y,
                settings["n_jobs"],
            )
            path = directory / "model.joblib"
            training._save_checked_model(
                model, path, local[winner["route"]]["X"], "classification", settings["n_jobs"]
            )
            break
        except Exception as exc:
            admission.append({"candidate": winner, "reason": f"{type(exc).__name__}:{exc}"})
    else:
        raise RuntimeError("No inner-CV candidate passed outer-training fit/persistence")
    contract = {
        "protocol": PROTOCOL,
        "warning": WARNING,
        "arm": arm,
        "input_hash": signature,
        "selection": winner,
        "search": {key: value for key, value in search.items() if key != "trials"},
        "chemical_feature_names": local[winner["route"]]["feature_names"],
        "context_fields": list(CONTEXT),
        "context_encoding": "one_hot_fitted_on_each_training_fold_only",
        "forbidden_inputs": [
            "source_ids",
            "study_ids",
            "assay_ids",
            "labels",
            "free_text",
            "exposure",
            "duration",
        ],
        "training_record_ids": [rows[i]["record_id"] for i in train],
        "rejected_fit_candidates": admission,
        "outer_labels_used_for_selection": False,
        "model_sha256": digest(path),
        "context_is_additional_chemical_method": False,
    }
    atomic_json(directory / "model_contract.json", contract)
    return model, contract, path


def train_seed(rows, features, split_evidence, run_dir, config):
    settings = training._settings(config)
    if settings["inner_folds"] != 5 or settings["search_backend"] != "optuna":
        raise ValueError("This benchmark requires Optuna and strict fivefold inner CV")
    rows, root = normalize_rows(rows), Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _verify_artifacts(features)
    loaded, common, coverage, excluded = training.prepare_routes(
        rows, features, {**settings, "min_rows": len(rows)}
    )
    if not loaded or not common.all() or excluded:
        raise ValueError("Every declared method must cover every admitted source record")
    plan = nested_plan(rows, split_evidence, settings["seed"])
    context = np.asarray([[row[field] for field in CONTEXT] for row in rows], dtype=object)
    molecule = {
        route: {
            **payload,
            "chemical_width": payload["X"].shape[1],
            "X": np.column_stack((payload["X"], context)),
        }
        for route, payload in loaded.items()
    }
    context_payload = {"context": {"X": context, "chemical_width": 0, "feature_names": []}}
    y = np.asarray([float(row["target"]) for row in rows])
    predictions = {arm: np.full(len(rows), np.nan) for arm in ARMS}
    fold_ids, fold_results = np.full(len(rows), -1), []
    for fold, (train, test, inner) in enumerate(plan):
        fold_ids[test] = fold
        fold_result = {
            "fold": fold,
            "actual_inner_folds": 5,
            "train_record_ids": [rows[i]["record_id"] for i in train],
            "test_record_ids": [rows[i]["record_id"] for i in test],
            "arms": {},
        }
        for arm in ARMS[:3]:
            matrices = context_payload if arm == "context_only" else molecule
            directory = root / "models" / f"outer_{fold}" / arm
            model, contract, path = _fit_arm(matrices, rows, train, inner, arm, directory, settings)
            selected = matrices[contract["selection"]["route"]]["X"]
            predictions[arm][test] = training._predict(model, selected[test], "classification")
            fold_result["arms"][arm] = {
                "selected_route": contract["selection"]["route"],
                "selected_model": contract["selection"]["model"],
                "model_path": str(path.relative_to(root)),
                "model_sha256": digest(path),
                "contract_path": str((directory / "model_contract.json").relative_to(root)),
                "contract_sha256": digest(directory / "model_contract.json"),
                "context_coverage": context_coverage(model, selected[test]),
                "actual_inner_folds": 5,
                "search": contract["search"],
            }
        baseline = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), y[train])
        baseline_path = root / "models" / f"outer_{fold}" / "prevalence.joblib"
        training._save_checked_model(
            baseline, baseline_path, np.zeros((len(train), 1)), "classification", 1
        )
        predictions["prevalence"][test] = training._predict(
            baseline, np.zeros((len(test), 1)), "classification"
        )
        fold_result["arms"]["prevalence"] = {
            "model_path": str(baseline_path.relative_to(root)),
            "model_sha256": digest(baseline_path),
        }
        fold_results.append(fold_result)
        atomic_json(root / "fold_progress.json", fold_results)
    output = root / "outer_oof.tsv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "activity_id",
                "group_id",
                "structure_key",
                "study_id",
                "target",
                "outer_fold",
                *ARMS,
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{
                        field: row[field]
                        for field in (
                            "record_id",
                            "activity_id",
                            "structure_key",
                            "study_id",
                            "target",
                        )
                    },
                    "group_id": row["split_group"],
                    "outer_fold": int(fold_ids[index]),
                    **{arm: float(predictions[arm][index]) for arm in ARMS},
                }
            )
    result = {
        "status": "completed",
        "protocol": PROTOCOL,
        "warning": WARNING,
        "seed": settings["seed"],
        "actual_outer_folds": 5,
        "actual_inner_folds_per_outer": 5,
        "n_records": len(rows),
        "n_graphs": len({row["structure_key"] for row in rows}),
        "n_groups": len({row["split_group"] for row in rows}),
        "coverage": coverage,
        "source_rows_hash": training._digest(rows),
        "feature_fingerprint": features["input_fingerprint"],
        "folds": fold_results,
        "outer_oof_sha256": digest(output),
        "performance": {arm: performance(rows, predictions[arm], settings["seed"]) for arm in ARMS},
        "outer_scores_used_to_choose_arm": False,
        "deployment_model_fitted": False,
        "independent_external_validation": False,
        "context_counted_as_chemical_method": False,
        "outer_evaluation_includes_unseen_contexts_for_diagnostics": True,
        "operational_inference_abstains_on_unseen_context": True,
    }
    atomic_json(root / "result.json", result)
    verify_seed(rows, features, split_evidence, root, result)
    return result


def verify_seed(rows, features, split_evidence, root, result):
    rows, root = normalize_rows(rows), Path(root)
    if result["protocol"] != PROTOCOL or result["source_rows_hash"] != training._digest(rows):
        raise ValueError("Result protocol/source rows mismatch")
    _verify_artifacts(features)
    if result["feature_fingerprint"] != features["input_fingerprint"]:
        raise ValueError("Feature bank fingerprint differs from evaluated source")
    plan = nested_plan(rows, split_evidence, result["seed"])
    if result["actual_outer_folds"] != 5 or result["actual_inner_folds_per_outer"] != 5:
        raise ValueError("Result does not prove real nested fivefold")
    path = root / "outer_oof.tsv"
    if digest(path) != result["outer_oof_sha256"]:
        raise ValueError("OOF artifact hash mismatch")
    values = load_rows(path)
    if [row["record_id"] for row in values] != [row["record_id"] for row in rows]:
        raise ValueError("OOF record order differs from source")
    context = np.asarray([[row[field] for field in CONTEXT] for row in rows], dtype=object)
    for fold, (train, test, _) in enumerate(plan):
        evidence = result["folds"][fold]
        if evidence["train_record_ids"] != [rows[i]["record_id"] for i in train] or evidence[
            "test_record_ids"
        ] != [rows[i]["record_id"] for i in test]:
            raise ValueError("Persisted fold membership differs from reviewed plan")
        for arm in ARMS:
            receipt = evidence["arms"][arm]
            model_path = root / receipt["model_path"]
            if digest(model_path) != receipt["model_sha256"]:
                raise ValueError("Model hash mismatch")
            model = joblib.load(model_path)
            if arm == "prevalence":
                X = np.zeros((len(rows), 1))
            else:
                contract_path = root / receipt["contract_path"]
                if digest(contract_path) != receipt["contract_sha256"]:
                    raise ValueError("Model contract hash mismatch")
                contract = read_json(contract_path)
                if contract["search"].get("backend") != "optuna":
                    raise ValueError("Result did not use the requested Optuna backend")
                if contract["outer_labels_used_for_selection"] or contract[
                    "training_record_ids"
                ] != [rows[i]["record_id"] for i in train]:
                    raise ValueError("Model selection used a non-training cohort")
                if arm == "context_only":
                    X = context
                else:
                    payload = training._load_feature_route(
                        rows, features, receipt["selected_route"]
                    )
                    X = np.column_stack((payload["X"], context))
            prediction = training._predict(model, X[test], "classification")
            actual = np.asarray([float(values[i][arm]) for i in test])
            if not np.allclose(prediction, actual, rtol=1e-7, atol=1e-8):
                raise ValueError("Reloaded outer predictions differ from OOF artifact")
        for i in test:
            if (
                float(values[i]["target"]) != float(rows[i]["target"])
                or int(values[i]["outer_fold"]) != fold
            ):
                raise ValueError("OOF target/fold differs from reviewed source")
    proof = {
        "status": "verified",
        "protocol": PROTOCOL,
        "seed": result["seed"],
        "outer_folds": 5,
        "inner_folds_per_outer": 5,
        "graph_study_overlap": 0,
        "model_reload_verified": True,
        "records": len(rows),
    }
    atomic_json(root / "verification.json", proof)
    return proof


def freeze_inputs(template):
    """Pin staged inputs before waiting, including code and cache pointer metadata."""
    files = {}
    for path in sorted(Path(template).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("Prepared source inputs cannot be symlinks")
        files[path.relative_to(template).as_posix()] = digest(path)
    return files


def verify_inputs(template, expected):
    if freeze_inputs(template) != expected:
        raise ValueError("Prepared table/splits/features/source inputs changed while waiting")


def method_model_coverage(result, methods):
    required = {method["route"]: method["method_id"] for method in methods}
    proofs = []
    for fold in result["folds"]:
        for arm in ARMS[:3]:
            search = fold["arms"][arm]["search"]
            pairs = search.get("coverage_attempted", [])
            routes, models = {pair[0] for pair in pairs}, {pair[1] for pair in pairs}
            methods_used = sorted(required[route] for route in routes if route in required)
            proofs.append(
                {
                    "fold": fold["fold"],
                    "arm": arm,
                    "attempted_independent_chemical_methods": methods_used,
                    "attempted_model_families": sorted(models),
                    "requirements_met": search.get("backend") == "optuna"
                    and len(models) >= 7
                    and (arm == "context_only" or set(required) <= routes),
                }
            )
    return {
        "requirements_met": all(proof["requirements_met"] for proof in proofs),
        "fold_arms": proofs,
    }


def run_followup(config_path):
    """One process/thread after the parent gate; exactly three finite seed runs."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    root, durable = Path(config["output_root"]).resolve(), Path(config["durable_root"]).resolve()
    template = Path(config["template_run"]).resolve()
    if any(
        a == b or a in b.parents or b in a.parents for a, b in ((root, durable), (root, template))
    ):
        raise ValueError("Template, scratch output and durable roots must be separate")
    if os.name != "posix" or socket.gethostname() != config.get("training_host", "node23"):
        raise ValueError("Followup must run on its declared Linux host")
    settings = training._settings(config["training"])
    _, methods = representation_registry(config["registry_path"])
    if (
        settings["n_jobs"] != 1
        or settings["inner_folds"] != 5
        or settings["search_backend"] != "optuna"
    ):
        raise ValueError("Followup requires one thread, strict fivefold and Optuna")
    if len(set(settings["models"])) < 7 or any(
        method["route"] not in settings["routes"] for method in methods
    ):
        raise ValueError(
            "All ten chemical methods and at least seven model families must participate"
        )
    if config.get("required_seeds", [42, 2026, 3407]) != [42, 2026, 3407]:
        raise ValueError("The three required independent seeds are 42, 2026 and 3407")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "qualitative_status.json").exists():
        raise FileExistsError("Existing followup requires an explicit recovery decision")
    descriptor = os.open(root / "qualitative.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    journal = {
        "status": "waiting_parent",
        "protocol": PROTOCOL,
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "config_sha256": digest(config_path),
        "rounds": [],
        "additional_twenty_hour_loop": False,
        "thread_budget": 1,
    }
    frozen_inputs = freeze_inputs(template)
    journal["input_files_sha256"] = frozen_inputs

    def save():
        if digest(config_path) != journal["config_sha256"]:
            raise ValueError("Running followup configuration changed")
        atomic_json(root / "qualitative_status.json", journal)
        atomic_json(durable / "qualitative_status.json", journal)

    try:
        save()
        while (evidence := parent_ready(config)) is None:
            time.sleep(min(60, max(1, float(config.get("poll_seconds", 5)))))
            save()
        journal["parent_requirement_evidence"] = evidence
        verify_inputs(template, frozen_inputs)
        os.nice(10)
        rows = load_rows(template / config["table_path"])
        features = portable_features(template)
        split_evidence = read_json(template / config["split_evidence_path"])
        for seed in (42, 2026, 3407):
            verify_inputs(template, frozen_inputs)
            run = root / f"seed_{seed}"
            if run.exists():
                raise FileExistsError(run)
            entry = {"seed": seed, "status": "running"}
            journal["rounds"].append(entry)
            journal["status"] = "running"
            save()
            with threadpool_limits(limits=1):
                result = train_seed(rows, features, split_evidence, run, {**settings, "seed": seed})
            entry.update(
                status=result["status"],
                verification=read_json(run / "verification.json"),
                result_sha256=digest(run / "result.json"),
                method_model_evidence=method_model_coverage(result, methods),
            )
            shutil.copytree(run, durable / run.name)
            for source in run.rglob("*"):
                if source.is_file() and digest(source) != digest(
                    durable / run.name / source.relative_to(run)
                ):
                    raise ValueError("Durable artifact copy hash mismatch")
            save()
        summary = {
            "protocol": PROTOCOL,
            "warning": WARNING,
            "seeds": [entry["seed"] for entry in journal["rounds"]],
            "arm_selection_from_outer_scores": False,
            "scores": {
                str(entry["seed"]): read_json(root / f"seed_{entry['seed']}" / "result.json")[
                    "performance"
                ]
                for entry in journal["rounds"]
            },
            "independent_external_validation": False,
        }
        atomic_json(root / "performance_summary.json", summary)
        atomic_json(durable / "performance_summary.json", summary)
        journal["status"] = (
            "completed"
            if all(
                entry["method_model_evidence"]["requirements_met"] for entry in journal["rounds"]
            )
            else "requirement_incomplete"
        )
        save()
        return journal
    except BaseException as exc:
        journal.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_followup(args.config)


if __name__ == "__main__":
    # Persist classes under their importable module name, including direct CLI use.
    from peptide_modeling_qualitative_cyto import main as canonical_main

    canonical_main()
