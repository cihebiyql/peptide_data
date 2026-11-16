"""Exploratory five-fold internal CV for tiny, scientifically defined tasks.

These scores are used for tuning and are selection-optimistic. There is no
independent holdout. They must never be pooled with outer-holdout benchmarks.

CLI: python peptide_modeling_internal_cv.py --config followup.json
The bounded runner accepts template_run, template_config, output_root,
durable_root, task_ids, parent_campaign_root, parent_config_sha256,
required_seeds (42,2026,3407), workers (1 or 2), and poll_seconds. It waits for
the parent's verified twenty-hour run and executes exactly those three seeds.
The template contains only its declared tasks. Its training configuration must
request Optuna, five folds, at least seven models and ten independent methods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import peptide_modeling_train as training
import sklearn
from peptide_modeling_campaign import (
    process_identity,
    representation_registry,
    stage_round,
    template_contract,
)
from peptide_modeling_features import _verify_artifacts
from peptide_modeling_followup_seeds import parent_ready
from peptide_modeling_pipeline import load_rows, portable_features
from peptide_modeling_supervise import archive_run, atomic_json, digest, read_json
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

PROTOCOL = "internal_grouped_cv_only_no_independent_holdout"
WARNING = (
    "Tiny-sample exploration. These same five-fold predictions select the model; "
    "CV scores are selection-optimistic, not unbiased generalization estimates. "
    "Additional random seeds do not create independent observations."
)
SAFE = re.compile(r"[A-Za-z0-9_.-]+")


def runtime_contract():
    return {
        "hostname": socket.gethostname(),
        "python": list(sys.version_info[:3]),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "training_source_sha256": digest(training.__file__),
        "internal_cv_source_sha256": digest(__file__),
    }


def internal_splits(y, groups, kind, config):
    if int(config.get("inner_folds", 5)) != 5:
        raise ValueError("Internal exploration requires exactly five folds")
    if len(set(groups)) < 5:
        raise training.DeferredTraining("requested_5_fold_insufficient_independent_groups")
    if kind == "classification":
        return training._inner_splits(y, groups, kind, config)
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=int(config["seed"]))
    splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    if any(set(groups[a]) & set(groups[b]) for a, b in splits):
        raise ValueError("An identity group crosses an internal fold")
    return splits


def _settings(config):
    settings = training._settings(
        {
            "min_rows": 5,
            "min_groups": 5,
            "n_jobs": 1,
            "search_backend": "optuna",
            "search_budget": 128,
            **config,
        }
    )
    if settings["inner_folds"] != 5:
        raise ValueError("Internal exploration requires exactly five folds")
    return settings


def _result_path(root, task_id):
    if not SAFE.fullmatch(task_id):
        raise ValueError("Task IDs must be safe filenames")
    return Path(root) / "results/internal_cv" / (task_id + ".json")


def _external_context_module(task, external):
    """Bind each supported physical target to its explicit context contract."""
    if task.get("task_kind") != "regression" or not isinstance(external, dict):
        raise ValueError("External context requires a supported regression contract")
    if task.get("endpoint") == "Cmax":
        import peptide_modeling_conditional_cmax as context
    elif task.get("endpoint") == "CL_absolute":
        import peptide_modeling_conditional_cl as context
    elif task.get("endpoint") == "Vd_per_kg":
        import peptide_modeling_conditional_volume as context
    else:
        raise ValueError("Unsupported external-context endpoint")
    if external.get("schema") != context.SCHEMA:
        raise ValueError("External context schema does not match the endpoint")
    context.validate_contract(external)
    return context


def train_internal_task(task, rows, features, run_dir, config):
    """Tune on real five-fold OOF predictions, then fit all eligible observations."""
    settings, root = _settings(config), Path(run_dir).resolve()
    task_id, kind = str(task["task_id"]), task["task_kind"]
    result_path = _result_path(root, task_id)
    directory = root / "models/internal_cv" / task_id
    directory.mkdir(parents=True, exist_ok=True)
    result = {
        "task_id": task_id,
        "endpoint": task.get("endpoint"),
        "task_kind": kind,
        "protocol": PROTOCOL,
        "status": "failed",
        "seed": settings["seed"],
        "requested_inner_folds": 5,
        "independent_holdout_available": False,
        "selection_optimism_warning": WARNING,
        "n_task_rows": len(rows),
        "runtime": runtime_contract(),
    }
    try:
        if features.get("artifact_hashes"):
            _verify_artifacts(features)
        y, groups = training.validate_rows(rows, kind)
        bounds = task.get("target_bounds", task.get("task_definition", {}).get("target_bounds"))
        bounds = training._validated_target_bounds(bounds) if kind == "regression" else None
        transform = task.get("target_transform", rows[0].get("target_transform", "identity"))
        result.update(
            target_transform=transform,
            target_unit=task.get("unit", rows[0].get("unit", "unknown")),
            metrics_space="declared_transformed_target_space",
        )
        if bounds is not None and transform != "identity":
            raise ValueError("declared_target_bounds_require_explicit_identity_target_space")
        if bounds is not None and (np.any(y < bounds[0]) or np.any(y > bounds[1])):
            raise ValueError("observed_targets_violate_declared_target_bounds")
        settings["target_bounds"] = list(bounds) if bounds is not None else None
        loaded, common, coverage, excluded = training.prepare_routes(rows, features, settings)
        external = task.get("fixed_external_covariates")
        if external is not None:
            context = _external_context_module(task, external)
            loaded = {
                route: context.append_context(payload, rows, external)
                for route, payload in loaded.items()
            }
            result["external_context"] = {
                "schema": external["schema"],
                "sha256": external["sha256"],
                "chemical_method_count_increment": 0,
                "warning": external["warning"],
                "context_columns": len(external["feature_names"]),
                "implementation_sha256": digest(context.__file__),
            }
        result.update(coverage=coverage, excluded_routes=excluded)
        indices = np.flatnonzero(common)
        if len(indices) < max(5, settings["min_rows"]):
            raise training.DeferredTraining("insufficient_common_feature_rows_for_fivefold")
        splits = internal_splits(y[indices], groups[indices], kind, settings)
        if kind == "regression" and len(np.unique(y[indices])) < 2:
            raise training.DeferredTraining("regression_target_has_no_variation_after_feature_mask")
        result.update(
            actual_inner_folds=5,
            n_internal_cv_rows=len(indices),
            n_internal_cv_groups=len(set(groups[indices])),
        )
        feature_contract = {
            route: {
                "names": payload["feature_names"],
                "matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(payload["X"]).tobytes()
                ).hexdigest(),
                "valid_sha256": hashlib.sha256(payload["valid"].tobytes()).hexdigest(),
            }
            for route, payload in loaded.items()
        }
        signature = training._digest(
            {
                "protocol": PROTOCOL,
                "task": task,
                "config": settings,
                "rows": [(r["record_id"], r["target"], r["split_group"]) for r in rows],
                "features": feature_contract,
                "feature_fingerprint": features.get("input_fingerprint"),
                "runtime": result["runtime"],
                **(
                    {"external_context": result["external_context"]} if external is not None else {}
                ),
            }
        )
        contract_path = directory / "input_contract.json"
        if contract_path.exists() and read_json(contract_path)["input_hash"] != signature:
            raise ValueError("Existing internal task has different inputs; use a fresh run")
        atomic_json(
            contract_path,
            {
                "protocol": PROTOCOL,
                "input_hash": signature,
                "task": task,
                "config": settings,
                "features": feature_contract,
                "runtime": result["runtime"],
                "source_rows_hash": training._digest(rows),
            },
        )
        result["input_hash"] = signature
        if result_path.exists() and read_json(result_path).get("status") == "completed":
            prior = read_json(result_path)
            if prior["input_hash"] != signature:
                raise ValueError("Existing completed internal result has different inputs")
            verify_internal_task(task, rows, features, root, prior)
            return {**prior, "resumed": True}
        pairs, capabilities = [], {}
        for route, payload in loaded.items():
            capabilities[route] = {}
            for model in settings["models"]:
                capability = training.model_capability(
                    model, kind, len(indices), payload["X"].shape[1], settings
                )
                capabilities[route][model] = capability
                if capability["eligible"]:
                    pairs.append((route, model))
        if not pairs:
            raise training.DeferredTraining("no_eligible_model_method_pair")
        result["model_capabilities"] = capabilities
        trial_directory = directory / "candidate_cv"
        trial_directory.mkdir(exist_ok=True)

        def evaluate(route, model, params):
            key = training._digest([signature, route, model, params])
            path = trial_directory / (key + ".json")
            if path.exists():
                return read_json(path)
            candidate = {
                "route": route,
                "model": model,
                "params": params,
                "input_hash": signature,
                "requested_folds": 5,
                "actual_folds": 5,
            }
            try:
                estimator = training._estimator(
                    model, kind, params, settings, loaded[route]["X"].shape[1]
                )
                predictions, _ = training._cv_predictions(
                    estimator,
                    loaded[route]["X"][indices],
                    y[indices],
                    splits,
                    kind,
                    settings["n_jobs"],
                )
                metrics = training._metrics(y[indices], predictions, kind)
                candidate.update(
                    status="completed", metrics=metrics, score=training._score(metrics, kind)
                )
            except Exception as exc:
                candidate.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
            atomic_json(path, candidate)
            return candidate

        if settings["search_backend"] == "optuna":
            from peptide_modeling_search import search_candidates

            search = search_candidates(
                pairs, evaluate, settings, directory / "search", signature, kind
            )
            candidates = search["trials"]
            result["search"] = {k: v for k, v in search.items() if k != "trials"}
        elif settings["search_backend"] == "random_per_pair":
            # Explicit local test/development backend, never an Optuna fallback.
            candidates = [
                evaluate(route, model, params)
                for route, model in pairs
                for params in training._parameter_candidates(model, {**settings, "task_kind": kind})
            ]
            result["search"] = {
                "backend": "random_per_pair",
                "coverage_attempted": [list(pair) for pair in pairs],
            }
        else:
            raise ValueError("Unsupported explicit internal-CV search backend")
        successful = [candidate for candidate in candidates if candidate["status"] == "completed"]
        if not successful:
            raise RuntimeError("All internal-CV candidates failed; no dummy fallback")
        winner, _, fitted, oof, folds, model_path, admission = training._fit_cv_ranked_candidate(
            successful, loaded, indices, y[indices], splits, kind, settings, directory
        )
        admission.update(
            policy="internal_cv_rank_then_full_internal_cohort_fit_and_persistence",
            validation_labels="all_internal_cv_records_no_independent_holdout",
        )
        admission_path = directory / "candidate_fit_admission.json"
        atomic_json(admission_path, admission)
        prediction_path = directory / "internal_oof.tsv"
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["record_id", "split_group", "fold", "target", "prediction"],
                delimiter="\t",
            )
            writer.writeheader()
            for index, prediction, fold in zip(indices, oof, folds, strict=True):
                writer.writerow(
                    {
                        "record_id": rows[index]["record_id"],
                        "split_group": str(groups[index]),
                        "fold": int(fold),
                        "target": float(y[index]),
                        "prediction": float(prediction),
                    }
                )
        metrics = training._metrics(y[indices], oof, kind)
        model_contract_path = directory / "model_contract.json"
        model_contract = {
            "protocol": PROTOCOL,
            "task": task,
            "input_hash": signature,
            "runtime": result["runtime"],
            "requested_inner_folds": 5,
            "actual_inner_folds": 5,
            "independent_holdout_available": False,
            "selection_optimism_warning": WARNING,
            "fit_scope": "all_internal_cv_rows",
            "selected_route": winner["route"],
            "selected_model": winner["model"],
            "selected_params": winner["params"],
            "feature_names": loaded[winner["route"]]["feature_names"],
            "record_ids": [rows[index]["record_id"] for index in indices],
            "target_transform": transform,
            "target_unit": result["target_unit"],
            "target_bounds": settings["target_bounds"],
            "selection_fit_admission": admission,
        }
        if external is not None:
            model_contract["fixed_external_covariates"] = external
            model_contract["external_context"] = result["external_context"]
        atomic_json(model_contract_path, model_contract)
        paths = {
            "model": model_path,
            "internal_oof": prediction_path,
            "input_contract": contract_path,
            "model_contract": model_contract_path,
            "candidate_fit_admission": admission_path,
        }
        if external is not None:
            diagnostics = context.contextual_diagnostics(
                [rows[index] for index in indices],
                external,
                loaded[winner["route"]]["X"][indices, : -len(context.FEATURE_NAMES)],
                y[indices],
                groups[indices],
                splits,
                settings,
                training,
            )
            diagnostics["selected_chemical_route"] = winner["route"]
            diagnostics_path = directory / "context_ablations.json"
            atomic_json(diagnostics_path, diagnostics)
            paths["context_ablations"] = diagnostics_path
        result.update(
            status="completed",
            selected_route=winner["route"],
            selected_model=winner["model"],
            selected_params=winner["params"],
            internal_cv_selection_metrics=metrics,
            internal_cv_selection_score=training._score(metrics, kind),
            artifacts={name: str(path) for name, path in paths.items()},
            artifact_hashes={str(path): {"sha256": digest(path)} for path in paths.values()},
        )
        # Validate the actual fitted artifact on the same host, with identical feature order.
        verify_internal_task(task, rows, features, root, result, fitted=fitted)
    except training.DeferredTraining as exc:
        result.update(status="deferred", reason=str(exc))
    except Exception as exc:
        result.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
    atomic_json(result_path, result)
    return result


def verify_internal_task(task, rows, features, run_dir, result, *, fitted=None):
    """Check target/group provenance, real fivefold OOF and same-host persistence."""
    root = Path(run_dir).resolve()
    if (
        result.get("protocol") != PROTOCOL
        or result.get("independent_holdout_available") is not False
    ):
        raise ValueError("Internal-only protocol must explicitly exclude independent holdout")
    if result.get("actual_inner_folds") != 5 or result.get("requested_inner_folds") != 5:
        raise ValueError("Exactly five actual internal folds are required")
    if (
        "holdout" in result
        or "holdout_metrics" in result
        or "holdout" in result.get("artifacts", {})
    ):
        raise ValueError("Internal-only results cannot contain fabricated holdout evidence")
    if result["runtime"] != runtime_contract():
        raise ValueError("Model reload requires the same training host, versions and source")
    for name, receipt in result["artifact_hashes"].items():
        path = Path(name).resolve()
        path.relative_to(root)
        if digest(path) != receipt["sha256"]:
            raise ValueError("Internal artifact hash mismatch")
    contract = read_json(result["artifacts"]["model_contract"])
    if contract["input_hash"] != result["input_hash"] or contract["task"] != task:
        raise ValueError("Internal model contract does not match task or input")
    if any(
        result.get(key) != contract.get(key)
        for key in ("selected_route", "selected_model", "selected_params")
    ):
        raise ValueError("Selected internal model metadata differs from its contract")
    training.validate_rows(rows, task["task_kind"])
    inputs = read_json(result["artifacts"]["input_contract"])
    if inputs.get("source_rows_hash") != training._digest(rows):
        raise ValueError("Internal source row provenance changed")
    by_id = {row["record_id"]: row for row in rows}
    oof = load_rows(result["artifacts"]["internal_oof"])
    ids = [row["record_id"] for row in oof]
    if len(set(ids)) != len(ids) or ids != contract["record_ids"] or set(ids) - set(by_id):
        raise ValueError("Internal OOF row identity mismatch")
    group_folds = {}
    for row in oof:
        original = by_id[row["record_id"]]
        if row["split_group"] != original["split_group"]:
            raise ValueError("Internal OOF group does not match source")
        if float(row["target"]) != float(original["target"]):
            raise ValueError("Internal OOF target does not match source")
        fold = int(row["fold"])
        if not 0 <= fold < 5:
            raise ValueError("Internal OOF fold outside the fivefold contract")
        if group_folds.setdefault(row["split_group"], fold) != fold:
            raise ValueError("An identity group crosses internal folds")
    if {int(row["fold"]) for row in oof} != set(range(5)):
        raise ValueError("Internal OOF is missing an actual fold")
    target = np.asarray([float(row["target"]) for row in oof])
    predictions = np.asarray([float(row["prediction"]) for row in oof])
    actual_groups = np.asarray([row["split_group"] for row in oof])
    expected_folds = np.full(len(oof), -1, dtype=int)
    for fold, (_, validation) in enumerate(
        internal_splits(target, actual_groups, task["task_kind"], inputs["config"])
    ):
        expected_folds[validation] = fold
    if not np.array_equal(expected_folds, [int(row["fold"]) for row in oof]):
        raise ValueError("Internal fold assignments do not match the declared seed")
    metrics = training._metrics(target, predictions, task["task_kind"])
    for key, value in metrics.items():
        expected = result["internal_cv_selection_metrics"][key]
        if isinstance(value, float):
            if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("Saved internal CV metrics do not match OOF predictions")
        elif value != expected:
            raise ValueError("Saved internal CV metric metadata does not match")
    selected = [by_id[record_id] for record_id in ids]
    payload = training._load_feature_route(selected, features, result["selected_route"])
    if task.get("fixed_external_covariates") is not None:
        external = task["fixed_external_covariates"]
        context = _external_context_module(task, external)
        if (
            contract.get("fixed_external_covariates") != external
            or contract.get("external_context") != result.get("external_context")
            or result["external_context"]["implementation_sha256"] != digest(context.__file__)
        ):
            raise ValueError("External context provenance or implementation changed")
        payload = context.append_context(payload, selected, external)
    if payload["record_ids"] != ids or payload["feature_names"] != contract["feature_names"]:
        raise ValueError("Reload feature order/schema differs from training")
    if not np.asarray(payload["valid"], dtype=bool).all():
        raise ValueError("A saved training row no longer has valid features")
    model = joblib.load(result["artifacts"]["model"])
    X = np.asarray(payload["X"], dtype=np.float32)
    with threadpool_limits(limits=1):
        actual = training._predict(model, X, task["task_kind"])
        if not np.isfinite(actual).all():
            raise ValueError("Reloaded internal model has nonfinite predictions")
        if fitted is not None and not np.allclose(
            actual, training._predict(fitted, X, task["task_kind"]), rtol=1e-7, atol=1e-8
        ):
            raise ValueError("Internal model predictions changed after persistence")
    return {
        "status": "verified",
        "protocol": PROTOCOL,
        "actual_folds": 5,
        "record_count": len(ids),
        "group_count": len(group_folds),
        "artifact_hashes_verified": len(result["artifact_hashes"]),
        "independent_holdout_available": False,
        "model_reload_verified_on_training_host": True,
    }


def _validate_runner(config):
    template, output = Path(config["template_run"]).resolve(), Path(config["output_root"]).resolve()
    durable = Path(config["durable_root"]).resolve()
    for other in (template, durable, Path(config["parent_campaign_root"]).resolve()):
        if output == other or output in other.parents or other in output.parents:
            raise ValueError(
                "Internal output must be separate from template, parent and durable roots"
            )
    required = config.get("required_seeds", [42, 2026, 3407])
    if (
        len(required) != 3
        or len(set(required)) != 3
        or any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in required)
    ):
        raise ValueError("Declare exactly three distinct uint32 seeds")
    cfg = read_json(config["template_config"])
    settings = _settings(cfg["training"])
    if settings["search_backend"] != "optuna" or len(set(settings["models"])) < 7:
        raise ValueError("Production internal exploration requires Optuna and seven model families")
    workers = int(config.get("workers", 1))
    if not 1 <= workers <= 2 or workers * settings["n_jobs"] > 4:
        raise ValueError("Internal followup CPU budget is at most two processes and four threads")
    if not 0 < float(config.get("poll_seconds", 5)) <= 60:
        raise ValueError("poll_seconds must be in (0,60]")
    tasks = read_json(template / "task_manifest.json")["tasks"]
    task_ids = config.get("task_ids", [])
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or len(tasks) != len(task_ids)
        or set(task_ids) != {task["task_id"] for task in tasks}
        or any(
            task["status"] != "eligible" or not SAFE.fullmatch(task["task_id"]) for task in tasks
        )
    ):
        raise ValueError("Template must contain exactly its declared new eligible tasks")
    registry_path = Path(config.get("registry_path", template / "representation_methods.json"))
    _, methods = representation_registry(registry_path)
    available = [method for method in methods if method["route"] in settings["routes"]]
    if len(available) < 10:
        raise ValueError("At least ten genuine available internal methods are required")
    return cfg, tasks, available, registry_path


def _mirror_artifacts(root, durable, result):
    if result["status"] == "completed":
        for name, receipt in result["artifact_hashes"].items():
            source = Path(name).resolve()
            relative = source.relative_to(root)
            if digest(source) != receipt["sha256"]:
                raise ValueError("Internal artifact changed before durable copy")
            target = durable / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".writing")
            shutil.copy2(source, temporary)
            if digest(temporary) != receipt["sha256"]:
                raise ValueError("Durable internal artifact copy failed verification")
            temporary.replace(target)
    atomic_json(durable / "results/internal_cv" / (result["task_id"] + ".json"), result)


def write_internal_report(root, durable, receipt):
    """Describe seed variation of selected CV scores, never external performance."""
    selected, grouped = [], {}
    for entry in receipt["rounds"]:
        if entry["status"] != "completed":
            continue
        for path in sorted((Path(entry["run_dir"]) / "results/internal_cv").glob("*.json")):
            result = read_json(path)
            if result["status"] != "completed":
                continue
            row = {
                "task_id": result["task_id"],
                "endpoint": result["endpoint"],
                "seed": entry["seed"],
                "selected_route": result["selected_route"],
                "selected_model": result["selected_model"],
                "selected_params": result["selected_params"],
                "internal_cv_selection_metrics": result["internal_cv_selection_metrics"],
                "target_transform": result["target_transform"],
                "target_unit": result["target_unit"],
                "n_records": result["n_internal_cv_rows"],
                "n_groups": result["n_internal_cv_groups"],
                "result_sha256": digest(path),
            }
            selected.append(row)
            grouped.setdefault(result["task_id"], []).append(row)
    summaries = []
    for task_id, values in sorted(grouped.items()):
        metrics = {}
        for key in sorted(values[0]["internal_cv_selection_metrics"]):
            observations = [row["internal_cv_selection_metrics"][key] for row in values]
            if all(isinstance(value, (int, float)) for value in observations):
                metrics[key] = {
                    "mean": statistics.mean(observations),
                    "seed_std": statistics.stdev(observations) if len(observations) > 1 else None,
                }
        summaries.append(
            {
                "task_id": task_id,
                "seed_count": len(values),
                "all_required_seeds": set(receipt["required_seeds"]).issubset(
                    {row["seed"] for row in values}
                ),
                "selected_cv_score_summary": metrics,
            }
        )
    report = {
        "protocol": PROTOCOL,
        "independent_holdout_available": False,
        "selection_optimism_warning": WARNING,
        "selected_models": selected,
        "seed_summaries": summaries,
        "consensus_model_fit": "not_claimed",
    }
    atomic_json(Path(root) / "internal_performance_summary.json", report)
    atomic_json(Path(durable) / "internal_performance_summary.json", report)
    return {
        "path": "internal_performance_summary.json",
        "sha256": digest(Path(root) / "internal_performance_summary.json"),
        "protocol": PROTOCOL,
        "selected_models": len(selected),
    }


def _run_seed(run, settings, tasks, workers, poll_seconds, durable):
    pending, active, results = list(tasks), {}, []
    dispatch = {
        "status": "running",
        "pid": os.getpid(),
        "protocol": PROTOCOL,
        "config_sha256": digest(run / "config.json"),
        "tasks": {},
    }

    def save_dispatch():
        if digest(run / "config.json") != dispatch["config_sha256"]:
            raise ValueError("Internal seed configuration changed while running")
        atomic_json(run / "internal_dispatch.json", dispatch)
        atomic_json(durable / "internal_dispatch.json", dispatch)

    logs = run / "logs/internal_cv"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    threads = str(settings["training"]["n_jobs"])
    env.update(
        {
            key: threads
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    try:
        while pending or active:
            while pending and len(active) < workers:
                task = pending.pop(0)
                command = [
                    sys.executable,
                    str(run / "scripts/peptide_modeling_internal_cv.py"),
                    "--config",
                    str(run / "config.json"),
                    "--run-dir",
                    str(run),
                    "--train-task",
                    task["task_id"],
                ]
                with (logs / (task["task_id"] + ".log")).open("a", encoding="utf-8") as handle:
                    child = subprocess.Popen(
                        command,
                        cwd=run,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=os.name == "posix",
                    )
                active[task["task_id"]] = child
                dispatch["tasks"][task["task_id"]] = {
                    "status": "running",
                    "pid": child.pid,
                    "process_identity": process_identity(child.pid),
                    "argv": command,
                    "threads": int(threads),
                    "started_at": time.time(),
                }
                save_dispatch()
            for task_id, child in list(active.items()):
                if child.poll() is None:
                    continue
                result_path = _result_path(run, task_id)
                if child.returncode != 0 or not result_path.is_file():
                    raise RuntimeError(f"Internal task {task_id} failed, exit {child.returncode}")
                result = read_json(result_path)
                if result["status"] not in {"completed", "deferred"}:
                    raise RuntimeError("Internal task did not produce a valid terminal result")
                _mirror_artifacts(run, durable, result)
                results.append(result)
                dispatch["tasks"][task_id].update(
                    status=result["status"], exit_code=child.returncode, finished_at=time.time()
                )
                del active[task_id]
                save_dispatch()
            if pending or active:
                time.sleep(poll_seconds)
        dispatch["status"] = "completed"
        save_dispatch()
        return results
    except BaseException as exc:
        dispatch.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(run / "internal_dispatch.json", dispatch)
        atomic_json(durable / "internal_dispatch.json", dispatch)
        raise
    finally:
        for child in active.values():
            if child.poll() is None:
                child.terminate()
        for child in active.values():
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # A live handle is retained as an error; do not launch a replacement.
                    raise RuntimeError("Owned internal task did not release its process") from None


def run_internal_followup(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    cfg, tasks, methods, registry = _validate_runner(config)
    if os.name != "posix" or socket.gethostname() != cfg["resources"]["training_host"]:
        raise RuntimeError("Internal followup must execute on its declared Linux training host")
    root, durable = Path(config["output_root"]).resolve(), Path(config["durable_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    journal = root / "internal_campaign_status.json"
    if journal.exists():
        raise FileExistsError("Existing internal journal requires an explicit recovery decision")
    lock, token = root / "internal_followup.lock", uuid.uuid4().hex
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "token": token}, handle)
    receipt = {
        "schema": "peptide_tiny_internal_cv_followup_v1",
        "protocol": PROTOCOL,
        "selection_optimism_warning": WARNING,
        "status": "waiting_parent",
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "config_sha256": digest(config_path),
        "required_seeds": config.get("required_seeds", [42, 2026, 3407]),
        "task_ids": config["task_ids"],
        "rounds": [],
        "started_at": time.time(),
        "additional_twenty_hour_loop": False,
    }

    def save():
        if digest(config_path) != receipt["config_sha256"]:
            raise ValueError("Internal followup configuration changed")
        receipt["last_checked_at"] = time.time()
        atomic_json(journal, receipt)
        atomic_json(durable / journal.name, receipt)

    try:
        save()
        while True:
            evidence = parent_ready(config)
            if evidence is not None:
                break
            save()
            time.sleep(float(config.get("poll_seconds", 5)))
        receipt["parent_requirement_evidence"] = evidence
        template = Path(config["template_run"]).resolve()
        contract = template_contract(template, config["template_config"], registry)
        atomic_json(root / "template_contract.json", contract)
        for index, seed in enumerate(receipt["required_seeds"]):
            run = root / f"round_{index:03d}_seed_{seed}"
            settings = stage_round(template, run, cfg, seed, durable / run.name, contract, registry)
            entry = {
                "seed": seed,
                "run_dir": str(run),
                "status": "running",
                "config_sha256": digest(run / "config.json"),
            }
            receipt["rounds"].append(entry)
            receipt["status"] = "running"
            save()
            results = _run_seed(
                run,
                settings,
                tasks,
                int(config.get("workers", 1)),
                float(config.get("poll_seconds", 5)),
                durable / run.name,
            )
            features = portable_features(run)
            by_id = {task["task_id"]: task for task in tasks}
            verifications = []
            for result in results:
                if result["status"] == "completed":
                    task = by_id[result["task_id"]]
                    verification = verify_internal_task(
                        task, load_rows(run / task["table_path"]), features, run, result
                    )
                    verification.update(
                        task_id=task["task_id"],
                        source_table_sha256=digest(run / task["table_path"]),
                        result_sha256=digest(_result_path(run, task["task_id"])),
                    )
                    verifications.append(verification)
            seed_verification = {
                "status": "verified",
                "protocol": PROTOCOL,
                "independent_holdout_available": False,
                "source_task_manifest_sha256": digest(run / "task_manifest.json"),
                "completed_results": verifications,
            }
            atomic_json(run / "internal_verification.json", seed_verification)
            atomic_json(durable / run.name / "internal_verification.json", seed_verification)
            routes, families = set(), set()
            for result in results:
                if result["status"] == "completed":
                    for route, model in result["search"]["coverage_attempted"]:
                        routes.add(route)
                        families.add(model)
            actual_methods = [
                method["method_id"] for method in methods if method["route"] in routes
            ]
            entry.update(
                status="completed",
                task_states=dict(Counter(r["status"] for r in results)),
                attempted_methods=actual_methods,
                audit={
                    "status": "verified",
                    "protocol": PROTOCOL,
                    "verification_sha256": digest(run / "internal_verification.json"),
                },
                attempted_models=sorted(families),
                all_completed_results_have_five_folds=all(
                    result.get("actual_inner_folds") == 5
                    for result in results
                    if result["status"] == "completed"
                ),
                requirements_met=all(r["status"] == "completed" for r in results)
                and len(actual_methods) >= 10
                and len(families) >= 7,
            )
            save()
            receipt["internal_performance_report"] = write_internal_report(root, durable, receipt)
            save()
        receipt["status"] = (
            "completed"
            if all(entry["requirements_met"] for entry in receipt["rounds"])
            else "requirement_incomplete"
        )
        receipt["finished_at"] = time.time()
        save()
        # Per-task immutable models are already durable; archive all diagnostics once.
        receipt["archive"] = archive_run(root, durable, f"internal_cv_{int(time.time())}")
        save()
        return receipt
    except BaseException as exc:
        receipt.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(journal, receipt)
        atomic_json(durable / journal.name, receipt)
        raise
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            lock.unlink()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--train-task")
    args = parser.parse_args()
    if args.train_task:
        if args.run_dir is None:
            parser.error("--train-task requires --run-dir")
        config = read_json(args.config)
        if os.name == "posix" and int(config.get("resources", {}).get("nice", 0)) > 0:
            os.nice(int(config["resources"]["nice"]))
        task = next(
            task
            for task in read_json(args.run_dir / "task_manifest.json")["tasks"]
            if task["task_id"] == args.train_task
        )
        result = train_internal_task(
            task,
            load_rows(args.run_dir / task["table_path"]),
            portable_features(args.run_dir),
            args.run_dir,
            config["training"],
        )
        print(json.dumps({"task_id": task["task_id"], "status": result["status"]}), flush=True)
        raise SystemExit(0 if result["status"] in {"completed", "deferred"} else 2)
    result = run_internal_followup(args.config)
    print(json.dumps({"status": result["status"], "protocol": PROTOCOL}), flush=True)
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
