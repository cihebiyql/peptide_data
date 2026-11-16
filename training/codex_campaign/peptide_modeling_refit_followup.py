"""Recover optional full-data ElasticNet refits after the audited parent releases CPUs.

This never reruns selection or CV. The original five-fold evaluation and held-out
models remain immutable. Only a saved estimator's solver iteration ceiling may
increase; convergence tolerance, statistical parameters and cohort stay fixed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from peptide_modeling_campaign import process_identity
from peptide_modeling_supervise import atomic_json, digest, read_json

SAFE = re.compile(r"[A-Za-z0-9_.-]+")
LIMITS = (48000, 192000)
REQUIRED = (42, 2026, 3407)


def checked_file(root, name):
    path = Path(name)
    if not path.is_absolute():
        path = Path(root) / path
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked source file: " + str(path))
    path = path.resolve()
    path.relative_to(Path(root).resolve())
    return path


def verify_files(files):
    for name, expected in files.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError("Frozen source changed: " + name)


def _artifact(result, root, name):
    path = checked_file(root, result["artifacts"][name])
    expected = result["artifact_hashes"][str(path)]["sha256"]
    if digest(path) != expected:
        raise ValueError("Original evaluated artifact hash mismatch: " + name)
    return path


def freeze_candidate(root, result_path, seed, round_receipt):
    root, result_path = Path(root).resolve(), Path(result_path).resolve()
    result = read_json(result_path)
    if (
        result.get("status") != "completed"
        or result.get("refit_status") != "failed"
        or result.get("selected_model") != "elasticnet"
        or result.get("task_kind") != "regression"
        or result.get("actual_inner_folds") != 5
        or result.get("requested_inner_folds") != 5
        or result.get("search", {}).get("backend") != "optuna"
        or not str(result.get("refit_failure_reason", "")).startswith("ConvergenceWarning:")
    ):
        raise ValueError("Recovery requires a completed five-fold Optuna ElasticNet evaluation")
    task, scope = result["task_id"], result["scope_label"]
    if not SAFE.fullmatch(task) or not SAFE.fullmatch(scope):
        raise ValueError("Unsafe source task or scope")
    config_path = checked_file(root, "config.json")
    if digest(config_path) != round_receipt["config_sha256"]:
        raise ValueError("Source round config hash mismatch")
    if read_json(config_path)["training"]["seed"] != seed:
        raise ValueError("Source round belongs to another random seed")
    source_input = read_json(checked_file(root, "campaign_input_contract.json"))
    if source_input["config_sha256"] != round_receipt["config_sha256"]:
        raise ValueError("Source round input/config contract mismatch")
    pinned = source_input["template_contract"]["files"]
    manifest_path = checked_file(root, "task_manifest.json")
    manifest = read_json(manifest_path)
    source_task = next(t for t in manifest["tasks"] if t["task_id"] == task)
    table_path = checked_file(root, source_task["table_path"])
    verification_path = checked_file(root, "verification.json")
    verification = read_json(verification_path)
    if (
        digest(verification_path) != round_receipt["audit"]["verification_sha256"]
        or verification.get("status") != "verified"
        or verification.get("source_task_manifest_sha256") != digest(manifest_path)
    ):
        raise ValueError("Source round prediction/artifact verification is not pinned")
    artifacts = {name: _artifact(result, root, name) for name in result["artifacts"]}
    model_contract = read_json(artifacts["model_contract"])
    input_contract = read_json(artifacts["input_contract"])
    if (
        model_contract["input_hash"] != result["input_hash"]
        or input_contract["input_hash"] != result["input_hash"]
        or model_contract["selected_params"] != result["selected_params"]
        or model_contract["selected_route"] != result["selected_route"]
        or model_contract["selected_model"] != "elasticnet"
        or model_contract["task"] != input_contract["task"]
        or model_contract["task"]["task_id"] != task
        or model_contract["common_record_ids"] == []
    ):
        raise ValueError("Selected estimator, cohort and source contracts differ")
    files = {}
    paths = [
        result_path,
        config_path,
        manifest_path,
        table_path,
        verification_path,
        checked_file(root, "campaign_input_contract.json"),
        checked_file(root, "features/manifest.json"),
        *artifacts.values(),
    ]
    source_scripts = sorted((root / "scripts").rglob("*.py"))
    if not source_scripts:
        raise ValueError("Original runtime scripts are absent")
    for path in [manifest_path, table_path, root / "features/manifest.json", *source_scripts]:
        relative = path.relative_to(root).as_posix()
        if relative not in pinned or digest(path) != pinned[relative]:
            raise ValueError("Original template input hash mismatch: " + relative)
    training_path = checked_file(root, "scripts/peptide_modeling_train.py")
    if input_contract["implementation_sha256"] != digest(training_path):
        raise ValueError("Original training implementation is unavailable")
    for path in [*paths, *source_scripts]:
        files[str(path)] = digest(path)
    return {
        "source_run": str(root),
        "seed": seed,
        "task_id": task,
        "scope": scope,
        "result_path": str(result_path),
        "table_path": str(table_path),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "files": files,
        "selected_route": result["selected_route"],
        "selected_params": result["selected_params"],
        "common_record_ids": model_contract["common_record_ids"],
        "target_transform": model_contract["target_transform"],
        "target_bounds": model_contract.get("target_bounds"),
        "original_five_fold_evaluation_unchanged": True,
    }


def discover_candidates(config):
    journal = read_json(Path(config["parent_campaign_root"]) / "campaign_status.json")
    if journal["config_sha256"] != config["parent_config_sha256"]:
        raise ValueError("Parent config does not match the pinned SHA")
    candidates = []
    for seed in REQUIRED:
        rounds = [r for r in journal["rounds"] if r.get("seed") == seed]
        if len(rounds) != 1 or rounds[0].get("status") != "completed":
            raise ValueError("Pinning recovery requires each audited required seed to exist")
        source = rounds[0]
        if source.get("audit", {}).get("status") != "verified":
            raise ValueError("Source seed has not passed its independent audit")
        root = Path(source["run_dir"]).resolve()
        for path in sorted((root / "results/full").glob("*.json")):
            result = read_json(path)
            if result.get("status") == "completed" and result.get("refit_status") == "failed":
                candidates.append(freeze_candidate(root, path, seed, source))
    if len(candidates) > int(config.get("max_tasks", 20)):
        raise ValueError("Optional refit recovery exceeds the declared finite task cap")
    return candidates


def parameter_signature(estimator):
    """Compare all leaf parameters while excluding only solver iteration limits."""
    values = {}
    for key, value in estimator.get_params(deep=True).items():
        if key.split("__")[-1] == "max_iter" or hasattr(value, "get_params"):
            continue
        if isinstance(value, (list, tuple)) and any(
            hasattr(item, "get_params")
            or (isinstance(item, tuple) and any(hasattr(part, "get_params") for part in item))
            for item in value
        ):
            continue  # Pipeline step estimators are expanded in the remaining deep leaf keys.
        values[key] = repr(value)
    return values


def extend_iterations(estimator, limit):
    from sklearn.base import clone
    from sklearn.linear_model import ElasticNet

    if limit not in LIMITS:
        raise ValueError("Only the two declared finite solver ceilings are permitted")
    changes = {}
    for prefix, component in [("", estimator), *estimator.get_params(deep=True).items()]:
        if isinstance(component, ElasticNet):
            key = prefix + "__max_iter" if prefix else "max_iter"
            if not 0 < int(component.max_iter) < limit:
                raise ValueError("Recovery cannot decrease or repeat a solver ceiling")
            changes[key] = limit
    if len(changes) != 1:
        raise ValueError("Exactly one saved ElasticNet solver is required")
    recovered = clone(estimator).set_params(**changes)
    if parameter_signature(recovered) != parameter_signature(estimator):
        raise ValueError("Recovery changes a statistical parameter")
    return recovered, changes


def run_worker(spec_path):
    """Execute in a fresh interpreter with the original run's scripts first on sys.path."""
    import joblib
    import numpy as np
    import peptide_modeling_train as training
    from peptide_modeling_features import _verify_artifacts
    from peptide_modeling_pipeline import load_rows, portable_features
    from sklearn.exceptions import ConvergenceWarning

    spec = read_json(spec_path)
    candidate, output = spec["candidate"], Path(spec["output_dir"])
    original = Path(candidate["source_run"]).resolve()
    resolved_output = output.resolve()
    if (
        resolved_output == original
        or original in resolved_output.parents
        or resolved_output in original.parents
    ):
        raise ValueError("Worker output cannot overlap an original run")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "solver_limit": spec["solver_limit"]}
    try:
        verify_files(candidate["files"])
        if Path(training.__file__).resolve() != original / "scripts/peptide_modeling_train.py":
            raise ValueError("Worker did not import the exact original training runtime")
        contract = read_json(candidate["artifacts"]["model_contract"])
        inputs = read_json(candidate["artifacts"]["input_contract"])
        expected = {
            "common_record_ids": candidate["common_record_ids"],
            "selected_route": candidate["selected_route"],
            "selected_params": candidate["selected_params"],
            "target_transform": candidate["target_transform"],
            "target_bounds": candidate["target_bounds"],
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("Worker specification changes the frozen cohort or model contract")
        actual_versions = {
            "numpy": training.np.__version__,
            "scikit_learn": training.sklearn.__version__,
            "joblib": training.joblib.__version__,
        }
        if any(contract["versions"][key] != value for key, value in actual_versions.items()):
            raise ValueError("Worker library versions differ from the evaluated model")
        rows = load_rows(candidate["table_path"])
        features = portable_features(original)
        _verify_artifacts(features)
        payload = training._load_feature_route(rows, features, candidate["selected_route"])
        evidence = inputs["feature_hashes"][candidate["selected_route"]]
        if (
            payload["feature_names"] != contract["feature_names"]
            or training._digest(payload["feature_names"]) != training._digest(evidence["names"])
            or training.hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest()
            != evidence["matrix_sha256"]
            or training.hashlib.sha256(payload["valid"].tobytes()).hexdigest()
            != evidence["mask_sha256"]
        ):
            raise ValueError("Original feature matrix or valid mask differs from evaluation")
        ids = [r["record_id"] for r in rows]
        common = candidate["common_record_ids"]
        if len(ids) != len(set(ids)) or len(common) != len(set(common)):
            raise ValueError("Source cohort contains duplicate record IDs")
        indices = np.asarray([ids.index(record) for record in common])
        if not payload["valid"][indices].all():
            raise ValueError("A frozen common-cohort feature row became invalid")
        X, y = payload["X"][indices], np.asarray([float(rows[i]["target"]) for i in indices])
        selected = joblib.load(candidate["artifacts"]["model"])
        estimator, overrides = extend_iterations(selected, int(spec["solver_limit"]))
        fitted = training._fit_model(estimator, X, y, 1)
        model = output / "refit_model.joblib"
        training._save_checked_model(fitted, model, X, "regression", 1)
        verify_files(candidate["files"])
        recovered_contract = {
            **contract,
            "model_fit_scope": "all_common_after_original_evaluation",
            "optional_refit_fit_scope": "all_common_after_evaluation",
            "refit_status": "completed",
            "deployment_ready": True,
            "model_path": "refit_model.joblib",
            "model_sha256": digest(model),
            "source_evaluation_model_sha256": digest(candidate["artifacts"]["model"]),
            "source_evaluation_contract_sha256": digest(candidate["artifacts"]["model_contract"]),
            "solver_iteration_overrides": overrides,
            "new_cv_or_search_run": False,
            "performance_claim_from_full_refit": False,
            "original_five_fold_evaluation_unchanged": True,
        }
        atomic_json(output / "model_contract.json", recovered_contract)
        result.update(
            status="completed",
            model_sha256=digest(model),
            model_contract_sha256=digest(output / "model_contract.json"),
            solver_iteration_overrides=overrides,
            source_files_sha256=candidate["files"],
            common_record_ids=common,
            selected_route=candidate["selected_route"],
            selected_params=candidate["selected_params"],
            target_transform=candidate["target_transform"],
            target_bounds=candidate["target_bounds"],
            leaf_parameters_except_max_iter=parameter_signature(selected),
            actual_runtime_versions=actual_versions,
            reload_predictions_verified=True,
            training_rows=len(common),
            original_five_fold_evaluation_unchanged=True,
            model_selection_repeated=False,
            performance_estimated_from_refit=False,
        )
    except Exception as exc:
        result.update(
            status="non_converged" if isinstance(exc, ConvergenceWarning) else "failed",
            reason=f"{type(exc).__name__}:{exc}",
        )
    atomic_json(output / "attempt_result.json", result)
    return result


def stop_child(child):
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def run_attempt(candidate, limit, directory, timeout, popen=subprocess.Popen):
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    spec = directory.with_suffix(".json")
    atomic_json(spec, {"candidate": candidate, "solver_limit": limit, "output_dir": str(directory)})
    bootstrap = (
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        str(Path(candidate["source_run"]) / "scripts"),
        str(Path(__file__).resolve()),
        "--worker-spec",
        str(spec),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    with directory.with_suffix(".log").open("x", encoding="utf-8") as log:
        child = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=os.name == "posix",
        )
        try:
            exit_code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_child(child)
            return {"status": "timeout", "solver_limit": limit, "pid": child.pid}
        except BaseException:
            stop_child(child)
            raise
    result_path = directory / "attempt_result.json"
    if exit_code or not result_path.is_file():
        return {
            "status": "failed",
            "solver_limit": limit,
            "exit_code": exit_code,
            "reason": "worker_exit_or_missing_receipt",
        }
    return read_json(result_path)


def recover_one(candidate, output, seconds, attempt=run_attempt):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline, attempts = time.monotonic() + seconds, []
    selected = None
    for limit in LIMITS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            attempts.append({"status": "timeout", "solver_limit": limit})
            break
        result = attempt(candidate, limit, output / f"attempt_{limit}", remaining)
        attempts.append(result)
        if result["status"] == "completed":
            selected = output / f"attempt_{limit}" / "refit_model.joblib"
            if digest(selected) != result["model_sha256"]:
                raise ValueError("Recovered model hash differs from worker receipt")
            contract = selected.with_name("model_contract.json")
            if digest(contract) != result["model_contract_sha256"]:
                raise ValueError("Recovered model contract hash differs from worker receipt")
            shutil.copy2(selected, output / "refit_model.joblib")
            shutil.copy2(contract, output / "model_contract.json")
            break
        if result["status"] != "non_converged":
            break
    verify_files(candidate["files"])
    receipt = {
        "status": "completed" if selected else "failed",
        "candidate": candidate,
        "attempts": attempts,
        "original_evaluation_unchanged": True,
        "new_cv_or_search_run": False,
        "performance_claim_from_full_refit": False,
    }
    if selected:
        receipt["model_sha256"] = digest(output / "refit_model.joblib")
        receipt["model_contract_sha256"] = digest(output / "model_contract.json")
    atomic_json(output / "refit_recovery.json", receipt)
    return receipt


def mirror_files(source, destination):
    for path in Path(source).rglob("*"):
        if not path.is_file():
            continue
        target = Path(destination) / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".writing")
        shutil.copy2(path, temporary)
        if digest(path) != digest(temporary):
            raise ValueError("Durable recovery copy hash mismatch")
        temporary.replace(target)


def run_followup(config_path):
    from peptide_modeling_followup_seeds import parent_ready

    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if os.name != "posix" or socket.gethostname() != config.get("training_host", "node23"):
        raise ValueError("Refit controller must run on its declared Linux host")
    if tuple(config.get("required_seeds", REQUIRED)) != REQUIRED:
        raise ValueError("The original three required seeds must be retained")
    if tuple(config.get("solver_iterations", LIMITS)) != LIMITS:
        raise ValueError("Only the two bounded solver iteration ceilings are permitted")
    seconds = float(config.get("max_seconds_per_model", 120))
    if not 0 < seconds <= 120 or not 0 < int(config.get("max_tasks", 20)) <= 20:
        raise ValueError("Recovery exceeds its time or task-count cap")
    poll = float(config.get("poll_seconds", 10))
    if not 0 < poll <= 60:
        raise ValueError("Polling interval must stay within one minute")
    root, durable = Path(config["output_root"]).resolve(), Path(config["durable_root"]).resolve()
    source_root = Path(config["parent_campaign_root"]).resolve()
    template = config_path.parent
    roots = [root, durable, source_root, template]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError("Template, source, output and durable roots must not overlap")
    if (root / "refit_status.json").exists():
        raise FileExistsError("Existing recovery requires a separate explicit recovery decision")
    candidates = discover_candidates(config)
    runtime = {
        str(config_path): digest(config_path),
        str(Path(__file__).resolve()): digest(__file__),
    }
    runtime.update({str(p): digest(p) for p in Path(__file__).parent.glob("peptide_modeling_*.py")})
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / "refit.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    journal = {
        "status": "waiting_parent",
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "config_sha256": digest(config_path),
        "runtime_sha256": runtime,
        "candidates": candidates,
        "recoveries": [],
        "thread_budget": 1,
        "additional_twenty_hour_loop": False,
        "new_cv_or_search_run": False,
    }

    def save():
        atomic_json(root / "refit_status.json", journal)
        atomic_json(durable / "refit_status.json", journal)

    try:
        save()
        while (evidence := parent_ready(config)) is None:
            time.sleep(poll)
            verify_files(runtime)
            save()
        journal["parent_requirement_evidence"] = evidence
        os.nice(10)
        for candidate in candidates:
            verify_files(runtime)
            verify_files(candidate["files"])
            journal["status"] = "running"
            save()
            target = (
                root
                / f"seed_{candidate['seed']}"
                / (candidate["task_id"] + "_" + candidate["scope"])
            )
            receipt = recover_one(candidate, target, seconds)
            for name, expected in candidate["files"].items():
                path = Path(name)
                if path.suffix == ".py":
                    relative = path.relative_to(Path(candidate["source_run"]) / "scripts")
                    copied = target / "source_runtime/scripts" / relative
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, copied)
                    if digest(copied) != expected:
                        raise ValueError("Copied original runtime differs from frozen source")
            mirror_files(target, durable / target.relative_to(root))
            journal["recoveries"].append(
                {
                    "seed": candidate["seed"],
                    "task_id": candidate["task_id"],
                    "scope": candidate["scope"],
                    "status": receipt["status"],
                    "receipt": str(target / "refit_recovery.json"),
                    "receipt_sha256": digest(target / "refit_recovery.json"),
                }
            )
            save()
        journal["status"] = (
            "completed"
            if all(r["status"] == "completed" for r in journal["recoveries"])
            else "requirement_incomplete"
        )
        journal["finished_at"] = time.time()
        verify_files(runtime)
        save()
        return journal
    except BaseException as exc:
        journal.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        save()
        raise


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--worker-spec")
    args = parser.parse_args()
    if args.worker_spec:
        result = run_worker(args.worker_spec)
        raise SystemExit(0 if result["status"] in {"completed", "non_converged"} else 2)
    run_followup(args.config)


if __name__ == "__main__":
    main()
