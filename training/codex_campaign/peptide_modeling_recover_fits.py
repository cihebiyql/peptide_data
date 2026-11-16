"""Recover failed fit tasks without changing their original data or held-out split.

CLI: python peptide_modeling_recover_fits.py --config recovery.json

The recovery JSON declares source_runs=[{run_dir, config_sha256}], template_run,
template_config (default template_run/template_config.json), output_run,
durable_root, workers (default32), and poll_seconds (default5). The prepared
template supplies the repaired runtime, the same feature bank and every source
task table. Source supervisors and dispatchers must actually release their CPUs
before recovery starts. Completed and scientifically deferred tasks are retained
in their original runs; only failed/interrupted task IDs receive a new run.

materialize_config(training, output, durable) is deterministic. Its JSON SHA can
be calculated by materialized_config_sha256 before the controller starts, so a
following campaign can pin the recovery's expected configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

from peptide_modeling_campaign import predecessor_terminal, process_identity
from peptide_modeling_supervise import atomic_json, digest, read_json

SAFE_TASK = re.compile(r"[A-Za-z0-9_.-]+")
OPTIONAL_INPUTS = (
    "feature_rows_index.tsv",
    "final_feature_bank_verification.json",
    "representation_methods.json",
)


def materialize_config(training, output, durable):
    def target_name(value):
        # Permit the parent to calculate a Linux deployment hash from Windows.
        if isinstance(value, str) and value.startswith("/"):
            return PurePosixPath(value).as_posix()
        return str(Path(value).resolve())

    configuration = copy.deepcopy(training)
    configuration["run_dir"] = target_name(output)
    resources = configuration.setdefault("resources", {})
    resources["remote_training_scratch"] = target_name(output)
    resources["remote_durable_root"] = target_name(durable)
    return configuration


def materialized_config_sha256(training, output, durable):
    configuration = materialize_config(training, output, durable)
    return hashlib.sha256(
        json.dumps(configuration, indent=2, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_run_config(path, configuration):
    """Use identical UTF8/LF bytes on Windows and Linux for the pinned SHA."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_bytes(json.dumps(configuration, indent=2, ensure_ascii=False).encode("utf-8"))
    temporary.replace(path)


def _regular_files(directory):
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Recovery inputs cannot contain symlinks: {path}")
        if path.is_file():
            yield path


def _relative_table(task):
    relative = Path(task["table_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("task_tables",):
        raise ValueError("Task table must stay inside task_tables")
    return relative


def _training_contract(configuration):
    return {
        "training": configuration["training"],
        "coverage_cohorts": configuration.get("coverage_cohorts", {}),
        "task_trial_budgets": configuration.get("task_trial_budgets", []),
    }


def validate_recovery(configuration):
    template = Path(configuration["template_run"]).resolve()
    output = Path(configuration["output_run"]).resolve()
    durable = Path(configuration["durable_root"]).resolve()
    if template == output or template in output.parents or output in template.parents:
        raise ValueError("Template and recovery output must be separate")
    if durable == output or durable in output.parents or output in durable.parents:
        raise ValueError("Scratch and durable roots must be separate")
    template_config = Path(configuration.get("template_config", template / "template_config.json"))
    training = read_json(template_config)
    settings = training["training"]
    if settings.get("seed") != 42 or float(settings.get("test_size", 0)) != 0.2:
        raise ValueError("Recovery must preserve source seed42 and test_size0.2")
    if settings.get("inner_folds") != 5 or settings.get("search_backend") != "optuna":
        raise ValueError("Recovery requires strict five-fold Optuna training")
    workers = int(configuration.get("workers", 32))
    threads = int(settings.get("n_jobs", 2))
    if workers < 1 or threads < 1 or workers * threads > 64:
        raise ValueError("Recovery CPU budget exceeds64")
    if not 0 < float(configuration.get("poll_seconds", 5)) <= 60:
        raise ValueError("poll_seconds must be in (0,60]")
    sources = configuration.get("source_runs", [])
    if not sources:
        raise ValueError("At least one frozen source run is required")
    roots = [Path(source["run_dir"]).resolve() for source in sources]
    if len(set(roots)) != len(roots):
        raise ValueError("Source run roots must be distinct")
    for source, root in zip(sources, roots, strict=True):
        if output == root or root in output.parents or output in root.parents:
            raise ValueError("Recovery output cannot overlap a frozen source run")
        if digest(root / "config.json") != source["config_sha256"]:
            raise ValueError("Source configuration hash mismatch")
        if _training_contract(read_json(root / "config.json")) != _training_contract(training):
            raise ValueError("Recovery changes the frozen source training parameters")
    return template, output, durable, template_config, training


def failed_tasks(source):
    """Inspect terminal, frozen evidence; a missing observation is never a restart signal."""
    root = Path(source["run_dir"]).resolve()
    if digest(root / "config.json") != source["config_sha256"]:
        raise ValueError("Source configuration hash mismatch")
    dispatch = read_json(root / "dispatch/latest_full.json")
    if dispatch.get("config_sha256") != source["config_sha256"]:
        raise ValueError("Source dispatch configuration hash mismatch")
    if dispatch.get("status") not in {"completed", "failed"}:
        raise ValueError("Source dispatcher is not terminal")
    tasks = dispatch.get("tasks", {})
    if any(task.get("status") == "running" for task in tasks.values()):
        raise ValueError("A source task is still running")
    selected = {
        task_id
        for task_id, task in tasks.items()
        if task.get("status") in {"failed", "interrupted"}
    }
    for path in sorted((root / "results/full").glob("*.json")):
        result = read_json(path)
        infrastructure_deferral = result.get("status") == "deferred" and any(
            str(reason).startswith("feature_route_unavailable:")
            and any(
                token in str(reason) for token in ("FileNotFoundError:", "artifact hash mismatch")
            )
            for reason in result.get("excluded_routes", {}).values()
        )
        if result.get("status") != "failed" and not infrastructure_deferral:
            continue
        task_id = result["task_id"]
        if tasks.get(task_id, {}).get("status") == "completed" and not infrastructure_deferral:
            raise ValueError("Source completed task contradicts a failed scope result")
        selected.add(task_id)
    if any(not SAFE_TASK.fullmatch(task_id) for task_id in selected):
        raise ValueError("Unsafe source task ID")
    return selected


def select_failures(source_runs):
    selected = {}
    for source in source_runs:
        for task_id in sorted(failed_tasks(source)):
            selected.setdefault(task_id, []).append(source)
    return selected


def template_snapshot(template, template_config):
    files = {"task_manifest.json": digest(template / "task_manifest.json")}
    for directory in ("features", "scripts", "task_tables"):
        for path in _regular_files(template / directory):
            files[path.relative_to(template).as_posix()] = digest(path)
    for name in OPTIONAL_INPUTS:
        if (template / name).is_file():
            files[name] = digest(template / name)
    if "features/manifest.json" not in files:
        raise FileNotFoundError("Prepared source feature bank is required")
    if "scripts/peptide_modeling_supervise.py" not in files:
        raise FileNotFoundError("Prepared repaired supervisor runtime is required")
    return {"files": files, "template_config_sha256": digest(template_config)}


def _copy_verified(source, destination, expected):
    if destination.exists():
        raise FileExistsError(destination)
    if digest(source) != expected:
        raise ValueError(f"Frozen input changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if digest(destination) != expected:
        raise ValueError(f"Copied input hash mismatch: {destination}")


def prepare_recovery(template, destination, selected, snapshot):
    """Stage a new subset and byte-identical originating outer splits."""
    destination = Path(destination)
    if digest(template / "task_manifest.json") != snapshot["files"]["task_manifest.json"]:
        raise ValueError("Recovery template manifest changed")
    manifest = read_json(template / "task_manifest.json")
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if set(selected) - set(by_id):
        raise ValueError("Failed source task is missing from the prepared template")
    for task_id in selected:
        if by_id[task_id]["status"] != "eligible":
            raise ValueError("A failed source task is not eligible in the recovery template")
    for name, expected in snapshot["files"].items():
        if name == "task_manifest.json" or name.startswith("task_tables/"):
            continue
        _copy_verified(template / name, destination / name, expected)
    source_features = read_json(template / "features/manifest.json")["input_fingerprint"]
    provenance = {}
    for task_id, sources in sorted(selected.items()):
        task = by_id[task_id]
        table = _relative_table(task)
        expected = snapshot["files"][table.as_posix()]
        split_bytes = None
        origins = []
        for source in sources:
            root = Path(source["run_dir"]).resolve()
            source_manifest = read_json(root / "task_manifest.json")
            original = next(t for t in source_manifest["tasks"] if t["task_id"] == task_id)
            scientific_fields = (
                "endpoint",
                "task_kind",
                "target_transform",
                "inverse_transform",
                "task_definition",
                "n_records",
                "n_groups",
                "unit",
            )
            if any(task.get(key) != original.get(key) for key in scientific_fields):
                raise ValueError("Recovery scientific task contract differs from its source")
            original_table = root / _relative_table(original)
            if digest(original_table) != expected:
                raise ValueError("Recovery task table differs from its frozen source")
            if read_json(root / "features/manifest.json")["input_fingerprint"] != source_features:
                raise ValueError("Recovery feature bank differs from its frozen source")
            split = root / "splits" / task_id / "group_assignment.json"
            payload = split.read_bytes()
            if split_bytes is not None and payload != split_bytes:
                raise ValueError("Duplicated failed task has incompatible source holdouts")
            split_bytes = payload
            origins.append(
                {
                    "run_dir": str(root),
                    "config_sha256": source["config_sha256"],
                    "table_sha256": expected,
                    "outer_split_sha256": digest(split),
                }
            )
        _copy_verified(template / table, destination / table, expected)
        split_out = destination / "splits" / task_id / "group_assignment.json"
        split_out.parent.mkdir(parents=True, exist_ok=True)
        if split_out.exists():
            raise FileExistsError(split_out)
        split_out.write_bytes(split_bytes)
        provenance[task_id] = origins
    output_manifest = copy.deepcopy(manifest)
    output_manifest["tasks"] = [by_id[task_id] for task_id in sorted(selected)]
    output_manifest["source_template_counts"] = manifest.get("counts", {})
    output_manifest["counts"] = {
        "eligible_tasks": len(selected),
        "eligible_table_records": sum(int(by_id[t].get("n_records", 0)) for t in selected),
        "endpoints_with_eligible_tasks": len({by_id[t]["endpoint"] for t in selected}),
    }
    output_manifest["recovery_policy"] = "failed_tasks_only;unchanged_tables_and_outer_splits"
    atomic_json(destination / "task_manifest.json", output_manifest)
    atomic_json(destination / "recovery_source_evidence.json", provenance)
    return output_manifest


def no_recovery_markers(output, durable, config_sha256):
    """Only certify resource release; do not invent models or validation metrics."""
    marker = {
        "schema": "peptide_no_recovery_resource_marker_v1",
        "status": "completed",
        "config_sha256": config_sha256,
        "model_training_performed": False,
        "reason": "no_recovery_needed",
        "purpose": "resource_release_only",
        "tasks": {},
    }
    for root in (output, durable):
        atomic_json(root / "full_supervisor.json", marker)
        atomic_json(root / "dispatch/latest_full.json", marker)


def run_recovery(config_path):
    config_path = Path(config_path).resolve()
    configuration = read_json(config_path)
    template, output, durable, template_config, training = validate_recovery(configuration)
    if socket.gethostname() != training["resources"]["training_host"]:
        raise RuntimeError("Recovery must run on the declared training host")
    output.mkdir(parents=True, exist_ok=False)
    durable.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    lock = output / "recovery.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "token": token}, handle)
    run_configuration = materialize_config(training, output, durable)
    write_run_config(output / "config.json", run_configuration)
    receipt = {
        "schema": "peptide_failed_fit_recovery_v1",
        "status": "waiting_sources",
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "host": socket.gethostname(),
        "controller_config_sha256": digest(config_path),
        "config_sha256": digest(output / "config.json"),
        "source_runs": configuration["source_runs"],
        "started_at": time.time(),
        "model_training_performed": False,
    }
    child = None

    def save(check_contract=True):
        if check_contract and digest(config_path) != receipt["controller_config_sha256"]:
            raise ValueError("Recovery controller configuration changed")
        receipt["last_checked_at"] = time.time()
        atomic_json(output / "recovery_status.json", receipt)
        atomic_json(durable / "recovery_status.json", receipt)

    try:
        snapshot = template_snapshot(template, template_config)
        atomic_json(output / "recovery_template_contract.json", snapshot)
        save()
        while not all(predecessor_terminal(source) for source in configuration["source_runs"]):
            for source in configuration["source_runs"]:
                if digest(Path(source["run_dir"]) / "config.json") != source["config_sha256"]:
                    raise ValueError("Source configuration changed while recovery waited")
            save()
            time.sleep(float(configuration.get("poll_seconds", 5)))
        if digest(template_config) != snapshot["template_config_sha256"]:
            raise ValueError("Recovery template configuration changed while waiting")
        selected = select_failures(configuration["source_runs"])
        receipt["failed_task_ids"] = sorted(selected)
        receipt["source_dispatch_sha256"] = {
            source["run_dir"]: digest(Path(source["run_dir"]) / "dispatch/latest_full.json")
            for source in configuration["source_runs"]
        }
        prepare_recovery(template, output, selected, snapshot)
        if not selected:
            receipt.update(status="no_recovery_needed", finished_at=time.time())
            no_recovery_markers(output, durable, receipt["config_sha256"])
            save()
            return receipt
        receipt.update(status="running", model_training_performed=True)
        command = [
            sys.executable,
            str(output / "scripts/peptide_modeling_supervise.py"),
            "--run-dir",
            str(output),
            "--config",
            str(output / "config.json"),
            "--workers",
            str(configuration.get("workers", 32)),
        ]
        with (output / "recovery_supervisor.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                command,
                cwd=output,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt.update(
                supervisor_pid=child.pid,
                supervisor_identity=process_identity(child.pid),
                argv=command,
            )
            save()
            while child.poll() is None:
                save()
                time.sleep(float(configuration.get("poll_seconds", 5)))
        receipt["supervisor_exit_code"] = child.returncode
        if child.returncode != 0:
            raise RuntimeError(f"Recovery supervisor failed with exit{child.returncode}")
        supervisor = read_json(output / "full_supervisor.json")
        dispatch = read_json(output / "dispatch/latest_full.json")
        if any(r.get("status") != "completed" for r in (supervisor, dispatch)):
            raise RuntimeError("Recovery did not produce complete terminal receipts")
        if any(r.get("config_sha256") != receipt["config_sha256"] for r in (supervisor, dispatch)):
            raise RuntimeError("Recovery terminal configuration contract mismatch")
        receipt.update(status="completed", finished_at=time.time())
        save()
        return receipt
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                receipt["owned_supervisor_still_live"] = True
        receipt.update(
            status="failed", reason=f"{type(exc).__name__}:{exc}", finished_at=time.time()
        )
        save(check_contract=False)
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
    args = parser.parse_args()
    result = run_recovery(args.config)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["status"] in {"completed", "no_recovery_needed"} else 1)


if __name__ == "__main__":
    main()
