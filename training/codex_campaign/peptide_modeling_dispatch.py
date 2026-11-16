"""Run eligible modelling tasks in isolated, bounded Python processes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from peptide_modeling_pipeline import training_scopes


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def eligible_tasks(manifest):
    tasks = [t["task_id"] for t in manifest["tasks"] if t["status"] == "eligible"]
    if len(set(tasks)) != len(tasks) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", t) for t in tasks):
        raise ValueError("Task IDs must be unique safe filenames")
    return tasks


def available_memory_gib():
    """Linux MemAvailable includes reclaimable cache; other hosts use the static cap."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return None


def task_thread_plan(metadata, tasks, policy, default_threads, effective_cpu):
    """Allocate bounded threads from immutable task sizes, never from labels."""
    if policy is None:
        return dict.fromkeys(tasks, default_threads)
    if not isinstance(policy, list) or not policy:
        raise ValueError("task_cpu_by_rows must be a nonempty threshold list")
    thresholds = []
    for rule in policy:
        if set(rule) != {"min_records", "threads"}:
            raise ValueError("Task CPU rules require only min_records and threads")
        minimum, threads = int(rule["min_records"]), int(rule["threads"])
        if minimum < 0 or threads < 1 or threads > effective_cpu:
            raise ValueError("Task CPU rule exceeds the available CPU budget")
        thresholds.append((minimum, threads))
    minima = [minimum for minimum, _ in thresholds]
    if minima != sorted(set(minima), reverse=True):
        raise ValueError("Task CPU thresholds must be distinct and descending")
    if any("n_records" not in metadata[task] for task in tasks):
        raise ValueError("Task CPU allocation requires known record counts")
    allocation = {
        task: next(
            (
                threads
                for minimum, threads in thresholds
                if int(metadata[task]["n_records"]) >= minimum
            ),
            default_threads,
        )
        for task in tasks
    }
    if any(threads > effective_cpu for threads in allocation.values()):
        raise ValueError("Default task threads exceed the available CPU budget")
    return allocation


def resource_plan(run_dir, settings, manifest, workers):
    resources = settings.get("resources", {})
    threads = int(settings["training"].get("n_jobs", 4))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpu_budget = int(resources.get("cpu_budget", os.cpu_count() or 1))
    effective_cpu = min(cpu_budget, len(affinity)) if affinity is not None else cpu_budget
    policy = resources.get("task_cpu_by_rows")
    if threads < 1 or effective_cpu < 1 or (policy is None and workers * threads > effective_cpu):
        raise ValueError(
            f"CPU budget exceeded: {workers} workers * {threads} threads > {effective_cpu}"
        )
    if int(settings["training"].get("optuna_n_jobs", 1)) != 1:
        raise ValueError("Optuna trials must remain sequential within each task")
    budget = float(resources.get("memory_budget_gib", 72))
    reserve = float(resources.get("memory_reserve_gib", 16))
    if not math.isfinite(budget + reserve) or budget <= 0 or reserve < 0:
        raise ValueError("Memory budget must be positive and reserve nonnegative")
    dimensions = 0
    routes = settings["training"].get("routes", [])
    if routes:
        features = read_json(run_dir / "features/manifest.json")
        for route in routes:
            names = features["routes"][route] if isinstance(route, str) else route
            dimensions += sum(int(features["groups"][name]["shape"][1]) for name in names)
    metadata = {t["task_id"]: t for t in manifest["tasks"]}
    tasks = eligible_tasks(manifest)
    if any(
        int(metadata[t].get("n_records", 0)) < 0 or (dimensions and "n_records" not in metadata[t])
        for t in tasks
    ):
        raise ValueError("resource_budget: task record counts must be known and nonnegative")
    task_threads = task_thread_plan(metadata, tasks, policy, threads, effective_cpu)
    estimates = {
        task: 0.35 + int(metadata[task].get("n_records", 0)) * dimensions * 4 * 3 / 1024**3
        for task in tasks
    }
    oversized = {t: estimate for t, estimate in estimates.items() if estimate > budget}
    if oversized:
        raise ValueError(f"resource_budget: single-task memory exceeds {budget} GiB: {oversized}")
    tasks.sort(key=lambda task: (-estimates[task], task))
    return (
        tasks,
        estimates,
        {
            "cpu_budget": cpu_budget,
            "effective_cpu_budget": effective_cpu,
            "cpu_affinity": affinity,
            "memory_budget_gib": budget,
            "memory_reserve_gib": reserve,
            "total_route_dimensions": dimensions,
            "optuna_trials_parallel_per_task": 1,
            "task_cpu_threads": task_threads,
            "task_cpu_by_rows": policy,
            "memory_formula": "0.35 GiB + n_records * sum(route_dimensions) * 4 bytes * 3",
        },
        threads,
    )


def task_configuration(settings, parent_config, directory, parent_sha, task_id, threads):
    """Create an immutable child config only when heterogeneous allocation is enabled."""
    if settings.get("resources", {}).get("task_cpu_by_rows") is None:
        return parent_config
    path = directory / "task_configs" / parent_sha[:16] / (task_id + ".json")
    configuration = copy.deepcopy(settings)
    configuration["training"]["n_jobs"] = threads
    configuration["resources"]["threads_per_task"] = threads
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != configuration:
            raise ValueError("Immutable task configuration differs from the allocation plan")
    else:
        write_json(path, configuration)
    return path


def mirror_status(run_dir, settings, profile, receipt, task_id=None):
    """Copy only small status artifacts; model archival remains the parent's responsibility."""
    root = settings.get("resources", {}).get("remote_durable_root")
    if not root:
        return
    try:
        destination = Path(root) / "status" / profile
        destination.mkdir(parents=True, exist_ok=True)
        files = (
            [
                run_dir / "results" / profile / (task_id + "_" + scope + ".json")
                for scope in receipt["expected_scopes"]
            ]
            if task_id
            else [
                run_dir / name
                for name in (
                    "REPORT.md",
                    "run_summary.json",
                    "verification.json",
                    "model_inventory.tsv",
                )
            ]
        )
        for source in files:
            if source.is_file() and source.stat().st_size <= 8 * 1024**2:
                temporary = destination / (source.name + ".writing")
                temporary.write_bytes(source.read_bytes())
                temporary.replace(destination / source.name)
        write_json(destination / "dispatch_receipt.json", receipt)
    except OSError as exc:
        receipt.setdefault("durable_status_errors", []).append(str(exc)[:500])
        receipt["durable_status_errors"] = receipt["durable_status_errors"][-10:]


def finish_reports(run_dir, remote_root, profile):
    from peptide_modeling_pipeline import report_stage
    from peptide_modeling_summarize import summarize_run

    report_stage(run_dir)
    return summarize_run(run_dir, remote_root or str(run_dir), profile=profile)


def stop_owned(active):
    """Each POSIX child owns its session; never target an unrelated process group."""
    for process, _ in active.values():
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
    for process, _ in active.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)


def dispatch(
    run_dir,
    config,
    profile="full",
    workers=4,
    remote_root=None,
    pipeline=None,
    finalize=True,
    poll_seconds=0.25,
):
    if not 1 <= workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    run_dir, config = Path(run_dir).resolve(), Path(config).resolve()
    pipeline = Path(pipeline or Path(__file__).with_name("peptide_modeling_pipeline.py")).resolve()
    if not pipeline.is_file():
        raise FileNotFoundError(pipeline)
    settings = read_json(config)
    scopes = training_scopes(settings, profile)
    tasks, estimates, budgets, threads = resource_plan(
        run_dir, settings, read_json(run_dir / "task_manifest.json"), workers
    )
    directory, logs = run_dir / "dispatch", run_dir / "logs/dispatch"
    directory.mkdir(exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    lock = directory / "active.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Dispatcher lock exists: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "token": token}, handle)
    receipt = {
        "schema": "peptide_process_dispatch_v1",
        "status": "running",
        "exit_code": None,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "config": str(config),
        "config_sha256": sha(config),
        "task_manifest_sha256": sha(run_dir / "task_manifest.json"),
        "profile": profile,
        "workers": workers,
        "threads_per_task": threads
        if settings.get("resources", {}).get("task_cpu_by_rows") is None
        else None,
        "default_threads_per_task": threads,
        "cpu_allocation": "task_record_count_policy"
        if settings.get("resources", {}).get("task_cpu_by_rows") is not None
        else "fixed_threads_per_task",
        "expected_scopes": list(scopes),
        "cohort_search_budgets": {
            scope: training.get("search_budget", training.get("n_trials"))
            for scope, training in scopes.items()
        },
        "cohort_execution": "sequential_scopes_within_each_task_process",
        "peak_active": 0,
        "tasks": {},
        "resource_budget": budgets,
        "estimated_task_memory_gib": estimates,
        "peak_estimated_memory_gib": 0,
        "peak_reserved_cpu_threads": 0,
        "launch_order": [],
        "memory_wait_events": 0,
        "cpu_wait_events": 0,
        "resource_wait_events": 0,
    }
    receipt_path = directory / f"{profile}_{time.time_ns()}_{os.getpid()}.json"
    latest = directory / f"latest_{profile}.json"

    def save():
        write_json(receipt_path, receipt)
        write_json(latest, receipt)

    pending, active = deque(tasks), {}
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    task_threads = budgets["task_cpu_threads"]
    try:
        # Publish once under the dispatcher lock before isolated children start.
        # Otherwise all first-time workers can race over the shared .writing file.
        write_json(run_dir / "execution_config.json", settings)
        save()
        last_wait_saved = 0
        while pending or active:
            while pending and len(active) < workers:
                if sha(config) != receipt["config_sha256"]:
                    raise RuntimeError("Configuration changed during dispatch")
                active_memory = sum(entry["estimated_memory_gib"] for _, entry in active.values())
                active_cpu = sum(entry["threads"] for _, entry in active.values())
                available = available_memory_gib()

                def memory_fits(task, active_memory=active_memory, available=available):
                    return active_memory + estimates[task] <= budgets["memory_budget_gib"] and (
                        available is None
                        or estimates[task] + budgets["memory_reserve_gib"] <= available
                    )

                task_id = next(
                    (
                        t
                        for t in pending
                        if memory_fits(t)
                        and active_cpu + task_threads[t] <= budgets["effective_cpu_budget"]
                    ),
                    None,
                )
                if task_id is None:
                    receipt["resource_wait_events"] += 1
                    receipt["memory_wait_events"] += int(not any(memory_fits(t) for t in pending))
                    receipt["cpu_wait_events"] += int(
                        not any(
                            active_cpu + task_threads[t] <= budgets["effective_cpu_budget"]
                            for t in pending
                        )
                    )
                    receipt["resource_wait"] = {
                        "at": time.time(),
                        "available_memory_gib": available,
                        "active_estimated_gib": active_memory,
                        "active_cpu_threads": active_cpu,
                        "available_cpu_threads": budgets["effective_cpu_budget"] - active_cpu,
                        "pending_tasks": len(pending),
                    }
                    if time.time() - last_wait_saved >= 5:
                        save()
                        last_wait_saved = time.time()
                    break
                pending.remove(task_id)
                receipt.pop("resource_wait", None)
                threads_for_task = task_threads[task_id]
                child_config = task_configuration(
                    settings, config, directory, receipt["config_sha256"], task_id, threads_for_task
                )
                child_config_sha = sha(child_config)
                env = os.environ.copy()
                env.update({name: str(threads_for_task) for name in names})
                command = [
                    sys.executable,
                    str(pipeline),
                    "--run-dir",
                    str(run_dir),
                    "--config",
                    str(child_config),
                    "--stage",
                    "train",
                    "--profile",
                    profile,
                    "--only-task",
                    task_id,
                ]
                log_path = logs / (task_id + ".log")
                with log_path.open("a", encoding="utf-8") as output:
                    process = subprocess.Popen(
                        command,
                        cwd=run_dir,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        env=env,
                        start_new_session=os.name == "posix",
                    )
                entry = {
                    "status": "running",
                    "pid": process.pid,
                    "started_at": time.time(),
                    "argv": command,
                    "log": str(log_path),
                    "estimated_memory_gib": estimates[task_id],
                    "threads": threads_for_task,
                    "config": str(child_config),
                    "config_sha256": child_config_sha,
                }
                receipt["tasks"][task_id] = entry
                active[task_id] = (process, entry)
                receipt["peak_active"] = max(receipt["peak_active"], len(active))
                receipt["peak_estimated_memory_gib"] = max(
                    receipt["peak_estimated_memory_gib"], active_memory + estimates[task_id]
                )
                receipt["peak_reserved_cpu_threads"] = max(
                    receipt["peak_reserved_cpu_threads"], active_cpu + threads_for_task
                )
                receipt["launch_order"].append(task_id)
                save()
            for task_id, (process, entry) in list(active.items()):
                if sha(entry["config"]) != entry["config_sha256"]:
                    raise RuntimeError("Task configuration changed during dispatch")
                code = process.poll()
                if code is None:
                    continue
                entry.update(exit_code=code, finished_at=time.time(), status="failed")
                scope_results = {}
                for scope in scopes:
                    result_file = run_dir / "results" / profile / (task_id + "_" + scope + ".json")
                    if result_file.exists():
                        result = read_json(result_file)
                        scope_results[scope] = {
                            "status": result.get("status"),
                            "reason": result.get("reason", ""),
                        }
                entry["scopes"] = scope_results
                if (
                    code == 0
                    and len(scope_results) == len(scopes)
                    and all(
                        result["status"] in {"completed", "deferred"}
                        for result in scope_results.values()
                    )
                ):
                    entry["status"] = (
                        "completed"
                        if any(r["status"] == "completed" for r in scope_results.values())
                        else "deferred"
                    )
                    entry["reason"] = "all_expected_scopes_terminal"
                if entry["status"] == "failed":
                    entry["reason"] = "child_failed_or_missing_terminal_result"
                del active[task_id]
                mirror_status(run_dir, settings, profile, receipt, task_id)
                save()
            if active or pending:
                time.sleep(poll_seconds)
        failed = any(t["status"] == "failed" for t in receipt["tasks"].values())
        manifest_hash = sha(run_dir / "task_manifest.json")
        if (
            sha(config) != receipt["config_sha256"]
            or manifest_hash != receipt["task_manifest_sha256"]
            or any(sha(t["config"]) != t["config_sha256"] for t in receipt["tasks"].values())
        ):
            raise RuntimeError("Dispatch inputs changed during execution")
        if finalize:
            if failed:
                from peptide_modeling_pipeline import report_stage

                report_stage(run_dir)
            else:
                receipt["verification"] = finish_reports(run_dir, remote_root, profile)
        receipt.update(status="failed" if failed else "completed", exit_code=int(failed))
    except BaseException as exc:
        stop_owned(active)
        receipt.update(status="failed", exit_code=1, reason=f"{type(exc).__name__}:{exc}")
        for process, entry in active.values():
            entry.update(status="interrupted", exit_code=process.poll(), finished_at=time.time())
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        receipt["finished_at"] = time.time()
        mirror_status(run_dir, settings, profile, receipt)
        try:
            save()
        finally:
            if read_json(lock).get("token") == token:
                lock.unlink()
    return receipt


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--remote-root")
    args = parser.parse_args()
    result = dispatch(args.run_dir, args.config, args.profile, args.workers, args.remote_root)
    print(json.dumps({k: result[k] for k in ("status", "exit_code", "peak_active")}), flush=True)
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()
