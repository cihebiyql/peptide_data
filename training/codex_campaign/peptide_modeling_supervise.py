"""Supervise one frozen node23 dispatcher and archive verified outputs durably."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def mirror_terminal(run_dir, durable, mirrored):
    """Copy only terminal, hash-verified artifacts; never copy a live SQLite DB."""
    copied = 0
    for result_path in sorted((run_dir / "results/full").glob("*.json")):
        result = read_json(result_path)
        if result.get("status") not in {"completed", "deferred"}:
            continue
        receipt_key = (str(result_path), digest(result_path))
        if receipt_key in mirrored:
            continue
        for name, receipt in result.get("artifact_hashes", {}).items():
            source = Path(name).resolve()
            relative = source.relative_to(run_dir)
            if digest(source) != receipt["sha256"]:
                raise ValueError(f"Artifact SHA mismatch: {relative}")
            destination = durable / "live" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".writing")
            shutil.copy2(source, temporary)
            if digest(temporary) != receipt["sha256"]:
                raise ValueError(f"Copied artifact SHA mismatch: {relative}")
            temporary.replace(destination)
            copied += 1
        destination = durable / "live" / result_path.relative_to(run_dir)
        atomic_json(destination, result)
        mirrored.add(receipt_key)
    return copied


def archive_run(run_dir, durable, label):
    destination = durable / "archives" / (label + ".tar.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".writing")
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        archive.add(run_dir, arcname=run_dir.name, recursive=True)
    receipt = {
        "path": str(destination),
        "bytes": temporary.stat().st_size,
        "sha256": digest(temporary),
    }
    # Stream every archived member to check gzip/tar integrity before publishing.
    with tarfile.open(temporary, "r:gz") as archive:
        for member in archive:
            if member.isfile():
                with archive.extractfile(member) as handle:
                    while handle.read(1024 * 1024):
                        pass
    temporary.replace(destination)
    return receipt


def supervise(run_dir, config, workers=32):
    run_dir, config = Path(run_dir).resolve(), Path(config).resolve()
    settings = read_json(config)
    resources = settings["resources"]
    if socket.gethostname() != resources["training_host"]:
        raise RuntimeError("Supervisor must run on the declared training host")
    durable = Path(resources["remote_durable_root"]).resolve()
    if durable == run_dir or run_dir in durable.parents or durable in run_dir.parents:
        raise ValueError("Scratch and durable roots must be separate")
    if settings["training"]["inner_folds"] != 5:
        raise ValueError("Expanded experiment requires strict five-fold CV")
    if settings["training"]["search_backend"] != "optuna":
        raise ValueError("Expanded experiment requires Optuna")
    durable.mkdir(parents=True, exist_ok=True)
    dispatcher = Path(__file__).with_name("peptide_modeling_dispatch.py")
    command = [
        sys.executable,
        str(dispatcher),
        "--run-dir",
        str(run_dir),
        "--config",
        str(config),
        "--profile",
        "full",
        "--workers",
        str(workers),
        "--remote-root",
        str(run_dir),
    ]
    label = time.strftime("expanded5_%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    receipt = {
        "schema": "peptide_supervision_v1",
        "status": "running",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "argv": command,
        "config_sha256": digest(config),
        "task_manifest_sha256": digest(run_dir / "task_manifest.json"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts_mirrored": 0,
        "sync_errors": [],
    }
    lock = run_dir / "supervisor.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "label": label}, handle)
    child = None

    def save():
        atomic_json(run_dir / "full_supervisor.json", receipt)
        atomic_json(durable / "full_supervisor.json", receipt)

    try:
        save()
        with (run_dir / "full_dispatch.log").open("a", encoding="utf-8") as output:
            child = subprocess.Popen(command, cwd=run_dir, stdout=output, stderr=subprocess.STDOUT)
            receipt["dispatcher_pid"] = child.pid
            save()
            mirrored = set()
            last_sync = 0
            while child.poll() is None:
                if time.monotonic() - last_sync >= 30:
                    try:
                        receipt["artifacts_mirrored"] += mirror_terminal(run_dir, durable, mirrored)
                    except OSError as exc:
                        receipt["sync_errors"] = (receipt["sync_errors"] + [str(exc)])[-10:]
                    receipt["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    save()
                    last_sync = time.monotonic()
                time.sleep(2)
        receipt["dispatcher_exit_code"] = child.returncode
        receipt["artifacts_mirrored"] += mirror_terminal(run_dir, durable, mirrored)
        verification = run_dir / "verification.json"
        if child.returncode == 0:
            result = read_json(verification)
            task_counts = result.get("scientific_tasks_any_coverage", result["primary_tasks"])
            if result.get("status") != "verified" or task_counts["pending"]:
                raise RuntimeError("Dispatcher ended without complete verification")
            receipt["verification_sha256"] = digest(verification)
        receipt["status"] = "archiving"
        save()
        receipt["archive"] = archive_run(run_dir, durable, label)
        receipt["status"] = "completed" if child.returncode == 0 else "failed"
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        return receipt
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=60)
        receipt.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        save()
        raise
    finally:
        if read_json(lock).get("label") == label:
            lock.unlink()


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    receipt = supervise(args.run_dir, args.config, args.workers)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    raise SystemExit(0 if receipt["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
