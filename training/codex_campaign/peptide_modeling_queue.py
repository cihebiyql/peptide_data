"""Wait for a frozen predecessor before starting a supplementary CPU batch."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from peptide_modeling_supervise import atomic_json, read_json


def predecessor_terminal(predecessor, expected_config_sha256):
    receipt_path = Path(predecessor) / "full_supervisor.json"
    if not receipt_path.exists():
        return False
    receipt = read_json(receipt_path)
    if receipt["config_sha256"] != expected_config_sha256:
        raise ValueError("Predecessor configuration changed; refusing a stale completion")
    if receipt["status"] not in {"completed", "failed"}:
        return False
    dispatch_path = Path(predecessor) / "dispatch/latest_full.json"
    if not dispatch_path.exists():
        return False
    dispatch = read_json(dispatch_path)
    return dispatch.get("status") in {"completed", "failed"} and all(
        task.get("status") != "running" for task in dispatch.get("tasks", {}).values()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--after-config-sha256", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    receipt_path = args.run_dir / "queue_status.json"
    receipt = {
        "status": "waiting",
        "predecessor": str(args.after_run),
        "predecessor_config_sha256": args.after_config_sha256,
        "cpu_budgets_overlap": False,
    }
    try:
        while not predecessor_terminal(args.after_run, args.after_config_sha256):
            receipt["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            atomic_json(receipt_path, receipt)
            time.sleep(30)
        receipt["status"] = "running"
        receipt["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(receipt_path, receipt)
        command = [
            sys.executable,
            str(Path(__file__).with_name("peptide_modeling_supervise.py")),
            "--run-dir",
            str(args.run_dir),
            "--config",
            str(args.config),
            "--workers",
            str(args.workers),
        ]
        code = subprocess.run(command, check=False).returncode
        receipt.update(status="completed" if code == 0 else "failed", exit_code=code)
        atomic_json(receipt_path, receipt)
        raise SystemExit(code)
    except Exception as exc:
        receipt.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    main()
