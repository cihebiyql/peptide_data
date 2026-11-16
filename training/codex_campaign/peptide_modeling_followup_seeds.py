"""Run only newly admitted tasks for three seeds after the twenty-hour campaign.

CLI: python peptide_modeling_followup_seeds.py --config followup.json

Required configuration: template_run, template_config, output_root, durable_root,
task_ids, parent_campaign_root, parent_config_sha256. Optional required_seeds
defaults to [42, 2026, 3407], workers to32, poll_seconds to5, and registry_path to
template_run/representation_methods.json. The template manifest must contain
exactly task_ids, all eligible. This followup runs exactly the requested seeds;
the twenty-hour requirement is proved by its predecessor, never repeated here.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path

from peptide_modeling_campaign import (
    audit_round,
    predecessor_terminal,
    process_identity,
    stage_round,
    template_contract,
    validate_campaign,
)
from peptide_modeling_campaign_report import build_report
from peptide_modeling_supervise import atomic_json, digest, read_json

SAFE_TASK = re.compile(r"[A-Za-z0-9_.-]+")


def parent_ready(config, proc_root=Path("/proc")):
    """Require productive work, seed audits and actual release of owned processes."""
    parent = Path(config["parent_campaign_root"])
    path = parent / "campaign_status.json"
    if not path.is_file():
        return None
    receipt = read_json(path)
    if receipt.get("config_sha256") != config["parent_config_sha256"]:
        raise ValueError("Parent campaign configuration does not match its pinned SHA")
    state = receipt.get("status")
    if state == "failed":
        raise RuntimeError("Parent campaign failed before followup prerequisites were established")
    if state not in {"completed", "requirement_incomplete"}:
        return None
    productive = float(receipt.get("productive_wall_seconds", 0))
    if not math.isfinite(productive) or productive < 20 * 3600:
        raise RuntimeError("Terminal parent has not proved twenty productive hours")
    required = set(config.get("required_seeds", [42, 2026, 3407]))
    rounds = receipt.get("rounds", [])
    verified = {
        entry["seed"]
        for entry in rounds
        if entry.get("status") == "completed" and entry.get("audit", {}).get("status") == "verified"
    }
    if not required.issubset(verified):
        raise RuntimeError("Terminal parent is missing a required verified seed")
    global_method_evidence = {}
    if config.get("method_coverage_requirement_scope", "own_followup") == "parent_campaign":
        for seed in sorted(required):
            audits = [
                entry["audit"]
                for entry in rounds
                if entry.get("seed") == seed and entry.get("status") == "completed"
            ]
            proof = next(
                (
                    audit
                    for audit in audits
                    if len(set(audit.get("attempted_independent_methods", []))) >= 10
                    and len(set(audit.get("attempted_model_families", []))) >= 7
                ),
                None,
            )
            if proof is None:
                raise RuntimeError("Parent does not prove the global method/model requirement")
            global_method_evidence[str(seed)] = {
                "methods": proof["attempted_independent_methods"],
                "model_families": proof["attempted_model_families"],
            }
    if not rounds or rounds[-1].get("status") != "completed":
        raise RuntimeError("Parent last round did not terminate successfully")
    pid = receipt.get("pid")
    if pid:
        current = process_identity(pid, proc_root)
        expected = receipt.get("process_identity", {})
        if current is not None and current["state"] != "Z":
            # Without a recorded start identity, an existing PID is not release evidence.
            if not expected or current["start_ticks"] == expected.get("start_ticks"):
                return None
    last = rounds[-1]
    if not predecessor_terminal(
        {
            "run_dir": last["run_dir"],
            "config_sha256": last["config_sha256"],
            "require_success": True,
        },
        proc_root,
    ):
        return None
    return {
        "campaign_root": str(parent),
        "campaign_journal_sha256": digest(path),
        "config_sha256": receipt["config_sha256"],
        "status": state,
        "productive_wall_seconds": productive,
        "verified_required_seeds": sorted(required),
        "parent_owned_processes_released": True,
        "global_method_model_evidence_by_seed": global_method_evidence,
    }


def validate_followup(config):
    """Validate the bounded subset without creating an additional twenty-hour loop."""
    proxy = {
        **config,
        "campaign_root": config["output_root"],
        "minimum_productive_hours": 20,
        "max_rounds": len(config.get("required_seeds", [42, 2026, 3407])),
    }
    training, registry, methods = validate_campaign(proxy)
    if config.get("method_coverage_requirement_scope", "own_followup") not in {
        "own_followup",
        "parent_campaign",
    }:
        raise ValueError("Unknown representation coverage requirement scope")
    root = Path(config["output_root"]).resolve()
    parent = Path(config["parent_campaign_root"]).resolve()
    if root == parent or root in parent.parents or parent in root.parents:
        raise ValueError("Followup output cannot overlap the parent campaign")
    task_ids = config.get("task_ids", [])
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(
            not isinstance(task_id, str) or not SAFE_TASK.fullmatch(task_id) for task_id in task_ids
        )
    ):
        raise ValueError("Declare unique safe task_ids for only the newly admitted tasks")
    manifest = read_json(Path(config["template_run"]) / "task_manifest.json")
    tasks = manifest.get("tasks", [])
    if (
        len(tasks) != len(task_ids)
        or {task["task_id"] for task in tasks} != set(task_ids)
        or any(task["status"] != "eligible" for task in tasks)
    ):
        raise ValueError("Followup template must contain exactly its declared eligible task_ids")
    if not 0 < float(config.get("poll_seconds", 5)) <= 60:
        raise ValueError("poll_seconds must be in (0,60]")
    return training, registry, methods, sorted({task["endpoint"] for task in tasks})


def verify_runtime(training):
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Live parent process verification requires Linux /proc")
    if socket.gethostname() != training["resources"]["training_host"]:
        raise RuntimeError("Followup must run on the declared training host")


def run_followup(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    training, registry_path, methods, endpoints = validate_followup(config)
    verify_runtime(training)
    root, durable = Path(config["output_root"]).resolve(), Path(config["durable_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    journal = root / "campaign_status.json"
    if journal.exists():
        raise FileExistsError("Existing followup journal requires an explicit recovery decision")
    token = uuid.uuid4().hex
    lock = root / "followup.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "token": token, "hostname": socket.gethostname()}, handle)
    receipt = {
        "schema": "peptide_bounded_seed_followup_v1",
        "status": "waiting_parent",
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "host": socket.gethostname(),
        "config_sha256": digest(config_path),
        "started_at": time.time(),
        "required_seeds": config.get("required_seeds", [42, 2026, 3407]),
        "task_ids": config["task_ids"],
        "rounds": [],
        "time_requirement_source": "verified_parent_campaign_only",
        "additional_twenty_hour_loop": False,
        "completion_scope": "declared_new_tasks_only_not_the_entire_project",
        "method_coverage_requirement_scope": config.get(
            "method_coverage_requirement_scope", "own_followup"
        ),
    }

    def save(check_config=True):
        if check_config and digest(config_path) != receipt["config_sha256"]:
            raise ValueError("Followup configuration changed during execution")
        receipt["last_checked_at"] = time.time()
        atomic_json(journal, receipt)
        atomic_json(durable / "campaign_status.json", receipt)

    child = None
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
        contract = template_contract(template, config["template_config"], registry_path)
        atomic_json(root / "template_contract.json", contract)
        atomic_json(durable / "template_contract.json", contract)
        for index, seed in enumerate(receipt["required_seeds"]):
            label = f"round_{index:03d}_seed_{seed}"
            run = root / label
            settings = stage_round(
                template, run, training, seed, durable / label, contract, registry_path
            )
            entry = {
                "index": index,
                "seed": seed,
                "run_dir": str(run),
                "status": "starting",
                "config_sha256": digest(run / "config.json"),
            }
            receipt["rounds"].append(entry)
            command = [
                sys.executable,
                str(run / "scripts/peptide_modeling_supervise.py"),
                "--run-dir",
                str(run),
                "--config",
                str(run / "config.json"),
                "--workers",
                str(config.get("workers", 32)),
            ]
            with (run / "followup_supervisor.log").open("a", encoding="utf-8") as output:
                child = subprocess.Popen(
                    command,
                    cwd=run,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                entry.update(
                    pid=child.pid,
                    process_identity=process_identity(child.pid),
                    status="running",
                    started_at=time.time(),
                    argv=command,
                )
                receipt["status"] = "running"
                save()
                while child.poll() is None:
                    save()
                    time.sleep(float(config.get("poll_seconds", 5)))
            entry.update(exit_code=child.returncode, finished_at=time.time())
            if child.returncode != 0:
                entry["status"] = "failed"
                raise RuntimeError(f"Followup seed {seed} supervisor failed")
            audit = audit_round(run, settings, endpoints)
            completed_tasks = {
                item["task_id"]
                for items in audit["endpoint_completed_scopes"].values()
                for item in items
            }
            audit["missing_completed_task_ids"] = sorted(set(config["task_ids"]) - completed_tasks)
            audit["attempted_independent_methods"] = [
                method["method_id"]
                for method in methods
                if method["route"] in audit["attempted_routes"]
            ]
            audit["method_model_coverage_met"] = (
                len(audit["attempted_independent_methods"]) >= 10
                or bool(evidence.get("global_method_model_evidence_by_seed", {}))
            ) and len(audit["attempted_model_families"]) >= 7
            audit["own_method_count"] = len(audit["attempted_independent_methods"])
            audit["global_method_requirement_basis"] = (
                "verified_parent_campaign"
                if evidence.get("global_method_model_evidence_by_seed", {})
                else "this_followup"
            )
            entry.update(status="completed", audit=audit)
            save()
            receipt["cross_seed_report"] = build_report(root, durable)
            save()
        complete = all(
            not entry["audit"]["missing_completed_task_ids"]
            and entry["audit"]["method_model_coverage_met"]
            for entry in receipt["rounds"]
        )
        receipt.update(
            status="completed" if complete else "requirement_incomplete", finished_at=time.time()
        )
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
        save(check_config=False)
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
    receipt = run_followup(args.config)
    print(
        json.dumps({"status": receipt["status"], "required_seeds": receipt["required_seeds"]}),
        flush=True,
    )
    raise SystemExit(0 if receipt["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
