"""Explicitly restore empty followup controllers after an audited parent recovery.

Prepare a pinned plan while controllers are stopped, then execute it only after
the same parent configuration has a live running controller. No model output is
removed, no template is changed, and no partially executed followup is resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

PARENT_FAILURE = (
    "RuntimeError:Parent campaign failed before followup prerequisites were established"
)
JOURNAL_LOCKS = {
    "campaign_status.json": "followup.lock",
    "internal_campaign_status.json": "internal_followup.lock",
    "qualitative_status.json": "qualitative.lock",
    "refit_status.json": "refit.lock",
}
CONTROLLERS = {
    "peptide_modeling_followup_seeds.py",
    "peptide_modeling_internal_cv.py",
    "peptide_modeling_qualitative_cyto.py",
    "peptide_modeling_refit_followup.py",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_pin(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked pinned file: " + str(path))
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError("Pinned file changed: " + str(path))
    return str(path.resolve()), actual


def verify_files(files):
    for path, expected in files.items():
        file_pin(path, expected)


def identity(pid, proc_root=Path("/proc")):
    try:
        p = Path(proc_root) / str(int(pid))
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": int(pid),
            "state": fields[0],
            "start_ticks": int(fields[19]),
            "argv": [x.decode() for x in (p / "cmdline").read_bytes().split(b"\0") if x],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def old_released(expected, proc_root):
    actual = identity(expected["pid"], proc_root)
    if actual is None or actual["state"] == "Z":
        return "exited"
    if actual["start_ticks"] != expected["start_ticks"]:
        return "pid_reused"
    raise ValueError("Original controller is still live")


def matching_controllers(argv, proc_root):
    matches = []
    for p in Path(proc_root).iterdir():
        if not p.name.isdigit():
            continue
        try:
            actual = identity(int(p.name), proc_root)
        except PermissionError:
            continue
        if actual and actual["state"] != "Z" and actual["argv"] == argv:
            matches.append(actual)
    return matches


def validate_argv(argv, config_path):
    if (
        not isinstance(argv, list)
        or len(argv) != 4
        or Path(argv[0]).name not in {"python", "python3", "python3.10", "python3.11", "python.exe"}
        or Path(argv[1]).name not in CONTROLLERS
        or argv[2] != "--config"
        or Path(argv[3]).resolve() != Path(config_path).resolve()
    ):
        raise ValueError("Controller argv is not the original bounded Python/config command")


def empty_failure(journal, output):
    if journal.get("status") != "failed" or journal.get("reason") != PARENT_FAILURE:
        raise ValueError("Only the exact parent-triggered failure may be restored")
    for key in ("rounds", "recoveries"):
        if key in journal and journal[key] != []:
            raise ValueError("Partial followup work must not be restarted")
    for pattern in ("round_*", "seed_*", "models", "results", "training", "internal_dispatch.json"):
        if list(Path(output).glob(pattern)):
            raise ValueError("Followup output contains launched work: " + pattern)


def find_logs(output, argv, explicit=()):
    """Keep old logs untouched; discover bounded direct-parent traceback logs too."""
    logs = {Path(p).resolve() for p in explicit}
    logs.update(p.resolve() for p in Path(output).rglob("*.log"))
    needles = [str(argv[1]).encode(), str(argv[3]).encode()]
    for p in Path(output).parent.glob("peptide_modeling*.log"):
        if p.is_symlink() or p.stat().st_size > 1024 * 1024:
            continue
        raw = p.read_bytes()
        if any(needle in raw for needle in needles):
            logs.add(p.resolve())
    return sorted(logs)


def pin_entry(source, proc_root=Path("/proc")):
    output = Path(source["output_root"]).resolve()
    journal_path = Path(source["journal_path"]).resolve()
    if journal_path.parent != output or journal_path.name not in JOURNAL_LOCKS:
        raise ValueError("Unexpected followup journal location")
    journal = read(journal_path)
    empty_failure(journal, output)
    expected = source["owned_process_identity"]
    if (
        journal.get("pid") != expected["pid"]
        or journal.get("process_identity", {}).get("start_ticks") != expected["start_ticks"]
    ):
        raise ValueError("Failed journal no longer belongs to the pinned original controller")
    released = old_released(expected, proc_root)
    argv = expected["argv"]
    config_path = Path(source["config_path"]).resolve()
    validate_argv(argv, config_path)
    if matching_controllers(argv, proc_root):
        raise ValueError("A matching controller is already live")
    files = dict([file_pin(config_path, source["config_sha256"])])
    if journal.get("config_sha256") != source["config_sha256"]:
        raise ValueError("Failed controller configuration does not match registry")
    config = read(config_path)
    if Path(config["output_root"]).resolve() != output:
        raise ValueError("Config output differs from pinned journal")
    template = Path(config.get("template_run", config_path.parent)).resolve()
    durable = Path(config["durable_root"]).resolve()
    if any(
        a == b or a in b.parents or b in a.parents
        for a, b in ((output, template), (output, durable))
    ):
        raise ValueError("Source template and output roots must stay separate")
    for path in Path(argv[1]).parent.glob("peptide_modeling_*.py"):
        files.update([file_pin(path)])
    files.update([file_pin(argv[1])])
    for key in ("manifest_path",):
        if source.get(key):
            files.update([file_pin(source[key], source["manifest_sha256"])])
    if source.get("template_config"):
        value = source["template_config"]
        files.update([file_pin(value["path"], value["sha256"])])
    for task in source.get("tasks", []):
        files.update([file_pin(template / task["table_relative"], task["table_sha256"])])
    for key in ("runtime_sha256", "input_files_sha256"):
        for name, expected_sha in journal.get(key, {}).items():
            path = Path(name) if key == "runtime_sha256" else template / name
            files.update([file_pin(path, expected_sha)])
    for candidate in journal.get("candidates", []):
        for name, expected_sha in candidate["files"].items():
            files.update([file_pin(name, expected_sha)])
    for name in ("features/manifest.json", "representation_methods.json"):
        p = template / name
        if p.is_file():
            files.update([file_pin(p)])
    durable_journal = durable / journal_path.name
    journal_sha = sha(journal_path)
    if file_pin(durable_journal)[1] != journal_sha:
        raise ValueError("Durable failed journal differs from local failed proof")
    logs = find_logs(output, argv, source.get("logs", []))
    log_files = dict(file_pin(p) for p in logs)
    lock = output / JOURNAL_LOCKS[journal_path.name]
    retired = [journal_path, durable_journal]
    if lock.exists():
        if lock.stat().st_size:
            owner = read(lock)
            if owner.get("pid") != expected["pid"]:
                raise ValueError("Stale lock does not belong to the failed controller")
        retired.append(lock)
    mutable = dict(file_pin(p) for p in retired)
    return {
        "controller_id": source.get("controller_id", output.name),
        "output_root": str(output),
        "durable_root": str(durable),
        "journal_path": str(journal_path),
        "config_path": str(config_path),
        "config_sha256": source["config_sha256"],
        "argv": argv,
        "old_identity": expected,
        "old_pid_disposition": released,
        "source_files_sha256": files,
        "failed_files_sha256": mutable,
        "log_files_sha256": log_files,
        "retire_paths": [str(p) for p in retired],
        "parent_campaign_root": config["parent_campaign_root"],
        "parent_config_sha256": config["parent_config_sha256"],
        "threads": int(read(config["template_config"])["training"]["n_jobs"])
        if config.get("template_config")
        else int(config.get("training", {}).get("n_jobs", 1)),
    }


def prepare(
    registry_path, refit_launch_path, output, proc_root=Path("/proc"), source_precheck_path=None
):
    output = Path(output)
    if (output / "restore_plan.json").exists():
        raise FileExistsError("A restoration plan already exists")
    registry = read(registry_path)
    sources = list(registry["controllers"])
    launch = read(refit_launch_path)
    config_path = launch["argv"][3]
    config = read(config_path)
    failed = Path(config["output_root"]) / "refit_status.json"
    sources.append(
        {
            "controller_id": Path(config["output_root"]).name,
            "output_root": config["output_root"],
            "journal_path": str(failed),
            "config_path": config_path,
            "config_sha256": read(failed)["config_sha256"],
            "owned_process_identity": launch["process_identity"],
            "logs": [launch["log"]],
        }
    )
    if len(sources) != 12 or len({x["output_root"] for x in sources}) != 12:
        raise ValueError("The explicit restoration scope is eleven supplements and one refit")
    entries = [pin_entry(source, proc_root) for source in sources]
    if source_precheck_path is not None:
        earlier = read(source_precheck_path)
        expected = {entry["controller_id"]: entry for entry in earlier["controllers"]}
        if set(expected) != {entry["controller_id"] for entry in entries}:
            raise ValueError("Read-only source precheck has a different controller scope")
        for entry in entries:
            for key in ("source_files_sha256", "failed_files_sha256", "log_files_sha256"):
                if entry[key] != expected[entry["controller_id"]][key]:
                    raise ValueError("Controller source changed after read-only precheck")
    parents = {(e["parent_campaign_root"], e["parent_config_sha256"]) for e in entries}
    if len(parents) != 1:
        raise ValueError("Controllers do not reference the same pinned parent")
    parent_root, parent_sha = next(iter(parents))
    plan = {
        "schema": "peptide_empty_waiter_restore_plan_v1",
        "host": socket.gethostname(),
        "prepared_at": time.time(),
        "parent_campaign_root": parent_root,
        "parent_config_sha256": parent_sha,
        "registry_path": str(Path(registry_path).resolve()),
        "registry_sha256": sha(registry_path),
        "refit_launch_path": str(Path(refit_launch_path).resolve()),
        "refit_launch_sha256": sha(refit_launch_path),
        "controllers": entries,
        "restoration_implementation_sha256": sha(__file__),
        "script_pinning_scope": (
            "failed-controller preparation; journal input/runtime pins checked where present"
        ),
    }
    if source_precheck_path is not None:
        plan["source_precheck_path"] = str(Path(source_precheck_path).resolve())
        plan["source_precheck_sha256"] = sha(source_precheck_path)
    write(output / "restore_plan.json", plan)
    (output / "restore_plan.sha256").write_text(sha(output / "restore_plan.json") + "\n")
    return plan


def parent_running(plan, proc_root=Path("/proc")):
    journal = read(Path(plan["parent_campaign_root"]) / "campaign_status.json")
    if (
        journal.get("status") != "running"
        or journal.get("config_sha256") != plan["parent_config_sha256"]
    ):
        raise ValueError("Same pinned parent must have an actual running recovered controller")
    expected = journal.get("process_identity", {})
    actual = identity(journal["pid"], proc_root)
    if (
        actual is None
        or actual["state"] == "Z"
        or actual["start_ticks"] != expected.get("start_ticks")
    ):
        raise ValueError("Recovered parent process identity is not live")
    argv = actual["argv"]
    if (
        len(argv) < 4
        or Path(argv[1]).name != "peptide_modeling_campaign.py"
        or "--config" not in argv
    ):
        raise ValueError("Recovered parent command does not expose its pinned configuration")
    file_pin(argv[argv.index("--config") + 1], plan["parent_config_sha256"])
    return {
        "pid": journal["pid"],
        "process_identity": actual,
        "config_sha256": journal["config_sha256"],
        "journal_sha256": sha(Path(plan["parent_campaign_root"]) / "campaign_status.json"),
    }


def copy_checked(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(destination) != expected or sha(source) != expected:
        raise ValueError("Backup failed byte-exact verification")


def restore_one(
    entry,
    plan,
    token,
    proc_root=Path("/proc"),
    launch=subprocess.Popen,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    parent_proof = parent_running(plan, proc_root)
    old_released(entry["old_identity"], proc_root)
    if matching_controllers(entry["argv"], proc_root):
        raise ValueError("A matching controller became live")
    empty_failure(read(entry["journal_path"]), entry["output_root"])
    for key in ("source_files_sha256", "failed_files_sha256", "log_files_sha256"):
        verify_files(entry[key])
    history = Path(entry["output_root"]) / "recovery_history" / token
    durable_history = Path(entry["durable_root"]) / "recovery_history" / token
    history.mkdir(parents=True, exist_ok=False)
    durable_history.mkdir(parents=True, exist_ok=False)
    originals = {**entry["failed_files_sha256"], **entry["log_files_sha256"]}
    backup = {}
    for number, (name, expected) in enumerate(originals.items()):
        relative = Path("files") / f"{number:03d}_{Path(name).name}"
        for root in (history, durable_history):
            copy_checked(name, root / relative, expected)
        backup[name] = {"sha256": expected, "backup_relative": relative.as_posix()}
    manifest = {
        "original_files": backup,
        "source_files_sha256": entry["source_files_sha256"],
        "old_identity": entry["old_identity"],
        "parent_running_evidence": parent_proof,
    }
    for root in (history, durable_history):
        write(root / "failed_proof_manifest.json", manifest)
    parent_running(plan, proc_root)
    verify_files(entry["failed_files_sha256"])
    moved = []
    for number, name in enumerate(entry["retire_paths"]):
        source = Path(name)
        destination = (
            (durable_history if source.parent == Path(entry["durable_root"]) else history)
            / "retired"
            / f"{number:03d}_{source.name}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved.append((source, destination))
    command = entry["argv"]
    env = os.environ.copy()
    env.update(
        {
            name: str(entry["threads"])
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log = history / "restored_controller.log"
    try:
        with log.open("xb") as handle:
            child = launch(
                command,
                cwd=str(Path(entry["config_path"]).parent),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except BaseException:
        for source, destination in reversed(moved):
            if source.exists():
                raise RuntimeError(
                    "Launch failed after another process created a new journal"
                ) from None
            destination.replace(source)
        raise
    pending = {
        "status": "launched_awaiting_journal",
        "pid": child.pid,
        "identity_at_launch": identity(child.pid, proc_root),
        "argv": command,
        "journal_path": entry["journal_path"],
        "launch_log": str(log),
        "retired_files": {str(a): str(b) for a, b in moved},
    }
    for root in (history, durable_history):
        write(root / "launch_pending.json", pending)
    deadline = monotonic() + 30
    while monotonic() < deadline:
        path = Path(entry["journal_path"])
        if path.exists():
            journal = read(path)
            if journal.get("pid") == child.pid:
                actual = identity(child.pid, proc_root)
                if (
                    journal.get("status") != "waiting_parent"
                    or journal.get("config_sha256") != entry["config_sha256"]
                    or actual is None
                    or actual["state"] == "Z"
                    or actual["start_ticks"]
                    != journal.get("process_identity", {}).get("start_ticks")
                    or actual["argv"] != command
                ):
                    raise RuntimeError(
                        "Restored controller did not establish a verified waiting journal"
                    )
                empty = not journal.get("rounds") and not journal.get("recoveries")
                if not empty:
                    raise RuntimeError("Restored controller unexpectedly launched model work")
                result = {
                    "controller_id": entry["controller_id"],
                    "status": "restored_waiting_parent",
                    "old_identity": entry["old_identity"],
                    "new_identity": actual,
                    "config_sha256": entry["config_sha256"],
                    "argv": command,
                    "journal_path": entry["journal_path"],
                    "history": str(history),
                    "durable_history": str(durable_history),
                    "launch_log": str(log),
                    "retired_files": {str(a): str(b) for a, b in moved},
                    "parent_running_evidence": parent_proof,
                }
                for root in (history, durable_history):
                    write(root / "restoration_receipt.json", result)
                return result
        if child.poll() is not None:
            raise RuntimeError(
                "Restored controller exited; retain its new log and journal for review"
            )
        sleep(0.1)
    raise RuntimeError("Restored controller has no verified journal yet; retain its live handle")


def execute(plan_path, output, proc_root=Path("/proc"), launch=subprocess.Popen):
    plan_path, output = Path(plan_path), Path(output)
    if output.resolve() != plan_path.parent.resolve():
        raise ValueError("Execute into the directory containing the pinned restoration plan")
    if sha(plan_path) != plan_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Restoration plan SHA changed")
    plan = read(plan_path)
    if sha(__file__) != plan.get("restoration_implementation_sha256"):
        raise ValueError("Restoration implementation changed after preparation")
    verify_files(
        {
            plan["registry_path"]: plan["registry_sha256"],
            plan["refit_launch_path"]: plan["refit_launch_sha256"],
        }
    )
    if plan.get("source_precheck_path"):
        file_pin(plan["source_precheck_path"], plan["source_precheck_sha256"])
    parent_running(plan, proc_root)
    if (output / "restoration_receipt.json").exists():
        raise FileExistsError("Restoration already has a receipt; do not launch duplicates")
    for entry in plan["controllers"]:
        old_released(entry["old_identity"], proc_root)
        if matching_controllers(entry["argv"], proc_root):
            raise ValueError("A matching controller became live")
        empty_failure(read(entry["journal_path"]), entry["output_root"])
        for key in ("source_files_sha256", "failed_files_sha256", "log_files_sha256"):
            verify_files(entry[key])
    descriptor = os.open(output / "restore.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "plan_sha256": sha(plan_path)}, handle)
    token = f"waiter_restore_{time.time_ns()}"
    receipt = {
        "schema": "peptide_empty_waiter_restore_receipt_v1",
        "status": "running",
        "plan_sha256": sha(plan_path),
        "token": token,
        "controllers": [],
    }
    write(output / "restoration_receipt.json", receipt)
    try:
        for entry in plan["controllers"]:
            receipt["controllers"].append(restore_one(entry, plan, token, proc_root, launch))
            write(output / "restoration_receipt.json", receipt)
        receipt["status"] = "completed"
        receipt["registry_overlay"] = {
            item["controller_id"]: item["new_identity"] for item in receipt["controllers"]
        }
        write(output / "restoration_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status="partial_or_failed", reason=f"{type(exc).__name__}:{exc}")
        write(output / "restoration_receipt.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--execute", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--refit-launch", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--source-precheck", type=Path)
    parser.add_argument("--source-precheck-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix" or socket.gethostname() != "node23":
        raise RuntimeError("Restore only on the declared node23 Linux host")
    if args.prepare:
        if not all(
            (args.registry, args.refit_launch, args.source_precheck, args.source_precheck_sha256)
        ):
            parser.error("--prepare requires registry, refit launch and SHA-pinned source precheck")
        file_pin(args.source_precheck, args.source_precheck_sha256)
        result = prepare(
            args.registry,
            args.refit_launch,
            args.output_dir,
            source_precheck_path=args.source_precheck,
        )
        print(
            json.dumps(
                {"status": "prepared_not_launched", "controllers": len(result["controllers"])}
            )
        )
    else:
        if not args.plan:
            parser.error("--execute requires --plan")
        result = execute(args.plan, args.output_dir)
        print(json.dumps({"status": result["status"], "controllers": len(result["controllers"])}))


if __name__ == "__main__":
    main()
