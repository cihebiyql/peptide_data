"""Run isolated seeded Optuna experiments for at least twenty productive hours.

CLI: python peptide_modeling_campaign.py --config campaign.json

The campaign JSON declares template_run, template_config, campaign_root,
durable_root, predecessors=[{run_dir, config_sha256}], required_seeds,
minimum_productive_hours, workers, max_rounds and sample_seconds. Optional
registry_path defaults to template_run/representation_methods.json. Optional
expected_endpoints is the complete source endpoint name list. Waiting for
predecessors and final archival do not count toward productive wall time.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

from peptide_modeling_supervise import atomic_json, digest, read_json

TERMINAL = {"completed", "failed", "requirement_incomplete"}
INPUT_DIRECTORIES = ("features", "task_tables", "scripts")
INPUT_FILES = (
    "task_manifest.json",
    "task_summary.tsv",
    "endpoint_admission_summary.tsv",
    "record_dispositions.tsv",
    "deferred_records.tsv",
    "feature_rows_index.tsv",
    "representation_methods.json",
    "final_feature_bank_verification.json",
)
NUMERIC_POLICY = "finite_transformed_metrics_with_explicit_unrepresentable_native_v1"
POSTPROCESS_FILE = "scripts/peptide_modeling_summarize.py"
PIPELINE_POSTPROCESS_FILE = "scripts/peptide_modeling_pipeline.py"


def report_stage_only_replacement(original, replacement):
    """Allow a pipeline report body change while retaining every other Python AST node."""
    trees = []
    for source in (original, replacement):
        tree = ast.parse(Path(source).read_text(encoding="utf-8-sig"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "report_stage"
        ]
        if len(functions) != 1:
            raise ValueError(
                "Pipeline override requires exactly one ordinary report_stage function"
            )
        functions[0].body = []
        trees.append(ast.dump(tree, include_attributes=False))
    if trees[0] != trees[1]:
        raise ValueError(
            "Pipeline override changes code outside report_stage or its signature/decorators"
        )


def runtime_revision(path, contract):
    """Permit only a SHA-pinned reporting replacement, never a training-code override."""
    if path is None:
        return None
    path = Path(path).resolve()
    value = read_json(path)
    if value.get("schema") != "peptide_postprocess_override_v1":
        raise ValueError("Unknown postprocessing runtime revision schema")
    overrides = value.get("overrides", {})
    if (
        not isinstance(overrides, dict)
        or not overrides
        or set(overrides) - {POSTPROCESS_FILE, PIPELINE_POSTPROCESS_FILE}
    ):
        raise ValueError(
            "Only the numerical summarizer and pipeline report_stage may be overridden"
        )
    for name, item in overrides.items():
        source = Path(item["path"])
        if source.is_symlink() or not source.is_file() or digest(source) != item["sha256"]:
            raise ValueError("Pinned postprocessing replacement changed")
        if item.get("base_sha256") != contract["files"].get(name):
            raise ValueError("Postprocessing revision does not match the original template")
        if name == PIPELINE_POSTPROCESS_FILE:
            original = Path(contract["template"]) / name
            if (
                original.is_symlink()
                or not original.is_file()
                or digest(original) != item["base_sha256"]
            ):
                raise ValueError("Original pipeline source differs from its frozen template hash")
            report_stage_only_replacement(original, source)
    return {
        "manifest_path": str(path),
        "manifest_sha256": digest(path),
        "overrides": overrides,
    }


def verify_runtime_revision(revision, contract):
    if revision is not None and runtime_revision(revision["manifest_path"], contract) != revision:
        raise ValueError("Frozen postprocessing runtime revision changed")


def round_files(contract, revision):
    files = dict(contract["files"])
    if revision:
        files.update({name: item["sha256"] for name, item in revision["overrides"].items()})
    return files


def process_identity(pid, proc_root=Path("/proc")):
    """Read a Linux process identity and aggregate CPU ticks, including all threads."""
    try:
        directory = Path(proc_root) / str(int(pid))
        raw = (directory / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        command = (directory / "cmdline").read_bytes().split(b"\0")
        return {
            "pid": int(pid),
            "start_ticks": int(fields[19]),
            "cpu_ticks": int(fields[11]) + int(fields[12]),
            "state": fields[0],
            "argv": [item.decode("utf-8", errors="replace") for item in command if item],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def has_argument(identity, flag, expected):
    if identity is None or identity["state"] == "Z":
        return False
    argv = identity["argv"]
    return any(a == flag and b == str(expected) for a, b in zip(argv, argv[1:], strict=False))


def owned_worker(identity, run_dir, task_id):
    return has_argument(identity, "--run-dir", run_dir) and has_argument(
        identity, "--only-task", task_id
    )


def predecessor_terminal(specification, proc_root=Path("/proc")):
    """A receipt alone never proves that a predecessor released its CPU budget."""
    root = Path(specification["run_dir"])
    supervisor_path = root / "full_supervisor.json"
    dispatch_path = root / "dispatch/latest_full.json"
    if not supervisor_path.is_file() or not dispatch_path.is_file():
        return False
    supervisor, dispatch = read_json(supervisor_path), read_json(dispatch_path)
    expected = specification["config_sha256"]
    if supervisor.get("config_sha256") != expected or dispatch.get("config_sha256") != expected:
        raise ValueError("Predecessor configuration does not match its frozen contract")
    if supervisor.get("status") not in TERMINAL or dispatch.get("status") not in TERMINAL:
        return False
    if specification.get("require_success", False) and (
        supervisor.get("status") != "completed" or dispatch.get("status") != "completed"
    ):
        raise RuntimeError("Required predecessor ended without successful recovery")
    for task_id, task in dispatch.get("tasks", {}).items():
        if task.get("status") == "running":
            return False
        if task.get("pid") and owned_worker(
            process_identity(task["pid"], proc_root), root, task_id
        ):
            return False
    for receipt in (supervisor, dispatch):
        if receipt.get("pid") and has_argument(
            process_identity(receipt["pid"], proc_root), "--run-dir", root
        ):
            return False
    return True


def collect_workers(run_dir, proc_root=Path("/proc")):
    path = Path(run_dir) / "dispatch/latest_full.json"
    if not path.exists():
        return {}
    receipt = read_json(path)
    workers = {}
    for task_id, task in receipt.get("tasks", {}).items():
        if task.get("status") != "running" or not task.get("pid"):
            continue
        identity = process_identity(task["pid"], proc_root)
        if owned_worker(identity, run_dir, task_id):
            key = f"{identity['pid']}:{identity['start_ticks']}"
            workers[key] = {"cpu_ticks": identity["cpu_ticks"], "task_id": task_id}
    return workers


def productive_delta(previous, current, elapsed, sample_seconds, ticks_per_second):
    """Credit one capped wall interval only when an identified owned worker progressed."""
    ticks = sum(
        max(0, worker["cpu_ticks"] - previous[key]["cpu_ticks"])
        for key, worker in current.items()
        if key in previous and worker["task_id"] == previous[key]["task_id"]
    )
    return {
        "productive_wall_seconds": min(max(0.0, elapsed), sample_seconds) if ticks else 0.0,
        "owned_worker_cpu_seconds": ticks / ticks_per_second,
    }


def seed_for_round(round_index, required_seeds):
    """Extra rounds are independent robustness evaluations, never cached repetitions."""
    if round_index < len(required_seeds):
        return int(required_seeds[round_index])
    candidate = 1_000_003 + (round_index - len(required_seeds)) * 104_729
    while candidate in required_seeds:
        candidate += 104_729
    return candidate


def representation_registry(path):
    registry = read_json(path)
    methods = [method for method in registry["methods"] if method.get("independent") is True]
    ids = [method["method_id"] for method in methods]
    families = [method["algorithm_family_id"] for method in methods]
    if len(ids) != len(set(ids)) or len(families) != len(set(families)):
        raise ValueError("Independent representation methods must have distinct algorithm families")
    if len(methods) < 10:
        raise ValueError("At least ten independent representation methods are required")
    return registry, methods


def validate_campaign(config):
    required = config.get("required_seeds", [42, 2026, 3407])
    if len(required) < 3 or len(set(required)) != len(required):
        raise ValueError("At least three distinct required seeds must be declared")
    if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in required):
        raise ValueError("Seeds must be uint32 integers")
    if float(config.get("minimum_productive_hours", 20)) < 20:
        raise ValueError("The campaign requires at least twenty productive wall hours")
    if int(config.get("max_rounds", 60)) < len(required):
        raise ValueError("max_rounds must cover all required seeds")
    if not 0 < float(config.get("sample_seconds", 5)) <= 60:
        raise ValueError("sample_seconds must be in (0, 60]")
    template, root = Path(config["template_run"]).resolve(), Path(config["campaign_root"]).resolve()
    durable = Path(config["durable_root"]).resolve()
    if template == root or template in root.parents or root in template.parents:
        raise ValueError("Template and campaign output roots must be separate")
    if durable == root or durable in root.parents or root in durable.parents:
        raise ValueError("Scratch and durable campaign roots must be separate")
    training = read_json(config["template_config"])
    if training["training"].get("inner_folds") != 5:
        raise ValueError("Every campaign evaluation must use strict five-fold CV")
    if training["training"].get("search_backend") != "optuna":
        raise ValueError("Optuna is required")
    models = training["training"].get("models", [])
    if len(set(models)) < 7:
        raise ValueError("At least seven different model families are required")
    workers = int(config.get("workers", 32))
    threads = int(training["training"].get("n_jobs", 2))
    if workers < 1 or threads < 1 or workers * threads > 64:
        raise ValueError("Campaign concurrency exceeds the sixty-four CPU budget")
    registry_path = Path(config.get("registry_path", template / "representation_methods.json"))
    _, methods = representation_registry(registry_path)
    routes = training["training"].get("routes", [])
    if any(method["route"] not in routes for method in methods):
        raise ValueError("All independent methods must participate in configured training routes")
    return training, registry_path, methods


def template_contract(template, template_config, registry_path):
    files = {}
    for directory in INPUT_DIRECTORIES:
        for path in sorted((template / directory).rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("Template inputs must be regular files, not symlinks")
                files[path.relative_to(template).as_posix()] = digest(path)
    for name in INPUT_FILES:
        path = template / name
        if path.is_file():
            files[name] = digest(path)
    if "task_manifest.json" not in files or "features/manifest.json" not in files:
        raise FileNotFoundError("Prepared template tasks and feature manifest are required")
    return {
        "template": str(template),
        "files": files,
        "template_config_sha256": digest(template_config),
        "registry_sha256": digest(registry_path),
    }


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_round(
    template, destination, training, seed, durable, contract, registry_path, revision=None
):
    """Only immutable inputs are shared. Splits, results and databases always start fresh."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Round output already exists: {destination}")
    verify_runtime_revision(revision, contract)
    destination.mkdir(parents=True)
    for name, expected_sha in contract["files"].items():
        source = template / name
        # The template was hashed before scheduling and must remain immutable.
        if digest(source) != expected_sha:
            raise ValueError(f"Template input changed: {name}")
        link_or_copy(source, destination / name)
    if revision:
        for name, item in revision["overrides"].items():
            target = destination / name
            temporary = target.with_name(target.name + ".replacement")
            shutil.copy2(item["path"], temporary)
            if digest(temporary) != item["sha256"]:
                raise ValueError("Copied postprocessing replacement failed its SHA check")
            temporary.replace(target)  # Replace the hardlink, never write through it.
    if not (destination / "representation_methods.json").exists():
        shutil.copy2(registry_path, destination / "representation_methods.json")
    configuration = copy.deepcopy(training)
    configuration["training"]["seed"] = seed
    configuration["run_dir"] = str(destination)
    configuration["resources"]["remote_training_scratch"] = str(destination)
    configuration["resources"]["remote_durable_root"] = str(durable)
    configuration["resources"]["max_parallel_tasks"] = 32
    atomic_json(destination / "config.json", configuration)
    provenance = {
        "seed": seed,
        "template_contract": contract,
        "config_sha256": digest(destination / "config.json"),
    }
    if revision:
        provenance["runtime_revision"] = revision
    atomic_json(destination / "campaign_input_contract.json", provenance)
    return configuration


def audit_round(run_dir, config, expected_endpoints, verification_path=None):
    verification_path = Path(verification_path or run_dir / "verification.json")
    manifest = read_json(run_dir / "task_manifest.json")
    verification = read_json(verification_path)
    if verification.get("status") != "verified":
        raise ValueError("Seed run did not pass artifact and prediction verification")
    if verification.get("source_task_manifest_sha256") != digest(run_dir / "task_manifest.json"):
        raise ValueError("Verification does not match the current task manifest")
    scopes = {"common", *config.get("coverage_cohorts", {})}
    completed = defaultdict(list)
    deferred = defaultdict(list)
    statuses = defaultdict(dict)
    attempted_routes, attempted_models = set(), set()
    for task in manifest["tasks"]:
        endpoint = task["endpoint"]
        if task["status"] != "eligible":
            deferred[endpoint].append(task.get("reason", "not_eligible"))
            continue
        for scope in sorted(scopes):
            path = run_dir / "results/full" / (task["task_id"] + "_" + scope + ".json")
            result = read_json(path)
            state = result.get("status")
            if state not in {"completed", "deferred"}:
                raise ValueError(f"Nonterminal or failed task scope: {path.name}")
            statuses[task["task_id"]][scope] = state
            if state == "completed":
                if result.get("actual_inner_folds") != 5:
                    raise ValueError("A completed result does not prove five actual folds")
                if result.get("requested_inner_folds") != 5:
                    raise ValueError("A completed result requested a different fold count")
                if result.get("search", {}).get("backend") != "optuna":
                    raise ValueError("A completed result did not use Optuna")
                for route, model in result.get("search", {}).get("coverage_attempted", []):
                    attempted_routes.add(route)
                    attempted_models.add(model)
                completed[endpoint].append({"task_id": task["task_id"], "scope": scope})
            else:
                deferred[endpoint].append(result.get("reason", "deferred"))
    expected = set(expected_endpoints) or {task["endpoint"] for task in manifest["tasks"]}
    return {
        "status": "verified",
        "verification_sha256": digest(verification_path),
        "eligible_tasks": len(statuses),
        "completed_endpoints": sorted(completed),
        "missing_completed_endpoints": sorted(expected - set(completed)),
        "endpoint_completed_scopes": dict(completed),
        "endpoint_deferral_reasons": {key: sorted(set(value)) for key, value in deferred.items()},
        "all_completed_results_have_five_folds": True,
        "attempted_routes": sorted(attempted_routes),
        "attempted_model_families": sorted(attempted_models),
        **(
            {"verification_path": str(verification_path)}
            if verification_path != run_dir / "verification.json"
            else {}
        ),
    }


def completion_status(receipt, config):
    required = set(config.get("required_seeds", [42, 2026, 3407]))
    verified = {
        r["seed"] for r in receipt["rounds"] if r.get("audit", {}).get("status") == "verified"
    }
    if not required.issubset(verified):
        return "running"
    if (
        receipt["productive_wall_seconds"]
        < float(config.get("minimum_productive_hours", 20)) * 3600
    ):
        return "running"
    missing = {
        r["seed"]: r["audit"]["missing_completed_endpoints"]
        for r in receipt["rounds"]
        if r["seed"] in required and r["audit"]["missing_completed_endpoints"]
    }
    return "requirement_incomplete" if missing else "completed"


def verify_runtime(training):
    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Productive CPU accounting requires Linux /proc")
    if socket.gethostname() != training["resources"]["training_host"]:
        raise RuntimeError("Campaign must run on the declared training host")


def _verify_snapshot_files(directory, files):
    directory = Path(directory).resolve()
    for name, expected in files.items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Resume input escapes its frozen root")
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Resume input changed: {name}")


def repaired_verification(entry, run, revision):
    """Validate a separate repair without rewriting the original failed receipts."""
    pointer = entry.get("postprocess_repair")
    if not pointer:
        return run / "verification.json"
    path = Path(pointer["receipt_path"])
    if digest(path) != pointer["sha256"]:
        raise ValueError("Postprocessing repair receipt changed")
    repair = read_json(path)
    if (
        repair.get("schema") != "peptide_postprocess_repair_v1"
        or repair.get("source_run") != str(run)
        or repair.get("seed") != entry["seed"]
        or repair.get("source_config_sha256") != entry["config_sha256"]
        or repair.get("original_exit_code") != 1
        or repair.get("runtime_revision") != revision
    ):
        raise ValueError("Postprocessing repair belongs to a different frozen round")
    for source, evidence in repair["original_files"].items():
        if digest(evidence["backup"]) != evidence["sha256"]:
            raise ValueError("Original failed-receipt backup changed")
        if not evidence.get("mutable_after_adoption") and digest(source) != evidence["sha256"]:
            raise ValueError("Original failed round receipt changed")
    _verify_snapshot_files(run, repair["result_files_sha256"])
    verification = Path(repair["verification_path"])
    if digest(verification) != repair["verification_sha256"]:
        raise ValueError("Repaired numerical verification changed")
    proof = read_json(verification)
    if (
        proof.get("status") != "verified"
        or proof.get("numeric_policy") != NUMERIC_POLICY
        or proof.get("assertions", {}).get("all_checks_passed") is not True
    ):
        raise ValueError("Repaired numerical verification is not complete")
    return verification


def _completed_training_failure(run, entry):
    """Only the known reporting failure may adopt already finished training."""
    if entry.get("status") != "failed" or entry.get("exit_code") != 1:
        raise ValueError("Repair requires the failed final supervisor exit to be retained")
    failure = read_json(run / "verification.json")
    if failure.get("status") != "failed" or failure.get("reason") != (
        "VerificationError:Nonfinite inverse-transformed predictions"
    ):
        raise ValueError("Failed round is not the supported inverse-reporting failure")
    if not predecessor_terminal({"run_dir": str(run), "config_sha256": entry["config_sha256"]}):
        raise RuntimeError("Failed seed has not released its owned processes")
    manifest = read_json(run / "task_manifest.json")
    expected = {task["task_id"] for task in manifest["tasks"] if task["status"] == "eligible"}
    dispatch = read_json(run / "dispatch/latest_full.json")
    tasks = dispatch.get("tasks", {})
    if set(tasks) != expected or any(
        item.get("status") not in {"completed", "deferred"} for item in tasks.values()
    ):
        raise ValueError("Repair refuses missing, partial or failed training tasks")


def prepare_postprocess_repair(config_path, output, overrides):
    """Reverify a finished numeric-report failure in a fresh sidecar directory."""
    from peptide_modeling_summarize import summarize_run

    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_json(config_path)
    training, registry_path, methods = validate_campaign(config)
    root, durable = Path(config["campaign_root"]).resolve(), Path(config["durable_root"]).resolve()
    journal_path = root / "campaign_status.json"
    prior = read_json(journal_path)
    _failed_controller(prior, config_path)
    _valid_accounting(prior)
    contract = read_json(root / "template_contract.json")
    revision = runtime_revision(overrides, contract)
    if revision is None:
        raise ValueError("A pinned numerical postprocessing revision is required")
    if POSTPROCESS_FILE not in revision["overrides"]:
        raise ValueError("Inverse-report repair requires a pinned numerical summarizer replacement")
    import peptide_modeling_summarize

    if (
        digest(peptide_modeling_summarize.__file__)
        != revision["overrides"][POSTPROCESS_FILE]["sha256"]
    ):
        raise ValueError("Repair must execute the pinned numerical summarizer")
    _verify_snapshot_files(Path(config["template_run"]), contract["files"])
    rounds = prior["rounds"]
    if not rounds or any(entry.get("status") != "completed" for entry in rounds[:-1]):
        raise ValueError("Only the final fully trained round may be repaired")
    for entry in rounds[:-1]:
        _resume_round_audit(entry, root, durable, training, contract, methods, config)
    entry = rounds[-1]
    run = Path(entry["run_dir"]).resolve()
    _completed_training_failure(run, entry)
    if any(
        output == source or output in source.parents or source in output.parents
        for source in (root, durable, Path(config["template_run"]).resolve())
    ):
        raise ValueError("Repair output must be separate from campaign and template roots")
    output.mkdir(parents=True, exist_ok=False)
    original_files = {}
    sources = {
        "campaign_status.json": journal_path,
        "verification.json": run / "verification.json",
        "full_supervisor.json": run / "full_supervisor.json",
        "dispatch_latest_full.json": run / "dispatch/latest_full.json",
    }
    for name, source in sources.items():
        target = output / "original" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        original_files[str(source)] = {
            "backup": str(target),
            "sha256": digest(source),
            "mutable_after_adoption": source == journal_path,
        }
    summary_dir = output / "numerical_verification"
    summary = summarize_run(run, str(run), output_dir=summary_dir)
    if summary.get("status") != "verified" or summary.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError("Revised numerical policy did not verify the completed training")
    verification = summary_dir / "verification.json"
    repair = {
        "schema": "peptide_postprocess_repair_v1",
        "source_run": str(run),
        "seed": entry["seed"],
        "source_config_sha256": entry["config_sha256"],
        "failed_campaign_sha256": digest(journal_path),
        "original_exit_code": entry["exit_code"],
        "original_files": original_files,
        "verification_path": str(verification),
        "verification_sha256": digest(verification),
        "runtime_revision": revision,
        "result_files_sha256": {
            path.relative_to(run).as_posix(): digest(path)
            for path in sorted((run / "results/full").glob("*.json"))
        },
        "productive_time_added": 0,
        "model_selection_repeated": False,
        "original_models_and_predictions_modified": False,
    }
    path = output / "repair_receipt.json"
    atomic_json(path, repair)
    adopted = _adopt_repair(prior, path, revision, config, methods)
    _resume_round_audit(adopted["rounds"][-1], root, durable, training, contract, methods, config)
    return {"repair_receipt": str(path), "sha256": digest(path), "status": "verified"}


def _adopt_repair(prior, path, revision, config, methods):
    repair, receipt = read_json(path), copy.deepcopy(prior)
    backup = next(
        item["backup"]
        for item in repair["original_files"].values()
        if item.get("mutable_after_adoption")
    )
    if read_json(backup) != prior:
        raise ValueError("Repair refers to a different failed campaign journal")
    entry = receipt["rounds"][-1]
    entry["postprocess_repair"] = {
        "receipt_path": str(Path(path).resolve()),
        "sha256": digest(path),
    }
    entry["verification_runtime_revision"] = revision
    run = Path(entry["run_dir"]).resolve()
    verification = repaired_verification(entry, run, revision)
    entry["status"] = "completed"
    entry["audit"] = audit_round(
        run, read_json(run / "config.json"), config.get("expected_endpoints", []), verification
    )
    entry["audit"]["attempted_independent_methods"] = [
        method["method_id"]
        for method in methods
        if method["route"] in entry["audit"]["attempted_routes"]
    ]
    return receipt


def _failed_controller(prior, config_path):
    if prior.get("status") != "failed":
        raise ValueError("Explicit resume accepts only a failed campaign")
    if prior.get("config_sha256") != digest(config_path):
        raise ValueError("Resume campaign configuration changed")
    if prior.get("host") != socket.gethostname():
        raise ValueError("Resume cannot verify a controller from another host")
    identity = process_identity(prior["pid"])
    if identity is not None and identity["state"] != "Z":
        expected = prior.get("process_identity", {}).get("start_ticks")
        if (
            expected is None
            or identity["start_ticks"] == expected
            or has_argument(identity, "--config", config_path)
        ):
            raise RuntimeError("Prior campaign controller is still live")


def _valid_accounting(prior):
    rounds = prior.get("rounds", [])
    if [entry.get("index") for entry in rounds] != list(range(len(rounds))):
        raise ValueError("Resume round indices are incomplete or repeated")
    for key in ("productive_wall_seconds", "owned_worker_cpu_seconds"):
        total = float(prior.get(key, -1))
        values = [float(entry.get(key, -1)) for entry in rounds]
        if (
            not math.isfinite(total)
            or total < 0
            or any(not math.isfinite(value) or value < 0 for value in values)
            or not math.isclose(sum(values), total, rel_tol=1e-10, abs_tol=1e-6)
        ):
            raise ValueError("Resume time accounting is incomplete or inconsistent")


def _resume_round_audit(entry, root, durable, training, contract, methods, config):
    """Recheck saved evidence without rewriting any completed seed artifact."""
    from peptide_modeling_summarize import read_table, resolve_artifact, verified_predictions

    index = entry["index"]
    expected_seed = seed_for_round(index, config.get("required_seeds", [42, 2026, 3407]))
    run = root / f"round_{index:03d}_seed_{expected_seed}"
    if entry.get("seed") != expected_seed or Path(entry.get("run_dir", "")).resolve() != run:
        raise ValueError("Resume seed/run identity differs from its deterministic plan")
    if entry.get("status") != "completed" or entry.get("audit", {}).get("status") != "verified":
        raise ValueError("Resume requires all existing rounds to be completed and verified")
    revision = entry.get("runtime_revision")
    verify_runtime_revision(revision, contract)
    verification_revision = entry.get("verification_runtime_revision")
    verify_runtime_revision(verification_revision, contract)
    verification_path = repaired_verification(entry, run, verification_revision)
    if not predecessor_terminal(
        {
            "run_dir": str(run),
            "config_sha256": entry["config_sha256"],
            "require_success": not bool(entry.get("postprocess_repair")),
        }
    ):
        raise RuntimeError("A completed seed still has an owned live process")
    if digest(run / "config.json") != entry["config_sha256"]:
        raise ValueError("Completed seed configuration changed")
    settings = copy.deepcopy(training)
    settings["training"]["seed"] = expected_seed
    settings["run_dir"] = str(run)
    settings["resources"]["remote_training_scratch"] = str(run)
    settings["resources"]["remote_durable_root"] = str(durable / run.name)
    settings["resources"]["max_parallel_tasks"] = 32
    if read_json(run / "config.json") != settings:
        raise ValueError("Completed seed configuration differs from its frozen template")
    provenance = read_json(run / "campaign_input_contract.json")
    expected_provenance = {
        "seed": expected_seed,
        "template_contract": contract,
        "config_sha256": entry["config_sha256"],
    }
    if revision:
        expected_provenance["runtime_revision"] = revision
    if provenance != expected_provenance:
        raise ValueError("Completed seed input provenance differs from the frozen template")
    _verify_snapshot_files(run, round_files(contract, revision))
    old_audit = entry["audit"]
    if digest(verification_path) != old_audit.get("verification_sha256"):
        raise ValueError("Completed seed verification receipt changed")
    current = audit_round(run, settings, config.get("expected_endpoints", []), verification_path)
    for name in (
        "completed_endpoints",
        "missing_completed_endpoints",
        "attempted_routes",
        "attempted_model_families",
    ):
        if current[name] != old_audit.get(name):
            raise ValueError(f"Completed seed audit changed: {name}")
    attempted = [m["method_id"] for m in methods if m["route"] in current["attempted_routes"]]
    if len(attempted) < 10 or len(current["attempted_model_families"]) < 7:
        raise ValueError("Completed seed lacks the required actual method/model coverage")
    tasks = {task["task_id"]: task for task in read_json(run / "task_manifest.json")["tasks"]}
    checked = 0
    for path in sorted((run / "results/full").glob("*.json")):
        result = read_json(path)
        if result.get("status") != "completed":
            continue
        evidence = verified_predictions(run, str(run), result, tasks[result["task_id"]])
        oof = resolve_artifact(run, str(run), result["artifacts"]["oof"])
        if {int(row["fold"]) for row in read_table(oof)} != set(range(5)):
            raise ValueError("Completed seed OOF records do not prove all five folds")
        checked += evidence["artifact_hashes_verified"]
    return {
        "index": index,
        "seed": expected_seed,
        "artifact_hashes_revalidated": checked,
        "verification_sha256": current["verification_sha256"],
        "saved_predictions_and_metrics_revalidated": True,
    }


def resume_state(
    root,
    durable,
    receipt_path,
    config_path,
    config,
    training,
    registry_path,
    methods,
    repair_path=None,
    revision=None,
):
    """Adopt only a dead, failed controller whose completed rounds remain valid."""
    raw = receipt_path.read_bytes()
    prior = json.loads(raw)
    _failed_controller(prior, config_path)
    _valid_accounting(prior)
    rounds = prior.get("rounds", [])
    contract = read_json(root / "template_contract.json")
    template = Path(config["template_run"]).resolve()
    if contract.get("template") != str(template):
        raise ValueError("Resume template root changed")
    if contract.get("template_config_sha256") != digest(config["template_config"]):
        raise ValueError("Resume template configuration changed")
    if contract.get("registry_sha256") != digest(registry_path):
        raise ValueError("Resume representation registry changed")
    _verify_snapshot_files(template, contract["files"])
    verify_runtime_revision(revision, contract)
    if prior.get("active_runtime_revision") and revision != prior["active_runtime_revision"]:
        raise ValueError("Resume must retain its previously pinned reporting revision")
    adopted = prior
    if repair_path:
        repair = read_json(repair_path)
        if repair.get("failed_campaign_sha256") != digest(receipt_path):
            raise ValueError("Repair is not pinned to these exact failed campaign bytes")
        _completed_training_failure(Path(rounds[-1]["run_dir"]), rounds[-1])
        adopted = _adopt_repair(prior, repair_path, revision, config, methods)
    revalidated = [
        _resume_round_audit(entry, root, durable, training, contract, methods, config)
        for entry in adopted["rounds"]
    ]
    expected_roots = {Path(entry["run_dir"]).resolve() for entry in rounds}
    if any(path.resolve() not in expected_roots for path in root.glob("round_*")):
        raise ValueError("Unrecorded round output requires separate recovery review")
    # Back up the exact failed bytes only after all revalidation has passed.
    stamp = f"{time.time_ns()}_{digest(receipt_path)[:16]}"
    backup = Path("recovery_history") / ("journal_" + stamp + ".json")
    for parent in (root, durable):
        target = parent / backup
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
    receipt = copy.deepcopy(adopted)
    if revision:
        receipt["active_runtime_revision"] = revision
    receipt.setdefault("recovery_history", []).append(
        {
            "prior_journal_sha256": digest(receipt_path),
            "backup": backup.as_posix(),
            "prior_pid": prior["pid"],
            "prior_process_identity": prior.get("process_identity"),
            "prior_status": prior["status"],
            "prior_reason": prior.get("reason"),
            "resumed_at": time.time(),
            "retained_productive_wall_seconds": prior["productive_wall_seconds"],
            "round_revalidation": revalidated,
            "downtime_credited": False,
        }
    )
    for key in ("reason", "finished_at", "owned_supervisor_still_live"):
        receipt.pop(key, None)
    receipt.update(
        status="resuming", pid=os.getpid(), process_identity=process_identity(os.getpid())
    )
    return receipt, contract


def run_campaign(config_path, resume=False, repair_path=None, overrides=None):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    training, registry_path, methods = validate_campaign(config)
    verify_runtime(training)
    root, durable = Path(config["campaign_root"]).resolve(), Path(config["durable_root"]).resolve()
    if (repair_path or overrides) and not resume:
        raise ValueError("Postprocessing repair and runtime overrides require explicit --resume")
    revision = (
        runtime_revision(overrides, read_json(root / "template_contract.json"))
        if overrides
        else None
    )
    root.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    lock = root / "campaign.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "hostname": socket.gethostname(), "token": token}, handle)
    receipt_path = root / "campaign_status.json"
    receipt = None

    def save(check_contract=True):
        if check_contract and digest(config_path) != receipt["config_sha256"]:
            raise ValueError("Campaign configuration changed during execution")
        receipt["last_checked_at"] = time.time()
        atomic_json(receipt_path, receipt)
        atomic_json(durable / "campaign_status.json", receipt)

    child = None
    try:
        if receipt_path.exists():
            if not resume:
                raise FileExistsError("Existing campaign journal requires explicit --resume")
            receipt, contract = resume_state(
                root,
                durable,
                receipt_path,
                config_path,
                config,
                training,
                registry_path,
                methods,
                repair_path,
                revision,
            )
        elif resume:
            raise FileNotFoundError("There is no campaign journal to resume")
        else:
            receipt = {
                "schema": "peptide_multiseed_productive_campaign_v1",
                "status": "waiting_predecessors",
                "pid": os.getpid(),
                "process_identity": process_identity(os.getpid()),
                "host": socket.gethostname(),
                "config_sha256": digest(config_path),
                "started_at": time.time(),
                "required_seeds": config.get("required_seeds", [42, 2026, 3407]),
                "independent_method_count": len(methods),
                "independent_methods": [m["method_id"] for m in methods],
                "model_families": training["training"]["models"],
                "productive_wall_seconds": 0.0,
                "owned_worker_cpu_seconds": 0.0,
                "time_accounting": (
                    "capped sample intervals with same identified owned worker CPU progress"
                ),
                "waiting_and_archival_credited": False,
                "rounds": [],
            }
        save()
        template = Path(config["template_run"]).resolve()
        if resume:
            from peptide_modeling_campaign_report import build_report

            receipt["cross_seed_report"] = build_report(root, durable)
            save()
            state = completion_status(receipt, config)
            if state != "running":
                receipt.update(status=state, finished_at=time.time())
                receipt["required_seed_endpoint_gaps"] = {
                    str(r["seed"]): r["audit"]["missing_completed_endpoints"]
                    for r in receipt["rounds"]
                    if r["seed"] in receipt["required_seeds"]
                }
                save()
                return receipt
        else:
            while not all(predecessor_terminal(p) for p in config.get("predecessors", [])):
                save()
                time.sleep(min(30, float(config.get("sample_seconds", 5))))
            receipt["status"] = "preparing_template_contract"
            save()
            contract = template_contract(template, config["template_config"], registry_path)
            atomic_json(root / "template_contract.json", contract)
            atomic_json(durable / "template_contract.json", contract)
        sample_seconds = float(config.get("sample_seconds", 5))
        ticks_per_second = os.sysconf("SC_CLK_TCK")
        for index in range(len(receipt["rounds"]), int(config.get("max_rounds", 60))):
            seed = seed_for_round(index, receipt["required_seeds"])
            label = f"round_{index:03d}_seed_{seed}"
            run_dir = root / label
            settings = stage_round(
                template,
                run_dir,
                training,
                seed,
                durable / label,
                contract,
                registry_path,
                revision,
            )
            entry = {
                "index": index,
                "seed": seed,
                "run_dir": str(run_dir),
                "status": "starting",
                "config_sha256": digest(run_dir / "config.json"),
                "productive_wall_seconds": 0.0,
                "owned_worker_cpu_seconds": 0.0,
            }
            if revision:
                entry["runtime_revision"] = revision
            receipt["rounds"].append(entry)
            command = [
                sys.executable,
                str(run_dir / "scripts/peptide_modeling_supervise.py"),
                "--run-dir",
                str(run_dir),
                "--config",
                str(run_dir / "config.json"),
                "--workers",
                str(config.get("workers", 32)),
            ]
            with (run_dir / "campaign_supervisor.log").open("a", encoding="utf-8") as output:
                child = subprocess.Popen(
                    command,
                    cwd=run_dir,
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
                previous, last_sample = {}, time.monotonic()
                while child.poll() is None:
                    time.sleep(sample_seconds)
                    current, now = collect_workers(run_dir), time.monotonic()
                    delta = productive_delta(
                        previous, current, now - last_sample, sample_seconds, ticks_per_second
                    )
                    for key, value in delta.items():
                        entry[key] += value
                        receipt[key] += value
                    entry["last_active_worker_count"] = len(current)
                    previous, last_sample = current, now
                    save()
            entry.update(exit_code=child.returncode, finished_at=time.time())
            if child.returncode != 0:
                entry["status"] = "failed"
                raise RuntimeError(f"Seed {seed} supervisor failed with exit {child.returncode}")
            entry["audit"] = audit_round(run_dir, settings, config.get("expected_endpoints", []))
            entry["audit"]["attempted_independent_methods"] = [
                method["method_id"]
                for method in methods
                if method["route"] in entry["audit"]["attempted_routes"]
            ]
            if len(entry["audit"]["attempted_independent_methods"]) < 10:
                raise ValueError("Fewer than ten independent methods were actually attempted")
            if len(entry["audit"]["attempted_model_families"]) < 7:
                raise ValueError("Fewer than seven model families were actually attempted")
            entry["status"] = "completed"
            save()
            from peptide_modeling_campaign_report import build_report

            receipt["cross_seed_report"] = build_report(root, durable)
            save()
            state = completion_status(receipt, config)
            if state != "running":
                receipt["status"] = state
                break
            if not entry["audit"]["completed_endpoints"]:
                receipt.update(
                    status="requirement_incomplete",
                    reason="No endpoint completed; additional seeds cannot justify work",
                )
                break
        else:
            receipt.update(
                status="requirement_incomplete",
                reason="Finite search rounds exhausted before all requirements were proven",
            )
        receipt["finished_at"] = time.time()
        receipt["required_seed_endpoint_gaps"] = {
            str(r["seed"]): r["audit"]["missing_completed_endpoints"]
            for r in receipt["rounds"]
            if "audit" in r and r["seed"] in receipt["required_seeds"]
        }
        save()
        return receipt
    except BaseException as exc:
        if child is not None and child.poll() is None:
            # Stop only this campaign's own supervisor. Never launch a replacement.
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                receipt["owned_supervisor_still_live"] = True
        if receipt is not None:
            receipt.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
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
    parser.add_argument(
        "--resume", action="store_true", help="Revalidate a failed completed-round journal"
    )
    parser.add_argument(
        "--runtime-overrides", type=Path, help="Pinned numerical summarizer revision"
    )
    parser.add_argument(
        "--postprocess-repair", type=Path, help="Adopt a separate repaired verification"
    )
    parser.add_argument(
        "--prepare-postprocess-repair", type=Path, help="Write only a fresh repair sidecar"
    )
    args = parser.parse_args()
    if args.prepare_postprocess_repair:
        if args.resume or args.postprocess_repair or not args.runtime_overrides:
            parser.error("Preparation requires --runtime-overrides and no resume/adoption flag")
        print(
            json.dumps(
                prepare_postprocess_repair(
                    args.config, args.prepare_postprocess_repair, args.runtime_overrides
                )
            ),
            flush=True,
        )
        return
    result = run_campaign(
        args.config,
        resume=args.resume,
        repair_path=args.postprocess_repair,
        overrides=args.runtime_overrides,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "productive_wall_seconds", "required_seed_endpoint_gaps")
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()
