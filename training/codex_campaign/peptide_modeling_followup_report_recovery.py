"""Repair an identified, fully trained group-followup report failure and finish its seeds.

The retained seed is never staged or fitted again. Original failed process,
dispatcher, verification and archive receipts remain immutable historical evidence.
Only missing seeds run, using SHA-pinned report-only replacements.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

import peptide_modeling_campaign as campaign
import peptide_modeling_followup_seeds as followup
from peptide_modeling_supervise import archive_run, atomic_json, digest, read_json

SCHEMA = "peptide_group_followup_report_recovery_v1"
COUNTS_FAILURE = "KeyError:'counts'"
NUMERIC_FAILURE = "VerificationError:Nonfinite inverse-transformed predictions"
SUPPORTED_FAILURES = {COUNTS_FAILURE, NUMERIC_FAILURE}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def contained(root, value):
    root, value = Path(root).resolve(), Path(value)
    path = (value if value.is_absolute() else root / value).resolve()
    require(path.is_relative_to(root) and path != root, "Recovery evidence path escapes its run")
    return path


def snapshot_files(root, files):
    for relative, expected in files.items():
        path = contained(root, relative)
        require(
            path.is_file() and not path.is_symlink() and digest(path) == expected,
            "Frozen source/feature input changed: " + str(relative),
        )


def same_live(expected, actual):
    return (
        actual is not None
        and actual.get("state") != "Z"
        and (expected is None or actual.get("start_ticks") == expected.get("start_ticks"))
    )


def released(identity, *, expected_argv=None, proc_root=Path("/proc")):
    require(
        isinstance(identity, dict) and identity.get("pid") and identity.get("start_ticks"),
        "An original process identity is required for recovery",
    )
    if expected_argv is not None:
        require(
            identity.get("argv") == expected_argv, "Recorded process argv differs from its launch"
        )
    actual = campaign.process_identity(identity["pid"], proc_root)
    require(not same_live(identity, actual), "An original controller or supervisor is still live")


def dead_seed(run, entry, proc_root=Path("/proc"), *, workers=None):
    """Check recorded handles and the live process inventory, without signalling anything."""
    run = Path(run).resolve()
    supervisor = read_json(run / "full_supervisor.json")
    dispatcher = read_json(run / "dispatch/latest_full.json")
    require(
        supervisor.get("config_sha256")
        == dispatcher.get("config_sha256")
        == entry["config_sha256"],
        "Seed process receipts do not bind the original configuration",
    )
    require(
        supervisor.get("task_manifest_sha256")
        == dispatcher.get("task_manifest_sha256")
        == digest(run / "task_manifest.json"),
        "Seed task-manifest process contract changed",
    )
    require(
        supervisor.get("pid") == entry.get("pid"), "Supervisor PID differs from the launch receipt"
    )
    require(
        entry.get("process_identity", {}).get("pid") == entry.get("pid"),
        "Recorded supervisor process identity has a different PID",
    )
    identity, command = entry["process_identity"], entry.get("argv")
    require(
        isinstance(command, list)
        and len(command) == 8
        and isinstance(workers, int)
        and workers > 0,
        "Original supervisor Popen command and configured worker count are required",
    )
    expected_command = [
        command[0],
        str(run / "scripts/peptide_modeling_supervise.py"),
        "--run-dir",
        str(run),
        "--config",
        str(run / "config.json"),
        "--workers",
        str(workers),
    ]
    require(
        command == expected_command,
        "Original supervisor Popen command differs from the frozen run/config/workers",
    )
    observed = identity.get("argv")
    identity_kind = "recorded_nonempty_launch_identity"
    if observed == []:
        expected_dispatch = [
            command[0],
            str(run / "scripts/peptide_modeling_dispatch.py"),
            "--run-dir",
            str(run),
            "--config",
            str(run / "config.json"),
            "--profile",
            "full",
            "--workers",
            str(workers),
            "--remote-root",
            str(run),
        ]
        require(
            supervisor.get("argv") == expected_dispatch,
            "Empty startup observation lacks an exact bound final dispatcher command",
        )
        require(
            isinstance(dispatcher.get("pid"), int)
            and dispatcher["pid"] > 0
            and supervisor.get("dispatcher_pid") == dispatcher["pid"],
            "Empty startup observation lacks a bound supervisor/dispatcher PID relationship",
        )
        identity_kind = "empty_exec_startup_observation_with_bound_final_receipts"
        # Retain the empty observation; final receipts support the command instead of inventing it.
        released(identity, proc_root=proc_root)
    else:
        require(
            isinstance(observed, list) and observed,
            "A missing launch argv is not an observed empty-exec race",
        )
        released(identity, expected_argv=command, proc_root=proc_root)
    for receipt in (supervisor, dispatcher):
        require(
            receipt.get("status") in campaign.TERMINAL, "A seed process receipt is not terminal"
        )
        if receipt.get("pid"):
            actual = campaign.process_identity(receipt["pid"], proc_root)
            require(
                not campaign.has_argument(actual, "--run-dir", str(run)),
                "An owned dispatcher/supervisor is still live",
            )
    for task_id, task in dispatcher.get("tasks", {}).items():
        require(
            task.get("status") in {"completed", "deferred"}, "A training worker was not terminal"
        )
        if task.get("pid"):
            actual = campaign.process_identity(task["pid"], proc_root)
            require(
                not campaign.owned_worker(actual, run, task_id),
                "An owned training worker is still live",
            )
    # A crash may leave unrecorded handles. Refuse every process with this exact run argument.
    for directory in Path(proc_root).iterdir() if Path(proc_root).is_dir() else []:
        if directory.name.isdigit():
            actual = campaign.process_identity(int(directory.name), proc_root)
            require(
                not campaign.has_argument(actual, "--run-dir", str(run)),
                "An unrecorded process still owns this seed run",
            )
    return (
        supervisor,
        dispatcher,
        dict(
            identity_evidence=identity_kind,
            observed_argv=copy.deepcopy(observed),
            recorded_process_identity=copy.deepcopy(identity),
            recorded_popen_argv=copy.deepcopy(command),
            final_supervisor_dispatcher_argv=supervisor.get("argv"),
            dispatcher_pid=dispatcher.get("pid"),
        ),
    )


def load_report_module(path, expected):
    require(digest(path) == expected, "Pinned reporting implementation changed")
    name = "_followup_report_recovery_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def report_modules(contract, revision):
    selected = {}
    for name in (campaign.POSTPROCESS_FILE, campaign.PIPELINE_POSTPROCESS_FILE):
        override = revision["overrides"].get(name)
        path = Path(override["path"]) if override else Path(contract["template"]) / name
        expected = override["sha256"] if override else contract["files"][name]
        selected[name] = load_report_module(path, expected)
    return selected[campaign.POSTPROCESS_FILE], selected[campaign.PIPELINE_POSTPROCESS_FILE]


def expected_settings(training, root, durable, index, seed):
    run = root / f"round_{index:03d}_seed_{seed}"
    settings = copy.deepcopy(training)
    settings["training"]["seed"] = seed
    settings["run_dir"] = str(run)
    settings["resources"].update(
        remote_training_scratch=str(run),
        remote_durable_root=str(durable / run.name),
        max_parallel_tasks=32,
    )
    return run, settings


def fitted_evidence(run, manifest, settings, summarizer):
    """Recheck every retained result/model/prediction hash; never unpickle or fit estimators."""
    from peptide_modeling_campaign_report import canonical, unique_candidates

    scopes = {"common", *settings.get("coverage_cohorts", {})}
    tasks = {task["task_id"]: task for task in manifest["tasks"] if task["status"] == "eligible"}
    expected = {f"{task}_{scope}.json" for task in tasks for scope in scopes}
    paths = list((run / "results/full").glob("*.json"))
    require({path.name for path in paths} == expected, "Missing or unrecorded task-scope results")
    hashes, completed = {}, 0
    for path in paths:
        result = read_json(path)
        require(
            result.get("status") in {"completed", "deferred"},
            "Actual training failure is not report-recoverable",
        )
        require(result.get("task_id") in tasks, "A retained result belongs to another task")
        hashes[str(path)] = digest(path)
        if result["status"] == "completed":
            require(
                result.get("actual_inner_folds") == result.get("requested_inner_folds") == 5,
                "A retained model does not prove five actual folds",
            )
            require(
                result.get("search", {}).get("backend") == "optuna",
                "A retained model did not use Optuna",
            )
            summarizer.verified_predictions(run, str(run), result, tasks[result["task_id"]])
            candidates, _ = unique_candidates(run, result)
            selected = (
                result["selected_route"],
                result["selected_model"],
                canonical(result["selected_params"]),
            )
            require(
                any(
                    (c["route"], c["model"], canonical(c["params"])) == selected for c in candidates
                ),
                "Selected model has no matching successful Optuna candidate evidence",
            )
            for candidate in candidates:
                path = contained(run, candidate["trial_file"])
                hashes[str(path)] = candidate["trial_sha256"]
            for name, receipt in result["artifact_hashes"].items():
                artifact = contained(run, name)
                require(digest(artifact) == receipt["sha256"], "Retained fitted artifact changed")
                hashes[str(artifact)] = receipt["sha256"]
            completed += 1
    return hashes, completed


def unchanged(hashes):
    require(
        all(Path(path).is_file() and digest(path) == expected for path, expected in hashes.items()),
        "An original fitted result/model/prediction changed during reporting",
    )


def verify_original_archive(run, archive_receipt, hashes, bound_receipts=None):
    """Bind current retained fitted bytes to the already-published pre-repair archive."""
    import hashlib

    require(isinstance(archive_receipt, dict), "The original failed seed archive is required")
    path = Path(archive_receipt["path"])
    require(
        path.is_file() and digest(path) == archive_receipt["sha256"],
        "Original failed archive changed",
    )
    expected = {
        str(Path(run.name) / Path(name).relative_to(run)).replace("\\", "/"): value
        for name, value in hashes.items()
    }
    found = set()
    receipt_members = {}
    if bound_receipts is not None:
        receipt_members = {
            str(Path(run.name) / relative).replace("\\", "/"): expected
            for relative, expected in bound_receipts.items()
        }
    receipt_found = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name in receipt_members:
                require(member.isfile(), "Archived final process receipt is not a regular member")
                with archive.extractfile(member) as handle:
                    archived_receipt = json.load(handle)
                require(
                    all(
                        archived_receipt.get(k) == v
                        for k, v in receipt_members[member.name].items()
                    ),
                    "Empty startup observation has conflicting archived final process receipts",
                )
                receipt_found.add(member.name)
            if member.name not in expected:
                continue
            require(member.isfile(), "Archived fitted artifact is not a regular member")
            value = hashlib.sha256()
            with archive.extractfile(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            require(
                value.hexdigest() == expected[member.name],
                "Current fitted artifact differs from its original archive",
            )
            found.add(member.name)
    require(found == set(expected), "Original archive omits a retained result/model/prediction")
    require(
        receipt_found == set(receipt_members), "Original archive lacks bound final process receipts"
    )


def task_audit(run, settings, endpoints, methods, task_ids, verification, parent=None):
    audit = campaign.audit_round(run, settings, endpoints, verification)
    completed = {
        item["task_id"] for values in audit["endpoint_completed_scopes"].values() for item in values
    }
    audit["missing_completed_task_ids"] = sorted(set(task_ids) - completed)
    audit["attempted_independent_methods"] = [
        m["method_id"] for m in methods if m["route"] in audit["attempted_routes"]
    ]
    audit["own_method_count"] = len(audit["attempted_independent_methods"])
    parent_coverage = bool((parent or {}).get("global_method_model_evidence_by_seed", {}))
    audit["method_model_coverage_met"] = (
        audit["own_method_count"] >= 10 or parent_coverage
    ) and len(audit["attempted_model_families"]) >= 7
    audit["global_method_requirement_basis"] = (
        "verified_parent_campaign" if parent_coverage else "this_followup"
    )
    return audit


def mirror_preserved_models(run, durable, hashes):
    copied = 0
    for source, expected in hashes.items():
        path = Path(source)
        destination = durable / "live" / path.relative_to(run)
        if destination.exists():
            require(
                digest(destination) == expected,
                "Existing durable fitted copy differs; refusing overwrite",
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".report_recovery_writing")
        require(not temporary.exists(), "A previous durable recovery write remains unresolved")
        shutil.copy2(path, temporary)
        require(digest(temporary) == expected, "Durable fitted copy failed its SHA check")
        temporary.replace(destination)
        copied += 1
    return copied


def prepare_recovery(config_path, overrides, *, proc_root=Path("/proc")):
    """Verify and repair only reports, returning an adoption plan without launching a seed."""
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    training, registry, methods, endpoints = followup.validate_followup(config)
    followup.verify_runtime(training)
    root, durable, template = (
        Path(config[k]).resolve() for k in ("output_root", "durable_root", "template_run")
    )
    prior_path = root / "campaign_status.json"
    prior = read_json(prior_path)
    require(
        prior.get("schema") == "peptide_bounded_seed_followup_v1"
        and prior.get("status") == "failed",
        "Recovery accepts only a failed bounded group-followup journal",
    )
    require(
        prior.get("config_sha256") == digest(config_path), "Original followup configuration changed"
    )
    require(
        prior.get("host") == socket.gethostname(),
        "Recovery cannot verify another host's process identity",
    )
    identity = prior.get("process_identity")
    require(
        isinstance(identity, dict) and identity.get("pid") == prior.get("pid"),
        "Controller PID identity is inconsistent",
    )
    require(
        campaign.has_argument({**identity, "state": "S"}, "--config", str(config_path)),
        "Original controller identity refers to another configuration",
    )
    released(identity, proc_root=proc_root)
    require(
        prior.get("required_seeds") == config.get("required_seeds", [42, 2026, 3407]),
        "Required seeds changed",
    )
    contract = read_json(root / "template_contract.json")
    require(
        contract == campaign.template_contract(template, Path(config["template_config"]), registry),
        "Original template/config/registry contract changed",
    )
    snapshot_files(template, contract["files"])
    revision = campaign.runtime_revision(overrides, contract)
    require(revision is not None, "A SHA-pinned report-only runtime revision is required")
    summarizer, pipeline = report_modules(contract, revision)
    rounds, seeds = prior.get("rounds", []), config.get("required_seeds", [42, 2026, 3407])
    require(
        0 < len(rounds) <= len(seeds) == 3,
        "Recovery requires a nonempty prefix of exactly three seeds",
    )
    require(
        all(entry.get("status") == "completed" for entry in rounds[:-1])
        and rounds[-1].get("status") == "failed",
        "Only the final fully trained seed may have a report failure",
    )
    require(
        {p.name for p in root.glob("round_*") if p.is_dir()}
        == {f"round_{i:03d}_seed_{seed}" for i, seed in enumerate(seeds[: len(rounds)])},
        "Unrecorded seed directory exists; refusing replacement work",
    )
    all_hashes, seed_details = {}, []
    for index, entry in enumerate(rounds):
        seed = seeds[index]
        run, settings = expected_settings(training, root, durable, index, seed)
        require(
            entry.get("index") == index
            and entry.get("seed") == seed
            and Path(entry.get("run_dir", "")).resolve() == run,
            "Seed journal is not the exact required prefix",
        )
        require(
            read_json(run / "config.json") == settings
            and digest(run / "config.json") == entry["config_sha256"],
            "Retained seed configuration differs from the frozen template",
        )
        existing_revision = entry.get("runtime_revision")
        campaign.verify_runtime_revision(existing_revision, contract)
        snapshot_files(run, campaign.round_files(contract, existing_revision))
        provenance = dict(
            seed=seed, template_contract=contract, config_sha256=entry["config_sha256"]
        )
        if existing_revision:
            provenance["runtime_revision"] = existing_revision
        require(
            read_json(run / "campaign_input_contract.json") == provenance,
            "Retained seed input provenance changed",
        )
        supervisor, dispatcher, identity_observation = dead_seed(
            run, entry, proc_root, workers=int(config.get("workers", 32))
        )
        manifest = read_json(run / "task_manifest.json")
        expected_tasks = {
            task["task_id"] for task in manifest["tasks"] if task["status"] == "eligible"
        }
        require(
            set(dispatcher.get("tasks", {})) == expected_tasks, "Worker journal omits or adds tasks"
        )
        failure = None
        if index == len(rounds) - 1:
            failure = dispatcher.get("reason")
            require(
                dispatcher.get("status") == supervisor.get("status") == "failed"
                and failure in SUPPORTED_FAILURES,
                "Failure is not the explicitly supported counts/numerical reporting defect",
            )
            if failure == COUNTS_FAILURE:
                require(
                    "counts" not in manifest
                    and campaign.PIPELINE_POSTPROCESS_FILE in revision["overrides"],
                    "Counts repair requires absent counts and a pipeline report-only override",
                )
            else:
                require(
                    campaign.POSTPROCESS_FILE in revision["overrides"],
                    "Numerical repair requires the approved summarizer override",
                )
                previous_verification = read_json(run / "verification.json")
                require(
                    previous_verification.get("reason") == NUMERIC_FAILURE,
                    "Numerical failure receipt differs from its dispatch reason",
                )
        else:
            require(
                supervisor.get("status") == dispatcher.get("status") == "completed",
                "Previous successful seed process receipts changed",
            )
            verify_path = Path(
                entry.get("audit", {}).get("verification_path", run / "verification.json")
            )
            require(
                digest(verify_path) == entry.get("audit", {}).get("verification_sha256"),
                "Earlier seed verification changed",
            )
            task_audit(run, settings, endpoints, methods, config["task_ids"], verify_path)
        hashes, completed = fitted_evidence(run, manifest, settings, summarizer)
        bound_receipts = None
        if (
            identity_observation["identity_evidence"]
            == "empty_exec_startup_observation_with_bound_final_receipts"
        ):
            bound_receipts = {
                "full_supervisor.json": {
                    key: supervisor[key]
                    for key in (
                        "pid",
                        "config_sha256",
                        "task_manifest_sha256",
                        "argv",
                        "dispatcher_pid",
                    )
                },
                "dispatch/latest_full.json": {
                    key: dispatcher[key]
                    for key in ("pid", "config_sha256", "task_manifest_sha256", "status", "tasks")
                },
            }
        verify_original_archive(run, supervisor.get("archive"), hashes, bound_receipts)
        if bound_receipts is not None:
            identity_observation["archived_final_receipts_bound"] = True
        all_hashes.update(hashes)
        seed_details.append(
            dict(
                index=index,
                seed=seed,
                run_dir=str(run),
                completed_model_scopes=completed,
                original_dispatch_reason=failure,
                supervisor=supervisor,
                dispatcher=dispatcher,
                identity_observation=identity_observation,
            )
        )
    parent = followup.parent_ready(config, proc_root)
    require(parent is not None, "Verified twenty-hour parent has not released its owned processes")
    recovery = (
        root / "report_recovery" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:12])
    )
    recovery.mkdir(parents=True, exist_ok=False)
    original_files = {}
    run = Path(rounds[-1]["run_dir"])
    for label, path in (
        ("campaign_status.json", prior_path),
        ("full_supervisor.json", run / "full_supervisor.json"),
        ("dispatch_latest_full.json", run / "dispatch/latest_full.json"),
        ("verification.json", run / "verification.json"),
    ):
        if path.is_file():
            target = recovery / "original" / label
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            original_files[str(path)] = dict(backup=str(target), sha256=digest(path))
    archive = seed_details[-1]["supervisor"].get("archive")
    if archive:
        path = Path(archive["path"])
        require(
            path.is_file() and digest(path) == archive["sha256"], "Original failed archive changed"
        )
    # Run report_stage on exact copies of the small report inputs, never the old model run.
    report_inputs = recovery / "report_inputs"
    report_inputs.mkdir()
    shutil.copy2(run / "task_manifest.json", report_inputs / "task_manifest.json")
    shutil.copytree(run / "results", report_inputs / "results")
    pipeline.report_stage(report_inputs)
    verified_dir = recovery / "verified_reports"
    summary = summarizer.summarize_run(run, str(run), output_dir=verified_dir)
    require(
        summary.get("status") == "verified"
        and summary.get("assertions", {}).get("all_checks_passed") is True,
        "Report-only revalidation did not verify retained predictions/artifacts",
    )
    verification = verified_dir / "verification.json"
    audit = task_audit(
        run,
        read_json(run / "config.json"),
        endpoints,
        methods,
        config["task_ids"],
        verification,
        parent,
    )
    unchanged(all_hashes)
    snapshot_files(template, contract["files"])
    require(
        digest(prior_path) == digest(recovery / "original/campaign_status.json"),
        "Controller journal changed during recovery preparation",
    )
    mirrored = mirror_preserved_models(
        run,
        durable / run.name,
        {p: h for p, h in all_hashes.items() if Path(p).is_relative_to(run)},
    )
    receipt = dict(
        schema=SCHEMA,
        status="report_repair_verified_not_adopted",
        source_config_sha256=digest(config_path),
        source_followup_config=str(config_path),
        failed_journal_sha256=digest(prior_path),
        source_run=str(run),
        repaired_seed=rounds[-1]["seed"],
        original_failure=seed_details[-1]["original_dispatch_reason"],
        identity_evidence=seed_details[-1]["identity_observation"]["identity_evidence"],
        identity_observations=[
            dict(seed=s["seed"], **s["identity_observation"]) for s in seed_details
        ],
        original_files=original_files,
        retained_failed_archive=archive,
        retained_fitted_files_sha256=all_hashes,
        runtime_revision=revision,
        verification_path=str(verification),
        verification_sha256=digest(verification),
        parent_requirement_evidence=parent,
        missing_seeds=seeds[len(rounds) :],
        models_refitted=0,
        selection_repeated=False,
        productive_time_added=0,
        additional_twenty_hour_loop=False,
        original_models_results_inputs_modified=False,
        durable_files_copied=mirrored,
        audit=audit,
    )
    receipt_path = recovery / "repair_receipt.json"
    atomic_json(receipt_path, receipt)
    durable_recovery = durable / "report_recovery" / recovery.name
    shutil.copytree(recovery, durable_recovery)
    archived = archive_run(recovery, durable_recovery, "verified_report_repair")
    return dict(
        config=config,
        training=training,
        registry=registry,
        methods=methods,
        endpoints=endpoints,
        root=root,
        durable=durable,
        prior=prior,
        contract=contract,
        revision=revision,
        receipt=receipt,
        receipt_path=receipt_path,
        recovery_dir=recovery,
        durable_recovery=durable_recovery,
        repair_archive=archived,
        original_artifact_hashes=all_hashes,
    )


def recover_followup(config_path, overrides):
    """Adopt one verified report repair, then run only the absent required seed suffix."""
    config_path = Path(config_path).resolve()
    initial = read_json(config_path)
    followup.verify_runtime(read_json(initial["template_config"]))
    root = Path(initial["output_root"]).resolve()
    lock, token = root / "followup_report_recovery.lock", uuid.uuid4().hex
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(pid=os.getpid(), token=token), handle)
    receipt, child = None, None
    try:
        plan = prepare_recovery(config_path, overrides)
        config, training, durable = plan["config"], plan["training"], plan["durable"]
        journal_path = root / "campaign_status.json"
        prior = plan["prior"]
        require(
            digest(journal_path) == plan["receipt"]["failed_journal_sha256"],
            "Failed journal changed before adoption",
        )
        receipt = copy.deepcopy(prior)
        old_identity = receipt["process_identity"]
        original_reason = receipt.pop("reason", "")
        receipt.pop("finished_at", None)
        identity = campaign.process_identity(os.getpid())
        require(
            identity and identity["state"] != "Z", "New recovery controller identity unavailable"
        )
        require(
            campaign.has_argument(identity, "--config", str(config_path)),
            "Recovery controller argv lacks the original config",
        )
        receipt.update(
            status="running",
            pid=os.getpid(),
            process_identity=identity,
            active_runtime_revision=plan["revision"],
        )
        last = receipt["rounds"][-1]
        last.update(
            status="completed",
            audit=plan["receipt"]["audit"],
            postprocess_exit_code=0,
            verification_runtime_revision=plan["revision"],
            followup_report_repair=dict(
                receipt_path=str(plan["receipt_path"]),
                sha256=digest(plan["receipt_path"]),
                verification_path=plan["receipt"]["verification_path"],
                verification_sha256=plan["receipt"]["verification_sha256"],
                repaired_reports_archive=plan["repair_archive"],
                original_supervisor_exit_code=last.get("exit_code"),
            ),
        )
        receipt.setdefault("recovery_history", []).append(
            dict(
                schema=SCHEMA,
                recovered_at=time.time(),
                previous_controller_identity=old_identity,
                previous_failure_reason=original_reason,
                source_failed_journal_sha256=plan["receipt"]["failed_journal_sha256"],
                repair_receipt_path=str(plan["receipt_path"]),
                repair_receipt_sha256=digest(plan["receipt_path"]),
                retained_seeds=[e["seed"] for e in prior["rounds"]],
                missing_seeds=plan["receipt"]["missing_seeds"],
                configuration_unchanged=True,
                models_refitted=0,
                additional_twenty_hour_loop=False,
            )
        )

        def save():
            require(
                digest(config_path) == receipt["config_sha256"],
                "Original configuration changed during continuation",
            )
            receipt["last_checked_at"] = time.time()
            atomic_json(journal_path, receipt)
            atomic_json(durable / "campaign_status.json", receipt)

        save()
        seeds = receipt["required_seeds"]
        for index in range(len(receipt["rounds"]), len(seeds)):
            seed = seeds[index]
            run = root / f"round_{index:03d}_seed_{seed}"
            settings = campaign.stage_round(
                Path(config["template_run"]),
                run,
                training,
                seed,
                durable / run.name,
                plan["contract"],
                plan["registry"],
                plan["revision"],
            )
            entry = dict(
                index=index,
                seed=seed,
                run_dir=str(run),
                status="starting",
                config_sha256=digest(run / "config.json"),
                runtime_revision=plan["revision"],
            )
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
            with (run / "followup_report_recovery_supervisor.log").open(
                "x", encoding="utf-8"
            ) as handle:
                child = subprocess.Popen(
                    command,
                    cwd=run,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                entry.update(
                    status="running",
                    pid=child.pid,
                    process_identity=campaign.process_identity(child.pid),
                    argv=command,
                    started_at=time.time(),
                )
                save()
                while child.poll() is None:
                    time.sleep(float(config.get("poll_seconds", 5)))
                    save()
            entry.update(exit_code=child.returncode, finished_at=time.time())
            if child.returncode != 0:
                entry["status"] = "failed"
            require(
                child.returncode == 0,
                "A newly required seed failed; inspect its actual training/report receipts",
            )
            summarizer, _ = report_modules(plan["contract"], plan["revision"])
            snapshot_files(run, campaign.round_files(plan["contract"], plan["revision"]))
            fitted_evidence(run, read_json(run / "task_manifest.json"), settings, summarizer)
            entry.update(
                status="completed",
                audit=task_audit(
                    run,
                    settings,
                    plan["endpoints"],
                    plan["methods"],
                    config["task_ids"],
                    run / "verification.json",
                    plan["receipt"]["parent_requirement_evidence"],
                ),
            )
            unchanged(plan["original_artifact_hashes"])
            snapshot_files(Path(config["template_run"]), plan["contract"]["files"])
            save()
        from peptide_modeling_campaign_report import build_report

        receipt["cross_seed_report"] = build_report(root, durable)
        complete = all(
            not e["audit"]["missing_completed_task_ids"] and e["audit"]["method_model_coverage_met"]
            for e in receipt["rounds"]
        )
        receipt.update(
            status="completed" if complete else "requirement_incomplete", finished_at=time.time()
        )
        unchanged(plan["original_artifact_hashes"])
        snapshot_files(Path(config["template_run"]), plan["contract"]["files"])
        save()
        return receipt
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        if receipt is not None:
            receipt.update(
                status="failed", reason=f"{type(exc).__name__}:{exc}", finished_at=time.time()
            )
            atomic_json(root / "campaign_status.json", receipt)
            atomic_json(Path(initial["durable_root"]) / "campaign_status.json", receipt)
        raise
    finally:
        if lock.exists() and read_json(lock).get("token") == token:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-overrides", type=Path, required=True)
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="Verify and materialize report sidecars without adopting or launching seeds",
    )
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if args.repair_only:
        value = prepare_recovery(args.config, args.runtime_overrides)
        result = dict(
            status="report_repair_verified_not_adopted",
            receipt_path=str(value["receipt_path"]),
            missing_seeds=value["receipt"]["missing_seeds"],
        )
    else:
        value = recover_followup(args.config, args.runtime_overrides)
        result = dict(status=value["status"], seeds=[e["seed"] for e in value["rounds"]])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
