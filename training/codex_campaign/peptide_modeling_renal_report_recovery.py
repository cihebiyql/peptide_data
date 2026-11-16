"""Plan or explicitly recover renal reports after all three learning seeds finished.

Inspection uses only the standard library and never changes trained directories.
An explicit apply restores absent, pre-fit-pinned provenance bytes, then calls the
unchanged frozen reporting function. It never resumes learning or rewrites the
original failure journal, fitted artifacts, contracts, labels, or feature banks.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

SCHEMA = "renal_native_report_only_recovery_v1"
PRODUCER_SHA = "bfb175601178c99835c1f680b70f357d139312bf15d9811c8b2a095755429a56"
PROOF_SHA = "4e51728d86e60a7f325bbd02ef39bbd47170dc0f537028a049d50f8783ea0e7a"
NODE_SHA = "4da4b84725554f6b6556ae9c8432d50ed071897583a76a8c301d301520428d13"
TASK = "TASK_RENAL_KG_ANNOTATION_b77ceacb6b28f7f632f34c97"
SEEDS = [42, 2026, 3407]
PRODUCER = "scripts/peptide_modeling_renal_kg_native_bridge.py"
CONTROL = "renal_native_fixed_controls.json"
REPORT = "reports/renal_native_diagnostics.json"
CAMPAIGN = "renal_native_campaign_diagnostics.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def contained(root, relative):
    root = Path(root).resolve()
    relative = PurePosixPath(str(relative).replace("\\", "/"))
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Recovery relative path escapes its declared directory",
    )
    path = root.joinpath(*relative.parts)
    require(path != root and path.resolve().is_relative_to(root), "Recovery path escapes root")
    require(
        not any(p.is_symlink() for p in [path, *path.parents] if p != root.parent),
        "Recovery paths must not traverse symlinks",
    )
    return path


def file_hashes(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Snapshot root must be a regular directory")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Snapshot refuses symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def process_identity(pid, proc_root=Path("/proc")):
    directory = Path(proc_root) / str(pid)
    try:
        stat = (directory / "stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        argv = (directory / "cmdline").read_bytes().split(b"\0")
        return dict(
            pid=int(pid),
            state=fields[0],
            start_ticks=int(fields[19]),
            argv=[v.decode(errors="replace") for v in argv if v],
        )
    except FileNotFoundError:
        return None


def released_processes(expected, template, runs, *, proc_root=Path("/proc")):
    require(
        expected.get("pid") == 3281577 and expected.get("start_ticks") == 95813586,
        "The original controller PID/start identity is mandatory",
    )
    actual = process_identity(expected["pid"], proc_root)
    same_live = (
        actual is not None
        and actual["state"] != "Z"
        and actual["start_ticks"] == expected["start_ticks"]
    )
    require(not same_live, "Original renal controller is still live")
    expected_source = str(Path(template) / PRODUCER)
    owned = set(str(Path(run)) for run in runs)
    checked = 0
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        identity = process_identity(directory.name, proc_root)
        if identity is None or identity["state"] == "Z":
            continue
        checked += 1
        argv = identity["argv"]
        require(
            expected_source not in argv and not any(arg in owned for arg in argv),
            "A live process still owns renal training inputs or a seed directory",
        )
    return dict(
        expected_controller=expected,
        actual_controller=actual,
        original_controller_released=True,
        active_owned_processes=[],
        proc_entries_checked=checked,
        observed_at=time.time(),
    )


def provenance_closure(template, proof):
    """Every restored byte is bound by the published pre-fit proof/node receipt."""
    template = Path(template)
    evidence = proof["nodefold_evidence"]
    require(evidence["sha256"] == NODE_SHA, "The original pre-fit node receipt changed")
    node_path = contained(template, evidence["path"])
    require(digest(node_path) == NODE_SHA, "Pre-fit node receipt bytes changed")
    node = read(node_path)
    require(
        node["producer_sha256"] == PRODUCER_SHA
        and node["probe_scope"] == "actual_node23_pretraining_no_fit"
        and not any(node["guard_counters"].values()),
        "Original no-fit evidence is invalid",
    )
    closure = {}
    for relative, sha in node["input_hashes_before"].items():
        relative = relative.replace("\\", "/")
        if relative.startswith("source_provenance/"):
            closure[relative] = dict(sha256=sha, pin_kind="original_node_receipt_input_hash_before")
    baseline = evidence["pre_node_proof_path"]
    closure[baseline] = dict(
        sha256=evidence["pre_node_proof_sha256"], pin_kind="final_prefit_source_proof_baseline_hash"
    )
    require(len(closure) == 14, "The complete original14-file provenance closure is required")
    for entry in proof["pinned_artifacts"]:
        if entry["path"].startswith("source_provenance/"):
            require(
                closure[entry["path"]]["sha256"] == entry["sha256"],
                "Source-proof and original-node provenance pins conflict",
            )
    for relative, entry in closure.items():
        require(relative.startswith("source_provenance/"), "Only source_provenance is restorable")
        source = contained(template, relative)
        require(
            source.is_file() and digest(source) == entry["sha256"],
            "Immutable template provenance bytes changed: " + relative,
        )
        entry.update(source=str(source), bytes=source.stat().st_size)
    return closure


def inspect_recovery(config, *, proc_root=Path("/proc")):
    """Read-only plan; no ML imports, model loads, fitted controls, or old-file writes."""
    require(config["schema"] == SCHEMA, "Unsupported renal recovery schema")
    template, output, durable = (
        Path(config[k]).resolve() for k in ["template", "output_root", "durable_root"]
    )
    roots = dict(template=str(template), output=str(output), durable=str(durable))
    for new in ["recovery_root", "recovery_durable_root"]:
        path = Path(config[new]).resolve()
        require(
            all(
                path != Path(old)
                and not path.is_relative_to(Path(old))
                and not Path(old).is_relative_to(path)
                for old in roots.values()
            ),
            "Recovery evidence must live outside every protected input/output root",
        )
    followup_path = template / "followup_config.json"
    require(
        digest(followup_path) == config["followup_config_sha256"],
        "Original followup configuration changed",
    )
    cfg = read(followup_path)
    require(
        cfg["template_run"] == str(template)
        and cfg["output_root"] == str(output)
        and cfg["durable_root"] == str(durable)
        and cfg["required_seeds"] == SEEDS,
        "Recovery paths/seeds differ from the original configuration",
    )
    require(digest(template / PRODUCER) == PRODUCER_SHA, "Frozen bfb175 producer changed")
    journal_path = output / "internal_campaign_status.json"
    log_path = Path(config["failed_log"])
    require(
        digest(journal_path) == config["failed_journal_sha256"]
        and digest(log_path) == config["failed_log_sha256"],
        "Original failure evidence changed",
    )
    journal = read(journal_path)
    require(
        journal["status"] == "requirement_incomplete"
        and journal["config_sha256"] == config["followup_config_sha256"]
        and journal["renal_annotation_reporting"]["status"] == "failed"
        and "source_provenance/NAB739_exact_primary_abstract.json"
        in journal["renal_annotation_reporting"]["reason"],
        "Unexpected failure stage",
    )
    require([entry["seed"] for entry in journal["rounds"]] == SEEDS, "All three seeds required")
    runs = [output / f"round_{i:03d}_seed_{seed}" for i, seed in enumerate(SEEDS)]
    released = released_processes(
        config["original_controller"], template, runs, proc_root=proc_root
    )
    task = read(template / "task_manifest.json")["tasks"][0]
    proof_path = task["native_source_observation_coverage"]["proof_path"]
    require(
        task["task_id"] == TASK
        and task["native_source_observation_coverage"]["proof_sha256"] == PROOF_SHA
        and digest(template / proof_path) == PROOF_SHA,
        "Original source proof changed",
    )
    proof = read(template / proof_path)
    closure = provenance_closure(template, proof)
    require(
        not (output / CAMPAIGN).exists() and not (durable / CAMPAIGN).exists(),
        "First campaign reports must not already exist",
    )
    restores, rounds = [], []
    for entry, run in zip(journal["rounds"], runs, strict=True):
        require(
            entry["run_dir"] == str(run)
            and entry["status"] == "completed"
            and entry["requirements_met"]
            and entry["audit"]["status"] == "verified",
            "A learning seed is incomplete or unverified",
        )
        require(digest(run / "config.json") == entry["config_sha256"], "Seed configuration changed")
        contract = read(run / "campaign_input_contract.json")["template_contract"]
        require(
            contract["files"][proof_path] == PROOF_SHA
            and contract["files"][PRODUCER] == PRODUCER_SHA,
            "Learning-time input contract does not bind the frozen proof/producer",
        )
        for relative, sha in contract["files"].items():
            require(
                digest(contained(template, relative)) == sha,
                "Original template contract changed: " + relative,
            )
        require(digest(run / proof_path) == PROOF_SHA, "Round source proof changed")
        result_path = run / "results/internal_cv" / (TASK + ".json")
        result = read(result_path)
        require(
            result["status"] == "completed"
            and result["seed"] == entry["seed"]
            and result["actual_inner_folds"] == result["requested_inner_folds"] == 5,
            "Existing learned result is not completed actual fivefold",
        )
        for path, expected in result["artifact_hashes"].items():
            path = Path(path)
            require(
                path.resolve().is_relative_to(run) and digest(path) == expected["sha256"],
                "Existing learned artifact changed",
            )
            mirror = durable / run.name / path.relative_to(run)
            require(
                mirror.is_file() and digest(mirror) == expected["sha256"],
                "Original durable learned artifact differs from its recorded SHA",
            )
        require(
            digest(durable / run.name / result_path.relative_to(run)) == digest(result_path),
            "Original durable selected result differs",
        )
        require(
            digest(run / "internal_verification.json") == entry["audit"]["verification_sha256"],
            "Original learned verification changed",
        )
        for target_root in [run, durable / run.name]:
            require(target_root.is_dir(), "Each original durable round must already exist")
            require(
                not (target_root / CONTROL).exists() and not (target_root / REPORT).exists(),
                "Controls/reports already exist; never overwrite or refit them",
            )
            for relative, pin in closure.items():
                target = contained(target_root, relative)
                if target.exists():
                    require(
                        target.is_file() and digest(target) == pin["sha256"],
                        "Existing provenance differs; absent-only restoration refused",
                    )
                else:
                    restores.append(
                        dict(
                            destination=str(target),
                            relative=relative,
                            source=pin["source"],
                            sha256=pin["sha256"],
                            pin_kind=pin["pin_kind"],
                            absent_before=True,
                            phase="post_training_provenance_file_completion",
                        )
                    )
        rounds.append(
            dict(
                seed=entry["seed"],
                run_dir=str(run),
                result_path=str(result_path),
                result_sha256=digest(result_path),
                learned_status="completed_verified",
                source_proof_sha256=PROOF_SHA,
                config_sha256=entry["config_sha256"],
            )
        )
    snapshots = {name: file_hashes(path) for name, path in roots.items()}
    snapshots["failed_log"] = {str(log_path): digest(log_path)}
    return dict(
        schema=SCHEMA,
        status="dry_run_reviewable_no_apply",
        roots=roots,
        original_failure=dict(
            journal=str(journal_path),
            journal_sha256=digest(journal_path),
            log=str(log_path),
            log_sha256=digest(log_path),
            journal_status=journal["status"],
        ),
        controller_release_evidence=released,
        before_files=snapshots,
        source_provenance_closure=closure,
        restore_closure=restores,
        rounds=rounds,
        producer_sha256=PRODUCER_SHA,
        source_proof_sha256=PROOF_SHA,
        original_node_receipt_sha256=NODE_SHA,
        planned_control_fit_calls=54,
        selected_model_fit_calls=0,
        search_calls=0,
        staging_calls=0,
        current_fixed_control_fit_calls=0,
        trained_data_modified=False,
        first_reporting_only=True,
        snapshot_created_now_not_historical=True,
        original_failure_receipts_will_remain_unchanged=True,
    )


def verify_preserved(snapshot):
    for name, root in snapshot["roots"].items():
        for relative, sha in snapshot["before_files"][name].items():
            require(
                digest(contained(root, relative)) == sha,
                "Protected original artifact changed: " + name + "/" + relative,
            )
    for path, sha in snapshot["before_files"]["failed_log"].items():
        require(digest(path) == sha, "Original failed controller log changed")


@contextlib.contextmanager
def report_fit_guard(frozen_source):
    """Permit only the fixed controls in frozen reporting, never learning/search/staging."""
    counters = dict(
        fixed_control_fit_calls=0,
        selected_model_fit_calls=0,
        search_calls=0,
        staging_calls=0,
        forbidden_calls=0,
    )
    frozen_source = str(Path(frozen_source).resolve())
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event != "call":
            return
        name, module = frame.f_code.co_name, frame.f_globals.get("__name__", "")
        if name in {
            "run_internal_followup",
            "run_followup",
            "train_internal_task",
            "train_task",
            "_fit_model",
            "_cv_predictions",
            "_fit_cv_ranked_candidate",
            "_run_seed",
        }:
            counters["forbidden_calls"] += 1
            raise RuntimeError("Recovery may never enter learning/refit orchestration")
        if name == "stage_round":
            counters["staging_calls"] += 1
            raise RuntimeError("Recovery may never stage a new learning round")
        if (module.startswith("optuna") and name in {"optimize", "create_study"}) or (
            module == "peptide_modeling_search" and name == "search_candidates"
        ):
            counters["search_calls"] += 1
            raise RuntimeError("Recovery may never search/select new parameters")
        if name in {"fit", "fit_transform", "partial_fit"} and module.startswith(
            ("sklearn", "xgboost", "lightgbm")
        ):
            ancestor = frame.f_back
            allowed = False
            while ancestor:
                if (
                    str(Path(ancestor.f_code.co_filename).resolve()) == frozen_source
                    and ancestor.f_code.co_name == "fixed_ridge_controls"
                ):
                    allowed = True
                    break
                ancestor = ancestor.f_back
            if not allowed:
                counters["selected_model_fit_calls"] += 1
                raise RuntimeError("Only the predeclared fixed float64 controls may be fitted")
            if name == "fit" and type(frame.f_locals.get("self")).__name__ == "Ridge":
                counters["fixed_control_fit_calls"] += 1
                require(counters["fixed_control_fit_calls"] <= 54, "Control fit budget exceeded54")

    sys.setprofile(profile)
    try:
        yield counters
    finally:
        sys.setprofile(previous)


def apply_recovery(config_path, reviewed_path, reviewed_sha):
    """Explicit future action only. Inspection never calls this function."""
    require(
        socket.gethostname() == "node23"
        and sys.executable == "/usr/bin/python3"
        and sys.flags.dont_write_bytecode,
        "Apply requires node23 /usr/bin/python3 -B",
    )
    require(digest(reviewed_path) == reviewed_sha, "Reviewed dry-run evidence changed")
    config, reviewed = read(config_path), read(reviewed_path)
    require(
        reviewed["configuration_sha256"] == digest(config_path)
        and reviewed["recovery_implementation_sha256"] == digest(__file__),
        "Apply must use the exact reviewed recovery configuration and implementation",
    )
    current = inspect_recovery(config)
    for key in ["before_files", "restore_closure", "source_provenance_closure", "rounds"]:
        require(current[key] == reviewed[key], "Original inputs changed after dry-run: " + key)
    root, durable = Path(config["recovery_root"]), Path(config["recovery_durable_root"])
    token = uuid.uuid4().hex
    owner = dict(
        schema=SCHEMA,
        token=token,
        pid=os.getpid(),
        process_identity=process_identity(os.getpid()),
        status="running_report_recovery",
        started_at=time.time(),
        reviewed_sha256=reviewed_sha,
        configuration_sha256=digest(config_path),
        recovery_implementation_sha256=digest(__file__),
    )
    for dest in [root, durable]:
        exclusive_json(dest / "owner.json", owner)
        exclusive_json(dest / "before_snapshot.json", current)
        for kind, path in [
            ("failed_journal", current["original_failure"]["journal"]),
            ("failed_controller_log", current["original_failure"]["log"]),
        ]:
            target = dest / (kind + ".original_bytes")
            with target.open("xb") as handle:
                handle.write(Path(path).read_bytes())
    restored = []
    counters = {}
    status = "failed"
    failure = None
    try:
        verify_preserved(current)
        for item in current["restore_closure"]:
            source, target = Path(item["source"]), Path(item["destination"])
            require(
                digest(source) == item["sha256"] and not target.exists(),
                "Absent-only restoration precondition changed",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(source.read_bytes())
            require(
                digest(target) == item["sha256"], "Restored bytes differ from their pre-fit pin"
            )
            restored.append({**item, "restored_sha256": digest(target)})
        template = Path(config["template"])
        sys.path.insert(0, str(template / "scripts"))
        name = "peptide_modeling_renal_kg_native_bridge"
        require(name not in sys.modules, "Use a fresh report-recovery process")
        source = template / PRODUCER
        require(digest(source) == PRODUCER_SHA, "Frozen reporting producer changed")
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cfg = read(template / "followup_config.json")
        journal = read(current["original_failure"]["journal"])
        with report_fit_guard(source) as counters:
            report = module._complete_followup_reports(cfg, journal)
        require(
            counters["fixed_control_fit_calls"] == 54
            and not any(
                value for key, value in counters.items() if key != "fixed_control_fit_calls"
            ),
            "Recovery must create exactly54 controls and zero learning/search/staging calls",
        )
        require(report["status"] == "verified_internal_annotation_only", "Report did not verify")
        verify_preserved(current)
        generated = {}
        for entry in current["rounds"]:
            for target_root in [
                Path(entry["run_dir"]),
                Path(config["durable_root"]) / Path(entry["run_dir"]).name,
            ]:
                for relative in [CONTROL, REPORT]:
                    target = target_root / relative
                    generated[str(target)] = digest(target)
        for target_root in [Path(config["output_root"]), Path(config["durable_root"])]:
            generated[str(target_root / CAMPAIGN)] = digest(target_root / CAMPAIGN)
        after = {name: file_hashes(path) for name, path in current["roots"].items()}
        allowed = {item["destination"] for item in restored} | set(generated)
        for key, path in current["roots"].items():
            new = set(after[key]) - set(current["before_files"][key])
            require(
                all(str(Path(path) / relative) in allowed for relative in new),
                "Unexpected artifact created during report-only recovery",
            )
        status = "verified_report_recovery_without_retraining"
        result = dict(
            schema=SCHEMA,
            status=status,
            owner=owner,
            restored=restored,
            fit_guard_counters=counters,
            generated_report_sha256=generated,
            before_snapshot_sha256=digest(root / "before_snapshot.json"),
            before_files=current["before_files"],
            after_files=after,
            original_failure=current["original_failure"],
            original_failure_journal_status_retained="requirement_incomplete",
            all_original_fitted_search_contract_bytes_unchanged=True,
            post_training_provenance_completion=True,
            no_retraining=True,
            recovery_reports_are_new_evidence=True,
            finished_at=time.time(),
        )
        for dest in [root, durable]:
            exclusive_json(dest / "recovery_result.json", result)
        return result
    except BaseException as exc:
        failure = dict(exception_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        terminal = dict(
            schema=SCHEMA,
            token=token,
            pid=os.getpid(),
            status=status,
            finished_at=time.time(),
            restored=restored,
            fit_guard_counters=counters,
            failure=failure,
            owner_process_exits_after_receipt=True,
        )
        for dest in [root, durable]:
            exclusive_json(dest / "terminal_receipt.json", terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.inspect:
        require(args.output is not None, "Inspection requires a new evidence output path")
        config = read(args.config)
        require(
            any(
                args.output.resolve().is_relative_to(Path(config[key]).resolve())
                for key in ["recovery_root", "recovery_durable_root"]
            ),
            "Dry-run evidence output must stay within the separate recovery directories",
        )
        result = inspect_recovery(config)
        result.update(
            configuration_sha256=digest(args.config),
            recovery_implementation_sha256=digest(__file__),
        )
        exclusive_json(args.output, result)
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    restores=len(result["restore_closure"]),
                    review_sha256=digest(args.output),
                )
            )
        )
    else:
        require(
            args.review is not None and args.review_sha256, "Apply requires pinned dry-run review"
        )
        result = apply_recovery(args.config, args.review, args.review_sha256)
        print(result["status"])


if __name__ == "__main__":
    main()
