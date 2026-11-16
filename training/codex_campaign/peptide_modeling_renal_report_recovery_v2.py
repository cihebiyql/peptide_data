"""Continue report-only recovery, preserving v1 failure and completed controls.

V1 counted both Ridge.fit and _BaseRidge.fit for the same estimator. This new
driver counts only an outermost fit frame for each self object, verifies/reuses
saved controls, and fits only the two missing complete seed-control artifacts.
Nothing runs on import. --inspect never fits; --apply needs a pinned review.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path

import peptide_modeling_renal_report_recovery as old

SCHEMA = "renal_native_report_only_recovery_v2"
V1_SHA = "f79b58e605d3db72fc397c4ff65d0c3988a82df67c8ed16990c3478ccb3ea44b"
REUSED_SEEDS = [42]
MISSING_SEEDS = [2026, 3407]
DESIGNS = {"molecule_only", "context_only", "molecule_plus_context"}


@contextlib.contextmanager
def report_fit_guard(frozen_source, budget):
    """Count only the outermost same-self Ridge.fit and its successful return."""
    old.require(
        isinstance(budget, int) and budget >= 0, "Explicit nonnegative control budget required"
    )
    counters = dict(
        fixed_control_fit_calls=0,
        fixed_control_fit_completed=0,
        fixed_control_fit_attempts=0,
        inherited_same_self_fit_frames=0,
        fixed_control_fit_failed=0,
        selected_model_fit_calls=0,
        search_calls=0,
        staging_calls=0,
        forbidden_calls=0,
        budget_exceeded_calls=0,
    )
    source = str(Path(frozen_source).resolve())
    outer_frames = {}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        key = id(frame)
        if event == "return" and key in outer_frames:
            estimator = outer_frames.pop(key)
            if arg is estimator:
                counters["fixed_control_fit_completed"] += 1
            else:
                counters["fixed_control_fit_failed"] += 1
            return
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
        if name not in {"fit", "fit_transform", "partial_fit"} or not module.startswith(
            ("sklearn", "xgboost", "lightgbm")
        ):
            return
        ancestor, allowed = frame.f_back, False
        while ancestor:
            if (
                str(Path(ancestor.f_code.co_filename).resolve()) == source
                and ancestor.f_code.co_name == "fixed_ridge_controls"
            ):
                allowed = True
                break
            ancestor = ancestor.f_back
        if not allowed:
            counters["selected_model_fit_calls"] += 1
            raise RuntimeError("Only frozen fixed controls may fit during recovery")
        estimator = frame.f_locals.get("self")
        if name != "fit" or type(estimator).__name__ != "Ridge":
            return
        ancestor = frame.f_back
        while ancestor:
            if ancestor.f_code.co_name == "fit" and ancestor.f_locals.get("self") is estimator:
                counters["inherited_same_self_fit_frames"] += 1
                return
            ancestor = ancestor.f_back
        counters["fixed_control_fit_attempts"] += 1
        if counters["fixed_control_fit_calls"] >= budget:
            counters["budget_exceeded_calls"] += 1
            raise RuntimeError("Outermost control fit budget exceeded")
        old.require(
            estimator.alpha == 1.0 and estimator.solver == "lsqr",
            "Only fixed alpha1 LSQR controls are permitted",
        )
        counters["fixed_control_fit_calls"] += 1
        outer_frames[key] = estimator

    sys.setprofile(profile)
    try:
        yield counters
    finally:
        sys.setprofile(previous)


def load_producer(template):
    source = Path(template) / old.PRODUCER
    old.require(old.digest(source) == old.PRODUCER_SHA, "Frozen bfb175 producer changed")
    name = "peptide_modeling_renal_kg_native_bridge"
    if name in sys.modules:
        module = sys.modules[name]
        old.require(
            Path(module.__file__).resolve() == source.resolve(),
            "A different renal producer is already imported",
        )
        return module
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def captured_report(module, run):
    """Run original numeric report validation while capturing its only output write."""
    captured = []
    previous = module.write
    expected = Path(run) / old.REPORT

    def capture(path, value):
        old.require(Path(path).resolve() == expected.resolve(), "Unexpected report-only write")
        captured.append(value)

    module.write = capture
    try:
        result = module.write_report(run, run_dir=run)
    finally:
        module.write = previous
    old.require(captured == [result], "Exactly one metadata report must be captured")
    return result


def verify_saved_seed(module, run, seed):
    from peptide_modeling_internal_cv import verify_internal_task

    run = Path(run)
    task = module.read(run / "task_manifest.json")["tasks"][0]
    values = module.rows(module.contained(run, task["table_path"]))
    features = module.read(run / "features/manifest.json")
    result = module.read(run / "results/internal_cv" / (old.TASK + ".json"))
    with report_fit_guard(Path(module.__file__), 0) as counters:
        module.verify_source_proof(run, task, values)
        verify_internal_task(task, values, features, run, result)
        report = captured_report(module, run)
    old.require(not any(counters.values()), "Verifying a saved seed must never fit/search/stage")
    old.require(
        report["seed"] == seed and report == module.read(run / old.REPORT),
        "Saved completed report differs from the frozen report validator",
    )
    return report


def inspect_recovery(config, *, proc_root=Path("/proc"), saved_verifier=None):
    """Verify v1 history/restores/partial products and plan only missing controls."""
    old.require(
        config["schema"] == SCHEMA and old.digest(old.__file__) == V1_SHA,
        "Recovery v2 requires the unchanged historical f79 v1 implementation",
    )
    v1, v1_durable = (Path(config[key]).resolve() for key in ["v1_root", "v1_durable_root"])
    before = old.read(v1 / "before_snapshot.json")
    owner, terminal = old.read(v1 / "owner.json"), old.read(v1 / "terminal_receipt.json")
    old.require(
        old.digest(v1 / "before_snapshot.json") == config["v1_before_snapshot_sha256"]
        and old.digest(v1 / "terminal_receipt.json") == config["v1_terminal_sha256"]
        and old.digest(v1 / "owner.json") == config["v1_owner_sha256"],
        "Original v1 failure bytes changed",
    )
    old.require(
        owner["token"] == terminal["token"]
        and owner["recovery_implementation_sha256"] == V1_SHA
        and terminal["status"] == "failed"
        and terminal["failure"]["reason"] == "Control fit budget exceeded54"
        and terminal["fit_guard_counters"]
        == dict(
            fixed_control_fit_calls=55,
            selected_model_fit_calls=0,
            search_calls=0,
            staging_calls=0,
            forbidden_calls=0,
        )
        and not (v1 / "recovery_result.json").exists(),
        "V1 is historical failed recovery, never a successful54-fit sidecar",
    )
    actual_owner = old.process_identity(owner["pid"], proc_root)
    old.require(
        actual_owner is None
        or actual_owner["state"] == "Z"
        or actual_owner["start_ticks"] != owner["process_identity"]["start_ticks"],
        "Original v1 recovery owner is still live",
    )
    old.verify_preserved(before)
    for relative in [
        "owner.json",
        "terminal_receipt.json",
        "before_snapshot.json",
        "failed_journal.original_bytes",
        "failed_controller_log.original_bytes",
    ]:
        old.require(
            old.digest(v1 / relative) == old.digest(v1_durable / relative),
            "V1 scratch/durable history differs",
        )
    template = Path(before["roots"]["template"])
    output, durable = (Path(before["roots"][key]) for key in ["output", "durable"])
    roots = {**before["roots"], "v1_recovery": str(v1), "v1_recovery_durable": str(v1_durable)}
    for key in ["recovery_root", "recovery_durable_root"]:
        path = Path(config[key]).resolve()
        old.require(
            all(
                path != Path(root)
                and not path.is_relative_to(Path(root))
                and not Path(root).is_relative_to(path)
                for root in roots.values()
            ),
            "V2 evidence roots must be outside all protected roots",
        )
    cfg = old.read(template / "followup_config.json")
    journal = old.read(output / "internal_campaign_status.json")
    old.require(
        journal["status"] == "requirement_incomplete"
        and journal["renal_annotation_reporting"]["status"] == "failed",
        "Original main failure journal must stay unchanged",
    )
    release = old.released_processes(
        dict(pid=3281577, start_ticks=95813586),
        template,
        [item["run_dir"] for item in before["rounds"]],
        proc_root=proc_root,
    )
    task = old.read(template / "task_manifest.json")["tasks"][0]
    proof = old.read(template / task["native_source_observation_coverage"]["proof_path"])
    old.require(
        old.digest(template / task["native_source_observation_coverage"]["proof_path"])
        == old.PROOF_SHA,
        "Frozen source proof changed",
    )
    closure = old.provenance_closure(template, proof)
    old.require(len(terminal["restored"]) == 84, "V1 must have restored all84 provenanced files")
    expected_destinations = {item["destination"] for item in before["restore_closure"]}
    old.require(
        {item["destination"] for item in terminal["restored"]} == expected_destinations,
        "V1 restored destination set differs from its reviewed plan",
    )
    for item in terminal["restored"]:
        old.require(
            item["absent_before"]
            and item["relative"] in closure
            and item["sha256"] == closure[item["relative"]]["sha256"]
            and old.digest(item["destination"]) == item["sha256"],
            "A v1 absent-only restored byte changed",
        )
    old.require(
        not (output / old.CAMPAIGN).exists() and not (durable / old.CAMPAIGN).exists(),
        "Campaign success report already exists; do not overwrite",
    )
    rounds, reusable, missing = [], [], []
    module = None
    for entry in before["rounds"]:
        run = Path(entry["run_dir"])
        products = [run / name for name in [old.CONTROL, old.REPORT]]
        mirrors = [durable / run.name / name for name in [old.CONTROL, old.REPORT]]
        exists = [path.exists() for path in [*products, *mirrors]]
        if any(exists):
            old.require(
                all(exists), "Incomplete saved seed-control/report mirror; do not recompute"
            )
            old.require(
                all(old.digest(a) == old.digest(b) for a, b in zip(products, mirrors, strict=True)),
                "Saved controls/reports differ from durable mirrors",
            )
            controls = old.read(products[0])
            old.require(
                controls["seed"] == entry["seed"]
                and set(controls["controls"]) == DESIGNS
                and all(
                    len(item["per_fold_fit_trace"]) == 6 for item in controls["controls"].values()
                ),
                "Saved control artifact must contain complete18 actual traces",
            )
            if saved_verifier is None:
                module = module or load_producer(template)
                report = verify_saved_seed(module, run, entry["seed"])
            else:
                report = saved_verifier(run, entry["seed"])
            old.require(
                report["status"] == "verified_internal_annotation_only",
                "Reusable saved report is not verified",
            )
            reusable.append(entry["seed"])
            action = "reuse_verified_existing_controls_and_report_without_writing"
        else:
            missing.append(entry["seed"])
            action = "compute_first_complete_saved_controls_then_report"
        rounds.append(
            {
                **entry,
                "recovery_action": action,
                "retained_control_traces": 18 if all(exists) else 0,
                "new_control_fit_budget": 0 if all(exists) else 18,
                "existing_product_sha256": {
                    str(path): old.digest(path) for path in [*products, *mirrors] if path.exists()
                },
            }
        )
    old.require(
        reusable == REUSED_SEEDS and missing == MISSING_SEEDS,
        "Bounded v2 requires saved seed42 and only missing2026/3407 controls",
    )
    files = {name: old.file_hashes(path) for name, path in roots.items()}
    files["failed_log"] = before["before_files"]["failed_log"]
    return dict(
        schema=SCHEMA,
        status="dry_run_reviewable_no_apply",
        roots=roots,
        before_files=files,
        original_failure=before["original_failure"],
        original_controller_release=release,
        v1_history=dict(
            owner=owner,
            terminal=terminal,
            before_snapshot_sha256=old.digest(v1 / "before_snapshot.json"),
            owner_sha256=old.digest(v1 / "owner.json"),
            terminal_sha256=old.digest(v1 / "terminal_receipt.json"),
            owner_released=True,
            actual_owner=actual_owner,
            restored_files_verified=84,
            historical_failure_preserved=True,
            raw_fit_frame_count55_not_estimator_fit_count=True,
            completed_outer_fits_inferred_from_two_frame_chain=27,
            retained_complete_control_traces=18,
            unpersisted_completed_controls_inferred=9,
            next_outer_fit_blocked_at_entry=True,
            inference_is_not_a_recovered_perfit_trace=True,
        ),
        source_provenance_closure=closure,
        restore_closure=[],
        rounds=rounds,
        reused_control_seeds=reusable,
        missing_control_seeds=missing,
        planned_new_control_fit_calls=36,
        final_retained_control_traces=54,
        allowed_necessary_recomputation_of_unpersisted_controls=9,
        planned_total_completed_controls_across_v1_v2_inferred=63,
        current_control_fit_calls=0,
        selected_model_fit_calls=0,
        search_calls=0,
        staging_calls=0,
        original_source_proof_sha256=old.PROOF_SHA,
        original_producer_sha256=old.PRODUCER_SHA,
        original_followup_config=cfg,
        historical_v1_implementation_sha256=V1_SHA,
        original_failure_journal_will_not_be_rewritten=True,
    )


def complete_missing_reports(module, snapshot):
    """Reuse the saved first seed; unchanged frozen numeric APIs handle missing seeds."""
    import joblib
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_internal_cv import verify_internal_task

    cfg = snapshot["original_followup_config"]
    durable = Path(cfg["durable_root"])
    reports = []
    for entry in snapshot["rounds"]:
        run, seed = Path(entry["run_dir"]), entry["seed"]
        if seed in snapshot["reused_control_seeds"]:
            reports.append(verify_saved_seed(module, run, seed))
            continue
        old.require(
            not (run / old.CONTROL).exists() and not (run / old.REPORT).exists(),
            "Never recompute an existing complete saved control artifact",
        )
        task = module.read(run / "task_manifest.json")["tasks"][0]
        values = module.rows(module.contained(run, task["table_path"]))
        module.verify_source_proof(run, task, values)
        features = module.read(run / "features/manifest.json")
        result_path = run / "results/internal_cv" / (old.TASK + ".json")
        result = module.read(result_path)
        verify_internal_task(task, values, features, run, result)
        payload = load_feature_route(values, features, result["selected_route"])
        model = joblib.load(result["artifacts"]["model"])
        check = next(
            c
            for c in module.read(run / "task_tables/predeclared_splits.json")["seeds"]
            if c["seed"] == seed
        )
        artifact = dict(
            source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
            seed=seed,
            selected_result_sha256=module.sha(result_path),
            frozen_check_sha256=module.object_hash(check),
            selected_estimator_provenance=module.capture_solver_provenance(model, result, payload),
            controls=module.fixed_ridge_controls(
                values, payload["X"][:, :-4], payload["X"][:, -4:], check
            ),
        )
        old.exclusive_json(run / old.CONTROL, artifact)
        reports.append(module.write_report(run, run_dir=run))
        for relative in [old.CONTROL, old.REPORT]:
            destination = durable / run.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write((run / relative).read_bytes())
            old.require(
                old.digest(destination) == old.digest(run / relative), "New durable report differs"
            )
    old.require(
        len(reports) == 3 and {r["seed"] for r in reports} == set(old.SEEDS),
        "All54 retained control traces and three reports are mandatory",
    )
    campaign = dict(
        schema=module.SCHEMA,
        status="verified_internal_annotation_only",
        reports=reports,
        warning=module.WARNING,
        additional_twenty_hour_loop=False,
    )
    for target in [Path(cfg["output_root"]), durable]:
        old.exclusive_json(target / old.CAMPAIGN, campaign)
    return campaign


def apply_recovery(config_path, review_path, review_sha):
    old.require(
        socket.gethostname() == "node23"
        and sys.executable == "/usr/bin/python3"
        and sys.flags.dont_write_bytecode,
        "Apply requires node23 /usr/bin/python3 -B",
    )
    old.require(old.digest(review_path) == review_sha, "Reviewed v2 snapshot changed")
    config, reviewed = old.read(config_path), old.read(review_path)
    old.require(
        reviewed["configuration_sha256"] == old.digest(config_path)
        and reviewed["recovery_implementation_sha256"] == old.digest(__file__),
        "V2 apply must match the exact reviewed code/configuration",
    )
    snapshot = inspect_recovery(config)
    for key in ["before_files", "rounds", "source_provenance_closure", "missing_control_seeds"]:
        old.require(snapshot[key] == reviewed[key], "Inputs changed after v2 review: " + key)
    root, durable = Path(config["recovery_root"]), Path(config["recovery_durable_root"])
    owner = dict(
        schema=SCHEMA,
        token=uuid.uuid4().hex,
        pid=os.getpid(),
        process_identity=old.process_identity(os.getpid()),
        status="running_report_recovery",
        started_at=time.time(),
        reviewed_sha256=review_sha,
        configuration_sha256=old.digest(config_path),
        recovery_implementation_sha256=old.digest(__file__),
        historical_v1_implementation_sha256=V1_SHA,
    )
    for dest in [root, durable]:
        old.exclusive_json(dest / "owner.json", owner)
        old.exclusive_json(dest / "before_snapshot.json", snapshot)
    counters, failure, status = {}, None, "failed"
    try:
        old.verify_preserved(snapshot)
        module = load_producer(snapshot["roots"]["template"])
        with report_fit_guard(
            module.__file__, snapshot["planned_new_control_fit_calls"]
        ) as counters:
            complete_missing_reports(module, snapshot)
        old.require(
            counters["fixed_control_fit_calls"] == counters["fixed_control_fit_completed"] == 36
            and counters["fixed_control_fit_attempts"] == 36
            and not any(
                counters[key]
                for key in [
                    "fixed_control_fit_failed",
                    "selected_model_fit_calls",
                    "search_calls",
                    "staging_calls",
                    "forbidden_calls",
                    "budget_exceeded_calls",
                ]
            ),
            "V2 requires exactly36 completed outermost fits and zero unauthorized calls",
        )
        old.verify_preserved(snapshot)
        generated = {}
        for entry in snapshot["rounds"]:
            for folder in [
                Path(entry["run_dir"]),
                Path(snapshot["roots"]["durable"]) / Path(entry["run_dir"]).name,
            ]:
                for relative in [old.CONTROL, old.REPORT]:
                    generated[str(folder / relative)] = old.digest(folder / relative)
        for folder in [Path(snapshot["roots"]["output"]), Path(snapshot["roots"]["durable"])]:
            generated[str(folder / old.CAMPAIGN)] = old.digest(folder / old.CAMPAIGN)
        after = {key: old.file_hashes(path) for key, path in snapshot["roots"].items()}
        for key, root_path in snapshot["roots"].items():
            new = set(after[key]) - set(snapshot["before_files"][key])
            old.require(
                all(str(Path(root_path) / relative) in generated for relative in new),
                "Unexpected artifact created during v2 recovery",
            )
        status = "verified_report_recovery_without_retraining"
        result = dict(
            schema=SCHEMA,
            status=status,
            owner=owner,
            v1_history=snapshot["v1_history"],
            before_files=snapshot["before_files"],
            after_files=after,
            fit_guard_counters=counters,
            generated_report_sha256=generated,
            reused_control_seeds=REUSED_SEEDS,
            newly_computed_control_seeds=MISSING_SEEDS,
            new_completed_control_fits=36,
            final_retained_control_traces=54,
            historical_v1_retained_control_traces=18,
            unpersisted_v1_completed_controls_inferred=9,
            total_completed_control_fits_across_v1_v2_inferred=63,
            all_original_fitted_search_contract_bytes_unchanged=True,
            original_failure_journal_status_retained="requirement_incomplete",
            first_seed_control_and_report_bytes_preserved=True,
            historical_failed_v1_all_bytes_preserved=True,
            no_retraining=True,
            finished_at=time.time(),
        )
        for dest in [root, durable]:
            old.exclusive_json(dest / "recovery_result.json", result)
        return result
    except BaseException as exc:
        failure = dict(exception_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        terminal = dict(
            schema=SCHEMA,
            token=owner["token"],
            pid=os.getpid(),
            status=status,
            finished_at=time.time(),
            failure=failure,
            fit_guard_counters=counters,
            owner_process_exits_after_receipt=True,
        )
        for dest in [root, durable]:
            old.exclusive_json(dest / "terminal_receipt.json", terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inspect", action="store_true")
    modes.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.inspect:
        config = old.read(args.config)
        old.require(
            args.output is not None
            and any(
                args.output.resolve().is_relative_to(Path(config[key]).resolve())
                for key in ["recovery_root", "recovery_durable_root"]
            ),
            "Write v2 dry-run evidence only in its own new recovery roots",
        )
        result = inspect_recovery(config)
        result.update(
            configuration_sha256=old.digest(args.config),
            recovery_implementation_sha256=old.digest(__file__),
        )
        old.exclusive_json(args.output, result)
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    reused_seeds=REUSED_SEEDS,
                    new_fit_budget=result["planned_new_control_fit_calls"],
                    review_sha256=old.digest(args.output),
                )
            )
        )
    else:
        old.require(args.review is not None and args.review_sha256, "Pinned v2 review is required")
        print(apply_recovery(args.config, args.review, args.review_sha256)["status"])


if __name__ == "__main__":
    main()
