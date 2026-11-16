"""Repair only HC50 report precision bookkeeping; never fit or alter predictions.

The 32-ULP ceiling is an engineering output-consistency budget, not a theorem
about BLAS error. It applies only to exact binary32 predictions from the frozen
linear/LSQR policy after duplicate chemical input rows are proved byte-identical.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import shutil
import socket
import struct
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import peptide_modeling_hc50_primary_bridge as legacy

LEGACY_SHA = "5bc218673c9cde599c24e7e66a2b9b3e11bd411254419f91b9bbbdfe97e6cb39"
POLICY = "byte_identical_linear_binary32_five_guard_bits_v1"
MAX_ULPS = 32
MIN_NORMAL_FLOAT32 = 2.0**-126
KNOWN_FAILURE = (
    "ValueError:One structure-only graph received inconsistent predictions within its role"
)
SCHEMA = "hc50_precision_report_repair_v1"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def bits32(value):
    require(math.isfinite(value), "Prediction is nonfinite")
    try:
        packed = struct.pack("!f", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError("Prediction is outside binary32 range") from exc
    require(struct.unpack("!f", packed)[0] == value, "Prediction is not exact binary32")
    bits = struct.unpack("!I", packed)[0]
    # Monotone signed ordering, with both zero encodings at the same position.
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits


def feature_evidence(source, payload):
    ids = [row["record_id"] for row in source]
    X = np.asarray(payload["X"])
    require(payload["record_ids"] == ids, "Selected feature record order changed")
    require(
        X.dtype == np.dtype("float32") and X.ndim == 2 and X.shape[0] == 34,
        "The precision exception requires the actual float32 chemical matrix",
    )
    valid = np.asarray(payload["valid"])
    require(
        len(payload["feature_names"]) == X.shape[1]
        and valid.shape == (34,)
        and valid.dtype == np.dtype("bool")
        and valid.all()
        and not np.isinf(X).any()
        and not np.isnan(X).all(axis=1).any(),
        "Selected feature schema or validity changed",
    )
    require(
        not any(
            "context:" in name or name.startswith("rbc_species_context:")
            for name in payload["feature_names"]
        ),
        "Context features are not structure-only inputs",
    )
    evidence, groups = {}, defaultdict(list)
    for index, row in enumerate(source):
        groups[row["split_group"]].append((index, row))
    for group, items in groups.items():
        graphs = {row["canonical_identity_group"] for _, row in items}
        require(len(graphs) == 1, "A precision group contains different graphs")
        first = X[items[0][0]].view(np.uint8)
        require(
            all(np.array_equal(first, X[index].view(np.uint8)) for index, _ in items),
            "Duplicate graph input rows are not byte-identical",
        )
        digest = hashlib.sha256(first.tobytes()).hexdigest()
        for _, row in items:
            evidence[row["record_id"]] = dict(
                split_group=group,
                graph=next(iter(graphs)),
                row_sha256=digest,
                feature_width=int(X.shape[1]),
                dtype="float32",
            )
    return dict(
        record_rows=evidence,
        selected_matrix_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        selected_mask_sha256=hashlib.sha256(np.asarray(payload["valid"]).tobytes()).hexdigest(),
        feature_names=list(payload["feature_names"]),
        feature_width=int(X.shape[1]),
    )


def prediction_consistency(source, group, values, predictions, evidence, model_context):
    require(bool(values) and len(values) == len(predictions), "Prediction group is empty")
    ids = [row["record_id"] for row in values]
    require(len(set(ids)) == len(ids), "Precision group repeats a record")
    require(all(math.isfinite(value) for value in predictions), "Prediction is nonfinite")
    bound = [evidence["record_rows"][record_id] for record_id in ids]
    require(
        all(
            item["split_group"] == group
            and item["graph"] == source[record_id]["canonical_identity_group"]
            for item, record_id in zip(bound, ids, strict=True)
        ),
        "Precision group/source binding changed",
    )
    require(
        len({item["graph"] for item in bound}) == len({item["row_sha256"] for item in bound}) == 1,
        "Precision allowance needs one graph and identical selected input bytes",
    )
    require(
        len({int(row["fold"]) for row in values}) == 1
        and len({row["evaluation_role"] for row in values}) == 1,
        "Precision comparison crossed folds or evaluation roles",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        original_pass = bool(np.allclose(predictions, predictions[0], rtol=1e-7, atol=1e-8))
    ulps = None
    try:
        ordered = [bits32(value) for value in predictions]
        ulps = max(ordered) - min(ordered)
    except ValueError:
        if not original_pass:
            raise
    mode = "legacy_float_tolerance"
    if not original_pass:
        policy = model_context.get("linear_solver_policy", {})
        require(
            model_context.get("selected_model") == "linear"
            and policy.get("version") == "dimension_based_linear_policy_v2"
            and policy.get("solver") == "lsqr"
            and policy.get("selection_basis") == "raw_route_feature_count_only"
            and policy.get("scaler_with_mean") is False
            and policy.get("feature_count") == evidence["feature_width"],
            "Precision exception is restricted to the frozen float32 linear LSQR policy",
        )
        require(
            all(abs(value) >= MIN_NORMAL_FLOAT32 for value in predictions)
            and len({math.copysign(1.0, value) for value in predictions}) == 1,
            "Precision exception requires normal binary32 values of one sign",
        )
        require(
            ulps is not None and ulps <= MAX_ULPS, "Binary32 prediction drift exceeds32ULP budget"
        )
        mode = "binary32_five_guard_bits"
    return dict(
        policy=POLICY,
        mode=mode,
        max_allowed_binary32_ulps=MAX_ULPS,
        engineering_budget_not_mathematical_BLAS_bound=True,
        observed_absolute_spread=max(predictions) - min(predictions),
        observed_binary32_ulp_distance=ulps,
        raw_predictions_unchanged=True,
        same_graph_and_fold=True,
        byte_identical_selected_chemical_inputs=True,
        selected_input_row_sha256=bound[0]["row_sha256"],
        selected_feature_width=evidence["feature_width"],
        source_record_ids=ids,
        canonical_identity_group=bound[0]["graph"],
    )


def evaluate_with_precision(source_rows, inner_oof, outer_holdout, payload, model_context):
    """Reuse the pinned evaluator with exactly its single constancy guard replaced."""
    require(sha(legacy.__file__) == LEGACY_SHA, "Original HC50 report implementation changed")
    evidence = feature_evidence(source_rows, payload)
    metadata = {}

    def guard(source, group, values, p):
        item = prediction_consistency(source, group, values, p, evidence, model_context)
        metadata[(values[0]["evaluation_role"], group)] = item
        return True

    tree = ast.parse(inspect.getsource(legacy.evaluation_diagnostics))
    old_call = ast.parse("np.allclose(p, p[0], rtol=1e-7, atol=1e-8)", mode="eval").body

    class ReplaceGuard(ast.NodeTransformer):
        replaced = 0

        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == ast.dump(
                old_call, include_attributes=False
            ):
                self.replaced += 1
                return ast.copy_location(
                    ast.parse(
                        "_bounded_precision_guard(source, group, values, p)", mode="eval"
                    ).body,
                    node,
                )
            return self.generic_visit(node)

    transform = ReplaceGuard()
    updated = transform.visit(copy.deepcopy(tree))
    require(transform.replaced == 1, "Pinned evaluator no longer has the single reviewed guard")
    ast.fix_missing_locations(updated)
    namespace = dict(legacy.__dict__)
    namespace["_bounded_precision_guard"] = guard
    exec(compile(updated, "<private_hc50_precision_evaluator>", "exec"), namespace)
    report = namespace["evaluation_diagnostics"](source_rows, inner_oof, outer_holdout)
    exceptional = []
    for role, result in report["evaluation_by_role"].items():
        for group in result["per_molecule"]:
            item = metadata[(role, group["split_group"])]
            group["prediction_consistency"] = item
            if item["mode"] == "binary32_five_guard_bits":
                exceptional.append(
                    dict(evaluation_role=role, split_group=group["split_group"], **item)
                )
    report["precision_consistency"] = dict(
        policy=POLICY,
        max_allowed_binary32_ulps=MAX_ULPS,
        engineering_budget_not_mathematical_BLAS_bound=True,
        groups_using_allowance=len(exceptional),
        exceptions=exceptional,
        selected_matrix_sha256=evidence["selected_matrix_sha256"],
        selected_mask_sha256=evidence["selected_mask_sha256"],
        feature_width=evidence["feature_width"],
        raw_predictions_unchanged=True,
        private_legacy_guard_replacements=1,
        legacy_evaluator_ast_sha256=object_sha(ast.dump(tree, include_attributes=False)),
        revised_evaluator_ast_sha256=object_sha(ast.dump(updated, include_attributes=False)),
    )
    return report


def evaluation_key(source, inner, holdout):
    def values(rows):
        return [
            (
                row["record_id"],
                row["split_group"],
                int(row["fold"]),
                float(row["target"]),
                float(row["prediction"]),
            )
            for row in rows
        ]

    return object_sha(dict(source=source, inner=values(inner), holdout=values(holdout)))


def prepare_report_plan(config_path, journal=None):
    """Read and verify existing artifacts and compile reports without writing or fitting."""
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features
    from peptide_modeling_summarize import verified_predictions

    require(sha(legacy.__file__) == LEGACY_SHA, "Original HC50 implementation changed")
    config_path = Path(config_path).resolve()
    config = legacy.read(config_path)
    root = Path(config["output_root"]).resolve()
    journal = journal or legacy.read(root / "campaign_status.json")
    require(
        journal.get("config_sha256") == sha(config_path),
        "Original controller configuration changed",
    )
    require(
        len(journal["rounds"]) == 3
        and {entry["seed"] for entry in journal["rounds"]} == set(legacy.SEEDS),
        "All three original fitted seeds are required",
    )
    reports, evidence, protected = (
        {},
        [],
        {str(config_path): sha(config_path), legacy.__file__: LEGACY_SHA},
    )
    for entry in journal["rounds"]:
        require(
            entry["status"] == "completed"
            and entry["audit"].get("method_model_coverage_met")
            and not entry["audit"].get("missing_completed_task_ids"),
            "A fitted seed is incomplete",
        )
        run = Path(entry["run_dir"]).resolve()
        manifest_path = run / "task_manifest.json"
        task = legacy.read(manifest_path)["tasks"][0]
        source_path = run / task["table_path"]
        source = legacy.rows(source_path)
        proof = legacy.verify_coverage_proof(run, task, source)
        result_path = run / "results/full" / (legacy.TASK + "_common.json")
        result = legacy.read(result_path)
        require(
            result["status"] == result["refit_status"] == "completed"
            and result["deployment_ready"] is True,
            "All three final refits must already be complete",
        )
        verified_predictions(run, run, result, task)
        verification_path = run / "verification.json"
        verification = legacy.read(verification_path)
        require(
            sha(verification_path) == entry["audit"]["verification_sha256"]
            and verification.get("status") == "verified"
            and verification.get("source_task_manifest_sha256") == sha(manifest_path),
            "Original seed verification changed",
        )
        paths = {
            name: legacy.artifact_path(run, value) for name, value in result["artifacts"].items()
        }
        require("refit_model" in paths, "Final all34 refit is missing")
        for path, receipt in result["artifact_hashes"].items():
            actual = legacy.artifact_path(run, path)
            require(sha(actual) == receipt["sha256"], "An original fitted artifact changed")
            protected[str(actual)] = receipt["sha256"]
        inputs, contract = (
            legacy.read(paths["input_contract"]),
            legacy.read(paths["model_contract"]),
        )
        split = legacy.read(paths["outer_split"])
        signature = dict(
            implementation_sha256=inputs["implementation_sha256"],
            search_implementation=inputs["search_implementation"],
            task=task,
            rows=[(row["record_id"], row["target"], row["split_group"]) for row in source],
            config=inputs["config"],
            features=inputs["feature_hashes"],
            linear_solver_policies=inputs["linear_solver_policies"],
            versions=inputs["versions"],
            outer_split=split["input_hash"],
        )
        require(
            object_sha(signature)
            == inputs["input_hash"]
            == contract["input_hash"]
            == result["input_hash"],
            "Frozen training-input signature changed",
        )
        require(
            inputs["implementation_sha256"] == sha(run / "scripts/peptide_modeling_train.py"),
            "Training implementation changed",
        )
        require(
            contract["common_record_ids"] == [r["record_id"] for r in source]
            and contract["optional_refit_fit_scope"] == "all_common_after_evaluation",
            "Final all34 membership changed",
        )
        features = portable_features(run)
        payload = load_feature_route(source, features, result["selected_route"])
        for relative, expected in features["artifact_hashes"].items():
            protected[str(Path(features["cache_dir"]) / relative)] = expected
        protected[str(run / "features/manifest.json")] = sha(run / "features/manifest.json")
        binding = feature_evidence(source, payload)
        expected = inputs["feature_hashes"][result["selected_route"]]
        require(
            expected["matrix_sha256"] == binding["selected_matrix_sha256"]
            and expected["mask_sha256"] == binding["selected_mask_sha256"]
            and expected["names"] == payload["feature_names"] == contract["feature_names"],
            "Selected input matrix/schema differs from fitted provenance",
        )
        inner, holdout = legacy.rows(paths["oof"]), legacy.rows(paths["holdout"])
        checks_path = legacy.contained(run, proof["predeclared_splits"]["path"])
        checks = legacy.read(checks_path)
        expected_check = next(item for item in checks["seeds"] if item["seed"] == entry["seed"])
        expected_folds = {
            rid: fold["fold"]
            for fold in expected_check["folds"]
            for rid in fold["validation_record_ids"]
        }
        require(
            {r["record_id"] for r in inner} == set(expected_check["train_record_ids"])
            and {r["record_id"] for r in holdout} == set(expected_check["holdout_record_ids"])
            and {r["record_id"]: int(r["fold"]) for r in inner} == expected_folds,
            "Actual evaluation membership differs from the frozen node plan",
        )
        context = dict(
            selected_model=result["selected_model"],
            linear_solver_policy=contract.get("linear_solver_policy"),
        )
        if result["selected_model"] == "linear":
            require(
                context["linear_solver_policy"]
                == inputs["linear_solver_policies"][result["selected_route"]],
                "Linear precision policy changed",
            )
        report = evaluate_with_precision(source, inner, holdout, payload, context)
        cache_key = evaluation_key(source, inner, holdout)
        require(
            cache_key not in reports, "Seed evaluation inputs are ambiguous for report delegation"
        )
        reports[cache_key] = report
        evidence.append(
            dict(
                seed=entry["seed"],
                selected_route=result["selected_route"],
                selected_model=result["selected_model"],
                numerical_context=context,
                result_sha256=sha(result_path),
                source_table_sha256=sha(source_path),
                input_contract_sha256=sha(paths["input_contract"]),
                input_hash=inputs["input_hash"],
                oof_sha256=sha(paths["oof"]),
                holdout_sha256=sha(paths["holdout"]),
                refit_model_sha256=sha(paths["refit_model"]),
                feature_evidence=binding,
                precision_consistency=report["precision_consistency"],
            )
        )
        for path in (
            result_path,
            manifest_path,
            source_path,
            checks_path,
            run / task["observation_endpoint_coverage"][0]["proof_path"],
            run / "verification.json",
            run / "scripts/peptide_modeling_train.py",
        ):
            protected[str(path)] = sha(path)
    return dict(
        config=config,
        root=root,
        journal=copy.deepcopy(journal),
        reports=reports,
        evidence=evidence,
        protected=protected,
        config_path=config_path,
    )


def unchanged(protected):
    require(
        all(Path(path).is_file() and sha(path) == expected for path, expected in protected.items()),
        "An original source/model/prediction artifact changed during reporting",
    )


def runtime_guard():
    require(
        os.name == "posix" and socket.gethostname() == "node23",
        "Report repair must run on its existing node23 host",
    )


def active_run_processes(root, process_identity, proc_root=Path("/proc")):
    root = str(Path(root).resolve())
    active = []
    if proc_root.is_dir():
        for directory in proc_root.iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            actual = process_identity(int(directory.name))
            if (
                actual
                and actual["state"] != "Z"
                and any(arg == root or arg.startswith(root + "/") for arg in actual.get("argv", []))
            ):
                active.append(dict(pid=actual["pid"], start_ticks=actual["start_ticks"]))
    return active


def repair_reports(config_path):
    """Apply only after root reviews this tool; no model/trial entrypoint is called."""
    from peptide_modeling_campaign import process_identity
    from peptide_modeling_supervise import atomic_json

    runtime_guard()
    plan = prepare_report_plan(config_path)
    root, config, journal = plan["root"], plan["config"], plan["journal"]
    journal_path = root / "campaign_status.json"
    original = journal_path.read_bytes()
    require(
        journal["status"] == "requirement_incomplete"
        and journal.get("molecular_reporting", {}).get("reason") == KNOWN_FAILURE,
        "Only the identified completed-training precision-report failure can be repaired",
    )
    identity = journal["process_identity"]
    actual = process_identity(identity["pid"])
    require(
        actual is None
        or actual["state"] == "Z"
        or actual["start_ticks"] != identity["start_ticks"],
        "Original report controller is still live",
    )
    for entry in journal["rounds"]:
        actual = process_identity(entry["pid"])
        expected = entry.get("process_identity", {})
        require(
            actual is None
            or actual["state"] == "Z"
            or actual["start_ticks"] != expected.get("start_ticks"),
            "Original seed supervisor is still live",
        )
    require(
        not active_run_processes(root, process_identity), "A process still owns this run namespace"
    )
    require(
        not (root / "hc50_primary_molecular_diagnostics.json").exists(),
        "An existing mandatory report requires separate review",
    )
    unchanged(plan["protected"])
    token = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    recovery = root / "report_recovery" / ("hc50_precision_" + token)
    recovery.mkdir(parents=True, exist_ok=False)
    backup = recovery / "failed_campaign_status.json"
    backup.write_bytes(original)
    evidence_path = recovery / "precision_evidence.json"
    evidence = dict(
        schema=SCHEMA,
        policy=POLICY,
        max_allowed_binary32_ulps=MAX_ULPS,
        engineering_budget_not_mathematical_BLAS_bound=True,
        legacy_delegate_sha256=LEGACY_SHA,
        implementation_sha256=sha(__file__),
        config_sha256=sha(plan["config_path"]),
        prior_journal_sha256=hashlib.sha256(original).hexdigest(),
        protected_artifact_hashes=plan["protected"],
        reports=plan["evidence"],
        production_model_fits=0,
        model_selection_repeated=False,
        raw_predictions_modified=False,
    )
    atomic_json(evidence_path, evidence)
    require(
        journal_path.read_bytes() == original, "Controller journal changed before report adoption"
    )
    journal.update(
        status="reporting_precision_repair",
        pid=os.getpid(),
        process_identity=process_identity(os.getpid()),
    )
    journal.setdefault("recovery_history", []).append(
        dict(
            kind=SCHEMA,
            prior_journal_sha256=evidence["prior_journal_sha256"],
            backup=str(backup),
            evidence_path=str(evidence_path),
            evidence_sha256=sha(evidence_path),
            source_model_prediction_changes=False,
            no_model_fit=True,
        )
    )
    durable = Path(config["durable_root"])
    durable.mkdir(parents=True, exist_ok=True)
    atomic_json(journal_path, journal)
    atomic_json(durable / journal_path.name, journal)
    old_evaluator = legacy.evaluation_diagnostics

    def verified_cached_evaluator(source, inner, holdout):
        key = evaluation_key(source, inner, holdout)
        require(key in plan["reports"], "Delegate requested unverified prediction inputs")
        return copy.deepcopy(plan["reports"][key])

    try:
        legacy.evaluation_diagnostics = verified_cached_evaluator
        delegated = legacy.write_diagnostics(config_path, journal)
        unchanged(plan["protected"])
        path = Path(delegated["path"])
        payload = legacy.read(path)
        revision = dict(
            schema=SCHEMA,
            policy=POLICY,
            max_allowed_binary32_ulps=MAX_ULPS,
            engineering_budget_not_mathematical_BLAS_bound=True,
            implementation_sha256=sha(__file__),
            legacy_delegate_sha256=LEGACY_SHA,
            evidence_path=str(evidence_path),
            evidence_sha256=sha(evidence_path),
            failed_journal_backup=str(backup),
            failed_journal_sha256=evidence["prior_journal_sha256"],
            model_selection_repeated=False,
            production_model_fits=0,
            source_model_prediction_files_changed=False,
            raw_predictions_modified=False,
        )
        payload["precision_reporting_revision"] = revision
        atomic_json(path, payload)
        destination = Path(delegated["durable_path"])
        shutil.copy2(path, destination)
        require(sha(path) == sha(destination), "Durable precision report differs")
        receipt = {
            **delegated,
            "sha256": sha(path),
            "durable_sha256": sha(destination),
            "implementation_sha256": sha(__file__),
            "precision_reporting_revision": revision,
            "implementation_dependencies_sha256": {
                **delegated["implementation_dependencies_sha256"],
                "peptide_modeling_hc50_primary_bridge.py": LEGACY_SHA,
            },
        }
        journal["molecular_reporting"] = receipt
        journal["status"] = "completed"
        unchanged(plan["protected"])
        atomic_json(journal_path, journal)
        atomic_json(durable / journal_path.name, journal)
        evidence_durable = durable / "report_recovery" / recovery.name
        shutil.copytree(recovery, evidence_durable)
        require(
            all(
                sha(p) == sha(evidence_durable / p.relative_to(recovery))
                for p in recovery.rglob("*")
                if p.is_file()
            ),
            "Durable precision evidence differs",
        )
        return receipt
    except BaseException as exc:
        journal["status"] = "requirement_incomplete"
        journal["molecular_reporting"] = dict(
            status="failed",
            reason=f"{type(exc).__name__}:{exc}",
            precision_repair=True,
            evidence_path=str(evidence_path),
        )
        atomic_json(journal_path, journal)
        atomic_json(durable / journal_path.name, journal)
        raise
    finally:
        legacy.evaluation_diagnostics = old_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        plan = prepare_report_plan(args.config)
        print(
            json.dumps(
                dict(
                    status="verified_readonly",
                    evidence=plan["evidence"],
                    original_files_verified=len(plan["protected"]),
                    production_model_fits=0,
                )
            )
        )
    else:
        print(json.dumps(repair_reports(args.config)))


if __name__ == "__main__":
    main()
