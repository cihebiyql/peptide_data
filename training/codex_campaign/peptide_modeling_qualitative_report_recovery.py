"""Recover saved qualitative reports; preserve learning code and seed42 fitted bytes.

No action runs on import. ``--inspect`` reads pinned saved artifacts and builds the
report in memory. ``--run-recovery`` requires a reviewed, pinned recovery config;
it publishes seed42 reports and trains only fresh seed2026/3407 directories.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = "qualitative_zero_mass_ece_report_recovery_v1"
LEGACY_SHA256 = "a90f6aebfa3ad2c7409540c410c2c227504193d7cf9ad0fd7c43204453abe7ae"
METRIC_STATUS = "zero_effective_weight_calibration_bins_skipped"
METRIC_KEYS = ("pr_auc", "roc_auc", "brier", "ece_fixed_10_bins")


def digest(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def file_hashes(root):
    root = Path(root)
    values = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Recovery evidence cannot contain symlinks")
        if path.is_file():
            values[path.relative_to(root).as_posix()] = digest(path)
    return values


def weighted_metrics(y, prediction, weights, *, api=None):
    """Preserve old arithmetic, omitting only calibration bins with zero mass."""
    if api is None:
        from sklearn import metrics as api

    y, prediction, weights = (np.asarray(value, dtype=float) for value in (y, prediction, weights))
    if (
        y.ndim != 1
        or y.shape != prediction.shape
        or y.shape != weights.shape
        or not len(y)
        or not np.isfinite([y, prediction, weights]).all()
        or not np.isin(y, [0, 1]).all()
        or ((prediction < 0) | (prediction > 1)).any()
        or (weights < 0).any()
        or not math.isfinite(float(weights.sum()))
        or weights.sum() <= 0
        or len(set(y[weights > 0])) != 2
    ):
        raise ValueError(
            "Finite aligned binary predictions and valid positive effective weights required"
        )
    bins = np.minimum((prediction * 10).astype(int), 9)
    calibration = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any() and weights[mask].sum() > 0:
            calibration += (
                weights[mask].sum()
                / weights.sum()
                * abs(
                    np.average(y[mask], weights=weights[mask])
                    - np.average(prediction[mask], weights=weights[mask])
                )
            )
    return {
        "pr_auc": float(api.average_precision_score(y, prediction, sample_weight=weights)),
        "roc_auc": float(api.roc_auc_score(y, prediction, sample_weight=weights)),
        "brier": float(api.brier_score_loss(y, prediction, sample_weight=weights)),
        "ece_fixed_10_bins": float(calibration),
    }


@contextlib.contextmanager
def report_override(legacy):
    """Only the reporting helper changes; fitted functions and source pins do not."""
    previous = legacy._weighted_metrics
    legacy._weighted_metrics = lambda y, p, w: weighted_metrics(y, p, w, api=legacy)
    try:
        yield
    finally:
        legacy._weighted_metrics = previous


def load_legacy(template):
    template = Path(template).resolve()
    source = template / "scripts/peptide_modeling_qualitative_cyto.py"
    if digest(source) != LEGACY_SHA256:
        raise ValueError("The frozen learning implementation must remain a90f6a")
    sys.path.insert(0, str(source.parent))
    legacy = importlib.import_module("peptide_modeling_qualitative_cyto")
    if Path(legacy.__file__).resolve() != source:
        raise ValueError("A different qualitative module is already imported")
    return legacy


def capture_verification(legacy, rows, features, splits, seed_root, result):
    captured = []
    previous = legacy.atomic_json

    def capture(path, value):
        if Path(path).resolve() != (Path(seed_root) / "verification.json").resolve():
            raise ValueError("The report-only verifier attempted an unexpected write")
        captured.append(value)

    legacy.atomic_json = capture
    try:
        legacy.verify_seed(rows, features, splits, seed_root, result)
    finally:
        legacy.atomic_json = previous
    if len(captured) != 1 or captured[0].get("model_reload_verified") is not True:
        raise ValueError("All saved models must reload to their saved OOF predictions")
    return captured[0]


def saved_result(legacy, template, seed_root, training_config, *, expected_files=None):
    """Assemble seed42 from saved predictions. Never call train_seed or _fit_arm."""
    template, seed_root = Path(template), Path(seed_root)
    before = file_hashes(seed_root)
    if expected_files is not None and before != expected_files:
        raise ValueError("Saved seed42 evidence changed before report assembly")
    if (seed_root / "result.json").exists() or (seed_root / "verification.json").exists():
        raise FileExistsError("Existing seed42 report must not be overwritten")
    if training_config.get("seed", 42) != 42:
        raise ValueError("Only the already fitted seed42 can use saved-report recovery")
    config = legacy.read_json(template / "followup_config.json")
    rows = legacy.normalize_rows(legacy.load_rows(template / config["table_path"]))
    features = legacy.portable_features(template)
    splits = legacy.read_json(template / config["split_evidence_path"])
    settings = legacy.training._settings(training_config)
    legacy._verify_artifacts(features)
    loaded, common, coverage, excluded = legacy.training.prepare_routes(
        rows, features, {**settings, "min_rows": len(rows)}
    )
    if not loaded or not common.all() or excluded:
        raise ValueError("Every original feature method must retain every source record")
    folds = legacy.read_json(seed_root / "fold_progress.json")
    if len(folds) != 5 or [fold["fold"] for fold in folds] != list(range(5)):
        raise ValueError("Recovery requires all five completed saved outer folds")
    output = seed_root / "outer_oof.tsv"
    oof = legacy.load_rows(output)
    predictions = {arm: np.asarray([float(row[arm]) for row in oof]) for arm in legacy.ARMS}
    result = dict(
        status="completed",
        protocol=legacy.PROTOCOL,
        warning=legacy.WARNING,
        seed=42,
        actual_outer_folds=5,
        actual_inner_folds_per_outer=5,
        n_records=len(rows),
        n_graphs=len({row["structure_key"] for row in rows}),
        n_groups=len({row["split_group"] for row in rows}),
        coverage=coverage,
        source_rows_hash=legacy.training._digest(rows),
        feature_fingerprint=features["input_fingerprint"],
        folds=folds,
        outer_oof_sha256=digest(output),
        outer_scores_used_to_choose_arm=False,
        deployment_model_fitted=False,
        independent_external_validation=False,
        context_counted_as_chemical_method=False,
        outer_evaluation_includes_unseen_contexts_for_diagnostics=True,
        operational_inference_abstains_on_unseen_context=True,
    )
    verification = capture_verification(legacy, rows, features, splits, seed_root, result)
    with report_override(legacy):
        result["performance"] = {
            arm: legacy.performance(rows, prediction, 42) for arm, prediction in predictions.items()
        }
    provenance = dict(
        schema=SCHEMA,
        report_revision=SCHEMA,
        empty_support_metric_status=METRIC_STATUS,
        learning_implementation_sha256=digest(legacy.__file__),
        training_implementation_sha256=digest(legacy.training.__file__),
        reporting_implementation_sha256=digest(__file__),
        seed42_reused_all_saved_models=True,
        production_model_fits=0,
        candidate_searches=0,
        outer_oof_rewritten=False,
        fold_progress_rewritten=False,
    )
    result["report_recovery"] = provenance
    verification["report_recovery"] = provenance
    if file_hashes(seed_root) != before:
        raise ValueError("Report assembly changed saved production evidence")
    return result, verification


def validate_adoption(legacy, config, evidence):
    if config.get("schema") != SCHEMA or config.get("helper_sha256") != digest(__file__):
        raise ValueError("Recovery implementation/config is not pinned")
    if config.get("remaining_seeds") != [2026, 3407]:
        raise ValueError("Only fresh seeds2026/3407 may perform future learning")
    recovery_id = config.get("recovery_id", "")
    if not recovery_id or Path(recovery_id).name != recovery_id or recovery_id in {".", ".."}:
        raise ValueError("Recovery history requires one safe directory name")
    template, root = Path(config["template"]), Path(config["run_root"])
    if digest(template / "followup_config.json") != config["legacy_config_sha256"]:
        raise ValueError("The original execution and training configuration changed")
    if digest(config["evidence_path"]) != config["evidence_sha256"]:
        raise ValueError("Original read-only evidence pin changed")
    journal_path = root / "qualitative_status.json"
    if digest(journal_path) != evidence["journal_sha256"]:
        raise ValueError("Failed journal changed; recovery adoption needs a new decision")
    journal = read(journal_path)
    if journal.get("status") != "failed" or not journal.get("reason", "").startswith(
        "ZeroDivisionError:Weights sum to zero"
    ):
        raise ValueError("Expected the specific report-only failure")
    if (
        evidence.get("status") != "verified_readonly_fitted_report_failure"
        or evidence.get("models") != 20
        or evidence.get("selected_model_contracts") != 15
        or len(evidence.get("signatures", [])) != 15
    ):
        raise ValueError("The independently checked fitted artifact census is required")
    for proc in Path("/proc").glob("[0-9]*"):
        current = legacy.process_identity(int(proc.name))
        if (
            current
            and current["pid"] != os.getpid()
            and current["state"] != "Z"
            and any(str(template) in arg or str(root) in arg for arg in current["argv"])
        ):
            raise RuntimeError("An old or competing production handle is still live")
    legacy.verify_inputs(template, evidence["original_input_files_sha256"])
    if file_hashes(root / "seed_42") != evidence["file_sha256"]:
        raise ValueError("Saved model/prediction/search evidence changed")
    if any((root / f"seed_{seed}").exists() for seed in config["remaining_seeds"]):
        raise FileExistsError("Fresh seed directories are required; no automatic refit/restart")
    original = read(template / "followup_config.json")
    if (
        Path(original["template_run"]).resolve() != template.resolve()
        or Path(original["output_root"]).resolve() != root.resolve()
    ):
        raise ValueError("The recovery cannot relocate the original learning or output roots")
    if original["required_seeds"] != [42, 2026, 3407] or original["training"]["n_jobs"] != 1:
        raise ValueError("Original bounded three-seed/one-thread contract changed")
    if (Path(original["durable_root"]) / "seed_42").exists():
        raise FileExistsError("An existing durable seed42 archive needs a separate reconciliation")
    return journal, original


def snapshot_fitted(root, template, destination, evidence):
    """Create a snapshot NOW; never describe it as a preexisting seed archive."""
    root, template, destination = map(Path, (root, template, destination))
    destination.mkdir(parents=True, exist_ok=False)
    for prefix, source, files in (
        ("seed_42", root / "seed_42", evidence["file_sha256"]),
        ("template", template, evidence["original_input_files_sha256"]),
    ):
        for name, expected in files.items():
            target = destination / prefix / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
            if digest(target) != expected:
                raise ValueError("Pre-repair fitted snapshot copy failed its exact SHA")
    shutil.copy2(root / "qualitative_status.json", destination / "failed_journal.json")
    exclusive_json(
        destination / "snapshot_receipt.json",
        dict(
            schema="qualitative_fitted_snapshot_created_now_v1",
            created_now_not_prior_archive=True,
            created_at=time.time(),
            seed_files=len(evidence["file_sha256"]),
            template_files=len(evidence["original_input_files_sha256"]),
            models=20,
            selected_contracts=15,
            failed_journal_sha256=evidence["journal_sha256"],
        ),
    )


def run_recovery(config_path):
    """Explicit apply entrypoint. No caller may run it as a metadata preflight."""
    config_path = Path(config_path).resolve()
    config, helper_sha = read(config_path), digest(__file__)
    config_sha = digest(config_path)
    if os.name != "posix" or socket.gethostname() != "node23":
        raise ValueError("Production recovery requires its declared node23 runtime")
    if any(
        os.environ.get(key) != "1"
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("The bounded recovery process must retain all thread limits at one")
    template, root = Path(config["template"]), Path(config["run_root"])
    legacy = load_legacy(template)
    evidence = read(config["evidence_path"])
    old_journal, original = validate_adoption(legacy, config, evidence)
    durable = Path(original["durable_root"])
    parent = legacy.parent_ready(original)
    if not parent:
        raise RuntimeError("The original verified parent must have released its processes")
    history = root / "recovery_history" / config["recovery_id"]
    token = uuid.uuid4().hex
    owner = dict(
        pid=os.getpid(),
        token=token,
        process_identity=legacy.process_identity(os.getpid()),
        helper_sha256=helper_sha,
    )
    ownership_path = root / "qualitative_report_recovery.lock"
    exclusive_json(ownership_path, owner)
    history.mkdir(parents=True, exist_ok=False)
    exclusive_json(history / "owner.json", owner)
    snapshot_fitted(root, template, history / "fitted_snapshot_created_now", evidence)
    result, verification = saved_result(
        legacy,
        template,
        root / "seed_42",
        original["training"],
        expected_files=evidence["file_sha256"],
    )
    if digest(root / "qualitative_status.json") != evidence["journal_sha256"]:
        raise ValueError("Failed journal compare-and-swap guard changed before adoption")
    exclusive_json(root / "seed_42/result.json", result)
    exclusive_json(root / "seed_42/verification.json", verification)
    source_proof = dict(
        schema=SCHEMA,
        helper_sha256=helper_sha,
        config_sha256=config_sha,
        legacy_learning_sha256=LEGACY_SHA256,
        old_failed_pid=old_journal["pid"],
        old_failure_reason=old_journal["reason"],
        old_journal_sha256=evidence["journal_sha256"],
        pre_repair_snapshot_created_now=str(history / "fitted_snapshot_created_now"),
        recovered_seed42_result=str(root / "seed_42/result.json"),
        result_sha256=digest(root / "seed_42/result.json"),
        seed42_training_calls=0,
        remaining_seeds=[2026, 3407],
        runtime_override_only="_weighted_metrics",
        empty_support_metric_status=METRIC_STATUS,
        owner_token=token,
    )
    exclusive_json(history / "repair_proof.json", source_proof)
    _, methods = legacy.representation_registry(original["registry_path"])
    journal = dict(old_journal)
    journal.pop("reason", None)
    journal.update(
        status="running",
        pid=os.getpid(),
        process_identity=legacy.process_identity(os.getpid()),
        report_revision=SCHEMA,
        report_recovery=source_proof,
        parent_requirement_evidence=parent,
        recovery_history=[*old_journal.get("recovery_history", []), source_proof],
    )
    journal["rounds"] = [
        dict(
            seed=42,
            status="completed",
            report_only_recovery=True,
            verification=verification,
            result_sha256=digest(root / "seed_42/result.json"),
            method_model_evidence=legacy.method_model_coverage(result, methods),
        )
    ]

    adopted = False

    def check_owner():
        current = read(history / "owner.json")
        if (
            current.get("token") != token
            or read(ownership_path).get("token") != token
            or digest(__file__) != helper_sha
            or digest(config_path) != config_sha
        ):
            raise ValueError("Recovery ownership or runtime pins changed")
        legacy.verify_inputs(template, evidence["original_input_files_sha256"])
        current_seed = file_hashes(root / "seed_42")
        if any(
            current_seed.get(name) != expected for name, expected in evidence["file_sha256"].items()
        ):
            raise ValueError("Protected seed42 model/OOF/fold/search bytes changed during recovery")
        active = read(root / "qualitative_status.json")
        if adopted:
            if (
                active.get("pid") != os.getpid()
                or active.get("report_recovery", {}).get("owner_token") != token
            ):
                raise ValueError("The adopted production journal belongs to another recovery owner")
        elif digest(root / "qualitative_status.json") != evidence["journal_sha256"]:
            raise ValueError("Failed journal compare-and-swap changed before first publication")

    def save():
        nonlocal adopted
        check_owner()
        legacy.atomic_json(root / "qualitative_status.json", journal)
        adopted = True
        legacy.atomic_json(durable / "qualitative_status.json", journal)

    def archive(seed):
        source, target = root / f"seed_{seed}", durable / f"seed_{seed}"
        shutil.copytree(source, target)
        if file_hashes(source) != file_hashes(target):
            raise ValueError("Durable seed archive differs from exact fitted/report bytes")

    os.nice(10)
    save()
    archive(42)
    rows = legacy.load_rows(template / original["table_path"])
    features = legacy.portable_features(template)
    splits = legacy.read_json(template / original["split_evidence_path"])
    settings = legacy.training._settings(original["training"])
    try:
        with report_override(legacy):
            for seed in config["remaining_seeds"]:
                check_owner()
                run = root / f"seed_{seed}"
                if run.exists():
                    raise FileExistsError("Never refit an existing production seed")
                entry = dict(seed=seed, status="running")
                journal["rounds"].append(entry)
                save()
                with legacy.threadpool_limits(limits=1):
                    future = legacy.train_seed(
                        rows, features, splits, run, {**settings, "seed": seed}
                    )
                exclusive_json(
                    run / "report_recovery.json",
                    dict(
                        schema=SCHEMA,
                        report_revision=SCHEMA,
                        empty_support_metric_status=METRIC_STATUS,
                        learning_implementation_sha256=LEGACY_SHA256,
                        reporting_implementation_sha256=helper_sha,
                        result_sha256=digest(run / "result.json"),
                        verification_sha256=digest(run / "verification.json"),
                        seed=seed,
                        original_learning_executed=True,
                        runtime_override_only="_weighted_metrics",
                    ),
                )
                entry.update(
                    status="completed",
                    verification=read(run / "verification.json"),
                    result_sha256=digest(run / "result.json"),
                    method_model_evidence=legacy.method_model_coverage(future, methods),
                )
                archive(seed)
                save()
        summary = dict(
            protocol=legacy.PROTOCOL,
            warning=legacy.WARNING,
            seeds=[42, 2026, 3407],
            arm_selection_from_outer_scores=False,
            independent_external_validation=False,
            report_revision=SCHEMA,
            report_recovery=source_proof,
            scores={
                str(seed): read(root / f"seed_{seed}/result.json")["performance"]
                for seed in [42, 2026, 3407]
            },
        )
        legacy.atomic_json(root / "performance_summary.json", summary)
        legacy.atomic_json(durable / "performance_summary.json", summary)
        journal["status"] = (
            "completed"
            if all(e["method_model_evidence"]["requirements_met"] for e in journal["rounds"])
            else "requirement_incomplete"
        )
        save()
        return journal
    except BaseException as exc:
        journal.update(status="failed", reason=f"{type(exc).__name__}:{exc}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--run-recovery", action="store_true")
    args = parser.parse_args()
    if args.run_recovery:
        print(json.dumps(run_recovery(args.config)))
    else:
        config = read(args.config)
        legacy = load_legacy(config["template"])
        evidence = read(config["evidence_path"])
        _, original = validate_adoption(legacy, config, evidence)
        result, verification = saved_result(
            legacy,
            config["template"],
            Path(config["run_root"]) / "seed_42",
            original["training"],
            expected_files=evidence["file_sha256"],
        )
        print(
            json.dumps(
                dict(
                    status="verified_report_only_in_memory",
                    result=result,
                    verification=verification,
                )
            )
        )


if __name__ == "__main__":
    main()
