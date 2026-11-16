"""Verify saved predictions and artifacts without unpickling cross-version models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

NUMERIC_POLICY = "finite_transformed_metrics_with_explicit_unrepresentable_native_v1"


class VerificationError(ValueError):
    """A saved run does not satisfy its declared evidence contract."""


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_table(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(path, fields, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def resolve_artifact(run_dir, remote_root, artifact):
    root = Path(run_dir).resolve()
    raw = str(artifact).replace("\\", "/")
    remote = PurePosixPath(str(remote_root).replace("\\", "/"))
    path = PurePosixPath(raw)
    if ".." in path.parts:
        raise VerificationError("Artifact path contains parent traversal")
    native = Path(artifact)
    if native.is_absolute() and native.resolve().is_relative_to(root):
        return native.resolve()
    if path.is_absolute() or (path.parts and path.parts[0].endswith(":")):
        try:
            relative = path.relative_to(remote)
        except ValueError as exc:
            # Also permit a run already generated on this machine.
            local = Path(artifact).resolve()
            if not local.is_relative_to(root):
                raise VerificationError("Artifact lies outside the declared run") from exc
            return local
    else:
        relative = path
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root):
        raise VerificationError("Artifact resolves outside the declared run")
    return result


def metrics(target, prediction, kind):
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if len(target) == 0 or target.shape != prediction.shape:
        raise VerificationError("Empty or inconsistent prediction arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise VerificationError("Nonfinite target or prediction")
    if kind == "classification":
        if set(target) != {0, 1} or np.any((prediction < 0) | (prediction > 1)):
            raise VerificationError(
                "Classification holdout needs both labels and valid probabilities"
            )
        return {
            "pr_auc": float(average_precision_score(target, prediction)),
            "roc_auc": float(roc_auc_score(target, prediction)),
            "mcc": float(matthews_corrcoef(target, prediction >= 0.5)),
            "brier": float(brier_score_loss(target, prediction)),
        }
    if kind != "regression":
        raise VerificationError("Unknown task kind")
    rho = (
        float(spearmanr(target, prediction).statistic)
        if (len(set(target)) > 1 and len(set(prediction)) > 1)
        else float("nan")
    )
    return {
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "spearman": rho if math.isfinite(rho) else None,
    }


def compare_metrics(recomputed, recorded, description):
    for key, actual in recomputed.items():
        if key not in recorded:
            raise VerificationError(f"Missing {description} metric: {key}")
        expected = recorded[key]
        if actual is None and (expected is None or not math.isfinite(float(expected))):
            continue
        if (
            expected is None
            or actual is None
            or not math.isclose(actual, float(expected), rel_tol=1e-8, abs_tol=1e-10)
        ):
            raise VerificationError(
                f"Inconsistent {description} metric {key}: {actual} vs {expected}"
            )


def native_metrics(records, transform, unit):
    """Report the entire cohort, without clipping or deleting extreme predictions.

    Saved transformed predictions remain the evidence. An inverse transform that
    cannot be represented by float64 is recorded per row, and makes the whole
    cohort's original-unit error metrics unavailable. Finite original-unit
    values use scaled error arithmetic to avoid unnecessary square overflow.
    """
    if transform not in {"identity", "log10"}:
        raise VerificationError("Unsupported inverse target transform")
    target = np.asarray([float(row["target"]) for row in records])
    prediction = np.asarray([float(row["prediction"]) for row in records])
    if not len(records) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise VerificationError("Nonfinite saved prediction or target")
    if transform == "log10":
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            native_target = np.power(10.0, target)
            native_prediction = np.power(10.0, prediction)
    else:
        native_target, native_prediction = target, prediction

    def inverse_status(value):
        if not math.isfinite(value):
            return "overflow"
        if transform == "log10" and value == 0:
            return "underflow_to_zero"
        return "finite"

    issues = []
    for index, row in enumerate(records):
        left, right = float(native_target[index]), float(native_prediction[index])
        left_status, right_status = inverse_status(left), inverse_status(right)
        if left_status != "finite" or right_status != "finite":
            issues.append(
                {
                    "record_id": row["record_id"],
                    "target_transformed": float(target[index]),
                    "prediction_transformed": float(prediction[index]),
                    "target_native": left if left_status == "finite" else None,
                    "prediction_native": right if right_status == "finite" else None,
                    "target_inverse_status": left_status,
                    "prediction_inverse_status": right_status,
                }
            )
    result = {
        "native_mae": None,
        "native_rmse": None,
        "native_unit": unit,
        "native_metric_status": "unavailable_inverse_range" if issues else "available",
        "native_unrepresentable_rows": len(issues),
        "native_total_rows": len(records),
        "native_issues": issues,
        "native_metric_range_issues": [],
        "quantitative_use_restricted": bool(issues),
        "native_metrics_use_entire_cohort": True,
    }
    if issues:
        return result
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        difference = native_target - native_prediction
    finite_difference = bool(np.isfinite(difference).all())
    if finite_difference:
        # Scale errors rather than absolute levels: a large exactly-correct row
        # must not erase a small nonzero error on another row.
        scale = float(np.max(np.abs(difference)))
    else:
        # Opposite-sign finite inputs can have an unrepresentable difference,
        # while their mean/RMS error across a cohort can still be representable.
        scale = float(max(np.max(np.abs(native_target)), np.max(np.abs(native_prediction))))
        difference = native_target / scale - native_prediction / scale
    if scale == 0:
        mae, rmse = 0.0, 0.0
    else:
        if finite_difference:
            difference = difference / scale
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            mae = float(np.mean(np.abs(difference)) * scale)
            rmse = float(np.sqrt(np.mean(difference**2)) * scale)
    unavailable = [
        name for name, value in (("mae", mae), ("rmse", rmse)) if not math.isfinite(value)
    ]
    if unavailable:
        result.update(
            native_metric_status="unavailable_metric_range",
            native_metric_range_issues=unavailable,
            quantitative_use_restricted=True,
        )
    else:
        result.update(native_mae=mae, native_rmse=rmse)
    return result


def verified_predictions(run_dir, remote_root, result, task):
    artifact_receipts = result.get("artifact_hashes", {})
    if not artifact_receipts:
        raise VerificationError("Completed result has no artifact hashes")
    checked = 0
    for artifact, receipt in artifact_receipts.items():
        path = resolve_artifact(run_dir, remote_root, artifact)
        if not path.is_file() or file_sha(path) != receipt["sha256"]:
            raise VerificationError("Artifact missing or changed: " + str(path))
        checked += 1
    required = {"model", "oof", "holdout", "outer_split", "model_contract"}
    artifacts = result.get("artifacts", {})
    if not required <= set(artifacts):
        raise VerificationError("Completed result lacks required model/prediction contracts")
    if any(artifacts[k] not in artifact_receipts for k in required):
        raise VerificationError("Required artifact is not covered by a hash")
    paths = {k: resolve_artifact(run_dir, remote_root, v) for k, v in artifacts.items()}
    contract = json.loads(paths["model_contract"].read_text(encoding="utf-8"))
    if not contract.get("holdout_not_used_for_selection"):
        raise VerificationError("Holdout selection independence is not declared")
    if contract.get("input_hash") != result.get("input_hash"):
        raise VerificationError("Model and result input hashes differ")
    source_rows = read_table(Path(run_dir) / task["table_path"])
    source = {r["record_id"]: r for r in source_rows}
    if len(source) != len(source_rows):
        raise VerificationError("Duplicate source record IDs")
    train, test = read_table(paths["oof"]), read_table(paths["holdout"])
    train_ids, test_ids = {r["record_id"] for r in train}, {r["record_id"] for r in test}
    if len(train_ids) != len(train) or len(test_ids) != len(test) or train_ids & test_ids:
        raise VerificationError("Repeated or overlapping train/test record IDs")
    if {r["split_group"] for r in train} & {r["split_group"] for r in test}:
        raise VerificationError("Train/test group leakage")
    if not train_ids | test_ids <= set(source):
        raise VerificationError("Prediction includes a record outside its task")
    if train_ids | test_ids != set(contract["common_record_ids"]):
        raise VerificationError("Prediction membership differs from feature-comparison contract")
    if len(train) != result["n_train"] or len(test) != result["n_test"]:
        raise VerificationError("Prediction count differs from reported train/test sizes")
    split = json.loads(paths["outer_split"].read_text(encoding="utf-8"))
    if not train_ids <= set(split["train_records"]) or not test_ids <= set(split["test_records"]):
        raise VerificationError("Prediction records violate the frozen outer split")
    folds = defaultdict(set)
    for r in train + test:
        original = source[r["record_id"]]
        if r["split_group"] != original["split_group"]:
            raise VerificationError("Prediction group differs from the task table")
        if split["record_groups"][r["record_id"]] != r["split_group"]:
            raise VerificationError("Prediction group differs from the frozen split")
        if not math.isclose(float(r["target"]), float(original["target"]), abs_tol=1e-10):
            raise VerificationError("Saved prediction target differs from the source task")
        if not all(
            math.isfinite(float(r[k])) for k in ("target", "prediction", "dummy_prediction")
        ):
            raise VerificationError("Nonfinite saved prediction")
    for r in train:
        folds[r["split_group"]].add(r["fold"])
        if int(r["fold"]) < 0:
            raise VerificationError("OOF fold is not an inner validation fold")
    if any(len(values) != 1 for values in folds.values()):
        raise VerificationError("Group crosses inner OOF folds")
    if len({r["fold"] for r in train}) < 2:
        raise VerificationError("OOF predictions need at least two validation folds")
    if "requested_inner_folds" in result or "actual_inner_folds" in result:
        if (
            result.get("requested_inner_folds") != 5
            or result.get("actual_inner_folds") != 5
            or len({r["fold"] for r in train}) != 5
        ):
            raise VerificationError("Declared strict five-fold evidence is inconsistent")
    kind = task["task_kind"]
    outer = metrics([r["target"] for r in test], [r["prediction"] for r in test], kind)
    inner = metrics([r["target"] for r in train], [r["prediction"] for r in train], kind)
    compare_metrics(outer, result["holdout"], "holdout")
    if "inner_cv" in result:
        compare_metrics(inner, result["inner_cv"], "inner CV")
    baseline = metrics([r["target"] for r in test], [r["dummy_prediction"] for r in test], kind)
    if "baseline_holdout" in result:
        compare_metrics(baseline, result["baseline_holdout"], "dummy holdout")
    native, native_cohorts = {}, {}
    if kind == "regression":
        transform = contract["target_transform"]
        native_cohorts = {
            name: native_metrics(records, transform, contract["target_unit"])
            for name, records in (("oof", train), ("holdout", test))
        }
        native = native_cohorts["holdout"]
    return {
        "artifact_hashes_verified": checked,
        "holdout": outer,
        "baseline_holdout": baseline,
        "native": native,
        "native_cohorts": native_cohorts,
        "quantitative_use_restricted": any(
            value["quantitative_use_restricted"] for value in native_cohorts.values()
        ),
        "target_transform": contract["target_transform"],
    }


def _summarize(run_dir, remote_root, profile, output_dir=None):
    run_dir = Path(run_dir)
    output_dir = run_dir if output_dir is None else Path(output_dir)
    manifest = json.loads((run_dir / "task_manifest.json").read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    rows, verified = [], []
    primary = {}
    scope_statuses = defaultdict(dict)
    trained_ids = set()
    for path in sorted((run_dir / "results" / profile).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        task = tasks[result["task_id"]]
        row = {
            "task_id": task["task_id"],
            "endpoint": task["endpoint"],
            "task_kind": task["task_kind"],
            "scope": result["scope_label"],
            "status": result["status"],
            "reason": result.get("reason", ""),
            "scientific_definition": json.dumps(
                task["task_definition"], ensure_ascii=False, sort_keys=True
            ),
            "scope_warning": task.get("scope_warning", ""),
            "selected_route": result.get("selected_route") or "",
            "selected_model": result.get("selected_model") or "",
            "refit_status": result.get("refit_status")
            or (
                "completed"
                if "refit_model" in result.get("artifacts", {})
                else "not_requested_or_unavailable"
            ),
            "refit_failure_reason": result.get("refit_failure_reason", ""),
            "refit_attempt_count": len(result.get("refit_attempts", [])),
            "deployment_ready": result.get("deployment_ready", "not_recorded"),
            "n_task_records": task.get("n_records", ""),
            "n_train": result.get("n_train", ""),
            "n_test": result.get("n_test", ""),
            "train_groups": result.get("train_groups", ""),
            "test_groups": result.get("test_groups", ""),
            "target_transform": "",
            "log_target_mae": "",
            "log_target_rmse": "",
            "spearman": "",
            "native_mae": "",
            "native_rmse": "",
            "native_unit": "",
            "native_metric_status": "not_applicable",
            "native_unrepresentable_rows": 0,
            "oof_native_metric_status": "not_applicable",
            "oof_native_unrepresentable_rows": 0,
            "quantitative_use_restricted": False,
            "pr_auc": "",
            "roc_auc": "",
            "mcc": "",
            "brier": "",
            "source_holdout": result.get("source_holdout", {}).get("status", "not_evaluated"),
            "probability_calibration": "uncalibrated"
            if task["task_kind"] == "classification"
            else "not_applicable",
        }
        if result["scope_label"] == "common":
            primary[task["task_id"]] = result["status"]
        scope_statuses[task["task_id"]][result["scope_label"]] = result["status"]
        if result["status"] == "failed":
            raise VerificationError("Training failure remains: " + task["task_id"])
        if result["status"] == "completed":
            evidence = verified_predictions(run_dir, remote_root, result, task)
            contract_path = resolve_artifact(
                run_dir, remote_root, result["artifacts"]["model_contract"]
            )
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            trained_ids.update(contract.get("common_record_ids", []))
            row["target_transform"] = evidence["target_transform"]
            outer = evidence["holdout"]
            if task["task_kind"] == "regression":
                row.update(
                    log_target_mae=outer["mae"],
                    log_target_rmse=outer["rmse"],
                    spearman=outer["spearman"],
                    **{
                        key: evidence["native"][key]
                        for key in (
                            "native_mae",
                            "native_rmse",
                            "native_unit",
                            "native_metric_status",
                            "native_unrepresentable_rows",
                        )
                    },
                    oof_native_metric_status=evidence["native_cohorts"]["oof"][
                        "native_metric_status"
                    ],
                    oof_native_unrepresentable_rows=evidence["native_cohorts"]["oof"][
                        "native_unrepresentable_rows"
                    ],
                    quantitative_use_restricted=evidence["quantitative_use_restricted"],
                )
            else:
                row.update(outer)
            verified.append(
                {
                    "result_file": str(path.relative_to(run_dir)),
                    "source_result_sha256": file_sha(path),
                    **evidence,
                }
            )
        rows.append(row)
    if not rows:
        raise VerificationError("No model results found for profile " + profile)
    primary_counts = Counter(primary.values())
    pending = sum(t["status"] == "eligible" and t["task_id"] not in primary for t in tasks.values())
    primary_summary = {s: primary_counts.get(s, 0) for s in ("completed", "deferred", "failed")}
    primary_summary["pending"] = pending
    expected_scopes = {"common"}
    config_path = run_dir / "execution_config.json"
    if profile == "full" and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_scopes.update(config.get("coverage_cohorts", {}))
    scientific_counts = Counter()
    for task_id, task in tasks.items():
        if task["status"] != "eligible":
            continue
        states = scope_statuses[task_id]
        if any(state == "failed" for state in states.values()):
            scientific_counts["failed"] += 1
        elif any(states.get(scope) not in {"completed", "deferred"} for scope in expected_scopes):
            scientific_counts["pending"] += 1
        else:
            scientific_counts["completed" if "completed" in states.values() else "deferred"] += 1
    native_warnings = [
        {"result_file": evidence["result_file"], "cohort": cohort, **values}
        for evidence in verified
        for cohort, values in evidence["native_cohorts"].items()
        if values["quantitative_use_restricted"]
    ]
    summary = {
        "schema": "peptide_modeling_verification_v2",
        "status": "verified",
        "profile": profile,
        "numeric_policy": NUMERIC_POLICY,
        "source_run_dir": str(run_dir.resolve()),
        "source_task_manifest_sha256": file_sha(run_dir / "task_manifest.json"),
        "primary_tasks": primary_summary,
        "scientific_tasks_any_coverage": {
            state: scientific_counts[state]
            for state in ("completed", "deferred", "failed", "pending")
        },
        "expected_scopes": sorted(expected_scopes),
        "unique_records_in_completed_model_cohorts": len(trained_ids),
        "scope_status_counts": {
            scope: dict(
                Counter(states[scope] for states in scope_statuses.values() if scope in states)
            )
            for scope in sorted({s for states in scope_statuses.values() for s in states})
        },
        "branches": dict(Counter(r["status"] for r in rows)),
        "artifact_hashes_verified": sum(r["artifact_hashes_verified"] for r in verified),
        "assertions": {
            "all_checks_passed": True,
            "artifacts_match_hashes": True,
            "record_and_group_holdout_disjoint": True,
            "targets_match_task_tables": True,
            "oof_groups_do_not_cross_folds": True,
            "saved_metrics_recomputed": True,
            "native_target_inverse_finite_or_explicitly_unavailable": True,
        },
        "numeric_diagnostics": {
            "all_native_inverses_finite": not any(
                value["native_unrepresentable_rows"] for value in native_warnings
            ),
            "all_native_error_metrics_available": not native_warnings,
            "results_with_native_warnings": len({v["result_file"] for v in native_warnings}),
            "cohorts_with_native_warnings": len(native_warnings),
            "affected_prediction_rows": sum(
                value["native_unrepresentable_rows"] for value in native_warnings
            ),
            "warnings": native_warnings,
            "policy": (
                "No clipping, row deletion or replacement of nonrepresentable inverses; "
                "original-unit metrics are unavailable for the entire affected cohort. "
                "Saved transformed predictions and their verified metrics remain unchanged."
            ),
            "warning_scope": "Numerical extrapolation restricts original-unit quantitative use",
            "model_selection_repeated": False,
            "source_predictions_modified": False,
        },
        "scientific_scope": (
            "internal_grouped_development_holdout; "
            "source external validation and probability calibration not completed"
        ),
        "results": verified,
    }
    write_table(output_dir / "model_inventory.tsv", list(rows[0]), rows)
    write_json(output_dir / "verification.json", summary)
    lines = [
        "# 自动化多肽建模结果与验收",
        "",
        f"正式阶段 `{profile}`：{primary_summary['completed']} 个科学任务完成，"
        f"{primary_summary['deferred']} 个延期，{pending} 个待运行。",
        f"训练/覆盖检查分支：{summary['branches']}。分支数量不等于端点或科学任务数量。",
        "",
        f"独立核验 {summary['artifact_hashes_verified']} 个模型及预测相关文件哈希；"
        "保存的目标、分组、OOF/留出记录和指标重算全部通过。",
        "",
        "common 表示固定共同样本上的路线比较；full_* 为单路线覆盖报告，不能跨子集直接比较排名。",
        "回归表中的 MAE/RMSE 是 log10 目标尺度。"
        "物理单位误差见 model_inventory.tsv 的 native_mae/native_rmse。",
        f"{len(native_warnings)}个回归队列的原单位数值风险已逐行记录；"
        "受影响整队列的原单位误差留空，有限目标变换尺度的指标保留。"
        "quantitative_use_restricted标记不改变原选模或原deployment_ready记录。",
        "BBB 与总体毒性为弱标签 benchmark，分数不代表真实生物学终点已完成独立验证。",
        "当前均为内部开发验证；论文/来源留出验证及分类概率校准尚未完成。",
        "",
        "| 端点 | 科学任务定义 | 训练/测试记录 | 路线/模型 | 留出指标 | 状态 |",
        "|---|---|---:|---|---|---|",
    ]
    for r in rows:
        if r["scope"] != "common":
            continue
        definition = r["scientific_definition"].replace("|", "/")
        rho = "NA" if r["spearman"] in (None, "") else f"{r['spearman']:.3f}"
        indicator = (
            f"Spearman={rho}; log10 MAE={r['log_target_mae']:.3f}"
            if r["task_kind"] == "regression" and r["status"] == "completed"
            else f"AP={r['pr_auc']:.3f}; ROC-AUC={r['roc_auc']:.3f}"
            if r["status"] == "completed"
            else r["reason"]
        )
        lines.append(
            f"| {r['endpoint']} | {definition} | {r['n_train']}/{r['n_test']} | "
            f"{r['selected_route']}/{r['selected_model']} | {indicator} | {r['status']} |"
        )
    lines += [
        "",
        "完整逐分支统计：model_inventory.tsv。验收证据：verification.json。",
        "实际拟合模型、预处理、OOF、留出预测及反变换合同保存在 models/。"
        "本验收不跨版本反序列化模型。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def summarize_run(run_dir, remote_root, profile="full", *, output_dir=None):
    try:
        return _summarize(run_dir, remote_root, profile, output_dir)
    except Exception as exc:
        write_json(
            Path(run_dir if output_dir is None else output_dir) / "verification.json",
            {
                "status": "failed",
                "profile": profile,
                "reason": f"{type(exc).__name__}:{exc}",
                "assertions": {"all_checks_passed": False},
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = summarize_run(args.run_dir, args.remote_root, args.profile, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "primary_tasks", "branches", "artifact_hashes_verified")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
