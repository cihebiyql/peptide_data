"""Compare preselected modelling routes only on exactly matching saved holdouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath


class ComparisonError(ValueError):
    """A comparison input does not satisfy its saved evidence contract."""


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_table(path, fields, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def artifact_path(run_dir, raw):
    """Relocate a saved model artifact using its run-relative models/ suffix."""
    root = Path(run_dir).resolve()
    normalized = str(raw).replace("\\", "/")
    remote = PurePosixPath(normalized)
    if ".." in remote.parts:
        raise ComparisonError("Artifact path contains parent traversal")
    native = Path(raw)
    if native.is_absolute() and native.resolve().is_relative_to(root):
        candidate = native.resolve()
    elif "models" in remote.parts:
        index = remote.parts.index("models")
        candidate = root.joinpath(*remote.parts[index:]).resolve()
    elif not remote.is_absolute() and not (remote.parts and remote.parts[0].endswith(":")):
        candidate = root.joinpath(*remote.parts).resolve()
    else:
        raise ComparisonError("Cannot rebase artifact to its run: " + str(raw))
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ComparisonError("Artifact is missing or outside its run: " + str(candidate))
    return candidate


def scientific_name(task):
    endpoint = task["endpoint"]
    definition = task.get("task_definition", {})
    if endpoint == "hemolysis":
        contract = definition.get("label_contract", "")
        if contract == "HC50_le_100_uM":
            return "HC50≤100 μM 溶血分类"
        if contract == "20_percent_lysis_at_50_uM":
            return "人红细胞20%@50 μM溶血分类"
        return "溶血：" + contract
    if endpoint == "half_life":
        species = {"mouse": "小鼠", "rat": "大鼠", "human": "人"}.get(
            definition.get("species"), definition.get("species", "物种未明确")
        )
        system = {
            "systemic_terminal_pk": "体内终末药代",
            "protease_or_intestinal_stability": "蛋白酶/肠道稳定性",
            "plasma_serum_stability": "血浆/血清稳定性",
        }.get(definition.get("assay_system"), definition.get("assay_system", ""))
        matrix = {
            "intestinal": "肠道",
            "plasma": "血浆",
            "serum": "血清",
            "systemic": "系统性",
        }.get(definition.get("matrix"), definition.get("matrix", ""))
        basis = {
            "immunoreactive_peptide_decay": "免疫反应性肽衰减",
            "chromatographic_intact_peptide_decay": "色谱完整肽衰减",
        }.get(definition.get("measurement_basis"), "")
        extra = "；" + basis if basis else ""
        label = "≥1 h分类" if task["task_kind"] == "classification" else "回归"
        return f"{species} {matrix}/{system}半衰期{label}{extra}"
    if endpoint == "BBB":
        return "BBB弱标签benchmark"
    if endpoint == "overall_peptide_toxicity":
        return "ToxinPred3总体毒性弱标签benchmark"
    if endpoint in {"PAMPA", "Caco2", "RRCK", "MDCK"}:
        return endpoint + "协议内渗透率"
    return endpoint


def _load_run(run_dir, inputs):
    root = Path(run_dir).resolve()
    verification_path = root / "verification.json"
    if not verification_path.is_file():
        raise ComparisonError("Run must be verified before comparison: " + str(root))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "verified" or not verification.get("assertions", {}).get(
        "all_checks_passed"
    ):
        raise ComparisonError("Run must be verified before comparison: " + str(root))
    manifest_path = root / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    inputs[str(verification_path)] = file_sha(verification_path)
    inputs[str(manifest_path)] = file_sha(manifest_path)
    proven_metrics = {r["result_file"]: r["holdout"] for r in verification.get("results", [])}
    results = {}
    for path in sorted((root / "results/full").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["scope_label"] != "common":
            continue
        if result["task_id"] in results:
            raise ComparisonError("Multiple common results for the same task")
        inputs[str(path)] = file_sha(path)
        relative = str(path.relative_to(root)).replace("\\", "/")
        if result["status"] == "completed" and relative in proven_metrics:
            for key, value in proven_metrics[relative].items():
                saved = result["holdout"].get(key)
                consistent = (saved is None and value is None) or (
                    saved is not None
                    and value is not None
                    and math.isclose(float(saved), float(value), rel_tol=1e-8, abs_tol=1e-10)
                )
                if not consistent:
                    raise ComparisonError("Saved metrics changed after verification: " + relative)
        results[result["task_id"]] = result
    if not results:
        raise ComparisonError("No full/common results in run: " + str(root))
    return root, tasks, results


def _holdout(run_dir, result, inputs):
    raw_path = result.get("artifacts", {}).get("holdout")
    if not raw_path:
        raise ComparisonError("Completed result lacks holdout artifact")
    path = artifact_path(run_dir, raw_path)
    receipt = result.get("artifact_hashes", {}).get(raw_path, {})
    actual = file_sha(path)
    if receipt.get("sha256") != actual:
        raise ComparisonError("Holdout artifact changed after training: " + str(path))
    inputs[str(path)] = actual
    rows = read_table(path)
    indexed = {r["record_id"]: (r["target"], r["split_group"]) for r in rows}
    if len(indexed) != len(rows):
        raise ComparisonError("Duplicate holdout IDs")
    if len(indexed) != result["n_test"]:
        raise ComparisonError("Holdout size differs from recorded result")
    return indexed


def _formatted_metrics(result, kind):
    if not result or result.get("status") != "completed":
        return "—"
    values = result["holdout"]

    def number(key):
        value = values.get(key)
        return "NA" if value is None else f"{float(value):.3f}"

    if kind == "regression":
        return "MAE=" + number("mae") + " / ρ=" + number("spearman")
    return "AP=" + number("pr_auc") + " / AUC=" + number("roc_auc")


def _compare(cpu_run, esm_run):
    inputs = {}
    cpu, cpu_tasks, cpu_results = _load_run(cpu_run, inputs)
    esm, esm_tasks, esm_results = _load_run(esm_run, inputs)
    rows, markdown = [], []
    task_ids = sorted(set(cpu_results) | set(esm_results))
    for tid in task_ids:
        task = cpu_tasks.get(tid) or esm_tasks[tid]
        c, e = cpu_results.get(tid), esm_results.get(tid)
        row = {
            "task_id": tid,
            "task_name": scientific_name(task),
            "endpoint": task["endpoint"],
            "task_kind": task["task_kind"],
            "cpu_status": c.get("status", "missing") if c else "missing",
            "esm_status": e.get("status", "missing") if e else "missing",
            "cpu_n_test": c.get("n_test", "") if c else "",
            "esm_n_test": e.get("n_test", "") if e else "",
            "common_n_test": "",
            "holdout_exact_match": "false",
            "comparison_status": "not_completed",
            "cpu_route": c.get("selected_route") or "" if c else "",
            "cpu_model": c.get("selected_model") or "" if c else "",
            "esm_candidate_set_route": e.get("selected_route") or "" if e else "",
            "esm_candidate_set_model": e.get("selected_model") or "" if e else "",
            "esm_route_selected": "false",
            "feature_attribution": "not_comparable",
            "cpu_mae_log10": "",
            "esm_mae_log10": "",
            "cpu_spearman": "",
            "esm_spearman": "",
            "cpu_pr_auc": "",
            "esm_pr_auc": "",
            "cpu_roc_auc": "",
            "esm_roc_auc": "",
            "cpu_inner_cv": json.dumps(c.get("inner_cv", {}) if c else {}, sort_keys=True),
            "esm_inner_cv": json.dumps(e.get("inner_cv", {}) if e else {}, sort_keys=True),
            "cpu_only_records": "",
            "esm_only_records": "",
            "target_mismatches": "",
            "group_mismatches": "",
            "interpretation": "任务未在两次实验中均完成，不比较排名。",
        }
        esm_selected = bool(e and "esm" in str(e.get("selected_route", "")).lower())
        row["esm_route_selected"] = (
            str(esm_selected).lower() if e and e["status"] == "completed" else ""
        )
        for result, prefix in ((c, "cpu"), (e, "esm")):
            if result and result["status"] == "completed":
                values = result["holdout"]
                for metric, suffix in (
                    ("mae", "mae_log10"),
                    ("spearman", "spearman"),
                    ("pr_auc", "pr_auc"),
                    ("roc_auc", "roc_auc"),
                ):
                    row[prefix + "_" + suffix] = values.get(metric, "")
        if c and e and c["status"] == e["status"] == "completed":
            ch, eh = _holdout(cpu, c, inputs), _holdout(esm, e, inputs)
            shared = set(ch) & set(eh)
            targets = sum(ch[k][0] != eh[k][0] for k in shared)
            groups = sum(ch[k][1] != eh[k][1] for k in shared)
            row.update(
                cpu_only_records=len(set(ch) - set(eh)),
                esm_only_records=len(set(eh) - set(ch)),
                target_mismatches=targets,
                group_mismatches=groups,
            )
            same_contract = all(
                cpu_tasks[tid].get(k) == esm_tasks[tid].get(k)
                for k in ("task_kind", "target_transform", "task_definition")
            )
            identical = ch == eh and same_contract
            row["holdout_exact_match"] = str(identical).lower()
            if identical:
                row["common_n_test"] = len(ch)
                row["comparison_status"] = "exact_holdout_comparison"
                if not esm_selected:
                    row["feature_attribution"] = "non_esm_route_selected_no_esm_attribution"
                    row["interpretation"] = (
                        "未选中ESM路线；扩展候选集仍选择非ESM表示，不归因于ESM。"
                    )
                else:
                    row["feature_attribution"] = "esm_selected_observational_comparison"
                    row["interpretation"] = (
                        "内层CV选择ESM路线；展示固定留出差异，不据留出重新选模。"
                    )
            else:
                row["comparison_status"] = "not_comparable_holdout_or_task_mismatch"
                row["interpretation"] = (
                    "测试ID、目标、分组或科学任务合同不完全相同，不可直接比较或排名。"
                )
        if task["endpoint"] == "overall_peptide_toxicity":
            row["interpretation"] += " 高维实验求解器工程版本与CPU低维不同，不作纯特征归因。"
            if row["holdout_exact_match"] == "true" and esm_selected:
                row["feature_attribution"] = "solver_change_prevents_pure_feature_attribution"
        rows.append(row)
        common = (
            str(row["common_n_test"])
            if row["holdout_exact_match"] == "true"
            else (f"不可比（{row['cpu_n_test']}/{row['esm_n_test']}）")
        )
        markdown.append(
            f"| {row['task_name']} | {common} | {row['cpu_route']}/{row['cpu_model']} | "
            f"{_formatted_metrics(c, task['task_kind'])} | "
            f"{row['esm_candidate_set_route']}/{row['esm_candidate_set_model']} | "
            f"{_formatted_metrics(e, task['task_kind'])} | "
            f"{'是' if esm_selected else '否' if row['esm_route_selected'] else '—'} | "
            f"{row['interpretation']} |"
        )
    counts = Counter(r["comparison_status"] for r in rows)
    summary = {
        "schema": "peptide_modeling_comparison_v1",
        "status": "verified",
        "cpu_run": str(cpu),
        "esm_run": str(esm),
        "counts": {
            "tasks": len(rows),
            "directly_comparable": counts["exact_holdout_comparison"],
            "holdout_or_contract_mismatch": counts["not_comparable_holdout_or_task_mismatch"],
            "not_completed_in_both": counts["not_completed"],
            "esm_selected_comparable": sum(
                r["holdout_exact_match"] == "true" and r["esm_route_selected"] == "true"
                for r in rows
            ),
        },
        "assertions": {
            "both_runs_independently_verified": True,
            "comparisons_require_exact_ids_targets_groups_and_task_contract": True,
            "no_holdout_driven_model_reselection": True,
            "non_esm_selected_routes_not_attributed_to_esm": True,
            "source_models_and_results_not_modified": True,
        },
        "inputs": inputs,
    }
    if any(file_sha(path) != expected for path, expected in inputs.items()):
        raise ComparisonError("A source result changed during comparison")
    write_table(cpu / "experiment_comparison.tsv", list(rows[0]), rows)
    lines = [
        "# CPU 与 ESM 候选集实验对照",
        "",
        f"逐任务核验：{summary['counts']['directly_comparable']} 个具有完全相同的"
        "留出记录、目标、分组及任务合同；"
        f"{summary['counts']['holdout_or_contract_mismatch']} 个不具备直接比较条件。",
        "",
        "表中路线均由各实验内层CV预先选择，留出指标用于展示，不用于重新挑选模型或全局获胜方案。",
        "第二次实验加入ESM候选路线，并不代表每个任务都选中了ESM。",
        "回归指标为log10目标尺度MAE和Spearman；分类为AP及ROC-AUC。",
        "",
        "| 科学任务 | 共同测试N | CPU路线/模型 | CPU留出指标 | ESM候选集路线/模型 | "
        "ESM候选集留出指标 | 选中ESM | 解释 |",
        "|---|---:|---|---|---|---|---|---|",
        *markdown,
        "",
        "当前验证按明确母体序列及完整规范结构分组，属于内部开发验证；"
        "尚非覆盖全部数据的90%同源族隔离，也未完成论文或来源外测。",
        "分类概率尚未校准。BBB与ToxinPred3为弱标签benchmark，不能作为真实生物学终点已验证的证据。",
        "高维弱ToxinPred3实验使用的求解器工程版本与CPU低维不同，不能将差异纯粹归因于ESM特征。",
        "完整表和证据分别见 experiment_comparison.tsv、comparison_verification.json。",
    ]
    (cpu / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["outputs"] = {
        name: {"sha256": file_sha(cpu / name), "bytes": (cpu / name).stat().st_size}
        for name in ("COMPARISON.md", "experiment_comparison.tsv")
    }
    write_json(cpu / "comparison_verification.json", summary)
    return summary


def compare_runs(cpu_run, esm_run):
    try:
        return _compare(cpu_run, esm_run)
    except Exception as exc:
        write_json(
            Path(cpu_run) / "comparison_verification.json",
            {"status": "failed", "reason": f"{type(exc).__name__}:{exc}"},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_run", type=Path)
    parser.add_argument("esm_run", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.cpu_run, args.esm_run)
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
