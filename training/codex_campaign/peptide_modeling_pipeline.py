"""Resumable coordinator for the frozen peptide modelling dataset.

Preparation and CPU extraction run beside the frozen dataset. The train stage is
portable: it needs only task tables, feature arrays and the modelling modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_rows(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def environment_receipt():
    import importlib.metadata

    versions = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "optuna",
        "joblib",
        "rdkit",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def portable_features(run_dir):
    """Relocate cache metadata without modifying source arrays or frozen data."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "features/manifest.json")
    cache_name = str(manifest["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    cache = run_dir / "features" / cache_name
    if not cache.is_dir():
        raise FileNotFoundError(f"Feature cache not staged: {cache}")
    manifest["cache_dir"] = str(cache.resolve())
    manifest["manifest_path"] = str((cache / "manifest.json").resolve())
    return manifest


def completed_checkpoint(path, expected):
    if not Path(path).exists():
        return None
    checkpoint = read_json(path)
    if checkpoint.get("input_signature") != expected:
        return None
    result = checkpoint.get("result", {})
    if result.get("status") == "deferred":
        return result
    if result.get("status") != "completed":
        return None
    artifacts = result.get("artifact_hashes", {})
    if not artifacts:
        return None
    for artifact, receipt in artifacts.items():
        if not Path(artifact).is_file() or file_sha(artifact) != receipt.get("sha256"):
            return None
    return result


def training_scopes(config, profile, include_full_coverage=False):
    """Keep route comparisons and broader coverage cohorts scientifically separate."""
    training = dict(config["training"])
    if profile == "smoke":
        training.update(config.get("smoke", {}))
    training["scope_label"] = "common"
    scopes = {"common": training}
    if profile == "smoke":
        return scopes
    for scope, specification in config.get("coverage_cohorts", {}).items():
        if not re.fullmatch(r"full_[A-Za-z0-9_.-]+", scope):
            raise ValueError("Coverage cohort scopes must be safe full_* names")
        options = specification if isinstance(specification, dict) else {"routes": specification}
        if set(options) - {"routes", "search_budget"}:
            raise ValueError("Coverage cohorts may override only routes and search_budget")
        routes = options.get("routes", [])
        if (
            not isinstance(routes, list)
            or not routes
            or any(route not in training["routes"] for route in routes)
            or len(set(routes)) != len(routes)
        ):
            raise ValueError("Coverage cohort routes must be distinct configured routes")
        cohort = {**training, **options, "scope_label": scope}
        if int(cohort.get("search_budget", 1)) < 1:
            raise ValueError("Coverage cohort search_budget must be positive")
        scopes[scope] = cohort
    if include_full_coverage:
        for route in training["routes"]:
            scope = "full_" + route
            if route == "fusion" or scope in scopes:
                continue
            scopes[scope] = {
                **training,
                "routes": [route],
                "scope_label": scope,
                "n_trials": int(config.get("full_coverage_trials", 5)),
            }
    return scopes


def jobs_for_profile(tasks, config, profile, only_tasks=None, include_full_coverage=False):
    eligible = [t for t in tasks if t["status"] == "eligible"]
    if only_tasks:
        eligible = [
            t for t in eligible if t["task_id"] in only_tasks or t["endpoint"] in only_tasks
        ]
    if profile == "smoke":
        first = {}
        for t in eligible:
            first.setdefault(t["task_kind"], t)
        eligible = list(first.values())
    scopes = training_scopes(config, profile, include_full_coverage)
    jobs = []
    for task in eligible:
        policy = None
        if profile == "full":
            for candidate in config.get("task_trial_budgets", []):
                maximum = candidate.get("max_records")
                if maximum is None or int(task.get("n_records", 0)) <= int(maximum):
                    policy = candidate
                    break
        for scope, training in scopes.items():
            settings = dict(training)
            if policy is not None:
                requested = int(policy["budgets_by_scope"][scope])
                if requested < 1:
                    raise ValueError("Per-task Optuna trial budgets must be positive")
                # Each eligible method/model pair still receives a first attempt.
                pair_ceiling = len(settings["routes"]) * len(settings["models"])
                settings["search_budget"] = max(requested, pair_ceiling)
                settings["task_budget_policy"] = {
                    "n_records": int(task.get("n_records", 0)),
                    "max_records": policy.get("max_records"),
                    "requested": requested,
                    "pair_coverage_floor": pair_ceiling,
                }
            jobs.append((task, settings))
    return jobs


def train_stage(run_dir, config, profile, only_tasks=None, include_full_coverage=False):
    from peptide_modeling_train import train_task

    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "task_manifest.json")
    features = portable_features(run_dir)
    jobs = jobs_for_profile(manifest["tasks"], config, profile, only_tasks, include_full_coverage)
    environment = environment_receipt()
    implementation = {
        "versions": environment["versions"],
        "python": list(sys.version_info[:3]),
        "code": {
            name: file_sha(Path(__file__).with_name(name))
            for name in (
                "peptide_modeling_train.py",
                "peptide_modeling_features.py",
                "peptide_modeling_search.py",
            )
            if Path(__file__).with_name(name).is_file()
        },
    }
    environment["implementation_contract"] = implementation
    receipt_name = (
        profile
        + ("_task_" + signature(sorted(only_tasks))[:12] if only_tasks else "")
        + "_environment.json"
    )
    write_json(run_dir / "receipts" / receipt_name, environment)
    results = []

    def one(task, training):
        scope = training["scope_label"]
        training["implementation_contract"] = implementation
        implementation_id = signature(implementation)[:12]
        name = task["task_id"] + "_" + scope
        checkpoint = run_dir / "checkpoints" / profile / (name + ".json")
        training["outer_split_path"] = str(
            (run_dir / "splits" / task["task_id"] / "group_assignment.json").resolve()
        )
        sig = signature(
            {
                "task_table": file_sha(run_dir / task["table_path"]),
                "feature_fingerprint": features["input_fingerprint"],
                "groups": features["groups"],
                "training": training,
                "implementation": implementation,
            }
        )
        cached = completed_checkpoint(checkpoint, sig)
        if cached is not None:
            print(f"[{profile}] cached {name}: {cached['status']}", flush=True)
            return cached
        started = time.time()
        write_json(
            checkpoint,
            {
                "status": "running",
                "input_signature": sig,
                "task_id": task["task_id"],
                "scope": scope,
                "host": socket.gethostname(),
                "pid": os.getpid(),
            },
        )
        try:
            rows = load_rows(run_dir / task["table_path"])
            result = train_task(
                task, rows, features, run_dir / "models" / profile / implementation_id, training
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "task_id": task["task_id"],
                "endpoint": task["endpoint"],
                "task_kind": task["task_kind"],
                "scope_label": scope,
                "reason": str(exc),
                "traceback": traceback.format_exc(),
            }
        result["elapsed_seconds_coordinator"] = round(time.time() - started, 3)
        result["scope_label"] = scope
        result["profile"] = profile
        write_json(checkpoint, {"input_signature": sig, "result": result})
        write_json(run_dir / "results" / profile / (name + ".json"), result)
        print(
            f"[{profile}] {name}: {result.get('status')} ({time.time() - started:.1f}s)", flush=True
        )
        return result

    workers = int(config.get("resources", {}).get("max_parallel_tasks", 2))
    # Different coverage scopes share one frozen split. Run scopes for each task
    # sequentially to prevent concurrent first-time split-file creation.
    by_task = {}
    for task, training in jobs:
        by_task.setdefault(task["task_id"], []).append((task, training))

    def sequence(task_jobs):
        return [one(task, training) for task, training in task_jobs]

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(by_task)))) as pool:
        futures = [pool.submit(sequence, j) for j in by_task.values()]
        for future in as_completed(futures):
            results.extend(future.result())
            if not only_tasks:
                write_json(run_dir / "results" / (profile + "_summary.json"), results)
    return results


def report_stage(run_dir):
    run_dir = Path(run_dir)
    tasks = read_json(run_dir / "task_manifest.json")
    task_counts = (
        tasks["counts"]
        if "counts" in tasks
        else {
            "declared_tasks": len(tasks["tasks"]),
            "eligible_tasks": sum(task.get("status") == "eligible" for task in tasks["tasks"]),
        }
    )
    records = []
    for p in sorted((run_dir / "results").glob("*/*.json")):
        records.append(read_json(p))
    counts = {
        s: sum(r.get("status") == s for r in records) for s in ("completed", "deferred", "failed")
    }
    profiles = {}
    for r in records:
        key = r.get("profile", "unknown")
        profiles.setdefault(key, {s: 0 for s in ("completed", "deferred", "failed")})
        profiles[key][r["status"]] += 1
    write_json(
        run_dir / "run_summary.json",
        {"tasks": task_counts, "model_runs": counts, "profiles": profiles, "results": records},
    )
    text = [
        "# 自动化多肽建模运行报告",
        "",
        f"科学任务准入：{task_counts['eligible_tasks']} 个；模型运行状态：{counts}。",
        "",
        "每个任务以科学定义、单位、标签规则和条件准入；来源信息仅用于审计与划分。",
        "common 为固定共同样本上的路线比较；full_* 为单路线覆盖报告，不能跨子集直接比较排名。",
        "内层CV用于选参选路线，外层留出用于评估；结果属于内部开发验证。",
        "",
        "| 阶段 | 端点 | 任务 | 范围 | 状态 | 选择路线/模型 |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        text.append(
            "| "
            + " | ".join(
                str(x)
                for x in (
                    r.get("profile", ""),
                    r.get("endpoint", ""),
                    r.get("task_id", ""),
                    r.get("scope_label", ""),
                    r.get("status", ""),
                    str(r.get("selected_route", "")) + "/" + str(r.get("selected_model", "")),
                )
            )
            + " |"
        )
    text += [
        "",
        "任务与延期原因见 task_summary.tsv/deferred_records.tsv；",
        "完整指标、真实模型、逐样本预测和预处理合同见 models/ 及 results/。",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/peptide_modeling_20261113_v1.json"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--stage", choices=["prepare", "features", "train", "report", "all"], default="all"
    )
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--only-task", action="append")
    parser.add_argument("--include-full-coverage", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if os.name == "posix" and args.stage in {"train", "all"}:
        requested_nice = int(config.get("resources", {}).get("nice", 0))
        if requested_nice > 0:
            os.nice(requested_nice)
    run_dir = (args.run_dir or ROOT / config["run_dir"]).resolve()
    dataset = (args.dataset or ROOT / config["dataset"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    standalone_worker = args.stage == "train" and bool(args.only_task)
    execution_config = run_dir / "execution_config.json"
    # The dispatcher publishes the global config once. A child may have a
    # task-specific CPU allocation and must never replace this shared receipt.
    if not standalone_worker and (
        not execution_config.exists() or read_json(execution_config) != config
    ):
        write_json(execution_config, config)
    if args.stage in {"prepare", "all"}:
        task_settings = dict(config["task_compilation"])
        compiler = task_settings.pop("compiler", "priority")
        if compiler == "all_endpoints":
            from peptide_modeling_all_tasks import compile_all_tasks

            base = ROOT / task_settings.pop("base_run")
            manifest = compile_all_tasks(dataset, run_dir, base_run=base, **task_settings)
        else:
            from peptide_modeling_tasks import compile_tasks

            manifest = compile_tasks(dataset, run_dir, **task_settings)
        print("TASKS", manifest["counts"], flush=True)
    if args.stage in {"features", "all"}:
        if config.get("feature_builder") == "precomputed":
            from peptide_modeling_features import _verify_artifacts

            features = portable_features(run_dir)
            _verify_artifacts(features)
        elif config.get("feature_builder") == "expanded":
            from peptide_modeling_expand_features import expand_features

            feature_pointer = run_dir / "features/manifest.json"
            current = read_json(feature_pointer) if feature_pointer.exists() else {}
            attached = current.get("configuration", {}).get("full_esm_attachment", {})
            source_index_sha = file_sha(run_dir / "feature_rows_index.tsv")
            if attached.get("source_feature_rows_sha256") == source_index_sha:
                from peptide_modeling_features import _verify_artifacts

                _verify_artifacts(current)
                features = current
            else:
                rows_index = load_rows(run_dir / "feature_rows_index.tsv")
                features = expand_features(
                    ROOT / config["base_features_run"], dataset, run_dir, rows_index=rows_index
                )
        else:
            from peptide_modeling_features import build_features

            manifest = read_json(run_dir / "task_manifest.json")
            paths = [
                run_dir / t["table_path"] for t in manifest["tasks"] if t["status"] == "eligible"
            ]
            features = build_features(dataset, run_dir, record_task_paths=paths)
        print("FEATURES", features["coverage"], flush=True)
        sequence_group = features["groups"]["sequence"]
        import shutil

        shutil.copy2(
            Path(features["cache_dir"]) / sequence_group["index"],
            run_dir / "esm_input_sequences.tsv",
        )
    if args.stage in {"train", "all"}:
        if config.get("require_complete_feature_bank", False):
            feature_contract = portable_features(run_dir)
            from peptide_modeling_features import _verify_artifacts

            # A missing/corrupted deployment artifact is an engineering failure,
            # never a scientific reason to defer every feature route.
            _verify_artifacts(feature_contract)
            if (
                feature_contract["groups"]["esm2"]["shape"][0]
                != feature_contract["groups"]["sequence"]["shape"][0]
            ):
                raise SystemExit(
                    "Full sequence-universe ESM outcomes must be computed before training"
                )
        trained = train_stage(
            run_dir, config, args.profile, args.only_task, args.include_full_coverage
        )
        if any(r["status"] == "failed" for r in trained):
            raise SystemExit(2)
    if args.stage in {"report", "train", "all"} and not standalone_worker:
        counts = report_stage(run_dir)
        print("REPORT", counts, flush=True)
        if args.stage in {"train", "all"} and counts["failed"]:
            raise SystemExit(2)
        if args.stage in {"train", "all"} and not args.only_task:
            from peptide_modeling_summarize import summarize_run

            verified = summarize_run(run_dir, str(run_dir), profile=args.profile)
            print("VERIFIED", verified, flush=True)


if __name__ == "__main__":
    main()
