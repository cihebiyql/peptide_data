"""Compare seeded searches using training CV within each task and coverage scope.

Only verified rounds contribute. A recommendation requires the declared required
seeds, and never uses holdout metrics. The report describes candidate recipes;
it does not claim that a consensus model has been fitted or externally validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_table(path, fields, rows):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def five_folds(value, *, required=False):
    keys = ("actual_folds", "requested_folds", "actual_inner_folds", "requested_inner_folds")
    if required and (
        value.get("actual_inner_folds") != 5 or value.get("requested_inner_folds") != 5
    ):
        raise ValueError("Completed result must prove five actual and requested inner folds")
    if any(key in value and value[key] != 5 for key in (*keys, "n_inner_folds")):
        raise ValueError("Candidate evidence contains a fold count other than five")


def search_directory(run_dir, result):
    directory = Path(result["search"]["search_dir"])
    directory = directory.resolve() if directory.is_absolute() else (run_dir / directory).resolve()
    if not directory.is_relative_to(run_dir):
        raise ValueError("Search evidence lies outside its seed run")
    if not (directory / "trials").is_dir():
        raise FileNotFoundError(directory / "trials")
    return directory


def unique_candidates(run_dir, result):
    """Repeated materializations of one parameter choice are one observation."""
    unique, duplicates = {}, 0
    rejected = {
        (item["route"], item["model"], canonical(item["params"]))
        for item in result.get("search", {})
        .get("candidate_fit_admission", {})
        .get("rejected_candidates", [])
    }
    for path in sorted((search_directory(run_dir, result) / "trials").glob("*.json")):
        trial = read_json(path)
        if "evaluation" in trial:
            from peptide_modeling_search import _read_trial

            trial = _read_trial(path)
        if trial.get("status") != "completed":
            continue
        five_folds(trial)
        for key, expected in (
            ("input_hash", result.get("input_hash")),
            ("search_input_signature", result["search"].get("namespace")),
        ):
            if expected is not None and key in trial and trial[key] != expected:
                raise ValueError("Candidate evidence belongs to a different input/search contract")
        score = float(trial["score"])
        if not math.isfinite(score):
            raise ValueError("A successful candidate has a nonfinite training-CV score")
        key = (trial["route"], trial["model"], canonical(trial["params"]))
        if key in rejected:
            continue
        if key in unique:
            duplicates += 1
            if not math.isclose(unique[key]["score"], score, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Repeated parameters have inconsistent scores within one seed")
            continue
        unique[key] = {
            "route": trial["route"],
            "model": trial["model"],
            "params": trial["params"],
            "score": score,
            "trial_file": str(path.relative_to(run_dir)),
            "trial_sha256": file_sha(path),
        }
    if not unique:
        raise ValueError("Completed model has no successful search candidate evidence")
    return list(unique.values()), duplicates


def build_report(campaign_root, durable_root=None):
    """Write reports and return compact artifact receipts suitable for a journal."""
    root = Path(campaign_root).resolve()
    journal_path = root / "campaign_status.json"
    journal_bytes = journal_path.read_bytes()
    journal = json.loads(journal_bytes)
    required = [int(seed) for seed in journal.get("required_seeds", [42, 2026, 3407])]
    if len(required) < 3 or len(required) != len(set(required)):
        raise ValueError("Cross-seed recommendations require at least three distinct seeds")
    rounds = [r for r in journal["rounds"] if r.get("audit", {}).get("status") == "verified"]
    seeds = [int(r["seed"]) for r in rounds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated seed rounds cannot count as independent replicates")
    candidates = defaultdict(dict)
    contexts, scope_states, selected, provenance = {}, defaultdict(dict), [], []
    duplicates = 0
    for entry in rounds:
        seed, run_dir = int(entry["seed"]), Path(entry["run_dir"]).resolve()
        manifest_path = run_dir / "task_manifest.json"
        manifest = read_json(manifest_path)
        tasks = {t["task_id"]: t for t in manifest["tasks"]}
        provenance.append(
            {"seed": seed, "run_dir": str(run_dir), "task_manifest_sha256": file_sha(manifest_path)}
        )
        for path in sorted((run_dir / "results/full").glob("*.json")):
            result = read_json(path)
            task = tasks[result["task_id"]]
            scope = result["scope_label"]
            group = (task["task_id"], scope)
            context = {
                "endpoint": task["endpoint"],
                "task_kind": task["task_kind"],
                "task_definition": task["task_definition"],
                "target_transform": task.get("target_transform"),
                "inverse_transform": task.get("inverse_transform"),
                "unit": task.get("unit"),
            }
            if group in contexts and contexts[group] != context:
                raise ValueError("A task/scope has different scientific contracts across seeds")
            contexts[group] = context
            if str(seed) in scope_states[group]:
                raise ValueError("One seed contains repeated results for the same task/scope")
            scope_states[group][str(seed)] = result["status"]
            if result["status"] == "deferred":
                continue
            if result["status"] != "completed":
                raise ValueError("Verified round contains a failed or nonterminal scope")
            five_folds(result, required=True)
            if result.get("search", {}).get("backend") != "optuna":
                raise ValueError("Completed results must provide an Optuna search")
            values, duplicate_count = unique_candidates(run_dir, result)
            duplicates += duplicate_count
            best = {}
            for value in values:
                pair = (value["route"], value["model"])
                prior = best.get(pair)
                if prior is None or (-value["score"], canonical(value["params"])) < (
                    -prior["score"],
                    canonical(prior["params"]),
                ):
                    best[pair] = value
            if seed in required:
                for (route, model), value in best.items():
                    candidates[(*group, route, model)][seed] = value
            selected.append(
                {
                    "task_id": task["task_id"],
                    "endpoint": task["endpoint"],
                    "scope": scope,
                    "seed": seed,
                    "required_seed": seed in required,
                    "selected_route": result.get("selected_route"),
                    "selected_model": result.get("selected_model"),
                    "selected_params": result.get("selected_params", {}),
                    "refit_status": result.get("refit_status", "not_recorded"),
                    "deployment_ready": result.get("deployment_ready", "not_recorded"),
                    "inner_cv": result.get("inner_cv", {}),
                    "holdout_metrics_display_only": result.get("holdout", {}),
                    "model_artifacts": result.get("artifacts", {}),
                    "result_file": str(path.relative_to(run_dir)),
                    "run_dir": str(run_dir),
                    "result_sha256": file_sha(path),
                    "unique_successful_parameter_choices": len(values),
                    "cached_duplicate_materializations": duplicate_count,
                }
            )
    pair_rows, recommendations = [], []
    grouped_pairs = defaultdict(list)
    for (task_id, scope, route, model), by_seed in sorted(candidates.items()):
        values = [by_seed[seed]["score"] for seed in required if seed in by_seed]
        complete = set(required).issubset(by_seed)
        row = {
            "task_id": task_id,
            "scope": scope,
            "endpoint": contexts[task_id, scope]["endpoint"],
            "route": route,
            "model": model,
            "seed_count": len(values),
            "required_seed_count": len(required),
            "complete_required_seeds": complete,
            "cv_score_mean": statistics.mean(values),
            "cv_score_std": statistics.stdev(values) if len(values) > 1 else None,
            "rank": None,
            "per_seed_best_training_candidates": {
                str(seed): by_seed[seed] for seed in required if seed in by_seed
            },
        }
        pair_rows.append(row)
        if complete:
            grouped_pairs[task_id, scope].append(row)
    for group, rows in sorted(grouped_pairs.items()):
        rows.sort(
            key=lambda row: (-row["cv_score_mean"], row["cv_score_std"], row["route"], row["model"])
        )
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        best = rows[0]
        recommendations.append(
            {
                "task_id": group[0],
                "scope": group[1],
                "endpoint": contexts[group]["endpoint"],
                "recommended_route": best["route"],
                "recommended_model": best["model"],
                "cv_score_mean": best["cv_score_mean"],
                "cv_score_std": best["cv_score_std"],
                "seed_count": best["seed_count"],
                "per_seed_tuned_parameters": {
                    seed: value["params"]
                    for seed, value in best["per_seed_best_training_candidates"].items()
                },
                "selection_basis": "required-seed mean of each pair's best training-only CV score",
                "consensus_model_fit": "not_claimed",
            }
        )
    status = "interim" if set(required) - set(seeds) else "required_seed_summary"
    report = {
        "schema": "peptide_cross_seed_cv_report_v1",
        "status": status,
        "campaign_journal_sha256": hashlib.sha256(journal_bytes).hexdigest(),
        "required_seeds": required,
        "available_verified_seeds": seeds,
        "missing_required_seeds": sorted(set(required) - set(seeds)),
        "ranking_direction": "higher_training_cv_score_is_better",
        "cross_scope_ranking": False,
        "holdout_used_for_selection": False,
        "goal_completion_claimed": False,
        "scientific_limit": "training-CV recipe recommendation; no consensus fit or external claim",
        "duplicate_materializations_not_counted_as_replicates": duplicates,
        "source_rounds": provenance,
        "pair_scores": pair_rows,
        "recommendations": recommendations,
        "selected_models_by_seed": selected,
        "task_scope_seed_status": [
            {"task_id": task, "scope": scope, "seed_status": states}
            for (task, scope), states in sorted(scope_states.items())
        ],
    }
    directory = root / "reports"
    directory.mkdir(exist_ok=True)
    write_json(directory / "cross_seed_report.json", report)
    pair_table = [
        {
            **row,
            "per_seed_best_training_candidates": canonical(
                row["per_seed_best_training_candidates"]
            ),
        }
        for row in pair_rows
    ]
    write_table(
        directory / "cross_seed_pairs.tsv",
        [
            "task_id",
            "endpoint",
            "scope",
            "route",
            "model",
            "seed_count",
            "required_seed_count",
            "complete_required_seeds",
            "cv_score_mean",
            "cv_score_std",
            "rank",
            "per_seed_best_training_candidates",
        ],
        pair_table,
    )
    selected_table = [
        {
            **row,
            **{
                key: canonical(row[key])
                for key in (
                    "selected_params",
                    "inner_cv",
                    "holdout_metrics_display_only",
                    "model_artifacts",
                )
            },
        }
        for row in selected
    ]
    write_table(
        directory / "seed_selected_models.tsv",
        [
            "task_id",
            "endpoint",
            "scope",
            "seed",
            "required_seed",
            "selected_route",
            "selected_model",
            "selected_params",
            "refit_status",
            "deployment_ready",
            "inner_cv",
            "holdout_metrics_display_only",
            "unique_successful_parameter_choices",
            "cached_duplicate_materializations",
            "run_dir",
            "result_file",
            "result_sha256",
            "model_artifacts",
        ],
        selected_table,
    )
    text = [
        "# Cross-seed training-CV report",
        "",
        f"Status: `{status}`.",
        "",
        f"Required seeds: {required}; verified seeds: {seeds}.",
        f"Task/scope recommendations with every required seed: {len(recommendations)}.",
        "",
        "Ranks use only the mean best inner-CV score for each route/model pair across required "
        "seeds. Hyperparameters remain seed-specific. Different tasks, assay conditions, units "
        "and coverage scopes are never pooled. Holdout metrics are display-only.",
        "",
        "This report does not claim a fitted consensus model, complete endpoint coverage, "
        "twenty productive hours, or external validation.",
        "",
        "See cross_seed_pairs.tsv for rankings and seed_selected_models.tsv for actual "
        "saved-model results and provenance.",
    ]
    path = directory / "CROSS_SEED_REPORT.md"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text("\n".join(text) + "\n", encoding="utf-8")
    temporary.replace(path)
    names = [
        "cross_seed_report.json",
        "cross_seed_pairs.tsv",
        "seed_selected_models.tsv",
        "CROSS_SEED_REPORT.md",
    ]
    artifacts = {
        name: {"path": str(directory / name), "sha256": file_sha(directory / name)}
        for name in names
    }
    if durable_root is not None:
        destination = Path(durable_root).resolve() / "reports"
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            temporary = destination / (name + ".writing")
            temporary.write_bytes((directory / name).read_bytes())
            if file_sha(temporary) != artifacts[name]["sha256"]:
                raise ValueError("Durable report copy failed its SHA check")
            temporary.replace(destination / name)
    return {
        "status": status,
        "available_verified_seeds": seeds,
        "missing_required_seeds": report["missing_required_seeds"],
        "ranked_task_scopes": len(recommendations),
        "artifacts": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.campaign_root, args.durable_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
