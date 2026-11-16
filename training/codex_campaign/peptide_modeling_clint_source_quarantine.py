"""Prepare a conservative six-point CLint revision without changing its sources.

The ambiguity concerns one paper's compound-code to graph association. The
graph itself is retained in the feature bank and is not globally prohibited.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import shutil
from decimal import Decimal
from pathlib import Path

import numpy as np
from peptide_modeling_clint_bridge import (
    DEFAULT_SOURCE,
    DOI,
    RECORD_IDS,
    SOURCE_SHA256,
    UNIT,
    VALUES,
    canonical_graph,
    pinned_source,
)
from peptide_modeling_features import _verify_artifacts, load_feature_route
from peptide_modeling_internal_cv import internal_splits
from peptide_modeling_train import validate_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/peptide_modeling_clint_followup_20261114_v1/training_template"
FAMILIES = ROOT / "data/peptide_modeling_clint_followup_20261114_v1/feature_rows_index.tsv"
IMPACT = ROOT / "data/acs_cell_ic50_source_audit_20261114_v1/clint_shared_graph_impact.json"
OUTPUT = ROOT / "data/peptide_modeling_clint_source_quarantine_20261114_v2"
TASK = "TASK_d9fe27fb90efc229ca43dd24"
EXCLUDED = "OLD:PV13:b64d32204089e229d8fdd569"
GRAPH = "CANON_SMILES:92a0902990200f6c3f96c001eea35fe440493b000aa884084292d3c659f46a57"
TABLE_SHA = "783e8a4d421e95862700f224cf11acfe6ff69dc0317c9984f4d96fc9df94e2ed"
SEEDS = (42, 2026, 3407)
REASON = "paper_scoped_compound_code_to_complete_graph_association_ambiguous"
DEFINITION = {
    "endpoint": "CL_per_mg_protein",
    "task_kind": "regression",
    "quantity_unit": UNIT,
    "assay_system": "in_vitro_intrinsic_microsomal_clearance",
    "species": "mouse",
    "matrix": "liver_microsomes",
    "route": "not_applicable_in_vitro",
    "measurement_basis": "intrinsic_protein_normalized_clearance",
    "source_assay_protocol": "acs_jmedchem_5c00544_table6_microsomal",
    "target_transform": "log10",
}
MODELS = {
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def source_conflict(source_file=DEFAULT_SOURCE):
    """Re-read the authored source; central estimates differ before any modelling."""
    seven = pinned_source(source_file)
    with Path(source_file).open(encoding="cp1252", newline="") as handle:
        data = list(csv.reader(handle))
    if data[0][2] != "D283-med IC50 (nM)":
        raise ValueError("Source cell column or unit changed")
    pairs = {row[0]: row for row in data[2:] if row and row[0] in {"15", "16"}}
    if len([row for row in data[2:] if row and row[0] in {"15", "16"}]) != 2:
        raise ValueError("Source compound identifiers are absent or repeated")
    graph15, key15 = canonical_graph(pairs["15"][1])
    graph16, key16 = canonical_graph(pairs["16"][1])
    means = {
        code: Decimal(pairs[code][2].split("±")[0].replace(",", "").strip())
        for code in ("15", "16")
    }
    if graph15 != graph16 or key15 != GRAPH or key16 != GRAPH:
        raise ValueError("The paper-scoped shared complete graph is not confirmed")
    if means != {"15": Decimal("52.7"), "16": Decimal("6094")}:
        raise ValueError("The source activity discrepancy is not the pinned discrepancy")
    return seven, {
        "source_doi": DOI,
        "source_csv_sha256": SOURCE_SHA256,
        "compound_codes": ["15", "16"],
        "shared_full_graph": key15,
        "raw_cell_values_nM": {code: pairs[code][2] for code in ("15", "16")},
        "central_estimates_nM": {code: str(value) for code, value in means.items()},
        "which_author_structure_is_correct": "not_established",
        "scope": "this_paper_compound_code_association_only_not_a_global_graph_ban",
        "heldout_results_consulted": False,
    }


def validate_impact(impact):
    expected = {
        "affected_task": TASK,
        "source_table_sha256": TABLE_SHA,
        "source_graph": GRAPH,
        "affected_record_ids": [EXCLUDED],
        "affected_source_entries": ["compound_15:microsomal_clint"],
        "original_rows": 7,
        "conservative_rows": 6,
        "conservative_graphs": 6,
        "conservative_families": 6,
    }
    if any(impact.get(key) != value for key, value in expected.items()):
        raise ValueError("Impact evidence differs from the bounded one-point revision")
    if set(impact.get("ambiguous_source_compound_codes", [])) != {"15", "16"}:
        raise ValueError("Impact paper compound-code scope changed")


def validate_task(task):
    if (
        task.get("task_id") != TASK
        or task.get("endpoint") != "CL_per_mg_protein"
        or task.get("task_kind") != "regression"
        or task.get("target_transform") != "log10"
        or task.get("n_records") != 7
        or task.get("n_groups") != 7
        or task.get("task_definition") != DEFINITION
    ):
        raise ValueError("Original scientific CLint task contract differs")


def table_revision(raw, expected_sha=TABLE_SHA):
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Original seven-point table SHA changed")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter="\t"))
    ids = [row["record_id"] for row in rows]
    if len(rows) != 7 or len(set(ids)) != 7 or set(ids) != set(RECORD_IDS.values()):
        raise ValueError("Original seven source records differ")
    lines = raw.splitlines(keepends=True)
    if len(lines) != 8:
        raise ValueError("Byte-preserving revision requires one physical line per observation")
    if [line.split(b"\t", 1)[0].decode("utf-8") for line in lines[1:]] != ids:
        raise ValueError("TSV physical lines do not match parsed record identifiers")
    kept = [row for row in rows if row["record_id"] != EXCLUDED]
    kept_bytes = lines[0] + b"".join(
        line for row, line in zip(rows, lines[1:], strict=True) if row["record_id"] != EXCLUDED
    )
    return rows, kept, kept_bytes


def family_evidence(path, ids):
    found = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["record_id"] in ids:
                if row["record_id"] in found:
                    raise ValueError("Original FAMILY index repeats a source record")
                found[row["record_id"]] = row
    if set(found) != set(ids):
        raise ValueError("Original FAMILY evidence is incomplete")
    return found


def validate_original_rows(rows, seven, families):
    by_id = {record: code for code, record in RECORD_IDS.items()}
    for row in rows:
        code = by_id[row["record_id"]]
        _, graph = canonical_graph(seven[code]["smiles"])
        if (
            row["source_code"] != "acs_jmedchem_5c00544"
            or row["source_entry_id"] != f"compound_{code}:microsomal_clint"
            or row["source_group"] != DOI
            or row["canonical_identity_group"] != graph
            or row["unit"] != UNIT
            or row["target_transform"] != "log10"
            or abs(float(row["target"]) - math.log10(float(VALUES[code]))) > 1e-10
            or row["split_group"] != families[row["record_id"]]["split_group"]
            or graph != families[row["record_id"]]["canonical_identity_group"]
        ):
            raise ValueError("Original source identity, label, quantity or FAMILY differs")
    excluded = next(row for row in rows if row["record_id"] == EXCLUDED)
    if excluded["canonical_identity_group"] != GRAPH:
        raise ValueError("Excluded observation is not the ambiguous paper association")


def prechecks(rows, config):
    y, groups = validate_rows(rows, "regression")
    if len(rows) != 6 or len(set(groups)) != 6:
        raise ValueError("Revised internal cohort must have six unchanged source groups")
    result = []
    for seed in SEEDS:
        folds = internal_splits(y, groups, "regression", {**config, "seed": seed})
        if len(folds) != 5:
            raise ValueError("Revised cohort did not produce five actual folds")
        members, seen = [], []
        for train, valid in folds:
            if set(groups[train]) & set(groups[valid]) or len(np.unique(y[train])) < 2:
                raise ValueError("A revised fold leaks a group or lacks target variation")
            seen.extend(valid.tolist())
            members.append(
                {
                    "train_record_ids": [rows[i]["record_id"] for i in train],
                    "validation_record_ids": [rows[i]["record_id"] for i in valid],
                }
            )
        if sorted(seen) != list(range(6)):
            raise ValueError("Revised folds do not cover every observation once")
        result.append({"seed": seed, "actual_folds": 5, "members": members})
    return result


def prepare(
    source=SOURCE,
    output=OUTPUT,
    impact_path=IMPACT,
    source_csv=DEFAULT_SOURCE,
    family_path=FAMILIES,
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            "Use a new revision directory; existing evidence is never overwritten"
        )
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Revision output must not overlap its source template")
    impact = read(impact_path)
    validate_impact(impact)
    table_path = source / "task_tables" / (TASK + ".tsv")
    all_rows, kept, table_bytes = table_revision(table_path.read_bytes())
    manifest = read(source / "task_manifest.json")
    if len(manifest["tasks"]) != 1 or manifest["tasks"][0]["task_id"] != TASK:
        raise ValueError("Source template contains a different task scope")
    original_task = manifest["tasks"][0]
    validate_task(original_task)
    seven, conflict = source_conflict(source_csv)
    families = family_evidence(family_path, {row["record_id"] for row in all_rows})
    validate_original_rows(all_rows, seven, families)
    pointer_path = source / "features/manifest.json"
    feature = read(pointer_path)
    _verify_artifacts(feature)
    old_cache = Path(feature["cache_dir"]).resolve()
    if output == old_cache or old_cache in output.parents or output in old_cache.parents:
        raise ValueError("Revision output must not overlap its source feature bank")
    protected_paths = [
        table_path,
        source / "task_manifest.json",
        source / "template_config.json",
        source / "representation_methods.json",
        Path(source_csv),
        Path(impact_path),
        Path(family_path),
        pointer_path,
        *old_cache.rglob("*"),
    ]
    protected = {str(path): sha(path) for path in protected_paths if path.is_file()}
    config = read(source / "template_config.json")
    if set(config["training"]["models"]) != MODELS:
        raise ValueError("Original eight-model search scope differs")
    config["training"].update(
        n_jobs=1,
        search_backend="optuna",
        search_budget=128,
        inner_folds=5,
        min_rows=5,
        min_groups=5,
        refit_full=False,
    )
    config["resources"].update(n_jobs=1, nice=10, cpu_budget=1, training_host="node23")
    config["run_dir"] = str(output)
    config["scientific_scope"] = (
        "Six retained paper-source mouse liver microsomal CLint observations; one ambiguous "
        "compound-code/graph association quarantined before training; internal tuning-CV only."
    )
    partitions = prechecks(kept, config["training"])
    revision = {
        "source_table_sha256": TABLE_SHA,
        "source_manifest_sha256": sha(source / "task_manifest.json"),
        "impact_evidence_sha256": sha(impact_path),
        "source_csv_sha256": SOURCE_SHA256,
        "source_doi": DOI,
        "excluded_record_ids": [EXCLUDED],
        "excluded_source_entries": ["compound_15:microsomal_clint"],
        "ambiguous_source_graph": GRAPH,
        "ambiguous_author_compound_codes": ["15", "16"],
        "reason": REASON,
        "scope": "this_paper_compound_code_to_graph_association_only",
        "graph_globally_banned": False,
        "same_scientific_task_id_retained": True,
        "original_data_or_evaluation_modified": False,
        "model_selection_or_heldout_results_consulted": False,
    }
    task = copy.deepcopy(original_task)
    task.update(n_records=6, n_groups=6, source_revision=revision)
    derived = {
        "schema": "derived_clint_source_quarantine_v2",
        "tasks": [task],
        "counts": {
            "eligible_tasks": 1,
            "eligible_table_records": 6,
            "endpoints_with_eligible_tasks": 1,
            "quarantined_source_records": 1,
        },
        "source_ids": [row["record_id"] for row in kept],
        "protocol": "internal_grouped_cv_only_no_independent_holdout",
        "source_revision": revision,
        "historical_source_counts_not_current_revision_counts": {
            key: value for key, value in manifest.items() if "counts" in key
        },
    }
    output.mkdir(parents=True)
    target = output / "task_tables" / (TASK + ".tsv")
    target.parent.mkdir()
    target.write_bytes(table_bytes)
    if read_rows(target) != kept:
        raise ValueError("Revised table did not preserve retained parsed values")
    registry = read(source / "representation_methods.json")
    methods = [m for m in registry["methods"] if m.get("independent")]
    if len(methods) != 10 or any(m["route"] not in config["training"]["routes"] for m in methods):
        raise ValueError("Original ten-method configuration differs")
    new_cache = output / "features" / old_cache.name
    shutil.copytree(old_cache, new_cache)
    relocated = {
        **feature,
        "cache_dir": str(new_cache),
        "manifest_path": str(new_cache / "manifest.json"),
    }
    write(output / "features/manifest.json", relocated)
    _verify_artifacts(relocated)
    coverage = {}
    for method in methods:
        payload = load_feature_route(kept, relocated, method["route"])
        if (
            payload["record_ids"] != [row["record_id"] for row in kept]
            or not payload["valid"].all()
        ):
            raise ValueError("Revised observations lack a valid original feature method")
        coverage[method["method_id"]] = {
            "records": 6,
            "dimensions": payload["X"].shape[1],
            "all_valid": True,
        }
    copied = {str(p.relative_to(old_cache)): sha(p) for p in old_cache.rglob("*") if p.is_file()}
    if any(sha(new_cache / name) != value for name, value in copied.items()):
        raise ValueError("A copied seven-graph feature artifact changed")
    for filename in ["task_manifest.json", "targeted_task_manifest.json"]:
        write(output / filename, derived)
    write(output / "recommended_training_config.json", config)
    write(output / "representation_methods.json", registry)
    write(
        output / "fivefold_prechecks.json",
        {
            "actual_seed_checks": partitions,
            "independent_holdout_available": False,
            "models_fitted": 0,
        },
    )
    write(output / "source_revision.json", {**revision, "source_conflict": conflict})
    write(
        output / "source_rows.json",
        [
            {
                "record_id": row["record_id"],
                "source_entry_id": row["source_entry_id"],
                "disposition": "quarantined" if row["record_id"] == EXCLUDED else "retained",
                "original_target": row["target"],
                "original_unit": row["unit"],
                "original_group": row["split_group"],
            }
            for row in all_rows
        ],
    )
    write(
        output / "feature_reuse_plan.json",
        {
            "source_template": str(source),
            "source_pointer_sha256": protected[str(pointer_path)],
            "fingerprint": feature["input_fingerprint"],
            "source_bank_structures_retained": 7,
            "requested_revised_records": 6,
            "unused_feature_record_ids": [EXCLUDED],
            "original_bank_files_unchanged": True,
            "copied_files_byte_identical": len(copied),
            "new_feature_extractions": 0,
        },
    )
    write(output / "feature_coverage.json", coverage)
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("A protected source file changed during preparation")
    write(
        output / "source_integrity.json",
        {
            "protected_sha256": protected,
            "all_unchanged": True,
            "retained_rows_byte_identical": True,
            "retained_labels_and_groups_unchanged": True,
        },
    )
    write(
        output / "admission_receipt.json",
        {
            "status": "prepared_not_trained",
            "task_id": TASK,
            "source_records": 7,
            "retained_records": 6,
            "quarantined_records": 1,
            "independent_groups": 6,
            "feature_methods": 10,
            "feature_artifact_hashes_verified": len(feature["artifact_hashes"]),
            "parent_bank_unchanged": True,
            "models_fitted": 0,
            "remote_operations": False,
            "source_labels_modified": False,
        },
    )
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--impact", type=Path, default=IMPACT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.impact)
    print(json.dumps({"status": "prepared_not_trained", **result["counts"]}))


if __name__ == "__main__":
    main()
