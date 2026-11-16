"""Admit pinned source safety points with independently derived covalent graphs.

No labels, original gates, graphs or training outputs are rewritten. The derived
structures are chemical-specification interpretations, not measured source SMILES.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
from peptide_modeling_dramp_bridge import pinned_source
from peptide_modeling_features import _verify_artifacts, load_feature_route
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/remaining_safety_source_audit_20261114_v1"
BANK = ROOT / "data/peptide_modeling_safety_spec_features_20261114_v1"
LC50_OUTPUT = ROOT / "data/peptide_modeling_lc50_aib_followup_20261114_v1"
CC50_OUTPUT = ROOT / "data/peptide_modeling_cc50_followup_20261114_v1"
BANK_RECEIPT_SHA = "347227a89e0deb0c4135caf5aa35b7f56f27534f6f77287ba30c1c83f88b93bf"
PRODUCER_SHA = "7ea91efc99f5437bb3ad6035c701c60cfb8423d1f3301e3822a1d122738ff6fe"
SEEDS = (42, 2026, 3407)
INTERNAL = "internal_grouped_cv_only_no_independent_holdout"
HOLDOUT = "fixed_outer_group_holdout_with_fivefold_inner_selection"
SCOPES = {
    "LC50_AIB": ("cytotoxicity_LC50", "LC50", "PMID:17767978", "L929", "uM", 11),
    "CC50_COPIES": ("cytotoxicity_CC50", "CC50", "PMID:29290802", "Vero", "ug/mL", 5),
    "CC50_NAL": ("cytotoxicity_CC50", "CC50", "PMID:20718719", "Vero", "uM", 6),
}
WARNING = (
    "Same-publication source-reported cell safety benchmark. Primary article tables, "
    "complete exposure/readout protocol and physiological protonation/salt state are "
    "not independently validated. Covalent graphs are derived from explicit source "
    "chemical specifications. No clinical, source-external or scaffold generalization "
    "is established; this task does not design or enhance molecules."
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(values)


def project_path(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def frozen_module(path, name):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_member(row, frozen, check, source_number, source_row, points):
    """Bind one original member to one exact cell-specific source observation."""
    endpoint, metric, paper, cell, expected_unit, _ = SCOPES[row["scope"]]
    raw = json.loads(frozen["raw_json"])
    ctx = json.loads(raw["context_json"])
    if (
        frozen["record_id"] != check["record_id"]
        or frozen["endpoint"] != endpoint
        or frozen["source_code"] != "dramp_v4"
        or frozen["source_entry_id"] != str(source_row["DRAMP_ID"])
        or frozen["source_entry_id"] != check["source_entry_id"]
        or frozen["raw_relation"] != "="
        or frozen["relation"] != "="
        or raw["relation"] != "="
        or int(check["source_xlsx_row"]) != int(source_number)
    ):
        raise ValueError("Original identity, endpoint, point relation or worksheet row differs")
    annotation = str(source_row.get("Cytotoxicity", ""))
    source_spec = {
        "sequence": str(source_row["Sequence"]),
        "chemistry": {
            field: str(source_row.get(field, "")).strip() for field in row["spec"]["chemistry"]
        },
    }
    if (
        source_spec != row["spec"]
        or raw["sequence"] != source_spec["sequence"]
        or ctx["raw_annotation"] != annotation
        or annotation != check["source_annotation"]
        or hashlib.sha256(annotation.encode()).hexdigest() != check["source_annotation_sha256"]
        or any(ctx.get(k) != v for k, v in source_spec["chemistry"].items())
    ):
        raise ValueError("Original source sequence, annotation or chemical specification differs")
    value = Decimal(row["model_value"])
    matching = [
        point
        for point in points
        if point["metric"] == metric
        and point["relation"] == "="
        and point["unit"] == expected_unit
        and Decimal(point["scalar"]) == value
    ]
    if len(matching) != 1 or matching[0]["cell"] != cell or matching[0]["source_paper"] != paper:
        raise ValueError("Source point is ambiguous or belongs to a different cell/publication")
    point = matching[0]
    if (
        point != check["resolved_point"]
        or row["model_unit"] != expected_unit
        or frozen["raw_unit"] != expected_unit
        or raw["unit"] != expected_unit
        or any(Decimal(v) != value for v in (frozen["raw_value"], frozen["value"], raw["value"]))
        or not value.is_finite()
        or value <= 0
        or not math.isclose(float(row["target"]), math.log10(float(value)), abs_tol=1e-14)
    ):
        raise ValueError("Source scalar, unconverted unit or target transform differs")
    return {
        "record_id": frozen["record_id"],
        "source_entry_id": frozen["source_entry_id"],
        "source_xlsx_row": source_number,
        "source_paper": paper,
        "cell": cell,
        "raw_value": frozen["raw_value"],
        "raw_unit": frozen["raw_unit"],
        "relation": "=",
        "annotation_sha256": check["source_annotation_sha256"],
        "source_point": point,
    }


def validate_derivation(row, ledger, mapping, producer, components):
    if (
        ledger["record_id"] != row["record_id"]
        or ledger["scope"] != row["scope"]
        or ledger["source_chemical_specification"] != row["spec"]
        or ledger["source_paper"] != row["source_paper"]
        or ledger["cell"] != row["cell"]
        or ledger["model_unit"] != row["model_unit"]
        or Decimal(ledger["model_value"]) != Decimal(row["model_value"])
        or ledger["members"] != row["members"]
        or ledger["source_measured_smiles"] is not False
        or ledger["protonation_or_salt_state_verified"] is not False
        or ledger["independent_assemblies_equal"] is not True
    ):
        raise ValueError("Derived graph provenance differs from the selected source point")
    actual = producer.derive_graph(row["spec"], row["source_paper"], components)
    if any(ledger.get(key) != value for key, value in actual.items()):
        raise ValueError("Independent graph assembly, formula, stereochemistry or caps differ")
    selected = [item for item in mapping if item["representative_record_id"] == row["record_id"]]
    if {item["record_id"] for item in selected} != set(row["members"]) or any(
        item["structure_key"] != ledger["structure_key"]
        or item["split_group"] != ledger["split_group"]
        for item in selected
    ):
        raise ValueError("Original copy members do not share the verified graph/group")


def validate_groups(rows):
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Representatives contain duplicate source IDs")
    seen = {}
    for row in rows:
        identities = [row["canonical_identity_group"], *json.loads(row["existing_families_json"])]
        for key in identities:
            prior = seen.setdefault(key, row["split_group"])
            if prior != row["split_group"]:
                raise ValueError("A full graph or known FAMILY crosses identity groups")


def internal_precheck(rows):
    validate_groups(rows)
    groups = np.asarray([row["split_group"] for row in rows])
    y = np.asarray([float(row["target"]) for row in rows])
    if len(set(groups)) < 5 or not np.isfinite(y).all():
        raise ValueError("Insufficient finite independent groups for fivefold")
    probes, partitions = [], set()
    for seed in SEEDS:
        folds = []
        for a, b in GroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups):
            if set(groups[a]) & set(groups[b]) or len(set(y[a])) < 2:
                raise ValueError("Fivefold overlap or constant training target")
            folds.append(
                {
                    "train_record_ids": [rows[i]["record_id"] for i in a],
                    "validation_record_ids": [rows[i]["record_id"] for i in b],
                }
            )
        partitions.add(canonical(sorted(sorted(f["validation_record_ids"]) for f in folds)))
        probes.append({"seed": seed, "actual_inner_folds": 5, "folds": folds})
    return {"status": "verified", "probes": probes, "distinct_partitions": len(partitions)}


def holdout_precheck(rows, directory, settings):
    import peptide_modeling_train as training

    validate_groups(rows)
    probes = []
    y, groups = training.validate_rows(rows, "regression")
    for seed in SEEDS:
        cfg = {**settings, "seed": seed}
        split = training.freeze_outer_split(
            rows, "regression", directory / f"seed_{seed}.json", cfg
        )
        a = np.asarray(
            [i for i, row in enumerate(rows) if row["record_id"] in split["train_records"]]
        )
        b = np.asarray(
            [i for i, row in enumerate(rows) if row["record_id"] in split["test_records"]]
        )
        folds = training._inner_splits(y[a], groups[a], "regression", cfg)
        if len(folds) != 5 or set(groups[a]) & set(groups[b]):
            raise ValueError("Actual outer holdout/inner-fivefold plan differs")
        probes.append(
            {
                "seed": seed,
                "train_rows": len(a),
                "holdout_rows": len(b),
                "actual_inner_folds": len(folds),
                "train_record_ids": [rows[i]["record_id"] for i in a],
                "holdout_record_ids": [rows[i]["record_id"] for i in b],
                "inner_folds": [
                    {
                        "train_record_ids": [rows[a[i]]["record_id"] for i in x],
                        "validation_record_ids": [rows[a[i]]["record_id"] for i in v],
                    }
                    for x, v in folds
                ],
            }
        )
    return {"status": "verified", "probes": probes, "protocol": HOLDOUT}


def configuration(methods, holdout):
    return {
        "resources": {"training_host": "node23", "n_jobs": 1, "nice": 10, "cpu_budget": 1},
        "training": {
            "routes": [m["route"] for m in methods] + ["structure_methods_all"],
            "models": [
                "linear",
                "elasticnet",
                "random_forest",
                "extra_trees",
                "histgb",
                "lightgbm",
                "xgboost",
                "svm",
            ],
            "search_backend": "optuna",
            "search_budget": 128,
            "inner_folds": 5,
            "seed": 42,
            "n_jobs": 1,
            "min_rows": 10 if holdout else 5,
            "min_groups": 8 if holdout else 5,
            "test_size": 0.3,
            "refit_full": holdout,
            "svm_max_features": 16384,
            "histgb_max_features": 16384,
            "elasticnet_max_features": 16384,
        },
        "required_seeds": list(SEEDS),
        "validation_protocol": HOLDOUT if holdout else INTERNAL,
    }


def prepare(source=SOURCE, bank=BANK, lc50_output=LC50_OUTPUT, cc50_output=CC50_OUTPUT):
    import peptide_modeling_safety_spec_features as producer

    source, bank = Path(source), Path(bank)
    outputs = {
        "LC50_AIB": Path(lc50_output),
        "CC50_COPIES": Path(cc50_output),
        "CC50_NAL": Path(cc50_output),
    }
    if any(
        path.exists()
        and any(p.name not in {"development_snapshots", "deployment"} for p in path.iterdir())
        for path in set(outputs.values())
    ):
        raise FileExistsError("Prepared admissions are immutable; choose new output directories")
    if (
        sha(producer.__file__) != PRODUCER_SHA
        or sha(bank / "delivery_receipt.json") != BANK_RECEIPT_SHA
    ):
        raise ValueError("Graph producer or final feature delivery receipt changed")
    protected = {bank / "delivery_receipt.json": BANK_RECEIPT_SHA}
    delivery = read(bank / "delivery_receipt.json")
    for name, expected in delivery["files"].items():
        path = project_path(name)
        if sha(path) != expected:
            raise ValueError("Frozen graph bank artifact changed: " + name)
        protected[path] = expected
    selected, _, source_pins, audit = producer.validate_sources(source)
    protected.update({Path(p): h for p, h in source_pins.items()})
    parser = frozen_module(source / "audit.py", "_pinned_safety_source_parser")
    components = producer.registered_components(bank / "component_references")
    ledger, members = read(bank / "derivation_ledger.json"), read(bank / "member_mapping.json")
    derived, mapping = copy.deepcopy(ledger), copy.deepcopy(members)
    producer.reconcile_groups(derived, mapping, audit)
    if derived != ledger or mapping != members:
        raise ValueError("Known-family reconciliation differs from the frozen graph bank")
    lookup = {item["record_id"]: item for item in ledger}
    if len(lookup) != 22:
        raise ValueError("Derived representatives are missing or repeated")
    db = project_path(next(k for k in audit["source_sha256"] if k.endswith("catalogue.sqlite")))
    workbook = project_path(
        next(k for k in audit["source_sha256"] if k.endswith("general_amps.xlsx"))
    )
    pin = project_path(next(k for k in audit["source_sha256"] if k.endswith("27233508.json")))
    sources = pinned_source(workbook, pin)
    checks = {item["record_id"]: item for item in audit["record_checks"]}
    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    features = read(bank / "features/manifest.json")
    _verify_artifacts(features)
    registry = read(bank / "representation_methods.json")
    methods = registry["methods"]
    if len(methods) != 10 or any(m["source_kind"] != "structure" for m in methods):
        raise ValueError("Ten genuine complete-structure methods are required")
    scope_rows, source_proofs, copy_rows = {s: [] for s in SCOPES}, [], []
    for row in selected:
        item = lookup[row["record_id"]]
        validate_derivation(row, item, members, producer, components)
        for record in row["members"]:
            frozen = connection.execute(
                "SELECT * FROM records WHERE record_id=?", (record,)
            ).fetchone()
            if frozen is None:
                raise ValueError("An original source member is missing")
            number, original = sources[checks[record]["source_entry_id"]]
            proof = validate_member(
                row, dict(frozen), checks[record], number, original, parser.source_points(original)
            )
            proof.update(scope=row["scope"], representative_record_id=row["record_id"])
            source_proofs.append(proof)
            copy_rows.append(
                {
                    "record_id": record,
                    "representative_record_id": row["record_id"],
                    "scope": row["scope"],
                    "source_entry_id": checks[record]["source_entry_id"],
                    "structure_key": item["structure_key"],
                    "split_group": item["split_group"],
                }
            )
        scope_rows[row["scope"]].append(
            {
                "record_id": row["record_id"],
                "split_group": item["split_group"],
                "target": format(math.log10(float(row["model_value"])), ".17g"),
                "value": row["model_value"],
                "unit": row["model_unit"],
                "relation": "=",
                "target_transform": "log10",
                "structure_eligible": "true",
                "sequence_eligible": "false",
                "canonical_identity_group": item["structure_key"],
                "canonical_isomeric_smiles": item["canonical_isomeric_smiles"],
                "existing_families_json": canonical(item["existing_families"]),
                "source_paper": row["source_paper"],
                "source_cell": row["cell"],
                "source_sequence": row["spec"]["sequence"],
                "source_chemical_specification_json": canonical(row["spec"]),
                "source_member_record_ids_json": canonical(row["members"]),
                "source_raw_value": row.get("source_value", row["model_value"]),
                "source_raw_unit": row.get("source_unit", row["model_unit"]),
                "source_relation": "=",
                "source_measured_smiles": "false",
                "chemical_specification_derived_graph": "true",
                "full_protocol_verified": "false",
                "primary_article_table_values_verified": "false",
                "protonation_or_salt_state_verified": "false",
            }
        )
    connection.close()
    manifests = {}
    for output in set(outputs.values()):
        output.mkdir(parents=True, exist_ok=True)
        tasks, coverage, prechecks = [], {}, {}
        scopes = [s for s in SCOPES if outputs[s] == output]
        cfg = configuration(methods, scopes == ["LC50_AIB"])
        for scope in scopes:
            endpoint, metric, paper, cell, source_unit, count = SCOPES[scope]
            rows = scope_rows[scope]
            if len(rows) != count or len(set(r["split_group"] for r in rows)) != count:
                raise ValueError("Verified graph/FAMILY groups collapsed below declared scope")
            definition = {
                "canonical_measure": f"source_reported_{cell}_{metric}",
                "endpoint_lineage": endpoint,
                "source_paper": paper,
                "cell": cell,
                "unit": source_unit,
                "raw_relation": "=",
                "molecular_weight_conversion": "none",
                "chemical_representation": (
                    "neutral_covalent_graph_from_explicit_source_specification"
                ),
                "source_measured_smiles": False,
                "physiological_protonation_or_salt_state_verified": False,
                "primary_article_table_values_verified": False,
                "full_protocol_verified": False,
                "unknown_protocol_fields": [
                    "complete_exposure_duration",
                    "complete_readout_method_and_timing",
                ],
                "source_copy_members": sum(
                    len(json.loads(r["source_member_record_ids_json"])) for r in rows
                ),
                "source_copy_representatives": count,
                "exact_source_copy_collapse": scope == "CC50_COPIES",
                "source_safety_benchmark_only": True,
            }
            task_id = "TASK_" + signature(definition)[:24]
            task = {
                "task_id": task_id,
                "endpoint": endpoint,
                "task_kind": "regression",
                "status": "eligible",
                "training_admitted": True,
                "table_path": f"task_tables/{task_id}.tsv",
                "n_records": count,
                "n_groups": count,
                "unit": source_unit,
                "target_transform": "log10",
                "inverse_transform": "pow10",
                "task_definition": definition,
                "validation_protocol": cfg["validation_protocol"],
                "requested_inner_folds": 5,
                "source_evidence_tier": (
                    "pinned_secondary_point_and_explicit_chemical_specification_derived_graph"
                ),
                "scientific_warning": WARNING,
                "models_fitted": 0,
            }
            table(output / task["table_path"], rows)
            task["table_sha256"] = sha(output / task["table_path"])
            prechecks[task_id] = (
                holdout_precheck(rows, output / "split_prechecks", cfg["training"])
                if scope == "LC50_AIB"
                else internal_precheck(rows)
            )
            coverage[task_id] = {}
            for method in methods:
                payload = load_feature_route(rows, features, method["route"])
                if not payload["valid"].all() or payload["record_ids"] != [
                    r["record_id"] for r in rows
                ]:
                    raise ValueError("A representative lacks an admitted full-graph method")
                coverage[task_id][method["method_id"]] = {
                    "records": count,
                    "dimensions": payload["X"].shape[1],
                }
            tasks.append(task)
        manifest = {
            "schema": "source_safety_specification_admission_v1",
            "tasks": tasks,
            "derived_admission_only": True,
            "feature_fingerprint": features["input_fingerprint"],
            "source_data_modified": False,
            "models_fitted": 0,
        }
        manifests[str(output)] = manifest
        write(output / "task_manifest.json", manifest)
        write(output / "targeted_task_manifest.json", manifest)
        write(output / "recommended_training_config.json", cfg)
        write(output / "representation_methods.json", registry)
        write(output / "fivefold_prechecks.json", prechecks)
        write(
            output / "feature_coverage.json",
            {
                "status": "verified",
                "tasks": coverage,
                "feature_fingerprint": features["input_fingerprint"],
            },
        )
        write(
            output / "source_verification.json",
            {
                "status": "verified",
                "records": [p for p in source_proofs if p["scope"] in scopes],
                "graph_delivery_sha256": BANK_RECEIPT_SHA,
                "original_source_labels_modified": False,
            },
        )
        table(output / "source_members.tsv", [r for r in copy_rows if r["scope"] in scopes])
        write(
            output / "feature_reuse_plan.json",
            {
                "source": str(bank),
                "pointer_sha256": sha(bank / "features/manifest.json"),
                "feature_fingerprint": features["input_fingerprint"],
                "new_extraction_required": False,
                "graphs_are_chemical_specification_derived": True,
            },
        )
    if any(sha(p) != expected for p, expected in protected.items()):
        raise ValueError("Protected source/feature artifacts changed during admission")
    for output in set(outputs.values()):
        write(
            output / "admission_receipt.json",
            {
                "status": "prepared_not_trained",
                "source_and_feature_files_unchanged": len(protected),
                "graphs_are_derived_not_measured": True,
                "model_training": False,
                "files": {
                    p.relative_to(output).as_posix(): sha(p)
                    for p in output.rglob("*")
                    if p.is_file()
                    and not {"development_snapshots", "deployment"}
                    & set(p.relative_to(output).parts)
                },
            },
        )
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--lc50-output", type=Path, default=LC50_OUTPUT)
    parser.add_argument("--cc50-output", type=Path, default=CC50_OUTPUT)
    args = parser.parse_args()
    value = prepare(args.source, args.bank, args.lc50_output, args.cc50_output)
    print(
        json.dumps(
            {
                "status": "prepared_not_trained",
                "tasks": sum(len(x["tasks"]) for x in value.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
