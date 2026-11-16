"""Prepare a source-qualified MHC benchmark in its original mass unit.

Only the pinned archive's point observations are admitted. Missing molecular-weight
units are never guessed, and no universal hemolysis threshold is inferred.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sqlite3
from decimal import Decimal
from pathlib import Path

import numpy as np
from peptide_modeling_add_smiles_methods import rdkit_tools
from peptide_modeling_clint_features import STRUCTURE_GROUPS, atomic_array, extract_all
from peptide_modeling_expand_features import safe_path, write_tsv
from peptide_modeling_features import _verify_artifacts, load_feature_route
from peptide_modeling_qualitative_cyto_features import check_matrix
from peptide_modeling_train import _inner_splits, freeze_outer_split, validate_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/remaining_safety_source_audit_20261114_v1"
BASE = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_mhc_mass_followup_20261114_v1"
SOURCE_RECEIPT_SHA = "a0d03ebcb86f93393f05b6834086cf858f65e0d26401d9b71dcd83b2077e1ec0"
PAPER = "PMID:14584933"
DESCRIPTION = "Tested for minimum concentration for hemolysis of fresh human erythrocytes"
MEASURE = "source_reported_MHC_original_mass_threshold_unverified"
VERSION = "peptide_mhc_original_mass_full_graph_v1"
SEEDS = (42, 2026, 3407)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path, delimiter="\t"):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def project_path(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def scalar(value):
    if not re.fullmatch(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(value)):
        raise ValueError("Original MHC value must be an exact scalar")
    result = Decimal(str(value))
    if not result.is_finite() or result <= 0:
        raise ValueError("Original MHC value must be positive finite")
    return result


def graph(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or any(
        value == "?" for _, value in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    ):
        raise ValueError("Source full graph is invalid or stereochemically unspecified")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return canonical, "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest()


def verify_sources(source):
    source = Path(source)
    receipt = source / "delivery_receipt.json"
    if sha(receipt) != SOURCE_RECEIPT_SHA:
        raise ValueError("Pinned source audit receipt changed")
    protected = {receipt: SOURCE_RECEIPT_SHA}
    for name, expected in read(receipt)["files"].items():
        relative = Path(name.replace("\\", "/")).relative_to(SOURCE.relative_to(ROOT))
        path = source / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Pinned source audit artifact changed: " + name)
        protected[path] = expected
    audit = read(source / "audit.json")
    for name, expected in audit["source_sha256"].items():
        path = project_path(name)
        if sha(path) != expected:
            raise ValueError("Original source data changed: " + name)
        protected[path] = expected
    return audit, protected


def validate_record(candidate, frozen, original, compound, assay, reference):
    """Verify the original point and full graph, rather than invert the old uM label."""
    raw = json.loads(frozen["raw_json"])
    if original != raw:
        raise ValueError("Original archive observation differs from frozen raw JSON")
    if (
        frozen["endpoint"] != "MHC"
        or frozen["source_code"] != "CHEMBL"
        or frozen["lane"] not in {"review", "strict_numeric"}
        or frozen["raw_relation"] != "="
        or raw["qualifier"] != "="
        or raw["endpoint"] != "MHC"
        or raw["unit"] != "ug ml-1"
        or raw["original_unit"] != "ug ml-1"
        or candidate["model_unit"] != "ug/mL"
        or candidate["raw_relation"] != "="
        or candidate["source_paper"] != PAPER
        or candidate["record_id"] != frozen["record_id"]
    ):
        raise ValueError("Source MHC endpoint, point relation or original mass unit differs")
    value = scalar(raw["original_value"])
    if any(
        scalar(item) != value
        for item in (raw["value"], frozen["raw_value"], candidate["model_value"])
    ):
        raise ValueError("Original mass label was changed or replaced by normalized uM")
    if (
        raw["assay_id"] != frozen["assay_id"]
        or assay["assay_id"] != frozen["assay_id"]
        or assay["description"] != DESCRIPTION
        or assay["system_description"] != "Homo sapiens"
        or assay["source_code"] != "CHEMBL"
        or raw["compound_id"] != frozen["compound_id"]
        or compound["compound_id"] != frozen["compound_id"]
        or raw["reference_id"] != frozen["reference_ids"]
        or reference["reference_id"] != frozen["reference_ids"]
        or reference["pmid"] != PAPER.removeprefix("PMID:")
    ):
        raise ValueError("Original observation/assay/compound/publication binding differs")
    canonical, key = graph(compound["smiles"])
    if (
        canonical != frozen["canonical_smiles"]
        or canonical != candidate["canonical_isomeric_smiles"]
        or key != frozen["canonical_identity_group"]
        or key != candidate["canonical_identity_group"]
        or frozen["structure_model_eligible"] != "true"
    ):
        raise ValueError("Original full stereochemical graph or structure gate differs")
    if (
        candidate["hemolysis_fraction_threshold_verified"] != "False"
        or candidate["full_protocol_verified"] != "False"
        or candidate["molecular_weight_guessed"] != "False"
    ):
        raise ValueError("Unverified threshold/protocol or molecular-weight assumption changed")
    return canonical, key, value


def reconcile_groups(candidates, families):
    graphs = {row["canonical_identity_group"] for row in candidates}
    ids = {row["record_id"] for row in candidates}
    by_graph, by_id, parent = {key: set() for key in graphs}, {}, {}
    for item in families:
        if item["canonical_identity_group"] in graphs:
            by_graph[item["canonical_identity_group"]].add(item["split_group"])
        if item["record_id"] in ids:
            by_id[item["record_id"]] = item["split_group"]

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in candidates:
        key = row["canonical_identity_group"]
        known = by_graph[key] | ({by_id[row["record_id"]]} if row["record_id"] in by_id else set())
        if sorted(known) != json.loads(row["existing_families"]):
            raise ValueError("Existing same-graph/FAMILY evidence differs")
        find(key)
        for family in known:
            a, b = find(key), find(family)
            if a != b:
                parent[max(a, b)] = min(a, b)
    result = {}
    for row in candidates:
        value = "IDENTITY:" + signature(find(row["canonical_identity_group"]))[:24]
        if row["split_group"] != value:
            raise ValueError("Source graph/FAMILY connected group changed")
        result[row["record_id"]] = value
    return result


def source_rows(source=SOURCE):
    audit, protected = verify_sources(source)
    candidates = table(Path(source) / "candidate_tables/MHC_PMID14584933_original_mass.tsv")
    if len(candidates) != 24 or len({row["record_id"] for row in candidates}) != 24:
        raise ValueError("Exactly 24 unique original MHC observations are required")
    db = project_path(
        next(name for name in audit["source_sha256"] if name.endswith("catalogue.sqlite"))
    )
    connection = sqlite3.connect("file:" + db.as_posix() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    marks = ",".join("?" for _ in candidates)
    frozen = {
        row["record_id"]: dict(row)
        for row in connection.execute(
            f"SELECT * FROM records_with_structure_qc WHERE record_id IN ({marks})",
            [row["record_id"] for row in candidates],
        )
    }
    connection.close()
    if len(frozen) != 24:
        raise ValueError("A selected frozen source record is missing")
    csv_root = project_path(next(iter(frozen.values()))["input_file"]).parent
    keys = {
        "observations": (
            "08_other_activity.csv",
            "observation_id",
            {json.loads(r["raw_json"])["observation_id"] for r in frozen.values()},
        ),
        "compounds": (
            "01_compounds.csv",
            "compound_id",
            {r["compound_id"] for r in frozen.values()},
        ),
        "assays": ("13_experiments.csv", "assay_id", {r["assay_id"] for r in frozen.values()}),
        "references": (
            "15_references.csv",
            "reference_id",
            {r["reference_ids"] for r in frozen.values()},
        ),
    }
    lookup = {}
    for name, (filename, key, wanted) in keys.items():
        with (csv_root / filename).open(encoding="utf-8-sig", newline="") as stream:
            lookup[name] = {r[key]: r for r in csv.DictReader(stream) if r[key] in wanted}
        if set(lookup[name]) != wanted:
            raise ValueError("An exact original archive row is missing: " + name)
    family_path = project_path(
        next(name for name in audit["source_sha256"] if name.endswith("feature_rows_index.tsv"))
    )
    groups = reconcile_groups(candidates, table(family_path))
    result, proofs = [], []
    for item in candidates:
        row = frozen[item["record_id"]]
        raw = json.loads(row["raw_json"])
        canonical, key, value = validate_record(
            item,
            row,
            lookup["observations"][raw["observation_id"]],
            lookup["compounds"][row["compound_id"]],
            lookup["assays"][row["assay_id"]],
            lookup["references"][row["reference_ids"]],
        )
        result.append(
            {
                **item,
                "target": format(math.log10(float(value)), ".17g"),
                "value": str(value),
                "unit": "ug/mL",
                "structure_key": key,
                "structure_eligible": "true",
                "sequence_eligible": "false",
                "sequence": "",
                "canonical_isomeric_smiles": canonical,
                "split_group": groups[item["record_id"]],
                "source_relation": "=",
                "source_raw_value": raw["original_value"],
                "source_raw_unit": raw["original_unit"],
                "canonical_measure": MEASURE,
                "training_admitted": "true",
                "model_training_admitted": "true",
                "primary_article_method_verified": "false",
                "hemolysis_fraction_threshold_verified": "false",
                "full_protocol_verified": "false",
                "molecular_weight_guessed": "false",
            }
        )
        proofs.append(
            {
                "record_id": item["record_id"],
                "source_original_mass": str(value),
                "source_unit": raw["original_unit"],
                "source_lane": row["lane"],
                "source_normalized_value_unchanged": row["value"],
                "source_normalized_unit": row["unit"],
                "source_mw_unit": lookup["compounds"][row["compound_id"]]["molecular_weight_unit"],
                "source_graph_key": key,
                "source_paper": PAPER,
                "original_archive_observation_sha256": signature(raw),
            }
        )
    if (
        len({r["canonical_identity_group"] for r in result}) != 24
        or len(set(groups.values())) != 23
    ):
        raise ValueError("The 24 full graphs and 23 conservative identity groups must be retained")
    return result, proofs, protected


def build_features(task_rows, source_proof, protected, output, base_run=BASE):
    output, base_run = Path(output).resolve(), Path(base_run).resolve()
    if output == base_run or output in base_run.parents or base_run in output.parents:
        raise ValueError("Feature output must be separate from immutable parent bank")
    base_path = base_run / "features/manifest.json"
    pointer_sha = sha(base_path)
    base = read(base_path)
    _verify_artifacts(base)
    registry_path = base_run / "representation_methods.json"
    registry = read(registry_path)
    methods = [
        m for m in registry["methods"] if m["source_kind"] == "structure" and m["independent"]
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten independent complete-graph methods are required")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if base["configuration"]["rdkit_version"] != rdkit_tools()["version"]:
        raise ValueError("RDKit differs from frozen parent algorithms")
    cache_source = Path(base["cache_dir"])
    old_index = {
        r["key"]: int(r["row_idx"]) for r in table(cache_source / groups["morgan"]["index"])
    }
    old_graphs = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in table(cache_source / "structure_source_smiles.tsv")
    }
    graphs = sorted(
        ({"key": r["structure_key"], "smiles": r["canonical_isomeric_smiles"]} for r in task_rows),
        key=lambda r: r["key"],
    )
    if len({r["key"] for r in graphs}) != len(graphs):
        raise ValueError("This 24-graph bank must not duplicate graph keys")
    reused = {r["key"] for r in graphs if r["key"] in old_index}
    new = [r for r in graphs if r["key"] not in reused]
    if len(reused) != 3 or len(new) != 21:
        raise ValueError("Expected exactly three reusable and 21 new source graphs")
    if any(
        old_graphs[key] != next(r["smiles"] for r in graphs if r["key"] == key) for key in reused
    ):
        raise ValueError("Parent full graph differs from original archive")
    config = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "parent_fingerprint": base["input_fingerprint"],
        "parent_pointer_sha256": pointer_sha,
        "registry_sha256": sha(registry_path),
        "source_sha256": {str(p): h for p, h in protected.items()},
        "source_proof": source_proof,
        "rdkit_version": rdkit_tools()["version"],
        "inherited_algorithm_configuration": base["configuration"],
        "extracted_graphs": len(new),
        "reused_graphs": len(reused),
        "sequence_policy": "No sequence projection or ESM generated",
        "max_structure_atoms": 1000,
    }
    fingerprint = signature(config)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        old = read(cache / "manifest.json")
        _verify_artifacts(old)
        return old
    extracted = {r["key"]: extract_all(r["smiles"], r["key"]) for r in new}
    for name, metadata in groups.items():
        before = np.load(safe_path(cache_source, metadata["matrix"]), mmap_mode="r")
        before_valid = np.load(safe_path(cache_source, metadata["valid"]), mmap_mode="r")
        matrix = np.asarray(
            [
                before[old_index[r["key"]]] if r["key"] in reused else extracted[r["key"]][0][name]
                for r in graphs
            ],
            dtype=metadata["dtype"],
        )
        valid = np.asarray(
            [
                before_valid[old_index[r["key"]]]
                if r["key"] in reused
                else extracted[r["key"]][1][name]
                for r in graphs
            ],
            dtype=bool,
        )
        metadata["shape"] = [len(graphs), metadata["shape"][1]]
        metadata["index"] = "structure_bank_index.tsv"
        if matrix.shape != tuple(metadata["shape"]):
            raise ValueError("Full graph feature shape differs: " + name)
        check_matrix(matrix, valid, metadata, name)
        for index, row in enumerate(graphs):
            if (
                row["key"] in reused
                and before[old_index[row["key"]]].tobytes() != matrix[index].tobytes()
            ):
                raise ValueError("Reused graph feature bytes changed: " + name)
        atomic_array(safe_path(cache, metadata["matrix"]), matrix)
        atomic_array(safe_path(cache, metadata["valid"]), valid)
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": r["key"], "row_idx": i} for i, r in enumerate(graphs)],
    )
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "canonical_isomeric_smiles"],
        [{"key": r["key"], "canonical_isomeric_smiles": r["smiles"]} for r in graphs],
    )
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {"record_id": r["record_id"], "structure_key": r["structure_key"], "sequence_key": ""}
            for r in task_rows
        ],
    )
    registry = {
        **copy.deepcopy(registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": MEASURE,
    }
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", config)
    write(
        cache / "new_graph_extraction_audit.json",
        [{"key": key, "errors": value[2]} for key, value in extracted.items()],
    )
    routes = {m["route"]: m["group_names"] for m in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": config,
        "record_index": "record_bank_index.tsv",
        "requested_records": 24,
        "structure_keys": 24,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "coverage": {name: {"representations": 24, "valid": 24, "invalid": 0} for name in groups},
        "scope": MEASURE,
    }
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    _verify_artifacts(base)
    if sha(base_path) != pointer_sha:
        raise ValueError("Parent feature pointer changed")
    for method in methods:
        payload = load_feature_route(task_rows, manifest, method["route"])
        if not payload["valid"].all() or payload["X"].shape != (24, method["dimensions"]):
            raise ValueError("A method did not load all source observations")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        {
            "status": "verified",
            "records": 24,
            "graphs": 24,
            "reused_graphs": 3,
            "new_graphs": 21,
            "feature_groups": len(groups),
            "independent_methods": 10,
            "fusion_dimensions": sum(m["shape"][1] for m in groups.values()),
            "artifact_hashes_verified": len(manifest["artifact_hashes"]),
            "parent_artifacts_unchanged": len(base["artifact_hashes"]),
            "parent_pointer_unchanged": True,
            "reused_rows_byte_identical": True,
            "all_records_all_methods_loaded": True,
            "no_sequence_or_esm_generated": True,
            "fingerprint": fingerprint,
        },
    )
    return manifest


def split_prechecks(rows, output, config):
    y, groups = validate_rows(rows, "regression")
    result = []
    for seed in SEEDS:
        settings = {**config, "seed": seed}
        split = freeze_outer_split(
            rows, "regression", Path(output) / f"split_prechecks/seed_{seed}.json", settings
        )
        train = np.flatnonzero(np.isin(groups, split["train_groups"]))
        test = np.flatnonzero(np.isin(groups, split["test_groups"]))
        inner = _inner_splits(y[train], groups[train], "regression", settings)
        if len(inner) != 5 or set(groups[train]) & set(groups[test]):
            raise ValueError("Source holdout or actual inner fivefold proof failed")
        result.append(
            {
                "seed": seed,
                "n_train": len(train),
                "n_holdout": len(test),
                "actual_inner_folds": len(inner),
                "train_record_ids": [rows[i]["record_id"] for i in train],
                "holdout_record_ids": [rows[i]["record_id"] for i in test],
                "folds": [
                    {
                        "train_record_ids": [rows[train[i]]["record_id"] for i in a],
                        "validation_record_ids": [rows[train[i]]["record_id"] for i in b],
                        "group_overlap": len(set(groups[train[a]]) & set(groups[train[b]])),
                    }
                    for a, b in inner
                ],
            }
        )
    return {
        "status": "verified_partitions_not_fitted",
        "protocol": "fixed_outer_group_holdout_with_fivefold_inner_selection",
        "source_paper": PAPER,
        "independent_external_validation": False,
        "seeds": result,
    }


def prepare(source=SOURCE, output=OUTPUT, base_run=BASE):
    output = Path(output).resolve()
    if (output / "delivery_receipt.json").exists():
        raise FileExistsError("Frozen derived admission already exists; use a new version")
    rows, proofs, protected = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_features(rows, proofs, protected, output, base_run)
    task_id = (
        "TASK_"
        + signature({"measure": MEASURE, "paper": PAPER, "assay": DESCRIPTION, "unit": "ug/mL"})[
            :24
        ]
    )
    table_path = "task_tables/" + task_id + ".tsv"
    (output / "task_tables").mkdir(exist_ok=True)
    write_tsv(output / table_path, list(rows[0]), rows)
    task = {
        "task_id": task_id,
        "endpoint": "MHC",
        "task_kind": "regression",
        "status": "eligible",
        "table_path": table_path,
        "n_records": 24,
        "n_groups": 23,
        "unit": "ug/mL",
        "target_transform": "log10",
        "inverse_transform": "pow10",
        "requested_inner_folds": 5,
        "task_definition": {
            "quantity": MEASURE,
            "raw_endpoint": "MHC",
            "source_paper": PAPER,
            "reported_assay_description": DESCRIPTION,
            "reported_species": "Homo sapiens",
            "fraction_hemolysis_threshold_verified": False,
            "full_protocol_verified": False,
            "primary_article_method_verified": False,
            "independent_official_activity_cache_verified": False,
            "molecular_weight_assumed": False,
            "original_source_unit_preserved": True,
            "unknown_protocol_fields": [
                "fraction_hemolysis_threshold",
                "exposure_duration",
                "readout_method",
            ],
        },
        "source_evidence_tier": (
            "pinned_archive_point_assay_full_graph_reference_original_mass_"
            "not_independent_primary_table"
        ),
        "scientific_warning": (
            "Same-paper source-qualified MHC benchmark in original mass units; "
            "threshold and methods incompletely verified; no external or clinical validation."
        ),
        "validation_protocol": "fixed_outer_group_holdout_with_fivefold_inner_selection",
    }
    tasks = {"schema": VERSION, "tasks": [task], "models_fitted": 0}
    write(output / "task_manifest.json", tasks)
    write(output / "targeted_task_manifest.json", tasks)
    config = {
        "resources": {"training_host": "node23", "n_jobs": 1, "nice": 10, "cpu_budget": 1},
        "training": {
            "routes": list(manifest["routes"]),
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
            "inner_folds": 5,
            "test_size": 0.3,
            "seed": 42,
            "min_rows": 10,
            "min_groups": 8,
            "search_backend": "optuna",
            "search_budget": 128,
            "n_jobs": 1,
            "refit_full": True,
            "svm_max_features": 16384,
            "histgb_max_features": 16384,
            "elasticnet_max_features": 16384,
        },
        "required_seeds": list(SEEDS),
        "validation_protocol": task["validation_protocol"],
    }
    write(output / "recommended_training_config.json", config)
    write(output / "fivefold_prechecks.json", split_prechecks(rows, output, config["training"]))
    write(
        output / "source_verification.json",
        {
            "records": proofs,
            "point_unit": "ug/mL",
            "no_mw_unit_guess": True,
            "no_inverse_uM_conversion": True,
            "protected_files": {str(p): h for p, h in protected.items()},
            "source_records_unchanged": True,
            "model_training_performed": False,
        },
    )
    if any(sha(p) != h for p, h in protected.items()):
        raise ValueError("Protected source data changed during preparation")
    write(
        output / "admission_receipt.json",
        {
            "status": "prepared_not_trained",
            "tasks": 1,
            "records": 24,
            "graphs": 24,
            "identity_groups": 23,
            "unit": "ug/mL",
            "measure": MEASURE,
            "protected_source_files_unchanged": len(protected),
            "no_mw_guess_or_molar_conversion": True,
            "source_threshold_fraction_verified": False,
            "feature_fingerprint": manifest["input_fingerprint"],
            "models_fitted": 0,
            "remote_operations": False,
        },
    )
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-run", type=Path, default=BASE)
    args = parser.parse_args()
    tasks = prepare(args.source, args.output, args.base_run)
    print(
        json.dumps(
            {
                "status": "prepared_not_trained",
                "task_id": tasks["tasks"][0]["task_id"],
                "records": 24,
            }
        )
    )


if __name__ == "__main__":
    main()
