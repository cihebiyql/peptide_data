"""Build an independent, structure-only bank for seven exact-source CLint rows."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from peptide_modeling_add_methods import extract_graph_methods
from peptide_modeling_add_smiles_methods import extract_smiles_methods, rdkit_tools
from peptide_modeling_clint_bridge import (
    SOURCE_SHA256,
    canonical_graph,
    load_admissions,
    read_tsv,
)
from peptide_modeling_expand_features import (
    extract_expanded_structure,
    read_json,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts, load_feature_route

VERSION = "peptide_exact_clint_structure_features_v1"
STRUCTURE_GROUPS = (
    "morgan",
    "descriptors",
    "graph_modifications",
    "morgan_r3",
    "atompair",
    "torsion",
    "maccs",
    "rdkit_full",
    "rdkit_path",
    "pattern",
    "layered",
    "estate",
    "erg",
)


def source_graphs(evidence_map, dataset_dir):
    """Revalidate source identity/measurement proof before extracting any graph."""
    evidence_map, dataset_dir = Path(evidence_map), Path(dataset_dir)
    manifest = read_json(evidence_map.parent / "manifest.json")
    if manifest.get("source_csv_sha256") != SOURCE_SHA256:
        raise ValueError("CLint supporting source SHA differs from the pinned source")
    records, gates, admission = load_admissions(evidence_map, dataset_dir)
    proof = {row["record_id"]: row for row in read_tsv(evidence_map)}
    result = []
    for row in records:
        item, gate = proof[row["record_id"]], gates[row["record_id"]]
        canonical, key = canonical_graph(item["source_smiles"])
        if canonical != item["canonical_smiles"] or key != item["canonical_identity_group"]:
            raise ValueError("Full canonical stereochemical source identity differs")
        if (
            gate["canonical_identity_group"] != key
            or gate["structure_model_eligible"] != "true"
            or gate["sequence_model_eligible"] != "false"
            or row.get("sequence")
        ):
            raise ValueError("CLint admission must remain full-graph-only")
        result.append(
            {"record_id": row["record_id"], "key": key, "canonical_isomeric_smiles": canonical}
        )
    if len(result) != 7 or len({row["record_id"] for row in result}) != 7:
        raise ValueError("Exactly seven unique source observations are required")
    if len({row["key"] for row in result}) != 7:
        raise ValueError("The seven source compounds must retain distinct full graphs")
    return sorted(result, key=lambda row: row["key"]), admission


def check_rows_index(path, graphs):
    if path is None:
        return None
    expected = {row["record_id"]: row["key"] for row in graphs}
    found, count = {}, 0
    for row in read_tsv(path):
        count += 1
        if row["record_id"] not in expected:
            continue
        if row["record_id"] in found:
            raise ValueError("Derived feature index repeats a CLint record")
        if (
            row["canonical_identity_group"] != expected[row["record_id"]]
            or row["structure_eligible"] != "true"
            or row["sequence_eligible"] != "false"
            or row.get("sequence")
        ):
            raise ValueError("Derived feature index differs from full-graph-only source proof")
        found[row["record_id"]] = row["canonical_identity_group"]
    if found != expected:
        raise ValueError("Derived feature index does not cover all seven exact source records")
    return {"sha256": sha(path), "rows": count, "matched_new_records": len(found)}


def atomic_array(path, array):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def extract_all(smiles, expected_key):
    expanded = extract_expanded_structure(smiles, max_atoms=1000, include_base=True)
    paths = extract_graph_methods(smiles, max_atoms=1000)
    additional = extract_smiles_methods(smiles, max_atoms=1000, expected_key=expected_key)
    values = {**expanded["values"], **paths["values"], **additional["values"]}
    valid = {**expanded["valid"], **paths["valid"], **additional["valid"]}
    if set(values) != set(STRUCTURE_GROUPS) or set(valid) != set(STRUCTURE_GROUPS):
        raise ValueError("Structural extractor schemas differ from the declared 13 groups")
    return values, valid, expanded["errors"] + paths["errors"] + additional["errors"]


def build_features(base_run, evidence_map, dataset_dir, run_dir, rows_index=None):
    base_run, run_dir = Path(base_run).resolve(), Path(run_dir).resolve()
    evidence_map, dataset_dir = Path(evidence_map).resolve(), Path(dataset_dir).resolve()
    if base_run == run_dir or base_run in run_dir.parents or run_dir in base_run.parents:
        raise ValueError("The new CLint run must be separate from the immutable source bank")
    base = read_json(base_run / "features/manifest.json")
    _verify_artifacts(base)
    registry_path = base_run / "representation_methods.json"
    original_registry = read_json(registry_path)
    methods = [
        item
        for item in original_registry["methods"]
        if item["source_kind"] == "structure" and item["independent"] is True
    ]
    if len(methods) != 10 or len({item["algorithm_family_id"] for item in methods}) != 10:
        raise ValueError("The parent must provide ten distinct structural algorithm families")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if any(group["key_kind"] != "canonical_identity_group" for group in groups.values()):
        raise ValueError("No sequence or synthetic-parent feature group may enter this bank")
    source_version = base["configuration"]["additional_smiles_methods"]["rdkit_version"]
    if rdkit_tools()["version"] != source_version:
        raise ValueError("RDKit differs from the frozen parent feature algorithm version")
    graphs, admission = source_graphs(evidence_map, dataset_dir)
    rows_receipt = check_rows_index(rows_index, graphs)
    modules = (
        Path(__file__),
        Path(__file__).with_name("peptide_modeling_features.py"),
        Path(__file__).with_name("peptide_modeling_expand_features.py"),
        Path(__file__).with_name("peptide_modeling_add_methods.py"),
        Path(__file__).with_name("peptide_modeling_add_smiles_methods.py"),
        Path(__file__).with_name("peptide_modeling_clint_bridge.py"),
    )
    configuration = {
        "version": VERSION,
        "source_feature_fingerprint": base["input_fingerprint"],
        "source_feature_manifest_sha256": sha(base_run / "features/manifest.json"),
        "source_registry_sha256": sha(registry_path),
        "source_admission": admission,
        "source_graphs": graphs,
        "rows_index": rows_receipt,
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "implementation_sha256": {path.name: sha(path) for path in modules},
        "inherited_algorithm_configuration": base["configuration"],
        "sequence_policy": "No sequence, parent projection or ESM is constructed",
        "max_structure_atoms": 1000,
    }
    fingerprint = signature(configuration)
    cache_root = run_dir / "features"
    cache = cache_root / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        manifest = read_json(cache / "manifest.json")
        if manifest["input_fingerprint"] != fingerprint:
            raise ValueError("Feature cache fingerprint collision")
        _verify_artifacts(manifest)
        write_json(cache_root / "manifest.json", manifest)
        return manifest
    write_json(cache / "pending.json", configuration)
    computed = [extract_all(row["canonical_isomeric_smiles"], row["key"]) for row in graphs]
    errors = []
    for name, metadata in groups.items():
        metadata["shape"] = [len(graphs), metadata["shape"][1]]
        metadata["index"] = "structure_bank_index.tsv"
        matrix = np.asarray([row[0][name] for row in computed], dtype=metadata["dtype"])
        masks = np.asarray([row[1][name] for row in computed], dtype=bool)
        if matrix.shape != tuple(metadata["shape"]):
            raise ValueError("Extracted structural feature shape differs: " + name)
        if np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
            raise ValueError("A new structural feature row is nonfinite or entirely missing")
        if not metadata.get("allow_partial_missing", False) and not np.isfinite(matrix).all():
            raise ValueError("Undeclared partial missing feature values: " + name)
        if (
            np.isnan(matrix).any()
            and (np.isnan(matrix).mean(axis=1) > metadata["max_missing_fraction"]).any()
        ):
            raise ValueError("Partial descriptor missingness exceeds its frozen contract")
        if not masks.all():
            raise ValueError("A source-verified CLint graph failed a declared method: " + name)
        atomic_array(safe_path(cache, metadata["matrix"]), matrix)
        atomic_array(safe_path(cache, metadata["valid"]), masks)
    for row, computed_row in zip(graphs, computed, strict=True):
        errors.append({"record_id": row["record_id"], "key": row["key"], "errors": computed_row[2]})
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": row["key"], "row_idx": i} for i, row in enumerate(graphs)],
    )
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["record_id", "key", "canonical_isomeric_smiles"],
        graphs,
    )
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {"record_id": row["record_id"], "structure_key": row["key"], "sequence_key": ""}
            for row in graphs
        ],
    )
    registry = {
        **copy.deepcopy(original_registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": "structure_only_exact_source_clint",
    }
    write_json(cache / "representation_methods.json", registry)
    write_json(cache / "source_parent_registry.json", original_registry)
    write_json(cache / "extraction_errors.json", errors)
    for name in ("manifest.json", "primary_support.json", "verified_mapping.tsv"):
        shutil.copy2(evidence_map.parent / name, cache / ("source_clint_" + name))
    routes = {item["route"]: item["group_names"] for item in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": configuration,
        "record_index": "record_bank_index.tsv",
        "requested_records": 7,
        "structure_keys": 7,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "coverage": {name: {"representations": 7, "valid": 7, "invalid": 0} for name in groups},
        "scope": "Seven exact-source modified graphs only; no parent-sequence inference",
    }
    manifest["artifact_hashes"] = {
        path.relative_to(cache).as_posix(): sha(path)
        for path in cache.rglob("*")
        if path.is_file() and path.name != "pending.json"
    }
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    _verify_artifacts(base)
    write_json(cache_root / "manifest.json", manifest)
    write_json(run_dir / "representation_methods.json", registry)
    receipt = {
        "status": "verified",
        "input_fingerprint": fingerprint,
        "new_records": 7,
        "new_structures": 7,
        "structural_groups": len(groups),
        "independent_methods": 10,
        "structure_fusion_dimensions": sum(group["shape"][1] for group in groups.values()),
        "source_artifact_hashes_unchanged": len(base["artifact_hashes"]),
        "artifact_hashes_verified": len(manifest["artifact_hashes"]),
        "no_sequence_or_esm_constructed": True,
        "rows_index": rows_receipt,
        "model_training_performed": False,
    }
    for method in methods:
        payload = load_feature_route(
            [
                {
                    "record_id": row["record_id"],
                    "structure_eligible": "true",
                    "sequence_eligible": "false",
                }
                for row in graphs
            ],
            manifest,
            method["route"],
        )
        if not payload["valid"].all():
            raise ValueError("Generic feature loader does not verify all seven method rows")
    write_json(run_dir / "clint_feature_verification.json", receipt)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--evidence-map", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows-index", type=Path)
    args = parser.parse_args()
    manifest = build_features(
        args.base_run, args.evidence_map, args.dataset_dir, args.run_dir, args.rows_index
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "fingerprint": manifest["input_fingerprint"],
                "cache_dir": manifest["cache_dir"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    csv.field_size_limit(10_000_000)
    main()
