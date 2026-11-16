"""Freeze a graph-only feature bank for the reviewed source-label safety benchmark."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from peptide_modeling_add_smiles_methods import rdkit_tools
from peptide_modeling_clint_features import STRUCTURE_GROUPS, atomic_array, extract_all
from peptide_modeling_expand_features import (
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts, canonical_key, load_feature_route

VERSION = "peptide_qualitative_cyto_structure_features_v1"
PINNED_EVIDENCE = {
    "audit.json": "574dcbaed44295f80c26048ca1de9f8c408c58c4bba1b46333716e15ba8ebbd3",
    "benchmark_admission_review.json": (
        "fca58c872cf0370143f52a4d6d9ca853b80fa171ad666529728be6ec168c2fa5"
    ),
    "official_crosschecks.json": "c36f8b04c162688b2fd38ac13670981e2c826c9e604b9385448bb1ede539272b",
    "frozen_lineage_crosschecks.json": (
        "3d8661030bcdabfeed0aa7e4dfccb8f4050a22de392e4b2a5bc3c75f600ae82a"
    ),
}


def validate_inputs(evidence_dir, task_table, source_graphs):
    """Require the exact reviewed identities/labels, independent of input row order."""
    evidence_dir = Path(evidence_dir)
    for relative, expected in PINNED_EVIDENCE.items():
        if sha(evidence_dir / relative) != expected:
            raise ValueError("Pinned qualitative evidence changed: " + relative)
    audit = read_json(evidence_dir / "audit.json")
    review = read_json(evidence_dir / "benchmark_admission_review.json")
    lineage = read_json(evidence_dir / "frozen_lineage_crosschecks.json")["records"]
    official = read_json(evidence_dir / "official_crosschecks.json")
    selected = set(review["admitted_candidate_activity_ids"])
    candidates = {row["activity_id"]: row for row in audit["candidate_records"]}
    original_ids = {
        row["activity_id"]: row["record_id"] for row in lineage if row["activity_id"] in selected
    }
    if len(selected) != 150 or set(candidates) != selected or set(original_ids) != selected:
        raise ValueError("Reviewed 150-record evidence set differs")
    if official["issue_counts"] or any(
        not all(item["checks"].values()) for item in official["records"]
    ):
        raise ValueError("Official identity and label verification is incomplete")
    expected = {
        original_ids[activity]: {
            "key": canonical_key(item["canonical_smiles"]),
            "smiles": item["canonical_smiles"],
            "label": item["label"],
        }
        for activity, item in candidates.items()
    }
    task_rows = list(rows(task_table))
    if len(task_rows) != 150 or len({item["record_id"] for item in task_rows}) != 150:
        raise ValueError("Task table must retain exactly 150 distinct frozen record IDs")
    for item in task_rows:
        proof = expected.get(item["record_id"])
        if proof is None or item.get("structure_key") != proof["key"]:
            raise ValueError("Task identity differs from reviewed full source graph")
        if float(item["y"]) != proof["label"]:
            raise ValueError("Task label differs from reviewed categorical evidence")
    graph_rows = list(rows(source_graphs))
    graph_map = {item["key"]: item for item in graph_rows}
    if len(graph_rows) != 100 or len(graph_map) != 100:
        raise ValueError("Source graph table must contain exactly 100 distinct full graphs")
    expected_graphs = {item["key"]: item["smiles"] for item in expected.values()}
    if set(graph_map) != set(expected_graphs):
        raise ValueError("Source graph universe differs from reviewed task records")
    for key, item in graph_map.items():
        if item["canonical_isomeric_smiles"] != expected_graphs[key]:
            raise ValueError("Source stereochemical SMILES differs from reviewed source")
    ordered = sorted(graph_map.values(), key=lambda item: item["key"])
    return task_rows, ordered


def check_matrix(matrix, masks, metadata, name):
    if not masks.all() or np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
        raise ValueError("Structural method failed for reviewed full graph: " + name)
    missing = np.isnan(matrix)
    if missing.any() and not metadata.get("allow_partial_missing", False):
        raise ValueError("Undeclared partial feature missingness: " + name)
    if missing.any() and (missing.mean(axis=1) > metadata["max_missing_fraction"]).any():
        raise ValueError("Partial feature missingness exceeds frozen contract: " + name)


def build_features(base_run, evidence_dir, task_table, source_graphs, run_dir):
    base_run, run_dir = Path(base_run).resolve(), Path(run_dir).resolve()
    if base_run == run_dir or base_run in run_dir.parents or run_dir in base_run.parents:
        raise ValueError("Output must be separate from immutable parent feature bank")
    task_rows, graphs = validate_inputs(evidence_dir, task_table, source_graphs)
    base_pointer = base_run / "features/manifest.json"
    base = read_json(base_pointer)
    _verify_artifacts(base)
    cache_source = Path(base["cache_dir"])
    registry_path = base_run / "representation_methods.json"
    original_registry = read_json(registry_path)
    methods = [
        item
        for item in original_registry["methods"]
        if item["source_kind"] == "structure" and item["independent"] is True
    ]
    if len(methods) != 10 or len({item["algorithm_family_id"] for item in methods}) != 10:
        raise ValueError("Parent must provide ten independent structural methods")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if any(item["key_kind"] != "canonical_identity_group" for item in groups.values()):
        raise ValueError("A sequence feature cannot enter this graph-only bank")
    if base["configuration"]["rdkit_version"] != rdkit_tools()["version"]:
        raise ValueError("RDKit differs from pinned parent algorithm version")
    parent_graphs = {
        item["key"]: item for item in rows(cache_source / "structure_source_smiles.tsv")
    }
    parent_index = {
        item["key"]: int(item["row_idx"]) for item in rows(cache_source / groups["morgan"]["index"])
    }
    for item in graphs:
        old = parent_graphs.get(item["key"])
        if (
            old is not None
            and old["canonical_isomeric_smiles"] != item["canonical_isomeric_smiles"]
        ):
            raise ValueError("Existing parent full graph does not match reviewed source")
    reused = {item["key"] for item in graphs if item["key"] in parent_index}
    missing_graphs = [item for item in graphs if item["key"] not in parent_index]
    parent_record_ids = {item["record_id"] for item in rows(cache_source / base["record_index"])}
    record_overlap = len({item["record_id"] for item in task_rows} & parent_record_ids)
    if len(reused) != 77 or len(missing_graphs) != 23:
        raise ValueError("Expected 77 frozen parent graphs and 23 source-verified new graphs")
    configuration = {
        "version": VERSION,
        "parent_feature_fingerprint": base["input_fingerprint"],
        "parent_manifest_sha256": sha(base_pointer),
        "parent_registry_sha256": sha(registry_path),
        "evidence_hashes": PINNED_EVIDENCE,
        "task_table_sha256": sha(task_table),
        "source_graphs_sha256": sha(source_graphs),
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "implementation_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "peptide_modeling_clint_features.py",
                "peptide_modeling_expand_features.py",
                "peptide_modeling_add_methods.py",
                "peptide_modeling_add_smiles_methods.py",
            )
        },
        "inherited_algorithm_configuration": base["configuration"],
        "sequence_policy": "No sequence, lossy parent projection or ESM constructed",
        "reused_graphs": len(reused),
        "extracted_graphs": len(missing_graphs),
        "parent_record_id_overlap": record_overlap,
        "records_using_reused_graphs": sum(item["structure_key"] in reused for item in task_rows),
        "max_structure_atoms": 1000,
    }
    fingerprint = signature(configuration)
    cache = run_dir / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        manifest = read_json(cache / "manifest.json")
        if manifest["input_fingerprint"] != fingerprint:
            raise ValueError("Feature fingerprint collision")
        _verify_artifacts(manifest)
        return manifest
    write_json(cache / "pending.json", configuration)
    extracted = {
        item["key"]: extract_all(item["canonical_isomeric_smiles"], item["key"])
        for item in missing_graphs
    }
    for name, metadata in groups.items():
        old_matrix = np.load(safe_path(cache_source, metadata["matrix"]), mmap_mode="r")
        old_valid = np.load(safe_path(cache_source, metadata["valid"]), mmap_mode="r")
        values, masks = [], []
        for item in graphs:
            key = item["key"]
            if key in reused:
                values.append(old_matrix[parent_index[key]])
                masks.append(old_valid[parent_index[key]])
            else:
                values.append(extracted[key][0][name])
                masks.append(extracted[key][1][name])
        matrix = np.asarray(values, dtype=metadata["dtype"])
        valid = np.asarray(masks, dtype=bool)
        metadata["shape"] = [100, metadata["shape"][1]]
        metadata["index"] = "structure_bank_index.tsv"
        if matrix.shape != tuple(metadata["shape"]):
            raise ValueError("Structural group shape differs: " + name)
        check_matrix(matrix, valid, metadata, name)
        atomic_array(safe_path(cache, metadata["matrix"]), matrix)
        atomic_array(safe_path(cache, metadata["valid"]), valid)
        for index, item in enumerate(graphs):
            if item["key"] in reused:
                before = np.ascontiguousarray(old_matrix[parent_index[item["key"]]]).view(np.uint8)
                after = np.ascontiguousarray(matrix[index]).view(np.uint8)
                if not np.array_equal(before, after):
                    raise ValueError("Existing feature row changed byte payload: " + name)
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": item["key"], "row_idx": i} for i, item in enumerate(graphs)],
    )
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "canonical_isomeric_smiles"],
        [
            {"key": item["key"], "canonical_isomeric_smiles": item["canonical_isomeric_smiles"]}
            for item in graphs
        ],
    )
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {
                "record_id": item["record_id"],
                "structure_key": item["structure_key"],
                "sequence_key": "",
            }
            for item in task_rows
        ],
    )
    registry = {
        **copy.deepcopy(original_registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": "reviewed_qualitative_cyto_full_graph_only",
    }
    write_json(cache / "representation_methods.json", registry)
    write_json(cache / "source_parent_manifest.json", base)
    write_json(cache / "source_evidence_receipt.json", configuration)
    write_json(
        cache / "new_graph_extraction_audit.json",
        [{"key": key, "errors": value[2]} for key, value in extracted.items()],
    )
    routes = {item["route"]: item["group_names"] for item in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": configuration,
        "record_index": "record_bank_index.tsv",
        "requested_records": 150,
        "structure_keys": 100,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "coverage": {name: {"representations": 100, "valid": 100, "invalid": 0} for name in groups},
        "scope": (
            "Source-reported qualitative cell-injury safety-screening benchmark; full graph only"
        ),
    }
    manifest["artifact_hashes"] = {
        path.relative_to(cache).as_posix(): sha(path)
        for path in cache.rglob("*")
        if path.is_file() and path.name != "pending.json"
    }
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    _verify_artifacts(base)
    if sha(base_pointer) != configuration["parent_manifest_sha256"]:
        raise ValueError("Parent feature pointer changed during extraction")
    for item in methods:
        payload = load_feature_route(task_rows, manifest, item["route"])
        if not payload["valid"].all() or payload["X"].shape != (150, item["dimensions"]):
            raise ValueError("Published method did not load all 150 reviewed records")
    write_json(run_dir / "features/manifest.json", manifest)
    write_json(run_dir / "representation_methods.json", registry)
    write_json(
        run_dir / "qualitative_cyto_feature_verification.json",
        {
            "status": "verified",
            "fingerprint": fingerprint,
            "records": 150,
            "graphs": 100,
            "reused_graphs": 77,
            "new_graphs": 23,
            "parent_record_id_overlap": record_overlap,
            "new_record_aliases": 150 - record_overlap,
            "records_using_reused_graphs": sum(
                item["structure_key"] in reused for item in task_rows
            ),
            "independent_methods": 10,
            "groups": len(groups),
            "fusion_dimensions": sum(item["shape"][1] for item in groups.values()),
            "artifact_hashes_verified": len(manifest["artifact_hashes"]),
            "parent_artifact_hashes_unchanged": len(base["artifact_hashes"]),
            "parent_pointer_unchanged": True,
            "reused_matrix_rows_byte_identical": True,
            "all_150_records_all_10_methods_loaded": True,
            "no_sequence_or_esm_constructed": True,
            "model_training_performed": False,
            "configuration": configuration,
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-run", "evidence-dir", "task-table", "source-graphs", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = build_features(
        args.base_run, args.evidence_dir, args.task_table, args.source_graphs, args.run_dir
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "cache_dir": manifest["cache_dir"],
                "fingerprint": manifest["input_fingerprint"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
