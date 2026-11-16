"""Append source-proven IC50 sequences and resolved graphs to an immutable bank.

Cysteine-bearing primary sequences remain sequence-only. Fresh ESM embeddings
must match the frozen model, pooling and extraction implementation contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from peptide_modeling_add_methods import ordered_index
from peptide_modeling_apply_dramp_overlay import (
    DONE_ARRAYS,
    extract_overlay_graph,
)
from peptide_modeling_apply_dramp_overlay import (
    append_array as append_source_array,
)
from peptide_modeling_dramp_cytotoxicity_bridge import load_representation_overlay
from peptide_modeling_expand_features import (
    copy_verified,
    definitions,
    extract_expanded_sequence,
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts, extract_sequence

VERSION = "peptide_cytotoxicity_proven_sequence_graph_append_v1"
SEQUENCE_GROUPS = {"sequence", "dpc", "sequence_extra", "esm2"}


def append_array(source, destination, tail):
    append_source_array(source, destination, tail)
    before = np.load(source, mmap_mode="r", allow_pickle=False)
    after = np.load(destination, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(before.view(np.uint8), after[: len(before)].view(np.uint8)):
        raise ValueError("Inherited array prefix changed its exact bytes")


def load_verified_overlay(dataset_dir, overlay_path):
    dataset_dir, overlay_path = Path(dataset_dir), Path(overlay_path)
    database = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    database.row_factory = sqlite3.Row
    try:
        originals = [
            dict(row)
            for row in database.execute(
                "SELECT * FROM records WHERE source_code='dramp_v4' "
                "AND endpoint='cytotoxicity_IC50' AND lane='strict_numeric'"
            )
        ]
    finally:
        database.close()
    overlays, _ = load_representation_overlay(
        overlay_path, originals, sha(dataset_dir / "manifest.json")
    )
    return sorted(overlays.values(), key=lambda row: row["record_id"]), read_json(
        overlay_path.parent / "manifest.json"
    )


def validate_overlay(overlays):
    if not overlays or len({row["record_id"] for row in overlays}) != len(overlays):
        raise ValueError("Source overlay needs unique nonempty record IDs")
    graphs = {}
    for row in overlays:
        sequence = row["derived_sequence"]
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Only exact complete standard primary sequences can be appended")
        if row["sequence_eligible"] != "true":
            raise ValueError("A source-proven primary sequence must have a derived sequence gate")
        if "C" in sequence:
            if row["structure_eligible"] != "false" or any(
                row.get(field)
                for field in (
                    "derived_smiles",
                    "derived_smiles_sha256",
                    "derived_canonical_identity_group",
                )
            ):
                raise ValueError("Unresolved cysteine topology cannot acquire a guessed graph")
        else:
            key, smiles = row["derived_canonical_identity_group"], row["derived_smiles"]
            digest = hashlib.sha256(smiles.encode()).hexdigest()
            if (
                row["structure_eligible"] != "true"
                or not smiles
                or row["derived_smiles_sha256"] != digest
                or key != "CANON_SMILES:" + digest
            ):
                raise ValueError("Source-proven graph identity or its SHA differs")
            if key in graphs and graphs[key] != smiles:
                raise ValueError("Canonical graph SHA collision")
            graphs[key] = smiles
    return graphs


def check_compiled_rows(path, aliases, overlays):
    compiled = {row["record_id"]: row for row in rows(path)}
    if len(compiled) != len(aliases) or set(compiled) != {row["record_id"] for row in aliases}:
        raise ValueError("Compiled feature inventory changes the full labelled ID set")
    for row in overlays:
        record = compiled[row["record_id"]]
        for source, field in (
            ("derived_sequence", "sequence"),
            ("derived_canonical_identity_group", "canonical_identity_group"),
            ("derived_smiles_sha256", "structure_smiles_sha256"),
            ("sequence_eligible", "sequence_eligible"),
            ("structure_eligible", "structure_eligible"),
        ):
            if record.get(field) != row[source]:
                raise ValueError("Compiled feature row contradicts its source proof: " + field)


def plan_overlay(original, overlays):
    source = Path(original["cache_dir"])
    graphs = validate_overlay(overlays)
    indexed_graphs = ordered_index(source / original["groups"]["morgan"]["index"])
    indexed_sequences = ordered_index(source / original["groups"]["sequence"]["index"])
    graph_keys = {row["key"] for row in indexed_graphs}
    sequence_keys = {row["key"] for row in indexed_sequences}
    graph_additions = sorted(set(graphs) - graph_keys)
    sequence_additions = sorted({row["derived_sequence"] for row in overlays} - sequence_keys)
    aliases = list(rows(source / original["record_index"]))
    by_id = {row["record_id"]: dict(row) for row in aliases}
    if len(by_id) != len(aliases) or len(aliases) != original["requested_records"]:
        raise ValueError("Source record inventory is incomplete or repeated")
    for row in overlays:
        if row["record_id"] not in by_id:
            raise ValueError("Proven source record is absent from the feature bank")
        alias = by_id[row["record_id"]]
        sequence, key = row["derived_sequence"], row["derived_canonical_identity_group"]
        if alias["sequence_key"] not in {"", sequence} or alias["structure_key"] not in {"", key}:
            raise ValueError("Derived identity would replace a conflicting verified alias")
        alias.update(sequence_key=sequence, structure_key=key)
    for name, group in original["groups"].items():
        expected = graph_keys if group["key_kind"] == "canonical_identity_group" else sequence_keys
        if {row["key"] for row in ordered_index(source / group["index"])} != expected:
            raise ValueError("Source group has a different identity universe: " + name)
    return {
        "graphs": graphs,
        "graph_keys": graph_additions,
        "sequence_keys": sequence_additions,
        "graph_index": indexed_graphs,
        "sequence_index": indexed_sequences,
        "aliases": aliases,
        "updated_aliases": [by_id[row["record_id"]] for row in aliases],
    }


def prepare_esm_input(base_run, dataset_dir, overlay_path, output_run):
    original = read_json(Path(base_run) / "features/manifest.json")
    _verify_artifacts(original)
    overlays, _ = load_verified_overlay(dataset_dir, overlay_path)
    plan = plan_overlay(original, overlays)
    output_run = Path(output_run)
    output_run.mkdir(parents=True, exist_ok=True)
    path = output_run / "esm_input_new_sequences.tsv"
    records = [
        {
            "key": sequence,
            "sequence": sequence,
            "sequence_kind": "source_complete_standard_sequence",
        }
        for sequence in plan["sequence_keys"]
    ]
    if path.exists():
        if list(rows(path)) != records:
            raise ValueError("Existing ESM input differs from the verified source sequences")
    else:
        write_tsv(path, ["key", "sequence", "sequence_kind"], records)
    return path, plan


def verified_esm_tail(source_cache, source_group, fresh_dir, expected_sequences, input_path):
    """Verify original and newly computed model provenance before joining arrays."""
    fresh_dir = Path(fresh_dir)
    historical_path = Path(source_group["matrix"]).parent / "esm2_manifest.json"
    historical = read_json(source_cache / historical_path)
    fresh = read_json(fresh_dir / "esm2_manifest.json")
    if fresh.get("status") != "complete" or not fresh.get("eligible_for_primary_comparison"):
        raise ValueError("Fresh ESM extraction is not complete primary evidence")
    for field in (
        "model_assets",
        "pooling",
        "max_length",
        "expected_dim",
        "dtype",
        "script_sha256",
    ):
        if fresh["contract"].get(field) != historical["contract"].get(field):
            raise ValueError("Fresh ESM model or extraction contract differs: " + field)
    if fresh["contract"]["input_sha256"] != sha(input_path):
        raise ValueError("Fresh ESM was computed from a different input list")
    if fresh["contract"]["expected_dim"] != source_group["shape"][1]:
        raise ValueError("Fresh ESM dimensions differ from the source group")
    for name, receipt in fresh["outputs"].items():
        path = safe_path(fresh_dir, name)
        if sha(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
            raise ValueError("Fresh ESM output artifact SHA mismatch")
    indexed = ordered_index(fresh_dir / "esm2_index.tsv")
    if {row["key"] for row in indexed} != set(expected_sequences):
        raise ValueError("Fresh ESM sequence universe is not the exact new key set")
    matrix = np.load(fresh_dir / "esm2.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(fresh_dir / "esm2_valid.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(indexed), source_group["shape"][1]) or valid.shape != (len(indexed),):
        raise ValueError("Fresh ESM arrays have a different shape")
    if str(matrix.dtype) != source_group["dtype"] or not valid.all():
        raise ValueError("New short source sequences must have valid matching-dtype ESM vectors")
    order = {row["key"]: int(row["row_idx"]) for row in indexed}
    tail = np.asarray([matrix[order[key]] for key in expected_sequences])
    if not np.isfinite(tail).all() or np.any(~np.any(tail != 0, axis=1)):
        raise ValueError("Fresh ESM vectors are nonfinite or falsely zero-filled")
    return tail, fresh, historical


def append_coverage(cache, group):
    matrix = np.load(cache / group["matrix"], mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / group["valid"], mmap_mode="r", allow_pickle=False)
    return {
        "representations": len(valid),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "all_zero_rows": int(np.sum(valid & ~np.any(matrix != 0, axis=1))),
    }


def apply_overlay(base_run, dataset_dir, overlay_path, output_run, rows_index, esm_dir):
    base_run, dataset_dir, overlay_path, output_run, rows_index, esm_dir = map(
        Path, (base_run, dataset_dir, overlay_path, output_run, rows_index, esm_dir)
    )
    for source_root in (base_run, dataset_dir):
        if (
            source_root.resolve() == output_run.resolve()
            or source_root.resolve() in output_run.resolve().parents
            or output_run.resolve() in source_root.resolve().parents
        ):
            raise ValueError("Derived bank must be independent of source and frozen dataset")
    pointer = base_run / "features/manifest.json"
    original = read_json(pointer)
    _verify_artifacts(original)
    source = Path(original["cache_dir"])
    overlays, proof = load_verified_overlay(dataset_dir, overlay_path)
    plan = plan_overlay(original, overlays)
    check_compiled_rows(rows_index, plan["aliases"], overlays)
    input_path, _ = prepare_esm_input(base_run, dataset_dir, overlay_path, output_run)
    esm_tail, fresh_esm, old_esm = verified_esm_tail(
        source, original["groups"]["esm2"], esm_dir, plan["sequence_keys"], input_path
    )
    structural = {
        name: group
        for name, group in original["groups"].items()
        if group["key_kind"] == "canonical_identity_group"
    }
    if len(structural) != 11 or set(original["groups"]) - set(structural) != SEQUENCE_GROUPS:
        raise ValueError("Source needs the frozen eleven graph and four sequence groups")
    if original["groups"]["rdkit_full"]["feature_names"] != definitions()["rdkit_full"][2]:
        raise ValueError("Descriptor schema differs from the frozen source algorithm")
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "helper_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                "peptide_modeling_features.py",
                "peptide_modeling_expand_features.py",
                "peptide_modeling_add_methods.py",
                "peptide_modeling_apply_dramp_overlay.py",
                "peptide_modeling_dramp_cytotoxicity_bridge.py",
                "peptide_modeling_dramp_bridge.py",
            )
        },
        "source_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "source_gates_sha256": sha(dataset_dir / "representation_gates.tsv"),
        "base_manifest_sha256": sha(pointer),
        "base_fingerprint": original["input_fingerprint"],
        "overlay_sha256": sha(overlay_path),
        "compiled_rows_sha256": sha(rows_index),
        "fresh_esm_manifest_sha256": sha(esm_dir / "esm2_manifest.json"),
        "new_sequence_input_sha256": sha(input_path),
        "source_xlsx_sha256": proof["source_xlsx_sha256"],
        "source_pin_sha256": proof["source_pin_sha256"],
        "new_graph_keys": plan["graph_keys"],
        "new_sequence_keys": plan["sequence_keys"],
        "cysteine_policy": "exact_primary_sequence_only;no_inference_of_covalent_topology",
    }
    fingerprint = signature(configuration)
    destination_pointer = output_run / "features/manifest.json"
    if destination_pointer.exists():
        manifest = read_json(destination_pointer)
        if manifest["input_fingerprint"] != fingerprint:
            raise ValueError("Refusing to replace an unrelated derived feature pointer")
        _verify_artifacts(manifest)
        return manifest
    cache = output_run / "features" / fingerprint[:16]
    if cache.exists():
        raise FileExistsError("Partial feature append needs explicit inspection before retry")
    cache.mkdir(parents=True)
    changed = {original["record_index"], "structure_source_smiles.tsv"}
    for group in original["groups"].values():
        changed.update((group["matrix"], group["valid"], group["index"]))
    changed.update(name for name in DONE_ARRAYS if (source / name).exists())
    esm_folder = Path(original["groups"]["esm2"]["matrix"]).parent
    for name in ("esm2_manifest.json", "checkpoint.json", "input_members.tsv"):
        if (source / esm_folder / name).exists():
            changed.add((esm_folder / name).as_posix())
    for relative in original["artifact_hashes"]:
        if relative not in changed:
            copy_verified(safe_path(source, relative), safe_path(cache, relative))
    graph_values = [extract_overlay_graph(plan["graphs"][key]) for key in plan["graph_keys"]]
    new_sequences = plan["sequence_keys"]
    basic = [extract_sequence(sequence) for sequence in new_sequences]
    extended = [extract_expanded_sequence(sequence) for sequence in new_sequences]
    if not all(row["valid"] for row in basic) or not all(row["valid"] for row in extended):
        raise ValueError("A new standard source sequence failed deterministic extraction")
    tails = {
        "sequence": [row["sequence"] for row in basic],
        "dpc": [row["dpc"] for row in extended],
        "sequence_extra": [row["sequence_extra"] for row in extended],
        "esm2": esm_tail,
    }
    manifest = copy.deepcopy(original)
    for name, group in original["groups"].items():
        destination = cache / group["matrix"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in structural:
            tail = [row["values"][name] for row in graph_values]
            valid = [row["valid"][name] for row in graph_values]
        else:
            tail, valid = tails[name], np.ones(len(new_sequences), dtype=bool)
        append_array(source / group["matrix"], destination, tail)
        append_array(source / group["valid"], cache / group["valid"], valid)
        manifest["groups"][name]["shape"][0] += len(tail)
        manifest["coverage"][name] = append_coverage(cache, manifest["groups"][name])
    indexes = {}
    for group in original["groups"].values():
        index = group["index"]
        if index in indexes:
            continue
        before = ordered_index(source / index)
        keys = (
            plan["graph_keys"] if group["key_kind"] == "canonical_identity_group" else new_sequences
        )
        indexes[index] = [
            *before,
            *[{"key": key, "row_idx": len(before) + i} for i, key in enumerate(keys)],
        ]
        write_tsv(cache / index, ["key", "row_idx"], indexes[index])
    for name in DONE_ARRAYS:
        if (source / name).exists():
            append_array(source / name, cache / name, np.ones(len(graph_values), dtype=bool))
    source_smiles = list(rows(source / "structure_source_smiles.tsv"))
    write_tsv(
        cache / "structure_source_smiles.tsv",
        list(source_smiles[0]),
        [
            *source_smiles,
            *[
                {
                    "key": key,
                    "row_idx": len(source_smiles) + i,
                    "canonical_isomeric_smiles": plan["graphs"][key],
                }
                for i, key in enumerate(plan["graph_keys"])
            ],
        ],
    )
    write_tsv(cache / original["record_index"], list(plan["aliases"][0]), plan["updated_aliases"])
    old_members_path = source / esm_folder / "input_members.tsv"
    if old_members_path.exists():
        members = list(rows(old_members_path))
        fresh_members = list(rows(esm_dir / "input_members.tsv"))
        key_positions = {
            row["key"]: row["row_idx"] for row in indexes[original["groups"]["esm2"]["index"]]
        }
        row_start = max(int(row["input_row"]) for row in members)
        additions = [
            {**row, "row_idx": key_positions[row["key"]], "input_row": row_start + i + 1}
            for i, row in enumerate(fresh_members)
        ]
        write_tsv(
            cache / esm_folder / "input_members.tsv", list(members[0]), [*members, *additions]
        )
    esm_provenance = {
        "schema": "peptide_esm_verified_prefix_append_v1",
        "status": "complete",
        "eligible_for_primary_comparison": True,
        "contract": old_esm["contract"],
        "source_manifest_sha256": sha(source / esm_folder / "esm2_manifest.json"),
        "fresh_manifest_sha256": sha(esm_dir / "esm2_manifest.json"),
        "prefix_rows": original["groups"]["esm2"]["shape"][0],
        "appended_rows": len(new_sequences),
        "counts": manifest["coverage"]["esm2"],
        "interpretation": "Exact primary sequences do not resolve missing covalent topology",
    }
    write_json(cache / esm_folder / "esm2_manifest.json", esm_provenance)
    if (source / esm_folder / "checkpoint.json").exists():
        write_json(
            cache / esm_folder / "checkpoint.json",
            {
                **esm_provenance,
                "resume_policy": "merged_bank_is_immutable;resume_fresh_producer_only",
            },
        )
    extras = [
        "cytotoxicity_representation_overlay.tsv",
        "cytotoxicity_structure_audit.json",
        "cytotoxicity_source_feature_manifest.json",
        "cytotoxicity_source_esm_manifest.json",
    ]
    write_tsv(cache / extras[0], list(overlays[0]), overlays)
    write_json(
        cache / extras[1],
        [
            {"graph_key": key, "errors": result["errors"]}
            for key, result in zip(plan["graph_keys"], graph_values, strict=True)
        ],
    )
    copy_verified(pointer, cache / extras[2])
    copy_verified(source / esm_folder / "esm2_manifest.json", cache / extras[3])
    for path in sorted(esm_dir.iterdir()):
        if path.is_file():
            relative = "cytotoxicity_fresh_esm/" + path.name
            copy_verified(path, cache / relative)
            extras.append(relative)
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        cache_reused=False,
    )
    manifest["configuration"] = {
        "source_configuration": original["configuration"],
        "cytotoxicity_overlay": configuration,
    }
    manifest["artifact_hashes"] = {
        name: sha(safe_path(cache, name)) for name in [*original["artifact_hashes"], *extras]
    }
    _verify_artifacts(original)
    for path, expected in (
        (pointer, configuration["base_manifest_sha256"]),
        (dataset_dir / "manifest.json", configuration["source_manifest_sha256"]),
        (dataset_dir / "representation_gates.tsv", configuration["source_gates_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError("Frozen source changed during the independent feature append")
    receipt = {
        "status": "completed",
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "source_fingerprint": original["input_fingerprint"],
        "configuration": configuration,
        "source_artifacts_unchanged": len(original["artifact_hashes"]),
        "record_ids_unchanged": True,
        "record_count": len(plan["aliases"]),
        "derived_records": len(overlays),
        "new_graph_rows": len(graph_values),
        "new_sequence_rows": len(new_sequences),
        "structure_count": len(plan["graph_index"]) + len(graph_values),
        "sequence_count": len(plan["sequence_index"]) + len(new_sequences),
        "sequence_only_topology_unresolved_records": sum(
            "C" in row["derived_sequence"] for row in overlays
        ),
        "all_old_group_prefixes_byte_equal": True,
        "all_new_graph_groups_valid": all(all(row["valid"].values()) for row in graph_values),
        "all_new_sequence_groups_valid": True,
        "frozen_dataset_and_gates_unchanged": True,
        "method_count": original["representation_methods"]["independent_method_count"],
        "feature_groups": len(manifest["groups"]),
        "feature_dimensions": sum(group["shape"][1] for group in manifest["groups"].values()),
    }
    write_json(cache / "cytotoxicity_feature_receipt.json", receipt)
    manifest["artifact_hashes"]["cytotoxicity_feature_receipt.json"] = sha(
        cache / "cytotoxicity_feature_receipt.json"
    )
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    write_json(output_run / "cytotoxicity_feature_receipt.json", receipt)
    write_json(output_run / "representation_methods.json", manifest["representation_methods"])
    write_json(destination_pointer, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows-index", type=Path)
    parser.add_argument("--esm-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        path, plan = prepare_esm_input(args.base_run, args.dataset, args.overlay, args.run_dir)
        print(json.dumps({"input": str(path), "new_sequences": len(plan["sequence_keys"])}))
    else:
        if args.rows_index is None or args.esm_dir is None:
            parser.error("Append requires --rows-index and --esm-dir")
        manifest = apply_overlay(
            args.base_run, args.dataset, args.overlay, args.run_dir, args.rows_index, args.esm_dir
        )
        print(
            json.dumps(
                {
                    "cache_dir": manifest["cache_dir"],
                    "input_fingerprint": manifest["input_fingerprint"],
                }
            )
        )


if __name__ == "__main__":
    main()
