"""Append source-proven DRAMP graphs to a new, immutable feature-bank version."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
from peptide_modeling_add_methods import extract_graph_methods, ordered_index
from peptide_modeling_dramp_bridge import (
    CHEMISTRY_REQUIRED,
    derived_representation,
    load_evidence_map,
    text_sha,
)
from peptide_modeling_expand_features import (
    copy_verified,
    definitions,
    extract_expanded_structure,
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts

VERSION = "peptide_dramp_proven_representation_overlay_v1"
DONE_ARRAYS = ("expanded_structure_done.npy", "additional_methods_done.npy")


def load_verified_overlay(dataset_dir, overlay_path, evidence_map=None):
    """Recheck original records, pinned source bytes, chemistry and graph derivation."""
    dataset_dir, overlay_path = Path(dataset_dir), Path(overlay_path)
    evidence_map = Path(evidence_map or overlay_path.parent / "verified_mapping.tsv")
    source = read_json(evidence_map.parent / "manifest.json")
    if sha(overlay_path) != source["representation_candidates_sha256"]:
        raise ValueError("DRAMP representation overlay SHA mismatch")
    database = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    database.row_factory = sqlite3.Row
    try:
        originals = [
            dict(row)
            for row in database.execute(
                "SELECT * FROM records WHERE source_code='dramp_v4' "
                "AND endpoint='cytotoxicity_EC50' AND lane='strict_numeric'"
            )
        ]
    finally:
        database.close()
    proofs, _ = load_evidence_map(evidence_map, originals, sha(dataset_dir / "manifest.json"))
    candidates = list(rows(overlay_path))
    return validate_candidates(candidates, proofs), source


def validate_candidates(candidates, proofs):
    """Bound this repair to the explicitly sourced all-L free-terminal chemistry."""
    if not candidates or len({item["record_id"] for item in candidates}) != len(candidates):
        raise ValueError("DRAMP overlay needs unique, nonempty source records")
    if {item["record_id"] for item in candidates} != set(proofs):
        raise ValueError("DRAMP overlay and verified source records differ")
    result = []
    for item in candidates:
        evidence = proofs[item["record_id"]]
        chemistry = json.loads(evidence["source_chemistry_json"])
        if any(chemistry.get(key) != value for key, value in CHEMISTRY_REQUIRED.items()):
            raise ValueError("DRAMP chemistry is outside the explicit free-terminal L scope")
        sequence = evidence["source_sequence"]
        if (
            not sequence
            or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
            or "C" in sequence
            or text_sha(sequence) != evidence["source_sequence_sha256"]
        ):
            raise ValueError("DRAMP sequence or its identity proof is invalid")
        derived = derived_representation(evidence)
        if any(item.get(key) != value for key, value in derived.items()):
            raise ValueError("DRAMP graph differs from its exact source derivation")
        if item.get("source_entry_id") != evidence["source_entry_id"]:
            raise ValueError("DRAMP source identity differs from its verified observation")
        result.append(dict(item))
    if len({item["derived_sequence"] for item in result}) != len(result):
        raise ValueError("This bounded overlay requires distinct source sequences")
    if len({item["derived_canonical_identity_group"] for item in result}) != len(result):
        raise ValueError("DRAMP overlay has a canonical graph collision")
    return sorted(result, key=lambda item: item["record_id"])


def checked_aliases(original_rows, overlays, structure_keys, sequence_keys):
    """Change only proven record aliases, retaining the complete source record ID set."""
    aliases = {item["record_id"]: dict(item) for item in original_rows}
    if len(aliases) != len(original_rows):
        raise ValueError("Source bank contains duplicate record IDs")
    for item in overlays:
        record_id, sequence = item["record_id"], item["derived_sequence"]
        key = item["derived_canonical_identity_group"]
        if record_id not in aliases:
            raise ValueError("DRAMP overlay record is absent from the labelled source bank")
        if sequence not in sequence_keys:
            raise ValueError("DRAMP sequence needs a separately verified sequence/ESM extraction")
        if key in structure_keys:
            raise ValueError("DRAMP graph is already present; do not append a duplicate identity")
        row = aliases[record_id]
        if row["structure_key"] or row["sequence_key"] not in {"", sequence}:
            raise ValueError("DRAMP overlay would replace an existing conflicting representation")
        row.update(structure_key=key, sequence_key=sequence)
    return [aliases[item["record_id"]] for item in original_rows]


def extract_overlay_graph(smiles):
    result = extract_expanded_structure(smiles, include_base=True)
    additional = extract_graph_methods(smiles)
    result["values"].update(additional["values"])
    result["valid"].update(additional["valid"])
    result["errors"].extend(additional["errors"])
    return result


def append_array(source, destination, tail):
    """Write a new inode and verify the old prefix without editing source memmaps."""
    original = np.load(source, mmap_mode="r", allow_pickle=False)
    tail = np.asarray(tail, dtype=original.dtype)
    if tail.shape[1:] != original.shape[1:]:
        raise ValueError("Derived feature dimensions do not match the frozen algorithm contract")
    target = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=original.dtype,
        shape=(len(original) + len(tail), *original.shape[1:]),
    )
    target[: len(original)] = original
    target[len(original) :] = tail
    target.flush()
    if not np.array_equal(target[: len(original)], original, equal_nan=True):
        raise ValueError("Inherited feature prefix changed")
    del target


def apply_overlay(
    base_run, dataset_dir, overlay_path, output_run, rows_index=None, evidence_map=None
):
    base_run, dataset_dir, overlay_path, output_run = map(
        Path, (base_run, dataset_dir, overlay_path, output_run)
    )
    if (
        base_run.resolve() == output_run.resolve()
        or base_run.resolve() in output_run.resolve().parents
        or output_run.resolve() in base_run.resolve().parents
    ):
        raise ValueError("Derived representations need a separate sibling output run")
    if (
        dataset_dir.resolve() == output_run.resolve()
        or dataset_dir.resolve() in output_run.resolve().parents
        or output_run.resolve() in dataset_dir.resolve().parents
    ):
        raise ValueError("Derived features must not overlap the frozen dataset")
    pointer = base_run / "features/manifest.json"
    original = read_json(pointer)
    _verify_artifacts(original)
    source_cache = Path(original["cache_dir"])
    overlays, source_proof = load_verified_overlay(dataset_dir, overlay_path, evidence_map)
    structural = {
        name: group
        for name, group in original["groups"].items()
        if group["key_kind"] == "canonical_identity_group"
    }
    if len(structural) != 11:
        raise ValueError("The versioned source bank must provide all eleven structure groups")
    if original["groups"]["rdkit_full"]["feature_names"] != definitions()["rdkit_full"][2]:
        raise ValueError("RDKit descriptor names or transforms differ from the source bank")
    structure_index = structural["morgan"]["index"]
    indexed = ordered_index(source_cache / structure_index)
    sequence_index = ordered_index(source_cache / original["groups"]["sequence"]["index"])
    original_aliases = list(rows(source_cache / original["record_index"]))
    if len(original_aliases) != int(original["requested_records"]):
        raise ValueError("The labelled source record inventory is incomplete")
    updated_aliases = checked_aliases(
        original_aliases,
        overlays,
        {item["key"] for item in indexed},
        {item["key"] for item in sequence_index},
    )
    for group in original["groups"].values():
        if group["key_kind"] == "sequence":
            present = {item["key"] for item in ordered_index(source_cache / group["index"])}
            if any(item["derived_sequence"] not in present for item in overlays):
                raise ValueError("A reused sequence/ESM group lacks the proven sequence key")
    if rows_index is not None:
        indexed_records = {item["record_id"]: item for item in rows(rows_index)}
        if set(indexed_records) != {item["record_id"] for item in original_aliases}:
            raise ValueError("Derived row inventory changes the labelled source record ID set")
        for item in overlays:
            record = indexed_records[item["record_id"]]
            if (
                record.get("sequence") != item["derived_sequence"]
                or record.get("canonical_identity_group")
                != item["derived_canonical_identity_group"]
                or record.get("structure_smiles_sha256") != item["derived_smiles_sha256"]
                or record.get("sequence_eligible", "").lower() != "true"
                or record.get("structure_eligible", "").lower() != "true"
            ):
                raise ValueError(
                    "Derived row inventory does not match the exact representation proof"
                )
    count = len(indexed)
    if any(
        group["shape"][0] != count or group["index"] != structure_index
        for group in structural.values()
    ):
        raise ValueError("Source structural groups do not share one canonical graph universe")
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "expanded_helper_sha256": sha(
            Path(__file__).with_name("peptide_modeling_expand_features.py")
        ),
        "method_helper_sha256": sha(Path(__file__).with_name("peptide_modeling_add_methods.py")),
        "base_helper_sha256": sha(Path(__file__).with_name("peptide_modeling_features.py")),
        "proof_loader_sha256": sha(Path(__file__).with_name("peptide_modeling_dramp_bridge.py")),
        "base_manifest_sha256": sha(pointer),
        "base_input_fingerprint": original["input_fingerprint"],
        "overlay_sha256": sha(overlay_path),
        "evidence_mapping_sha256": sha(
            evidence_map or overlay_path.parent / "verified_mapping.tsv"
        ),
        "source_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "frozen_gates_sha256": (
            sha(dataset_dir / "representation_gates.tsv")
            if (dataset_dir / "representation_gates.tsv").exists()
            else None
        ),
        "source_xlsx_sha256": source_proof["source_xlsx_sha256"],
        "source_pin_sha256": source_proof["source_pin_sha256"],
        "rows_index_sha256": sha(rows_index) if rows_index is not None else None,
        "rdkit_version": overlays[0]["rdkit_version"],
        "new_graphs": [item["derived_canonical_identity_group"] for item in overlays],
        "gate_policy": "derived_exact_source_proof_only;frozen_gates_unchanged",
    }
    fingerprint = signature(configuration)
    cache = output_run / "features" / fingerprint[:16]
    destination_pointer = output_run / "features/manifest.json"
    if destination_pointer.exists():
        if read_json(destination_pointer)["input_fingerprint"] != fingerprint:
            raise ValueError("Refusing to overwrite an unrelated derived feature pointer")
        manifest = read_json(destination_pointer)
        _verify_artifacts(manifest)
        return manifest
    if cache.exists():
        raise FileExistsError("Partial derived cache requires explicit inspection before retry")
    cache.mkdir(parents=True)
    changed = {structure_index, original["record_index"], "structure_source_smiles.tsv"}
    for group in structural.values():
        changed.update((group["matrix"], group["valid"]))
    changed.update(name for name in DONE_ARRAYS if (source_cache / name).exists())
    for relative in original["artifact_hashes"]:
        if relative not in changed:
            copy_verified(safe_path(source_cache, relative), safe_path(cache, relative))
    extracted = [extract_overlay_graph(item["derived_smiles"]) for item in overlays]
    manifest = copy.deepcopy(original)
    for name, group in structural.items():
        if any(name not in item["values"] or name not in item["valid"] for item in extracted):
            raise ValueError("A source-proven graph lacks a requested structure algorithm")
        append_array(
            source_cache / group["matrix"],
            cache / group["matrix"],
            [item["values"][name] for item in extracted],
        )
        append_array(
            source_cache / group["valid"],
            cache / group["valid"],
            [item["valid"][name] for item in extracted],
        )
        manifest["groups"][name]["shape"][0] += len(overlays)
        matrix = np.load(cache / group["matrix"], mmap_mode="r")
        valid = np.load(cache / group["valid"], mmap_mode="r")
        manifest["coverage"][name] = {
            "representations": len(valid),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "all_zero_rows": int(np.sum(valid & ~np.any(matrix != 0, axis=1))),
        }
        del matrix, valid
    for name in DONE_ARRAYS:
        if (source_cache / name).exists():
            append_array(source_cache / name, cache / name, np.ones(len(overlays), dtype=bool))
    additions = [
        {"key": item["derived_canonical_identity_group"], "row_idx": count + position}
        for position, item in enumerate(overlays)
    ]
    write_tsv(cache / structure_index, ["key", "row_idx"], [*indexed, *additions])
    source_smiles = list(rows(source_cache / "structure_source_smiles.tsv"))
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "row_idx", "canonical_isomeric_smiles"],
        [
            *source_smiles,
            *[
                {**index, "canonical_isomeric_smiles": item["derived_smiles"]}
                for index, item in zip(additions, overlays, strict=True)
            ],
        ],
    )
    write_tsv(cache / original["record_index"], list(original_aliases[0]), updated_aliases)
    write_tsv(cache / "derived_representation_overlay.tsv", list(overlays[0]), overlays)
    errors = [
        {"record_id": item["record_id"], "errors": extracted[index]["errors"]}
        for index, item in enumerate(overlays)
    ]
    write_json(cache / "derived_structure_audit.json", errors)
    copy_verified(pointer, cache / "source_overlay_feature_manifest.json")
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        cache_reused=False,
    )
    manifest["configuration"] = {
        "source_configuration": original["configuration"],
        "derived_representation_overlay": configuration,
    }
    manifest["artifact_hashes"] = {
        relative: sha(safe_path(cache, relative)) for relative in original["artifact_hashes"]
    }
    extras = [
        "derived_representation_overlay.tsv",
        "derived_structure_audit.json",
        "source_overlay_feature_manifest.json",
    ]
    manifest["artifact_hashes"].update({name: sha(cache / name) for name in extras})
    unchanged = [name for name in original["artifact_hashes"] if name not in changed]
    if any(
        manifest["artifact_hashes"][name] != original["artifact_hashes"][name] for name in unchanged
    ):
        raise ValueError("An unchanged sequence, ESM or historical artifact changed")
    if sha(pointer) != configuration["base_manifest_sha256"]:
        raise ValueError("Source feature pointer changed during derived extraction")
    if sha(dataset_dir / "manifest.json") != configuration["source_manifest_sha256"]:
        raise ValueError("Frozen dataset manifest changed during derived extraction")
    if configuration["frozen_gates_sha256"] is not None and (
        sha(dataset_dir / "representation_gates.tsv") != configuration["frozen_gates_sha256"]
    ):
        raise ValueError("Frozen representation gates changed during derived extraction")
    _verify_artifacts(original)
    receipt = {
        "schema": VERSION,
        "status": "completed",
        "input_fingerprint": fingerprint,
        "source_input_fingerprint": original["input_fingerprint"],
        "source_pointer_unchanged": True,
        "source_artifacts_verified": len(original["artifact_hashes"]),
        "unchanged_artifacts_byte_identical": len(unchanged),
        "structure_groups_old_prefix_equal": sorted(structural),
        "original_structure_count": count,
        "new_structure_count": count + len(overlays),
        "derived_records": len(overlays),
        "record_ids_unchanged": True,
        "record_count": len(updated_aliases),
        "sequence_universe_unchanged": len(sequence_index),
        "all_new_graph_groups_valid": all(all(item["valid"].values()) for item in extracted),
        "frozen_dataset_and_gates_unchanged": True,
        "historical_artifacts": (
            "Inactive source snapshots retain their original extraction universes"
        ),
        "method_count": original["representation_methods"]["independent_method_count"],
        "feature_dimensions": sum(group["shape"][1] for group in manifest["groups"].values()),
        "configuration": configuration,
    }
    write_json(cache / "derived_feature_overlay_receipt.json", receipt)
    manifest["artifact_hashes"]["derived_feature_overlay_receipt.json"] = sha(
        cache / "derived_feature_overlay_receipt.json"
    )
    write_json(cache / "manifest.json", manifest)
    output_run.mkdir(parents=True, exist_ok=True)
    write_json(output_run / "derived_feature_overlay_receipt.json", receipt)
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
    parser.add_argument("--evidence-map", type=Path)
    args = parser.parse_args()
    manifest = apply_overlay(
        args.base_run,
        args.dataset,
        args.overlay,
        args.run_dir,
        args.rows_index,
        args.evidence_map,
    )
    print(
        json.dumps(
            {"input_fingerprint": manifest["input_fingerprint"], "cache_dir": manifest["cache_dir"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
