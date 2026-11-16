"""Recover only source-proven, all-L linear DRAMP human-RBC MHC10 structures."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import peptide_modeling_dramp_bridge as shared
from peptide_modeling_add_methods import ordered_index
from peptide_modeling_apply_dramp_overlay import (
    validate_candidates,
)
from peptide_modeling_expand_features import (
    copy_verified,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts

SCHEMA = "dramp_exact_human_mhc10_representation_bridge_v1"
FEATURE_SCHEMA = "peptide_dramp_mhc10_feature_followup_v1"
ANNOTATION = re.compile(
    r"^\s*\[Ref\.(?P<pmid>\d+)\]\s*MHC10\s*=\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*[\u00b5\u03bcu]M\s+"
    r"against\s+human\s+red\s+blood\s+cells\s*$",
    re.IGNORECASE,
)


def verify_observation(raw, source):
    if raw.get("source_code") != "dramp_v4" or raw.get("endpoint") != "hemolysis_MHC10":
        return None, "source_or_endpoint_outside_scope"
    if raw.get("upstream_eligible") != "true":
        return None, "upstream_source_gate_rejected"
    if raw.get("task_kind") != "regression" or raw.get("source_entry_id") != source["DRAMP_ID"]:
        return None, "source_identity_or_task_kind_mismatch"
    annotation = str(source.get("Hemolytic_activity", ""))
    matched = ANNOTATION.fullmatch(annotation)
    if matched is None:
        return None, "not_single_exact_human_mhc10_statement"
    if str(source.get("Pubmed_ID", "")).strip() != matched["pmid"]:
        return None, "source_reference_mismatch"
    if raw.get("raw_relation") not in {"", "="} or raw.get("relation") != "=":
        return None, "observation_not_exact"
    try:
        number = Decimal(matched["value"])
        if number != Decimal(raw["value"]) or number != Decimal(raw["raw_value"]):
            return None, "source_scalar_mismatch"
    except (InvalidOperation, KeyError):
        return None, "invalid_source_scalar"
    normalized = str(raw.get("raw_unit", "")).replace("\u00b5", "u").replace("\u03bc", "u")
    if normalized != "uM" or raw.get("unit") != "uM":
        return None, "source_unit_mismatch"
    context = json.loads(raw["condition_json"])
    original = json.loads(raw["raw_json"])
    source_context = context.get("source_context", {})
    if isinstance(source_context, str):
        source_context = json.loads(source_context)
    if source_context.get("raw_annotation", "").strip() != annotation.strip():
        return None, "original_source_annotation_mismatch"
    parsed = source_context.get("parsed_observation", {})
    if parsed.get("metric") != "MHC10" or parsed.get("measurement_snippet") != annotation:
        return None, "original_metric_or_scalar_alignment_missing"
    if context.get("semantics", {}).get("species") not in {"Homo sapiens", "human"}:
        return None, "original_human_rbc_species_not_explicit"
    sequence = str(source.get("Sequence", ""))
    if original.get("sequence") != sequence:
        return None, "original_sequence_mismatch"
    if original.get("identity_key") != "sequence:" + shared.text_sha(sequence):
        return None, "original_sequence_identity_hash_mismatch"
    if any(source.get(key) != value for key, value in shared.CHEMISTRY_REQUIRED.items()):
        return None, "source_chemistry_outside_scope"
    if any(source_context.get(key) != source.get(key) for key in shared.CHEMISTRY_REQUIRED):
        return None, "original_source_chemistry_mismatch"
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY") or "C" in sequence:
        return None, "ambiguous_sequence_or_possible_disulfide"
    return {
        "metric": "MHC10",
        "species": "human",
        "source_sequence": sequence,
        "source_sequence_sha256": shared.text_sha(sequence),
        "source_annotation_sha256": shared.text_sha(annotation.strip()),
        "source_reference_pmid": matched["pmid"],
        "source_chemistry_json": json.dumps(
            {key: source.get(key) for key in shared.CHEMISTRY_FIELDS}, sort_keys=True
        ),
    }, ""


def build_bridge(
    dataset_dir, output_dir, source_file=shared.DEFAULT_SOURCE, pin_file=shared.DEFAULT_PIN
):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    source_file, pin_file = Path(source_file).resolve(), Path(pin_file).resolve()
    sources = shared.pinned_source(source_file, pin_file)
    connection = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    matches, overlays, audit = [], [], []
    try:
        for item in connection.execute(
            "SELECT * FROM records WHERE source_code='dramp_v4' "
            "AND endpoint='hemolysis_MHC10' AND lane='strict_numeric'"
        ):
            raw = dict(item)
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = (
                verify_observation(raw, source) if source else (None, "source_id_missing")
            )
            audit.append({"record_id": raw["record_id"], "reason": reason or "verified"})
            if proof is None:
                continue
            evidence = {field: raw.get(field, "") for field in shared.RAW_MATCH_FIELDS}
            evidence.update(
                catalog_raw_json_sha256=shared.text_sha(raw["raw_json"]),
                source_xlsx_row=str(number),
                **proof,
            )
            matches.append(evidence)
            overlays.append(
                {
                    "record_id": raw["record_id"],
                    "source_entry_id": raw["source_entry_id"],
                    **shared.derived_representation(proof),
                }
            )
    finally:
        connection.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared.write_tsv(
        output_dir / "verified_mapping.tsv", list(matches[0]) if matches else ["record_id"], matches
    )
    shared.write_tsv(
        output_dir / "representation_overlay.tsv",
        list(overlays[0]) if overlays else ["record_id"],
        overlays,
    )
    shared.write_tsv(output_dir / "audit.tsv", ["record_id", "reason"], audit)
    manifest = {
        "schema": SCHEMA,
        "source_dataset": str(dataset_dir.resolve()),
        "source_catalogue": str((dataset_dir / "catalogue.sqlite").resolve()),
        "source_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "source_xlsx": str(source_file),
        "source_xlsx_sha256": sha(source_file),
        "source_pin": str(pin_file),
        "source_pin_sha256": sha(pin_file),
        "verified_mapping_sha256": sha(output_dir / "verified_mapping.tsv"),
        "representation_candidates_sha256": sha(output_dir / "representation_overlay.tsv"),
        "script_sha256": sha(__file__),
        "counts": {
            "observations": len(audit),
            "verified": len(matches),
            "rejected": len(audit) - len(matches),
        },
        "policy": "exact_source_chemistry_only;frozen_gates_and_labels_unchanged",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_representation_overlay(path, records, dataset_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest["source_manifest_sha256"] != dataset_manifest_sha256
    ):
        raise ValueError("MHC10 source schema or frozen dataset mismatch")
    for actual, expected in (
        (path, manifest["representation_candidates_sha256"]),
        (path.parent / "verified_mapping.tsv", manifest["verified_mapping_sha256"]),
        (manifest["source_xlsx"], manifest["source_xlsx_sha256"]),
        (manifest["source_pin"], manifest["source_pin_sha256"]),
    ):
        if sha(actual) != expected:
            raise ValueError("MHC10 source or evidence SHA mismatch")
    sources = shared.pinned_source(manifest["source_xlsx"], manifest["source_pin"])
    original = {row["record_id"]: row for row in records}
    proofs = {}
    connection = sqlite3.connect(
        Path(manifest["source_catalogue"]).resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        for evidence in shared.read_tsv(path.parent / "verified_mapping.tsv"):
            raw = original.get(evidence["record_id"])
            if raw is None or evidence["record_id"] in proofs:
                raise ValueError("MHC10 observation missing or repeated")
            if any(raw.get(field, "") != evidence[field] for field in shared.RAW_MATCH_FIELDS):
                raise ValueError("MHC10 proof differs from frozen observation")
            item = connection.execute(
                "SELECT raw_json FROM records WHERE record_id=?", (raw["record_id"],)
            ).fetchone()
            if item is None or shared.text_sha(item[0]) != evidence["catalog_raw_json_sha256"]:
                raise ValueError("MHC10 catalogue source payload mismatch")
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = (
                verify_observation({**raw, "raw_json": item[0]}, source)
                if source
                else (None, "missing_source")
            )
            if proof is None or any(evidence[key] != value for key, value in proof.items()):
                raise ValueError("MHC10 exact source proof failed: " + reason)
            if evidence["source_xlsx_row"] != str(number):
                raise ValueError("MHC10 source row mismatch")
            proofs[raw["record_id"]] = evidence
    finally:
        connection.close()
    candidates = validate_candidates(list(shared.read_tsv(path)), proofs)
    return {item["record_id"]: item for item in candidates}, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": sha(path),
        "verified_records": len(candidates),
        "frozen_gates_unchanged": True,
        "source_xlsx_sha256": manifest["source_xlsx_sha256"],
        "source_pin_sha256": manifest["source_pin_sha256"],
    }


def append_features(base_run, dataset_dir, overlay_path, output_run, rows_index):
    """Publish proven aliases to existing graphs; never duplicate an existing graph."""
    base_run, dataset_dir, overlay_path, output_run, rows_index = map(
        Path, (base_run, dataset_dir, overlay_path, output_run, rows_index)
    )
    if (
        base_run.resolve() == output_run.resolve()
        or base_run.resolve() in output_run.resolve().parents
        or output_run.resolve() in base_run.resolve().parents
        or dataset_dir.resolve() in output_run.resolve().parents
    ):
        raise ValueError("MHC10 output must be independent of frozen inputs")
    pointer = base_run / "features/manifest.json"
    original = json.loads(pointer.read_text(encoding="utf-8"))
    _verify_artifacts(original)
    source = Path(original["cache_dir"])
    connection = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        records = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM records WHERE source_code='dramp_v4' "
                "AND endpoint='hemolysis_MHC10' AND lane='strict_numeric'"
            )
        ]
    finally:
        connection.close()
    overlays, proof = load_representation_overlay(
        overlay_path, records, sha(dataset_dir / "manifest.json")
    )
    aliases = list(rows(source / original["record_index"]))
    by_id = {item["record_id"]: dict(item) for item in aliases}
    derived = {item["record_id"]: item for item in rows(rows_index)}
    if len(by_id) != len(aliases) or set(derived) != set(by_id):
        raise ValueError("MHC10 overlay changes the unique source record inventory")
    graph_index = {
        item["key"]: int(item["row_idx"])
        for item in ordered_index(source / original["groups"]["morgan"]["index"])
    }
    sequence_index = {
        item["key"]: int(item["row_idx"])
        for item in ordered_index(source / original["groups"]["sequence"]["index"])
    }
    graph_smiles = {
        item["key"]: item["canonical_isomeric_smiles"]
        for item in rows(source / "structure_source_smiles.tsv")
    }
    for rid, item in overlays.items():
        key, sequence = item["derived_canonical_identity_group"], item["derived_sequence"]
        if key not in graph_index or sequence not in sequence_index:
            raise ValueError("MHC10 identity needs a separate graph/sequence extraction")
        if graph_smiles.get(key) != item["derived_smiles"]:
            raise ValueError("MHC10 graph identity does not exactly match source SMILES")
        previous, compact = by_id[rid], derived[rid]
        if previous["structure_key"] not in {"", key} or previous["sequence_key"] not in {
            "",
            sequence,
        }:
            raise ValueError("MHC10 alias would replace a different verified representation")
        if (
            compact["sequence"] != sequence
            or compact["canonical_identity_group"] != key
            or compact["structure_smiles_sha256"] != item["derived_smiles_sha256"]
            or compact["sequence_eligible"] != "true"
            or compact["structure_eligible"] != "true"
        ):
            raise ValueError("MHC10 compact row disagrees with the source-proven identity")
        for group in original["groups"].values():
            position = (
                graph_index[key]
                if group["key_kind"] == "canonical_identity_group"
                else sequence_index[sequence]
            )
            valid = np.load(source / group["valid"], mmap_mode="r", allow_pickle=False)
            if not bool(valid[position]):
                raise ValueError("MHC10 alias points to a representation not valid for all methods")
        previous.update(structure_key=key, sequence_key=sequence)
    configuration = {
        "version": FEATURE_SCHEMA,
        "implementation_sha256": sha(__file__),
        "base_manifest_sha256": sha(pointer),
        "base_fingerprint": original["input_fingerprint"],
        "overlay_sha256": sha(overlay_path),
        "rows_index_sha256": sha(rows_index),
        "source_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "source_gates_sha256": sha(dataset_dir / "representation_gates.tsv"),
        "source_proof": proof,
    }
    fingerprint = signature(configuration)
    cache = output_run / "features" / fingerprint[:16]
    if (output_run / "features/manifest.json").exists() or cache.exists():
        raise FileExistsError("MHC10 followup cache already exists")
    cache.mkdir(parents=True)
    for name in original["artifact_hashes"]:
        if name != original["record_index"]:
            copy_verified(safe_path(source, name), safe_path(cache, name))
    write_tsv(
        cache / original["record_index"],
        list(aliases[0]),
        [by_id[item["record_id"]] for item in aliases],
    )
    overlay_rows = [overlays[rid] for rid in sorted(overlays)]
    write_tsv(cache / "mhc10_representation_overlay.tsv", list(overlay_rows[0]), overlay_rows)
    copy_verified(pointer, cache / "mhc10_source_feature_manifest.json")
    manifest = copy.deepcopy(original)
    manifest.update(
        schema=FEATURE_SCHEMA,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        cache_reused=False,
    )
    manifest["configuration"] = {
        "source_configuration": original["configuration"],
        "mhc10_overlay": configuration,
    }
    manifest["artifact_hashes"] = {name: sha(cache / name) for name in original["artifact_hashes"]}
    unchanged = set(original["artifact_hashes"]) - {original["record_index"]}
    if any(
        manifest["artifact_hashes"][name] != original["artifact_hashes"][name] for name in unchanged
    ):
        raise ValueError("MHC10 alias repair changed a feature or historical artifact")
    receipt = {
        "status": "completed",
        "input_fingerprint": fingerprint,
        "source_fingerprint": original["input_fingerprint"],
        "source_artifact_count": len(original["artifact_hashes"]),
        "source_structure_count": len(graph_index),
        "new_structure_count": len(graph_index),
        "derived_records": len(overlays),
        "reused_exact_graphs": len(overlays),
        "new_graph_rows": 0,
        "record_ids_unchanged": True,
        "sequence_universe_unchanged": len(sequence_index),
        "unchanged_artifacts_verified": len(unchanged),
        "all_method_masks_valid": True,
        "configuration": configuration,
    }
    write_json(cache / "mhc10_feature_receipt.json", receipt)
    for name in (
        "mhc10_representation_overlay.tsv",
        "mhc10_source_feature_manifest.json",
        "mhc10_feature_receipt.json",
    ):
        manifest["artifact_hashes"][name] = sha(cache / name)
    _verify_artifacts(original)
    if (
        sha(pointer) != configuration["base_manifest_sha256"]
        or sha(dataset_dir / "manifest.json") != configuration["source_manifest_sha256"]
        or sha(dataset_dir / "representation_gates.tsv") != configuration["source_gates_sha256"]
    ):
        raise ValueError("MHC10 alias repair changed frozen inputs")
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    write_json(output_run / "features/manifest.json", manifest)
    write_json(output_run / "mhc10_feature_receipt.json", receipt)
    write_json(output_run / "representation_methods.json", manifest["representation_methods"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=shared.ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output", type=Path, default=shared.ROOT / "data/dramp_mhc10_followup_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.dataset, args.output)["counts"]))


if __name__ == "__main__":
    main()
