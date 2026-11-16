"""Verify exact HUVEC/HDF IC50 source statements and primary-sequence chemistry.

This creates independent derived evidence. It never changes the frozen records
or representation gates. Cysteine-bearing sequences remain structure-ineligible
because the pinned source does not establish covalent/disulfide topology.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import peptide_modeling_dramp_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dramp_exact_huvec_hdf_ic50_bridge_v1"
STATEMENT = re.compile(
    r"^\s*(?P<cell>HUVEC|HDF|AHDF)\s*:\s*IC50\s*=\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:\u00b1\s*(?P<uncertainty>\d+(?:\.\d+)?))?"
    r"\s*[u\u00b5\u03bc]M\s*$",
    re.IGNORECASE,
)


def verify_observation(raw, source):
    if raw.get("source_code") != "dramp_v4" or raw.get("endpoint") != "cytotoxicity_IC50":
        return None, "source_or_endpoint_outside_scope"
    if raw.get("upstream_eligible") in {"false", "0"}:
        return None, "upstream_source_gate_rejected"
    if raw.get("task_kind") != "regression" or raw.get("source_entry_id") != source["DRAMP_ID"]:
        return None, "source_identity_or_task_kind_mismatch"
    annotation = str(source.get("Cytotoxicity", ""))
    clauses = annotation.strip().split(";")
    if not 1 <= len(clauses) <= 2:
        return None, "annotation_not_in_bounded_statement_grammar"
    matches = [STATEMENT.fullmatch(clause) for clause in clauses]
    if any(match is None for match in matches):
        return None, "annotation_not_in_bounded_statement_grammar"
    if raw.get("relation") != "=" or raw.get("raw_relation") not in {"", "="}:
        return None, "observation_not_exact"

    def normalize(unit):
        return str(unit).replace("\u00b5", "u").replace("\u03bc", "u")

    if normalize(raw.get("raw_unit")) != "uM" or raw.get("unit") != "uM":
        return None, "source_unit_mismatch"
    try:
        aligned = [
            match
            for match in matches
            if Decimal(match["value"]) == Decimal(raw["value"])
            and Decimal(match["value"]) == Decimal(raw["raw_value"])
        ]
    except (InvalidOperation, KeyError):
        return None, "invalid_original_scalar"
    if len(aligned) != 1:
        return None, "source_scalar_not_unique"
    matched = aligned[0]
    if matched["cell"].upper() not in {"HUVEC", "HDF"}:
        return None, "cell_outside_bounded_scope"
    context = json.loads(raw["condition_json"]).get("source_context", {})
    if isinstance(context, str):
        context = json.loads(context)
    if context.get("raw_annotation", "").strip() != annotation.strip():
        return None, "annotation_differs_from_original_source"
    if context.get("source_file") != "general_amps.xlsx":
        return None, "source_file_outside_bounded_scope"
    original = json.loads(raw["raw_json"])
    sequence = str(source.get("Sequence", ""))
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        return None, "source_sequence_not_complete_standard_residues"
    if original.get("sequence") != sequence or original.get("identity_key") != (
        "sequence:" + shared.text_sha(sequence)
    ):
        return None, "original_source_sequence_or_identity_mismatch"
    if any(source.get(field) != value for field, value in shared.CHEMISTRY_REQUIRED.items()):
        return None, "source_chemistry_not_in_bounded_scope"
    if any(context.get(field) != source.get(field) for field in shared.CHEMISTRY_REQUIRED):
        return None, "original_source_chemistry_mismatch"
    return {
        "cell_line": matched["cell"].upper(),
        "metric": "IC50",
        "source_sequence": sequence,
        "source_sequence_sha256": shared.text_sha(sequence),
        "source_annotation_sha256": shared.text_sha(annotation.strip()),
        "matched_statement": matched.group(0).strip(),
        "statement_count": str(len(matches)),
        "matching_scalar_count": "1",
        "reported_uncertainty": matched["uncertainty"] or "",
        "source_chemistry_json": json.dumps(
            {field: source.get(field) for field in shared.CHEMISTRY_FIELDS},
            sort_keys=True,
        ),
        "representation_scope": "sequence_only_covalent_topology_unresolved"
        if "C" in sequence
        else "complete_native_graph_and_sequence",
    }, ""


def derived_representation(proof):
    sequence = proof["source_sequence"]
    if "C" in sequence:
        return {
            "derived_sequence": sequence,
            "derived_smiles": "",
            "derived_canonical_identity_group": "",
            "derived_smiles_sha256": "",
            "sequence_eligible": "true",
            "structure_eligible": "false",
            "sequence_kind": "source_complete_sequence_topology_unresolved",
            "chem_modified": "true",
            "derivation": "exact_source_primary_sequence;no_inference_of_disulfide_topology",
            "rdkit_version": "not_used_for_unresolved_structure",
            "representation_gate_status": "candidate_not_published",
        }
    full = shared.derived_representation(proof)
    return {
        **full,
        "sequence_eligible": "true",
        "structure_eligible": "true",
        "sequence_kind": "bare_standard",
        "chem_modified": "false",
    }


def build_bridge(
    dataset_dir, output_dir, source_file=shared.DEFAULT_SOURCE, pin_file=shared.DEFAULT_PIN
):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if output_dir.exists() and (output_dir / "manifest.json").exists():
        raise FileExistsError("A frozen cytotoxicity proof artifact already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file, pin_file = Path(source_file).resolve(), Path(pin_file).resolve()
    sources = shared.pinned_source(source_file, pin_file)
    database = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    database.row_factory = sqlite3.Row
    mapped, overlays, audited = [], [], []
    try:
        for row in database.execute(
            "SELECT * FROM records WHERE endpoint='cytotoxicity_IC50' "
            "AND lane='strict_numeric' AND source_code='dramp_v4'"
        ):
            raw = dict(row)
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = verify_observation(raw, source) if source else (None, "source_missing")
            audited.append(
                {
                    "record_id": raw["record_id"],
                    "source_entry_id": raw["source_entry_id"],
                    "reason": reason or "verified",
                }
            )
            if proof is None:
                continue
            evidence = {field: raw.get(field, "") for field in shared.RAW_MATCH_FIELDS}
            evidence.update(
                proof,
                source_xlsx_row=str(number),
                catalog_raw_json_sha256=shared.text_sha(raw["raw_json"]),
            )
            mapped.append(evidence)
            overlays.append(
                {
                    "record_id": raw["record_id"],
                    "source_entry_id": raw["source_entry_id"],
                    **derived_representation(proof),
                }
            )
    finally:
        database.close()
    if not mapped:
        raise ValueError("No source-proven bounded IC50 observations were found")
    shared.write_tsv(output_dir / "verified_mapping.tsv", list(mapped[0]), mapped)
    shared.write_tsv(output_dir / "representation_overlay.tsv", list(overlays[0]), overlays)
    shared.write_tsv(output_dir / "audit.tsv", list(audited[0]), audited)
    manifest = {
        "schema": SCHEMA,
        "source_dataset": str(dataset_dir.resolve()),
        "source_catalogue": str((dataset_dir / "catalogue.sqlite").resolve()),
        "source_manifest_sha256": shared.sha_file(dataset_dir / "manifest.json"),
        "source_xlsx": str(source_file),
        "source_xlsx_sha256": shared.sha_file(source_file),
        "source_pin": str(pin_file),
        "source_pin_sha256": shared.sha_file(pin_file),
        "verified_mapping_sha256": shared.sha_file(output_dir / "verified_mapping.tsv"),
        "representation_overlay_sha256": shared.sha_file(output_dir / "representation_overlay.tsv"),
        "implementation_sha256": shared.sha_file(__file__),
        "counts": {
            "observations": len(audited),
            "verified": len(mapped),
            "cells": dict(Counter(row["cell_line"] for row in mapped)),
            "representation_scopes": dict(Counter(row["representation_scope"] for row in mapped)),
        },
        "policy": "derived_compact_only;raw_labels_and_source_gates_unchanged;"
        "Cys_sequence_only_no_inferred_graph",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_evidence_map(path, records, source_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or (
        manifest["source_manifest_sha256"] != source_manifest_sha256
    ):
        raise ValueError("Cytotoxicity proof schema or frozen dataset mismatch")
    for actual, expected in [
        (path, manifest["verified_mapping_sha256"]),
        (manifest["source_xlsx"], manifest["source_xlsx_sha256"]),
        (manifest["source_pin"], manifest["source_pin_sha256"]),
    ]:
        if shared.sha_file(actual) != expected:
            raise ValueError("Cytotoxicity evidence or source SHA mismatch")
    sources = shared.pinned_source(manifest["source_xlsx"], manifest["source_pin"])
    originals = {row["record_id"]: row for row in records}
    database = sqlite3.connect(
        Path(manifest["source_catalogue"]).resolve().as_uri() + "?mode=ro", uri=True
    )
    mapped = {}
    try:
        for evidence in shared.read_tsv(path):
            raw = originals.get(evidence["record_id"])
            if raw is None or evidence["record_id"] in mapped:
                raise ValueError("Cytotoxicity proof record missing or repeated")
            if any(evidence[field] != raw.get(field, "") for field in shared.RAW_MATCH_FIELDS):
                raise ValueError("Cytotoxicity proof differs from frozen observation")
            payload = database.execute(
                "SELECT raw_json FROM records WHERE record_id=?", (raw["record_id"],)
            ).fetchone()
            if (
                payload is None
                or shared.text_sha(payload[0]) != evidence["catalog_raw_json_sha256"]
            ):
                raise ValueError("Original source payload differs from frozen proof")
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = (
                verify_observation({**raw, "raw_json": payload[0]}, source)
                if source
                else (None, "source_missing")
            )
            if not proof or any(evidence.get(field) != value for field, value in proof.items()):
                raise ValueError("Cytotoxicity source proof invalid: " + reason)
            if evidence["source_xlsx_row"] != str(number):
                raise ValueError("Cytotoxicity source row mismatch")
            mapped[raw["record_id"]] = evidence
    finally:
        database.close()
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": shared.sha_file(path),
        "verified_records": len(mapped),
        "applied_records": 0,
        "source_gates_unchanged": True,
    }


def load_representation_overlay(path, records, source_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if shared.sha_file(path) != manifest["representation_overlay_sha256"]:
        raise ValueError("Cytotoxicity representation proof SHA mismatch")
    evidence, summary = load_evidence_map(
        path.parent / "verified_mapping.tsv", records, source_manifest_sha256
    )
    overlays = {}
    for candidate in shared.read_tsv(path):
        rid = candidate["record_id"]
        if rid not in evidence or rid in overlays:
            raise ValueError("Cytotoxicity derived identity lacks unique source proof")
        proof = derived_representation(evidence[rid])
        if any(candidate.get(key) != value for key, value in proof.items()):
            raise ValueError("Derived cytotoxicity representation contradicts source proof")
        if candidate["source_entry_id"] != evidence[rid]["source_entry_id"]:
            raise ValueError("Cytotoxicity representation source mismatch")
        overlays[rid] = candidate
    if set(overlays) != set(evidence):
        raise ValueError("Cytotoxicity proof does not cover every admitted observation")
    return overlays, {**summary, "overlay_sha256": shared.sha_file(path)}


def enrich_cell_copy(raw, evidence):
    return shared.enrich_cell_copy(raw, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/dramp_cytotoxicity_followup_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.dataset_dir, args.output_dir)["counts"]), flush=True)


if __name__ == "__main__":
    main()
