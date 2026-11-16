"""Audit exact DRAMP HDMEC observations without changing frozen representation gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    ROOT / "data/external_downloads/peptide_property_expansion_v13/sequence_databases/dramp_v4"
)
DEFAULT_SOURCE = SOURCE_ROOT / "general_amps.xlsx"
DEFAULT_PIN = SOURCE_ROOT / "figshare_article_27233508.json"
SCHEMA = "dramp_exact_hdmec_condition_bridge_v1"
RAW_MATCH_FIELDS = (
    "record_id",
    "origin",
    "input_file",
    "input_row",
    "source_code",
    "source_entry_id",
    "source_record_id",
    "endpoint",
    "task_kind",
    "raw_value",
    "raw_relation",
    "raw_unit",
    "value",
    "unit",
    "relation",
    "sequence",
    "sequence_kind",
    "smiles",
    "has_modifications",
    "condition_json",
    "upstream_eligible",
)
CHEMISTRY_FIELDS = (
    "N-terminal_Modification",
    "C-terminal_Modification",
    "Other_Modifications",
    "Linear/Cyclic/Branched",
    "Stereochemistry",
    "Structure",
    "Structure_Description",
    "Comments",
)
CHEMISTRY_REQUIRED = {
    "N-terminal_Modification": "Free",
    "C-terminal_Modification": "Free",
    "Other_Modifications": "None",
    "Linear/Cyclic/Branched": "Linear",
    "Stereochemistry": "L",
}
ANNOTATION = re.compile(
    r"^\s*HDMEC\s*:\s*EC50\s*=\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:\u00b1\s*\d+(?:\.\d+)?)?\s*[\u00b5\u03bcu]M\s*$",
    re.IGNORECASE,
)


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def text_sha(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def read_tsv(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def pinned_source(source_file, pin_file):
    """The existing Figshare v2 receipt pins both the XLSX bytes and source version."""
    import openpyxl

    source_file = Path(source_file)
    pin = json.loads(Path(pin_file).read_text(encoding="utf-8"))
    if pin.get("id") != 27233508 or pin.get("version") != 2:
        raise ValueError("DRAMP source is not the pinned Figshare article version")
    files = [entry for entry in pin["files"] if entry["name"] == source_file.name]
    if len(files) != 1:
        raise ValueError("DRAMP source file is absent or ambiguous in source pin")
    expected = files[0]
    if source_file.stat().st_size != expected["size"]:
        raise ValueError("DRAMP pinned source size mismatch")
    md5 = hashlib.md5(source_file.read_bytes(), usedforsecurity=False).hexdigest()
    if md5 != expected["computed_md5"] or md5 != expected["supplied_md5"]:
        raise ValueError("DRAMP pinned source MD5 mismatch")
    workbook = openpyxl.load_workbook(source_file, read_only=True, data_only=True)
    try:
        iterator = workbook.active.iter_rows(values_only=True)
        headers = next(iterator)
        sources = {}
        for number, values in enumerate(iterator, start=2):
            source = dict(zip(headers, values, strict=True))
            key = str(source["DRAMP_ID"])
            if key in sources:
                raise ValueError("Duplicate DRAMP source ID")
            sources[key] = (number, source)
        return sources
    finally:
        workbook.close()


def verify_observation(raw, source):
    """A single anchored metric/cell statement is matched to the original observation."""
    if raw.get("source_code") != "dramp_v4" or raw.get("endpoint") != "cytotoxicity_EC50":
        return None, "source_or_endpoint_not_in_bounded_scope"
    if raw.get("upstream_eligible") in {"false", "0"}:
        return None, "upstream_source_gate_rejected"
    if raw.get("task_kind") != "regression" or raw.get("source_entry_id") != source["DRAMP_ID"]:
        return None, "source_identity_or_task_kind_mismatch"
    annotation = str(source.get("Cytotoxicity", ""))
    match = ANNOTATION.fullmatch(annotation)
    if match is None:
        return None, "not_single_exact_hdmec_ec50_statement"
    if raw.get("raw_relation") not in {"", "="} or raw.get("relation") != "=":
        return None, "observation_not_exact"
    try:
        value = Decimal(match["value"])
        if value != Decimal(raw["value"]) or value != Decimal(raw["raw_value"]):
            return None, "source_scalar_mismatch"
    except (InvalidOperation, KeyError):
        return None, "invalid_source_scalar"

    def normalize_unit(unit):
        return str(unit).replace("\u00b5", "u").replace("\u03bc", "u")

    if normalize_unit(raw.get("raw_unit")) != "uM" or raw.get("unit") != "uM":
        return None, "source_unit_mismatch"
    context = json.loads(raw["condition_json"])
    original = json.loads(raw["raw_json"])
    source_context = context.get("source_context", {})
    if isinstance(source_context, str):
        source_context = json.loads(source_context)
    if source_context.get("raw_annotation", "").strip() != annotation.strip():
        return None, "annotation_not_same_original_source_text"
    sequence = str(source.get("Sequence", ""))
    if not sequence or original.get("sequence") != sequence:
        return None, "original_source_sequence_mismatch"
    if original.get("identity_key") != "sequence:" + text_sha(sequence):
        return None, "original_sequence_identity_hash_mismatch"
    if any(source.get(field) != value for field, value in CHEMISTRY_REQUIRED.items()):
        return None, "source_chemistry_not_fully_in_bounded_scope"
    if any(source_context.get(field) != source.get(field) for field in CHEMISTRY_REQUIRED):
        return None, "original_source_chemistry_mismatch"
    if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY") or "C" in sequence:
        return None, "ambiguous_sequence_or_possible_disulfide_outside_scope"
    return {
        "cell_line": "HDMEC",
        "metric": "EC50",
        "source_sequence": sequence,
        "source_sequence_sha256": text_sha(sequence),
        "source_annotation_sha256": text_sha(annotation.strip()),
        "source_chemistry_json": json.dumps(
            {field: source.get(field) for field in CHEMISTRY_FIELDS},
            sort_keys=True,
        ),
    }, ""


def derived_representation(proof):
    """Candidate only: no source-record mutation or gate publication happens here."""
    from rdkit import Chem, rdBase

    molecule = Chem.MolFromFASTA(proof["source_sequence"], flavor=0)
    if molecule is None:
        raise ValueError("Exact all-L free-terminal peptide did not parse")
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("Derived complete isomeric SMILES did not round-trip")
    return {
        "derived_sequence": proof["source_sequence"],
        "derived_smiles": smiles,
        "derived_canonical_identity_group": "CANON_SMILES:" + text_sha(smiles),
        "derived_smiles_sha256": text_sha(smiles),
        "rdkit_version": rdBase.rdkitVersion,
        "derivation": "MolFromFASTA(flavor=0);explicit_FreeN_FreeC_Linear_L_OtherNone_noCys",
        "representation_gate_status": "candidate_not_published",
    }


def build_bridge(dataset_dir, output_dir, source_file=DEFAULT_SOURCE, pin_file=DEFAULT_PIN):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    source_file, pin_file = Path(source_file).resolve(), Path(pin_file).resolve()
    sources = pinned_source(source_file, pin_file)
    source_hash = sha_file(dataset_dir / "manifest.json")
    database = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    database.row_factory = sqlite3.Row
    matches, overlays, audits = [], [], []
    try:
        for row in database.execute(
            "SELECT * FROM records WHERE source_code='dramp_v4' "
            "AND endpoint='cytotoxicity_EC50' AND lane='strict_numeric'"
        ):
            raw = dict(row)
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = (
                verify_observation(raw, source) if source else (None, "source_id_missing")
            )
            audits.append(
                {
                    "record_id": raw["record_id"],
                    "status": "verified" if proof else "rejected",
                    "reason": reason,
                    "source_xlsx_row": number,
                }
            )
            if proof:
                evidence = {field: raw.get(field, "") for field in RAW_MATCH_FIELDS}
                evidence["catalog_raw_json_sha256"] = text_sha(raw["raw_json"])
                evidence.update(source_xlsx_row=str(number), **proof)
                matches.append(evidence)
                overlays.append(
                    {
                        "record_id": raw["record_id"],
                        "source_entry_id": raw["source_entry_id"],
                        **derived_representation(proof),
                    }
                )
    finally:
        database.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = output_dir / "verified_mapping.tsv"
    write_tsv(mapping, list(matches[0]) if matches else ["record_id"], matches)
    overlay_fields = list(overlays[0]) if overlays else ["record_id", "source_entry_id"]
    write_tsv(output_dir / "representation_candidates.tsv", overlay_fields, overlays)
    (output_dir / "representation_overlay.tsv").write_bytes(
        (output_dir / "representation_candidates.tsv").read_bytes()
    )
    write_tsv(
        output_dir / "audit.tsv",
        list(audits[0]) if audits else ["record_id", "status", "reason", "source_xlsx_row"],
        audits,
    )
    manifest = {
        "schema": SCHEMA,
        "source_dataset": str(dataset_dir.resolve()),
        "source_catalogue": str((dataset_dir / "catalogue.sqlite").resolve()),
        "source_manifest_sha256": source_hash,
        "source_xlsx": str(source_file),
        "source_xlsx_sha256": sha_file(source_file),
        "source_pin": str(pin_file),
        "source_pin_sha256": sha_file(pin_file),
        "verified_mapping_sha256": sha_file(mapping),
        "representation_candidates_sha256": sha_file(output_dir / "representation_candidates.tsv"),
        "script_sha256": sha_file(__file__),
        "counts": {
            "observations": len(audits),
            "verified": len(matches),
            "rejected": len(audits) - len(matches),
        },
        "reasons": dict(Counter(row["reason"] or "verified" for row in audits)),
        "policy": (
            "condition_copy_only;frozen_representation_gates_unchanged;"
            "derived_graphs_candidates_only"
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_evidence_map(path, records, dataset_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest["source_manifest_sha256"] != dataset_manifest_sha256
    ):
        raise ValueError("DRAMP evidence schema or frozen dataset mismatch")
    for actual, expected in [
        (path, manifest["verified_mapping_sha256"]),
        (manifest["source_xlsx"], manifest["source_xlsx_sha256"]),
        (manifest["source_pin"], manifest["source_pin_sha256"]),
    ]:
        if sha_file(actual) != expected:
            raise ValueError("DRAMP evidence source or mapping SHA mismatch")
    sources = pinned_source(manifest["source_xlsx"], manifest["source_pin"])
    originals, mapped = {raw["record_id"]: raw for raw in records}, {}
    database = sqlite3.connect(
        Path(manifest["source_catalogue"]).resolve().as_uri() + "?mode=ro", uri=True
    )
    try:
        for evidence in read_tsv(path):
            raw = originals.get(evidence["record_id"])
            if raw is None or evidence["record_id"] in mapped:
                raise ValueError("DRAMP evidence observation missing or repeated")
            if any(evidence[field] != raw.get(field, "") for field in RAW_MATCH_FIELDS):
                raise ValueError("DRAMP evidence differs from frozen original observation")
            catalog = database.execute(
                "SELECT raw_json FROM records WHERE record_id=?", (raw["record_id"],)
            ).fetchone()
            if catalog is None or text_sha(catalog[0]) != evidence["catalog_raw_json_sha256"]:
                raise ValueError(
                    "DRAMP original catalogue source payload differs from frozen proof"
                )
            verified_raw = {**raw, "raw_json": catalog[0]}
            number, source = sources.get(raw["source_entry_id"], (None, None))
            proof, reason = (
                verify_observation(verified_raw, source) if source else (None, "source_id_missing")
            )
            if not proof or any(evidence[field] != value for field, value in proof.items()):
                raise ValueError("DRAMP evidence lacks exact source proof: " + reason)
            if evidence["source_xlsx_row"] != str(number):
                raise ValueError("DRAMP evidence source row mismatch")
            mapped[evidence["record_id"]] = evidence
    finally:
        database.close()
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "verified_records": len(mapped),
        "applied_records": 0,
        "representation_gates_unchanged": True,
    }


def enrich_cell_copy(raw, evidence):
    result = dict(raw)
    context = json.loads(raw["condition_json"])
    parameters = context.setdefault("semantics", {}).setdefault("reported_parameters", {})
    parameters["cell_line"] = [[evidence["cell_line"], ""]]
    result["condition_json"] = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return result


def load_representation_overlay(path, records, dataset_manifest_sha256):
    """Recheck complete source chemistry before enabling a separate derived representation."""
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha_file(path) != manifest["representation_candidates_sha256"]:
        raise ValueError("DRAMP representation candidate SHA mismatch")
    evidence, summary = load_evidence_map(
        path.parent / "verified_mapping.tsv", records, dataset_manifest_sha256
    )
    overlays = {}
    for candidate in read_tsv(path):
        rid = candidate["record_id"]
        if rid not in evidence or rid in overlays:
            raise ValueError(
                "DRAMP representation candidate lacks a unique exact observation proof"
            )
        proof = derived_representation(evidence[rid])
        if any(candidate.get(field) != value for field, value in proof.items()):
            raise ValueError("DRAMP derived complete structure differs from source chemistry proof")
        if candidate["source_entry_id"] != evidence[rid]["source_entry_id"]:
            raise ValueError("DRAMP representation source entry mismatch")
        overlays[rid] = candidate
    if set(overlays) != set(evidence):
        raise ValueError("DRAMP representation proof does not cover all exact observations")
    return overlays, {
        **summary,
        "overlay_path": str(path.resolve()),
        "overlay_sha256": sha_file(path),
        "policy": "derived_compact_representation_only;frozen_source_gates_unchanged",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/dramp_condition_bridge_20261114_v1"
    )
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pin-file", type=Path, default=DEFAULT_PIN)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bridge(args.dataset_dir, args.output_dir, args.source_file, args.pin_file)[
                "counts"
            ]
        )
    )


if __name__ == "__main__":
    main()
