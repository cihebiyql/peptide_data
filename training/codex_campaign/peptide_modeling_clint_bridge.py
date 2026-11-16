"""Admit seven exact ACS Table 6 microsomal-clearance cells as derived evidence.

Frozen review rows and representation gates are never edited. The explicit
species, assay compartment and numeric relation are supported by the primary
article and the pinned machine-readable supporting CSV, not by the mCLint name.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "acs_exact_mouse_microsomal_clint_admission_v1"
DOI = "10.1021/acs.jmedchem.5c00544"
PRIMARY_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13626384/"
SOURCE_SHA256 = "f74711637f133163a3b3e045a12e8906753f361cfe489a90d7d45ae076be496c"
DEFAULT_SOURCE = (
    ROOT
    / "data/external_downloads/peptide_property_expansion_v13/cyclic_numeric"
    / "acs_jmedchem_5c00544/jm5c00544_si_002.csv"
)
UNIT = "uL/min/mg_protein"
VALUES = {"10": "202", "11": "43", "15": "160", "31": "182", "35": "10", "39": "33", "40": "323"}
RECORD_IDS = {
    "10": "OLD:PV13:21b16d20d306149c74047a6c",
    "11": "OLD:PV13:9f7471786860cefc089f2ccc",
    "15": "OLD:PV13:b64d32204089e229d8fdd569",
    "31": "OLD:PV13:90139874f88022c1231f44fa",
    "35": "OLD:PV13:9b023addf8f33d09f74611c9",
    "39": "OLD:PV13:ecf5169456044e90f8895af6",
    "40": "OLD:PV13:d7e496fb56aa8c0afef17ee6",
}
ASSAY_CONTRACT = {
    "source_doi": DOI,
    "source_table": "6",
    "species": "mouse",
    "matrix": "liver_microsomes",
    "route": "not_applicable_in_vitro",
    "quantity_unit": UNIT,
    "measurement_basis": "intrinsic_protein_normalized_clearance",
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def canonical_graph(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Source full SMILES is not parseable")
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
    return canonical, "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest()


def primary_support():
    return {
        "schema": "acs_clint_primary_support_v1",
        "doi": DOI,
        "primary_url": PRIMARY_URL,
        "retrieval_method": "official PMC full-text web retrieval",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "primary_locator": "Results stability subsection and Table 6; PMC lines 383-405",
        "short_species_quote": "in mouse plasma and microsomes",
        "species": "mouse",
        "matrix": "liver_microsomes",
        "raw_table_header": "mCLint (μL/min/mg)",
        "table_6_numeric_cells": VALUES,
        "interpretation": (
            "The surrounding results describe liver microsomal stability. These cells are "
            "in-vitro intrinsic clearance per microsomal protein mass; no administration "
            "route, bodyweight, or systemic clearance conversion is supplied or inferred."
        ),
    }


def validate_support(support):
    expected = {
        "schema": "acs_clint_primary_support_v1",
        "doi": DOI,
        "primary_url": PRIMARY_URL,
        "species": "mouse",
        "matrix": "liver_microsomes",
        "table_6_numeric_cells": VALUES,
        "short_species_quote": "in mouse plasma and microsomes",
    }
    if any(support.get(key) != value for key, value in expected.items()):
        raise ValueError("Primary species, compartment or exact Table 6 proof differs")


def pinned_source(path):
    path = Path(path)
    if sha_file(path) != SOURCE_SHA256:
        raise ValueError("Supporting CSV differs from the verified pinned SHA256")
    with path.open(encoding="cp1252", newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0][6] != "Microsomal Stability" or rows[1][7] != "mCLint (µL/min/mg)":
        raise ValueError("Supporting CSV microsomal column/header changed")
    found = {}
    for number, row in enumerate(rows[2:], start=3):
        if row[0] not in VALUES:
            continue
        if row[0] in found:
            raise ValueError("Supporting CSV has repeated compound identifiers")
        found[row[0]] = {"compound": row[0], "smiles": row[1], "cell": row[7], "row": number}
    if set(found) != set(VALUES):
        raise ValueError("Supporting CSV does not contain the seven exact compounds")
    return found


def verify_observation(raw, source, qc):
    """Validate original scalar/identity; a blank relation is not itself equality proof."""
    compound = source["compound"]
    if compound not in VALUES or raw.get("record_id") != RECORD_IDS[compound]:
        raise ValueError("Record is outside the bounded seven-source admission")
    required = {
        "source_code": "acs_jmedchem_5c00544",
        "source_entry_id": f"compound_{compound}:microsomal_clint",
        "endpoint": "CL_per_mg_protein",
        "task_kind": "regression",
        "lane": "review",
        "reason": "missing_relation",
        "relation": "",
        "raw_relation": "",
        "unit": UNIT,
        "upstream_eligible": "true",
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen source identity, review reason, supervision or unit differs")
    if raw.get("lower") or raw.get("upper") or raw.get("raw_lower") or raw.get("raw_upper"):
        raise ValueError("A range cannot be restored as an exact concentration")
    if not re.fullmatch(r"\d+(?:\.\d+)?", source["cell"]):
        raise ValueError("Source clearance cell is not a plain numeric exact measurement")
    try:
        value = Decimal(source["cell"])
        if (
            not value.is_finite()
            or value <= 0
            or any(
                Decimal(item) != value
                for item in (VALUES[compound], raw["value"], raw["raw_value"])
            )
        ):
            raise ValueError("Source Table 6 and original scalar do not exactly agree")
    except InvalidOperation as exc:
        raise ValueError("Invalid source scalar") from exc
    unit = raw.get("raw_unit", "").replace("µ", "u").replace("μ", "u").replace("_", " ")
    if unit != "uL/min/mg protein":
        raise ValueError("No bodyweight-normalized or different clearance unit is admitted")
    graph, key = canonical_graph(source["smiles"])
    frozen_graph, frozen_key = canonical_graph(raw.get("smiles", ""))
    if graph != frozen_graph or key != frozen_key:
        raise ValueError("Source and frozen complete isomeric graphs differ")
    if qc is None or any(
        qc.get(field) != expected
        for field, expected in (
            ("structure_status", "parsed"),
            ("status", "parsed"),
            ("key_comparison", "exact_match"),
            ("canonical_smiles", graph),
            ("canonical_identity_group", key),
        )
    ):
        raise ValueError("Frozen source structure QC does not verify the same complete graph")
    return {
        "record_id": raw["record_id"],
        "source_entry_id": raw["source_entry_id"],
        "compound": compound,
        "source_csv_row": str(source["row"]),
        "source_cell": source["cell"],
        "source_smiles": source["smiles"],
        "canonical_smiles": graph,
        "canonical_identity_group": key,
        "source_smiles_sha256": key.split(":", 1)[1],
        "frozen_record_sha256": signature(raw),
        "derived_relation": "=",
        "species": "mouse",
        "matrix": "liver_microsomes",
        "route": "not_applicable_in_vitro",
        "unit": UNIT,
    }


def query_raw_and_qc(connection, record_id):
    row = connection.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
    if row is None:
        raise ValueError("Verified source record is absent from the frozen catalogue")
    raw = dict(row)
    qc = connection.execute(
        "SELECT q.structure_status,q.key_comparison,v.status,v.canonical_smiles,"
        "v.canonical_identity_group "
        "FROM source_structure_qc q JOIN validated_source_structures v USING(smiles_sha256) "
        "WHERE q.origin=? AND q.source_key=?",
        (raw["origin"], raw["source_record_id"]),
    ).fetchone()
    return raw, dict(qc) if qc is not None else None


def connection_for(dataset_dir):
    connection = sqlite3.connect(
        (Path(dataset_dir) / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    return connection


def build_bridge(dataset_dir, output_dir, source_file=DEFAULT_SOURCE):
    dataset_dir, output_dir, source_file = Path(dataset_dir), Path(output_dir), Path(source_file)
    sources = pinned_source(source_file)
    support = primary_support()
    validate_support(support)
    evidence = []
    connection = connection_for(dataset_dir)
    try:
        for compound in VALUES:
            raw, qc = query_raw_and_qc(connection, RECORD_IDS[compound])
            evidence.append(verify_observation(raw, sources[compound], qc))
    finally:
        connection.close()
    if len({row["canonical_identity_group"] for row in evidence}) != 7:
        raise ValueError("The seven compounds do not define seven independent full graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "primary_support.json", support)
    write_tsv(output_dir / "verified_mapping.tsv", evidence)
    write_tsv(
        output_dir / "source_structure_smiles.tsv",
        [
            {
                "record_id": row["record_id"],
                "key": row["canonical_identity_group"],
                "canonical_smiles": row["canonical_smiles"],
                "sequence_eligible": "false",
            }
            for row in evidence
        ],
    )
    manifest = {
        "schema": SCHEMA,
        "source_manifest_sha256": sha_file(dataset_dir / "manifest.json"),
        "source_csv": str(source_file.resolve()),
        "source_csv_sha256": SOURCE_SHA256,
        "verified_mapping_sha256": sha_file(output_dir / "verified_mapping.tsv"),
        "primary_support_sha256": sha_file(output_dir / "primary_support.json"),
        "script_sha256": sha_file(__file__),
        "assay_contract": ASSAY_CONTRACT,
        "record_count": 7,
        "derived_only": True,
        "frozen_source_rows_edited": False,
        "sequence_policy": "No parent sequence or missing monomer is inferred from the graph",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def derived_copy(raw, proof):
    row = copy.deepcopy(raw)
    context = json.loads(row["condition_json"])
    historical = context.get("historical_condition", {})
    if isinstance(historical, str):
        historical = json.loads(historical)
    historical.update(species="mouse", matrix="liver_microsomes", route="not_applicable_in_vitro")
    context["historical_condition"] = json.dumps(historical, sort_keys=True)
    sem = context.setdefault("semantics", {})
    sem.update(species="mouse", matrix="liver_microsomes", route="not_applicable_in_vitro")
    sem["verified_clint_assay_contract"] = ASSAY_CONTRACT
    row.update(
        relation="=",
        lane="strict_numeric",
        normalization_status="derived_source_verified",
        normalization_reason="exact_source_table_restores_missing_relation",
        training_gate="derived_exact_source_clint_only",
        condition_json=json.dumps(context),
        sequence="",
        sequence_kind="unparsed_or_missing",
        sequence_group="",
        _derived_clint_proof_validated=True,
    )
    gate = {
        "record_id": row["record_id"],
        "lane": "derived_strict_numeric",
        "structure_model_eligible": "true",
        "sequence_model_eligible": "false",
        "sequence_kind": "unparsed_or_missing",
        "canonical_identity_group": proof["canonical_identity_group"],
        "structure_qc_status": "parsed",
        "final_training_gate": "derived_exact_source_clint_only",
    }
    return row, gate


def load_admissions(path, dataset_dir):
    if path is None:
        return [], {}, {"enabled": False}
    path, dataset_dir = Path(path), Path(dataset_dir)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("assay_contract") != ASSAY_CONTRACT:
        raise ValueError("Unsupported derived CLint assay contract")
    if manifest["source_manifest_sha256"] != sha_file(dataset_dir / "manifest.json"):
        raise ValueError("CLint evidence belongs to another frozen dataset")
    if manifest["verified_mapping_sha256"] != sha_file(path):
        raise ValueError("CLint evidence mapping hash differs")
    support_path = path.parent / "primary_support.json"
    if manifest["primary_support_sha256"] != sha_file(support_path):
        raise ValueError("CLint primary-support artifact hash differs")
    validate_support(json.loads(support_path.read_text(encoding="utf-8")))
    sources = pinned_source(manifest["source_csv"])
    items = list(read_tsv(path))
    if len(items) != 7 or {item["record_id"] for item in items} != set(RECORD_IDS.values()):
        raise ValueError("CLint evidence must cover exactly its seven source observations")
    records, gates = [], {}
    connection = connection_for(dataset_dir)
    try:
        for item in items:
            raw, qc = query_raw_and_qc(connection, item["record_id"])
            proof = verify_observation(raw, sources[item["compound"]], qc)
            if item != proof:
                raise ValueError("CLint evidence differs from the revalidated original observation")
            row, gate = derived_copy(raw, proof)
            records.append(row)
            gates[row["record_id"]] = gate
    finally:
        connection.close()
    return (
        records,
        gates,
        {
            "enabled": True,
            "mapping_path": str(path.resolve()),
            "mapping_sha256": sha_file(path),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "records_admitted_from_review": len(records),
            "frozen_source_rows_edited": False,
            "assay_contract": ASSAY_CONTRACT,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/clint_source_recovery_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.dataset_dir, args.output_dir)), flush=True)


if __name__ == "__main__":
    main()
