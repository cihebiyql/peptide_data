"""Admit one explicitly secondary-curated HC10 family with bounded chemistry proof.

This is independent of the original candidate-only primary-proof audit. Numeric
labels remain pinned secondary observations; no primary table cells are verified.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import peptide_modeling_hc10_terminal_bridge as terminal
import peptide_modeling_hemolytik_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "hemolytik_hc10_reviewed_secondary_family_bridge_v1"
TIER = "pinned_secondary_quantitative_labels_with_primary_supported_chemical_family"
SOURCE_IDS = frozenset(str(value) for value in range(2355, 2367))
PARENT = "GLPALISWIKRKRQQ"
REVIEW = ROOT / "data/hc10_terminal_recovery_20261114_v1/secondary_benchmark_review.json"
REVIEW_SHA256 = "627c37e1e5bc5c42a306507451205aea33aa944cc867e2ede5dc39fbff66f4f8"
SOURCE_SHA256 = "60ea939eab999a375eb0b132afdce3f5ba0e21c753e32b50296df9b265647c42"
PIN_SHA256 = "16b202ef95a0669614a505e164a8e107c7533644cd4ae2a3f3ebe9fd531b3f64"
RAW_FIELDS = (*shared.RAW_MATCH_FIELDS, "lane")


def stable(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def reviewed_scope(path=REVIEW):
    if shared.sha_file(path) != REVIEW_SHA256:
        raise ValueError("Independent secondary benchmark review hash differs")
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        review["recommended_data_tier"] != TIER
        or set(review["supported_source_ids"]) != {"HEMOLYTIK2:" + k for k in SOURCE_IDS}
        or review["counts"]["primary_table_individually_verified_rows"] != 0
    ):
        raise ValueError("Independent review scope differs")
    return review


def pinned_source(source_file, pin_file):
    if shared.sha_file(source_file) != SOURCE_SHA256 or shared.sha_file(pin_file) != PIN_SHA256:
        raise ValueError("Reviewed-secondary source/pin differs from fixed reviewed version")
    return shared.pinned_source(source_file, pin_file)


def verify_observation(raw, source):
    if not source or source.get("id") not in SOURCE_IDS:
        return None, "outside_reviewed_12_member_secondary_scope"
    sequence = source.get("seq", "")
    if len(sequence) != len(PARENT):
        return None, "outside_primary_supported_15mer_family"
    positions = [i + 1 for i, (a, b) in enumerate(zip(PARENT, sequence, strict=True)) if a != b]
    if len(positions) > 1 or any(position > 9 for position in positions):
        return None, "outside_parent_or_single_substitution_positions_1_to_9"
    proof, reason = terminal.verify_candidate(raw, source)
    if proof is None:
        return None, reason
    return {
        **proof,
        "admitted": True,
        "evidence_tier": TIER,
        "primary_proof_status": "primary_family_supported_secondary_quantitative_cells_only",
        "primary_table_values_verified": False,
        "source_export_mechanism_proven": False,
        "primary_family_supported": True,
        "parent_distance": len(positions),
        "changed_positions": positions,
        "response_metric": "HC10",
        "response_percent_lysis": 10,
        "independent_studies": 1,
        "independent_chemical_families": 1,
    }, ""


def build_bridge(
    dataset_dir,
    output_dir,
    source_file=shared.DEFAULT_SOURCE,
    pin_file=shared.DEFAULT_PIN,
    review_file=REVIEW,
):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    protected = ROOT / "data/hc10_terminal_recovery_20261114_v1"
    target = output_dir.resolve()
    if any(
        target == p.resolve() or p.resolve() in target.parents for p in (dataset_dir, protected)
    ):
        raise ValueError("Secondary output must be independent of frozen and candidate evidence")
    reviewed_scope(review_file)
    sources = pinned_source(source_file, pin_file)
    source_hash = shared.sha_file(dataset_dir / "manifest.json")
    gates_hash = shared.sha_file(dataset_dir / "representation_gates.tsv")
    con = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    con.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in con.execute(
                "SELECT * FROM records WHERE source_code='HEMOLYTIK2' "
                "AND endpoint='hemolysis_HC10' AND lane='strict_numeric'"
            )
        ]
    finally:
        con.close()
    admitted, audit = [], []
    for raw in rows:
        source_row, source = sources.get(raw["source_entry_id"].split(":")[-1], (None, None))
        proof, reason = verify_observation(raw, source)
        audit.append(
            {
                "record_id": raw["record_id"],
                "source_entry_id": raw["source_entry_id"],
                "status": "admitted_secondary" if proof else "rejected",
                "reason": reason,
            }
        )
        if proof:
            admitted.append(
                {
                    "record_id": raw["record_id"],
                    "source_csv_row": source_row,
                    **{field: raw.get(field, "") for field in RAW_FIELDS},
                    "proof_json": stable(proof),
                }
            )
    if len(admitted) != 12 or {r["source_entry_id"] for r in admitted} != {
        "HEMOLYTIK2:" + k for k in SOURCE_IDS
    }:
        raise ValueError("Reviewed secondary source membership must be exactly twelve")
    output_dir.mkdir(parents=True, exist_ok=True)
    shared.write_tsv(output_dir / "secondary_mapping.tsv", list(admitted[0]), admitted)
    shared.write_tsv(output_dir / "audit.tsv", list(audit[0]), audit)
    manifest = {
        "schema": SCHEMA,
        "evidence_tier": TIER,
        "source_manifest_sha256": source_hash,
        "source_representation_gates_sha256": gates_hash,
        "source_csv": str(Path(source_file).resolve()),
        "source_csv_sha256": shared.sha_file(source_file),
        "source_pin": str(Path(pin_file).resolve()),
        "source_pin_sha256": shared.sha_file(pin_file),
        "review_file": str(Path(review_file).resolve()),
        "review_sha256": REVIEW_SHA256,
        "pinned_commit": shared.PINNED_COMMIT,
        "secondary_mapping_sha256": shared.sha_file(output_dir / "secondary_mapping.tsv"),
        "audit_sha256": shared.sha_file(output_dir / "audit.tsv"),
        "script_sha256": shared.sha_file(__file__),
        "terminal_validator_sha256": shared.sha_file(terminal.__file__),
        "counts": {
            "observations": len(rows),
            "admitted_secondary": len(admitted),
            "primary_table_values_verified": 0,
            "rejected": len(rows) - len(admitted),
        },
        "primary_table_values_verified": False,
        "source_export_mechanism_proven": False,
        "primary_reference_url": "https://pubmed.ncbi.nlm.nih.gov/14596922/",
        "scientific_scope": (
            "one_study_one_chemical_family_within_series_exploratory_secondary_benchmark"
        ),
        "missing_assay_conditions_inferred": False,
        "original_candidate_only_contract_unchanged": True,
    }
    if (
        shared.sha_file(dataset_dir / "manifest.json") != source_hash
        or shared.sha_file(dataset_dir / "representation_gates.tsv") != gates_hash
    ):
        raise ValueError("Frozen input changed during read-only proof extraction")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_evidence_map(path, records, dataset_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("evidence_tier") != TIER
        or manifest.get("source_manifest_sha256") != dataset_manifest_sha256
        or manifest.get("primary_table_values_verified") is not False
        or manifest.get("source_export_mechanism_proven") is not False
        or manifest.get("pinned_commit") != shared.PINNED_COMMIT
    ):
        raise ValueError("Unsupported reviewed-secondary evidence scope or primary claim")
    reviewed_scope(manifest["review_file"])
    for actual, expected in (
        (path, manifest["secondary_mapping_sha256"]),
        (manifest["source_csv"], manifest["source_csv_sha256"]),
        (manifest["source_pin"], manifest["source_pin_sha256"]),
    ):
        if shared.sha_file(actual) != expected:
            raise ValueError("Reviewed-secondary mapping/source SHA256 differs")
    sources = pinned_source(manifest["source_csv"], manifest["source_pin"])
    originals = {row["record_id"]: row for row in records}
    mapped = {}
    for evidence in shared.read_tsv(path):
        raw = originals.get(evidence["record_id"])
        if evidence["record_id"] in mapped or raw is None:
            raise ValueError("Duplicate or missing reviewed-secondary record")
        if any(evidence[field] != str(raw.get(field, "")) for field in RAW_FIELDS):
            raise ValueError("Reviewed-secondary frozen observation differs")
        source_row, source = sources.get(raw["source_entry_id"].split(":")[-1], (None, None))
        proof, reason = verify_observation(raw, source)
        if (
            not proof
            or evidence["proof_json"] != stable(proof)
            or evidence["source_csv_row"] != str(source_row)
        ):
            raise ValueError("Reviewed-secondary source proof differs: " + reason)
        mapped[evidence["record_id"]] = proof
    if len(mapped) != 12 or {originals[k]["source_entry_id"] for k in mapped} != {
        "HEMOLYTIK2:" + k for k in SOURCE_IDS
    }:
        raise ValueError("Reviewed-secondary membership is not exactly twelve")
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": shared.sha_file(path),
        "evidence_tier": TIER,
        "verified_records": len(mapped),
        "applied_records": 0,
        "primary_table_values_verified": False,
        "independent_studies": 1,
        "independent_chemical_families": 1,
        "source_manifest_sha256": dataset_manifest_sha256,
    }


def enrich_species_copy(raw, evidence):
    if (
        evidence.get("evidence_tier") != TIER
        or evidence.get("primary_table_values_verified") is not False
        or not evidence.get("admitted")
    ):
        raise ValueError("HC10 enrichment requires reviewed-secondary proof")
    return shared.enrich_species_copy(raw, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/hc10_secondary_benchmark_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.dataset_dir, args.output_dir)["counts"]))


if __name__ == "__main__":
    main()
