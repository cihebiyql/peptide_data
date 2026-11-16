"""Recover RBC species only from exact pinned Hemolytik2 observation matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import build_hemolytik2_v13 as source_parser

SCHEMA = "hemolytik_exact_species_bridge_v1"
PINNED_COMMIT = source_parser.PINNED_COMMIT
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = source_parser.DEFAULT_RAW_DIR / "Hemolytik2_complete_data.csv"
DEFAULT_PIN = source_parser.DEFAULT_MANIFEST
RAW_MATCH_FIELDS = (
    "origin",
    "input_file",
    "input_row",
    "source_code",
    "source_entry_id",
    "endpoint",
    "source_record_id",
    "compound_id",
    "assay_id",
    "raw_value",
    "raw_relation",
    "raw_unit",
    "value",
    "unit",
    "sequence",
    "smiles",
    "condition_json",
    "task_kind",
    "relation",
    "has_modifications",
    "sequence_kind",
)


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def text_sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def read_tsv(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pinned_source(source_file, pin_file):
    source_file, pin_file = Path(source_file), Path(pin_file)
    pin = json.loads(pin_file.read_text(encoding="utf-8"))
    if pin["source"]["pinned_commit"] != PINNED_COMMIT:
        raise ValueError("Hemolytik source commit is not the approved pinned version")
    matches = [entry for entry in pin["inputs"] if Path(entry["path"]).name == source_file.name]
    if len(matches) != 1 or sha_file(source_file) != matches[0]["sha256"]:
        raise ValueError("Hemolytik source file differs from its pinned SHA256")
    with source_file.open(encoding="utf-8-sig", newline="") as handle:
        sources = {}
        for index, source in enumerate(csv.DictReader(handle), start=2):
            if source["id"] in sources:
                raise ValueError("Pinned Hemolytik source contains duplicate IDs")
            sources[source["id"]] = (index, source)
    return sources


def canonical_graph(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule is not None else None


def verify_observation(raw, source):
    """No scalar rounding, positional matching, graph simplification or identity projection."""
    if raw.get("source_code") != "HEMOLYTIK2":
        return None, "source_not_hemolytik2"
    if raw.get("task_kind") != "regression":
        return None, "source_metric_requires_regression_observation"
    if raw.get("source_entry_id") != "HEMOLYTIK2:" + source["id"]:
        return None, "source_entry_id_mismatch"
    match = source_parser.METRIC_PATTERN.match(source.get("activity", ""))
    if match is None:
        return None, "source_metric_not_unambiguously_parsed"
    if (
        len(
            re.findall(
                r"\b(?:HC10|HC50|LC50|EC25|EC50|IC50|LD50|HD50|HL50|MHC)\b",
                source["activity"],
                flags=re.IGNORECASE,
            )
        )
        != 1
    ):
        return None, "multiple_source_metric_mentions"
    parsed = match.groupdict()
    metric = parsed["metric"].upper()
    if raw.get("endpoint") != "hemolysis_" + metric:
        return None, "endpoint_metric_mismatch"
    try:
        value = Decimal(parsed["value"])
        exponent = parsed.get("exponent") or parsed.get("compact_exponent")
        if exponent:
            value *= Decimal(10) ** int(exponent)
        observed = Decimal(raw["raw_value"])
        if not value.is_finite() or not observed.is_finite() or value != observed:
            return None, "raw_scalar_mismatch"
    except (InvalidOperation, ValueError, KeyError):
        return None, "raw_scalar_invalid"
    relation = source_parser.normalize_relation(parsed["relation"])
    if relation != "=" or raw.get("raw_relation") != relation:
        return None, "raw_relation_not_same_exact_measurement"
    if source_parser.normalize_unit(raw.get("raw_unit", "")) != source_parser.normalize_unit(
        parsed["unit"]
    ):
        return None, "raw_unit_mismatch"
    if not raw.get("sequence") or raw["sequence"] != source.get("seq"):
        return None, "exact_sequence_mismatch"
    raw_smiles, source_smiles = raw.get("smiles", ""), source.get("smiles", "")
    if not raw_smiles or not source_smiles:
        return None, "missing_source_or_observation_smiles"
    source_graph, graph = canonical_graph(source_smiles), canonical_graph(raw_smiles)
    if source_graph is None or graph is None:
        return None, "invalid_source_or_observation_smiles"
    if graph != source_graph:
        return None, "different_full_canonical_isomeric_graph"
    # This is the RBC-source column. Peptide donor origin is never substituted.
    species = source.get("source", "").strip()
    if species.lower() not in {"human", "horse"}:
        return None, "rbc_source_not_one_explicit_verified_species"
    return {
        "metric": metric,
        "rbc_species": species,
        "canonical_isomeric_smiles_sha256": text_sha(graph),
    }, ""


def build_bridge(dataset_dir, output_dir, source_file=DEFAULT_SOURCE, pin_file=DEFAULT_PIN):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    source_file, pin_file = Path(source_file).resolve(), Path(pin_file).resolve()
    source_sha = sha_file(dataset_dir / "manifest.json")
    sources = pinned_source(source_file, pin_file)
    matches, audit = [], []
    lane = dataset_dir / "lanes/strict_numeric.tsv"
    for raw in read_tsv(lane):
        if raw.get("source_code") != "HEMOLYTIK2":
            continue
        entry_id = raw.get("source_entry_id", "").split(":")[-1]
        source_row, source = sources.get(entry_id, (None, None))
        proof, reason = verify_observation(raw, source) if source else (None, "source_id_missing")
        audit.append(
            {
                "record_id": raw["record_id"],
                "endpoint": raw["endpoint"],
                "source_csv_row": source_row,
                "status": "verified" if proof else "rejected",
                "reason": reason,
            }
        )
        if proof:
            mapping = {"record_id": raw["record_id"], "source_csv_row": str(source_row), **proof}
            mapping.update({field: raw.get(field, "") for field in RAW_MATCH_FIELDS})
            mapping["source_smiles_sha256"] = text_sha(source["smiles"])
            matches.append(mapping)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "verified_mapping.tsv"
    fields = [
        "record_id",
        "source_csv_row",
        "metric",
        "rbc_species",
        "canonical_isomeric_smiles_sha256",
        *RAW_MATCH_FIELDS,
        "source_smiles_sha256",
    ]
    write_tsv(path, fields, matches)
    write_tsv(
        output_dir / "audit.tsv",
        list(audit[0])
        if audit
        else ["record_id", "endpoint", "source_csv_row", "status", "reason"],
        audit,
    )
    manifest = {
        "schema": SCHEMA,
        "source_dataset": str(dataset_dir.resolve()),
        "source_manifest_sha256": source_sha,
        "source_label_lane_sha256": sha_file(lane),
        "source_csv": str(source_file),
        "source_csv_sha256": sha_file(source_file),
        "source_pin_manifest": str(pin_file),
        "source_pin_manifest_sha256": sha_file(pin_file),
        "pinned_commit": PINNED_COMMIT,
        "script_sha256": sha_file(__file__),
        "verified_mapping_sha256": sha_file(path),
        "audit_sha256": sha_file(output_dir / "audit.tsv"),
        "counts": {
            "observations": len(audit),
            "verified": len(matches),
            "rejected": len(audit) - len(matches),
        },
        "reasons": dict(Counter(row["reason"] or "verified" for row in audit)),
        "groups": dict(Counter(row["endpoint"] + "|" + row["rbc_species"] for row in matches)),
        "scientific_scope": "species_from_explicit_RBC_source_column_only;"
        "no_target_or_representation_mutation;no_exposure_duration_inference",
    }
    if sha_file(dataset_dir / "manifest.json") != source_sha:
        raise ValueError("Frozen dataset manifest changed during evidence extraction")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_evidence_map(path, records, dataset_manifest_sha256):
    """Revalidate the pin, original observation, and complete stereo graph at admission."""
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("pinned_commit") != PINNED_COMMIT:
        raise ValueError("Unsupported Hemolytik evidence schema or source pin")
    if manifest.get("source_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("Hemolytik evidence refers to a different frozen dataset manifest")
    for actual, expected in [
        (path, manifest["verified_mapping_sha256"]),
        (manifest["source_csv"], manifest["source_csv_sha256"]),
        (manifest["source_pin_manifest"], manifest["source_pin_manifest_sha256"]),
    ]:
        if sha_file(actual) != expected:
            raise ValueError("Hemolytik evidence source or mapping SHA256 mismatch")
    sources = pinned_source(manifest["source_csv"], manifest["source_pin_manifest"])
    originals = {row["record_id"]: row for row in records}
    mapped = {}
    for evidence in read_tsv(path):
        raw = originals.get(evidence["record_id"])
        if evidence["record_id"] in mapped:
            raise ValueError("Duplicate Hemolytik evidence record")
        if raw is None or any(evidence[field] != raw.get(field, "") for field in RAW_MATCH_FIELDS):
            raise ValueError("Hemolytik evidence differs from frozen observation fields")
        source_row, source = sources.get(raw["source_entry_id"].split(":")[-1], (None, None))
        proof, reason = verify_observation(raw, source) if source else (None, "source_id_missing")
        if not proof or any(evidence[field] != value for field, value in proof.items()):
            raise ValueError("Hemolytik evidence lacks exact source proof: " + reason)
        if evidence["source_csv_row"] != str(source_row) or evidence[
            "source_smiles_sha256"
        ] != text_sha(source["smiles"]):
            raise ValueError("Hemolytik evidence source row differs from pinned source")
        mapped[evidence["record_id"]] = evidence
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "source_manifest_sha256": dataset_manifest_sha256,
        "source_csv_sha256": manifest["source_csv_sha256"],
        "pinned_commit": PINNED_COMMIT,
        "verified_records": len(mapped),
        "applied_records": 0,
    }


def enrich_species_copy(raw, evidence):
    result = dict(raw)
    context = json.loads(raw["condition_json"])
    semantics = context.setdefault("semantics", {})
    semantics["species"] = evidence["rbc_species"]
    result["condition_json"] = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/hemolytik_condition_bridge_20261114_v1"
    )
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pin-file", type=Path, default=DEFAULT_PIN)
    args = parser.parse_args()
    receipt = build_bridge(args.dataset_dir, args.output_dir, args.source_file, args.pin_file)
    print(json.dumps(receipt["counts"]), flush=True)


if __name__ == "__main__":
    main()
