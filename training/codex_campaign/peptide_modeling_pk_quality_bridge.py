"""Apply a bounded PK source-quality revision without changing frozen records.

Only the 25 observations in the pinned audit are in scope. Explicit assay
matrices are retained; control-normalized affinity is not promoted to PPB, and
the conflicting cyclosporine clearance observation is quarantined unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import peptide_modeling_tasks as baseline

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "bounded_pk_source_quality_revision_v1"
SOURCE_MANIFEST = "baf3f83f809a2038acb5e4c22239ee96b69826d664dd6aa44600716f5b87b418"
AUDIT_SHA256 = "f5b91950f219011513e519037bf883e410e34bf727c6d5951c41c03726c3857f"
DEFAULT_AUDIT = ROOT / "data/peptide_modeling_remaining_pk_audit_20261114_v1/audit.json"
CSA_ID = "NEW:07:40000000017805418"
PROTEIN_BINDING = re.compile(r"protein binding to human serum albumin", re.IGNORECASE)
AFFINITY = re.compile(
    r"binding affinity to human serum albumin at (?P<value>\d+(?:\.\d+)?) "
    r"ug/ml(?: after (?P<hours>\d+(?:\.\d+)?) hrs relative to control)?",
    re.IGNORECASE,
)
SOURCE_FIELDS = (
    "record_id",
    "origin",
    "input_file",
    "input_row",
    "source_record_id",
    "compound_id",
    "source_code",
    "source_entry_id",
    "endpoint",
    "task_kind",
    "raw_value",
    "raw_relation",
    "raw_unit",
    "value",
    "relation",
    "unit",
    "assay_id",
    "study_id",
    "reference_ids",
    "condition_json",
    "identity_key",
    "sequence",
    "sequence_kind",
    "smiles",
    "upstream_eligible",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_signature(raw):
    return hashlib.sha256(
        canonical({key: raw.get(key) or "" for key in SOURCE_FIELDS}).encode()
    ).hexdigest()


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def audit_rows(audit):
    rows = {row["record_id"]: row for row in audit["scheduled_matrix_review_candidates"]}
    csa = next(row for row in audit["quality_flags"] if row["record_id"] == CSA_ID)
    if len(rows) != 24 or CSA_ID in rows:
        raise ValueError("Pinned audit must contain 24 distinct matrix rows and one CsA row")
    return {CSA_ID: csa, **rows}


def decide(raw, evidence, csa_verification):
    """Derive a decision solely from matching source statements, never scores."""
    for key in (
        "record_id",
        "endpoint",
        "source_code",
        "source_entry_id",
        "assay_id",
        "study_id",
        "reference_ids",
        "input_file",
        "input_row",
        "value",
        "unit",
    ):
        if str(raw.get(key) or "") != str(evidence.get(key) or ""):
            raise ValueError(f"PK audit no longer matches frozen {key}")
    if raw.get("task_kind") != "regression" or raw.get("relation") != "=":
        raise ValueError("PK quality scope requires existing exact regression observations")
    sem, _, _ = baseline.contexts(raw)
    info = sem.get("assay", {})
    if not isinstance(info, dict) or normalize(baseline.assay_text(info)) != normalize(
        evidence["source_description"]
    ):
        raise ValueError("Frozen assay description differs from the audited source")
    description = normalize(info.get("description"))
    system = normalize(info.get("system_description"))
    base = {
        "record_id": raw["record_id"],
        "endpoint": raw["endpoint"],
        "source_record_sha256": record_signature(raw),
        "source_description": description,
        "source_system_description": system,
        "matrix": "",
        "assay_scope": "",
        "source_conditions": {},
        "original_task_id": evidence.get("scheduled_task", {}).get(
            "task_id",
            evidence.get("scheduled_intersections", {}).get("active32", {}).get("task_id", ""),
        ),
    }
    if raw["record_id"] == CSA_ID:
        if (
            csa_verification.get("activity_id") != 17805418
            or csa_verification.get("properties", {}).get("ROUTE") != "Oral"
            or raw["endpoint"] != "CL_per_kg"
            or raw["value"] != "57"
            or raw["unit"] != "mL/min/kg"
        ):
            raise ValueError("The exact oral-route conflict proof is missing")
        return {
            **base,
            "action": "quarantine",
            "reason": "source_route_conflict_CL_or_CLF_unresolved_no_numeric_correction",
        }
    flags = evidence["quality_flags"]
    if "explicit_assay_matrix_not_used_in_task_key" in flags:
        matrix = evidence.get("explicit_matrix")
        if raw["endpoint"] != "CL_per_kg" or matrix not in {"blood", "plasma"}:
            raise ValueError("Clearance matrix is outside the bounded source scope")
        if system != normalize(evidence.get("explicit_system_description")) or (
            len(system.split("|")) != 2 or system.split("|")[-1].strip().lower() != matrix
        ):
            raise ValueError("Clearance matrix is not explicit in its observation system")
        return {
            **base,
            "action": "matrix_override",
            "matrix": matrix,
            "reason": "explicit_observation_system_matrix",
            "assay_scope": "systemic_clearance",
        }
    if "assay_system_matrix_conflicts_with_narrative_fallback" in flags:
        if (
            raw["endpoint"] != "PPB"
            or raw["unit"] != "fraction_bound"
            or (
                system != normalize(evidence.get("explicit_system_description"))
                or len(system.split("|")) != 2
                or system.split("|")[-1].strip().lower() != "serum"
            )
        ):
            raise ValueError("Serum override is not explicit in the bounded source system")
        return {
            **base,
            "action": "matrix_override",
            "matrix": "serum",
            "reason": "explicit_system_serum_over_generic_plasma_narrative",
            "assay_scope": "protein_binding",
        }
    if "purified_albumin_vs_whole_serum_requires_source_review" in flags:
        if raw["endpoint"] != "PPB" or raw["unit"] != "fraction_bound":
            raise ValueError("Albumin source scope conflicts with frozen target")
        if PROTEIN_BINDING.fullmatch(description):
            return {
                **base,
                "action": "albumin_source_response",
                "matrix": "albumin",
                "assay_scope": "source_defined_albumin_binding_not_whole_serum",
                "reason": "explicit_human_serum_albumin_binding_target",
                "source_conditions": {
                    "concentration": "not_reported",
                    "exposure_h": "not_reported",
                },
            }
        match = AFFINITY.fullmatch(description)
        if match:
            return {
                **base,
                "action": "quarantine",
                "assay_scope": "affinity_response_not_proven_PPB_fraction",
                "reason": (
                    "source_affinity_or_control_normalized_signal_not_validated_fraction_bound"
                ),
                "source_conditions": {
                    "reported_concentration": match["value"],
                    "reported_unit": "ug/mL",
                    "concentration_role": "not_reported",
                    "exposure_h": match["hours"] or "not_reported",
                    "reference": "relative_to_control" if match["hours"] else "not_reported",
                },
            }
        return {
            **base,
            "action": "quarantine",
            "reason": "albumin_target_or_quantity_requires_source_review",
        }
    raise ValueError("Quality flag is outside the bounded correction policy")


def read_frozen_records(dataset, ids):
    path = (Path(dataset) / "catalogue.sqlite").resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in ids)
        return {
            row["record_id"]: dict(row)
            for row in connection.execute(
                f"SELECT * FROM records WHERE record_id IN ({placeholders})", sorted(ids)
            )
        }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_evidence(dataset, audit_path, output):
    dataset, audit_path, output = Path(dataset), Path(audit_path), Path(output)
    if digest(dataset / "manifest.json") != SOURCE_MANIFEST or digest(audit_path) != AUDIT_SHA256:
        raise ValueError("Dataset or source audit differs from the pinned bounded evidence")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = audit_rows(audit)
    records = read_frozen_records(dataset, candidates)
    if set(records) != set(candidates):
        raise ValueError("A quality audit observation is missing from the frozen catalogue")
    rows = [
        decide(records[rid], evidence, audit["csa_external_primary_verification"])
        for rid, evidence in sorted(candidates.items())
    ]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_path, output / "source_audit.json")
    path = output / "verified_mapping.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [key for key in rows[0] if key != "source_conditions"] + ["source_conditions_json"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: value for key, value in row.items() if key != "source_conditions"},
                    "source_conditions_json": canonical(row["source_conditions"]),
                }
            )
    manifest = {
        "schema": SCHEMA,
        "source_manifest_sha256": SOURCE_MANIFEST,
        "source_audit_sha256": AUDIT_SHA256,
        "verified_mapping_sha256": digest(path),
        "policy_source_sha256": digest(__file__),
        "counts": dict(Counter(row["action"] for row in rows)),
        "source_records": len(rows),
        "value_unit_structure_identity_unchanged": True,
        "scope": "source_quality_only_no_model_scores_or_holdout_used",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_evidence_map(path, records, source_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    audit_path = path.parent / "source_audit.json"
    if (
        manifest.get("schema") != SCHEMA
        or source_manifest_sha256 != SOURCE_MANIFEST
        or manifest.get("source_manifest_sha256") != SOURCE_MANIFEST
        or manifest.get("source_audit_sha256") != AUDIT_SHA256
        or digest(audit_path) != AUDIT_SHA256
        or digest(path) != manifest["verified_mapping_sha256"]
    ):
        raise ValueError("PK source-quality artifact or dataset hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = audit_rows(audit)
    originals = {row["record_id"]: row for row in records}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 25 or {row["record_id"] for row in rows} != set(expected):
        raise ValueError("PK quality whitelist must contain exactly its 25 unique observations")
    loaded = {}
    for row in rows:
        raw = originals.get(row["record_id"])
        if raw is None:
            raise ValueError("PK quality observation is outside the labelled frozen input")
        decision = decide(
            raw, expected[row["record_id"]], audit["csa_external_primary_verification"]
        )
        observed = {key: value for key, value in row.items() if key != "source_conditions_json"}
        observed["source_conditions"] = json.loads(row["source_conditions_json"])
        if observed != decision:
            raise ValueError("PK quality decision differs from its exact pinned source proof")
        loaded[row["record_id"]] = decision
    return loaded, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": digest(path),
        "source_audit_sha256": AUDIT_SHA256,
        "verified_records": len(loaded),
        "actions": dict(Counter(row["action"] for row in loaded.values())),
        "source_values_and_representations_unchanged": True,
    }


def enrich_quality_copy(raw, decision):
    """Return only a derived context copy; numeric values and features never change."""
    if record_signature(raw) != decision["source_record_sha256"]:
        raise ValueError("PK quality proof cannot replace a different source observation")
    result = dict(raw)
    context = json.loads(raw["condition_json"])
    context.setdefault("semantics", {})["verified_pk_quality"] = {
        "action": decision["action"],
        "matrix": decision["matrix"],
        "assay_scope": decision["assay_scope"],
        "reason": decision["reason"],
        "source_conditions": copy.deepcopy(decision["source_conditions"]),
    }
    result["condition_json"] = canonical(context)
    result["_derived_pk_quality_validated"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/pk_quality_revision_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_evidence(args.dataset, args.audit, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
