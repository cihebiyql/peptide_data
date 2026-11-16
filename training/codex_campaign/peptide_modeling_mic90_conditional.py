"""Audit MIC90 conditional-model candidates without inventing assay conditions.

The bounded seven-record cohort has verified labels and graphs. Its reported
salt fields are incomplete, however, and cannot yet define an observed
experiment-state predictor. This tool emits a candidate ledger and an empty
training manifest. It never trains, imputes assay metadata or admits a task.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

import peptide_modeling_dbaasp_primary_bridge as bridge

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data/dbaasp_primary_recovery_20261114_v1"
BANK = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
SCHEMA = "mic90_conditional_candidate_audit_v1"
STUDY_PMID = "38401304"
STUDY_DOI = "10.1016/j.bbrc.2024.149700"
INCOMPLETE_CONDITION_REASON = "observed_experiment_conditions_not_proven"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_conditions(proofs, primary_payloads):
    """Separate metadata equality from proof that an assay field was fixed."""
    if len(proofs) != 7 or len({p["record_id"] for p in proofs}) != 7:
        raise ValueError("Expected the seven distinct bounded MIC90 candidates")
    patterns, missing, studies = Counter(), Counter(), set()
    records = []
    for proof in proofs:
        if (proof["endpoint"], proof["organism"], proof["medium"]) != bridge.MIC90_SCOPE:
            raise ValueError("MIC90 species, medium or endpoint changed")
        primary = primary_payloads[proof["peptide_id"]]
        activities = [a for a in primary["targetActivities"] if a["id"] == proof["activity_id"]]
        if len(activities) != 1:
            raise ValueError("Unique pinned primary activity is required")
        activity = activities[0]
        context = bridge.context_fields(activity)
        if context != proof["context_fields"]:
            raise ValueError("Source context conflicts with pinned proof")
        # Reference is a provenance locator only, never a predictor.
        try:
            article = primary["articles"][int(activity["reference"]) - 1]
            pmid = str(article["pubmed"]["pubmedId"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Cannot resolve primary activity study") from exc
        if pmid != STUDY_PMID:
            raise ValueError("Bounded source study changed")
        studies.add(pmid)
        salt = context.get("saltType", "")
        strength = context.get("ionicStrength", "")
        # Preserve literal units/strings. Blank means unreported, never zero.
        pattern = bridge.canonical({"saltType": salt, "ionicStrength": strength})
        patterns[pattern] += 1
        absent = [key for key in ("ph", "cfu", "saltType", "ionicStrength") if not context.get(key)]
        absent += [key for key in ("incubationTime", "temperature") if key not in context or not context[key]]
        missing.update(absent)
        records.append({
            **proof,
            "source_study_pmid": pmid,
            "salt_metadata_pattern": json.loads(pattern),
            "ionic_strength_unit": None,
            "unknown_or_unreported_conditions": absent,
            "condition_is_known_experiment_state": False,
            "training_admitted": False,
            "exclusion_reason": INCOMPLETE_CONDITION_REASON,
        })
    known_groups = len({p["canonical_identity_group"] for p in proofs})
    family_groups = len({p["split_group"] for p in proofs})
    return {
        "schema": SCHEMA,
        "status": "candidate_only",
        "candidate_records": len(records),
        "full_graph_groups": known_groups,
        "split_groups": family_groups,
        "fivefold_identity_group_count_sufficient": min(known_groups, family_groups) >= 5,
        "source_study_pmids": sorted(studies),
        "reported_salt_metadata_patterns": [
            {"reported_fields": json.loads(pattern), "records": count}
            for pattern, count in sorted(patterns.items())
        ],
        "missing_condition_field_counts": dict(sorted(missing.items())),
        "all_other_assay_conditions_proven_fixed": False,
        "known_source_salt_vocabulary_proven": False,
        "admitted_endpoint_count": 0,
        "admitted_tasks": 0,
        "admitted_records": 0,
        "reason": INCOMPLETE_CONDITION_REASON,
        "condition_encoding_created": False,
        "models_created": False,
        "records": records,
    }


def verified_candidates(evidence=EVIDENCE, bank=BANK):
    """Re-run exact source activity, value, unit, graph and family checks."""
    evidence, bank = Path(evidence), Path(bank)
    manifest = read_json(evidence / "manifest.json")
    checks = bridge.selected_checks(read_json(evidence / "census.json"))
    for filename, key in (
        ("census.json", "bounded_census_sha256"),
        ("conditional_mic90_evidence.json", "conditional_mic90_sha256"),
        ("source_compounds.json", "source_compounds_sha256"),
    ):
        if bridge.digest(evidence / filename) != manifest[key]:
            raise ValueError("Pinned source artifact hash changed: " + filename)
    proofs = read_json(evidence / "conditional_mic90_evidence.json")
    compounds = read_json(evidence / "source_compounds.json")
    pointer = bank / "features/manifest.json"
    feature_manifest = read_json(pointer)
    cache = Path(feature_manifest["cache_dir"])
    pins = feature_manifest["artifact_hashes"]
    # Read-only verification also proves this audit did not mutate a hardlink.
    for relative, expected in pins.items():
        if bridge.digest(cache / relative) != expected:
            raise ValueError("Unified feature artifact hash changed: " + relative)
    wanted = {p["record_id"] for p in proofs}
    feature_index = ROOT / "data/peptide_modeling_multiseed20h_followup1_v1/feature_rows_index.tsv"
    if bridge.digest(feature_index) != manifest["feature_index_sha256"]:
        raise ValueError("Source eligibility index changed")
    features = {r["record_id"]: r for r in bridge.read_tsv(feature_index) if r["record_id"] in wanted}
    unified = {r["record_id"]: r for r in bridge.read_tsv(cache / "record_bank_index.tsv") if r["record_id"] in wanted}
    dataset = ROOT / "data/dataset_reclean_20261113_v1"
    if bridge.digest(dataset / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Frozen dataset source manifest changed")
    primary_payloads = {}
    with sqlite3.connect((dataset / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        for proof in proofs:
            check = checks[proof["record_id"]]
            path = evidence / "primary_json" / (str(proof["peptide_id"]) + ".json")
            if bridge.digest(path) != proof["primary_json_sha256"]:
                raise ValueError("Pinned primary JSON changed")
            primary = read_json(path)
            raw = dict(con.execute("SELECT * FROM records WHERE record_id=?", (proof["record_id"],)).fetchone())
            actual = bridge.verify_observation(raw, primary, check, features[proof["record_id"]], compounds[raw["compound_id"]])
            if actual != proof:
                raise ValueError("Source label/full graph proof changed")
            if unified.get(proof["record_id"], {}).get("structure_key") != proof["canonical_identity_group"]:
                raise ValueError("Unified bank full graph alias differs")
            primary_payloads[proof["peptide_id"]] = primary
    methods = feature_manifest["representation_methods"]["methods"]
    structure_methods = [method for method in methods if method["source_kind"] == "structure" and method["independent"]]
    for method in structure_methods:
        for group_name in method["group_names"]:
            group = feature_manifest["groups"][group_name]
            indices = {r["key"]: int(r["row_idx"]) for r in bridge.read_tsv(cache / group["index"])}
            valid = np.load(cache / group["valid"], mmap_mode="r", allow_pickle=False)
            matrix = np.load(cache / group["matrix"], mmap_mode="r", allow_pickle=False)
            rows = [indices[p["canonical_identity_group"]] for p in proofs]
            values = matrix[rows]
            # RDKit may leave individual descriptors unavailable. Existing
            # train-fold imputation handles NaN; an invalid or empty row does
            # not become eligible through this audit.
            if not np.all(valid[rows]) or not np.isfinite(values).any(axis=1).all() or np.isinf(values).any():
                raise ValueError("Candidate has invalid existing full-graph features: " + group_name)
    if len(structure_methods) != 10:
        raise ValueError("Expected ten independent existing full-SMILES methods")
    return proofs, primary_payloads, {
        "unified_feature_fingerprint": feature_manifest["input_fingerprint"],
        "unified_feature_pointer_sha256": bridge.digest(pointer),
        "unified_runtime_artifact_hashes_verified_unchanged": len(pins),
        "source_conditional_evidence_sha256": bridge.digest(evidence / "conditional_mic90_evidence.json"),
        "exact_source_value_unit_full_graph_and_family_revalidated_records": len(proofs),
        "independent_structure_methods_valid_for_all_candidates": len(structure_methods),
    }


def build_audit(output, followup, evidence=EVIDENCE, bank=BANK):
    proofs, payloads, verification = verified_candidates(evidence, bank)
    audit = audit_conditions(proofs, payloads)
    audit["verification"] = verification
    audit["source_resolution"] = {
        "primary_article": "https://pubmed.ncbi.nlm.nih.gov/38401304/",
        "doi": STUDY_DOI,
        "publisher_url": "https://www.sciencedirect.com/science/article/pii/S0006291X24002365",
        "publisher_full_text_observation": "HTTP 403 on 2026-11-14; accessible abstract has no exact assay methods/table proof",
        "exact_condition_completion_evidence_available": False,
    }
    for directory in (Path(output), Path(followup)):
        directory.mkdir(parents=True, exist_ok=True)
    bridge.write_json(Path(output) / "audit.json", audit)
    bridge.write_json(Path(followup) / "manifest.json", {
        "schema": SCHEMA,
        "status": "candidate_only_not_trainable",
        "tasks": [],
        "candidate_endpoint": "MIC90",
        "reason": INCOMPLETE_CONDITION_REASON,
        "independent_holdout_created": False,
        "condition_covariates_created": False,
        "training_launched": False,
        "audit_sha256": bridge.digest(Path(output) / "audit.json"),
    })
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/mic90_conditional_recovery_20261114_v1")
    parser.add_argument("--followup", type=Path, default=ROOT / "data/peptide_modeling_mic90_conditional_followup_v1")
    args = parser.parse_args()
    audit = build_audit(args.output, args.followup)
    print(json.dumps({k: audit[k] for k in ("status", "candidate_records", "admitted_tasks", "reason", "verification")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
