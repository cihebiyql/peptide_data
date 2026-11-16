"""Freeze the reviewed source-reported cell-injury benchmark independently.

This source bridge preserves categorical observations and full chemical graphs.
It performs no fitting and does not change the frozen catalogue or default tasks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "TASK_qualitative_cyto_source_weak_v1"
TIER = "reviewed_official_source_categorical_benchmark_not_primary_table_protocol_validated"
AUDIT_DIR = ROOT / "data/qualitative_cytotoxicity_feasibility_20261114_v1"
DATASET = ROOT / "data/dataset_reclean_20261113_v1"
UNIFIED = ROOT / "data/peptide_modeling_unified_features_20261114_v1/features/ee232f37b6acb5c3"
RECOVERY = ROOT / "data/qualitative_cyto_source_recovery_20261114_v1"
FOLLOWUP = ROOT / "data/peptide_modeling_qualitative_cyto_followup_v1"
PINS = {
    "audit.json": "574dcbaed44295f80c26048ca1de9f8c408c58c4bba1b46333716e15ba8ebbd3",
    "benchmark_admission_review.json": (
        "fca58c872cf0370143f52a4d6d9ca853b80fa171ad666529728be6ec168c2fa5"
    ),
    "frozen_lineage_crosschecks.json": (
        "3d8661030bcdabfeed0aa7e4dfccb8f4050a22de392e4b2a5bc3c75f600ae82a"
    ),
    "local_candidate_rows.json": "c7441daf0dbb24ff70a4f81723bc16e8a0e5dbd37eab14e9827d771be649057e",
    "official_api_snapshots.json": (
        "86b2ed1cb8a8cfc72a14fc7a607b68b428ee8ed863bb3ae973e2d8d985dea99b"
    ),
    "official_crosschecks.json": "c36f8b04c162688b2fd38ac13670981e2c826c9e604b9385448bb1ede539272b",
}
DATASET_MANIFEST_SHA = "baf3f83f809a2038acb5e4c22239ee96b69826d664dd6aa44600716f5b87b418"
UNIFIED_FP = "ee232f37b6acb5c3bfe1ed0ccc79cd7a6cf4330b2c1dc14464e3031a0fb9f823"
CONTEXT_FIELDS = ("cell", "species", "readout_family")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def source_label(text):
    """Map only the three source labels already admitted by independent review."""
    labels = {"Toxic": 1, "Not toxic": 0, "Non-toxic": 0}
    if text not in labels:
        raise ValueError(f"Not an admitted explicit source label: {text!r}")
    return labels[text]


def readout_family(description):
    """Prespecified method ontology; label text, doses and IDs are never inspected."""
    text = description.lower()
    if re.search(r"\b(mtt|mts|wst(?:-1)?|xtt|cck-?8)\b", text):
        return "metabolic_reduction"
    if any(term in text for term in ("resazurin", "alamar blue", "presto blue", "prestoblue")):
        return "metabolic_reduction"
    if any(term in text for term in ("lactate dehydrogenase", "ldh", "cytotox96")):
        return "membrane_integrity"
    if any(
        term in text for term in ("trypan blue", "propidium iodide", "calcein-am/pi", "annexin-v")
    ):
        return "membrane_integrity"
    if "differential interference contrast microscopy" in text:
        return "morphological_viability"
    if "celltiter-glo" in text:
        return "atp_viability"
    if "cell viability" in text:
        # The readout concept is explicit; its unreported method is not guessed.
        return "cell_viability_method_unreported"
    raise ValueError(f"Readout outside the reviewed ontology: {description!r}")


def component_groups(rows):
    """Join all shared molecules and all shared studies transitively."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in rows:
        graph = "G:" + row["graph_sha256"]
        study = "S:" + row["document_chembl_id"]
        if not row["document_chembl_id"]:
            raise ValueError("Missing source study identity")
        parent[find(graph)] = find(study)
    members = {}
    for row in rows:
        members.setdefault(find("G:" + row["graph_sha256"]), []).append(row)
    mapping, receipts = {}, []
    for component in members.values():
        nodes = sorted(
            {"G:" + r["graph_sha256"] for r in component}
            | {"S:" + r["document_chembl_id"] for r in component}
        )
        group = "COMP_" + digest_text("\n".join(nodes))[:20]
        ids = sorted(r["activity_id"] for r in component)
        mapping.update(dict.fromkeys(ids, group))
        receipts.append({"component_id": group, "activity_ids": ids, "nodes": nodes})
    return mapping, sorted(receipts, key=lambda item: item["component_id"])


def validate_split_memberships(rows, probes):
    """Independently validate every saved outer/inner partition before any fit."""
    by_id = {r["activity_id"]: r for r in rows}
    universe = set(by_id)
    if len(universe) != len(rows):
        raise ValueError("Duplicate source activity identity")

    def check(train, test):
        if not train or not test or train & test:
            raise ValueError("Empty or overlapping split")
        for part in (train, test):
            if {int(by_id[k]["label"]) for k in part} != {0, 1}:
                raise ValueError("Fold lacks one source class")
        for field in ("graph_sha256", "document_chembl_id"):
            if {by_id[k][field] for k in train} & {by_id[k][field] for k in test}:
                raise ValueError(f"Split leaks {field}")

    result = []
    for probe in probes:
        outer = probe["outer_folds"]
        if len(outer) != 5:
            raise ValueError("Expected exactly five outer folds")
        seen = []
        for fold in outer:
            test = set(fold["test_activity_ids"])
            if not test <= universe:
                raise ValueError("Split contains unknown activity")
            train = universe - test
            check(train, test)
            seen.extend(test)
            inner = fold["inner_folds"]
            if len(inner) != 5:
                raise ValueError("Expected exactly five inner folds")
            inner_seen = []
            for item in inner:
                validation = set(item["test_activity_ids"])
                if not validation <= train:
                    raise ValueError("Inner split escapes outer training set")
                check(train - validation, validation)
                inner_seen.extend(validation)
            if Counter(inner_seen) != Counter(dict.fromkeys(train, 1)):
                raise ValueError("Inner validation partition is incomplete or duplicated")
        if Counter(seen) != Counter(dict.fromkeys(universe, 1)):
            raise ValueError("Outer test partition is incomplete or duplicated")
        result.append(
            {
                "seed": probe["seed"],
                "outer_folds": 5,
                "inner_folds_per_outer": 5,
                "both_classes_all_splits": True,
                "graph_study_overlap_all_splits": 0,
            }
        )
    if {p["seed"] for p in probes} != {42, 2026, 3407} or len(probes) != 3:
        raise ValueError("Required three seeds differ")
    signatures = {
        tuple(tuple(sorted(f["test_activity_ids"])) for f in p["outer_folds"]) for p in probes
    }
    if len(signatures) != 3:
        raise ValueError("Seed partitions are identical")
    return result


def pinned_inputs(audit_dir):
    loaded = {}
    for name, expected in PINS.items():
        path = Path(audit_dir) / name
        if sha_file(path) != expected:
            raise ValueError(f"Reviewed evidence hash differs: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def verify_source(row, resources):
    """Check pinned official resource fields, full stereo graphs and label."""
    from rdkit import Chem

    activity = resources["activity"][row["activity_id"]]
    assay = resources["assay"][row["assay_chembl_id"]]
    molecule = resources["molecule"][row["chembl_id"]]
    exact = {
        "assay_chembl_id": row["assay_chembl_id"],
        "molecule_chembl_id": row["chembl_id"],
        "document_chembl_id": row["document_chembl_id"],
        "assay_description": row["assay_description"],
    }
    if any(activity.get(k) != v for k, v in exact.items()):
        raise ValueError("Official activity metadata differs")
    if (activity.get("standard_text_value") or "") != row["standard_text_value"]:
        raise ValueError("Official categorical source field differs")
    label_text = activity.get("standard_text_value") or activity.get("activity_comment")
    if (activity.get("activity_comment") or "") != row["activity_comment"]:
        raise ValueError("Official categorical comment field differs")
    if source_label(label_text) != int(row["label"]):
        raise ValueError("Official source label differs")
    for key in ("assay_cell_type", "assay_organism", "description"):
        expected = row["assay_description" if key == "description" else key]
        if assay.get(key) != expected:
            raise ValueError(f"Official assay {key} differs")
    if activity.get("data_validity_comment") or activity.get("potential_duplicate"):
        raise ValueError("Official source validity/duplicate flag")
    smiles = row["canonical_smiles"]
    if digest_text(smiles) != row["graph_sha256"]:
        raise ValueError("Source graph hash differs")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(
        label == "?" for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    ):
        raise ValueError("Invalid or unassigned tetrahedral source graph")
    canon = Chem.MolToSmiles(mol, isomericSmiles=True)
    for value in (
        activity["canonical_smiles"],
        molecule["molecule_structures"]["canonical_smiles"],
    ):
        other = Chem.MolFromSmiles(value)
        if other is None or Chem.MolToSmiles(other, isomericSmiles=True) != canon:
            raise ValueError("Official full isomeric graph differs")
    if Chem.MolToInchiKey(mol) != row["standard_inchi_key"]:
        raise ValueError("Full source graph InChIKey differs")
    if not row["assay_cell_type"] or not row["assay_organism"]:
        raise ValueError("Missing verified required biological context")
    return {
        "activity_id": row["activity_id"],
        "official_activity_verified": True,
        "official_assay_verified": True,
        "official_molecule_verified": True,
        "full_isomeric_graph_verified": True,
        "explicit_label_verified": True,
        "primary_article_tables_verified": False,
    }


def build_bridge(
    dataset_dir=DATASET,
    audit_dir=AUDIT_DIR,
    unified_dir=UNIFIED,
    recovery_dir=RECOVERY,
    followup_dir=FOLLOWUP,
):
    dataset_dir, audit_dir, unified_dir = map(Path, (dataset_dir, audit_dir, unified_dir))
    recovery_dir, followup_dir = map(Path, (recovery_dir, followup_dir))
    for output in (recovery_dir, followup_dir):
        for protected in (dataset_dir, audit_dir, unified_dir):
            if (
                output.resolve() == protected.resolve()
                or protected.resolve() in output.resolve().parents
            ):
                raise ValueError("Derived output overlaps protected source")
    if sha_file(dataset_dir / "manifest.json") != DATASET_MANIFEST_SHA:
        raise ValueError("Frozen dataset manifest differs")
    evidence = pinned_inputs(audit_dir)
    review, audit = evidence["benchmark_admission_review.json"], evidence["audit.json"]
    if review["evidence_grade"] != TIER:
        raise ValueError("Independent review evidence tier differs")
    rows = sorted(audit["candidate_records"], key=lambda row: row["activity_id"])
    if len(rows) != 150 or {r["activity_id"] for r in rows} != set(
        review["admitted_candidate_activity_ids"]
    ):
        raise ValueError("Exact reviewed 150-activity whitelist differs")
    if Counter(r["label"] for r in rows) != Counter({0: 131, 1: 19}):
        raise ValueError("Reviewed class counts differ")
    resources = {"activity": {}, "assay": {}, "molecule": {}}
    for snapshot in evidence["official_api_snapshots.json"]:
        resource = snapshot["resource"]
        key = "activity_id" if resource == "activity" else resource + "_chembl_id"
        plural = {"activity": "activities", "assay": "assays", "molecule": "molecules"}[resource]
        for item in snapshot["response"][plural]:
            identity = str(item[key])
            previous = resources[resource].get(identity)
            if previous is not None and previous != item:
                raise ValueError("Conflicting pinned official snapshots")
            resources[resource][identity] = item
    proofs = [verify_source(row, resources) for row in rows]
    lineage = {r["activity_id"]: r for r in evidence["frozen_lineage_crosschecks.json"]["records"]}
    record_ids = [lineage[r["activity_id"]]["record_id"] for r in rows]
    if len(set(record_ids)) != 150:
        raise ValueError("Original record identities are not one-to-one")
    connection = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" for _ in record_ids)
        raw = {
            r["record_id"]: dict(r)
            for r in connection.execute(
                f"SELECT * FROM records WHERE record_id IN ({placeholders})", record_ids
            )
        }
    finally:
        connection.close()
    groups, components = component_groups(rows)
    positive_groups = {groups[r["activity_id"]] for r in rows if r["label"] == 1}
    negative_groups = {groups[r["activity_id"]] for r in rows if r["label"] == 0}
    if (len(positive_groups), len(negative_groups)) != (9, 39):
        raise ValueError("Reviewed class-bearing component counts differ")
    expected_members = {
        frozenset(c["activity_ids"]) for c in review["validation_contract"]["component_receipts"]
    }
    if {frozenset(c["activity_ids"]) for c in components} != expected_members or len(
        components
    ) != 45:
        raise ValueError("Molecule-study connected components differ from independent review")
    split_checks = validate_split_memberships(rows, review["validation_contract"]["probes"])
    unified_records = {r["record_id"]: r for r in read_tsv(unified_dir / "record_bank_index.tsv")}
    unified_graphs = {r["key"]: r for r in read_tsv(unified_dir / "structure_source_smiles.tsv")}
    table, alias_rows, structures = [], [], {}
    for row in rows:
        frozen = raw[lineage[row["activity_id"]]["record_id"]]
        if (
            frozen["endpoint"] != "cytotoxicity"
            or frozen["lane"] not in {"review", "duplicate"}
            or frozen["smiles"] != row["canonical_smiles"]
            or int(frozen["value"]) != row["label"]
            or frozen["source_entry_id"]
            != "CHEMBL37BIN:"
            + row["activity_id"]
            + ":cell_line_cytotoxicity_ChEMBL_source_binary_candidate"
        ):
            raise ValueError("Frozen original observation differs from exact source whitelist")
        key = "CANON_SMILES:" + row["graph_sha256"]
        structures[key] = row["canonical_smiles"]
        if (
            key in unified_graphs
            and unified_graphs[key]["canonical_isomeric_smiles"] != row["canonical_smiles"]
        ):
            raise ValueError("Existing feature graph string differs")
        record = {
            "record_id": frozen["record_id"],
            "activity_id": row["activity_id"],
            "task_id": TASK_ID,
            "endpoint": "cytotoxicity",
            "task_kind": "classification",
            "y": row["label"],
            "group_id": groups[row["activity_id"]],
            "structure_key": key,
            "canonical_isomeric_smiles": row["canonical_smiles"],
            "cell": row["assay_cell_type"],
            "species": row["assay_organism"],
            "readout_family": readout_family(row["assay_description"]),
            "study_id": row["document_chembl_id"],
            "assay_id": row["assay_chembl_id"],
            "evidence_tier": TIER,
        }
        table.append(record)
        alias_rows.append(
            {
                "record_id": frozen["record_id"],
                "activity_id": row["activity_id"],
                "structure_key": key,
                "sequence_key": "",
                "exact_record_id_in_unified": frozen["record_id"] in unified_records,
                "graph_in_unified": key in unified_graphs,
                "original_frozen_lane": frozen["lane"],
            }
        )
    graph_rows = [
        {"key": key, "row_idx": index, "canonical_isomeric_smiles": structures[key]}
        for index, key in enumerate(sorted(structures))
    ]
    missing = [row for row in graph_rows if row["key"] not in unified_graphs]
    overlap = {
        "unified_fingerprint": UNIFIED_FP,
        "candidate_records": 150,
        "candidate_graphs": len(structures),
        "exact_record_id_overlap": sum(r["exact_record_id_in_unified"] for r in alias_rows),
        "records_with_existing_graph_vectors": sum(r["graph_in_unified"] for r in alias_rows),
        "existing_graphs": len(structures.keys() & unified_graphs.keys()),
        "new_graphs_required": len(missing),
        "meaning": (
            "Graph-vector reuse does not admit, duplicate, or alias any existing training label."
        ),
    }
    if (
        overlap["candidate_graphs"],
        overlap["records_with_existing_graph_vectors"],
        overlap["existing_graphs"],
        overlap["new_graphs_required"],
    ) != (100, 119, 77, 23):
        raise ValueError("Reviewed graph coverage changed")
    recovery_dir.mkdir(parents=True, exist_ok=True)
    (followup_dir / "task_tables").mkdir(parents=True, exist_ok=True)
    write_tsv(followup_dir / "task_tables" / (TASK_ID + ".tsv"), table)
    write_tsv(followup_dir / "source_structure_smiles.tsv", graph_rows)
    write_tsv(followup_dir / "missing_structure_feature_request.tsv", missing)
    write_tsv(followup_dir / "feature_rows_index.tsv", alias_rows)
    write_tsv(followup_dir / "record_index.tsv", alias_rows)
    write_json(
        followup_dir / "source_component_groups.json",
        {
            "components": components,
            "split_only_identifiers": [
                "study_id",
                "assay_id",
                "activity_id",
                "record_id",
                "group_id",
            ],
        },
    )
    write_json(
        followup_dir / "fivefold_prechecks.json",
        {
            "schema": "qualitative_cyto_nested_split_admission_v1",
            "verified": split_checks,
            "probes": review["validation_contract"]["probes"],
            "performance_claim": "split-only validation; no model fitted",
        },
    )
    context = {
        "allowed_predictor_columns": list(CONTEXT_FIELDS),
        "required_at_inference": list(CONTEXT_FIELDS),
        "categorical_vocabulary": {
            field: sorted({r[field] for r in table}) for field in CONTEXT_FIELDS
        },
        "encoding": (
            "Fit one-hot encoding inside each training fold only; report unseen contexts"
            " and abstain on missing/unsupported inference context."
        ),
        "forbidden_predictors": [
            "activity_id",
            "record_id",
            "assay_id",
            "study_id",
            "group_id",
            "DOI",
            "source_label_text",
            "free_text",
            "dose",
            "concentration",
            "relation",
            "duration",
        ],
        "readout_ontology": (
            "A priori method families: metabolic reduction; membrane integrity; "
            "morphological viability; ATP viability; explicitly reported viability with "
            "method unreported. No label words are used to select a family."
        ),
        "exposure_guard": (
            "All exposure/duration expressions remain source evidence only; four "
            "positive expressions may be outcome-associated thresholds. No exposure "
            "predictor is admitted."
        ),
        "ablations_required": ["prevalence", "context_only", "molecule_only", "joint"],
        "evaluation": (
            "Same outer/inner graph-study component folds for all ablations; molecule-"
            "equal-weight and component-aware uncertainty; calibration and unseen-"
            "context coverage."
        ),
    }
    write_json(followup_dir / "context_plan.json", context)
    manifest = {
        "schema": "qualitative_cyto_source_weak_task_v1",
        "task_id": TASK_ID,
        "raw_endpoint": "cytotoxicity",
        "raw_endpoint_count": 1,
        "task_kind": "classification",
        "evidence_tier": TIER,
        "records": 150,
        "positive_records": 19,
        "negative_records": 131,
        "full_stereochemical_graphs": 100,
        "source_studies": 55,
        "source_assays": 84,
        "molecule_study_components": 45,
        "positive_bearing_components": 9,
        "negative_bearing_components": 39,
        "target": review["task_contract"]["target"],
        "primary_article_tables_verified": False,
        "source_qualified_only": True,
        "default_frozen_tasks_modified": False,
        "table": "task_tables/" + TASK_ID + ".tsv",
        "label_column": "y",
        "group_column": "group_id",
        "split_protocol": "strict_nested_5_outer_x_5_inner_graph_study_components",
        "seeds": [42, 2026, 3407],
        "scope_limits": review["task_contract"]["scope_limits"],
        "feature_overlap": overlap,
        "training_performed": False,
    }
    write_json(followup_dir / "source_task_manifest.json", manifest)
    write_json(recovery_dir / "source_checks.json", {"records": proofs, "all_150_verified": True})
    write_json(
        recovery_dir / "source_evidence.json",
        {
            "candidate_records": rows,
            "source_raw_text_is_evidence_only": True,
            "task_contract": review["task_contract"],
        },
    )
    write_json(
        recovery_dir / "frozen_source_observations.json",
        {
            "read_only_frozen_records": [raw[key] for key in record_ids],
            "raw_lane_or_gate_edits": False,
            "source_whitelist_only": True,
        },
    )
    write_json(recovery_dir / "feature_overlap.json", overlap)
    receipt_paths = [audit_dir / name for name in PINS] + [
        dataset_dir / "manifest.json",
        unified_dir / "record_bank_index.tsv",
        unified_dir / "structure_source_smiles.tsv",
        unified_dir / "manifest.json",
    ]
    write_json(
        recovery_dir / "input_sha256.json",
        {str(p.relative_to(ROOT)): sha_file(p) for p in receipt_paths},
    )
    report = (
        "# Reviewed conditional cell-injury source bridge\n\n"
        "150 original observations (19 Toxic, 131 explicit source negatives), 100 "
        "full isomeric graphs, "
        "55 studies, 84 assays. Raw endpoint: cytotoxicity only. Original frozen "
        "review/duplicate lanes are unchanged.\n\n"
        "Official activity, assay, molecule, label and full-graph snapshots are "
        "pinned and rechecked. Primary article "
        "table values/protocols remain unverified. This weak source-report benchmark"
        " supports ADMET safety screening "
        "evaluation within its recorded source distribution; it does not establish "
        "universal or clinical toxicity.\n\n"
        "45 molecule-study components prevent both graph and study leakage. All 15 "
        "outer and 75 inner folds across "
        "42/2026/3407 contain both classes. These checks fit no model. All context "
        "encoders must be trained within folds.\n\n"
        "Actual exact record-ID overlap with the 213807-record unified bank: "
        f"{overlap['exact_record_id_overlap']}/150. "
        "119 observations reuse 77 graph vectors; 31 observations require vectors "
        "for 23 new graphs. Graph reuse does "
        "not imply label overlap or admission into existing training tasks.\n\n"
        "Allowed biological predictors are verified cell, species and readout "
        "family. Exposure, duration, IDs, free "
        "text and label wording remain outside X. Preserve the shared graph carrying"
        " opposite labels across different "
        "contexts. Require prevalence, context-only, molecule-only and joint "
        "ablations on identical folds.\n"
    )
    (recovery_dir / "REPORT.md").write_text(report, encoding="utf-8")
    outputs = [followup_dir / "task_tables" / (TASK_ID + ".tsv")] + [
        followup_dir / name
        for name in (
            "source_structure_smiles.tsv",
            "missing_structure_feature_request.tsv",
            "feature_rows_index.tsv",
            "record_index.tsv",
            "source_component_groups.json",
            "fivefold_prechecks.json",
            "context_plan.json",
            "source_task_manifest.json",
        )
    ]
    write_json(
        recovery_dir / "source_delivery_sha256.json",
        {str(p.relative_to(ROOT)): sha_file(p) for p in outputs},
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--unified-dir", type=Path, default=UNIFIED)
    parser.add_argument("--recovery-dir", type=Path, default=RECOVERY)
    parser.add_argument("--followup-dir", type=Path, default=FOLLOWUP)
    args = parser.parse_args()
    result = build_bridge(**vars(args))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("task_id", "records", "full_stereochemical_graphs", "feature_overlap")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
