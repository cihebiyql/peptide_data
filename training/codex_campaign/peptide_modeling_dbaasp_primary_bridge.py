"""Recover bounded DBAASP activity contexts from exact primary JSON objects.

This is a second source-proof route, not a relaxation of the flattened Tier A
bridge. Labels, representations and original family assignments never change.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "dbaasp_exact_primary_activity_bridge_v1"
CENSUS_SHA256 = "f85c42117b9763a439faa08d1b49e43b9938a60edfaae6a202d67abc27d0ac63"
DEFAULT_CENSUS = ROOT / "data/peptide_modeling_remaining_review_census_20261114_v1/census.json"
DEFAULT_DATABASE = (
    ROOT
    / "data/external_downloads/peptide_property_expansion_v13/dbaasp_live_v4_0_1"
    / "dbaasp_hemolysis_cytotoxicity_records.sqlite"
)
SCOPES = {
    ("MFC", "Saccharomyces cerevisiae", "RPMI-1640"),
    ("MFC", "Candida albicans ATCC 10231", "RPMI-1640"),
    ("LD50", "Escherichia coli HB101", "MHA"),
}
MIC90_SCOPE = ("MIC90", "Candida parapsilosis KCTC 7214", "YEPD")
RAW_FIELDS = (
    "record_id source_code source_entry_id source_record_id compound_id assay_id endpoint "
    "task_kind lane relation raw_relation raw_value raw_unit value unit smiles condition_json"
).split()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def raw_signature(raw):
    return signature({key: raw.get(key, "") for key in RAW_FIELDS})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unit(value):
    return str(value).strip().replace("\u00b5", "u").replace("\u03bc", "u").lower()


def exact_scalar(value):
    match = re.fullmatch(r"\s*(=)?\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", str(value))
    if not match:
        raise ValueError("Primary concentration is not an exact positive scalar")
    try:
        number = Decimal(match[2])
    except InvalidOperation as exc:
        raise ValueError("Invalid scalar") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("Nonpositive or nonfinite scalar")
    return number


def graph(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid complete primary SMILES")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def selected_checks(census):
    return {
        r["record_id"]: r
        for r in census["dbaasp_primary_record_checks"]
        if (r["endpoint"], r["organism"], r["medium"]) in SCOPES | {MIC90_SCOPE}
    }


def context_fields(activity):
    # Every source assay field is retained; only row identity, response and
    # reference locators are excluded from the scientific task signature.
    excluded = {
        "id",
        "peptideId",
        "activityMeasureGroup",
        "activityMeasureValue",
        "concentration",
        "activity",
        "unit",
        "reference",
    }
    return {key: copy.deepcopy(value) for key, value in activity.items() if key not in excluded}


def verify_observation(raw, primary, check, feature, source_compound=None):
    if raw.get("record_id") != check["record_id"]:
        raise ValueError("Frozen ID differs from bounded census identity")
    if (
        raw.get("source_code") != "DBAASP"
        or raw.get("source_entry_id") != f"DBAASP:{check['peptide_id']}"
        or raw.get("endpoint") != check["endpoint"]
        or raw.get("endpoint") not in {"MFC", "LD50", "MIC90"}
        or raw.get("task_kind") != "regression"
        or raw.get("lane") != "strict_numeric"
        or raw.get("relation") != "="
        or raw.get("raw_relation") != "="
        or any(raw.get(k) for k in ("lower", "upper", "raw_lower", "raw_upper"))
        or raw.get("unit") != "uM"
    ):
        raise ValueError("Source endpoint, lane, exact relation or concentration unit differs")
    if primary.get("id") != check["peptide_id"]:
        raise ValueError("Primary peptide identity differs")
    original = exact_scalar(raw["raw_value"])
    matches = []
    for activity in primary.get("targetActivities", []):
        try:
            scalar = exact_scalar(activity.get("concentration", ""))
        except ValueError:
            continue
        if (
            activity.get("peptideId") == primary["id"]
            and activity.get("activityMeasureGroup", {}).get("name") == raw["endpoint"]
            and activity.get("activityMeasureValue") == raw["endpoint"]
            and scalar == original
            and unit(activity.get("unit", {}).get("name", "")) == unit(raw["raw_unit"])
            and activity.get("targetSpecies", {}).get("name") == check["organism"]
            and activity.get("medium", {}).get("name") == check["medium"]
        ):
            matches.append(activity)
    if len(matches) != 1 or matches[0].get("id") != check["activity_id"]:
        raise ValueError("Primary activity must match endpoint/value/unit/species/medium uniquely")
    activity = matches[0]
    expected_context = check["primary_activity_context"]
    for key, value in expected_context.items():
        if activity.get(key) != value:
            raise ValueError("Pinned activity context differs")
    integrated = graph(raw["smiles"])
    primary_graphs = {
        graph(item["smiles"]) for item in primary.get("smiles", []) if item.get("smiles")
    }
    if integrated not in primary_graphs:
        raise ValueError("Full stereochemical graph is not present in primary JSON")
    graph_key = "CANON_SMILES:" + hashlib.sha256(integrated.encode()).hexdigest()
    if (
        feature.get("record_id") != raw["record_id"]
        or feature.get("structure_eligible") != "true"
        or feature.get("canonical_identity_group") != graph_key
        or feature.get("split_group") != check["split_group"]
        or not check.get("structure_gate_eligible")
        or not check.get("full_stereo_graph_exact_primary_match")
    ):
        raise ValueError(
            "Existing full-graph representation gate or family identity is not verified"
        )
    raw_unit = unit(raw["raw_unit"])
    if raw_unit == "um":
        normalized = format(float(original), ".15g")
        factor, mw = 1.0, None
    elif raw_unit == "ug/ml":
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        if (
            source_compound is None
            or source_compound.get("compound_id") != raw["compound_id"]
            or source_compound.get("molecular_weight_unit") not in {"g/mol", "g mol-1", "Da"}
            or graph(source_compound.get("smiles", "")) != integrated
        ):
            raise ValueError("Pinned original compound mass/unit/full graph is absent")
        mw = float(exact_scalar(source_compound["molecular_weight"]))
        # The catalogue normalizer used the provided g/mol scalar. Retain it
        # exactly; RDKit summation order can change its last machine digits.
        computed = Descriptors.MolWt(Chem.MolFromSmiles(raw["smiles"]))
        if format(computed, ".3f") != format(mw, ".3f"):
            raise ValueError(
                "Source molecular mass disagrees with the full graph at source precision"
            )
        factor = 1000.0 / mw
        # Reproduce the frozen normalizer's exact multiplication/formatting.
        # This is not a floating tolerance or nearest-activity lookup.
        normalized = format(float(original) * factor, ".15g")
    else:
        raise ValueError("Unsupported concentration conversion")
    if raw["value"] != normalized:
        raise ValueError("Frozen canonical concentration is not the exact source normalization")
    return {
        "record_id": raw["record_id"],
        "endpoint": raw["endpoint"],
        "source_record_sha256": raw_signature(raw),
        "primary_json_sha256": check["primary_json_sha256"],
        "peptide_id": primary["id"],
        "activity_id": activity["id"],
        "organism": check["organism"],
        "medium": check["medium"],
        "original_value": str(original),
        "original_unit": raw["raw_unit"],
        "canonical_value": raw["value"],
        "canonical_unit": raw["unit"],
        "normalization_factor": factor,
        "full_graph_molecular_weight": mw,
        "canonical_identity_group": graph_key,
        "split_group": feature["split_group"],
        "context_fields": context_fields(activity),
        "primary_articles": copy.deepcopy(primary.get("articles", [])),
        "training_admitted": (raw["endpoint"], check["organism"], check["medium"]) in SCOPES,
        "semantic_scope": "microbial_lethal_concentration_not_mammalian_systemic_ld50"
        if raw["endpoint"] == "LD50"
        else "organism_conditioned_antimicrobial_concentration",
    }


def enrich_condition_copy(raw, evidence):
    if raw_signature(raw) != evidence["source_record_sha256"] or not evidence["training_admitted"]:
        raise ValueError("Primary source proof cannot enrich a different observation")
    result = dict(raw)
    context = json.loads(raw["condition_json"])
    sem = context.setdefault("semantics", {})
    params = sem.setdefault("reported_parameters", {})
    params["target_species"] = [[evidence["organism"], ""]]
    params["medium"] = [[evidence["medium"], ""]]
    sem["verified_dbaasp_primary_activity"] = {
        "organism": evidence["organism"].lower(),
        "medium": evidence["medium"].lower(),
        "reported_condition_details": copy.deepcopy(evidence["context_fields"]),
        "semantic_scope": evidence["semantic_scope"],
    }
    result["condition_json"] = canonical(context)
    result["_derived_dbaasp_primary_validated"] = True
    return result


def build_evidence(
    dataset, feature_index, output, census_path=DEFAULT_CENSUS, database=DEFAULT_DATABASE
):
    dataset, output, census_path = Path(dataset), Path(output), Path(census_path)
    if digest(census_path) != CENSUS_SHA256:
        raise ValueError("Bounded primary census SHA differs")
    census = json.loads(census_path.read_text(encoding="utf-8"))
    checks = selected_checks(census)
    output.mkdir(parents=True, exist_ok=True)
    (output / "primary_json").mkdir(exist_ok=True)
    (output / "census.json").write_bytes(census_path.read_bytes())
    features = {r["record_id"]: r for r in read_tsv(feature_index) if r["record_id"] in checks}
    con = sqlite3.connect(
        Path(dataset / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    con.row_factory = sqlite3.Row
    source = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    compound_input = next(
        item for item in dataset_manifest["inputs"] if item["path"].endswith("01_compounds.csv")
    )
    compound_path = ROOT / compound_input["path"]
    if digest(compound_path) != compound_input["sha256"]:
        raise ValueError("Original compound CSV differs from frozen source manifest")
    wanted_compounds = {
        con.execute("SELECT compound_id FROM records WHERE record_id=?", (rid,)).fetchone()[0]
        for rid in checks
    }
    csv.field_size_limit(10000000)
    with compound_path.open(encoding="utf-8-sig", newline="") as handle:
        compounds = {
            row["compound_id"]: row
            for row in csv.DictReader(handle)
            if row["compound_id"] in wanted_compounds
        }
    write_json(output / "source_compounds.json", compounds)
    admitted, conditional, rejected, pins = [], [], [], {}
    for rid, check in sorted(checks.items()):
        raw = dict(con.execute("SELECT * FROM records WHERE record_id=?", (rid,)).fetchone())
        payload, sha = source.execute(
            "SELECT json_gzip,json_sha256 FROM peptide_records WHERE peptide_id=?",
            (check["peptide_id"],),
        ).fetchone()
        decompressed = gzip.decompress(payload)
        if hashlib.sha256(decompressed).hexdigest() != sha or sha != check["primary_json_sha256"]:
            raise ValueError("Primary JSON hash differs from database and pinned census")
        path = output / "primary_json" / f"{check['peptide_id']}.json"
        path.write_bytes(decompressed)
        pins[str(path.relative_to(output)).replace("\\", "/")] = sha
        try:
            proof = verify_observation(
                raw,
                json.loads(decompressed),
                check,
                features.get(rid, {}),
                compounds.get(raw["compound_id"]),
            )
        except ValueError as exc:
            rejected.append({"record_id": rid, "reason": str(exc)})
            continue
        (admitted if proof["training_admitted"] else conditional).append(proof)
    con.close()
    source.close()
    write_json(output / "verified_mapping.json", admitted)
    write_json(output / "conditional_mic90_evidence.json", conditional)
    write_json(output / "rejected_candidates.json", rejected)
    manifest = {
        "schema": SCHEMA,
        "source_manifest_sha256": digest(dataset / "manifest.json"),
        "bounded_census_sha256": CENSUS_SHA256,
        "feature_index_sha256": digest(feature_index),
        "verified_mapping_sha256": digest(output / "verified_mapping.json"),
        "conditional_mic90_sha256": digest(output / "conditional_mic90_evidence.json"),
        "source_compounds_sha256": digest(output / "source_compounds.json"),
        "original_compound_csv_sha256": compound_input["sha256"],
        "primary_json_sha256": pins,
        "admitted_records": len(admitted),
        "conditional_only_records": len(conditional),
        "admitted_endpoint_counts": dict(Counter(r["endpoint"] for r in admitted)),
        "rejected_records": len(rejected),
        "labels_graphs_family_assignments_unchanged": True,
        "tier_a_policy_unchanged": True,
        "mic90_conditions_must_not_be_unconditionally_pooled": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_evidence_map(path, records, source_manifest_sha256):
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("source_manifest_sha256") != source_manifest_sha256
        or manifest.get("bounded_census_sha256") != CENSUS_SHA256
        or digest(path.parent / "census.json") != CENSUS_SHA256
        or digest(path) != manifest["verified_mapping_sha256"]
    ):
        raise ValueError("Primary condition artifact/source hash mismatch")
    checks = selected_checks(json.loads((path.parent / "census.json").read_text(encoding="utf-8")))
    if digest(path.parent / "source_compounds.json") != manifest["source_compounds_sha256"]:
        raise ValueError("Source compound normalization evidence changed")
    compounds = json.loads((path.parent / "source_compounds.json").read_text(encoding="utf-8"))
    originals = {r["record_id"]: r for r in records}
    mapped = {}
    for proof in json.loads(path.read_text(encoding="utf-8")):
        rid = proof["record_id"]
        if (
            rid in mapped
            or rid not in checks
            or rid not in originals
            or not proof["training_admitted"]
        ):
            raise ValueError("Primary artifact contains duplicate or unapproved identity")
        check = checks[rid]
        source = path.parent / "primary_json" / f"{check['peptide_id']}.json"
        if digest(source) != check["primary_json_sha256"]:
            raise ValueError("Pinned primary payload changed")
        feature = {
            "record_id": rid,
            "structure_eligible": "true",
            "canonical_identity_group": proof["canonical_identity_group"],
            "split_group": proof["split_group"],
        }
        actual = verify_observation(
            originals[rid],
            json.loads(source.read_text(encoding="utf-8")),
            check,
            feature,
            compounds.get(originals[rid]["compound_id"]),
        )
        if actual != proof:
            raise ValueError("Primary proof differs from exact source revalidation")
        mapped[rid] = proof
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": digest(path),
        "verified_records": len(mapped),
        "applied_records": 0,
        "tier_a_policy_unchanged": True,
        "source_manifest_sha256": source_manifest_sha256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset_reclean_20261113_v1")
    parser.add_argument(
        "--feature-index",
        type=Path,
        default=ROOT / "data/peptide_modeling_multiseed20h_followup1_v1/feature_rows_index.tsv",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/dbaasp_primary_recovery_20261114_v1"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_evidence(args.dataset, args.feature_index, args.output), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
