"""Prepare bounded paper/context-specific reported-MIC90 label regressions.

The source measurement basis remains explicitly unverified. Frozen labels,
representations and the existing task compiler are never rewritten.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from peptide_modeling_infection_ic50_bridge import (
    EXCLUDED_ACTIVITY_KEYS,
    ROOT,
    canonical,
    graph,
    paper,
    precheck,
    read,
    read_table,
    scalar,
    sha,
    signature,
    unit,
    write,
)

SOURCE = ROOT / "data/mic90_offline_primary_audit_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_mic90_source_followup_20261114_v1"
QUANTITY = "source_reported_MIC90_measurement_basis_unverified"
PINS = {
    "delivery_receipt.json": "b06a3b06f36c4483d501c167e1f0da17e6fbccd09422bc0035a79acc24a02adf",
    "supplemental_review/delivery_receipt.json": (
        "66e4563d24dd546c711f422255eca45173b55000b6cd24aef382928645d92d12"
    ),
}
BASIS_FLAGS = (
    "growth_fraction_90_percent_established",
    "population_percentile_90_percent_established",
    "universal_MIC90_definition_verified",
)
TASK_IDS = {
    "TASK_864ca3b1af452924e8e49c9d",
    "TASK_ad2b81e74768780d0dda985b",
    "TASK_5de107f3d44f386e348a624b",
    "TASK_7b3b8666bafdd703ba81da99",
    "TASK_657a5a4c3e2c35d3ab8190db",
}


def verify_sources(source):
    protected = {}
    for name, expected in PINS.items():
        path = source / name
        if sha(path) != expected:
            raise ValueError("Pinned MIC90 source receipt changed")
        protected[path] = expected
        for relative, digest in read(path)["files"].items():
            artifact = path.parent / relative
            if not artifact.is_file() or sha(artifact) != digest:
                raise ValueError("Pinned MIC90 source artifact changed")
            protected[artifact] = digest
    return protected


def validate_definition(definition):
    if (
        definition.get("quantity") != QUANTITY
        or definition.get("legacy_endpoint") != "MIC90"
        or definition.get("raw_measure_group") != "MIC90"
        or any(definition.get(flag) is not False for flag in BASIS_FLAGS)
        or definition.get("independent_source_validation") is not False
        or not definition.get("unresolved_protocol_fields")
        or "growth_fraction_vs_isolate_population_percentile_basis"
        not in definition["unresolved_protocol_fields"]
    ):
        raise ValueError("MIC90 measurement basis must remain explicitly unverified")


def validate_record(row, frozen, check, primary, definition, compound=None):
    """Recheck literal measure, unique exact activity, graph, paper and conversion."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    validate_definition(definition)
    if (
        frozen["record_id"] != row["record_id"]
        or check["record_id"] != row["record_id"]
        or frozen["source_code"] != "DBAASP"
        or frozen["endpoint"] != "MIC90"
        or frozen["lane"] != "strict_numeric"
        or frozen["relation"] != "="
        or frozen["raw_relation"] != "="
        or row["relation"] != "="
        or row["source_relation"] != "="
        or row["original_endpoint"] != "MIC90"
        or row["quantity_semantics"] != QUANTITY
        or frozen["structure_model_eligible"] != "true"
        or frozen["unit"] != "uM"
        or row["unit"] != "uM"
        or check["status"] != "direct_primary_point_graph_proven"
    ):
        raise ValueError("Exact frozen MIC90 source/label/gate differs")
    pid = int(frozen["source_entry_id"].split(":")[1])
    if primary["id"] != pid or int(row["source_peptide_id"]) != pid:
        raise ValueError("Primary peptide identity differs")
    if (
        scalar(row["source_raw_value"]) != scalar(frozen["raw_value"])
        or row["source_raw_unit"] != frozen["raw_unit"]
    ):
        raise ValueError("Original scalar/unit lineage changed")
    matches = []
    for activity in primary.get("targetActivities", []):
        try:
            value = scalar(activity.get("concentration", ""))
        except ValueError:
            continue
        if (
            activity.get("activityMeasureValue") == "MIC90"
            and (activity.get("activityMeasureGroup") or {}).get("name") == "MIC90"
            and activity.get("peptideId") == pid
            and value == scalar(frozen["raw_value"])
            and unit((activity.get("unit") or {}).get("name")) == unit(frozen["raw_unit"])
        ):
            matches.append(activity)
    if len(matches) != 1 or str(matches[0]["id"]) != row["source_activity_id"]:
        raise ValueError("Exact MIC90 source activity is missing or ambiguous")
    activity = matches[0]
    context = {k: v for k, v in activity.items() if k not in EXCLUDED_ACTIVITY_KEYS}
    if (
        context != definition["source_context"]
        or context != check["source_context"]
        or signature(context) != row["source_context_signature"]
    ):
        raise ValueError("Reported MIC90 source context differs")
    actual_paper = paper(primary, activity.get("reference"))
    if actual_paper != definition["source_paper"] or actual_paper != row["source_paper"]:
        raise ValueError("Source publication differs")
    full_graph, key = graph(frozen["canonical_smiles"])
    if (
        full_graph not in {graph(s["smiles"])[0] for s in primary.get("smiles", [])}
        or graph(row["canonical_isomeric_smiles"])[0] != full_graph
        or key != row["canonical_identity_group"]
        or key != check["canonical_identity_group"]
        or row["exact_family_group"] != check["split_group"]
    ):
        raise ValueError("Complete source graph or exact family differs")
    original = scalar(frozen["raw_value"])
    if unit(frozen["raw_unit"]) == "um":
        factor, mass = 1.0, None
    elif unit(frozen["raw_unit"]) == "ug/ml":
        molecule = Chem.MolFromSmiles(full_graph)
        if (
            compound is None
            or compound["compound_id"] != frozen["compound_id"]
            or graph(compound["smiles"])[0] != full_graph
            or compound["molecular_weight_unit"] not in {"g/mol", "g mol-1", "Da"}
        ):
            raise ValueError("Mass conversion lacks an exact source graph and mass unit")
        mass = float(scalar(compound["molecular_weight"]))
        if format(mass, ".3f") != format(Descriptors.MolWt(molecule), ".3f"):
            raise ValueError("Source molecular weight differs from the full source graph")
        factor = 1000.0 / mass
    else:
        raise ValueError("Unsupported original MIC90 concentration unit")
    normalized = format(float(original) * factor, ".15g")
    if (
        normalized != frozen["value"]
        or row["value"] != frozen["value"]
        or not math.isclose(
            float(row["source_normalization_factor"]), factor, rel_tol=1e-15, abs_tol=0.0
        )
    ):
        raise ValueError("Original mass/molar normalization or target changed")
    target = math.log10(float(normalized))
    if not math.isclose(float(row["target"]), target, rel_tol=1e-14, abs_tol=1e-14):
        raise ValueError("Source target transform differs")
    return dict(
        record_id=row["record_id"],
        source_peptide_id=pid,
        source_activity_id=activity["id"],
        source_paper=actual_paper,
        source_raw_value=frozen["raw_value"],
        source_raw_unit=frozen["raw_unit"],
        original_value=frozen["value"],
        normalized_value=normalized,
        source_molecular_weight=mass,
        source_normalization_factor=factor,
        canonical_identity_group=key,
        target=target,
        measurement_basis_unverified=True,
        **{flag: False for flag in BASIS_FLAGS},
    )


def variation_audit(rows):
    by_unit = defaultdict(set)
    normalized_by_original = defaultdict(set)
    for row in rows:
        by_unit[unit(row["source_raw_unit"])].add(str(scalar(row["source_raw_value"])))
        original = (unit(row["source_raw_unit"]), str(scalar(row["source_raw_value"])))
        normalized_by_original[original].add(str(scalar(row["value"])))
    normalized = {str(scalar(row["value"])) for row in rows}
    pairs = {(unit(row["source_raw_unit"]), str(scalar(row["source_raw_value"]))) for row in rows}
    return dict(
        original_units=dict(Counter(row["source_raw_unit"] for row in rows)),
        original_scalar_distinct_by_unit={u: len(v) for u, v in by_unit.items()},
        original_unit_scalar_pairs_distinct=len(pairs),
        normalized_value_distinct=len(normalized),
        normalized_log10_target_distinct=len({float(row["target"]) for row in rows}),
        original_unit_scalar_pairs_with_multiple_normalized_values=sum(
            len(values) > 1 for values in normalized_by_original.values()
        ),
        molecular_weight_conversion_can_induce_target_variation=any(
            original[0] == "ug/ml" and len(values) > 1
            for original, values in normalized_by_original.items()
        ),
        original_mass_label_constant=(len(pairs) == 1 and next(iter(pairs))[0] == "ug/ml"),
        conversion_variation_is_not_new_activity_evidence=True,
    )


def prepare(source=SOURCE, output=OUTPUT):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    source, output = Path(source), Path(output)
    protected = verify_sources(source)
    if output.exists():
        raise FileExistsError("Existing derived admission is immutable; use a new output")
    audit = read(source / "audit.json")
    candidate = read(source / "supplemental_review/candidate_task_manifest.json")
    reviews = {r["task_id"]: r for r in read(source / "supplemental_review/review.json")["tasks"]}
    checks = {r["record_id"]: r for r in audit["record_checks"]}
    if {task["task_id"] for task in candidate["tasks"]} != TASK_IDS:
        raise ValueError("Only the five explicitly reviewed source scopes are admitted")
    prepared = {
        task["task_id"]: read_table(source / "supplemental_review" / task["table_path"])
        for task in candidate["tasks"]
    }
    ids = [r["record_id"] for rows in prepared.values() for r in rows]
    if len(ids) != 43 or len(set(ids)) != 43:
        raise ValueError("Forty-three distinct source observations are required")
    family_path = ROOT / "data/peptide_modeling_multiseed20h_overlay_v1/feature_rows_index.tsv"
    bank = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
    pointer = bank / "features/manifest.json"
    for path in (family_path, pointer):
        expected = audit["source_sha256"][str(path.relative_to(ROOT))]
        if sha(path) != expected:
            raise ValueError("Original family index or feature pointer changed")
        protected[path] = expected
    features = read(pointer)
    _verify_artifacts(features)
    registry = read(bank / "representation_methods.json")
    methods = [m for m in registry["methods"] if m["source_kind"] == "structure"]
    if len(methods) != 10 or len(features["artifact_hashes"]) != 48:
        raise ValueError("Expected ten methods and the frozen 48-artifact feature bank")
    with family_path.open(encoding="utf-8-sig", newline="") as stream:
        families = {
            r["record_id"]: r["split_group"]
            for r in csv.DictReader(stream, delimiter="\t")
            if r["record_id"] in set(ids)
        }
    dataset = ROOT / "data/dataset_reclean_20261113_v1"
    for path in (dataset / "manifest.json", dataset / "catalogue.sqlite"):
        expected = audit["source_sha256"][str(path.relative_to(ROOT))]
        if sha(path) != expected:
            raise ValueError("Frozen dataset source changed")
        protected[path] = expected
    compound_input = next(
        i
        for i in read(dataset / "manifest.json")["inputs"]
        if i["path"].endswith("01_compounds.csv")
    )
    compound_path = ROOT / compound_input["path"]
    if (
        sha(compound_path) != compound_input["sha256"]
        or sha(compound_path) != audit["source_compound_csv_sha256"]
    ):
        raise ValueError("Original compound/MW source changed")
    protected[compound_path] = compound_input["sha256"]
    connection = sqlite3.connect((dataset / "catalogue.sqlite").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    frozen = {
        rid: dict(
            connection.execute(
                "SELECT * FROM records_with_structure_qc WHERE record_id=?", (rid,)
            ).fetchone()
        )
        for rid in ids
    }
    connection.close()
    wanted = {r["compound_id"] for r in frozen.values() if unit(r["raw_unit"]) == "ug/ml"}
    csv.field_size_limit(10000000)
    with compound_path.open(encoding="utf-8-sig", newline="") as stream:
        compounds = {
            r["compound_id"]: r for r in csv.DictReader(stream) if r["compound_id"] in wanted
        }
    tasks, evidence, splits, coverage = [], {}, {}, {}
    activity_keys = []
    for proposed in candidate["tasks"]:
        validate_definition(proposed["task_definition"])
        path = source / "supplemental_review" / proposed["table_path"]
        if sha(path) != proposed["table_sha256"]:
            raise ValueError("Candidate source table changed")
        rows = prepared[proposed["task_id"]]
        proofs = []
        for row in rows:
            original, check = frozen[row["record_id"]], checks[row["record_id"]]
            if families.get(row["record_id"]) != row["exact_family_group"]:
                raise ValueError("Original FAMILY differs")
            primary_path = source / "primary_json" / f"{check['peptide_id']}.json"
            if (
                sha(primary_path) != check["primary_json_sha256"]
                or sha(primary_path) != row["source_json_sha256"]
            ):
                raise ValueError("Pinned primary JSON changed")
            proof = validate_record(
                row,
                original,
                check,
                read(primary_path),
                proposed["task_definition"],
                compounds.get(original["compound_id"]),
            )
            proofs.append({**proof, "primary_sha256": sha(primary_path)})
            activity_keys.append((proof["source_peptide_id"], proof["source_activity_id"]))
            row.update(
                canonical_quantity=QUANTITY,
                source_context_json=canonical(proposed["task_definition"]["source_context"]),
                unknown_reported_fields_json=canonical(
                    proposed["task_definition"]["unknown_reported_fields"]
                ),
                unresolved_protocol_fields_json=canonical(
                    proposed["task_definition"]["unresolved_protocol_fields"]
                ),
                **{flag: "false" for flag in BASIS_FLAGS},
            )
        split = precheck(rows)
        reviewed = reviews[proposed["task_id"]]["precheck"]
        for current, prior in zip(split["probes"], reviewed["seeds"], strict=True):
            if current["seed"] != prior["seed"] or any(
                set(a[field]) != set(b[field])
                for a, b in zip(current["folds"], prior["folds"], strict=True)
                for field in ("train_record_ids", "validation_record_ids")
            ):
                raise ValueError("Actual graph/FAMILY folds differ from independent review")
        splits[proposed["task_id"]] = split
        evidence[proposed["task_id"]] = dict(records=proofs, variation_audit=variation_audit(rows))
        coverage[proposed["task_id"]] = {}
        for method in methods:
            payload = load_feature_route(rows, features, method["route"])
            if (
                payload["record_ids"] != [r["record_id"] for r in rows]
                or not payload["valid"].all()
            ):
                raise ValueError("Existing complete-graph method coverage is incomplete")
            coverage[proposed["task_id"]][method["method_id"]] = dict(
                records=len(rows), dimensions=payload["X"].shape[1]
            )
        task = copy.deepcopy(proposed)
        task.update(
            status="eligible",
            training_admitted=True,
            table_path=f"task_tables/{task['task_id']}.tsv",
        )
        task.pop("table_sha256")
        task["task_definition"].update(
            canonical_quantity=QUANTITY,
            unknown_conditions_remain_unknown=True,
            source_protocol_complete=False,
        )
        tasks.append(task)
    if len(set(activity_keys)) != 43:
        raise ValueError("One primary activity cannot be duplicated as independent observations")
    output.mkdir(parents=True)
    for task in tasks:
        path = output / task["table_path"]
        path.parent.mkdir(exist_ok=True)
        rows = prepared[task["task_id"]]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        task["table_sha256"] = sha(path)
    manifest = dict(
        schema="source_defined_MIC90_admission_v1",
        tasks=tasks,
        source_candidate_manifest_sha256=sha(
            source / "supplemental_review/candidate_task_manifest.json"
        ),
        original_source_labels_modified=False,
        measurement_basis_unverified=True,
    )
    for name, value in [
        ("task_manifest.json", manifest),
        ("targeted_task_manifest.json", manifest),
        ("source_verification.json", dict(status="verified", tasks=evidence)),
        ("fivefold_prechecks.json", splits),
        (
            "feature_coverage.json",
            dict(
                fingerprint=features["input_fingerprint"],
                tasks=coverage,
                verified_feature_artifacts=48,
                methods=10,
                new_extraction=False,
            ),
        ),
        (
            "representation_methods.json",
            {**registry, "methods": methods, "independent_method_count": 10},
        ),
    ]:
        write(output / name, value)
    config = read(
        ROOT
        / "data"
        / "peptide_modeling_infection_ic50_followup_20261114_v1"
        / "recommended_training_config.json"
    )
    write(output / "recommended_training_config.json", config)
    write(
        output / "feature_reuse_plan.json",
        dict(
            source=str(bank),
            pointer_sha256=sha(pointer),
            fingerprint=features["input_fingerprint"],
            methods=10,
            artifacts=48,
            records=43,
            new_extraction_required=False,
            modified_structures_preserved=True,
        ),
    )
    if any(sha(p) != expected for p, expected in protected.items()):
        raise ValueError("Protected source changed during admission")
    write(
        output / "admission_receipt.json",
        dict(
            status="prepared_not_trained",
            tasks=5,
            records=43,
            unique_primary_activities=43,
            global_unique_graphs=len(
                {r["canonical_identity_group"] for rows in prepared.values() for r in rows}
            ),
            protected_source_artifacts_unchanged=len(protected),
            methods=10,
            feature_artifacts_verified=48,
            actual_fivefold_splits=75,
            source_measurement_basis_verified=False,
            model_fits=0,
            remote_operations=False,
            files={
                p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps(dict(status="prepared_not_trained", tasks=len(manifest["tasks"]), records=43)))


if __name__ == "__main__":
    main()
