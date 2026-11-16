"""Prepare explicitly source-qualified infection-inhibition IC50 internal tasks.

This derived admission never rewrites frozen labels, representation gates or the
generic binding compiler. It preserves incomplete protocol as unknown evidence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ic50i_source_feasibility_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_infection_ic50_followup_20261114_v1"
RECEIPT_PINS = {
    "delivery_receipt.json": "7e5444dddaedfbc1fb0b94ee62dcd06b2486c0c87531497bc458e198446610b0",
    "supplemental_review/delivery_receipt.json": (
        "c65d2ae1a4441d41538122baa3b29338ddd53f8071f24efeba2d63bc19f8edc5"
    ),
}
MEASURE = "source_reported_infection_inhibition_IC50_not_binding_affinity"
PROTOCOL = "internal_grouped_cv_only_no_independent_holdout"
SEEDS = (42, 2026, 3407)
EXCLUDED_ACTIVITY_KEYS = {
    "id",
    "peptideId",
    "activityMeasureGroup",
    "activityMeasureValue",
    "concentration",
    "activity",
    "unit",
    "reference",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_sources(source):
    protected = {}
    for name, pin in RECEIPT_PINS.items():
        receipt_path = source / name
        if sha(receipt_path) != pin:
            raise ValueError("Pinned source receipt changed: " + name)
        protected[receipt_path] = pin
        for relative, expected in read(receipt_path)["files"].items():
            path = receipt_path.parent / relative
            if not path.is_file() or sha(path) != expected:
                raise ValueError("Source artifact changed: " + str(path))
            protected[path] = expected
    return protected


def scalar(value):
    if not re.fullmatch(r"\s*[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*", str(value)):
        raise ValueError("Source value is not an exact scalar")
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError("Invalid source decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("Source value must be positive finite")
    return result


def unit(value):
    return str(value).strip().replace("\u00b5", "u").replace("\u03bc", "u").lower()


def graph(value):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError("Invalid full structure")
    if any(v == "?" for _, v in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)):
        raise ValueError("Unassigned stereochemical centre")
    result = Chem.MolToSmiles(molecule, isomericSmiles=True)
    return result, "CANON_SMILES:" + hashlib.sha256(result.encode()).hexdigest()


def paper(primary, reference):
    if not re.fullmatch(r"[1-9][0-9]*", str(reference)):
        raise ValueError("Explicit single paper reference is required")
    index = int(reference) - 1
    articles = primary.get("articles", [])
    if not 0 <= index < len(articles):
        raise ValueError("Paper reference outside source articles")
    article = articles[index]
    pmid = str((article.get("pubmed") or {}).get("pubmedId", ""))
    if not re.fullmatch(r"[0-9]+", pmid) or not article.get("title"):
        raise ValueError("Paper identity is missing")
    return "PMID:" + pmid


def validate_record(row, frozen, check, primary, definition):
    """Recheck a selected row against exact activity, frozen label and paper/context."""
    if (
        frozen["record_id"] != row["record_id"]
        or check["record_id"] != row["record_id"]
        or frozen["endpoint"] != "IC50_inhibitor"
        or frozen["endpoint_raw"] != "DBAASP IC50 I"
        or frozen["lane"] != "strict_numeric"
        or frozen["relation"] != "="
        or frozen["raw_relation"] != "="
        or frozen["structure_model_eligible"] != "true"
        or frozen["unit"] != "uM"
        or row["original_endpoint"] != "IC50_inhibitor"
        or row["source_endpoint_raw"] != "DBAASP IC50 I"
        or row["quantity_semantics"] != "infection_inhibition_IC50"
    ):
        raise ValueError("Frozen endpoint, label lane or representation gate differs")
    if check["status"] != "direct_primary_point_graph_proven":
        raise ValueError("Primary audit did not prove this row")
    pid = int(frozen["source_entry_id"].split(":")[1])
    if primary["id"] != pid or check["peptide_id"] != pid:
        raise ValueError("Primary peptide identity differs")
    # These seventeen explicitly pinned rows all use exact micromolar values.
    if unit(frozen["raw_unit"]) != "um" or scalar(frozen["raw_value"]) != scalar(frozen["value"]):
        raise ValueError("This bounded admission does not convert other source units")
    matches = [
        activity
        for activity in primary.get("targetActivities", [])
        if activity.get("activityMeasureValue") == "IC50 I"
        and (activity.get("activityMeasureGroup") or {}).get("name") == "IC50 I"
        and activity.get("peptideId") == pid
        and unit((activity.get("unit") or {}).get("name")) == "um"
        and str(activity.get("concentration", "")).strip()
        and re.fullmatch(
            r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", str(activity["concentration"]).strip()
        )
        and scalar(activity["concentration"]) == scalar(frozen["raw_value"])
    ]
    if len(matches) != 1 or str(matches[0]["id"]) != row["source_activity_id"]:
        raise ValueError("Exact source activity is ambiguous or differs")
    activity = matches[0]
    if (activity["activityMeasureGroup"].get("description") or "").lower() != (
        "50% inhibition of infection"
    ):
        raise ValueError("Infection-inhibition measure definition differs")
    context = {key: value for key, value in activity.items() if key not in EXCLUDED_ACTIVITY_KEYS}
    if (
        context != definition["source_context"]
        or signature(context) != row["source_context_signature"]
        or context != check["source_context"]
    ):
        raise ValueError("Full reported source context differs")
    source_paper = paper(primary, activity.get("reference"))
    if source_paper != row["source_paper"] or source_paper != definition["source_paper"]:
        raise ValueError("Source activity belongs to a different paper")
    structure, key = graph(frozen["canonical_smiles"])
    source_graphs = {graph(value["smiles"])[0] for value in primary.get("smiles", [])}
    if (
        structure not in source_graphs
        or graph(row["canonical_isomeric_smiles"])[0] != structure
        or row["canonical_identity_group"] != key
        or check["canonical_identity_group"] != key
        or row["exact_family_group"] != check["split_group"]
    ):
        raise ValueError("Full source graph or original family identity differs")
    value = scalar(frozen["value"])
    if scalar(row["value"]) != value or row["unit"] != "uM":
        raise ValueError("Candidate changed the frozen target")
    target = math.log10(float(value))
    if not math.isclose(float(row["target"]), target, abs_tol=1e-14, rel_tol=1e-14):
        raise ValueError("Candidate target transform differs")
    return dict(
        record_id=row["record_id"],
        activity_id=activity["id"],
        source_paper=source_paper,
        context_sha256=signature(context),
        canonical_identity_group=key,
        original_frozen_value=frozen["value"],
        target=target,
        source_graph_and_label_verified=True,
    )


def precheck(rows):
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated source record ID")
    for identity in ("canonical_identity_group", "exact_family_group"):
        memberships = {}
        for row in rows:
            prior = memberships.setdefault(row[identity], row["split_group"])
            if prior != row["split_group"]:
                raise ValueError("Graph or exact family splits across groups")
    groups = np.asarray([row["split_group"] for row in rows])
    y = np.asarray([float(row["target"]) for row in rows])
    if len(set(groups)) < 5 or not np.isfinite(y).all():
        raise ValueError("Insufficient finite independent fivefold inputs")
    probes, partitions = [], set()
    for seed in SEEDS:
        folds = []
        for train, valid in GroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups):
            if len(set(y[train])) < 2:
                raise ValueError("Training fold has no target variation")
            for identity in ("canonical_identity_group", "exact_family_group", "split_group"):
                if {rows[i][identity] for i in train} & {rows[i][identity] for i in valid}:
                    raise ValueError("Identity leaks across fivefold split")
            folds.append(
                dict(
                    train_record_ids=[ids[i] for i in train],
                    validation_record_ids=[ids[i] for i in valid],
                )
            )
        if sorted(rid for fold in folds for rid in fold["validation_record_ids"]) != sorted(ids):
            raise ValueError("Incomplete validation fold coverage")
        partitions.add(canonical(sorted(sorted(f["validation_record_ids"]) for f in folds)))
        probes.append(dict(seed=seed, folds=folds, actual_inner_folds=5))
    return dict(
        status="verified",
        probes=probes,
        distinct_partitions=len(partitions),
        graph_and_family_crossfold_overlap=0,
        models_fitted=0,
    )


def prepare(source=SOURCE, output=OUTPUT):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    source, output = Path(source), Path(output)
    protected = verify_sources(source)
    if output.exists():
        raise FileExistsError("Use a new derived output; existing admission remains immutable")
    candidate = read(source / "supplemental_review/candidate_task_manifest.json")
    audit = read(source / "audit.json")
    checks = {row["record_id"]: row for row in audit["record_checks"]}
    reviewed = {
        scope["review_scope_id"]: scope
        for scope in read(source / "supplemental_review/review.json")["scopes"]
    }
    bank = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
    pointer = bank / "features/manifest.json"
    if sha(pointer) != candidate["feature_reuse"]["pointer_sha256"]:
        raise ValueError("Frozen feature pointer changed")
    features = read(pointer)
    _verify_artifacts(features)
    registry = read(bank / "representation_methods.json")
    methods = [m for m in registry["methods"] if m["source_kind"] == "structure"]
    if len(methods) != 10:
        raise ValueError("Exactly ten existing full-graph methods are required")
    registry = {**registry, "methods": methods, "independent_method_count": 10}
    family_path = ROOT / "data/peptide_modeling_multiseed20h_overlay_v1/feature_rows_index.tsv"
    expected_family_sha = audit["source_sha256"][str(family_path.relative_to(ROOT))]
    if sha(family_path) != expected_family_sha:
        raise ValueError("Original exact-family feature index changed")
    wanted = {
        row["record_id"]
        for task in candidate["tasks"]
        for row in read_table(source / "supplemental_review" / task["table_path"])
    }
    with family_path.open(encoding="utf-8-sig", newline="") as stream:
        original_families = {
            row["record_id"]: row["split_group"]
            for row in csv.DictReader(stream, delimiter="\t")
            if row["record_id"] in wanted
        }
    protected.update({pointer: sha(pointer), family_path: expected_family_sha})
    connection = sqlite3.connect(
        (ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite").as_uri() + "?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    tasks, tables, evidence, splits, coverage = [], {}, [], {}, {}
    for proposed in candidate["tasks"]:
        rows = read_table(source / "supplemental_review" / proposed["table_path"])
        if sha(source / "supplemental_review" / proposed["table_path"]) != proposed["table_sha256"]:
            raise ValueError("Candidate table changed")
        for row in rows:
            frozen = connection.execute(
                "SELECT * FROM records_with_structure_qc WHERE record_id=?", (row["record_id"],)
            ).fetchone()
            if frozen is None:
                raise ValueError("Candidate is absent from the frozen source")
            if original_families.get(row["record_id"]) != row["exact_family_group"]:
                raise ValueError("Candidate family differs from the original identity index")
            check = checks[row["record_id"]]
            path = source / "primary_json" / f"{check['peptide_id']}.json"
            if sha(path) != check["primary_json_sha256"]:
                raise ValueError("Exact primary source JSON changed")
            proof = validate_record(
                row, dict(frozen), check, read(path), proposed["task_definition"]
            )
            evidence.append({**proof, "primary_sha256": sha(path), "task_id": proposed["task_id"]})
            row.update(
                source_relation=frozen["raw_relation"],
                relation=frozen["relation"],
                source_raw_value=frozen["raw_value"],
                source_raw_unit=frozen["raw_unit"],
                source_context_json=canonical(proposed["task_definition"]["source_context"]),
                unknown_conditions_json=canonical(
                    proposed["task_definition"]["unknown_conditions"]
                ),
                undeclared_protocol_fields_json=canonical(
                    proposed["task_definition"]["undeclared_protocol_fields"]
                ),
            )
        task = copy.deepcopy(proposed)
        task.update(
            status="eligible",
            training_admitted=True,
            models_fitted=0,
            table_path=f"task_tables/{task['task_id']}.tsv",
        )
        task.pop("table_sha256")
        task["task_definition"].update(
            canonical_measure=MEASURE,
            unknown_conditions_remain_unknown=True,
            original_raw_endpoint_lineage_preserved=True,
            feature_inputs="complete_source_stereochemical_graph_only",
        )
        splits[task["task_id"]] = precheck(rows)
        if splits[task["task_id"]]["distinct_partitions"] != proposed["distinct_partitions"]:
            raise ValueError("Actual source split evidence differs from review")
        prior_folds = reviewed[proposed["split_evidence_scope"]]["split_precheck"]["seeds"]
        for actual, prior in zip(splits[task["task_id"]]["probes"], prior_folds, strict=True):
            if actual["seed"] != prior["seed"] or any(
                set(a[field]) != set(b[field])
                for a, b in zip(actual["folds"], prior["folds"], strict=True)
                for field in ("train_record_ids", "validation_record_ids")
            ):
                raise ValueError("Actual fold members differ from reviewed evidence")
        task_coverage = {}
        for method in methods:
            payload = load_feature_route(rows, features, method["route"])
            if (
                payload["record_ids"] != [r["record_id"] for r in rows]
                or not payload["valid"].all()
            ):
                raise ValueError("Incomplete graph feature coverage")
            task_coverage[method["method_id"]] = dict(
                records=len(rows), dimensions=payload["X"].shape[1]
            )
        coverage[task["task_id"]] = task_coverage
        tables[task["task_id"]] = rows
        tasks.append(task)
    connection.close()
    all_ids = [row["record_id"] for rows in tables.values() for row in rows]
    if len(tasks) != 3 or len(all_ids) != 17 or len(set(all_ids)) != 17:
        raise ValueError("This admission is limited to three distinct tasks and seventeen records")
    output.mkdir(parents=True)
    for task in tasks:
        path = output / task["table_path"]
        path.parent.mkdir(exist_ok=True)
        rows = tables[task["task_id"]]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        task["table_sha256"] = sha(path)
    manifest = dict(
        schema="source_qualified_infection_ic50_admission_v1",
        tasks=tasks,
        source_candidate_manifest_sha256=sha(
            source / "supplemental_review/candidate_task_manifest.json"
        ),
        derived_admission_only=True,
        original_frozen_data_modified=False,
        feature_fingerprint=features["input_fingerprint"],
    )
    write(output / "task_manifest.json", manifest)
    write(output / "targeted_task_manifest.json", manifest)
    write(output / "fivefold_prechecks.json", splits)
    write(output / "source_verification.json", dict(status="verified", records=evidence))
    write(
        output / "feature_coverage.json",
        dict(
            fingerprint=features["input_fingerprint"], tasks=coverage, new_feature_extraction=False
        ),
    )
    write(output / "representation_methods.json", registry)
    config = dict(
        resources=dict(training_host="node23", n_jobs=1, nice=10, cpu_budget=1),
        training=dict(
            routes=[m["route"] for m in methods] + ["structure_methods_all"],
            models=[
                "linear",
                "elasticnet",
                "random_forest",
                "extra_trees",
                "histgb",
                "lightgbm",
                "xgboost",
                "svm",
            ],
            search_backend="optuna",
            search_budget=128,
            inner_folds=5,
            min_rows=5,
            min_groups=5,
            n_jobs=1,
            seed=42,
            refit_full=False,
            svm_max_features=16384,
            histgb_max_features=16384,
            elasticnet_max_features=16384,
        ),
        required_seeds=list(SEEDS),
        validation_protocol=PROTOCOL,
    )
    write(output / "recommended_training_config.json", config)
    write(
        output / "feature_reuse_plan.json",
        dict(
            source=str(bank),
            pointer_sha256=sha(pointer),
            fingerprint=features["input_fingerprint"],
            new_extraction_required=False,
            preserve_complete_modified_smiles=True,
            records=len(all_ids),
        ),
    )
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("Protected source artifacts changed during admission")
    write(
        output / "admission_receipt.json",
        dict(
            status="prepared_not_trained",
            tasks=3,
            records=17,
            original_and_supplemental_files_unchanged=len(protected),
            feature_pointer_unchanged=True,
            actual_feature_methods=10,
            source_labels_unchanged=True,
            models_fitted=0,
            remote_operations=False,
            source_protocol_complete=False,
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
    print(json.dumps(dict(status="prepared_not_trained", tasks=len(manifest["tasks"]), records=17)))


if __name__ == "__main__":
    main()
