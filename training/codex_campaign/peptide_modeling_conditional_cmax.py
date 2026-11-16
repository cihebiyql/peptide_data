"""Prepare and guard source-reported Cmax conditional on minimum known context.

This is internal exploration, not an intrinsic molecular property, a causal dose
response, or a fully specified regimen/patient prediction. No training is run here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

SCHEMA = "conditional_cmax_minimum_context_v1"
ESTIMAND = "source_reported_parent_peak_concentration_given_minimum_known_context"
WARNING = (
    "Internal five-fold model-selection scores are selection-optimistic; no independent "
    "holdout. Compound, context and study are confounded. Rare contexts lack support. "
    "Frequency, infusion duration, study day, formulation, age and health are not fully "
    "controlled. Not a causal dose response or a fully specified regimen/patient model."
)
# Fixed scientific ontology: no categories or scaling are learned from all labels.
ONTOLOGY = {
    "species": ["human", "mouse", "rat", "dog", "monkey", "rabbit"],
    "route": ["oral", "intravenous", "subcutaneous", "intraperitoneal", "intramuscular"],
    "matrix": ["plasma", "serum", "whole_blood"],
}
DOSE_UNITS = {
    "mg": ("mass_absolute_mg", 1),
    "g": ("mass_absolute_mg", 1000),
    "ug": ("mass_absolute_mg", 0.001),
    "mg/kg": ("mass_per_kg_mg", 1),
    "ug/kg": ("mass_per_kg_mg", 0.001),
    "umol": ("molar_absolute_umol", 1),
    "nmol": ("molar_absolute_umol", 0.001),
    "umol/kg": ("molar_per_kg_umol", 1),
    "nmol/kg": ("molar_per_kg_umol", 0.001),
    "mg/m'2": ("mass_per_m2_mg", 1),
    "mg/m2": ("mass_per_m2_mg", 1),
    "mg/kg/day": ("daily_mass_per_kg_mg", 1),
}
BASES = sorted({basis for basis, _ in DOSE_UNITS.values()})
CONTEXT_KEYS = {"species", "route", "matrix", "dose_value", "dose_reported_unit"}
FEATURE_NAMES = (
    [f"context:dose_{basis}" for basis in BASES]
    + [f"context:basis_{basis}" for basis in BASES]
    + [f"context:{key}_{value}" for key, values in ONTOLOGY.items() for value in values]
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def encode_context(context):
    """Require explicit inputs; never default dose, unknown states or unit conversions."""
    if not isinstance(context, dict) or set(context) != CONTEXT_KEYS:
        raise ValueError("Exactly the five known context inputs are required; no IDs or outcomes")
    for key, values in ONTOLOGY.items():
        if context[key] not in values:
            raise ValueError("Unknown or unsupported known context: " + key)
    unit = context["dose_reported_unit"]
    if unit not in DOSE_UNITS:
        raise ValueError("Unsupported/ambiguous dose basis or source unit")
    dose = float(context["dose_value"])
    if isinstance(context["dose_value"], bool) or not math.isfinite(dose) or dose <= 0:
        raise ValueError("Explicit dose must be positive and finite")
    basis, factor = DOSE_UNITS[unit]
    # Separate numeric channels prevent mg/kg, mg/m2, and mg/kg/day from sharing a quantity.
    vector = [dose * factor if item == basis else 0.0 for item in BASES]
    vector += [float(item == basis) for item in BASES]
    vector += [float(context[key] == item) for key, items in ONTOLOGY.items() for item in items]
    result = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Dose exceeds finite feature range")
    return result


def context_contract(records, audit_sha):
    contexts = {r["record_id"]: r["conditional_core_context"] for r in records}
    for value in contexts.values():
        encode_context(value)
    value = {
        "schema": SCHEMA,
        "estimand": ESTIMAND,
        "feature_names": FEATURE_NAMES,
        "ontology": ONTOLOGY,
        "dose_units": DOSE_UNITS,
        "source_audit_sha256": audit_sha,
        "contexts_by_record": contexts,
        "inference_requires_all_known_context": True,
        "warning": WARNING,
        "chemical_method_count_increment": 0,
    }
    # JSON-normalize tuples so the contract survives persistence without changing its hash.
    value = json.loads(json.dumps(value))
    return {**value, "sha256": object_hash(value)}


def validate_contract(contract):
    unsigned = {k: v for k, v in contract.items() if k != "sha256"}
    if object_hash(unsigned) != contract.get("sha256"):
        raise ValueError("External context contract hash mismatch")
    if (
        contract.get("schema") != SCHEMA
        or contract.get("estimand") != ESTIMAND
        or contract.get("feature_names") != FEATURE_NAMES
        or contract.get("ontology") != ONTOLOGY
        or contract.get("dose_units") != json.loads(json.dumps(DOSE_UNITS))
        or contract.get("inference_requires_all_known_context") is not True
        or contract.get("chemical_method_count_increment") != 0
        or not re.fullmatch(r"[0-9a-f]{64}", contract.get("source_audit_sha256", ""))
    ):
        raise ValueError("Unsupported external context contract")
    for value in contract["contexts_by_record"].values():
        encode_context(value)
    return contract


def context_matrix(rows, contract):
    validate_contract(contract)
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) - set(contract["contexts_by_record"]):
        raise ValueError("Context source record identity missing or duplicated")
    return np.vstack([encode_context(contract["contexts_by_record"][item]) for item in ids])


def append_context(payload, rows, contract):
    if payload["record_ids"] != [r["record_id"] for r in rows]:
        raise ValueError("Chemical/context row order mismatch")
    if any(name.startswith("context:") for name in payload["feature_names"]):
        raise ValueError("Context was already appended")
    return {
        **payload,
        "X": np.column_stack((payload["X"], context_matrix(rows, contract))),
        "feature_names": payload["feature_names"] + FEATURE_NAMES,
    }


def predict_contextual(model, chemical_X, chemical_names, contexts, model_contract):
    """Supported inference entry point: caller supplies context for every full graph."""
    contract = validate_contract(model_contract["fixed_external_covariates"])
    X = np.asarray(chemical_X, dtype=np.float32)
    if X.ndim != 2 or len(contexts) != len(X) or not contexts:
        raise ValueError("Explicit known context is required per chemical row")
    if chemical_names + contract["feature_names"] != model_contract["feature_names"]:
        raise ValueError("Chemical or context feature schema differs from fitted model")
    if any(str(name).startswith("context:") for name in chemical_names):
        raise ValueError("Chemical inputs must not contain outcome/context columns")
    return model.predict(np.column_stack((X, np.vstack([encode_context(c) for c in contexts]))))


def contextual_diagnostics(rows, contract, chemical_X, y, groups, splits, settings, training):
    """Fixed ridge ablations on identical folds, kept separate from Optuna method counts."""
    context = context_matrix(rows, contract)
    designs = {
        "context_only": context,
        "molecule_only": chemical_X,
        "molecule_plus_context": np.column_stack((chemical_X, context)),
    }
    values = {}
    for name, X in designs.items():
        estimator = training._estimator(
            "linear", "regression", {"regularization": 1.0}, settings, X.shape[1]
        )
        predictions, _ = training._cv_predictions(
            estimator, X, y, splits, "regression", settings["n_jobs"]
        )
        errors = np.abs(y - predictions)
        values[name] = {
            "model": "fixed_ridge_regularization_1_not_optuna_selected",
            "metrics": training._metrics(y, predictions, "regression"),
            "compound_equal_weight_mae": float(
                np.mean([errors[groups == group].mean() for group in set(groups)])
            ),
            "predictions": [float(x) for x in predictions],
        }
    coverage = []
    contexts = contract["contexts_by_record"]
    for fold, (train, test) in enumerate(splits):
        unsupported = {}
        for key in ONTOLOGY:
            training_values = {contexts[rows[i]["record_id"]][key] for i in train}
            unsupported[key] = sorted(
                {contexts[rows[i]["record_id"]][key] for i in test} - training_values
            )
        train_bases = {
            DOSE_UNITS[contexts[rows[i]["record_id"]]["dose_reported_unit"]][0] for i in train
        }
        unsupported["dose_basis"] = sorted(
            {DOSE_UNITS[contexts[rows[i]["record_id"]]["dose_reported_unit"]][0] for i in test}
            - train_bases
        )
        coverage.append(
            {
                "fold": fold,
                "train_groups": sorted(set(groups[train])),
                "validation_groups": sorted(set(groups[test])),
                "unsupported_validation_context": unsupported,
            }
        )
    return {
        "warning": WARNING,
        "chemical_method_count_increment": 0,
        "ablations": values,
        "fold_context_coverage": coverage,
        "record_ids": [r["record_id"] for r in rows],
        "molecular_signal_established": False,
        "interpretation": (
            "Diagnostic ablations; do not attribute improvement to chemistry "
            "without source/context support"
        ),
    }


def validate_source_record(record, official, frozen):
    """Revalidate exact official activity/scalar/units/full stereo graph and frozen rows."""
    from rdkit import Chem

    if (
        record.get("endpoint") != "Cmax"
        or not record.get("conditional_core_candidate")
        or record.get("conditional_core_exclusion_reasons")
        or set(record.get("official_exact_match", {}).values()) != {True}
    ):
        raise ValueError("Candidate source proof is incomplete or conflicted")
    for key, expected in record["frozen_source"].items():
        if frozen.get(key) != expected:
            raise ValueError("Frozen source changed: " + key)
    if (
        frozen["record_id"] != record["record_id"]
        or record["source_entry_id"] != "activity:" + str(official["activity_id"])
        or frozen["source_entry_id"] != record["source_entry_id"]
        or frozen["value"] != record["value"]
        or frozen["unit"] != record["unit"]
        or frozen["endpoint"] != "Cmax"
        or frozen["relation"] != "="
        or official["standard_type"].lower() != "cmax"
        or official["standard_relation"] != "="
        or official.get("data_validity_comment")
    ):
        raise ValueError("Official/frozen identity, scalar or endpoint conflict")
    scale = {"nM": Decimal("0.001"), "uM": Decimal("1"), "M": Decimal("1000000")}
    if (
        record["unit"] != "uM"
        or official["standard_units"] not in scale
        or Decimal(str(official["standard_value"])) * scale[official["standard_units"]]
        != Decimal(record["value"])
        or Decimal(record["value"]) <= 0
    ):
        raise ValueError("Canonical Cmax concentration unit/value conflict")
    graphs = [Chem.MolFromSmiles(s) for s in (official["canonical_smiles"], frozen["smiles"])]
    if (
        any(g is None for g in graphs)
        or len({Chem.MolToSmiles(g, isomericSmiles=True) for g in graphs}) != 1
    ):
        raise ValueError("Official full stereochemical graph mismatch")
    context = record["conditional_core_context"]
    encode_context(context)
    description = official["assay_description"].lower()
    if record["source_description"].strip() != official["assay_description"].strip():
        # Source audit may concatenate indexed system context; exact description must be present.
        if official["assay_description"].strip() not in record["source_description"]:
            raise ValueError("Official context description mismatch")
    dose_pattern = (
        rf"(?<![\d.]){float(context['dose_value']):g}(?:\.0+)?\s*"
        rf"{re.escape(context['dose_reported_unit'])}(?![\w/'])"
    )
    if not re.search(dose_pattern, description):
        raise ValueError("Dose quantity/basis absent from official context")
    route_patterns = {
        "oral": r"\b(?:po|oral)\b",
        "intravenous": r"\b(?:iv|intravenous)\b",
        "intraperitoneal": r"\b(?:ip|intraperitoneal)\b",
        "subcutaneous": r"\b(?:sc|subcutaneous)\b",
    }
    if not re.search(route_patterns[context["route"]], description):
        raise ValueError("Route not explicitly supported by official description")
    species_patterns = {
        "human": r"human|patient|subjects|children|adolescents|infants|toddlers",
        "mouse": r"mouse|mice",
        "rat": r"\brats?\b",
    }
    if not (
        re.search(species_patterns[context["species"]], description)
        or {"human": "Homo sapiens", "mouse": "Mus musculus", "rat": "Rattus norvegicus"}[
            context["species"]
        ]
        == official.get("target_organism")
    ):
        raise ValueError("Species lacks explicit official evidence")
    if not re.search(
        {"plasma": r"\bplasma\b", "serum": r"\bserum\b", "whole_blood": r"\bblood\b"}[
            context["matrix"]
        ],
        description,
    ):
        raise ValueError("Matrix lacks explicit official evidence")
    return Chem.MolToSmiles(graphs[0], isomericSmiles=True)


def prepare(audit_path, source_db, feature_root, output, prepared):
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features

    audit_path, output, prepared = map(Path, (audit_path, output, prepared))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest_path = Path(source_db).resolve().parent / "manifest.json"
    if sha(manifest_path) != audit["input_hashes"]["dataset_manifest"]:
        raise ValueError("Frozen dataset manifest changed since source audit")
    rows = [r for r in audit["candidate_records"] if r.get("conditional_core_candidate")]
    declared = audit["summary"]["Cmax"]["minimum_context_source_reported_Cmax"]
    if len(rows) != len({r["record_id"] for r in rows}) or {r["record_id"] for r in rows} != set(
        declared["record_ids"]
    ):
        raise ValueError("Declared candidate IDs differ from audit rows")
    official = {
        str(a["activity_id"]): a for b in audit["official_api_batches"] for a in b["activities"]
    }
    source_db = Path(source_db).resolve()
    proof, accepted, excluded, graph_groups = [], [], [], {}
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in rows:
            try:
                frozen = dict(
                    db.execute(
                        "SELECT * FROM records WHERE record_id=?", (row["record_id"],)
                    ).fetchone()
                )
                activity = official[row["source_entry_id"].split(":")[1]]
                graph = validate_source_record(row, activity, frozen)
                if graph in graph_groups and graph_groups[graph] != row["split_group"]:
                    raise ValueError("Same full graph is split across molecule groups")
                graph_groups[graph] = row["split_group"]
                accepted.append(row)
                proof.append(
                    {
                        "record_id": row["record_id"],
                        "official_activity_id": activity["activity_id"],
                        "official_object_sha256": object_hash(activity),
                        "frozen_row_sha256": object_hash(frozen),
                        "full_stereo_graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                        "split_group": row["split_group"],
                        "value_uM": row["value"],
                        "context": row["conditional_core_context"],
                        "primary_article_table_values_independently_verified": False,
                    }
                )
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({"record_id": row["record_id"], "reason": str(exc)})
    features = portable_features(feature_root)
    registry = json.loads(
        (Path(feature_root) / "representation_methods.json").read_text(encoding="utf-8")
    )
    methods = [
        m for m in registry["methods"] if m["source_kind"] == "structure" and m["independent"]
    ]
    if len(methods) != 10:
        raise ValueError("Require the ten independently registered SMILES methods")
    coverage, valid = {}, np.ones(len(accepted), dtype=bool)
    if accepted:
        for method in methods:
            payload = load_feature_route(accepted, features, method["route"])
            valid &= payload["valid"]
            coverage[method["route"]] = {
                "valid_records": int(payload["valid"].sum()),
                "dimensions": payload["X"].shape[1],
            }
    selected = [r for r, ok in zip(accepted, valid, strict=True) if ok]
    group_count = len({r["split_group"] for r in selected})
    eligible = group_count >= 5 and len({r["value"] for r in selected}) >= 2
    external = context_contract(selected, sha(audit_path))
    task_id = "TASK_" + object_hash({"estimand": ESTIMAND, "contract": external["sha256"]})[:24]
    task = {
        "task_id": task_id,
        "endpoint": "Cmax",
        "task_kind": "regression",
        "status": "eligible" if eligible else "deferred",
        "table_path": f"task_tables/{task_id}.tsv",
        "n_records": len(selected),
        "n_groups": group_count,
        "unit": "uM",
        "target_transform": "log10",
        "inverse_transform": "10**prediction",
        "protocol": "internal_grouped_cv_only_no_independent_holdout",
        "scope_warning": WARNING,
        "fixed_external_covariates": external,
        "source_evidence": [
            p for p in proof if p["record_id"] in {r["record_id"] for r in selected}
        ],
        "task_definition": {
            "endpoint": "Cmax",
            "quantity_unit": "uM",
            "estimand": ESTIMAND,
            "context_schema": SCHEMA,
            "molecule_only_property": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    table_rows = [
        {
            "record_id": r["record_id"],
            "split_group": r["split_group"],
            "target": str(math.log10(float(r["value"]))),
            "value": r["value"],
            "unit": "uM",
            "target_transform": "log10",
            "structure_eligible": "true",
            "sequence_eligible": "false",
            "smiles": r["frozen_source"]["smiles"],
        }
        for r in selected
    ]
    table = prepared / task["table_path"]
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(table_rows[0]) if table_rows else ["record_id"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(table_rows)
    manifest = {
        "schema": "peptide_modeling_all_endpoint_tasks_v1",
        "tasks": [task],
        "source_manifest_sha256": audit["input_hashes"]["dataset_manifest"],
        "parameters": {"requested_inner_folds": 5, "min_groups": 5},
    }
    config = {
        "resources": {"nice": 10},
        "training": {
            "routes": [m["route"] for m in methods] + ["structure_methods_all"],
            "models": [
                "linear",
                "elasticnet",
                "svm",
                "random_forest",
                "extra_trees",
                "histgb",
                "lightgbm",
                "xgboost",
            ],
            "inner_folds": 5,
            "min_rows": 5,
            "min_groups": 5,
            "seed": 42,
            "search_backend": "optuna",
            "search_budget": 128,
            "n_jobs": 1,
        },
        "required_seeds": [42, 2026, 3407],
    }
    report = {
        "schema": SCHEMA,
        "status": "prepared_not_trained" if eligible else "candidate_only",
        "estimand": ESTIMAND,
        "warning": WARNING,
        "source_audit_sha256": sha(audit_path),
        "feature_fingerprint": features["input_fingerprint"],
        "source_proofs": proof,
        "source_exclusions": excluded,
        "coverage": coverage,
        "records": len(selected),
        "molecule_groups": group_count,
        "actual_independent_chemical_methods": 10,
        "context_columns": len(FEATURE_NAMES),
        "context_is_additional_chemical_method": False,
        "known_frequency_intersection_is_regimen_complete": False,
        "dose_basis_counts": dict(
            Counter(
                DOSE_UNITS[r["conditional_core_context"]["dose_reported_unit"]][0] for r in selected
            )
        ),
        "independent_holdout_available": False,
        "no_training_performed": True,
        "feature_invalid_records": [
            r["record_id"] for r, ok in zip(accepted, valid, strict=True) if not ok
        ],
        "task_id": task_id,
        "task_table_sha256": sha(table),
    }
    for path, value in [
        (prepared / "task_manifest.json", manifest),
        (prepared / "config.json", config),
        (prepared / "representation_methods.json", registry),
        (output / "audit.json", report),
    ]:
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output / "REPORT.md").write_text(
        f"# Contextual Cmax preparation\n\n{report['status']}: "
        f"{len(selected)} records / {group_count} compound groups.\n\n"
        f"{WARNING}\n\nTarget: log10 source-reported parent Cmax in uM. "
        "Ten independent full-graph methods, "
        f"each with the same {len(FEATURE_NAMES)} fixed context columns; "
        "covariates add zero chemical methods. "
        "Dose bases retain separate numeric channels; no bodyweight, "
        "bioavailability or salt/free-base conversion. "
        "mg/kg/day is a daily total, not inferred once-daily administration. "
        "Official database objects support the labels; "
        "original article tables were not independently verified.\n\n"
        "Planned protocol: grouped internal five-fold Optuna, eight model families, "
        "128 trials, seeds 42/2026/3407. "
        "No training or remote launch performed by preparation. No independent holdout. "
        "Supported inference entry point requires explicit known context for every chemical row.\n",
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="data/conditional_pk_feasibility_20261114_v1/audit.json")
    parser.add_argument("--source-db", default="data/dataset_reclean_20261113_v1/catalogue.sqlite")
    parser.add_argument(
        "--feature-root", default="data/peptide_modeling_unified_features_20261114_v1"
    )
    parser.add_argument("--output", default="data/conditional_cmax_recovery_20261114_v1")
    parser.add_argument("--prepared", default="data/peptide_modeling_conditional_cmax_followup_v1")
    args = parser.parse_args()
    result = prepare(args.audit, args.source_db, args.feature_root, args.output, args.prepared)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "records", "molecule_groups", "task_id", "source_exclusions")
            }
        )
    )


if __name__ == "__main__":
    main()
