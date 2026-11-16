"""Typed nominal source-reported volume per kg, with explicit missing-dose state.

Vz, Vss and Vz/F are distinct estimands; pooled CV is a stress-test selection score.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_conditional_cmax as base_context

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/vd_per_kg_primary_recovery_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_conditional_volume_followup_20261115_v1"
DB = ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite"
SCHEMA = "conditional_volume_estimand_context_v1"
REPORT_SCHEMA = "conditional_volume_estimand_molecular_diagnostics_v1"
ESTIMAND = (
    "typed_nominal_source_reported_volume_per"
    "_bodyweight_given_estimand_F_basis_and_r"
    "eported_context"
)
SEEDS = (42, 2026, 3407)
ESTIMANDS = ["Vz", "Vss", "Vz_over_F"]
F_BASES = ["not_divided_by_F", "divided_by_unknown_F"]
DOSE_STATUSES = ["reported", "not_reported"]
DOSE_UNITS = base_context.DOSE_UNITS
ONTOLOGY = {
    **base_context.ONTOLOGY,
    "volume_estimand": ESTIMANDS,
    "F_basis": F_BASES,
    "dose_status": DOSE_STATUSES,
}
CONTEXT_KEYS = {
    "volume_estimand",
    "F_basis",
    "species",
    "route",
    "matrix",
    "dose_status",
    "dose_value",
    "dose_reported_unit",
}
FEATURE_NAMES = (
    base_context.FEATURE_NAMES
    + [f"context:volume_estimand_{v}" for v in ESTIMANDS]
    + [f"context:F_basis_{v}" for v in F_BASES]
    + [f"context:dose_status_{v}" for v in DOSE_STATUSES]
)
GROUPS = (
    "morgan",
    "descriptors",
    "graph_modifications",
    "morgan_r3",
    "atompair",
    "torsion",
    "maccs",
    "rdkit_full",
    "rdkit_path",
    "pattern",
    "layered",
    "estate",
    "erg",
)
CP33 = "OLD:PV13:a71fcd2390fa1f00d593a3d7"
WARNING = (
    "Typed nominal source volume-per-bodyweight multitask exploration. Vz, Vss and Vz/F "
    "are different physical estimands and must be explicit query inputs with F basis. "
    "There are five families overall, but only 3/1/1 per estimand; three seeds repeat one"
    " LOGO partition. Overall CV selects the model and is a joint stress test, not "
    "independent or subtype-generalization validation. Six OOF rows have unseen "
    "estimands. One Vss dose is explicitly unknown, with None/None and a missingness "
    "mask; zero channels are placeholders, never an assumed physical dose. Species, "
    "matrix, dose, visit, regimen, analytical method and molecular identity are "
    "confounded. No numerical F conversion, causal dose response, linear-PK assumption, "
    "cross-subtype/species/matrix, or clinical generalization is established."
)
sha = base_context.sha
object_hash = base_context.object_hash
REVIEW_SHA = "bdae9068dd777ca2db03418b02f334e9065f232b28598e358b889ec0bc3f3d3b"
VERIFICATION_SHA = "814ae9e6da5e50ef219a647910b124df214370831a29915868e4839dbb6463c0"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def error_metrics(target, prediction):
    """Stable finite transformed-scale errors, without clipping predictions."""
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite source or diagnostic prediction")
    scale = max(float(np.abs(target).max()), float(np.abs(prediction).max()))
    normalized = prediction / scale - target / scale if scale else np.zeros(len(target))
    result = dict(
        mae=float(np.mean(np.abs(normalized))) * scale,
        rmse=float(np.sqrt(np.mean(normalized * normalized))) * scale,
    )
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Diagnostic error exceeds representable numeric range")
    return result


def encode_context(context):
    if not isinstance(context, dict) or set(context) != CONTEXT_KEYS:
        raise ValueError("Exactly eight typed context inputs are required; no IDs or outcomes")
    estimand, basis = context["volume_estimand"], context["F_basis"]
    if estimand not in ESTIMANDS or basis != (
        "divided_by_unknown_F" if estimand == "Vz_over_F" else "not_divided_by_F"
    ):
        raise ValueError("Volume estimand and F basis are missing or inconsistent")
    for key, values in base_context.ONTOLOGY.items():
        if context[key] not in values:
            raise ValueError("Unknown core condition: " + key)
    status = context["dose_status"]
    if status == "reported":
        vector = base_context.encode_context({k: context[k] for k in base_context.CONTEXT_KEYS})
    elif status == "not_reported":
        if context["dose_value"] is not None or context["dose_reported_unit"] is not None:
            raise ValueError("An unreported dose must retain explicit None value and unit")
        # Zero channels are structural placeholders only; the separate mask declares missingness.
        vector = np.asarray(
            [0.0] * (2 * len(base_context.BASES))
            + [
                float(context[key] == item)
                for key, items in base_context.ONTOLOGY.items()
                for item in items
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError("Explicit reported/not_reported dose status is required")
    suffix = [float(estimand == v) for v in ESTIMANDS]
    suffix += [float(basis == v) for v in F_BASES]
    suffix += [float(status == v) for v in DOSE_STATUSES]
    return np.concatenate((vector, np.asarray(suffix, dtype=np.float32)))


def _context(row):
    value = row["conditional_core_context"]
    return json.loads(value) if isinstance(value, str) else value


def context_contract(records, audit_sha):
    contexts = {r["record_id"]: _context(r) for r in records}
    if len(contexts) != len(records):
        raise ValueError("Duplicate context record identity")
    for context in contexts.values():
        encode_context(context)
    counts = {
        e: dict(
            records=sum(_context(r)["volume_estimand"] == e for r in records),
            molecules=len(
                {r["split_group"] for r in records if _context(r)["volume_estimand"] == e}
            ),
        )
        for e in ESTIMANDS
    }
    value = dict(
        schema=SCHEMA,
        estimand=ESTIMAND,
        feature_names=FEATURE_NAMES,
        ontology=ONTOLOGY,
        dose_units=DOSE_UNITS,
        encoding_dependency_sha256=sha(base_context.__file__),
        source_audit_sha256=audit_sha,
        contexts_by_record=contexts,
        dose_optional=True,
        missing_dose_is_explicit_state=True,
        missing_dose_is_physical_zero=False,
        inference_requires_all_declared_inputs=True,
        no_numeric_F_conversion=True,
        chemical_method_count_increment=0,
        warning=WARNING,
        support_metadata=dict(
            per_estimand=counts,
            independent_holdout=False,
            subtype_generalization_validated=False,
            joint_stress_test_selection_only=True,
        ),
    )
    value = json.loads(json.dumps(value))
    return {**value, "sha256": object_hash(value)}


def validate_contract(contract):
    if not isinstance(contract, dict) or object_hash(
        {k: v for k, v in contract.items() if k != "sha256"}
    ) != contract.get("sha256"):
        raise ValueError("Typed context contract hash mismatch")
    expected = dict(
        schema=SCHEMA,
        estimand=ESTIMAND,
        feature_names=FEATURE_NAMES,
        ontology=ONTOLOGY,
        dose_units=json.loads(json.dumps(DOSE_UNITS)),
        dose_optional=True,
        encoding_dependency_sha256=sha(base_context.__file__),
        missing_dose_is_explicit_state=True,
        missing_dose_is_physical_zero=False,
        inference_requires_all_declared_inputs=True,
        no_numeric_F_conversion=True,
        chemical_method_count_increment=0,
        warning=WARNING,
    )
    if any(contract.get(k) != v for k, v in expected.items()) or not re.fullmatch(
        "[0-9a-f]{64}", contract.get("source_audit_sha256", "")
    ):
        raise ValueError("Unsupported typed volume contract")
    if not isinstance(contract.get("contexts_by_record"), dict):
        raise ValueError("Context record mapping is missing")
    for context in contract["contexts_by_record"].values():
        encode_context(context)
    support = contract.get("support_metadata", {})
    if (
        support.get("independent_holdout") is not False
        or support.get("subtype_generalization_validated") is not False
        or support.get("joint_stress_test_selection_only") is not True
    ):
        raise ValueError("Unsupported validation claim in typed-volume support metadata")
    for estimand in ESTIMANDS:
        item = support.get("per_estimand", {}).get(estimand, {})
        count = sum(
            c["volume_estimand"] == estimand for c in contract["contexts_by_record"].values()
        )
        if (
            item.get("records") != count
            or not isinstance(item.get("molecules"), int)
            or not 0 < item["molecules"] <= count
        ):
            raise ValueError("Typed-volume support counts do not match the source contexts")
    return contract


def inference_support(context, contract):
    validate_contract(contract)
    encode_context(context)
    support = contract["support_metadata"]["per_estimand"][context["volume_estimand"]]
    return dict(
        requested_volume_estimand=context["volume_estimand"],
        F_basis=context["F_basis"],
        dose_status=context["dose_status"],
        source_records=support["records"],
        source_molecules=support["molecules"],
        dose_is_imputed=False,
        subtype_generalization_validated=False,
        independent_holdout=False,
        warning=WARNING,
    )


def _dose_basis(context):
    return (
        DOSE_UNITS[context["dose_reported_unit"]][0]
        if context["dose_status"] == "reported"
        else "not_reported"
    )


def _fold_coverage(rows, contexts, groups, splits):
    result, unsupported_ids = [], set()
    for fold, (train, test) in enumerate(splits):
        unsupported = {}
        for key in ONTOLOGY:
            known = {contexts[rows[i]["record_id"]][key] for i in train}
            unsupported[key] = sorted({contexts[rows[i]["record_id"]][key] for i in test} - known)
        known_bases = {_dose_basis(contexts[rows[i]["record_id"]]) for i in train}
        unsupported["dose_basis"] = sorted(
            {_dose_basis(contexts[rows[i]["record_id"]]) for i in test} - known_bases
        )
        for i in test:
            if contexts[rows[i]["record_id"]]["volume_estimand"] in unsupported["volume_estimand"]:
                unsupported_ids.add(rows[i]["record_id"])
        result.append(
            dict(
                fold=fold,
                train_groups=sorted(set(groups[train])),
                validation_groups=sorted(set(groups[test])),
                unsupported_validation_context=unsupported,
            )
        )
    return result, sorted(unsupported_ids)


def equal_metrics(target, prediction, groups):
    per = [
        error_metrics(np.asarray(target)[groups == group], np.asarray(prediction)[groups == group])
        for group in set(groups)
    ]
    ma = max(p["mae"] for p in per)
    rm = max(p["rmse"] for p in per)
    return dict(
        mae=float(np.mean([p["mae"] / ma for p in per])) * ma if ma else 0.0,
        rmse=float(np.sqrt(np.mean([(p["rmse"] / rm) ** 2 for p in per]))) * rm if rm else 0.0,
    )


def contextual_diagnostics(rows, contract, chemical_X, y, groups, splits, settings, training):
    context = context_matrix(rows, contract)
    designs = dict(
        context_only=context,
        molecule_only=chemical_X,
        molecule_plus_context=np.column_stack((chemical_X, context)),
    )
    ablations = {}
    for name, X in designs.items():
        estimator = training._estimator(
            "linear", "regression", {"regularization": 1.0}, settings, X.shape[1]
        )
        prediction, _ = training._cv_predictions(
            estimator, X, y, splits, "regression", settings["n_jobs"]
        )
        equal = equal_metrics(y, prediction, groups)
        ablations[name] = dict(
            model="fixed_ridge_regularization_1_not_optuna_selected",
            metrics=training._metrics(y, prediction, "regression"),
            compound_equal_weight_mae=equal["mae"],
            compound_equal_weight_rmse=equal["rmse"],
            predictions=[float(v) for v in prediction],
        )
    coverage, unsupported = _fold_coverage(rows, contract["contexts_by_record"], groups, splits)
    return dict(
        warning=WARNING,
        chemical_method_count_increment=0,
        ablations=ablations,
        fold_context_coverage=coverage,
        unsupported_estimand_record_ids=unsupported,
        unknown_dose_record_ids=sorted(
            r["record_id"] for r in rows if _context(r)["dose_status"] == "not_reported"
        ),
        record_ids=[r["record_id"] for r in rows],
        molecular_signal_established=False,
        interpretation=(
            "Joint stress-test selection only; per-estimand unsupported transfer is not validated"
        ),
    )


CEND_IDS = [
    "OLD:PV14:4538e03905e83a621c72c48f",
    "OLD:PV14:326078d47a356a15aaaa1270",
    "OLD:PV14:173df3c69218747ffc2c7f50",
]
DAPTO_IDS = ["NEW:07:400000000051622" + str(i) for i in range(44, 51)]
SCY_IDS = ["NEW:07:400000000033887" + str(i) for i in range(23, 27)]
VOS_IDS = [
    "OLD:PV14:c796b87275c8c64e0559d657",
    "OLD:PV14:0a3f58b2b46223a19d1b9bd5",
    "OLD:PV14:e3b9325ae8e9df35e131345d",
    "OLD:PV14:9ec2dd53e74c09897e40369d",
    "OLD:PV14:1ad688811632dd42ad2b1ce4",
]
RECORD_IDS = CEND_IDS + DAPTO_IDS + SCY_IDS + VOS_IDS + [CP33]
GRAPH_BY_FAMILY = {
    "FAMILY:d4d25384dbd26339988f9522": (
        "bd9ae50f60f988c34dc7fb6cd2a917ac03a46aaf14eab60c2706a4869e44ed3c"
    ),
    "FAMILY:270cfb85bc0a02dc7857b1ec": (
        "4035341513fcf17a52c6766d09b3e38d280a35160cfb4c9c3a08d62a0a6f4691"
    ),
    "FAMILY:5867c4ad4646755d96f8640d": (
        "b4042c8df8d74ace911f19f7af8cd66963bd650cba2fe8dddc7b6ecf4d17e026"
    ),
    "FAMILY:fb8fa129a58f936536a480c9": (
        "fa389dd758f175a5dc9c5cbe7dc80b35457761a9a14eebec44da25b82a400a88"
    ),
    "FAMILY:177d88b010b64c71156adec9": (
        "c8ad0c9df69d99a9cc72a378219d957b552f1394ed9f4a6a93d5cfe2344f0961"
    ),
}
PRIMARY_SHAS = {
    "raw/compound33_PMC12990038.xml": (
        "a09b92d62195599cf30c00d7bbef809f4f1925b95bc5fa5e3e01fb806af1dc5d"
    ),
    "raw/scy635_PMC2812147.html": (
        "e58a09dc29e46d487a385d621ab72f03972e69c4ed0e1c616d55b435c405b450"
    ),
    "raw/daptomycin_PMC152488.html": (
        "b7d05111c60b8095e8da33550feb60653804e678ff6e824fba61eabe2b3abaae"
    ),
    "raw/cend1_ijms24065700.xml": (
        "3c70ffc3084ef0354dd44ffe790a11b187d5d4a0f9828d5ece42ae1b2c4544d8"
    ),
    "raw/vosoritide_01059.xml": "ec64fbbf6af4f8e43871613f077e1e674e6372f995a7868833fc88bd2f8b3f6e",
}
DATASET_MANIFEST_SHA = "baf3f83f809a2038acb5e4c22239ee96b69826d664dd6aa44600716f5b87b418"
rows = table


def write_table(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(values)


def verify_sources(source=SOURCE):
    source = Path(source)
    review = source / "record_review.json"
    verification = source / "verification.json"
    if sha(review) != REVIEW_SHA or sha(verification) != VERIFICATION_SHA:
        raise ValueError("Pinned volume primary review changed")
    pins = {review: REVIEW_SHA, verification: VERIFICATION_SHA}
    for name, expected in read(verification)["files"].items():
        rel = Path(name.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe primary source path")
        p = source / rel
        if p.is_symlink() or sha(p) != expected:
            raise ValueError("Primary source artifact changed: " + name)
        pins[p] = expected
    for name, expected in PRIMARY_SHAS.items():
        p = source / name
        if sha(p) != expected:
            raise ValueError("Independent primary anchor changed")
        pins[p] = expected
    if sha(DB.parent / "manifest.json") != DATASET_MANIFEST_SHA:
        raise ValueError("Frozen dataset manifest changed")
    pins[DB.parent / "manifest.json"] = DATASET_MANIFEST_SHA
    return read(review), pins


def validate_cohort(values, *, verify_chemistry=True):
    if verify_chemistry:
        from rdkit import Chem
    if len(values) != 20 or {r["record_id"] for r in values} != set(RECORD_IDS):
        raise ValueError("Exactly the twenty qualified original observations are required")
    graphs = {}
    families = set()
    counts = Counter()
    for row in values:
        context = _context(row)
        encode_context(context)
        if any(row[k] != context[k] for k in ("volume_estimand", "F_basis", "dose_status")):
            raise ValueError("Typed source fields differ from the explicit context")
        canonical = row["canonical_isomeric_smiles"]
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if (
            row["smiles"] != canonical
            or row["structure_key"] != "CANON_SMILES:" + digest
            or GRAPH_BY_FAMILY.get(row["split_group"]) != digest
        ):
            raise ValueError("Exact canonical graph and known-family binding changed")
        if verify_chemistry:
            mol = Chem.MolFromSmiles(canonical)
            if (
                mol is None
                or any(c == "?" for _, c in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                or Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) != canonical
            ):
                raise ValueError("Assigned-stereo complete reference graph is required")
        if graphs.setdefault(canonical, row["split_group"]) != row["split_group"]:
            raise ValueError("A graph crosses families")
        families.add(row["split_group"])
        counts[context["volume_estimand"]] += 1
        if (
            row["unit"] != "L/kg"
            or row["relation"] != "="
            or Decimal(row["value"]) <= 0
            or not math.isclose(
                float(row["target"]), math.log10(float(row["value"])), abs_tol=1e-14
            )
        ):
            raise ValueError("Exact normalized volume label or transform changed")
    if len(graphs) != 5 or len(families) != 5 or counts != {"Vz": 14, "Vz_over_F": 5, "Vss": 1}:
        raise ValueError("Typed dataset must have five families with14/5/1 records")
    unknown = [r for r in values if _context(r)["dose_status"] == "not_reported"]
    if len(unknown) != 1 or unknown[0]["record_id"] != CP33:
        raise ValueError("Only the compound33 Table9 dose is explicitly unknown")
    if any(
        len({r["split_group"] for r in values if _context(r)["volume_estimand"] == e}) != n
        for e, n in {"Vz": 3, "Vz_over_F": 1, "Vss": 1}.items()
    ):
        raise ValueError("Per-estimand molecular support changed")
    return values


def source_rows(source=SOURCE):
    from rdkit import Chem

    reviewed, pins = verify_sources(source)
    by_id = {r["record_id"]: r for r in reviewed}
    accepted = []
    proofs = []
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for record_id in RECORD_IDS:
            r = by_id[record_id]
            q = r["source_recovery"]
            f = dict(db.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone())
            if any(f.get(k) != v for k, v in r["frozen_source"].items()):
                raise ValueError("Frozen volume source changed")
            if (
                f["endpoint"] != "Vd_per_kg"
                or f["lane"] != "strict_numeric"
                or f["unit"] != "L/kg"
                or f["relation"] != "="
            ):
                raise ValueError("Candidate is not an exact frozen per-bodyweight volume label")
            context0 = q["context"]
            original_taxon = context0["species"]
            primary_taxon = original_taxon
            correction = "none"
            if record_id in CEND_IDS:
                estimand = "Vz"
                dose = 3.2
                unit = "mg/kg"
                species = "human"
                route = "intravenous"
                matrix = "plasma"
                primary_scalar = {CEND_IDS[0]: "220.9", CEND_IDS[1]: "269.2", CEND_IDS[2]: "262.6"}[
                    record_id
                ]
                primary_unit = "mL/kg"
                factor = "0.001"
            elif record_id in DAPTO_IDS:
                if not q.get("strict_primary_terminal_definition") or q["source_files"] != [
                    "raw/daptomycin_PMC152488.html"
                ]:
                    raise ValueError(
                        "Generic V or secondary terminal label cannot replace direct primary Vz"
                    )
                estimand = "Vz"
                dose = context0["dose_value"]
                unit = "mg/kg"
                species = "human"
                route = "intravenous"
                matrix = "plasma"
                primary_scalar = re.match(
                    r"[+]?\d+(?:\.\d+)?", q["proof"]["source_scalar_with_dispersion"]
                ).group()
                primary_unit = "L/kg"
                factor = "1"
            elif record_id in SCY_IDS:
                if not q.get("strict_primary_terminal_definition"):
                    raise ValueError("SCY Vz definition missing")
                estimand = "Vz"
                dose = context0["dose_value"]
                unit = "mg/kg"
                route = "intravenous"
                matrix = "whole_blood"
                species = "rat" if record_id in SCY_IDS[:2] else "monkey"
                if species == "monkey":
                    if original_taxon != "Macaca mulatta":
                        raise ValueError("Expected preserved SCY raw taxonomy conflict")
                    primary_taxon = "cynomolgus monkey (Macaca fascicularis)"
                    correction = (
                        "primary_cynomolgus_overrides_raw_Macaca_mulatta_for_derived_context_only"
                    )
                primary_scalar = r["value"]
                primary_unit = "L/kg"
                factor = "1"
            elif record_id in VOS_IDS:
                estimand = "Vz_over_F"
                dose = 30 if record_id == VOS_IDS[-1] else 15
                unit = "ug/kg"
                species = "human"
                route = "subcutaneous"
                matrix = "plasma"
                primary_scalar = {
                    VOS_IDS[0]: "2880",
                    VOS_IDS[1]: "2560",
                    VOS_IDS[2]: "2910",
                    VOS_IDS[3]: "3020",
                    VOS_IDS[4]: "1830",
                }[record_id]
                primary_unit = "mL/kg"
                factor = "0.001"
                if q["quantity"] != "Vz_over_F" or str(dose) not in context0["dose"]:
                    raise ValueError("Apparent volume primary context changed")
            else:
                if record_id != CP33 or q["quantity"] != "Vss" or q["table_anchor"] != "Table9":
                    raise ValueError("Unexpected steady-state volume candidate")
                estimand = "Vss"
                dose = None
                unit = None
                species = "mouse"
                route = "intravenous"
                matrix = "plasma"
                primary_scalar = "1.1"
                primary_unit = "L/kg"
                factor = "1"
            if q["quantity"] != estimand or Decimal(primary_scalar) * Decimal(factor) != Decimal(
                f["value"]
            ):
                raise ValueError("Primary volume type/value/unit conversion mismatch")
            context = dict(
                volume_estimand=estimand,
                F_basis="divided_by_unknown_F" if estimand == "Vz_over_F" else "not_divided_by_F",
                species=species,
                route=route,
                matrix=matrix,
                dose_status="not_reported" if dose is None else "reported",
                dose_value=dose,
                dose_reported_unit=unit,
            )
            encode_context(context)
            mol = Chem.MolFromSmiles(f["smiles"])
            if mol is None:
                raise ValueError("Invalid full graph")
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            key = "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest()
            descriptive = dict(
                core=context,
                source_recovery_context_original=context0,
                source_original_taxon=original_taxon,
                primary_taxon=primary_taxon,
                primary_taxon_correction=correction,
                primary_source_files=q["source_files"],
                protocol_or_visit=q.get("table_anchor", q.get("proof", {}).get("table_anchor")),
                missing_dose_is_not_physical_zero=True,
                compound33_dose_policy=(
                    (
                        "Table9/sec4.10 administered dose not rep"
                        "orted; reject generic sec4.9 2mg/kg, com"
                        "pound2 SI5mg/kg and formulation2mg/mL bo"
                        "rrowing"
                    )
                    if record_id == CP33
                    else "not_applicable"
                ),
            )
            row = dict(
                record_id=record_id,
                split_group=r["split_group"],
                target=repr(math.log10(float(f["value"]))),
                value=f["value"],
                unit="L/kg",
                relation="=",
                target_transform="log10",
                structure_eligible="true",
                sequence_eligible="false",
                smiles=canonical,
                canonical_isomeric_smiles=canonical,
                structure_key=key,
                canonical_identity_group=key,
                volume_estimand=estimand,
                F_basis=context["F_basis"],
                dose_status=context["dose_status"],
                conditional_core_context=context,
                raw_scalar=f["raw_value"],
                raw_unit=f["raw_unit"],
                raw_relation=f["raw_relation"],
                raw_endpoint=f["endpoint_raw"],
                frozen_endpoint=f["endpoint"],
                primary_scalar=primary_scalar,
                primary_unit=primary_unit,
                primary_normalization_factor=factor,
                source_paper="|".join(q["source_files"]),
                source_locator=q.get("table_anchor", q.get("proof", {}).get("table_anchor", "")),
                source_original_taxon=original_taxon,
                primary_taxon=primary_taxon,
                primary_taxon_correction=correction,
                reported_context=json.dumps(descriptive, sort_keys=True, separators=(",", ":")),
                graph_identity_tier=(
                    "pinned_source_complete_reference_graph_not_sample_salt_or_microstate_proof"
                ),
            )
            accepted.append(row)
            proofs.append(
                dict(
                    record_id=record_id,
                    frozen_row_sha256=object_hash(f),
                    raw_source_fields=r["frozen_source"],
                    primary_source_files={
                        name: pins[Path(source) / name] for name in q["source_files"]
                    },
                    source_recovery_original=q,
                    derived_context=context,
                    primary_scalar=primary_scalar,
                    primary_unit=primary_unit,
                    normalization_factor=factor,
                    normalized_value_L_per_kg=f["value"],
                    canonical_graph_sha256=key.split(":", 1)[1],
                    source_original_taxon=original_taxon,
                    primary_taxon=primary_taxon,
                    primary_taxon_correction=correction,
                    compound33_administered_dose_not_reported=record_id == CP33,
                )
            )
    validate_cohort(accepted)
    return accepted, proofs, pins, reviewed


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
    implementation = model_contract.get("external_context", {})
    if (
        implementation.get("implementation_sha256") != sha(__file__)
        or implementation.get("schema") != SCHEMA
        or implementation.get("sha256") != contract["sha256"]
    ):
        raise ValueError("Typed volume inference implementation/contract hash differs")
    X = np.asarray(chemical_X, dtype=np.float32)
    if X.ndim != 2 or len(contexts) != len(X) or not contexts:
        raise ValueError("Explicit known context is required per chemical row")
    if chemical_names + contract["feature_names"] != model_contract["feature_names"]:
        raise ValueError("Chemical or context feature schema differs from fitted model")
    if any(str(name).startswith("context:") for name in chemical_names):
        raise ValueError("Chemical inputs must not contain outcome/context columns")
    return model.predict(np.column_stack((X, np.vstack([encode_context(c) for c in contexts]))))


def split_prechecks(values):
    from peptide_modeling_internal_cv import internal_splits
    from peptide_modeling_train import validate_rows

    y, groups = validate_rows(values, "regression")
    result, partitions = [], set()
    for seed in SEEDS:
        splits = internal_splits(y, groups, "regression", {"seed": seed, "inner_folds": 5})
        folds = []
        for fold, (train, test) in enumerate(splits):
            if (
                set(groups[train]) & set(groups[test])
                or len(set(groups[train])) != 4
                or len(set(groups[test])) != 1
                or len(set(y[train])) < 2
            ):
                raise ValueError("Invalid independent five-fold training partition")
            folds.append(
                dict(
                    fold=fold,
                    train_record_ids=[values[i]["record_id"] for i in train],
                    validation_record_ids=[values[i]["record_id"] for i in test],
                    train_groups=sorted(set(groups[train])),
                    validation_groups=sorted(set(groups[test])),
                )
            )
        partitions.add(tuple(sorted(tuple(sorted(f["validation_record_ids"])) for f in folds)))
        result.append(dict(seed=seed, actual_folds=5, folds=folds))
    if len(partitions) != 1:
        raise ValueError("Five groups imply one LOGO molecular partition")
    return dict(seeds=result, distinct_molecular_partitions=1, independent_holdout=False)


def write_diagnostics(config_path, journal=None):
    from peptide_modeling_internal_cv import verify_internal_task
    from peptide_modeling_pipeline import load_rows, portable_features
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "internal_campaign_status.json")
    if len(journal["rounds"]) != 3 or {e["seed"] for e in journal["rounds"]} != set(SEEDS):
        raise ValueError("Molecular reporting requires exactly the three required seeds")
    reports, observed_tasks = [], set()
    for entry in journal["rounds"]:
        if entry["status"] != "completed":
            raise ValueError("Molecular reporting requires completed internal seed work")
        run = Path(entry["run_dir"])
        tasks = read(run / "task_manifest.json")["tasks"]
        if len(tasks) != 1 or tasks[0]["endpoint"] != "Vd_per_kg":
            raise ValueError("Unexpected task in molecular reporting")
        task = tasks[0]
        validate_contract(task["fixed_external_covariates"])
        expected = (
            "TASK_"
            + object_hash(
                {
                    "estimand": ESTIMAND,
                    "external_context": task["fixed_external_covariates"]["sha256"],
                }
            )[:24]
        )
        if task["task_id"] != expected or set(config["task_ids"]) != {expected}:
            raise ValueError("Typed volume task identity changed")
        observed_tasks.add(expected)
        source_rows = load_rows(run / task["table_path"])
        result_path = run / "results/internal_cv" / (expected + ".json")
        result = read(result_path)
        verify_internal_task(task, source_rows, portable_features(run), run, result)
        oof_path = Path(result["artifacts"]["internal_oof"])
        if sha(oof_path) != result["artifact_hashes"][str(oof_path)]["sha256"]:
            raise ValueError("OOF artifact changed before diagnostics")
        report = molecule_diagnostics(source_rows, load_rows(oof_path))
        for metric in ("mae", "rmse"):
            if not math.isclose(
                report["record_weighted"][metric],
                result["internal_cv_selection_metrics"][metric],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError("Record-weighted diagnostics disagree with saved OOF metrics")
        report.update(
            seed=entry["seed"],
            result_sha256=sha(result_path),
            oof_sha256=sha(oof_path),
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=sha(run / task["table_path"]),
        )
        reports.append(report)
    if len(observed_tasks) != 1:
        raise ValueError("Required seeds use different scientific tasks")
    payload = dict(
        schema=REPORT_SCHEMA,
        status="verified",
        task_id=next(iter(observed_tasks)),
        seeds=list(SEEDS),
        reports=reports,
        model_selection_repeated=False,
        independent_holdout=False,
        warning=WARNING,
    )
    destination = root / "vd_per_kg_molecular_diagnostics.json"
    atomic_json(destination, payload)
    durable.mkdir(parents=True, exist_ok=True)
    target = durable / destination.name
    shutil.copy2(destination, target)
    if sha(destination) != sha(target):
        raise ValueError("Durable molecular report differs")
    receipt = dict(
        status="verified",
        path=str(destination),
        sha256=sha(destination),
        durable_path=str(target),
        durable_sha256=sha(target),
        implementation_sha256=sha(__file__),
    )
    journal["molecular_reporting"] = receipt
    if all(e["status"] == "completed" for e in journal["rounds"]):
        journal["status"] = "completed"
    atomic_json(root / "internal_campaign_status.json", journal)
    atomic_json(durable / "internal_campaign_status.json", journal)
    return receipt


def run_followup(config_path, diagnostics_only=False):
    from peptide_modeling_internal_cv import run_internal_followup
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = (
        read(root / "internal_campaign_status.json")
        if diagnostics_only
        else run_internal_followup(config_path)
    )
    try:
        return write_diagnostics(config_path, journal)
    except BaseException as exc:
        if journal.get("status") == "completed":
            journal["status"] = "requirement_incomplete"
        journal["molecular_reporting"] = dict(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(root / "internal_campaign_status.json", journal)
        atomic_json(durable / "internal_campaign_status.json", journal)
        raise


def slice_features(values, source, parent, output, *, expected_parent_sha):
    """Copy only five frozen rows; no descriptors or embeddings are computed."""
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    parent, output = Path(parent), Path(output)
    pointer = parent / "features/manifest.json"
    if sha(pointer) != expected_parent_sha:
        raise ValueError("Parent pointer differs from independently pinned source evidence")
    base = read(pointer)
    _verify_artifacts(base)
    before = sha(pointer)
    root = Path(base["cache_dir"])
    registry = copy.deepcopy(base["representation_methods"])
    methods = [
        item
        for item in registry["methods"]
        if item["source_kind"] == "structure" and item["independent"] is True
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten independent structure algorithms are required")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    if any(item["key_kind"] != "canonical_identity_group" for item in groups.values()):
        raise ValueError("A sequence representation entered a structure-only bank")
    old_graphs = {item["key"]: item for item in rows(root / "structure_source_smiles.tsv")}
    graph_index = {
        item["key"]: int(item["row_idx"]) for item in rows(root / groups["morgan"]["index"])
    }
    graph_map = {}
    for row in values:
        key = row["canonical_identity_group"]
        if key not in graph_index or (
            old_graphs[key]["canonical_isomeric_smiles"] != row["canonical_isomeric_smiles"]
        ):
            raise ValueError("A source graph does not exactly match a frozen parent graph")
        graph_map[key] = old_graphs[key]["canonical_isomeric_smiles"]
    if len(graph_map) != 5:
        raise ValueError("Exactly five old complete graphs must be reused")
    keys = sorted(graph_map)
    configuration = dict(
        version="typed_volume_frozen_graph_slices_v1",
        parent_sha256=before,
        parent_fingerprint=base["input_fingerprint"],
        source_review_sha256=REVIEW_SHA,
        source_admission_sha256=sha(Path(output) / "source_admission.json"),
        implementation_sha256=sha(__file__),
        encoding_dependency_sha256=sha(base_context.__file__),
        selected_graphs=keys,
        record_ids=[r["record_id"] for r in values],
        feature_computations=0,
        sequence_or_lossy_projection=False,
        typed_estimand_multitask=True,
        dose_optional=True,
    )
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    indices = [graph_index[key] for key in keys]
    for name, meta in groups.items():
        matrix = np.load(root / meta["matrix"], mmap_mode="r")
        valid = np.load(root / meta["valid"], mmap_mode="r")
        selected, masks = matrix[indices].copy(), valid[indices].copy()
        if not masks.all() or np.isinf(selected).any() or np.isnan(selected).all(axis=1).any():
            raise ValueError("Frozen graph method has an invalid selected row: " + name)
        if np.isnan(selected).any() and (
            not meta.get("allow_partial_missing", False)
            or (np.isnan(selected).mean(axis=1) > meta["max_missing_fraction"]).any()
        ):
            raise ValueError("Feature partial missingness differs from the parent contract")
        if not np.array_equal(selected.view(np.uint8), matrix[indices].copy().view(np.uint8)):
            raise ValueError("Copied graph vector changed byte payload")
        for filename, array in ((meta["matrix"], selected), (meta["valid"], masks)):
            (cache / filename).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache / filename, array, allow_pickle=False)
        meta.update(shape=[5, meta["shape"][1]], index="structure_bank_index.tsv")
    write_table(
        cache / "structure_bank_index.tsv", [dict(key=key, row_idx=i) for i, key in enumerate(keys)]
    )
    write_table(
        cache / "structure_source_smiles.tsv",
        [dict(key=key, canonical_isomeric_smiles=graph_map[key]) for key in keys],
    )
    write_table(
        cache / "record_bank_index.tsv",
        [
            dict(
                record_id=row["record_id"],
                structure_key=row["canonical_identity_group"],
                sequence_key="",
            )
            for row in values
        ],
    )
    registry = {
        **registry,
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": "conditional_typed_volume_reference_graphs_only",
    }
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", configuration)
    write(
        cache / "copied_vector_audit.json",
        dict(
            reused_graphs=5,
            new_graphs=0,
            byte_identical=True,
            parent_artifacts_checked=len(base["artifact_hashes"]),
        ),
    )
    routes = {item["route"]: item["group_names"] for item in methods}
    routes["structure_methods_all"] = list(GROUPS)
    manifest = dict(
        schema="conditional_typed_volume_structure_slices_v1",
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=20,
        structure_keys=5,
        groups=groups,
        routes=routes,
        representation_methods=registry,
        coverage={name: dict(representations=5, valid=5, invalid=0) for name in groups},
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    if len(manifest["artifact_hashes"]) != 33:
        raise ValueError("Unexpected structure-bank artifact count")
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for method in methods:
        payload = load_feature_route(values, manifest, method["route"])
        if not payload["valid"].all() or payload["X"].shape != (20, method["dimensions"]):
            raise ValueError("A structural method did not cover every admitted source row")
    _verify_artifacts(base)
    if sha(pointer) != before:
        raise ValueError("Parent feature pointer changed")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        dict(
            status="verified",
            records=20,
            graphs=5,
            methods=10,
            groups=13,
            dimensions=sum(g["shape"][1] for g in groups.values()),
            output_hashes=33,
            parent_hashes_unchanged=len(base["artifact_hashes"]),
            parent_pointer_unchanged=True,
            copied_vectors_byte_identical=True,
            new_graphs=0,
            feature_computations=0,
            new_observation_aliases=0,
        ),
    )
    return manifest


def molecule_diagnostics(source_rows, predictions):
    validate_cohort(source_rows, verify_chemistry=False)
    source = {r["record_id"]: r for r in source_rows}
    if (
        len(predictions) != 20
        or len({r["record_id"] for r in predictions}) != 20
        or {r["record_id"] for r in predictions} != set(source)
    ):
        raise ValueError("OOF record identity differs from the twenty source observations")
    grouped, group_folds = defaultdict(list), {}
    for row in predictions:
        original = source[row["record_id"]]
        group, fold = row["split_group"], int(row["fold"])
        target, prediction = float(row["target"]), float(row["prediction"])
        if (
            group != original["split_group"]
            or target != float(original["target"])
            or not 0 <= fold < 5
            or not math.isfinite(prediction)
        ):
            raise ValueError("OOF source target, group, fold or prediction changed")
        if group_folds.setdefault(group, fold) != fold:
            raise ValueError("A molecule crosses OOF folds")
        grouped[group].append(
            dict(
                record_id=row["record_id"],
                target=target,
                prediction=prediction,
                source_paper=original["source_paper"],
                reported_context=original["reported_context"],
                graph_identity_tier=original["graph_identity_tier"],
                volume_estimand=original["volume_estimand"],
                F_basis=original["F_basis"],
                dose_status=original["dose_status"],
                source_original_taxon=original["source_original_taxon"],
                primary_taxon=original["primary_taxon"],
                primary_taxon_correction=original["primary_taxon_correction"],
            )
        )
    if len(grouped) != 5 or set(group_folds.values()) != set(range(5)):
        raise ValueError("OOF must contain five actual held-out molecular groups")
    per_group = []
    for group, values in sorted(grouped.items()):
        y = [r["target"] for r in values]
        p = [r["prediction"] for r in values]
        per_group.append(
            dict(
                split_group=group,
                observations=len(values),
                fold=group_folds[group],
                **error_metrics(y, p),
                target_min=min(y),
                target_max=max(y),
                predictions=values,
            )
        )
    mae_scale = max(g["mae"] for g in per_group)
    rmse_scale = max(g["rmse"] for g in per_group)
    mae = (
        float(np.mean([g["mae"] / mae_scale for g in per_group])) * mae_scale if mae_scale else 0.0
    )
    rmse = (
        float(np.sqrt(np.mean([(g["rmse"] / rmse_scale) ** 2 for g in per_group]))) * rmse_scale
        if rmse_scale
        else 0.0
    )
    report = dict(
        status="verified",
        records=20,
        molecules=5,
        metrics_space="log10_L_per_kg",
        record_weighted=error_metrics(
            [r["target"] for r in predictions], [r["prediction"] for r in predictions]
        ),
        molecule_equal=dict(mae=mae, rmse=rmse),
        per_molecule=per_group,
        weighting="each molecular family contributes total weight one",
        selection_criterion_changed=False,
        extra_model_fit=False,
        independent_holdout=False,
        distinct_molecular_partitions=1,
        warning=WARNING,
    )
    order = {row["record_id"]: row for row in predictions}
    targets = np.asarray([float(row["target"]) for row in source_rows])
    predicted = np.asarray([float(order[row["record_id"]]["prediction"]) for row in source_rows])
    groups = np.asarray([row["split_group"] for row in source_rows])
    folds = np.asarray([int(order[row["record_id"]]["fold"]) for row in source_rows])
    splits = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)) for fold in range(5)]
    contexts = {row["record_id"]: _context(row) for row in source_rows}
    coverage, unsupported = _fold_coverage(source_rows, contexts, groups, splits)
    by_input = {}
    for row, prediction in zip(source_rows, predicted, strict=True):
        key = (row["structure_key"], object_hash(contexts[row["record_id"]]))
        if key in by_input and not math.isclose(
            float(prediction), by_input[key], rel_tol=1e-7, abs_tol=1e-8
        ):
            raise ValueError(
                "Identical graph and typed context received inconsistent OOF predictions"
            )
        by_input[key] = float(prediction)
    per = []
    for estimand in ESTIMANDS:
        indices = np.asarray(
            [
                i
                for i, row in enumerate(source_rows)
                if contexts[row["record_id"]]["volume_estimand"] == estimand
            ],
            dtype=int,
        )
        ids = sorted(source_rows[i]["record_id"] for i in indices)
        per.append(
            dict(
                volume_estimand=estimand,
                records=len(indices),
                molecules=len(set(groups[indices])),
                record_weighted=error_metrics(targets[indices], predicted[indices]),
                molecule_equal=equal_metrics(targets[indices], predicted[indices], groups[indices]),
                source_record_ids=ids,
                validation_scope="unsupported_estimand_transfer_only"
                if set(ids) & set(unsupported)
                else "internal_selection_cv_not_independent_validation",
            )
        )
    report.update(
        per_estimand=per,
        unsupported_estimand_record_ids=unsupported,
        unknown_dose_record_ids=sorted(
            row["record_id"]
            for row in source_rows
            if contexts[row["record_id"]]["dose_status"] == "not_reported"
        ),
        fold_context_coverage=coverage,
        typed_estimand_multitask=True,
        dose_optional=True,
        overall_score_interpretation="joint_stress_test_model_selection_not_subtype_validation",
    )
    return report


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if (output / "task_manifest.json").exists():
        raise FileExistsError("Typed volume preparation already exists; preserveit")
    values, proofs, pins, all_review = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    admission = dict(
        schema=SCHEMA,
        status="qualified_typed_multitask_not_trained",
        source_proofs=proofs,
        source_pins={str(p): h for p, h in pins.items()},
        records=20,
        molecular_groups=5,
        per_estimand_records={"Vz": 14, "Vz_over_F": 5, "Vss": 1},
        per_estimand_molecular_groups={"Vz": 3, "Vz_over_F": 1, "Vss": 1},
        same_quantity_five_family_Vz_proven=False,
        typed_estimand_multitask=True,
        dose_optional=True,
        unknown_dose_record_ids=[CP33],
        unknown_dose_is_imputed_physical_value=False,
        no_numerical_F_conversion=True,
        frozen_rows_modified=False,
        warning=WARNING,
        excluded_records=[
            dict(
                record_id=r["record_id"],
                reason="outside_fixed20reviewedtypedcohort",
                original_status=r.get("source_recovery", {}).get("status"),
            )
            for r in all_review
            if r["record_id"] not in RECORD_IDS
        ],
    )
    write(output / "source_admission.json", admission)
    external = context_contract(values, sha(output / "source_admission.json"))
    task_id = (
        "TASK_" + object_hash({"estimand": ESTIMAND, "external_context": external["sha256"]})[:24]
    )
    table_path = "task_tables/" + task_id + ".tsv"
    exported = [
        {
            k: json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, dict) else v
            for k, v in r.items()
        }
        for r in values
    ]
    write_table(output / table_path, exported)
    definition = dict(
        endpoint="Vd_per_kg",
        estimand=ESTIMAND,
        quantity_unit="L/kg",
        typed_estimand_multitask=True,
        dose_optional=True,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=20,
        volume_estimands=ESTIMANDS,
        F_bases=F_BASES,
        per_estimand_records={"Vz": 14, "Vz_over_F": 5, "Vss": 1},
        per_estimand_molecular_groups={"Vz": 3, "Vz_over_F": 1, "Vss": 1},
        unknown_dose_record_ids=[CP33],
        dose_missing_is_not_physical_zero=True,
        no_numeric_F_conversion=True,
        subtype_generalization_validated=False,
        independent_holdout=False,
        pooled_CV_scope="joint_stress_test_selection_only",
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=task_id,
        endpoint="Vd_per_kg",
        task_kind="regression",
        status="eligible",
        table_path=table_path,
        n_records=20,
        n_groups=5,
        unit="L/kg",
        target_transform="log10",
        inverse_transform="pow10",
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        requested_inner_folds=5,
        fixed_external_covariates=external,
        scientific_definition=definition,
        task_definition=copy.deepcopy(definition),
        scientific_warning=WARNING,
    )
    manifest = dict(schema=SCHEMA, tasks=[task], models_fitted=0)
    write(output / "task_manifest.json", manifest)
    write(output / "targeted_task_manifest.json", manifest)
    write(output / "fivefold_prechecks.json", split_prechecks(values))
    feature = slice_features(
        values,
        source,
        parent,
        output,
        expected_parent_sha=sha(Path(parent) / "features/manifest.json"),
    )
    config = dict(
        dataset="data/dataset_reclean_20261113_v1/catalogue.sqlite",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        mandatory_postprocessing=(
            "selected_record_molecule_equal_per_estimand_OOF_support_diagnostics"
        ),
        training=dict(
            routes=list(feature["routes"]),
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
            inner_folds=5,
            min_rows=5,
            min_groups=5,
            seed=42,
            search_backend="optuna",
            search_budget=128,
            n_jobs=1,
            refit_full=False,
            svm_max_features=16384,
            histgb_max_features=16384,
            elasticnet_max_features=16384,
        ),
    )
    write(output / "recommended_training_config.json", config)
    write(output / "config.json", config)
    snapshots = output / "task_tables/source_evidence"
    snapshots.mkdir(parents=True, exist_ok=True)
    for name in (
        "record_review.json",
        "source_definition_cards.json",
        "verification.json",
        "summary.json",
    ):
        shutil.copy2(Path(source) / name, snapshots / name)
    for name in PRIMARY_SHAS:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source) / name, target)
    write(
        snapshots / "independent_corrections.json",
        dict(
            compound33_dose_status="not_reported",
            compound33_dose_value=None,
            compound33_dose_unit=None,
            rejected_inferred_doses=["genericmouseTable8_sec4.9_2mgkg", "compound2SI_5mgkg"],
            compound33_formulation_2mg_per_mL_is_not_administered_dose=True,
            SCY_primary_cynomolgus_over_raw_Macaca_mulatta=[
                r["record_id"] for r in values if r["primary_taxon_correction"] != "none"
            ],
        ),
    )
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("Protected volume source changed duringpreparation")
    report = dict(
        status="prepared_not_trained",
        task_id=task_id,
        records=20,
        molecules=5,
        chemical_methods=10,
        context_columns=len(FEATURE_NAMES),
        chemical_method_count_increment=0,
        feature_fingerprint=feature["input_fingerprint"],
        typed_estimand_multitask=True,
        dose_optional=True,
        per_estimand_records={"Vz": 14, "Vz_over_F": 5, "Vss": 1},
        per_estimand_molecules={"Vz": 3, "Vz_over_F": 1, "Vss": 1},
        unknown_dose_record_ids=[CP33],
        distinct_molecular_partitions=1,
        models_fitted=0,
        remote_operations=False,
        warning=WARNING,
    )
    write(output / "preparation_validation.json", report)
    (output / "REPORT.md").write_text(
        (
            "# Typed nominal source-reported volume per kg\n\nThe cohort contains 20 observ"
            "ations and 5 molecular families: Vz 14 rows / 3 families, Vz/F 5 rows / 1 fa"
            "mily, and Vss 1 row / 1 family. These are distinct typed targets, not a unif"
            "orm Vz dataset or three separately validated five-family models. Every query"
            " requires an estimand, F basis, known species/route/matrix, and an explicit "
            "dose-known or dose-missing state.\n\n"
        )
        + WARNING
        + (
            "\n\nCompound 33 Table 9 has no reported administered dose: value and unit stay"
            " None with a missingness mask. The generic mouse 2 mg/kg protocol, compound "
            "2 SI 5 mg/kg, and formulation concentration 2 mg/mL are not borrowed. The tw"
            "o SCY original Macaca mulatta fields remain intact; derived context records "
            "the primary cynomolgus correction and coarse monkey encoding. CEND-1 and vos"
            "oritide primary mL/kg values convert by exactly 0.001 to L/kg. Vz/F is never"
            " converted to Vz.\n\nFive exact complete graphs and 20 aliases are copied byte"
            "-for-byte, with zero feature extraction. Ten chemical methods plus fusion, e"
            "ight model families, Optuna 128 trials, and seeds 42/2026/3407 use strict in"
            "ternal five-fold CV with one LOGO partition. Mandatory selected-OOF record-w"
            "eighted, molecule-equal and per-estimand diagnostics, three fixed ridge abla"
            "tions, and fold condition coverage retain all six unseen-estimand rows and t"
            "he one unknown-dose case. There is no independent holdout.\n"
        ),
        encoding="utf-8",
    )
    write(
        output / "delivery_receipt.json",
        dict(
            schema=SCHEMA,
            status="prepared_not_trained",
            task_id=task_id,
            files={
                p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()
            },
        ),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.run_followup or args.diagnostics_only:
        if args.config is None:
            parser.error("--config isrequiredforfollowupreporting")
        result = run_followup(args.config, diagnostics_only=args.diagnostics_only)
    else:
        result = prepare(args.source, args.parent, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
