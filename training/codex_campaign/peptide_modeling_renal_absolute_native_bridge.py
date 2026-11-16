"""Bounded, typed source-annotation exploration of absolute-unit urinary-parent CLR.

The frozen observations retain estimator, normalization and mean-kind differences.
Preparation slices existing complete-graph features; reporting never fits/unpickles.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import struct
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_mhc10_native_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/renal_absolute_native_admission_review_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_renal_absolute_native_followup_20261115_v1"
SCHEMA = "renal_absolute_native_estimator_annotation_v1"
QUANTITY = "as_reported_urinary_parent_CLR_absolute_unit_estimator_annotation"
ENDPOINT = "CL_renal_absolute_source_annotation"
UNIT = "L/h_as_reported_normalization_annotation"
MODE = "source_annotation_only"
CONTEXT_GROUP = "renal_absolute_annotation_context"
REPORT_FILE = "reports/renal_absolute_native_molecular_diagnostics.json"
CONTROL_FILE = "renal_absolute_native_fixed_controls.json"
SEEDS = (42, 2026, 3407)
NOFIT_COUNTERS = {
    "estimator_fit_attempts",
    "search_attempts",
    "model_unpickle_attempts",
    "chemistry_import_attempts",
    "production_fits_launched",
}
MODELS = [
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
]
GROUPS = shared.GROUPS
sha, read, rows, write, table, object_hash, contained = (
    shared.sha,
    shared.read,
    shared.rows,
    shared.write,
    shared.table,
    shared.object_hash,
    shared.contained,
)
SHARED_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
CORE_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
INTERNAL_SHA = "329cf60e16657eea737bdd1de2469fa8acf60dce4ab968d52859e9de609c1d27"
PINS = {
    "review.json": "1bdb3cc1e48701a566d82c48f38ef2201819bd6079b90a03dbae44df34e8e099",
    "delivery_receipt.json": "823871afaef89ce6b26e68d2c2a7546e583d7a004dee7cf83193ae705aa2ea52",
    "observations.json": "a2f7c198dee4cc6a67e112ba80d5a8967bb56a239830dcdf72cfe4e0476f8fcb",
    "identity_union_review.json": (
        "0a29803c03631a0abacd3534107969e1c4cceaa8887dc7baa78595d8984e8101"
    ),
    "REPORT.md": "a8181c27b0fa12cd5f5c0a37870176d0e005a055989dceff7749d699d4d5450b",
    "review_admission.py": "0bcb879d2bb7fdb3b1115d738919ef6527c9d6b316028ca05eea9869bdba0794",
}
TARGET_POLICY = dict(
    schema="source_log10_15g_text_portability_only_v1",
    maximum_binary64_ulp_distance=128,
    maximum_fixed15_significant_digit_decimal_quanta=1,
    all_non_target_source_fields_exact=True,
    frozen_training_target_bytes_unchanged=True,
    applies_to_model_predictions=False,
)
CATEGORY_LEVELS = {
    "estimator_family": [
        "joint_plasma_urine_NONMEM_EBE",
        "open_two_compartment_PK_source_reported_urinary_CLR",
        "urine_plasma_NCA",
    ],
    "estimator_detail": [
        "Phoenix_WinNonlin8.3_noncompartmental; unchanged_parent_urine_PK_parameter",
        "joint_NONMEM_final_population_model; individual_EBEs; "
        "observed_urine_amount=CLr×matched_plasma_AUC",
        "source_open_two_compartment_PK_after_serum70kg_normalization;reported_CLR_column",
        "urine_parent_amount0_24 / total_plasma_AUC0_24",
        "urine_recovered_parent_amount / matched_same_interval_plasma_AUC",
    ],
    "normalization_annotation": [
        "as_reported_absolute_row_no_body_reference_scaling_described",
        "if_applicable_70kg_reference_normalization_scope_unresolved",
        "reported_unscaled_absolute_row_separate_from_TBW_IBW_FFW_rows",
        "source_L_h_CLR_no_body_divisor;CrCL_covariate_BSA_reference_separate",
    ],
    "mean_kind": ["arithmetic_group_mean_of_EBEs", "arithmetic_mean", "geometric_mean"],
    "dose_unit_and_basis": ["mg", "mg/kg_TBW", "microgram_desmopressin_acetate"],
    "regimen": [
        "not_reported",
        "single_dose",
        "q6h_7day_last_dose",
        "q6h_14day_last_dose",
        "daily_14day_last_dose",
        "source_IV_alone",
        "source_IV_with_ceftazidime2g",
    ],
    "renal_status": [
        "not_reported",
        "normal",
        "mild_impairment",
        "moderate_impairment",
        "severe_impairment",
    ],
    "CrCL_unit_and_basis": ["not_reported", "mL/min_BSA_reference1.73m2"],
}
NUMERIC_FIELDS = ("dose_value", "regimen_day", "dose_interval_hours", "CrCL_group_mean")
CONTEXT_FIELDS = tuple(CATEGORY_LEVELS) + NUMERIC_FIELDS
CONTEXT_NAMES = [
    field + "=" + level for field, levels in CATEGORY_LEVELS.items() for level in levels
] + [n for field in NUMERIC_FIELDS for n in (field, field + "_missing")]
WIDTH = len(CONTEXT_NAMES)
TASK = (
    "TASK_RENAL_ABSOLUTE_NATIVE_"
    + object_hash(
        dict(quantity=QUANTITY, source=PINS, context=CATEGORY_LEVELS, numeric=NUMERIC_FIELDS)
    )[:24]
)
WARNING = (
    "Typed source-reported urinary-parent renal-clearance absolute-unit annotations only. "
    "Estimator, normalization annotation and mean-kind are mandatory query inputs. "
    "Vancomycin70kg reference scope remains unresolved; conversion to L/h does not remove it. "
    "Desmopressin rows are group means of joint plasma/urine NONMEM EBEs, "
    "not22 individual observations. Assays, intactness, sample salts/microstates, "
    "dose/regimen and populations are not fully harmonized. Five complete registered "
    "parent-reference identity/FAMILY components are not five independent scaffolds or sources. "
    "Internal fivefold OOF selects models, without independent holdout; "
    "three seeds repeat one LOGO partition. Scores are selection-optimistic exploratory "
    "annotation errors, not validated renal physiology or clinical predictions. "
    "Only2/8 original CL_renal_absolute observations are included; "
    "six normalized misclassifications remain excluded."
)


def check_dependency():
    if sha(shared.__file__) != SHARED_SHA:
        raise ValueError("Pinned numeric helper changed")


def encode_context(context):
    if not isinstance(context, dict) or set(context) != set(CONTEXT_FIELDS):
        raise ValueError("Query requires the complete explicit annotation context")
    result = []
    for field, levels in CATEGORY_LEVELS.items():
        if context[field] not in levels:
            raise ValueError("Unsupported context level: " + field)
        result.extend(context[field] == level for level in levels)
    for field in NUMERIC_FIELDS:
        value = context[field]
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0
        ):
            raise ValueError(
                "Context requires finite nonnegative native quantities or explicit missingness"
            )
        result.extend([np.nan if value is None else float(value), value is None])
    return np.asarray(result, dtype=np.float32)


def _context(row):
    return json.loads(row["semantic_context_json"])


def source_context(observation):
    """Only source-bound dose/regimen/renal covariates; no outcome/IDs/uncertainty inputs."""
    o = observation
    context = {k: o[k] for k in list(CATEGORY_LEVELS)[:4]}
    context.update(
        dose_unit_and_basis="mg",
        regimen="not_reported",
        renal_status="not_reported",
        CrCL_unit_and_basis="not_reported",
        **{k: None for k in NUMERIC_FIELDS},
    )
    drug = o["drug"]
    evidence = {k: [k] for k in list(CATEGORY_LEVELS)[:4]}
    if drug == "daptomycin":
        if o["nominal_dose"] != "4 mg/kg TBW":
            raise ValueError("Daptomycin native nominal dose changed")
        context.update(dose_value="4", dose_unit_and_basis="mg/kg_TBW")
        evidence["dose"] = ["nominal_dose"]
    elif drug == "BRII693_QPX9003_F365":
        match = re.match(r"^(\d+) mg \(", o["dose_header"])
        if not match:
            raise ValueError("BRII dose must come from this exact table column")
        context.update(dose_value=match[1], regimen=o["regimen"])
        if o["regimen"] != "single_dose":
            context.update(
                regimen_day="7" if o["regimen"] == "q6h_7day_last_dose" else "14",
                dose_interval_hours="6",
            )
        evidence["dose"] = ["dose_header"]
        evidence["regimen"] = ["regimen"]
    elif drug == "caspofungin":
        if o["nominal_dose"] != "70 mg daily ×14":
            raise ValueError("Caspofungin source day14 nominal regimen changed")
        context.update(
            dose_value="70",
            regimen="daily_14day_last_dose",
            regimen_day="14",
            dose_interval_hours="24",
        )
        evidence["dose_and_regimen"] = ["nominal_dose", "source_locator"]
    elif drug == "desmopressin":
        if (
            o["nominal_dose"] != "single 2 microgram desmopressin acetate"
            or o["CrCL_source_covariate_reference"] != "1.73m2"
        ):
            raise ValueError("Desmopressin dose/CrCL basis changed")
        group = int(o["review_observation_id"].rsplit("Group", 1)[1])
        context.update(
            dose_value="2",
            dose_unit_and_basis="microgram_desmopressin_acetate",
            regimen="single_dose",
            renal_status=["normal", "mild_impairment", "moderate_impairment", "severe_impairment"][
                group - 1
            ],
            CrCL_group_mean=o["CrCL_group_mean_native"],
            CrCL_unit_and_basis="mL/min_BSA_reference1.73m2",
        )
        evidence["dose"] = ["nominal_dose"]
        evidence["CrCL_group_mean"] = [
            "CrCL_group_mean_native",
            "CrCL_source_covariate_reference",
            "source_locator",
        ]
        evidence["renal_status"] = [
            "primary_sec7_group1_normal_group2_mild_group3_moderate_group4_severe"
        ]
    elif drug == "vancomycin":
        regimen = o["regimen"]
        if regimen not in {
            "Vancomycin0.5g_alone",
            "Vancomycin0.5g_withceftazidime2g",
            "Vancomycin1.0g_alone",
        }:
            raise ValueError("Vancomycin bound source regimen changed")
        context.update(
            dose_value="1000" if "1.0g" in regimen else "500",
            regimen="source_IV_with_ceftazidime2g"
            if "withceftazidime" in regimen
            else "source_IV_alone",
        )
        evidence["dose_and_regimen"] = ["regimen", "source_locator"]
    else:
        raise ValueError("Unreviewed parent")
    encode_context(context)
    return context, evidence


def compare_source_rows(frozen, regenerated):
    """One source-log display quantum AND128ULP; never a prediction tolerance."""
    if len(frozen) != len(regenerated):
        raise ValueError("Source row count changed")
    differences = []
    for a, b in zip(frozen, regenerated, strict=True):
        if {k: v for k, v in a.items() if k != "target"} != {
            k: v for k, v in b.items() if k != "target"
        }:
            raise ValueError("Non-target reconstructed source field changed")
        if a["target"] == b["target"]:
            continue
        x, y = float(a["target"]), float(b["target"])
        if not (math.isfinite(x) and math.isfinite(y)) or b["target"] != format(y, ".15g"):
            raise ValueError("Noncanonical regenerated source log target")
        quantum = Decimal(1).scaleb(Decimal(b["target"]).adjusted() - 14)
        ulps = abs(
            struct.unpack("!q", struct.pack("!d", x))[0]
            - struct.unpack("!q", struct.pack("!d", y))[0]
        )
        if (
            abs(Decimal(a["target"]) - Decimal(b["target"])) > quantum
            or ulps > 128
            or math.copysign(1, x) != math.copysign(1, y)
        ):
            raise ValueError("Source-only log portability bound exceeded")
        differences.append(
            dict(
                record_id=a["record_id"],
                frozen_target_text=a["target"],
                regenerated_target_text=b["target"],
                binary64_ULPs=ulps,
            )
        )
    return differences


def source_rows(source=SOURCE):
    source = Path(source)
    for name, expected in PINS.items():
        if sha(contained(source, name)) != expected:
            raise ValueError("Independent source review pin changed: " + name)
    review, observations, identities = (
        read(source / "review.json"),
        read(source / "observations.json"),
        read(source / "identity_union_review.json"),
    )
    if (
        review["verdict"] != "QUALIFIED_APPROVE_SOURCE_ANNOTATION_ONLY"
        or len(observations) != 22
        or len(identities) != 5
    ):
        raise ValueError("Independent qualified22/5 source contract changed")
    values = []
    for o in observations:
        identity = identities[o["drug"]]
        if (
            o["canonical_graph_key"] != identity["canonical_graph_key"]
            or o["known_FAMILY"] != identity["known_FAMILY"]
            or o["native_qualifier"] != "reported_table_central_point_no_inequality"
        ):
            raise ValueError("Reviewed exact graph/FAMILY/scalar qualifier changed")
        factor = {"L/h": Decimal(1), "mL/min": Decimal("0.06")}[o["native_unit"]]
        converted = Decimal(o["native_central_value"]) * factor
        if converted <= 0 or converted != Decimal(o["comparison_value_L_h_unit_conversion_only"]):
            raise ValueError("Exact source unit conversion disagrees")
        context, evidence = source_context(o)
        semantic = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        record = dict(
            record_id=o["review_observation_id"],
            endpoint=ENDPOINT,
            task_id=TASK,
            task_kind="regression",
            value=format(converted, "f"),
            unit=UNIT,
            target=format(math.log10(float(converted)), ".15g"),
            target_transform="log10",
            source_native_qualifier=o["native_qualifier"],
            source_native_value=o["native_central_value"],
            source_native_unit=o["native_unit"],
            source_original_record_id=o.get("original_frozen_record", {}).get("record_id", ""),
            source_original_endpoint=o.get("original_frozen_record", {}).get("endpoint", ""),
            canonical_identity_group=o["canonical_graph_key"],
            canonical_smiles=identity["canonical_smiles"],
            smiles=identity["canonical_smiles"],
            known_family="|".join(o["known_FAMILY"]),
            split_group=identity["union_component"],
            structure_eligible="true",
            renal_annotation_context_eligible="true",
            sequence_eligible="false",
            semantic_context_key=object_hash(context),
            semantic_context_json=semantic,
            context_provenance_json=json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            source_observation_json=json.dumps(o, ensure_ascii=False, sort_keys=True),
            source_study=o["source_path"],
            source_grade="primary_table_source_annotation",
            readout_verification_status="source_parent_assignment_assay_specific_not_independent_chemical_intactness",
        )
        record.update({k: "not_reported" if v is None else str(v) for k, v in context.items()})
        values.append(record)
    if len({r["record_id"] for r in values}) != 22 or len({r["split_group"] for r in values}) != 5:
        raise ValueError("Reviewed global graph/FAMILY components changed")
    return values, observations, identities


def split_prechecks(values):
    from peptide_modeling_internal_cv import internal_splits
    from peptide_modeling_train import validate_rows as validate_training

    y, groups = validate_training(values, "regression")
    checks, partitions = [], set()
    for seed in SEEDS:
        folds = []
        for fold, (train, test) in enumerate(
            internal_splits(y, groups, "regression", dict(seed=seed, inner_folds=5))
        ):
            if (
                set(groups[train]) & set(groups[test])
                or len(set(groups[test])) != 1
                or len(set(groups[train])) != 4
            ):
                raise ValueError("Frozen molecular LOGO fivefold changed")
            known = {values[i]["semantic_context_key"] for i in train}
            folds.append(
                dict(
                    fold=fold,
                    train_record_ids=[values[i]["record_id"] for i in train],
                    validation_record_ids=[values[i]["record_id"] for i in test],
                    train_groups=sorted(set(groups[train])),
                    validation_groups=sorted(set(groups[test])),
                    unsupported_joint_context_record_ids=[
                        values[i]["record_id"]
                        for i in test
                        if values[i]["semantic_context_key"] not in known
                    ],
                )
            )
        for f in folds:
            a = [i for i, r in enumerate(values) if r["record_id"] in f["train_record_ids"]]
            b = [i for i, r in enumerate(values) if r["record_id"] in f["validation_record_ids"]]
            f["per_field_context_support"] = {}
            for field in CONTEXT_FIELDS:
                observed = {_context(values[i])[field] for i in a}
                unsupported = []
                for i in b:
                    value = _context(values[i])[field]
                    if field in CATEGORY_LEVELS:
                        supported = value in observed
                    elif value is None:
                        supported = None in observed
                    else:
                        numeric = [float(v) for v in observed if v is not None]
                        supported = bool(numeric) and min(numeric) <= float(value) <= max(numeric)
                    if not supported:
                        unsupported.append(values[i]["record_id"])
                f["per_field_context_support"][field] = dict(
                    unsupported_record_ids=unsupported,
                    criterion="seen_level"
                    if field in CATEGORY_LEVELS
                    else "missingness_seen_or_training_numeric_range_not_validation",
                )
        partitions.add(tuple(sorted(tuple(sorted(f["validation_record_ids"])) for f in folds)))
        checks.append(dict(seed=seed, folds=folds, actual_folds=5))
    if len(partitions) != 1:
        raise ValueError("Five identity groups must have one LOGO partition")
    return dict(
        seeds=checks,
        distinct_molecular_partitions=1,
        independent_holdout=False,
        identity_groups=5,
        known_scaffold_groups_not_claimed=True,
        strict_scaffold_validation_claimed=False,
    )


def slice_features(values, parent, output):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    pointer = Path(parent) / "features/manifest.json"
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Frozen parent pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    root = Path(base["cache_dir"])
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in values}
    source_smiles = {
        r["key"]: r["canonical_isomeric_smiles"] for r in rows(root / "structure_source_smiles.tsv")
    }
    if any(source_smiles.get(k) != v for k, v in graph_map.items()) or len(graph_map) != 5:
        raise ValueError("All five exact SMILES bytes must match the parent graph bank")
    index = {r["key"]: int(r["row_idx"]) for r in rows(root / "structure_bank_index.tsv")}
    keys = sorted(graph_map)
    indices = [index[k] for k in keys]
    cache = Path(output) / "features" / object_hash([QUANTITY, graph_map, values])[:16]
    cache.mkdir(parents=True, exist_ok=True)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    for meta in groups.values():
        matrix = np.load(root / meta["matrix"], mmap_mode="r")
        selected = matrix[indices].copy()
        valid = np.load(root / meta["valid"])[indices].copy()
        if not valid.all() or selected.tobytes() != matrix[indices].tobytes():
            raise ValueError("Cached chemical vectors must be byte-identical and valid")
        np.save(cache / meta["matrix"], selected, allow_pickle=False)
        np.save(cache / meta["valid"], valid, allow_pickle=False)
        meta.update(shape=[5, selected.shape[1]], index="structure_bank_index.tsv")
    context = np.vstack([encode_context(_context(r)) for r in values])
    np.save(cache / "semantic_context.npy", context, allow_pickle=False)
    np.save(cache / "semantic_context_valid.npy", np.ones(22, dtype=bool), allow_pickle=False)
    groups[CONTEXT_GROUP] = dict(
        matrix="semantic_context.npy",
        valid="semantic_context_valid.npy",
        index="context_index.tsv",
        key_kind="record_id",
        eligibility_field="renal_annotation_context_eligible",
        shape=[22, WIDTH],
        feature_names=CONTEXT_NAMES,
        chemical_method=False,
        source_kind="semantic_context",
        allow_partial_missing=True,
        max_missing_fraction=0.1,
    )
    table(cache / "structure_bank_index.tsv", [dict(key=k, row_idx=i) for i, k in enumerate(keys)])
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=k, canonical_isomeric_smiles=graph_map[k]) for k in keys],
    )
    table(
        cache / "context_index.tsv",
        [dict(key=r["record_id"], row_idx=i) for i, r in enumerate(values)],
    )
    table(
        cache / "record_bank_index.tsv",
        [
            dict(
                record_id=r["record_id"],
                structure_key=r["canonical_identity_group"],
                sequence_key="",
            )
            for r in values
        ],
    )
    methods = copy.deepcopy(
        [
            m
            for m in base["representation_methods"]["methods"]
            if m["source_kind"] == "structure" and m["independent"]
        ]
    )
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Exactly ten independent chemical methods are required")
    registry = dict(
        methods=methods,
        independent_method_count=10,
        chemical_dimensions=15212,
        chemical_groups=13,
        context_columns=WIDTH,
        context_method_increment=0,
    )
    routes = {m["route"]: m["group_names"] + [CONTEXT_GROUP] for m in methods}
    routes["structure_methods_all"] = list(GROUPS) + [CONTEXT_GROUP]
    write(cache / "source_parent_manifest.json", base)
    write(cache / "representation_methods.json", registry)
    manifest = dict(
        schema=SCHEMA,
        input_fingerprint=object_hash(
            [QUANTITY, keys, hashlib.sha256(context.tobytes()).hexdigest()]
        ),
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        record_index="record_bank_index.tsv",
        requested_records=22,
        structure_keys=5,
        groups=groups,
        routes=routes,
        representation_methods=registry,
        parent_pointer_sha256=PARENT_SHA,
        chemical_feature_extractions=0,
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p)
        for p in cache.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for route, names in routes.items():
        if names.count(CONTEXT_GROUP) != 1:
            raise ValueError("Context must be appended exactly once on every route")
        loaded = load_feature_route(values, manifest, route)
        if (
            not loaded["valid"].all()
            or loaded["X"][:, -WIDTH:].tobytes() != context.tobytes()
            or loaded["feature_names"][-WIDTH:] != [CONTEXT_GROUP + ":" + n for n in CONTEXT_NAMES]
        ):
            raise ValueError("Actual route loading failed context/identity alignment")
    if sum(groups[n]["shape"][1] for n in GROUPS) != 15212:
        raise ValueError("Chemical fusion width changed")
    _verify_artifacts(base)
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Preparation changed the immutable parent pointer")
    write(Path(output) / "features/manifest.json", manifest)
    write(Path(output) / "representation_methods.json", registry)
    return manifest


def _metrics(values, prediction):
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Prediction vector must cover every exact source record")
    counts = Counter(r["split_group"] for r in values)
    weights = np.asarray([1 / counts[r["split_group"]] for r in values])
    error = y - p
    return dict(
        records=len(values),
        groups=len(counts),
        record_weighted=dict(
            mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2)))
        ),
        molecular_equal=dict(
            mae=float(np.average(abs(error), weights=weights)),
            rmse=float(np.sqrt(np.average(error**2, weights=weights))),
        ),
    )


def precision_evidence(values, payload):
    X = np.asarray(payload["X"])
    if X.dtype != np.float32 or payload["record_ids"] != [r["record_id"] for r in values]:
        raise ValueError("Selected-model precision evidence requires actual float32 route rows")
    return dict(
        feature_width=X.shape[1],
        matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        record_rows={
            r["record_id"]: dict(
                graph=r["canonical_identity_group"],
                context=r["semantic_context_key"],
                group=r["split_group"],
                row_sha256=hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest(),
            )
            for i, r in enumerate(values)
        },
    )


def _bits32(value):
    if not math.isfinite(value) or float(np.float32(value)) != value or abs(value) < 2.0**-126:
        raise ValueError("Drift guard requires exact normal binary32 values")
    return struct.unpack("!I", struct.pack("!f", value))[0]


def prediction_consistency(predictions, evidence, provenance):
    bound = [evidence["record_rows"][r["record_id"]] for r in predictions]
    if (
        not bound
        or len({(b["graph"], b["context"], b["group"], b["row_sha256"]) for b in bound}) != 1
        or len({(r.get("evaluation_role", "internal_oof"), int(r["fold"])) for r in predictions})
        != 1
    ):
        raise ValueError("Drift comparison requires the same full input/graph/context/group/fold")
    p = [float(r["prediction"]) for r in predictions]
    if not all(math.isfinite(v) for v in p):
        raise ValueError("Nonfinite prediction")
    if len(set(p)) == 1:
        return dict(mode="exact", raw_predictions_unchanged=True)
    bits = [_bits32(v) for v in p]
    required = dict(
        selected_model="linear",
        actual_estimator="Ridge",
        actual_solver="lsqr",
        actual_coef_dtype="float32",
        actual_scaler_with_mean=False,
        actual_imputer_strategy="median",
        actual_add_indicator=True,
        input_dtype="float32",
        feature_width=evidence["feature_width"],
        full_matrix_sha256=evidence["matrix_sha256"],
    )
    from peptide_modeling_train import __file__ as training_file

    if (
        any(provenance.get(k) != v for k, v in required.items())
        or provenance.get("training_source_sha256") != sha(training_file)
        or not provenance.get("actual_model_artifact_sha256")
        or len({math.copysign(1, v) for v in p}) != 1
        or max(bits) - min(bits) > 32
    ):
        raise ValueError("Identical-input drift lacks the predeclared bounded32ULP LSQR proof")
    return dict(
        mode="exact_normal_binary32_actual_LSQR_at_most32ULP",
        observed_binary32_ULPs=max(bits) - min(bits),
        maximum_ULPs=32,
        engineering_guard_not_a_universal_BLAS_bound=True,
        raw_predictions_unchanged=True,
    )


def evaluation_diagnostics(values, oof, check, payload, provenance):
    by_id = {r["record_id"]: r for r in values}
    if [r["record_id"] for r in oof] != list(by_id):
        raise ValueError("Actual OOF rows must exactly preserve all22 source identities/order")
    folds = {rid: f["fold"] for f in check["folds"] for rid in f["validation_record_ids"]}
    comparisons = defaultdict(list)
    for r in oof:
        old = by_id[r["record_id"]]
        if (
            r["split_group"] != old["split_group"]
            or int(r["fold"]) != folds[r["record_id"]]
            or float(r["target"]) != float(old["target"])
        ):
            raise ValueError("OOF target, group or frozen actual fold changed")
        comparisons[
            (old["canonical_identity_group"], old["semantic_context_key"], int(r["fold"]))
        ].append(r)
    evidence = precision_evidence(values, payload)
    consistency = [prediction_consistency(v, evidence, provenance) for v in comparisons.values()]
    pred = [float(r["prediction"]) for r in oof]
    report = _metrics(values, pred)
    report["strata"] = {}
    for field in [*CONTEXT_FIELDS, "source_grade", "readout_verification_status"]:
        report["strata"][field] = {}
        for label in sorted({r[field] for r in values}):
            idx = [i for i, r in enumerate(values) if r[field] == label]
            report["strata"][field][label] = _metrics(
                [values[i] for i in idx], [pred[i] for i in idx]
            )
    report.update(
        annotation_only=True,
        validated_physiology=False,
        independent_holdout=False,
        BRII_record_dominance=dict(records=12, denominator=22),
        fold_context_support=check["folds"],
        prediction_consistency=consistency,
        source_grade_used_only_for_reporting=True,
        raw_predictions_mutated=False,
    )
    report["per_molecule_errors"] = {
        group: _metrics(
            [r for r in values if r["split_group"] == group],
            [pred[i] for i, r in enumerate(values) if r["split_group"] == group],
        )
        for group in sorted({r["split_group"] for r in values})
    }
    report["original_endpoint_observation_coverage"] = dict(
        original_endpoint="CL_renal_absolute",
        covered=2,
        total_original=8,
        included_original_record_ids=[
            r["source_original_record_id"] for r in values if r["source_original_record_id"]
        ],
        all_original_eight_covered=False,
    )
    report["independent_sources_or_scaffolds_validated"] = False
    return report


def fixed_control_estimator():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )
    # Instrument each concrete public method once. Never count both an ancestor
    # and its delegated implementation; these are actual Ridge boundary calls.
    learner = estimator.named_steps["model"]
    fit, predict = learner.fit, learner.predict
    learner._native_calls = dict(fit=0, predict=0)

    def fit_once(X, y, **kwargs):
        learner._native_calls["fit"] += 1
        learner._native_fit_boundary = array_boundary(X, y)
        return fit(X, y, **kwargs)

    def predict_once(X, **kwargs):
        learner._native_calls["predict"] += 1
        learner._native_predict_boundary = array_boundary(X)
        return predict(X, **kwargs)

    learner.fit, learner.predict = fit_once, predict_once
    return estimator


def array_boundary(X, y=None):
    x = np.asarray(X)
    out = dict(
        X_dtype=str(x.dtype),
        X_shape=list(x.shape),
        X_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
    )
    if y is not None:
        y = np.asarray(y)
        out.update(
            y_dtype=str(y.dtype),
            y_shape=list(y.shape),
            y_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        )
    return out


def _fixed_control_trace(
    estimator, X_train, y_train, X_predict, prediction, train_ids, predict_ids, *, fold, role
):
    steps = estimator.named_steps
    learner = steps["model"]
    coef = np.asarray(learner.coef_)
    predicted = np.asarray(prediction)
    params = dict(
        alpha=float(learner.alpha),
        solver=learner.solver,
        imputer_strategy=steps["impute"].strategy,
        imputer_add_indicator=steps["impute"].add_indicator,
        imputer_keep_empty_features=steps["impute"].keep_empty_features,
        scaler_with_mean=steps["scale"].with_mean,
    )
    if (
        type(learner).__name__ != "Ridge"
        or params
        != dict(
            alpha=1.0,
            solver="lsqr",
            imputer_strategy="median",
            imputer_add_indicator=True,
            imputer_keep_empty_features=True,
            scaler_with_mean=False,
        )
        or any(a.dtype != np.float64 for a in [X_train, y_train, X_predict, predicted, coef])
        or learner._native_calls != dict(fit=1, predict=1)
        or learner._native_fit_boundary["X_dtype"] != "float64"
        or learner._native_fit_boundary["y_dtype"] != "float64"
        or learner._native_predict_boundary["X_dtype"] != "float64"
    ):
        raise ValueError("Actual fixed-control fit/predict/coef must remain float64 RidgeLSQR")

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    return dict(
        fold=fold,
        fit_role=role,
        train_record_ids=train_ids,
        predict_record_ids=predict_ids,
        actual_estimator=type(learner).__name__,
        actual_parameters=params,
        actual_fit_X_dtype=str(X_train.dtype),
        actual_fit_y_dtype=str(y_train.dtype),
        actual_predict_X_dtype=str(X_predict.dtype),
        actual_prediction_dtype=str(predicted.dtype),
        actual_coef_dtype=str(coef.dtype),
        fit_X_sha256=digest(X_train),
        fit_y_sha256=digest(y_train),
        predict_X_sha256=digest(X_predict),
        predictions_sha256=digest(predicted),
        actual_coef_sha256=digest(coef),
        actual_coef_shape=list(coef.shape),
        actual_public_Ridge_calls=learner._native_calls.copy(),
        transformed_Ridge_fit_boundary=learner._native_fit_boundary,
        transformed_Ridge_predict_boundary=learner._native_predict_boundary,
        control_fit_calls=1,
        selected_model_fit_calls=0,
        search_calls=0,
    )


def fixed_ridge_controls(values, chemical_X, context_X, check):
    """Explicit run phase:5OOF+1full-cohort control fits/design; never report-only."""
    from threadpoolctl import threadpool_limits

    chemical = np.asarray(chemical_X, dtype=np.float64)
    context = np.asarray(context_X, dtype=np.float64)
    designs = dict(
        molecule_only=chemical,
        context_only=context,
        molecule_plus_context=np.column_stack((chemical, context)),
    )
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    positions = {r["record_id"]: i for i, r in enumerate(values)}
    all_ids = [r["record_id"] for r in values]
    result = {}
    for name, X in designs.items():
        assert X.dtype == np.float64
        pred = np.full(len(values), np.nan, dtype=np.float64)
        traces = []
        for fold in check["folds"]:
            a = [positions[r] for r in fold["train_record_ids"]]
            b = [positions[r] for r in fold["validation_record_ids"]]
            estimator = fixed_control_estimator()
            with threadpool_limits(limits=1):
                estimator.fit(X[a], y[a])
                part = estimator.predict(X[b])
            pred[b] = part
            traces.append(
                _fixed_control_trace(
                    estimator,
                    X[a],
                    y[a],
                    X[b],
                    part,
                    fold["train_record_ids"],
                    fold["validation_record_ids"],
                    fold=fold["fold"],
                    role="inner_training_fold",
                )
            )
        full_estimator = fixed_control_estimator()
        with threadpool_limits(limits=1):
            full_estimator.fit(X, y)
            full_prediction = full_estimator.predict(X)
        traces.append(
            _fixed_control_trace(
                full_estimator,
                X,
                y,
                X,
                full_prediction,
                all_ids,
                all_ids,
                fold=-1,
                role="all_internal_cohort_control_fit",
            )
        )
        result[name] = dict(
            model="fixed_Ridge_alpha1_LSQR_not_searched",
            dtype_from_initial_boundary="float64",
            source_encoded_values_rewritten=False,
            same_frozen_inner_fivefold=True,
            source_value_origin="upcast_exact_existing_encoded_float32_values",
            prediction_record_ids=all_ids,
            predictions=pred.tolist(),
            metrics=_metrics(values, pred),
            matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
            per_fold_fit_trace=traces,
            control_fit_calls=6,
            selected_model_fit_calls=0,
            search_calls=0,
            all_cohort_fit_predictions=np.asarray(full_prediction).tolist(),
            full_cohort_evaluation_role="training_inputs_only_no_independent_performance_score",
        )
    return result


def write_report(prepared, *, run_dir=None):
    """Pure artifact/numeric reporting: no chemistry, unpickling, search, or fitting."""
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_train import _digest as training_digest

    root = Path(prepared)
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    verify_source_proof(root, task, values)
    report = dict(
        schema=SCHEMA,
        task_id=TASK,
        status="prepared_not_trained",
        records=22,
        identity_groups=5,
        known_scaffold_groups_not_claimed=True,
        selected_models_fitted_by_reporting=0,
        any_estimators_fitted_by_reporting=0,
        annotation_only=True,
        validated_physiology=False,
        warning=WARNING,
    )
    if run_dir is not None:
        run = Path(run_dir)
        result = read(run / "results/internal_cv" / (TASK + ".json"))
        if (
            result.get("status") != "completed"
            or result.get("task_id") != TASK
            or result.get("endpoint") != ENDPOINT
            or result.get("actual_inner_folds") != 5
            or result.get("requested_inner_folds") != 5
            or result.get("protocol") != "internal_grouped_cv_only_no_independent_holdout"
            or result.get("target_unit") != UNIT
            or result.get("target_transform") != "log10"
            or result.get("independent_holdout_available") is not False
        ):
            raise ValueError("Metadata reporting requires completed real internal fivefold OOF")
        for name, receipt in result["artifact_hashes"].items():
            path = Path(name).resolve()
            path.relative_to(run.resolve())
            if sha(path) != receipt["sha256"]:
                raise ValueError("Result artifact hash changed")
        actual_task = read(run / "task_manifest.json")["tasks"][0]
        actual_values = rows(contained(run, actual_task["table_path"]))
        verify_source_proof(run, actual_task, actual_values)
        if actual_values != values or actual_task != task:
            raise ValueError("Runtime source contract differs from prepared task")
        contract = read(result["artifacts"]["model_contract"])
        inputs = read(result["artifacts"]["input_contract"])
        if (
            contract["task"] != task
            or contract.get("fit_scope") != "all_internal_cv_rows"
            or result.get("n_internal_cv_rows") != 22
            or result.get("n_internal_cv_groups") != 5
            or contract["input_hash"] != result["input_hash"]
            or any(
                contract.get(k) != result.get(k)
                for k in ("selected_route", "selected_model", "selected_params")
            )
            or contract["record_ids"] != [r["record_id"] for r in values]
            or inputs["source_rows_hash"] != training_digest(values)
        ):
            raise ValueError("Runtime model/source metadata mismatch")
        feature = feature_manifest(run)
        payload = load_feature_route(values, feature, result["selected_route"])
        if payload["feature_names"] != contract["feature_names"] or not payload["valid"].all():
            raise ValueError("Actual selected route schema changed")
        selected_input = inputs["features"][result["selected_route"]]
        if (
            selected_input["names"] != payload["feature_names"]
            or selected_input["matrix_sha256"]
            != hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest()
            or selected_input["valid_sha256"]
            != hashlib.sha256(payload["valid"].tobytes()).hexdigest()
        ):
            raise ValueError("Selected numeric route differs from the actual frozen fit input")
        check = next(
            c
            for c in read(root / "task_tables/predeclared_splits.json")["seeds"]
            if c["seed"] == result["seed"]
        )
        controls_path = run / CONTROL_FILE
        controls = read(controls_path)
        if (
            controls["source_proof_sha256"]
            != task["native_source_observation_coverage"]["proof_sha256"]
            or controls["seed"] != result["seed"]
            or controls["selected_result_sha256"]
            != sha(run / "results/internal_cv" / (TASK + ".json"))
            or controls.get("frozen_check_sha256") != object_hash(check)
            or controls["selected_estimator_provenance"].get("actual_model_artifact_sha256")
            != sha(result["artifacts"]["model"])
        ):
            raise ValueError("Mandatory precomputed controls are absent or stale")
        validate_controls(values, payload, check, controls["controls"])
        diagnostic = evaluation_diagnostics(
            values,
            rows(result["artifacts"]["internal_oof"]),
            check,
            payload,
            controls["selected_estimator_provenance"],
        )
        for metric in ("mae", "rmse"):
            if not math.isclose(
                diagnostic["record_weighted"][metric],
                result["internal_cv_selection_metrics"][metric],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError("OOF metrics disagree with saved selection results")
        report.update(
            status="verified_internal_annotation_only",
            seed=result["seed"],
            diagnostics=diagnostic,
            fixed_ridge_controls=controls["controls"],
            controls_sha256=sha(controls_path),
            result_input_hash=result["input_hash"],
        )
    write(root / REPORT_FILE, report)
    return report


def validate_controls(values, payload, check, controls):
    """Metadata-only checks on saved fixed controls; never recreate their estimators."""
    if set(controls) != {"molecule_only", "context_only", "molecule_plus_context"}:
        raise ValueError("Exactly the three predeclared fixed controls are mandatory")
    X = np.asarray(payload["X"], dtype=np.float64)
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    ids = [r["record_id"] for r in values]
    positions = {rid: i for i, rid in enumerate(ids)}
    designs = dict(molecule_only=X[:, :-WIDTH], context_only=X[:, -WIDTH:], molecule_plus_context=X)
    for name, artifact in controls.items():
        if (
            artifact["model"] != "fixed_Ridge_alpha1_LSQR_not_searched"
            or artifact["dtype_from_initial_boundary"] != "float64"
            or artifact["same_frozen_inner_fivefold"] is not True
            or artifact["source_encoded_values_rewritten"] is not False
            or artifact["prediction_record_ids"] != [r["record_id"] for r in values]
            or artifact["matrix_sha256"]
            != hashlib.sha256(np.ascontiguousarray(designs[name]).tobytes()).hexdigest()
            or artifact["metrics"] != _metrics(values, artifact["predictions"])
        ):
            raise ValueError("Fixed control dtype, immutable values, membership or metrics changed")
        traces = artifact.get("per_fold_fit_trace", [])
        if (
            len(traces) != 6
            or artifact.get("control_fit_calls") != 6
            or artifact.get("selected_model_fit_calls") != 0
            or artifact.get("search_calls") != 0
        ):
            raise ValueError("Actual5OOF+1full control trace is mandatory without search")
        for index, trace in enumerate(traces):
            if index < 5:
                frozen = check["folds"][index]
                train_ids, predict_ids = frozen["train_record_ids"], frozen["validation_record_ids"]
                role, fold = "inner_training_fold", frozen["fold"]
                pred = np.asarray(
                    [artifact["predictions"][positions[r]] for r in predict_ids], dtype=np.float64
                )
            else:
                train_ids, predict_ids, role, fold = ids, ids, "all_internal_cohort_control_fit", -1
                pred = np.asarray(artifact["all_cohort_fit_predictions"], dtype=np.float64)
            a, b = [positions[r] for r in train_ids], [positions[r] for r in predict_ids]
            design = designs[name]
            required = dict(
                fold=fold,
                fit_role=role,
                train_record_ids=train_ids,
                predict_record_ids=predict_ids,
                actual_estimator="Ridge",
                actual_fit_X_dtype="float64",
                actual_fit_y_dtype="float64",
                actual_predict_X_dtype="float64",
                actual_prediction_dtype="float64",
                actual_coef_dtype="float64",
                control_fit_calls=1,
                selected_model_fit_calls=0,
                search_calls=0,
                actual_parameters=dict(
                    alpha=1.0,
                    solver="lsqr",
                    imputer_strategy="median",
                    imputer_add_indicator=True,
                    imputer_keep_empty_features=True,
                    scaler_with_mean=False,
                ),
            )
            for key, array in [
                ("fit_X_sha256", design[a]),
                ("fit_y_sha256", y[a]),
                ("predict_X_sha256", design[b]),
                ("predictions_sha256", pred),
            ]:
                required[key] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if (
                any(trace.get(k) != v for k, v in required.items())
                or not trace.get("actual_coef_sha256")
                or not np.isfinite(pred).all()
                or trace.get("actual_public_Ridge_calls") != dict(fit=1, predict=1)
            ):
                raise ValueError("Actual float64 fit/predict membership/dtype/hash trace changed")
            fit_boundary = trace.get("transformed_Ridge_fit_boundary", {})
            predict_boundary = trace.get("transformed_Ridge_predict_boundary", {})
            if (
                fit_boundary.get("X_dtype") != "float64"
                or fit_boundary.get("y_dtype") != "float64"
                or predict_boundary.get("X_dtype") != "float64"
                or fit_boundary.get("y_sha256") != required["fit_y_sha256"]
                or fit_boundary.get("X_shape", [None])[0] != len(a)
                or predict_boundary.get("X_shape", [None])[0] != len(b)
                or not fit_boundary.get("X_sha256")
                or not predict_boundary.get("X_sha256")
            ):
                raise ValueError("Actual transformed Ridge boundaries are mandatory")
            duplicate = defaultdict(list)
            for row, prediction in zip(design[b], pred, strict=True):
                duplicate[np.ascontiguousarray(row).tobytes()].append(float(prediction))
            if any(not np.allclose(p, p[0], rtol=1e-7, atol=1e-8) for p in duplicate.values()):
                raise ValueError(
                    "Float64 fixed-control identical inputs disagree; inspect, do not widen"
                )
    if set(r for f in check["folds"] for r in f["validation_record_ids"]) != {
        r["record_id"] for r in values
    }:
        raise ValueError("Fixed control folds do not cover all exact native records")


def capture_solver_provenance(model, result, payload):
    from peptide_modeling_train import __file__ as training_file

    out = dict(
        selected_model=result["selected_model"],
        input_dtype=str(payload["X"].dtype),
        feature_width=payload["X"].shape[1],
        full_matrix_sha256=hashlib.sha256(payload["X"].tobytes()).hexdigest(),
        actual_model_artifact_sha256=sha(result["artifacts"]["model"]),
        training_source_sha256=sha(training_file),
    )
    if result["selected_model"] == "linear":
        steps = model.named_steps
        out.update(
            actual_estimator=type(steps["model"]).__name__,
            actual_solver=steps["model"].solver,
            actual_coef_dtype=str(steps["model"].coef_.dtype),
            actual_scaler_with_mean=steps["scale"].with_mean,
            actual_imputer_strategy=steps["impute"].strategy,
            actual_add_indicator=steps["impute"].add_indicator,
        )
    return out


def run_followup(config_path):
    """Explicit future node execution; preparation and report-only never enter here."""
    import peptide_modeling_internal_cv as internal

    cfg = read(config_path)
    template = Path(cfg["template_run"])
    verify_source_proof(template)
    validate_node_runtime(template)
    settings = read(cfg["template_config"])
    if (
        cfg.get("required_seeds") != list(SEEDS)
        or cfg.get("workers", 1) != 1
        or settings["training"]["models"] != MODELS
        or settings["training"]["search_backend"] != "optuna"
        or settings["training"]["search_budget"] != 128
        or settings["training"]["n_jobs"] != 1
        or settings["resources"]["cpu_budget"] != 1
        or cfg.get("additional_twenty_hour_loop", False)
    ):
        raise ValueError("Bounded three-seed/Optuna128/one-CPU native configuration changed")
    original_run_seed = internal._run_seed

    def checked_run_seed(run, settings, tasks, workers, poll_seconds, durable):
        # Parent dispatcher stages fresh rounds. Validate their complete source,
        # arrays and actual node splitter output BEFORE it creates a fit process.
        preparation_review(run)
        validate_node_runtime(run)
        if split_prechecks(rows(contained(run, tasks[0]["table_path"]))) != read(
            Path(run) / "task_tables/predeclared_splits.json"
        ):
            raise ValueError("Actual staged node plans changed before fitting")
        return original_run_seed(run, settings, tasks, workers, poll_seconds, durable)

    internal._run_seed = checked_run_seed
    try:
        journal = internal.run_internal_followup(config_path)
    finally:
        internal._run_seed = original_run_seed
    try:
        result = _complete_followup_reports(cfg, journal)
    except BaseException as exc:
        journal.update(
            status="requirement_incomplete",
            renal_absolute_annotation_reporting=dict(
                status="failed", reason=f"{type(exc).__name__}:{exc}"
            ),
        )
        for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
            write(dest / "internal_campaign_status.json", journal)
        raise
    journal["renal_absolute_annotation_reporting"] = dict(status="verified", task_id=TASK)
    for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
        write(dest / "internal_campaign_status.json", journal)
    return result


def _complete_followup_reports(cfg, journal):
    import joblib
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_internal_cv import verify_internal_task

    reports = []
    for entry in journal["rounds"]:
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        values = rows(contained(run, task["table_path"]))
        verify_source_proof(run, task, values)
        features = feature_manifest(run)
        result_path = run / "results/internal_cv" / (TASK + ".json")
        result = read(result_path)
        verify_internal_task(task, values, features, run, result)
        payload = load_feature_route(values, features, result["selected_route"])
        model = joblib.load(result["artifacts"]["model"])
        check = next(
            c
            for c in read(run / "task_tables/predeclared_splits.json")["seeds"]
            if c["seed"] == entry["seed"]
        )
        write(
            run / CONTROL_FILE,
            dict(
                source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
                seed=entry["seed"],
                selected_result_sha256=sha(result_path),
                frozen_check_sha256=object_hash(check),
                selected_estimator_provenance=capture_solver_provenance(model, result, payload),
                controls=fixed_ridge_controls(
                    values, payload["X"][:, :-WIDTH], payload["X"][:, -WIDTH:], check
                ),
            ),
        )
        reports.append(write_report(run, run_dir=run))
        for relative in [
            CONTROL_FILE,
            REPORT_FILE,
        ]:
            destination = Path(cfg["durable_root"]) / run.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / relative, destination)
            if sha(run / relative) != sha(destination):
                raise ValueError("Durable mandatory native diagnostics differ")
    if len(reports) != 3 or {r["seed"] for r in reports} != set(SEEDS):
        raise ValueError("All three mandatory metadata reports must complete")
    report = dict(
        schema=SCHEMA,
        status="verified_internal_annotation_only",
        reports=reports,
        warning=WARNING,
        additional_twenty_hour_loop=False,
    )
    for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
        write(dest / "renal_absolute_native_campaign_diagnostics.json", report)
    return report


def validate_node_runtime(prepared):
    """Production entry must match the pre-fit splitter host/code/runtime, without fitting."""
    import socket
    import sys

    from peptide_modeling_internal_cv import runtime_contract

    proof = verify_source_proof(prepared)
    if "nodefold_evidence" not in proof:
        raise ValueError("Freeze actual node23 no-fit folds before any production execution")
    receipt = read(contained(prepared, proof["nodefold_evidence"]["path"]))
    if (
        socket.gethostname() != "node23"
        or Path(sys.executable).resolve() != Path("/usr/bin/python3").resolve()
        or runtime_contract() != receipt["runtime"]
    ):
        raise ValueError("Runtime changed from the actual node23 pre-fit fold receipt")
    return receipt["runtime"]


def snapshot_sources(source, output):
    """Verify the full independent audit dependency set, and stage its exact bytes."""
    source, output = Path(source), Path(output)
    destination = output / "task_tables/source_snapshots"
    names = []
    for name, expected in PINS.items():
        src, dst = source / name, destination / name
        if sha(src) != expected:
            raise ValueError("Approved source package changed")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        names.append(dst.relative_to(output).as_posix())
    dependencies = read(source / "review.json")["source_input_sha256"]
    for name, expected in dependencies.items():
        src, dst = contained(ROOT, name), destination / "dependencies" / name
        if sha(src) != expected:
            raise ValueError("Independently reviewed dependency changed: " + name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        names.append(dst.relative_to(output).as_posix())
    return names


def numeric_policy():
    return dict(
        schema=SCHEMA,
        target_text_portability_policy=TARGET_POLICY,
        selected_binary32_identical_input_guard="same_exact_full_input_graph_context_group_fold_actual_LSQR_at_most32ULP",
        fixed_controls_dtype="float64",
        fixed_controls_identity_rtol=1e-7,
        fixed_controls_identity_atol=1e-8,
        fixed_controls_no_binary32_tolerance=True,
        frozen_targets_rewritten=False,
        missing_numeric_context="IEEE_NaN_byte_preserved_with_explicit_missing_mask",
        selection_objective="negative_MAE",
        core_sha256=CORE_SHA,
        internal_sha256=INTERNAL_SHA,
    )


def context_contract():
    return dict(
        schema=SCHEMA,
        quantity_id=QUANTITY,
        mandatory_query_mode=MODE,
        mandatory_semantic_fields=list(CATEGORY_LEVELS)[:4],
        categorical_levels=CATEGORY_LEVELS,
        numeric_fields=list(NUMERIC_FIELDS),
        feature_names=CONTEXT_NAMES,
        width=WIDTH,
        categories_frozen_from_unlabelled_source_annotations=True,
        dose_mass_unit_conversion_only_no_bodyweight_guess=True,
        CrCL_reference_is_separate_from_drug_CLR=True,
        source_unknowns_explicit=True,
        IDs_outcomes_uncertainties_study_free_text_are_features=False,
        urine_collection_windows_are_not_predictors=True,
    )


def verify_source_proof(root, task=None, values=None):
    """Portable source/array policy verification without chemistry, fitting or unpickle."""
    check_dependency()
    root = Path(root)
    task = task or read(root / "task_manifest.json")["tasks"][0]
    values = values or rows(contained(root, task["table_path"]))
    descriptor = task["native_source_observation_coverage"]
    if (
        descriptor.get("covered") != 2
        or descriptor.get("total_original") != 8
        or descriptor.get("original_endpoint") != "CL_renal_absolute"
        or descriptor.get("included_original_record_ids")
        != ["NEW:07:40000000002573915", "NEW:07:40000000002573916"]
    ):
        raise ValueError("Native original-endpoint coverage is exactly two of eight")
    proof_path = contained(root, descriptor["proof_path"])
    if sha(proof_path) != descriptor["proof_sha256"]:
        raise ValueError("Frozen source proof changed")
    proof = read(proof_path)
    required = dict(
        schema=SCHEMA,
        task_id=TASK,
        records=22,
        identity_groups=5,
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        source_pins=PINS,
        target_text_portability_policy=TARGET_POLICY,
        annotation_only=True,
        validated_physiology=False,
        independent_holdout=False,
    )
    if (
        any(proof.get(k) != v for k, v in required.items())
        or task.get("task_id") != TASK
        or task.get("endpoint") != ENDPOINT
        or task.get("unit") != UNIT
        or task.get("n_records") != 22
        or task.get("n_groups") != 5
    ):
        raise ValueError("Frozen source-annotation scope changed")
    definition = task["scientific_definition"]
    if definition != task["task_definition"] or any(
        definition.get(k) != v
        for k, v in dict(
            canonical_quantity_id=QUANTITY,
            query_mode=MODE,
            context_fields=list(CONTEXT_FIELDS),
            implementation_sha256=sha(__file__),
            implementation_dependency_sha256=SHARED_SHA,
            mandatory_molecular_reporting=True,
            annotation_only=True,
            validated_physiology=False,
            independent_holdout=False,
            original_endpoint_observations_covered=2,
            original_endpoint_observations_total=8,
        ).items()
    ):
        raise ValueError("Typed query and original2/8 scope cannot be strengthened")
    if (
        task.get("validation_protocol") != "internal_grouped_cv_only_no_independent_holdout"
        or task.get("target_transform") != "log10"
    ):
        raise ValueError("Strict internal fivefold log-target protocol changed")
    for item in proof["pinned_artifacts"]:
        if sha(contained(root, item["path"])) != item["sha256"]:
            raise ValueError("Pinned staged dependency changed: " + item["path"])
    snapshots = root / "task_tables/source_snapshots"
    for name, expected in read(snapshots / "review.json")["source_input_sha256"].items():
        if sha(contained(snapshots / "dependencies", name)) != expected:
            raise ValueError("Staged independent source audit dependency changed")
    reconstructed, observations, identities = source_rows(snapshots)
    compare_source_rows(values, reconstructed)
    if (
        proof["row_hashes"] != {r["record_id"]: object_hash(r) for r in values}
        or read(root / "task_tables/frozen_training_rows.json") != values
    ):
        raise ValueError("Frozen row text/targets differ from immutable ledger")
    if (
        read(root / "task_tables/semantic_context_contract.json") != context_contract()
        or read(root / "task_tables/numeric_policy.json") != numeric_policy()
    ):
        raise ValueError("Staged context or numeric policy changed")
    feature = feature_manifest(root)
    if proof.get("feature_contract") != frozen_feature_contract(feature):
        raise ValueError("Frozen chemical/context artifacts, routes or methods changed")
    from peptide_modeling_features import _verify_artifacts

    _verify_artifacts(feature)
    if (
        len({r["split_group"] for r in values}) != 5
        or len({r["canonical_identity_group"] for r in values}) != 5
        or len(observations) != 22
        or len(identities) != 5
    ):
        raise ValueError("Do not invent independent molecules")
    if "nodefold_evidence" in proof:
        e = proof["nodefold_evidence"]
        receipt, baseline = (
            read(contained(root, e["path"])),
            contained(root, e["pre_node_proof_path"]),
        )
        if (
            sha(contained(root, e["path"])) != e["sha256"]
            or sha(baseline) != e["pre_node_proof_sha256"]
            or receipt.get("hostname") != "node23"
            or receipt.get("producer_sha256") != sha(__file__)
            or set(receipt.get("guard_counters", {})) != NOFIT_COUNTERS
            or any(receipt.get("guard_counters", {}).values())
            or receipt.get("source_proof_sha256") != sha(baseline)
            or receipt["actual_checks"] != read(root / "task_tables/predeclared_splits.json")
            or receipt["training_source_sha256"] != CORE_SHA
            or receipt["internal_source_sha256"] != INTERNAL_SHA
        ):
            raise ValueError("Actual node pre-fit proof changed")
    return proof


def frozen_feature_contract(feature):
    return dict(
        input_fingerprint=feature["input_fingerprint"],
        artifact_hashes=feature["artifact_hashes"],
        groups_sha256=object_hash(feature["groups"]),
        routes_sha256=object_hash(feature["routes"]),
        representation_methods_sha256=object_hash(feature["representation_methods"]),
        parent_pointer_sha256=PARENT_SHA,
    )


def script_closure(output):
    """Freeze only transitive repo-local Python imports, including explicit runner entries."""
    import ast

    pending = [
        Path(__file__).stem,
        "peptide_modeling_internal_cv",
        "peptide_modeling_campaign",
        "peptide_modeling_pipeline",
    ]
    seen, hashes = set(), {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        src = ROOT / "scripts" / (name + ".py")
        if not src.is_file():
            continue
        target = Path(output) / "scripts" / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        hashes["scripts/" + src.name] = sha(target)
        for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("peptide_modeling"))
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("peptide_modeling")
            ):
                pending.append(node.module)
    return hashes


def refresh_delivery(output):
    output = Path(output)
    task = read(output / "task_manifest.json")["tasks"][0]
    receipt = dict(
        schema=SCHEMA,
        status="node23_folds_prepared_not_trained"
        if task["scientific_definition"]["pretraining_fold_state"] == "node23_actual_no_fit_frozen"
        else "prepared_not_trained",
        task_id=TASK,
        records=22,
        identity_groups=5,
        models_fitted=0,
        chemical_feature_extractions=0,
        original_endpoint_observation_coverage=dict(covered=2, total=8),
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file() and p.name != "delivery_receipt.json" and "__pycache__" not in p.parts
        },
    )
    write(output / "delivery_receipt.json", receipt)
    return receipt


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if (output / "delivery_receipt.json").exists():
        raise FileExistsError("Preserve the existing prepared payload")
    values, observations, identities = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    table_path = "task_tables/" + TASK + ".tsv"
    table(output / table_path, values)
    values = rows(output / table_path)
    write(output / "task_tables/frozen_training_rows.json", values)
    write(output / "task_tables/predeclared_splits.json", split_prechecks(values))
    write(output / "task_tables/semantic_context_contract.json", context_contract())
    write(output / "task_tables/numeric_policy.json", numeric_policy())
    write(
        output / "task_tables/observation_context_evidence.json",
        [
            dict(
                record_id=r["record_id"],
                context=_context(r),
                evidence=json.loads(r["context_provenance_json"]),
                source_path=o["source_path"],
                source_sha256=o["source_sha256"],
                source_locator=o["source_locator"],
            )
            for r, o in zip(values, observations, strict=True)
        ],
    )
    names = snapshot_sources(source, output)
    names += [
        table_path,
        "task_tables/frozen_training_rows.json",
        "task_tables/predeclared_splits.json",
        "task_tables/semantic_context_contract.json",
        "task_tables/numeric_policy.json",
        "task_tables/observation_context_evidence.json",
    ]
    definition = dict(
        endpoint=ENDPOINT,
        canonical_quantity_id=QUANTITY,
        quantity_unit=UNIT,
        query_mode=MODE,
        context_fields=list(CONTEXT_FIELDS),
        context_group=CONTEXT_GROUP,
        annotation_only=True,
        validated_physiology=False,
        independent_holdout=False,
        identity_groups=5,
        independent_scaffold_or_source_validation_claimed=False,
        implementation_module=Path(__file__).stem,
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        mandatory_molecular_reporting=True,
        required_report_schema=SCHEMA,
        original_endpoint_observations_covered=2,
        original_endpoint_observations_total=8,
        target_text_portability_policy=TARGET_POLICY,
        pretraining_fold_state="awaiting_actual_node23_no_fit_receipt",
        warning=WARNING,
    )
    proof_path = "task_tables/renal_absolute_native_source_proof.json"
    proof = dict(
        schema=SCHEMA,
        task_id=TASK,
        records=22,
        identity_groups=5,
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        source_pins=PINS,
        annotation_only=True,
        validated_physiology=False,
        independent_holdout=False,
        target_text_portability_policy=TARGET_POLICY,
        row_hashes={r["record_id"]: object_hash(r) for r in values},
        pinned_artifacts=[dict(path=n, sha256=sha(output / n)) for n in names],
    )
    write(output / proof_path, proof)
    task = dict(
        task_id=TASK,
        endpoint=ENDPOINT,
        task_kind="regression",
        status="eligible",
        table_path=table_path,
        n_records=22,
        n_groups=5,
        unit=UNIT,
        target_transform="log10",
        inverse_transform="pow10",
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        requested_inner_folds=5,
        scientific_definition=definition,
        task_definition=definition,
        native_source_observation_coverage=dict(
            proof_path=proof_path,
            proof_sha256=sha(output / proof_path),
            original_endpoint="CL_renal_absolute",
            included_original_record_ids=[
                r["source_original_record_id"] for r in values if r["source_original_record_id"]
            ],
            covered=2,
            total_original=8,
        ),
    )
    for name in ["task_manifest.json", "targeted_task_manifest.json"]:
        write(output / name, dict(schema=SCHEMA, tasks=[task], models_fitted=0))
    table(output / "feature_rows_index.tsv", values)
    feature = slice_features(values, parent, output)
    config = dict(
        dataset="data/dataset_reclean_20261113_v1",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        additional_twenty_hour_loop=False,
        validation_protocol=task["validation_protocol"],
        mandatory_postprocessing="renal_absolute_native_precomputed_float64_controls_and_metadata_diagnostics",
        training=dict(
            routes=list(feature["routes"]),
            models=MODELS,
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
    for name in ["config.json", "recommended_training_config.json"]:
        write(output / name, config)
    closure = script_closure(output)
    write(output / "task_tables/import_closure.json", closure)
    proof["feature_contract"] = frozen_feature_contract(feature)
    proof["pinned_artifacts"].extend(
        dict(path=name, sha256=sha(output / name))
        for name in ["features/manifest.json", "task_tables/import_closure.json"]
    )
    write(output / proof_path, proof)
    task["native_source_observation_coverage"]["proof_sha256"] = sha(output / proof_path)
    for name in ["task_manifest.json", "targeted_task_manifest.json"]:
        write(output / name, dict(schema=SCHEMA, tasks=[task], models_fitted=0))
    verify_source_proof(output)
    write_report(output)
    (output / "README.md").write_text(
        WARNING + "\n\n22 source observations;5 exact graph/FAMILY components;"
        "10 chemical methods/13groups/15212D + "
        + str(WIDTH)
        + " source-context columns. Optuna128/8models/negativeMAE;internal5/noholdout/3seeds. "
        "Fixed float64 controls18fits/seed,54total. All22 selected-model finalfit; "
        "no consensus/external validation. Producer wrapper required. "
        "Stage source dependencies/policies undertask_tables. "
        "No production fitting during preparation or reporting.\n",
        encoding="utf-8",
    )
    return refresh_delivery(output)


def feature_manifest(root):
    from peptide_modeling_pipeline import portable_features

    return portable_features(root)


def preparation_review(prepared):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    root = Path(prepared)
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    proof = verify_source_proof(root, task, values)
    regenerated, _, _ = source_rows(root / "task_tables/source_snapshots")
    differences = compare_source_rows(values, regenerated)
    feature = feature_manifest(root)
    _verify_artifacts(feature)
    expected_context = np.vstack([encode_context(_context(r)) for r in values])
    routes = {}
    for route, names in feature["routes"].items():
        payload = load_feature_route(values, feature, route)
        if (
            names.count(CONTEXT_GROUP) != 1
            or not payload["valid"].all()
            or payload["X"].dtype != np.float32
            or payload["X"][:, -WIDTH:].tobytes() != expected_context.tobytes()
        ):
            raise ValueError("Actual route/context/NaN byte alignment changed")
        routes[route] = dict(
            shape=list(payload["X"].shape),
            matrix_sha256=hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest(),
            feature_names_sha256=object_hash(payload["feature_names"]),
        )
    closure = read(root / "task_tables/import_closure.json")
    for path, expected in closure.items():
        if sha(contained(root, path)) != expected:
            raise ValueError("Frozen staged import closure changed")
    policy = numeric_policy()
    from peptide_modeling_internal_cv import __file__ as internal_file
    from peptide_modeling_train import __file__ as core_file
    from peptide_modeling_train import _score

    if (
        sha(core_file) != CORE_SHA
        or sha(internal_file) != INTERNAL_SHA
        or _score(dict(mae=2.5), "regression") != -2.5
    ):
        raise ValueError("Actual core objective/runtime pin changed")
    return dict(
        schema=SCHEMA,
        task_id=TASK,
        records=22,
        identity_groups=5,
        source_proof_sha256=sha(
            contained(root, task["native_source_observation_coverage"]["proof_path"])
        ),
        task_table_sha256=sha(contained(root, task["table_path"])),
        row_hashes=proof["row_hashes"],
        routes=routes,
        context_width=WIDTH,
        context_matrix_sha256=hashlib.sha256(expected_context.tobytes()).hexdigest(),
        numeric_policy=policy,
        import_closure=closure,
        platform_log_text_differences=differences,
        frozen_targets_rewritten=False,
        models_fitted=0,
        chemical_feature_extractions=0,
    )


def nofit_nodefold_probe(prepared, receipt_output, *, stage_root=None, require_node=True):
    """Exercise real stage_round three times under fit/search/unpickle/RDKit guards."""
    import importlib.abc
    import platform
    import socket
    import sys
    import time

    hostname = socket.gethostname()
    if require_node and (
        hostname != "node23" or Path(sys.executable).resolve() != Path("/usr/bin/python3").resolve()
    ):
        raise ValueError("Node proof requires node23 /usr/bin/python3 -B")
    if not sys.flags.dont_write_bytecode or any(
        n == "rdkit" or n.startswith("rdkit.") for n in sys.modules
    ):
        raise ValueError("Use a fresh -B process without RDKit")
    counters = dict(
        estimator_fit_attempts=0,
        search_attempts=0,
        model_unpickle_attempts=0,
        chemistry_import_attempts=0,
        production_fits_launched=0,
    )

    class ChemistryBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "rdkit" or fullname.startswith("rdkit."):
                counters["chemistry_import_attempts"] += 1
                raise AssertionError("RDKit forbidden during no-fit node proof")

    def profile(frame, event, arg):
        if event != "call":
            return
        name, module = frame.f_code.co_name, frame.f_globals.get("__name__", "")
        if name in {
            "fit",
            "fit_transform",
            "partial_fit",
            "_fit_model",
            "_cv_predictions",
            "train_task",
            "train_internal_task",
            "_fit_cv_ranked_candidate",
        } and module.startswith(("sklearn", "lightgbm", "xgboost", "peptide_modeling")):
            counters["estimator_fit_attempts"] += 1
            raise AssertionError("Fit forbidden during no-fit node proof")
        if (module.startswith("optuna") and name in {"optimize", "create_study"}) or (
            module == "peptide_modeling_search" and name == "search_candidates"
        ):
            counters["search_attempts"] += 1
            raise AssertionError("Search forbidden during no-fit node proof")
        if module.startswith("joblib") and name == "load":
            counters["model_unpickle_attempts"] += 1
            raise AssertionError("Unpickle forbidden during no-fit node proof")

    blocker, previous = ChemistryBlocker(), sys.getprofile()
    sys.meta_path.insert(0, blocker)
    sys.setprofile(profile)
    started = time.time()
    try:
        import peptide_modeling_internal_cv as internal
        from peptide_modeling_campaign import stage_round, template_contract
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=1):
            root = Path(prepared).resolve()
            stage_root = Path(
                stage_root or Path(receipt_output).parent / "actual_staged_rounds"
            ).resolve()
            if stage_root == root or root in stage_root.parents or stage_root.exists():
                raise FileExistsError("Use a fresh isolated sibling stage directory")
            review = preparation_review(root)
            task = read(root / "task_manifest.json")["tasks"][0]
            proof_sha = task["native_source_observation_coverage"]["proof_sha256"]
            checks = split_prechecks(rows(root / task["table_path"]))
            configuration = read(root / "config.json")
            contract = template_contract(
                root, root / "config.json", root / "representation_methods.json"
            )
            staged = []
            for seed in SEEDS:
                run = stage_root / ("seed_" + str(seed))
                stage_round(
                    root,
                    run,
                    configuration,
                    seed,
                    stage_root / "durable" / run.name,
                    contract,
                    root / "representation_methods.json",
                )
                actual_review = preparation_review(run)
                actual_checks = split_prechecks(rows(run / task["table_path"]))
                if actual_checks != checks or actual_review != review:
                    raise ValueError("Staging changed source/features/policies or actual splits")
                # Verify report-only executes inside stage with all guards still active.
                report = write_report(run)
                if report["any_estimators_fitted_by_reporting"] != 0:
                    raise ValueError("Report-only attempted fitting")
                staged.append(
                    dict(
                        seed=seed,
                        run_dir=str(run),
                        source_review=actual_review,
                        actual_seed_plan=next(
                            x for x in actual_checks["seeds"] if x["seed"] == seed
                        ),
                        stage_contract_sha256=object_hash(contract),
                        stage_config_sha256=sha(run / "config.json"),
                        policy_sha256=sha(run / "task_tables/numeric_policy.json"),
                        source_proof_sha256=proof_sha,
                        report_only_estimator_fits=0,
                    )
                )
            if "nodefold_evidence" in verify_source_proof(root) and checks != read(
                root / "task_tables/predeclared_splits.json"
            ):
                raise ValueError("Actual runtime plans differ from pre-fit node freeze")
            if any(counters.values()):
                raise ValueError("Forbidden work attempted")
            receipt = dict(
                schema="renal_absolute_native_actual_node_stage_no_fit_v1",
                probe_scope="actual_node23_pretraining_no_fit"
                if require_node
                else "local_test_only",
                hostname=hostname,
                python_executable=sys.executable,
                python_version=list(sys.version_info[:3]),
                python_build=platform.python_build(),
                dont_write_bytecode=sys.flags.dont_write_bytecode,
                runtime=internal.runtime_contract(),
                producer_sha256=sha(__file__),
                helper_sha256=SHARED_SHA,
                training_source_sha256=CORE_SHA,
                internal_source_sha256=INTERNAL_SHA,
                source_proof_sha256=proof_sha,
                task_table_sha256=review["task_table_sha256"],
                task_id=TASK,
                payload_source_review=review,
                actual_checks=checks,
                actual_stage_rounds=staged,
                guard_counters=counters,
                report_only_estimator_fits=0,
                chemical_feature_extractions=0,
                frozen_targets_rewritten=False,
                annotation_only=True,
                validated_physiology=False,
                elapsed_seconds=time.time() - started,
                threadpools=[
                    {k: v for k, v in p.items() if k in {"internal_api", "num_threads", "version"}}
                    for p in threadpool_info()
                ],
            )
    finally:
        sys.setprofile(previous)
        sys.meta_path.remove(blocker)
    write(receipt_output, receipt)
    return receipt


def adopt_nodefold_receipt(prepared, receipt_path):
    root, receipt = Path(prepared), read(receipt_path)
    if (
        receipt.get("hostname") != "node23"
        or receipt.get("probe_scope") != "actual_node23_pretraining_no_fit"
        or receipt.get("producer_sha256") != sha(__file__)
        or receipt.get("training_source_sha256") != CORE_SHA
        or receipt.get("internal_source_sha256") != INTERNAL_SHA
        or receipt.get("dont_write_bytecode") != 1
        or receipt.get("python_executable") != "/usr/bin/python3"
        or receipt.get("report_only_estimator_fits") != 0
        or receipt.get("frozen_targets_rewritten") is not False
        or set(receipt.get("guard_counters", {})) != NOFIT_COUNTERS
        or any(receipt.get("guard_counters", {}).values())
        or len(receipt.get("actual_stage_rounds", [])) != 3
    ):
        raise ValueError("Only actual current three-seed zero-fit node evidence can be adopted")
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(root / task["table_path"])
    proof = verify_source_proof(root)
    source_path = task["native_source_observation_coverage"]["proof_path"]
    if receipt["source_proof_sha256"] != sha(root / source_path) or receipt[
        "task_table_sha256"
    ] != sha(root / task["table_path"]):
        raise ValueError("Node receipt belongs to another immutable payload")
    checks = receipt["actual_checks"]
    validate_plan_memberships(values, checks)
    baseline_path = "task_tables/pre_node_source_proof.json"
    shutil.copyfile(root / source_path, root / baseline_path)
    saved = "task_tables/node23_actual_no_fit_fold_receipt.json"
    if Path(receipt_path).resolve() != (root / saved).resolve():
        shutil.copyfile(receipt_path, root / saved)
    write(root / "task_tables/predeclared_splits.json", checks)
    for entry in proof["pinned_artifacts"]:
        entry["sha256"] = sha(contained(root, entry["path"]))
    proof["nodefold_evidence"] = dict(
        path=saved,
        sha256=sha(root / saved),
        pre_node_proof_path=baseline_path,
        pre_node_proof_sha256=sha(root / baseline_path),
        frozen_before_any_production_fit=True,
        selection_by_performance=False,
    )
    proof["pinned_artifacts"].extend(
        [
            dict(path=saved, sha256=sha(root / saved)),
            dict(path=baseline_path, sha256=sha(root / baseline_path)),
        ]
    )
    write(root / source_path, proof)
    task["native_source_observation_coverage"]["proof_sha256"] = sha(root / source_path)
    for field in ["scientific_definition", "task_definition"]:
        task[field]["pretraining_fold_state"] = "node23_actual_no_fit_frozen"
    for name in ["task_manifest.json", "targeted_task_manifest.json"]:
        write(
            root / name,
            dict(
                schema=SCHEMA,
                tasks=[task],
                models_fitted=0,
                stage="node23_folds_prepared_not_trained",
            ),
        )
    verify_source_proof(root)
    write_report(root)
    return refresh_delivery(root)


def validate_plan_memberships(values, checks):
    by_id = {r["record_id"]: r for r in values}
    if (
        [x["seed"] for x in checks["seeds"]] != list(SEEDS)
        or checks["distinct_molecular_partitions"] != 1
        or checks["independent_holdout"] is not False
    ):
        raise ValueError("Strict three-seed oneLOGO protocol changed")
    for seed in checks["seeds"]:
        seen = []
        if len(seed["folds"]) != 5:
            raise ValueError("Five actual folds required")
        for fold in seed["folds"]:
            a, b = fold["train_record_ids"], fold["validation_record_ids"]
            if (
                set(a) & set(b)
                or set(a) | set(b) != set(by_id)
                or len({by_id[i]["split_group"] for i in a}) != 4
                or len({by_id[i]["split_group"] for i in b}) != 1
            ):
                raise ValueError("Actual whole-molecule fold membership changed")
            seen.extend(b)
        if sorted(seen) != sorted(by_id):
            raise ValueError("Every source observation must have one OOF prediction")


def predict_contextual(model, chemical_X, contexts, *, quantity_id, source_annotation_mode):
    """Explicit annotation query only; never a unified physiological-clearance API."""
    if quantity_id != QUANTITY or source_annotation_mode != MODE:
        raise ValueError("Only the explicit source-annotation quantity is supported")
    chemistry = np.asarray(chemical_X, dtype=np.float32)
    if chemistry.ndim != 2 or len(chemistry) != len(contexts):
        raise ValueError("Exact full-graph and context rows must align")
    context = np.vstack([encode_context(c) for c in contexts])
    predictions = np.asarray(model.predict(np.column_stack((chemistry, context))))
    if predictions.shape != (len(contexts),) or not np.isfinite(predictions).all():
        raise ValueError("Finite source-annotation prediction required")
    return dict(
        target_space="log10_" + UNIT,
        predictions=predictions.tolist(),
        annotation_only=True,
        validated_physiology=False,
        quantity_id=QUANTITY,
        warning=WARNING,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in [
        "prepare",
        "report-only",
        "run-followup",
        "review-preparation",
        "nofit-nodefold-probe",
        "adopt-nodefold-receipt",
    ]:
        modes.add_argument("--" + flag, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--receipt-input", type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(output=args.output)
    elif args.report_only:
        result = write_report(args.output, run_dir=args.run_dir)
    elif args.run_followup:
        if args.config is None:
            parser.error("--config required")
        result = run_followup(args.config)
    elif args.review_preparation:
        result = preparation_review(args.output)
    elif args.adopt_nodefold_receipt:
        if args.receipt_input is None:
            parser.error("--receipt-input required")
        result = adopt_nodefold_receipt(args.output, args.receipt_input)
    else:
        if args.prepared_input is None or args.receipt_output is None:
            parser.error("Probe requires --prepared-input and --receipt-output")
        result = nofit_nodefold_probe(
            args.prepared_input, args.receipt_output, stage_root=args.stage_root
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "task_id",
                    "records",
                    "identity_groups",
                    "context_width",
                    "models_fitted",
                    "guard_counters",
                ]
                if k in result
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
