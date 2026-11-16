"""Independent stdlib audit of source-typed absolute-unit urinary-parent CLR.

Never imports the producer, chemistry, array/ML libraries or a model loader.
Call source gate before learning; final reporting additionally requires the
shared auditor's persisted-model proof plus released owner and durable files.
"""

import ast
import base64
import csv
import hashlib
import json
import math
import sqlite3
import struct
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

RA_TASK = "TASK_RENAL_ABSOLUTE_NATIVE_b0e47e12d9821d488c138588"
RA_SCHEMA = "renal_absolute_native_estimator_annotation_v1"
RA_ENDPOINT = "CL_renal_absolute_source_annotation"
RA_QUANTITY = "as_reported_urinary_parent_CLR_absolute_unit_estimator_annotation"
RA_UNIT = "L/h_as_reported_normalization_annotation"
RA_PRODUCER_SHA = "9917223fa959c2a711512f8ab19d9111697835c45e4b30146bf3447c341f24c4"
RA_HELPER_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
RA_CORE_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
RA_INTERNAL_SHA = "329cf60e16657eea737bdd1de2469fa8acf60dce4ab968d52859e9de609c1d27"
RA_PROOF_SHA = "b588fcd44dacbf723706a070dc4577fd3639234619333a860180311a60568513"
RA_PLAN_SHA = "67cba9c677580bbab91d8f0bbc6d47155a217368687b3449b9d33ba5727feec6"
RA_NODE_SHA = "a5acc20dbdcbf386027bd7e92d5cb5e57bef190d6a836e90ef496a2cf579ad5d"
RA_STAGE_SHA = "1fcd4338bb25caa44b182fb4c8ecbe3d2344e809f486023a4cc3fff1d265d10f"
RA_CONTEXT_SHA = "810b899b465c4ac316be58bc4e91196e1ea2d53e9e8ce5d9b49296be2ff6fb5a"
RA_POLICY_SHA = "5b197e7855f48c96b45ba7f3fdba1b01524fec22018f21796e1df68a6b4aefea"
RA_CLOSURE_SHA = "88b3105949d3431e9e6fce6ad8c7a87d1dcd75995b0d24af1302c9b0d0a2e364"
RA_MODELS = {
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
}
RA_ORIGINAL_IDS = ["NEW:07:40000000002573915", "NEW:07:40000000002573916"]
RA_CONTEXT_GROUP = "renal_absolute_annotation_context"
RA_NUMERIC = ("dose_value", "regimen_day", "dose_interval_hours", "CrCL_group_mean")
RA_MANDATORY = ("estimator_family", "estimator_detail", "normalization_annotation", "mean_kind")
RA_CONTROL_FILE = "renal_absolute_native_fixed_controls.json"
RA_REPORT_FILE = "reports/renal_absolute_native_molecular_diagnostics.json"


def is_task(task):
    return task.get("task_id") == RA_TASK


def ra_published_stage(root):
    path = Path(root) / "preparation_checks/node23_final_stage_closure_receipt_v4.json"
    if path.is_file():
        raw = path.read_bytes()
    else:
        receipt = globals().get("RA_TRANSPORTED_STAGE_RECEIPT")
        ra_require(
            isinstance(receipt, dict) and receipt.get("sha256") == RA_STAGE_SHA,
            "published post-stage1fcd receipt must be separately transported; never patch the frozen template",
        )
        raw = base64.b64decode(receipt["source_base64"], validate=True)
    ra_require(
        hashlib.sha256(raw).hexdigest() == RA_STAGE_SHA, "published final-v4 actual stage receipt"
    )
    return json.loads(raw)


def ra_context(o, categories):
    context = {k: o[k] for k in RA_MANDATORY}
    context.update(
        dose_unit_and_basis="mg",
        regimen="not_reported",
        renal_status="not_reported",
        CrCL_unit_and_basis="not_reported",
        **{k: None for k in RA_NUMERIC},
    )
    evidence = {k: [k] for k in RA_MANDATORY}
    drug = o["drug"]
    if drug == "daptomycin":
        ra_require(o["nominal_dose"] == "4 mg/kg TBW", "Dapt native dose")
        context.update(dose_value="4", dose_unit_and_basis="mg/kg_TBW")
        evidence["dose"] = ["nominal_dose"]
    elif drug == "BRII693_QPX9003_F365":
        header = o["dose_header"].split(" mg (", 1)
        ra_require(len(header) == 2 and header[0].isdigit(), "BRII exact native table dose")
        context.update(dose_value=header[0], regimen=o["regimen"])
        if o["regimen"] != "single_dose":
            context.update(
                regimen_day="7" if o["regimen"] == "q6h_7day_last_dose" else "14",
                dose_interval_hours="6",
            )
        evidence.update(dose=["dose_header"], regimen=["regimen"])
    elif drug == "caspofungin":
        ra_require(o["nominal_dose"] == "70 mg daily ×14", "Casp distinct StudyB dose")
        context.update(
            dose_value="70",
            regimen="daily_14day_last_dose",
            regimen_day="14",
            dose_interval_hours="24",
        )
        evidence["dose_and_regimen"] = ["nominal_dose", "source_locator"]
    elif drug == "desmopressin":
        ra_require(
            o["nominal_dose"] == "single 2 microgram desmopressin acetate"
            and o["CrCL_source_covariate_reference"] == "1.73m2",
            "Desmo native dose/CrCL basis",
        )
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
        evidence.update(
            dose=["nominal_dose"],
            CrCL_group_mean=[
                "CrCL_group_mean_native",
                "CrCL_source_covariate_reference",
                "source_locator",
            ],
            renal_status=["primary_sec7_group1_normal_group2_mild_group3_moderate_group4_severe"],
        )
    elif drug == "vancomycin":
        regimen = o["regimen"]
        ra_require(
            regimen
            in {"Vancomycin0.5g_alone", "Vancomycin0.5g_withceftazidime2g", "Vancomycin1.0g_alone"},
            "Vanco original native regimen",
        )
        context.update(
            dose_value="1000" if "1.0g" in regimen else "500",
            regimen="source_IV_with_ceftazidime2g"
            if "withceftazidime" in regimen
            else "source_IV_alone",
        )
        evidence["dose_and_regimen"] = ["regimen", "source_locator"]
    else:
        raise ValueError("Unapproved absolute-unit CLR parent")
    ra_require(set(context) == set(categories) | set(RA_NUMERIC), "full12field source context")
    for key, levels in categories.items():
        ra_require(context[key] in levels, "frozen categorical context level")
    return context, evidence


def ra_encode_context(context, contract):
    values = [
        float(context[field] == level)
        for field, levels in contract["categorical_levels"].items()
        for level in levels
    ]
    for field in RA_NUMERIC:
        value = context[field]
        ra_require(
            value is None
            or (not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0),
            "finite nonnegative native context or explicit missingness",
        )
        values.extend([math.nan if value is None else float(value), float(value is None)])
    ra_require(len(values) == 40, "exact40context columns")
    return list(struct.unpack("<40f", struct.pack("<40f", *values)))


def ra_reconstruct_rows(observations, identities, contract):
    rows, evidence_rows = [], []
    for o in observations:
        identity = identities[o["drug"]]
        ra_require(
            o["canonical_graph_key"] == identity["canonical_graph_key"]
            and o["known_FAMILY"] == identity["known_FAMILY"]
            and o["native_qualifier"] == "reported_table_central_point_no_inequality",
            "source graph/qualifier",
        )
        value = (
            Decimal(o["native_central_value"])
            * {"L/h": Decimal(1), "mL/min": Decimal("0.06")}[o["native_unit"]]
        )
        ra_require(
            value > 0 and value == Decimal(o["comparison_value_L_h_unit_conversion_only"]),
            "exact decimal unit-only conversion",
        )
        context, evidence = ra_context(o, contract["categorical_levels"])
        row = dict(
            record_id=o["review_observation_id"],
            endpoint=RA_ENDPOINT,
            task_id=RA_TASK,
            task_kind="regression",
            value=format(value, "f"),
            unit=RA_UNIT,
            target=format(math.log10(float(value)), ".15g"),
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
            semantic_context_key=ra_object_sha(context),
            semantic_context_json=json.dumps(
                context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            context_provenance_json=json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            source_observation_json=json.dumps(o, ensure_ascii=False, sort_keys=True),
            source_study=o["source_path"],
            source_grade="primary_table_source_annotation",
            readout_verification_status="source_parent_assignment_assay_specific_not_independent_chemical_intactness",
        )
        row.update({k: "not_reported" if v is None else str(v) for k, v in context.items()})
        rows.append(row)
        evidence_rows.append(
            dict(
                record_id=row["record_id"],
                context=context,
                evidence=evidence,
                source_path=o["source_path"],
                source_sha256=o["source_sha256"],
                source_locator=o["source_locator"],
            )
        )
    return rows, evidence_rows


def ra_source_target_text(frozen, regenerated):
    """Only source log text portability; it never validates model predictions."""
    ra_require(len(frozen) == len(regenerated), "source row count")
    differences = []
    for a, b in zip(frozen, regenerated, strict=True):
        ra_require(
            {k: v for k, v in a.items() if k != "target"}
            == {k: v for k, v in b.items() if k != "target"},
            "every non-target reconstructed source field must remain exact",
        )
        if a["target"] == b["target"]:
            continue
        x, y = float(a["target"]), float(b["target"])
        ra_require(
            math.isfinite(x) and math.isfinite(y) and b["target"] == format(y, ".15g"),
            "canonical source15g log text",
        )
        quantum = Decimal(1).scaleb(Decimal(b["target"]).adjusted() - 14)
        ulps = abs(
            struct.unpack("!q", struct.pack("!d", x))[0]
            - struct.unpack("!q", struct.pack("!d", y))[0]
        )
        ra_require(
            abs(Decimal(a["target"]) - Decimal(b["target"])) <= quantum
            and ulps <= 128
            and math.copysign(1, x) == math.copysign(1, y),
            "source-only portability bound exceeded",
        )
        differences.append(dict(record_id=a["record_id"], binary64_ULPs=ulps))
    return differences


def ra_verify_splits(rows, plan, contract):
    by_id = {r["record_id"]: r for r in rows}
    ra_require(
        len(by_id) == len(rows) == 22 and [c["seed"] for c in plan["seeds"]] == [42, 2026, 3407],
        "22unique rows/three seed prefix",
    )
    ra_require(
        plan["identity_groups"] == 5
        and plan["distinct_molecular_partitions"] == 1
        and plan["independent_holdout"] is False
        and plan["known_scaffold_groups_not_claimed"] is True
        and plan["strict_scaffold_validation_claimed"] is False,
        "single identity LOGO, no scaffold/holdout promotion",
    )
    partitions = set()
    for check in plan["seeds"]:
        seen, parts = [], []
        ra_require(check["actual_folds"] == len(check["folds"]) == 5, "five actual folds")
        for i, fold in enumerate(check["folds"]):
            a, b = fold["train_record_ids"], fold["validation_record_ids"]
            ra_require(
                fold["fold"] == i
                and len(set(a)) == len(a)
                and len(set(b)) == len(b)
                and not set(a) & set(b)
                and set(a) | set(b) == set(by_id),
                "actual fold partition",
            )
            ga, gb = (
                sorted({by_id[r]["split_group"] for r in a}),
                sorted({by_id[r]["split_group"] for r in b}),
            )
            ra_require(
                len(ga) == 4
                and len(gb) == 1
                and ga == fold["train_groups"]
                and gb == fold["validation_groups"]
                and not set(ga) & set(gb),
                "FAMILY membership/leakage",
            )
            ra_require(
                not (
                    {by_id[r]["canonical_identity_group"] for r in a}
                    & {by_id[r]["canonical_identity_group"] for r in b}
                ),
                "exact graph leakage",
            )
            known = {by_id[r]["semantic_context_key"] for r in a}
            ra_require(
                fold["unsupported_joint_context_record_ids"]
                == [r for r in b if by_id[r]["semantic_context_key"] not in known],
                "joint context support",
            )
            for field in list(contract["categorical_levels"]) + list(RA_NUMERIC):
                observed = {json.loads(by_id[r]["semantic_context_json"])[field] for r in a}
                unsupported = []
                for rid in b:
                    value = json.loads(by_id[rid]["semantic_context_json"])[field]
                    if field in contract["categorical_levels"]:
                        supported = value in observed
                    elif value is None:
                        supported = None in observed
                    else:
                        numeric = [float(v) for v in observed if v is not None]
                        supported = bool(numeric) and min(numeric) <= float(value) <= max(numeric)
                    if not supported:
                        unsupported.append(rid)
                criterion = (
                    "seen_level"
                    if field in contract["categorical_levels"]
                    else "missingness_seen_or_training_numeric_range_not_validation"
                )
                ra_require(
                    fold["per_field_context_support"][field]
                    == dict(unsupported_record_ids=unsupported, criterion=criterion),
                    "fold-local categorical/numeric support",
                )
            ra_require(len({by_id[r]["target"] for r in a}) > 1, "nonconstant training target")
            seen.extend(b)
            parts.append(tuple(sorted(b)))
        ra_require(Counter(seen) == Counter(by_id.keys()), "each source OOF exactly once")
        partitions.add(tuple(sorted(parts)))
    ra_require(len(partitions) == 1, "three seeds do not create independent LOGO partitions")


def verify_prepared(root, scripts_root=None, require_published_stage=True):
    root = Path(root)
    manifest = ra_read(root / "task_manifest.json")
    ra_require(
        manifest["schema"] == RA_SCHEMA and len(manifest["tasks"]) == 1,
        "approved absolute source manifest",
    )
    task = manifest["tasks"][0]
    d = task["native_source_observation_coverage"]
    ra_require(
        is_task(task)
        and task["endpoint"] == RA_ENDPOINT
        and task["unit"] == RA_UNIT
        and task["n_records"] == 22
        and task["n_groups"] == 5
        and task["target_transform"] == "log10"
        and task["validation_protocol"] == "internal_grouped_cv_only_no_independent_holdout",
        "typed22/5 internal task",
    )
    ra_require(
        d
        == dict(
            proof_path="task_tables/renal_absolute_native_source_proof.json",
            proof_sha256=RA_PROOF_SHA,
            original_endpoint="CL_renal_absolute",
            included_original_record_ids=RA_ORIGINAL_IDS,
            covered=2,
            total_original=8,
        ),
        "original coverage exactly2/8",
    )
    ra_require(ra_sha(root / d["proof_path"]) == RA_PROOF_SHA, "frozen source proof")
    proof = ra_read(root / d["proof_path"])
    ra_require(
        proof["schema"] == RA_SCHEMA
        and proof["task_id"] == RA_TASK
        and proof["implementation_sha256"] == RA_PRODUCER_SHA
        and proof["implementation_dependency_sha256"] == RA_HELPER_SHA
        and proof["records"] == 22
        and proof["identity_groups"] == 5
        and proof["annotation_only"] is True
        and proof["validated_physiology"] is False
        and proof["independent_holdout"] is False,
        "source annotation scope/producer pins",
    )
    for item in proof["pinned_artifacts"]:
        ra_require(
            ra_sha(ra_path(root, item["path"])) == item["sha256"],
            "staged source dependency changed: " + item["path"],
        )
    for relative, expected in [
        ("task_tables/predeclared_splits.json", RA_PLAN_SHA),
        ("task_tables/semantic_context_contract.json", RA_CONTEXT_SHA),
        ("task_tables/numeric_policy.json", RA_POLICY_SHA),
        ("task_tables/import_closure.json", RA_CLOSURE_SHA),
    ]:
        ra_require(ra_sha(root / relative) == expected, "frozen plan/context/numeric/import policy")
    contract = ra_read(root / "task_tables/semantic_context_contract.json")
    policy = ra_read(root / "task_tables/numeric_policy.json")
    ra_require(
        contract["mandatory_semantic_fields"] == list(RA_MANDATORY)
        and contract["quantity_id"] == RA_QUANTITY
        and contract["width"] == 40
        and contract["numeric_fields"] == list(RA_NUMERIC)
        and contract["IDs_outcomes_uncertainties_study_free_text_are_features"] is False
        and contract["CrCL_reference_is_separate_from_drug_CLR"] is True,
        "only40 predeclared source-bound channels; no outcomes/IDs",
    )
    definition = task["scientific_definition"]
    ra_require(
        definition == task["task_definition"]
        and definition["canonical_quantity_id"] == RA_QUANTITY
        and definition["annotation_only"] is True
        and definition["validated_physiology"] is False
        and definition["independent_holdout"] is False
        and definition["mandatory_molecular_reporting"] is True
        and definition["implementation_sha256"] == RA_PRODUCER_SHA
        and definition["original_endpoint_observations_covered"] == 2
        and definition["original_endpoint_observations_total"] == 8,
        "mandatory reporting and qualified coverage limits",
    )
    snapshots = root / "task_tables/source_snapshots"
    for name, expected in proof["source_pins"].items():
        ra_require(ra_sha(ra_path(snapshots, name)) == expected, "independent admission source pin")
    review = ra_read(snapshots / "review.json")
    for name, expected in review["source_input_sha256"].items():
        ra_require(
            ra_sha(ra_path(snapshots / "dependencies", name)) == expected,
            "full independent source closure",
        )
    observations, identities = (
        ra_read(snapshots / "observations.json"),
        ra_read(snapshots / "identity_union_review.json"),
    )
    ra_require(
        review["verdict"] == "QUALIFIED_APPROVE_SOURCE_ANNOTATION_ONLY"
        and len(observations) == 22
        and len(identities) == 5
        and Counter(o["drug"] for o in observations)
        == {
            "daptomycin": 2,
            "BRII693_QPX9003_F365": 12,
            "caspofungin": 1,
            "desmopressin": 4,
            "vancomycin": 3,
        },
        "exclude extra drugs/EBEs/subintervals",
    )
    rows = ra_rows(root / task["table_path"])
    ra_require(
        {r["record_id"]: ra_object_sha(r) for r in rows} == proof["row_hashes"]
        and ra_read(root / "task_tables/frozen_training_rows.json") == rows,
        "immutable row/target bytes",
    )
    rebuilt, evidence = ra_reconstruct_rows(observations, identities, contract)
    portability = ra_source_target_text(rows, rebuilt)
    ra_require(
        ra_read(root / "task_tables/observation_context_evidence.json") == evidence,
        "dose/regimen/renal evidence binding",
    )
    ra_require(
        [r["source_original_record_id"] for r in rows if r["source_original_record_id"]]
        == RA_ORIGINAL_IDS,
        "only original Dapt2, extra20 source augmentation",
    )
    ra_require(
        len({r["canonical_identity_group"] for r in rows})
        == len({r["split_group"] for r in rows})
        == 5,
        "five graph/FAMILY components",
    )
    for row in rows:
        ra_require(
            row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest(),
            "exact reference graph bytes",
        )
    plan = ra_read(root / "task_tables/predeclared_splits.json")
    ra_verify_splits(rows, plan, contract)
    node = ra_read(root / proof["nodefold_evidence"]["path"])
    ra_require(
        ra_sha(root / proof["nodefold_evidence"]["path"]) == RA_NODE_SHA
        and node["hostname"] == "node23"
        and node["python_executable"] == "/usr/bin/python3"
        and node["dont_write_bytecode"] == 1
        and node["actual_checks"] == plan
        and node["producer_sha256"] == RA_PRODUCER_SHA
        and node["training_source_sha256"] == RA_CORE_SHA
        and node["internal_source_sha256"] == RA_INTERNAL_SHA
        and node["payload_source_review"]["row_hashes"] == proof["row_hashes"],
        "actual pre-fit node plan/source/runtime",
    )
    guard_names = {
        "estimator_fit_attempts",
        "search_attempts",
        "model_unpickle_attempts",
        "chemistry_import_attempts",
        "production_fits_launched",
    }
    ra_require(
        set(node["guard_counters"]) == guard_names
        and all(type(v) is int and v == 0 for v in node["guard_counters"].values()),
        "prefit zero-fit/unpickle/chemistry guard",
    )
    closure = ra_read(root / "task_tables/import_closure.json")
    ra_require(
        len(closure) == 33
        and closure["scripts/peptide_modeling_train.py"] == RA_CORE_SHA
        and closure["scripts/peptide_modeling_internal_cv.py"] == RA_INTERNAL_SHA
        and closure["scripts/peptide_modeling_renal_absolute_native_bridge.py"] == RA_PRODUCER_SHA,
        "33module immutable import closure",
    )
    for relative, expected in closure.items():
        path = (
            ra_path(root, relative)
            if scripts_root is None
            else ra_path(scripts_root, Path(relative).name)
        )
        ra_require(ra_sha(path) == expected, "actual script closure changed: " + relative)
    feature = ra_read(root / "features/manifest.json")
    actual_contract = dict(
        input_fingerprint=feature["input_fingerprint"],
        artifact_hashes=feature["artifact_hashes"],
        groups_sha256=ra_object_sha(feature["groups"]),
        routes_sha256=ra_object_sha(feature["routes"]),
        representation_methods_sha256=ra_object_sha(feature["representation_methods"]),
        parent_pointer_sha256=feature["parent_pointer_sha256"],
    )
    ra_require(
        actual_contract == proof["feature_contract"],
        "frozen chemical/context route artifact schema",
    )
    if require_published_stage:
        stage = ra_published_stage(root)
        ra_require(
            stage["source_proof_sha256"] == RA_PROOF_SHA
            and stage["actual_checks"] == plan
            and stage["producer_sha256"] == RA_PRODUCER_SHA
            and stage["runtime"] == node["runtime"]
            and stage["hostname"] == "node23"
            and stage["guard_counters"] == node["guard_counters"]
            and [s["seed"] for s in stage["actual_stage_rounds"]] == [42, 2026, 3407],
            "actual final-v4 three-stage no-fit closure",
        )
        for actual, check in zip(stage["actual_stage_rounds"], plan["seeds"], strict=True):
            ra_require(
                actual["actual_seed_plan"] == check
                and actual["source_proof_sha256"]
                == actual["source_review"]["source_proof_sha256"]
                == RA_PROOF_SHA
                and actual["source_review"]["row_hashes"] == proof["row_hashes"]
                and actual["source_review"]["routes"] == node["payload_source_review"]["routes"]
                and actual["report_only_estimator_fits"] == 0,
                "each frozen staged source/array/plan bound",
            )
    return dict(
        status="source_verified_annotation_only_not_training_completion",
        root=str(root),
        task=task,
        proof=proof,
        rows=rows,
        observations=observations,
        identities=identities,
        plan=plan,
        node=node,
        context_contract=contract,
        numeric_policy=policy,
        feature=feature,
        import_closure=closure,
        source_log_portability_differences=portability,
    )


def ra_feature_cache(root, feature):
    base = str(feature["cache_dir"]).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return ra_path(Path(root) / "features", base)


def ra_design(root, source, route):
    feature = source["feature"]
    cache = ra_feature_cache(root, feature)
    for name, expected in feature["artifact_hashes"].items():
        ra_require(ra_sha(ra_path(cache, name)) == expected, "immutable feature artifact changed")
    methods = feature["representation_methods"]
    ra_require(
        methods["independent_method_count"] == 10
        and methods["chemical_groups"] == 13
        and methods["chemical_dimensions"] == 15212,
        "10chemical methods/13groups/15212D",
    )
    groups = feature["routes"][route]
    ra_require(
        groups[-1] == RA_CONTEXT_GROUP and groups.count(RA_CONTEXT_GROUP) == 1,
        "append exactly40semantic channels once",
    )
    matrix, names = [[] for _ in source["rows"]], []
    for group_name in groups:
        group = feature["groups"][group_name]
        shape, data = ra_npy(ra_path(cache, group["matrix"]))
        valid_shape, valid = ra_npy(ra_path(cache, group["valid"]))
        ra_require(
            list(shape) == group["shape"]
            and len(shape) == 2
            and valid_shape == (shape[0],)
            and shape[1] == len(group["feature_names"]),
            "primitive NPY shape/names/masks",
        )
        idx = {r["key"]: int(r["row_idx"]) for r in ra_rows(ra_path(cache, group["index"]))}
        ra_require(len(idx) == shape[0], "unique frozen feature keys")
        for i, row in enumerate(source["rows"]):
            key = (
                row["record_id"]
                if group["key_kind"] == "record_id"
                else row["canonical_identity_group"]
            )
            pos = idx[key]
            ra_require(0 <= pos < shape[0] and valid[pos], "all22 feature records valid")
            raw = data[pos * shape[1] : (pos + 1) * shape[1]]
            part = list(
                struct.unpack(
                    "<" + str(len(raw)) + "f", struct.pack("<" + str(len(raw)) + "f", *raw)
                )
            )
            matrix[i].extend(part)
            if group_name == RA_CONTEXT_GROUP:
                encoded = ra_encode_context(
                    json.loads(row["semantic_context_json"]), source["context_contract"]
                )
                ra_require(
                    group["feature_names"] == source["context_contract"]["feature_names"]
                    and ra_array_sha(part, "f") == ra_array_sha(encoded, "f"),
                    "source-derived categorical/numeric/missingness byte equality",
                )
        names.extend(group_name + ":" + name for name in group["feature_names"])
    frozen = source["node"]["payload_source_review"]["routes"][route]
    ra_require(
        frozen["shape"] == [22, len(names)]
        and frozen["matrix_sha256"] == ra_array_sha(matrix, "f")
        and frozen["feature_names_sha256"] == ra_object_sha(names),
        "actual node loaded route bytes/names",
    )
    return matrix, names


def verify_source(entry, task, api=None):
    ra_require(
        entry["protocol"] == "internal5_no_holdout" and is_task(task),
        "registry native internal-only protocol",
    )
    ra_require(
        ra_sha(entry["manifest_path"]) == entry["manifest_sha256"], "registered manifest pin"
    )
    root = Path(entry["manifest_path"]).parent
    source = verify_prepared(root)
    frozen = source["task"]
    ra_require(
        task["table_relative"] == frozen["table_path"]
        and task["records"] == 22
        and task["table_sha256"] == ra_sha(root / frozen["table_path"])
        and task["scientific_definition"] == frozen["scientific_definition"]
        and task["native_source_observation_coverage"]
        == frozen["native_source_observation_coverage"],
        "registered source/task qualified definition",
    )
    return source


def ra_verify_stage_contract(run, source, seed):
    run = Path(run)
    staged = ra_read(run / "campaign_input_contract.json")
    files = staged["template_contract"]["files"]
    required = {item["path"]: item["sha256"] for item in source["proof"]["pinned_artifacts"]}
    required[source["task"]["table_path"]] = ra_sha(run / source["task"]["table_path"])
    required["task_tables/renal_absolute_native_source_proof.json"] = RA_PROOF_SHA
    required.update(source["import_closure"])
    cache = ra_feature_cache(run, source["feature"])
    for name, expected in source["feature"]["artifact_hashes"].items():
        required[(cache / name).relative_to(run).as_posix()] = expected
    ra_require(
        staged["seed"] == seed and staged["config_sha256"] == ra_sha(run / "config.json"),
        "actual staged seed/config binding",
    )
    for name, expected in required.items():
        ra_require(
            files.get(name) == ra_sha(ra_path(run, name)) == expected,
            "source/import/array closure not in actual learning-time stage contract: " + name,
        )
    return len(required)


def ra_sum(values):
    # NumPy's C-contiguous, non-fast-axis0 accumulation is sequential per column.
    total = 0.0
    for value in values:
        total += value
    return total


def ra_transform(train, predict):
    """Reconstruct fixed median/indicator/keep-empty + uncentered float64 scaling.

    Only arithmetic on stored encoded values. No estimator is constructed/fitted.
    Returns actual public-Ridge-boundary arrays, not an estimate of coefficients.
    """
    ra_require(
        train and predict and len({len(r) for r in train + predict}) == 1,
        "fixed transform matrix shape",
    )
    width, n = len(train[0]), len(train)
    medians, indicators = [], []
    for j in range(width):
        available = sorted(r[j] for r in train if not math.isnan(r[j]))
        ra_require(all(math.isfinite(v) for v in available), "finite raw values or IEEE NaN")
        if len(available) != n:
            indicators.append(j)
        k = len(available)
        medians.append(
            0.0
            if not k
            else available[k // 2]
            if k % 2
            else ra_sum(available[k // 2 - 1 : k // 2 + 1]) / 2.0
        )

    def fill(rows):
        result = []
        for row in rows:
            ra_require(
                all(math.isfinite(v) or math.isnan(v) for v in row), "no infinite encoded values"
            )
            result.append(
                [medians[j] if math.isnan(v) else v for j, v in enumerate(row)]
                + [float(math.isnan(row[j])) for j in indicators]
            )
        return result

    a, b = fill(train), fill(predict)
    scales = []
    eps = 2.0**-52
    for j in range(len(a[0])):
        mean = ra_sum(row[j] for row in a) / n
        residuals = [row[j] - mean for row in a]
        correction = ra_sum(residuals)
        unnormalized = ra_sum(v * v for v in residuals) - correction * correction / n
        variance = unnormalized / n
        bound = n * eps * variance + (n * mean * eps) ** 2
        scales.append(1.0 if variance <= bound else math.sqrt(variance))
    return [[v / scales[j] for j, v in enumerate(row)] for row in a], [
        [v / scales[j] for j, v in enumerate(row)] for row in b
    ]


def ra_boundary(X, y=None):
    result = dict(X_dtype="float64", X_shape=[len(X), len(X[0])], X_sha256=ra_array_sha(X))
    if y is not None:
        result.update(y_dtype="float64", y_shape=[len(y)], y_sha256=ra_array_sha(y))
    return result


def ra_verify_controls(rows, matrix, check, controls):
    ids = [r["record_id"] for r in rows]
    positions = {rid: i for i, rid in enumerate(ids)}
    designs = dict(
        molecule_only=[r[:-40] for r in matrix],
        context_only=[r[-40:] for r in matrix],
        molecule_plus_context=matrix,
    )
    ra_require(set(controls) == set(designs), "three predeclared fixed float64 controls")
    y = [float(r["target"]) for r in rows]
    for name, X in designs.items():
        c = controls[name]
        ra_require(
            c["model"] == "fixed_Ridge_alpha1_LSQR_not_searched"
            and c["dtype_from_initial_boundary"] == "float64"
            and c["source_encoded_values_rewritten"] is False
            and c["same_frozen_inner_fivefold"] is True
            and c["source_value_origin"] == "upcast_exact_existing_encoded_float32_values"
            and c["prediction_record_ids"] == ids
            and c["matrix_sha256"] == ra_array_sha(X)
            and c["control_fit_calls"] == 6
            and c["selected_model_fit_calls"] == c["search_calls"] == 0,
            "fixed-control source/dtype/membership/fit-count policy",
        )
        ra_require(
            ra_numeric_equal(c["metrics"], ra_metrics(rows, c["predictions"])),
            "actual fixed-control OOF metrics",
        )
        ra_require(
            c["full_cohort_evaluation_role"]
            == "training_inputs_only_no_independent_performance_score",
            "all22 control fit is not independent evaluation",
        )
        traces = c["per_fold_fit_trace"]
        ra_require(len(traces) == 6, "actual5OOF+1all22 trace per design")
        for i, trace in enumerate(traces):
            train_ids = check["folds"][i]["train_record_ids"] if i < 5 else ids
            predict_ids = check["folds"][i]["validation_record_ids"] if i < 5 else ids
            a, b = [X[positions[r]] for r in train_ids], [X[positions[r]] for r in predict_ids]
            target = [y[positions[r]] for r in train_ids]
            pred = (
                [c["predictions"][positions[r]] for r in predict_ids]
                if i < 5
                else c["all_cohort_fit_predictions"]
            )
            ra_require(
                len(pred) == len(b) and all(math.isfinite(float(p)) for p in pred),
                "finite actual control predictions",
            )
            transformed_a, transformed_b = ra_transform(a, b)
            required = dict(
                fold=i if i < 5 else -1,
                fit_role="inner_training_fold" if i < 5 else "all_internal_cohort_control_fit",
                train_record_ids=train_ids,
                predict_record_ids=predict_ids,
                actual_estimator="Ridge",
                actual_parameters=dict(
                    alpha=1.0,
                    solver="lsqr",
                    imputer_strategy="median",
                    imputer_add_indicator=True,
                    imputer_keep_empty_features=True,
                    scaler_with_mean=False,
                ),
                actual_fit_X_dtype="float64",
                actual_fit_y_dtype="float64",
                actual_predict_X_dtype="float64",
                actual_prediction_dtype="float64",
                actual_coef_dtype="float64",
                fit_X_sha256=ra_array_sha(a),
                fit_y_sha256=ra_array_sha(target),
                predict_X_sha256=ra_array_sha(b),
                predictions_sha256=ra_array_sha(pred),
                actual_public_Ridge_calls=dict(fit=1, predict=1),
                actual_coef_shape=[len(transformed_a[0])],
                transformed_Ridge_fit_boundary=ra_boundary(transformed_a, target),
                transformed_Ridge_predict_boundary=ra_boundary(transformed_b),
                control_fit_calls=1,
                selected_model_fit_calls=0,
                search_calls=0,
            )
            changed = [k for k, v in required.items() if not ra_numeric_equal(trace.get(k), v)]
            ra_require(
                not changed,
                "exact raw/transformed64 trace changed: "
                + name
                + ":"
                + str(i)
                + ":"
                + ",".join(changed),
            )
            digest = trace.get("actual_coef_sha256", "")
            ra_require(
                len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "actual float64 coefficient hash",
            )
            duplicate = defaultdict(list)
            for row, p in zip(b, pred, strict=True):
                duplicate[ra_array_sha(row)].append(float(p))
            ra_require(
                all(
                    all(math.isclose(p, values[0], rel_tol=1e-7, abs_tol=1e-8) for p in values)
                    for values in duplicate.values()
                ),
                "strict float64 same-input control consistency; no f32/source-log exception",
            )
    return 18


def ra_require(condition, message):
    if not condition:
        raise ValueError("Renal annotation audit: " + message)


def ra_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ra_object_sha(value, *, compact=True):
    kwargs = {"separators": (",", ":")} if compact else {}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, **kwargs).encode("utf-8")
    ).hexdigest()


def ra_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ra_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def ra_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    ra_require(path.is_relative_to(root), "artifact escapes its root")
    return path


def ra_npy(path):
    """Read primitive little-endian, C-order arrays without numpy/pickle."""
    data = Path(path).read_bytes()
    ra_require(data[:6] == b"\x93NUMPY", "invalid NPY")
    major = data[6]
    ra_require(major in (1, 2, 3), "NPY version")
    size = 2 if major == 1 else 4
    header_size = int.from_bytes(data[8 : 8 + size], "little")
    start = 8 + size
    header = ast.literal_eval(data[start : start + header_size].decode("utf-8"))
    ra_require(header["fortran_order"] is False, "C-order NPY required")
    code = {"<f4": "f", "<f8": "d", "|b1": "?", "|u1": "B"}.get(header["descr"])
    ra_require(code is not None, "primitive float/bool NPY only")
    shape = header["shape"]
    ra_require(
        isinstance(shape, tuple)
        and 1 <= len(shape) <= 2
        and all(type(v) is int and v > 0 for v in shape),
        "NPY shape",
    )
    body = data[start + header_size :]
    count = math.prod(shape)
    ra_require(len(body) == count * struct.calcsize(code), "NPY body length")
    values = struct.unpack("<" + str(count) + code, body)
    return (shape, values)


def ra_array_sha(matrix, dtype="d"):
    flat = (
        [v for row in matrix for v in row]
        if matrix and isinstance(matrix[0], (list, tuple))
        else matrix
    )
    return hashlib.sha256(struct.pack("<" + str(len(flat)) + dtype, *flat)).hexdigest()


def ra_metrics(rows, prediction):
    ra_require(
        len(prediction) == len(rows) and all(math.isfinite(float(v)) for v in prediction),
        "finite prediction for every observation",
    )
    counts = Counter(r["split_group"] for r in rows)
    e = [float(r["target"]) - float(p) for r, p in zip(rows, prediction, strict=True)]
    w = [1 / counts[r["split_group"]] for r in rows]
    return {
        "records": len(rows),
        "groups": len(counts),
        "record_weighted": {
            "mae": math.fsum(abs(v) for v in e) / len(e),
            "rmse": math.sqrt(math.fsum(v * v for v in e) / len(e)),
        },
        "molecular_equal": {
            "mae": math.fsum((abs(v) * z for v, z in zip(e, w, strict=True))) / math.fsum(w),
            "rmse": math.sqrt(
                math.fsum((v * v * z for v, z in zip(e, w, strict=True))) / math.fsum(w)
            ),
        },
    }


def ra_numeric_equal(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all((ra_numeric_equal(actual[k], v) for k, v in expected.items()))
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(ra_numeric_equal(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if isinstance(expected, float):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11)
        )
    return type(actual) is type(expected) and actual == expected


def ra_consistency(rows, predictions, matrix, provenance):
    """Recompute the narrow predeclared binary32 LSQR guard; do not alter OOF."""
    positions = {r["record_id"]: i for i, r in enumerate(rows)}
    buckets = defaultdict(list)
    for prediction in predictions:
        row = rows[positions[prediction["record_id"]]]
        buckets[
            row["canonical_identity_group"], row["semantic_context_key"], int(prediction["fold"])
        ].append(prediction)
    output = []
    for bucket in buckets.values():
        ix = [positions[p["record_id"]] for p in bucket]
        ra_require(
            len({ra_array_sha(matrix[i], "f") for i in ix}) == 1,
            "same identity/context/fold requires exact same route bytes",
        )
        values = [float(p["prediction"]) for p in bucket]
        ra_require(all(math.isfinite(v) for v in values), "finite selected predictions")
        if len(set(values)) == 1:
            output.append({"mode": "exact", "raw_predictions_unchanged": True})
            continue
        required = {
            "selected_model": "linear",
            "actual_estimator": "Ridge",
            "actual_solver": "lsqr",
            "actual_coef_dtype": "float32",
            "actual_scaler_with_mean": False,
            "actual_imputer_strategy": "median",
            "actual_add_indicator": True,
            "input_dtype": "float32",
            "feature_width": len(matrix[0]),
            "full_matrix_sha256": ra_array_sha(matrix, "f"),
            "training_source_sha256": RA_CORE_SHA,
        }
        ra_require(
            all((provenance.get(k) == v for k, v in required.items())),
            "drift requires actual pinned float32 Ridge LSQR provenance",
        )
        bits = []
        for value in values:
            ra_require(
                abs(value) >= 2.0 ** (-126)
                and struct.unpack("<f", struct.pack("<f", value))[0] == value,
                "drift values must be exact normal binary32",
            )
            bits.append(struct.unpack("<I", struct.pack("<f", value))[0])
        ra_require(
            len({math.copysign(1, v) for v in values}) == 1 and max(bits) - min(bits) <= 32,
            "selected drift exceeds predeclared 32ULP guard",
        )
        output.append(
            {
                "mode": "exact_normal_binary32_actual_LSQR_at_most32ULP",
                "observed_binary32_ULPs": max(bits) - min(bits),
                "maximum_ULPs": 32,
                "engineering_guard_not_a_universal_BLAS_bound": True,
                "raw_predictions_unchanged": True,
            }
        )
    return output


def ra_cv_score(metrics):
    """Frozen37bd regression model selection maximizes negative MAE."""
    return -float(metrics["mae"])


def ra_verify_trials(run, result, settings, admission):
    """Reconcile all128 attempt receipts, candidate caches and read-only Optuna DB."""
    search = result["search"]
    ra_require(
        search.get("backend") == "optuna" and search.get("status") == "completed",
        "actual completed Optuna search required",
    )
    contract = search["contract"]
    namespace = ra_object_sha(contract)
    run = Path(run)
    directory = ra_path(run, search["search_dir"])
    ra_require(
        namespace == search["namespace"]
        and directory.name == namespace[:24]
        and (contract["input_signature"] == result["input_hash"])
        and (contract["backend"] == "optuna")
        and (contract["seed"] == result["seed"])
        and (contract["task_kind"] == "regression")
        and (ra_read(directory / "contract.json") == contract),
        "Optuna namespace/seed/input contract",
    )
    ra_require(
        contract["implementation_sha256"] == ra_sha(run / "scripts/peptide_modeling_search.py"),
        "actual search implementation pin",
    )
    pairs = {(route, model) for route in settings["routes"] for model in settings["models"]}
    ra_require(
        set(map(tuple, contract["pairs"])) == pairs and len(contract["pairs"]) == len(pairs) == 88,
        "all11routes x8models must participate",
    )
    stored_result = ra_read(directory / "search_result.json")
    ra_require(
        {k: v for k, v in stored_result.items() if k != "trials"} == search,
        "published search result differs from selected result",
    )
    records = []
    candidate_dir = Path(result["artifacts"]["model"]).parent / "candidate_cv"
    expected_cache = set()
    for number, path in enumerate(sorted((directory / "trials").glob("*.json"))):
        wrapped = ra_read(path)
        value = wrapped["evaluation"]
        ra_require(
            wrapped["record_sha256"] == ra_object_sha(value)
            and value["search_trial"] == number
            and (path.name == f"{number:06d}.json")
            and (value["search_input_signature"] == namespace),
            "trial receipt content/number/input pin",
        )
        ra_require(
            (value["route"], value["model"]) in pairs
            and value["status"] in {"completed", "failed"},
            "terminal declared candidate",
        )
        if value["status"] == "completed":
            ra_require(
                value["actual_folds"] == value["requested_folds"] == 5
                and value["input_hash"] == result["input_hash"]
                and math.isfinite(value["score"])
                and math.isclose(
                    value["score"], ra_cv_score(value["metrics"]), rel_tol=1e-10, abs_tol=1e-11
                ),
                "successful candidate has real fivefold regression score",
            )
        else:
            ra_require(
                value.get("score") is None and bool(value.get("reason")),
                "failed candidate cannot gain a numeric fallback score",
            )
        key = ra_object_sha(
            [result["input_hash"], value["route"], value["model"], value["params"]], compact=False
        )
        cache_path = candidate_dir / (key + ".json")
        if cache_path.exists():
            cached = ra_read(cache_path)
            ra_require(
                all((value.get(k) == v for k, v in cached.items())), "candidate cache changed"
            )
            expected_cache.add(cache_path.name)
        else:
            ra_require(value["status"] == "failed", "successful candidate cache missing")
        records.append(value)
    ra_require(
        len(records) == 128 == settings["search_budget"]
        and search["budget"] == {"requested": 128, "used": 128, "remaining": 0, "pairs": 88}
        and (stored_result["trials"] == records)
        and ({p.name for p in candidate_dir.glob("*.json")} == expected_cache)
        and (
            set(map(tuple, search["coverage_attempted"]))
            == {(r["route"], r["model"]) for r in records}
            == pairs
        ),
        "complete128attempts/all88pairs/no unrecorded candidate cache",
    )
    ra_require(
        search["completed"] == sum(v["status"] == "completed" for v in records)
        and search["failed"] == sum(v["status"] == "failed" for v in records),
        "actual successful/failed counts",
    )
    database = ra_path(run, search["storage_path"])
    ra_require(
        database == directory / "study.sqlite3"
        and search["study_name"] == "peptide_" + namespace[:24],
        "Optuna study database path/name",
    )
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        study_names = connection.execute("SELECT study_name FROM studies").fetchall()
        stored = connection.execute(
            "SELECT t.number,t.state,a.value_json FROM trials t LEFT JOIN trial_user_attributes a ON a.trial_id=t.trial_id AND a.key='evaluation_result' ORDER BY t.number"
        ).fetchall()
        objectives = connection.execute(
            "SELECT t.number,v.value FROM trials t JOIN trial_values v ON v.trial_id=t.trial_id WHERE t.state='COMPLETE' ORDER BY t.number"
        ).fetchall()
    ra_require(
        study_names == [(search["study_name"],)]
        and len(stored) == len(records)
        and all(
            (
                number == i
                and state == ("COMPLETE" if records[i]["status"] == "completed" else "FAIL")
                and (raw is not None)
                and (json.loads(raw) == records[i])
                for i, (number, state, raw) in enumerate(stored)
            )
        ),
        "SQLite evaluations/statuses differ from trial receipts",
    )
    ra_require(
        objectives
        == [(v["search_trial"], v["score"]) for v in records if v["status"] == "completed"],
        "SQLite objective values differ from actual candidate scores",
    )
    ra_require(
        admission.get("policy") == "internal_cv_rank_then_full_internal_cohort_fit_and_persistence"
        and admission.get("validation_labels") == "all_internal_cv_records_no_independent_holdout"
        and (admission.get("holdout_used") is False),
        "internal final-fit admission policy",
    )

    def candidate_key(record):
        return ra_object_sha([record["route"], record["model"], record["params"]], compact=False)

    rejected = set()
    for record in admission["rejected_candidates"]:
        matches = [
            v
            for v in records
            if v["status"] == "completed"
            and all(v[k] == record[k] for k in ("route", "model", "params"))
            and (v["score"] == record["inner_cv_score"])
        ]
        ra_require(
            matches
            and bool(record.get("reason"))
            and (
                record.get("stage")
                in {
                    "estimator_construction",
                    "outer_train_fit",
                    "training_probe_persistence",
                    "development_oof_reconstruction",
                }
            ),
            "rejected full-fit candidate lacks a real successful CV attempt",
        )
        rejected.add(candidate_key(record))
    selected = {
        "route": result["selected_route"],
        "model": result["selected_model"],
        "params": result["selected_params"],
    }
    key = candidate_key(selected)
    admitted = [
        r for r in records if r["status"] == "completed" and candidate_key(r) not in rejected
    ]
    winners = [r for r in admitted if candidate_key(r) == key]
    ra_require(
        winners
        and key == admission["selected_candidate_key"]
        and (winners[0]["score"] == max(r["score"] for r in admitted))
        and math.isclose(
            winners[0]["score"], result["internal_cv_selection_score"], rel_tol=1e-10, abs_tol=1e-11
        ),
        "selected model is not the highest admitted successful CV candidate",
    )
    return {"attempts": 128, "pairs": 88, "backend": "optuna", "sqlite_verified": True}


def ra_files(root):
    root = Path(root)
    ra_require(root.is_dir() and (not root.is_symlink()), "regular inventory root required")
    result = {}
    for path in sorted(root.rglob("*")):
        ra_require(not path.is_symlink(), "inventory refuses symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = ra_sha(path)
    return result


def ra_owner_released(identity, api):
    ra_require(
        isinstance(identity, dict)
        and type(identity.get("pid")) is int
        and (type(identity.get("start_ticks")) is int),
        "owner identity required",
    )
    state = api.identity_state(identity, api.process_identity(identity["pid"]))
    ra_require(
        state in {"exited", "pid_reused"}, "original/recovery owner is still live or unverified"
    )


def ra_no_live_owned_workers(entry, api):
    proc = Path(getattr(api, "renal_proc_root", "/proc"))
    ra_require(proc.is_dir(), "Linux process inventory required for final owner release")
    prefixes = [str(Path(entry[k])).replace("\\", "/") + "/" for k in ("template", "output_root")]
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        identity = api.process_identity(int(directory.name))
        if identity is None or identity.get("state") == "Z":
            continue
        ra_require(
            not any(
                arg.replace("\\", "/").startswith(prefix)
                for arg in identity.get("argv", [])
                for prefix in prefixes
            ),
            "live worker still owns renal input/run paths",
        )


def ra_verify_learning_archive(journal, before_files, output, durable):
    receipt = journal["archive"]
    path = ra_path(durable, receipt["path"])
    ra_require(
        ra_sha(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"],
        "original durable learning archive changed",
    )

    def reporting_member(name):
        relative = name.split("/", 1)[1] if "/" in name else ""
        return relative in {RA_CONTROL_FILE, RA_REPORT_FILE}

    expected = {
        name: sha
        for name, sha in before_files["output"].items()
        if name.startswith("round_") and not reporting_member(name)
    }
    archived = {}
    prefix = Path(output).name + "/"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        named = {m.name: m for m in members}
        ra_require(len(named) == len(members), "duplicate learning archive member")
        for member in members:
            ra_require(
                not member.issym()
                and (member.isfile() or member.isdir() or member.islnk())
                and (not PurePosixPath(member.name).is_absolute())
                and (".." not in PurePosixPath(member.name.replace("\\", "/")).parts),
                "learning archive must not contain symlinks/devices/path escapes",
            )
            target, visited = (member, set())
            while target.islnk():
                ra_require(
                    target.name not in visited
                    and target.linkname in named
                    and target.linkname.startswith(prefix)
                    and (".." not in PurePosixPath(target.linkname.replace("\\", "/")).parts),
                    "hardlink must resolve inside the same archive without cycles",
                )
                visited.add(target.name)
                target = named[target.linkname]
            if member.islnk():
                ra_require(target.isfile(), "hardlink target must be a regular archived file")
            if (member.isfile() or member.islnk()) and member.name.startswith(prefix):
                relative = member.name[len(prefix) :]
                if relative.startswith("round_"):
                    if reporting_member(relative):
                        continue
                    ra_require(relative not in archived, "duplicate learning archive member")
                    with archive.extractfile(target) as handle:
                        archived[relative] = hashlib.sha256(handle.read()).hexdigest()
    ra_require(
        archived == expected,
        "every original learned seed/source/model/trial byte must be durably archived",
    )


def verify_seed(root, run, seed):
    """Renal supplement only: caller must also audit Optuna/persistence/owner/durable."""
    source = verify_prepared(root)
    actual = verify_prepared(run, scripts_root=Path(run) / "scripts", require_published_stage=False)
    ra_require(
        actual["task"] == source["task"] and actual["rows"] == source["rows"],
        "runtime source changed",
    )
    run = Path(run).resolve()
    staged_artifacts = ra_verify_stage_contract(run, source, seed)
    result_path = run / "results/internal_cv" / (RA_TASK + ".json")
    result = ra_read(result_path)
    ra_require(
        result["status"] == "completed"
        and result["task_id"] == RA_TASK
        and (result["endpoint"] == source["task"]["endpoint"])
        and (result["target_unit"] == source["task"]["unit"])
        and (result["target_transform"] == "log10")
        and (result["seed"] == seed)
        and (result["actual_inner_folds"] == result["requested_inner_folds"] == 5)
        and (result["protocol"] == "internal_grouped_cv_only_no_independent_holdout")
        and (result["independent_holdout_available"] is False)
        and (result["n_internal_cv_rows"] == 22)
        and (result["n_internal_cv_groups"] == 5),
        "actual all22 internal fivefold result",
    )
    for name, receipt in result["artifact_hashes"].items():
        ra_require(ra_sha(ra_path(run, name)) == receipt["sha256"], "fitted artifact changed")
    paths = {k: ra_path(run, v) for k, v in result["artifacts"].items()}
    ra_require(
        all(str(p) in result["artifact_hashes"] for p in paths.values()),
        "all artifacts hash-bound",
    )
    contract, inputs = (ra_read(paths["model_contract"]), ra_read(paths["input_contract"]))
    ids = [r["record_id"] for r in source["rows"]]
    ra_require(
        contract["fit_scope"] == "all_internal_cv_rows"
        and contract["protocol"] == result["protocol"]
        and (contract["actual_inner_folds"] == contract["requested_inner_folds"] == 5)
        and (contract["independent_holdout_available"] is False)
        and (contract["target_unit"] == result["target_unit"])
        and (contract["target_transform"] == result["target_transform"])
        and (contract["record_ids"] == ids)
        and (contract["task"] == source["task"] == inputs["task"])
        and (contract["input_hash"] == inputs["input_hash"] == result["input_hash"])
        and (inputs["source_rows_hash"] == ra_object_sha(source["rows"], compact=False)),
        "all22 selected final fit and input contract",
    )
    for key in ["selected_route", "selected_model", "selected_params"]:
        ra_require(contract[key] == result[key], "selected model contract")
    admission = ra_read(paths["candidate_fit_admission"])
    ra_require(
        contract["selection_fit_admission"] == admission, "model final-fit admission binding"
    )
    matrix, names = ra_design(run, source, result["selected_route"])
    selected = inputs["features"][result["selected_route"]]
    ra_require(
        selected["names"] == contract["feature_names"] == names
        and selected["matrix_sha256"] == ra_array_sha(matrix, "f")
        and (selected["valid_sha256"] == hashlib.sha256(b"\x01" * 22).hexdigest()),
        "actual selected full route bytes",
    )
    settings = inputs["config"]
    ra_require(
        set(settings["models"]) == RA_MODELS
        and settings["seed"] == seed
        and (settings["search_backend"] == "optuna")
        and (settings["search_budget"] == 128)
        and (settings["inner_folds"] == 5)
        and (settings["n_jobs"] == 1),
        "8 models, Optuna128 and frozen resource/seed policy",
    )
    features = ra_read(run / "features/manifest.json")
    ra_require(
        set(inputs["features"]) == set(features["routes"]) == set(settings["routes"]),
        "all10 methods plus fusion in actual training signature",
    )
    for route, record in inputs["features"].items():
        X, route_names = ra_design(run, source, route)
        ra_require(
            record
            == {
                "names": route_names,
                "matrix_sha256": ra_array_sha(X, "f"),
                "valid_sha256": hashlib.sha256(b"\x01" * 22).hexdigest(),
            },
            "all route inputs must match predeclared matrix bytes",
        )
    signature = {
        "protocol": result["protocol"],
        "task": source["task"],
        "config": settings,
        "rows": [(r["record_id"], r["target"], r["split_group"]) for r in source["rows"]],
        "features": inputs["features"],
        "feature_fingerprint": features["input_fingerprint"],
        "runtime": result["runtime"],
    }
    ra_require(
        result["input_hash"] == ra_object_sha(signature, compact=False)
        and contract["runtime"] == inputs["runtime"] == result["runtime"],
        "actual learning input signature",
    )
    ra_require(
        result["runtime"] == source["node"]["runtime"], "actual frozen node learning runtime"
    )
    search_evidence = ra_verify_trials(run, result, settings, admission)
    check = next(c for c in source["plan"]["seeds"] if c["seed"] == seed)
    oof = ra_rows(paths["internal_oof"])
    ra_require([r["record_id"] for r in oof] == ids, "OOF all22/order/once")
    folds = {rid: f["fold"] for f in check["folds"] for rid in f["validation_record_ids"]}
    for row, prediction in zip(source["rows"], oof, strict=True):
        ra_require(
            prediction["split_group"] == row["split_group"]
            and int(prediction["fold"]) == folds[row["record_id"]]
            and (float(prediction["target"]) == float(row["target"])),
            "actual frozen OOF membership/target",
        )
    controls_path = run / RA_CONTROL_FILE
    controls = ra_read(controls_path)
    ra_require(
        controls["source_proof_sha256"] == RA_PROOF_SHA
        and controls["seed"] == seed
        and (controls["selected_result_sha256"] == ra_sha(result_path))
        and (controls["frozen_check_sha256"] == ra_object_sha(check))
        and (
            controls["selected_estimator_provenance"]["actual_model_artifact_sha256"]
            == ra_sha(paths["model"])
        ),
        "control-to-source/model/result binding",
    )
    provenance = controls["selected_estimator_provenance"]
    ra_require(
        provenance.get("selected_model") == result["selected_model"]
        and provenance.get("input_dtype") == "float32"
        and (provenance.get("feature_width") == len(names))
        and (provenance.get("full_matrix_sha256") == ra_array_sha(matrix, "f"))
        and (provenance.get("training_source_sha256") == RA_CORE_SHA),
        "actual selected estimator input/source provenance",
    )
    fits = ra_verify_controls(source["rows"], matrix, check, controls["controls"])
    report_path = run / RA_REPORT_FILE
    report = ra_read(report_path)
    ra_require(
        report["schema"] == RA_SCHEMA
        and report["task_id"] == RA_TASK
        and report["records"] == 22
        and report["identity_groups"] == 5
        and report["annotation_only"] is True
        and report["validated_physiology"] is False
        and report["known_scaffold_groups_not_claimed"] is True,
        "typed qualified molecular report scope",
    )
    ra_require(
        report["status"] == "verified_internal_annotation_only"
        and report["seed"] == seed
        and (report["controls_sha256"] == ra_sha(controls_path))
        and (report["fixed_ridge_controls"] == controls["controls"])
        and (report["result_input_hash"] == result["input_hash"])
        and (
            report["selected_models_fitted_by_reporting"]
            == report["any_estimators_fitted_by_reporting"]
            == 0
        ),
        "report-only controls provenance",
    )
    diagnostic = report["diagnostics"]
    expected = ra_metrics(source["rows"], [float(r["prediction"]) for r in oof])
    ra_require(
        all(
            ra_numeric_equal(
                result["internal_cv_selection_metrics"][metric],
                expected["record_weighted"][metric],
            )
            for metric in ("mae", "rmse")
        ),
        "selection result metrics disagree with actual OOF",
    )
    ra_require(
        all((ra_numeric_equal(diagnostic[k], v) for k, v in expected.items()))
        and diagnostic["fold_context_support"] == check["folds"]
        and (diagnostic["annotation_only"] is True)
        and (diagnostic["validated_physiology"] is False)
        and (diagnostic["independent_holdout"] is False)
        and (diagnostic["source_grade_used_only_for_reporting"] is True)
        and (diagnostic["raw_predictions_mutated"] is False)
        and (diagnostic["BRII_record_dominance"] == {"records": 12, "denominator": 22}),
        "actual OOF metrics/support/interpretation",
    )
    ra_require(
        diagnostic["prediction_consistency"]
        == ra_consistency(source["rows"], oof, matrix, controls["selected_estimator_provenance"]),
        "actual selected prediction precision proof",
    )
    for field in [
        *source["context_contract"]["categorical_levels"],
        *RA_NUMERIC,
        "source_grade",
        "readout_verification_status",
    ]:
        labels = sorted({r[field] for r in source["rows"]})
        ra_require(set(diagnostic["strata"][field]) == set(labels), "all provenance strata")
        for label in labels:
            ix = [i for i, r in enumerate(source["rows"]) if r[field] == label]
            metrics = ra_metrics(
                [source["rows"][i] for i in ix], [float(oof[i]["prediction"]) for i in ix]
            )
            ra_require(
                ra_numeric_equal(diagnostic["strata"][field][label], metrics), "stratum metrics"
            )
    molecule_errors = {
        g: ra_metrics(
            [r for r in source["rows"] if r["split_group"] == g],
            [
                float(oof[i]["prediction"])
                for i, r in enumerate(source["rows"])
                if r["split_group"] == g
            ],
        )
        for g in sorted({r["split_group"] for r in source["rows"]})
    }
    ra_require(
        ra_numeric_equal(diagnostic["per_molecule_errors"], molecule_errors)
        and diagnostic["independent_sources_or_scaffolds_validated"] is False
        and diagnostic["original_endpoint_observation_coverage"]
        == dict(
            original_endpoint="CL_renal_absolute",
            covered=2,
            total_original=8,
            included_original_record_ids=RA_ORIGINAL_IDS,
            all_original_eight_covered=False,
        ),
        "per-molecule errors and original2/8 coverage cannot be strengthened",
    )
    return {
        "status": "renal_source_and_numeric_reporting_verified_not_campaign_completion",
        "seed": seed,
        "control_fits": fits,
        "learning_stage_artifacts_verified": staged_artifacts,
        "search": search_evidence,
        "selected_fits_by_auditor": 0,
        "models_unpickled_by_auditor": 0,
        "required_external_gates": [
            "same_host_model_persistence",
            "exclusive_owner_released",
            "durable_source_model_trials_reports_and_campaign_journal",
        ],
    }


def verify_reporting(entry, journal, task, api):
    source = verify_source(entry, task, api)
    cfg = ra_read(entry["config_path"])
    ra_require(
        journal["status"] == "completed"
        and journal.get("renal_absolute_annotation_reporting")
        == dict(status="verified", task_id=RA_TASK)
        and [r["seed"] for r in journal["rounds"]] == [42, 2026, 3407]
        and cfg["required_seeds"] == [42, 2026, 3407]
        and cfg.get("workers", 1) == 1
        and cfg.get("additional_twenty_hour_loop") is False
        and cfg["output_root"] == entry["output_root"]
        and cfg["template_run"] == entry["template"],
        "three completed learned seeds alone do not satisfy mandatory reporting",
    )
    ra_require(
        ra_sha(entry["config_path"]) == entry["config_sha256"] == journal["config_sha256"],
        "registered controller config unchanged",
    )
    ra_require(
        journal["pid"] == entry["owned_process_identity"]["pid"]
        and journal["process_identity"]["start_ticks"]
        == entry["owned_process_identity"]["start_ticks"],
        "frozen original controller journal identity changed",
    )
    ra_owner_released(entry["owned_process_identity"], api)
    ra_no_live_owned_workers(entry, api)
    reports, verified = [], []
    for index, r in enumerate(journal["rounds"]):
        seed = r["seed"]
        run = ra_path(entry["output_root"], r["run_dir"])
        ra_require(
            run.name == f"round_{index:03d}_seed_{seed}"
            and r["status"] == "completed"
            and r["audit"]["status"] == "verified"
            and r.get("requirements_met") is True
            and r["config_sha256"] == ra_sha(run / "config.json"),
            "actual seed namespace/config/terminal state",
        )
        base = api.check_seed({**entry, "tasks": [task]}, r, seed)
        ra_require(
            base.get(RA_TASK, {}).get("status") == "completed_verified",
            "same-host persisted-model/fold verification is mandatory",
        )
        verified.append(verify_seed(entry["template"], run, seed))
        reports.append(ra_read(run / RA_REPORT_FILE))
        result_path = run / "results/internal_cv" / (RA_TASK + ".json")
        result = ra_read(result_path)
        durable = Path(cfg["durable_root"]) / run.name
        for path in [
            result_path,
            *[ra_path(run, p) for p in result["artifacts"].values()],
            run / RA_CONTROL_FILE,
            run / RA_REPORT_FILE,
            run / "internal_verification.json",
        ]:
            ra_require(
                ra_sha(path) == ra_sha(ra_path(durable, path.relative_to(run))),
                "fitted artifact/control/report durable byte equality",
            )
    campaign_path = Path(entry["output_root"]) / "renal_absolute_native_campaign_diagnostics.json"
    campaign = ra_read(campaign_path)
    ra_require(
        campaign["schema"] == RA_SCHEMA
        and campaign["status"] == "verified_internal_annotation_only"
        and campaign["reports"] == reports
        and campaign["additional_twenty_hour_loop"] is False
        and campaign["warning"] == source["task"]["scientific_definition"]["warning"]
        and ra_sha(campaign_path) == ra_sha(Path(cfg["durable_root"]) / campaign_path.name)
        and ra_sha(entry["journal_path"])
        == ra_sha(Path(cfg["durable_root"]) / "internal_campaign_status.json"),
        "three mandatory molecular reports/campaign/journal durable closure",
    )
    file_map = {}
    for r in journal["rounds"]:
        run = ra_path(entry["output_root"], r["run_dir"])
        file_map.update({run.name + "/" + name: digest for name, digest in ra_files(run).items()})
    ra_verify_learning_archive(
        journal, {"output": file_map}, entry["output_root"], cfg["durable_root"]
    )
    return dict(
        status="verified",
        task_id=RA_TASK,
        required_seed_reports=3,
        source_records_per_seed=22,
        identity_groups=5,
        fixed_float64_control_traces=54,
        all22_final_internal_fits_verified=3,
        source_annotation_only=True,
        validated_physiology=False,
        original_observations_covered=2,
        original_observations_total=8,
        all_original_eight_covered=False,
        selected_models_fitted_by_auditor=0,
        models_unpickled_by_auditor=0,
        seed_checks=verified,
    )


def coverage(entry, journal, task, state, api):
    verify_source(entry, task, api)
    result = dict(
        original_endpoint="CL_renal_absolute",
        canonical_task_id=RA_TASK,
        quantity_id=RA_QUANTITY,
        source_proof_verified=True,
        source_annotation_only=True,
        original_labelled_records=8,
        included_original_records=2,
        included_original_record_ids=RA_ORIGINAL_IDS,
        all_original_records_covered=False,
        source_augmentation_records=20,
        training_pool_records=22,
        training_pool_identity_groups=5,
        validated_physiology=False,
        uniform_actual_body_normalization_validated=False,
        independent_scaffold_source_or_external_validation=False,
        state="source_annotation_two_of_eight_observations_training_pending",
        required_seed_final_training_verified=0,
    )
    if state == "trained_three_seeds":
        verify_reporting(entry, journal, task, api)
        result.update(
            state="source_annotation_two_of_eight_observations_trained",
            required_seed_final_training_verified=3,
        )
    return result


def transport_payload():
    data = Path(__file__).read_bytes()
    stage = Path(__file__).resolve().parents[1] / (
        "data/peptide_modeling_renal_absolute_native_followup_20261115_v1/"
        "preparation_checks/node23_final_stage_closure_receipt_v4.json"
    )
    raw = stage.read_bytes()
    ra_require(
        hashlib.sha256(raw).hexdigest() == RA_STAGE_SHA, "local immutable published stage receipt"
    )
    return dict(
        source_base64=base64.b64encode(data).decode(),
        sha256=hashlib.sha256(data).hexdigest(),
        published_stage_receipt=dict(
            source_base64=base64.b64encode(raw).decode(), sha256=RA_STAGE_SHA
        ),
    )


def load_transport(payload):
    data = base64.b64decode(payload["source_base64"], validate=True)
    ra_require(
        hashlib.sha256(data).hexdigest() == payload["sha256"],
        "absolute-native adapter transport hash",
    )
    stage = payload["published_stage_receipt"]
    raw = base64.b64decode(stage["source_base64"], validate=True)
    ra_require(
        stage["sha256"] == hashlib.sha256(raw).hexdigest() == RA_STAGE_SHA,
        "published stage evidence transport hash",
    )
    namespace = {
        "__name__": "frozen_stdlib_renal_absolute_native_audit",
        "RA_TRANSPORTED_STAGE_RECEIPT": stage,
    }
    exec(compile(data, "<frozen_stdlib_renal_absolute_native_audit>", "exec"), namespace)
    return SimpleNamespace(**namespace)
