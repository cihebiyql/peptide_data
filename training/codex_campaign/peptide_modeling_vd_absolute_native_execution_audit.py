"""Independent stdlib Vd annotation execution adapter; no producer or learner imports.

The caller passes the shared auditor as ``api``.  This adapter supplements its
artifact/trial, process-identity and durable-archive primitives; it never fits,
deserializes a model, changes source files or declares completion from a status.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
from collections import Counter, defaultdict
from decimal import Decimal
from functools import wraps
from pathlib import Path

TASK = "TASK_VD_ABSOLUTE_NATIVE_83c52b8c0afb06f712070e93"
SCHEMA = "vd_absolute_native_typed_annotation_v1"
QUANTITY = "native_source_absolute_volume_annotation_given_reported_estimand"
MODULE = "peptide_modeling_vd_absolute_native_bridge.py"
PRODUCER_SHA = "77113b69b655bd13d208c285dbc477468d806a48b71038f7c4236fe145e35cb3"
PROOF_SHA = "ec5b1d3e62780d1237d8cfe96f642a5f55bba0abae44fe2b20c4df7e82b6add4"
CLOSURE_SHA = "35f993947db5614fa4d8c09635fb465fc424b9ce7c3e0970c361c2f6656209da"
POST_RECEIPT_SHA = "7b0b5d128391684e7fa9ecad10391a21b8a5413b1f23f390219e39d929808ded"
LAUNCH_CONTRACT_SHA = "db46d0839cf2a53457d61c9a65da224a3a1811700ef5ffaf0fd8d182f2bac5d7"
LAUNCH_RECEIPT_SHA = "d225108b2c5e0c81eacaaff3fb7d59f58305952ec697feb08772a288e648b12d"
LAUNCH_PREFLIGHT_SHA = "0f9f8ef78991b532afcbaf5675152f40d4715d1117c0a4a754d1c980c5594fdd"
LAUNCH_CONFIG_SHA = "35494d5ef6e857fa524b154f9e926168b88dd5c347550b857c89e07a4f42e59a"
PREPARED_DELIVERY_SHA = "f2f2fab702eb4bb0e518f32acbbeab42c359f58275d224492cf3f6a7ea040cf1"
HELPER_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
SOURCE_SHA = "83c52b8c0afb06f712070e93374949c3b072a38727a86eb7633291064b4ccf6e"
REVIEW_SHA = "452e2dfc0058f46fb7c1135774951dadee8571b34570693160a8629d9be026bd"
CORRECTIONS_SHA = "56b0e9ed3852926aee12efade3f4ce8c2ddd93e4b78d5efc598357aebc3e8269"
TABLE_SHA = "462cf2f301b0529b2e5601bdf8c10b66e0ae518f243df25ee8cbee428512213b"
ONTOLOGY_SHA = "97a5c528e8db326fa418ad5c9b967138b5ef8d6f6103aeefda9fc646ea47759d"
CONTEXT = "vd_absolute_native_context"
SEEDS = (42, 2026, 3407)
MODELS = {
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
}
CATEGORICAL = (
    "estimand",
    "statistic_basis",
    "reported_F_basis",
    "steady_state_modifier",
    "baseline_modifier",
    "route",
    "species",
    "matrix",
    "population",
    "sex",
    "formulation",
    "dose_status",
    "dose_unit",
    "frequency_status",
    "visit_status",
    "reference_covariate_status",
)
NUMERIC = (
    "dose_min_reported_unit",
    "dose_max_reported_unit",
    "frequency_per_day",
    "visit_day",
    "dose_level_count",
    "dose_level_1",
    "dose_level_2",
    "dose_level_3",
    "dose_level_4",
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
PARAMETERS = dict(
    alpha=1.0,
    solver="lsqr",
    imputer_strategy="median",
    imputer_add_indicator=True,
    imputer_keep_empty_features=True,
    scaler_with_mean=False,
)


def require(value, message):
    if not value:
        raise ValueError("Vd native audit: " + message)


def audit_errors(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError:
            raise
        except (
            OSError,
            KeyError,
            TypeError,
            IndexError,
            ArithmeticError,
            sqlite3.Error,
            struct.error,
            StopIteration,
        ) as exc:
            raise ValueError("Vd native audit: malformed or incomplete evidence") from exc

    return checked


def same(actual, expected):
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, (float, int)) and not isinstance(actual, bool) and actual == expected
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(same(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(same(a, b) for a, b in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def object_sha(value, *, compact=True):
    kwargs = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def is_task(task):
    return task.get("task_id") == TASK


def target_text(frozen, value):
    concentration = Decimal(value)
    require(concentration.is_finite() and concentration > 0, "nonpositive/nonfinite raw L value")
    expected = format(math.log10(float(concentration)), ".15g")
    a, b = Decimal(frozen), Decimal(expected)
    x, y = float(frozen), float(expected)
    require(
        a.is_finite()
        and b.is_finite()
        and math.isfinite(x)
        and math.isfinite(y)
        and format(x, ".15g") == frozen,
        "noncanonical/nonfinite frozen log10 text",
    )
    if frozen == expected:
        return
    require(a and b and math.copysign(1, x) == math.copysign(1, y), "zero/sign target drift")

    def ordered(v):
        n = struct.unpack("!Q", struct.pack("!d", v))[0]
        return 2**63 - (n & (2**63 - 1)) if n >> 63 else 2**63 + n

    require(
        abs(a - b) <= Decimal(10) ** (min(a.adjusted(), b.adjusted()) - 14)
        and abs(ordered(x) - ordered(y)) <= 128,
        "derived target exceeds .15g/binary64 text bound",
    )


def corrected_context(record, correction):
    require(
        correction["source_contract_sha256"] == SOURCE_SHA
        and correction["original_frozen_context"]
        == {
            k: record[k] for k in ("frozen_route", "frozen_species", "frozen_matrix", "frozen_dose")
        },
        "independent context correction no longer binds source",
    )
    c = correction["typed_context"]
    mapping = dict(
        estimand="source_estimand",
        statistic_basis="statistic_basis",
        reported_F_basis="source_F_basis",
        steady_state_modifier="source_steady_state",
        baseline_modifier="baseline_adjustment",
        route="route",
        species="species",
        matrix="matrix",
        population="population",
        sex="sex",
        formulation="formulation",
        dose_status="dose_status",
        dose_unit="dose_unit",
    )
    q = {k: c[v] for k, v in mapping.items()}
    q.update(
        reference_covariate_status=c.get("reference_covariate_status", "not_reported"),
        frequency_status=c["frequency"],
        visit_status="explicit_day" if c["day"] is not None else "not_reported",
        frequency_per_day=3.0 if c["frequency"] == "three_times_daily" else 0.0,
        visit_day=float(c["day"]) if c["day"] is not None else 0.0,
    )
    doses = [Decimal(v) for v in c["dose_values"]]
    require(
        len(doses) <= 4 and all(v.is_finite() and v > 0 for v in doses), "invalid dose-set source"
    )
    q.update(
        dose_min_reported_unit=float(min(doses)) if doses else 0.0,
        dose_max_reported_unit=float(max(doses)) if doses else 0.0,
        dose_level_count=float(len(doses)),
    )
    q.update(
        {"dose_level_" + str(i + 1): float(doses[i]) if i < len(doses) else 0.0 for i in range(4)}
    )
    require(
        set(q) == set(CATEGORICAL + NUMERIC) and q["matrix"] not in {"male", "healthy", "patients"},
        "unapproved source/identifier/sex/population field entered context",
    )
    return q


def support(query, fitted):
    contexts = [json.loads(r["semantic_context_json"]) for r in fitted]
    types = {q["estimand"] for q in contexts}
    type_groups = {r["split_group"] for r in fitted if r["source_estimand"] == query["estimand"]}
    joint_groups = {
        r["split_group"] for r in fitted if r["semantic_context_key"] == object_sha(query)
    }
    return dict(
        estimand_seen=query["estimand"] in types,
        statistic_basis_seen=query["statistic_basis"] in {q["statistic_basis"] for q in contexts},
        joint_context_seen=bool(joint_groups),
        fitted_molecular_groups=len({r["split_group"] for r in fitted}),
        estimand_molecular_groups=len(type_groups),
        joint_context_molecular_groups=len(joint_groups),
        low_estimand_molecular_support=len(type_groups) < 2,
        new_molecule_generalization_validated=False,
    )


def verify_plan(source, plan):
    by_id = {r["record_id"]: r for r in source}
    require(
        plan.get("protocol") == "group_holdout_inner5_then_separate_all_common_refit"
        and plan.get("refit_full_required") is True
        and plan.get("independent_holdout") is True
        and plan.get("final_training_records") == 27
        and plan.get("physiological_or_subtype_generalization_claim") is False
        and [c["seed"] for c in plan["seeds"]] == list(SEEDS),
        "outer/inner/refit plan scope",
    )
    result = {}
    for check in plan["seeds"]:
        a, b = check["train_record_ids"], check["holdout_record_ids"]
        require(
            len(set(a)) == len(a)
            and len(set(b)) == len(b)
            and set(a) | set(b) == set(by_id)
            and not set(a) & set(b)
            and len(b) >= 4,
            "outer record membership",
        )
        ga, gb = ({by_id[i]["split_group"] for i in v} for v in (a, b))
        require(
            len(ga) == check["train_groups"] == 8
            and len(gb) == check["holdout_groups"] == 2
            and not ga & gb,
            "outer graph/known-family leakage or counts",
        )
        require(
            check["actual_folds"] == 5
            and len(check["folds"]) == 5
            and [f["fold"] for f in check["folds"]] == list(range(5)),
            "actual inner fivefold",
        )
        seen = []
        for f in check["folds"]:
            tr, va = f["train_record_ids"], f["validation_record_ids"]
            require(
                set(tr) | set(va) == set(a)
                and not set(tr) & set(va)
                and len(set(tr)) == len(tr)
                and len(set(va)) == len(va),
                "inner memberships",
            )
            require(
                not {by_id[i]["split_group"] for i in tr} & {by_id[i]["split_group"] for i in va}
                and not {by_id[i]["canonical_identity_group"] for i in tr}
                & {by_id[i]["canonical_identity_group"] for i in va}
                and len({by_id[i]["target"] for i in tr}) > 1,
                "inner graph leakage/constant target",
            )
            expected = {
                i: support(json.loads(by_id[i]["semantic_context_json"]), [by_id[j] for j in tr])
                for i in va
            }
            require(
                f["context_support"] == expected,
                "inner support not computed from actual fit members",
            )
            seen += va
        require(Counter(seen) == Counter(a), "inner OOF not exactly once")
        require(
            check["holdout_context_support"]
            == {
                i: support(json.loads(by_id[i]["semantic_context_json"]), [by_id[j] for j in a])
                for i in b
            },
            "outer support not actual fit members",
        )
        require(
            check["final_refit_record_ids"] == list(by_id)
            and check["all27_refit_support"]
            == {
                i: support(json.loads(r["semantic_context_json"]), source) for i, r in by_id.items()
            },
            "all27 refit coverage/support",
        )
        result[check["seed"]] = check
    return result


@audit_errors
def source_at(root, api, *, require_node=True, scripts_root=None):
    root = Path(root)
    manifest = api.read(root / "task_manifest.json")
    require(
        manifest.get("schema") == SCHEMA and len(manifest["tasks"]) == 1, "only one exact Vd task"
    )
    task = manifest["tasks"][0]
    definition = task["scientific_definition"]
    require(
        is_task(task)
        and task.get("endpoint") == "Vd_absolute"
        and task.get("unit") == "L"
        and task.get("target_transform") == "log10"
        and task.get("inverse_transform") == "pow10"
        and task.get("n_records") == 27
        and task.get("n_groups") == 10
        and definition == task["task_definition"],
        "Vd quantity/counts/scientific definition",
    )
    required = dict(
        canonical_quantity_id=QUANTITY,
        query_mode="source_annotation",
        annotation_only=True,
        validated_physiology=False,
        context_group=CONTEXT,
        context_fields=list(CATEGORICAL + NUMERIC),
        implementation_module=MODULE[:-3],
        implementation_sha256=PRODUCER_SHA,
        molecular_groups=10,
        independent_scaffolds_claimed=False,
        mandatory_molecular_reporting=True,
        mandatory_float64_fixed_controls=True,
        mandatory_all27_refit=True,
        primary_inner_cv_role="cross_estimand_extrapolation_stress_test_for_selection",
        each_estimand_independently_fivefold_validated=False,
        all27_refit_creates_no_new_molecular_generalization_support=True,
    )
    require(
        all(definition.get(k) == v for k, v in required.items()),
        "annotation/query limits or producer pin changed",
    )
    descriptor = task["native_source_observation_coverage"]
    proof_path = api.contained(root, descriptor["proof_path"])
    require(api.sha(proof_path) == descriptor["proof_sha256"] == PROOF_SHA, "source proof hash")
    proof = api.read(proof_path)
    require(
        proof.get("schema") == SCHEMA
        and proof.get("task_id") == TASK
        and proof.get("implementation_sha256") == PRODUCER_SHA
        and proof.get("source_contract_sha256") == SOURCE_SHA
        and (proof.get("rows"), proof.get("graphs"), proof.get("molecular_groups")) == (27, 10, 10),
        "source proof scope",
    )
    for item in proof["pinned_artifacts"]:
        require(
            api.sha(api.contained(root, item["path"])) == item["sha256"],
            "pinned source closure artifact changed",
        )
    require(
        api.sha(root / "source_provenance/Vd_main_candidate_source_contract.json") == SOURCE_SHA
        and api.sha(root / "source_provenance/typed_context_corrections.json") == CORRECTIONS_SHA
        and api.sha(root / "independent_source_review.json") == REVIEW_SHA,
        "independent source/correction/review pins",
    )
    contract = api.read(root / "source_provenance/Vd_main_candidate_source_contract.json")
    corrections = {
        r["record_id"]: r
        for r in api.read(root / "source_provenance/typed_context_corrections.json")["records"]
    }
    source = api.table(api.contained(root, task["table_path"]))
    require(
        api.sha(api.contained(root, task["table_path"])) == TABLE_SHA
        and len(source) == len({r["record_id"] for r in source}) == 27
        and {r["record_id"]: object_sha(r) for r in source} == proof["row_hashes"],
        "exact source table/rows",
    )
    native = {r["record_id"]: r for r in contract["records"]}
    require(
        len(native) == 27 and set(native) == {r["record_id"] for r in source},
        "original27 source membership",
    )
    components = {}

    def find(k):
        components.setdefault(k, k)
        if components[k] != k:
            components[k] = find(components[k])
        return components[k]

    for r in contract["records"]:
        graph = r["canonical_graph"]
        find(graph)
        for family in r["known_family_links"]:
            components[find(graph)] = find(family)
    for r in source:
        old = native[r["record_id"]]
        q = corrected_context(old, corrections[r["record_id"]])
        require(
            r["source_original_record_id"] == r["record_id"]
            and r["endpoint"] == "Vd_absolute"
            and r["value"] == r["raw_target"] == old["value"]
            and Decimal(old["raw_value"]) == Decimal(r["value"])
            and old["raw_relation"] == old["relation"] == "="
            and old["raw_unit"] == old["unit"] == r["unit"] == "L"
            and r["canonical_smiles"] == r["smiles"] == old["registered_canonical_smiles"]
            and r["canonical_identity_group"]
            == old["canonical_graph"]
            == "CANON_SMILES:" + hashlib.sha256(r["canonical_smiles"].encode()).hexdigest()
            and r["split_group"] == old["wholegraph_union_known_family_group"]
            and r["measurement_group"] == old["source_observation_key"]
            and r["source_record_id"] == old["source_record_id"]
            and r["source_entry_id"] == old["source_entry_id"]
            and r["source_estimand"] == q["estimand"]
            and json.loads(r["semantic_context_json"]) == q
            and r["semantic_context_key"] == object_sha(q)
            and r["vd_native_context_eligible"] == "true"
            and r["canonical_quantity_id"] == QUANTITY
            and r["structure_eligible"] == "true"
            and r["sequence_eligible"] == "false"
            and not r["sequence"],
            "source scalar/graph/typed role drift",
        )
        target_text(r["target"], r["value"])
    require(
        len({r["canonical_identity_group"] for r in source})
        == len({r["split_group"] for r in source})
        == 10,
        "10 registered graph/groups",
    )
    require(
        all(
            (find(a["canonical_identity_group"]) == find(b["canonical_identity_group"]))
            == (a["split_group"] == b["split_group"])
            for a in source
            for b in source
        ),
        "known-FAMILY transitive union mismatch",
    )
    review = api.read(root / "independent_source_review.json")
    require(
        review.get("scientific_candidate_approved") is True
        and review.get("blocked_record_ids") == []
        and set(review["allowed_record_ids"]) == set(native)
        and review.get("typed_context_sha256") == CORRECTIONS_SHA,
        "independent reviewed27 gate changed",
    )
    binding = api.read(root / "independent_source_review_binding.json")
    require(
        binding.get("status") == "bound_narrow_source_annotation_only"
        and binding.get("review_sha256") == REVIEW_SHA
        and binding.get("source_contract_sha256") == SOURCE_SHA
        and binding.get("implementation_sha256") == PRODUCER_SHA
        and set(binding["allowed_record_ids"]) == set(native)
        and binding.get("malformed_matrix_corrected") is True
        and binding.get("statistical_dispersion_not_inferred") is True,
        "independent review execution binding missing",
    )
    ontology = api.read(root / "task_tables/semantic_context_contract.json")
    expected_categories = {
        k: sorted({json.loads(r["semantic_context_json"])[k] for r in source}) for k in CATEGORICAL
    }
    expected_names = [f"{k}={v}" for k in CATEGORICAL for v in expected_categories[k]] + list(
        NUMERIC
    )
    require(
        api.sha(root / "task_tables/semantic_context_contract.json") == ONTOLOGY_SHA
        and ontology
        == dict(
            schema=SCHEMA,
            categorical=expected_categories,
            numeric=list(NUMERIC),
            feature_names=expected_names,
            unknown_is_explicit=True,
            label_or_identifier_columns=0,
        )
        and len(expected_names) == 65,
        "unapproved/outcome/ID features or semantic ontology",
    )
    if scripts_root is not None:
        for name, expected in [
            (MODULE, PRODUCER_SHA),
            ("peptide_modeling_mhc10_native_bridge.py", HELPER_SHA),
            ("peptide_modeling_train.py", TRAIN_SHA),
        ]:
            require(
                api.sha(Path(scripts_root) / name) == expected,
                "frozen implementation byte mismatch",
            )
    if not require_node:
        return dict(proof=proof, task=task, source=source, ontology=ontology, checks=None)
    closure = verify_closure(root, api)
    require(
        task.get("status") == "eligible"
        and manifest.get("execution_gate") == "both_concrete_gates_bound_before_any_fit"
        and definition.get("source_review_state")
        == "independent_review_bound_with_context_corrections"
        and definition.get("pretraining_fold_state") == "node23_actual_no_fit_frozen",
        "pending/local preparation is not execution eligible",
    )
    node_binding = api.read(root / "nodefold_binding.json")
    node_path = api.contained(root, node_binding["receipt_path"])
    require(
        api.sha(node_path) == node_binding["receipt_sha256"]
        and node_binding["source_proof_sha256"] == descriptor["proof_sha256"],
        "node-fold proof binding",
    )
    node = api.read(node_path)
    require(
        node.get("probe_scope") == "actual_node23_pretraining_no_fit"
        and node.get("hostname") == "node23"
        and node.get("python_executable") == "/usr/bin/python3"
        and node.get("dont_write_bytecode") == 1
        and node.get("producer_sha256") == PRODUCER_SHA
        and node.get("training_source_sha256") == TRAIN_SHA
        and node.get("source_contract_sha256") == SOURCE_SHA
        and node.get("task_id") == TASK
        and node.get("input_hashes_before") == node.get("input_hashes_after")
        and bool(node.get("input_hashes_before"))
        and node.get("guard_counters")
        == dict(
            estimator_fit_attempts=0,
            search_attempts=0,
            model_unpickle_attempts=0,
            chemistry_import_attempts=0,
            production_fits_launched=0,
        )
        and node.get("frozen_targets_rewritten") is False
        and node.get("report_only_estimator_fits") == 0,
        "local/changed/fitted node probe is not execution evidence",
    )
    checks = verify_plan(source, node["actual_checks"])
    for seed in SEEDS:
        check = checks[seed]
        outer = node["outer_split_receipts"][str(seed)]
        require(
            outer["train_records"] == check["train_record_ids"]
            and outer["test_records"] == check["holdout_record_ids"]
            and outer["record_groups"] == {r["record_id"]: r["split_group"] for r in source},
            "actual outer receipt/member mismatch",
        )
    verify_probe_staging(node, root, closure, api, require_bound=False)
    post_binding = api.read(root / "post_adoption_round_staging_binding.json")
    post_path = api.contained(root, post_binding["path"])
    require(
        post_binding.get("status") == "actual_node23_postadoption_round_closure_verified"
        and api.sha(post_path) == post_binding["sha256"] == POST_RECEIPT_SHA
        and post_binding.get("producer_sha256") == PRODUCER_SHA
        and post_binding.get("source_closure_sha256") == CLOSURE_SHA
        and post_binding.get("actual_seed_staging_reviews") == 3
        and post_binding.get("estimator_fits") == 0,
        "third actual staged-round closure gate missing",
    )
    post = api.read(post_path)
    require(
        post.get("probe_scope") == "actual_node23_pretraining_no_fit"
        and post.get("hostname") == "node23"
        and post.get("producer_sha256") == PRODUCER_SHA
        and post.get("training_source_sha256") == TRAIN_SHA
        and post.get("actual_checks") == node["actual_checks"]
        and post.get("outer_split_receipts") == node["outer_split_receipts"]
        and post.get("guard_counters") == node["guard_counters"]
        and post.get("report_only_estimator_fits") == 0
        and post.get("input_hashes_before") == post.get("input_hashes_after")
        and post.get("python_executable") == "/usr/bin/python3"
        and post.get("dont_write_bytecode") == 1,
        "post-adoption closure probe scope/runtime/fold mismatch",
    )
    verify_probe_staging(post, root, closure, api, require_bound=True)
    return dict(
        proof=proof,
        task=task,
        source=source,
        ontology=ontology,
        checks=checks,
        node=node,
        closure=closure,
    )


def verify_closure(root, api):
    root = Path(root)
    path = root / "source_staging_closure.json"
    require(api.sha(path) == CLOSURE_SHA, "source staging closure pin")
    c = api.read(path)
    require(
        c.get("schema") == "vd_absolute_source_staging_closure_v2"
        and c.get("task_id") == TASK
        and c.get("producer_sha256") == PRODUCER_SHA
        and c.get("actual_controller_stage_function") == "stage_seed"
        and c.get("missing_source_file_is_fatal") is True
        and len(c["files"]) == 79
        and c.get("runtime_receipts_required_when_bound")
        == [
            "nodefold_binding.json",
            "node23_actual_no_fit_fold_receipt.json",
            "post_adoption_round_staging_binding.json",
            "post_adoption_round_staging_receipt.json",
        ],
        "source staging closure definition",
    )
    required = {
        "source_provenance/Vd_main_candidate_source_contract.json",
        "source_provenance/typed_context_corrections.json",
        "independent_source_review.json",
        "independent_source_review_binding.json",
        "task_tables/semantic_context_contract.json",
    }
    require(required <= set(c["files"]), "round-copy closure omits required source/review/context")
    for name, pin in c["files"].items():
        p = api.contained(root, name)
        require(
            not p.is_symlink() and p.is_file() and api.sha(p) == pin,
            "round source closure byte mismatch: " + name,
        )
    return c


def verify_probe_staging(receipt, root, closure, api, *, require_bound):
    staged = receipt.get("staged_seed_closure_reviews", [])
    require(
        receipt.get("source_closure_sha256") == CLOSURE_SHA
        and len(staged) == 3
        and [r.get("seed") for r in staged] == list(SEEDS)
        and all(p.get("num_threads") == 1 for p in receipt.get("threadpools", [])),
        "actual staged3seed probe/thread budget",
    )
    for item in staged:
        require(
            item.get("status") == "verified_no_fit_staged_source_closure"
            and item.get("source_closure_sha256") == CLOSURE_SHA
            and item.get("source_closure_files_checked") == len(closure["files"])
            and item.get("exact_source_proof_sha256") == PROOF_SHA
            and item.get("exact_task_table_sha256") == TABLE_SHA
            and item.get("portable_source_and_all_routes_passed") is True
            and item.get("actual_staged_outer_and_inner_folds_rederived") is True
            and all(
                item.get(k) == 0
                for k in ("estimator_fits", "searches", "model_unpickles", "chemistry_imports")
            ),
            "source closure canary not actual staged/no-fit",
        )
        if require_bound:
            for name in ("nodefold_binding.json", "node23_actual_no_fit_fold_receipt.json"):
                require(
                    item["runtime_receipts_copied"].get(name) == api.sha(Path(root) / name),
                    "staged node receipt bytes differ",
                )


def verify_source(entry, task, api):
    if entry.get("verified_launch_contract_post_exit") is not None:
        verify_post_exit_binding(entry, api)
    root = Path(entry["template"])
    require(
        api.sha(root / "task_manifest.json") == entry["manifest_sha256"],
        "registry manifest differs",
    )
    binding = source_at(root, api, scripts_root=root / "scripts")
    require(binding["task"]["task_id"] == task["task_id"], "registry task differs")
    return binding


@audit_errors
def verify_post_exit_binding(entry, api):
    """Revalidate recorded historical live evidence without claiming a process is live now."""
    pins = entry["verified_launch_contract_post_exit"]
    require(
        entry.get("state_at_freeze") == "completed_post_exit_from_verified_launch_contract"
        and pins.get("schema") == "vd_absolute_verified_launch_post_exit_freeze_v1",
        "unapproved post-exit freeze semantics",
    )
    expected = {
        "launch_contract": LAUNCH_CONTRACT_SHA,
        "verified_live_launch": LAUNCH_RECEIPT_SHA,
        "preflight": LAUNCH_PREFLIGHT_SHA,
        "followup_config": LAUNCH_CONFIG_SHA,
    }
    values = {}
    for name, pin in expected.items():
        item = pins[name]
        require(
            item["sha256"] == pin and api.sha(item["path"]) == pin,
            "historical launch evidence changed",
        )
        values[name] = api.read(item["path"])
    contract, launch, preflight, cfg = (
        values[k]
        for k in ("launch_contract", "verified_live_launch", "preflight", "followup_config")
    )
    require(
        contract.get("schema") == "vd_absolute_native_launch_contract_v1"
        and contract.get("prepared_delivery_sha256")
        == launch.get("prepared_delivery_sha256")
        == PREPARED_DELIVERY_SHA
        and contract.get("followup_config_sha256")
        == launch.get("config_sha256")
        == LAUNCH_CONFIG_SHA
        and contract.get("followup_config") == cfg
        and contract.get("followup_config_path") == pins["followup_config"]["path"]
        and launch.get("launch_contract_sha256") == LAUNCH_CONTRACT_SHA
        and launch.get("preflight_receipt_sha256") == LAUNCH_PREFLIGHT_SHA,
        "prefit contract/config/launch chain",
    )
    require(
        launch.get("status") == "waiting_parent_verified"
        and launch.get("rounds_observed") == 0
        and launch.get("controller_identity_verified_live") is True
        and launch.get("task_id") == TASK
        and launch.get("source_annotation_only") is True
        and launch.get("cpu_budget") == 1
        and launch.get("additional_twenty_hour_loop") is False,
        "launch was not a recorded verified-live zero-round startup",
    )
    owner = launch["process_identity"]
    require(
        owner.get("pid") == launch.get("pid") == 3542333
        and owner.get("start_ticks") == 96262211
        and owner.get("argv") == launch.get("argv")
        and owner.get("state") != "Z"
        and entry["owned_process_identity"] == owner,
        "historical PID/start/argv was reconstructed or replaced",
    )
    template = Path(cfg["template_run"])
    output = Path(cfg["output_root"])
    durable = Path(cfg["durable_root"])
    require(
        str(template) == entry["template"] == launch["template"]
        and str(output) == entry["output_root"] == launch["output_root"]
        and str(durable) == launch["durable_root"]
        and entry["config_path"] == launch["config_path"] == pins["followup_config"]["path"],
        "launch namespace changed",
    )
    actual_files = {
        p.relative_to(template).as_posix(): api.sha(p) for p in template.rglob("*") if p.is_file()
    }
    require(
        len(actual_files) == len(contract["files"]) == 113
        and actual_files == contract["files"]
        and not any(p.is_symlink() for p in template.rglob("*")),
        "current template is not the exact113-file prefit payload",
    )
    require(
        preflight.get("status") == "ready"
        and preflight.get("host") == "node23"
        and preflight.get("producer_sha256") == PRODUCER_SHA
        and preflight.get("source_proof_sha256") == PROOF_SHA
        and preflight.get("source_closure_sha256") == CLOSURE_SHA
        and preflight.get("config_sha256") == LAUNCH_CONFIG_SHA
        and preflight.get("verified_template_files") == 113
        and preflight.get("source_closure_files") == 79
        and preflight.get("script_dependencies") == 29
        and preflight.get("preflight_model_fits") == 0
        and preflight.get("required_seeds") == list(SEEDS),
        "prefit source/runtime/no-fit gate changed",
    )
    parent = preflight["parent_requirement_evidence"]
    require(
        parent.get("productive_wall_seconds", 0) >= 72000
        and parent.get("verified_required_seeds") == list(SEEDS)
        and parent.get("parent_owned_processes_released") is True,
        "parent productive-time/seed/release gate absent",
    )
    for name, item in pins.items():
        if name in expected:
            copy_path = api.contained(durable, "deployment/" + Path(item["path"]).name)
            require(
                api.sha(copy_path) == item["sha256"], "durable deployment launch evidence differs"
            )
    journal = api.read(entry["journal_path"])
    require(
        api.sha(entry["journal_path"]) == pins["terminal_journal_sha256"]
        and journal.get("status") == "completed"
        and journal.get("mandatory_reporting_state") == "verified"
        and journal.get("pid") == owner["pid"]
        and journal.get("process_identity", {}).get("start_ticks") == owner["start_ticks"]
        and journal.get("process_identity", {}).get("argv") == owner["argv"]
        and journal.get("config_sha256") == LAUNCH_CONFIG_SHA,
        "post-exit terminal journal changed",
    )
    actual = api.process_identity(owner["pid"])
    require(
        actual is None
        or actual.get("state") == "Z"
        or actual.get("start_ticks") != owner["start_ticks"],
        "historical controller remains live; use ordinary live freeze instead",
    )
    return values


@audit_errors
def freeze_post_exit_entry(
    launch_contract_path, launch_receipt_path, preflight_path, config_path, api
):
    """Specific terminal adoption from immutable prefit/verified-live receipts; pure read-only."""
    launch = api.read(launch_receipt_path)
    cfg = api.read(config_path)
    template = Path(cfg["template_run"])
    output = Path(cfg["output_root"])
    manifest_path = template / "task_manifest.json"
    manifest = api.read(manifest_path)
    task = manifest["tasks"][0]
    source_path = api.contained(template, task["table_path"])
    pinned = dict(
        task_id=TASK,
        endpoint="Vd_absolute",
        table_relative=task["table_path"],
        table_sha256=api.sha(source_path),
        records=27,
        evidence_tier="source_native_typed_annotation_not_unified_physiology",
        scientific_definition=task["task_definition"],
        native_source_observation_coverage=task["native_source_observation_coverage"],
    )
    evidence = dict(
        schema="vd_absolute_verified_launch_post_exit_freeze_v1",
        launch_contract=dict(
            path=str(Path(launch_contract_path).resolve()), sha256=LAUNCH_CONTRACT_SHA
        ),
        verified_live_launch=dict(
            path=str(Path(launch_receipt_path).resolve()), sha256=LAUNCH_RECEIPT_SHA
        ),
        preflight=dict(path=str(Path(preflight_path).resolve()), sha256=LAUNCH_PREFLIGHT_SHA),
        followup_config=dict(path=str(Path(config_path).resolve()), sha256=LAUNCH_CONFIG_SHA),
        terminal_journal_sha256=api.sha(output / "campaign_status.json"),
        historical_live_observation_reused_without_new_live_claim=True,
    )
    feature = template / "features/manifest.json"
    registry = template / "representation_methods.json"
    entry = dict(
        controller_id=output.name,
        output_root=str(output),
        journal_path=str(output / "campaign_status.json"),
        protocol="group_holdout_inner5",
        controller_execution="vd_native_bounded_in_process",
        config_path=str(Path(config_path).resolve()),
        config_sha256=LAUNCH_CONFIG_SHA,
        template=str(template),
        manifest_path=str(manifest_path),
        manifest_sha256=api.sha(manifest_path),
        template_config=dict(path=cfg["template_config"], sha256=api.sha(cfg["template_config"])),
        tasks=[pinned],
        expected_scopes=["common"],
        required_seeds=list(SEEDS),
        owned_process_identity=launch["process_identity"],
        journal_sha256_at_freeze=api.sha(output / "campaign_status.json"),
        state_at_freeze="completed_post_exit_from_verified_launch_contract",
        verified_launch_contract_post_exit=evidence,
        native_feature_contract=dict(
            path=str(feature),
            sha256=api.sha(feature),
            registry_path=str(registry),
            registry_sha256=api.sha(registry),
        ),
    )
    verify_post_exit_binding(entry, api)
    verify_reporting(entry, api.read(entry["journal_path"]), pinned, api)
    return entry


def coverage(entry, journal, task, state, api):
    binding = verify_source(entry, task, api)
    contract = api.read(
        Path(entry["template"]) / "source_provenance/Vd_main_candidate_source_contract.json"
    )
    original = [
        r["record_id"] for r in contract["records"] if r["original_lane"] == "strict_numeric"
    ]
    require(
        len(original) == 15 and set(original) <= {r["record_id"] for r in binding["source"]},
        "original15 strict observation coverage",
    )
    evidence = dict(
        original_endpoint="Vd_absolute",
        quantity_id=QUANTITY,
        task_id=TASK,
        original_labelled_records=15,
        included_original_records=15,
        record_ids=original,
        training_pool_records=27,
        training_pool_identity_groups=10,
        source_estimand_types=8,
        source_annotation_only=True,
        validated_physiology=False,
        each_estimand_fivefold_validated=False,
        source_proof_verified=True,
        coverage_kind="native_typed_volume_L_source_annotation_observation_participation",
        required_seed_final_training_verified=0,
        state="source_annotation_observation_pending",
    )
    if state == "trained_three_seeds":
        verify_reporting(entry, journal, task, api)
        evidence.update(
            required_seed_final_training_verified=3, state="source_annotation_observation_trained"
        )
    return evidence


def verify_features(root, source, ontology, api):
    root = Path(root)
    f = api.read(root / "features/manifest.json")
    cache = api.contained(root, "features/" + f["input_fingerprint"][:16])
    require(
        f.get("schema") == SCHEMA
        and f.get("requested_records") == 27
        and f.get("structure_keys") == 10
        and set(f["groups"]) == set(GROUPS) | {CONTEXT},
        "feature schema/scope/groups",
    )
    receipt = api.read(root / "feature_delivery_receipt.json")
    require(
        receipt.get("status") == "verified_feature_preparation_only"
        and receipt.get("complete_feature_artifact_hashes") == f["artifact_hashes"]
        and receipt.get("reused_exact_graphs") == 7
        and receipt.get("newly_extracted_complete_graphs") == 3
        and receipt.get("independent_methods") == 10
        and receipt.get("chemical_groups") == 13
        and receipt.get("chemical_dimensions") == 15212
        and receipt.get("context_columns") == 65
        and receipt.get("parent_pointer_unchanged") is True
        and receipt.get("parent_artifacts_unchanged") == 48
        and receipt.get("descriptor_columns_compared_per_canary") == 217
        and receipt.get("guessed_graphs") == receipt.get("model_fits") == 0,
        "feature prep provenance/canary scope",
    )
    for name, expected in f["artifact_hashes"].items():
        require(api.sha(api.contained(cache, name)) == expected, "feature bytes changed: " + name)
    aliases = api.table(api.contained(cache, f["record_index"]))
    alias = {r["record_id"]: r for r in aliases}
    require(
        len(aliases) == len(alias) == 27
        and set(alias) == {r["record_id"] for r in source}
        and all(
            alias[r["record_id"]]["structure_key"] == r["canonical_identity_group"] for r in source
        ),
        "exact27 graph aliases",
    )
    methods = api.read(root / "representation_methods.json")
    require(
        methods == f["representation_methods"]
        and len(methods["methods"]) == 10
        and len({m["algorithm_family_id"] for m in methods["methods"]}) == 10
        and all(
            m.get("independent") is True and m.get("source_kind") == "structure"
            for m in methods["methods"]
        ),
        "ten independent chemical algorithms",
    )
    require(
        set(f["routes"]) == {m["route"] for m in methods["methods"]} | {"structure_methods_all"}
        and f["routes"]["structure_methods_all"] == list(GROUPS) + [CONTEXT],
        "chemical fusion/route scope",
    )
    for method in methods["methods"]:
        require(
            f["routes"][method["route"]] == method["group_names"] + [CONTEXT]
            and sum(f["groups"][g]["shape"][1] for g in method["group_names"])
            == method["dimensions"],
            "context must append once without counting as method",
        )
    require(
        sum(f["groups"][g]["shape"][1] for g in GROUPS) == 15212, "chemical dimensionality changed"
    )
    parts, names = {}, {}
    for name, meta in f["groups"].items():
        is_context = name == CONTEXT
        require(
            meta["key_kind"] == ("record_id" if is_context else "canonical_identity_group")
            and (
                not is_context
                or meta.get("eligibility_field") == "vd_native_context_eligible"
                and meta.get("chemical_method") is False
                and meta["shape"] == [27, 65]
            ),
            "record context/graph key boundary changed",
        )
        indices = api.table(api.contained(cache, meta["index"]))
        index = {r["key"]: int(r["row_idx"]) for r in indices}
        require(
            len(indices) == len(index) == (27 if is_context else 10)
            and set(index.values()) == set(range(len(index))),
            "feature index missing/duplicate",
        )
        header, data = api.hc50_precision_npy(api.contained(cache, meta["matrix"]))
        mask_header, mask = api.hc50_precision_npy(api.contained(cache, meta["valid"]))
        width = len(meta["feature_names"])
        require(
            header["shape"] == (len(index), width) == tuple(meta["shape"])
            and header["descr"] in {"<f4", "|u1"}
            and mask_header["descr"] == "|b1"
            and mask_header["shape"] == (len(index),),
            "primitive feature dtype/shape changed",
        )
        bytes_per = width * (4 if header["descr"] == "<f4" else 1)
        selected = {}
        for row in source:
            key = row["record_id"] if is_context else row["canonical_identity_group"]
            require(key in index and mask[index[key]] == 1, "missing/invalid exact representation")
            raw = data[index[key] * bytes_per : (index[key] + 1) * bytes_per]
            raw = struct.pack("<" + "f" * width, *raw) if header["descr"] == "|u1" else raw
            values = struct.unpack("<" + "f" * width, raw)
            missing = sum(math.isnan(v) for v in values)
            require(
                not any(math.isinf(v) for v in values)
                and missing < width
                and (
                    not missing
                    or meta.get("allow_partial_missing") is True
                    and missing / width <= meta.get("max_missing_fraction", 0)
                ),
                "frozen missingness policy",
            )
            if is_context:
                q = json.loads(row["semantic_context_json"])
                vector = [float(q[k] == v) for k in CATEGORICAL for v in ontology["categorical"][k]]
                vector += [q[k] for k in NUMERIC]
                require(
                    raw == struct.pack("<" + "f" * 65, *vector),
                    "context numeric bytes do not encode exact source roles",
                )
            selected[row["record_id"]] = raw
        parts[name] = selected
        names[name] = [name + ":" + str(v) for v in meta["feature_names"]]
    return dict(features=f, parts=parts, names=names)


def designs(binding, route, source, api):
    f = binding["features"]
    require(route in f["routes"], "unknown chemical route")
    selected = f["routes"][route]
    require(selected[-1] == CONTEXT and selected.count(CONTEXT) == 1, "context missing/repeated")
    chemical = [g for g in selected if g != CONTEXT]
    ids = [r["record_id"] for r in source]
    result = {}
    for name, groups in (
        ("molecule_only", chemical),
        ("context_only", [CONTEXT]),
        ("molecule_plus_context", chemical + [CONTEXT]),
    ):
        parts = {i: b"".join(binding["parts"][g][i] for g in groups) for i in ids}
        result[name] = dict(
            rows=parts,
            names=sum((binding["names"][g] for g in groups), []),
            **api.control_precision_matrix_bytes(parts, ids),
        )
    return result


def weighted_metrics(predictions, by_id):
    require(
        predictions and len({r["record_id"] for r in predictions}) == len(predictions),
        "empty/repeated metric rows",
    )
    counts = Counter(by_id[r["record_id"]]["split_group"] for r in predictions)
    errors = [abs(float(r["target"]) - float(r["prediction"])) for r in predictions]
    require(all(math.isfinite(v) for v in errors), "nonfinite role metrics")
    scale = max(errors)

    def metric(weights):
        total = math.fsum(weights)
        return dict(
            mae=(
                scale
                * math.fsum((e / scale) * w for e, w in zip(errors, weights, strict=True))
                / total
                if scale
                else 0.0
            ),
            rmse=(
                scale
                * math.sqrt(
                    math.fsum((e / scale) ** 2 * w for e, w in zip(errors, weights, strict=True))
                    / total
                )
                if scale
                else 0.0
            ),
        )

    return dict(
        record_weighted=metric([1.0] * len(errors)),
        molecule_equal=metric(
            [1 / counts[by_id[r["record_id"]]["split_group"]] for r in predictions]
        ),
        record_counts_per_molecule=dict(counts),
        molecular_groups=len(counts),
    )


def check_metrics(actual, expected):
    for kind in ("record_weighted", "molecule_equal"):
        for key in ("mae", "rmse"):
            value = actual[kind][key]
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and math.isclose(value, expected[kind][key], rel_tol=1e-9, abs_tol=1e-10),
                "role/molecule metric differs",
            )
    require(
        actual.get("record_counts_per_molecule") == expected["record_counts_per_molecule"]
        and actual.get("molecular_groups") == expected["molecular_groups"],
        "molecular weighting denominator changed",
    )


def _close(a, b):
    return abs(a - b) <= 1e-8 + 1e-7 * abs(b)


def selected_consistency(records, source, design, provenance):
    by_id = {r["record_id"]: r for r in source}
    grouped = defaultdict(list)
    for row in records:
        old = by_id[row["record_id"]]
        grouped[
            (
                old["canonical_identity_group"],
                old["split_group"],
                old["semantic_context_key"],
                hashlib.sha256(design["rows"][row["record_id"]]).hexdigest(),
                int(row["fold"]),
            )
        ].append(row)
    expected = []
    for (_, _, _, _row_sha, _), rows in grouped.items():
        p = [float(r["prediction"]) for r in rows]
        require(all(math.isfinite(v) for v in p), "selected prediction nonfinite")
        ordinary = len(set(p)) == 1
        ulps = None

        def bits32(value):
            raw = struct.pack("!f", value)
            require(struct.unpack("!f", raw)[0] == value, "selected exception is not exact float32")
            n = struct.unpack("!I", raw)[0]
            return 2**31 - (n & 0x7FFFFFFF) if n >> 31 else 2**31 + n

        if not ordinary:
            require(
                provenance.get("actual_estimator") == "Ridge"
                and provenance.get("actual_coef_dtype") == "float32"
                and provenance.get("actual_solver") == "lsqr"
                and provenance.get("input_dtype") == "float32"
                and provenance.get("selected_model") == "linear"
                and provenance.get("actual_scaler_with_mean") is False
                and provenance.get("actual_imputer_strategy") == "median"
                and provenance.get("actual_add_indicator") is True
                and provenance.get("training_source_sha256") == TRAIN_SHA
                and provenance.get("feature_width") == len(design["names"])
                and provenance.get("full_matrix_sha256") == design["source_float32_sha256"]
                and bool(provenance.get("actual_model_artifact_sha256"))
                and all(abs(v) >= 2**-126 for v in p)
                and len({math.copysign(1, v) for v in p}) == 1,
                "selected drift lacks actual normal float32 Ridge LSQR provenance",
            )
            positions = [bits32(v) for v in p]
            ulps = max(positions) - min(positions)
            require(ulps <= 32, "selected prediction drift exceeds32ULP")
        expected.append(
            dict(mode="exact", raw_predictions_unchanged=True)
            if ordinary
            else dict(
                mode="exact_binary32_LSQR_at_most32ULP",
                maximum_binary32_ulp_budget=32,
                observed_binary32_ulp_distance=ulps,
                raw_predictions_unchanged=True,
            )
        )
    return expected


def verify_evaluation(diagnostics, source, inner, held, check, design, provenance):
    by_id = {r["record_id"]: r for r in source}
    require(
        diagnostics.get("source_annotation_only") is True
        and diagnostics.get("full_refit_scores_not_external") is True
        and diagnostics.get("raw_predictions_mutated") is False
        and set(diagnostics["evaluation_by_role"]) == {"inner_oof", "outer_holdout"},
        "evaluation annotation/refit claim",
    )
    ids = [r["record_id"] for r in inner + held]
    require(
        len(ids) == len(set(ids)) == 27 and set(ids) == set(by_id), "exact27 prediction coverage"
    )
    fold_by = {i: f for f in check["folds"] for i in f["validation_record_ids"]}
    missing_type = []
    missing_joint = []
    for role, predictions, expected in (
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", held, check["holdout_record_ids"]),
    ):
        require(
            len(predictions) == len(expected)
            and {r["record_id"] for r in predictions} == set(expected),
            "evaluation roles differ from frozen node plan",
        )
        reported = diagnostics["evaluation_by_role"][role]
        require(
            {r["record_id"] for r in reported["rows"]} == set(expected)
            and len(reported["rows"]) == len(expected),
            "enriched role repeated/missing records",
        )
        enriched = {r["record_id"]: r for r in reported["rows"]}
        for row in predictions:
            i = row["record_id"]
            old = by_id[i]
            f = fold_by[i] if role == "inner_oof" else None
            fit = f["train_record_ids"] if f else check["train_record_ids"]
            support_value = support(
                json.loads(old["semantic_context_json"]), [by_id[j] for j in fit]
            )
            target, p = float(row["target"]), float(row["prediction"])
            require(
                math.isfinite(target)
                and math.isfinite(p)
                and target == float(old["target"])
                and row["split_group"] == old["split_group"]
                and int(row["fold"]) == (f["fold"] if f else -1),
                "saved target/group/actualfold changed",
            )
            require(
                enriched[i]
                == dict(
                    record_id=i,
                    evaluation_role=role,
                    fold=int(row["fold"]),
                    source_estimand=old["source_estimand"],
                    context_support=support_value,
                    target_log10_L=target,
                    prediction_log10_L=p,
                ),
                "per-fit source/context support not actual members",
            )
            if not support_value["estimand_seen"]:
                missing_type.append(i)
            if not support_value["joint_context_seen"]:
                missing_joint.append(i)
        computed = weighted_metrics(predictions, by_id)
        check_metrics(reported, computed)
        require(
            set(reported["per_estimand"])
            == {by_id[r["record_id"]]["source_estimand"] for r in predictions},
            "per-estimand strata",
        )
        for typ, summary in reported["per_estimand"].items():
            check_metrics(
                summary,
                weighted_metrics(
                    [r for r in predictions if by_id[r["record_id"]]["source_estimand"] == typ],
                    by_id,
                ),
            )
        require(
            reported["unsupported_estimand_record_ids"]
            == [
                r["record_id"]
                for r in reported["rows"]
                if not r["context_support"]["estimand_seen"]
            ]
            and reported["unsupported_joint_context_record_ids"]
            == [
                r["record_id"]
                for r in reported["rows"]
                if not r["context_support"]["joint_context_seen"]
            ],
            "unsupported role lists changed",
        )
        require(
            reported["unchanged_prediction_consistency"]
            == selected_consistency(predictions, source, design, provenance),
            "selected exact/32ULP evidence differs",
        )
    require(
        len(missing_type) == 22 and len(missing_joint) == 27,
        "pooled27 unsupported-estimand/joint scope changed",
    )
    return dict(unseen_estimand_records=22, unseen_joint_context_records=27)


def verify_controls(source, check, controls, full_designs):
    require(
        set(controls)
        == set(full_designs)
        == {"molecule_only", "context_only", "molecule_plus_context"},
        "three fixed designs",
    )
    ids = [r["record_id"] for r in source]
    by_id = {r["record_id"]: r for r in source}
    planned = [
        ("inner_training_fold", f["fold"], f["train_record_ids"], f["validation_record_ids"])
        for f in check["folds"]
    ]
    planned += [
        ("outer_train_only", -1, check["train_record_ids"], check["holdout_record_ids"]),
        ("all27_training_inputs_only_no_independent_score", -2, ids, ids),
    ]
    role_metrics = {}
    for name, design in full_designs.items():
        saved = controls[name]
        width = len(design["names"])
        raw = design["rows"]
        upcast = {
            i: struct.pack("<" + "d" * width, *struct.unpack("<" + "f" * width, v))
            for i, v in raw.items()
        }
        require(
            saved.get("model") == "fixed_Ridge_alpha1_LSQR_not_Optuna_selected"
            and saved.get("dtype_from_initial_boundary") == "float64"
            and saved.get("actual_fit_calls") == 7
            and len(saved["per_fit_trace"]) == len(saved["predictions"]) == 7
            and saved.get("search_calls") == saved.get("selected_model_fit_calls") == 0
            and saved.get("exact_upcast_of_stored_float32_design") is True
            and saved.get("source_encoded_values_rewritten") is False
            and saved.get("stored_matrix_sha256") == design["source_float32_sha256"]
            and saved.get("matrix_sha256") == design["upcast_float64_sha256"]
            and saved.get("full_refit_evaluation_role")
            == "training_inputs_only_no_independent_score",
            "fixed control design/precision/fit-count boundary",
        )
        all_inner = []
        outer_rows = []
        for (role, fold, train, predict), trace, item in zip(
            planned, saved["per_fit_trace"], saved["predictions"], strict=True
        ):
            p = item["predictions"]
            require(
                item["record_ids"] == predict
                and item["fit_role"] == role
                and item["fold"] == fold
                and len(p) == len(predict)
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in p
                ),
                "control prediction role/membership/values",
            )
            expected = dict(
                fit_role=role,
                fold=fold,
                train_record_ids=train,
                predict_record_ids=predict,
                actual_estimator="Ridge",
                actual_parameters=PARAMETERS,
                actual_fit_X_dtype="float64",
                actual_fit_y_dtype="float64",
                actual_predict_X_dtype="float64",
                actual_prediction_dtype="float64",
                actual_coef_dtype="float64",
                fit_X_sha256=hashlib.sha256(b"".join(upcast[i] for i in train)).hexdigest(),
                predict_X_sha256=hashlib.sha256(b"".join(upcast[i] for i in predict)).hexdigest(),
                fit_y_sha256=hashlib.sha256(
                    struct.pack("<" + "d" * len(train), *[float(by_id[i]["target"]) for i in train])
                ).hexdigest(),
                predictions_sha256=hashlib.sha256(struct.pack("<" + "d" * len(p), *p)).hexdigest(),
                control_fit_calls=1,
                selected_model_fit_calls=0,
                search_calls=0,
            )
            missing_columns = {
                j
                for i in train
                for j, v in enumerate(struct.unpack("<" + "f" * width, raw[i]))
                if math.isnan(v)
            }
            require(
                all(same(trace.get(k), v) for k, v in expected.items())
                and trace.get("actual_coef_shape") == [width + len(missing_columns)]
                and bool(re.fullmatch(r"[0-9a-f]{64}", str(trace.get("actual_coef_sha256", "")))),
                "actual float64 fit/dtype/parameters/member/data/coef trace changed",
            )
            identical = defaultdict(list)
            for i, value in zip(predict, p, strict=True):
                identical[upcast[i]].append(float(value))
            require(
                all(
                    all(_close(value, group[0]) for value in group) for group in identical.values()
                ),
                "float64 control identical-input strict tolerance",
            )
            rows = [
                dict(
                    record_id=i,
                    split_group=by_id[i]["split_group"],
                    target=by_id[i]["target"],
                    prediction=value,
                    fold=fold,
                )
                for i, value in zip(predict, p, strict=True)
            ]
            if role == "inner_training_fold":
                all_inner += rows
            elif role == "outer_train_only":
                outer_rows = rows
        require(
            Counter(r["record_id"] for r in all_inner) == Counter(check["train_record_ids"]),
            "control OOF coverage",
        )
        role_metrics[name] = dict(
            inner_oof_stress_test=weighted_metrics(all_inner, by_id),
            outer_holdout_stress_test=weighted_metrics(outer_rows, by_id),
        )
    return dict(
        status="verified_numeric_traces_no_refit",
        actual_control_fit_calls=21,
        fixed_chemical_route="structure_methods_all",
        role_metrics=role_metrics,
        full_cohort_metric_not_reported=True,
    )


def verify_search(run, result, routes, api):
    """Cross-check retained trial wrappers, Optuna SQLite and the admitted CV winner."""
    search = result["search"]
    directory = api.contained(run, search["search_dir"])
    require(
        search.get("backend") == "optuna"
        and search.get("sampler") == "TPESampler"
        and search.get("pruner") == "NopPruner"
        and search.get("scope") == "training_only_grouped_cv; heldout_not_available_to_search"
        and search.get("budget") == dict(requested=128, used=128, remaining=0, pairs=88)
        and search.get("eligible_model_route_pairs") == 88,
        "Optuna128/eight models/11routes scope",
    )
    c = search["contract"]
    require(
        c.get("input_signature") == result["input_hash"]
        and c.get("backend") == "optuna"
        and c.get("task_kind") == "regression"
        and len(c["pairs"]) == 88
        and {tuple(p) for p in c["pairs"]}
        == {(route, model) for route in routes for model in MODELS}
        and c.get("implementation_sha256")
        == api.sha(Path(run) / "scripts/peptide_modeling_search.py")
        and object_sha(c) == search["namespace"]
        and api.read(directory / "contract.json") == c,
        "search namespace/actual source/pairs",
    )
    paths = sorted((directory / "trials").glob("*.json"))
    require(len(paths) == 128, "actual128 trial receipts")
    records = []
    hashes = {}
    for number, path in enumerate(paths):
        wrapped = api.read(path)
        r = wrapped["evaluation"]
        require(
            path.name == f"{number:06d}.json"
            and wrapped["record_sha256"] == object_sha(r)
            and r.get("search_trial") == number
            and r.get("search_input_signature") == search["namespace"]
            and r.get("status") in {"completed", "failed"}
            and r.get("input_hash") == result["input_hash"]
            and r.get("requested_folds") == r.get("actual_folds") == 5
            and (r["route"], r["model"]) in {tuple(p) for p in c["pairs"]},
            "trial wrapper/namespace/status",
        )
        if r["status"] == "completed":
            require(
                math.isfinite(float(r["score"]))
                and math.isclose(r["score"], -r["metrics"]["mae"], rel_tol=1e-9, abs_tol=1e-10)
                and all(
                    r[k] == 5
                    for k in ("requested_inner_folds", "actual_inner_folds", "n_inner_folds")
                    if k in r
                ),
                "successful CV trial score/fivefold",
            )
        records.append(r)
        hashes[str(path.relative_to(run))] = api.sha(path)
    require(
        {(r["route"], r["model"]) for r in records} == {tuple(p) for p in c["pairs"]}
        and sorted(search["coverage_attempted"]) == sorted(c["pairs"])
        and search["completed"] == sum(r["status"] == "completed" for r in records)
        and search["failed"] == sum(r["status"] == "failed" for r in records),
        "actual method/model coverage or attempt counts",
    )
    db = api.contained(run, search["storage_path"])
    require(db == directory / "study.sqlite3", "unexpected Optuna storage")
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        studies = connection.execute("SELECT study_id, study_name FROM studies").fetchall()
        require(
            len(studies) == 1
            and studies[0][1] == search["study_name"] == "peptide_" + search["namespace"][:24],
            "SQLite study namespace",
        )
        study_id = studies[0][0]
        require(
            connection.execute(
                "SELECT direction FROM study_directions WHERE study_id=?", (study_id,)
            ).fetchall()
            == [("MAXIMIZE",)],
            "SQLite search direction",
        )
        sql_trials = connection.execute(
            "SELECT trial_id, number, state FROM trials WHERE study_id=? ORDER BY number",
            (study_id,),
        ).fetchall()
        require(
            len(sql_trials) == 128 and [t[1] for t in sql_trials] == list(range(128)),
            "SQLite trial membership",
        )
        for r, (tid, _number, state) in zip(records, sql_trials, strict=True):
            attrs = {
                k: json.loads(v)
                for k, v in connection.execute(
                    "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (tid,)
                )
            }
            require(
                state == ("COMPLETE" if r["status"] == "completed" else "FAIL")
                and attrs.get("evaluation_result") == r
                and attrs.get("candidate") == {k: r[k] for k in ("route", "model", "params")},
                "SQLite evaluation/candidate/state differs from saved trial",
            )
            objectives = connection.execute(
                "SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective", (tid,)
            ).fetchall()
            require(
                objectives == ([(r["score"],)] if r["status"] == "completed" else []),
                "SQLite objective differs from CV score",
            )
            actual = {}
            for name, value, distribution in connection.execute(
                "SELECT param_name, param_value, distribution_json "
                "FROM trial_params WHERE trial_id=?",
                (tid,),
            ):
                spec = json.loads(distribution)
                if name == "candidate_pair":
                    require(
                        spec.get("name") == "CategoricalDistribution"
                        and spec["attributes"]["choices"] == [f"pair_{i:04d}" for i in range(88)],
                        "SQLite candidate pair distribution is not the frozen88-pair panel",
                    )
                actual[name] = (
                    spec["attributes"]["choices"][int(value)]
                    if spec["name"] == "CategoricalDistribution"
                    else int(value)
                    if spec["name"] == "IntDistribution"
                    else value
                )
            pair = actual.pop("candidate_pair", None)
            require(
                isinstance(pair, str)
                and re.fullmatch(r"pair_\d{4}", pair)
                and tuple(c["pairs"][int(pair.split("_")[1])]) == (r["route"], r["model"])
                and actual == {r["model"] + "." + k: v for k, v in r["params"].items()},
                "SQLite actual suggested parameters differ",
            )
    finally:
        connection.close()
    admission = search["candidate_fit_admission"]
    require(
        admission.get("policy") == "descending_inner_cv_score_then_outer_train_fit_and_persistence"
        and admission.get("validation_labels") == "outer_train_only"
        and admission.get("holdout_used") is False,
        "CV fit admission used holdout/refit",
    )

    def key(r):
        return object_sha([r["route"], r["model"], r["params"]], compact=False)

    rejected = set()
    for r in admission["rejected_candidates"]:
        require(
            any(
                v["status"] == "completed"
                and all(v[k] == r[k] for k in ("route", "model", "params"))
                and v["score"] == r["inner_cv_score"]
                for v in records
            )
            and r.get("reason")
            and r.get("stage")
            in {
                "estimator_construction",
                "outer_train_fit",
                "training_probe_persistence",
                "development_oof_reconstruction",
            },
            "rejected candidate lacks successful CV evidence",
        )
        rejected.add(key(r))
    admitted = [r for r in records if r["status"] == "completed" and key(r) not in rejected]
    chosen = {
        "route": result["selected_route"],
        "model": result["selected_model"],
        "params": result["selected_params"],
    }
    require(
        admitted
        and admission.get("selected_candidate_key") == key(chosen)
        and any(
            key(r) == key(chosen) and r["score"] == max(v["score"] for v in admitted)
            for r in admitted
        ),
        "selected model not highest admitted inner-CV candidate",
    )
    # Search scores belong to retained candidate executions; saved OOF belongs
    # to the separately reconstructed admitted estimator. Each is checked
    # against its own original records rather than silently equating refits.
    api.report_repair_trials(run, result)
    hashes[str(db.relative_to(run))] = api.sha(db)
    return hashes


def verify_refit(run, result, task, source, feature_binding, check, api):
    paths, _ = api.verify_hashes(
        run,
        result,
        {
            "model",
            "refit_model",
            "oof",
            "holdout",
            "outer_split",
            "input_contract",
            "model_contract",
        },
    )
    contract, inputs, outer = (
        api.read(paths[k]) for k in ("model_contract", "input_contract", "outer_split")
    )
    require(
        result.get("status") == result.get("refit_status") == "completed"
        and result.get("deployment_ready") is True
        and result.get("task_id") == TASK
        and result.get("scope_label") == "common"
        and result.get("requested_inner_folds") == result.get("actual_inner_folds") == 5
        and paths["model"] != paths["refit_model"]
        and result.get("refit_attempts")
        and result["refit_attempts"][-1].get("status") == "completed"
        and contract.get("refit_attempts") == result["refit_attempts"]
        and contract.get("task") == inputs.get("task") == task
        and contract.get("common_record_ids") == [r["record_id"] for r in source]
        and contract.get("model_fit_scope") == "outer_train_only"
        and contract.get("optional_refit_fit_scope") == "all_common_after_evaluation"
        and contract.get("holdout_not_used_for_selection") is True
        and contract.get("target_unit") == "L"
        and contract.get("target_transform") == "log10"
        and contract.get("selected_route") == result.get("selected_route")
        and contract.get("selected_model") == result.get("selected_model"),
        "distinct all27 refit/source/evaluation scope",
    )
    settings = inputs["config"]
    routes = feature_binding["features"]["routes"]
    require(
        settings.get("refit_full") is True
        and settings.get("inner_folds") == 5
        and settings.get("search_backend") == "optuna"
        and settings.get("search_budget") == 128
        and set(settings.get("models", [])) == MODELS
        and len(settings["models"]) == 8
        and settings.get("n_jobs") == 1
        and set(settings["routes"]) == set(routes)
        and inputs.get("implementation_sha256")
        == api.sha(Path(run) / "scripts/peptide_modeling_train.py")
        == TRAIN_SHA,
        "frozen actual trainer/Optuna8models128/CPU settings",
    )
    require(
        outer == check["outer_receipt"]
        and outer["train_records"] == check["train_record_ids"]
        and outer["test_records"] == check["holdout_record_ids"],
        "evaluated outer split differs from node prefreeze",
    )
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=task,
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=settings,
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=outer["input_hash"],
    )
    require(
        object_sha(signature, compact=False)
        == inputs.get("input_hash")
        == result.get("input_hash")
        == contract.get("input_hash"),
        "signed training inputs differ",
    )
    require(
        set(inputs["feature_hashes"]) == set(routes) == set(inputs["linear_solver_policies"]),
        "all signed feature routes/policies",
    )
    for route in routes:
        design = designs(feature_binding, route, source, api)["molecule_plus_context"]
        expected = dict(
            names=design["names"],
            matrix_sha256=design["source_float32_sha256"],
            mask_sha256=hashlib.sha256(b"\x01" * 27).hexdigest(),
        )
        require(inputs["feature_hashes"][route] == expected, "actual signed float32 design changed")
        policy = dict(
            version="dimension_based_linear_policy_v2",
            selection_basis="raw_route_feature_count_only",
            feature_count=len(design["names"]),
            dense_classification_threshold=256,
            solver="lsqr",
            scaler_with_mean=False,
            max_iter=None,
            tol=1e-4,
            convergence_warning="candidate_failure",
        )
        require(inputs["linear_solver_policies"][route] == policy, "actual regression LSQR policy")
    joint = designs(feature_binding, result["selected_route"], source, api)["molecule_plus_context"]
    require(contract.get("feature_names") == joint["names"], "selected model feature schema")
    if result["selected_model"] == "linear":
        require(
            contract.get("linear_solver_policy")
            == inputs["linear_solver_policies"][result["selected_route"]],
            "selected solver differs",
        )
    trials = verify_search(Path(run), result, routes, api)
    return paths, contract, inputs, joint, trials


@audit_errors
def verify_seed(entry, round_entry, seed, task, api):
    template = verify_source(entry, task, api)
    source = template["source"]
    run = api.contained(entry["output_root"], round_entry["run_dir"])
    require(
        seed in SEEDS
        and round_entry.get("seed") == seed
        and round_entry.get("status") == "completed"
        and round_entry.get("phase") == "verified_terminal"
        and round_entry.get("all27_refit_verified") is True
        and round_entry.get("config_sha256") == api.sha(run / "config.json"),
        "custom seed terminal/config evidence",
    )
    actual = source_at(run, api, scripts_root=run / "scripts")
    require(
        actual["source"] == source and actual["task"] == template["task"],
        "round source closure/proof/task differs",
    )
    require(api.read(run / "task_manifest.json")["tasks"] == [actual["task"]], "round task count")
    check = dict(
        actual["checks"][seed], outer_receipt=actual["node"]["outer_split_receipts"][str(seed)]
    )
    staged = api.read(run / "round_source_closure_verification.json")
    require(
        staged == round_entry.get("staged_source_closure")
        and staged.get("status") == "verified_no_fit_staged_source_closure"
        and staged.get("seed") == seed
        and staged.get("source_closure_sha256") == CLOSURE_SHA
        and staged.get("source_closure_files_checked") == 79
        and staged.get("exact_source_proof_sha256") == PROOF_SHA
        and staged.get("exact_task_table_sha256") == TABLE_SHA
        and staged.get("portable_source_and_all_routes_passed") is True
        and staged.get("outer_split_sha256")
        == api.sha(run / "splits" / TASK / "group_assignment.json")
        and staged.get("config_sha256") == api.sha(run / "config.json")
        and all(
            staged.get(k) == 0
            for k in ("estimator_fits", "searches", "model_unpickles", "chemistry_imports")
        ),
        "actual production round was not source-closure staged",
    )
    for name in actual["closure"]["runtime_receipts_required_when_bound"]:
        require(
            staged["runtime_receipts_copied"].get(name)
            == api.sha(run / name)
            == api.sha(Path(entry["template"]) / name),
            "round lost runtime receipt/dependency",
        )
    verification_path = run / "verification.json"
    verification = api.read(verification_path)
    require(
        verification.get("status") == "verified"
        and verification.get("source_task_manifest_sha256") == api.sha(run / "task_manifest.json")
        and round_entry["audit"].get("verification_sha256") == api.sha(verification_path)
        and round_entry["audit"].get("actual_folds") == 5
        and round_entry["audit"].get("backend") == "optuna"
        and round_entry["audit"].get("method_model_coverage_met") is True
        and round_entry["audit"].get("attempted_independent_methods") == 10
        and round_entry["audit"].get("attempted_model_families") == 8,
        "seed verification/search admission",
    )
    features = verify_features(run, source, actual["ontology"], api)
    result_path = run / "results/full" / (TASK + "_common.json")
    result = api.read(result_path)
    paths, contract, inputs, joint, trials = verify_refit(
        run, result, actual["task"], source, features, check, api
    )
    require(
        inputs["versions"].get("numpy") == actual["node"]["numpy_version"]
        and inputs["versions"].get("scikit_learn") == actual["node"]["sklearn_version"]
        and contract.get("versions") == inputs["versions"] == result.get("versions")
        and result["search"].get("optuna_version")
        == inputs["versions"].get("optuna")
        == result["search"]["contract"].get("backend_version"),
        "fitted library versions differ from actual pre-frozen node runtime/search",
    )
    require(
        inputs["config"]["seed"] == seed
        and result.get("n_train") == len(check["train_record_ids"])
        and result.get("n_test") == len(check["holdout_record_ids"]),
        "actual seed/train/holdout counts",
    )
    current_config = api.read(run / "config.json")
    require(
        all(
            current_config["training"][key] == inputs["config"][key]
            for key in (
                "seed",
                "routes",
                "models",
                "inner_folds",
                "search_backend",
                "search_budget",
                "n_jobs",
                "refit_full",
            )
        )
        and current_config["resources"].get("training_host") == "node23"
        and current_config["resources"].get("cpu_budget") == 1
        and current_config["resources"].get("max_parallel_tasks") == 1,
        "actual run configuration differs from signed training/resource policy",
    )
    inner = api.actual_folds(paths["oof"])
    held = api.table(paths["holdout"])
    api.source_membership(inner + held, source, require_all=True)
    provenance = api.read(run / "run_phase_numeric_provenance.json")
    solver = provenance["selected_solver"]
    require(
        provenance.get("explicit_run_phase_only") is True
        and provenance.get("report_only_fit_calls") == 0
        and provenance.get("result_sha256") == api.sha(result_path)
        and provenance.get("controls_sha256") == api.sha(run / "fixed_float64_controls.json")
        and solver.get("actual_model_artifact_sha256") == api.sha(paths["model"])
        and solver.get("training_source_sha256") == TRAIN_SHA
        and solver.get("selected_model") == result["selected_model"]
        and solver.get("input_dtype") == "float32"
        and solver.get("feature_width") == len(joint["names"])
        and solver.get("full_matrix_sha256") == joint["source_float32_sha256"],
        "actual run-only model/control numeric provenance",
    )
    report_path = run / "reports/vd_absolute_native_diagnostics.json"
    report = api.read(report_path)
    require(
        round_entry["molecular_reporting"]
        == dict(path="reports/vd_absolute_native_diagnostics.json", sha256=api.sha(report_path)),
        "mandatory per-round report hash",
    )
    require(
        report.get("schema") == SCHEMA
        and report.get("status") == "verified_typed_source_annotation_stress_test_only"
        and report.get("task_id") == TASK
        and report.get("seed") == seed
        and report.get("required_seeds") == list(SEEDS)
        and report.get("result_sha256") == api.sha(result_path)
        and report.get("source_proof_sha256") == PROOF_SHA
        and report.get("oof_sha256") == api.sha(paths["oof"])
        and report.get("holdout_sha256") == api.sha(paths["holdout"])
        and report.get("model_sha256") == api.sha(paths["model"])
        and report.get("report_only_estimator_fits") == 0
        and report.get("model_selection_repeated") is False
        and report.get("every_estimand_independently_validated") is False,
        "mandatory report scientific/source/prediction scope",
    )
    require(
        report.get("warning") == actual["task"]["scientific_definition"]["warning"],
        "source-native stress-test warning altered",
    )
    verify_evaluation(report["diagnostics"], source, inner, held, check, joint, solver)
    full = designs(features, "structure_methods_all", source, api)
    controls = api.read(run / "fixed_float64_controls.json")
    summary = verify_controls(source, check, controls, full)
    r_control = report["fixed_controls"]
    require(
        r_control.get("path") == "fixed_float64_controls.json"
        and r_control.get("sha256") == api.sha(run / "fixed_float64_controls.json"),
        "control trace report hash",
    )
    reported_summary = r_control["verification"]
    require(
        set(reported_summary)
        == {
            "status",
            "actual_control_fit_calls",
            "fixed_chemical_route",
            "role_metrics",
            "full_cohort_metric_not_reported",
        }
        and set(reported_summary["role_metrics"]) == set(full)
        and all(
            set(v) == {"inner_oof_stress_test", "outer_holdout_stress_test"}
            for v in reported_summary["role_metrics"].values()
        ),
        "fixed-control report added a full-refit performance score",
    )
    require(
        all(
            reported_summary.get(k) == summary[k]
            for k in (
                "status",
                "actual_control_fit_calls",
                "fixed_chemical_route",
                "full_cohort_metric_not_reported",
            )
        ),
        "control report fit/count/scope",
    )
    for name in full:
        for role in ("inner_oof_stress_test", "outer_holdout_stress_test"):
            check_metrics(
                reported_summary["role_metrics"][name][role], summary["role_metrics"][name][role]
            )
    refit = report["separate_all27_refit"]
    require(
        refit.get("status") == "verified"
        and refit.get("records") == 27
        and refit.get("molecular_groups") == 10
        and refit.get("training_record_ids") == [r["record_id"] for r in source]
        and refit.get("model_sha256") == api.sha(paths["refit_model"])
        and refit.get("model_contract_sha256") == api.sha(paths["model_contract"])
        and refit.get("independently_scored") is False
        and refit.get("new_molecule_or_subtype_generalization_created") is False
        and refit.get("all_observation_context_support") == check["all27_refit_support"],
        "all27 final refit or unsupported-type claims changed",
    )
    return dict(
        status="verified",
        run_dir=str(run),
        actual_folds=5,
        source_records=27,
        molecular_groups=10,
        fixed_float64_control_fits=21,
        unseen_estimand_records=22,
        unseen_joint_context_records=27,
        all27_refit_verified=True,
        report_sha256=api.sha(report_path),
        trials=trials,
    )


@audit_errors
def verify_reporting(entry, journal, task, api):
    verify_source(entry, task, api)
    root = Path(entry["output_root"])
    config = api.read(entry["config_path"])
    require(
        api.sha(entry["config_path"]) == entry["config_sha256"] == journal.get("config_sha256")
        and config.get("template_run") == entry["template"]
        and Path(config["output_root"]) == root
        and config.get("required_seeds", list(SEEDS)) == list(SEEDS)
        and config.get("task_ids") == [TASK]
        and config.get("workers", 1) == 1
        and config.get("additional_twenty_hour_loop", False) is False,
        "custom bounded controller configuration",
    )
    require(
        journal.get("schema") == SCHEMA
        and journal.get("task_id") == TASK
        and journal.get("status") == "completed"
        and journal.get("mandatory_reporting_state") == "verified"
        and journal.get("required_seeds") == list(SEEDS)
        and journal.get("additional_twenty_hour_loop") is False
        and len(journal.get("rounds", [])) == 3
        and [r["seed"] for r in journal["rounds"]] == list(SEEDS)
        and api.read(root / "campaign_status.json") == journal,
        "completed without mandatory three-seed report proof",
    )
    owner = entry["owned_process_identity"]
    recorded = journal["process_identity"]
    current_owner = api.process_identity(owner["pid"])
    require(
        owner["pid"] == recorded["pid"] == journal["pid"]
        and owner["start_ticks"] == recorded["start_ticks"]
        and (
            current_owner is None
            or current_owner.get("state") == "Z"
            or current_owner.get("start_ticks") != owner["start_ticks"]
        )
        and api.identity_state(owner, current_owner) != "same_live_process",
        "owned custom controller has not released",
    )
    require(
        recorded.get("argv")
        and MODULE in " ".join(recorded["argv"])
        and str(entry["config_path"]) in recorded["argv"],
        "custom owner invocation not source/config bound",
    )
    durable = Path(config["durable_root"])
    require(
        api.read(durable / "campaign_status.json") == journal,
        "durable controller/report state differs",
    )
    reports = []
    for i, round_entry in enumerate(journal["rounds"]):
        seed = SEEDS[i]
        run = api.contained(root, round_entry["run_dir"])
        require(run == root / f"round_{i:03d}_seed_{seed}", "unexpected custom round namespace")
        reports.append(verify_seed(entry, round_entry, seed, task, api))
        expected = {
            p.relative_to(run).as_posix(): api.sha(p) for p in run.rglob("*") if p.is_file()
        }
        require(
            not any(p.is_symlink() for p in run.rglob("*")),
            "durable scientific archive uses symlinks",
        )
        archive = round_entry["archive"]
        path = api.contained(durable, archive["path"])
        api.report_repair_archive(path, archive, expected, run.name)
    return dict(
        status="verified",
        source_records_per_seed=27,
        molecular_groups_per_seed=10,
        required_seed_reports=3,
        fixed_float64_control_fits=63,
        all27_final_refits_verified=3,
        unseen_estimand_records_per_seed=22,
        unseen_joint_context_records_per_seed=27,
        source_annotation_only=True,
        every_estimand_independently_validated=False,
        mandatory_reporting_and_durable_archives_verified=True,
        reports=reports,
    )
