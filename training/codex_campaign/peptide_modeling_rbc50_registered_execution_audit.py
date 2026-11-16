"""Read-only, standard-library audit of the frozen registered-reference RBC50 lane.

The ``api`` argument is the already-loaded shared execution auditor.  Passing it
explicitly avoids imports of either the learner or the parent auditor, including
NumPy, RDKit and model deserializers.  No function fits or changes an artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

TASK = "TASK_RBC50_REGISTERED_270d2dbf870da342cf2e85e8"
QUANTITY = "source_reported_nominal_RBC50_given_species_and_registered_reference"
SCHEMA = "rbc50_registered_reference_species_v1"
REPORT = "rbc50_registered_species_molecular_diagnostics_v1"
MODULE = "peptide_modeling_rbc50_registered_bridge.py"
TOOL_SHA = "be5b54f5f647d0a2321ffe846f72d39e2197182741e65fea65dedd9742486a71"
PROOF_SHA = "6cc423fb6dcd2a6f5d159dad8c260c8e1dc714fe6dd1ddaf22ec0c053a0dd95b"
TABLE_SHA = "4b274379506f4bc9127969c6942bc528a509f5f21eecba82da0e626cda9c1669"
BASE_SHA = "1d5d2afeae4aac319a15614b80dd0af57a467b25ee23873fb877324e2971d44a"
PLAN_SHA = "43bc211d76f8e1e8f538d203039e92cd7c84c015835a5276c19cf20955dc5e7a"
FEATURE_SHA = "c4d3590ff27c09f71f31b7843e0e90c7f1412fd7eb5de8ec8d2e2b54284f1078"
ARTIFACT_MAP_SHA = "38410bcb79cc5f1d9addc5c1f6786bca9b2991cf56b4a18f9e3c06d0e3773469"
TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
SEEDS = (42, 2026, 3407)
CONTEXT = "rbc_species_context"
NEW_IDS = (
    "OLD:PV13:2e3023a37d5a1e2edaa7d9ad",
    "OLD:PV13:000c5469597c9cdb35104212",
    "OLD:PV13:278ae838c9210f75ca437655",
    "OLD:PV13:33bc227b010794efc5d50f24",
)
HORSE = NEW_IDS[0]
POINTS = {
    NEW_IDS[0]: ("53.12", "horse", "DBAASP:16101", [[15, 21]]),
    NEW_IDS[1]: ("11.6", "human", "DBAASP:4286", [[3, 19], [7, 21], [12, 26]]),
    NEW_IDS[2]: ("25.5", "human", "DBAASP:4287", [[4, 20], [8, 22], [13, 27]]),
    NEW_IDS[3]: ("13.3", "human", "DBAASP:4289", [[3, 19], [7, 21], [12, 26]]),
}
DEPENDENCIES = {
    "peptide_modeling_hc50_primary_bridge.py": (
        "5bc218673c9cde599c24e7e66a2b9b3e11bd411254419f91b9bbbdfe97e6cb39"
    ),
    "peptide_modeling_mhc10_native_bridge.py": (
        "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
    ),
    "peptide_modeling_hc50_precision_report.py": (
        "bf83afa1ffb341ad103c723952ac34d7465d6a9646414f6170a70f47046be701"
    ),
    "peptide_modeling_control_float64_recovery.py": (
        "a7583f837bb12d3a5086b9c73f52746564872d894f7f1e5e376703a1dc515c12"
    ),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def object_sha(value, *, compact=True):
    kwargs = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def is_task(task):
    return task.get("task_id") == TASK


def _source_at(root, frozen, api):
    root = Path(root)
    descriptors = frozen.get("registered_reference_observation_coverage")
    require(isinstance(descriptors, list) and len(descriptors) == 2, "RBC dual coverage missing")
    proof_path = api.contained(root, descriptors[0]["proof_path"])
    require(
        all(
            d["proof_path"] == descriptors[0]["proof_path"] and d["proof_sha256"] == PROOF_SHA
            for d in descriptors
        )
        and api.sha(proof_path) == PROOF_SHA,
        "RBC source proof or dual descriptor binding changed",
    )
    proof = api.read(proof_path)
    expected = [dict(d) for d in descriptors]
    for d in expected:
        d.pop("proof_path")
        d.pop("proof_sha256")
    require(
        expected == proof["original_endpoint_coverage"]
        and [d["original_endpoint"] for d in expected] == ["cytotoxicity_HC50", "hemolysis_HD50"]
        and [d["included_original_records"] for d in expected] == [1, 3]
        and [d["original_labelled_records"] for d in expected] == [1, 4]
        and [d["record_ids"] for d in expected] == [[HORSE], list(NEW_IDS[1:])],
        "RBC original source coverage was enlarged or relabelled",
    )
    definition = frozen.get("scientific_definition", {})
    require(
        frozen.get("task_id") == proof["task_id"] == TASK
        and frozen.get("endpoint") == "hemolysis_HC50"
        and frozen.get("canonical_quantity_id") == proof["quantity_id"] == QUANTITY
        and proof["query_mode"] == "nominal_registered_reference_only"
        and definition == frozen.get("task_definition")
        and definition.get("implementation_sha256") == proof["implementation_sha256"] == TOOL_SHA
        and definition.get("molecular_reporting_schema") == REPORT
        and definition.get("scientific_warning") == proof["warning"] == frozen["scientific_warning"]
        and definition.get("mandatory_prediction_inputs")
        == ["rbc_species", "quantity_id", "reference_mode", "identity_tier"]
        and definition.get("context_is_chemical_method") is False
        and definition.get("separate_all_common_refit_required") is True
        and definition.get("horse_independent_molecular_groups") == 1,
        "RBC source/query/training definition changed",
    )
    for key in (
        "primary_sample_graph_verified",
        "primary_sample_disulfide_assignment_verified",
        "biological_state_contrast_supported",
        "reference_tier_is_input_feature",
        "all38_fit_method_uniform",
    ):
        require(
            proof.get(key) is False and definition.get(key) is False,
            "RBC reference or measurement limits were promoted: " + key,
        )
    require(
        proof.get("sample_redox_state") == "not_independently_verified"
        and proof.get("registered_disulfide_maps_verified") is True
        and proof.get("registered_disulfide_map_record_ids") == list(NEW_IDS)
        and all(
            proof.get(k) is False
            for k in (
                "clinical_queries_supported",
                "same_parent_ChaC_series_independence_proven",
            )
        )
        and all(
            proof.get(k) is True
            for k in (
                "primary_censored_RN3_not_used",
                "secondary178_33_not_used",
                "old_RN3_250_not_used",
                "LHH1_old200_or_new187_2_not_used",
            )
        ),
        "RBC source exclusions, registered/sample scope or homology claims changed",
    )
    for name, expected_sha in {MODULE: TOOL_SHA, **DEPENDENCIES}.items():
        require(
            api.sha(root / "scripts" / name) == expected_sha,
            "RBC frozen implementation changed: " + name,
        )
    path = api.contained(root, proof["source_table"]["path"])
    require(
        path == api.contained(root, frozen["table_path"])
        and api.sha(path) == proof["source_table"]["sha256"] == TABLE_SHA,
        "RBC exact frozen training table changed",
    )
    source = api.table(path)
    old_path = api.contained(root, proof["baseline34"]["path"])
    require(
        api.sha(old_path) == proof["baseline34"]["sha256"] == BASE_SHA,
        "RBC baseline34 source snapshot changed",
    )
    old = api.table(old_path)
    require(
        len(source) == len({r["record_id"] for r in source}) == 38
        and len(old) == 34
        and all(all(r[k] == b[k] for k in b) for r, b in zip(source[:34], old, strict=True)),
        "RBC baseline34 fields or complete38 membership changed",
    )
    by_id = {r["record_id"]: r for r in source}
    require(
        set(by_id) - {r["record_id"] for r in old} == set(NEW_IDS)
        and Counter(r["rbc_species"] for r in source) == {"human": 37, "horse": 1}
        and len({r["canonical_identity_group"] for r in source})
        == len({r["split_group"] for r in source})
        == 34,
        "RBC exact molecular/species counts changed",
    )
    graph_families = {}
    for r in source:
        graph = r["canonical_smiles"]
        require(
            r["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
            and graph_families.setdefault(graph, r["split_group"]) == r["split_group"]
            and r["unit"] == "uM"
            and r["target_transform"] == "log10"
            and r["rbc50_quantity_id"] == QUANTITY
            and r["rbc50_reference_mode"] == proof["query_mode"]
            and math.isfinite(float(r["target"])),
            "RBC source graph, quantity or transformed target changed",
        )
    for rid, (value, species, registry, pairs) in POINTS.items():
        row = by_id[rid]
        require(
            row["value"] == value
            and row["rbc_species"] == species
            and row["registered_reference_id"] == registry
            and json.loads(row["registered_SS_pairs"]) == pairs
            and math.isclose(
                float(row["target"]), math.log10(float(value)), rel_tol=0, abs_tol=1e-12
            )
            and row["identity_tier"] == "registered_reference_only"
            and row["reference_backbone"] == "linear"
            and row["head_to_tail_bond"] == "false"
            and row["primary_sample_graph_verified"] == "false"
            and row["sample_redox_state"] == "not_independently_verified",
            "RBC registered reference point, topology or limited sample claim changed",
        )
    for witness in proof["source_objects"].values():
        require(
            api.sha(api.contained(root, witness["path"])) == witness["sha256"],
            "RBC primary/registered raw snapshot changed",
        )
    observations = proof["observations"]
    require(
        len(observations) == 4 and {r["record_id"] for r in observations} == set(NEW_IDS),
        "RBC source observations changed",
    )
    for item in observations:
        rid = item["record_id"]
        registration = item["registration"]
        require(
            all(by_id[rid].get(k) == v for k, v in item["registered_reference_model_row"].items())
            and registration["primary_sample_graph_verified"] is False
            and registration["peptide_backbone_head_to_tail_closed"] is False
            and registration["disulfide_pairs_1based_from_raw_intrachainBonds"] == POINTS[rid][3]
            and registration["canonical_isomeric_smiles"] == by_id[rid]["canonical_smiles"],
            "RBC registered point lost exact lineage or explicit registered S-S graph",
        )
    plan_path = api.contained(root, proof["predeclared_splits"]["path"])
    require(
        api.sha(plan_path) == proof["predeclared_splits"]["sha256"] == PLAN_SHA,
        "RBC actual-node fivefold master changed",
    )
    checks = api.read(plan_path)
    require(
        len(checks["seeds"]) == 3 and {c["seed"] for c in checks["seeds"]} == set(SEEDS),
        "RBC predeclared seeds missing",
    )
    for check in checks["seeds"]:
        require(
            check["actual_folds"] == 5
            and len(check["folds"]) == 5
            and [f["fold"] for f in check["folds"]] == list(range(5))
            and check["final_refit_record_ids"] == [r["record_id"] for r in source]
            and api.sha(api.contained(root, check["split_path"])) == check["split_sha256"],
            "RBC actual-node predeclared memberships changed",
        )
        train, held = set(check["train_record_ids"]), set(check["holdout_record_ids"])
        require(not train & held and train | held == set(by_id), "RBC outer membership changed")
        require(
            {by_id[i]["split_group"] for i in train}.isdisjoint(
                {by_id[i]["split_group"] for i in held}
            ),
            "RBC outer molecule leakage",
        )
        seen = []
        for fold in check["folds"]:
            a, b = set(fold["train_record_ids"]), set(fold["validation_record_ids"])
            require(
                not a & b
                and a | b == train
                and a
                and b
                and {by_id[i]["split_group"] for i in a}.isdisjoint(
                    {by_id[i]["split_group"] for i in b}
                ),
                "RBC inner molecule leakage",
            )
            seen.extend(fold["validation_record_ids"])
        require(
            len(seen) == len(set(seen)) == len(train) and set(seen) == train,
            "RBC actual inner folds do not partition outer train",
        )
    return proof, source, checks


def verify_source(entry, task, api):
    root = Path(entry["template"])
    manifest = api.read(root / "task_manifest.json")
    require(len(manifest["tasks"]) == 1, "RBC template contains unrelated tasks")
    frozen = manifest["tasks"][0]
    require(
        task.get("task_id") == TASK
        and task.get("scientific_definition") == frozen["task_definition"]
        and task.get("registered_reference_observation_coverage")
        == frozen["registered_reference_observation_coverage"],
        "RBC registry task changed",
    )
    proof, source, checks = _source_at(root, frozen, api)
    require(
        task["table_relative"] == frozen["table_path"]
        and task["table_sha256"] == TABLE_SHA
        and task["records"] == 38,
        "RBC registry source table changed",
    )
    pin = entry.get("native_feature_contract")
    require(
        isinstance(pin, dict) and {"sha256", "registry_sha256"} <= set(pin),
        "RBC registry omitted its frozen feature/schema contract",
    )
    verify_features(root, source, api, pin)
    return proof, frozen, source, checks


def verify_features(root, source, api, frozen_pin=None):
    root = Path(root)
    pointer, registry_path = root / "features/manifest.json", root / "representation_methods.json"
    if frozen_pin is not None:
        require(
            api.sha(pointer) == frozen_pin["sha256"]
            and api.sha(registry_path) == frozen_pin["registry_sha256"],
            "RBC frozen feature pointer/registry changed",
        )
    f = api.read(pointer)
    require(
        f["schema"] == SCHEMA
        and f["input_fingerprint"] == FEATURE_SHA
        and object_sha(f["configuration"]) == FEATURE_SHA
        and object_sha(f["artifact_hashes"]) == ARTIFACT_MAP_SHA
        and f["configuration"]["species_by_record"]
        == {r["record_id"]: r["rbc_species"] for r in source}
        and f["configuration"]["record_ids"] == [r["record_id"] for r in source],
        "RBC feature/source/configuration fingerprint changed",
    )
    cache_name = str(f["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    require(cache_name == FEATURE_SHA[:16], "RBC cache namespace changed")
    cache = api.contained(root, "features/" + cache_name)
    for name, pin in f["artifact_hashes"].items():
        require(api.sha(api.contained(cache, name)) == pin, "RBC feature artifact changed")
    registry = api.read(registry_path)
    methods = registry["methods"]
    require(
        registry == f["representation_methods"]
        and registry["independent_method_count"] == 10
        and len(methods) == 10
        and len({m["algorithm_family_id"] for m in methods}) == 10
        and all(m["independent"] is True and m["source_kind"] == "structure" for m in methods)
        and registry["context_method_increment"] == 0,
        "RBC ten independent chemical methods/context accounting changed",
    )
    groups = f["groups"]
    require(
        len(groups) == 14
        and sum(g["shape"][1] for k, g in groups.items() if k != CONTEXT) == 15212
        and groups[CONTEXT]["shape"] == [38, 2]
        and groups[CONTEXT]["key_kind"] == "record_id"
        and groups[CONTEXT]["feature_names"] == ["human", "horse"]
        and groups[CONTEXT]["chemical_method"] is False,
        "RBC chemical/species dimensions or key schema changed",
    )
    require(
        api.table(cache / f["record_index"])
        == [
            dict(
                record_id=r["record_id"],
                structure_key=r["canonical_identity_group"],
                sequence_key="",
            )
            for r in source
        ],
        "RBC complete graph aliases changed",
    )
    require(
        set(f["routes"]) == {m["route"] for m in methods} | {"structure_methods_all"}
        and f["routes"]["structure_methods_all"] == list(groups)
        and all(f["routes"][m["route"]] == m["group_names"] + [CONTEXT] for m in methods)
        and all(v.count(CONTEXT) == 1 for v in f["routes"].values()),
        "RBC route repeats or omits species context",
    )
    parts, names = {}, {}
    for name, g in groups.items():
        index_rows = api.table(cache / g["index"])
        index = {r["key"]: int(r["row_idx"]) for r in index_rows}
        n = 38 if name == CONTEXT else 34
        header, body = api.hc50_precision_npy(cache / g["matrix"])
        mh, masks = api.hc50_precision_npy(cache / g["valid"])
        require(
            len(index) == len(index_rows) == n
            and set(index.values()) == set(range(n))
            and header["shape"] == tuple(g["shape"])
            and header["shape"][0] == n
            and len(header["shape"]) == 2
            and header["shape"][1] == len(g["feature_names"])
            and mh["shape"] == (n,)
            and mh["descr"] == "|b1"
            and all(b == 1 for b in masks),
            "RBC feature array/index/mask shape changed",
        )
        if name != CONTEXT:
            require(g["key_kind"] == "canonical_identity_group", "RBC graph key scheme changed")
        rows = {}
        width = header["shape"][1]
        stride = width * (4 if header["descr"] == "<f4" else 1)
        for r in source:
            key = r["record_id"] if name == CONTEXT else r["canonical_identity_group"]
            i = index[key]
            data = body[i * stride : (i + 1) * stride]
            if header["descr"] == "|u1":
                data = struct.pack("<" + "f" * width, *data)
            else:
                require(header["descr"] == "<f4", "RBC nonprimitive binary32 feature")
            values = struct.unpack("<" + "f" * width, data)
            missing = sum(math.isnan(v) for v in values)
            require(
                not any(math.isinf(v) for v in values)
                and missing < width
                and (
                    not missing
                    or (
                        g.get("allow_partial_missing") is True
                        and missing / width <= g.get("max_missing_fraction", 0)
                    )
                ),
                "RBC feature finite/missing-value policy changed",
            )
            if name == CONTEXT:
                require(
                    values == ((1.0, 0.0) if r["rbc_species"] == "human" else (0.0, 1.0)),
                    "RBC species array differs from source",
                )
            rows[r["record_id"]] = data
        parts[name], names[name] = rows, [name + ":" + str(v) for v in g["feature_names"]]
    return dict(features=f, parts=parts, names=names)


def designs(binding, route, source, api):
    f, parts = binding["features"], binding["parts"]
    require(route in f["routes"], "RBC selected route absent from frozen registry")
    chemical = [n for n in f["routes"][route] if n != CONTEXT]
    ids = [r["record_id"] for r in source]
    output = {}
    for name, members in (
        ("molecule_only", chemical),
        ("context_only", [CONTEXT]),
        ("molecule_plus_context", chemical + [CONTEXT]),
    ):
        rows = {i: b"".join(parts[k][i] for k in members) for i in ids}
        output[name] = dict(
            rows=rows,
            names=sum((binding["names"][k] for k in members), []),
            **api.control_precision_matrix_bytes(rows, ids),
        )
    return output


def verify_refit(run, result, frozen, source, api):
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
    contract, inputs = api.read(paths["model_contract"]), api.read(paths["input_contract"])
    attempts = result.get("refit_attempts", [])
    require(
        result.get("status") == result.get("refit_status") == "completed"
        and result.get("deployment_ready") is True
        and paths["model"] != paths["refit_model"]
        and attempts
        and attempts[-1]["status"] == "completed"
        and contract.get("refit_attempts") == attempts
        and contract.get("task") == inputs.get("task") == frozen
        and contract.get("common_record_ids") == [r["record_id"] for r in source]
        and contract.get("model_fit_scope") == "outer_train_only"
        and contract.get("optional_refit_fit_scope") == "all_common_after_evaluation"
        and contract.get("holdout_not_used_for_selection") is True
        and contract.get("target_unit") == "uM"
        and contract.get("target_transform") == "log10"
        and contract.get("selected_route") == result.get("selected_route")
        and contract.get("selected_model") == result.get("selected_model"),
        "RBC separate all38 final refit/outer evaluation contract changed",
    )
    config = inputs["config"]
    require(
        config.get("refit_full") is True
        and config.get("inner_folds") == 5
        and config.get("search_backend") == "optuna"
        and config.get("search_budget") == 128
        and len(config.get("models", [])) == 8
        and inputs.get("implementation_sha256")
        == api.sha(Path(run) / "scripts/peptide_modeling_train.py")
        == TRAIN_SHA,
        "RBC frozen trainer, Optuna8 models128 or fivefold/full-refit settings changed",
    )
    outer = api.read(paths["outer_split"])
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=frozen,
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=config,
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=outer["input_hash"],
    )
    require(
        object_sha(signature, compact=False)
        == inputs["input_hash"]
        == result["input_hash"]
        == contract["input_hash"],
        "RBC signed training inputs changed",
    )
    binding = verify_features(run, source, api)
    inputs_by_design = designs(binding, result["selected_route"], source, api)
    joint = inputs_by_design["molecule_plus_context"]
    recorded = inputs["feature_hashes"][result["selected_route"]]
    require(
        recorded["names"] == contract["feature_names"] == joint["names"]
        and recorded["matrix_sha256"] == joint["source_float32_sha256"]
        and recorded["mask_sha256"] == hashlib.sha256(bytes([1]) * 38).hexdigest(),
        "RBC selected exact joint matrix differs from signed feature bytes",
    )
    if result["selected_model"] == "linear":
        require(
            contract.get("linear_solver_policy")
            == inputs["linear_solver_policies"][result["selected_route"]],
            "RBC selected LSQR solver provenance changed",
        )
    api.report_repair_trials(run, result)
    return paths, contract, inputs, inputs_by_design


def verify_evaluation(report, source, inner, held, check, api):
    require(
        report.get("status") == "verified"
        and report.get("records") == 38
        and report.get("molecules") == 34
        and report.get("metrics_space") == "log10_uM"
        and report.get("registered_reference_prediction_only") is True
        and all(
            report.get(k) is False
            for k in (
                "horse_effect_validated",
                "independently_evaluated_refit",
                "primary_sample_graph_verified",
                "biological_state_contrast_supported",
            )
        ),
        "RBC evaluated annotation scope or independent refit/species claims changed",
    )
    by_id = {r["record_id"]: r for r in source}
    require(
        len(inner) + len(held) == len({r["record_id"] for r in inner + held}) == 38,
        "RBC evaluation does not cover every source record once",
    )
    api.source_membership(inner + held, source, require_all=True)
    folds = {f["fold"]: f for f in check["folds"]}
    fold_by_id = {i: f["fold"] for f in check["folds"] for i in f["validation_record_ids"]}
    roles, unsupported, original_roles = report.get("evaluation_by_role", {}), [], {}
    require(set(roles) == {"inner_oof", "outer_holdout"}, "RBC evaluation roles pooled or missing")
    copied_fields = (
        "rbc_species",
        "identity_tier",
        "source_stratum",
        "source_original_endpoint",
        "estimate_method",
        "extrapolation_status",
        "exposure_evidence_status",
        "registered_reference_id",
        "registered_disulfide_maps_verified",
        "primary_sample_disulfide_assignment_verified",
        "primary_sample_graph_verified",
        "sample_redox_state",
    )
    for role, rows, expected in (
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", held, check["holdout_record_ids"]),
    ):
        item = roles[role]
        require(
            len(rows) == len(expected)
            and {r["record_id"] for r in rows} == set(expected)
            and item.get("score_role")
            == ("model_selection" if role == "inner_oof" else "outer_train_model_evaluation")
            and item.get("score_applies_to_all38_refit") is False,
            "RBC frozen evaluation membership or score role changed",
        )
        actual = {r["record_id"]: r for r in rows}
        saved = item.get("predictions", [])
        require(
            len(saved) == len(rows)
            and {r["record_id"] for r in saved} == set(expected)
            and len({r["record_id"] for r in saved}) == len(saved),
            "RBC reported predictions omitted/repeated source rows",
        )
        missing = []
        for r in saved:
            rid, fold = r["record_id"], int(r["fold"])
            old, raw = by_id[rid], actual[rid]
            require(
                fold == (-1 if role == "outer_holdout" else fold_by_id[rid]),
                "RBC reported fold differs from frozen node plan",
            )
            train = (
                check["train_record_ids"]
                if role == "outer_holdout"
                else folds[fold]["train_record_ids"]
            )
            species = sorted({by_id[i]["rbc_species"] for i in train})
            unsupported_horse = old["rbc_species"] == "horse" and "horse" not in species
            require(
                old["split_group"] not in {by_id[i]["split_group"] for i in train}
                and r.get("evaluation_role") == role
                and r["split_group"] == old["split_group"]
                and float(r["target"]) == float(raw["target"]) == float(old["target"])
                and float(r["prediction"]) == float(raw["prediction"])
                and all(math.isfinite(float(r[k])) for k in ("target", "prediction"))
                and all(r.get(k) == old[k] for k in copied_fields)
                and r.get("species_seen_in_training") == species
                and r.get("species_context_supported") is (old["rbc_species"] in species)
                and r.get("unsupported_horse_context") is unsupported_horse
                and r.get("biological_state_contrast_supported") is False,
                "RBC saved numeric prediction/source condition/species support changed",
            )
            if unsupported_horse:
                missing.append(rid)
            if rid in NEW_IDS:
                original_roles[rid] = role
        require(
            item.get("unsupported_horse_record_ids") == missing,
            "RBC unsupported horse observations hidden",
        )
        api.check_mhc10_native_weighted(item, saved)
        strata = item.get("strata", {})
        require(
            set(strata) == {"rbc_species", "identity_tier", "source_stratum"},
            "RBC species/reference/source diagnostic strata missing",
        )
        for field in strata:
            require(set(strata[field]) == {r[field] for r in source}, "RBC stratum labels changed")
            for label, value in strata[field].items():
                api.check_mhc10_native_weighted(value, [r for r in saved if r[field] == label])
        unsupported += missing
    require(
        unsupported == [HORSE]
        and report.get("unsupported_horse_record_ids") == unsupported
        and report.get("original_observation_evaluation_roles") == original_roles,
        "RBC single-group horse or four source observation roles changed",
    )


def selected_consistency(report, source, inner, held, design, result, contract, api):
    by_id = {r["record_id"]: r for r in source}
    reported = report.get("selected_prediction_consistency", {})
    require(
        set(reported) == {"inner_oof", "outer_holdout"}, "RBC selected precision checks missing"
    )
    for role, values in (("inner_oof", inner), ("outer_holdout", held)):
        groups = defaultdict(list)
        for row in values:
            old = by_id[row["record_id"]]
            groups[(old["canonical_identity_group"], old["rbc_species"], int(row["fold"]))].append(
                row
            )
        expected = []
        for (graph, species, fold), rows in groups.items():
            ids = [r["record_id"] for r in rows]
            require(
                len({design["rows"][i] for i in ids}) == 1,
                "RBC same graph/species has distinct input bytes",
            )
            p = [float(r["prediction"]) for r in rows]
            close = all(abs(v - p[0]) <= 1e-8 + 1e-7 * abs(p[0]) for v in p)
            ulps = None
            if not close:
                policy = contract.get("linear_solver_policy") or {}
                require(
                    result["selected_model"] == "linear"
                    and policy.get("solver") == "lsqr"
                    and policy.get("selection_basis") == "raw_route_feature_count_only"
                    and policy.get("version") == "dimension_based_linear_policy_v2"
                    and policy.get("feature_count") == len(design["names"])
                    and policy.get("scaler_with_mean") is False,
                    "RBC selected numerical exception lacks signed LSQR provenance",
                )
                positions = [api.hc50_precision_binary32(v) for v in p]
                ulps = max(positions) - min(positions)
                require(
                    ulps <= 32
                    and all(abs(v) >= 2.0**-126 for v in p)
                    and len({math.copysign(1.0, v) for v in p}) == 1,
                    "RBC selected float32 spread exceeds bounded32ULP policy",
                )
            expected.append(
                dict(
                    canonical_identity_group=graph,
                    rbc_species=species,
                    fold=fold,
                    source_record_ids=ids,
                    full_input_row_sha256=hashlib.sha256(design["rows"][ids[0]]).hexdigest(),
                    full_input_width=len(design["names"]),
                    raw_predictions_unchanged=True,
                    byte_identical_full_inputs=True,
                    mode="legacy_float_tolerance" if close else "verified_linear_binary32_32ulp",
                    observed_absolute_spread=max(p) - min(p),
                    observed_binary32_ulp_distance=ulps,
                    max_allowed_binary32_ulps=32,
                    engineering_budget_not_mathematical_BLAS_bound=True,
                )
            )
        require(reported[role] == expected, "RBC selected same-input consistency evidence changed")


def verify_reporting(entry, journal, task, api):
    proof, frozen, source, checks = verify_source(entry, task, api)
    expected_owner = entry["owned_process_identity"]
    require(
        api.identity_state(expected_owner, api.process_identity(expected_owner["pid"]))
        != "same_live_process",
        "RBC reporting owner has not released",
    )
    receipt = journal.get("molecular_reporting", {})
    require(
        journal.get("status") == "completed"
        and receipt.get("status") == "verified"
        and receipt.get("implementation_sha256") == TOOL_SHA
        and receipt.get("implementation_dependencies_sha256") == DEPENDENCIES
        and receipt.get("selected_model_fits_during_reporting") == 0
        and receipt.get("fixed_diagnostic_fits") == 54,
        "RBC mandatory source/diagnostic report receipt missing or changed",
    )
    path = api.contained(entry["output_root"], receipt["path"])
    require(
        path.name == "rbc50_registered_molecular_diagnostics.json"
        and api.sha(path) == receipt.get("sha256"),
        "RBC report path/SHA changed",
    )
    payload = api.read(path)
    require(
        payload.get("schema") == REPORT
        and payload.get("status") == "verified"
        and payload.get("task_id") == TASK
        and payload.get("quantity_id") == QUANTITY
        and payload.get("required_seeds") == list(SEEDS)
        and payload.get("original_endpoint_coverage") == proof["original_endpoint_coverage"]
        and payload.get("warning") == proof["warning"]
        and payload.get("registered_disulfide_maps_verified") is True
        and payload.get("registered_disulfide_map_record_ids") == list(NEW_IDS)
        and payload.get("sample_redox_state") == "not_independently_verified"
        and payload.get("fixed_diagnostic_precision") == "float64"
        and payload.get("fixed_diagnostic_fits") == 54
        and all(
            payload.get(k) is False
            for k in (
                "original_human_only_contract_changed",
                "primary_sample_graph_verified",
                "primary_sample_disulfide_assignment_verified",
                "biological_state_contrast_supported",
                "horse_effect_validated",
                "all38_fit_method_uniform",
                "selected_model_selection_repeated",
            )
        ),
        "RBC report promoted reference/sample/refit/species/measurement scope",
    )
    rounds, reports = journal.get("rounds", []), payload.get("reports", [])
    require(
        len(rounds) == len(reports) == 3
        and {r["seed"] for r in rounds} == {r["seed"] for r in reports} == set(SEEDS),
        "RBC mandatory three seed reports missing/repeated",
    )
    round_by_seed = {r["seed"]: r for r in rounds}
    for report in reports:
        round_entry = round_by_seed[report["seed"]]
        require(
            round_entry.get("status") == "completed"
            and round_entry.get("audit", {}).get("method_model_coverage_met") is True
            and not round_entry["audit"].get("missing_completed_task_ids"),
            "RBC seed task/method/model coverage incomplete",
        )
        run = api.contained(entry["output_root"], round_entry["run_dir"])
        verification_path = run / "verification.json"
        verification = api.read(verification_path)
        require(
            api.sha(verification_path) == round_entry["audit"]["verification_sha256"]
            and verification["status"] == "verified"
            and verification["source_task_manifest_sha256"] == api.sha(run / "task_manifest.json")
            and api.read(run / "task_manifest.json")["tasks"] == [frozen],
            "RBC saved verification/manifest changed",
        )
        _, run_source, _ = _source_at(run, frozen, api)
        require(run_source == source, "RBC seed source changed")
        result_path = run / "results/full" / (TASK + "_common.json")
        result = api.read(result_path)
        require(
            result.get("actual_inner_folds") == result.get("requested_inner_folds") == 5
            and result.get("search", {}).get("backend") == "optuna",
            "RBC actual fivefold/Optuna missing",
        )
        paths, contract, inputs, d = verify_refit(run, result, frozen, source, api)
        check = next(c for c in checks["seeds"] if c["seed"] == report["seed"])
        outer = api.read(paths["outer_split"])
        require(
            outer.get("seed") == inputs["config"]["seed"] == report["seed"]
            and outer.get("train_records") == check["train_record_ids"]
            and outer.get("test_records") == check["holdout_record_ids"]
            and outer.get("record_groups") == {r["record_id"]: r["split_group"] for r in source},
            "RBC selected outer memberships differ from prefit node plan",
        )
        inner, held = api.actual_folds(paths["oof"]), api.table(paths["holdout"])
        verify_evaluation(report, source, inner, held, check, api)
        selected_consistency(
            report, source, inner, held, d["molecule_plus_context"], result, contract, api
        )
        bindings = dict(
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=TABLE_SHA,
            source_proof_sha256=PROOF_SHA,
            result_sha256=api.sha(result_path),
            oof_sha256=api.sha(paths["oof"]),
            holdout_sha256=api.sha(paths["holdout"]),
            model_sha256=api.sha(paths["model"]),
        )
        require(
            all(report.get(k) == v for k, v in bindings.items()),
            "RBC selected report bindings changed",
        )
        for role, key in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            api.check_metrics(
                result[key],
                {
                    k: report["evaluation_by_role"][role]["record_weighted"][k]
                    for k in ("mae", "rmse")
                },
            )
        controls = report.get("fixed_ridge_controls", {})
        require(set(controls) == set(d), "RBC fixed diagnostic design missing")
        for control in controls.values():
            require(
                control.get("model") == "fixed_ridge_alpha1_float64_not_Optuna_selected"
                and control.get("same_frozen_splits") is True
                and control.get("model_selection_repeated") is False,
                "RBC diagnostic control was selected/reparameterized",
            )
            verify_evaluation(
                control,
                source,
                control["evaluation_by_role"]["inner_oof"]["predictions"],
                control["evaluation_by_role"]["outer_holdout"]["predictions"],
                check,
                api,
            )
        precision = report.get("fixed_ridge_control_precision", {})
        require(
            precision.get("dataset_kind") == "rbc50_registered"
            and precision.get("source_table_sha256") == TABLE_SHA
            and precision.get("result_sha256") == api.sha(result_path)
            and precision.get("old_control_reference")
            == {"status": "new_task_float64_controls_from_outset_no_old_control_artifact"}
            and precision.get("ast_provenance")
            == api.control_float64_ast_proof(run / "scripts" / MODULE, "native"),
            "RBC float64 control dataset/source/result/AST binding changed",
        )
        # The shared verifier's 'native' branch is the generic five inner + one outer
        # fit wire. Validate the original RBC namespace first; normalize only the
        # local copy passed to that structural verifier, never the saved evidence.
        wire = dict(precision, dataset_kind="native_mhc10")
        api.verify_control_float64_seed(
            wire,
            {"designs": {k: v["rows"] for k, v in d.items()}},
            source,
            check,
            controls,
            "native",
        )
        refit = report.get("separate_all_common_refit", {})
        require(
            refit.get("status") == "verified"
            and refit.get("records") == 38
            and refit.get("molecular_groups") == 34
            and refit.get("training_record_ids") == contract["common_record_ids"]
            and refit.get("new_original_record_ids") == list(NEW_IDS)
            and refit.get("all_four_original_observations_in_training") is True
            and api.contained(run, refit["model_path"]) == paths["refit_model"]
            and refit.get("model_sha256") == api.sha(paths["refit_model"])
            and refit.get("model_contract_sha256") == api.sha(paths["model_contract"])
            and refit.get("fit_scope") == "all_common_after_evaluation"
            and refit.get("independently_evaluated_on_these38") is False,
            "RBC source observation coverage lacks distinct all38 training refit",
        )
    durable = api.read(entry["config_path"])["durable_root"]
    require(
        api.sha(api.contained(durable, receipt["durable_path"]))
        == receipt.get("durable_sha256")
        == receipt["sha256"],
        "RBC durable report changed",
    )
    return dict(
        status="verified",
        path=str(path),
        sha256=receipt["sha256"],
        required_seed_reports=3,
        source_records_per_seed=38,
        molecular_groups_per_seed=34,
        all38_full_refit_verified=3,
        registered_reference_only=True,
        primary_sample_graph_verified=False,
        independent_nucleated_cytotoxicity_model=False,
        horse_effect_validated=False,
        fixed_diagnostic_precision="float64",
        fixed_diagnostic_fits=54,
    )


def coverage(entry, journal, task, state, api):
    proof, _, _, _ = verify_source(entry, task, api)
    if state != "trained_three_seeds":
        return [
            dict(
                d,
                canonical_task_id=TASK,
                state="registered_reference_observation_training_pending",
                required_seed_final_refit_training_verified=0,
            )
            for d in proof["original_endpoint_coverage"]
        ]
    verify_reporting(entry, journal, task, api)
    return [
        dict(
            d,
            canonical_task_id=TASK,
            state="registered_reference_observation_trained",
            coverage_kind="registered_reference_only_source_corrected_observation_participation",
            source_proof_verified=True,
            required_seed_final_refit_training_verified=3,
            independent_raw_endpoint_predictor=False,
        )
        for d in proof["original_endpoint_coverage"]
    ]
