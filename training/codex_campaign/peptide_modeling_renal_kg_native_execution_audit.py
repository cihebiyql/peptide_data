"""Pure stdlib renal annotation source/report adapter; never fit or unpickle.

This supplements, rather than replaces, the campaign auditor's owner, Optuna,
model persistence, immutable-template and durable-publication checks. Call
``renal_verify_source`` before learning and ``renal_verify_seed`` after the
standard internal-CV check. Neither function declares a campaign complete.
"""

import ast
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

RENAL_TASK = "TASK_RENAL_KG_ANNOTATION_b77ceacb6b28f7f632f34c97"
RENAL_SCHEMA = "renal_kg_native_reported_annotation_v2"
RENAL_PROOF_SHA = "4e51728d86e60a7f325bbd02ef39bbd47170dc0f537028a049d50f8783ea0e7a"
RENAL_PRODUCER_SHA = "bfb175601178c99835c1f680b70f357d139312bf15d9811c8b2a095755429a56"
RENAL_HELPER_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
RENAL_TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
RENAL_BODY = "source_reported_body_weight_annotation"
RENAL_UNKNOWN = "reported_kg_basis_unspecified"
RENAL_CONTEXT = "renal_kg_annotation_context"
RENAL_CONTEXT_NAMES = [
    "source_reported_body_weight_annotation",
    "reported_kg_basis_unspecified_mask",
    "reported_in_vivo_organism_human",
    "reported_in_vivo_organism_rat",
]
RENAL_MODELS = {
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
}
RENAL_NATIVE = {
    "5162164": "0.064",
    "2672768": "1.03",
    "2671534": "0.09",
    "5162163": "0.04533",
    "5162158": "0.1558",
    "5118245": "3.78",
    "2996378": "1",
    "2671530": "0.08",
    "5162162": "0.06083",
    "2990610": "0.1",
    "5162161": "0.098",
    "5162157": "0.07117",
    "5162156": "0.12",
    "5162160": "0.1343",
    "5162159": "0.1395",
}
RENAL_UNVERIFIED = {"2672768", "2996378", "2671530", "2990610", "2671534"}
RENAL_NAB739 = "RENAL_KG_NATIVE_SOURCE:PMC2916294:T5:NAB739"
RENAL_RECOVERY_SCHEMA = "renal_native_report_only_recovery_v1"
RENAL_RECOVERY_IMPL_SHA = "f79b58e605d3db72fc397c4ff65d0c3988a82df67c8ed16990c3478ccb3ea44b"
RENAL_RECOVERY_CFG_SHA = "15f8c14c93f9b4c2fc0bbaf026f23f3debf288a6b6b6413a7e666f1bbcd53a8d"
RENAL_RECOVERY_REVIEW_SHA = "d3078e5bdf1a1bac5b9e07d27a663c2208f920452a6495aa48a69a802df29a43"
RENAL_RECOVERY_V2_IMPL_SHA = "de75d4663e14a0e17f7dd06d4bd63826399cbd7a3fe688bb8459eaae6e367be4"
RENAL_RECOVERY_V2_CFG_SHA = "cacc75bfcfb52a6a91d6e25e4333668b4898c5c6159001d85db7446566e638ff"
RENAL_RECOVERY_V2_REVIEW_SHA = "2e72b9e6c610d65aea4ff918675824c0a9b800d7d17c3f21301b515a2fd1a169"
RENAL_RECOVERY_V2_RESULT_SHA = "b6ee2d8cd90f15b6c140938ea7804407e548febabd884480a4f7a5bdbfda17d6"
RENAL_RECOVERY_V2_TERMINAL_SHA = "c218bd87ae65e48306cd5c5058a32d4e7d5c251b405a9c09670bc9ec673bef78"
RENAL_REUSED42_PRODUCTS = {
    "renal_native_fixed_controls.json": (
        "171684a80d26538c2f167eff84a5743951986644dba8a1ab122e1a92290b455e"
    ),
    "reports/renal_native_diagnostics.json": (
        "3834e4f4735d879cd88d9a45e1f10439a28005b0beedd4705ef25e3180825127"
    ),
}


def renal_require(condition, message):
    if not condition:
        raise ValueError("Renal annotation audit: " + message)


def renal_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def renal_object_sha(value, *, compact=True):
    kwargs = {"separators": (",", ":")} if compact else {}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, **kwargs).encode("utf-8")
    ).hexdigest()


def renal_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def renal_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def renal_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    renal_require(path.is_relative_to(root), "artifact escapes its root")
    return path


def is_renal_native_task(task):
    return task.get("task_id") == RENAL_TASK


def renal_verify_splits(rows, plan):
    ids = [r["record_id"] for r in rows]
    by_id = {r["record_id"]: r for r in rows}
    renal_require(len(ids) == len(by_id) == 18, "18 unique observations required")
    renal_require(
        plan.get("independent_holdout") is False
        and plan.get("strict_scaffold_fivefold_possible") is False
        and plan.get("identity_groups") == 5
        and plan.get("known_scaffold_groups") == 4
        and plan.get("distinct_molecular_partitions") == 1,
        "identity LOGO is not independent holdout or five-scaffold validation",
    )
    renal_require([s["seed"] for s in plan["seeds"]] == [42, 2026, 3407], "three seed order")
    partitions = []
    for check in plan["seeds"]:
        renal_require(check["actual_folds"] == len(check["folds"]) == 5, "five actual folds")
        seen, partition = [], []
        for number, fold in enumerate(check["folds"]):
            a, b = fold["train_record_ids"], fold["validation_record_ids"]
            renal_require(fold["fold"] == number, "fold numbering")
            renal_require(
                len(set(a)) == len(a)
                and len(set(b)) == len(b)
                and not set(a) & set(b)
                and set(a) | set(b) == set(ids),
                "fold membership",
            )
            ag = sorted({by_id[r]["split_group"] for r in a})
            bg = sorted({by_id[r]["split_group"] for r in b})
            renal_require(
                ag == fold["train_groups"]
                and bg == fold["validation_groups"]
                and len(ag) == 4
                and len(bg) == 1
                and not set(ag) & set(bg),
                "FAMILY leakage",
            )
            ai = {by_id[r]["canonical_identity_group"] for r in a}
            bi = {by_id[r]["canonical_identity_group"] for r in b}
            renal_require(not ai & bi, "exact graph leakage")
            asc = sorted({by_id[r]["scaffold_group"] for r in a})
            bsc = sorted({by_id[r]["scaffold_group"] for r in b})
            renal_require(
                asc == fold["train_scaffolds"]
                and bsc == fold["validation_scaffolds"]
                and sorted(set(asc) & set(bsc)) == fold["shared_scaffolds_across_identity_fold"],
                "known shared NAB scaffold must be reported",
            )
            contexts = {by_id[r]["semantic_context_key"] for r in a}
            renal_require(
                [r for r in b if by_id[r]["semantic_context_key"] not in contexts]
                == fold["unsupported_joint_context_record_ids"],
                "context support changed",
            )
            renal_require(len({by_id[r]["target"] for r in a}) > 1, "constant training target")
            seen.extend(b)
            partition.append(tuple(sorted(b)))
        renal_require(Counter(seen) == Counter(ids), "OOF coverage exactly once")
        partitions.append(tuple(sorted(partition)))
    renal_require(len(set(partitions)) == 1, "one unique molecular partition")


def renal_verify_source(root, *, scripts_root=None):
    """Validate immutable v2 source, weak annotations and actual-node memberships."""
    root = Path(root)
    manifest = renal_read(root / "task_manifest.json")
    renal_require(
        manifest.get("schema") == RENAL_SCHEMA and len(manifest["tasks"]) == 1,
        "only the approved v2 task is supported",
    )
    task = manifest["tasks"][0]
    renal_require(is_renal_native_task(task), "unknown task")
    renal_require(
        task.get("endpoint") == "CL_renal_per_reported_kg_annotation"
        and task.get("unit") == "L/h/kg_reported_basis_annotation"
        and task.get("task_kind") == "regression"
        and task.get("target_transform") == "log10"
        and task.get("inverse_transform") == "pow10"
        and task.get("n_records") == 18
        and task.get("n_groups") == 5
        and task.get("requested_inner_folds") == 5
        and task.get("validation_protocol") == "internal_grouped_cv_only_no_independent_holdout",
        "native quantity/task metadata changed",
    )
    definition = task["scientific_definition"]
    renal_require(definition == task["task_definition"], "scientific/task definitions differ")
    required_definition = {
        "query_mode": "source_annotation_only",
        "annotation_only": True,
        "validated_physiology": False,
        "no_absolute_clearance_query": True,
        "source_grade_is_predictor": False,
        "readout_verification_is_predictor": False,
        "context_fields": ["reported_weight_basis", "source_supported_organism_annotation"],
        "context_group": RENAL_CONTEXT,
        "identity_groups": 5,
        "known_scaffold_groups": 4,
        "strict_scaffold_fivefold_possible": False,
        "independent_holdout": False,
        "mandatory_molecular_reporting": True,
        "pretraining_fold_state": "node23_actual_no_fit_frozen",
        "implementation_sha256": RENAL_PRODUCER_SHA,
        "implementation_dependency_sha256": RENAL_HELPER_SHA,
    }
    renal_require(
        all(definition.get(k) == v for k, v in required_definition.items()),
        "annotation limits and source/runtime pins changed",
    )
    descriptor = task["native_source_observation_coverage"]
    proof_path = renal_path(root, descriptor["proof_path"])
    renal_require(
        renal_sha(proof_path) == descriptor["proof_sha256"] == RENAL_PROOF_SHA,
        "unapproved source proof",
    )
    proof = renal_read(proof_path)
    renal_require(
        proof["implementation_sha256"] == RENAL_PRODUCER_SHA
        and proof["implementation_dependency_sha256"] == RENAL_HELPER_SHA,
        "producer/dependency pin",
    )
    for artifact in proof["pinned_artifacts"]:
        renal_require(
            renal_sha(renal_path(root, artifact["path"])) == artifact["sha256"],
            "pinned source artifact changed: " + artifact["path"],
        )
    if scripts_root is not None:
        for name, expected in [
            ("peptide_modeling_renal_kg_native_bridge.py", RENAL_PRODUCER_SHA),
            ("peptide_modeling_mhc10_native_bridge.py", RENAL_HELPER_SHA),
            ("peptide_modeling_train.py", RENAL_TRAIN_SHA),
        ]:
            renal_require(
                renal_sha(Path(scripts_root) / name) == expected, "implementation changed"
            )
    rows = renal_rows(renal_path(root, task["table_path"]))
    renal_require(
        {r["record_id"]: renal_object_sha(r) for r in rows} == proof["row_hashes"],
        "source rows changed",
    )
    renal_require(
        len(rows) == 18
        and len({r["split_group"] for r in rows}) == 5
        and len({r["canonical_identity_group"] for r in rows}) == 5
        and len({r["scaffold_group"] for r in rows}) == 4,
        "18/5/4 source counts",
    )
    curation = renal_read(root / "task_tables/native_curation_proof.json")
    observations = {o["record_id"]: o for o in curation["observations"]}
    renal_require(
        len(observations) == 18
        and len(curation["input_source_ledger"]) == 24
        and len(curation["excluded_from_main18"]) == 7
        and curation["absolute_renal_head_five_molecules_claimed"] is False,
        "preserved original ledger and excluded absolute/component records",
    )
    expected = {
        "NEW:07:" + str(40000000000000000 + int(a)): Decimal(v) * Decimal("0.06")
        for a, v in RENAL_NATIVE.items()
    }
    expected.update(
        {
            "RENAL_KG_NATIVE_CORRECTION:2573917": Decimal("0.0044"),
            "RENAL_KG_NATIVE_CORRECTION:2573918": Decimal("0.0095"),
            RENAL_NAB739: Decimal("0.0318"),
        }
    )
    unknown = {"NEW:07:" + str(40000000000000000 + int(a)) for a in RENAL_UNVERIFIED}
    renal_require(set(expected) == set(observations), "unexpected original lineage")
    for row in rows:
        rid = row["record_id"]
        observation = observations[rid]
        renal_require(
            Decimal(row["value"]) == expected[rid]
            and Decimal(observation["normalized_decimal"]) == expected[rid],
            "exact source decimal normalization",
        )
        renal_require(
            math.isclose(
                float(row["target"]), math.log10(float(expected[rid])), rel_tol=1e-14, abs_tol=1e-14
            ),
            "log10 source target",
        )
        context = json.loads(row["semantic_context_json"])
        renal_require(context == observation["context"], "annotation context alignment")
        basis = RENAL_UNKNOWN if rid in unknown | {RENAL_NAB739} else RENAL_BODY
        renal_require(
            row["reported_weight_basis"] == context["reported_weight_basis"] == basis,
            "6 unspecified bases are distinct from 5 readout unknowns",
        )
        evidence = observation["basis_evidence"]
        renal_require(
            evidence["actual_total_bodyweight_denominator_claimed"] is False,
            "weak annotation must not become a TBW claim",
        )
        if basis == RENAL_BODY:
            renal_require(
                bool(
                    evidence["source_file"]
                    and evidence["source_anchor"]
                    and evidence["explicit_weight_text"]
                ),
                "body-weight text required",
            )
        renal_require(
            observation["original_five_unknown_readout_and_conditions"] == (rid in unknown),
            "original five readout limitations",
        )
        renal_require(
            observation["SD_is_provenance_only_not_an_extra_observation"] is True, "SD is metadata"
        )
        renal_require(
            row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest(),
            "full canonical graph hash",
        )
        renal_require(
            row["sequence_eligible"] == "false" and row["structure_eligible"] == "true",
            "no lossy sequence projection",
        )
    renal_require(
        sum(
            o["basis_evidence"]["exact_TBW_denominator_primary_verified"]
            for o in observations.values()
        )
        == 2,
        "two TBW provenance-only facts",
    )
    nab = next(r for r in rows if r["record_id"] == RENAL_NAB739)
    provenance = json.loads(nab["reported_context_provenance_json"])
    renal_require(
        provenance["n"] is None
        and provenance["formulation"] is None
        and provenance["dose_unit"] == "mg/kg_reported_basis",
        "NAB739 unsupported n/formulation or dose denominator",
    )
    plan = renal_read(root / "task_tables/predeclared_splits.json")
    renal_verify_splits(rows, plan)
    node = renal_read(renal_path(root, proof["nodefold_evidence"]["path"]))
    renal_require(
        node["hostname"] == "node23"
        and node["python_executable"] == "/usr/bin/python3"
        and node["dont_write_bytecode"] == 1
        and node["actual_checks"] == plan,
        "actual-node plan binding",
    )
    renal_require(
        set(node["guard_counters"])
        == {
            "estimator_fit_attempts",
            "search_attempts",
            "model_unpickle_attempts",
            "chemistry_import_attempts",
            "production_fits_launched",
        }
        and all(type(v) is int and v == 0 for v in node["guard_counters"].values()),
        "pretraining probe must perform zero fits/search/unpickle/chemistry",
    )
    renal_require(
        node["producer_sha256"] == RENAL_PRODUCER_SHA
        and node["training_source_sha256"] == RENAL_TRAIN_SHA
        and node["payload_source_review"]["frozen_row_hashes"] == proof["row_hashes"],
        "node source/runtime binding",
    )
    return {
        "task": task,
        "rows": rows,
        "proof": proof,
        "plan": plan,
        "node": node,
        "status": "source_approved_annotation_only_not_training_completion",
    }


def renal_npy(path):
    """Read primitive little-endian, C-order arrays without numpy/pickle."""
    data = Path(path).read_bytes()
    renal_require(data[:6] == b"\x93NUMPY", "invalid NPY")
    major = data[6]
    renal_require(major in (1, 2, 3), "NPY version")
    size = 2 if major == 1 else 4
    header_size = int.from_bytes(data[8 : 8 + size], "little")
    start = 8 + size
    header = ast.literal_eval(data[start : start + header_size].decode("utf-8"))
    renal_require(header["fortran_order"] is False, "C-order NPY required")
    code = {"<f4": "f", "<f8": "d", "|b1": "?", "|u1": "B"}.get(header["descr"])
    renal_require(code is not None, "primitive float/bool NPY only")
    shape = header["shape"]
    renal_require(
        isinstance(shape, tuple)
        and 1 <= len(shape) <= 2
        and all(type(v) is int and v > 0 for v in shape),
        "NPY shape",
    )
    body = data[start + header_size :]
    count = math.prod(shape)
    renal_require(len(body) == count * struct.calcsize(code), "NPY body length")
    values = struct.unpack("<" + str(count) + code, body)
    return shape, values


def renal_array_sha(matrix, dtype="d"):
    flat = (
        [v for row in matrix for v in row]
        if matrix and isinstance(matrix[0], (list, tuple))
        else matrix
    )
    return hashlib.sha256(struct.pack("<" + str(len(flat)) + dtype, *flat)).hexdigest()


def renal_design(root, source, route):
    """Reconstruct exact float32 loaded route and compare to pinned node no-fit bytes."""
    root = Path(root)
    manifest = renal_read(root / "features/manifest.json")
    renal_require(
        manifest["input_fingerprint"]
        == "363a612a8d46e51a391adf3edffefbb3398a9ddd4e6b6e5777372d4911e40e43",
        "frozen feature fingerprint changed",
    )
    # Cache paths are intentionally rebasable; require the existing in-root basename.
    basename = str(manifest["cache_dir"]).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    cache = renal_path(root / "features", basename)
    for name, expected in manifest["artifact_hashes"].items():
        renal_require(renal_sha(renal_path(cache, name)) == expected, "feature artifact changed")
    renal_require(
        manifest["representation_methods"]["independent_method_count"] == 10
        and manifest["representation_methods"]["chemical_groups"] == 13
        and manifest["representation_methods"]["chemical_dimensions"] == 15212,
        "ten chemical methods and 13 groups required",
    )
    groups = manifest["routes"][route]
    renal_require(
        groups[-1] == RENAL_CONTEXT and groups.count(RENAL_CONTEXT) == 1,
        "exactly one context suffix",
    )
    values, names = [[] for _ in source["rows"]], []
    for group_name in groups:
        group = manifest["groups"][group_name]
        shape, flat = renal_npy(renal_path(cache, group["matrix"]))
        valid_shape, valid = renal_npy(renal_path(cache, group["valid"]))
        renal_require(list(shape) == group["shape"] and valid_shape == (shape[0],), "group shape")
        index = {r["key"]: int(r["row_idx"]) for r in renal_rows(renal_path(cache, group["index"]))}
        renal_require(len(index) == shape[0], "unique feature keys")
        for i, row in enumerate(source["rows"]):
            key = (
                row["record_id"]
                if group["key_kind"] == "record_id"
                else row["canonical_identity_group"]
            )
            pos = index[key]
            renal_require(0 <= pos < shape[0] and valid[pos], "all18 features eligible")
            part = flat[pos * shape[1] : (pos + 1) * shape[1]]
            # Loader always casts every group to float32 before concatenation.
            part = struct.unpack(
                "<" + str(len(part)) + "f", struct.pack("<" + str(len(part)) + "f", *part)
            )
            values[i].extend(part)
            if group_name == RENAL_CONTEXT:
                expected = [
                    float(row["reported_weight_basis"] == RENAL_BODY),
                    float(row["reported_weight_basis"] == RENAL_UNKNOWN),
                    float(row["source_supported_organism_annotation"] == "human"),
                    float(row["source_supported_organism_annotation"] == "rat"),
                ]
                renal_require(
                    list(part) == expected and group["feature_names"] == RENAL_CONTEXT_NAMES,
                    "semantic feature values/names",
                )
        names.extend(group_name + ":" + n for n in group["feature_names"])
    frozen = source["node"]["payload_source_review"]["routes"][route]
    renal_require(
        frozen["shape"] == [18, len(names)]
        and frozen["feature_names"] == names
        and frozen["matrix_sha256"] == renal_array_sha(values, "f"),
        "actual-node exact route bytes/names",
    )
    return values, names


def renal_metrics(rows, prediction):
    renal_require(
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
            "mae": math.fsum(abs(v) * z for v, z in zip(e, w, strict=True)) / math.fsum(w),
            "rmse": math.sqrt(
                math.fsum(v * v * z for v, z in zip(e, w, strict=True)) / math.fsum(w)
            ),
        },
    }


def renal_numeric_equal(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(renal_numeric_equal(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, float):
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11)
        )
    return type(actual) is type(expected) and actual == expected


def renal_consistency(rows, predictions, matrix, provenance):
    """Recompute the narrow predeclared binary32 LSQR guard; do not alter OOF."""
    positions = {r["record_id"]: i for i, r in enumerate(rows)}
    buckets = defaultdict(list)
    for prediction in predictions:
        row = rows[positions[prediction["record_id"]]]
        buckets[
            (row["canonical_identity_group"], row["semantic_context_key"], int(prediction["fold"]))
        ].append(prediction)
    output = []
    for bucket in buckets.values():
        ix = [positions[p["record_id"]] for p in bucket]
        renal_require(
            len({renal_array_sha(matrix[i], "f") for i in ix}) == 1,
            "same identity/context/fold requires exact same route bytes",
        )
        values = [float(p["prediction"]) for p in bucket]
        renal_require(all(math.isfinite(v) for v in values), "finite selected predictions")
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
            "full_matrix_sha256": renal_array_sha(matrix, "f"),
            "training_source_sha256": RENAL_TRAIN_SHA,
        }
        renal_require(
            all(provenance.get(k) == v for k, v in required.items()),
            "drift requires actual pinned float32 Ridge LSQR provenance",
        )
        bits = []
        for value in values:
            renal_require(
                abs(value) >= 2.0**-126
                and struct.unpack("<f", struct.pack("<f", value))[0] == value,
                "drift values must be exact normal binary32",
            )
            bits.append(struct.unpack("<I", struct.pack("<f", value))[0])
        renal_require(
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


def renal_verify_controls(rows, matrix, check, controls):
    """Verify all 18 actual traces/seed, numeric metrics and frozen input hashes."""
    ids = [r["record_id"] for r in rows]
    positions = {rid: i for i, rid in enumerate(ids)}
    designs = {
        "molecule_only": [r[:-4] for r in matrix],
        "context_only": [r[-4:] for r in matrix],
        "molecule_plus_context": matrix,
    }
    renal_require(set(controls) == set(designs), "three fixed controls")
    y = [float(r["target"]) for r in rows]
    for name, X in designs.items():
        control = controls[name]
        renal_require(
            control["model"] == "fixed_Ridge_alpha1_LSQR_not_searched"
            and control["dtype_from_initial_boundary"] == "float64"
            and control["source_encoded_values_rewritten"] is False
            and control["same_frozen_inner_fivefold"] is True
            and control["source_value_origin"] == "upcast_exact_existing_encoded_float32_values"
            and control["prediction_record_ids"] == ids
            and control["matrix_sha256"] == renal_array_sha(X)
            and control["control_fit_calls"] == 6
            and control["selected_model_fit_calls"] == control["search_calls"] == 0,
            "fixed control schema/input/dtype/count",
        )
        renal_require(
            renal_numeric_equal(control["metrics"], renal_metrics(rows, control["predictions"])),
            "fixed control OOF metrics",
        )
        renal_require(
            control["full_cohort_evaluation_role"]
            == "training_inputs_only_no_independent_performance_score",
            "full control fit is not independent performance",
        )
        traces = control["per_fold_fit_trace"]
        renal_require(len(traces) == 6, "five OOF plus all18 actual control traces")
        for i, trace in enumerate(traces):
            a = check["folds"][i]["train_record_ids"] if i < 5 else ids
            b = check["folds"][i]["validation_record_ids"] if i < 5 else ids
            pred = (
                [control["predictions"][positions[r]] for r in b]
                if i < 5
                else control["all_cohort_fit_predictions"]
            )
            renal_require(
                len(pred) == len(b) and all(math.isfinite(float(v)) for v in pred),
                "trace prediction length/finite",
            )
            required = {
                "fold": i if i < 5 else -1,
                "fit_role": "inner_training_fold" if i < 5 else "all_internal_cohort_control_fit",
                "train_record_ids": a,
                "predict_record_ids": b,
                "actual_estimator": "Ridge",
                "actual_parameters": {
                    "alpha": 1.0,
                    "solver": "lsqr",
                    "imputer_strategy": "median",
                    "imputer_add_indicator": True,
                    "imputer_keep_empty_features": True,
                    "scaler_with_mean": False,
                },
                "control_fit_calls": 1,
                "selected_model_fit_calls": 0,
                "search_calls": 0,
                "fit_X_sha256": renal_array_sha([X[positions[r]] for r in a]),
                "fit_y_sha256": renal_array_sha([y[positions[r]] for r in a]),
                "predict_X_sha256": renal_array_sha([X[positions[r]] for r in b]),
                "predictions_sha256": renal_array_sha(pred),
            }
            required.update(
                {
                    k: "float64"
                    for k in [
                        "actual_fit_X_dtype",
                        "actual_fit_y_dtype",
                        "actual_predict_X_dtype",
                        "actual_prediction_dtype",
                        "actual_coef_dtype",
                    ]
                }
            )
            renal_require(
                all(trace.get(k) == v for k, v in required.items()),
                "actual control membership/parameters/dtype/hash",
            )
            coef_sha = trace.get("actual_coef_sha256", "")
            renal_require(
                len(coef_sha) == 64
                and all(c in "0123456789abcdef" for c in coef_sha)
                and trace.get("actual_coef_shape")
                == [
                    len(X[0])
                    + sum(any(math.isnan(X[positions[r]][j]) for r in a) for j in range(len(X[0])))
                ],
                "actual coefficient hash/shape",
            )
            identical = defaultdict(list)
            for rid, p in zip(b, pred, strict=True):
                identical[renal_array_sha(X[positions[rid]])].append(float(p))
            renal_require(
                all(
                    all(math.isclose(p, ps[0], rel_tol=1e-7, abs_tol=1e-8) for p in ps)
                    for ps in identical.values()
                ),
                "float64 identical-input consistency",
            )
    return 18


def renal_cv_score(metrics):
    """Frozen37bd regression model selection maximizes negative MAE."""
    return -float(metrics["mae"])


def renal_verify_trials(run, result, settings, admission):
    """Reconcile all128 attempt receipts, candidate caches and read-only Optuna DB."""
    search = result["search"]
    renal_require(
        search.get("backend") == "optuna" and search.get("status") == "completed",
        "actual completed Optuna search required",
    )
    contract = search["contract"]
    namespace = renal_object_sha(contract)
    run = Path(run)
    directory = renal_path(run, search["search_dir"])
    renal_require(
        namespace == search["namespace"]
        and directory.name == namespace[:24]
        and contract["input_signature"] == result["input_hash"]
        and contract["backend"] == "optuna"
        and contract["seed"] == result["seed"]
        and contract["task_kind"] == "regression"
        and renal_read(directory / "contract.json") == contract,
        "Optuna namespace/seed/input contract",
    )
    renal_require(
        contract["implementation_sha256"] == renal_sha(run / "scripts/peptide_modeling_search.py"),
        "actual search implementation pin",
    )
    pairs = {(route, model) for route in settings["routes"] for model in settings["models"]}
    renal_require(
        set(map(tuple, contract["pairs"])) == pairs and len(contract["pairs"]) == len(pairs) == 88,
        "all11routes x8models must participate",
    )
    stored_result = renal_read(directory / "search_result.json")
    renal_require(
        {k: v for k, v in stored_result.items() if k != "trials"} == search,
        "published search result differs from selected result",
    )
    records = []
    candidate_dir = Path(result["artifacts"]["model"]).parent / "candidate_cv"
    expected_cache = set()
    for number, path in enumerate(sorted((directory / "trials").glob("*.json"))):
        wrapped = renal_read(path)
        value = wrapped["evaluation"]
        renal_require(
            wrapped["record_sha256"] == renal_object_sha(value)
            and value["search_trial"] == number
            and path.name == f"{number:06d}.json"
            and value["search_input_signature"] == namespace,
            "trial receipt content/number/input pin",
        )
        renal_require(
            (value["route"], value["model"]) in pairs
            and value["status"] in {"completed", "failed"},
            "terminal declared candidate",
        )
        if value["status"] == "completed":
            renal_require(
                value["actual_folds"] == value["requested_folds"] == 5
                and value["input_hash"] == result["input_hash"]
                and math.isfinite(value["score"])
                and math.isclose(
                    value["score"], renal_cv_score(value["metrics"]), rel_tol=1e-10, abs_tol=1e-11
                ),
                "successful candidate has real fivefold regression score",
            )
        else:
            renal_require(
                value.get("score") is None and bool(value.get("reason")),
                "failed candidate cannot gain a numeric fallback score",
            )
        key = renal_object_sha(
            [result["input_hash"], value["route"], value["model"], value["params"]], compact=False
        )
        cache_path = candidate_dir / (key + ".json")
        if cache_path.exists():
            cached = renal_read(cache_path)
            renal_require(
                all(value.get(k) == v for k, v in cached.items()), "candidate cache changed"
            )
            expected_cache.add(cache_path.name)
        else:
            renal_require(value["status"] == "failed", "successful candidate cache missing")
        records.append(value)
    renal_require(
        len(records) == 128 == settings["search_budget"]
        and search["budget"] == {"requested": 128, "used": 128, "remaining": 0, "pairs": 88}
        and stored_result["trials"] == records
        and {p.name for p in candidate_dir.glob("*.json")} == expected_cache
        and set(map(tuple, search["coverage_attempted"]))
        == {(r["route"], r["model"]) for r in records}
        == pairs,
        "complete128attempts/all88pairs/no unrecorded candidate cache",
    )
    renal_require(
        search["completed"] == sum(v["status"] == "completed" for v in records)
        and search["failed"] == sum(v["status"] == "failed" for v in records),
        "actual successful/failed counts",
    )
    database = renal_path(run, search["storage_path"])
    renal_require(
        database == directory / "study.sqlite3"
        and search["study_name"] == "peptide_" + namespace[:24],
        "Optuna study database path/name",
    )
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        study_names = connection.execute("SELECT study_name FROM studies").fetchall()
        stored = connection.execute(
            "SELECT t.number,t.state,a.value_json FROM trials t LEFT JOIN trial_user_attributes a "
            "ON a.trial_id=t.trial_id AND a.key='evaluation_result' ORDER BY t.number"
        ).fetchall()
        objectives = connection.execute(
            "SELECT t.number,v.value FROM trials t JOIN trial_values v ON v.trial_id=t.trial_id "
            "WHERE t.state='COMPLETE' ORDER BY t.number"
        ).fetchall()
    renal_require(
        study_names == [(search["study_name"],)]
        and len(stored) == len(records)
        and all(
            number == i
            and state == ("COMPLETE" if records[i]["status"] == "completed" else "FAIL")
            and raw is not None
            and json.loads(raw) == records[i]
            for i, (number, state, raw) in enumerate(stored)
        ),
        "SQLite evaluations/statuses differ from trial receipts",
    )
    renal_require(
        objectives
        == [(v["search_trial"], v["score"]) for v in records if v["status"] == "completed"],
        "SQLite objective values differ from actual candidate scores",
    )
    renal_require(
        admission.get("policy") == "internal_cv_rank_then_full_internal_cohort_fit_and_persistence"
        and admission.get("validation_labels") == "all_internal_cv_records_no_independent_holdout"
        and admission.get("holdout_used") is False,
        "internal final-fit admission policy",
    )

    def candidate_key(record):
        return renal_object_sha([record["route"], record["model"], record["params"]], compact=False)

    rejected = set()
    for record in admission["rejected_candidates"]:
        matches = [
            v
            for v in records
            if v["status"] == "completed"
            and all(v[k] == record[k] for k in ("route", "model", "params"))
            and v["score"] == record["inner_cv_score"]
        ]
        renal_require(
            matches
            and bool(record.get("reason"))
            and record.get("stage")
            in {
                "estimator_construction",
                "outer_train_fit",
                "training_probe_persistence",
                "development_oof_reconstruction",
            },
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
    renal_require(
        winners
        and key == admission["selected_candidate_key"]
        and winners[0]["score"] == max(r["score"] for r in admitted)
        and math.isclose(
            winners[0]["score"], result["internal_cv_selection_score"], rel_tol=1e-10, abs_tol=1e-11
        ),
        "selected model is not the highest admitted successful CV candidate",
    )
    return {"attempts": 128, "pairs": 88, "backend": "optuna", "sqlite_verified": True}


def renal_verify_seed(root, run, seed):
    """Renal supplement only: caller must also audit Optuna/persistence/owner/durable."""
    source = renal_verify_source(root)
    actual = renal_verify_source(run, scripts_root=Path(run) / "scripts")
    renal_require(
        actual["task"] == source["task"] and actual["rows"] == source["rows"],
        "runtime source changed",
    )
    run = Path(run)
    result_path = run / "results/internal_cv" / (RENAL_TASK + ".json")
    result = renal_read(result_path)
    renal_require(
        result["status"] == "completed"
        and result["task_id"] == RENAL_TASK
        and result["endpoint"] == source["task"]["endpoint"]
        and result["target_unit"] == source["task"]["unit"]
        and result["target_transform"] == "log10"
        and result["seed"] == seed
        and result["actual_inner_folds"] == result["requested_inner_folds"] == 5
        and result["protocol"] == "internal_grouped_cv_only_no_independent_holdout"
        and result["independent_holdout_available"] is False
        and result["n_internal_cv_rows"] == 18
        and result["n_internal_cv_groups"] == 5,
        "actual all18 internal fivefold result",
    )
    for name, receipt in result["artifact_hashes"].items():
        renal_require(
            renal_sha(renal_path(run, name)) == receipt["sha256"], "fitted artifact changed"
        )
    paths = {k: renal_path(run, v) for k, v in result["artifacts"].items()}
    renal_require(
        all(str(p) in result["artifact_hashes"] for p in paths.values()), "all artifacts hash-bound"
    )
    contract, inputs = renal_read(paths["model_contract"]), renal_read(paths["input_contract"])
    ids = [r["record_id"] for r in source["rows"]]
    renal_require(
        contract["fit_scope"] == "all_internal_cv_rows"
        and contract["protocol"] == result["protocol"]
        and contract["actual_inner_folds"] == contract["requested_inner_folds"] == 5
        and contract["independent_holdout_available"] is False
        and contract["target_unit"] == result["target_unit"]
        and contract["target_transform"] == result["target_transform"]
        and contract["record_ids"] == ids
        and contract["task"] == source["task"] == inputs["task"]
        and contract["input_hash"] == inputs["input_hash"] == result["input_hash"]
        and inputs["source_rows_hash"] == renal_object_sha(source["rows"], compact=False),
        "all18 selected final fit and input contract",
    )
    for key in ["selected_route", "selected_model", "selected_params"]:
        renal_require(contract[key] == result[key], "selected model contract")
    admission = renal_read(paths["candidate_fit_admission"])
    renal_require(
        contract["selection_fit_admission"] == admission, "model final-fit admission binding"
    )
    matrix, names = renal_design(run, source, result["selected_route"])
    selected = inputs["features"][result["selected_route"]]
    renal_require(
        selected["names"] == contract["feature_names"] == names
        and selected["matrix_sha256"] == renal_array_sha(matrix, "f")
        and selected["valid_sha256"] == hashlib.sha256(b"\1" * 18).hexdigest(),
        "actual selected full route bytes",
    )
    settings = inputs["config"]
    renal_require(
        set(settings["models"]) == RENAL_MODELS
        and settings["seed"] == seed
        and settings["search_backend"] == "optuna"
        and settings["search_budget"] == 128
        and settings["inner_folds"] == 5
        and settings["n_jobs"] == 1,
        "8 models, Optuna128 and frozen resource/seed policy",
    )
    features = renal_read(run / "features/manifest.json")
    renal_require(
        set(inputs["features"]) == set(features["routes"]) == set(settings["routes"]),
        "all10 methods plus fusion in actual training signature",
    )
    for route, record in inputs["features"].items():
        X, route_names = renal_design(run, source, route)
        renal_require(
            record
            == {
                "names": route_names,
                "matrix_sha256": renal_array_sha(X, "f"),
                "valid_sha256": hashlib.sha256(b"\1" * 18).hexdigest(),
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
    renal_require(
        result["input_hash"] == renal_object_sha(signature, compact=False)
        and contract["runtime"] == inputs["runtime"] == result["runtime"],
        "actual learning input signature",
    )
    renal_require(
        result["runtime"] == source["node"]["runtime"], "actual frozen node learning runtime"
    )
    search_evidence = renal_verify_trials(run, result, settings, admission)
    check = next(c for c in source["plan"]["seeds"] if c["seed"] == seed)
    oof = renal_rows(paths["internal_oof"])
    renal_require([r["record_id"] for r in oof] == ids, "OOF all18/order/once")
    folds = {rid: f["fold"] for f in check["folds"] for rid in f["validation_record_ids"]}
    for row, prediction in zip(source["rows"], oof, strict=True):
        renal_require(
            prediction["split_group"] == row["split_group"]
            and int(prediction["fold"]) == folds[row["record_id"]]
            and float(prediction["target"]) == float(row["target"]),
            "actual frozen OOF membership/target",
        )
    controls_path = run / "renal_native_fixed_controls.json"
    controls = renal_read(controls_path)
    renal_require(
        controls["source_proof_sha256"] == RENAL_PROOF_SHA
        and controls["seed"] == seed
        and controls["selected_result_sha256"] == renal_sha(result_path)
        and controls["frozen_check_sha256"] == renal_object_sha(check)
        and controls["selected_estimator_provenance"]["actual_model_artifact_sha256"]
        == renal_sha(paths["model"]),
        "control-to-source/model/result binding",
    )
    provenance = controls["selected_estimator_provenance"]
    renal_require(
        provenance.get("selected_model") == result["selected_model"]
        and provenance.get("input_dtype") == "float32"
        and provenance.get("feature_width") == len(names)
        and provenance.get("full_matrix_sha256") == renal_array_sha(matrix, "f")
        and provenance.get("training_source_sha256") == RENAL_TRAIN_SHA,
        "actual selected estimator input/source provenance",
    )
    fits = renal_verify_controls(source["rows"], matrix, check, controls["controls"])
    report_path = run / "reports/renal_native_diagnostics.json"
    report = renal_read(report_path)
    renal_require(
        report["status"] == "verified_internal_annotation_only"
        and report["seed"] == seed
        and report["controls_sha256"] == renal_sha(controls_path)
        and report["fixed_ridge_controls"] == controls["controls"]
        and report["result_input_hash"] == result["input_hash"]
        and report["selected_models_fitted_by_reporting"]
        == report["any_estimators_fitted_by_reporting"]
        == 0,
        "report-only controls provenance",
    )
    diagnostic = report["diagnostics"]
    expected = renal_metrics(source["rows"], [float(r["prediction"]) for r in oof])
    renal_require(
        all(
            renal_numeric_equal(
                result["internal_cv_selection_metrics"][metric], expected["record_weighted"][metric]
            )
            for metric in ("mae", "rmse")
        ),
        "selection result metrics disagree with actual OOF",
    )
    renal_require(
        all(renal_numeric_equal(diagnostic[k], v) for k, v in expected.items())
        and diagnostic["fold_context_support"] == check["folds"]
        and diagnostic["annotation_only"] is True
        and diagnostic["validated_physiology"] is False
        and diagnostic["independent_holdout"] is False
        and diagnostic["source_grade_used_only_for_reporting"] is True
        and diagnostic["raw_predictions_mutated"] is False
        and diagnostic["Daptomycin_record_dominance"] == {"records": 12, "denominator": 18},
        "actual OOF metrics/support/interpretation",
    )
    renal_require(
        diagnostic["prediction_consistency"]
        == renal_consistency(
            source["rows"], oof, matrix, controls["selected_estimator_provenance"]
        ),
        "actual selected prediction precision proof",
    )
    for field in [
        "reported_weight_basis",
        "source_supported_organism_annotation",
        "source_grade",
        "readout_verification_status",
    ]:
        labels = sorted({r[field] for r in source["rows"]})
        renal_require(set(diagnostic["strata"][field]) == set(labels), "all provenance strata")
        for label in labels:
            ix = [i for i, r in enumerate(source["rows"]) if r[field] == label]
            metrics = renal_metrics(
                [source["rows"][i] for i in ix], [float(oof[i]["prediction"]) for i in ix]
            )
            renal_require(
                renal_numeric_equal(diagnostic["strata"][field][label], metrics), "stratum metrics"
            )
    return {
        "status": "renal_source_and_numeric_reporting_verified_not_campaign_completion",
        "seed": seed,
        "control_fits": fits,
        "search": search_evidence,
        "selected_fits_by_auditor": 0,
        "models_unpickled_by_auditor": 0,
        "required_external_gates": [
            "same_host_model_persistence",
            "exclusive_owner_released",
            "durable_source_model_trials_reports_and_campaign_journal",
        ],
    }


def verify_source(entry, task, api=None):
    """Registry-compatible source gate, with no training/model imports."""
    renal_require(entry["protocol"] == "internal5_no_holdout", "registry protocol")
    renal_require(
        renal_sha(entry["manifest_path"]) == entry["manifest_sha256"], "registry source manifest"
    )
    root = Path(entry["manifest_path"]).parent
    source = renal_verify_source(root)
    renal_require(
        task["task_id"] == RENAL_TASK
        and task["table_relative"] == source["task"]["table_path"]
        and task["table_sha256"] == renal_sha(root / task["table_relative"])
        and task["records"] == 18
        and task["scientific_definition"] == source["task"]["scientific_definition"]
        and task["native_source_observation_coverage"]
        == source["task"]["native_source_observation_coverage"],
        "registry/native source descriptor mismatch",
    )
    return source


def renal_files(root):
    root = Path(root)
    renal_require(root.is_dir() and not root.is_symlink(), "regular inventory root required")
    result = {}
    for path in sorted(root.rglob("*")):
        renal_require(not path.is_symlink(), "inventory refuses symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = renal_sha(path)
    return result


def renal_closure(source, template):
    node, proof = source["node"], source["proof"]
    closure = {
        name.replace("\\", "/"): {
            "sha256": value,
            "pin_kind": "original_node_receipt_input_hash_before",
        }
        for name, value in node["input_hashes_before"].items()
        if name.replace("\\", "/").startswith("source_provenance/")
    }
    evidence = proof["nodefold_evidence"]
    closure[evidence["pre_node_proof_path"]] = {
        "sha256": evidence["pre_node_proof_sha256"],
        "pin_kind": "final_prefit_source_proof_baseline_hash",
    }
    renal_require(len(closure) == 14, "complete14file pre-fit source-provenance closure")
    for relative, item in closure.items():
        path = renal_path(template, relative)
        renal_require(renal_sha(path) == item["sha256"], "original provenance source changed")
        item.update(source=str(path), bytes=path.stat().st_size)
    return closure


def renal_owner_released(identity, api):
    renal_require(
        isinstance(identity, dict)
        and type(identity.get("pid")) is int
        and type(identity.get("start_ticks")) is int,
        "owner identity required",
    )
    state = api.identity_state(identity, api.process_identity(identity["pid"]))
    renal_require(
        state in {"exited", "pid_reused"}, "original/recovery owner is still live or unverified"
    )


def renal_no_live_owned_workers(entry, api):
    proc = Path(getattr(api, "renal_proc_root", "/proc"))
    renal_require(proc.is_dir(), "Linux process inventory required for final owner release")
    prefixes = [str(Path(entry[k])).replace("\\", "/") + "/" for k in ("template", "output_root")]
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        identity = api.process_identity(int(directory.name))
        if identity is None or identity.get("state") == "Z":
            continue
        renal_require(
            not any(
                arg.replace("\\", "/").startswith(prefix)
                for arg in identity.get("argv", [])
                for prefix in prefixes
            ),
            "live worker still owns renal input/run paths",
        )


def renal_verify_learning_archive(journal, before_files, output, durable):
    receipt = journal["archive"]
    path = renal_path(durable, receipt["path"])
    renal_require(
        renal_sha(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"],
        "original durable learning archive changed",
    )
    expected = {
        name: sha for name, sha in before_files["output"].items() if name.startswith("round_")
    }
    archived = {}
    prefix = Path(output).name + "/"
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        named = {m.name: m for m in members}
        renal_require(len(named) == len(members), "duplicate learning archive member")
        for member in members:
            renal_require(
                not member.issym()
                and (member.isfile() or member.isdir() or member.islnk())
                and not PurePosixPath(member.name).is_absolute()
                and ".." not in PurePosixPath(member.name.replace("\\", "/")).parts,
                "learning archive must not contain symlinks/devices/path escapes",
            )
            target, visited = member, set()
            while target.islnk():
                renal_require(
                    target.name not in visited
                    and target.linkname in named
                    and target.linkname.startswith(prefix)
                    and ".." not in PurePosixPath(target.linkname.replace("\\", "/")).parts,
                    "hardlink must resolve inside the same archive without cycles",
                )
                visited.add(target.name)
                target = named[target.linkname]
            if member.islnk():
                renal_require(target.isfile(), "hardlink target must be a regular archived file")
            if (member.isfile() or member.islnk()) and member.name.startswith(prefix):
                relative = member.name[len(prefix) :]
                if relative.startswith("round_"):
                    renal_require(relative not in archived, "duplicate learning archive member")
                    with archive.extractfile(target) as handle:
                        archived[relative] = hashlib.sha256(handle.read()).hexdigest()
    renal_require(
        archived == expected,
        "every original learned seed/source/model/trial byte must be durably archived",
    )


def verify_recovery(entry, descriptor, api):
    """Only the approved v2 can succeed; v1 remains immutable failed history."""
    cfg = renal_read(descriptor["configuration"]["path"])
    renal_require(
        cfg.get("schema") == "renal_native_report_only_recovery_v2",
        "v1 is historical failed recovery and cannot be adopted as success",
    )
    return verify_recovery_v2(entry, descriptor, api)


def renal_verify_v2_fit_provenance(result, terminal):
    """Separate durable54traces from36new fits and the explicitly inferred v1 waste."""
    counter = result["fit_guard_counters"]
    expected = {
        "fixed_control_fit_calls": 36,
        "fixed_control_fit_completed": 36,
        "fixed_control_fit_attempts": 36,
        "fixed_control_fit_failed": 0,
        "selected_model_fit_calls": 0,
        "search_calls": 0,
        "staging_calls": 0,
        "forbidden_calls": 0,
        "budget_exceeded_calls": 0,
    }
    inherited = counter.get("inherited_same_self_fit_frames")
    renal_require(type(inherited) is int and inherited >= 0, "inherited frames are diagnostic only")
    expected["inherited_same_self_fit_frames"] = inherited
    renal_require(
        counter == terminal["fit_guard_counters"] == expected,
        "exact36completed outermost same-self fits, no unauthorized calls",
    )
    claims = {
        "new_completed_control_fits": 36,
        "reused_control_seeds": [42],
        "newly_computed_control_seeds": [2026, 3407],
        "final_retained_control_traces": 54,
        "historical_v1_retained_control_traces": 18,
        "unpersisted_v1_completed_controls_inferred": 9,
        "total_completed_control_fits_across_v1_v2_inferred": 63,
        "all_original_fitted_search_contract_bytes_unchanged": True,
        "original_failure_journal_status_retained": "requirement_incomplete",
        "first_seed_control_and_report_bytes_preserved": True,
        "historical_failed_v1_all_bytes_preserved": True,
        "no_retraining": True,
    }
    renal_require(
        all(result.get(k) == v for k, v in claims.items()),
        "retained54traces and inferred9/63 must not become fabricated fit evidence",
    )
    history = result["v1_history"]
    renal_require(
        history["inference_is_not_a_recovered_perfit_trace"] is True
        and history["raw_fit_frame_count55_not_estimator_fit_count"] is True
        and history["completed_outer_fits_inferred_from_two_frame_chain"] == 27
        and history["retained_complete_control_traces"] == 18
        and history["unpersisted_completed_controls_inferred"] == 9,
        "historical frame count and transient inference must remain distinct "
        "from retained evidence",
    )


def verify_recovery_v2(entry, descriptor, api):
    """Retain failed v1 and seed42; validate only36 new successful fixed-control fits."""
    for key, expected in [
        ("configuration", RENAL_RECOVERY_V2_CFG_SHA),
        ("review", RENAL_RECOVERY_V2_REVIEW_SHA),
        ("implementation", RENAL_RECOVERY_V2_IMPL_SHA),
    ]:
        renal_require(
            descriptor[key]["sha256"] == renal_sha(descriptor[key]["path"]) == expected,
            "unapproved v2 recovery code/config/review",
        )
    cfg = renal_read(descriptor["configuration"]["path"])
    review = renal_read(descriptor["review"]["path"])
    schema = "renal_native_report_only_recovery_v2"
    renal_require(
        cfg["schema"] == review["schema"] == schema
        and review["configuration_sha256"] == RENAL_RECOVERY_V2_CFG_SHA
        and review["recovery_implementation_sha256"] == RENAL_RECOVERY_V2_IMPL_SHA,
        "v2 reviewed implementation/configuration provenance",
    )
    recovery, durable_recovery = Path(cfg["recovery_root"]), Path(cfg["recovery_durable_root"])
    sidecars = [
        "owner.json",
        "before_snapshot.json",
        "recovery_result.json",
        "terminal_receipt.json",
    ]
    renal_require(set(descriptor["sidecars"]) == set(sidecars), "v2 recovery sidecar closure")
    renal_require(
        descriptor["sidecars"]["recovery_result.json"] == RENAL_RECOVERY_V2_RESULT_SHA
        and descriptor["sidecars"]["terminal_receipt.json"] == RENAL_RECOVERY_V2_TERMINAL_SHA,
        "actual successful v2 transaction receipt pin",
    )
    for name in sidecars:
        renal_require(
            renal_sha(recovery / name)
            == descriptor["sidecars"][name]
            == renal_sha(durable_recovery / name),
            "v2 sidecar durable pin",
        )
    owner = renal_read(recovery / "owner.json")
    terminal = renal_read(recovery / "terminal_receipt.json")
    result = renal_read(recovery / "recovery_result.json")
    before = renal_read(recovery / "before_snapshot.json")
    renal_require(
        owner["schema"] == terminal["schema"] == result["schema"] == before["schema"] == schema
        and owner["reviewed_sha256"] == RENAL_RECOVERY_V2_REVIEW_SHA
        and owner["configuration_sha256"] == RENAL_RECOVERY_V2_CFG_SHA
        and owner["recovery_implementation_sha256"] == RENAL_RECOVERY_V2_IMPL_SHA
        and owner["historical_v1_implementation_sha256"] == RENAL_RECOVERY_IMPL_SHA
        and result["owner"] == owner
        and owner["pid"] == terminal["pid"] == owner["process_identity"]["pid"]
        and owner["token"] == terminal["token"]
        and terminal["failure"] is None
        and result["status"] == terminal["status"] == "verified_report_recovery_without_retraining",
        "v2 owner/token/terminal binding",
    )
    for key in [
        "roots",
        "before_files",
        "rounds",
        "v1_history",
        "source_provenance_closure",
        "missing_control_seeds",
        "reused_control_seeds",
        "original_failure",
    ]:
        renal_require(before[key] == review[key], "v2 pre-apply review/snapshot mismatch")
    roots = before["roots"]
    renal_require(
        roots["template"] == entry["template"]
        and roots["output"] == entry["output_root"]
        and roots["v1_recovery"] == cfg["v1_root"]
        and roots["v1_recovery_durable"] == cfg["v1_durable_root"]
        and set(roots) == {"template", "output", "durable", "v1_recovery", "v1_recovery_durable"},
        "v2 protected roots must preserve original and v1 evidence",
    )
    v1 = Path(cfg["v1_root"])
    v1_durable = Path(cfg["v1_durable_root"])
    v1_before, v1_owner, v1_terminal = [
        renal_read(v1 / n) for n in ["before_snapshot.json", "owner.json", "terminal_receipt.json"]
    ]
    history = before["v1_history"]
    for field, name in [
        ("v1_before_snapshot_sha256", "before_snapshot.json"),
        ("v1_owner_sha256", "owner.json"),
        ("v1_terminal_sha256", "terminal_receipt.json"),
    ]:
        renal_require(
            renal_sha(v1 / name) == cfg[field] == renal_sha(v1_durable / name),
            "historical failed v1 receipt changed",
        )
    renal_require(
        history["owner"] == v1_owner
        and history["terminal"] == v1_terminal
        and history["before_snapshot_sha256"] == cfg["v1_before_snapshot_sha256"]
        and history["owner_sha256"] == cfg["v1_owner_sha256"]
        and history["terminal_sha256"] == cfg["v1_terminal_sha256"]
        and v1_owner["reviewed_sha256"] == RENAL_RECOVERY_REVIEW_SHA
        and v1_owner["configuration_sha256"] == RENAL_RECOVERY_CFG_SHA
        and v1_owner["recovery_implementation_sha256"] == RENAL_RECOVERY_IMPL_SHA
        and v1_terminal["token"] == v1_owner["token"]
        and v1_terminal["status"] == "failed"
        and v1_terminal["failure"]["reason"] == "Control fit budget exceeded54"
        and not (v1 / "recovery_result.json").exists()
        and not (v1_durable / "recovery_result.json").exists(),
        "v1 must remain failed, never fabricated as successful54fits",
    )
    renal_require(
        v1_terminal["fit_guard_counters"]
        == {
            "fixed_control_fit_calls": 55,
            "selected_model_fit_calls": 0,
            "search_calls": 0,
            "staging_calls": 0,
            "forbidden_calls": 0,
        },
        "historical55frames are not55estimator fits",
    )
    required_history = {
        "owner_released": True,
        "restored_files_verified": 84,
        "historical_failure_preserved": True,
        "raw_fit_frame_count55_not_estimator_fit_count": True,
        "completed_outer_fits_inferred_from_two_frame_chain": 27,
        "retained_complete_control_traces": 18,
        "unpersisted_completed_controls_inferred": 9,
        "next_outer_fit_blocked_at_entry": True,
        "inference_is_not_a_recovered_perfit_trace": True,
    }
    renal_require(
        all(history.get(k) == v for k, v in required_history.items()),
        "v1 transient-fit inference limits",
    )
    renal_owner_released(entry["owned_process_identity"], api)
    renal_owner_released(v1_owner["process_identity"], api)
    renal_owner_released(owner["process_identity"], api)
    renal_no_live_owned_workers(entry, api)
    cfg_original = renal_read(entry["config_path"])
    renal_require(
        renal_sha(entry["config_path"]) == entry["config_sha256"]
        and cfg_original == before["original_followup_config"]
        and cfg_original["durable_root"] == roots["durable"]
        and cfg_original["template_run"] == roots["template"]
        and cfg_original["output_root"] == roots["output"]
        and cfg_original["required_seeds"] == [42, 2026, 3407]
        and cfg_original["additional_twenty_hour_loop"] is False,
        "original learning configuration must remain unchanged",
    )
    source = renal_verify_source(entry["template"])
    closure = renal_closure(source, entry["template"])
    renal_require(
        before["source_provenance_closure"] == v1_before["source_provenance_closure"] == closure,
        "v1/v2 original pre-fit provenance closure",
    )
    # Independently recover the historical review pin from its protected v1 namespace.
    reviews = [
        name
        for name, sha in before["before_files"]["v1_recovery"].items()
        if sha == RENAL_RECOVERY_REVIEW_SHA
    ]
    renal_require(len(reviews) == 1, "original immutable v1 review must remain present")
    original_review = renal_read(v1 / reviews[0])
    for key in (
        "roots",
        "before_files",
        "source_provenance_closure",
        "restore_closure",
        "rounds",
        "original_failure",
    ):
        renal_require(
            v1_before[key] == original_review[key], "original v1 pre-repair inventory provenance"
        )
    failure = v1_before["original_failure"]
    renal_require(
        renal_sha(entry["journal_path"])
        == failure["journal_sha256"]
        == renal_sha(v1 / "failed_journal.original_bytes")
        == renal_sha(v1_durable / "failed_journal.original_bytes")
        and renal_sha(failure["log"])
        == failure["log_sha256"]
        == renal_sha(v1 / "failed_controller_log.original_bytes")
        == renal_sha(v1_durable / "failed_controller_log.original_bytes"),
        "original failure journal/log must remain byte-identical",
    )
    journal = renal_read(entry["journal_path"])
    renal_require(
        journal["status"] == "requirement_incomplete"
        and journal["renal_annotation_reporting"]["status"] == "failed"
        and "source_provenance/NAB739_exact_primary_abstract.json"
        in journal["renal_annotation_reporting"]["reason"],
        "original missing-source failure scope",
    )
    renal_verify_learning_archive(
        journal, v1_before["before_files"], roots["output"], roots["durable"]
    )
    restored = []
    expected_before = {
        key: dict(v1_before["before_files"][key]) for key in ("template", "output", "durable")
    }
    for index, seed in enumerate([42, 2026, 3407]):
        run_name = f"round_{index:03d}_seed_{seed}"
        for kind in ("output", "durable"):
            for relative, pin in closure.items():
                key = run_name + "/" + relative
                renal_require(
                    key not in expected_before[kind], "historical absent-only restoration changed"
                )
                path = Path(roots[kind]) / key
                restored.append(
                    {
                        "destination": str(path),
                        "relative": relative,
                        "source": pin["source"],
                        "sha256": pin["sha256"],
                        "pin_kind": pin["pin_kind"],
                        "absent_before": True,
                        "phase": "post_training_provenance_file_completion",
                        "restored_sha256": pin["sha256"],
                    }
                )
                expected_before[kind][key] = pin["sha256"]
    renal_require(
        v1_terminal["restored"] == restored
        and len(restored) == 84
        and v1_before["restore_closure"]
        == [{k: v for k, v in r.items() if k != "restored_sha256"} for r in restored],
        "preserve the exact84historical absent-only restorations",
    )
    existing = before["rounds"][0]["existing_product_sha256"]
    expected_existing = {}
    for kind in ("output", "durable"):
        for relative, digest in RENAL_REUSED42_PRODUCTS.items():
            key = "round_000_seed_42/" + relative
            expected_before[kind][key] = digest
            expected_existing[str(Path(roots[kind]) / key)] = digest
    renal_require(existing == expected_existing, "reuse only exact saved42 controls/report bytes")
    for key, files in expected_before.items():
        renal_require(
            before["before_files"][key] == files,
            "v2 may begin only after84restores and saved42 products",
        )
    for index, (current, original) in enumerate(
        zip(before["rounds"], v1_before["rounds"], strict=True)
    ):
        expected = {
            **original,
            "recovery_action": "reuse_verified_existing_controls_and_report_without_writing"
            if index == 0
            else "compute_first_complete_saved_controls_then_report",
            "retained_control_traces": 18 if index == 0 else 0,
            "new_control_fit_budget": 0 if index == 0 else 18,
            "existing_product_sha256": expected_existing if index == 0 else {},
        }
        renal_require(
            current == expected, "v2 must skip42 fitting/writing and only compute missing2026/3407"
        )
    renal_require(
        before["reused_control_seeds"] == result["reused_control_seeds"] == [42]
        and before["missing_control_seeds"]
        == result["newly_computed_control_seeds"]
        == [2026, 3407]
        and before["planned_new_control_fit_calls"] == result["new_completed_control_fits"] == 36
        and before["current_control_fit_calls"] == 0
        and before["restore_closure"] == [],
        "no repeated restoration/selected training/extra seeds",
    )
    renal_verify_v2_fit_provenance(result, terminal)
    renal_require(
        result["v1_history"] == history and result["before_files"] == before["before_files"],
        "retained54traces and inferred9/63 must not become fabricated fit evidence",
    )
    products = set()
    for index, seed in enumerate([42, 2026, 3407]):
        for kind in ("output", "durable"):
            for relative in [
                "renal_native_fixed_controls.json",
                "reports/renal_native_diagnostics.json",
            ]:
                products.add(str(Path(roots[kind]) / f"round_{index:03d}_seed_{seed}" / relative))
    products.update(
        str(Path(roots[k]) / "renal_native_campaign_diagnostics.json")
        for k in ("output", "durable")
    )
    renal_require(
        set(result["generated_report_sha256"]) == products, "all14final products must be hash-bound"
    )
    for path, sha in result["generated_report_sha256"].items():
        renal_require(renal_sha(path) == sha, "final reporting product changed")
    for key, root in roots.items():
        current = renal_files(root)
        originals = before["before_files"][key]
        renal_require(
            current == result["after_files"][key]
            and all(current.get(k) == v for k, v in originals.items()),
            "old/v1/reused42 bytes changed",
        )
        renal_require(
            all(str(Path(root) / name) in products for name in set(current) - set(originals)),
            "unexpected file outside the exact missing-report publication closure",
        )
    return {
        "status": "verified",
        "version": 2,
        "new_control_fit_calls": 36,
        "retained_control_traces": 54,
        "retained_v1_control_traces": 18,
        "transient_v1_completed_controls_inferred": 9,
        "total_completed_controls_inferred": 63,
        "inference_is_not_recovered_perfit_trace": True,
        "selected_model_fit_calls": 0,
        "historical_source_restorations": 84,
        "original_failed_journal_retained": True,
        "historical_failed_v1_retained": True,
        "owner_released": True,
    }


def freeze_registry_entry(
    previous_entry, recovery_config_path, review_path, implementation_path, api
):
    """Pure post-exit metadata migration; never alters the old journal or artifacts."""
    cfg = renal_read(recovery_config_path)
    root = Path(cfg["recovery_root"])
    sidecars = [
        "owner.json",
        "before_snapshot.json",
        "recovery_result.json",
        "terminal_receipt.json",
    ]
    if cfg["schema"] == RENAL_RECOVERY_SCHEMA:
        sidecars += ["failed_journal.original_bytes", "failed_controller_log.original_bytes"]
    descriptor = {
        "configuration": {
            "path": str(recovery_config_path),
            "sha256": renal_sha(recovery_config_path),
        },
        "review": {"path": str(review_path), "sha256": renal_sha(review_path)},
        "implementation": {
            "path": str(implementation_path),
            "sha256": renal_sha(implementation_path),
        },
        "sidecars": {name: renal_sha(root / name) for name in sidecars},
    }
    migrated = json.loads(json.dumps(previous_entry))
    renal_require("renal_report_recovery" not in migrated, "recovery adoption already exists")
    migrated["renal_report_recovery"] = descriptor
    verify_recovery(migrated, descriptor, api)
    return migrated


def verify_reporting(entry, journal, task, api):
    source = verify_source(entry, task, api)
    renal_require(
        [r["seed"] for r in journal["rounds"]] == [42, 2026, 3407], "exact three seed prefix"
    )
    renal_owner_released(entry["owned_process_identity"], api)
    recovery = verify_recovery(entry, entry["renal_report_recovery"], api)
    cfg = renal_read(entry["config_path"])
    reports, seed_checks = [], []
    for index, entry_round in enumerate(journal["rounds"]):
        seed = entry_round["seed"]
        run = renal_path(entry["output_root"], entry_round["run_dir"])
        renal_require(
            run.name == f"round_{index:03d}_seed_{seed}"
            and entry_round["status"] == "completed"
            and entry_round["audit"]["status"] == "verified",
            "original completed seed scope",
        )
        seed_checks.append(renal_verify_seed(entry["template"], run, seed))
        reports.append(renal_read(run / "reports/renal_native_diagnostics.json"))
        result_path = run / "results/internal_cv" / (RENAL_TASK + ".json")
        result = renal_read(result_path)
        durable_run = Path(cfg["durable_root"]) / run.name
        for path in [
            result_path,
            *[renal_path(run, p) for p in result["artifacts"].values()],
            run / "renal_native_fixed_controls.json",
            run / "reports/renal_native_diagnostics.json",
        ]:
            renal_require(
                renal_sha(path) == renal_sha(renal_path(durable_run, path.relative_to(run))),
                "mandatory fitted artifact/report durable mirror",
            )
        for relative, item in renal_closure(source, entry["template"]).items():
            renal_require(
                renal_sha(run / relative) == renal_sha(durable_run / relative) == item["sha256"],
                "restored source closure durable pin",
            )
    path = Path(entry["output_root"]) / "renal_native_campaign_diagnostics.json"
    campaign = renal_read(path)
    renal_require(
        campaign["schema"] == RENAL_SCHEMA
        and campaign["status"] == "verified_internal_annotation_only"
        and campaign["reports"] == reports
        and campaign["additional_twenty_hour_loop"] is False
        and renal_sha(path) == renal_sha(Path(cfg["durable_root"]) / path.name),
        "all3first reports/campaign/durable/noextra20h",
    )
    return {
        "status": "verified",
        "required_seed_reports": 3,
        "source_records_per_seed": 18,
        "identity_groups": 5,
        "known_scaffold_groups": 4,
        "fixed_diagnostic_fits": 54,
        "retained_fixed_diagnostic_traces": 54,
        "all18_final_internal_fit_verified": 3,
        "source_annotation_only": True,
        "validated_physiology": False,
        "recovery": recovery,
        "seed_checks": seed_checks,
    }


def coverage(entry, journal, task, state, api):
    source = verify_source(entry, task, api)
    result = {
        "original_endpoint": "CL_renal_per_kg",
        "quantity_id": source["task"]["scientific_definition"]["canonical_quantity_id"],
        "task_id": RENAL_TASK,
        "canonical_task_id": RENAL_TASK,
        "source_proof_verified": True,
        "original_labelled_records": 15,
        "included_original_records": 15,
        "all_original_records_covered": True,
        "training_pool_records": 18,
        "training_pool_identity_groups": 5,
        "known_scaffold_groups": 4,
        "basis_unknown_records": 6,
        "readout_and_conditions_unverified_records": 5,
        "source_corrected_perkg_records": 2,
        "new_NAB739_records": 1,
        "source_annotation_only": True,
        "native_annotation_only": True,
        "validated_physiology": False,
        "absolute_renal_endpoint_coverage": False,
        "independent_holdout": False,
        "strict_scaffold_fivefold_possible": False,
        "required_seed_final_training_verified": 0,
        "state": "source_annotation_observation_training_pending",
    }
    if state == "trained_three_seeds":
        verify_reporting(entry, journal, task, api)
        result.update(
            state="source_annotation_observation_trained", required_seed_final_training_verified=3
        )
    return result
