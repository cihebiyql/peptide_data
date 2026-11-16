"""Read-only stdlib audit for the DRAMP native29 source-annotation classifier.

The learner, NumPy, sklearn, RDKit, pickle and joblib are never imported. Models
are opaque SHA-bound artifacts. Callbacks supplement the shared auditor's
ordinary learning/owner checks; a completed controller alone proves nothing.
"""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import math
import re
import sqlite3
import struct
import tarfile
import types
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

TASK = "TASK_DRAMP_NATIVE_BINARY_e4dc11dc410d800c40f828e6"
SCHEMA = "DRAMP_native_hemolytic_annotation_graph_only_v1"
REPORT_SCHEMA = "DRAMP_native_binary_graph_only_calibrated_diagnostics_v1"
QUANTITY = "DRAMP_source_native_hemolytic_presence_judgment_given_complete_reference_graph"
QUERY = "source_native_annotation_registered_reference_only"
MODULE = "peptide_modeling_dramp_hemolysis_native_binary.py"
PRODUCER_SHA = "b5f4323f34c5e784105a83c0355177f62a291de89a65c6495855163e810c7402"
TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
PROOF_SHA = "462aa44df7dd3d6bcd4d6d0b2930734b219aef990d588edd619a6dd93a2ffe9e"
TABLE_SHA = "5d1cb8186a9eb7bcae8d5d371928c73e2eb20962a30016895a6ac327bb9db28a"
PLAN_SHA = "af06aa8bd11a5ba54467e2df583fb156abc50af2cdc8d41c29a92bc6c8885a1c"
ADMISSION_SHA = "ce205a474834d69406abff84bd9e47d3490b7f1b329125f2e7060f88c6ecf74d"
ADMISSION_DELIVERY_SHA = "3766da4228112013b8b52f36ffcc1930de90fb886fc54620ec9fc0aba53c7768"
STRUCTURE_DELIVERY_SHA = "46690e5e478573e905e203b73bb90a2f1a273f727e828e800ad04dc7252d7d14"
NEGATIVE_SHA = "b8d162ae695a991a60160e3c633bcf1ea09ca35cc67e92783b14333e7082ceeb"
FEATURE_FINGERPRINT = "afd414273ff18d0810d5115ecb7490f82bbd3e9847053abc90259c883b1180f8"
POLICY_SHA = "514ac0fb173f38fd02175cf7ba0956835aad04e22b2bce10113454f3d7b22a0d"
FEATURE_ARTIFACT_MAP_SHA = "67e587fd2b7d1da9a432ec11012e6c26b445cc4bda8caeb187b16985d4133b7c"
FEATURE_GROUPS_SHA = "6dfcbfa9a2f7bef043b665c0468f359e1f471131ef5efc0a4712dce6161c4b73"
FEATURE_METHODS_SHA = "f12fdf36a54ac5c2a1a4647b06eb5cf83dfe988a0efc9bc5a5e9f8d8bf09f8fa"
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
CHEMICAL_GROUPS = (
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
POSITIVE_IDS = {
    "DRAMP00833",
    "DRAMP01219",
    "DRAMP02009",
    "DRAMP03186",
    "DRAMP03724",
    "DRAMP28985",
    "DRAMP29031",
}
EXPLICIT_RBC_POSITIVE_IDS = {"DRAMP01219", "DRAMP03724", "DRAMP28985", "DRAMP29031"}
DIAGNOSTIC = "learning_diagnostics/native_binary"
REPORT_FILE = "dramp_native_binary_molecular_diagnostics.json"
WRAPPER = "peptide_modeling_dramp_native_source_complete_runner.py"
WRAPPER_SHA = "94ee449aaa9b3b76cf3457e162fe33854e357e8eb2e22862fd860a0d2bd57625"
CLOSURE_SCHEMA = "DRAMP_native_binary_prefit_source_closure_v1"
PIN_SCHEMA = "DRAMP_native_binary_source_complete_runtime_pin_v1"
BASE_DELIVERY_SHA = "8e183998467d02a9db3ab01a9904e601669fa6234c5d595d81cdc28569c12ee2"
ORIGINAL_CONFIG_SHA = "edce724a9b121b9c7bd9fb49df119f797a67bceac9a55cb43312e5bae921172d"
TRAINING_CONFIG_SHA = "ed4590422a416640a4361b50166ac8d8f42efd0950454b5262fa6559315ca2ba"
POLICY_FILE_SHA = "95cdd5fc3c8f6eca579554973ee74b1affc3fbddf93198af7b90f9fc1988b681"
STOCK_OBJECTIVE = "AP/pr_auc_original37bd_unchanged_ROC_evaluation_only"


SCRIPT_PINS = {
    "scripts/build_hemolytik2_v13.py": (
        "5be78be5f9971ea8923a3a271ac2c87e6d9451736ac3371c1cc67c1a7c2e3e80"
    ),
    "scripts/peptide_modeling_add_methods.py": (
        "80d4ba3e0293a2b3784dc62bfa74142696d68939b2c82366862d8d8f16332852"
    ),
    "scripts/peptide_modeling_add_smiles_methods.py": (
        "6cddff94edf5a069e1654cd4ef56d3a5bb8758d45c662c06fc729bfa11986b3a"
    ),
    "scripts/peptide_modeling_all_tasks.py": (
        "d47f2608340e5d78ace67c4660d1b419eff0d9e7f7b7c2d36e02b5d60f418489"
    ),
    "scripts/peptide_modeling_apply_dramp_overlay.py": (
        "eaf6f62d27561ac6cf45201dccaf476d1bd29657afe1015fe96d6bfa8db8ad21"
    ),
    "scripts/peptide_modeling_campaign.py": (
        "ec06551f36a7c30e31292bce6965e8f772ae3de03011e394ae3b0a90861a3742"
    ),
    "scripts/peptide_modeling_campaign_report.py": (
        "a00e87bfc9697fd0580e6c89696bd51f8a7bdb304d21ba9df42d68968939220b"
    ),
    "scripts/peptide_modeling_clint_bridge.py": (
        "ad1a85fba2038bd3673427624a5637fe5eb458b9a7c8b3630e4dff2fd89efc7e"
    ),
    "scripts/peptide_modeling_clint_features.py": (
        "f36139fbadf790d5215ae7d7a8d315ef9d6c8d4a1705aaf06d26a486c8fb1e4a"
    ),
    "scripts/peptide_modeling_dbaasp_primary_bridge.py": (
        "da297c0c845497c137abea469cd50990818292f8aade1fa53cafb3ac5c56fa54"
    ),
    "scripts/peptide_modeling_dispatch.py": (
        "1f0d9a2f806ded7d6a6b818b3d47748e12262a9830ae93e675f2c9e58918667a"
    ),
    "scripts/peptide_modeling_dramp_binary_reference_features.py": (
        "cdfaf5390f6138a1fa00fa163ef10f5c96621d9be1e9560222fcf706444e637b"
    ),
    "scripts/peptide_modeling_dramp_bridge.py": (
        "b4f68e4467bc9f7ace58cb2c51c5fadfb0c2ccbda6eda7b7df868efd3ddaff8f"
    ),
    "scripts/peptide_modeling_dramp_cytotoxicity_bridge.py": (
        "3b148815eeaa9c4fc4648ab633f1861e316536fbcd5759c45dfd68f40c542d9a"
    ),
    "scripts/peptide_modeling_dramp_hemolysis_native_binary.py": (
        "b5f4323f34c5e784105a83c0355177f62a291de89a65c6495855163e810c7402"
    ),
    "scripts/peptide_modeling_dramp_mhc10_bridge.py": (
        "3ca3393ee989611c4f15832f6fcfd5124f196a6c85b23171f4738c73c7b77464"
    ),
    "scripts/peptide_modeling_expand_features.py": (
        "107e609b314e6b249df23c659103633d9221be7b7b15d4f4cd9c3dba95ba7ce8"
    ),
    "scripts/peptide_modeling_features.py": (
        "80f42130aeee59478954f8d0e57134253dde391e006493f43e8be858cc63d2ba"
    ),
    "scripts/peptide_modeling_followup_seeds.py": (
        "8027b1c4c9b02816fa40c1a1f6e0f7dc58983e784f498ff76ed9fddf0a7ddfa2"
    ),
    "scripts/peptide_modeling_hc10_secondary_bridge.py": (
        "c31f41cbe8f2eb3c77ccdd7a279318414c00c42f71434367589e6f6355693de2"
    ),
    "scripts/peptide_modeling_hc10_terminal_bridge.py": (
        "335edfa0a37dacbd2c45d921a39be8f33451d209c09978ad6d81ce532453675a"
    ),
    "scripts/peptide_modeling_hemolytik_bridge.py": (
        "70b79e94dbef60c840376cfb126ffe4ea5db37845ed791ac19fb7cc6935f334c"
    ),
    "scripts/peptide_modeling_pipeline.py": (
        "46c739c09c5be29bc152acef2a0bf31354cc7d89c2514ee1915314dda8f86e30"
    ),
    "scripts/peptide_modeling_pk_quality_bridge.py": (
        "4c6c14fa31121c83a4630b27c1c147a2fb2bb3dbec881027c5bdf6dad2d2cdd2"
    ),
    "scripts/peptide_modeling_qualitative_cyto_features.py": (
        "f1ae694643d7eea1e230bfd9cb5900404a781847fdb490d2b3197fbe330c6863"
    ),
    "scripts/peptide_modeling_search.py": (
        "419960dfa5f1ffcc5f093f848344eb250f7939daa3927fe891002df7957bf36d"
    ),
    "scripts/peptide_modeling_summarize.py": (
        "28d2c8486cee1364d4cc500f68a9cf39d113e626b32adaf59da26610ec4b5112"
    ),
    "scripts/peptide_modeling_supervise.py": (
        "011d1b8a75749fd82f6a651eb282a689239964f8031ee944f06a314e987326d4"
    ),
    "scripts/peptide_modeling_tasks.py": (
        "debb1e00f23969877ae5f7a09a7372f68a796aa96bd63f689e3be3e160449584"
    ),
    "scripts/peptide_modeling_train.py": (
        "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
    ),
}


def require(value, reason):
    if not value:
        raise ValueError("DRAMP native binary audit: " + reason)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def table(path):
    with Path(path).open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def path_at(root, name):
    root = Path(root).resolve()
    path = Path(name)
    path = path if path.is_absolute() else root / path
    require(
        not any(p.is_symlink() for p in [path, *path.parents]), "artifact symlink is not allowed"
    )
    path = path.resolve()
    require(path != root and path.is_relative_to(root), "artifact path escapes its owned root")
    require(
        not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)),
        "artifact symlink is not allowed",
    )
    return path


def object_sha(value, compact=True):
    kwargs = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        kwargs["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def close(actual, expected, reason="numeric metadata differs", *, rtol=1e-9, atol=1e-11):
    require(
        math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
        reason,
    )


def tree_equal(actual, expected, reason="saved numeric report differs"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), reason + ": keys")
        for key in expected:
            tree_equal(actual[key], expected[key], reason + "/" + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), reason + ": length")
        for a, b in zip(actual, expected, strict=True):
            tree_equal(a, b, reason)
    elif isinstance(expected, float):
        close(actual, expected, reason)
    else:
        require(
            actual == expected and (not isinstance(expected, bool) or type(actual) is bool), reason
        )


def signature(values, dtype="float64", shape=None):
    code = {"float64": "d", "float32": "f", "bool": "?"}[dtype]
    body = struct.pack("<" + str(len(values)) + code, *values)
    return dict(
        dtype=dtype, shape=shape or [len(values)], bytes_sha256=hashlib.sha256(body).hexdigest()
    )


def npy(path):
    """Parse only bounded C-order numeric NPY arrays; never accept object/pickle data."""
    raw = Path(path).read_bytes()
    require(
        len(raw) >= 10 and raw[:6] == b"\x93NUMPY" and raw[6] in {1, 2, 3},
        "invalid NPY magic/version",
    )
    size = 2 if raw[6] == 1 else 4
    header_size = int.from_bytes(raw[8 : 8 + size], "little")
    offset = 8 + size + header_size
    require(header_size <= 65536 and offset <= len(raw), "invalid NPY header extent")
    try:
        header = ast.literal_eval(
            raw[8 + size : offset].decode("utf8" if raw[6] == 3 else "latin1")
        )
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise ValueError("DRAMP native binary audit: invalid literal numeric NPY header") from exc
    require(
        isinstance(header, dict)
        and set(header) == {"descr", "fortran_order", "shape"}
        and header["fortran_order"] is False,
        "only numeric C-order NPY arrays are supported",
    )
    shape = header["shape"]
    require(
        isinstance(shape, tuple)
        and 1 <= len(shape) <= 2
        and all(type(v) is int and 0 < v <= 20000 for v in shape),
        "invalid NPY shape",
    )
    code = {"|u1": "B", "|b1": "?", "<f4": "f", "<f8": "d"}.get(header["descr"])
    require(code is not None, "NPY object, structured, or unsupported dtype rejected")
    body = raw[offset:]
    require(
        math.prod(shape) <= 1000000 and len(body) == math.prod(shape) * struct.calcsize(code),
        "NPY body extent differs from its declared numeric shape",
    )
    return header, body


def np_sum(values):
    """Match NumPy's short contiguous float64 pairwise sum, without importing it."""
    values = list(map(float, values))
    if len(values) < 8:
        result = -0.0
        for value in values:
            result += value
        return result
    require(len(values) <= 128, "native29 arithmetic is bounded")
    totals = values[:8]
    end = len(values) - len(values) % 8
    for offset in range(8, end, 8):
        for i in range(8):
            totals[i] += values[offset + i]
    result = ((totals[0] + totals[1]) + (totals[2] + totals[3])) + (
        (totals[4] + totals[5]) + (totals[6] + totals[7])
    )
    for value in values[end:]:
        result += value
    return result


def parent_weights(records):
    counts = Counter(r["split_group"] for r in records)
    values = [1.0 / counts[r["split_group"]] for r in records]
    mean = np_sum(values) / len(values)
    return [value / mean for value in values]


def verified_delivery(root, expected):
    require(
        sha(Path(root) / "delivery_receipt.json") == expected, "source snapshot delivery changed"
    )
    receipt = read(Path(root) / "delivery_receipt.json")
    for name, pin in receipt["files"].items():
        require(sha(path_at(root, name)) == pin, "source snapshot file changed: " + name)
    return receipt


def verify_folds(source, folds, ids):
    by_id = {r["record_id"]: r for r in source}
    require(
        len(folds) == 5 and [f["fold"] for f in folds] == list(range(5)),
        "exactly five named folds required",
    )
    seen = []
    for fold in folds:
        a, b = fold["train_record_ids"], fold["validation_record_ids"]
        require(
            len(a) == len(set(a))
            and len(b) == len(set(b))
            and a
            and b
            and not set(a) & set(b)
            and set(a) | set(b) == set(ids),
            "fold membership changed",
        )
        require(
            not {by_id[i]["split_group"] for i in a} & {by_id[i]["split_group"] for i in b},
            "known-parent group enters its own validation fold",
        )
        require(
            not {by_id[i]["canonical_identity_group"] for i in a}
            & {by_id[i]["canonical_identity_group"] for i in b},
            "wholegraph leakage",
        )
        require(
            {by_id[i]["target"] for i in a} == {by_id[i]["target"] for i in b} == {"0", "1"},
            "each actual fold needs both source classes",
        )
        seen.extend(b)
    require(Counter(seen) == Counter(ids), "OOF membership must cover each declared record once")


def verify_plans(source, plans):
    ids = [r["record_id"] for r in source]
    by_id = {r["record_id"]: r for r in source}
    require(
        plans["requested_seeds"] == list(SEEDS)
        and not plans["performance_used_for_split_choice"]
        and plans["model_fits"] == 0
        and plans["full29_selected_parameter_OOF_is_deployment_calibration_not_holdout"] is True,
        "prefit seed/selection/deployment scope changed",
    )
    runtime = plans["runtime"]
    require(
        runtime["host"] == "node23"
        and runtime["trainer_sha256"] == TRAIN_SHA
        and runtime["production_fits"] == runtime["model_loads"] == 0
        and runtime["no_bytecode"] is True,
        "actual node23 prefit runtime evidence missing",
    )
    require(
        [s["seed"] for s in plans["seeds"]] == list(SEEDS), "three actual seeded plans required"
    )
    for plan in plans["seeds"]:
        a, b = plan["outer_train_record_ids"], plan["outer_holdout_record_ids"]
        require(
            len(a) == len(set(a))
            and len(b) == len(set(b))
            and a
            and b
            and not set(a) & set(b)
            and set(a) | set(b) == set(ids),
            "outer split membership changed",
        )
        ag, bg = {by_id[i]["split_group"] for i in a}, {by_id[i]["split_group"] for i in b}
        require(
            not ag & bg
            and {by_id[i]["target"] for i in a} == {by_id[i]["target"] for i in b} == {"0", "1"},
            "outer holdout must be group-disjoint with both classes",
        )
        outer = plan["outer_split"]
        require(
            outer["seed"] == plan["actual_outer_seed"]
            and outer["record_groups"] == {r["record_id"]: r["split_group"] for r in source}
            and set(outer["train_records"]) == set(a)
            and set(outer["test_records"]) == set(b)
            and set(outer["train_groups"]) == ag
            and set(outer["test_groups"]) == bg
            and plan["selected_model_fit_scope"] == "outer_train_only"
            and plan["all29_refit_required"] is True,
            "actual outer split/model fit scope differs from prefit plan",
        )
        verify_folds(source, plan["inner_folds"], a)
        verify_folds(source, plan["full29_oof_folds"], ids)
    return plans


def verify_features(root, source, policy):
    root = Path(root)
    manifest = read(root / "features/manifest.json")
    require(
        object_sha(manifest["artifact_hashes"]) == FEATURE_ARTIFACT_MAP_SHA
        and object_sha(manifest["groups"]) == FEATURE_GROUPS_SHA
        and object_sha(manifest["representation_methods"]) == FEATURE_METHODS_SHA,
        "original feature artifact map, schema or representation registry changed",
    )
    configuration = manifest["configuration"]
    require(
        object_sha(configuration) == manifest["input_fingerprint"] == FEATURE_FINGERPRINT
        and configuration["implementation_sha256"] == PRODUCER_SHA
        and configuration["diagnostic_policy_sha256"] == object_sha(policy) == POLICY_SHA
        and configuration["contexts_used_as_covariates"] is False
        and configuration["standard_sequence_projection_used"] is False
        and configuration["record_ids"] == [r["record_id"] for r in source]
        and configuration["graphs"]
        == {r["record_id"]: r["canonical_identity_group"] for r in source},
        "graph-only feature fingerprint/configuration changed",
    )
    require(
        set(manifest["groups"]) == set(CHEMICAL_GROUPS)
        and manifest["routes"]["structure_methods_all"] == list(CHEMICAL_GROUPS)
        and len(manifest["routes"]) == 11,
        "13 chemical groups/11 routes required; no context arms",
    )
    methods = manifest["representation_methods"]
    require(
        methods["independent_method_count"] == len(methods["methods"]) == 10
        and len({m["algorithm_family_id"] for m in methods["methods"]}) == 10
        and all(
            m["source_kind"] == "structure" and m["independent"] is True for m in methods["methods"]
        ),
        "ten independent structural methods required; fusion is not another method",
    )
    cache_name = re.split(r"[/\\]", manifest["cache_dir"])[-1]
    require(cache_name == FEATURE_FINGERPRINT[:16], "feature-cache identity changed")
    cache = path_at(root, "features/" + cache_name)
    cache_manifest = read(cache / "manifest.json")
    require(cache_manifest == manifest, "feature pointer/cache manifest disagree")
    for name, pin in manifest["artifact_hashes"].items():
        require(sha(path_at(cache, name)) == pin, "numeric feature artifact changed: " + name)
    registry = read(root / "representation_methods.json")
    require(registry == methods, "representation registry differs from frozen feature methods")
    bank = table(cache / "structure_bank_index.tsv")
    index = {r["key"]: int(r["row_idx"]) for r in bank}
    require(
        len(bank) == len(index) == 29
        and set(index.values()) == set(range(29))
        and set(index) == {r["canonical_identity_group"] for r in source},
        "wholegraph feature index mismatch",
    )
    graphs = table(cache / "structure_source_smiles.tsv")
    require(
        {r["key"]: r["canonical_isomeric_smiles"] for r in graphs}
        == {r["canonical_identity_group"]: r["canonical_smiles"] for r in source},
        "full SMILES bank changed",
    )
    group_rows, width = {}, 0
    for name in CHEMICAL_GROUPS:
        group = manifest["groups"][name]
        require(
            group["index"] == "structure_bank_index.tsv"
            and group["key_kind"] == "canonical_identity_group",
            "chemical group has an undeclared record/context mapping",
        )
        h, body = npy(cache / group["matrix"])
        mh, mask = npy(cache / group["valid"])
        require(
            list(h["shape"]) == group["shape"]
            and h["shape"][0] == 29
            and len(group["feature_names"]) == h["shape"][1]
            and mh["descr"] == "|b1"
            and mh["shape"] == (29,)
            and set(mask) == {1},
            "chemical array shape/name/validity mismatch",
        )
        columns = h["shape"][1]
        code = {"|u1": "B", "<f4": "f"}.get(h["descr"])
        require(code is not None, "stored chemical arrays must be uint8 or float32")
        stride = columns * struct.calcsize(code)
        rows = {}
        for key, number in index.items():
            values = body[number * stride : (number + 1) * stride]
            rows[key] = values if code == "f" else struct.pack("<" + str(columns) + "f", *values)
        group_rows[name] = rows
        width += columns
    require(
        width == 15212 and len(manifest["groups"]) == 13, "exact15212 chemical dimensions required"
    )
    return dict(manifest=manifest, cache=cache, group_rows=group_rows, chemical_dimensions=width)


def design(binding, source, route, dtype="float32"):
    groups = binding["manifest"]["routes"][route]
    require(
        groups and len(groups) == len(set(groups)) and set(groups) <= set(CHEMICAL_GROUPS),
        "route contains duplicate or nonchemical features",
    )
    names = [
        group + ":" + name
        for group in groups
        for name in binding["manifest"]["groups"][group]["feature_names"]
    ]
    result = {}
    for record in source:
        raw = b"".join(
            binding["group_rows"][group][record["canonical_identity_group"]] for group in groups
        )
        if dtype == "float64":
            raw = struct.pack(
                "<" + str(len(names)) + "d", *struct.unpack("<" + str(len(names)) + "f", raw)
            )
        result[record["record_id"]] = raw
    return dict(dtype=dtype, names=names, rows=result)


def matrix_signature(matrix, ids):
    return dict(
        dtype=matrix["dtype"],
        shape=[len(ids), len(matrix["names"])],
        bytes_sha256=hashlib.sha256(b"".join(matrix["rows"][rid] for rid in ids)).hexdigest(),
    )


def verify_report_only_ast(path):
    """The fixed producer's report call graph cannot reach fitting or model loading."""
    require(sha(path) == PRODUCER_SHA, "report purity requires the original frozen producer")
    tree = ast.parse(Path(path).read_text(encoding="utf8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    pending, visited = ["render_report", "verify_diagnostics_and_build_report"], set()
    forbidden = {
        "learn_seed_diagnostics",
        "fit_platt",
        "calibration_artifact",
        "fit_prediction_artifact",
        "oof_learning",
        "run_followup",
        "_fit_model",
        "_estimator",
        "search_candidates",
    }
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        require(name not in forbidden, "report call graph reaches diagnostic learning")
        visited.add(name)
        for call in (node for node in ast.walk(functions[name]) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name):
                require(call.func.id not in forbidden, "report calls fitting/search factory")
                if call.func.id in functions:
                    pending.append(call.func.id)
            elif isinstance(call.func, ast.Attribute):
                require(
                    call.func.attr not in {"fit", "fit_transform", "partial_fit", "optimize"},
                    "report calls estimator fitting",
                )
                require(
                    not (
                        call.func.attr in {"load", "loads"}
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id in {"joblib", "pickle"}
                    ),
                    "report deserializes models",
                )
    return dict(
        report_call_graph_functions=len(visited),
        producer_sha256=PRODUCER_SHA,
        model_fits=0,
        model_loads=0,
        purity_scope="frozen_source_AST_no_execution",
    )


def native_source(root):
    """Bind source29, reference parents, all32 ledger, actual node folds and features."""
    root = Path(root)
    manifest = read(root / "task_manifest.json")
    require(len(manifest["tasks"]) == 1, "isolated native29 task required")
    task = manifest["tasks"][0]
    require(
        task["task_id"] == TASK
        and task["task_kind"] == "classification"
        and task["endpoint"] == "hemolysis_SOURCE_DEFINED_BINARY"
        and task["canonical_quantity_id"] == QUANTITY
        and task["unit"] == "label01"
        and task["target_transform"] == "identity"
        and task["inverse_transform"] == "identity"
        and task["requested_inner_folds"] == 5,
        "source-native classification task/quantity changed",
    )
    descriptor = task["source_native_binary_observation_coverage"]
    require(
        descriptor["proof_sha256"] == PROOF_SHA
        and sha(root / descriptor["proof_path"]) == PROOF_SHA,
        "frozen native source proof changed",
    )
    proof = read(root / descriptor["proof_path"])
    require(
        proof["implementation_sha256"] == PRODUCER_SHA
        and proof["admission_review_sha256"] == ADMISSION_SHA
        and proof["structure_delivery_sha256"] == STRUCTURE_DELIVERY_SHA
        and proof["schema"] == SCHEMA,
        "source proof producer/admission/structure binding changed",
    )
    for item in proof["artifacts"].values():
        require(
            sha(path_at(root, item["path"])) == item["sha256"],
            "source-stage artifact missing or changed: " + item["path"],
        )
    for relative, expected in SCRIPT_PINS.items():
        require(
            sha(path_at(root, relative)) == expected,
            "original frozen script closure changed: " + relative,
        )
    require(
        sha(root / "scripts" / MODULE) == PRODUCER_SHA
        and sha(root / "scripts/peptide_modeling_train.py") == TRAIN_SHA,
        "frozen source/trainer bytes changed",
    )
    admission = root / proof["admission_snapshot"]
    structure = root / proof["structure_snapshot"]
    verified_delivery(admission, ADMISSION_DELIVERY_SHA)
    verified_delivery(structure, STRUCTURE_DELIVERY_SHA)
    negative_path = (
        root / proof["negative_snapshot"] / "fifty_two_source_objects_and_reference_graphs.json"
    )
    require(sha(negative_path) == NEGATIVE_SHA, "original52 negative-source snapshot changed")
    review = read(admission / "admission_review.json")
    require(
        sha(admission / "admission_review.json") == ADMISSION_SHA
        and review["hemolysis_explicit_RBC_material_only"]["class_group_counts"]["1"] == 4,
        "native annotation cannot become the stricter explicit-RBC contract",
    )
    contract = review["hemolysis_field_aware_native_inactivity_extension"]
    parent_of = {
        i: "DRAMP_NATIVE_GROUP:" + hashlib.sha256("||".join(sorted(m)).encode()).hexdigest()[:24]
        for m in contract["components"].values()
        for i in m
    }
    require(
        (
            contract["total_rows"],
            contract["positive_rows"],
            contract["negative_rows"],
            contract["total_components"],
            contract["class_group_counts"],
        )
        == (29, 7, 22, 19, {"0": 13, "1": 7}),
        "admitted native29 source/group counts changed",
    )
    selected = {
        r["source_id"]: r
        for r in read(admission / "hemolysis_native_candidate_observations_review_only.json")
        + read(admission / "six_original_native_inactivity_field_extension.json")
        if r["source_id"] in parent_of
    }
    negatives = {r["record"]["source_entry_id"]: r for r in read(negative_path)}
    positive_cards = {r["source_id"]: r for r in read(structure / "source_structure_cards.json")}
    contexts = {r["source_id"]: r for r in read(admission / "per_source_context_role_review.json")}
    source_path = root / task["table_path"]
    require(sha(source_path) == TABLE_SHA, "exact source29 table changed")
    source = table(source_path)
    require(
        len(source) == len({r["record_id"] for r in source}) == 29
        and Counter(r["target"] for r in source) == {"0": 22, "1": 7}
        and len({r["split_group"] for r in source}) == 19
        and len({r["canonical_identity_group"] for r in source}) == 29,
        "source29 unique row/class/reference/parent counts changed",
    )
    require(
        {r["source_entry_id"] for r in source if r["target"] == "1"} == POSITIVE_IDS,
        "seven source-affirmative objects changed",
    )
    for row in source:
        sid, rid = row["source_entry_id"], row["record_id"]
        item = selected[sid]
        require(
            rid
            == "DRAMP_NATIVE:" + hashlib.sha256(item["source_entry_key"].encode()).hexdigest()[:24]
            and row["target"] == row["value"] == str(item["reviewed_native_polarity"])
            and row["native_annotation"] == item["annotation"]
            and row["polarity_spans_json"]
            and json.loads(row["polarity_spans_json"]) == item["polarity_spans"]
            and row["split_group"] == parent_of[sid]
            and row["source_entry_key"] == item["source_entry_key"],
            "source-native label/span/parent lineage changed",
        )
        for item_span in item["polarity_spans"]:
            require(
                item["annotation"][item_span["start"] : item_span["end"]]
                == item_span["exact_text"],
                "native polarity span differs from its original field",
            )
        if row["target"] == "1":
            card = positive_cards[sid]
            require(
                card["raw_affirmative_evidence"]["source_reference_tier"]
                == "DRAMP_secondary_curated_annotation_not_primary_verified",
                "SwissProt-only layer entered labels",
            )
            graph = card["candidate_graph"]
            old_ids = [
                r["record_id"]
                for r in json.loads(card["raw_affirmative_evidence"]["exact_catalogue_memberships"])
            ]
        else:
            card = negatives[sid]
            require(card["record"]["value"] == "0", "an original negative was flipped")
            graph = card["reference_graph"]
            old_ids = [card["record"]["record_id"]]
        smiles = graph["canonical_isomeric_smiles"]
        require(
            row["canonical_smiles"] == row["smiles"] == smiles
            and row["canonical_identity_group"]
            == item["whole_graph_key"]
            == "CANON_SMILES:" + hashlib.sha256(smiles.encode()).hexdigest()
            and json.loads(row["original_catalogue_record_ids"]) == old_ids
            and row["sequence"] == ""
            and row["sequence_eligible"] == "false"
            and row["structure_eligible"] == "true"
            and row["annotation_only"] == "true"
            and row["primary_sample_identity_verified"] == "false",
            "complete reference graph/source identity changed",
        )
        context = contexts[sid]
        require(
            json.loads(row["context_roles_json"]) == context
            and row["assay_material_literal"]
            == (" | ".join(s["exact_text"] for s in context["material_spans"]) or "not_reported")
            and row["assay_species_literal"]
            == (
                " | ".join(s["exact_text"] for s in context["assay_species_word_spans"])
                or "not_reported"
            ),
            "source context was inferred or borrowed",
        )
    dispositions = read(root / proof["artifacts"]["original32_dispositions"]["path"])
    require(
        len(dispositions) == len({r["record_id"] for r in dispositions}) == 32
        and sum(r["included_in_this29_source_annotation_task"] for r in dispositions) == 22
        and all(r["old_label"] == "0" and r["old_label_unchanged"] is True for r in dispositions),
        "original32 ledger/qualified22 participation changed",
    )
    included = {
        r["record_id"] for r in dispositions if r["included_in_this29_source_annotation_task"]
    }
    require(
        set(descriptor["original_record_ids"]) == included
        and descriptor["all_original_records_covered"] is False
        and (
            descriptor["original_labelled_records"],
            descriptor["included_original_records"],
            descriptor["new_affirmative_source_records"],
            descriptor["pooled_records"],
        )
        == (32, 22, 7, 29),
        "22/32 source participation cannot be reported as full32 coverage",
    )
    policy = proof["diagnostic_policy"]
    require(
        object_sha(policy) == POLICY_SHA and read(root / "diagnostic_policy.json") == policy,
        "39-fit/one-control/calibration policy changed",
    )
    definition = task["scientific_definition"]
    require(
        task["task_definition"] == definition
        and definition["diagnostic_policy"] == policy
        and definition["source_native_annotation_only"] is True
        and definition["query_mode"] == QUERY
        and definition["prediction_covariates"] == "complete_graph_only"
        and definition["context_features"] is False
        and definition["physical_shared_threshold"] is False
        and definition["clinical_inference_supported"] is False
        and definition["original_negative_observation_coverage"] == "22/32"
        and definition["implementation_sha256"] == PRODUCER_SHA
        and task["scientific_warning"] == proof["warning"],
        "annotation-only/scientific scope changed",
    )
    plan_path = root / proof["artifacts"]["node_prefit_plan"]["path"]
    require(sha(plan_path) == PLAN_SHA, "actual node prefit plan changed")
    plans = verify_plans(source, read(plan_path))
    features = verify_features(root, source, policy)
    config = read(root / "config.json")
    require(
        config["diagnostic_policy"] == policy
        and set(config["training"]["models"]) == MODELS
        and set(config["training"]["routes"]) == set(features["manifest"]["routes"])
        and config["training"]["search_backend"] == "optuna"
        and config["training"]["search_budget"] == 128
        and config["training"]["inner_folds"] == 5
        and config["training"]["refit_full"] is True
        and config["training"]["n_jobs"] == config["resources"]["cpu_budget"] == 1
        and config["resources"]["training_host"] == "node23",
        "frozen source training/diagnostic resource contract changed",
    )
    purity = verify_report_only_ast(root / "scripts" / MODULE)
    return dict(
        task=task,
        source=source,
        proof=proof,
        descriptor=descriptor,
        dispositions=dispositions,
        plans=plans,
        features=features,
        policy=policy,
        report_purity=purity,
    )


def is_task(task):
    return task.get("task_id") == TASK


def verify_source(entry, task, api=None):
    manifest = Path(entry["manifest_path"])
    require(
        sha(manifest) == entry["manifest_sha256"] and entry["protocol"] == "group_holdout_inner5",
        "registry manifest/protocol changed",
    )
    bound = native_source(manifest.parent)
    require(
        task["task_id"] == TASK
        and task["records"] == 29
        and task["table_relative"] == bound["task"]["table_path"]
        and task["table_sha256"] == TABLE_SHA
        and task["scientific_definition"] == bound["task"]["scientific_definition"],
        "registry native source descriptor mismatch",
    )
    return bound


def adapter_payload(path=None):
    raw = Path(path or __file__).read_bytes()
    return dict(
        source_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest()
    )


def load_adapter(payload):
    raw = base64.b64decode(payload["source_base64"], validate=True)
    require(hashlib.sha256(raw).hexdigest() == payload["sha256"], "adapter transport SHA changed")
    namespace = {"__name__": "peptide_modeling_dramp_hemolysis_native_binary_execution_audit"}
    exec(compile(raw, "<frozen_stdlib_DRAMP_native_binary_audit>", "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def weighted_mcc(y, p, weights, threshold):
    totals = Counter()
    for target, probability, weight in zip(y, p, weights, strict=True):
        totals[(int(target), int(probability >= threshold))] += weight
    tp, tn, fp, fn = totals[(1, 1)], totals[(0, 0)], totals[(0, 1)], totals[(1, 0)]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def probability_metrics(y, p, weights, threshold):
    require(
        len(y) == len(p) == len(weights) > 0
        and set(y) <= {0, 1}
        and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        and all(math.isfinite(v) and v >= 0 for v in weights)
        and sum(weights) > 0,
        "finite binary probabilities/positive-mass weights required",
    )
    mass = sum(weights)
    positive = sum(w for v, w in zip(y, weights, strict=True) if v == 1)
    negative = mass - positive
    both = positive > 0 and negative > 0
    roc, average_precision = None, None
    if both:
        roc = sum(
            wp * wn * (1.0 if pp > pn else 0.5 if pp == pn else 0.0)
            for yp, pp, wp in zip(y, p, weights, strict=True)
            if yp == 1
            for yn, pn, wn in zip(y, p, weights, strict=True)
            if yn == 0
        ) / (positive * negative)
        tp, fp, average_precision = 0.0, 0.0, 0.0
        for value in sorted(set(p), reverse=True):
            added_positive = sum(
                w
                for target, score, w in zip(y, p, weights, strict=True)
                if score == value and target == 1
            )
            tp += added_positive
            fp += sum(
                w
                for target, score, w in zip(y, p, weights, strict=True)
                if score == value and target == 0
            )
            average_precision += added_positive / positive * tp / (tp + fp)
    bins, ece = [], 0.0
    for number in range(5):
        members = [i for i, value in enumerate(p) if min(int(value * 5), 4) == number]
        weight = sum(weights[i] for i in members)
        mean_p = sum(weights[i] * p[i] for i in members) / weight if weight else None
        mean_y = sum(weights[i] * y[i] for i in members) / weight if weight else None
        if weight:
            ece += weight / mass * abs(mean_p - mean_y)
        bins.append(
            dict(
                bin=number,
                records=len(members),
                weight=weight,
                mean_probability=mean_p,
                observed_fraction=mean_y,
            )
        )
    safe = [min(max(v, 1e-15), 1 - 1e-15) for v in p]
    balanced = (
        (
            sum(
                w
                for v, score, w in zip(y, p, weights, strict=True)
                if v == 1 and score >= threshold
            )
            / positive
            + sum(
                w for v, score, w in zip(y, p, weights, strict=True) if v == 0 and score < threshold
            )
            / negative
        )
        / 2
        if both
        else None
    )
    return dict(
        n=len(y),
        positive=y.count(1),
        negative=y.count(0),
        two_classes_present=both,
        threshold=float(threshold),
        pr_auc=average_precision,
        roc_auc=roc,
        mcc=weighted_mcc(y, p, weights, threshold),
        brier=sum(w * (score - target) ** 2 for target, score, w in zip(y, p, weights, strict=True))
        / mass,
        log_loss=sum(
            w * (-target * math.log(score) - (1 - target) * math.log(1 - score))
            for target, score, w in zip(y, safe, weights, strict=True)
        )
        / mass,
        balanced_accuracy=balanced,
        ece_fixed5_bins=ece,
        reliability_bins=bins,
    )


def threshold_choice(records, p, scope, policy):
    y = [int(r["target"]) for r in records]
    weights = parent_weights(records)
    unique = sorted(set(p))
    candidates = sorted(
        {
            0.0,
            0.5,
            1.0,
            *unique,
            *[(a + b) / 2 for a, b in zip(unique[:-1], unique[1:], strict=True)],
        }
    )
    scores = [weighted_mcc(y, p, weights, threshold) for threshold in candidates]
    best = max(scores)
    tied = [
        t
        for t, score in zip(candidates, scores, strict=True)
        if best - score <= policy["threshold"]["tie_score_tolerance"]
    ]
    selected = min(tied, key=lambda value: (abs(value - 0.5), value))
    deployment = scope == "deployment_full29_fixed_parameter_OOF"
    return dict(
        threshold=float(selected),
        objective="known_parent_equal_MCC",
        score=float(best),
        candidates=candidates,
        policy=policy["threshold"],
        source_record_ids=[r["record_id"] for r in records],
        target_float64_sha256=signature(y)["bytes_sha256"],
        probability_float64_sha256=signature(p)["bytes_sha256"],
        weight_float64_sha256=signature(weights)["bytes_sha256"],
        fit_scope=scope,
        labels_from_declared_fit_records_only=True,
        outer_evaluation_holdout_labels_used=None if deployment else False,
        former_holdout_now_in_deployment_training=deployment,
    )


def apply_platt(artifact, probability, policy):
    require(
        artifact["schema"] == "fixed_Platt_logit_C1_native_source_v1"
        and artifact["policy"] == policy["platt"]
        and artifact["classes"] == [0.0, 1.0],
        "portable Platt policy/classes changed",
    )
    coefficient, intercept = float(artifact["coefficient"]), float(artifact["intercept"])
    require(math.isfinite(coefficient) and math.isfinite(intercept), "nonfinite Platt coefficients")
    values = []
    for p in probability:
        p = min(max(p, policy["platt"]["clip"]), 1 - policy["platt"]["clip"])
        z = math.log(p / (1 - p)) * coefficient + intercept
        require(math.isfinite(z), "nonfinite portable calibration linear predictor")
        values.append(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    return values


def verify_calibration(path, records, p, seed, scope, policy):
    artifact = read(path)
    clipped = [min(max(v, policy["platt"]["clip"]), 1 - policy["platt"]["clip"]) for v in p]
    logits = [math.log(v / (1 - v)) for v in clipped]
    deployment = scope == "deployment_full29_fixed_parameter_OOF"
    expected = dict(
        scope=scope,
        seed=seed,
        fit_record_ids=[r["record_id"] for r in records],
        fit_parent_groups=sorted({r["split_group"] for r in records}),
        raw_probability_signature=signature(p),
        logit_input_signature=signature(logits, shape=[len(p), 1]),
        target_signature=signature([int(r["target"]) for r in records]),
        weight_signature=signature(parent_weights(records)),
        policy=policy["platt"],
        fitted_coefficient_dtype="float64",
        independent_performance_claim=False,
        labels_from_declared_fit_records_only=True,
        outer_evaluation_holdout_labels_used=None if deployment else False,
        former_holdout_now_in_deployment_training=deployment,
    )
    for key, value in expected.items():
        require(artifact.get(key) == value, "Platt fit input/membership/hash/scope differs: " + key)
    return artifact, apply_platt(artifact, p, policy)


def prediction_values(source, path, ids, folds):
    by_id = {r["record_id"]: r for r in source}
    values = table(path)
    require(
        len(values) == len({r["record_id"] for r in values}) == len(ids)
        and {r["record_id"] for r in values} == set(ids),
        "saved probability membership changed",
    )
    probabilities = {}
    for row in values:
        rid = row["record_id"]
        p = float(row["prediction"])
        require(
            row["split_group"] == by_id[rid]["split_group"]
            and float(row["target"]) == int(by_id[rid]["target"])
            and int(row["fold"]) == folds[rid]
            and math.isfinite(p)
            and 0 <= p <= 1,
            "source/target/fold/probability differs from frozen source",
        )
        probabilities[rid] = p
    return [probabilities[rid] for rid in ids]


def enriched(source, path, ids, folds, role, raw, calibrated=None):
    observed = table(path)
    require(
        [r["record_id"] for r in observed] == ids, "enriched probability order/membership changed"
    )
    by_id = {r["record_id"]: r for r in source}
    expected = []
    for i, rid in enumerate(ids):
        value = dict(
            record_id=rid,
            split_group=by_id[rid]["split_group"],
            target=float(by_id[rid]["target"]),
            prediction=raw[i],
            fold=folds[rid],
            evaluation_role=role,
            calibrated_probability=calibrated[i] if calibrated is not None else None,
        )
        row = observed[i]
        require(
            row["record_id"] == rid
            and row["split_group"] == value["split_group"]
            and row["evaluation_role"] == role
            and int(row["fold"]) == value["fold"],
            "enriched source/evaluation role changed",
        )
        close(row["target"], value["target"])
        require(float(row["prediction"]) == value["prediction"], "enriched raw probability changed")
        if calibrated is None:
            require(
                row["calibrated_probability"] == "",
                "unrequested calibration attached to fixed control",
            )
        else:
            close(
                row["calibrated_probability"],
                calibrated[i],
                "calibrated probability differs from portable JSON",
                rtol=1e-12,
                atol=1e-14,
            )
            # Replay checks the formula; persisted binary64 values bind threshold hashes.
            value["calibrated_probability"] = float(row["calibrated_probability"])
            require(
                0 <= value["calibrated_probability"] <= 1,
                "saved calibrated probability outside[0,1]",
            )
        expected.append(value)
    return expected


def role_metrics(source, predictions, raw_threshold, calibrated_threshold=None):
    by_id = {r["record_id"]: r for r in source}
    records = [by_id[r["record_id"]] for r in predictions]
    if not records:
        return dict(records=0, parent_groups=0, raw=None, calibrated=None)
    y = [int(r["target"]) for r in records]

    def calculate(values, threshold):
        return dict(
            record_weighted=probability_metrics(y, values, [1.0] * len(y), threshold),
            parent_equal=probability_metrics(y, values, parent_weights(records), threshold),
        )

    return dict(
        records=len(records),
        parent_groups=len({r["split_group"] for r in records}),
        raw=calculate([r["prediction"] for r in predictions], raw_threshold),
        calibrated=calculate(
            [r["calibrated_probability"] for r in predictions], calibrated_threshold
        )
        if calibrated_threshold is not None
        else None,
    )


def verify_stage_contract(run, bound):
    """Demand source closure in the real pre-fit round contract, not a later repair."""
    run = Path(run)
    contract = read(run / "campaign_input_contract.json")["template_contract"]
    required = {item["path"]: item["sha256"] for item in bound["proof"]["artifacts"].values()}
    required[bound["task"]["source_native_binary_observation_coverage"]["proof_path"]] = PROOF_SHA
    for snapshot, expected in [
        (bound["proof"]["admission_snapshot"], ADMISSION_DELIVERY_SHA),
        (bound["proof"]["structure_snapshot"], STRUCTURE_DELIVERY_SHA),
    ]:
        prefix = snapshot.rstrip("/") + "/"
        required[prefix + "delivery_receipt.json"] = expected
        required.update(
            {
                prefix + name: pin
                for name, pin in read(run / snapshot / "delivery_receipt.json")["files"].items()
            }
        )
    required[
        bound["proof"]["negative_snapshot"] + "/fifty_two_source_objects_and_reference_graphs.json"
    ] = NEGATIVE_SHA
    required.update(SCRIPT_PINS)
    for name, expected in required.items():
        require(
            contract["files"].get(name) == expected and sha(path_at(run, name)) == expected,
            "actual pre-fit source-stage contract omits or changes: " + name,
        )
    extension = contract.get("runtime_source_closure")
    require(isinstance(extension, dict), "explicit source-complete prefit extension missing")
    expected = dict(
        schema=CLOSURE_SCHEMA,
        wrapper_module=WRAPPER,
        wrapper_sha256=WRAPPER_SHA,
        base_builder_module="peptide_modeling_campaign.py",
        base_builder_sha256=SCRIPT_PINS["scripts/peptide_modeling_campaign.py"],
        base_followup_module="peptide_modeling_followup_seeds.py",
        base_followup_sha256=SCRIPT_PINS["scripts/peptide_modeling_followup_seeds.py"],
        producer_module=MODULE,
        producer_sha256=PRODUCER_SHA,
        training_source_sha256=TRAIN_SHA,
        base_delivery_receipt_sha256=BASE_DELIVERY_SHA,
        added_files={"diagnostic_policy.json": POLICY_FILE_SHA},
        discovery="frozen_native_source_proof_artifacts",
        before_any_fit=True,
        source_proof_path=bound["task"]["source_native_binary_observation_coverage"]["proof_path"],
        source_proof_sha256=PROOF_SHA,
        base_contract_files_sha256=object_sha(
            {k: v for k, v in contract["files"].items() if k != "diagnostic_policy.json"}
        ),
        base_config_sha256=contract["template_config_sha256"],
        base_original_config_sha256=ORIGINAL_CONFIG_SHA,
        registry_sha256=sha(Path(run) / "representation_methods.json"),
        numerical_training_config_sha256=TRAINING_CONFIG_SHA,
        numerical_learning_logic_changed=False,
        original_base_files_changed=False,
        stock_search_objective=STOCK_OBJECTIVE,
    )
    require(
        extension == expected
        and contract["registry_sha256"] == expected["registry_sha256"]
        and contract["files"].get("scripts/" + WRAPPER) == WRAPPER_SHA
        and sha(Path(run) / "scripts" / WRAPPER) == WRAPPER_SHA,
        "source-complete wrapper/base inventory differs from the frozen prefit contract",
    )
    settings = dict(read(Path(run) / "config.json")["training"])
    settings["seed"] = 42
    require(
        object_sha(settings) == TRAINING_CONFIG_SHA,
        "source inventory patch altered numerical training settings",
    )
    return contract


def verify_runtime_source_closure(entry, journal, bound):
    """Verify the actual wrapper owner and its before-producer sidecar on both storage roots."""
    cfg = read(entry["config_path"])
    require(
        cfg.get("required_seeds") == list(SEEDS)
        and cfg.get("additional_twenty_hour_loop") is False,
        "native29 is exactly three bounded seeds without another time loop",
    )
    root, durable, template = (
        Path(entry["output_root"]),
        Path(cfg["durable_root"]),
        Path(entry["template"]),
    )
    require(
        sha(entry["config_path"]) == entry["config_sha256"] == journal["config_sha256"],
        "wrapper followup configuration changed",
    )
    base_receipt = template / "source_base_delivery_receipt.json"
    require(
        sha(base_receipt) == BASE_DELIVERY_SHA, "original169-file source delivery binding changed"
    )
    base = read(base_receipt)
    require(len(base["files"]) == 169, "original169-file source inventory changed")
    for name, expected in base["files"].items():
        require(
            sha(path_at(template, name)) == expected,
            "original base bytes changed after deployment: " + name,
        )
    for name, expected in {
        "QA_SIDECAR.json": "d2512ad8cb60186d3f51590e9c6b1c4c046aceaeee7a305f1a06bf1ba1d541a0",
        "RUNBOOK.md": "0753c02bdb24d5a6a044cdf29720160a5c0f55f9b8d20143672258c7ce4e2825",
    }.items():
        require(sha(template / name) == expected, "original preparation QA/runbook sidecar changed")
    require(
        sha(template / "scripts" / WRAPPER) == WRAPPER_SHA,
        "actual controller wrapper bytes changed",
    )
    pin = cfg["runtime_source_closure"]
    core_names = [
        "peptide_modeling_campaign.py",
        "peptide_modeling_followup_seeds.py",
        MODULE,
        "peptide_modeling_train.py",
        "peptide_modeling_search.py",
        "peptide_modeling_pipeline.py",
        "peptide_modeling_features.py",
        "peptide_modeling_dispatch.py",
        "peptide_modeling_supervise.py",
    ]
    expected_pin = dict(
        schema=PIN_SCHEMA,
        wrapper_module=WRAPPER,
        wrapper_sha256=WRAPPER_SHA,
        base_delivery_receipt_sha256=BASE_DELIVERY_SHA,
        producer_sha256=PRODUCER_SHA,
        core_sources_sha256={name: SCRIPT_PINS["scripts/" + name] for name in core_names},
        source_proof_path=bound["task"]["source_native_binary_observation_coverage"]["proof_path"],
        source_proof_sha256=PROOF_SHA,
        template_config_sha256=sha(cfg["template_config"]),
        registry_sha256=sha(template / "representation_methods.json"),
        base_original_config_sha256=ORIGINAL_CONFIG_SHA,
        numerical_training_config_sha256=TRAINING_CONFIG_SHA,
        diagnostic_policy_sha256=POLICY_FILE_SHA,
        allowed_extension="prefit_source_file_inventory_only",
        alias_target="peptide_modeling_followup_seeds.template_contract",
        stock_search_objective=STOCK_OBJECTIVE,
    )
    require(pin == expected_pin, "followup wrapper runtime pin changed numerical/source scope")
    descriptor = journal["runtime_source_closure_receipt"]
    path = path_at(root, descriptor["path"])
    mirror = path_at(durable, descriptor["durable_path"])
    require(
        path.name == "source_closure_runtime.json"
        and sha(path) == descriptor["sha256"] == sha(mirror) == descriptor["durable_sha256"],
        "prefit wrapper sidecar/durable SHA changed",
    )
    runtime = read(path)
    contract = read(root / "template_contract.json")
    require(
        runtime["schema"] == CLOSURE_SCHEMA
        and runtime["runtime_source_closure"]
        == journal["runtime_source_closure"]
        == contract["runtime_source_closure"]
        and runtime["config_sha256"] == entry["config_sha256"]
        and Path(runtime["config_path"]).resolve() == Path(entry["config_path"]).resolve()
        and runtime["closed_contract_sha256"] == object_sha(contract)
        and runtime["written_before_producer_call"] is True
        and runtime["original_entrypoint_not_impersonated"] is True
        and runtime["recorded_at"] <= journal["started_at"],
        "wrapper sidecar was not bound before actual producer start",
    )
    expected_owner = entry["owned_process_identity"]
    observed_owner = runtime["actual_wrapper_owner"]
    for name in ["pid", "start_ticks"]:
        require(
            observed_owner[name] == expected_owner[name] == journal["process_identity"][name],
            "wrapper sidecar/controller uses another process identity",
        )
    argv = runtime["actual_wrapper_argv"]
    require(
        argv == observed_owner["argv"]
        and argv[0] == "/usr/bin/python3"
        and "-B" in argv
        and str(template / "scripts" / WRAPPER) in argv
        and "--config" in argv
        and argv[argv.index("--config") + 1] == str(Path(entry["config_path"]))
        and "--preflight" not in argv
        and "--runtime-pin" not in argv,
        "actual owner argv does not run the declared source-complete wrapper",
    )
    if expected_owner.get("argv"):
        require(expected_owner["argv"] == argv, "registered owner argv differs from actual wrapper")
    return contract


def verify_trials(run, result, settings):
    """Frozen37bd classification maximizes AP/pr_auc, not ROC-AUC or regression MAE."""
    search = result["search"]
    require(
        search["backend"] == "optuna" and search["status"] == "completed",
        "completed Optuna search required",
    )
    contract = search["contract"]
    namespace = object_sha(contract)
    directory = path_at(run, search["search_dir"])
    require(
        namespace == search["namespace"]
        and directory.name == namespace[:24]
        and contract["input_signature"] == result["input_hash"]
        and contract["backend"] == "optuna"
        and contract["task_kind"] == "classification"
        and contract["seed"] == settings["seed"]
        and read(directory / "contract.json") == contract
        and contract["implementation_sha256"]
        == sha(Path(run) / "scripts/peptide_modeling_search.py"),
        "actual search namespace/kind/source binding changed",
    )
    pairs = {(route, model) for route in settings["routes"] for model in settings["models"]}
    require(
        len(pairs) == 88
        and set(map(tuple, contract["pairs"])) == pairs
        and len(contract["pairs"]) == 88,
        "all11chemical routes x8classifier families required",
    )
    stored = read(directory / "search_result.json")
    require(
        {k: v for k, v in stored.items() if k != "trials"}
        == {
            k: v
            for k, v in search.items()
            if k not in {"candidate_fit_admission", "eligible_model_route_pairs"}
        }
        and search["eligible_model_route_pairs"] == 88,
        "published search summary differs",
    )
    records, caches = [], set()
    candidate_dir = path_at(run, result["artifacts"]["model"]).parent / "trials"
    for number, path in enumerate(sorted((directory / "trials").glob("*.json"))):
        envelope = read(path)
        value = envelope["evaluation"]
        require(
            envelope["record_sha256"] == object_sha(value)
            and value["search_trial"] == number
            and path.name == f"{number:06d}.json"
            and value["search_input_signature"] == namespace
            and (value["route"], value["model"]) in pairs
            and value["status"] in {"completed", "failed"},
            "Optuna trial envelope/number/declared pair changed",
        )
        if value["status"] == "completed":
            require(
                value["actual_folds"] == value["requested_folds"] == 5
                and value["input_hash"] == result["input_hash"],
                "successful classification candidate is not real fivefold",
            )
            close(
                value["score"],
                value["metrics"]["pr_auc"],
                "classification objective must be AP/pr_auc",
            )
            require(0 <= value["score"] <= 1, "classification AP objective outside[0,1]")
        else:
            require(
                value.get("score") is None and bool(value.get("reason")),
                "failed candidate gained a numeric score",
            )
        key = object_sha(
            [result["input_hash"], value["route"], value["model"], value["params"]], compact=False
        )
        cache = candidate_dir / (key + ".json")
        if cache.exists():
            require(
                all(value.get(k) == v for k, v in read(cache).items()),
                "candidate cache differs from actual trial",
            )
            caches.add(cache.name)
        else:
            require(value["status"] == "failed", "successful candidate cache missing")
        records.append(value)
    require(
        len(records) == settings["search_budget"] == 128
        and search["budget"] == {"requested": 128, "used": 128, "remaining": 0, "pairs": 88}
        and stored["trials"] == records
        and {p.name for p in candidate_dir.glob("*.json")} == caches
        and set(map(tuple, search["coverage_attempted"]))
        == {(v["route"], v["model"]) for v in records}
        == pairs,
        "128 actual trials/88 pairs/cache coverage incomplete",
    )
    require(
        search["completed"] == sum(v["status"] == "completed" for v in records)
        and search["failed"] == sum(v["status"] == "failed" for v in records),
        "trial status counts changed",
    )
    database = path_at(run, search["storage_path"])
    require(
        database == directory / "study.sqlite3"
        and search["study_name"] == "peptide_" + namespace[:24],
        "Optuna DB identity changed",
    )
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        names = connection.execute("SELECT study_name FROM studies").fetchall()
        directions = connection.execute("SELECT direction FROM study_directions").fetchall()
        values = connection.execute(
            "SELECT t.number,t.state,a.value_json FROM trials t LEFT JOIN trial_user_attributes a "
            "ON a.trial_id=t.trial_id AND a.key='evaluation_result' ORDER BY t.number"
        ).fetchall()
        objectives = connection.execute(
            "SELECT t.number,v.value FROM trials t JOIN trial_values v ON v.trial_id=t.trial_id "
            "WHERE t.state='COMPLETE' ORDER BY t.number"
        ).fetchall()
        candidates = connection.execute(
            "SELECT t.number,a.value_json FROM trials t JOIN trial_user_attributes a "
            "ON a.trial_id=t.trial_id AND a.key='candidate' ORDER BY t.number"
        ).fetchall()
        parameters = connection.execute(
            "SELECT t.number,p.param_name,p.param_value,p.distribution_json FROM trials t "
            "JOIN trial_params p ON p.trial_id=t.trial_id ORDER BY t.number,p.param_name"
        ).fetchall()
    require(
        names == [(search["study_name"],)]
        and directions == [("MAXIMIZE",)]
        and len(values) == 128
        and all(
            n == i
            and state == ("COMPLETE" if records[i]["status"] == "completed" else "FAIL")
            and raw is not None
            and json.loads(raw) == records[i]
            for i, (n, state, raw) in enumerate(values)
        ),
        "SQLite trial state/evaluation differs",
    )
    require(
        candidates
        == [
            (
                v["search_trial"],
                json.dumps(dict(route=v["route"], model=v["model"], params=v["params"])),
            )
            for v in records
        ]
        or (
            len(candidates) == 128
            and all(
                number == i
                and json.loads(raw)
                == dict(
                    route=records[i]["route"],
                    model=records[i]["model"],
                    params=records[i]["params"],
                )
                for i, (number, raw) in enumerate(candidates)
            )
        ),
        "SQLite candidate parameter attributes differ",
    )
    decoded = defaultdict(dict)
    for number, name, value, distribution_raw in parameters:
        distribution = json.loads(distribution_raw)
        require(
            name not in decoded[number]
            and distribution["name"]
            in {"CategoricalDistribution", "FloatDistribution", "IntDistribution"},
            "duplicate or unsupported SQLite parameter distribution",
        )
        if distribution["name"] == "CategoricalDistribution":
            choices = distribution["attributes"]["choices"]
            require(
                float(value).is_integer() and 0 <= int(value) < len(choices),
                "invalid categorical parameter index",
            )
            decoded[number][name] = choices[int(value)]
        else:
            decoded[number][name] = (
                int(value) if distribution["name"] == "IntDistribution" else float(value)
            )
    for number, value in enumerate(records):
        pair_number = [list(pair) for pair in contract["pairs"]].index(
            [value["route"], value["model"]]
        )
        expected_parameters = {
            "candidate_pair": f"pair_{pair_number:04d}",
            **{
                value["model"] + "." + name: parameter
                for name, parameter in value["params"].items()
            },
        }
        tree_equal(
            decoded[number],
            expected_parameters,
            "SQLite sampled pair/parameters differ from evaluated candidate",
        )
    require(
        objectives
        == [(v["search_trial"], v["score"]) for v in records if v["status"] == "completed"],
        "SQLite objective values differ from actual AP scores",
    )
    admission = search["candidate_fit_admission"]
    require(
        admission["policy"] == "descending_inner_cv_score_then_outer_train_fit_and_persistence"
        and admission["validation_labels"] == "outer_train_only"
        and admission["holdout_used"] is False,
        "candidate admission used holdout labels or changed selection policy",
    )
    rejected = set()
    for item in admission["rejected_candidates"]:
        matching = [
            v
            for v in records
            if v["status"] == "completed"
            and all(v[k] == item[k] for k in ["route", "model", "params"])
        ]
        require(
            matching
            and item["reason"]
            and item["stage"]
            in {
                "estimator_construction",
                "outer_train_fit",
                "training_probe_persistence",
                "development_oof_reconstruction",
            }
            and all(v["score"] == item["inner_cv_score"] for v in matching),
            "unproven candidate rejection",
        )
        rejected.add(object_sha([item["route"], item["model"], item["params"]], compact=False))
    key = object_sha(
        [result["selected_route"], result["selected_model"], result["selected_params"]],
        compact=False,
    )
    eligible = [
        v
        for v in records
        if v["status"] == "completed"
        and object_sha([v["route"], v["model"], v["params"]], compact=False) not in rejected
    ]
    winners = [
        v
        for v in eligible
        if all(v[k] == result["selected_" + k] for k in ["route", "model", "params"])
    ]
    require(
        winners
        and admission["selected_candidate_key"] == key
        and winners[0]["score"] == max(v["score"] for v in eligible),
        "selected classifier is not highest admitted AP candidate",
    )
    close(
        winners[0]["score"],
        result["inner_cv"]["pr_auc"],
        "retained OOF AP differs from its selected objective",
    )
    return dict(attempts=128, pairs=88, objective="pr_auc_average_precision", sqlite_verified=True)


def selected_bundle(run, bound):
    """Verify opaque original evaluation/refit artifacts and source/fold/input signatures."""
    run, source, task = Path(run), bound["source"], bound["task"]
    verify_stage_contract(run, bound)
    path = run / "results/full" / (TASK + "_common.json")
    result = read(path)
    require(
        result["status"] == result["refit_status"] == "completed"
        and result["deployment_ready"] is True
        and result["task_id"] == TASK
        and result["actual_inner_folds"] == result["requested_inner_folds"] == 5,
        "selected classifier/full29 refit not completed",
    )
    required = {
        "model",
        "refit_model",
        "oof",
        "holdout",
        "model_contract",
        "input_contract",
        "outer_split",
    }
    require(
        required <= set(result["artifacts"])
        and all(result["artifacts"][name] in result["artifact_hashes"] for name in required),
        "evaluation/refit/probability/source artifacts missing SHA bindings",
    )
    for name, pin in result["artifact_hashes"].items():
        require(
            sha(path_at(run, name)) == pin["sha256"], "opaque original learned artifact changed"
        )
    paths = {name: path_at(run, name_path) for name, name_path in result["artifacts"].items()}
    require(
        paths["model"] != paths["refit_model"],
        "outer-train evaluation model cannot alias all29 refit",
    )
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    ids = [r["record_id"] for r in source]
    require(
        contract["task"] == inputs["task"] == task
        and contract["common_record_ids"] == ids
        and contract["model_fit_scope"] == "outer_train_only"
        and contract["holdout_not_used_for_selection"] is True
        and contract["optional_refit_fit_scope"] == "all_common_after_evaluation"
        and inputs["config"]["refit_full"] is True
        and inputs["implementation_sha256"]
        == sha(run / "scripts/peptide_modeling_train.py")
        == TRAIN_SHA,
        "original task/refit/membership/trainer contract changed",
    )
    require(
        contract["refit_status"] == result["refit_status"] == "completed"
        and contract["deployment_ready"] is True
        and contract["refit_attempts"] == result["refit_attempts"]
        and 1 <= len(result["refit_attempts"]) <= 2
        and result["refit_attempts"][-1]["status"] == "completed",
        "actual all29 refit/persistence attempt evidence missing or inconsistent",
    )
    for attempt in result["refit_attempts"]:
        require(
            all(
                name.endswith("max_iter") and type(value) is int and value > 0
                for name, value in attempt["solver_iteration_overrides"].items()
            ),
            "all29 refit changed a statistical parameter instead of solver iterations",
        )
    require(
        contract["versions"] == inputs["versions"]
        and inputs["versions"]["numpy"] == bound["plans"]["runtime"]["numpy"]
        and inputs["versions"]["scikit_learn"] == bound["plans"]["runtime"]["sklearn"],
        "actual selected classifier runtime differs from frozen node plan",
    )
    require(
        all(
            contract["selected_" + key] == result["selected_" + key]
            for key in ["route", "model", "params"]
        ),
        "selected classifier metadata differs from model contract",
    )
    settings = inputs["config"]
    require(
        set(settings["models"]) == MODELS
        and set(settings["routes"]) == set(bound["features"]["manifest"]["routes"])
        and settings["search_backend"] == "optuna"
        and settings["search_budget"] == 128
        and settings["inner_folds"] == 5
        and settings["n_jobs"] == 1
        and settings["seed"] in SEEDS,
        "actual eight-model/fivefold/128-trial/one-CPU settings changed",
    )
    plan = next(s for s in bound["plans"]["seeds"] if s["seed"] == settings["seed"])
    outer = read(paths["outer_split"])
    require(
        outer == plan["outer_split"], "actual training outer split differs from prefit node plan"
    )
    signature_value = dict(
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
        object_sha(signature_value, compact=False)
        == inputs["input_hash"]
        == result["input_hash"]
        == contract["input_hash"],
        "original selected training input signature changed",
    )
    all_designs = {}
    for route in settings["routes"]:
        matrix = design(bound["features"], source, route)
        all_designs[route] = matrix
        evidence = inputs["feature_hashes"][route]
        require(
            evidence["names"] == matrix["names"]
            and evidence["matrix_sha256"] == matrix_signature(matrix, ids)["bytes_sha256"]
            and evidence["mask_sha256"] == signature([True] * 29, "bool")["bytes_sha256"],
            "original all-route chemical feature schema/values/mask changed",
        )
    selected = all_designs[result["selected_route"]]
    require(contract["feature_names"] == selected["names"], "selected-route feature schema differs")
    trials = verify_trials(run, result, settings)
    return dict(
        result=result,
        result_path=path,
        paths=paths,
        contract=contract,
        inputs=inputs,
        plan=plan,
        seed=settings["seed"],
        matrix=selected,
        all_designs=all_designs,
        original_search=trials,
    )


def verify_named_fit(
    run,
    source,
    matrix,
    folder,
    train_ids,
    predict_ids,
    scope,
    fold,
    seed,
    policy,
    binding,
    require_float64=False,
):
    folder = path_at(run, str(Path(folder)))
    by_id = {r["record_id"]: r for r in source}
    receipt = read(folder / "fit_receipt.json")
    expected = dict(
        scope=scope,
        fold=fold,
        seed=seed,
        model_policy=policy,
        train_record_ids=train_ids,
        predict_record_ids=predict_ids,
        train_X=matrix_signature(matrix, train_ids),
        predict_X=matrix_signature(matrix, predict_ids),
        train_y=signature([int(by_id[i]["target"]) for i in train_ids]),
        source_binding=binding,
        implementation_sha256=PRODUCER_SHA,
        diagnostic_policy_sha256=POLICY_SHA,
        parameter_search_performed=False,
    )
    require(
        receipt["schema"] == "native_binary_named_diagnostic_fit_v1"
        and receipt["status"] == "completed"
        and receipt["input_contract"] == read(folder / "fit_input_contract.json") == expected
        and receipt["input_signature"] == object_sha(expected)
        and receipt["model_reload_verified_on_training_host"] is True
        and receipt["selected_model_search_repeated"] is False
        and receipt["learning_phase"] is True
        and receipt["host"] == "node23",
        "named diagnostic fit source/member/parameter/provenance changed",
    )
    require(
        set(receipt["artifacts"]) == {"model.joblib", "predictions.tsv", "fit_input_contract.json"},
        "named diagnostic fit needs exact opaque-model/probability/input artifacts",
    )
    for name, pin in receipt["artifacts"].items():
        require(
            sha(path_at(folder, name)) == pin,
            "diagnostic opaque model/input/prediction artifact changed",
        )
    require(
        receipt["fit_input_dtype"] == receipt["predict_input_dtype"] == matrix["dtype"],
        "diagnostic input dtype changed",
    )
    if require_float64:
        require(
            receipt["fitted_coefficient_dtype"] == matrix["dtype"] == "float64",
            "fixed Logistic must start with float64 and keep float64 coefficients",
        )
    require(
        set(train_ids).isdisjoint(predict_ids)
        and {by_id[i]["split_group"] for i in train_ids}.isdisjoint(
            {by_id[i]["split_group"] for i in predict_ids}
        ),
        "diagnostic learner predicts its own training record/parent",
    )
    p = prediction_values(
        source,
        folder / "predictions.tsv",
        predict_ids,
        {i: -1 if fold is None else fold for i in predict_ids},
    )
    require(
        receipt["prediction_signature"] == signature(p),
        "diagnostic probability bytes differ from actual fit receipt",
    )
    if require_float64:
        identical = defaultdict(list)
        for rid, probability in zip(predict_ids, p, strict=True):
            identical[matrix["rows"][rid]].append(probability)
        require(
            all(
                all(math.isclose(v, values[0], rel_tol=1e-10, abs_tol=1e-12) for v in values)
                for values in identical.values()
            ),
            "same float64 complete input has inconsistent fixed probabilities",
        )
    return p, receipt


def verify_diagnostics(run, bound, bundle):
    source, plan, seed, policy = bound["source"], bundle["plan"], bundle["seed"], bound["policy"]
    run = Path(run)
    folder = run / DIAGNOSTIC
    learning_path = folder / "learning_receipt.json"
    learning = read(learning_path)
    require(
        learning["schema"] == "native_binary_diagnostic_learning_receipt_v1"
        and learning["status"] == "completed"
        and learning["seed"] == seed
        and learning["implementation_sha256"] == PRODUCER_SHA
        and learning["diagnostic_policy"] == policy
        and learning["source_result_sha256"] == sha(bundle["result_path"])
        and (
            learning["diagnostic_learning_fits"],
            learning["full29_fixed_selected_parameter_OOF_fits"],
            learning["fixed_Logistic_float64_fits"],
            learning["scalar_Platt_calibrator_fits"],
        )
        == (13, 5, 6, 2)
        and learning["model_selection_repeated"] is False
        and learning["report_stage_fits"] == learning["report_stage_model_loads"] == 0
        and learning["original_selected_model_and_prediction_files_unchanged"] is True,
        "actual diagnostic-learning13-fit roles/source/report-only contract changed",
    )
    for relative, pin in learning["artifacts"].items():
        require(
            sha(path_at(folder, relative)) == pin, "persisted diagnostic-learning artifact changed"
        )
    existing = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(
        existing == set(learning["artifacts"]) | {"learning_receipt.json"},
        "unrecorded or missing diagnostic learning artifacts",
    )
    binding = dict(
        source_table_sha256=TABLE_SHA,
        source_result_sha256=sha(bundle["result_path"]),
        selected_input_hash=bundle["result"]["input_hash"],
        node_plan_sha256=PLAN_SHA,
    )
    require(
        learning["source_binding"] == binding, "diagnostics bind a different source/result/plan"
    )
    by_id = {r["record_id"]: r for r in source}
    train_ids, held_ids = plan["outer_train_record_ids"], plan["outer_holdout_record_ids"]
    train = [by_id[i] for i in train_ids]
    all_ids = [r["record_id"] for r in source]
    inner_map = {rid: f["fold"] for f in plan["inner_folds"] for rid in f["validation_record_ids"]}
    raw_inner = prediction_values(source, bundle["paths"]["oof"], train_ids, inner_map)
    raw_held = prediction_values(
        source, bundle["paths"]["holdout"], held_ids, {i: -1 for i in held_ids}
    )
    evaluation_scope = "evaluation_outer_train_inner_OOF"
    cal, calibrated_inner = verify_calibration(
        folder / "evaluation_calibration.json", train, raw_inner, seed, evaluation_scope, policy
    )
    calibrated_held = apply_platt(cal, raw_held, policy)
    raw_threshold = threshold_choice(train, raw_inner, evaluation_scope, policy)
    tree_equal(
        read(folder / "evaluation_raw_threshold.json"),
        raw_threshold,
        "raw evaluation threshold escaped training OOF",
    )
    inner = enriched(
        source,
        folder / "selected_inner_evaluation.tsv",
        train_ids,
        inner_map,
        "inner_oof_calibration_fit",
        raw_inner,
        calibrated_inner,
    )
    held = enriched(
        source,
        folder / "selected_outer_holdout.tsv",
        held_ids,
        {i: -1 for i in held_ids},
        "outer_holdout",
        raw_held,
        calibrated_held,
    )
    calibrated_threshold = threshold_choice(
        train, [r["calibrated_probability"] for r in inner], evaluation_scope, policy
    )
    tree_equal(
        read(folder / "evaluation_threshold.json"),
        calibrated_threshold,
        "calibrated evaluation threshold escaped training OOF",
    )
    selected_policy = dict(
        model=bundle["result"]["selected_model"],
        params=bundle["result"]["selected_params"],
        route=bundle["result"]["selected_route"],
        new_parameter_search=False,
    )
    require(
        learning["selected_params_frozen"] == selected_policy,
        "deployment parameters were reselected",
    )
    full, full_trace = {}, []
    for fold in plan["full29_oof_folds"]:
        values, receipt = verify_named_fit(
            run,
            source,
            bundle["matrix"],
            folder / f"full29_selected_parameter_oof/fold_{fold['fold']}",
            fold["train_record_ids"],
            fold["validation_record_ids"],
            "full29_deployment_selected_parameter_OOF",
            fold["fold"],
            seed,
            selected_policy,
            binding,
        )
        full_trace.append(receipt)
        full.update(
            {
                rid: (p, fold["fold"])
                for rid, p in zip(fold["validation_record_ids"], values, strict=True)
            }
        )
    require(
        set(full) == set(all_ids) and learning["selected_parameter_learning_trace"] == full_trace,
        "full29 selected-parameter fivefold trace/membership changed",
    )
    full_raw = [full[i][0] for i in all_ids]
    deploy_scope = "deployment_full29_fixed_parameter_OOF"
    deploy_cal, calibrated_full = verify_calibration(
        folder / "deployment_calibration.json", source, full_raw, seed, deploy_scope, policy
    )
    full_predictions = enriched(
        source,
        folder / "full29_deployment_calibration_oof.tsv",
        all_ids,
        {i: full[i][1] for i in all_ids},
        "deployment_calibration_fit_not_independent_evaluation",
        full_raw,
        calibrated_full,
    )
    deploy_threshold = threshold_choice(
        source, [r["calibrated_probability"] for r in full_predictions], deploy_scope, policy
    )
    tree_equal(
        read(folder / "deployment_threshold.json"),
        deploy_threshold,
        "deployment threshold escaped full29 fixed-parameter OOF",
    )
    fixed_matrix = design(bound["features"], source, "structure_methods_all", "float64")
    fixed_values, fixed_trace = {}, []
    for fold in plan["inner_folds"]:
        values, receipt = verify_named_fit(
            run,
            source,
            fixed_matrix,
            folder / f"fixed_Logistic_float64/fold_{fold['fold']}",
            fold["train_record_ids"],
            fold["validation_record_ids"],
            "fixed_Logistic_C1_molecule_only_innerOOF",
            fold["fold"],
            seed,
            policy["fixed_logistic"],
            binding,
            require_float64=True,
        )
        fixed_trace.append(receipt)
        fixed_values.update(dict(zip(fold["validation_record_ids"], values, strict=True)))
    fixed_held, receipt = verify_named_fit(
        run,
        source,
        fixed_matrix,
        folder / "fixed_Logistic_float64/outer_train",
        train_ids,
        held_ids,
        "fixed_Logistic_C1_outer_holdout",
        None,
        seed,
        policy["fixed_logistic"],
        binding,
        require_float64=True,
    )
    fixed_trace.append(receipt)
    require(
        learning["fixed_control_learning_trace"] == fixed_trace
        and set(fixed_values) == set(train_ids),
        "single-arm six-fit fixed Logistic trace/membership changed",
    )
    fixed_inner = [fixed_values[i] for i in train_ids]
    fixed_threshold = threshold_choice(
        train, fixed_inner, "fixed_Logistic_outer_train_inner_OOF", policy
    )
    tree_equal(
        read(folder / "fixed_Logistic_threshold.json"),
        fixed_threshold,
        "fixed threshold used heldout data",
    )
    fixed_oof = enriched(
        source,
        folder / "fixed_Logistic_inner.tsv",
        train_ids,
        inner_map,
        "inner_oof_fixed_control",
        fixed_inner,
    )
    fixed_test = enriched(
        source,
        folder / "fixed_Logistic_holdout.tsv",
        held_ids,
        {i: -1 for i in held_ids},
        "outer_holdout_fixed_control",
        fixed_held,
    )
    return dict(
        learning=learning,
        learning_path=learning_path,
        inner=inner,
        held=held,
        full=full_predictions,
        cal=cal,
        raw_threshold=raw_threshold,
        threshold=calibrated_threshold,
        deployment_cal=deploy_cal,
        deployment_threshold=deploy_threshold,
        fixed_inner=fixed_oof,
        fixed_held=fixed_test,
        fixed_threshold=fixed_threshold,
        actual_named_model_fit_receipts=len(full_trace) + len(fixed_trace),
        actual_Platt_fit_artifacts=2,
    )


def support_diagnostics(source, predictions, plan, role):
    by_id = {r["record_id"]: r for r in source}
    folds = {f["fold"]: f for f in plan["inner_folds"]}
    result = []
    for row in predictions:
        original = by_id[row["record_id"]]
        trained = (
            plan["outer_train_record_ids"]
            if role == "outer_holdout"
            else folds[int(row["fold"])]["train_record_ids"]
        )
        require(
            original["split_group"] not in {by_id[i]["split_group"] for i in trained},
            "reported support leaked parent",
        )
        result.append(
            dict(
                record_id=row["record_id"],
                evaluation_role=role,
                fold=int(row["fold"]),
                unknown_material=original["assay_material_literal"] == "not_reported",
                unknown_species=original["assay_species_literal"] == "not_reported",
                material_literal_seen_in_training=original["assay_material_literal"]
                in {by_id[i]["assay_material_literal"] for i in trained},
                species_literal_seen_in_training=original["assay_species_literal"]
                in {by_id[i]["assay_species_literal"] for i in trained},
                unknown_literal_category_is_not_biological_support=True,
                uniform_physiological_threshold_or_species_effect_validated=False,
                source_context_is_prediction_covariate=False,
            )
        )
    return result


def expected_seed_report(bound, bundle, diagnostics):
    source, plan = bound["source"], bundle["plan"]
    result, paths, contract = bundle["result"], bundle["paths"], bundle["contract"]
    d, by_id = diagnostics, {r["record_id"]: r for r in source}
    raw_threshold, threshold = d["raw_threshold"]["threshold"], d["threshold"]["threshold"]
    original = {}
    for name, predictions, result_name in [
        ("inner_oof", d["inner"], "inner_cv"),
        ("outer_holdout", d["held"], "holdout"),
    ]:
        metrics = probability_metrics(
            [int(by_id[r["record_id"]]["target"]) for r in predictions],
            [r["prediction"] for r in predictions],
            [1.0] * len(predictions),
            0.5,
        )
        for metric in ["pr_auc", "roc_auc", "mcc", "brier"]:
            close(
                result[result_name][metric],
                metrics[metric],
                "original heldout/OOF classifier metric changed",
            )
        original[name] = metrics
    strata = {}
    for field in [
        "source_subset",
        "identity_tier",
        "assay_material_literal",
        "assay_species_literal",
    ]:
        strata[field] = {
            value: role_metrics(
                source,
                [r for r in d["held"] if by_id[r["record_id"]][field] == value],
                raw_threshold,
                threshold,
            )
            for value in sorted({r[field] for r in source})
        }
    return dict(
        seed=bundle["seed"],
        actual_outer_seed=plan["actual_outer_seed"],
        status="verified",
        task_id=TASK,
        quantity_id=QUANTITY,
        source_annotation_only=True,
        primary_sample_identity_verified=False,
        selected_route=result["selected_route"],
        selected_model=result["selected_model"],
        source_result_sha256=sha(bundle["result_path"]),
        selected_input_hash=result["input_hash"],
        source_table_sha256=TABLE_SHA,
        node_plan_sha256=PLAN_SHA,
        original_selected_model_sha256=sha(paths["model"]),
        original_selected_refit_sha256=sha(paths["refit_model"]),
        original_oof_sha256=sha(paths["oof"]),
        original_holdout_sha256=sha(paths["holdout"]),
        base_probability_calibration_role=contract["probability_calibration"],
        original_selected_metrics_at_default_0_5=original,
        evaluation_calibration_model=d["cal"],
        evaluation_threshold=d["threshold"],
        raw_OOF_threshold=d["raw_threshold"],
        evaluation_by_role=dict(
            inner_oof=dict(
                metrics=role_metrics(source, d["inner"], raw_threshold, threshold),
                role="model_selection_then_calibration_threshold_fit_diagnostic_not_independent",
                predictions=d["inner"],
                source_support=support_diagnostics(source, d["inner"], plan, "inner_oof"),
            ),
            outer_holdout=dict(
                metrics=role_metrics(source, d["held"], raw_threshold, threshold),
                role="independent_outer_train_model_plus_innerOOF_calibration_evaluation",
                predictions=d["held"],
                source_support=support_diagnostics(source, d["held"], plan, "outer_holdout"),
                strata=strata,
                scores_apply_to_full29_refit=False,
            ),
        ),
        deployment_full29=dict(
            calibration_model=d["deployment_cal"],
            threshold=d["deployment_threshold"],
            metrics=role_metrics(source, d["full"], 0.5, d["deployment_threshold"]["threshold"]),
            predictions=d["full"],
            role="deployment_calibration_fit_diagnostic_not_independent_evaluation",
            selected_parameter_OOF_fits=5,
            parameter_search_repeated=False,
            all29_refit_training_record_ids=contract["common_record_ids"],
            original_outer_holdout_now_training_not_independent=True,
        ),
        fixed_molecule_only_Logistic=dict(
            policy=bound["policy"]["fixed_logistic"],
            threshold=d["fixed_threshold"],
            inner_oof=role_metrics(source, d["fixed_inner"], d["fixed_threshold"]["threshold"]),
            outer_holdout=role_metrics(source, d["fixed_held"], d["fixed_threshold"]["threshold"]),
            predictions=dict(inner_oof=d["fixed_inner"], outer_holdout=d["fixed_held"]),
            fits=6,
            input_dtype="float64",
            coefficient_dtype="float64",
            context_designs_exist=False,
            context_control_omission=bound["policy"]["context_control_omission"],
        ),
        diagnostic_learning_fits=13,
        report_stage_fits=0,
        report_stage_model_loads=0,
        warning=bound["proof"]["warning"],
    )


def verify_diagnostic_archive(run, durable, descriptor, learning):
    archive = descriptor["archive"]
    path = path_at(run, archive["path"])
    mirror = path_at(durable, archive["durable_path"])
    require(
        sha(path) == archive["sha256"] == sha(mirror) == archive["durable_sha256"],
        "diagnostic archive/durable mirror changed",
    )
    folder = Path(run) / DIAGNOSTIC
    expected = {
        **learning["artifacts"],
        "learning_receipt.json": sha(folder / "learning_receipt.json"),
    }
    observed = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            require(
                member.isfile()
                and not member.issym()
                and not member.islnk()
                and not name.is_absolute()
                and ".." not in name.parts
                and member.name not in observed,
                "diagnostic archive contains links, duplicate or unsafe members",
            )
            with tar.extractfile(member) as stream:
                observed[member.name] = hashlib.sha256(stream.read()).hexdigest()
    require(
        observed == expected,
        "durable diagnostic archive omits/changes actual model, trace, input or probability bytes",
    )


def verify_seed(entry, round_entry, seed, task, api):
    """Strict per-seed evidence; never call the shared check_seed recursively."""
    bound = verify_source(entry, task, api)
    run = path_at(entry["output_root"], round_entry["run_dir"])
    require(
        sha(run / "task_manifest.json") == entry["manifest_sha256"]
        and sha(run / "config.json") == round_entry["config_sha256"]
        and read(run / "config.json")["training"]["seed"] == seed,
        "actual seed input/config differs from frozen registry",
    )
    verification_path = run / "verification.json"
    verification = read(verification_path)
    require(
        round_entry["status"] == "completed"
        and round_entry.get("exit_code", 0) == 0
        and round_entry["audit"]["method_model_coverage_met"] is True
        and not round_entry["audit"]["missing_completed_task_ids"]
        and verification["status"] == "verified"
        and sha(verification_path) == round_entry["audit"]["verification_sha256"]
        and verification["source_task_manifest_sha256"] == sha(run / "task_manifest.json"),
        "actual completed seed/same-host verification receipt changed",
    )
    actual_bound = native_source(run)
    require(
        actual_bound["task"] == bound["task"] and actual_bound["source"] == bound["source"],
        "seed native source changed",
    )
    bundle = selected_bundle(run, actual_bound)
    require(bundle["seed"] == seed, "actual learned seed differs")
    result_relative = bundle["result_path"].relative_to(run).as_posix()
    proofs = [p for p in verification["results"] if p.get("result_file") == result_relative]
    require(
        len(proofs) == 1
        and proofs[0].get("artifact_hashes_verified", 0) > 0
        and verification["assertions"]["all_checks_passed"] is True,
        "native selected result is absent from actual same-host holdout verification",
    )
    api.verify_standard_result(run, bundle["result"], task, "group_holdout_inner5", proofs[0])
    diagnostic = verify_diagnostics(run, actual_bound, bundle)
    learning_descriptor = round_entry["native_binary_diagnostic_learning"]
    require(
        learning_descriptor["status"] == "completed"
        and learning_descriptor["fit_count"] == 13
        and learning_descriptor["implementation_sha256"] == PRODUCER_SHA
        and path_at(run, learning_descriptor["receipt_path"])
        == diagnostic["learning_path"].resolve()
        and sha(diagnostic["learning_path"]) == learning_descriptor["receipt_sha256"],
        "actual diagnostic13-fit journal binding changed",
    )
    cfg = read(entry["config_path"])
    durable = Path(cfg["durable_root"])
    verify_diagnostic_archive(run, durable, learning_descriptor, diagnostic["learning"])
    for name in [
        "model",
        "refit_model",
        "oof",
        "holdout",
        "model_contract",
        "input_contract",
        "outer_split",
    ]:
        original = bundle["paths"][name]
        # The frozen full-task supervisor publishes these exact artifacts under live/.
        mirror = path_at(durable / run.name / "live", original.relative_to(run))
        require(
            sha(original) == sha(mirror),
            "original model/refit/prediction/contract lacks matching durable bytes",
        )
    expected = expected_seed_report(actual_bound, bundle, diagnostic)
    return dict(
        status="completed_verified",
        task_id=TASK,
        actual_inner_folds=5,
        actual_full29_OOF_folds=5,
        actual_backend="optuna",
        selection_objective="pr_auc_average_precision",
        diagnostic_learning_fits=13,
        independent_outer_holdout=True,
        all29_refit_independently_scored=False,
        original32_participation="22/32",
        source_proof_verified=True,
        source_annotation_only=True,
        opaque_models_never_loaded_by_auditor=True,
        expected_report=expected,
    )


def verify_reporting(entry, journal, task, api):
    bound = verify_source(entry, task, api)
    owner = entry["owned_process_identity"]
    require(
        api.identity_state(owner, api.process_identity(owner["pid"])) in {"exited", "pid_reused"},
        "native binary controller is live or release is unverified",
    )
    proc_root = Path(getattr(api, "native_binary_proc_root", "/proc"))
    require(proc_root.is_dir(), "Linux process inventory required for final completion")
    prefixes = [
        str(Path(entry[key])).replace("\\", "/").rstrip("/") + "/"
        for key in ["template", "output_root"]
    ]
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        actual = api.process_identity(int(directory.name))
        if actual is not None and actual.get("state") != "Z":
            require(
                not any(
                    str(arg).replace("\\", "/").startswith(prefix)
                    for arg in actual.get("argv", [])
                    for prefix in prefixes
                ),
                "live process still owns native binary artifacts",
            )
    require(
        journal["status"] == "completed"
        and journal["required_seeds"] == list(SEEDS)
        and journal.get("additional_twenty_hour_loop") is False
        and len(journal["rounds"]) == 3
        and {r["seed"] for r in journal["rounds"]} == set(SEEDS),
        "three completed actual seeds required",
    )
    receipt = journal.get("molecular_reporting")
    require(
        isinstance(receipt, dict)
        and receipt.get("status") == "verified"
        and receipt.get("implementation_sha256") == PRODUCER_SHA
        and receipt.get("diagnostic_policy_sha256") == POLICY_SHA
        and receipt.get("diagnostic_learning_fits") == 39
        and receipt.get("report_stage_model_fits") == receipt.get("report_stage_model_loads") == 0
        and receipt.get("source_native_binary_original_observation_participation") == "22/32"
        and receipt.get("separate_full29_refits_verified") == 3,
        "completed controller alone cannot replace39-fit calibrated molecular report evidence",
    )
    report_path = path_at(entry["output_root"], receipt["path"])
    cfg = read(entry["config_path"])
    durable_path = path_at(cfg["durable_root"], receipt["durable_path"])
    require(
        report_path.name == REPORT_FILE
        and sha(report_path) == receipt["sha256"] == sha(durable_path) == receipt["durable_sha256"],
        "final report/durable receipt changed",
    )
    require(
        sha(entry["journal_path"]) == sha(Path(cfg["durable_root"]) / "campaign_status.json"),
        "final original campaign journal lacks exact durable mirror",
    )
    closed_contract = verify_runtime_source_closure(entry, journal, bound)
    payload = read(report_path)
    required = dict(
        schema=REPORT_SCHEMA,
        status="verified",
        task_id=TASK,
        quantity_id=QUANTITY,
        query_mode=QUERY,
        required_seeds=list(SEEDS),
        records=29,
        positive=7,
        negative=22,
        known_parent_groups=19,
        class_parent_groups={"0": 13, "1": 7},
        independent_methods=10,
        diagnostic_policy=bound["policy"],
        diagnostic_learning_fits=39,
        full29_fixed_selected_parameter_OOF_fits=15,
        fixed_Logistic_float64_fits=18,
        scalar_Platt_calibration_fits=6,
        report_stage_model_fits=0,
        report_stage_model_loads=0,
        original32_observation_participation=bound["descriptor"],
        old32_negative_values_unchanged=True,
        all_original32_records_covered=False,
        source_annotation_only=True,
        calibrated_source_annotation_probability_is_not_clinical_hemolysis_risk=True,
        source_context_covariates=False,
        common_physiological_threshold_validated=False,
        primary_sample_identity_verified=False,
        all29_refits_independently_scored=False,
        source_parent_union_is_not_homology_or_scaffold_isolation=True,
        warning=bound["proof"]["warning"],
    )
    for key, value in required.items():
        tree_equal(payload.get(key), value, "final report scientifically misstates " + key)
    require(
        set(payload) == set(required) | {"model_families", "reports"},
        "final report contains an undeclared score or scientific claim",
    )
    require(
        set(payload["model_families"]) == MODELS and len(payload["model_families"]) == 8,
        "reported eight actual model families required",
    )
    reports = {r["seed"]: r for r in payload["reports"]}
    require(
        len(reports) == 3 and set(reports) == set(SEEDS), "three complete per-seed reports required"
    )
    for round_entry in journal["rounds"]:
        run = path_at(entry["output_root"], round_entry["run_dir"])
        require(
            read(run / "campaign_input_contract.json")["template_contract"] == closed_contract,
            "round source closure differs from actual before-producer controller contract",
        )
        evidence = verify_seed(entry, round_entry, round_entry["seed"], task, api)
        tree_equal(
            reports[round_entry["seed"]],
            evidence["expected_report"],
            "native per-seed metric/calibration/role report changed",
        )
    return dict(
        status="verified",
        task_id=TASK,
        actual_seeds=3,
        diagnostic_learning_fits=39,
        full29_refit_training_seeds=3,
        original32_participation="22/32",
        source_annotation_only=True,
        independent_full29_scores_available=False,
        clinical_toxicity_validated=False,
    )


def coverage(entry, journal, task, state, api):
    bound = verify_source(entry, task, api)
    trained = state == "trained_three_seeds"
    if trained:
        verify_reporting(entry, journal, task, api)
    return [
        dict(
            bound["descriptor"],
            canonical_task_id=TASK,
            state="qualified_source_annotation_subset_trained"
            if trained
            else "qualified_source_annotation_subset_pending",
            coverage_kind="source_native_annotation_qualified_original_subset_participation",
            covered_original_records=22,
            total_original_records=32,
            new_native_affirmative_records=7,
            excluded_original_records=10,
            source_proof_verified=True,
            required_seed_final_refit_training_verified=3 if trained else 0,
            all_original_records_covered=False,
            independent_raw_endpoint_predictor=False,
            physiological_or_clinical_toxicity_predictor=False,
        )
    ]
