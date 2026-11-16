"""Qualified human-IV total CL preparation, context, and molecular diagnostics.

Source-reported nominal clearance, not an intrinsic molecular or clinical model.
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
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_conditional_cmax as base_context
from peptide_modeling_add_smiles_methods import rdkit_tools
from peptide_modeling_clint_features import STRUCTURE_GROUPS, atomic_array, extract_all
from peptide_modeling_features import _verify_artifacts, load_feature_route
from peptide_modeling_mhc_mass_bridge import check_matrix, graph, safe_path, write_tsv
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/cl_absolute_primary_recovery_20261114_v1"
AUDIT = ROOT / "data/conditional_pk_feasibility_20261114_v1/audit.json"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
BASE = PARENT
OUTPUT = ROOT / "data/peptide_modeling_conditional_cl_followup_20261114_v1"
DB = ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite"
SCHEMA = "conditional_total_cl_minimum_context_v1"
VERSION = "conditional_total_cl_complete_graph_bank_v1"
ESTIMAND = "human_iv_source_reported_total_systemic_clearance_given_known_matrix_and_dose"
MEASURE = ESTIMAND
WARNING = (
    "Human IV source-reported nominal total systemic CL, internally tuned on these same "
    "five molecular-group folds. Scores are selection-optimistic; no independent holdout. "
    "Five molecular families yield one LOGO partition across seeds. Matrix is confounded "
    "with teriparatide/study; species and route are constants. Regimen, sex, cohort, "
    "analytical method and bodyweight heterogeneity are not fully controlled. "
    "No clinical, cross-species, cross-route or cross-matrix generalization is established."
)
SEEDS = (42, 2026, 3407)
ONTOLOGY = base_context.ONTOLOGY
DOSE_UNITS = base_context.DOSE_UNITS
FEATURE_NAMES = base_context.FEATURE_NAMES
CONTEXT_KEYS = base_context.CONTEXT_KEYS
sha = base_context.sha
object_hash = base_context.object_hash
signature = object_hash


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


def encode_context(context):
    result = base_context.encode_context(context)
    if context["species"] != "human" or context["route"] != "intravenous":
        raise ValueError("This estimand supports known human IV observations only")
    return result


def extract_reference_graph(smiles, key):
    """Match the frozen single-BLAS-thread descriptor calculation exactly."""
    with threadpool_limits(limits=1):
        return extract_all(smiles, key)


def context_contract(records, audit_sha):
    contexts = {r["record_id"]: r["conditional_core_context"] for r in records}
    if len(contexts) != len(records):
        raise ValueError("Duplicate source record IDs")
    for value in contexts.values():
        encode_context(value)
    value = {
        "schema": SCHEMA,
        "encoding_dependency_sha256": sha(base_context.__file__),
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
    if not isinstance(contract, dict):
        raise ValueError("Context contract must be an object")
    unsigned = {k: v for k, v in contract.items() if k != "sha256"}
    if object_hash(unsigned) != contract.get("sha256"):
        raise ValueError("External context contract hash mismatch")
    if (
        contract.get("schema") != SCHEMA
        or contract.get("encoding_dependency_sha256") != sha(base_context.__file__)
        or contract.get("estimand") != ESTIMAND
        or contract.get("feature_names") != FEATURE_NAMES
        or contract.get("ontology") != ONTOLOGY
        or contract.get("dose_units") != json.loads(json.dumps(DOSE_UNITS))
        or contract.get("inference_requires_all_known_context") is not True
        or contract.get("chemical_method_count_increment") != 0
        or contract.get("warning") != WARNING
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
    implementation = model_contract.get("external_context", {})
    if (
        implementation.get("implementation_sha256") != sha(__file__)
        or implementation.get("schema") != SCHEMA
        or implementation.get("sha256") != contract["sha256"]
    ):
        raise ValueError("Conditional CL inference implementation/contract hash differs")
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
            "compound_equal_weight_rmse": float(
                np.sqrt(
                    np.mean(
                        [
                            np.mean((y[groups == group] - predictions[groups == group]) ** 2)
                            for group in set(groups)
                        ]
                    )
                )
            ),
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


def build_features(task_rows, source_proof, protected, output, base_run=BASE):
    output, base_run = Path(output).resolve(), Path(base_run).resolve()
    if output == base_run or output in base_run.parents or base_run in output.parents:
        raise ValueError("Feature output must be separate from immutable parent bank")
    base_path = base_run / "features/manifest.json"
    pointer_sha = sha(base_path)
    base = read(base_path)
    _verify_artifacts(base)
    registry_path = base_run / "representation_methods.json"
    registry = read(registry_path)
    methods = [
        m for m in registry["methods"] if m["source_kind"] == "structure" and m["independent"]
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten independent complete-graph methods are required")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if base["configuration"]["rdkit_version"] != rdkit_tools()["version"]:
        raise ValueError("RDKit differs from frozen parent algorithms")
    cache_source = Path(base["cache_dir"])
    old_index = {
        r["key"]: int(r["row_idx"]) for r in table(cache_source / groups["morgan"]["index"])
    }
    old_graphs = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in table(cache_source / "structure_source_smiles.tsv")
    }
    graph_map = {}
    for row in task_rows:
        key, smiles = row["structure_key"], row["canonical_isomeric_smiles"]
        if key in graph_map and graph_map[key] != smiles:
            raise ValueError("Source graph key has conflicting full structures")
        graph_map[key] = smiles
    graphs = [{"key": key, "smiles": graph_map[key]} for key in sorted(graph_map)]
    if len(graphs) != 5 or len(task_rows) != 7:
        raise ValueError("Expected seven observations and five complete graphs")
    reused = {r["key"] for r in graphs if r["key"] in old_index}
    new = [r for r in graphs if r["key"] not in reused]
    if len(reused) < 4 or len(new) > 1:
        raise ValueError("Only the exact teriparatide reference graph may need extraction")
    if any(
        old_graphs[key] != next(r["smiles"] for r in graphs if r["key"] == key) for key in reused
    ):
        raise ValueError("Parent full graph differs from original archive")
    config = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "implementation_dependencies_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                "peptide_modeling_mhc_mass_bridge.py",
                "peptide_modeling_clint_features.py",
                "peptide_modeling_expand_features.py",
                "peptide_modeling_add_methods.py",
                "peptide_modeling_add_smiles_methods.py",
                "peptide_modeling_features.py",
            )
        },
        "parent_fingerprint": base["input_fingerprint"],
        "parent_pointer_sha256": pointer_sha,
        "registry_sha256": sha(registry_path),
        "source_sha256": {str(p): h for p, h in protected.items()},
        "source_proof": source_proof,
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "feature_extraction_blas_threads": 1,
        "reused_algorithm_canary_policy": "all_columns_exact_equal_nan_no_tolerance",
        "inherited_algorithm_configuration": base["configuration"],
        "extracted_graphs": len(new),
        "reused_graphs": len(reused),
        "sequence_policy": "No sequence projection or ESM generated",
        "max_structure_atoms": 1000,
    }
    fingerprint = signature(config)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        old = read(cache / "manifest.json")
        _verify_artifacts(old)
        return old
    extracted = {r["key"]: extract_reference_graph(r["smiles"], r["key"]) for r in new}
    compatibility = {
        r["key"]: extract_reference_graph(r["smiles"], r["key"])
        for r in graphs
        if r["key"] in reused
    }
    for name, metadata in groups.items():
        before = np.load(safe_path(cache_source, metadata["matrix"]), mmap_mode="r")
        before_valid = np.load(safe_path(cache_source, metadata["valid"]), mmap_mode="r")
        for key, values in compatibility.items():
            recomputed = np.asarray(values[0][name], dtype=metadata["dtype"])
            if not values[1][name] or not np.array_equal(
                before[old_index[key]], recomputed, equal_nan=True
            ):
                raise ValueError("Current extractor differs from frozen reused vectors: " + name)
        matrix = np.asarray(
            [
                before[old_index[r["key"]]] if r["key"] in reused else extracted[r["key"]][0][name]
                for r in graphs
            ],
            dtype=metadata["dtype"],
        )
        valid = np.asarray(
            [
                before_valid[old_index[r["key"]]]
                if r["key"] in reused
                else extracted[r["key"]][1][name]
                for r in graphs
            ],
            dtype=bool,
        )
        metadata["shape"] = [len(graphs), metadata["shape"][1]]
        metadata["index"] = "structure_bank_index.tsv"
        if matrix.shape != tuple(metadata["shape"]):
            raise ValueError("Full graph feature shape differs: " + name)
        check_matrix(matrix, valid, metadata, name)
        for index, row in enumerate(graphs):
            if (
                row["key"] in reused
                and before[old_index[row["key"]]].tobytes() != matrix[index].tobytes()
            ):
                raise ValueError("Reused graph feature bytes changed: " + name)
        atomic_array(safe_path(cache, metadata["matrix"]), matrix)
        atomic_array(safe_path(cache, metadata["valid"]), valid)
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": r["key"], "row_idx": i} for i, r in enumerate(graphs)],
    )
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "canonical_isomeric_smiles"],
        [{"key": r["key"], "canonical_isomeric_smiles": r["smiles"]} for r in graphs],
    )
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {"record_id": r["record_id"], "structure_key": r["structure_key"], "sequence_key": ""}
            for r in task_rows
        ],
    )
    registry = {
        **copy.deepcopy(registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": MEASURE,
    }
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", config)
    write(
        cache / "new_graph_extraction_audit.json",
        [{"key": key, "errors": value[2]} for key, value in extracted.items()],
    )
    routes = {m["route"]: m["group_names"] for m in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": config,
        "record_index": "record_bank_index.tsv",
        "requested_records": 7,
        "structure_keys": 5,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "coverage": {name: {"representations": 5, "valid": 5, "invalid": 0} for name in groups},
        "scope": MEASURE,
    }
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    _verify_artifacts(base)
    if sha(base_path) != pointer_sha:
        raise ValueError("Parent feature pointer changed")
    if any(
        sha(Path(__file__).with_name(name)) != expected
        for name, expected in config["implementation_dependencies_sha256"].items()
    ):
        raise ValueError("Feature implementation changed during extraction")
    for method in methods:
        payload = load_feature_route(task_rows, manifest, method["route"])
        if not payload["valid"].all() or payload["X"].shape != (7, method["dimensions"]):
            raise ValueError("A method did not load all source observations")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        {
            "status": "verified",
            "records": 7,
            "graphs": 5,
            "reused_graphs": len(reused),
            "new_graphs": len(new),
            "feature_groups": len(groups),
            "independent_methods": 10,
            "fusion_dimensions": sum(m["shape"][1] for m in groups.values()),
            "artifact_hashes_verified": len(manifest["artifact_hashes"]),
            "parent_artifacts_unchanged": len(base["artifact_hashes"]),
            "parent_pointer_unchanged": True,
            "reused_rows_byte_identical": True,
            "reused_algorithm_canaries_verified": len(compatibility),
            "all_records_all_methods_loaded": True,
            "no_sequence_or_esm_generated": True,
            "fingerprint": fingerprint,
        },
    )
    return manifest


SOURCE_VERIFICATION_SHA = "5cece4c23e89fb8ab226fdac8c78d9ff6a3e8fd8bd3b61b8cdc754f6c94530a8"
CONDITIONAL_AUDIT_SHA = "25b2a3957748b8c89f29efef93795a838cafca6c07c704f932ca75c06d546f66"

SPECS = [
    ("OLD:PV14:ee3e2f914d070a69da580291", "1052.4", 10, "mg/kg", "NX210c"),
    ("OLD:PV14:f426603a3be9bae869896153", "1416.5", 5, "mg/kg", "NX210c"),
    ("NEW:07:40000000005091286", "0.913", 100, "mg", "anidulafungin"),
    ("NEW:07:40000000005099417", "0.7", 150, "mg", "micafungin"),
    ("NEW:07:40000000002573932", "0.82", 4, "mg/kg", "daptomycin"),
    ("NEW:07:40000000002573933", "0.73", 4, "mg/kg", "daptomycin"),
]
PRIMARY_ID = "PRIMARY_CLABS:FDA021318:GHBI:NCA:90.34"
TERI_FAMILY = "FAMILY:59776a6ab3eb23c944c8a61d"
NX_XML = (
    ROOT
    / "data/external_downloads/peptide_property_expansion_v14/primary_pk"
    / "10_1007_s40120_024_00691_w/fulltext.xml"
)
NX_SHA = "30cb400c732c5767b10ea7729124b9eb55b0fbfedb2259dd6fdee29075060447"


def verify_sources(source=SOURCE, audit_path=AUDIT):
    source, audit_path = Path(source), Path(audit_path)
    receipt = source / "verification.json"
    if sha(receipt) != SOURCE_VERIFICATION_SHA or sha(audit_path) != CONDITIONAL_AUDIT_SHA:
        raise ValueError("Pinned primary/conditional source receipt changed")
    protected = {receipt: sha(receipt), audit_path: sha(audit_path), NX_XML: NX_SHA}
    for name, expected in read(receipt)["files"].items():
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Primary evidence path escapes source directory")
        path = source / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError("Primary evidence changed: " + name)
        protected[path] = expected
    if sha(NX_XML) != NX_SHA:
        raise ValueError("NX210c original XML changed")
    audit = read(audit_path)
    manifest = DB.parent / "manifest.json"
    if sha(manifest) != audit["input_hashes"]["dataset_manifest"]:
        raise ValueError("Frozen data manifest changed")
    protected[manifest] = sha(manifest)
    return audit, protected


def validate_cohort(values, *, verify_chemistry=True):
    """Keep chemistry QC local; reports bind already-validated graph hashes."""
    if verify_chemistry:
        from rdkit import Chem
    if len(values) != 7 or len({r["record_id"] for r in values}) != 7:
        raise ValueError("Exactly seven distinct source observations are required")
    graph_groups, families = {}, set()
    for row in values:
        context = row["conditional_core_context"]
        encode_context(json.loads(context) if isinstance(context, str) else context)
        canonical = row["canonical_isomeric_smiles"]
        if (
            not isinstance(canonical, str)
            or not canonical
            or row["smiles"] != canonical
            or row["structure_key"]
            != "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest()
        ):
            raise ValueError("Validated canonical graph/hash binding changed")
        if verify_chemistry:
            mol = Chem.MolFromSmiles(row["smiles"])
            if (
                mol is None
                or any(c == "?" for _, c in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                or Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) != canonical
            ):
                raise ValueError("Complete assigned-stereo graph is required")
        if graph_groups.setdefault(canonical, row["split_group"]) != row["split_group"]:
            raise ValueError("One complete graph crosses molecule families")
        families.add(row["split_group"])
        if row["unit"] != "L/h" or row["quantity"] != "total_systemic_clearance":
            raise ValueError("Only absolute total systemic CL in L/h is admitted")
        value = Decimal(str(row["value"]))
        if not value.is_finite() or value <= 0 or row["relation"] != "=":
            raise ValueError("A positive exact source point is required")
        if not math.isclose(float(row["target"]), math.log10(float(value)), abs_tol=1e-14):
            raise ValueError("Source label transform changed")
    if len(graph_groups) != 5 or len(families) != 5 or len(set(graph_groups.values())) != 5:
        raise ValueError("Graph union known-FAMILY must contain exactly five molecules")
    return values


def source_rows(source=SOURCE, audit_path=AUDIT):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    audit, protected = verify_sources(source, audit_path)
    candidates = {r["record_id"]: r for r in audit["candidate_records"]}
    official = {
        str(a["activity_id"]): a for b in audit["official_api_batches"] for a in b["activities"]
    }
    primary = read(Path(source) / "observation_evidence.json")["evidence_observations"]
    primary_by_old = {
        r.get("original_record_id"): r for r in primary if r.get("original_record_id")
    }
    identity = read(Path(source) / "identity_verification.json")
    accepted, proof = [], []
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        for record_id, expected, dose, dose_unit, name in SPECS:
            candidate = candidates[record_id]
            fetched = db.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
            if fetched is None:
                raise ValueError("Frozen source record missing")
            frozen = dict(fetched)
            if any(frozen.get(k) != v for k, v in candidate["frozen_source"].items()):
                raise ValueError("Frozen source row changed: " + record_id)
            if (
                frozen["endpoint"] != "CL_absolute"
                or frozen["unit"] != "L/h"
                or frozen["relation"] != "="
                or Decimal(frozen["value"]) != Decimal(expected)
            ):
                raise ValueError("Frozen exact total-CL scalar/unit changed")
            canonical, key = graph(frozen["smiles"])
            if name in ("anidulafungin", "micafungin"):
                activity = official[candidate["source_entry_id"].split(":")[1]]
                activity_graph, _ = graph(activity["canonical_smiles"])
                if (
                    activity_graph != canonical
                    or activity["standard_type"] != "CL"
                    or activity["standard_relation"] != "="
                    or activity["standard_units"] != "L/hr"
                    or Decimal(activity["standard_value"]) != Decimal(expected)
                    or not re.search(rf"\b{dose}\s*mg\b", activity["assay_description"])
                    or not re.search(
                        r"human.*plasma|human plasma", activity["assay_description"], re.I
                    )
                    or not re.search(r"\biv\b", activity["assay_description"], re.I)
                ):
                    raise ValueError("Official activity/graph/core-context mismatch")
            if name == "daptomycin":
                original = primary_by_old[record_id]
                reference, _ = graph(identity["daptomycin"]["canonical_isomeric_smiles"])
                if (
                    reference != canonical
                    or Decimal(str(original["value"])) != Decimal(expected)
                    or original["quantity"] != "total_systemic_clearance"
                    or original["matrix"] != "plasma"
                    or original["dose_value"] != 4
                    or original["route"] != "intravenous_infusion"
                ):
                    raise ValueError("Daptomycin primary exact observation mismatch")
            elif candidate.get("resolved_quantity_candidate") != "source_reported_total_CL_L_per_h":
                raise ValueError("Unresolved clearance definition")
            context = dict(
                species="human",
                route="intravenous",
                matrix="plasma",
                dose_value=dose,
                dose_reported_unit=dose_unit,
            )
            source_paper = (
                "PMID:17548489"
                if name == "daptomycin"
                else "DOI:10.1007/s40120-024-00691-w"
                if name == "NX210c"
                else "CHEMBL_REF:" + candidate["reference_ids"]
            )
            reported = dict(
                context,
                cohort=name,
                regimen=(
                    "thrice_weekly_four_weeks"
                    if name == "NX210c"
                    else "single_dose"
                    if name == "daptomycin"
                    else "source_literal_TID_frequency_not_resolved"
                    if name == "micafungin"
                    else "not_reported"
                ),
            )
            if name == "daptomycin":
                reported.update(
                    sex="female",
                    n=7,
                    infusion_minutes=30,
                    bodyweight_basis="actual_bodyweight",
                    cohort=primary_by_old[record_id]["evidence_observation_id"].split(":")[-1],
                )
            elif name == "NX210c":
                reported.update(
                    table_n=30,
                    table_n_is_independent_subject_count=False,
                    infusion_within_minutes=10,
                    sex="not_reported",
                )
            else:
                reported.update(sex="not_reported")
            row = dict(
                record_id=record_id,
                split_group=candidate["split_group"],
                value=expected,
                unit="L/h",
                target=repr(math.log10(float(expected))),
                relation="=",
                quantity="total_systemic_clearance",
                target_transform="log10",
                structure_eligible="true",
                sequence_eligible="false",
                smiles=canonical,
                canonical_isomeric_smiles=canonical,
                structure_key=key,
                conditional_core_context=context,
                source_paper=source_paper,
                reported_context=json.dumps(reported, sort_keys=True, separators=(",", ":")),
                graph_identity_tier="frozen_complete_graph_official_or_primary_exact_match",
                source_statistic=(
                    "reported_arithmetic_mean"
                    if name in ("NX210c", "daptomycin")
                    else "source_point_statistic_unreported"
                ),
                reported_sd_L_per_h=(
                    primary_by_old[record_id]["reported_sd"]
                    if name == "daptomycin"
                    else "not_reported"
                ),
                reported_n=(
                    7 if name == "daptomycin" else 30 if name == "NX210c" else "not_reported"
                ),
                n_interpretation=(
                    "subjects_in_source_cohort"
                    if name == "daptomycin"
                    else "source_table_PK_profiles_not_proven_independent_subjects"
                    if name == "NX210c"
                    else "not_reported"
                ),
            )
            accepted.append(row)
            proof.append(
                dict(
                    record_id=record_id,
                    frozen_row_sha256=object_hash(frozen),
                    complete_graph_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                    value_L_per_h=expected,
                    source_paper=source_paper,
                    raw_value=frozen["raw_value"],
                    raw_unit=frozen["raw_unit"],
                    source_context=candidate["frozen_source"]["condition_json"],
                )
            )
    new = next(r for r in primary if r["evidence_observation_id"] == PRIMARY_ID)
    ref = identity["teriparatide"]
    canonical, key = graph(ref["canonical_isomeric_smiles"])
    mol = Chem.MolFromSmiles(canonical)
    from_sequence = Chem.MolToSmiles(
        Chem.MolFromFASTA(ref["primary_sequence"]), isomericSmiles=True
    )
    if (
        canonical != from_sequence
        or rdMolDescriptors.CalcMolFormula(mol) != "C181H291N55O51S2"
        or Chem.MolToInchiKey(mol) != "OGBMKVWORPGQRR-UMXFMPSGSA-N"
        or len(Chem.FindMolChiralCenters(mol)) != 34
        or Decimal(str(new["value"])) != Decimal("90.34")
        or new["matrix"] != "serum"
        or new["route"] != "intravenous_infusion"
    ):
        raise ValueError("Independent FDA point or reference identity changed")
    context = dict(
        species="human",
        route="intravenous",
        matrix="serum",
        dose_value=17.54,
        dose_reported_unit="ug",
    )
    reported = dict(
        context,
        study="B3D-LC-GHBI",
        infusion_seconds=28,
        regimen="single_dose_crossover",
        n=22,
        men=11,
        women=11,
        analysis="noncompartmental_reported_cohort_summary",
        analyte="serum_immunoreactive_LY333334_validated_IRMA",
        SD="not_reported",
        individual_bodyweights="not_reported",
        sex="mixed_11_men_11_women",
    )
    accepted.append(
        dict(
            record_id=PRIMARY_ID,
            split_group=TERI_FAMILY,
            value="90.34",
            unit="L/h",
            target=repr(math.log10(90.34)),
            relation="=",
            quantity="total_systemic_clearance",
            target_transform="log10",
            structure_eligible="true",
            sequence_eligible="false",
            smiles=canonical,
            canonical_isomeric_smiles=canonical,
            structure_key=key,
            conditional_core_context=context,
            source_paper="FDA:NDA021318:GHBI",
            reported_context=json.dumps(reported, sort_keys=True, separators=(",", ":")),
            graph_identity_tier="FDA_sequence_PubChem_registered_neutral_reference_not_sample_microspecies",
            source_statistic="reported_NCA_cohort_summary_statistic_SD_not_specified",
            reported_sd_L_per_h="not_reported",
            reported_n=22,
            n_interpretation="single_crossover_cohort_11_men_11_women_not_22_molecules",
        )
    )
    proof.append(
        dict(
            record_id=PRIMARY_ID,
            value_L_per_h="90.34",
            original_record_replaced=False,
            source_sha256=new["source_sha256"],
            complete_graph_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            old_62_94_unbackfilled=True,
            population_77_1_not_added=True,
        )
    )
    validate_cohort(accepted)
    return accepted, proof, protected


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


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if (output / "task_manifest.json").exists():
        raise FileExistsError("Prepared CL task already exists; preserve the existing run")
    values, proofs, protected = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_features(values, proofs, protected, output, parent)
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("Source evidence changed during feature preparation")
    evidence = dict(
        schema=SCHEMA,
        status="qualified_not_trained",
        source_proofs=proofs,
        source_pins={str(p): h for p, h in protected.items()},
        warning=WARNING,
        original_rows_preserved=True,
        new_primary_observations=1,
        excluded_old_teriparatide_records=["NEW:07:40000000002633391", "NEW:07:40000000002633392"],
        exclusions=(
            "Old62/94 unbackfilled; SC CL/F62.2/94.3 and population77.1 are not this estimand"
        ),
    )
    write(output / "source_admission.json", evidence)
    contract = context_contract(values, sha(output / "source_admission.json"))
    task_id = (
        "TASK_" + object_hash({"estimand": ESTIMAND, "external_context": contract["sha256"]})[:24]
    )
    path = "task_tables/" + task_id + ".tsv"
    exported = [
        {
            k: (json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, dict) else v)
            for k, v in row.items()
        }
        for row in values
    ]
    (output / "task_tables").mkdir(exist_ok=True)
    write_tsv(output / path, list(exported[0]), exported)
    task = dict(
        task_id=task_id,
        endpoint="CL_absolute",
        task_kind="regression",
        status="eligible",
        table_path=path,
        n_records=7,
        n_groups=5,
        unit="L/h",
        target_transform="log10",
        inverse_transform="pow10",
        protocol="internal_grouped_cv_only_no_independent_holdout",
        fixed_external_covariates=contract,
        scope_warning=WARNING,
        scientific_definition=dict(
            endpoint="CL_absolute",
            quantity="total_systemic_clearance",
            quantity_unit="L/h",
            estimand=ESTIMAND,
            species="human",
            route="intravenous",
            molecular_equal_diagnostics_required=True,
            independent_holdout=False,
            graph_identity_is_reference_tier=True,
            context_fields=list(CONTEXT_KEYS),
            context_is_chemical_method=False,
        ),
        task_definition=dict(
            endpoint="CL_absolute",
            quantity_unit="L/h",
            estimand=ESTIMAND,
            molecular_equal_diagnostics_required=True,
            context_schema=SCHEMA,
        ),
    )
    write(output / "task_manifest.json", {"schema": SCHEMA, "tasks": [task], "models_fitted": 0})
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    methods = manifest["representation_methods"]["methods"]
    config = dict(
        dataset="data/dataset_reclean_20261113_v1/catalogue.sqlite",
        require_complete_feature_bank=False,
        resources={"nice": 10},
        required_seeds=list(SEEDS),
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
            inner_folds=5,
            min_rows=5,
            min_groups=5,
            seed=42,
            search_backend="optuna",
            search_budget=128,
            n_jobs=1,
        ),
    )
    write(output / "recommended_training_config.json", config)
    write(output / "config.json", config)
    write(output / "fivefold_prechecks.json", split_prechecks(values))
    evidence_dir = output / "source_evidence"
    evidence_dir.mkdir(exist_ok=True)
    for name in (
        "observation_evidence.json",
        "identity_verification.json",
        "verification.json",
        "retrieval_ledger.json",
    ):
        shutil.copy2(Path(source) / name, evidence_dir / name)
    for name in (
        "fda_original_biopharm.pdf",
        "fda_original_label.pdf",
        "daptomycin_PMC1932544.html",
        "pubchem_teriparatide.json",
        "pubchem_daptomycin.json",
    ):
        shutil.copy2(Path(source) / "raw" / name, evidence_dir / name)
    shutil.copy2(NX_XML, evidence_dir / "NX210c_primary.xml")
    write(evidence_dir / "frozen_six_observation_source_proofs.json", proofs[:6])
    report = dict(
        status="prepared_not_trained",
        task_id=task_id,
        records=7,
        groups=5,
        independent_methods=10,
        context_columns=len(FEATURE_NAMES),
        context_method_increment=0,
        feature_fingerprint=manifest["input_fingerprint"],
        warning=WARNING,
        molecular_report_required=True,
        three_seeds=list(SEEDS),
        independent_partitions=1,
    )
    write(output / "preparation_validation.json", report)
    (output / "REPORT.md").write_text(
        "# Human IV total absolute clearance\n\n7 source observations /5 molecule families; "
        "ten complete-graph methods plus known context, eight model families, strict internal "
        "five-fold Optuna128/seeds42,2026,3407. No fitting or remote launch by preparation.\n\n"
        + WARNING
        + "\n\nFDA90.34 is an independent primary observation; old62/94, SC CL/F "
        "and population77.1 are excluded. New teriparatide exact reference graph uses its own "
        "feature hash; no loss of modifications or replacement of old graphs.\n\n"
        f"Exact graph rows reused: {manifest['configuration']['reused_graphs']}; "
        f"new graph extractions: {manifest['configuration']['extracted_graphs']}. "
        "The PubChem/primary-sequence teriparatide exact graph was independently matched "
        "to the parent graph bank; this does not equate the different old ChEMBL tautomer "
        "hash. Source vectors are byte-reused. All217 RDKit descriptor columns and all "
        "other feature groups reproduce the parent with scoped BLAS1 canaries. "
        "Multithread characteristic-polynomial Ipc sensitivity is recorded separately; "
        "no tolerance, column removal or zero-filling was used. Existing NaNs remain "
        "missing for fold-local imputation.\n",
        encoding="utf-8",
    )
    write(
        output / "delivery_receipt.json",
        {
            "schema": SCHEMA,
            "status": "prepared_not_trained",
            "files": {
                p.relative_to(output).as_posix(): sha(p)
                for p in output.rglob("*")
                if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
            },
            "historical_intermediate_artifacts_not_execution_inputs": {
                p.relative_to(output).as_posix(): sha(p)
                for p in (output / "diagnostic_history").rglob("*")
                if p.is_file()
            },
        },
    )
    return report


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


def molecule_diagnostics(source_rows, predictions):
    validate_cohort(source_rows, verify_chemistry=False)
    source = {r["record_id"]: r for r in source_rows}
    if (
        len(predictions) != 7
        or len({r["record_id"] for r in predictions}) != 7
        or {r["record_id"] for r in predictions} != set(source)
    ):
        raise ValueError("OOF record identity differs from the seven source observations")
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
    return dict(
        status="verified",
        records=7,
        molecules=5,
        metrics_space="log10_L_per_h",
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
        if len(tasks) != 1 or tasks[0]["endpoint"] != "CL_absolute":
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
            raise ValueError("Conditional CL task identity changed")
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
        schema="conditional_cl_absolute_molecular_diagnostics_v1",
        status="verified",
        task_id=next(iter(observed_tasks)),
        seeds=list(SEEDS),
        reports=reports,
        model_selection_repeated=False,
        independent_holdout=False,
        warning=WARNING,
    )
    destination = root / "cl_absolute_molecular_diagnostics.json"
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
            parser.error("--config is required for followup reporting")
        result = run_followup(args.config, diagnostics_only=args.diagnostics_only)
    else:
        result = prepare(args.source, args.parent, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
