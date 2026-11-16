"""Prepare and audit a typed, source-native absolute-volume annotation experiment.

Different source estimands remain explicit query inputs. Preparation, verification,
node-fold probing and report-only commands never fit models or unpickle artifacts.
Only the explicitly selected future run command performs fitting.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import socket
import struct
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_mhc10_native_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/endpoint_gap_feasibility_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_vd_absolute_native_followup_20261115_v1"
NODE_OUTPUT = ROOT / "data/peptide_modeling_vd_absolute_native_nodefold_20261115_v2"
PREVIOUS_PRODUCER_SHA = "d517980295ea660e35505a372515558d8aadabc6c1e0ad45c801b7b1b85332bd"
SOURCE_FILE = "Vd_main_candidate_source_contract.json"
SOURCE_SHA = "83c52b8c0afb06f712070e93374949c3b072a38727a86eb7633291064b4ccf6e"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
REVIEW_SOURCE = ROOT / "data/vd_absolute_native_admission_review_20261115_v1"
REVIEW_SHA = "452e2dfc0058f46fb7c1135774951dadee8571b34570693160a8629d9be026bd"
CORRECTIONS_SHA = "56b0e9ed3852926aee12efade3f4ce8c2ddd93e4b78d5efc598357aebc3e8269"
SCHEMA = "vd_absolute_native_typed_annotation_v1"
ENDPOINT = "Vd_absolute"
QUANTITY = "native_source_absolute_volume_annotation_given_reported_estimand"
TASK = "TASK_VD_ABSOLUTE_NATIVE_" + SOURCE_SHA[:24]
CONTEXT_GROUP = "vd_absolute_native_context"
SEEDS = (42, 2026, 3407)
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
UNKNOWN = "not_reported_in_observation_context"
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
CONTEXT_FIELDS = (*CATEGORICAL, *NUMERIC)
WARNING = (
    "Source-native absolute-volume annotation exploration, conditional on the exact reported "
    "estimand. Vc, Vp, Vz, Vz/F, baseline-adjusted or steady-state variants and source-named "
    "Vss are not one physiological Vd. Most estimands have one molecule; unsupported types "
    "and joint contexts require explicit per-fit diagnostics and query rejection. Repeated "
    "doses/cohorts are not independent molecules. This is neither clinical validation nor "
    "proof of each estimand's predictive generalization. All27 refit has no independent score."
)
sha, read, rows, write, table, object_hash, contained = (
    shared.sha,
    shared.read,
    shared.rows,
    shared.write,
    shared.table,
    shared.object_hash,
    shared.contained,
)


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare_target_text(frozen, value):
    """Only bound platform log10/.15g text; never rewrite the frozen training target."""
    derived = format(math.log10(float(Decimal(value))), ".15g")
    a, b = Decimal(frozen), Decimal(derived)
    x, y = float(frozen), float(derived)
    if not all(math.isfinite(v) for v in (x, y, float(value))) or Decimal(value) <= 0:
        raise ValueError("Positive finite source volume and target text required")
    if format(x, ".15g") != frozen:
        raise ValueError("Frozen target is not canonical .15g text")
    if frozen == derived:
        return "exact_text"
    if not a or not b or math.copysign(1, x) != math.copysign(1, y):
        raise ValueError("Zero/sign target changes cannot use portability tolerance")

    def ordered(v):
        n = struct.unpack("!Q", struct.pack("!d", v))[0]
        return 0x8000000000000000 - (n & 0x7FFFFFFFFFFFFFFF) if n >> 63 else n + 2**63

    if abs(a - b) > Decimal(10) ** (min(a.adjusted(), b.adjusted()) - 14):
        raise ValueError("Target differs by more than one final .15g decimal place")
    if abs(ordered(x) - ordered(y)) > 128:
        raise ValueError("Target text differs by more than128binary64ULP")
    return "bounded_platform_log10_text_only_no_target_rewrite"


def source_context(record, corrections=None):
    """Use the exact independent source-bound correction; old parsed fields stay provenance."""
    if corrections is None:
        path = REVIEW_SOURCE / "typed_context_corrections.json"
        if sha(path) != CORRECTIONS_SHA:
            raise ValueError("Independent typed-context correction changed")
        corrections = {r["record_id"]: r for r in read(path)["records"]}
    corrected = corrections[record["record_id"]]
    if corrected["source_contract_sha256"] != SOURCE_SHA or corrected[
        "original_frozen_context"
    ] != {k: record[k] for k in ("frozen_route", "frozen_species", "frozen_matrix", "frozen_dose")}:
        raise ValueError("Context correction no longer binds the exact frozen source fields")
    c = corrected["typed_context"]
    if c["source_estimand"] != record["source_estimand"]:
        raise ValueError("Independent source estimand changed")
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
    out = {key: c[source] for key, source in mapping.items()}
    out["reference_covariate_status"] = c.get("reference_covariate_status", "not_reported")
    out.update(
        frequency_status=c["frequency"],
        visit_status="explicit_day" if c["day"] is not None else "not_reported",
        frequency_per_day=3.0 if c["frequency"] == "three_times_daily" else 0.0,
        visit_day=float(c["day"]) if c["day"] is not None else 0.0,
    )
    levels = [Decimal(x) for x in c["dose_values"]]
    out.update(
        dose_min_reported_unit=float(min(levels)) if levels else 0.0,
        dose_max_reported_unit=float(max(levels)) if levels else 0.0,
    )
    # The exact pooled dose set is included as typed numeric levels; neither
    # an assumed reference weight nor the source's free-text dose is a predictor.
    out["dose_level_count"] = float(len(levels))
    if len(levels) > 4:
        raise ValueError("Unreviewed additional pooled-dose levels")
    out.update(
        {"dose_level_" + str(i + 1): float(levels[i]) if i < len(levels) else 0.0 for i in range(4)}
    )
    return out, [corrected["source_binding"]]


def ontology_for(values):
    ontology = {
        k: sorted({json.loads(r["semantic_context_json"])[k] for r in values}) for k in CATEGORICAL
    }
    names = [f"{k}={v}" for k in CATEGORICAL for v in ontology[k]] + list(NUMERIC)
    return dict(
        schema=SCHEMA,
        categorical=ontology,
        numeric=list(NUMERIC),
        feature_names=names,
        unknown_is_explicit=True,
        label_or_identifier_columns=0,
    )


def encode_context(context, ontology):
    if set(context) != set(CONTEXT_FIELDS):
        raise ValueError("Typed semantic fields only; identifiers/outcomes/free text forbidden")
    vector = []
    for key in CATEGORICAL:
        if context[key] not in ontology["categorical"][key]:
            raise ValueError("Unsupported semantic category: " + key)
        vector.extend(float(context[key] == v) for v in ontology["categorical"][key])
    for key in NUMERIC:
        value = context[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("Finite typed numeric context required")
        vector.append(float(value))
    missing = {UNKNOWN, "not_reported", "not_reported_for_bound_volume_estimate"}
    for status, columns in [
        (
            "dose_status",
            (
                "dose_min_reported_unit",
                "dose_max_reported_unit",
                "dose_level_count",
                "dose_level_1",
                "dose_level_2",
                "dose_level_3",
                "dose_level_4",
            ),
        ),
        ("frequency_status", ("frequency_per_day",)),
        ("visit_status", ("visit_day",)),
    ]:
        if context[status] in missing and any(context[k] != 0 for k in columns):
            raise ValueError("Unknown context uses masked zero channels, not assumed values")
    if context["dose_status"] in missing and context["dose_unit"] not in missing:
        raise ValueError("Missing dose cannot silently retain an assumed unit")
    return np.asarray(vector, dtype=np.float32)


def source_rows(source=SOURCE, *, verify_chemistry=False):
    path = Path(source) / SOURCE_FILE
    if sha(path) != SOURCE_SHA:
        raise ValueError("Frozen27-observation main source contract changed")
    contract = read(path)
    corrections_path = Path(source) / "typed_context_corrections.json"
    if not corrections_path.exists():
        corrections_path = REVIEW_SOURCE / "typed_context_corrections.json"
    if sha(corrections_path) != CORRECTIONS_SHA:
        raise ValueError("Independent source-context correction SHA changed")
    corrections = {r["record_id"]: r for r in read(corrections_path)["records"]}
    values, provenance = [], []
    for record in contract["records"]:
        if record["relation"] != "=" or record["unit"] != "L":
            raise ValueError("Exact source-native L values only")
        graph, smiles = record["canonical_graph"], record["registered_canonical_smiles"]
        if graph != "CANON_SMILES:" + hashlib.sha256(smiles.encode()).hexdigest():
            raise ValueError("Whole-graph key differs from exact pinned canonical SMILES")
        if verify_chemistry:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None or Chem.MolToSmiles(mol, isomericSmiles=True) != smiles:
                raise ValueError("Frozen graph did not round-trip exactly")
            if any(
                label == "?" for _, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            ):
                raise ValueError("Source complete reference graph has unspecified stereochemistry")
        context, anchors = source_context(record, corrections)
        semantic = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        values.append(
            dict(
                record_id=record["record_id"],
                source_original_record_id=record["record_id"],
                endpoint=ENDPOINT,
                value=record["value"],
                raw_target=record["value"],
                unit="L",
                target=format(math.log10(float(Decimal(record["value"]))), ".15g"),
                target_transform="log10",
                inverse_transform="pow10",
                canonical_identity_group=graph,
                canonical_smiles=smiles,
                smiles=smiles,
                structure_eligible="true",
                sequence_eligible="false",
                sequence="",
                sequence_kind="",
                split_group=record["wholegraph_union_known_family_group"],
                measurement_group=record["source_observation_key"],
                source_group=record["source_entry_id"].split("|", 1)[0],
                source_record_id=record["source_record_id"],
                source_entry_id=record["source_entry_id"],
                semantic_context_json=semantic,
                semantic_context_key=object_hash(context),
                source_estimand=context["estimand"],
                vd_native_context_eligible="true",
                canonical_quantity_id=QUANTITY,
                identity_tier=record["identity_tier"],
            )
        )
        provenance.append(
            dict(
                record_id=record["record_id"],
                context=context,
                context_source_anchors=anchors,
                frozen_source_record=record,
                frozen_fields_are_provenance_only=True,
            )
        )
    validate_rows(values)
    return values, provenance


def validate_rows(values):
    if len(values) != 27 or len({r["record_id"] for r in values}) != 27:
        raise ValueError("Exactly27 frozen source observations are required")
    graphs, groups, graph_groups = set(), set(), {}
    for r in values:
        compare_target_text(r["target"], r["value"])
        if r["unit"] != "L" or r["canonical_quantity_id"] != QUANTITY:
            raise ValueError("Wrong quantity or normalized unit")
        key = r["canonical_identity_group"]
        if key != "CANON_SMILES:" + hashlib.sha256(r["canonical_smiles"].encode()).hexdigest():
            raise ValueError("Source graph bytes changed")
        if graph_groups.setdefault(key, r["split_group"]) != r["split_group"]:
            raise ValueError("One complete graph crosses molecular groups")
        context = json.loads(r["semantic_context_json"])
        if set(context) != set(CONTEXT_FIELDS) or r["semantic_context_key"] != object_hash(context):
            raise ValueError("Context membership/key changed")
        if context["matrix"] in {"male", "healthy", "patients"}:
            raise ValueError("Old population/sex words cannot become matrix features")
        graphs.add(key)
        groups.add(r["split_group"])
    if len(graphs) != 10 or len(groups) != 10:
        raise ValueError("Must preserve10 complete graphs and graph-union-known-FAMILY groups")


def context_support(query, fitted_rows):
    contexts = [json.loads(r["semantic_context_json"]) for r in fitted_rows]
    known_types = {r["estimand"] for r in contexts}
    estimand_groups = len(
        {r["split_group"] for r in fitted_rows if r["source_estimand"] == query["estimand"]}
    )
    joint_groups = len(
        {r["split_group"] for r in fitted_rows if r["semantic_context_key"] == object_hash(query)}
    )
    return dict(
        estimand_seen=query["estimand"] in known_types,
        statistic_basis_seen=query["statistic_basis"] in {r["statistic_basis"] for r in contexts},
        joint_context_seen=object_hash(query) in {r["semantic_context_key"] for r in fitted_rows},
        fitted_molecular_groups=len({r["split_group"] for r in fitted_rows}),
        estimand_molecular_groups=estimand_groups,
        joint_context_molecular_groups=joint_groups,
        low_estimand_molecular_support=estimand_groups < 2,
        new_molecule_generalization_validated=False,
    )


def split_prechecks(values, output):
    from peptide_modeling_train import _inner_splits, freeze_outer_split
    from peptide_modeling_train import validate_rows as validate

    y, groups = validate(values, "regression")
    checks = []
    for seed in SEEDS:
        settings = dict(
            seed=seed, inner_folds=5, test_size=0.2, min_groups=5, min_rows=5, min_train_rows=6
        )
        path = Path(output) / "task_tables/predeclared_outer_splits" / f"seed_{seed}.json"
        split = freeze_outer_split(values, "regression", path, settings)
        tr = np.flatnonzero(np.isin(groups, split["train_groups"]))
        te = np.flatnonzero(np.isin(groups, split["test_groups"]))
        folds = _inner_splits(y[tr], groups[tr], "regression", settings)
        if len(folds) != 5 or len(set(groups[tr])) < 5 or set(groups[tr]) & set(groups[te]):
            raise ValueError("Strict fivefold grouped holdout protocol unavailable")
        fold_rows = []
        for f, (a, b) in enumerate(folds):
            fitted = [values[tr[i]] for i in a]
            validation = [values[tr[i]] for i in b]
            fold_rows.append(
                dict(
                    fold=f,
                    train_record_ids=[r["record_id"] for r in fitted],
                    validation_record_ids=[r["record_id"] for r in validation],
                    context_support={
                        r["record_id"]: context_support(
                            json.loads(r["semantic_context_json"]), fitted
                        )
                        for r in validation
                    },
                )
            )
        fitted = [values[i] for i in tr]
        checks.append(
            dict(
                seed=seed,
                actual_folds=5,
                split_path=path.relative_to(output).as_posix(),
                split_sha256=sha(path),
                train_record_ids=[values[i]["record_id"] for i in tr],
                holdout_record_ids=[values[i]["record_id"] for i in te],
                train_groups=len(set(groups[tr])),
                holdout_groups=len(set(groups[te])),
                folds=fold_rows,
                holdout_context_support={
                    values[i]["record_id"]: context_support(
                        json.loads(values[i]["semantic_context_json"]), fitted
                    )
                    for i in te
                },
                final_refit_record_ids=[r["record_id"] for r in values],
                all27_refit_support={
                    r["record_id"]: context_support(json.loads(r["semantic_context_json"]), values)
                    for r in values
                },
            )
        )
    return dict(
        protocol="group_holdout_inner5_then_separate_all_common_refit",
        seeds=checks,
        refit_full_required=True,
        independent_holdout=True,
        final_training_records=27,
        physiological_or_subtype_generalization_claim=False,
    )


def slice_features(values, ontology, parent, output):
    """Reuse exact reference vectors; extract only absent complete registered graphs."""
    from peptide_modeling_clint_features import extract_all
    from peptide_modeling_dramp_binary_reference_features import (
        canary_check,
        checked_vector,
        reuse_partition,
    )
    from peptide_modeling_features import _verify_artifacts, load_feature_route
    from threadpoolctl import threadpool_limits

    pointer = Path(parent) / "features/manifest.json"
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Frozen unified feature parent pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    root = Path(base["cache_dir"])
    index = {r["key"]: int(r["row_idx"]) for r in rows(root / "structure_bank_index.tsv")}
    parent_graphs = {r["key"]: r for r in rows(root / "structure_source_smiles.tsv")}
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in values}
    graph_rows = [dict(key=k, canonical_isomeric_smiles=v) for k, v in sorted(graph_map.items())]
    reused, missing = reuse_partition(graph_rows, parent_graphs, index)
    original_hashes = copy.deepcopy(base["artifact_hashes"])
    extraction_modules = [
        "peptide_modeling_clint_features.py",
        "peptide_modeling_expand_features.py",
        "peptide_modeling_add_methods.py",
        "peptide_modeling_add_smiles_methods.py",
        "peptide_modeling_features.py",
        "peptide_modeling_dramp_binary_reference_features.py",
    ]
    pins = {n: sha(ROOT / "scripts" / n) for n in extraction_modules}
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    with threadpool_limits(limits=1):
        canaries = canary_check(base, reused, index)
        extracted = {
            r["key"]: extract_all(r["canonical_isomeric_smiles"], r["key"]) for r in missing
        }
    cache = Path(output) / "features" / object_hash([QUANTITY, graph_rows, values, ontology])[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {n: copy.deepcopy(base["groups"][n]) for n in GROUPS}
    reuse_proof = []
    for name, meta in groups.items():
        old = np.load(root / meta["matrix"], mmap_mode="r")
        old_mask = np.load(root / meta["valid"], mmap_mode="r")
        selected = []
        for row in graph_rows:
            key = row["key"]
            if key in index:
                vector = old[index[key]].copy()
                if not old_mask[index[key]]:
                    raise ValueError("An exact parent graph lacks a valid structural method")
                if vector.tobytes() != old[index[key]].tobytes():
                    raise ValueError("Parent vector bytes changed during copying")
                selected.append(vector)
                reuse_proof.append(
                    dict(
                        key=key, group=name, byte_identical=True, vector_sha256=digest_array(vector)
                    )
                )
            else:
                vectors, masks, _ = extracted[key]
                selected.append(checked_vector(vectors[name], masks[name], meta, name))
        matrix = np.asarray(selected, dtype=np.dtype(meta["dtype"]))
        mask = np.ones(len(graph_rows), dtype=bool)
        if np.isinf(matrix).any() or np.isnan(matrix).all(axis=1).any():
            raise ValueError("No Inf/all-NaN valid graph rows are accepted")
        np.save(cache / meta["matrix"], matrix, allow_pickle=False)
        np.save(cache / meta["valid"], mask, allow_pickle=False)
        meta.update(shape=list(matrix.shape), index="structure_bank_index.tsv")
    context = np.vstack(
        [encode_context(json.loads(r["semantic_context_json"]), ontology) for r in values]
    )
    np.save(cache / "semantic_context.npy", context, allow_pickle=False)
    np.save(
        cache / "semantic_context_valid.npy", np.ones(len(values), dtype=bool), allow_pickle=False
    )
    groups[CONTEXT_GROUP] = dict(
        matrix="semantic_context.npy",
        valid="semantic_context_valid.npy",
        index="context_index.tsv",
        key_kind="record_id",
        eligibility_field="vd_native_context_eligible",
        shape=list(context.shape),
        feature_names=ontology["feature_names"],
        dtype="float32",
        chemical_method=False,
        source_kind="semantic_context",
    )
    table(
        cache / "structure_bank_index.tsv",
        [dict(key=r["key"], row_idx=i) for i, r in enumerate(graph_rows)],
    )
    table(cache / "structure_source_smiles.tsv", graph_rows)
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
        raise ValueError("Ten independent chemical methods required, variants not recounted")
    registry = dict(
        methods=methods,
        independent_method_count=10,
        chemical_dimensions=15212,
        chemical_groups=13,
        context_columns=context.shape[1],
        context_method_increment=0,
    )
    routes = {m["route"]: m["group_names"] + [CONTEXT_GROUP] for m in methods}
    routes["structure_methods_all"] = list(GROUPS) + [CONTEXT_GROUP]
    write(cache / "representation_methods.json", registry)
    manifest = dict(
        schema=SCHEMA,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        input_fingerprint=object_hash([graph_rows, values, ontology]),
        record_index="record_bank_index.tsv",
        requested_records=len(values),
        structure_keys=len(graph_rows),
        groups=groups,
        routes=routes,
        representation_methods=registry,
        parent_pointer_sha256=PARENT_SHA,
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for route in routes:
        loaded = load_feature_route(values, manifest, route)
        if (
            not loaded["valid"].all()
            or loaded["record_ids"] != [r["record_id"] for r in values]
            or not np.array_equal(loaded["X"][:, -context.shape[1] :], context)
        ):
            raise ValueError("Route lost exact graph/record context alignment")
    if sum(groups[n]["shape"][1] for n in GROUPS) != 15212:
        raise ValueError("Declared chemical width changed")
    _verify_artifacts(base)
    if sha(pointer) != PARENT_SHA or base["artifact_hashes"] != original_hashes:
        raise ValueError("Immutable parent bank/pointer changed")
    if any(sha(ROOT / "scripts" / n) != pin for n, pin in pins.items()):
        raise ValueError("Extractor source changed while building the isolated bank")
    write(Path(output) / "features/manifest.json", manifest)
    write(Path(output) / "representation_methods.json", registry)
    write(
        Path(output) / "feature_delivery_receipt.json",
        dict(
            status="verified_feature_preparation_only",
            records=27,
            complete_graphs=10,
            reused_exact_graphs=len(reused),
            newly_extracted_complete_graphs=len(missing),
            exact_parent_vectors=reuse_proof,
            algorithm_byte_canaries=canaries,
            descriptor_columns_compared_per_canary=217,
            algorithm_source_pins=pins,
            parent_pointer_unchanged=True,
            parent_artifacts_unchanged=len(original_hashes),
            complete_feature_artifact_hashes=manifest["artifact_hashes"],
            independent_methods=10,
            chemical_groups=13,
            chemical_dimensions=15212,
            context_columns=context.shape[1],
            BLAS_policy="one_thread",
            modified_molecule_source_graphs_preserved=True,
            guessed_graphs=0,
            model_fits=0,
        ),
    )
    return manifest


def verify_source_proof(prepared, task=None, values=None):
    """Portable no-chemistry/no-pickle/no-fit proof verification."""
    root = Path(prepared)
    task = task or read(root / "task_manifest.json")["tasks"][0]
    values = values or rows(contained(root, task["table_path"]))
    descriptor = task["native_source_observation_coverage"]
    proof_path = contained(root, descriptor["proof_path"])
    if sha(proof_path) != descriptor["proof_sha256"]:
        raise ValueError("Native source proof changed")
    proof = read(proof_path)
    if (
        task["task_id"] != TASK
        or task["endpoint"] != ENDPOINT
        or len(values) != 27
        or proof["implementation_sha256"] != sha(__file__)
        or proof["source_contract_sha256"] != SOURCE_SHA
    ):
        raise ValueError("Unexpected producer/source/task identity")
    for item in proof["pinned_artifacts"]:
        if sha(contained(root, item["path"])) != item["sha256"]:
            raise ValueError("Pinned preparation artifact changed: " + item["path"])
    expected, _ = source_rows(root / "source_provenance")
    if len(expected) != len(values):
        raise ValueError("Source observation count changed")
    for frozen, regenerated in zip(values, expected, strict=True):
        if set(frozen) != set(regenerated) or any(
            frozen[k] != str(regenerated[k]) for k in frozen if k != "target"
        ):
            raise ValueError("An exact source label/context/graph/non-target field changed")
        compare_target_text(frozen["target"], regenerated["value"])
    if proof["row_hashes"] != {r["record_id"]: object_hash(r) for r in values}:
        raise ValueError("Frozen target/table bytes or membership changed")
    validate_rows(values)
    if proof.get("source_closure"):
        closure = proof["source_closure"]
        if sha(contained(root, closure["path"])) != closure["sha256"]:
            raise ValueError("Explicit source-staging closure manifest changed")
        verify_source_closure(root)
    return proof


def verify_source_closure(root):
    root = Path(root)
    closure = read(root / "source_staging_closure.json")
    if (
        closure.get("schema") != "vd_absolute_source_staging_closure_v2"
        or closure.get("producer_sha256") != sha(__file__)
        or closure.get("task_id") != TASK
    ):
        raise ValueError("Unexpected source-staging closure identity")
    for name, pin in closure["files"].items():
        path = contained(root, name)
        if not path.is_file() or path.is_symlink() or sha(path) != pin:
            raise ValueError("Round lacks exact source-closure input: " + name)
    required = {
        "source_provenance/" + SOURCE_FILE,
        "source_provenance/typed_context_corrections.json",
        "independent_source_review.json",
        "independent_source_review_binding.json",
    }
    if not required <= set(closure["files"]):
        raise ValueError("Closure silently excludes source/review/correction dependencies")
    return closure


def stage_seed(template, destination, seed, outer_split, settings):
    """Actual controller staging, also exercised by every no-fit node probe."""
    template, destination = Path(template), Path(destination)
    if seed not in SEEDS or destination.exists():
        raise ValueError("New isolated round and one declared seed are required")
    closure = verify_source_closure(template)
    if any(p.is_symlink() for p in template.rglob("*")):
        raise ValueError("Template closure must use regular contained files")
    if any((template / n).exists() for n in ("models", "results", "checkpoints")):
        raise ValueError("Immutable source template already contains training outputs")
    shutil.copytree(template, destination, ignore=shutil.ignore_patterns("__pycache__"))
    verify_source_closure(destination)
    runtime_receipts = {}
    for name in closure["runtime_receipts_required_when_bound"]:
        source = template / name
        if source.exists():
            copied = contained(destination, name)
            if not copied.is_file() or sha(copied) != sha(source):
                raise ValueError("Round lost actual node review/receipt: " + name)
            runtime_receipts[name] = sha(copied)
    write(destination / "splits" / TASK / "group_assignment.json", outer_split)
    current = copy.deepcopy(settings)
    current["training"]["seed"] = seed
    current["resources"]["max_parallel_tasks"] = 1
    current["run_dir"] = str(destination)
    write(destination / "config.json", current)
    portable = preparation_review(destination)
    evidence = dict(
        status="verified_no_fit_staged_source_closure",
        seed=seed,
        source_closure_sha256=sha(destination / "source_staging_closure.json"),
        source_closure_files_checked=len(closure["files"]),
        exact_source_proof_sha256=portable["source_proof_sha256"],
        exact_task_table_sha256=portable["task_table_sha256"],
        runtime_receipts_copied=runtime_receipts,
        portable_source_and_all_routes_passed=True,
        outer_split_sha256=sha(destination / "splits" / TASK / "group_assignment.json"),
        config_sha256=sha(destination / "config.json"),
        estimator_fits=0,
        searches=0,
        model_unpickles=0,
        chemistry_imports=0,
    )
    write(destination / "round_source_closure_verification.json", evidence)
    return current, evidence


def clone_nodefold_version(previous, output=NODE_OUTPUT):
    """Version the no-fit closure without rewriting the frozen v1 labels or arrays."""
    import ast

    from peptide_modeling_features import _verify_artifacts

    previous, output = Path(previous).resolve(), Path(output).resolve()
    if output.exists() or previous in output.parents or output in previous.parents:
        raise ValueError("New disjoint prepared namespace required")
    old_task = read(previous / "task_manifest.json")["tasks"][0]
    old_proof_path = old_task["native_source_observation_coverage"]["proof_path"]
    old_proof = read(previous / old_proof_path)
    if (
        old_proof["implementation_sha256"] != PREVIOUS_PRODUCER_SHA
        or old_proof["source_contract_sha256"] != SOURCE_SHA
        or sha(previous / old_proof_path)
        != old_task["native_source_observation_coverage"]["proof_sha256"]
    ):
        raise ValueError("Only the exact frozen v1 source lineage may be versioned here")
    for item in old_proof["pinned_artifacts"]:
        if sha(contained(previous, item["path"])) != item["sha256"]:
            raise ValueError("Historical source input changed before cloning")
    old_feature = read(previous / "features/manifest.json")
    _verify_artifacts(old_feature)
    before = {
        p.relative_to(previous).as_posix(): sha(p)
        for p in previous.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    shutil.copytree(previous, output, ignore=shutil.ignore_patterns("__pycache__"))
    history = output / "previous_preparation_v1"
    history.mkdir()
    for name in (
        "task_manifest.json",
        "targeted_task_manifest.json",
        "delivery_receipt.json",
        "active_file_list.json",
        "active_file_list.tsv",
        "preparation_review.json",
        "local_no_fit_probe.json",
        "REPORT.md",
        "RUNBOOK.md",
    ):
        if (output / name).exists():
            shutil.copyfile(output / name, history / name)
    shutil.copyfile(output / old_proof_path, history / "source_proof.json")
    feature = read(output / "features/manifest.json")
    old_cache_name = Path(feature["cache_dir"]).name
    old_fp = feature["input_fingerprint"]
    new_fp = object_hash([old_fp, sha(__file__), "explicit_source_staging_closure_v2"])
    new_cache = output / "features" / new_fp[:16]
    (output / "features" / old_cache_name).rename(new_cache)
    feature.update(
        cache_dir=str(new_cache),
        manifest_path=str(new_cache / "manifest.json"),
        input_fingerprint=new_fp,
        previous_input_fingerprint=old_fp,
        numerical_feature_arrays_unchanged=True,
    )
    write(new_cache / "manifest.json", feature)
    write(output / "features/manifest.json", feature)
    copied_arrays = {}
    for name, meta in feature["groups"].items():
        for field in ("matrix", "valid"):
            rel = meta[field]
            old = previous / "features" / old_cache_name / rel
            new = new_cache / rel
            if sha(old) != sha(new):
                raise ValueError("Versioning altered a numerical feature array")
            copied_arrays[f"{name}:{field}"] = sha(new)
    if sha(output / old_task["table_path"]) != sha(previous / old_task["table_path"]):
        raise ValueError("Versioning rewrote source labels or target bytes")
    proof = read(output / old_proof_path)
    proof.update(
        implementation_sha256=sha(__file__),
        previous_source_proof_sha256=sha(previous / old_proof_path),
        previous_implementation_sha256=PREVIOUS_PRODUCER_SHA,
    )
    binding = read(output / "independent_source_review_binding.json")
    binding["implementation_sha256"] = sha(__file__)
    write(output / "independent_source_review_binding.json", binding)
    modules, pending = {}, [Path(__file__).stem]
    while pending:
        name = pending.pop()
        if name in modules:
            continue
        source = Path(__file__).with_name(name + ".py")
        if not source.exists():
            continue
        modules[name] = source
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            candidates = []
            if isinstance(node, ast.Import):
                candidates.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                candidates.append(node.module.split(".")[0])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("peptide_modeling_") and node.value.isidentifier():
                    candidates.append(node.value)
            pending.extend(n for n in candidates if n.startswith("peptide_modeling_"))
    for name, source in modules.items():
        destination = output / "scripts" / (name + ".py")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    closure_files = {}
    proof_rel = Path(old_proof_path).as_posix()
    for folder in ("source_provenance", "task_tables", "features", "scripts"):
        for path in (output / folder).rglob("*"):
            rel = path.relative_to(output).as_posix()
            if path.is_file() and rel != proof_rel and "__pycache__" not in path.parts:
                closure_files[rel] = sha(path)
    for name in (
        "independent_source_review.json",
        "independent_source_review_binding.json",
        "feature_delivery_receipt.json",
        "representation_methods.json",
        "feature_rows_index.tsv",
    ):
        closure_files[name] = sha(output / name)
    closure = dict(
        schema="vd_absolute_source_staging_closure_v2",
        task_id=TASK,
        producer_sha256=sha(__file__),
        files=closure_files,
        independently_bound_inputs=["task_manifest.json", proof_rel],
        runtime_receipts_required_when_bound=[
            "nodefold_binding.json",
            "node23_actual_no_fit_fold_receipt.json",
            "post_adoption_round_staging_binding.json",
            "post_adoption_round_staging_receipt.json",
        ],
        actual_controller_stage_function="stage_seed",
        missing_source_file_is_fatal=True,
    )
    write(output / "source_staging_closure.json", closure)
    proof["source_closure"] = dict(
        path="source_staging_closure.json", sha256=sha(output / "source_staging_closure.json")
    )
    write(output / old_proof_path, proof)
    manifest = read(output / "task_manifest.json")
    task = manifest["tasks"][0]
    task["status"] = "prepared_pending_source_review_and_nodefold"
    for field in ("scientific_definition", "task_definition"):
        task[field].update(
            implementation_sha256=sha(__file__),
            source_staging_closure_required=True,
            post_adoption_round_staging_required=True,
            pretraining_fold_state="v2_actual_node_and_round_closure_receipts_required",
        )
    task["native_source_observation_coverage"]["proof_sha256"] = sha(output / old_proof_path)
    manifest["execution_gate"] = "review_nodefold_and_postadoption_round_closure_required"
    write(output / "task_manifest.json", manifest)
    write(output / "targeted_task_manifest.json", manifest)
    review = preparation_review(output)
    write(output / "preparation_review.json", review)
    after = {
        p.relative_to(previous).as_posix(): sha(p)
        for p in previous.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if before != after:
        raise ValueError("Cloning changed immutable previous inputs")
    receipt = dict(
        status="v2_cloned_no_fit",
        previous=str(previous),
        output=str(output),
        producer_sha256=sha(__file__),
        previous_all_file_hashes=before,
        previous_all_file_bytes_unchanged=True,
        task_table_sha256=sha(output / task["table_path"]),
        all_label_target_bytes_unchanged=True,
        numerical_arrays_sha256=copied_arrays,
        source_closure_sha256=sha(output / "source_staging_closure.json"),
        source_closure_files=len(closure_files),
        local_script_dependency_files=len(modules),
        source_proof_sha256=sha(output / old_proof_path),
        previous_feature_fingerprint=old_fp,
        feature_fingerprint=new_fp,
        feature_pointer_sha256=sha(output / "features/manifest.json"),
        model_fits=0,
        feature_extractions=0,
    )
    write(output / "v2_clone_receipt.json", receipt)
    return receipt


def preparation_review(prepared):
    root = Path(prepared)
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    proof = verify_source_proof(root, task, values)
    from peptide_modeling_features import _verify_artifacts, load_feature_route
    from peptide_modeling_pipeline import portable_features

    features = portable_features(root)
    _verify_artifacts(features)
    for route in features["routes"]:
        if not load_feature_route(values, features, route)["valid"].all():
            raise ValueError("A prepared route lacks complete source coverage")
    return dict(
        status="prepared_no_fit_not_execution_approval",
        task_id=TASK,
        records=27,
        complete_graphs=10,
        molecular_groups=10,
        implementation_sha256=sha(__file__),
        source_contract_sha256=SOURCE_SHA,
        source_proof_sha256=sha(root / task["native_source_observation_coverage"]["proof_path"]),
        task_table_sha256=sha(root / task["table_path"]),
        frozen_row_hashes=proof["row_hashes"],
        verified_routes=len(features["routes"]),
        estimator_fits=0,
        searches=0,
        model_unpickles=0,
        warning=WARNING,
    )


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("New output namespace required; never overwrite prior preparation")
    values, provenance = source_rows(source, verify_chemistry=True)
    output.mkdir(parents=True, exist_ok=False)
    table_path = "task_tables/" + TASK + ".tsv"
    table(output / table_path, values)
    values = rows(output / table_path)
    checks = split_prechecks(values, output)
    write(output / "task_tables/predeclared_splits.json", checks)
    ontology = ontology_for(values)
    write(output / "task_tables/semantic_context_contract.json", ontology)
    write(output / "task_tables/observation_context_provenance.json", provenance)
    source_dir = output / "source_provenance"
    source_dir.mkdir()
    shutil.copyfile(Path(source) / SOURCE_FILE, source_dir / SOURCE_FILE)
    shutil.copyfile(
        REVIEW_SOURCE / "typed_context_corrections.json",
        source_dir / "typed_context_corrections.json",
    )
    raw_pins = read(Path(source) / "source_pins.json")
    for n, pin in raw_pins.items():
        if sha(ROOT / n) != pin:
            raise ValueError("Original evidence snapshot changed before preparation")
    write(source_dir / "original_source_pins.json", raw_pins)
    feature = slice_features(values, ontology, parent, output)
    definition = dict(
        endpoint=ENDPOINT,
        canonical_quantity_id=QUANTITY,
        quantity_unit="L",
        query_mode="source_annotation",
        annotation_only=True,
        validated_physiology=False,
        context_group=CONTEXT_GROUP,
        context_fields=list(CONTEXT_FIELDS),
        implementation_module="peptide_modeling_vd_absolute_native_bridge",
        implementation_sha256=sha(__file__),
        molecular_groups=10,
        independent_scaffolds_claimed=False,
        mandatory_molecular_reporting=True,
        mandatory_float64_fixed_controls=True,
        mandatory_all27_refit=True,
        primary_inner_cv_role="cross_estimand_extrapolation_stress_test_for_selection",
        each_estimand_independently_fivefold_validated=False,
        all27_refit_creates_no_new_molecular_generalization_support=True,
        pretraining_fold_state="local_precheck_only_node23_actual_receipt_required",
        source_review_state="independent_final_review_binding_required",
        warning=WARNING,
    )
    proof_names = [
        table_path,
        "task_tables/predeclared_splits.json",
        "task_tables/semantic_context_contract.json",
        "task_tables/observation_context_provenance.json",
        "source_provenance/" + SOURCE_FILE,
        "source_provenance/typed_context_corrections.json",
        "source_provenance/original_source_pins.json",
        *[c["split_path"] for c in checks["seeds"]],
        "feature_delivery_receipt.json",
    ]
    proof_path = "task_tables/vd_native_source_proof.json"
    proof = dict(
        schema=SCHEMA,
        task_id=TASK,
        implementation_sha256=sha(__file__),
        source_contract_sha256=SOURCE_SHA,
        rows=27,
        graphs=10,
        molecular_groups=10,
        original_source_pins=raw_pins,
        row_hashes={r["record_id"]: object_hash(r) for r in values},
        pinned_artifacts=[dict(path=n, sha256=sha(output / n)) for n in proof_names],
    )
    write(output / proof_path, proof)
    task = dict(
        task_id=TASK,
        endpoint=ENDPOINT,
        task_kind="regression",
        status="prepared_pending_source_review_and_nodefold",
        table_path=table_path,
        n_records=27,
        n_groups=10,
        unit="L",
        target_transform="log10",
        inverse_transform="pow10",
        scientific_definition=definition,
        task_definition=definition,
        native_source_observation_coverage=dict(
            proof_path=proof_path, proof_sha256=sha(output / proof_path)
        ),
    )
    write(
        output / "task_manifest.json",
        dict(
            schema=SCHEMA,
            tasks=[task],
            models_fitted=0,
            execution_gate="review_and_nodefold_required",
        ),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    table(output / "feature_rows_index.tsv", values)
    config = dict(
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        additional_twenty_hour_loop=False,
        require_complete_feature_bank=False,
        training=dict(
            routes=list(feature["routes"]),
            models=MODELS,
            inner_folds=5,
            min_rows=5,
            min_groups=5,
            min_train_rows=6,
            test_size=0.2,
            seed=42,
            search_backend="optuna",
            search_budget=128,
            n_jobs=1,
            refit_full=True,
            coverage_cohorts=False,
            svm_max_features=16384,
            histgb_max_features=16384,
            elasticnet_max_features=16384,
        ),
    )
    write(output / "config.json", config)
    write(output / "recommended_training_config.json", config)
    write(
        output / "deployment_requirements.json",
        dict(
            wrapper_entrypoint="scripts/peptide_modeling_vd_absolute_native_bridge.py",
            wrapper_arguments=["--run-followup", "--config", "<followup.json>"],
            required_before_any_fit=[
                "independent_source_review_binding.json",
                "node23_actual_no_fit_fold_receipt.json",
            ],
            explicit_modules=[
                "peptide_modeling_vd_absolute_native_bridge",
                "peptide_modeling_followup_seeds",
                "peptide_modeling_train",
            ],
            actual_node_python="/usr/bin/python3",
            node_command_requires_minus_B=True,
            fixed_controls="three_designs_float64_RidgeLSQR_alpha1_run_phase_only",
            all27_refit_independent_score=False,
            generic_direct_train_entry_forbidden=True,
        ),
    )
    report = preparation_review(output)
    write(output / "preparation_review.json", report)
    (output / "RUNBOOK.md").write_text(
        "# Typed native absolute-volume source annotation\n\n" + WARNING + "\n\n"
        "Prepared only; zero fits. Root must bind independent review and genuine node23 no-fit "
        "fold evidence before invoking the wrapper. Do not invoke the generic trainer directly. "
        "Frozen source fields are provenance; male/healthy/patients are not matrix features. "
        "Every inference requires the exact estimand and fitted-role joint context support. "
        "All27 refit is separate from outer-train-only evaluation and has no independent score.\n",
        encoding="utf-8",
    )
    return report


def bind_independent_review(prepared, review_path):
    """Explicitly bind a final independent review; a draft contract is never approval."""
    root = Path(prepared)
    preparation_review(root)
    if sha(review_path) != REVIEW_SHA:
        raise ValueError("Independent final review SHA changed")
    review = read(review_path)
    allowed = review.get("allowed_record_ids", review.get("candidate_record_ids", []))
    if (
        review.get("source_contract_sha256") != SOURCE_SHA
        or review.get("decision") != "qualified_approve_narrow_typed_source_annotation_candidate"
        or review.get("scientific_candidate_approved") is not True
        or review.get("blocked_record_ids") != []
        or review.get("blocking_after_required_context_roles_adopted") is not False
        or review.get("typed_context_sha256") != CORRECTIONS_SHA
        or set(allowed) != {r["record_id"] for r in source_rows(root / "source_provenance")[0]}
        or len(allowed) != 27
    ):
        raise ValueError("Final independent review must explicitly permit these exact27 IDs")
    destination = root / "independent_source_review.json"
    if destination.exists():
        raise FileExistsError("Review binding is immutable; use a new prepared version")
    shutil.copyfile(review_path, destination)
    write(
        root / "independent_source_review_binding.json",
        dict(
            status="bound_narrow_source_annotation_only",
            review_path=destination.name,
            review_sha256=sha(destination),
            source_contract_sha256=SOURCE_SHA,
            task_id=TASK,
            allowed_record_ids=allowed,
            implementation_sha256=sha(__file__),
            malformed_matrix_corrected=True,
            statistical_dispersion_not_inferred=True,
        ),
    )
    activate_execution_gate(root)
    return read(root / "independent_source_review_binding.json")


def activate_execution_gate(root):
    """A generic trainer cannot admit this prepared task before both concrete gates."""
    root = Path(root)
    review_path = root / "independent_source_review_binding.json"
    node_path = root / "nodefold_binding.json"
    if not review_path.exists() or not node_path.exists():
        return False
    if not (root / "post_adoption_round_staging_binding.json").exists():
        return False
    review, node = read(review_path), read(node_path)
    if (
        review.get("status") != "bound_narrow_source_annotation_only"
        or node.get("status") != "node23_actual_no_fit_frozen"
        or review.get("implementation_sha256") != sha(__file__)
        or node.get("producer_sha256") != sha(__file__)
        or sha(contained(root, review["review_path"])) != review["review_sha256"]
        or sha(contained(root, node["receipt_path"])) != node["receipt_sha256"]
    ):
        raise ValueError("A bound independent review or actual node-fold gate changed")
    post = read(root / "post_adoption_round_staging_binding.json")
    if (
        post.get("status") != "actual_node23_postadoption_round_closure_verified"
        or sha(contained(root, post["path"])) != post["sha256"]
    ):
        raise ValueError("Actual post-adoption round closure receipt changed")
    manifest = read(root / "task_manifest.json")
    task = manifest["tasks"][0]
    task["status"] = "eligible"
    for name in ("scientific_definition", "task_definition"):
        task[name]["source_review_state"] = "independent_review_bound_with_context_corrections"
        task[name]["pretraining_fold_state"] = "node23_actual_no_fit_frozen"
    manifest["execution_gate"] = "both_concrete_gates_bound_before_any_fit"
    write(root / "task_manifest.json", manifest)
    write(root / "targeted_task_manifest.json", manifest)
    return True


def nofit_nodefold_probe(prepared, receipt_output, *, require_node=True):
    """Freeze actual host split memberships under fit/search/pickle/import tripwires."""
    import importlib.abc
    import inspect
    import platform
    import tempfile

    from peptide_modeling_train import freeze_outer_split

    hostname = socket.gethostname()
    if require_node and (
        hostname != "node23"
        or sys.executable != "/usr/bin/python3"
        or sys.flags.dont_write_bytecode != 1
    ):
        raise ValueError("Actual node23 /usr/bin/python3 -B is required for execution evidence")
    counters = dict(
        estimator_fit_attempts=0,
        search_attempts=0,
        model_unpickle_attempts=0,
        chemistry_import_attempts=0,
        production_fits_launched=0,
    )

    class BlockChemistry(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "rdkit" or fullname.startswith("rdkit."):
                counters["chemistry_import_attempts"] += 1
                raise RuntimeError("Chemistry imports forbidden in the node-fold no-fit probe")
            return None

    def profile(frame, event, arg):
        if event != "call":
            return
        name, filename = frame.f_code.co_name, frame.f_code.co_filename.replace("\\", "/")
        if name in {"fit", "fit_transform", "_fit_model", "_cv_predictions", "train_task"}:
            counters["estimator_fit_attempts"] += 1
            raise RuntimeError("A model-fit call entered the no-fit probe")
        if name == "search_candidates":
            counters["search_attempts"] += 1
            raise RuntimeError("Optuna/search calls forbidden in the no-fit probe")
        if name in {"load", "loads"} and any(x in filename for x in ("joblib", "pickle")):
            counters["model_unpickle_attempts"] += 1
            raise RuntimeError("Model unpickling forbidden in the no-fit probe")

    blocker, previous = BlockChemistry(), sys.getprofile()
    sys.meta_path.insert(0, blocker)
    sys.setprofile(profile)
    try:
        root = Path(prepared)
        before = {
            str(p.relative_to(root)): sha(p)
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        reviewed = preparation_review(root)
        values = rows(root / read(root / "task_manifest.json")["tasks"][0]["table_path"])
        with tempfile.TemporaryDirectory(prefix="vd_absolute_nodefold_") as folder:
            checks = split_prechecks(values, Path(folder))
            outer = {
                str(seed["seed"]): read(Path(folder) / seed["split_path"])
                for seed in checks["seeds"]
            }
            staged = []
            settings = read(root / "config.json")
            for seed in SEEDS:
                destination = Path(folder) / f"staged_round_seed_{seed}"
                _, evidence = stage_seed(root, destination, seed, outer[str(seed)], settings)
                staged_values = rows(
                    destination / read(destination / "task_manifest.json")["tasks"][0]["table_path"]
                )
                check_root = Path(folder) / f"rederived_staged_plans_{seed}"
                staged_checks = split_prechecks(staged_values, check_root)
                if staged_checks != checks:
                    raise ValueError("Actual staged-round fold plans differ from template plans")
                evidence["actual_staged_outer_and_inner_folds_rederived"] = True
                staged.append(evidence)
        after = {
            str(p.relative_to(root)): sha(p)
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        if before != after or any(counters.values()):
            raise ValueError("No-fit probing changed input bytes or crossed a guarded boundary")
        import peptide_modeling_train as training
        import sklearn
        from threadpoolctl import threadpool_info

        pools = [
            {k: v for k, v in pool.items() if k in {"internal_api", "num_threads", "version"}}
            for pool in threadpool_info()
        ]
        if require_node and any(pool.get("num_threads") != 1 for pool in pools):
            raise ValueError("Node no-fit prefreeze requires one thread in every loaded BLAS pool")

        receipt = dict(
            schema=SCHEMA,
            probe_scope="actual_node23_pretraining_no_fit"
            if require_node
            else "local_test_only_not_deployment_evidence",
            hostname=hostname,
            python_executable=sys.executable,
            dont_write_bytecode=sys.flags.dont_write_bytecode,
            python_version=list(sys.version_info[:3]),
            python_build=platform.python_build(),
            numpy_version=np.__version__,
            sklearn_version=sklearn.__version__,
            producer_sha256=sha(__file__),
            training_source_sha256=sha(training.__file__),
            outer_splitter_sha256=hashlib.sha256(
                inspect.getsource(freeze_outer_split).encode()
            ).hexdigest(),
            payload_source_review=reviewed,
            actual_checks=checks,
            outer_split_receipts=outer,
            input_hashes_before=before,
            input_hashes_after=after,
            guard_counters=counters,
            frozen_targets_rewritten=False,
            report_only_estimator_fits=0,
            source_contract_sha256=SOURCE_SHA,
            task_id=TASK,
            staged_seed_closure_reviews=staged,
            source_closure_sha256=sha(root / "source_staging_closure.json"),
            runtime_receipts_present_in_template=[
                name
                for name in verify_source_closure(root)["runtime_receipts_required_when_bound"]
                if (root / name).is_file()
            ],
            threadpools=pools,
        )
    finally:
        sys.setprofile(previous)
        sys.meta_path.remove(blocker)
    write(receipt_output, receipt)
    return receipt


def adopt_nodefold_receipt(prepared, receipt_path):
    root, receipt = Path(prepared), read(receipt_path)
    reviewed = preparation_review(root)
    if (
        receipt.get("probe_scope") != "actual_node23_pretraining_no_fit"
        or receipt.get("hostname") != "node23"
        or receipt.get("python_executable") != "/usr/bin/python3"
        or receipt.get("dont_write_bytecode") != 1
        or receipt.get("producer_sha256") != sha(__file__)
        or receipt.get("payload_source_review") != reviewed
        or receipt.get("input_hashes_before") != receipt.get("input_hashes_after")
        or set(receipt.get("guard_counters", {}))
        != {
            "estimator_fit_attempts",
            "search_attempts",
            "model_unpickle_attempts",
            "chemistry_import_attempts",
            "production_fits_launched",
        }
        or any(receipt["guard_counters"].values())
    ):
        raise ValueError("Only genuine exact-payload node23 no-fit evidence can be adopted")
    checks = receipt["actual_checks"]
    validate_staging_receipt(root, receipt, require_bound_receipts=False)
    values = rows(root / read(root / "task_manifest.json")["tasks"][0]["table_path"])
    validate_checks(values, checks)
    if (root / "node23_actual_no_fit_fold_receipt.json").exists():
        raise FileExistsError("Node-fold binding is immutable")
    shutil.copyfile(receipt_path, root / "node23_actual_no_fit_fold_receipt.json")
    # Local source/table targets remain byte-identical; node memberships are kept
    # in a separate receipt and placed into each future round before any fitting.
    write(
        root / "nodefold_binding.json",
        dict(
            status="node23_actual_no_fit_frozen",
            receipt_path="node23_actual_no_fit_fold_receipt.json",
            receipt_sha256=sha(root / "node23_actual_no_fit_fold_receipt.json"),
            producer_sha256=sha(__file__),
            source_proof_sha256=reviewed["source_proof_sha256"],
            frozen_targets_rewritten=False,
        ),
    )
    activate_execution_gate(root)
    return read(root / "nodefold_binding.json")


def validate_staging_receipt(root, receipt, *, require_bound_receipts):
    root = Path(root)
    closure = verify_source_closure(root)
    staged = receipt.get("staged_seed_closure_reviews", [])
    if (
        receipt.get("source_closure_sha256") != sha(root / "source_staging_closure.json")
        or len(staged) != 3
        or [x.get("seed") for x in staged] != list(SEEDS)
    ):
        raise ValueError("All three actual staged source-closure reviews are mandatory")
    expected = preparation_review(root)
    required = {"nodefold_binding.json", "node23_actual_no_fit_fold_receipt.json"}
    for item in staged:
        if (
            item.get("status") != "verified_no_fit_staged_source_closure"
            or item.get("source_closure_sha256") != receipt["source_closure_sha256"]
            or item.get("source_closure_files_checked") != len(closure["files"])
            or item.get("exact_source_proof_sha256") != expected["source_proof_sha256"]
            or item.get("exact_task_table_sha256") != expected["task_table_sha256"]
            or item.get("portable_source_and_all_routes_passed") is not True
            or item.get("actual_staged_outer_and_inner_folds_rederived") is not True
            or any(
                item.get(k) != 0
                for k in ("estimator_fits", "searches", "model_unpickles", "chemistry_imports")
            )
        ):
            raise ValueError("Actual staged round has incomplete portable source/fold proof")
        if require_bound_receipts:
            if not required <= set(item.get("runtime_receipts_copied", {})):
                raise ValueError("Post-adoption round lost actual node receipts")
            if any(item["runtime_receipts_copied"][name] != sha(root / name) for name in required):
                raise ValueError("Post-adoption round receipt bytes do not match template")


def adopt_postadoption_staging(prepared, receipt_path):
    root, receipt = Path(prepared), read(receipt_path)
    first = read(root / "node23_actual_no_fit_fold_receipt.json")
    if (
        receipt.get("probe_scope") != "actual_node23_pretraining_no_fit"
        or receipt.get("hostname") != "node23"
        or receipt.get("python_executable") != "/usr/bin/python3"
        or receipt.get("dont_write_bytecode") != 1
        or receipt.get("producer_sha256") != sha(__file__)
        or receipt.get("actual_checks") != first["actual_checks"]
        or receipt.get("outer_split_receipts") != first["outer_split_receipts"]
        or receipt.get("guard_counters") != first["guard_counters"]
        or any(receipt.get("guard_counters", {}).values())
    ):
        raise ValueError("Genuine same-runtime zero-fit post-adoption evidence is required")
    validate_staging_receipt(root, receipt, require_bound_receipts=True)
    destination = root / "post_adoption_round_staging_receipt.json"
    if destination.exists():
        raise FileExistsError("Post-adoption closure receipt binding is immutable")
    shutil.copyfile(receipt_path, destination)
    write(
        root / "post_adoption_round_staging_binding.json",
        dict(
            status="actual_node23_postadoption_round_closure_verified",
            path=destination.name,
            sha256=sha(destination),
            producer_sha256=sha(__file__),
            source_closure_sha256=sha(root / "source_staging_closure.json"),
            actual_seed_staging_reviews=3,
            estimator_fits=0,
        ),
    )
    activate_execution_gate(root)
    preparation_review(root)
    return read(root / "post_adoption_round_staging_binding.json")


def validate_checks(values, checks):
    by_id = {r["record_id"]: r for r in values}
    if len(checks["seeds"]) != 3 or [s["seed"] for s in checks["seeds"]] != list(SEEDS):
        raise ValueError("Exactly the three required seed plans are mandatory")
    for seed in checks["seeds"]:
        train, test = set(seed["train_record_ids"]), set(seed["holdout_record_ids"])
        if (
            train | test != set(by_id)
            or train & test
            or len(test) < 4
            or {by_id[i]["split_group"] for i in train} & {by_id[i]["split_group"] for i in test}
        ):
            raise ValueError("Frozen outer holdout membership is invalid")
        seen = []
        if len(seed["folds"]) != 5 or seed["actual_folds"] != 5:
            raise ValueError("Exactly five actual inner folds required")
        for fold in seed["folds"]:
            a, b = set(fold["train_record_ids"]), set(fold["validation_record_ids"])
            if (
                a | b != train
                or a & b
                or {by_id[i]["split_group"] for i in a} & {by_id[i]["split_group"] for i in b}
            ):
                raise ValueError("Inner fold changed membership or leaks a graph/family group")
            seen.extend(b)
        if sorted(seen) != sorted(train) or seed["final_refit_record_ids"] != list(by_id):
            raise ValueError("OOF/refit plan does not preserve exact source coverage")


def verify_ready_for_execution(prepared):
    root = Path(prepared)
    reviewed = preparation_review(root)
    binding = read(root / "independent_source_review_binding.json")
    if (
        binding.get("status") != "bound_narrow_source_annotation_only"
        or binding.get("implementation_sha256") != sha(__file__)
        or binding.get("source_contract_sha256") != SOURCE_SHA
        or sha(contained(root, binding["review_path"])) != binding["review_sha256"]
    ):
        raise ValueError("Independent final source review binding missing or changed")
    node_binding = read(root / "nodefold_binding.json")
    receipt_path = contained(root, node_binding["receipt_path"])
    if (
        sha(receipt_path) != node_binding["receipt_sha256"]
        or node_binding["source_proof_sha256"] != reviewed["source_proof_sha256"]
    ):
        raise ValueError("Actual node-fold binding missing or changed")
    receipt = read(receipt_path)
    post_binding = read(root / "post_adoption_round_staging_binding.json")
    post_path = contained(root, post_binding["path"])
    if sha(post_path) != post_binding["sha256"] or post_binding.get("producer_sha256") != sha(
        __file__
    ):
        raise ValueError("Actual post-adoption staged closure evidence changed")
    validate_staging_receipt(root, read(post_path), require_bound_receipts=True)
    import peptide_modeling_train as training
    import sklearn

    if (
        socket.gethostname() != "node23"
        or sys.executable != "/usr/bin/python3"
        or receipt["producer_sha256"] != sha(__file__)
        or receipt["training_source_sha256"] != sha(training.__file__)
        or receipt["numpy_version"] != np.__version__
        or receipt["sklearn_version"] != sklearn.__version__
    ):
        raise ValueError("Training runtime/source differs from actual frozen node runtime")
    return receipt


def error_metrics(target, prediction, groups):
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite aligned prediction evidence required")
    error = np.abs(y - p)

    def metric(e, weights=None):
        scale = float(np.max(e))
        normalized = e / scale if scale else e
        return dict(
            mae=float(np.average(e, weights=weights)),
            rmse=scale * math.sqrt(float(np.average(normalized**2, weights=weights))),
        )

    count = Counter(groups)
    w = np.asarray([1 / count[g] for g in groups], dtype=np.float64)
    return dict(
        record_weighted=metric(error),
        molecule_equal=metric(error, w),
        record_counts_per_molecule=dict(count),
        molecular_groups=len(count),
    )


def prediction_consistency(predictions, evidence, provenance):
    """The existing bounded32ULP float32 LSQR policy; never broaden or mutate values."""
    p = [float(r["prediction"]) for r in predictions]
    if not p or not all(math.isfinite(v) for v in p):
        raise ValueError("Empty/nonfinite identical-input prediction evidence")
    bound = [evidence["record_rows"][r["record_id"]] for r in predictions]
    if (
        len({(r["graph"], r["group"], r["context"], r["row_sha256"]) for r in bound}) != 1
        or len({(r["evaluation_role"], int(r["fold"])) for r in predictions}) != 1
    ):
        raise ValueError("Consistency needs identical graph/group/context/full-input/fold")
    if len(set(p)) == 1:
        return dict(mode="exact", raw_predictions_unchanged=True)
    needed = dict(
        selected_model="linear",
        actual_estimator="Ridge",
        actual_solver="lsqr",
        actual_coef_dtype="float32",
        actual_scaler_with_mean=False,
        actual_imputer_strategy="median",
        actual_add_indicator=True,
        input_dtype="float32",
    )
    needed.update(
        feature_width=evidence["feature_width"], full_matrix_sha256=evidence["matrix_sha256"]
    )
    from peptide_modeling_train import __file__ as training_file

    if (
        any(provenance.get(k) != v for k, v in needed.items())
        or provenance.get("training_source_sha256") != sha(training_file)
        or not provenance.get("actual_model_artifact_sha256")
    ):
        raise ValueError("No bounded float32 LSQR exception without actual saved-model provenance")
    bits = []
    for value in p:
        if abs(value) < 2**-126:
            raise ValueError("Subnormal/zero cannot use float32 LSQR exception")
        packed = struct.pack("!f", value)
        if struct.unpack("!f", packed)[0] != value:
            raise ValueError("Bounded exception requires exact binary32 prediction values")
        n = struct.unpack("!I", packed)[0]
        bits.append(0x80000000 - (n & 0x7FFFFFFF) if n >> 31 else 0x80000000 + n)
    if len({math.copysign(1, v) for v in p}) != 1 or max(bits) - min(bits) > 32:
        raise ValueError("Same-input LSQR discrepancy exceeds unchanged32ULP bound")
    return dict(
        mode="exact_binary32_LSQR_at_most32ULP",
        maximum_binary32_ulp_budget=32,
        observed_binary32_ulp_distance=max(bits) - min(bits),
        raw_predictions_unchanged=True,
    )


def evaluation_diagnostics(values, inner, holdout, check, payload, provenance):
    by_id = {r["record_id"]: r for r in values}
    X = np.asarray(payload["X"])
    if (
        payload["record_ids"] != list(by_id)
        or X.dtype != np.float32
        or not payload["valid"].all()
        or np.isinf(X).any()
    ):
        raise ValueError("Selected numeric evidence must bind exact stored float32 inputs")
    full_rows = {
        r["record_id"]: dict(
            graph=r["canonical_identity_group"],
            group=r["split_group"],
            context=r["semantic_context_key"],
            row_sha256=digest_array(X[i]),
        )
        for i, r in enumerate(values)
    }
    evidence = dict(record_rows=full_rows, feature_width=X.shape[1], matrix_sha256=digest_array(X))
    report, fold_by_id = {}, {}
    for fold in check["folds"]:
        for rid in fold["validation_record_ids"]:
            fold_by_id[rid] = fold
    for role, records, expected in [
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", holdout, check["holdout_record_ids"]),
    ]:
        if len(records) != len(expected) or {r["record_id"] for r in records} != set(expected):
            raise ValueError("Exact fit-role prediction coverage is mandatory")
        enriched, duplicates = [], defaultdict(list)
        for row in records:
            original = by_id[row["record_id"]]
            fit_ids = (
                check["train_record_ids"]
                if role == "outer_holdout"
                else fold_by_id[row["record_id"]]["train_record_ids"]
            )
            fold = -1 if role == "outer_holdout" else fold_by_id[row["record_id"]]["fold"]
            if (
                int(row["fold"]) != fold
                or row["split_group"] != original["split_group"]
                or float(row["target"]) != float(original["target"])
            ):
                raise ValueError("Prediction row changed target/group/frozen inner fold")
            enriched.append(
                dict(
                    record_id=row["record_id"],
                    evaluation_role=role,
                    fold=fold,
                    source_estimand=original["source_estimand"],
                    context_support=context_support(
                        json.loads(original["semantic_context_json"]), [by_id[i] for i in fit_ids]
                    ),
                    target_log10_L=float(row["target"]),
                    prediction_log10_L=float(row["prediction"]),
                )
            )
            b = full_rows[row["record_id"]]
            duplicates[(b["graph"], b["context"], b["row_sha256"], fold)].append(
                dict(row, evaluation_role=role)
            )
        consistency = [prediction_consistency(v, evidence, provenance) for v in duplicates.values()]
        result = error_metrics(
            [r["target"] for r in records],
            [r["prediction"] for r in records],
            [r["split_group"] for r in records],
        )
        by_type = {}
        for typ in sorted({r["source_estimand"] for r in enriched}):
            rr = [r for r in records if by_id[r["record_id"]]["source_estimand"] == typ]
            by_type[typ] = error_metrics(
                [r["target"] for r in rr],
                [r["prediction"] for r in rr],
                [r["split_group"] for r in rr],
            )
        report[role] = dict(
            **result,
            rows=enriched,
            per_estimand=by_type,
            unchanged_prediction_consistency=consistency,
            unsupported_estimand_record_ids=[
                r["record_id"] for r in enriched if not r["context_support"]["estimand_seen"]
            ],
            unsupported_joint_context_record_ids=[
                r["record_id"] for r in enriched if not r["context_support"]["joint_context_seen"]
            ],
        )
    return dict(
        evaluation_by_role=report,
        source_annotation_only=True,
        full_refit_scores_not_external=True,
        raw_predictions_mutated=False,
    )


def fixed_control_estimator():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )


def fixed_control_trace(
    estimator, fit_X, fit_y, predict_X, prediction, train_ids, predict_ids, *, role, fold
):
    steps, pred = estimator.named_steps, np.asarray(prediction)
    model, coef = steps["model"], np.asarray(steps["model"].coef_)
    parameters = dict(
        alpha=float(model.alpha),
        solver=model.solver,
        imputer_strategy=steps["impute"].strategy,
        imputer_add_indicator=steps["impute"].add_indicator,
        imputer_keep_empty_features=steps["impute"].keep_empty_features,
        scaler_with_mean=steps["scale"].with_mean,
    )
    if (
        type(model).__name__ != "Ridge"
        or parameters
        != dict(
            alpha=1.0,
            solver="lsqr",
            imputer_strategy="median",
            imputer_add_indicator=True,
            imputer_keep_empty_features=True,
            scaler_with_mean=False,
        )
        or any(a.dtype != np.float64 for a in (fit_X, fit_y, predict_X, pred, coef))
        or not np.isfinite(pred).all()
        or not np.isfinite(coef).all()
    ):
        raise ValueError("Actual fixed-control fit/predict/coef must be float64 RidgeLSQR")
    return dict(
        fit_role=role,
        fold=fold,
        train_record_ids=train_ids,
        predict_record_ids=predict_ids,
        actual_estimator=type(model).__name__,
        actual_parameters=parameters,
        actual_fit_X_dtype=str(fit_X.dtype),
        actual_fit_y_dtype=str(fit_y.dtype),
        actual_predict_X_dtype=str(predict_X.dtype),
        actual_prediction_dtype=str(pred.dtype),
        actual_coef_dtype=str(coef.dtype),
        actual_coef_sha256=digest_array(coef),
        actual_coef_shape=list(coef.shape),
        fit_X_sha256=digest_array(fit_X),
        fit_y_sha256=digest_array(fit_y),
        predict_X_sha256=digest_array(predict_X),
        predictions_sha256=digest_array(pred),
        control_fit_calls=1,
        selected_model_fit_calls=0,
        search_calls=0,
    )


def fixed_ridge_controls(values, chemical_X, context_X, check):
    """Explicit run phase only:5OOF+outer-train+all27 fits per fixed float64 design."""
    from threadpoolctl import threadpool_limits

    chemical, context = np.asarray(chemical_X), np.asarray(context_X)
    if (
        chemical.dtype != np.float32
        or context.dtype != np.float32
        or len(chemical) != 27
        or len(context) != 27
        or not chemical.shape[1]
        or not context.shape[1]
    ):
        raise ValueError("Controls must start from exact stored float32 design values")
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    ids = [r["record_id"] for r in values]
    position = {rid: i for i, rid in enumerate(ids)}
    report = {}
    for name, stored in dict(
        molecule_only=chemical,
        context_only=context,
        molecule_plus_context=np.column_stack((chemical, context)),
    ).items():
        # Precision conversion happens before any estimator sees this design.
        X = np.asarray(stored, dtype=np.float64)
        plans = [
            ("inner_training_fold", f["fold"], f["train_record_ids"], f["validation_record_ids"])
            for f in check["folds"]
        ]
        plans += [
            ("outer_train_only", -1, check["train_record_ids"], check["holdout_record_ids"]),
            ("all27_training_inputs_only_no_independent_score", -2, ids, ids),
        ]
        traces, predictions = [], []
        for role, fold, fit_ids, predict_ids in plans:
            a, b = [position[i] for i in fit_ids], [position[i] for i in predict_ids]
            model = fixed_control_estimator()
            with threadpool_limits(limits=1):
                model.fit(X[a], y[a])
                part = np.asarray(model.predict(X[b]))
            traces.append(
                fixed_control_trace(
                    model, X[a], y[a], X[b], part, fit_ids, predict_ids, role=role, fold=fold
                )
            )
            predictions.append(
                dict(fit_role=role, fold=fold, record_ids=predict_ids, predictions=part.tolist())
            )
        report[name] = dict(
            model="fixed_Ridge_alpha1_LSQR_not_Optuna_selected",
            dtype_from_initial_boundary="float64",
            actual_fit_calls=7,
            search_calls=0,
            selected_model_fit_calls=0,
            exact_upcast_of_stored_float32_design=True,
            source_encoded_values_rewritten=False,
            stored_matrix_sha256=digest_array(stored),
            matrix_sha256=digest_array(X),
            per_fit_trace=traces,
            predictions=predictions,
            full_refit_evaluation_role="training_inputs_only_no_independent_score",
        )
    return report


def validate_controls(values, chemical_X, context_X, check, controls):
    """No estimator recreation: validate saved numeric traces and exact fit memberships."""
    if set(controls) != {"molecule_only", "context_only", "molecule_plus_context"}:
        raise ValueError("Exactly the three fixed designs are required")
    ids = [r["record_id"] for r in values]
    pos = {rid: i for i, rid in enumerate(ids)}
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    stored_designs = dict(
        molecule_only=chemical_X,
        context_only=context_X,
        molecule_plus_context=np.column_stack((chemical_X, context_X)),
    )
    role_metrics = {}
    for name, saved in controls.items():
        X = np.asarray(stored_designs[name], dtype=np.float64)
        if (
            saved["matrix_sha256"] != digest_array(X)
            or saved["stored_matrix_sha256"] != digest_array(stored_designs[name])
            or saved["dtype_from_initial_boundary"] != "float64"
            or saved["actual_fit_calls"] != 7
            or len(saved["per_fit_trace"]) != 7
            or len(saved["predictions"]) != 7
            or saved["search_calls"] != 0
        ):
            raise ValueError("Fixed designs/dtypes/actual fit counts changed")
        plans = [
            ("inner_training_fold", f["fold"], f["train_record_ids"], f["validation_record_ids"])
            for f in check["folds"]
        ]
        plans += [
            ("outer_train_only", -1, check["train_record_ids"], check["holdout_record_ids"]),
            ("all27_training_inputs_only_no_independent_score", -2, ids, ids),
        ]
        for plan, trace, item in zip(
            plans, saved["per_fit_trace"], saved["predictions"], strict=True
        ):
            role, fold, a_ids, b_ids = plan
            a, b = [pos[i] for i in a_ids], [pos[i] for i in b_ids]
            p = np.asarray(item["predictions"], dtype=np.float64)
            expected = dict(
                fit_role=role,
                fold=fold,
                train_record_ids=a_ids,
                predict_record_ids=b_ids,
                actual_estimator="Ridge",
                actual_fit_X_dtype="float64",
                actual_fit_y_dtype="float64",
                actual_predict_X_dtype="float64",
                actual_prediction_dtype="float64",
                actual_coef_dtype="float64",
                fit_X_sha256=digest_array(X[a]),
                fit_y_sha256=digest_array(y[a]),
                predict_X_sha256=digest_array(X[b]),
                predictions_sha256=digest_array(p),
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
            if (
                any(trace.get(k) != v for k, v in expected.items())
                or item["record_ids"] != b_ids
                or item["fit_role"] != role
                or item["fold"] != fold
                or not trace.get("actual_coef_sha256")
                or not trace.get("actual_coef_shape")
                or not np.isfinite(p).all()
                or p.shape != (len(b),)
            ):
                raise ValueError("Actual float64 control membership/dtype/parameter trace changed")
            identical = defaultdict(list)
            for row, value in zip(X[b], p, strict=True):
                identical[row.tobytes()].append(value)
            if any(not np.allclose(v, v[0], rtol=1e-7, atol=1e-8) for v in identical.values()):
                raise ValueError("Float64 controls fail unchanged strict identical-input tolerance")
        by_id = {r["record_id"]: r for r in values}
        inner_ids = [rid for item in saved["predictions"][:5] for rid in item["record_ids"]]
        inner_p = [p for item in saved["predictions"][:5] for p in item["predictions"]]
        outer_item = saved["predictions"][5]
        role_metrics[name] = {
            role: error_metrics(
                [float(by_id[r]["target"]) for r in members],
                prediction,
                [by_id[r]["split_group"] for r in members],
            )
            for role, members, prediction in [
                ("inner_oof_stress_test", inner_ids, inner_p),
                ("outer_holdout_stress_test", outer_item["record_ids"], outer_item["predictions"]),
            ]
        }
    return dict(
        status="verified_numeric_traces_no_refit",
        actual_control_fit_calls=21,
        fixed_chemical_route="structure_methods_all",
        role_metrics=role_metrics,
        full_cohort_metric_not_reported=True,
    )


def artifact_path(root, raw):
    path = Path(raw)
    return (
        contained(root, path.relative_to(Path(root).resolve()).as_posix())
        if path.is_absolute()
        else contained(root, raw)
    )


def selected_solver_provenance(model, result, payload):
    """Called in explicit run phase, never from metadata-only reporting."""
    from peptide_modeling_train import __file__ as training_file

    out = dict(
        training_source_sha256=sha(training_file),
        selected_model=result["selected_model"],
        input_dtype=str(payload["X"].dtype),
        feature_width=payload["X"].shape[1],
        full_matrix_sha256=digest_array(payload["X"]),
        actual_model_artifact_sha256=sha(result["artifacts"]["model"]),
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


def write_report(prepared, *, run_dir=None, seed=None):
    """Pure metadata/numeric report; no chemistry, estimator creation or model unpickling."""
    root = Path(run_dir or prepared)
    review = preparation_review(root)
    if run_dir is None:
        return dict(status="prepared_no_fit", preparation=review, warning=WARNING)
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features
    from peptide_modeling_summarize import verified_predictions

    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(root / task["table_path"])
    result_path = root / "results/full" / (TASK + "_common.json")
    result = read(result_path)
    if (
        result.get("status") != "completed"
        or result.get("actual_inner_folds") != 5
        or result.get("requested_inner_folds") != 5
        or result.get("search", {}).get("backend") != "optuna"
        or result.get("refit_status") != "completed"
    ):
        raise ValueError("Actual Optuna/strict5/all27 separate refit completion is required")
    verified_predictions(root, root, result, task)
    paths = {k: artifact_path(root, v) for k, v in result["artifacts"].items()}
    contract = read(paths["model_contract"])
    all_ids = [r["record_id"] for r in values]
    if (
        contract["common_record_ids"] != all_ids
        or contract["model_fit_scope"] != "outer_train_only"
        or contract["optional_refit_fit_scope"] != "all_common_after_evaluation"
        or not contract["holdout_not_used_for_selection"]
        or "refit_model" not in paths
        or len(all_ids) != 27
    ):
        raise ValueError("All27 refit membership/evaluation separation changed")
    node = read(root / "node23_actual_no_fit_fold_receipt.json")
    check = next(s for s in node["actual_checks"]["seeds"] if s["seed"] == seed)
    split = read(paths["outer_split"])
    if split != node["outer_split_receipts"][str(seed)]:
        raise ValueError("Actual evaluated holdout differs from node-frozen assignment")
    payload = load_feature_route(values, portable_features(root), result["selected_route"])
    if contract["feature_names"] != payload["feature_names"]:
        raise ValueError("Saved model schema changed chemical/context order")
    provenance_path = root / "run_phase_numeric_provenance.json"
    provenance = read(provenance_path)
    if (
        provenance["result_sha256"] != sha(result_path)
        or provenance["controls_sha256"] != sha(root / "fixed_float64_controls.json")
        or provenance["selected_solver"]["actual_model_artifact_sha256"] != sha(paths["model"])
        or provenance["selected_solver"]["full_matrix_sha256"] != digest_array(payload["X"])
    ):
        raise ValueError("Actual run-phase solver/control provenance missing or stale")
    diagnostics = evaluation_diagnostics(
        values,
        rows(paths["oof"]),
        rows(paths["holdout"]),
        check,
        payload,
        provenance["selected_solver"],
    )
    for role, key in [("inner_oof", "inner_cv"), ("outer_holdout", "holdout")]:
        for metric in ("mae", "rmse"):
            if not math.isclose(
                diagnostics["evaluation_by_role"][role]["record_weighted"][metric],
                result[key][metric],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError("Saved role metrics disagree with exact prediction evidence")
    ontology = read(root / "task_tables/semantic_context_contract.json")
    full = load_feature_route(values, portable_features(root), "structure_methods_all")
    width = len(ontology["feature_names"])
    controls = read(root / "fixed_float64_controls.json")
    control_validation = validate_controls(
        values, full["X"][:, :-width], full["X"][:, -width:], check, controls
    )
    report = dict(
        schema=SCHEMA,
        status="verified_typed_source_annotation_stress_test_only",
        task_id=TASK,
        seed=seed,
        required_seeds=list(SEEDS),
        warning=WARNING,
        result_sha256=sha(result_path),
        source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
        oof_sha256=sha(paths["oof"]),
        holdout_sha256=sha(paths["holdout"]),
        model_sha256=sha(paths["model"]),
        diagnostics=diagnostics,
        fixed_controls=dict(
            verification=control_validation,
            path="fixed_float64_controls.json",
            sha256=sha(root / "fixed_float64_controls.json"),
        ),
        separate_all27_refit=dict(
            status="verified",
            records=27,
            molecular_groups=10,
            training_record_ids=contract["common_record_ids"],
            model_sha256=sha(paths["refit_model"]),
            model_contract_sha256=sha(paths["model_contract"]),
            independently_scored=False,
            all_observation_context_support=check["all27_refit_support"],
            new_molecule_or_subtype_generalization_created=False,
        ),
        report_only_estimator_fits=0,
        model_selection_repeated=False,
        every_estimand_independently_validated=False,
    )
    write(root / "reports/vd_absolute_native_diagnostics.json", report)
    return report


def predict_contextual(
    model,
    chemical_X,
    chemical_names,
    contexts,
    model_contract,
    *,
    quantity_id,
    source_proof,
    training_record_ids,
    unsupported="raise",
):
    """Reject unseen and single-molecule estimands even after the separate full refit."""
    if quantity_id != QUANTITY or unsupported not in {"raise", "status"}:
        raise ValueError("Explicit typed source-annotation quantity/mode required")
    task = model_contract["task"]
    root = Path(source_proof).resolve().parents[1]
    values = rows(contained(root, task["table_path"]))
    verify_source_proof(root, task, values)
    if (
        task["task_id"] != TASK
        or sha(source_proof) != task["native_source_observation_coverage"]["proof_sha256"]
    ):
        raise ValueError("Model/source quantity or proof does not match this producer")
    by_id = {r["record_id"]: r for r in values}
    if (
        not training_record_ids
        or len(set(training_record_ids)) != len(training_record_ids)
        or not set(training_record_ids) <= set(by_id)
    ):
        raise ValueError("Actual fitted-role training membership must be supplied")
    fitted = [by_id[i] for i in training_record_ids]
    ontology = read(root / "task_tables/semantic_context_contract.json")
    names = [CONTEXT_GROUP + ":" + n for n in ontology["feature_names"]]
    X = np.asarray(chemical_X, dtype=np.float32)
    if (
        X.ndim != 2
        or len(X) != len(contexts)
        or not len(contexts)
        or X.shape[1] != len(chemical_names)
        or list(chemical_names) + names != model_contract["feature_names"]
        or np.isinf(X).any()
        or np.isnan(X).all(axis=1).any()
    ):
        raise ValueError("Exact chemical/context schema and usable graph vectors required")
    result, accepted, vectors = [], [], []
    for i, query in enumerate(contexts):
        try:
            vector = encode_context(query, ontology)
            support = context_support(query, fitted)
            if not support["estimand_seen"] or support["low_estimand_molecular_support"]:
                raise ValueError("Estimand is unseen or has fewer than two fitted molecular groups")
            if not support["joint_context_seen"]:
                raise ValueError("This fitted model has no exact typed joint-context support")
        except (ValueError, TypeError, KeyError) as exc:
            if unsupported == "raise":
                raise ValueError("Unsupported native annotation query: " + str(exc)) from exc
            result.append(
                dict(
                    status="unsupported",
                    reason=str(exc),
                    prediction_log10_L=None,
                    new_molecule_generalization_validated=False,
                )
            )
            continue
        accepted.append(i)
        vectors.append(vector)
        result.append(
            dict(
                status="source_annotation_only",
                quantity_id=QUANTITY,
                context_support=support,
                new_molecule_generalization_validated=False,
                physiological_or_subtype_validation=False,
            )
        )
    if accepted:
        predicted = model.predict(np.column_stack((X[accepted], np.vstack(vectors))))
        for i, p in zip(accepted, predicted, strict=True):
            if not math.isfinite(float(p)):
                raise ValueError("Nonfinite source-annotation prediction")
            result[i]["prediction_log10_L"] = float(p)
    return result


def predict_from_artifacts(
    result_path,
    chemical_X,
    chemical_names,
    contexts,
    *,
    quantity_id,
    source_proof,
    run_dir,
    use_refit=True,
    unsupported="raise",
):
    import joblib

    root = Path(run_dir).resolve()
    result = read(result_path)
    paths = {k: artifact_path(root, p) for k, p in result["artifacts"].items()}
    for key, path in paths.items():
        if sha(path) != result["artifact_hashes"][result["artifacts"][key]]["sha256"]:
            raise ValueError("Saved artifact changed before inference")
    contract = read(paths["model_contract"])
    if use_refit:
        if result.get("refit_status") != "completed":
            raise ValueError("All27 separate refit is unavailable")
        membership, path = contract["common_record_ids"], paths["refit_model"]
    else:
        membership = read(paths["outer_split"])["train_records"]
        path = paths["model"]
    return predict_contextual(
        joblib.load(path),
        chemical_X,
        chemical_names,
        contexts,
        contract,
        quantity_id=quantity_id,
        source_proof=source_proof,
        training_record_ids=membership,
        unsupported=unsupported,
    )


def run_followup(config_path):
    """Future explicit production entry; exactly3seeds after parent release, one CPU."""
    import time

    import joblib
    from peptide_modeling_campaign import process_identity
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_followup_seeds import parent_ready
    from peptide_modeling_pipeline import portable_features, report_stage, train_stage
    from peptide_modeling_summarize import summarize_run
    from peptide_modeling_supervise import archive_run

    cfg = read(config_path)
    template, output, durable = map(
        Path, (cfg["template_run"], cfg["output_root"], cfg["durable_root"])
    )
    node = verify_ready_for_execution(template)
    settings = read(cfg["template_config"])
    if (
        cfg.get("required_seeds", list(SEEDS)) != list(SEEDS)
        or cfg.get("task_ids") != [TASK]
        or cfg.get("workers", 1) != 1
        or cfg.get("additional_twenty_hour_loop", False)
        or settings["training"]["models"] != MODELS
        or settings["training"]["search_backend"] != "optuna"
        or settings["training"]["search_budget"] != 128
        or settings["training"]["inner_folds"] != 5
        or settings["training"]["n_jobs"] != 1
        or settings["training"]["refit_full"] is not True
        or settings["resources"]["cpu_budget"] != 1
    ):
        raise ValueError("Bounded three-seed/strict5/Optuna128/eight-model/all27 protocol changed")
    roots = [p.resolve() for p in (template, output, durable)]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError("Immutable template, scratch and durable roots must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    durable.mkdir(parents=True, exist_ok=True)
    journal = dict(
        schema=SCHEMA,
        status="waiting_parent",
        pid=os.getpid(),
        process_identity=process_identity(os.getpid()),
        config_sha256=sha(config_path),
        task_id=TASK,
        required_seeds=list(SEEDS),
        rounds=[],
        mandatory_reporting_state="not_started",
        additional_twenty_hour_loop=False,
    )

    def persist():
        for dest in (output, durable):
            write(dest / "campaign_status.json", journal)

    persist()
    try:
        while parent_ready(cfg) is None:
            time.sleep(min(30, max(1, float(cfg.get("poll_seconds", 5)))))
            persist()
        os.nice(10)
        for i, seed in enumerate(SEEDS):
            verify_ready_for_execution(template)
            run = output / f"round_{i:03d}_seed_{seed}"
            check = next(s for s in node["actual_checks"]["seeds"] if s["seed"] == seed)
            current, staged_source = stage_seed(
                template, run, seed, node["outer_split_receipts"][str(seed)], settings
            )
            values = rows(run / read(run / "task_manifest.json")["tasks"][0]["table_path"])
            ontology = read(run / "task_tables/semantic_context_contract.json")
            full = load_feature_route(values, portable_features(run), "structure_methods_all")
            width = len(ontology["feature_names"])
            entry = dict(
                seed=seed,
                run_dir=str(run),
                status="running",
                config_sha256=sha(run / "config.json"),
                phase="fixed_float64_controls",
                staged_source_closure=staged_source,
            )
            journal["rounds"].append(entry)
            journal["status"] = "running"
            persist()
            controls = fixed_ridge_controls(
                values, full["X"][:, :-width], full["X"][:, -width:], check
            )
            validate_controls(values, full["X"][:, :-width], full["X"][:, -width:], check, controls)
            write(run / "fixed_float64_controls.json", controls)
            entry["phase"] = "selected_model_search_then_separate_all27_refit"
            persist()
            results = train_stage(run, current, "full")
            if len(results) != 1 or results[0].get("status") != "completed":
                raise RuntimeError("Required real model task did not complete")
            result = results[0]
            pairs = result.get("search", {}).get("coverage_attempted", [])
            if len({m for _, m in pairs}) < 8 or not {
                m["route"] for m in read(run / "representation_methods.json")["methods"]
            } <= {r for r, _ in pairs}:
                raise ValueError("Actual search lacks the ten methods/eight model families")
            payload = load_feature_route(values, portable_features(run), result["selected_route"])
            selected = joblib.load(result["artifacts"]["model"])
            result_path = run / "results/full" / (TASK + "_common.json")
            write(
                run / "run_phase_numeric_provenance.json",
                dict(
                    result_sha256=sha(result_path),
                    controls_sha256=sha(run / "fixed_float64_controls.json"),
                    selected_solver=selected_solver_provenance(selected, result, payload),
                    explicit_run_phase_only=True,
                    report_only_fit_calls=0,
                ),
            )
            report_stage(run)
            summarize_run(run, run)
            report = write_report(run, run_dir=run, seed=seed)
            entry.update(
                status="completed",
                phase="verified_terminal",
                audit=dict(
                    status="verified",
                    actual_folds=5,
                    backend="optuna",
                    verification_sha256=sha(run / "verification.json"),
                    method_model_coverage_met=True,
                    attempted_independent_methods=10,
                    attempted_model_families=8,
                ),
                molecular_reporting=dict(
                    path="reports/vd_absolute_native_diagnostics.json",
                    sha256=sha(run / "reports/vd_absolute_native_diagnostics.json"),
                ),
                all27_refit_verified=report["separate_all27_refit"]["status"] == "verified",
            )
            entry["archive"] = archive_run(run, durable, run.name)
            persist()
        journal.update(status="completed", mandatory_reporting_state="verified")
        persist()
    except BaseException as exc:
        journal.update(
            status="requirement_incomplete",
            reason=f"{type(exc).__name__}:{exc}",
            mandatory_reporting_state="incomplete",
        )
        persist()
        raise
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--clone-prepared", type=Path)
    parser.add_argument("--bind-review", type=Path)
    parser.add_argument("--nofit-nodefold-probe", type=Path)
    parser.add_argument("--adopt-nodefold-receipt", type=Path)
    parser.add_argument("--adopt-postadoption-staging", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.clone_prepared:
        result = clone_nodefold_version(
            args.clone_prepared, args.output if args.output != OUTPUT else NODE_OUTPUT
        )
    elif args.run_followup:
        if args.config is None:
            parser.error("--run-followup requires --config")
        result = run_followup(args.config)
    elif (
        args.bind_review
        or args.nofit_nodefold_probe
        or args.adopt_nodefold_receipt
        or args.adopt_postadoption_staging
        or args.report_only
    ):
        if args.prepared is None:
            parser.error("This no-fit operation requires --prepared")
        if args.bind_review:
            result = bind_independent_review(args.prepared, args.bind_review)
        elif args.nofit_nodefold_probe:
            result = nofit_nodefold_probe(args.prepared, args.nofit_nodefold_probe)
        elif args.adopt_nodefold_receipt:
            result = adopt_nodefold_receipt(args.prepared, args.adopt_nodefold_receipt)
        elif args.adopt_postadoption_staging:
            result = adopt_postadoption_staging(args.prepared, args.adopt_postadoption_staging)
        else:
            result = write_report(
                args.prepared,
                run_dir=args.prepared if args.seed is not None else None,
                seed=args.seed,
            )
    else:
        result = prepare(args.source, args.parent, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
