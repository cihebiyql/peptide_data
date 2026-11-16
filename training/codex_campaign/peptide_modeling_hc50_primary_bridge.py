"""A separately versioned nominal human-RBC HC50 cohort with one primary correction.

Preparation copies exact existing graph vectors. Training delegates to the frozen
group-holdout/inner-fivefold runner; its optional all-data refit is a distinct model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import signal
from collections import defaultdict
from pathlib import Path

import numpy as np
import peptide_modeling_hc10_human_bridge as legacy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/dramp_hemolysis_ic50_primary_recovery_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_hc50_primary_followup_20261115_v1"
OLD_TASK = "TASK_e879fa2991add3daba5f1b0e"
OLD_TABLE = ROOT / "data/peptide_modeling_multiseed20h_overlay_v1/task_tables" / (OLD_TASK + ".tsv")
OLD_TABLE_SHA = "7cab364576f1eb01c30d7f0d2013bfe3932ed658322fc8cc45c4b84b1ab7a195"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
REVIEW_SHA = "0ab13cd339e2bb229b2b36610f0543867eb932b425ad6718d807f120da030ba4"
LEGACY_IMPLEMENTATION_SHA = "27d4aa3d737ea88178a6ffb00e3a047ebf3e529af3885bde0fc271dfe4c140f2"
NEW_ID = "OLD:PV13:712120ca59fa82219e3e3920"
NEW_GRAPH = "CANON_SMILES:259c12a291ad369dc5f2df0034afd4ebc3c3450590d813e1fa3a05342d17c887"
NEW_FAMILY = "FAMILY:d5ae42256f5c2fe6aa4b79f9"
RAW_FAMILY = "FAMILY:1ec6ec05884fb27225c98e9c"
QUANTITY = "source_reported_nominal_human_erythrocyte_half_lysis_concentration"
PROOF_SCHEMA = "hc50_primary_corrected_observation_coverage_v1"
REPORT_SCHEMA = "hc50_primary_corrected_molecular_diagnostics_v1"
SOURCE_OBJECT_HASHES = {
    "evidence": REVIEW_SHA,
    "identity": "644f62ec38c1be8e8cfb725377c1d87c9aa539573e2161d6cbc231e5158dbd0f",
    "original_record": "5379777048bc730a50ade1714464e67d0cf3fbe7e826e310b2bda9ccaa230ceb",
    "article_pdf": "292923913137cb2cad6fa969c8f26e5fc7ecb8844dfb664bdf424c87f07a6879",
}
SEEDS = (42, 2026, 3407)
GROUPS = legacy.GROUPS
sha, read, rows, write, table, canonical, error_metrics = (
    legacy.sha,
    legacy.read,
    legacy.rows,
    legacy.write,
    legacy.table,
    legacy.canonical,
    legacy.error_metrics,
)
WARNING = (
    "Nominal source-reported human-RBC half-lysis concentration. The original33 records "
    "have unknown fit methodology; only the primary-corrected155uM observation is a "
    "source-reported estimate beyond the stated100uM tested range. Its half-lysis mapping "
    "is inferred from the IC50 table and dedicated percent-lysis method, not an independent "
    "HC50 definition or a directly observed155uM response. Dedicated RBC1h and general "
    "table48h evidence conflict. Acetyl/amidated reference termini correct the secondary "
    "Free/Free annotation. No standardized time, method, clinical or sample-stereopurity "
    "claim. Holdout scores belong to the outer-train model; the separate all34-data refit "
    "has no independent evaluation on those now-included records."
)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


TASK = (
    "TASK_HC50_PRIMARY_"
    + object_hash(dict(quantity=QUANTITY, old33=OLD_TABLE_SHA, original=NEW_ID, review=REVIEW_SHA))[
        :24
    ]
)


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source proof escapes the versioned task")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Source proof requires a contained regular file")
    return path


def validate_cohort(values, *, verify_chemistry=True):
    """Chemistry runs at preparation; reports bind the frozen canonical strings."""
    if len(values) != 34 or len({r["record_id"] for r in values}) != 34:
        raise ValueError("The revised cohort requires34 distinct observations")
    graph_families = {}
    for row in values:
        graph = row["canonical_smiles"]
        key = "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
        if not graph or key != row["canonical_identity_group"]:
            raise ValueError("Full canonical graph binding changed")
        if verify_chemistry:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(row["smiles"])
            if (
                molecule is None
                or canonical(row["smiles"]) != graph
                or any(
                    value == "?"
                    for _, value in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
                )
            ):
                raise ValueError("Complete assigned-stereo reference graphs are required")
        if graph_families.setdefault(key, row["split_group"]) != row["split_group"]:
            raise ValueError("An exact complete graph crosses molecular families")
        if (
            row["unit"] != "uM"
            or row["target_transform"] != "log10"
            or row["canonical_quantity_id"] != QUANTITY
            or row["canonical_species"] != "human"
            or row["canonical_material"] != "erythrocytes"
            or not math.isfinite(float(row["target"]))
        ):
            raise ValueError("Nominal source quantity or target changed")
    if len(graph_families) != 30 or len(set(graph_families.values())) != 30:
        raise ValueError("34 observations must retain30 whole-graph/FAMILY groups")
    new = [row for row in values if row["record_id"] == NEW_ID]
    if len(new) != 1:
        raise ValueError("Exactly one primary-corrected original observation is required")
    row = new[0]
    expected = dict(
        canonical_identity_group=NEW_GRAPH,
        split_group=NEW_FAMILY,
        source_original_endpoint="hemolysis_IC50",
        raw_source_split_group=RAW_FAMILY,
        value="155",
        reported_sd="13.04",
        reported_statistic="mean_plus_minus_SD",
        reported_donors="5",
        raw_relation="=",
        raw_unit="uM",
        primary_N_terminus="acetylated",
        primary_C_terminus="amidated",
        raw_N_terminus="Free",
        raw_C_terminus="Free",
        primary_identity_correction_applied="true",
        estimate_method="source_reported_estimate",
        extrapolation_status="inferred_beyond_stated_assay_range",
        exposure_evidence_status="conflicted_RBC_method1h_general_caption48h",
        mapping_evidence_status="inferred_from_IC50_table_and_percent_lysis_method",
        independently_defined_HC50="false",
        directly_observed_half_lysis_point="false",
    )
    if any(row.get(k) != value for k, value in expected.items()) or not math.isclose(
        float(row["target"]), math.log10(155), rel_tol=0, abs_tol=1e-10
    ):
        raise ValueError("Primary correction, extrapolation, conflict or original scalar changed")
    if any(
        r["estimate_method"] != "unknown_in_frozen_contract"
        for r in values
        if r["record_id"] != NEW_ID
    ):
        raise ValueError("The old33 methods must not be relabelled as fitted measurements")
    return values


def source_rows(source=SOURCE, parent=PARENT):
    source, parent = Path(source), Path(parent)
    pins = {
        source / "evidence.json": REVIEW_SHA,
        source / "identity_graph_comparison.json": (
            "644f62ec38c1be8e8cfb725377c1d87c9aa539573e2161d6cbc231e5158dbd0f"
        ),
        source / "original_frozen_and_dramp_source.json": (
            "5379777048bc730a50ade1714464e67d0cf3fbe7e826e310b2bda9ccaa230ceb"
        ),
        source / "article.pdf": "292923913137cb2cad6fa969c8f26e5fc7ecb8844dfb664bdf424c87f07a6879",
        OLD_TABLE: OLD_TABLE_SHA,
        parent / "features/manifest.json": PARENT_SHA,
        Path(legacy.__file__): LEGACY_IMPLEMENTATION_SHA,
    }
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("Pinned primary/old-cohort evidence changed: " + str(path))
    evidence, comparison, raw = [
        read(source / n)
        for n in (
            "evidence.json",
            "identity_graph_comparison.json",
            "original_frozen_and_dramp_source.json",
        )
    ]
    numeric, exposure = evidence["primary_numeric_evidence"], evidence["exposure_conflict"]
    if (
        evidence["original_record"]["record_id"] != NEW_ID
        or numeric["reported_mean"] != "155"
        or numeric["reported_SD"] != "13.04"
        or numeric["donors"] != 5
        or evidence["physical_readout"]["reported_experiment_range_uM"] != [1.56, 100]
        or not evidence["physical_readout"]["reported155_exceeds_stated_assay_maximum"]
        or exposure["RBC_method_hours"] != 1
        or exposure["generic_Table1_caption_hours"] != 48
        or not exposure["primary_table_caption_and_RBC_method_conflict"]
        or raw["record_id"] != NEW_ID
        or raw["value"] != "155"
        or raw["unit"] != "uM"
        or raw["relation"] != "="
        or raw["endpoint"] != "hemolysis_IC50"
    ):
        raise ValueError("Original scalar, readout or exposure evidence changed")
    variants = comparison["variants"]
    matches = [v for v in variants if v["primary_sequence_and_caps_match"]]
    if len(matches) != 1 or matches[0]["graph_key"] != NEW_GRAPH or len(variants) != 4:
        raise ValueError("Primary terminal correction lacks its exact complete graph")
    graph = matches[0]["canonical_isomeric_smiles"]
    if any(
        v["canonical_isomeric_smiles"] == graph
        for v in variants
        if not v["primary_sequence_and_caps_match"]
    ):
        raise ValueError("Wrong-terminal negative controls match the primary reference graph")
    old = rows(OLD_TABLE)
    base = read(parent / "features/manifest.json")
    graph_map = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in rows(Path(base["cache_dir"]) / "structure_source_smiles.tsv")
    }
    if (
        len(old) != 33
        or len({r["split_group"] for r in old}) != 29
        or any(r["record_id"] == NEW_ID for r in old)
    ):
        raise ValueError("Original33 cohort membership changed")
    values = []
    extra = dict(
        canonical_quantity_id=QUANTITY,
        canonical_species="human",
        canonical_material="erythrocytes",
        estimate_method="unknown_in_frozen_contract",
        extrapolation_status="unknown_in_frozen_contract",
        exposure_evidence_status="not_reported_in_frozen_contract",
        source_original_endpoint="hemolysis_HC50",
        primary_identity_correction_applied="false",
        value="",
        reported_sd="",
        reported_statistic="",
        reported_donors="",
        raw_relation="",
        raw_unit="",
        primary_N_terminus="",
        primary_C_terminus="",
        raw_N_terminus="",
        raw_C_terminus="",
        raw_source_split_group="",
        mapping_evidence_status="source_HC50_label",
        independently_defined_HC50="not_independently_reaudited",
        directly_observed_half_lysis_point="unknown",
    )
    for item in old:
        canonical_graph = graph_map[item["canonical_identity_group"]]
        values.append(
            {**item, **extra, "smiles": canonical_graph, "canonical_smiles": canonical_graph}
        )
    item = copy.deepcopy(evidence["identity"]["original_frozen_feature_row"])
    # Preserve the incorrect secondary identity in the proof, not as the model input.
    item.update(
        structure_eligible="true",
        canonical_identity_group=NEW_GRAPH,
        structure_smiles_sha256=NEW_GRAPH.split(":", 1)[1],
        split_group=NEW_FAMILY,
        scope_warning="primary_caps_corrected;reported_extrapolated_estimate;RBC1h_caption48h_conflict",
    )
    item.update(extra)
    item.update(
        smiles=graph,
        canonical_smiles=graph,
        source_original_endpoint="hemolysis_IC50",
        raw_source_split_group=RAW_FAMILY,
        value="155",
        reported_sd="13.04",
        reported_statistic="mean_plus_minus_SD",
        reported_donors="5",
        raw_relation="=",
        raw_unit="uM",
        primary_N_terminus="acetylated",
        primary_C_terminus="amidated",
        raw_N_terminus="Free",
        raw_C_terminus="Free",
        primary_identity_correction_applied="true",
        estimate_method="source_reported_estimate",
        extrapolation_status="inferred_beyond_stated_assay_range",
        exposure_evidence_status="conflicted_RBC_method1h_general_caption48h",
        mapping_evidence_status="inferred_from_IC50_table_and_percent_lysis_method",
        independently_defined_HC50="false",
        directly_observed_half_lysis_point="false",
    )
    values.append(item)
    fields = list(dict.fromkeys(k for row in values for k in row))
    values = [{key: row.get(key, "") for key in fields} for row in values]
    if any(
        any(current[k] != previous[k] for k in previous)
        for current, previous in zip(values[:33], old, strict=True)
    ):
        raise ValueError("Original33 source cells were not preserved")
    validate_cohort(values)
    return values, pins, evidence, comparison, raw


def split_prechecks(values, output):
    from peptide_modeling_train import _inner_splits, freeze_outer_split, validate_rows

    y, groups = validate_rows(values, "regression")
    checks = []
    for seed in SEEDS:
        settings = dict(
            seed=seed, inner_folds=5, test_size=0.2, min_groups=8, min_rows=10, min_train_rows=6
        )
        split_path = Path(output) / "task_tables/predeclared_splits" / f"seed_{seed}.json"
        split = freeze_outer_split(values, "regression", split_path, settings)
        train = np.flatnonzero(np.isin(groups, split["train_groups"]))
        test = np.flatnonzero(np.isin(groups, split["test_groups"]))
        folds = _inner_splits(y[train], groups[train], "regression", settings)
        if (
            len(folds) != 5
            or set(groups[train]) & set(groups[test])
            or len(set(groups[train])) != 24
        ):
            raise ValueError("Actual grouped outer holdout/inner fivefold is unavailable")
        checks.append(
            dict(
                seed=seed,
                actual_folds=5,
                split_path=split_path.relative_to(output).as_posix(),
                split_sha256=sha(split_path),
                train_record_ids=[values[i]["record_id"] for i in train],
                holdout_record_ids=[values[i]["record_id"] for i in test],
                train_groups=len(set(groups[train])),
                holdout_groups=len(set(groups[test])),
                corrected_record_evaluation_role="inner_oof"
                if NEW_ID in split["train_records"]
                else "outer_holdout",
                final_refit_record_ids=[r["record_id"] for r in values],
                folds=[
                    dict(
                        fold=f,
                        train_record_ids=[values[train[i]]["record_id"] for i in a],
                        validation_record_ids=[values[train[i]]["record_id"] for i in b],
                    )
                    for f, (a, b) in enumerate(folds)
                ],
            )
        )
    return dict(
        protocol="group_holdout_inner5_then_separate_all_common_refit",
        seeds=checks,
        all_common_refit_after_evaluation=True,
        selection_holdout_not_used=True,
    )


def verify_coverage_proof(root, task, values):
    validate_cohort(values, verify_chemistry=False)
    descriptors = task.get("observation_endpoint_coverage")
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        raise ValueError("One explicit primary-corrected observation proof is required")
    descriptor = descriptors[0]
    expected = dict(
        original_endpoint="hemolysis_IC50",
        canonical_endpoint="hemolysis_HC50",
        canonical_quantity_id=QUANTITY,
        record_ids=[NEW_ID],
        original_labelled_records=1,
        included_original_records=1,
        included_original_molecular_groups=1,
        canonical_molecular_groups=30,
        inference_alias_is_global=False,
        raw_uncorrected_identity_covered=False,
        coverage_kind="primary_source_corrected_observation_only",
    )
    if task.get("endpoint") != "hemolysis_HC50" or any(
        descriptor.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Corrected-observation coverage scope changed")
    proof_path = contained(root, descriptor["proof_path"])
    if sha(proof_path) != descriptor["proof_sha256"]:
        raise ValueError("Primary-correction proof changed")
    proof = read(proof_path)
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("canonical_task_id") != task["task_id"]
        or any(proof.get(k) != v for k, v in expected.items())
        or proof.get("canonical_record_ids") != [r["record_id"] for r in values]
        or proof.get("full_refit_required_for_each_seed_training_coverage") is not True
    ):
        raise ValueError("Primary-correction proof declaration changed")
    current_path = contained(root, proof["canonical_source_table"]["path"])
    if (
        str(proof["canonical_source_table"]["path"]) != task["table_path"]
        or sha(current_path) != proof["canonical_source_table"]["sha256"]
        or proof["canonical_source_table"].get("records") != 34
    ):
        raise ValueError("Canonical source table differs from the proof")
    original = contained(root, proof["parent_thirtythree"]["path"])
    if (
        sha(original) != OLD_TABLE_SHA
        or proof["parent_thirtythree"]["sha256"] != OLD_TABLE_SHA
        or proof["parent_thirtythree"].get("task_id") != OLD_TASK
    ):
        raise ValueError("Original33 source snapshot changed")
    old = rows(original)
    if [r["record_id"] for r in values[:33]] != [r["record_id"] for r in old] or any(
        any(row[k] != previous[k] for k in previous)
        for row, previous in zip(values[:33], old, strict=True)
    ):
        raise ValueError("Original33 source fields were dropped or rewritten")
    if set(proof["source_objects"]) != set(SOURCE_OBJECT_HASHES):
        raise ValueError("Required primary/raw evidence object missing")
    for name, item in proof["source_objects"].items():
        if (
            sha(contained(root, item["path"])) != item["sha256"]
            or item["sha256"] != SOURCE_OBJECT_HASHES[name]
        ):
            raise ValueError("Primary/raw source snapshot SHA changed")
    native = {
        k: read(contained(root, v["path"]))
        for k, v in proof["source_objects"].items()
        if k != "article_pdf"
    }
    raw, evidence, comparison = native["original_record"], native["evidence"], native["identity"]
    if (
        raw["record_id"] != NEW_ID
        or raw["endpoint"] != "hemolysis_IC50"
        or raw["value"] != "155"
        or raw["audit_source_row"]["N-terminal_Modification"] != "Free"
        or raw["audit_source_row"]["C-terminal_Modification"] != "Free"
        or evidence["identity"]["old_placeholder_family"] != RAW_FAMILY
        or comparison["variants"][-1]["graph_key"] != NEW_GRAPH
        or not evidence["exposure_conflict"]["primary_table_caption_and_RBC_method_conflict"]
    ):
        raise ValueError("Primary identity correction lost the raw conflicting evidence")
    new = next(r for r in values if r["record_id"] == NEW_ID)
    if proof["corrected_observation"] != new:
        raise ValueError("Corrected observation differs from its source-bound proof")
    required_readout = dict(
        status="combined_table_and_percent_lysis_readout_inference",
        independently_defined_HC50=False,
        direct_observed_155uM_half_lysis=False,
        reported_mean="155",
        reported_SD="13.04",
        statistic="mean_plus_minus_SD",
        donors=5,
        observed_range_uM=[1.56, 100],
        maximum_tested_lysis_percent="30-35",
        extrapolation_inferred=True,
        original_curve_refitted=False,
        RBC_method_hours=1,
        general_caption_hours=48,
        exposure_conflicted=True,
    )
    if (
        proof["primary_readout_mapping"] != required_readout
        or proof["original_secondary_feature_row"]
        != evidence["identity"]["original_frozen_feature_row"]
    ):
        raise ValueError(
            "Primary readout inference, extrapolation or raw identity evidence was promoted"
        )
    prechecks = contained(root, proof["predeclared_splits"]["path"])
    if sha(prechecks) != proof["predeclared_splits"]["sha256"]:
        raise ValueError("Predeclared split source proof changed")
    return proof


def slice_features(values, source, parent, output, *, expected_parent_sha):
    """Copy only30 frozen graph rows; no descriptors or embeddings are computed."""
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
        if key not in graph_index or canonical(
            old_graphs[key]["canonical_isomeric_smiles"]
        ) != canonical(row["smiles"]):
            raise ValueError("A source graph does not exactly match a frozen parent graph")
        graph_map[key] = old_graphs[key]["canonical_isomeric_smiles"]
    if len(graph_map) != 30:
        raise ValueError("Exactly30 old complete graphs must be reused")
    keys = sorted(graph_map)
    configuration = dict(
        version="hc50_primary_corrected_graph_slices_v1",
        parent_sha256=before,
        parent_fingerprint=base["input_fingerprint"],
        source_review_sha256=REVIEW_SHA,
        old_thirtythree_table_sha256=OLD_TABLE_SHA,
        implementation_sha256=sha(__file__),
        legacy_helper_sha256=LEGACY_IMPLEMENTATION_SHA,
        selected_graphs=keys,
        record_ids=[r["record_id"] for r in values],
        canonical_quantity_id=QUANTITY,
        feature_computations=0,
        sequence_or_lossy_projection=False,
        observation_alias_record=NEW_ID,
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
        meta.update(shape=[30, meta["shape"][1]], index="structure_bank_index.tsv")
    table(
        cache / "structure_bank_index.tsv", [dict(key=key, row_idx=i) for i, key in enumerate(keys)]
    )
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=key, canonical_isomeric_smiles=graph_map[key]) for key in keys],
    )
    table(
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
        "bank_scope": "hc50_primary_corrected_complete_reference_graphs_only",
    }
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", configuration)
    write(
        cache / "copied_vector_audit.json",
        dict(
            reused_graphs=30,
            new_graphs=0,
            byte_identical=True,
            parent_artifacts_checked=len(base["artifact_hashes"]),
        ),
    )
    routes = {item["route"]: item["group_names"] for item in methods}
    routes["structure_methods_all"] = list(GROUPS)
    manifest = dict(
        schema="hc50_primary_corrected_structure_slices_v1",
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=34,
        structure_keys=30,
        groups=groups,
        routes=routes,
        representation_methods=registry,
        coverage={name: dict(representations=30, valid=30, invalid=0) for name in groups},
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
        if not payload["valid"].all() or payload["X"].shape != (34, method["dimensions"]):
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
            records=34,
            graphs=30,
            methods=10,
            groups=13,
            dimensions=sum(g["shape"][1] for g in groups.values()),
            output_hashes=33,
            parent_hashes_unchanged=len(base["artifact_hashes"]),
            parent_pointer_unchanged=True,
            copied_vectors_byte_identical=True,
            new_graphs=0,
            feature_computations=0,
            new_observation_aliases=1,
        ),
    )
    return manifest


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output, source, parent = Path(output).resolve(), Path(source).resolve(), Path(parent).resolve()
    if output.exists() and any(path.name != "diagnostic_history" for path in output.iterdir()):
        raise FileExistsError("A prepared version cannot be overwritten")
    values, pins, evidence, comparison, raw = source_rows(source, parent)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "task_tables/source_snapshots"
    snapshots.mkdir(parents=True)
    shutil.copy2(OLD_TABLE, snapshots / "old_hc50_thirtythree.tsv")
    objects = {}
    for key, filename in (
        ("evidence", "evidence.json"),
        ("identity", "identity_graph_comparison.json"),
        ("original_record", "original_frozen_and_dramp_source.json"),
        ("article_pdf", "article.pdf"),
    ):
        target = snapshots / filename
        shutil.copy2(source / filename, target)
        objects[key] = dict(path=target.relative_to(output).as_posix(), sha256=sha(target))
    task_path = "task_tables/" + TASK + ".tsv"
    table(output / task_path, values)
    checks = split_prechecks(values, output)
    checks_path = output / "task_tables/predeclared_fivefold_membership.json"
    write(checks_path, checks)
    write(output / "fivefold_prechecks.json", checks)
    scope = dict(
        original_endpoint="hemolysis_IC50",
        canonical_endpoint="hemolysis_HC50",
        canonical_quantity_id=QUANTITY,
        record_ids=[NEW_ID],
        original_labelled_records=1,
        included_original_records=1,
        included_original_molecular_groups=1,
        canonical_molecular_groups=30,
        inference_alias_is_global=False,
        raw_uncorrected_identity_covered=False,
        coverage_kind="primary_source_corrected_observation_only",
    )
    proof = dict(
        schema=PROOF_SCHEMA,
        canonical_task_id=TASK,
        **scope,
        canonical_record_ids=[r["record_id"] for r in values],
        canonical_source_table=dict(path=task_path, sha256=sha(output / task_path), records=34),
        parent_thirtythree=dict(
            task_id=OLD_TASK,
            path="task_tables/source_snapshots/old_hc50_thirtythree.tsv",
            sha256=OLD_TABLE_SHA,
        ),
        source_objects=objects,
        corrected_observation=next(r for r in values if r["record_id"] == NEW_ID),
        original_secondary_feature_row=evidence["identity"]["original_frozen_feature_row"],
        primary_identity_correction=dict(
            secondary_N_terminus="Free",
            secondary_C_terminus="Free",
            primary_N_terminus="acetylated",
            primary_C_terminus="amidated",
            original_placeholder_family=RAW_FAMILY,
            corrected_graph=NEW_GRAPH,
            corrected_family=NEW_FAMILY,
            identity_tier="L_reference_parent_graph_not_sample_stereopurity_or_salt_state",
        ),
        primary_readout_mapping=dict(
            status="combined_table_and_percent_lysis_readout_inference",
            independently_defined_HC50=False,
            direct_observed_155uM_half_lysis=False,
            reported_mean="155",
            reported_SD="13.04",
            statistic="mean_plus_minus_SD",
            donors=5,
            observed_range_uM=[1.56, 100],
            maximum_tested_lysis_percent="30-35",
            extrapolation_inferred=True,
            original_curve_refitted=False,
            RBC_method_hours=1,
            general_caption_hours=48,
            exposure_conflicted=True,
        ),
        predeclared_splits=dict(
            path=checks_path.relative_to(output).as_posix(), sha256=sha(checks_path)
        ),
        full_refit_required_for_each_seed_training_coverage=True,
        warning=WARNING,
    )
    proof_path = output / "task_tables/observation_endpoint_coverage.json"
    write(proof_path, proof)
    write(output / "observation_endpoint_coverage.json", proof)
    definition = dict(
        quantity=QUANTITY,
        canonical_quantity_id=QUANTITY,
        species="human",
        assay_material="reported_erythrocytes",
        target_unit="uM",
        exposure_time_conditioned=False,
        standardized_protocol_verified=False,
        all_records_fitted=False,
        legacy_fit_method_status="unknown_in_frozen_contract",
        corrected_estimate_extrapolated=True,
        corrected_exposure_conflicted=True,
        primary_identity_correction_required=True,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=34,
        separate_all_common_refit_required=True,
        inference_alias_is_global=False,
        raw_uncorrected_identity_covered=False,
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="hemolysis_HC50",
        task_kind="regression",
        status="eligible",
        training_admitted=True,
        table_path=task_path,
        n_records=34,
        n_groups=30,
        unit="uM",
        target_transform="log10",
        inverse_transform="pow10",
        requested_inner_folds=5,
        seeds=list(SEEDS),
        validation_protocol="group_holdout_inner5_then_separate_all_common_refit",
        scientific_definition=definition,
        task_definition=copy.deepcopy(definition),
        observation_endpoint_coverage=[
            dict(
                **scope,
                proof_path=proof_path.relative_to(output).as_posix(),
                proof_sha256=sha(proof_path),
            )
        ],
        canonical_quantity_id=QUANTITY,
        model_count_for_canonical_target=1,
        scientific_warning=WARNING,
    )
    verify_coverage_proof(output, task, values)
    write(
        output / "task_manifest.json",
        dict(schema="hc50_primary_corrected_admission_v1", tasks=[task]),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    features = slice_features(values, source, parent, output, expected_parent_sha=PARENT_SHA)
    config = dict(
        dataset="data/dataset_reclean_20261113_v1",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        training=dict(
            routes=list(features["routes"]),
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
            test_size=0.2,
            min_rows=10,
            min_groups=8,
            min_train_rows=6,
            seed=42,
            search_backend="optuna",
            search_budget=128,
            n_jobs=1,
            refit_full=True,
            svm_max_features=16384,
            histgb_max_features=16384,
            elasticnet_max_features=16384,
        ),
        required_seeds=list(SEEDS),
        validation_protocol="group_holdout_inner5_then_separate_all_common_refit",
        mandatory_postprocessing="source_corrected_observation_roles_and_molecule_equal_diagnostics",
    )
    write(output / "config.json", config)
    write(output / "recommended_training_config.json", config)
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("Protected evidence or old source changed during preparation")
    write(
        output / "source_admission.json",
        dict(
            status="prepared_not_trained",
            quantity=QUANTITY,
            source_pins={str(path): expected for path, expected in pins.items()},
            source_observation_coverage=scope,
            old33_source_fields_unchanged=True,
            old33_fit_method_not_relabelled=True,
            secondary_identity_conflict_preserved=True,
            corrected_reference_graph=NEW_GRAPH,
            corrected_molecular_family=NEW_FAMILY,
            new_record=NEW_ID,
            records=34,
            molecular_groups=30,
            parent_artifacts_unchanged=48,
            feature_computations=0,
            donors_do_not_inflate_molecule_count=True,
            training_or_remote_operations=False,
        ),
    )
    (output / "REPORT.md").write_text(
        "# Primary-corrected nominal human-RBC half-lysis cohort\n\n"
        "34 observations /30 complete-graph molecular families; the original33 source records "
        "are preserved. One DRAMP hemolysis_IC50 observation is represented only through an "
        "explicit primary-source terminal-identity correction and inferred half-lysis readout.\n\n"
        + WARNING
        + "\n\nTen structure methods plus fusion, eight model families, Optuna128, "
        "seeds42/2026/3407. "
        "Each seed has a frozen molecular outer holdout and actual inner fivefold. The selected "
        "outer-train model retains its evaluation artifacts; a distinct all34-data refit must "
        "succeed before claiming training coverage for the corrected observation. No fitting "
        "or remote operation was performed by preparation.\n",
        encoding="utf-8",
    )
    write(
        output / "delivery_receipt.json",
        dict(
            status="prepared_not_trained",
            task_id=TASK,
            records=34,
            molecules=30,
            feature_fingerprint=features["input_fingerprint"],
            files={
                p.relative_to(output).as_posix(): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
            },
            historical_intermediate_artifacts_not_execution_inputs={
                p.relative_to(output).as_posix(): sha(p)
                for p in sorted((output / "diagnostic_history").rglob("*"))
                if p.is_file()
            },
        ),
    )
    return task


def evaluation_diagnostics(source_rows, inner_oof, outer_holdout):
    """Keep tuning OOF and outer holdout errors separate; neither scores the full refit."""
    validate_cohort(source_rows, verify_chemistry=False)
    source = {r["record_id"]: r for r in source_rows}
    all_predictions = list(inner_oof) + list(outer_holdout)
    if (
        len(all_predictions) != 34
        or len({r["record_id"] for r in all_predictions}) != 34
        or {r["record_id"] for r in all_predictions} != set(source)
    ):
        raise ValueError("Actual evaluation membership must cover all34 source observations once")
    if {r["split_group"] for r in inner_oof} & {r["split_group"] for r in outer_holdout}:
        raise ValueError("Whole molecular groups cross evaluation roles")
    reports = {}
    corrected = None
    for role, predictions in (("inner_oof", inner_oof), ("outer_holdout", outer_holdout)):
        grouped, group_folds = defaultdict(list), {}
        for row in predictions:
            original = source[row["record_id"]]
            group, fold = row["split_group"], int(row["fold"])
            target, prediction = float(row["target"]), float(row["prediction"])
            if (
                group != original["split_group"]
                or target != float(original["target"])
                or not math.isfinite(prediction)
            ):
                raise ValueError("Evaluation source target, group or prediction changed")
            if (role == "inner_oof" and not 0 <= fold < 5) or (
                role == "outer_holdout" and fold != -1
            ):
                raise ValueError("OOF and holdout fold roles were confused")
            if group_folds.setdefault(group, fold) != fold:
                raise ValueError("A molecular group crosses inner validation folds")
            item = dict(
                record_id=row["record_id"],
                split_group=group,
                fold=fold,
                target=target,
                prediction=prediction,
                evaluation_role=role,
                estimate_method=original["estimate_method"],
                extrapolation_status=original["extrapolation_status"],
                exposure_evidence_status=original["exposure_evidence_status"],
                primary_identity_correction_applied=original["primary_identity_correction_applied"],
            )
            grouped[group].append(item)
            if row["record_id"] == NEW_ID:
                corrected = {
                    **item,
                    **error_metrics([target], [prediction]),
                    "original_endpoint": "hemolysis_IC50",
                    "raw_uncorrected_identity_covered": False,
                    "primary_corrected_graph": NEW_GRAPH,
                    "reported_mean_uM": "155",
                    "reported_SD_uM": "13.04",
                    "donors": 5,
                    "molecules": 1,
                    "independently_defined_HC50": False,
                    "directly_observed_half_lysis_point": False,
                }
        if role == "inner_oof" and set(group_folds.values()) != set(range(5)):
            raise ValueError("Saved inner OOF must contain five actual folds")
        per_group = []
        for group, values in sorted(grouped.items()):
            p = [r["prediction"] for r in values]
            if not np.allclose(p, p[0], rtol=1e-7, atol=1e-8):
                raise ValueError(
                    "One structure-only graph received inconsistent predictions within its role"
                )
            per_group.append(
                dict(
                    split_group=group,
                    observations=len(values),
                    fold=group_folds[group],
                    **error_metrics([r["target"] for r in values], p),
                    predictions=values,
                )
            )
        records = [r for values in grouped.values() for r in values]
        scale = max(g["rmse"] for g in per_group)
        mae_scale = max(g["mae"] for g in per_group)
        reports[role] = dict(
            records=len(records),
            molecular_groups=len(grouped),
            actual_folds=5 if role == "inner_oof" else None,
            record_weighted=error_metrics(
                [r["target"] for r in records], [r["prediction"] for r in records]
            ),
            molecule_equal=dict(
                mae=float(np.mean([g["mae"] / mae_scale for g in per_group])) * mae_scale
                if mae_scale
                else 0.0,
                rmse=float(np.sqrt(np.mean([(g["rmse"] / scale) ** 2 for g in per_group])) * scale)
                if scale
                else 0.0,
            ),
            per_molecule=per_group,
            score_role="model_selection" if role == "inner_oof" else "outer_train_model_evaluation",
            full_refit_performance_claimed=False,
        )
        without = [r for r in records if r["record_id"] != NEW_ID]
        reports[role]["excluding_corrected_observation_score_only"] = dict(
            records=len(without),
            metrics=error_metrics(
                [r["target"] for r in without], [r["prediction"] for r in without]
            ),
            model_retrained=False,
            ablation_experiment=False,
        )
    return dict(
        status="verified",
        records=34,
        molecules=30,
        metrics_space="log10_uM",
        evaluation_by_role=reports,
        corrected_observation=corrected,
        new_model_selection=False,
        extra_model_fit=False,
        warning=WARNING,
    )


def artifact_path(root, value):
    root = Path(root).resolve()
    value = Path(value)
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Model artifact escapes its run")
    return path


def write_diagnostics(config_path, journal=None):
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features
    from peptide_modeling_summarize import verified_predictions
    from peptide_modeling_supervise import atomic_json

    if sha(legacy.__file__) != LEGACY_IMPLEMENTATION_SHA:
        raise ValueError("Pinned numeric helper changed")
    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "campaign_status.json")
    if len(journal["rounds"]) != 3 or {r["seed"] for r in journal["rounds"]} != set(SEEDS):
        raise ValueError("Mandatory diagnostics require all three declared seed runs")
    reports = []
    for entry in journal["rounds"]:
        if (
            entry["status"] != "completed"
            or entry["audit"].get("missing_completed_task_ids")
            or not entry["audit"].get("method_model_coverage_met")
        ):
            raise ValueError(
                "A required seed has not completed its scientific task/search coverage"
            )
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        if task["task_id"] != TASK:
            raise ValueError("Unexpected task in primary-corrected reporting")
        source = rows(run / task["table_path"])
        proof = verify_coverage_proof(run, task, source)
        result_path = run / "results/full" / (TASK + "_common.json")
        result = read(result_path)
        if (
            result["status"] != "completed"
            or result["refit_status"] != "completed"
            or result["deployment_ready"] is not True
        ):
            raise ValueError(
                "A completed separate all-common refit is mandatory for training coverage"
            )
        verification_path = run / "verification.json"
        verification = read(verification_path)
        if (
            sha(verification_path) != entry["audit"]["verification_sha256"]
            or verification.get("status") != "verified"
            or verification.get("source_task_manifest_sha256") != sha(run / "task_manifest.json")
        ):
            raise ValueError("Seed verification does not bind the actual source manifest")
        verified_predictions(run, run, result, task)
        paths = {name: artifact_path(run, value) for name, value in result["artifacts"].items()}
        if "refit_model" not in paths or str(paths["refit_model"]) not in result["artifact_hashes"]:
            raise ValueError("The all-data refit is not independently hash-covered")
        contract = read(paths["model_contract"])
        ids = [r["record_id"] for r in source]
        if (
            contract["common_record_ids"] != ids
            or contract.get("optional_refit_fit_scope") != "all_common_after_evaluation"
            or contract.get("refit_status") != "completed"
            or contract.get("model_fit_scope") != "outer_train_only"
            or not contract.get("holdout_not_used_for_selection")
            or contract["target_unit"] != "uM"
        ):
            raise ValueError("The all34-data refit and independent evaluation scopes were confused")
        input_contract = read(paths["input_contract"])
        if input_contract["config"].get("refit_full") is not True or input_contract[
            "implementation_sha256"
        ] != sha(run / "scripts/peptide_modeling_train.py"):
            raise ValueError(
                "The reported all-common refit lacks its frozen training implementation"
            )
        feature_payload = load_feature_route(
            source, portable_features(run), result["selected_route"]
        )
        feature_proof = input_contract["feature_hashes"][result["selected_route"]]
        if (
            not feature_payload["valid"].all()
            or feature_payload["record_ids"] != ids
            or feature_payload["feature_names"] != contract["feature_names"]
            or feature_proof["names"] != feature_payload["feature_names"]
            or feature_proof["matrix_sha256"]
            != hashlib.sha256(np.ascontiguousarray(feature_payload["X"]).tobytes()).hexdigest()
            or feature_proof["mask_sha256"]
            != hashlib.sha256(feature_payload["valid"].tobytes()).hexdigest()
        ):
            raise ValueError(
                "The final-refit feature schema/matrix does not cover all34 source rows"
            )
        inner, holdout = rows(paths["oof"]), rows(paths["holdout"])
        report = evaluation_diagnostics(source, inner, holdout)
        prechecks = read(contained(run, proof["predeclared_splits"]["path"]))
        expected = next(item for item in prechecks["seeds"] if item["seed"] == entry["seed"])
        if {r["record_id"] for r in inner} != set(expected["train_record_ids"]) or {
            r["record_id"] for r in holdout
        } != set(expected["holdout_record_ids"]):
            raise ValueError("Actual evaluation roles differ from the predeclared source split")
        expected_folds = {
            record: fold["fold"]
            for fold in expected["folds"]
            for record in fold["validation_record_ids"]
        }
        if {row["record_id"]: int(row["fold"]) for row in inner} != expected_folds:
            raise ValueError("Actual inner fold membership differs from the predeclared split")
        for role, stored in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            for key in ("mae", "rmse"):
                if not math.isclose(
                    report["evaluation_by_role"][role]["record_weighted"][key],
                    result[stored][key],
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "Molecular diagnostics do not reproduce saved evaluation metrics"
                    )
        report.update(
            seed=entry["seed"],
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=sha(run / task["table_path"]),
            result_sha256=sha(result_path),
            oof_sha256=sha(paths["oof"]),
            holdout_sha256=sha(paths["holdout"]),
            source_proof_sha256=task["observation_endpoint_coverage"][0]["proof_sha256"],
            separate_all_common_refit=dict(
                status="verified",
                records=34,
                molecules=30,
                training_record_ids=ids,
                corrected_observation_in_training=True,
                model_path=str(paths["refit_model"]),
                model_sha256=sha(paths["refit_model"]),
                model_contract_sha256=sha(paths["model_contract"]),
                refit_attempts=result["refit_attempts"],
                fit_scope="all_common_after_evaluation",
                independently_evaluated_on_these34=False,
            ),
        )
        reports.append(report)
    payload = dict(
        schema=REPORT_SCHEMA,
        status="verified",
        task_id=TASK,
        canonical_quantity_id=QUANTITY,
        seeds=list(SEEDS),
        reports=reports,
        source_corrected_observation=NEW_ID,
        model_selection_repeated=False,
        raw_uncorrected_identity_covered=False,
        warning=WARNING,
    )
    path = root / "hc50_primary_molecular_diagnostics.json"
    atomic_json(path, payload)
    durable.mkdir(parents=True, exist_ok=True)
    destination = durable / path.name
    shutil.copy2(path, destination)
    if sha(path) != sha(destination):
        raise ValueError("Durable mandatory report differs")
    receipt = dict(
        status="verified",
        path=str(path),
        sha256=sha(path),
        durable_path=str(destination),
        durable_sha256=sha(destination),
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={
            "peptide_modeling_hc10_human_bridge.py": LEGACY_IMPLEMENTATION_SHA
        },
    )
    journal["molecular_reporting"] = receipt
    journal["status"] = "completed"
    atomic_json(root / "campaign_status.json", journal)
    atomic_json(durable / "campaign_status.json", journal)
    return receipt


def run_followup(config_path, diagnostics_only=False):
    from peptide_modeling_followup_seeds import run_followup as run_seed_followup
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = (
        read(root / "campaign_status.json") if diagnostics_only else run_seed_followup(config_path)
    )
    try:
        return write_diagnostics(config_path, journal)
    except BaseException as exc:
        if journal.get("status") == "completed":
            journal["status"] = "requirement_incomplete"
        journal["molecular_reporting"] = dict(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(root / "campaign_status.json", journal)
        atomic_json(durable / "campaign_status.json", journal)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.run_followup or args.diagnostics_only:
        if args.config is None:
            parser.error("--config is required for training/reporting")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        result = run_followup(args.config, args.diagnostics_only)
    else:
        task = prepare(args.source, args.parent, args.output)
        result = dict(
            status="prepared_not_trained", task_id=task["task_id"], records=34, molecules=30
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
