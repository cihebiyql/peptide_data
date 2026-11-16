"""One source-proven observation alias in a canonical nominal human-RBC10% target.

This is an explicit observation-level contract, never a global HC10/MHC10 alias.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import peptide_modeling_hc10_human_bridge as legacy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/peptide_modeling_remaining_gap_contract_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_hc10_human_followup_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_hc10_canonical_followup_20261115_v1"
DB = ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite"
OLD_TASK = "TASK_HC10_HUMAN_SOURCE_799fd3266d92284a"
OLD_TABLE = PARENT / "task_tables" / (OLD_TASK + ".tsv")
OLD_TABLE_SHA = "c2ae46d5b1d072aac26d2811feceb3013f1b5b906a479911cf9e366245319e3f"
NEW_ID = "NEW:07:40000000022951890"
MEL_GROUP = "FAMILY:e3630b19efc0fb3b36837673"
MEL_GRAPH = "CANON_SMILES:f9420c9f1b030adf9d9fe7626706f6eca5574280a79f1daea103aba9000109c0"
QUANTITY = "source_reported_nominal_human_erythrocyte_10pct_release_concentration"
REPORT_SCHEMA = "hc10_human_canonical_molecular_diagnostics_v2"
PROOF_SCHEMA = "hc10_observation_endpoint_coverage_v1"
SEEDS = (42, 2026, 3407)
GROUPS = legacy.GROUPS
sha, read, rows, write, table, canonical = (
    legacy.sha,
    legacy.read,
    legacy.rows,
    legacy.write,
    legacy.table,
    legacy.canonical,
)
error_metrics = legacy.error_metrics
WARNING = (
    "Source-curated nominal human-RBC10% release concentration; exposure, pH, preparation, "
    "readout, concentration grid and study are not standardized. No equivalence of fitted "
    "continuous HC10 and minimum/discrete criteria is established. Ten observations are "
    "five molecules; melittin contributes six rows. Three seeds repeat one LOGO partition. "
    "Tuning CV is selection-optimistic with no independent holdout. Reference graphs do not "
    "prove sample salt/stereopurity. No clinical, protocol, species or matrix generalization. "
    "Only one of six original MHC10 observations has an explicit observation-level alias; "
    "there is no global inference alias and no MHC10-only five-molecule model."
)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


REVIEW_SHA = "0fce12f02dec174df639c21f321ff0ad136f48e1430da4ebfde656ff4e57f53f"
LEGACY_IMPLEMENTATION_SHA = "27d4aa3d737ea88178a6ffb00e3a047ebf3e529af3885bde0fc271dfe4c140f2"
TASK = (
    "TASK_HC10_CANONICAL_"
    + object_hash(
        {"quantity": QUANTITY, "old9": OLD_TABLE_SHA, "new": NEW_ID, "source_review": REVIEW_SHA}
    )[:24]
)


def _csv_unique(path, field, value):
    matches = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for data_row, row in enumerate(csv.DictReader(stream), start=1):
            if row.get(field) == value:
                matches.append((row, data_row))
    if len(matches) != 1:
        raise ValueError(f"Raw {field} linkage is not unique: {value}")
    return matches[0]


def validate_cohort(values, *, verify_chemistry=True):
    """Chemistry is local-only; node reports bind already-pinned canonical hashes."""
    if len(values) != 10 or len({r["record_id"] for r in values}) != 10:
        raise ValueError("Canonical HC10 requires exactly ten distinct observations")
    graph_groups = {}
    for row in values:
        graph = row["canonical_smiles"]
        if not isinstance(graph, str) or not graph:
            raise ValueError("A pinned complete canonical graph is required")
        if verify_chemistry:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(row["smiles"])
            if (
                molecule is None
                or any(
                    value == "?"
                    for _, value in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
                )
                or canonical(row["smiles"]) != graph
            ):
                raise ValueError("Complete assigned-stereo reference graphs are required")
        if row["canonical_identity_group"] != (
            "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
        ):
            raise ValueError("Canonical graph binding changed")
        if graph_groups.setdefault(graph, row["split_group"]) != row["split_group"]:
            raise ValueError("A complete graph crosses molecular families")
        if (
            row["rbc_species"] != "Homo sapiens"
            or row["source_matrix"] != "erythrocytes"
            or row["canonical_quantity_id"] != QUANTITY
            or row["canonical_species"] != "human"
            or row["canonical_material"] != "erythrocytes"
            or row["unit"] != "uM"
            or row["raw_relation"] != "="
            or not math.isclose(
                float(row["target"]), math.log10(float(row["value"])), abs_tol=1e-14
            )
        ):
            raise ValueError("Canonical quantity, source identity or target changed")
    multiplicities = sorted(Counter(r["split_group"] for r in values).values())
    if (
        len(graph_groups) != 5
        or len(set(graph_groups.values())) != 5
        or multiplicities != [1, 1, 1, 1, 6]
    ):
        raise ValueError(
            "Ten observations must represent five families, including six melittin rows"
        )
    if sum(r["record_id"] == NEW_ID for r in values) != 1:
        raise ValueError("The exact newly qualified observation must occur once")
    return values


def source_rows():
    """Revalidate the old cohort and one explicit raw-observation alias."""
    from rdkit import Chem

    review_path = SOURCE / "MHC10_human_candidate_review.json"
    if sha(review_path) != REVIEW_SHA or sha(legacy.__file__) != LEGACY_IMPLEMENTATION_SHA:
        raise ValueError("Pinned source review or legacy helper implementation changed")
    if sha(OLD_TABLE) != OLD_TABLE_SHA:
        raise ValueError("The protected nine-row source table changed")
    old_task, old_values, pins, old_proofs = legacy.load_admission()
    old_table = rows(OLD_TABLE)
    if old_values != old_table or old_task["task_id"] != OLD_TASK:
        raise ValueError("Prepared old nine rows differ from independently revalidated admission")
    pins = dict(pins)
    pins.update(
        {
            review_path: REVIEW_SHA,
            Path(legacy.__file__): LEGACY_IMPLEMENTATION_SHA,
            OLD_TABLE: OLD_TABLE_SHA,
        }
    )
    old_delivery_path = PARENT / "delivery_receipt.json"
    old_delivery = read(old_delivery_path)
    pins[old_delivery_path] = sha(old_delivery_path)
    for name, expected in old_delivery["files"].items():
        path = legacy.project_path(name)
        if (
            not path.resolve().is_relative_to(ROOT.resolve())
            or path.is_symlink()
            or sha(path) != expected
        ):
            raise ValueError("Protected old prepared artifact changed: " + name)
        pins[path] = expected
    review = read(review_path)
    candidates = (
        review["human_original_records"] + review["mouse_original_records_explicitly_out_of_scope"]
    )
    if len(candidates) != 6 or len({r["record_id"] for r in candidates}) != 6:
        raise ValueError("Original MHC10 denominator is not six unique labelled observations")
    candidate = next(r for r in candidates if r["record_id"] == NEW_ID)
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        frozen = dict(db.execute("SELECT * FROM records WHERE record_id=?", (NEW_ID,)).fetchone())
        actual_ids = {
            r[0]
            for r in db.execute(
                "SELECT record_id FROM records WHERE endpoint='MHC10' AND lane='strict_numeric'"
            )
        }
    if actual_ids != {r["record_id"] for r in candidates}:
        raise ValueError("Original MHC10 labelled membership changed")
    archive = read(ROOT / "data/hc10_primary_species_lookup_20261114_v1/pinned_source_links.json")
    obs_path = legacy.project_path(candidate["source_file"])
    assay_path = legacy.project_path(archive["13_experiments.csv"]["source_path"])
    ref_path = legacy.project_path(archive["15_references.csv"]["source_path"])
    compound_path = legacy.project_path(archive["01_compounds.csv"]["source_path"])
    source_objects = {}
    for kind, path, field, value in (
        ("observation", obs_path, "observation_id", candidate["source_observation_id"]),
        ("assay", assay_path, "assay_id", candidate["assay_id"]),
        ("reference", ref_path, "reference_id", candidate["reference"]["reference_id"]),
        ("compound", compound_path, "compound_id", frozen["compound_id"]),
    ):
        raw, source_row = _csv_unique(path, field, value)
        expected = review["source_sha256"].get(str(path.relative_to(ROOT)).replace("/", "\\"))
        if expected is None:
            expected = pins.get(path)
        if expected is None or sha(path) != expected:
            raise ValueError("Raw source file is not independently pinned: " + str(path))
        pins[path] = expected
        source_objects[kind] = dict(
            row=raw,
            source_file=str(path.relative_to(ROOT)),
            source_file_sha256=expected,
            source_row=source_row,
            source_row_convention="one_based_data_row_excluding_header",
            id_field=field,
            id=value,
        )
    observation, assay, reference, compound = [
        source_objects[k]["row"] for k in ("observation", "assay", "reference", "compound")
    ]
    description = candidate["assay_description"]
    graph = canonical(frozen["smiles"])
    if (
        json.loads(frozen["raw_json"]) != observation
        or candidate["native_observation"] != observation
        or candidate["native_assay"] != assay
        or observation["endpoint"] != "MHC10"
        or observation["qualifier"] != "="
        or observation["value"] != "0.350000000000"
        or observation["unit"] != "uM"
        or frozen["raw_value"] != observation["value"]
        or frozen["raw_unit"] != "uM"
        or frozen["value"] != "0.35"
        or frozen["unit"] != "uM"
        or frozen["relation"] != "="
        or frozen["raw_relation"] != "="
        or frozen["source_entry_id"] != "activity:22951890"
        or frozen["assay_id"] != assay["assay_id"]
        or observation["reference_id"] != reference["reference_id"]
        or observation["compound_id"] != compound["compound_id"]
        or reference["pmid"] != "28002968"
        or reference["doi"] != "10.1021/acs.jmedchem.6b00922"
        or assay["description"] != description
        or "concentration required to cause 10% hemoglobin release" not in description
        or "human red blood cells" not in description
        or "after 1 hr" not in description
        or "microplate reader" not in description
        or assay["system_description"] != "Homo sapiens"
        or assay["cell_line"] != "Erythrocyte"
        or canonical(compound["smiles"]) != graph
        or candidate["wholegraph_group"] != MEL_GRAPH
        or candidate["family"] != MEL_GROUP
        or "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest() != MEL_GRAPH
        or Chem.MolToInchiKey(Chem.MolFromSmiles(graph)) != "VDXZNPDIRNWWCW-JFTDCZMZSA-N"
        or candidate["pH"] is not None
    ):
        raise ValueError("Exact MHC10 observation/readout/reference/graph binding changed")
    melittin = next(r for r in old_values if r["canonical_identity_group"] == MEL_GRAPH)
    new = {key: "" for key in old_values[0]}
    new.update(
        record_id=NEW_ID,
        split_group=MEL_GROUP,
        existing_family=MEL_GROUP,
        canonical_identity_group=MEL_GRAPH,
        target=repr(math.log10(0.35)),
        value="0.35",
        unit="uM",
        target_transform="log10",
        structure_eligible="true",
        sequence_eligible="false",
        smiles=graph,
        source_paper="PMID:28002968",
        source_assay_id=assay["assay_id"],
        source_endpoint="MHC10",
        rbc_species="Homo sapiens",
        source_matrix="erythrocytes",
        reported_time_expressions=json.dumps(["1 hr"]),
        raw_scalar="0.350000000000",
        raw_unit="uM",
        raw_relation="=",
        graph_identity_tier=melittin["graph_identity_tier"],
        quantity_evidence="original_activity_explicit_human_RBC_10percent_hemoglobin_release_1hr",
        primary_full_protocol_independently_verified="false",
        existing_feature_record_id=melittin["existing_feature_record_id"],
        source_lineage="original_MHC10_observation_explicit_canonical_quantity_alias_only",
    )
    values = copy.deepcopy(old_values) + [new]
    for row in values:
        row.update(
            canonical_smiles=canonical(row["smiles"]),
            canonical_quantity_id=QUANTITY,
            canonical_species="human",
            canonical_material="erythrocytes",
            source_original_endpoint=row["source_endpoint"],
            standardized_protocol_verified="false",
            minimum_discrete_vs_interpolated_rule_verified="false",
            reported_pH="not_reported" if row["record_id"] == NEW_ID else "",
        )
    if any(
        any(row[key] != original[key] for key in original)
        for row, original in zip(values[:9], old_values, strict=True)
    ):
        raise ValueError("An existing nine-row field was overwritten")
    validate_cohort(values)
    excluded = [
        dict(
            record_id=r["record_id"],
            source_original_endpoint="MHC10",
            reason=(
                "nonhuman_mouse_outside_canonical_human_target"
                if r["species_evidence"] != "Homo sapiens"
                else "acronym_only_no_explicit_10percent_readout_observation_alias_unproven"
            ),
        )
        for r in candidates
        if r["record_id"] != NEW_ID
    ]
    return (
        values,
        pins,
        dict(
            candidate=candidate,
            frozen=frozen,
            source_objects=source_objects,
            excluded_original_records=excluded,
            old_source_proofs=old_proofs,
            protected_old_prepared_files=len(old_delivery["files"]),
        ),
    )


def split_prechecks(values):
    from peptide_modeling_internal_cv import internal_splits
    from peptide_modeling_train import validate_rows

    y, groups = validate_rows(values, "regression")
    checks, signatures = [], set()
    for seed in SEEDS:
        splits = internal_splits(y, groups, "regression", dict(seed=seed, inner_folds=5))
        folds = []
        for fold, (train, test) in enumerate(splits):
            if (
                set(groups[train]) & set(groups[test])
                or len(set(groups[train])) != 4
                or len(np.unique(y[train])) < 2
            ):
                raise ValueError("Invalid actual five-fold molecular training partition")
            folds.append(
                dict(
                    fold=fold,
                    train_record_ids=[values[i]["record_id"] for i in train],
                    validation_record_ids=[values[i]["record_id"] for i in test],
                    train_groups=sorted(set(groups[train])),
                    validation_groups=sorted(set(groups[test])),
                )
            )
        signatures.add(tuple(sorted(tuple(sorted(f["validation_record_ids"])) for f in folds)))
        checks.append(dict(seed=seed, actual_folds=5, folds=folds))
    if len(signatures) != 1:
        raise ValueError("Five groups must retain one LOGO partition across seeds")
    return dict(seeds=checks, distinct_molecular_partitions=1, independent_holdout=False)


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
        if key not in graph_index or canonical(
            old_graphs[key]["canonical_isomeric_smiles"]
        ) != canonical(row["smiles"]):
            raise ValueError("A source graph does not exactly match a frozen parent graph")
        graph_map[key] = old_graphs[key]["canonical_isomeric_smiles"]
    if len(graph_map) != 5:
        raise ValueError("Exactly five old complete graphs must be reused")
    keys = sorted(graph_map)
    configuration = dict(
        version="hc10_human_canonical_graph_slices_v2",
        parent_sha256=before,
        parent_fingerprint=base["input_fingerprint"],
        source_review_sha256=REVIEW_SHA,
        old_nine_table_sha256=OLD_TABLE_SHA,
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
        meta.update(shape=[5, meta["shape"][1]], index="structure_bank_index.tsv")
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
        "bank_scope": "hc10_human_canonical_registered_reference_graphs_only",
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
        schema="hc10_human_canonical_structure_slices_v2",
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=10,
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
        if not payload["valid"].all() or payload["X"].shape != (10, method["dimensions"]):
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
            records=10,
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
            new_observation_aliases=1,
        ),
    )
    return manifest


def _contained(root, name):
    root = Path(root).resolve()
    relative = Path(str(name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Proof path escapes canonical task root")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Proof path is not a contained regular artifact")
    return path


def verify_coverage_proof(root, task, values):
    descriptors = task.get("observation_endpoint_coverage")
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        raise ValueError("Exactly one observation-level coverage descriptor is required")
    descriptor = descriptors[0]
    expected = dict(
        original_endpoint="MHC10",
        canonical_endpoint="HC10",
        canonical_quantity_id=QUANTITY,
        record_ids=[NEW_ID],
        original_labelled_records=6,
        included_original_records=1,
        included_original_molecular_groups=1,
        canonical_molecular_groups=5,
        inference_alias_is_global=False,
        all_original_records_covered=False,
    )
    if any(descriptor.get(k) != v for k, v in expected.items()):
        raise ValueError("Observation coverage scope was enlarged or changed")
    proof_path = _contained(root, descriptor["proof_path"])
    if sha(proof_path) != descriptor["proof_sha256"]:
        raise ValueError("Observation coverage proof changed")
    proof = read(proof_path)
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("canonical_task_id") != task["task_id"]
        or any(proof.get(k) != v for k, v in expected.items())
        or proof.get("canonical_record_ids") != [r["record_id"] for r in values]
    ):
        raise ValueError("Observation proof does not bind the canonical task membership")
    table_path = _contained(root, proof["canonical_source_table"]["path"])
    if (
        str(proof["canonical_source_table"]["path"]) != task["table_path"]
        or sha(table_path) != proof["canonical_source_table"]["sha256"]
    ):
        raise ValueError("Canonical table differs from the observation proof")
    old = _contained(root, proof["parent_nine"]["table_path"])
    if sha(old) != OLD_TABLE_SHA or proof["parent_nine"]["sha256"] != OLD_TABLE_SHA:
        raise ValueError("The old nine-row source table is not byte-preserved")
    old_values = rows(old)
    if len(old_values) != 9 or any(
        any(row[k] != previous[k] for k in previous)
        for row, previous in zip(values[:9], old_values, strict=True)
    ):
        raise ValueError("Original nine source fields changed")
    observations = proof.get("included_observations", [])
    if len(observations) != 1 or observations[0].get("record_id") != NEW_ID:
        raise ValueError("Proof must bind only the exact qualified original observation")
    included = observations[0]
    actual = next(r for r in values if r["record_id"] == NEW_ID)
    for name in (
        "canonical_quantity_id",
        "canonical_smiles",
        "canonical_identity_group",
        "split_group",
        "source_paper",
        "source_assay_id",
        "reported_time_expressions",
        "graph_identity_tier",
    ):
        if included.get(name) != actual[name]:
            raise ValueError("Included observation differs from canonical source: " + name)
    for item in included["source_objects"].values():
        path = _contained(root, item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("Raw source-object snapshot changed")
    native = {
        name: read(_contained(root, item["path"]))
        for name, item in included["source_objects"].items()
    }
    obs, assay, ref, compound = [
        native[k] for k in ("observation", "assay", "reference", "compound")
    ]
    readout = included["readout_evidence"]
    if (
        obs["endpoint"] != "MHC10"
        or obs["observation_id"] != "40000000022951890"
        or obs["value"] != "0.350000000000"
        or obs["qualifier"] != "="
        or obs["unit"] != "uM"
        or obs["assay_id"] != assay["assay_id"]
        or assay["assay_id"] != actual["source_assay_id"]
        or obs["reference_id"] != ref["reference_id"]
        or ref["pmid"] != "28002968"
        or ref["doi"] != "10.1021/acs.jmedchem.6b00922"
        or obs["compound_id"] != compound["compound_id"]
        or compound["smiles"] != actual["canonical_smiles"]
        or hashlib.sha256(compound["smiles"].encode()).hexdigest()
        != included["canonical_smiles_sha256"]
        or included["canonical_identity_group"] != MEL_GRAPH
        or included["split_group"] != MEL_GROUP
        or readout.get("threshold_percent") != 10
        or readout.get("species") != "human"
        or readout.get("matrix") != "erythrocytes"
        or readout.get("pH") is not None
        or readout.get("reported_time_expressions") != ["1 hr"]
        or readout.get("standardized_protocol_verified") is not False
        or readout.get("minimum_discrete_vs_interpolated_verified") is not False
        or readout.get("assay_description") != assay["description"]
        or "concentration required to cause 10% hemoglobin release" not in assay["description"]
        or "human red blood cells" not in assay["description"]
        or "after 1 hr" not in assay["description"]
    ):
        raise ValueError(
            "Original observation/readout/graph proof is not the exact qualified point"
        )
    if len(proof["excluded_original_records"]) != 5 or {
        r["record_id"] for r in proof["excluded_original_records"]
    } | {NEW_ID} != set(proof["original_record_ids"]):
        raise ValueError("Excluded original MHC10 membership changed")
    return proof


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if (output / "task_manifest.json").exists():
        raise FileExistsError("Canonical task preparation already exists; preserve it")
    values, pins, evidence = source_rows()
    output.mkdir(parents=True, exist_ok=True)
    task_path = "task_tables/" + TASK + ".tsv"
    table(output / task_path, values)
    snapshots = output / "task_tables/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OLD_TABLE, snapshots / "old_hc10_nine.tsv")
    source_objects = {}
    for kind, item in evidence["source_objects"].items():
        path = snapshots / (kind + ".json")
        write(path, item["row"])
        source_objects[kind] = {
            **{k: v for k, v in item.items() if k != "row"},
            "path": path.relative_to(output).as_posix(),
            "sha256": sha(path),
            "object_sha256": object_hash(item["row"]),
        }
    write(snapshots / "frozen_record.json", evidence["frozen"])
    candidate = evidence["candidate"]
    included = dict(
        record_id=NEW_ID,
        original_endpoint="MHC10",
        canonical_endpoint="HC10",
        canonical_quantity_id=QUANTITY,
        value="0.35",
        raw_scalar="0.350000000000",
        relation="=",
        raw_relation="=",
        unit="uM",
        raw_unit="uM",
        canonical_smiles=values[-1]["canonical_smiles"],
        canonical_smiles_sha256=MEL_GRAPH.split(":", 1)[1],
        canonical_identity_group=MEL_GRAPH,
        split_group=MEL_GROUP,
        source_paper=values[-1]["source_paper"],
        source_assay_id=values[-1]["source_assay_id"],
        reported_time_expressions=values[-1]["reported_time_expressions"],
        graph_identity_tier=values[-1]["graph_identity_tier"],
        source_objects=source_objects,
        readout_evidence=dict(
            threshold_percent=10,
            species="human",
            matrix="erythrocytes",
            assay_description=candidate["assay_description"],
            reported_time_expressions=["1 hr"],
            pH=None,
            minimum_discrete_vs_interpolated_verified=False,
            standardized_protocol_verified=False,
        ),
    )
    scope = dict(
        original_endpoint="MHC10",
        canonical_endpoint="HC10",
        canonical_quantity_id=QUANTITY,
        record_ids=[NEW_ID],
        original_labelled_records=6,
        included_original_records=1,
        included_original_molecular_groups=1,
        canonical_molecular_groups=5,
        inference_alias_is_global=False,
        all_original_records_covered=False,
    )
    proof = dict(
        schema=PROOF_SCHEMA,
        canonical_task_id=TASK,
        **scope,
        canonical_record_ids=[r["record_id"] for r in values],
        canonical_source_table=dict(path=task_path, sha256=sha(output / task_path), records=10),
        parent_nine=dict(
            task_id=OLD_TASK,
            table_path="task_tables/source_snapshots/old_hc10_nine.tsv",
            original_table_path=str(OLD_TABLE.relative_to(ROOT)),
            sha256=OLD_TABLE_SHA,
        ),
        included_observations=[included],
        original_record_ids=sorted(
            [NEW_ID] + [r["record_id"] for r in evidence["excluded_original_records"]]
        ),
        excluded_original_records=evidence["excluded_original_records"],
        source_pins={str(p): h for p, h in pins.items()},
        warning=WARNING,
    )
    proof_path = output / "task_tables/observation_endpoint_coverage.json"
    write(proof_path, proof)
    write(output / "observation_endpoint_coverage.json", proof)
    definition = dict(
        quantity=QUANTITY,
        canonical_quantity_id=QUANTITY,
        species="Homo sapiens",
        assay_material="reported_erythrocytes",
        target_unit="uM",
        standardized_protocol_verified=False,
        fitted_continuous_threshold_verified=False,
        exposure_time_conditioned=False,
        pH_conditioned=False,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=10,
        distinct_molecular_partitions=1,
        inference_alias_is_global=False,
        all_original_MHC10_records_covered=False,
        source_original_endpoint_preserved=True,
        registered_reference_graph_identity=True,
        primary_sample_stereopurity_or_salt_state_verified=False,
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="HC10",
        task_kind="regression",
        status="eligible",
        training_admitted=True,
        table_path=task_path,
        n_records=10,
        n_groups=5,
        unit="uM",
        target_transform="log10",
        inverse_transform="pow10",
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        requested_inner_folds=5,
        seeds=list(SEEDS),
        distinct_molecular_partitions=1,
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
        dict(schema="hc10_human_canonical_admission_v2", tasks=[task]),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    write(output / "fivefold_prechecks.json", split_prechecks(values))
    features = slice_features(
        values, source, parent, output, expected_parent_sha=pins[PARENT / "features/manifest.json"]
    )
    # The old slice itself pins the earlier unified parent, whose48 artifacts remain unchanged.
    from peptide_modeling_features import _verify_artifacts

    unified = read(
        Path(read(PARENT / "features/manifest.json")["cache_dir"]) / "source_parent_manifest.json"
    )
    _verify_artifacts(unified)
    config = dict(
        dataset="data/dataset_reclean_20261113_v1/catalogue.sqlite",
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
        required_seeds=list(SEEDS),
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        mandatory_postprocessing="record_and_molecule_equal_OOF_diagnostics",
    )
    write(output / "recommended_training_config.json", config)
    write(output / "config.json", config)
    if any(sha(p) != expected for p, expected in pins.items()):
        raise ValueError("A protected source or old parent changed during preparation")
    write(
        output / "source_verification.json",
        dict(
            status="verified",
            source_pins=len(pins),
            source_inputs_sha256={str(p): h for p, h in pins.items()},
            old_nine_fields_unchanged=True,
            raw_source_objects_unique_and_exact=True,
            old_prepared_files_unchanged=evidence["protected_old_prepared_files"],
            original_unified_artifacts_unchanged=len(unified["artifact_hashes"]),
            new_record=NEW_ID,
            observation_alias_scope=scope,
            remote_operations=False,
            models_fitted=0,
        ),
    )
    write(
        output / "admission_receipt.json",
        dict(
            status="prepared_not_trained",
            tasks=1,
            records=10,
            molecules=5,
            multiplicities=[1, 1, 1, 1, 6],
            new_feature_computations=0,
            new_graphs=0,
            canonical_quantity_id=QUANTITY,
            original_MHC10_included_records=1,
            original_MHC10_labelled_records=6,
            original_MHC10_included_molecules=1,
            original_MHC10_all_records_covered=False,
            original_MHC10_independent_model=False,
        ),
    )
    (output / "REPORT.md").write_text(
        "# Canonical nominal human-RBC10% release concentration\n\n"
        "10 observations /5 molecular families, one contributing6rows; one canonical HC10target. "
        "Exactly one original MHC10 observation of six is included under an observation-level "
        "explicit10%humanRBCreadout proof. Five other original records remain uncovered. "
        "No independent MHC10-only five-molecule model or global inference alias is created.\n\n"
        + WARNING
        + "\n\nAll9original cells/provenance preserved; primaryPexiganan staysHD10. "
        "Old5complete graph vectors are copied byte-for-byte with10aliases, no feature extraction. "
        "Ten structural methods+fusion, eight models, Optuna128, seeds42/2026/3407, "
        "strict internal5fold "
        "with oneLOGOpartition. Mandatory selected-OOF record/moleculeequal diagnostics and source "
        "bindings; no independent holdout or model-selection repetition in reporting.\n",
        encoding="utf-8",
    )
    diagnostic = output / "source_preparation_diagnostic.txt"
    if diagnostic.exists():
        history = output / "diagnostic_history"
        history.mkdir(exist_ok=True)
        diagnostic.rename(history / diagnostic.name)
    write(
        output / "delivery_receipt.json",
        dict(
            status="prepared_not_trained",
            task_id=TASK,
            records=10,
            molecules=5,
            feature_fingerprint=features["input_fingerprint"],
            files={
                p.relative_to(output).as_posix(): sha(p)
                for p in output.rglob("*")
                if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
            },
            historical_intermediate_artifacts_not_execution_inputs={
                p.relative_to(output).as_posix(): sha(p)
                for p in (output / "diagnostic_history").rglob("*")
                if p.is_file()
            },
        ),
    )
    return task


def molecule_diagnostics(source_rows, predictions):
    """Verify source/fold identity, then compare row and molecule-equal log errors."""
    validate_cohort(source_rows, verify_chemistry=False)
    source = {row["record_id"]: row for row in source_rows}
    if (
        len(predictions) != len(source_rows)
        or len({row["record_id"] for row in predictions}) != len(predictions)
        or {row["record_id"] for row in predictions} != set(source)
    ):
        raise ValueError("OOF record identity differs from the source cohort")
    grouped, group_folds = defaultdict(list), {}
    for row in predictions:
        old = source[row["record_id"]]
        fold, target, prediction = int(row["fold"]), float(row["target"]), float(row["prediction"])
        group = row["split_group"]
        if (
            group != old["split_group"]
            or target != float(old["target"])
            or not 0 <= fold < 5
            or not math.isfinite(prediction)
        ):
            raise ValueError("OOF source target, group, fold or prediction changed")
        if group_folds.setdefault(group, fold) != fold:
            raise ValueError("An OOF molecular group crossed folds")
        grouped[group].append(
            dict(
                record_id=row["record_id"],
                target=target,
                prediction=prediction,
                source_paper=old["source_paper"],
                reported_time_expressions=old["reported_time_expressions"],
                graph_identity_tier=old["graph_identity_tier"],
                source_original_endpoint=old.get("source_original_endpoint"),
                canonical_quantity_id=old["canonical_quantity_id"],
            )
        )
    if set(group_folds.values()) != set(range(5)):
        raise ValueError("OOF does not contain five actual held-out molecular groups")
    all_target = [row["target"] for values in grouped.values() for row in values]
    all_prediction = [row["prediction"] for values in grouped.values() for row in values]
    per_group = []
    for group, values in sorted(grouped.items()):
        p = np.asarray([row["prediction"] for row in values])
        y = np.asarray([row["target"] for row in values])
        if not np.allclose(p, p[0], rtol=1e-7, atol=1e-8):
            raise ValueError(
                "One complete graph received inconsistent structure-only OOF predictions"
            )
        metrics = error_metrics(y, p)
        per_group.append(
            dict(
                split_group=group,
                observations=len(values),
                fold=group_folds[group],
                **metrics,
                target_min=float(y.min()),
                target_max=float(y.max()),
                predictions=values,
            )
        )
    mae_scale = max(group["mae"] for group in per_group)
    rmse_scale = max(group["rmse"] for group in per_group)
    molecule_mae = (
        float(np.mean([g["mae"] / mae_scale for g in per_group])) * mae_scale if mae_scale else 0.0
    )
    molecule_rmse = (
        float(np.sqrt(np.mean([(g["rmse"] / rmse_scale) ** 2 for g in per_group]))) * rmse_scale
        if rmse_scale
        else 0.0
    )
    return dict(
        status="verified",
        records=10,
        molecules=5,
        metrics_space="log10_uM",
        record_weighted=error_metrics(all_target, all_prediction),
        molecule_equal=dict(mae=molecule_mae, rmse=molecule_rmse),
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

    if sha(legacy.__file__) != LEGACY_IMPLEMENTATION_SHA:
        raise ValueError("Legacy numeric helper implementation changed")
    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "internal_campaign_status.json")
    if len(journal["rounds"]) != 3 or {entry["seed"] for entry in journal["rounds"]} != set(SEEDS):
        raise ValueError("Molecular reporting requires all three finite seed runs")
    reports = []
    for entry in journal["rounds"]:
        if entry["status"] != "completed":
            raise ValueError("Molecular reporting requires completed internal seed work")
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        if task["task_id"] != TASK:
            raise ValueError("Unexpected task in molecular reporting")
        source_rows = load_rows(run / task["table_path"])
        verify_coverage_proof(run, task, source_rows)
        result = read(run / "results/internal_cv" / (TASK + ".json"))
        verify_internal_task(task, source_rows, portable_features(run), run, result)
        path = Path(result["artifacts"]["internal_oof"])
        if sha(path) != result["artifact_hashes"][str(path)]["sha256"]:
            raise ValueError("OOF artifact changed before molecular diagnostics")
        report = molecule_diagnostics(source_rows, load_rows(path))
        for name in ("mae", "rmse"):
            if not math.isclose(
                report["record_weighted"][name],
                result["internal_cv_selection_metrics"][name],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Record-weighted diagnostics do not reproduce the saved OOF metrics"
                )
        report.update(
            seed=entry["seed"],
            result_sha256=sha(run / "results/internal_cv" / (TASK + ".json")),
            oof_sha256=sha(path),
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=sha(run / task["table_path"]),
            canonical_quantity_id=QUANTITY,
        )
        reports.append(report)
    payload = dict(
        schema=REPORT_SCHEMA,
        canonical_quantity_id=QUANTITY,
        status="verified",
        task_id=TASK,
        seeds=list(SEEDS),
        reports=reports,
        model_selection_repeated=False,
        independent_holdout=False,
        warning=WARNING,
    )
    destination = root / "hc10_molecular_diagnostics.json"
    atomic_json(destination, payload)
    durable.mkdir(parents=True, exist_ok=True)
    target = durable / destination.name
    shutil.copy2(destination, target)
    if sha(destination) != sha(target):
        raise ValueError("Durable molecular diagnostics differ")
    receipt = dict(
        status="verified",
        path=str(destination),
        sha256=sha(destination),
        implementation_sha256=sha(__file__),
    )
    receipt.update(
        durable_path=str(target),
        durable_sha256=sha(target),
        implementation_dependencies_sha256={
            "peptide_modeling_hc10_human_bridge.py": LEGACY_IMPLEMENTATION_SHA
        },
    )
    journal["molecular_reporting"] = receipt
    if all(entry["status"] == "completed" for entry in journal["rounds"]):
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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.run_followup or args.diagnostics_only:
        if args.config is None:
            parser.error("--config is required for followup reporting")
        result = run_followup(args.config, diagnostics_only=args.diagnostics_only)
    else:
        task = prepare(parent=args.parent, output=args.output)
        result = dict(
            status="prepared_not_trained",
            task_id=task["task_id"],
            records=10,
            canonical_molecular_groups=5,
            original_MHC10_included_records=1,
            original_MHC10_labelled_records=6,
            inference_alias_is_global=False,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
