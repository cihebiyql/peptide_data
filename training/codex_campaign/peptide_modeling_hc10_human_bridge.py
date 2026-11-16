"""Prepare and report a small, source-curated human-erythrocyte HC10 benchmark.

Preparation only copies existing structure vectors. The optional training entry
delegates to the established internal-CV controller; this module adds read-only
molecule-equal OOF diagnostics, never another model-selection criterion.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import signal
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hc10_conditional_primary_review_20261114_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_hc10_human_followup_20261114_v1"
RECEIPT_SHA = "236db50be7156db991df54ba99bfe3229a7f24f470f4840d7d311149970e2034"
TASK = "TASK_HC10_HUMAN_SOURCE_799fd3266d92284a"
PRIMARY = "PRIMARY:PMID18287037:T1:PEXIGANAN:HD10"
OLD_UNKNOWN = "NEW:07:40000000012053095"
SEEDS = (42, 2026, 3407)
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
WARNING = (
    "Source-curated nominal human-RBC HC10 only. Time, preparation, readout and study "
    "are uncontrolled; reference graphs do not establish primary sample stereopurity "
    "or salt state. Nine observations are five molecules, one contributing five rows. "
    "The three seeds repeat one LOGO partition. Tuning CV is selection-optimistic; "
    "no independent holdout, time/species effect or clinical generalization is claimed."
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(values)


def project_path(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def canonical(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid registered source graph")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def csv_select(path, field, requested):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        found = {row[field]: row for row in csv.DictReader(handle) if row[field] in requested}
    if set(found) != set(requested):
        raise ValueError("Pinned source CSV lacks requested original identities")
    return found


def verify_source(source=SOURCE):
    source = Path(source)
    if sha(source / "delivery_receipt.json") != RECEIPT_SHA:
        raise ValueError("Peer-review receipt differs from the fixed admission")
    pins = {}
    for relative, expected in read(source / "delivery_receipt.json")["files"].items():
        item = Path(relative)
        if item.is_absolute() or ".." in item.parts:
            raise ValueError("Unsafe peer-review artifact path")
        path = source / item
        if sha(path) != expected:
            raise ValueError("Peer-review artifact changed: " + relative)
        pins[path] = expected
    for name, expected in read(source / "source_pins.json").items():
        path = project_path(name)
        if sha(path) != expected:
            raise ValueError("Underlying HC10 source changed: " + name)
        pins[path] = expected
    return pins


def validate_primary(primary, reference, compound):
    if (
        primary["candidate_id"] != PRIMARY
        or primary["raw_value"] != "73"
        or primary["raw_relation"] != "="
        or primary["raw_unit"] != "uM"
        or primary["reported_assay_context"]["rbc_species"] != "human"
        or primary["source_H50_comparator_preserved"]
        != {"raw_value": "200", "raw_relation": ">", "raw_unit": "uM", "is_exact_point": False}
        or primary["old_observation_not_modified"] != OLD_UNKNOWN
        or primary["old_observation_species_still_unknown"] is not True
        or primary["primary_sequence_with_terminal"] != "GIGKFLKKAKKFGKAFVKILKK-NH2"
        or primary["explicit_all_L_statement_in_accessed_primary_main_text"] is not False
        or primary["N_terminal_specific_capping_explicit"] is not False
    ):
        raise ValueError("Independent primary point or its identity limits changed")
    if (
        reference["CID"] != 16132253
        or compound["name"] != "PEXIGANAN"
        or reference["InChIKey"] != compound["inchikey"]
        or reference["MolecularFormula"] != compound["molecular_formula"]
        or canonical(reference["SMILES"]) != canonical(compound["smiles"])
        or canonical(reference["SMILES"])
        != canonical(primary["existing_structure_mapping"]["isomeric_smiles"])
    ):
        raise ValueError("Named registered PEX reference identity differs")


def validate_cohort(values):
    ids = [row["record_id"] for row in values]
    if len(values) != 9 or len(set(ids)) != 9 or PRIMARY not in ids or OLD_UNKNOWN in ids:
        raise ValueError("HC10 admission must retain the exact nine distinct candidate records")
    by_graph, by_group = defaultdict(set), defaultdict(set)
    for row in values:
        if row["rbc_species"] != "Homo sapiens" or row["source_matrix"] != "erythrocytes":
            raise ValueError("Outside the predeclared human-erythrocyte scope")
        if row["unit"] != "uM" or row["raw_relation"] != "=":
            raise ValueError("HC10 source label is not the declared molar point")
        value, target = float(row["value"]), float(row["target"])
        if not math.isfinite(value) or value <= 0 or not math.isfinite(target):
            raise ValueError("Invalid positive HC10 point")
        if not math.isclose(math.log10(value), target, rel_tol=1e-13, abs_tol=1e-13):
            raise ValueError("HC10 transformed label changed")
        if row["split_group"] != row["existing_family"]:
            raise ValueError("An existing molecular family was split")
        by_graph[row["canonical_identity_group"]].add(row["split_group"])
        by_group[row["split_group"]].add(row["canonical_identity_group"])
    if len(by_graph) != 5 or len(by_group) != 5 or any(len(v) != 1 for v in by_graph.values()):
        raise ValueError("Expected five complete graphs and five conservative families")
    if any(len(v) != 1 for v in by_group.values()):
        raise ValueError("HC10 family/graph mapping differs from the reviewed cohort")
    if sorted(Counter(row["split_group"] for row in values).values()) != [1, 1, 1, 1, 5]:
        raise ValueError("Source observation multiplicity changed")


def validate_archived(row, frozen, observation, compound, assay, reference, alias):
    raw = json.loads(frozen["raw_json"])
    if (
        raw != observation
        or frozen["record_id"] != row["record_id"]
        or raw["qualifier"] != frozen["raw_relation"]
        or frozen["raw_relation"] != frozen["relation"]
        or frozen["relation"] != "="
        or frozen["unit"] != row["unit"]
        or Decimal(frozen["value"]) != Decimal(row["value"])
        or frozen["raw_value"] != row["raw_scalar"]
        or frozen["raw_unit"] != row["raw_unit"]
        or frozen["assay_id"] != row["source_assay_id"]
        or frozen["canonical_identity_group"] != row["canonical_identity_group"]
        or alias["existing_family"] != row["split_group"]
        or not alias["RBC_species"].startswith("Homo sapiens")
        or alias["source_matrix"] != row["source_matrix"]
        or alias["reported_time_expressions"] != row["reported_time_expressions"]
        or row["source_paper"] != "PMID:" + reference["pmid"]
        or alias["assay_description"] not in (assay["description"], assay["title"])
        or canonical(frozen["smiles"]) != canonical(compound["smiles"])
        or canonical(frozen["smiles"]) != canonical(row["smiles"])
        or frozen["structure_model_eligible"] not in (1, True, "1", "true")
    ):
        raise ValueError("Archived HC10 point, assay, reference, graph or family changed")
    unit = row["raw_unit"].replace("µ", "u").replace("μ", "u").strip().lower()
    if unit == "um":
        expected, policy = Decimal(row["raw_scalar"]), "source_uM_preserved"
    else:
        if unit not in {"ug ml-1", "ug/ml", "ug.ml-1"}:
            raise ValueError("Unsupported original HC10 unit")
        if compound["molecular_weight_unit"].strip() != "g/mol":
            raise ValueError("HC10 mass normalization requires explicit source g/mol")
        weight = Decimal(compound["molecular_weight"])
        if weight <= 0:
            raise ValueError("Invalid source molecular weight")
        expected = Decimal(row["raw_scalar"]) * 1000 / weight
        policy = "source_mass_scalar_and_explicit_g_per_mol"
    if not math.isclose(float(expected), float(row["value"]), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("HC10 original-unit normalization differs")
    return {"record_id": row["record_id"], "normalization_policy": policy}


def load_admission(source=SOURCE):
    source = Path(source)
    pins = verify_source(source)
    task = copy.deepcopy(read(source / "candidate_task_manifest.json")["tasks"][0])
    if task["task_id"] != TASK or task["status"] != "candidate_not_admitted":
        raise ValueError("Unexpected peer-review task")
    values = rows(source / task["table_path"])
    validate_cohort(values)
    review = read(source / "review.json")
    if review["identity_reference_mapping_accepted_at_stated_tier"] is not True:
        raise ValueError("Registered identity review was not accepted")
    trace = ROOT / "data/hc10_cited_primary_trace_20261114_v1"
    primary = read(trace / "primary_new_observation_candidate.json")
    if sha(trace / "request01_ref3_PMC2268539.html") != primary["source_html_sha256"]:
        raise ValueError("Primary article bytes changed")
    archive = read(ROOT / "data/hc10_primary_species_lookup_20261114_v1/pinned_source_links.json")
    reference = read(source / "pubchem_pexiganan.json")["PropertyTable"]["Properties"]
    if len(reference) != 1:
        raise ValueError("Registered reference lookup is not unique")
    validate_primary(primary, reference[0], archive["01_compounds.csv"]["row"])
    aliases = {
        item["record_id"]: item
        for item in rows(
            ROOT
            / "data/remaining_safety_source_audit_20261114_v1/supplemental_review"
            / "HC10_alias_context_records.tsv"
        )
    }
    conn = sqlite3.connect(
        "file:" + str(ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite") + "?mode=ro",
        uri=True,
    )
    conn.row_factory = sqlite3.Row
    old = {
        item["record_id"]: dict(
            conn.execute(
                "select * from records_with_structure_qc where record_id=?", (item["record_id"],)
            ).fetchone()
        )
        for item in values
        if item["record_id"] != PRIMARY
    }
    conn.close()
    compounds = csv_select(
        project_path(archive["01_compounds.csv"]["source_path"]),
        "compound_id",
        {item["compound_id"] for item in old.values()},
    )
    assays = csv_select(
        project_path(archive["13_experiments.csv"]["source_path"]),
        "assay_id",
        {item["assay_id"] for item in old.values()},
    )
    refs = {json.loads(item["raw_json"])["reference_id"] for item in old.values()}
    references = csv_select(
        project_path(archive["15_references.csv"]["source_path"]), "reference_id", refs
    )
    observations = {}
    for filename in {item["input_file"] for item in old.values()}:
        wanted = {
            json.loads(item["raw_json"])["observation_id"]
            for item in old.values()
            if item["input_file"] == filename
        }
        observations.update(csv_select(project_path(filename), "observation_id", wanted))
    proofs = []
    for row in values:
        if row["record_id"] == PRIMARY:
            mapping = primary["existing_structure_mapping"]
            if (
                row["value"] != "73"
                or row["source_endpoint"] != "HD10"
                or row["existing_feature_record_id"] != OLD_UNKNOWN
                or row["split_group"] != mapping["existing_family"]
                or row["canonical_identity_group"] != mapping["canonical_identity_group"]
                or canonical(row["smiles"]) != canonical(reference[0]["SMILES"])
            ):
                raise ValueError("New primary observation was changed or attached to the old label")
            proofs.append(
                {
                    "record_id": PRIMARY,
                    "normalization_policy": "primary_uM_no_conversion",
                    "primary_sample_stereo_verified": False,
                }
            )
        else:
            f = old[row["record_id"]]
            raw = json.loads(f["raw_json"])
            proofs.append(
                validate_archived(
                    row,
                    f,
                    observations[raw["observation_id"]],
                    compounds[f["compound_id"]],
                    assays[f["assay_id"]],
                    references[raw["reference_id"]],
                    aliases[row["record_id"]],
                )
            )
    task.update(
        status="eligible", training_admitted=True, table_path="task_tables/" + TASK + ".tsv"
    )
    task["task_definition"].update(
        required_reporting=review["required_reporting"],
        optimization_metric="record_weighted_internal_log_MAE_not_molecule_equal_selection",
        molecular_equal_diagnostics_required=True,
        primary_sample_stereochemistry_independently_measured=False,
        reference_graph_identity_verified=True,
        scientific_warning=WARNING,
    )
    return task, values, pins, proofs


def split_prechecks(values, expected):
    from peptide_modeling_internal_cv import internal_splits
    from peptide_modeling_train import validate_rows

    y, groups = validate_rows(values, "regression")
    checks = []
    signatures = set()
    for seed in SEEDS:
        splits = internal_splits(y, groups, "regression", {"inner_folds": 5, "seed": seed})
        memberships = []
        original = next(item for item in expected["seeds"] if item["seed"] == seed)
        for fold, (train, valid) in enumerate(splits):
            train_ids = [values[i]["record_id"] for i in train]
            valid_ids = [values[i]["record_id"] for i in valid]
            if (
                train_ids != original["folds"][fold]["train_record_ids"]
                or valid_ids != original["folds"][fold]["validation_record_ids"]
                or set(groups[train]) & set(groups[valid])
                or len(np.unique(y[train])) < 2
            ):
                raise ValueError("Actual grouped folds differ from the independently reviewed plan")
            memberships.append(dict(train_record_ids=train_ids, validation_record_ids=valid_ids))
        signatures.add(
            tuple(sorted(tuple(sorted(item["validation_record_ids"])) for item in memberships))
        )
        checks.append(dict(seed=seed, actual_folds=5, folds=memberships))
    if len(signatures) != 1:
        raise ValueError("The five-group task unexpectedly reports independent partitions")
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
    reuse = read(Path(source) / "existing_structure_reuse_plan.json")
    if reuse["feature_fingerprint"] != base["input_fingerprint"]:
        raise ValueError("Parent feature fingerprint differs from the peer review")
    links = reuse["links"]
    by_id = {item["record_id"]: item for item in links}
    if len(links) != len(by_id) or set(by_id) != {row["record_id"] for row in values}:
        raise ValueError("Reviewed source/feature aliases are not one-to-one")
    old_records = {item["record_id"]: item for item in rows(root / base["record_index"])}
    old_graphs = {item["key"]: item for item in rows(root / "structure_source_smiles.tsv")}
    graph_index = {
        item["key"]: int(item["row_idx"]) for item in rows(root / groups["morgan"]["index"])
    }
    graph_map = {}
    for row in values:
        link = by_id[row["record_id"]]
        key = row["canonical_identity_group"]
        old = old_records[link["existing_feature_record_id"]]
        if (
            old["structure_key"] != key
            or link["structure_key"] != key
            or graph_index[key] != link["existing_structure_row_idx"]
            or canonical(old_graphs[key]["canonical_isomeric_smiles"]) != canonical(row["smiles"])
        ):
            raise ValueError("Reviewed parent feature row or full graph changed")
        graph_map[key] = old_graphs[key]["canonical_isomeric_smiles"]
    keys = sorted(graph_map)
    configuration = dict(
        version="hc10_human_frozen_graph_slices_v1",
        parent_sha256=before,
        parent_fingerprint=base["input_fingerprint"],
        source_receipt_sha256=RECEIPT_SHA,
        source_table_sha256=sha(Path(source) / "candidate_task_tables" / (TASK + ".tsv")),
        implementation_sha256=sha(__file__),
        selected_graphs=keys,
        feature_computations=0,
        sequence_or_lossy_projection=False,
        new_primary_alias=PRIMARY,
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
        "bank_scope": "hc10_human_registered_reference_graphs_only",
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
        schema="hc10_human_structure_slices_v1",
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=9,
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
        if not payload["valid"].all() or payload["X"].shape != (9, method["dimensions"]):
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
            records=9,
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
            new_primary_aliases=1,
        ),
    )
    return manifest


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    source, output = Path(source), Path(output)
    for protected in (source.resolve(), Path(parent).resolve()):
        if (
            output.resolve() == protected
            or output.resolve() in protected.parents
            or protected in output.resolve().parents
        ):
            raise ValueError("Admission output cannot overlap protected source or parent bank")
    if (output / "delivery_receipt.json").exists():
        raise FileExistsError("Frozen HC10 preparation must not be overwritten")
    task, values, pins, proofs = load_admission(source)
    output.mkdir(parents=True, exist_ok=True)
    table(output / task["table_path"], values)
    write(
        output / "task_manifest.json", dict(schema="hc10_human_source_admission_v1", tasks=[task])
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    write(
        output / "source_verification.json",
        dict(
            status="verified",
            rows=proofs,
            primary_old_species_not_patched=True,
            observed_records=9,
            molecules=5,
            source_inputs_sha256={str(p): h for p, h in pins.items()},
            scientific_warning=WARNING,
        ),
    )
    write(
        output / "fivefold_prechecks.json",
        split_prechecks(values, read(source / "fivefold_prechecks.json")),
    )
    features = slice_features(
        values, source, parent, output, expected_parent_sha=pins[PARENT / "features/manifest.json"]
    )
    config = dict(
        resources=dict(training_host="node23", n_jobs=1, nice=10, cpu_budget=1),
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
            seed=42,
            min_rows=5,
            min_groups=5,
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
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError("A protected source changed during admission")
    write(
        output / "admission_receipt.json",
        dict(
            status="prepared_not_trained",
            tasks=1,
            records=9,
            molecular_groups=5,
            primary_new_record=PRIMARY,
            source_pins_unchanged=len(pins),
            new_graphs=0,
            new_feature_computations=0,
            models_fitted=0,
            remote_operations=False,
        ),
    )
    return task


def error_metrics(target, prediction):
    """Avoid intermediate squares/sums overflowing for finite log-scale errors."""
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite diagnostic source or prediction")
    scale = max(float(np.abs(target).max()), float(np.abs(prediction).max()))
    normalized = (prediction / scale - target / scale) if scale else np.zeros(len(target))
    result = dict(
        mae=float(np.mean(np.abs(normalized))) * scale,
        rmse=float(np.sqrt(np.mean(normalized * normalized))) * scale,
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Diagnostic error is outside the representable numeric range")
    return result


def molecule_diagnostics(source_rows, predictions):
    """Verify source/fold identity, then compare row and molecule-equal log errors."""
    validate_cohort(source_rows)
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
        records=9,
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

    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "internal_campaign_status.json")
    if {entry["seed"] for entry in journal["rounds"]} != set(SEEDS):
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
        )
        reports.append(report)
    payload = dict(
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
    journal["molecular_reporting"] = receipt
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-followup", action="store_true")
    mode.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.source, args.parent, args.output)
        print(json.dumps(dict(status="prepared_not_trained", task_id=result["task_id"])))
    else:
        if args.config is None:
            parser.error("Training/reporting modes require --config")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        print(json.dumps(run_followup(args.config, args.diagnostics_only)))


if __name__ == "__main__":
    main()
