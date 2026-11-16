"""Source-native MHC10 annotations with explicit RBC species and whole-molecule splits.

The target is a reported annotation, not a harmonized continuous HC10 quantity.
Preparation checks chemistry locally; training/reporting load frozen numeric arrays.
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
import signal
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/peptide_modeling_remaining_gap_contract_20261115_v1"
DRAMP = ROOT / "data/dramp_mhc10_followup_20261114_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_mhc10_native_followup_20261115_v1"
OLD_TASK = "TASK_a6776cdc37c9d693d9ed675c"
OLD_TABLE = ROOT / "data/peptide_modeling_pk_quality_followup_v1/task_tables" / (OLD_TASK + ".tsv")
OLD_TABLE_SHA = "248ee6fd81f49c3604e152ee81d84fea4807b02bb5feb4ef4228aad22623f478"
REVIEW_SHA = "0fce12f02dec174df639c21f321ff0ad136f48e1430da4ebfde656ff4e57f53f"
LABELS_SHA = "6ada80bbec5cb3d959f1cabb34cdd2a428899e68de44374fad9470289035ef0b"
DRAMP_MANIFEST_SHA = "7858b1693112a8ad2241be8d4b415421afe689ef966bf3e403c0295307b8d668"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
MEL_GRAPH = "CANON_SMILES:f9420c9f1b030adf9d9fe7626706f6eca5574280a79f1daea103aba9000109c0"
MEL_GROUP = "FAMILY:e3630b19efc0fb3b36837673"
QUANTITY = "native_source_reported_MHC10_concentration_annotation"
SCHEMA = "mhc10_native_species_conditioned_annotation_v1"
REPORT_SCHEMA = "mhc10_native_species_molecular_diagnostics_v1"
PROOF_SCHEMA = "mhc10_native_observation_source_coverage_v1"
SEEDS = (42, 2026, 3407)
CONTEXT_GROUP = "rbc_species_context"
CONTEXT_NAMES = [CONTEXT_GROUP + ":human", CONTEXT_GROUP + ":mouse"]
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
POINTS = {
    "NEW:06:40000000024425451": ("2", "human", "35294199", "6.5"),
    "NEW:06:40000000024425438": ("4", "human", "35294199", "7.4"),
    "NEW:07:40000000022951890": ("0.35", "human", "28002968", ""),
    "NEW:07:40000000019038346": ("1", "human", "30958004", ""),
    "NEW:07:40000000023221359": ("2.26", "mouse", "34180670", ""),
    "NEW:07:40000000022808186": ("2", "mouse", "33422980", ""),
}
WARNING = (
    "Source-native MHC10 concentration annotations, not a harmonized continuous HC10 or "
    "independently standardized minimum/discrete threshold. Species is an explicit input; "
    "time, pH, preparation, readout and estimation rules remain partly unknown and are not "
    "controlled model inputs. All six original observations are one melittin molecular group. "
    "Both mouse observations belong only to that group: an unseen-melittin fold has no mouse "
    "training support. No validated mouse effect, cross-molecule species correction, clinical "
    "or cross-protocol generalization. DRAMP21 are one paper's related peptide series; exact "
    "graph/FAMILY grouping does not prove homology/series isolation. Inner OOF selects the "
    "model; outer holdout evaluates the outer-train model. Separate all27-data refits have "
    "no independent performance score on their now-included records. No global HC10 alias; "
    "the separate canonical HC10 one-of-six mapping is unchanged."
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


TASK = (
    "TASK_MHC10_NATIVE_"
    + object_hash(
        dict(quantity=QUANTITY, review=REVIEW_SHA, old21=OLD_TABLE_SHA, original_ids=sorted(POINTS))
    )[:24]
)


def contained(root, relative):
    root = Path(root).resolve()
    relative = Path(str(relative).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Native MHC10 proof escapes the isolated task root")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Native MHC10 source must be a contained regular artifact")
    return path


def canonical(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Source complete graph cannot be parsed")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def species_context(species):
    if species not in {"human", "mouse"}:
        raise ValueError("Explicit source-known human or mouse RBC species is mandatory")
    return np.asarray([species == "human", species == "mouse"], dtype=np.float32)


def validate_cohort(values, *, verify_chemistry=True):
    if len(values) != 27 or len({r["record_id"] for r in values}) != 27:
        raise ValueError("Native MHC10 requires exactly27 unique source observations")
    graph_families = {}
    for row in values:
        graph = row["canonical_smiles"]
        key = "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
        if key != row["canonical_identity_group"] or not graph:
            raise ValueError("Source canonical graph hash changed")
        if verify_chemistry:
            from rdkit import Chem

            molecule = Chem.MolFromSmiles(row["smiles"])
            if (
                molecule is None
                or canonical(row["smiles"]) != graph
                or any(
                    tag == "?"
                    for _, tag in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
                )
            ):
                raise ValueError("Source graph needs complete assigned stereochemistry")
        if graph_families.setdefault(key, row["split_group"]) != row["split_group"]:
            raise ValueError("A complete graph crosses whole-molecule groups")
        species_context(row["rbc_species"])
        if (
            row["canonical_quantity_id"] != QUANTITY
            or row["unit"] != "uM"
            or row["native_metric_annotation"] != "MHC10"
            or row["rbc_species_eligible"] != "true"
            or row["rbc_material"] != "erythrocytes"
            or row["target_transform"] != "log10"
            or row["raw_relation"] != "="
            or row["minimum_discrete_vs_interpolated_verified"] != "false"
            or row["standardized_10percent_quantity_verified"] != "false"
            or row["explicit_10percent_readout"]
            != str(row["record_id"] == "NEW:07:40000000022951890").lower()
            or not math.isclose(
                float(row["target"]), math.log10(float(row["value"])), rel_tol=0, abs_tol=1e-10
            )
        ):
            raise ValueError("Native annotation quantity, exact target or input context changed")
        if row["record_id"] in POINTS:
            value, species, pmid, ph = POINTS[row["record_id"]]
            if (
                Decimal(row["value"]) != Decimal(value)
                or row["rbc_species"] != species
                or row["source_pmid"] != pmid
                or row["reported_pH"] != ph
                or row["reported_exposure_h"] != "1"
                or key != MEL_GRAPH
                or row["split_group"] != MEL_GROUP
                or row["source_original_endpoint"] != "MHC10"
                or row["source_stratum"] != "Melittin_CHEMBL"
            ):
                raise ValueError("One of the six original native MHC10 observations changed")
        elif (
            row["source_original_endpoint"] != "hemolysis_MHC10"
            or row["rbc_species"] != "human"
            or row["source_pmid"] != "23894079"
            or row["reported_pH"] != ""
            or row["reported_exposure_h"] != ""
            or row["source_stratum"] != "DRAMP21_PMID23894079"
        ):
            raise ValueError("The21 native DRAMP source annotations changed")
    if (
        len(graph_families) != 22
        or len(set(graph_families.values())) != 22
        or set(POINTS) - {r["record_id"] for r in values}
        or Counter(r["rbc_species"] for r in values) != {"human": 25, "mouse": 2}
        or Counter(r["split_group"] for r in values)[MEL_GROUP] != 6
    ):
        raise ValueError("27rows/22molecular groups/25human/2mouse invariant changed")


def _csv_matches(path, field, keys):
    matched = defaultdict(list)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for data_row, row in enumerate(csv.DictReader(stream), start=2):
            if row.get(field) in keys:
                matched[row[field]].append((row, data_row))
    if set(matched) != set(keys) or any(len(v) != 1 for v in matched.values()):
        raise ValueError("Exact source object is missing or duplicated")
    return {key: value[0] for key, value in matched.items()}


def source_rows(source=SOURCE, dramp=DRAMP):
    """Re-read the fixed raw sources; never infer species or labels from a peptide name."""
    source, dramp = Path(source), Path(dramp)
    pins = {
        source / "MHC10_human_candidate_review.json": REVIEW_SHA,
        source / "source_labelled_records.json": LABELS_SHA,
        dramp / "manifest.json": DRAMP_MANIFEST_SHA,
        OLD_TABLE: OLD_TABLE_SHA,
    }
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("Independently pinned native MHC10 evidence changed")
    review, dm = read(source / "MHC10_human_candidate_review.json"), read(dramp / "manifest.json")
    points = (
        review["human_original_records"] + review["mouse_original_records_explicitly_out_of_scope"]
    )
    if {r["record_id"] for r in points} != set(POINTS):
        raise ValueError("Native MHC10 source membership differs from all six original IDs")
    for name, expected in review["source_sha256"].items():
        path = ROOT / Path(name)
        if sha(path) != expected:
            raise ValueError("Pinned original CSV/source evidence changed: " + str(path))
        pins[path] = expected
    labelled = {
        r["record_id"]: r
        for r in read(source / "source_labelled_records.json")
        if r.get("endpoint") == "MHC10"
    }
    if set(labelled) != set(POINTS):
        raise ValueError("Original clean MHC10 denominator differs from six exact observations")
    csv_root = (ROOT / Path(points[0]["source_file"])).parent
    observations = {}
    for path in {ROOT / Path(p["source_file"]) for p in points}:
        keys = {p["source_observation_id"] for p in points if ROOT / Path(p["source_file"]) == path}
        observations.update(_csv_matches(path, "observation_id", keys))
    assays = _csv_matches(
        csv_root / "13_experiments.csv", "assay_id", {p["assay_id"] for p in points}
    )
    references = _csv_matches(
        csv_root / "15_references.csv",
        "reference_id",
        {p["reference"]["reference_id"] for p in points},
    )
    compounds_path = csv_root / "01_compounds.csv"
    compound = _csv_matches(compounds_path, "compound_id", {"cmp_3d54915f882550c95c7054d3"})
    compound = next(iter(compound.values()))[0]
    if (
        canonical(compound["smiles"])
        != review["shared_complete_graph"]["canonical_isomeric_smiles"]
    ):
        raise ValueError("The original source melittin graph differs from its registered graph")
    pins[compounds_path] = sha(compounds_path)
    payloads, values = {}, []
    for point in sorted(points, key=lambda p: p["record_id"]):
        record = labelled[point["record_id"]]
        obs, line = observations[point["source_observation_id"]]
        assay = assays[point["assay_id"]][0]
        ref = references[point["reference"]["reference_id"]][0]
        if (
            obs != point["native_observation"]
            or line != point["source_row"]
            or assay != point["native_assay"]
            or any(ref[k] != value for k, value in point["reference"].items())
            or obs["assay_id"] != assay["assay_id"]
            or obs["reference_id"] != ref["reference_id"]
            or obs["compound_id"] != compound["compound_id"]
            or obs["endpoint"] != "MHC10"
            or obs["qualifier"] != "="
            or obs["unit"] != "uM"
            or record["raw_value"] != obs["value"]
            or record["raw_relation"] != obs["qualifier"]
            or record["raw_unit"] != obs["unit"]
        ):
            raise ValueError("Native original observation/assay/reference/compound linkage changed")
        species = (
            "human"
            if assay["system_description"] == "Homo sapiens"
            else ("mouse" if assay["system_description"] == "Mus musculus" else "")
        )
        material = assay["description"].lower()
        if species != POINTS[point["record_id"]][1] or not any(
            s in material for s in ("rbc", "erythrocyte", "red blood cells")
        ):
            raise ValueError("Species/material is not explicitly observation-aligned")
        graph = canonical(record["smiles"])
        row = dict(
            record_id=record["record_id"],
            merged_record_ids=record["record_id"],
            origin="NEW",
            compound_id=record["compound_id"],
            target=format(math.log10(float(record["value"])), ".15g"),
            value=record["value"],
            unit="uM",
            target_transform="log10",
            inverse_transform="pow10",
            sequence="",
            sequence_kind="unparsed_or_missing",
            sequence_eligible="false",
            canonical_identity_group=MEL_GRAPH,
            structure_smiles_sha256=MEL_GRAPH.split(":", 1)[1],
            structure_eligible="true",
            smiles=graph,
            canonical_smiles=graph,
            split_group=MEL_GROUP,
            source_code=record["source_code"],
            source_entry_id=record["source_entry_id"],
            source_group="PMID:" + ref["pmid"] + "|DOI:" + ref["doi"],
            raw_value=record["raw_value"],
            raw_relation="=",
            raw_unit="uM",
            source_original_endpoint="MHC10",
            canonical_quantity_id=QUANTITY,
            native_metric_annotation="MHC10",
            rbc_species=species,
            rbc_species_eligible="true",
            rbc_material="erythrocytes",
            source_pmid=ref["pmid"],
            source_assay_id=assay["assay_id"],
            source_study_id=obs["study_id"],
            source_reference_id=ref["reference_id"],
            source_stratum="Melittin_CHEMBL",
            reported_pH=POINTS[record["record_id"]][3],
            reported_exposure_h="1",
            reported_exposure="1 hr",
            reported_readout=assay["description"],
            explicit_10percent_readout=str(
                point["ten_percent_readout_explicit_in_source_description"]
            ).lower(),
            minimum_discrete_vs_interpolated_verified="false",
            standardized_10percent_quantity_verified="false",
            graph_identity_tier="registered_source_reference_graph_not_sample_salt_or_stereopurity",
            scope_warning=WARNING,
        )
        values.append(row)
        payloads[record["record_id"]] = dict(
            observation=obs, assay=assay, reference=ref, compound=compound, frozen_record=record
        )
    for field in ("source_xlsx", "source_pin"):
        path = Path(dm[field])
        if sha(path) != dm[field + "_sha256"]:
            raise ValueError("Fixed DRAMP source package changed")
        pins[path] = dm[field + "_sha256"]
    mapping_path = dramp / "verified_mapping.tsv"
    if sha(mapping_path) != dm["verified_mapping_sha256"]:
        raise ValueError("DRAMP21 source observation mapping changed")
    pins[mapping_path] = dm["verified_mapping_sha256"]
    mapping, old = rows(mapping_path), rows(OLD_TABLE)
    if len(mapping) != 21 or {r["record_id"] for r in old} != {r["record_id"] for r in mapping}:
        raise ValueError("The exact21 DRAMP observations are missing or repeated")
    import openpyxl
    from rdkit import Chem

    workbook = openpyxl.load_workbook(dm["source_xlsx"], read_only=True, data_only=True)
    source_rows_by_number = {}
    try:
        sheet = workbook.worksheets[0]
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        wanted = {int(r["source_xlsx_row"]) for r in mapping}
        for number, cells in enumerate(
            sheet.iter_rows(min_row=2, max_row=max(wanted), values_only=True), start=2
        ):
            if number in wanted:
                source_rows_by_number[number] = dict(zip(header, cells, strict=True))
    finally:
        workbook.close()
    by_id = {r["record_id"]: r for r in old}
    connection = sqlite3.connect(
        Path(dm["source_catalogue"]).resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        for evidence in sorted(mapping, key=lambda r: r["record_id"]):
            x = source_rows_by_number[int(evidence["source_xlsx_row"])]
            original = connection.execute(
                "SELECT * FROM records WHERE record_id=?", (evidence["record_id"],)
            ).fetchone()
            if original is None:
                raise ValueError("DRAMP original observation missing from the frozen catalogue")
            original = dict(original)
            chemistry = {
                "N-terminal_Modification": "Free",
                "C-terminal_Modification": "Free",
                "Other_Modifications": "None",
                "Linear/Cyclic/Branched": "Linear",
                "Stereochemistry": "L",
            }
            sequence = str(x["Sequence"])
            if (
                x["DRAMP_ID"] != evidence["source_entry_id"]
                or sequence != evidence["source_sequence"]
                or "C" in sequence
                or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
                or any(x[k] != v for k, v in chemistry.items())
                or str(x["Pubmed_ID"]) != "23894079"
                or hashlib.sha256(str(x["Hemolytic_activity"]).strip().encode()).hexdigest()
                != evidence["source_annotation_sha256"]
                or hashlib.sha256(original["raw_json"].encode()).hexdigest()
                != evidence["catalog_raw_json_sha256"]
                or evidence["metric"] != "MHC10"
                or evidence["species"] != "human"
                or evidence["raw_relation"] != "="
                or evidence["unit"] != "uM"
            ):
                raise ValueError("DRAMP source annotation, identity or chemistry changed")
            graph = Chem.MolToSmiles(Chem.MolFromSequence(sequence), isomericSmiles=True)
            prior = by_id[evidence["record_id"]]
            if "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest() != prior[
                "canonical_identity_group"
            ] or not math.isclose(
                float(prior["target"]),
                math.log10(float(evidence["value"])),
                abs_tol=1e-10,
                rel_tol=0,
            ):
                raise ValueError("The old21 graph or target differs from the exact source")
            row = {
                **prior,
                "value": evidence["value"],
                "smiles": graph,
                "canonical_smiles": graph,
                "raw_value": evidence["raw_value"],
                "raw_relation": "=",
                "raw_unit": "uM",
                "source_original_endpoint": "hemolysis_MHC10",
                "canonical_quantity_id": QUANTITY,
                "native_metric_annotation": "MHC10",
                "rbc_species": "human",
                "rbc_species_eligible": "true",
                "rbc_material": "erythrocytes",
                "source_pmid": "23894079",
                "source_assay_id": "",
                "source_study_id": "",
                "source_reference_id": "23894079",
                "source_stratum": "DRAMP21_PMID23894079",
                "reported_pH": "",
                "reported_exposure_h": "",
                "reported_exposure": "not_reported",
                "reported_readout": "not_reported",
                "explicit_10percent_readout": "false",
                "minimum_discrete_vs_interpolated_verified": "false",
                "standardized_10percent_quantity_verified": "false",
                "graph_identity_tier": "source_proven_L_linear_Free_Free_no_other_modifications",
            }
            values.append(row)
            payloads[row["record_id"]] = dict(
                xlsx_source_row=x,
                frozen_record=original,
                verified_mapping=evidence,
                old_task_row=prior,
            )
    finally:
        connection.close()
    values.sort(key=lambda r: r["record_id"])
    validate_cohort(values)
    return values, payloads, pins, dm


def split_prechecks(values, output):
    from peptide_modeling_train import _inner_splits, freeze_outer_split, validate_rows

    y, groups = validate_rows(values, "regression")
    checks = []
    for seed in SEEDS:
        settings = dict(
            seed=seed, inner_folds=5, test_size=0.2, min_groups=8, min_rows=10, min_train_rows=6
        )
        path = Path(output) / "task_tables/predeclared_splits" / f"seed_{seed}.json"
        split = freeze_outer_split(values, "regression", path, settings)
        train = np.flatnonzero(np.isin(groups, split["train_groups"]))
        test = np.flatnonzero(np.isin(groups, split["test_groups"]))
        folds = _inner_splits(y[train], groups[train], "regression", settings)
        if (
            len(folds) != 5
            or set(groups[train]) & set(groups[test])
            or len(set(groups[train])) != 17
            or len(set(groups[test])) != 5
        ):
            raise ValueError("Native27 grouped outer holdout/actual inner5 is unavailable")
        checks.append(
            dict(
                seed=seed,
                actual_folds=5,
                split_path=path.relative_to(output).as_posix(),
                split_sha256=sha(path),
                train_record_ids=[values[i]["record_id"] for i in train],
                holdout_record_ids=[values[i]["record_id"] for i in test],
                train_groups=17,
                holdout_groups=5,
                final_refit_record_ids=[r["record_id"] for r in values],
                melittin_evaluation_role="inner_oof"
                if MEL_GROUP in groups[train]
                else "outer_holdout",
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
        refit_full_required=True,
        final_training_records=27,
        holdout_never_scores_all_common_refit=True,
    )


def slice_features(values, parent, output):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    parent, output = Path(parent), Path(output)
    pointer = parent / "features/manifest.json"
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Independently pinned parent feature pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    root = Path(base["cache_dir"])
    graph_source = {
        r["key"]: r["canonical_isomeric_smiles"] for r in rows(root / "structure_source_smiles.tsv")
    }
    graph_index = {r["key"]: int(r["row_idx"]) for r in rows(root / "structure_bank_index.tsv")}
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in values}
    if len(graph_map) != 22 or any(graph_source.get(k) != v for k, v in graph_map.items()):
        raise ValueError("Every exact full source graph must already match the frozen parent")
    keys = sorted(graph_map)
    methods = copy.deepcopy(
        [
            m
            for m in base["representation_methods"]["methods"]
            if m["source_kind"] == "structure" and m["independent"] is True
        ]
    )
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten independent chemical methods are required; context is not one")
    configuration = dict(
        schema=SCHEMA,
        implementation_sha256=sha(__file__),
        parent_pointer_sha256=PARENT_SHA,
        parent_fingerprint=base["input_fingerprint"],
        source_review_sha256=REVIEW_SHA,
        old21_table_sha256=OLD_TABLE_SHA,
        chemical_graph_keys=keys,
        record_ids=[r["record_id"] for r in values],
        context_by_record={r["record_id"]: r["rbc_species"] for r in values},
        context_method_increment=0,
        chemical_feature_extractions=0,
    )
    fingerprint = object_hash(configuration)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    indices = [graph_index[k] for k in keys]
    for name, meta in groups.items():
        if meta["key_kind"] != "canonical_identity_group":
            raise ValueError("A chemical group uses a nonchemical identity key")
        matrix, valid = np.load(root / meta["matrix"], mmap_mode="r"), np.load(root / meta["valid"])
        selected, masks = matrix[indices].copy(), valid[indices].copy()
        if (
            not masks.all()
            or np.isinf(selected).any()
            or np.isnan(selected).all(axis=1).any()
            or (
                np.isnan(selected).any()
                and (
                    not meta.get("allow_partial_missing", False)
                    or (np.isnan(selected).mean(axis=1) > meta["max_missing_fraction"]).any()
                )
            )
        ):
            raise ValueError("Frozen chemical feature validity/missingness policy changed: " + name)
        if selected.tobytes() != matrix[indices].tobytes():
            raise ValueError("The copied chemical row bytes changed")
        for filename, array in ((meta["matrix"], selected), (meta["valid"], masks)):
            np.save(cache / filename, array, allow_pickle=False)
        meta.update(shape=[22, meta["shape"][1]], index="structure_bank_index.tsv")
    context = np.vstack([species_context(r["rbc_species"]) for r in values])
    np.save(cache / "rbc_species_context.npy", context, allow_pickle=False)
    np.save(cache / "rbc_species_context_valid.npy", np.ones(27, dtype=bool), allow_pickle=False)
    groups[CONTEXT_GROUP] = dict(
        matrix="rbc_species_context.npy",
        valid="rbc_species_context_valid.npy",
        index="rbc_species_index.tsv",
        key_kind="record_id",
        eligibility_field="rbc_species_eligible",
        shape=[27, 2],
        feature_names=["human", "mouse"],
        chemical_method=False,
        source_kind="known_experimental_context",
    )
    table(cache / "structure_bank_index.tsv", [dict(key=k, row_idx=i) for i, k in enumerate(keys)])
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=k, canonical_isomeric_smiles=graph_map[k]) for k in keys],
    )
    table(
        cache / "rbc_species_index.tsv",
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
    registry = dict(
        schema="mhc10_native_ten_chemical_methods_v1",
        methods=methods,
        independent_method_count=10,
        context_method_increment=0,
        chemical_dimensions=sum(g["shape"][1] for n, g in groups.items() if n != CONTEXT_GROUP),
        context_columns=2,
    )
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", configuration)
    write(
        cache / "copied_vector_audit.json",
        dict(
            reused_graphs=22,
            new_graphs=0,
            chemical_feature_extractions=0,
            byte_identical=True,
            parent_artifacts_unchanged=len(base["artifact_hashes"]),
        ),
    )
    routes = {m["route"]: m["group_names"] + [CONTEXT_GROUP] for m in methods}
    routes["structure_methods_all"] = list(GROUPS) + [CONTEXT_GROUP]
    manifest = dict(
        schema=SCHEMA,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=27,
        structure_keys=22,
        groups=groups,
        routes=routes,
        representation_methods=registry,
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for method in methods:
        loaded = load_feature_route(values, manifest, method["route"])
        if (
            not loaded["valid"].all()
            or loaded["X"].shape != (27, method["dimensions"] + 2)
            or loaded["feature_names"][-2:] != CONTEXT_NAMES
            or loaded["feature_names"].count(CONTEXT_NAMES[0]) != 1
            or not np.array_equal(loaded["X"][:, -2:], context)
        ):
            raise ValueError("A chemical route must cover27rows and exactlyone species context")
    _verify_artifacts(base)
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Parent feature pointer changed during byte-copy preparation")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        dict(
            status="verified",
            records=27,
            graphs=22,
            independent_chemical_methods=10,
            chemical_groups=13,
            context_groups=1,
            chemical_dimensions=registry["chemical_dimensions"],
            context_columns=2,
            context_counted_as_method=False,
            chemical_feature_extractions=0,
            parent_pointer_unchanged=True,
            parent_artifacts_unchanged=len(base["artifact_hashes"]),
            copied_vectors_byte_identical=True,
            output_artifacts=len(manifest["artifact_hashes"]),
        ),
    )
    return manifest


def verify_source_proof(root, task, values):
    """Verify portable, immutable source snapshots without chemistry/runtime imports."""
    validate_cohort(values, verify_chemistry=False)
    definition = task.get("scientific_definition", {})
    if (
        task.get("task_id") != TASK
        or task.get("endpoint") != "MHC10"
        or task.get("canonical_quantity_id") != QUANTITY
        or definition.get("canonical_quantity_id") != QUANTITY
        or definition.get("implementation_sha256") != sha(__file__)
        or definition.get("mandatory_prediction_inputs") != ["rbc_species", "quantity_id"]
        or definition.get("context_columns") != 2
        or definition.get("standardized_10percent_quantity_verified") is not False
        or definition.get("mouse_effect_validated") is not False
        or definition.get("cross_molecule_species_correction_validated") is not False
        or definition.get("separate_all_common_refit_required") is not True
    ):
        raise ValueError("Native source quantity, species support or mandatory refit was promoted")
    descriptor = task.get("native_source_observation_coverage", {})
    scope = dict(
        original_endpoint="MHC10",
        target_endpoint="MHC10",
        quantity_id=QUANTITY,
        original_labelled_records=6,
        included_original_records=6,
        original_record_ids=sorted(POINTS),
        included_original_molecular_groups=1,
        pooled_records=27,
        pooled_molecular_groups=22,
        species_counts={"human": 25, "mouse": 2},
        global_HC10_alias=False,
        unified_10percent_quantity_verified=False,
        full_refit_required_for_training_coverage=True,
    )
    if any(descriptor.get(k) != v for k, v in scope.items()):
        raise ValueError("Native annotation coverage scope changed or claims a false HC10 alias")
    path = contained(root, descriptor["proof_path"])
    if sha(path) != descriptor["proof_sha256"]:
        raise ValueError("Native MHC10 source proof hash changed")
    proof = read(path)
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("task_id") != TASK
        or any(proof.get(k) != v for k, v in scope.items())
        or proof.get("implementation_sha256") != sha(__file__)
        or proof.get("record_ids") != [r["record_id"] for r in values]
    ):
        raise ValueError("Native MHC10 proof does not bind its exact implementation/source members")
    source = contained(root, proof["task_table"]["path"])
    if (
        proof["task_table"]["path"] != task["table_path"]
        or sha(source) != proof["task_table"]["sha256"]
        or rows(source) != values
    ):
        raise ValueError("Native source task table differs from its hash/member proof")
    old = contained(root, proof["old21"]["path"])
    if sha(old) != OLD_TABLE_SHA or proof["old21"]["sha256"] != OLD_TABLE_SHA:
        raise ValueError("The original21 source table was not preserved byte-for-byte")
    old_rows = {r["record_id"]: r for r in rows(old)}
    current = {r["record_id"]: r for r in values}
    for record_id, prior in old_rows.items():
        if any(current[record_id].get(k) != v for k, v in prior.items()):
            raise ValueError("An existing21 task source field changed")
    for item in proof["source_files"].values():
        if sha(contained(root, item["path"])) != item["sha256"]:
            raise ValueError("A source-native evidence snapshot changed")
    if (
        proof["source_files"]["review"]["sha256"] != REVIEW_SHA
        or proof["source_files"]["dramp_manifest"]["sha256"] != DRAMP_MANIFEST_SHA
        or proof["source_files"]["dramp_xlsx"]["sha256"]
        != "2196e49100cae4553045873ce640eadb6168555dec5097686842e316ba462f2b"
    ):
        raise ValueError("Independently fixed native source package pins changed")
    if set(proof["observations"]) != set(current):
        raise ValueError("Source observation proof is missing or repeated")
    for record_id, item in proof["observations"].items():
        snapshot = contained(root, item["path"])
        if sha(snapshot) != item["sha256"]:
            raise ValueError("Raw source-object snapshot hash changed")
        obj, row = read(snapshot), current[record_id]
        if record_id in POINTS:
            obs, assay, ref, compound = (
                obj[k] for k in ("observation", "assay", "reference", "compound")
            )
            frozen = obj["frozen_record"]
            species = (
                "human"
                if assay["system_description"] == "Homo sapiens"
                else ("mouse" if assay["system_description"] == "Mus musculus" else "")
            )
            if (
                obs["endpoint"] != "MHC10"
                or obs["qualifier"] != "="
                or obs["unit"] != "uM"
                or obs["source_entry_id"] != row["source_entry_id"]
                or obs["observation_id"] != record_id.split(":")[-1]
                or Decimal(obs["value"]) != Decimal(row["value"])
                or obs["assay_id"] != assay["assay_id"]
                or assay["assay_id"] != row["source_assay_id"]
                or obs["reference_id"] != ref["reference_id"]
                or ref["pmid"] != row["source_pmid"]
                or obs["compound_id"] != compound["compound_id"]
                or compound["smiles"] != row["canonical_smiles"]
                or species != row["rbc_species"]
                or assay["description"] != row["reported_readout"]
                or frozen["record_id"] != record_id
                or frozen["endpoint"] != "MHC10"
                or frozen["raw_value"] != obs["value"]
                or frozen["raw_relation"] != "="
                or frozen["raw_unit"] != "uM"
            ):
                raise ValueError(
                    "An original source scalar, native endpoint or species was changed"
                )
        else:
            evidence, x = obj["verified_mapping"], obj["xlsx_source_row"]
            original = obj["frozen_record"]
            match = re.fullmatch(
                r"\s*\[Ref\.23894079\]\s*MHC10\s*=\s*(\d+(?:\.\d+)?)\s*[µμu]M\s+"
                r"against\s+human\s+red\s+blood\s+cells\s*",
                str(x["Hemolytic_activity"]),
                flags=re.IGNORECASE,
            )
            if (
                not match
                or Decimal(match[1]) != Decimal(row["value"])
                or x["DRAMP_ID"] != row["source_entry_id"]
                or x["Sequence"] != row["sequence"]
                or hashlib.sha256(str(x["Hemolytic_activity"]).strip().encode()).hexdigest()
                != evidence["source_annotation_sha256"]
                or hashlib.sha256(original["raw_json"].encode()).hexdigest()
                != evidence["catalog_raw_json_sha256"]
                or evidence["record_id"] != record_id
                or evidence["raw_relation"] != "="
                or evidence["raw_unit"] != "uM"
                or evidence["species"] != "human"
                or row["source_original_endpoint"] != "hemolysis_MHC10"
                or obj["old_task_row"] != old_rows[record_id]
            ):
                raise ValueError("A DRAMP source-native annotation or original identity changed")
    precheck = contained(root, proof["predeclared_splits"]["path"])
    if sha(precheck) != proof["predeclared_splits"]["sha256"]:
        raise ValueError("Predeclared group/inner5 memberships changed")
    return proof


def prepare(source=SOURCE, dramp=DRAMP, parent=PARENT, output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    values, payloads, pins, dramp_manifest = source_rows(source, dramp)
    source_table = "task_tables/" + TASK + ".tsv"
    table(output / source_table, values)
    # Read the serialized form so blank cells and the proof use exactly the same schema.
    values = rows(output / source_table)
    checks = split_prechecks(values, output)
    write(output / "fivefold_prechecks.json", checks)
    write(output / "task_tables/predeclared_splits/manifest.json", checks)
    archived = {}
    sources = dict(
        review=Path(source) / "MHC10_human_candidate_review.json",
        dramp_manifest=Path(dramp) / "manifest.json",
        dramp_mapping=Path(dramp) / "verified_mapping.tsv",
        dramp_xlsx=Path(dramp_manifest["source_xlsx"]),
        dramp_source_pin=Path(dramp_manifest["source_pin"]),
    )
    for key, path in sources.items():
        destination = output / "task_tables/source_snapshots" / (key + path.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        archived[key] = dict(
            path=destination.relative_to(output).as_posix(), sha256=sha(destination)
        )
    old_path = output / "task_tables/source_snapshots/old21.tsv"
    shutil.copy2(OLD_TABLE, old_path)
    observations = {}
    for record_id, payload in payloads.items():
        path = (
            output
            / "task_tables/source_snapshots/observations"
            / (object_hash(record_id) + ".json")
        )
        write(path, payload)
        observations[record_id] = dict(path=path.relative_to(output).as_posix(), sha256=sha(path))
    scope = dict(
        original_endpoint="MHC10",
        target_endpoint="MHC10",
        quantity_id=QUANTITY,
        original_labelled_records=6,
        included_original_records=6,
        original_record_ids=sorted(POINTS),
        included_original_molecular_groups=1,
        pooled_records=27,
        pooled_molecular_groups=22,
        species_counts={"human": 25, "mouse": 2},
        global_HC10_alias=False,
        unified_10percent_quantity_verified=False,
        full_refit_required_for_training_coverage=True,
    )
    proof = dict(
        schema=PROOF_SCHEMA,
        task_id=TASK,
        **scope,
        implementation_sha256=sha(__file__),
        record_ids=[r["record_id"] for r in values],
        task_table=dict(path=source_table, sha256=sha(output / source_table)),
        old21=dict(path=old_path.relative_to(output).as_posix(), sha256=OLD_TABLE_SHA),
        source_files=archived,
        observations=observations,
        original_source_file_pins={str(path): expected for path, expected in pins.items()},
        predeclared_splits=dict(
            path="task_tables/predeclared_splits/manifest.json",
            sha256=sha(output / "task_tables/predeclared_splits/manifest.json"),
        ),
        warning=WARNING,
    )
    proof_path = output / "task_tables/native_observation_coverage.json"
    write(proof_path, proof)
    definition = dict(
        endpoint="MHC10",
        canonical_quantity_id=QUANTITY,
        quantity_unit="uM",
        native_metric_annotation="MHC10",
        assay_system="source_reported_RBC_annotation",
        species=["human", "mouse"],
        mandatory_prediction_inputs=["rbc_species", "quantity_id"],
        context_group=CONTEXT_GROUP,
        context_columns=2,
        context_is_chemical_method=False,
        standardized_10percent_quantity_verified=False,
        minimum_discrete_vs_interpolated_verified=False,
        exposure_time_conditioned=False,
        pH_conditioned=False,
        mouse_independent_molecular_groups=1,
        mouse_effect_validated=False,
        cross_molecule_species_correction_validated=False,
        homologous_series_isolation_proven=False,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=27,
        separate_all_common_refit_required=True,
        implementation_module="peptide_modeling_mhc10_native_bridge",
        implementation_sha256=sha(__file__),
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="MHC10",
        task_kind="regression",
        status="eligible",
        training_admitted=True,
        table_path=source_table,
        n_records=27,
        n_groups=22,
        unit="uM",
        target_transform="log10",
        inverse_transform="pow10",
        requested_inner_folds=5,
        seeds=list(SEEDS),
        canonical_quantity_id=QUANTITY,
        validation_protocol="group_holdout_inner5_then_separate_all_common_refit",
        scientific_definition=definition,
        task_definition=copy.deepcopy(definition),
        native_source_observation_coverage={
            **scope,
            "proof_path": proof_path.relative_to(output).as_posix(),
            "proof_sha256": sha(proof_path),
        },
        scientific_warning=WARNING,
    )
    verify_source_proof(output, task, values)
    counts = dict(
        input_records=27,
        feature_index_records=27,
        endpoints_accounted=1,
        endpoints_with_eligible_tasks=1,
        eligible_tasks=1,
        eligible_table_records=27,
        deferred_records=0,
        collapsed_source_copies=0,
        label_conflict_records=0,
    )
    write(
        output / "task_manifest.json",
        dict(schema=SCHEMA, tasks=[task], counts=counts, models_fitted=0),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    table(
        output / "task_summary.tsv",
        [dict(task_id=TASK, endpoint="MHC10", status="eligible", records=27, molecular_groups=22)],
    )
    table(output / "feature_rows_index.tsv", values)
    features = slice_features(values, parent, output)
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
        validation_protocol=task["validation_protocol"],
        mandatory_postprocessing="native_species_context_and_all27_refit_diagnostics",
    )
    write(output / "config.json", config)
    write(output / "recommended_training_config.json", config)
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("Original data or source evidence changed during preparation")
    write(
        output / "source_admission.json",
        dict(
            status="prepared_not_trained",
            task_id=TASK,
            original_MHC10_records=6,
            original_MHC10_molecules=1,
            canonical_HC10_alias_changed=False,
            records=27,
            molecule_groups=22,
            species_counts={"human": 25, "mouse": 2},
            source_pins_checked=len(pins),
            source_proof_sha256=sha(proof_path),
            warning=WARNING,
        ),
    )
    (output / "REPORT.md").write_text(
        "# Source-native MHC10 concentration annotation\n\nPrepared27 observations/22 full-graph "
        "plus known-FAMILY groups;25human/2mouse. All6original MHC10 observations are one "
        "Melittin group; DRAMP21 original task fields and exact source table are preserved. "
        "Ten chemical methods/13groups,15,212 chemical dimensions plus2record-specific RBCspecies "
        "columns; no new chemical extraction. Eight models/Optuna128, seeds42/2026/3407, "
        "outer molecular holdout/actual inner5, mandatory separate all27-data refit.\n\n"
        + WARNING
        + "\n\nNo fitting, remote operation or extra20-hour loop by preparation.\n",
        encoding="utf-8",
    )
    from peptide_modeling_pipeline import report_stage

    smoke = output / "diagnostic_history/report_stage_smoke"
    smoke.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output / "task_manifest.json", smoke / "task_manifest.json")
    if report_stage(smoke) != dict(completed=0, deferred=0, failed=0):
        raise ValueError("Prepared task report-stage smoke unexpectedly contains model runs")
    if read(smoke / "run_summary.json")["tasks"] != counts:
        raise ValueError("Report-stage smoke did not preserve actual prepared task counts")
    write(
        output / "preparation_validation.json",
        dict(
            status="prepared_not_trained",
            report_stage_smoke="passed_without_model_fitting",
            task_counts=counts,
            predeclared_split_manifest_staged_under_task_tables=True,
        ),
    )
    receipt = dict(
        status="prepared_not_trained",
        task_id=TASK,
        records=27,
        molecules=22,
        feature_fingerprint=features["input_fingerprint"],
        implementation_sha256=sha(__file__),
        source_proof_sha256=sha(proof_path),
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
        },
    )
    write(output / "delivery_receipt.json", receipt)
    return receipt


def error_metrics(target, prediction):
    y, p = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isfinite([y, p]).all():
        raise ValueError("Finite nonempty aligned diagnostic values are required")
    residuals = np.abs(y - p)
    scale = float(residuals.max())
    return dict(
        n=len(y),
        mae=0.0 if not scale else float(scale * (residuals / scale).mean()),
        rmse=0.0 if not scale else float(scale * np.sqrt(np.square(residuals / scale).mean())),
    )


def weighted_errors(predictions):
    if not predictions:
        return dict(records=0, molecular_groups=0, record_weighted=None, molecule_equal=None)
    groups = defaultdict(list)
    for row in predictions:
        groups[row["split_group"]].append(row)
    per_group = {
        key: error_metrics([r["target"] for r in group], [r["prediction"] for r in group])
        for key, group in sorted(groups.items())
    }
    group_mae = np.asarray([v["mae"] for v in per_group.values()])
    group_rmse = np.asarray([v["rmse"] for v in per_group.values()])
    mae_scale, rmse_scale = float(group_mae.max()), float(group_rmse.max())
    return dict(
        records=len(predictions),
        molecular_groups=len(groups),
        record_weighted=error_metrics(
            [r["target"] for r in predictions], [r["prediction"] for r in predictions]
        ),
        molecule_equal=dict(
            mae=0.0 if not mae_scale else float(mae_scale * (group_mae / mae_scale).mean()),
            rmse=0.0
            if not rmse_scale
            else float(rmse_scale * np.sqrt(np.square(group_rmse / rmse_scale).mean())),
        ),
        per_molecule=per_group,
    )


def evaluation_diagnostics(source, inner_oof, outer_holdout, check):
    """Keep evaluation roles and unsupported species explicit; never score the refit."""
    validate_cohort(source, verify_chemistry=False)
    by_id = {r["record_id"]: r for r in source}
    all_predictions = list(inner_oof) + list(outer_holdout)
    ids = [r["record_id"] for r in all_predictions]
    if len(ids) != 27 or len(set(ids)) != 27 or set(ids) != set(by_id):
        raise ValueError("Exactly one evaluation prediction per27source record is required")
    roles = {
        "inner_oof": (inner_oof, check["train_record_ids"]),
        "outer_holdout": (outer_holdout, check["holdout_record_ids"]),
    }
    if {by_id[r]["split_group"] for r in check["train_record_ids"]} & {
        by_id[r]["split_group"] for r in check["holdout_record_ids"]
    }:
        raise ValueError("An exact molecule crosses outer training and holdout")
    fold_by_id = {
        record_id: f["fold"] for f in check["folds"] for record_id in f["validation_record_ids"]
    }
    if len(check["folds"]) != 5 or {f["fold"] for f in check["folds"]} != set(range(5)):
        raise ValueError("The actual inner fivefold source partition is incomplete")
    folds = {f["fold"]: f for f in check["folds"]}
    report = dict(
        evaluation_by_role={},
        source_context_by_fold=[],
        independently_evaluated_refit=False,
        mouse_effect_validated=False,
        native_annotation_not_unified_HC10=True,
        warning=WARNING,
    )
    for role, (predictions, expected) in roles.items():
        if {r["record_id"] for r in predictions} != set(expected):
            raise ValueError("Saved prediction evaluation role differs from frozen source split")
        enriched = []
        for predicted in predictions:
            row = by_id[predicted["record_id"]]
            fold = int(predicted["fold"])
            if (
                predicted["split_group"] != row["split_group"]
                or not math.isclose(
                    float(predicted["target"]), float(row["target"]), rel_tol=0, abs_tol=1e-10
                )
                or not all(math.isfinite(float(predicted[k])) for k in ("target", "prediction"))
                or (role == "inner_oof" and fold_by_id[row["record_id"]] != fold)
                or (role == "outer_holdout" and fold != -1)
            ):
                raise ValueError("A saved native MHC10 group/target/fold is invalid")
            fitted_ids = (
                folds[fold]["train_record_ids"]
                if role == "inner_oof"
                else check["train_record_ids"]
            )
            if row["split_group"] in {by_id[r]["split_group"] for r in fitted_ids}:
                raise ValueError("A validation molecule entered its own training fold")
            known_species = sorted({by_id[r]["rbc_species"] for r in fitted_ids})
            supported = row["rbc_species"] in known_species
            enriched.append(
                dict(
                    **predicted,
                    evaluation_role=role,
                    rbc_species=row["rbc_species"],
                    source_stratum=row["source_stratum"],
                    original_endpoint=row["source_original_endpoint"],
                    source_pmid=row["source_pmid"],
                    reported_pH=row["reported_pH"],
                    reported_exposure_h=row["reported_exposure_h"],
                    species_seen_in_training=known_species,
                    species_context_supported=supported,
                    unsupported_mouse_context=(row["rbc_species"] == "mouse" and not supported),
                    performance_target="source_native_annotation_not_standardized10percent",
                )
            )
        strata = {}
        for field, labels in (
            ("rbc_species", ["human", "mouse"]),
            ("source_stratum", ["Melittin_CHEMBL", "DRAMP21_PMID23894079"]),
        ):
            strata[field] = {
                label: weighted_errors([r for r in enriched if r[field] == label])
                for label in labels
            }
        report["evaluation_by_role"][role] = dict(
            **weighted_errors(enriched),
            strata=strata,
            predictions=enriched,
            unsupported_mouse_record_ids=[
                r["record_id"] for r in enriched if r["unsupported_mouse_context"]
            ],
            score_role="model_selection" if role == "inner_oof" else "outer_train_model_evaluation",
            score_applies_to_all27_refit=False,
        )
    for fold in check["folds"]:
        train_species = sorted({by_id[r]["rbc_species"] for r in fold["train_record_ids"]})
        validation_species = sorted(
            {by_id[r]["rbc_species"] for r in fold["validation_record_ids"]}
        )
        report["source_context_by_fold"].append(
            dict(
                fold=fold["fold"],
                train_species=train_species,
                validation_species=validation_species,
                unsupported_species=sorted(set(validation_species) - set(train_species)),
                train_record_ids=fold["train_record_ids"],
                validation_record_ids=fold["validation_record_ids"],
            )
        )
    unsupported = [
        r["record_id"]
        for item in report["evaluation_by_role"].values()
        for r in item["predictions"]
        if r["unsupported_mouse_context"]
    ]
    if set(unsupported) != {r["record_id"] for r in source if r["rbc_species"] == "mouse"}:
        raise ValueError(
            "Both single-family mouse evaluations must expose missing training support"
        )
    report["original_MHC10_evaluation_roles"] = {
        record_id: next(
            role
            for role, (predictions, _) in roles.items()
            if any(r["record_id"] == record_id for r in predictions)
        )
        for record_id in sorted(POINTS)
    }
    return report


def fixed_ridge_controls(source, chemical_X, check):
    """Prespecified diagnostic ablations on identical frozen folds; no model selection."""
    from peptide_modeling_train import _cv_predictions, _estimator, _fit_model

    by_id = {r["record_id"]: i for i, r in enumerate(source)}
    train = np.asarray([by_id[r] for r in check["train_record_ids"]])
    test = np.asarray([by_id[r] for r in check["holdout_record_ids"]])
    train_pos = {r: i for i, r in enumerate(check["train_record_ids"])}
    splits = [
        (
            np.asarray([train_pos[r] for r in f["train_record_ids"]]),
            np.asarray([train_pos[r] for r in f["validation_record_ids"]]),
        )
        for f in check["folds"]
    ]
    context = np.vstack([species_context(r["rbc_species"]) for r in source])
    y = np.asarray([float(r["target"]) for r in source])
    settings = dict(seed=check["seed"], n_jobs=1)
    reports = {}
    for name, X in (
        ("molecule_only", np.asarray(chemical_X, dtype=np.float32)),
        ("context_only", context),
        ("molecule_plus_context", np.column_stack((chemical_X, context))),
    ):
        estimator = _estimator(
            "linear", "regression", {"regularization": 1.0}, settings, X.shape[1]
        )
        oof, folds = _cv_predictions(estimator, X[train], y[train], splits, "regression", 1)
        fitted = _fit_model(estimator, X[train], y[train], 1)
        prediction = fitted.predict(X[test])
        inner = [
            dict(
                record_id=source[i]["record_id"],
                split_group=source[i]["split_group"],
                target=float(y[i]),
                prediction=float(p),
                fold=int(f),
            )
            for i, p, f in zip(train, oof, folds, strict=True)
        ]
        outer = [
            dict(
                record_id=source[i]["record_id"],
                split_group=source[i]["split_group"],
                target=float(y[i]),
                prediction=float(p),
                fold=-1,
            )
            for i, p in zip(test, prediction, strict=True)
        ]
        reports[name] = dict(
            model="fixed_ridge_alpha1_not_Optuna_selected",
            same_frozen_splits=True,
            model_selection_repeated=False,
            **evaluation_diagnostics(source, inner, outer, check),
        )
    return reports


def artifact_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    root = Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Saved native-model artifact escapes its run")
    return path


def verify_refit_contract(root, task, source, result):
    """Bind the all27 refit to the exact rows and schema, apart from holdout evaluation."""
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features

    if (
        result.get("status") != "completed"
        or result.get("refit_status") != "completed"
        or result.get("deployment_ready") is not True
    ):
        raise ValueError(
            "A completed all27-data refit is mandatory, even when melittin is held out"
        )
    paths = {name: artifact_path(root, path) for name, path in result["artifacts"].items()}
    if "refit_model" not in paths or str(paths["refit_model"]) not in result["artifact_hashes"]:
        raise ValueError("The separate all27 refit model is not hash covered")
    for path, expected in result["artifact_hashes"].items():
        if sha(artifact_path(root, path)) != expected["sha256"]:
            raise ValueError("A saved source prediction/model/refit artifact changed")
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    ids = [r["record_id"] for r in source]
    if (
        contract.get("common_record_ids") != ids
        or inputs["task"] != task
        or contract.get("task") != task
        or contract.get("refit_status") != "completed"
        or contract.get("optional_refit_fit_scope") != "all_common_after_evaluation"
        or contract.get("model_fit_scope") != "outer_train_only"
        or contract.get("holdout_not_used_for_selection") is not True
        or inputs["config"].get("refit_full") is not True
        or inputs["implementation_sha256"] != sha(Path(root) / "scripts/peptide_modeling_train.py")
        or contract.get("target_unit") != "uM"
    ):
        raise ValueError(
            "All27 training coverage, model source or independent evaluation scope changed"
        )
    payload = load_feature_route(source, portable_features(root), result["selected_route"])
    frozen = inputs["feature_hashes"][result["selected_route"]]
    context = np.vstack([species_context(r["rbc_species"]) for r in source])
    if (
        not payload["valid"].all()
        or payload["record_ids"] != ids
        or payload["feature_names"] != contract["feature_names"]
        or payload["feature_names"][-2:] != CONTEXT_NAMES
        or payload["feature_names"].count(CONTEXT_NAMES[0]) != 1
        or not np.array_equal(payload["X"][:, -2:], context)
        or frozen["names"] != payload["feature_names"]
        or frozen["matrix_sha256"]
        != hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest()
        or frozen["mask_sha256"] != hashlib.sha256(payload["valid"].tobytes()).hexdigest()
    ):
        raise ValueError("The all27 refit lost exact chemical/species context membership or schema")
    return paths, contract, payload


def write_diagnostics(config_path, journal=None):
    from peptide_modeling_summarize import verified_predictions
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "campaign_status.json")
    if len(journal["rounds"]) != 3 or {r["seed"] for r in journal["rounds"]} != set(SEEDS):
        raise ValueError("Native MHC10 diagnostics need all three declared seeds")
    reports = []
    for entry in journal["rounds"]:
        if (
            entry["status"] != "completed"
            or entry["audit"].get("missing_completed_task_ids")
            or not entry["audit"].get("method_model_coverage_met")
        ):
            raise ValueError("A required seed lacks actual task/method/model completion")
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        if task["task_id"] != TASK or task["endpoint"] != "MHC10":
            raise ValueError("Unexpected native MHC10 target task")
        source = rows(run / task["table_path"])
        proof = verify_source_proof(run, task, source)
        path = run / "results/full" / (TASK + "_common.json")
        result = read(path)
        verification_path = run / "verification.json"
        verification = read(verification_path)
        if (
            sha(verification_path) != entry["audit"]["verification_sha256"]
            or verification.get("status") != "verified"
            or verification.get("source_task_manifest_sha256") != sha(run / "task_manifest.json")
        ):
            raise ValueError("Required seed verification does not bind its exact task manifest")
        verified_predictions(run, run, result, task)
        paths, contract, features = verify_refit_contract(run, task, source, result)
        check = next(
            r
            for r in read(contained(run, proof["predeclared_splits"]["path"]))["seeds"]
            if r["seed"] == entry["seed"]
        )
        inner, holdout = rows(paths["oof"]), rows(paths["holdout"])
        report = evaluation_diagnostics(source, inner, holdout, check)
        for role, saved in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            for key in ("mae", "rmse"):
                if not math.isclose(
                    report["evaluation_by_role"][role]["record_weighted"][key],
                    result[saved][key],
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                ):
                    raise ValueError("Native diagnostics do not reproduce the saved role metrics")
        report["fixed_ridge_controls"] = fixed_ridge_controls(source, features["X"][:, :-2], check)
        report.update(
            seed=entry["seed"],
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=sha(run / task["table_path"]),
            result_sha256=sha(path),
            oof_sha256=sha(paths["oof"]),
            holdout_sha256=sha(paths["holdout"]),
            model_sha256=sha(paths["model"]),
            source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
            separate_all_common_refit=dict(
                status="verified",
                records=27,
                molecular_groups=22,
                training_record_ids=contract["common_record_ids"],
                original_MHC10_ids=sorted(POINTS),
                all_six_original_observations_in_training=True,
                model_path=str(paths["refit_model"]),
                model_sha256=sha(paths["refit_model"]),
                model_contract_sha256=sha(paths["model_contract"]),
                fit_scope="all_common_after_evaluation",
                independently_evaluated_on_these27=False,
            ),
        )
        reports.append(report)
    payload = dict(
        schema=REPORT_SCHEMA,
        status="verified",
        task_id=TASK,
        quantity_id=QUANTITY,
        required_seeds=list(SEEDS),
        reports=reports,
        original_MHC10_ids=sorted(POINTS),
        original_MHC10_records=6,
        original_MHC10_molecular_groups=1,
        canonical_HC10_mapping_changed=False,
        mouse_effect_validated=False,
        model_selection_repeated=False,
        warning=WARNING,
    )
    path = root / "mhc10_native_molecular_diagnostics.json"
    atomic_json(path, payload)
    durable.mkdir(parents=True, exist_ok=True)
    destination = durable / path.name
    shutil.copy2(path, destination)
    if sha(destination) != sha(path):
        raise ValueError("Durable native MHC10 mandatory report SHA differs")
    receipt = dict(
        status="verified",
        path=str(path),
        sha256=sha(path),
        durable_path=str(destination),
        durable_sha256=sha(destination),
        implementation_sha256=sha(__file__),
    )
    journal["molecular_reporting"] = receipt
    journal["status"] = "completed"
    atomic_json(root / "campaign_status.json", journal)
    atomic_json(durable / "campaign_status.json", journal)
    return receipt


def predict_contextual(
    model, chemical_X, chemical_names, species, model_contract, *, quantity_id, source_proof
):
    """Prediction requires species and this exact source-native target, never a HC10 alias."""
    task = model_contract["task"]
    definition = task["scientific_definition"]
    descriptor = task["native_source_observation_coverage"]
    if (
        quantity_id != QUANTITY
        or task.get("canonical_quantity_id") != QUANTITY
        or task.get("endpoint") != "MHC10"
        or task.get("task_id") != TASK
        or definition.get("implementation_sha256") != sha(__file__)
        or definition.get("context_columns") != 2
        or definition.get("mandatory_prediction_inputs") != ["rbc_species", "quantity_id"]
        or descriptor.get("global_HC10_alias") is not False
        or sha(source_proof) != descriptor["proof_sha256"]
    ):
        raise ValueError("Prediction target, source proof or native MHC10 implementation differs")
    X = np.asarray(chemical_X, dtype=np.float32)
    if (
        X.ndim != 2
        or not species
        or len(species) != len(X)
        or len(chemical_names) != X.shape[1]
        or any(str(name).startswith(CONTEXT_GROUP + ":") for name in chemical_names)
        or list(chemical_names) + CONTEXT_NAMES != model_contract["feature_names"]
    ):
        raise ValueError(
            "Explicit species per chemical row and the exact fitted schema are required"
        )
    proof = read(source_proof)
    if (
        proof.get("quantity_id") != QUANTITY
        or proof.get("implementation_sha256") != sha(__file__)
        or proof.get("global_HC10_alias") is not False
    ):
        raise ValueError("Source-native inference proof changed")
    context = np.vstack([species_context(s) for s in species])
    return model.predict(np.column_stack((X, context)))


def predict_from_artifacts(
    result_path,
    chemical_X,
    chemical_names,
    species,
    *,
    quantity_id,
    source_proof,
    run_dir,
    use_refit=True,
):
    """Load only a hash-verified fitted model and its source-pinned feature/target contract."""
    import joblib

    result = read(result_path)
    name = "refit_model" if use_refit else "model"
    if result.get("status") != "completed" or (
        use_refit and result.get("refit_status") != "completed"
    ):
        raise ValueError("A completed distinct model artifact is required for prediction")
    paths = {
        key: artifact_path(run_dir, result["artifacts"][key]) for key in (name, "model_contract")
    }
    for key, path in paths.items():
        original = result["artifacts"][key]
        if sha(path) != result["artifact_hashes"][original]["sha256"]:
            raise ValueError("Prediction model or source contract hash differs")
    contract = read(paths["model_contract"])
    if contract.get("input_hash") != result.get("input_hash"):
        raise ValueError("Prediction model and source contract have different input bindings")
    return predict_contextual(
        joblib.load(paths[name]),
        chemical_X,
        chemical_names,
        species,
        contract,
        quantity_id=quantity_id,
        source_proof=source_proof,
    )


def run_followup(config_path, diagnostics_only=False):
    from peptide_modeling_followup_seeds import run_followup as run_seeds
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    training = read(config["template_config"])
    if training["training"].get("refit_full") is not True or config.get(
        "required_seeds", list(SEEDS)
    ) != list(SEEDS):
        raise ValueError("All27-data refit and all three required seeds are mandatory")
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = read(root / "campaign_status.json") if diagnostics_only else run_seeds(config_path)
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
    parser.add_argument("--dramp", type=Path, default=DRAMP)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.run_followup or args.diagnostics_only:
        if args.config is None:
            parser.error("--config is required for the bounded followup/report")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        print(json.dumps(run_followup(args.config, args.diagnostics_only)))
    else:
        receipt = prepare(args.source, args.dramp, args.parent, args.output)
        print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    main()
