"""Isolated source-reported renal-CL per-reported-kg annotation exploration.

This target is neither validated renal physiology nor an absolute-clearance head.
Preparation reuses cached chemistry. Reporting reads frozen numeric artifacts only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
import struct
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_mhc10_native_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/renal_cl_primary_recovery_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_renal_kg_native_followup_20261115_v1"
SCHEMA = "renal_kg_native_reported_annotation_v2"
QUANTITY = "native_source_reported_renal_clearance_per_reported_kg_annotation"
ENDPOINT = "CL_renal_per_reported_kg_annotation"
UNIT = "L/h/kg_reported_basis_annotation"
MODE = "source_annotation_only"
CONTEXT_GROUP = "renal_kg_annotation_context"
BODY_WEIGHT = "source_reported_body_weight_annotation"
UNSPECIFIED = "reported_kg_basis_unspecified"
CONTEXT_FIELDS = ("reported_weight_basis", "source_supported_organism_annotation")
CONTEXT_NAMES = [
    "source_reported_body_weight_annotation",
    "reported_kg_basis_unspecified_mask",
    "reported_in_vivo_organism_human",
    "reported_in_vivo_organism_rat",
]
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
sha, read, rows, write, table, object_hash, contained = (
    shared.sha,
    shared.read,
    shared.rows,
    shared.write,
    shared.table,
    shared.object_hash,
    shared.contained,
)
SHARED_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
PINS = {
    "frozen23_plus_component.json": (
        "06fead99ae92b0bed6795fcf5a8e6df3400d4a7f55eb5fb699b619cec3718d47"
    ),
    "renal_record_review.json": (
        "e6aacc03cf3b62fc06db3a78cfae499412062855f5d606569c060d2be3abf351"
    ),
    "corrected_basis_candidates.json": (
        "e8b88b2777ba3aac3c5fbec72740461d4841c0aed0a14bea345c4bcc87bf4ee2"
    ),
    "nominal_actual_bodyweight_candidates.json": (
        "2669567fc8904249e7bf32c1bfd677a3315fba68e28c99b985aacf76ac4f850f"
    ),
    "NAB739_graph_and_source_candidate.json": (
        "044e6b35552e99d1e767dd3820554babbe9bd1f4f158742b3ee3d4d2a3db92d4"
    ),
    "raw/daptomycin_PMC1932544.html": (
        "17893379cb3ee6381f0196fa4fa12435bac2266c6f7f3adffeb53429b13c2d39"
    ),
    "raw/daptomycin_PMC2630654.html": (
        "91158758525db14c115aa51a95293f7fce83adf5e108de79a9dfbceb50da764d"
    ),
    "raw/nab741_PMC2916294.html": (
        "dd055f079e7f1a3634ae3ed7e4a49d6bb8335fd907015f4709b91373a49cf6ee"
    ),
    "raw/cend1_population_cpdd1502.xml": (
        "03e5d4a73ca9047f204bfdef7dad79d29c7f7adf66e6488d245ae1a84272847b"
    ),
}
UNKNOWN = {"2672768", "2996378", "2671530", "2990610", "2671534"}
UNSPECIFIED_WEIGHT = UNKNOWN | {"NAB739"}
LEGACY_TASK_V1 = "TASK_RENAL_KG_ANNOTATION_fffd317ba69dcab1df8a8c44"
NAB739_ABSTRACT_SHA = "b05cb02ddc0a2752c3b04e0ef7df35d62109217c2173c7b1bcaceb405fd939fe"
NATIVE = {
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
CORRECTIONS = {"2573917": "0.0044", "2573918": "0.0095"}
DAPT = "FAMILY:270cfb85bc0a02dc7857b1ec"
NAB741 = "FAMILY:061ecd1d36a88c80b01c58eb"
NAB739 = "FAMILY:8791f9d19f06695d08688c0a"
SCAFFOLD = "KNOWN_SHARED_NAB741_NAB739_PEPTIDE_PORTION"
TASK = (
    "TASK_RENAL_KG_ANNOTATION_"
    + object_hash(
        dict(
            quantity=QUANTITY,
            source_pins=PINS,
            unspecified=sorted(UNKNOWN),
            native=NATIVE,
            corrections=CORRECTIONS,
            context_fields=CONTEXT_FIELDS,
            body_weight_semantics=BODY_WEIGHT,
            extra_unspecified_weight=["NAB739"],
        )
    )[:24]
)
WARNING = (
    "Source-reported renal-clearance per-reported-kg annotations only. Reported kg is not "
    "automatically actual total bodyweight. Reported body-weight text is a source annotation, "
    "not proof of a total-bodyweight denominator. Six kg bases are unspecified, including "
    "NAB739; the original five also have independently unverified readout/conditions. "
    "Route/matrix/dose remain unknown where not "
    "source-verified. Mixed source tiers and assays are not harmonized intact-parent renal "
    "physiology, absolute clearance, or causal dose/species effects. Five identity/FAMILY "
    "groups are four known scaffold groups: NAB741/NAB739 share a peptide portion. Strict "
    "scaffold five-fold CV is impossible. Daptomycin contributes12/18 records. The same "
    "internal five-fold OOF selects models; no independent holdout, selection-optimistic "
    "scores, one LOGO partition despite three seeds. Source grades/readout verification "
    "are provenance only. No clinical interpretation or conversion to absolute L/h."
)


def check_dependency():
    if sha(shared.__file__) != SHARED_SHA:
        raise ValueError("Pinned numeric helper changed")


def encode_context(context):
    if not isinstance(context, dict) or set(context) != set(CONTEXT_FIELDS):
        raise ValueError("Only explicit reported weight basis and source in-vivo organism apply")
    basis, organism = (context[k] for k in CONTEXT_FIELDS)
    if basis not in {BODY_WEIGHT, UNSPECIFIED} or organism not in {"human", "rat"}:
        raise ValueError("Unsupported reported kg basis or source-supported organism annotation")
    return np.asarray(
        [basis == BODY_WEIGHT, basis == UNSPECIFIED, organism == "human", organism == "rat"],
        dtype=np.float32,
    )


def _context(row):
    import json

    return json.loads(row["semantic_context_json"])


def _source_pins(source):
    for name, expected in PINS.items():
        if sha(contained(source, name)) != expected:
            raise ValueError("Pinned source changed: " + name)


def weight_basis_evidence(aid, source):
    """Table-level body-weight text supports annotations, never an inferred TBW divisor."""
    import html
    import re

    result = dict(
        actual_total_bodyweight_denominator_claimed=False,
        exact_TBW_denominator_primary_verified=aid in CORRECTIONS,
    )
    if aid in CORRECTIONS:
        result.update(
            source_file="raw/daptomycin_PMC1932544.html",
            source_anchor="#t2",
            explicit_weight_text="liter/h/kg TBW",
            semantic_role="primary_TBW_denominator_provenance_only",
            predictive_category=BODY_WEIGHT,
        )
    elif aid.startswith("51621"):
        displays = {
            "5162164": "15% of IBW (cohort inclusion, not a denominator proof)",
            "5162163": "84.18",
            "5162158": "85.7 +/-9.0",
            "5162162": "92.66",
            "5162161": None,
            "5162157": "77.0 +/-8.16",
            "5162156": "64.9 +/-9.76",
            "5162160": "125.8 +/-17.0",
            "5162159": "64.3 +/-7.0",
        }
        result.update(
            source_file="raw/daptomycin_PMC2630654.html",
            source_anchor="#t2",
            explicit_weight_text="Wt (kg)",
            perkg_quantity_text="CL R (ml/h/kg)",
            row_weight_display=displays[aid],
            semantic_role="table_level_body_weight_annotation_not_TBW_divisor_proof",
            missing_row_weight_not_imputed=True,
            predictive_category=BODY_WEIGHT,
        )
    elif aid == "5118245":
        result.update(
            source_file="raw/nab741_PMC2916294.html",
            source_anchor="NAB741 pharmacokinetic methods; #t5 NAB741 column",
            explicit_weight_text="body weight, 245 to 285 g",
            perkg_quantity_text="Renal clearance (ml/min/kg)",
            semantic_role="same_NAB741_experiment_body_weight_annotation",
            predictive_category=BODY_WEIGHT,
        )
    else:
        result.update(
            source_file=None,
            source_anchor=None,
            explicit_weight_text=None,
            semantic_role="reported_kg_only_weight_basis_unspecified",
            predictive_category=UNSPECIFIED,
            not_borrowed_from_another_experiment=True,
        )
        if aid == "NAB739":
            result["reason"] = "Exact primary abstract gives mg/kg but no body-weight text"
            result["NAB741_245_285g_not_borrowed"] = True
    if result["source_file"]:
        text = html.unescape(
            re.sub(
                "<[^>]+>", " ", (Path(source) / result["source_file"]).read_text(encoding="utf-8")
            )
        )
        text = " ".join(text.split())
        if result["explicit_weight_text"] not in text:
            raise ValueError("Explicit source weight annotation was not found: " + aid)
    return result


def source_rows(source=SOURCE, *, verify_chemistry=True):
    """Override the recovery ledger's stronger wording in a new explicit curation proof."""
    import json

    check_dependency()
    source = Path(source)
    _source_pins(source)
    ledger = read(source / "frozen23_plus_component.json")
    originals = {r["record_id"]: r for r in ledger}
    candidates = read(source / "nominal_actual_bodyweight_candidates.json")
    if len(candidates) != 18:
        raise ValueError("Approved recovery frame must contain exactly18 candidates")
    output, proof = [], []
    for candidate in candidates:
        old_id = candidate.get("original_record_id")
        if old_id:
            original = originals[old_id]
            aid = original["source_entry_id"].split(":")[-1]
            graph = original["graph_check"]["canonical_isomeric_smiles"]
            raw = original["frozen_record"]
            if aid in CORRECTIONS:
                source_value = Decimal(CORRECTIONS[aid])
                if Decimal(original["value"]) != source_value or original["unit"] != "L/h":
                    raise ValueError("Primary TBW correction lineage changed")
                value = source_value  # Never round-trip through recovery binary64 mL/min values.
                record_id = "RENAL_KG_NATIVE_CORRECTION:" + aid
                native_unit = "L/h/kg_TBW_primary"
                conversion = "direct_primary_decimal_L_h_per_kg_TBW"
            else:
                if (
                    aid not in NATIVE
                    or Decimal(original["value"]) != Decimal(NATIVE[aid])
                    or original["unit"] != "mL/min/kg"
                    or original["endpoint"] != "CL_renal_per_kg"
                ):
                    raise ValueError("One of the15 original native points changed")
                source_value = Decimal(original["value"])
                value = source_value * Decimal("0.06")
                record_id, native_unit = old_id, "mL/min/kg"
                conversion = "native_displayed_decimal_times_exact_0.06"
            source_entry = original["source_entry_id"]
            source_endpoint, raw_unit = original["endpoint"], original["unit"]
            raw_value = original["value"]
            group = original["split_group"]
            if original["structure_eligible"] != "true" or raw["relation"] != "=":
                raise ValueError("Exact-label/full-reference-graph source gate changed")
        else:
            if (
                candidate["candidate_id"] != "SOURCE_CLR_PERKG:PMC2916294:Table5:NAB739"
                or Decimal(str(candidate["value_mL_min_per_basis_kg"])) != Decimal("0.53")
                or candidate["family_group"] != NAB739
            ):
                raise ValueError("Independent reported NAB739 point changed")
            aid = "NAB739"
            record_id = "RENAL_KG_NATIVE_SOURCE:PMC2916294:T5:NAB739"
            graph, group = candidate["complete_graph"], NAB739
            value = Decimal("0.53") * Decimal("0.06")
            source_value, native_unit = Decimal("0.53"), "mL/min/kg"
            source_entry, source_endpoint = (
                candidate["candidate_id"],
                "reported_renal_CL_annotation",
            )
            raw_value, raw_unit = "0.53", "mL/min/kg"
            conversion = "native_displayed_decimal_times_exact_0.06"
        basis = UNSPECIFIED if aid in UNSPECIFIED_WEIGHT else BODY_WEIGHT
        organism = "rat" if aid in {"5118245", "NAB739"} else "human"
        context = dict(reported_weight_basis=basis, source_supported_organism_annotation=organism)
        encode_context(context)
        descriptive = copy.deepcopy(candidate.get("context", {}))
        if aid == "NAB739":
            descriptive.update(n=None, formulation=None, dose_unit="mg/kg_reported_basis")
        if aid in UNKNOWN:
            descriptive.update(route=None, matrix_reference=None, dose=None, regimen=None)
            readout = "not_independently_verified"
        elif aid in CORRECTIONS or aid == "5118245":
            readout = "primary_reported_parent_urinary_clearance_context"
        elif aid == "NAB739":
            readout = "secondary_numeric_table_exact_primary_abstract_protocol_only"
        else:
            readout = "secondary_cross_study_reported_renal_clearance_annotation"
        key = "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
        row = dict(
            record_id=record_id,
            merged_record_ids=record_id,
            source_record_id=source_entry,
            source_original_record_id=old_id or "",
            source_original_endpoint=source_endpoint,
            source_original_value=raw_value,
            source_original_unit=raw_unit,
            native_point_value=str(source_value),
            native_point_unit=native_unit,
            raw_relation="=",
            value=str(value),
            unit=UNIT,
            target=str(math.log10(float(value))),
            target_transform="log10",
            inverse_transform="pow10",
            endpoint=ENDPOINT,
            canonical_quantity_id=QUANTITY,
            split_group=group,
            scaffold_group=SCAFFOLD if group in {NAB739, NAB741} else group,
            smiles=graph,
            canonical_smiles=graph,
            canonical_identity_group=key,
            structure_smiles_sha256=key.split(":", 1)[1],
            structure_eligible="true",
            sequence="",
            sequence_kind="not_used_registered_full_graph",
            sequence_eligible="false",
            measurement_group="",
            source_code="source_reported_annotation",
            source_entry_id=source_entry,
            source_group="provenance_only_not_a_feature",
            reported_weight_basis=basis,
            source_supported_organism_annotation=organism,
            semantic_context_json=json.dumps(context, sort_keys=True, separators=(",", ":")),
            semantic_context_key=object_hash(context),
            renal_annotation_context_eligible="true",
            source_grade=candidate.get("source_grade", "primary_TBW_unit_correction"),
            readout_verification_status=readout,
            reported_context_provenance_json=json.dumps(descriptive, sort_keys=True),
            annotation_only="true",
            validated_physiology="false",
            normalization=conversion,
        )
        output.append(row)
        sd_native = {
            "2573917": "0.0010",
            "2573918": "0.0036",
            "5118245": "1.11",
            "NAB739": "0.30",
        }.get(aid)
        proof.append(
            dict(
                record_id=record_id,
                original_record_id=old_id,
                source_candidate_id=candidate["candidate_id"],
                native_point=str(source_value),
                native_unit=native_unit,
                normalized_decimal=str(value),
                normalized_unit=UNIT,
                normalization=conversion,
                context=context,
                context_role="source_annotation_only",
                override_of_old_actual_TBW_wording=aid not in CORRECTIONS,
                original_five_unknown_readout_and_conditions=aid in UNKNOWN,
                additional_NAB739_basis_unspecified=aid == "NAB739",
                basis_evidence=weight_basis_evidence(aid, source),
                original_candidate_context_raw_provenance=copy.deepcopy(
                    candidate.get("context", {})
                ),
                NAB739_unverified_n_and_formulation_cleared=aid == "NAB739",
                primary_intact_parent_verification_claimed=False,
                readout_verification_status=readout,
                source_grade=row["source_grade"],
                SD_is_provenance_only_not_an_extra_observation=True,
                reported_SD_metadata={k: v for k, v in candidate.items() if "sd" in k.lower()},
                source_reported_SD_native=sd_native,
                source_reported_SD_native_unit=native_unit if sd_native else None,
                source_reported_SD_target_annotation=(
                    str(
                        Decimal(sd_native)
                        if aid in CORRECTIONS
                        else Decimal(sd_native) * Decimal("0.06")
                    )
                    if sd_native
                    else None
                ),
                registered_reference_identity_not_experimental_sample_purity=True,
                row_hash=object_hash(row),
            )
        )
    validate_rows(output, verify_chemistry=verify_chemistry)
    return output, proof, ledger


def validate_rows(values, *, verify_chemistry=False):
    if len(values) != 18 or len({r["record_id"] for r in values}) != 18:
        raise ValueError("Exactly18 unique native annotation records are required")
    if (
        Counter(r["split_group"] for r in values)
        != {
            DAPT: 12,
            NAB741: 1,
            NAB739: 1,
            "FAMILY:38950029595db06e677c510b": 2,
            "FAMILY:15265cbabdbbbab744d5b4f7": 2,
        }
        or len({r["scaffold_group"] for r in values}) != 4
    ):
        raise ValueError("Frozen identity groups or shared NAB scaffold changed")
    unknown_ids = {"NEW:07:" + str(40000000000000000 + int(a)) for a in UNKNOWN}
    if {r["record_id"] for r in values if r["reported_weight_basis"] == UNSPECIFIED} != (
        unknown_ids | {"RENAL_KG_NATIVE_SOURCE:PMC2916294:T5:NAB739"}
    ):
        raise ValueError("Unspecified kg bases must remain masked, without a TBW claim")
    graphs = {}
    for row in values:
        key = "CANON_SMILES:" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
        if (
            key != row["canonical_identity_group"]
            or row["smiles"] != row["canonical_smiles"]
            or graphs.setdefault(key, row["split_group"]) != row["split_group"]
            or row["unit"] != UNIT
            or row["canonical_quantity_id"] != QUANTITY
            or row["endpoint"] != ENDPOINT
            or row["raw_relation"] != "="
            or row["annotation_only"] != "true"
            or row["validated_physiology"] != "false"
            or row["renal_annotation_context_eligible"] != "true"
        ):
            raise ValueError("Annotation quantity, graph, group or interpretation changed")
        context = _context(row)
        encode_context(context)
        if context != {k: row[k] for k in CONTEXT_FIELDS} or row[
            "semantic_context_key"
        ] != object_hash(context):
            raise ValueError("Only the frozen semantic fields can enter prediction inputs")
        value = Decimal(row["value"])
        if value <= 0 or not math.isclose(
            float(row["target"]), math.log10(float(value)), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Native decimal target changed")
        if row["record_id"] in unknown_ids:
            import json

            info = json.loads(row["reported_context_provenance_json"])
            if row["readout_verification_status"] != "not_independently_verified" or any(
                info.get(k) is not None for k in ["route", "matrix_reference", "dose", "regimen"]
            ):
                raise ValueError("Unknown source readout/route/matrix/dose was fabricated")
        if verify_chemistry:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors

            mol = Chem.MolFromSmiles(row["smiles"])
            if (
                mol is None
                or Chem.MolToSmiles(mol, isomericSmiles=True) != row["smiles"]
                or any(c == "?" for _, c in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            ):
                raise ValueError("Registered full graph needs assigned stereochemistry")
            expected_formula = {NAB741: "C44H73N13O13", NAB739: "C50H85N13O13"}.get(
                row["split_group"]
            )
            if expected_formula and rdMolDescriptors.CalcMolFormula(mol) != expected_formula:
                raise ValueError("Registered NAB acyl-chain C6H12 difference changed")
    if len(graphs) != 5:
        raise ValueError("Exactly five registered complete graphs are required")


def split_prechecks(values):
    from peptide_modeling_internal_cv import internal_splits
    from peptide_modeling_train import validate_rows as validate_training

    y, groups = validate_training(values, "regression")
    checks, partitions = [], set()
    for seed in SEEDS:
        folds = []
        for fold, (train, test) in enumerate(
            internal_splits(y, groups, "regression", dict(seed=seed, inner_folds=5))
        ):
            if (
                set(groups[train]) & set(groups[test])
                or len(set(groups[test])) != 1
                or len(set(groups[train])) != 4
            ):
                raise ValueError("Frozen molecular LOGO fivefold changed")
            known = {values[i]["semantic_context_key"] for i in train}
            folds.append(
                dict(
                    fold=fold,
                    train_record_ids=[values[i]["record_id"] for i in train],
                    validation_record_ids=[values[i]["record_id"] for i in test],
                    train_groups=sorted(set(groups[train])),
                    validation_groups=sorted(set(groups[test])),
                    unsupported_joint_context_record_ids=[
                        values[i]["record_id"]
                        for i in test
                        if values[i]["semantic_context_key"] not in known
                    ],
                )
            )
        partitions.add(tuple(sorted(tuple(sorted(f["validation_record_ids"])) for f in folds)))
        checks.append(dict(seed=seed, folds=folds, actual_folds=5))
    if len(partitions) != 1:
        raise ValueError("Five identity groups must have one LOGO partition")
    return dict(
        seeds=checks,
        distinct_molecular_partitions=1,
        independent_holdout=False,
        identity_groups=5,
        known_scaffold_groups=4,
        strict_scaffold_fivefold_possible=False,
    )


def slice_features(values, parent, output):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    pointer = Path(parent) / "features/manifest.json"
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Frozen parent pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    root = Path(base["cache_dir"])
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in values}
    source_smiles = {
        r["key"]: r["canonical_isomeric_smiles"] for r in rows(root / "structure_source_smiles.tsv")
    }
    if any(source_smiles.get(k) != v for k, v in graph_map.items()) or len(graph_map) != 5:
        raise ValueError("All five exact SMILES bytes must match the parent graph bank")
    index = {r["key"]: int(r["row_idx"]) for r in rows(root / "structure_bank_index.tsv")}
    keys = sorted(graph_map)
    indices = [index[k] for k in keys]
    cache = Path(output) / "features" / object_hash([QUANTITY, graph_map, values])[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    for meta in groups.values():
        matrix = np.load(root / meta["matrix"], mmap_mode="r")
        selected = matrix[indices].copy()
        valid = np.load(root / meta["valid"])[indices].copy()
        if not valid.all() or selected.tobytes() != matrix[indices].tobytes():
            raise ValueError("Cached chemical vectors must be byte-identical and valid")
        np.save(cache / meta["matrix"], selected, allow_pickle=False)
        np.save(cache / meta["valid"], valid, allow_pickle=False)
        meta.update(shape=[5, selected.shape[1]], index="structure_bank_index.tsv")
    context = np.vstack([encode_context(_context(r)) for r in values])
    np.save(cache / "semantic_context.npy", context, allow_pickle=False)
    np.save(cache / "semantic_context_valid.npy", np.ones(18, dtype=bool), allow_pickle=False)
    groups[CONTEXT_GROUP] = dict(
        matrix="semantic_context.npy",
        valid="semantic_context_valid.npy",
        index="context_index.tsv",
        key_kind="record_id",
        eligibility_field="renal_annotation_context_eligible",
        shape=[18, 4],
        feature_names=CONTEXT_NAMES,
        chemical_method=False,
        source_kind="semantic_context",
    )
    table(cache / "structure_bank_index.tsv", [dict(key=k, row_idx=i) for i, k in enumerate(keys)])
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=k, canonical_isomeric_smiles=graph_map[k]) for k in keys],
    )
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
        raise ValueError("Exactly ten independent chemical methods are required")
    registry = dict(
        methods=methods,
        independent_method_count=10,
        chemical_dimensions=15212,
        chemical_groups=13,
        context_columns=4,
        context_method_increment=0,
    )
    routes = {m["route"]: m["group_names"] + [CONTEXT_GROUP] for m in methods}
    routes["structure_methods_all"] = list(GROUPS) + [CONTEXT_GROUP]
    write(cache / "source_parent_manifest.json", base)
    write(cache / "representation_methods.json", registry)
    manifest = dict(
        schema=SCHEMA,
        input_fingerprint=object_hash([QUANTITY, keys, context.tolist()]),
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        record_index="record_bank_index.tsv",
        requested_records=18,
        structure_keys=5,
        groups=groups,
        routes=routes,
        representation_methods=registry,
        parent_pointer_sha256=PARENT_SHA,
        chemical_feature_extractions=0,
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for route, names in routes.items():
        if names.count(CONTEXT_GROUP) != 1:
            raise ValueError("Context must be appended exactly once on every route")
        loaded = load_feature_route(values, manifest, route)
        if (
            not loaded["valid"].all()
            or not np.array_equal(loaded["X"][:, -4:], context)
            or loaded["feature_names"][-4:] != [CONTEXT_GROUP + ":" + n for n in CONTEXT_NAMES]
        ):
            raise ValueError("Actual route loading failed context/identity alignment")
    if sum(groups[n]["shape"][1] for n in GROUPS) != 15212:
        raise ValueError("Chemical fusion width changed")
    _verify_artifacts(base)
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Preparation changed the immutable parent pointer")
    write(Path(output) / "features/manifest.json", manifest)
    write(Path(output) / "representation_methods.json", registry)
    return manifest


def verify_source_proof(root, task=None, values=None):
    """Portable metadata-only validation; intentionally no RDKit or model fit imports."""
    root = Path(root)
    check_dependency()
    task = task or read(root / "task_manifest.json")["tasks"][0]
    values = values or rows(contained(root, task["table_path"]))
    proof_path = contained(root, task["native_source_observation_coverage"]["proof_path"])
    proof = read(proof_path)
    if sha(proof_path) != task["native_source_observation_coverage"]["proof_sha256"]:
        raise ValueError("Source proof hash changed")
    if (
        task["task_id"] != TASK
        or task["endpoint"] != ENDPOINT
        or task["scientific_definition"]["canonical_quantity_id"] != QUANTITY
        or proof["schema"] != SCHEMA
        or proof["task_id"] != TASK
        or proof["validated_physiology"] is not False
        or proof["query_mode"] != MODE
        or proof["source_grade_is_predictor"] is not False
        or proof["implementation_sha256"] != sha(__file__)
        or proof["implementation_dependency_sha256"] != SHARED_SHA
        or proof["source_pins"] != PINS
    ):
        raise ValueError("Native annotation scope, implementation or namespace changed")
    definition = task["scientific_definition"]
    required_definition = dict(
        quantity_unit=UNIT,
        query_mode=MODE,
        context_fields=list(CONTEXT_FIELDS),
        context_group=CONTEXT_GROUP,
        annotation_only=True,
        validated_physiology=False,
        no_absolute_clearance_query=True,
        source_grade_is_predictor=False,
        readout_verification_is_predictor=False,
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        identity_groups=5,
        known_scaffold_groups=4,
        strict_scaffold_fivefold_possible=False,
        independent_holdout=False,
    )
    if (
        any(definition.get(k) != v for k, v in required_definition.items())
        or task.get("unit") != UNIT
        or task.get("target_transform") != "log10"
        or task.get("fixed_external_covariates") is not None
        or task.get("validation_protocol") != "internal_grouped_cv_only_no_independent_holdout"
    ):
        raise ValueError("The weaker annotation contract cannot become a physiological head")
    validate_rows(values)
    for entry in proof["pinned_artifacts"]:
        if sha(contained(root, entry["path"])) != entry["sha256"]:
            raise ValueError("Frozen curation/context/split/source artifact changed")
    if proof["row_hashes"] != {r["record_id"]: object_hash(r) for r in values}:
        raise ValueError("Row normalization, provenance or source context changed")
    if (
        proof["records"] != 18
        or proof["identity_groups"] != 5
        or proof["known_scaffold_groups"] != 4
        or proof["independent_holdout"] is not False
        or proof["strict_scaffold_fivefold_possible"] is not False
    ):
        raise ValueError("Molecular and scaffold coverage cannot be inflated")
    if definition.get("pretraining_fold_state") == "node23_actual_no_fit_frozen":
        evidence = proof.get("nodefold_evidence", {})
        receipt_path = contained(root, evidence.get("path", "missing_node_receipt"))
        baseline = contained(root, evidence.get("pre_node_proof_path", "missing_baseline"))
        receipt = read(receipt_path)
        required_counters = {
            "estimator_fit_attempts",
            "search_attempts",
            "model_unpickle_attempts",
            "chemistry_import_attempts",
            "production_fits_launched",
        }
        if (
            sha(receipt_path) != evidence.get("sha256")
            or sha(baseline) != evidence.get("pre_node_proof_sha256")
            or receipt.get("probe_scope") != "actual_node23_pretraining_no_fit"
            or receipt.get("hostname") != "node23"
            or receipt.get("producer_sha256") != sha(__file__)
            or set(receipt.get("guard_counters", {})) != required_counters
            or any(receipt["guard_counters"].values())
            or receipt.get("frozen_targets_rewritten") is not False
            or receipt.get("report_only_estimator_fits") != 0
            or receipt["payload_source_review"]["source_proof_sha256"] != sha(baseline)
            or receipt["payload_source_review"]["frozen_row_hashes"] != proof["row_hashes"]
            or receipt["payload_source_review"]["task_table_sha256"]
            != sha(contained(root, task["table_path"]))
            or read(root / "task_tables/predeclared_splits.json") != receipt["actual_checks"]
        ):
            raise ValueError("Actual node-fold no-fit receipt or preserved baseline changed")
    return proof


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    output = Path(output).resolve()
    if (output / "task_manifest.json").exists():
        raise FileExistsError("Preserve the existing prepared native annotation task")
    values, observations, ledger = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    table_path = "task_tables/" + TASK + ".tsv"
    table(output / table_path, values)
    values = rows(output / table_path)
    checks = split_prechecks(values)
    write(output / "task_tables/predeclared_splits.json", checks)
    write(
        output / "task_tables/semantic_context_contract.json",
        dict(
            schema=SCHEMA,
            fields=list(CONTEXT_FIELDS),
            feature_names=CONTEXT_NAMES,
            unknown_kg_is_missingness_mask=True,
            unspecified_weight_basis_records=6,
            original_unverified_readout_condition_records=5,
            reported_body_weight_annotation_is_not_TBW_denominator_proof=True,
            source_grade_and_readout_are_provenance_only=True,
            organism_role="source_reported_in_vivo_organism_not_peptide_origin",
        ),
    )
    write(
        output / "task_tables/native_curation_proof.json",
        dict(
            schema=SCHEMA,
            canonical_quantity_id=QUANTITY,
            observations=observations,
            supersedes_old_recovery_actual_bodyweight_wording_for_five_unknown_rows=True,
            input_source_ledger=ledger,
            source_ledger_24_records_preserved=True,
            excluded_from_main18=[
                dict(
                    record_id=r["record_id"],
                    original_endpoint=r["endpoint"],
                    value=r["value"],
                    unit=r["unit"],
                    reason="absolute_or_IBW_or_FFW_or_population_component",
                )
                for r in ledger
                if r["record_id"] not in {v["source_original_record_id"] for v in values}
            ],
            absolute_source_corrections_descriptive_only=True,
            absolute_renal_head_five_molecules_claimed=False,
            NAB_shared_scaffold_evidence=dict(
                identity_groups=[NAB741, NAB739],
                NAB741_formula="C44H73N13O13",
                NAB739_formula="C50H85N13O13",
                N_cap_difference="C6H12",
                peptide_portion_shared=True,
                source="raw/nab741_PMC2916294.html introduction; registered graphs",
                five_independent_scaffolds_claimed=False,
            ),
        ),
    )
    snapshots = output / "task_tables/source_snapshots"
    for name in PINS:
        dest = snapshots / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / name, dest)
        if sha(dest) != PINS[name]:
            raise ValueError("Source snapshot changed while copying its frozen bytes")
    definition = dict(
        endpoint=ENDPOINT,
        canonical_quantity_id=QUANTITY,
        quantity_unit=UNIT,
        query_mode=MODE,
        context_fields=list(CONTEXT_FIELDS),
        context_group=CONTEXT_GROUP,
        annotation_only=True,
        validated_physiology=False,
        no_absolute_clearance_query=True,
        source_grade_is_predictor=False,
        readout_verification_is_predictor=False,
        implementation_module="peptide_modeling_renal_kg_native_bridge",
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        identity_groups=5,
        known_scaffold_groups=4,
        strict_scaffold_fivefold_possible=False,
        mandatory_molecular_reporting=True,
        independent_holdout=False,
        warning=WARNING,
    )
    proof_names = [
        table_path,
        "task_tables/native_curation_proof.json",
        "task_tables/predeclared_splits.json",
        "task_tables/semantic_context_contract.json",
        *["task_tables/source_snapshots/" + name for name in PINS],
    ]
    proof = dict(
        schema=SCHEMA,
        task_id=TASK,
        query_mode=MODE,
        validated_physiology=False,
        source_grade_is_predictor=False,
        records=18,
        identity_groups=5,
        known_scaffold_groups=4,
        strict_scaffold_fivefold_possible=False,
        independent_holdout=False,
        source_pins=PINS,
        implementation_sha256=sha(__file__),
        implementation_dependency_sha256=SHARED_SHA,
        row_hashes={r["record_id"]: object_hash(r) for r in values},
        pinned_artifacts=[dict(path=n, sha256=sha(output / n)) for n in proof_names],
    )
    proof_path = "task_tables/renal_native_source_proof.json"
    write(output / proof_path, proof)
    task = dict(
        task_id=TASK,
        endpoint=ENDPOINT,
        task_kind="regression",
        status="eligible",
        table_path=table_path,
        n_records=18,
        n_groups=5,
        unit=UNIT,
        target_transform="log10",
        inverse_transform="pow10",
        validation_protocol="internal_grouped_cv_only_no_independent_holdout",
        requested_inner_folds=5,
        scientific_definition=definition,
        task_definition=definition,
        native_source_observation_coverage=dict(
            proof_path=proof_path, proof_sha256=sha(output / proof_path)
        ),
    )
    write(output / "task_manifest.json", dict(schema=SCHEMA, tasks=[task], models_fitted=0))
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    table(output / "feature_rows_index.tsv", values)
    feature = slice_features(values, parent, output)
    config = dict(
        dataset="data/dataset_reclean_20261113_v1",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        additional_twenty_hour_loop=False,
        validation_protocol=task["validation_protocol"],
        mandatory_postprocessing="renal_native_metadata_diagnostics_and_precomputed_fixed_controls",
        training=dict(
            routes=list(feature["routes"]),
            models=MODELS,
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
    )
    write(output / "config.json", config)
    write(output / "recommended_training_config.json", config)
    write(
        output / "deployment_requirements.json",
        dict(
            wrapper_entrypoint="scripts/peptide_modeling_renal_kg_native_bridge.py",
            wrapper_arguments=["--run-followup", "--config", "<followup.json>"],
            explicitly_include_before_recursive_import_closure=[
                "peptide_modeling_renal_kg_native_bridge",
                "peptide_modeling_internal_cv",
                "peptide_modeling_mhc10_native_bridge",
            ],
            public_inference="predict_from_artifacts",
            report_only_selected_model_fits=0,
            report_only_any_estimator_fits=0,
            report_only_requires_no_RDKit=True,
            fixed_controls_generated_only_in_explicit_run_phase=True,
            freeze_module_and_source_proof_before_deployment=True,
            prediction_consistency_guard="same_exact_binary32_full_input_graph_context_fold_LSQR32ULP",
        ),
    )
    verify_source_proof(output)
    _source_pins(source)
    from peptide_modeling_pipeline import report_stage

    smoke = output / "preparation_checks/generic_report_smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "task_manifest.json", smoke / "task_manifest.json")
    if report_stage(smoke) != dict(completed=0, deferred=0, failed=0):
        raise ValueError("Preparation report unexpectedly claimed fits")
    write_report(output)
    (output / "README.md").write_text(
        "# Source-reported renal CL per-reported-kg annotation\n\n"
        + WARNING
        + "\n\n18 records, five exact reference graphs, ten chemical methods/13 groups/15212D "
        "plus four semantic columns appended once on every route. Cached chemistry and SMILES "
        "bytes are reused; no extraction. Original15 native points multiply their displayed "
        "Decimal by0.06. Two TBW corrections use primary0.0044/0.0095L/h/kg directly; original "
        "wrongL/h units and IDs remain provenance. NAB739 retains reportedSD0.30 as metadata. "
        "Six kg bases are unspecified; the original five also have unverified readout/conditions. "
        "Body-weight annotation is not a verified TBW divisor; no clinical/intact-parent claim. "
        "Absolute2, IBW/FFW4 and CEND model component are outside this head.\n\n"
        "Eight models, Optuna128, n_jobs1/CPUbudget1, seeds42/2026/3407, internal fivefold only. "
        "Fixed controls: alpha1 RidgeLSQR, median imputation+indicators+keep-empty, scaler "
        "with_mean=False, float64 from the initial numeric boundary using unchanged encoded "
        "float32 values. Report-only reads precomputed controls and never fits.\n\n"
        "Prepare: --prepare. Validate/report metadata: --report-only --output <prepared>. "
        "Root-deployed explicit training wrapper: --run-followup --config <followup.json>. "
        "Inference requires source_annotation_mode, reported_weight_basis and "
        "source_supported_organism_annotation, and always returns annotation_only and "
        "validated_physiology=False. AbsoluteL/h queries are unsupported.\n",
        encoding="utf-8",
    )
    receipt = dict(
        schema=SCHEMA,
        status="prepared_not_trained",
        task_id=TASK,
        records=18,
        identity_groups=5,
        known_scaffold_groups=4,
        models_fitted=0,
        chemical_feature_extractions=0,
        parent_artifacts_verified=len(
            read(Path(parent) / "features/manifest.json")["artifact_hashes"]
        ),
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file() and not any(part.startswith("_") for part in p.relative_to(output).parts)
        },
    )
    write(output / "delivery_receipt.json", receipt)
    return receipt


def _metrics(values, prediction):
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all():
        raise ValueError("Prediction vector must cover every exact source record")
    counts = Counter(r["split_group"] for r in values)
    weights = np.asarray([1 / counts[r["split_group"]] for r in values])
    error = y - p
    return dict(
        records=len(values),
        groups=len(counts),
        record_weighted=dict(
            mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2)))
        ),
        molecular_equal=dict(
            mae=float(np.average(abs(error), weights=weights)),
            rmse=float(np.sqrt(np.average(error**2, weights=weights))),
        ),
    )


def precision_evidence(values, payload):
    X = np.asarray(payload["X"])
    if X.dtype != np.float32 or payload["record_ids"] != [r["record_id"] for r in values]:
        raise ValueError("Selected-model precision evidence requires actual float32 route rows")
    return dict(
        feature_width=X.shape[1],
        matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        record_rows={
            r["record_id"]: dict(
                graph=r["canonical_identity_group"],
                context=r["semantic_context_key"],
                group=r["split_group"],
                row_sha256=hashlib.sha256(np.ascontiguousarray(X[i]).tobytes()).hexdigest(),
            )
            for i, r in enumerate(values)
        },
    )


def _bits32(value):
    if not math.isfinite(value) or float(np.float32(value)) != value or abs(value) < 2.0**-126:
        raise ValueError("Drift guard requires exact normal binary32 values")
    return struct.unpack("!I", struct.pack("!f", value))[0]


def prediction_consistency(predictions, evidence, provenance):
    bound = [evidence["record_rows"][r["record_id"]] for r in predictions]
    if (
        not bound
        or len({(b["graph"], b["context"], b["group"], b["row_sha256"]) for b in bound}) != 1
        or len({(r.get("evaluation_role", "internal_oof"), int(r["fold"])) for r in predictions})
        != 1
    ):
        raise ValueError("Drift comparison requires the same full input/graph/context/group/fold")
    p = [float(r["prediction"]) for r in predictions]
    if not all(math.isfinite(v) for v in p):
        raise ValueError("Nonfinite prediction")
    if len(set(p)) == 1:
        return dict(mode="exact", raw_predictions_unchanged=True)
    bits = [_bits32(v) for v in p]
    required = dict(
        selected_model="linear",
        actual_estimator="Ridge",
        actual_solver="lsqr",
        actual_coef_dtype="float32",
        actual_scaler_with_mean=False,
        actual_imputer_strategy="median",
        actual_add_indicator=True,
        input_dtype="float32",
        feature_width=evidence["feature_width"],
        full_matrix_sha256=evidence["matrix_sha256"],
    )
    from peptide_modeling_train import __file__ as training_file

    if (
        any(provenance.get(k) != v for k, v in required.items())
        or provenance.get("training_source_sha256") != sha(training_file)
        or not provenance.get("actual_model_artifact_sha256")
        or len({math.copysign(1, v) for v in p}) != 1
        or max(bits) - min(bits) > 32
    ):
        raise ValueError("Identical-input drift lacks the predeclared bounded32ULP LSQR proof")
    return dict(
        mode="exact_normal_binary32_actual_LSQR_at_most32ULP",
        observed_binary32_ULPs=max(bits) - min(bits),
        maximum_ULPs=32,
        engineering_guard_not_a_universal_BLAS_bound=True,
        raw_predictions_unchanged=True,
    )


def evaluation_diagnostics(values, oof, check, payload, provenance):
    by_id = {r["record_id"]: r for r in values}
    if [r["record_id"] for r in oof] != list(by_id):
        raise ValueError("Actual OOF rows must exactly preserve all18 source identities/order")
    folds = {rid: f["fold"] for f in check["folds"] for rid in f["validation_record_ids"]}
    comparisons = defaultdict(list)
    for r in oof:
        old = by_id[r["record_id"]]
        if (
            r["split_group"] != old["split_group"]
            or int(r["fold"]) != folds[r["record_id"]]
            or float(r["target"]) != float(old["target"])
        ):
            raise ValueError("OOF target, group or frozen actual fold changed")
        comparisons[
            (old["canonical_identity_group"], old["semantic_context_key"], int(r["fold"]))
        ].append(r)
    evidence = precision_evidence(values, payload)
    consistency = [prediction_consistency(v, evidence, provenance) for v in comparisons.values()]
    pred = [float(r["prediction"]) for r in oof]
    report = _metrics(values, pred)
    report["strata"] = {}
    for field in [*CONTEXT_FIELDS, "source_grade", "readout_verification_status"]:
        report["strata"][field] = {}
        for label in sorted({r[field] for r in values}):
            idx = [i for i, r in enumerate(values) if r[field] == label]
            report["strata"][field][label] = _metrics(
                [values[i] for i in idx], [pred[i] for i in idx]
            )
    report.update(
        annotation_only=True,
        validated_physiology=False,
        independent_holdout=False,
        Daptomycin_record_dominance=dict(records=12, denominator=18),
        fold_context_support=check["folds"],
        prediction_consistency=consistency,
        source_grade_used_only_for_reporting=True,
        raw_predictions_mutated=False,
    )
    return report


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


def _fixed_control_trace(
    estimator, X_train, y_train, X_predict, prediction, train_ids, predict_ids, *, fold, role
):
    steps = estimator.named_steps
    learner = steps["model"]
    coef = np.asarray(learner.coef_)
    predicted = np.asarray(prediction)
    params = dict(
        alpha=float(learner.alpha),
        solver=learner.solver,
        imputer_strategy=steps["impute"].strategy,
        imputer_add_indicator=steps["impute"].add_indicator,
        imputer_keep_empty_features=steps["impute"].keep_empty_features,
        scaler_with_mean=steps["scale"].with_mean,
    )
    if (
        type(learner).__name__ != "Ridge"
        or params
        != dict(
            alpha=1.0,
            solver="lsqr",
            imputer_strategy="median",
            imputer_add_indicator=True,
            imputer_keep_empty_features=True,
            scaler_with_mean=False,
        )
        or any(a.dtype != np.float64 for a in [X_train, y_train, X_predict, predicted, coef])
    ):
        raise ValueError("Actual fixed-control fit/predict/coef must remain float64 RidgeLSQR")

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    return dict(
        fold=fold,
        fit_role=role,
        train_record_ids=train_ids,
        predict_record_ids=predict_ids,
        actual_estimator=type(learner).__name__,
        actual_parameters=params,
        actual_fit_X_dtype=str(X_train.dtype),
        actual_fit_y_dtype=str(y_train.dtype),
        actual_predict_X_dtype=str(X_predict.dtype),
        actual_prediction_dtype=str(predicted.dtype),
        actual_coef_dtype=str(coef.dtype),
        fit_X_sha256=digest(X_train),
        fit_y_sha256=digest(y_train),
        predict_X_sha256=digest(X_predict),
        predictions_sha256=digest(predicted),
        actual_coef_sha256=digest(coef),
        actual_coef_shape=list(coef.shape),
        control_fit_calls=1,
        selected_model_fit_calls=0,
        search_calls=0,
    )


def fixed_ridge_controls(values, chemical_X, context_X, check):
    """Explicit run phase:5OOF+1full-cohort control fits/design; never report-only."""
    from threadpoolctl import threadpool_limits

    chemical = np.asarray(chemical_X, dtype=np.float64)
    context = np.asarray(context_X, dtype=np.float64)
    designs = dict(
        molecule_only=chemical,
        context_only=context,
        molecule_plus_context=np.column_stack((chemical, context)),
    )
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    positions = {r["record_id"]: i for i, r in enumerate(values)}
    all_ids = [r["record_id"] for r in values]
    result = {}
    for name, X in designs.items():
        assert X.dtype == np.float64
        pred = np.full(len(values), np.nan, dtype=np.float64)
        traces = []
        for fold in check["folds"]:
            a = [positions[r] for r in fold["train_record_ids"]]
            b = [positions[r] for r in fold["validation_record_ids"]]
            estimator = fixed_control_estimator()
            with threadpool_limits(limits=1):
                estimator.fit(X[a], y[a])
                part = estimator.predict(X[b])
            pred[b] = part
            traces.append(
                _fixed_control_trace(
                    estimator,
                    X[a],
                    y[a],
                    X[b],
                    part,
                    fold["train_record_ids"],
                    fold["validation_record_ids"],
                    fold=fold["fold"],
                    role="inner_training_fold",
                )
            )
        full_estimator = fixed_control_estimator()
        with threadpool_limits(limits=1):
            full_estimator.fit(X, y)
            full_prediction = full_estimator.predict(X)
        traces.append(
            _fixed_control_trace(
                full_estimator,
                X,
                y,
                X,
                full_prediction,
                all_ids,
                all_ids,
                fold=-1,
                role="all_internal_cohort_control_fit",
            )
        )
        result[name] = dict(
            model="fixed_Ridge_alpha1_LSQR_not_searched",
            dtype_from_initial_boundary="float64",
            source_encoded_values_rewritten=False,
            same_frozen_inner_fivefold=True,
            source_value_origin="upcast_exact_existing_encoded_float32_values",
            prediction_record_ids=all_ids,
            predictions=pred.tolist(),
            metrics=_metrics(values, pred),
            matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
            per_fold_fit_trace=traces,
            control_fit_calls=6,
            selected_model_fit_calls=0,
            search_calls=0,
            all_cohort_fit_predictions=np.asarray(full_prediction).tolist(),
            full_cohort_evaluation_role="training_inputs_only_no_independent_performance_score",
        )
    return result


def write_report(prepared, *, run_dir=None):
    """Pure artifact/numeric reporting: no chemistry, unpickling, search, or fitting."""
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_train import _digest as training_digest

    root = Path(prepared)
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    verify_source_proof(root, task, values)
    report = dict(
        schema=SCHEMA,
        task_id=TASK,
        status="prepared_not_trained",
        records=18,
        identity_groups=5,
        known_scaffold_groups=4,
        selected_models_fitted_by_reporting=0,
        any_estimators_fitted_by_reporting=0,
        annotation_only=True,
        validated_physiology=False,
        warning=WARNING,
    )
    if run_dir is not None:
        run = Path(run_dir)
        result = read(run / "results/internal_cv" / (TASK + ".json"))
        if (
            result.get("status") != "completed"
            or result.get("task_id") != TASK
            or result.get("endpoint") != ENDPOINT
            or result.get("actual_inner_folds") != 5
            or result.get("requested_inner_folds") != 5
            or result.get("protocol") != "internal_grouped_cv_only_no_independent_holdout"
            or result.get("target_unit") != UNIT
            or result.get("target_transform") != "log10"
            or result.get("independent_holdout_available") is not False
        ):
            raise ValueError("Metadata reporting requires completed real internal fivefold OOF")
        for name, receipt in result["artifact_hashes"].items():
            path = Path(name).resolve()
            path.relative_to(run.resolve())
            if sha(path) != receipt["sha256"]:
                raise ValueError("Result artifact hash changed")
        actual_task = read(run / "task_manifest.json")["tasks"][0]
        actual_values = rows(contained(run, actual_task["table_path"]))
        verify_source_proof(run, actual_task, actual_values)
        if actual_values != values or actual_task != task:
            raise ValueError("Runtime source contract differs from prepared task")
        contract = read(result["artifacts"]["model_contract"])
        inputs = read(result["artifacts"]["input_contract"])
        if (
            contract["task"] != task
            or contract["input_hash"] != result["input_hash"]
            or any(
                contract.get(k) != result.get(k)
                for k in ("selected_route", "selected_model", "selected_params")
            )
            or contract["record_ids"] != [r["record_id"] for r in values]
            or inputs["source_rows_hash"] != training_digest(values)
        ):
            raise ValueError("Runtime model/source metadata mismatch")
        feature = read(run / "features/manifest.json")
        payload = load_feature_route(values, feature, result["selected_route"])
        if payload["feature_names"] != contract["feature_names"] or not payload["valid"].all():
            raise ValueError("Actual selected route schema changed")
        selected_input = inputs["features"][result["selected_route"]]
        if (
            selected_input["names"] != payload["feature_names"]
            or selected_input["matrix_sha256"]
            != hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest()
            or selected_input["valid_sha256"]
            != hashlib.sha256(payload["valid"].tobytes()).hexdigest()
        ):
            raise ValueError("Selected numeric route differs from the actual frozen fit input")
        check = next(
            c
            for c in read(root / "task_tables/predeclared_splits.json")["seeds"]
            if c["seed"] == result["seed"]
        )
        controls_path = run / "renal_native_fixed_controls.json"
        controls = read(controls_path)
        if (
            controls["source_proof_sha256"]
            != task["native_source_observation_coverage"]["proof_sha256"]
            or controls["seed"] != result["seed"]
            or controls["selected_result_sha256"]
            != sha(run / "results/internal_cv" / (TASK + ".json"))
            or controls.get("frozen_check_sha256") != object_hash(check)
            or controls["selected_estimator_provenance"].get("actual_model_artifact_sha256")
            != sha(result["artifacts"]["model"])
        ):
            raise ValueError("Mandatory precomputed controls are absent or stale")
        validate_controls(values, payload, check, controls["controls"])
        diagnostic = evaluation_diagnostics(
            values,
            rows(result["artifacts"]["internal_oof"]),
            check,
            payload,
            controls["selected_estimator_provenance"],
        )
        for metric in ("mae", "rmse"):
            if not math.isclose(
                diagnostic["record_weighted"][metric],
                result["internal_cv_selection_metrics"][metric],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError("OOF metrics disagree with saved selection results")
        report.update(
            status="verified_internal_annotation_only",
            seed=result["seed"],
            diagnostics=diagnostic,
            fixed_ridge_controls=controls["controls"],
            controls_sha256=sha(controls_path),
            result_input_hash=result["input_hash"],
        )
    write(root / "reports/renal_native_diagnostics.json", report)
    return report


def validate_controls(values, payload, check, controls):
    """Metadata-only checks on saved fixed controls; never recreate their estimators."""
    if set(controls) != {"molecule_only", "context_only", "molecule_plus_context"}:
        raise ValueError("Exactly the three predeclared fixed controls are mandatory")
    X = np.asarray(payload["X"], dtype=np.float64)
    y = np.asarray([float(r["target"]) for r in values], dtype=np.float64)
    ids = [r["record_id"] for r in values]
    positions = {rid: i for i, rid in enumerate(ids)}
    designs = dict(molecule_only=X[:, :-4], context_only=X[:, -4:], molecule_plus_context=X)
    for name, artifact in controls.items():
        if (
            artifact["model"] != "fixed_Ridge_alpha1_LSQR_not_searched"
            or artifact["dtype_from_initial_boundary"] != "float64"
            or artifact["same_frozen_inner_fivefold"] is not True
            or artifact["source_encoded_values_rewritten"] is not False
            or artifact["prediction_record_ids"] != [r["record_id"] for r in values]
            or artifact["matrix_sha256"]
            != hashlib.sha256(np.ascontiguousarray(designs[name]).tobytes()).hexdigest()
            or artifact["metrics"] != _metrics(values, artifact["predictions"])
        ):
            raise ValueError("Fixed control dtype, immutable values, membership or metrics changed")
        traces = artifact.get("per_fold_fit_trace", [])
        if (
            len(traces) != 6
            or artifact.get("control_fit_calls") != 6
            or artifact.get("selected_model_fit_calls") != 0
            or artifact.get("search_calls") != 0
        ):
            raise ValueError("Actual5OOF+1full control trace is mandatory without search")
        for index, trace in enumerate(traces):
            if index < 5:
                frozen = check["folds"][index]
                train_ids, predict_ids = frozen["train_record_ids"], frozen["validation_record_ids"]
                role, fold = "inner_training_fold", frozen["fold"]
                pred = np.asarray(
                    [artifact["predictions"][positions[r]] for r in predict_ids], dtype=np.float64
                )
            else:
                train_ids, predict_ids, role, fold = ids, ids, "all_internal_cohort_control_fit", -1
                pred = np.asarray(artifact["all_cohort_fit_predictions"], dtype=np.float64)
            a, b = [positions[r] for r in train_ids], [positions[r] for r in predict_ids]
            design = designs[name]
            required = dict(
                fold=fold,
                fit_role=role,
                train_record_ids=train_ids,
                predict_record_ids=predict_ids,
                actual_estimator="Ridge",
                actual_fit_X_dtype="float64",
                actual_fit_y_dtype="float64",
                actual_predict_X_dtype="float64",
                actual_prediction_dtype="float64",
                actual_coef_dtype="float64",
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
            for key, array in [
                ("fit_X_sha256", design[a]),
                ("fit_y_sha256", y[a]),
                ("predict_X_sha256", design[b]),
                ("predictions_sha256", pred),
            ]:
                required[key] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            if (
                any(trace.get(k) != v for k, v in required.items())
                or not trace.get("actual_coef_sha256")
                or not np.isfinite(pred).all()
            ):
                raise ValueError("Actual float64 fit/predict membership/dtype/hash trace changed")
            duplicate = defaultdict(list)
            for row, prediction in zip(design[b], pred, strict=True):
                duplicate[np.ascontiguousarray(row).tobytes()].append(float(prediction))
            if any(not np.allclose(p, p[0], rtol=1e-7, atol=1e-8) for p in duplicate.values()):
                raise ValueError(
                    "Float64 fixed-control identical inputs disagree; inspect, do not widen"
                )
    if set(r for f in check["folds"] for r in f["validation_record_ids"]) != {
        r["record_id"] for r in values
    }:
        raise ValueError("Fixed control folds do not cover all exact native records")


def capture_solver_provenance(model, result, payload):
    from peptide_modeling_train import __file__ as training_file

    out = dict(
        selected_model=result["selected_model"],
        input_dtype=str(payload["X"].dtype),
        feature_width=payload["X"].shape[1],
        full_matrix_sha256=hashlib.sha256(payload["X"].tobytes()).hexdigest(),
        actual_model_artifact_sha256=sha(result["artifacts"]["model"]),
        training_source_sha256=sha(training_file),
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


def run_followup(config_path):
    """Explicit future node execution; preparation and report-only never enter here."""
    from peptide_modeling_internal_cv import run_internal_followup

    cfg = read(config_path)
    template = Path(cfg["template_run"])
    verify_source_proof(template)
    validate_node_runtime(template)
    settings = read(cfg["template_config"])
    if (
        cfg.get("required_seeds") != list(SEEDS)
        or cfg.get("workers", 1) != 1
        or settings["training"]["models"] != MODELS
        or settings["training"]["search_backend"] != "optuna"
        or settings["training"]["search_budget"] != 128
        or settings["training"]["n_jobs"] != 1
        or settings["resources"]["cpu_budget"] != 1
        or cfg.get("additional_twenty_hour_loop", False)
    ):
        raise ValueError("Bounded three-seed/Optuna128/one-CPU native configuration changed")
    journal = run_internal_followup(config_path)
    try:
        result = _complete_followup_reports(cfg, journal)
    except BaseException as exc:
        journal.update(
            status="requirement_incomplete",
            renal_annotation_reporting=dict(status="failed", reason=f"{type(exc).__name__}:{exc}"),
        )
        for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
            write(dest / "internal_campaign_status.json", journal)
        raise
    journal["renal_annotation_reporting"] = dict(status="verified", task_id=TASK)
    for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
        write(dest / "internal_campaign_status.json", journal)
    return result


def _complete_followup_reports(cfg, journal):
    import joblib
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_internal_cv import verify_internal_task

    reports = []
    for entry in journal["rounds"]:
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        values = rows(contained(run, task["table_path"]))
        verify_source_proof(run, task, values)
        features = read(run / "features/manifest.json")
        result_path = run / "results/internal_cv" / (TASK + ".json")
        result = read(result_path)
        verify_internal_task(task, values, features, run, result)
        payload = load_feature_route(values, features, result["selected_route"])
        model = joblib.load(result["artifacts"]["model"])
        check = next(
            c
            for c in read(run / "task_tables/predeclared_splits.json")["seeds"]
            if c["seed"] == entry["seed"]
        )
        write(
            run / "renal_native_fixed_controls.json",
            dict(
                source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
                seed=entry["seed"],
                selected_result_sha256=sha(result_path),
                frozen_check_sha256=object_hash(check),
                selected_estimator_provenance=capture_solver_provenance(model, result, payload),
                controls=fixed_ridge_controls(
                    values, payload["X"][:, :-4], payload["X"][:, -4:], check
                ),
            ),
        )
        reports.append(write_report(run, run_dir=run))
        for relative in [
            "renal_native_fixed_controls.json",
            "reports/renal_native_diagnostics.json",
        ]:
            destination = Path(cfg["durable_root"]) / run.name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / relative, destination)
            if sha(run / relative) != sha(destination):
                raise ValueError("Durable mandatory native diagnostics differ")
    if len(reports) != 3 or {r["seed"] for r in reports} != set(SEEDS):
        raise ValueError("All three mandatory metadata reports must complete")
    report = dict(
        schema=SCHEMA,
        status="verified_internal_annotation_only",
        reports=reports,
        warning=WARNING,
        additional_twenty_hour_loop=False,
    )
    for dest in [Path(cfg["output_root"]), Path(cfg["durable_root"])]:
        write(dest / "renal_native_campaign_diagnostics.json", report)
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
    unsupported="raise",
):
    if quantity_id != QUANTITY or unsupported not in {"raise", "status"}:
        raise ValueError("Only the explicit native reported-kg annotation quantity is supported")
    root = Path(source_proof).resolve().parents[1]
    task = model_contract["task"]
    proof = verify_source_proof(root, task)
    if sha(source_proof) != task["native_source_observation_coverage"]["proof_sha256"]:
        raise ValueError("Inference source proof differs from fitted task")
    source = rows(contained(root, task["table_path"]))
    selected_ids = model_contract["record_ids"]
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) != set(proof["row_hashes"]):
        raise ValueError("Actual fitted training members must cover the declared internal cohort")
    known = {r["semantic_context_key"] for r in source}
    X = np.asarray(chemical_X, dtype=np.float32)
    suffix = [CONTEXT_GROUP + ":" + n for n in CONTEXT_NAMES]
    if (
        X.ndim != 2
        or len(X) != len(contexts)
        or not len(X)
        or X.shape[1] != len(chemical_names)
        or any(CONTEXT_GROUP + ":" in n for n in chemical_names)
        or list(chemical_names) + suffix != model_contract["feature_names"]
        or model_contract.get("target_unit") != UNIT
        or model_contract.get("target_transform") != "log10"
    ):
        raise ValueError("Exact chemical/context schema and reported-kg target unit are mandatory")
    output, accepted, encoded = [], [], []
    for i, query in enumerate(contexts):
        try:
            if (
                not isinstance(query, dict)
                or set(query) != {*CONTEXT_FIELDS, "source_annotation_mode"}
                or query["source_annotation_mode"] != MODE
            ):
                raise ValueError(
                    "Explicit source_annotation_mode/basis/organism only; no L/h query"
                )
            context = {k: query[k] for k in CONTEXT_FIELDS}
            vector = encode_context(context)
            if object_hash(context) not in known:
                raise ValueError("Joint source-annotation context has no fitted support")
        except (ValueError, TypeError, KeyError) as exc:
            if unsupported == "raise":
                raise ValueError("Unsupported annotation query: " + str(exc)) from exc
            output.append(
                dict(
                    status="unsupported_annotation_query",
                    reason=str(exc),
                    annotation_only=True,
                    validated_physiology=False,
                )
            )
            continue
        accepted.append(i)
        encoded.append(vector)
        output.append(
            dict(
                status="annotation_only",
                annotation_only=True,
                validated_physiology=False,
                canonical_quantity_id=QUANTITY,
                output_unit=UNIT,
                absolute_clearance_conversion_available=False,
                warning=WARNING,
            )
        )
    if accepted:
        pred = model.predict(np.column_stack((X[accepted], np.vstack(encoded))))
        for i, value in zip(accepted, pred, strict=True):
            if not math.isfinite(float(value)):
                raise ValueError("Nonfinite source annotation prediction")
            output[i]["prediction_log10_L_h_per_reported_kg_annotation"] = float(value)
            output[i]["prediction_L_h_per_reported_kg_annotation"] = 10.0 ** float(value)
    return output


def predict_from_artifacts(
    run_dir, chemical_X, chemical_names, contexts, *, quantity_id, unsupported="raise"
):
    import joblib
    from peptide_modeling_train import _digest as training_digest

    run = Path(run_dir)
    result = read(run / "results/internal_cv" / (TASK + ".json"))
    if (
        result["status"] != "completed"
        or result.get("task_id") != TASK
        or result.get("endpoint") != ENDPOINT
        or result.get("actual_inner_folds") != 5
        or result.get("target_unit") != UNIT
        or result.get("target_transform") != "log10"
        or result.get("protocol") != "internal_grouped_cv_only_no_independent_holdout"
        or result.get("independent_holdout_available") is not False
    ):
        raise ValueError("Only a completed native annotation artifact can predict")
    for name, receipt in result["artifact_hashes"].items():
        path = Path(name).resolve()
        path.relative_to(run.resolve())
        if sha(path) != receipt["sha256"]:
            raise ValueError("Inference artifact changed")
    contract = read(result["artifacts"]["model_contract"])
    inputs = read(result["artifacts"]["input_contract"])
    task = contract["task"]
    source = rows(contained(run, task["table_path"]))
    if (
        contract["input_hash"] != result["input_hash"]
        or inputs["source_rows_hash"] != training_digest(source)
        or any(
            contract.get(k) != result.get(k)
            for k in ("selected_route", "selected_model", "selected_params")
        )
    ):
        raise ValueError("Inference model identity differs from actual native fit contract")
    return predict_contextual(
        joblib.load(result["artifacts"]["model"]),
        chemical_X,
        chemical_names,
        contexts,
        contract,
        quantity_id=quantity_id,
        source_proof=contained(
            run, contract["task"]["native_source_observation_coverage"]["proof_path"]
        ),
        unsupported=unsupported,
    )


def preparation_review(prepared):
    """Reconstruct the18 source-qualified annotations, without chemistry or fitting."""
    from peptide_modeling_features import load_feature_route

    root = Path(prepared)
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    proof = verify_source_proof(root, task, values)
    reconstructed, curation, ledger = source_rows(
        root / "task_tables/source_snapshots", verify_chemistry=False
    )
    target_differences = []
    for frozen, rebuilt in zip(values, reconstructed, strict=True):
        if {k: v for k, v in frozen.items() if k != "target"} != {
            k: v for k, v in rebuilt.items() if k != "target"
        }:
            raise ValueError("Source reconstruction changed a non-target field")
        if not math.isclose(
            float(frozen["target"]), float(rebuilt["target"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Frozen log target disagrees with the exact normalized source")
        if frozen["target"] != rebuilt["target"]:
            target_differences.append(
                dict(
                    record_id=frozen["record_id"],
                    frozen_target_text=frozen["target"],
                    platform_recomputed_text=rebuilt["target"],
                )
            )
    feature = read(root / "features/manifest.json")
    routes = {}
    for route in feature["routes"]:
        payload = load_feature_route(values, feature, route)
        if not payload["valid"].all() or payload["X"].dtype != np.float32:
            raise ValueError("No-fit review requires the exact complete numeric payload")
        routes[route] = dict(
            shape=list(payload["X"].shape),
            feature_names=payload["feature_names"],
            matrix_sha256=hashlib.sha256(payload["X"].tobytes()).hexdigest(),
            context_appended_exactly_once=feature["routes"][route].count(CONTEXT_GROUP) == 1,
        )
    unknown_basis = [r for r in values if r["reported_weight_basis"] == UNSPECIFIED]
    unknown_readout = [
        r for r in values if r["readout_verification_status"] == "not_independently_verified"
    ]
    return dict(
        schema=SCHEMA,
        task_id=TASK,
        source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
        task_table_sha256=sha(root / task["table_path"]),
        record_ids=[r["record_id"] for r in values],
        frozen_row_hashes=proof["row_hashes"],
        source_reconstructed_records=len(reconstructed),
        input_ledger_records=len(ledger),
        source_grade_is_predictor=False,
        unspecified_weight_basis_record_ids=[r["record_id"] for r in unknown_basis],
        unverified_readout_and_conditions_record_ids=[r["record_id"] for r in unknown_readout],
        unspecified_kg_readout_record_ids=[r["record_id"] for r in unknown_readout],
        unknown_primary_conditions_not_inferred=True,
        identity_groups=5,
        known_scaffold_groups=4,
        strict_scaffold_fivefold_possible=False,
        qualified_curation_records=len(curation),
        annotation_only=True,
        validated_physiology=False,
        routes=routes,
        platform_log_text_differences=target_differences,
        target_comparison="frozen_text_and_hash_preserved; recomputation_numeric_check_only",
        frozen_targets_rewritten=False,
    )


def clone_nodefold_payload(previous, output):
    """Reprepare a new weaker semantic version; old payload and numeric labels stay frozen."""
    import json

    previous, output = Path(previous).resolve(), Path(output).resolve()
    if previous == output or (output / "task_manifest.json").exists():
        raise FileExistsError("Use a new node-fold preparation directory")
    old = read(previous / "delivery_receipt.json")
    if (
        old.get("status") != "prepared_not_trained"
        or old.get("models_fitted") != 0
        or old.get("task_id") != LEGACY_TASK_V1
    ):
        raise ValueError("Only the unfitted original source-native18 payload can be versioned")
    for relative, expected in old["files"].items():
        if sha(contained(previous, relative)) != expected:
            raise ValueError("Original published payload changed: " + relative)
    provenance = output / "source_provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    old_task = read(previous / "task_manifest.json")["tasks"][0]
    old_proof_path = old_task["native_source_observation_coverage"]["proof_path"]
    old_proof = read(previous / old_proof_path)
    old_producer = provenance / "original_v1_producer.py"
    if not old_producer.is_file() or sha(old_producer) != old_proof["implementation_sha256"]:
        raise ValueError("Preserve the original producer's exact bytes before editing it")
    for source, name in [
        (previous / "delivery_receipt.json", "previous_delivery_receipt.json"),
        (
            previous / "task_tables/predeclared_splits.json",
            "original_local_predeclared_splits.json",
        ),
        (previous / old_task["table_path"], "original_v1_task_table.tsv"),
        (previous / old_proof_path, "original_v1_source_proof.json"),
    ]:
        shutil.copyfile(source, provenance / name)
    old_feature = read(previous / "features/manifest.json")
    old_cache = Path(old_feature["cache_dir"])
    old_context = old_feature["groups"][CONTEXT_GROUP]
    shutil.copyfile(old_cache / old_context["matrix"], provenance / "original_v1_context.npy")
    write(provenance / "original_v1_context_group.json", old_context)
    abstract = provenance / "NAB739_exact_primary_abstract.json"
    if not abstract.is_file() or sha(abstract) != NAB739_ABSTRACT_SHA:
        raise ValueError("The new exact primary abstract snapshot must be pinned before cloning")
    primary = read(abstract)["resultList"]["result"][0]
    if primary.get("id") != "19744982" or primary.get("doi", "").lower() != "10.1093/jac/dkp331":
        raise ValueError("NAB739 protocol abstract identity changed")
    receipt = prepare(source=SOURCE, parent=PARENT, output=output)
    task = read(output / "task_manifest.json")["tasks"][0]
    values = rows(contained(output, task["table_path"]))
    old_values = rows(previous / old_task["table_path"])
    approved_semantics = {
        "reported_weight_basis",
        "semantic_context_json",
        "semantic_context_key",
        "reported_context_provenance_json",
    }
    for prior, current in zip(old_values, values, strict=True):
        if {k: v for k, v in prior.items() if k not in approved_semantics} != {
            k: v for k, v in current.items() if k not in approved_semantics
        }:
            raise ValueError("A non-semantic field, raw value or frozen target changed")
    for prior, current in zip(old_values, values, strict=True):
        before_context = json.loads(prior["reported_context_provenance_json"])
        after_context = json.loads(current["reported_context_provenance_json"])
        if current["split_group"] == NAB739:
            before_context.update(n=None, formulation=None, dose_unit="mg/kg_reported_basis")
        if before_context != after_context:
            raise ValueError("Only unsupported NAB739 descriptive fields may be downgraded")
    new_feature = read(output / "features/manifest.json")
    new_cache = Path(new_feature["cache_dir"])
    chemical_equal = {}
    for group in GROUPS:
        before = np.load(old_cache / old_feature["groups"][group]["matrix"])
        after = np.load(new_cache / new_feature["groups"][group]["matrix"])
        chemical_equal[group] = before.tobytes() == after.tobytes()
    old_X = np.load(provenance / "original_v1_context.npy")
    new_X = np.load(new_cache / new_feature["groups"][CONTEXT_GROUP]["matrix"])
    changed = np.flatnonzero(np.any(old_X != new_X, axis=1)).tolist()
    nab_index = next(i for i, row in enumerate(values) if row["split_group"] == NAB739)
    if not all(chemical_equal.values()) or changed != [nab_index]:
        raise ValueError(
            "Only NAB739's two basis bits may change; cached chemistry stays identical"
        )
    if not np.array_equal(old_X[nab_index], [1, 0, 0, 1]) or not np.array_equal(
        new_X[nab_index], [0, 1, 0, 1]
    ):
        raise ValueError("NAB739 unspecified-basis downgrade did not preserve organism context")
    equivalence = dict(
        original_task_id=old_task["task_id"],
        new_task_id=TASK,
        old_payload_modified=False,
        all18_raw_native_normalized_values_and_targets_unchanged=True,
        approved_changed_metadata_fields=sorted(approved_semantics),
        chemical_vectors_byte_identical=chemical_equal,
        context_feature_names_before=old_context["feature_names"],
        context_feature_names_after=CONTEXT_NAMES,
        changed_context_numeric_record_ids=[values[nab_index]["record_id"]],
        known_body_weight_annotations=12,
        unspecified_weight_basis_records=6,
        NAB739_n_and_formulation_unknown_without_primary_support=True,
        original_unverified_readout_condition_records=5,
        body_weight_annotation_not_TBW_denominator=True,
    )
    write(provenance / "v1_numeric_and_semantic_equivalence.json", equivalence)
    task["status"] = "nodefold_pending_not_trainable"
    for field in ("scientific_definition", "task_definition"):
        task[field]["pretraining_fold_state"] = "awaiting_actual_node23_no_fit_receipt"
    proof_path = task["native_source_observation_coverage"]["proof_path"]
    proof = read(output / proof_path)
    proof["primary_abstract_protocol_evidence"] = dict(
        path="source_provenance/NAB739_exact_primary_abstract.json",
        sha256=NAB739_ABSTRACT_SHA,
        fulltext_verified=False,
    )
    for relative in [
        "source_provenance/NAB739_exact_primary_abstract.json",
        "source_provenance/v1_numeric_and_semantic_equivalence.json",
    ]:
        proof["pinned_artifacts"].append(dict(path=relative, sha256=sha(output / relative)))
    write(output / proof_path, proof)
    task["native_source_observation_coverage"]["proof_sha256"] = sha(output / proof_path)
    for name in ("task_manifest.json", "targeted_task_manifest.json"):
        write(
            output / name,
            dict(
                schema=SCHEMA, tasks=[task], models_fitted=0, stage="nodefold_pending_not_trainable"
            ),
        )
    review = preparation_review(output)
    write(output / "source_preparation_review.json", review)
    receipt.update(
        status="nodefold_pending_not_trainable",
        original_payload_modified=False,
        original_delivery_sha256=sha(previous / "delivery_receipt.json"),
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and not any(part.startswith("_") for part in p.relative_to(output).parts)
            and p.name != "delivery_receipt.json"
        },
    )
    write(output / "delivery_receipt.json", receipt)
    return review


def nofit_nodefold_probe(prepared, receipt_output, *, require_node=True):
    """Freeze actual splitter output in a fresh -B process; prohibit fit/search/unpickle."""
    import importlib.abc
    import inspect
    import platform
    import socket
    import sys
    import time

    hostname = socket.gethostname()
    if require_node and (
        hostname != "node23" or Path(sys.executable).resolve() != Path("/usr/bin/python3").resolve()
    ):
        raise ValueError("Actual receipt requires node23 /usr/bin/python3 -B")
    if not sys.flags.dont_write_bytecode:
        raise ValueError("No-fit probing requires -B / disabled bytecode writes")
    if any(n == "rdkit" or n.startswith("rdkit.") for n in sys.modules):
        raise ValueError("Use a fresh process without RDKit already imported")
    counters = dict(
        estimator_fit_attempts=0,
        search_attempts=0,
        model_unpickle_attempts=0,
        chemistry_import_attempts=0,
        production_fits_launched=0,
    )

    class ChemistryBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "rdkit" or fullname.startswith("rdkit."):
                counters["chemistry_import_attempts"] += 1
                raise AssertionError("RDKit is forbidden in the no-fit fold probe")

    def profile(frame, event, arg):
        if event != "call":
            return
        name = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        if name in {
            "fit",
            "fit_transform",
            "partial_fit",
            "_fit_model",
            "_cv_predictions",
            "train_task",
            "train_internal_task",
            "_fit_cv_ranked_candidate",
        } and (module.startswith(("sklearn", "lightgbm", "xgboost", "peptide_modeling"))):
            counters["estimator_fit_attempts"] += 1
            raise AssertionError("Estimator fitting is forbidden in the fold probe")
        if (module.startswith("optuna") and name in {"optimize", "create_study"}) or (
            module == "peptide_modeling_search" and name == "search_candidates"
        ):
            counters["search_attempts"] += 1
            raise AssertionError("Search is forbidden in the fold probe")
        if module.startswith("joblib") and name == "load":
            counters["model_unpickle_attempts"] += 1
            raise AssertionError("Model unpickling is forbidden in the fold probe")

    blocker, old_profile = ChemistryBlocker(), sys.getprofile()
    sys.meta_path.insert(0, blocker)
    sys.setprofile(profile)
    started = time.time()
    try:
        import peptide_modeling_internal_cv as internal
        import peptide_modeling_train as training
        from threadpoolctl import threadpool_info, threadpool_limits

        with threadpool_limits(limits=1):
            root = Path(prepared)
            before = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
            review = preparation_review(root)  # Calls actual verify_source_proof on this host.
            task = read(root / "task_manifest.json")["tasks"][0]
            values = rows(contained(root, task["table_path"]))
            checks = split_prechecks(values)  # Calls actual internal.internal_splits three times.
            by_id = {r["record_id"]: r for r in values}
            for seed in checks["seeds"]:
                for fold in seed["folds"]:
                    train_scaffolds = {by_id[i]["scaffold_group"] for i in fold["train_record_ids"]}
                    validation_scaffolds = {
                        by_id[i]["scaffold_group"] for i in fold["validation_record_ids"]
                    }
                    fold["train_scaffolds"] = sorted(train_scaffolds)
                    fold["validation_scaffolds"] = sorted(validation_scaffolds)
                    fold["shared_scaffolds_across_identity_fold"] = sorted(
                        train_scaffolds & validation_scaffolds
                    )
            # Entire report-only path must work with RDKit and all fit/search/load blocked.
            report = write_report(root)
            allowed = {"reports/renal_native_diagnostics.json"}
            after = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}
            if any(after.get(k) != v for k, v in before.items() if k not in allowed):
                raise ValueError("No-fit probe changed an input artifact")
            if set(after) - set(before) - allowed:
                raise ValueError("No-fit probe created a non-report artifact in the payload")
            if any(counters.values()) or report["any_estimators_fitted_by_reporting"]:
                raise ValueError("A forbidden fit/search/import/unpickle was attempted")
            receipt = dict(
                schema="renal_native_actual_nodefold_no_fit_v1",
                probe_scope="actual_node23_pretraining_no_fit"
                if require_node
                else "local_test_only",
                hostname=hostname,
                python_executable=sys.executable,
                python_version=list(sys.version_info[:3]),
                python_build=platform.python_build(),
                dont_write_bytecode=sys.flags.dont_write_bytecode,
                runtime=internal.runtime_contract(),
                producer_sha256=sha(__file__),
                helper_sha256=SHARED_SHA,
                training_source_sha256=sha(training.__file__),
                splitter_source_sha256=hashlib.sha256(
                    inspect.getsource(internal.internal_splits).encode()
                ).hexdigest(),
                task_id=TASK,
                payload_source_review=review,
                actual_checks=checks,
                guard_counters=counters,
                report_only_status=report["status"],
                report_only_estimator_fits=0,
                chemical_feature_extractions=0,
                input_hashes_before=before,
                input_hashes_after=after,
                threadpools=[
                    {k: v for k, v in x.items() if k in {"internal_api", "num_threads", "version"}}
                    for x in threadpool_info()
                ],
                elapsed_seconds=time.time() - started,
                frozen_targets_rewritten=False,
                annotation_only=True,
                validated_physiology=False,
                unrelated_workloads_not_touched=True,
                no_claim_about_global_cluster_fit_count=True,
            )
    finally:
        sys.setprofile(old_profile)
        sys.meta_path.remove(blocker)
    write(receipt_output, receipt)
    return receipt


def adopt_nodefold_receipt(prepared, receipt_path):
    """Pin genuine node output into this new, still-unfitted version only."""
    root = Path(prepared)
    receipt = read(receipt_path)
    required_counters = {
        "estimator_fit_attempts",
        "search_attempts",
        "model_unpickle_attempts",
        "chemistry_import_attempts",
        "production_fits_launched",
    }
    if (
        receipt.get("probe_scope") != "actual_node23_pretraining_no_fit"
        or receipt.get("hostname") != "node23"
        or receipt.get("dont_write_bytecode") != 1
        or receipt.get("python_executable") != "/usr/bin/python3"
        or receipt.get("producer_sha256") != sha(__file__)
        or set(receipt.get("guard_counters", {})) != required_counters
        or any(receipt["guard_counters"].values())
        or receipt.get("report_only_estimator_fits") != 0
        or receipt.get("validated_physiology") is not False
    ):
        raise ValueError("Only genuine current node23 no-fit evidence may freeze these folds")
    task = read(root / "task_manifest.json")["tasks"][0]
    values = rows(contained(root, task["table_path"]))
    verify_source_proof(root, task, values)
    source_proof = task["native_source_observation_coverage"]["proof_path"]
    if (
        sha(root / source_proof) != receipt["payload_source_review"]["source_proof_sha256"]
        or sha(root / task["table_path"]) != receipt["payload_source_review"]["task_table_sha256"]
    ):
        raise ValueError("Node receipt belongs to another pre-freeze source payload")
    checks = receipt["actual_checks"]
    if (
        [s["seed"] for s in checks["seeds"]] != list(SEEDS)
        or checks["distinct_molecular_partitions"] != 1
        or checks["independent_holdout"] is not False
        or checks["strict_scaffold_fivefold_possible"] is not False
    ):
        raise ValueError(
            "Actual node fold scope differs from the five-identity/four-scaffold contract"
        )
    by_id = {r["record_id"]: r for r in values}
    for seed in checks["seeds"]:
        seen = []
        for fold in seed["folds"]:
            a, b = fold["train_record_ids"], fold["validation_record_ids"]
            if (
                set(a) & set(b)
                or set(a) | set(b) != set(by_id)
                or len({by_id[i]["split_group"] for i in a}) != 4
                or len({by_id[i]["split_group"] for i in b}) != 1
            ):
                raise ValueError("Node identity fold membership is invalid")
            seen.extend(b)
        if len(seed["folds"]) != 5 or sorted(seen) != sorted(by_id):
            raise ValueError("Node actual fivefold does not cover every source row once")
    provenance = root / "source_provenance"
    shutil.copyfile(root / source_proof, provenance / "pre_node_source_proof.json")
    saved_receipt = root / "task_tables/node23_actual_no_fit_fold_receipt.json"
    if Path(receipt_path).resolve() != saved_receipt.resolve():
        shutil.copyfile(receipt_path, saved_receipt)
    write(root / "task_tables/predeclared_splits.json", checks)
    proof = read(root / source_proof)
    proof["nodefold_evidence"] = dict(
        path="task_tables/node23_actual_no_fit_fold_receipt.json",
        sha256=sha(saved_receipt),
        pre_node_proof_path="source_provenance/pre_node_source_proof.json",
        pre_node_proof_sha256=sha(provenance / "pre_node_source_proof.json"),
        frozen_before_any_production_fit=True,
        selection_by_performance=False,
    )
    for entry in proof["pinned_artifacts"]:
        entry["sha256"] = sha(root / entry["path"])
    proof["pinned_artifacts"].append(
        dict(path=proof["nodefold_evidence"]["path"], sha256=sha(saved_receipt))
    )
    write(root / source_proof, proof)
    task["status"] = "eligible"
    for field in ("scientific_definition", "task_definition"):
        task[field]["pretraining_fold_state"] = "node23_actual_no_fit_frozen"
    task["native_source_observation_coverage"]["proof_sha256"] = sha(root / source_proof)
    for name in ("task_manifest.json", "targeted_task_manifest.json"):
        write(
            root / name,
            dict(
                schema=SCHEMA,
                tasks=[task],
                models_fitted=0,
                stage="node23_folds_prepared_not_trained",
            ),
        )
    verify_source_proof(root)
    write_report(root)
    return dict(
        status="node23_folds_prepared_not_trained",
        task_id=TASK,
        actual_seeds=list(SEEDS),
        actual_folds=5,
        identity_groups=5,
        known_scaffold_groups=4,
        no_production_fits=True,
        source_proof_sha256=sha(root / source_proof),
        node_receipt_sha256=sha(saved_receipt),
    )


def validate_node_runtime(prepared):
    """Production entry must match the pre-fit splitter host/code/runtime, without fitting."""
    import socket
    import sys

    from peptide_modeling_internal_cv import runtime_contract

    proof = verify_source_proof(prepared)
    if "nodefold_evidence" not in proof:
        raise ValueError("Freeze actual node23 no-fit folds before any production execution")
    receipt = read(contained(prepared, proof["nodefold_evidence"]["path"]))
    if (
        socket.gethostname() != "node23"
        or Path(sys.executable).resolve() != Path("/usr/bin/python3").resolve()
        or runtime_contract() != receipt["runtime"]
    ):
        raise ValueError("Runtime changed from the actual node23 pre-fit fold receipt")
    return receipt["runtime"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--nofit-nodefold-probe", action="store_true")
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()
    if args.nofit_nodefold_probe:
        if args.prepared_input is None or args.receipt_output is None:
            parser.error("No-fit probe requires --prepared-input and --receipt-output")
        print(nofit_nodefold_probe(args.prepared_input, args.receipt_output)["probe_scope"])
    elif args.run_followup:
        if args.config is None:
            parser.error("--run-followup requires --config")
        print(run_followup(args.config)["status"])
    elif args.report_only:
        print(write_report(args.output, run_dir=args.run_dir)["status"])
    elif args.prepare:
        print(prepare(output=args.output)["status"])
    else:
        parser.error("Choose --prepare, --report-only or explicit --run-followup")


if __name__ == "__main__":
    main()
