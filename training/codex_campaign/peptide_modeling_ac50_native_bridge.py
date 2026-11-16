"""Source-native AC50 annotations given a qualified semantic assay context.

This lane retains 78 clear source records, trains the77-context core and keeps
unresolved NR13 as feature-only evidence. It does not harmonize biological potency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import signal
import struct
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
import peptide_modeling_mhc10_native_bridge as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ac50_native_condition_contract_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
V1_OUTPUT = ROOT / "data/peptide_modeling_ac50_native_followup_20261115_v1"
OUTPUT = ROOT / "data/peptide_modeling_ac50_native_float64_followup_20261115_v2"
V1_IMPLEMENTATION_SHA = "fc23f6b361b5d899379569b5fd231a26c94565d631d4d23f72bea4a5b8973f3c"
V1_DELIVERY_SHA = "7885b9298d38f27c1a00e0c194a3a906582af3a27eabec3527380933f8b874d9"
CONTROL_PRECISION_POLICY = "uniform_float64_fixed_ridge_from_exact_stored_float32_designs_v2"
TARGET_TEXT_POLICY = dict(
    schema="source_log10_15g_text_portability_only_v1",
    maximum_binary64_ulp_distance=128,
    maximum_fixed15_significant_digit_decimal_quanta=1,
    all_non_target_source_fields_exact=True,
    frozen_training_target_bytes_unchanged=True,
    applies_to_model_predictions=False,
)
SOURCE_DELIVERY_SHA = "ed344f07c505999cf694fbf992fd3c7a742cf573ca30ee272c90d0ca0131e71b"
SHARED_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
NR13 = "NEW:08:40000000025164313"
PDK1 = "NEW:08:40000000003455553"
P21 = "NEW:08:40000000001291778"
QUANTITY = "native_source_reported_AC50_concentration_given_semantic_assay_annotation"
SCHEMA = "ac50_native_semantic_source_annotation_v1"
PROOF_SCHEMA = "ac50_native_core77_original78_source_coverage_v1"
REPORT_SCHEMA = "ac50_native_semantic_molecular_diagnostics_v2"
CONTEXT_GROUP = "ac50_semantic_context"
SEEDS = (42, 2026, 3407)
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
error_metrics, weighted_errors = shared.error_metrics, shared.weighted_errors
CONTEXT_FIELDS = (
    "native_assay_type",
    "native_readout_family",
    "measured_direction",
    "target_semantic_category",
    "target_resolution",
    "native_target_entity_ids",
    "target_role",
    "species",
    "species_semantic_role",
    "assay_cell_or_expression_host_species",
    "assay_format",
    "detection_method_categories",
    "cell_line",
    "pH",
    "exposure",
    "half_response_normalization_and_fit_rule",
)
TARGET_FIELDS = (
    "target_semantic_category",
    "native_target_entity_ids",
    "target_role",
    "species",
    "species_semantic_role",
)
TASK = (
    "TASK_AC50_NATIVE_"
    + object_hash(
        dict(
            quantity=QUANTITY,
            source_delivery_sha256=SOURCE_DELIVERY_SHA,
            excluded=NR13,
            context_fields=list(CONTEXT_FIELDS),
            core77=True,
        )
    )[:24]
)
WARNING = (
    "Source-native AC50 concentration annotations given readout, direction, target annotation "
    "role and other recorded semantic context. Binding, inhibition, antagonism, activation, "
    "agonism and p21 induction are distinct annotations, not harmonized biological potencies, "
    "generic50%activation or EC50/IC50/Ki conversions. B/F is not a measured direction. "
    "Human is a target/system-organism annotation; assay host, donor and cell-line species "
    "remain not_reported. Exact PDK1/mitochondrial-pyruvate-DH-kinase1 annotation retains an "
    "unresolved biological identity conflict, never a PDPK1 remapping. p21 is the measured "
    "induction phenotype/marker; annotated HDAC is upstream inhibitor-family context, not "
    "direct p21 or HDAC binding. NR13 is unresolved: all78 retained with features, only77 "
    "core records train.75/77 come from one paper; its holdout leaves only2training groups, "
    "so no independent source generalization is established. Unseen target/readout/joint "
    "contexts must be flagged, and queries abstain outside the qualified source annotation "
    "support. Inner OOF selects the model; outer holdout evaluates only the outer-train "
    "model. Separate all77 refits have no independent score on their now-included records."
)


def check_dependency():
    if sha(shared.__file__) != SHARED_SHA:
        raise ValueError("Pinned portable numeric/source helper changed")


def compare_frozen_target_text(frozen, regenerated, exact_value_uM):
    """Compare only platform log10/.15g rendering; never replace a frozen training target."""
    try:
        concentration = Decimal(exact_value_uM)
        a, b = Decimal(frozen), Decimal(regenerated)
        x, y = float(frozen), float(regenerated)
        value = float(concentration)
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise ValueError("Source log10 target text is not finite numeric data") from exc
    if (
        not concentration.is_finite()
        or concentration <= 0
        or not math.isfinite(value)
        or value <= 0
        or not a.is_finite()
        or not b.is_finite()
        or not math.isfinite(x)
        or not math.isfinite(y)
        or format(x, ".15g") != frozen
        or format(y, ".15g") != regenerated
        or regenerated != format(math.log10(value), ".15g")
    ):
        raise ValueError("Target comparison requires canonical source log10/.15g text")
    if frozen == regenerated:
        return dict(
            mode="exact_text", binary64_ulp_distance=0, frozen_training_target_bytes_unchanged=True
        )
    if not a or not b or math.copysign(1.0, x) != math.copysign(1.0, y):
        raise ValueError("Zero/sign-changing target text cannot use portability tolerance")

    def ordered(value):
        bits = struct.unpack("!Q", struct.pack("!d", value))[0]
        return (
            0x8000000000000000 - (bits & 0x7FFFFFFFFFFFFFFF)
            if bits & 0x8000000000000000
            else 0x8000000000000000 + bits
        )

    ulps = abs(ordered(x) - ordered(y))
    quantum = Decimal(10) ** (min(a.adjusted(), b.adjusted()) - 14)
    if abs(a - b) > quantum or ulps > TARGET_TEXT_POLICY["maximum_binary64_ulp_distance"]:
        raise ValueError("Source target drift exceeds one15g decimal quantum or128binary64 ULP")
    return dict(
        mode="bounded_log10_15g_platform_text",
        binary64_ulp_distance=ulps,
        decimal_difference=str(abs(a - b)),
        maximum_decimal_quantum=str(quantum),
        frozen_training_target_bytes_unchanged=True,
    )


def compare_source_rows(frozen_rows, regenerated_rows):
    """All ordered source fields are exact except the narrowly bounded derived target text."""
    if len(frozen_rows) != len(regenerated_rows):
        raise ValueError("Source observation membership count changed")
    differences = []
    for frozen, regenerated in zip(frozen_rows, regenerated_rows, strict=True):
        if set(frozen) != set(regenerated) or any(
            frozen[field] != regenerated[field] for field in regenerated if field != "target"
        ):
            raise ValueError("An exact non-target source field, concentration or member changed")
        comparison = compare_frozen_target_text(
            frozen["target"], regenerated["target"], regenerated["value"]
        )
        if comparison["mode"] != "exact_text":
            differences.append(dict(record_id=frozen["record_id"], **comparison))
    return differences


def _token(value):
    if isinstance(value, list):
        if not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError("Context lists require explicit nonempty semantic categories")
        return json.dumps(sorted(value), ensure_ascii=False, separators=(",", ":"))
    if not isinstance(value, str) or not value:
        raise ValueError("Every semantic field needs an explicit value or not_reported state")
    return value


def semantic_context(value):
    """Select only the approved source-semantic fields; identifiers/outcomes never enter X."""
    if not isinstance(value, dict) or not set(CONTEXT_FIELDS) <= set(value):
        raise ValueError("A complete qualified source semantic ontology tuple is required")
    return {field: _token(value[field]) for field in CONTEXT_FIELDS}


def context_key(value):
    return object_hash(semantic_context(value))


def target_key(value):
    context = semantic_context(value)
    return object_hash({field: context[field] for field in TARGET_FIELDS})


def make_ontology(candidates):
    """Fix categories from the unlabeled78-source panel; no target/ID-derived statistics."""
    contexts = [semantic_context(row["semantic_context"]) for row in candidates]
    categories = {field: sorted({row[field] for row in contexts}) for field in CONTEXT_FIELDS}
    names = [field + "=" + category for field in CONTEXT_FIELDS for category in categories[field]]
    return dict(
        schema="fixed_source_semantic_AC50_ontology_v1",
        fields=list(CONTEXT_FIELDS),
        categories=categories,
        feature_names=names,
        dimensions=len(names),
        construction="unlabelled_source_panel78_only_no_outcome_statistics",
        identifiers_and_outcomes_are_features=False,
        permitted_entity_ID_role="target_concept_annotation_only_not_activity_or_assay_ID",
    )


def encode_context(value, ontology):
    context = semantic_context(value)
    if ontology.get("fields") != list(CONTEXT_FIELDS):
        raise ValueError("The fitted qualified ontology field schema changed")
    output = []
    for field in CONTEXT_FIELDS:
        categories = ontology["categories"][field]
        if context[field] not in categories:
            raise ValueError("Unsupported unseen semantic context category: " + field)
        output.extend(context[field] == category for category in categories)
    vector = np.asarray(output, dtype=np.float32)
    if len(vector) != ontology["dimensions"]:
        raise ValueError("The fixed semantic ontology dimension changed")
    return vector


def source_rows(source=SOURCE, *, verify_chemistry=True, verify_raw_source_pins=True):
    """Verify all78 exact observations and their same-activity source joins."""
    check_dependency()
    source = Path(source)
    if sha(source / "delivery_receipt.json") != SOURCE_DELIVERY_SHA:
        raise ValueError("Independent AC50 source/semantic review delivery changed")
    delivery = read(source / "delivery_receipt.json")
    for relative, expected in delivery["artifact_sha256"].items():
        if sha(source / relative) != expected:
            raise ValueError("Fixed AC50 source evidence changed: " + relative)
    if verify_raw_source_pins:
        for relative, expected in delivery["source_hashes"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError("Original exact AC50 source input changed")
    candidates = read(source / "candidate_context_records.json")
    joined = read(source / "joined_exact_observation_evidence.json")
    assessment = read(source / "assessment.json")
    if (
        len(candidates) != 78
        or len(joined) != 78
        or assessment["recommended_core_candidate_records"] != 77
        or assessment["unknown_target_row_ids"] != [NR13]
    ):
        raise ValueError("The reviewed78-original/77-core scope changed")
    by_id = {item["record"]["record_id"]: item for item in joined}
    if len(by_id) != 78 or set(by_id) != {item["record_id"] for item in candidates}:
        raise ValueError("AC50 same-observation source membership is missing or duplicated")
    values = []
    for candidate in sorted(candidates, key=lambda r: r["record_id"]):
        evidence = by_id[candidate["record_id"]]
        record, obs, assay, compound, reference, identity = [
            evidence[key]
            for key in (
                "record",
                "native_observation",
                "native_assay",
                "native_compound",
                "native_reference",
                "feature_identity",
            )
        ]
        context = candidate["semantic_context"]
        selected = semantic_context(context)
        if (
            json.loads(record["raw_json"]) != obs
            or record["endpoint"] != "AC50"
            or obs["endpoint"] != "AC50"
            or obs["qualifier"] != "="
            or record["raw_relation"] != "="
            or obs["unit"] != "nM"
            or candidate["relation"] != "="
            or candidate["unit"] != "uM"
            or Decimal(obs["value"]) / 1000 != Decimal(candidate["value_uM"])
            or Decimal(record["value"]) != Decimal(candidate["value_uM"])
            or obs["original_value"] != candidate["original_scalar"]
            or obs["original_unit"] != candidate["original_unit"]
            or obs["source_entry_id"] != candidate["activity_id"]
            or obs["compound_id"] != compound["compound_id"]
            or obs["assay_id"] != assay["assay_id"]
            or obs["study_id"] != assay["study_id"]
            or obs["reference_id"] != reference["reference_id"]
            or reference["pmid"] != candidate["pmid"]
            or record["smiles"] != compound["smiles"]
            or candidate["complete_graph"] != identity["canonical_identity_group"]
            or candidate["family"] != identity["split_group"]
            or context["raw_assay_description_provenance_only"] != assay["description"]
            or context["native_assay_type"] != assay["assay_type"]
        ):
            raise ValueError(
                "AC50 exact observation/assay/compound/target/reference linkage changed"
            )
        original_unit = obs["original_unit"]
        factor = {"nM": Decimal("0.001"), "uM": Decimal(1)}.get(original_unit)
        if factor is None or Decimal(obs["original_value"]) * factor != Decimal(
            candidate["value_uM"]
        ):
            raise ValueError("AC50 original concentration unit conversion differs")
        links = evidence["same_activity_binding_target_rows"]
        targets = evidence["same_activity_target_entities"]
        for link in links:
            if (
                any(
                    link[key] != obs[key]
                    for key in (
                        "compound_id",
                        "substance_id",
                        "assay_id",
                        "reference_id",
                        "source_entry_id",
                    )
                )
                or link["measurement_type"] != "AC50"
                or link["relation"] != "="
                or link["unit"] != "nM"
                or Decimal(link["value_num"]) != Decimal(obs["value"])
            ):
                raise ValueError("AC50 target association came from a different observation")
        linked_ids = sorted({link["target_entity_id"] for link in links}) or ["not_reported"]
        if (
            linked_ids != sorted(context["native_target_entity_ids"])
            or {t["bio_entity_id"] for t in targets} != {link["target_entity_id"] for link in links}
            or context["species"] != "Homo sapiens"
            or context["species_semantic_role"]
            != "source_target_or_system_organism_annotation_not_assay_host_or_donor_species"
            or any(
                context[field] != "not_reported"
                for field in (
                    "assay_cell_or_expression_host_species",
                    "cell_line",
                    "pH",
                    "exposure",
                )
            )
        ):
            raise ValueError("Target annotation or unknown assay-host species was invented")
        graph = (
            shared.canonical(record["smiles"])
            if verify_chemistry
            else next(
                item["canonical_smiles"]
                for item in read(source / "existing_fullgraph_validation.json")[
                    "existing_catalogue_validations"
                ]
                if item["canonical_identity_group"] == candidate["complete_graph"]
            )
        )
        if (
            "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
            != candidate["complete_graph"]
        ):
            raise ValueError("Exact canonical source graph and frozen graph key differ")
        if verify_chemistry:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(record["smiles"])
            if any(tag == "?" for _, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)):
                raise ValueError("Complete source reference graphs need assigned stereochemistry")
        value = {
            **identity,
            "smiles": graph,
            "canonical_smiles": graph,
            "value": candidate["value_uM"],
            "target": format(math.log10(float(candidate["value_uM"])), ".15g"),
            "raw_value": obs["value"],
            "raw_relation": "=",
            "raw_unit": "nM",
            "original_scalar": obs["original_value"],
            "original_unit": obs["original_unit"],
            "source_original_endpoint": "AC50",
            "canonical_quantity_id": QUANTITY,
            "native_metric_annotation": "AC50",
            "source_pmid": reference["pmid"],
            "source_doi": reference["doi"],
            "source_reference_id": reference["reference_id"],
            "source_assay_id": assay["assay_id"],
            "source_study_id": assay["study_id"],
            "semantic_context_json": json.dumps(context, sort_keys=True, ensure_ascii=False),
            "semantic_context_key": object_hash(selected),
            "semantic_target_key": target_key(context),
            "native_readout_family": selected["native_readout_family"],
            "measured_direction": selected["measured_direction"],
            "target_semantic_category": selected["target_semantic_category"],
            "target_role": selected["target_role"],
            "target_resolution": selected["target_resolution"],
            "annotated_organism": selected["species"],
            "annotated_organism_role": selected["species_semantic_role"],
            "assay_host_species": "not_reported",
            "ac50_context_eligible": "true",
            "core_training_eligible": str(candidate["core_candidate"]).lower(),
            "disposition": "core77_training_candidate"
            if candidate["core_candidate"]
            else "NR13_feature_ledger_only",
            "uniform_biological_potency_verified": "false",
            "primary_curve_fit_verified": "false",
            "source_annotation_only": "true",
            "scope_warning": WARNING,
        }
        values.append(value)
    validate_rows(values, require_all78=True, verify_chemistry=False)
    return values, candidates, joined, delivery


def validate_rows(values, *, require_all78=False, verify_chemistry=True):
    check_dependency()
    expected = 78 if require_all78 else 77
    if len(values) != expected or len({r["record_id"] for r in values}) != expected:
        raise ValueError("AC50 requires its exact original78 ledger or77-core scope")
    graph_families = {}
    for row in values:
        compare_frozen_target_text(
            row["target"], format(math.log10(float(row["value"])), ".15g"), row["value"]
        )
        context = json.loads(row["semantic_context_json"])
        selected = semantic_context(context)
        graph = row["canonical_smiles"]
        if (
            row["canonical_identity_group"]
            != "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
            or row["canonical_quantity_id"] != QUANTITY
            or row["source_original_endpoint"] != "AC50"
            or row["raw_relation"] != "="
            or row["raw_unit"] != "nM"
            or row["unit"] != "uM"
            or row["native_metric_annotation"] != "AC50"
            or row["target_transform"] != "log10"
            or Decimal(row["raw_value"]) / 1000 != Decimal(row["value"])
            or row["semantic_context_key"] != object_hash(selected)
            or row["semantic_target_key"] != target_key(context)
            or row["ac50_context_eligible"] != "true"
            or row["source_annotation_only"] != "true"
            or row["uniform_biological_potency_verified"] != "false"
            or row["primary_curve_fit_verified"] != "false"
            or row["assay_host_species"] != "not_reported"
            or row["annotated_organism"] != "Homo sapiens"
            or selected["assay_cell_or_expression_host_species"] != "not_reported"
        ):
            raise ValueError("Exact native target, source context role or graph contract changed")
        if (
            graph_families.setdefault(row["canonical_identity_group"], row["split_group"])
            != row["split_group"]
        ):
            raise ValueError("An exact AC50 graph crosses known molecular families")
        for field in (
            "native_readout_family",
            "measured_direction",
            "target_semantic_category",
            "target_role",
            "target_resolution",
        ):
            if row[field] != selected[field]:
                raise ValueError(
                    "Semantic diagnostic fields differ from the actual encoded context"
                )
        if row["record_id"] == NR13:
            if (
                not require_all78
                or row["core_training_eligible"] != "false"
                or row["target_semantic_category"] != "NR13"
                or row["target_resolution"] != "unresolved_source_literal"
                or selected["native_target_entity_ids"] != '["not_reported"]'
            ):
                raise ValueError(
                    "NR13 is only a literal feature/ledger row, never a resolved training target"
                )
        elif (
            row["core_training_eligible"] != "true"
            or row["target_resolution"] != "source_literal_or_same_activity_annotation"
        ):
            raise ValueError("The semantic77 core membership changed")
        if (
            selected["native_readout_family"] == "binding_affinity"
            and selected["measured_direction"] != "not_reported_functional_direction"
        ):
            raise ValueError("A binding annotation was silently promoted to activation/inhibition")
        if row["record_id"] == PDK1 and (
            selected["target_semantic_category"] != "PDK1_source_literal"
            or selected["native_target_entity_ids"] != '["42000000000012683"]'
            or selected["native_readout_family"] != "functional_activation"
            or selected["native_assay_type"] != "B"
        ):
            raise ValueError(
                "The PDK1 source annotation was remapped to a different biological target"
            )
        if row["record_id"] == P21 and (
            selected["native_readout_family"] != "phenotypic_p21_induction"
            or selected["target_semantic_category"] != "p21_induction_with_HDAC_inhibitor_context"
            or selected["target_role"]
            != "annotated_HDAC_inhibitor_target_family_not_direct_p21_readout"
        ):
            raise ValueError("p21 induction marker was conflated with direct HDAC binding")
        if verify_chemistry:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(row["smiles"])
            if (
                mol is None
                or shared.canonical(row["smiles"]) != graph
                or any(
                    tag == "?" for _, tag in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                )
            ):
                raise ValueError("AC50 complete graph chemistry/stereochemistry changed")
    if len(graph_families) != 17 or len(set(graph_families.values())) != 17:
        raise ValueError("The17 exact complete-graph/FAMILY components changed")
    if (NR13 in {r["record_id"] for r in values}) != require_all78:
        raise ValueError("Unresolved NR13 core/ledger membership changed")


def split_prechecks(values, output):
    from peptide_modeling_train import (
        _inner_splits,
        freeze_outer_split,
    )
    from peptide_modeling_train import (
        validate_rows as validate_training,
    )

    y, groups = validate_training(values, "regression")
    checks = []
    for seed in SEEDS:
        settings = dict(
            seed=seed, inner_folds=5, test_size=0.2, min_groups=8, min_rows=10, min_train_rows=6
        )
        path = Path(output) / "task_tables/predeclared_splits" / f"seed_{seed}.json"
        split = freeze_outer_split(values, "regression", path, settings)
        train, test = (
            np.flatnonzero(np.isin(groups, split["train_groups"])),
            np.flatnonzero(np.isin(groups, split["test_groups"])),
        )
        folds = _inner_splits(y[train], groups[train], "regression", settings)
        if (
            len(folds) != 5
            or set(groups[train]) & set(groups[test])
            or len(set(groups[train])) != 13
            or len(set(groups[test])) != 4
            or any(len(np.unique(y[train[a]])) < 2 for a, _ in folds)
        ):
            raise ValueError("Actual grouped AC50 outer holdout/inner5 is not feasible")
        checks.append(
            dict(
                seed=seed,
                actual_folds=5,
                split_path=path.relative_to(output).as_posix(),
                split_sha256=sha(path),
                train_record_ids=[values[i]["record_id"] for i in train],
                holdout_record_ids=[values[i]["record_id"] for i in test],
                train_groups=13,
                holdout_groups=4,
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
        refit_full_required=True,
        final_training_records=77,
        reference_internal5_context_counts_are_not_this_outer_inner_plan=True,
    )


def slice_features(all_values, ontology, parent, output):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    parent, output = Path(parent), Path(output)
    pointer = parent / "features/manifest.json"
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Independently pinned parent feature pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    root = Path(base["cache_dir"])
    source_graphs = {
        r["key"]: r["canonical_isomeric_smiles"] for r in rows(root / "structure_source_smiles.tsv")
    }
    graph_index = {r["key"]: int(r["row_idx"]) for r in rows(root / "structure_bank_index.tsv")}
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in all_values}
    if len(graph_map) != 17 or any(source_graphs.get(k) != v for k, v in graph_map.items()):
        raise ValueError("All17 full source graphs must exactly match the frozen parent bank")
    keys = sorted(graph_map)
    methods = copy.deepcopy(
        [
            m
            for m in base["representation_methods"]["methods"]
            if m["source_kind"] == "structure" and m["independent"] is True
        ]
    )
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten distinct chemical algorithms are required; context adds zero methods")
    configuration = dict(
        schema=SCHEMA,
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA},
        parent_pointer_sha256=PARENT_SHA,
        parent_fingerprint=base["input_fingerprint"],
        source_delivery_sha256=SOURCE_DELIVERY_SHA,
        selected_complete_graphs=keys,
        all78_record_ids=[r["record_id"] for r in all_values],
        context_by_record={r["record_id"]: r["semantic_context_key"] for r in all_values},
        semantic_ontology_sha256=object_hash(ontology),
        trained_records=77,
        unresolved_feature_only_record=NR13,
        chemical_feature_extractions=0,
        context_method_increment=0,
    )
    fingerprint = object_hash(configuration)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    indices = [graph_index[k] for k in keys]
    for name, meta in groups.items():
        if meta["key_kind"] != "canonical_identity_group":
            raise ValueError("A chemical group uses nonchemical record identity")
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
            raise ValueError("Exact chemical-row validity/missingness changed: " + name)
        if selected.tobytes() != matrix[indices].tobytes():
            raise ValueError("A copied exact chemical vector changed bytes")
        for filename, array in ((meta["matrix"], selected), (meta["valid"], masks)):
            np.save(cache / filename, array, allow_pickle=False)
        meta.update(shape=[17, meta["shape"][1]], index="structure_bank_index.tsv")
    context = np.vstack(
        [encode_context(json.loads(r["semantic_context_json"]), ontology) for r in all_values]
    )
    np.save(cache / "ac50_semantic_context.npy", context, allow_pickle=False)
    np.save(cache / "ac50_semantic_context_valid.npy", np.ones(78, dtype=bool), allow_pickle=False)
    groups[CONTEXT_GROUP] = dict(
        matrix="ac50_semantic_context.npy",
        valid="ac50_semantic_context_valid.npy",
        index="ac50_context_index.tsv",
        key_kind="record_id",
        eligibility_field="ac50_context_eligible",
        shape=[78, ontology["dimensions"]],
        feature_names=ontology["feature_names"],
        chemical_method=False,
        source_kind="unlabelled_source_semantic_context",
    )
    table(cache / "structure_bank_index.tsv", [dict(key=k, row_idx=i) for i, k in enumerate(keys)])
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=k, canonical_isomeric_smiles=graph_map[k]) for k in keys],
    )
    table(
        cache / "ac50_context_index.tsv",
        [dict(key=r["record_id"], row_idx=i) for i, r in enumerate(all_values)],
    )
    table(
        cache / "record_bank_index.tsv",
        [
            dict(
                record_id=r["record_id"],
                structure_key=r["canonical_identity_group"],
                sequence_key="",
            )
            for r in all_values
        ],
    )
    registry = dict(
        schema="ac50_native_ten_chemical_methods_v1",
        methods=methods,
        independent_method_count=10,
        context_method_increment=0,
        chemical_dimensions=sum(g["shape"][1] for n, g in groups.items() if n != CONTEXT_GROUP),
        context_columns=ontology["dimensions"],
        biological_equivalence_of_readouts_claimed=False,
    )
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", configuration)
    write(cache / "semantic_context_ontology.json", ontology)
    write(
        cache / "copied_vector_audit.json",
        dict(
            reused_graphs=17,
            new_graphs=0,
            chemical_feature_extractions=0,
            all78_record_aliases_preserved=True,
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
        requested_records=78,
        training_records=77,
        feature_only_records=[NR13],
        structure_keys=17,
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
        payload = load_feature_route(all_values, manifest, method["route"])
        if (
            not payload["valid"].all()
            or payload["X"].shape != (78, method["dimensions"] + ontology["dimensions"])
            or payload["feature_names"][-ontology["dimensions"] :]
            != [CONTEXT_GROUP + ":" + n for n in ontology["feature_names"]]
            or not np.array_equal(payload["X"][:, -ontology["dimensions"] :], context)
        ):
            raise ValueError("A chemical route lacks exact once-only78-row semantic context")
    _verify_artifacts(base)
    if sha(pointer) != PARENT_SHA:
        raise ValueError("Immutable parent bank pointer changed during feature slicing")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        dict(
            status="verified",
            records=78,
            training_records=77,
            feature_only_records=[NR13],
            graphs=17,
            independent_chemical_methods=10,
            chemical_groups=13,
            context_groups=1,
            chemical_dimensions=registry["chemical_dimensions"],
            context_columns=ontology["dimensions"],
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
    """Portable source/role/label proof; no RDKit, original CSV or network dependency."""
    check_dependency()
    validate_rows(values, verify_chemistry=False)
    descriptor = task.get("native_source_observation_coverage", {})
    scope = dict(
        original_endpoint="AC50",
        target_endpoint="AC50",
        quantity_id=QUANTITY,
        original_clear_records=78,
        included_core_records=77,
        original_molecular_groups=17,
        core_molecular_groups=17,
        feature_precomputed_records=78,
        excluded_feature_only_ids=[NR13],
        query_mode="source_annotation",
        biological_potency_equivalence_verified=False,
        assay_host_species_known=False,
        full_refit_required_for_training_coverage=True,
    )
    definition = task.get("scientific_definition", {})
    if (
        task.get("task_id") != TASK
        or task.get("endpoint") != "AC50"
        or task.get("canonical_quantity_id") != QUANTITY
        or any(descriptor.get(k) != v for k, v in scope.items())
        or definition.get("implementation_sha256") != sha(__file__)
        or definition.get("query_mode") != "source_annotation"
        or definition.get("context_fields") != list(CONTEXT_FIELDS)
        or definition.get("biological_potency_equivalence_verified") is not False
        or definition.get("assay_host_species_known") is not False
        or definition.get("separate_all_common_refit_required") is not True
        or definition.get("context_is_chemical_method") is not False
        or definition.get("PDK1_biology_unresolved_exact_annotation_only") is not True
        or definition.get("p21_is_induction_marker_not_HDAC_binding") is not True
        or definition.get("unresolved_NR13_feature_only") is not True
        or definition.get("source_external_validation_available") is not False
        or definition.get("diagnostic_controls_precision") != "float64"
        or definition.get("diagnostic_control_precision_policy") != CONTROL_PRECISION_POLICY
        or definition.get("target_text_portability_policy") != TARGET_TEXT_POLICY
    ):
        raise ValueError("Native77/78 coverage, role/quantity or query scope changed")
    path = contained(root, descriptor["proof_path"])
    if sha(path) != descriptor["proof_sha256"]:
        raise ValueError("Native AC50 source proof SHA changed")
    proof = read(path)
    if (
        proof.get("schema") != PROOF_SCHEMA
        or proof.get("task_id") != TASK
        or any(proof.get(k) != v for k, v in scope.items())
        or proof.get("implementation_sha256") != sha(__file__)
        or proof.get("implementation_dependencies_sha256")
        != {"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA}
        or proof.get("source_delivery_sha256") != SOURCE_DELIVERY_SHA
        or proof.get("literal10000nM_censored_rows_excluded") != 244
        or proof.get("raw_predictions_may_not_be_mutated") is not True
        or proof.get("diagnostic_controls_precision") != "float64"
        or proof.get("diagnostic_control_precision_policy") != CONTROL_PRECISION_POLICY
        or proof.get("target_text_portability_policy") != TARGET_TEXT_POLICY
    ):
        raise ValueError("Source-native AC50 proof schema/dependency or scientific scope changed")
    for name in ("task_table", "all78_ledger", "ontology", "predeclared_splits"):
        item = proof[name]
        if sha(contained(root, item["path"])) != item["sha256"]:
            raise ValueError("A source/native context or frozen split artifact changed: " + name)
    if (
        proof["task_table"]["path"] != task["table_path"]
        or rows(contained(root, proof["task_table"]["path"])) != values
    ):
        raise ValueError("The exact77 source task table differs from its proof")
    snapshots = contained(root, proof["source_snapshot_directory"])
    all_expected, candidates, _, _ = source_rows(
        snapshots, verify_chemistry=False, verify_raw_source_pins=False
    )
    expected_fields = list(dict.fromkeys(field for row in all_expected for field in row))
    all_expected = [{field: r.get(field, "") for field in expected_fields} for r in all_expected]
    frozen_ledger = rows(contained(root, proof["all78_ledger"]["path"]))
    compare_source_rows(frozen_ledger, all_expected)
    if values != [r for r in frozen_ledger if r["core_training_eligible"] == "true"]:
        raise ValueError("The77 core differs from the exact78 source ledger")
    compare_source_rows(values, [r for r in all_expected if r["core_training_eligible"] == "true"])
    ontology = read(contained(root, proof["ontology"]["path"]))
    if (
        ontology != make_ontology(candidates)
        or definition["context_columns"] != ontology["dimensions"]
    ):
        raise ValueError(
            "The ontology includes inferred, outcome-derived or unapproved input fields"
        )
    if proof["record_ids"] != [r["record_id"] for r in values] or proof["all78_record_ids"] != [
        r["record_id"] for r in all_expected
    ]:
        raise ValueError("Source proof member IDs were enlarged or collapsed")
    return proof


def _baseline_provenance(output):
    """Keep the exact unfitted v1 source/inputs separate from the new diagnostic policy."""
    output = Path(output)
    provenance = output / "source_provenance"
    if not provenance.exists():
        shutil.copytree(OUTPUT / "source_provenance", provenance)
    if (
        sha(provenance / "peptide_modeling_ac50_native_bridge_v1.py") != V1_IMPLEMENTATION_SHA
        or sha(provenance / "baseline_v1_delivery_receipt.json") != V1_DELIVERY_SHA
        or sha(V1_OUTPUT / "delivery_receipt.json") != V1_DELIVERY_SHA
    ):
        raise ValueError("The original v1 source/delivery snapshot was not preserved")
    baseline = read(V1_OUTPUT / "delivery_receipt.json")
    if len(baseline["files"]) != 75 or any(
        sha(V1_OUTPUT / name) != expected for name, expected in baseline["files"].items()
    ):
        raise ValueError("A protected v1 prepared input or cached array changed")
    return baseline


def prepare(source=SOURCE, parent=PARENT, output=OUTPUT):
    check_dependency()
    output = Path(output)
    if output.exists():
        if {p.name for p in output.iterdir()} != {"source_provenance"}:
            raise FileExistsError(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
    baseline = _baseline_provenance(output)
    all_values, candidates, _, delivery = source_rows(source)
    ontology = make_ontology(candidates)
    ledger_path = output / "task_tables/native78_ledger.tsv"
    table(ledger_path, all_values)
    all_values = rows(ledger_path)
    core = [r for r in all_values if r["core_training_eligible"] == "true"]
    validate_rows(core)
    source_table = "task_tables/" + TASK + ".tsv"
    table(output / source_table, core)
    checks = split_prechecks(core, output)
    check_path = output / "task_tables/predeclared_splits/manifest.json"
    write(check_path, checks)
    write(output / "fivefold_prechecks.json", checks)
    ontology_path = output / "task_tables/ac50_context_ontology.json"
    write(ontology_path, ontology)
    snapshots = output / "task_tables/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=False)
    for relative in ["delivery_receipt.json", *delivery["artifact_sha256"]]:
        target = snapshots / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(source) / relative, target)
    scope = dict(
        original_endpoint="AC50",
        target_endpoint="AC50",
        quantity_id=QUANTITY,
        original_clear_records=78,
        included_core_records=77,
        original_molecular_groups=17,
        core_molecular_groups=17,
        feature_precomputed_records=78,
        excluded_feature_only_ids=[NR13],
        query_mode="source_annotation",
        biological_potency_equivalence_verified=False,
        assay_host_species_known=False,
        full_refit_required_for_training_coverage=True,
    )
    proof = dict(
        schema=PROOF_SCHEMA,
        task_id=TASK,
        **scope,
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA},
        record_ids=[r["record_id"] for r in core],
        all78_record_ids=[r["record_id"] for r in all_values],
        task_table=dict(path=source_table, sha256=sha(output / source_table)),
        all78_ledger=dict(path=ledger_path.relative_to(output).as_posix(), sha256=sha(ledger_path)),
        ontology=dict(path=ontology_path.relative_to(output).as_posix(), sha256=sha(ontology_path)),
        predeclared_splits=dict(
            path=check_path.relative_to(output).as_posix(), sha256=sha(check_path)
        ),
        source_snapshot_directory=snapshots.relative_to(output).as_posix(),
        source_delivery_sha256=SOURCE_DELIVERY_SHA,
        literal10000nM_censored_rows_excluded=244,
        raw_predictions_may_not_be_mutated=True,
        diagnostic_controls_precision="float64",
        diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
        target_text_portability_policy=copy.deepcopy(TARGET_TEXT_POLICY),
        warning=WARNING,
    )
    proof_path = output / "task_tables/native_observation_coverage.json"
    write(proof_path, proof)
    definition = dict(
        endpoint="AC50",
        canonical_quantity_id=QUANTITY,
        quantity_unit="uM",
        native_metric_annotation="AC50",
        source_scope="exact_native_AC50_given_semantic_readout_and_target_role",
        query_mode="source_annotation",
        context_fields=list(CONTEXT_FIELDS),
        context_group=CONTEXT_GROUP,
        context_columns=ontology["dimensions"],
        context_is_chemical_method=False,
        biological_potency_equivalence_verified=False,
        assay_host_species_known=False,
        PDK1_biology_unresolved_exact_annotation_only=True,
        p21_is_induction_marker_not_HDAC_binding=True,
        unresolved_NR13_feature_only=True,
        source_external_validation_available=False,
        diagnostic_controls_precision="float64",
        diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
        target_text_portability_policy=copy.deepcopy(TARGET_TEXT_POLICY),
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=77,
        separate_all_common_refit_required=True,
        implementation_module="peptide_modeling_ac50_native_bridge",
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA},
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="AC50",
        task_kind="regression",
        status="eligible",
        training_admitted=True,
        table_path=source_table,
        n_records=77,
        n_groups=17,
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
    verify_source_proof(output, task, core)
    counts = dict(
        input_records=78,
        feature_index_records=78,
        endpoints_accounted=1,
        endpoints_with_eligible_tasks=1,
        eligible_tasks=1,
        eligible_table_records=77,
        feature_only_records=1,
        deferred_records=0,
        collapsed_source_copies=0,
        label_conflict_records=0,
    )
    write(
        output / "task_manifest.json",
        dict(schema=SCHEMA, tasks=[task], counts=counts, models_fitted=0),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    table(output / "feature_rows_index.tsv", all_values)
    table(
        output / "record_dispositions.tsv",
        [
            dict(
                record_id=r["record_id"],
                disposition=r["disposition"],
                training_eligible=r["core_training_eligible"],
            )
            for r in all_values
        ],
    )
    table(
        output / "task_summary.tsv",
        [dict(task_id=TASK, endpoint="AC50", status="eligible", records=77, molecular_groups=17)],
    )
    features = slice_features(all_values, ontology, parent, output)
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
        mandatory_postprocessing="source_semantic_support_strata_and_all77_refit_diagnostics",
    )
    write(output / "config.json", config)
    write(output / "recommended_training_config.json", config)
    write(
        output / "deployment_requirements.json",
        dict(
            wrapper_entrypoint="scripts/peptide_modeling_ac50_native_bridge.py",
            wrapper_arguments=["--run-followup", "--config", "<followup_config.json>"],
            explicitly_include_before_recursive_import_closure=[
                "peptide_modeling_ac50_native_bridge",
                "peptide_modeling_followup_seeds",
                "peptide_modeling_mhc10_native_bridge",
            ],
            implementation_sha256=sha(__file__),
            implementation_dependencies_sha256={
                "peptide_modeling_mhc10_native_bridge.py": SHARED_SHA
            },
            record_id_context_feature_loader_required=True,
            actual_node_inner_plan_frozen_before_any_fit=True,
            report_schema=REPORT_SCHEMA,
            report_filename="ac50_native_molecular_diagnostics.json",
            diagnostic_controls_precision="float64",
            diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
            supported_public_inference_entrypoint="predict_from_artifacts",
            do_not_edit_frozen_implementation_after_model_fitting=True,
        ),
    )
    write(
        output / "source_admission.json",
        dict(
            status="prepared_not_trained",
            task_id=TASK,
            original_clear_records=78,
            core_training_records=77,
            molecular_groups=17,
            feature_only_records=[NR13],
            source_annotation_not_biological_equivalence=True,
            source_proof_sha256=sha(proof_path),
            warning=WARNING,
        ),
    )
    from peptide_modeling_pipeline import report_stage

    smoke = output / "diagnostic_history/report_stage_smoke"
    smoke.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output / "task_manifest.json", smoke / "task_manifest.json")
    if report_stage(smoke) != dict(completed=0, deferred=0, failed=0):
        raise ValueError("Prepared AC50 report-stage smoke unexpectedly reports model fits")
    write(
        output / "preparation_validation.json",
        dict(
            status="prepared_not_trained",
            report_stage_smoke="passed_without_model_fitting",
            task_counts=counts,
            ontology_dimensions=ontology["dimensions"],
            source_original78_preserved=True,
        ),
    )
    (output / "REPORT.md").write_text(
        "# Native AC50 semantic source annotations\n\n77training candidates/17whole-graph plus "
        "known-FAMILY groups; all78clear observations have exact chemical/context features. "
        "Unresolved NR13 remains feature/ledger-only. Ten chemical methods/13groups/15,212D "
        f"plus{ontology['dimensions']}fixed semantic context columns; context adds no methods. "
        "17existing graph vectors are byte-reused, zero chemical feature extraction. Eight "
        "models/Optuna128/seeds42,2026,3407; actual outer group holdout/inner5 and mandatory "
        "separate all77 refit. All three fixed-Ridge diagnostic designs use uniform float64 "
        "fit/predict inputs obtained solely by exact upcasting of stored float32 design values. "
        "Selected model training, inference and its32ULP provenance guard remain unchanged.\n\n"
        + WARNING
        + "\n\nNo fitting or remote operations by preparation. "
        "Freeze actual node inner plans before search; retain strict fold membership checks. "
        "Deployment includes the CLI module explicitly and its recursive scripts import closure "
        "and the generic followup_seeds entrypoint; numerical helpers are SHA-pinned.\n",
        encoding="utf-8",
    )
    receipt = dict(
        status="prepared_not_trained",
        task_id=TASK,
        original_records=78,
        training_records=77,
        molecules=17,
        feature_fingerprint=features["input_fingerprint"],
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA},
        source_proof_sha256=sha(proof_path),
        preparation_revision="prespecified_float64_diagnostic_controls_v2",
        diagnostic_controls_precision="float64",
        baseline_v1_delivery_sha256=V1_DELIVERY_SHA,
        files={
            p.relative_to(output).as_posix(): sha(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
        },
    )
    prior_feature = read(V1_OUTPUT / "features/manifest.json")
    current_root, prior_root = Path(features["cache_dir"]), Path(prior_feature["cache_dir"])
    for name, meta in features["groups"].items():
        previous = prior_feature["groups"][name]
        for field in ("matrix", "valid", "index"):
            if sha(current_root / meta[field]) != sha(prior_root / previous[field]):
                raise ValueError("A v2 numeric feature/validity/index byte differs from v1")
    if (
        sha(output / source_table) != baseline["files"][source_table]
        or sha(ledger_path) != baseline["files"][ledger_path.relative_to(output).as_posix()]
        or sha(ontology_path) != baseline["files"][ontology_path.relative_to(output).as_posix()]
        or sha(check_path) != baseline["files"][check_path.relative_to(output).as_posix()]
        or config["training"] != read(V1_OUTPUT / "config.json")["training"]
    ):
        raise ValueError("Source77/78 rows, ontology, frozen folds or model settings changed in v2")
    _baseline_provenance(output)
    write(
        output / "v1_numeric_equivalence.json",
        dict(
            status="verified_before_any_production_fit",
            baseline_v1_implementation_sha256=V1_IMPLEMENTATION_SHA,
            baseline_v1_delivery_sha256=V1_DELIVERY_SHA,
            baseline_active_files_unchanged=75,
            all17_chemical_graph_arrays_and_masks_byte_identical=True,
            all78_semantic_context_rows_and_masks_byte_identical=True,
            all_feature_indices_byte_identical=True,
            core77_source_table_sha256=sha(output / source_table),
            original78_ledger_sha256=sha(ledger_path),
            ontology_sha256=sha(ontology_path),
            frozen_outer_and_inner_plan_sha256=sha(check_path),
            source_tables_ontology_plans_and_model_settings_unchanged=True,
            selected_model_training_input_precision="stored_float32_unchanged",
            diagnostic_controls_precision="float64",
            diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
            controls_only_exact_upcast_of_existing_float32_designs=True,
            selected_32ULP_guard_unchanged=True,
            production_models_fitted=0,
        ),
    )
    receipt["files"] = {
        p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and "diagnostic_history" not in p.relative_to(output).parts
    }
    write(output / "delivery_receipt.json", receipt)
    return receipt


def _bits32(value):
    if not math.isfinite(value):
        raise ValueError("A prediction is nonfinite")
    try:
        packed = struct.pack("!f", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError("A prediction is outside binary32 range") from exc
    if struct.unpack("!f", packed)[0] != value:
        raise ValueError("Precision allowance requires exact binary32 predictions")
    bits = struct.unpack("!I", packed)[0]
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits


def precision_evidence(source, payload, *, diagnostic_precision=None):
    X, valid = np.asarray(payload["X"]), np.asarray(payload["valid"])
    if (
        payload["record_ids"] != [r["record_id"] for r in source]
        or diagnostic_precision not in {None, "float64"}
        or X.dtype != np.dtype("float64" if diagnostic_precision == "float64" else "float32")
        or X.ndim != 2
        or len(X) != 77
        or valid.shape != (77,)
        or valid.dtype != np.dtype("bool")
        or not valid.all()
        or X.shape[1] != len(payload["feature_names"])
        or np.isinf(X).any()
        or np.isnan(X).all(axis=1).any()
    ):
        raise ValueError(
            "Precision evidence needs the exact selected/control input dtype and schema"
        )
    if diagnostic_precision == "float64" and not np.array_equal(
        X, X.astype(np.float32).astype(np.float64), equal_nan=True
    ):
        raise ValueError("Float64 controls must exactly upcast stored float32 design values")
    return dict(
        record_rows={
            r["record_id"]: dict(
                graph=r["canonical_identity_group"],
                group=r["split_group"],
                context=r["semantic_context_key"],
                row_sha256=hashlib.sha256(X[i].tobytes()).hexdigest(),
            )
            for i, r in enumerate(source)
        },
        feature_width=X.shape[1],
        matrix_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        feature_names=payload["feature_names"],
        input_dtype=str(X.dtype),
        diagnostic_precision=diagnostic_precision,
    )


def prediction_consistency(values, evidence, model_context):
    """Bound only same-graph/context/fold binary32 LSQR drift, never mutate predictions."""
    if not values:
        raise ValueError("Empty prediction consistency group")
    predictions = [float(r["prediction"]) for r in values]
    bound = [evidence["record_rows"][r["record_id"]] for r in values]
    if (
        not all(math.isfinite(p) for p in predictions)
        or len({b["graph"] for b in bound}) != 1
        or len({b["context"] for b in bound}) != 1
        or len({b["group"] for b in bound}) != 1
        or len({b["row_sha256"] for b in bound}) != 1
        or len({(r["evaluation_role"], int(r["fold"])) for r in values}) != 1
    ):
        raise ValueError("Consistency comparison lacks identical full input/graph/context/fold")
    original_pass = bool(np.allclose(predictions, predictions[0], rtol=1e-7, atol=1e-8))
    if evidence.get("diagnostic_precision") == "float64":
        if not original_pass:
            raise ValueError("Float64 fixed controls fail the unchanged strict consistency guard")
        return dict(
            policy="same_full_input_graph_context_fold_float64_original_tolerance_v2",
            mode="original_float_tolerance",
            diagnostic_precision="float64",
            binary32_exception_allowed=False,
            full_selected_row_sha256=bound[0]["row_sha256"],
            raw_predictions_unchanged=True,
            source_record_ids=[r["record_id"] for r in values],
        )
    ulps = None
    try:
        bits = [_bits32(p) for p in predictions]
        ulps = max(bits) - min(bits)
    except ValueError:
        if not original_pass:
            raise
    mode = "original_float_tolerance"
    if not original_pass:
        policy = model_context.get("linear_solver_policy", {})
        if (
            model_context.get("selected_model") != "linear"
            or policy.get("version") != "dimension_based_linear_policy_v2"
            or policy.get("solver") != "lsqr"
            or policy.get("selection_basis") != "raw_route_feature_count_only"
            or policy.get("scaler_with_mean") is not False
            or policy.get("feature_count") != evidence["feature_width"]
            or any(abs(p) < 2.0**-126 for p in predictions)
            or len({math.copysign(1.0, p) for p in predictions}) != 1
            or ulps is None
            or ulps > 32
        ):
            raise ValueError(
                "Identical full-input prediction drift lacks the bounded32ULP LSQR proof"
            )
        mode = "exact_binary32_LSQR_at_most32ULP"
    return dict(
        policy="same_full_input_graph_context_fold_binary32_LSQR_32ULP_v1",
        mode=mode,
        observed_binary32_ulp_distance=ulps,
        engineering_budget_not_mathematical_BLAS_bound=True,
        maximum_binary32_ulp_budget=32,
        full_selected_row_sha256=bound[0]["row_sha256"],
        raw_predictions_unchanged=True,
        source_record_ids=[r["record_id"] for r in values],
    )


def context_support(row, fitted_rows):
    target = row["semantic_target_key"]
    readout = row["native_readout_family"]
    target_seen = target in {r["semantic_target_key"] for r in fitted_rows}
    readout_seen = readout in {r["native_readout_family"] for r in fitted_rows}
    joint_seen = row["semantic_context_key"] in {r["semantic_context_key"] for r in fitted_rows}
    return dict(
        target_annotation_seen_in_training=target_seen,
        target_category_seen_in_training=row["target_semantic_category"]
        in {r["target_semantic_category"] for r in fitted_rows},
        readout_seen_in_training=readout_seen,
        joint_context_seen_in_training=joint_seen,
        coarse_target_and_readout_supported=target_seen and readout_seen,
        qualified_joint_query_supported=joint_seen and target_seen and readout_seen,
        source_paper_shared_with_training=row["source_pmid"]
        in {r["source_pmid"] for r in fitted_rows},
    )


def evaluation_diagnostics(source, inner_oof, outer_holdout, check, payload, model_context):
    validate_rows(source, verify_chemistry=False)
    by_id = {r["record_id"]: r for r in source}
    predictions = list(inner_oof) + list(outer_holdout)
    ids = [r["record_id"] for r in predictions]
    if len(ids) != 77 or len(set(ids)) != 77 or set(ids) != set(by_id) or NR13 in ids:
        raise ValueError("Exactly the77 admitted observations must have one evaluation prediction")
    if {by_id[r]["split_group"] for r in check["train_record_ids"]} & {
        by_id[r]["split_group"] for r in check["holdout_record_ids"]
    }:
        raise ValueError("An exact AC50 graph/FAMILY crosses outer training and holdout")
    folds = {f["fold"]: f for f in check["folds"]}
    if set(folds) != set(range(5)) or len(check["folds"]) != 5:
        raise ValueError("The actual source-frozen inner fivefold plan is incomplete")
    fold_by_id = {r: f["fold"] for f in check["folds"] for r in f["validation_record_ids"]}
    evidence = precision_evidence(
        source, payload, diagnostic_precision=model_context.get("diagnostic_precision")
    )
    report = dict(
        evaluation_by_role={},
        model_selection_repeated=False,
        independently_evaluated_all77_refit=False,
        quantity_id=QUANTITY,
        original_clear_records=78,
        core_records=77,
        feature_only_ids=[NR13],
        source_external_validation_claimed=False,
        biological_potency_equivalence_claimed=False,
        warning=WARNING,
    )
    for role, predicted, expected in (
        ("inner_oof", inner_oof, check["train_record_ids"]),
        ("outer_holdout", outer_holdout, check["holdout_record_ids"]),
    ):
        if {r["record_id"] for r in predicted} != set(expected):
            raise ValueError("Saved AC50 prediction roles differ from the frozen outer split")
        enriched, comparisons = [], defaultdict(list)
        for value in predicted:
            row = by_id[value["record_id"]]
            fold = int(value["fold"])
            if (
                value["split_group"] != row["split_group"]
                or not math.isclose(
                    float(value["target"]), float(row["target"]), rel_tol=0, abs_tol=1e-10
                )
                or not math.isfinite(float(value["prediction"]))
                or (role == "inner_oof" and fold_by_id[row["record_id"]] != fold)
                or (role == "outer_holdout" and fold != -1)
            ):
                raise ValueError("A saved AC50 source target/group/actual fold changed")
            fitted_ids = (
                folds[fold]["train_record_ids"]
                if role == "inner_oof"
                else check["train_record_ids"]
            )
            fitted_rows = [by_id[r] for r in fitted_ids]
            if row["split_group"] in {r["split_group"] for r in fitted_rows}:
                raise ValueError("An AC50 evaluation molecule entered its own training fold")
            enriched_row = dict(
                **value,
                evaluation_role=role,
                native_readout_family=row["native_readout_family"],
                measured_direction=row["measured_direction"],
                target_semantic_category=row["target_semantic_category"],
                target_role=row["target_role"],
                semantic_target_key=row["semantic_target_key"],
                semantic_context_key=row["semantic_context_key"],
                source_pmid=row["source_pmid"],
                annotated_organism_role=row["annotated_organism_role"],
                assay_host_species="not_reported",
                **context_support(row, fitted_rows),
            )
            enriched.append(enriched_row)
            comparisons[
                (row["canonical_identity_group"], row["semantic_context_key"], fold)
            ].append(enriched_row)
        consistency = [
            prediction_consistency(group, evidence, model_context) for group in comparisons.values()
        ]
        strata = {}
        for field in (
            "native_readout_family",
            "measured_direction",
            "target_semantic_category",
            "target_role",
            "semantic_context_key",
            "source_pmid",
        ):
            strata[field] = {
                category: weighted_errors([r for r in enriched if r[field] == category])
                for category in sorted({r[field] for r in enriched})
            }
        support_counts = dict(
            unseen_target_annotation=sum(
                not r["target_annotation_seen_in_training"] for r in enriched
            ),
            unseen_target_category=sum(not r["target_category_seen_in_training"] for r in enriched),
            unseen_readout=sum(not r["readout_seen_in_training"] for r in enriched),
            unseen_joint_context=sum(not r["joint_context_seen_in_training"] for r in enriched),
            qualified_joint_supported=sum(r["qualified_joint_query_supported"] for r in enriched),
            coarse_target_readout_supported=sum(
                r["coarse_target_and_readout_supported"] for r in enriched
            ),
            shared_source_paper=sum(r["source_paper_shared_with_training"] for r in enriched),
        )
        report["evaluation_by_role"][role] = dict(
            **weighted_errors(enriched),
            strata=strata,
            support_counts=support_counts,
            predictions=enriched,
            prediction_consistency=consistency,
            score_role="model_selection" if role == "inner_oof" else "outer_train_model_evaluation",
            score_applies_to_all77_refit=False,
        )
    report["singleton_context_diagnostics"] = {
        record_id: next(
            r
            for role in report["evaluation_by_role"].values()
            for r in role["predictions"]
            if r["record_id"] == record_id
        )
        for record_id in [PDK1, P21]
        + [row["record_id"] for row in source if row["native_readout_family"] == "agonist_activity"]
    }
    return report


def fixed_ridge_controls(source, chemical_X, context_X, check):
    from peptide_modeling_train import _cv_predictions, _estimator, _fit_model, _linear_policy

    validate_rows(source, verify_chemistry=False)
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
    y = np.asarray([float(r["target"]) for r in source])
    settings = dict(seed=check["seed"], n_jobs=1)
    chemical_X, context_X = np.asarray(chemical_X), np.asarray(context_X)
    if (
        chemical_X.dtype != np.dtype("float32")
        or context_X.dtype != np.dtype("float32")
        or chemical_X.ndim != 2
        or context_X.ndim != 2
        or len(chemical_X) != len(source)
        or len(context_X) != len(source)
        or not chemical_X.shape[1]
        or not context_X.shape[1]
        or np.isinf(chemical_X).any()
        or np.isinf(context_X).any()
        or np.isnan(chemical_X).all(axis=1).any()
        or np.isnan(context_X).all(axis=1).any()
    ):
        raise ValueError("Fixed controls start from the exact stored float32 input values")
    reports = {}
    for name, stored_design in (
        ("molecule_only", chemical_X),
        ("context_only", context_X),
        ("molecule_plus_context", np.column_stack((chemical_X, context_X))),
    ):
        X = np.asarray(stored_design, dtype=np.float64)
        estimator = _estimator(
            "linear", "regression", {"regularization": 1.0}, settings, X.shape[1]
        )
        oof, folds = _cv_predictions(estimator, X[train], y[train], splits, "regression", 1)
        fitted = _fit_model(estimator, X[train], y[train], 1)
        pred = fitted.predict(X[test])
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
            for i, p in zip(test, pred, strict=True)
        ]
        payload = dict(
            X=X,
            valid=np.ones(len(source), dtype=bool),
            feature_names=["fixed_control_column" + str(i) for i in range(X.shape[1])],
            record_ids=[r["record_id"] for r in source],
        )
        reports[name] = dict(
            model="fixed_ridge_alpha1_LSQR_float64_not_Optuna_selected",
            diagnostic_precision="float64",
            diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
            exact_upcast_of_stored_float32_design=True,
            stored_float32_design_sha256=hashlib.sha256(
                np.ascontiguousarray(stored_design).tobytes()
            ).hexdigest(),
            diagnostic_float64_design_sha256=hashlib.sha256(
                np.ascontiguousarray(X).tobytes()
            ).hexdigest(),
            same_frozen_splits=True,
            **evaluation_diagnostics(
                source,
                inner,
                outer,
                check,
                payload,
                dict(
                    selected_model="linear",
                    linear_solver_policy=_linear_policy("regression", X.shape[1]),
                    diagnostic_precision="float64",
                ),
            ),
        )
    return reports


def verify_refit_contract(run, task, source, result, proof):
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features
    from peptide_modeling_train import _digest

    if (
        result.get("status") != "completed"
        or result.get("refit_status") != "completed"
        or result.get("deployment_ready") is not True
    ):
        raise ValueError("A distinct completed all77-data refit is mandatory for training coverage")
    paths = {name: shared.artifact_path(run, value) for name, value in result["artifacts"].items()}
    if "refit_model" not in paths or str(paths["refit_model"]) not in result["artifact_hashes"]:
        raise ValueError("All77 refit artifact is missing or not hash covered")
    for path, expected in result["artifact_hashes"].items():
        if sha(shared.artifact_path(run, path)) != expected["sha256"]:
            raise ValueError("An AC50 model/refit/prediction/source artifact changed")
    contract, inputs, split = (
        read(paths["model_contract"]),
        read(paths["input_contract"]),
        read(paths["outer_split"]),
    )
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=task,
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=inputs["config"],
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=split["input_hash"],
    )
    if (
        inputs["implementation_sha256"] != sha(Path(run) / "scripts/peptide_modeling_train.py")
        or not (
            _digest(signature)
            == inputs["input_hash"]
            == result["input_hash"]
            == contract["input_hash"]
        )
        or inputs["config"].get("refit_full") is not True
        or inputs["task"] != task
        or contract["task"] != task
        or contract.get("common_record_ids") != [r["record_id"] for r in source]
        or contract.get("optional_refit_fit_scope") != "all_common_after_evaluation"
        or contract.get("model_fit_scope") != "outer_train_only"
        or contract.get("holdout_not_used_for_selection") is not True
        or contract.get("refit_status") != "completed"
        or contract.get("target_unit") != "uM"
    ):
        raise ValueError("The all77 refit/input signature and independent evaluation scopes differ")
    ontology = read(contained(run, proof["ontology"]["path"]))
    payload = load_feature_route(source, portable_features(run), result["selected_route"])
    expected = inputs["feature_hashes"][result["selected_route"]]
    context = np.vstack(
        [encode_context(json.loads(r["semantic_context_json"]), ontology) for r in source]
    )
    names = [CONTEXT_GROUP + ":" + n for n in ontology["feature_names"]]
    if (
        not payload["valid"].all()
        or payload["record_ids"] != [r["record_id"] for r in source]
        or payload["feature_names"] != contract["feature_names"]
        or payload["feature_names"][-ontology["dimensions"] :] != names
        or sum(n.startswith(CONTEXT_GROUP + ":") for n in payload["feature_names"])
        != ontology["dimensions"]
        or not np.array_equal(payload["X"][:, -ontology["dimensions"] :], context)
        or expected["names"] != payload["feature_names"]
        or expected["matrix_sha256"]
        != hashlib.sha256(np.ascontiguousarray(payload["X"]).tobytes()).hexdigest()
        or expected["mask_sha256"] != hashlib.sha256(payload["valid"].tobytes()).hexdigest()
    ):
        raise ValueError("All77 chemical/context feature membership or schema changed")
    if (
        result["selected_model"] == "linear"
        and contract.get("linear_solver_policy")
        != inputs["linear_solver_policies"][result["selected_route"]]
    ):
        raise ValueError("The exact selected linear solver policy differs from its fitted input")
    return paths, contract, payload, ontology


def write_diagnostics(config_path, journal=None):
    from peptide_modeling_summarize import verified_predictions
    from peptide_modeling_supervise import atomic_json

    check_dependency()
    config = read(config_path)
    root, durable = Path(config["output_root"]), Path(config["durable_root"])
    journal = journal or read(root / "campaign_status.json")
    if len(journal["rounds"]) != 3 or {r["seed"] for r in journal["rounds"]} != set(SEEDS):
        raise ValueError("All three exact AC50 seed runs are required for mandatory reporting")
    reports = []
    for entry in journal["rounds"]:
        if (
            entry["status"] != "completed"
            or entry["audit"].get("missing_completed_task_ids")
            or not entry["audit"].get("method_model_coverage_met")
        ):
            raise ValueError("A required AC50 seed lacks scientific task/method/model completion")
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
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
            raise ValueError("The actual AC50 task manifest is not seed-verification bound")
        verified_predictions(run, run, result, task)
        paths, contract, payload, ontology = verify_refit_contract(run, task, source, result, proof)
        check = next(
            c
            for c in read(contained(run, proof["predeclared_splits"]["path"]))["seeds"]
            if c["seed"] == entry["seed"]
        )
        inner, holdout = rows(paths["oof"]), rows(paths["holdout"])
        report = evaluation_diagnostics(
            source,
            inner,
            holdout,
            check,
            payload,
            dict(
                selected_model=result["selected_model"],
                linear_solver_policy=contract.get("linear_solver_policy"),
            ),
        )
        for role, saved in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            for key in ("mae", "rmse"):
                if not math.isclose(
                    report["evaluation_by_role"][role]["record_weighted"][key],
                    result[saved][key],
                    rel_tol=1e-9,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        "Native AC50 molecular report does not reproduce saved role metrics"
                    )
        width = ontology["dimensions"]
        report["fixed_ridge_controls"] = fixed_ridge_controls(
            source, payload["X"][:, :-width], payload["X"][:, -width:], check
        )
        report.update(
            seed=entry["seed"],
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            selected_feature_width=payload["X"].shape[1],
            selected_input_hash=result["input_hash"],
            source_table_sha256=sha(run / task["table_path"]),
            result_sha256=sha(path),
            oof_sha256=sha(paths["oof"]),
            holdout_sha256=sha(paths["holdout"]),
            outer_model_sha256=sha(paths["model"]),
            input_contract_sha256=sha(paths["input_contract"]),
            source_proof_sha256=task["native_source_observation_coverage"]["proof_sha256"],
            predeclared_node_plan_sha256=proof["predeclared_splits"]["sha256"],
            precision_consistency_policy="same_full_input_graph_context_fold_binary32_LSQR_32ULP_v1",
            separate_all_common_refit=dict(
                status="verified",
                records=77,
                molecular_groups=17,
                original_clear_records=78,
                excluded_feature_only_ids=[NR13],
                training_record_ids=contract["common_record_ids"],
                model_path=str(paths["refit_model"]),
                model_sha256=sha(paths["refit_model"]),
                model_contract_sha256=sha(paths["model_contract"]),
                fit_scope="all_common_after_evaluation",
                independently_evaluated_on_these77=False,
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
        original_clear_records=78,
        core_training_records=77,
        original_molecular_groups=17,
        feature_only_ids=[NR13],
        biological_potency_equivalence_claimed=False,
        source_external_validation_claimed=False,
        model_selection_repeated=False,
        raw_predictions_unchanged=True,
        diagnostic_controls_precision="float64",
        diagnostic_control_precision_policy=CONTROL_PRECISION_POLICY,
        warning=WARNING,
    )
    path = root / "ac50_native_molecular_diagnostics.json"
    atomic_json(path, payload)
    durable.mkdir(parents=True, exist_ok=True)
    destination = durable / path.name
    shutil.copy2(path, destination)
    if sha(path) != sha(destination):
        raise ValueError("Durable mandatory AC50 report SHA differs")
    receipt = dict(
        status="verified",
        path=str(path),
        sha256=sha(path),
        durable_path=str(destination),
        durable_sha256=sha(destination),
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256={"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA},
    )
    journal["molecular_reporting"] = receipt
    journal["status"] = "completed"
    atomic_json(root / "campaign_status.json", journal)
    atomic_json(durable / "campaign_status.json", journal)
    return receipt


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
    query_mode="source_annotation",
    unsupported="raise",
):
    """Query qualified source annotations without asserting supported biology."""
    check_dependency()
    task, definition = model_contract["task"], model_contract["task"]["scientific_definition"]
    if (
        query_mode != "source_annotation"
        or quantity_id != QUANTITY
        or task.get("task_id") != TASK
        or task.get("endpoint") != "AC50"
        or definition.get("query_mode") != "source_annotation"
        or definition.get("implementation_sha256") != sha(__file__)
        or definition.get("implementation_dependencies_sha256")
        != {"peptide_modeling_mhc10_native_bridge.py": SHARED_SHA}
        or sha(source_proof) != task["native_source_observation_coverage"]["proof_sha256"]
        or unsupported not in {"raise", "status"}
    ):
        raise ValueError(
            "Explicit source-annotation target/mode and exact model/source pins are required"
        )
    root = Path(source_proof).resolve().parents[1]
    source = rows(contained(root, task["table_path"]))
    proof = verify_source_proof(root, task, source)
    by_id = {r["record_id"]: r for r in source}
    if (
        not training_record_ids
        or len(set(training_record_ids)) != len(training_record_ids)
        or not set(training_record_ids) <= set(by_id)
    ):
        raise ValueError("Actual fitted model training members must be explicitly bound")
    fitted_rows = [by_id[r] for r in training_record_ids]
    ontology = read(contained(root, proof["ontology"]["path"]))
    X = np.asarray(chemical_X, dtype=np.float32)
    context_names = [CONTEXT_GROUP + ":" + n for n in ontology["feature_names"]]
    if (
        X.ndim != 2
        or len(X) != len(contexts)
        or not len(contexts)
        or X.shape[1] != len(chemical_names)
        or any(n.startswith(CONTEXT_GROUP + ":") for n in chemical_names)
        or list(chemical_names) + context_names != model_contract["feature_names"]
    ):
        raise ValueError("Chemical/context feature schema differs from the actual fitted model")
    output, accepted, encoded = [], [], []
    for index, query in enumerate(contexts):
        reason = None
        try:
            if set(query) != set(CONTEXT_FIELDS):
                raise ValueError(
                    "Qualified fields only; provenance IDs, outcomes and free text are forbidden"
                )
            selected = semantic_context(query)
            vector = encode_context(query, ontology)
            if (
                selected["target_resolution"] == "unresolved_source_literal"
                or selected["target_semantic_category"] == "NR13"
            ):
                raise ValueError("NR13 has no resolved qualified target annotation support")
            if object_hash(selected) not in {r["semantic_context_key"] for r in fitted_rows}:
                raise ValueError(
                    "Qualified joint target/readout context was unseen by this fitted model"
                )
            if selected["assay_cell_or_expression_host_species"] != "not_reported":
                raise ValueError("Target/system organism cannot establish assay-host species")
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc)
        if reason is not None:
            if unsupported == "raise":
                raise ValueError("Unsupported source-annotation query: " + reason)
            output.append(
                dict(
                    status="unsupported",
                    reason=reason,
                    prediction_log10_uM=None,
                    supported_biology=False,
                )
            )
            continue
        accepted.append(index)
        encoded.append(vector)
        output.append(
            dict(
                status="source_annotation_only",
                quantity_id=QUANTITY,
                supported_biology=False,
                biological_identity_unresolved=selected["target_semantic_category"]
                == "PDK1_source_literal",
                p21_marker_not_direct_HDAC_binding=selected["native_readout_family"]
                == "phenotypic_p21_induction",
            )
        )
    if accepted:
        predictions = model.predict(np.column_stack((X[accepted], np.vstack(encoded))))
        for index, value in zip(accepted, predictions, strict=True):
            if not math.isfinite(float(value)):
                raise ValueError("Source-annotation model returned a nonfinite prediction")
            output[index]["prediction_log10_uM"] = float(value)
    return output


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

    result = read(shared.artifact_path(run_dir, result_path))
    name = "refit_model" if use_refit else "model"
    if result.get("status") != "completed" or (
        use_refit and result.get("refit_status") != "completed"
    ):
        raise ValueError("A distinct completed fitted model artifact is required")
    paths = {
        k: shared.artifact_path(run_dir, result["artifacts"][k])
        for k in (name, "model_contract", "outer_split")
    }
    for key, path in paths.items():
        if sha(path) != result["artifact_hashes"][result["artifacts"][key]]["sha256"]:
            raise ValueError("Fitted model/source/evaluation contract artifact SHA differs")
    contract = read(paths["model_contract"])
    if contract.get("input_hash") != result.get("input_hash"):
        raise ValueError("Source-annotation model/input contract hash differs")
    members = (
        contract["common_record_ids"] if use_refit else read(paths["outer_split"])["train_records"]
    )
    return predict_contextual(
        joblib.load(paths[name]),
        chemical_X,
        chemical_names,
        contexts,
        contract,
        quantity_id=quantity_id,
        source_proof=source_proof,
        training_record_ids=members,
        unsupported=unsupported,
    )


def run_followup(config_path, diagnostics_only=False):
    from peptide_modeling_followup_seeds import run_followup as run_seeds
    from peptide_modeling_supervise import atomic_json

    config = read(config_path)
    training = read(config["template_config"])
    if training["training"].get("refit_full") is not True or config.get(
        "required_seeds", list(SEEDS)
    ) != list(SEEDS):
        raise ValueError("Three exact seeds and a separate all77-data refit are mandatory")
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
        receipt = prepare(args.source, args.parent, args.output)
        print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))


if __name__ == "__main__":
    main()
