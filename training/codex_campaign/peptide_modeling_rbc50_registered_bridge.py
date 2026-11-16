"""Nominal RBC50 observations given species and an explicitly limited reference graph.

Four registered S-S maps are verified; experimental sample assignments are not.
Preparation copies cached graphs, while report/inference paths do not import RDKit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import signal
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import peptide_modeling_hc50_primary_bridge as baseline
import peptide_modeling_mhc10_native_bridge as shared
from peptide_modeling_hc50_precision_report import bits32

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/peptide_modeling_hc50_primary_nodefold_followup_20261115_v2"
BASE_TASK = baseline.TASK
BASE_TABLE = BASE / "task_tables" / (BASE_TASK + ".tsv")
BASE_TABLE_SHA = "1d5d2afeae4aac319a15614b80dd0af57a467b25ee23873fb877324e2971d44a"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
PARENT_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
OUTPUT = ROOT / "data/peptide_modeling_rbc50_registered_followup_20261115_v1"
ALIAS = ROOT / "data/hemolysis_alias_structure_recovery_20261115_v1"
REGISTRY = ROOT / "data/hemolysis_registered_reference_followup_20261115_v1"
SCHEMA = "rbc50_registered_reference_species_v1"
PROOF_SCHEMA = "rbc50_registered_reference_observation_coverage_v1"
REPORT_SCHEMA = "rbc50_registered_species_molecular_diagnostics_v1"
QUANTITY = "source_reported_nominal_RBC50_given_species_and_registered_reference"
QUERY_MODE = "nominal_registered_reference_only"
SEEDS = (42, 2026, 3407)
GROUPS = shared.GROUPS
CONTEXT_GROUP = "rbc_species_context"
CONTEXT_NAMES = [CONTEXT_GROUP + ":human", CONTEXT_GROUP + ":horse"]
TIERS = ("legacy_source_reference", "primary_caps_corrected_reference", "registered_reference_only")
DEPENDENCY_PINS = {
    "peptide_modeling_hc50_primary_bridge.py": (
        "5bc218673c9cde599c24e7e66a2b9b3e11bd411254419f91b9bbbdfe97e6cb39"
    ),
    "peptide_modeling_mhc10_native_bridge.py": (
        "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
    ),
    "peptide_modeling_hc50_precision_report.py": (
        "bf83afa1ffb341ad103c723952ac34d7465d6a9646414f6170a70f47046be701"
    ),
    "peptide_modeling_control_float64_recovery.py": (
        "a7583f837bb12d3a5086b9c73f52746564872d894f7f1e5e376703a1dc515c12"
    ),
}
SOURCE_FILES = {
    "review": (
        ALIAS / "six_case_review.json",
        "3f8ed596b8c77a96d4d12778739548f57449547c3bcb71bc87a3ba18cecf1aa0",
    ),
    "frozen_six": (
        ALIAS / "frozen_six_cases.json",
        "5734ae75172f2c07960d3255227bd7ad85e304b65f35dc8da5de438e24e1c578",
    ),
    "registered": (
        REGISTRY / "registered_reference_candidates.json",
        "d82b353a67ea363a5537d69bf0b164c3b08566ad98fbfdd081fc1d6238a66b4d",
    ),
    "coherence": (
        REGISTRY / "registered_graph_coherence_checks.json",
        "728cb3bdb3345209070f1a7af273d88aee1ba3414fc904034be9886a715a6429",
    ),
    "conflicts": (
        REGISTRY / "registered_state_conflicts.json",
        "f78ce791f4b99f8c20cd17558d7e1e3802faad01653133a5cf5d0b5d61fc5631",
    ),
    "bank_matches": (
        REGISTRY / "existing_bank_reference_matches.json",
        "d6189edd11c1296b7019d783e686796f644981846f88e7d0a11499124426a4e6",
    ),
    "compounds": (
        REGISTRY / "raw/original_compound_table_matches.json",
        "4287ae75f995a3eb1ef3f48583da8c166d4f112baefbcd2f0bd4d3621c1bda07",
    ),
    "linked_rows": (
        REGISTRY / "raw/original_registry_linked_rows.json",
        "28420d59210aca7c1d0ebe18afe73356885a9a0b36f92bd3c4115e89d3ce068e",
    ),
    "dramp_targets": (
        REGISTRY / "exact_target_registry_definitions.json",
        "4a87bbfa57ea9b473577d2d8529a0dbddd20270524f503f4a9df010ff9db3e71",
    ),
    "brevinin_primary": (
        ALIAS / "raw/brevinin1h_qub.pdf",
        "23bd40074699bb61377906a5a82f6c9f1fa7f956bf0253ab41c86ac96bd9069e",
    ),
    "chac_primary": (
        ALIAS / "raw/chassatides_PMC3366795.html",
        "3630da627838ef7f99bdf90e1b4cee5bd84c07621347c1aa6460753352ad1b2d",
    ),
}
TARGETS = {
    "DRAMP32300": dict(
        record_id="OLD:PV13:2e3023a37d5a1e2edaa7d9ad",
        value="53.12",
        species="horse",
        pmid="32812694",
        dbaasp="DBAASP:16101",
        sequence="FALGAVTKVLPKLFCLITRKC",
        pairs=[[15, 21]],
        family="FAMILY:77b7f020f0e36751c4fc846b",
        graph="87bf52caabec6a25b80ae74e274a6cd88221acdda491a0f28d06783cbd1b7322",
    ),
    "DRAMP00336": dict(
        record_id="OLD:PV13:000c5469597c9cdb35104212",
        value="11.6",
        species="human",
        pmid="22467870",
        dbaasp="DBAASP:4286",
        sequence="IPCGESCVWIPCITAIAGCSCKNKVCYT",
        pairs=[[3, 19], [7, 21], [12, 26]],
        family="FAMILY:3b3378f03f195ee8c1b4245b",
        graph="494a3f814602a656b0ac5054f9fcaa8065f4c5f468276007e617fd0af1ee7389",
    ),
    "DRAMP00337": dict(
        record_id="OLD:PV13:278ae838c9210f75ca437655",
        value="25.5",
        species="human",
        pmid="22467870",
        dbaasp="DBAASP:4287",
        sequence="AIPCGESCVWIPCISTVIGCSCSNKVCYR",
        pairs=[[4, 20], [8, 22], [13, 27]],
        family="FAMILY:db1cf6104928ae33c06b0857",
        graph="8ba9d886fe18d8c5e9ed2efd3a26cb75323572a7eba8a60b4723e86e6103afe4",
    ),
    "DRAMP00338": dict(
        record_id="OLD:PV13:33bc227b010794efc5d50f24",
        value="13.3",
        species="human",
        pmid="22467870",
        dbaasp="DBAASP:4289",
        sequence="IPCGESCVWIPCISGMFGCSCKDKVCYS",
        pairs=[[3, 19], [7, 21], [12, 26]],
        family="FAMILY:c1fb29309b91698624b26097",
        graph="4d234d737a00d929b23c653f5d195f156059f5edcfc1dd67f3f6cd56ad5471c9",
    ),
}
NEW_IDS = tuple(t["record_id"] for t in TARGETS.values())
HORSE_ID = TARGETS["DRAMP32300"]["record_id"]
WARNING = (
    "Nominal source-reported RBC50 concentrations given species and a limited reference graph. "
    "The four new S-S maps are exact database registrations, not verified primary sample "
    "disulfide assignments, redox states, salt forms, microstates or purity. Reduced HEMOLYTIK2 "
    "graphs are retained as unresolved representation-provenance discrepancies, not proven "
    "separate assayed reduced samples. No head-to-tail bond or inferred S-S pairing was added. "
    "The original34 source fields, unknown methods and155uM extrapolation/time/cap correction "
    "remain unchanged. Horse occurs in one molecule only: its validation lacks horse support. "
    "The three ChaC references are a related same-paper series; exact graph/FAMILY splits do "
    "not prove homology/series isolation. No state-specific biological potency, clinical, "
    "cross-species effect or primary-sample structure claim. Holdout scores evaluate only the "
    "outer-train model; separate all38-data refits have no independent score on those records."
)
sha, read, rows, write, table, object_hash = (
    shared.sha,
    shared.read,
    shared.rows,
    shared.write,
    shared.table,
    shared.object_hash,
)
TASK = (
    "TASK_RBC50_REGISTERED_"
    + object_hash(
        dict(
            quantity=QUANTITY,
            baseline=BASE_TABLE_SHA,
            sources={k: value[1] for k, value in SOURCE_FILES.items()},
            ids=NEW_IDS,
        )
    )[:24]
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def verify_dependencies():
    require(
        all(sha(Path(__file__).with_name(name)) == pin for name, pin in DEPENDENCY_PINS.items()),
        "Pinned source/numeric helper implementation changed",
    )


def contained(root, value):
    return shared.contained(root, value)


def species_context(species):
    require(species in {"human", "horse"}, "Exactly known human/horse RBC species is required")
    return np.asarray([species == "human", species == "horse"], dtype=np.float32)


def _registered_graph_qc(candidate, raw, sequence, modification, details, target):
    """Read existing bonds only; never construct a peptide, S-S edge or reduced graph."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecule = Chem.MolFromSmiles(raw["smiles"])
    require(molecule is not None, "Registered source graph is not parseable")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    key = "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest()
    require(
        canonical == candidate["canonical_isomeric_smiles"]
        and key == candidate["structure_key"] == "CANON_SMILES:" + target["graph"],
        "Exact registered canonical graph changed",
    )
    molecule = Chem.MolFromSmiles(canonical)
    chiral = dict(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    require(
        "?" not in chiral.values() and len(chiral) == candidate["assigned_chiral_centres"],
        "Registered stereochemistry is incomplete",
    )
    require(
        rdMolDescriptors.CalcMolFormula(molecule)
        == candidate["formula"]
        == raw["molecular_formula"]
        and Chem.MolToInchiKey(molecule) == candidate["computed_InChIKey"] == raw["inchikey"]
        and Chem.GetFormalCharge(molecule) == 0,
        "Registered formula/InChIKey/charge changed",
    )
    require(
        sequence["sequence"] == candidate["sequence"] == target["sequence"],
        "Exact target sequence changed",
    )
    checks = candidate["residue_checks"]
    require(len(checks) == len(target["sequence"]), "Registered residue map changed")
    nitrogens, carbonyls, sulfur_positions = [], [], {}
    for check in checks:
        alpha = molecule.GetAtomWithIdx(check["alpha_atom_index"])
        require(
            chiral.get(alpha.GetIdx()) == check["alpha_CIP"],
            "Registered residue stereochemistry changed",
        )
        neighbors = list(alpha.GetNeighbors())
        ns = [a for a in neighbors if a.GetAtomicNum() == 7]
        cs = [
            a
            for a in neighbors
            if a.GetAtomicNum() == 6
            and any(
                b.GetOtherAtom(a).GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE
                for b in a.GetBonds()
            )
        ]
        require(len(ns) == len(cs) == 1, "Registered peptide backbone map is ambiguous")
        nitrogens.append(ns[0].GetIdx())
        carbonyls.append(cs[0].GetIdx())
        if check["residue"] == "C":
            sulfurs = [
                s
                for cb in neighbors
                if cb.GetAtomicNum() == 6 and cb.GetIdx() != cs[0].GetIdx()
                for s in cb.GetNeighbors()
                if s.GetAtomicNum() == 16
            ]
            require(len(sulfurs) == 1, "Registered cysteine sidechain map changed")
            sulfur_positions[sulfurs[0].GetIdx()] = check["position"]
    require(
        all(
            molecule.GetBondBetweenAtoms(carbonyls[i], nitrogens[i + 1]) is not None
            for i in range(len(checks) - 1)
        )
        and molecule.GetBondBetweenAtoms(carbonyls[-1], nitrogens[0]) is None,
        "A head-to-tail bond was added or backbone order changed",
    )
    require(
        not any(
            n.GetIdx() in carbonyls for n in molecule.GetAtomWithIdx(nitrogens[0]).GetNeighbors()
        ),
        "Registered N terminus is not free",
    )
    require(
        any(
            a.GetAtomicNum() == 8 and a.GetTotalNumHs() == 1
            for a in molecule.GetAtomWithIdx(carbonyls[-1]).GetNeighbors()
        ),
        "Registered C-terminal acid changed",
    )
    graph_pairs = sorted(
        sorted([sulfur_positions[b.GetBeginAtomIdx()], sulfur_positions[b.GetEndAtomIdx()]])
        for b in molecule.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == b.GetEndAtom().GetAtomicNum() == 16
    )
    mapping = {
        r["field"]: r["value"]
        for r in details
        if r["modification_id"] == modification["modification_id"]
    }
    raw_pairs = []
    for index in range(10):
        prefix = f"intrachainBonds[{index}]."
        if prefix + "type.name" not in mapping:
            continue
        require(
            mapping[prefix + "type.name"] == "DSB"
            and mapping[prefix + "chainParticipating.name"] == "SSB",
            "Registered link is not the explicit sidechain disulfide field",
        )
        raw_pairs.append(
            sorted([int(mapping[prefix + "position1"]), int(mapping[prefix + "position2"])])
        )
    require(
        graph_pairs
        == sorted(raw_pairs)
        == target["pairs"]
        == candidate["disulfide_pairs_1based_from_graph"]
        == candidate["disulfide_pairs_1based_from_raw_intrachainBonds"],
        "Registered S-S residue pairs changed",
    )
    if target["dbaasp"] == "DBAASP:4289":
        require(
            not any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and {b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()} == {8, 16}
                for b in molecule.GetBonds()
            ),
            "ChaC11A Met oxidation cannot replace native ChaC11",
        )
    return canonical, key


def source_rows():
    from rdkit import Chem

    verify_dependencies()
    require(sha(BASE_TABLE) == BASE_TABLE_SHA, "Frozen baseline34 table changed")
    source = {
        key: read(path)
        for key, (path, expected) in SOURCE_FILES.items()
        if path.suffix != ".pdf" and key != "chac_primary" and sha(path) == expected
    }
    require(
        set(source) == set(SOURCE_FILES) - {"brevinin_primary", "chac_primary"},
        "A pinned source object changed",
    )
    for path, expected in SOURCE_FILES.values():
        require(sha(path) == expected, "Primary/registration source snapshot changed")
    old = rows(BASE_TABLE)
    baseline.validate_cohort(old, verify_chemistry=True)
    values = []
    for row in old:
        values.append(
            {
                **row,
                "rbc_species": "human",
                "rbc_species_eligible": "true",
                "rbc50_quantity_id": QUANTITY,
                "rbc50_reference_mode": QUERY_MODE,
                "identity_tier": "primary_caps_corrected_reference"
                if row["record_id"] == baseline.NEW_ID
                else "legacy_source_reference",
                "source_stratum": "primary_corrected_DRAMP32143"
                if row["record_id"] == baseline.NEW_ID
                else "legacy_HC50_unknown_method",
                "registered_disulfide_maps_verified": "not_reassessed_in_baseline",
                "primary_sample_disulfide_assignment_verified": "not_reassessed_in_baseline",
                "primary_sample_graph_verified": "not_reassessed_in_baseline",
                "sample_redox_state": "not_reassessed_in_baseline",
                "reported_exposure_h": "",
                "registered_reference_id": "",
            }
        )
    registered = {r["DRAMP_ID"]: r for r in source["registered"]}
    review = {r["DRAMP_ID"]: r for r in source["review"]}
    frozen = {r["record_id"]: r for r in source["frozen_six"]}
    conflicts = {r["DRAMP_ID"]: r for r in source["conflicts"]}
    coherence = {r["DRAMP_ID"]: r for r in source["coherence"]}
    linked = source["linked_rows"]
    new_proofs = []
    old_keys, old_groups = (
        {r["canonical_identity_group"] for r in old},
        {r["split_group"] for r in old},
    )
    for dramp, target in TARGETS.items():
        candidate, primary, raw = registered[dramp], review[dramp], frozen[target["record_id"]]
        require(
            str(primary["original_value"]) == str(raw["value"]) == target["value"]
            and float(primary["primary_value"]) == float(target["value"])
            and raw["relation"] == primary["primary_relation"] == "="
            and raw["unit"] == primary["primary_unit"] == "uM",
            "Primary scalar/relation/unit differs from the original observation",
        )
        require(
            primary["primary_species"]
            == ("Equus caballus" if target["species"] == "horse" else "human_type_A_erythrocytes"),
            "Primary assay species must not be inferred from peptide origin",
        )
        require(
            candidate["DBAASP_ID"] == target["dbaasp"]
            and coherence[dramp]["existing_family_groups"] == [target["family"]],
            "Registered identifier/family changed",
        )
        compound = [r for r in source["compounds"] if r["compound_id"] == candidate["compound_id"]]
        sequence = [
            r for r in linked["03_sequences.csv"] if r["compound_id"] == candidate["compound_id"]
        ]
        modification = [
            r
            for r in linked["04_modifications.csv"]
            if r["compound_id"] == candidate["compound_id"]
        ]
        forms = [
            r for r in linked["02_source_forms.csv"] if r["compound_id"] == candidate["compound_id"]
        ]
        require(
            len(compound) == len(sequence) == len(modification) == len(forms) == 1,
            "Exact raw compound/sequence/modification/registry form is not unique",
        )
        require(
            sequence[0]["source_entry_id"] == target["dbaasp"]
            and sequence[0]["sequence_id"] == modification[0]["sequence_id"]
            and sequence[0]["substance_id"] == forms[0]["substance_id"],
            "Raw registry identifiers do not join",
        )
        graph, key = _registered_graph_qc(
            candidate,
            compound[0],
            sequence[0],
            modification[0],
            linked["04b_modification_details.csv"],
            target,
        )
        alternative = conflicts[dramp]
        reduced = Chem.MolToSmiles(
            Chem.MolFromSmiles(alternative["alternative_source_graph_preserved_verbatim"]),
            canonical=True,
            isomericSmiles=True,
        )
        reduced_key = "CANON_SMILES:" + hashlib.sha256(reduced.encode()).hexdigest()
        require(
            key not in old_keys
            and reduced_key not in old_keys
            and target["family"] not in old_groups,
            "New reference or reduced alternative overlaps the frozen baseline identity",
        )
        row = dict(
            record_id=target["record_id"],
            merged_record_ids=target["record_id"],
            origin="OLD",
            compound_id=candidate["compound_id"],
            target=repr(math.log10(float(target["value"]))),
            value=target["value"],
            unit="uM",
            target_transform="log10",
            inverse_transform="10**prediction",
            sequence=target["sequence"],
            sequence_kind="registered_reference_sequence",
            sequence_eligible="false",
            structure_eligible="true",
            smiles=graph,
            canonical_smiles=graph,
            canonical_identity_group=key,
            structure_smiles_sha256=key.split(":", 1)[1],
            split_group=target["family"],
            canonical_quantity_id=QUANTITY,
            canonical_species=target["species"],
            canonical_material="erythrocytes",
            rbc_species=target["species"],
            rbc_species_eligible="true",
            rbc50_quantity_id=QUANTITY,
            rbc50_reference_mode=QUERY_MODE,
            identity_tier="registered_reference_only",
            registered_reference_id=target["dbaasp"],
            reported_RBC_material=primary["primary_species"],
            registered_disulfide_maps_verified="true",
            primary_sample_disulfide_assignment_verified="false",
            primary_sample_graph_verified="false",
            sample_redox_state="not_independently_verified",
            reference_backbone="linear",
            head_to_tail_bond="false",
            registered_SS_pairs=json.dumps(target["pairs"], separators=(",", ":")),
            parallel_reduced_reference_key=reduced_key,
            parallel_reduced_source_id=alternative["alternative_source_ID"],
            parallel_representation_discrepancy="unresolved_provenance_not_two_verified_assay_states",
            experimental_redox_contrast_verified="false",
            representation_omission_mechanism_proven="false",
            source_original_endpoint=raw["endpoint"],
            source_entry_id=dramp,
            source_code="DRAMP_primary_plus_DBAASP_registered_reference",
            source_pmid=target["pmid"],
            source_primary_anchors=json.dumps(primary["primary_anchors"]),
            raw_original_identity_key=raw["identity_key"],
            raw_relation=raw["raw_relation"],
            raw_unit=raw["raw_unit"],
            reported_sd="not_reported",
            reported_statistic="source_reported_scalar",
            estimate_method="source_reported_RBC50_point_method_not_independently_refitted",
            extrapolation_status="within_reported_assay_range_not_direct50percent_point"
            if target["species"] == "horse"
            else "not_independently_recovered",
            exposure_evidence_status="primary_horse_RBC_method2h"
            if target["species"] == "horse"
            else "not_independently_recovered",
            reported_exposure_h="2" if target["species"] == "horse" else "",
            source_stratum="Brevinin1H_primary_horse"
            if target["species"] == "horse"
            else "ChaC7_8_11_primary_human_related_series",
            mapping_evidence_status="primary_RBC50_scalar_with_registered_reference_not_sample_graph",
            source_classification_correction_applied="true"
            if target["species"] == "horse"
            else "false",
            independently_defined_RBC50="true",
            directly_observed_half_lysis_point="not_independently_verified",
        )
        values.append(row)
        new_proofs.append(
            dict(
                record_id=target["record_id"],
                DRAMP_ID=dramp,
                raw_original=raw,
                primary_review=primary,
                registration=candidate,
                coherence=coherence[dramp],
                raw_compound=compound[0],
                raw_sequence=sequence[0],
                raw_modification=modification[0],
                raw_details=[
                    r
                    for r in linked["04b_modification_details.csv"]
                    if r["modification_id"] == modification[0]["modification_id"]
                ],
                conflicting_reference=alternative,
                reduced_reference_key=reduced_key,
                registered_reference_model_row=row,
            )
        )
    validate_cohort(values)
    return values, new_proofs


def validate_cohort(values, *, verify_chemistry=True):
    require(
        len(values) == len({r["record_id"] for r in values}) == 38,
        "Exactly38 distinct RBC50 observations are required",
    )
    baseline.validate_cohort(values[:34], verify_chemistry=verify_chemistry)
    require(
        sha(BASE_TABLE) == BASE_TABLE_SHA, "Baseline table changed"
    ) if BASE_TABLE.exists() else None
    if BASE_TABLE.exists():
        require(
            all(
                all(row[k] == old[k] for k in old)
                for row, old in zip(values[:34], rows(BASE_TABLE), strict=True)
            ),
            "Original34 source fields changed",
        )
    graph_groups = {}
    for row in values:
        species_context(row["rbc_species"])
        require(
            row["rbc50_quantity_id"] == QUANTITY
            and row["rbc50_reference_mode"] == QUERY_MODE
            and row["identity_tier"] in TIERS
            and row["unit"] == "uM"
            and row["target_transform"] == "log10",
            "Registered-reference quantity/species/tier changed",
        )
        graph = row["canonical_smiles"]
        require(
            row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest(),
            "Canonical reference graph binding changed",
        )
        require(
            graph_groups.setdefault(graph, row["split_group"]) == row["split_group"],
            "One graph crosses families",
        )
        if verify_chemistry:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(row["smiles"])
            require(
                mol is not None
                and "?"
                not in [c for _, c in Chem.FindMolChiralCenters(mol, includeUnassigned=True)]
                and Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) == graph,
                "A full assigned-stereo reference graph is required",
            )
    require(
        len(graph_groups) == len(set(graph_groups.values())) == 34
        and sum(r["rbc_species"] == "human" for r in values) == 37
        and [r["record_id"] for r in values if r["rbc_species"] == "horse"] == [HORSE_ID],
        "38records must retain34molecular groups and37human/1horse",
    )
    require(
        {r["record_id"] for r in values[34:]} == set(NEW_IDS),
        "Only the four approved original labels may be added",
    )
    for target in TARGETS.values():
        row = next(r for r in values[34:] if r["record_id"] == target["record_id"])
        expected = dict(
            value=target["value"],
            rbc_species=target["species"],
            registered_reference_id=target["dbaasp"],
            identity_tier="registered_reference_only",
            canonical_identity_group="CANON_SMILES:" + target["graph"],
            split_group=target["family"],
            registered_disulfide_maps_verified="true",
            primary_sample_disulfide_assignment_verified="false",
            primary_sample_graph_verified="false",
            sample_redox_state="not_independently_verified",
            reference_backbone="linear",
            head_to_tail_bond="false",
            experimental_redox_contrast_verified="false",
            representation_omission_mechanism_proven="false",
            directly_observed_half_lysis_point="not_independently_verified",
            estimate_method="source_reported_RBC50_point_method_not_independently_refitted",
            parallel_representation_discrepancy="unresolved_provenance_not_two_verified_assay_states",
        )
        require(
            all(row.get(k) == value for k, value in expected.items())
            and json.loads(row["registered_SS_pairs"]) == target["pairs"]
            and math.isclose(
                float(row["target"]), math.log10(float(target["value"])), rel_tol=0, abs_tol=1e-12
            ),
            "Registered reference, primary scalar or limited sample-state claim changed",
        )
    return values


def slice_features(values, parent, output):
    from peptide_modeling_features import _verify_artifacts, load_feature_route

    parent, output = Path(parent).resolve(), Path(output).resolve()
    require(
        output != parent and parent not in output.parents and output not in parent.parents,
        "A new feature namespace is required",
    )
    pointer = parent / "features/manifest.json"
    require(sha(pointer) == PARENT_SHA, "Frozen unified feature pointer changed")
    base = read(pointer)
    _verify_artifacts(base)
    source = Path(base["cache_dir"])
    graph_source = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in rows(source / "structure_source_smiles.tsv")
    }
    graph_index = {r["key"]: int(r["row_idx"]) for r in rows(source / "structure_bank_index.tsv")}
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in values}
    require(
        len(graph_map) == 34
        and all(graph_source.get(key) == graph for key, graph in graph_map.items()),
        "All34 exact registered/baseline graphs must already exist in the parent",
    )
    methods = copy.deepcopy(
        [
            m
            for m in base["representation_methods"]["methods"]
            if m["source_kind"] == "structure" and m["independent"] is True
        ]
    )
    require(
        len(methods) == len({m["algorithm_family_id"] for m in methods}) == 10,
        "Ten independent chemical methods are required",
    )
    keys = sorted(graph_map)
    configuration = dict(
        schema=SCHEMA,
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256=DEPENDENCY_PINS,
        parent_pointer_sha256=PARENT_SHA,
        parent_fingerprint=base["input_fingerprint"],
        baseline34_table_sha256=BASE_TABLE_SHA,
        source_pins={name: pin for name, (_, pin) in SOURCE_FILES.items()},
        chemical_graph_keys=keys,
        record_ids=[r["record_id"] for r in values],
        species_by_record={r["record_id"]: r["rbc_species"] for r in values},
        context_method_increment=0,
        chemical_feature_extractions=0,
        reference_tier_is_input_feature=False,
        experimental_redox_state_is_input_feature=False,
    )
    fingerprint = object_hash(configuration)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in GROUPS}
    indices = [graph_index[key] for key in keys]
    for meta in groups.values():
        require(
            meta["key_kind"] == "canonical_identity_group",
            "Chemical methods must use complete graph keys",
        )
        matrix, valid = (
            np.load(source / meta["matrix"], mmap_mode="r"),
            np.load(source / meta["valid"], mmap_mode="r"),
        )
        selected, masks = matrix[indices].copy(), valid[indices].copy()
        require(
            masks.all()
            and not np.isinf(selected).any()
            and not np.isnan(selected).all(axis=1).any(),
            "A cached chemical row is invalid",
        )
        if np.isnan(selected).any():
            require(
                meta.get("allow_partial_missing")
                and (np.isnan(selected).mean(axis=1) <= meta["max_missing_fraction"]).all(),
                "Cached missing-value policy changed",
            )
        require(selected.tobytes() == matrix[indices].tobytes(), "Copied chemical values changed")
        np.save(cache / meta["matrix"], selected, allow_pickle=False)
        np.save(cache / meta["valid"], masks, allow_pickle=False)
        meta.update(shape=[34, meta["shape"][1]], index="structure_bank_index.tsv")
    context = np.vstack([species_context(r["rbc_species"]) for r in values])
    np.save(cache / "rbc_species_context.npy", context, allow_pickle=False)
    np.save(cache / "rbc_species_context_valid.npy", np.ones(38, dtype=bool), allow_pickle=False)
    groups[CONTEXT_GROUP] = dict(
        matrix="rbc_species_context.npy",
        valid="rbc_species_context_valid.npy",
        index="rbc_species_index.tsv",
        key_kind="record_id",
        eligibility_field="rbc_species_eligible",
        shape=[38, 2],
        feature_names=["human", "horse"],
        chemical_method=False,
        source_kind="known_experimental_species_only",
    )
    table(
        cache / "structure_bank_index.tsv", [dict(key=key, row_idx=i) for i, key in enumerate(keys)]
    )
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=key, canonical_isomeric_smiles=graph_map[key]) for key in keys],
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
        schema="rbc50_registered_ten_chemical_methods_v1",
        methods=methods,
        independent_method_count=10,
        chemical_dimensions=sum(groups[name]["shape"][1] for name in GROUPS),
        chemical_groups=13,
        context_columns=2,
        context_method_increment=0,
        reference_tier_is_input_feature=False,
    )
    require(
        registry["chemical_dimensions"] == 15212, "Expected complete cached chemical schema changed"
    )
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", configuration)
    write(
        cache / "copied_vector_audit.json",
        dict(
            reused_graphs=34,
            aliases=38,
            chemical_feature_extractions=0,
            byte_identical=True,
            parent_artifacts_checked=len(base["artifact_hashes"]),
        ),
    )
    routes = {m["route"]: m["group_names"] + [CONTEXT_GROUP] for m in methods}
    routes["structure_methods_all"] = list(GROUPS) + [CONTEXT_GROUP]
    manifest = dict(
        schema=SCHEMA,
        input_fingerprint=fingerprint,
        cache_dir=str(cache),
        manifest_path=str(cache / "manifest.json"),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=38,
        structure_keys=34,
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
        require(
            loaded["valid"].all()
            and loaded["X"].shape == (38, method["dimensions"] + 2)
            and loaded["feature_names"][-2:] == CONTEXT_NAMES
            and loaded["feature_names"].count(CONTEXT_NAMES[0]) == 1
            and np.array_equal(loaded["X"][:, -2:], context),
            "Chemical/species route coverage changed",
        )
        require(
            not any(
                "tier" in name.lower() or "redox_state" in name.lower()
                for name in loaded["feature_names"]
            ),
            "Representation grade is not a model feature",
        )
    _verify_artifacts(base)
    require(sha(pointer) == PARENT_SHA, "Parent pointer changed during preparation")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        dict(
            status="verified",
            records=38,
            graphs=34,
            aliases=38,
            chemical_methods=10,
            chemical_groups=13,
            chemical_dimensions=15212,
            species_columns=2,
            chemical_feature_extractions=0,
            copied_vectors_byte_identical=True,
            parent_artifacts_unchanged=len(base["artifact_hashes"]),
            parent_pointer_unchanged=True,
            reference_grade_not_a_feature=True,
        ),
    )
    return manifest


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
        train, test = (
            np.flatnonzero(np.isin(groups, split["train_groups"])),
            np.flatnonzero(np.isin(groups, split["test_groups"])),
        )
        folds = _inner_splits(y[train], groups[train], "regression", settings)
        require(
            len(folds) == 5
            and not set(groups[train]) & set(groups[test])
            and all(len(np.unique(y[train[a]])) > 1 for a, _ in folds),
            "Actual grouped outer/inner-fivefold is unavailable",
        )
        checks.append(
            dict(
                seed=seed,
                actual_folds=5,
                split_path=path.relative_to(output).as_posix(),
                split_sha256=sha(path),
                train_record_ids=[values[i]["record_id"] for i in train],
                holdout_record_ids=[values[i]["record_id"] for i in test],
                train_groups=len(set(groups[train])),
                holdout_groups=len(set(groups[test])),
                final_refit_record_ids=[r["record_id"] for r in values],
                new_observation_evaluation_roles={
                    rid: "inner_oof" if rid in split["train_records"] else "outer_holdout"
                    for rid in NEW_IDS
                },
                folds=[
                    dict(
                        fold=fold,
                        train_record_ids=[values[train[i]]["record_id"] for i in a],
                        validation_record_ids=[values[train[i]]["record_id"] for i in b],
                    )
                    for fold, (a, b) in enumerate(folds)
                ],
            )
        )
    return dict(
        protocol="group_holdout_inner5_then_separate_all_common_refit",
        seeds=checks,
        refit_full_required=True,
        final_training_records=38,
        node_runtime_plans_required_before_production_fit=True,
    )


def coverage_descriptors():
    return [
        dict(
            original_endpoint="cytotoxicity_HC50",
            canonical_endpoint="hemolysis_HC50",
            quantity_id=QUANTITY,
            record_ids=[HORSE_ID],
            original_labelled_records=1,
            included_original_records=1,
            included_original_molecular_groups=1,
            canonical_records=38,
            canonical_molecular_groups=34,
            registered_reference_only=True,
            primary_sample_graph_verified=False,
            independent_nucleated_cytotoxicity_model=False,
            raw_cell_classification_corrected_to_RBC=True,
            all_original_records_covered=True,
            inference_alias_is_global=False,
        ),
        dict(
            original_endpoint="hemolysis_HD50",
            canonical_endpoint="hemolysis_HC50",
            quantity_id=QUANTITY,
            record_ids=sorted(rid for rid in NEW_IDS if rid != HORSE_ID),
            original_labelled_records=4,
            included_original_records=3,
            included_original_molecular_groups=3,
            canonical_records=38,
            canonical_molecular_groups=34,
            registered_reference_only=True,
            primary_sample_graph_verified=False,
            all_original_records_covered=False,
            excluded_original_record_ids=["OLD:PV13:80fda950de1f472de66b3184"],
            inference_alias_is_global=False,
        ),
    ]


def prepare(output=OUTPUT, parent=PARENT):
    output = Path(output).resolve()
    require(not (output / "task_manifest.json").exists(), "Prepared RBC50 version already exists")
    values, observations = source_rows()
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "task_tables/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_TABLE, snapshots / "baseline34.tsv")
    source_objects = {}
    for key, (path, pin) in SOURCE_FILES.items():
        destination = snapshots / (key + path.suffix)
        shutil.copy2(path, destination)
        require(sha(destination) == pin, "Source snapshot copy changed")
        source_objects[key] = dict(path=destination.relative_to(output).as_posix(), sha256=pin)
    checks = split_prechecks(values, output)
    write(output / "fivefold_prechecks.json", checks)
    write(output / "task_tables/predeclared_splits/manifest.json", checks)
    task_table = "task_tables/" + TASK + ".tsv"
    table(output / task_table, values)
    descriptors = coverage_descriptors()
    proof = dict(
        schema=PROOF_SCHEMA,
        task_id=TASK,
        quantity_id=QUANTITY,
        query_mode=QUERY_MODE,
        implementation_sha256=sha(__file__),
        source_table=dict(path=task_table, sha256=sha(output / task_table)),
        baseline34=dict(
            path="task_tables/source_snapshots/baseline34.tsv",
            sha256=BASE_TABLE_SHA,
            fields_preserved=True,
        ),
        records=38,
        molecular_groups=34,
        species_records=dict(human=37, horse=1),
        species_molecular_groups=dict(human=33, horse=1),
        source_objects=source_objects,
        observations=observations,
        original_endpoint_coverage=descriptors,
        registered_disulfide_maps_verified=True,
        registered_disulfide_map_record_ids=list(NEW_IDS),
        primary_sample_disulfide_assignment_verified=False,
        primary_sample_graph_verified=False,
        sample_redox_state="not_independently_verified",
        clinical_queries_supported=False,
        reference_tier_is_input_feature=False,
        biological_state_contrast_supported=False,
        same_parent_ChaC_series_independence_proven=False,
        all38_fit_method_uniform=False,
        original_human_only_contract_unchanged=True,
        primary_censored_RN3_not_used=True,
        secondary178_33_not_used=True,
        old_RN3_250_not_used=True,
        LHH1_old200_or_new187_2_not_used=True,
        predeclared_splits=dict(
            path="task_tables/predeclared_splits/manifest.json",
            sha256=sha(output / "task_tables/predeclared_splits/manifest.json"),
        ),
        full_refit_required_for_every_original_observation_training_claim=True,
        warning=WARNING,
    )
    proof_path = output / "task_tables/registered_reference_observation_coverage.json"
    write(proof_path, proof)
    for descriptor in descriptors:
        descriptor.update(
            proof_path=proof_path.relative_to(output).as_posix(), proof_sha256=sha(proof_path)
        )
    definition = dict(
        endpoint="hemolysis_HC50",
        quantity_id=QUANTITY,
        quantity_unit="uM",
        nominal_reference_prediction_only=True,
        mandatory_prediction_inputs=[
            "rbc_species",
            "quantity_id",
            "reference_mode",
            "identity_tier",
        ],
        species=["human", "horse"],
        identity_tiers=list(TIERS),
        context_group=CONTEXT_GROUP,
        context_columns=2,
        context_is_chemical_method=False,
        reference_tier_is_input_feature=False,
        experimental_redox_state_is_input_feature=False,
        primary_sample_graph_verified=False,
        primary_sample_disulfide_assignment_verified=False,
        registered_disulfide_maps_verified=True,
        registered_disulfide_map_record_ids=list(NEW_IDS),
        biological_state_contrast_supported=False,
        all38_fit_method_uniform=False,
        exposure_time_conditioned=False,
        horse_independent_molecular_groups=1,
        horse_effect_validated=False,
        cross_molecule_species_correction_validated=False,
        ChaC_homology_series_isolation_proven=False,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        molecular_reporting_record_count=38,
        separate_all_common_refit_required=True,
        fixed_ridge_diagnostic_precision="float64",
        fixed_ridge_diagnostic_alpha=1.0,
        implementation_module="peptide_modeling_rbc50_registered_bridge",
        implementation_sha256=sha(__file__),
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="hemolysis_HC50",
        task_kind="regression",
        status="eligible",
        training_admitted=True,
        table_path=task_table,
        n_records=38,
        n_groups=34,
        unit="uM",
        target_transform="log10",
        inverse_transform="pow10",
        requested_inner_folds=5,
        seeds=list(SEEDS),
        canonical_quantity_id=QUANTITY,
        validation_protocol="group_holdout_inner5_then_separate_all_common_refit",
        task_definition=copy.deepcopy(definition),
        scientific_definition=definition,
        registered_reference_observation_coverage=descriptors,
        scientific_warning=WARNING,
    )
    write(
        output / "task_manifest.json",
        dict(schema=SCHEMA, tasks=[task], counts=dict(declared_tasks=1, eligible_tasks=1)),
    )
    write(output / "targeted_task_manifest.json", read(output / "task_manifest.json"))
    features = slice_features(values, parent, output)
    configuration = dict(
        dataset="data/dataset_reclean_20261113_v1",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        validation_protocol=task["validation_protocol"],
        mandatory_postprocessing="registered_reference_source_coverage_species_strata_float64_controls_all38_refit",
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
    )
    write(output / "config.json", configuration)
    write(output / "recommended_training_config.json", configuration)
    write(
        output / "deployment_requirements.json",
        dict(
            dynamic_CLI_entrypoints=[
                "peptide_modeling_dispatch.py",
                "peptide_modeling_supervise.py",
                "peptide_modeling_pipeline.py",
            ],
            node_actual_inner_fold_memberships_must_be_frozen_before_fit=True,
            no_additional_twenty_hour_loop=True,
            mandatory_wrapper=(
                "peptide_modeling_rbc50_registered_bridge.py "
                "--run-followup --config <followup_config>"
            ),
        ),
    )
    verify_source_proof(output, task, rows(output / task_table))
    (output / "REPORT.md").write_text(
        "# Registered-reference nominal RBC50 cohort\n\n"
        "38 observations /34 whole-graph molecular groups;37human/1horse. "
        "Original34 fields are preserved and four primary scalar observations use "
        "exact cached registered S-S references.\n\n"
        + WARNING
        + "\n\nPreparation computed no chemical features, optimized no parameters "
        "and fitted no model. "
        "Fixed Ridge diagnostic controls will use float64 on exact cached float32 values "
        "from the first production report. Actual node fivefold memberships must be "
        "frozen before training.\n",
        encoding="utf-8",
    )
    write(output / "stage_report_no_fit_QA.json", qa_stage_no_fit(output))
    write(
        output / "delivery_receipt.json",
        dict(
            status="prepared_not_trained",
            task_id=TASK,
            records=38,
            molecules=34,
            feature_fingerprint=features["input_fingerprint"],
            implementation_sha256=sha(__file__),
            source_proof_sha256=sha(proof_path),
            files={
                p.relative_to(output).as_posix(): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and "diagnostic_history" not in p.parts
            },
        ),
    )
    return task


def verify_source_proof(root, task, values):
    verify_dependencies()
    validate_cohort(values, verify_chemistry=False)
    require(
        task.get("task_id") == TASK and task.get("canonical_quantity_id") == QUANTITY,
        "Registered-reference task identity changed",
    )
    definition = task["scientific_definition"]
    require(
        definition["quantity_id"] == QUANTITY
        and definition["implementation_sha256"] == sha(__file__)
        and definition["primary_sample_graph_verified"] is False
        and definition["primary_sample_disulfide_assignment_verified"] is False
        and definition["biological_state_contrast_supported"] is False
        and definition["reference_tier_is_input_feature"] is False
        and definition["registered_disulfide_map_record_ids"] == list(NEW_IDS)
        and definition["all38_fit_method_uniform"] is False,
        "Task definition promoted a reference to primary sample or uniform fitted truth",
    )
    declared = task.get("registered_reference_observation_coverage")
    require(
        isinstance(declared, list) and len(declared) == 2,
        "Two separated raw-endpoint coverage declarations are required",
    )
    expected = coverage_descriptors()
    for observed, scoped in zip(declared, expected, strict=True):
        require(
            all(observed.get(k) == v for k, v in scoped.items()),
            "Raw endpoint coverage scope changed",
        )
    require(
        declared[0]["proof_path"] == declared[1]["proof_path"]
        and declared[0]["proof_sha256"] == declared[1]["proof_sha256"],
        "Coverage declarations use different proofs",
    )
    path = contained(root, declared[0]["proof_path"])
    require(sha(path) == declared[0]["proof_sha256"], "Registered-reference source proof changed")
    proof = read(path)
    require(
        proof.get("schema") == PROOF_SCHEMA
        and proof.get("implementation_sha256") == sha(__file__)
        and proof.get("task_id") == TASK
        and proof.get("quantity_id") == QUANTITY
        and proof.get("query_mode") == QUERY_MODE
        and proof.get("registered_disulfide_maps_verified") is True
        and proof.get("registered_disulfide_map_record_ids") == list(NEW_IDS)
        and proof.get("primary_sample_disulfide_assignment_verified") is False
        and proof.get("primary_sample_graph_verified") is False
        and proof.get("sample_redox_state") == "not_independently_verified"
        and proof.get("biological_state_contrast_supported") is False
        and proof.get("reference_tier_is_input_feature") is False
        and proof.get("all38_fit_method_uniform") is False
        and proof.get("original_endpoint_coverage") == expected,
        "Reference/sample-state or endpoint claims were promoted",
    )
    source_table = contained(root, proof["source_table"]["path"])
    require(
        proof["source_table"]["path"] == task["table_path"]
        and sha(source_table) == proof["source_table"]["sha256"],
        "Source table is not bound to proof",
    )
    old_path = contained(root, proof["baseline34"]["path"])
    require(
        sha(old_path) == proof["baseline34"]["sha256"] == BASE_TABLE_SHA,
        "Original34 source snapshot changed",
    )
    old = rows(old_path)
    require(
        [r["record_id"] for r in old] == [r["record_id"] for r in values[:34]]
        and all(
            all(row[k] == previous[k] for k in previous)
            for row, previous in zip(values[:34], old, strict=True)
        ),
        "An original34 field was rewritten",
    )
    require(
        set(proof["source_objects"]) == set(SOURCE_FILES),
        "A primary/registry source object is missing",
    )
    native_sources = {}
    for key, (_, expected_sha) in SOURCE_FILES.items():
        witness = proof["source_objects"][key]
        actual = contained(root, witness["path"])
        require(
            sha(actual) == witness["sha256"] == expected_sha,
            "Primary/registry raw snapshot changed",
        )
        if key not in {"brevinin_primary", "chac_primary"}:
            native_sources[key] = read(actual)
    observed = proof.get("observations", [])
    require(
        len(observed) == 4 and {r["record_id"] for r in observed} == set(NEW_IDS),
        "Only four registered-reference observations are approved",
    )
    model_rows = {r["record_id"]: r for r in values}
    raw_source = {r["record_id"]: r for r in native_sources["frozen_six"]}
    registries = {r["DRAMP_ID"]: r for r in native_sources["registered"]}
    reviews = {r["DRAMP_ID"]: r for r in native_sources["review"]}
    for item in observed:
        dramp, rid = item["DRAMP_ID"], item["record_id"]
        require(
            dramp in TARGETS
            and TARGETS[dramp]["record_id"] == rid
            and item["raw_original"] == raw_source[rid]
            and item["registration"] == registries[dramp]
            and item["primary_review"] == reviews[dramp],
            "A new observation lost its exact raw lineage",
        )
        require(
            all(
                model_rows[rid].get(k) == value
                for k, value in item["registered_reference_model_row"].items()
            ),
            "Admitted row differs from its source proof",
        )
        require(
            item["registration"]["primary_sample_graph_verified"] is False
            and item["registration"]["peptide_backbone_head_to_tail_closed"] is False
            and item["registration"]["disulfide_pairs_1based_from_raw_intrachainBonds"]
            == TARGETS[dramp]["pairs"],
            "Registered topology was promoted to sample proof or head-tail chemistry",
        )
    check_path = contained(root, proof["predeclared_splits"]["path"])
    require(
        sha(check_path) == proof["predeclared_splits"]["sha256"],
        "Predeclared fivefold master changed",
    )
    return proof


def error_metrics(target, prediction):
    return shared.error_metrics(target, prediction)


def weighted_errors(predictions):
    return shared.weighted_errors(predictions)


def selected_input_consistency(source, predictions, feature_payload, model_context):
    X = np.asarray(feature_payload["X"])
    require(
        X.dtype == np.dtype("float32")
        and X.shape[0] == 38
        and feature_payload["record_ids"] == [r["record_id"] for r in source]
        and np.asarray(feature_payload["valid"]).all(),
        "Exact selected full-input float32 rows are required",
    )
    index = {r["record_id"]: i for i, r in enumerate(source)}
    by_id = {r["record_id"]: r for r in source}
    grouped = defaultdict(list)
    for row in predictions:
        old = by_id[row["record_id"]]
        grouped[(old["canonical_identity_group"], old["rbc_species"], int(row["fold"]))].append(row)
    result = []
    for (graph, species, fold), group in grouped.items():
        selected = X[[index[row["record_id"]] for row in group]]
        require(
            all(np.array_equal(selected[0].view("u1"), row.view("u1")) for row in selected),
            "One graph/species has different full input bytes",
        )
        p = [float(row["prediction"]) for row in group]
        require(all(math.isfinite(value) for value in p), "Selected prediction is nonfinite")
        close = bool(np.allclose(p, p[0], rtol=1e-7, atol=1e-8))
        ulps = None
        if not close:
            policy = model_context.get("linear_solver_policy") or {}
            require(
                model_context.get("selected_model") == "linear"
                and policy.get("solver") == "lsqr"
                and policy.get("selection_basis") == "raw_route_feature_count_only"
                and policy.get("version") == "dimension_based_linear_policy_v2"
                and policy.get("feature_count") == X.shape[1]
                and policy.get("scaler_with_mean") is False,
                "Selected numeric exception needs verified linear LSQR provenance",
            )
            positions = [bits32(value) for value in p]
            ulps = max(positions) - min(positions)
            require(
                ulps <= 32
                and all(abs(value) >= 2.0**-126 for value in p)
                and len({math.copysign(1.0, value) for value in p}) == 1,
                "Selected binary32 spread exceeds the bounded32ULP policy",
            )
        result.append(
            dict(
                canonical_identity_group=graph,
                rbc_species=species,
                fold=fold,
                source_record_ids=[row["record_id"] for row in group],
                full_input_row_sha256=hashlib.sha256(selected[0].tobytes()).hexdigest(),
                full_input_width=X.shape[1],
                raw_predictions_unchanged=True,
                byte_identical_full_inputs=True,
                mode="legacy_float_tolerance" if close else "verified_linear_binary32_32ulp",
                observed_absolute_spread=max(p) - min(p),
                observed_binary32_ulp_distance=ulps,
                max_allowed_binary32_ulps=32,
                engineering_budget_not_mathematical_BLAS_bound=True,
            )
        )
    return result


def evaluation_diagnostics(source, inner, outer, check):
    validate_cohort(source, verify_chemistry=False)
    by_id = {r["record_id"]: r for r in source}
    require(
        len(inner) + len(outer) == 38
        and {r["record_id"] for r in inner + outer} == set(by_id)
        and len({r["record_id"] for r in inner + outer}) == 38,
        "Actual evaluation must cover38 records exactly once",
    )
    require(
        not ({r["split_group"] for r in inner} & {r["split_group"] for r in outer}),
        "A molecular group crosses evaluation roles",
    )
    fold_by_id = {
        rid: fold["fold"] for fold in check["folds"] for rid in fold["validation_record_ids"]
    }
    require(
        len(check["folds"]) == 5 and {f["fold"] for f in check["folds"]} == set(range(5)),
        "Actual inner fivefold is incomplete",
    )
    reports, unsupported = {}, []
    for role, predictions, expected in (
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", outer, check["holdout_record_ids"]),
    ):
        require(
            {r["record_id"] for r in predictions} == set(expected),
            "Evaluation role differs from frozen source split",
        )
        enriched = []
        for row in predictions:
            original = by_id[row["record_id"]]
            fold = int(row["fold"])
            require(
                row["split_group"] == original["split_group"]
                and float(row["target"]) == float(original["target"])
                and math.isfinite(float(row["prediction"]))
                and (
                    fold == -1 if role == "outer_holdout" else fold == fold_by_id[row["record_id"]]
                ),
                "Saved source target/group/fold/prediction changed",
            )
            trained = (
                check["train_record_ids"]
                if role == "outer_holdout"
                else next(f["train_record_ids"] for f in check["folds"] if f["fold"] == fold)
            )
            require(
                original["split_group"] not in {by_id[r]["split_group"] for r in trained},
                "A validation molecule entered its own training fold",
            )
            species = sorted({by_id[r]["rbc_species"] for r in trained})
            unsupported_horse = original["rbc_species"] == "horse" and "horse" not in species
            item = {
                **row,
                "evaluation_role": role,
                "rbc_species": original["rbc_species"],
                "identity_tier": original["identity_tier"],
                "source_stratum": original["source_stratum"],
                "source_original_endpoint": original["source_original_endpoint"],
                "estimate_method": original["estimate_method"],
                "extrapolation_status": original["extrapolation_status"],
                "exposure_evidence_status": original["exposure_evidence_status"],
                "registered_reference_id": original["registered_reference_id"],
                "registered_disulfide_maps_verified": original[
                    "registered_disulfide_maps_verified"
                ],
                "primary_sample_disulfide_assignment_verified": original[
                    "primary_sample_disulfide_assignment_verified"
                ],
                "primary_sample_graph_verified": original["primary_sample_graph_verified"],
                "sample_redox_state": original["sample_redox_state"],
                "species_seen_in_training": species,
                "species_context_supported": original["rbc_species"] in species,
                "unsupported_horse_context": unsupported_horse,
                "biological_state_contrast_supported": False,
            }
            enriched.append(item)
            if unsupported_horse:
                unsupported.append(row["record_id"])
        strata = {
            field: {
                label: weighted_errors([r for r in enriched if r[field] == label])
                for label in sorted({r[field] for r in source})
            }
            for field in ("rbc_species", "identity_tier", "source_stratum")
        }
        reports[role] = dict(
            **weighted_errors(enriched),
            predictions=enriched,
            strata=strata,
            unsupported_horse_record_ids=[
                r["record_id"] for r in enriched if r["unsupported_horse_context"]
            ],
            score_role="model_selection" if role == "inner_oof" else "outer_train_model_evaluation",
            score_applies_to_all38_refit=False,
        )
    require(
        unsupported == [HORSE_ID],
        "The single-family horse validation must expose unseen-species support",
    )
    return dict(
        status="verified",
        records=38,
        molecules=34,
        metrics_space="log10_uM",
        evaluation_by_role=reports,
        unsupported_horse_record_ids=unsupported,
        horse_effect_validated=False,
        independently_evaluated_refit=False,
        primary_sample_graph_verified=False,
        registered_reference_prediction_only=True,
        biological_state_contrast_supported=False,
        original_observation_evaluation_roles={
            rid: next(
                role
                for role, report in reports.items()
                if any(r["record_id"] == rid for r in report["predictions"])
            )
            for rid in NEW_IDS
        },
        warning=WARNING,
    )


def fixed_ridge_controls(source, chemical_X, check):
    """Prespecified float64 diagnostic controls; no parameter search or selected-model fit."""
    from peptide_modeling_train import _cv_predictions, _estimator, _fit_model

    by_id = {r["record_id"]: i for i, r in enumerate(source)}
    train = np.asarray([by_id[r] for r in check["train_record_ids"]])
    test = np.asarray([by_id[r] for r in check["holdout_record_ids"]])
    local = {r: i for i, r in enumerate(check["train_record_ids"])}
    folds = [
        (
            np.asarray([local[r] for r in f["train_record_ids"]]),
            np.asarray([local[r] for r in f["validation_record_ids"]]),
        )
        for f in check["folds"]
    ]
    context = np.vstack([species_context(r["rbc_species"]) for r in source])
    y = np.asarray([float(r["target"]) for r in source], dtype=np.float64)
    settings = dict(seed=check["seed"], n_jobs=1)
    reports = {}
    for name, X in (
        ("molecule_only", np.asarray(chemical_X, dtype=np.float32)),
        ("context_only", context),
        ("molecule_plus_context", np.column_stack((chemical_X, context))),
    ):
        X = np.asarray(X, dtype=np.float64)
        estimator = _estimator(
            "linear", "regression", {"regularization": 1.0}, settings, X.shape[1]
        )
        p, fold_ids = _cv_predictions(estimator, X[train], y[train], folds, "regression", 1)
        fitted = _fit_model(estimator, X[train], y[train], 1)
        held = fitted.predict(X[test])
        inner = [
            dict(
                record_id=source[i]["record_id"],
                split_group=source[i]["split_group"],
                target=float(y[i]),
                prediction=float(v),
                fold=int(f),
            )
            for i, v, f in zip(train, p, fold_ids, strict=True)
        ]
        outer = [
            dict(
                record_id=source[i]["record_id"],
                split_group=source[i]["split_group"],
                target=float(y[i]),
                prediction=float(v),
                fold=-1,
            )
            for i, v in zip(test, held, strict=True)
        ]
        reports[name] = dict(
            model="fixed_ridge_alpha1_float64_not_Optuna_selected",
            same_frozen_splits=True,
            model_selection_repeated=False,
            **evaluation_diagnostics(source, inner, outer, check),
        )
    return reports


def instrumented_float64_controls(source, chemical_X, check, source_table_sha256, result_sha256):
    """Reuse the reviewed dtype/count instrumentation; production reporting fits18 diagnostics."""
    import sys

    import peptide_modeling_control_float64_recovery as controls
    import peptide_modeling_train as training

    context = np.vstack([species_context(r["rbc_species"]) for r in source])
    designs = dict(
        molecule_only=np.asarray(chemical_X, dtype=np.float32),
        context_only=context,
        molecule_plus_context=np.column_stack((chemical_X, context)),
    )
    plan = dict(
        kind="rbc50_registered",
        seed=check["seed"],
        source=source,
        y=np.asarray([float(r["target"]) for r in source], dtype=np.float64),
        groups=np.asarray([r["split_group"] for r in source]),
        chemical=chemical_X,
        designs=designs,
        design_proofs={
            name: dict(
                **controls.matrix_proof(X), source_record_ids=[r["record_id"] for r in source]
            )
            for name, X in designs.items()
        },
        operations=controls.native_operations(source, check),
        call=dict(check=check),
        source_table_sha256=source_table_sha256,
        result_sha256=result_sha256,
        old_control_reference=dict(
            status="new_task_float64_controls_from_outset_no_old_control_artifact"
        ),
        source_module=sys.modules[__name__],
        training_module=training,
    )
    result, evidence = controls.run_seed_controls(plan)
    require(
        evidence["diagnostic_fits"] == 18 and evidence["original_strict_guard_passed"],
        "All18 float64 fixed diagnostic fits are required",
    )
    return result, evidence


def verify_refit_contract(root, task, source, result):
    from peptide_modeling_control_float64_recovery import training_signature
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import portable_features

    require(
        result["status"] == result["refit_status"] == "completed"
        and result["deployment_ready"] is True,
        "A separate completed all38 refit is mandatory for training coverage",
    )
    paths = {name: shared.artifact_path(root, value) for name, value in result["artifacts"].items()}
    require("refit_model" in paths, "The all38 refit artifact is missing")
    for name, pin in result["artifact_hashes"].items():
        require(
            sha(shared.artifact_path(root, name)) == pin["sha256"],
            "An original selected/refit artifact changed",
        )
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    ids = [r["record_id"] for r in source]
    require(
        contract["task"] == inputs["task"] == task
        and contract["common_record_ids"] == ids
        and contract["model_fit_scope"] == "outer_train_only"
        and contract["holdout_not_used_for_selection"] is True
        and contract["optional_refit_fit_scope"] == "all_common_after_evaluation"
        and inputs["config"].get("refit_full") is True
        and inputs["implementation_sha256"] == sha(Path(root) / "scripts/peptide_modeling_train.py")
        and contract["target_unit"] == "uM",
        "Selected evaluation and all38 refit source scopes differ",
    )
    loaded = load_feature_route(source, portable_features(root), result["selected_route"])
    expected = inputs["feature_hashes"][result["selected_route"]]
    require(
        loaded["valid"].all()
        and loaded["record_ids"] == ids
        and loaded["feature_names"] == contract["feature_names"] == expected["names"]
        and loaded["feature_names"][-2:] == CONTEXT_NAMES
        and loaded["feature_names"].count(CONTEXT_NAMES[0]) == 1
        and np.array_equal(
            loaded["X"][:, -2:], np.vstack([species_context(r["rbc_species"]) for r in source])
        )
        and expected["matrix_sha256"]
        == hashlib.sha256(np.ascontiguousarray(loaded["X"]).tobytes()).hexdigest()
        and expected["mask_sha256"] == hashlib.sha256(loaded["valid"].tobytes()).hexdigest(),
        "The selected joint chemical/species matrix changed",
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
        outer_split=read(paths["outer_split"])["input_hash"],
    )
    require(
        training_signature(signature)
        == inputs["input_hash"]
        == result["input_hash"]
        == contract["input_hash"],
        "Signed selected training inputs changed",
    )
    policy = contract.get("linear_solver_policy")
    if result["selected_model"] == "linear":
        require(
            policy == inputs["linear_solver_policies"][result["selected_route"]],
            "Selected LSQR precision provenance changed",
        )
    return (
        paths,
        contract,
        loaded,
        dict(selected_model=result["selected_model"], linear_solver_policy=policy),
    )


def write_diagnostics(config_path, journal=None):
    from peptide_modeling_summarize import verified_predictions
    from peptide_modeling_supervise import atomic_json

    cfg = read(config_path)
    root, durable = Path(cfg["output_root"]), Path(cfg["durable_root"])
    journal = journal or read(root / "campaign_status.json")
    require(
        len(journal["rounds"]) == 3
        and {entry["seed"] for entry in journal["rounds"]} == set(SEEDS),
        "All three original source seeds are required",
    )
    reports = []
    for entry in journal["rounds"]:
        require(
            entry["status"] == "completed"
            and not entry["audit"].get("missing_completed_task_ids")
            and entry["audit"].get("method_model_coverage_met"),
            "A selected seed lacks its task/search coverage",
        )
        run = Path(entry["run_dir"])
        task = read(run / "task_manifest.json")["tasks"][0]
        source = rows(run / task["table_path"])
        proof = verify_source_proof(run, task, source)
        result_path = run / "results/full" / (TASK + "_common.json")
        result = read(result_path)
        verification_path = run / "verification.json"
        verification = read(verification_path)
        require(
            sha(verification_path) == entry["audit"]["verification_sha256"]
            and verification["status"] == "verified"
            and verification["source_task_manifest_sha256"] == sha(run / "task_manifest.json"),
            "Selected seed verification does not bind its source manifest",
        )
        verified_predictions(run, run, result, task)
        paths, contract, features, numeric_context = verify_refit_contract(
            run, task, source, result
        )
        check = next(
            c
            for c in read(contained(run, proof["predeclared_splits"]["path"]))["seeds"]
            if c["seed"] == entry["seed"]
        )
        inner, outer = rows(paths["oof"]), rows(paths["holdout"])
        report = evaluation_diagnostics(source, inner, outer, check)
        report["selected_prediction_consistency"] = dict(
            inner_oof=selected_input_consistency(source, inner, features, numeric_context),
            outer_holdout=selected_input_consistency(source, outer, features, numeric_context),
        )
        for role, stored in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            for key in ("mae", "rmse"):
                require(
                    math.isclose(
                        report["evaluation_by_role"][role]["record_weighted"][key],
                        result[stored][key],
                        rel_tol=1e-9,
                        abs_tol=1e-10,
                    ),
                    "Selected metrics disagree with unchanged raw predictions",
                )
        controls, precision = instrumented_float64_controls(
            source, features["X"][:, :-2], check, sha(run / task["table_path"]), sha(result_path)
        )
        report["fixed_ridge_controls"] = controls
        report["fixed_ridge_control_precision"] = precision
        report.update(
            seed=entry["seed"],
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            source_table_sha256=sha(run / task["table_path"]),
            result_sha256=sha(result_path),
            oof_sha256=sha(paths["oof"]),
            holdout_sha256=sha(paths["holdout"]),
            model_sha256=sha(paths["model"]),
            source_proof_sha256=task["registered_reference_observation_coverage"][0][
                "proof_sha256"
            ],
            separate_all_common_refit=dict(
                status="verified",
                records=38,
                molecular_groups=34,
                training_record_ids=contract["common_record_ids"],
                new_original_record_ids=list(NEW_IDS),
                all_four_original_observations_in_training=True,
                model_path=str(paths["refit_model"]),
                model_sha256=sha(paths["refit_model"]),
                model_contract_sha256=sha(paths["model_contract"]),
                fit_scope="all_common_after_evaluation",
                independently_evaluated_on_these38=False,
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
        original_endpoint_coverage=coverage_descriptors(),
        original_human_only_contract_changed=False,
        registered_disulfide_maps_verified=True,
        registered_disulfide_map_record_ids=list(NEW_IDS),
        primary_sample_graph_verified=False,
        primary_sample_disulfide_assignment_verified=False,
        sample_redox_state="not_independently_verified",
        biological_state_contrast_supported=False,
        horse_effect_validated=False,
        all38_fit_method_uniform=False,
        selected_model_selection_repeated=False,
        fixed_diagnostic_precision="float64",
        fixed_diagnostic_fits=54,
        warning=WARNING,
    )
    path = root / "rbc50_registered_molecular_diagnostics.json"
    atomic_json(path, payload)
    durable.mkdir(parents=True, exist_ok=True)
    destination = durable / path.name
    shutil.copy2(path, destination)
    require(sha(destination) == sha(path), "Durable RBC50 report differs")
    receipt = dict(
        status="verified",
        path=str(path),
        sha256=sha(path),
        durable_path=str(destination),
        durable_sha256=sha(destination),
        implementation_sha256=sha(__file__),
        selected_model_fits_during_reporting=0,
        fixed_diagnostic_fits=54,
        implementation_dependencies_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in (
                "peptide_modeling_hc50_primary_bridge.py",
                "peptide_modeling_mhc10_native_bridge.py",
                "peptide_modeling_hc50_precision_report.py",
                "peptide_modeling_control_float64_recovery.py",
            )
        },
    )
    journal["molecular_reporting"], journal["status"] = receipt, "completed"
    atomic_json(root / "campaign_status.json", journal)
    atomic_json(durable / "campaign_status.json", journal)
    return receipt


def predict_contextual(
    model,
    chemical_X,
    chemical_names,
    species,
    tiers,
    model_contract,
    *,
    quantity_id,
    reference_mode,
    source_proof,
    query_kind="nominal_observation",
    allow_unsupported_diagnostic=False,
):
    task = model_contract["task"]
    definition = task["scientific_definition"]
    descriptors = task["registered_reference_observation_coverage"]
    require(
        quantity_id == QUANTITY
        and reference_mode == QUERY_MODE
        and query_kind == "nominal_observation"
        and task["task_id"] == TASK
        and definition["implementation_sha256"] == sha(__file__)
        and definition["primary_sample_graph_verified"] is False
        and definition["biological_state_contrast_supported"] is False,
        "Only nominal reference observations are supported; "
        "no primary-sample or biological-state contrast query",
    )
    require(
        sha(source_proof) == descriptors[0]["proof_sha256"] == descriptors[1]["proof_sha256"],
        "Registered reference inference proof changed",
    )
    proof = read(source_proof)
    require(
        proof["quantity_id"] == QUANTITY
        and proof["primary_sample_graph_verified"] is False
        and proof["sample_redox_state"] == "not_independently_verified",
        "Sample-state inference is outside this reference contract",
    )
    X = np.asarray(chemical_X, dtype=np.float32)
    require(
        X.ndim == 2
        and len(species) == len(tiers) == len(X)
        and len(X) > 0
        and len(chemical_names) == X.shape[1]
        and list(chemical_names) + CONTEXT_NAMES == model_contract["feature_names"]
        and not any(name.startswith(CONTEXT_GROUP + ":") for name in chemical_names),
        "Explicit species/tier and exact fitted chemical schema are required",
    )
    require(
        not np.isinf(X).any() and not np.isnan(X).all(axis=1).any(),
        "Chemical query features are missing or invalid",
    )
    require(
        all(tier in TIERS for tier in tiers),
        "Experimental-sample or redox-specific identity tiers are not supported",
    )
    require(
        allow_unsupported_diagnostic or "horse" not in species,
        "Horse lacks cross-molecule support; diagnostic-only numeric output must be explicit",
    )
    context = np.vstack([species_context(s) for s in species])
    prediction = np.asarray(model.predict(np.column_stack((X, context))))
    require(np.isfinite(prediction).all(), "Reference prediction is nonfinite")
    return dict(
        predictions_log10_uM=prediction,
        quantity_id=QUANTITY,
        reference_mode=QUERY_MODE,
        support=[
            dict(
                rbc_species=s,
                identity_tier=t,
                unsupported_single_registered_molecule=(s == "horse"),
                diagnostic_numeric_only=(s == "horse"),
                clinical_or_primary_sample_prediction=False,
                primary_sample_graph_verified=False,
                biological_state_contrast_supported=False,
            )
            for s, t in zip(species, tiers, strict=True)
        ],
    )


def predict_from_artifacts(
    result_path,
    chemical_X,
    chemical_names,
    species,
    tiers,
    *,
    quantity_id,
    reference_mode,
    source_proof,
    run_dir,
    use_refit=True,
    query_kind="nominal_observation",
    allow_unsupported_diagnostic=False,
):
    import joblib

    result = read(result_path)
    key = "refit_model" if use_refit else "model"
    require(
        result["status"] == "completed"
        and (not use_refit or result["refit_status"] == "completed"),
        "A completed reference model is required",
    )
    paths = {
        name: shared.artifact_path(run_dir, result["artifacts"][name])
        for name in (key, "model_contract")
    }
    for name, path in paths.items():
        require(
            sha(path) == result["artifact_hashes"][result["artifacts"][name]]["sha256"],
            "Inference model/contract hash changed",
        )
    contract = read(paths["model_contract"])
    require(
        contract["input_hash"] == result["input_hash"], "Inference model and source contract differ"
    )
    return predict_contextual(
        joblib.load(paths[key]),
        chemical_X,
        chemical_names,
        species,
        tiers,
        contract,
        quantity_id=quantity_id,
        reference_mode=reference_mode,
        source_proof=source_proof,
        query_kind=query_kind,
        allow_unsupported_diagnostic=allow_unsupported_diagnostic,
    )


def qa_stage_no_fit(output):
    from peptide_modeling_campaign import stage_round, template_contract
    from peptide_modeling_pipeline import report_stage

    output = Path(output)
    before = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    with tempfile.TemporaryDirectory(prefix="rbc50_registered_stage_qa_") as scratch:
        root = Path(scratch)
        template = root / "template"
        shutil.copytree(output, template)
        cfg = read(template / "config.json")
        cfg["run_dir"] = str(template)
        write(template / "template_config.json", cfg)
        contract = template_contract(
            template, template / "template_config.json", template / "representation_methods.json"
        )
        run = root / "run"
        stage_round(
            template,
            run,
            cfg,
            42,
            root / "durable",
            contract,
            template / "representation_methods.json",
        )
        status = report_stage(run)
        require(
            status == dict(completed=0, deferred=0, failed=0),
            "No-fit report smoke executed unexpected model work",
        )
        require(
            sha(run / "task_manifest.json") == sha(output / "task_manifest.json"),
            "Actual stage_round changed the source manifest",
        )
    require(
        all(sha(output / name) == pin for name, pin in before.items()),
        "No-fit staging QA changed active prepared artifacts",
    )
    return dict(
        status="verified_no_fit",
        actual_stage_round=True,
        actual_report_stage=True,
        model_fits=0,
        searches=0,
        supervised_dynamic_CLI_requirements=[
            "peptide_modeling_dispatch.py",
            "peptide_modeling_supervise.py",
            "peptide_modeling_pipeline.py",
        ],
    )


def run_followup(config_path, diagnostics_only=False):
    from peptide_modeling_followup_seeds import run_followup as run_seeds
    from peptide_modeling_supervise import atomic_json

    cfg = read(config_path)
    root, durable = Path(cfg["output_root"]), Path(cfg["durable_root"])
    journal = read(root / "campaign_status.json") if diagnostics_only else run_seeds(config_path)
    try:
        return write_diagnostics(config_path, journal)
    except BaseException as exc:
        journal["status"] = "requirement_incomplete"
        journal["molecular_reporting"] = dict(status="failed", reason=f"{type(exc).__name__}:{exc}")
        atomic_json(root / "campaign_status.json", journal)
        atomic_json(durable / "campaign_status.json", journal)
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
        require(args.config is not None, "Followup/report mode requires --config")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal{signum}")

        signal.signal(signal.SIGTERM, interrupted)
        result = run_followup(args.config, args.diagnostics_only)
    else:
        task = prepare(args.output, args.parent)
        result = dict(
            status="prepared_not_trained", task_id=task["task_id"], records=38, molecular_groups=34
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
