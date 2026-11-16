"""Audit a pinned HC10 terminal conflict without promoting incomplete primary proof.

This module deliberately publishes candidate evidence only. A database terminal
annotation and a matching frozen graph do not independently prove all paper rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import peptide_modeling_hemolytik_bridge as shared

SCHEMA = "hemolytik_hc10_terminal_candidate_audit_v1"
SOURCE_IDS = frozenset(str(value) for value in range(2354, 2367))
PMID = "14596922"
CHEMISTRY = {
    "nter": "Free",
    "cter": "Amidation",
    "lyn_cyc": "Linear",
    "ldmix": "L",
    "non_nat": "None",
}
PRIMARY_GAP = "primary_member_and_terminal_coverage_not_independently_pinned"


def terminal_amide_candidate(sequence, source_smiles):
    """Locate the final backbone oxygen, never an arbitrary carboxylic acid.

    The independently assembled all-L acid must equal the entire source graph,
    including stereo, before a terminal atom can be mapped. Asp/Glu and Cys are
    outside this narrow source series and intentionally rejected.
    """
    from rdkit import Chem

    if not sequence or set(sequence) - set("AFGHIKLMNPQRSTVWY"):
        raise ValueError("sequence_outside_unambiguous_source_series")
    reference = Chem.MolFromSequence(sequence)
    observed = Chem.MolFromSmiles(source_smiles)
    if reference is None or observed is None:
        raise ValueError("invalid_source_graph_or_sequence")

    def canonical(molecule):
        return Chem.MolToSmiles(molecule, isomericSmiles=True)

    if canonical(reference) != canonical(observed):
        raise ValueError("source_full_graph_not_exact_all_L_sequence_acid")
    terminal = [
        atom.GetIdx()
        for atom in reference.GetAtoms()
        if atom.GetPDBResidueInfo()
        and atom.GetPDBResidueInfo().GetResidueNumber() == len(sequence)
        and atom.GetPDBResidueInfo().GetName().strip() == "OXT"
    ]
    if len(terminal) != 1:
        raise ValueError("terminal_backbone_oxygen_not_unique")
    oxygen = reference.GetAtomWithIdx(terminal[0])
    neighbors = list(oxygen.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("terminal_oxygen_not_single_connected")
    carbon = neighbors[0]
    info = carbon.GetPDBResidueInfo()
    if (
        oxygen.GetAtomicNum() != 8
        or oxygen.GetTotalNumHs() != 1
        or carbon.GetAtomicNum() != 6
        or info.GetName().strip() != "C"
        or info.GetResidueNumber() != len(sequence)
        or not any(
            neighbor.GetPDBResidueInfo()
            and neighbor.GetPDBResidueInfo().GetName().strip() == "CA"
            and neighbor.GetPDBResidueInfo().GetResidueNumber() == len(sequence)
            for neighbor in carbon.GetNeighbors()
        )
    ):
        raise ValueError("site_not_the_final_alpha_carbon_backbone_carboxyl")
    matches = observed.GetSubstructMatches(reference, useChirality=True, uniquify=False)
    source_sites = {match[terminal[0]] for match in matches}
    if len(source_sites) != 1:
        raise ValueError("source_terminal_atom_mapping_ambiguous")
    edited = Chem.RWMol(observed)
    atom = edited.GetAtomWithIdx(next(iter(source_sites)))
    atom.SetAtomicNum(7)
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)
    repaired = edited.GetMol()
    Chem.SanitizeMol(repaired)
    Chem.AssignStereochemistry(repaired, cleanIt=True, force=True)
    if repaired.GetNumAtoms() != observed.GetNumAtoms():
        raise ValueError("repair_changed_atom_inventory")
    return canonical(repaired), {
        "terminal_atom_mapping_count": 1,
        "terminal_source_atom_index": next(iter(source_sites)),
        "terminal_reference_residue": len(sequence),
        "terminal_reference_atom_name": "OXT",
        "source_full_L_sequence_acid_equal": True,
        "repair": "single_C_terminal_OH_to_NH2_no_other_graph_change",
    }


def verify_candidate(raw, source):
    """Prove graph/label consistency, retaining the independent primary gap."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    if source.get("id") not in SOURCE_IDS or source.get("pmid") != PMID:
        return None, "source_id_or_reference_outside_pinned_scope"
    if raw.get("endpoint") != "hemolysis_HC10" or raw.get("lane") != "strict_numeric":
        return None, "endpoint_or_lane_outside_scope"
    if any(source.get(key) != value for key, value in CHEMISTRY.items()):
        return None, "explicit_source_chemistry_not_supported"
    if source.get("MAP") != source.get("seq", "") + "{ct:Amid}":
        return None, "source_MAP_terminal_annotation_conflict"
    try:
        repaired, site = terminal_amide_candidate(source["seq"], source["smiles"])
    except (ValueError, KeyError) as error:
        return None, str(error)
    repaired_source = {**source, "smiles": repaired}
    proof, reason = shared.verify_observation(raw, repaired_source)
    if not proof:
        return None, reason
    if proof["metric"] != "HC10" or proof["rbc_species"] != "Human":
        return None, "metric_or_species_outside_scope"
    unit = str(raw.get("raw_unit", "")).replace("µ", "u").replace("μ", "u").lower()
    if unit != "ug/ml":
        return None, "mass_concentration_unit_outside_scope"
    if raw.get("unit") != "uM" or raw.get("relation") != "=":
        return None, "normalized_target_not_exact_uM"
    try:
        molecular_weight = Descriptors.MolWt(Chem.MolFromSmiles(repaired))
        converted = float(raw["raw_value"]) * 1000.0 / molecular_weight
        observed = float(raw["value"])
        relative_error = abs(converted / observed - 1.0)
        if (
            not all(map(math.isfinite, [converted, observed]))
            or observed <= 0
            or relative_error > 1e-12
        ):
            return None, "frozen_molar_label_not_consistent_with_declared_amide"
    except (ValueError, KeyError, ZeroDivisionError):
        return None, "invalid_molar_conversion"
    return {
        **proof,
        **site,
        "source_csv_smiles_sha256": shared.text_sha(source["smiles"]),
        "source_acid_graph_sha256": shared.text_sha(shared.canonical_graph(source["smiles"])),
        "declared_amide_graph_sha256": shared.text_sha(repaired),
        "declared_amide_graph_equal_frozen": True,
        "declared_amide_MW": molecular_weight,
        "molar_conversion_relative_error": relative_error,
        "canonical_identity_group": "CANON_SMILES:" + shared.text_sha(repaired),
        "source_sequence": source["seq"],
        "primary_reference_pmid": PMID,
        "primary_proof_status": PRIMARY_GAP,
        "admitted": False,
    }, ""


def build_bridge(
    dataset_dir, output_dir, source_file=shared.DEFAULT_SOURCE, pin_file=shared.DEFAULT_PIN
):
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if (
        output_dir.resolve() == dataset_dir.resolve()
        or dataset_dir.resolve() in output_dir.resolve().parents
    ):
        raise ValueError("audit_output_must_be_independent_of_frozen_dataset")
    sources = shared.pinned_source(source_file, pin_file)
    source_hash = shared.sha_file(dataset_dir / "manifest.json")
    gate_hash = shared.sha_file(dataset_dir / "representation_gates.tsv")
    connection = sqlite3.connect(
        (dataset_dir / "catalogue.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        records = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM records WHERE source_code='HEMOLYTIK2' "
                "AND endpoint='hemolysis_HC10' AND lane='strict_numeric'"
            )
        ]
    finally:
        connection.close()
    audit, candidates = [], []
    for raw in records:
        row_number, source = sources.get(raw["source_entry_id"].split(":")[-1], (None, None))
        proof, reason = verify_candidate(raw, source) if source else (None, "source_missing")
        audit.append(
            {
                "record_id": raw["record_id"],
                "status": "candidate_only" if proof else "rejected",
                "reason": reason or PRIMARY_GAP,
            }
        )
        if proof:
            candidates.append(
                {
                    "record_id": raw["record_id"],
                    "source_csv_row": row_number,
                    **{field: raw.get(field, "") for field in shared.RAW_MATCH_FIELDS},
                    "catalogue_raw_json_sha256": shared.text_sha(raw["raw_json"]),
                    **proof,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    shared.write_tsv(
        output_dir / "candidate_mapping.tsv",
        list(candidates[0]) if candidates else ["record_id"],
        candidates,
    )
    shared.write_tsv(output_dir / "verified_mapping.tsv", ["record_id"], [])
    shared.write_tsv(output_dir / "audit.tsv", ["record_id", "status", "reason"], audit)
    manifest = {
        "schema": SCHEMA,
        "source_manifest_sha256": source_hash,
        "source_representation_gates_sha256": gate_hash,
        "source_csv": str(Path(source_file).resolve()),
        "source_csv_sha256": shared.sha_file(source_file),
        "source_pin": str(Path(pin_file).resolve()),
        "source_pin_sha256": shared.sha_file(pin_file),
        "pinned_commit": shared.PINNED_COMMIT,
        "script_sha256": shared.sha_file(__file__),
        "candidate_mapping_sha256": shared.sha_file(output_dir / "candidate_mapping.tsv"),
        "verified_mapping_sha256": shared.sha_file(output_dir / "verified_mapping.tsv"),
        "audit_sha256": shared.sha_file(output_dir / "audit.tsv"),
        "counts": {
            "observations": len(records),
            "source_consistent_candidates": len(candidates),
            "admitted": 0,
        },
        "rbc_species": dict(Counter(item["rbc_species"] for item in candidates)),
        "independent_primary_coverage_complete": False,
        "primary_reference_url": "https://pubmed.ncbi.nlm.nih.gov/14596922/",
        "primary_boundary": (
            "Abstract supports the amidated 15-residue parent and its substitution series; "
            "it does not independently enumerate all thirteen source members, terminal states "
            "or HC10 cells. The 26-residue member especially remains unverified against "
            "primary full text."
        ),
        "policy": "candidate_only_no_gate_label_graph_or_task_changes",
    }
    if (
        shared.sha_file(dataset_dir / "manifest.json") != source_hash
        or shared.sha_file(dataset_dir / "representation_gates.tsv") != gate_hash
    ):
        raise ValueError("frozen_inputs_changed_during_read_only_audit")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_evidence_map(path, records, dataset_manifest_sha256):
    """Fail closed: candidate graph evidence alone cannot admit training rows."""
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("source_manifest_sha256") != dataset_manifest_sha256
    ):
        raise ValueError("HC10 candidate schema or frozen source mismatch")
    if (
        path.name != "verified_mapping.tsv"
        or shared.sha_file(path) != manifest["verified_mapping_sha256"]
    ):
        raise ValueError("HC10 candidate evidence is not an admission mapping")
    for file_path, expected in [
        (manifest["source_csv"], manifest["source_csv_sha256"]),
        (manifest["source_pin"], manifest["source_pin_sha256"]),
        (path.parent / "candidate_mapping.tsv", manifest["candidate_mapping_sha256"]),
        (path.parent / "audit.tsv", manifest["audit_sha256"]),
    ]:
        if shared.sha_file(file_path) != expected:
            raise ValueError("HC10 candidate evidence source hash mismatch")
    if list(shared.read_tsv(path)) or manifest.get("counts", {}).get("admitted") != 0:
        raise ValueError("HC10 primary coverage remains incomplete; admission is unsupported")
    return {}, {"enabled": True, "verified_records": 0, "applied_records": 0, "status": PRIMARY_GAP}


def enrich_species_copy(raw, evidence):
    raise ValueError("HC10 candidate-only evidence cannot enrich or admit training records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=shared.ROOT / "data/dataset_reclean_20261113_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=shared.ROOT / "data/hc10_terminal_recovery_20261114_v1"
    )
    args = parser.parse_args()
    print(json.dumps(build_bridge(args.dataset_dir, args.output_dir)["counts"]))


if __name__ == "__main__":
    main()
