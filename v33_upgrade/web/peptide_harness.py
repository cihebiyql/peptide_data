"""Peptide-Harness (v35 / OmniPanel 1.5) — Controlled Evolutionary Peptide Optimization Kernel.

Hardened Step 1a Core Module (all R1a Red-Team P0 & P1 defects resolved):
- Track-aware dynamic Endpoint Tier Classifier (strict track validation; Tier A primary only for objectives)
- CoreLockSpec (strict sequence-vs-SMILES verification -> never silently drops C-term amides/mods or locks)
- Activity preservation & cumulative initial-seed edit budgets (`functional_guided`, `developability_only`, `chemical_leaf_only`)
- Two-stage Fail-Closed DeterministicGatekeeper:
  * Stage A (Pre-Eval): derives edits independently from molecules (never trusts caller labels!), verifies `sequence_to_mol(seq) == can`, rejects parent-identical candidates, enforces RDKit hygiene, consistent SMARTS charge estimator across parent & leaves, accurate multi-site chem edit counting (`Di-N-Me` = 2, `c[C-seq-C]` = 3), core-lock integrity, and cumulative initial-seed budget + relative biophysical soft warnings
  * Stage B (Post-Eval): FAIL-CLOSED on missing required endpoints/AD similarity, strictly `mechanism_anchor=None`, no silent cross-track parent fallback, uniform frozen Geometric `Feas-gMPO` (disulfide parents score `low_toxicity` cleanly from mammalian cytotoxicity for both parent & candidate without ad-hoc inflation), hard pillar vetoes with non-noise improvement floor (`>=0.15` prob drop + HC50/CC50 non-worsening + `<=0.70` cap), per-endpoint relative AD floor (`max_tc >= min(0.40, parent_tc - 0.10)`), and scale-appropriate functional endpoint non-degradation gate
- NegativeSampleMemory enforced across ALL sequence & chemical generators
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import Descriptors

from .batch_mpo import (
    AD_PENALTY_BY_STATUS,
    _desirability_low_risk_prob,
    _desirability_safety_ic50,
    _normalize_weights,
    compute_peptide_mpo,
)
from .chem_utils import (
    estimate_mol_charge_ph74,
    parse_smiles_safe,
    sequence_to_mol,
)
from .peptide_opt import (
    _CandidateSpec,
    _TARGETED_SWAPS,
    _gen_ala_scan,
    _gen_cyclization,
    _gen_lipidation,
    _gen_n_methylation,
    _gen_point_mutations,
    _sanitize_and_canonicalize,
)
from .schemas import EndpointPrediction, MpoWeights

# ---------------------------------------------------------------------------
# Constants & SMARTS patterns
# ---------------------------------------------------------------------------

CANONICAL_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_ATOMIC_NUMS: frozenset[int] = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
VALID_TRACKS: frozenset[str] = frozenset({"dual_modality", "smiles_only"})

_PAT_DISULFIDE = Chem.MolFromSmarts("[SX2][SX2]")
_PAT_FREE_THIOL = Chem.MolFromSmarts("[SX2H1]")
_PAT_TERT_NME_AMIDE = Chem.MolFromSmarts("[NX3;!R]([CH3])C(=O)")
_PAT_LONG_LIPID_CHAIN = Chem.MolFromSmarts("[CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2][CH2]")
_MUT_LABEL_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

# Eisenberg consensus hydrophobicity scale (1982), 100-deg alpha-helix, residue-mean normalized
_EISENBERG_SCALE: dict[str, float] = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

_KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_PKA_NTERM = 8.6
_PKA_CTERM = 3.6
_PKA_POS = {"K": 10.8, "R": 12.5, "H": 6.5}
_PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}
_HYDROPHOBIC_AA: frozenset[str] = frozenset("ILVFWYM")

# Core MPO pillar endpoints (all 8 are Tier A on both tracks)
MPO_PILLAR_ENDPOINTS: tuple[str, ...] = (
    "PAMPA",
    "Caco2",
    "half_life_reg",
    "hemolysis_cls",
    "hemolysis_HC50",
    "overall_peptide_toxicity",
    "cytotoxicity_cls",
    "cytotoxicity_CC50",
)

_ENDPOINT_HIGHER_IS_BETTER: dict[str, bool] = {
    "PAMPA": True,
    "Caco2": True,
    "MDCK": True,
    "RRCK": True,
    "BBB": True,
    "half_life_reg": True,
    "half_life_cls": True,
    "F": True,
    "solubility_cls": True,
    "solubility_reg": True,
    "hemolysis_HC50": True,
    "hemolysis_MHC": True,
    "cytotoxicity_CC50": True,
    "hemolysis_cls": False,
    "overall_peptide_toxicity": False,
    "cytotoxicity_cls": False,
    "efflux_ratio": False,
    "CL_absolute": False,
    "CL_per_kg": False,
}


def default_direction_for_endpoint(endpoint: str) -> str:
    """Return canonical optimization direction ('max' or 'min') or raise ValueError if ambiguous (e.g., LogD7.4)."""
    if endpoint.startswith(("MIC_", "MBC_", "MFC_", "MBIC_", "MBEC_", "LC_")):
        return "min"
    if endpoint in _ENDPOINT_HIGHER_IS_BETTER:
        return "max" if _ENDPOINT_HIGHER_IS_BETTER[endpoint] else "min"
    raise ValueError(
        f"Endpoint '{endpoint}' has no unambiguous default optimization direction; specify 'max' or 'min' explicitly."
    )


# ---------------------------------------------------------------------------
# 1. Dynamic Track-Aware Endpoint Tier Classifier (Red Line 4 / P1-2 & P1-4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointTierInfo:
    endpoint: str
    track: str
    tier: str  # "tier_a_primary" | "tier_b_soft_only" | "advisory_macrocycle_only" | "tier_c_forbidden"
    evidence_grade: str
    uq_reliability: Optional[str]
    n_eff_clusters: int
    reason: str


def classify_endpoint_tier(
    manifest: dict[str, Any],
    endpoint: str,
    track: str = "dual_modality",
    is_macrocycle: bool = False,
) -> EndpointTierInfo:
    """Dynamically classify an endpoint for an explicit valid track directly from `manifest['endpoints']`."""
    if track not in VALID_TRACKS:
        raise ValueError(f"Unknown track '{track}'; must be one of {sorted(VALID_TRACKS)}")
    ep_map = manifest.get("endpoints") or {}
    if endpoint not in ep_map:
        raise ValueError(f"Unknown endpoint '{endpoint}'")

    ep_meta = ep_map[endpoint]
    grade = str(ep_meta.get("evidence_grade") or "rank_only")
    task_type = str(ep_meta.get("task_type") or "regression")
    if track not in ep_meta or not isinstance(ep_meta[track], dict):
        return EndpointTierInfo(
            endpoint=endpoint,
            track=track,
            tier="tier_c_forbidden",
            evidence_grade=grade,
            uq_reliability=None,
            n_eff_clusters=0,
            reason=f"Track '{track}' metadata missing for endpoint '{endpoint}'.",
        )
    tr_meta = ep_meta[track]
    ho = tr_meta.get("holdout_metrics") or {}
    conf_cov = tr_meta.get("conformal_holdout_coverage") or {}
    conf = tr_meta.get("conformal") or {}
    ho_boot = tr_meta.get("holdout_bootstrap_ci95") or {}
    uq_rel = conf_cov.get("uq_reliability")
    n_eff = int(conf.get("n_eff_clusters") or ho_boot.get("n_clusters") or 0)

    if grade == "exploratory_not_recommended":
        return EndpointTierInfo(
            endpoint=endpoint,
            track=track,
            tier="tier_c_forbidden",
            evidence_grade=grade,
            uq_reliability=uq_rel,
            n_eff_clusters=n_eff,
            reason=f"{endpoint} is graded exploratory_not_recommended and forbidden as an optimization objective.",
        )

    if endpoint == "LogD_decadiene":
        if not is_macrocycle:
            return EndpointTierInfo(
                endpoint=endpoint,
                track=track,
                tier="tier_c_forbidden",
                evidence_grade=grade,
                uq_reliability=uq_rel,
                n_eff_clusters=n_eff,
                reason="LogD_decadiene is restricted to cyclic macrocycles (out-of-scope for linear peptides).",
            )
        return EndpointTierInfo(
            endpoint=endpoint,
            track=track,
            tier="advisory_macrocycle_only",
            evidence_grade=grade,
            uq_reliability=uq_rel,
            n_eff_clusters=n_eff,
            reason="LogD_decadiene is macrocycle chameleonicity diagnostic only (n_eff_clusters=5, uq_reliability=low).",
        )

    if grade == "quantitative":
        return EndpointTierInfo(
            endpoint=endpoint,
            track=track,
            tier="tier_a_primary",
            evidence_grade=grade,
            uq_reliability=uq_rel,
            n_eff_clusters=n_eff,
            reason="Quantitative evidence grade (CI-verified); allowed as primary optimization objective.",
        )

    if task_type == "classification":
        auc = float(ho.get("roc_auc") or 0.0)
        if auc >= 0.70 and n_eff >= 25:
            return EndpointTierInfo(
                endpoint=endpoint,
                track=track,
                tier="tier_b_soft_only",
                evidence_grade=grade,
                uq_reliability=uq_rel,
                n_eff_clusters=n_eff,
                reason=f"rank_only classification (AUC={auc:.3f}, n_eff={n_eff}): report & non-degradation constraint only.",
            )
    else:
        sp = float(ho.get("spearman") or -1.0)
        r2 = float(ho.get("r2") or -9.0)
        if sp >= 0.45 and r2 >= 0.15 and n_eff >= 25 and uq_rel != "low":
            return EndpointTierInfo(
                endpoint=endpoint,
                track=track,
                tier="tier_b_soft_only",
                evidence_grade=grade,
                uq_reliability=uq_rel,
                n_eff_clusters=n_eff,
                reason=f"rank_only regression on {track} (Sp={sp:.3f}, R2={r2:.3f}, n_eff={n_eff}, uq={uq_rel}): report & non-degradation constraint only.",
            )

    return EndpointTierInfo(
        endpoint=endpoint,
        track=track,
        tier="tier_c_forbidden",
        evidence_grade=grade,
        uq_reliability=uq_rel,
        n_eff_clusters=n_eff,
        reason=(
            f"Endpoint {endpoint} on {track} fails Tier A/B quality gates "
            f"(grade={grade}, uq={uq_rel}, n_eff={n_eff}, R2={ho.get('r2')}, Sp={ho.get('spearman')}); forbidden as optimization objective."
        ),
    )


def classify_all_endpoints(
    manifest: dict[str, Any],
    track: str = "dual_modality",
    is_macrocycle: bool = False,
) -> dict[str, EndpointTierInfo]:
    """Classify every endpoint in `manifest['endpoints']` for the requested modality track."""
    ep_map = manifest.get("endpoints") or {}
    return {
        ep: classify_endpoint_tier(manifest, ep, track=track, is_macrocycle=is_macrocycle)
        for ep in ep_map
    }


def validate_optimization_objectives(
    manifest: dict[str, Any],
    objective_x: str,
    objective_y: str,
    functional_endpoint: Optional[str] = None,
    tracks: Sequence[str] = ("dual_modality",),
    is_macrocycle: bool = False,
) -> dict[str, EndpointTierInfo]:
    """Validate requested optimization objectives across ALL active tracks.

    Per Plan v2.0 Red Line 4 & R1a P1-2: ONLY Tier A (`tier_a_primary`) endpoints are allowed as
    `objective_x`, `objective_y`, or `functional_endpoint`. Tier B is report/non-degradation only.
    """
    if not tracks:
        raise ValueError("tracks must contain at least one track")
    info_map: dict[str, EndpointTierInfo] = {}
    check_list: list[tuple[str, str]] = [("objective_x", objective_x), ("objective_y", objective_y)]
    if functional_endpoint:
        check_list.append(("functional_endpoint", functional_endpoint))

    for tr in tracks:
        for role, ep in check_list:
            info = classify_endpoint_tier(manifest, ep, track=tr, is_macrocycle=is_macrocycle)
            if info.tier != "tier_a_primary":
                raise ValueError(
                    f"Invalid {role}='{ep}' on track='{tr}': requires Tier A (quantitative); got {info.tier} ({info.reason})"
                )
            info_map[f"{ep}@{tr}"] = info
            info_map[ep] = info
    return info_map


# ---------------------------------------------------------------------------
# 2. Biophysical Descriptors (Consistent SMARTS Charge for Stage A Comparisons)
# ---------------------------------------------------------------------------

def compute_biophysical_profile(sequence: Optional[str], mol: Chem.Mol) -> dict[str, Any]:
    """Compute RDKit & sequence biophysical metrics.

    IMPORTANT (R1a P1-7): `mol_charge_ph74` ALWAYS uses the RDKit SMARTS estimator (`sequence=None`)
    so parent-vs-chemical-leaf charge deltas are computed on the exact same estimator (zero jump on LL-37).
    `seq_charge_ph74` is stored separately for display when canonical sequence is present.
    """
    mw = float(Descriptors.MolWt(mol))
    tpsa = float(Descriptors.TPSA(mol))
    hbd = int(Descriptors.NumHDonors(mol))
    mol_charge = float(estimate_mol_charge_ph74(mol, None))
    seq_charge = float(estimate_mol_charge_ph74(mol, sequence)) if sequence else None
    n_disulfides = len(mol.GetSubstructMatches(_PAT_DISULFIDE))
    n_free_thiols = len(mol.GetSubstructMatches(_PAT_FREE_THIOL))
    n_lipid_chains = _count_distinct_lipid_chains(mol)
    n_nme_amides = len(mol.GetSubstructMatches(_PAT_TERT_NME_AMIDE))
    has_macrocycle = any(len(r) >= 12 for r in mol.GetRingInfo().AtomRings())

    max_hydro_patch = 0
    f_hydro: Optional[float] = None
    gravy: Optional[float] = None
    mu_h: Optional[float] = None
    pi_est: Optional[float] = None

    if sequence and all(aa in CANONICAL_AA for aa in sequence):
        L = len(sequence)
        run = 0
        for aa in sequence:
            if aa in _HYDROPHOBIC_AA:
                run += 1
                if run > max_hydro_patch:
                    max_hydro_patch = run
            else:
                run = 0
        f_hydro = sum(1 for aa in sequence if aa in _HYDROPHOBIC_AA) / float(L)
        gravy = sum(_KYTE_DOOLITTLE[aa] for aa in sequence) / float(L)
        angle_rad = math.radians(100.0)
        s_sin = sum(_EISENBERG_SCALE[aa] * math.sin(i * angle_rad) for i, aa in enumerate(sequence))
        s_cos = sum(_EISENBERG_SCALE[aa] * math.cos(i * angle_rad) for i, aa in enumerate(sequence))
        mu_h = math.hypot(s_sin, s_cos) / float(L)
        pi_est = _estimate_sequence_pi(sequence)

    return {
        "mw": round(mw, 2),
        "tpsa": round(tpsa, 2),
        "tpsa_per_mw": round(tpsa / max(mw, 1.0), 4),
        "hbd": hbd,
        "net_charge_ph74": round(mol_charge, 3),  # consistent SMARTS estimator across parent & leaves
        "seq_charge_ph74": round(seq_charge, 3) if seq_charge is not None else None,
        "n_disulfides": n_disulfides,
        "n_free_thiols": n_free_thiols,
        "n_lipid_chains": n_lipid_chains,
        "n_nme_amides": n_nme_amides,
        "has_macrocycle": has_macrocycle,
        "max_hydro_patch": max_hydro_patch,
        "f_hydro": round(f_hydro, 4) if f_hydro is not None else None,
        "gravy": round(gravy, 4) if gravy is not None else None,
        "eisenberg_mu_h": round(mu_h, 4) if mu_h is not None else None,
        "pi_est": round(pi_est, 2) if pi_est is not None else None,
    }


def _count_distinct_lipid_chains(mol: Chem.Mol) -> int:
    matches = mol.GetSubstructMatches(_PAT_LONG_LIPID_CHAIN)
    if not matches:
        return 0
    comps: list[set[int]] = []
    for m in matches:
        aset = set(int(i) for i in m)
        merged: list[set[int]] = []
        acc = aset
        for c in comps:
            if acc & c:
                acc = acc | c
            else:
                merged.append(c)
        merged.append(acc)
        comps = merged
    return len(comps)


def _estimate_sequence_pi(sequence: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        pos = 1.0 / (1.0 + 10.0 ** (mid - _PKA_NTERM))
        neg = 1.0 / (1.0 + 10.0 ** (_PKA_CTERM - mid))
        for aa in sequence:
            if aa in _PKA_POS:
                pos += 1.0 / (1.0 + 10.0 ** (mid - _PKA_POS[aa]))
            elif aa in _PKA_NEG:
                neg += 1.0 / (1.0 + 10.0 ** (_PKA_NEG[aa] - mid))
        if (pos - neg) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 3. CoreLockSpec & Strict Sequence-vs-SMILES Parity (R1a P0-3)
# ---------------------------------------------------------------------------

@dataclass
class CoreLockSpec:
    locked_positions: set[int] = field(default_factory=set)  # 1-indexed sequence positions
    lock_disulfides: bool = True
    locked_smarts: list[str] = field(default_factory=list)
    parent_sequence: Optional[str] = None
    parent_canonical_smiles: str = ""
    parent_disulfide_count: int = 0
    parent_smarts_counts: dict[str, int] = field(default_factory=dict)
    locked_atom_indices: set[int] = field(default_factory=set)
    compiled_smarts: list[tuple[str, Chem.Mol]] = field(default_factory=list)
    harness_mode: str = "developability_only"  # "functional_guided" | "developability_only" | "chemical_leaf_only"
    activity_unmodeled: bool = True
    max_total_aa_edits: int = 2
    max_total_chem_edits: int = 1
    sequence_smiles_mismatch_reason: Optional[str] = None


def resolve_core_locks(
    parent_mol: Chem.Mol,
    parent_sequence: Optional[str],
    locked_positions: Optional[Sequence[int]] = None,
    lock_disulfides: bool = True,
    locked_smarts: Optional[Sequence[str]] = None,
    functional_endpoint: Optional[str] = None,
    requested_max_aa_edits: Optional[int] = None,
    requested_max_chem_edits: Optional[int] = None,
) -> CoreLockSpec:
    """Validate and resolve immutable pharmacophore locks, mode, and cumulative seed edit budgets.

    P0-3 invariant: if `parent_sequence` is supplied but `canon(sequence_to_mol(seq)) != parent_can`
    (e.g. C-term amidation, D-AA, Aib, existing lipidation), NEVER silently drop parent chemistry or locks:
    - Raise ValueError if `locked_positions` was requested (cannot map 1-indexed positions safely).
    - Force `chemical_leaf_only` with `parent_sequence=None` so sequence edits never strip modifications!
    """
    parent_can = _sanitize_and_canonicalize(parent_mol)
    if not parent_can:
        raise ValueError("Parent molecule failed RDKit sanitization or is disconnected.")
    # P1-6: Always align locked_atom_indices on canonical-SMILES atom order so SMILES-only ops match exact indices!
    canon_mol = Chem.MolFromSmiles(parent_can)
    if canon_mol is None:
        raise ValueError("Parent canonical SMILES round-trip failed.")

    if requested_max_aa_edits is not None and int(requested_max_aa_edits) < 0:
        raise ValueError("requested_max_aa_edits must be >= 0")
    if requested_max_chem_edits is not None and int(requested_max_chem_edits) < 0:
        raise ValueError("requested_max_chem_edits must be >= 0")

    has_macro = any(len(r) >= 12 for r in canon_mol.GetRingInfo().AtomRings())
    raw_seq = parent_sequence.strip().upper() if parent_sequence else None
    seq_mismatch_reason: Optional[str] = None
    editable_seq: Optional[str] = None
    seq_mol_verified: Optional[Chem.Mol] = None

    if raw_seq and not has_macro and 3 <= len(raw_seq) <= 65 and all(aa in CANONICAL_AA for aa in raw_seq):
        smol = sequence_to_mol(raw_seq)
        scan = _sanitize_and_canonicalize(smol) if smol is not None else None
        if smol is not None and scan == parent_can:
            editable_seq = raw_seq
            seq_mol_verified = smol
        else:
            seq_mismatch_reason = (
                "Parent SMILES differs from canonical unmodified sequence_to_mol(sequence) "
                "(e.g., C-term amidation, D-AA, or conjugation); switched to chemical_leaf_only."
            )
            if locked_positions:
                raise ValueError(
                    "Cannot apply 1-indexed locked_positions when parent SMILES has non-canonical modifications "
                    "(canonical sequence SMILES != parent SMILES); supply locked_smarts instead."
                )

    user_locked: set[int] = set()
    if locked_positions:
        if editable_seq is None:
            raise ValueError("locked_positions requires a linear unmodified canonical 20-AA sequence (length 3-65).")
        L = len(editable_seq)
        for p in locked_positions:
            pi = int(p)
            if pi < 1 or pi > L:
                raise ValueError(f"locked_position {pi} out of range [1, {L}]")
            user_locked.add(pi)

    disulfide_matches = canon_mol.GetSubstructMatches(_PAT_DISULFIDE)
    parent_disulfide_count = len(disulfide_matches)
    locked_atoms: set[int] = set()

    if lock_disulfides:
        for m in disulfide_matches:
            for a_idx in m:
                locked_atoms.add(int(a_idx))

    effective_locked_positions = set(user_locked)
    if editable_seq is not None and seq_mol_verified is not None:
        if lock_disulfides and parent_disulfide_count > 0:
            for i, aa in enumerate(editable_seq, start=1):
                if aa == "C":
                    effective_locked_positions.add(i)
        # Map PDB residue-locked atoms onto canon_mol atom indices via exact isomorphism match
        iso = canon_mol.GetSubstructMatch(seq_mol_verified)
        for atom in seq_mol_verified.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None and int(info.GetResidueNumber()) in effective_locked_positions:
                idx = atom.GetIdx()
                locked_atoms.add(int(iso[idx]) if iso and idx < len(iso) else idx)

    compiled: list[tuple[str, Chem.Mol]] = []
    smarts_counts: dict[str, int] = {}
    for sm in (locked_smarts or []):
        pat = Chem.MolFromSmarts(sm)
        if pat is None:
            raise ValueError(f"Invalid locked_smarts pattern: '{sm}'")
        matches = canon_mol.GetSubstructMatches(pat)
        if not matches:
            raise ValueError(f"locked_smarts pattern '{sm}' does not match parent molecule")
        compiled.append((sm, pat))
        smarts_counts[sm] = len(matches)
        for m in matches:
            for a_idx in m:
                locked_atoms.add(int(a_idx))

    # Mode & cumulative edit caps (Red Line 1 / P0-1)
    max_chem = min(2, int(requested_max_chem_edits)) if requested_max_chem_edits is not None else 1
    if editable_seq is None:
        mode = "chemical_leaf_only"
        activity_unmodeled = functional_endpoint is None
        max_aa = 0
    elif functional_endpoint is not None or len(user_locked) > 0:
        mode = "functional_guided"
        activity_unmodeled = functional_endpoint is None
        max_aa = min(3, int(requested_max_aa_edits)) if requested_max_aa_edits is not None else 3
    else:
        mode = "developability_only"
        activity_unmodeled = True
        max_aa = min(2, int(requested_max_aa_edits)) if requested_max_aa_edits is not None else 2

    return CoreLockSpec(
        locked_positions=effective_locked_positions,
        lock_disulfides=lock_disulfides,
        locked_smarts=list(locked_smarts or []),
        parent_sequence=editable_seq,
        parent_canonical_smiles=parent_can,
        parent_disulfide_count=parent_disulfide_count,
        parent_smarts_counts=smarts_counts,
        locked_atom_indices=locked_atoms,
        compiled_smarts=compiled,
        harness_mode=mode,
        activity_unmodeled=activity_unmodeled,
        max_total_aa_edits=max_aa,
        max_total_chem_edits=max_chem,
        sequence_smiles_mismatch_reason=seq_mismatch_reason,
    )


def sequence_hamming_distance(seq_a: str, seq_b: str) -> int:
    if len(seq_a) != len(seq_b):
        return abs(len(seq_a) - len(seq_b)) + sum(a != b for a, b in zip(seq_a, seq_b))
    return sum(a != b for a, b in zip(seq_a, seq_b))


def _build_peptide_scaffold_query(ref_mol: Chem.Mol) -> Optional[tuple[Chem.Mol, Optional[int]]]:
    """Build a heavy-atom scaffold query from `ref_mol` removing ONLY C-terminal OXT (-OH),
    and record the C-terminal carbonyl carbon index in the resulting query graph.

    Genuine chemical-leaf operators (N-methylation, N-term/Lys-epsilon lipidation, head-to-tail lactam,
    Cys-Cys disulfide) preserve 100% of this heavy-atom graph AND only ever increase heavy-atom degree
    on Nitrogen (N), Sulfur (S), or the C-terminal carbonyl C (where OXT was removed).
    """
    try:
        rw = Chem.RWMol(ref_mol)
        oxt_idx: Optional[int] = None
        cterm_c_orig: Optional[int] = None
        for atom in rw.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is not None and info.GetName().strip() == "OXT":
                oxt_idx = atom.GetIdx()
                nbrs = atom.GetNeighbors()
                if nbrs:
                    cterm_c_orig = nbrs[0].GetIdx()
                break
        if oxt_idx is not None:
            rw.RemoveAtom(oxt_idx)
            if cterm_c_orig is not None and cterm_c_orig > oxt_idx:
                cterm_c_orig -= 1
        # Clear non-aromatic explicit H / charge / stereo query constraints so N-Me, S-S, amide bonds match cleanly.
        # Keep aromatic explicit Hs intact (e.g., Trp indole [nH], His imidazole [nH]) so RDKit sanitization succeeds!
        for atom in rw.GetAtoms():
            if not atom.GetIsAromatic():
                atom.SetNumExplicitHs(0)
                atom.SetNoImplicit(False)
            atom.SetFormalCharge(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        q = rw.GetMol()
        Chem.SanitizeMol(q)
        return q, cterm_c_orig
    except Exception:
        return None


def _verify_chemical_leaf_scaffold(cand_mol: Chem.Mol, ref_mol: Chem.Mol) -> tuple[bool, Optional[str]]:
    """Verify `cand_mol` is a genuine chemical-leaf derivative of `ref_mol` (blocks G->A/W, A->L/F, S->T, K->R, W->Y!)."""
    built = _build_peptide_scaffold_query(ref_mol)
    if built is None:
        return False, "CHEMICAL_LEAF_SCAFFOLD_QUERY_ERROR"
    scaffold_q, cterm_c_idx = built
    matches = cand_mol.GetSubstructMatches(scaffold_q, uniquify=False, useChirality=False, maxMatches=32)
    if not matches:
        return False, "CHEMICAL_LEAF_SUBSTRUCTURE_MISMATCH"

    # At least one scaffold embedding must preserve heavy-atom degree on ALL carbon/oxygen atoms
    # (except the single C-terminal carbonyl C whose OXT was removed). Only N (7) and S (16) may accept new bonds!
    q_atoms = list(scaffold_q.GetAtoms())
    for match in matches:
        ok_match = True
        for q_idx, m_idx in enumerate(match):
            qa = q_atoms[q_idx]
            z = qa.GetAtomicNum()
            if z in (7, 16) or (cterm_c_idx is not None and q_idx == cterm_c_idx):
                continue
            if cand_mol.GetAtomWithIdx(m_idx).GetDegree() > qa.GetDegree():
                ok_match = False
                break
        if ok_match:
            return True, None
    return False, "CHEMICAL_LEAF_SIDECHAIN_MUTATION"


def _infer_chemical_edit_count(
    parent_bio: dict[str, Any],
    cand_bio: dict[str, Any],
    cand_mol: Chem.Mol,
    mutation_label: Optional[str] = None,
) -> int:
    """Independently infer chemical modification count from molecular graph deltas (never trust caller labels!)."""
    edits = 0
    delta_nme = max(0, int(cand_bio["n_nme_amides"]) - int(parent_bio["n_nme_amides"]))
    delta_lipids = max(0, int(cand_bio["n_lipid_chains"]) - int(parent_bio["n_lipid_chains"]))
    delta_ss = max(0, int(cand_bio["n_disulfides"]) - int(parent_bio["n_disulfides"]))
    new_macro = (not bool(parent_bio["has_macrocycle"])) and bool(cand_bio["has_macrocycle"])

    edits += delta_nme
    edits += delta_lipids
    # c[C-seq-C] inserts 2 flanking Cys (MW increases by ~204 Da + new disulfide) -> count as 3 edits
    if delta_ss > 0:
        mw_jump = float(cand_bio["mw"]) - float(parent_bio["mw"])
        if mw_jump > 150.0:
            edits += 2 + delta_ss  # 2 Cys insertions + disulfide cyclization = 3
        else:
            edits += delta_ss
    elif new_macro:
        edits += 1

    # Also detect Di-N-Me or explicit multi-site labels if SMARTS missed a ring/variant edge
    if mutation_label and "Di-N-Me" in mutation_label:
        edits = max(edits, 2)
    if mutation_label and "c[C-seq-C]" in mutation_label:
        edits = max(edits, 3)

    # Any non-identical chemical leaf with 0 detected standard deltas counts as at least 1 uncharacterized chem edit
    if edits == 0:
        edits = 1
    return edits


# ---------------------------------------------------------------------------
# 4. Stage A & Stage B Fail-Closed Deterministic Gatekeeper
# ---------------------------------------------------------------------------

@dataclass
class StageAResult:
    passed: bool
    reject_codes: list[str]
    soft_warnings: list[str]
    canonical_smiles: Optional[str]
    biophysical: dict[str, Any]
    aa_edits_from_seed: int
    chem_edits_from_seed: int


@dataclass
class StageBResult:
    passed: bool
    reject_codes: list[str]
    gmpo_score: float
    pillar_scores: dict[str, float]
    ad_factor: float
    ad_overall: str
    ad_floor_failures: list[str]
    functional_delta: Optional[float]
    modality_track: str
    track_switched: bool


class DeterministicGatekeeper:
    """Two-stage FAIL-CLOSED deterministic Python gatekeeper strictly outside any LLM."""

    def __init__(
        self,
        core_lock: CoreLockSpec,
        parent_biophysical: dict[str, Any],
        parent_preds_by_track: dict[str, dict[str, EndpointPrediction]],
        objective_x: str,
        objective_y: str,
        functional_endpoint: Optional[str] = None,
        functional_direction: Optional[str] = None,
        weights: Optional[MpoWeights] = None,
        allow_hem_veto_floor: float = 0.25,
        allow_tox_veto_floor: float = 0.25,
        require_tanimoto_ad: bool = True,
    ) -> None:
        if not parent_preds_by_track:
            raise ValueError("parent_preds_by_track must contain at least one evaluated parent track")
        for tr in parent_preds_by_track:
            if tr not in VALID_TRACKS:
                raise ValueError(f"Unknown parent track '{tr}'")

        self.core_lock = core_lock
        self.parent_bio = parent_biophysical
        self.parent_preds_by_track = parent_preds_by_track
        self.objective_x = objective_x
        self.objective_y = objective_y
        self.functional_endpoint = functional_endpoint
        self.functional_direction = functional_direction or (
            default_direction_for_endpoint(functional_endpoint) if functional_endpoint else None
        )
        self.weights = weights or MpoWeights()
        self.hem_veto_floor = allow_hem_veto_floor
        self.tox_veto_floor = allow_tox_veto_floor
        self.require_tanimoto_ad = require_tanimoto_ad
        self.primary_track = "dual_modality" if "dual_modality" in parent_preds_by_track else "smiles_only"

        # Fail-closed validation of parent baseline predictions (R1a P0-1!)
        self._required_endpoints: tuple[str, ...] = tuple(
            dict.fromkeys([objective_x, objective_y, *MPO_PILLAR_ENDPOINTS, *([functional_endpoint] if functional_endpoint else [])])
        )
        for tr, pmap in parent_preds_by_track.items():
            missing = [ep for ep in self._required_endpoints if ep not in pmap or pmap[ep].value is None]
            if missing:
                raise ValueError(f"Parent predictions on track '{tr}' missing required endpoints: {missing}")
            nonfinite = [
                ep
                for ep in self._required_endpoints
                if not math.isfinite(float(pmap[ep].value))
                or (pmap[ep].raw_value is not None and not math.isfinite(float(pmap[ep].raw_value)))
                or (pmap[ep].max_tanimoto is not None and not math.isfinite(float(pmap[ep].max_tanimoto)))
            ]
            if nonfinite:
                raise ValueError(f"Parent predictions on track '{tr}' contain non-finite (NaN/Inf) values: {nonfinite}")

        # Precompute parent overall AD status per track (used to apply ad_factor=0.72 ONLY when parent started OOD)
        self._parent_ad_overall_by_track: dict[str, str] = {}
        for tr, pmap in parent_preds_by_track.items():
            statuses = [
                pmap[ep].ad_status if pmap[ep].ad_status in ("in_domain", "borderline", "ood") else "ood"
                for ep in self._required_endpoints
            ]
            self._parent_ad_overall_by_track[tr] = (
                "ood" if "ood" in statuses else ("borderline" if "borderline" in statuses else "in_domain")
            )

    def check_stage_a(
        self,
        smiles: str,
        sequence: Optional[str],
        is_chemical_leaf: bool = False,
        parent_was_chemical_leaf: bool = False,
        seen_keys: Optional[set[tuple[str, Optional[str]]]] = None,
        lineage_sequence: Optional[str] = None,
        mutation_label: Optional[str] = None,
    ) -> StageAResult:
        """Pre-Eval FAIL-CLOSED chemistry, core-lock, dedup, and independent edit-budget gate.

        R1a P0-2 invariant: NEVER trusts caller `sequence` or `is_chemical_leaf` labels!
        """
        rejects: list[str] = []
        warnings: list[str] = []

        if parent_was_chemical_leaf:
            return StageAResult(False, ["CHEMICAL_LEAF_NON_EXPANDABLE"], [], None, {}, 0, 2)

        mol = parse_smiles_safe(smiles)
        if mol is None:
            return StageAResult(False, ["RDKIT_PARSE_FAIL"], [], None, {}, 0, 1 if is_chemical_leaf else 0)

        can = _sanitize_and_canonicalize(mol)
        if can is None:
            return StageAResult(False, ["RDKIT_SANITIZE_OR_FRAGMENT_FAIL"], [], None, {}, 0, 1 if is_chemical_leaf else 0)

        mol = Chem.MolFromSmiles(can)
        if mol is None or mol.GetNumAtoms() < 2:
            return StageAResult(False, ["RDKIT_ROUNDTRIP_FAIL"], [], can, {}, 0, 1 if is_chemical_leaf else 0)

        # P1-10: Always reject parent-identical candidates even if seen_keys is None
        if can == self.core_lock.parent_canonical_smiles and sequence == self.core_lock.parent_sequence:
            rejects.append("IDENTICAL_TO_PARENT")

        if any(a.GetNumRadicalElectrons() != 0 for a in mol.GetAtoms()):
            rejects.append("RADICAL_ELECTRON")
        if any(a.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS for a in mol.GetAtoms()):
            rejects.append("DISALLOWED_ELEMENT")
        if any(abs(a.GetFormalCharge()) > 1 for a in mol.GetAtoms()) or abs(Chem.GetFormalCharge(mol)) > 12:
            rejects.append("EXTREME_FORMAL_CHARGE")

        if seen_keys is not None and (can, sequence) in seen_keys:
            rejects.append("DUPLICATE_CANDIDATE")

        # P0-2: Independently verify sequence <-> SMILES parity & derive edit counts
        aa_edits = 0
        chem_edits = 0
        seq_for_bio = sequence or lineage_sequence
        bio = compute_biophysical_profile(seq_for_bio, mol)

        if sequence is not None:
            if self.core_lock.parent_sequence is None:
                rejects.append("SEQ_EDIT_DISALLOWED_IN_CHEMICAL_LEAF_MODE")
            else:
                # Must match canonical SMILES of sequence_to_mol(sequence)! Blocks forged sequence labels!
                seq_mol = sequence_to_mol(sequence)
                seq_can = _sanitize_and_canonicalize(seq_mol) if seq_mol is not None else None
                if seq_can is None or seq_can != can:
                    rejects.append("SEQUENCE_SMILES_MISMATCH")
                else:
                    p_seq = self.core_lock.parent_sequence
                    if len(sequence) != len(p_seq):
                        rejects.append("SEQUENCE_LENGTH_CHANGED")
                    else:
                        for pos1 in self.core_lock.locked_positions:
                            if sequence[pos1 - 1] != p_seq[pos1 - 1]:
                                rejects.append(f"LOCKED_POS_VIOLATION:{pos1}")
                        aa_edits = sequence_hamming_distance(p_seq, sequence)
                        if aa_edits > self.core_lock.max_total_aa_edits:
                            rejects.append(f"AA_BUDGET_EXCEEDED:{aa_edits}>{self.core_lock.max_total_aa_edits}")
        else:
            # sequence is None -> MUST be a chemical leaf derived cleanly; never allow unlabeled arbitrary SMILES bypass!
            if not is_chemical_leaf:
                rejects.append("UNLABELED_NON_SEQUENCE_CANDIDATE")
            # Derive chem_edits independently from molecular graph deltas
            chem_edits = _infer_chemical_edit_count(self.parent_bio, bio, mol, mutation_label=mutation_label)
            if chem_edits > self.core_lock.max_total_chem_edits:
                rejects.append(f"CHEM_EDIT_BUDGET_EXCEEDED:{chem_edits}>{self.core_lock.max_total_chem_edits}")
            # If parent had a canonical sequence, verify lineage_sequence respects locked_positions & AA budget
            if self.core_lock.parent_sequence is not None:
                ref_seq = lineage_sequence or self.core_lock.parent_sequence
                if len(ref_seq) != len(self.core_lock.parent_sequence):
                    rejects.append("LINEAGE_SEQUENCE_LENGTH_CHANGED")
                else:
                    for pos1 in self.core_lock.locked_positions:
                        if ref_seq[pos1 - 1] != self.core_lock.parent_sequence[pos1 - 1]:
                            rejects.append(f"LOCKED_POS_VIOLATION:{pos1}")
                    aa_edits = sequence_hamming_distance(self.core_lock.parent_sequence, ref_seq)
                    if aa_edits > self.core_lock.max_total_aa_edits:
                        rejects.append(f"AA_BUDGET_EXCEEDED:{aa_edits}>{self.core_lock.max_total_aa_edits}")
                    # P0-3: True molecular scaffold + degree-preservation check (blocks K->R, W->Y, G->A/W, A->L/F, S->T!)
                    ref_mol = sequence_to_mol(ref_seq)
                    if ref_mol is not None:
                        if mol.GetNumHeavyAtoms() < int(0.85 * ref_mol.GetNumHeavyAtoms()):
                            rejects.append("CHEMICAL_LEAF_SCAFFOLD_LOSS")
                        sc_ok, sc_err = _verify_chemical_leaf_scaffold(mol, ref_mol)
                        if not sc_ok and sc_err:
                            rejects.append(sc_err)

        # MW hard bounds + relative drift
        p_mw = float(self.parent_bio["mw"])
        c_mw = float(bio["mw"])
        if c_mw < 200.0 or c_mw > 6500.0:
            rejects.append("MW_OUT_OF_RANGE")
        if (c_mw - p_mw) > 800.0 or c_mw < 0.50 * p_mw or c_mw > 1.80 * p_mw:
            rejects.append("MW_DRIFT_EXCEEDED")

        # Lipid ceiling (<= max(1, parent_lipid_chains); blocks multi-lipidation hacking)
        max_lipids = max(1, int(self.parent_bio["n_lipid_chains"]))
        if int(bio["n_lipid_chains"]) > max_lipids:
            rejects.append("MULTI_LIPIDATION_EXCEEDED")

        # Unpaired free thiols ceiling
        max_thiols = max(1, int(self.parent_bio["n_free_thiols"]))
        if int(bio["n_free_thiols"]) > max_thiols:
            rejects.append("UNPAIRED_THIOL_EXCEEDED")

        # Consistent SMARTS-based relative charge window (R1a P1-7: parent & candidate use exact same estimator!)
        p_chg = float(self.parent_bio["net_charge_ph74"])
        c_chg = float(bio["net_charge_ph74"])
        lo_chg = -max(5.0, abs(p_chg) + 1.0)
        hi_chg = max(6.5, abs(p_chg) + 1.0)
        if abs(c_chg - p_chg) > 4.0 or c_chg < lo_chg or c_chg > hi_chg:
            rejects.append("CHARGE_WINDOW_EXCEEDED")

        # Disulfide & SMARTS locks
        if self.core_lock.lock_disulfides and int(bio["n_disulfides"]) < self.core_lock.parent_disulfide_count:
            rejects.append("DISULFIDE_LOCK_BROKEN")
        for sm, pat in self.core_lock.compiled_smarts:
            if len(mol.GetSubstructMatches(pat)) < self.core_lock.parent_smarts_counts[sm]:
                rejects.append(f"LOCKED_SMARTS_BROKEN:{sm}")

        # Soft biophysical warnings (relative caps so native leads like LL-37 never self-reject)
        p_patch = int(self.parent_bio.get("max_hydro_patch") or 0)
        if int(bio["max_hydro_patch"]) > max(4, p_patch):
            warnings.append(f"Hydrophobic patch length {bio['max_hydro_patch']} > {max(4, p_patch)}")

        p_mu = self.parent_bio.get("eisenberg_mu_h")
        c_mu = bio.get("eisenberg_mu_h")
        if c_mu is not None:
            p_seq_chg = float(self.parent_bio.get("seq_charge_ph74") if self.parent_bio.get("seq_charge_ph74") is not None else p_chg)
            base_mu_cap = 0.55 if p_seq_chg >= 2.0 else 0.45
            eff_mu_cap = max(base_mu_cap, (float(p_mu) + 0.10) if p_mu is not None else base_mu_cap)
            if float(c_mu) > eff_mu_cap:
                warnings.append(f"Eisenberg uH {c_mu:.3f} > {eff_mu_cap:.3f}")

        p_gravy = self.parent_bio.get("gravy")
        c_gravy = bio.get("gravy")
        if c_gravy is not None:
            eff_gravy_cap = max(0.80, (float(p_gravy) + 0.25) if p_gravy is not None else 0.80)
            if float(c_gravy) > eff_gravy_cap:
                warnings.append(f"GRAVY {c_gravy:.2f} > {eff_gravy_cap:.2f}")

        c_pi = bio.get("pi_est")
        p_pi = self.parent_bio.get("pi_est")
        if c_pi is not None:
            parent_in_pi_zone = p_pi is not None and abs(float(p_pi) - 7.4) < 0.4 and abs(p_chg) < 0.8
            if abs(float(c_pi) - 7.4) < 0.4 and abs(c_chg) < 0.8 and not parent_in_pi_zone:
                warnings.append(f"Neutral pI precipitation zone (pI={c_pi:.2f}, chg={c_chg:+.2f})")

        return StageAResult(
            passed=len(rejects) == 0,
            reject_codes=rejects,
            soft_warnings=warnings,
            canonical_smiles=can,
            biophysical=bio,
            aa_edits_from_seed=aa_edits,
            chem_edits_from_seed=chem_edits,
        )

    def _compute_toxicity_pillar(
        self,
        by_ep: dict[str, EndpointPrediction],
        has_parent_disulfide: bool,
        base_tox_pillar: float,
    ) -> float:
        """Compute `low_toxicity` pillar uniformly for parent & candidate.

        For disulfide scaffolds (ToxinPred cystine-knot bias), anchor `low_toxicity` strictly on
        mammalian cytotoxicity (`cytotoxicity_cls` + `cytotoxicity_CC50`) for BOTH parent and candidate
        so there is zero ad-hoc inflation (`max(d_tox, 1-cyto)` removed per R1a P1-5).
        """
        if not has_parent_disulfide:
            return base_tox_pillar
        parts: list[tuple[float, float]] = []
        cyto_cls = by_ep.get("cytotoxicity_cls")
        if cyto_cls is not None and cyto_cls.value is not None:
            parts.append((0.60, _desirability_low_risk_prob(float(cyto_cls.value))))
        cyto_cc50 = by_ep.get("cytotoxicity_CC50")
        if cyto_cc50 is not None and cyto_cc50.value is not None and float(cyto_cc50.value) > 0:
            parts.append((0.40, _desirability_safety_ic50(float(cyto_cc50.value))))
        if not parts:
            return base_tox_pillar
        w_sum = sum(w for w, _ in parts)
        return sum(w * d for w, d in parts) / w_sum

    def check_stage_b(
        self,
        predictions: list[EndpointPrediction],
        modality_track: str,
    ) -> StageBResult:
        """Post-Eval FAIL-CLOSED Pure-ML Geometric MPO, hard pillar vetoes, relative AD floor & functional gate."""
        if modality_track not in self.parent_preds_by_track:
            raise ValueError(
                f"No track-matched parent baseline for modality_track='{modality_track}'; "
                f"available={sorted(self.parent_preds_by_track.keys())}"
            )

        rejects: list[str] = []
        by_ep = {p.endpoint: p for p in predictions}
        parent_track_preds = self.parent_preds_by_track[modality_track]
        parent_ad_overall = self._parent_ad_overall_by_track[modality_track]

        # 1. Fail-closed required endpoint presence & finiteness check (R1a P0-1 & R2 P1-6!)
        missing_eps = [ep for ep in self._required_endpoints if ep not in by_ep or by_ep[ep].value is None]
        if missing_eps:
            return StageBResult(
                passed=False,
                reject_codes=[f"MISSING_REQUIRED_ENDPOINT:{ep}" for ep in missing_eps],
                gmpo_score=0.0,
                pillar_scores={},
                ad_factor=0.0,
                ad_overall="ood",
                ad_floor_failures=[],
                functional_delta=None,
                modality_track=modality_track,
                track_switched=(modality_track != self.primary_track),
            )
        nonfinite_eps = [
            ep
            for ep in self._required_endpoints
            if not math.isfinite(float(by_ep[ep].value))
            or (by_ep[ep].raw_value is not None and not math.isfinite(float(by_ep[ep].raw_value)))
            or (by_ep[ep].max_tanimoto is not None and not math.isfinite(float(by_ep[ep].max_tanimoto)))
        ]
        if nonfinite_eps:
            return StageBResult(
                passed=False,
                reject_codes=[f"NONFINITE_ENDPOINT:{ep}" for ep in nonfinite_eps],
                gmpo_score=0.0,
                pillar_scores={},
                ad_factor=0.0,
                ad_overall="ood",
                ad_floor_failures=[],
                functional_delta=None,
                modality_track=modality_track,
                track_switched=(modality_track != self.primary_track),
            )

        # 2. Pure-ML MPO pillars (STRICTLY mechanism_anchor=None — Red Line 2!)
        mpo_raw = compute_peptide_mpo(predictions, weights=self.weights, mechanism_anchor=None)
        pillars = dict(mpo_raw["pillar_scores"])
        has_parent_disulfide = self.core_lock.parent_disulfide_count > 0
        pillars["low_toxicity"] = self._compute_toxicity_pillar(by_ep, has_parent_disulfide, float(pillars["low_toxicity"]))

        # 3. Hard pillar vetoes with non-noise bad-parent exemption (R1a P1-5 & P1-6)
        d_hem = float(pillars["low_hemolysis"])
        d_tox = float(pillars["low_toxicity"])
        hem_p = by_ep["hemolysis_cls"]
        p_hem_p = parent_track_preds["hemolysis_cls"]
        hc50_p = by_ep["hemolysis_HC50"]
        p_hc50_p = parent_track_preds["hemolysis_HC50"]

        if d_hem < self.hem_veto_floor:
            # Strict exemption: parent >= 0.45, candidate improves by >= 0.15 prob, HC50 non-worsening, AND prob <= 0.70
            hem_exempt = (
                float(p_hem_p.value) >= 0.45
                and float(hem_p.value) <= float(p_hem_p.value) - 0.15
                and float(hem_p.value) <= 0.70
                and float(hc50_p.value) >= float(p_hc50_p.value) * 0.90
            )
            if not hem_exempt:
                rejects.append(f"HEMOLYSIS_PILLAR_VETO:d_hem={d_hem:.3f}<{self.hem_veto_floor}")
        # P1-7: Relative classification deterioration hard veto (blocks prob spiking 0.12->0.90 even when HC50 stays high!)
        if float(hem_p.value) - float(p_hem_p.value) > 0.25 and float(hem_p.value) >= 0.45:
            rejects.append(f"HEMOLYSIS_CLS_SPIKE:{float(p_hem_p.value):.3f}->{float(hem_p.value):.3f}")

        tox_p = by_ep["overall_peptide_toxicity"]
        p_tox_p = parent_track_preds["overall_peptide_toxicity"]
        cyto_p = by_ep["cytotoxicity_cls"]
        p_cyto_p = parent_track_preds["cytotoxicity_cls"]
        cc50_p = by_ep["cytotoxicity_CC50"]
        p_cc50_p = parent_track_preds["cytotoxicity_CC50"]

        if float(cyto_p.value) - float(p_cyto_p.value) > 0.25 and float(cyto_p.value) >= 0.45:
            rejects.append(f"CYTOTOXICITY_CLS_SPIKE:{float(p_cyto_p.value):.3f}->{float(cyto_p.value):.3f}")

        if has_parent_disulfide:
            tox_cap = max(0.55, float(p_tox_p.value) + 0.05)
            if float(tox_p.value) > tox_cap:
                rejects.append(f"DISULFIDE_TOX_REGRESSION:{float(tox_p.value):.3f}>{tox_cap:.3f}")
            # Relative mammalian cytotoxicity veto (symmetric with hemolysis!)
            if d_tox < self.tox_veto_floor or float(cyto_p.value) >= 0.55:
                cyto_exempt = (
                    float(p_cyto_p.value) >= 0.55
                    and float(cyto_p.value) <= float(p_cyto_p.value) - 0.15
                    and float(cyto_p.value) <= 0.70
                    and float(cc50_p.value) >= float(p_cc50_p.value) * 0.90
                )
                if not cyto_exempt:
                    rejects.append(f"CYTOTOXICITY_PILLAR_VETO:d_tox={d_tox:.3f},cyto={float(cyto_p.value):.3f}")
        else:
            if float(tox_p.value) - float(p_tox_p.value) > 0.25 and float(tox_p.value) >= 0.50:
                rejects.append(f"TOXICITY_CLS_SPIKE:{float(p_tox_p.value):.3f}->{float(tox_p.value):.3f}")
            if d_tox < self.tox_veto_floor:
                tox_exempt = (
                    float(p_tox_p.value) >= 0.55
                    and float(tox_p.value) <= float(p_tox_p.value) - 0.15
                    and float(tox_p.value) <= 0.70
                    and float(cc50_p.value) >= float(p_cc50_p.value) * 0.90
                )
                if not tox_exempt:
                    rejects.append(f"TOXICITY_PILLAR_VETO:d_tox={d_tox:.3f}<{self.tox_veto_floor}")

        # 4. Fail-closed per-endpoint relative AD floor across ALL required endpoints (R1a P0-1 & P1-6)
        ad_floor_failures: list[str] = []
        ad_statuses: list[str] = []
        for ep in self._required_endpoints:
            cp = by_ep[ep]
            pp = parent_track_preds[ep]
            c_status = cp.ad_status if cp.ad_status in ("in_domain", "borderline", "ood") else "ood"
            ad_statuses.append(c_status)
            c_tc = cp.max_tanimoto
            p_tc = pp.max_tanimoto
            if c_tc is None or p_tc is None:
                if self.require_tanimoto_ad:
                    ad_floor_failures.append(f"{ep}:missing_max_tanimoto")
                elif pp.ad_status != "ood" and c_status == "ood":
                    ad_floor_failures.append(f"{ep}:drifted_to_ood")
            else:
                floor_tc = min(0.40, float(p_tc) - 0.10)
                if float(c_tc) < floor_tc - 1e-9:
                    ad_floor_failures.append(f"{ep}:tc={float(c_tc):.3f}<{floor_tc:.3f}")

        if ad_floor_failures:
            rejects.append(f"OOD_DRIFT:{';'.join(ad_floor_failures[:3])}")

        # 5. Scale-appropriate functional endpoint non-degradation gate (R1a P1-8)
        func_delta: Optional[float] = None
        if self.functional_endpoint:
            cf = by_ep[self.functional_endpoint]
            pf = parent_track_preds[self.functional_endpoint]
            c_raw = float(cf.raw_value) if cf.raw_value is not None else float(cf.value)
            p_raw = float(pf.raw_value) if pf.raw_value is not None else float(pf.value)
            func_delta = c_raw - p_raw
            tol = 0.05 if cf.kind == "classification" else 0.15
            if self.functional_direction == "min":
                if func_delta > tol:
                    rejects.append(f"FUNCTIONAL_DEGRADED:{self.functional_endpoint}({func_delta:+.3f}>+{tol:.2f})")
            else:
                if func_delta < -tol:
                    rejects.append(f"FUNCTIONAL_DEGRADED:{self.functional_endpoint}({func_delta:+.3f}<-{tol:.2f})")

        # 6. Frozen Weighted Geometric Mean MPO (R1a P1-4: ad_factor=0.72 ONLY when parent itself started OOD)
        ad_overall = "ood" if "ood" in ad_statuses else ("borderline" if "borderline" in ad_statuses else "in_domain")
        ad_factor = (
            AD_PENALTY_BY_STATUS["ood"]
            if (parent_ad_overall == "ood" and ad_overall == "ood" and self.weights.penalize_ood)
            else 1.0
        )

        norm_w = _normalize_weights(self.weights)
        log_g = 0.0
        for pillar_name, w in norm_w.items():
            if w <= 0.0:
                continue
            d_val = max(float(pillars[pillar_name]), 0.05)
            log_g += w * math.log(d_val)
        gmpo_score = round(100.0 * ad_factor * math.exp(log_g), 2)

        return StageBResult(
            passed=len(rejects) == 0,
            reject_codes=rejects,
            gmpo_score=gmpo_score,
            pillar_scores={k: round(float(v), 4) for k, v in pillars.items()},
            ad_factor=round(ad_factor, 4),
            ad_overall=ad_overall,
            ad_floor_failures=ad_floor_failures,
            functional_delta=round(func_delta, 4) if func_delta is not None else None,
            modality_track=modality_track,
            track_switched=(modality_track != self.primary_track),
        )


# ---------------------------------------------------------------------------
# 5. NegativeSampleMemory & Sequence-First / Chemical-Leaf Candidate Builders
# ---------------------------------------------------------------------------

@dataclass
class NegativeMemoryEntry:
    round_index: int
    mutation_label: str
    strategy: str
    canonical_smiles: Optional[str]
    reject_codes: list[str]
    position: Optional[int] = None
    to_aa: Optional[str] = None


_NON_FAILURE_CODES: frozenset[str] = frozenset({"DUPLICATE_CANDIDATE", "IDENTICAL_TO_PARENT"})


class NegativeSampleMemory:
    """Per-run episodic negative memory enforced across ALL generation paths (never polluted by duplicates)."""

    def __init__(self) -> None:
        self.entries: list[NegativeMemoryEntry] = []
        self.seen_keys: set[tuple[str, Optional[str]]] = set()
        self.failed_seq_edits: dict[tuple[int, str], list[str]] = {}
        self.failed_chem_labels: set[str] = set()

    def record_seen(self, canonical_smiles: str, sequence: Optional[str]) -> None:
        self.seen_keys.add((canonical_smiles, sequence))

    def record_failure(
        self,
        round_index: int,
        mutation_label: str,
        strategy: str,
        reject_codes: list[str],
        canonical_smiles: Optional[str] = None,
        sequence: Optional[str] = None,
        position: Optional[int] = None,
        to_aa: Optional[str] = None,
    ) -> None:
        # P1-3: Never pollute negative_memory or tabu sets with DUPLICATE_CANDIDATE or IDENTICAL_TO_PARENT!
        real_codes = [c for c in reject_codes if c not in _NON_FAILURE_CODES]
        if not real_codes:
            return
        self.entries.append(
            NegativeMemoryEntry(
                round_index=round_index,
                mutation_label=mutation_label,
                strategy=strategy,
                canonical_smiles=canonical_smiles,
                reject_codes=list(real_codes),
                position=position,
                to_aa=to_aa,
            )
        )
        if canonical_smiles:
            self.seen_keys.add((canonical_smiles, sequence))
        if position is not None and to_aa is not None:
            self.failed_seq_edits[(position, to_aa)] = list(real_codes)
        else:
            m = _MUT_LABEL_RE.match(mutation_label.strip())
            if m:
                self.failed_seq_edits[(int(m.group(2)), m.group(3))] = list(real_codes)
            else:
                self.failed_chem_labels.add(f"{strategy}:{mutation_label}")

    def is_tabu_seq_edit(self, position: int, to_aa: str) -> bool:
        return (position, to_aa) in self.failed_seq_edits

    def is_tabu_spec(self, spec: _CandidateSpec) -> bool:
        if (spec.smiles, spec.sequence) in self.seen_keys:
            return True
        if spec.sequence is not None:
            m = _MUT_LABEL_RE.match(spec.mutation_label.strip())
            if m and self.is_tabu_seq_edit(int(m.group(2)), m.group(3)):
                return True
        elif self.is_tabu_chem(spec.strategy, spec.mutation_label):
            return True
        return False

    def is_tabu_chem(self, strategy: str, mutation_label: str) -> bool:
        return f"{strategy}:{mutation_label}" in self.failed_chem_labels


def generate_sequence_candidates(
    parent_sequence: str,
    parent_can: str,
    core_lock: CoreLockSpec,
    memory: NegativeSampleMemory,
    strategies: Sequence[str] = ("ala_scan", "point_mutation"),
    prioritized_positions: Optional[Sequence[int]] = None,
    limit: int = 8,
) -> list[_CandidateSpec]:
    """Generate lock-compliant, NegativeSampleMemory-filtered single-step canonical sequence mutations."""
    if core_lock.parent_sequence is None:
        return []
    if len(parent_sequence) != len(core_lock.parent_sequence):
        raise ValueError("parent_sequence length does not match core_lock.parent_sequence")

    out: list[_CandidateSpec] = []
    local_seen: set[str] = {parent_can}

    if prioritized_positions and "point_mutation" in strategies:
        for pos1 in prioritized_positions:
            if len(out) >= limit:
                break
            if pos1 < 1 or pos1 > len(parent_sequence) or pos1 in core_lock.locked_positions:
                continue
            orig = parent_sequence[pos1 - 1]
            if core_lock.lock_disulfides and core_lock.parent_disulfide_count > 0 and orig == "C":
                continue
            for tgt in _TARGETED_SWAPS.get(orig, ["A", "S", "Q", "E", "K"]):
                if len(out) >= limit:
                    break
                if tgt == orig or memory.is_tabu_seq_edit(pos1, tgt):
                    continue
                mut_seq = parent_sequence[: pos1 - 1] + tgt + parent_sequence[pos1:]
                mol = sequence_to_mol(mut_seq)
                if mol is None:
                    continue
                smi = _sanitize_and_canonicalize(mol)
                if smi and smi not in local_seen and (smi, mut_seq) not in memory.seen_keys:
                    local_seen.add(smi)
                    out.append(
                        _CandidateSpec(
                            strategy="point_mutation",
                            mutation_label=f"{orig}{pos1}{tgt}",
                            description=f"Attribution-guided substitution at pos {pos1} ({orig}→{tgt})",
                            sequence=mut_seq,
                            smiles=smi,
                        )
                    )

    # Request extra headroom so tabu filtering doesn't starve the quota.
    # When both ala_scan and point_mutation are enabled, interleave round-robin
    # (point_mutation first, then ala_scan) so neither operator starves the other!
    overfetch = max(limit * 3, 24)
    lock_cys = core_lock.lock_disulfides and core_lock.parent_disulfide_count > 0

    seq_buckets: list[list[_CandidateSpec]] = []
    if len(out) < limit and "point_mutation" in strategies:
        pt_specs, _ = _gen_point_mutations(
            parent_sequence,
            parent_can,
            limit=overfetch,
            locked_positions=core_lock.locked_positions,
            lock_disulfides=lock_cys,
        )
        seq_buckets.append([sp for sp in pt_specs if not memory.is_tabu_spec(sp)])
    if len(out) < limit and "ala_scan" in strategies:
        ala_specs, _ = _gen_ala_scan(
            parent_sequence,
            parent_can,
            limit=overfetch,
            locked_positions=core_lock.locked_positions,
            lock_disulfides=lock_cys,
        )
        seq_buckets.append([sp for sp in ala_specs if not memory.is_tabu_spec(sp)])

    if seq_buckets and len(out) < limit:
        max_len = max((len(b) for b in seq_buckets), default=0)
        for i in range(max_len):
            for b in seq_buckets:
                if len(out) >= limit:
                    break
                if i < len(b):
                    sp = b[i]
                    if sp.smiles not in local_seen and not memory.is_tabu_spec(sp):
                        local_seen.add(sp.smiles)
                        out.append(sp)
            if len(out) >= limit:
                break

    return out[:limit]


def generate_chemical_leaf_candidates(
    parent_mol: Chem.Mol,
    parent_sequence: Optional[str],
    parent_can: str,
    core_lock: CoreLockSpec,
    memory: NegativeSampleMemory,
    strategies: Sequence[str] = ("n_methylation", "cyclization", "lipidation_c16_c18"),
    limit: int = 6,
) -> list[_CandidateSpec]:
    """Generate terminal chemical-graph leaf modifications honoring CoreLockSpec and memory."""
    out: list[_CandidateSpec] = []
    local_seen: set[str] = {parent_can}

    working_mol = parent_mol
    locked_atoms = set(core_lock.locked_atom_indices)
    if parent_sequence and all(aa in CANONICAL_AA for aa in parent_sequence):
        smol = sequence_to_mol(parent_sequence)
        if smol is not None and _sanitize_and_canonicalize(smol) == parent_can:
            working_mol = smol
            locked_atoms = set()
            for atom in smol.GetAtoms():
                info = atom.GetPDBResidueInfo()
                if info is not None and int(info.GetResidueNumber()) in core_lock.locked_positions:
                    locked_atoms.add(atom.GetIdx())
            for _, pat in core_lock.compiled_smarts:
                for m in smol.GetSubstructMatches(pat):
                    locked_atoms.update(int(i) for i in m)

    # P1-1: Collect per-strategy buckets, pre-filter over-budget chem leaves (e.g. Di-N-Me=2, c[C-seq-C]=3),
    # then interleave ROUND-ROBIN across enabled strategies so lipidation & cyclization are never starved!
    per_strat = max(3, limit)
    buckets: list[list[_CandidateSpec]] = []
    parent_bio_ref = compute_biophysical_profile(parent_sequence, working_mol)
    for strat in strategies:
        specs: list[_CandidateSpec] = []
        if strat == "n_methylation":
            specs, _ = _gen_n_methylation(working_mol, parent_can, limit=per_strat, locked_atom_indices=locked_atoms)
        elif strat == "cyclization":
            specs, _ = _gen_cyclization(
                working_mol,
                parent_sequence,
                parent_can,
                limit=per_strat,
                locked_positions=core_lock.locked_positions,
                locked_atom_indices=locked_atoms,
                lock_disulfides=core_lock.lock_disulfides,
            )
        elif strat == "lipidation_c16_c18":
            specs, _ = _gen_lipidation(working_mol, parent_can, limit=per_strat, locked_atom_indices=locked_atoms)
        valid_bucket: list[_CandidateSpec] = []
        for sp in specs:
            if sp.smiles in local_seen or memory.is_tabu_spec(sp):
                continue
            cand_m = parse_smiles_safe(sp.smiles)
            if cand_m is None:
                continue
            cand_bio = compute_biophysical_profile(parent_sequence, cand_m)
            if _infer_chemical_edit_count(parent_bio_ref, cand_bio, cand_m, mutation_label=sp.mutation_label) > core_lock.max_total_chem_edits:
                continue
            valid_bucket.append(sp)
        if valid_bucket:
            buckets.append(valid_bucket)

    # Round-robin interleave across strategy buckets
    idx_ptrs = [0] * len(buckets)
    while len(out) < limit and any(idx_ptrs[b] < len(buckets[b]) for b in range(len(buckets))):
        for b in range(len(buckets)):
            if len(out) >= limit:
                break
            while idx_ptrs[b] < len(buckets[b]):
                sp = buckets[b][idx_ptrs[b]]
                idx_ptrs[b] += 1
                if sp.smiles not in local_seen:
                    local_seen.add(sp.smiles)
                    out.append(sp)
                    break

    return out[:limit]



# ---------------------------------------------------------------------------
# 6. Step 1b: Multi-Round Controlled Evolutionary Engine (`run_peptide_harness`)
# ---------------------------------------------------------------------------

import threading
import time
from typing import Callable

from .adapter import EngineBusyError, EngineUnavailableError, OmniPanelAdapter
from .direct_attribution import compute_headless_multi_endpoint_attribution
from .schemas import AssayConditions, BatchMoleculeResult, BatchParsedRow

SEQ_STRATEGIES: frozenset[str] = frozenset({"ala_scan", "point_mutation"})
CHEM_LEAF_STRATEGIES: frozenset[str] = frozenset({"n_methylation", "cyclization", "lipidation_c16_c18"})
ALL_HARNESS_STRATEGIES: tuple[str, ...] = (
    "ala_scan",
    "point_mutation",
    "n_methylation",
    "cyclization",
    "lipidation_c16_c18",
)


@dataclass
class HarnessConfig:
    objective_x: str = "PAMPA"
    direction_x: Optional[str] = None
    objective_y: str = "half_life_reg"
    direction_y: Optional[str] = None
    functional_endpoint: Optional[str] = None
    functional_direction: Optional[str] = None
    strategies: tuple[str, ...] = ("ala_scan", "point_mutation", "n_methylation", "cyclization", "lipidation_c16_c18")
    locked_positions: tuple[int, ...] = ()
    lock_disulfides: bool = True
    locked_smarts: tuple[str, ...] = ()
    rounds: int = 3
    population_size: int = 6
    top_k: int = 5
    critic_max_perturbations: int = 8
    max_total_evals: int = 36
    wall_clock_budget_s: float = 120.0
    chunk_size: int = 4
    requested_max_aa_edits: Optional[int] = None
    requested_max_chem_edits: Optional[int] = None
    weights: MpoWeights = field(default_factory=MpoWeights)
    conditions: AssayConditions = field(default_factory=AssayConditions)
    require_remote_oracle: bool = False
    require_tanimoto_ad: bool = True
    compute_elite_cross_track_audit: bool = True
    # Explicit ablation switches for Step 3 benchmark (always True in production)
    use_negative_memory: bool = True
    use_ad_safety_gate: bool = True
    use_critic: bool = True


def _extract_raw_and_display(by_ep: dict[str, EndpointPrediction], ep: str) -> tuple[Optional[float], Optional[float], str]:
    p = by_ep.get(ep)
    if p is None or p.value is None:
        return None, None, "N/A"
    val = float(p.value)
    raw = float(p.raw_value) if p.raw_value is not None else val
    return val, raw, p.value_display


def compute_pareto_flags_and_hv(
    feasible_records: list[dict[str, Any]],
    dir_x: str,
    dir_y: str,
    ref_dx: float,
    ref_dy: float,
) -> tuple[list[dict[str, Any]], float]:
    """Mark 2D Pareto dominance on TRACK-MATCHED DELTAS `(delta_x_raw, delta_y_raw)` and compute exact 2D HV.

    P0-1 invariant: Operating strictly in track-matched delta space `(delta_x_raw, delta_y_raw)` ensures
    `dual_modality` and `smiles_only` candidates are compared strictly by improvement over their OWN
    track-matched parent baseline `(0.0, 0.0)`, eliminating cross-track intercept bias!
    Orientations are transformed to maximization space (`sx`, `sy`) where higher is strictly better:
    - `max`: `s = delta_raw`
    - `min`: `s = -delta_raw`
    `ref_dx`, `ref_dy` are specified directly in maximization-oriented delta space (negative numbers < 0,
    e.g. -0.50 for regression log10 or -0.20 for probability).
    """
    sx_sign = 1.0 if dir_x == "max" else -1.0
    sy_sign = 1.0 if dir_y == "max" else -1.0
    srx = float(ref_dx)
    sry = float(ref_dy)

    pts: list[tuple[int, float, float]] = []
    for idx, rec in enumerate(feasible_records):
        rec["is_pareto_optimal"] = False
        dx = rec.get("delta_x_raw")
        dy = rec.get("delta_y_raw")
        if dx is None or dy is None:
            continue
        pts.append((idx, sx_sign * float(dx), sy_sign * float(dy)))

    pareto_pts: list[tuple[int, float, float]] = []
    for i, xi, yi in pts:
        dominated = False
        for j, xj, yj in pts:
            if i == j:
                continue
            if xj >= xi - 1e-12 and yj >= yi - 1e-12 and (xj > xi + 1e-12 or yj > yi + 1e-12):
                dominated = True
                break
        if not dominated:
            feasible_records[i]["is_pareto_optimal"] = True
            pareto_pts.append((i, xi, yi))

    pareto_pts.sort(key=lambda t: (t[1], -t[2]))
    hv = 0.0
    prev_x = srx
    for _, sx, sy in pareto_pts:
        if sx <= srx or sy <= sry:
            continue
        width = max(0.0, sx - prev_x)
        height = max(0.0, sy - sry)
        hv += width * height
        prev_x = max(prev_x, sx)

    frontier = [feasible_records[i] for i, _, _ in pareto_pts]
    frontier.sort(key=lambda r: float(r.get("delta_x_raw") or 0.0))
    return frontier, round(hv, 6)


def _rank_critic_positions(
    impacts_by_ep: dict[str, list[dict[str, Any]]],
    objective_x: str,
    dir_x: str,
    objective_y: str,
    dir_y: str,
    locked_positions: set[int],
) -> tuple[list[int], list[str]]:
    """Score unlocked positions from multi-endpoint Critic deltas & generate human-readable reflection lines."""
    pos_score: dict[int, float] = {}
    reflections: list[str] = []

    # Removing a residue gives `delta_raw = base - masked`:
    # - For a liability endpoint (direction='min', e.g., hemolysis_cls), delta_raw > 0 means removing it REDUCED risk!
    # - For a maximize endpoint (direction='max', e.g., PAMPA), delta_raw < 0 means removing it IMPROVED objective!
    scored_eps: list[tuple[str, str, float]] = [
        (objective_x, dir_x, 1.0),
        (objective_y, dir_y, 1.0),
        ("hemolysis_cls", "min", 0.9),
        ("cytotoxicity_cls", "min", 0.7),
        ("overall_peptide_toxicity", "min", 0.5),
    ]
    seen_ep_weights: dict[str, tuple[str, float]] = {}
    for ep, d, w in scored_eps:
        if ep not in seen_ep_weights:
            seen_ep_weights[ep] = (d, w)

    for ep, (direction, w) in seen_ep_weights.items():
        rows = impacts_by_ep.get(ep) or []
        if not rows:
            continue
        max_abs = max(max(abs(float(r["delta_raw"])) for r in rows), 1e-6)
        top_liability: Optional[dict[str, Any]] = None
        for r in rows:
            pos = r.get("position")
            if pos is None or int(pos) in locked_positions:
                continue
            dr = float(r["delta_raw"])
            # Positive gain_on_removal means mutating this site is promising
            gain_on_removal = (dr if direction == "min" else -dr) / max_abs
            if gain_on_removal > 0:
                pos_score[int(pos)] = pos_score.get(int(pos), 0.0) + w * gain_on_removal
                if top_liability is None or gain_on_removal > (
                    (float(top_liability["delta_raw"]) if direction == "min" else -float(top_liability["delta_raw"])) / max_abs
                ):
                    top_liability = r
        if top_liability is not None:
            reflections.append(
                f"{ep} ({direction}): masking {top_liability['label']} shifted Δ={top_liability['delta_raw']:+.3f} -> target pos {top_liability['position']}."
            )

    ranked = [p for p, _ in sorted(pos_score.items(), key=lambda kv: kv[1], reverse=True)]
    if not reflections:
        reflections.append("No dominant single-site liability detected; exploring balanced unlocked substitutions.")
    return ranked, reflections


def _build_combinatorial_stack_candidates(
    seed_sequence: str,
    beneficial_single_edits: list[tuple[int, str, float]],
    core_lock: CoreLockSpec,
    memory: NegativeSampleMemory,
    limit: int = 3,
) -> list[_CandidateSpec]:
    """Combine top-ranked individually beneficial single-site edits (within cumulative initial-seed budget)."""
    if core_lock.max_total_aa_edits < 2 or len(beneficial_single_edits) < 2:
        return []
    # Keep best target AA per unique unlocked position
    best_per_pos: dict[int, tuple[str, float]] = {}
    for pos1, to_aa, gain in sorted(beneficial_single_edits, key=lambda x: x[2], reverse=True):
        if pos1 in core_lock.locked_positions or pos1 in best_per_pos:
            continue
        best_per_pos[pos1] = (to_aa, gain)
    top_sites = list(best_per_pos.items())[:5]

    out: list[_CandidateSpec] = []
    local_seen: set[str] = set()
    for i in range(len(top_sites)):
        for j in range(i + 1, len(top_sites)):
            if len(out) >= limit:
                break
            (p1, (aa1, _)), (p2, (aa2, _)) = top_sites[i], top_sites[j]
            chars = list(seed_sequence)
            o1, o2 = chars[p1 - 1], chars[p2 - 1]
            chars[p1 - 1] = aa1
            chars[p2 - 1] = aa2
            comb_seq = "".join(chars)
            if sequence_hamming_distance(seed_sequence, comb_seq) > core_lock.max_total_aa_edits:
                continue
            mol = sequence_to_mol(comb_seq)
            if mol is None:
                continue
            smi = _sanitize_and_canonicalize(mol)
            if not smi or smi in local_seen or (smi, comb_seq) in memory.seen_keys:
                continue
            local_seen.add(smi)
            out.append(
                _CandidateSpec(
                    strategy="point_mutation",
                    mutation_label=f"{o1}{p1}{aa1}+{o2}{p2}{aa2}",
                    description=f"Combinatorial stack of top beneficial edits ({o1}{p1}{aa1} + {o2}{p2}{aa2})",
                    sequence=comb_seq,
                    smiles=smi,
                )
            )
    return out


def preflight_validate_harness_request(
    adapter: OmniPanelAdapter,
    *,
    query: Optional[str] = None,
    sequence: Optional[str] = None,
    smiles: Optional[str] = None,
    config: Optional[HarnessConfig] = None,
) -> dict[str, Any]:
    """Synchronously validate parent resolution, core locks, strategies, and Tier-A objectives before queuing a job."""
    cfg = config or HarnessConfig()
    if cfg.require_remote_oracle and adapter.mode != "v33_remote":
        raise EngineUnavailableError(
            f"Peptide-Harness requires live v33_remote oracle, but adapter.mode='{adapter.mode}'."
        )
    if cfg.objective_x == cfg.objective_y:
        raise ValueError(f"objective_x and objective_y must be distinct endpoints (both set to '{cfg.objective_x}').")
    for d_name, d_val in (
        ("direction_x", cfg.direction_x),
        ("direction_y", cfg.direction_y),
        ("functional_direction", cfg.functional_direction),
    ):
        if d_val is not None and d_val not in ("max", "min"):
            raise ValueError(f"Invalid {d_name}='{d_val}'; must be 'max' or 'min'.")
    strategies = [s for s in cfg.strategies if s in ALL_HARNESS_STRATEGIES]
    if not strategies:
        raise ValueError(f"At least one valid strategy required from {ALL_HARNESS_STRATEGIES}")
    seq_strategies = [s for s in strategies if s in SEQ_STRATEGIES]
    chem_strategies = [s for s in strategies if s in CHEM_LEAF_STRATEGIES]

    resolved = adapter.resolve_query(query=query, sequence=sequence, smiles=smiles)
    if resolved.mol is None:
        raise ValueError("Parent molecule could not be parsed into an RDKit atomic graph.")
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    if not parent_can:
        raise ValueError("Parent molecule failed RDKit single-fragment sanitization.")

    core_lock = resolve_core_locks(
        parent_mol=resolved.mol,
        parent_sequence=resolved.sequence,
        locked_positions=cfg.locked_positions,
        lock_disulfides=cfg.lock_disulfides,
        locked_smarts=cfg.locked_smarts,
        functional_endpoint=cfg.functional_endpoint,
        requested_max_aa_edits=cfg.requested_max_aa_edits,
        requested_max_chem_edits=cfg.requested_max_chem_edits,
    )
    if core_lock.harness_mode == "chemical_leaf_only" and not chem_strategies:
        raise ValueError(
            "Parent molecule runs in chemical_leaf_only mode (cyclic/non-canonical/modified scaffold), "
            "which requires at least one chemical strategy (n_methylation, cyclization, lipidation_c16_c18)."
        )
    parent_bio = compute_biophysical_profile(core_lock.parent_sequence, resolved.mol)
    active_tracks: list[str] = []
    if core_lock.parent_sequence is not None:
        active_tracks.append("dual_modality")
        if chem_strategies or cfg.compute_elite_cross_track_audit:
            active_tracks.append("smiles_only")
    else:
        active_tracks.append("smiles_only")

    tier_info = validate_optimization_objectives(
        adapter.manifest,
        cfg.objective_x,
        cfg.objective_y,
        functional_endpoint=cfg.functional_endpoint,
        tracks=active_tracks,
        is_macrocycle=bool(parent_bio["has_macrocycle"]),
    )
    return {
        "query_name": resolved.query_name,
        "resolved_type": resolved.resolved_type,
        "harness_mode": core_lock.harness_mode,
        "activity_unmodeled": core_lock.activity_unmodeled,
        "active_tracks": active_tracks,
        "locked_positions": sorted(core_lock.locked_positions),
        "has_seq_strategies": bool(seq_strategies),
        "has_chem_strategies": bool(chem_strategies),
        "objective_tiers": {k: v.tier for k, v in tier_info.items()},
    }


def run_peptide_harness(
    adapter: OmniPanelAdapter,
    *,
    query: Optional[str] = None,
    sequence: Optional[str] = None,
    smiles: Optional[str] = None,
    config: Optional[HarnessConfig] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    eval_batch_fn: Optional[Callable[[list[BatchParsedRow], bool, AssayConditions], list[BatchMoleculeResult]]] = None,
) -> dict[str, Any]:
    """Execute multi-round controlled evolutionary peptide optimization with strict anti-reward-hacking gates."""
    t0 = time.perf_counter()
    cfg = config or HarnessConfig()

    # Clamp runtime budgets defensively
    n_rounds = max(1, min(4, int(cfg.rounds)))
    pop_size = max(2, min(8, int(cfg.population_size)))
    top_k = max(1, min(5, int(cfg.top_k)))
    critic_max_p = max(2, min(10, int(cfg.critic_max_perturbations)))
    max_total_evals = max(4, min(60, int(cfg.max_total_evals)))
    chunk_size = max(1, min(4, int(cfg.chunk_size)))
    wall_budget_s = max(5.0, float(cfg.wall_clock_budget_s))

    dir_x = cfg.direction_x or default_direction_for_endpoint(cfg.objective_x)
    dir_y = cfg.direction_y or default_direction_for_endpoint(cfg.objective_y)
    func_dir = (
        cfg.functional_direction or default_direction_for_endpoint(cfg.functional_endpoint)
        if cfg.functional_endpoint
        else None
    )

    # Validate strategies
    if cfg.objective_x == cfg.objective_y:
        raise ValueError(f"objective_x and objective_y must be distinct endpoints (both set to '{cfg.objective_x}').")
    for d_name, d_val in (
        ("direction_x", cfg.direction_x),
        ("direction_y", cfg.direction_y),
        ("functional_direction", cfg.functional_direction),
    ):
        if d_val is not None and d_val not in ("max", "min"):
            raise ValueError(f"Invalid {d_name}='{d_val}'; must be 'max' or 'min'.")
    strategies = [s for s in cfg.strategies if s in ALL_HARNESS_STRATEGIES]
    if not strategies:
        raise ValueError(f"At least one valid strategy required from {ALL_HARNESS_STRATEGIES}")
    seq_strategies = [s for s in strategies if s in SEQ_STRATEGIES]
    chem_strategies = [s for s in strategies if s in CHEM_LEAF_STRATEGIES]

    # Assert strict remote oracle upfront if requested (P1-12)
    def _assert_oracle_mode() -> None:
        if cfg.require_remote_oracle and adapter.mode != "v33_remote":
            raise EngineUnavailableError(
                f"Peptide-Harness requires live v33_remote oracle, but adapter.mode='{adapter.mode}'."
            )

    _assert_oracle_mode()

    # Resolve parent structure
    resolved = adapter.resolve_query(query=query, sequence=sequence, smiles=smiles)
    if resolved.mol is None:
        raise ValueError("Parent molecule could not be parsed into an RDKit atomic graph.")
    parent_mol = resolved.mol
    parent_can = _sanitize_and_canonicalize(parent_mol)
    if not parent_can:
        raise ValueError("Parent molecule failed RDKit single-fragment sanitization.")

    core_lock = resolve_core_locks(
        parent_mol=parent_mol,
        parent_sequence=resolved.sequence,
        locked_positions=cfg.locked_positions,
        lock_disulfides=cfg.lock_disulfides,
        locked_smarts=cfg.locked_smarts,
        functional_endpoint=cfg.functional_endpoint,
        requested_max_aa_edits=cfg.requested_max_aa_edits,
        requested_max_chem_edits=cfg.requested_max_chem_edits,
    )
    parent_bio = compute_biophysical_profile(core_lock.parent_sequence, parent_mol)

    # Determine active tracks & whether Tier-2 endpoints are required (fixes R1a P0-1!)
    is_linear_editable = core_lock.parent_sequence is not None
    will_emit_chem_leaves = len(chem_strategies) > 0
    active_tracks: list[str] = []
    if is_linear_editable:
        active_tracks.append("dual_modality")
        if will_emit_chem_leaves or cfg.compute_elite_cross_track_audit:
            active_tracks.append("smiles_only")
    else:
        active_tracks.append("smiles_only")

    tier_info = validate_optimization_objectives(
        adapter.manifest,
        cfg.objective_x,
        cfg.objective_y,
        functional_endpoint=cfg.functional_endpoint,
        tracks=active_tracks,
        is_macrocycle=bool(parent_bio["has_macrocycle"]),
    )

    ep_docs = adapter.manifest.get("endpoints") or {}
    req_eps = [cfg.objective_x, cfg.objective_y, *MPO_PILLAR_ENDPOINTS]
    if cfg.functional_endpoint:
        req_eps.append(cfg.functional_endpoint)
    include_tier2 = any(int((ep_docs.get(ep) or {}).get("tier", 1)) == 2 for ep in req_eps)

    # Helper for chunked evaluation with budget, single retry on EngineBusyError, partial degradation & cancel tracking
    evals_used = 0
    budget_exhausted = False
    cancelled = False
    partial_oracle_error: Optional[str] = None
    total_oracle_errors = 0

    # P1-4: Reserve budget for cross-track epistemic audit when enabled & budget permits
    audit_reserve = (
        min(3, top_k)
        if (
            cfg.compute_elite_cross_track_audit
            and "dual_modality" in active_tracks
            and "smiles_only" in active_tracks
            and max_total_evals >= 10
        )
        else 0
    )
    loop_eval_cap = max_total_evals - audit_reserve
    # R1b P1-3: Reserve cross-round budget for terminal chemical leaves so early sequence rounds never starve them!
    leaf_reserve = (
        min(5, max(len(chem_strategies), 4))
        if (
            chem_strategies
            and core_lock.harness_mode != "chemical_leaf_only"
            and loop_eval_cap >= len(active_tracks) + 12
        )
        else (
            len(chem_strategies)
            if (
                chem_strategies
                and core_lock.harness_mode != "chemical_leaf_only"
                and loop_eval_cap >= len(active_tracks) + len(chem_strategies) + 2
            )
            else 0
        )
    )
    seq_eval_cap = loop_eval_cap - leaf_reserve
    leaf_pass_done = False

    def _time_left() -> float:
        return wall_budget_s - (time.perf_counter() - t0)

    def _run_batch(rows: list[BatchParsedRow], allow_partial: bool = False) -> list[BatchMoleculeResult]:
        nonlocal evals_used, partial_oracle_error, total_oracle_errors
        if not rows:
            return []
        _assert_oracle_mode()
        last_exc: Optional[Exception] = None
        for attempt in range(2):
            try:
                if eval_batch_fn is not None:
                    res_list = eval_batch_fn(rows, include_tier2, cfg.conditions)
                else:
                    res_list = adapter.predict_batch(rows, include_tier2=include_tier2, conditions=cfg.conditions)
                if cancel_event is not None and cancel_event.is_set():
                    evals_used += len(res_list)
                    return []
                if len(res_list) != len(rows):
                    raise RuntimeError(f"Oracle batch returned {len(res_list)} results for {len(rows)} requested rows.")
                evals_used += len(res_list)
                return res_list
            except EngineBusyError as exc:
                last_exc = exc
                if attempt == 0 and _time_left() > 3.0 and (cancel_event is None or not cancel_event.is_set()):
                    time.sleep(min(1.5, max(0.05, _time_left() * 0.1)))
                    continue
                if allow_partial:
                    partial_oracle_error = f"EngineBusyError after retry: {exc}"
                    total_oracle_errors += len(rows)
                    return []
                raise
            except EngineUnavailableError:
                raise
            except Exception as exc:
                if allow_partial:
                    partial_oracle_error = f"{type(exc).__name__}: {exc}"
                    total_oracle_errors += len(rows)
                    return []
                raise
        if last_exc is not None and not allow_partial:
            raise last_exc
        return []

    def _eval_chunked(rows: list[BatchParsedRow], cap_override: Optional[int] = None) -> list[BatchMoleculeResult]:
        nonlocal budget_exhausted, cancelled
        cap = max_total_evals if cap_override is None else cap_override
        collected: list[BatchMoleculeResult] = []
        for start in range(0, len(rows), chunk_size):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if partial_oracle_error is not None:
                break
            if _time_left() <= 1.0 or evals_used >= cap:
                budget_exhausted = True
                break
            rem = cap - evals_used
            chunk = rows[start : start + min(chunk_size, rem)]
            if not chunk:
                budget_exhausted = True
                break
            out_chunk = _run_batch(chunk, allow_partial=True)
            if not out_chunk and partial_oracle_error is not None:
                break
            collected.extend(out_chunk)
        return collected

    # Step 0: Evaluate parent on active track(s)
    if progress_callback:
        progress_callback({"phase": "init_parent", "current_round": 0, "total_rounds": n_rounds, "progress_pct": 5.0})

    parent_rows: list[BatchParsedRow] = []
    if "dual_modality" in active_tracks:
        parent_rows.append(
            BatchParsedRow(
                index=0,
                name=f"{resolved.query_name}_dual",
                detected_type="sequence",
                sequence=core_lock.parent_sequence,
                smiles=parent_can,
                valid=True,
            )
        )
    if "smiles_only" in active_tracks:
        parent_rows.append(
            BatchParsedRow(
                index=len(parent_rows),
                name=f"{resolved.query_name}_smiles",
                detected_type="smiles",
                sequence=None,
                smiles=parent_can,
                valid=True,
            )
        )
    if cancel_event is not None and cancel_event.is_set():
        return {
            "status": "cancelled",
            "partial_error": None,
            "harness_mode": core_lock.harness_mode,
            "activity_unmodeled": core_lock.activity_unmodeled,
            "no_improvement_over_parent": True,
            "oracle_mode": adapter.mode,
            "oracle_is_surrogate_heuristic": adapter.mode != "v33_remote",
            "rounds": [],
            "candidates": [],
            "pareto_frontier": [],
            "final_recommendations": [],
            "lineage_edges": [],
            "negative_memory": [],
            "cross_track_epistemic_audit": [],
            "metrics": {
                "evals_used": evals_used,
                "max_total_evals": max_total_evals,
                "budget_exhausted": False,
                "cancelled": True,
                "runtime_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        }
    parent_bm_list = _run_batch(parent_rows)
    if cancel_event is not None and cancel_event.is_set():
        return {
            "status": "cancelled",
            "partial_error": None,
            "harness_mode": core_lock.harness_mode,
            "activity_unmodeled": core_lock.activity_unmodeled,
            "no_improvement_over_parent": True,
            "oracle_mode": adapter.mode,
            "oracle_is_surrogate_heuristic": adapter.mode != "v33_remote",
            "rounds": [],
            "candidates": [],
            "pareto_frontier": [],
            "final_recommendations": [],
            "lineage_edges": [],
            "negative_memory": [],
            "cross_track_epistemic_audit": [],
            "metrics": {
                "evals_used": evals_used,
                "max_total_evals": max_total_evals,
                "budget_exhausted": False,
                "cancelled": True,
                "runtime_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            },
        }
    parent_preds_by_track: dict[str, dict[str, EndpointPrediction]] = {}
    parent_bm_by_track: dict[str, BatchMoleculeResult] = {}
    for tr, bm in zip(active_tracks, parent_bm_list):
        if bm.status != "ok":
            raise ValueError(f"Parent evaluation failed on track '{tr}': {bm.error}")
        parent_bm_by_track[tr] = bm
        parent_preds_by_track[tr] = {p.endpoint: p for p in bm.predictions}

    gatekeeper = DeterministicGatekeeper(
        core_lock=core_lock,
        parent_biophysical=parent_bio,
        parent_preds_by_track=parent_preds_by_track,
        objective_x=cfg.objective_x,
        objective_y=cfg.objective_y,
        functional_endpoint=cfg.functional_endpoint,
        functional_direction=func_dir,
        weights=cfg.weights,
        require_tanimoto_ad=cfg.require_tanimoto_ad,
    )
    primary_track = gatekeeper.primary_track

    # Compute parent baseline summary per track
    parent_summary_by_track: dict[str, dict[str, Any]] = {}
    for tr in active_tracks:
        p_by_ep = parent_preds_by_track[tr]
        p_sb = gatekeeper.check_stage_b(list(p_by_ep.values()), tr)
        xv, xr, xd = _extract_raw_and_display(p_by_ep, cfg.objective_x)
        yv, yr, yd = _extract_raw_and_display(p_by_ep, cfg.objective_y)
        parent_summary_by_track[tr] = {
            "modality_track": tr,
            "gmpo_score": p_sb.gmpo_score,
            "pillar_scores": p_sb.pillar_scores,
            "x_value": xv,
            "x_raw": xr,
            "x_display": xd,
            "y_value": yv,
            "y_raw": yr,
            "y_display": yd,
            "ad_overall": p_sb.ad_overall,
        }

    p_prim = parent_summary_by_track[primary_track]
    p_prim_by_ep = parent_preds_by_track[primary_track]
    parent_record: dict[str, Any] = {
        "id": "parent",
        "parent_id": None,
        "round": 0,
        "name": resolved.query_name,
        "strategy": "parent",
        "mutation_label": "Parent",
        "lineage_path": ["Parent"],
        "sequence": core_lock.parent_sequence,
        "smiles": parent_can,
        "is_parent": True,
        "is_chemical_leaf": not is_linear_editable,
        "is_pareto_optimal": False,
        "feasible": True,
        "reject_codes": [],
        "soft_warnings": [],
        "aa_edits_from_seed": 0,
        "chem_edits_from_seed": 0,
        "modality_track": primary_track,
        "track_switched": False,
        "gmpo_score": p_prim["gmpo_score"],
        "delta_gmpo": 0.0,
        "pillar_scores": p_prim["pillar_scores"],
        "x_value": p_prim["x_value"],
        "x_raw": p_prim["x_raw"],
        "x_display": p_prim["x_display"],
        "y_value": p_prim["y_value"],
        "y_raw": p_prim["y_raw"],
        "y_display": p_prim["y_display"],
        "delta_x_raw": 0.0,
        "delta_y_raw": 0.0,
        "ad_overall": p_prim["ad_overall"],
        "ad_factor": 1.0,
        "functional_delta": 0.0 if cfg.functional_endpoint else None,
        "biophysical": parent_bio,
        "key_metrics": {
            ep: {
                "value": p_prim_by_ep[ep].value,
                "raw_value": p_prim_by_ep[ep].raw_value,
                "display": p_prim_by_ep[ep].value_display,
                "ad_status": p_prim_by_ep[ep].ad_status,
                "max_tanimoto": p_prim_by_ep[ep].max_tanimoto,
            }
            for ep in (cfg.objective_x, cfg.objective_y, "hemolysis_cls", "cytotoxicity_cls", "overall_peptide_toxicity", "half_life_reg", "PAMPA")
            if ep in p_prim_by_ep
        },
    }

    # Fixed scale-appropriate delta-space Pareto HV reference point (P0-1 & P2-8: parent sits at (0.0, 0.0)!)
    x_is_cls = str((ep_docs.get(cfg.objective_x) or {}).get("task_type") or "") == "classification"
    y_is_cls = str((ep_docs.get(cfg.objective_y) or {}).get("task_type") or "") == "classification"
    ref_dx = -0.20 if x_is_cls else -0.50
    ref_dy = -0.20 if y_is_cls else -0.50

    memory = NegativeSampleMemory()
    memory.record_seen(parent_can, core_lock.parent_sequence)
    if "smiles_only" in active_tracks:
        memory.record_seen(parent_can, None)

    all_candidates: list[dict[str, Any]] = []
    feasible_pool: list[dict[str, Any]] = [parent_record]
    bm_by_candidate_id: dict[str, BatchMoleculeResult] = {"parent": parent_bm_by_track[primary_track]}
    lineage_edges: list[dict[str, Any]] = []
    rounds_log: list[dict[str, Any]] = []
    beneficial_single_edits: list[tuple[int, str, float]] = []

    cand_seq_counter = 0
    last_critic_elite_id: Optional[str] = None
    last_critic_positions: list[int] = []
    last_critic_reflections: list[str] = []

    no_editable_positions = (
        core_lock.parent_sequence is not None
        and len(core_lock.locked_positions) >= len(core_lock.parent_sequence)
    )

    def _score_evaluated_candidate(
        *,
        spec: _CandidateSpec,
        round_idx: int,
        parent_node: dict[str, Any],
        sa: StageAResult,
        bm: BatchMoleculeResult,
        is_chem_leaf: bool,
        lineage_seq: Optional[str],
    ) -> dict[str, Any]:
        nonlocal cand_seq_counter
        cand_seq_counter += 1
        cid = f"r{round_idx}_c{cand_seq_counter}"
        track = "dual_modality" if (spec.sequence is not None and "dual_modality" in parent_preds_by_track) else "smiles_only"
        by_ep = {p.endpoint: p for p in bm.predictions}
        sb = gatekeeper.check_stage_b(bm.predictions, track)

        # In ablation mode (`use_ad_safety_gate=False`), allow Stage B OOD/safety rejects into pool while still recording them
        feasible = sb.passed if cfg.use_ad_safety_gate else not any(
            c.startswith("MISSING_REQUIRED_ENDPOINT") for c in sb.reject_codes
        )

        # Track-matched deltas (Red Line 3!)
        p_ref = parent_summary_by_track[track]
        xv, xr, xd = _extract_raw_and_display(by_ep, cfg.objective_x)
        yv, yr, yd = _extract_raw_and_display(by_ep, cfg.objective_y)
        dx_raw = round(xr - float(p_ref["x_raw"]), 5) if (xr is not None and p_ref["x_raw"] is not None) else None
        dy_raw = round(yr - float(p_ref["y_raw"]), 5) if (yr is not None and p_ref["y_raw"] is not None) else None
        d_gmpo = round(sb.gmpo_score - float(p_ref["gmpo_score"]), 2)

        rec: dict[str, Any] = {
            "id": cid,
            "parent_id": parent_node["id"],
            "round": round_idx,
            "name": f"{resolved.query_name} [{spec.mutation_label}]",
            "strategy": spec.strategy,
            "mutation_label": spec.mutation_label,
            "description": spec.description,
            "lineage_path": [*parent_node["lineage_path"], spec.mutation_label],
            "sequence": spec.sequence,
            "lineage_sequence": lineage_seq,
            "smiles": sa.canonical_smiles or spec.smiles,
            "is_parent": False,
            "is_chemical_leaf": is_chem_leaf,
            "is_pareto_optimal": False,
            "feasible": feasible,
            "stage_b_passed": sb.passed,
            "reject_codes": list(sb.reject_codes),
            "soft_warnings": list(sa.soft_warnings),
            "aa_edits_from_seed": sa.aa_edits_from_seed,
            "chem_edits_from_seed": sa.chem_edits_from_seed,
            "modality_track": track,
            "track_switched": sb.track_switched,
            "gmpo_score": sb.gmpo_score,
            "delta_gmpo": d_gmpo,
            "pillar_scores": sb.pillar_scores,
            "x_value": xv,
            "x_raw": xr,
            "x_display": xd,
            "y_value": yv,
            "y_raw": yr,
            "y_display": yd,
            "delta_x_raw": dx_raw,
            "delta_y_raw": dy_raw,
            "ad_overall": sb.ad_overall,
            "ad_factor": sb.ad_factor,
            "functional_delta": sb.functional_delta,
            "biophysical": sa.biophysical,
            "key_metrics": {
                ep: {
                    "value": by_ep[ep].value,
                    "raw_value": by_ep[ep].raw_value,
                    "display": by_ep[ep].value_display,
                    "ad_status": by_ep[ep].ad_status,
                    "max_tanimoto": by_ep[ep].max_tanimoto,
                }
                for ep in (cfg.objective_x, cfg.objective_y, "hemolysis_cls", "cytotoxicity_cls", "overall_peptide_toxicity", "half_life_reg", "PAMPA")
                if ep in by_ep
            },
        }
        all_candidates.append(rec)
        bm_by_candidate_id[cid] = bm
        lineage_edges.append(
            {
                "parent_id": parent_node["id"],
                "child_id": cid,
                "round": round_idx,
                "strategy": spec.strategy,
                "mutation_label": spec.mutation_label,
                "delta_gmpo": d_gmpo,
                "feasible": feasible,
            }
        )
        if feasible:
            feasible_pool.append(rec)
            # Record single-site beneficial sequence edits for Round>=2 combinatorial stacking
            if spec.sequence is not None and sa.aa_edits_from_seed == 1 and d_gmpo > 0.0:
                m = _MUT_LABEL_RE.match(spec.mutation_label.strip())
                if m:
                    beneficial_single_edits.append((int(m.group(2)), m.group(3), d_gmpo))
        elif cfg.use_negative_memory:
            memory.record_failure(
                round_idx,
                spec.mutation_label,
                spec.strategy,
                sb.reject_codes,
                canonical_smiles=rec["smiles"],
                sequence=spec.sequence,
            )
        return rec

    # Multi-round loop (respecting reserved cross-track audit budget `loop_eval_cap`)
    for r_idx in range(1, n_rounds + 1):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        if partial_oracle_error is not None:
            break
        if _time_left() <= 1.0 or evals_used >= loop_eval_cap:
            budget_exhausted = True
            break
        _assert_oracle_mode()

        # Pick current best expandable parent (ranked by track-matched `delta_gmpo` — P0-1!)
        if core_lock.harness_mode == "chemical_leaf_only":
            expandable_elites = [parent_record]
        else:
            expandable_elites = [
                c
                for c in feasible_pool
                if (not c["is_chemical_leaf"])
                and c["sequence"] is not None
                and int(c["aa_edits_from_seed"]) < core_lock.max_total_aa_edits
            ]
            if not expandable_elites:
                expandable_elites = [parent_record]
        expandable_elites.sort(key=lambda c: (float(c["delta_gmpo"]), -len(c["soft_warnings"])), reverse=True)
        elite_parent = expandable_elites[0]

        round_candidates: list[dict[str, Any]] = []
        round_stage_a_rejects: list[dict[str, Any]] = []
        round_oracle_errors = 0
        critic_impacts_summary: list[dict[str, Any]] = []
        critic_reflections: list[str] = list(last_critic_reflections)
        recycled_count = 0

        # --- Phase A: Headless Multi-Endpoint Critic (skip in chemical_leaf_only or when elite unchanged!) ---
        is_final_round = r_idx == n_rounds
        run_leaf_this_round = bool(chem_strategies) and (
            core_lock.harness_mode == "chemical_leaf_only"
            or ((not leaf_pass_done) and (is_final_round or evals_used >= seq_eval_cap))
        )
        round_eval_cap = loop_eval_cap if (run_leaf_this_round or core_lock.harness_mode == "chemical_leaf_only" or not chem_strategies) else seq_eval_cap
        proposal_reserve = (
            max(2, len(chem_strategies) + 1)
            if (run_leaf_this_round and chem_strategies and core_lock.harness_mode != "chemical_leaf_only")
            else 2
        )
        critic_budget_avail = max(0, round_eval_cap - evals_used - proposal_reserve)
        critic_cap = (
            min(critic_max_p, critic_budget_avail)
            if "ala_scan" in seq_strategies
            else min(critic_max_p, critic_budget_avail, max(2, max_total_evals // 4))
        )
        # Run headless Critic on Round 1 (100% recycled as ala_scan); only re-run in Round>=2 when budget is large (>=44)
        # so tight 24-36 eval budgets invest Round 2+ evals into Critic-guided point mutations & combinatorial stacking!
        allow_critic_this_round = (r_idx == 1) or (max_total_evals >= 44)
        if (
            cfg.use_critic
            and allow_critic_this_round
            and core_lock.harness_mode != "chemical_leaf_only"
            and elite_parent["sequence"] is not None
            and elite_parent["id"] != last_critic_elite_id
            and critic_cap >= 2
            and _time_left() > 3.0
        ):
            if progress_callback:
                progress_callback(
                    {
                        "phase": "critic_attribution",
                        "current_round": r_idx,
                        "total_rounds": n_rounds,
                        "progress_pct": round(10.0 + ((r_idx - 1) / n_rounds) * 75.0, 1),
                    }
                )
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            elite_mol = parse_smiles_safe(elite_parent["smiles"])
            if elite_mol is not None:
                critic_eps = list(
                    dict.fromkeys([cfg.objective_x, cfg.objective_y, "hemolysis_cls", "cytotoxicity_cls", "overall_peptide_toxicity"])
                )
                # P1-3: Exclude locked positions AND positions already mutated from initial seed
                already_mutated_pos: set[int] = set()
                if core_lock.parent_sequence and elite_parent["sequence"]:
                    for idx_p, (a0, a1) in enumerate(zip(core_lock.parent_sequence, elite_parent["sequence"]), start=1):
                        if a0 != a1:
                            already_mutated_pos.add(idx_p)
                critic_excl_pos = set(core_lock.locked_positions) | already_mutated_pos
                base_bm = bm_by_candidate_id.get(elite_parent["id"])
                critic_out = compute_headless_multi_endpoint_attribution(
                    adapter=adapter,
                    mol=elite_mol,
                    sequence=elite_parent["sequence"],
                    canonical_smiles=elite_parent["smiles"],
                    target_endpoints=critic_eps,
                    locked_positions=critic_excl_pos,
                    locked_atom_indices=core_lock.locked_atom_indices,
                    seen_keys=memory.seen_keys,
                    max_perturbations=critic_cap,
                    include_tier2=include_tier2,
                    conditions=cfg.conditions,
                    chunk_size=chunk_size,
                    base_result=base_bm,
                    eval_batch_fn=(lambda rows: _eval_chunked(rows, cap_override=round_eval_cap)),
                    cancel_event=cancel_event,
                )
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if partial_oracle_error is not None:
                    break
                if critic_out.get("base_ok"):
                    last_critic_elite_id = elite_parent["id"]
                    last_critic_positions, critic_reflections = _rank_critic_positions(
                        critic_out["impacts_by_endpoint"],
                        cfg.objective_x,
                        dir_x,
                        cfg.objective_y,
                        dir_y,
                        critic_excl_pos,
                    )
                    last_critic_reflections = list(critic_reflections)
                    critic_impacts_summary = (critic_out["impacts_by_endpoint"].get(cfg.objective_x) or [])[:6]

                    # Zero-cost Ala/Gly-masking recycling directly into this round's candidate pool!
                    if "ala_scan" in seq_strategies:
                        for pl, bm in critic_out["reusable_evals"]:
                            if pl.perturbed_sequence is None or pl.position is None:
                                continue
                            mut_lbl = f"{pl.original_residue}{pl.position}{pl.masked_to}"
                            recycled_strat = "gly_mask" if pl.masked_to == "G" else "ala_scan"
                            sp = _CandidateSpec(
                                strategy=recycled_strat,
                                mutation_label=mut_lbl,
                                description=f"Recycled Critic masking ({pl.label})",
                                sequence=pl.perturbed_sequence,
                                smiles=pl.perturbed_smiles,
                            )
                            sa = gatekeeper.check_stage_a(
                                sp.smiles,
                                sp.sequence,
                                is_chemical_leaf=False,
                                parent_was_chemical_leaf=False,
                                seen_keys=memory.seen_keys,
                                lineage_sequence=sp.sequence,
                                mutation_label=sp.mutation_label,
                            )
                            if not sa.passed:
                                round_stage_a_rejects.append({"mutation_label": sp.mutation_label, "codes": sa.reject_codes})
                                if cfg.use_negative_memory:
                                    memory.record_failure(r_idx, sp.mutation_label, sp.strategy, sa.reject_codes, sp.smiles, sp.sequence)
                                continue
                            memory.record_seen(sa.canonical_smiles or sp.smiles, sp.sequence)
                            rec = _score_evaluated_candidate(
                                spec=sp,
                                round_idx=r_idx,
                                parent_node=elite_parent,
                                sa=sa,
                                bm=bm,
                                is_chem_leaf=False,
                                lineage_seq=sp.sequence,
                            )
                            round_candidates.append(rec)
                            recycled_count += 1

        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        if partial_oracle_error is not None:
            break

        # --- Phase B: Proposal Generation (Sequence-First, Chemical-Leaf on Final Round or chemical_leaf_only) ---
        if progress_callback:
            progress_callback(
                {
                    "phase": "proposing",
                    "current_round": r_idx,
                    "total_rounds": n_rounds,
                    "progress_pct": round(10.0 + ((r_idx - 0.5) / n_rounds) * 75.0, 1),
                }
            )
        seq_proposals: list[tuple[_CandidateSpec, dict[str, Any], bool, Optional[str]]] = []
        leaf_proposals: list[tuple[_CandidateSpec, dict[str, Any], bool, Optional[str]]] = []
        active_mem = memory if cfg.use_negative_memory else NegativeSampleMemory()
        if not cfg.use_negative_memory:
            active_mem.seen_keys = set(memory.seen_keys)

        round_local_seen: set[tuple[str, Optional[str]]] = set(active_mem.seen_keys)

        if core_lock.harness_mode != "chemical_leaf_only" and seq_strategies and elite_parent["sequence"] is not None:
            if r_idx >= 2 and core_lock.parent_sequence is not None and "point_mutation" in seq_strategies:
                comb_specs = _build_combinatorial_stack_candidates(
                    core_lock.parent_sequence,
                    beneficial_single_edits,
                    core_lock,
                    active_mem,
                    limit=min(3, pop_size),
                )
                for sp in comb_specs:
                    if (sp.smiles, sp.sequence) not in round_local_seen:
                        round_local_seen.add((sp.smiles, sp.sequence))
                        seq_proposals.append((sp, parent_record, False, sp.sequence))

            seq_quota = max(1, pop_size - len(seq_proposals))
            # If Critic already recycled ala_scan this round and point_mutation is enabled, focus Phase B on point_mutation!
            eff_seq_strats = (
                ["point_mutation"]
                if (recycled_count > 0 and "point_mutation" in seq_strategies)
                else seq_strategies
            )
            seq_specs = generate_sequence_candidates(
                parent_sequence=elite_parent["sequence"],
                parent_can=elite_parent["smiles"],
                core_lock=core_lock,
                memory=active_mem,
                strategies=eff_seq_strats,
                prioritized_positions=last_critic_positions,
                limit=seq_quota,
            )
            for sp in seq_specs:
                if (sp.smiles, sp.sequence) not in round_local_seen:
                    round_local_seen.add((sp.smiles, sp.sequence))
                    seq_proposals.append((sp, elite_parent, False, sp.sequence))

        # Terminal chemical-leaf modifications on final round (or as soon as seq_eval_cap is reached / chemical_leaf_only)
        if run_leaf_this_round:
            if core_lock.harness_mode != "chemical_leaf_only":
                leaf_pass_done = True
            leaf_parent = elite_parent if core_lock.harness_mode != "chemical_leaf_only" else parent_record
            leaf_mol = parse_smiles_safe(leaf_parent["smiles"])
            if leaf_mol is not None:
                leaf_quota = (
                    pop_size
                    if core_lock.harness_mode == "chemical_leaf_only"
                    else max(len(chem_strategies), leaf_reserve, pop_size // 2)
                )
                leaf_specs = generate_chemical_leaf_candidates(
                    parent_mol=leaf_mol,
                    parent_sequence=leaf_parent["sequence"],
                    parent_can=leaf_parent["smiles"],
                    core_lock=core_lock,
                    memory=active_mem,
                    strategies=chem_strategies,
                    limit=leaf_quota,
                )
                for sp in leaf_specs:
                    if (sp.smiles, None) not in round_local_seen:
                        round_local_seen.add((sp.smiles, None))
                        leaf_proposals.append((sp, leaf_parent, True, leaf_parent["sequence"]))

        # Order chemical-leaf & sequence proposals on leaf round so tight eval caps never starve leaves.
        # When `evals_used >= seq_eval_cap` (remaining slots are the leaf_reserve), place
        # `min(len(chem_strategies), leaf_reserve)` round-robin leaves FIRST before interleaving sequence specs!
        proposals: list[tuple[_CandidateSpec, dict[str, Any], bool, Optional[str]]] = []
        if leaf_proposals and seq_proposals:
            front_leaves = (
                min(len(chem_strategies), leaf_reserve, len(leaf_proposals))
                if (run_leaf_this_round and evals_used >= seq_eval_cap)
                else 0
            )
            proposals.extend(leaf_proposals[:front_leaves])
            li, si = front_leaves, 0
            while li < len(leaf_proposals) or si < len(seq_proposals):
                if li < len(leaf_proposals):
                    proposals.append(leaf_proposals[li])
                    li += 1
                if si < len(seq_proposals):
                    proposals.append(seq_proposals[si])
                    si += 1
        else:
            proposals = seq_proposals + leaf_proposals

        # --- Phase C: Stage A Gatekeeper Filter (0 Oracle calls if rejected!) ---
        passed_stage_a: list[tuple[_CandidateSpec, dict[str, Any], StageAResult, bool, Optional[str]]] = []
        # Reserve slots for chemical leaves on leaf round so sequence proposals don't crowd them out,
        # and cap to remaining round budget so Phase C never pre-marks unevaluated specs as seen!
        max_round_evals = min(
            pop_size + (len(chem_strategies) if (run_leaf_this_round and chem_strategies and core_lock.harness_mode != "chemical_leaf_only") else 0),
            max(0, round_eval_cap - evals_used),
        )
        for sp, p_node, is_leaf, lin_seq in proposals:
            if len(passed_stage_a) >= max_round_evals:
                break
            sa = gatekeeper.check_stage_a(
                sp.smiles,
                sp.sequence,
                is_chemical_leaf=is_leaf,
                parent_was_chemical_leaf=bool(p_node["is_chemical_leaf"]) and (not p_node["is_parent"]),
                seen_keys=memory.seen_keys,
                lineage_sequence=lin_seq,
                mutation_label=sp.mutation_label,
            )
            if not sa.passed:
                round_stage_a_rejects.append({"mutation_label": sp.mutation_label, "strategy": sp.strategy, "codes": sa.reject_codes})
                if cfg.use_negative_memory:
                    memory.record_failure(r_idx, sp.mutation_label, sp.strategy, sa.reject_codes, sa.canonical_smiles, sp.sequence)
                continue
            memory.record_seen(sa.canonical_smiles or sp.smiles, sp.sequence)
            passed_stage_a.append((sp, p_node, sa, is_leaf, lin_seq))

        # --- Phase D: Chunked Oracle Evaluation + Stage B Gatekeeper ---
        if passed_stage_a:
            if progress_callback:
                progress_callback(
                    {
                        "phase": "scoring",
                        "current_round": r_idx,
                        "total_rounds": n_rounds,
                        "progress_pct": round(10.0 + (r_idx / n_rounds) * 75.0, 1),
                    }
                )
            batch_rows = [
                BatchParsedRow(
                    index=i,
                    name=sp.mutation_label,
                    detected_type="sequence" if sp.sequence else "smiles",
                    sequence=sp.sequence,
                    smiles=sa.canonical_smiles or sp.smiles,
                    valid=True,
                )
                for i, (sp, _, sa, _, _) in enumerate(passed_stage_a)
            ]
            bm_results = _eval_chunked(batch_rows, cap_override=round_eval_cap)
            for (sp, p_node, sa, is_leaf, lin_seq), bm in zip(passed_stage_a, bm_results):
                if bm.status != "ok":
                    round_oracle_errors += 1
                    total_oracle_errors += 1
                    continue
                rec = _score_evaluated_candidate(
                    spec=sp,
                    round_idx=r_idx,
                    parent_node=p_node,
                    sa=sa,
                    bm=bm,
                    is_chem_leaf=is_leaf,
                    lineage_seq=lin_seq,
                )
                round_candidates.append(rec)

        frontier_now, hv_now = compute_pareto_flags_and_hv(feasible_pool, dir_x, dir_y, ref_dx, ref_dy)
        best_delta_now = max((float(c["delta_gmpo"]) for c in feasible_pool), default=0.0)
        best_mpo_now = max((float(c["gmpo_score"]) for c in feasible_pool), default=float(parent_record["gmpo_score"]))
        rounds_log.append(
            {
                "round_index": r_idx,
                "elite_parent_id": elite_parent["id"],
                "elite_parent_label": elite_parent["mutation_label"],
                "critic_reflections": critic_reflections,
                "critic_top_impacts": critic_impacts_summary,
                "recycled_critic_evals": recycled_count,
                "n_stage_a_rejected": len(round_stage_a_rejects),
                "n_oracle_errors": round_oracle_errors,
                "stage_a_reject_sample": round_stage_a_rejects[:6],
                "n_evaluated": len(round_candidates),
                "n_feasible": sum(1 for c in round_candidates if c["feasible"]),
                "best_gmpo_so_far": best_mpo_now,
                "best_delta_gmpo_so_far": round(best_delta_now, 2),
                "pareto_count": len(frontier_now),
                "feasible_pareto_hv": hv_now,
                "evals_used_cumulative": evals_used,
            }
        )

    # P0-1: Delta-space Pareto & Top-K ranking strictly by track-matched `delta_gmpo`, requiring `delta_gmpo > 0`!
    frontier_final, hv_final = compute_pareto_flags_and_hv(feasible_pool, dir_x, dir_y, ref_dx, ref_dy)
    feasible_non_parent = [c for c in feasible_pool if not c["is_parent"]]
    improved_feasible = [c for c in feasible_non_parent if float(c["delta_gmpo"]) > 0.0]
    improved_feasible.sort(
        key=lambda c: (bool(c["is_pareto_optimal"]), float(c["delta_gmpo"]), -len(c["soft_warnings"])),
        reverse=True,
    )
    final_top_k = improved_feasible[:top_k]

    # --- Post-Loop Cross-Track Epistemic Audit (P1-4 & P1-5: reserved budget, offset-free |d_dual - d_smiles|, Stage B check!) ---
    cross_track_audit: list[dict[str, Any]] = []
    audit_skipped_reason: Optional[str] = None
    if not cfg.compute_elite_cross_track_audit:
        audit_skipped_reason = "disabled_by_config"
    elif "dual_modality" not in parent_summary_by_track or "smiles_only" not in parent_summary_by_track:
        audit_skipped_reason = "single_track_run"
    elif cancelled or partial_oracle_error is not None:
        audit_skipped_reason = "run_cancelled_or_partial"
    elif evals_used >= max_total_evals or _time_left() <= 1.5:
        audit_skipped_reason = "budget_or_time_exhausted"
    else:
        dual_elites = [c for c in final_top_k if c["modality_track"] == "dual_modality" and c["sequence"] is not None][:3]
        if not dual_elites:
            audit_skipped_reason = "no_improved_dual_elites"
        else:
            audit_rows = [
                BatchParsedRow(
                    index=i,
                    name=f"xtrack_{c['id']}",
                    detected_type="smiles",
                    sequence=None,
                    smiles=c["smiles"],
                    valid=True,
                )
                for i, c in enumerate(dual_elites)
            ]
            audit_bms = _eval_chunked(audit_rows, cap_override=max_total_evals)
            p_smi = parent_summary_by_track["smiles_only"]
            for c, bm in zip(dual_elites, audit_bms):
                if bm.status != "ok":
                    continue
                by_ep = {p.endpoint: p for p in bm.predictions}
                sb_smi = gatekeeper.check_stage_b(bm.predictions, "smiles_only")
                _, xr_s, _ = _extract_raw_and_display(by_ep, cfg.objective_x)
                _, yr_s, _ = _extract_raw_and_display(by_ep, cfg.objective_y)
                dx_smi = (float(xr_s) - float(p_smi["x_raw"])) if (xr_s is not None and p_smi["x_raw"] is not None) else None
                dy_smi = (float(yr_s) - float(p_smi["y_raw"])) if (yr_s is not None and p_smi["y_raw"] is not None) else None
                # P1-5: Offset-free epistemic disagreement = |delta_dual - delta_smiles|!
                dx_dis = round(abs(float(c["delta_x_raw"]) - dx_smi), 4) if (c["delta_x_raw"] is not None and dx_smi is not None) else None
                dy_dis = round(abs(float(c["delta_y_raw"]) - dy_smi), 4) if (c["delta_y_raw"] is not None and dy_smi is not None) else None
                delta_gmpo_smi = round(sb_smi.gmpo_score - float(p_smi["gmpo_score"]), 2)
                retention = (
                    round(100.0 * delta_gmpo_smi / max(float(c["delta_gmpo"]), 1e-6), 1)
                    if float(c["delta_gmpo"]) > 0.5
                    else None
                )
                audit_entry = {
                    "candidate_id": c["id"],
                    "mutation_label": c["mutation_label"],
                    "dual_gmpo": c["gmpo_score"],
                    "smiles_only_gmpo": sb_smi.gmpo_score,
                    "dual_delta_gmpo": c["delta_gmpo"],
                    "smiles_only_delta_gmpo": delta_gmpo_smi,
                    "smiles_only_stage_b_passed": sb_smi.passed,
                    "smiles_only_reject_codes": list(sb_smi.reject_codes),
                    "winners_curse_retention_pct": retention,
                    "epistemic_disagreement_dx": dx_dis,
                    "epistemic_disagreement_dy": dy_dis,
                }
                c["cross_track_audit"] = audit_entry
                cross_track_audit.append(audit_entry)

    if progress_callback:
        progress_callback({"phase": "done", "current_round": len(rounds_log), "total_rounds": n_rounds, "progress_pct": 100.0})

    final_status = "cancelled" if cancelled else ("partial" if partial_oracle_error is not None else "completed")
    return {
        "status": final_status,
        "partial_error": partial_oracle_error,
        "harness_mode": core_lock.harness_mode,
        "activity_unmodeled": core_lock.activity_unmodeled,
        "activity_warning": (
            "Activity/target potency was not modeled (developability_only mode; AA edits capped at <=2). "
            "Verify target binding experimentally or re-run with a Tier-A functional_endpoint / locked_positions."
            if core_lock.activity_unmodeled
            else None
        ),
        "no_editable_positions": no_editable_positions,
        "no_improvement_over_parent": len(final_top_k) == 0,
        "oracle_mode": adapter.mode,
        "oracle_is_surrogate_heuristic": adapter.mode != "v33_remote",
        "ablation_config": {
            "use_negative_memory": cfg.use_negative_memory,
            "use_stage_b_gate": cfg.use_ad_safety_gate,
            "use_critic": cfg.use_critic,
            "strategies": list(strategies),
        },
        "objective_x": {"endpoint": cfg.objective_x, "direction": dir_x, "tier": tier_info[cfg.objective_x].tier},
        "objective_y": {"endpoint": cfg.objective_y, "direction": dir_y, "tier": tier_info[cfg.objective_y].tier},
        "functional_endpoint": (
            {"endpoint": cfg.functional_endpoint, "direction": func_dir} if cfg.functional_endpoint else None
        ),
        "core_lock": {
            "locked_positions": sorted(core_lock.locked_positions),
            "lock_disulfides": core_lock.lock_disulfides,
            "locked_smarts": core_lock.locked_smarts,
            "max_total_aa_edits": core_lock.max_total_aa_edits,
            "max_total_chem_edits": core_lock.max_total_chem_edits,
            "sequence_smiles_mismatch_reason": core_lock.sequence_smiles_mismatch_reason,
        },
        "parent": parent_record,
        "parent_by_track": parent_summary_by_track,
        "rounds": rounds_log,
        "candidates": all_candidates,
        "pareto_frontier": frontier_final,
        "final_recommendations": final_top_k,
        "lineage_edges": lineage_edges,
        "negative_memory": [
            {
                "round": e.round_index,
                "mutation_label": e.mutation_label,
                "strategy": e.strategy,
                "reject_codes": e.reject_codes,
            }
            for e in memory.entries[:64]
        ],
        "cross_track_epistemic_audit": cross_track_audit,
        "audit_skipped_reason": audit_skipped_reason,
        "metrics": {
            "total_evaluated_candidates": len(all_candidates),
            "total_feasible_candidates": len(feasible_non_parent),
            "total_improved_candidates": len(improved_feasible),
            "total_oracle_errors": total_oracle_errors,
            "pareto_count": len(frontier_final),
            "feasible_pareto_hv": hv_final,
            "pareto_hv_reference_delta": {"ref_dx": ref_dx, "ref_dy": ref_dy},
            "best_gmpo": max((float(c["gmpo_score"]) for c in feasible_pool), default=float(parent_record["gmpo_score"])),
            "best_delta_gmpo": round(
                max((float(c["delta_gmpo"]) for c in feasible_non_parent), default=0.0), 2
            ),
            "evals_used": evals_used,
            "max_total_evals": max_total_evals,
            "budget_exhausted": budget_exhausted,
            "cancelled": cancelled,
            "runtime_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        },
    }
