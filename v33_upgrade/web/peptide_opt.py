"""PeptideOpt: RDKit-Sanitized Peptide Modification Enumeration & Multi-Objective Pareto Engine (P1-5).

Supports 5 chemically valid, RDKit-sanitized medicinal chemistry strategies:
  1. `ala_scan`            — Alanine scanning across representative sequence positions
  2. `point_mutation`      — Targeted polarity/charge/rigidity substitutions (K/R/E/A/F/P)
  3. `n_methylation`       — Backbone amide N-methylation on non-proline secondary amides via RWMol
  4. `cyclization`         — Head-to-tail backbone lactam OR Cys-Cys disulfide cyclization
  5. `lipidation_c16_c18`  — Lys-epsilon C16-monoacyl-γGlu and C18/C16-diacid-γGlu lipidation

Evaluates Parent + deduplicated variants through OmniPanelAdapter, computes exact 2D Pareto
dominance on (objective_x, objective_y), checks safety guardrails, and scores Peptide-MPO [0, 100].
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

from rdkit import Chem, RDLogger

from .adapter import ENDPOINT_UI_META, OmniPanelAdapter
from .batch_mpo import compute_peptide_mpo
from .chem_utils import InputValidationError, ResolvedInput, parse_smiles_safe, sequence_to_mol
from .schemas import (
    AssayConditions,
    BatchParsedRow,
    EndpointPrediction,
    PeptideOptObjectiveMeta,
    PeptideOptRequest,
    PeptideOptResponse,
    PeptideOptVariant,
)

RDLogger.DisableLog("rdApp.*")

# Compiled RDKit SMARTS for deterministic chemical modification sites
_PAT_SEC_AMIDE = Chem.MolFromSmarts("[NX3;H1]([#6])[CX3](=O)")
_PAT_FREE_THIOL = Chem.MolFromSmarts("[SX2H1]")
_PAT_N_TERM_AMINE = Chem.MolFromSmarts("[NX3;H2][CH1,CH2][CX3](=O)")
# Atom 0 = hydroxyl O, Atom 1 = carbonyl C (invariant across both specific and generic patterns)
_PAT_C_TERM_ACID = Chem.MolFromSmarts("[OX2H1][CX3](=O)[CH1,CH2][NX3]")
_PAT_GENERIC_CARBOXYL = Chem.MolFromSmarts("[OX2H1][CX3]=O")
_PAT_LYS_EPS_AMINE = Chem.MolFromSmarts("[NX3;H2][CH2][CH2][CH2][CH2][CH]([NX3])[CX3](=O)")
_PAT_PRIM_AMINE = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(NS=O)]")

# Lipidation acyl-γGlu fragments with dummy attachment point [*] on the γ-carboxyl carbonyl
_LIPID_FRAGMENTS: list[tuple[str, str, str]] = [
    (
        "C18-diacid-γGlu",
        "OC(=O)CCCCCCCCCCCCCCCCC(=O)N[C@@H](C(=O)O)CCC(=O)[*]",
        "Semaglutide-like C18 α,ω-fatty-diacid-γGlu albumin-binding lipidation",
    ),
    (
        "C16-monoacyl-γGlu",
        "CCCCCCCCCCCCCCCC(=O)N[C@@H](C(=O)O)CCC(=O)[*]",
        "Liraglutide-like C16 palmitoyl-γGlu albumin-binding lipidation",
    ),
    (
        "C16-diacid-γGlu",
        "OC(=O)CCCCCCCCCCCCCCC(=O)N[C@@H](C(=O)O)CCC(=O)[*]",
        "Degludec-like C16 α,ω-fatty-diacid-γGlu depot/albumin lipidation",
    ),
]

# Targeted charge / polarity / conformation substitutions for point_mutation
_TARGETED_SWAPS: dict[str, list[str]] = {
    "K": ["E", "A", "R"],
    "R": ["K", "A", "E"],
    "W": ["F", "A"],
    "F": ["A", "P"],
    "Y": ["F", "A"],
    "L": ["A", "P"],
    "I": ["A", "P"],
    "V": ["A", "P"],
    "E": ["K", "A"],
    "D": ["E", "A"],
    "G": ["P", "A"],
    "S": ["F", "A"],
    "T": ["A", "P"],
    "N": ["A", "E"],
    "Q": ["E", "K"],
    "H": ["K", "F"],
    "M": ["L", "A"],
    "P": ["A", "F"],
    "A": ["F", "P", "K"],
    "C": ["A"],
}


@dataclass
class _CandidateSpec:
    strategy: str
    mutation_label: str
    description: str
    sequence: Optional[str]
    smiles: str  # canonical, RDKit-sanitized SMILES


def _sanitize_and_canonicalize(mol: Chem.Mol) -> Optional[str]:
    """Sanitize an edited RDKit Mol and verify round-trip SMILES validity (single connected fragment)."""
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1:
            return None
        smi = Chem.MolToSmiles(m, canonical=True)
        chk = Chem.MolFromSmiles(smi)
        if chk is None or chk.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(chk, canonical=True)
    except Exception:
        return None


def _evenly_spaced_indices(indices: list[int], k: int) -> list[int]:
    if len(indices) <= k:
        return list(indices)
    if k <= 1:
        return [indices[len(indices) // 2]]
    step = (len(indices) - 1) / float(k - 1)
    chosen: list[int] = []
    seen: set[int] = set()
    for i in range(k):
        idx = indices[round(i * step)]
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)
    return chosen


def _gen_ala_scan(
    sequence: Optional[str],
    parent_can: str,
    limit: int = 10,
    locked_positions: Optional[set[int]] = None,
    lock_disulfides: bool = False,
) -> tuple[list[_CandidateSpec], Optional[str]]:
    if not sequence or len(sequence) < 3:
        return [], "Requires canonical 20-AA sequence (length >= 3)"
    if len(sequence) > 65:
        return [], "Canonical sequence exceeds 65 AA RDKit atomic graph limit"
    locked = locked_positions or set()
    non_ala = [
        i
        for i, aa in enumerate(sequence)
        if aa != "A" and (i + 1) not in locked and not (lock_disulfides and aa == "C")
    ]
    if not non_ala:
        return [], "No unlocked non-Alanine positions available"
    positions = _evenly_spaced_indices(non_ala, limit)
    out: list[_CandidateSpec] = []
    for pos in positions:
        orig = sequence[pos]
        mut_seq = sequence[:pos] + "A" + sequence[pos + 1 :]
        mol = sequence_to_mol(mut_seq)
        if mol is None:
            continue
        smi = _sanitize_and_canonicalize(mol)
        if smi and smi != parent_can:
            out.append(
                _CandidateSpec(
                    strategy="ala_scan",
                    mutation_label=f"{orig}{pos + 1}A",
                    description=f"Alanine scan at position {pos + 1} ({orig}→A)",
                    sequence=mut_seq,
                    smiles=smi,
                )
            )
    return out, (None if out else "No valid Ala-scan variants generated")


def _gen_point_mutations(
    sequence: Optional[str],
    parent_can: str,
    limit: int = 10,
    locked_positions: Optional[set[int]] = None,
    lock_disulfides: bool = False,
) -> tuple[list[_CandidateSpec], Optional[str]]:
    if not sequence or len(sequence) < 3:
        return [], "Requires canonical 20-AA sequence (length >= 3)"
    if len(sequence) > 65:
        return [], "Canonical sequence exceeds 65 AA RDKit atomic graph limit"
    locked = locked_positions or set()

    # Prioritize charged (K/R/E/D), aromatic (W/F/Y), and flexible (G/L/S) unlocked positions
    priority_pos = [
        i
        for i, aa in enumerate(sequence)
        if aa in "KRWFYEDGLS" and (i + 1) not in locked and not (lock_disulfides and aa == "C")
    ]
    if not priority_pos:
        priority_pos = [
            i for i, aa in enumerate(sequence) if (i + 1) not in locked and not (lock_disulfides and aa == "C")
        ]
    if not priority_pos:
        return [], "All sequence positions are locked"
    positions = _evenly_spaced_indices(priority_pos, min(len(priority_pos), max(limit, 8)))

    out: list[_CandidateSpec] = []
    seen_seqs: set[str] = {sequence}
    for swap_rank in (0, 1):
        for pos in positions:
            if len(out) >= limit:
                break
            orig = sequence[pos]
            targets = _TARGETED_SWAPS.get(orig, ["A", "F", "K", "E", "P"])
            if swap_rank >= len(targets):
                continue
            tgt = targets[swap_rank]
            if tgt == orig or (tgt == "A" and orig != "A" and swap_rank == 1):
                continue
            mut_seq = sequence[:pos] + tgt + sequence[pos + 1 :]
            if mut_seq in seen_seqs:
                continue
            mol = sequence_to_mol(mut_seq)
            if mol is None:
                continue
            smi = _sanitize_and_canonicalize(mol)
            if smi and smi != parent_can:
                seen_seqs.add(mut_seq)
                out.append(
                    _CandidateSpec(
                        strategy="point_mutation",
                        mutation_label=f"{orig}{pos + 1}{tgt}",
                        description=f"Targeted polarity/rigidity point mutation at pos {pos + 1} ({orig}→{tgt})",
                        sequence=mut_seq,
                        smiles=smi,
                    )
                )
    return out, (None if out else "No valid point mutations generated")


def _n_methylate_sites(mol: Chem.Mol, n_indices: list[int]) -> Optional[str]:
    try:
        rw = Chem.RWMol(mol)
        for n_idx in n_indices:
            atom = rw.GetAtomWithIdx(n_idx)
            if atom.GetAtomicNum() != 7 or atom.GetTotalNumHs() < 1:
                return None
            c_idx = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        return _sanitize_and_canonicalize(rw.GetMol())
    except Exception:
        return None


def _gen_n_methylation(
    mol: Optional[Chem.Mol],
    parent_can: str,
    limit: int = 8,
    locked_atom_indices: Optional[set[int]] = None,
) -> tuple[list[_CandidateSpec], Optional[str]]:
    if mol is None:
        return [], "Requires RDKit atomic structure (SMILES or <=80 AA sequence)"
    locked_atoms = locked_atom_indices or set()
    matches = mol.GetSubstructMatches(_PAT_SEC_AMIDE)
    # Exclude 5-membered proline ring nitrogens and locked core atoms
    candidate_n: list[int] = []
    seen_n: set[int] = set()
    for m in matches:
        n_idx = int(m[0])
        if n_idx in seen_n or n_idx in locked_atoms:
            continue
        atom = mol.GetAtomWithIdx(n_idx)
        if atom.GetTotalNumHs() >= 1 and not atom.IsInRingSize(5):
            seen_n.add(n_idx)
            candidate_n.append(n_idx)
    if not candidate_n:
        return [], "No available unlocked non-proline secondary amide NH sites for N-methylation"

    selected = _evenly_spaced_indices(candidate_n, max(1, limit - 1))
    out: list[_CandidateSpec] = []
    for k, n_idx in enumerate(selected, start=1):
        smi = _n_methylate_sites(mol, [n_idx])
        if smi and smi != parent_can:
            out.append(
                _CandidateSpec(
                    strategy="n_methylation",
                    mutation_label=f"N-Me (amide #{k})",
                    description=f"Backbone secondary amide N-methylation at nitrogen atom {n_idx} (reduces 1 HBD)",
                    sequence=None,
                    smiles=smi,
                )
            )
    # Also add one dual N-methylated analog if >= 2 amide sites exist
    if len(selected) >= 2 and len(out) < limit:
        pair = [selected[0], selected[-1]]
        smi2 = _n_methylate_sites(mol, pair)
        if smi2 and smi2 != parent_can:
            out.append(
                _CandidateSpec(
                    strategy="n_methylation",
                    mutation_label="Di-N-Me (2 amides)",
                    description=f"Dual backbone amide N-methylation at nitrogens {pair[0]} & {pair[1]} (reduces 2 HBDs)",
                    sequence=None,
                    smiles=smi2,
                )
            )
    return out, (None if out else "N-methylation did not yield valid sanitized variants")


def _cyclize_disulfide(mol: Chem.Mol, s1: int, s2: int) -> Optional[str]:
    try:
        rw = Chem.RWMol(mol)
        a1 = rw.GetAtomWithIdx(s1)
        a2 = rw.GetAtomWithIdx(s2)
        if a1.GetAtomicNum() != 16 or a2.GetAtomicNum() != 16:
            return None
        if rw.GetBondBetweenAtoms(s1, s2) is not None:
            return None
        rw.AddBond(s1, s2, Chem.BondType.SINGLE)
        a1.SetNumExplicitHs(0)
        a2.SetNumExplicitHs(0)
        a1.SetNoImplicit(False)
        a2.SetNoImplicit(False)
        return _sanitize_and_canonicalize(rw.GetMol())
    except Exception:
        return None


def _cyclize_head_to_tail(mol: Chem.Mol) -> Optional[str]:
    """Form head-to-tail backbone lactam between N-terminal primary amine and C-terminal carboxyl."""
    n_matches = mol.GetSubstructMatches(_PAT_N_TERM_AMINE)
    c_matches = mol.GetSubstructMatches(_PAT_C_TERM_ACID)
    if not c_matches:
        c_matches = mol.GetSubstructMatches(_PAT_GENERIC_CARBOXYL)
    if not n_matches or not c_matches:
        return None

    # Choose the pair with the largest topological separation so we close the main backbone loop
    best_pair: Optional[tuple[int, int, int, int]] = None  # (dist, n_idx, c_idx, oh_idx)
    dm = Chem.GetDistanceMatrix(mol)
    for nm in n_matches:
        n_idx = int(nm[0])
        for cm in c_matches:
            oh_idx = int(cm[0])
            c_idx = int(cm[1])
            dist = int(dm[n_idx][c_idx])
            if dist >= 10:
                if best_pair is None or dist > best_pair[0]:
                    best_pair = (dist, n_idx, c_idx, oh_idx)
    if best_pair is None:
        return None

    _, n_idx, c_idx, oh_idx = best_pair
    try:
        rw = Chem.RWMol(mol)
        if rw.GetBondBetweenAtoms(n_idx, c_idx) is not None:
            return None
        rw.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
        n_atom = rw.GetAtomWithIdx(n_idx)
        n_atom.SetNumExplicitHs(0)
        n_atom.SetNoImplicit(False)
        rw.RemoveAtom(oh_idx)
        res_mol = rw.GetMol()
        smi = _sanitize_and_canonicalize(res_mol)
        if not smi:
            return None
        chk = Chem.MolFromSmiles(smi)
        if chk is None:
            return None
        # Confirm a macrocycle (>=12 ring atoms) was formed
        if not any(len(r) >= 12 for r in chk.GetRingInfo().AtomRings()):
            return None
        return smi
    except Exception:
        return None


def _gen_cyclization(
    mol: Optional[Chem.Mol],
    sequence: Optional[str],
    parent_can: str,
    limit: int = 4,
    locked_positions: Optional[set[int]] = None,
    locked_atom_indices: Optional[set[int]] = None,
    lock_disulfides: bool = False,
) -> tuple[list[_CandidateSpec], Optional[str]]:
    if mol is None:
        return [], "Requires RDKit atomic structure"

    locked_pos = locked_positions or set()
    locked_atoms = locked_atom_indices or set()
    out: list[_CandidateSpec] = []
    has_macro = any(len(r) >= 12 for r in mol.GetRingInfo().AtomRings())

    # 1. Head-to-tail backbone lactam cyclization (skip if N-term or C-term is locked)
    termini_locked = bool(sequence and (1 in locked_pos or len(sequence) in locked_pos))
    if not has_macro and not termini_locked:
        htt_smi = _cyclize_head_to_tail(mol)
        if htt_smi and htt_smi != parent_can:
            out.append(
                _CandidateSpec(
                    strategy="cyclization",
                    mutation_label="Head-to-Tail Cyclo",
                    description="Head-to-tail backbone amide macrocyclization (N-term to C-term lactam)",
                    sequence=None,
                    smiles=htt_smi,
                )
            )

    # 2. Existing free Cys-Cys disulfide bridge (ALWAYS respect explicitly locked sulfur atoms)
    thiol_matches = [
        int(m[0])
        for m in mol.GetSubstructMatches(_PAT_FREE_THIOL)
        if int(m[0]) not in locked_atoms
    ]
    if len(thiol_matches) >= 2:
        dm = Chem.GetDistanceMatrix(mol)
        best_ss: Optional[tuple[int, int, int]] = None
        for i in range(len(thiol_matches)):
            for j in range(i + 1, len(thiol_matches)):
                s1, s2 = thiol_matches[i], thiol_matches[j]
                d = int(dm[s1][s2])
                if d >= 8 and (best_ss is None or d > best_ss[0]):
                    best_ss = (d, s1, s2)
        if best_ss is not None:
            ss_smi = _cyclize_disulfide(mol, best_ss[1], best_ss[2])
            if ss_smi and ss_smi != parent_can:
                out.append(
                    _CandidateSpec(
                        strategy="cyclization",
                        mutation_label="Cys-Cys Disulfide",
                        description="Intramolecular oxidative Cys-Cys disulfide bridge cyclization",
                        sequence=None,
                        smiles=ss_smi,
                    )
                )

    # 3. For canonical linear peptides without >=2 free Cys, synthesize a Cys-flanked disulfide macrocycle (chemical leaf)
    # BOTH branches modify N-term (pos 1) and C-term (pos L) — FORBIDDEN if either terminus is locked!
    if (
        len(out) < limit
        and sequence
        and 4 <= len(sequence) <= 45
        and not has_macro
        and len(thiol_matches) < 2
        and not termini_locked
    ):
        cys_seq = f"C{sequence}C" if len(sequence) <= 30 else ("C" + sequence[1:-1] + "C")
        cys_mol = sequence_to_mol(cys_seq)
        if cys_mol is not None:
            t_idxs = [int(m[0]) for m in cys_mol.GetSubstructMatches(_PAT_FREE_THIOL)]
            if len(t_idxs) >= 2:
                ss_smi = _cyclize_disulfide(cys_mol, t_idxs[0], t_idxs[-1])
                if ss_smi and ss_smi != parent_can:
                    out.append(
                        _CandidateSpec(
                            strategy="cyclization",
                            mutation_label="c[C-seq-C] Disulfide",
                            description="Engineered terminal Cys-Cys disulfide-bridged cyclic analog (chemical leaf)",
                            sequence=None,
                            smiles=ss_smi,
                        )
                    )

    if not out:
        reason = "Parent molecule is already macrocyclic with no free thiol pairs" if has_macro else "No compatible unlocked termini or free Cys pairs found for cyclization"
        return [], reason
    return out[:limit], None


def _attach_lipid_fragment(mol: Chem.Mol, amine_idx: int, frag_smiles: str) -> Optional[str]:
    """Attach a fatty-acyl-γGlu fragment (`[*]`) onto a primary amine nitrogen via an amide bond."""
    try:
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return None
        combo = Chem.CombineMols(mol, frag)
        offset = mol.GetNumAtoms()
        dummy_idx: Optional[int] = None
        carbonyl_idx: Optional[int] = None
        for a in combo.GetAtoms():
            if a.GetIdx() >= offset and a.GetAtomicNum() == 0:
                dummy_idx = a.GetIdx()
                nbs = a.GetNeighbors()
                if len(nbs) == 1:
                    carbonyl_idx = nbs[0].GetIdx()
                break
        if dummy_idx is None or carbonyl_idx is None:
            return None

        rw = Chem.RWMol(combo)
        rw.AddBond(amine_idx, carbonyl_idx, Chem.BondType.SINGLE)
        n_atom = rw.GetAtomWithIdx(amine_idx)
        n_atom.SetNumExplicitHs(0)
        n_atom.SetNoImplicit(False)
        rw.RemoveAtom(dummy_idx)
        return _sanitize_and_canonicalize(rw.GetMol())
    except Exception:
        return None


def _gen_lipidation(
    mol: Optional[Chem.Mol],
    parent_can: str,
    limit: int = 6,
    locked_atom_indices: Optional[set[int]] = None,
) -> tuple[list[_CandidateSpec], Optional[str]]:
    if mol is None:
        return [], "Requires RDKit atomic structure"

    locked_atoms = locked_atom_indices or set()
    # Prefer unlocked Lysine ε-amines first (clinical standard: Semaglutide / Liraglutide / Tirzepatide)
    lys_sites = [int(m[0]) for m in mol.GetSubstructMatches(_PAT_LYS_EPS_AMINE) if int(m[0]) not in locked_atoms]
    site_kind = "Lys-ε"
    sites = _evenly_spaced_indices(lys_sites, 2)
    if not sites:
        prim_sites = [int(m[0]) for m in mol.GetSubstructMatches(_PAT_PRIM_AMINE) if int(m[0]) not in locked_atoms]
        sites = _evenly_spaced_indices(prim_sites, 2)
        site_kind = "N-term/amine"

    if not sites:
        return [], "No unlocked Lys ε-amine or primary amine handle available for C16/C18 lipidation"

    out: list[_CandidateSpec] = []
    for s_rank, n_idx in enumerate(sites, start=1):
        for frag_label, frag_smi, frag_desc in _LIPID_FRAGMENTS:
            if len(out) >= limit:
                break
            smi = _attach_lipid_fragment(mol, n_idx, frag_smi)
            if smi and smi != parent_can:
                suffix = f" ({site_kind} #{s_rank})" if len(sites) > 1 else f" ({site_kind})"
                out.append(
                    _CandidateSpec(
                        strategy="lipidation_c16_c18",
                        mutation_label=f"{frag_label}{suffix}",
                        description=f"{frag_desc} conjugated at {site_kind} atom {n_idx}",
                        sequence=None,
                        smiles=smi,
                    )
                )
    return out, (None if out else "Lipidation conjugation did not produce valid sanitized variants")


def _sig6(val: float) -> float:
    return float(f"{val:.6g}")


def _extract_objective(
    preds_by_ep: dict[str, EndpointPrediction],
    endpoint: str,
) -> tuple[Optional[float], Optional[float], str, str]:
    p = preds_by_ep.get(endpoint)
    if p is None:
        return None, None, "N/A", "borderline"
    val = float(p.value) if p.value is not None else None
    raw = float(p.raw_value) if p.raw_value is not None else val
    return val, raw, p.value_display, p.ad_status if p.ad_status in ("in_domain", "borderline", "ood") else "in_domain"


def _evaluate_safety_guardrails(preds_by_ep: dict[str, EndpointPrediction]) -> tuple[bool, list[str]]:
    flags: list[str] = []
    hem = preds_by_ep.get("hemolysis_cls")
    if hem is not None and hem.value is not None and float(hem.value) >= 0.50:
        flags.append(f"High hemolysis risk ({hem.value_display})")
    tox = preds_by_ep.get("overall_peptide_toxicity")
    if tox is not None and tox.value is not None and float(tox.value) >= 0.55:
        flags.append(f"Elevated peptide toxicity ({tox.value_display})")
    cyto = preds_by_ep.get("cytotoxicity_cls")
    if cyto is not None and cyto.value is not None and float(cyto.value) >= 0.55:
        flags.append(f"Cytotoxicity alert ({cyto.value_display})")
    return len(flags) == 0, flags


def _compute_pareto_flags(
    all_items: list[PeptideOptVariant],
    dir_x: str,
    dir_y: str,
) -> list[PeptideOptVariant]:
    """Compute exact 2D non-dominated Pareto set on (x_raw, y_raw) and return frontier sorted by x_raw."""
    sx = 1.0 if dir_x == "max" else -1.0
    sy = 1.0 if dir_y == "max" else -1.0
    eps = 1e-9

    valid = [it for it in all_items if it.x_raw is not None and it.y_raw is not None]
    frontier: list[PeptideOptVariant] = []
    for cand in valid:
        cx = sx * float(cand.x_raw)  # type: ignore[arg-type]
        cy = sy * float(cand.y_raw)  # type: ignore[arg-type]
        dominated = False
        for other in valid:
            if other.id == cand.id:
                continue
            ox = sx * float(other.x_raw)  # type: ignore[arg-type]
            oy = sy * float(other.y_raw)  # type: ignore[arg-type]
            if ox >= cx - eps and oy >= cy - eps and (ox > cx + eps or oy > cy + eps):
                dominated = True
                break
        cand.is_pareto_optimal = not dominated
        if not dominated:
            frontier.append(cand)

    # Sort Pareto frontier left-to-right by x_raw for crisp SVG polyline rendering
    frontier.sort(key=lambda v: (float(v.x_raw or 0.0), float(v.y_raw or 0.0)))
    return frontier


def run_peptide_opt(
    adapter: OmniPanelAdapter,
    req: PeptideOptRequest,
) -> PeptideOptResponse:
    """Enumerate RDKit-valid peptide modifications, evaluate via OmniPanelAdapter, and compute Pareto frontier."""
    t0 = time.perf_counter()

    # 1. Validate endpoints against manifest catalog
    eps_manifest = adapter.manifest.get("endpoints") or {}
    for axis_name, obj in (("objective_x", req.objective_x), ("objective_y", req.objective_y)):
        if eps_manifest and obj.endpoint not in eps_manifest:
            raise InputValidationError(
                f"Unknown {axis_name} endpoint '{obj.endpoint}'. Must be one of the {len(eps_manifest)} supported V33 endpoints."
            )

    # 2. Resolve parent peptide
    resolved: ResolvedInput = adapter.resolve_query(
        query=req.query,
        sequence=req.sequence,
        smiles=req.smiles,
    )
    if resolved.mol is None:
        raise InputValidationError(
            "PeptideOpt requires an RDKit-parseable molecular structure (SMILES or canonical sequence <= 80 AA)."
        )
    parent_can = _sanitize_and_canonicalize(resolved.mol)
    if not parent_can:
        raise InputValidationError("Parent molecule failed RDKit sanitization.")

    # Check if sequence really corresponds to an unmodified linear parent (for sequence-based strategies)
    has_macro = any(len(r) >= 12 for r in resolved.mol.GetRingInfo().AtomRings())
    seq_for_mut = resolved.sequence if (resolved.sequence and not has_macro and resolved.resolved_type in ("sequence", "dual", "drug_name")) else None
    if seq_for_mut:
        # Only use sequence mutations when the parent is unmodified linear or drug_name without lipid/non-canonical scaffold
        seq_mol = sequence_to_mol(seq_for_mut)
        seq_can = _sanitize_and_canonicalize(seq_mol) if seq_mol is not None else None
        if seq_can != parent_can and resolved.smiles:
            # e.g., Semaglutide/Liraglutide have modified covalent SMILES -> still allow seq-based ala/point scan ONLY if explicitly requested via sequence-only, otherwise keep covalent modifications honest
            if resolved.resolved_type == "smiles":
                seq_for_mut = None

    # 3. Generate per-strategy candidate pools
    requested_strategies = list(dict.fromkeys(req.strategies))
    pools: dict[str, list[_CandidateSpec]] = {}
    strategies_skipped: list[dict[str, str]] = []
    max_v = min(max(1, int(req.max_variants)), 32)
    per_strat_cap = max(4, math.ceil(max_v / max(1, len(requested_strategies))) + 3)

    for strat in requested_strategies:
        if strat == "ala_scan":
            cands, skip_reason = _gen_ala_scan(seq_for_mut, parent_can, limit=per_strat_cap)
        elif strat == "point_mutation":
            cands, skip_reason = _gen_point_mutations(seq_for_mut, parent_can, limit=per_strat_cap)
        elif strat == "n_methylation":
            cands, skip_reason = _gen_n_methylation(resolved.mol, parent_can, limit=per_strat_cap)
        elif strat == "cyclization":
            cands, skip_reason = _gen_cyclization(resolved.mol, seq_for_mut, parent_can, limit=min(4, per_strat_cap))
        elif strat == "lipidation_c16_c18":
            cands, skip_reason = _gen_lipidation(resolved.mol, parent_can, limit=min(6, per_strat_cap))
        else:
            cands, skip_reason = [], f"Unsupported strategy '{strat}'"

        if cands:
            pools[strat] = cands
        elif skip_reason:
            strategies_skipped.append({"strategy": strat, "reason": skip_reason})

    # 4. Round-robin select & deduplicate by canonical SMILES up to max_variants
    selected_specs: list[_CandidateSpec] = []
    seen_smiles: set[str] = {parent_can}
    total_generated = sum(len(v) for v in pools.values())
    active_strats = list(pools.keys())
    cursors = {s: 0 for s in active_strats}

    while len(selected_specs) < max_v and active_strats:
        progressed = False
        for s in list(active_strats):
            pool = pools[s]
            while cursors[s] < len(pool):
                cand = pool[cursors[s]]
                cursors[s] += 1
                if cand.smiles not in seen_smiles:
                    seen_smiles.add(cand.smiles)
                    selected_specs.append(cand)
                    progressed = True
                    break
            if cursors[s] >= len(pool):
                active_strats.remove(s)
            if len(selected_specs) >= max_v:
                break
        if not progressed:
            break

    # 5. Evaluate Parent + all selected variants in a single vectorized adapter.predict_batch call
    include_tier2 = any(
        int((eps_manifest.get(ep) or {}).get("tier", 1)) == 2
        for ep in (req.objective_x.endpoint, req.objective_y.endpoint)
    )
    batch_rows: list[BatchParsedRow] = [
        BatchParsedRow(
            index=0,
            name=resolved.query_name,
            detected_type=resolved.resolved_type,
            sequence=resolved.sequence,
            smiles=parent_can,
            valid=True,
        )
    ]
    for idx, spec in enumerate(selected_specs, start=1):
        batch_rows.append(
            BatchParsedRow(
                index=idx,
                name=spec.mutation_label,
                detected_type="dual" if spec.sequence else "smiles",
                sequence=spec.sequence,
                smiles=spec.smiles,
                valid=True,
            )
        )

    cond = req.conditions or AssayConditions()
    batch_res = adapter.predict_batch(batch_rows, include_tier2=include_tier2, conditions=cond)
    parent_bm = batch_res[0]
    if parent_bm.status != "ok":
        raise InputValidationError(f"Parent evaluation failed: {parent_bm.error}")

    parent_preds = {p.endpoint: p for p in parent_bm.predictions}
    px_val, px_raw, px_disp, px_ad = _extract_objective(parent_preds, req.objective_x.endpoint)
    py_val, py_raw, py_disp, py_ad = _extract_objective(parent_preds, req.objective_y.endpoint)
    p_mpo = compute_peptide_mpo(parent_bm.predictions, weights=req.mpo_weights, mechanism_anchor=parent_bm.mechanism_anchor)
    p_safe, p_flags = _evaluate_safety_guardrails(parent_preds)

    parent_var = PeptideOptVariant(
        id="parent",
        name=resolved.query_name,
        strategy="parent",
        mutation_label="Parent",
        description=f"Original parent molecule ({resolved.query_name})",
        sequence=resolved.sequence,
        smiles=parent_can,
        is_parent=True,
        is_pareto_optimal=False,
        x_value=px_val,
        x_raw=px_raw,
        x_display=px_disp,
        y_value=py_val,
        y_raw=py_raw,
        y_display=py_disp,
        delta_x=0.0,
        delta_y=0.0,
        delta_x_raw=0.0,
        delta_y_raw=0.0,
        delta_x_pct=0.0,
        delta_y_pct=0.0,
        ad_status=p_mpo["ad_overall"],  # type: ignore[arg-type]
        ad_status_x=px_ad,  # type: ignore[arg-type]
        ad_status_y=py_ad,  # type: ignore[arg-type]
        mpo_score=p_mpo["peptide_mpo_score"],
        safety_guardrail_pass=p_safe,
        safety_flags=p_flags,
        mw=parent_bm.mw,
        logp=parent_bm.logp,
        tpsa=parent_bm.physchem.tpsa if parent_bm.physchem else None,
        hbd=parent_bm.physchem.hbd if parent_bm.physchem else None,
        net_charge_ph74=parent_bm.net_charge_ph74,
        key_metrics=parent_bm.key_metrics,
    )

    variants: list[PeptideOptVariant] = []
    for idx, (spec, bm) in enumerate(zip(selected_specs, batch_res[1:]), start=1):
        if bm.status != "ok":
            continue
        v_preds = {p.endpoint: p for p in bm.predictions}
        vx_val, vx_raw, vx_disp, vx_ad = _extract_objective(v_preds, req.objective_x.endpoint)
        vy_val, vy_raw, vy_disp, vy_ad = _extract_objective(v_preds, req.objective_y.endpoint)
        v_mpo = compute_peptide_mpo(bm.predictions, weights=req.mpo_weights, mechanism_anchor=bm.mechanism_anchor)
        v_safe, v_flags = _evaluate_safety_guardrails(v_preds)

        dx = _sig6(vx_val - px_val) if (vx_val is not None and px_val is not None) else None
        dy = _sig6(vy_val - py_val) if (vy_val is not None and py_val is not None) else None
        dx_raw = round(vx_raw - px_raw, 5) if (vx_raw is not None and px_raw is not None) else None
        dy_raw = round(vy_raw - py_raw, 5) if (vy_raw is not None and py_raw is not None) else None
        dx_pct = round(((vx_val - px_val) / abs(px_val)) * 100.0, 2) if (vx_val is not None and px_val is not None and abs(px_val) > 1e-12) else None
        dy_pct = round(((vy_val - py_val) / abs(py_val)) * 100.0, 2) if (vy_val is not None and py_val is not None and abs(py_val) > 1e-12) else None

        variants.append(
            PeptideOptVariant(
                id=f"var_{idx:02d}",
                name=spec.mutation_label,
                strategy=spec.strategy,
                mutation_label=spec.mutation_label,
                description=spec.description,
                sequence=spec.sequence,
                smiles=spec.smiles,
                is_parent=False,
                is_pareto_optimal=False,
                x_value=vx_val,
                x_raw=vx_raw,
                x_display=vx_disp,
                y_value=vy_val,
                y_raw=vy_raw,
                y_display=vy_disp,
                delta_x=dx,
                delta_y=dy,
                delta_x_raw=dx_raw,
                delta_y_raw=dy_raw,
                delta_x_pct=dx_pct,
                delta_y_pct=dy_pct,
                ad_status=v_mpo["ad_overall"],  # type: ignore[arg-type]
                ad_status_x=vx_ad,  # type: ignore[arg-type]
                ad_status_y=vy_ad,  # type: ignore[arg-type]
                mpo_score=v_mpo["peptide_mpo_score"],
                safety_guardrail_pass=v_safe,
                safety_flags=v_flags,
                mw=bm.mw,
                logp=bm.logp,
                tpsa=bm.physchem.tpsa if bm.physchem else None,
                hbd=bm.physchem.hbd if bm.physchem else None,
                net_charge_ph74=bm.net_charge_ph74,
                key_metrics=bm.key_metrics,
            )
        )

    # 6. Compute exact 2D Pareto dominance across Parent + variants
    frontier = _compute_pareto_flags([parent_var] + variants, req.objective_x.direction, req.objective_y.direction)
    variants.sort(key=lambda v: (not v.is_pareto_optimal, not v.safety_guardrail_pass, -v.mpo_score, v.id))

    def _obj_meta(endpoint: str, direction: str) -> PeptideOptObjectiveMeta:
        ep_doc = eps_manifest.get(endpoint) or {}
        ui = ENDPOINT_UI_META.get(endpoint) or {}
        is_cls = ep_doc.get("task_type") == "classification"
        unit = "probability" if is_cls else str(ep_doc.get("linear_unit") or ep_doc.get("unit") or "")
        return PeptideOptObjectiveMeta(
            endpoint=endpoint,
            direction=direction,  # type: ignore[arg-type]
            display_en=ui.get("display_en") or str(ep_doc.get("display_en") or endpoint),
            display_zh=ui.get("display_zh") or str(ep_doc.get("display_zh") or endpoint),
            unit=unit,
            log10_space=bool(ep_doc.get("log10_space", False)),
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return PeptideOptResponse(
        parent=parent_var,
        variants=variants,
        pareto_frontier=frontier,
        objective_x=_obj_meta(req.objective_x.endpoint, req.objective_x.direction),
        objective_y=_obj_meta(req.objective_y.endpoint, req.objective_y.direction),
        strategies_used=list(pools.keys()),
        strategies_skipped=strategies_skipped,
        total_generated=total_generated,
        total_evaluated=len(variants),
        pareto_count=len(frontier),
        runtime_ms=elapsed_ms,
        adapter_mode=adapter.mode,
    )
