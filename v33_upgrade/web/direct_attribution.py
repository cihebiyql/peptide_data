"""Direct Ensemble Perturbation Attribution Engine (P2-8).

100% production-ensemble-faithful attribution across ALL 47 Tier-1 and Tier-2 endpoints,
solving the <1.0x anti-faithful E-GAT 2D-graph surrogate limitation on PAMPA, hemolysis_HC50,
and cytotoxicity_CC50:

  - Mode 1 (`residue_alanine_glycine_masking`):
      For unmodified canonical linear peptides, performs Leave-One-Out Ala/Gly residue masking
      at each position (A -> G, non-A -> A), calls the production ensemble via `adapter.predict_batch`,
      computes `delta_raw = pred_full_raw - pred_masked_raw`, and maps residue impacts directly
      onto the exact RDKit backbone/sidechain atom indices via `AtomPDBResidueInfo`.
  - Mode 2 (`chemical_substituent_pruning`):
      For cyclic, N-methylated, lipidated, or modified SMILES peptides, performs strictly chemically
      valid substituent/sidechain/N-methyl/lipid-tail pruning (replacing exocyclic groups with -H
      across non-ring single bonds and verifying single-fragment RDKit sanitization — NEVER creating
      broken-valence or disconnected subgraphs!) and maps `delta_raw` back onto the ablated heavy atoms.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .adapter import ENDPOINT_UI_META, OmniPanelAdapter, _TTLCache
from .chem_utils import InputValidationError, ResolvedInput, sequence_to_mol
from .schemas import (
    AssayConditions,
    BatchParsedRow,
    DirectAtomWeight,
    DirectAttributionRequest,
    DirectAttributionResponse,
    ResidueOrGroupImpact,
    TopAtomContribution,
)

RDLogger.DisableLog("rdApp.*")

# Amino acid 3-letter names for crisp residue labels
_AA_3 = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
}

_PAT_N_METHYL = Chem.MolFromSmarts("[NX3;H0]([CH3])[CX3](=O)")
_PAT_BACKBONE_CA = Chem.MolFromSmarts("[NX3]-[CH;X4]-[CX3](=O)")
_PAT_FATTY_AMIDE = Chem.MolFromSmarts("[CX3](=O)[CH2][CH2][CH2][CH2][CH2][CH2][CH2]")
_PAT_GUANIDINO = Chem.MolFromSmarts("NC(=N)N")
_PAT_INDOLE = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")
_PAT_PHENOL = Chem.MolFromSmarts("c1ccc(O)cc1")
_PAT_PHENYL = Chem.MolFromSmarts("c1ccccc1")
_PAT_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")

_DIRECT_ATTR_CACHE = _TTLCache(maxsize=256, ttl_seconds=3600.0)


@dataclass
class _PerturbationPlan:
    id: str
    label: str
    kind: str  # "residue" | "sidechain" | "n_methyl" | "lipid_tail" | "functional_group"
    position: Optional[int]
    original_residue: Optional[str]
    masked_to: Optional[str]
    perturbed_sequence: Optional[str]
    perturbed_smiles: str
    atom_indices: list[int]
    backbone_indices: list[int]


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _palette_diverging(t: float) -> tuple[float, float, float]:
    """Cool-neutral-warm perceptual diverging palette on [0, 1] centered at 0.50."""
    t = max(0.0, min(1.0, float(t)))
    stops = [
        (0.00, (0.14, 0.48, 0.72)),  # deep cobalt-blue (decreases endpoint)
        (0.32, (0.56, 0.80, 0.90)),  # soft sky
        (0.50, (0.95, 0.96, 0.97)),  # neutral ice
        (0.68, (0.98, 0.72, 0.32)),  # warm amber
        (1.00, (0.85, 0.22, 0.18)),  # crimson (increases endpoint)
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = (t - t0) / max(1e-8, t1 - t0)
            return (
                c0[0] + u * (c1[0] - c0[0]),
                c0[1] + u * (c1[1] - c0[1]),
                c0[2] + u * (c1[2] - c0[2]),
            )
    return stops[-1][1]


def _sanitize_single_frag(mol: Chem.Mol) -> Optional[str]:
    """Verify chemically valid single-fragment RDKit sanitization and round-trip SMILES."""
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        if m.GetNumAtoms() < 2 or len(Chem.GetMolFrags(m)) != 1:
            return None
        smi = Chem.MolToSmiles(m, canonical=True)
        chk = Chem.MolFromSmiles(smi)
        if chk is None or chk.GetNumAtoms() < 2 or len(Chem.GetMolFrags(chk)) != 1:
            return None
        return Chem.MolToSmiles(chk, canonical=True)
    except Exception:
        return None


def _prune_exocyclic_branch(mol: Chem.Mol, keep_atom: int, cut_atom: int) -> Optional[tuple[str, list[int]]]:
    """Cut a single non-ring bond (keep_atom, cut_atom), remove the branch containing cut_atom, replace with -H."""
    bond = mol.GetBondBetweenAtoms(keep_atom, cut_atom)
    if bond is None or bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
        return None

    # BFS from cut_atom without crossing back into keep_atom
    visited: set[int] = {cut_atom}
    q = deque([cut_atom])
    while q:
        cur = q.popleft()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            nidx = nb.GetIdx()
            if nidx == keep_atom:
                continue
            if nidx not in visited:
                visited.add(nidx)
                q.append(nidx)

    # Reject if cutting didn't split cleanly (e.g. cycle path back to keep_atom) or removes >= half the molecule
    n_total = mol.GetNumAtoms()
    if len(visited) == 0 or len(visited) >= max(n_total - 2, int(n_total * 0.60)):
        return None

    try:
        rw = Chem.RWMol(mol)
        # Remove branch atoms in descending index order so indices stay valid
        for aidx in sorted(visited, reverse=True):
            rw.RemoveAtom(aidx)
        # Re-adjust implicit hydrogens on the attachment atom (index shifted by count of removed atoms < keep_atom)
        shift = sum(1 for aidx in visited if aidx < keep_atom)
        new_keep = keep_atom - shift
        ka = rw.GetAtomWithIdx(new_keep)
        ka.SetNumExplicitHs(0)
        ka.SetNoImplicit(False)
        smi = _sanitize_single_frag(rw.GetMol())
        if not smi:
            return None
        return smi, sorted(visited)
    except Exception:
        return None


def _describe_substructure(mol: Chem.Mol, sub_atoms: list[int], default_prefix: str, idx_num: int) -> str:
    """Assign a readable medicinal-chemistry label to an ablated substituent fragment."""
    n = len(sub_atoms)
    if n == 1:
        sym = mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol()
        if sym == "C":
            return f"Methyl group #{idx_num}"
        if sym == "O":
            return f"Hydroxyl / Oxo #{idx_num}"
        return f"{sym} substituent #{idx_num}"

    try:
        sub_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms, canonical=True)
        sub_mol = Chem.MolFromSmiles(sub_smi)
    except Exception:
        sub_mol = None

    if sub_mol is not None:
        if sub_mol.HasSubstructMatch(_PAT_INDOLE):
            return f"Indole / Trp sidechain (#{idx_num})"
        if sub_mol.HasSubstructMatch(_PAT_GUANIDINO):
            return f"Guanidino / Arg sidechain (#{idx_num})"
        if sub_mol.HasSubstructMatch(_PAT_PHENOL):
            return f"Phenol / Tyr sidechain (#{idx_num})"
        if sub_mol.HasSubstructMatch(_PAT_PHENYL):
            return f"Aryl / Phe sidechain (#{idx_num})"
        if sub_mol.HasSubstructMatch(_PAT_CARBOXYL):
            return f"Carboxyl sidechain (#{idx_num}, {n} atoms)"
        syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in sub_atoms}
        if "S" in syms:
            return f"Thio / Sulfur sidechain (#{idx_num}, {n} atoms)"
        if "N" in syms and "O" not in syms:
            return f"Amine / Lys-like sidechain (#{idx_num}, {n} atoms)"
        if "N" in syms and "O" in syms:
            return f"Amide / Polar sidechain (#{idx_num}, {n} atoms)"
        if "O" in syms:
            return f"Hydroxyl / Ether group (#{idx_num}, {n} atoms)"
        if syms == {"C"}:
            return f"Hydrophobic alkyl ({n}C, #{idx_num})"
    return f"{default_prefix} #{idx_num} ({n} atoms)"


def _evenly_pick(items: list[Any], k: int) -> list[Any]:
    if len(items) <= k:
        return items
    if k <= 1:
        return [items[0]]
    step = (len(items) - 1) / float(k - 1)
    chosen = []
    seen = set()
    for i in range(k):
        idx = round(i * step)
        if idx not in seen:
            chosen.append(items[idx])
            seen.add(idx)
    return chosen


def _build_residue_masking_plans(
    mol: Chem.Mol,
    sequence: str,
    max_perturbations: int,
    exclude_positions: Optional[set[int]] = None,
) -> list[_PerturbationPlan]:
    """Build Leave-One-Out Ala/Gly residue masking plans with exact RDKit PDB residue atom mapping."""
    res_sidechain: dict[int, list[int]] = {}
    res_backbone: dict[int, list[int]] = {}
    backbone_names = {"N", "CA", "C", "O", "OXT"}
    excl = exclude_positions or set()

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        rnum = int(info.GetResidueNumber())
        aname = info.GetName().strip()
        if aname in backbone_names:
            res_backbone.setdefault(rnum, []).append(atom.GetIdx())
        else:
            res_sidechain.setdefault(rnum, []).append(atom.GetIdx())

    # Map unlocked positions 1..len(sequence)
    positions = [p for p in range(1, len(sequence) + 1) if p not in excl]
    if not positions:
        return []
    selected_pos = _evenly_pick(positions, max_perturbations)

    plans: list[_PerturbationPlan] = []
    for pos in selected_pos:
        orig_aa = sequence[pos - 1]
        masked_aa = "G" if orig_aa == "A" else "A"
        mut_seq = sequence[: pos - 1] + masked_aa + sequence[pos:]
        mut_mol = sequence_to_mol(mut_seq)
        if mut_mol is None:
            continue
        mut_smi = _sanitize_single_frag(mut_mol)
        if not mut_smi:
            continue
        sc_atoms = res_sidechain.get(pos, [])
        bb_atoms = res_backbone.get(pos, [])
        all_atoms = sorted(set(sc_atoms + bb_atoms))
        plans.append(
            _PerturbationPlan(
                id=f"pos_{pos}_{orig_aa}",
                label=f"{_AA_3.get(orig_aa, orig_aa)}{pos} → {_AA_3.get(masked_aa, masked_aa)}",
                kind="residue",
                position=pos,
                original_residue=orig_aa,
                masked_to=masked_aa,
                perturbed_sequence=mut_seq,
                perturbed_smiles=mut_smi,
                atom_indices=all_atoms,
                backbone_indices=sorted(bb_atoms),
            )
        )
    return plans


def _build_substituent_pruning_plans(
    mol: Chem.Mol,
    max_perturbations: int,
) -> list[_PerturbationPlan]:
    """Build chemically valid exocyclic substituent/sidechain/N-methyl/lipid-tail pruning plans."""
    plans: list[_PerturbationPlan] = []
    seen_atom_sets: set[frozenset[int]] = set()
    seen_smiles: set[str] = set()

    def _try_add(keep_a: int, cut_a: int, kind: str, custom_label: Optional[str] = None) -> bool:
        res = _prune_exocyclic_branch(mol, keep_a, cut_a)
        if res is None:
            return False
        smi, sub_atoms = res
        aset = frozenset(sub_atoms)
        if aset in seen_atom_sets or smi in seen_smiles:
            return False
        seen_atom_sets.add(aset)
        seen_smiles.add(smi)
        idx_num = len(plans) + 1
        lbl = custom_label or _describe_substructure(mol, sub_atoms, kind.replace("_", " ").title(), idx_num)
        plans.append(
            _PerturbationPlan(
                id=f"grp_{idx_num}_{kind}",
                label=lbl,
                kind=kind,
                position=None,
                original_residue=None,
                masked_to="H",
                perturbed_sequence=None,
                perturbed_smiles=smi,
                atom_indices=sub_atoms,
                backbone_indices=[],
            )
        )
        return True

    # 1. Fatty-acyl / lipid tails (prune acyl chain at amide C(=O)-N bond)
    fatty_c = 0
    for m in mol.GetSubstructMatches(_PAT_FATTY_AMIDE):
        carb_c = int(m[0])
        for nb in mol.GetAtomWithIdx(carb_c).GetNeighbors():
            if nb.GetAtomicNum() == 7:
                fatty_c += 1
                _try_add(nb.GetIdx(), carb_c, "lipid_tail", f"Fatty-acyl lipid tail #{fatty_c}")
                break

    # 2. Backbone N-methyl groups (demethylate N-Me -> N-H)
    nme_c = 0
    for m in mol.GetSubstructMatches(_PAT_N_METHYL):
        n_idx, me_idx = int(m[0]), int(m[1])
        if _try_add(n_idx, me_idx, "n_methyl", f"N-Methyl #{nme_c + 1}"):
            nme_c += 1

    # 3. Peptide alpha-carbon sidechains (prune non-ring exocyclic bond off CA -> Gly-like backbone)
    ca_matches = mol.GetSubstructMatches(_PAT_BACKBONE_CA)
    ca_sidechains: list[tuple[int, int]] = []
    for m in ca_matches:
        n_idx, ca_idx, c_idx = int(m[0]), int(m[1]), int(m[2])
        for nb in mol.GetAtomWithIdx(ca_idx).GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in (n_idx, c_idx):
                bond = mol.GetBondBetweenAtoms(ca_idx, nb_idx)
                if bond is not None and not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
                    ca_sidechains.append((ca_idx, nb_idx))

    for ca_idx, sc_root in ca_sidechains:
        _try_add(ca_idx, sc_root, "sidechain")

    # 4. Other exocyclic functional groups (e.g. OH, O-alkyl, aryl, sulfates) if room remains
    if len(plans) < max_perturbations:
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            at1, at2 = mol.GetAtomWithIdx(a1), mol.GetAtomWithIdx(a2)
            # Prune leaf or small exocyclic branch attached to a ring or degree>=3 atom
            if at1.IsInRing() and not at2.IsInRing():
                _try_add(a1, a2, "functional_group")
            elif at2.IsInRing() and not at1.IsInRing():
                _try_add(a2, a1, "functional_group")
            elif at1.GetDegree() >= 3 and at2.GetDegree() <= 2:
                _try_add(a1, a2, "functional_group")
            elif at2.GetDegree() >= 3 and at1.GetDegree() <= 2:
                _try_add(a2, a1, "functional_group")
            if len(plans) >= max_perturbations * 2:
                break

    # Prioritize lipid tails & N-methyls + evenly spaced sidechains/groups up to max_perturbations
    if len(plans) > max_perturbations:
        priority = [p for p in plans if p.kind in ("lipid_tail", "n_methyl")]
        rest = [p for p in plans if p.kind not in ("lipid_tail", "n_methyl")]
        budget = max(1, max_perturbations - len(priority))
        plans = priority[:max_perturbations] + _evenly_pick(rest, budget)

    return plans[:max_perturbations]


def render_direct_attribution_svg(
    mol: Chem.Mol,
    weights_01: list[float],
    endpoint: str,
    endpoint_label: str,
    method_tag: str,
    title: str,
    prediction_text: Optional[str] = None,
    width: int = 760,
    height: int = 460,
) -> str:
    """Render publication-grade RDKit 2D SVG for Direct Ensemble Perturbation Attribution."""
    mol_obj = Chem.Mol(mol)
    n_atoms = mol_obj.GetNumAtoms()
    if mol_obj.GetNumConformers() == 0:
        try:
            rdDepictor.SetPreferCoordGen(n_atoms <= 140)
            rdDepictor.Compute2DCoords(mol_obj)
        except Exception:
            rdDepictor.SetPreferCoordGen(False)
            AllChem.Compute2DCoords(mol_obj)

    header_h = 54
    footer_h = 54
    draw_h = max(240, height - header_h - footer_h)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, draw_h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.06
    opts.bondLineWidth = 1.6 if n_atoms <= 120 else 1.2
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.continuousHighlight = True
    opts.highlightBondWidthMultiplier = 10 if n_atoms <= 120 else 7

    atom_colors: dict[int, tuple[float, float, float]] = {}
    atom_radii: dict[int, float] = {}
    highlight_atoms: list[int] = []
    for i in range(n_atoms):
        wi = float(weights_01[i]) if i < len(weights_01) else 0.5
        mag = abs(wi - 0.5) * 2.0
        atom_colors[i] = _palette_diverging(wi)
        atom_radii[i] = 0.24 + 0.28 * mag
        highlight_atoms.append(i)

    bond_colors: dict[int, tuple[float, float, float]] = {}
    highlight_bonds: list[int] = []
    for bond in mol_obj.GetBonds():
        bidx = bond.GetIdx()
        w1 = float(weights_01[bond.GetBeginAtomIdx()])
        w2 = float(weights_01[bond.GetEndAtomIdx()])
        bond_colors[bidx] = _palette_diverging(0.5 * (w1 + w2))
        highlight_bonds.append(bidx)

    drawer.DrawMolecule(
        mol_obj,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    raw_svg = drawer.GetDrawingText()

    start_tag_end = raw_svg.find(">", raw_svg.find("<svg"))
    end_tag_start = raw_svg.rfind("</svg>")
    inner_mol_svg = raw_svg[start_tag_end + 1 : end_tag_start] if (start_tag_end != -1 and end_tag_start != -1) else raw_svg

    pred_badge = f" | {prediction_text}" if prediction_text else ""
    grad_id = f"directAttrGrad_{abs(hash((endpoint, method_tag))) % 100000}"
    legend_w = min(380, width - 200)
    legend_x = (width - legend_w) // 2
    legend_y = header_h + draw_h + 14

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#247AB8"/>
      <stop offset="32%" stop-color="#8FCCE6"/>
      <stop offset="50%" stop-color="#F2F5F7"/>
      <stop offset="68%" stop-color="#FAB852"/>
      <stop offset="100%" stop-color="#D9382E"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2"/>
  <rect x="0" y="0" width="{width}" height="{header_h}" rx="10" fill="#F8FAFC"/>
  <line x1="0" y1="{header_h}" x2="{width}" y2="{header_h}" stroke="#E2E8F0" stroke-width="1"/>
  <text x="18" y="23" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="14.5" font-weight="600" fill="#0F172A">{_xml_escape(title)}</text>
  <text x="18" y="42" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11.5" font-weight="500" fill="#0F766E">Direct Ensemble Perturbation (100% Production-Faithful · {_xml_escape(method_tag)}, n={n_atoms}){_xml_escape(pred_badge)}</text>
  <g transform="translate(0, {header_h})">
    {inner_mol_svg}
  </g>
  <line x1="16" y1="{header_h + draw_h + 2}" x2="{width - 16}" y2="{header_h + draw_h + 2}" stroke="#F1F5F9" stroke-width="1"/>
  <rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="12" rx="6" fill="url(#{grad_id})" stroke="#CBD5E1" stroke-width="0.8"/>
  <text x="{legend_x}" y="{legend_y + 27}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" fill="#1E40AF">Decreases { _xml_escape(endpoint) } (−Δ)</text>
  <text x="{legend_x + legend_w // 2}" y="{legend_y + 27}" text-anchor="middle" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" fill="#475569">Neutral (0)</text>
  <text x="{legend_x + legend_w}" y="{legend_y + 27}" text-anchor="end" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" fill="#991B1B">Increases { _xml_escape(endpoint) } (+Δ)</text>
</svg>
"""


def run_direct_attribution(
    adapter: OmniPanelAdapter,
    req: DirectAttributionRequest,
) -> DirectAttributionResponse:
    """Compute 100% production-ensemble-faithful perturbation attribution for any of the 47 endpoints."""
    t0 = time.perf_counter()
    available_eps = adapter._ordered_endpoints()
    if available_eps and req.endpoint not in available_eps:
        raise InputValidationError(
            f"Unsupported endpoint '{req.endpoint}'. Must be one of the {len(available_eps)} supported endpoints."
        )

    resolved: ResolvedInput = adapter.resolve_query(
        query=req.query,
        sequence=req.sequence,
        smiles=req.smiles,
    )
    if resolved.mol is None:
        raise InputValidationError(
            "Direct attribution requires an RDKit atomic graph (SMILES or canonical sequence <= 80 AA)."
        )

    cond = req.conditions or AssayConditions()
    v33_cond = cond.to_v33_conditions()
    can_smi = resolved.canonical_smiles or Chem.MolToSmiles(resolved.mol, canonical=True)

    # Check TTL cache
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "smi": can_smi,
                "seq": resolved.sequence or "",
                "ep": req.endpoint,
                "cond": v33_cond,
                "max_p": req.max_perturbations,
                "w": req.width,
                "h": req.height,
                "mode": adapter.mode,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cached = _DIRECT_ATTR_CACHE.get(cache_key)
    if cached is not None:
        resp_copy = cached.model_copy(update={"cache_hit": True, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return resp_copy

    # Decide Mode A (residue Ala/Gly masking) vs Mode B (chemical substituent pruning)
    mol = resolved.mol
    use_residue_mode = False
    if resolved.sequence and len(resolved.sequence) >= 3 and len(resolved.sequence) <= 80:
        seq_mol = sequence_to_mol(resolved.sequence)
        if seq_mol is not None:
            seq_can = _sanitize_single_frag(seq_mol)
            # Only use pure sequence masking if the input IS that unmodified canonical linear peptide
            if seq_can == can_smi:
                mol = seq_mol  # preserves AtomPDBResidueInfo for exact residue -> atom mapping!
                use_residue_mode = True

    if use_residue_mode and resolved.sequence:
        method = "residue_alanine_glycine_masking"
        method_tag = "Residue Ala/Gly Masking"
        method_desc = (
            "Leave-one-out residue masking (non-Ala → Ala, Ala → Gly) evaluated directly on the production "
            "ensemble; Δ = pred(full) − pred(masked) mapped onto each residue's RDKit atoms."
        )
        plans = _build_residue_masking_plans(mol, resolved.sequence, req.max_perturbations)
    else:
        method = "chemical_substituent_pruning"
        method_tag = "Valid Substituent Pruning"
        method_desc = (
            "Chemically valid exocyclic sidechain, N-methyl, fatty-acyl tail, and functional-group pruning "
            "(replaced by −H, single-fragment RDKit sanitized) evaluated directly on the production ensemble."
        )
        plans = _build_substituent_pruning_plans(mol, req.max_perturbations)

    # Evaluate base molecule + all perturbations in ONE vectorized batch call
    ep_doc = (adapter.manifest.get("endpoints") or {}).get(req.endpoint) or {}
    include_tier2 = int(ep_doc.get("tier", 1)) == 2

    batch_rows: list[BatchParsedRow] = [
        BatchParsedRow(
            index=0,
            name=resolved.query_name,
            detected_type=resolved.resolved_type,
            sequence=resolved.sequence if use_residue_mode else None,
            smiles=can_smi,
            valid=True,
        )
    ]
    for idx, pl in enumerate(plans, start=1):
        batch_rows.append(
            BatchParsedRow(
                index=idx,
                name=pl.label,
                detected_type="sequence" if pl.perturbed_sequence else "smiles",
                sequence=pl.perturbed_sequence,
                smiles=pl.perturbed_smiles,
                valid=True,
            )
        )

    batch_out = adapter.predict_batch(batch_rows, include_tier2=include_tier2, conditions=cond)
    base_bm = batch_out[0]
    if base_bm.status != "ok":
        raise InputValidationError(f"Base prediction failed: {base_bm.error}")
    base_ep = next((p for p in base_bm.predictions if p.endpoint == req.endpoint), None)
    if base_ep is None or base_ep.value is None:
        raise InputValidationError(f"Endpoint '{req.endpoint}' was not returned for this molecule.")

    base_val = float(base_ep.value)
    base_raw = float(base_ep.raw_value) if base_ep.raw_value is not None else base_val

    n_atoms = mol.GetNumAtoms()
    atom_delta_sum = [0.0] * n_atoms
    atom_delta_cnt = [0] * n_atoms
    atom_group_id: list[Optional[str]] = [None] * n_atoms

    raw_impacts: list[tuple[_PerturbationPlan, float, float, float, float]] = []
    for pl, bm in zip(plans, batch_out[1:]):
        if bm.status != "ok":
            continue
        p_ep = next((p for p in bm.predictions if p.endpoint == req.endpoint), None)
        if p_ep is None or p_ep.value is None:
            continue
        p_val = float(p_ep.value)
        p_raw = float(p_ep.raw_value) if p_ep.raw_value is not None else p_val
        delta_raw = base_raw - p_raw
        delta_disp = base_val - p_val
        raw_impacts.append((pl, p_val, p_raw, delta_raw, delta_disp))

        bb_set = set(pl.backbone_indices)
        for aidx in pl.atom_indices:
            if 0 <= aidx < n_atoms:
                # For residue masking, emphasize sidechain atoms slightly over backbone atoms
                scale = 0.55 if (use_residue_mode and aidx in bb_set and len(pl.atom_indices) > len(bb_set)) else 1.0
                atom_delta_sum[aidx] += delta_raw * scale
                atom_delta_cnt[aidx] += 1
                if atom_group_id[aidx] is None:
                    atom_group_id[aidx] = pl.id

    atom_raw_deltas = [
        (atom_delta_sum[i] / atom_delta_cnt[i]) if atom_delta_cnt[i] > 0 else 0.0
        for i in range(n_atoms)
    ]

    max_abs_impact = max(
        [abs(x[3]) for x in raw_impacts] + [abs(d) for d in atom_raw_deltas] + [0.0]
    )
    denom = max_abs_impact if max_abs_impact > 1e-9 else 1.0

    impacts_out: list[ResidueOrGroupImpact] = []
    for pl, p_val, p_raw, delta_raw, delta_disp in raw_impacts:
        norm_w = max(-1.0, min(1.0, delta_raw / denom)) if max_abs_impact > 1e-9 else 0.0
        direction = "increases" if norm_w > 0.03 else ("decreases" if norm_w < -0.03 else "neutral")
        impacts_out.append(
            ResidueOrGroupImpact(
                id=pl.id,
                label=pl.label,
                kind=pl.kind,  # type: ignore[arg-type]
                position=pl.position,
                original_residue=pl.original_residue,
                masked_to=pl.masked_to,
                perturbed_smiles=pl.perturbed_smiles,
                perturbed_value=float(f"{p_val:.6g}"),
                perturbed_raw=round(p_raw, 5),
                delta_raw=round(delta_raw, 6),
                delta_display=float(f"{delta_disp:.6g}"),
                normalized_weight=round(norm_w, 4),
                atom_indices=pl.atom_indices,
                direction=direction,  # type: ignore[arg-type]
            )
        )
    impacts_out.sort(key=lambda x: abs(x.delta_raw), reverse=True)

    atom_weights_out: list[DirectAtomWeight] = []
    weights_01: list[float] = []
    for i in range(n_atoms):
        d_i = atom_raw_deltas[i]
        w_signed = max(-1.0, min(1.0, d_i / denom)) if max_abs_impact > 1e-9 else 0.0
        w_01 = max(0.0, min(1.0, 0.5 + 0.5 * w_signed))
        weights_01.append(w_01)
        atom_weights_out.append(
            DirectAtomWeight(
                atom_index=i,
                symbol=mol.GetAtomWithIdx(i).GetSymbol(),
                weight=round(w_signed, 4),
                weight_01=round(w_01, 4),
                raw_delta=round(d_i, 6),
                group_id=atom_group_id[i],
            )
        )

    top_order = sorted(range(n_atoms), key=lambda i: abs(atom_weights_out[i].weight), reverse=True)[:8]
    top_atoms = [
        TopAtomContribution(
            atom_index=i,
            symbol=atom_weights_out[i].symbol,
            weight=atom_weights_out[i].weight,
        )
        for i in top_order
    ]

    ui = ENDPOINT_UI_META.get(req.endpoint) or {}
    ep_label = ui.get("display_en") or str(ep_doc.get("display_en") or req.endpoint)
    ep_label_zh = ui.get("display_zh") or str(ep_doc.get("display_zh") or req.endpoint)
    u_part = f" {base_ep.unit}" if base_ep.unit and base_ep.unit not in ("probability", "fraction", "fraction_bound") else ""
    pred_badge = f"Ensemble: {base_ep.value_display}{u_part}"

    svg_str = render_direct_attribution_svg(
        mol=mol,
        weights_01=weights_01,
        endpoint=req.endpoint,
        endpoint_label=ep_label,
        method_tag=method_tag,
        title=f"{resolved.query_name} · {ep_label}",
        prediction_text=pred_badge,
        width=req.width,
        height=req.height,
    )

    resp = DirectAttributionResponse(
        endpoint=req.endpoint,
        endpoint_label=ep_label,
        endpoint_label_zh=ep_label_zh,
        interpretability_tier="direct_ensemble_faithful",
        method=method,  # type: ignore[arg-type]
        method_description=method_desc,
        base_prediction={
            "value": base_val,
            "raw_value": base_raw,
            "value_display": base_ep.value_display,
            "unit": base_ep.unit,
            "kind": base_ep.kind,
            "ad_status": base_ep.ad_status,
            "evidence_grade": base_ep.evidence_grade,
        },
        n_atoms=n_atoms,
        n_perturbations=len(impacts_out),
        atom_weights=atom_weights_out,
        top_atoms=top_atoms,
        residue_or_group_impacts=impacts_out,
        svg_2d=svg_str,
        svg=svg_str,
        available_endpoints=available_eps,
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
        cache_hit=False,
        adapter_mode=adapter.mode,
    )
    _DIRECT_ATTR_CACHE.set(cache_key, resp)
    return resp



def compute_headless_multi_endpoint_attribution(
    adapter: OmniPanelAdapter,
    mol: Chem.Mol,
    sequence: Optional[str],
    canonical_smiles: str,
    target_endpoints: Sequence[str],
    *,
    locked_positions: Optional[set[int]] = None,
    locked_atom_indices: Optional[set[int]] = None,
    seen_keys: Optional[set[tuple[str, Optional[str]]]] = None,
    max_perturbations: int = 8,
    include_tier2: bool = False,
    conditions: Optional[AssayConditions] = None,
    chunk_size: int = 4,
    base_result: Optional[Any] = None,
    eval_batch_fn: Optional[Any] = None,
    cancel_event: Optional[Any] = None,
) -> dict[str, Any]:
    """Headless multi-endpoint Critic attribution (skip_svg=True, excludes locked & already-seen sites).

    Evaluates unlocked leave-one-out perturbations in chunked batches (`<= chunk_size`) and returns
    per-endpoint `delta_raw = base_raw - perturbed_raw` across ALL `target_endpoints` simultaneously,
    plus reusable `(plan, BatchMoleculeResult)` pairs so Round 1 can recycle Ala-scan evaluations at zero extra cost.
    """
    t0 = time.perf_counter()
    cond = conditions or AssayConditions()
    excl_pos = set(locked_positions or ())
    excl_atoms = set(locked_atom_indices or ())
    seen = seen_keys or set()

    use_residue_mode = False
    working_mol = mol
    if sequence and 3 <= len(sequence) <= 80:
        seq_mol = sequence_to_mol(sequence)
        if seq_mol is not None and _sanitize_single_frag(seq_mol) == canonical_smiles:
            working_mol = seq_mol
            use_residue_mode = True

    if use_residue_mode and sequence:
        method = "residue_alanine_glycine_masking"
        raw_plans = _build_residue_masking_plans(
            working_mol, sequence, max(1, int(max_perturbations)) * 2, exclude_positions=excl_pos
        )
        plans = [
            pl
            for pl in raw_plans
            if (pl.perturbed_smiles, pl.perturbed_sequence) not in seen
            and pl.perturbed_smiles != canonical_smiles
        ][: max(1, int(max_perturbations))]
    else:
        method = "chemical_substituent_pruning"
        raw_plans = _build_substituent_pruning_plans(working_mol, max(1, int(max_perturbations)) * 2)
        plans = [
            pl
            for pl in raw_plans
            if not (set(pl.atom_indices) & excl_atoms)
            and (pl.perturbed_smiles, None) not in seen
            and pl.perturbed_smiles != canonical_smiles
        ][: max(1, int(max_perturbations))]

    eval_fn = eval_batch_fn or (lambda rows: adapter.predict_batch(rows, include_tier2=include_tier2, conditions=cond))

    n_evals_used = 0
    if base_result is None:
        if cancel_event is not None and cancel_event.is_set():
            return {
                "method": method,
                "n_perturbations": 0,
                "n_evals_used": 0,
                "base_ok": False,
                "impacts_by_endpoint": {ep: [] for ep in target_endpoints},
                "reusable_evals": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }
        base_row = BatchParsedRow(
            index=0,
            name="critic_base",
            detected_type="sequence" if use_residue_mode else "smiles",
            sequence=sequence if use_residue_mode else None,
            smiles=canonical_smiles,
            valid=True,
        )
        base_out = eval_fn([base_row])
        n_evals_used += len(base_out)
        base_result = base_out[0] if base_out else None

    if base_result is None or base_result.status != "ok":
        return {
            "method": method,
            "n_perturbations": 0,
            "n_evals_used": n_evals_used,
            "base_ok": False,
            "impacts_by_endpoint": {ep: [] for ep in target_endpoints},
            "reusable_evals": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    base_by_ep = {p.endpoint: p for p in base_result.predictions}
    pert_results: list[Any] = []
    if plans:
        csize = max(1, int(chunk_size))
        for start in range(0, len(plans), csize):
            if cancel_event is not None and cancel_event.is_set():
                break
            chunk_plans = plans[start : start + csize]
            rows = [
                BatchParsedRow(
                    index=start + i + 1,
                    name=pl.label,
                    detected_type="sequence" if pl.perturbed_sequence else "smiles",
                    sequence=pl.perturbed_sequence,
                    smiles=pl.perturbed_smiles,
                    valid=True,
                )
                for i, pl in enumerate(chunk_plans)
            ]
            out_chunk = eval_fn(rows)
            n_evals_used += len(rows)
            pert_results.extend(out_chunk)

    impacts_by_ep: dict[str, list[dict[str, Any]]] = {ep: [] for ep in target_endpoints}
    reusable_evals: list[tuple[_PerturbationPlan, Any]] = []

    for pl, bm in zip(plans, pert_results):
        if bm is None or bm.status != "ok":
            continue
        reusable_evals.append((pl, bm))
        p_by_ep = {p.endpoint: p for p in bm.predictions}
        for ep in target_endpoints:
            bp = base_by_ep.get(ep)
            pp = p_by_ep.get(ep)
            if bp is None or pp is None or bp.value is None or pp.value is None:
                continue
            b_raw = float(bp.raw_value) if bp.raw_value is not None else float(bp.value)
            p_raw = float(pp.raw_value) if pp.raw_value is not None else float(pp.value)
            delta_raw = b_raw - p_raw
            impacts_by_ep[ep].append(
                {
                    "id": pl.id,
                    "label": pl.label,
                    "kind": pl.kind,
                    "position": pl.position,
                    "original_residue": pl.original_residue,
                    "masked_to": pl.masked_to,
                    "perturbed_smiles": pl.perturbed_smiles,
                    "perturbed_sequence": pl.perturbed_sequence,
                    "base_raw": round(b_raw, 5),
                    "perturbed_raw": round(p_raw, 5),
                    "delta_raw": round(delta_raw, 5),
                    "direction": "increases" if delta_raw > 1e-6 else ("decreases" if delta_raw < -1e-6 else "neutral"),
                }
            )

    for ep in target_endpoints:
        impacts_by_ep[ep].sort(key=lambda r: abs(r["delta_raw"]), reverse=True)

    return {
        "method": method,
        "n_perturbations": len(reusable_evals),
        "n_evals_used": n_evals_used,
        "base_ok": True,
        "impacts_by_endpoint": impacts_by_ep,
        "reusable_evals": reusable_evals,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
