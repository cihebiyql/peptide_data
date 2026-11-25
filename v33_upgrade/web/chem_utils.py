"""RDKit molecular rendering, substructure motif detection, physicochemical calculation, and smart input parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from .schemas import HighlightLegendItem, MechanismAnchor, ModificationBadge, PhysChemSummary

RDLogger.DisableLog("rdApp.*")

# Standard 20 canonical amino acid 1-letter codes
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# Subset that also happens to be valid organic-subset single-letter uppercase SMILES atoms
SMILES_OVERLAP_ATOMS = set("CNSPFIH")
AA_ONLY_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,1022}$")
SMILES_HINT_RE = re.compile(r"[a-z0-9=()\[\]#@\\/\+\-\.\*:]|Cl|Br|O|B")

# Compiled SMARTS patterns for peptide modification & structural motif recognition
_PAT_DISULFIDE = Chem.MolFromSmarts("[#16X2]-[#16X2]")
_PAT_N_METHYL = Chem.MolFromSmarts("[NX3;H0]([CH3])C(=O)")
_PAT_AIB = Chem.MolFromSmarts("NC(C)(C)C(=O)")
_PAT_PEG = Chem.MolFromSmarts("OCCOCC")
_PAT_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
_PAT_GUANIDINO = Chem.MolFromSmarts("NC(=N)N")
_PAT_AMINE_PRIM = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(NS=O)]")
_PAT_IMIDAZOLE = Chem.MolFromSmarts("c1cnc[nH]1")
_PAT_AMIDE_BOND = Chem.MolFromSmarts("C(=O)N")

# Precompiled diacid / monoacyl SMARTS from longest to shortest
_DIACID_PATTERNS: list[tuple[int, Chem.Mol]] = [
    (L, Chem.MolFromSmarts("C(=O)" + "C" * L + "C(=O)"))
    for L in range(22, 9, -1)
]
_MONOACYL_PATTERNS: list[tuple[int, Chem.Mol]] = [
    (L, Chem.MolFromSmarts("C(=O)" + "C" * L))
    for L in range(22, 8, -1)
]
# Exact covalent amide/ester-linked saturated mono-fatty-acyl chain for mechanism rule overlay
# CH3-(CH2)_{L-1}-C(=O)-[N,O] -> total chain carbons = L + 1 (e.g. C14 myristoyl, C16 palmitoyl)
_MONOACYL_ANCHOR_PATTERNS: list[tuple[int, Chem.Mol]] = [
    (L + 1, Chem.MolFromSmarts("[CH3]" + ("[CH2]" * (L - 1)) + "C(=O)[N,O]"))
    for L in range(21, 12, -1)
]

# Henderson-Hasselbalch pKa constants for physiological pH 7.4 net charge estimation (canonical sequences)
_PKA_POS = {"K": 10.53, "R": 12.48, "H": 6.00}
_PKA_NEG = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07}

# Scientific muted heteroatom palette matching OmniPanel UI palette
_ATOM_PALETTE = {
    6: (0.118, 0.161, 0.231),   # C: #1E293B deep slate
    7: (0.008, 0.518, 0.780),   # N: #0284C7 ocean cobalt
    8: (0.863, 0.149, 0.149),   # O: #DC2626 crisp crimson
    16: (0.051, 0.580, 0.533),  # S: #0D9488 scientific teal
    15: (0.851, 0.467, 0.024),  # P: #D97706 amber
    9: (0.020, 0.588, 0.412),   # F: #059669 emerald
    17: (0.020, 0.588, 0.412),  # Cl: emerald
    35: (0.576, 0.200, 0.918),  # Br: violet
}


class InputValidationError(ValueError):
    """Raised when user input cannot be interpreted as a valid drug name, sequence, or SMILES."""


@dataclass
class ResolvedInput:
    query_name: str
    resolved_type: str  # "smiles" | "sequence" | "dual" | "drug_name"
    sequence: Optional[str]
    smiles: Optional[str]
    canonical_smiles: Optional[str]
    mol: Optional[Chem.Mol]
    drug_entry: Optional[dict[str, Any]] = None


def clean_sequence_text(text: str) -> str:
    """Strip FASTA headers and whitespace from a candidate sequence string."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip() and not ln.strip().startswith(">")]
    return re.sub(r"\s+", "", "".join(lines)).upper()


def is_canonical_sequence(text: str) -> bool:
    """Return True if text is clearly a canonical 20-AA amino acid sequence."""
    raw = (text or "").strip()
    if not raw:
        return False
    # If there are lowercase letters, digits, or SMILES punctuation, it is not a raw 20-AA sequence
    if SMILES_HINT_RE.search(raw):
        return False
    cleaned = clean_sequence_text(raw)
    if not AA_ONLY_RE.fullmatch(cleaned):
        return False
    # Guard against short pure-carbon/nitrogen strings like "CCCC" or "CCN"
    if len(cleaned) < 4:
        return False
    if set(cleaned).issubset(SMILES_OVERLAP_ATOMS) and len(set(cleaned)) <= 2:
        return False
    return True


def parse_smiles_safe(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES safely with RDKit, returning None on invalid input."""
    s = (smiles or "").strip()
    if not s or len(s) > 8000:
        return None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol
    except Exception:
        return None
    return None


def sequence_to_mol(sequence: str, max_residues_for_mol: int = 80) -> Optional[Chem.Mol]:
    """Convert canonical 20-AA sequence to an RDKit Mol via Chem.MolFromSequence."""
    seq = clean_sequence_text(sequence)
    if not seq or not AA_ONLY_RE.fullmatch(seq):
        return None
    if len(seq) > max_residues_for_mol:
        return None
    try:
        return Chem.MolFromSequence(seq)
    except Exception:
        return None


def estimate_sequence_charge_ph74(sequence: str) -> float:
    """Estimate physiological net charge at pH 7.4 via Henderson-Hasselbalch."""
    pH = 7.4
    # N-terminus (~8.0) and C-terminus (~3.1)
    pos = 1.0 / (1.0 + 10.0 ** (pH - 8.0))
    neg = 1.0 / (1.0 + 10.0 ** (3.1 - pH))
    for aa in sequence:
        if aa in _PKA_POS:
            pos += 1.0 / (1.0 + 10.0 ** (pH - _PKA_POS[aa]))
        elif aa in _PKA_NEG:
            neg += 1.0 / (1.0 + 10.0 ** (_PKA_NEG[aa] - pH))
    return round(pos - neg, 2)


def estimate_mol_charge_ph74(mol: Chem.Mol, sequence: Optional[str] = None) -> float:
    """Estimate net charge at pH 7.4 from canonical sequence or ionizable SMARTS groups."""
    if sequence:
        return estimate_sequence_charge_ph74(sequence)
    n_acid = len(mol.GetSubstructMatches(_PAT_CARBOXYL))
    n_guan = len(mol.GetSubstructMatches(_PAT_GUANIDINO))
    n_amine = len(mol.GetSubstructMatches(_PAT_AMINE_PRIM))
    n_imid = len(mol.GetSubstructMatches(_PAT_IMIDAZOLE))
    # Formal charge + ionized estimate (carboxylate -1, primary amine/guanidinium +1, imidazole ~+0.04 at pH 7.4)
    formal = Chem.GetFormalCharge(mol)
    est = float(formal) + (n_guan * 1.0) + (n_amine * 0.95) + (n_imid * 0.04) - (n_acid * 0.99)
    return round(est, 2)


def estimate_residue_count(mol: Chem.Mol, sequence: Optional[str] = None) -> Optional[int]:
    """Return exact AA length for sequence inputs, or estimate peptide residue count from amide bonds."""
    if sequence:
        return len(sequence)
    amide_matches = len(mol.GetSubstructMatches(_PAT_AMIDE_BOND))
    if amide_matches >= 2:
        # Approximate residues (subtract 1-3 for side-chain amides like Asn/Gln/linkers heuristically)
        return max(2, amide_matches)
    return None


def detect_acylation_anchor(
    mol: Optional[Chem.Mol],
    exclude_drug: Optional[str] = None,
) -> MechanismAnchor:
    """RDKit SMARTS-based fatty-acylation structural motif detector & literature PK class rule overlay.

    Scientifically separated from Pure-ML predictions (reported strictly as a transparent
    literature rule overlay, never conflated with ML test metrics). Uses exact RDKit
    graph SMARTS (zero fragile SMILES string regex) and strictly fatty-acylated reference
    classes (Fc/albumin fusion proteins removed). Supports leave-one-drug-out reference
    reporting via `exclude_drug`.
    """
    if mol is None:
        return MechanismAnchor(detected=False)

    ex_low = (exclude_drug or "").strip().lower()

    diacid_len = 0
    for L, pat in _DIACID_PATTERNS:
        if pat is not None and mol.HasSubstructMatch(pat):
            diacid_len = L
            break

    # Check total aliphatic carbons in diacid chain (L internal CH2 + 2 carbonyls = C(L+2))
    if diacid_len >= 16:
        c_total = diacid_len + 2
        # Strictly C18/C20 fatty-diacid acylated peptide literature class (NO Fc/albumin fusion proteins!)
        all_refs = [
            ("semaglutide", "Semaglutide (C18 diacid, ~168 h FDA label)"),
            ("tirzepatide", "Tirzepatide (C20 diacid, ~120 h FDA label)"),
            ("cagrilintide", "Cagrilintide (C20 diacid, ~159–195 h clinical Ph1/2)"),
        ]
        shown_refs = [lbl for k, lbl in all_refs if not ex_low or k not in ex_low]
        return MechanismAnchor(
            detected=True,
            acylation_class="weekly_diacid",
            label_en=f"C{c_total} Fatty Diacid · Once-Weekly Albumin-Binding Rule Overlay",
            label_zh=f"C{c_total} 脂肪二羧酸脂化 · 周给药白蛋白结合机制规则层",
            chain_carbons=c_total,
            value_h=144.0,
            range_h=[120.0, 168.0],
            anchor_drugs=shown_refs,
            note=(
                f"RDKit SMARTS detected covalent C{c_total} α,ω-fatty-diacid acylation motif "
                "(reversible human serum albumin binding + reduced renal filtration). "
                "Displayed strictly as a literature class rule overlay (typical once-weekly class 120–168 h), "
                "separate from the Pure-ML super-ensemble prediction."
            ),
        )
    if diacid_len >= 12:
        c_total = diacid_len + 2
        all_refs = [
            ("degludec", "Insulin degludec (C16 diacid, ~25 h FDA label)"),
            ("icodec", "C14–C16 diacid multi-hexamer/albumin depot class (typical 20–30 h)"),
        ]
        shown_refs = [lbl for k, lbl in all_refs if not ex_low or k not in ex_low]
        return MechanismAnchor(
            detected=True,
            acylation_class="depot_diacid",
            label_en=f"C{c_total} Fatty Diacid · Intermediate/Depot Albumin-Binding Rule Overlay",
            label_zh=f"C{c_total} 脂肪二羧酸脂化 · 中长效缓释/白蛋白结合规则层",
            chain_carbons=c_total,
            value_h=25.0,
            range_h=[20.0, 30.0],
            anchor_drugs=shown_refs,
            note=(
                f"RDKit SMARTS detected covalent C{c_total} fatty-diacid acylation motif "
                "(subcutaneous depot self-association + albumin binding). "
                "Displayed strictly as a literature class rule overlay (20–30 h), separate from Pure-ML."
            ),
        )

    mono_c = 0
    for c_tot, pat in _MONOACYL_ANCHOR_PATTERNS:
        if pat is not None and mol.HasSubstructMatch(pat):
            mono_c = c_tot
            break
    if mono_c >= 14:
        all_refs = [
            ("detemir", "Insulin detemir (C14 monoacyl, 5–7 h FDA label)"),
            ("liraglutide", "Liraglutide (C16 monoacyl, ~13 h FDA label)"),
        ]
        shown_refs = [lbl for k, lbl in all_refs if not ex_low or k not in ex_low]
        return MechanismAnchor(
            detected=True,
            acylation_class="daily_acyl",
            label_en=f"C{mono_c} Mono-Fatty-Acyl · Once-Daily Albumin-Binding Rule Overlay",
            label_zh=f"C{mono_c} 单酰基脂链 · 日给药白蛋白结合机制规则层",
            chain_carbons=mono_c,
            value_h=9.0,
            range_h=[5.0, 13.0],
            anchor_drugs=shown_refs,
            note=(
                f"RDKit SMARTS detected covalent C{mono_c} mono-fatty-acyl amide/ester chain "
                "(albumin binding slows proteolytic clearance). "
                "Displayed strictly as a literature class rule overlay (5–13 h), separate from Pure-ML."
            ),
        )

    return MechanismAnchor(detected=False)


def detect_motifs_and_highlights(
    mol: Optional[Chem.Mol],
    sequence: Optional[str] = None,
) -> tuple[list[ModificationBadge], dict[str, dict[str, Any]]]:
    """Detect peptide modifications and return badges plus RDKit atom/bond highlight maps."""
    badges: list[ModificationBadge] = []
    highlight_groups: dict[str, dict[str, Any]] = {}

    if mol is None:
        if sequence:
            badges.append(
                ModificationBadge(
                    key="linear_seq",
                    label_en=f"Linear Canonical Peptide ({len(sequence)} AA)",
                    label_zh=f"线性标准多肽 ({len(sequence)} AA)",
                    tone="slate",
                )
            )
            chg = estimate_sequence_charge_ph74(sequence)
            if chg >= 2.0:
                badges.append(
                    ModificationBadge(
                        key="cationic",
                        label_en=f"Cationic (+{chg:.1f} at pH 7.4)",
                        label_zh=f"阳离子肽 (pH 7.4 净电荷 +{chg:.1f})",
                        tone="teal",
                    )
                )
        return badges, highlight_groups

    # 1. Fatty Diacid or Monoacyl lipid tail (amber/gold)
    fatty_atoms: set[int] = set()
    fatty_label_en = ""
    fatty_label_zh = ""
    diacid_matched = 0
    for L, pat in _DIACID_PATTERNS:
        if pat is not None:
            matches = mol.GetSubstructMatches(pat)
            if matches:
                diacid_matched = L
                for m in matches:
                    fatty_atoms.update(m)
                    # Include terminal hydroxyl/oxygen on the carboxyl ends
                    for idx in m:
                        atom = mol.GetAtomWithIdx(idx)
                        for nb in atom.GetNeighbors():
                            if nb.GetAtomicNum() == 8:
                                fatty_atoms.add(nb.GetIdx())
                break

    if diacid_matched >= 10:
        c_total = diacid_matched + 2
        fatty_label_en = f"C{c_total} Fatty Diacid"
        fatty_label_zh = f"C{c_total} 脂肪二羧酸"
        badges.append(
            ModificationBadge(
                key="fatty_diacid",
                label_en=fatty_label_en,
                label_zh=fatty_label_zh,
                tone="amber",
                detail="Reversible serum albumin-binding fatty diacid modification",
            )
        )
    else:
        mono_matched = 0
        for L, pat in _MONOACYL_PATTERNS:
            if pat is not None:
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    mono_matched = L
                    for m in matches:
                        fatty_atoms.update(m)
                        for idx in m:
                            atom = mol.GetAtomWithIdx(idx)
                            for nb in atom.GetNeighbors():
                                if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1:
                                    fatty_atoms.add(nb.GetIdx())
                    break
        if mono_matched >= 9:
            c_total = mono_matched + 1
            fatty_label_en = f"C{c_total} Lipopeptide / Monoacyl" if c_total < 14 else f"C{c_total} Fatty Monoacyl"
            fatty_label_zh = f"C{c_total} 脂酰链修饰"
            badges.append(
                ModificationBadge(
                    key="fatty_monoacyl",
                    label_en=fatty_label_en,
                    label_zh=fatty_label_zh,
                    tone="amber",
                    detail="Aliphatic fatty acyl side-chain modification",
                )
            )

    if fatty_atoms:
        highlight_groups["fatty_acyl"] = {
            "atoms": fatty_atoms,
            "rgba": (0.961, 0.620, 0.043, 0.42),  # Warm Amber/Gold
            "hex": "#F59E0B",
            "label_en": fatty_label_en or "Fatty Acyl Chain",
            "label_zh": fatty_label_zh or "脂肪酰基侧链",
            "priority": 4,
        }

    # 2. Disulfide bonds S-S (ocean teal)
    ss_matches = mol.GetSubstructMatches(_PAT_DISULFIDE)
    if ss_matches:
        ss_atoms = {idx for m in ss_matches for idx in m}
        n_ss = len(ss_matches)
        badges.append(
            ModificationBadge(
                key="disulfide",
                label_en=f"Disulfide Bond (×{n_ss})" if n_ss > 1 else "Disulfide Bond",
                label_zh=f"二硫键 (×{n_ss})" if n_ss > 1 else "二硫键环化",
                tone="teal",
                detail="Intramolecular or interchain cysteine disulfide bridge(s)",
            )
        )
        highlight_groups["disulfide"] = {
            "atoms": ss_atoms,
            "rgba": (0.051, 0.580, 0.533, 0.52),  # Teal
            "hex": "#0D9488",
            "label_en": f"Disulfide Bond (×{n_ss})",
            "label_zh": f"二硫键 (×{n_ss})",
            "priority": 5,
        }

    # 3. Macrocycle rings (size >= 12)
    ring_info = mol.GetRingInfo()
    macro_rings = [r for r in ring_info.AtomRings() if len(r) >= 12]
    if macro_rings:
        macro_atoms = {idx for r in macro_rings for idx in r}
        sizes_str = "/".join(str(len(r)) for r in sorted(macro_rings, key=len))
        badges.append(
            ModificationBadge(
                key="macrocycle",
                label_en=f"Macrocyclic ({sizes_str}-ring)",
                label_zh=f"大环结构 ({sizes_str}元环)",
                tone="cobalt",
                detail="Backbone or side-chain macrocyclization improves proteolytic stability",
            )
        )
        highlight_groups["macrocycle"] = {
            "atoms": macro_atoms,
            "rgba": (0.012, 0.412, 0.631, 0.16),  # Soft Cobalt
            "hex": "#0369A1",
            "label_en": f"Macrocycle Ring ({sizes_str}-atom)",
            "label_zh": f"大环骨架 ({sizes_str}元环)",
            "priority": 1,
        }

    # 4. N-Methylated amides (Cyclosporine A hallmark)
    nme_matches = mol.GetSubstructMatches(_PAT_N_METHYL)
    if nme_matches:
        nme_atoms = {idx for m in nme_matches for idx in m[:2]}  # N + methyl C
        n_nme = len(nme_matches)
        badges.append(
            ModificationBadge(
                key="n_methyl",
                label_en=f"N-Methylated (×{n_nme})",
                label_zh=f"N-甲基化修饰 (×{n_nme})",
                tone="violet",
                detail="N-methylation removes H-bond donors to boost passive membrane permeability",
            )
        )
        highlight_groups["n_methyl"] = {
            "atoms": nme_atoms,
            "rgba": (0.486, 0.227, 0.929, 0.34),  # Soft Violet
            "hex": "#7C3AED",
            "label_en": f"N-Methylated Amide (×{n_nme})",
            "label_zh": f"N-甲基化酰胺 (×{n_nme})",
            "priority": 3,
        }

    # 5. Aib (2-aminoisobutyric acid) — DPP-4 protection in Semaglutide / Tirzepatide
    aib_matches = mol.GetSubstructMatches(_PAT_AIB)
    if aib_matches:
        badges.append(
            ModificationBadge(
                key="aib",
                label_en=f"Aib Protected (×{len(aib_matches)})",
                label_zh=f"Aib 抗酶解修饰 (×{len(aib_matches)})",
                tone="emerald",
                detail="Alpha-aminoisobutyric acid resists DPP-4 proteolytic cleavage",
            )
        )

    # 6. PEG / OEG spacer
    peg_matches = mol.GetSubstructMatches(_PAT_PEG)
    if peg_matches:
        badges.append(
            ModificationBadge(
                key="peg_spacer",
                label_en="OEG / PEG Linker",
                label_zh="OEG / PEG 柔性间隔臂",
                tone="slate",
                detail="Oligoethylene glycol spacer enhances solubility and albumin-binding geometry",
            )
        )

    # Fallback badge for unmodified canonical linear peptides
    if not badges:
        badges.append(
            ModificationBadge(
                key="linear_peptide",
                label_en="Linear L-Peptide",
                label_zh="线性标准 L-多肽",
                tone="slate",
                detail="Canonical linear backbone without detected lipid or macrocycle modifications",
            )
        )

    # Add cationic badge for strongly positively charged AMPs
    chg = estimate_mol_charge_ph74(mol, sequence=sequence)
    if chg >= 2.5 and not any(b.key == "fatty_diacid" for b in badges):
        badges.append(
            ModificationBadge(
                key="cationic_amp",
                label_en=f"Cationic (+{chg:.1f})",
                label_zh=f"阳离子特征 (+{chg:.1f})",
                tone="teal",
                detail="Positive physiological net charge typical of membrane-active / antimicrobial peptides",
            )
        )

    return badges, highlight_groups


def compute_physchem(mol: Optional[Chem.Mol], sequence: Optional[str] = None) -> PhysChemSummary:
    """Compute physicochemical descriptors using RDKit (or sequence fallback for >80 AA peptides)."""
    if mol is not None:
        ring_info = mol.GetRingInfo()
        macro_rings = sorted(len(r) for r in ring_info.AtomRings() if len(r) >= 12)
        return PhysChemSummary(
            mw=round(float(Descriptors.MolWt(mol)), 2),
            logp=round(float(Crippen.MolLogP(mol)), 2),
            tpsa=round(float(rdMolDescriptors.CalcTPSA(mol)), 1),
            hbd=int(Lipinski.NumHDonors(mol)),
            hba=int(Lipinski.NumHAcceptors(mol)),
            rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)),
            ring_count=int(rdMolDescriptors.CalcNumRings(mol)),
            macrocycle_rings=macro_rings,
            heavy_atoms=int(mol.GetNumHeavyAtoms()),
            formal_charge=int(Chem.GetFormalCharge(mol)),
            net_charge_ph74=estimate_mol_charge_ph74(mol, sequence=sequence),
            formula=rdMolDescriptors.CalcMolFormula(mol),
            length_aa=estimate_residue_count(mol, sequence=sequence),
            frac_csp3=round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 3),
        )

    # Fallback for ultra-long canonical sequences (>80 AA) where full atomic graph is skipped
    seq = sequence or ""
    aa_mw = {
        "A": 71.08, "C": 103.14, "D": 115.09, "E": 129.12, "F": 147.18,
        "G": 57.05, "H": 137.14, "I": 113.16, "K": 128.17, "L": 113.16,
        "M": 131.19, "N": 114.10, "P": 97.12, "Q": 128.13, "R": 156.19,
        "S": 87.08, "T": 101.11, "V": 99.13, "W": 186.21, "Y": 163.18,
    }
    kd_scale = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
        "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    }
    mw = 18.02 + sum(aa_mw.get(c, 110.0) for c in seq)
    gravy = sum(kd_scale.get(c, 0.0) for c in seq) / max(len(seq), 1)
    return PhysChemSummary(
        mw=round(mw, 2),
        logp=round(gravy * 1.4, 2),
        tpsa=round(len(seq) * 44.0, 1),
        hbd=int(len(seq) * 1.6),
        hba=int(len(seq) * 1.8),
        rotatable_bonds=int(len(seq) * 3.2),
        ring_count=sum(1 for c in seq if c in "FWYHP"),
        macrocycle_rings=[],
        heavy_atoms=int(round(mw / 14.0)),
        formal_charge=int(round(estimate_sequence_charge_ph74(seq))),
        net_charge_ph74=estimate_sequence_charge_ph74(seq),
        formula=f"Peptide ({len(seq)} AA)",
        length_aa=len(seq),
        frac_csp3=0.62,
    )


def render_molecule_svg(
    mol: Optional[Chem.Mol],
    sequence: Optional[str] = None,
    width: int = 560,
    height: int = 340,
) -> tuple[str, list[HighlightLegendItem], list[ModificationBadge]]:
    """Render clean RDKit 2D SVG with substructure motif highlighting."""
    badges, highlight_groups = detect_motifs_and_highlights(mol, sequence=sequence)
    legend: list[HighlightLegendItem] = []

    if mol is None:
        # Clean stylized sequence ribbon SVG fallback for ultra-long sequences (>80 AA)
        seq = sequence or ""
        chunks = [seq[i : i + 28] for i in range(0, min(len(seq), 140), 28)]
        lines_svg = "".join(
            f'<text x="280" y="{135 + i * 26}" text-anchor="middle" '
            f'font-family="JetBrains Mono, Consolas, monospace" font-size="14" fill="#0F172A" letter-spacing="1.5">{c}</text>'
            for i, c in enumerate(chunks)
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">'
            f'<rect width="{width}" height="{height}" fill="#FFFFFF" rx="8"/>'
            f'<text x="280" y="92" text-anchor="middle" font-family="Inter, sans-serif" font-size="13" font-weight="600" fill="#0369A1">'
            f'Canonical Peptide Sequence ({len(seq)} AA)</text>'
            f"{lines_svg}"
            f'<text x="280" y="{height - 32}" text-anchor="middle" font-family="Inter, sans-serif" font-size="11" fill="#64748B">'
            f'Long sequence (&gt;80 AA) displayed in residue ribbon view</text></svg>'
        )
        return svg, legend, badges

    draw_mol = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(draw_mol)
    except Exception:
        pass

    # Merge highlight groups by priority (higher priority wins for shared atoms)
    atom_colors: dict[int, tuple[float, float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float, float]] = {}
    atom_radii: dict[int, float] = {}

    ordered_groups = sorted(highlight_groups.items(), key=lambda kv: kv[1]["priority"])
    for key, info in ordered_groups:
        atoms = info["atoms"]
        rgba = info["rgba"]
        for a in atoms:
            atom_colors[a] = rgba
            atom_radii[a] = 0.45 if key != "macrocycle" else 0.30
        for bond in draw_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in atoms and a2 in atoms:
                bond_colors[bond.GetIdx()] = rgba
        legend.append(
            HighlightLegendItem(
                key=key,
                label_en=info["label_en"],
                label_zh=info["label_zh"],
                color=info["hex"],
                atom_count=len(atoms),
            )
        )

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    opts.updateAtomPalette(_ATOM_PALETTE)
    opts.bondLineWidth = 1.65 if draw_mol.GetNumHeavyAtoms() < 160 else 1.25
    opts.padding = 0.04
    opts.addStereoAnnotation = False
    opts.fillHighlights = True
    opts.highlightBondWidthMultiplier = 10
    opts.minFontSize = 7
    opts.maxFontSize = 14

    hl_atoms = sorted(atom_colors.keys())
    hl_bonds = sorted(bond_colors.keys())
    drawer.DrawMolecule(
        draw_mol,
        highlightAtoms=hl_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=hl_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    raw_svg = drawer.GetDrawingText()

    # Make SVG responsive while preserving exact viewBox aspect ratio
    raw_svg = re.sub(
        r"<svg([^>]*?)width=['\"][^'\"]+['\"]([^>]*?)height=['\"][^'\"]+['\"]",
        f'<svg\\1width="100%"\\2height="100%"',
        raw_svg,
        count=1,
    )
    return raw_svg, legend, badges


def morgan_fp(mol: Optional[Chem.Mol]) -> Optional[Any]:
    """Compute Morgan r=2, 2048-bit fingerprint for Tanimoto AD & analog lookup."""
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def tanimoto_similarity(fp1: Any, fp2: Any) -> float:
    """Compute Tanimoto similarity between two RDKit bit vectors."""
    if fp1 is None or fp2 is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def sequence_kmer_similarity(seq1: str, seq2: str, k: int = 2) -> float:
    """Compute Jaccard similarity over k-mers for sequence-only analogs."""
    if not seq1 or not seq2:
        return 0.0
    s1 = {seq1[i : i + k] for i in range(max(len(seq1) - k + 1, 1))}
    s2 = {seq2[i : i + k] for i in range(max(len(seq2) - k + 1, 1))}
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0
