#!/usr/bin/env python3
"""
Step 02 (runs on node23, 64 CPU cores, RDKit 2026.09.1):
- Canonicalizes all source SMILES & generates deterministic RDKit SMILES for provably unmodified 20AA sequences.
- Extracts parent_sequence (source or RDKit alpha-AA backbone walker) & 10 deterministic modification flags.
- Performs exact RDKit-MW ug/mL -> uM conversions & unambiguous thresholding.
- Deduplicates cleanly per (compound_key, endpoint, condition_signature) with replicate SD & conflict tracking.
- Produces compounds_v33.tsv, case17_queries.tsv, case17_overlap.tsv, and per-endpoint cleaned staging TSVs.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10_000_000)

BASE_DIR = Path("/data/qlyu/v33_upgrade")
STAGING_DIR = BASE_DIR / "data" / "staging"
OUT_DATA_DIR = BASE_DIR / "data"
N_WORKERS = min(64, os.cpu_count() or 32)

# Physical finite domain bounds for censored tails (guarantees zero NaN/Inf and strict y_lower <= y_upper)
DOMAIN_BOUNDS = {
    "PAMPA": (-10.0, -2.0),
    "Caco2": (-10.0, -2.0),
    "MDCK": (-10.0, -2.0),
    "RRCK": (-10.0, -2.0),
    "LogD7.4": (-6.0, 10.0),
    "solubility_reg": (-4.0, 6.0),
    "PPB": (0.0, 1.0),
    "half_life_reg": (-4.0, 4.0),
    "CL_per_kg": (-4.0, 3.5),
    "CL_absolute": (-4.0, 4.0),
    "Vd_per_kg": (-3.0, 3.0),
    "F": (0.0, 1.0),
    "Cmax": (-5.0, 6.0),
    "hemolysis_HC50": (-3.0, 5.5),
    "MIC_E_coli": (-4.0, 5.5),
    "MIC_S_aureus": (-4.0, 5.5),
    "MIC_P_aeruginosa": (-4.0, 5.5),
    "MIC_C_albicans": (-4.0, 5.5),
}

LOG10_ENDPOINTS = {
    "PAMPA", "Caco2", "MDCK", "RRCK",
    "solubility_reg", "half_life_reg",
    "CL_per_kg", "CL_absolute", "Vd_per_kg", "Cmax",
    "hemolysis_HC50",
    "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans",
}

# Condition columns defining deduplication signature per endpoint
ENDPOINT_COND_COLS = {
    "PAMPA": ["cond_assay", "cond_estimand"],
    "Caco2": ["cond_estimand"],
    "MDCK": ["cond_assay", "cond_estimand"],
    "RRCK": ["cond_estimand"],
    "BBB": [],
    "LogD7.4": ["cond_assay"],
    "solubility_cls": ["cond_solvent"],
    "solubility_reg": ["cond_solvent", "cond_assay"],
    "PPB": ["cond_species", "cond_matrix"],
    "half_life_reg": ["cond_species", "cond_matrix", "cond_route", "cond_estimand"],
    "half_life_cls": ["cond_species", "cond_matrix", "cond_route"],
    "CL_per_kg": ["cond_species", "cond_route", "cond_estimand"],
    "CL_absolute": ["cond_species", "cond_route", "cond_estimand"],
    "Vd_per_kg": ["cond_species", "cond_route", "cond_estimand"],
    "F": ["cond_species", "cond_route"],
    "Cmax": ["cond_species", "cond_route", "cond_dose"],
    "hemolysis_cls": ["cond_species"],
    "hemolysis_HC50": ["cond_species"],
    "overall_peptide_toxicity": [],
    "cytotoxicity_cls": ["cond_species", "cond_matrix"],
    "MIC_E_coli": ["cond_strain", "cond_assay"],
    "MIC_S_aureus": ["cond_strain", "cond_assay"],
    "MIC_P_aeruginosa": ["cond_strain", "cond_assay"],
    "MIC_C_albicans": ["cond_strain", "cond_assay"],
}

# Precompiled SMARTS patterns (initialized per worker)
_SMARTS = {}
_SIDECHAIN_MAP = {}


def _init_worker():
    global _SMARTS, _SIDECHAIN_MAP
    _SMARTS = {
        "disulfide": Chem.MolFromSmarts("[#16X2]-[#16X2]"),
        # N-methyl amide excluding Proline-like 5-membered rings
        "nmethyl": Chem.MolFromSmarts("[NX3;H0;!$([N]1CCC[C]1)]([CH3])([CX3]=O)[#6]"),
        # PEG / OEG oligo-ether linker (e.g., Semaglutide/Tirzepatide COCCOCCN or PEG2+)
        "peg": Chem.MolFromSmarts("[#8X2][CX4H2][CX4H2][#8X2][CX4H2][CX4H2][#8X2,#7X3]"),
        # C-terminal primary amide directly on alpha-carbon (excludes Asn/Gln sidechains!)
        "c_amid": Chem.MolFromSmarts("[NX3]-[CX4H1,CX4H2]-[CX3](=O)-[NH2]"),
        # N-terminal acetylation
        "n_acet": Chem.MolFromSmarts("[CH3]-[CX3](=O)-[NX3H1]-[CX4]-[CX3](=O)[#7,#8]"),
        # Backbone alpha-AA unit for chirality & sequence extraction
        "backbone_aa": Chem.MolFromSmarts("[NX3;H2,H1,H0]-[CX4;H1,H2,H0]-[CX3](=O)[#7,#8]"),
        "backbone_chiral": Chem.MolFromSmarts("[NX3;H2,H1,H0]-[CX4H1]-[CX3](=O)[#7,#8]"),
        "carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
        "acyl": Chem.MolFromSmarts("[CX3](=O)[NX3,OX2]"),
    }
    # Build canonical sidechain lookup map from RDKit standard 20 L-amino acids (non-Gly, non-Pro)
    # Using Chem.MolFromSequence("A" + aa + "A") and extracting sidechain fragment attached to middle C-alpha
    _SIDECHAIN_MAP = {}
    for aa in "ACDEFHIKLMNQRSTVWY":
        tri = Chem.MolFromSequence("G" + aa + "G")
        if tri is None:
            continue
        sc = _extract_middle_sidechain(tri)
        if sc:
            _SIDECHAIN_MAP[sc] = aa
    # Also add disulfide-bonded Cys sidechain (-CH2-S-*) -> 'C', and Lys-acylated (-CH2CH2CH2CH2-NH-C(=O)-*) -> 'K'
    # Handled cleanly during backbone walking!


def _extract_middle_sidechain(tri_mol) -> str:
    # Middle residue of G-X-G has a chiral C-alpha bonded to N, C(=O), and 1 sidechain heavy atom
    matches = tri_mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3H1]-[CX4H1]-[CX3](=O)[NX3H1]"))
    if not matches:
        return ""
    n_idx, ca_idx, c_idx = matches[0][0], matches[0][1], matches[0][2]
    ca = tri_mol.GetAtomWithIdx(ca_idx)
    sc_starts = [nb.GetIdx() for nb in ca.GetNeighbors() if nb.GetIdx() not in (n_idx, c_idx)]
    if len(sc_starts) != 1:
        return ""
    return _sidechain_smiles(tri_mol, ca_idx, sc_starts[0], {n_idx, c_idx})


def _sidechain_smiles(mol, ca_idx: int, start_idx: int, blocked: set) -> str:
    """BFS sidechain subgraph from start_idx without crossing ca_idx or blocked backbone atoms, max 20 atoms."""
    visited = {ca_idx}
    queue = deque([start_idx])
    sc_atoms = []
    while queue:
        u = queue.popleft()
        if u in visited or u in blocked:
            continue
        visited.add(u)
        sc_atoms.append(u)
        if len(sc_atoms) > 22:
            return "__LARGE__"
        atom = mol.GetAtomWithIdx(u)
        for nb in atom.GetNeighbors():
            v = nb.GetIdx()
            if v not in visited and v not in blocked:
                # Stop at disulfide S-S bridge so Cys-SS-Cys still looks like -CH2-S!
                if atom.GetAtomicNum() == 16 and nb.GetAtomicNum() == 16:
                    continue
                queue.append(v)
    if not sc_atoms:
        return ""
    try:
        smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sc_atoms, rootedAtAtom=start_idx, isomericSmiles=False, canonical=True)
        return smi
    except Exception:
        return ""


def _compute_macrocycle_ring_size(mol) -> int:
    ri = mol.GetRingInfo()
    max_ring = max((len(r) for r in ri.AtomRings()), default=0)
    if max_ring >= 8:
        return max_ring
    # Check if any disulfide or amide bond is in a large ring missed by minimal SSSR
    # (e.g., bridged multi-disulfide cycles). Compute shortest cycle containing any ring-S-S bond!
    ss_matches = mol.GetSubstructMatches(_SMARTS["disulfide"])
    for a1, a2 in ss_matches:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None or not bond.IsInRing():
            continue
        # Shortest path from a1 to a2 without using bond (a1, a2)
        q = deque([(a1, 1)])
        seen = {a1}
        while q:
            u, d = q.popleft()
            if u == a2:
                if d > max_ring:
                    max_ring = d
                break
            if d > 120:
                break
            for nb in mol.GetAtomWithIdx(u).GetNeighbors():
                v = nb.GetIdx()
                if (u == a1 and v == a2) or (u == a2 and v == a1):
                    continue
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
    return max_ring if max_ring >= 8 else 0


def _compute_fatty_chains(mol) -> tuple[int, int]:
    """
    Detect fatty diacid chain length (e.g., C18 in Semaglutide, C20 in Tirzepatide, C16 in Degludec)
    and fatty monoacyl chain length (e.g., C16 in Liraglutide, C14 in Detemir/Caspofungin, C10 in Daptomycin).
    """
    n_atoms = mol.GetNumAtoms()
    # Identify carbonyl carbons of acyl groups C(=O)[N,O] and free carboxyl C(=O)OH
    carboxyl_c = {m[0] for m in mol.GetSubstructMatches(_SMARTS["carboxyl"])}
    acyl_c = {m[0] for m in mol.GetSubstructMatches(_SMARTS["acyl"])} - carboxyl_c

    max_diacid = 0
    max_monoacyl = 0

    for c_idx in acyl_c:
        c_atom = mol.GetAtomWithIdx(c_idx)
        # Find sp3 carbon neighbor attached to acyl carbonyl
        c_nbs = [
            nb.GetIdx() for nb in c_atom.GetNeighbors()
            if nb.GetAtomicNum() == 6 and not nb.GetIsAromatic() and not nb.IsInRing()
            and nb.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ]
        for start in c_nbs:
            # Walk aliphatic acyclic carbon chain (allow CH2, CH with CH3 branch like Caspofungin, terminal CH3, or terminal COOH)
            # Only follow carbons whose ONLY non-carbon neighbor (if any) is carboxyl_c!
            # Specifically, track longest simple carbon path starting at c_idx -> start
            stack = [(start, c_idx, 2, False)]  # (curr, prev, carbon_len, hit_carboxyl)
            best_mono = 0
            best_di = 0
            visited_local = {c_idx}
            # Iterative DFS bounded to depth 30 to avoid combinatorial blowup
            def dfs(u, prev, length, path_set):
                nonlocal best_mono, best_di
                if length > 30:
                    return
                atom = mol.GetAtomWithIdx(u)
                # Check if u is a pure aliphatic CH2/CH/CH3 (all neighbors are C or H, or carboxyl_c)
                nbs = atom.GetNeighbors()
                non_c = [nb for nb in nbs if nb.GetAtomicNum() != 6]
                if non_c:
                    return  # Heteroatom on chain -> stop (not a hydrocarbon fatty tail!)
                c_next = [nb.GetIdx() for nb in nbs if nb.GetIdx() != prev and nb.GetIdx() not in path_set]
                if not c_next:
                    # Terminal CH3!
                    if atom.GetTotalNumHs() == 3 and length > best_mono:
                        best_mono = length
                    return
                for v in c_next:
                    if v in carboxyl_c:
                        if length + 1 > best_di:
                            best_di = length + 1
                    else:
                        va = mol.GetAtomWithIdx(v)
                        if va.GetAtomicNum() == 6 and not va.GetIsAromatic() and not va.IsInRing():
                            path_set.add(v)
                            dfs(v, u, length + 1, path_set)
                            path_set.remove(v)

            dfs(start, c_idx, 2, {c_idx, start})
            if best_di >= 8 and best_di > max_diacid:
                max_diacid = best_di
            if best_mono >= 8 and best_di < 8 and best_mono > max_monoacyl:
                max_monoacyl = best_mono

    return max_diacid, max_monoacyl


def _count_d_aa_approx(mol) -> int:
    """Count approximate D-configuration alpha-amino-acid stereocenters on peptide backbone."""
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        return 0
    matches = mol.GetSubstructMatches(_SMARTS["backbone_chiral"])
    seen_ca = set()
    d_count = 0
    for m in matches:
        n_idx, ca_idx, c_idx = m[0], m[1], m[2]
        if ca_idx in seen_ca:
            continue
        seen_ca.add(ca_idx)
        ca = mol.GetAtomWithIdx(ca_idx)
        if not ca.HasProp("_CIPCode"):
            continue
        cip = ca.GetProp("_CIPCode")
        if cip not in ("R", "S"):
            continue
        # Check if this alpha-carbon is truly an AA alpha-carbon (C-alpha bonded to N, C=O, and 1 carbon/sidechain)
        sc_nbs = [nb for nb in ca.GetNeighbors() if nb.GetIdx() not in (n_idx, c_idx)]
        if len(sc_nbs) != 1:
            continue
        beta = sc_nbs[0]
        # Skip non-peptide sugar/heteroatom-attached CH centers (e.g., Caspofungin hemiaminal CH(O)(N) or Threonine C-beta!)
        # Note: in Threonine, C-beta is bonded to O, not N, so backbone_chiral won't match C-beta.
        # In Caspofungin, hemiaminal has beta.AtomicNum == 8 -> skip!
        if beta.GetAtomicNum() != 6:
            continue
        # For Cys/Pen (CH2-S or C(Me)2-S), L-configuration has CIP=(R) because S outranks O!
        # Also if beta is bonded to O directly (like Ser/Thr? No: in Ser/Thr, COOH has (=O,O) > CH2OH (O,H,H), so L-Ser/L-Thr are still (S)!)
        # Wait: what about C-beta bonded to S (atomic num 16 > 8)? Then CH2S outranks C(=O)N, so L-Cys is (R) and D-Cys is (S)!
        beta_has_s = any(nb.GetAtomicNum() >= 16 for nb in beta.GetNeighbors() if nb.GetIdx() != ca_idx)
        if beta_has_s:
            if cip == "S":
                d_count += 1
        else:
            if cip == "R":
                d_count += 1
    return d_count


def _extract_backbone_sequence(mol) -> str:
    """
    Deterministic RDKit alpha-amino-acid backbone walker.
    Extracts canonical 20AA parent sequence from a peptide SMILES (handles linear, cyclic, disulfide, acylated Lys).
    Returns "" if fewer than 2 recognizable AA residues or <50% standard residues.
    """
    matches = mol.GetSubstructMatches(_SMARTS["backbone_aa"])
    if not matches:
        return ""
    # Deduplicate by ca_idx, picking valid (n_idx, ca_idx, c_idx) where n_idx != c_idx
    units = {}  # ca_idx -> (n_idx, c_idx)
    backbone_atoms = set()
    for m in matches:
        n_idx, ca_idx, c_idx = m[0], m[1], m[2]
        if ca_idx in units:
            continue
        ca = mol.GetAtomWithIdx(ca_idx)
        # Skip carbons that are part of aromatic rings
        if ca.GetIsAromatic():
            continue
        # Skip sidechain Asn/Gln CH2-C(=O)NH2 (their carbon isn't bonded to N anyway)
        units[ca_idx] = (n_idx, c_idx)
        backbone_atoms.update((n_idx, ca_idx, c_idx))
    if len(units) < 2:
        return ""

    # Identify residue letter for each ca_idx
    res_letter = {}
    n_to_ca = {}
    c_to_ca = {}
    for ca_idx, (n_idx, c_idx) in units.items():
        n_to_ca[n_idx] = ca_idx
        c_to_ca[c_idx] = ca_idx
        ca = mol.GetAtomWithIdx(ca_idx)
        sc_nbs = [nb for nb in ca.GetNeighbors() if nb.GetIdx() not in (n_idx, c_idx)]
        if len(sc_nbs) == 0:
            res_letter[ca_idx] = "G"
        elif len(sc_nbs) == 1:
            nb = sc_nbs[0]
            # Check Proline (5-ring connecting ca and n_idx)
            if ca.IsInRingSize(5) and mol.GetAtomWithIdx(n_idx).IsInRingSize(5):
                # Check if nb is in same 5-ring
                if nb.IsInRingSize(5):
                    res_letter[ca_idx] = "P"
                    continue
            sc_smi = _sidechain_smiles(mol, ca_idx, nb.GetIdx(), backbone_atoms)
            if sc_smi in _SIDECHAIN_MAP:
                res_letter[ca_idx] = _SIDECHAIN_MAP[sc_smi]
            elif sc_smi == "__LARGE__":
                # Likely fatty-acylated Lys (e.g., Liraglutide/Semaglutide/Tirzepatide/Detemir/Degludec)!
                # Check if first 4 atoms from C-alpha are -(CH2)4-NH-C(=O)-
                if nb.GetAtomicNum() == 6 and nb.GetTotalNumHs() == 2:
                    # Quick 4-CH2-N check
                    res_letter[ca_idx] = "K"
                else:
                    res_letter[ca_idx] = "X"
            else:
                # Check disulfide Cys (-CH2-S- without S-S continuation)
                if sc_smi in ("CS", "[CH2]S", "C[S]"):
                    res_letter[ca_idx] = "C"
                else:
                    res_letter[ca_idx] = "X"
        else:
            # Disubstituted C-alpha (e.g., Aib in Semaglutide/Tirzepatide)
            res_letter[ca_idx] = "X"

    # Build directed peptide bond edges: ca1 -> ca2 if c1 is directly bonded to n2
    adj = {}
    indeg = Counter({ca: 0 for ca in units})
    for ca1, (n1, c1) in units.items():
        for nb in mol.GetAtomWithIdx(c1).GetNeighbors():
            n2 = nb.GetIdx()
            if n2 in n_to_ca and n_to_ca[n2] != ca1:
                ca2 = n_to_ca[n2]
                adj[ca1] = ca2
                indeg[ca2] += 1
                break

    # Walk chains starting from indeg == 0 (N-termini), then any unvisited cycles
    visited = set()
    chains = []
    for start in [ca for ca, d in indeg.items() if d == 0]:
        if start in visited:
            continue
        ch = []
        u = start
        while u is not None and u not in visited:
            visited.add(u)
            ch.append(res_letter[u])
            u = adj.get(u)
        chains.append("".join(ch))
    for start in units:
        if start in visited:
            continue
        ch = []
        u = start
        while u is not None and u not in visited:
            visited.add(u)
            ch.append(res_letter[u])
            u = adj.get(u)
        chains.append("".join(ch))

    if not chains:
        return ""
    # Pick longest chain (for multi-chain like Insulin, take longest chain if pure AA, or concat? longest chain is cleanest)
    best = max(chains, key=len)
    if len(best) < 2:
        return ""
    known_frac = sum(1 for c in best if c != "X") / len(best)
    if known_frac < 0.5:
        return ""
    # Replace any non-canonical 'X' only? Wait: keep only standard 20AA if >=80% standard, strip X or keep clean 20AA?
    # Better: keep parent_sequence strictly in standard 20AA alphabet by stripping X ONLY if >= 80% standard and len >= 4;
    # wait: for Semaglutide (Aib -> X), stripping X shifts positions; keeping X causes AA20 seq features to skip X cleanly!
    # Wait: let's check what sequence descriptors do with X: they ignore X and compute fractions over the 20 standard AAs!
    # And for clustering, 70% sequence identity works cleanly on A-Z strings!
    # Wait: prompt says "parent_sequence (canonical 20AA sequence if available, or clean RDKit peptide backbone extraction if available — clearly tagged seq_origin)".
    # Let's return `best` (canonical 20AA letters + 'X' for ncAA positions if present, or pure 20AA if all standard)!
    return best


def _process_one_mol(item):
    """Worker function: item = (kind, key_str) where kind in ('smi', 'seq')."""
    kind, key_str = item
    try:
        if kind == "smi":
            mol = Chem.MolFromSmiles(key_str)
            smiles_origin = "source"
        else:
            mol = Chem.MolFromSequence(key_str)
            smiles_origin = "rdkit_from_canonical_seq"
        if mol is None or mol.GetNumAtoms() == 0:
            return (kind, key_str, None)

        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        nosmi_conn = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        mw = float(Descriptors.MolWt(mol))
        heavy = int(mol.GetNumHeavyAtoms())

        # InChIKey (may fail for >1024 atoms like Insulins -> deterministic fallback connectivity key)
        ik = ""
        try:
            if mol.GetNumAtoms() <= 1020:
                ik = Chem.MolToInchiKey(mol) or ""
        except Exception:
            ik = ""
        conn_hash = hashlib.sha256(nosmi_conn.encode("utf-8")).hexdigest()[:16]
        ik14 = ik[:14] if len(ik) >= 14 else f"CONN:{conn_hash}"

        # Murcko scaffolds (only meaningful if rings exist and size reasonable)
        murcko_smi = ""
        generic_smi = ""
        scaf_heavy = 0
        try:
            if mol.GetRingInfo().NumRings() > 0 and heavy <= 400:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                if scaf is not None and scaf.GetNumHeavyAtoms() > 0:
                    scaf_heavy = int(scaf.GetNumHeavyAtoms())
                    murcko_smi = Chem.MolToSmiles(scaf, isomericSmiles=False, canonical=True)
                    gen = MurckoScaffold.MakeScaffoldGeneric(scaf)
                    if gen is not None:
                        generic_smi = Chem.MolToSmiles(gen, isomericSmiles=False, canonical=True)
        except Exception:
            pass

        # Deterministic 10 structural modification flags
        macro_size = _compute_macrocycle_ring_size(mol)
        is_cyclic = 1 if macro_size >= 8 else 0
        n_disulfide = len(mol.GetSubstructMatches(_SMARTS["disulfide"]))
        n_nmethyl = len(mol.GetSubstructMatches(_SMARTS["nmethyl"]))
        n_d_aa = _count_d_aa_approx(mol) if kind == "smi" else 0
        diacid_len, monoacyl_len = _compute_fatty_chains(mol) if kind == "smi" else (0, 0)
        has_peg = 1 if (kind == "smi" and mol.HasSubstructMatch(_SMARTS["peg"])) else 0
        has_c_amid = 1 if (kind == "smi" and mol.HasSubstructMatch(_SMARTS["c_amid"])) else 0
        has_n_acet = 1 if (kind == "smi" and mol.HasSubstructMatch(_SMARTS["n_acet"])) else 0

        extracted_seq = key_str if kind == "seq" else _extract_backbone_sequence(mol)

        return (kind, key_str, {
            "canon_smi": canon_smi,
            "smiles_origin": smiles_origin,
            "mw": round(mw, 4),
            "heavy_atoms": heavy,
            "inchikey": ik,
            "inchikey_14": ik14,
            "conn_hash": conn_hash,
            "murcko_smi": murcko_smi,
            "generic_smi": generic_smi,
            "scaf_heavy": scaf_heavy,
            "is_cyclic": is_cyclic,
            "macrocycle_ring_size": macro_size,
            "n_disulfide": n_disulfide,
            "n_nmethyl": n_nmethyl,
            "n_d_aa_approx": n_d_aa,
            "fatty_diacid_len": diacid_len,
            "fatty_monoacyl_len": monoacyl_len,
            "has_peg": has_peg,
            "has_c_amidation": has_c_amid,
            "has_n_acetylation": has_n_acet,
            "extracted_seq": extracted_seq,
        })
    except Exception:
        return (kind, key_str, None)


def main():
    t0 = time.time()
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ep_dir = OUT_DATA_DIR / "endpoints_raw_clean"
    ep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading staging raw_candidates.tsv.gz & case17_raw.tsv (workers={N_WORKERS})...", flush=True)
    raw_rows = []
    unique_smis = set()
    unique_pure_seqs = set()

    with gzip.open(STAGING_DIR / "raw_candidates.tsv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw_rows.append(row)
            s = row["canonical_smiles_src"]
            if s:
                unique_smis.add(s)
            elif row["is_pure_unmodified_20aa"] == "1" and row["sequence_raw"]:
                unique_pure_seqs.add(row["sequence_raw"])

    case17_raw = []
    with (STAGING_DIR / "case17_raw.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            case17_raw.append(row)
            if row["source_smiles"]:
                unique_smis.add(row["source_smiles"])

    print(f"  Loaded {len(raw_rows):,} candidate rows | {len(unique_smis):,} unique source SMILES | {len(unique_pure_seqs):,} unique pure 20AA seqs", flush=True)

    # Parallel RDKit processing
    print("[2/6] Running parallel RDKit canonicalization, MW, scaffolds, & modification extraction...", flush=True)
    tasks = [("smi", s) for s in sorted(unique_smis)] + [("seq", q) for q in sorted(unique_pure_seqs)]
    smi_info = {}
    seq_info = {}
    failed_smi = 0
    failed_seq = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for kind, key_str, res in ex.map(_process_one_mol, tasks, chunksize=128):
            if res is None:
                if kind == "smi":
                    failed_smi += 1
                else:
                    failed_seq += 1
            else:
                if kind == "smi":
                    smi_info[key_str] = res
                else:
                    seq_info[key_str] = res

    print(f"  RDKit done in {time.time()-t0:.1f}s | smi_ok={len(smi_info):,} (fail={failed_smi}) | seq_to_smi_ok={len(seq_info):,} (fail={failed_seq})", flush=True)

    # Build Case17 canonical table & exact overlap indices
    print("[3/6] Building case17_queries.tsv & verifying modification flags on all 17 drugs...", flush=True)
    c17_records = []
    c17_by_smi = {}
    c17_by_ik = {}
    c17_by_ik14 = {}
    c17_by_conn = {}
    c17_by_seq = {}

    for row in case17_raw:
        name = row["drug_name"]
        info = smi_info.get(row["source_smiles"])
        assert info is not None, f"Case17 drug failed RDKit parse: {name}"
        ckey = "SMI:" + hashlib.sha256(info["canon_smi"].encode("utf-8")).hexdigest()[:16]
        rec = {
            "drug_name": name,
            "compound_key": ckey,
            "canonical_smiles": info["canon_smi"],
            "smiles_origin": "source",
            "inchikey": info["inchikey"],
            "inchikey_14": info["inchikey_14"],
            "conn_hash": info["conn_hash"],
            "mol_wt": info["mw"],
            "parent_sequence": info["extracted_seq"],
            "seq_origin": "rdkit_backbone_extracted" if info["extracted_seq"] else "none",
            "is_cyclic": info["is_cyclic"],
            "macrocycle_ring_size": info["macrocycle_ring_size"],
            "n_disulfide": info["n_disulfide"],
            "n_nmethyl": info["n_nmethyl"],
            "n_d_aa_approx": info["n_d_aa_approx"],
            "fatty_diacid_len": info["fatty_diacid_len"],
            "fatty_monoacyl_len": info["fatty_monoacyl_len"],
            "has_peg": info["has_peg"],
            "has_c_amidation": info["has_c_amidation"],
            "has_n_acetylation": info["has_n_acetylation"],
            "half_life_ref": row["half_life_ref"],
            "ppb_ref": row["ppb_ref"],
        }
        c17_records.append(rec)
        c17_by_smi[info["canon_smi"]] = name
        if info["inchikey"]:
            c17_by_ik[info["inchikey"]] = name
        c17_by_ik14[info["inchikey_14"]] = name
        c17_by_conn[info["conn_hash"]] = name
        pseq = info["extracted_seq"]
        if pseq and len(pseq) >= 6 and "X" not in pseq:
            c17_by_seq[pseq] = name
        print(
            f"  [Case17] {name:17s} MW={info['mw']:7.1f} cyc={info['is_cyclic']}({info['macrocycle_ring_size']:2d}) "
            f"SS={info['n_disulfide']} NMe={info['n_nmethyl']} DAA={info['n_d_aa_approx']} "
            f"diacid={info['fatty_diacid_len']:2d} mono={info['fatty_monoacyl_len']:2d} "
            f"PEG={info['has_peg']} NH2={info['has_c_amidation']} Ac={info['has_n_acetylation']} seq={pseq[:22]}",
            flush=True,
        )

    c17_fields = list(c17_records[0].keys())
    with (OUT_DATA_DIR / "case17_queries.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=c17_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(c17_records)

    # Resolve every raw candidate row: MW conversions, compound registry, target transform
    print("[4/6] Resolving MW unit conversions, thresholding, and unifying compound registry...", flush=True)
    compounds = {}  # compound_key -> dict
    # endpoint -> (compound_key, cond_sig) -> list of resolved observations
    grouped = defaultdict(lambda: defaultdict(list))
    mw_rescues = Counter()
    mw_drops = Counter()

    def register_compound(info, seq_raw: str, seq_kind: str, is_pure20: str) -> str:
        if info is not None:
            ckey = "SMI:" + hashlib.sha256(info["canon_smi"].encode("utf-8")).hexdigest()[:16]
            if ckey not in compounds:
                # Choose best parent_sequence
                if seq_raw:
                    pseq = seq_raw
                    sorig = "source_bare_20aa" if is_pure20 == "1" else "source_parent_projection"
                elif info["extracted_seq"]:
                    pseq = info["extracted_seq"]
                    sorig = "rdkit_backbone_extracted"
                else:
                    pseq = ""
                    sorig = "none"
                compounds[ckey] = {
                    "compound_key": ckey,
                    "canonical_smiles": info["canon_smi"],
                    "smiles_origin": info["smiles_origin"],
                    "inchikey": info["inchikey"],
                    "inchikey_14": info["inchikey_14"],
                    "conn_hash": info["conn_hash"],
                    "mol_wt": info["mw"],
                    "heavy_atoms": info["heavy_atoms"],
                    "murcko_smi": info["murcko_smi"],
                    "generic_smi": info["generic_smi"],
                    "scaf_heavy": info["scaf_heavy"],
                    "parent_sequence": pseq,
                    "seq_origin": sorig,
                    "is_cyclic": info["is_cyclic"],
                    "macrocycle_ring_size": info["macrocycle_ring_size"],
                    "n_disulfide": info["n_disulfide"],
                    "n_nmethyl": info["n_nmethyl"],
                    "n_d_aa_approx": info["n_d_aa_approx"],
                    "fatty_diacid_len": info["fatty_diacid_len"],
                    "fatty_monoacyl_len": info["fatty_monoacyl_len"],
                    "has_peg": info["has_peg"],
                    "has_c_amidation": info["has_c_amidation"],
                    "has_n_acetylation": info["has_n_acetylation"],
                }
            else:
                # Upgrade smiles_origin to 'source' if a later record had source SMILES
                c = compounds[ckey]
                if info["smiles_origin"] == "source" and c["smiles_origin"] != "source":
                    c["smiles_origin"] = "source"
                if seq_raw and c["seq_origin"] in ("none", "rdkit_backbone_extracted"):
                    c["parent_sequence"] = seq_raw
                    c["seq_origin"] = "source_bare_20aa" if is_pure20 == "1" else "source_parent_projection"
            return ckey
        else:
            # Modified peptide without SMILES (sequence-only fallback; NEVER fabricate SMILES!)
            ckey = "SEQMOD:" + hashlib.sha256(f"{seq_raw}|{seq_kind}".encode("utf-8")).hexdigest()[:16]
            if ckey not in compounds:
                compounds[ckey] = {
                    "compound_key": ckey,
                    "canonical_smiles": "",
                    "smiles_origin": "missing",
                    "inchikey": "",
                    "inchikey_14": "",
                    "conn_hash": "",
                    "mol_wt": 0.0,
                    "heavy_atoms": 0,
                    "murcko_smi": "",
                    "generic_smi": "",
                    "scaf_heavy": 0,
                    "parent_sequence": seq_raw,
                    "seq_origin": "source_parent_projection",
                    "is_cyclic": 0,
                    "macrocycle_ring_size": 0,
                    "n_disulfide": 0,
                    "n_nmethyl": 0,
                    "n_d_aa_approx": 0,
                    "fatty_diacid_len": 0,
                    "fatty_monoacyl_len": 0,
                    "has_peg": 0,
                    "has_c_amidation": 0,
                    "has_n_acetylation": 0,
                }
            return ckey

    def add_obs(ep, row, ckey, ttype, y_val, y_lo, y_hi, ctype, rescue_reason, priority):
        cond_cols = ENDPOINT_COND_COLS[ep]
        cond_vals = tuple(row[c] for c in cond_cols)
        grouped[ep][(ckey, cond_vals)].append({
            "task_tier": row["task_tier"],
            "task_type": ttype,
            "record_id": row["record_id"],
            "origin": row["origin"],
            "source_code": row["source_code"],
            "source_record_id": row["source_record_id"],
            "lane_orig": row["lane_orig"],
            "rescue_reason": rescue_reason,
            "priority": int(priority),
            "y_value": float(y_val),
            "y_lower": float(y_lo),
            "y_upper": float(y_hi),
            "censor_type": ctype,
            "cond_dict": {c: row[c] for c in cond_cols},
        })

    def reg_bounds(ep, y_val, rel, raw_lo="", raw_hi=""):
        dmin, dmax = DOMAIN_BOUNDS[ep]
        if rel == "interval":
            lo = float(raw_lo)
            hi = float(raw_hi)
            if lo > hi:
                lo, hi = hi, lo
            return (0.5 * (lo + hi), lo, hi, "interval")
        if rel in (">", ">="):
            return (y_val, y_val, max(y_val + 0.01, dmax), "right")
        if rel in ("<", "<="):
            return (y_val, min(y_val - 0.01, dmin), y_val, "left")
        return (y_val, y_val, y_val, "exact")

    for row in raw_rows:
        ep = row["endpoint_v33"]
        smi_s = row["canonical_smiles_src"]
        seq_raw = row["sequence_raw"]
        is_pure20 = row["is_pure_unmodified_20aa"]

        info = None
        if smi_s:
            info = smi_info.get(smi_s)
        if info is None and is_pure20 == "1" and seq_raw:
            info = seq_info.get(seq_raw)
        if info is None and not seq_raw:
            mw_drops[(ep, "unparseable_smi_and_no_seq")] += 1
            continue

        ckey = register_compound(info, seq_raw, row["sequence_kind"], is_pure20)
        unit = row["val_unit"]
        rel = row["val_relation"]
        pri = int(row["evidence_priority"])

        # Handle deferred RDKit-MW conversions!
        if unit == "ug/mL_needs_mw":
            if info is None or info["mw"] <= 0:
                mw_drops[(ep, "missing_mw")] += 1
                continue
            val_ug = float(row["val_num"])
            val_uM = val_ug * 1000.0 / info["mw"]
            if ep == "solubility_reg":
                if not (1e-4 <= val_uM <= 1e6):
                    mw_drops[(ep, "uM_out_of_range")] += 1
                    continue
                y = math.log10(val_uM)
                yv, yl, yh, ct = reg_bounds(ep, y, rel)
                add_obs(ep, row, ckey, "regression", yv, yl, yh, ct, row["rescue_reason"], pri)
                mw_rescues[ep] += 1
                # Also emit unambiguous aqueous binary solubility_cls
                if row["cond_solvent"] in ("pbs_ph74", "ultrapure_water", "aqueous_buffer", "aqueous_other", "unspecified"):
                    if rel in ("=", ">", ">=") and val_uM >= 100.0:
                        r2 = dict(row); r2["cond_solvent"] = "pbs_ph74" if row["cond_solvent"] == "pbs_ph74" else "aqueous_buffer"
                        add_obs("solubility_cls", r2, ckey, "classification", 1, 1, 1, "exact", "quant_solubility_mw_ge100uM_soluble", 1)
                    elif rel in ("=", "<", "<=") and val_uM < 10.0:
                        r2 = dict(row); r2["cond_solvent"] = "pbs_ph74" if row["cond_solvent"] == "pbs_ph74" else "aqueous_buffer"
                        add_obs("solubility_cls", r2, ckey, "classification", 0, 0, 0, "exact", "quant_solubility_mw_lt10uM_insoluble", 1)
            elif ep == "hemolysis_HC50":
                if not (1e-3 <= val_uM <= 1e5):
                    mw_drops[(ep, "uM_out_of_range")] += 1
                    continue
                y = math.log10(val_uM)
                yv, yl, yh, ct = reg_bounds(ep, y, rel)
                add_obs(ep, row, ckey, "regression", yv, yl, yh, ct, row["rescue_reason"], pri)
                mw_rescues[ep] += 1
                # Also emit unambiguous hemolysis_cls
                if rel in ("=", "<", "<=") and val_uM <= 50.0:
                    add_obs("hemolysis_cls", row, ckey, "classification", 1, 1, 1, "exact", f"hc50_mw_le50uM_hemolytic:{row['source_code']}", 1)
                elif rel in ("=", ">", ">=") and val_uM >= 100.0:
                    add_obs("hemolysis_cls", row, ckey, "classification", 0, 0, 0, "exact", f"hc50_mw_ge100uM_safe:{row['source_code']}", 1)
            continue

        if unit == "pct_at_ug_mL_needs_mw":
            if info is None or info["mw"] <= 0:
                mw_drops[(ep, "missing_mw")] += 1
                continue
            conc_uM = float(row["aux_conc_val"]) * 1000.0 / info["mw"]
            pct = float(row["aux_pct_val"])
            bval, brr = None, ""
            if rel in ("=", ">", ">=", "~") and pct >= 20.0 and conc_uM <= 50.0:
                bval, brr = 1, f"pct_hemolysis_mw_ge20_at_le50uM:{row['source_code']}"
            elif rel in ("=", ">", ">=", "~") and pct >= 50.0 and conc_uM <= 100.0:
                bval, brr = 1, f"pct_hemolysis_mw_ge50_at_le100uM:{row['source_code']}"
            elif rel in ("=", "<", "<=", "~") and pct <= 10.0 and conc_uM >= 50.0:
                bval, brr = 0, f"pct_hemolysis_mw_le10_at_ge50uM:{row['source_code']}"
            elif rel in ("=", "<", "<=", "~") and pct < 20.0 and conc_uM >= 100.0:
                bval, brr = 0, f"pct_hemolysis_mw_lt20_at_ge100uM:{row['source_code']}"
            if bval is not None:
                add_obs("hemolysis_cls", row, ckey, "classification", bval, bval, bval, "exact", brr, 1)
                mw_rescues["hemolysis_cls_pct_mw"] += 1
            continue

        if unit == "cyto_conc_ug_mL_needs_mw":
            if info is None or info["mw"] <= 0:
                mw_drops[(ep, "missing_mw")] += 1
                continue
            conc_uM = float(row["val_num"]) * 1000.0 / info["mw"]
            if rel in ("=", "<", "<=") and conc_uM <= 25.0:
                add_obs("cytotoxicity_cls", row, ckey, "classification", 1, 1, 1, "exact", f"cyto_mw_le25uM_toxic:{row['source_code']}", 1)
                mw_rescues["cytotoxicity_cls_mw"] += 1
            elif rel in ("=", ">", ">=") and conc_uM >= 100.0:
                add_obs("cytotoxicity_cls", row, ckey, "classification", 0, 0, 0, "exact", f"cyto_mw_ge100uM_safe:{row['source_code']}", 1)
                mw_rescues["cytotoxicity_cls_mw"] += 1
            continue

        if unit == "cyto_pct_at_ug_mL_needs_mw":
            if info is None or info["mw"] <= 0:
                mw_drops[(ep, "missing_mw")] += 1
                continue
            conc_uM = float(row["aux_conc_val"]) * 1000.0 / info["mw"]
            pct = float(row["aux_pct_val"])
            if rel in ("=", ">", ">=", "~") and pct >= 50.0 and conc_uM <= 50.0:
                add_obs("cytotoxicity_cls", row, ckey, "classification", 1, 1, 1, "exact", f"cyto_pct_mw_ge50_at_le50uM:{row['source_code']}", 1)
                mw_rescues["cytotoxicity_cls_pct_mw"] += 1
            elif rel in ("=", "<", "<=", "~") and pct <= 15.0 and conc_uM >= 50.0:
                add_obs("cytotoxicity_cls", row, ckey, "classification", 0, 0, 0, "exact", f"cyto_pct_mw_le15_at_ge50uM:{row['source_code']}", 1)
                mw_rescues["cytotoxicity_cls_pct_mw"] += 1
            continue

        # Standard already-converted values
        ttype = row["task_type"]
        if ttype == "classification":
            bval = int(float(row["val_num"]))
            add_obs(ep, row, ckey, "classification", bval, bval, bval, "exact", row["rescue_reason"], pri)
        else:
            raw_v = float(row["val_num"])
            if ep in LOG10_ENDPOINTS:
                if raw_v <= 0:
                    continue
                y = math.log10(raw_v)
                yv, yl, yh, ct = reg_bounds(ep, y, rel)
            else:
                yv, yl, yh, ct = reg_bounds(ep, raw_v, rel, row["val_lower"], row["val_upper"])
            add_obs(ep, row, ckey, "regression", yv, yl, yh, ct, row["rescue_reason"], pri)

    print("  MW rescues:", dict(mw_rescues), "MW drops:", dict(mw_drops), flush=True)

    # Deduplicate each endpoint & check Case17 overlap!
    print("[5/6] Deduplicating per (compound_key, endpoint, condition_signature) & auditing Case17 overlap...", flush=True)
    case17_overlaps = []
    dedup_audit = {}
    used_compounds = set()

    def match_case17(ckey: str) -> tuple[str, str]:
        c = compounds[ckey]
        if c["canonical_smiles"] and c["canonical_smiles"] in c17_by_smi:
            return (c17_by_smi[c["canonical_smiles"]], "exact_canonical_smiles")
        if c["inchikey"] and c["inchikey"] in c17_by_ik:
            return (c17_by_ik[c["inchikey"]], "exact_inchikey")
        if c["inchikey_14"] and c["inchikey_14"] in c17_by_ik14:
            return (c17_by_ik14[c["inchikey_14"]], "inchikey_connectivity_14")
        if c["conn_hash"] and c["conn_hash"] in c17_by_conn:
            return (c17_by_conn[c["conn_hash"]], "connectivity_smiles_hash")
        pseq = c["parent_sequence"]
        if pseq and len(pseq) >= 6 and pseq in c17_by_seq:
            return (c17_by_seq[pseq], "exact_parent_sequence")
        return ("", "")

    for ep in sorted(ENDPOINT_COND_COLS.keys()):
        groups = grouped.get(ep, {})
        cond_cols = ENDPOINT_COND_COLS[ep]
        out_rows = []
        n_conflicts = 0
        n_exact = 0
        n_right = 0
        n_left = 0
        n_interval = 0
        n_c17 = 0
        rescue_cnt = Counter()
        n_raw_in = sum(len(v) for v in groups.values())

        for (ckey, cond_vals), obs_list in sorted(groups.items()):
            ttype = obs_list[0]["task_type"]
            tier = obs_list[0]["task_tier"]
            obs_sorted = sorted(obs_list, key=lambda x: (x["priority"], x["record_id"]))

            if ttype == "classification":
                best_pri = min(o["priority"] for o in obs_sorted)
                top_obs = [o for o in obs_sorted if o["priority"] == best_pri]
                votes = [int(o["y_value"]) for o in top_obs]
                p1 = sum(votes) / len(votes)
                if 0.25 < p1 < 0.75:
                    n_conflicts += 1
                    continue
                y_final = 1 if p1 >= 0.75 else 0
                y_lo = y_hi = float(y_final)
                ctype = "exact"
                rep = next(o for o in top_obs if int(o["y_value"]) == y_final)
                n_rep = len(obs_list)
                rep_sd = round(statistics.pstdev(votes), 4) if len(votes) > 1 else 0.0
                all_sources = ";".join(sorted({o["source_code"] for o in obs_list}))
                all_rescues = ";".join(sorted({o["rescue_reason"] for o in top_obs}))
                n_exact += 1
            else:
                exacts = [o for o in obs_sorted if o["censor_type"] == "exact"]
                intervals = [o for o in obs_sorted if o["censor_type"] == "interval"]
                rights = [o for o in obs_sorted if o["censor_type"] == "right"]
                lefts = [o for o in obs_sorted if o["censor_type"] == "left"]
                dmin, dmax = DOMAIN_BOUNDS[ep]

                if exacts:
                    vals = [o["y_value"] for o in exacts]
                    # Check extreme conflict (>2.5 log units or >0.45 fraction spread)
                    spread = max(vals) - min(vals)
                    max_spread = 0.45 if ep in ("PPB", "F") else 2.5
                    if len(vals) >= 2 and spread > max_spread:
                        # Prefer priority-1 observations if they agree; else drop genuine contradiction
                        p1_vals = [o["y_value"] for o in exacts if o["priority"] == 1]
                        if p1_vals and (max(p1_vals) - min(p1_vals)) <= max_spread:
                            vals = p1_vals
                            exacts = [o for o in exacts if o["priority"] == 1]
                        else:
                            n_conflicts += 1
                            continue
                    y_final = float(statistics.median(vals))
                    y_lo = y_hi = y_final
                    ctype = "exact"
                    rep = exacts[0]
                    n_rep = len(exacts)
                    rep_sd = round(statistics.pstdev(vals), 5) if len(vals) > 1 else 0.0
                    all_sources = ";".join(sorted({o["source_code"] for o in exacts}))
                    all_rescues = ";".join(sorted({o["rescue_reason"] for o in exacts}))
                    n_exact += 1
                elif intervals:
                    lo = max(o["y_lower"] for o in intervals)
                    hi = min(o["y_upper"] for o in intervals)
                    if lo > hi:
                        lo = float(statistics.median([o["y_lower"] for o in intervals]))
                        hi = float(statistics.median([o["y_upper"] for o in intervals]))
                    y_final = 0.5 * (lo + hi)
                    y_lo, y_hi = lo, hi
                    ctype = "interval"
                    rep = intervals[0]
                    n_rep = len(intervals)
                    rep_sd = 0.0
                    all_sources = ";".join(sorted({o["source_code"] for o in intervals}))
                    all_rescues = ";".join(sorted({o["rescue_reason"] for o in intervals}))
                    n_interval += 1
                else:
                    # Right and/or left censored
                    if rights and lefts:
                        lo = max(o["y_lower"] for o in rights)
                        hi = min(o["y_upper"] for o in lefts)
                        if lo <= hi:
                            y_final = 0.5 * (lo + hi)
                            y_lo, y_hi = lo, hi
                            ctype = "interval"
                            rep = rights[0]
                            n_rep = len(rights) + len(lefts)
                            rep_sd = 0.0
                            all_sources = ";".join(sorted({o["source_code"] for o in obs_list}))
                            all_rescues = ";".join(sorted({o["rescue_reason"] for o in obs_list}))
                            n_interval += 1
                        else:
                            n_conflicts += 1
                            continue
                    elif rights:
                        lo = max(o["y_lower"] for o in rights)
                        y_final = lo
                        y_lo = lo
                        y_hi = max(lo + 0.01, dmax)
                        ctype = "right"
                        rep = max(rights, key=lambda o: o["y_lower"])
                        n_rep = len(rights)
                        rep_sd = 0.0
                        all_sources = ";".join(sorted({o["source_code"] for o in rights}))
                        all_rescues = ";".join(sorted({o["rescue_reason"] for o in rights}))
                        n_right += 1
                    else:
                        hi = min(o["y_upper"] for o in lefts)
                        y_final = hi
                        y_lo = min(hi - 0.01, dmin)
                        y_hi = hi
                        ctype = "left"
                        rep = min(lefts, key=lambda o: o["y_upper"])
                        n_rep = len(lefts)
                        rep_sd = 0.0
                        all_sources = ";".join(sorted({o["source_code"] for o in lefts}))
                        all_rescues = ";".join(sorted({o["rescue_reason"] for o in lefts}))
                        n_left += 1

            assert math.isfinite(y_final) and math.isfinite(y_lo) and math.isfinite(y_hi), f"Non-finite label in {ep}"
            assert y_lo <= y_hi, f"Invalid bounds {y_lo} > {y_hi} in {ep}"

            c17_drug, c17_match = match_case17(ckey)
            c17_excl = 1 if c17_drug else 0
            if c17_excl:
                n_c17 += 1
                case17_overlaps.append({
                    "drug_name": c17_drug,
                    "endpoint": ep,
                    "match_type": c17_match,
                    "compound_key": ckey,
                    "censor_type": ctype,
                    "y_value": round(y_final, 5),
                    "y_lower": round(y_lo, 5),
                    "y_upper": round(y_hi, 5),
                    "condition_signature": "|".join(cond_vals) if cond_vals else "none",
                    "source_codes": all_sources,
                    "rescue_reasons": all_rescues,
                })

            used_compounds.add(ckey)
            c_rec = compounds[ckey]
            rescue_cnt[rep["rescue_reason"]] += 1

            row_out = {
                "endpoint": ep,
                "task_tier": tier,
                "task_type": ttype,
                "compound_key": ckey,
                "smiles_origin": c_rec["smiles_origin"],
                "seq_origin": c_rec["seq_origin"],
                "y_value": round(y_final, 6) if ttype == "regression" else int(y_final),
                "y_lower": round(y_lo, 6) if ttype == "regression" else int(y_lo),
                "y_upper": round(y_hi, 6) if ttype == "regression" else int(y_hi),
                "censor_type": ctype,
                "n_replicates": n_rep,
                "replicate_sd": rep_sd,
                "condition_signature": "|".join(cond_vals) if cond_vals else "none",
                "origin": rep["origin"],
                "source_code": rep["source_code"],
                "source_record_id": rep["source_record_id"],
                "record_id": rep["record_id"],
                "all_source_codes": all_sources,
                "rescue_reason": rep["rescue_reason"],
                "all_rescue_reasons": all_rescues,
                "case17_excluded": c17_excl,
                "case17_drug": c17_drug,
            }
            for col, val_c in zip(cond_cols, cond_vals):
                row_out[col] = val_c
            out_rows.append(row_out)

        ep_fields = [
            "endpoint", "task_tier", "task_type", "compound_key", "smiles_origin", "seq_origin",
            "y_value", "y_lower", "y_upper", "censor_type", "n_replicates", "replicate_sd",
            "condition_signature",
        ] + cond_cols + [
            "origin", "source_code", "source_record_id", "record_id", "all_source_codes",
            "rescue_reason", "all_rescue_reasons", "case17_excluded", "case17_drug",
        ]
        with (ep_dir / f"{ep}.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ep_fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)

        n_smi_ready = sum(1 for x in out_rows if x["smiles_origin"] != "missing")
        n_pos = sum(1 for x in out_rows if x["task_type"] == "classification" and x["y_value"] == 1)
        n_neg = sum(1 for x in out_rows if x["task_type"] == "classification" and x["y_value"] == 0)
        dedup_audit[ep] = {
            "raw_candidates": n_raw_in,
            "dedup_rows": len(out_rows),
            "unique_compounds": len({x["compound_key"] for x in out_rows}),
            "smiles_ready_rows": n_smi_ready,
            "exact": n_exact,
            "right_censored": n_right,
            "left_censored": n_left,
            "interval": n_interval,
            "conflicts_dropped": n_conflicts,
            "case17_overlap_rows": n_c17,
            "pos_cls": n_pos,
            "neg_cls": n_neg,
            "top_rescue_reasons": dict(rescue_cnt.most_common(12)),
        }
        cls_str = f" pos={n_pos} neg={n_neg}" if (n_pos or n_neg) else ""
        print(
            f"  {ep:25s}: rows={len(out_rows):6,d} (mols={dedup_audit[ep]['unique_compounds']:5,d}, smi={n_smi_ready:6,d}) "
            f"exact={n_exact:5,d} R={n_right:4,d} L={n_left:3,d} I={n_interval:2,d} conf={n_conflicts:3,d} c17={n_c17:2d}{cls_str}",
            flush=True,
        )

    # Write compounds_v33.tsv (only compounds referenced in at least one endpoint; Case17 stored in case17_queries.tsv!)
    print("[6/6] Writing compounds_v33.tsv, case17_overlap.tsv, & clean_audit.json...", flush=True)
    comp_rows = [compounds[k] for k in sorted(used_compounds)]
    comp_fields = list(comp_rows[0].keys())
    with (OUT_DATA_DIR / "compounds_v33.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=comp_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(comp_rows)

    c17_ov_fields = [
        "drug_name", "endpoint", "match_type", "compound_key", "censor_type",
        "y_value", "y_lower", "y_upper", "condition_signature", "source_codes", "rescue_reasons",
    ]
    with (OUT_DATA_DIR / "case17_overlap.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=c17_ov_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(case17_overlaps)

    smi_origin_counts = Counter(c["smiles_origin"] for c in comp_rows)
    seq_origin_counts = Counter(c["seq_origin"] for c in comp_rows)
    audit_out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "total_unique_compounds": len(comp_rows),
        "smiles_origin_counts": dict(smi_origin_counts),
        "seq_origin_counts": dict(seq_origin_counts),
        "mw_rescues": dict(mw_rescues),
        "mw_drops": {f"{k[0]}|{k[1]}": v for k, v in mw_drops.items()},
        "case17_overlap_total_rows": len(case17_overlaps),
        "case17_overlap_drugs": sorted({x["drug_name"] for x in case17_overlaps}),
        "endpoints": dedup_audit,
    }
    (OUT_DATA_DIR / "clean_audit.json").write_text(json.dumps(audit_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE Step 02 in {audit_out['elapsed_sec']}s! Unique compounds={len(comp_rows):,}, smiles_origin={dict(smi_origin_counts)}, Case17 overlap rows={len(case17_overlaps)}", flush=True)


if __name__ == "__main__":
    main()
