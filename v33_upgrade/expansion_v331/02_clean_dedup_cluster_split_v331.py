#!/usr/bin/env python3
"""
Wave 5A — Step 02 (runs on node23, 64 CPU cores, RDKit 2026.09.1):
1. Loads staging raw_candidates_v331.tsv.gz (strictly READ-ONLY on existing V33 data/ & features/).
2. Parallel RDKit canonicalization, MW conversions (ug/mL -> uM), modification flags, backbone sequence extraction.
3. Deduplicates per (compound_key, endpoint, condition_signature) with replicate SD, conflict rejection,
   and conservative Case17 overlap flagging (`case17_excluded = 1`).
4. Matches compounds against existing 46,598 V33+Case17 compound registry (`0..46597`), reusing existing
   `compound_idx` & `cluster_id`.
5. Assigns incremental indices `46598..46598+K-1` ONLY to genuinely new compounds used in admitted endpoints,
   runs incremental Non-Chaining Dual-Track clustering (matching existing V33 leaders first at >=70% seq ID
   or >=0.70 Morgan Tanimoto, then leader-clustering new compounds), and applies strict admission gates.
6. Freezes untouched ~20% outer cluster holdout + balanced 5-fold grouped CV (`[10%, 35%]`), asserting
   ZERO leakage across cluster_id, compound_key, inchikey_14, conn_hash, canonical_smiles and 0 NaN/Inf.
7. Computes exact V33-identical compact features for ONLY the K new compounds and saves concatenated
   V33.1 feature bank under `/data/qlyu/v33_upgrade/expansion_v331/features/` (rows 0..46597 bit-for-bit identical!).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10_000_000)

V33_BASE = Path("/data/qlyu/v33_upgrade")
V33_DATA = V33_BASE / "data"
V33_FEAT = V33_BASE / "features"

EXP_BASE = V33_BASE / "expansion_v331"
STAGING_DIR = EXP_BASE / "staging"
OUT_DATA = EXP_BASE / "data"
OUT_FEAT = EXP_BASE / "features"
OUT_EP_DIR = OUT_DATA / "endpoints"

N_WORKERS = min(64, os.cpu_count() or 32)
TANIMOTO_THRESHOLD = 0.70
SEQ_IDENTITY_THRESHOLD = 0.70

AA20 = "ACDEFGHIKLMNPQRSTVWY"
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
EISENBERG_SCALE = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}
BOMAN_SCALE = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35,
    "W": -2.33, "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14,
    "T": 2.57, "S": 3.40, "H": 4.66, "Q": 5.54, "K": 5.55,
    "N": 6.64, "E": 6.81, "D": 8.72, "R": 14.92, "P": 0.0,
}
PKA_POS = {"K": 10.54, "R": 12.48, "H": 6.04}
PKA_NEG = {"D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46}

RDKIT_DESC_NAMES = [
    "MolWt", "Log10_MolWt", "ExactMolWt", "MolLogP", "MolMR",
    "TPSA", "TPSA_per_MW", "LabuteASA", "NumHDonors", "NumHAcceptors",
    "HBD_per_HBA", "NumRotatableBonds", "RotB_per_Heavy", "FractionCSP3",
    "HeavyAtomCount", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeteroatoms", "Hetero_per_Heavy", "NHOHCount",
    "NOCount", "FormalCharge", "AbsFormalCharge", "QED", "BertzCT",
    "Kappa1", "Kappa2", "HallKierAlpha", "Chi0v", "Chi1v",
    "NumAmideBonds", "ChiralCenterCount", "PosIonizableCount",
    "NegIonizableCount", "EstNetCharge_pH74", "LogP_per_TPSA",
]
MOD_DESC_NAMES = [
    "is_cyclic", "macrocycle_ring_size", "n_disulfide", "n_nmethyl",
    "n_d_aa_approx", "fatty_diacid_len", "fatty_monoacyl_len",
    "has_peg", "has_c_amidation", "has_n_acetylation",
]
SEQ_DESC_NAMES = (
    [f"aa_frac_{a}" for a in AA20]
    + [
        "seq_len", "log10_seq_len", "seq_net_charge_pH74", "seq_charge_density",
        "seq_hydrophobicity_kd", "seq_hydrophobicity_eisenberg",
        "seq_muH_alpha_100", "seq_muH_beta_160",
        "seq_aromatic_frac", "seq_basic_frac", "seq_acidic_frac",
        "seq_polar_frac", "seq_aliphatic_frac", "seq_boman_index", "seq_isoelectric_point",
    ]
)
ALL_PHYSCHEM_NAMES = RDKIT_DESC_NAMES + MOD_DESC_NAMES + SEQ_DESC_NAMES

# Physical finite domain bounds for all V33.1 candidate regression endpoints
DOMAIN_BOUNDS = {
    "efflux_ratio": (-2.0, 3.0),
    "cytotoxicity_CC50": (-4.0, 5.5),
    "hemolysis_MHC": (-3.0, 5.5),
    "Vd_absolute": (-3.0, 5.0),
    "LogP": (-6.0, 10.0),
    "MIC_B_subtilis": (-4.0, 5.5),
    "MIC_A_baumannii": (-4.0, 5.5),
    "MIC_K_pneumoniae": (-4.0, 5.5),
    "MIC_S_epidermidis": (-4.0, 5.5),
    "MIC_Salmonella": (-4.0, 5.5),
    "MIC_E_faecium": (-4.0, 5.5),
    "MIC_E_faecalis": (-4.0, 5.5),
    "MIC_L_monocytogenes": (-4.0, 5.5),
    "MIC_C_neoformans": (-4.0, 5.5),
    "MBC_E_coli": (-4.0, 5.5),
    "MBC_S_aureus": (-4.0, 5.5),
    "MBC_P_aeruginosa": (-4.0, 5.5),
    "MBC_A_baumannii": (-4.0, 5.5),
    "MBC_K_pneumoniae": (-4.0, 5.5),
    "MBC_B_subtilis": (-4.0, 5.5),
    "MBC_S_epidermidis": (-4.0, 5.5),
    "MBC_E_faecalis": (-4.0, 5.5),
    "MFC_C_albicans": (-4.0, 5.5),
    "MFC_antifungal": (-4.0, 5.5),
    "MBIC_biofilm": (-4.0, 5.5),
    "MBEC_biofilm": (-4.0, 5.5),
    "LC_antimicrobial": (-4.0, 5.5),
}

LOG10_ENDPOINTS = set(DOMAIN_BOUNDS.keys()) - {"LogP"}

ENDPOINT_COND_COLS = {
    "efflux_ratio": ["cond_assay"],
    "cytotoxicity_CC50": ["cond_matrix", "cond_species", "cond_estimand"],
    "hemolysis_MHC": ["cond_species", "cond_estimand"],
    "Vd_absolute": ["cond_species", "cond_route", "cond_estimand"],
    "LogP": ["cond_assay"],
    "MIC_B_subtilis": ["cond_strain", "cond_assay"],
    "MIC_A_baumannii": ["cond_strain", "cond_assay"],
    "MIC_K_pneumoniae": ["cond_strain", "cond_assay"],
    "MIC_S_epidermidis": ["cond_strain", "cond_assay"],
    "MIC_Salmonella": ["cond_strain", "cond_assay"],
    "MIC_E_faecium": ["cond_strain", "cond_assay"],
    "MIC_E_faecalis": ["cond_strain", "cond_assay"],
    "MIC_L_monocytogenes": ["cond_strain", "cond_assay"],
    "MIC_C_neoformans": ["cond_strain", "cond_assay"],
    "MBC_E_coli": ["cond_strain", "cond_assay"],
    "MBC_S_aureus": ["cond_strain", "cond_assay"],
    "MBC_P_aeruginosa": ["cond_strain", "cond_assay"],
    "MBC_A_baumannii": ["cond_strain", "cond_assay"],
    "MBC_K_pneumoniae": ["cond_strain", "cond_assay"],
    "MBC_B_subtilis": ["cond_strain", "cond_assay"],
    "MBC_S_epidermidis": ["cond_strain", "cond_assay"],
    "MBC_E_faecalis": ["cond_strain", "cond_assay"],
    "MFC_C_albicans": ["cond_assay"],
    "MFC_antifungal": ["cond_species", "cond_assay"],
    "MBIC_biofilm": ["cond_species", "cond_estimand", "cond_assay"],
    "MBEC_biofilm": ["cond_species", "cond_estimand", "cond_assay"],
    "LC_antimicrobial": ["cond_species", "cond_assay"],
}

# Admission gates: (min_rows, min_clusters, min_verified_compounds)
ADMISSION_GATES = {
    "efflux_ratio": (40, 20, 15),
    "cytotoxicity_CC50": (40, 20, 15),
    "hemolysis_MHC": (40, 20, 15),
    "Vd_absolute": (40, 20, 15),
    "LogP": (40, 25, 15),
    "LC_antimicrobial": (100, 50, 40),
}
DEFAULT_TIER2_GATE = (80, 40, 30)

# --- Worker globals (exact match to Wave-1 02 & 04 scripts!) ---
_SMARTS = {}
_SIDECHAIN_MAP = {}
_GENS = {}
_ION_SMARTS = {}


def _init_mol_worker():
    global _SMARTS, _SIDECHAIN_MAP
    _SMARTS = {
        "disulfide": Chem.MolFromSmarts("[#16X2]-[#16X2]"),
        "nmethyl": Chem.MolFromSmarts("[NX3;H0;!$([N]1CCC[C]1)]([CH3])([CX3]=O)[#6]"),
        "peg": Chem.MolFromSmarts("[#8X2][CX4H2][CX4H2][#8X2][CX4H2][CX4H2][#8X2,#7X3]"),
        "c_amid": Chem.MolFromSmarts("[NX3]-[CX4H1,CX4H2]-[CX3](=O)-[NH2]"),
        "n_acet": Chem.MolFromSmarts("[CH3]-[CX3](=O)-[NX3H1]-[CX4]-[CX3](=O)[#7,#8]"),
        "backbone_aa": Chem.MolFromSmarts("[NX3;H2,H1,H0]-[CX4;H1,H2,H0]-[CX3](=O)[#7,#8]"),
        "backbone_chiral": Chem.MolFromSmarts("[NX3;H2,H1,H0]-[CX4H1]-[CX3](=O)[#7,#8]"),
        "carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
        "acyl": Chem.MolFromSmarts("[CX3](=O)[NX3,OX2]"),
    }
    _SIDECHAIN_MAP = {}
    for aa in "ACDEFHIKLMNQRSTVWY":
        tri = Chem.MolFromSequence("G" + aa + "G")
        if tri is None:
            continue
        sc = _extract_middle_sidechain(tri)
        if sc:
            _SIDECHAIN_MAP[sc] = aa


def _extract_middle_sidechain(tri_mol) -> str:
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
                if atom.GetAtomicNum() == 16 and nb.GetAtomicNum() == 16:
                    continue
                queue.append(v)
    if not sc_atoms:
        return ""
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=sc_atoms, rootedAtAtom=start_idx, isomericSmiles=False, canonical=True)
    except Exception:
        return ""


def _compute_macrocycle_ring_size(mol) -> int:
    ri = mol.GetRingInfo()
    max_ring = max((len(r) for r in ri.AtomRings()), default=0)
    if max_ring >= 8:
        return max_ring
    ss_matches = mol.GetSubstructMatches(_SMARTS["disulfide"])
    for a1, a2 in ss_matches:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None or not bond.IsInRing():
            continue
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
    carboxyl_c = {m[0] for m in mol.GetSubstructMatches(_SMARTS["carboxyl"])}
    acyl_c = {m[0] for m in mol.GetSubstructMatches(_SMARTS["acyl"])} - carboxyl_c
    max_diacid = 0
    max_monoacyl = 0
    for c_idx in acyl_c:
        c_atom = mol.GetAtomWithIdx(c_idx)
        c_nbs = [
            nb.GetIdx() for nb in c_atom.GetNeighbors()
            if nb.GetAtomicNum() == 6 and not nb.GetIsAromatic() and not nb.IsInRing()
            and nb.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ]
        for start in c_nbs:
            best_mono = 0
            best_di = 0

            def dfs(u, prev, length, path_set):
                nonlocal best_mono, best_di
                if length > 30:
                    return
                atom = mol.GetAtomWithIdx(u)
                nbs = atom.GetNeighbors()
                if any(nb.GetAtomicNum() != 6 for nb in nbs):
                    return
                c_next = [nb.GetIdx() for nb in nbs if nb.GetIdx() != prev and nb.GetIdx() not in path_set]
                if not c_next:
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
        sc_nbs = [nb for nb in ca.GetNeighbors() if nb.GetIdx() not in (n_idx, c_idx)]
        if len(sc_nbs) != 1 or sc_nbs[0].GetAtomicNum() != 6:
            continue
        beta = sc_nbs[0]
        beta_has_s = any(nb.GetAtomicNum() >= 16 for nb in beta.GetNeighbors() if nb.GetIdx() != ca_idx)
        if beta_has_s:
            if cip == "S":
                d_count += 1
        else:
            if cip == "R":
                d_count += 1
    return d_count


def _extract_backbone_sequence(mol) -> str:
    matches = mol.GetSubstructMatches(_SMARTS["backbone_aa"])
    if not matches:
        return ""
    units = {}
    backbone_atoms = set()
    for m in matches:
        n_idx, ca_idx, c_idx = m[0], m[1], m[2]
        if ca_idx in units:
            continue
        if mol.GetAtomWithIdx(ca_idx).GetIsAromatic():
            continue
        units[ca_idx] = (n_idx, c_idx)
        backbone_atoms.update((n_idx, ca_idx, c_idx))
    if len(units) < 2:
        return ""
    res_letter = {}
    n_to_ca = {}
    for ca_idx, (n_idx, c_idx) in units.items():
        n_to_ca[n_idx] = ca_idx
        ca = mol.GetAtomWithIdx(ca_idx)
        sc_nbs = [nb for nb in ca.GetNeighbors() if nb.GetIdx() not in (n_idx, c_idx)]
        if len(sc_nbs) == 0:
            res_letter[ca_idx] = "G"
        elif len(sc_nbs) == 1:
            nb = sc_nbs[0]
            if ca.IsInRingSize(5) and mol.GetAtomWithIdx(n_idx).IsInRingSize(5) and nb.IsInRingSize(5):
                res_letter[ca_idx] = "P"
                continue
            sc_smi = _sidechain_smiles(mol, ca_idx, nb.GetIdx(), backbone_atoms)
            if sc_smi in _SIDECHAIN_MAP:
                res_letter[ca_idx] = _SIDECHAIN_MAP[sc_smi]
            elif sc_smi == "__LARGE__":
                res_letter[ca_idx] = "K" if (nb.GetAtomicNum() == 6 and nb.GetTotalNumHs() == 2) else "X"
            elif sc_smi in ("CS", "[CH2]S", "C[S]"):
                res_letter[ca_idx] = "C"
            else:
                res_letter[ca_idx] = "X"
        else:
            res_letter[ca_idx] = "X"
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
    visited = set()
    chains = []
    for start in [ca for ca, d in indeg.items() if d == 0] + list(units.keys()):
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
    best = max(chains, key=len)
    if len(best) < 2 or sum(1 for c in best if c != "X") / len(best) < 0.5:
        return ""
    return best


def _process_one_mol(item):
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
        ik = ""
        try:
            if mol.GetNumAtoms() <= 1020:
                ik = Chem.MolToInchiKey(mol) or ""
        except Exception:
            ik = ""
        conn_hash = hashlib.sha256(nosmi_conn.encode("utf-8")).hexdigest()[:16]
        ik14 = ik[:14] if len(ik) >= 14 else f"CONN:{conn_hash}"
        murcko_smi = generic_smi = ""
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
            "canon_smi": canon_smi, "smiles_origin": smiles_origin,
            "mw": round(mw, 4), "heavy_atoms": heavy,
            "inchikey": ik, "inchikey_14": ik14, "conn_hash": conn_hash,
            "murcko_smi": murcko_smi, "generic_smi": generic_smi, "scaf_heavy": scaf_heavy,
            "is_cyclic": is_cyclic, "macrocycle_ring_size": macro_size,
            "n_disulfide": n_disulfide, "n_nmethyl": n_nmethyl, "n_d_aa_approx": n_d_aa,
            "fatty_diacid_len": diacid_len, "fatty_monoacyl_len": monoacyl_len,
            "has_peg": has_peg, "has_c_amidation": has_c_amid, "has_n_acetylation": has_n_acet,
            "extracted_seq": extracted_seq,
        })
    except Exception:
        return (kind, key_str, None)


# --- Feature computation workers (100% identical to 04_compute_compact_features.py!) ---
def _init_feat_worker():
    global _GENS, _ION_SMARTS
    _GENS = {
        "morgan_cnt": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
        "morgan_bit": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
        "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=512, includeChirality=True),
    }
    _ION_SMARTS = {
        "pos_amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(NS=O);!$([n])][CX4]"),
        "pos_guanidine": Chem.MolFromSmarts("[NX3;H1,H2]C(=[NX2;H1,H0])[NX3;H1,H2]"),
        "neg_carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
        "neg_sulfate_phos": Chem.MolFromSmarts("[SX4,PX4](=O)(=O)[OX2H1,OX1-]"),
    }


def _safe_float(x, lo=-1e6, hi=1e6) -> float:
    try:
        v = float(x)
        return max(lo, min(hi, v)) if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _seq_charge_at_ph(seq: str, ph: float) -> float:
    q = 1.0 / (1.0 + 10.0 ** (ph - 8.0)) - 1.0 / (1.0 + 10.0 ** (3.1 - ph))
    for ch in seq:
        if ch in PKA_POS:
            q += 1.0 / (1.0 + 10.0 ** (ph - PKA_POS[ch]))
        elif ch in PKA_NEG:
            q -= 1.0 / (1.0 + 10.0 ** (PKA_NEG[ch] - ph))
    return q


def _compute_seq_descriptors(seq: str) -> tuple[bool, list[float]]:
    clean = [c for c in (seq or "").upper() if c in KD_SCALE]
    if len(clean) < 2:
        return False, [0.0] * len(SEQ_DESC_NAMES)
    L = len(clean)
    cnt = {a: 0 for a in AA20}
    for c in clean:
        cnt[c] += 1
    aa_fracs = [cnt[a] / L for a in AA20]
    s_clean = "".join(clean)
    net_q = _seq_charge_at_ph(s_clean, 7.4)
    q_dens = net_q / L
    kd_mean = sum(KD_SCALE[c] for c in clean) / L
    eis_vals = [EISENBERG_SCALE[c] for c in clean]
    eis_mean = sum(eis_vals) / L

    def mu_h(angle_deg: float) -> float:
        rad = math.radians(angle_deg)
        sx = sum(v * math.sin(i * rad) for i, v in enumerate(eis_vals))
        cx = sum(v * math.cos(i * rad) for i, v in enumerate(eis_vals))
        return math.sqrt(sx * sx + cx * cx) / L

    arom = (cnt["F"] + cnt["W"] + cnt["Y"]) / L
    basic = (cnt["K"] + cnt["R"] + cnt["H"]) / L
    acidic = (cnt["D"] + cnt["E"]) / L
    polar = (cnt["S"] + cnt["T"] + cnt["N"] + cnt["Q"] + cnt["C"]) / L
    aliph = (cnt["A"] + cnt["V"] + cnt["I"] + cnt["L"] + cnt["M"]) / L
    boman = -sum(BOMAN_SCALE[c] for c in clean) / L

    lo, hi = 1.0, 14.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _seq_charge_at_ph(s_clean, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    pi_val = 0.5 * (lo + hi)
    feats = aa_fracs + [
        float(L), math.log10(float(L)), net_q, q_dens,
        kd_mean, eis_mean, mu_h(100.0), mu_h(160.0),
        arom, basic, acidic, polar, aliph, boman, pi_val,
    ]
    return True, [_safe_float(x) for x in feats]


def _compute_one_feat(item):
    loc_i, smi, seq, mod_vals = item
    seq_valid, seq_feats = _compute_seq_descriptors(seq)
    if not smi:
        return (loc_i, False, seq_valid, np.array([0.0] * len(RDKIT_DESC_NAMES) + mod_vals + seq_feats, dtype=np.float32), None, None, None, None, None)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (loc_i, False, seq_valid, np.array([0.0] * len(RDKIT_DESC_NAMES) + mod_vals + seq_feats, dtype=np.float32), None, None, None, None, None)
    mw = _safe_float(Descriptors.MolWt(mol), 1.0, 1e6)
    exact_mw = _safe_float(Descriptors.ExactMolWt(mol), 1.0, 1e6)
    logp = _safe_float(Crippen.MolLogP(mol), -50.0, 50.0)
    mr = _safe_float(Crippen.MolMR(mol), 0.0, 1e5)
    tpsa = _safe_float(rdMolDescriptors.CalcTPSA(mol), 0.0, 1e5)
    labute = _safe_float(rdMolDescriptors.CalcLabuteASA(mol), 0.0, 1e5)
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotb = float(Lipinski.NumRotatableBonds(mol))
    fcsp3 = _safe_float(rdMolDescriptors.CalcFractionCSP3(mol), 0.0, 1.0)
    heavy = float(max(1, mol.GetNumHeavyAtoms()))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    arom_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    aliph_rings = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    sat_rings = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
    hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    nhoh = float(Lipinski.NHOHCount(mol))
    no_cnt = float(Lipinski.NOCount(mol))
    fchg = float(Chem.GetFormalCharge(mol))
    try:
        qed_val = _safe_float(QED.qed(mol), 0.0, 1.0)
    except Exception:
        qed_val = 0.0
    try:
        bertz = _safe_float(Descriptors.BertzCT(mol), 0.0, 1e6) if heavy <= 350 else _safe_float(heavy * math.log2(max(2.0, heavy)) * 10.0, 0.0, 1e6)
    except Exception:
        bertz = 0.0
    k1 = _safe_float(rdMolDescriptors.CalcKappa1(mol), 0.0, 1e5)
    k2 = _safe_float(rdMolDescriptors.CalcKappa2(mol), 0.0, 1e5)
    hka = _safe_float(rdMolDescriptors.CalcHallKierAlpha(mol), -1e4, 1e4)
    chi0v = _safe_float(rdMolDescriptors.CalcChi0v(mol), 0.0, 1e5)
    chi1v = _safe_float(rdMolDescriptors.CalcChi1v(mol), 0.0, 1e5)
    amide = float(rdMolDescriptors.CalcNumAmideBonds(mol))
    chiral = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=True)))
    pos_ion = float(len(mol.GetSubstructMatches(_ION_SMARTS["pos_amine"])) + len(mol.GetSubstructMatches(_ION_SMARTS["pos_guanidine"])))
    neg_ion = float(len(mol.GetSubstructMatches(_ION_SMARTS["neg_carboxyl"])) + len(mol.GetSubstructMatches(_ION_SMARTS["neg_sulfate_phos"])))
    est_q74 = pos_ion - neg_ion
    rd_feats = [
        mw, math.log10(max(1.0, mw)), exact_mw, logp, mr,
        tpsa, tpsa / max(1.0, mw), labute, hbd, hba,
        hbd / max(1.0, hba), rotb, rotb / heavy, fcsp3,
        heavy, rings, arom_rings, aliph_rings,
        sat_rings, hetero, hetero / heavy, nhoh,
        no_cnt, fchg, abs(fchg), qed_val, bertz,
        k1, k2, hka, chi0v, chi1v,
        amide, chiral, pos_ion, neg_ion, est_q74, logp / max(1.0, tpsa),
    ]
    physchem = np.array([_safe_float(x) for x in rd_feats] + mod_vals + seq_feats, dtype=np.float32)
    m_cnt = np.clip(_GENS["morgan_cnt"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    m_bit = _GENS["morgan_bit"].GetFingerprintAsNumPy(mol).astype(np.uint8)
    ap_cnt = np.clip(_GENS["atompair"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    tt_cnt = np.clip(_GENS["torsion"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint8)
    return (loc_i, True, seq_valid, physchem, m_cnt, ap_cnt, maccs, tt_cnt, m_bit)


# --- Clustering & Split helpers (exact match to 03_cluster_and_split.py!) ---
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def levenshtein_bounded(s1: str, s2: str, max_dist: int) -> int:
    n, m = len(s1), len(s2)
    if abs(n - m) > max_dist:
        return max_dist + 1
    if s1 == s2:
        return 0
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        c1 = s1[i - 1]
        lo = max(1, i - max_dist)
        hi = min(m, i + max_dist)
        curr = [max_dist + 1] * (m + 1)
        if lo == 1:
            curr[0] = i
        row_min = curr[0]
        for j in range(lo, hi + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            v = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr[j] = v
            if v < row_min:
                row_min = v
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[m]


def incremental_cluster_sequences(
    existing_seq_to_cid: dict[str, str],
    new_seqs: list[str],
    thresh: float = SEQ_IDENTITY_THRESHOLD,
) -> dict[str, str]:
    """Match new sequences against existing V33 Track-S sequences first (preventing analog leakage!),
    then leader-cluster remaining unmatched new sequences."""
    out_map: dict[str, str] = {}
    # Build k-mer index over existing sequences
    reps: list[tuple[str, str]] = []
    kmer2: dict[str, list[int]] = defaultdict(list)
    kmer3: dict[str, list[int]] = defaultdict(list)
    exact_short: dict[str, str] = {}

    for seq, cid in sorted(existing_seq_to_cid.items(), key=lambda kv: (-len(kv[0]), kv[0])):
        L = len(seq)
        if L < 4:
            exact_short[seq] = cid
            continue
        rid = len(reps)
        reps.append((seq, cid))
        if L < 10:
            for km in {seq[i : i + 2] for i in range(L - 1)}:
                kmer2[km].append(rid)
        else:
            for km in {seq[i : i + 3] for i in range(L - 2)}:
                kmer3[km].append(rid)
            if L <= 13:
                for km in {seq[i : i + 2] for i in range(L - 1)}:
                    kmer2[km].append(rid)

    next_new_id = 0
    for seq in sorted(set(new_seqs), key=lambda s: (-len(s), s)):
        if seq in existing_seq_to_cid:
            out_map[seq] = existing_seq_to_cid[seq]
            continue
        L = len(seq)
        if L < 4:
            if seq in exact_short:
                out_map[seq] = exact_short[seq]
            else:
                cid = f"CLU_V331_S_{next_new_id:06d}"
                next_new_id += 1
                exact_short[seq] = cid
                out_map[seq] = cid
            continue

        min_len_rep = int(math.ceil(L * thresh))
        max_len_rep = int(math.floor(L / thresh))
        cand_counts: Counter[int] = Counter()
        if L < 10:
            for km in {seq[i : i + 2] for i in range(L - 1)}:
                for rid in kmer2.get(km, ()):
                    cand_counts[rid] += 1
            min_hits = 1
        else:
            for km in {seq[i : i + 3] for i in range(L - 2)}:
                for rid in kmer3.get(km, ()):
                    cand_counts[rid] += 1
            min_hits = 1 if L < 15 else 2

        best_cid = None
        best_sim = thresh - 1e-6
        for rid, hits in cand_counts.most_common(96):
            if hits < min_hits:
                break
            rseq, rcid = reps[rid]
            rL = len(rseq)
            if rL < min_len_rep or rL > max_len_rep:
                continue
            denom = max(L, rL)
            max_ed = int(math.floor((1.0 - thresh) * denom + 1e-9))
            ed = levenshtein_bounded(seq, rseq, max_ed)
            if ed <= max_ed:
                sim = 1.0 - (ed / denom)
                if sim >= thresh and sim > best_sim:
                    best_sim = sim
                    best_cid = rcid
                    if sim >= 0.95:
                        break
        if best_cid is not None:
            out_map[seq] = best_cid
        else:
            rid = len(reps)
            cid = f"CLU_V331_S_{next_new_id:06d}"
            next_new_id += 1
            reps.append((seq, cid))
            out_map[seq] = cid
            if L < 10:
                for km in {seq[i : i + 2] for i in range(L - 1)}:
                    kmer2[km].append(rid)
            else:
                for km in {seq[i : i + 3] for i in range(L - 2)}:
                    kmer3[km].append(rid)
                if L <= 13:
                    for km in {seq[i : i + 2] for i in range(L - 1)}:
                        kmer2[km].append(rid)
    return out_map


def incremental_cluster_morgan(
    existing_bits: np.ndarray,
    existing_cids: list[str],
    new_bits: np.ndarray,
    thresh: float = TANIMOTO_THRESHOLD,
    block_size: int = 1024,
) -> list[str]:
    """Match new Track-M compounds against ALL existing V33 Track-M compounds at Tanimoto >= 0.70 first,
    then Taylor-Butina cluster unmatched new Track-M compounds."""
    n_new = new_bits.shape[0]
    if n_new == 0:
        return []
    assigned: list[Optional[str]] = [None] * n_new

    if existing_bits.shape[0] > 0:
        E = existing_bits.astype(np.float32, copy=False)
        Pe = E.sum(axis=1)
        Q = new_bits.astype(np.float32, copy=False)
        Pq = Q.sum(axis=1)
        for i0 in range(0, n_new, block_size):
            i1 = min(n_new, i0 + block_size)
            dot = Q[i0:i1] @ E.T
            denom = Pq[i0:i1, None] + Pe[None, :] - dot
            sim = np.where(denom > 0.0, dot / denom, 0.0)
            best_j = np.argmax(sim, axis=1)
            best_s = sim[np.arange(i1 - i0), best_j]
            for loc_i in range(i1 - i0):
                if float(best_s[loc_i]) >= thresh:
                    assigned[i0 + loc_i] = existing_cids[int(best_j[loc_i])]

    unmatched_pos = [i for i, a in enumerate(assigned) if a is None]
    if unmatched_pos:
        B = new_bits[unmatched_pos].astype(np.float32, copy=False)
        m = B.shape[0]
        P = B.sum(axis=1)
        nbrs: list[list[int]] = [[] for _ in range(m)]
        for i0 in range(0, m, block_size):
            i1 = min(m, i0 + block_size)
            dot = B[i0:i1] @ B.T
            denom = P[i0:i1, None] + P[None, :] - dot
            sim = np.where(denom > 0.0, dot / denom, 0.0)
            rs, cs = np.where(sim >= thresh)
            for r_loc, c_loc in zip(rs, cs):
                nbrs[i0 + int(r_loc)].append(int(c_loc))
        order = sorted(range(m), key=lambda u: (-len(nbrs[u]), u))
        loc_asg = [-1] * m
        next_cid = 0
        for u in order:
            if loc_asg[u] != -1:
                continue
            cid = next_cid
            next_cid += 1
            for v in nbrs[u]:
                if loc_asg[v] == -1:
                    loc_asg[v] = cid
        for loc_i, orig_i in enumerate(unmatched_pos):
            assigned[orig_i] = f"CLU_V331_M_{loc_asg[loc_i]:06d}"

    return [str(x) for x in assigned]


def _det_tie(ep_name: str, cid: str) -> int:
    return int(hashlib.blake2b(f"{ep_name}|{cid}".encode("utf-8"), digest_size=8).hexdigest(), 16)


def _bin_pack_clusters(
    cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    cluster_strata: dict[str, np.ndarray],
    n_bins: int,
    ep_name: str,
    max_bin_cap: int | None = None,
) -> dict[str, int]:
    n_strata = len(next(iter(cluster_strata.values())))
    total_rows = sum(cluster_sizes[c] for c in cluster_ids)
    total_strata = np.sum([cluster_strata[c] for c in cluster_ids], axis=0).astype(float)
    target_row = max(1.0, total_rows / n_bins)
    target_strata = np.maximum(1.0, total_strata / n_bins)
    bin_rows = np.zeros(n_bins, dtype=float)
    bin_strata = np.zeros((n_bins, n_strata), dtype=float)
    assignment: dict[str, int] = {}
    ordered = sorted(cluster_ids, key=lambda c: (-cluster_sizes[c], _det_tie(ep_name, c)))

    for cid in ordered:
        s = float(cluster_sizes[cid])
        q = cluster_strata[cid].astype(float)
        eligible = list(range(n_bins))
        if max_bin_cap is not None and len(eligible) > 1:
            under_cap = [b for b in eligible if bin_rows[b] + s <= max_bin_cap]
            if under_cap:
                eligible = under_cap
        best_b = eligible[0]
        best_cost = float("inf")
        for b in eligible:
            dv_row = ((bin_rows[b] + s) ** 2 - bin_rows[b] ** 2) / (target_row ** 2)
            dv_str = float(np.sum(((bin_strata[b] + q) ** 2 - bin_strata[b] ** 2) / (target_strata ** 2)))
            over_pen = 5.0 * max(0.0, (bin_rows[b] + s - target_row) / target_row) ** 2 if bin_rows[b] >= target_row else 0.0
            cost = 4.0 * dv_row + 1.0 * dv_str + over_pen
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_b = b
        assignment[cid] = best_b
        bin_rows[best_b] += s
        bin_strata[best_b] += q

    def total_obj(br: np.ndarray, bs: np.ndarray) -> float:
        return 4.0 * float(np.sum(((br - target_row) / target_row) ** 2)) + float(np.sum(((bs - target_strata[None, :]) / target_strata[None, :]) ** 2))

    cur_obj = total_obj(bin_rows, bin_strata)
    for _ in range(30):
        improved = False
        for cid in ordered:
            s = float(cluster_sizes[cid])
            q = cluster_strata[cid].astype(float)
            b_from = assignment[cid]
            if bin_rows[b_from] - s <= 0:
                continue
            best_to = b_from
            best_obj = cur_obj
            for b_to in range(n_bins):
                if b_to == b_from:
                    continue
                if max_bin_cap is not None and bin_rows[b_to] + s > max_bin_cap and bin_rows[b_from] <= max_bin_cap:
                    continue
                br_new = bin_rows.copy()
                bs_new = bin_strata.copy()
                br_new[b_from] -= s; br_new[b_to] += s
                bs_new[b_from] -= q; bs_new[b_to] += q
                cand_obj = total_obj(br_new, bs_new)
                if cand_obj < best_obj - 1e-9:
                    best_obj = cand_obj
                    best_to = b_to
            if best_to != b_from:
                bin_rows[b_from] -= s; bin_rows[best_to] += s
                bin_strata[b_from] -= q; bin_strata[best_to] += q
                assignment[cid] = best_to
                cur_obj = best_obj
                improved = True
        if not improved:
            break
    return assignment


def assign_splits_regression(
    rows: list[dict],
    comp_cluster: dict[str, str],
    comp_map: dict[str, dict],
    ep_name: str,
) -> tuple[list[dict], dict]:
    n = len(rows)
    groups = np.array([comp_cluster[r["compound_key"]] for r in rows])
    uniq_groups = sorted(set(groups.tolist()))
    n_groups = len(uniq_groups)
    assert n_groups >= 10, f"{ep_name}: too few clusters ({n_groups})"

    yv = np.array([float(r["y_value"]) for r in rows], dtype=float)
    n_bins_q = 5 if n_groups >= 40 else (3 if n_groups >= 18 else 2)
    qs = np.unique(np.quantile(yv, np.linspace(0, 1, n_bins_q + 1)[1:-1]))
    y_strat = np.digitize(yv, qs, right=False).astype(int)
    n_strata = len(qs) + 1

    cluster_sizes: dict[str, int] = Counter()
    cluster_strata: dict[str, np.ndarray] = {c: np.zeros(n_strata, dtype=int) for c in uniq_groups}
    for i, c in enumerate(groups):
        cluster_sizes[c] += 1
        cluster_strata[c][y_strat[i]] += 1

    outer_asg = _bin_pack_clusters(uniq_groups, cluster_sizes, cluster_strata, n_bins=5, ep_name=f"{ep_name}|outer")
    best_ho_bin = None
    best_ho_score = float("inf")
    for b in range(5):
        ho_c = {c for c, ab in outer_asg.items() if ab == b}
        cv_c = set(uniq_groups) - ho_c
        ho_n = sum(cluster_sizes[c] for c in ho_c)
        cv_n = n - ho_n
        if ho_n == 0 or cv_n == 0 or len(cv_c) < 5:
            continue
        # Ensure holdout has >= 2 exact rows!
        ho_exact = sum(1 for i, c in enumerate(groups) if c in ho_c and rows[i]["censor_type"] == "exact")
        if ho_exact < 2:
            continue
        max_cv_clu = max(cluster_sizes[c] for c in cv_c)
        cv_cap = max(10, int(0.35 * cv_n))
        cap_viol = max(0, max_cv_clu - cv_cap)
        score = 100.0 * cap_viol + 2.0 * abs((ho_n / n) - 0.20)
        if score < best_ho_score:
            best_ho_score = score
            best_ho_bin = b

    assert best_ho_bin is not None, f"Failed to select valid outer holdout for {ep_name}!"
    ho_clusters = {c for c, ab in outer_asg.items() if ab == best_ho_bin}
    cv_clusters = sorted(set(uniq_groups) - ho_clusters)

    n_cv_rows = sum(cluster_sizes[c] for c in cv_clusters)
    cv_min_req = max(3, int(0.10 * n_cv_rows))
    cv_max_req = max(10, int(0.35 * n_cv_rows))
    cv_asg = _bin_pack_clusters(cv_clusters, cluster_sizes, cluster_strata, n_bins=5, ep_name=f"{ep_name}|inner", max_bin_cap=cv_max_req)

    is_holdout = np.zeros(n, dtype=int)
    fold_id = np.full(n, -1, dtype=int)
    for i, c in enumerate(groups):
        if c in ho_clusters:
            is_holdout[i] = 1
            fold_id[i] = -1
        else:
            is_holdout[i] = 0
            fold_id[i] = cv_asg[c]

    # HARD ZERO-LEAKAGE & FOLD-BALANCE ASSERTIONS
    assert len(ho_clusters & set(cv_clusters)) == 0, f"CLUSTER LEAKAGE holdout vs CV in {ep_name}!"

    def _id_tokens(mask: np.ndarray):
        cks, ik14s, chs, smis = set(), set(), set(), set()
        for idx in np.where(mask)[0]:
            ck = rows[idx]["compound_key"]
            cks.add(ck)
            cm = comp_map[ck]
            if cm["inchikey_14"]:
                ik14s.add(cm["inchikey_14"])
            if cm["conn_hash"]:
                chs.add(cm["conn_hash"])
            if cm["canonical_smiles"]:
                smis.add(cm["canonical_smiles"])
        return cks, ik14s, chs, smis

    ho_mask = is_holdout == 1
    cv_mask = is_holdout == 0
    ho_t = _id_tokens(ho_mask)
    cv_t = _id_tokens(cv_mask)
    for label, s1, s2 in zip(("compound_key", "inchikey_14", "conn_hash", "canonical_smiles"), ho_t, cv_t):
        assert len(s1 & s2) == 0, f"{label} LEAKAGE holdout vs CV in {ep_name}!"

    fold_cluster_sets = []
    fold_id_sets = []
    fold_sizes = {}
    for fi in range(5):
        mask = (is_holdout == 0) & (fold_id == fi)
        cnt = int(mask.sum())
        assert cnt > 0, f"Empty CV fold {fi} in {ep_name}!"
        fc = set(groups[mask])
        ftoks = _id_tokens(mask)
        for pi, (prev_c, prev_t) in enumerate(zip(fold_cluster_sets, fold_id_sets)):
            assert len(fc & prev_c) == 0, f"CLUSTER LEAKAGE fold {fi} vs {pi} in {ep_name}!"
            for label, s1, s2 in zip(("compound_key", "inchikey_14", "conn_hash", "canonical_smiles"), ftoks, prev_t):
                assert len(s1 & s2) == 0, f"{label} LEAKAGE fold {fi} vs {pi} in {ep_name}!"
        fold_cluster_sets.append(fc)
        fold_id_sets.append(ftoks)
        fold_sizes[f"fold_{fi}"] = cnt

    min_f = min(fold_sizes.values())
    max_f = max(fold_sizes.values())
    assert min_f >= cv_min_req, f"{ep_name}: min_f={min_f} < {cv_min_req} ({fold_sizes})"
    assert max_f <= cv_max_req, f"{ep_name}: max_f={max_f} > {cv_max_req} ({fold_sizes})"

    out_rows = []
    for i, r in enumerate(rows):
        nr = dict(r)
        nr["cluster_id"] = str(groups[i])
        nr["is_holdout"] = int(is_holdout[i])
        nr["fold_id"] = int(fold_id[i])
        out_rows.append(nr)

    info = {
        "n_rows": n,
        "n_unique_compounds": len({r["compound_key"] for r in rows}),
        "n_clusters": int(n_groups),
        "n_cv_rows": int(n_cv_rows),
        "n_holdout_rows": int(ho_mask.sum()),
        "holdout_frac": round(float(ho_mask.sum() / n), 4),
        "n_censored": sum(1 for r in rows if r["censor_type"] != "exact"),
        "n_exact": sum(1 for r in rows if r["censor_type"] == "exact"),
        "n_case17_excluded": sum(1 for r in rows if int(r["case17_excluded"]) == 1),
        "fold_sizes": fold_sizes,
        "min_fold_frac": round(float(min_f / n_cv_rows), 4),
        "max_fold_frac": round(float(max_f / n_cv_rows), 4),
        "fold_balance_pass": True,
        "zero_leakage_pass": True,
    }
    return out_rows, info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    OUT_EP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FEAT.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen V33 compounds, cluster IDs, Case17 queries, and feature index (STRICTLY READ-ONLY)
    print(f"[1/7] Loading frozen V33 registry, cluster map, & feature bank (workers={N_WORKERS})...", flush=True)
    v33_comps: dict[str, dict] = {}
    with (V33_DATA / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v33_comps[r["compound_key"]] = r

    # Recover exact frozen V33 cluster_id from existing 24 endpoint TSVs (authoritative post-03 clustering!)
    v33_cluster_map: dict[str, str] = {}
    for ep_fp in sorted((V33_DATA / "endpoints").glob("*.tsv")):
        with ep_fp.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ck = r["compound_key"]
                cid = r["cluster_id"]
                if ck in v33_cluster_map:
                    assert v33_cluster_map[ck] == cid, f"Inconsistent V33 cluster_id for {ck}"
                else:
                    v33_cluster_map[ck] = cid
    for ck, r in v33_comps.items():
        if ck not in v33_cluster_map and r.get("cluster_id"):
            v33_cluster_map[ck] = r["cluster_id"]
    print(f"  Loaded {len(v33_comps):,} V33 compounds | {len(v33_cluster_map):,} frozen V33 cluster assignments.", flush=True)

    # Load Case17 overlap lookup tables
    c17_by_smi, c17_by_ik, c17_by_ik14, c17_by_conn, c17_by_seq = {}, {}, {}, {}, {}
    c17_rows = []
    with (V33_DATA / "case17_queries.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c17_rows.append(r)
            dn = r["drug_name"]
            if r["canonical_smiles"]:
                c17_by_smi[r["canonical_smiles"]] = dn
            if r["inchikey"]:
                c17_by_ik[r["inchikey"]] = dn
            if r["inchikey_14"]:
                c17_by_ik14[r["inchikey_14"]] = dn
            if r["conn_hash"]:
                c17_by_conn[r["conn_hash"]] = dn
            pseq = r["parent_sequence"]
            if pseq and len(pseq) >= 6 and "X" not in pseq:
                c17_by_seq[pseq] = dn
            # Also register Case17 compound metadata if not in compounds_v33 (Tirzepatide!)
            if r["compound_key"] not in v33_comps:
                v33_comps[r["compound_key"]] = {
                    "compound_key": r["compound_key"],
                    "canonical_smiles": r["canonical_smiles"],
                    "smiles_origin": r["smiles_origin"],
                    "inchikey": r["inchikey"],
                    "inchikey_14": r["inchikey_14"],
                    "conn_hash": r["conn_hash"],
                    "mol_wt": r["mol_wt"],
                    "heavy_atoms": "0",
                    "murcko_smi": "",
                    "generic_smi": "",
                    "scaf_heavy": "0",
                    "parent_sequence": r["parent_sequence"],
                    "seq_origin": r["seq_origin"],
                    **{k: r[k] for k in MOD_DESC_NAMES},
                }

    # Load existing V33 feature arrays & compound_index.tsv (46,598 rows)
    v33_idx_rows = []
    v33_key_to_idx: dict[str, int] = {}
    with (V33_FEAT / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v33_idx_rows.append(r)
            v33_key_to_idx[r["compound_key"]] = int(r["compound_idx"])
    N_V33 = len(v33_idx_rows)
    assert N_V33 == 46598, f"Expected 46598 V33 feature rows, got {N_V33}"
    v33_feat_manifest = json.loads((V33_FEAT / "feature_manifest.json").read_text(encoding="utf-8"))

    # 2. Load raw_candidates_v331.tsv.gz & run parallel RDKit canonicalization
    print("[2/7] Loading raw_candidates_v331.tsv.gz & canonicalizing structures...", flush=True)
    raw_rows = []
    unique_smis = set()
    unique_pure_seqs = set()
    with gzip.open(STAGING_DIR / "raw_candidates_v331.tsv.gz", "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw_rows.append(row)
            if row["canonical_smiles_src"]:
                unique_smis.add(row["canonical_smiles_src"])
            elif row["is_pure_unmodified_20aa"] == "1" and row["sequence_raw"]:
                unique_pure_seqs.add(row["sequence_raw"])
    print(f"  Loaded {len(raw_rows):,} candidate rows | {len(unique_smis):,} unique SMILES | {len(unique_pure_seqs):,} pure 20AA seqs", flush=True)

    tasks = [("smi", s) for s in sorted(unique_smis)] + [("seq", q) for q in sorted(unique_pure_seqs)]
    smi_info, seq_info = {}, {}
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_mol_worker) as ex:
        for kind, key_str, res in ex.map(_process_one_mol, tasks, chunksize=128):
            if res is not None:
                if kind == "smi":
                    smi_info[key_str] = res
                else:
                    seq_info[key_str] = res
    print(f"  RDKit done in {time.time()-t0:.1f}s | smi_ok={len(smi_info):,} | seq_to_smi_ok={len(seq_info):,}", flush=True)

    # 3. Resolve MW conversions, unify compound registry, deduplicate per (compound_key, endpoint, cond_sig)
    print("[3/7] Resolving MW conversions & deduplicating per (compound_key, endpoint, condition_signature)...", flush=True)
    all_comps: dict[str, dict] = dict(v33_comps)
    grouped = defaultdict(lambda: defaultdict(list))
    mw_rescues = Counter()
    mw_drops = Counter()

    def register_comp(info, seq_raw: str, seq_kind: str, is_pure20: str) -> str:
        if info is not None:
            ckey = "SMI:" + hashlib.sha256(info["canon_smi"].encode("utf-8")).hexdigest()[:16]
            if ckey not in all_comps:
                if seq_raw:
                    pseq = seq_raw
                    sorig = "source_bare_20aa" if is_pure20 == "1" else "source_parent_projection"
                elif info["extracted_seq"]:
                    pseq = info["extracted_seq"]
                    sorig = "rdkit_backbone_extracted"
                else:
                    pseq = ""
                    sorig = "none"
                all_comps[ckey] = {
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
            return ckey
        else:
            ckey = "SEQMOD:" + hashlib.sha256(f"{seq_raw}|{seq_kind}".encode("utf-8")).hexdigest()[:16]
            if ckey not in all_comps:
                all_comps[ckey] = {
                    "compound_key": ckey,
                    "canonical_smiles": "",
                    "smiles_origin": "missing",
                    "inchikey": "", "inchikey_14": "", "conn_hash": "",
                    "mol_wt": 0.0, "heavy_atoms": 0,
                    "murcko_smi": "", "generic_smi": "", "scaf_heavy": 0,
                    "parent_sequence": seq_raw, "seq_origin": "source_parent_projection",
                    **{k: 0 for k in MOD_DESC_NAMES},
                }
            return ckey

    def reg_bounds(ep, y_val, rel, raw_lo="", raw_hi=""):
        dmin, dmax = DOMAIN_BOUNDS[ep]
        if rel == "interval":
            lo, hi = float(raw_lo), float(raw_hi)
            if lo > hi:
                lo, hi = hi, lo
            return (0.5 * (lo + hi), lo, hi, "interval")
        if rel in (">", ">="):
            return (y_val, y_val, max(y_val + 0.01, dmax), "right")
        if rel in ("<", "<="):
            return (y_val, min(y_val - 0.01, dmin), y_val, "left")
        return (y_val, y_val, y_val, "exact")

    for row in raw_rows:
        ep = row["endpoint_v331"]
        if ep not in ENDPOINT_COND_COLS:
            continue
        smi_s = row["canonical_smiles_src"]
        seq_raw = row["sequence_raw"]
        is_pure20 = row["is_pure_unmodified_20aa"]

        info = smi_info.get(smi_s) if smi_s else None
        if info is None and is_pure20 == "1" and seq_raw:
            info = seq_info.get(seq_raw)
        if info is None and not seq_raw:
            mw_drops[(ep, "unparseable_smi_no_seq")] += 1
            continue

        ckey = register_comp(info, seq_raw, row["sequence_kind"], is_pure20)
        unit = row["val_unit"]
        rel = row["val_relation"]
        pri = int(row["evidence_priority"])

        if unit == "ug/mL_needs_mw":
            if info is None or info["mw"] <= 0:
                mw_drops[(ep, "missing_mw")] += 1
                continue
            val_uM = float(row["val_num"]) * 1000.0 / info["mw"]
            lo_lim = 1e-3 if ep == "hemolysis_MHC" else 1e-4
            if not (lo_lim <= val_uM <= 1e5):
                mw_drops[(ep, "uM_out_of_range")] += 1
                continue
            y = math.log10(val_uM)
            yv, yl, yh, ct = reg_bounds(ep, y, rel)
            mw_rescues[ep] += 1
        else:
            raw_v = float(row["val_num"])
            if ep in LOG10_ENDPOINTS:
                if raw_v <= 0:
                    continue
                y = math.log10(raw_v)
                yv, yl, yh, ct = reg_bounds(ep, y, rel, row["val_lower"], row["val_upper"])
            else:
                yv, yl, yh, ct = reg_bounds(ep, raw_v, rel, row["val_lower"], row["val_upper"])

        cond_cols = ENDPOINT_COND_COLS[ep]
        cond_vals = tuple(row[c] for c in cond_cols)
        grouped[ep][(ckey, cond_vals)].append({
            "task_tier": row["task_tier"],
            "task_type": "regression",
            "record_id": row["record_id"],
            "origin": row["origin"],
            "source_code": row["source_code"],
            "source_record_id": row["source_record_id"],
            "rescue_reason": row["rescue_reason"],
            "priority": pri,
            "y_value": float(yv),
            "y_lower": float(yl),
            "y_upper": float(yh),
            "censor_type": ct,
        })

    print("  MW rescues:", dict(mw_rescues), "| MW drops:", dict(mw_drops), flush=True)

    def match_case17(ckey: str) -> tuple[str, str]:
        c = all_comps[ckey]
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

    dedup_by_ep: dict[str, list[dict]] = {}
    dedup_conflicts: dict[str, int] = {}
    for ep in sorted(ENDPOINT_COND_COLS.keys()):
        groups = grouped.get(ep, {})
        cond_cols = ENDPOINT_COND_COLS[ep]
        out_rows = []
        n_conf = 0
        dmin, dmax = DOMAIN_BOUNDS[ep]
        for (ckey, cond_vals), obs_list in sorted(groups.items()):
            obs_sorted = sorted(obs_list, key=lambda x: (x["priority"], x["record_id"]))
            tier = obs_sorted[0]["task_tier"]
            exacts = [o for o in obs_sorted if o["censor_type"] == "exact"]
            intervals = [o for o in obs_sorted if o["censor_type"] == "interval"]
            rights = [o for o in obs_sorted if o["censor_type"] == "right"]
            lefts = [o for o in obs_sorted if o["censor_type"] == "left"]

            if exacts:
                vals = [o["y_value"] for o in exacts]
                if len(vals) >= 2 and (max(vals) - min(vals)) > 2.5:
                    p1_vals = [o["y_value"] for o in exacts if o["priority"] == 1]
                    if p1_vals and (max(p1_vals) - min(p1_vals)) <= 2.5:
                        vals = p1_vals
                        exacts = [o for o in exacts if o["priority"] == 1]
                    else:
                        n_conf += 1
                        continue
                y_final = float(statistics.median(vals))
                y_lo = y_hi = y_final
                ctype = "exact"
                rep = exacts[0]
                n_rep = len(exacts)
                rep_sd = round(statistics.pstdev(vals), 5) if len(vals) > 1 else 0.0
                all_src = ";".join(sorted({o["source_code"] for o in exacts}))
                all_rr = ";".join(sorted({o["rescue_reason"] for o in exacts}))
            elif intervals:
                lo = max(o["y_lower"] for o in intervals)
                hi = min(o["y_upper"] for o in intervals)
                if lo > hi:
                    lo = float(statistics.median([o["y_lower"] for o in intervals]))
                    hi = float(statistics.median([o["y_upper"] for o in intervals]))
                y_final, y_lo, y_hi, ctype = 0.5 * (lo + hi), lo, hi, "interval"
                rep, n_rep, rep_sd = intervals[0], len(intervals), 0.0
                all_src = ";".join(sorted({o["source_code"] for o in intervals}))
                all_rr = ";".join(sorted({o["rescue_reason"] for o in intervals}))
            else:
                if rights and lefts:
                    lo = max(o["y_lower"] for o in rights)
                    hi = min(o["y_upper"] for o in lefts)
                    if lo <= hi:
                        y_final, y_lo, y_hi, ctype = 0.5 * (lo + hi), lo, hi, "interval"
                        rep, n_rep, rep_sd = rights[0], len(rights) + len(lefts), 0.0
                        all_src = ";".join(sorted({o["source_code"] for o in obs_list}))
                        all_rr = ";".join(sorted({o["rescue_reason"] for o in obs_list}))
                    else:
                        n_conf += 1
                        continue
                elif rights:
                    lo = max(o["y_lower"] for o in rights)
                    y_final, y_lo, y_hi, ctype = lo, lo, max(lo + 0.01, dmax), "right"
                    rep = max(rights, key=lambda o: o["y_lower"])
                    n_rep, rep_sd = len(rights), 0.0
                    all_src = ";".join(sorted({o["source_code"] for o in rights}))
                    all_rr = ";".join(sorted({o["rescue_reason"] for o in rights}))
                else:
                    hi = min(o["y_upper"] for o in lefts)
                    y_final, y_lo, y_hi, ctype = hi, min(hi - 0.01, dmin), hi, "left"
                    rep = min(lefts, key=lambda o: o["y_upper"])
                    n_rep, rep_sd = len(lefts), 0.0
                    all_src = ";".join(sorted({o["source_code"] for o in lefts}))
                    all_rr = ";".join(sorted({o["rescue_reason"] for o in lefts}))

            assert math.isfinite(y_final) and math.isfinite(y_lo) and math.isfinite(y_hi) and y_lo <= y_hi
            c17_drug, _ = match_case17(ckey)
            c_rec = all_comps[ckey]
            row_out = {
                "endpoint": ep,
                "task_tier": tier,
                "task_type": "regression",
                "compound_key": ckey,
                "smiles_origin": c_rec["smiles_origin"],
                "seq_origin": c_rec["seq_origin"],
                "y_value": round(y_final, 6),
                "y_lower": round(y_lo, 6),
                "y_upper": round(y_hi, 6),
                "censor_type": ctype,
                "n_replicates": n_rep,
                "replicate_sd": rep_sd,
                "condition_signature": "|".join(cond_vals) if cond_vals else "none",
                "origin": rep["origin"],
                "source_code": rep["source_code"],
                "source_record_id": rep["source_record_id"],
                "record_id": rep["record_id"],
                "all_source_codes": all_src,
                "rescue_reason": rep["rescue_reason"],
                "all_rescue_reasons": all_rr,
                "case17_excluded": 1 if c17_drug else 0,
                "case17_drug": c17_drug,
            }
            for col, vc in zip(cond_cols, cond_vals):
                row_out[col] = vc
            out_rows.append(row_out)
        dedup_by_ep[ep] = out_rows
        dedup_conflicts[ep] = n_conf

    # 4. Compute Morgan bits for ALL candidate new compounds so we can run EXACT incremental clustering BEFORE admission gating!
    print("[4/7] Running incremental Non-Chaining Dual-Track clustering (reusing V33 cluster_ids first!)...", flush=True)
    all_cand_keys = sorted({r["compound_key"] for rows in dedup_by_ep.values() for r in rows})
    new_cand_keys = [ck for ck in all_cand_keys if ck not in v33_key_to_idx]
    print(f"  Candidate compounds total={len(all_cand_keys):,} | already in V33={len(all_cand_keys)-len(new_cand_keys):,} | genuinely NEW={len(new_cand_keys):,}", flush=True)

    # Load existing V33 Morgan bits
    v33_mb = np.load(V33_FEAT / "morgan_bits_1024.npz", allow_pickle=False)["morgan_bits_1024"]

    # Compute exact features for new candidate compounds now (we'll slice down to admitted-only at Step 6!)
    feat_tasks = [
        (i, all_comps[ck]["canonical_smiles"], all_comps[ck]["parent_sequence"], [float(all_comps[ck][m]) for m in MOD_DESC_NAMES])
        for i, ck in enumerate(new_cand_keys)
    ]
    n_new_cand = len(new_cand_keys)
    new_phys = np.zeros((n_new_cand, len(ALL_PHYSCHEM_NAMES)), dtype=np.float32)
    new_m_cnt = np.zeros((n_new_cand, 1024), dtype=np.uint16)
    new_ap = np.zeros((n_new_cand, 1024), dtype=np.uint16)
    new_maccs = np.zeros((n_new_cand, 167), dtype=np.uint8)
    new_tt = np.zeros((n_new_cand, 512), dtype=np.uint16)
    new_m_bit = np.zeros((n_new_cand, 1024), dtype=np.uint8)
    new_smi_valid = np.zeros(n_new_cand, dtype=bool)
    new_seq_valid = np.zeros(n_new_cand, dtype=bool)

    if n_new_cand > 0:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_feat_worker) as ex:
            for loc_i, smi_ok, seq_ok, phys, m_cnt, ap_cnt, maccs, tt_cnt, m_bit in ex.map(_compute_one_feat, feat_tasks, chunksize=64):
                new_smi_valid[loc_i] = smi_ok
                new_seq_valid[loc_i] = seq_ok
                new_phys[loc_i] = phys
                if smi_ok:
                    new_m_cnt[loc_i] = m_cnt
                    new_ap[loc_i] = ap_cnt
                    new_maccs[loc_i] = maccs
                    new_tt[loc_i] = tt_cnt
                    new_m_bit[loc_i] = m_bit

    # Build lookup maps over existing V33 compounds for exact connectivity + Track S + Track M matching
    v33_ik14_to_cid: dict[str, str] = {}
    v33_ch_to_cid: dict[str, str] = {}
    v33_smi_to_cid: dict[str, str] = {}
    v33_track_s_seq_to_cid: dict[str, str] = {}
    v33_track_m_idxs: list[int] = []
    v33_track_m_cids: list[str] = []

    for ck, cid in v33_cluster_map.items():
        c = v33_comps[ck]
        if c["inchikey_14"]:
            v33_ik14_to_cid.setdefault(c["inchikey_14"], cid)
        if c["conn_hash"]:
            v33_ch_to_cid.setdefault(c["conn_hash"], cid)
        if c["canonical_smiles"]:
            v33_smi_to_cid.setdefault(c["canonical_smiles"], cid)
        fidx = v33_key_to_idx[ck]
        is_cyc = int(c["is_cyclic"]) == 1
        pseq = c["parent_sequence"]
        sorig = c["seq_origin"]
        has_smi = bool(v33_mb[fidx].sum() > 0)
        n_known = sum(1 for ch in pseq if ch != "X")
        has_good_lin = (
            (not is_cyc) and bool(pseq)
            and (
                (not has_smi)
                or (sorig in ("source_bare_20aa", "source_parent_projection") and len(pseq) >= 6 and n_known >= 5 and (n_known / len(pseq)) >= 0.75)
                or (sorig == "rdkit_backbone_extracted" and len(pseq) >= 8 and n_known >= 6 and (n_known / len(pseq)) >= 0.80)
            )
        )
        if has_good_lin:
            v33_track_s_seq_to_cid.setdefault(pseq, cid)
        elif has_smi:
            v33_track_m_idxs.append(fidx)
            v33_track_m_cids.append(cid)

    # Assign cluster_id to every candidate compound (reusing V33 first, then incremental Dual-Track!)
    comp_cluster: dict[str, str] = dict(v33_cluster_map)
    # Handle any compound already in v33_key_to_idx but not in v33_cluster_map (e.g., Tirzepatide!)
    unclustered_keys = [ck for ck in all_cand_keys if ck not in comp_cluster]

    # Stage (a): exact connectivity match to existing V33 OR group new compounds with identical ik14/conn_hash
    still_new = []
    for ck in unclustered_keys:
        c = all_comps[ck]
        matched_cid = None
        if c["canonical_smiles"] and c["canonical_smiles"] in v33_smi_to_cid:
            matched_cid = v33_smi_to_cid[c["canonical_smiles"]]
        elif c["inchikey_14"] and c["inchikey_14"] in v33_ik14_to_cid:
            matched_cid = v33_ik14_to_cid[c["inchikey_14"]]
        elif c["conn_hash"] and c["conn_hash"] in v33_ch_to_cid:
            matched_cid = v33_ch_to_cid[c["conn_hash"]]
        if matched_cid is not None:
            comp_cluster[ck] = matched_cid
        else:
            still_new.append(ck)

    # Group remaining new compounds by exact connectivity (ik14 / conn_hash / smi)
    new_key_to_loc = {ck: i for i, ck in enumerate(new_cand_keys)}
    uf_new = UnionFind(len(still_new))
    by_conn_new: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ck in enumerate(still_new):
        c = all_comps[ck]
        if c["inchikey_14"]:
            by_conn_new[("ik14", c["inchikey_14"])].append(i)
        if c["conn_hash"]:
            by_conn_new[("ch", c["conn_hash"])].append(i)
        if c["canonical_smiles"]:
            by_conn_new[("smi", c["canonical_smiles"])].append(i)
    for idxs in by_conn_new.values():
        for j in idxs[1:]:
            uf_new.union(idxs[0], j)

    idg_members = defaultdict(list)
    for i in range(len(still_new)):
        idg_members[uf_new.find(i)].append(i)
    idg_roots = sorted(idg_members.keys())

    new_track_s: dict[int, str] = {}
    new_track_m: list[tuple[int, np.ndarray]] = []
    new_track_o: list[int] = []

    for r in idg_roots:
        ms = idg_members[r]
        any_cyc = any(int(all_comps[still_new[i]]["is_cyclic"]) == 1 for i in ms)
        best_seq, best_sorig = "", "none"
        for i in ms:
            c = all_comps[still_new[i]]
            ps, so = c["parent_sequence"], c["seq_origin"]
            if ps:
                if so == "source_bare_20aa":
                    best_seq, best_sorig = ps, so
                    break
                elif so == "source_parent_projection" and best_sorig != "source_parent_projection":
                    best_seq, best_sorig = ps, so
                elif not best_seq:
                    best_seq, best_sorig = ps, so

        def _get_bits(ck_s: str) -> np.ndarray:
            if ck_s in v33_key_to_idx:
                return v33_mb[v33_key_to_idx[ck_s]]
            return new_m_bit[new_key_to_loc[ck_s]]

        ms_smi = [i for i in ms if _get_bits(still_new[i]).sum() > 0]
        n_known = sum(1 for ch in best_seq if ch != "X")
        has_good_lin = (
            (not any_cyc) and bool(best_seq)
            and (
                (not ms_smi)
                or (best_sorig in ("source_bare_20aa", "source_parent_projection") and len(best_seq) >= 6 and n_known >= 5 and (n_known / len(best_seq)) >= 0.75)
                or (best_sorig == "rdkit_backbone_extracted" and len(best_seq) >= 8 and n_known >= 6 and (n_known / len(best_seq)) >= 0.80)
            )
        )
        if has_good_lin:
            new_track_s[r] = best_seq
        elif ms_smi:
            ms_smi.sort(key=lambda i: (0 if all_comps[still_new[i]]["smiles_origin"] == "source" else 1, still_new[i]))
            new_track_m.append((r, _get_bits(still_new[ms_smi[0]])))
        else:
            new_track_o.append(r)

    root_to_cid: dict[int, str] = {}
    if new_track_s:
        s_map = incremental_cluster_sequences(v33_track_s_seq_to_cid, list(new_track_s.values()))
        for r, seq in new_track_s.items():
            root_to_cid[r] = s_map[seq]
    if new_track_m:
        m_bits_new = np.stack([b for _, b in new_track_m], axis=0)
        ex_m_bits = v33_mb[np.array(v33_track_m_idxs, dtype=np.int64)]
        m_cids = incremental_cluster_morgan(ex_m_bits, v33_track_m_cids, m_bits_new)
        for (r, _), cid in zip(new_track_m, m_cids):
            root_to_cid[r] = cid
    for oi, r in enumerate(new_track_o):
        root_to_cid[r] = f"CLU_V331_O_{oi:06d}"

    for i, ck in enumerate(still_new):
        comp_cluster[ck] = root_to_cid[uf_new.find(i)]

    n_reused_clu = sum(1 for ck in still_new if not comp_cluster[ck].startswith("CLU_V331_"))
    print(f"  New candidate compounds clustered: {len(still_new):,} ({n_reused_clu:,} merged into existing V33 analog clusters, {len(still_new)-n_reused_clu:,} in new V33.1 clusters).", flush=True)

    # 5. Apply smart MFC / MBIC / MBEC policy & strict admission gates!
    print("[5/7] Evaluating strict scientific admission gates & freezing splits...", flush=True)
    # Smart MFC choice: prefer MFC_C_albicans if it passes >=80 rows & >=40 clusters; else use fungal-only MFC_antifungal!
    mfc_ca_rows = dedup_by_ep.get("MFC_C_albicans", [])
    mfc_ca_clu = len({comp_cluster[r["compound_key"]] for r in mfc_ca_rows})
    if len(mfc_ca_rows) >= 80 and mfc_ca_clu >= 40:
        print(f"  [MFC Decision] MFC_C_albicans PASSED ({len(mfc_ca_rows)} rows, {mfc_ca_clu} clusters) -> keeping MFC_C_albicans, dropping pooled MFC_antifungal.", flush=True)
        dedup_by_ep.pop("MFC_antifungal", None)
    else:
        print(f"  [MFC Decision] MFC_C_albicans below gate ({len(mfc_ca_rows)} rows, {mfc_ca_clu} clusters) -> using unified fungal-only MFC_antifungal with cond_species!", flush=True)
        dedup_by_ep.pop("MFC_C_albicans", None)

    # Smart MBIC / MBEC strict-vs-conditioned choice:
    for bf_ep, strict_tag, min_r_strict, min_c_strict in [
        ("MBIC_biofilm", "MBIC", 150, 60),
        ("MBEC_biofilm", "MBEC", 80, 40),
    ]:
        all_bf = dedup_by_ep.get(bf_ep, [])
        strict_bf = [r for r in all_bf if r.get("cond_estimand") == strict_tag]
        strict_clu = len({comp_cluster[r["compound_key"]] for r in strict_bf})
        if len(strict_bf) >= min_r_strict and strict_clu >= min_c_strict:
            print(f"  [{bf_ep}] Strict {strict_tag} alone passed ({len(strict_bf)} rows, {strict_clu} clusters) -> restricting strictly to {strict_tag}!", flush=True)
            dedup_by_ep[bf_ep] = strict_bf
        else:
            all_clu = len({comp_cluster[r["compound_key"]] for r in all_bf})
            print(f"  [{bf_ep}] Strict {strict_tag} ({len(strict_bf)} rows, {strict_clu} clu) < ({min_r_strict}, {min_c_strict}) -> keeping {strict_tag}/{strict_tag}50/{strict_tag}90 with explicit cond_estimand ({len(all_bf)} rows, {all_clu} clu)!", flush=True)

    admitted_summary: dict[str, dict] = {}
    rejected_summary: dict[str, dict] = {}
    admitted_used_keys: set[str] = set()

    for ep in sorted(dedup_by_ep.keys()):
        rows = dedup_by_ep[ep]
        min_r, min_c, min_m = ADMISSION_GATES.get(ep, DEFAULT_TIER2_GATE)
        n_rows = len(rows)
        n_clu = len({comp_cluster[r["compound_key"]] for r in rows})
        n_mols = len({r["compound_key"] for r in rows})
        n_ver_mols = len({
            r["compound_key"] for r in rows
            if all_comps[r["compound_key"]]["canonical_smiles"] or all_comps[r["compound_key"]]["parent_sequence"]
        })
        n_exact = sum(1 for r in rows if r["censor_type"] == "exact")

        if n_rows < min_r or n_clu < min_c or n_ver_mols < min_m or n_exact < 25:
            reason = f"Failed gate (rows={n_rows}/{min_r}, clusters={n_clu}/{min_c}, verified_mols={n_ver_mols}/{min_m}, exact={n_exact}/25)"
            rejected_summary[ep] = {
                "n_rows": n_rows, "n_unique_compounds": n_mols, "n_clusters": n_clu,
                "n_exact": n_exact, "conflicts_dropped": dedup_conflicts.get(ep, 0),
                "reason": reason,
            }
            print(f"  [REJECT] {ep:24s}: {reason}", flush=True)
            continue

        out_rows, info = assign_splits_regression(rows, comp_cluster, all_comps, ep)
        cond_cols = ENDPOINT_COND_COLS[ep]
        out_fields = [
            "endpoint", "task_tier", "task_type", "compound_key", "smiles_origin", "seq_origin",
            "y_value", "y_lower", "y_upper", "censor_type", "n_replicates", "replicate_sd",
            "condition_signature",
        ] + cond_cols + [
            "origin", "source_code", "source_record_id", "record_id", "all_source_codes",
            "rescue_reason", "all_rescue_reasons", "case17_excluded", "case17_drug",
            "cluster_id", "is_holdout", "fold_id",
        ]
        with (OUT_EP_DIR / f"{ep}.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)

        info["task_tier"] = rows[0]["task_tier"]
        info["conflicts_dropped"] = dedup_conflicts.get(ep, 0)
        info["smiles_ready_rows"] = sum(1 for r in rows if r["smiles_origin"] != "missing")
        info["condition_counts"] = {
            c: dict(Counter(r.get(c, "") for r in out_rows).most_common(12))
            for c in cond_cols
        }
        admitted_summary[ep] = info
        for r in out_rows:
            admitted_used_keys.add(r["compound_key"])

        fs = [info["fold_sizes"][f"fold_{i}"] for i in range(5)]
        print(
            f"  [ADMIT ] {ep:24s} ({info['task_tier']}): n={info['n_rows']:5,d} mols={info['n_unique_compounds']:4,d} "
            f"clu={info['n_clusters']:4,d} cens={info['n_censored']:4,d} c17={info['n_case17_excluded']:2d} | "
            f"ho={info['n_holdout_rows']:4,d} ({info['holdout_frac']*100:4.1f}%) | folds={fs}",
            flush=True,
        )

    # 6. Build incremental delta compound table & concatenated V33.1 feature bank (rows 0..46597 BIT-FOR-BIT identical!)
    print("[6/7] Writing delta compounds table & concatenated V33.1 feature bank...", flush=True)
    delta_keys = sorted(k for k in admitted_used_keys if k not in v33_key_to_idx)
    K = len(delta_keys)
    N_TOTAL = N_V33 + K
    print(f"  Admitted endpoints={len(admitted_summary)} | Rejected={len(rejected_summary)} | Delta genuinely NEW compounds K={K:,} -> Total N={N_TOTAL:,}", flush=True)

    # Load frozen V33 arrays to concatenate cleanly (read-only!)
    v33_pc = np.load(V33_FEAT / "physchem_core.npz", allow_pickle=False)
    v33_fp = np.load(V33_FEAT / "fp_compact.npz", allow_pickle=False)
    v33_ad = np.load(V33_FEAT / "morgan_bits_1024.npz", allow_pickle=False)

    if K > 0:
        d_locs = np.array([new_key_to_loc[ck] for ck in delta_keys], dtype=np.int64)
        d_phys = new_phys[d_locs]
        d_smi_v = new_smi_valid[d_locs]
        d_seq_v = new_seq_valid[d_locs]
        d_m_cnt = new_m_cnt[d_locs]
        d_ap = new_ap[d_locs]
        d_maccs = new_maccs[d_locs]
        d_tt = new_tt[d_locs]
        d_m_bit = new_m_bit[d_locs]
        d_m_pack = np.packbits(d_m_bit, axis=1)

        assert np.all(np.isfinite(d_phys)), "Non-finite value in delta physchem_core!"

        all_phys = np.concatenate([v33_pc["X"], d_phys], axis=0)
        all_smi_v = np.concatenate([v33_pc["smiles_valid_mask"], d_smi_v], axis=0)
        all_seq_v = np.concatenate([v33_pc["seq_valid_mask"], d_seq_v], axis=0)
        all_keys_arr = np.concatenate([v33_pc["compound_keys"], np.array(delta_keys)], axis=0)

        all_m_cnt = np.concatenate([v33_fp["morgan_count_r2_1024"], d_m_cnt], axis=0)
        all_ap = np.concatenate([v33_fp["atompair_1024"], d_ap], axis=0)
        all_maccs = np.concatenate([v33_fp["maccs_167"], d_maccs], axis=0)
        all_tt = np.concatenate([v33_fp["torsion_512"], d_tt], axis=0)

        all_m_bit = np.concatenate([v33_ad["morgan_bits_1024"], d_m_bit], axis=0)
        all_m_pack = np.concatenate([v33_ad["morgan_bits_packed"], d_m_pack], axis=0)
    else:
        all_phys = v33_pc["X"]
        all_smi_v = v33_pc["smiles_valid_mask"]
        all_seq_v = v33_pc["seq_valid_mask"]
        all_keys_arr = v33_pc["compound_keys"]
        all_m_cnt = v33_fp["morgan_count_r2_1024"]
        all_ap = v33_fp["atompair_1024"]
        all_maccs = v33_fp["maccs_167"]
        all_tt = v33_fp["torsion_512"]
        all_m_bit = v33_ad["morgan_bits_1024"]
        all_m_pack = v33_ad["morgan_bits_packed"]

    # Hard assertion: rows 0..46597 are BIT-FOR-BIT identical to V33!
    assert np.array_equal(all_phys[:N_V33], v33_pc["X"])
    assert np.array_equal(all_m_bit[:N_V33], v33_ad["morgan_bits_1024"])

    # Write delta_compounds.tsv (for local RTX 5080 delta 4-FM extraction!)
    delta_fields = ["compound_idx", "compound_key", "canonical_smiles", "parent_sequence", "smiles_origin", "seq_origin", "smiles_valid", "seq_valid"]
    with (OUT_DATA / "delta_compounds.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=delta_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for j, ck in enumerate(delta_keys):
            cidx = N_V33 + j
            c = all_comps[ck]
            w.writerow({
                "compound_idx": cidx,
                "compound_key": ck,
                "canonical_smiles": c["canonical_smiles"],
                "parent_sequence": c["parent_sequence"],
                "smiles_origin": c["smiles_origin"],
                "seq_origin": c["seq_origin"],
                "smiles_valid": int(all_smi_v[cidx]),
                "seq_valid": int(all_seq_v[cidx]),
            })

    # Write 5 existing V33 anchor compounds (spanning modified SMILES + AMP sequences) for cosine > 0.999 verification!
    anchor_indices = [
        v33_feat_manifest["case17_indices"]["Semaglutide"],
        v33_feat_manifest["case17_indices"]["Cyclosporine_A"],
        v33_feat_manifest["case17_indices"]["Caspofungin"],
    ]
    # Pick 2 linear AMP compounds from existing V33 with both valid SMILES & sequence
    for r_i in range(N_V33):
        if len(anchor_indices) >= 5:
            break
        if r_i in anchor_indices:
            continue
        ck = str(all_keys_arr[r_i])
        c = v33_comps.get(ck)
        if c and all_smi_v[r_i] and all_seq_v[r_i] and c["smiles_origin"] == "rdkit_from_canonical_seq" and 12 <= len(c["parent_sequence"]) <= 25:
            anchor_indices.append(r_i)
    with (OUT_DATA / "anchor_5_compounds.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=delta_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for cidx in anchor_indices:
            ck = str(all_keys_arr[cidx])
            c = v33_comps[ck]
            w.writerow({
                "compound_idx": cidx,
                "compound_key": ck,
                "canonical_smiles": c["canonical_smiles"],
                "parent_sequence": c["parent_sequence"],
                "smiles_origin": c["smiles_origin"],
                "seq_origin": c["seq_origin"],
                "smiles_valid": int(all_smi_v[cidx]),
                "seq_valid": int(all_seq_v[cidx]),
            })

    # Write compounds_v331_merged.tsv (preserving all existing compounds_v33.tsv rows + appending K new compounds!)
    with (V33_DATA / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fin:
        r_in = csv.DictReader(fin, delimiter="\t")
        comp_fields = list(r_in.fieldnames or [])
        existing_comp_rows = list(r_in)
    with (OUT_DATA / "compounds_v331_merged.tsv").open("w", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=comp_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(existing_comp_rows)
        for ck in delta_keys:
            rec = dict(all_comps[ck])
            rec["cluster_id"] = comp_cluster[ck]
            w.writerow({k: rec[k] for k in comp_fields})

    # Write concatenated compound_index.tsv & .npz feature bank
    idx_fields = ["compound_idx", "compound_key", "is_case17", "case17_drug_name", "smiles_origin", "seq_origin", "smiles_valid", "seq_valid"]
    with (OUT_FEAT / "compound_index.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=idx_fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(v33_idx_rows)
        for j, ck in enumerate(delta_keys):
            cidx = N_V33 + j
            c = all_comps[ck]
            w.writerow({
                "compound_idx": cidx,
                "compound_key": ck,
                "is_case17": 0,
                "case17_drug_name": "",
                "smiles_origin": c["smiles_origin"],
                "seq_origin": c["seq_origin"],
                "smiles_valid": int(all_smi_v[cidx]),
                "seq_valid": int(all_seq_v[cidx]),
            })

    phys_path = OUT_FEAT / "physchem_core.npz"
    np.savez_compressed(
        phys_path,
        X=all_phys,
        feature_names=np.array(ALL_PHYSCHEM_NAMES),
        smiles_valid_mask=all_smi_v,
        seq_valid_mask=all_seq_v,
        compound_keys=all_keys_arr,
    )
    fp_path = OUT_FEAT / "fp_compact.npz"
    np.savez_compressed(
        fp_path,
        morgan_count_r2_1024=all_m_cnt,
        atompair_1024=all_ap,
        maccs_167=all_maccs,
        torsion_512=all_tt,
        smiles_valid_mask=all_smi_v,
        compound_keys=all_keys_arr,
    )
    ad_path = OUT_FEAT / "morgan_bits_1024.npz"
    np.savez_compressed(
        ad_path,
        morgan_bits_1024=all_m_bit,
        morgan_bits_packed=all_m_pack,
        smiles_valid_mask=all_smi_v,
        compound_keys=all_keys_arr,
    )

    feat_manifest = {
        "schema": "peptide_omnipanel_v331_compact_feature_bank_v1",
        "rdkit_version": rdkit.__version__,
        "numpy_version": np.__version__,
        "n_v33_base": N_V33,
        "n_delta_new": K,
        "n_compounds_total": N_TOTAL,
        "n_smiles_valid": int(all_smi_v.sum()),
        "n_seq_valid": int(all_seq_v.sum()),
        "n_delta_smiles_valid": int(all_smi_v[N_V33:].sum()) if K > 0 else 0,
        "n_delta_seq_valid": int(all_seq_v[N_V33:].sum()) if K > 0 else 0,
        "case17_indices": v33_feat_manifest["case17_indices"],
        "physchem_core": {
            "file": str(phys_path),
            "sha256": sha256_file(phys_path),
            "shape": list(all_phys.shape),
            "feature_names": ALL_PHYSCHEM_NAMES,
        },
        "fp_compact": {"file": str(fp_path), "sha256": sha256_file(fp_path)},
        "morgan_bits_1024": {"file": str(ad_path), "sha256": sha256_file(ad_path)},
    }
    (OUT_FEAT / "feature_manifest.json").write_text(json.dumps(feat_manifest, indent=2), encoding="utf-8")

    # 7. Save comprehensive audit summary
    summary = {
        "schema": "peptide_omnipanel_v331_expansion_data_summary_v1",
        "elapsed_sec": round(time.time() - t0, 2),
        "n_v33_existing_endpoints": 24,
        "n_new_admitted_endpoints": len(admitted_summary),
        "n_total_endpoints": 24 + len(admitted_summary),
        "n_rejected_candidates": len(rejected_summary),
        "n_v33_compounds": N_V33,
        "n_delta_new_compounds": K,
        "n_delta_smiles_valid": feat_manifest["n_delta_smiles_valid"],
        "n_delta_seq_valid": feat_manifest["n_delta_seq_valid"],
        "n_total_unified_compounds": N_TOTAL,
        "admitted_endpoints": admitted_summary,
        "rejected_candidates": rejected_summary,
    }
    (OUT_DATA / "v331_split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[7/7] DONE Step 02 in {summary['elapsed_sec']}s! Admitted {len(admitted_summary)} new endpoints (24 + {len(admitted_summary)} = {24+len(admitted_summary)} total), K={K:,} delta compounds (smi={feat_manifest['n_delta_smiles_valid']:,}, seq={feat_manifest['n_delta_seq_valid']:,})!", flush=True)


if __name__ == "__main__":
    main()
