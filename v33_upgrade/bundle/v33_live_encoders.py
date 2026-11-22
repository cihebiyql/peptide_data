#!/usr/bin/env python3
"""Self-contained live featurizer, 4-foundation-model encoder stack (ncAA-RepDistill v2,
PeptideCLM-2, ChemBERTa-MTR, ESM-C 600M via persistent IPC worker), 46,598-compound
precomputed bank cache + LRU cache, and duck-typed LiveMiniBank for Peptide OmniPanel V33.

Reproduces Wave-1 (`02_clean_canonicalize_dedup.py`, `04_compute_compact_features.py`)
and Wave-2B (`extract_*.py`) preprocessing & pooling contracts with exact numerical parity.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import json
import math
import os
import re
import socket
import struct
import sys
import threading
import time
import types
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors
from torch import nn

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10_000_000)

BASE_DIR = Path(os.environ.get("V33_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
_PORTABLE_PRETRAINED = BASE_DIR.parent / "pretrained"


def _resolve_pretrained(env_var: str, portable_rel: str, legacy_path: str) -> Path:
    val = os.environ.get(env_var, "").strip()
    if val:
        return Path(val)
    cand = _PORTABLE_PRETRAINED / portable_rel
    if cand.exists():
        return cand
    return Path(legacy_path)


NCAA_REPO = _resolve_pretrained("NCAA_REPO_DIR", "ncaadistill", "/data/qlyu/ncaadistill")
NCAA_CONFIG_PATH = Path(os.environ.get("NCAA_CONFIG_PATH", str(NCAA_REPO / "config/phase2/phase2_repro_sweep24_e0_b0.yaml")))
NCAA_CKPT_PATH = Path(os.environ.get("NCAA_CKPT_PATH", str(NCAA_REPO / "artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt")))
NCAA_VOCAB_PATH = Path(os.environ.get("NCAA_VOCAB_PATH", str(NCAA_REPO / "data/raw/adaptation/vocab_smiles.txt")))
EXPECTED_VOCAB_SHA256 = "7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92"

PEPTIDECLM2_DIR = _resolve_pretrained(
    "V33_PEPTIDECLM2_DIR",
    "peptideclm2",
    "/data/qlyu/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/models/peptideclm-2-hybrid-small",
)
CHEMBERTA_DIR = _resolve_pretrained(
    "V33_CHEMBERTA_DIR",
    "chemberta_mtr",
    "/data/qlyu/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca",
)

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = set(AA20)
AA20_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,1022}$")

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

# Precompiled SMARTS & RDKit fingerprint generators (100% matching Wave-1)
_MOD_SMARTS = {
    "disulfide": Chem.MolFromSmarts("[#16X2]-[#16X2]"),
    "nmethyl": Chem.MolFromSmarts("[NX3;H0;!$([N]1CCC[C]1)]([CH3])([CX3]=O)[#6]"),
    "peg": Chem.MolFromSmarts("[#8X2][CX4H2][CX4H2][#8X2][CX4H2][CX4H2][#8X2,#7X3]"),
    "c_amid": Chem.MolFromSmarts("[NX3]-[CX4H1,CX4H2]-[CX3](=O)-[NH2]"),
    "n_acet": Chem.MolFromSmarts("[CH3]-[CX3](=O)-[NX3H1]-[CX4]-[CX3](=O)[#7,#8]"),
    "backbone_chiral": Chem.MolFromSmarts("[NX3;H2,H1,H0]-[CX4H1]-[CX3](=O)[#7,#8]"),
    "carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "acyl": Chem.MolFromSmarts("[CX3](=O)[NX3,OX2]"),
}
_ION_SMARTS = {
    "pos_amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(NS=O);!$([n])][CX4]"),
    "pos_guanidine": Chem.MolFromSmarts("[NX3;H1,H2]C(=[NX2;H1,H0])[NX3;H1,H2]"),
    "neg_carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "neg_sulfate_phos": Chem.MolFromSmarts("[SX4,PX4](=O)(=O)[OX2H1,OX1-]"),
}
_FP_GENS = {
    "morgan_cnt": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
    "morgan_bit": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
    "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
    "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=512, includeChirality=True),
}
_RDKIT_LOCK = threading.Lock()

_NCAA_TOKEN_PAT = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Zn|[B-IK-PR-Z]|[bcnops]|\(|\)|=|#|-|\+|\\|/|\.|:|~|@|\?|>|\*|\$|%\d\d|\d)"
)
_NCAA_ATOM_SET = {
    "B", "C", "N", "O", "P", "S", "F", "I", "Cl", "Br", "Si", "Se",
    "Na", "Ca", "Li", "Mg", "Al", "Zn", "K", "H",
}
_NCAA_AROM_SET = {"b", "c", "n", "o", "p", "s"}


def _safe_float(x: Any, lo: float = -1e6, hi: float = 1e6) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return 0.0
        return max(lo, min(hi, v))
    except Exception:
        return 0.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def clean_sequence(seq: Optional[str]) -> str:
    if not seq:
        return ""
    lines = [ln.strip() for ln in str(seq).splitlines() if ln.strip() and not ln.strip().startswith(">")]
    return re.sub(r"\s+", "", "".join(lines)).upper()


def is_valid_canonical_seq(seq: str) -> bool:
    return bool(seq and AA20_RE.fullmatch(seq))


def _seq_charge_at_ph(seq: str, ph: float) -> float:
    q = 1.0 / (1.0 + 10.0 ** (ph - 8.0))
    q -= 1.0 / (1.0 + 10.0 ** (3.1 - ph))
    for ch in seq:
        if ch in PKA_POS:
            q += 1.0 / (1.0 + 10.0 ** (ph - PKA_POS[ch]))
        elif ch in PKA_NEG:
            q -= 1.0 / (1.0 + 10.0 ** (PKA_NEG[ch] - ph))
    return q


def compute_seq_descriptors(seq: str) -> Tuple[bool, np.ndarray]:
    """Exact 35D sequence descriptors matching 04_compute_compact_features.py."""
    clean = [c for c in (seq or "").upper() if c in KD_SCALE]
    if len(clean) < 2:
        return False, np.zeros(35, dtype=np.float32)
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
    return True, np.array([_safe_float(x) for x in feats], dtype=np.float32)


def _compute_macrocycle_ring_size(mol: Chem.Mol) -> int:
    ri = mol.GetRingInfo()
    max_ring = max((len(r) for r in ri.AtomRings()), default=0)
    if max_ring >= 8:
        return max_ring
    ss_matches = mol.GetSubstructMatches(_MOD_SMARTS["disulfide"])
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


def _compute_fatty_chains(mol: Chem.Mol) -> Tuple[int, int]:
    carboxyl_c = {m[0] for m in mol.GetSubstructMatches(_MOD_SMARTS["carboxyl"])}
    acyl_c = {m[0] for m in mol.GetSubstructMatches(_MOD_SMARTS["acyl"])} - carboxyl_c
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

            def dfs(u: int, prev: int, length: int, path_set: set):
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


def _count_d_aa_approx(mol: Chem.Mol) -> int:
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        return 0
    matches = mol.GetSubstructMatches(_MOD_SMARTS["backbone_chiral"])
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
        if len(sc_nbs) != 1:
            continue
        beta = sc_nbs[0]
        if beta.GetAtomicNum() != 6:
            continue
        beta_has_s = any(nb.GetAtomicNum() >= 16 for nb in beta.GetNeighbors() if nb.GetIdx() != ca_idx)
        if beta_has_s:
            if cip == "S":
                d_count += 1
        else:
            if cip == "R":
                d_count += 1
    return d_count


def compute_smiles_features_live(smiles: str, from_canonical_seq: bool = False) -> Dict[str, Any]:
    """Compute exact 48D SMILES+Mod physchem + compact chiral FPs matching Wave-1 bit-for-bit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return {"valid": False}

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
        bertz = (
            _safe_float(Descriptors.BertzCT(mol), 0.0, 1e6)
            if heavy <= 350
            else _safe_float(heavy * math.log2(max(2.0, heavy)) * 10.0, 0.0, 1e6)
        )
    except Exception:
        bertz = 0.0
    k1 = _safe_float(rdMolDescriptors.CalcKappa1(mol), 0.0, 1e5)
    k2 = _safe_float(rdMolDescriptors.CalcKappa2(mol), 0.0, 1e5)
    hka = _safe_float(rdMolDescriptors.CalcHallKierAlpha(mol), -1e4, 1e4)
    chi0v = _safe_float(rdMolDescriptors.CalcChi0v(mol), 0.0, 1e5)
    chi1v = _safe_float(rdMolDescriptors.CalcChi1v(mol), 0.0, 1e5)
    amide = float(rdMolDescriptors.CalcNumAmideBonds(mol))
    chiral = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=True)))
    pos_ion = float(
        len(mol.GetSubstructMatches(_ION_SMARTS["pos_amine"]))
        + len(mol.GetSubstructMatches(_ION_SMARTS["pos_guanidine"]))
    )
    neg_ion = float(
        len(mol.GetSubstructMatches(_ION_SMARTS["neg_carboxyl"]))
        + len(mol.GetSubstructMatches(_ION_SMARTS["neg_sulfate_phos"]))
    )
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

    # Exact 10 modification descriptors matching 02_clean_canonicalize_dedup.py
    macro_size = _compute_macrocycle_ring_size(mol)
    is_cyclic = 1.0 if macro_size >= 8 else 0.0
    n_disulfide = float(len(mol.GetSubstructMatches(_MOD_SMARTS["disulfide"])))
    n_nmethyl = float(len(mol.GetSubstructMatches(_MOD_SMARTS["nmethyl"])))
    if from_canonical_seq:
        n_d_aa = 0.0
        diacid_len, monoacyl_len = 0.0, 0.0
        has_peg, has_c_amid, has_n_acet = 0.0, 0.0, 0.0
    else:
        n_d_aa = float(_count_d_aa_approx(mol))
        dl, ml = _compute_fatty_chains(mol)
        diacid_len, monoacyl_len = float(dl), float(ml)
        has_peg = 1.0 if mol.HasSubstructMatch(_MOD_SMARTS["peg"]) else 0.0
        has_c_amid = 1.0 if mol.HasSubstructMatch(_MOD_SMARTS["c_amid"]) else 0.0
        has_n_acet = 1.0 if mol.HasSubstructMatch(_MOD_SMARTS["n_acet"]) else 0.0

    mod_feats = [
        is_cyclic, float(macro_size), n_disulfide, n_nmethyl,
        n_d_aa, diacid_len, monoacyl_len,
        has_peg, has_c_amid, has_n_acet,
    ]
    phys48 = np.array([_safe_float(x) for x in rd_feats] + mod_feats, dtype=np.float32)

    with _RDKIT_LOCK:
        m_cnt = np.clip(_FP_GENS["morgan_cnt"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
        m_bit = _FP_GENS["morgan_bit"].GetFingerprintAsNumPy(mol).astype(np.uint8)
        ap_cnt = np.clip(_FP_GENS["atompair"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
        tt_cnt = np.clip(_FP_GENS["torsion"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint8)

    m_packed = np.packbits(m_bit).astype(np.uint8)
    return {
        "valid": True,
        "phys48": phys48,
        "morgan_count": m_cnt,
        "atompair": ap_cnt,
        "maccs": maccs,
        "torsion": tt_cnt,
        "morgan_bits": m_bit,
        "morgan_packed": m_packed,
    }


def _compute_ncaa_graph_distances(smiles: str, max_length: int = 768, max_dist: int = 8) -> np.ndarray:
    """Exact in-memory reproduction of ncaadistill/data/dataset.py _get_graph_distances."""
    mol = Chem.MolFromSmiles(smiles)
    L_out = min(len(smiles) + 2, max_length)
    if mol is None:
        return np.zeros((L_out, L_out), dtype=np.int16)

    spans = []
    idx = 0
    for match in _NCAA_TOKEN_PAT.finditer(smiles):
        if match.start() != idx:
            return np.zeros((L_out, L_out), dtype=np.int16)
        spans.append((match.start(), match.end(), match.group()))
        idx = match.end()
    if idx != len(smiles):
        return np.zeros((L_out, L_out), dtype=np.int16)

    mapping = [-1] * len(smiles)
    atom_idx = 0
    for start, end, token in spans:
        if token.startswith("[") or token in _NCAA_ATOM_SET or token in _NCAA_AROM_SET:
            for pos in range(start, end):
                mapping[pos] = atom_idx
            atom_idx += 1
    if atom_idx != mol.GetNumAtoms():
        return np.zeros((L_out, L_out), dtype=np.int16)

    dist = Chem.GetDistanceMatrix(mol).astype(np.int16)
    max_smiles_len = max_length - 2
    mapping = mapping[:max_smiles_len]

    token_to_atom = np.full((L_out,), -1, dtype=np.int16)
    n_map = min(len(mapping), L_out - 1)
    token_to_atom[1 : 1 + n_map] = np.array(mapping[:n_map], dtype=np.int16)

    ai = token_to_atom[:, None]
    aj = token_to_atom[None, :]
    mask = (ai >= 0) & (aj >= 0)

    dist_tok = np.zeros((L_out, L_out), dtype=np.int16)
    if mask.any():
        dist_clipped = np.minimum(dist, max_dist).astype(np.int16)
        ai_safe = np.maximum(ai, 0)
        aj_safe = np.maximum(aj, 0)
        dist_tok = np.where(mask, dist_clipped[ai_safe, aj_safe] + 1, 0).astype(np.int16)
    return dist_tok


class NCAALiveEncoder:
    """Warm in-RAM ncAA-RepDistill v2 encoder (exact EMA weights + VocabManager + graph_attention)."""

    def __init__(self):
        if str(NCAA_REPO) not in sys.path:
            sys.path.insert(0, str(NCAA_REPO))
        from data.vocab_manager import VocabManager
        from model.single_input_dual_teacher import SingleInputDualTeacher

        t0 = time.time()
        vocab_sha = _sha256_file(NCAA_VOCAB_PATH)
        assert vocab_sha == EXPECTED_VOCAB_SHA256, f"Vocab SHA mismatch: {vocab_sha}"
        self.vocab_mgr = VocabManager(str(NCAA_VOCAB_PATH))
        assert len(self.vocab_mgr) == 37

        with NCAA_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        smiles_cfg = cfg["student"]["smiles_encoder"]
        proj_cfg = cfg["student"]["projection"]
        ga_cfg = dict(cfg.get("student", {}).get("graph_attention") or {})
        ga_cfg["cache_size"] = 1
        cproj_cfg = cfg.get("student", {}).get("contrastive_proj") or {}
        fusion_cfg = cfg.get("fusion") or {}

        self.max_length = int(smiles_cfg["max_length"])
        self.payload_limit = self.max_length - 2
        self.max_dist = int(ga_cfg.get("max_distance", 8))

        model = SingleInputDualTeacher(
            smiles_vocab_size=int(smiles_cfg["vocab_size"]),
            smiles_embed_dim=int(smiles_cfg["embed_dim"]),
            smiles_num_layers=int(smiles_cfg["num_layers"]),
            smiles_num_heads=int(smiles_cfg["num_heads"]),
            smiles_max_length=self.max_length,
            esm2_dim=int(proj_cfg["esm2_dim"]),
            chemberta_dim=int(proj_cfg["chemberta_dim"]),
            graph_attention=ga_cfg,
            contrastive_proj_cfg=cproj_cfg,
            fusion_cfg=fusion_cfg,
            use_smiles_mlm=bool(cfg["student"].get("use_smiles_mlm", True)),
            dropout=float(smiles_cfg.get("dropout", 0.1)),
            use_gradient_checkpointing=False,
        )
        ckpt = torch.load(str(NCAA_CKPT_PATH), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        non_rope_missing = [k for k in missing if "rope.inv_freq" not in k]
        assert not non_rope_missing, f"ncAA unexpected missing keys: {non_rope_missing}"
        model.eval()
        self.model = model
        w_gb = self.model.smiles_encoder.graph_bias_embed.weight.detach()
        self.neutral_graph_idx = int(w_gb[1:].norm(dim=1).argmin().item()) + 1
        self.lock = threading.Lock()
        self.load_sec = round(time.time() - t0, 2)
        print(
            f"[enc] ncAA-RepDistill v2 ready in {self.load_sec}s (neutral_graph_idx={self.neutral_graph_idx}, vocab_sha={vocab_sha[:8]})",
            flush=True,
        )

    def encode_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(smiles_list)
        sr = np.zeros((n, 768), dtype=np.float32)
        fh = np.zeros((n, 2048), dtype=np.float32)
        pp = np.zeros((n, 1280), dtype=np.float32)
        if n == 0:
            return sr, fh, pp

        pad_id = int(self.vocab_mgr.pad_token_id)
        std_items = [(i, s) for i, s in enumerate(smiles_list) if len(s) <= self.payload_limit]
        long_items = [(i, s) for i, s in enumerate(smiles_list) if len(s) > self.payload_limit]

        with self.lock, torch.inference_mode():
            if std_items:
                std_items.sort(key=lambda x: len(x[1]))
                for pos in range(0, len(std_items), 16):
                    sub = std_items[pos : pos + 16]
                    encoded = [
                        self.vocab_mgr.encode(list(s), add_special_tokens=True, max_length=self.max_length, padding=False, truncation=True)
                        for _, s in sub
                    ]
                    L = max(len(e) for e in encoded)
                    ids_t = torch.full((len(sub), L), pad_id, dtype=torch.long)
                    attn_t = torch.zeros((len(sub), L), dtype=torch.long)
                    gd_t = torch.zeros((len(sub), L, L), dtype=torch.long)
                    for bi, ((_, s), e) in enumerate(zip(sub, encoded)):
                        le = len(e)
                        ids_t[bi, :le] = torch.tensor(e, dtype=torch.long)
                        attn_t[bi, :le] = 1
                        gd = _compute_ncaa_graph_distances(s, self.max_length, self.max_dist)
                        gL = min(le, gd.shape[0])
                        gd_t[bi, :gL, :gL] = torch.from_numpy(gd[:gL, :gL].astype(np.int64))
                    out = self.model(ids_t, attn_t, graph_distances=gd_t, return_all=False)
                    sr_np = out["smiles_repr"].float().numpy()
                    fh_np = out["fusion_hf"].float().numpy()
                    pp_np = out["protein_proj"].float().numpy()
                    for bi, (orig_i, _) in enumerate(sub):
                        sr[orig_i] = sr_np[bi]
                        fh[orig_i] = fh_np[bi]
                        pp[orig_i] = pp_np[bi]

            if long_items:
                for orig_i, smi in long_items:
                    windows = []
                    weights = []
                    for st in range(0, len(smi), self.payload_limit):
                        w = smi[st : st + self.payload_limit]
                        if w:
                            windows.append(w)
                            weights.append(len(w))
                    encoded = [
                        self.vocab_mgr.encode(list(w), add_special_tokens=True, max_length=self.max_length, padding=False, truncation=True)
                        for w in windows
                    ]
                    L = max(len(e) for e in encoded)
                    ids_t = torch.full((len(windows), L), pad_id, dtype=torch.long)
                    attn_t = torch.zeros((len(windows), L), dtype=torch.long)
                    gd_t = torch.zeros((len(windows), L, L), dtype=torch.long)
                    for bi, e in enumerate(encoded):
                        lv = len(e)
                        ids_t[bi, :lv] = torch.tensor(e, dtype=torch.long)
                        attn_t[bi, :lv] = 1
                        gd_t[bi, :lv, :lv] = self.neutral_graph_idx
                    out = self.model(ids_t, attn_t, graph_distances=gd_t, return_all=False)
                    warr = np.asarray(weights, dtype=np.float32)
                    wn = warr[:, None] / warr.sum()
                    sr[orig_i] = (out["smiles_repr"].float().numpy() * wn).sum(axis=0)
                    fh[orig_i] = (out["fusion_hf"].float().numpy() * wn).sum(axis=0)
                    pp[orig_i] = (out["protein_proj"].float().numpy() * wn).sum(axis=0)

        return sr, fh, pp


class PeptideCLM2LiveEncoder:
    """Warm in-RAM PeptideCLM-2 (hybrid-small 512D) encoder."""

    def __init__(self):
        from safetensors.torch import load_file as safetensors_load_file
        from transformers import PreTrainedTokenizerFast

        t0 = time.time()
        pkg_name = "v33_live_peptideclm2"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(PEPTIDECLM2_DIR.resolve())]
            sys.modules[pkg_name] = pkg
        chem_mod = importlib.import_module(pkg_name + ".ChemPepMTR")
        cfg_cls = importlib.import_module(pkg_name + ".config").model_config
        cfg = cfg_cls(**json.loads((PEPTIDECLM2_DIR / "config.json").read_text(encoding="utf-8")))

        encoder = chem_mod.MLM_model(cfg)
        sd = safetensors_load_file(str(PEPTIDECLM2_DIR / "model.safetensors"), device="cpu")
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        non_buf_missing = [k for k in missing if not (k.endswith(".theta") or k.endswith(".cache"))]
        assert not non_buf_missing and not unexpected, f"PeptideCLM-2 load mismatch: {missing}"
        encoder.model.sequence_head = nn.Identity()
        encoder.eval()
        self.encoder = encoder
        self.max_len = 2048
        self.payload = self.max_len - 2
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(PEPTIDECLM2_DIR / "tokenizer.json"),
            pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
            model_max_length=self.max_len,
        )
        self.lock = threading.Lock()
        self.load_sec = round(time.time() - t0, 2)
        print(f"[enc] PeptideCLM-2 ready in {self.load_sec}s", flush=True)

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        n = len(smiles_list)
        out_arr = np.zeros((n, 512), dtype=np.float32)
        if n == 0:
            return out_arr

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        std_items = []
        long_items = []
        for i, smi in enumerate(smiles_list):
            ids = self.tokenizer(smi, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
            if len(ids) <= self.payload:
                std_items.append((i, ids))
            else:
                long_items.append((i, ids))

        with self.lock, torch.inference_mode():
            if std_items:
                std_items.sort(key=lambda x: len(x[1]))
                for pos in range(0, len(std_items), 16):
                    sub = std_items[pos : pos + 16]
                    seqs = [[cls_id] + it[1] + [sep_id] for it in sub]
                    L = max(len(s) for s in seqs)
                    ids_t = torch.full((len(sub), L), pad_id, dtype=torch.long)
                    mask_t = torch.zeros((len(sub), L), dtype=torch.long)
                    for bi, s in enumerate(seqs):
                        ids_t[bi, : len(s)] = torch.tensor(s, dtype=torch.long)
                        mask_t[bi, : len(s)] = 1
                    vecs = self.encoder(input_ids=ids_t, attention_mask=mask_t).mean_pool.float().numpy()
                    for bi, (orig_i, _) in enumerate(sub):
                        out_arr[orig_i] = vecs[bi]

            for orig_i, full_ids in long_items:
                windows = []
                weights = []
                for start in range(0, len(full_ids), self.payload):
                    w = full_ids[start : start + self.payload]
                    if w:
                        windows.append([cls_id] + w + [sep_id])
                        weights.append(len(w))
                L = max(len(w) for w in windows)
                ids_t = torch.full((len(windows), L), pad_id, dtype=torch.long)
                mask_t = torch.zeros((len(windows), L), dtype=torch.long)
                for bi, w in enumerate(windows):
                    ids_t[bi, : len(w)] = torch.tensor(w, dtype=torch.long)
                    mask_t[bi, : len(w)] = 1
                wmat = self.encoder(input_ids=ids_t, attention_mask=mask_t).mean_pool.float().numpy()
                warr = np.asarray(weights, dtype=np.float32)
                out_arr[orig_i] = (wmat * (warr[:, None] / warr.sum())).sum(axis=0)

        return out_arr


class ChemBERTaLiveEncoder:
    """Warm in-RAM ChemBERTa-77M-MTR (384D) encoder (exact BOS/EOS-excluded mean pool)."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        t0 = time.time()
        self.max_len = 512
        self.payload = self.max_len - 2
        self.tok = AutoTokenizer.from_pretrained(str(CHEMBERTA_DIR), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(CHEMBERTA_DIR), local_files_only=True, torch_dtype=torch.float32)
        self.model.eval()
        self.lock = threading.Lock()
        self.load_sec = round(time.time() - t0, 2)
        print(f"[enc] ChemBERTa-MTR ready in {self.load_sec}s", flush=True)

    @staticmethod
    def _pool_no_bos_eos(hidden: torch.Tensor, attn: torch.Tensor) -> np.ndarray:
        am = attn.bool()
        tm = am.clone()
        tm[:, 0] = False
        lens = am.sum(dim=1)
        for bi in range(tm.size(0)):
            Lk = int(lens[bi].item())
            if Lk > 2:
                tm[bi, Lk - 1] = False
            else:
                tm[bi] = am[bi]
        denom = tm.sum(dim=1, keepdim=True).clamp(min=1)
        return ((hidden * tm.unsqueeze(-1)).sum(dim=1) / denom).numpy()

    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        n = len(smiles_list)
        out_arr = np.zeros((n, 384), dtype=np.float32)
        if n == 0:
            return out_arr

        raw_ids = self.tok(smiles_list, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        short_idx = [i for i, ids in enumerate(raw_ids) if len(ids) <= self.payload]
        long_items = [(i, ids) for i, ids in enumerate(raw_ids) if len(ids) > self.payload]

        with self.lock, torch.inference_mode():
            if short_idx:
                for pos in range(0, len(short_idx), 32):
                    sub_i = short_idx[pos : pos + 32]
                    sub_s = [smiles_list[i] for i in sub_i]
                    enc = self.tok(sub_s, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
                    h = self.model(**enc).last_hidden_state.float()
                    pooled = self._pool_no_bos_eos(h, enc["attention_mask"])
                    for bi, orig_i in enumerate(sub_i):
                        out_arr[orig_i] = pooled[bi]

            if long_items:
                cls_id = self.tok.cls_token_id if self.tok.cls_token_id is not None else self.tok.bos_token_id
                sep_id = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.tok.eos_token_id
                pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                for orig_i, full_ids in long_items:
                    windows = []
                    weights = []
                    for start in range(0, len(full_ids), self.payload):
                        w = full_ids[start : start + self.payload]
                        if w:
                            windows.append([cls_id] + w + [sep_id])
                            weights.append(len(w))
                    L = max(len(w) for w in windows)
                    w_ids = torch.full((len(windows), L), pad_id, dtype=torch.long)
                    w_attn = torch.zeros((len(windows), L), dtype=torch.long)
                    for bi, w in enumerate(windows):
                        w_ids[bi, : len(w)] = torch.tensor(w, dtype=torch.long)
                        w_attn[bi, : len(w)] = 1
                    h = self.model(input_ids=w_ids, attention_mask=w_attn).last_hidden_state.float()
                    wmat = self._pool_no_bos_eos(h, w_attn)
                    warr = np.asarray(weights, dtype=np.float32)
                    out_arr[orig_i] = (wmat * (warr[:, None] / warr.sum())).sum(axis=0)

        return out_arr


class ESMCWorkerClient:
    """Thread-safe persistent TCP client for warm esmc_worker.py on localhost."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18902, timeout: float = 180.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.lock = threading.Lock()
        self._sock: Optional[socket.socket] = None

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        s.settimeout(self.timeout)
        self._sock = s
        return s

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise EOFError("esmc_worker closed connection")
            buf.extend(chunk)
        return bytes(buf)

    def _call(self, payload: dict) -> dict:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        frame = struct.pack("!I", len(raw)) + raw
        with self.lock:
            for attempt in range(2):
                try:
                    s = self._connect()
                    s.sendall(frame)
                    hdr = self._recv_exact(s, 4)
                    length = struct.unpack("!I", hdr)[0]
                    resp_bytes = self._recv_exact(s, length)
                    return json.loads(resp_bytes.decode("utf-8"))
                except Exception:
                    self._close()
                    if attempt == 1:
                        raise

    def health(self) -> dict:
        try:
            return self._call({"cmd": "health"})
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        if not sequences:
            return np.zeros((0, 4, 1152), dtype=np.float32)
        resp = self._call({"cmd": "encode", "sequences": sequences})
        if not resp.get("ok"):
            raise RuntimeError(f"esmc_worker error: {resp.get('error')}")
        raw = base64.b64decode(resp["data_b64"])
        shape = tuple(resp["shape"])
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


class LiveMiniBank:
    """Duck-typed batch bank matching BOTH Wave-2C `FeatureBank` and Wave-2D `FusionBank`
    interfaces so pickled `FoldPreprocessor.transform()` and `EmbeddingFoldPreprocessor.transform()`
    execute identically on live unseen molecules!
    """

    def __init__(
        self,
        physchem: np.ndarray,
        smiles_valid: np.ndarray,
        seq_valid: np.ndarray,
        morgan_count: np.ndarray,
        atompair: np.ndarray,
        maccs: np.ndarray,
        torsion: np.ndarray,
        morgan_bits: np.ndarray,
        morgan_packed: np.ndarray,
        emb_blocks: Dict[str, np.ndarray],
        emb_valid: Dict[str, np.ndarray],
    ):
        self.physchem = np.ascontiguousarray(physchem, dtype=np.float32)
        self.smiles_valid = np.ascontiguousarray(smiles_valid, dtype=bool)
        self.seq_valid = np.ascontiguousarray(seq_valid, dtype=bool)
        self.morgan_count = np.ascontiguousarray(morgan_count, dtype=np.uint16)
        self.atompair = np.ascontiguousarray(atompair, dtype=np.uint16)
        self.maccs = np.ascontiguousarray(maccs, dtype=np.uint8)
        self.torsion = np.ascontiguousarray(torsion, dtype=np.uint16)
        self.morgan_bits = np.ascontiguousarray(morgan_bits, dtype=np.uint8)
        self.morgan_packed = np.ascontiguousarray(morgan_packed, dtype=np.uint8)
        self.emb_blocks = emb_blocks
        self.emb_valid = emb_valid

    def get_fp_block(self, idx: np.ndarray, kind: str) -> np.ndarray:
        if kind == "morgan":
            return np.log1p(self.morgan_count[idx].astype(np.float32))
        if kind == "svd_src":
            return np.log1p(
                np.concatenate(
                    [
                        self.morgan_count[idx].astype(np.float32),
                        self.atompair[idx].astype(np.float32),
                        self.maccs[idx].astype(np.float32),
                    ],
                    axis=1,
                )
            )
        if kind == "fp_all":
            return np.log1p(
                np.concatenate(
                    [
                        self.morgan_count[idx].astype(np.float32),
                        self.atompair[idx].astype(np.float32),
                        self.maccs[idx].astype(np.float32),
                        self.torsion[idx].astype(np.float32),
                    ],
                    axis=1,
                )
            )
        raise ValueError(f"Unknown fp block kind: {kind}")

    def get_emb_f32(self, block_name: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.emb_blocks[block_name][idx].astype(np.float32),
            self.emb_valid[block_name][idx].copy(),
        )


class V33EncodingHub:
    """Unified smart-cache + true live 4-foundation-model encoding engine."""

    def __init__(self, esmc_port: int = 18902, torch_threads: int = 16, lru_capacity: int = 4096):
        t0 = time.time()
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

        print("[hub] Loading 46,598-compound precomputed V33 feature & embedding bank into RAM...", flush=True)
        key_to_idx: Dict[str, int] = {}
        with (FEATURES_DIR / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                key_to_idx[r["compound_key"]] = int(r["compound_idx"])

        pc = np.load(FEATURES_DIR / "physchem_core.npz")
        self.bank_physchem = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.bank_smi_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.bank_seq_valid = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        fp = np.load(FEATURES_DIR / "fp_compact.npz")
        self.bank_morgan_cnt = np.ascontiguousarray(fp["morgan_count_r2_1024"], dtype=np.uint16)
        self.bank_atompair = np.ascontiguousarray(fp["atompair_1024"], dtype=np.uint16)
        self.bank_maccs = np.ascontiguousarray(fp["maccs_167"], dtype=np.uint8)
        self.bank_torsion = np.ascontiguousarray(fp["torsion_512"], dtype=np.uint16)
        fp.close()

        mb = np.load(FEATURES_DIR / "morgan_bits_1024.npz")
        self.bank_morgan_bits = np.ascontiguousarray(mb["morgan_bits_1024"], dtype=np.uint8)
        self.bank_morgan_packed = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        # Map canonical SMILES -> compound_idx AND canonical parent_sequence -> compound_idx
        self.smi_to_bank_idx: Dict[str, int] = {}
        self.seq_to_bank_idx: Dict[str, int] = {}
        for fn in ("compounds_v33.tsv", "case17_queries.tsv"):
            with (DATA_DIR / fn).open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    cidx = key_to_idx.get(r["compound_key"])
                    if cidx is None:
                        continue
                    csmi = r.get("canonical_smiles", "")
                    if csmi and self.bank_smi_valid[cidx] and csmi not in self.smi_to_bank_idx:
                        self.smi_to_bank_idx[csmi] = cidx
                    pseq = r.get("parent_sequence", "")
                    if pseq and self.bank_seq_valid[cidx] and pseq not in self.seq_to_bank_idx:
                        self.seq_to_bank_idx[pseq] = cidx

        # Precomputed embeddings (compact float16)
        d_nc = np.load(EMBEDDINGS_DIR / "ncaa_repdistill_v2.npz")
        self.bank_ncaa_smi = np.ascontiguousarray(d_nc["smiles_repr"], dtype=np.float16)
        self.bank_ncaa_fus = np.ascontiguousarray(d_nc["fusion_hf"], dtype=np.float16)
        self.bank_ncaa_prot = np.ascontiguousarray(d_nc["protein_proj"], dtype=np.float16)
        d_nc.close()

        d_pc = np.load(EMBEDDINGS_DIR / "peptideclm2.npz")
        self.bank_pclm2 = np.ascontiguousarray(d_pc["embeddings"], dtype=np.float16)
        d_pc.close()

        d_cb = np.load(EMBEDDINGS_DIR / "chemberta_mtr.npz")
        self.bank_chemberta = np.ascontiguousarray(d_cb["embeddings"], dtype=np.float16)
        d_cb.close()

        d_es = np.load(EMBEDDINGS_DIR / "esmc_600m_multiregion.npz")
        self.bank_esmc_mean = np.ascontiguousarray(d_es["mean"], dtype=np.float16)
        self.bank_esmc_max = np.ascontiguousarray(d_es["max"], dtype=np.float16)
        self.bank_esmc_nt5 = np.ascontiguousarray(d_es["nterm5"], dtype=np.float16)
        self.bank_esmc_ct5 = np.ascontiguousarray(d_es["cterm5"], dtype=np.float16)
        d_es.close()

        print(
            f"[hub] Bank loaded in {time.time()-t0:.2f}s | smi_cache={len(self.smi_to_bank_idx):,} | seq_cache={len(self.seq_to_bank_idx):,}",
            flush=True,
        )

        # Warm live encoders in main process + IPC client to warm ESM-C worker
        self.ncaa = NCAALiveEncoder()
        self.pclm2 = PeptideCLM2LiveEncoder()
        self.chemberta = ChemBERTaLiveEncoder()
        self.esmc = ESMCWorkerClient(host="127.0.0.1", port=esmc_port)

        self.lru_capacity = lru_capacity
        self._smi_lru: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._seq_lru: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lru_lock = threading.Lock()

    def encoders_status(self) -> Dict[str, Any]:
        esmc_h = self.esmc.health()
        return {
            "ncaa_repdistill_v2": {"ready": True, "load_sec": self.ncaa.load_sec},
            "peptideclm2": {"ready": True, "load_sec": self.pclm2.load_sec},
            "chemberta_mtr": {"ready": True, "load_sec": self.chemberta.load_sec},
            "esmc_600m": {"ready": bool(esmc_h.get("ok")), **esmc_h},
            "bank_smiles_cached": len(self.smi_to_bank_idx),
            "bank_sequences_cached": len(self.seq_to_bank_idx),
        }

    def resolve_item(self, smiles: Optional[str], sequence: Optional[str]) -> Dict[str, Any]:
        """Resolve modality track & canonical representations strictly per Wave 3A contract:
          - Only valid canonical 20-AA sequence -> Chem.MolFromSequence(seq) -> dual_modality
          - Only SMILES -> smiles_only (ZERO sequence/ESM-C features!)
          - Both SMILES & sequence -> dual_modality
        """
        raw_smi = (smiles or "").strip()
        raw_seq = clean_sequence(sequence)

        canon_smi = ""
        seq_clean = ""
        from_canon_seq = False
        track = "smiles_only"

        if raw_smi and raw_seq:
            mol = Chem.MolFromSmiles(raw_smi)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("Invalid SMILES string provided.")
            canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            seq_clean = raw_seq
            track = "dual_modality"
        elif raw_smi:
            mol = Chem.MolFromSmiles(raw_smi)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("Invalid SMILES string provided.")
            canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            seq_clean = ""
            track = "smiles_only"
        elif raw_seq:
            if not is_valid_canonical_seq(raw_seq):
                raise ValueError("Invalid canonical 20-AA sequence provided (must contain 2-1022 standard AA letters).")
            seq_clean = raw_seq
            track = "dual_modality"
            # Deterministically generate L-peptide RDKit SMILES for dual-modality coverage!
            if len(seq_clean) <= 150:
                try:
                    mol = Chem.MolFromSequence(seq_clean)
                    if mol is not None and mol.GetNumAtoms() > 0:
                        canon_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                        from_canon_seq = True
                except Exception:
                    canon_smi = ""
        else:
            raise ValueError("Either 'smiles' or 'sequence' must be provided.")

        return {
            "canonical_smiles": canon_smi,
            "canonical_sequence": seq_clean,
            "from_canonical_seq": from_canon_seq,
            "track": track,
        }

    def encode_resolved_batch(
        self,
        resolved_items: List[Dict[str, Any]],
        force_live_encode: bool = False,
    ) -> Tuple[LiveMiniBank, List[bool], Dict[str, float]]:
        """Encode a batch of resolved items into a LiveMiniBank (using bank/LRU cache unless force_live_encode=True)."""
        t0 = time.time()
        B = len(resolved_items)

        physchem = np.zeros((B, 83), dtype=np.float32)
        smi_valid = np.zeros(B, dtype=bool)
        seq_valid = np.zeros(B, dtype=bool)
        m_cnt = np.zeros((B, 1024), dtype=np.uint16)
        ap_cnt = np.zeros((B, 1024), dtype=np.uint16)
        maccs = np.zeros((B, 167), dtype=np.uint8)
        tt_cnt = np.zeros((B, 512), dtype=np.uint16)
        m_bit = np.zeros((B, 1024), dtype=np.uint8)
        m_pack = np.zeros((B, 128), dtype=np.uint8)

        ncaa_smi = np.zeros((B, 768), dtype=np.float32)
        ncaa_fus = np.zeros((B, 2048), dtype=np.float32)
        ncaa_prot = np.zeros((B, 1280), dtype=np.float32)
        pclm2 = np.zeros((B, 512), dtype=np.float32)
        chemberta = np.zeros((B, 384), dtype=np.float32)
        esmc_mean = np.zeros((B, 1152), dtype=np.float32)
        esmc_max = np.zeros((B, 1152), dtype=np.float32)
        esmc_nt5 = np.zeros((B, 1152), dtype=np.float32)
        esmc_ct5 = np.zeros((B, 1152), dtype=np.float32)

        smi_cache_hits = [True] * B
        seq_cache_hits = [True] * B

        # 1. Resolve SMILES features & embeddings (bank cache -> LRU -> live encode)
        live_smi_map: Dict[Tuple[str, bool], List[int]] = {}
        for i, item in enumerate(resolved_items):
            csmi = item["canonical_smiles"]
            if not csmi:
                continue
            from_seq = item["from_canonical_seq"]
            if not force_live_encode:
                bidx = self.smi_to_bank_idx.get(csmi)
                if bidx is not None:
                    smi_valid[i] = True
                    physchem[i, :48] = self.bank_physchem[bidx, :48]
                    m_cnt[i] = self.bank_morgan_cnt[bidx]
                    ap_cnt[i] = self.bank_atompair[bidx]
                    maccs[i] = self.bank_maccs[bidx]
                    tt_cnt[i] = self.bank_torsion[bidx]
                    m_bit[i] = self.bank_morgan_bits[bidx]
                    m_pack[i] = self.bank_morgan_packed[bidx]
                    ncaa_smi[i] = self.bank_ncaa_smi[bidx].astype(np.float32)
                    ncaa_fus[i] = self.bank_ncaa_fus[bidx].astype(np.float32)
                    ncaa_prot[i] = self.bank_ncaa_prot[bidx].astype(np.float32)
                    pclm2[i] = self.bank_pclm2[bidx].astype(np.float32)
                    chemberta[i] = self.bank_chemberta[bidx].astype(np.float32)
                    continue
                with self._lru_lock:
                    cached = self._smi_lru.get(csmi)
                    if cached is not None:
                        self._smi_lru.move_to_end(csmi)
                if cached is not None:
                    smi_valid[i] = True
                    physchem[i, :48] = cached["phys48"]
                    m_cnt[i] = cached["morgan_count"]
                    ap_cnt[i] = cached["atompair"]
                    maccs[i] = cached["maccs"]
                    tt_cnt[i] = cached["torsion"]
                    m_bit[i] = cached["morgan_bits"]
                    m_pack[i] = cached["morgan_packed"]
                    ncaa_smi[i] = cached["ncaa_smi"]
                    ncaa_fus[i] = cached["ncaa_fus"]
                    ncaa_prot[i] = cached["ncaa_prot"]
                    pclm2[i] = cached["pclm2"]
                    chemberta[i] = cached["chemberta"]
                    continue
            smi_cache_hits[i] = False
            live_smi_map.setdefault((csmi, from_seq), []).append(i)

        t_smi_live = 0.0
        if live_smi_map:
            ts0 = time.time()
            uniq_keys = list(live_smi_map.keys())
            uniq_smis = [k[0] for k in uniq_keys]
            rd_list = [compute_smiles_features_live(smi, from_canonical_seq=fseq) for smi, fseq in uniq_keys]
            ok_pos = [j for j, d in enumerate(rd_list) if d["valid"]]
            ok_smis = [uniq_smis[j] for j in ok_pos]
            if ok_smis:
                sr_arr, fh_arr, pp_arr = self.ncaa.encode_batch(ok_smis)
                pc_arr = self.pclm2.encode_batch(ok_smis)
                cb_arr = self.chemberta.encode_batch(ok_smis)
                for loc_j, orig_j in enumerate(ok_pos):
                    csmi = uniq_smis[orig_j]
                    rd = rd_list[orig_j]
                    entry = {
                        "phys48": rd["phys48"],
                        "morgan_count": rd["morgan_count"],
                        "atompair": rd["atompair"],
                        "maccs": rd["maccs"],
                        "torsion": rd["torsion"],
                        "morgan_bits": rd["morgan_bits"],
                        "morgan_packed": rd["morgan_packed"],
                        "ncaa_smi": sr_arr[loc_j],
                        "ncaa_fus": fh_arr[loc_j],
                        "ncaa_prot": pp_arr[loc_j],
                        "pclm2": pc_arr[loc_j],
                        "chemberta": cb_arr[loc_j],
                    }
                    if not force_live_encode:
                        with self._lru_lock:
                            self._smi_lru[csmi] = entry
                            self._smi_lru.move_to_end(csmi)
                            if len(self._smi_lru) > self.lru_capacity:
                                self._smi_lru.popitem(last=False)
                    for i in live_smi_map[uniq_keys[orig_j]]:
                        smi_valid[i] = True
                        physchem[i, :48] = entry["phys48"]
                        m_cnt[i] = entry["morgan_count"]
                        ap_cnt[i] = entry["atompair"]
                        maccs[i] = entry["maccs"]
                        tt_cnt[i] = entry["torsion"]
                        m_bit[i] = entry["morgan_bits"]
                        m_pack[i] = entry["morgan_packed"]
                        ncaa_smi[i] = entry["ncaa_smi"]
                        ncaa_fus[i] = entry["ncaa_fus"]
                        ncaa_prot[i] = entry["ncaa_prot"]
                        pclm2[i] = entry["pclm2"]
                        chemberta[i] = entry["chemberta"]
            t_smi_live = (time.time() - ts0) * 1000.0

        # 2. Resolve Sequence features & ESM-C 600M (ONLY for dual_modality items!)
        live_seq_map: Dict[str, List[int]] = {}
        for i, item in enumerate(resolved_items):
            if item["track"] != "dual_modality":
                continue
            seq = item["canonical_sequence"]
            if not seq:
                continue
            if not force_live_encode:
                bidx = self.seq_to_bank_idx.get(seq)
                if bidx is not None:
                    seq_valid[i] = True
                    physchem[i, 48:83] = self.bank_physchem[bidx, 48:83]
                    esmc_mean[i] = self.bank_esmc_mean[bidx].astype(np.float32)
                    esmc_max[i] = self.bank_esmc_max[bidx].astype(np.float32)
                    esmc_nt5[i] = self.bank_esmc_nt5[bidx].astype(np.float32)
                    esmc_ct5[i] = self.bank_esmc_ct5[bidx].astype(np.float32)
                    continue
                with self._lru_lock:
                    cached_q = self._seq_lru.get(seq)
                    if cached_q is not None:
                        self._seq_lru.move_to_end(seq)
                if cached_q is not None:
                    seq_valid[i] = True
                    physchem[i, 48:83] = cached_q["phys35"]
                    esmc_mean[i] = cached_q["esmc_4reg"][0]
                    esmc_max[i] = cached_q["esmc_4reg"][1]
                    esmc_nt5[i] = cached_q["esmc_4reg"][2]
                    esmc_ct5[i] = cached_q["esmc_4reg"][3]
                    continue
            seq_cache_hits[i] = False
            live_seq_map.setdefault(seq, []).append(i)

        t_esmc_live = 0.0
        if live_seq_map:
            te0 = time.time()
            uniq_seqs = list(live_seq_map.keys())
            esmc_4 = self.esmc.encode_batch(uniq_seqs)  # (U, 4, 1152)
            for j, seq in enumerate(uniq_seqs):
                sv_ok, phys35 = compute_seq_descriptors(seq)
                if not sv_ok:
                    continue
                q_entry = {"phys35": phys35, "esmc_4reg": esmc_4[j]}
                if not force_live_encode:
                    with self._lru_lock:
                        self._seq_lru[seq] = q_entry
                        self._seq_lru.move_to_end(seq)
                        if len(self._seq_lru) > self.lru_capacity:
                            self._seq_lru.popitem(last=False)
                for i in live_seq_map[seq]:
                    seq_valid[i] = True
                    physchem[i, 48:83] = phys35
                    esmc_mean[i] = esmc_4[j, 0]
                    esmc_max[i] = esmc_4[j, 1]
                    esmc_nt5[i] = esmc_4[j, 2]
                    esmc_ct5[i] = esmc_4[j, 3]
            t_esmc_live = (time.time() - te0) * 1000.0

        emb_blocks = {
            "ncaa_smiles_repr": ncaa_smi,
            "ncaa_fusion_hf": ncaa_fus,
            "ncaa_protein_proj": ncaa_prot,
            "peptideclm2": pclm2,
            "chemberta_mtr": chemberta,
            "esmc_mean": esmc_mean,
            "esmc_max": esmc_max,
            "esmc_nterm5": esmc_nt5,
            "esmc_cterm5": esmc_ct5,
        }
        emb_valid = {
            "ncaa_smiles_repr": smi_valid,
            "ncaa_fusion_hf": smi_valid,
            "ncaa_protein_proj": smi_valid,
            "peptideclm2": smi_valid,
            "chemberta_mtr": smi_valid,
            "esmc_mean": seq_valid,
            "esmc_max": seq_valid,
            "esmc_nterm5": seq_valid,
            "esmc_cterm5": seq_valid,
        }
        mini_bank = LiveMiniBank(
            physchem=physchem,
            smiles_valid=smi_valid,
            seq_valid=seq_valid,
            morgan_count=m_cnt,
            atompair=ap_cnt,
            maccs=maccs,
            torsion=tt_cnt,
            morgan_bits=m_bit,
            morgan_packed=m_pack,
            emb_blocks=emb_blocks,
            emb_valid=emb_valid,
        )
        cache_hits = [smi_cache_hits[i] and seq_cache_hits[i] for i in range(B)]
        timings = {
            "encode_total_ms": round((time.time() - t0) * 1000.0, 2),
            "smiles_live_ms": round(t_smi_live, 2),
            "esmc_live_ms": round(t_esmc_live, 2),
        }
        return mini_bank, cache_hits, timings
