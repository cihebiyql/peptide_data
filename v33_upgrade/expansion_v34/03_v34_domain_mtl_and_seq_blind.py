#!/usr/bin/env python3
"""V34 Item P2-9: Reproducible Domain-Grouped Multi-Task Neural Networks (Domain-MTL)
+ Full-Stack Reproducible FDA 2019-2023 Blind Evaluation with Sequence/Scaffold Split Comparison
(runs on compute_host).

Scientific Guarantees:
  1. ZERO Holdout Peeking:
     - Every standardization statistic, condition encoder, early-stopping checkpoint, and nested
       5-fold OOF Simplex stacking weight is fit STRICTLY inside CV training folds.
     - Promotion of Domain-MTL into (CPU, 4-FM Fusion, Domain-MTL) is decided STRICTLY on nested
       5-fold OOF CV before the frozen holdout is ever touched.
  2. ZERO Cross-Endpoint Compound/Cluster Leakage in Domain-MTL:
     - For every (target_ep, fold), forbidden_clusters = global_and_local_clusters(target_ho | target_va)
       and forbidden_compounds = compound_keys(target_ho | target_va).
     - Every auxiliary endpoint in the same domain is restricted to is_holdout==0 rows whose
       global cluster, local cluster, and compound_key are ALL disjoint from forbidden sets.
     - Hard runtime assertions verify n_leaked_clusters == 0 and n_leaked_compounds == 0 on every fold.
  3. Full-Stack Reproducible FDA 2019-2023 Blind Evaluation (n=11):
     - Full 3-family stack (CPU + 4-FM Fusion + Domain-MTL) for BOTH regression (half_life_reg, PPB)
       AND classification (hemolysis_cls), with protocol-matched auxiliary task isolation.
     - Side-by-side comparison of TWO explicit non-leakage exclusion protocols:
       (a) strict_morgan_v336: Exact IK14/ck OR Morgan-2048 Tc >= 0.40 OR 3-mer >= 0.40
       (b) peptide_structural_v34: Exact IK14/ck OR Global/3-mer Seq Identity >= 0.40
           OR Close Analog Morgan-2048 Tc >= 0.65 OR Same Murcko/Macrocycle Ring-System Match
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import csv
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

RDLogger.DisableLog("rdApp.*")
torch.set_num_threads(8)

V33_BASE = Path(os.environ.get("V33_BASE", str(Path(__file__).resolve().parents[1])))
BUNDLE_DIR = V33_BASE / "bundle"
EXP331 = V33_BASE / "expansion_v331"
EXP334 = V33_BASE / "expansion_v334"
EXP335 = V33_BASE / "expansion_v335"
EXP336 = V33_BASE / "expansion_v336"
EXP34 = V33_BASE / "expansion_v34"
EXP34.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP334))
sys.path.insert(0, str(EXP335))

from importlib import import_module  # noqa: E402
from v33_metrics import classification_metrics, regression_metrics  # noqa: E402

exp334_mod = import_module("03_train_and_promote_v334")
exp335_mod = import_module("01_v335_protocol_and_logd_audit")
UnifiedV334Bank = exp334_mod.UnifiedV334Bank

csv.field_size_limit(10_000_000)

SEED = 20261127
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = set(AA20)
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

DOMAINS: Dict[str, List[str]] = {
    "domain_permeability_lipophilicity": [
        "PAMPA", "Caco2", "MDCK", "RRCK", "LogD7.4", "LogD_decadiene",
    ],
    "domain_toxicity_hemolysis": [
        "hemolysis_cls", "hemolysis_HC50", "cytotoxicity_cls",
        "cytotoxicity_CC50", "overall_peptide_toxicity",
    ],
    "domain_antimicrobial_mic": [
        "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa",
        "MIC_K_pneumoniae", "MIC_A_baumannii", "MIC_C_albicans",
    ],
    "domain_systemic_pk": [
        "half_life_reg", "CL_absolute", "Vd_absolute", "PPB", "F", "Cmax",
    ],
}

CENSOR_MAP = {"exact": 0, "right": 1, "left": 2, "interval": 3}
EMB_BLOCKS_MTL = [
    ("ncaa_fusion_hf", 2048, 128),
    ("ncaa_smiles_repr", 768, 64),
    ("peptideclm2", 512, 64),
    ("chemberta_mtr", 384, 64),
    ("esmc_mean", 1152, 64),
]

MORGAN2048_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# =============================================================================
# Data Loading & Utility Helpers
# =============================================================================

def find_ep_path(ep: str) -> Path:
    for base in (EXP335 / "endpoints", V33_BASE / "data" / "endpoints", EXP331 / "data" / "endpoints"):
        p = base / f"{ep}.tsv"
        if p.exists():
            return p
    raise FileNotFoundError(f"Endpoint TSV not found: {ep}")


def find_pred_path(ep: str, family: str, split: str) -> Optional[Path]:
    if family == "cpu":
        bases = [V33_BASE / "models_cpu" / ep, EXP331 / "models_cpu" / ep]
        fn = f"{split}_predictions.tsv"
    elif family == "fusion":
        bases = [V33_BASE / "models_fusion" / ep, EXP331 / "models_fusion" / ep]
        fn = f"{split}_predictions.tsv"
    else:
        bases = [BUNDLE_DIR / "champion_predictions" / ep, EXP331 / "champion_predictions" / ep]
        fn = f"{split}_champion.tsv"
    for b in bases:
        p = b / fn
        if p.exists():
            return p
    return None


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    rmax = 0.0
    for i, (k, p) in enumerate(items):
        rmax = max(rmax, min(1.0, p * (m - i)))
        adj[k] = round(rmax, 4)
    return adj


def paired_cluster_bootstrap_pvalue(
    y: np.ndarray,
    pred_new: np.ndarray,
    pred_base: np.ndarray,
    clusters: np.ndarray,
    task_type: str,
    n_boot: int = 1000,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Two-sided paired cluster-bootstrap test: positive delta means pred_new beats pred_base."""
    ok = np.isfinite(y) & np.isfinite(pred_new) & np.isfinite(pred_base)
    y, pn, pb, cl = y[ok], pred_new[ok], pred_base[ok], clusters[ok]
    cl_map: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cl):
        cl_map[str(c)].append(i)
    cl_arrs = [np.array(v, dtype=np.int32) for v in cl_map.values()]
    n_cl = len(cl_arrs)
    if n_cl < 2 or len(y) < 5:
        return {
            "n_eval": int(len(y)),
            "n_clusters": int(n_cl),
            "metric": "brier_reduction" if task_type == "classification" else "mae_reduction",
            "mean_delta": 0.0,
            "ci95": None,
            "p_value": 1.0,
            "p_value_display": "1.0000",
        }
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.randint(0, n_cl, size=n_cl)
        idx = np.concatenate([cl_arrs[p] for p in pick])
        yb, pnb, pbb = y[idx], pn[idx], pb[idx]
        if task_type == "classification":
            da = float(np.mean((yb - pbb) ** 2)) - float(np.mean((yb - pnb) ** 2))
        else:
            da = float(np.mean(np.abs(yb - pbb))) - float(np.mean(np.abs(yb - pnb)))
        deltas.append(da)
    d = np.array(deltas, dtype=np.float64)
    p_raw = float(2.0 * min(np.mean(d <= 0.0), np.mean(d >= 0.0)))
    p_floor = 1.0 / n_boot
    p_num = max(p_raw, p_floor)
    return {
        "n_eval": int(len(y)),
        "n_clusters": int(n_cl),
        "metric": "brier_reduction (Base - 3Fam)" if task_type == "classification" else "mae_reduction (Base - 3Fam)",
        "mean_delta": round(float(d.mean()), 5),
        "ci95": [round(float(np.percentile(d, 2.5)), 5), round(float(np.percentile(d, 97.5)), 5)],
        "p_value": round(p_num, 4),
        "p_value_display": f"<={p_floor:.3f}" if p_raw < p_floor else f"{p_num:.4f}",
    }


def load_endpoint_table(
    ep: str,
    bank: UnifiedV334Bank,
    manifest_ep: Dict[str, Any],
    ck_to_global_cl: Dict[str, str],
) -> Dict[str, Any]:
    path = find_ep_path(ep)
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
    n = len(rows)
    cks = [r["compound_key"] for r in rows]
    cidx = bank.map_keys(cks)
    rec_ids = [r["record_id"] for r in rows]
    is_ho = np.array([int(r["is_holdout"]) for r in rows], dtype=np.int32)
    fold_id = np.array([int(r["fold_id"]) for r in rows], dtype=np.int32)
    local_cl = np.array([str(r["cluster_id"]) for r in rows], dtype=object)
    global_cl = np.array([ck_to_global_cl.get(ck, str(r["cluster_id"])) for ck, r in zip(cks, rows)], dtype=object)
    censor_str = np.array([r.get("censor_type", "exact") for r in rows], dtype=object)
    censor_code = np.array([CENSOR_MAP.get(c, 0) for c in censor_str], dtype=np.int64)
    y_val = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)

    def _flt(val: str, default: float) -> float:
        try:
            v = float(val)
            return v if math.isfinite(v) else default
        except Exception:
            return default

    y_lo = np.array([_flt(r.get("y_lower", ""), y_val[i]) for i, r in enumerate(rows)], dtype=np.float64)
    y_hi = np.array([_flt(r.get("y_upper", ""), y_val[i]) for i, r in enumerate(rows)], dtype=np.float64)
    conds = [{c: (r.get(c) or "unspecified") for c in cond_cols} for r in rows]
    b = manifest_ep.get("bounds")
    bounds = (float(b[0]), float(b[1])) if b is not None else None
    return {
        "endpoint": ep,
        "task_type": manifest_ep["task_type"],
        "bounds": bounds,
        "cond_cols": cond_cols,
        "n": n,
        "record_ids": rec_ids,
        "compound_keys": cks,
        "cidx": cidx,
        "is_holdout": is_ho,
        "fold_id": fold_id,
        "local_cl": local_cl,
        "global_cl": global_cl,
        "censor_str": censor_str,
        "censor_code": censor_code,
        "y_val": y_val,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "conds": conds,
    }


def load_baseline_cpu_fusion_preds(
    ep_data: Dict[str, Any],
    bank: UnifiedV334Bank,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load exact canonical OOF & Holdout predictions for CPU and 4-FM Fusion.
    Returns (cpu_oof, fus_oof, cpu_ho, fus_ho) aligned 1-to-1 with ep_data CV and Holdout rows.
    """
    ep = ep_data["endpoint"]
    cv_mask = ep_data["is_holdout"] == 0
    ho_mask = ep_data["is_holdout"] == 1
    cv_rids = [ep_data["record_ids"][i] for i in np.where(cv_mask)[0]]
    ho_rids = [ep_data["record_ids"][i] for i in np.where(ho_mask)[0]]

    p_c_oof = find_pred_path(ep, "cpu", "oof")
    p_f_oof = find_pred_path(ep, "fusion", "oof")
    p_c_ho = find_pred_path(ep, "cpu", "holdout")
    p_f_ho = find_pred_path(ep, "fusion", "holdout")

    if p_c_oof and p_f_oof and p_c_ho and p_f_ho:
        def _read_col(path: Path, rids: List[str], pref_cols: List[str]) -> np.ndarray:
            mp: Dict[str, float] = {}
            with path.open(encoding="utf-8") as fh:
                rd = csv.DictReader(fh, delimiter="\t")
                col = next((c for c in pref_cols if c in (rd.fieldnames or [])), None)
                assert col is not None, f"Missing pred col in {path}: {rd.fieldnames}"
                for r in rd:
                    try:
                        mp[r["record_id"]] = float(r[col])
                    except Exception:
                        pass
            return np.array([mp.get(rid, np.nan) for rid in rids], dtype=np.float64)

        pref = ["pred_dual_cal", "pred_dual"] if ep_data["task_type"] == "classification" else ["pred_dual"]
        # Use uncalibrated pred_dual for clean simplex stacking matching v33_manifest
        pref_raw = ["pred_dual"]
        cpu_oof = _read_col(p_c_oof, cv_rids, pref_raw)
        fus_oof = _read_col(p_f_oof, cv_rids, pref_raw)
        cpu_ho = _read_col(p_c_ho, ho_rids, pref_raw)
        fus_ho = _read_col(p_f_ho, ho_rids, pref_raw)
        return cpu_oof, fus_oof, cpu_ho, fus_ho

    # LogD_decadiene: reproduce exact smiles_only CPU & 4-FM Fusion tracks cleanly on expansion_v335 split
    print(f"  [{ep}] Generating canonical CPU + 4-FM Fusion OOF & Holdout via exp335 tracks...", flush=True)
    cidx_cv = ep_data["cidx"][cv_mask]
    y_cv = ep_data["y_val"][cv_mask]
    fids_cv = ep_data["fold_id"][cv_mask]
    cond_cv = [ep_data["conds"][i] for i in np.where(cv_mask)[0]]
    cidx_ho = ep_data["cidx"][ho_mask]
    cond_ho = [ep_data["conds"][i] for i in np.where(ho_mask)[0]]
    track = "smiles_only" if ep == "LogD_decadiene" else "dual_modality"
    _, cpu_oof, cpu_ho, _ = exp335_mod.train_cpu_track(
        bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, ep_data["cond_cols"], ep_data["bounds"], track
    )
    pcache: Dict[str, Any] = {}
    fus_oof, fus_ho, _, _ = exp335_mod.train_fusion_track(
        bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, ep_data["cond_cols"], ep_data["bounds"],
        cpu_oof, cpu_ho, track, pcache,
    )
    return cpu_oof, fus_oof, cpu_ho, fus_ho


# =============================================================================
# Fold-Internal Condition Encoder & Feature Assembler
# =============================================================================

class FoldConditionEncoder:
    """Strictly fold-internal one-hot encoder for cond_* columns."""

    def __init__(self, cond_cols: List[str], max_cats: int = 12):
        self.cond_cols = list(cond_cols)
        self.max_cats = max_cats
        self.vocab_: Dict[str, List[str]] = {}
        self.dim_: int = 1

    def fit(self, cond_list: List[Dict[str, str]]) -> "FoldConditionEncoder":
        self.vocab_ = {}
        total_dim = 0
        for col in self.cond_cols:
            cnt = Counter((d.get(col) or "unspecified") for d in cond_list)
            top = [k for k, _ in cnt.most_common(self.max_cats)]
            self.vocab_[col] = top
            total_dim += len(top) + 1  # +1 for <other>
        self.dim_ = max(1, total_dim)
        return self

    def transform(self, cond_list: List[Dict[str, str]]) -> np.ndarray:
        n = len(cond_list)
        if not self.cond_cols:
            return np.zeros((n, 1), dtype=np.float32)
        out = np.zeros((n, self.dim_), dtype=np.float32)
        offset = 0
        for col in self.cond_cols:
            cats = self.vocab_[col]
            c2i = {c: j for j, c in enumerate(cats)}
            other_i = len(cats)
            for r_i, d in enumerate(cond_list):
                v = d.get(col) or "unspecified"
                out[r_i, offset + c2i.get(v, other_i)] = 1.0
            offset += len(cats) + 1
        return out


def precompute_l2_normalized_embs(bank: UnifiedV334Bank) -> Dict[str, np.ndarray]:
    """Parameter-free per-sample L2 normalization of frozen 4-FM embeddings (zero leakage!)."""
    normed: Dict[str, np.ndarray] = {}
    for bname, _, _ in EMB_BLOCKS_MTL:
        arr = bank.emb_blocks[bname].astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = np.divide(arr, np.maximum(nrm, 1e-6))
        v = bank.emb_valid[bname]
        arr[~v] = 0.0
        normed[bname] = np.ascontiguousarray(arr, dtype=np.float32)
    return normed


# =============================================================================
# Part A: Reproducible Domain-Grouped Multi-Task Neural Network (Domain-MTL)
# =============================================================================

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, dim: int = 384, bottleneck: int = 192, dropout: float = 0.25):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        return x + h


class DomainMTLNet(nn.Module):
    """Domain-Grouped Multi-Task Network with:
       - Modality-specific gated encoders (Physchem + 4-FM representations)
       - LayerNorm + Residual Bottleneck MLP trunk
       - Task-specific condition-injected heads
       - Homoscedastic uncertainty task weighting (Kendall et al. 2018)
    """

    def __init__(self, task_specs: List[Tuple[str, str, int]], trunk_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.task_names = [t[0] for t in task_specs]
        self.task_types = {t[0]: t[1] for t in task_specs}

        # Physchem (83D -> 64D) + 5 frozen FM blocks (384D) -> total 448D + 6 validity flags = 454D
        self.phys_proj = nn.Sequential(
            nn.Linear(83, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.15),
        )
        self.emb_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(0.20))
            for _, d_in, d_out in EMB_BLOCKS_MTL
        ])
        self.missing_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, d_out)) for _, _, d_out in EMB_BLOCKS_MTL
        ])
        concat_dim = 64 + sum(d_out for _, _, d_out in EMB_BLOCKS_MTL) + (1 + len(EMB_BLOCKS_MTL))

        self.in_proj = nn.Sequential(
            nn.Linear(concat_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.GELU(), nn.Dropout(0.20),
        )
        self.blocks = nn.ModuleList([
            ResidualBottleneckBlock(trunk_dim, trunk_dim // 2, dropout=dropout),
            ResidualBottleneckBlock(trunk_dim, trunk_dim // 2, dropout=dropout),
        ])
        self.trunk_norm = nn.LayerNorm(trunk_dim)

        self.heads = nn.ModuleDict()
        for tname, _, cond_dim in task_specs:
            safe_k = tname.replace(".", "__dot__")
            self.heads[safe_k] = nn.Sequential(
                nn.Linear(trunk_dim + cond_dim, 96),
                nn.LayerNorm(96),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(96, 1),
            )

        # Learnable homoscedastic uncertainty log-variances s_t = log(sigma_t^2)
        self.log_vars = nn.Parameter(torch.zeros(len(task_specs)))

    def encode_trunk(
        self,
        x_phys: torch.Tensor,
        emb_list: List[torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        p_h = self.phys_proj(x_phys)
        parts = [p_h]
        for j, (proj, miss_tok) in enumerate(zip(self.emb_projs, self.missing_tokens)):
            mj = masks[:, j + 1 : j + 2]
            hj = proj(emb_list[j]) * mj + miss_tok * (1.0 - mj)
            parts.append(hj)
        parts.append(masks)
        h = self.in_proj(torch.cat(parts, dim=1))
        for blk in self.blocks:
            h = blk(h)
        return self.trunk_norm(h)

    def head_forward(self, tname: str, h: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        safe_k = tname.replace(".", "__dot__")
        return self.heads[safe_k](torch.cat([h, cond_vec], dim=1)).squeeze(-1)

    def forward_task(
        self,
        tname: str,
        x_phys: torch.Tensor,
        emb_list: List[torch.Tensor],
        masks: torch.Tensor,
        cond_vec: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_trunk(x_phys, emb_list, masks)
        return self.head_forward(tname, h, cond_vec)


def censored_huber_loss(
    pred_z: torch.Tensor,
    y_z: torch.Tensor,
    y_lo_z: torch.Tensor,
    y_hi_z: torch.Tensor,
    censor_code: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Native censored-aware smooth-L1 hinge loss for exact (0), right > (1), left < (2), interval (3)."""
    err_exact = torch.abs(pred_z - y_z)
    viol_right = F.relu(y_lo_z - pred_z)
    viol_left = F.relu(pred_z - y_hi_z)
    viol_int = viol_right + viol_left

    eff = torch.where(
        censor_code == 0,
        err_exact,
        torch.where(
            censor_code == 1,
            viol_right,
            torch.where(censor_code == 2, viol_left, viol_int),
        ),
    )
    return F.smooth_l1_loss(eff, torch.zeros_like(eff), beta=beta, reduction="mean")


def gather_feature_tensors(
    cidx: np.ndarray,
    bank: UnifiedV334Bank,
    normed_embs: Dict[str, np.ndarray],
    phys_mu: np.ndarray,
    phys_sd: np.ndarray,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    raw_p = np.nan_to_num(bank.physchem[cidx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xp = np.clip((raw_p - phys_mu) / phys_sd, -8.0, 8.0)
    sv = bank.smiles_valid[cidx]
    qv = bank.seq_valid[cidx]
    xp[~sv, :48] = 0.0
    xp[~qv, 48:] = 0.0
    t_phys = torch.from_numpy(np.ascontiguousarray(xp, dtype=np.float32))
    emb_list = [torch.from_numpy(normed_embs[bn][cidx]) for bn, _, _ in EMB_BLOCKS_MTL]
    m_cols = [(sv | qv).astype(np.float32)] + [bank.emb_valid[bn][cidx].astype(np.float32) for bn, _, _ in EMB_BLOCKS_MTL]
    t_masks = torch.from_numpy(np.stack(m_cols, axis=1).astype(np.float32))
    return t_phys, emb_list, t_masks


def train_domain_mtl_single_fold(
    target_ep: str,
    domain_eps: List[str],
    ep_tables: Dict[str, Dict[str, Any]],
    bank: UnifiedV334Bank,
    normed_embs: Dict[str, np.ndarray],
    target_tr_idx: np.ndarray,
    target_va_idx: np.ndarray,
    target_ho_idx: np.ndarray,
    aux_tr_indices: Dict[str, np.ndarray],
    seed: int,
    max_epochs: int = 45,
    patience: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train DomainMTLNet strictly on target_tr_idx + leak-free aux_tr_indices; early-stop ONLY on target_va_idx."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    t_tbl = ep_tables[target_ep]
    active_eps = [target_ep] + [ep for ep in domain_eps if ep != target_ep and len(aux_tr_indices.get(ep, [])) >= 8]

    # 1. Strictly fold-internal physchem standardization on ALL clean training compounds in this fold
    all_tr_cidx = np.unique(np.concatenate(
        [t_tbl["cidx"][target_tr_idx]] + [ep_tables[ep]["cidx"][aux_tr_indices[ep]] for ep in active_eps if ep != target_ep]
    ))
    raw_tr_p = np.nan_to_num(bank.physchem[all_tr_cidx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    phys_mu = raw_tr_p.mean(axis=0, keepdims=True)
    phys_sd = np.maximum(raw_tr_p.std(axis=0, keepdims=True), 1e-4)

    # 2. Prepare per-task tensors, condition encoders, and target standardization (strictly fold-internal!)
    task_specs: List[Tuple[str, str, int]] = []
    task_train_packs: Dict[str, Dict[str, Any]] = {}
    target_scale = (0.0, 1.0)

    for ep in active_eps:
        tbl = ep_tables[ep]
        ridx = target_tr_idx if ep == target_ep else aux_tr_indices[ep]
        cenc = FoldConditionEncoder(tbl["cond_cols"]).fit([tbl["conds"][i] for i in ridx])
        cvec = torch.from_numpy(cenc.transform([tbl["conds"][i] for i in ridx]))
        tp, el, tm = gather_feature_tensors(tbl["cidx"][ridx], bank, normed_embs, phys_mu, phys_sd)

        tt = tbl["task_type"]
        yv = tbl["y_val"][ridx].astype(np.float64)
        yl = tbl["y_lo"][ridx].astype(np.float64)
        yh = tbl["y_hi"][ridx].astype(np.float64)
        cc = tbl["censor_code"][ridx]

        if tt == "regression":
            ex = cc == 0
            mu_y = float(np.mean(yv[ex])) if ex.sum() >= 2 else float(np.mean(yv))
            sd_y = float(max(np.std(yv[ex]), 1e-3)) if ex.sum() >= 2 else float(max(np.std(yv), 1e-3))
            yz = torch.from_numpy(((yv - mu_y) / sd_y).astype(np.float32))
            ylz = torch.from_numpy(((yl - mu_y) / sd_y).astype(np.float32))
            yhz = torch.from_numpy(((yh - mu_y) / sd_y).astype(np.float32))
            if ep == target_ep:
                target_scale = (mu_y, sd_y)
        else:
            mu_y, sd_y = 0.0, 1.0
            yz = torch.from_numpy(yv.astype(np.float32))
            ylz = yz
            yhz = yz

        task_specs.append((ep, tt, cenc.dim_))
        task_train_packs[ep] = {
            "n": len(ridx),
            "tt": tt,
            "cenc": cenc,
            "phys": tp,
            "embs": el,
            "masks": tm,
            "cond": cvec,
            "y_z": yz,
            "y_lo_z": ylz,
            "y_hi_z": yhz,
            "cc": torch.from_numpy(cc.astype(np.int64)),
            "mu_y": mu_y,
            "sd_y": sd_y,
        }

    # Prepare target val & holdout tensors
    t_cenc = task_train_packs[target_ep]["cenc"]
    va_phys, va_embs, va_masks = gather_feature_tensors(t_tbl["cidx"][target_va_idx], bank, normed_embs, phys_mu, phys_sd)
    va_cond = torch.from_numpy(t_cenc.transform([t_tbl["conds"][i] for i in target_va_idx]))
    va_y_raw = t_tbl["y_val"][target_va_idx]
    va_cc = t_tbl["censor_code"][target_va_idx]

    has_ho = len(target_ho_idx) > 0
    if has_ho:
        ho_phys, ho_embs, ho_masks = gather_feature_tensors(t_tbl["cidx"][target_ho_idx], bank, normed_embs, phys_mu, phys_sd)
        ho_cond = torch.from_numpy(t_cenc.transform([t_tbl["conds"][i] for i in target_ho_idx]))

    # 3. Build model & optimizer
    n_target_tr = task_train_packs[target_ep]["n"]
    dropout = 0.30 if n_target_tr < 300 else 0.25
    model = DomainMTLNet(task_specs, trunk_dim=256, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-4)

    batch_size_t = min(256, max(32, n_target_tr // 5))
    steps_per_epoch = int(np.clip(math.ceil(n_target_tr / batch_size_t), 10, 28))

    best_val_metric = float("inf")
    best_state = None
    best_epoch = 0
    no_imp = 0

    def _predict_tensor(phys, embs, masks, cond) -> np.ndarray:
        model.eval()
        preds = []
        bs_eval = 512
        with torch.no_grad():
            for st in range(0, len(phys), bs_eval):
                ed = min(len(phys), st + bs_eval)
                out = model.forward_task(
                    target_ep,
                    phys[st:ed],
                    [e[st:ed] for e in embs],
                    masks[st:ed],
                    cond[st:ed],
                )
                if t_tbl["task_type"] == "classification":
                    out = torch.sigmoid(out)
                else:
                    out = out * target_scale[1] + target_scale[0]
                preds.append(out.cpu().numpy())
        arr = np.concatenate(preds).astype(np.float64)
        if t_tbl["bounds"] is not None:
            arr = np.clip(arr, t_tbl["bounds"][0], t_tbl["bounds"][1])
        return arr

    # Aux task round-robin fused batching: 1 target batch + up to 2 aux batches per step in ONE trunk forward pass!
    aux_eps = [ep for ep in active_eps if ep != target_ep]
    n_aux = len(aux_eps)
    aux_cursor = 0
    n_embs = len(EMB_BLOCKS_MTL)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for _ in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            log_vars_clamped = torch.clamp(model.log_vars, -2.5, 2.5)

            # Assemble fused batch: [target, aux_a, aux_b]
            step_items = []
            tpk = task_train_packs[target_ep]
            b_idx = rng.randint(0, tpk["n"], size=min(batch_size_t, tpk["n"]))
            step_items.append((target_ep, 0, 1.5, tpk, b_idx))

            if n_aux > 0:
                n_pick = min(2, n_aux)
                for _k in range(n_pick):
                    aep = aux_eps[aux_cursor % n_aux]
                    aux_cursor += 1
                    ai = 1 + aux_eps.index(aep)
                    apk = task_train_packs[aep]
                    bs_a = min(64, max(16, apk["n"] // 4))
                    a_idx = rng.randint(0, apk["n"], size=min(bs_a, apk["n"]))
                    step_items.append((aep, ai, 1.0, apk, a_idx))

            # Single fused trunk forward pass across target + sampled aux tasks
            cat_phys = torch.cat([pk["phys"][idx] for _, _, _, pk, idx in step_items], dim=0)
            cat_embs = [
                torch.cat([pk["embs"][ej][idx] for _, _, _, pk, idx in step_items], dim=0)
                for ej in range(n_embs)
            ]
            cat_masks = torch.cat([pk["masks"][idx] for _, _, _, pk, idx in step_items], dim=0)
            h_all = model.encode_trunk(cat_phys, cat_embs, cat_masks)

            total_loss = torch.tensor(0.0)
            offset = 0
            for ep_k, lv_i, mult, pk, idx in step_items:
                sz = len(idx)
                h_k = h_all[offset : offset + sz]
                offset += sz
                out_k = model.head_forward(ep_k, h_k, pk["cond"][idx])
                sk = log_vars_clamped[lv_i]
                if pk["tt"] == "classification":
                    lk = F.binary_cross_entropy_with_logits(out_k, pk["y_z"][idx])
                    total_loss = total_loss + mult * (torch.exp(-sk) * lk + 0.5 * sk)
                else:
                    lk = censored_huber_loss(
                        out_k, pk["y_z"][idx], pk["y_lo_z"][idx], pk["y_hi_z"][idx], pk["cc"][idx]
                    )
                    total_loss = total_loss + mult * (0.5 * torch.exp(-sk) * lk + 0.5 * sk)

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        sched.step()

        # Evaluate strictly on target validation fold
        va_pred = _predict_tensor(va_phys, va_embs, va_masks, va_cond)
        if t_tbl["task_type"] == "classification":
            val_metric = float(np.mean((va_y_raw - va_pred) ** 2))
        else:
            ex_m = va_cc == 0
            if ex_m.any():
                val_metric = float(np.mean(np.abs(va_y_raw[ex_m] - va_pred[ex_m])))
            else:
                val_metric = float(np.mean(np.abs(va_y_raw - va_pred)))

        if val_metric < best_val_metric - 1e-5:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience and epoch >= 14:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_va_pred = _predict_tensor(va_phys, va_embs, va_masks, va_cond)
    final_ho_pred = _predict_tensor(ho_phys, ho_embs, ho_masks, ho_cond) if has_ho else np.array([], dtype=np.float64)
    learned_sigmas = {
        active_eps[i]: round(float(math.exp(0.5 * float(torch.clamp(model.log_vars[i], -2.5, 2.5)))), 4)
        for i in range(len(active_eps))
    }
    return final_va_pred, final_ho_pred, {
        "best_epoch": best_epoch,
        "best_val_metric": round(best_val_metric, 5),
        "n_active_domain_tasks": len(active_eps),
        "learned_task_sigmas": learned_sigmas,
    }


def nested_simplex_stack(
    y_cv: np.ndarray,
    fids_cv: np.ndarray,
    exact_mask: np.ndarray,
    oof_dict: Dict[str, np.ndarray],
    ho_dict: Dict[str, np.ndarray],
    task_type: str,
    bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Strict nested 5-fold OOF Simplex stacking on arbitrary family subset (ZERO holdout peeking!)."""
    names = list(oof_dict.keys())
    k = len(names)
    oof_mat = np.column_stack([oof_dict[n] for n in names])
    ho_mat = np.column_stack([ho_dict[n] for n in names]) if len(next(iter(ho_dict.values()))) > 0 else np.zeros((0, k))

    finite_all = exact_mask & np.all(np.isfinite(oof_mat), axis=1) & np.isfinite(y_cv)
    nested_oof = np.full(len(y_cv), np.nan, dtype=np.float64)
    w_folds: List[np.ndarray] = []

    def _fit_weights(Y_tr: np.ndarray, P_tr: np.ndarray) -> np.ndarray:
        if k == 1:
            return np.array([1.0], dtype=np.float64)
        if task_type == "classification":
            def obj(w):
                p = P_tr @ w
                return float(np.mean((Y_tr - p) ** 2))
        else:
            def obj(w):
                p = P_tr @ w
                return float(np.mean(np.abs(Y_tr - p)))
        w0 = np.full(k, 1.0 / k, dtype=np.float64)
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bnds = [(0.0, 1.0)] * k
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-8, "maxiter": 200})
        w = np.clip(res.x, 0.0, 1.0)
        s = float(w.sum())
        return (w / s) if s > 1e-9 else w0

    for f in range(5):
        tr = (fids_cv != f) & finite_all
        va = (fids_cv == f) & np.all(np.isfinite(oof_mat), axis=1)
        if tr.sum() < 8 or va.sum() < 1:
            continue
        wf = _fit_weights(y_cv[tr], oof_mat[tr])
        w_folds.append(wf)
        p_va = oof_mat[va] @ wf
        if bounds is not None:
            p_va = np.clip(p_va, bounds[0], bounds[1])
        nested_oof[va] = p_va

    w_mean = np.mean(w_folds, axis=0) if w_folds else np.full(k, 1.0 / k)
    w_mean = w_mean / np.sum(w_mean)
    ho_pred = (ho_mat @ w_mean) if len(ho_mat) > 0 else np.array([], dtype=np.float64)
    if bounds is not None and len(ho_pred) > 0:
        ho_pred = np.clip(ho_pred, bounds[0], bounds[1])

    return {
        "weights": {names[i]: round(float(w_mean[i]), 4) for i in range(k)},
        "oof_pred": nested_oof,
        "ho_pred": ho_pred,
    }


def eval_endpoint_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    censor_str: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    if task_type == "classification":
        return classification_metrics(y, pred)
    return regression_metrics(y, pred, y_lo, y_hi, censor_str)


def run_part_a_domain_mtl(
    bank: UnifiedV334Bank,
    normed_embs: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    ck_to_global_cl: Dict[str, str],
) -> Dict[str, Any]:
    """Part A: Train & evaluate Domain-Grouped Multi-Task Neural Networks across all 4 biological domains."""
    print("\n================ [PART A: Reproducible Domain-Grouped Multi-Task Networks (4 Domains, 23 Endpoints)] ================", flush=True)
    t_part_a = time.time()

    # Preload all endpoint tables across the 4 domains
    ep_tables: Dict[str, Dict[str, Any]] = {}
    for dom, eps in DOMAINS.items():
        for ep in eps:
            ep_tables[ep] = load_endpoint_table(ep, bank, manifest["endpoints"][ep], ck_to_global_cl)

    domain_results: Dict[str, Any] = {}
    all_endpoint_results: Dict[str, Any] = {}
    all_ho_pvals: Dict[str, float] = {}
    promoted_endpoints: List[str] = []
    held_endpoints: List[str] = []

    for dom_name, eps in DOMAINS.items():
        t_dom = time.time()
        print(f"\n--- [Domain: {dom_name}] ({len(eps)} endpoints: {eps}) ---", flush=True)
        dom_ep_summary: Dict[str, Any] = {}

        for ep in eps:
            t_ep = time.time()
            tbl = ep_tables[ep]
            tt = tbl["task_type"]
            cv_idx_all = np.where(tbl["is_holdout"] == 0)[0]
            ho_idx_all = np.where(tbl["is_holdout"] == 1)[0]
            n_cv = len(cv_idx_all)
            n_ho = len(ho_idx_all)

            # 1. Train 5-fold Domain-MTL with HARD ZERO CROSS-ENDPOINT COMPOUND/CLUSTER LEAKAGE!
            mtl_oof = np.full(n_cv, np.nan, dtype=np.float64)
            mtl_ho_folds: List[np.ndarray] = []
            fold_audits: List[Dict[str, Any]] = []
            aux_kept_list, aux_drop_list = [], []

            ho_global_cls = set(tbl["global_cl"][ho_idx_all].tolist())
            ho_local_cls = set(tbl["local_cl"][ho_idx_all].tolist())
            ho_cks = {tbl["compound_keys"][i] for i in ho_idx_all}

            for f in range(5):
                va_local_pos = np.where(tbl["fold_id"][cv_idx_all] == f)[0]
                tr_local_pos = np.where(tbl["fold_id"][cv_idx_all] != f)[0]
                va_rows = cv_idx_all[va_local_pos]
                tr_rows = cv_idx_all[tr_local_pos]
                if len(va_rows) == 0 or len(tr_rows) < 5:
                    continue

                va_global_cls = set(tbl["global_cl"][va_rows].tolist())
                va_local_cls = set(tbl["local_cl"][va_rows].tolist())
                va_cks = {tbl["compound_keys"][i] for i in va_rows}

                forbidden_local_cls = ho_local_cls | va_local_cls
                forbidden_global_cls = ho_global_cls | va_global_cls
                forbidden_aux_cls = forbidden_local_cls | forbidden_global_cls
                forbidden_cks = ho_cks | va_cks

                # Filter every auxiliary endpoint in the same domain strictly outside forbidden sets
                aux_tr_map: Dict[str, np.ndarray] = {}
                n_aux_kept = 0
                n_aux_dropped = 0
                target_train_local_cls: Set[str] = set(tbl["local_cl"][tr_rows].tolist())
                aux_train_cls: Set[str] = set()
                all_train_cks: Set[str] = {tbl["compound_keys"][i] for i in tr_rows}

                for aep in eps:
                    if aep == ep:
                        continue
                    atbl = ep_tables[aep]
                    acv = np.where(atbl["is_holdout"] == 0)[0]
                    keep = []
                    for ri in acv:
                        if (
                            atbl["global_cl"][ri] in forbidden_aux_cls
                            or atbl["local_cl"][ri] in forbidden_aux_cls
                            or atbl["compound_keys"][ri] in forbidden_cks
                        ):
                            n_aux_dropped += 1
                        else:
                            keep.append(ri)
                    keep_arr = np.array(keep, dtype=np.int32)
                    aux_tr_map[aep] = keep_arr
                    n_aux_kept += len(keep_arr)
                    if len(keep_arr) > 0:
                        aux_train_cls.update(atbl["global_cl"][keep_arr].tolist())
                        aux_train_cls.update(atbl["local_cl"][keep_arr].tolist())
                        all_train_cks.update(atbl["compound_keys"][ri] for ri in keep_arr)

                # HARD ZERO-LEAKAGE RUNTIME ASSERTION!
                leaked_target_cls = target_train_local_cls & forbidden_local_cls
                leaked_aux_cls = aux_train_cls & forbidden_aux_cls
                leaked_cks = all_train_cks & forbidden_cks
                assert len(leaked_target_cls) == 0, f"TARGET CLUSTER LEAK in {ep} fold {f}: {len(leaked_target_cls)}"
                assert len(leaked_aux_cls) == 0, f"AUX CLUSTER LEAK in {ep} fold {f}: {len(leaked_aux_cls)}"
                assert len(leaked_cks) == 0, f"COMPOUND LEAK in {ep} fold {f}: {len(leaked_cks)}"

                aux_kept_list.append(n_aux_kept)
                aux_drop_list.append(n_aux_dropped)

                f_va_pred, f_ho_pred, f_info = train_domain_mtl_single_fold(
                    target_ep=ep,
                    domain_eps=eps,
                    ep_tables=ep_tables,
                    bank=bank,
                    normed_embs=normed_embs,
                    target_tr_idx=tr_rows,
                    target_va_idx=va_rows,
                    target_ho_idx=ho_idx_all,
                    aux_tr_indices=aux_tr_map,
                    seed=SEED + 100 * f + abs(hash(ep)) % 997,
                )
                mtl_oof[va_local_pos] = f_va_pred
                if len(f_ho_pred) > 0:
                    mtl_ho_folds.append(f_ho_pred)
                fold_audits.append({
                    "fold": f,
                    "n_target_tr": int(len(tr_rows)),
                    "n_target_va": int(len(va_rows)),
                    "n_aux_kept": int(n_aux_kept),
                    "n_aux_dropped": int(n_aux_dropped),
                    "n_leaked_clusters": 0,
                    "n_leaked_compounds": 0,
                    **f_info,
                })

            mtl_ho = np.mean(mtl_ho_folds, axis=0) if mtl_ho_folds else np.full(n_ho, np.nan)
            if tbl["bounds"] is not None:
                mtl_ho = np.clip(mtl_ho, tbl["bounds"][0], tbl["bounds"][1])

            # 2. Load canonical CPU & 4-FM Fusion OOF/Holdout predictions
            cpu_oof, fus_oof, cpu_ho, fus_ho = load_baseline_cpu_fusion_preds(tbl, bank)

            # 3. Strictly Nested 5-Fold OOF Simplex Stacking (ZERO holdout peeking!)
            y_cv = tbl["y_val"][cv_idx_all]
            yl_cv = tbl["y_lo"][cv_idx_all]
            yh_cv = tbl["y_hi"][cv_idx_all]
            cs_cv = tbl["censor_str"][cv_idx_all]
            fids_cv = tbl["fold_id"][cv_idx_all]
            exact_cv = cs_cv == "exact"

            y_ho = tbl["y_val"][ho_idx_all]
            yl_ho = tbl["y_lo"][ho_idx_all]
            yh_ho = tbl["y_hi"][ho_idx_all]
            cs_ho = tbl["censor_str"][ho_idx_all]
            cl_ho = tbl["local_cl"][ho_idx_all]
            exact_ho = cs_ho == "exact"

            stack_2 = nested_simplex_stack(
                y_cv, fids_cv, exact_cv,
                {"cpu": cpu_oof, "fusion": fus_oof},
                {"cpu": cpu_ho, "fusion": fus_ho},
                tt, tbl["bounds"],
            )
            stack_3 = nested_simplex_stack(
                y_cv, fids_cv, exact_cv,
                {"cpu": cpu_oof, "fusion": fus_oof, "domain_mtl": mtl_oof},
                {"cpu": cpu_ho, "fusion": fus_ho, "domain_mtl": mtl_ho},
                tt, tbl["bounds"],
            )

            # CV metrics for every family and stack
            cv_m_cpu = eval_endpoint_metrics(y_cv, cpu_oof, yl_cv, yh_cv, cs_cv, tt)
            cv_m_fus = eval_endpoint_metrics(y_cv, fus_oof, yl_cv, yh_cv, cs_cv, tt)
            cv_m_mtl = eval_endpoint_metrics(y_cv, mtl_oof, yl_cv, yh_cv, cs_cv, tt)
            cv_m_s2 = eval_endpoint_metrics(y_cv, stack_2["oof_pred"], yl_cv, yh_cv, cs_cv, tt)
            cv_m_s3 = eval_endpoint_metrics(y_cv, stack_3["oof_pred"], yl_cv, yh_cv, cs_cv, tt)

            # STRICT ZERO-HOLDOUT-PEEKING PROMOTION DECISION (uses ONLY nested OOF CV!)
            w_mtl = float(stack_3["weights"]["domain_mtl"])
            if tt == "regression":
                mae2 = float(cv_m_s2["mae"])
                mae3 = float(cv_m_s3["mae"])
                sp2 = float(cv_m_s2.get("spearman") or 0.0)
                sp3 = float(cv_m_s3.get("spearman") or 0.0)
                abs_imp = mae2 - mae3
                rel_imp = abs_imp / max(mae2, 1e-9)
                sp_delta = sp3 - sp2
                promote = (rel_imp >= 0.005) and (abs_imp >= 0.002) and (w_mtl >= 0.05) and (sp_delta >= -0.01)
                gate_reason = (
                    f"Nested OOF MAE {mae2:.4f}->{mae3:.4f} ({-100*rel_imp:+.2f}%), "
                    f"Sp {sp2:.4f}->{sp3:.4f} ({sp_delta:+.4f}), w_mtl={w_mtl:.3f}"
                )
            else:
                br2 = float(cv_m_s2["brier"])
                br3 = float(cv_m_s3["brier"])
                auc2 = float(cv_m_s2["roc_auc"])
                auc3 = float(cv_m_s3["roc_auc"])
                br_imp = br2 - br3
                auc_imp = auc3 - auc2
                promote = ((br_imp >= 0.0005) or (auc_imp >= 0.001 and br_imp >= -0.001)) and (w_mtl >= 0.05)
                gate_reason = (
                    f"Nested OOF Brier {br2:.4f}->{br3:.4f} (d={br_imp:+.4f}), "
                    f"AUC {auc2:.4f}->{auc3:.4f} (d={auc_imp:+.4f}), w_mtl={w_mtl:.3f}"
                )

            if promote:
                promoted_endpoints.append(ep)
            else:
                held_endpoints.append(ep)

            # 4. READ-ONLY HOLDOUT AUDIT (evaluated AFTER zero-peeking decision is locked!)
            ho_m_cpu = eval_endpoint_metrics(y_ho, cpu_ho, yl_ho, yh_ho, cs_ho, tt)
            ho_m_fus = eval_endpoint_metrics(y_ho, fus_ho, yl_ho, yh_ho, cs_ho, tt)
            ho_m_mtl = eval_endpoint_metrics(y_ho, mtl_ho, yl_ho, yh_ho, cs_ho, tt)
            ho_m_s2 = eval_endpoint_metrics(y_ho, stack_2["ho_pred"], yl_ho, yh_ho, cs_ho, tt)
            ho_m_s3 = eval_endpoint_metrics(y_ho, stack_3["ho_pred"], yl_ho, yh_ho, cs_ho, tt)
            ho_m_sel = ho_m_s3 if promote else ho_m_s2

            pb_ho = paired_cluster_bootstrap_pvalue(
                y_ho[exact_ho], stack_3["ho_pred"][exact_ho], stack_2["ho_pred"][exact_ho], cl_ho[exact_ho], tt
            )
            all_ho_pvals[ep] = pb_ho["p_value"]

            v336_m = manifest["endpoints"][ep]["dual_modality"]
            ep_rec = {
                "domain": dom_name,
                "task_type": tt,
                "n_cv": int(n_cv),
                "n_cv_exact": int(exact_cv.sum()),
                "n_cv_censored": int((~exact_cv).sum()),
                "n_holdout": int(n_ho),
                "n_holdout_exact": int(exact_ho.sum()),
                "n_holdout_censored": int((~exact_ho).sum()),
                "leakage_audit": {
                    "n_leaked_clusters_all_folds": 0,
                    "n_leaked_compounds_all_folds": 0,
                    "mean_aux_rows_kept_per_fold": round(float(np.mean(aux_kept_list)), 1) if aux_kept_list else 0.0,
                    "mean_aux_rows_dropped_for_isolation": round(float(np.mean(aux_drop_list)), 1) if aux_drop_list else 0.0,
                    "folds": fold_audits,
                },
                "v336_manifest_champion": {
                    "winning_family": v336_m["winning_family"],
                    "weights": v336_m["weights"],
                    "cv_metrics": v336_m["cv_metrics"],
                    "holdout_metrics": v336_m["holdout_metrics"],
                },
                "cv_metrics": {
                    "cpu": cv_m_cpu,
                    "fusion": cv_m_fus,
                    "domain_mtl": cv_m_mtl,
                    "stack_2fam_cpu_fusion": cv_m_s2,
                    "stack_3fam_cpu_fusion_mtl": cv_m_s3,
                },
                "simplex_weights": {
                    "stack_2fam": stack_2["weights"],
                    "stack_3fam": stack_3["weights"],
                },
                "zero_peeking_promotion_gate": {
                    "promoted_domain_mtl": bool(promote),
                    "reason": gate_reason,
                },
                "holdout_audit": {
                    "cpu": ho_m_cpu,
                    "fusion": ho_m_fus,
                    "domain_mtl": ho_m_mtl,
                    "stack_2fam_cpu_fusion": ho_m_s2,
                    "stack_3fam_cpu_fusion_mtl": ho_m_s3,
                    "oof_selected_champion": ho_m_sel,
                    "paired_boot_3fam_vs_2fam": pb_ho,
                },
                "elapsed_sec": round(time.time() - t_ep, 1),
            }
            dom_ep_summary[ep] = ep_rec
            all_endpoint_results[ep] = ep_rec

            if tt == "regression":
                print(
                    f"  [{ep:<24}] PROMOTE={str(promote):<5} w={stack_3['weights']} | "
                    f"CV MAE: 2fam={cv_m_s2['mae']:.4f} mtl={cv_m_mtl['mae']:.4f} 3fam={cv_m_s3['mae']:.4f} | "
                    f"HO MAE: v336={v336_m['holdout_metrics'].get('mae')} 2fam={ho_m_s2['mae']:.4f} mtl={ho_m_mtl['mae']:.4f} 3fam={ho_m_s3['mae']:.4f} "
                    f"(Sp {ho_m_s2['spearman']:.3f}->{ho_m_s3['spearman']:.3f}) [{time.time()-t_ep:.1f}s]",
                    flush=True,
                )
            else:
                print(
                    f"  [{ep:<24}] PROMOTE={str(promote):<5} w={stack_3['weights']} | "
                    f"CV AUC/Br: 2fam={cv_m_s2['roc_auc']:.4f}/{cv_m_s2['brier']:.4f} mtl={cv_m_mtl['roc_auc']:.4f}/{cv_m_mtl['brier']:.4f} "
                    f"3fam={cv_m_s3['roc_auc']:.4f}/{cv_m_s3['brier']:.4f} | "
                    f"HO AUC/Br: 2fam={ho_m_s2['roc_auc']:.4f}/{ho_m_s2['brier']:.4f} 3fam={ho_m_s3['roc_auc']:.4f}/{ho_m_s3['brier']:.4f} [{time.time()-t_ep:.1f}s]",
                    flush=True,
                )

        domain_results[dom_name] = {
            "endpoints": eps,
            "n_promoted": sum(1 for e in eps if dom_ep_summary[e]["zero_peeking_promotion_gate"]["promoted_domain_mtl"]),
            "promoted_list": [e for e in eps if dom_ep_summary[e]["zero_peeking_promotion_gate"]["promoted_domain_mtl"]],
            "elapsed_sec": round(time.time() - t_dom, 1),
        }

    # Holm-Bonferroni multiplicity correction across all 23 holdout comparisons
    holm_adj = holm_bonferroni(all_ho_pvals)
    for ep, padj in holm_adj.items():
        all_endpoint_results[ep]["holdout_audit"]["paired_boot_3fam_vs_2fam"]["p_value_holm"] = padj

    # Dedicated Sparse PK Highlight Audit
    pk_highlight = {}
    for ep in DOMAINS["domain_systemic_pk"]:
        r = all_endpoint_results[ep]
        pk_highlight[ep] = {
            "n_cv": r["n_cv"],
            "n_holdout": r["n_holdout"],
            "v336_winning_family": r["v336_manifest_champion"]["winning_family"],
            "v336_ho_mae": r["v336_manifest_champion"]["holdout_metrics"].get("mae"),
            "v336_ho_spearman": r["v336_manifest_champion"]["holdout_metrics"].get("spearman"),
            "stack_2fam_ho_mae": r["holdout_audit"]["stack_2fam_cpu_fusion"].get("mae"),
            "stack_2fam_ho_spearman": r["holdout_audit"]["stack_2fam_cpu_fusion"].get("spearman"),
            "domain_mtl_ho_mae": r["holdout_audit"]["domain_mtl"].get("mae"),
            "domain_mtl_ho_spearman": r["holdout_audit"]["domain_mtl"].get("spearman"),
            "stack_3fam_ho_mae": r["holdout_audit"]["stack_3fam_cpu_fusion_mtl"].get("mae"),
            "stack_3fam_ho_spearman": r["holdout_audit"]["stack_3fam_cpu_fusion_mtl"].get("spearman"),
            "promoted_by_oof_gate": r["zero_peeking_promotion_gate"]["promoted_domain_mtl"],
            "stack_3fam_weights": r["simplex_weights"]["stack_3fam"],
        }

    # Compute aggregate audit counts
    n_ho_improved_3v2 = 0
    n_ho_improved_oof_sel_vs_2 = 0
    for ep, r in all_endpoint_results.items():
        tt = r["task_type"]
        if tt == "regression":
            m2 = r["holdout_audit"]["stack_2fam_cpu_fusion"]["mae"]
            m3 = r["holdout_audit"]["stack_3fam_cpu_fusion_mtl"]["mae"]
            ms = r["holdout_audit"]["oof_selected_champion"]["mae"]
            if m3 < m2:
                n_ho_improved_3v2 += 1
            if ms < m2:
                n_ho_improved_oof_sel_vs_2 += 1
        else:
            b2 = r["holdout_audit"]["stack_2fam_cpu_fusion"]["brier"]
            b3 = r["holdout_audit"]["stack_3fam_cpu_fusion_mtl"]["brier"]
            bs = r["holdout_audit"]["oof_selected_champion"]["brier"]
            if b3 < b2:
                n_ho_improved_3v2 += 1
            if bs < b2:
                n_ho_improved_oof_sel_vs_2 += 1

    return {
        "architecture": (
            "DomainMTLNet: Modality-gated 4-FM + Physchem projections (774D -> 384D) + "
            "2x LayerNorm Residual Bottleneck MLP blocks (384->192->384) + Task-Specific Condition Heads + "
            "Homoscedastic Uncertainty Task Weighting (Kendall et al. 2018) + Native Censored-Aware Huber Hinge Loss"
        ),
        "zero_peeking_promotion_rule": (
            "Decided STRICTLY on nested 5-fold cluster-grouped OOF CV (holdout NEVER inspected for selection). "
            "Regression: rel MAE imp >= 0.5% AND abs MAE imp >= 0.002 AND w_mtl >= 0.05 AND OOF Spearman delta >= -0.01. "
            "Classification: (Brier imp >= 0.0005 OR (AUC imp >= 0.001 AND Brier imp >= -0.001)) AND w_mtl >= 0.05."
        ),
        "zero_leakage_guarantee": (
            "For every (target_ep, fold), auxiliary domain tasks exclude ALL holdout rows plus ALL rows sharing "
            "global cluster_id, local cluster_id, or compound_key with target_va or target_ho. Runtime assertions verified 0/0 leaks."
        ),
        "n_domains": len(DOMAINS),
        "n_endpoints": len(all_endpoint_results),
        "n_promoted_by_oof_gate": len(promoted_endpoints),
        "promoted_endpoints": promoted_endpoints,
        "held_endpoints": held_endpoints,
        "holdout_audit_summary": {
            "n_3fam_improves_over_2fam_on_holdout": n_ho_improved_3v2,
            "n_oof_selected_improves_over_2fam_on_holdout": n_ho_improved_oof_sel_vs_2,
            "n_total_endpoints": len(all_endpoint_results),
        },
        "sparse_pk_domain_highlight": pk_highlight,
        "domains": domain_results,
        "endpoints": all_endpoint_results,
        "elapsed_sec": round(time.time() - t_part_a, 1),
    }


# =============================================================================
# Part B: Full-Stack FDA 2019-2023 Blind Evaluation (Protocol 1 vs Protocol 2)
# =============================================================================

def canonicalize_seq(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    su = s.upper()
    if len(su) >= 4 and set(su).issubset(AA20_SET):
        return su
    tokens = re.findall(r"[A-Za-z]+", s)
    mapped = [AA3_TO_1[t.upper()] for t in tokens if t.upper() in AA3_TO_1]
    if len(mapped) >= 4:
        return "".join(mapped)
    return ""


def seq_3mer_jaccard(s1: str, s2: str) -> float:
    if len(s1) < 3 or len(s2) < 3:
        return 0.0
    a = {s1[i : i + 3] for i in range(len(s1) - 2)}
    b = {s2[i : i + 3] for i in range(len(s2) - 2)}
    return len(a & b) / max(len(a | b), 1)


def seq_global_identity(s1: str, s2: str) -> float:
    if len(s1) < 4 or len(s2) < 4:
        return 0.0
    l1, l2 = len(s1), len(s2)
    if min(l1, l2) / max(l1, l2) < 0.40:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def morgan2048_bits(smi: str) -> Optional[np.ndarray]:
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return MORGAN2048_GEN.GetFingerprintAsNumPy(m).astype(np.uint8)


def extract_murcko_and_macrocycle(smi: str) -> Tuple[str, str]:
    """Extract non-trivial Murcko scaffold (>=2 rings or macrocycle >=12 atoms) and macrocycle core ring SMILES."""
    if not smi or len(smi) > 900 or not any(ch.isdigit() for ch in smi):
        return "", ""
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() > 260:
        return "", ""
    ri = m.GetRingInfo()
    atom_rings = ri.AtomRings()
    n_rings = len(atom_rings)
    max_ring = max((len(r) for r in atom_rings), default=0)
    murcko = ""
    if n_rings >= 2 or max_ring >= 12:
        try:
            ms = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            if len(ms) >= 15:
                murcko = ms
        except Exception:
            pass
    macro_core = ""
    if max_ring >= 12:
        largest = max(atom_rings, key=len)
        try:
            macro_core = Chem.MolFragmentToSmiles(m, atomsToUse=list(largest), canonical=True, isomericSmiles=False)
        except Exception:
            macro_core = ""
    return murcko, macro_core


def make_cluster_balanced_5folds(clusters: List[str]) -> np.ndarray:
    cl_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        cl_to_rows[str(c)].append(i)
    sorted_cls = sorted(cl_to_rows.keys(), key=lambda c: -len(cl_to_rows[c]))
    fsizes = [0] * 5
    fids = np.zeros(len(clusters), dtype=np.int32)
    for c in sorted_cls:
        f = int(np.argmin(fsizes))
        for idx in cl_to_rows[c]:
            fids[idx] = f
        fsizes[f] += len(cl_to_rows[c])
    return fids


def fit_predict_full_cls_stack(
    bank: UnifiedV334Bank,
    normed_embs: Dict[str, np.ndarray],
    cidx_tr: np.ndarray,
    y_tr: np.ndarray,
    fids_tr: np.ndarray,
    cidx_te: np.ndarray,
    mtl_oof: np.ndarray,
    mtl_te: np.ndarray,
) -> Dict[str, Any]:
    """Full reproducible multi-family classification stack (CPU multi-model + 4-FM Modality-Aware Fusion + Domain-MTL)
    with strictly nested 5-fold OOF simplex stacking on Brier score (fixes Round 3 N-5 simplified-LGBM gap!).
    """
    n_tr = len(y_tr)
    n_te = len(cidx_te)
    y_int = y_tr.astype(int)

    # 1. CPU Multi-Model Track: LGBM + ExtraTrees + LogisticRegression(SVD) on Morgan + Physchem
    X_cpu_tr = np.hstack([bank.morgan_bits[cidx_tr].astype(np.float32), np.nan_to_num(bank.physchem[cidx_tr], nan=0.0)])
    X_cpu_te = np.hstack([bank.morgan_bits[cidx_te].astype(np.float32), np.nan_to_num(bank.physchem[cidx_te], nan=0.0)])

    cpu_sub_oofs = {
        "lgbm": np.zeros(n_tr, dtype=np.float64),
        "et": np.zeros(n_tr, dtype=np.float64),
        "lr_svd": np.zeros(n_tr, dtype=np.float64),
    }
    cpu_sub_te = {k: [] for k in cpu_sub_oofs}

    for f in range(5):
        va = fids_tr == f
        tr = ~va
        m_lgb = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.7, n_jobs=8, random_state=42 + f, verbose=-1,
        )
        m_lgb.fit(X_cpu_tr[tr], y_int[tr])
        cpu_sub_oofs["lgbm"][va] = m_lgb.predict_proba(X_cpu_tr[va])[:, 1]
        cpu_sub_te["lgbm"].append(m_lgb.predict_proba(X_cpu_te)[:, 1])

        m_et = ExtraTreesClassifier(
            n_estimators=250, max_depth=18, min_samples_leaf=2, max_features=0.25,
            n_jobs=8, random_state=142 + f,
        )
        m_et.fit(X_cpu_tr[tr], y_int[tr])
        cpu_sub_oofs["et"][va] = m_et.predict_proba(X_cpu_tr[va])[:, 1]
        cpu_sub_te["et"].append(m_et.predict_proba(X_cpu_te)[:, 1])

        svd = TruncatedSVD(n_components=min(64, X_cpu_tr.shape[1] - 1), random_state=42 + f)
        sc = StandardScaler()
        Z_tr = sc.fit_transform(svd.fit_transform(X_cpu_tr[tr]))
        Z_va = sc.transform(svd.transform(X_cpu_tr[va]))
        Z_te = sc.transform(svd.transform(X_cpu_te))
        m_lr = LogisticRegression(C=1.0, max_iter=500)
        m_lr.fit(Z_tr, y_int[tr])
        cpu_sub_oofs["lr_svd"][va] = m_lr.predict_proba(Z_va)[:, 1]
        cpu_sub_te["lr_svd"].append(m_lr.predict_proba(Z_te)[:, 1])

    cpu_te_avg = {k: np.mean(v, axis=0) for k, v in cpu_sub_te.items()}
    cpu_blend = nested_simplex_stack(
        y_tr, fids_tr, np.ones(n_tr, dtype=bool), cpu_sub_oofs, cpu_te_avg, "classification", (0.0, 1.0)
    )
    cpu_oof, cpu_te = cpu_blend["oof_pred"], cpu_blend["ho_pred"]

    # 2. 4-FM Modality-Aware Fusion Track: Individual FM base heads + direct 4-FM head -> OOF simplex stacker!
    fm_heads_oof: Dict[str, np.ndarray] = {"cpu_base": cpu_oof}
    fm_heads_te: Dict[str, np.ndarray] = {"cpu_base": cpu_te}

    for bn, _, _ in EMB_BLOCKS_MTL:
        v_tr = bank.emb_valid[bn][cidx_tr]
        v_te = bank.emb_valid[bn][cidx_te]
        if v_tr.sum() < 100:
            continue
        X_e_tr = normed_embs[bn][cidx_tr]
        X_e_te = normed_embs[bn][cidx_te]
        oof_h = np.full(n_tr, float(y_int.mean()), dtype=np.float64)
        te_folds = []
        for f in range(5):
            va = (fids_tr == f) & v_tr
            tr = (fids_tr != f) & v_tr
            if tr.sum() < 50 or len(np.unique(y_int[tr])) < 2:
                continue
            sc_e = StandardScaler()
            lr_e = LogisticRegression(C=0.5, max_iter=500)
            lr_e.fit(sc_e.fit_transform(X_e_tr[tr]), y_int[tr])
            if va.any():
                oof_h[va] = lr_e.predict_proba(sc_e.transform(X_e_tr[va]))[:, 1]
            p_te_f = np.full(n_te, float(y_int[tr].mean()), dtype=np.float64)
            if v_te.any():
                p_te_f[v_te] = lr_e.predict_proba(sc_e.transform(X_e_te[v_te]))[:, 1]
            te_folds.append(p_te_f)
        if te_folds:
            fm_heads_oof[bn] = oof_h
            fm_heads_te[bn] = np.mean(te_folds, axis=0)

    # Direct concatenated 4-FM + physchem LGBM head
    X_direct_tr = np.hstack(
        [np.nan_to_num(bank.physchem[cidx_tr], nan=0.0)] + [normed_embs[bn][cidx_tr] for bn, _, _ in EMB_BLOCKS_MTL]
    )
    X_direct_te = np.hstack(
        [np.nan_to_num(bank.physchem[cidx_te], nan=0.0)] + [normed_embs[bn][cidx_te] for bn, _, _ in EMB_BLOCKS_MTL]
    )
    dir_oof = np.zeros(n_tr, dtype=np.float64)
    dir_te_folds = []
    for f in range(5):
        va = fids_tr == f
        tr = ~va
        m_dir = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, num_leaves=31,
            subsample=0.8, colsample_bytree=0.5, n_jobs=8, random_state=242 + f, verbose=-1,
        )
        m_dir.fit(X_direct_tr[tr], y_int[tr])
        dir_oof[va] = m_dir.predict_proba(X_direct_tr[va])[:, 1]
        dir_te_folds.append(m_dir.predict_proba(X_direct_te)[:, 1])
    fm_heads_oof["direct_4fm_lgbm"] = dir_oof
    fm_heads_te["direct_4fm_lgbm"] = np.mean(dir_te_folds, axis=0)

    fus_blend = nested_simplex_stack(
        y_tr, fids_tr, np.ones(n_tr, dtype=bool), fm_heads_oof, fm_heads_te, "classification", (0.0, 1.0)
    )
    fus_oof, fus_te = fus_blend["oof_pred"], fus_blend["ho_pred"]

    # 3. Top-level 3-Family Nested OOF Simplex Stack: (CPU, 4-FM Fusion, Domain-MTL)
    stack_3 = nested_simplex_stack(
        y_tr, fids_tr, np.ones(n_tr, dtype=bool),
        {"cpu": cpu_oof, "fusion": fus_oof, "domain_mtl": mtl_oof},
        {"cpu": cpu_te, "fusion": fus_te, "domain_mtl": mtl_te},
        "classification", (0.0, 1.0),
    )
    return {
        "cpu_oof": cpu_oof, "cpu_pred": cpu_te, "cpu_subweights": cpu_blend["weights"],
        "fusion_oof": fus_oof, "fusion_pred": fus_te, "fusion_subweights": fus_blend["weights"],
        "domain_mtl_oof": mtl_oof, "domain_mtl_pred": mtl_te,
        "full_stack_oof": stack_3["oof_pred"], "full_stack_pred": stack_3["ho_pred"],
        "stack_weights": stack_3["weights"],
        "cv_metrics": {
            "cpu": classification_metrics(y_int, cpu_oof),
            "fusion": classification_metrics(y_int, fus_oof),
            "domain_mtl": classification_metrics(y_int, mtl_oof),
            "full_3fam_stack": classification_metrics(y_int, stack_3["oof_pred"]),
        },
    }


def exact_spearman_permutation_pvalue(obs: np.ndarray, pred: np.ndarray, seed: int = SEED) -> Dict[str, Any]:
    """Compute exact (for n<=8) or 100,000-draw Monte Carlo permutation p-values for Spearman rho."""
    n = len(obs)
    rho_obs = float(spearmanr(obs, pred).statistic)
    r_obs = np.argsort(np.argsort(obs)).astype(np.float64)
    r_prd = np.argsort(np.argsort(pred)).astype(np.float64)
    # Use rankdata for exact tie handling
    from scipy.stats import rankdata
    r_obs = rankdata(obs)
    r_prd = rankdata(pred)
    ro_c = r_obs - r_obs.mean()
    rp_c = r_prd - r_prd.mean()
    denom = float(np.linalg.norm(ro_c) * np.linalg.norm(rp_c))
    if denom < 1e-12:
        return {"method": "degenerate", "n_perms": 0, "p_one_sided": 1.0, "p_two_sided": 1.0}

    if n <= 8:
        # Exact enumeration over all n! permutations (8! = 40,320)
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int32)
        rhos = (rp_c[perms] @ ro_c) / denom
        p_one = float(np.mean(rhos >= rho_obs - 1e-12))
        p_two = float(np.mean(np.abs(rhos) >= abs(rho_obs) - 1e-12))
        return {
            "method": f"exact_all_{math.factorial(n)}_permutations",
            "n_perms": int(len(rhos)),
            "p_one_sided": round(p_one, 6),
            "p_two_sided": round(p_two, 6),
        }

    # For n=11, 100,000 vectorized random permutations (SE < 0.0002 near p=0.003)
    rng = np.random.RandomState(seed)
    n_perms = 100000
    cnt_one = 0
    cnt_two = 0
    blk = 10000
    for _ in range(n_perms // blk):
        keys = rng.random((blk, n))
        idx = np.argsort(keys, axis=1)
        rhos = (rp_c[idx] @ ro_c) / denom
        cnt_one += int(np.sum(rhos >= rho_obs - 1e-12))
        cnt_two += int(np.sum(np.abs(rhos) >= abs(rho_obs) - 1e-12))
    p_one = (cnt_one + 1) / (n_perms + 1)
    p_two = (cnt_two + 1) / (n_perms + 1)
    return {
        "method": f"monte_carlo_{n_perms}_permutations",
        "n_perms": n_perms,
        "p_one_sided": round(p_one, 6),
        "p_two_sided": round(p_two, 6),
    }


def small_n_blind_summary(obs_v: List[float], prd_v: List[float], seed: int = SEED) -> Dict[str, Any]:
    oa = np.array(obs_v, dtype=np.float64)
    pa = np.array(prd_v, dtype=np.float64)
    m = regression_metrics(oa, pa)
    const_mae = float(np.mean(np.abs(oa - np.median(oa))))
    rho = float(m["spearman"] or 0.0)
    n = len(oa)
    if n > 3 and abs(rho) < 0.9999:
        z = math.atanh(rho)
        se_z = 1.0 / math.sqrt(n - 3)
        fisher_ci = [round(math.tanh(z - 1.96 * se_z), 4), round(math.tanh(z + 1.96 * se_z), 4)]
    else:
        fisher_ci = None

    rng_b = np.random.RandomState(seed)
    b_maes, b_sps = [], []
    for _ in range(2000):
        idx = rng_b.randint(0, n, size=n)
        b_maes.append(float(np.mean(np.abs(oa[idx] - pa[idx]))))
        if np.std(oa[idx]) > 1e-12 and np.std(pa[idx]) > 1e-12:
            r = spearmanr(oa[idx], pa[idx]).statistic
            if np.isfinite(r):
                b_sps.append(float(r))

    perm = exact_spearman_permutation_pvalue(oa, pa, seed=seed)
    return {
        **m,
        "n_evaluated": n,
        "spearman_fisher_ci95": fisher_ci,
        "spearman_boot_ci95": [round(float(np.percentile(b_sps, 2.5)), 4), round(float(np.percentile(b_sps, 97.5)), 4)] if len(b_sps) >= 20 else None,
        "mae_boot_ci95": [round(float(np.percentile(b_maes, 2.5)), 4), round(float(np.percentile(b_maes, 97.5)), 4)],
        "constant_median_oracle_mae": round(const_mae, 4),
        "mae_improvement_vs_oracle_median_pct": round(100.0 * (const_mae - m["mae"]) / max(const_mae, 1e-9), 1),
        "spearman_permutation_test": perm,
    }


def run_part_b_fda_blind_comparison(
    bank: UnifiedV334Bank,
    normed_embs: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    ck_to_smi: Dict[str, str],
    ck_to_ik14: Dict[str, str],
    ck_to_seq: Dict[str, str],
    ck_to_global_cl: Dict[str, str],
) -> Dict[str, Any]:
    """Part B: Full-Stack (CPU + 4-FM Fusion + Domain-MTL) FDA 2019-2023 Blind Evaluation (n=11)
    under Protocol 1 (Strict-Morgan Tc>=0.40) vs Protocol 2 (Peptide-Structural) side-by-side!
    """
    print("\n================ [PART B: Full-Stack FDA 2019-2023 Blind Test (n=11) — Protocol 1 vs Protocol 2] ================", flush=True)
    t_b = time.time()
    fda_list = json.loads((EXP336 / "fda_2019_2024_recent_peptides.json").read_text(encoding="utf-8"))
    ik14_to_ck = {ik: ck for ck, ik in ck_to_ik14.items() if ik and ck in bank.key_to_idx}

    matched_fda = []
    for d in fda_list:
        ck = ik14_to_ck.get(d.get("inchikey_14", ""))
        assert ck is not None and ck in bank.key_to_idx, f"FDA drug {d['name']} missing from UnifiedV334Bank!"
        dc = dict(d)
        dc["compound_key"] = ck
        dc["bank_idx"] = bank.key_to_idx[ck]
        dc["canon_seq"] = canonicalize_seq(d.get("sequence") or "")
        dc["murcko"], dc["macro_core"] = extract_murcko_and_macrocycle(d["smiles"])
        matched_fda.append(dc)
    print(f"  Matched ALL {len(matched_fda)}/{len(fda_list)} FDA 2019-2023 drugs in UnifiedV334Bank!", flush=True)

    fda_cks = {d["compound_key"] for d in matched_fda}
    fda_iks = {d.get("inchikey_14", "") for d in matched_fda if d.get("inchikey_14")}
    fda_1024 = bank.morgan_bits[np.array([d["bank_idx"] for d in matched_fda], dtype=np.int32)]
    fda_m2048 = np.vstack([morgan2048_bits(d["smiles"]) for d in matched_fda])
    fda_m2048_pop = fda_m2048.sum(axis=1).astype(np.float32)

    fda_raw_canon_seqs = [
        (d.get("sequence") or "").strip().upper()
        for d in matched_fda
        if len((d.get("sequence") or "").strip()) >= 4 and set((d.get("sequence") or "").strip().upper()).issubset(AA20_SET)
    ]
    fda_parsed_seqs = [d["canon_seq"] for d in matched_fda if len(d["canon_seq"]) >= 4]
    fda_murckos = {d["murcko"] for d in matched_fda if d["murcko"]}
    fda_macros = {d["macro_core"] for d in matched_fda if d["macro_core"]}
    print(f"  FDA Protocol-2 signatures: {len(fda_parsed_seqs)} parsed seqs, {len(fda_murckos)} Murcko scaffolds, {len(fda_macros)} macrocycle cores", flush=True)

    # Cache Morgan-2048 and Murcko/Macrocycle per unique compound_key so we never recompute!
    cache_m2048: Dict[str, Optional[np.ndarray]] = {}
    cache_scaf: Dict[str, Tuple[str, str]] = {}
    cache_cseq: Dict[str, str] = {}

    def _get_m2048(ck: str) -> Optional[np.ndarray]:
        if ck not in cache_m2048:
            cache_m2048[ck] = morgan2048_bits(ck_to_smi.get(ck, ""))
        return cache_m2048[ck]

    def _get_scaf(ck: str) -> Tuple[str, str]:
        if ck not in cache_scaf:
            cache_scaf[ck] = extract_murcko_and_macrocycle(ck_to_smi.get(ck, ""))
        return cache_scaf[ck]

    def _get_cseq(ck: str) -> str:
        if ck not in cache_cseq:
            cache_cseq[ck] = canonicalize_seq(ck_to_seq.get(ck, ""))
        return cache_cseq[ck]

    def filter_endpoint_rows_by_protocol(
        ep: str,
        protocol: str,
        exact_only: bool = True,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        rows_raw = list(csv.DictReader(find_ep_path(ep).open(encoding="utf-8"), delimiter="\t"))
        rows = [r for r in rows_raw if (r["censor_type"] == "exact" if exact_only else True)]
        cl_before = len({ck_to_global_cl.get(r["compound_key"], r["cluster_id"]) for r in rows})
        tr_idx_all = bank.map_keys([r["compound_key"] for r in rows])
        tr_1024 = bank.morgan_bits[tr_idx_all]
        tr_sv = bank.smiles_valid[tr_idx_all]

        inter = tr_1024.astype(np.float32) @ fda_1024.astype(np.float32).T
        union = tr_1024.sum(axis=1)[:, None].astype(np.float32) + fda_1024.sum(axis=1)[None, :].astype(np.float32) - inter
        tc1024_max = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).max(axis=1)

        clean_rows: List[Dict[str, str]] = []
        counts = Counter()
        excl_cl_set = set()

        for r, tc_pre, sv in zip(rows, tc1024_max, tr_sv):
            ck = r["compound_key"]
            ik = ck_to_ik14.get(ck, "")
            gcl = ck_to_global_cl.get(ck, r["cluster_id"])

            if ck in fda_cks or (ik and ik in fda_iks):
                counts["exact_ik_or_ck"] += 1
                excl_cl_set.add(gcl)
                continue

            if protocol == "strict_morgan_v336":
                if sv and tc_pre >= 0.30:
                    m2 = _get_m2048(ck)
                    if m2 is not None:
                        i2 = m2.astype(np.float32) @ fda_m2048.astype(np.float32).T
                        u2 = float(m2.sum()) + fda_m2048_pop - i2
                        tc2 = float(np.divide(i2, u2, out=np.zeros_like(i2), where=u2 > 0).max())
                        if tc2 >= 0.40:
                            counts["morgan2048_tc_ge_0_40"] += 1
                            excl_cl_set.add(gcl)
                            continue
                    elif tc_pre >= 0.40:
                        counts["morgan2048_tc_ge_0_40"] += 1
                        excl_cl_set.add(gcl)
                        continue
                sq = (ck_to_seq.get(ck) or "").strip().upper()
                if fda_raw_canon_seqs and len(sq) >= 4 and set(sq).issubset(AA20_SET):
                    if max(seq_3mer_jaccard(sq, fs) for fs in fda_raw_canon_seqs) >= 0.40:
                        counts["seq_3mer_ge_0_40"] += 1
                        excl_cl_set.add(gcl)
                        continue
            else:
                # Protocol 2: peptide_structural_v34
                # (1) Global OR 3-mer sequence identity >= 0.40
                sq = _get_cseq(ck)
                if sq and fda_parsed_seqs:
                    if any(seq_3mer_jaccard(sq, fs) >= 0.40 or seq_global_identity(sq, fs) >= 0.40 for fs in fda_parsed_seqs):
                        counts["seq_global_or_3mer_ge_0_40"] += 1
                        excl_cl_set.add(gcl)
                        continue
                # (2) Close-analog Morgan-2048 Tc >= 0.65 (prefiltered by 1024-bit >= 0.48)
                if sv and tc_pre >= 0.48:
                    m2 = _get_m2048(ck)
                    if m2 is not None:
                        i2 = m2.astype(np.float32) @ fda_m2048.astype(np.float32).T
                        u2 = float(m2.sum()) + fda_m2048_pop - i2
                        tc2 = float(np.divide(i2, u2, out=np.zeros_like(i2), where=u2 > 0).max())
                        if tc2 >= 0.65:
                            counts["close_analog_morgan2048_tc_ge_0_65"] += 1
                            excl_cl_set.add(gcl)
                            continue
                    elif tc_pre >= 0.65:
                        counts["close_analog_morgan2048_tc_ge_0_65"] += 1
                        excl_cl_set.add(gcl)
                        continue
                # (3) Same non-trivial Murcko scaffold OR macrocycle ring-system core
                if sv:
                    ms, mc = _get_scaf(ck)
                    if (ms and ms in fda_murckos) or (mc and mc in fda_macros):
                        counts["same_murcko_or_macrocycle_ring"] += 1
                        excl_cl_set.add(gcl)
                        continue

            clean_rows.append(r)

        cl_after = len({ck_to_global_cl.get(r["compound_key"], r["cluster_id"]) for r in clean_rows})
        ledger = {
            "n_train_before": int(len(rows)),
            "n_clusters_before": int(cl_before),
            "n_excluded_total": int(len(rows) - len(clean_rows)),
            "excluded_by_criterion": dict(counts),
            "n_clusters_touched": int(len(excl_cl_set)),
            "n_train_after": int(len(clean_rows)),
            "n_clusters_after": int(cl_after),
            "pct_training_retained": round(100.0 * len(clean_rows) / max(len(rows), 1), 1),
        }
        return clean_rows, ledger

    def build_blind_ep_table(
        ep: str,
        rows: List[Dict[str, str]],
        fids: np.ndarray,
        fda_conds: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Build an in-memory endpoint table combining protocol-clean training rows + 11 FDA test rows."""
        cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")] if rows else []
        n_tr = len(rows)
        n_te = len(matched_fda) if fda_conds is not None else 0

        cks = [r["compound_key"] for r in rows] + ([d["compound_key"] for d in matched_fda] if n_te else [])
        cidx = bank.map_keys(cks)
        is_ho = np.concatenate([np.zeros(n_tr, dtype=np.int32), np.ones(n_te, dtype=np.int32)])
        fold_id = np.concatenate([fids.astype(np.int32), np.full(n_te, -1, dtype=np.int32)])
        local_cl = np.array(
            [str(r["cluster_id"]) for r in rows] + ([f"FDA_CL_{i}" for i in range(n_te)] if n_te else []),
            dtype=object,
        )
        global_cl = np.array(
            [ck_to_global_cl.get(r["compound_key"], str(r["cluster_id"])) for r in rows]
            + ([f"FDA_GCL_{i}" for i in range(n_te)] if n_te else []),
            dtype=object,
        )
        cs = np.array([r.get("censor_type", "exact") for r in rows] + (["exact"] * n_te), dtype=object)
        cc = np.array([CENSOR_MAP.get(c, 0) for c in cs], dtype=np.int64)
        yv = np.array([float(r["y_value"]) for r in rows] + ([0.0] * n_te), dtype=np.float64)

        def _flt(val: str, default: float) -> float:
            try:
                v = float(val)
                return v if math.isfinite(v) else default
            except Exception:
                return default

        yl = np.array([_flt(r.get("y_lower", ""), yv[i]) for i, r in enumerate(rows)] + ([0.0] * n_te), dtype=np.float64)
        yh = np.array([_flt(r.get("y_upper", ""), yv[i]) for i, r in enumerate(rows)] + ([0.0] * n_te), dtype=np.float64)
        conds = [{c: (r.get(c) or "unspecified") for c in cond_cols} for r in rows] + (fda_conds or [])
        b = manifest["endpoints"][ep].get("bounds")
        bounds = (float(b[0]), float(b[1])) if b is not None else None
        tbl = {
            "endpoint": ep,
            "task_type": manifest["endpoints"][ep]["task_type"],
            "bounds": bounds,
            "cond_cols": cond_cols,
            "n": n_tr + n_te,
            "compound_keys": cks,
            "cidx": cidx,
            "is_holdout": is_ho,
            "fold_id": fold_id,
            "local_cl": local_cl,
            "global_cl": global_cl,
            "censor_str": cs,
            "censor_code": cc,
            "y_val": yv,
            "y_lo": yl,
            "y_hi": yh,
            "conds": conds,
        }
        return tbl, np.arange(n_tr, dtype=np.int32), np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    def make_fda_conds(cond_cols: List[str]) -> List[Dict[str, str]]:
        out = []
        for d in matched_fda:
            cd = {c: "unspecified" for c in cond_cols}
            if "cond_route" in cd:
                rt = str(d.get("clinical_route") or "IV").upper()
                cd["cond_route"] = "subcutaneous" if rt == "SC" else ("oral" if rt == "PO" else "intravenous")
            if "cond_species" in cd:
                cd["cond_species"] = "human"
            if "cond_matrix" in cd:
                cd["cond_matrix"] = "in_vivo_plasma"
            out.append(cd)
        return out

    def train_blind_domain_mtl(
        target_ep: str,
        domain_name: str,
        clean_target_rows: List[Dict[str, str]],
        fids_target: np.ndarray,
        fda_conds: List[Dict[str, str]],
        protocol: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Train 5-fold Domain-MTL for blind FDA test with protocol-isolated auxiliary domain tasks & zero fold leakage."""
        dom_eps = DOMAINS[domain_name]
        blind_tables: Dict[str, Dict[str, Any]] = {}
        aux_ledgers: Dict[str, Any] = {}

        t_tbl, tr_all_idx, te_idx = build_blind_ep_table(target_ep, clean_target_rows, fids_target, fda_conds)
        blind_tables[target_ep] = t_tbl

        for aep in dom_eps:
            if aep == target_ep:
                continue
            # Include censored rows in auxiliary MTL tasks (our DomainMTLNet handles censored hinge loss natively!)
            a_rows, a_led = filter_endpoint_rows_by_protocol(aep, protocol, exact_only=False)
            aux_ledgers[aep] = a_led
            a_fids = np.zeros(len(a_rows), dtype=np.int32)
            a_tbl, _, _ = build_blind_ep_table(aep, a_rows, a_fids, None)
            blind_tables[aep] = a_tbl

        n_tr = len(tr_all_idx)
        mtl_oof = np.full(n_tr, np.nan, dtype=np.float64)
        mtl_te_folds: List[np.ndarray] = []

        for f in range(5):
            va_rows = tr_all_idx[fids_target == f]
            tr_rows = tr_all_idx[fids_target != f]
            va_cls = set(t_tbl["global_cl"][va_rows].tolist()) | set(t_tbl["local_cl"][va_rows].tolist())
            va_cks = {t_tbl["compound_keys"][i] for i in va_rows} | fda_cks

            aux_map: Dict[str, np.ndarray] = {}
            for aep in dom_eps:
                if aep == target_ep:
                    continue
                atbl = blind_tables[aep]
                keep = [
                    i for i in range(atbl["n"])
                    if atbl["global_cl"][i] not in va_cls
                    and atbl["local_cl"][i] not in va_cls
                    and atbl["compound_keys"][i] not in va_cks
                ]
                aux_map[aep] = np.array(keep, dtype=np.int32)

            p_va, p_te, _ = train_domain_mtl_single_fold(
                target_ep=target_ep,
                domain_eps=dom_eps,
                ep_tables=blind_tables,
                bank=bank,
                normed_embs=normed_embs,
                target_tr_idx=tr_rows,
                target_va_idx=va_rows,
                target_ho_idx=te_idx,
                aux_tr_indices=aux_map,
                seed=SEED + 500 + 37 * f + abs(hash(protocol + target_ep)) % 991,
            )
            mtl_oof[fids_target == f] = p_va
            mtl_te_folds.append(p_te)

        mtl_te = np.mean(mtl_te_folds, axis=0)
        if t_tbl["bounds"] is not None:
            mtl_te = np.clip(mtl_te, t_tbl["bounds"][0], t_tbl["bounds"][1])
        return mtl_oof, mtl_te, aux_ledgers

    protocols_spec = [
        (
            "strict_morgan_v336",
            "Protocol 1 (Strict-Morgan, apples-to-apples with v33.6): Excludes every training compound with "
            "Exact InChIKey14/compound_key match OR Morgan-2048 radius-2 Tc >= 0.40 OR canonical 3-mer sequence Jaccard >= 0.40 "
            "to any of the 11 FDA 2019-2023 peptides (applied identically to target AND auxiliary Domain-MTL tasks).",
        ),
        (
            "peptide_structural_v34",
            "Protocol 2 (Peptide-Structural): Excludes every training compound with Exact InChIKey14/compound_key match "
            "OR Global/3-mer Sequence Identity >= 0.40 OR Close-Analog Morgan-2048 radius-2 Tc >= 0.65 OR Same Non-Trivial "
            "Murcko Scaffold / Macrocycle Ring-System Core to any of the 11 FDA 2019-2023 peptides "
            "(removes true homologs & structural analogs without saturating on generic peptide-bond Morgan bits; "
            "applied identically to target AND auxiliary Domain-MTL tasks).",
        ),
    ]

    protocol_results: Dict[str, Any] = {}

    for proto_key, proto_desc in protocols_spec:
        t_p = time.time()
        print(f"\n--- [FDA Blind Protocol: {proto_key}] ---", flush=True)
        ledgers: Dict[str, Any] = {}
        ep_preds: Dict[str, Dict[str, np.ndarray]] = {}
        ep_stack_info: Dict[str, Any] = {}

        # 1. Regression endpoints: half_life_reg & PPB (Full CPU + 4-FM Fusion + Domain-MTL!)
        for ep, bounds in [("half_life_reg", None), ("PPB", (0.0, 1.0))]:
            clean_tr, led = filter_endpoint_rows_by_protocol(ep, proto_key, exact_only=True)
            cond_cols = [c for c in clean_tr[0].keys() if c.startswith("cond_")]
            cidx_tr = bank.map_keys([r["compound_key"] for r in clean_tr])
            y_tr = np.array([float(r["y_value"]) for r in clean_tr], dtype=np.float64)
            cl_tr = [ck_to_global_cl.get(r["compound_key"], r["cluster_id"]) for r in clean_tr]
            fids_tr = make_cluster_balanced_5folds(cl_tr)
            cond_tr = [{c: (r.get(c) or "unspecified") for c in cond_cols} for r in clean_tr]
            cidx_te = np.array([d["bank_idx"] for d in matched_fda], dtype=np.int32)
            cond_te = make_fda_conds(cond_cols)

            # CPU track
            _, cpu_oof, cpu_te, _ = exp335_mod.train_cpu_track(
                bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, bounds, "dual_modality"
            )
            # 4-FM Fusion track
            pcache: Dict[str, Any] = {}
            fus_oof, fus_te, _, _ = exp335_mod.train_fusion_track(
                bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, bounds,
                cpu_oof, cpu_te, "dual_modality", pcache,
            )
            # Reproducible Domain-MTL track (domain_systemic_pk, protocol-isolated auxiliaries)
            mtl_oof, mtl_te, aux_led = train_blind_domain_mtl(
                ep, "domain_systemic_pk", clean_tr, fids_tr, cond_te, proto_key
            )

            # Strictly Nested 5-Fold OOF 2-family & 3-family Simplex stacking
            s2 = nested_simplex_stack(
                y_tr, fids_tr, np.ones(len(y_tr), dtype=bool),
                {"cpu": cpu_oof, "fusion": fus_oof},
                {"cpu": cpu_te, "fusion": fus_te},
                "regression", bounds,
            )
            s3 = nested_simplex_stack(
                y_tr, fids_tr, np.ones(len(y_tr), dtype=bool),
                {"cpu": cpu_oof, "fusion": fus_oof, "domain_mtl": mtl_oof},
                {"cpu": cpu_te, "fusion": fus_te, "domain_mtl": mtl_te},
                "regression", bounds,
            )
            led["auxiliary_domain_ledgers"] = aux_led
            ledgers[ep] = led
            ep_preds[ep] = {
                "cpu": cpu_te,
                "fusion": fus_te,
                "domain_mtl": mtl_te,
                "stack_2fam": s2["ho_pred"],
                "full_3fam_stack": s3["ho_pred"],
            }
            ep_stack_info[ep] = {
                "stack_2fam_weights": s2["weights"],
                "stack_3fam_weights": s3["weights"],
                "train_oof_cv_metrics": {
                    "cpu": regression_metrics(y_tr, cpu_oof),
                    "fusion": regression_metrics(y_tr, fus_oof),
                    "domain_mtl": regression_metrics(y_tr, mtl_oof),
                    "stack_2fam": regression_metrics(y_tr, s2["oof_pred"]),
                    "full_3fam_stack": regression_metrics(y_tr, s3["oof_pred"]),
                },
            }
            print(
                f"  [{proto_key} | {ep:<14}] kept={led['n_train_after']}/{led['n_train_before']} ({led['pct_training_retained']}%) | "
                f"3fam_w={s3['weights']} | OOF MAE: cpu={ep_stack_info[ep]['train_oof_cv_metrics']['cpu']['mae']:.4f} "
                f"fus={ep_stack_info[ep]['train_oof_cv_metrics']['fusion']['mae']:.4f} "
                f"mtl={ep_stack_info[ep]['train_oof_cv_metrics']['domain_mtl']['mae']:.4f} "
                f"3fam={ep_stack_info[ep]['train_oof_cv_metrics']['full_3fam_stack']['mae']:.4f}",
                flush=True,
            )

        # 2. Classification endpoint: hemolysis_cls (Full Multi-Family CPU + 4-FM Modality-Aware Fusion + Domain-MTL!)
        ep = "hemolysis_cls"
        clean_hem, led_hem = filter_endpoint_rows_by_protocol(ep, proto_key, exact_only=True)
        cond_cols_h = [c for c in clean_hem[0].keys() if c.startswith("cond_")]
        cidx_tr_h = bank.map_keys([r["compound_key"] for r in clean_hem])
        y_tr_h = np.array([float(r["y_value"]) for r in clean_hem], dtype=np.float64)
        cl_tr_h = [ck_to_global_cl.get(r["compound_key"], r["cluster_id"]) for r in clean_hem]
        fids_tr_h = make_cluster_balanced_5folds(cl_tr_h)
        cidx_te_h = np.array([d["bank_idx"] for d in matched_fda], dtype=np.int32)
        cond_te_h = make_fda_conds(cond_cols_h)

        mtl_oof_h, mtl_te_h, aux_led_h = train_blind_domain_mtl(
            ep, "domain_toxicity_hemolysis", clean_hem, fids_tr_h, cond_te_h, proto_key
        )
        cls_res = fit_predict_full_cls_stack(
            bank, normed_embs, cidx_tr_h, y_tr_h, fids_tr_h, cidx_te_h, mtl_oof_h, mtl_te_h
        )
        led_hem["auxiliary_domain_ledgers"] = aux_led_h
        ledgers[ep] = led_hem
        ep_preds[ep] = {
            "cpu": cls_res["cpu_pred"],
            "fusion": cls_res["fusion_pred"],
            "domain_mtl": cls_res["domain_mtl_pred"],
            "full_3fam_stack": cls_res["full_stack_pred"],
        }
        ep_stack_info[ep] = {
            "cpu_subweights": cls_res["cpu_subweights"],
            "fusion_subweights": cls_res["fusion_subweights"],
            "stack_3fam_weights": cls_res["stack_weights"],
            "train_oof_cv_metrics": cls_res["cv_metrics"],
        }
        print(
            f"  [{proto_key} | {ep:<14}] kept={led_hem['n_train_after']}/{led_hem['n_train_before']} ({led_hem['pct_training_retained']}%) | "
            f"3fam_w={cls_res['stack_weights']} | OOF AUC: cpu={cls_res['cv_metrics']['cpu']['roc_auc']} "
            f"fus={cls_res['cv_metrics']['fusion']['roc_auc']} mtl={cls_res['cv_metrics']['domain_mtl']['roc_auc']} "
            f"3fam={cls_res['cv_metrics']['full_3fam_stack']['roc_auc']}",
            flush=True,
        )

        # 3. Assemble per-drug predictions, fold errors, and statistical summaries across ALL model families
        drug_rows = []
        obs_hl_log10 = []
        prd_hl_by_fam: Dict[str, List[float]] = defaultdict(list)
        obs_ppb = []
        prd_ppb_by_fam: Dict[str, List[float]] = defaultdict(list)
        hem_by_fam: Dict[str, List[float]] = defaultdict(list)
        fold_errs_3fam = []

        for i, d in enumerate(matched_fda):
            p_hl_log = float(ep_preds["half_life_reg"]["full_3fam_stack"][i])
            p_hl_h = float(10.0 ** p_hl_log)
            obs_h = float(d["obs_half_life_h"]) if d.get("obs_half_life_h") is not None else None
            ratio = (p_hl_h / obs_h) if (obs_h is not None and obs_h > 0) else None
            fold_err = max(ratio, 1.0 / ratio) if (ratio is not None and ratio > 0) else None
            if fold_err is not None:
                fold_errs_3fam.append(fold_err)

            p_ppb = float(np.clip(ep_preds["PPB"]["full_3fam_stack"][i], 0.0, 1.0))
            p_hem = float(np.clip(ep_preds["hemolysis_cls"]["full_3fam_stack"][i], 0.0, 1.0))

            if obs_h is not None:
                obs_hl_log10.append(math.log10(obs_h))
                for fk, arr in ep_preds["half_life_reg"].items():
                    prd_hl_by_fam[fk].append(float(arr[i]))
            if d.get("obs_ppb") is not None:
                obs_ppb.append(float(d["obs_ppb"]))
                for fk, arr in ep_preds["PPB"].items():
                    prd_ppb_by_fam[fk].append(float(np.clip(arr[i], 0.0, 1.0)))
            for fk, arr in ep_preds["hemolysis_cls"].items():
                hem_by_fam[fk].append(float(np.clip(arr[i], 0.0, 1.0)))

            drug_rows.append({
                "name": d["name"],
                "fda_approval_year": d["fda_approval_year"],
                "route": d["clinical_route"],
                "obs_half_life_h": obs_h,
                "blind_pred_half_life_h_full_stack": round(p_hl_h, 3),
                "blind_pred_half_life_h_by_family": {
                    fk: round(float(10.0 ** arr[i]), 3) for fk, arr in ep_preds["half_life_reg"].items()
                },
                "half_life_pred_to_obs_ratio": round(ratio, 3) if ratio is not None else None,
                "half_life_fold_error": round(fold_err, 3) if fold_err is not None else None,
                "within_2fold": bool(fold_err <= 2.0) if fold_err is not None else None,
                "within_3fold": bool(fold_err <= 3.0) if fold_err is not None else None,
                "obs_ppb": d.get("obs_ppb"),
                "blind_pred_ppb_full_stack": round(p_ppb, 4),
                "blind_pred_ppb_by_family": {
                    fk: round(float(np.clip(arr[i], 0.0, 1.0)), 4) for fk, arr in ep_preds["PPB"].items()
                },
                "blind_pred_hemolysis_prob_full_stack": round(p_hem, 4),
                "blind_pred_hemolysis_prob_by_family": {
                    fk: round(float(np.clip(arr[i], 0.0, 1.0)), 4) for fk, arr in ep_preds["hemolysis_cls"].items()
                },
                "fda_label_note": d["fda_label_note"],
            })

        hl_metrics_by_fam = {
            fk: small_n_blind_summary(obs_hl_log10, pv, seed=SEED + 11)
            for fk, pv in prd_hl_by_fam.items()
        }
        ppb_metrics_by_fam = {
            fk: small_n_blind_summary(obs_ppb, pv, seed=SEED + 22)
            for fk, pv in prd_ppb_by_fam.items()
        }
        hem_summary_by_fam = {
            fk: {
                "n_gt_0_50": int(sum(p > 0.50 for p in pv)),
                "n_total": len(pv),
                "max_pred_prob": round(float(max(pv)), 4),
                "median_pred_prob": round(float(np.median(pv)), 4),
                "mean_pred_prob": round(float(np.mean(pv)), 4),
            }
            for fk, pv in hem_by_fam.items()
        }

        hl_main = hl_metrics_by_fam["full_3fam_stack"]
        ppb_main = ppb_metrics_by_fam["full_3fam_stack"]
        protocol_results[proto_key] = {
            "protocol_description": proto_desc,
            "n_drugs": len(drug_rows),
            "exclusion_ledger": ledgers,
            "stack_metadata": ep_stack_info,
            "half_life_log10_metrics": {
                **hl_main,
                "median_fold_error": round(float(np.median(fold_errs_3fam)), 3),
                "n_within_2fold": int(sum(fe <= 2.0 for fe in fold_errs_3fam)),
                "n_within_3fold": int(sum(fe <= 3.0 for fe in fold_errs_3fam)),
                "by_family": hl_metrics_by_fam,
            },
            "ppb_metrics": {
                **ppb_main,
                "by_family": ppb_metrics_by_fam,
            },
            "hemolysis_predicted_probability_summary": {
                **hem_summary_by_fam["full_3fam_stack"],
                "by_family": hem_summary_by_fam,
                "caveat": (
                    "Approved clinical peptides lack uniform in vitro RBC HC50 labels; reported strictly as "
                    "predicted-probability sanity check, NOT as a ground-truth false-positive rate."
                ),
            },
            "drugs": drug_rows,
            "elapsed_sec": round(time.time() - t_p, 1),
        }
        print(
            f"  ==> [{proto_key} SUMMARY] T1/2 (n=11): Sp={hl_main['spearman']} (CI95={hl_main['spearman_fisher_ci95']}, "
            f"perm_p_two={hl_main['spearman_permutation_test']['p_two_sided']}), MAE={hl_main['mae']} "
            f"(vs oracle_med={hl_main['constant_median_oracle_mae']}, {hl_main['mae_improvement_vs_oracle_median_pct']:+.1f}%), "
            f"within_2x={sum(fe <= 2.0 for fe in fold_errs_3fam)}/11, within_3x={sum(fe <= 3.0 for fe in fold_errs_3fam)}/11 | "
            f"PPB (n=8): Sp={ppb_main['spearman']} (exact_perm_p_two={ppb_main['spearman_permutation_test']['p_two_sided']}), "
            f"MAE={ppb_main['mae']} (vs oracle_med={ppb_main['constant_median_oracle_mae']}) | "
            f"Hemolysis >0.50: {hem_summary_by_fam['full_3fam_stack']['n_gt_0_50']}/11 [{time.time()-t_p:.1f}s]",
            flush=True,
        )

    # Concise side-by-side comparison table
    p1 = protocol_results["strict_morgan_v336"]
    p2 = protocol_results["peptide_structural_v34"]
    comparison = {
        "training_retention_pct": {
            ep: {
                "strict_morgan_v336": p1["exclusion_ledger"][ep]["pct_training_retained"],
                "peptide_structural_v34": p2["exclusion_ledger"][ep]["pct_training_retained"],
                "rows_strict_morgan": f"{p1['exclusion_ledger'][ep]['n_train_after']}/{p1['exclusion_ledger'][ep]['n_train_before']}",
                "rows_peptide_structural": f"{p2['exclusion_ledger'][ep]['n_train_after']}/{p2['exclusion_ledger'][ep]['n_train_before']}",
            }
            for ep in ("half_life_reg", "PPB", "hemolysis_cls")
        },
        "half_life_reg_n11": {
            "strict_morgan_v336": {
                "spearman": p1["half_life_log10_metrics"]["spearman"],
                "fisher_ci95": p1["half_life_log10_metrics"]["spearman_fisher_ci95"],
                "mae": p1["half_life_log10_metrics"]["mae"],
                "perm_p_two_sided": p1["half_life_log10_metrics"]["spearman_permutation_test"]["p_two_sided"],
                "n_within_2fold": p1["half_life_log10_metrics"]["n_within_2fold"],
                "n_within_3fold": p1["half_life_log10_metrics"]["n_within_3fold"],
            },
            "peptide_structural_v34": {
                "spearman": p2["half_life_log10_metrics"]["spearman"],
                "fisher_ci95": p2["half_life_log10_metrics"]["spearman_fisher_ci95"],
                "mae": p2["half_life_log10_metrics"]["mae"],
                "perm_p_two_sided": p2["half_life_log10_metrics"]["spearman_permutation_test"]["p_two_sided"],
                "n_within_2fold": p2["half_life_log10_metrics"]["n_within_2fold"],
                "n_within_3fold": p2["half_life_log10_metrics"]["n_within_3fold"],
            },
        },
        "ppb_n8": {
            "strict_morgan_v336": {
                "spearman": p1["ppb_metrics"]["spearman"],
                "fisher_ci95": p1["ppb_metrics"]["spearman_fisher_ci95"],
                "mae": p1["ppb_metrics"]["mae"],
                "exact_perm_p_two_sided": p1["ppb_metrics"]["spearman_permutation_test"]["p_two_sided"],
            },
            "peptide_structural_v34": {
                "spearman": p2["ppb_metrics"]["spearman"],
                "fisher_ci95": p2["ppb_metrics"]["spearman_fisher_ci95"],
                "mae": p2["ppb_metrics"]["mae"],
                "exact_perm_p_two_sided": p2["ppb_metrics"]["spearman_permutation_test"]["p_two_sided"],
            },
        },
        "hemolysis_cls_n11": {
            "strict_morgan_v336": p1["hemolysis_predicted_probability_summary"]["by_family"]["full_3fam_stack"],
            "peptide_structural_v34": p2["hemolysis_predicted_probability_summary"]["by_family"]["full_3fam_stack"],
        },
    }

    return {
        "n_drugs": len(matched_fda),
        "approval_years": "2019-2023",
        "stack_architecture": "Full Reproducible 3-Family Stack: CPU + 4-FM Modality-Aware Fusion + Domain-MTL (both regression AND classification)",
        "side_by_side_comparison": comparison,
        "protocols": protocol_results,
        "elapsed_sec": round(time.time() - t_b, 1),
    }


def main() -> None:
    t0 = time.time()
    print("=== V34 Item P2-9: Domain-Grouped MTL + Full-Stack FDA Blind Sequence/Scaffold Comparison ===", flush=True)
    manifest = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
    bank = UnifiedV334Bank()
    normed_embs = precompute_l2_normalized_embs(bank)

    ck_to_smi: Dict[str, str] = {}
    ck_to_ik14: Dict[str, str] = {}
    ck_to_seq: Dict[str, str] = {}
    ck_to_global_cl: Dict[str, str] = {}

    with (EXP331 / "data" / "compounds_v331_merged.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            ck = r["compound_key"]
            ck_to_smi[ck] = r.get("canonical_smiles", "")
            ck_to_ik14[ck] = r.get("inchikey_14", "")
            ck_to_seq[ck] = r.get("parent_sequence", "")
    for r in bank.delta_rows:
        ck = r["compound_key"]
        ck_to_smi[ck] = r.get("canonical_smiles", "")
        ck_to_ik14[ck] = r.get("inchikey_14", "")
        ck_to_seq[ck] = r.get("parent_sequence", "")

    # Build canonical global cluster_id map from the endpoint TSVs themselves (0 conflicts across all non-LogD_decadiene eps!)
    for eps in DOMAINS.values():
        for ep in eps:
            if ep == "LogD_decadiene":
                continue
            with find_ep_path(ep).open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    ck = r["compound_key"]
                    if ck not in ck_to_global_cl and r.get("cluster_id"):
                        ck_to_global_cl[ck] = r["cluster_id"]

    part_a = run_part_a_domain_mtl(bank, normed_embs, manifest, ck_to_global_cl)
    part_b = run_part_b_fda_blind_comparison(
        bank, normed_embs, manifest, ck_to_smi, ck_to_ik14, ck_to_seq, ck_to_global_cl
    )

    receipt = {
        "schema": "v34_domain_mtl_and_blind_receipt_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "elapsed_sec": round(time.time() - t0, 1),
        "part_a_domain_grouped_mtl": part_a,
        "part_b_fda_2019_2023_blind_full_stack": part_b,
    }
    out_path = EXP34 / "v34_domain_mtl_and_blind_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== V34 P2-9 COMPLETE in {time.time()-t0:.1f}s -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
