#!/usr/bin/env python3
"""v33.3 Step 3 (runs on compute-node 64-core CPU ONLY):
  Part 3A: 3-Lineage Domain-Factored Multi-Task Kernel Transfer + `MBC >= MIC` Monotonic Cascade
           - Lineage I  (Gram-Negative, 8 EPs): MIC/MBC of E. coli, K. pneumoniae, P. aeruginosa, A. baumannii, Salmonella
           - Lineage II (Gram-Positive + Fungal + Biofilm, 14 EPs): S. aureus, S. epidermidis, E. faecalis, E. faecium,
                        B. subtilis, L. monocytogenes, C. albicans, C. neoformans, LC_antimicrobial, MBIC/MBEC_biofilm
           - Lineage III (Permeability/Efflux, 5 EPs): PAMPA, Caco2, MDCK, RRCK, efflux_ratio
           STRICT CROSS-TASK CLUSTER MASKING: When evaluating target endpoint T fold f (or holdout),
           EVERY row in ALL sibling tasks whose cluster_id is in ValFold(T, f) U Holdout(T) is MASKED!
           Plus physics-constrained monotonic residual cascade `log10(MBC) = log10(MIC) + softplus(delta(x))`.

  Part 3B: Small-N PK Zero-Winner's-Curse + Allometric Bodyweight Unification + 20D Proteolytic Cleavage Features
           - Allometric bodyweight unification (`CL_absolute <-> CL_per_kg`, `Vd_absolute <-> Vd_per_kg`)
             with strict cluster isolation and domain-offset indicator.
           - Logit-space Composite GPR / TabPFN v2 for bounded `PPB` & `F`.
           - 20D RDKit SMARTS + Sequence proteolytic cleavage-site x local modification shielding features
             for `half_life_reg`.
           - Zero-tuning `MultiBlockCompositeGPR` + `TabPFN v2` (when run under tabpfn_venv) evaluated
             under strict 5-fold CV and ONCE on untouched 20% Holdout.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

V33_BASE = Path("${V33_BASE_DIR}")
EXP_BASE = V33_BASE / "expansion_v331"
BUNDLE_DIR = V33_BASE / "bundle"
SMALLN_DIR = V33_BASE / "v2_upgrade" / "smalln_cqr"
OUT_DIR = V33_BASE / "v333_science"
TABPFN_CKPT = Path("${TABPFN_CACHE_DIR}/tabpfn-v2-regressor.ckpt")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SMALLN_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))

from run_plan_b1_smalln import UnifiedSmallNBank, _load_endpoint_frozen  # noqa: E402
from smalln_models import (  # noqa: E402
    CompactTabularPreprocessor,
    MultiBlockCompositeGPR,
    MultiBlockKernelPreprocessor,
    compute_multiblock_grams,
)
from v33_metrics import regression_metrics  # noqa: E402
from v333_models import inv_logit, logit_transform_bounds  # noqa: E402

csv.field_size_limit(10_000_000)

LINEAGES: Dict[str, List[str]] = {
    "gram_negative": [
        "MIC_E_coli",
        "MBC_E_coli",
        "MIC_K_pneumoniae",
        "MIC_P_aeruginosa",
        "MBC_P_aeruginosa",
        "MIC_A_baumannii",
        "MBC_A_baumannii",
        "MIC_Salmonella",
    ],
    "gram_positive_fungal_biofilm": [
        "MIC_S_aureus",
        "MBC_S_aureus",
        "MIC_S_epidermidis",
        "MIC_E_faecalis",
        "MBC_E_faecalis",
        "MIC_E_faecium",
        "MIC_B_subtilis",
        "MIC_L_monocytogenes",
        "MIC_C_albicans",
        "MFC_C_albicans",
        "MIC_C_neoformans",
        "LC_antimicrobial",
        "MBIC_biofilm",
        "MBEC_biofilm",
    ],
    "permeability_efflux": [
        "PAMPA",
        "Caco2",
        "MDCK",
        "RRCK",
        "efflux_ratio",
    ],
}

# Monotonic physical cascade pairs: (child_endpoint >= parent_endpoint in log10(uM))
MBC_MIC_PAIRS: List[Tuple[str, str]] = [
    ("MBC_E_coli", "MIC_E_coli"),
    ("MBC_P_aeruginosa", "MIC_P_aeruginosa"),
    ("MBC_A_baumannii", "MIC_A_baumannii"),
    ("MBC_S_aureus", "MIC_S_aureus"),
    ("MBC_E_faecalis", "MIC_E_faecalis"),
    ("MFC_C_albicans", "MIC_C_albicans"),
    ("MBEC_biofilm", "MBIC_biofilm"),
]

# Standard allometric species bodyweights (kg) for CL/Vd physical unification
SPECIES_BW_KG: Dict[str, float] = {
    "human": 70.0,
    "monkey": 3.5,
    "dog": 10.0,
    "rabbit": 3.0,
    "rat": 0.25,
    "mouse": 0.025,
}

SMALLN_PK_ENDPOINTS = ["CL_absolute", "Vd_absolute", "Cmax", "F", "PPB", "Vd_per_kg", "CL_per_kg"]

G_BANK: Optional[UnifiedSmallNBank] = None
G_MANIFEST: Optional[Dict[str, Any]] = None
G_EP_DATA: Dict[str, Dict[str, Any]] = {}
G_TABPFN_OK: bool = False


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _fast_mol_kernel_from_bank(bank: UnifiedSmallNBank, idx_a: np.ndarray, idx_b: np.ndarray, track: str = "dual_modality") -> np.ndarray:
    """Fast, strictly positive-definite frozen multi-block molecular kernel (4-FM exp-cosine + Morgan Tanimoto + Physchem RBF).
    Uses global raw L2-normalized blocks so it's O(1) to slice across multi-task lineages without leakage!
    """
    blocks = ["ncaa_smiles_repr", "ncaa_fusion_hf", "ncaa_protein_proj", "peptideclm2", "chemberta_mtr"]
    if track == "dual_modality":
        blocks = blocks + ["esmc_mean", "esmc_max", "esmc_nterm5", "esmc_cterm5"]
    n_fm = len(blocks)
    w_fm = 0.65 / n_fm
    w_tan = 0.20
    w_phys = 0.15

    K = np.zeros((len(idx_a), len(idx_b)), dtype=np.float64)
    for b in blocks:
        Ea, va = bank.get_emb_f32(b, idx_a)
        Eb, vb = bank.get_emb_f32(b, idx_b)
        na = np.maximum(np.linalg.norm(Ea, axis=1, keepdims=True), 1e-8)
        nb = np.maximum(np.linalg.norm(Eb, axis=1, keepdims=True), 1e-8)
        Ua = Ea / na
        Ub = Eb / nb
        cos = np.clip((Ua @ Ub.T).astype(np.float64), -1.0, 1.0)
        vm = va.astype(np.float64)[:, None] * vb.astype(np.float64)[None, :]
        K += w_fm * (np.exp(-1.25 * np.maximum(1.0 - cos, 0.0)) * vm)

    # Morgan Tanimoto
    Aa = bank.morgan_bits[idx_a].astype(np.float64)
    Ab = bank.morgan_bits[idx_b].astype(np.float64)
    va_s = bank.smiles_valid[idx_a].astype(np.float64)[:, None]
    vb_s = bank.smiles_valid[idx_b].astype(np.float64)[None, :]
    inter = Aa @ Ab.T
    union = Aa.sum(axis=1)[:, None] + Ab.sum(axis=1)[None, :] - inter
    K_tan = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-12) * (va_s * vb_s)
    K += w_tan * K_tan

    # Standardized Physchem RBF (robust clipping, fixed unit scale per dim)
    pdim = 48 if track == "smiles_only" else 83
    Pa = np.nan_to_num(bank.physchem[idx_a, :pdim].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Pb = np.nan_to_num(bank.physchem[idx_b, :pdim].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # Use log1p-signed robust scaling so heavy-tailed MW/Area never dominate
    Pa = np.sign(Pa) * np.log1p(np.abs(Pa))
    Pb = np.sign(Pb) * np.log1p(np.abs(Pb))
    D2 = np.maximum((Pa * Pa).sum(axis=1)[:, None] + (Pb * Pb).sum(axis=1)[None, :] - 2.0 * (Pa @ Pb.T), 0.0)
    K += w_phys * np.exp(-0.5 * D2 / max(float(pdim), 1.0))
    return K


def _softplus(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return np.where(x64 > 25.0, x64, np.log1p(np.exp(np.clip(x64, -30.0, 25.0))))


def _inv_softplus(y: np.ndarray, eps: float = 0.02) -> np.ndarray:
    yc = np.maximum(np.asarray(y, dtype=np.float64), eps)
    return np.where(yc > 25.0, yc, np.log(np.expm1(yc)))


def _fit_blend_w_mae(p_base: np.ndarray, p_new: np.ndarray, y: np.ndarray, reg: float = 0.002) -> float:
    fin = np.isfinite(p_base) & np.isfinite(p_new) & np.isfinite(y)
    if fin.sum() < 4:
        return 0.0
    pb, pn, yt = p_base[fin], p_new[fin], y[fin]
    best_w, best_v = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        v = float(np.mean(np.abs(yt - ((1.0 - w) * pb + w * pn)))) + reg * (w ** 2)
        if v < best_v:
            best_v, best_w = v, float(w)
    return best_w


# =============================================================================
# PART 3A: 3-Lineage Domain-Factored Multi-Task Kernel Transfer + MBC>=MIC
# =============================================================================

def _run_lineage_target(args: Tuple[str, str, List[str], str]) -> Dict[str, Any]:
    """Run strict cross-task-cluster-masked Multi-Task KRR for ONE target endpoint inside its lineage."""
    lineage_name, target_ep, lineage_eps, track = args
    t0 = time.time()
    assert G_BANK is not None and G_MANIFEST is not None

    d_t = G_EP_DATA[target_ep]
    bounds = d_t["bounds"]
    cv_idx = d_t["cv_idx"]
    ho_idx = d_t["ho_idx"]
    cidx_cv = d_t["compound_idx"][cv_idx]
    cidx_ho = d_t["compound_idx"][ho_idx]
    cl_cv = d_t["cluster_id"][cv_idx]
    cl_ho = d_t["cluster_id"][ho_idx]
    fids_cv = d_t["fold_id"][cv_idx]
    y_cv = d_t["y_value"][cv_idx]
    yl_cv, yu_cv, ct_cv = d_t["y_lower"][cv_idx], d_t["y_upper"][cv_idx], d_t["censor_type"][cv_idx]
    y_ho = d_t["y_value"][ho_idx]
    yl_ho, yu_ho, ct_ho = d_t["y_lower"][ho_idx], d_t["y_upper"][ho_idx], d_t["censor_type"][ho_idx]

    valid_cv = G_BANK.smiles_valid[cidx_cv] if track == "smiles_only" else np.ones(len(cv_idx), dtype=bool)
    valid_ho = G_BANK.smiles_valid[cidx_ho] if track == "smiles_only" else np.ones(len(ho_idx), dtype=bool)

    ho_cluster_set = set(str(c) for c in cl_ho)
    siblings = [s for s in lineage_eps if s != target_ep]

    # Helper to fit & predict MT-KRR given target training positions & forbidden cluster set
    def _fit_predict_mtkrr(
        tr_pos_t: np.ndarray,
        q_cidx: np.ndarray,
        forbidden_clusters: Set[str],
        alpha: float = 0.25,
        tau: float = 0.55,
        max_sib_per_task: int = 450,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        rng = np.random.RandomState(seed)
        # Target training rows (exact only for KRR)
        t_cidx = cidx_cv[tr_pos_t]
        t_y = y_cv[tr_pos_t]
        t_cl = cl_cv[tr_pos_t]
        assert set(str(c) for c in t_cl).isdisjoint(forbidden_clusters)
        mu_t = float(np.mean(t_y))
        sd_t = max(float(np.std(t_y)), 1e-4)
        t_z = (t_y - mu_t) / sd_t

        # Build target compound -> z lookup for empirical overlap correlation
        t_map: Dict[int, List[float]] = {}
        for ci, zv in zip(t_cidx, t_z):
            t_map.setdefault(int(ci), []).append(float(zv))
        t_mean_map = {ci: float(np.mean(vs)) for ci, vs in t_map.items()}

        pool_cidx = [t_cidx]
        pool_z = [t_z]
        pool_task = [np.zeros(len(t_cidx), dtype=np.int32)]
        pool_alpha = [np.full(len(t_cidx), alpha, dtype=np.float64)]
        task_rhos: Dict[str, float] = {target_ep: 1.0}
        task_names_used = [target_ep]

        for s_ep in siblings:
            d_s = G_EP_DATA[s_ep]
            # STRICT CROSS-TASK CLUSTER MASKING: drop ANY row in sibling whose cluster is in forbidden_clusters!
            s_cidx_all = d_s["compound_idx"]
            s_cl_all = d_s["cluster_id"]
            s_ct_all = d_s["censor_type"]
            s_y_all = d_s["y_value"]
            s_val_ok = G_BANK.smiles_valid[s_cidx_all] if track == "smiles_only" else np.ones(len(s_cidx_all), dtype=bool)
            s_keep = (
                (s_ct_all == "exact")
                & s_val_ok
                & np.array([str(c) not in forbidden_clusters for c in s_cl_all], dtype=bool)
            )
            if s_keep.sum() < 15:
                continue
            s_pos = np.where(s_keep)[0]
            # Hard assertion: ZERO cluster overlap with target validation fold OR target holdout!
            assert set(str(c) for c in s_cl_all[s_pos]).isdisjoint(forbidden_clusters), (
                f"CROSS-TASK LEAKAGE: {s_ep} leaked into {target_ep}!"
            )

            s_cidx = s_cidx_all[s_pos]
            s_y = s_y_all[s_pos]
            mu_s = float(np.mean(s_y))
            sd_s = max(float(np.std(s_y)), 1e-4)
            s_z = (s_y - mu_s) / sd_s

            # Compute shrunk empirical correlation on overlapping compounds strictly inside allowed training pool
            ov_t, ov_s = [], []
            for ci, zv in zip(s_cidx, s_z):
                ci_int = int(ci)
                if ci_int in t_mean_map:
                    ov_t.append(t_mean_map[ci_int])
                    ov_s.append(float(zv))
            n_ov = len(ov_t)
            if n_ov >= 6 and np.std(ov_t) > 1e-6 and np.std(ov_s) > 1e-6:
                r_emp = float(np.corrcoef(ov_t, ov_s)[0, 1])
                if not math.isfinite(r_emp):
                    r_emp = 0.0
                # Empirical-Bayes shrinkage toward weak positive prior inside antimicrobial lineages, 0 in permeability
                prior_r = 0.20 if lineage_name != "permeability_efflux" else 0.0
                rho = (n_ov * r_emp + 10.0 * prior_r) / (n_ov + 10.0)
            else:
                rho = 0.20 if lineage_name != "permeability_efflux" else 0.0
            rho = float(np.clip(rho, -0.80, 0.85))
            if abs(rho) < 0.08:
                continue

            # Subsample huge sibling tasks cleanly while keeping overlapping compounds
            if len(s_pos) > max_sib_per_task:
                ov_mask = np.array([int(ci) in t_mean_map for ci in s_cidx], dtype=bool)
                ov_i = np.where(ov_mask)[0]
                non_i = np.where(~ov_mask)[0]
                n_rem = max(0, max_sib_per_task - len(ov_i))
                pick_non = rng.choice(non_i, min(len(non_i), n_rem), replace=False) if n_rem > 0 else np.array([], dtype=np.int32)
                pick = np.sort(np.concatenate([ov_i[:max_sib_per_task], pick_non]).astype(np.int32))
                s_cidx = s_cidx[pick]
                s_z = s_z[pick]

            tid = len(task_names_used)
            task_names_used.append(s_ep)
            task_rhos[s_ep] = round(rho, 4)
            pool_cidx.append(s_cidx)
            pool_z.append(s_z)
            pool_task.append(np.full(len(s_cidx), tid, dtype=np.int32))
            # Slightly higher noise regularization on auxiliary sibling observations
            pool_alpha.append(np.full(len(s_cidx), alpha * 1.6, dtype=np.float64))

        all_cidx = np.concatenate(pool_cidx)
        all_z = np.concatenate(pool_z)
        all_tid = np.concatenate(pool_task)
        all_alpha = np.concatenate(pool_alpha)

        # Build strictly PSD task covariance matrix B (rank-1 star correlation + diagonal)
        # B[0, j] = tau * rho_j; B[j, k] = tau^2 * rho_j * rho_k (j != k); B[j, j] = 1.0 -> ALWAYS strictly PSD!
        n_tasks = len(task_names_used)
        rho_vec = np.array([1.0] + [tau * task_rhos[nm] for nm in task_names_used[1:]], dtype=np.float64)
        B = np.outer(rho_vec, rho_vec)
        np.fill_diagonal(B, 1.0)
        # Verify strict PSD
        B_idx = B[np.ix_(all_tid, all_tid)]

        # If target itself is huge (>1400), keep all target rows up to 1400 via Nystrom-free cap? No, all antimicrobials <= 2210 CV exact <= 1600!
        K_mol = _fast_mol_kernel_from_bank(G_BANK, all_cidx, all_cidx, track=track)
        K_mt = K_mol * B_idx
        K_mt[np.diag_indices(len(all_cidx))] += all_alpha + 1e-6

        L, lower = cho_factor(K_mt, lower=True, check_finite=False)
        dual = cho_solve((L, lower), all_z, check_finite=False)

        # Query target task (tid = 0)
        K_q_mol = _fast_mol_kernel_from_bank(G_BANK, q_cidx, all_cidx, track=track)
        b_q = B[0, all_tid][None, :]  # (1, n_pool)
        z_pred = (K_q_mol * b_q) @ dual
        y_pred = mu_t + sd_t * z_pred
        if bounds is not None:
            y_pred = np.clip(y_pred, bounds[0], bounds[1])
        return y_pred.astype(np.float64), task_rhos

    # Evaluate 2 multi-task transfer strengths (tau=0.45, tau=0.70) + single-task baseline kernel (tau=0.0) on 5-fold CV
    configs = [("st_krr", 0.25, 0.0), ("mt_krr_t45", 0.25, 0.45), ("mt_krr_t70", 0.25, 0.70)]
    oof_preds = {nm: np.full(len(cv_idx), np.nan, dtype=np.float64) for nm, _, _ in configs}
    ho_preds = {}
    last_rhos = {}

    for nm, alp, tau in configs:
        for f in range(5):
            va_m = (fids_cv == f) & valid_cv
            tr_m = (fids_cv != f) & valid_cv & (ct_cv == "exact")
            if tr_m.sum() < 8 or va_m.sum() == 0:
                continue
            tr_p = np.where(tr_m)[0]
            va_p = np.where(va_m)[0]
            forb = set(str(c) for c in cl_cv[fids_cv == f]) | ho_cluster_set
            p_va, rhos = _fit_predict_mtkrr(tr_p, cidx_cv[va_p], forb, alpha=alp, tau=tau, seed=142 + f)
            oof_preds[nm][va_p] = p_va
            if f == 0 and tau > 0:
                last_rhos = rhos

        # Fit once on full CV (masking ONLY ho_cluster_set from all siblings!) and predict untouched Holdout
        tr_all_p = np.where(valid_cv & (ct_cv == "exact"))[0]
        ho_preds[nm], _ = _fit_predict_mtkrr(tr_all_p, cidx_ho, ho_cluster_set, alpha=alp, tau=tau, seed=999)

    base_oof = d_t["base_champ_oof"][track]
    base_ho = d_t["base_champ_ho"][track]

    m_cv_ex = valid_cv & (ct_cv == "exact")
    m_ho_ex = valid_ho & (ct_ho == "exact")
    base_cv_m = regression_metrics(y_cv[m_cv_ex], base_oof[m_cv_ex], yl_cv[m_cv_ex], yu_cv[m_cv_ex], ct_cv[m_cv_ex])
    base_ho_m = regression_metrics(y_ho[m_ho_ex], base_ho[m_ho_ex], yl_ho[m_ho_ex], yu_ho[m_ho_ex], ct_ho[m_ho_ex])

    cand_cv = {nm: regression_metrics(y_cv[m_cv_ex], oof_preds[nm][m_cv_ex], yl_cv[m_cv_ex], yu_cv[m_cv_ex], ct_cv[m_cv_ex]) for nm, _, _ in configs}
    cand_ho = {nm: regression_metrics(y_ho[m_ho_ex], ho_preds[nm][m_ho_ex], yl_ho[m_ho_ex], yu_ho[m_ho_ex], ct_ho[m_ho_ex]) for nm, _, _ in configs}

    best_mt = min(("mt_krr_t45", "mt_krr_t70"), key=lambda k: cand_cv[k]["mae"])
    # Nested 5-fold blend with frozen Wave 5A champion (strictly OOF!)
    blend_oof = np.full(len(cv_idx), np.nan, dtype=np.float64)
    for f in range(5):
        tr_m = (fids_cv != f) & m_cv_ex
        va_m = (fids_cv == f) & valid_cv
        wf = _fit_blend_w_mae(base_oof[tr_m], oof_preds[best_mt][tr_m], y_cv[tr_m])
        blend_oof[va_m] = (1.0 - wf) * base_oof[va_m] + wf * oof_preds[best_mt][va_m]
    w_full = _fit_blend_w_mae(base_oof[m_cv_ex], oof_preds[best_mt][m_cv_ex], y_cv[m_cv_ex])
    blend_ho = (1.0 - w_full) * base_ho + w_full * ho_preds[best_mt]

    blend_cv_m = regression_metrics(y_cv[m_cv_ex], blend_oof[m_cv_ex], yl_cv[m_cv_ex], yu_cv[m_cv_ex], ct_cv[m_cv_ex])
    blend_ho_m = regression_metrics(y_ho[m_ho_ex], blend_ho[m_ho_ex], yl_ho[m_ho_ex], yu_ho[m_ho_ex], ct_ho[m_ho_ex])

    d_cv_mae = base_cv_m["mae"] - blend_cv_m["mae"]
    d_st_vs_mt = cand_cv["st_krr"]["mae"] - cand_cv[best_mt]["mae"]
    promoted = (d_cv_mae >= 0.004) and (w_full >= 0.10)

    print(
        f"  [3A {lineage_name[:8]}|{target_ep:<18}|{track[:4]}] ST={cand_cv['st_krr']['mae']:.4f} MT={cand_cv[best_mt]['mae']:.4f} "
        f"(dMT={d_st_vs_mt:+.4f}) | BaseCV={base_cv_m['mae']:.4f}->BlendCV={blend_cv_m['mae']:.4f} (w={w_full:.2f}) | "
        f"HO: {base_ho_m['mae']:.4f}->{blend_ho_m['mae']:.4f} Sp:{base_ho_m['spearman']:.3f}->{blend_ho_m['spearman']:.3f} "
        f"PROM={promoted} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {
        "lineage": lineage_name,
        "endpoint": target_ep,
        "track": track,
        "sibling_rhos_fold0": last_rhos,
        "baseline_cv": base_cv_m,
        "baseline_ho": base_ho_m,
        "candidates_cv": cand_cv,
        "candidates_ho": cand_ho,
        "best_mt": best_mt,
        "mt_gain_over_st_cv": round(d_st_vs_mt, 5),
        "blend_weight": round(w_full, 4),
        "blend_nested_cv": blend_cv_m,
        "blend_ho": blend_ho_m,
        "oof_delta_mae": round(d_cv_mae, 5),
        "promoted_strict_oof": promoted,
    }


def _run_mbc_mic_cascade() -> Dict[str, Any]:
    """Physics-constrained `log10(MBC) = log10(MIC) + softplus(delta(x))` cascade with strict cluster masking."""
    assert G_BANK is not None
    results: Dict[str, Any] = {}
    for child_ep, parent_ep in MBC_MIC_PAIRS:
        d_c = G_EP_DATA[child_ep]
        d_p = G_EP_DATA[parent_ep]
        cv_idx, ho_idx = d_c["cv_idx"], d_c["ho_idx"]
        cidx_cv, cidx_ho = d_c["compound_idx"][cv_idx], d_c["compound_idx"][ho_idx]
        cl_cv, cl_ho = d_c["cluster_id"][cv_idx], d_c["cluster_id"][ho_idx]
        fids_cv = d_c["fold_id"][cv_idx]
        y_cv, ct_cv = d_c["y_value"][cv_idx], d_c["censor_type"][cv_idx]
        y_ho, ct_ho = d_c["y_value"][ho_idx], d_c["censor_type"][ho_idx]
        ho_cl_set = set(str(c) for c in cl_ho)

        # Helper: predict parent MIC strictly excluding forbidden clusters
        def _pred_parent_strict(q_cidx: np.ndarray, forb: Set[str]) -> np.ndarray:
            p_keep = (d_p["censor_type"] == "exact") & np.array([str(c) not in forb for c in d_p["cluster_id"]], dtype=bool)
            p_pos = np.where(p_keep)[0]
            assert set(str(c) for c in d_p["cluster_id"][p_pos]).isdisjoint(forb)
            p_cidx = d_p["compound_idx"][p_pos]
            p_y = d_p["y_value"][p_pos]
            mu_p = float(np.mean(p_y))
            K_pp = _fast_mol_kernel_from_bank(G_BANK, p_cidx, p_cidx, track="dual_modality")
            K_pp[np.diag_indices(len(p_cidx))] += 0.25
            L, lo = cho_factor(K_pp, lower=True, check_finite=False)
            dual = cho_solve((L, lo), p_y - mu_p, check_finite=False)
            K_qp = _fast_mol_kernel_from_bank(G_BANK, q_cidx, p_cidx, track="dual_modality")
            return mu_p + K_qp @ dual

        oof_casc = np.full(len(cv_idx), np.nan, dtype=np.float64)
        oof_mic = np.full(len(cv_idx), np.nan, dtype=np.float64)
        for f in range(5):
            va_m = fids_cv == f
            tr_m = (fids_cv != f) & (ct_cv == "exact")
            if tr_m.sum() < 8 or va_m.sum() == 0:
                continue
            tr_p, va_p = np.where(tr_m)[0], np.where(va_m)[0]
            forb = set(str(c) for c in cl_cv[va_p]) | ho_cl_set
            mic_tr = _pred_parent_strict(cidx_cv[tr_p], forb)
            mic_va = _pred_parent_strict(cidx_cv[va_p], forb)
            oof_mic[va_p] = mic_va

            # Fit non-negative delta in inverse-softplus space via KRR!
            delta_tr = np.maximum(y_cv[tr_p] - mic_tr, 0.02)
            z_tr = _inv_softplus(delta_tr, eps=0.02)
            mu_z = float(np.mean(z_tr))
            K_tt = _fast_mol_kernel_from_bank(G_BANK, cidx_cv[tr_p], cidx_cv[tr_p], track="dual_modality")
            K_tt[np.diag_indices(len(tr_p))] += 0.40
            L, lo = cho_factor(K_tt, lower=True, check_finite=False)
            dual_z = cho_solve((L, lo), z_tr - mu_z, check_finite=False)
            K_vt = _fast_mol_kernel_from_bank(G_BANK, cidx_cv[va_p], cidx_cv[tr_p], track="dual_modality")
            z_va = mu_z + K_vt @ dual_z
            oof_casc[va_p] = mic_va + _softplus(z_va)

        # Full CV -> Holdout
        tr_all = np.where(ct_cv == "exact")[0]
        mic_tr_all = _pred_parent_strict(cidx_cv[tr_all], ho_cl_set)
        mic_ho = _pred_parent_strict(cidx_ho, ho_cl_set)
        z_all = _inv_softplus(np.maximum(y_cv[tr_all] - mic_tr_all, 0.02), eps=0.02)
        mu_za = float(np.mean(z_all))
        K_aa = _fast_mol_kernel_from_bank(G_BANK, cidx_cv[tr_all], cidx_cv[tr_all], track="dual_modality")
        K_aa[np.diag_indices(len(tr_all))] += 0.40
        L, lo = cho_factor(K_aa, lower=True, check_finite=False)
        dual_za = cho_solve((L, lo), z_all - mu_za, check_finite=False)
        K_ha = _fast_mol_kernel_from_bank(G_BANK, cidx_ho, cidx_cv[tr_all], track="dual_modality")
        ho_casc = mic_ho + _softplus(mu_za + K_ha @ dual_za)

        base_oof = d_c["base_champ_oof"]["dual_modality"]
        base_ho = d_c["base_champ_ho"]["dual_modality"]
        m_cv = ct_cv == "exact"
        m_ho = ct_ho == "exact"
        base_cv_m = regression_metrics(y_cv[m_cv], base_oof[m_cv], d_c["y_lower"][cv_idx][m_cv], d_c["y_upper"][cv_idx][m_cv], ct_cv[m_cv])
        casc_cv_m = regression_metrics(y_cv[m_cv], oof_casc[m_cv], d_c["y_lower"][cv_idx][m_cv], d_c["y_upper"][cv_idx][m_cv], ct_cv[m_cv])
        base_ho_m = regression_metrics(y_ho[m_ho], base_ho[m_ho], d_c["y_lower"][ho_idx][m_ho], d_c["y_upper"][ho_idx][m_ho], ct_ho[m_ho])
        casc_ho_m = regression_metrics(y_ho[m_ho], ho_casc[m_ho], d_c["y_lower"][ho_idx][m_ho], d_c["y_upper"][ho_idx][m_ho], ct_ho[m_ho])

        # Check physical violation rate (% where pred_MBC < pred_MIC on holdout)
        viol_base = float(np.mean(base_ho < mic_ho))
        viol_casc = float(np.mean(ho_casc < mic_ho))
        results[child_ep] = {
            "parent": parent_ep,
            "baseline_cv": base_cv_m,
            "cascade_cv": casc_cv_m,
            "baseline_ho": base_ho_m,
            "cascade_ho": casc_ho_m,
            "ho_violation_rate_baseline": round(viol_base, 4),
            "ho_violation_rate_cascade": round(viol_casc, 4),
        }
        print(
            f"  [3A-Casc {child_ep:<16} >= {parent_ep:<16}] CV MAE: {base_cv_m['mae']:.4f} -> {casc_cv_m['mae']:.4f} | "
            f"HO MAE: {base_ho_m['mae']:.4f} -> {casc_ho_m['mae']:.4f} | Viol: {100*viol_base:.1f}% -> {100*viol_casc:.1f}%",
            flush=True,
        )
    return results


# =============================================================================
# PART 3B: Small-N PK Zero-Winner's-Curse + Allometric Unification + Proteolytic
# =============================================================================

def _compute_20d_proteolytic_features(smiles_list: Sequence[str], seq_list: Sequence[str]) -> np.ndarray:
    """20D RDKit SMARTS + Sequence proteolytic cleavage-site x local modification shielding descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    pat_nme_amide = Chem.MolFromSmarts("[NX3]([CH3])C(=O)")
    pat_aib = Chem.MolFromSmarts("[NX3][CX4]([CH3])([CH3])C(=O)")
    pat_disulfide = Chem.MolFromSmarts("[SX2]-[SX2]")
    pat_c8 = Chem.MolFromSmarts("CCCCCCCC")
    pat_c14 = Chem.MolFromSmarts("CCCCCCCCCCCCCC")
    pat_peg = Chem.MolFromSmarts("OCCOCC")
    pat_n_acetyl = Chem.MolFromSmarts("[CH3]C(=O)[NX3H1][CX4]C(=O)")
    pat_free_nterm = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)][CX4]C(=O)")
    pat_amide_cterm = Chem.MolFromSmarts("[CX4]C(=O)[NX3H2]")
    pat_acid_cterm = Chem.MolFromSmarts("[CX4]C(=O)[OX2H1]")
    pat_guanidinium = Chem.MolFromSmarts("NC(=[NH2,NH])N")
    pat_chiral_alpha = Chem.MolFromSmarts("[NX3][CX4H1]([*])C(=O)")

    out = np.zeros((len(smiles_list), 20), dtype=np.float32)
    for i, (smi, seq) in enumerate(zip(smiles_list, seq_list)):
        s = (seq or "").strip().upper()
        aa = [c for c in s if c in "ACDEFGHIKLMNPQRSTVWY"]
        L = max(len(aa), 1)
        # Sequence cleavage motifs (8 dims)
        n_rk = sum(1 for c in aa if c in ("R", "K"))
        n_tryp = sum(1 for j in range(len(aa) - 1) if aa[j] in ("R", "K") and aa[j + 1] != "P")
        n_dibasic = sum(1 for j in range(len(aa) - 1) if aa[j] in ("R", "K") and aa[j + 1] in ("R", "K"))
        n_chymo = sum(1 for c in aa if c in ("F", "Y", "W", "L"))
        dpp4_motif = 1.0 if (len(aa) >= 2 and aa[1] in ("P", "A")) else 0.0
        n_pro = sum(1 for c in aa if c == "P")
        n_cys = sum(1 for c in aa if c == "C")
        out[i, 0] = n_rk / float(L)
        out[i, 1] = n_tryp / float(L)
        out[i, 2] = float(n_dibasic)
        out[i, 3] = n_chymo / float(L)
        out[i, 4] = dpp4_motif
        out[i, 5] = n_pro / float(L)
        out[i, 6] = n_cys / float(L)
        out[i, 7] = math.log1p(len(aa))

        # RDKit SMARTS chemical shielding motifs (12 dims)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            ri = mol.GetRingInfo()
            max_ring = max((len(r) for r in ri.AtomRings()), default=0)
            is_macro = 1.0 if max_ring >= 12 else 0.0
            n_nme = len(mol.GetSubstructMatches(pat_nme_amide)) if pat_nme_amide else 0
            n_aib = len(mol.GetSubstructMatches(pat_aib)) if pat_aib else 0
            n_ss = len(mol.GetSubstructMatches(pat_disulfide)) if pat_disulfide else 0
            has_c8 = 1.0 if (pat_c8 and mol.HasSubstructMatch(pat_c8)) else 0.0
            has_c14 = 1.0 if (pat_c14 and mol.HasSubstructMatch(pat_c14)) else 0.0
            n_peg = len(mol.GetSubstructMatches(pat_peg)) if pat_peg else 0
            has_nac = 1.0 if (pat_n_acetyl and mol.HasSubstructMatch(pat_n_acetyl)) else 0.0
            has_free_n = 1.0 if (pat_free_nterm and mol.HasSubstructMatch(pat_free_nterm)) else 0.0
            has_camide = 1.0 if (pat_amide_cterm and mol.HasSubstructMatch(pat_amide_cterm)) else 0.0
            n_guan = len(mol.GetSubstructMatches(pat_guanidinium)) if pat_guanidinium else 0
            # Approximate D-amino acid / stereochemical inversion fraction via CIP R-vs-S at alpha carbons
            n_alpha = 0
            n_d_like = 0
            if pat_chiral_alpha:
                try:
                    Chem.AssignStereochemistry(mol, cleanIt=False, force=False)
                    for match in mol.GetSubstructMatches(pat_chiral_alpha):
                        atom = mol.GetAtomWithIdx(match[1])
                        if atom.HasProp("_CIPCode"):
                            n_alpha += 1
                            # Most natural L-amino acids are S (except Cys); R indicates D-residue!
                            if atom.GetProp("_CIPCode") == "R":
                                n_d_like += 1
                except Exception:
                    pass
            d_frac = (n_d_like / float(n_alpha)) if n_alpha > 0 else 0.0
            out[i, 8] = is_macro
            out[i, 9] = math.log1p(max_ring)
            out[i, 10] = float(n_nme)
            out[i, 11] = float(n_aib)
            out[i, 12] = float(n_ss)
            out[i, 13] = has_c8
            out[i, 14] = has_c14
            out[i, 15] = math.log1p(n_peg)
            out[i, 16] = has_nac - has_free_n  # N-terminal shielding index
            out[i, 17] = has_camide
            out[i, 18] = float(n_guan)
            out[i, 19] = d_frac
    return out


def _run_halflife_proteolytic_ablation() -> Dict[str, Any]:
    """Evaluate 20D proteolytic cleavage-site x local shielding features on half_life_reg (strict 5-fold CV + Holdout)."""
    t0 = time.time()
    assert G_BANK is not None
    d = G_EP_DATA["half_life_reg"]
    # Load canonical_smiles + parent_sequence from compounds_v331_merged.tsv
    comp_meta: Dict[str, Tuple[str, str]] = {}
    with (EXP_BASE / "data/compounds_v331_merged.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            comp_meta[r["compound_key"]] = (r.get("canonical_smiles", ""), r.get("parent_sequence", ""))

    all_keys = [r["compound_key"] for r in d["rows"]]
    smi_all = [comp_meta.get(k, ("", ""))[0] for k in all_keys]
    seq_all = [comp_meta.get(k, ("", ""))[1] for k in all_keys]
    X_prot_all = _compute_20d_proteolytic_features(smi_all, seq_all)

    cv_idx, ho_idx = d["cv_idx"], d["ho_idx"]
    fids_cv = d["fold_id"][cv_idx]
    y_cv, ct_cv = d["y_value"][cv_idx], d["censor_type"][cv_idx]
    y_ho, ct_ho = d["y_value"][ho_idx], d["censor_type"][ho_idx]
    cidx_cv, cidx_ho = d["compound_idx"][cv_idx], d["compound_idx"][ho_idx]
    cond_cv = [d["cond_rows"][i] for i in cv_idx]
    cond_ho = [d["cond_rows"][i] for i in ho_idx]
    Xp_cv, Xp_ho = X_prot_all[cv_idx], X_prot_all[ho_idx]

    # Compare CompactTabular LightGBM WITHOUT vs WITH 20D proteolytic features + blend with frozen champion!
    oof_no = np.full(len(cv_idx), np.nan)
    oof_yes = np.full(len(cv_idx), np.nan)
    ho_no_f, ho_yes_f = [], []

    for f in range(5):
        tr_m = (fids_cv != f) & (ct_cv == "exact")
        va_m = fids_cv == f
        tr_p, va_p = np.where(tr_m)[0], np.where(va_m)[0]
        pre = CompactTabularPreprocessor(track="dual_modality", cond_cols=d["cond_cols"])
        Xt_tr = pre.fit_transform(G_BANK, cidx_cv[tr_p], [cond_cv[i] for i in tr_p])
        Xt_va = pre.transform(G_BANK, cidx_cv[va_p], [cond_cv[i] for i in va_p])
        Xt_ho = pre.transform(G_BANK, cidx_ho, cond_rows=cond_ho)

        # Standardize 20D proteolytic block strictly on training fold
        mu = Xp_cv[tr_p].mean(axis=0)
        sd = Xp_cv[tr_p].std(axis=0)
        sd[sd < 1e-6] = 1.0
        P_tr = np.clip((Xp_cv[tr_p] - mu) / sd, -6.0, 6.0)
        P_va = np.clip((Xp_cv[va_p] - mu) / sd, -6.0, 6.0)
        P_ho = np.clip((Xp_ho - mu) / sd, -6.0, 6.0)

        lgb_kw = dict(
            n_estimators=320, learning_rate=0.04, num_leaves=23, max_depth=5,
            min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=5.0, random_state=142 + f, n_jobs=2, verbose=-1,
        )
        m_no = lgb.LGBMRegressor(**lgb_kw).fit(Xt_tr, y_cv[tr_p])
        oof_no[va_p] = m_no.predict(Xt_va)
        ho_no_f.append(m_no.predict(Xt_ho))

        m_yes = lgb.LGBMRegressor(**lgb_kw).fit(np.hstack([Xt_tr, P_tr]), y_cv[tr_p])
        oof_yes[va_p] = m_yes.predict(np.hstack([Xt_va, P_va]))
        ho_yes_f.append(m_yes.predict(np.hstack([Xt_ho, P_ho])))

    ho_no = np.mean(np.stack(ho_no_f, axis=0), axis=0)
    ho_yes = np.mean(np.stack(ho_yes_f, axis=0), axis=0)

    # Also test residual booster on top of frozen champion using 20D proteolytic features + cond!
    base_oof = d["base_champ_oof"]["dual_modality"]
    base_ho = d["base_champ_ho"]["dual_modality"]
    oof_blend = np.full(len(cv_idx), np.nan)
    for f in range(5):
        tr_m = (fids_cv != f) & (ct_cv == "exact")
        va_m = fids_cv == f
        wf = _fit_blend_w_mae(base_oof[tr_m], oof_yes[tr_m], y_cv[tr_m])
        oof_blend[va_m] = (1.0 - wf) * base_oof[va_m] + wf * oof_yes[va_m]
    m_cv_ex = ct_cv == "exact"
    m_ho_ex = ct_ho == "exact"
    w_full = _fit_blend_w_mae(base_oof[m_cv_ex], oof_yes[m_cv_ex], y_cv[m_cv_ex])
    ho_blend = (1.0 - w_full) * base_ho + w_full * ho_yes

    res = {
        "n_cv_exact": int(m_cv_ex.sum()),
        "n_ho_exact": int(m_ho_ex.sum()),
        "lgb_no_prot_cv": regression_metrics(y_cv[m_cv_ex], oof_no[m_cv_ex], d["y_lower"][cv_idx][m_cv_ex], d["y_upper"][cv_idx][m_cv_ex], ct_cv[m_cv_ex]),
        "lgb_with_prot20_cv": regression_metrics(y_cv[m_cv_ex], oof_yes[m_cv_ex], d["y_lower"][cv_idx][m_cv_ex], d["y_upper"][cv_idx][m_cv_ex], ct_cv[m_cv_ex]),
        "lgb_no_prot_ho": regression_metrics(y_ho[m_ho_ex], ho_no[m_ho_ex], d["y_lower"][ho_idx][m_ho_ex], d["y_upper"][ho_idx][m_ho_ex], ct_ho[m_ho_ex]),
        "lgb_with_prot20_ho": regression_metrics(y_ho[m_ho_ex], ho_yes[m_ho_ex], d["y_lower"][ho_idx][m_ho_ex], d["y_upper"][ho_idx][m_ho_ex], ct_ho[m_ho_ex]),
        "base_champ_cv": regression_metrics(y_cv[m_cv_ex], base_oof[m_cv_ex], d["y_lower"][cv_idx][m_cv_ex], d["y_upper"][cv_idx][m_cv_ex], ct_cv[m_cv_ex]),
        "base_champ_ho": regression_metrics(y_ho[m_ho_ex], base_ho[m_ho_ex], d["y_lower"][ho_idx][m_ho_ex], d["y_upper"][ho_idx][m_ho_ex], ct_ho[m_ho_ex]),
        "blend_weight": round(w_full, 4),
        "blend_nested_cv": regression_metrics(y_cv[m_cv_ex], oof_blend[m_cv_ex], d["y_lower"][cv_idx][m_cv_ex], d["y_upper"][cv_idx][m_cv_ex], ct_cv[m_cv_ex]),
        "blend_ho": regression_metrics(y_ho[m_ho_ex], ho_blend[m_ho_ex], d["y_lower"][ho_idx][m_ho_ex], d["y_upper"][ho_idx][m_ho_ex], ct_ho[m_ho_ex]),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(
        f"  [3B-Prot20 half_life_reg] LGB no_prot CV={res['lgb_no_prot_cv']['mae']:.4f} -> +prot20 CV={res['lgb_with_prot20_cv']['mae']:.4f} | "
        f"HO: {res['lgb_no_prot_ho']['mae']:.4f} -> {res['lgb_with_prot20_ho']['mae']:.4f} | "
        f"Champ HO: {res['base_champ_ho']['mae']:.4f} -> Blend(w={w_full:.2f}) HO={res['blend_ho']['mae']:.4f}",
        flush=True,
    )
    return res


def _run_smalln_pk_allometric_and_tabpfn() -> Dict[str, Any]:
    """Zero-Winner's-Curse Small-N PK Benchmark:
    - MultiBlockCompositeGPR (native + allometric bodyweight unification + logit for PPB/F)
    - TabPFN v2 (when available in tabpfn_venv)
    """
    assert G_BANK is not None and G_MANIFEST is not None
    tabpfn_cls = None
    if G_TABPFN_OK:
        from tabpfn import TabPFNRegressor  # type: ignore
        tabpfn_cls = TabPFNRegressor

    # Physical allometric partner pairs: (target, source, formula_fn(y_src, bw_kg))
    allometric_pairs: Dict[str, Tuple[str, Any]] = {
        "CL_absolute": ("CL_per_kg", lambda y_s, bw: y_s + math.log10(0.06 * bw)),
        "CL_per_kg": ("CL_absolute", lambda y_s, bw: y_s - math.log10(0.06 * bw)),
        "Vd_absolute": ("Vd_per_kg", lambda y_s, bw: y_s + math.log10(bw)),
        "Vd_per_kg": ("Vd_absolute", lambda y_s, bw: y_s - math.log10(bw)),
    }

    pk_results: Dict[str, Any] = {}
    for ep in SMALLN_PK_ENDPOINTS:
        t0 = time.time()
        d = G_EP_DATA[ep]
        bounds = d["bounds"]
        use_logit = ep in ("PPB", "F")
        cv_idx, ho_idx = d["cv_idx"], d["ho_idx"]
        cidx_cv, cidx_ho = d["compound_idx"][cv_idx], d["compound_idx"][ho_idx]
        cl_cv, cl_ho = d["cluster_id"][cv_idx], d["cluster_id"][ho_idx]
        fids_cv = d["fold_id"][cv_idx]
        y_cv, ct_cv = d["y_value"][cv_idx], d["censor_type"][cv_idx]
        y_ho, ct_ho = d["y_value"][ho_idx], d["censor_type"][ho_idx]
        cond_cv = [d["cond_rows"][i] for i in cv_idx]
        cond_ho = [d["cond_rows"][i] for i in ho_idx]
        ho_cl_set = set(str(c) for c in cl_ho)

        # Helper to gather strictly cluster-masked allometric augmentation rows
        def _get_allo_aug(forb: Set[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
            if ep not in allometric_pairs:
                return np.array([], dtype=np.int32), np.array([], dtype=np.float64), []
            src_ep, conv_fn = allometric_pairs[ep]
            ds = G_EP_DATA[src_ep]
            aug_cidx, aug_y, aug_cond = [], [], []
            for i in range(len(ds["rows"])):
                if ds["censor_type"][i] != "exact":
                    continue
                if str(ds["cluster_id"][i]) in forb:
                    continue
                sp = ds["cond_rows"][i].get("cond_species", "unspecified").lower()
                if sp not in SPECIES_BW_KG:
                    continue
                aug_cidx.append(int(ds["compound_idx"][i]))
                aug_y.append(float(conv_fn(ds["y_value"][i], SPECIES_BW_KG[sp])))
                c_row = {c: ds["cond_rows"][i].get(c, "unspecified") for c in d["cond_cols"]}
                c_row["cond_allo_src"] = "converted"
                aug_cond.append(c_row)
            return np.array(aug_cidx, dtype=np.int32), np.array(aug_y, dtype=np.float64), aug_cond

        cands = ["gpr_native"]
        if ep in allometric_pairs:
            cands.append("gpr_allometric")
        if tabpfn_cls is not None:
            cands.append("tabpfn_native")
            if ep in allometric_pairs:
                cands.append("tabpfn_allometric")

        oof_map = {k: np.full(len(cv_idx), np.nan) for k in cands}
        ho_folds = {k: [] for k in cands}
        n_allo_added = 0

        for f in range(5):
            tr_m = (fids_cv != f) & (ct_cv == "exact")
            va_m = fids_cv == f
            if tr_m.sum() < 6 or va_m.sum() == 0:
                continue
            tr_p, va_p = np.where(tr_m)[0], np.where(va_m)[0]
            forb = set(str(c) for c in cl_cv[va_p]) | ho_cl_set

            # 1. Native zero-tuning MultiBlockCompositeGPR (in logit space for PPB/F!)
            pre_k = MultiBlockKernelPreprocessor(track="dual_modality", cond_cols=d["cond_cols"])
            Xk_tr = pre_k.fit_transform(G_BANK, cidx_cv[tr_p], [cond_cv[i] for i in tr_p])
            Xk_va = pre_k.transform(G_BANK, cidx_cv[va_p], [cond_cv[i] for i in va_p])
            Xk_ho = pre_k.transform(G_BANK, cidx_ho, cond_rows=cond_ho)

            y_tr_fit = logit(np.clip(y_cv[tr_p], 0.005, 0.995)) if use_logit else y_cv[tr_p]
            gpr = MultiBlockCompositeGPR(fm_kernel="exp_cosine", exp_gamma=1.25, bounds=(None if use_logit else bounds))
            gpr.fit(Xk_tr, y_tr_fit)
            p_va = inv_logit(gpr.predict(Xk_va)) if use_logit else gpr.predict(Xk_va)
            p_ho = inv_logit(gpr.predict(Xk_ho)) if use_logit else gpr.predict(Xk_ho)
            oof_map["gpr_native"][va_p] = p_va
            ho_folds["gpr_native"].append(p_ho)

            # 2. Allometric-Unified GPR (strictly cluster-masked!)
            if "gpr_allometric" in cands:
                acidx, ay, acond = _get_allo_aug(forb)
                if f == 0:
                    n_allo_added = len(acidx)
                tr_cond_tag = [{**cond_cv[i], "cond_allo_src": "native"} for i in tr_p]
                va_cond_tag = [{**cond_cv[i], "cond_allo_src": "native"} for i in va_p]
                ho_cond_tag = [{**c, "cond_allo_src": "native"} for c in cond_ho]
                pre_a = MultiBlockKernelPreprocessor(track="dual_modality", cond_cols=list(d["cond_cols"]) + ["cond_allo_src"])
                comb_cidx = np.concatenate([cidx_cv[tr_p], acidx]) if len(acidx) > 0 else cidx_cv[tr_p]
                comb_y = np.concatenate([y_cv[tr_p], ay]) if len(acidx) > 0 else y_cv[tr_p]
                comb_cond = tr_cond_tag + acond
                Xa_tr = pre_a.fit_transform(G_BANK, comb_cidx, comb_cond)
                Xa_va = pre_a.transform(G_BANK, cidx_cv[va_p], va_cond_tag)
                Xa_ho = pre_a.transform(G_BANK, cidx_ho, ho_cond_tag)
                gpr_a = MultiBlockCompositeGPR(fm_kernel="exp_cosine", exp_gamma=1.25, bounds=bounds)
                gpr_a.fit(Xa_tr, comb_y)
                oof_map["gpr_allometric"][va_p] = gpr_a.predict(Xa_va)
                ho_folds["gpr_allometric"].append(gpr_a.predict(Xa_ho))

            # 3. Zero-tuning TabPFN v2 (native + allometric!)
            if tabpfn_cls is not None:
                pre_t = CompactTabularPreprocessor(track="dual_modality", cond_cols=d["cond_cols"])
                Xt_tr = pre_t.fit_transform(G_BANK, cidx_cv[tr_p], [cond_cv[i] for i in tr_p])
                Xt_va = pre_t.transform(G_BANK, cidx_cv[va_p], [cond_cv[i] for i in va_p])
                Xt_ho = pre_t.transform(G_BANK, cidx_ho, cond_rows=cond_ho)
                tp = tabpfn_cls(device="cpu", model_path=str(TABPFN_CKPT), n_estimators=4, ignore_pretraining_limits=True)
                tp.fit(Xt_tr, y_tr_fit)
                pv = inv_logit(tp.predict(Xt_va)) if use_logit else tp.predict(Xt_va)
                ph = inv_logit(tp.predict(Xt_ho)) if use_logit else tp.predict(Xt_ho)
                if bounds is not None:
                    pv, ph = np.clip(pv, bounds[0], bounds[1]), np.clip(ph, bounds[0], bounds[1])
                oof_map["tabpfn_native"][va_p] = pv
                ho_folds["tabpfn_native"].append(ph)

                if "tabpfn_allometric" in cands:
                    acidx, ay, acond = _get_allo_aug(forb)
                    tr_cond_tag = [{**cond_cv[i], "cond_allo_src": "native"} for i in tr_p]
                    va_cond_tag = [{**cond_cv[i], "cond_allo_src": "native"} for i in va_p]
                    ho_cond_tag = [{**c, "cond_allo_src": "native"} for c in cond_ho]
                    pre_ta = CompactTabularPreprocessor(track="dual_modality", cond_cols=list(d["cond_cols"]) + ["cond_allo_src"])
                    comb_cidx = np.concatenate([cidx_cv[tr_p], acidx]) if len(acidx) > 0 else cidx_cv[tr_p]
                    comb_y = np.concatenate([y_cv[tr_p], ay]) if len(acidx) > 0 else y_cv[tr_p]
                    Xta_tr = pre_ta.fit_transform(G_BANK, comb_cidx, tr_cond_tag + acond)
                    Xta_va = pre_ta.transform(G_BANK, cidx_cv[va_p], va_cond_tag)
                    Xta_ho = pre_ta.transform(G_BANK, cidx_ho, ho_cond_tag)
                    tpa = tabpfn_cls(device="cpu", model_path=str(TABPFN_CKPT), n_estimators=4, ignore_pretraining_limits=True)
                    tpa.fit(Xta_tr, comb_y)
                    pva = tpa.predict(Xta_va)
                    pha = tpa.predict(Xta_ho)
                    if bounds is not None:
                        pva, pha = np.clip(pva, bounds[0], bounds[1]), np.clip(pha, bounds[0], bounds[1])
                    oof_map["tabpfn_allometric"][va_p] = pva
                    ho_folds["tabpfn_allometric"].append(pha)

        ho_map = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in ho_folds.items() if v}
        # Also evaluate equal-weight zero-tuning ensemble of GPR + TabPFN (zero hyperparameter tuning!)
        if "tabpfn_native" in ho_map:
            oof_map["zero_tune_ens"] = 0.5 * oof_map["gpr_native"] + 0.5 * oof_map["tabpfn_native"]
            ho_map["zero_tune_ens"] = 0.5 * ho_map["gpr_native"] + 0.5 * ho_map["tabpfn_native"]
            cands.append("zero_tune_ens")
        if "tabpfn_allometric" in ho_map:
            oof_map["zero_tune_allo_ens"] = 0.5 * oof_map["gpr_allometric"] + 0.5 * oof_map["tabpfn_allometric"]
            ho_map["zero_tune_allo_ens"] = 0.5 * ho_map["gpr_allometric"] + 0.5 * ho_map["tabpfn_allometric"]
            cands.append("zero_tune_allo_ens")

        m_cv = ct_cv == "exact"
        m_ho = ct_ho == "exact"
        base_oof = d["base_champ_oof"]["dual_modality"]
        base_ho = d["base_champ_ho"]["dual_modality"]
        base_cv_m = regression_metrics(y_cv[m_cv], base_oof[m_cv], d["y_lower"][cv_idx][m_cv], d["y_upper"][cv_idx][m_cv], ct_cv[m_cv])
        base_ho_m = regression_metrics(y_ho[m_ho], base_ho[m_ho], d["y_lower"][ho_idx][m_ho], d["y_upper"][ho_idx][m_ho], ct_ho[m_ho])

        c_cv = {k: regression_metrics(y_cv[m_cv], oof_map[k][m_cv], d["y_lower"][cv_idx][m_cv], d["y_upper"][cv_idx][m_cv], ct_cv[m_cv]) for k in cands}
        c_ho = {k: regression_metrics(y_ho[m_ho], ho_map[k][m_ho], d["y_lower"][ho_idx][m_ho], d["y_upper"][ho_idx][m_ho], ct_ho[m_ho]) for k in cands}
        best_c = min(cands, key=lambda k: c_cv[k]["mae"])

        pk_results[ep] = {
            "n_cv_exact": int(m_cv.sum()),
            "n_ho_exact": int(m_ho.sum()),
            "n_allometric_aug_fold0": n_allo_added,
            "use_logit": use_logit,
            "baseline_cv": base_cv_m,
            "baseline_ho": base_ho_m,
            "candidates_cv": c_cv,
            "candidates_ho": c_ho,
            "best_zero_tune_cv": best_c,
            "best_zero_tune_ho": c_ho[best_c],
        }
        print(
            f"  [3B-PK {ep:<14}] aug={n_allo_added:3d} | Base CV={base_cv_m['mae']:.4f} HO={base_ho_m['mae']:.4f} (R2={base_ho_m['r2']:.3f}) | "
            f"BestZT={best_c:<18} CV={c_cv[best_c]['mae']:.4f} HO={c_ho[best_c]['mae']:.4f} (R2={c_ho[best_c]['r2']:.3f}) ({time.time()-t0:.1f}s)",
            flush=True,
        )
    return pk_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-3a", action="store_true")
    parser.add_argument("--skip-3b", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global G_BANK, G_MANIFEST, G_EP_DATA, G_TABPFN_OK

    try:
        import tabpfn  # noqa: F401
        G_TABPFN_OK = TABPFN_CKPT.exists()
    except Exception:
        G_TABPFN_OK = False
    print(f"[Step 3] TabPFN v2 available: {G_TABPFN_OK}", flush=True)

    G_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
    G_BANK = UnifiedSmallNBank()

    # Preload all needed endpoints once in parent process (CoW-shared across fork workers!)
    needed_eps: Set[str] = set(SMALLN_PK_ENDPOINTS) | {"half_life_reg"}
    for eps in LINEAGES.values():
        needed_eps.update(eps)
    for ep in sorted(needed_eps):
        if ep in G_MANIFEST["endpoints"]:
            G_EP_DATA[ep] = _load_endpoint_frozen(ep, G_MANIFEST["endpoints"][ep], G_BANK)
    print(f"[Step 3] Loaded {len(G_EP_DATA)} frozen endpoints in {time.time()-t_all:.1f}s", flush=True)

    res_3a_lineage: Dict[str, Any] = {}
    res_3a_cascade: Dict[str, Any] = {}
    if not args.skip_3a:
        tasks = []
        for lname, eps in LINEAGES.items():
            present = [e for e in eps if e in G_EP_DATA]
            for ep in present:
                tasks.append((lname, ep, present, "dual_modality"))
        print(f"\n[Step 3A] Running 3-Lineage Domain-Factored Multi-Task Kernel Transfer ({len(tasks)} endpoints)...", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(args.workers, len(tasks))) as pool:
            for r in pool.imap_unordered(_run_lineage_target, tasks):
                res_3a_lineage[r["endpoint"]] = r

        print("\n[Step 3A-Cascade] Running Physics-Constrained MBC >= MIC Monotonic Residual Cascade...", flush=True)
        res_3a_cascade = _run_mbc_mic_cascade()

    res_3b_pk: Dict[str, Any] = {}
    res_3b_prot: Dict[str, Any] = {}
    if not args.skip_3b:
        print("\n[Step 3B-PK] Running Zero-Winner's-Curse Small-N PK + Allometric Bodyweight Unification...", flush=True)
        res_3b_pk = _run_smalln_pk_allometric_and_tabpfn()

        print("\n[Step 3B-Prot20] Running 20D RDKit Proteolytic Cleavage-Site x Shielding Ablation on half_life_reg...", flush=True)
        res_3b_prot = _run_halflife_proteolytic_ablation()

    out_doc = {
        "schema": "v333_step3_multitask_smalln_pk_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "tabpfn_enabled": G_TABPFN_OK,
        "part3a_lineage_transfer": res_3a_lineage,
        "part3a_mbc_mic_cascade": res_3a_cascade,
        "part3b_smalln_pk": res_3b_pk,
        "part3b_halflife_prot20": res_3b_prot,
        "elapsed_sec": round(time.time() - t_all, 2),
    }
    out_path = OUT_DIR / "step3_multitask_smalln_pk_results.json"
    out_path.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Step 3 DONE] Wrote {out_path} in {out_doc['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
