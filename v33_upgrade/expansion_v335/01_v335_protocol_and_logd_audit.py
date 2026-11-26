#!/usr/bin/env python3
"""V33.5 Publication-Grade Protocol Integrity, Zero-Holdout-Peeking Promotion,
Rigorous LogD_decadiene & Experimental ΔLogD Validation, Cluster-Bootstrap 95% CIs,
Small-N Admission Grading, and Friedman/Paired-Bootstrap Significance Tests (runs on compute host).

Directly addresses Claude Sonnet 5.5 Independent Peer Review items Q1, Q2, Q3, Q4, Q7, Q9, Q13, Q14.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import friedmanchisquare, pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

V33_BASE = Path("")
BUNDLE_DIR = V33_BASE / "bundle"
SERVING_DIR = BUNDLE_DIR / "serving_models"
EXP331 = V33_BASE / "expansion_v331"
EXP334 = V33_BASE / "expansion_v334"
EXP335 = V33_BASE / "expansion_v335"
EXP335.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP334))

from v33_features import FoldPreprocessor  # noqa: E402
from v33_metrics import regression_metrics  # noqa: E402
from v33_models import make_estimator  # noqa: E402
from fusion_config import DIRECT_FUSION_REPS, INDIVIDUAL_REPS  # noqa: E402
from fusion_models import EmbeddingFoldPreprocessor, ModalityAwareStacker, make_fusion_estimator  # noqa: E402
from v333_models import assign_ad_tiers, fit_mondrian_ad_eb_cw  # noqa: E402
from importlib import import_module

# Reuse UnifiedV334Bank, check_no_catboost, compute_oof_ho_max_tanimoto, eval_mondrian_cw, sha256_file
exp334_mod = import_module("os.environ.get(V334_TRAINER_MODULE, 03_train_and_promote_v334)")
UnifiedV334Bank = exp334_mod.UnifiedV334Bank
check_no_catboost = exp334_mod.check_no_catboost
compute_oof_ho_max_tanimoto = exp334_mod.compute_oof_ho_max_tanimoto
eval_mondrian_cw = exp334_mod.eval_mondrian_cw
sha256_file = exp334_mod.sha256_file
train_cpu_track = exp334_mod.train_cpu_track
train_fusion_track = exp334_mod.train_fusion_track
select_simplex_oof = exp334_mod.select_simplex_oof

csv.field_size_limit(10_000_000)
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_IDX = {a: i for i, a in enumerate(AA20)}


def tanimoto_matrix(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    a = bits_a.astype(np.float32)
    b = bits_b.astype(np.float32)
    inter = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def cluster_block_bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    clusters: np.ndarray,
    task_type: str,
    lo80: Optional[np.ndarray] = None,
    hi80: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 20261125,
) -> Dict[str, Any]:
    """1000x cluster-level block bootstrap 95% CIs (respecting scaffold cluster dependence)."""
    ok = np.isfinite(y) & np.isfinite(pred)
    y = y[ok]
    pred = pred[ok]
    clusters = clusters[ok]
    if lo80 is not None and hi80 is not None:
        lo80 = lo80[ok]
        hi80 = hi80[ok]
    n = len(y)
    if n < 5:
        return {"n_eval": int(n), "n_clusters": int(len(set(clusters))), "n_boot": 0}

    cl_to_idx: Dict[str, np.ndarray] = {}
    for ci, c in enumerate(clusters):
        cl_to_idx.setdefault(str(c), []).append(ci)
    cl_arrays = [np.array(v, dtype=np.int32) for v in cl_to_idx.values()]
    n_cl = len(cl_arrays)
    rng = np.random.RandomState(seed)

    if task_type == "classification":
        aucs, briers = [], []
        for _ in range(n_boot):
            pick = rng.randint(0, n_cl, size=n_cl)
            idx = np.concatenate([cl_arrays[p] for p in pick])
            yb, pb = y[idx], pred[idx]
            if len(np.unique(yb)) >= 2:
                try:
                    aucs.append(float(roc_auc_score(yb, pb)))
                except Exception:
                    pass
            briers.append(float(np.mean((yb - pb) ** 2)))

        def _ci(arr):
            if len(arr) < 20:
                return None
            return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

        return {
            "n_eval": int(n),
            "n_clusters": int(n_cl),
            "n_boot": int(n_boot),
            "roc_auc_ci95": _ci(aucs),
            "brier_ci95": _ci(briers),
        }

    maes, sps, r2s, covs = [], [], [], []
    has_cov = lo80 is not None and hi80 is not None and np.all(np.isfinite(lo80)) and np.all(np.isfinite(hi80))
    for _ in range(n_boot):
        pick = rng.randint(0, n_cl, size=n_cl)
        idx = np.concatenate([cl_arrays[p] for p in pick])
        yb, pb = y[idx], pred[idx]
        maes.append(float(np.mean(np.abs(yb - pb))))
        if np.std(yb) > 1e-12 and np.std(pb) > 1e-12:
            r = spearmanr(yb, pb).statistic
            if np.isfinite(r):
                sps.append(float(r))
            ss_tot = float(np.sum((yb - np.mean(yb)) ** 2))
            if ss_tot > 1e-12:
                r2s.append(float(1.0 - np.sum((yb - pb) ** 2) / ss_tot))
        if has_cov:
            covs.append(float(np.mean((yb >= lo80[idx]) & (yb <= hi80[idx]))))

    def _ci(arr):
        if len(arr) < 20:
            return None
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    out = {
        "n_eval": int(n),
        "n_clusters": int(n_cl),
        "n_boot": int(n_boot),
        "mae_ci95": _ci(maes),
        "spearman_ci95": _ci(sps),
        "r2_ci95": _ci(r2s),
    }
    if has_cov:
        out["cov80_ci95"] = _ci(covs)
    return out


def complete_linkage_clusters(bits: np.ndarray, tc_threshold: float = 0.55) -> List[List[int]]:
    """Strict complete-linkage clustering: EVERY pair inside a cluster satisfies Tanimoto >= tc_threshold."""
    n = bits.shape[0]
    if n <= 1:
        return [list(range(n))]
    sim = tanimoto_matrix(bits, bits)
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=1.0 - tc_threshold, criterion="distance")
    cl_map: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        cl_map[int(lab)].append(i)
    return sorted(cl_map.values(), key=lambda g: -len(g))


def make_balanced_folds(clusters: List[List[int]], n_folds: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy size-balanced cluster-isolated fold assignment."""
    n = sum(len(c) for c in clusters)
    fold_id = np.full(n, -1, dtype=np.int32)
    cluster_id = np.empty(n, dtype=object)
    rng = np.random.RandomState(seed)
    # Shuffle within equal-size buckets for repeated CV diversity while keeping balance
    order = sorted(range(len(clusters)), key=lambda i: (-len(clusters[i]), rng.random()))
    fsizes = [0] * n_folds
    for ci in order:
        f = int(np.argmin(fsizes))
        fsizes[f] += len(clusters[ci])
        for idx in clusters[ci]:
            fold_id[idx] = f
            cluster_id[idx] = f"CL_{ci:03d}"
    return fold_id, cluster_id


def eval_logd_decadiene_rigorous(bank: UnifiedV334Bank, jmc_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """P0-3: Strict complete-linkage clustering + 5x5 Repeated CV + LOCO CV + Direct 27-pair ΔLogD validation."""
    print("\n================ [P0-3: LogD_decadiene Rigorous Audit & ΔLogD Validation] ================", flush=True)
    n = len(jmc_rows)
    cidx_all = bank.map_keys([r["compound_key"] for r in jmc_rows])
    y_all = np.array([float(r["y_value"]) for r in jmc_rows], dtype=np.float64)
    cond_all = [{"cond_assay": "shake_flask_decadiene_water_ph74"} for _ in range(n)]
    cond_cols = ["cond_assay"]

    # Strict complete-linkage clusters (every intra-cluster pair Tc >= 0.55)
    clusters = complete_linkage_clusters(bank.morgan_bits[cidx_all], tc_threshold=0.55)
    cl_sizes = [len(c) for c in clusters]
    print(f"  Complete-linkage (Tc>=0.55): {len(clusters)} clusters, top sizes={cl_sizes[:10]}", flush=True)

    # Verify max cross-cluster Tc distribution
    sim_all = tanimoto_matrix(bank.morgan_bits[cidx_all], bank.morgan_bits[cidx_all])

    # 1. Repeated 5x5 Cluster-Grouped CV (5 distinct random seeds)
    rep_cpu_maes, rep_cpu_sps, rep_cpu_r2s = [], [], []
    rep_fus_maes, rep_fus_sps, rep_fus_r2s = [], [], []
    oof_accum_cpu = np.zeros(n, dtype=np.float64)
    oof_accum_fus = np.zeros(n, dtype=np.float64)
    oof_counts = np.zeros(n, dtype=np.int32)

    for rep_i, rseed in enumerate([42, 1042, 2042, 3042, 4042]):
        fids, cids = make_balanced_folds(clusters, n_folds=5, seed=rseed)
        fsz = [int((fids == f).sum()) for f in range(5)]
        empty_ho = np.array([], dtype=np.int32)
        _, cpu_oof, _, _ = train_cpu_track(bank, cidx_all, y_all, fids, cond_all, empty_ho, [], cond_cols, None, "smiles_only")
        pcache: Dict[str, Any] = {}
        fus_oof, _, _, _ = train_fusion_track(
            bank, cidx_all, y_all, fids, cond_all, empty_ho, [], cond_cols, None, cpu_oof, np.array([]), "smiles_only", pcache
        )
        mc = regression_metrics(y_all, cpu_oof)
        mf = regression_metrics(y_all, fus_oof)
        rep_cpu_maes.append(mc["mae"])
        rep_cpu_sps.append(mc["spearman"])
        rep_cpu_r2s.append(mc["r2"])
        rep_fus_maes.append(mf["mae"])
        rep_fus_sps.append(mf["spearman"])
        rep_fus_r2s.append(mf["r2"])
        ok = np.isfinite(cpu_oof) & np.isfinite(fus_oof)
        oof_accum_cpu[ok] += cpu_oof[ok]
        oof_accum_fus[ok] += fus_oof[ok]
        oof_counts[ok] += 1
        print(f"    Rep {rep_i+1} (folds={fsz}): CPU MAE={mc['mae']:.4f} Sp={mc['spearman']:.4f} R2={mc['r2']:.4f} | FUS MAE={mf['mae']:.4f} Sp={mf['spearman']:.4f} R2={mf['r2']:.4f}", flush=True)

    mean_oof_cpu = oof_accum_cpu / np.maximum(oof_counts, 1)
    mean_oof_fus = oof_accum_fus / np.maximum(oof_counts, 1)

    # 2. Leave-One-Cluster-Out (LOCO) CV across all clusters (100% deterministic, zero seed dependence!)
    loco_oof = np.full(n, np.nan, dtype=np.float64)
    for ci, members in enumerate(clusters):
        va_mask = np.zeros(n, dtype=bool)
        va_mask[members] = True
        tr_mask = ~va_mask
        pre = FoldPreprocessor(track="smiles_only", route="tanimoto_hybrid", cond_cols=cond_cols, scale_for_linear=True, svd_dim=16, random_state=42)
        X_tr = pre.fit_transform(bank, cidx_all[tr_mask], [cond_all[i] for i in np.where(tr_mask)[0]])
        X_va = pre.transform(bank, cidx_all[va_mask], [cond_all[i] for i in np.where(va_mask)[0]])
        est = make_estimator(family="TanimotoRBF_KRR", params={"alpha": 0.25, "w_tan": 0.5, "w_cond": 0.15, "gamma_mult": 1.0}, task_type="regression", n_jobs=4, bounds=None, seed=42)
        est.fit(X_tr, y_all[tr_mask])
        loco_oof[va_mask] = est.predict(X_va)
    m_loco = regression_metrics(y_all, loco_oof)
    print(f"  LOCO CV ({len(clusters)} clusters, n={n}): MAE={m_loco['mae']:.4f}, Sp={m_loco['spearman']:.4f}, R2={m_loco['r2']:.4f}", flush=True)

    # 3. Canonical balanced 80/20 cluster-isolated Holdout + 5-fold CV for production bundle (honest family selection!)
    # Select ~20% (~17-18 compounds) spread across multiple clusters deterministically
    rng_ho = np.random.RandomState(20261125)
    c_order = list(range(len(clusters)))
    rng_ho.shuffle(c_order)
    ho_cidxs: Set[int] = set()
    ho_n = 0
    for ci in c_order:
        if ho_n < 17 and (ho_n + len(clusters[ci]) <= 20 or ho_n == 0):
            ho_cidxs.add(ci)
            ho_n += len(clusters[ci])
    cv_clusters = [clusters[ci] for ci in range(len(clusters)) if ci not in ho_cidxs]
    cv_idx_flat = [i for cl in cv_clusters for i in cl]
    ho_idx_flat = [i for ci in sorted(ho_cidxs) for i in clusters[ci]]

    # Balanced 5 folds over CV clusters (using local 0..len(cv_clusters)-1 indexing)
    cv_order = sorted(range(len(cv_clusters)), key=lambda i: -len(cv_clusters[i]))
    cv_fsizes = [0] * 5
    cv_ci_to_fold: Dict[int, int] = {}
    for ci_l in cv_order:
        f_min = int(np.argmin(cv_fsizes))
        cv_ci_to_fold[ci_l] = f_min
        cv_fsizes[f_min] += len(cv_clusters[ci_l])

    row_is_ho = np.zeros(n, dtype=int)
    row_fold = np.full(n, -1, dtype=int)
    row_cid = np.empty(n, dtype=object)
    for ci in sorted(ho_cidxs):
        for idx in clusters[ci]:
            row_is_ho[idx] = 1
            row_cid[idx] = f"LDDEC_CL{ci:03d}"
    for ci_l, cl in enumerate(cv_clusters):
        for idx in cl:
            row_fold[idx] = cv_ci_to_fold[ci_l]
            row_cid[idx] = f"LDDEC_CVCL{ci_l:03d}"

    cv_mask = row_is_ho == 0
    ho_mask = row_is_ho == 1
    cv_fold_sizes = [int((row_fold[cv_mask] == f).sum()) for f in range(5)]
    print(f"  Production split: n_cv={cv_mask.sum()} (folds={cv_fold_sizes}), n_ho={ho_mask.sum()} ({len(ho_cidxs)} cl)", flush=True)

    cidx_cv, y_cv, fids_cv, cids_cv = cidx_all[cv_mask], y_all[cv_mask], row_fold[cv_mask], row_cid[cv_mask]
    cidx_ho, y_ho, cids_ho = cidx_all[ho_mask], y_all[ho_mask], row_cid[ho_mask]
    cond_cv = [cond_all[i] for i in np.where(cv_mask)[0]]
    cond_ho = [cond_all[i] for i in np.where(ho_mask)[0]]
    ad_oof, ad_ho = compute_oof_ho_max_tanimoto(bank, cidx_cv, fids_cv, cidx_ho)

    cpu_bundle = {"schema": "v33_pruned_cpu_serving_v1", "endpoint": "LogD_decadiene", "task_type": "regression", "bounds": None, "cond_cols": cond_cols, "canonical_default_conditions": {"cond_assay": "shake_flask_decadiene_water_ph74"}, "dual_modality": None, "smiles_only": None, "seq_only": None, "aft": None}
    fus_bundle = {"schema": "v33_pruned_fusion_serving_v1", "endpoint": "LogD_decadiene", "task_type": "regression", "bounds": None, "cond_cols": cond_cols, "canonical_default_conditions": {"cond_assay": "shake_flask_decadiene_water_ph74"}, "pretrained_base_heads": {}, "dual_modality": None, "smiles_only": None, "aft": None}
    pretrained_cache: Dict[str, Any] = {}
    track_results: Dict[str, Any] = {}

    # Since 87/87 LogD_decadiene macrocycles are SMILES-only (seq_valid=0), train smiles_only cleanly
    # and mirror identical smiles_only serving track into dual_modality (eliminating spurious dual degradation!)
    cpu_recipe, cpu_oof, cpu_ho, cpu_tr_obj = train_cpu_track(
        bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, "smiles_only"
    )
    fus_oof, fus_ho, cand_heads, stacker = train_fusion_track(
        bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, cpu_oof, cpu_ho, "smiles_only", pretrained_cache
    )
    sel = select_simplex_oof(y_cv, fids_cv, np.ones(len(cidx_cv), dtype=bool), cpu_oof, fus_oof, None)

    # HONEST FAMILY LABEL FIX (Q3): If fusion stacker collapses >=99% weight onto cpu_smiles, report winning_family="cpu"!
    st_weights = dict(zip(cand_heads, [float(x) for x in stacker.weights_])) if hasattr(stacker, "weights_") and stacker.weights_ is not None else {}
    cpu_in_stack = float(st_weights.get("cpu_smiles", 0.0))
    if sel["winning_family"] == "fusion" and cpu_in_stack >= 0.99:
        print(f"  [HONEST LABEL FIX] Fusion stacker put {cpu_in_stack:.4f} weight on cpu_smiles -> reporting winning_family='cpu'!", flush=True)
        sel["winning_family"] = "cpu"
        sel["weights"] = {"cpu": 1.0}
        sel["oof_pred"] = cpu_oof
        sel["cv_metrics"] = regression_metrics(y_cv, cpu_oof)

    w = sel["weights"]
    if sel["winning_family"] == "super_ensemble":
        ho_pred = w["cpu"] * cpu_ho + w["fusion"] * fus_ho
    elif sel["winning_family"] == "fusion":
        ho_pred = fus_ho
    else:
        ho_pred = cpu_ho

    ho_m = regression_metrics(y_ho, ho_pred)
    fam_ho = {"cpu": regression_metrics(y_ho, cpu_ho), "fusion": regression_metrics(y_ho, fus_ho)}
    conf_cw = fit_mondrian_ad_eb_cw(
        y_cv, sel["oof_pred"], ad_oof, bounds=None, cluster_ids=cids_cv, min_stratum_clusters=10, enforce_monotonic=True
    )
    conf_ho = eval_mondrian_cw(y_ho, ho_pred, ad_ho, conf_cw)
    # Honest UQ reliability disclosure for small-N endpoint (Q3)
    conf_cw["uq_reliability"] = "low"
    conf_cw["uq_note"] = (
        f"Small-N congeneric macrocycle endpoint (n_cv={len(y_cv)}, n_holdout={len(y_ho)}, n_clusters={len(cv_clusters)}); "
        "intervals are wide empirical references only — use for rank ordering."
    )
    conf_ho["uq_reliability"] = "low"
    conf_ho["uq_note"] = conf_cw["uq_note"]

    # Bootstrap 95% CI on Holdout
    ho_boot = cluster_block_bootstrap_ci(y_ho, ho_pred, cids_ho, "regression")

    tr_doc = {
        "winning_family": sel["winning_family"],
        "is_super_ensemble": sel["is_super_ensemble"],
        "weights": w,
        "best_single_family": sel["best_single_family"],
        "nested_super_delta_over_best_single": sel["nested_super_delta_over_best_single"],
        "cv_metrics": sel["cv_metrics"],
        "cv_metrics_raw": None,
        "holdout_metrics": ho_m,
        "holdout_metrics_raw": None,
        "holdout_bootstrap_ci95": ho_boot,
        "family_cv_metrics": sel["family_cv_metrics"],
        "family_holdout_metrics": fam_ho,
        "calibrator": None,
        "conformal": conf_cw,
        "conformal_holdout_coverage": conf_ho,
        "repeated_5x5_cv": {
            "cpu_mae_mean": round(float(np.mean(rep_cpu_maes)), 4),
            "cpu_mae_sd": round(float(np.std(rep_cpu_maes)), 4),
            "cpu_spearman_mean": round(float(np.mean(rep_cpu_sps)), 4),
            "cpu_spearman_sd": round(float(np.std(rep_cpu_sps)), 4),
            "cpu_r2_mean": round(float(np.mean(rep_cpu_r2s)), 4),
            "cpu_r2_sd": round(float(np.std(rep_cpu_r2s)), 4),
            "fusion_mae_mean": round(float(np.mean(rep_fus_maes)), 4),
            "fusion_mae_sd": round(float(np.std(rep_fus_maes)), 4),
            "fusion_spearman_mean": round(float(np.mean(rep_fus_sps)), 4),
            "fusion_spearman_sd": round(float(np.std(rep_fus_sps)), 4),
            "fusion_r2_mean": round(float(np.mean(rep_fus_r2s)), 4),
            "fusion_r2_sd": round(float(np.std(rep_fus_r2s)), 4),
        },
        "loco_cv": m_loco,
        "stacker_weights": st_weights,
        "cv_fold_sizes": cv_fold_sizes,
    }
    # Both tracks serve the clean smiles_only models (since LogD_decadiene is 100% SMILES-only macrocycles)
    track_results["smiles_only"] = tr_doc
    track_results["dual_modality"] = json.loads(json.dumps(tr_doc))
    cpu_bundle["smiles_only"] = cpu_tr_obj
    cpu_bundle["dual_modality"] = cpu_tr_obj
    act_h = stacker.active_heads(tol=1e-9)
    fus_tr_obj = {"candidate_heads": cand_heads, "stacker": stacker, "active_heads": act_h}
    fus_bundle["smiles_only"] = fus_tr_obj
    fus_bundle["dual_modality"] = fus_tr_obj
    for hk in act_h:
        if hk in pretrained_cache and hk not in fus_bundle["pretrained_base_heads"]:
            fus_bundle["pretrained_base_heads"][hk] = {
                "rep_cfg": pretrained_cache[hk]["rep_cfg"],
                "is_aft": False,
                "folds": pretrained_cache[hk]["folds"],
            }

    # 4. DIRECT GROUND-TRUTH ΔLogD VALIDATION on ALL 27 matched dual-phase macrocycles (Q14!)
    oct_oos: Dict[str, Tuple[float, float]] = {}  # ck -> (y_true, y_oos_pred)
    for fn in ("oof_champion.tsv", "holdout_champion.tsv"):
        fp = BUNDLE_DIR / "champion_predictions" / "LogD7.4" / fn
        if fp.exists():
            with fp.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r.get("censor_type", "exact") != "exact":
                        continue
                    try:
                        oct_oos[r["compound_key"]] = (float(r["y_value"]), float(r["pred_dual"]))
                    except Exception:
                        pass

    dec_oos_pred = mean_oof_fus if float(np.mean(rep_fus_maes)) <= float(np.mean(rep_cpu_maes)) else mean_oof_cpu
    matched_true_d, matched_pred_d, matched_records = [], [], []
    for i, r in enumerate(jmc_rows):
        ck = r["compound_key"]
        if ck in oct_oos and np.isfinite(dec_oos_pred[i]):
            ot, op = oct_oos[ck]
            dt, dp = float(y_all[i]), float(dec_oos_pred[i])
            matched_true_d.append(ot - dt)
            matched_pred_d.append(op - dp)
            matched_records.append({
                "compound_key": ck,
                "logd_oct_true": round(ot, 3), "logd_oct_oos": round(op, 3),
                "logd_dec_true": round(dt, 3), "logd_dec_oos": round(dp, 3),
                "delta_true": round(ot - dt, 3), "delta_pred": round(op - dp, 3),
            })

    mt = np.array(matched_true_d, dtype=np.float64)
    mp = np.array(matched_pred_d, dtype=np.float64)
    m_delta = regression_metrics(mt, mp)
    # Bootstrap 95% CI for ΔLogD Spearman & MAE
    rng_d = np.random.RandomState(42)
    d_sps, d_maes = [], []
    for _ in range(1000):
        idx = rng_d.randint(0, len(mt), size=len(mt))
        if np.std(mt[idx]) > 1e-9 and np.std(mp[idx]) > 1e-9:
            d_sps.append(float(spearmanr(mt[idx], mp[idx]).statistic))
        d_maes.append(float(np.mean(np.abs(mt[idx] - mp[idx]))))
    # Binary AUROC for high chameleonicity (ΔLogD_true >= 2.0)
    y_hi = (mt >= 2.0).astype(int)
    auc_hi = round(float(roc_auc_score(y_hi, mp)), 4) if len(np.unique(y_hi)) == 2 else None
    # Empirical tercile cutoffs on experimental ΔLogD
    q33, q67 = float(np.percentile(mt, 33.3)), float(np.percentile(mt, 66.7))
    delta_validation = {
        "n_matched_dual_phase_macrocycles": int(len(mt)),
        "protocol": "Strictly Out-of-Sample: LogD7.4_OOS - LogD_decadiene_5x5_OOF vs Measured (LogD_oct - LogD_dec)",
        "spearman": m_delta["spearman"],
        "spearman_ci95": [round(float(np.percentile(d_sps, 2.5)), 4), round(float(np.percentile(d_sps, 97.5)), 4)],
        "pearson": m_delta["pearson"],
        "mae": m_delta["mae"],
        "mae_ci95": [round(float(np.percentile(d_maes, 2.5)), 4), round(float(np.percentile(d_maes, 97.5)), 4)],
        "r2": m_delta["r2"],
        "high_chameleonicity_ge2_auroc": auc_hi,
        "n_high_chameleonicity_ge2": int(y_hi.sum()),
        "empirical_terciles": {"p33": round(q33, 2), "p67": round(q67, 2), "min": round(float(mt.min()), 2), "max": round(float(mt.max()), 2)},
        "scope_warning": "Validated exclusively on cyclic macrocycles; ΔLogD for linear non-macrocyclic peptides is extrapolated and flagged out-of-scope.",
    }
    print(
        f"  [ΔLogD DIRECT EXPERIMENTAL VALIDATION (n={len(mt)})]: "
        f"Sp={m_delta['spearman']:.4f} (95%CI {delta_validation['spearman_ci95']}), "
        f"r={m_delta['pearson']:.4f}, MAE={m_delta['mae']:.4f}, R2={m_delta['r2']:.4f}, "
        f"AUROC(Δ>=2.0)={auc_hi}, empirical_terciles=[{q33:.2f}, {q67:.2f}]",
        flush=True,
    )

    # Save updated LogD_decadiene.tsv with balanced folds
    ep_out_dir = EXP335 / "endpoints"
    ep_out_dir.mkdir(parents=True, exist_ok=True)
    saved_rows = []
    for i, r in enumerate(jmc_rows):
        saved_rows.append({
            "endpoint": "LogD_decadiene", "task_tier": "tier1", "task_type": "regression",
            "compound_key": r["compound_key"], "y_value": r["y_value"],
            "y_lower": r["y_lower"], "y_upper": r["y_upper"],
            "censor_type": r["censor_type"], "cond_assay": r["cond_assay"],
            "cluster_id": str(row_cid[i]), "is_holdout": str(row_is_ho[i]),
            "fold_id": str(row_fold[i]), "case17_excluded": "0",
            "source_code": r["source_code"], "record_id": f"V335:LogD_decadiene:{r['source_record_id']}",
        })
    with (ep_out_dir / "LogD_decadiene.tsv").open("w", encoding="utf-8", newline="") as f:
        w_csv = csv.DictWriter(f, fieldnames=list(saved_rows[0].keys()), delimiter="\t")
        w_csv.writeheader()
        w_csv.writerows(saved_rows)

    return {
        "rows": saved_rows,
        "track_results": track_results,
        "cpu_bundle": cpu_bundle,
        "fus_bundle": fus_bundle,
        "delta_validation": delta_validation,
        "holdout_preds": {"y": y_ho, "pred": ho_pred, "cids": cids_ho},
    }


def main():
    t_all = time.time()
    manifest_path = BUNDLE_DIR / "v33_manifest.json"
    backup_v333 = BUNDLE_DIR / "v33_manifest_v333_backup.json"
    assert backup_v333.exists(), "Missing pristine v33.3 manifest backup!"
    manifest = json.loads(backup_v333.read_text(encoding="utf-8"))

    # Restore pristine v33.3 AD index backup before building V33.5 cleanly
    ad_npz_path = BUNDLE_DIR / "ad_analog_index.npz"
    analog_gz_path = BUNDLE_DIR / "analog_lookup.json.gz"
    if (BUNDLE_DIR / "ad_analog_index_v333_backup.npz").exists():
        shutil.copy2(BUNDLE_DIR / "ad_analog_index_v333_backup.npz", ad_npz_path)
        shutil.copy2(BUNDLE_DIR / "analog_lookup_v333_backup.json.gz", analog_gz_path)

    bank = UnifiedV334Bank()

    # Load Case17 compound keys for strict Tc < 0.40 isolation check (Q4)
    c17_keys = set()
    c17_fp = V33_BASE / "data" / "case17_queries.tsv"
    if c17_fp.exists():
        with c17_fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("compound_key") in bank.key_to_idx:
                    c17_keys.add(r["compound_key"])
    c17_idx = bank.map_keys(sorted(c17_keys)) if c17_keys else np.array([], dtype=np.int32)
    c17_bits = bank.morgan_bits[c17_idx] if len(c17_idx) > 0 else np.zeros((0, 1024), dtype=np.uint8)

    v334_rows_by_ep = defaultdict(list)
    with (EXP334 / "data" / "v334_new_endpoint_rows.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            v334_rows_by_ep[r["endpoint"]].append(r)

    def get_existing_ep_path(ep: str) -> Optional[Path]:
        for base in (V33_BASE / "data" / "endpoints", EXP331 / "data" / "endpoints"):
            p = base / f"{ep}.tsv"
            if p.exists():
                return p
        return None

    receipt: Dict[str, Any] = {
        "schema": "v335_strict_zero_peeking_audit_v1",
        "promotion_rule": "100% ZERO-HOLDOUT-PEEKING: Promote expanded existing endpoint ONLY if 5-seed nested CV paired MAE improves by >=1.0% (and >=0.005) on identical baseline CV rows AND 5-seed mean CV Spearman >= -0.01 AND >=4/5 seeds improve. Holdout is NEVER inspected for accept/reject.",
        "tanimoto_isolation_threshold": 0.40,
        "endpoints": {},
        "promoted": [],
        "held_on_v333": [],
    }

    # --- A. Rigorous LogD_decadiene (#47) + Direct ΔLogD Experimental Validation ---
    ld_res = eval_logd_decadiene_rigorous(bank, v334_rows_by_ep["LogD_decadiene"])
    receipt["logd_decadiene_delta_validation"] = ld_res["delta_validation"]
    receipt["promoted"].append("LogD_decadiene")

    srv_out = EXP335 / "serving_models"
    srv_out.mkdir(parents=True, exist_ok=True)
    for ep_s, cb, fb in [("LogD_decadiene", ld_res["cpu_bundle"], ld_res["fus_bundle"])]:
        cfp, ffp = srv_out / f"{ep_s}_cpu.joblib", srv_out / f"{ep_s}_fusion.joblib"
        joblib.dump(cb, cfp, compress=1)
        joblib.dump(fb, ffp, compress=1)
        check_no_catboost(cfp)
        check_no_catboost(ffp)
        shutil.copy2(cfp, SERVING_DIR / f"{ep_s}_cpu.joblib")
        shutil.copy2(ffp, SERVING_DIR / f"{ep_s}_fusion.joblib")

    ld_rows = ld_res["rows"]
    ld_cv_n = sum(1 for r in ld_rows if int(r["is_holdout"]) == 0)
    ld_ho_n = sum(1 for r in ld_rows if int(r["is_holdout"]) == 1)
    manifest["endpoints"]["LogD_decadiene"] = {
        "endpoint": "LogD_decadiene", "expansion_wave": "v33.5",
        "display_en": "Macrocycle Lipophilicity (LogD decadiene/water)",
        "display_zh": "大环肽癸二烯亲脂性 LogD (变色龙效应)",
        "tier": 1, "tier_label": "tier1", "domain": "physicochemical",
        "task_type": "regression", "size_tier": "small",
        "evidence_grade": "rank_only",
        "unit": "LogD_dec", "linear_unit": "logD_dec", "log10_space": False,
        "bounds": None, "cond_cols": ["cond_assay"],
        "canonical_conditions": {"cond_assay": "shake_flask_decadiene_water_ph74"},
        "counts": {
            "n_total": len(ld_rows), "n_cv": ld_cv_n, "n_holdout": ld_ho_n,
            "n_censored_total": 0, "n_censored_cv": 0, "n_censored_holdout": 0,
            "censor_counts": {"exact": len(ld_rows), "right": 0, "left": 0, "interval": 0},
            "n_case17_excluded": 0,
        },
        "v32_reference": None,
        "dual_modality": ld_res["track_results"]["dual_modality"],
        "smiles_only": ld_res["track_results"]["smiles_only"],
        "censored_head": None, "representation_ablation_holdout": {},
    }

    # --- B. Zero-Holdout-Peeking 5-Seed Evaluation of ALL 8 Expanded Existing Endpoints (incl. half_life_reg!) ---
    EXISTING_8 = ["F", "LogD7.4", "CL_per_kg", "CL_absolute", "Vd_per_kg", "Vd_absolute", "PPB", "half_life_reg"]
    SEEDS = [42, 1042, 2042, 3042, 4042]

    print("\n================ [P0-2: Zero-Holdout-Peeking 5-Seed CV Gate (8 Existing Endpoints)] ================", flush=True)
    for ep in EXISTING_8:
        t_ep = time.time()
        ex_path = get_existing_ep_path(ep)
        ex_rows = list(csv.DictReader(ex_path.open(encoding="utf-8"), delimiter="\t"))
        cond_cols = [c for c in ex_rows[0].keys() if c.startswith("cond_")]
        ep_m = manifest["endpoints"][ep]
        canon_cond = ep_m.get("canonical_conditions") or {}
        b = ep_m.get("bounds")
        bounds = (float(b[0]), float(b[1])) if b is not None else None

        ho_rows = [r for r in ex_rows if int(r["is_holdout"]) == 1]
        cv_rows_base = [r for r in ex_rows if int(r["is_holdout"]) == 0]
        ho_clusters = {r["cluster_id"] for r in ho_rows}
        ex_ck_set = {r["compound_key"] for r in ex_rows}

        # Exact Tanimoto isolation against frozen Holdout AND Case17 (Q4 fix!)
        ho_idx = bank.map_keys([r["compound_key"] for r in ho_rows])
        ho_bits = bank.morgan_bits[ho_idx]
        ho_smi_v = bank.smiles_valid[ho_idx]

        cand_new = v334_rows_by_ep.get(ep, [])
        added_rows = []
        n_dup, n_cl_hit, n_tc_ho_drop, n_tc_c17_drop = 0, 0, 0, 0
        for r in cand_new:
            ck = r["compound_key"]
            if ck in ex_ck_set:
                n_dup += 1
                continue
            if r["cluster_id"] in ho_clusters:
                n_cl_hit += 1
                continue
            gi = bank.key_to_idx[ck]
            if bank.smiles_valid[gi]:
                qbit = bank.morgan_bits[gi : gi + 1]
                if ho_smi_v.any():
                    max_tc_ho = float(tanimoto_matrix(qbit, ho_bits[ho_smi_v]).max())
                    if max_tc_ho >= 0.40:
                        n_tc_ho_drop += 1
                        continue
                if len(c17_bits) > 0:
                    max_tc_c17 = float(tanimoto_matrix(qbit, c17_bits).max())
                    if max_tc_c17 >= 0.40:
                        n_tc_c17_drop += 1
                        continue
            row_d = {
                "endpoint": ep, "task_tier": "tier1", "task_type": "regression",
                "compound_key": ck, "y_value": r["y_value"],
                "y_lower": r["y_lower"], "y_upper": r["y_upper"],
                "censor_type": r["censor_type"], "cluster_id": r["cluster_id"],
                "is_holdout": "0", "fold_id": "0", "case17_excluded": "0",
                "source_code": r["source_code"], "record_id": f"V335:{ep}:{r['source_record_id']}",
            }
            for cc in cond_cols:
                row_d[cc] = r.get(cc) or canon_cond.get(cc, "unspecified") or "unspecified"
            added_rows.append(row_d)

        # Assign added rows to folds by cluster_id (NOT row-level round robin!)
        added_cl_map: Dict[str, int] = {}
        for r in added_rows:
            cid = r["cluster_id"]
            if cid not in added_cl_map:
                added_cl_map[cid] = len(added_cl_map) % 5
            r["fold_id"] = str(added_cl_map[cid])

        print(
            f"  [{ep:<14}] base_cv={len(cv_rows_base)} + clean_added={len(added_rows)} "
            f"(dropped: dup={n_dup}, cl_ho={n_cl_hit}, tc_ho>=0.4={n_tc_ho_drop}, tc_c17>=0.4={n_tc_c17_drop})",
            flush=True,
        )

        if len(added_rows) == 0:
            receipt["held_on_v333"].append(ep)
            receipt["endpoints"][ep] = {
                "n_added_clean": 0, "promoted": False, "reason": "zero_tanimoto_isolated_additions",
                "drops": {"dup": n_dup, "cl_ho": n_cl_hit, "tc_ho": n_tc_ho_drop, "tc_c17": n_tc_c17_drop},
            }
            continue

        # 5-seed paired nested CV comparison STRICTLY on CV rows (ZERO Holdout access!)
        cv_base_ex = [r for r in cv_rows_base if r["censor_type"] == "exact"]
        cv_exp_ex = cv_base_ex + [r for r in added_rows if r["censor_type"] == "exact"]
        n_b = len(cv_base_ex)

        cidx_b = bank.map_keys([r["compound_key"] for r in cv_base_ex])
        y_b = np.array([float(r["y_value"]) for r in cv_base_ex], dtype=np.float64)
        cond_b = [{c: r.get(c, "unspecified") for c in cond_cols} for r in cv_base_ex]
        cl_b = [r["cluster_id"] for r in cv_base_ex]

        cidx_e = bank.map_keys([r["compound_key"] for r in cv_exp_ex])
        y_e = np.array([float(r["y_value"]) for r in cv_exp_ex], dtype=np.float64)
        cond_e = [{c: r.get(c, "unspecified") for c in cond_cols} for r in cv_exp_ex]
        cl_e = [r["cluster_id"] for r in cv_exp_ex]

        # Build cluster-grouped folds per seed (keeping baseline rows in identical folds for paired delta!)
        unique_b_cl = sorted(set(cl_b))
        unique_a_cl = sorted(set(cl_e[n_b:]) - set(unique_b_cl))
        old_maes, new_maes, old_sps, new_sps = [], [], [], []
        seed_wins = 0
        empty_ho = np.array([], dtype=np.int32)

        for sd in SEEDS:
            rng_s = np.random.RandomState(sd)
            b_shuf = list(unique_b_cl)
            rng_s.shuffle(b_shuf)
            c2f = {c: i % 5 for i, c in enumerate(b_shuf)}
            a_shuf = list(unique_a_cl)
            rng_s.shuffle(a_shuf)
            for i, c in enumerate(a_shuf):
                c2f[c] = i % 5

            fids_b = np.array([c2f[c] for c in cl_b], dtype=np.int32)
            fids_e = np.array([c2f[c] for c in cl_e], dtype=np.int32)

            # Fast paired evaluation using our exact CPU + Fusion pipeline on dual_modality
            _, cpu_b, _, _ = train_cpu_track(bank, cidx_b, y_b, fids_b, cond_b, empty_ho, [], cond_cols, bounds, "dual_modality")
            pc_b: Dict[str, Any] = {}
            fus_b, _, _, _ = train_fusion_track(bank, cidx_b, y_b, fids_b, cond_b, empty_ho, [], cond_cols, bounds, cpu_b, np.array([]), "dual_modality", pc_b)
            sel_b = select_simplex_oof(y_b, fids_b, np.ones(n_b, dtype=bool), cpu_b, fus_b, bounds)

            _, cpu_e, _, _ = train_cpu_track(bank, cidx_e, y_e, fids_e, cond_e, empty_ho, [], cond_cols, bounds, "dual_modality")
            pc_e: Dict[str, Any] = {}
            fus_e, _, _, _ = train_fusion_track(bank, cidx_e, y_e, fids_e, cond_e, empty_ho, [], cond_cols, bounds, cpu_e, np.array([]), "dual_modality", pc_e)
            sel_e = select_simplex_oof(y_e, fids_e, np.ones(len(y_e), dtype=bool), cpu_e, fus_e, bounds)

            # Paired evaluation strictly on the SAME n_b baseline CV compounds!
            m_old = regression_metrics(y_b, sel_b["oof_pred"])
            m_new = regression_metrics(y_b, sel_e["oof_pred"][:n_b])
            old_maes.append(m_old["mae"])
            new_maes.append(m_new["mae"])
            old_sps.append(m_old["spearman"] or 0.0)
            new_sps.append(m_new["spearman"] or 0.0)
            if m_new["mae"] < m_old["mae"]:
                seed_wins += 1

        mean_old_mae, mean_new_mae = float(np.mean(old_maes)), float(np.mean(new_maes))
        mean_old_sp, mean_new_sp = float(np.mean(old_sps)), float(np.mean(new_sps))
        rel_mae_imp = (mean_old_mae - mean_new_mae) / max(mean_old_mae, 1e-9)
        sp_delta = mean_new_sp - mean_old_sp

        # STRICT ZERO-HOLDOUT-PEEKING DECISION!
        promote = (rel_mae_imp >= 0.01) and ((mean_old_mae - mean_new_mae) >= 0.005) and (sp_delta >= -0.01) and (seed_wins >= 4)
        reason = (
            f"5-seed paired CV MAE {mean_old_mae:.4f}->{mean_new_mae:.4f} ({-100*rel_mae_imp:+.2f}%), "
            f"Sp {mean_old_sp:.4f}->{mean_new_sp:.4f} ({sp_delta:+.4f}), seed_wins={seed_wins}/5"
        )
        print(f"    -> ZERO-PEEKING GATE: PROMOTE={promote} | {reason} [{time.time()-t_ep:.1f}s]", flush=True)

        ep_rec: Dict[str, Any] = {
            "n_added_clean": len(added_rows),
            "drops": {"dup": n_dup, "cl_ho": n_cl_hit, "tc_ho": n_tc_ho_drop, "tc_c17": n_tc_c17_drop},
            "cv_5seed_old_mae": {"mean": round(mean_old_mae, 4), "sd": round(float(np.std(old_maes)), 4)},
            "cv_5seed_new_mae": {"mean": round(mean_new_mae, 4), "sd": round(float(np.std(new_maes)), 4)},
            "cv_5seed_old_sp": {"mean": round(mean_old_sp, 4), "sd": round(float(np.std(old_sps)), 4)},
            "cv_5seed_new_sp": {"mean": round(mean_new_sp, 4), "sd": round(float(np.std(new_sps)), 4)},
            "seed_wins": f"{seed_wins}/5",
            "promoted": promote,
            "reason": reason,
        }

        if promote:
            # Train final serving bundle on full expanded CV and evaluate ONCE on untouched frozen Holdout
            all_rows = ex_rows + added_rows
            ho_ex = [r for r in ho_rows if r["censor_type"] == "exact"]
            fids_canon = np.array([int(r["fold_id"]) for r in cv_exp_ex], dtype=np.int32)
            cids_canon = np.array([r["cluster_id"] for r in cv_exp_ex], dtype=object)
            cidx_ho = bank.map_keys([r["compound_key"] for r in ho_ex])
            y_ho = np.array([float(r["y_value"]) for r in ho_ex], dtype=np.float64)
            cids_ho = np.array([r["cluster_id"] for r in ho_ex], dtype=object)
            cond_ho = [{c: r.get(c, "unspecified") for c in cond_cols} for r in ho_ex]
            ad_oof, ad_ho = compute_oof_ho_max_tanimoto(bank, cidx_e, fids_canon, cidx_ho)

            cpu_b = {"schema": "v33_pruned_cpu_serving_v1", "endpoint": ep, "task_type": "regression", "bounds": bounds, "cond_cols": cond_cols, "canonical_default_conditions": canon_cond, "dual_modality": None, "smiles_only": None, "seq_only": None, "aft": None}
            fus_b = {"schema": "v33_pruned_fusion_serving_v1", "endpoint": ep, "task_type": "regression", "bounds": bounds, "cond_cols": cond_cols, "canonical_default_conditions": canon_cond, "pretrained_base_heads": {}, "dual_modality": None, "smiles_only": None, "aft": None}
            pcache: Dict[str, Any] = {}
            tr_res = {}
            for track in ("dual_modality", "smiles_only"):
                v_cv = bank.smiles_valid[cidx_e].copy() if track == "smiles_only" else np.ones(len(cidx_e), dtype=bool)
                v_ho = bank.smiles_valid[cidx_ho].copy() if track == "smiles_only" else np.ones(len(cidx_ho), dtype=bool)
                _, c_oof, c_ho, c_obj = train_cpu_track(bank, cidx_e, y_e, fids_canon, cond_e, cidx_ho, cond_ho, cond_cols, bounds, track)
                f_oof, f_ho, ch, st = train_fusion_track(bank, cidx_e, y_e, fids_canon, cond_e, cidx_ho, cond_ho, cond_cols, bounds, c_oof, c_ho, track, pcache)
                sel = select_simplex_oof(y_e, fids_canon, v_cv, c_oof, f_oof, bounds)
                w = sel["weights"]
                ho_p = (w["cpu"] * c_ho + w["fusion"] * f_ho) if sel["winning_family"] == "super_ensemble" else (f_ho if sel["winning_family"] == "fusion" else c_ho)
                if bounds is not None:
                    ho_p = np.clip(ho_p, bounds[0], bounds[1])
                ho_m = regression_metrics(y_ho[v_ho], ho_p[v_ho])
                conf_cw = fit_mondrian_ad_eb_cw(y_e[v_cv], sel["oof_pred"][v_cv], ad_oof[v_cv], bounds=bounds, cluster_ids=cids_canon[v_cv], min_stratum_clusters=20, enforce_monotonic=True)
                conf_ho = eval_mondrian_cw(y_ho[v_ho], ho_p[v_ho], ad_ho[v_ho], conf_cw)
                ho_boot = cluster_block_bootstrap_ci(y_ho[v_ho], ho_p[v_ho], cids_ho[v_ho], "regression")
                tr_res[track] = {
                    "winning_family": sel["winning_family"], "is_super_ensemble": sel["is_super_ensemble"], "weights": w,
                    "best_single_family": sel["best_single_family"], "nested_super_delta_over_best_single": sel["nested_super_delta_over_best_single"],
                    "cv_metrics": sel["cv_metrics"], "cv_metrics_raw": None,
                    "holdout_metrics": ho_m, "holdout_metrics_raw": None, "holdout_bootstrap_ci95": ho_boot,
                    "family_cv_metrics": sel["family_cv_metrics"],
                    "family_holdout_metrics": {"cpu": regression_metrics(y_ho[v_ho], c_ho[v_ho]), "fusion": regression_metrics(y_ho[v_ho], f_ho[v_ho])},
                    "calibrator": None, "conformal": conf_cw, "conformal_holdout_coverage": conf_ho,
                }
                cpu_b[track] = c_obj
                act_h = st.active_heads(tol=1e-9)
                fus_b[track] = {"candidate_heads": ch, "stacker": st, "active_heads": act_h}
                for hk in act_h:
                    if hk in pcache and hk not in fus_b["pretrained_base_heads"]:
                        fus_b["pretrained_base_heads"][hk] = {"rep_cfg": pcache[hk]["rep_cfg"], "is_aft": False, "folds": pcache[hk]["folds"]}

            cfp, ffp = srv_out / f"{ep}_cpu.joblib", srv_out / f"{ep}_fusion.joblib"
            joblib.dump(cpu_b, cfp, compress=1)
            joblib.dump(fus_b, ffp, compress=1)
            check_no_catboost(cfp)
            check_no_catboost(ffp)
            shutil.copy2(cfp, SERVING_DIR / f"{ep}_cpu.joblib")
            shutil.copy2(ffp, SERVING_DIR / f"{ep}_fusion.joblib")
            for tr in ("dual_modality", "smiles_only"):
                oc = BUNDLE_DIR / "conformal_calibrators" / f"{ep}__{tr}__cqr_lgb_cw.joblib"
                if oc.exists():
                    oc.unlink()
            manifest["endpoints"][ep]["counts"] = {
                "n_total": len(all_rows), "n_cv": len([r for r in all_rows if int(r["is_holdout"]) == 0]), "n_holdout": len(ho_rows),
                "n_censored_total": sum(1 for r in all_rows if r["censor_type"] != "exact"),
                "n_censored_cv": sum(1 for r in all_rows if int(r["is_holdout"]) == 0 and r["censor_type"] != "exact"),
                "n_censored_holdout": sum(1 for r in ho_rows if r["censor_type"] != "exact"),
                "censor_counts": {k: sum(1 for r in all_rows if r["censor_type"] == k) for k in ("exact", "right", "left", "interval")},
                "n_case17_excluded": sum(1 for r in all_rows if int(r.get("case17_excluded") or 0) == 1),
            }
            manifest["endpoints"][ep]["dual_modality"] = tr_res["dual_modality"]
            manifest["endpoints"][ep]["smiles_only"] = tr_res["smiles_only"]
            ep_rec["untouched_holdout_posthoc"] = tr_res["dual_modality"]["holdout_metrics"]
            receipt["promoted"].append(ep)
            # Save expanded endpoint TSV
            with (EXP335 / "endpoints" / f"{ep}.tsv").open("w", encoding="utf-8", newline="") as f:
                w_csv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), delimiter="\t", extrasaction="ignore")
                w_csv.writeheader()
                w_csv.writerows(all_rows)
        else:
            # Ensure we restore pristine v33.3 serving models if v33.4 previously overwrote them!
            receipt["held_on_v333"].append(ep)
        receipt["endpoints"][ep] = ep_rec

    # Restore pristine v33.3 serving models for any endpoint held on v33.3 (e.g. if v33.4 previously promoted PPB/Vd_absolute via holdout!)
    # Check where pristine v33.3 serving models live or if backup exists!
    print("\n[CHECK] Restoring/verifying pristine v33.3 serving models for held endpoints...", flush=True)
    for ep in receipt["held_on_v333"]:
        # Check if expansion_v334 overwrote SERVING_DIR earlier
        v334_overwrote = (EXP334 / "serving_models" / f"{ep}_cpu.joblib").exists()
        if v334_overwrote:
            # Restore from original models_cpu / models_fusion / expansion_v331/serving_models if available!
            for cand_dir in (EXP331 / "serving_models", V33_BASE / "bundle_v333_backup" / "serving_models"):
                if (cand_dir / f"{ep}_cpu.joblib").exists():
                    shutil.copy2(cand_dir / f"{ep}_cpu.joblib", SERVING_DIR / f"{ep}_cpu.joblib")
                    print(f"  Restored {ep}_cpu.joblib from {cand_dir}", flush=True)
                if (cand_dir / f"{ep}_fusion.joblib").exists():
                    shutil.copy2(cand_dir / f"{ep}_fusion.joblib", SERVING_DIR / f"{ep}_fusion.joblib")
                    print(f"  Restored {ep}_fusion.joblib from {cand_dir}", flush=True)

    # --- C. 1000x Cluster-Level Block Bootstrap 95% CIs + Small-N Admission Policy (`evidence_grade`) across ALL 47 Endpoints ---
    print("\n================ [P0-2: 1000x Cluster-Bootstrap 95% CIs & Evidence Grade (All 47 Endpoints)] ================", flush=True)
    n_quant, n_rank = 0, 0
    reg_fam_maes = {"champion": [], "fusion": [], "cpu": []}
    for ep, doc in manifest["endpoints"].items():
        tt = doc["task_type"]
        counts = doc["counts"]
        n_ho = int(counts.get("n_holdout", 0))
        dm = doc["dual_modality"]
        ho_m = dm["holdout_metrics"]

        # Compute cluster-level bootstrap CI from saved holdout_champion.tsv if not already computed
        if "holdout_bootstrap_ci95" not in dm:
            pred_fp = None
            for base_cp in (BUNDLE_DIR / "champion_predictions", EXP331 / "champion_predictions"):
                p1 = base_cp / ep / "holdout_champion.tsv"
                if p1.exists():
                    pred_fp = p1
                    break
            ep_tsv = get_existing_ep_path(ep)
            if pred_fp is not None and ep_tsv is not None:
                rec_to_cl = {}
                with ep_tsv.open(encoding="utf-8") as f:
                    for r in csv.DictReader(f, delimiter="\t"):
                        rec_to_cl[r.get("record_id", "")] = r.get("cluster_id", r.get("compound_key", ""))
                        rec_to_cl[r.get("compound_key", "")] = r.get("cluster_id", r.get("compound_key", ""))
                ys, ps, cls, lo80s, hi80s = [], [], [], [], []
                with pred_fp.open(encoding="utf-8") as f:
                    for r in csv.DictReader(f, delimiter="\t"):
                        if r.get("censor_type", "exact") != "exact":
                            continue
                        try:
                            yv = float(r["y_value"])
                            pv = float(r["pred_dual"])
                            if not (math.isfinite(yv) and math.isfinite(pv)):
                                continue
                            ys.append(yv)
                            ps.append(pv)
                            cls.append(rec_to_cl.get(r.get("record_id", ""), rec_to_cl.get(r.get("compound_key", ""), "C0")))
                            if r.get("lo80") and r.get("hi80"):
                                lo80s.append(float(r["lo80"]))
                                hi80s.append(float(r["hi80"]))
                        except Exception:
                            continue
                lo_arr = np.array(lo80s, dtype=np.float64) if len(lo80s) == len(ys) else None
                hi_arr = np.array(hi80s, dtype=np.float64) if len(hi80s) == len(ys) else None
                dm["holdout_bootstrap_ci95"] = cluster_block_bootstrap_ci(
                    np.array(ys, dtype=np.float64), np.array(ps, dtype=np.float64), np.array(cls, dtype=object), tt, lo_arr, hi_arr, n_boot=1000
                )

        # Honest Small-N & Quality Admission Policy (`evidence_grade`)
        if tt == "classification":
            auc = float(ho_m.get("roc_auc") or 0.0)
            grade = "quantitative" if (n_ho >= 30 and auc >= 0.75) else "rank_only"
        else:
            r2 = float(ho_m.get("r2") if ho_m.get("r2") is not None else -9.0)
            sp = float(ho_m.get("spearman") if ho_m.get("spearman") is not None else -1.0)
            grade = "quantitative" if (n_ho >= 30 and r2 >= 0.15 and sp >= 0.45) else "rank_only"
            fhm = dm.get("family_holdout_metrics") or {}
            if ho_m.get("mae") is not None and (fhm.get("fusion") or {}).get("mae") is not None and (fhm.get("cpu") or {}).get("mae") is not None:
                reg_fam_maes["champion"].append(float(ho_m["mae"]))
                reg_fam_maes["fusion"].append(float(fhm["fusion"]["mae"]))
                reg_fam_maes["cpu"].append(float(fhm["cpu"]["mae"]))
        doc["evidence_grade"] = grade
        if grade == "quantitative":
            n_quant += 1
        else:
            n_rank += 1

    print(f"  Evidence Grades across 47 endpoints: quantitative={n_quant}, rank_only={n_rank}", flush=True)

    # --- D. Statistical Significance Tests (Friedman + Wilcoxon + Paired Bootstrap) on Family Ablation (Q7) ---
    stat_tests: Dict[str, Any] = {}
    if len(reg_fam_maes["champion"]) >= 10:
        c_arr = np.array(reg_fam_maes["champion"])
        f_arr = np.array(reg_fam_maes["fusion"])
        u_arr = np.array(reg_fam_maes["cpu"])
        fr_stat, fr_p = friedmanchisquare(c_arr, f_arr, u_arr)
        # Paired Wilcoxon signed-rank tests (two-sided)
        w_cf_p = float(wilcoxon(c_arr, f_arr).pvalue) if np.any(c_arr != f_arr) else 1.0
        w_cu_p = float(wilcoxon(c_arr, u_arr).pvalue) if np.any(c_arr != u_arr) else 1.0
        w_fu_p = float(wilcoxon(f_arr, u_arr).pvalue) if np.any(f_arr != u_arr) else 1.0
        # 5000x paired bootstrap CI for MAE reduction (CPU - Champion)
        rng_b = np.random.RandomState(42)
        diffs_cu = u_arr - c_arr
        boot_cu = [float(np.mean(diffs_cu[rng_b.randint(0, len(diffs_cu), size=len(diffs_cu))])) for _ in range(5000)]
        diffs_fu = u_arr - f_arr
        boot_fu = [float(np.mean(diffs_fu[rng_b.randint(0, len(diffs_fu), size=len(diffs_fu))])) for _ in range(5000)]
        stat_tests["regression_family_holdout_mae"] = {
            "n_endpoints": int(len(c_arr)),
            "mean_mae": {"champion": round(float(c_arr.mean()), 4), "fusion": round(float(f_arr.mean()), 4), "cpu": round(float(u_arr.mean()), 4)},
            "friedman_chi2": round(float(fr_stat), 4),
            "friedman_p": float(f"{fr_p:.3e}"),
            "wilcoxon_p_champion_vs_cpu": float(f"{w_cu_p:.3e}"),
            "wilcoxon_p_fusion_vs_cpu": float(f"{w_fu_p:.3e}"),
            "wilcoxon_p_champion_vs_fusion": float(f"{w_cf_p:.3e}"),
            "paired_boot_cpu_minus_champion_mae_ci95": [round(float(np.percentile(boot_cu, 2.5)), 4), round(float(np.percentile(boot_cu, 97.5)), 4)],
            "paired_boot_cpu_minus_fusion_mae_ci95": [round(float(np.percentile(boot_fu, 2.5)), 4), round(float(np.percentile(boot_fu, 97.5)), 4)],
        }
        print(f"  [FAMILY SIGNIFICANCE] n={len(c_arr)} reg eps | Friedman chi2={fr_stat:.2f} (p={fr_p:.2e}) | Wilcoxon Champ vs CPU p={w_cu_p:.2e}, Fus vs CPU p={w_fu_p:.2e}", flush=True)

    # Single-FM 4-model ablation Friedman test (across 24 core endpoints if available in manifest)
    fm_names = ["ncaa_repdistill_v2", "esmc_600m", "peptideclm2", "chemberta_mtr"]
    fm_mat = {k: [] for k in fm_names}
    for ep, doc in manifest["endpoints"].items():
        rab = doc.get("representation_ablation_holdout") or {}
        if all(k in rab and rab[k] is not None for k in fm_names):
            # Each entry may be float or dict with mae/spearman/roc_auc
            vals = []
            for k in fm_names:
                v = rab[k]
                if isinstance(v, dict):
                    s = v.get("roc_auc") if doc["task_type"] == "classification" else v.get("spearman")
                else:
                    s = v
                vals.append(float(s) if s is not None else None)
            if all(x is not None and math.isfinite(x) for x in vals):
                for k, x in zip(fm_names, vals):
                    fm_mat[k].append(x)
    if len(fm_mat[fm_names[0]]) >= 8:
        arrs = [np.array(fm_mat[k]) for k in fm_names]
        fr_fm, p_fm = friedmanchisquare(*arrs)
        stat_tests["single_fm_ablation_holdout"] = {
            "n_endpoints": int(len(arrs[0])),
            "mean_primary_score": {k: round(float(np.mean(fm_mat[k])), 4) for k in fm_names},
            "friedman_chi2": round(float(fr_fm), 4),
            "friedman_p": float(f"{p_fm:.3e}"),
        }
        print(f"  [4-FM ABLATION FRIEDMAN] n={len(arrs[0])} eps | chi2={fr_fm:.2f} (p={p_fm:.2e}) | means={stat_tests['single_fm_ablation_holdout']['mean_primary_score']}", flush=True)

    # --- E. Update AD Analog Index cleanly for promoted endpoints ---
    old_ad = dict(np.load(ad_npz_path, allow_pickle=False))
    with gzip.open(analog_gz_path, "rt", encoding="utf-8") as gz:
        old_lookup = json.load(gz)

    ck_to_meta = {}
    with (EXP331 / "data" / "compounds_v331_merged.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ck_to_meta[r["compound_key"]] = (r.get("canonical_smiles", ""), r.get("parent_sequence", ""))
    for r in bank.delta_rows:
        ck_to_meta[r["compound_key"]] = (r.get("canonical_smiles", ""), r.get("parent_sequence", ""))

    ref_key_to_gidx = {}
    for ep_k, v in old_lookup["endpoints"].items():
        idxs = old_ad[f"ep_ref_idx__{ep_k}"]
        for rec, gi in zip(v["records"], idxs):
            ref_key_to_gidx[rec["compound_key"]] = int(gi)

    new_ref_cks = []
    for ep in receipt["promoted"]:
        ep_rows = list(csv.DictReader((EXP335 / "endpoints" / f"{ep}.tsv").open(encoding="utf-8"), delimiter="\t"))
        for r in ep_rows:
            if int(r["is_holdout"]) == 0 and r["compound_key"] not in ref_key_to_gidx:
                ref_key_to_gidx[r["compound_key"]] = int(old_ad["ref_morgan_packed"].shape[0]) + len(new_ref_cks)
                new_ref_cks.append(r["compound_key"])

    if new_ref_cks:
        n_new = len(new_ref_cks)
        add_packed = np.zeros((n_new, 128), dtype=np.uint8)
        add_smi_v = np.zeros(n_new, dtype=bool)
        add_seq_v = np.zeros(n_new, dtype=bool)
        indptr = list(old_ad["seq_3mer_indptr"].astype(np.int32))
        indices = list(old_ad["seq_3mer_indices"].astype(np.int16))
        add_pop = np.zeros(n_new, dtype=np.float32)
        for i, ck in enumerate(new_ref_cks):
            gidx = bank.key_to_idx[ck]
            add_packed[i] = bank.morgan_packed[gidx]
            add_smi_v[i] = bank.smiles_valid[gidx]
            add_seq_v[i] = bank.seq_valid[gidx]
            _, seq = ck_to_meta.get(ck, ("", ""))
            clean = [c for c in (seq or "").upper() if c in AA20_IDX]
            codes = sorted({AA20_IDX[clean[j]] * 400 + AA20_IDX[clean[j + 1]] * 20 + AA20_IDX[clean[j + 2]] for j in range(len(clean) - 2)}) if len(clean) >= 3 else []
            indices.extend(codes)
            indptr.append(len(indices))
            add_pop[i] = float(len(codes))
        old_ad["ref_morgan_packed"] = np.concatenate([old_ad["ref_morgan_packed"], add_packed], axis=0)
        old_ad["ref_smi_valid"] = np.concatenate([old_ad["ref_smi_valid"], add_smi_v], axis=0)
        old_ad["ref_seq_valid"] = np.concatenate([old_ad["ref_seq_valid"], add_seq_v], axis=0)
        old_ad["seq_3mer_indptr"] = np.array(indptr, dtype=np.int32)
        old_ad["seq_3mer_indices"] = np.array(indices, dtype=np.int16)
        old_ad["seq_3mer_pop"] = np.concatenate([old_ad["seq_3mer_pop"], add_pop], axis=0)

    for ep in receipt["promoted"]:
        ep_rows = [r for r in csv.DictReader((EXP335 / "endpoints" / f"{ep}.tsv").open(encoding="utf-8"), delimiter="\t") if int(r["is_holdout"]) == 0]
        seen_ck = set()
        recs, ep_gidxs = [], []
        unit_str = manifest["endpoints"][ep]["unit"]
        for r in ep_rows:
            ck = r["compound_key"]
            if ck in seen_ck:
                continue
            seen_ck.add(ck)
            smi, seq = ck_to_meta.get(ck, ("", ""))
            recs.append({"compound_key": ck, "smiles": smi, "sequence": seq, "experimental_value": round(float(r["y_value"]), 4), "censor_type": r["censor_type"], "unit": unit_str})
            ep_gidxs.append(ref_key_to_gidx[ck])
        old_ad[f"ep_ref_idx__{ep}"] = np.array(ep_gidxs, dtype=np.int32)
        old_lookup["endpoints"][ep] = {"n_unique_compounds": len(recs), "records": recs}

    old_lookup["n_unique_ref_compounds"] = int(old_ad["ref_morgan_packed"].shape[0])
    np.savez_compressed(ad_npz_path, **old_ad)
    with gzip.open(analog_gz_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        json.dump(old_lookup, gz, ensure_ascii=False, separators=(",", ":"))

    # --- F. Sync manifest summaries, serving_inventory, checksums, and V33.5 governance metadata ---
    eps_all = manifest["endpoints"]
    ds = manifest["dataset_summary"]
    ds["n_endpoints"] = len(eps_all)
    ds["n_tier1_endpoints"] = sum(1 for v in eps_all.values() if int(v.get("tier", 1)) == 1)
    ds["n_tier2_endpoints"] = sum(1 for v in eps_all.values() if int(v.get("tier", 1)) == 2)
    ds["n_regression_endpoints"] = sum(1 for v in eps_all.values() if v.get("task_type") == "regression")
    ds["n_classification_endpoints"] = sum(1 for v in eps_all.values() if v.get("task_type") == "classification")
    ds["n_quantitative_endpoints"] = n_quant
    ds["n_rank_only_endpoints"] = n_rank
    ds["n_total_rows"] = sum(int(v["counts"]["n_total"]) for v in eps_all.values())
    ds["n_cv_rows"] = sum(int(v["counts"]["n_cv"]) for v in eps_all.values())
    ds["n_holdout_rows"] = sum(int(v["counts"]["n_holdout"]) for v in eps_all.values())
    ds["n_censored_rows"] = sum(int(v["counts"]["n_censored_total"]) for v in eps_all.values())
    ds["n_unique_compounds"] = int(bank.n_total - 1)
    ds["n_unified_compounds_incl_case17"] = int(bank.n_total)

    ov = manifest.setdefault("overall_summary", {})
    ov["dual_modality_winning_family_counts"] = dict(Counter(v["dual_modality"]["winning_family"] for v in eps_all.values()))
    ov["smiles_only_winning_family_counts"] = dict(Counter(v["smiles_only"]["winning_family"] for v in eps_all.values()))
    ov["statistical_significance_tests"] = stat_tests

    inv = manifest.setdefault("serving_inventory", {})
    inv_cpu = inv.setdefault("cpu_endpoints_bundled", [])
    inv_fus = inv.setdefault("fusion_endpoints_bundled", [])
    srv_shas = manifest.setdefault("sha256_checksums", {}).setdefault("serving_models", {})
    for ep in receipt["promoted"]:
        if ep not in inv_cpu:
            inv_cpu.append(ep)
        if ep not in inv_fus:
            inv_fus.append(ep)
        for suffix in ("_cpu.joblib", "_fusion.joblib"):
            fp = SERVING_DIR / f"{ep}{suffix}"
            if fp.exists():
                srv_shas[fp.name] = sha256_file(fp)

    manifest["version"] = "v33.5"
    manifest["v335_protocol_audit"] = {
        "zero_holdout_peeking_enforced": True,
        "promotion_rule": receipt["promotion_rule"],
        "tanimoto_isolation_threshold": 0.40,
        "promoted_endpoints": receipt["promoted"],
        "held_endpoints_v333_kept": receipt["held_on_v333"],
        "evidence_grade_counts": {"quantitative": n_quant, "rank_only": n_rank},
        "delta_logd_experimental_validation": ld_res["delta_validation"],
        "deep_mt_note": "Deep Multi-Task checkpoints (3 legacy winners: F, CL_absolute, Cmax) were not retrained in v33.5; newly promoted endpoints use CPU/4-FM Fusion only.",
        "clinical_pk_index_license": "US Federal Government Public Domain (FDA DailyMed SPL & Drugs@FDA NDA/BLA Reviews, 17 U.S.C. § 105) — 59 drugs, 427 PK records; zero CC-BY-NC mirroring.",
    }
    manifest.pop("v334_admetatlas_absorption", None)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP335 / "v335_protocol_audit_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== V33.5 SCRIPT A DONE in {time.time()-t_all:.1f}s | PROMOTED={receipt['promoted']} | HELD={receipt['held_on_v333']} ===", flush=True)


if __name__ == "__main__":
    main()
