#!/usr/bin/env python3
"""v33.3 Step 2 (runs on compute-node 64-core CPU ONLY):
  Part 2A: Unified Censored-Aware Main Models + Logit-Space PPB/F
           (TobitKernelEM on zero-PCA-loss 4-FM Multi-Block Kernel + Custom Tobit LightGBM + Logit-XGB-AFT)
           on 14 high-censored / bounded regression endpoints under strict nested 5-fold OOF.
  Part 2B: Cluster-Weighted Conformalized Quantile Regression (`cqr_lgb_cw`) &
           Cluster-Weighted Mondrian AD-EB (`mondrian_ad_eb_cw`) across ALL 40 Regression Endpoints
           (dual_modality + smiles_only = 80 tracks) under strict nested 5-fold OOF.

ZERO-HOLDOUT-PEEKING GUARANTEE:
  - All candidate selections, alpha/hyperparameter choices, and convex blend weights are chosen
    STRICTLY on nested 5-fold OOF (is_holdout == 0).
  - Untouched 20% Holdout (and strict 19-cluster-excluded Leave-Case17-Out PPB) evaluated ONCE
    after all selections are frozen.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr

V33_BASE = Path("${V33_BASE_DIR}")
EXP_BASE = V33_BASE / "expansion_v331"
BUNDLE_DIR = V33_BASE / "bundle"
SMALLN_DIR = V33_BASE / "v2_upgrade" / "smalln_cqr"
OUT_DIR = V33_BASE / "v333_science"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SMALLN_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))

from run_plan_b1_conformal import (  # noqa: E402
    _apply_baseline_manifest_conformal,
    _compute_oof_and_ho_tanimoto,
    _eval_interval_metrics,
    _extract_phys_block,
    _fit_baseline_conformal_on_subset,
    _oof_coverage_efficiency_score,
)
import run_plan_b1_conformal as b1_conf  # noqa: E402
from run_plan_b1_smalln import UnifiedSmallNBank, _load_endpoint_frozen  # noqa: E402
from smalln_models import (  # noqa: E402
    CompactTabularPreprocessor,
    MultiBlockKernelPreprocessor,
    ResidualCQRCalibrator,
    apply_mondrian_ad_eb,
    fit_mondrian_ad_eb,
)
from v33_config import CASE17_FDA_NUMERIC  # noqa: E402
from v333_models import (  # noqa: E402
    ClusterWeightedCQRCalibrator,
    TobitKernelEMRegressor,
    censored_hinge_mae,
    fit_mondrian_ad_eb_cw,
    inv_logit,
    logit_transform_bounds,
    make_tobit_lgb_objective,
)

csv.field_size_limit(10_000_000)

PART2A_ENDPOINTS = [
    "PPB",
    "F",
    "hemolysis_MHC",
    "cytotoxicity_CC50",
    "hemolysis_HC50",
    "MFC_C_albicans",
    "MIC_K_pneumoniae",
    "MBC_E_faecalis",
    "MIC_P_aeruginosa",
    "MBC_S_aureus",
    "MIC_E_faecalis",
    "MIC_S_aureus",
    "MIC_E_coli",
    "half_life_reg",
]
LOGIT_BOUNDED_ENDPOINTS = {"PPB", "F"}

G_BANK: Optional[UnifiedSmallNBank] = None
G_MANIFEST: Optional[Dict[str, Any]] = None
G_ALL_C17_CLUSTERS: Set[str] = set()


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _fit_xgb_aft_fold(
    X_tr: np.ndarray,
    yv: np.ndarray,
    yl: np.ndarray,
    yh: np.ndarray,
    ct: np.ndarray,
    use_logit: bool,
    bounds: Optional[Tuple[float, float]],
    seed: int,
    n_est: int = 220,
    max_depth: int = 4,
    aft_scale: float = 0.8,
) -> Tuple[xgb.Booster, float]:
    if use_logit:
        yv_u, yl_u, yh_u = logit_transform_bounds(yv, yl, yh, eps=0.005)
    else:
        yv_u, yl_u, yh_u = yv.copy(), yl.copy(), yh.copy()
    is_ex = ct == "exact"
    y_ref = float(np.median(yv_u[is_ex])) if is_ex.any() else float(np.median(yv_u))

    lo_t = np.exp(np.clip(yl_u - y_ref, -18.0, 18.0)).astype(np.float32)
    hi_t = np.exp(np.clip(yh_u - y_ref, -18.0, 18.0)).astype(np.float32)
    is_r = ct == "right"
    is_l = ct == "left"
    hi_t[is_r] = np.inf
    lo_t[is_l] = 0.0
    lo_t[is_ex] = np.exp(np.clip(yv_u[is_ex] - y_ref, -18.0, 18.0)).astype(np.float32)
    hi_t[is_ex] = lo_t[is_ex]
    non_ex_fin = (~is_ex) & (~is_r)
    tight = non_ex_fin & (hi_t <= lo_t)
    if tight.any():
        hi_t[tight] = lo_t[tight] * 1.001 + 1e-6

    dtr = xgb.DMatrix(np.asarray(X_tr, dtype=np.float32))
    dtr.set_float_info("label_lower_bound", lo_t)
    dtr.set_float_info("label_upper_bound", hi_t)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": aft_scale,
        "max_depth": max_depth,
        "learning_rate": 0.04,
        "min_child_weight": 5.0 if len(yv) >= 300 else 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 8.0,
        "reg_alpha": 0.5,
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
    }
    bst = xgb.train(params, dtr, num_boost_round=n_est)
    return bst, y_ref


def _pred_xgb_aft(bst: xgb.Booster, y_ref: float, X: np.ndarray, use_logit: bool, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    dm = xgb.DMatrix(np.asarray(X, dtype=np.float32))
    z = bst.predict(dm, output_margin=True).astype(np.float64) + y_ref
    p = inv_logit(z) if use_logit else z
    if bounds is not None:
        p = np.clip(p, bounds[0], bounds[1])
    return p


def _fit_lgb_tobit_fold(
    X_tr: np.ndarray,
    yv: np.ndarray,
    yl: np.ndarray,
    yh: np.ndarray,
    ct: np.ndarray,
    use_logit: bool,
    seed: int,
) -> Tuple[lgb.Booster, float]:
    if use_logit:
        yv_u, yl_u, yh_u = logit_transform_bounds(yv, yl, yh, eps=0.005)
    else:
        yv_u, yl_u, yh_u = yv.copy(), yl.copy(), yh.copy()
    is_ex = ct == "exact"
    y_mean = float(np.mean(yv_u[is_ex])) if is_ex.any() else float(np.mean(yv_u))
    y_std = max(float(np.std(yv_u[is_ex])) if is_ex.sum() >= 3 else float(np.std(yv_u)), 0.15)
    sigma = 0.55 * y_std

    obj_fn = make_tobit_lgb_objective(yv_u - y_mean, yl_u - y_mean, yh_u - y_mean, ct, sigma=sigma)
    dtr = lgb.Dataset(np.asarray(X_tr, dtype=np.float32), label=np.zeros(len(yv_u), dtype=np.float32), free_raw_data=False)
    n_tr = len(yv_u)
    params = {
        "objective": obj_fn,
        "learning_rate": 0.04,
        "num_leaves": 7 if n_tr < 300 else (15 if n_tr < 1500 else 23),
        "max_depth": 3 if n_tr < 300 else 5,
        "min_child_samples": max(5, min(25, n_tr // 12)),
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 8.0,
        "n_jobs": 1,
        "seed": seed,
        "verbose": -1,
    }
    n_rounds = 140 if n_tr < 300 else (220 if n_tr < 1500 else 300)
    bst = lgb.train(params, dtr, num_boost_round=n_rounds)
    return bst, y_mean


def _pred_lgb_tobit(bst: lgb.Booster, y_mean: float, X: np.ndarray, use_logit: bool, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    z = bst.predict(np.asarray(X, dtype=np.float32)).astype(np.float64) + y_mean
    p = inv_logit(z) if use_logit else z
    if bounds is not None:
        p = np.clip(p, bounds[0], bounds[1])
    return p


def _fit_blend_weight_hinge(
    p_base: np.ndarray,
    p_new: np.ndarray,
    yv: np.ndarray,
    yl: np.ndarray,
    yh: np.ndarray,
    ct: np.ndarray,
    reg: float = 0.002,
) -> float:
    best_w = 0.0
    best_v = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        p = (1.0 - w) * p_base + w * p_new
        m = censored_hinge_mae(yv, p, yl, yh, ct)
        v = m["hinge_mae_all"] + reg * (w ** 2)
        if v < best_v:
            best_v = v
            best_w = float(w)
    return best_w


def _run_part2a_endpoint(ep: str) -> Dict[str, Any]:
    t0 = time.time()
    assert G_BANK is not None and G_MANIFEST is not None
    ep_doc = G_MANIFEST["endpoints"][ep]
    d = _load_endpoint_frozen(ep, ep_doc, G_BANK)
    bounds = d["bounds"]
    use_logit = ep in LOGIT_BOUNDED_ENDPOINTS

    cv_idx = d["cv_idx"]
    ho_idx = d["ho_idx"]
    cidx_cv = d["compound_idx"][cv_idx]
    cidx_ho = d["compound_idx"][ho_idx]
    cond_cv = [d["cond_rows"][i] for i in cv_idx]
    cond_ho = [d["cond_rows"][i] for i in ho_idx]
    fids_cv = d["fold_id"][cv_idx]
    yv_cv, yl_cv, yh_cv, ct_cv = d["y_value"][cv_idx], d["y_lower"][cv_idx], d["y_upper"][cv_idx], d["censor_type"][cv_idx]
    yv_ho, yl_ho, yh_ho, ct_ho = d["y_value"][ho_idx], d["y_lower"][ho_idx], d["y_upper"][ho_idx], d["censor_type"][ho_idx]

    out_ep: Dict[str, Any] = {
        "endpoint": ep,
        "use_logit": use_logit,
        "n_cv": len(cv_idx),
        "n_ho": len(ho_idx),
        "n_cens_cv": int((ct_cv != "exact").sum()),
        "n_cens_ho": int((ct_ho != "exact").sum()),
        "tracks": {},
    }

    for track in ("dual_modality", "smiles_only"):
        valid_cv = G_BANK.smiles_valid[cidx_cv] if track == "smiles_only" else np.ones(len(cv_idx), dtype=bool)
        valid_ho = G_BANK.smiles_valid[cidx_ho] if track == "smiles_only" else np.ones(len(ho_idx), dtype=bool)
        base_oof = d["base_champ_oof"][track]
        base_ho = d["base_champ_ho"][track]

        cand_names = ["tobit_krr_a03", "tobit_krr_a15", "lgb_tobit", "xgb_aft"]
        oof_preds = {k: np.full(len(cv_idx), np.nan, dtype=np.float64) for k in cand_names}
        ho_folds = {k: [] for k in cand_names}
        krr_saved_folds: Dict[str, List[Tuple[int, Any, Any]]] = {"tobit_krr_a03": [], "tobit_krr_a15": []}

        for f in range(5):
            tr_m = (fids_cv != f) & valid_cv
            va_m = (fids_cv == f) & valid_cv
            if tr_m.sum() < 10 or va_m.sum() == 0:
                continue
            tr_p = np.where(tr_m)[0]
            va_p = np.where(va_m)[0]

            # 1. MultiBlockKernelPreprocessor + TobitKernelEM (2 regularization strengths)
            pre_k = MultiBlockKernelPreprocessor(track=track, cond_cols=d["cond_cols"])
            Xk_tr = pre_k.fit_transform(G_BANK, cidx_cv[tr_p], [cond_cv[i] for i in tr_p])
            Xk_va = pre_k.transform(G_BANK, cidx_cv[va_p], [cond_cv[i] for i in va_p])
            Xk_ho = pre_k.transform(G_BANK, cidx_ho, cond_rows=cond_ho)

            for cname, alpha in (("tobit_krr_a03", 0.35), ("tobit_krr_a15", 1.5)):
                est_k = TobitKernelEMRegressor(
                    alpha=alpha, fm_kernel="exp_cosine", exp_gamma=1.25,
                    n_em_iters=8, use_logit=use_logit, bounds=bounds, max_exact_n=1400,
                )
                est_k.fit(Xk_tr, yv_cv[tr_p], yl_cv[tr_p], yh_cv[tr_p], ct_cv[tr_p], seed=142 + f)
                oof_preds[cname][va_p] = est_k.predict(Xk_va)
                ho_folds[cname].append(est_k.predict(Xk_ho))
                krr_saved_folds[cname].append((f, pre_k, est_k))

            # 2. CompactTabularPreprocessor + Custom Tobit LightGBM + XGBoost AFT
            pre_t = CompactTabularPreprocessor(track=track, cond_cols=d["cond_cols"])
            Xt_tr = pre_t.fit_transform(G_BANK, cidx_cv[tr_p], [cond_cv[i] for i in tr_p])
            Xt_va = pre_t.transform(G_BANK, cidx_cv[va_p], [cond_cv[i] for i in va_p])
            Xt_ho = pre_t.transform(G_BANK, cidx_ho, cond_rows=cond_ho)

            bst_l, ym_l = _fit_lgb_tobit_fold(Xt_tr, yv_cv[tr_p], yl_cv[tr_p], yh_cv[tr_p], ct_cv[tr_p], use_logit, seed=242 + f)
            oof_preds["lgb_tobit"][va_p] = _pred_lgb_tobit(bst_l, ym_l, Xt_va, use_logit, bounds)
            ho_folds["lgb_tobit"].append(_pred_lgb_tobit(bst_l, ym_l, Xt_ho, use_logit, bounds))

            bst_x, yr_x = _fit_xgb_aft_fold(
                Xt_tr, yv_cv[tr_p], yl_cv[tr_p], yh_cv[tr_p], ct_cv[tr_p], use_logit, bounds,
                seed=342 + f, aft_scale=0.6 if use_logit else 0.85,
            )
            oof_preds["xgb_aft"][va_p] = _pred_xgb_aft(bst_x, yr_x, Xt_va, use_logit, bounds)
            ho_folds["xgb_aft"].append(_pred_xgb_aft(bst_x, yr_x, Xt_ho, use_logit, bounds))

        ho_preds = {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in ho_folds.items() if v}

        # Strictly OOF-only evaluation & candidate selection
        base_cv_m = censored_hinge_mae(yv_cv[valid_cv], base_oof[valid_cv], yl_cv[valid_cv], yh_cv[valid_cv], ct_cv[valid_cv])
        base_ho_m = censored_hinge_mae(yv_ho[valid_ho], base_ho[valid_ho], yl_ho[valid_ho], yh_ho[valid_ho], ct_ho[valid_ho])

        cand_cv_metrics = {}
        cand_ho_metrics = {}
        for cname in cand_names:
            cand_cv_metrics[cname] = censored_hinge_mae(yv_cv[valid_cv], oof_preds[cname][valid_cv], yl_cv[valid_cv], yh_cv[valid_cv], ct_cv[valid_cv])
            cand_ho_metrics[cname] = censored_hinge_mae(yv_ho[valid_ho], ho_preds[cname][valid_ho], yl_ho[valid_ho], yh_ho[valid_ho], ct_ho[valid_ho])

        best_cand = min(cand_names, key=lambda k: cand_cv_metrics[k]["hinge_mae_all"])
        # Nested 5-fold OOF convex blend (1-w)*base_champ + w*best_cand (strictly inner-4-fold w!)
        blend_nested_oof = np.full(len(cv_idx), np.nan, dtype=np.float64)
        ws_inner = []
        for f in range(5):
            tr_m = (fids_cv != f) & valid_cv
            va_m = (fids_cv == f) & valid_cv
            w_f = _fit_blend_weight_hinge(base_oof[tr_m], oof_preds[best_cand][tr_m], yv_cv[tr_m], yl_cv[tr_m], yh_cv[tr_m], ct_cv[tr_m])
            ws_inner.append(w_f)
            blend_nested_oof[va_m] = (1.0 - w_f) * base_oof[va_m] + w_f * oof_preds[best_cand][va_m]
        w_full = _fit_blend_weight_hinge(base_oof[valid_cv], oof_preds[best_cand][valid_cv], yv_cv[valid_cv], yl_cv[valid_cv], yh_cv[valid_cv], ct_cv[valid_cv])
        blend_ho = (1.0 - w_full) * base_ho + w_full * ho_preds[best_cand]

        blend_cv_m = censored_hinge_mae(yv_cv[valid_cv], blend_nested_oof[valid_cv], yl_cv[valid_cv], yh_cv[valid_cv], ct_cv[valid_cv])
        blend_ho_m = censored_hinge_mae(yv_ho[valid_ho], blend_ho[valid_ho], yl_ho[valid_ho], yh_ho[valid_ho], ct_ho[valid_ho])

        # Promotion check strictly on nested OOF: hinge_mae improves >= 0.003 AND exact_mae does not regress > 0.002
        # Compare best_cand standalone vs nested blend
        opt_standalone = (best_cand, cand_cv_metrics[best_cand], cand_ho_metrics[best_cand], 1.0)
        opt_blend = (f"blend({best_cand},w={w_full:.2f})", blend_cv_m, blend_ho_m, w_full)
        winner_lbl, win_cv, win_ho, win_w = opt_blend if blend_cv_m["hinge_mae_all"] <= cand_cv_metrics[best_cand]["hinge_mae_all"] else opt_standalone

        d_hinge_cv = base_cv_m["hinge_mae_all"] - win_cv["hinge_mae_all"]
        d_exact_cv = (base_cv_m["exact_mae"] - win_cv["exact_mae"]) if (base_cv_m["exact_mae"] and win_cv["exact_mae"]) else 0.0
        promoted = (d_hinge_cv >= 0.003) and (d_exact_cv >= -0.002) and (win_w > 0.0)

        out_ep["tracks"][track] = {
            "baseline_cv": base_cv_m,
            "baseline_ho": base_ho_m,
            "candidates_cv": cand_cv_metrics,
            "candidates_ho": cand_ho_metrics,
            "best_standalone": best_cand,
            "blend_weight_full": round(w_full, 4),
            "blend_weights_inner": [round(x, 3) for x in ws_inner],
            "blend_nested_cv": blend_cv_m,
            "blend_ho": blend_ho_m,
            "selected_candidate": winner_lbl,
            "oof_delta_hinge_mae": round(d_hinge_cv, 5),
            "oof_delta_exact_mae": round(d_exact_cv, 5),
            "promoted_strict_oof": promoted,
            "selected_ho": win_ho,
        }

        # Special diagnostic: strict 19-cluster-excluded Leave-Case17-Out check for Logit-PPB!
        if ep == "PPB" and track == "dual_modality":
            keep_c17 = np.where(
                (d["case17_excluded"] == 0) & np.array([str(c) not in G_ALL_C17_CLUSTERS for c in d["cluster_id"]], dtype=bool)
            )[0]
            cidx_nc = d["compound_idx"][keep_c17]
            cond_nc = [d["cond_rows"][i] for i in keep_c17]
            pre_k_c17 = MultiBlockKernelPreprocessor(track="dual_modality", cond_cols=d["cond_cols"])
            Xk_nc = pre_k_c17.fit_transform(G_BANK, cidx_nc, cond_nc)
            Xk_q = pre_k_c17.transform(G_BANK, G_BANK.case17_idx, default_cond=d["canonical_conditions"])
            est_c17 = TobitKernelEMRegressor(
                alpha=0.35 if best_cand == "tobit_krr_a03" else 1.5,
                fm_kernel="exp_cosine", exp_gamma=1.25, n_em_iters=8, use_logit=True, bounds=bounds,
            )
            est_c17.fit(Xk_nc, d["y_value"][keep_c17], d["y_lower"][keep_c17], d["y_upper"][keep_c17], d["censor_type"][keep_c17])
            p_c17_logit = est_c17.predict(Xk_q)
            c17_diag = []
            yt_l, yp_l = [], []
            for i, q in enumerate(G_BANK.case17_queries):
                dn = q["drug_name"]
                fda_ppb = CASE17_FDA_NUMERIC.get(dn, {}).get("ppb")
                c17_diag.append({"drug_name": dn, "fda_ppb": fda_ppb, "logit_tobit_ppb": round(float(p_c17_logit[i]), 4)})
                if fda_ppb is not None:
                    yt_l.append(fda_ppb)
                    yp_l.append(float(p_c17_logit[i]))
            out_ep["ppb_case17_strict_logit_tobit"] = {
                "n": len(yt_l),
                "mae": round(float(np.mean(np.abs(np.array(yt_l) - np.array(yp_l)))), 4),
                "spearman": round(float(spearmanr(yt_l, yp_l).statistic), 4),
                "drugs": c17_diag,
            }

    d_tr = out_ep["tracks"]["dual_modality"]
    print(
        f"  [2A {ep:<20}] base_cv_h={d_tr['baseline_cv']['hinge_mae_all']:.4f} -> best={d_tr['selected_candidate']:<24} "
        f"cv_h={d_tr['blend_nested_cv']['hinge_mae_all'] if 'blend' in d_tr['selected_candidate'] else d_tr['candidates_cv'][d_tr['best_standalone']]['hinge_mae_all']:.4f} "
        f"| HO_h: {d_tr['baseline_ho']['hinge_mae_all']:.4f} -> {d_tr['selected_ho']['hinge_mae_all']:.4f} "
        f"| PROM={d_tr['promoted_strict_oof']} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return out_ep


def _run_part2b_endpoint(ep: str) -> Dict[str, Any]:
    """Compare 5 conformal methods (baseline, mondrian_ad_eb, mondrian_ad_eb_cw, cqr_lgb, cqr_lgb_cw)
    strictly on nested 5-fold OOF across both tracks, then evaluate ONCE on untouched 20% Holdout.
    """
    t0 = time.time()
    assert G_MANIFEST is not None and b1_conf.GLOBAL_KEY_TO_IDX is not None
    ep_doc = G_MANIFEST["endpoints"][ep]
    wave = ep_doc.get("expansion_wave", "v33.0")
    ep_dir = EXP_BASE / "data/endpoints" if wave == "v33.1" else V33_BASE / "data/endpoints"
    rows = _read_tsv(ep_dir / f"{ep}.tsv")
    cv_rows = [r for r in rows if int(r["is_holdout"]) == 0]
    ho_rows = [r for r in rows if int(r["is_holdout"]) == 1]

    cidx_cv = np.array([b1_conf.GLOBAL_KEY_TO_IDX[r["compound_key"]] for r in cv_rows], dtype=np.int32)
    cidx_ho = np.array([b1_conf.GLOBAL_KEY_TO_IDX[r["compound_key"]] for r in ho_rows], dtype=np.int32)
    fids_cv = np.array([int(r["fold_id"]) for r in cv_rows], dtype=np.int32)
    cids_cv = np.array([r["cluster_id"] for r in cv_rows], dtype=object)
    y_cv = np.array([float(r["y_value"]) for r in cv_rows], dtype=np.float64)
    y_ho = np.array([float(r["y_value"]) for r in ho_rows], dtype=np.float64)
    ct_cv = np.array([r["censor_type"] for r in cv_rows], dtype=object)
    ct_ho = np.array([r["censor_type"] for r in ho_rows], dtype=object)
    bounds_raw = ep_doc.get("bounds")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1])) if bounds_raw is not None else None

    oof_max_t, oof_top5_t, ho_max_t, ho_top5_t = _compute_oof_and_ho_tanimoto(cidx_cv, fids_cv, cidx_ho)

    # Load exact champion OOF & HO predictions (same logic as run_plan_b1_conformal.py)
    if wave == "v33.0":
        coof = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "oof_champion.tsv")
        cho = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "holdout_champion.tsv")
        pred_oof_map = {
            "dual_modality": np.array([float(r["pred_dual"]) if r["pred_dual"] != "" else np.nan for r in coof]),
            "smiles_only": np.array([float(r["pred_smiles_only"]) if r["pred_smiles_only"] != "" else np.nan for r in coof]),
        }
        pred_ho_map = {
            "dual_modality": np.array([float(r["pred_dual"]) if r["pred_dual"] != "" else np.nan for r in cho]),
            "smiles_only": np.array([float(r["pred_smiles_only"]) if r["pred_smiles_only"] != "" else np.nan for r in cho]),
        }
    else:
        cpu_oof = _read_tsv(EXP_BASE / "models_cpu" / ep / "oof_predictions.tsv")
        cpu_ho = _read_tsv(EXP_BASE / "models_cpu" / ep / "holdout_predictions.tsv")
        fus_oof = _read_tsv(EXP_BASE / "models_fusion" / ep / "oof_predictions.tsv")
        fus_ho = _read_tsv(EXP_BASE / "models_fusion" / ep / "holdout_predictions.tsv")
        pred_oof_map, pred_ho_map = {}, {}
        for tr, col in (("dual_modality", "pred_dual"), ("smiles_only", "pred_smiles_only")):
            w = ep_doc[tr]["weights"]
            c_o = np.array([float(r[col]) if r[col] != "" else np.nan for r in cpu_oof])
            f_o = np.array([float(r[col]) if r[col] != "" else np.nan for r in fus_oof])
            c_h = np.array([float(r[col]) if r[col] != "" else np.nan for r in cpu_ho])
            f_h = np.array([float(r[col]) if r[col] != "" else np.nan for r in fus_ho])
            wc, wf = float(w.get("cpu", 0.0)), float(w.get("fusion", 0.0))
            tot = max(wc + wf, 1e-12)
            po = (wc * np.nan_to_num(c_o) + wf * np.nan_to_num(f_o)) / tot
            ph = (wc * np.nan_to_num(c_h) + wf * np.nan_to_num(f_h)) / tot
            if bounds is not None:
                po, ph = np.clip(po, bounds[0], bounds[1]), np.clip(ph, bounds[0], bounds[1])
            pred_oof_map[tr], pred_ho_map[tr] = po, ph

    cal_dir = OUT_DIR / "cqr_cw_calibrators"
    cal_dir.mkdir(parents=True, exist_ok=True)
    ep_res: Dict[str, Any] = {"endpoint": ep, "wave": wave, "tracks": {}}

    for track in ("dual_modality", "smiles_only"):
        valid_cv = b1_conf.GLOBAL_SMI_VALID[cidx_cv] if track == "smiles_only" else np.ones(len(cv_rows), dtype=bool)
        valid_ho = b1_conf.GLOBAL_SMI_VALID[cidx_ho] if track == "smiles_only" else np.ones(len(ho_rows), dtype=bool)
        phys_cv = _extract_phys_block(cidx_cv, track)
        phys_ho = _extract_phys_block(cidx_ho, track)

        m_cv = valid_cv & (ct_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof_map[track])
        m_ho = valid_ho & (ct_ho == "exact") & np.isfinite(y_ho) & np.isfinite(pred_ho_map[track])
        ye, pe, fe, ce = y_cv[m_cv], pred_oof_map[track][m_cv], fids_cv[m_cv], cids_cv[m_cv]
        ph_e, t1_e, t5_e = phys_cv[m_cv], oof_max_t[m_cv], oof_top5_t[m_cv]
        yh, ph, ph_h, t1_h, t5_h = y_ho[m_ho], pred_ho_map[track][m_ho], phys_ho[m_ho], ho_max_t[m_ho], ho_top5_t[m_ho]

        base_conf = ep_doc[track]["conformal"]
        base_mode = base_conf.get("mode", "standard")
        methods = ("baseline", "mondrian_ad_eb", "mondrian_ad_eb_cw", "cqr_lgb", "cqr_lgb_cw")
        nested_ivs = {k: {b: np.full(len(ye), np.nan) for b in ("lo80", "hi80", "lo90", "hi90")} for k in methods}

        for f in sorted(set(fe.tolist())):
            va = fe == f
            tr = ~va
            if tr.sum() < 6 or va.sum() == 0:
                continue
            # 1. baseline
            b_sp = _fit_baseline_conformal_on_subset(ye[tr], pe[tr], t1_e[tr], mode=base_mode, bounds=bounds)
            iv1 = _apply_baseline_manifest_conformal(pe[va], b_sp, t1_e[va])
            # 2. mondrian_ad_eb (unweighted)
            m_sp = fit_mondrian_ad_eb(ye[tr], pe[tr], t1_e[tr], bounds=bounds, cluster_ids=ce[tr])
            iv2 = apply_mondrian_ad_eb(pe[va], m_sp, t1_e[va])
            # 3. mondrian_ad_eb_cw (cluster-weighted!)
            mcw_sp = fit_mondrian_ad_eb_cw(ye[tr], pe[tr], t1_e[tr], bounds=bounds, cluster_ids=ce[tr])
            iv3 = apply_mondrian_ad_eb(pe[va], mcw_sp, t1_e[va])
            # 4. cqr_lgb (unweighted)
            cqr_u = ResidualCQRCalibrator(bounds=bounds).fit(ye[tr], pe[tr], ph_e[tr], t1_e[tr], t5_e[tr], fold_ids=fe[tr], cluster_ids=ce[tr])
            iv4 = cqr_u.predict_intervals(pe[va], ph_e[va], t1_e[va], t5_e[va])
            # 5. cqr_lgb_cw (cluster-weighted!)
            cqr_cw = ClusterWeightedCQRCalibrator(bounds=bounds, cluster_weighted=True).fit(
                ye[tr], pe[tr], ph_e[tr], t1_e[tr], t5_e[tr], fold_ids=fe[tr], cluster_ids=ce[tr]
            )
            iv5 = cqr_cw.predict_intervals(pe[va], ph_e[va], t1_e[va], t5_e[va])

            for k, iv in zip(methods, (iv1, iv2, iv3, iv4, iv5)):
                for b in ("lo80", "hi80", "lo90", "hi90"):
                    nested_ivs[k][b][va] = iv[b]

        oof_m = {k: _eval_interval_metrics(ye, nested_ivs[k], t1_e) for k in methods}
        bw80 = float(oof_m["baseline"]["w80"] or 1.0)
        bw90 = float(oof_m["baseline"]["w90"] or 1.5)
        scores = {}
        for k in methods:
            sc, det = _oof_coverage_efficiency_score(oof_m[k], bw80, bw90)
            scores[k] = det

        # Select winner strictly on nested 5-fold OOF score (prefer simpler JSON when within 0.015)
        best_m = min(methods, key=lambda k: scores[k]["total_score"])
        if best_m in ("cqr_lgb", "cqr_lgb_cw"):
            best_json = min(("baseline", "mondrian_ad_eb", "mondrian_ad_eb_cw"), key=lambda k: scores[k]["total_score"])
            if scores[best_json]["total_score"] <= scores[best_m]["total_score"] + 0.015:
                best_m = best_json

        # Fit final calibrators on full exact CV & evaluate ONCE on untouched 20% Holdout!
        iv_ho_base = _apply_baseline_manifest_conformal(ph, base_conf, t1_h)
        spec_m = fit_mondrian_ad_eb(ye, pe, t1_e, bounds=bounds, cluster_ids=ce)
        iv_ho_m = apply_mondrian_ad_eb(ph, spec_m, t1_h)
        spec_mcw = fit_mondrian_ad_eb_cw(ye, pe, t1_e, bounds=bounds, cluster_ids=ce)
        iv_ho_mcw = apply_mondrian_ad_eb(ph, spec_mcw, t1_h)
        cal_u = ResidualCQRCalibrator(bounds=bounds).fit(ye, pe, ph_e, t1_e, t5_e, fold_ids=fe, cluster_ids=ce)
        iv_ho_u = cal_u.predict_intervals(ph, ph_h, t1_h, t5_h)
        cal_cw = ClusterWeightedCQRCalibrator(bounds=bounds, cluster_weighted=True).fit(
            ye, pe, ph_e, t1_e, t5_e, fold_ids=fe, cluster_ids=ce
        )
        iv_ho_cw = cal_cw.predict_intervals(ph, ph_h, t1_h, t5_h)

        ho_m = {
            "baseline": _eval_interval_metrics(yh, iv_ho_base, t1_h),
            "mondrian_ad_eb": _eval_interval_metrics(yh, iv_ho_m, t1_h),
            "mondrian_ad_eb_cw": _eval_interval_metrics(yh, iv_ho_mcw, t1_h),
            "cqr_lgb": _eval_interval_metrics(yh, iv_ho_u, t1_h),
            "cqr_lgb_cw": _eval_interval_metrics(yh, iv_ho_cw, t1_h),
        }

        cal_file = None
        cal_kb = 0.0
        if best_m == "cqr_lgb_cw":
            fp = cal_dir / f"{ep}__{track}__cqr_lgb_cw.joblib"
            joblib.dump(cal_cw, fp, compress=3)
            cal_file = str(fp)
            cal_kb = round(fp.stat().st_size / 1024.0, 2)
        elif best_m == "cqr_lgb":
            fp = cal_dir / f"{ep}__{track}__cqr_lgb.joblib"
            joblib.dump(cal_u, fp, compress=3)
            cal_file = str(fp)
            cal_kb = round(fp.stat().st_size / 1024.0, 2)

        ep_res["tracks"][track] = {
            "selected_method": best_m,
            "n_exact_cv": len(ye),
            "n_clusters_cv": int(len(set(str(c) for c in ce))),
            "n_exact_ho": len(yh),
            "oof_metrics": oof_m,
            "oof_scores": scores,
            "ho_metrics": ho_m,
            "selected_ho": ho_m[best_m],
            "mondrian_cw_spec": spec_mcw,
            "mondrian_spec": spec_m,
            "calibrator_joblib": cal_file,
            "calibrator_kb": cal_kb,
        }

    d_t = ep_res["tracks"]["dual_modality"]
    print(
        f"  [2B {ep:<24}] sel={d_t['selected_method']:<18} | HO Cov80: base={d_t['ho_metrics']['baseline']['cov80']:.3f} "
        f"mcw={d_t['ho_metrics']['mondrian_ad_eb_cw']['cov80']:.3f} cqr_cw={d_t['ho_metrics']['cqr_lgb_cw']['cov80']:.3f} "
        f"-> sel={d_t['selected_ho']['cov80']:.3f} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return ep_res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-2a", action="store_true")
    parser.add_argument("--skip-2b", action="store_true")
    args = parser.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global G_BANK, G_MANIFEST, G_ALL_C17_CLUSTERS

    G_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
    G_BANK = UnifiedSmallNBank()

    # Load exact 19 Case17 clusters for strict PPB diagnostic
    c17_keys = {q["compound_key"] for q in G_BANK.case17_queries}
    for ep, ed in G_MANIFEST["endpoints"].items():
        if ed.get("expansion_wave", "v33.0") != "v33.0":
            continue
        for r in _read_tsv(V33_BASE / "data/endpoints" / f"{ep}.tsv"):
            if int(r["case17_excluded"]) == 1 or r["compound_key"] in c17_keys:
                G_ALL_C17_CLUSTERS.add(r["cluster_id"])

    # Hydrate b1_conf globals (CoW-shared across fork workers!)
    b1_conf.GLOBAL_MANIFEST = G_MANIFEST
    b1_conf.GLOBAL_PHYS = G_BANK.physchem
    b1_conf.GLOBAL_SMI_VALID = G_BANK.smiles_valid
    b1_conf.GLOBAL_SEQ_VALID = G_BANK.seq_valid
    b1_conf.GLOBAL_MORGAN_PACKED = G_BANK.morgan_packed
    b1_conf.GLOBAL_KEY_TO_IDX = G_BANK.key_to_idx

    res_2a: Dict[str, Any] = {}
    if not args.skip_2a:
        print(f"\n[Step 2A] Running Censored-Aware + Logit-Space Main Models on {len(PART2A_ENDPOINTS)} endpoints...", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(args.workers, len(PART2A_ENDPOINTS))) as pool:
            for r in pool.imap_unordered(_run_part2a_endpoint, PART2A_ENDPOINTS):
                res_2a[r["endpoint"]] = r

    res_2b: Dict[str, Any] = {}
    if not args.skip_2b:
        reg_eps = sorted(ep for ep, ed in G_MANIFEST["endpoints"].items() if ed["task_type"] == "regression")
        print(f"\n[Step 2B] Running Cluster-Weighted CQR & Mondrian across all {len(reg_eps)} regression endpoints (80 tracks)...", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_run_part2b_endpoint, reg_eps):
                res_2b[r["endpoint"]] = r

    # Aggregate 2B Holdout Summary across all 80 tracks!
    summary_2b: Dict[str, Any] = {}
    if res_2b:
        for mname in ("baseline", "mondrian_ad_eb", "mondrian_ad_eb_cw", "cqr_lgb", "cqr_lgb_cw", "selected"):
            c80s, c90s, w80s, ood80s = [], [], [], []
            for ep, ed in res_2b.items():
                for tr in ("dual_modality", "smiles_only"):
                    hm = ed["tracks"][tr]["selected_ho"] if mname == "selected" else ed["tracks"][tr]["ho_metrics"][mname]
                    if hm["cov80"] is not None:
                        c80s.append(hm["cov80"])
                        c90s.append(hm["cov90"])
                        w80s.append(hm["w80"])
                    if "OOD" in hm.get("by_tier", {}) and hm["by_tier"]["OOD"]["n"] >= 3:
                        ood80s.append(hm["by_tier"]["OOD"]["cov80"])
            summary_2b[mname] = {
                "n_tracks": len(c80s),
                "mean_cov80": round(float(np.mean(c80s)), 4),
                "mean_cov90": round(float(np.mean(c90s)), 4),
                "min_cov80": round(float(np.min(c80s)), 4),
                "n_tracks_cov80_lt_70pct": int(sum(1 for x in c80s if x < 0.70)),
                "n_tracks_cov80_in_75_88pct": int(sum(1 for x in c80s if 0.75 <= x <= 0.88)),
                "mean_w80": round(float(np.mean(w80s)), 4),
                "mean_ood_cov80": round(float(np.mean(ood80s)), 4) if ood80s else None,
            }
        sel_counts: Dict[str, int] = {}
        tot_kb = 0.0
        for ep, ed in res_2b.items():
            for tr in ("dual_modality", "smiles_only"):
                sm = ed["tracks"][tr]["selected_method"]
                sel_counts[sm] = sel_counts.get(sm, 0) + 1
                tot_kb += float(ed["tracks"][tr]["calibrator_kb"])
        summary_2b["selected_method_counts"] = sel_counts
        summary_2b["total_cqr_joblib_mb"] = round(tot_kb / 1024.0, 3)

    out_doc = {
        "schema": "v333_step2_censored_logit_cqr_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "part2a_censored_logit": res_2a,
        "part2b_cqr_summary": summary_2b,
        "part2b_endpoints": res_2b,
        "elapsed_sec": round(time.time() - t_all, 2),
    }
    out_path = OUT_DIR / "step2_censored_logit_cqr_results.json"
    out_path.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Step 2 DONE] Wrote {out_path} in {out_doc['elapsed_sec']}s", flush=True)
    if summary_2b:
        print(json.dumps(summary_2b, indent=2), flush=True)


if __name__ == "__main__":
    main()
