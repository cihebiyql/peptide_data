#!/usr/bin/env python3
"""
Wave 5A — Step 04b (runs on node23, 64 CPU cores):
64-Core Parallel Pretrained 4-FM Multimodal Fusion, Fair Individual Representation Ablation,
Censored Multimodal AFT, Nested Anti-(p>>n) ModalityAwareStacker, Conformal Uncertainty, &
Strict Zero-Leakage Leave-Case17-Out Suite for ALL 22 new V33.1 endpoints.

Imports exact picklable classes from /data/qlyu/v33_upgrade/models_fusion/ so every .joblib
unpickles cleanly inside v33_inference_service.py with ZERO new class dependencies!
Writes strictly under /data/qlyu/v33_upgrade/expansion_v331/models_fusion/.
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
import gc
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.linalg import eigh

V33_BASE = Path("/data/qlyu/v33_upgrade")
EXP_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP_SCRIPTS))

from fusion_bank import TanimotoADIndex  # noqa: E402
from fusion_config import CASE17_CLINICAL_ROUTE, DIRECT_FUSION_REPS, INDIVIDUAL_REPS  # noqa: E402
from fusion_metrics import ConformalIntervalCalibrator, compute_ad_stratification, regression_metrics  # noqa: E402
from fusion_models import (  # noqa: E402
    BoundedWrapper,
    EmbeddingFoldPreprocessor,
    ExactRBFKRR,
    ModalityAwareStacker,
    PrecomputedRBFSVR,
    XGBoostAFTRegressor,
    compute_sqdist_matrix,
    get_multimodal_aft_candidates,
    get_rep_candidates,
    make_fusion_estimator,
)
from v331_config import EXP_CPU_DIR, EXP_FUSION_DIR, V331_ENDPOINT_SPECS, V331FusionBank  # noqa: E402

csv.field_size_limit(10_000_000)

GLOBAL_BANK: Optional[V331FusionBank] = None


def _score_metrics(m: Dict[str, Any], is_aft: bool = False) -> Tuple[float, float]:
    if is_aft:
        hinge = m["all_row_hinge_mae"] if m["all_row_hinge_mae"] is not None else 1e9
        conc = m["censored_concordance"] if m["censored_concordance"] is not None else 0.0
        return (-hinge, conc)
    mae = m["mae"] if m["mae"] is not None else 1e9
    sp = m["spearman"] if m["spearman"] is not None else -1.0
    return (-mae, sp)


def _track_valid_mask(bank: V331FusionBank, compound_idx: np.ndarray, track: str) -> np.ndarray:
    if track == "smiles_only":
        return bank.smiles_valid[compound_idx].copy()
    return np.ones(len(compound_idx), dtype=bool)


def _predict_single(est: Any, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
    if D2_star is not None:
        inner = est.est if isinstance(est, BoundedWrapper) else est
        if isinstance(inner, (ExactRBFKRR, PrecomputedRBFSVR)):
            p = inner.predict(X, D2_star=D2_star)
            if isinstance(est, BoundedWrapper) and est.bounds is not None:
                p = np.clip(p, est.bounds[0], est.bounds[1])
            return p.astype(np.float64)
    return np.asarray(est.predict(X), dtype=np.float64)


def _fast_ridge_oof_sweep(
    fold_entries: List[Optional[Dict[str, Any]]],
    y_cv: np.ndarray,
    n_cv: int,
    alphas: List[float],
    bounds: Optional[Tuple[float, float]] = None,
    winsor_cap: Optional[float] = None,
) -> Dict[float, np.ndarray]:
    oofs = {float(a): np.full(n_cv, np.nan, dtype=np.float64) for a in alphas}
    for entry in fold_entries:
        if entry is None:
            continue
        tr_pos, va_pos = entry["tr_pos"], entry["va_pos"]
        X_tr = entry["X_tr"].astype(np.float64)
        X_va = entry["X_va"].astype(np.float64)
        y_tr = y_cv[tr_pos].astype(np.float64)
        if winsor_cap is not None:
            y_tr = np.clip(y_tr, None, winsor_cap)
        x_mu = X_tr.mean(axis=0)
        y_mu = float(y_tr.mean())
        X_tr_c = X_tr - x_mu
        X_va_c = X_va - x_mu
        y_tr_c = y_tr - y_mu
        n_tr, d = X_tr_c.shape
        if n_tr <= d:
            s, U = eigh(X_tr_c @ X_tr_c.T)
            s = np.maximum(s, 0.0)
            Uty = U.T @ y_tr_c
            K_va = X_va_c @ X_tr_c.T
            for a in alphas:
                p = K_va @ (U @ (Uty / (s + float(a)))) + y_mu
                if bounds is not None:
                    p = np.clip(p, bounds[0], bounds[1])
                oofs[float(a)][va_pos] = p
        else:
            s, V = eigh(X_tr_c.T @ X_tr_c)
            s = np.maximum(s, 0.0)
            Vtxy = V.T @ (X_tr_c.T @ y_tr_c)
            XV = X_va_c @ V
            for a in alphas:
                p = XV @ (Vtxy / (s + float(a))) + y_mu
                if bounds is not None:
                    p = np.clip(p, bounds[0], bounds[1])
                oofs[float(a)][va_pos] = p
    return oofs


def refit_5fold_and_predict_ho(
    bank: V331FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    recipe: Dict[str, Any],
    n_jobs: int,
    is_aft: bool = False,
) -> Dict[str, Any]:
    cv_rows, ho_rows = ep_data["cv_idx"], ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)
    cidx_cv, cidx_ho = ep_data["compound_idx"][cv_rows], ep_data["compound_idx"][ho_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    cond_ho = [ep_data["cond_rows"][i] for i in ho_rows]
    bounds = spec.get("bounds")

    valid_cv = _track_valid_mask(bank, cidx_cv, rep_cfg["track"])
    valid_ho = _track_valid_mask(bank, cidx_ho, rep_cfg["track"])
    ho_vpos = np.where(valid_ho)[0]
    ho_cond_v = [cond_ho[i] for i in ho_vpos]

    oof_pred = np.full(n_cv, np.nan, dtype=np.float64)
    ho_fold_preds: List[np.ndarray] = []
    fitted_folds: List[Tuple[int, EmbeddingFoldPreprocessor, Any]] = []

    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv
        if not is_aft:
            tr_mask = tr_mask & (ct_cv == "exact")
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            continue
        tr_pos, va_pos = np.where(tr_mask)[0], np.where(va_mask)[0]
        pre = EmbeddingFoldPreprocessor(
            blocks=rep_cfg["blocks"],
            phys_mode=rep_cfg["phys_mode"],
            cond_cols=ep_data["cond_cols"],
            mode=recipe["mode"],
            pca_group_dims=rep_cfg["pca_group_dims"],
            random_state=42 + f,
        )
        X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
        X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])
        if is_aft:
            est = XGBoostAFTRegressor(**recipe["params"], nthread=n_jobs, seed=42 + f, bounds=bounds)
            est.fit(X_tr, y_cv[tr_pos], yl_cv[tr_pos], yu_cv[tr_pos], ct_cv[tr_pos])
            oof_pred[va_pos] = est.predict(X_va)
        else:
            est = make_fusion_estimator(
                family=recipe["family"],
                params=recipe["params"],
                task_type="regression",
                n_jobs=n_jobs,
                bounds=bounds,
                winsor_cap=recipe.get("winsor_cap"),
                seed=42 + f,
            )
            est.fit(X_tr, y_cv[tr_pos])
            oof_pred[va_pos] = _predict_single(est, X_va)

        if len(ho_vpos) > 0:
            X_ho_v = pre.transform(bank, cidx_ho[ho_vpos], cond_rows=ho_cond_v)
            ho_fold_preds.append(est.predict(X_ho_v) if is_aft else _predict_single(est, X_ho_v))
        fitted_folds.append((f, pre, est))

    ho_pred = np.full(n_ho, np.nan, dtype=np.float64)
    ho_sd = np.full(n_ho, np.nan, dtype=np.float64)
    if len(ho_vpos) > 0 and len(ho_fold_preds) > 0:
        P_ho = np.stack(ho_fold_preds, axis=0)
        ho_pred[ho_vpos] = np.mean(P_ho, axis=0)
        ho_sd[ho_vpos] = np.std(P_ho, axis=0, ddof=1) if P_ho.shape[0] > 1 else np.zeros(len(ho_vpos))

    m_cv = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
    m_ho = regression_metrics(ep_data["y_value"][ho_rows], ho_pred, ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows])
    return {
        "rep_name": rep_cfg["name"], "rep_cfg": rep_cfg, "recipe": recipe, "is_aft": is_aft,
        "oof_pred": oof_pred, "ho_pred": ho_pred, "ho_sd": ho_sd,
        "cv_metrics": m_cv, "ho_metrics": m_ho, "folds": fitted_folds,
    }


def benchmark_representation(
    bank: V331FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    n_jobs: int,
    is_direct_fusion: bool = False,
    keep_both_families: bool = False,
) -> Dict[str, Any]:
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge

    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    bounds = spec.get("bounds")
    winsor_cap = spec.get("outlier_winsor_cap")
    valid_cv = _track_valid_mask(bank, cidx_cv, rep_cfg["track"])

    cands = get_rep_candidates(size_tier=spec["size_tier"], task_type="regression", n_cv=n_cv, is_direct_fusion=is_direct_fusion, winsor_cap=winsor_cap)
    bal_cands = [i for i, c in enumerate(cands) if c["mode"] == "balanced"]
    pca_cands = [i for i, c in enumerate(cands) if c["mode"] in ("pca_gbdt", "pca_mlp")]
    oofs = [np.full(n_cv, np.nan, dtype=np.float64) for _ in cands]
    errors: Dict[int, str] = {}
    needs_d2 = any(cands[i]["family"] in ("KRR_RBF", "SVR_RBF") for i in bal_cands)

    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv & (ct_cv == "exact")
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            continue
        tr_pos, va_pos = np.where(tr_mask)[0], np.where(va_mask)[0]
        y_tr = y_cv[tr_pos]

        if bal_cands:
            pre_b = EmbeddingFoldPreprocessor(
                blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
                mode="balanced", pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f,
            )
            X_tr = pre_b.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            X_va = pre_b.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])

            D2_tr = D2_va = base_gamma = None
            if needs_d2:
                use_f32 = len(tr_pos) > 7500
                D2_tr = compute_sqdist_matrix(X_tr, X_tr, use_float32=use_f32)
                D2_va = compute_sqdist_matrix(X_va, X_tr, use_float32=use_f32)
                if len(tr_pos) > 1500:
                    sub_m = np.random.default_rng(42 + f).choice(len(tr_pos), size=1000, replace=False)
                    dsub = D2_tr[np.ix_(sub_m, sub_m)]
                    tri = dsub[np.triu_indices(len(sub_m), k=1)]
                else:
                    tri = D2_tr[np.triu_indices(len(tr_pos), k=1)]
                med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_tr.shape[1]))
                base_gamma = 1.0 / max(med_d2, 1e-6)

            ridge_ids = [i for i in bal_cands if cands[i]["family"] == "Ridge"]
            if ridge_ids:
                alphas = [float(cands[i]["params"]["alpha"]) for i in ridge_ids]
                r_oofs = _fast_ridge_oof_sweep([{"tr_pos": tr_pos, "va_pos": va_pos, "X_tr": X_tr, "X_va": X_va}], y_cv, n_cv, alphas, bounds=bounds, winsor_cap=winsor_cap)
                for i in ridge_ids:
                    oofs[i][va_pos] = r_oofs[float(cands[i]["params"]["alpha"])][va_pos]

            nys_ids = [i for i in bal_cands if cands[i]["family"] == "Nystrom_RBF"]
            if nys_ids:
                try:
                    n_comp = min(int(cands[nys_ids[0]]["params"].get("n_components", 800)), len(tr_pos))
                    gm = float(cands[nys_ids[0]]["params"].get("gamma_mult", 1.0))
                    sub = np.random.default_rng(42 + f).choice(len(tr_pos), size=min(len(tr_pos), 600), replace=False)
                    D2s = compute_sqdist_matrix(X_tr[sub], X_tr[sub])
                    tris = D2s[np.triu_indices(len(sub), k=1)]
                    md2 = float(np.median(tris[tris > 0])) if (tris > 0).any() else float(max(1, X_tr.shape[1]))
                    nys = Nystroem(kernel="rbf", gamma=gm / max(md2, 1e-6), n_components=n_comp, random_state=42 + f)
                    Z_tr = nys.fit_transform(X_tr).astype(np.float32)
                    Z_va = nys.transform(X_va).astype(np.float32)
                    y_fit = np.clip(y_tr, None, winsor_cap) if winsor_cap is not None else y_tr
                    for i in nys_ids:
                        p = Ridge(alpha=float(cands[i]["params"]["alpha"])).fit(Z_tr, y_fit).predict(Z_va).astype(np.float64)
                        oofs[i][va_pos] = np.clip(p, bounds[0], bounds[1]) if bounds is not None else p
                    del Z_tr, Z_va
                except Exception as exc:
                    for i in nys_ids:
                        errors[i] = str(exc)

            for i in bal_cands:
                if i in ridge_ids or i in nys_ids or i in errors:
                    continue
                c = cands[i]
                try:
                    est = make_fusion_estimator(family=c["family"], params=c["params"], task_type="regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=c.get("winsor_cap"), seed=42 + f)
                    inner = est.est if isinstance(est, BoundedWrapper) else est
                    if D2_tr is not None and isinstance(inner, (ExactRBFKRR, PrecomputedRBFSVR)):
                        y_fit = np.clip(y_tr, None, est.winsor_cap) if (isinstance(est, BoundedWrapper) and est.winsor_cap is not None) else y_tr
                        inner.fit(X_tr, y_fit, D2_pre=D2_tr, base_gamma=base_gamma)
                        oofs[i][va_pos] = _predict_single(est, X_va, D2_star=D2_va)
                    else:
                        est.fit(X_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, X_va)
                except Exception as exc:
                    errors[i] = str(exc)
            del X_tr, X_va, D2_tr, D2_va, pre_b

        if pca_cands:
            pre_p = EmbeddingFoldPreprocessor(
                blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
                mode="pca_gbdt", pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f,
            )
            Xp_tr = pre_p.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            Xp_va = pre_p.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])
            has_mlp = any(cands[i]["mode"] == "pca_mlp" for i in pca_cands)
            Xm_tr = Xm_va = None
            if has_mlp:
                n_c = len(pre_p.cond_enc_.feature_names_) if pre_p.cond_enc_ is not None else 0
                cut = Xp_tr.shape[1] - n_c
                Xm_tr, Xm_va = Xp_tr.copy(), Xp_va.copy()
                if cut > 0:
                    mu = Xp_tr[:, :cut].mean(axis=0)
                    sd = Xp_tr[:, :cut].std(axis=0)
                    sd[sd < 1e-6] = 1.0
                    Xm_tr[:, :cut] = np.clip((Xp_tr[:, :cut] - mu) / sd, -8.0, 8.0).astype(np.float32)
                    Xm_va[:, :cut] = np.clip((Xp_va[:, :cut] - mu) / sd, -8.0, 8.0).astype(np.float32)

            for i in pca_cands:
                if i in errors:
                    continue
                c = cands[i]
                try:
                    est = make_fusion_estimator(family=c["family"], params=c["params"], task_type="regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=c.get("winsor_cap"), seed=42 + f)
                    if c["mode"] == "pca_mlp":
                        est.fit(Xm_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xm_va)
                    else:
                        est.fit(Xp_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xp_va)
                except Exception as exc:
                    errors[i] = str(exc)
            del Xp_tr, Xp_va, Xm_tr, Xm_va, pre_p

    valid_res = []
    for i, c in enumerate(cands):
        if i in errors:
            continue
        m = regression_metrics(y_cv, oofs[i], yl_cv, yu_cv, ct_cv)
        valid_res.append({"recipe": c, "metrics": m, "score": _score_metrics(m), "oof_pred": oofs[i]})
    assert valid_res, f"{ep_data['endpoint']} {rep_cfg['name']}: all base learners failed!"
    valid_res.sort(key=lambda x: x["score"], reverse=True)

    best_rec = valid_res[0]["recipe"]
    best_overall = refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, best_rec, n_jobs=n_jobs, is_aft=False)

    out: Dict[str, Any] = {"best_overall": best_overall, "best_smooth": None, "best_tree": None}
    if keep_both_families:
        smooth_sub = [r for r in valid_res if r["recipe"]["family"] not in ("LightGBM", "XGBoost")]
        tree_sub = [r for r in valid_res if r["recipe"]["family"] in ("LightGBM", "XGBoost")]
        if smooth_sub:
            out["best_smooth"] = best_overall if smooth_sub[0]["recipe"] == best_rec else refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, smooth_sub[0]["recipe"], n_jobs=n_jobs)
        if tree_sub:
            out["best_tree"] = best_overall if tree_sub[0]["recipe"] == best_rec else refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, tree_sub[0]["recipe"], n_jobs=n_jobs)
    return out


def benchmark_multimodal_aft(
    bank: V331FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    n_jobs: int,
) -> Dict[str, Any]:
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    bounds = spec.get("bounds")
    valid_cv = _track_valid_mask(bank, cidx_cv, rep_cfg["track"])

    f_cache = []
    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            f_cache.append(None)
            continue
        tr_pos, va_pos = np.where(tr_mask)[0], np.where(va_mask)[0]
        pre = EmbeddingFoldPreprocessor(blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"], mode="pca_gbdt", pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f)
        f_cache.append({"fold": f, "tr_pos": tr_pos, "va_pos": va_pos, "X_tr": pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos]), "X_va": pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])})

    aft_params_list = get_multimodal_aft_candidates(spec["size_tier"])
    best_score, best_params = (-1e12, -1e12), aft_params_list[0]
    for p in aft_params_list:
        oof_p = np.full(n_cv, np.nan, dtype=np.float64)
        for entry in f_cache:
            if entry is None:
                continue
            tr_pos, va_pos = entry["tr_pos"], entry["va_pos"]
            est = XGBoostAFTRegressor(**p, nthread=n_jobs, seed=42 + entry["fold"], bounds=bounds)
            est.fit(entry["X_tr"], y_cv[tr_pos], yl_cv[tr_pos], yu_cv[tr_pos], ct_cv[tr_pos])
            oof_p[va_pos] = est.predict(entry["X_va"])
        sc = _score_metrics(regression_metrics(y_cv, oof_p, yl_cv, yu_cv, ct_cv), is_aft=True)
        if sc > best_score:
            best_score, best_params = sc, p
    del f_cache
    return refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, {"mode": "pca_gbdt", "family": "XGBoost_AFT", "params": best_params}, n_jobs=n_jobs, is_aft=True)


def run_nested_stacking(
    head_names: List[str],
    oof_dict: Dict[str, np.ndarray],
    ho_dict: Dict[str, np.ndarray],
    y_cv: np.ndarray, yl_cv: np.ndarray, yu_cv: np.ndarray, ct_cv: np.ndarray, fids_cv: np.ndarray,
    y_ho: np.ndarray, yl_ho: np.ndarray, yu_ho: np.ndarray, ct_ho: np.ndarray,
    bounds: Optional[Tuple[float, float]],
    eval_mask_cv: np.ndarray, eval_mask_ho: np.ndarray,
    is_aft_stack: bool = False,
) -> Dict[str, Any]:
    n_cv, n_ho = len(y_cv), len(y_ho)
    P_oof = np.column_stack([oof_dict[h] for h in head_names])
    P_ho = np.column_stack([ho_dict[h] for h in head_names])
    has_nan = bool((~np.isfinite(P_oof[eval_mask_cv])).any() or (~np.isfinite(P_ho[eval_mask_ho])).any())
    n_eff = int(eval_mask_cv.sum())

    if is_aft_stack:
        meta_cands = [{"method": "best_single", "params": {}}, {"method": "convex_hinge", "params": {"reg": 0.0}}, {"method": "convex_hinge", "params": {"reg": 0.02}}, {"method": "convex_hinge", "params": {"reg": 0.08}}, {"method": "topk_blend", "params": {"k": 2}}]
    else:
        meta_cands = [
            {"method": "best_single", "params": {}},
            {"method": "topk_blend", "params": {"k": 2}}, {"method": "topk_blend", "params": {"k": 3}}, {"method": "topk_blend", "params": {"k": 4}},
            {"method": "convex_simplex", "params": {"reg": 0.0}}, {"method": "convex_simplex", "params": {"reg": 0.01}}, {"method": "convex_simplex", "params": {"reg": 0.05}}, {"method": "convex_simplex", "params": {"reg": 0.15}},
        ]
        if n_eff >= 120 and not has_nan:
            for alpha in (5.0, 25.0, 100.0):
                meta_cands.append({"method": "ridge_nonneg", "params": {"alpha": alpha}})
            if n_eff >= 250:
                for alpha in (15.0, 60.0):
                    meta_cands.append({"method": "ridge_l2", "params": {"alpha": alpha}})

    best_cfg, best_score, best_nested = None, (-1e12, -1e12), None
    for mc in meta_cands:
        nested = np.full(n_cv, np.nan, dtype=np.float64)
        for f in range(5):
            va = (fids_cv == f) & eval_mask_cv
            tr = (fids_cv != f) & eval_mask_cv
            if not is_aft_stack:
                tr = tr & (ct_cv == "exact")
            if tr.sum() < 4 or va.sum() == 0:
                continue
            st = ModalityAwareStacker(task_type="regression", method=mc["method"], params=mc["params"], bounds=bounds)
            st.fit(P_oof[tr], y_cv[tr], head_names=head_names, y_lower=yl_cv[tr] if is_aft_stack else None, y_upper=yu_cv[tr] if is_aft_stack else None, censor_type=ct_cv[tr] if is_aft_stack else None)
            nested[va] = st.predict(P_oof[va])
        m = regression_metrics(y_cv[eval_mask_cv], nested[eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv])
        sc = _score_metrics(m, is_aft=is_aft_stack)
        if sc > best_score:
            best_score, best_cfg, best_nested = sc, mc, nested

    assert best_cfg is not None and best_nested is not None
    fit_mask = eval_mask_cv.copy() if is_aft_stack else (eval_mask_cv & (ct_cv == "exact"))
    final_st = ModalityAwareStacker(task_type="regression", method=best_cfg["method"], params=best_cfg["params"], bounds=bounds)
    final_st.fit(P_oof[fit_mask], y_cv[fit_mask], head_names=head_names, y_lower=yl_cv[fit_mask] if is_aft_stack else None, y_upper=yu_cv[fit_mask] if is_aft_stack else None, censor_type=ct_cv[fit_mask] if is_aft_stack else None)

    ho_pred = np.full(n_ho, np.nan, dtype=np.float64)
    if eval_mask_ho.any():
        ho_pred[eval_mask_ho] = final_st.predict(P_ho[eval_mask_ho])
    return {
        "best_cfg": best_cfg, "stacker": final_st, "weights": final_st.weight_dict(), "active_heads": final_st.active_heads(),
        "nested_oof": best_nested, "ho_pred": ho_pred,
        "cv_metrics": regression_metrics(y_cv[eval_mask_cv], best_nested[eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv]),
        "ho_metrics": regression_metrics(y_ho[eval_mask_ho], ho_pred[eval_mask_ho], yl_ho[eval_mask_ho], yu_ho[eval_mask_ho], ct_ho[eval_mask_ho]),
    }


def assign_balanced_cluster_folds(cluster_ids: np.ndarray, n_folds: int = 5) -> np.ndarray:
    cl_to_pos: Dict[str, List[int]] = {}
    for i, c in enumerate(cluster_ids):
        cl_to_pos.setdefault(str(c), []).append(i)
    sorted_cls = sorted(cl_to_pos.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    fold_counts = [0] * n_folds
    fold_ids = np.zeros(len(cluster_ids), dtype=np.int32)
    for _, pos_list in sorted_cls:
        best_f = int(np.argmin(fold_counts))
        for p in pos_list:
            fold_ids[p] = best_f
        fold_counts[best_f] += len(pos_list)
    return fold_ids


def refit_head_leave_case17_out(bank: V331FusionBank, ep_data: Dict[str, Any], spec: Dict[str, Any], head_info: Dict[str, Any], c17_idx: np.ndarray, n_jobs: int) -> Dict[str, Any]:
    keep_rows = np.where(ep_data["case17_excluded"] == 0)[0]
    cidx = ep_data["compound_idx"][keep_rows]
    c17_set = set(int(x) for x in c17_idx)
    assert all(int(x) not in c17_set for x in cidx)
    clids = ep_data["cluster_ids"][keep_rows]
    y, yl, yu, ct = ep_data["y_value"][keep_rows], ep_data["y_lower"][keep_rows], ep_data["y_upper"][keep_rows], ep_data["censor_type"][keep_rows]
    conds = [ep_data["cond_rows"][i] for i in keep_rows]
    rep_cfg, recipe, is_aft = head_info["rep_cfg"], head_info["recipe"], bool(head_info["is_aft"])
    bounds = spec.get("bounds")
    valid = _track_valid_mask(bank, cidx, rep_cfg["track"])
    fids = assign_balanced_cluster_folds(clids, n_folds=5)
    canon_cond = spec["canonical_conditions"]
    alt_conds = spec.get("alt_case17_conditions", {})
    canon_preds, alt_preds = [], {k: [] for k in alt_conds}

    for f in range(5):
        tr_mask = (fids != f) & valid
        if not is_aft:
            tr_mask = tr_mask & (ct == "exact")
        if tr_mask.sum() < 4:
            continue
        tr_pos = np.where(tr_mask)[0]
        pre = EmbeddingFoldPreprocessor(blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"], mode=recipe["mode"], pca_group_dims=rep_cfg["pca_group_dims"], random_state=142 + f)
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos])
        if is_aft:
            est = XGBoostAFTRegressor(**recipe["params"], nthread=n_jobs, seed=142 + f, bounds=bounds)
            est.fit(X_tr, y[tr_pos], yl[tr_pos], yu[tr_pos], ct[tr_pos])
            canon_preds.append(est.predict(pre.transform(bank, c17_idx, default_cond=canon_cond)))
            for ak, ac in alt_conds.items():
                alt_preds[ak].append(est.predict(pre.transform(bank, c17_idx, default_cond=ac)))
        else:
            est = make_fusion_estimator(family=recipe["family"], params=recipe["params"], task_type="regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=recipe.get("winsor_cap"), seed=142 + f)
            est.fit(X_tr, y[tr_pos])
            canon_preds.append(_predict_single(est, pre.transform(bank, c17_idx, default_cond=canon_cond)))
            for ak, ac in alt_conds.items():
                alt_preds[ak].append(_predict_single(est, pre.transform(bank, c17_idx, default_cond=ac)))
    P_c = np.stack(canon_preds, axis=0)
    return {"mean": np.mean(P_c, axis=0), "sd": np.std(P_c, axis=0, ddof=1) if P_c.shape[0] > 1 else np.zeros(len(c17_idx)), "alt": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in alt_preds.items()}}


def run_single_endpoint(args: Tuple[str, int]) -> Dict[str, Any]:
    endpoint, n_jobs = args
    t0 = time.time()
    bank = GLOBAL_BANK
    assert bank is not None
    spec = V331_ENDPOINT_SPECS[endpoint]
    ep_data = bank.load_endpoint_and_cpu(endpoint)
    bounds = spec.get("bounds")
    cv_rows, ho_rows = ep_data["cv_idx"], ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)

    y_cv, yl_cv, yu_cv, ct_cv, fids_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows], ep_data["fold_id"][cv_rows]
    y_ho, yl_ho, yu_ho, ct_ho = ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
    cidx_cv, cidx_ho = ep_data["compound_idx"][cv_rows], ep_data["compound_idx"][ho_rows]
    smi_cv_mask, smi_ho_mask = bank.smiles_valid[cidx_cv], bank.smiles_valid[cidx_ho]
    dual_cv_mask, dual_ho_mask = np.ones(n_cv, dtype=bool), np.ones(n_ho, dtype=bool)

    print(f"[FUS START] {endpoint:<22} (n={ep_data['n']}, cv={n_cv}, ho={n_ho}, jobs={n_jobs})", flush=True)
    cpu = ep_data["cpu"]
    rep_ablation_doc: Dict[str, Any] = {
        "cpu_smiles_only": {"label": "CPU (smiles_only)", "track": "smiles_only", "cv_metrics": regression_metrics(y_cv[smi_cv_mask], cpu["oof_smiles_only"][smi_cv_mask], yl_cv[smi_cv_mask], yu_cv[smi_cv_mask], ct_cv[smi_cv_mask]), "ho_metrics": regression_metrics(y_ho[smi_ho_mask], cpu["ho_smiles_only"][smi_ho_mask], yl_ho[smi_ho_mask], yu_ho[smi_ho_mask], ct_ho[smi_ho_mask])},
        "cpu_dual_modality": {"label": "CPU (dual_modality)", "track": "dual_modality", "cv_metrics": regression_metrics(y_cv, cpu["oof_dual"], yl_cv, yu_cv, ct_cv), "ho_metrics": regression_metrics(y_ho, cpu["ho_dual"], yl_ho, yu_ho, ct_ho)},
    }
    trained_heads: Dict[str, Dict[str, Any]] = {}

    for rep_cfg in INDIVIDUAL_REPS:
        rname = rep_cfg["name"]
        keep_both = (rname == "esmc_multiregion") and (n_cv >= 200)
        res = benchmark_representation(bank, ep_data, spec, rep_cfg, n_jobs=n_jobs, is_direct_fusion=False, keep_both_families=keep_both)
        bo = res["best_overall"]
        rep_ablation_doc[rname] = {"label": rep_cfg["label"], "track": rep_cfg["track"], "winning_family": bo["recipe"]["family"], "winning_mode": bo["recipe"]["mode"], "cv_metrics": bo["cv_metrics"], "ho_metrics": bo["ho_metrics"]}
        trained_heads[rname] = bo
        if keep_both:
            if res["best_smooth"] is not None:
                trained_heads[f"{rname}_smooth"] = res["best_smooth"]
            if res["best_tree"] is not None:
                trained_heads[f"{rname}_tree"] = res["best_tree"]

    for dname in ("direct_smiles", "direct_dual"):
        rep_cfg = DIRECT_FUSION_REPS[dname]
        res = benchmark_representation(bank, ep_data, spec, rep_cfg, n_jobs=n_jobs, is_direct_fusion=True, keep_both_families=True)
        bo = res["best_overall"]
        rep_ablation_doc[dname] = {"label": rep_cfg["label"], "track": rep_cfg["track"], "winning_family": bo["recipe"]["family"], "winning_mode": bo["recipe"]["mode"], "cv_metrics": bo["cv_metrics"], "ho_metrics": bo["ho_metrics"]}
        trained_heads[dname] = bo
        if res["best_smooth"] is not None:
            trained_heads[f"{dname}_smooth"] = res["best_smooth"]
        if res["best_tree"] is not None:
            trained_heads[f"{dname}_tree"] = res["best_tree"]

    n_cens_cv = int((ct_cv != "exact").sum())
    aft_direct_dual = aft_direct_smi = aft_stack_dual = aft_stack_smi = None
    if n_cens_cv >= 1:
        aft_direct_smi = benchmark_multimodal_aft(bank, ep_data, spec, DIRECT_FUSION_REPS["direct_smiles"], n_jobs=n_jobs)
        aft_direct_dual = benchmark_multimodal_aft(bank, ep_data, spec, DIRECT_FUSION_REPS["direct_dual"], n_jobs=n_jobs)
        trained_heads["aft_direct_smiles"] = aft_direct_smi
        trained_heads["aft_direct_dual"] = aft_direct_dual
        aft_stack_smi = run_nested_stacking(
            ["cpu_aft_smiles", "aft_direct_smiles"], {"cpu_aft_smiles": cpu["oof_aft_smiles_only"], "aft_direct_smiles": aft_direct_smi["oof_pred"]}, {"cpu_aft_smiles": cpu["ho_aft_smiles_only"], "aft_direct_smiles": aft_direct_smi["ho_pred"]},
            y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, smi_cv_mask, smi_ho_mask, is_aft_stack=True,
        )
        aft_stack_dual = run_nested_stacking(
            ["cpu_aft_dual", "aft_direct_dual"], {"cpu_aft_dual": cpu["oof_aft"], "aft_direct_dual": aft_direct_dual["oof_pred"]}, {"cpu_aft_dual": cpu["ho_aft"], "aft_direct_dual": aft_direct_dual["ho_pred"]},
            y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, dual_cv_mask, dual_ho_mask, is_aft_stack=True,
        )

    all_oof = {"cpu_smiles": cpu["oof_smiles_only"], "cpu_dual": cpu["oof_dual"]}
    all_ho = {"cpu_smiles": cpu["ho_smiles_only"], "cpu_dual": cpu["ho_dual"]}
    if cpu["oof_aft"] is not None:
        all_oof["cpu_aft_dual"], all_ho["cpu_aft_dual"] = cpu["oof_aft"], cpu["ho_aft"]
    if cpu["oof_aft_smiles_only"] is not None:
        all_oof["cpu_aft_smiles"], all_ho["cpu_aft_smiles"] = cpu["oof_aft_smiles_only"], cpu["ho_aft_smiles_only"]
    if cpu["oof_seq_only"] is not None:
        all_oof["cpu_seq_only"], all_ho["cpu_seq_only"] = cpu["oof_seq_only"], cpu["ho_seq_only"]
    for hname, hobj in trained_heads.items():
        all_oof[hname], all_ho[hname] = hobj["oof_pred"], hobj["ho_pred"]

    track_a_heads = ["cpu_smiles", "ncaa_smiles_repr", "ncaa_fusion_hf", "ncaa_protein_proj", "peptideclm2", "chemberta_mtr"]
    for extra in ("direct_smiles_smooth", "direct_smiles_tree"):
        if extra in all_oof:
            track_a_heads.append(extra)
    if aft_direct_smi is not None:
        track_a_heads.extend(["cpu_aft_smiles", "aft_direct_smiles"])

    stack_a = run_nested_stacking(track_a_heads, all_oof, all_ho, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, smi_cv_mask, smi_ho_mask)

    track_b_heads = ["cpu_dual", "ncaa_smiles_repr", "ncaa_fusion_hf", "ncaa_protein_proj", "peptideclm2", "chemberta_mtr", "esmc_mean"]
    if "esmc_multiregion_smooth" in all_oof:
        track_b_heads.append("esmc_multiregion_smooth")
    if "esmc_multiregion_tree" in all_oof:
        track_b_heads.append("esmc_multiregion_tree")
    if "esmc_multiregion_smooth" not in all_oof and "esmc_multiregion_tree" not in all_oof:
        track_b_heads.append("esmc_multiregion")
    for extra in ("direct_smiles_smooth", "direct_dual_smooth", "direct_dual_tree"):
        if extra in all_oof:
            track_b_heads.append(extra)
    if "cpu_seq_only" in all_oof and n_cv >= 300:
        track_b_heads.append("cpu_seq_only")
    if aft_direct_dual is not None:
        track_b_heads.extend(["cpu_aft_dual", "aft_direct_dual"])
    track_b_heads = list(dict.fromkeys(track_b_heads))

    stack_b = run_nested_stacking(track_b_heads, all_oof, all_ho, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, dual_cv_mask, dual_ho_mask)

    ad_index = TanimotoADIndex.from_indices(bank, cidx_cv)
    oof_max_t = cpu["oof_max_tanimoto"].astype(np.float32)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, cidx_ho)
    conf_smi = ConformalIntervalCalibrator(bounds=bounds).fit(y_cv[smi_cv_mask], stack_a["nested_oof"][smi_cv_mask], ct_cv[smi_cv_mask], oof_max_tanimoto=oof_max_t[smi_cv_mask])
    conf_dual = ConformalIntervalCalibrator(bounds=bounds).fit(y_cv, stack_b["nested_oof"], ct_cv, oof_max_tanimoto=oof_max_t)
    ho_iv_smi = conf_smi.predict_intervals(stack_a["ho_pred"], max_tanimoto=ho_max_t)
    ho_iv_dual = conf_dual.predict_intervals(stack_b["ho_pred"], max_tanimoto=ho_max_t)
    ad_index.tier_calibration = compute_ad_stratification(y_ho, stack_b["ho_pred"], ct_ho, ho_max_t, ho_ad_tiers, "regression")

    # Strict Zero-Leakage Leave-Case17-Out Refit
    c17_idx = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)
    keep_non_c17 = np.where(ep_data["case17_excluded"] == 0)[0]
    non_c17_cv_mask = ep_data["case17_excluded"][cv_rows] == 0

    def _fit_nc17(head_names, stack_res, extra_mask, is_aft_st=False):
        m_fit = non_c17_cv_mask & extra_mask
        if not is_aft_st:
            m_fit = m_fit & (ct_cv == "exact")
        if m_fit.sum() < 6:
            return stack_res["stacker"]
        st = ModalityAwareStacker(task_type="regression", method=stack_res["best_cfg"]["method"], params=stack_res["best_cfg"]["params"], bounds=bounds)
        st.fit(np.column_stack([all_oof[h][m_fit] for h in head_names]), y_cv[m_fit], head_names=head_names, y_lower=yl_cv[m_fit] if is_aft_st else None, y_upper=yu_cv[m_fit] if is_aft_st else None, censor_type=ct_cv[m_fit] if is_aft_st else None)
        return st

    c17_st_a = _fit_nc17(track_a_heads, stack_a, smi_cv_mask)
    c17_st_b = _fit_nc17(track_b_heads, stack_b, dual_cv_mask)
    c17_st_aft = _fit_nc17(["cpu_aft_dual", "aft_direct_dual"], aft_stack_dual, dual_cv_mask, True) if aft_stack_dual is not None else None

    needed_h = set(c17_st_a.active_heads()) | set(c17_st_b.active_heads())
    if c17_st_aft is not None:
        needed_h |= set(c17_st_aft.active_heads())
    needed_h = {h for h in needed_h if h in trained_heads}

    c17_base: Dict[str, np.ndarray] = {"cpu_smiles": cpu["c17_smiles_only"], "cpu_dual": cpu["c17_dual"]}
    c17_base_alt: Dict[str, Dict[str, np.ndarray]] = {}
    for ak, ad in cpu["c17_alt"].items():
        c17_base_alt[ak] = {"cpu_smiles": ad["smiles_only"], "cpu_dual": ad["dual"]}
        if ad.get("aft") is not None:
            c17_base_alt[ak]["cpu_aft_dual"] = ad["aft"]
            c17_base_alt[ak]["cpu_aft_smiles"] = ad["aft"]
    if cpu["c17_aft"] is not None:
        c17_base["cpu_aft_dual"] = cpu["c17_aft"]
        c17_base["cpu_aft_smiles"] = cpu["c17_aft"]
    if cpu["c17_seq_only"] is not None:
        c17_base["cpu_seq_only"] = cpu["c17_seq_only"]

    for hname in sorted(needed_h):
        c17_r = refit_head_leave_case17_out(bank, ep_data, spec, trained_heads[hname], c17_idx, n_jobs=n_jobs)
        c17_base[hname] = c17_r["mean"]
        for ak, arr in c17_r["alt"].items():
            c17_base_alt.setdefault(ak, {})[hname] = arr

    def _comb(st, hnames, smap):
        return st.predict(np.column_stack([smap[h] if (h in smap and smap[h] is not None) else np.full(len(c17_idx), np.nan) for h in hnames]))

    c17_p_smi = _comb(c17_st_a, track_a_heads, c17_base)
    c17_p_dual = _comb(c17_st_b, track_b_heads, c17_base)
    c17_p_aft = _comb(c17_st_aft, ["cpu_aft_dual", "aft_direct_dual"], c17_base) if c17_st_aft is not None else None
    c17_alt_out = {}
    for ak in spec.get("alt_case17_conditions", {}):
        asrc = dict(c17_base)
        asrc.update(c17_base_alt.get(ak, {}))
        c17_alt_out[ak] = {"smiles_only": _comb(c17_st_a, track_a_heads, asrc), "dual": _comb(c17_st_b, track_b_heads, asrc), "aft": _comb(c17_st_aft, ["cpu_aft_dual", "aft_direct_dual"], asrc) if c17_st_aft is not None else None}

    ad_c17 = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][keep_non_c17])
    c17_max_t, c17_top5_t, c17_ad_tiers = ad_c17.query(bank, c17_idx)

    # Write standardized outputs
    ep_out_dir = EXP_FUSION_DIR / endpoint
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.6f}"
        return str(v)

    oof_cols = ["record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type", "pred_smiles_only", "pred_dual"]
    if aft_stack_dual is not None:
        oof_cols.extend(["pred_aft", "pred_aft_smiles_only", "pred_aft_direct_dual"])
    oof_cols.append("max_tanimoto_oof")
    with (ep_out_dir / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(oof_cols)
        for loc_i, orig_i in enumerate(cv_rows):
            row = [ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i], int(fids_cv[loc_i]), _fmt(y_cv[loc_i]), _fmt(yl_cv[loc_i]), _fmt(yu_cv[loc_i]), str(ct_cv[loc_i]), _fmt(stack_a["nested_oof"][loc_i]), _fmt(stack_b["nested_oof"][loc_i])]
            if aft_stack_dual is not None:
                row.extend([_fmt(aft_stack_dual["nested_oof"][loc_i]), _fmt(aft_stack_smi["nested_oof"][loc_i]), _fmt(aft_direct_dual["oof_pred"][loc_i])])
            row.append(_fmt(oof_max_t[loc_i]))
            w.writerow(row)

    ho_cols = ["record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type", "pred_smiles_only", "pred_dual"]
    if aft_stack_dual is not None:
        ho_cols.extend(["pred_aft", "pred_aft_smiles_only", "pred_aft_direct_dual"])
    ho_cols.extend(["lo80", "hi80", "lo90", "hi90", "max_tanimoto", "top5_tanimoto", "ad_tier"])
    with (ep_out_dir / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ho_cols)
        for loc_i, orig_i in enumerate(ho_rows):
            row = [ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i], _fmt(y_ho[loc_i]), _fmt(yl_ho[loc_i]), _fmt(yu_ho[loc_i]), str(ct_ho[loc_i]), _fmt(stack_a["ho_pred"][loc_i]), _fmt(stack_b["ho_pred"][loc_i])]
            if aft_stack_dual is not None:
                row.extend([_fmt(aft_stack_dual["ho_pred"][loc_i]), _fmt(aft_stack_smi["ho_pred"][loc_i]), _fmt(aft_direct_dual["ho_pred"][loc_i])])
            row.extend([_fmt(ho_iv_dual["lo80"][loc_i]), _fmt(ho_iv_dual["hi80"][loc_i]), _fmt(ho_iv_dual["lo90"][loc_i]), _fmt(ho_iv_dual["hi90"][loc_i]), _fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i]])
            w.writerow(row)

    c17_cols = ["drug_name", "compound_key", "compound_idx", "clinical_route", "condition_used", "pred_smiles_only", "pred_dual"]
    if c17_p_aft is not None:
        c17_cols.append("pred_aft")
    c17_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])
    for ak in c17_alt_out:
        c17_cols.extend([f"pred_smiles_only_{ak}", f"pred_dual_{ak}"])
        if c17_alt_out[ak]["aft"] is not None:
            c17_cols.append(f"pred_aft_{ak}")
    cond_str = json.dumps(spec["canonical_conditions"], ensure_ascii=False)
    with (ep_out_dir / "case17_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(c17_cols)
        for i, q in enumerate(bank.case17_queries):
            dn = q["drug_name"]
            row = [dn, q["compound_key"], int(q["compound_idx"]), CASE17_CLINICAL_ROUTE.get(dn, "IV"), cond_str, _fmt(c17_p_smi[i]), _fmt(c17_p_dual[i])]
            if c17_p_aft is not None:
                row.append(_fmt(c17_p_aft[i]))
            row.extend([_fmt(c17_max_t[i]), _fmt(c17_top5_t[i]), c17_ad_tiers[i]])
            for ak, ad in c17_alt_out.items():
                row.extend([_fmt(ad["smiles_only"][i]), _fmt(ad["dual"][i])])
                if ad["aft"] is not None:
                    row.append(_fmt(ad["aft"][i]))
            w.writerow(row)

    elapsed = round(time.time() - t0, 2)
    metrics_doc = {
        "endpoint": endpoint, "tier": spec["tier"], "task_type": "regression", "size_tier": spec["size_tier"], "unit": spec["unit"],
        "bounds": list(bounds) if bounds is not None else None, "cond_cols": ep_data["cond_cols"], "canonical_default_conditions": spec["canonical_conditions"],
        "n_total": ep_data["n"], "n_cv": n_cv, "n_holdout": n_ho, "n_case17_excluded": int((ep_data["case17_excluded"] == 1).sum()),
        "censor_counts": {k: int((ep_data["censor_type"] == k).sum()) for k in ("exact", "right", "left", "interval")},
        "representation_ablation": rep_ablation_doc,
        "fusion_ablation": {
            "smiles_only": {"candidate_heads": track_a_heads, "winning_meta": stack_a["best_cfg"], "weights": stack_a["weights"], "active_heads": stack_a["active_heads"], "nested_cv_metrics": stack_a["cv_metrics"], "holdout_metrics": stack_a["ho_metrics"]},
            "dual_modality": {"candidate_heads": track_b_heads, "winning_meta": stack_b["best_cfg"], "weights": stack_b["weights"], "active_heads": stack_b["active_heads"], "nested_cv_metrics": stack_b["cv_metrics"], "holdout_metrics": stack_b["ho_metrics"]},
        },
        "cv_metrics": {"smiles_only": stack_a["cv_metrics"], "dual_modality": stack_b["cv_metrics"]},
        "holdout_metrics": {"smiles_only": stack_a["ho_metrics"], "dual_modality": stack_b["ho_metrics"]},
        "elapsed_sec": elapsed,
    }
    (ep_out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    save_keys = set(stack_a["active_heads"]) | set(stack_b["active_heads"])
    if aft_stack_dual is not None:
        save_keys |= set(aft_stack_dual["active_heads"]) | set(aft_stack_smi["active_heads"])
    ser_heads = {
        hk: {"rep_name": trained_heads[hk]["rep_name"], "rep_cfg": trained_heads[hk]["rep_cfg"], "recipe": trained_heads[hk]["recipe"], "is_aft": trained_heads[hk]["is_aft"], "folds": trained_heads[hk]["folds"]}
        for hk in sorted(save_keys) if hk in trained_heads
    }
    model_bundle = {
        "schema": "peptide_omnipanel_v33_fusion_5fold_v1",
        "endpoint": endpoint, "task_type": "regression", "unit": spec["unit"], "bounds": bounds,
        "cond_cols": ep_data["cond_cols"], "canonical_default_conditions": spec["canonical_conditions"],
        "cpu_model_ref": str(EXP_CPU_DIR / endpoint / "models_5fold.joblib"),
        "pretrained_base_heads": ser_heads,
        "smiles_only": {"candidate_heads": track_a_heads, "stacker": stack_a["stacker"], "weights": stack_a["weights"], "calibrator": None, "conformal": conf_smi},
        "dual_modality": {"candidate_heads": track_b_heads, "stacker": stack_b["stacker"], "weights": stack_b["weights"], "calibrator": None, "conformal": conf_dual},
        "aft": {"stacker_dual": aft_stack_dual["stacker"], "weights_dual": aft_stack_dual["weights"], "stacker_smiles_only": aft_stack_smi["stacker"], "weights_smiles_only": aft_stack_smi["weights"]} if aft_stack_dual is not None else None,
        "ad_index": ad_index,
    }
    joblib.dump(model_bundle, ep_out_dir / "fusion_models_5fold.joblib", compress=3)

    c_mae = rep_ablation_doc["cpu_dual_modality"]["ho_metrics"]["mae"]
    f_mae = stack_b["ho_metrics"]["mae"]
    f_sp = stack_b["ho_metrics"]["spearman"]
    print(f"[FUS DONE ] {endpoint:<22} ({elapsed:5.1f}s) | CPU MAE={c_mae:.4f} -> FUS MAE={f_mae:.4f} (Δ{f_mae-c_mae:+.4f}) Sp={f_sp:.3f} | meta={stack_b['best_cfg']['method']}", flush=True)
    gc.collect()
    return {"endpoint": endpoint, "metrics": metrics_doc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="*", default=None)
    ap.add_argument("--max-workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    EXP_FUSION_DIR.mkdir(parents=True, exist_ok=True)
    global GLOBAL_BANK
    GLOBAL_BANK = V331FusionBank()
    print(f"V331FusionBank ready in {time.time()-t0:.2f}s (N={GLOBAL_BANK.n_total:,}, {len(GLOBAL_BANK.emb_blocks)} emb blocks)", flush=True)

    lpt_order = [
        ("cytotoxicity_CC50", 8),
        ("MIC_B_subtilis", 6),
        ("MIC_K_pneumoniae", 5),
        ("MIC_A_baumannii", 5),
        ("MIC_S_epidermidis", 5),
        ("MIC_Salmonella", 5),
        ("MIC_E_faecalis", 5),
        ("MBC_S_aureus", 4),
        ("MBC_E_coli", 4),
        ("LC_antimicrobial", 4),
        ("hemolysis_MHC", 4),
        ("MBC_P_aeruginosa", 4),
        ("MBIC_biofilm", 4),
        ("MIC_C_neoformans", 4),
        ("MIC_E_faecium", 4),
        ("MIC_L_monocytogenes", 4),
        ("efflux_ratio", 3),
        ("MBEC_biofilm", 3),
        ("MBC_A_baumannii", 3),
        ("MFC_C_albicans", 3),
        ("MBC_E_faecalis", 3),
        ("Vd_absolute", 3),
    ]
    tasks = [(ep, nj) for ep, nj in lpt_order if (not args.endpoints or ep in args.endpoints)]
    ctx = mp.get_context("fork")
    summary_by_ep: Dict[str, Any] = {}
    n_workers = min(args.max_workers, len(tasks))
    print(f"=== V33.1 4-FM Fusion Suite: {len(tasks)} endpoints across {n_workers} workers ===", flush=True)
    with ctx.Pool(processes=n_workers) as pool:
        for item in pool.imap_unordered(run_single_endpoint, tasks):
            summary_by_ep[item["endpoint"]] = item["metrics"]

    master = {
        "schema": "peptide_omnipanel_v331_fusion_benchmark_summary_v1",
        "total_endpoints": len(summary_by_ep),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "endpoints": {ep: summary_by_ep[ep] for ep in sorted(summary_by_ep.keys())},
    }
    out_fp = EXP_FUSION_DIR / "fusion_benchmark_summary.json"
    out_fp.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL {len(summary_by_ep)} V33.1 FUSION ENDPOINTS DONE in {master['total_elapsed_sec']}s -> {out_fp}", flush=True)


if __name__ == "__main__":
    main()
