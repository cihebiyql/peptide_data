#!/usr/bin/env python3
"""Peptide OmniPanel V33 — Wave-2D 64-Core Parallel Pretrained Multimodal Fusion,
Fair Individual Representation Ablation, Censored Multimodal AFT, Calibration,
Conformal Uncertainty & Strict Zero-Leakage Leave-Case17-Out Suite.

Strictly owns and writes ONLY under /data/qlyu/v33_upgrade/models_fusion/.
Read-only access to data/, features/, embeddings/, and models_cpu/.
"""
from __future__ import annotations

import os

# Single-threaded BLAS/OpenMP BEFORE importing numpy/scipy/sklearn/lgb/xgb for 100% fork safety!
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.linalg import eigh

from fusion_bank import FusionBank, TanimotoADIndex
from fusion_config import (
    CASE17_CLINICAL_ROUTE,
    DIRECT_FUSION_REPS,
    ENDPOINT_SPECS,
    INDIVIDUAL_REPS,
    OUT_DIR,
    compute_halflife_mechanism_anchor,
)
from fusion_metrics import (
    ConformalIntervalCalibrator,
    ProbabilityCalibrator,
    classification_metrics,
    compute_ad_stratification,
    regression_metrics,
)
from fusion_models import (
    BoundedWrapper,
    EmbeddingFoldPreprocessor,
    ExactRBFKLR,
    ExactRBFKRR,
    FastPlattRBFSVC,
    ModalityAwareStacker,
    PrecomputedRBFSVC,
    PrecomputedRBFSVR,
    XGBoostAFTRegressor,
    compute_sqdist_matrix,
    get_multimodal_aft_candidates,
    get_rep_candidates,
    make_fusion_estimator,
)

csv.field_size_limit(10_000_000)

GLOBAL_BANK: Optional[FusionBank] = None


def _score_metrics(m: Dict[str, Any], task_type: str, is_aft: bool = False) -> Tuple[float, float]:
    if task_type == "classification":
        roc = m["roc_auc"] if m["roc_auc"] is not None else -1.0
        pr = m["pr_auc"] if m["pr_auc"] is not None else -1.0
        brier = m["brier"] if m["brier"] is not None else 1.0
        # Balanced PR-AUC + ROC-AUC with slight Brier tie-breaker
        return (0.5 * pr + 0.5 * roc - 0.05 * brier, pr)
    if is_aft:
        hinge = m["all_row_hinge_mae"] if m["all_row_hinge_mae"] is not None else 1e9
        conc = m["censored_concordance"] if m["censored_concordance"] is not None else 0.0
        return (-hinge, conc)
    mae = m["mae"] if m["mae"] is not None else 1e9
    sp = m["spearman"] if m["spearman"] is not None else -1.0
    return (-mae, sp)


def _track_valid_mask(bank: FusionBank, compound_idx: np.ndarray, track: str) -> np.ndarray:
    if track == "smiles_only":
        return bank.smiles_valid[compound_idx].copy()
    return np.ones(len(compound_idx), dtype=bool)


def _predict_single(est: Any, X: np.ndarray, task_type: str, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
    if task_type == "classification":
        if D2_star is not None and isinstance(est, (ExactRBFKLR, PrecomputedRBFSVC, FastPlattRBFSVC)):
            return est.predict_proba(X, D2_star=D2_star)[:, 1].astype(np.float64)
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1].astype(np.float64)
        dec = est.decision_function(X).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(dec, -40.0, 40.0)))
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
    """Exact dual/primal eigendecomposition Ridge sweep across all alphas in 1 shot per fold."""
    oofs = {float(a): np.full(n_cv, np.nan, dtype=np.float64) for a in alphas}
    for entry in fold_entries:
        if entry is None:
            continue
        tr_pos = entry["tr_pos"]
        va_pos = entry["va_pos"]
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
            G = X_tr_c @ X_tr_c.T
            s, U = eigh(G)
            s = np.maximum(s, 0.0)
            Uty = U.T @ y_tr_c
            K_va = X_va_c @ X_tr_c.T
            for a in alphas:
                dual = U @ (Uty / (s + float(a)))
                p = K_va @ dual + y_mu
                if bounds is not None:
                    p = np.clip(p, bounds[0], bounds[1])
                oofs[float(a)][va_pos] = p
        else:
            C_mat = X_tr_c.T @ X_tr_c
            s, V = eigh(C_mat)
            s = np.maximum(s, 0.0)
            Vtxy = V.T @ (X_tr_c.T @ y_tr_c)
            XV = X_va_c @ V
            for a in alphas:
                w_eig = Vtxy / (s + float(a))
                p = XV @ w_eig + y_mu
                if bounds is not None:
                    p = np.clip(p, bounds[0], bounds[1])
                oofs[float(a)][va_pos] = p
    return oofs


def _build_rep_fold_cache(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    mode: str,
    include_censored_in_train: bool = False,
    compute_d2: bool = False,
) -> List[Optional[Dict[str, Any]]]:
    """Build fold-internal preprocessed matrices (and optional exact pairwise D2) ONCE per mode."""
    cv_rows = ep_data["cv_idx"]
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    task_type = spec["task_type"]
    valid_cv = _track_valid_mask(bank, cidx_cv, rep_cfg["track"])

    cache: List[Optional[Dict[str, Any]]] = []
    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv
        if task_type == "regression" and not include_censored_in_train:
            tr_mask = tr_mask & (ct_cv == "exact")
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            cache.append(None)
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]
        pre = EmbeddingFoldPreprocessor(
            blocks=rep_cfg["blocks"],
            phys_mode=rep_cfg["phys_mode"],
            cond_cols=ep_data["cond_cols"],
            mode=mode,
            pca_group_dims=rep_cfg["pca_group_dims"],
            random_state=42 + f,
        )
        X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
        X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])
        item: Dict[str, Any] = {
            "fold": f,
            "tr_pos": tr_pos,
            "va_pos": va_pos,
            "X_tr": X_tr,
            "X_va": X_va,
            "D2_tr": None,
            "D2_va": None,
            "base_gamma": None,
        }
        if compute_d2:
            D2_tr = compute_sqdist_matrix(X_tr, X_tr)
            D2_va = compute_sqdist_matrix(X_va, X_tr)
            tri = D2_tr[np.triu_indices(len(tr_pos), k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_tr.shape[1]))
            item["D2_tr"] = D2_tr
            item["D2_va"] = D2_va
            item["base_gamma"] = 1.0 / max(med_d2, 1e-6)
        cache.append(item)
    return cache


def refit_5fold_and_predict_ho(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    recipe: Dict[str, Any],
    n_jobs: int,
    is_aft: bool = False,
) -> Dict[str, Any]:
    """Fit winning recipe on exact frozen 5 folds, returning fitted folds, OOF, and Holdout predictions."""
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv = len(cv_rows)
    n_ho = len(ho_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    cidx_ho = ep_data["compound_idx"][ho_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    cond_ho = [ep_data["cond_rows"][i] for i in ho_rows]
    task_type = spec["task_type"]
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
        if task_type == "regression" and not is_aft:
            tr_mask = tr_mask & (ct_cv == "exact")
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]

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
            y_tr = y_cv[tr_pos].astype(np.int32) if task_type == "classification" else y_cv[tr_pos]
            if task_type == "classification" and len(np.unique(y_tr)) < 2:
                continue
            est = make_fusion_estimator(
                family=recipe["family"],
                params=recipe["params"],
                task_type=task_type,
                n_jobs=n_jobs,
                bounds=bounds if task_type == "regression" else None,
                winsor_cap=recipe.get("winsor_cap"),
                seed=42 + f,
            )
            est.fit(X_tr, y_tr)
            oof_pred[va_pos] = _predict_single(est, X_va, task_type)

        if len(ho_vpos) > 0:
            X_ho_v = pre.transform(bank, cidx_ho[ho_vpos], cond_rows=ho_cond_v)
            p_ho_v = est.predict(X_ho_v) if is_aft else _predict_single(est, X_ho_v, task_type)
            ho_fold_preds.append(p_ho_v)
        fitted_folds.append((f, pre, est))

    ho_pred = np.full(n_ho, np.nan, dtype=np.float64)
    ho_sd = np.full(n_ho, np.nan, dtype=np.float64)
    if len(ho_vpos) > 0 and len(ho_fold_preds) > 0:
        P_ho = np.stack(ho_fold_preds, axis=0)
        ho_pred[ho_vpos] = np.mean(P_ho, axis=0)
        ho_sd[ho_vpos] = np.std(P_ho, axis=0, ddof=1) if P_ho.shape[0] > 1 else np.zeros(len(ho_vpos))

    if task_type == "classification":
        m_cv = classification_metrics(y_cv, oof_pred)
        m_ho = classification_metrics(ep_data["y_value"][ho_rows], ho_pred)
    else:
        m_cv = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
        m_ho = regression_metrics(
            ep_data["y_value"][ho_rows],
            ho_pred,
            ep_data["y_lower"][ho_rows],
            ep_data["y_upper"][ho_rows],
            ep_data["censor_type"][ho_rows],
        )

    return {
        "rep_name": rep_cfg["name"],
        "rep_cfg": rep_cfg,
        "recipe": recipe,
        "is_aft": is_aft,
        "oof_pred": oof_pred,
        "ho_pred": ho_pred,
        "ho_sd": ho_sd,
        "cv_metrics": m_cv,
        "ho_metrics": m_ho,
        "folds": fitted_folds,
    }


def benchmark_representation(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    n_jobs: int,
    is_direct_fusion: bool = False,
    keep_both_families: bool = False,
) -> Dict[str, Any]:
    """Benchmark all curated anti-(p >> n) base learners for ONE representation strictly on 5-fold CV.

    Processes fold-by-fold for 5x lower peak RAM, shares pairwise D2 across all RBF kernel learners,
    shares dual/primal Ridge eigendecomposition across all alphas, shares Nystrom basis across all C/alpha,
    and shares FastPCA projections between pca_gbdt and pca_mlp!
    """
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression, Ridge

    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    winsor_cap = spec.get("outlier_winsor_cap")
    valid_cv = _track_valid_mask(bank, cidx_cv, rep_cfg["track"])

    cands = get_rep_candidates(
        size_tier=spec["size_tier"],
        task_type=task_type,
        n_cv=n_cv,
        is_direct_fusion=is_direct_fusion,
        winsor_cap=winsor_cap,
    )

    bal_cands = [i for i, c in enumerate(cands) if c["mode"] == "balanced"]
    pca_cands = [i for i, c in enumerate(cands) if c["mode"] in ("pca_gbdt", "pca_mlp")]
    oofs = [np.full(n_cv, np.nan, dtype=np.float64) for _ in cands]
    errors: Dict[int, str] = {}

    needs_d2 = any(cands[i]["family"] in ("KRR_RBF", "SVR_RBF", "KLR_RBF", "SVC_RBF", "FastSVC_RBF") for i in bal_cands)

    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv
        if task_type == "regression":
            tr_mask = tr_mask & (ct_cv == "exact")
        if tr_mask.sum() < 4 or va_mask.sum() == 0:
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]
        y_tr = y_cv[tr_pos].astype(np.int32) if task_type == "classification" else y_cv[tr_pos]
        if task_type == "classification" and len(np.unique(y_tr)) < 2:
            continue

        # --- Block 1: mode='balanced' (Ridge, Logistic, KRR/SVR/KLR/SVC, Nystrom_RBF) ---
        if bal_cands:
            pre_b = EmbeddingFoldPreprocessor(
                blocks=rep_cfg["blocks"],
                phys_mode=rep_cfg["phys_mode"],
                cond_cols=ep_data["cond_cols"],
                mode="balanced",
                pca_group_dims=rep_cfg["pca_group_dims"],
                random_state=42 + f,
            )
            X_tr = pre_b.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            X_va = pre_b.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])

            D2_tr = D2_va = base_gamma = None
            if needs_d2:
                use_f32 = len(tr_pos) > 7500
                D2_tr = compute_sqdist_matrix(X_tr, X_tr, use_float32=use_f32)
                D2_va = compute_sqdist_matrix(X_va, X_tr, use_float32=use_f32)
                if len(tr_pos) > 1500:
                    rng_m = np.random.default_rng(42 + f)
                    sub_m = rng_m.choice(len(tr_pos), size=1000, replace=False)
                    dsub = D2_tr[np.ix_(sub_m, sub_m)]
                    tri = dsub[np.triu_indices(len(sub_m), k=1)]
                else:
                    tri = D2_tr[np.triu_indices(len(tr_pos), k=1)]
                med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_tr.shape[1]))
                base_gamma = 1.0 / max(med_d2, 1e-6)

            # Fast 1-shot dual/primal Ridge sweep
            ridge_ids = [i for i in bal_cands if cands[i]["family"] == "Ridge" and task_type == "regression"]
            if ridge_ids:
                alphas = [float(cands[i]["params"]["alpha"]) for i in ridge_ids]
                entry = {"tr_pos": tr_pos, "va_pos": va_pos, "X_tr": X_tr, "X_va": X_va}
                r_oofs = _fast_ridge_oof_sweep([entry], y_cv, n_cv, alphas, bounds=bounds, winsor_cap=winsor_cap)
                for i in ridge_ids:
                    oofs[i][va_pos] = r_oofs[float(cands[i]["params"]["alpha"])][va_pos]

            # Fast shared Nystrom basis per fold (fit Nystroem ONCE, sweep all alpha/C!)
            nys_ids = [i for i in bal_cands if cands[i]["family"] == "Nystrom_RBF"]
            if nys_ids:
                try:
                    n_comp = min(int(cands[nys_ids[0]]["params"].get("n_components", 800)), len(tr_pos))
                    gm = float(cands[nys_ids[0]]["params"].get("gamma_mult", 1.0))
                    rng = np.random.default_rng(42 + f)
                    sub = rng.choice(len(tr_pos), size=min(len(tr_pos), 600), replace=False)
                    D2s = compute_sqdist_matrix(X_tr[sub], X_tr[sub])
                    tris = D2s[np.triu_indices(len(sub), k=1)]
                    md2 = float(np.median(tris[tris > 0])) if (tris > 0).any() else float(max(1, X_tr.shape[1]))
                    nys = Nystroem(kernel="rbf", gamma=gm / max(md2, 1e-6), n_components=n_comp, random_state=42 + f)
                    Z_tr = nys.fit_transform(X_tr).astype(np.float32)
                    Z_va = nys.transform(X_va).astype(np.float32)
                    y_fit = np.clip(y_tr, None, winsor_cap) if (task_type == "regression" and winsor_cap is not None) else y_tr
                    for i in nys_ids:
                        if task_type == "regression":
                            hd = Ridge(alpha=float(cands[i]["params"]["alpha"])).fit(Z_tr, y_fit)
                            p = hd.predict(Z_va).astype(np.float64)
                            if bounds is not None:
                                p = np.clip(p, bounds[0], bounds[1])
                            oofs[i][va_pos] = p
                        else:
                            hd = LogisticRegression(C=float(cands[i]["params"]["C"]), max_iter=250, tol=1e-3, solver="lbfgs").fit(Z_tr, y_tr)
                            oofs[i][va_pos] = hd.predict_proba(Z_va)[:, 1].astype(np.float64)
                    del Z_tr, Z_va
                except Exception as exc:
                    for i in nys_ids:
                        errors[i] = str(exc)

            # Remaining balanced candidates (Logistic, KRR_RBF, SVR_RBF, KLR_RBF, SVC_RBF)
            for i in bal_cands:
                if i in ridge_ids or i in nys_ids or i in errors:
                    continue
                c = cands[i]
                try:
                    est = make_fusion_estimator(
                        family=c["family"],
                        params=c["params"],
                        task_type=task_type,
                        n_jobs=n_jobs,
                        bounds=bounds if task_type == "regression" else None,
                        winsor_cap=c.get("winsor_cap"),
                        seed=42 + f,
                    )
                    inner = est.est if isinstance(est, BoundedWrapper) else est
                    if D2_tr is not None and isinstance(inner, (ExactRBFKRR, PrecomputedRBFSVR, ExactRBFKLR, PrecomputedRBFSVC, FastPlattRBFSVC)):
                        y_fit = (
                            np.clip(y_tr, None, est.winsor_cap)
                            if (isinstance(est, BoundedWrapper) and est.winsor_cap is not None)
                            else y_tr
                        )
                        inner.fit(X_tr, y_fit, D2_pre=D2_tr, base_gamma=base_gamma)
                        oofs[i][va_pos] = _predict_single(est, X_va, task_type, D2_star=D2_va)
                    else:
                        est.fit(X_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, X_va, task_type)
                except Exception as exc:
                    errors[i] = str(exc)

            del X_tr, X_va, D2_tr, D2_va, pre_b

        # --- Block 2: mode='pca_gbdt' & 'pca_mlp' (FastPCA fit ONCE per fold, shared!) ---
        if pca_cands:
            pre_p = EmbeddingFoldPreprocessor(
                blocks=rep_cfg["blocks"],
                phys_mode=rep_cfg["phys_mode"],
                cond_cols=ep_data["cond_cols"],
                mode="pca_gbdt",
                pca_group_dims=rep_cfg["pca_group_dims"],
                random_state=42 + f,
            )
            Xp_tr = pre_p.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            Xp_va = pre_p.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])

            # Standardize non-condition columns once if any MLP candidate exists (matches mode='pca_mlp')
            has_mlp = any(cands[i]["mode"] == "pca_mlp" for i in pca_cands)
            Xm_tr = Xm_va = None
            if has_mlp:
                n_c = len(pre_p.cond_enc_.feature_names_) if pre_p.cond_enc_ is not None else 0
                cut = Xp_tr.shape[1] - n_c
                Xm_tr = Xp_tr.copy()
                Xm_va = Xp_va.copy()
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
                    est = make_fusion_estimator(
                        family=c["family"],
                        params=c["params"],
                        task_type=task_type,
                        n_jobs=n_jobs,
                        bounds=bounds if task_type == "regression" else None,
                        winsor_cap=c.get("winsor_cap"),
                        seed=42 + f,
                    )
                    if c["mode"] == "pca_mlp":
                        est.fit(Xm_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xm_va, task_type)
                    else:
                        est.fit(Xp_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xp_va, task_type)
                except Exception as exc:
                    errors[i] = str(exc)

            del Xp_tr, Xp_va, Xm_tr, Xm_va, pre_p

    all_results: List[Dict[str, Any]] = []
    for i, c in enumerate(cands):
        if i in errors:
            all_results.append({"recipe": c, "metrics": {}, "score": (-1e9, -1e9), "error": errors[i]})
            continue
        m = (
            classification_metrics(y_cv, oofs[i])
            if task_type == "classification"
            else regression_metrics(y_cv, oofs[i], yl_cv, yu_cv, ct_cv)
        )
        all_results.append({"recipe": c, "metrics": m, "score": _score_metrics(m, task_type), "oof_pred": oofs[i]})

    valid_res = [r for r in all_results if "error" not in r]
    assert valid_res, f"{ep_data['endpoint']} {rep_cfg['name']}: all base learners failed!"
    valid_res.sort(key=lambda x: x["score"], reverse=True)

    best_overall_rec = valid_res[0]["recipe"]
    best_overall = refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, best_overall_rec, n_jobs=n_jobs, is_aft=False)

    # Family breakdown for transparent ablation reporting (best linear, kernel/mlp, tree)
    fam_summary: Dict[str, Any] = {}
    for fam_group, fams in [
        ("linear", ("Ridge", "Logistic")),
        ("kernel_mlp", ("KRR_RBF", "SVR_RBF", "KLR_RBF", "SVC_RBF", "FastSVC_RBF", "Nystrom_RBF", "MLP", "FullEmb_MLP")),
        ("pca_gbdt", ("LightGBM", "XGBoost")),
    ]:
        sub = [r for r in valid_res if r["recipe"]["family"] in fams]
        if sub:
            fam_summary[fam_group] = {
                "family": sub[0]["recipe"]["family"],
                "params": sub[0]["recipe"]["params"],
                "cv_metrics": sub[0]["metrics"],
            }

    out: Dict[str, Any] = {
        "best_overall": best_overall,
        "family_cv_breakdown": fam_summary,
        "best_smooth": None,
        "best_tree": None,
    }

    if keep_both_families:
        smooth_sub = [r for r in valid_res if r["recipe"]["family"] not in ("LightGBM", "XGBoost")]
        tree_sub = [r for r in valid_res if r["recipe"]["family"] in ("LightGBM", "XGBoost")]
        if smooth_sub:
            if smooth_sub[0]["recipe"] == best_overall_rec:
                out["best_smooth"] = best_overall
            else:
                out["best_smooth"] = refit_5fold_and_predict_ho(
                    bank, ep_data, spec, rep_cfg, smooth_sub[0]["recipe"], n_jobs=n_jobs, is_aft=False
                )
        if tree_sub:
            if tree_sub[0]["recipe"] == best_overall_rec:
                out["best_tree"] = best_overall
            else:
                out["best_tree"] = refit_5fold_and_predict_ho(
                    bank, ep_data, spec, rep_cfg, tree_sub[0]["recipe"], n_jobs=n_jobs, is_aft=False
                )

    return out


def benchmark_multimodal_aft(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    n_jobs: int,
) -> Dict[str, Any]:
    """Tune & fit censored XGBoost survival:aft on compact multimodal PCA + physchem features."""
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    bounds = spec.get("bounds")

    aft_params_list = get_multimodal_aft_candidates(spec["size_tier"])
    f_cache = _build_rep_fold_cache(
        bank, ep_data, spec, rep_cfg, mode="pca_gbdt", include_censored_in_train=True, compute_d2=False
    )

    best_score = (-1e12, -1e12)
    best_params = aft_params_list[0]
    for p in aft_params_list:
        oof_p = np.full(n_cv, np.nan, dtype=np.float64)
        for entry in f_cache:
            if entry is None:
                continue
            tr_pos = entry["tr_pos"]
            va_pos = entry["va_pos"]
            est = XGBoostAFTRegressor(**p, nthread=n_jobs, seed=42 + entry["fold"], bounds=bounds)
            est.fit(entry["X_tr"], y_cv[tr_pos], yl_cv[tr_pos], yu_cv[tr_pos], ct_cv[tr_pos])
            oof_p[va_pos] = est.predict(entry["X_va"])
        m = regression_metrics(y_cv, oof_p, yl_cv, yu_cv, ct_cv)
        sc = _score_metrics(m, "regression", is_aft=True)
        if sc > best_score:
            best_score = sc
            best_params = p

    del f_cache
    best_rec = {"mode": "pca_gbdt", "family": "XGBoost_AFT", "params": best_params}
    return refit_5fold_and_predict_ho(bank, ep_data, spec, rep_cfg, best_rec, n_jobs=n_jobs, is_aft=True)


def run_nested_stacking(
    head_names: List[str],
    oof_dict: Dict[str, np.ndarray],
    ho_dict: Dict[str, np.ndarray],
    y_cv: np.ndarray,
    yl_cv: np.ndarray,
    yu_cv: np.ndarray,
    ct_cv: np.ndarray,
    fids_cv: np.ndarray,
    y_ho: np.ndarray,
    yl_ho: np.ndarray,
    yu_ho: np.ndarray,
    ct_ho: np.ndarray,
    task_type: str,
    bounds: Optional[Tuple[float, float]],
    eval_mask_cv: np.ndarray,
    eval_mask_ho: np.ndarray,
    is_aft_stack: bool = False,
) -> Dict[str, Any]:
    """Strictly fold-internal nested 5-fold CV selection of the optimal multimodal meta-stacker."""
    n_cv = len(y_cv)
    n_ho = len(y_ho)
    P_oof = np.column_stack([oof_dict[h] for h in head_names])
    P_ho = np.column_stack([ho_dict[h] for h in head_names])

    has_any_nan = bool((~np.isfinite(P_oof[eval_mask_cv])).any() or (~np.isfinite(P_ho[eval_mask_ho])).any())
    n_eff = int(eval_mask_cv.sum())

    if is_aft_stack:
        meta_candidates = [
            {"method": "best_single", "params": {}},
            {"method": "convex_hinge", "params": {"reg": 0.0}},
            {"method": "convex_hinge", "params": {"reg": 0.02}},
            {"method": "convex_hinge", "params": {"reg": 0.08}},
            {"method": "topk_blend", "params": {"k": 2}},
        ]
    elif task_type == "regression":
        meta_candidates = [
            {"method": "best_single", "params": {}},
            {"method": "topk_blend", "params": {"k": 2}},
            {"method": "topk_blend", "params": {"k": 3}},
            {"method": "topk_blend", "params": {"k": 4}},
            {"method": "convex_simplex", "params": {"reg": 0.0}},
            {"method": "convex_simplex", "params": {"reg": 0.01}},
            {"method": "convex_simplex", "params": {"reg": 0.05}},
            {"method": "convex_simplex", "params": {"reg": 0.15}},
        ]
        if n_eff >= 120 and not has_any_nan:
            for alpha in (5.0, 25.0, 100.0):
                meta_candidates.append({"method": "ridge_nonneg", "params": {"alpha": alpha}})
            if n_eff >= 250:
                for alpha in (15.0, 60.0):
                    meta_candidates.append({"method": "ridge_l2", "params": {"alpha": alpha}})
    else:
        meta_candidates = [
            {"method": "best_single", "params": {}},
            {"method": "topk_blend", "params": {"k": 2}},
            {"method": "topk_blend", "params": {"k": 3}},
            {"method": "topk_blend", "params": {"k": 4}},
            {"method": "convex_simplex", "params": {"reg": 0.0}},
            {"method": "convex_simplex", "params": {"reg": 0.01}},
            {"method": "convex_simplex", "params": {"reg": 0.05}},
            {"method": "logit_convex", "params": {"reg": 0.0}},
            {"method": "logit_convex", "params": {"reg": 0.02}},
            {"method": "logit_convex", "params": {"reg": 0.08}},
        ]
        if n_eff >= 150 and not has_any_nan:
            for C in (0.02, 0.1, 0.5):
                meta_candidates.append({"method": "logistic_l2", "params": {"C": C}})

    cand_logs: List[Dict[str, Any]] = []
    best_cfg = None
    best_score = (-1e12, -1e12)
    best_nested_oof = None

    for mc in meta_candidates:
        nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
        for f in range(5):
            va = (fids_cv == f) & eval_mask_cv
            tr = (fids_cv != f) & eval_mask_cv
            if task_type == "regression" and not is_aft_stack:
                tr = tr & (ct_cv == "exact")
            if tr.sum() < 4 or va.sum() == 0:
                continue
            st = ModalityAwareStacker(task_type=task_type, method=mc["method"], params=mc["params"], bounds=bounds)
            st.fit(
                P_oof[tr],
                y_cv[tr],
                head_names=head_names,
                y_lower=yl_cv[tr] if is_aft_stack else None,
                y_upper=yu_cv[tr] if is_aft_stack else None,
                censor_type=ct_cv[tr] if is_aft_stack else None,
            )
            nested_oof[va] = st.predict(P_oof[va])

        if task_type == "classification":
            m = classification_metrics(y_cv[eval_mask_cv], nested_oof[eval_mask_cv])
        else:
            m = regression_metrics(
                y_cv[eval_mask_cv], nested_oof[eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv]
            )
        sc = _score_metrics(m, task_type, is_aft=is_aft_stack)
        cand_logs.append({"method": mc["method"], "params": mc["params"], "nested_cv_metrics": m})
        if sc > best_score:
            best_score = sc
            best_cfg = mc
            best_nested_oof = nested_oof

    assert best_cfg is not None and best_nested_oof is not None

    # Fit final stacker strictly on full CV OOF (ZERO holdout peeking!)
    fit_mask = eval_mask_cv.copy()
    if task_type == "regression" and not is_aft_stack:
        fit_mask = fit_mask & (ct_cv == "exact")
    final_stacker = ModalityAwareStacker(
        task_type=task_type, method=best_cfg["method"], params=best_cfg["params"], bounds=bounds
    )
    final_stacker.fit(
        P_oof[fit_mask],
        y_cv[fit_mask],
        head_names=head_names,
        y_lower=yl_cv[fit_mask] if is_aft_stack else None,
        y_upper=yu_cv[fit_mask] if is_aft_stack else None,
        censor_type=ct_cv[fit_mask] if is_aft_stack else None,
    )

    ho_pred = np.full(n_ho, np.nan, dtype=np.float64)
    if eval_mask_ho.any():
        ho_pred[eval_mask_ho] = final_stacker.predict(P_ho[eval_mask_ho])

    if task_type == "classification":
        cv_m = classification_metrics(y_cv[eval_mask_cv], best_nested_oof[eval_mask_cv])
        ho_m = classification_metrics(y_ho[eval_mask_ho], ho_pred[eval_mask_ho])
    else:
        cv_m = regression_metrics(
            y_cv[eval_mask_cv], best_nested_oof[eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv]
        )
        ho_m = regression_metrics(
            y_ho[eval_mask_ho], ho_pred[eval_mask_ho], yl_ho[eval_mask_ho], yu_ho[eval_mask_ho], ct_ho[eval_mask_ho]
        )

    return {
        "best_cfg": best_cfg,
        "stacker": final_stacker,
        "weights": final_stacker.weight_dict(),
        "active_heads": final_stacker.active_heads(),
        "nested_oof": best_nested_oof,
        "ho_pred": ho_pred,
        "cv_metrics": cv_m,
        "ho_metrics": ho_m,
        "candidate_ablation": cand_logs,
    }


def assign_balanced_cluster_folds(cluster_ids: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Deterministic LPT cluster bin-packing for leave-Case17-out 5-fold refits."""
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


def refit_head_leave_case17_out(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    head_info: Dict[str, Any],
    c17_idx: np.ndarray,
    n_jobs: int,
) -> Dict[str, Any]:
    """Strictly Leave-Case17-Out (case17_excluded == 0) 5-fold refit & Case17 prediction for ONE pretrained head."""
    keep_rows = np.where(ep_data["case17_excluded"] == 0)[0]
    cidx = ep_data["compound_idx"][keep_rows]
    # Hard anti-leakage assertion: ZERO Case17 drug indices anywhere in training!
    c17_set = set(int(x) for x in c17_idx)
    assert all(int(x) not in c17_set for x in cidx), f"{ep_data['endpoint']}: Case17 index leaked into non-Case17 train!"

    clids = ep_data["cluster_ids"][keep_rows]
    y = ep_data["y_value"][keep_rows]
    yl = ep_data["y_lower"][keep_rows]
    yu = ep_data["y_upper"][keep_rows]
    ct = ep_data["censor_type"][keep_rows]
    conds = [ep_data["cond_rows"][i] for i in keep_rows]

    rep_cfg = head_info["rep_cfg"]
    recipe = head_info["recipe"]
    is_aft = bool(head_info["is_aft"])
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    valid = _track_valid_mask(bank, cidx, rep_cfg["track"])
    fids = assign_balanced_cluster_folds(clids, n_folds=5)

    canon_cond = spec["canonical_conditions"]
    alt_conds = spec.get("alt_case17_conditions", {})

    canon_preds: List[np.ndarray] = []
    alt_preds: Dict[str, List[np.ndarray]] = {k: [] for k in alt_conds}

    for f in range(5):
        tr_mask = (fids != f) & valid
        if task_type == "regression" and not is_aft:
            tr_mask = tr_mask & (ct == "exact")
        if tr_mask.sum() < 4:
            continue
        tr_pos = np.where(tr_mask)[0]
        pre = EmbeddingFoldPreprocessor(
            blocks=rep_cfg["blocks"],
            phys_mode=rep_cfg["phys_mode"],
            cond_cols=ep_data["cond_cols"],
            mode=recipe["mode"],
            pca_group_dims=rep_cfg["pca_group_dims"],
            random_state=142 + f,
        )
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos])

        if is_aft:
            est = XGBoostAFTRegressor(**recipe["params"], nthread=n_jobs, seed=142 + f, bounds=bounds)
            est.fit(X_tr, y[tr_pos], yl[tr_pos], yu[tr_pos], ct[tr_pos])
            X_q = pre.transform(bank, c17_idx, default_cond=canon_cond)
            canon_preds.append(est.predict(X_q))
            for alt_k, alt_c in alt_conds.items():
                X_qa = pre.transform(bank, c17_idx, default_cond=alt_c)
                alt_preds[alt_k].append(est.predict(X_qa))
        else:
            y_tr = y[tr_pos].astype(np.int32) if task_type == "classification" else y[tr_pos]
            if task_type == "classification" and len(np.unique(y_tr)) < 2:
                continue
            est = make_fusion_estimator(
                family=recipe["family"],
                params=recipe["params"],
                task_type=task_type,
                n_jobs=n_jobs,
                bounds=bounds if task_type == "regression" else None,
                winsor_cap=recipe.get("winsor_cap"),
                seed=142 + f,
            )
            est.fit(X_tr, y_tr)
            X_q = pre.transform(bank, c17_idx, default_cond=canon_cond)
            canon_preds.append(_predict_single(est, X_q, task_type))
            for alt_k, alt_c in alt_conds.items():
                X_qa = pre.transform(bank, c17_idx, default_cond=alt_c)
                alt_preds[alt_k].append(_predict_single(est, X_qa, task_type))

    P_c = np.stack(canon_preds, axis=0)
    out = {
        "mean": np.mean(P_c, axis=0),
        "sd": np.std(P_c, axis=0, ddof=1) if P_c.shape[0] > 1 else np.zeros(len(c17_idx)),
        "alt": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in alt_preds.items()},
    }
    return out


def compute_oof_tanimoto(bank: FusionBank, ep_data: Dict[str, Any]) -> np.ndarray:
    """Use Wave-2C's exact fold-internal max Tanimoto if available, else compute fresh."""
    t_cpu = ep_data["cpu"].get("oof_max_tanimoto")
    if t_cpu is not None and len(t_cpu) == len(ep_data["cv_idx"]):
        return t_cpu.astype(np.float32)
    cv_rows = ep_data["cv_idx"]
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    out = np.full(len(cv_rows), np.nan, dtype=np.float32)
    for f in range(5):
        va = np.where(fids_cv == f)[0]
        tr = np.where(fids_cv != f)[0]
        if len(va) == 0 or len(tr) == 0:
            continue
        ad_f = TanimotoADIndex.from_indices(bank, cidx_cv[tr])
        max_t, _, _ = ad_f.query(bank, cidx_cv[va])
        out[va] = max_t
    return out


def run_single_endpoint(args: Tuple[str, int]) -> Dict[str, Any]:
    """Full Wave-2D multimodal ablation, stacking, censored AFT, calibration, conformal & Case17 pipeline for ONE endpoint."""
    endpoint, n_jobs = args
    t0 = time.time()
    bank = GLOBAL_BANK
    assert bank is not None
    spec = ENDPOINT_SPECS[endpoint]
    ep_data = bank.load_endpoint_and_cpu(endpoint)
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv = len(cv_rows)
    n_ho = len(ho_rows)

    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]

    y_ho = ep_data["y_value"][ho_rows]
    yl_ho = ep_data["y_lower"][ho_rows]
    yu_ho = ep_data["y_upper"][ho_rows]
    ct_ho = ep_data["censor_type"][ho_rows]

    cidx_cv = ep_data["compound_idx"][cv_rows]
    cidx_ho = ep_data["compound_idx"][ho_rows]
    smi_cv_mask = bank.smiles_valid[cidx_cv]
    smi_ho_mask = bank.smiles_valid[cidx_ho]
    dual_cv_mask = np.ones(n_cv, dtype=bool)
    dual_ho_mask = np.ones(n_ho, dtype=bool)

    print(f"[START] {endpoint:<24} (n={ep_data['n']}, cv={n_cv}, ho={n_ho}, jobs={n_jobs})", flush=True)

    # =========================================================================
    # 1. Fair Individual Representation Benchmark (All 7 Pretrained Reps + 2 Direct Fusion Reps)
    # =========================================================================
    rep_ablation_doc: Dict[str, Any] = {}
    trained_heads: Dict[str, Dict[str, Any]] = {}

    # Seed CPU baselines into representation scorecard
    cpu = ep_data["cpu"]
    if task_type == "classification":
        rep_ablation_doc["cpu_smiles_only"] = {
            "label": "Wave-2C CPU (smiles_only)",
            "track": "smiles_only",
            "winning_family": cpu["metrics"]["winning_recipe_smiles_only"]["family"],
            "cv_metrics": classification_metrics(y_cv[smi_cv_mask], cpu["oof_smiles_only"][smi_cv_mask]),
            "ho_metrics": classification_metrics(y_ho[smi_ho_mask], cpu["ho_smiles_only"][smi_ho_mask]),
        }
        rep_ablation_doc["cpu_dual_modality"] = {
            "label": "Wave-2C CPU (dual_modality)",
            "track": "dual_modality",
            "winning_family": cpu["metrics"]["winning_recipe_dual"]["family"],
            "cv_metrics": classification_metrics(y_cv, cpu["oof_dual"]),
            "ho_metrics": classification_metrics(y_ho, cpu["ho_dual"]),
        }
    else:
        rep_ablation_doc["cpu_smiles_only"] = {
            "label": "Wave-2C CPU (smiles_only)",
            "track": "smiles_only",
            "winning_family": cpu["metrics"]["winning_recipe_smiles_only"]["family"],
            "cv_metrics": regression_metrics(
                y_cv[smi_cv_mask], cpu["oof_smiles_only"][smi_cv_mask], yl_cv[smi_cv_mask], yu_cv[smi_cv_mask], ct_cv[smi_cv_mask]
            ),
            "ho_metrics": regression_metrics(
                y_ho[smi_ho_mask], cpu["ho_smiles_only"][smi_ho_mask], yl_ho[smi_ho_mask], yu_ho[smi_ho_mask], ct_ho[smi_ho_mask]
            ),
        }
        rep_ablation_doc["cpu_dual_modality"] = {
            "label": "Wave-2C CPU (dual_modality)",
            "track": "dual_modality",
            "winning_family": cpu["metrics"]["winning_recipe_dual"]["family"],
            "cv_metrics": regression_metrics(y_cv, cpu["oof_dual"], yl_cv, yu_cv, ct_cv),
            "ho_metrics": regression_metrics(y_ho, cpu["ho_dual"], yl_ho, yu_ho, ct_ho),
        }

    # Evaluate all 7 individual pretrained representations
    for rep_cfg in INDIVIDUAL_REPS:
        rname = rep_cfg["name"]
        keep_both = (rname == "esmc_multiregion") and (n_cv >= 200)
        res = benchmark_representation(
            bank, ep_data, spec, rep_cfg, n_jobs=n_jobs, is_direct_fusion=False, keep_both_families=keep_both
        )
        bo = res["best_overall"]
        rep_ablation_doc[rname] = {
            "label": rep_cfg["label"],
            "track": rep_cfg["track"],
            "winning_family": bo["recipe"]["family"],
            "winning_mode": bo["recipe"]["mode"],
            "winning_params": bo["recipe"]["params"],
            "cv_metrics": bo["cv_metrics"],
            "ho_metrics": bo["ho_metrics"],
            "family_cv_breakdown": res["family_cv_breakdown"],
        }
        trained_heads[rname] = bo
        if keep_both:
            if res["best_smooth"] is not None:
                trained_heads[f"{rname}_smooth"] = res["best_smooth"]
            if res["best_tree"] is not None:
                trained_heads[f"{rname}_tree"] = res["best_tree"]

    # Evaluate 2 Direct Multimodal Fusion representations (Track A: direct_smiles, Track B: direct_dual)
    for dname in ("direct_smiles", "direct_dual"):
        rep_cfg = DIRECT_FUSION_REPS[dname]
        res = benchmark_representation(
            bank, ep_data, spec, rep_cfg, n_jobs=n_jobs, is_direct_fusion=True, keep_both_families=True
        )
        bo = res["best_overall"]
        rep_ablation_doc[dname] = {
            "label": rep_cfg["label"],
            "track": rep_cfg["track"],
            "winning_family": bo["recipe"]["family"],
            "winning_mode": bo["recipe"]["mode"],
            "winning_params": bo["recipe"]["params"],
            "cv_metrics": bo["cv_metrics"],
            "ho_metrics": bo["ho_metrics"],
            "family_cv_breakdown": res["family_cv_breakdown"],
        }
        trained_heads[dname] = bo
        if res["best_smooth"] is not None:
            trained_heads[f"{dname}_smooth"] = res["best_smooth"]
            rep_ablation_doc[f"{dname}_smooth"] = {
                "label": rep_cfg["label"] + " [Kernel/Linear/MLP]",
                "track": rep_cfg["track"],
                "winning_family": res["best_smooth"]["recipe"]["family"],
                "cv_metrics": res["best_smooth"]["cv_metrics"],
                "ho_metrics": res["best_smooth"]["ho_metrics"],
            }
        if res["best_tree"] is not None:
            trained_heads[f"{dname}_tree"] = res["best_tree"]
            rep_ablation_doc[f"{dname}_tree"] = {
                "label": rep_cfg["label"] + " [PCA-GBDT]",
                "track": rep_cfg["track"],
                "winning_family": res["best_tree"]["recipe"]["family"],
                "cv_metrics": res["best_tree"]["cv_metrics"],
                "ho_metrics": res["best_tree"]["ho_metrics"],
            }

    # =========================================================================
    # 2. Censored-Aware Multimodal XGBoost survival:aft (ALL endpoints with censored CV rows!)
    # =========================================================================
    n_cens_cv = int((ct_cv != "exact").sum())
    n_cens_ho = int((ct_ho != "exact").sum())
    n_cens_total = int((ep_data["censor_type"] != "exact").sum())

    aft_direct_dual = None
    aft_direct_smi = None
    aft_stack_dual = None
    aft_stack_smi = None
    aft_ablation_doc: Optional[Dict[str, Any]] = None

    if task_type == "regression" and n_cens_cv >= 1:
        aft_direct_smi = benchmark_multimodal_aft(bank, ep_data, spec, DIRECT_FUSION_REPS["direct_smiles"], n_jobs=n_jobs)
        aft_direct_dual = benchmark_multimodal_aft(bank, ep_data, spec, DIRECT_FUSION_REPS["direct_dual"], n_jobs=n_jobs)
        trained_heads["aft_direct_smiles"] = aft_direct_smi
        trained_heads["aft_direct_dual"] = aft_direct_dual

        # Strictly nested CV AFT stacking: CPU AFT + Multimodal Direct AFT
        aft_smi_names = ["cpu_aft_smiles", "aft_direct_smiles"]
        aft_smi_oof = {"cpu_aft_smiles": cpu["oof_aft_smiles_only"], "aft_direct_smiles": aft_direct_smi["oof_pred"]}
        aft_smi_ho = {"cpu_aft_smiles": cpu["ho_aft_smiles_only"], "aft_direct_smiles": aft_direct_smi["ho_pred"]}
        aft_stack_smi = run_nested_stacking(
            aft_smi_names, aft_smi_oof, aft_smi_ho,
            y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho,
            task_type="regression", bounds=bounds, eval_mask_cv=smi_cv_mask, eval_mask_ho=smi_ho_mask, is_aft_stack=True,
        )

        aft_dual_names = ["cpu_aft_dual", "aft_direct_dual"]
        aft_dual_oof = {"cpu_aft_dual": cpu["oof_aft"], "aft_direct_dual": aft_direct_dual["oof_pred"]}
        aft_dual_ho = {"cpu_aft_dual": cpu["ho_aft"], "aft_direct_dual": aft_direct_dual["ho_pred"]}
        aft_stack_dual = run_nested_stacking(
            aft_dual_names, aft_dual_oof, aft_dual_ho,
            y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho,
            task_type="regression", bounds=bounds, eval_mask_cv=dual_cv_mask, eval_mask_ho=dual_ho_mask, is_aft_stack=True,
        )

    # =========================================================================
    # 3. Anti-(p >> n) Nested Multimodal Stacking (Track A: smiles_only, Track B: dual_modality)
    # =========================================================================
    # Build unified base head OOF & HO lookup dicts
    all_oof: Dict[str, np.ndarray] = {
        "cpu_smiles": cpu["oof_smiles_only"],
        "cpu_dual": cpu["oof_dual"],
    }
    all_ho: Dict[str, np.ndarray] = {
        "cpu_smiles": cpu["ho_smiles_only"],
        "cpu_dual": cpu["ho_dual"],
    }
    if cpu["oof_aft"] is not None:
        all_oof["cpu_aft_dual"] = cpu["oof_aft"]
        all_ho["cpu_aft_dual"] = cpu["ho_aft"]
    if cpu["oof_aft_smiles_only"] is not None:
        all_oof["cpu_aft_smiles"] = cpu["oof_aft_smiles_only"]
        all_ho["cpu_aft_smiles"] = cpu["ho_aft_smiles_only"]
    if cpu["oof_seq_only"] is not None:
        all_oof["cpu_seq_only"] = cpu["oof_seq_only"]
        all_ho["cpu_seq_only"] = cpu["ho_seq_only"]

    for hname, hobj in trained_heads.items():
        all_oof[hname] = hobj["oof_pred"]
        all_ho[hname] = hobj["ho_pred"]

    # Track A (smiles_only — STRICTLY ZERO sequence/ESM-C features!)
    track_a_heads: List[str] = [
        "cpu_smiles",
        "ncaa_smiles_repr",
        "ncaa_fusion_hf",
        "ncaa_protein_proj",
        "peptideclm2",
        "chemberta_mtr",
    ]
    if "direct_smiles_smooth" in all_oof:
        track_a_heads.append("direct_smiles_smooth")
    if "direct_smiles_tree" in all_oof:
        track_a_heads.append("direct_smiles_tree")
    if aft_direct_smi is not None:
        track_a_heads.extend(["cpu_aft_smiles", "aft_direct_smiles"])

    stack_a = run_nested_stacking(
        track_a_heads, all_oof, all_ho,
        y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho,
        task_type=task_type, bounds=bounds, eval_mask_cv=smi_cv_mask, eval_mask_ho=smi_ho_mask, is_aft_stack=False,
    )

    # Track B (dual_modality — Full SMILES + Sequence + ESM-C 600M Multi-Region + CPU Dual!)
    track_b_heads: List[str] = [
        "cpu_dual",
        "ncaa_smiles_repr",
        "ncaa_fusion_hf",
        "ncaa_protein_proj",
        "peptideclm2",
        "chemberta_mtr",
        "esmc_mean",
    ]
    if "esmc_multiregion_smooth" in all_oof:
        track_b_heads.append("esmc_multiregion_smooth")
    if "esmc_multiregion_tree" in all_oof:
        track_b_heads.append("esmc_multiregion_tree")
    if "esmc_multiregion_smooth" not in all_oof and "esmc_multiregion_tree" not in all_oof:
        track_b_heads.append("esmc_multiregion")
    if "direct_smiles_smooth" in all_oof:
        track_b_heads.append("direct_smiles_smooth")
    if "direct_dual_smooth" in all_oof:
        track_b_heads.append("direct_dual_smooth")
    if "direct_dual_tree" in all_oof:
        track_b_heads.append("direct_dual_tree")
    if "cpu_seq_only" in all_oof and n_cv >= 300:
        track_b_heads.append("cpu_seq_only")
    if aft_direct_dual is not None:
        track_b_heads.extend(["cpu_aft_dual", "aft_direct_dual"])

    # Deduplicate preserving order
    track_b_heads = list(dict.fromkeys(track_b_heads))

    stack_b = run_nested_stacking(
        track_b_heads, all_oof, all_ho,
        y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho,
        task_type=task_type, bounds=bounds, eval_mask_cv=dual_cv_mask, eval_mask_ho=dual_ho_mask, is_aft_stack=False,
    )

    if aft_direct_dual is not None and aft_stack_dual is not None and aft_stack_smi is not None:
        aft_ablation_doc = {
            "n_censored_total": n_cens_total,
            "n_censored_cv": n_cens_cv,
            "n_censored_holdout": n_cens_ho,
            "aft_direct_dual_recipe": aft_direct_dual["recipe"],
            "aft_direct_smiles_recipe": aft_direct_smi["recipe"],
            "aft_stack_dual_weights": aft_stack_dual["weights"],
            "aft_stack_smiles_weights": aft_stack_smi["weights"],
            "cpu_exact_holdout": rep_ablation_doc["cpu_dual_modality"]["ho_metrics"],
            "cpu_aft_holdout": regression_metrics(y_ho, cpu["ho_aft"], yl_ho, yu_ho, ct_ho),
            "fusion_exact_holdout": stack_b["ho_metrics"],
            "aft_direct_multimodal_cv": aft_direct_dual["cv_metrics"],
            "aft_direct_multimodal_holdout": aft_direct_dual["ho_metrics"],
            "aft_stacked_multimodal_cv": aft_stack_dual["cv_metrics"],
            "aft_stacked_multimodal_holdout": aft_stack_dual["ho_metrics"],
            "aft_stacked_smiles_holdout": aft_stack_smi["ho_metrics"],
        }

    # Compute ensemble fold_sd on holdout cleanly across active heads
    def _approx_ho_sd(active_heads: List[str], weights: Dict[str, float], mask: np.ndarray) -> np.ndarray:
        sd_out = np.full(n_ho, np.nan, dtype=np.float64)
        valid_sds = []
        valid_ws = []
        for h in active_heads:
            w = abs(float(weights.get(h, 0.0)))
            if w <= 1e-6:
                continue
            if h in trained_heads:
                valid_sds.append(np.nan_to_num(trained_heads[h]["ho_sd"], nan=0.0))
                valid_ws.append(w)
        if valid_sds:
            w_arr = np.array(valid_ws)
            w_arr = w_arr / np.sum(w_arr)
            # Combine weighted fold SD + inter-head disagreement
            head_mat = np.column_stack([all_ho[h] for h in active_heads if h in all_ho])
            inter_sd = np.nanstd(head_mat, axis=1) if head_mat.shape[1] > 1 else np.zeros(n_ho)
            intra_sd = np.sum([w * s for w, s in zip(w_arr, valid_sds)], axis=0)
            sd_out[mask] = np.sqrt(intra_sd[mask] ** 2 + 0.25 * np.nan_to_num(inter_sd[mask], nan=0.0) ** 2)
        else:
            sd_out[mask] = 0.0
        return sd_out

    sd_ho_smi = _approx_ho_sd(stack_a["active_heads"], stack_a["weights"], smi_ho_mask)
    sd_ho_dual = _approx_ho_sd(stack_b["active_heads"], stack_b["weights"], dual_ho_mask)

    # =========================================================================
    # 4. Applicability Domain, Probability Calibration & Conformal Intervals
    # =========================================================================
    ad_index = TanimotoADIndex.from_indices(bank, cidx_cv)
    oof_max_t = compute_oof_tanimoto(bank, ep_data)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, cidx_ho)

    cal_smi = cal_dual = None
    conf_smi = conf_dual = None
    cal_summary: Optional[Dict[str, Any]] = None
    conf_summary: Optional[Dict[str, Any]] = None
    p_oof_smi_cal = p_oof_dual_cal = None
    p_ho_smi_cal = p_ho_dual_cal = None
    ho_iv_smi = ho_iv_dual = None

    if task_type == "classification":
        m_ho_smi_raw = stack_a["ho_metrics"]
        m_ho_dual_raw = stack_b["ho_metrics"]

        cal_smi = ProbabilityCalibrator().fit(stack_a["nested_oof"], y_cv, fids_cv)
        cal_dual = ProbabilityCalibrator().fit(stack_b["nested_oof"], y_cv, fids_cv)

        p_oof_smi_cal = cal_smi.transform(stack_a["nested_oof"])
        p_oof_dual_cal = cal_dual.transform(stack_b["nested_oof"])
        p_ho_smi_cal = cal_smi.transform(stack_a["ho_pred"])
        p_ho_dual_cal = cal_dual.transform(stack_b["ho_pred"])

        m_ho_smi_cal = classification_metrics(y_ho[smi_ho_mask], p_ho_smi_cal[smi_ho_mask])
        m_ho_dual_cal = classification_metrics(y_ho, p_ho_dual_cal)

        cal_summary = {
            "smiles_only": {
                "method": cal_smi.method_,
                "nested_cv_comparison": cal_smi.cv_comparison_,
                "holdout_raw_brier": m_ho_smi_raw["brier"],
                "holdout_cal_brier": m_ho_smi_cal["brier"],
                "holdout_raw_ece": m_ho_smi_raw["ece"],
                "holdout_cal_ece": m_ho_smi_cal["ece"],
            },
            "dual_modality": {
                "method": cal_dual.method_,
                "nested_cv_comparison": cal_dual.cv_comparison_,
                "holdout_raw_brier": m_ho_dual_raw["brier"],
                "holdout_cal_brier": m_ho_dual_cal["brier"],
                "holdout_raw_ece": m_ho_dual_raw["ece"],
                "holdout_cal_ece": m_ho_dual_cal["ece"],
            },
        }
        ho_metrics_smi_final = m_ho_smi_cal
        ho_metrics_dual_final = m_ho_dual_cal
        ad_strat_dual = compute_ad_stratification(y_ho, p_ho_dual_cal, ct_ho, ho_max_t, ho_ad_tiers, "classification")
    else:
        ho_metrics_smi_final = stack_a["ho_metrics"]
        ho_metrics_dual_final = stack_b["ho_metrics"]

        conf_smi = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv[smi_cv_mask], stack_a["nested_oof"][smi_cv_mask], ct_cv[smi_cv_mask], oof_max_tanimoto=oof_max_t[smi_cv_mask]
        )
        conf_dual = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv, stack_b["nested_oof"], ct_cv, oof_max_tanimoto=oof_max_t
        )
        ho_iv_smi = conf_smi.predict_intervals(stack_a["ho_pred"], max_tanimoto=ho_max_t)
        ho_iv_dual = conf_dual.predict_intervals(stack_b["ho_pred"], max_tanimoto=ho_max_t)

        conf_summary = {
            "smiles_only": {
                "cv_diagnostics": conf_smi.cv_diagnostics_,
                "holdout_coverage": conf_smi.evaluate_coverage(
                    y_ho[smi_ho_mask], stack_a["ho_pred"][smi_ho_mask], ct_ho[smi_ho_mask], max_tanimoto=ho_max_t[smi_ho_mask]
                ),
            },
            "dual_modality": {
                "cv_diagnostics": conf_dual.cv_diagnostics_,
                "holdout_coverage": conf_dual.evaluate_coverage(y_ho, stack_b["ho_pred"], ct_ho, max_tanimoto=ho_max_t),
            },
        }
        ad_strat_dual = compute_ad_stratification(y_ho, stack_b["ho_pred"], ct_ho, ho_max_t, ho_ad_tiers, "regression")

    ad_index.tier_calibration = ad_strat_dual

    # =========================================================================
    # 5. Strict Zero-Leakage Leave-Case17-Out Refit & Clinical Drug Evaluation!
    # =========================================================================
    c17_idx = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)
    keep_non_c17 = np.where(ep_data["case17_excluded"] == 0)[0]
    non_c17_cv_mask = ep_data["case17_excluded"][cv_rows] == 0

    # Refit winning stackers strictly on case17_excluded == 0 OOF rows!
    def _fit_non_c17_stacker(
        head_names: List[str],
        stack_res: Dict[str, Any],
        extra_mask: np.ndarray,
        is_aft_st: bool = False,
    ) -> ModalityAwareStacker:
        m_fit = non_c17_cv_mask & extra_mask
        if task_type == "regression" and not is_aft_st:
            m_fit = m_fit & (ct_cv == "exact")
        if m_fit.sum() < 6:
            return stack_res["stacker"]
        P_sub = np.column_stack([all_oof[h][m_fit] for h in head_names])
        st = ModalityAwareStacker(
            task_type=task_type,
            method=stack_res["best_cfg"]["method"],
            params=stack_res["best_cfg"]["params"],
            bounds=bounds,
        )
        st.fit(
            P_sub,
            y_cv[m_fit],
            head_names=head_names,
            y_lower=yl_cv[m_fit] if is_aft_st else None,
            y_upper=yu_cv[m_fit] if is_aft_st else None,
            censor_type=ct_cv[m_fit] if is_aft_st else None,
        )
        return st

    c17_stacker_a = _fit_non_c17_stacker(track_a_heads, stack_a, smi_cv_mask, is_aft_st=False)
    c17_stacker_b = _fit_non_c17_stacker(track_b_heads, stack_b, dual_cv_mask, is_aft_st=False)
    c17_stacker_aft = (
        _fit_non_c17_stacker(["cpu_aft_dual", "aft_direct_dual"], aft_stack_dual, dual_cv_mask, is_aft_st=True)
        if aft_stack_dual is not None
        else None
    )

    # Identify all active pretrained heads needed for Leave-Case17-Out refit
    needed_pretrained_heads = set(c17_stacker_a.active_heads()) | set(c17_stacker_b.active_heads())
    if c17_stacker_aft is not None:
        needed_pretrained_heads |= set(c17_stacker_aft.active_heads())
    needed_pretrained_heads = {h for h in needed_pretrained_heads if h in trained_heads}

    # Pre-seed Case17 base predictions with Wave-2C's strictly Leave-Case17-Out CPU predictions
    c17_base_mean: Dict[str, np.ndarray] = {
        "cpu_smiles": cpu["c17_smiles_only"],
        "cpu_dual": cpu["c17_dual"],
    }
    c17_base_sd: Dict[str, np.ndarray] = {}
    c17_base_alt: Dict[str, Dict[str, np.ndarray]] = {}
    for alt_k, alt_d in cpu["c17_alt"].items():
        c17_base_alt.setdefault(alt_k, {})
        c17_base_alt[alt_k]["cpu_smiles"] = alt_d["smiles_only"]
        c17_base_alt[alt_k]["cpu_dual"] = alt_d["dual"]
        if alt_d.get("aft") is not None:
            c17_base_alt[alt_k]["cpu_aft_dual"] = alt_d["aft"]
            c17_base_alt[alt_k]["cpu_aft_smiles"] = alt_d["aft"]
    if cpu["c17_aft"] is not None:
        c17_base_mean["cpu_aft_dual"] = cpu["c17_aft"]
        c17_base_mean["cpu_aft_smiles"] = cpu["c17_aft"]
    if cpu["c17_seq_only"] is not None:
        c17_base_mean["cpu_seq_only"] = cpu["c17_seq_only"]

    # Refit every active pretrained head strictly on case17_excluded == 0!
    for hname in sorted(needed_pretrained_heads):
        c17_res = refit_head_leave_case17_out(bank, ep_data, spec, trained_heads[hname], c17_idx, n_jobs=n_jobs)
        c17_base_mean[hname] = c17_res["mean"]
        c17_base_sd[hname] = c17_res["sd"]
        for alt_k, arr in c17_res["alt"].items():
            c17_base_alt.setdefault(alt_k, {})[hname] = arr

    def _combine_c17(stacker: ModalityAwareStacker, head_names: List[str], src_map: Dict[str, np.ndarray]) -> np.ndarray:
        cols = [
            src_map[h] if (h in src_map and src_map[h] is not None) else np.full(len(c17_idx), np.nan, dtype=np.float64)
            for h in head_names
        ]
        return stacker.predict(np.column_stack(cols))

    c17_p_smi = _combine_c17(c17_stacker_a, track_a_heads, c17_base_mean)
    c17_p_dual = _combine_c17(c17_stacker_b, track_b_heads, c17_base_mean)
    c17_p_aft = (
        _combine_c17(c17_stacker_aft, ["cpu_aft_dual", "aft_direct_dual"], c17_base_mean)
        if c17_stacker_aft is not None
        else None
    )

    # Fold SD for Case17 dual prediction
    c17_act_sds = [c17_base_sd[h] for h in c17_stacker_b.active_heads() if h in c17_base_sd]
    c17_act_preds = [c17_base_mean[h] for h in c17_stacker_b.active_heads() if h in c17_base_mean]
    if c17_act_sds:
        intra = np.mean(np.stack(c17_act_sds, axis=0), axis=0)
        inter = np.std(np.stack(c17_act_preds, axis=0), axis=0) if len(c17_act_preds) > 1 else np.zeros(len(c17_idx))
        c17_sd_dual = np.sqrt(intra ** 2 + 0.25 * inter ** 2)
    else:
        c17_sd_dual = np.zeros(len(c17_idx), dtype=np.float64)

    # Alternate clinical routes for Case17 (e.g. SC for half_life_reg & F, SC_CL_over_F for CL_absolute)
    c17_alt_out: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for alt_k in spec.get("alt_case17_conditions", {}):
        alt_src = dict(c17_base_mean)
        alt_src.update(c17_base_alt.get(alt_k, {}))
        c17_alt_out[alt_k] = {
            "smiles_only": _combine_c17(c17_stacker_a, track_a_heads, alt_src),
            "dual": _combine_c17(c17_stacker_b, track_b_heads, alt_src),
            "aft": (
                _combine_c17(c17_stacker_aft, ["cpu_aft_dual", "aft_direct_dual"], alt_src)
                if c17_stacker_aft is not None
                else None
            ),
        }

    # Case17 AD strictly against non-Case17 compounds + non-Case17 calibrators/conformal!
    ad_c17 = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][keep_non_c17])
    c17_max_t, c17_top5_t, c17_ad_tiers = ad_c17.query(bank, c17_idx)

    c17_p_smi_cal = c17_p_dual_cal = None
    c17_iv_dual = None
    if task_type == "classification":
        cal_smi_c17 = ProbabilityCalibrator().fit(stack_a["nested_oof"][non_c17_cv_mask], y_cv[non_c17_cv_mask], fids_cv[non_c17_cv_mask])
        cal_dual_c17 = ProbabilityCalibrator().fit(stack_b["nested_oof"][non_c17_cv_mask], y_cv[non_c17_cv_mask], fids_cv[non_c17_cv_mask])
        c17_p_smi_cal = cal_smi_c17.transform(c17_p_smi)
        c17_p_dual_cal = cal_dual_c17.transform(c17_p_dual)
    else:
        conf_dual_c17 = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv[non_c17_cv_mask], stack_b["nested_oof"][non_c17_cv_mask], ct_cv[non_c17_cv_mask], oof_max_tanimoto=oof_max_t[non_c17_cv_mask]
        )
        c17_iv_dual = conf_dual_c17.predict_intervals(c17_p_dual, max_tanimoto=c17_max_t)

    # =========================================================================
    # 6. Write Standardized Per-Endpoint Artifacts
    # =========================================================================
    ep_out_dir = OUT_DIR / endpoint
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.6f}"
        return str(v)

    rep_export_cols = [
        ("rep_cpu_smiles", "cpu_smiles"),
        ("rep_cpu_dual", "cpu_dual"),
        ("rep_ncaa_smiles", "ncaa_smiles_repr"),
        ("rep_ncaa_fusion", "ncaa_fusion_hf"),
        ("rep_ncaa_prot", "ncaa_protein_proj"),
        ("rep_peptideclm2", "peptideclm2"),
        ("rep_chemberta", "chemberta_mtr"),
        ("rep_esmc_mean", "esmc_mean"),
        ("rep_esmc_multiregion", "esmc_multiregion"),
        ("rep_direct_smiles", "direct_smiles"),
        ("rep_direct_dual", "direct_dual"),
    ]

    # (a) oof_predictions.tsv
    oof_cols = [
        "record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual",
    ]
    if aft_stack_dual is not None:
        oof_cols.extend(["pred_aft", "pred_aft_smiles_only", "pred_aft_direct_dual"])
    if task_type == "classification":
        oof_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    oof_cols.extend([c for c, _ in rep_export_cols])
    oof_cols.append("max_tanimoto_oof")

    with (ep_out_dir / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(oof_cols)
        for loc_i, orig_i in enumerate(cv_rows):
            row = [
                ep_data["record_ids"][orig_i],
                ep_data["compound_keys"][orig_i],
                int(fids_cv[loc_i]),
                _fmt(y_cv[loc_i]),
                _fmt(yl_cv[loc_i]),
                _fmt(yu_cv[loc_i]),
                str(ct_cv[loc_i]),
                _fmt(stack_a["nested_oof"][loc_i]),
                _fmt(stack_b["nested_oof"][loc_i]),
            ]
            if aft_stack_dual is not None:
                row.extend([
                    _fmt(aft_stack_dual["nested_oof"][loc_i]),
                    _fmt(aft_stack_smi["nested_oof"][loc_i]),
                    _fmt(aft_direct_dual["oof_pred"][loc_i]),
                ])
            if task_type == "classification":
                row.extend([_fmt(p_oof_smi_cal[loc_i]), _fmt(p_oof_dual_cal[loc_i])])
            for _, src_k in rep_export_cols:
                row.append(_fmt(all_oof[src_k][loc_i]) if src_k in all_oof else "")
            row.append(_fmt(oof_max_t[loc_i]))
            w.writerow(row)

    # (b) holdout_predictions.tsv
    ho_cols = [
        "record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual",
    ]
    if aft_stack_dual is not None:
        ho_cols.extend(["pred_aft", "pred_aft_smiles_only", "pred_aft_direct_dual"])
    ho_cols.extend(["fold_sd", "fold_sd_smiles_only"])
    if task_type == "classification":
        ho_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    else:
        ho_cols.extend(["lo80", "hi80", "lo90", "hi90", "lo80_smiles", "hi80_smiles", "lo90_smiles", "hi90_smiles"])
    ho_cols.extend([c for c, _ in rep_export_cols])
    ho_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])

    with (ep_out_dir / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ho_cols)
        for loc_i, orig_i in enumerate(ho_rows):
            row = [
                ep_data["record_ids"][orig_i],
                ep_data["compound_keys"][orig_i],
                _fmt(y_ho[loc_i]),
                _fmt(yl_ho[loc_i]),
                _fmt(yu_ho[loc_i]),
                str(ct_ho[loc_i]),
                _fmt(stack_a["ho_pred"][loc_i]),
                _fmt(stack_b["ho_pred"][loc_i]),
            ]
            if aft_stack_dual is not None:
                row.extend([
                    _fmt(aft_stack_dual["ho_pred"][loc_i]),
                    _fmt(aft_stack_smi["ho_pred"][loc_i]),
                    _fmt(aft_direct_dual["ho_pred"][loc_i]),
                ])
            row.extend([_fmt(sd_ho_dual[loc_i]), _fmt(sd_ho_smi[loc_i])])
            if task_type == "classification":
                row.extend([_fmt(p_ho_smi_cal[loc_i]), _fmt(p_ho_dual_cal[loc_i])])
            else:
                row.extend([
                    _fmt(ho_iv_dual["lo80"][loc_i]), _fmt(ho_iv_dual["hi80"][loc_i]),
                    _fmt(ho_iv_dual["lo90"][loc_i]), _fmt(ho_iv_dual["hi90"][loc_i]),
                    _fmt(ho_iv_smi["lo80"][loc_i]), _fmt(ho_iv_smi["hi80"][loc_i]),
                    _fmt(ho_iv_smi["lo90"][loc_i]), _fmt(ho_iv_smi["hi90"][loc_i]),
                ])
            for _, src_k in rep_export_cols:
                row.append(_fmt(all_ho[src_k][loc_i]) if src_k in all_ho else "")
            row.extend([_fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i]])
            w.writerow(row)

    # (c) case17_predictions.tsv
    c17_cols = [
        "drug_name", "compound_key", "compound_idx", "clinical_route", "condition_used",
        "pred_smiles_only", "pred_dual",
    ]
    if c17_p_aft is not None:
        c17_cols.append("pred_aft")
    if task_type == "classification":
        c17_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    c17_cols.append("fold_sd")
    if c17_iv_dual is not None:
        c17_cols.extend(["lo80", "hi80", "lo90", "hi90"])
    c17_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])
    for alt_k in c17_alt_out:
        c17_cols.extend([f"pred_smiles_only_{alt_k}", f"pred_dual_{alt_k}"])
        if c17_alt_out[alt_k]["aft"] is not None:
            c17_cols.append(f"pred_aft_{alt_k}")
    if endpoint == "half_life_reg":
        c17_cols.extend(["anchor_class", "anchor_mid_h", "anchor_range_h"])

    case17_records: List[Dict[str, Any]] = []
    cond_str = json.dumps(spec["canonical_conditions"], ensure_ascii=False)
    with (ep_out_dir / "case17_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(c17_cols)
        for i, q in enumerate(bank.case17_queries):
            dname = q["drug_name"]
            row = [
                dname,
                q["compound_key"],
                int(q["compound_idx"]),
                CASE17_CLINICAL_ROUTE.get(dname, "IV"),
                cond_str,
                _fmt(c17_p_smi[i]),
                _fmt(c17_p_dual[i]),
            ]
            rec: Dict[str, Any] = {
                "drug_name": dname,
                "clinical_route": CASE17_CLINICAL_ROUTE.get(dname, "IV"),
                "pred_smiles_only": round(float(c17_p_smi[i]), 5) if np.isfinite(c17_p_smi[i]) else None,
                "pred_dual": round(float(c17_p_dual[i]), 5) if np.isfinite(c17_p_dual[i]) else None,
                "fold_sd": round(float(c17_sd_dual[i]), 5) if np.isfinite(c17_sd_dual[i]) else None,
                "max_tanimoto": round(float(c17_max_t[i]), 4) if np.isfinite(c17_max_t[i]) else None,
                "ad_tier": c17_ad_tiers[i],
            }
            if c17_p_aft is not None:
                row.append(_fmt(c17_p_aft[i]))
                rec["pred_aft"] = round(float(c17_p_aft[i]), 5) if np.isfinite(c17_p_aft[i]) else None
            if task_type == "classification":
                row.extend([_fmt(c17_p_smi_cal[i]), _fmt(c17_p_dual_cal[i])])
                rec["pred_smiles_only_cal"] = round(float(c17_p_smi_cal[i]), 5)
                rec["pred_dual_cal"] = round(float(c17_p_dual_cal[i]), 5)
            row.append(_fmt(c17_sd_dual[i]))
            if c17_iv_dual is not None:
                row.extend([
                    _fmt(c17_iv_dual["lo80"][i]), _fmt(c17_iv_dual["hi80"][i]),
                    _fmt(c17_iv_dual["lo90"][i]), _fmt(c17_iv_dual["hi90"][i]),
                ])
                rec["ci80"] = [round(float(c17_iv_dual["lo80"][i]), 4), round(float(c17_iv_dual["hi80"][i]), 4)]
                rec["ci90"] = [round(float(c17_iv_dual["lo90"][i]), 4), round(float(c17_iv_dual["hi90"][i]), 4)]
            row.extend([_fmt(c17_max_t[i]), _fmt(c17_top5_t[i]), c17_ad_tiers[i]])
            for alt_k, ap_d in c17_alt_out.items():
                row.extend([_fmt(ap_d["smiles_only"][i]), _fmt(ap_d["dual"][i])])
                rec[f"pred_smiles_only_{alt_k}"] = round(float(ap_d["smiles_only"][i]), 5)
                rec[f"pred_dual_{alt_k}"] = round(float(ap_d["dual"][i]), 5)
                if ap_d["aft"] is not None:
                    row.append(_fmt(ap_d["aft"][i]))
                    rec[f"pred_aft_{alt_k}"] = round(float(ap_d["aft"][i]), 5)
            if endpoint == "half_life_reg":
                anch = compute_halflife_mechanism_anchor(q["canonical_smiles"])
                row.extend([
                    anch["anchor_class"],
                    "" if anch["anchor_mid_h"] is None else f"{anch['anchor_mid_h']:.1f}",
                    "" if anch["anchor_range_h"] is None else json.dumps(anch["anchor_range_h"]),
                ])
                rec["mechanism_anchor"] = anch
            w.writerow(row)
            case17_records.append(rec)

    # (d) metrics.json
    censor_counts = {k: int((ep_data["censor_type"] == k).sum()) for k in ("exact", "right", "left", "interval")}
    elapsed = round(time.time() - t0, 2)
    metrics_doc: Dict[str, Any] = {
        "endpoint": endpoint,
        "tier": spec["tier"],
        "task_type": task_type,
        "size_tier": spec["size_tier"],
        "unit": spec["unit"],
        "bounds": list(bounds) if bounds is not None else None,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": spec["canonical_conditions"],
        "n_total": ep_data["n"],
        "n_cv": n_cv,
        "n_holdout": n_ho,
        "n_case17_excluded": int((ep_data["case17_excluded"] == 1).sum()),
        "n_train_leave_case17_out": int(len(keep_non_c17)),
        "censor_counts": censor_counts,
        "smiles_valid_pct": round(100.0 * float(bank.smiles_valid[ep_data["compound_idx"]].mean()), 2),
        "seq_valid_pct": round(100.0 * float(bank.seq_valid[ep_data["compound_idx"]].mean()), 2),
        "representation_ablation": rep_ablation_doc,
        "fusion_ablation": {
            "smiles_only": {
                "candidate_heads": track_a_heads,
                "winning_meta": stack_a["best_cfg"],
                "weights": stack_a["weights"],
                "active_heads": stack_a["active_heads"],
                "nested_cv_metrics": stack_a["cv_metrics"],
                "holdout_metrics": ho_metrics_smi_final,
                "holdout_metrics_raw": stack_a["ho_metrics"] if task_type == "classification" else None,
                "meta_candidate_ablation": stack_a["candidate_ablation"],
            },
            "dual_modality": {
                "candidate_heads": track_b_heads,
                "winning_meta": stack_b["best_cfg"],
                "weights": stack_b["weights"],
                "active_heads": stack_b["active_heads"],
                "nested_cv_metrics": stack_b["cv_metrics"],
                "holdout_metrics": ho_metrics_dual_final,
                "holdout_metrics_raw": stack_b["ho_metrics"] if task_type == "classification" else None,
                "meta_candidate_ablation": stack_b["candidate_ablation"],
            },
        },
        "cv_metrics": {
            "smiles_only": stack_a["cv_metrics"],
            "dual_modality": stack_b["cv_metrics"],
        },
        "holdout_metrics": {
            "smiles_only": ho_metrics_smi_final,
            "dual_modality": ho_metrics_dual_final,
            "smiles_only_raw": stack_a["ho_metrics"] if task_type == "classification" else None,
            "dual_modality_raw": stack_b["ho_metrics"] if task_type == "classification" else None,
        },
        "aft_multimodal_ablation": aft_ablation_doc,
        "calibration": cal_summary,
        "conformal": conf_summary,
        "applicability_domain": ad_strat_dual,
        "case17_leaveout_predictions": case17_records,
        "elapsed_sec": elapsed,
    }
    (ep_out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # (e) fusion_models_5fold.joblib (complete inference-ready 5-fold base + meta stack + calibrator + conformal + AD)
    save_head_keys = set(stack_a["active_heads"]) | set(stack_b["active_heads"])
    if aft_stack_dual is not None:
        save_head_keys |= set(aft_stack_dual["active_heads"]) | set(aft_stack_smi["active_heads"])
    serialized_heads: Dict[str, Any] = {}
    for hk in sorted(save_head_keys):
        if hk in trained_heads:
            ho = trained_heads[hk]
            serialized_heads[hk] = {
                "rep_name": ho["rep_name"],
                "rep_cfg": ho["rep_cfg"],
                "recipe": ho["recipe"],
                "is_aft": ho["is_aft"],
                "folds": ho["folds"],
            }

    model_bundle: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v33_fusion_5fold_v1",
        "endpoint": endpoint,
        "task_type": task_type,
        "unit": spec["unit"],
        "bounds": bounds,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": spec["canonical_conditions"],
        "cpu_model_ref": f"/data/qlyu/v33_upgrade/models_cpu/{endpoint}/models_5fold.joblib",
        "pretrained_base_heads": serialized_heads,
        "smiles_only": {
            "candidate_heads": track_a_heads,
            "stacker": stack_a["stacker"],
            "weights": stack_a["weights"],
            "calibrator": cal_smi,
            "conformal": conf_smi,
        },
        "dual_modality": {
            "candidate_heads": track_b_heads,
            "stacker": stack_b["stacker"],
            "weights": stack_b["weights"],
            "calibrator": cal_dual,
            "conformal": conf_dual,
        },
        "aft": (
            {
                "stacker_dual": aft_stack_dual["stacker"],
                "weights_dual": aft_stack_dual["weights"],
                "stacker_smiles_only": aft_stack_smi["stacker"],
                "weights_smiles_only": aft_stack_smi["weights"],
            }
            if aft_stack_dual is not None
            else None
        ),
        "ad_index": ad_index,
    }
    joblib.dump(model_bundle, ep_out_dir / "fusion_models_5fold.joblib", compress=3)

    # Compact status log comparing Wave-2C CPU vs Wave-2D Multimodal Fusion
    if task_type == "classification":
        c_roc = rep_ablation_doc["cpu_dual_modality"]["ho_metrics"]["roc_auc"]
        f_roc = ho_metrics_dual_final["roc_auc"]
        f_pr = ho_metrics_dual_final["pr_auc"]
        s_roc = ho_metrics_smi_final["roc_auc"]
        s_str = f"CPU ROC={c_roc:.4f} -> FUSION Dual ROC={f_roc:.4f} (Δ{f_roc-c_roc:+.4f}) PR={f_pr:.4f} | SMI ROC={s_roc:.4f} | meta={stack_b['best_cfg']['method']}"
    else:
        c_mae = rep_ablation_doc["cpu_dual_modality"]["ho_metrics"]["mae"]
        c_sp = rep_ablation_doc["cpu_dual_modality"]["ho_metrics"]["spearman"]
        f_mae = ho_metrics_dual_final["mae"]
        f_sp = ho_metrics_dual_final["spearman"]
        s_mae = ho_metrics_smi_final["mae"]
        s_str = f"CPU MAE={c_mae:.4f}/Sp={c_sp:.3f} -> FUSION MAE={f_mae:.4f}(Δ{f_mae-c_mae:+.4f})/Sp={f_sp:.3f}(Δ{f_sp-c_sp:+.3f}) | SMI MAE={s_mae:.4f} | meta={stack_b['best_cfg']['method']}"
    print(f"[DONE ] {endpoint:<24} ({elapsed:5.1f}s) | {s_str}", flush=True)

    gc.collect()
    return {"endpoint": endpoint, "metrics": metrics_doc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test on BBB and PPB")
    parser.add_argument("--endpoints", nargs="*", default=None, help="Optional subset of endpoints")
    parser.add_argument("--max-workers", type=int, default=12, help="Parallel worker processes across 64 cores")
    parser.add_argument("--inner-jobs", type=int, default=5, help="Threads per worker for GBDT/AFT")
    args = parser.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading frozen V33 FeatureBank + 4 GPU Embedding Caches into CoW shared memory...", flush=True)
    global GLOBAL_BANK
    GLOBAL_BANK = FusionBank()
    print(
        f"FusionBank ready in {time.time() - t_all:.2f}s (46,598 compounds, {len(GLOBAL_BANK.emb_blocks)} embedding blocks).",
        flush=True,
    )

    ctx = mp.get_context("fork")

    if args.smoke_test:
        with ctx.Pool(processes=2) as pool:
            res = pool.map(run_single_endpoint, [("BBB", 8), ("PPB", 8)])
        print(f"Smoke test PASSED in {time.time() - t_all:.1f}s!", flush=True)
        return

    # LPT ordering: largest endpoints first so all 64 cores stay maximally saturated to the finish!
    lpt_order = [
        ("hemolysis_cls", 8),
        ("overall_peptide_toxicity", 8),
        ("PAMPA", 7),
        ("hemolysis_HC50", 7),
        ("solubility_cls", 5),
        ("half_life_cls", 5),
        ("half_life_reg", 5),
        ("cytotoxicity_cls", 5),
        ("MIC_E_coli", 5),
        ("MIC_S_aureus", 5),
        ("Caco2", 4),
        ("MIC_P_aeruginosa", 4),
        ("BBB", 4),
        ("LogD7.4", 4),
        ("MIC_C_albicans", 4),
        ("solubility_reg", 3),
        ("CL_per_kg", 3),
        ("MDCK", 3),
        ("RRCK", 3),
        ("Vd_per_kg", 3),
        ("PPB", 3),
        ("F", 3),
        ("Cmax", 3),
        ("CL_absolute", 3),
    ]

    if args.endpoints:
        req = set(args.endpoints)
        tasks = [(ep, nj) for ep, nj in lpt_order if ep in req]
    else:
        tasks = lpt_order

    n_workers = min(args.max_workers, len(tasks))
    print(f"=== Running {len(tasks)} endpoints across {n_workers} parallel workers (64-core node23) ===", flush=True)

    summary_by_ep: Dict[str, Dict[str, Any]] = {}
    with ctx.Pool(processes=n_workers) as pool:
        for item in pool.imap_unordered(run_single_endpoint, tasks):
            ep = item["endpoint"]
            summary_by_ep[ep] = item["metrics"]

    # If partial --endpoints run, merge any previously saved metrics.json into master summary
    if args.endpoints:
        for ep in sorted(ENDPOINT_SPECS.keys()):
            if ep not in summary_by_ep:
                mp_path = OUT_DIR / ep / "metrics.json"
                if mp_path.exists():
                    summary_by_ep[ep] = json.loads(mp_path.read_text(encoding="utf-8"))

    master = {
        "schema": "peptide_omnipanel_v33_fusion_benchmark_summary_v1",
        "total_endpoints": len(summary_by_ep),
        "total_elapsed_sec": round(time.time() - t_all, 2),
        "endpoints": {ep: summary_by_ep[ep] for ep in sorted(summary_by_ep.keys())},
    }
    master_path = OUT_DIR / "fusion_benchmark_summary.json"
    master_path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ALL {len(summary_by_ep)} ENDPOINTS COMPLETE in {time.time() - t_all:.1f}s! Saved {master_path}", flush=True)


if __name__ == "__main__":
    main()
