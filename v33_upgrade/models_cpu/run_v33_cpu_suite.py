#!/usr/bin/env python3
"""Peptide OmniPanel V33 — 64-Core Parallel CPU Modeling, Censored AFT, Small-Sample GPR/Stacking,
Calibration, Conformal Uncertainty, AD & Zero-Leakage Case17 Benchmark Suite.

Strictly owns and writes ONLY under <WORK_DIR>/v33_upgrade/models_cpu/.
Read-only access to <WORK_DIR>/v33_upgrade/data/ and <WORK_DIR>/v33_upgrade/features/.
"""
from __future__ import annotations

import os

# CRITICAL: Set single-threaded BLAS/OpenMP BEFORE importing numpy/scipy/sklearn/lgb/xgb
# to guarantee 100% fork safety and zero thread oversubscription across 64 cores!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from v33_config import (
    AUX_ENDPOINTS,
    CASE17_CLINICAL_ROUTE,
    CASE17_FDA_NUMERIC,
    ENDPOINT_SPECS,
    ENDPOINTS_DIR,
    OUT_DIR,
    PK_META_TARGETS,
    SEQ_ONLY_ENDPOINTS,
    compute_halflife_mechanism_anchor,
)
from v33_features import FeatureBank, FoldPreprocessor, TanimotoADIndex
from v33_metrics import (
    ConformalIntervalCalibrator,
    ProbabilityCalibrator,
    classification_metrics,
    compute_ad_stratification,
    regression_metrics,
)
from v33_models import (
    XGBoostAFTRegressor,
    get_aft_candidates,
    get_candidate_recipes,
    make_estimator,
)

csv.field_size_limit(10_000_000)

# Global read-only bank shared across forked processes via Copy-on-Write
GLOBAL_BANK: Optional[FeatureBank] = None
GLOBAL_META_CACHE: Optional[Dict[str, Any]] = None


def load_endpoint_data(endpoint: str, bank: FeatureBank) -> Dict[str, Any]:
    path = ENDPOINTS_DIR / f"{endpoint}.tsv"
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = list(reader.fieldnames or [])
        rows = list(reader)

    cond_cols = [c for c in cols if c.startswith("cond_")]
    n = len(rows)
    record_ids = [r["record_id"] for r in rows]
    compound_keys = [r["compound_key"] for r in rows]
    compound_idx = bank.map_keys(compound_keys)
    cluster_ids = np.array([r["cluster_id"] for r in rows], dtype=object)
    is_holdout = np.array([int(r["is_holdout"]) for r in rows], dtype=np.int32)
    fold_id = np.array([int(r["fold_id"]) for r in rows], dtype=np.int32)
    case17_excluded = np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32)
    censor_type = np.array([r["censor_type"] for r in rows], dtype=object)
    y_value = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)
    y_lower = np.array([float(r["y_lower"]) for r in rows], dtype=np.float64)
    y_upper = np.array([float(r["y_upper"]) for r in rows], dtype=np.float64)
    cond_rows = [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows]

    cv_mask = is_holdout == 0
    ho_mask = is_holdout == 1
    assert int(cv_mask.sum()) + int(ho_mask.sum()) == n
    assert set(cluster_ids[cv_mask]).isdisjoint(set(cluster_ids[ho_mask])), f"{endpoint}: cluster leakage!"

    return {
        "endpoint": endpoint,
        "n": n,
        "cond_cols": cond_cols,
        "record_ids": record_ids,
        "compound_keys": compound_keys,
        "compound_idx": compound_idx,
        "cluster_ids": cluster_ids,
        "is_holdout": is_holdout,
        "fold_id": fold_id,
        "case17_excluded": case17_excluded,
        "censor_type": censor_type,
        "y_value": y_value,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "cond_rows": cond_rows,
        "cv_idx": np.where(cv_mask)[0],
        "ho_idx": np.where(ho_mask)[0],
    }


def _track_valid_mask(bank: FeatureBank, compound_idx: np.ndarray, track: str) -> np.ndarray:
    if track == "smiles_only":
        return bank.smiles_valid[compound_idx].copy()
    if track == "seq_only":
        return bank.seq_valid[compound_idx].copy()
    return np.ones(len(compound_idx), dtype=bool)


def _predict_single(est: Any, X: Any, task_type: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if task_type == "classification":
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1].astype(np.float64), None
        dec = est.decision_function(X).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(dec, -40.0, 40.0))), None
    if hasattr(est, "predict_with_std"):
        m, s = est.predict_with_std(X)
        return m.astype(np.float64), s.astype(np.float64)
    return np.asarray(est.predict(X), dtype=np.float64), None


def _build_fold_cache(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    route: str,
    scale: bool,
    meta_cv: Optional[np.ndarray],
    include_censored_in_train: bool = False,
) -> List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]:
    """Build strictly fold-internal (X_tr, X_va) ONCE for a (route, scale) across all 5 folds."""
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    task_type = spec["task_type"]
    valid_cv = _track_valid_mask(bank, cidx_cv, track)

    cache: List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]] = []
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
        pre = FoldPreprocessor(
            track=track,
            route=route,
            cond_cols=ep_data["cond_cols"],
            scale_for_linear=scale,
            svd_dim=32,
            min_fp_support_frac=0.01 if n_cv >= 1000 else 0.005,
            random_state=42 + f,
        )
        m_tr = meta_cv[tr_pos] if meta_cv is not None else None
        m_va = meta_cv[va_pos] if meta_cv is not None else None
        X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=m_tr)
        X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=m_va)
        cache.append((f, tr_pos, va_pos, X_tr, X_va))
    return cache


def evaluate_recipe_cv(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    recipe: Dict[str, Any],
    n_jobs: int,
    meta_cv: Optional[np.ndarray] = None,
    return_folds: bool = False,
    fold_cache: Optional[List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]] = None,
) -> Dict[str, Any]:
    """Strictly leak-free 5-fold grouped CV evaluation of ONE (route, model_family, hyperparams)."""
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

    valid_cv = _track_valid_mask(bank, cidx_cv, track)
    oof_pred = np.full(n_cv, np.nan, dtype=np.float64)
    oof_sd = np.full(n_cv, np.nan, dtype=np.float64)
    has_sd = False
    fitted_folds: List[Tuple[int, FoldPreprocessor, Any]] = []

    for f in range(5):
        if fold_cache is not None and not return_folds:
            entry = fold_cache[f]
            if entry is None:
                continue
            _, tr_pos, va_pos, X_tr, X_va = entry
            pre = None
        else:
            va_mask = (fids_cv == f) & valid_cv
            tr_mask = (fids_cv != f) & valid_cv
            if task_type == "regression":
                tr_mask = tr_mask & (ct_cv == "exact")
            if tr_mask.sum() < 4 or va_mask.sum() == 0:
                continue
            tr_pos = np.where(tr_mask)[0]
            va_pos = np.where(va_mask)[0]
            pre = FoldPreprocessor(
                track=track,
                route=recipe["route"],
                cond_cols=ep_data["cond_cols"],
                scale_for_linear=recipe["scale"],
                svd_dim=32,
                min_fp_support_frac=0.01 if n_cv >= 1000 else 0.005,
                random_state=42 + f,
            )
            m_tr = meta_cv[tr_pos] if meta_cv is not None else None
            m_va = meta_cv[va_pos] if meta_cv is not None else None
            X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=m_tr)
            X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=m_va)

        y_tr = y_cv[tr_pos].astype(np.int32) if task_type == "classification" else y_cv[tr_pos]
        if task_type == "classification" and len(np.unique(y_tr)) < 2:
            continue

        est = make_estimator(
            family=recipe["family"],
            params=recipe["params"],
            task_type=task_type,
            n_jobs=n_jobs,
            bounds=bounds if task_type == "regression" else None,
            winsor_cap=recipe.get("winsor_cap"),
            seed=42 + f,
        )
        est.fit(X_tr, y_tr)
        p_va, sd_va = _predict_single(est, X_va, task_type)
        oof_pred[va_pos] = p_va
        if sd_va is not None:
            oof_sd[va_pos] = sd_va
            has_sd = True
        if return_folds:
            assert pre is not None
            fitted_folds.append((f, pre, est))

    if task_type == "classification":
        m = classification_metrics(y_cv, oof_pred)
        # Primary selection score: PR-AUC (AP), tie-break ROC-AUC - 0.1 * Brier
        primary = (m["pr_auc"] if m["pr_auc"] is not None else -1.0)
        tie = (m["roc_auc"] if m["roc_auc"] is not None else -1.0) - 0.1 * (m["brier"] if m["brier"] is not None else 1.0)
        score_tuple = (primary, tie)
    else:
        m = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
        # Primary selection score: minimize exact-row CV MAE (maximize -MAE), tie-break Spearman
        mae = m["mae"] if m["mae"] is not None else 1e9
        sp = m["spearman"] if m["spearman"] is not None else -1.0
        score_tuple = (-mae, sp)

    out: Dict[str, Any] = {
        "recipe": recipe,
        "metrics": m,
        "score_tuple": score_tuple,
        "oof_pred": oof_pred,
        "oof_sd": oof_sd if has_sd else None,
    }
    if return_folds:
        out["folds"] = fitted_folds
    return out


def evaluate_aft_cv(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    aft_cfg: Dict[str, Any],
    n_jobs: int,
    meta_cv: Optional[np.ndarray] = None,
    return_folds: bool = False,
    fold_cache: Optional[List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]] = None,
) -> Dict[str, Any]:
    """Strictly leak-free 5-fold CV evaluation of XGBoost survival:aft on ALL (exact + censored) rows."""
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    bounds = spec.get("bounds")

    valid_cv = _track_valid_mask(bank, cidx_cv, track)
    oof_pred = np.full(n_cv, np.nan, dtype=np.float64)
    fitted_folds: List[Tuple[int, FoldPreprocessor, XGBoostAFTRegressor]] = []

    for f in range(5):
        if fold_cache is not None and not return_folds:
            entry = fold_cache[f]
            if entry is None:
                continue
            _, tr_pos, va_pos, X_tr, X_va = entry
            pre = None
        else:
            va_mask = (fids_cv == f) & valid_cv
            tr_mask = (fids_cv != f) & valid_cv  # includes BOTH exact and censored rows!
            if tr_mask.sum() < 6 or va_mask.sum() == 0:
                continue
            tr_pos = np.where(tr_mask)[0]
            va_pos = np.where(va_mask)[0]
            pre = FoldPreprocessor(
                track=track,
                route=aft_cfg["route"],
                cond_cols=ep_data["cond_cols"],
                scale_for_linear=False,
                svd_dim=32,
                min_fp_support_frac=0.01 if n_cv >= 1000 else 0.005,
                random_state=42 + f,
            )
            m_tr = meta_cv[tr_pos] if meta_cv is not None else None
            m_va = meta_cv[va_pos] if meta_cv is not None else None
            X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=m_tr)
            X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=m_va)

        p = dict(aft_cfg["params"])
        est = XGBoostAFTRegressor(
            **p,
            nthread=n_jobs,
            seed=42 + f,
            bounds=bounds,
        )
        est.fit(X_tr, y_cv[tr_pos], yl_cv[tr_pos], yu_cv[tr_pos], ct_cv[tr_pos])
        oof_pred[va_pos] = est.predict(X_va)
        if return_folds:
            assert pre is not None
            fitted_folds.append((f, pre, est))

    m = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
    # Selection for AFT balances exact accuracy + censored bound adherence via all_row_hinge_mae!
    hinge = m["all_row_hinge_mae"] if m["all_row_hinge_mae"] is not None else 1e9
    conc = m["censored_concordance"] if m["censored_concordance"] is not None else 0.0
    out: Dict[str, Any] = {
        "recipe": {"route": aft_cfg["route"], "family": "XGBoost_AFT", "scale": False, "params": aft_cfg["params"]},
        "metrics": m,
        "score_tuple": (-hinge, conc),
        "oof_pred": oof_pred,
    }
    if return_folds:
        out["folds"] = fitted_folds
    return out


def predict_ensemble(
    folds: List[Tuple[int, FoldPreprocessor, Any]],
    bank: FeatureBank,
    compound_idx: np.ndarray,
    track: str,
    task_type: str,
    cond_rows: Optional[List[Dict[str, str]]] = None,
    default_cond: Optional[Dict[str, str]] = None,
    meta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Predict with a 5-fold ensemble, returning (mean_pred, fold_sd, gp_sd_if_available)."""
    n = len(compound_idx)
    valid = _track_valid_mask(bank, compound_idx, track)
    v_pos = np.where(valid)[0]
    mean_out = np.full(n, np.nan, dtype=np.float64)
    sd_out = np.full(n, np.nan, dtype=np.float64)
    gp_sd_out = np.full(n, np.nan, dtype=np.float64)
    has_gp_sd = False

    if len(v_pos) == 0 or len(folds) == 0:
        return mean_out, sd_out, None

    c_v = [cond_rows[i] for i in v_pos] if cond_rows is not None else None
    m_v = meta[v_pos] if meta is not None else None

    fold_preds = []
    gp_sds = []
    for _, pre, est in folds:
        X_v = pre.transform(bank, compound_idx[v_pos], cond_rows=c_v, default_cond=default_cond, meta=m_v)
        p, gsd = _predict_single(est, X_v, task_type)
        fold_preds.append(p)
        if gsd is not None:
            gp_sds.append(gsd)
            has_gp_sd = True

    P = np.stack(fold_preds, axis=0)  # (K, n_v)
    mean_out[v_pos] = np.mean(P, axis=0)
    sd_out[v_pos] = np.std(P, axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(len(v_pos))
    if has_gp_sd:
        gp_sd_out[v_pos] = np.mean(np.stack(gp_sds, axis=0), axis=0)
        return mean_out, sd_out, gp_sd_out
    return mean_out, sd_out, None


def compute_oof_tanimoto(bank: FeatureBank, ep_data: Dict[str, Any]) -> np.ndarray:
    """Strictly fold-internal max Tanimoto for every CV row against its 4 training folds (ZERO self-leakage!)."""
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    out = np.full(n_cv, np.nan, dtype=np.float32)
    for f in range(5):
        va = np.where(fids_cv == f)[0]
        tr = np.where(fids_cv != f)[0]
        if len(va) == 0 or len(tr) == 0:
            continue
        ad_f = TanimotoADIndex.from_indices(bank, cidx_cv[tr])
        max_t, _, _ = ad_f.query(bank, cidx_cv[va])
        out[va] = max_t
    return out


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


def fit_leave_case17_out_ensemble(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    recipe: Dict[str, Any],
    n_jobs: int,
    meta_non_c17: Optional[np.ndarray] = None,
    is_aft: bool = False,
) -> Tuple[List[Tuple[int, FoldPreprocessor, Any]], Dict[str, int], np.ndarray]:
    """Refit winning recipe strictly excluding ALL `case17_excluded == 1` rows across the entire endpoint."""
    keep_rows = np.where(ep_data["case17_excluded"] == 0)[0]
    cidx = ep_data["compound_idx"][keep_rows]
    clids = ep_data["cluster_ids"][keep_rows]
    y = ep_data["y_value"][keep_rows]
    yl = ep_data["y_lower"][keep_rows]
    yu = ep_data["y_upper"][keep_rows]
    ct = ep_data["censor_type"][keep_rows]
    conds = [ep_data["cond_rows"][i] for i in keep_rows]

    valid = _track_valid_mask(bank, cidx, track)
    fids = assign_balanced_cluster_folds(clids, n_folds=5)
    cluster_to_fold = {str(clids[i]): int(fids[i]) for i in range(len(keep_rows))}
    task_type = spec["task_type"]
    bounds = spec.get("bounds")

    folds_out: List[Tuple[int, FoldPreprocessor, Any]] = []
    for f in range(5):
        tr_mask = (fids != f) & valid
        if task_type == "regression" and not is_aft:
            tr_mask = tr_mask & (ct == "exact")
        if tr_mask.sum() < 4:
            continue
        tr_pos = np.where(tr_mask)[0]
        pre = FoldPreprocessor(
            track=track,
            route=recipe["route"],
            cond_cols=ep_data["cond_cols"],
            scale_for_linear=recipe.get("scale", False),
            svd_dim=32,
            min_fp_support_frac=0.01 if len(keep_rows) >= 1000 else 0.005,
            random_state=142 + f,
        )
        m_tr = meta_non_c17[tr_pos] if meta_non_c17 is not None else None
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos], meta=m_tr)

        if is_aft:
            est = XGBoostAFTRegressor(
                **recipe["params"],
                nthread=n_jobs,
                seed=142 + f,
                bounds=bounds,
            )
            est.fit(X_tr, y[tr_pos], yl[tr_pos], yu[tr_pos], ct[tr_pos])
        else:
            y_tr = y[tr_pos].astype(np.int32) if task_type == "classification" else y[tr_pos]
            if task_type == "classification" and len(np.unique(y_tr)) < 2:
                continue
            est = make_estimator(
                family=recipe["family"],
                params=recipe["params"],
                task_type=task_type,
                n_jobs=n_jobs,
                bounds=bounds if task_type == "regression" else None,
                winsor_cap=recipe.get("winsor_cap"),
                seed=142 + f,
            )
            est.fit(X_tr, y_tr)
        folds_out.append((f, pre, est))

    return folds_out, cluster_to_fold, cidx[valid]


def generate_strict_ooc_meta_features(
    bank: FeatureBank,
    meta_cache: Dict[str, Any],
    target_endpoint: str,
    track: str,
    compound_idx: np.ndarray,
    cluster_ids: np.ndarray,
    mode: str = "cv",
) -> Tuple[np.ndarray, List[str], int]:
    """Generate strictly out-of-cluster (ZERO cluster leakage!) predictions from the 5 auxiliary endpoints.

    For each auxiliary endpoint A != target_endpoint:
      - If cluster_id[i] is in A's training cluster map (fold f), predict using ONLY A's fold-f model
        (which was trained strictly WITHOUT cluster_id[i]!).
      - Otherwise (cluster_id[i] never appeared in A's training set), average all 5 fold models of A!
      - Hard-asserts `cluster_id[i] not in train_clusters(model)` for EVERY single row and aux endpoint!
    """
    aux_list = [a for a in AUX_ENDPOINTS if a != target_endpoint]
    n = len(compound_idx)
    M = np.zeros((n, len(aux_list)), dtype=np.float32)
    n_assertions = 0

    for j, aux_ep in enumerate(aux_list):
        bundle = meta_cache[aux_ep][track]
        if mode == "cv":
            folds = bundle["cv_folds"]
            cl_to_fold = bundle["cv_cluster_to_fold"]
            fold_train_clusters = bundle["cv_fold_train_clusters"]
        else:
            folds = bundle["c17_folds"]
            cl_to_fold = bundle["c17_cluster_to_fold"]
            fold_train_clusters = bundle["c17_fold_train_clusters"]
            # If querying Case17 drugs directly, assert zero overlap with aux leave-Case17-out train set
            if len(cluster_ids) > 0 and str(cluster_ids[0]).startswith("CASE17_QUERY::"):
                aux_cidx_set = bundle.get("c17_train_cidx_set", set())
                for cidx_val in compound_idx:
                    assert int(cidx_val) not in aux_cidx_set, (
                        f"CASE17 LEAKAGE: compound_idx {int(cidx_val)} found in aux {aux_ep} leave-Case17-out train set!"
                    )
                    n_assertions += 1

        default_cond = ENDPOINT_SPECS[aux_ep]["canonical_conditions"]
        task_type = ENDPOINT_SPECS[aux_ep]["task_type"]
        cal = bundle.get("calibrator")

        # Precompute all 5 fold predictions for the batch at once (fast vectorized!)
        fold_map = {f: (pre, est) for f, pre, est in folds}
        fold_preds: Dict[int, np.ndarray] = {}
        for f, (pre, est) in fold_map.items():
            X_q = pre.transform(bank, compound_idx, default_cond=default_cond)
            p, _ = _predict_single(est, X_q, task_type)
            if task_type == "classification" and cal is not None:
                p = cal.transform(p)
            fold_preds[f] = p

        all_f = sorted(fold_preds.keys())
        mean_all = np.mean(np.stack([fold_preds[f] for f in all_f], axis=0), axis=0)

        col_pred = np.empty(n, dtype=np.float64)
        for i in range(n):
            cid = str(cluster_ids[i])
            if cid in cl_to_fold:
                f_ooc = cl_to_fold[cid]
                assert cid not in fold_train_clusters[f_ooc], (
                    f"CLUSTER LEAKAGE DETECTED: {cid} in aux {aux_ep} fold {f_ooc}!"
                )
                col_pred[i] = fold_preds[f_ooc][i]
                n_assertions += 1
            else:
                for f_chk in all_f:
                    assert cid not in fold_train_clusters[f_chk], (
                        f"CLUSTER LEAKAGE DETECTED: {cid} in aux {aux_ep} fold {f_chk}!"
                    )
                    n_assertions += 1
                col_pred[i] = mean_all[i]

        M[:, j] = col_pred.astype(np.float32)

    return M, aux_list, n_assertions


def select_best_recipe(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    n_jobs: int,
    meta_cv: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Benchmark all candidate recipes strictly on 5-fold CV and return (best_with_folds, leaderboard)."""
    n_cv = len(ep_data["cv_idx"])
    candidates = get_candidate_recipes(
        size_tier=spec["size_tier"],
        task_type=spec["task_type"],
        track=track,
        n_cv=n_cv,
        outlier_winsor_cap=spec.get("outlier_winsor_cap"),
    )

    # Group candidates by (route, scale) so each fold preprocessing (SVD / FP filter / scaling) runs ONCE!
    grouped: Dict[Tuple[str, bool], List[Dict[str, Any]]] = {}
    for cand in candidates:
        grouped.setdefault((cand["route"], cand["scale"]), []).append(cand)

    results = []
    best_res = None
    for (route, scale), group_cands in grouped.items():
        f_cache = _build_fold_cache(
            bank, ep_data, spec, track, route=route, scale=scale, meta_cv=meta_cv, include_censored_in_train=False
        )
        for cand in group_cands:
            try:
                r = evaluate_recipe_cv(
                    bank, ep_data, spec, track, cand, n_jobs=n_jobs, meta_cv=meta_cv, return_folds=False, fold_cache=f_cache
                )
                results.append(r)
                if best_res is None or r["score_tuple"] > best_res["score_tuple"]:
                    best_res = r
            except Exception as exc:
                results.append({"recipe": cand, "error": str(exc), "score_tuple": (-1e9, -1e9), "metrics": {}})
        del f_cache

    assert best_res is not None, f"{ep_data['endpoint']} ({track}): all candidates failed!"
    # Refit winning recipe once with return_folds=True
    winner = evaluate_recipe_cv(
        bank, ep_data, spec, track, best_res["recipe"], n_jobs=n_jobs, meta_cv=meta_cv, return_folds=True
    )

    # Compact leaderboard (top 8 recipes)
    sorted_res = sorted([r for r in results if "error" not in r], key=lambda x: x["score_tuple"], reverse=True)
    lb = []
    for r in sorted_res[:8]:
        rec = r["recipe"]
        lb.append({
            "route": rec["route"],
            "family": rec["family"],
            "winsor_cap": rec.get("winsor_cap"),
            "params": rec["params"],
            "cv_metrics": r["metrics"],
        })
    return winner, lb


def run_single_endpoint(args: Tuple[str, int, bool]) -> Dict[str, Any]:
    """Full end-to-end modeling, ablation, calibration, conformal, AD, and Case17 pipeline for ONE endpoint."""
    endpoint, n_jobs, allow_meta = args
    t0 = time.time()
    bank = GLOBAL_BANK
    assert bank is not None
    spec = ENDPOINT_SPECS[endpoint]
    ep_data = load_endpoint_data(endpoint, bank)
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv = len(cv_rows)
    n_ho = len(ho_rows)

    print(f"[START] {endpoint} (n={ep_data['n']}, cv={n_cv}, ho={n_ho}, jobs={n_jobs}, meta={allow_meta})", flush=True)

    # 1. Track A: smiles_only & Track B: dual_modality (baseline without cross-endpoint meta)
    win_smi, lb_smi = select_best_recipe(bank, ep_data, spec, "smiles_only", n_jobs=n_jobs, meta_cv=None)
    win_dual, lb_dual = select_best_recipe(bank, ep_data, spec, "dual_modality", n_jobs=n_jobs, meta_cv=None)

    # 2. Leak-free Cross-Endpoint Meta-Stacking ablation for the 7 PK endpoints
    meta_ablation: Dict[str, Any] = {"tested": False}
    use_meta_smi = False
    use_meta_dual = False
    meta_aux_names: List[str] = []
    meta_cv_smi = meta_ho_smi = meta_c17_tr_smi = meta_c17_q_smi = None
    meta_cv_dual = meta_ho_dual = meta_c17_tr_dual = meta_c17_q_dual = None

    if allow_meta and endpoint in PK_META_TARGETS and GLOBAL_META_CACHE is not None:
        m_cv_s, aux_names, n_chk_s = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "smiles_only",
            ep_data["compound_idx"][cv_rows], ep_data["cluster_ids"][cv_rows], mode="cv"
        )
        m_ho_s, _, n_chk_sh = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "smiles_only",
            ep_data["compound_idx"][ho_rows], ep_data["cluster_ids"][ho_rows], mode="cv"
        )
        m_cv_d, _, n_chk_d = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "dual_modality",
            ep_data["compound_idx"][cv_rows], ep_data["cluster_ids"][cv_rows], mode="cv"
        )
        m_ho_d, _, n_chk_dh = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "dual_modality",
            ep_data["compound_idx"][ho_rows], ep_data["cluster_ids"][ho_rows], mode="cv"
        )

        win_smi_m, lb_smi_m = select_best_recipe(bank, ep_data, spec, "smiles_only", n_jobs=n_jobs, meta_cv=m_cv_s)
        win_dual_m, lb_dual_m = select_best_recipe(bank, ep_data, spec, "dual_modality", n_jobs=n_jobs, meta_cv=m_cv_d)

        base_mae_s = win_smi["metrics"]["mae"]
        base_sp_s = win_smi["metrics"]["spearman"]
        meta_mae_s = win_smi_m["metrics"]["mae"]
        base_mae_d = win_dual["metrics"]["mae"]
        base_sp_d = win_dual["metrics"]["spearman"]
        meta_mae_d = win_dual_m["metrics"]["mae"]

        # Adopt meta-stacking strictly if CV MAE improves!
        if meta_mae_s is not None and base_mae_s is not None and meta_mae_s < base_mae_s - 1e-4:
            use_meta_smi = True
            win_smi, lb_smi = win_smi_m, lb_smi_m
            meta_cv_smi, meta_ho_smi = m_cv_s, m_ho_s
        if meta_mae_d is not None and base_mae_d is not None and meta_mae_d < base_mae_d - 1e-4:
            use_meta_dual = True
            win_dual, lb_dual = win_dual_m, lb_dual_m
            meta_cv_dual, meta_ho_dual = m_cv_d, m_ho_d

        meta_aux_names = aux_names
        meta_ablation = {
            "tested": True,
            "aux_endpoints": aux_names,
            "zero_cluster_leakage_assertions": int(n_chk_s + n_chk_sh + n_chk_d + n_chk_dh),
            "smiles_only_no_meta_cv": {"mae": base_mae_s, "spearman": base_sp_s},
            "smiles_only_with_meta_cv": {"mae": meta_mae_s, "spearman": win_smi_m["metrics"]["spearman"]},
            "smiles_only_meta_adopted": use_meta_smi,
            "dual_no_meta_cv": {"mae": base_mae_d, "spearman": base_sp_d},
            "dual_with_meta_cv": {"mae": meta_mae_d, "spearman": win_dual_m["metrics"]["spearman"]},
            "dual_meta_adopted": use_meta_dual,
        }

    # 3. Track C: seq_only on the 12 sequence-heavy endpoints
    win_seq = None
    lb_seq: List[Dict[str, Any]] = []
    if endpoint in SEQ_ONLY_ENDPOINTS:
        win_seq, lb_seq = select_best_recipe(bank, ep_data, spec, "seq_only", n_jobs=n_jobs, meta_cv=None)

    # 4. MIC_S_aureus explicit Huber / winsorized outlier ablation summary
    mic_outlier_ablation: Optional[Dict[str, Any]] = None
    if spec.get("outlier_winsor_cap") is not None:
        cap = spec["outlier_winsor_cap"]
        ab_recipes = {
            "lgb_l2_raw": {"route": "physchem_fp_all", "family": "LightGBM", "scale": False, "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 5.0, "colsample_bytree": 0.6}},
            "lgb_huber_raw": {"route": "physchem_fp_all", "family": "LightGBM", "scale": False, "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 5.0, "colsample_bytree": 0.6, "objective": "huber"}},
            "lgb_huber_winsor5": {"route": "physchem_fp_all", "family": "LightGBM", "scale": False, "winsor_cap": cap, "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 5.0, "colsample_bytree": 0.6, "objective": "huber"}},
            "xgb_l2_raw": {"route": "physchem_fp_all", "family": "XGBoost", "scale": False, "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6}},
            "xgb_huber_winsor5": {"route": "physchem_fp_all", "family": "XGBoost", "scale": False, "winsor_cap": cap, "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6, "objective": "reg:pseudohubererror"}},
        }
        mic_outlier_ablation = {}
        ho_cidx = ep_data["compound_idx"][ho_rows]
        ho_cond = [ep_data["cond_rows"][i] for i in ho_rows]
        for k, rec in ab_recipes.items():
            r_ab = evaluate_recipe_cv(bank, ep_data, spec, "dual_modality", rec, n_jobs=n_jobs, return_folds=True)
            p_ho, _, _ = predict_ensemble(r_ab["folds"], bank, ho_cidx, "dual_modality", "regression", cond_rows=ho_cond)
            m_ho = regression_metrics(
                ep_data["y_value"][ho_rows], p_ho, ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
            )
            mic_outlier_ablation[k] = {"cv": r_ab["metrics"], "holdout": m_ho}

    # 5. Censored-Aware Modeling Ablation (XGBoost survival:aft) on ALL endpoints with censored rows!
    n_censored_total = int((ep_data["censor_type"] != "exact").sum())
    n_censored_cv = int((ep_data["censor_type"][cv_rows] != "exact").sum())
    win_aft_dual = None
    win_aft_smi = None
    aft_ablation: Optional[Dict[str, Any]] = None

    if task_type == "regression" and n_censored_cv >= 1:
        aft_cands = get_aft_candidates(spec["size_tier"])
        aft_by_route: Dict[str, List[Dict[str, Any]]] = {}
        for ac in aft_cands:
            aft_by_route.setdefault(ac["route"], []).append(ac)
        best_aft_d = None
        for r_name, r_cands in aft_by_route.items():
            aft_cache = _build_fold_cache(
                bank, ep_data, spec, "dual_modality", route=r_name, scale=False, meta_cv=meta_cv_dual, include_censored_in_train=True
            )
            for ac in r_cands:
                r_a = evaluate_aft_cv(
                    bank, ep_data, spec, "dual_modality", ac, n_jobs=n_jobs, meta_cv=meta_cv_dual, return_folds=False, fold_cache=aft_cache
                )
                if best_aft_d is None or r_a["score_tuple"] > best_aft_d["score_tuple"]:
                    best_aft_d = r_a
            del aft_cache
        assert best_aft_d is not None
        best_ac_d = {"route": best_aft_d["recipe"]["route"], "params": best_aft_d["recipe"]["params"]}
        win_aft_dual = evaluate_aft_cv(
            bank, ep_data, spec, "dual_modality", best_ac_d, n_jobs=n_jobs, meta_cv=meta_cv_dual, return_folds=True
        )
        # Also fit smiles_only AFT with the same winning AFT config
        win_aft_smi = evaluate_aft_cv(
            bank, ep_data, spec, "smiles_only", best_ac_d, n_jobs=n_jobs, meta_cv=meta_cv_smi, return_folds=True
        )

        # Apples-to-apples Exact-Only XGBoost on the same route & tree depth for clean ablation
        xgb_exact_rec = {
            "route": best_ac_d["route"],
            "family": "XGBoost",
            "scale": False,
            "params": {
                "n_estimators": best_ac_d["params"]["n_estimators"],
                "learning_rate": best_ac_d["params"]["learning_rate"],
                "max_depth": best_ac_d["params"]["max_depth"],
                "min_child_weight": best_ac_d["params"]["min_child_weight"],
                "reg_lambda": best_ac_d["params"]["reg_lambda"],
                "colsample_bytree": best_ac_d["params"].get("colsample_bytree", 0.7),
            },
        }
        xgb_exact_res = evaluate_recipe_cv(
            bank, ep_data, spec, "dual_modality", xgb_exact_rec, n_jobs=n_jobs, meta_cv=meta_cv_dual, return_folds=True
        )

        ho_cidx = ep_data["compound_idx"][ho_rows]
        ho_cond = [ep_data["cond_rows"][i] for i in ho_rows]
        p_ho_xgb_ex, _, _ = predict_ensemble(
            xgb_exact_res["folds"], bank, ho_cidx, "dual_modality", "regression", cond_rows=ho_cond, meta=meta_ho_dual
        )
        m_ho_xgb_ex = regression_metrics(
            ep_data["y_value"][ho_rows], p_ho_xgb_ex, ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
        )
        p_ho_aft_d, sd_ho_aft_d, _ = predict_ensemble(
            win_aft_dual["folds"], bank, ho_cidx, "dual_modality", "regression", cond_rows=ho_cond, meta=meta_ho_dual
        )
        m_ho_aft_d = regression_metrics(
            ep_data["y_value"][ho_rows], p_ho_aft_d, ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
        )
        p_ho_aft_s, _, _ = predict_ensemble(
            win_aft_smi["folds"], bank, ho_cidx, "smiles_only", "regression", cond_rows=ho_cond, meta=meta_ho_smi
        )
        m_ho_aft_s = regression_metrics(
            ep_data["y_value"][ho_rows], p_ho_aft_s, ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
        )

        aft_ablation = {
            "n_censored_total": n_censored_total,
            "n_censored_cv": n_censored_cv,
            "n_censored_holdout": int((ep_data["censor_type"][ho_rows] != "exact").sum()),
            "aft_winning_recipe": win_aft_dual["recipe"],
            "xgb_exact_only_cv": xgb_exact_res["metrics"],
            "xgb_exact_only_holdout": m_ho_xgb_ex,
            "xgb_aft_dual_cv": win_aft_dual["metrics"],
            "xgb_aft_dual_holdout": m_ho_aft_d,
            "xgb_aft_smiles_only_cv": win_aft_smi["metrics"],
            "xgb_aft_smiles_only_holdout": m_ho_aft_s,
        }
    else:
        p_ho_aft_d = p_ho_aft_s = None

    # 6. Evaluate winning 5-fold ensembles on the UNTOUCHED 20% Outer Holdout
    ho_cidx = ep_data["compound_idx"][ho_rows]
    ho_cond = [ep_data["cond_rows"][i] for i in ho_rows]
    y_cv = ep_data["y_value"][cv_rows]
    yl_cv = ep_data["y_lower"][cv_rows]
    yu_cv = ep_data["y_upper"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]

    y_ho = ep_data["y_value"][ho_rows]
    yl_ho = ep_data["y_lower"][ho_rows]
    yu_ho = ep_data["y_upper"][ho_rows]
    ct_ho = ep_data["censor_type"][ho_rows]

    p_ho_smi, sd_ho_smi, gpsd_ho_smi = predict_ensemble(
        win_smi["folds"], bank, ho_cidx, "smiles_only", task_type, cond_rows=ho_cond, meta=meta_ho_smi
    )
    p_ho_dual, sd_ho_dual, gpsd_ho_dual = predict_ensemble(
        win_dual["folds"], bank, ho_cidx, "dual_modality", task_type, cond_rows=ho_cond, meta=meta_ho_dual
    )
    p_ho_seq = sd_ho_seq = None
    if win_seq is not None:
        p_ho_seq, sd_ho_seq, _ = predict_ensemble(
            win_seq["folds"], bank, ho_cidx, "seq_only", task_type, cond_rows=ho_cond
        )

    # 7. Applicability Domain (packed-bit Tanimoto) — strictly against CV training compounds
    ad_index = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][cv_rows])
    oof_max_t = compute_oof_tanimoto(bank, ep_data)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, ho_cidx)

    # 8. Calibration (classification) OR Conformal Prediction Intervals (regression)
    cal_smi = cal_dual = cal_seq = None
    conf_smi = conf_dual = None
    cal_summary: Dict[str, Any] = {}
    conf_summary: Dict[str, Any] = {}

    p_ho_smi_cal = p_ho_dual_cal = p_ho_seq_cal = None
    p_oof_smi_cal = p_oof_dual_cal = None
    ho_iv_dual = ho_iv_smi = None

    if task_type == "classification":
        m_ho_smi_raw = classification_metrics(y_ho, p_ho_smi)
        m_ho_dual_raw = classification_metrics(y_ho, p_ho_dual)
        m_ho_seq_raw = classification_metrics(y_ho, p_ho_seq) if p_ho_seq is not None else None

        cal_smi = ProbabilityCalibrator().fit(win_smi["oof_pred"], y_cv, fids_cv)
        cal_dual = ProbabilityCalibrator().fit(win_dual["oof_pred"], y_cv, fids_cv)
        p_ho_smi_cal = cal_smi.transform(p_ho_smi)
        p_ho_dual_cal = cal_dual.transform(p_ho_dual)
        p_oof_smi_cal = cal_smi.transform(win_smi["oof_pred"])
        p_oof_dual_cal = cal_dual.transform(win_dual["oof_pred"])

        m_ho_smi_cal = classification_metrics(y_ho, p_ho_smi_cal)
        m_ho_dual_cal = classification_metrics(y_ho, p_ho_dual_cal)

        if win_seq is not None:
            cal_seq = ProbabilityCalibrator().fit(win_seq["oof_pred"], y_cv, fids_cv)
            p_ho_seq_cal = cal_seq.transform(p_ho_seq)
            m_ho_seq_cal = classification_metrics(y_ho, p_ho_seq_cal)
        else:
            m_ho_seq_cal = None

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
        ho_metrics_smi = m_ho_smi_cal
        ho_metrics_dual = m_ho_dual_cal
        ho_metrics_seq = m_ho_seq_cal
        ho_metrics_smi_raw = m_ho_smi_raw
        ho_metrics_dual_raw = m_ho_dual_raw

        ad_strat_dual = compute_ad_stratification(y_ho, p_ho_dual_cal, ct_ho, ho_max_t, ho_ad_tiers, "classification")
    else:
        ho_metrics_smi = regression_metrics(y_ho, p_ho_smi, yl_ho, yu_ho, ct_ho)
        ho_metrics_dual = regression_metrics(y_ho, p_ho_dual, yl_ho, yu_ho, ct_ho)
        ho_metrics_seq = regression_metrics(y_ho, p_ho_seq, yl_ho, yu_ho, ct_ho) if p_ho_seq is not None else None
        ho_metrics_smi_raw = ho_metrics_smi
        ho_metrics_dual_raw = ho_metrics_dual

        conf_smi = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv, win_smi["oof_pred"], ct_cv, oof_sd=win_smi["oof_sd"], oof_max_tanimoto=oof_max_t
        )
        conf_dual = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv, win_dual["oof_pred"], ct_cv, oof_sd=win_dual["oof_sd"], oof_max_tanimoto=oof_max_t
        )
        ho_iv_smi = conf_smi.predict_intervals(p_ho_smi, model_sd=gpsd_ho_smi, max_tanimoto=ho_max_t)
        ho_iv_dual = conf_dual.predict_intervals(p_ho_dual, model_sd=gpsd_ho_dual, max_tanimoto=ho_max_t)

        conf_summary = {
            "smiles_only": {
                "cv_diagnostics": conf_smi.cv_diagnostics_,
                "holdout_coverage": conf_smi.evaluate_coverage(
                    y_ho, p_ho_smi, ct_ho, model_sd=gpsd_ho_smi, max_tanimoto=ho_max_t
                ),
            },
            "dual_modality": {
                "cv_diagnostics": conf_dual.cv_diagnostics_,
                "holdout_coverage": conf_dual.evaluate_coverage(
                    y_ho, p_ho_dual, ct_ho, model_sd=gpsd_ho_dual, max_tanimoto=ho_max_t
                ),
            },
        }
        ad_strat_dual = compute_ad_stratification(y_ho, p_ho_dual, ct_ho, ho_max_t, ho_ad_tiers, "regression")

    ad_index.tier_calibration = ad_strat_dual

    # 9. Strict Zero-Leakage Case17 Leave-Out Refit & Evaluation (all 17 clinical drugs!)
    c17_idx = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)
    c17_names = [q["drug_name"] for q in bank.case17_queries]
    # Assign unique sentinel cluster IDs for Case17 drugs so strict OOC meta assertions verify zero overlap!
    c17_clusters = np.array([f"CASE17_QUERY::{name}" for name in c17_names], dtype=object)

    keep_non_c17 = np.where(ep_data["case17_excluded"] == 0)[0]
    if use_meta_smi and GLOBAL_META_CACHE is not None:
        meta_c17_tr_smi, _, _ = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "smiles_only",
            ep_data["compound_idx"][keep_non_c17], ep_data["cluster_ids"][keep_non_c17], mode="case17"
        )
        meta_c17_q_smi, _, _ = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "smiles_only", c17_idx, c17_clusters, mode="case17"
        )
    if use_meta_dual and GLOBAL_META_CACHE is not None:
        meta_c17_tr_dual, _, _ = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "dual_modality",
            ep_data["compound_idx"][keep_non_c17], ep_data["cluster_ids"][keep_non_c17], mode="case17"
        )
        meta_c17_q_dual, _, _ = generate_strict_ooc_meta_features(
            bank, GLOBAL_META_CACHE, endpoint, "dual_modality", c17_idx, c17_clusters, mode="case17"
        )

    c17_folds_smi, c17_cl_map_smi, c17_tr_cidx_smi = fit_leave_case17_out_ensemble(
        bank, ep_data, spec, "smiles_only", win_smi["recipe"], n_jobs=n_jobs, meta_non_c17=meta_c17_tr_smi, is_aft=False
    )
    c17_folds_dual, c17_cl_map_dual, c17_tr_cidx_dual = fit_leave_case17_out_ensemble(
        bank, ep_data, spec, "dual_modality", win_dual["recipe"], n_jobs=n_jobs, meta_non_c17=meta_c17_tr_dual, is_aft=False
    )
    c17_folds_aft = None
    if win_aft_dual is not None:
        c17_folds_aft, _, _ = fit_leave_case17_out_ensemble(
            bank, ep_data, spec, "dual_modality", win_aft_dual["recipe"], n_jobs=n_jobs, meta_non_c17=meta_c17_tr_dual, is_aft=True
        )
    c17_folds_seq = None
    if win_seq is not None:
        c17_folds_seq, _, _ = fit_leave_case17_out_ensemble(
            bank, ep_data, spec, "seq_only", win_seq["recipe"], n_jobs=n_jobs, meta_non_c17=None, is_aft=False
        )

    canon_cond = spec["canonical_conditions"]
    c17_p_smi, c17_sd_smi, c17_gpsd_smi = predict_ensemble(
        c17_folds_smi, bank, c17_idx, "smiles_only", task_type, default_cond=canon_cond, meta=meta_c17_q_smi
    )
    c17_p_dual, c17_sd_dual, c17_gpsd_dual = predict_ensemble(
        c17_folds_dual, bank, c17_idx, "dual_modality", task_type, default_cond=canon_cond, meta=meta_c17_q_dual
    )
    c17_p_aft = None
    if c17_folds_aft is not None:
        c17_p_aft, _, _ = predict_ensemble(
            c17_folds_aft, bank, c17_idx, "dual_modality", "regression", default_cond=canon_cond, meta=meta_c17_q_dual
        )
    c17_p_seq = None
    if c17_folds_seq is not None:
        c17_p_seq, _, _ = predict_ensemble(
            c17_folds_seq, bank, c17_idx, "seq_only", task_type, default_cond=canon_cond
        )

    # Alternate clinical route predictions for F (SC vs PO) and half_life_reg (SC vs IV)
    c17_alt_preds: Dict[str, Dict[str, np.ndarray]] = {}
    for alt_name, alt_cond in spec.get("alt_case17_conditions", {}).items():
        ap_s, _, _ = predict_ensemble(
            c17_folds_smi, bank, c17_idx, "smiles_only", task_type, default_cond=alt_cond, meta=meta_c17_q_smi
        )
        ap_d, _, _ = predict_ensemble(
            c17_folds_dual, bank, c17_idx, "dual_modality", task_type, default_cond=alt_cond, meta=meta_c17_q_dual
        )
        ap_a = None
        if c17_folds_aft is not None:
            ap_a, _, _ = predict_ensemble(
                c17_folds_aft, bank, c17_idx, "dual_modality", "regression", default_cond=alt_cond, meta=meta_c17_q_dual
            )
        c17_alt_preds[alt_name] = {"smiles_only": ap_s, "dual": ap_d, "aft": ap_a}

    # AD for Case17 strictly against non-Case17 training set
    ad_c17 = TanimotoADIndex.from_indices(bank, c17_tr_cidx_dual)
    c17_max_t, c17_top5_t, c17_ad_tiers = ad_c17.query(bank, c17_idx)

    c17_p_smi_cal = cal_smi.transform(c17_p_smi) if cal_smi is not None else None
    c17_p_dual_cal = cal_dual.transform(c17_p_dual) if cal_dual is not None else None
    c17_iv_dual = (
        conf_dual.predict_intervals(c17_p_dual, model_sd=c17_gpsd_dual, max_tanimoto=c17_max_t)
        if conf_dual is not None
        else None
    )

    # 10. Save ALL standardized per-endpoint outputs
    ep_out_dir = OUT_DIR / endpoint
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    # (a) oof_predictions.tsv
    oof_cols = [
        "record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual",
    ]
    if win_aft_dual is not None:
        oof_cols.extend(["pred_aft", "pred_aft_smiles_only"])
    oof_cols.append("fold_sd")
    if win_seq is not None:
        oof_cols.append("pred_seq_only")
    if task_type == "classification":
        oof_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    oof_cols.append("max_tanimoto_oof")

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.6f}"
        return str(v)

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
                _fmt(win_smi["oof_pred"][loc_i]),
                _fmt(win_dual["oof_pred"][loc_i]),
            ]
            if win_aft_dual is not None:
                row.append(_fmt(win_aft_dual["oof_pred"][loc_i]))
                row.append(_fmt(win_aft_smi["oof_pred"][loc_i]))
            sd_val = win_dual["oof_sd"][loc_i] if win_dual["oof_sd"] is not None else np.nan
            row.append(_fmt(sd_val))
            if win_seq is not None:
                row.append(_fmt(win_seq["oof_pred"][loc_i]))
            if task_type == "classification":
                row.append(_fmt(p_oof_smi_cal[loc_i]))
                row.append(_fmt(p_oof_dual_cal[loc_i]))
            row.append(_fmt(oof_max_t[loc_i]))
            w.writerow(row)

    # (b) holdout_predictions.tsv
    ho_cols = [
        "record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual",
    ]
    if p_ho_aft_d is not None:
        ho_cols.extend(["pred_aft", "pred_aft_smiles_only"])
    ho_cols.extend(["fold_sd", "fold_sd_smiles_only"])
    if p_ho_seq is not None:
        ho_cols.append("pred_seq_only")
    if task_type == "classification":
        ho_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
        if p_ho_seq_cal is not None:
            ho_cols.append("pred_seq_only_cal")
    else:
        ho_cols.extend(["lo80", "hi80", "lo90", "hi90", "lo80_smiles", "hi80_smiles", "lo90_smiles", "hi90_smiles"])
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
                _fmt(p_ho_smi[loc_i]),
                _fmt(p_ho_dual[loc_i]),
            ]
            if p_ho_aft_d is not None:
                row.append(_fmt(p_ho_aft_d[loc_i]))
                row.append(_fmt(p_ho_aft_s[loc_i]))
            row.append(_fmt(sd_ho_dual[loc_i]))
            row.append(_fmt(sd_ho_smi[loc_i]))
            if p_ho_seq is not None:
                row.append(_fmt(p_ho_seq[loc_i]))
            if task_type == "classification":
                row.append(_fmt(p_ho_smi_cal[loc_i]))
                row.append(_fmt(p_ho_dual_cal[loc_i]))
                if p_ho_seq_cal is not None:
                    row.append(_fmt(p_ho_seq_cal[loc_i]))
            else:
                row.extend([
                    _fmt(ho_iv_dual["lo80"][loc_i]), _fmt(ho_iv_dual["hi80"][loc_i]),
                    _fmt(ho_iv_dual["lo90"][loc_i]), _fmt(ho_iv_dual["hi90"][loc_i]),
                    _fmt(ho_iv_smi["lo80"][loc_i]), _fmt(ho_iv_smi["hi80"][loc_i]),
                    _fmt(ho_iv_smi["lo90"][loc_i]), _fmt(ho_iv_smi["hi90"][loc_i]),
                ])
            row.extend([_fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i]])
            w.writerow(row)

    # (c) case17_predictions.tsv
    c17_cols = [
        "drug_name", "compound_key", "compound_idx", "clinical_route", "condition_used",
        "pred_smiles_only", "pred_dual",
    ]
    if c17_p_aft is not None:
        c17_cols.append("pred_aft")
    if c17_p_seq is not None:
        c17_cols.append("pred_seq_only")
    if task_type == "classification":
        c17_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    c17_cols.append("fold_sd")
    if c17_iv_dual is not None:
        c17_cols.extend(["lo80", "hi80", "lo90", "hi90"])
    c17_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])
    for alt_name in c17_alt_preds:
        c17_cols.extend([f"pred_smiles_only_{alt_name}", f"pred_dual_{alt_name}"])
        if c17_alt_preds[alt_name]["aft"] is not None:
            c17_cols.append(f"pred_aft_{alt_name}")
    if endpoint == "half_life_reg":
        c17_cols.extend(["anchor_class", "anchor_mid_h", "anchor_range_h"])

    case17_records: List[Dict[str, Any]] = []
    cond_str = json.dumps(canon_cond, ensure_ascii=False)
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
            if c17_p_seq is not None:
                row.append(_fmt(c17_p_seq[i]))
                rec["pred_seq_only"] = round(float(c17_p_seq[i]), 5) if np.isfinite(c17_p_seq[i]) else None
            if task_type == "classification":
                row.append(_fmt(c17_p_smi_cal[i]))
                row.append(_fmt(c17_p_dual_cal[i]))
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
            for alt_name, ap_dict in c17_alt_preds.items():
                row.append(_fmt(ap_dict["smiles_only"][i]))
                row.append(_fmt(ap_dict["dual"][i]))
                rec[f"pred_smiles_only_{alt_name}"] = round(float(ap_dict["smiles_only"][i]), 5)
                rec[f"pred_dual_{alt_name}"] = round(float(ap_dict["dual"][i]), 5)
                if ap_dict["aft"] is not None:
                    row.append(_fmt(ap_dict["aft"][i]))
                    rec[f"pred_aft_{alt_name}"] = round(float(ap_dict["aft"][i]), 5)
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
    censor_counts = {
        k: int((ep_data["censor_type"] == k).sum())
        for k in ("exact", "right", "left", "interval")
    }
    elapsed = round(time.time() - t0, 2)
    metrics_doc: Dict[str, Any] = {
        "endpoint": endpoint,
        "tier": spec["tier"],
        "task_type": task_type,
        "size_tier": spec["size_tier"],
        "unit": spec["unit"],
        "bounds": list(bounds) if bounds is not None else None,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": canon_cond,
        "n_total": ep_data["n"],
        "n_cv": n_cv,
        "n_holdout": n_ho,
        "n_case17_excluded": int((ep_data["case17_excluded"] == 1).sum()),
        "n_train_leave_case17_out": int(len(keep_non_c17)),
        "censor_counts": censor_counts,
        "smiles_valid_pct": round(100.0 * float(bank.smiles_valid[ep_data["compound_idx"]].mean()), 2),
        "seq_valid_pct": round(100.0 * float(bank.seq_valid[ep_data["compound_idx"]].mean()), 2),
        "winning_recipe_smiles_only": {**win_smi["recipe"], "uses_meta": use_meta_smi, "meta_endpoints": meta_aux_names if use_meta_smi else []},
        "winning_recipe_dual": {**win_dual["recipe"], "uses_meta": use_meta_dual, "meta_endpoints": meta_aux_names if use_meta_dual else []},
        "winning_recipe_seq_only": win_seq["recipe"] if win_seq is not None else None,
        "cv_metrics": {
            "smiles_only": win_smi["metrics"],
            "dual_modality": win_dual["metrics"],
            "seq_only": win_seq["metrics"] if win_seq is not None else None,
        },
        "holdout_metrics": {
            "smiles_only": ho_metrics_smi,
            "dual_modality": ho_metrics_dual,
            "seq_only": ho_metrics_seq,
            "smiles_only_raw": ho_metrics_smi_raw if task_type == "classification" else None,
            "dual_modality_raw": ho_metrics_dual_raw if task_type == "classification" else None,
        },
        "meta_stacking_ablation": meta_ablation,
        "mic_outlier_ablation": mic_outlier_ablation,
        "aft_ablation": aft_ablation,
        "calibration": cal_summary if task_type == "classification" else None,
        "conformal": conf_summary if task_type == "regression" else None,
        "applicability_domain": ad_strat_dual,
        "case17_leaveout_predictions": case17_records,
        "selection_leaderboard_smiles_only": lb_smi,
        "selection_leaderboard_dual": lb_dual,
        "elapsed_sec": elapsed,
    }
    (ep_out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # (e) models_5fold.joblib (compressed to keep NFS footprint compact)
    model_bundle: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v33_cpu_5fold_v1",
        "endpoint": endpoint,
        "task_type": task_type,
        "unit": spec["unit"],
        "bounds": bounds,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": canon_cond,
        "smiles_only": {
            "recipe": win_smi["recipe"],
            "uses_meta": use_meta_smi,
            "meta_endpoints": meta_aux_names if use_meta_smi else [],
            "folds": win_smi["folds"],
            "calibrator": cal_smi,
            "conformal": conf_smi,
        },
        "dual_modality": {
            "recipe": win_dual["recipe"],
            "uses_meta": use_meta_dual,
            "meta_endpoints": meta_aux_names if use_meta_dual else [],
            "folds": win_dual["folds"],
            "calibrator": cal_dual,
            "conformal": conf_dual,
        },
        "seq_only": (
            {"recipe": win_seq["recipe"], "folds": win_seq["folds"], "calibrator": cal_seq}
            if win_seq is not None
            else None
        ),
        "aft": (
            {
                "recipe": win_aft_dual["recipe"],
                "folds_dual": win_aft_dual["folds"],
                "folds_smiles_only": win_aft_smi["folds"],
            }
            if win_aft_dual is not None
            else None
        ),
        "ad_index": ad_index,
    }
    joblib.dump(model_bundle, ep_out_dir / "models_5fold.joblib", compress=3)

    # Compact status log
    if task_type == "classification":
        s_str = f"dual HO ROC={ho_metrics_dual['roc_auc']:.4f} PR={ho_metrics_dual['pr_auc']:.4f} Brier={ho_metrics_dual_raw['brier']:.4f}->{ho_metrics_dual['brier']:.4f} ECE={ho_metrics_dual_raw['ece']:.4f}->{ho_metrics_dual['ece']:.4f}"
    else:
        s_str = f"dual HO MAE={ho_metrics_dual['mae']:.4f} Sp={ho_metrics_dual['spearman']:.4f} cov80/90={conf_summary['dual_modality']['holdout_coverage']['cov80']}/{conf_summary['dual_modality']['holdout_coverage']['cov90']}"
    print(f"[DONE ] {endpoint:<24} ({elapsed:5.1f}s) | {win_dual['recipe']['route']}+{win_dual['recipe']['family']} | {s_str}", flush=True)

    # If auxiliary endpoint, return lightweight meta-cache info for Stage 2!
    aux_cache_payload = None
    if endpoint in AUX_ENDPOINTS and not allow_meta:
        cv_cl = ep_data["cluster_ids"][cv_rows]
        cv_cl_to_fold = {str(cv_cl[i]): int(fids_cv[i]) for i in range(n_cv)}
        cv_fold_train_clusters = {
            f: set(str(cv_cl[i]) for i in range(n_cv) if fids_cv[i] != f) for f in range(5)
        }

        keep_cl = ep_data["cluster_ids"][keep_non_c17]
        c17_fids = assign_balanced_cluster_folds(keep_cl, n_folds=5)
        c17_fold_train_clusters = {
            f: set(str(keep_cl[i]) for i in range(len(keep_non_c17)) if c17_fids[i] != f) for f in range(5)
        }
        aux_cache_payload = {
            "smiles_only": {
                "cv_folds": win_smi["folds"],
                "cv_cluster_to_fold": cv_cl_to_fold,
                "cv_fold_train_clusters": cv_fold_train_clusters,
                "c17_folds": c17_folds_smi,
                "c17_cluster_to_fold": c17_cl_map_smi,
                "c17_fold_train_clusters": c17_fold_train_clusters,
                "c17_train_cidx_set": set(int(x) for x in c17_tr_cidx_smi),
                "calibrator": cal_smi,
            },
            "dual_modality": {
                "cv_folds": win_dual["folds"],
                "cv_cluster_to_fold": cv_cl_to_fold,
                "cv_fold_train_clusters": cv_fold_train_clusters,
                "c17_folds": c17_folds_dual,
                "c17_cluster_to_fold": c17_cl_map_dual,
                "c17_fold_train_clusters": c17_fold_train_clusters,
                "c17_train_cidx_set": set(int(x) for x in c17_tr_cidx_dual),
                "calibrator": cal_dual,
            },
        }

    return {"endpoint": endpoint, "metrics": metrics_doc, "aux_cache": aux_cache_payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Quick fork+pipeline test on PPB and BBB only")
    parser.add_argument("--endpoints", nargs="*", default=None, help="Optional subset of endpoints")
    parser.add_argument("--max-workers", type=int, default=10, help="Parallel worker processes in Stage 2")
    args = parser.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading frozen V33 FeatureBank into shared memory...", flush=True)
    global GLOBAL_BANK, GLOBAL_META_CACHE
    GLOBAL_BANK = FeatureBank()
    print(f"FeatureBank ready in {time.time() - t_all:.2f}s (46,598 compounds).", flush=True)

    ctx = mp.get_context("fork")

    if args.smoke_test:
        # Test 2 endpoints in parallel via fork (1 small-n censored + 1 classification)
        with ctx.Pool(processes=2) as pool:
            res = pool.map(run_single_endpoint, [("PPB", 6, False), ("BBB", 6, False)])
        print(f"Smoke test PASSED in {time.time() - t_all:.1f}s!", flush=True)
        return

    all_eps = args.endpoints if args.endpoints else list(ENDPOINT_SPECS.keys())

    # --- STAGE 1: 5 Auxiliary Endpoints (in parallel across ~54 threads) ---
    stage1_budget = {
        "hemolysis_cls": 16,
        "PAMPA": 14,
        "solubility_cls": 10,
        "Caco2": 8,
        "LogD7.4": 6,
    }
    stage1_tasks = [(ep, stage1_budget[ep], False) for ep in AUX_ENDPOINTS if ep in all_eps]
    summary_by_ep: Dict[str, Dict[str, Any]] = {}
    meta_cache: Dict[str, Any] = {}

    if stage1_tasks:
        print(f"\n=== STAGE 1: Running {len(stage1_tasks)} auxiliary endpoints in parallel ===", flush=True)
        with ctx.Pool(processes=len(stage1_tasks)) as pool:
            s1_res = pool.map(run_single_endpoint, stage1_tasks)
        for item in s1_res:
            ep = item["endpoint"]
            summary_by_ep[ep] = item["metrics"]
            if item["aux_cache"] is not None:
                meta_cache[ep] = item["aux_cache"]
        print(f"=== STAGE 1 COMPLETE in {time.time() - t_all:.1f}s ===\n", flush=True)

    GLOBAL_META_CACHE = meta_cache if len(meta_cache) == len(AUX_ENDPOINTS) else None

    # --- STAGE 2: Remaining 19 Endpoints (+ LogD7.4 meta-stacking upgrade check) ---
    # LPT ordering: largest endpoints first so all 64 cores stay balanced to the end!
    stage2_order = [
        ("overall_peptide_toxicity", 8),
        ("hemolysis_HC50", 8),
        ("half_life_cls", 6),
        ("half_life_reg", 6),
        ("cytotoxicity_cls", 6),
        ("MIC_E_coli", 6),
        ("MIC_S_aureus", 6),
        ("MIC_P_aeruginosa", 5),
        ("BBB", 5),
        ("MIC_C_albicans", 4),
        ("LogD7.4", 4),  # Re-run LogD7.4 WITH 4-endpoint meta-stacking ablation!
        ("solubility_reg", 4),
        ("CL_per_kg", 4),
        ("MDCK", 4),
        ("RRCK", 4),
        ("Vd_per_kg", 4),
        ("PPB", 4),
        ("F", 4),
        ("Cmax", 4),
        ("CL_absolute", 4),
    ]
    stage2_tasks = [
        (ep, nj, ep in PK_META_TARGETS)
        for ep, nj in stage2_order
        if ep in all_eps and (ep not in summary_by_ep or ep == "LogD7.4")
    ]

    if stage2_tasks:
        n_workers = min(args.max_workers, len(stage2_tasks))
        print(f"=== STAGE 2: Running {len(stage2_tasks)} endpoints across {n_workers} parallel workers ===", flush=True)
        with ctx.Pool(processes=n_workers) as pool:
            for item in pool.imap_unordered(run_single_endpoint, stage2_tasks):
                ep = item["endpoint"]
                summary_by_ep[ep] = item["metrics"]
        print(f"=== STAGE 2 COMPLETE in {time.time() - t_all:.1f}s ===\n", flush=True)

    # Save master cpu_benchmark_summary.json
    master = {
        "schema": "peptide_omnipanel_v33_cpu_benchmark_summary_v1",
        "total_endpoints": len(summary_by_ep),
        "total_elapsed_sec": round(time.time() - t_all, 2),
        "endpoints": {ep: summary_by_ep[ep] for ep in sorted(summary_by_ep.keys())},
    }
    master_path = OUT_DIR / "cpu_benchmark_summary.json"
    master_path.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ALL {len(summary_by_ep)} ENDPOINTS COMPLETE! Master summary saved to {master_path}", flush=True)


if __name__ == "__main__":
    main()
