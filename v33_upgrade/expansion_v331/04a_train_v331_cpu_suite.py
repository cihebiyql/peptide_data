#!/usr/bin/env python3
"""
Wave 5A — Step 04a (runs on node23, 64 CPU cores):
64-Core Parallel Multi-Family CPU Modeling, Censored XGBoost-AFT, Small-Sample GPR/KRR,
Conformal Uncertainty, AD, & Strict Zero-Leakage Case17 Suite for ALL 22 new V33.1 endpoints.

Imports exact picklable classes from <WORK_DIR>/v33_upgrade/models_cpu/ so every .joblib
unpickles cleanly inside v33_inference_service.py with ZERO new class dependencies!
Writes strictly under <WORK_DIR>/v33_upgrade/expansion_v331/models_cpu/.
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
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
EXP_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(EXP_SCRIPTS))

from v33_config import CASE17_CLINICAL_ROUTE  # noqa: E402
from v33_features import FoldPreprocessor, TanimotoADIndex  # noqa: E402
from v33_metrics import ConformalIntervalCalibrator, compute_ad_stratification, regression_metrics  # noqa: E402
from v33_models import XGBoostAFTRegressor, get_aft_candidates, get_candidate_recipes, make_estimator  # noqa: E402

from v331_config import (  # noqa: E402
    EXP_CPU_DIR,
    EXP_ENDPOINTS_DIR,
    V331_ENDPOINT_SPECS,
    V331_SEQ_ONLY_ENDPOINTS,
    V331FeatureBank,
)

csv.field_size_limit(10_000_000)

GLOBAL_BANK: Optional[V331FeatureBank] = None


def load_endpoint_data(endpoint: str, bank: V331FeatureBank) -> Dict[str, Any]:
    path = EXP_ENDPOINTS_DIR / f"{endpoint}.tsv"
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


def _track_valid_mask(bank: V331FeatureBank, compound_idx: np.ndarray, track: str) -> np.ndarray:
    if track == "smiles_only":
        return bank.smiles_valid[compound_idx].copy()
    if track == "seq_only":
        return bank.seq_valid[compound_idx].copy()
    return np.ones(len(compound_idx), dtype=bool)


def _predict_single(est: Any, X: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if hasattr(est, "predict_with_std"):
        m, s = est.predict_with_std(X)
        return m.astype(np.float64), s.astype(np.float64)
    return np.asarray(est.predict(X), dtype=np.float64), None


def _build_fold_cache(
    bank: V331FeatureBank,
    ep_data: Dict[str, Any],
    track: str,
    route: str,
    scale: bool,
    include_censored_in_train: bool = False,
) -> List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]:
    cv_rows = ep_data["cv_idx"]
    n_cv = len(cv_rows)
    cidx_cv = ep_data["compound_idx"][cv_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    ct_cv = ep_data["censor_type"][cv_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    valid_cv = _track_valid_mask(bank, cidx_cv, track)

    cache: List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]] = []
    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv
        if not include_censored_in_train:
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
        X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=None)
        X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=None)
        cache.append((f, tr_pos, va_pos, X_tr, X_va))
    return cache


def evaluate_recipe_cv(
    bank: V331FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    recipe: Dict[str, Any],
    n_jobs: int,
    return_folds: bool = False,
    fold_cache: Optional[List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]] = None,
) -> Dict[str, Any]:
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
            tr_mask = (fids_cv != f) & valid_cv & (ct_cv == "exact")
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
            X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=None)
            X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=None)

        est = make_estimator(
            family=recipe["family"],
            params=recipe["params"],
            task_type="regression",
            n_jobs=n_jobs,
            bounds=bounds,
            winsor_cap=recipe.get("winsor_cap"),
            seed=42 + f,
        )
        est.fit(X_tr, y_cv[tr_pos])
        p_va, sd_va = _predict_single(est, X_va)
        oof_pred[va_pos] = p_va
        if sd_va is not None:
            oof_sd[va_pos] = sd_va
            has_sd = True
        if return_folds:
            assert pre is not None
            fitted_folds.append((f, pre, est))

    m = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
    mae = m["mae"] if m["mae"] is not None else 1e9
    sp = m["spearman"] if m["spearman"] is not None else -1.0
    out: Dict[str, Any] = {
        "recipe": recipe,
        "metrics": m,
        "score_tuple": (-mae, sp),
        "oof_pred": oof_pred,
        "oof_sd": oof_sd if has_sd else None,
    }
    if return_folds:
        out["folds"] = fitted_folds
    return out


def evaluate_aft_cv(
    bank: V331FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    aft_cfg: Dict[str, Any],
    n_jobs: int,
    return_folds: bool = False,
    fold_cache: Optional[List[Optional[Tuple[int, np.ndarray, np.ndarray, Any, Any]]]] = None,
) -> Dict[str, Any]:
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
            tr_mask = (fids_cv != f) & valid_cv
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
            X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos], meta=None)
            X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos], meta=None)

        est = XGBoostAFTRegressor(**dict(aft_cfg["params"]), nthread=n_jobs, seed=42 + f, bounds=bounds)
        est.fit(X_tr, y_cv[tr_pos], yl_cv[tr_pos], yu_cv[tr_pos], ct_cv[tr_pos])
        oof_pred[va_pos] = est.predict(X_va)
        if return_folds:
            assert pre is not None
            fitted_folds.append((f, pre, est))

    m = regression_metrics(y_cv, oof_pred, yl_cv, yu_cv, ct_cv)
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
    bank: V331FeatureBank,
    compound_idx: np.ndarray,
    track: str,
    cond_rows: Optional[List[Dict[str, str]]] = None,
    default_cond: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
    fold_preds, gp_sds = [], []
    for _, pre, est in folds:
        X_v = pre.transform(bank, compound_idx[v_pos], cond_rows=c_v, default_cond=default_cond, meta=None)
        p, gsd = _predict_single(est, X_v)
        fold_preds.append(p)
        if gsd is not None:
            gp_sds.append(gsd)
            has_gp_sd = True
    P = np.stack(fold_preds, axis=0)
    mean_out[v_pos] = np.mean(P, axis=0)
    sd_out[v_pos] = np.std(P, axis=0, ddof=1) if P.shape[0] > 1 else np.zeros(len(v_pos))
    if has_gp_sd:
        gp_sd_out[v_pos] = np.mean(np.stack(gp_sds, axis=0), axis=0)
        return mean_out, sd_out, gp_sd_out
    return mean_out, sd_out, None


def compute_oof_tanimoto(bank: V331FeatureBank, ep_data: Dict[str, Any]) -> np.ndarray:
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
    bank: V331FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    recipe: Dict[str, Any],
    n_jobs: int,
    is_aft: bool = False,
) -> Tuple[List[Tuple[int, FoldPreprocessor, Any]], np.ndarray]:
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
    bounds = spec.get("bounds")

    folds_out: List[Tuple[int, FoldPreprocessor, Any]] = []
    for f in range(5):
        tr_mask = (fids != f) & valid
        if not is_aft:
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
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos], meta=None)
        if is_aft:
            est = XGBoostAFTRegressor(**recipe["params"], nthread=n_jobs, seed=142 + f, bounds=bounds)
            est.fit(X_tr, y[tr_pos], yl[tr_pos], yu[tr_pos], ct[tr_pos])
        else:
            est = make_estimator(
                family=recipe["family"],
                params=recipe["params"],
                task_type="regression",
                n_jobs=n_jobs,
                bounds=bounds,
                winsor_cap=recipe.get("winsor_cap"),
                seed=142 + f,
            )
            est.fit(X_tr, y[tr_pos])
        folds_out.append((f, pre, est))
    return folds_out, cidx[valid]


def select_best_recipe(
    bank: V331FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    track: str,
    n_jobs: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n_cv = len(ep_data["cv_idx"])
    candidates = get_candidate_recipes(
        size_tier=spec["size_tier"],
        task_type="regression",
        track=track,
        n_cv=n_cv,
        outlier_winsor_cap=spec.get("outlier_winsor_cap"),
    )
    grouped: Dict[Tuple[str, bool], List[Dict[str, Any]]] = {}
    for cand in candidates:
        grouped.setdefault((cand["route"], cand["scale"]), []).append(cand)

    results = []
    best_res = None
    for (route, scale), group_cands in grouped.items():
        f_cache = _build_fold_cache(bank, ep_data, track, route=route, scale=scale, include_censored_in_train=False)
        for cand in group_cands:
            try:
                r = evaluate_recipe_cv(bank, ep_data, spec, track, cand, n_jobs=n_jobs, return_folds=False, fold_cache=f_cache)
                results.append(r)
                if best_res is None or r["score_tuple"] > best_res["score_tuple"]:
                    best_res = r
            except Exception as exc:
                results.append({"recipe": cand, "error": str(exc), "score_tuple": (-1e9, -1e9), "metrics": {}})
        del f_cache

    assert best_res is not None, f"{ep_data['endpoint']} ({track}): all candidates failed!"
    winner = evaluate_recipe_cv(bank, ep_data, spec, track, best_res["recipe"], n_jobs=n_jobs, return_folds=True)
    sorted_res = sorted([r for r in results if "error" not in r], key=lambda x: x["score_tuple"], reverse=True)
    lb = [
        {"route": r["recipe"]["route"], "family": r["recipe"]["family"], "winsor_cap": r["recipe"].get("winsor_cap"), "params": r["recipe"]["params"], "cv_metrics": r["metrics"]}
        for r in sorted_res[:8]
    ]
    return winner, lb


def run_single_endpoint(args: Tuple[str, int]) -> Dict[str, Any]:
    endpoint, n_jobs = args
    t0 = time.time()
    bank = GLOBAL_BANK
    assert bank is not None
    spec = V331_ENDPOINT_SPECS[endpoint]
    ep_data = load_endpoint_data(endpoint, bank)
    bounds = spec.get("bounds")
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)
    print(f"[CPU START] {endpoint:<22} (n={ep_data['n']}, cv={n_cv}, ho={n_ho}, jobs={n_jobs})", flush=True)

    win_smi, lb_smi = select_best_recipe(bank, ep_data, spec, "smiles_only", n_jobs=n_jobs)
    win_dual, lb_dual = select_best_recipe(bank, ep_data, spec, "dual_modality", n_jobs=n_jobs)

    win_seq = None
    if endpoint in V331_SEQ_ONLY_ENDPOINTS:
        win_seq, _ = select_best_recipe(bank, ep_data, spec, "seq_only", n_jobs=n_jobs)

    # Censored XGBoost survival:aft on all endpoints with censored CV rows
    n_cens_total = int((ep_data["censor_type"] != "exact").sum())
    n_cens_cv = int((ep_data["censor_type"][cv_rows] != "exact").sum())
    win_aft_dual = win_aft_smi = None
    p_ho_aft_d = p_ho_aft_s = None
    aft_ablation = None

    ho_cidx = ep_data["compound_idx"][ho_rows]
    ho_cond = [ep_data["cond_rows"][i] for i in ho_rows]
    y_cv, yl_cv, yu_cv, ct_cv, fids_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows], ep_data["fold_id"][cv_rows]
    y_ho, yl_ho, yu_ho, ct_ho = ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]

    if n_cens_cv >= 1:
        aft_cands = get_aft_candidates(spec["size_tier"])
        aft_by_route: Dict[str, List[Dict[str, Any]]] = {}
        for ac in aft_cands:
            aft_by_route.setdefault(ac["route"], []).append(ac)
        best_aft_d = None
        for r_name, r_cands in aft_by_route.items():
            aft_cache = _build_fold_cache(bank, ep_data, "dual_modality", route=r_name, scale=False, include_censored_in_train=True)
            for ac in r_cands:
                r_a = evaluate_aft_cv(bank, ep_data, spec, "dual_modality", ac, n_jobs=n_jobs, return_folds=False, fold_cache=aft_cache)
                if best_aft_d is None or r_a["score_tuple"] > best_aft_d["score_tuple"]:
                    best_aft_d = r_a
            del aft_cache
        assert best_aft_d is not None
        best_ac_d = {"route": best_aft_d["recipe"]["route"], "params": best_aft_d["recipe"]["params"]}
        win_aft_dual = evaluate_aft_cv(bank, ep_data, spec, "dual_modality", best_ac_d, n_jobs=n_jobs, return_folds=True)
        win_aft_smi = evaluate_aft_cv(bank, ep_data, spec, "smiles_only", best_ac_d, n_jobs=n_jobs, return_folds=True)

        p_ho_aft_d, _, _ = predict_ensemble(win_aft_dual["folds"], bank, ho_cidx, "dual_modality", cond_rows=ho_cond)
        p_ho_aft_s, _, _ = predict_ensemble(win_aft_smi["folds"], bank, ho_cidx, "smiles_only", cond_rows=ho_cond)
        aft_ablation = {
            "n_censored_total": n_cens_total,
            "n_censored_cv": n_cens_cv,
            "n_censored_holdout": int((ct_ho != "exact").sum()),
            "aft_winning_recipe": win_aft_dual["recipe"],
            "xgb_aft_dual_cv": win_aft_dual["metrics"],
            "xgb_aft_dual_holdout": regression_metrics(y_ho, p_ho_aft_d, yl_ho, yu_ho, ct_ho),
            "xgb_aft_smiles_only_cv": win_aft_smi["metrics"],
            "xgb_aft_smiles_only_holdout": regression_metrics(y_ho, p_ho_aft_s, yl_ho, yu_ho, ct_ho),
        }

    # Evaluate on untouched 20% holdout
    p_ho_smi, sd_ho_smi, gpsd_ho_smi = predict_ensemble(win_smi["folds"], bank, ho_cidx, "smiles_only", cond_rows=ho_cond)
    p_ho_dual, sd_ho_dual, gpsd_ho_dual = predict_ensemble(win_dual["folds"], bank, ho_cidx, "dual_modality", cond_rows=ho_cond)
    p_ho_seq = None
    if win_seq is not None:
        p_ho_seq, _, _ = predict_ensemble(win_seq["folds"], bank, ho_cidx, "seq_only", cond_rows=ho_cond)

    ad_index = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][cv_rows])
    oof_max_t = compute_oof_tanimoto(bank, ep_data)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, ho_cidx)

    ho_metrics_smi = regression_metrics(y_ho, p_ho_smi, yl_ho, yu_ho, ct_ho)
    ho_metrics_dual = regression_metrics(y_ho, p_ho_dual, yl_ho, yu_ho, ct_ho)
    ho_metrics_seq = regression_metrics(y_ho, p_ho_seq, yl_ho, yu_ho, ct_ho) if p_ho_seq is not None else None

    conf_smi = ConformalIntervalCalibrator(bounds=bounds).fit(y_cv, win_smi["oof_pred"], ct_cv, oof_sd=win_smi["oof_sd"], oof_max_tanimoto=oof_max_t)
    conf_dual = ConformalIntervalCalibrator(bounds=bounds).fit(y_cv, win_dual["oof_pred"], ct_cv, oof_sd=win_dual["oof_sd"], oof_max_tanimoto=oof_max_t)
    ho_iv_smi = conf_smi.predict_intervals(p_ho_smi, model_sd=gpsd_ho_smi, max_tanimoto=ho_max_t)
    ho_iv_dual = conf_dual.predict_intervals(p_ho_dual, model_sd=gpsd_ho_dual, max_tanimoto=ho_max_t)
    conf_summary = {
        "smiles_only": {"cv_diagnostics": conf_smi.cv_diagnostics_, "holdout_coverage": conf_smi.evaluate_coverage(y_ho, p_ho_smi, ct_ho, model_sd=gpsd_ho_smi, max_tanimoto=ho_max_t)},
        "dual_modality": {"cv_diagnostics": conf_dual.cv_diagnostics_, "holdout_coverage": conf_dual.evaluate_coverage(y_ho, p_ho_dual, ct_ho, model_sd=gpsd_ho_dual, max_tanimoto=ho_max_t)},
    }
    ad_strat_dual = compute_ad_stratification(y_ho, p_ho_dual, ct_ho, ho_max_t, ho_ad_tiers, "regression")
    ad_index.tier_calibration = ad_strat_dual

    # Strict Leave-Case17-Out Refit & Evaluation
    c17_idx = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)
    keep_non_c17 = np.where(ep_data["case17_excluded"] == 0)[0]
    c17_folds_smi, _ = fit_leave_case17_out_ensemble(bank, ep_data, spec, "smiles_only", win_smi["recipe"], n_jobs=n_jobs, is_aft=False)
    c17_folds_dual, c17_tr_cidx_dual = fit_leave_case17_out_ensemble(bank, ep_data, spec, "dual_modality", win_dual["recipe"], n_jobs=n_jobs, is_aft=False)
    c17_folds_aft = None
    if win_aft_dual is not None:
        c17_folds_aft, _ = fit_leave_case17_out_ensemble(bank, ep_data, spec, "dual_modality", win_aft_dual["recipe"], n_jobs=n_jobs, is_aft=True)
    c17_folds_seq = None
    if win_seq is not None:
        c17_folds_seq, _ = fit_leave_case17_out_ensemble(bank, ep_data, spec, "seq_only", win_seq["recipe"], n_jobs=n_jobs, is_aft=False)

    canon_cond = spec["canonical_conditions"]
    c17_p_smi, _, _ = predict_ensemble(c17_folds_smi, bank, c17_idx, "smiles_only", default_cond=canon_cond)
    c17_p_dual, c17_sd_dual, c17_gpsd_dual = predict_ensemble(c17_folds_dual, bank, c17_idx, "dual_modality", default_cond=canon_cond)
    c17_p_aft = predict_ensemble(c17_folds_aft, bank, c17_idx, "dual_modality", default_cond=canon_cond)[0] if c17_folds_aft is not None else None
    c17_p_seq = predict_ensemble(c17_folds_seq, bank, c17_idx, "seq_only", default_cond=canon_cond)[0] if c17_folds_seq is not None else None

    c17_alt_preds: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
    for alt_name, alt_cond in spec.get("alt_case17_conditions", {}).items():
        ap_s, _, _ = predict_ensemble(c17_folds_smi, bank, c17_idx, "smiles_only", default_cond=alt_cond)
        ap_d, _, _ = predict_ensemble(c17_folds_dual, bank, c17_idx, "dual_modality", default_cond=alt_cond)
        ap_a = predict_ensemble(c17_folds_aft, bank, c17_idx, "dual_modality", default_cond=alt_cond)[0] if c17_folds_aft is not None else None
        c17_alt_preds[alt_name] = {"smiles_only": ap_s, "dual": ap_d, "aft": ap_a}

    ad_c17 = TanimotoADIndex.from_indices(bank, c17_tr_cidx_dual)
    c17_max_t, c17_top5_t, c17_ad_tiers = ad_c17.query(bank, c17_idx)
    c17_iv_dual = conf_dual.predict_intervals(c17_p_dual, model_sd=c17_gpsd_dual, max_tanimoto=c17_max_t)

    # Save standardized per-endpoint outputs
    ep_out_dir = EXP_CPU_DIR / endpoint
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{float(v):.6f}"
        return str(v)

    oof_cols = ["record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type", "pred_smiles_only", "pred_dual"]
    if win_aft_dual is not None:
        oof_cols.extend(["pred_aft", "pred_aft_smiles_only"])
    oof_cols.append("fold_sd")
    if win_seq is not None:
        oof_cols.append("pred_seq_only")
    oof_cols.append("max_tanimoto_oof")

    with (ep_out_dir / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(oof_cols)
        for loc_i, orig_i in enumerate(cv_rows):
            row = [
                ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i], int(fids_cv[loc_i]),
                _fmt(y_cv[loc_i]), _fmt(yl_cv[loc_i]), _fmt(yu_cv[loc_i]), str(ct_cv[loc_i]),
                _fmt(win_smi["oof_pred"][loc_i]), _fmt(win_dual["oof_pred"][loc_i]),
            ]
            if win_aft_dual is not None:
                row.extend([_fmt(win_aft_dual["oof_pred"][loc_i]), _fmt(win_aft_smi["oof_pred"][loc_i])])
            row.append(_fmt(win_dual["oof_sd"][loc_i] if win_dual["oof_sd"] is not None else np.nan))
            if win_seq is not None:
                row.append(_fmt(win_seq["oof_pred"][loc_i]))
            row.append(_fmt(oof_max_t[loc_i]))
            w.writerow(row)

    ho_cols = ["record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type", "pred_smiles_only", "pred_dual"]
    if p_ho_aft_d is not None:
        ho_cols.extend(["pred_aft", "pred_aft_smiles_only"])
    ho_cols.extend(["fold_sd", "fold_sd_smiles_only"])
    if p_ho_seq is not None:
        ho_cols.append("pred_seq_only")
    ho_cols.extend(["lo80", "hi80", "lo90", "hi90", "lo80_smiles", "hi80_smiles", "lo90_smiles", "hi90_smiles", "max_tanimoto", "top5_tanimoto", "ad_tier"])

    with (ep_out_dir / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ho_cols)
        for loc_i, orig_i in enumerate(ho_rows):
            row = [
                ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i],
                _fmt(y_ho[loc_i]), _fmt(yl_ho[loc_i]), _fmt(yu_ho[loc_i]), str(ct_ho[loc_i]),
                _fmt(p_ho_smi[loc_i]), _fmt(p_ho_dual[loc_i]),
            ]
            if p_ho_aft_d is not None:
                row.extend([_fmt(p_ho_aft_d[loc_i]), _fmt(p_ho_aft_s[loc_i])])
            row.extend([_fmt(sd_ho_dual[loc_i]), _fmt(sd_ho_smi[loc_i])])
            if p_ho_seq is not None:
                row.append(_fmt(p_ho_seq[loc_i]))
            row.extend([
                _fmt(ho_iv_dual["lo80"][loc_i]), _fmt(ho_iv_dual["hi80"][loc_i]),
                _fmt(ho_iv_dual["lo90"][loc_i]), _fmt(ho_iv_dual["hi90"][loc_i]),
                _fmt(ho_iv_smi["lo80"][loc_i]), _fmt(ho_iv_smi["hi80"][loc_i]),
                _fmt(ho_iv_smi["lo90"][loc_i]), _fmt(ho_iv_smi["hi90"][loc_i]),
                _fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i],
            ])
            w.writerow(row)

    c17_cols = ["drug_name", "compound_key", "compound_idx", "clinical_route", "condition_used", "pred_smiles_only", "pred_dual"]
    if c17_p_aft is not None:
        c17_cols.append("pred_aft")
    if c17_p_seq is not None:
        c17_cols.append("pred_seq_only")
    c17_cols.extend(["fold_sd", "lo80", "hi80", "lo90", "hi90", "max_tanimoto", "top5_tanimoto", "ad_tier"])
    for alt_name in c17_alt_preds:
        c17_cols.extend([f"pred_smiles_only_{alt_name}", f"pred_dual_{alt_name}"])
        if c17_alt_preds[alt_name]["aft"] is not None:
            c17_cols.append(f"pred_aft_{alt_name}")

    cond_str = json.dumps(canon_cond, ensure_ascii=False)
    with (ep_out_dir / "case17_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(c17_cols)
        for i, q in enumerate(bank.case17_queries):
            dname = q["drug_name"]
            row = [dname, q["compound_key"], int(q["compound_idx"]), CASE17_CLINICAL_ROUTE.get(dname, "IV"), cond_str, _fmt(c17_p_smi[i]), _fmt(c17_p_dual[i])]
            if c17_p_aft is not None:
                row.append(_fmt(c17_p_aft[i]))
            if c17_p_seq is not None:
                row.append(_fmt(c17_p_seq[i]))
            row.extend([
                _fmt(c17_sd_dual[i]), _fmt(c17_iv_dual["lo80"][i]), _fmt(c17_iv_dual["hi80"][i]),
                _fmt(c17_iv_dual["lo90"][i]), _fmt(c17_iv_dual["hi90"][i]),
                _fmt(c17_max_t[i]), _fmt(c17_top5_t[i]), c17_ad_tiers[i],
            ])
            for alt_name, ap_d in c17_alt_preds.items():
                row.extend([_fmt(ap_d["smiles_only"][i]), _fmt(ap_d["dual"][i])])
                if ap_d["aft"] is not None:
                    row.append(_fmt(ap_d["aft"][i]))
            w.writerow(row)

    elapsed = round(time.time() - t0, 2)
    metrics_doc = {
        "endpoint": endpoint,
        "tier": spec["tier"],
        "task_type": "regression",
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
        "censor_counts": {k: int((ep_data["censor_type"] == k).sum()) for k in ("exact", "right", "left", "interval")},
        "smiles_valid_pct": round(100.0 * float(bank.smiles_valid[ep_data["compound_idx"]].mean()), 2),
        "seq_valid_pct": round(100.0 * float(bank.seq_valid[ep_data["compound_idx"]].mean()), 2),
        "winning_recipe_smiles_only": {**win_smi["recipe"], "uses_meta": False, "meta_endpoints": []},
        "winning_recipe_dual": {**win_dual["recipe"], "uses_meta": False, "meta_endpoints": []},
        "winning_recipe_seq_only": win_seq["recipe"] if win_seq is not None else None,
        "cv_metrics": {"smiles_only": win_smi["metrics"], "dual_modality": win_dual["metrics"], "seq_only": win_seq["metrics"] if win_seq is not None else None},
        "holdout_metrics": {"smiles_only": ho_metrics_smi, "dual_modality": ho_metrics_dual, "seq_only": ho_metrics_seq},
        "aft_ablation": aft_ablation,
        "conformal": conf_summary,
        "applicability_domain": ad_strat_dual,
        "selection_leaderboard_smiles_only": lb_smi,
        "selection_leaderboard_dual": lb_dual,
        "elapsed_sec": elapsed,
    }
    (ep_out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    model_bundle = {
        "schema": "peptide_omnipanel_v33_cpu_5fold_v1",
        "endpoint": endpoint,
        "task_type": "regression",
        "unit": spec["unit"],
        "bounds": bounds,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": canon_cond,
        "smiles_only": {"recipe": win_smi["recipe"], "uses_meta": False, "meta_endpoints": [], "folds": win_smi["folds"], "calibrator": None, "conformal": conf_smi},
        "dual_modality": {"recipe": win_dual["recipe"], "uses_meta": False, "meta_endpoints": [], "folds": win_dual["folds"], "calibrator": None, "conformal": conf_dual},
        "seq_only": {"recipe": win_seq["recipe"], "folds": win_seq["folds"], "calibrator": None} if win_seq is not None else None,
        "aft": {"recipe": win_aft_dual["recipe"], "folds_dual": win_aft_dual["folds"], "folds_smiles_only": win_aft_smi["folds"]} if win_aft_dual is not None else None,
        "ad_index": ad_index,
    }
    joblib.dump(model_bundle, ep_out_dir / "models_5fold.joblib", compress=3)
    print(
        f"[CPU DONE ] {endpoint:<22} ({elapsed:5.1f}s) | {win_dual['recipe']['route']}+{win_dual['recipe']['family']} | "
        f"HO MAE={ho_metrics_dual['mae']:.4f} Sp={ho_metrics_dual['spearman']:.3f} R2={ho_metrics_dual['r2']:.3f}",
        flush=True,
    )
    return {"endpoint": endpoint, "metrics": metrics_doc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="*", default=None)
    ap.add_argument("--max-workers", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    EXP_CPU_DIR.mkdir(parents=True, exist_ok=True)
    global GLOBAL_BANK
    GLOBAL_BANK = V331FeatureBank()
    print(f"V331FeatureBank ready in {time.time()-t0:.2f}s (N={GLOBAL_BANK.n_total:,} compounds)", flush=True)

    # LPT order (largest endpoints first!)
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
    print(f"=== V33.1 CPU Suite: {len(tasks)} endpoints across {n_workers} workers ===", flush=True)
    with ctx.Pool(processes=n_workers) as pool:
        for item in pool.imap_unordered(run_single_endpoint, tasks):
            summary_by_ep[item["endpoint"]] = item["metrics"]

    master = {
        "schema": "peptide_omnipanel_v331_cpu_benchmark_summary_v1",
        "total_endpoints": len(summary_by_ep),
        "total_elapsed_sec": round(time.time() - t0, 2),
        "endpoints": {ep: summary_by_ep[ep] for ep in sorted(summary_by_ep.keys())},
    }
    out_fp = EXP_CPU_DIR / "cpu_benchmark_summary.json"
    out_fp.write_text(json.dumps(master, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nALL {len(summary_by_ep)} V33.1 CPU ENDPOINTS DONE in {master['total_elapsed_sec']}s -> {out_fp}", flush=True)


if __name__ == "__main__":
    main()
