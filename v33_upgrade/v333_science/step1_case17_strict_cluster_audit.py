#!/usr/bin/env python3
"""v33.3 Step 1: Strict Full-Cluster Leave-Case17-Out Fix & Re-Audit (L1).

Fixes BOTH Case17 evaluation leakage bugs identified in independent scientific audits:
  1. Analog-cluster leakage: `run_v33_cpu_suite.py` and `run_v33_fusion_suite.py` filtered only
     `case17_excluded == 0` (439 exact rows), leaving 472 same-cluster analog rows (across 19
     Case17 clusters, 911 total rows) in training — whereas `deep_data.py` properly excluded all
     19 clusters.
  2. Stacker-OOF leakage: `run_v33_fusion_suite.py:1341` refit `c17_stacker` on `all_oof`
     computed from base heads trained WITH Case17 rows in other folds.

This script:
  - Strictly excludes ALL 19 Case17 clusters (union of `case17_excluded==1` and `case17_queries`
    clusters across all 24 v33.0 endpoints) with hard zero-overlap assertions.
  - Refits all 5 AUX_ENDPOINTS strictly cluster-excluded first so CPU meta-features have 0 leak.
  - Refits CPU and Fusion base heads strictly on non-Case17-cluster rows using 5 balanced cluster
    folds, capturing strictly clean out-of-fold predictions on non-Case17-cluster rows.
  - Fits Fusion `c17_stacker` (dual_modality, smiles_only, aft) strictly on those clean
    non-Case17-cluster OOF predictions.
  - Combines with existing strictly cluster-excluded Deep Case17 predictions using the exact
    frozen Wave 5A `v33_manifest.json` champion weights & calibrators (verifying <1e-4 parity
    when plugging original Old CPU/Fusion predictions).
  - Writes full Before vs After comparison JSON under `v333_science/`.
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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

V33_BASE = Path("${V33_BASE_DIR}")
DATA_DIR = V33_BASE / "data"
CPU_DIR = V33_BASE / "models_cpu"
FUSION_DIR = V33_BASE / "models_fusion"
DEEP_DIR = V33_BASE / "models_gpu_deep"
BUNDLE_DIR = V33_BASE / "bundle"
OUT_DIR = V33_BASE / "v333_science"

sys.path.insert(0, str(CPU_DIR))
sys.path.insert(0, str(FUSION_DIR))

from fusion_bank import FusionBank  # noqa: E402
from fusion_models import (  # noqa: E402
    EmbeddingFoldPreprocessor,
    ModalityAwareStacker,
    XGBoostAFTRegressor as FusionXGBoostAFT,
    make_fusion_estimator,
)
from v33_config import (  # noqa: E402
    AUX_ENDPOINTS,
    CASE17_CLINICAL_ROUTE,
    CASE17_FDA_NUMERIC,
    ENDPOINT_SPECS,
    compute_halflife_mechanism_anchor,
)
ALL_ENDPOINTS = list(ENDPOINT_SPECS.keys())
from v33_features import FeatureBank, FoldPreprocessor  # noqa: E402
from v33_models import XGBoostAFTRegressor as CpuXGBoostAFT, make_estimator  # noqa: E402

csv.field_size_limit(10_000_000)

SCORECARD_ENDPOINTS = [
    "half_life_reg",
    "PPB",
    "hemolysis_cls",
    "hemolysis_HC50",
    "overall_peptide_toxicity",
    "BBB",
    "PAMPA",
    "Caco2",
    "LogD7.4",
    "CL_per_kg",
    "Vd_per_kg",
    "F",
    "solubility_cls",
]

# Global read-only state shared via fork CoW
G_CPU_BANK: Optional[FeatureBank] = None
G_FUS_BANK: Optional[FusionBank] = None
G_ALL_C17_CLUSTERS: Set[str] = set()
G_STRICT_META_CACHE: Dict[str, Any] = {}
G_MANIFEST: Dict[str, Any] = {}


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _col_arr(rows: List[Dict[str, str]], col: str) -> Optional[np.ndarray]:
    if not rows or col not in rows[0]:
        return None
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        s = r.get(col, "")
        if s != "":
            out[i] = float(s)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)))


def _apply_calibrator(p_raw: np.ndarray, cal_spec: Optional[Dict[str, Any]]) -> np.ndarray:
    p = np.asarray(p_raw, dtype=np.float64)
    if cal_spec is None:
        return np.clip(p, 1e-5, 1.0 - 1e-5)
    method = cal_spec.get("method", "none")
    params = cal_spec.get("params", {})
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    if method == "platt":
        return np.clip(_sigmoid(float(params["a"]) * _logit(pc) + float(params["b"])), 1e-5, 1.0 - 1e-5)
    if method == "isotonic":
        xs = np.asarray(params["x_thresholds"], dtype=np.float64)
        ys = np.asarray(params["y_thresholds"], dtype=np.float64)
        return np.clip(np.interp(pc, xs, ys), 1e-5, 1.0 - 1e-5)
    return np.clip(pc, 1e-5, 1.0 - 1e-5)


def _apply_simplex_blend(
    names: List[str],
    streams: Dict[str, np.ndarray],
    weights: Dict[str, float],
    task_type: str,
    bounds: Optional[Tuple[float, float]] = None,
    fallback: float = 0.0,
) -> np.ndarray:
    active = [s for s in names if float(weights.get(s, 0.0)) > 0.0 and s in streams]
    if not active:
        active = [names[0]]
    w = np.array([float(weights[s]) for s in active], dtype=np.float64)
    w = w / np.sum(w)
    M = np.column_stack([streams[s] for s in active])
    fin = np.isfinite(M)
    if task_type == "classification":
        Z = np.where(fin, _logit(np.where(fin, M, 0.5)), 0.0)
        W = np.where(fin, w[None, :], 0.0)
        wsum = W.sum(axis=1)
        z_out = np.divide((Z * W).sum(axis=1), wsum, out=np.zeros(M.shape[0]), where=wsum > 1e-12)
        return np.clip(_sigmoid(z_out), 1e-6, 1.0 - 1e-6)
    M0 = np.where(fin, M, 0.0)
    W = np.where(fin, w[None, :], 0.0)
    wsum = W.sum(axis=1)
    y_out = np.divide((M0 * W).sum(axis=1), wsum, out=np.full(M.shape[0], fallback), where=wsum > 1e-12)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


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


def _track_valid_mask(bank: Any, cidx: np.ndarray, track: str) -> np.ndarray:
    if track == "smiles_only":
        return bank.smiles_valid[cidx]
    if track == "seq_only":
        return bank.seq_valid[cidx]
    return np.ones(len(cidx), dtype=bool)


def _predict_est(est: Any, X: Any, task_type: str) -> np.ndarray:
    is_cb = "catboost" in type(est).__module__.lower()
    if task_type == "classification":
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X, thread_count=1) if is_cb else est.predict_proba(X)
            return np.asarray(proba, dtype=np.float64)[:, 1]
        return _sigmoid(np.asarray(est.decision_function(X), dtype=np.float64))
    pred = est.predict(X, thread_count=1) if is_cb else est.predict(X)
    return np.asarray(pred, dtype=np.float64)


def _strict_keep_indices(ep_data: Dict[str, Any], all_c17_clusters: Set[str], c17_idx: np.ndarray) -> np.ndarray:
    cids = ep_data["cluster_ids"]
    keep_mask = (ep_data["case17_excluded"] == 0) & np.array([str(c) not in all_c17_clusters for c in cids], dtype=bool)
    keep_rows = np.where(keep_mask)[0]
    # Hard zero-leakage assertions
    kept_clusters = set(str(c) for c in cids[keep_rows])
    assert kept_clusters.isdisjoint(all_c17_clusters), f"{ep_data['endpoint']}: Case17 cluster leaked!"
    kept_cidx = set(int(x) for x in ep_data["compound_idx"][keep_rows])
    assert kept_cidx.isdisjoint(set(int(x) for x in c17_idx)), f"{ep_data['endpoint']}: Case17 compound leaked!"
    return keep_rows


def _fit_cpu_strict_5fold(
    bank: FeatureBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    keep_rows: np.ndarray,
    fids: np.ndarray,
    track: str,
    recipe: Dict[str, Any],
    c17_idx: np.ndarray,
    meta_tr_all: Optional[np.ndarray] = None,
    meta_q: Optional[np.ndarray] = None,
    is_aft: bool = False,
    n_jobs: int = 2,
) -> Dict[str, Any]:
    cidx = ep_data["compound_idx"][keep_rows]
    clids = ep_data["cluster_ids"][keep_rows]
    y = ep_data["y_value"][keep_rows]
    yl = ep_data["y_lower"][keep_rows]
    yu = ep_data["y_upper"][keep_rows]
    ct = ep_data["censor_type"][keep_rows]
    conds = [ep_data["cond_rows"][i] for i in keep_rows]
    valid = _track_valid_mask(bank, cidx, track)
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    canon_cond = spec["canonical_conditions"]
    alt_conds = spec.get("alt_case17_conditions", {})

    oof = np.full(len(keep_rows), np.nan, dtype=np.float64)
    c17_canon_folds: List[np.ndarray] = []
    c17_alt_folds: Dict[str, List[np.ndarray]] = {k: [] for k in alt_conds}
    folds_saved: List[Tuple[int, FoldPreprocessor, Any]] = []
    fold_train_clusters: Dict[int, Set[str]] = {}
    cluster_to_fold = {str(clids[i]): int(fids[i]) for i in range(len(keep_rows))}

    for f in range(5):
        va_mask = (fids == f) & valid
        tr_mask = (fids != f) & valid
        if task_type == "regression" and not is_aft:
            tr_mask = tr_mask & (ct == "exact")
        if tr_mask.sum() < 4:
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]
        fold_train_clusters[f] = set(str(clids[p]) for p in tr_pos)
        assert fold_train_clusters[f].isdisjoint(G_ALL_C17_CLUSTERS)

        pre = FoldPreprocessor(
            track=track,
            route=recipe["route"],
            cond_cols=ep_data["cond_cols"],
            scale_for_linear=recipe.get("scale", False),
            svd_dim=32,
            min_fp_support_frac=0.01 if len(keep_rows) >= 1000 else 0.005,
            random_state=142 + f,
        )
        m_tr = meta_tr_all[tr_pos] if meta_tr_all is not None else None
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos], meta=m_tr)

        if is_aft:
            est = CpuXGBoostAFT(**recipe["params"], nthread=n_jobs, seed=142 + f, bounds=bounds)
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

        if len(va_pos) > 0:
            m_va = meta_tr_all[va_pos] if meta_tr_all is not None else None
            X_va = pre.transform(bank, cidx[va_pos], [conds[i] for i in va_pos], meta=m_va)
            oof[va_pos] = _predict_est(est, X_va, task_type)

        X_q = pre.transform(bank, c17_idx, default_cond=canon_cond, meta=meta_q)
        c17_canon_folds.append(_predict_est(est, X_q, task_type))
        for alt_k, alt_c in alt_conds.items():
            X_qa = pre.transform(bank, c17_idx, default_cond=alt_c, meta=meta_q)
            c17_alt_folds[alt_k].append(_predict_est(est, X_qa, task_type))

        folds_saved.append((f, pre, est))

    return {
        "oof": oof,
        "c17": np.mean(np.stack(c17_canon_folds, axis=0), axis=0),
        "c17_alt": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in c17_alt_folds.items()},
        "folds": folds_saved,
        "cluster_to_fold": cluster_to_fold,
        "fold_train_clusters": fold_train_clusters,
    }


def _gen_strict_ooc_meta(
    bank: FeatureBank,
    meta_cache: Dict[str, Any],
    target_ep: str,
    track: str,
    compound_idx: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    aux_list = [a for a in AUX_ENDPOINTS if a != target_ep]
    n = len(compound_idx)
    M = np.zeros((n, len(aux_list)), dtype=np.float32)
    for j, aux_ep in enumerate(aux_list):
        b = meta_cache[aux_ep][track]
        folds = b["folds"]
        cl_to_fold = b["cluster_to_fold"]
        fold_train_clusters = b["fold_train_clusters"]
        default_cond = ENDPOINT_SPECS[aux_ep]["canonical_conditions"]
        task_type = ENDPOINT_SPECS[aux_ep]["task_type"]
        cal = b.get("calibrator")

        fold_preds = {}
        for f, pre, est in folds:
            X_q = pre.transform(bank, compound_idx, default_cond=default_cond)
            p = _predict_est(est, X_q, task_type)
            if task_type == "classification" and cal is not None:
                p = cal.transform(p)
            fold_preds[f] = p

        all_f = sorted(fold_preds.keys())
        mean_all = np.mean(np.stack([fold_preds[f] for f in all_f], axis=0), axis=0)
        col = np.empty(n, dtype=np.float64)
        for i in range(n):
            cid = str(cluster_ids[i])
            if cid in cl_to_fold:
                f_ooc = cl_to_fold[cid]
                assert cid not in fold_train_clusters[f_ooc], f"OOC meta leak: {cid} in {aux_ep} f={f_ooc}"
                col[i] = fold_preds[f_ooc][i]
            else:
                for f_chk in all_f:
                    assert cid not in fold_train_clusters[f_chk], f"OOC meta leak: {cid} in {aux_ep} f={f_chk}"
                col[i] = mean_all[i]
        M[:, j] = col.astype(np.float32)
    return M


def _fit_fusion_head_strict_5fold(
    bank: FusionBank,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    keep_rows: np.ndarray,
    fids: np.ndarray,
    head_info: Dict[str, Any],
    c17_idx: np.ndarray,
    n_jobs: int = 2,
) -> Dict[str, Any]:
    cidx = ep_data["compound_idx"][keep_rows]
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
    canon_cond = spec["canonical_conditions"]
    alt_conds = spec.get("alt_case17_conditions", {})

    oof = np.full(len(keep_rows), np.nan, dtype=np.float64)
    c17_canon_folds: List[np.ndarray] = []
    c17_alt_folds: Dict[str, List[np.ndarray]] = {k: [] for k in alt_conds}

    for f in range(5):
        va_mask = (fids == f) & valid
        tr_mask = (fids != f) & valid
        if task_type == "regression" and not is_aft:
            tr_mask = tr_mask & (ct == "exact")
        if tr_mask.sum() < 4:
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]
        assert set(str(clids[p]) for p in tr_pos).isdisjoint(G_ALL_C17_CLUSTERS)

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
            est = FusionXGBoostAFT(**recipe["params"], nthread=n_jobs, seed=142 + f, bounds=bounds)
            est.fit(X_tr, y[tr_pos], yl[tr_pos], yu[tr_pos], ct[tr_pos])
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

        if len(va_pos) > 0:
            X_va = pre.transform(bank, cidx[va_pos], [conds[i] for i in va_pos])
            oof[va_pos] = _predict_est(est, X_va, task_type)

        X_q = pre.transform(bank, c17_idx, default_cond=canon_cond)
        c17_canon_folds.append(_predict_est(est, X_q, task_type))
        for alt_k, alt_c in alt_conds.items():
            X_qa = pre.transform(bank, c17_idx, default_cond=alt_c)
            c17_alt_folds[alt_k].append(_predict_est(est, X_qa, task_type))

    return {
        "oof": oof,
        "c17": np.mean(np.stack(c17_canon_folds, axis=0), axis=0),
        "c17_alt": {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in c17_alt_folds.items()},
    }


def _load_ep_data(bank: FeatureBank, ep: str) -> Dict[str, Any]:
    rows = _read_tsv(DATA_DIR / "endpoints" / f"{ep}.tsv")
    cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
    return {
        "endpoint": ep,
        "n": len(rows),
        "cond_cols": cond_cols,
        "record_ids": [r["record_id"] for r in rows],
        "compound_keys": [r["compound_key"] for r in rows],
        "compound_idx": bank.map_keys([r["compound_key"] for r in rows]),
        "cluster_ids": np.array([r["cluster_id"] for r in rows], dtype=object),
        "case17_excluded": np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32),
        "censor_type": np.array([r["censor_type"] for r in rows], dtype=object),
        "y_value": np.array([float(r["y_value"]) for r in rows], dtype=np.float64),
        "y_lower": np.array([float(r["y_lower"]) for r in rows], dtype=np.float64),
        "y_upper": np.array([float(r["y_upper"]) for r in rows], dtype=np.float64),
        "cond_rows": [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows],
    }


def _run_one_endpoint(ep: str, n_jobs: int = 12) -> Dict[str, Any]:
    t0 = time.time()
    assert G_CPU_BANK is not None and G_FUS_BANK is not None
    spec = ENDPOINT_SPECS[ep]
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    ep_data = _load_ep_data(G_CPU_BANK, ep)
    c17_idx = np.array([q["compound_idx"] for q in G_CPU_BANK.case17_queries], dtype=np.int32)
    c17_names = [q["drug_name"] for q in G_CPU_BANK.case17_queries]
    c17_sentinels = np.array([f"CASE17_QUERY::{nm}" for nm in c17_names], dtype=object)

    keep_rows = _strict_keep_indices(ep_data, G_ALL_C17_CLUSTERS, c17_idx)
    n_tot = ep_data["n"]
    n_ex1 = int((ep_data["case17_excluded"] == 1).sum())
    n_strict_excl = int(n_tot - len(keep_rows))
    clids_strict = ep_data["cluster_ids"][keep_rows]
    fids_strict = assign_balanced_cluster_folds(clids_strict, n_folds=5)

    y_st = ep_data["y_value"][keep_rows]
    yl_st = ep_data["y_lower"][keep_rows]
    yu_st = ep_data["y_upper"][keep_rows]
    ct_st = ep_data["censor_type"][keep_rows]
    cidx_st = ep_data["compound_idx"][keep_rows]
    smi_st_mask = G_CPU_BANK.smiles_valid[cidx_st]
    dual_st_mask = np.ones(len(keep_rows), dtype=bool)

    # 1. Load CPU winning recipes & refit strictly cluster-excluded (reuse AUX cache when available!)
    cpu_m = json.loads((CPU_DIR / ep / "metrics.json").read_text(encoding="utf-8"))
    rec_dual = cpu_m["winning_recipe_dual"]
    rec_smi = cpu_m["winning_recipe_smiles_only"]

    meta_tr_dual = meta_q_dual = meta_tr_smi = meta_q_smi = None
    if rec_dual.get("uses_meta"):
        meta_tr_dual = _gen_strict_ooc_meta(G_CPU_BANK, G_STRICT_META_CACHE, ep, "dual_modality", cidx_st, clids_strict)
        meta_q_dual = _gen_strict_ooc_meta(G_CPU_BANK, G_STRICT_META_CACHE, ep, "dual_modality", c17_idx, c17_sentinels)
    if rec_smi.get("uses_meta"):
        meta_tr_smi = _gen_strict_ooc_meta(G_CPU_BANK, G_STRICT_META_CACHE, ep, "smiles_only", cidx_st, clids_strict)
        meta_q_smi = _gen_strict_ooc_meta(G_CPU_BANK, G_STRICT_META_CACHE, ep, "smiles_only", c17_idx, c17_sentinels)

    if ep in G_STRICT_META_CACHE:
        cpu_dual_res = G_STRICT_META_CACHE[ep]["dual_modality"]
        cpu_smi_res = G_STRICT_META_CACHE[ep]["smiles_only"]
    else:
        cpu_dual_res = _fit_cpu_strict_5fold(
            G_CPU_BANK, ep_data, spec, keep_rows, fids_strict, "dual_modality", rec_dual, c17_idx,
            meta_tr_all=meta_tr_dual, meta_q=meta_q_dual, is_aft=False, n_jobs=n_jobs,
        )
        cpu_smi_res = _fit_cpu_strict_5fold(
            G_CPU_BANK, ep_data, spec, keep_rows, fids_strict, "smiles_only", rec_smi, c17_idx,
            meta_tr_all=meta_tr_smi, meta_q=meta_q_smi, is_aft=False, n_jobs=n_jobs,
        )
    cpu_aft_dual_res = cpu_aft_smi_res = cpu_seq_res = None
    if cpu_m.get("aft_ablation") is not None and cpu_m["aft_ablation"].get("aft_winning_recipe") is not None:
        aft_rec = cpu_m["aft_ablation"]["aft_winning_recipe"]
        cpu_aft_dual_res = _fit_cpu_strict_5fold(
            G_CPU_BANK, ep_data, spec, keep_rows, fids_strict, "dual_modality", aft_rec, c17_idx,
            meta_tr_all=meta_tr_dual, meta_q=meta_q_dual, is_aft=True, n_jobs=n_jobs,
        )
        cpu_aft_smi_res = _fit_cpu_strict_5fold(
            G_CPU_BANK, ep_data, spec, keep_rows, fids_strict, "smiles_only", aft_rec, c17_idx,
            meta_tr_all=meta_tr_smi, meta_q=meta_q_smi, is_aft=True, n_jobs=n_jobs,
        )
    if cpu_m.get("winning_recipe_seq_only") is not None:
        cpu_seq_res = _fit_cpu_strict_5fold(
            G_CPU_BANK, ep_data, spec, keep_rows, fids_strict, "seq_only", cpu_m["winning_recipe_seq_only"], c17_idx,
            is_aft=False, n_jobs=n_jobs,
        )

    # 2. Load Fusion joblib metadata ONLY, then free 500MB+ estimators immediately to prevent OOM!
    import gc
    fb = joblib.load(FUSION_DIR / ep / "fusion_models_5fold.joblib")
    track_a_heads = list(fb["smiles_only"]["candidate_heads"])
    track_b_heads = list(fb["dual_modality"]["candidate_heads"])
    st_smi_method = fb["smiles_only"]["stacker"].method
    st_smi_params = dict(fb["smiles_only"]["stacker"].params)
    st_dual_method = fb["dual_modality"]["stacker"].method
    st_dual_params = dict(fb["dual_modality"]["stacker"].params)
    old_fusion_dual_w = fb["dual_modality"]["stacker"].weight_dict()
    aft_block = fb.get("aft")
    has_aft_stack = aft_block is not None and aft_block.get("stacker_dual") is not None
    st_aft_method = aft_block["stacker_dual"].method if has_aft_stack else None
    st_aft_params = dict(aft_block["stacker_dual"].params) if has_aft_stack else None

    cand_union = set(track_a_heads) | set(track_b_heads)
    if has_aft_stack:
        cand_union |= {"aft_direct_dual", "aft_direct_smiles"}
    # Keep ONLY tiny config dicts (recipe, rep_cfg, is_aft), drop all fitted fold models!
    pre_head_configs: Dict[str, Dict[str, Any]] = {}
    for h in sorted(cand_union):
        if h in fb["pretrained_base_heads"]:
            src_h = fb["pretrained_base_heads"][h]
            pre_head_configs[h] = {
                "rep_cfg": src_h["rep_cfg"],
                "recipe": src_h["recipe"],
                "is_aft": bool(src_h["is_aft"]),
            }
    del fb, aft_block
    gc.collect()

    pre_heads_to_refit = sorted(pre_head_configs.keys())

    strict_oof: Dict[str, np.ndarray] = {
        "cpu_dual": cpu_dual_res["oof"],
        "cpu_smiles": cpu_smi_res["oof"],
    }
    strict_c17: Dict[str, np.ndarray] = {
        "cpu_dual": cpu_dual_res["c17"],
        "cpu_smiles": cpu_smi_res["c17"],
    }
    strict_c17_alt: Dict[str, Dict[str, np.ndarray]] = {}
    for alt_k in spec.get("alt_case17_conditions", {}):
        strict_c17_alt[alt_k] = {
            "cpu_dual": cpu_dual_res["c17_alt"][alt_k],
            "cpu_smiles": cpu_smi_res["c17_alt"][alt_k],
        }
    if cpu_aft_dual_res is not None:
        strict_oof["cpu_aft_dual"] = cpu_aft_dual_res["oof"]
        strict_c17["cpu_aft_dual"] = cpu_aft_dual_res["c17"]
        for alt_k in spec.get("alt_case17_conditions", {}):
            strict_c17_alt[alt_k]["cpu_aft_dual"] = cpu_aft_dual_res["c17_alt"][alt_k]
    if cpu_aft_smi_res is not None:
        strict_oof["cpu_aft_smiles"] = cpu_aft_smi_res["oof"]
        strict_c17["cpu_aft_smiles"] = cpu_aft_smi_res["c17"]
        for alt_k in spec.get("alt_case17_conditions", {}):
            strict_c17_alt[alt_k]["cpu_aft_smiles"] = cpu_aft_smi_res["c17_alt"][alt_k]
    if cpu_seq_res is not None:
        strict_oof["cpu_seq_only"] = cpu_seq_res["oof"]
        strict_c17["cpu_seq_only"] = cpu_seq_res["c17"]

    for hname in pre_heads_to_refit:
        hres = _fit_fusion_head_strict_5fold(
            G_FUS_BANK, ep_data, spec, keep_rows, fids_strict, pre_head_configs[hname], c17_idx, n_jobs=n_jobs
        )
        strict_oof[hname] = hres["oof"]
        strict_c17[hname] = hres["c17"]
        for alt_k, arr in hres["c17_alt"].items():
            strict_c17_alt.setdefault(alt_k, {})[hname] = arr

    def _fit_clean_stacker(
        head_names: List[str],
        method: str,
        params: Dict[str, Any],
        extra_mask: np.ndarray,
        is_aft_st: bool = False,
    ) -> ModalityAwareStacker:
        avail_heads = [h for h in head_names if h in strict_oof]
        m_fit = extra_mask.copy()
        if task_type == "regression" and not is_aft_st:
            m_fit = m_fit & (ct_st == "exact")
        P_sub = np.column_stack([strict_oof[h][m_fit] for h in avail_heads])
        st = ModalityAwareStacker(
            task_type=task_type,
            method=method,
            params=dict(params),
            bounds=bounds,
        )
        st.fit(
            P_sub,
            y_st[m_fit],
            head_names=avail_heads,
            y_lower=yl_st[m_fit] if is_aft_st else None,
            y_upper=yu_st[m_fit] if is_aft_st else None,
            censor_type=ct_st[m_fit] if is_aft_st else None,
        )
        return st

    def _pred_stacker(st: ModalityAwareStacker, src: Dict[str, np.ndarray]) -> np.ndarray:
        cols = [
            src[h] if (h in src and src[h] is not None) else np.full(len(c17_idx), np.nan, dtype=np.float64)
            for h in st.head_names_
        ]
        return st.predict(np.column_stack(cols))

    st_smi_clean = _fit_clean_stacker(track_a_heads, st_smi_method, st_smi_params, smi_st_mask, is_aft_st=False)
    st_dual_clean = _fit_clean_stacker(track_b_heads, st_dual_method, st_dual_params, dual_st_mask, is_aft_st=False)
    st_aft_clean = None
    if has_aft_stack and st_aft_method is not None and "cpu_aft_dual" in strict_oof and "aft_direct_dual" in strict_oof:
        st_aft_clean = _fit_clean_stacker(
            ["cpu_aft_dual", "aft_direct_dual"], st_aft_method, st_aft_params or {}, dual_st_mask, is_aft_st=True
        )

    fus_c17_dual = _pred_stacker(st_dual_clean, strict_c17)
    fus_c17_smi = _pred_stacker(st_smi_clean, strict_c17)
    fus_c17_aft = _pred_stacker(st_aft_clean, strict_c17) if st_aft_clean is not None else None

    fus_c17_alt_dual: Dict[str, np.ndarray] = {}
    fus_c17_alt_smi: Dict[str, np.ndarray] = {}
    for alt_k in spec.get("alt_case17_conditions", {}):
        alt_src = dict(strict_c17)
        alt_src.update(strict_c17_alt.get(alt_k, {}))
        fus_c17_alt_dual[alt_k] = _pred_stacker(st_dual_clean, alt_src)
        fus_c17_alt_smi[alt_k] = _pred_stacker(st_smi_clean, alt_src)

    # 3. Combine with existing strictly cluster-excluded Deep Case17 using exact frozen manifest weights
    ed = G_MANIFEST["endpoints"][ep]
    w_dual = ed["dual_modality"]["weights"]
    w_smi = ed["smiles_only"]["weights"]
    cal_dual = ed["dual_modality"].get("calibrator")
    cal_smi = ed["smiles_only"].get("calibrator")

    old_cpu_c17 = _read_tsv(CPU_DIR / ep / "case17_predictions.tsv")
    old_fus_c17 = _read_tsv(FUSION_DIR / ep / "case17_predictions.tsv")
    dep_c17 = _read_tsv(DEEP_DIR / ep / "case17_predictions.tsv")
    old_champ_c17 = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "case17_champion.tsv")

    old_champ_dual = _col_arr(old_champ_c17, "pred_dual")
    old_champ_smi = _col_arr(old_champ_c17, "pred_smiles_only")

    def _combine_champion(
        w_map: Dict[str, float],
        cal_spec: Optional[Dict[str, Any]],
        new_cpu_arr: np.ndarray,
        new_fus_arr: np.ndarray,
        old_cpu_arr: np.ndarray,
        old_fus_arr: np.ndarray,
        dep_arr: np.ndarray,
        old_champ_arr: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        # Handle half_life_reg/cls which split deep into deep_specialist + deep_multitask
        if "deep_specialist" in w_map or "deep_multitask" in w_map:
            wc = float(w_map.get("cpu", 0.0))
            wf = float(w_map.get("fusion", 0.0))
            if task_type == "regression":
                # Exact linear residual extraction of deep_specialist + deep_multitask contribution
                deep_part = old_champ_arr - wc * old_cpu_arr - wf * old_fus_arr
                new_raw = wc * new_cpu_arr + wf * new_fus_arr + deep_part
                if bounds is not None:
                    new_raw = np.clip(new_raw, bounds[0], bounds[1])
                return new_raw, 0.0
            else:
                # Classification fallback: use combined deep stream
                streams_new = {"cpu": new_cpu_arr, "fusion": new_fus_arr, "deep": dep_arr}
                w_collapse = {
                    "cpu": wc,
                    "fusion": wf,
                    "deep": float(w_map.get("deep_specialist", 0.0)) + float(w_map.get("deep_multitask", 0.0)),
                }
                new_raw = _apply_simplex_blend(list(w_collapse.keys()), streams_new, w_collapse, task_type, bounds)
                return _apply_calibrator(new_raw, cal_spec), 0.0

        # Standard 3-family blend + parity check against old_champ_arr!
        old_streams = {"cpu": old_cpu_arr, "fusion": old_fus_arr, "deep": dep_arr}
        repro_raw = _apply_simplex_blend(list(w_map.keys()), old_streams, w_map, task_type, bounds)
        repro_final = _apply_calibrator(repro_raw, cal_spec) if task_type == "classification" else repro_raw
        parity_err = float(np.nanmax(np.abs(repro_final - old_champ_arr)))

        new_streams = {"cpu": new_cpu_arr, "fusion": new_fus_arr, "deep": dep_arr}
        new_raw = _apply_simplex_blend(list(w_map.keys()), new_streams, w_map, task_type, bounds)
        new_final = _apply_calibrator(new_raw, cal_spec) if task_type == "classification" else new_raw
        return new_final, parity_err

    new_champ_dual, parity_dual = _combine_champion(
        w_dual, cal_dual,
        cpu_dual_res["c17"], fus_c17_dual,
        _col_arr(old_cpu_c17, "pred_dual"), _col_arr(old_fus_c17, "pred_dual"),
        _col_arr(dep_c17, "pred_dual"), old_champ_dual,
    )
    new_champ_smi, parity_smi = _combine_champion(
        w_smi, cal_smi,
        cpu_smi_res["c17"], fus_c17_smi,
        _col_arr(old_cpu_c17, "pred_smiles_only"), _col_arr(old_fus_c17, "pred_smiles_only"),
        _col_arr(dep_c17, "pred_smiles_only"), old_champ_smi,
    )

    # Censored head on Case17 (if present in manifest)
    new_champ_cens = None
    old_champ_cens = _col_arr(old_champ_c17, "pred_censored_dual")
    if ed.get("censored_head") is not None and ed["censored_head"].get("dual_modality") is not None:
        cw = ed["censored_head"]["dual_modality"]["weights"]
        c_streams = {
            "champion_exact": new_champ_dual,
            "cpu_aft": cpu_aft_dual_res["c17"] if cpu_aft_dual_res is not None else new_champ_dual,
            "fusion_aft": fus_c17_aft if fus_c17_aft is not None else new_champ_dual,
            "deep": _col_arr(dep_c17, "pred_dual"),
        }
        new_champ_cens = _apply_simplex_blend(list(cw.keys()), c_streams, cw, "regression", bounds)

    # Alternate clinical routes (e.g., SC for half_life_reg & F)
    new_alt_dual: Dict[str, np.ndarray] = {}
    old_alt_dual: Dict[str, np.ndarray] = {}
    for alt_k in spec.get("alt_case17_conditions", {}):
        oc_a = _col_arr(old_cpu_c17, f"pred_dual_{alt_k}")
        of_a = _col_arr(old_fus_c17, f"pred_dual_{alt_k}")
        od_a = _col_arr(dep_c17, f"pred_dual_{alt_k}")
        och_a = _col_arr(old_champ_c17, f"pred_dual_{alt_k}")
        if oc_a is None:
            oc_a = _col_arr(old_cpu_c17, "pred_dual")
        if of_a is None:
            of_a = _col_arr(old_fus_c17, "pred_dual")
        if od_a is None:
            od_a = _col_arr(dep_c17, "pred_dual")
        if och_a is None:
            och_a = old_champ_dual
        old_alt_dual[alt_k] = och_a
        na, _ = _combine_champion(
            w_dual, cal_dual,
            cpu_dual_res["c17_alt"][alt_k], fus_c17_alt_dual[alt_k],
            oc_a, of_a, od_a, och_a,
        )
        new_alt_dual[alt_k] = na

    max_diff = float(np.nanmax(np.abs(new_champ_dual - old_champ_dual)))
    mean_diff = float(np.nanmean(np.abs(new_champ_dual - old_champ_dual)))
    print(
        f"  [Step1 {ep:<24}] ex1={n_ex1:3d} -> strict_excl={n_strict_excl:3d} (+{n_strict_excl-n_ex1:3d}) | "
        f"parity={parity_dual:.5f} | mean|delta|={mean_diff:.4f} max|delta|={max_diff:.4f} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {
        "endpoint": ep,
        "n_total": n_tot,
        "n_old_exact_excluded": n_ex1,
        "n_strict_cluster_excluded": n_strict_excl,
        "extra_analog_rows_removed": n_strict_excl - n_ex1,
        "n_train_strict": len(keep_rows),
        "parity_err_dual": round(parity_dual, 6),
        "parity_err_smi": round(parity_smi, 6),
        "mean_abs_delta_dual": round(mean_diff, 5),
        "max_abs_delta_dual": round(max_diff, 5),
        "clean_fusion_dual_weights": st_dual_clean.weight_dict(),
        "old_fusion_dual_weights": old_fusion_dual_w,
        "old_dual": [round(float(x), 6) for x in old_champ_dual],
        "strict_dual": [round(float(x), 6) for x in new_champ_dual],
        "old_smi": [round(float(x), 6) for x in old_champ_smi],
        "strict_smi": [round(float(x), 6) for x in new_champ_smi],
        "old_cens": [round(float(x), 6) for x in old_champ_cens] if old_champ_cens is not None else None,
        "strict_cens": [round(float(x), 6) for x in new_champ_cens] if new_champ_cens is not None else None,
        "old_alt_dual": {k: [round(float(x), 6) for x in v] for k, v in old_alt_dual.items()},
        "strict_alt_dual": {k: [round(float(x), 6) for x in v] for k, v in new_alt_dual.items()},
    }


def _agg_metrics(yt: List[float], yp: List[float]) -> Dict[str, Any]:
    a = np.asarray(yt, dtype=np.float64)
    b = np.asarray(yp, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mae": round(float(mean_absolute_error(a, b)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(a, b))), 4),
        "r2": round(float(r2_score(a, b)), 4),
        "spearman": round(float(spearmanr(a, b).statistic), 4),
        "pearson": round(float(pearsonr(a, b).statistic), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global G_CPU_BANK, G_FUS_BANK, G_ALL_C17_CLUSTERS, G_STRICT_META_CACHE, G_MANIFEST
    print("[Step 1] Loading banks & identifying all 19 Case17 clusters across 24 v33.0 endpoints...", flush=True)
    G_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
    G_CPU_BANK = FeatureBank()
    G_FUS_BANK = FusionBank()

    c17_keys = {q["compound_key"] for q in G_CPU_BANK.case17_queries}
    c17_idx = np.array([q["compound_idx"] for q in G_CPU_BANK.case17_queries], dtype=np.int32)
    for ep in ALL_ENDPOINTS:
        rows = _read_tsv(DATA_DIR / "endpoints" / f"{ep}.tsv")
        for r in rows:
            if int(r["case17_excluded"]) == 1 or r["compound_key"] in c17_keys:
                G_ALL_C17_CLUSTERS.add(r["cluster_id"])
    print(f"[Step 1] Identified {len(G_ALL_C17_CLUSTERS)} unique Case17 clusters (expected 19).", flush=True)
    assert len(G_ALL_C17_CLUSTERS) == 19, f"Expected 19 Case17 clusters, got {len(G_ALL_C17_CLUSTERS)}"

    # Build strict cluster-excluded meta cache for the 5 AUX_ENDPOINTS first (with n_jobs=12 for speed)!
    print(f"[Step 1] Pre-fitting 5 AUX_ENDPOINTS strictly excluding all 19 Case17 clusters: {AUX_ENDPOINTS}", flush=True)
    for aux_ep in AUX_ENDPOINTS:
        t_aux = time.time()
        ep_data = _load_ep_data(G_CPU_BANK, aux_ep)
        spec = ENDPOINT_SPECS[aux_ep]
        keep_rows = _strict_keep_indices(ep_data, G_ALL_C17_CLUSTERS, c17_idx)
        fids = assign_balanced_cluster_folds(ep_data["cluster_ids"][keep_rows], n_folds=5)
        cpu_m = json.loads((CPU_DIR / aux_ep / "metrics.json").read_text(encoding="utf-8"))
        c_bundle = joblib.load(CPU_DIR / aux_ep / "models_5fold.joblib")
        cal_map = {tr: c_bundle[tr].get("calibrator") for tr in ("dual_modality", "smiles_only")}
        del c_bundle
        G_STRICT_META_CACHE[aux_ep] = {}
        for tr, rec_k in (("dual_modality", "winning_recipe_dual"), ("smiles_only", "winning_recipe_smiles_only")):
            res = _fit_cpu_strict_5fold(
                G_CPU_BANK, ep_data, spec, keep_rows, fids, tr, cpu_m[rec_k], c17_idx, is_aft=False, n_jobs=12
            )
            res["calibrator"] = cal_map[tr]
            G_STRICT_META_CACHE[aux_ep][tr] = res
        print(f"  [AUX {aux_ep}] done in {time.time()-t_aux:.1f}s", flush=True)
    print(f"[Step 1] Strict AUX_ENDPOINTS meta cache ready ({time.time()-t_start:.1f}s).", flush=True)

    # Target endpoints: all 13 Case17 scorecard endpoints (exact scorecard audit, in-process zero-fork!)
    target_eps = list(SCORECARD_ENDPOINTS)
    print(f"[Step 1] Running strict cluster-excluded refit on {len(target_eps)} Case17 scorecard endpoints (in-process, n_jobs=16)...", flush=True)

    ep_results: Dict[str, Dict[str, Any]] = {}
    for ep in target_eps:
        ep_results[ep] = _run_one_endpoint(ep, n_jobs=16)

    # Assemble Old vs Strict 17-FDA-Drug Scorecard!
    queries = G_CPU_BANK.case17_queries
    hl_true, hl_old_ml, hl_new_ml, hl_old_anch, hl_new_anch = [], [], [], [], []
    ppb_true, ppb_old_d, ppb_new_d, ppb_old_c, ppb_new_c = [], [], [], [], []
    drug_rows = []

    for i, q in enumerate(queries):
        dn = q["drug_name"]
        route = CASE17_CLINICAL_ROUTE.get(dn, "IV")
        fda = CASE17_FDA_NUMERIC.get(dn, {})

        hl_old_iv = ep_results["half_life_reg"]["old_dual"][i]
        hl_old_sc = ep_results["half_life_reg"]["old_alt_dual"].get("SC", ep_results["half_life_reg"]["old_dual"])[i]
        hl_old_rm = hl_old_sc if route == "SC" else hl_old_iv

        hl_new_iv = ep_results["half_life_reg"]["strict_dual"][i]
        hl_new_sc = ep_results["half_life_reg"]["strict_alt_dual"].get("SC", ep_results["half_life_reg"]["strict_dual"])[i]
        hl_new_rm = hl_new_sc if route == "SC" else hl_new_iv

        anch = compute_halflife_mechanism_anchor(q["canonical_smiles"])
        hl_old_hyb = anch["anchor_mid_h"] if anch["anchor_mid_h"] is not None else (10.0 ** hl_old_rm)
        hl_new_hyb = anch["anchor_mid_h"] if anch["anchor_mid_h"] is not None else (10.0 ** hl_new_rm)

        ppb_o = ep_results["PPB"]["old_dual"][i]
        ppb_n = ep_results["PPB"]["strict_dual"][i]
        ppb_oc = ep_results["PPB"]["old_cens"][i] if ep_results["PPB"]["old_cens"] else ppb_o
        ppb_nc = ep_results["PPB"]["strict_cens"][i] if ep_results["PPB"]["strict_cens"] else ppb_n

        hem_o = ep_results["hemolysis_cls"]["old_dual"][i]
        hem_n = ep_results["hemolysis_cls"]["strict_dual"][i]

        if fda.get("half_life_h") is not None:
            hl_true.append(math.log10(fda["half_life_h"]))
            hl_old_ml.append(hl_old_rm)
            hl_new_ml.append(hl_new_rm)
            hl_old_anch.append(math.log10(hl_old_hyb))
            hl_new_anch.append(math.log10(hl_new_hyb))
        if fda.get("ppb") is not None:
            ppb_true.append(fda["ppb"])
            ppb_old_d.append(ppb_o)
            ppb_new_d.append(ppb_n)
            ppb_old_c.append(ppb_oc)
            ppb_new_c.append(ppb_nc)

        drug_rows.append({
            "drug_name": dn,
            "clinical_route": route,
            "fda_half_life_h": fda.get("half_life_h"),
            "old_hl_rm_h": round(10.0 ** hl_old_rm, 3),
            "strict_hl_rm_h": round(10.0 ** hl_new_rm, 3),
            "delta_hl_log10": round(hl_new_rm - hl_old_rm, 4),
            "fda_ppb": fda.get("ppb"),
            "old_ppb": round(ppb_o, 4),
            "strict_ppb": round(ppb_n, 4),
            "delta_ppb": round(ppb_n - ppb_o, 4),
            "old_hemolysis_prob": round(hem_o, 4),
            "strict_hemolysis_prob": round(hem_n, 4),
        })

    old_hem_max = max(r["old_hemolysis_prob"] for r in drug_rows)
    new_hem_max = max(r["strict_hemolysis_prob"] for r in drug_rows)
    summary = {
        "schema": "v333_step1_strict_case17_audit_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "n_case17_clusters_excluded": len(G_ALL_C17_CLUSTERS),
        "case17_clusters": sorted(G_ALL_C17_CLUSTERS),
        "n_endpoints_audited": len(ep_results),
        "scorecard_comparison": {
            "half_life_pure_ml_log10": {
                "old_exact_only": _agg_metrics(hl_true, hl_old_ml),
                "strict_full_cluster": _agg_metrics(hl_true, hl_new_ml),
            },
            "half_life_with_mechanism_anchor_log10": {
                "old_exact_only": _agg_metrics(hl_true, hl_old_anch),
                "strict_full_cluster": _agg_metrics(hl_true, hl_new_anch),
            },
            "ppb_dual": {
                "old_exact_only": _agg_metrics(ppb_true, ppb_old_d),
                "strict_full_cluster": _agg_metrics(ppb_true, ppb_new_d),
            },
            "ppb_censored": {
                "old_exact_only": _agg_metrics(ppb_true, ppb_old_c),
                "strict_full_cluster": _agg_metrics(ppb_true, ppb_new_c),
            },
            "hemolysis_false_positive_check": {
                "old_max_prob": round(old_hem_max, 4),
                "old_n_gt_0_50": sum(1 for r in drug_rows if r["old_hemolysis_prob"] > 0.5),
                "strict_max_prob": round(new_hem_max, 4),
                "strict_n_gt_0_50": sum(1 for r in drug_rows if r["strict_hemolysis_prob"] > 0.5),
            },
        },
        "drugs": drug_rows,
        "endpoints": ep_results,
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    out_path = OUT_DIR / "step1_case17_strict_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Step 1 DONE] Wrote {out_path} in {summary['elapsed_sec']}s", flush=True)
    print(json.dumps(summary["scorecard_comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
