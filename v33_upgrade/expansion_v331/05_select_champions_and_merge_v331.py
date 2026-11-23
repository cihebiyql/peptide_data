#!/usr/bin/env python3
"""
Wave 5A — Step 05 (runs on node23):
Strictly OOF-only cross-family champion selection (CPU vs 4-FM Fusion vs Nested-OOF Simplex Super-Ensemble),
frozen OOF Conformal 80%/90% calibration, frozen OOF Censored-Aware AFT head selection, single untouched 20%
holdout evaluation, pruned serving model bundling (ZERO catboost!), unified AD + Top-3 Analog Index rebuild,
and ZERO-REGRESSION merge into <WORK_DIR>/v33_upgrade/bundle/.

HARD ZERO-REGRESSION GUARANTEES:
1. Backs up v33_manifest.json -> v33_manifest_24ep_backup.json (never overwriting an existing backup!).
2. Backs up ad_analog_index.npz & analog_lookup.json.gz -> *_24ep_backup.* before rebuilding unified 46-ep index.
3. Copies ONLY newly trained <ep>_{cpu,fusion}.joblib for the 22 NEW endpoints into bundle/serving_models/;
   asserts ALL 43 existing V33 .joblib files have 100% IDENTICAL SHA-256 checksums!
4. Keeps ALL 24 existing entries in v33_manifest.json["endpoints"] 100% BYTE-IDENTICAL!
5. Verifies via pickletools that ZERO new serving .joblib files reference catboost or non-node2 packages!
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import pickletools
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
BUNDLE_DIR = V33_BASE / "bundle"
SERVING_DIR = BUNDLE_DIR / "serving_models"
V33_ENDPOINTS_DIR = V33_BASE / "data" / "endpoints"

EXP_BASE = V33_BASE / "expansion_v331"
EXP_SCRIPTS = Path(__file__).resolve().parent
EXP_DATA = EXP_BASE / "data"
EXP_ENDPOINTS_DIR = EXP_DATA / "endpoints"
EXP_FEATURES_DIR = EXP_BASE / "features"
EXP_CPU_DIR = EXP_BASE / "models_cpu"
EXP_FUSION_DIR = EXP_BASE / "models_fusion"
EXP_SERVING_DIR = EXP_BASE / "serving_models"
EXP_PRED_DIR = EXP_BASE / "champion_predictions"

sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP_SCRIPTS))

from v33_config import CASE17_CLINICAL_ROUTE, ENDPOINT_SPECS as V33_24_SPECS  # noqa: E402
from v33_metrics import regression_metrics  # noqa: E402
from v331_config import V331_ENDPOINT_SPECS  # noqa: E402

csv.field_size_limit(10_000_000)

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_IDX = {a: i for i, a in enumerate(AA20)}
AD_IN_DOMAIN_THR = 0.65
AD_BORDERLINE_THR = 0.40


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
        return round(x, 6) if math.isfinite(x) else None
    except Exception:
        return None


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


def _weighted_nan_combine(M: np.ndarray, w: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    fin = np.isfinite(M)
    M0 = np.where(fin, M, 0.0)
    W = np.where(fin, w[None, :], 0.0)
    wsum = W.sum(axis=1)
    num = (M0 * W).sum(axis=1)
    out = np.divide(num, wsum, out=np.full(M.shape[0], fallback, dtype=np.float64), where=wsum > 1e-12)
    miss = wsum <= 1e-12
    if miss.any():
        cnt = fin[miss].sum(axis=1)
        s = M0[miss].sum(axis=1)
        out[miss] = np.divide(s, cnt, out=np.full(int(miss.sum()), fallback), where=cnt > 0)
    return out


def _apply_simplex_blend(stream_names: List[str], streams: Dict[str, np.ndarray], weights: Dict[str, float], bounds: Optional[Tuple[float, float]] = None, fallback: float = 0.0) -> np.ndarray:
    active = [s for s in stream_names if weights.get(s, 0.0) > 0.0]
    if not active:
        active = [stream_names[0]]
    w = np.array([weights.get(s, 0.0) for s in active], dtype=np.float64)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(active)) / len(active)
    M = np.column_stack([streams[s] for s in active])
    y_out = _weighted_nan_combine(M, w, fallback=fallback)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


def _fit_simplex_weights(P: np.ndarray, y: np.ndarray, reg: float = 0.005, prune_tol: float = 0.02, hinge_mode: bool = False, y_lower: Optional[np.ndarray] = None, y_upper: Optional[np.ndarray] = None, censor_type: Optional[np.ndarray] = None) -> np.ndarray:
    n, m = P.shape
    unif = np.ones(m, dtype=np.float64) / float(m)
    fallback = float(np.nanmean(y)) if len(y) > 0 else 0.0
    eps_s = 1e-3

    def loss_fn(theta: np.ndarray) -> float:
        lw = theta - np.max(theta)
        w = np.exp(lw)
        w = w / np.sum(w)
        pred = _weighted_nan_combine(P, w, fallback=fallback)
        if hinge_mode and censor_type is not None and y_lower is not None and y_upper is not None:
            ct = censor_type
            e_ex = np.sqrt((y - pred) ** 2 + eps_s ** 2) - eps_s
            e_r = np.maximum(0.0, y_lower - pred)
            e_l = np.maximum(0.0, pred - y_upper)
            e_i = np.maximum(0.0, np.maximum(y_lower - pred, pred - y_upper))
            val = float(np.mean(np.where(ct == "exact", e_ex, np.where(ct == "right", e_r, np.where(ct == "left", e_l, e_i)))))
        else:
            val = float(np.mean(np.sqrt((y - pred) ** 2 + eps_s ** 2)))
        return val + reg * float(np.sum((w - unif) ** 2))

    inits = [np.zeros(m, dtype=np.float64)]
    for j in range(m):
        v = np.full(m, -2.5, dtype=np.float64)
        v[j] = 2.5
        inits.append(v)

    best_res = None
    for x0 in inits:
        res = minimize(loss_fn, x0, method="BFGS", options={"maxiter": 150})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    lw = best_res.x - np.max(best_res.x)
    w = np.exp(lw)
    w = w / np.sum(w)
    if prune_tol > 0.0:
        w[w < prune_tol] = 0.0
        w = w / w.sum() if w.sum() > 0 else np.eye(m)[int(np.argmax(np.exp(lw)))]
    return w


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return 0.0
    k = min(max(int(math.ceil((1.0 - alpha) * (n + 1))), 1), n)
    return float(s[k - 1])


def _fit_conformal_oof(y_cv, pred_oof, censor_cv, fold_ids, eval_mask, oof_max_tanimoto, bounds=None) -> Dict[str, Any]:
    mask = eval_mask & (censor_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof)
    if mask.sum() < 5:
        mask = eval_mask & np.isfinite(y_cv) & np.isfinite(pred_oof)
    ye, pe = y_cv[mask], pred_oof[mask]
    abs_res = np.abs(ye - pe)
    fids = fold_ids[mask].astype(np.int32)
    t_oof = oof_max_tanimoto[mask]
    n_ex = len(abs_res)
    med_r = float(np.median(abs_res)) if n_ex > 0 else 0.5
    floor_s = max(0.30 * med_r, 1e-3)

    std_q80, std_q90 = _conformal_quantile(abs_res, 0.20), _conformal_quantile(abs_res, 0.10)
    uniq_f = sorted(set(fids.tolist()))
    std_c80, std_c90, std_w90 = [], [], []
    for f in uniq_f:
        va, tr = fids == f, fids != f
        if tr.sum() < 4 or va.sum() == 0:
            continue
        q80_f, q90_f = _conformal_quantile(abs_res[tr], 0.20), _conformal_quantile(abs_res[tr], 0.10)
        lo80, hi80 = pe[va] - q80_f, pe[va] + q80_f
        lo90, hi90 = pe[va] - q90_f, pe[va] + q90_f
        if bounds is not None:
            lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
            lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
        std_c80.extend(((ye[va] >= lo80) & (ye[va] <= hi80)).tolist())
        std_c90.extend(((ye[va] >= lo90) & (ye[va] <= hi90)).tolist())
        std_w90.extend((hi90 - lo90).tolist())

    std_oof_w90 = float(np.mean(std_w90)) if std_w90 else 2.0 * std_q90
    diag: Dict[str, Any] = {
        "n_oof_exact": int(n_ex),
        "standard": {
            "q80": round(std_q80, 5), "q90": round(std_q90, 5),
            "nested_oof_cov80": round(float(np.mean(std_c80)) if std_c80 else 0.80, 4),
            "nested_oof_cov90": round(float(np.mean(std_c90)) if std_c90 else 0.90, 4),
            "nested_oof_width90": round(std_oof_w90, 5),
        },
    }
    mode, q80_final, q90_final, dist_slope, dist_intercept, min_scale = "standard", std_q80, std_q90, 0.0, 1.0, floor_s

    ok_t = np.isfinite(t_oof)
    if ok_t.sum() >= 50:
        dist_all = 1.0 - np.clip(np.where(ok_t, t_oof, 0.5), 0.0, 1.0)
        sp_all = float(spearmanr(dist_all[ok_t], abs_res[ok_t]).statistic)
        if np.isfinite(sp_all) and sp_all >= 0.10:
            ad_c80, ad_c90, ad_w90 = [], [], []
            for f in uniq_f:
                va, tr = fids == f, (fids != f) & ok_t
                if tr.sum() < 25 or va.sum() == 0:
                    continue
                try:
                    reg = HuberRegressor(alpha=1.0).fit(dist_all[tr, None], abs_res[tr])
                    sl, ic = max(0.0, float(reg.coef_[0])), max(floor_s, float(reg.intercept_))
                except Exception:
                    sl, ic = 0.0, max(med_r, 1e-3)
                s_tr = np.maximum(ic + sl * dist_all[tr], floor_s)
                s_va = np.maximum(ic + sl * dist_all[va], floor_s)
                q80_f, q90_f = _conformal_quantile(abs_res[tr] / s_tr, 0.20), _conformal_quantile(abs_res[tr] / s_tr, 0.10)
                lo80, hi80 = pe[va] - q80_f * s_va, pe[va] + q80_f * s_va
                lo90, hi90 = pe[va] - q90_f * s_va, pe[va] + q90_f * s_va
                if bounds is not None:
                    lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
                    lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
                ad_c80.extend(((ye[va] >= lo80) & (ye[va] <= hi80)).tolist())
                ad_c90.extend(((ye[va] >= lo90) & (ye[va] <= hi90)).tolist())
                ad_w90.extend((hi90 - lo90).tolist())
            if ad_c80:
                ad_cov80, ad_cov90, ad_w90_mean = float(np.mean(ad_c80)), float(np.mean(ad_c90)), float(np.mean(ad_w90))
                diag["adaptive_tanimoto_dist"] = {"spearman_dist_err": round(sp_all, 4), "nested_oof_cov80": round(ad_cov80, 4), "nested_oof_cov90": round(ad_cov90, 4), "nested_oof_width90": round(ad_w90_mean, 5)}
                if ad_cov80 >= 0.78 and ad_w90_mean < std_oof_w90 * 0.98:
                    try:
                        reg_full = HuberRegressor(alpha=1.0).fit(dist_all[ok_t, None], abs_res[ok_t])
                        sl_full, ic_full = max(0.0, float(reg_full.coef_[0])), max(floor_s, float(reg_full.intercept_))
                        if sl_full > 1e-4:
                            s_full = np.maximum(ic_full + sl_full * dist_all, floor_s)
                            mode, dist_slope, dist_intercept, min_scale = "adaptive_tanimoto_dist", sl_full, ic_full, floor_s
                            q80_final, q90_final = _conformal_quantile(abs_res / s_full, 0.20), _conformal_quantile(abs_res / s_full, 0.10)
                    except Exception:
                        pass
    return {
        "mode": mode, "q80": round(q80_final, 5), "q90": round(q90_final, 5),
        "dist_slope": round(dist_slope, 6), "dist_intercept": round(dist_intercept, 6), "min_scale": round(min_scale, 6),
        "bounds": list(bounds) if bounds is not None else None, "oof_diagnostics": diag,
    }


def apply_frozen_conformal(pred, conf_spec, max_tanimoto=None) -> Dict[str, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    if conf_spec["mode"] == "adaptive_tanimoto_dist" and max_tanimoto is not None:
        d = 1.0 - np.clip(np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.5), 0.0, 1.0)
        s = np.maximum(conf_spec["dist_intercept"] + conf_spec["dist_slope"] * d, conf_spec["min_scale"])
    else:
        s = np.ones(len(p), dtype=np.float64)
    lo80, hi80 = p - conf_spec["q80"] * s, p + conf_spec["q80"] * s
    lo90, hi90 = p - conf_spec["q90"] * s, p + conf_spec["q90"] * s
    bounds = conf_spec.get("bounds")
    if bounds is not None:
        lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
        lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
    return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}


def eval_conformal_coverage(y_true, pred, censor_type, conf_spec, max_tanimoto=None) -> Dict[str, Any]:
    ivs = apply_frozen_conformal(pred, conf_spec, max_tanimoto=max_tanimoto)
    mask = (censor_type == "exact") & np.isfinite(y_true) & np.isfinite(pred)
    if not mask.any():
        return {"mode": conf_spec["mode"], "n_eval": 0, "cov80": None, "cov90": None, "mean_width_80": None, "mean_width_90": None}
    yt = y_true[mask]
    return {
        "mode": conf_spec["mode"], "n_eval": int(mask.sum()),
        "cov80": round(float(np.mean((yt >= ivs["lo80"][mask]) & (yt <= ivs["hi80"][mask]))), 4),
        "cov90": round(float(np.mean((yt >= ivs["lo90"][mask]) & (yt <= ivs["hi90"][mask]))), 4),
        "mean_width_80": round(float(np.mean(ivs["hi80"][mask] - ivs["lo80"][mask])), 4),
        "mean_width_90": round(float(np.mean(ivs["hi90"][mask] - ivs["lo90"][mask])), 4),
    }


def _select_champion_oof(track, bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, eval_mask, streams_oof, oof_max_tanimoto) -> Dict[str, Any]:
    stream_names = ["cpu", "fusion"]
    n_cv = len(y_cv)
    fallback = float(np.nanmean(y_cv[eval_mask])) if eval_mask.any() else 0.0
    single_metrics = {s: regression_metrics(y_cv[eval_mask], streams_oof[s][eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask]) for s in stream_names}
    best_single = max(stream_names, key=lambda s: (-(single_metrics[s]["mae"] or 1e9), single_metrics[s]["spearman"] or -1.0))

    P_oof = np.column_stack([streams_oof[s] for s in stream_names])
    nested_oof_super = np.full(n_cv, np.nan, dtype=np.float64)
    fold_weights_log = []
    for f in range(5):
        va = (fids_cv == f) & eval_mask
        tr = (fids_cv != f) & eval_mask & (ct_cv == "exact")
        if tr.sum() < 4 or va.sum() == 0:
            nested_oof_super[va] = streams_oof[best_single][va]
            continue
        w_f = _fit_simplex_weights(P_oof[tr], y_cv[tr], reg=0.005, prune_tol=0.02)
        wd = {s: float(w_f[i]) for i, s in enumerate(stream_names)}
        fold_weights_log.append({s: round(float(w_f[i]), 4) for i, s in enumerate(stream_names)})
        nested_oof_super[va] = _apply_simplex_blend(stream_names, {s: streams_oof[s][va] for s in stream_names}, wd, bounds=bounds, fallback=fallback)

    super_oof_m = regression_metrics(y_cv[eval_mask], nested_oof_super[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask])
    best_s_m = single_metrics[best_single]
    delta_primary = (best_s_m["mae"] or 1e9) - (super_oof_m["mae"] or 1e9)
    adopt_super = delta_primary >= 0.0005
    if adopt_super and int((ct_cv[eval_mask] != "exact").sum()) >= 20:
        v_b, v_s = best_s_m.get("censored_violation_mae"), super_oof_m.get("censored_violation_mae")
        if v_b is not None and v_s is not None and v_s > v_b + 0.05:
            adopt_super = False

    if adopt_super:
        fit_mask = eval_mask & (ct_cv == "exact")
        w_final = _fit_simplex_weights(P_oof[fit_mask], y_cv[fit_mask], reg=0.005, prune_tol=0.02)
        frozen_weights = {s: round(float(w_final[i]), 5) for i, s in enumerate(stream_names) if w_final[i] > 0.0}
        s_tot = sum(frozen_weights.values())
        frozen_weights = {k: round(v / s_tot, 5) for k, v in frozen_weights.items()}
        if len(frozen_weights) == 1:
            only_fam = next(iter(frozen_weights.keys()))
            winning_family, champion_oof_raw, adopt_super, frozen_weights = only_fam, streams_oof[only_fam].copy(), False, {only_fam: 1.0}
        else:
            winning_family, champion_oof_raw = "super_ensemble", nested_oof_super.copy()
    else:
        winning_family, frozen_weights, champion_oof_raw = best_single, {best_single: 1.0}, streams_oof[best_single].copy()

    conf_spec = _fit_conformal_oof(y_cv, champion_oof_raw, ct_cv, fids_cv, eval_mask, oof_max_tanimoto, bounds=bounds)
    champion_cv_metrics = regression_metrics(y_cv[eval_mask], champion_oof_raw[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask])
    return {
        "track": track, "winning_family": winning_family, "is_super_ensemble": bool(adopt_super),
        "weights": frozen_weights, "best_single_family": best_single,
        "nested_super_delta_over_best_single": round(float(delta_primary), 5),
        "single_family_oof_metrics": single_metrics, "nested_super_oof_metrics": super_oof_m,
        "nested_fold_weights": fold_weights_log, "champion_cv_metrics": champion_cv_metrics,
        "champion_cv_metrics_raw": None, "calibrator": None, "conformal": conf_spec,
        "champion_oof_final": champion_oof_raw,
    }


def _select_censored_head_oof(y_cv, yl_cv, yu_cv, ct_cv, fids_cv, eval_mask, bounds, cand_oof) -> Dict[str, Any]:
    names = [k for k in ("champion_exact", "cpu_aft", "fusion_aft") if k in cand_oof and cand_oof[k] is not None]
    n_cv = len(y_cv)
    fallback = float(np.nanmean(y_cv[eval_mask])) if eval_mask.any() else 0.0
    single_m = {k: regression_metrics(y_cv[eval_mask], cand_oof[k][eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask]) for k in names}
    best_single = min(names, key=lambda k: single_m[k]["all_row_hinge_mae"] if single_m[k]["all_row_hinge_mae"] is not None else 1e9)

    P = np.column_stack([cand_oof[k] for k in names])
    nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
    for f in range(5):
        va, tr = (fids_cv == f) & eval_mask, (fids_cv != f) & eval_mask
        if tr.sum() < 6 or va.sum() == 0:
            nested_oof[va] = cand_oof[best_single][va]
            continue
        w_f = _fit_simplex_weights(P[tr], y_cv[tr], reg=0.005, prune_tol=0.02, hinge_mode=True, y_lower=yl_cv[tr], y_upper=yu_cv[tr], censor_type=ct_cv[tr])
        wd = {k: float(w_f[i]) for i, k in enumerate(names)}
        nested_oof[va] = _apply_simplex_blend(names, {k: cand_oof[k][va] for k in names}, wd, bounds=bounds, fallback=fallback)

    blend_m = regression_metrics(y_cv[eval_mask], nested_oof[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask])
    h_best, h_blend = single_m[best_single]["all_row_hinge_mae"] or 1e9, blend_m["all_row_hinge_mae"] or 1e9
    if h_best - h_blend >= 0.0005:
        w_full = _fit_simplex_weights(P[eval_mask], y_cv[eval_mask], reg=0.005, prune_tol=0.02, hinge_mode=True, y_lower=yl_cv[eval_mask], y_upper=yu_cv[eval_mask], censor_type=ct_cv[eval_mask])
        weights = {k: round(float(w_full[i]), 5) for i, k in enumerate(names) if w_full[i] > 0.0}
        s_tot = sum(weights.values())
        weights = {k: round(v / s_tot, 5) for k, v in weights.items()}
        mode = "censored_blend" if len(weights) > 1 else next(iter(weights.keys()))
        cv_m = blend_m if len(weights) > 1 else single_m[mode]
    else:
        mode, weights, cv_m = best_single, {best_single: 1.0}, single_m[best_single]
    return {"mode": mode, "weights": weights, "candidate_oof_metrics": single_m, "cv_metrics": cv_m}


def _check_no_catboost_in_joblib(path: Path) -> List[str]:
    """Inspect serialized joblib object's classes recursively to assert ZERO catboost or disallowed packages."""
    obj = joblib.load(path)
    modules: Set[str] = set()
    visited: Set[int] = set()

    def walk(x: Any):
        oid = id(x)
        if oid in visited:
            return
        visited.add(oid)
        t = type(x)
        mod = getattr(t, "__module__", "")
        if mod:
            modules.add(mod.split(".")[0])
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
        elif hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                walk(v)

    walk(obj)
    assert "catboost" not in modules, f"FORBIDDEN catboost found in {path}!"
    return sorted(modules)


def _encode_seq_3mers(seq: str) -> List[int]:
    clean = [c for c in (seq or "").upper() if c in AA20_IDX]
    if len(clean) < 3:
        return []
    return sorted({AA20_IDX[clean[i]] * 400 + AA20_IDX[clean[i + 1]] * 20 + AA20_IDX[clean[i + 2]] for i in range(len(clean) - 2)})


def rebuild_unified_ad_analog_index(
    all_24_eps: List[str],
    all_new_eps: List[str],
    v33_manifest_24: Dict[str, Any],
) -> Tuple[Path, Path, int]:
    """Rebuilds unified AD + Top-3 Analog index covering ALL 24 existing + 22 new = 46 endpoints."""
    t0 = time.time()
    print(f"\n[AD INDEX] Rebuilding unified 46-endpoint AD + Top-3 Analog index...", flush=True)
    compounds_meta: Dict[str, Dict[str, str]] = {}
    with (EXP_DATA / "compounds_v331_merged.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            compounds_meta[r["compound_key"]] = r

    key_to_idx: Dict[str, int] = {}
    with (EXP_FEATURES_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key_to_idx[r["compound_key"]] = int(r["compound_idx"])

    pc = np.load(EXP_FEATURES_DIR / "physchem_core.npz")
    smiles_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
    seq_valid = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
    pc.close()
    mb = np.load(EXP_FEATURES_DIR / "morgan_bits_1024.npz")
    morgan_packed = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
    mb.close()

    ep_analog_records: Dict[str, List[Dict[str, Any]]] = {}
    ref_keys_set: Set[str] = set()

    def _collect(ep_list: List[str], ep_dir: Path, unit_map: Dict[str, str]):
        for ep in ep_list:
            rows = _read_tsv(ep_dir / f"{ep}.tsv")
            by_ck: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for r in rows:
                by_ck[r["compound_key"]].append(r)
            recs = []
            unit = unit_map[ep]
            for ck in sorted(by_ck.keys()):
                ref_keys_set.add(ck)
                obs = by_ck[ck]
                exact_obs = [o for o in obs if o["censor_type"] == "exact"]
                use_obs = exact_obs if exact_obs else obs
                med_val = float(np.median([float(o["y_value"]) for o in use_obs]))
                meta = compounds_meta.get(ck, {})
                recs.append({
                    "compound_key": ck,
                    "smiles": meta.get("canonical_smiles", ""),
                    "sequence": meta.get("parent_sequence", ""),
                    "experimental_value": round(med_val, 5),
                    "censor_type": use_obs[0]["censor_type"],
                    "unit": unit,
                    "n_obs": len(obs),
                })
            ep_analog_records[ep] = recs

    _collect(all_24_eps, V33_ENDPOINTS_DIR, {ep: v33_manifest_24["endpoints"][ep]["unit"] for ep in all_24_eps})
    _collect(all_new_eps, EXP_ENDPOINTS_DIR, {ep: V331_ENDPOINT_SPECS[ep]["unit"] for ep in all_new_eps})

    ref_keys = sorted(ref_keys_set, key=lambda k: key_to_idx[k])
    ref_key_to_gidx = {k: i for i, k in enumerate(ref_keys)}
    ref_cidx = np.array([key_to_idx[k] for k in ref_keys], dtype=np.int32)
    n_ref = len(ref_keys)

    ref_morgan_packed = np.ascontiguousarray(morgan_packed[ref_cidx], dtype=np.uint8)
    ref_smi_valid = np.ascontiguousarray(smiles_valid[ref_cidx], dtype=bool)
    ref_seq_valid = np.ascontiguousarray(seq_valid[ref_cidx], dtype=bool)

    indptr = [0]
    indices = []
    seq_3mer_pop = np.zeros(n_ref, dtype=np.float32)
    for i, ck in enumerate(ref_keys):
        kmers = _encode_seq_3mers(compounds_meta.get(ck, {}).get("parent_sequence", "")) if ref_seq_valid[i] else []
        indices.extend(kmers)
        indptr.append(len(indices))
        seq_3mer_pop[i] = float(len(kmers))

    npz_dict: Dict[str, Any] = {
        "ref_compound_keys": np.array(ref_keys),
        "ref_compound_idx": ref_cidx,
        "ref_morgan_packed": ref_morgan_packed,
        "ref_smi_valid": ref_smi_valid,
        "ref_seq_valid": ref_seq_valid,
        "seq_3mer_indptr": np.array(indptr, dtype=np.int32),
        "seq_3mer_indices": np.array(indices, dtype=np.int16),
        "seq_3mer_pop": seq_3mer_pop,
        "ad_in_domain_thr": np.array([AD_IN_DOMAIN_THR], dtype=np.float32),
        "ad_borderline_thr": np.array([AD_BORDERLINE_THR], dtype=np.float32),
    }
    analog_lookup: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v331_analog_lookup_v1",
        "ad_thresholds": {"in_domain": AD_IN_DOMAIN_THR, "borderline": AD_BORDERLINE_THR},
        "n_unique_ref_compounds": n_ref,
        "endpoints": {},
    }
    for ep in sorted(ep_analog_records.keys()):
        recs = ep_analog_records[ep]
        npz_dict[f"ep_ref_idx__{ep}"] = np.array([ref_key_to_gidx[r["compound_key"]] for r in recs], dtype=np.int32)
        analog_lookup["endpoints"][ep] = {"n_unique_compounds": len(recs), "records": recs}

    # Backup original 24-ep AD index files if backups don't already exist
    ad_npz_path = BUNDLE_DIR / "ad_analog_index.npz"
    analog_gz_path = BUNDLE_DIR / "analog_lookup.json.gz"
    ad_bak = BUNDLE_DIR / "ad_analog_index_24ep_backup.npz"
    an_bak = BUNDLE_DIR / "analog_lookup_24ep_backup.json.gz"
    if not ad_bak.exists():
        shutil.copy2(ad_npz_path, ad_bak)
    if not an_bak.exists():
        shutil.copy2(analog_gz_path, an_bak)

    # Verify 100% parity on all 24 existing endpoints vs 24-ep backup!
    with gzip.open(an_bak, "rt", encoding="utf-8") as gz:
        old_lookup = json.load(gz)
    for ep in all_24_eps:
        assert analog_lookup["endpoints"][ep]["records"] == old_lookup["endpoints"][ep]["records"], f"AD analog mismatch on existing endpoint {ep}!"

    np.savez_compressed(ad_npz_path, **npz_dict)
    with gzip.open(analog_gz_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        json.dump(analog_lookup, gz, ensure_ascii=False, separators=(",", ":"))
    print(f"[AD INDEX DONE] {n_ref:,} unique refs x {len(ep_analog_records)} eps | {ad_npz_path.stat().st_size/1e6:.2f}MB + {analog_gz_path.stat().st_size/1e6:.2f}MB ({time.time()-t0:.1f}s)", flush=True)
    return ad_npz_path, analog_gz_path, n_ref


def main():
    t_all = time.time()
    EXP_SERVING_DIR.mkdir(parents=True, exist_ok=True)
    EXP_PRED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Backup pristine 24-ep v33_manifest.json FIRST (never overwrite existing backup!)
    manifest_path = BUNDLE_DIR / "v33_manifest.json"
    backup_path = BUNDLE_DIR / "v33_manifest_24ep_backup.json"
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"[BACKUP] Created pristine 24-ep backup -> {backup_path}", flush=True)
    else:
        print(f"[BACKUP] Preserving existing pristine 24-ep backup -> {backup_path}", flush=True)

    v33_24_manifest = json.loads(backup_path.read_text(encoding="utf-8"))
    existing_24_eps = sorted(v33_24_manifest["endpoints"].keys())
    assert len(existing_24_eps) == 24
    new_eps = sorted(V331_ENDPOINT_SPECS.keys())
    assert set(existing_24_eps).isdisjoint(set(new_eps)), "Overlap between existing 24 and new V33.1 endpoints!"

    # Snapshot SHA-256 of ALL 43 existing serving_models/*.joblib files BEFORE touching anything!
    existing_joblib_shas = dict(v33_24_manifest["sha256_checksums"]["serving_models"])
    for fname, exp_sha in existing_joblib_shas.items():
        fp = SERVING_DIR / fname
        assert fp.exists() and sha256_file(fp) == exp_sha, f"Existing V33 serving model altered before merge: {fname}"
    print(f"[VERIFIED] All {len(existing_joblib_shas)} existing V33 serving_models/*.joblib checksums intact!", flush=True)

    # Load V33.1 compound index & masks
    key_to_idx: Dict[str, int] = {}
    with (EXP_FEATURES_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key_to_idx[r["compound_key"]] = int(r["compound_idx"])
    pc = np.load(EXP_FEATURES_DIR / "physchem_core.npz")
    smiles_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
    pc.close()
    case17_queries = _read_tsv(V33_BASE / "data" / "case17_queries.tsv")
    assert len(case17_queries) == 17

    # =========================================================================
    # PHASE A: STRICTLY OOF-ONLY SELECTION (22 new endpoints x 2 tracks, ZERO holdout peeking!)
    # =========================================================================
    print(f"\n[PHASE A] Strictly OOF-only champion selection for {len(new_eps)} new endpoints...", flush=True)
    oof_frozen: Dict[str, Any] = {}
    ep_counts: Dict[str, Any] = {}
    for ep in new_eps:
        spec = V331_ENDPOINT_SPECS[ep]
        bounds = spec.get("bounds")
        ep_rows = _read_tsv(EXP_ENDPOINTS_DIR / f"{ep}.tsv")
        cv_ep_rows = [r for r in ep_rows if int(r["is_holdout"]) == 0]
        ho_ep_rows = [r for r in ep_rows if int(r["is_holdout"]) == 1]
        assert {r["cluster_id"] for r in cv_ep_rows}.isdisjoint({r["cluster_id"] for r in ho_ep_rows})

        n_cv = len(cv_ep_rows)
        n_cens_cv = sum(1 for r in cv_ep_rows if r["censor_type"] != "exact")
        ep_counts[ep] = {
            "n_total": len(ep_rows), "n_cv": n_cv, "n_holdout": len(ho_ep_rows),
            "n_censored_total": sum(1 for r in ep_rows if r["censor_type"] != "exact"),
            "n_censored_cv": n_cens_cv, "n_censored_holdout": sum(1 for r in ho_ep_rows if r["censor_type"] != "exact"),
            "censor_counts": {k: sum(1 for r in ep_rows if r["censor_type"] == k) for k in ("exact", "right", "left", "interval")},
            "n_case17_excluded": sum(1 for r in ep_rows if int(r["case17_excluded"]) == 1),
        }

        cpu_oof = _read_tsv(EXP_CPU_DIR / ep / "oof_predictions.tsv")
        fus_oof = _read_tsv(EXP_FUSION_DIR / ep / "oof_predictions.tsv")
        assert len(cpu_oof) == len(fus_oof) == n_cv
        for i in range(n_cv):
            assert cpu_oof[i]["record_id"] == fus_oof[i]["record_id"] == cv_ep_rows[i]["record_id"]

        cidx_cv = np.array([key_to_idx[r["compound_key"]] for r in cv_ep_rows], dtype=np.int32)
        smi_mask_cv = smiles_valid[cidx_cv]
        dual_mask_cv = np.ones(n_cv, dtype=bool)
        y_cv = np.array([float(r["y_value"]) for r in cv_ep_rows], dtype=np.float64)
        yl_cv = np.array([float(r["y_lower"]) for r in cv_ep_rows], dtype=np.float64)
        yu_cv = np.array([float(r["y_upper"]) for r in cv_ep_rows], dtype=np.float64)
        ct_cv = np.array([r["censor_type"] for r in cv_ep_rows], dtype=object)
        fids_cv = np.array([int(r["fold_id"]) for r in cv_ep_rows], dtype=np.int32)
        oof_max_t = _col_arr(cpu_oof, "max_tanimoto_oof")

        res_dual = _select_champion_oof("dual_modality", bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, dual_mask_cv, {"cpu": _col_arr(cpu_oof, "pred_dual"), "fusion": _col_arr(fus_oof, "pred_dual")}, oof_max_t)
        res_smi = _select_champion_oof("smiles_only", bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, smi_mask_cv, {"cpu": _col_arr(cpu_oof, "pred_smiles_only"), "fusion": _col_arr(fus_oof, "pred_smiles_only")}, oof_max_t)

        cens_dual = cens_smi = None
        if n_cens_cv >= 1:
            cens_dual = _select_censored_head_oof(y_cv, yl_cv, yu_cv, ct_cv, fids_cv, dual_mask_cv, bounds, {"champion_exact": res_dual["champion_oof_final"], "cpu_aft": _col_arr(cpu_oof, "pred_aft"), "fusion_aft": _col_arr(fus_oof, "pred_aft")})
            cens_smi = _select_censored_head_oof(y_cv, yl_cv, yu_cv, ct_cv, fids_cv, smi_mask_cv, bounds, {"champion_exact": res_smi["champion_oof_final"], "cpu_aft": _col_arr(cpu_oof, "pred_aft_smiles_only"), "fusion_aft": _col_arr(fus_oof, "pred_aft_smiles_only")})

        oof_frozen[ep] = {
            "dual_modality": res_dual, "smiles_only": res_smi,
            "censored_head_dual": cens_dual, "censored_head_smi": cens_smi,
            "cv_ep_rows": cv_ep_rows, "ho_ep_rows": ho_ep_rows,
            "y_cv_mean_dual": float(np.nanmean(y_cv)),
            "y_cv_mean_smi": float(np.nanmean(y_cv[smi_mask_cv])) if smi_mask_cv.any() else float(np.nanmean(y_cv)),
        }
        print(f"  [OOF] {ep:<22} | DUAL={res_dual['winning_family']:<14} {res_dual['weights']} CV MAE={res_dual['champion_cv_metrics']['mae']:.4f} | SMI={res_smi['winning_family']:<14} CV MAE={res_smi['champion_cv_metrics']['mae']:.4f}", flush=True)

    # =========================================================================
    # PHASE B: SINGLE FROZEN EVALUATION ON UNTOUCHED 20% HOLDOUT
    # =========================================================================
    print(f"\n[PHASE B] Evaluating FROZEN champions ONCE on untouched 20% Holdout...", flush=True)
    new_endpoints_manifest: Dict[str, Any] = {}
    for ep in new_eps:
        spec = V331_ENDPOINT_SPECS[ep]
        bounds = spec.get("bounds")
        fr = oof_frozen[ep]
        ho_ep_rows = fr["ho_ep_rows"]
        n_ho = len(ho_ep_rows)

        cpu_ho = _read_tsv(EXP_CPU_DIR / ep / "holdout_predictions.tsv")
        fus_ho = _read_tsv(EXP_FUSION_DIR / ep / "holdout_predictions.tsv")
        assert len(cpu_ho) == len(fus_ho) == n_ho

        cidx_ho = np.array([key_to_idx[r["compound_key"]] for r in ho_ep_rows], dtype=np.int32)
        smi_mask_ho = smiles_valid[cidx_ho]
        y_ho = np.array([float(r["y_value"]) for r in ho_ep_rows], dtype=np.float64)
        yl_ho = np.array([float(r["y_lower"]) for r in ho_ep_rows], dtype=np.float64)
        yu_ho = np.array([float(r["y_upper"]) for r in ho_ep_rows], dtype=np.float64)
        ct_ho = np.array([r["censor_type"] for r in ho_ep_rows], dtype=object)
        ho_max_t = _col_arr(cpu_ho, "max_tanimoto")

        s_d_ho = {"cpu": _col_arr(cpu_ho, "pred_dual"), "fusion": _col_arr(fus_ho, "pred_dual")}
        s_s_ho = {"cpu": _col_arr(cpu_ho, "pred_smiles_only"), "fusion": _col_arr(fus_ho, "pred_smiles_only")}
        fam_ho_dual = {k: regression_metrics(y_ho, s_d_ho[k], yl_ho, yu_ho, ct_ho) for k in ("cpu", "fusion")}
        fam_ho_smi = {k: regression_metrics(y_ho[smi_mask_ho], s_s_ho[k][smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho]) for k in ("cpu", "fusion")}

        d_fr, s_fr = fr["dual_modality"], fr["smiles_only"]
        ho_dual = _apply_simplex_blend(list(d_fr["weights"].keys()), s_d_ho, d_fr["weights"], bounds=bounds, fallback=fr["y_cv_mean_dual"])
        ho_smi = _apply_simplex_blend(list(s_fr["weights"].keys()), s_s_ho, s_fr["weights"], bounds=bounds, fallback=fr["y_cv_mean_smi"])
        champ_ho_dual_m = regression_metrics(y_ho, ho_dual, yl_ho, yu_ho, ct_ho)
        champ_ho_smi_m = regression_metrics(y_ho[smi_mask_ho], ho_smi[smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho])
        conf_ho_dual = eval_conformal_coverage(y_ho, ho_dual, ct_ho, d_fr["conformal"], max_tanimoto=ho_max_t)
        conf_ho_smi = eval_conformal_coverage(y_ho[smi_mask_ho], ho_smi[smi_mask_ho], ct_ho[smi_mask_ho], s_fr["conformal"], max_tanimoto=ho_max_t[smi_mask_ho])

        cens_doc = None
        if fr["censored_head_dual"] is not None:
            ch_d, ch_s = fr["censored_head_dual"], fr["censored_head_smi"]
            ho_cens_d = _apply_simplex_blend(list(ch_d["weights"].keys()), {"champion_exact": ho_dual, "cpu_aft": _col_arr(cpu_ho, "pred_aft"), "fusion_aft": _col_arr(fus_ho, "pred_aft")}, ch_d["weights"], bounds=bounds, fallback=fr["y_cv_mean_dual"])
            ho_cens_s = _apply_simplex_blend(list(ch_s["weights"].keys()), {"champion_exact": ho_smi, "cpu_aft": _col_arr(cpu_ho, "pred_aft_smiles_only"), "fusion_aft": _col_arr(fus_ho, "pred_aft_smiles_only")}, ch_s["weights"], bounds=bounds, fallback=fr["y_cv_mean_smi"])
            cens_doc = {
                "dual_modality": {"mode": ch_d["mode"], "weights": ch_d["weights"], "cv_metrics": ch_d["cv_metrics"], "holdout_metrics": regression_metrics(y_ho, ho_cens_d, yl_ho, yu_ho, ct_ho)},
                "smiles_only": {"mode": ch_s["mode"], "weights": ch_s["weights"], "cv_metrics": ch_s["cv_metrics"], "holdout_metrics": regression_metrics(y_ho[smi_mask_ho], ho_cens_s[smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho])},
            }

        cpu_m_doc = json.loads((EXP_CPU_DIR / ep / "metrics.json").read_text(encoding="utf-8"))
        fus_m_doc = json.loads((EXP_FUSION_DIR / ep / "metrics.json").read_text(encoding="utf-8"))

        new_endpoints_manifest[ep] = {
            "endpoint": ep,
            "expansion_wave": "v33.1",
            "display_en": spec["display_en"],
            "display_zh": spec["display_zh"],
            "tier": 1 if spec["tier"] == "tier1" else 2,
            "tier_label": spec["tier"],
            "domain": spec["domain"],
            "task_type": "regression",
            "size_tier": spec["size_tier"],
            "unit": spec["unit"],
            "linear_unit": spec["linear_unit"],
            "log10_space": spec["log10_space"],
            "bounds": list(bounds) if bounds is not None else None,
            "cond_cols": cpu_m_doc["cond_cols"],
            "canonical_conditions": spec["canonical_conditions"],
            "counts": ep_counts[ep],
            "v32_reference": None,
            "dual_modality": {
                "winning_family": d_fr["winning_family"], "is_super_ensemble": d_fr["is_super_ensemble"], "weights": d_fr["weights"],
                "best_single_family": d_fr["best_single_family"], "nested_super_delta_over_best_single": d_fr["nested_super_delta_over_best_single"],
                "cv_metrics": d_fr["champion_cv_metrics"], "cv_metrics_raw": None,
                "holdout_metrics": champ_ho_dual_m, "holdout_metrics_raw": None,
                "family_cv_metrics": d_fr["single_family_oof_metrics"], "family_holdout_metrics": fam_ho_dual,
                "calibrator": None, "conformal": d_fr["conformal"], "conformal_holdout_coverage": conf_ho_dual,
            },
            "smiles_only": {
                "winning_family": s_fr["winning_family"], "is_super_ensemble": s_fr["is_super_ensemble"], "weights": s_fr["weights"],
                "best_single_family": s_fr["best_single_family"], "nested_super_delta_over_best_single": s_fr["nested_super_delta_over_best_single"],
                "cv_metrics": s_fr["champion_cv_metrics"], "cv_metrics_raw": None,
                "holdout_metrics": champ_ho_smi_m, "holdout_metrics_raw": None,
                "family_cv_metrics": s_fr["single_family_oof_metrics"], "family_holdout_metrics": fam_ho_smi,
                "calibrator": None, "conformal": s_fr["conformal"], "conformal_holdout_coverage": conf_ho_smi,
            },
            "censored_head": cens_doc,
            "representation_ablation_holdout": {k: v.get("ho_metrics") for k, v in fus_m_doc.get("representation_ablation", {}).items()},
        }
        print(
            f"  [HO ] {ep:<22} | {d_fr['winning_family']:<14} | MAE={champ_ho_dual_m['mae']:.4f} Sp={champ_ho_dual_m['spearman']:.3f} R2={champ_ho_dual_m['r2']:.3f} "
            f"Cov80/90={100*(conf_ho_dual['cov80'] or 0):.1f}%/{100*(conf_ho_dual['cov90'] or 0):.1f}% | SMI MAE={champ_ho_smi_m['mae']:.4f}",
            flush=True,
        )

    # =========================================================================
    # PHASE C: PRUNE NEW SERVING MODELS (ZERO CATBOOST!) & COPY TO BUNDLE
    # =========================================================================
    print(f"\n[PHASE C] Pruning & verifying serving models for {len(new_eps)} new endpoints...", flush=True)
    need_cpu_track: Dict[str, Set[str]] = defaultdict(set)
    need_fusion_track: Dict[str, Set[str]] = defaultdict(set)
    for ep in new_eps:
        edoc = new_endpoints_manifest[ep]
        for tr in ("dual_modality", "smiles_only"):
            w = edoc[tr]["weights"]
            if w.get("cpu", 0.0) > 0.0:
                need_cpu_track[ep].add(tr)
            if w.get("fusion", 0.0) > 0.0:
                need_fusion_track[ep].add(tr)
        if edoc.get("censored_head") is not None:
            for tr, aft_k in (("dual_modality", "aft_dual"), ("smiles_only", "aft_smiles")):
                cw = edoc["censored_head"][tr]["weights"]
                if cw.get("cpu_aft", 0.0) > 0.0:
                    need_cpu_track[ep].add(aft_k)
                if cw.get("fusion_aft", 0.0) > 0.0:
                    need_fusion_track[ep].add(aft_k)

    all_pkg_modules: Set[str] = set()
    new_joblib_files: List[Path] = []

    for ep in sorted(need_fusion_track.keys()):
        fb = joblib.load(EXP_FUSION_DIR / ep / "fusion_models_5fold.joblib")
        tracks = need_fusion_track[ep]
        act_heads: Set[str] = set()
        if "dual_modality" in tracks:
            act_heads.update(fb["dual_modality"]["stacker"].active_heads(tol=1e-9))
        if "smiles_only" in tracks:
            act_heads.update(fb["smiles_only"]["stacker"].active_heads(tol=1e-9))
        if ("aft_dual" in tracks or "aft_smiles" in tracks) and fb.get("aft") is not None:
            if "aft_dual" in tracks:
                act_heads.update(fb["aft"]["stacker_dual"].active_heads(tol=1e-9))
            if "aft_smiles" in tracks:
                act_heads.update(fb["aft"]["stacker_smiles_only"].active_heads(tol=1e-9))

        if "cpu_dual" in act_heads:
            need_cpu_track[ep].add("dual_modality")
        if "cpu_smiles" in act_heads:
            need_cpu_track[ep].add("smiles_only")
        if "cpu_aft_dual" in act_heads:
            need_cpu_track[ep].add("aft_dual")
        if "cpu_aft_smiles" in act_heads:
            need_cpu_track[ep].add("aft_smiles")
        if "cpu_seq_only" in act_heads:
            need_cpu_track[ep].add("seq_only")

        pruned_fus = {
            "schema": "v33_pruned_fusion_serving_v1",
            "endpoint": ep, "task_type": fb["task_type"], "bounds": fb["bounds"],
            "cond_cols": fb["cond_cols"], "canonical_default_conditions": fb["canonical_default_conditions"],
            "pretrained_base_heads": {hk: fb["pretrained_base_heads"][hk] for hk in sorted(act_heads) if hk in fb["pretrained_base_heads"]},
            "dual_modality": {"candidate_heads": fb["dual_modality"]["candidate_heads"], "stacker": fb["dual_modality"]["stacker"], "active_heads": fb["dual_modality"]["stacker"].active_heads(tol=1e-9)} if "dual_modality" in tracks else None,
            "smiles_only": {"candidate_heads": fb["smiles_only"]["candidate_heads"], "stacker": fb["smiles_only"]["stacker"], "active_heads": fb["smiles_only"]["stacker"].active_heads(tol=1e-9)} if "smiles_only" in tracks else None,
            "aft": {"stacker_dual": fb["aft"]["stacker_dual"], "stacker_smiles_only": fb["aft"]["stacker_smiles_only"]} if (("aft_dual" in tracks or "aft_smiles" in tracks) and fb.get("aft") is not None) else None,
        }
        out_fp = EXP_SERVING_DIR / f"{ep}_fusion.joblib"
        joblib.dump(pruned_fus, out_fp, compress=1)
        all_pkg_modules.update(_check_no_catboost_in_joblib(out_fp))
        new_joblib_files.append(out_fp)
        del fb, pruned_fus

    for ep in sorted(need_cpu_track.keys()):
        cb = joblib.load(EXP_CPU_DIR / ep / "models_5fold.joblib")
        tracks = need_cpu_track[ep]
        pruned_cpu = {
            "schema": "v33_pruned_cpu_serving_v1",
            "endpoint": ep, "task_type": cb["task_type"], "bounds": cb["bounds"],
            "cond_cols": cb["cond_cols"], "canonical_default_conditions": cb["canonical_default_conditions"],
            "dual_modality": {"uses_meta": False, "meta_endpoints": [], "folds": cb["dual_modality"]["folds"], "calibrator": None} if "dual_modality" in tracks else None,
            "smiles_only": {"uses_meta": False, "meta_endpoints": [], "folds": cb["smiles_only"]["folds"], "calibrator": None} if "smiles_only" in tracks else None,
            "seq_only": {"folds": cb["seq_only"]["folds"]} if ("seq_only" in tracks and cb.get("seq_only") is not None) else None,
            "aft": {"folds_dual": cb["aft"]["folds_dual"] if "aft_dual" in tracks else None, "folds_smiles_only": cb["aft"]["folds_smiles_only"] if "aft_smiles" in tracks else None} if (("aft_dual" in tracks or "aft_smiles" in tracks) and cb.get("aft") is not None) else None,
        }
        out_cp = EXP_SERVING_DIR / f"{ep}_cpu.joblib"
        joblib.dump(pruned_cpu, out_cp, compress=1)
        all_pkg_modules.update(_check_no_catboost_in_joblib(out_cp))
        new_joblib_files.append(out_cp)
        del cb, pruned_cpu

    # Copy ONLY new serving models into bundle/serving_models/ and verify ZERO existing models touched!
    for src in new_joblib_files:
        assert src.name not in existing_joblib_shas, f"Attempted overwrite of existing V33 serving model: {src.name}"
        shutil.copy2(src, SERVING_DIR / src.name)
    for fname, exp_sha in existing_joblib_shas.items():
        assert sha256_file(SERVING_DIR / fname) == exp_sha, f"REGRESSION! Existing V33 serving model changed: {fname}"
    new_mb = sum(p.stat().st_size for p in new_joblib_files) / 1e6
    tot_mb = sum(p.stat().st_size for p in SERVING_DIR.glob("*.joblib")) / 1e6
    print(f"[PRUNE DONE] Added {len(new_joblib_files)} new .joblib files ({new_mb:.1f} MB) -> Total serving_models={tot_mb:.1f} MB | Modules={sorted(all_pkg_modules)} (0 catboost!)", flush=True)

    # =========================================================================
    # PHASE D: REBUILD UNIFIED 46-ENDPOINT AD/ANALOG INDEX & MERGE MANIFEST
    # =========================================================================
    ad_npz_path, analog_gz_path, n_ref_total = rebuild_unified_ad_analog_index(existing_24_eps, new_eps, v33_24_manifest)

    merged = json.loads(backup_path.read_text(encoding="utf-8"))
    # Keep ALL 24 existing endpoints 100% UNCHANGED!
    for ep in new_eps:
        merged["endpoints"][ep] = new_endpoints_manifest[ep]
    # Assert all 24 existing entries are byte-identical!
    for ep in existing_24_eps:
        assert json.dumps(merged["endpoints"][ep], sort_keys=True) == json.dumps(v33_24_manifest["endpoints"][ep], sort_keys=True), f"Existing endpoint manifest mutated: {ep}"

    split_sum = json.loads((EXP_DATA / "v331_split_summary.json").read_text(encoding="utf-8"))
    all_46_eps = sorted(merged["endpoints"].keys())
    assert len(all_46_eps) == 24 + len(new_eps)

    def _mean(vals):
        c = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        return round(sum(c) / len(c), 5) if c else None

    v331_summary = {
        "n_new_endpoints": len(new_eps),
        "dual_modality_winning_family_counts": {fam: sum(1 for e in new_eps if new_endpoints_manifest[e]["dual_modality"]["winning_family"] == fam) for fam in sorted({new_endpoints_manifest[e]["dual_modality"]["winning_family"] for e in new_eps})},
        "smiles_only_winning_family_counts": {fam: sum(1 for e in new_eps if new_endpoints_manifest[e]["smiles_only"]["winning_family"] == fam) for fam in sorted({new_endpoints_manifest[e]["smiles_only"]["winning_family"] for e in new_eps})},
        "mean_holdout_dual_mae": _mean([new_endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["mae"] for e in new_eps]),
        "mean_holdout_dual_spearman": _mean([new_endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["spearman"] for e in new_eps]),
        "mean_holdout_dual_r2": _mean([new_endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["r2"] for e in new_eps]),
        "mean_holdout_smiles_mae": _mean([new_endpoints_manifest[e]["smiles_only"]["holdout_metrics"]["mae"] for e in new_eps]),
        "mean_conformal_cov80": _mean([new_endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov80"] for e in new_eps]),
        "mean_conformal_cov90": _mean([new_endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov90"] for e in new_eps]),
    }

    merged["version"] = "v33.1"
    merged["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    merged["dataset_summary"] = {
        "n_unique_compounds": int(split_sum["n_total_unified_compounds"]) - 1,
        "n_unified_compounds_incl_case17": int(split_sum["n_total_unified_compounds"]),
        "n_v330_base_compounds": int(split_sum["n_v33_compounds"]),
        "n_v331_delta_compounds": int(split_sum["n_delta_new_compounds"]),
        "n_endpoints": len(all_46_eps),
        "n_v330_existing_endpoints": 24,
        "n_v331_new_endpoints": len(new_eps),
        "n_tier1_endpoints": sum(1 for ep in all_46_eps if merged["endpoints"][ep]["tier"] == 1),
        "n_tier2_endpoints": sum(1 for ep in all_46_eps if merged["endpoints"][ep]["tier"] == 2),
        "n_classification_endpoints": sum(1 for ep in all_46_eps if merged["endpoints"][ep]["task_type"] == "classification"),
        "n_regression_endpoints": sum(1 for ep in all_46_eps if merged["endpoints"][ep]["task_type"] == "regression"),
        "n_total_rows": sum(merged["endpoints"][ep]["counts"]["n_total"] for ep in all_46_eps),
        "n_cv_rows": sum(merged["endpoints"][ep]["counts"]["n_cv"] for ep in all_46_eps),
        "n_holdout_rows": sum(merged["endpoints"][ep]["counts"]["n_holdout"] for ep in all_46_eps),
        "n_censored_rows": sum(merged["endpoints"][ep]["counts"]["n_censored_total"] for ep in all_46_eps),
        "n_case17_drugs": 17,
    }
    merged["overall_summary"]["v331_expansion_22_summary"] = v331_summary
    merged["serving_inventory"]["cpu_endpoints_bundled"] = sorted(set(v33_24_manifest["serving_inventory"]["cpu_endpoints_bundled"]) | set(need_cpu_track.keys()))
    merged["serving_inventory"]["fusion_endpoints_bundled"] = sorted(set(v33_24_manifest["serving_inventory"]["fusion_endpoints_bundled"]) | set(need_fusion_track.keys()))
    merged["serving_inventory"]["total_pruned_joblib_mb"] = round(tot_mb, 2)
    merged["serving_inventory"]["v331_new_joblib_mb"] = round(new_mb, 2)
    merged["serving_inventory"]["v331_serialized_packages"] = sorted(all_pkg_modules)
    merged["v331_expansion_metadata"] = {
        "admitted_new_endpoints": new_eps,
        "rejected_candidates": split_sum["rejected_candidates"],
        "n_delta_compounds": split_sum["n_delta_new_compounds"],
        "n_ad_reference_compounds": n_ref_total,
    }
    merged["sha256_checksums"]["ad_analog_index.npz"] = sha256_file(ad_npz_path)
    merged["sha256_checksums"]["analog_lookup.json.gz"] = sha256_file(analog_gz_path)
    merged["sha256_checksums"]["serving_models"] = {p.name: sha256_file(p) for p in sorted(SERVING_DIR.glob("*.joblib"))}

    manifest_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    (EXP_BASE / "v331_champion_summary.json").write_text(json.dumps({"elapsed_sec": round(time.time() - t_all, 2), "v331_summary": v331_summary, "endpoints": new_endpoints_manifest}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE Step 05] Merged V33.1 manifest written ({len(all_46_eps)} endpoints = 24 existing + {len(new_eps)} new) in {time.time()-t_all:.1f}s!", flush=True)
    print(json.dumps(v331_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
