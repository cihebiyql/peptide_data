#!/usr/bin/env python3
"""Wave 3A Part 1: Strictly Honest Cross-Family Super-Ensemble Champion Selection,
OOF Calibration & Conformal Intervals, Fast AD + Top-3 Analog Index, Pruned Serving
Model Bundling, and Authoritative `v33_manifest.json` Generation.

Runs on `node23` with `/usr/bin/python3` (64 CPU cores, 0 GPU usage).
Strictly read-only access to data/, features/, embeddings/, models_cpu/,
models_fusion/, and models_gpu_deep/. Writes ONLY under /data/qlyu/v33_upgrade/bundle/.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# Add read-only model modules to sys.path for joblib deserialization & deep checkpoint loading
BASE_DIR = Path("/data/qlyu/v33_upgrade")
sys.path.insert(0, str(BASE_DIR / "models_cpu"))
sys.path.insert(0, str(BASE_DIR / "models_fusion"))
sys.path.insert(0, str(BASE_DIR / "models_gpu_deep"))

from v33_config import (  # noqa: E402
    AUX_ENDPOINTS,
    CASE17_CLINICAL_ROUTE,
    CASE17_FDA_NUMERIC,
    ENDPOINT_SPECS,
    compute_halflife_mechanism_anchor,
)
from v33_metrics import (  # noqa: E402
    classification_metrics,
    compute_ece,
    regression_metrics,
)

csv.field_size_limit(10_000_000)

DATA_DIR = BASE_DIR / "data"
ENDPOINTS_DIR = DATA_DIR / "endpoints"
FEATURES_DIR = BASE_DIR / "features"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CPU_DIR = BASE_DIR / "models_cpu"
FUSION_DIR = BASE_DIR / "models_fusion"
DEEP_DIR = BASE_DIR / "models_gpu_deep"
BUNDLE_DIR = BASE_DIR / "bundle"
SERVING_DIR = BUNDLE_DIR / "serving_models"
PRED_OUT_DIR = BUNDLE_DIR / "champion_predictions"

ALL_ENDPOINTS: List[str] = sorted(ENDPOINT_SPECS.keys())
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_IDX = {a: i for i, a in enumerate(AA20)}

# Exact AD tier thresholds specified by Wave 3A
AD_IN_DOMAIN_THR = 0.65
AD_BORDERLINE_THR = 0.40

# Display metadata for all 24 endpoints (100% aligned with V33 schema)
ENDPOINT_DISPLAY: Dict[str, Dict[str, Any]] = {
    "BBB": {"display_en": "Blood-Brain Barrier (BBB)", "display_zh": "血脑屏障透过概率", "domain": "permeability", "log10_space": False, "linear_unit": "probability"},
    "CL_absolute": {"display_en": "Absolute Systemic Clearance (CL)", "display_zh": "绝对全身清除率 CL", "domain": "pk", "log10_space": True, "linear_unit": "L/h"},
    "CL_per_kg": {"display_en": "Weight-Normalized Clearance (CL/kg)", "display_zh": "体重归一化清除率 CL/kg", "domain": "pk", "log10_space": True, "linear_unit": "mL/min/kg"},
    "Caco2": {"display_en": "Caco-2 Permeability", "display_zh": "Caco-2 肠上皮渗透", "domain": "permeability", "log10_space": True, "linear_unit": "cm/s"},
    "Cmax": {"display_en": "Peak Plasma Concentration (Cmax)", "display_zh": "峰浓度 Cmax", "domain": "pk", "log10_space": True, "linear_unit": "uM"},
    "F": {"display_en": "Bioavailability (F)", "display_zh": "生物利用度 F", "domain": "pk", "log10_space": False, "linear_unit": "fraction"},
    "LogD7.4": {"display_en": "Distribution Coefficient (LogD7.4)", "display_zh": "表观分配系数 LogD(7.4)", "domain": "physicochemical", "log10_space": False, "linear_unit": "logD"},
    "MDCK": {"display_en": "MDCK Permeability", "display_zh": "MDCK 细胞渗透", "domain": "permeability", "log10_space": True, "linear_unit": "cm/s"},
    "MIC_C_albicans": {"display_en": "C. albicans MIC", "display_zh": "白色念珠菌 MIC (真菌)", "domain": "antimicrobial", "log10_space": True, "linear_unit": "uM"},
    "MIC_E_coli": {"display_en": "E. coli ATCC 25922 MIC", "display_zh": "大肠杆菌 MIC (革兰氏阴性)", "domain": "antimicrobial", "log10_space": True, "linear_unit": "uM"},
    "MIC_P_aeruginosa": {"display_en": "P. aeruginosa ATCC 27853 MIC", "display_zh": "铜绿假单胞菌 MIC (革兰氏阴性)", "domain": "antimicrobial", "log10_space": True, "linear_unit": "uM"},
    "MIC_S_aureus": {"display_en": "S. aureus ATCC 25923 MIC", "display_zh": "金黄色葡萄球菌 MIC (革兰氏阳性)", "domain": "antimicrobial", "log10_space": True, "linear_unit": "uM"},
    "PAMPA": {"display_en": "PAMPA Passive Permeability", "display_zh": "PAMPA 被动膜渗透", "domain": "permeability", "log10_space": True, "linear_unit": "cm/s"},
    "PPB": {"display_en": "Plasma Protein Binding (PPB)", "display_zh": "血浆蛋白结合率 PPB", "domain": "pk", "log10_space": False, "linear_unit": "fraction_bound"},
    "RRCK": {"display_en": "RRCK Permeability", "display_zh": "RRCK 低外排渗透", "domain": "permeability", "log10_space": True, "linear_unit": "cm/s"},
    "Vd_per_kg": {"display_en": "Volume of Distribution (Vd/kg)", "display_zh": "分布容积 Vd/kg", "domain": "pk", "log10_space": True, "linear_unit": "L/kg"},
    "cytotoxicity_cls": {"display_en": "Mammalian Cytotoxicity Risk", "display_zh": "哺乳动物细胞毒性概率", "domain": "safety", "log10_space": False, "linear_unit": "probability"},
    "half_life_cls": {"display_en": "Long Plasma Half-Life Prob", "display_zh": "长半衰期概率", "domain": "pk", "log10_space": False, "linear_unit": "probability"},
    "half_life_reg": {"display_en": "Plasma Half-Life (T1/2)", "display_zh": "血浆半衰期 T1/2", "domain": "pk", "log10_space": True, "linear_unit": "h"},
    "hemolysis_HC50": {"display_en": "Hemolysis HC50", "display_zh": "半数溶血浓度 HC50", "domain": "safety", "log10_space": True, "linear_unit": "uM"},
    "hemolysis_cls": {"display_en": "Hemolysis Risk (Binary)", "display_zh": "溶血风险概率", "domain": "safety", "log10_space": False, "linear_unit": "probability"},
    "overall_peptide_toxicity": {"display_en": "General Peptide Toxicity", "display_zh": "综合多肽毒性概率", "domain": "safety", "log10_space": False, "linear_unit": "probability"},
    "solubility_cls": {"display_en": "Aqueous Solubility Prob", "display_zh": "水溶性概率", "domain": "physicochemical", "log10_space": False, "linear_unit": "probability"},
    "solubility_reg": {"display_en": "Aqueous Solubility (uM)", "display_zh": "水溶性定量 (uM)", "domain": "physicochemical", "log10_space": True, "linear_unit": "uM"},
}

KEY_CENSORED_ENDPOINTS = {
    "hemolysis_HC50", "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans",
    "half_life_reg", "PPB", "solubility_reg", "F",
}


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


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)))


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
    """Row-wise weighted average over finite entries only (renormalizing available weights)."""
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


def _apply_simplex_blend(
    stream_names: List[str],
    streams: Dict[str, np.ndarray],
    weights: Dict[str, float],
    task_type: str,
    bounds: Optional[Tuple[float, float]] = None,
    fallback: float = 0.0,
) -> np.ndarray:
    active = [s for s in stream_names if weights.get(s, 0.0) > 0.0]
    if not active:
        active = [stream_names[0]]
    w = np.array([weights.get(s, 0.0) for s in active], dtype=np.float64)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(active)) / len(active)
    M = np.column_stack([streams[s] for s in active])

    if task_type == "classification":
        Z = np.where(np.isfinite(M), _logit(np.where(np.isfinite(M), M, 0.5)), np.nan)
        fb_z = float(_logit(np.array([np.clip(fallback, 1e-4, 1.0 - 1e-4)]))[0])
        z_out = _weighted_nan_combine(Z, w, fallback=fb_z)
        return np.clip(_sigmoid(z_out), 1e-6, 1.0 - 1e-6)

    y_out = _weighted_nan_combine(M, w, fallback=fallback)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


def _fit_simplex_weights(
    P: np.ndarray,
    y: np.ndarray,
    task_type: str,
    reg: float = 0.005,
    prune_tol: float = 0.02,
    hinge_mode: bool = False,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    censor_type: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fit non-negative simplex weights (sum=1) on training folds ONLY.

    - Classification: logit-space weighted average minimizing regularized binary log-loss.
    - Regression: value-space weighted average minimizing smooth L1 (or all-row censored hinge MAE).
    - Multi-start from uniform + each single-family vertex so optimizer never misses a pure corner.
    - Prunes negligible weights (< prune_tol) and renormalizes cleanly.
    """
    n, m = P.shape
    unif = np.ones(m, dtype=np.float64) / float(m)
    fallback = float(np.nanmean(y)) if len(y) > 0 else (0.5 if task_type == "classification" else 0.0)

    if task_type == "classification":
        Z = np.where(np.isfinite(P), _logit(np.where(np.isfinite(P), P, 0.5)), np.nan)
        fb_z = float(_logit(np.array([np.clip(fallback, 1e-4, 1.0 - 1e-4)]))[0])

        def loss_fn(theta: np.ndarray) -> float:
            lw = theta - np.max(theta)
            w = np.exp(lw)
            w = w / np.sum(w)
            z_comb = _weighted_nan_combine(Z, w, fallback=fb_z)
            pos_z = np.maximum(z_comb, 0.0)
            ll = float(np.mean(pos_z - y * z_comb + np.log1p(np.exp(-np.abs(z_comb)))))
            return ll + reg * float(np.sum((w - unif) ** 2))
    else:
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
                err = np.where(ct == "exact", e_ex, np.where(ct == "right", e_r, np.where(ct == "left", e_l, e_i)))
                val = float(np.mean(err))
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
        if w.sum() <= 0:
            w[int(np.argmax(np.exp(lw)))] = 1.0
        else:
            w = w / w.sum()
    return w


def _fit_and_select_calibrator_oof(
    p_oof: np.ndarray,
    y_cv: np.ndarray,
    fold_ids: np.ndarray,
    eval_mask: np.ndarray,
) -> Dict[str, Any]:
    """Strictly nested-OOF probability calibration (none vs platt vs isotonic).

    Selects calibrator that strictly improves OOF Brier + ECE without hurting OOF ROC-AUC.
    Returns explicit JSON-serializable frozen calibrator spec!
    """
    valid = eval_mask & np.isfinite(p_oof) & np.isfinite(y_cv)
    p = np.clip(p_oof[valid], 1e-6, 1.0 - 1e-6)
    y = y_cv[valid].astype(np.int32)
    fids = fold_ids[valid].astype(np.int32)
    uniq_folds = sorted(set(fids.tolist()))

    nested = {
        "none": p.copy(),
        "platt": np.zeros_like(p),
        "isotonic": np.zeros_like(p),
    }
    for f in uniq_folds:
        va = fids == f
        tr = ~va
        if len(np.unique(y[tr])) < 2:
            nested["platt"][va] = p[va]
            nested["isotonic"][va] = p[va]
            continue
        lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(p[tr])[:, None], y[tr])
        if float(lr.coef_[0, 0]) > 0.0:
            nested["platt"][va] = lr.predict_proba(_logit(p[va])[:, None])[:, 1]
        else:
            nested["platt"][va] = p[va]

        iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
        iso.fit(p[tr], y[tr])
        nested["isotonic"][va] = iso.predict(p[va])

    comp: Dict[str, Any] = {}
    for m in ("none", "platt", "isotonic"):
        pm = np.clip(nested[m], 1e-6, 1.0 - 1e-6)
        comp[m] = {
            "roc_auc": round(float(roc_auc_score(y, pm)), 5),
            "pr_auc": round(float(average_precision_score(y, pm)), 5),
            "brier": round(float(brier_score_loss(y, pm)), 5),
            "ece": round(float(compute_ece(y, pm, n_bins=10)), 5),
            "log_loss": round(float(log_loss(y, pm)), 5),
        }

    # Select calibrator strictly improving OOF Brier & ECE without hurting ROC-AUC
    raw_roc = comp["none"]["roc_auc"]
    raw_brier = comp["none"]["brier"]
    raw_ece = comp["none"]["ece"]

    chosen = "none"
    # Check Platt first (strictly monotonic -> preserves ROC-AUC identically!)
    platt_ok = (
        comp["platt"]["brier"] <= raw_brier - 1e-4
        and comp["platt"]["ece"] <= raw_ece + 0.002
        and comp["platt"]["roc_auc"] >= raw_roc - 0.0005
    ) or (
        comp["platt"]["ece"] <= raw_ece - 0.005
        and comp["platt"]["brier"] <= raw_brier + 1e-4
        and comp["platt"]["roc_auc"] >= raw_roc - 0.0005
    )
    if platt_ok:
        chosen = "platt"

    # Check Isotonic: adopt only if it strictly beats current best on Brier (>=0.0005) & does NOT hurt ROC-AUC (>0.001) or ECE
    cur_brier = comp[chosen]["brier"]
    cur_ece = comp[chosen]["ece"]
    iso_ok = (
        comp["isotonic"]["brier"] <= cur_brier - 0.0005
        and comp["isotonic"]["ece"] <= cur_ece + 0.002
        and comp["isotonic"]["roc_auc"] >= raw_roc - 0.001
    )
    if iso_ok:
        chosen = "isotonic"

    # Fit final frozen calibrator on full CV OOF
    params: Dict[str, Any] = {}
    if chosen == "platt":
        lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(p)[:, None], y)
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        if a <= 0.0:
            chosen = "none"
        else:
            params = {"a": round(a, 6), "b": round(b, 6)}
    elif chosen == "isotonic":
        iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
        iso.fit(p, y)
        params = {
            "x_thresholds": [round(float(x), 6) for x in iso.X_thresholds_],
            "y_thresholds": [round(float(v), 6) for v in iso.y_thresholds_],
        }

    # Nested-OOF calibrated vector (for honest OOF metrics)
    nested_cal_full = np.full(len(p_oof), np.nan, dtype=np.float64)
    nested_cal_full[valid] = nested[chosen]

    return {
        "method": chosen,
        "params": params,
        "nested_oof_comparison": comp,
        "nested_oof_calibrated": nested_cal_full,
    }


def apply_frozen_calibrator(p_raw: np.ndarray, cal_spec: Dict[str, Any]) -> np.ndarray:
    p = np.asarray(p_raw, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pc = np.clip(p[ok], 1e-6, 1.0 - 1e-6)
    method = cal_spec["method"]
    params = cal_spec.get("params", {})
    if method == "platt":
        z = float(params["a"]) * _logit(pc) + float(params["b"])
        out[ok] = _sigmoid(z)
    elif method == "isotonic":
        xs = np.asarray(params["x_thresholds"], dtype=np.float64)
        ys = np.asarray(params["y_thresholds"], dtype=np.float64)
        out[ok] = np.interp(pc, xs, ys)
    else:
        out[ok] = pc
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        return 0.0
    k = int(math.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def _fit_conformal_oof(
    y_cv: np.ndarray,
    pred_oof: np.ndarray,
    censor_cv: np.ndarray,
    fold_ids: np.ndarray,
    eval_mask: np.ndarray,
    oof_max_tanimoto: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Fit CV+ / split conformal 80% & 90% quantiles strictly on OOF exact residuals.

    Tests adaptive Tanimoto-distance residual scaling strictly via nested 5-fold CV and
    keeps it ONLY if nested-OOF 80% coverage >= 0.78 AND mean 90% width improves >= 2%.
    """
    mask = eval_mask & (censor_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof)
    if mask.sum() < 5:
        mask = eval_mask & np.isfinite(y_cv) & np.isfinite(pred_oof)

    ye = y_cv[mask]
    pe = pred_oof[mask]
    abs_res = np.abs(ye - pe)
    fids = fold_ids[mask].astype(np.int32)
    t_oof = oof_max_tanimoto[mask]
    n_ex = len(abs_res)
    med_r = float(np.median(abs_res)) if n_ex > 0 else 0.5
    floor_s = max(0.30 * med_r, 1e-3)

    # Full-OOF standard conformal quantiles
    std_q80 = _conformal_quantile(abs_res, 0.20)
    std_q90 = _conformal_quantile(abs_res, 0.10)

    # Nested 5-fold OOF check for standard conformal
    uniq_f = sorted(set(fids.tolist()))
    std_c80_hits, std_c90_hits, std_w90 = [], [], []
    for f in uniq_f:
        va = fids == f
        tr = ~va
        if tr.sum() < 4 or va.sum() == 0:
            continue
        q80_f = _conformal_quantile(abs_res[tr], 0.20)
        q90_f = _conformal_quantile(abs_res[tr], 0.10)
        lo80, hi80 = pe[va] - q80_f, pe[va] + q80_f
        lo90, hi90 = pe[va] - q90_f, pe[va] + q90_f
        if bounds is not None:
            lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
            lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
        std_c80_hits.extend(((ye[va] >= lo80) & (ye[va] <= hi80)).tolist())
        std_c90_hits.extend(((ye[va] >= lo90) & (ye[va] <= hi90)).tolist())
        std_w90.extend((hi90 - lo90).tolist())

    std_oof_cov80 = float(np.mean(std_c80_hits)) if std_c80_hits else 0.80
    std_oof_cov90 = float(np.mean(std_c90_hits)) if std_c90_hits else 0.90
    std_oof_w90 = float(np.mean(std_w90)) if std_w90 else 2.0 * std_q90

    diag: Dict[str, Any] = {
        "n_oof_exact": int(n_ex),
        "standard": {
            "q80": round(std_q80, 5),
            "q90": round(std_q90, 5),
            "nested_oof_cov80": round(std_oof_cov80, 4),
            "nested_oof_cov90": round(std_oof_cov90, 4),
            "nested_oof_width90": round(std_oof_w90, 5),
        },
    }

    mode = "standard"
    q80_final = std_q80
    q90_final = std_q90
    dist_slope = 0.0
    dist_intercept = 1.0
    min_scale = floor_s

    # Test adaptive Tanimoto-distance scaling strictly on nested 5-fold OOF
    ok_t = np.isfinite(t_oof)
    if ok_t.sum() >= 50:
        dist_all = 1.0 - np.clip(np.where(ok_t, t_oof, 0.5), 0.0, 1.0)
        sp_all = float(spearmanr(dist_all[ok_t], abs_res[ok_t]).statistic)
        if np.isfinite(sp_all) and sp_all >= 0.10:
            ad_c80_hits, ad_c90_hits, ad_w90 = [], [], []
            for f in uniq_f:
                va = fids == f
                tr = (~va) & ok_t
                if tr.sum() < 25 or va.sum() == 0:
                    continue
                try:
                    reg = HuberRegressor(alpha=1.0).fit(dist_all[tr, None], abs_res[tr])
                    sl = max(0.0, float(reg.coef_[0]))
                    ic = max(floor_s, float(reg.intercept_))
                except Exception:
                    sl, ic = 0.0, max(med_r, 1e-3)
                s_tr = np.maximum(ic + sl * dist_all[tr], floor_s)
                s_va = np.maximum(ic + sl * dist_all[va], floor_s)
                q80_f = _conformal_quantile(abs_res[tr] / s_tr, 0.20)
                q90_f = _conformal_quantile(abs_res[tr] / s_tr, 0.10)
                lo80, hi80 = pe[va] - q80_f * s_va, pe[va] + q80_f * s_va
                lo90, hi90 = pe[va] - q90_f * s_va, pe[va] + q90_f * s_va
                if bounds is not None:
                    lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
                    lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
                ad_c80_hits.extend(((ye[va] >= lo80) & (ye[va] <= hi80)).tolist())
                ad_c90_hits.extend(((ye[va] >= lo90) & (ye[va] <= hi90)).tolist())
                ad_w90.extend((hi90 - lo90).tolist())

            if ad_c80_hits:
                ad_cov80 = float(np.mean(ad_c80_hits))
                ad_cov90 = float(np.mean(ad_c90_hits))
                ad_w90_mean = float(np.mean(ad_w90))
                diag["adaptive_tanimoto_dist"] = {
                    "spearman_dist_err": round(sp_all, 4),
                    "nested_oof_cov80": round(ad_cov80, 4),
                    "nested_oof_cov90": round(ad_cov90, 4),
                    "nested_oof_width90": round(ad_w90_mean, 5),
                }
                if ad_cov80 >= 0.78 and ad_w90_mean < std_oof_w90 * 0.98:
                    try:
                        reg_full = HuberRegressor(alpha=1.0).fit(dist_all[ok_t, None], abs_res[ok_t])
                        sl_full = max(0.0, float(reg_full.coef_[0]))
                        ic_full = max(floor_s, float(reg_full.intercept_))
                        if sl_full > 1e-4:
                            s_full = np.maximum(ic_full + sl_full * dist_all, floor_s)
                            mode = "adaptive_tanimoto_dist"
                            dist_slope = sl_full
                            dist_intercept = ic_full
                            min_scale = floor_s
                            q80_final = _conformal_quantile(abs_res / s_full, 0.20)
                            q90_final = _conformal_quantile(abs_res / s_full, 0.10)
                    except Exception:
                        pass

    return {
        "mode": mode,
        "q80": round(q80_final, 5),
        "q90": round(q90_final, 5),
        "dist_slope": round(dist_slope, 6),
        "dist_intercept": round(dist_intercept, 6),
        "min_scale": round(min_scale, 6),
        "bounds": list(bounds) if bounds is not None else None,
        "oof_diagnostics": diag,
    }


def apply_frozen_conformal(
    pred: np.ndarray,
    conf_spec: Dict[str, Any],
    max_tanimoto: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    if conf_spec["mode"] == "adaptive_tanimoto_dist" and max_tanimoto is not None:
        t = np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.5)
        d = 1.0 - np.clip(t, 0.0, 1.0)
        s = np.maximum(conf_spec["dist_intercept"] + conf_spec["dist_slope"] * d, conf_spec["min_scale"])
    else:
        s = np.ones(len(p), dtype=np.float64)
    hw80 = conf_spec["q80"] * s
    hw90 = conf_spec["q90"] * s
    lo80, hi80 = p - hw80, p + hw80
    lo90, hi90 = p - hw90, p + hw90
    bounds = conf_spec.get("bounds")
    if bounds is not None:
        lo80 = np.clip(lo80, bounds[0], bounds[1])
        hi80 = np.clip(hi80, bounds[0], bounds[1])
        lo90 = np.clip(lo90, bounds[0], bounds[1])
        hi90 = np.clip(hi90, bounds[0], bounds[1])
    return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}


def eval_conformal_coverage(
    y_true: np.ndarray,
    pred: np.ndarray,
    censor_type: np.ndarray,
    conf_spec: Dict[str, Any],
    max_tanimoto: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    ivs = apply_frozen_conformal(pred, conf_spec, max_tanimoto=max_tanimoto)
    mask = (censor_type == "exact") & np.isfinite(y_true) & np.isfinite(pred)
    if not mask.any():
        return {"mode": conf_spec["mode"], "n_eval": 0, "cov80": None, "cov90": None, "mean_width_80": None, "mean_width_90": None}
    yt = y_true[mask]
    c80 = float(np.mean((yt >= ivs["lo80"][mask]) & (yt <= ivs["hi80"][mask])))
    c90 = float(np.mean((yt >= ivs["lo90"][mask]) & (yt <= ivs["hi90"][mask])))
    w80 = float(np.mean(ivs["hi80"][mask] - ivs["lo80"][mask]))
    w90 = float(np.mean(ivs["hi90"][mask] - ivs["lo90"][mask]))
    return {
        "mode": conf_spec["mode"],
        "n_eval": int(mask.sum()),
        "cov80": round(c80, 4),
        "cov90": round(c90, 4),
        "mean_width_80": round(w80, 4),
        "mean_width_90": round(w90, 4),
    }


def _select_champion_oof(
    endpoint: str,
    track: str,
    task_type: str,
    bounds: Optional[Tuple[float, float]],
    y_cv: np.ndarray,
    yl_cv: np.ndarray,
    yu_cv: np.ndarray,
    ct_cv: np.ndarray,
    fids_cv: np.ndarray,
    eval_mask: np.ndarray,
    streams_oof: Dict[str, np.ndarray],
    stream_names: List[str],
    oof_max_tanimoto: np.ndarray,
) -> Dict[str, Any]:
    """STRICTLY OOF-ONLY champion selection, nested-OOF simplex fitting, calibration & conformal."""
    n_cv = len(y_cv)
    fallback = float(np.nanmean(y_cv[eval_mask])) if eval_mask.any() else (0.5 if task_type == "classification" else 0.0)

    # 1. Evaluate every single base family strictly on OOF
    single_metrics: Dict[str, Dict[str, Any]] = {}
    for name in stream_names:
        pred = streams_oof[name]
        if task_type == "classification":
            single_metrics[name] = classification_metrics(y_cv[eval_mask], pred[eval_mask])
        else:
            single_metrics[name] = regression_metrics(
                y_cv[eval_mask], pred[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask]
            )

    def _cls_key(m: Dict[str, Any]) -> Tuple[float, float, float]:
        roc = m["roc_auc"] if m["roc_auc"] is not None else -1.0
        pr = m["pr_auc"] if m["pr_auc"] is not None else -1.0
        brier = m["brier"] if m["brier"] is not None else 1.0
        return (roc, pr, -brier)

    def _reg_key(m: Dict[str, Any]) -> Tuple[float, float]:
        mae = m["mae"] if m["mae"] is not None else 1e9
        sp = m["spearman"] if m["spearman"] is not None else -1.0
        return (-mae, sp)

    if task_type == "classification":
        best_single = max(stream_names, key=lambda s: _cls_key(single_metrics[s]))
    else:
        best_single = max(stream_names, key=lambda s: _reg_key(single_metrics[s]))

    # 2. Fit nested 5-fold OOF simplex super-ensemble (fit on 4 folds, evaluate on 5th fold)
    # Use clean non-redundant streams for simplex fitting: on half_life_*, include cpu, fusion,
    # deep_specialist, deep_multitask (which strictly spans deep = 0.5*sp + 0.5*mt!).
    if endpoint in ("half_life_reg", "half_life_cls") and "deep_specialist" in streams_oof and "deep_multitask" in streams_oof:
        simplex_names = [s for s in ("cpu", "fusion", "deep_specialist", "deep_multitask") if s in streams_oof]
    else:
        simplex_names = [s for s in ("cpu", "fusion", "deep") if s in streams_oof]

    P_oof = np.column_stack([streams_oof[s] for s in simplex_names])
    nested_oof_super = np.full(n_cv, np.nan, dtype=np.float64)
    fold_weights_log: List[Dict[str, float]] = []

    for f in range(5):
        va = (fids_cv == f) & eval_mask
        tr = (fids_cv != f) & eval_mask
        if task_type == "regression":
            tr = tr & (ct_cv == "exact")
        if tr.sum() < 4 or va.sum() == 0:
            nested_oof_super[va] = streams_oof[best_single][va]
            continue
        w_f = _fit_simplex_weights(P_oof[tr], y_cv[tr], task_type=task_type, reg=0.005, prune_tol=0.02)
        w_dict_f = {s: float(w_f[i]) for i, s in enumerate(simplex_names)}
        fold_weights_log.append({s: round(float(w_f[i]), 4) for i, s in enumerate(simplex_names)})
        sub_streams = {s: streams_oof[s][va] for s in simplex_names}
        nested_oof_super[va] = _apply_simplex_blend(
            simplex_names, sub_streams, w_dict_f, task_type=task_type, bounds=bounds, fallback=fallback
        )

    if task_type == "classification":
        super_oof_m = classification_metrics(y_cv[eval_mask], nested_oof_super[eval_mask])
        best_s_m = single_metrics[best_single]
        delta_primary = (super_oof_m["roc_auc"] or 0.0) - (best_s_m["roc_auc"] or 0.0)
        adopt_super = delta_primary >= 0.0005
    else:
        super_oof_m = regression_metrics(
            y_cv[eval_mask], nested_oof_super[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask]
        )
        best_s_m = single_metrics[best_single]
        delta_primary = (best_s_m["mae"] or 1e9) - (super_oof_m["mae"] or 1e9)  # positive = MAE reduction
        adopt_super = delta_primary >= 0.0005
        # Guard heavily censored endpoints against material degradation in censored violation MAE
        n_cens_cv = int((ct_cv[eval_mask] != "exact").sum())
        if adopt_super and n_cens_cv >= 20:
            v_best = best_s_m.get("censored_violation_mae")
            v_sup = super_oof_m.get("censored_violation_mae")
            if v_best is not None and v_sup is not None and v_sup > v_best + 0.05:
                adopt_super = False

    # 3. Freeze final champion weights on ALL 5 CV folds (still ZERO holdout peeking!)
    if adopt_super:
        fit_mask = eval_mask.copy()
        if task_type == "regression":
            fit_mask = fit_mask & (ct_cv == "exact")
        w_final = _fit_simplex_weights(P_oof[fit_mask], y_cv[fit_mask], task_type=task_type, reg=0.005, prune_tol=0.02)
        frozen_weights = {s: round(float(w_final[i]), 5) for i, s in enumerate(simplex_names) if w_final[i] > 0.0}
        # Re-normalize after rounding so sum == 1.00000
        s_tot = sum(frozen_weights.values())
        frozen_weights = {k: round(v / s_tot, 5) for k, v in frozen_weights.items()}
        # Check if pruning collapsed super_ensemble into a single family!
        if len(frozen_weights) == 1:
            only_fam = next(iter(frozen_weights.keys()))
            winning_family = only_fam
            champion_oof_raw = streams_oof[only_fam].copy()
            adopt_super = False
            frozen_weights = {only_fam: 1.0}
        else:
            winning_family = "super_ensemble"
            champion_oof_raw = nested_oof_super.copy()
    else:
        winning_family = best_single
        frozen_weights = {best_single: 1.0}
        champion_oof_raw = streams_oof[best_single].copy()

    # 4. Fit strict OOF calibration (cls) or Conformal intervals (reg)
    cal_spec = None
    conf_spec = None
    if task_type == "classification":
        cal_res = _fit_and_select_calibrator_oof(champion_oof_raw, y_cv, fids_cv, eval_mask)
        cal_spec = {
            "method": cal_res["method"],
            "params": cal_res["params"],
            "nested_oof_comparison": cal_res["nested_oof_comparison"],
        }
        champion_oof_final = cal_res["nested_oof_calibrated"]
        champion_cv_metrics = classification_metrics(y_cv[eval_mask], champion_oof_final[eval_mask])
        champion_cv_metrics_raw = classification_metrics(y_cv[eval_mask], champion_oof_raw[eval_mask])
    else:
        conf_spec = _fit_conformal_oof(
            y_cv, champion_oof_raw, ct_cv, fids_cv, eval_mask, oof_max_tanimoto, bounds=bounds
        )
        champion_oof_final = champion_oof_raw
        champion_cv_metrics = regression_metrics(
            y_cv[eval_mask], champion_oof_final[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask]
        )
        champion_cv_metrics_raw = None

    return {
        "track": track,
        "winning_family": winning_family,
        "is_super_ensemble": bool(adopt_super),
        "weights": frozen_weights,
        "best_single_family": best_single,
        "nested_super_delta_over_best_single": round(float(delta_primary), 5),
        "single_family_oof_metrics": single_metrics,
        "nested_super_oof_metrics": super_oof_m,
        "nested_fold_weights": fold_weights_log,
        "champion_cv_metrics": champion_cv_metrics,
        "champion_cv_metrics_raw": champion_cv_metrics_raw,
        "calibrator": cal_spec,
        "conformal": conf_spec,
        "champion_oof_raw": champion_oof_raw,
        "champion_oof_final": champion_oof_final,
    }


def _select_censored_head_oof(
    y_cv: np.ndarray,
    yl_cv: np.ndarray,
    yu_cv: np.ndarray,
    ct_cv: np.ndarray,
    fids_cv: np.ndarray,
    eval_mask: np.ndarray,
    bounds: Optional[Tuple[float, float]],
    cand_oof: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Build strictly OOF-optimized censored-aware diagnostic/blended head minimizing all-row hinge MAE."""
    names = [k for k in ("champion_exact", "cpu_aft", "fusion_aft", "deep") if k in cand_oof and cand_oof[k] is not None]
    n_cv = len(y_cv)
    fallback = float(np.nanmean(y_cv[eval_mask])) if eval_mask.any() else 0.0

    # Evaluate individual candidates
    single_m = {
        k: regression_metrics(y_cv[eval_mask], cand_oof[k][eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask])
        for k in names
    }
    best_single = min(names, key=lambda k: single_m[k]["all_row_hinge_mae"] if single_m[k]["all_row_hinge_mae"] is not None else 1e9)

    P = np.column_stack([cand_oof[k] for k in names])
    nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
    for f in range(5):
        va = (fids_cv == f) & eval_mask
        tr = (fids_cv != f) & eval_mask
        if tr.sum() < 6 or va.sum() == 0:
            nested_oof[va] = cand_oof[best_single][va]
            continue
        w_f = _fit_simplex_weights(
            P[tr], y_cv[tr], task_type="regression", reg=0.005, prune_tol=0.02,
            hinge_mode=True, y_lower=yl_cv[tr], y_upper=yu_cv[tr], censor_type=ct_cv[tr],
        )
        wd = {k: float(w_f[i]) for i, k in enumerate(names)}
        nested_oof[va] = _apply_simplex_blend(
            names, {k: cand_oof[k][va] for k in names}, wd, task_type="regression", bounds=bounds, fallback=fallback
        )

    blend_m = regression_metrics(y_cv[eval_mask], nested_oof[eval_mask], yl_cv[eval_mask], yu_cv[eval_mask], ct_cv[eval_mask])
    h_best = single_m[best_single]["all_row_hinge_mae"] or 1e9
    h_blend = blend_m["all_row_hinge_mae"] or 1e9

    if h_best - h_blend >= 0.0005:
        w_full = _fit_simplex_weights(
            P[eval_mask], y_cv[eval_mask], task_type="regression", reg=0.005, prune_tol=0.02,
            hinge_mode=True, y_lower=yl_cv[eval_mask], y_upper=yu_cv[eval_mask], censor_type=ct_cv[eval_mask],
        )
        weights = {k: round(float(w_full[i]), 5) for i, k in enumerate(names) if w_full[i] > 0.0}
        s_tot = sum(weights.values())
        weights = {k: round(v / s_tot, 5) for k, v in weights.items()}
        mode = "censored_blend" if len(weights) > 1 else next(iter(weights.keys()))
        oof_final = nested_oof if len(weights) > 1 else cand_oof[mode]
        cv_m = blend_m if len(weights) > 1 else single_m[mode]
    else:
        mode = best_single
        weights = {best_single: 1.0}
        oof_final = cand_oof[best_single]
        cv_m = single_m[best_single]

    return {
        "mode": mode,
        "weights": weights,
        "candidate_oof_metrics": single_m,
        "cv_metrics": cv_m,
        "oof_pred": oof_final,
    }


def _compute_deep_specialist_case17_exact(
    endpoint: str,
    bank_phys: np.ndarray,
    bank_smi_valid: np.ndarray,
    bank_seq_valid: np.ndarray,
    embs: Dict[str, np.ndarray],
    case17_idx: np.ndarray,
    cond_override: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """Run exact Leave-Case17-Out HalfLifeAttentionNet checkpoints on CPU to separate
    deep_specialist vs deep_multitask on Case17 without any approximation.
    """
    import torch
    from inference_api import load_deep_checkpoint, predict_from_numpy

    phys = bank_phys[case17_idx]
    sv = bank_smi_valid[case17_idx]
    qv = bank_seq_valid[case17_idx]
    nc_s = embs["ncaa_smi"][case17_idx]
    nc_f = embs["ncaa_fus"][case17_idx]
    nc_p = embs["ncaa_prot"][case17_idx]
    pc = embs["pclm2"][case17_idx]
    cb = embs["chemberta"][case17_idx]
    es = embs["esmc_4reg"][case17_idx]
    cond_rows = [cond_override] * len(case17_idx) if cond_override is not None else None

    out = {}
    for mode in ("dual_modality", "smiles_only"):
        preds = []
        for f in range(5):
            ck_path = DEEP_DIR / "halflife_specialist" / f"{endpoint}_halflife_specialist_case17_fold{f}.pt"
            model, ckpt = load_deep_checkpoint(ck_path, device="cpu")
            mu, _ = predict_from_numpy(
                model, ckpt, endpoint, phys, nc_s, nc_f, nc_p, pc, cb, es, sv, qv,
                cond_rows=cond_rows, eval_mode=mode, device="cpu",
            )
            preds.append(mu)
        out[mode] = np.mean(np.stack(preds, axis=0), axis=0)
    return out


def _encode_seq_3mers(seq: str) -> List[int]:
    clean = [c for c in (seq or "").upper() if c in AA20_IDX]
    if len(clean) < 3:
        return []
    seen: Set[int] = set()
    for i in range(len(clean) - 2):
        code = AA20_IDX[clean[i]] * 400 + AA20_IDX[clean[i + 1]] * 20 + AA20_IDX[clean[i + 2]]
        seen.add(code)
    return sorted(seen)


def build_ad_and_analog_index(
    compounds_meta: Dict[str, Dict[str, str]],
    key_to_idx: Dict[str, int],
    morgan_packed: np.ndarray,
    smiles_valid: np.ndarray,
    seq_valid: np.ndarray,
) -> Tuple[Path, Path]:
    """Build ultra-fast global AD + Top-3 training analog index across all 24 endpoints.

    Stores ONE global matrix of unique reference compounds (packed Morgan 128-byte + 3-mer CSR)
    plus per-endpoint index slices, enabling <15ms exact Tanimoto + Top-3 analog lookup across
    ALL 24 endpoints simultaneously!
    """
    t0 = time.time()
    print("\n[AD INDEX] Building global AD + Top-3 Nearest Analogs Index across 24 endpoints...", flush=True)

    # Collect unique reference compounds across all 24 frozen endpoint TSVs (CV + holdout reference corpus)
    ep_analog_records: Dict[str, List[Dict[str, Any]]] = {}
    ref_keys_set: Set[str] = set()

    for ep in ALL_ENDPOINTS:
        rows = _read_tsv(ENDPOINTS_DIR / f"{ep}.tsv")
        # Group by compound_key within endpoint so Top-3 analogs are distinct compounds!
        by_ck: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for r in rows:
            by_ck[r["compound_key"]].append(r)

        ep_recs: List[Dict[str, Any]] = []
        spec = ENDPOINT_SPECS[ep]
        unit = spec["unit"]
        for ck in sorted(by_ck.keys()):
            ref_keys_set.add(ck)
            obs = by_ck[ck]
            # Prefer exact observations if present, else censored
            exact_obs = [o for o in obs if o["censor_type"] == "exact"]
            use_obs = exact_obs if exact_obs else obs
            vals = [float(o["y_value"]) for o in use_obs]
            med_val = float(np.median(vals))
            ctype = use_obs[0]["censor_type"]
            meta = compounds_meta.get(ck, {})
            ep_recs.append({
                "compound_key": ck,
                "smiles": meta.get("canonical_smiles", ""),
                "sequence": meta.get("parent_sequence", ""),
                "experimental_value": round(med_val, 5),
                "censor_type": ctype,
                "unit": unit,
                "n_obs": len(obs),
            })
        ep_analog_records[ep] = ep_recs

    # Build unified global reference list (sorted by compound_idx)
    ref_keys = sorted(ref_keys_set, key=lambda k: key_to_idx[k])
    ref_key_to_gidx = {k: i for i, k in enumerate(ref_keys)}
    ref_cidx = np.array([key_to_idx[k] for k in ref_keys], dtype=np.int32)
    n_ref = len(ref_keys)

    ref_morgan_packed = np.ascontiguousarray(morgan_packed[ref_cidx], dtype=np.uint8)
    ref_smi_valid = np.ascontiguousarray(smiles_valid[ref_cidx], dtype=bool)
    ref_seq_valid = np.ascontiguousarray(seq_valid[ref_cidx], dtype=bool)

    # Build sparse 3-mer CSR matrix (n_ref x 8000) for fast exact sequence 3-mer Jaccard fallback
    indptr = [0]
    indices = []
    seq_3mer_pop = np.zeros(n_ref, dtype=np.float32)
    for i, ck in enumerate(ref_keys):
        seq = compounds_meta.get(ck, {}).get("parent_sequence", "")
        kmers = _encode_seq_3mers(seq) if ref_seq_valid[i] else []
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

    # Per-endpoint reference index arrays + compact analog lookup dict
    analog_lookup: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v33_analog_lookup_v1",
        "ad_thresholds": {"in_domain": AD_IN_DOMAIN_THR, "borderline": AD_BORDERLINE_THR},
        "n_unique_ref_compounds": n_ref,
        "endpoints": {},
    }
    for ep in ALL_ENDPOINTS:
        recs = ep_analog_records[ep]
        g_idx = np.array([ref_key_to_gidx[r["compound_key"]] for r in recs], dtype=np.int32)
        npz_dict[f"ep_ref_idx__{ep}"] = g_idx
        analog_lookup["endpoints"][ep] = {
            "n_unique_compounds": len(recs),
            "records": recs,
        }

    ad_npz_path = BUNDLE_DIR / "ad_analog_index.npz"
    np.savez_compressed(ad_npz_path, **npz_dict)

    analog_gz_path = BUNDLE_DIR / "analog_lookup.json.gz"
    with gzip.open(analog_gz_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        json.dump(analog_lookup, gz, ensure_ascii=False, separators=(",", ":"))

    print(
        f"[AD INDEX DONE] {n_ref:,} unique ref compounds | {ad_npz_path.name}={ad_npz_path.stat().st_size/1e6:.2f}MB "
        f"| {analog_gz_path.name}={analog_gz_path.stat().st_size/1e6:.2f}MB ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return ad_npz_path, analog_gz_path


def build_pruned_serving_models(
    champions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Create lean pruned serving copies under /data/qlyu/v33_upgrade/bundle/serving_models/
    containing ONLY the champion 5-fold estimators actually needed at inference time!
    """
    t0 = time.time()
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[PRUNE SERVING] Building lean champion-only serving models...", flush=True)

    # Step 1: Determine exact component requirements for every endpoint
    # Track which CPU sub-models and Fusion heads are needed for each endpoint
    need_cpu_track: Dict[str, Set[str]] = defaultdict(set)  # ep -> {'dual_modality', 'smiles_only', 'aft_dual', 'aft_smiles', 'seq_only'}
    need_fusion_track: Dict[str, Set[str]] = defaultdict(set)  # ep -> {'dual_modality', 'smiles_only', 'aft_dual', 'aft_smiles'}
    need_deep_ckpt: Dict[str, Set[str]] = defaultdict(set)  # ep -> {'multitask', 'halflife_specialist'}

    for ep, cdoc in champions.items():
        for track in ("dual_modality", "smiles_only"):
            w = cdoc[track]["weights"]
            if w.get("cpu", 0.0) > 0.0:
                need_cpu_track[ep].add(track)
            if w.get("fusion", 0.0) > 0.0:
                need_fusion_track[ep].add(track)
            if w.get("deep", 0.0) > 0.0:
                need_deep_ckpt[ep].add("multitask")
                if ep in ("half_life_reg", "half_life_cls"):
                    need_deep_ckpt[ep].add("halflife_specialist")
            if w.get("deep_multitask", 0.0) > 0.0:
                need_deep_ckpt[ep].add("multitask")
            if w.get("deep_specialist", 0.0) > 0.0:
                need_deep_ckpt[ep].add("halflife_specialist")

        # Also keep censored diagnostic head components for the 9 key censored endpoints
        if ep in KEY_CENSORED_ENDPOINTS and cdoc.get("censored_head_dual") is not None:
            cw = cdoc["censored_head_dual"]["weights"]
            if cw.get("cpu_aft", 0.0) > 0.0:
                need_cpu_track[ep].add("aft_dual")
                need_cpu_track[ep].add("aft_smiles")
            if cw.get("fusion_aft", 0.0) > 0.0:
                need_fusion_track[ep].add("aft_dual")
                need_fusion_track[ep].add("aft_smiles")
            if cw.get("deep", 0.0) > 0.0:
                need_deep_ckpt[ep].add("multitask")
                if ep in ("half_life_reg", "half_life_cls"):
                    need_deep_ckpt[ep].add("halflife_specialist")

    # Inspect fusion bundles first to discover any transitive CPU dependencies (`cpu_dual`, `cpu_smiles`, `cpu_aft_*`, `cpu_seq_only`)
    fusion_active_heads: Dict[str, Set[str]] = defaultdict(set)
    for ep in sorted(need_fusion_track.keys()):
        f_bundle = joblib.load(FUSION_DIR / ep / "fusion_models_5fold.joblib")
        tracks_needed = need_fusion_track[ep]
        act_heads: Set[str] = set()
        if "dual_modality" in tracks_needed:
            st = f_bundle["dual_modality"]["stacker"]
            act_heads.update(st.active_heads(tol=1e-9))
        if "smiles_only" in tracks_needed:
            st = f_bundle["smiles_only"]["stacker"]
            act_heads.update(st.active_heads(tol=1e-9))
        if ("aft_dual" in tracks_needed or "aft_smiles" in tracks_needed) and f_bundle.get("aft") is not None:
            if "aft_dual" in tracks_needed:
                act_heads.update(f_bundle["aft"]["stacker_dual"].active_heads(tol=1e-9))
            if "aft_smiles" in tracks_needed:
                act_heads.update(f_bundle["aft"]["stacker_smiles_only"].active_heads(tol=1e-9))
        fusion_active_heads[ep] = act_heads

        # Map CPU heads referenced inside fusion stackers
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

        # Write pruned fusion serving bundle (ONLY active pretrained_base_heads + needed stackers; no ad_index bloat!)
        pruned_heads = {
            hk: f_bundle["pretrained_base_heads"][hk]
            for hk in sorted(act_heads)
            if hk in f_bundle["pretrained_base_heads"]
        }
        pruned_fusion = {
            "schema": "v33_pruned_fusion_serving_v1",
            "endpoint": ep,
            "task_type": f_bundle["task_type"],
            "bounds": f_bundle["bounds"],
            "cond_cols": f_bundle["cond_cols"],
            "canonical_default_conditions": f_bundle["canonical_default_conditions"],
            "pretrained_base_heads": pruned_heads,
            "dual_modality": (
                {
                    "candidate_heads": f_bundle["dual_modality"]["candidate_heads"],
                    "stacker": f_bundle["dual_modality"]["stacker"],
                    "active_heads": f_bundle["dual_modality"]["stacker"].active_heads(tol=1e-9),
                }
                if "dual_modality" in tracks_needed
                else None
            ),
            "smiles_only": (
                {
                    "candidate_heads": f_bundle["smiles_only"]["candidate_heads"],
                    "stacker": f_bundle["smiles_only"]["stacker"],
                    "active_heads": f_bundle["smiles_only"]["stacker"].active_heads(tol=1e-9),
                }
                if "smiles_only" in tracks_needed
                else None
            ),
            "aft": (
                {
                    "stacker_dual": f_bundle["aft"]["stacker_dual"],
                    "stacker_smiles_only": f_bundle["aft"]["stacker_smiles_only"],
                }
                if (("aft_dual" in tracks_needed or "aft_smiles" in tracks_needed) and f_bundle.get("aft") is not None)
                else None
            ),
        }
        out_fp = SERVING_DIR / f"{ep}_fusion.joblib"
        joblib.dump(pruned_fusion, out_fp, compress=1)
        orig_mb = (FUSION_DIR / ep / "fusion_models_5fold.joblib").stat().st_size / 1e6
        new_mb = out_fp.stat().st_size / 1e6
        print(
            f"  [fusion prune] {ep:<24} heads={len(pruned_heads)}/{len(f_bundle['pretrained_base_heads'])} | {orig_mb:6.1f}MB -> {new_mb:6.1f}MB",
            flush=True,
        )
        del f_bundle, pruned_fusion

    # Check if any required CPU endpoint uses cross-endpoint meta-features -> add AUX_ENDPOINTS!
    aux_tracks_needed: Set[str] = set()
    for ep in list(need_cpu_track.keys()):
        cm = json.loads((CPU_DIR / ep / "metrics.json").read_text(encoding="utf-8"))
        if ("dual_modality" in need_cpu_track[ep] or "aft_dual" in need_cpu_track[ep]) and cm["winning_recipe_dual"].get("uses_meta"):
            aux_tracks_needed.add("dual_modality")
        if ("smiles_only" in need_cpu_track[ep] or "aft_smiles" in need_cpu_track[ep]) and cm["winning_recipe_smiles_only"].get("uses_meta"):
            aux_tracks_needed.add("smiles_only")
    for aux_ep in AUX_ENDPOINTS:
        for tr in aux_tracks_needed:
            need_cpu_track[aux_ep].add(tr)

    # Write pruned CPU serving bundles
    for ep in sorted(need_cpu_track.keys()):
        c_bundle = joblib.load(CPU_DIR / ep / "models_5fold.joblib")
        tracks_needed = need_cpu_track[ep]
        pruned_cpu = {
            "schema": "v33_pruned_cpu_serving_v1",
            "endpoint": ep,
            "task_type": c_bundle["task_type"],
            "bounds": c_bundle["bounds"],
            "cond_cols": c_bundle["cond_cols"],
            "canonical_default_conditions": c_bundle["canonical_default_conditions"],
            "dual_modality": (
                {
                    "uses_meta": c_bundle["dual_modality"]["uses_meta"],
                    "meta_endpoints": c_bundle["dual_modality"]["meta_endpoints"],
                    "folds": c_bundle["dual_modality"]["folds"],
                    "calibrator": c_bundle["dual_modality"]["calibrator"] if ep in AUX_ENDPOINTS else None,
                }
                if "dual_modality" in tracks_needed
                else None
            ),
            "smiles_only": (
                {
                    "uses_meta": c_bundle["smiles_only"]["uses_meta"],
                    "meta_endpoints": c_bundle["smiles_only"]["meta_endpoints"],
                    "folds": c_bundle["smiles_only"]["folds"],
                    "calibrator": c_bundle["smiles_only"]["calibrator"] if ep in AUX_ENDPOINTS else None,
                }
                if "smiles_only" in tracks_needed
                else None
            ),
            "seq_only": (
                {"folds": c_bundle["seq_only"]["folds"]}
                if ("seq_only" in tracks_needed and c_bundle.get("seq_only") is not None)
                else None
            ),
            "aft": (
                {
                    "folds_dual": c_bundle["aft"]["folds_dual"] if "aft_dual" in tracks_needed else None,
                    "folds_smiles_only": c_bundle["aft"]["folds_smiles_only"] if "aft_smiles" in tracks_needed else None,
                }
                if (("aft_dual" in tracks_needed or "aft_smiles" in tracks_needed) and c_bundle.get("aft") is not None)
                else None
            ),
        }
        out_cp = SERVING_DIR / f"{ep}_cpu.joblib"
        joblib.dump(pruned_cpu, out_cp, compress=1)
        del c_bundle, pruned_cpu

    # Collect deep checkpoint paths (referenced read-only from models_gpu_deep)
    deep_refs: Dict[str, Dict[str, List[str]]] = {}
    for ep in sorted(need_deep_ckpt.keys()):
        deep_refs[ep] = {}
        if "multitask" in need_deep_ckpt[ep]:
            deep_refs[ep]["multitask"] = [
                str(DEEP_DIR / "multitask_checkpoints" / "per_endpoint" / f"{ep}_multitask_fold{f}.pt")
                for f in range(5)
            ]
            for p in deep_refs[ep]["multitask"]:
                assert Path(p).exists(), f"Missing deep multitask ckpt: {p}"
        if "halflife_specialist" in need_deep_ckpt[ep]:
            deep_refs[ep]["halflife_specialist"] = [
                str(DEEP_DIR / "halflife_specialist" / f"{ep}_halflife_specialist_fold{f}.pt")
                for f in range(5)
            ]
            for p in deep_refs[ep]["halflife_specialist"]:
                assert Path(p).exists(), f"Missing deep specialist ckpt: {p}"

    total_pruned_mb = sum(p.stat().st_size for p in SERVING_DIR.glob("*.joblib")) / 1e6
    print(
        f"[PRUNE SERVING DONE] {len(need_cpu_track)} CPU bundles + {len(need_fusion_track)} Fusion bundles "
        f"= {total_pruned_mb:.1f} MB (vs 10,515 MB raw!) + {len(deep_refs)} Deep endpoints ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return {
        "serving_dir": str(SERVING_DIR),
        "total_pruned_joblib_mb": round(total_pruned_mb, 2),
        "cpu_endpoints_bundled": sorted(need_cpu_track.keys()),
        "cpu_aux_tracks_needed": sorted(aux_tracks_needed),
        "fusion_endpoints_bundled": sorted(need_fusion_track.keys()),
        "deep_checkpoint_refs": deep_refs,
    }


def main() -> None:
    t_all = time.time()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load V32 audited reference snapshot
    v32_snap_path = BUNDLE_DIR / "v32_reference_snapshot.json"
    assert v32_snap_path.exists(), f"Missing {v32_snap_path}! Upload from local first."
    v32_snap = json.loads(v32_snap_path.read_text(encoding="utf-8"))

    # Load compound registry & feature index
    print("[1/6] Loading compound registry, feature masks, and Case17 queries...", flush=True)
    compounds_meta: Dict[str, Dict[str, str]] = {}
    with (DATA_DIR / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            compounds_meta[r["compound_key"]] = r
    n_compounds_v33 = len(compounds_meta)

    key_to_idx: Dict[str, int] = {}
    with (FEATURES_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            key_to_idx[r["compound_key"]] = int(r["compound_idx"])

    pc = np.load(FEATURES_DIR / "physchem_core.npz")
    physchem = np.ascontiguousarray(pc["X"], dtype=np.float32)
    smiles_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
    seq_valid = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
    pc.close()

    mb = np.load(FEATURES_DIR / "morgan_bits_1024.npz")
    morgan_packed = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
    mb.close()

    case17_queries = _read_tsv(DATA_DIR / "case17_queries.tsv")
    for q in case17_queries:
        q["compound_idx"] = key_to_idx[q["compound_key"]]
    case17_idx = np.array([q["compound_idx"] for q in case17_queries], dtype=np.int32)
    assert len(case17_queries) == 17

    # =========================================================================
    # PHASE A: STRICTLY OOF-ONLY SELECTION (ZERO HOLDOUT PEEKING!)
    # =========================================================================
    print("[2/6] PHASE A — Strictly OOF-only cross-family super-ensemble selection (24 endpoints x 2 tracks)...", flush=True)

    oof_frozen: Dict[str, Dict[str, Any]] = {}
    ep_meta_counts: Dict[str, Dict[str, Any]] = {}
    total_rows_all = 0
    total_cv_all = 0
    total_ho_all = 0
    total_censored_all = 0

    for ep in ALL_ENDPOINTS:
        spec = ENDPOINT_SPECS[ep]
        task_type = spec["task_type"]
        bounds = spec.get("bounds")

        # Load frozen endpoint table to verify exact alignment & cluster isolation
        ep_rows = _read_tsv(ENDPOINTS_DIR / f"{ep}.tsv")
        cv_ep_rows = [r for r in ep_rows if int(r["is_holdout"]) == 0]
        ho_ep_rows = [r for r in ep_rows if int(r["is_holdout"]) == 1]
        cv_clusters = {r["cluster_id"] for r in cv_ep_rows}
        ho_clusters = {r["cluster_id"] for r in ho_ep_rows}
        assert cv_clusters.isdisjoint(ho_clusters), f"{ep}: cluster leakage between CV and Holdout!"

        n_tot = len(ep_rows)
        n_cv = len(cv_ep_rows)
        n_ho = len(ho_ep_rows)
        n_cens_tot = sum(1 for r in ep_rows if r["censor_type"] != "exact")
        n_cens_cv = sum(1 for r in cv_ep_rows if r["censor_type"] != "exact")
        n_cens_ho = sum(1 for r in ho_ep_rows if r["censor_type"] != "exact")
        total_rows_all += n_tot
        total_cv_all += n_cv
        total_ho_all += n_ho
        total_censored_all += n_cens_tot

        ep_meta_counts[ep] = {
            "n_total": n_tot,
            "n_cv": n_cv,
            "n_holdout": n_ho,
            "n_censored_total": n_cens_tot,
            "n_censored_cv": n_cens_cv,
            "n_censored_holdout": n_cens_ho,
            "censor_counts": {k: sum(1 for r in ep_rows if r["censor_type"] == k) for k in ("exact", "right", "left", "interval")},
            "n_case17_excluded": sum(1 for r in ep_rows if int(r["case17_excluded"]) == 1),
        }

        # Load ONLY OOF predictions from the 3 families (NEVER touch holdout_predictions.tsv in Phase A!)
        cpu_oof = _read_tsv(CPU_DIR / ep / "oof_predictions.tsv")
        fus_oof = _read_tsv(FUSION_DIR / ep / "oof_predictions.tsv")
        dep_oof = _read_tsv(DEEP_DIR / ep / "oof_predictions.tsv")

        assert len(cpu_oof) == len(fus_oof) == len(dep_oof) == n_cv, f"{ep}: OOF row count mismatch!"
        for i in range(n_cv):
            rid = cv_ep_rows[i]["record_id"]
            ck = cv_ep_rows[i]["compound_key"]
            assert cpu_oof[i]["record_id"] == fus_oof[i]["record_id"] == dep_oof[i]["record_id"] == rid, f"{ep}: OOF record_id mismatch at {i}"
            assert cpu_oof[i]["compound_key"] == fus_oof[i]["compound_key"] == dep_oof[i]["compound_key"] == ck, f"{ep}: OOF compound_key mismatch at {i}"

        cidx_cv = np.array([key_to_idx[r["compound_key"]] for r in cv_ep_rows], dtype=np.int32)
        smi_mask_cv = smiles_valid[cidx_cv]
        dual_mask_cv = np.ones(n_cv, dtype=bool)

        y_cv = np.array([float(r["y_value"]) for r in cv_ep_rows], dtype=np.float64)
        yl_cv = np.array([float(r["y_lower"]) for r in cv_ep_rows], dtype=np.float64)
        yu_cv = np.array([float(r["y_upper"]) for r in cv_ep_rows], dtype=np.float64)
        ct_cv = np.array([r["censor_type"] for r in cv_ep_rows], dtype=object)
        fids_cv = np.array([int(r["fold_id"]) for r in cv_ep_rows], dtype=np.int32)
        oof_max_t = _col_arr(cpu_oof, "max_tanimoto_oof")

        # Build OOF streams for both tracks
        streams_dual_oof: Dict[str, np.ndarray] = {
            "cpu": _col_arr(cpu_oof, "pred_dual"),
            "fusion": _col_arr(fus_oof, "pred_dual"),
            "deep": _col_arr(dep_oof, "pred_dual"),
        }
        streams_smi_oof: Dict[str, np.ndarray] = {
            "cpu": _col_arr(cpu_oof, "pred_smiles_only"),
            "fusion": _col_arr(fus_oof, "pred_smiles_only"),
            "deep": _col_arr(dep_oof, "pred_smiles_only"),
        }
        stream_names = ["cpu", "fusion", "deep"]

        if ep in ("half_life_reg", "half_life_cls"):
            streams_dual_oof["deep_specialist"] = _col_arr(dep_oof, "pred_halflife_specialist_dual")
            streams_dual_oof["deep_multitask"] = _col_arr(dep_oof, "pred_multitask_dual")
            streams_smi_oof["deep_specialist"] = _col_arr(dep_oof, "pred_halflife_specialist_smiles_only")
            streams_smi_oof["deep_multitask"] = _col_arr(dep_oof, "pred_multitask_smiles_only")
            stream_names = ["cpu", "fusion", "deep", "deep_specialist", "deep_multitask"]

        res_dual = _select_champion_oof(
            ep, "dual_modality", task_type, bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv,
            dual_mask_cv, streams_dual_oof, stream_names, oof_max_t,
        )
        res_smi = _select_champion_oof(
            ep, "smiles_only", task_type, bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv,
            smi_mask_cv, streams_smi_oof, stream_names, oof_max_t,
        )

        # Censored-aware diagnostic/blended head on all endpoints with censored CV rows
        cens_head_dual = None
        cens_head_smi = None
        if task_type == "regression" and n_cens_cv >= 1:
            cpu_aft_d = _col_arr(cpu_oof, "pred_aft")
            cpu_aft_s = _col_arr(cpu_oof, "pred_aft_smiles_only")
            fus_aft_d = _col_arr(fus_oof, "pred_aft")
            fus_aft_s = _col_arr(fus_oof, "pred_aft_smiles_only")
            cens_head_dual = _select_censored_head_oof(
                y_cv, yl_cv, yu_cv, ct_cv, fids_cv, dual_mask_cv, bounds,
                {
                    "champion_exact": res_dual["champion_oof_final"],
                    "cpu_aft": cpu_aft_d,
                    "fusion_aft": fus_aft_d,
                    "deep": streams_dual_oof["deep"],
                },
            )
            cens_head_smi = _select_censored_head_oof(
                y_cv, yl_cv, yu_cv, ct_cv, fids_cv, smi_mask_cv, bounds,
                {
                    "champion_exact": res_smi["champion_oof_final"],
                    "cpu_aft": cpu_aft_s,
                    "fusion_aft": fus_aft_s,
                    "deep": streams_smi_oof["deep"],
                },
            )

        oof_frozen[ep] = {
            "dual_modality": res_dual,
            "smiles_only": res_smi,
            "censored_head_dual": cens_head_dual,
            "censored_head_smi": cens_head_smi,
            "cv_ep_rows": cv_ep_rows,
            "ho_ep_rows": ho_ep_rows,
            "y_cv_mean_dual": float(np.nanmean(y_cv)),
            "y_cv_mean_smi": float(np.nanmean(y_cv[smi_mask_cv])) if smi_mask_cv.any() else float(np.nanmean(y_cv)),
        }

        if task_type == "classification":
            print(
                f"  [OOF] {ep:<24} | DUAL={res_dual['winning_family']:<14} {res_dual['weights']} CV ROC={res_dual['champion_cv_metrics']['roc_auc']:.4f} "
                f"| SMI={res_smi['winning_family']:<14} CV ROC={res_smi['champion_cv_metrics']['roc_auc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  [OOF] {ep:<24} | DUAL={res_dual['winning_family']:<14} {res_dual['weights']} CV MAE={res_dual['champion_cv_metrics']['mae']:.4f} Sp={res_dual['champion_cv_metrics']['spearman']:.3f} "
                f"| SMI={res_smi['winning_family']:<14} CV MAE={res_smi['champion_cv_metrics']['mae']:.4f}",
                flush=True,
            )

    assert total_rows_all == 69111, f"Expected 69,111 rows, got {total_rows_all}"
    assert total_cv_all == 55299, f"Expected 55,299 CV rows, got {total_cv_all}"
    assert total_ho_all == 13812, f"Expected 13,812 HO rows, got {total_ho_all}"
    assert total_censored_all == 2523, f"Expected 2,523 censored rows, got {total_censored_all}"
    print(f"  [PHASE A LOCKED] Verified 24 endpoints, {total_rows_all:,} rows ({total_cv_all:,} CV + {total_ho_all:,} HO, {total_censored_all:,} censored). ALL champion weights/calibrators/conformal FROZEN!", flush=True)

    # =========================================================================
    # PHASE B: SINGLE FROZEN EVALUATION ON UNTOUCHED 20% HOLDOUT & LEAVE-CASE17-OUT
    # =========================================================================
    print("\n[3/6] PHASE B — Evaluating FROZEN champions ONCE on untouched 20% Holdout & Leave-Case17-Out...", flush=True)

    # Pre-check if half_life_reg or half_life_cls needs separated deep_specialist / deep_multitask on Case17
    need_hl_c17_sep = set()
    for ep in ("half_life_reg", "half_life_cls"):
        for tr in ("dual_modality", "smiles_only"):
            w = oof_frozen[ep][tr]["weights"]
            if "deep_specialist" in w or "deep_multitask" in w:
                need_hl_c17_sep.add(ep)

    hl_c17_sp_exact: Dict[str, Dict[str, Any]] = {}
    if need_hl_c17_sep:
        print(f"  Loading embeddings to run exact Leave-Case17-Out HalfLifeAttentionNet for {sorted(need_hl_c17_sep)}...", flush=True)
        d_ncaa = np.load(EMBEDDINGS_DIR / "ncaa_repdistill_v2.npz")
        d_pclm = np.load(EMBEDDINGS_DIR / "peptideclm2.npz")
        d_cb = np.load(EMBEDDINGS_DIR / "chemberta_mtr.npz")
        d_esmc = np.load(EMBEDDINGS_DIR / "esmc_600m_multiregion.npz")
        emb_c17 = {
            "ncaa_smi": d_ncaa["smiles_repr"].astype(np.float32),
            "ncaa_fus": d_ncaa["fusion_hf"].astype(np.float32),
            "ncaa_prot": d_ncaa["protein_proj"].astype(np.float32),
            "pclm2": d_pclm["embeddings"].astype(np.float32),
            "chemberta": d_cb["embeddings"].astype(np.float32),
            "esmc_4reg": np.stack([d_esmc["mean"], d_esmc["max"], d_esmc["nterm5"], d_esmc["cterm5"]], axis=1).astype(np.float32),
        }
        d_ncaa.close(); d_pclm.close(); d_cb.close(); d_esmc.close()
        for ep in sorted(need_hl_c17_sep):
            hl_c17_sp_exact[ep] = {
                "canonical": _compute_deep_specialist_case17_exact(
                    ep, physchem, smiles_valid, seq_valid, emb_c17, case17_idx, cond_override=None
                )
            }
            if ep == "half_life_reg":
                hl_c17_sp_exact[ep]["SC"] = _compute_deep_specialist_case17_exact(
                    ep, physchem, smiles_valid, seq_valid, emb_c17, case17_idx,
                    cond_override=ENDPOINT_SPECS[ep]["alt_case17_conditions"]["SC"],
                )
        del emb_c17

    endpoints_manifest: Dict[str, Any] = {}
    case17_by_ep: Dict[str, List[Dict[str, Any]]] = {}

    for ep in ALL_ENDPOINTS:
        spec = ENDPOINT_SPECS[ep]
        disp = ENDPOINT_DISPLAY[ep]
        task_type = spec["task_type"]
        bounds = spec.get("bounds")
        fr = oof_frozen[ep]
        ho_ep_rows = fr["ho_ep_rows"]
        n_ho = len(ho_ep_rows)

        cpu_ho = _read_tsv(CPU_DIR / ep / "holdout_predictions.tsv")
        fus_ho = _read_tsv(FUSION_DIR / ep / "holdout_predictions.tsv")
        dep_ho = _read_tsv(DEEP_DIR / ep / "holdout_predictions.tsv")

        assert len(cpu_ho) == len(fus_ho) == len(dep_ho) == n_ho, f"{ep}: HO row count mismatch!"
        for i in range(n_ho):
            assert cpu_ho[i]["record_id"] == fus_ho[i]["record_id"] == dep_ho[i]["record_id"] == ho_ep_rows[i]["record_id"]
            assert cpu_ho[i]["compound_key"] == fus_ho[i]["compound_key"] == dep_ho[i]["compound_key"] == ho_ep_rows[i]["compound_key"]

        cidx_ho = np.array([key_to_idx[r["compound_key"]] for r in ho_ep_rows], dtype=np.int32)
        smi_mask_ho = smiles_valid[cidx_ho]
        dual_mask_ho = np.ones(n_ho, dtype=bool)

        y_ho = np.array([float(r["y_value"]) for r in ho_ep_rows], dtype=np.float64)
        yl_ho = np.array([float(r["y_lower"]) for r in ho_ep_rows], dtype=np.float64)
        yu_ho = np.array([float(r["y_upper"]) for r in ho_ep_rows], dtype=np.float64)
        ct_ho = np.array([r["censor_type"] for r in ho_ep_rows], dtype=object)
        ho_max_t = _col_arr(cpu_ho, "max_tanimoto")

        streams_dual_ho: Dict[str, np.ndarray] = {
            "cpu": _col_arr(cpu_ho, "pred_dual"),
            "fusion": _col_arr(fus_ho, "pred_dual"),
            "deep": _col_arr(dep_ho, "pred_dual"),
        }
        streams_smi_ho: Dict[str, np.ndarray] = {
            "cpu": _col_arr(cpu_ho, "pred_smiles_only"),
            "fusion": _col_arr(fus_ho, "pred_smiles_only"),
            "deep": _col_arr(dep_ho, "pred_smiles_only"),
        }
        if ep in ("half_life_reg", "half_life_cls"):
            streams_dual_ho["deep_specialist"] = _col_arr(dep_ho, "pred_halflife_specialist_dual")
            streams_dual_ho["deep_multitask"] = _col_arr(dep_ho, "pred_multitask_dual")
            streams_smi_ho["deep_specialist"] = _col_arr(dep_ho, "pred_halflife_specialist_smiles_only")
            streams_smi_ho["deep_multitask"] = _col_arr(dep_ho, "pred_multitask_smiles_only")

        # Evaluate each individual family on holdout (using each family's calibrated probs for cls comparison!)
        fam_ho_dual: Dict[str, Any] = {}
        fam_ho_smi: Dict[str, Any] = {}
        for fam, rows_ho in (("cpu", cpu_ho), ("fusion", fus_ho), ("deep", dep_ho)):
            if task_type == "classification":
                pd_eval = _col_arr(rows_ho, "pred_dual_cal")
                if pd_eval is None:
                    pd_eval = _col_arr(rows_ho, "pred_dual")
                ps_eval = _col_arr(rows_ho, "pred_smiles_only_cal")
                if ps_eval is None:
                    ps_eval = _col_arr(rows_ho, "pred_smiles_only")
                fam_ho_dual[fam] = classification_metrics(y_ho, pd_eval)
                fam_ho_smi[fam] = classification_metrics(y_ho[smi_mask_ho], ps_eval[smi_mask_ho])
            else:
                fam_ho_dual[fam] = regression_metrics(y_ho, streams_dual_ho[fam], yl_ho, yu_ho, ct_ho)
                fam_ho_smi[fam] = regression_metrics(
                    y_ho[smi_mask_ho], streams_smi_ho[fam][smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho]
                )
        if ep in ("half_life_reg", "half_life_cls"):
            for sub in ("deep_specialist", "deep_multitask"):
                if task_type == "classification":
                    fam_ho_dual[sub] = classification_metrics(y_ho, streams_dual_ho[sub])
                    fam_ho_smi[sub] = classification_metrics(y_ho[smi_mask_ho], streams_smi_ho[sub][smi_mask_ho])
                else:
                    fam_ho_dual[sub] = regression_metrics(y_ho, streams_dual_ho[sub], yl_ho, yu_ho, ct_ho)
                    fam_ho_smi[sub] = regression_metrics(
                        y_ho[smi_mask_ho], streams_smi_ho[sub][smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho]
                    )

        # Apply FROZEN champion weights + FROZEN calibrators/conformal on Holdout
        d_fr = fr["dual_modality"]
        s_fr = fr["smiles_only"]

        ho_dual_raw = _apply_simplex_blend(
            list(d_fr["weights"].keys()), streams_dual_ho, d_fr["weights"],
            task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_dual"],
        )
        ho_smi_raw = _apply_simplex_blend(
            list(s_fr["weights"].keys()), streams_smi_ho, s_fr["weights"],
            task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_smi"],
        )

        ho_dual_iv = ho_smi_iv = None
        if task_type == "classification":
            ho_dual_final = apply_frozen_calibrator(ho_dual_raw, d_fr["calibrator"])
            ho_smi_final = apply_frozen_calibrator(ho_smi_raw, s_fr["calibrator"])
            champ_ho_dual_raw_m = classification_metrics(y_ho, ho_dual_raw)
            champ_ho_dual_m = classification_metrics(y_ho, ho_dual_final)
            champ_ho_smi_raw_m = classification_metrics(y_ho[smi_mask_ho], ho_smi_raw[smi_mask_ho])
            champ_ho_smi_m = classification_metrics(y_ho[smi_mask_ho], ho_smi_final[smi_mask_ho])
            conf_ho_dual = conf_ho_smi = None
        else:
            ho_dual_final = ho_dual_raw
            ho_smi_final = ho_smi_raw
            champ_ho_dual_raw_m = champ_ho_smi_raw_m = None
            champ_ho_dual_m = regression_metrics(y_ho, ho_dual_final, yl_ho, yu_ho, ct_ho)
            champ_ho_smi_m = regression_metrics(
                y_ho[smi_mask_ho], ho_smi_final[smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho]
            )
            ho_dual_iv = apply_frozen_conformal(ho_dual_final, d_fr["conformal"], max_tanimoto=ho_max_t)
            ho_smi_iv = apply_frozen_conformal(ho_smi_final, s_fr["conformal"], max_tanimoto=ho_max_t)
            conf_ho_dual = eval_conformal_coverage(y_ho, ho_dual_final, ct_ho, d_fr["conformal"], max_tanimoto=ho_max_t)
            conf_ho_smi = eval_conformal_coverage(
                y_ho[smi_mask_ho], ho_smi_final[smi_mask_ho], ct_ho[smi_mask_ho], s_fr["conformal"], max_tanimoto=ho_max_t[smi_mask_ho]
            )

        # Evaluate FROZEN censored-aware diagnostic head on Holdout (for censored endpoints)
        cens_doc = None
        ho_cens_dual = None
        if fr["censored_head_dual"] is not None:
            ch_d = fr["censored_head_dual"]
            ch_s = fr["censored_head_smi"]
            cpu_aft_ho_d = _col_arr(cpu_ho, "pred_aft")
            fus_aft_ho_d = _col_arr(fus_ho, "pred_aft")
            cpu_aft_ho_s = _col_arr(cpu_ho, "pred_aft_smiles_only")
            fus_aft_ho_s = _col_arr(fus_ho, "pred_aft_smiles_only")
            ho_cens_dual = _apply_simplex_blend(
                list(ch_d["weights"].keys()),
                {"champion_exact": ho_dual_final, "cpu_aft": cpu_aft_ho_d, "fusion_aft": fus_aft_ho_d, "deep": streams_dual_ho["deep"]},
                ch_d["weights"], task_type="regression", bounds=bounds, fallback=fr["y_cv_mean_dual"],
            )
            ho_cens_smi = _apply_simplex_blend(
                list(ch_s["weights"].keys()),
                {"champion_exact": ho_smi_final, "cpu_aft": cpu_aft_ho_s, "fusion_aft": fus_aft_ho_s, "deep": streams_smi_ho["deep"]},
                ch_s["weights"], task_type="regression", bounds=bounds, fallback=fr["y_cv_mean_smi"],
            )
            cens_doc = {
                "dual_modality": {
                    "mode": ch_d["mode"],
                    "weights": ch_d["weights"],
                    "cv_metrics": ch_d["cv_metrics"],
                    "holdout_metrics": regression_metrics(y_ho, ho_cens_dual, yl_ho, yu_ho, ct_ho),
                    "cpu_aft_holdout": regression_metrics(y_ho, cpu_aft_ho_d, yl_ho, yu_ho, ct_ho),
                    "fusion_aft_holdout": regression_metrics(y_ho, fus_aft_ho_d, yl_ho, yu_ho, ct_ho),
                },
                "smiles_only": {
                    "mode": ch_s["mode"],
                    "weights": ch_s["weights"],
                    "cv_metrics": ch_s["cv_metrics"],
                    "holdout_metrics": regression_metrics(
                        y_ho[smi_mask_ho], ho_cens_smi[smi_mask_ho], yl_ho[smi_mask_ho], yu_ho[smi_mask_ho], ct_ho[smi_mask_ho]
                    ),
                },
            }

        # Strict Zero-Leakage Leave-Case17-Out evaluation using EXACT SAME frozen weights & calibrators!
        cpu_c17 = _read_tsv(CPU_DIR / ep / "case17_predictions.tsv")
        fus_c17 = _read_tsv(FUSION_DIR / ep / "case17_predictions.tsv")
        dep_c17 = _read_tsv(DEEP_DIR / ep / "case17_predictions.tsv")
        assert len(cpu_c17) == len(fus_c17) == len(dep_c17) == 17
        for i in range(17):
            dn = case17_queries[i]["drug_name"]
            assert cpu_c17[i]["drug_name"] == fus_c17[i]["drug_name"] == dep_c17[i]["drug_name"] == dn

        c17_d_streams = {
            "cpu": _col_arr(cpu_c17, "pred_dual"),
            "fusion": _col_arr(fus_c17, "pred_dual"),
            "deep": _col_arr(dep_c17, "pred_dual"),
        }
        c17_s_streams = {
            "cpu": _col_arr(cpu_c17, "pred_smiles_only"),
            "fusion": _col_arr(fus_c17, "pred_smiles_only"),
            "deep": _col_arr(dep_c17, "pred_smiles_only"),
        }
        if ep in hl_c17_sp_exact:
            sp_d = hl_c17_sp_exact[ep]["canonical"]["dual_modality"]
            sp_s = hl_c17_sp_exact[ep]["canonical"]["smiles_only"]
            # deep = 0.5*mt + 0.5*sp -> mt = 2*deep - sp (exact linear inverse of train_gpu_deep._blend_case17)
            mt_d = 2.0 * c17_d_streams["deep"] - sp_d
            mt_s = 2.0 * c17_s_streams["deep"] - sp_s
            if task_type == "classification":
                mt_d = np.clip(mt_d, 1e-5, 1.0 - 1e-5)
                mt_s = np.clip(mt_s, 1e-5, 1.0 - 1e-5)
            c17_d_streams["deep_specialist"] = sp_d
            c17_d_streams["deep_multitask"] = mt_d
            c17_s_streams["deep_specialist"] = sp_s
            c17_s_streams["deep_multitask"] = mt_s

        c17_dual_raw = _apply_simplex_blend(
            list(d_fr["weights"].keys()), c17_d_streams, d_fr["weights"],
            task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_dual"],
        )
        c17_smi_raw = _apply_simplex_blend(
            list(s_fr["weights"].keys()), c17_s_streams, s_fr["weights"],
            task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_smi"],
        )
        if task_type == "classification":
            c17_dual_final = apply_frozen_calibrator(c17_dual_raw, d_fr["calibrator"])
            c17_smi_final = apply_frozen_calibrator(c17_smi_raw, s_fr["calibrator"])
            c17_iv = None
        else:
            c17_dual_final = c17_dual_raw
            c17_smi_final = c17_smi_raw
            c17_max_t = _col_arr(cpu_c17, "max_tanimoto")
            c17_iv = apply_frozen_conformal(c17_dual_final, d_fr["conformal"], max_tanimoto=c17_max_t)

        # Censored head on Case17
        c17_cens_dual = None
        if fr["censored_head_dual"] is not None:
            ch_d = fr["censored_head_dual"]
            c17_cens_dual = _apply_simplex_blend(
                list(ch_d["weights"].keys()),
                {
                    "champion_exact": c17_dual_final,
                    "cpu_aft": _col_arr(cpu_c17, "pred_aft"),
                    "fusion_aft": _col_arr(fus_c17, "pred_aft"),
                    "deep": c17_d_streams["deep"],
                },
                ch_d["weights"], task_type="regression", bounds=bounds, fallback=fr["y_cv_mean_dual"],
            )

        # Alternate clinical route on Case17 (SC for half_life_reg & F, SC_CL_over_F for CL_absolute)
        c17_alt_dual: Dict[str, np.ndarray] = {}
        c17_alt_smi: Dict[str, np.ndarray] = {}
        for alt_k in spec.get("alt_case17_conditions", {}):
            a_d = {
                "cpu": _col_arr(cpu_c17, f"pred_dual_{alt_k}") if _col_arr(cpu_c17, f"pred_dual_{alt_k}") is not None else c17_d_streams["cpu"],
                "fusion": _col_arr(fus_c17, f"pred_dual_{alt_k}") if _col_arr(fus_c17, f"pred_dual_{alt_k}") is not None else c17_d_streams["fusion"],
                "deep": _col_arr(dep_c17, f"pred_dual_{alt_k}") if _col_arr(dep_c17, f"pred_dual_{alt_k}") is not None else c17_d_streams["deep"],
            }
            a_s = {
                "cpu": _col_arr(cpu_c17, f"pred_smiles_only_{alt_k}") if _col_arr(cpu_c17, f"pred_smiles_only_{alt_k}") is not None else c17_s_streams["cpu"],
                "fusion": _col_arr(fus_c17, f"pred_smiles_only_{alt_k}") if _col_arr(fus_c17, f"pred_smiles_only_{alt_k}") is not None else c17_s_streams["fusion"],
                "deep": _col_arr(dep_c17, f"pred_smiles_only_{alt_k}") if _col_arr(dep_c17, f"pred_smiles_only_{alt_k}") is not None else c17_s_streams["deep"],
            }
            if ep == "half_life_reg" and alt_k == "SC" and ep in hl_c17_sp_exact:
                sp_d_sc = hl_c17_sp_exact[ep]["SC"]["dual_modality"]
                sp_s_sc = hl_c17_sp_exact[ep]["SC"]["smiles_only"]
                a_d["deep_specialist"] = sp_d_sc
                a_d["deep_multitask"] = 2.0 * a_d["deep"] - sp_d_sc
                a_s["deep_specialist"] = sp_s_sc
                a_s["deep_multitask"] = 2.0 * a_s["deep"] - sp_s_sc
            c17_alt_dual[alt_k] = _apply_simplex_blend(
                list(d_fr["weights"].keys()), a_d, d_fr["weights"],
                task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_dual"],
            )
            c17_alt_smi[alt_k] = _apply_simplex_blend(
                list(s_fr["weights"].keys()), a_s, s_fr["weights"],
                task_type=task_type, bounds=bounds, fallback=fr["y_cv_mean_smi"],
            )

        # Save auditable champion predictions TSVs
        ep_pred_dir = PRED_OUT_DIR / ep
        ep_pred_dir.mkdir(parents=True, exist_ok=True)

        def _fstr(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, (float, np.floating)):
                return "" if not np.isfinite(v) else f"{float(v):.6f}"
            return str(v)

        with (ep_pred_dir / "oof_champion.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["record_id", "compound_key", "fold_id", "y_value", "censor_type", "pred_dual", "pred_smiles_only"])
            for i, r in enumerate(fr["cv_ep_rows"]):
                w.writerow([
                    r["record_id"], r["compound_key"], r["fold_id"], r["y_value"], r["censor_type"],
                    _fstr(d_fr["champion_oof_final"][i]), _fstr(s_fr["champion_oof_final"][i]),
                ])

        with (ep_pred_dir / "holdout_champion.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            cols = ["record_id", "compound_key", "y_value", "censor_type", "pred_dual", "pred_smiles_only"]
            if ho_dual_iv is not None:
                cols.extend(["lo80", "hi80", "lo90", "hi90"])
            if ho_cens_dual is not None:
                cols.append("pred_censored_dual")
            w.writerow(cols)
            for i, r in enumerate(ho_ep_rows):
                row = [
                    r["record_id"], r["compound_key"], r["y_value"], r["censor_type"],
                    _fstr(ho_dual_final[i]), _fstr(ho_smi_final[i]),
                ]
                if ho_dual_iv is not None:
                    row.extend([_fstr(ho_dual_iv["lo80"][i]), _fstr(ho_dual_iv["hi80"][i]), _fstr(ho_dual_iv["lo90"][i]), _fstr(ho_dual_iv["hi90"][i])])
                if ho_cens_dual is not None:
                    row.append(_fstr(ho_cens_dual[i]))
                w.writerow(row)

        c17_recs: List[Dict[str, Any]] = []
        with (ep_pred_dir / "case17_champion.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            cols = ["drug_name", "compound_key", "clinical_route", "pred_dual", "pred_smiles_only"]
            if c17_cens_dual is not None:
                cols.append("pred_censored_dual")
            for alt_k in c17_alt_dual:
                cols.extend([f"pred_dual_{alt_k}", f"pred_smiles_only_{alt_k}"])
            w.writerow(cols)
            for i, q in enumerate(case17_queries):
                dn = q["drug_name"]
                rec: Dict[str, Any] = {
                    "drug_name": dn,
                    "compound_key": q["compound_key"],
                    "clinical_route": CASE17_CLINICAL_ROUTE.get(dn, "IV"),
                    "pred_dual": _safe_float(c17_dual_final[i]),
                    "pred_smiles_only": _safe_float(c17_smi_final[i]),
                }
                row = [dn, q["compound_key"], rec["clinical_route"], _fstr(c17_dual_final[i]), _fstr(c17_smi_final[i])]
                if c17_iv is not None:
                    rec["ci80"] = [_safe_float(c17_iv["lo80"][i]), _safe_float(c17_iv["hi80"][i])]
                    rec["ci90"] = [_safe_float(c17_iv["lo90"][i]), _safe_float(c17_iv["hi90"][i])]
                if c17_cens_dual is not None:
                    rec["pred_censored_dual"] = _safe_float(c17_cens_dual[i])
                    row.append(_fstr(c17_cens_dual[i]))
                for alt_k in c17_alt_dual:
                    rec[f"pred_dual_{alt_k}"] = _safe_float(c17_alt_dual[alt_k][i])
                    rec[f"pred_smiles_only_{alt_k}"] = _safe_float(c17_alt_smi[alt_k][i])
                    row.extend([_fstr(c17_alt_dual[alt_k][i]), _fstr(c17_alt_smi[alt_k][i])])
                if ep == "half_life_reg":
                    rec["mechanism_anchor"] = compute_halflife_mechanism_anchor(q["canonical_smiles"])
                w.writerow(row)
                c17_recs.append(rec)
        case17_by_ep[ep] = c17_recs

        # Load CPU/Fusion/Deep summary metadata for transparency
        cpu_m_doc = json.loads((CPU_DIR / ep / "metrics.json").read_text(encoding="utf-8"))
        fus_m_doc = json.loads((FUSION_DIR / ep / "metrics.json").read_text(encoding="utf-8"))

        # Assemble authoritative endpoint entry for v33_manifest.json
        endpoints_manifest[ep] = {
            "endpoint": ep,
            "display_en": disp["display_en"],
            "display_zh": disp["display_zh"],
            "tier": 1 if spec["tier"] == "tier1" else 2,
            "tier_label": spec["tier"],
            "domain": disp["domain"],
            "task_type": task_type,
            "size_tier": spec["size_tier"],
            "unit": spec["unit"],
            "linear_unit": disp["linear_unit"],
            "log10_space": disp["log10_space"],
            "bounds": list(bounds) if bounds is not None else None,
            "cond_cols": cpu_m_doc["cond_cols"],
            "canonical_conditions": spec["canonical_conditions"],
            "counts": ep_meta_counts[ep],
            "v32_reference": v32_snap["endpoints"].get(ep),
            "dual_modality": {
                "winning_family": d_fr["winning_family"],
                "is_super_ensemble": d_fr["is_super_ensemble"],
                "weights": d_fr["weights"],
                "best_single_family": d_fr["best_single_family"],
                "nested_super_delta_over_best_single": d_fr["nested_super_delta_over_best_single"],
                "cv_metrics": d_fr["champion_cv_metrics"],
                "cv_metrics_raw": d_fr["champion_cv_metrics_raw"],
                "holdout_metrics": champ_ho_dual_m,
                "holdout_metrics_raw": champ_ho_dual_raw_m,
                "family_cv_metrics": d_fr["single_family_oof_metrics"],
                "family_holdout_metrics": fam_ho_dual,
                "calibrator": d_fr["calibrator"],
                "conformal": d_fr["conformal"],
                "conformal_holdout_coverage": conf_ho_dual,
            },
            "smiles_only": {
                "winning_family": s_fr["winning_family"],
                "is_super_ensemble": s_fr["is_super_ensemble"],
                "weights": s_fr["weights"],
                "best_single_family": s_fr["best_single_family"],
                "nested_super_delta_over_best_single": s_fr["nested_super_delta_over_best_single"],
                "cv_metrics": s_fr["champion_cv_metrics"],
                "cv_metrics_raw": s_fr["champion_cv_metrics_raw"],
                "holdout_metrics": champ_ho_smi_m,
                "holdout_metrics_raw": champ_ho_smi_raw_m,
                "family_cv_metrics": s_fr["single_family_oof_metrics"],
                "family_holdout_metrics": fam_ho_smi,
                "calibrator": s_fr["calibrator"],
                "conformal": s_fr["conformal"],
                "conformal_holdout_coverage": conf_ho_smi,
            },
            "censored_head": cens_doc,
            "representation_ablation_holdout": {
                k: v.get("ho_metrics") for k, v in fus_m_doc.get("representation_ablation", {}).items()
            },
        }

        if task_type == "classification":
            print(
                f"  [HO ] {ep:<24} | {d_fr['winning_family']:<14} | HO ROC={champ_ho_dual_m['roc_auc']:.4f} PR={champ_ho_dual_m['pr_auc']:.4f} "
                f"Brier={champ_ho_dual_m['brier']:.4f} ECE={champ_ho_dual_m['ece']:.4f} | SMI ROC={champ_ho_smi_m['roc_auc']:.4f}",
                flush=True,
            )
        else:
            print(
                f"  [HO ] {ep:<24} | {d_fr['winning_family']:<14} | HO MAE={champ_ho_dual_m['mae']:.4f} Sp={champ_ho_dual_m['spearman']:.3f} "
                f"Cov80/90={100*conf_ho_dual['cov80']:.1f}%/{100*conf_ho_dual['cov90']:.1f}% | SMI MAE={champ_ho_smi_m['mae']:.4f}",
                flush=True,
            )

    # =========================================================================
    # BUILD CASE17 17-DRUG SUPER-ENSEMBLE SCORECARD (ZERO LEAKAGE)
    # =========================================================================
    print("\n[4/6] Assembling strict Leave-Case17-Out 17-FDA-Drug Scorecard...", flush=True)
    case17_scorecard_drugs: List[Dict[str, Any]] = []
    hl_true_log, hl_pred_ml_log, hl_pred_anch_log = [], [], []
    ppb_true, ppb_pred_dual, ppb_pred_cens = [], [], []

    for i, q in enumerate(case17_queries):
        dn = q["drug_name"]
        route = CASE17_CLINICAL_ROUTE.get(dn, "IV")
        hl_rec = case17_by_ep["half_life_reg"][i]
        ppb_rec = case17_by_ep["PPB"][i]
        hem_rec = case17_by_ep["hemolysis_cls"][i]
        hc50_rec = case17_by_ep["hemolysis_HC50"][i]
        tox_rec = case17_by_ep["overall_peptide_toxicity"][i]
        bbb_rec = case17_by_ep["BBB"][i]
        pampa_rec = case17_by_ep["PAMPA"][i]
        caco2_rec = case17_by_ep["Caco2"][i]
        logd_rec = case17_by_ep["LogD7.4"][i]
        cl_rec = case17_by_ep["CL_per_kg"][i]
        vd_rec = case17_by_ep["Vd_per_kg"][i]
        f_rec = case17_by_ep["F"][i]
        sol_rec = case17_by_ep["solubility_cls"][i]

        hl_iv_log = hl_rec["pred_dual"]
        hl_sc_log = hl_rec.get("pred_dual_SC", hl_iv_log)
        hl_rm_log = hl_sc_log if route == "SC" else hl_iv_log
        hl_iv_h = round(10.0 ** hl_iv_log, 3) if hl_iv_log is not None else None
        hl_sc_h = round(10.0 ** hl_sc_log, 3) if hl_sc_log is not None else None
        hl_rm_h = round(10.0 ** hl_rm_log, 3) if hl_rm_log is not None else None
        anch = hl_rec["mechanism_anchor"]
        hl_hybrid_h = anch["anchor_mid_h"] if anch["anchor_mid_h"] is not None else hl_rm_h

        fda_num = CASE17_FDA_NUMERIC.get(dn, {})
        if fda_num.get("half_life_h") is not None and hl_rm_log is not None:
            hl_true_log.append(math.log10(fda_num["half_life_h"]))
            hl_pred_ml_log.append(hl_rm_log)
            hl_pred_anch_log.append(math.log10(hl_hybrid_h))
        if fda_num.get("ppb") is not None and ppb_rec["pred_dual"] is not None:
            ppb_true.append(fda_num["ppb"])
            ppb_pred_dual.append(ppb_rec["pred_dual"])
            ppb_pred_cens.append(ppb_rec.get("pred_censored_dual", ppb_rec["pred_dual"]))

        v32_c17_row = v32_snap["case17_v32"].get(dn, {})
        fda_ref_row = v32_snap["case17_fda_refs"].get(dn, {})
        case17_scorecard_drugs.append({
            "drug_name": dn,
            "compound_key": q["compound_key"],
            "clinical_route": route,
            "fda_half_life_ref": fda_ref_row.get("half_life_ref", q.get("half_life_ref", "")),
            "fda_half_life_h": fda_num.get("half_life_h"),
            "fda_ppb_ref": fda_ref_row.get("ppb_ref", q.get("ppb_ref", "")),
            "fda_ppb": fda_num.get("ppb"),
            "v32_half_life": v32_c17_row.get("platform_half_life"),
            "v32_ppb": v32_c17_row.get("platform_ppb"),
            "v32_hemolysis_prob": _safe_float(v32_c17_row.get("hemolysis_prob")),
            "v33_half_life_iv_h": hl_iv_h,
            "v33_half_life_sc_h": hl_sc_h,
            "v33_half_life_route_matched_h": hl_rm_h,
            "v33_half_life_smiles_only_sc_h": round(10.0 ** hl_rec.get("pred_smiles_only_SC", hl_rec["pred_smiles_only"]), 3),
            "v33_half_life_mechanism_anchor": anch,
            "v33_half_life_hybrid_h": hl_hybrid_h,
            "v33_ppb_dual": ppb_rec["pred_dual"],
            "v33_ppb_smiles_only": ppb_rec["pred_smiles_only"],
            "v33_ppb_censored": ppb_rec.get("pred_censored_dual"),
            "v33_hemolysis_prob_cal": hem_rec["pred_dual"],
            "v33_hemolysis_hc50_uM": round(10.0 ** hc50_rec["pred_dual"], 1) if hc50_rec["pred_dual"] is not None else None,
            "v33_hemolysis_hc50_censored_uM": round(10.0 ** hc50_rec["pred_censored_dual"], 1) if hc50_rec.get("pred_censored_dual") is not None else None,
            "v33_toxicity_prob": tox_rec["pred_dual"],
            "v33_bbb_prob": bbb_rec["pred_dual"],
            "v33_pampa_log10": pampa_rec["pred_dual"],
            "v33_caco2_log10": caco2_rec["pred_dual"],
            "v33_logd74": logd_rec["pred_dual"],
            "v33_cl_per_kg": round(10.0 ** cl_rec["pred_dual"], 3) if cl_rec["pred_dual"] is not None else None,
            "v33_vd_per_kg": round(10.0 ** vd_rec["pred_dual"], 3) if vd_rec["pred_dual"] is not None else None,
            "v33_f_po": f_rec["pred_dual"],
            "v33_f_sc": f_rec.get("pred_dual_SC"),
            "v33_solubility_prob": sol_rec["pred_dual"],
        })

    def _c17_agg(yt: List[float], yp: List[float]) -> Dict[str, Any]:
        a, b = np.array(yt, dtype=np.float64), np.array(yp, dtype=np.float64)
        return {
            "n": len(a),
            "mae": round(float(mean_absolute_error(a, b)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(a, b))), 4),
            "r2": round(float(r2_score(a, b)), 4),
            "spearman": round(float(spearmanr(a, b).statistic), 4),
            "pearson": round(float(pearsonr(a, b).statistic), 4),
        }

    c17_hem_max = max(d["v33_hemolysis_prob_cal"] for d in case17_scorecard_drugs)
    c17_hem_gt05 = sum(1 for d in case17_scorecard_drugs if d["v33_hemolysis_prob_cal"] > 0.5)
    case17_summary = {
        "protocol": "Strict Leave-Case17-Out (case17_excluded==0, 0 overlapping SMILES/InChIKey/InChIKey-14/clusters across all 3 families)",
        "half_life_pure_ml_route_matched_log10": _c17_agg(hl_true_log, hl_pred_ml_log),
        "half_life_with_mechanism_anchor_log10": _c17_agg(hl_true_log, hl_pred_anch_log),
        "ppb_dual": _c17_agg(ppb_true, ppb_pred_dual),
        "ppb_censored_head": _c17_agg(ppb_true, ppb_pred_cens),
        "hemolysis_false_positive_check": {
            "max_prob_across_17_fda_drugs": round(c17_hem_max, 4),
            "n_drugs_gt_0_50": c17_hem_gt05,
            "v32_n_drugs_gt_0_60": 8,
        },
        "drugs": case17_scorecard_drugs,
    }
    print(
        f"  [Case17] Half-life pure ML (n=16): MAE={case17_summary['half_life_pure_ml_route_matched_log10']['mae']:.4f}, Sp={case17_summary['half_life_pure_ml_route_matched_log10']['spearman']:.4f} "
        f"| +Anchor: MAE={case17_summary['half_life_with_mechanism_anchor_log10']['mae']:.4f}, Sp={case17_summary['half_life_with_mechanism_anchor_log10']['spearman']:.4f} "
        f"| PPB (n=14): MAE={case17_summary['ppb_dual']['mae']:.4f}, Sp={case17_summary['ppb_dual']['spearman']:.4f} "
        f"| Max Hemol Prob={c17_hem_max:.4f}",
        flush=True,
    )

    # =========================================================================
    # BUILD AD + TOP-3 ANALOG INDEX & PRUNED SERVING MODELS
    # =========================================================================
    ad_npz_path, analog_gz_path = build_ad_and_analog_index(
        compounds_meta, key_to_idx, morgan_packed, smiles_valid, seq_valid
    )

    # Build pruned serving models
    pruning_input = {
        ep: {
            "dual_modality": endpoints_manifest[ep]["dual_modality"],
            "smiles_only": endpoints_manifest[ep]["smiles_only"],
            "censored_head_dual": oof_frozen[ep]["censored_head_dual"],
        }
        for ep in ALL_ENDPOINTS
    }
    serving_inventory = build_pruned_serving_models(pruning_input)

    # Aggregate representation ablation summary from Wave-2D-1 fusion_benchmark_summary.json
    fus_summary = json.loads((FUSION_DIR / "fusion_benchmark_summary.json").read_text(encoding="utf-8"))
    rep_wins = defaultdict(int)
    for ep in ALL_ENDPOINTS:
        ra = fus_summary["endpoints"][ep].get("representation_ablation", {})
        is_cls = ENDPOINT_SPECS[ep]["task_type"] == "classification"
        cands = []
        for rk in ("ncaa_smiles_repr", "ncaa_fusion_hf", "ncaa_protein_proj", "peptideclm2", "chemberta_mtr", "esmc_mean", "esmc_multiregion"):
            if rk not in ra or not ra[rk].get("ho_metrics"):
                continue
            hm = ra[rk]["ho_metrics"]
            sc = hm["roc_auc"] if is_cls else -(hm["mae"] if hm["mae"] is not None else 1e9)
            fam = "ncaa_repdistill_v2" if rk.startswith("ncaa_") else ("esmc_600m" if rk.startswith("esmc_") else rk)
            cands.append((sc, fam, rk))
        if cands:
            best_fam = max(cands, key=lambda x: x[0])[1]
            rep_wins[best_fam] += 1

    # Aggregate overall champion summary statistics
    cls_eps = [ep for ep in ALL_ENDPOINTS if ENDPOINT_SPECS[ep]["task_type"] == "classification"]
    reg_eps = [ep for ep in ALL_ENDPOINTS if ENDPOINT_SPECS[ep]["task_type"] == "regression"]

    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        return round(sum(clean) / len(clean), 5) if clean else None

    overall_summary = {
        "dual_modality_winning_family_counts": {
            fam: sum(1 for ep in ALL_ENDPOINTS if endpoints_manifest[ep]["dual_modality"]["winning_family"] == fam)
            for fam in sorted({endpoints_manifest[ep]["dual_modality"]["winning_family"] for ep in ALL_ENDPOINTS})
        },
        "smiles_only_winning_family_counts": {
            fam: sum(1 for ep in ALL_ENDPOINTS if endpoints_manifest[ep]["smiles_only"]["winning_family"] == fam)
            for fam in sorted({endpoints_manifest[ep]["smiles_only"]["winning_family"] for ep in ALL_ENDPOINTS})
        },
        "classification_6_holdout_dual_means": {
            "cpu_roc_auc": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["cpu"]["roc_auc"] for e in cls_eps]),
            "fusion_roc_auc": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["fusion"]["roc_auc"] for e in cls_eps]),
            "deep_roc_auc": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["deep"]["roc_auc"] for e in cls_eps]),
            "v33_champion_roc_auc": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["roc_auc"] for e in cls_eps]),
            "v33_champion_pr_auc": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["pr_auc"] for e in cls_eps]),
            "v33_champion_brier": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["brier"] for e in cls_eps]),
            "v33_champion_ece": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["ece"] for e in cls_eps]),
        },
        "regression_18_holdout_dual_means": {
            "cpu_mae": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["cpu"]["mae"] for e in reg_eps]),
            "fusion_mae": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["fusion"]["mae"] for e in reg_eps]),
            "deep_mae": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["deep"]["mae"] for e in reg_eps]),
            "v33_champion_mae": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["mae"] for e in reg_eps]),
            "cpu_spearman": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["cpu"]["spearman"] for e in reg_eps]),
            "fusion_spearman": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["fusion"]["spearman"] for e in reg_eps]),
            "deep_spearman": _mean([endpoints_manifest[e]["dual_modality"]["family_holdout_metrics"]["deep"]["spearman"] for e in reg_eps]),
            "v33_champion_spearman": _mean([endpoints_manifest[e]["dual_modality"]["holdout_metrics"]["spearman"] for e in reg_eps]),
            "v33_conformal_cov80": _mean([endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov80"] for e in reg_eps]),
            "v33_conformal_cov90": _mean([endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov90"] for e in reg_eps]),
            "v33_conformal_cov80_medium_large_10": _mean([
                endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov80"]
                for e in reg_eps if ENDPOINT_SPECS[e]["size_tier"] in ("medium", "large")
            ]),
            "v33_conformal_cov90_medium_large_10": _mean([
                endpoints_manifest[e]["dual_modality"]["conformal_holdout_coverage"]["cov90"]
                for e in reg_eps if ENDPOINT_SPECS[e]["size_tier"] in ("medium", "large")
            ]),
        },
    }

    # Compute SHA-256 checksums of key bundle artifacts
    print("\n[5/6] Computing SHA-256 checksums and writing authoritative v33_manifest.json...", flush=True)
    checksums = {
        "ad_analog_index.npz": sha256_file(ad_npz_path),
        "analog_lookup.json.gz": sha256_file(analog_gz_path),
        "v32_reference_snapshot.json": sha256_file(v32_snap_path),
        "serving_models": {
            p.name: sha256_file(p) for p in sorted(SERVING_DIR.glob("*.joblib"))
        },
    }

    manifest_doc: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v33_authoritative_manifest_v1",
        "version": "v33.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "selection_protocol": {
            "zero_holdout_peeking": True,
            "super_ensemble_adoption_margin": 0.0005,
            "weight_pruning_threshold": 0.02,
            "classification_blend_space": "logit_simplex",
            "regression_blend_space": "value_simplex",
            "ad_thresholds": {"in_domain": AD_IN_DOMAIN_THR, "borderline": AD_BORDERLINE_THR},
        },
        "dataset_summary": {
            "n_unique_compounds": n_compounds_v33,
            "n_unified_compounds_incl_case17": len(key_to_idx),
            "n_endpoints": len(ALL_ENDPOINTS),
            "n_tier1_endpoints": sum(1 for ep in ALL_ENDPOINTS if ENDPOINT_SPECS[ep]["tier"] == "tier1"),
            "n_tier2_endpoints": sum(1 for ep in ALL_ENDPOINTS if ENDPOINT_SPECS[ep]["tier"] == "tier2"),
            "n_classification_endpoints": len(cls_eps),
            "n_regression_endpoints": len(reg_eps),
            "n_total_rows": total_rows_all,
            "n_cv_rows": total_cv_all,
            "n_holdout_rows": total_ho_all,
            "n_censored_rows": total_censored_all,
            "n_case17_drugs": 17,
        },
        "overall_summary": overall_summary,
        "representation_ablation_summary": {
            "single_foundation_model_holdout_wins_out_of_24": dict(rep_wins),
            "foundation_models": [
                {"name": "ncaa_repdistill_v2", "dims": "768 + 2048 + 1280", "modality": "smiles_only"},
                {"name": "peptideclm2", "dims": "512", "modality": "smiles_only"},
                {"name": "chemberta_mtr", "dims": "384", "modality": "smiles_only"},
                {"name": "esmc_600m_multiregion", "dims": "4 x 1152 = 4608 (mean/max/nterm5/cterm5)", "modality": "sequence"},
            ],
        },
        "serving_inventory": serving_inventory,
        "case17_scorecard": case17_summary,
        "endpoints": endpoints_manifest,
        "sha256_checksums": checksums,
        "build_elapsed_sec": round(time.time() - t_all, 2),
    }

    manifest_path = BUNDLE_DIR / "v33_manifest.json"
    manifest_path.write_text(json.dumps(manifest_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[6/6] COMPLETE! Wrote {manifest_path} ({manifest_path.stat().st_size/1e6:.2f} MB) in {manifest_doc['build_elapsed_sec']}s", flush=True)
    print(json.dumps(overall_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
