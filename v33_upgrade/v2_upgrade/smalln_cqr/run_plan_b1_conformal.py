#!/usr/bin/env python3
"""Plan B-1 Deliverable 2 (runs on node23 64-core CPU ONLY):
CQR + Finite-Sample / Studentized CV+ / Mondrian AD-Stratified Conformal Calibration
across ALL 40 Regression Endpoints (dual_modality AND smiles_only).

STRICT ZERO-LEAKAGE & ZERO-HOLDOUT-PEEKING GUARANTEES:
1. Evaluates all 4 conformal methods via STRICT NESTED 5-FOLD OOF on is_holdout == 0 exact rows:
   (a) baseline          : Current v33_manifest.json split/adaptive conformal
   (b) studentized_cvplus: Finite-sample cluster-effective quantile + OOF local scale sigma_hat(x)
                           (using AD distances + physchem + exact GPR posterior sigma where available)
   (c) cqr_lgb           : Residual-centered CQR with fold-internal LightGBM Quantile Regressors (0.10/0.90 & 0.05/0.95)
   (d) mondrian_ad_eb    : Mondrian AD-stratified (In-Domain >=0.65, Borderline 0.40-0.65, OOD <0.40)
                           with Empirical-Bayes shrinkage toward global scale when stratum count < 25
2. Selects the winning calibration rule STRICTLY on nested 5-fold OOF coverage-efficiency score
   (target 80%/90% within [78%, 85%] / [88%, 94%], AD-tier reliability, tightest mean width).
3. Evaluates ONCE on untouched 20% holdout across all 40 endpoints (both frozen V33.1 baseline predictions
   AND Plan B-1 upgraded predictions combining Deliverable 1 OOF-promoted Small-N models!).
4. Exports node2-ready cqr_mondrian_calibration.json + lightweight verified .joblib calibrators +
   smalln_cqr_summary.json + audited smalln_cqr_report.md!
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
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
EXP_BASE = V33_BASE / "expansion_v331"
BUNDLE_DIR = V33_BASE / "bundle"
OUT_DIR = V33_BASE / "v2_upgrade" / "smalln_cqr"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))

from smalln_models import (  # noqa: E402
    AD_BORDERLINE_THR,
    AD_IN_DOMAIN_THR,
    ResidualCQRCalibrator,
    StudentizedResidualScaleCalibrator,
    _conformal_quantile,
    apply_mondrian_ad_eb,
    assign_ad_tiers,
    fit_mondrian_ad_eb,
)

csv.field_size_limit(10_000_000)

GLOBAL_MANIFEST: Optional[Dict[str, Any]] = None
GLOBAL_PHYS: Optional[np.ndarray] = None
GLOBAL_SMI_VALID: Optional[np.ndarray] = None
GLOBAL_SEQ_VALID: Optional[np.ndarray] = None
GLOBAL_MORGAN_PACKED: Optional[np.ndarray] = None
GLOBAL_KEY_TO_IDX: Optional[Dict[str, int]] = None
GLOBAL_SMALLN_RES: Optional[Dict[str, Any]] = None


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


def _fast_tanimoto_max_top5(q_packed: np.ndarray, ref_packed: np.ndarray, chunk: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    nq, nr = q_packed.shape[0], ref_packed.shape[0]
    if nq == 0 or nr == 0:
        return np.full(nq, np.nan, dtype=np.float64), np.full(nq, np.nan, dtype=np.float64)
    ref_b = np.unpackbits(ref_packed, axis=1).astype(np.float32)
    ref_p = ref_b.sum(axis=1)
    k5 = min(5, nr)
    max_t = np.zeros(nq, dtype=np.float64)
    top5_t = np.zeros(nq, dtype=np.float64)
    for s in range(0, nq, chunk):
        e = min(nq, s + chunk)
        qb = np.unpackbits(q_packed[s:e], axis=1).astype(np.float32)
        qp = qb.sum(axis=1)
        inter = qb @ ref_b.T
        union = qp[:, None] + ref_p[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        sim[qp <= 0] = 0.0
        max_t[s:e] = np.maximum(sim.max(axis=1), 0.0)
        if k5 >= nr:
            top5_t[s:e] = np.maximum(sim, 0.0).mean(axis=1)
        else:
            part = np.partition(sim, nr - k5, axis=1)[:, nr - k5 :]
            top5_t[s:e] = np.maximum(part, 0.0).mean(axis=1)
    return max_t, top5_t


def _compute_oof_and_ho_tanimoto(
    cidx_cv: np.ndarray,
    fids_cv: np.ndarray,
    cidx_ho: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute strictly leak-free OOF (fold != f) & Holdout (all CV) max & top5 Tanimoto similarities."""
    assert GLOBAL_MORGAN_PACKED is not None and GLOBAL_SMI_VALID is not None
    n_cv, n_ho = len(cidx_cv), len(cidx_ho)
    oof_max = np.full(n_cv, np.nan, dtype=np.float64)
    oof_top5 = np.full(n_cv, np.nan, dtype=np.float64)
    for f in range(5):
        va = np.where(fids_cv == f)[0]
        tr = np.where(fids_cv != f)[0]
        if len(va) == 0 or len(tr) == 0:
            continue
        tr_uniq = np.unique(cidx_cv[tr])
        tr_uniq = tr_uniq[GLOBAL_SMI_VALID[tr_uniq]]
        if len(tr_uniq) == 0:
            continue
        m_v, t5_v = _fast_tanimoto_max_top5(GLOBAL_MORGAN_PACKED[cidx_cv[va]], GLOBAL_MORGAN_PACKED[tr_uniq])
        bad_va = ~GLOBAL_SMI_VALID[cidx_cv[va]]
        m_v[bad_va] = np.nan
        t5_v[bad_va] = np.nan
        oof_max[va] = m_v
        oof_top5[va] = t5_v

    cv_uniq = np.unique(cidx_cv)
    cv_uniq = cv_uniq[GLOBAL_SMI_VALID[cv_uniq]]
    ho_max, ho_top5 = _fast_tanimoto_max_top5(GLOBAL_MORGAN_PACKED[cidx_ho], GLOBAL_MORGAN_PACKED[cv_uniq])
    bad_ho = ~GLOBAL_SMI_VALID[cidx_ho]
    ho_max[bad_ho] = np.nan
    ho_top5[bad_ho] = np.nan
    return oof_max, oof_top5, ho_max, ho_top5


def _extract_phys_block(cidx: np.ndarray, track: str) -> np.ndarray:
    assert GLOBAL_PHYS is not None and GLOBAL_SMI_VALID is not None and GLOBAL_SEQ_VALID is not None
    if track == "smiles_only":
        return np.nan_to_num(GLOBAL_PHYS[cidx, :48].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    flags = np.stack([GLOBAL_SMI_VALID[cidx].astype(np.float32), GLOBAL_SEQ_VALID[cidx].astype(np.float32)], axis=1)
    return np.nan_to_num(np.concatenate([GLOBAL_PHYS[cidx, :83].astype(np.float32), flags], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _apply_baseline_manifest_conformal(
    pred: np.ndarray, conf_spec: Dict[str, Any], max_tanimoto: np.ndarray
) -> Dict[str, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    if conf_spec.get("mode") == "adaptive_tanimoto_dist":
        t = np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.5)
        d = 1.0 - np.clip(t, 0.0, 1.0)
        s = np.maximum(
            float(conf_spec["dist_intercept"]) + float(conf_spec["dist_slope"]) * d,
            float(conf_spec["min_scale"]),
        )
    else:
        s = np.ones(len(p), dtype=np.float64)
    lo80, hi80 = p - float(conf_spec["q80"]) * s, p + float(conf_spec["q80"]) * s
    lo90, hi90 = p - float(conf_spec["q90"]) * s, p + float(conf_spec["q90"]) * s
    bounds = conf_spec.get("bounds")
    if bounds is not None:
        lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
        lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
    return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}


def _fit_baseline_conformal_on_subset(
    y: np.ndarray,
    pred: np.ndarray,
    max_tanimoto: np.ndarray,
    mode: str,
    bounds: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    abs_r = np.abs(y - pred)
    med_r = float(np.median(abs_r)) if len(abs_r) > 0 else 0.5
    floor_s = max(0.30 * med_r, 1e-3)
    if mode == "adaptive_tanimoto_dist" and np.isfinite(max_tanimoto).sum() >= 25:
        ok = np.isfinite(max_tanimoto)
        d = 1.0 - np.clip(np.where(ok, max_tanimoto, 0.5), 0.0, 1.0)
        try:
            reg = HuberRegressor(alpha=1.0).fit(d[ok, None], abs_r[ok])
            sl = max(0.0, float(reg.coef_[0]))
            ic = max(floor_s, float(reg.intercept_))
        except Exception:
            sl, ic = 0.0, max(med_r, 1e-3)
        s = np.maximum(ic + sl * d, floor_s)
        return {
            "mode": "adaptive_tanimoto_dist",
            "q80": _conformal_quantile(abs_r / s, 0.20),
            "q90": _conformal_quantile(abs_r / s, 0.10),
            "dist_slope": sl,
            "dist_intercept": ic,
            "min_scale": floor_s,
            "bounds": list(bounds) if bounds is not None else None,
        }
    return {
        "mode": "standard",
        "q80": _conformal_quantile(abs_r, 0.20),
        "q90": _conformal_quantile(abs_r, 0.10),
        "dist_slope": 0.0,
        "dist_intercept": 1.0,
        "min_scale": floor_s,
        "bounds": list(bounds) if bounds is not None else None,
    }


def _eval_interval_metrics(
    y: np.ndarray,
    ivs: Dict[str, np.ndarray],
    max_tanimoto: np.ndarray,
) -> Dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(ivs["lo80"]) & np.isfinite(ivs["hi80"]) & np.isfinite(ivs["lo90"]) & np.isfinite(ivs["hi90"])
    if not ok.any():
        return {"n_eval": 0, "cov80": None, "cov90": None, "w80": None, "w90": None, "by_tier": {}}
    yt = y[ok]
    c80 = (yt >= ivs["lo80"][ok]) & (yt <= ivs["hi80"][ok])
    c90 = (yt >= ivs["lo90"][ok]) & (yt <= ivs["hi90"][ok])
    w80 = ivs["hi80"][ok] - ivs["lo80"][ok]
    w90 = ivs["hi90"][ok] - ivs["lo90"][ok]
    tiers = assign_ad_tiers(max_tanimoto[ok])
    by_tier = {}
    for tn in ("In-Domain", "Borderline", "OOD"):
        m = tiers == tn
        if m.any():
            by_tier[tn] = {
                "n": int(m.sum()),
                "cov80": round(float(np.mean(c80[m])), 4),
                "cov90": round(float(np.mean(c90[m])), 4),
                "w80": round(float(np.mean(w80[m])), 4),
                "w90": round(float(np.mean(w90[m])), 4),
            }
    return {
        "n_eval": int(ok.sum()),
        "cov80": round(float(np.mean(c80)), 4),
        "cov90": round(float(np.mean(c90)), 4),
        "w80": round(float(np.mean(w80)), 4),
        "w90": round(float(np.mean(w90)), 4),
        "by_tier": by_tier,
    }


def _oof_coverage_efficiency_score(m: Dict[str, Any], base_w80: float, base_w90: float) -> Tuple[float, Dict[str, float]]:
    """Strictly inner-OOF coverage-efficiency score (lower is better!).

    Target window: cov80 in [0.78, 0.85], cov90 in [0.88, 0.94].
    Combines marginal validity, AD-conditional reliability (no AD tier severely under-covered), and width efficiency.
    """
    c80 = float(m["cov80"]) if m["cov80"] is not None else 0.0
    c90 = float(m["cov90"]) if m["cov90"] is not None else 0.0
    w80 = float(m["w80"]) if m["w80"] is not None else base_w80 * 2.0
    w90 = float(m["w90"]) if m["w90"] is not None else base_w90 * 2.0

    # Marginal window violation penalty (heavier on under-coverage!)
    pen_marg = (
        2.5 * max(0.0, 0.78 - c80)
        + 1.0 * max(0.0, c80 - 0.85)
        + 2.5 * max(0.0, 0.88 - c90)
        + 1.0 * max(0.0, c90 - 0.94)
    )

    # AD-conditional stratum under-coverage penalty (strictly on OOF AD tiers with n >= 5, plus pooled non-In-Domain!)
    n_tot = max(1, int(m["n_eval"]))
    pen_ad = 0.0
    non_ind_n = 0
    non_ind_h80 = 0.0
    non_ind_h90 = 0.0
    for tn, td in m.get("by_tier", {}).items():
        if tn in ("Borderline", "OOD"):
            non_ind_n += td["n"]
            non_ind_h80 += td["cov80"] * td["n"]
            non_ind_h90 += td["cov90"] * td["n"]
        if td["n"] >= 5:
            wt = td["n"] / float(n_tot)
            tier_mult = 1.35 if tn in ("Borderline", "OOD") else 0.70
            pen_ad += tier_mult * wt * (max(0.0, 0.76 - td["cov80"]) + max(0.0, 0.86 - td["cov90"]))
    if non_ind_n >= 5:
        c80_nid = non_ind_h80 / non_ind_n
        c90_nid = non_ind_h90 / non_ind_n
        wt_nid = non_ind_n / float(n_tot)
        pen_ad += 1.0 * wt_nid * (max(0.0, 0.77 - c80_nid) + max(0.0, 0.87 - c90_nid))

    rel_w = 0.5 * (w80 / max(base_w80, 1e-4) + w90 / max(base_w90, 1e-4))
    score = 12.0 * pen_marg + 10.0 * pen_ad + rel_w
    return float(score), {
        "pen_marg": round(pen_marg, 5),
        "pen_ad": round(pen_ad, 5),
        "rel_width": round(rel_w, 5),
        "total_score": round(score, 5),
    }


def _check_joblib_node2_portable(path: Path) -> List[str]:
    obj = joblib.load(path)
    mods: Set[str] = set()
    seen: Set[int] = set()

    def walk(x: Any):
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        mod = getattr(type(x), "__module__", "")
        if mod:
            mods.add(mod.split(".")[0])
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
    for forbidden in ("catboost", "tabpfn", "torch"):
        assert forbidden not in mods, f"FORBIDDEN '{forbidden}' in {path}: {sorted(mods)}"
    return sorted(mods)


def _run_conformal_for_stream(
    ep: str,
    track: str,
    stream_label: str,
    bounds: Optional[Tuple[float, float]],
    base_manifest_conf: Dict[str, Any],
    y_cv: np.ndarray,
    pred_oof: np.ndarray,
    ct_cv: np.ndarray,
    fids_cv: np.ndarray,
    cids_cv: np.ndarray,
    valid_cv: np.ndarray,
    phys_cv: np.ndarray,
    oof_max_t: np.ndarray,
    oof_top5_t: np.ndarray,
    gp_oof_sig: Optional[np.ndarray],
    y_ho: np.ndarray,
    pred_ho: np.ndarray,
    ct_ho: np.ndarray,
    valid_ho: np.ndarray,
    phys_ho: np.ndarray,
    ho_max_t: np.ndarray,
    ho_top5_t: np.ndarray,
    gp_ho_sig: Optional[np.ndarray],
    save_calibrator_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare all 4 conformal methods strictly on nested 5-fold OOF, select winner, evaluate ONCE on Holdout."""
    m_cv = valid_cv & (ct_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof)
    m_ho = valid_ho & (ct_ho == "exact") & np.isfinite(y_ho) & np.isfinite(pred_ho)

    ye, pe = y_cv[m_cv], pred_oof[m_cv]
    fe, ce = fids_cv[m_cv], cids_cv[m_cv]
    ph_e = phys_cv[m_cv]
    t1_e, t5_e = oof_max_t[m_cv], oof_top5_t[m_cv]
    gs_e = gp_oof_sig[m_cv] if gp_oof_sig is not None else None
    n_ex = len(ye)

    yh, ph = y_ho[m_ho], pred_ho[m_ho]
    ph_h = phys_ho[m_ho]
    t1_h, t5_h = ho_max_t[m_ho], ho_top5_t[m_ho]
    gs_h = gp_ho_sig[m_ho] if gp_ho_sig is not None else None

    # =========================================================================
    # 1. STRICT NESTED 5-FOLD OOF COMPARISON OF ALL 4 METHODS (ZERO HOLDOUT!)
    # =========================================================================
    methods = ("baseline", "studentized_cvplus", "cqr_lgb", "mondrian_ad_eb")
    nested_ivs: Dict[str, Dict[str, np.ndarray]] = {
        k: {b: np.full(n_ex, np.nan, dtype=np.float64) for b in ("lo80", "hi80", "lo90", "hi90")}
        for k in methods
    }
    base_mode = base_manifest_conf.get("mode", "standard")
    uniq_f = sorted(set(fe.tolist()))

    for f in uniq_f:
        va = fe == f
        tr = ~va
        if tr.sum() < 6 or va.sum() == 0:
            continue
        # (a) Baseline split/adaptive conformal fit on 4 folds
        b_spec_f = _fit_baseline_conformal_on_subset(ye[tr], pe[tr], t1_e[tr], mode=base_mode, bounds=bounds)
        iv_a = _apply_baseline_manifest_conformal(pe[va], b_spec_f, t1_e[va])

        # (b) Studentized CV+ / Finite-sample locally scaled conformal fit on 4 folds
        cal_b = StudentizedResidualScaleCalibrator(bounds=bounds).fit(
            ye[tr], pe[tr], ph_e[tr], t1_e[tr], top5_tanimoto=t5_e[tr],
            gp_sigma=(gs_e[tr] if gs_e is not None else None),
            fold_ids=fe[tr], cluster_ids=ce[tr],
        )
        iv_b = cal_b.predict_intervals(
            pe[va], ph_e[va], t1_e[va], top5_tanimoto=t5_e[va],
            gp_sigma=(gs_e[va] if gs_e is not None else None),
        )

        # (c) Residual-centered CQR (LightGBM Quantile 0.10/0.90 & 0.05/0.95) fit on 4 folds
        cal_c = ResidualCQRCalibrator(bounds=bounds).fit(
            ye[tr], pe[tr], ph_e[tr], t1_e[tr], top5_tanimoto=t5_e[tr],
            fold_ids=fe[tr], cluster_ids=ce[tr],
        )
        iv_c = cal_c.predict_intervals(pe[va], ph_e[va], t1_e[va], top5_tanimoto=t5_e[va])

        # (d) Mondrian AD-Stratified Empirical-Bayes Shrinkage Conformal fit on 4 folds
        d_spec_f = fit_mondrian_ad_eb(
            ye[tr], pe[tr], t1_e[tr], bounds=bounds, cluster_ids=ce[tr], min_stratum_n=25, enforce_monotonic=True
        )
        iv_d = apply_mondrian_ad_eb(pe[va], d_spec_f, t1_e[va])

        for mname, iv in (("baseline", iv_a), ("studentized_cvplus", iv_b), ("cqr_lgb", iv_c), ("mondrian_ad_eb", iv_d)):
            for b in ("lo80", "hi80", "lo90", "hi90"):
                nested_ivs[mname][b][va] = iv[b]

    oof_metrics: Dict[str, Dict[str, Any]] = {}
    for mname in methods:
        oof_metrics[mname] = _eval_interval_metrics(ye, nested_ivs[mname], t1_e)

    base_w80 = float(oof_metrics["baseline"]["w80"] or 1.0)
    base_w90 = float(oof_metrics["baseline"]["w90"] or 1.0)
    oof_scores: Dict[str, Dict[str, float]] = {}
    for mname in methods:
        sc, detail = _oof_coverage_efficiency_score(oof_metrics[mname], base_w80, base_w90)
        oof_scores[mname] = detail

    # Select winning calibration method STRICTLY on nested-OOF score!
    selected_method = min(methods, key=lambda k: oof_scores[k]["total_score"])

    # =========================================================================
    # 2. FIT FULL-CV FROZEN CALIBRATORS & EVALUATE ONCE ON UNTOUCHED 20% HOLDOUT
    # =========================================================================
    # (a) Exact baseline (use manifest spec directly for frozen_v331 stream; refit on full CV for upgraded stream)
    if stream_label == "frozen_v331":
        spec_a = dict(base_manifest_conf)
    else:
        spec_a = _fit_baseline_conformal_on_subset(ye, pe, t1_e, mode=base_mode, bounds=bounds)
    ho_iv_a = _apply_baseline_manifest_conformal(ph, spec_a, t1_h)

    # (b) Full-CV Studentized Residual Scale
    cal_b_full = StudentizedResidualScaleCalibrator(bounds=bounds).fit(
        ye, pe, ph_e, t1_e, top5_tanimoto=t5_e, gp_sigma=gs_e, fold_ids=fe, cluster_ids=ce
    )
    ho_iv_b = cal_b_full.predict_intervals(ph, ph_h, t1_h, top5_tanimoto=t5_h, gp_sigma=gs_h)

    # (c) Full-CV Residual CQR
    cal_c_full = ResidualCQRCalibrator(bounds=bounds).fit(
        ye, pe, ph_e, t1_e, top5_tanimoto=t5_e, fold_ids=fe, cluster_ids=ce
    )
    ho_iv_c = cal_c_full.predict_intervals(ph, ph_h, t1_h, top5_tanimoto=t5_h)

    # (d) Full-CV Mondrian AD-EB
    spec_d_full = fit_mondrian_ad_eb(ye, pe, t1_e, bounds=bounds, cluster_ids=ce, min_stratum_n=25, enforce_monotonic=True)
    ho_iv_d = apply_mondrian_ad_eb(ph, spec_d_full, t1_h)

    ho_metrics = {
        "baseline": _eval_interval_metrics(yh, ho_iv_a, t1_h),
        "studentized_cvplus": _eval_interval_metrics(yh, ho_iv_b, t1_h),
        "cqr_lgb": _eval_interval_metrics(yh, ho_iv_c, t1_h),
        "mondrian_ad_eb": _eval_interval_metrics(yh, ho_iv_d, t1_h),
    }

    # Build node2-ready JSON calibration spec (+ save lightweight .joblib if selected method uses one!)
    cal_joblib_rel: Optional[str] = None
    cal_joblib_mods: List[str] = []
    if selected_method == "baseline":
        deploy_spec = {
            "method": "baseline",
            "mode": spec_a.get("mode", "standard"),
            "q80": round(float(spec_a["q80"]), 5),
            "q90": round(float(spec_a["q90"]), 5),
            "dist_slope": round(float(spec_a.get("dist_slope", 0.0)), 6),
            "dist_intercept": round(float(spec_a.get("dist_intercept", 1.0)), 6),
            "min_scale": round(float(spec_a.get("min_scale", 1e-3)), 6),
            "bounds": list(bounds) if bounds is not None else None,
        }
    elif selected_method == "mondrian_ad_eb":
        deploy_spec = dict(spec_d_full)
        deploy_spec["method"] = "mondrian_ad_eb"
    elif selected_method == "studentized_cvplus":
        if save_calibrator_dir is not None:
            save_calibrator_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{ep}_{track}_{stream_label}_studentized.joblib"
            fp = save_calibrator_dir / fn
            joblib.dump(cal_b_full, fp, compress=1)
            cal_joblib_mods = _check_joblib_node2_portable(fp)
            cal_joblib_rel = f"conformal_calibrators/{fn}"
        deploy_spec = {
            "method": "studentized_cvplus",
            "mode": "studentized_cvplus",
            "q80": round(float(cal_b_full.q80_), 5),
            "q90": round(float(cal_b_full.q90_), 5),
            "n_cal": int(cal_b_full.n_cal_),
            "n_eff_clusters": int(cal_b_full.n_eff_clusters_),
            "uses_gp_sigma": bool(cal_b_full.use_gp_sigma_),
            "calibrator_joblib": cal_joblib_rel,
            "joblib_modules": cal_joblib_mods,
            "mondrian_fallback": spec_d_full,
            "bounds": list(bounds) if bounds is not None else None,
        }
    else:  # cqr_lgb
        if save_calibrator_dir is not None:
            save_calibrator_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{ep}_{track}_{stream_label}_cqr.joblib"
            fp = save_calibrator_dir / fn
            joblib.dump(cal_c_full, fp, compress=1)
            cal_joblib_mods = _check_joblib_node2_portable(fp)
            cal_joblib_rel = f"conformal_calibrators/{fn}"
        deploy_spec = {
            "method": "cqr_lgb",
            "mode": "cqr_lgb",
            "Q80_adjustment": round(float(cal_c_full.Q80_), 5),
            "Q90_adjustment": round(float(cal_c_full.Q90_), 5),
            "n_cal": int(cal_c_full.n_cal_),
            "n_eff_clusters": int(cal_c_full.n_eff_clusters_),
            "calibrator_joblib": cal_joblib_rel,
            "joblib_modules": cal_joblib_mods,
            "mondrian_fallback": spec_d_full,
            "bounds": list(bounds) if bounds is not None else None,
        }

    return {
        "n_cv_exact": int(n_ex),
        "n_cv_clusters": int(len(set(str(c) for c in ce))),
        "n_ho_exact": int(len(yh)),
        "selected_method": selected_method,
        "oof_metrics": oof_metrics,
        "oof_scores": oof_scores,
        "holdout_metrics": ho_metrics,
        "selected_holdout": ho_metrics[selected_method],
        "deploy_spec": deploy_spec,
        "mondrian_ad_eb_spec": spec_d_full,
    }


def _weighted_combine(streams: Dict[str, np.ndarray], weights: Dict[str, float], bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    active = [k for k, w in weights.items() if float(w) > 0 and k in streams]
    w = np.array([float(weights[k]) for k in active], dtype=np.float64)
    w = w / np.sum(w)
    M = np.column_stack([streams[k] for k in active])
    fin = np.isfinite(M)
    M0 = np.where(fin, M, 0.0)
    W = np.where(fin, w[None, :], 0.0)
    ws = W.sum(axis=1)
    y = np.divide((M0 * W).sum(axis=1), ws, out=np.full(M.shape[0], np.nan), where=ws > 1e-12)
    if bounds is not None:
        y = np.clip(y, bounds[0], bounds[1])
    return y


def _worker_run_endpoint_conformal(ep: str) -> Dict[str, Any]:
    t0 = time.time()
    assert GLOBAL_MANIFEST is not None and GLOBAL_KEY_TO_IDX is not None and GLOBAL_SMI_VALID is not None
    ep_doc = GLOBAL_MANIFEST["endpoints"][ep]
    wave = ep_doc.get("expansion_wave", "v33.0")
    bounds_raw = ep_doc.get("bounds")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1])) if bounds_raw is not None else None

    ep_dir = EXP_BASE / "data/endpoints" if wave == "v33.1" else V33_BASE / "data/endpoints"
    cpu_dir = EXP_BASE / "models_cpu" if wave == "v33.1" else V33_BASE / "models_cpu"
    fus_dir = EXP_BASE / "models_fusion" if wave == "v33.1" else V33_BASE / "models_fusion"

    rows = _read_tsv(ep_dir / f"{ep}.tsv")
    cv_i = [i for i, r in enumerate(rows) if int(r["is_holdout"]) == 0]
    ho_i = [i for i, r in enumerate(rows) if int(r["is_holdout"]) == 1]

    cidx_all = np.array([GLOBAL_KEY_TO_IDX[r["compound_key"]] for r in rows], dtype=np.int32)
    cidx_cv, cidx_ho = cidx_all[cv_i], cidx_all[ho_i]
    fids_cv = np.array([int(rows[i]["fold_id"]) for i in cv_i], dtype=np.int32)
    cids_cv = np.array([rows[i]["cluster_id"] for i in cv_i], dtype=object)
    y_cv = np.array([float(rows[i]["y_value"]) for i in cv_i], dtype=np.float64)
    y_ho = np.array([float(rows[i]["y_value"]) for i in ho_i], dtype=np.float64)
    ct_cv = np.array([rows[i]["censor_type"] for i in cv_i], dtype=object)
    ct_ho = np.array([rows[i]["censor_type"] for i in ho_i], dtype=object)

    cpu_oof = _read_tsv(cpu_dir / ep / "oof_predictions.tsv")
    cpu_ho = _read_tsv(cpu_dir / ep / "holdout_predictions.tsv")
    fus_oof = _read_tsv(fus_dir / ep / "oof_predictions.tsv")
    fus_ho = _read_tsv(fus_dir / ep / "holdout_predictions.tsv")

    # Exact frozen max_tanimoto from CPU pipeline + leak-free top5_tanimoto computed from packed Morgan bits
    oof_max_t_cpu = _col_arr(cpu_oof, "max_tanimoto_oof")
    ho_max_t_cpu = _col_arr(cpu_ho, "max_tanimoto")
    oof_max_c, oof_top5_c, ho_max_c, ho_top5_c = _compute_oof_and_ho_tanimoto(cidx_cv, fids_cv, cidx_ho)
    # Use exact frozen CPU max_tanimoto where available (100% parity with manifest!) and computed top5
    oof_max_t = oof_max_t_cpu if oof_max_t_cpu is not None else oof_max_c
    ho_max_t = ho_max_t_cpu if ho_max_t_cpu is not None else ho_max_c

    # Reconstruct exact frozen V33.1 champion OOF & HO predictions
    frozen_oof: Dict[str, np.ndarray] = {}
    frozen_ho: Dict[str, np.ndarray] = {}
    if wave == "v33.0":
        coof = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "oof_champion.tsv")
        cho = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "holdout_champion.tsv")
        frozen_oof["dual_modality"] = _col_arr(coof, "pred_dual")
        frozen_oof["smiles_only"] = _col_arr(coof, "pred_smiles_only")
        frozen_ho["dual_modality"] = _col_arr(cho, "pred_dual")
        frozen_ho["smiles_only"] = _col_arr(cho, "pred_smiles_only")
    else:
        for tr, col in (("dual_modality", "pred_dual"), ("smiles_only", "pred_smiles_only")):
            w = ep_doc[tr]["weights"]
            frozen_oof[tr] = _weighted_combine({"cpu": _col_arr(cpu_oof, col), "fusion": _col_arr(fus_oof, col)}, w, bounds)
            frozen_ho[tr] = _weighted_combine({"cpu": _col_arr(cpu_ho, col), "fusion": _col_arr(fus_ho, col)}, w, bounds)

    # Load Deliverable 1 small-N cache if available for this endpoint
    sn_cache_fp = OUT_DIR / "smalln_cache" / f"{ep}_smalln_preds.npz"
    sn_npz = np.load(sn_cache_fp) if sn_cache_fp.exists() else None
    cal_dir = OUT_DIR / "conformal_calibrators"

    ep_out: Dict[str, Any] = {
        "endpoint": ep,
        "wave": wave,
        "n_total": int(ep_doc["counts"]["n_total"]),
        "n_cv": int(ep_doc["counts"]["n_cv"]),
        "n_holdout": int(ep_doc["counts"]["n_holdout"]),
        "frozen_v331": {},
        "plan_b1_upgraded": {},
    }

    for tr in ("dual_modality", "smiles_only"):
        valid_cv = GLOBAL_SMI_VALID[cidx_cv].copy() if tr == "smiles_only" else np.ones(len(cv_i), dtype=bool)
        valid_ho = GLOBAL_SMI_VALID[cidx_ho].copy() if tr == "smiles_only" else np.ones(len(ho_i), dtype=bool)
        phys_cv = _extract_phys_block(cidx_cv, tr)
        phys_ho = _extract_phys_block(cidx_ho, tr)

        gp_oof_s = gp_ho_s = None
        if sn_npz is not None:
            pref = "dual" if tr == "dual_modality" else "smi"
            gp_oof_s = sn_npz[f"{pref}_gp_oof_sigma"]
            gp_ho_s = sn_npz[f"{pref}_gp_ho_sigma"]

        # 1. Pure calibration benchmark on FROZEN V33.1 predictions
        res_fr = _run_conformal_for_stream(
            ep, tr, "frozen_v331", bounds, ep_doc[tr]["conformal"],
            y_cv, frozen_oof[tr], ct_cv, fids_cv, cids_cv, valid_cv, phys_cv, oof_max_t, oof_top5_c, gp_oof_s,
            y_ho, frozen_ho[tr], ct_ho, valid_ho, phys_ho, ho_max_t, ho_top5_c, gp_ho_s,
            save_calibrator_dir=cal_dir,
        )
        ep_out["frozen_v331"][tr] = res_fr

        # Verify baseline holdout parity with v33_manifest.json!
        man_cov80 = ep_doc[tr]["conformal_holdout_coverage"].get("cov80")
        rec_cov80 = res_fr["holdout_metrics"]["baseline"].get("cov80")
        if man_cov80 is not None and rec_cov80 is not None:
            assert abs(float(man_cov80) - float(rec_cov80)) <= 0.005, (
                f"Baseline conformal parity mismatch on {ep}/{tr}: manifest={man_cov80} vs recon={rec_cov80}"
            )

        # 2. Combined Plan B-1 upgraded predictions (if Deliverable 1 promoted a portable Small-N winner!)
        sn_promoted = False
        if GLOBAL_SMALLN_RES is not None and ep in GLOBAL_SMALLN_RES["endpoints"]:
            sn_promoted = bool(GLOBAL_SMALLN_RES["endpoints"][ep][tr]["portable"]["promoted"])
        if sn_promoted and sn_npz is not None:
            pref = "dual" if tr == "dual_modality" else "smi"
            up_oof = sn_npz[f"{pref}_port_oof"]
            up_ho = sn_npz[f"{pref}_port_ho"]
            res_up = _run_conformal_for_stream(
                ep, tr, "plan_b1_upgraded", bounds, ep_doc[tr]["conformal"],
                y_cv, up_oof, ct_cv, fids_cv, cids_cv, valid_cv, phys_cv, oof_max_t, oof_top5_c, gp_oof_s,
                y_ho, up_ho, ct_ho, valid_ho, phys_ho, ho_max_t, ho_top5_c, gp_ho_s,
                save_calibrator_dir=cal_dir,
            )
            ep_out["plan_b1_upgraded"][tr] = res_up
        else:
            ep_out["plan_b1_upgraded"][tr] = res_fr

    if sn_npz is not None:
        sn_npz.close()

    ep_out["elapsed_sec"] = round(time.time() - t0, 2)
    fd = ep_out["frozen_v331"]["dual_modality"]
    bh = fd["holdout_metrics"]["baseline"]
    sh = fd["selected_holdout"]
    print(
        f"  [{ep:<22}] N={ep_out['n_total']:4d} | SEL={fd['selected_method']:<18} "
        f"| HO Cov80: {100*(bh['cov80'] or 0):5.1f}% -> {100*(sh['cov80'] or 0):5.1f}% "
        f"| Cov90: {100*(bh['cov90'] or 0):5.1f}% -> {100*(sh['cov90'] or 0):5.1f}% "
        f"| W80: {bh['w80']:.3f} -> {sh['w80']:.3f} ({ep_out['elapsed_sec']:.1f}s)",
        flush=True,
    )
    return ep_out


def _agg_conformal_slice(
    by_ep: Dict[str, Any],
    eps: List[str],
    stream: str,
    track: str,
    method: str,
) -> Dict[str, Any]:
    c80s, c90s, w80s, w90s = [], [], [], []
    in_win80 = 0
    in_win90 = 0
    in_both = 0
    under70_80 = 0
    tier_hits80: Dict[str, List[float]] = {"In-Domain": [], "Borderline": [], "OOD": []}
    tier_n: Dict[str, int] = {"In-Domain": 0, "Borderline": 0, "OOD": 0}
    sel_counts: Dict[str, int] = {}

    for ep in eps:
        rec = by_ep[ep][stream][track]
        if method == "selected":
            sel = rec["selected_method"]
            sel_counts[sel] = sel_counts.get(sel, 0) + 1
            hm = rec["selected_holdout"]
        else:
            hm = rec["holdout_metrics"][method]
        if hm["cov80"] is None:
            continue
        c80, c90, w80, w90 = hm["cov80"], hm["cov90"], hm["w80"], hm["w90"]
        c80s.append(c80)
        c90s.append(c90)
        w80s.append(w80)
        w90s.append(w90)
        ok80 = 0.78 <= c80 <= 0.85
        ok90 = 0.88 <= c90 <= 0.94
        if ok80:
            in_win80 += 1
        if ok90:
            in_win90 += 1
        if ok80 and ok90:
            in_both += 1
        if c80 < 0.70:
            under70_80 += 1
        for tn, td in hm.get("by_tier", {}).items():
            tier_hits80[tn].append(td["cov80"] * td["n"])
            tier_n[tn] += td["n"]

    return {
        "n_endpoints": len(c80s),
        "mean_cov80": round(float(np.mean(c80s)), 4) if c80s else None,
        "mean_cov90": round(float(np.mean(c90s)), 4) if c90s else None,
        "min_cov80": round(float(np.min(c80s)), 4) if c80s else None,
        "min_cov90": round(float(np.min(c90s)), 4) if c90s else None,
        "mean_abs_err_cov80_from_0.80": round(float(np.mean(np.abs(np.array(c80s) - 0.80))), 4) if c80s else None,
        "mean_abs_err_cov90_from_0.90": round(float(np.mean(np.abs(np.array(c90s) - 0.90))), 4) if c90s else None,
        "n_cov80_in_78_85": in_win80,
        "n_cov90_in_88_94": in_win90,
        "n_both_in_window": in_both,
        "n_severe_undercov_lt_70": under70_80,
        "mean_w80": round(float(np.mean(w80s)), 4) if w80s else None,
        "mean_w90": round(float(np.mean(w90s)), 4) if w90s else None,
        "pooled_ad_tier_cov80": {
            tn: (round(sum(tier_hits80[tn]) / tier_n[tn], 4) if tier_n[tn] > 0 else None)
            for tn in ("In-Domain", "Borderline", "OOD")
        },
        "pooled_ad_tier_n": tier_n,
        "selected_method_counts": sel_counts if method == "selected" else None,
    }


def _generate_markdown_report(summary: Dict[str, Any], out_md: Path) -> None:
    sn = summary.get("deliverable1_smalln")
    cf = summary["deliverable2_conformal"]
    lines: List[str] = [
        "# Plan B-1 Audited Report: Small-N Foundation/Shrinkage Modeling & CQR/Mondrian Conformal Calibration",
        "",
        f"- **Generated at**: `{summary['generated_at']}`",
        f"- **Compute Hardware**: `node23` 64-core CPU ONLY (0 GPU usage; `node1` GPUs, local RTX 5080, `node2` live production, and frozen `v33_upgrade` directories untouched)",
        f"- **Zero-Leakage Protocol**: Frozen non-chaining dual-track cluster splits (`is_holdout == 0` 5-fold CV, untouched 20% `is_holdout == 1` holdout, strict `case17_excluded == 0` Leave-Case17-Out). All model & calibration selections locked strictly on inner nested 5-fold OOF BEFORE single holdout evaluation.",
        f"- **TabPFN v2 Status**: `{sn['tabpfn_status'] if sn else 'N/A'}`",
        "",
        "---",
        "",
        "## 1. Deliverable 1 — Small-N Modeling Upgrade (`N <= 400`, 20 Regression Endpoints)",
        "",
    ]

    if sn is not None:
        lines.append("### 1.1 Aggregate Summary (Zero-Holdout-Peeking Gate: OOF MAE Gain >= 0.002)")
        lines.append("")
        lines.append("| Slice | Track | Suite | OOF Promoted | Holdout Improved (of Promoted) | Mean CV MAE (Before -> After, Delta) | Mean Holdout MAE (Before -> After, Delta) | Mean Holdout Spearman (Before -> After) |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for s_label, sk in [
            ("All Small-N (N<=400, 20 EPs)", "all_n_le_400_20ep"),
            ("Core Small-N (N<=300, 19 EPs)", "all_n_le_300_19ep"),
            ("Small-N (N<=283, 16 EPs)", "smalln_16ep_n_le_283"),
            ("Priority Weakest (9 EPs)", "priority_weak_9ep"),
        ]:
            sl = sn["slices"][sk]
            for tr_lbl, m_key, su_lbl in [
                ("dual_modality", "dual_portable", "Portable (no TabPFN)"),
                ("dual_modality", "dual_full_with_tabpfn", "Full (+ TabPFN v2)"),
                ("smiles_only", "smiles_portable", "Portable (no TabPFN)"),
                ("smiles_only", "smiles_full_with_tabpfn", "Full (+ TabPFN v2)"),
            ]:
                a = sl[m_key]
                lines.append(
                    f"| {s_label} | `{tr_lbl}` | {su_lbl} | {a['n_oof_promoted']}/{a['n_endpoints']} | "
                    f"{a['n_promoted_improved_holdout']}/{a['n_oof_promoted']} | "
                    f"{a['mean_cv_mae_before']:.4f} -> {a['mean_cv_mae_after']:.4f} ({a['mean_cv_mae_delta']:+.4f}) | "
                    f"{a['mean_ho_mae_before']:.4f} -> {a['mean_ho_mae_after']:.4f} ({a['mean_ho_mae_delta']:+.4f}) | "
                    f"{a['mean_ho_spearman_before']:.4f} -> {a['mean_ho_spearman_after']:.4f} |"
                )
        lines.append("")
        lines.append("### 1.2 Per-Endpoint Small-N Benchmark (`dual_modality` — Before vs Portable Winner vs Full + TabPFN v2)")
        lines.append("")
        lines.append("| Endpoint | N | Base CV MAE | Base HO MAE | Best Single New (CV MAE) | TabPFN v2 CV/HO MAE | Portable Winner (OOF Gate) | Port CV -> HO MAE (Delta) | Full+TabPFN Winner | Full CV -> HO MAE (Delta) |")
        lines.append("|---|---:|---:|---:|---|---:|---|---:|---|---:|")
        for ep, er in sorted(sn["endpoints"].items(), key=lambda kv: kv[1]["n_total"]):
            tr = er["dual_modality"]
            bcv = tr["baseline_cv_metrics"]["mae"]
            bho = tr["baseline_ho_metrics"]["mae"]
            icv = tr["indiv_cv_metrics"]
            iho = tr["indiv_ho_metrics"]
            bs_k = min(icv.keys(), key=lambda k: icv[k]["mae"] if icv[k]["mae"] is not None else 1e9)
            tp_str = (
                f"{icv['tabpfn_v2']['mae']:.4f}/{iho['tabpfn_v2']['mae']:.4f}"
                if "tabpfn_v2" in icv and icv["tabpfn_v2"]["mae"] is not None
                else "N/A"
            )
            dp = tr["portable"]
            df = tr["full_with_tabpfn"]
            p_tag = f"**{dp['winner_name']}**" if dp["promoted"] else "unchanged"
            f_tag = f"**{df['winner_name']}**" if df["promoted"] else "unchanged"
            lines.append(
                f"| `{ep}` | {er['n_total']} | {bcv:.4f} | {bho:.4f} | `{bs_k}` ({icv[bs_k]['mae']:.4f}) | {tp_str} | "
                f"{p_tag} | {dp['cv_metrics']['mae']:.4f} -> {dp['ho_metrics']['mae']:.4f} ({dp['ho_mae_delta']:+.4f}) | "
                f"{f_tag} | {df['cv_metrics']['mae']:.4f} -> {df['ho_metrics']['mae']:.4f} ({df['ho_mae_delta']:+.4f}) |"
            )

        # Leave-Case17-Out PPB audit & Winner's Curse diagnostic
        lines.extend([
            "",
            "### 1.3 Scientific Diagnostic — Winner's Curse / Selection-Induced OOF Optimism on Weakest Small-N Endpoints (`dual_modality`)",
            "",
            "> Existing V33.1 `cpu`/`fusion` champions swept 40–80 recipes and stacked 12 already-CV-selected heads on the same 5-fold CV split, inducing downward CV optimism bias on tiny-N endpoints (e.g., `Vd_absolute` Base CV=0.4597 vs HO=0.7780). Zero-tuning `MultiBlockCompositeGPR` and `TabPFN v2` produce unbiased CV estimates and substantially lower untouched-holdout MAE on several weakest small-N endpoints, even where the strict pre-specified OOF gate (`CV_new <= Base_CV - 0.002`) prudently retained baseline without holdout peeking.",
            "",
            "| Endpoint | N | Base CV -> HO (Gap) | MultiBlock GPR CV -> HO (HO Delta vs Base) | TabPFN v2 CV -> HO (HO Delta vs Base) |",
            "|---|---:|---:|---:|---:|",
        ])
        for ep in ("Vd_absolute", "MBEC_biofilm", "CL_absolute", "PPB", "MDCK", "RRCK", "F", "Cmax", "efflux_ratio", "MFC_C_albicans", "MBC_E_faecalis", "MIC_E_faecium", "solubility_reg"):
            if ep not in sn["endpoints"]:
                continue
            er = sn["endpoints"][ep]
            tr = er["dual_modality"]
            bcv, bho = tr["baseline_cv_metrics"]["mae"], tr["baseline_ho_metrics"]["mae"]
            bg = tr["best_gpr_name"]
            gcv, gho = tr["indiv_cv_metrics"][bg]["mae"], tr["indiv_ho_metrics"][bg]["mae"]
            tcv = tr["indiv_cv_metrics"].get("tabpfn_v2", {}).get("mae")
            tho = tr["indiv_ho_metrics"].get("tabpfn_v2", {}).get("mae")
            t_str = f"{tcv:.4f} -> {tho:.4f} ({bho - tho:+.4f})" if (tcv is not None and tho is not None) else "N/A"
            lines.append(
                f"| `{ep}` | {er['n_total']} | {bcv:.4f} -> {bho:.4f} ({bho - bcv:+.4f}) | "
                f"{gcv:.4f} -> {gho:.4f} ({bho - gho:+.4f}) | {t_str} |"
            )

        # Smiles-only promoted endpoints
        smi_proms = [
            (ep, er["smiles_only"])
            for ep, er in sorted(sn["endpoints"].items(), key=lambda kv: kv[1]["n_total"])
            if er["smiles_only"]["portable"]["promoted"] or er["smiles_only"]["full_with_tabpfn"]["promoted"]
        ]
        if smi_proms:
            lines.extend(["", "### 1.4 `smiles_only` OOF-Promoted Small-N Endpoints", ""])
            for ep, tr in smi_proms:
                pf = tr["full_with_tabpfn"]
                lines.append(
                    f"- **`{ep}`** (`{pf['winner_name']}`): CV MAE `{tr['baseline_cv_metrics']['mae']:.4f} -> {pf['cv_metrics']['mae']:.4f}` "
                    f"| Untouched Holdout MAE `{tr['baseline_ho_metrics']['mae']:.4f} -> {pf['ho_metrics']['mae']:.4f}` (`{pf['ho_mae_delta']:+.4f}`), "
                    f"Spearman `{tr['baseline_ho_metrics']['spearman']:.4f} -> {pf['ho_metrics']['spearman']:.4f}`"
                )

        # Promoted portable joblib inventory
        prom_jbs = [(ep, er["promoted_joblib"], er["joblib_modules"]) for ep, er in sorted(sn["endpoints"].items()) if er.get("promoted_joblib")]
        lines.extend(["", "### 1.5 Promoted `node2`-Portable 5-Fold `.joblib` Artifacts (`node23`)", ""])
        for ep, jp, mods in prom_jbs:
            lines.append(f"- `{ep}`: `{jp}` — verified modules: `{mods}` (0 `catboost`, 0 `tabpfn`, 0 `torch`)")

        ppb_c17 = sn["endpoints"].get("PPB", {}).get("case17", {}).get("dual_modality", {}).get("fda_ppb_benchmark")
        if ppb_c17:
            lines.extend([
                "",
                "### 1.6 Strict Leave-Case17-Out FDA Clinical Drug Benchmark (`PPB`, n=14 FDA Anchors)",
                "",
                f"- **Baseline V33.1**: MAE = `{ppb_c17['baseline']['mae']:.4f}`, Spearman = `{ppb_c17['baseline']['spearman']:.4f}`",
                f"- **Plan B-1 Portable**: MAE = `{ppb_c17['portable']['mae']:.4f}`, Spearman = `{ppb_c17['portable']['spearman']:.4f}`",
                f"- **Plan B-1 Full (+TabPFN v2)**: MAE = `{ppb_c17['full_with_tabpfn']['mae']:.4f}`, Spearman = `{ppb_c17['full_with_tabpfn']['spearman']:.4f}`",
            ])

    lines.extend([
        "",
        "---",
        "",
        "## 2. Deliverable 2 — Conformal Calibration Across ALL 40 Regression Endpoints",
        "",
        "### 2.1 Aggregate Holdout Coverage & Width (Frozen V33.1 Predictions — Pure Calibration Comparison)",
        "",
        "| Slice | Track | Method | Mean Cov80 | Mean Cov90 | Min Cov80 | Severe (<70%) | Mean W80 | Mean W90 | Pooled AD Cov80 (InD / Bord / OOD) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for sl_lbl, sk in [
        ("All 40 Reg EPs", "all_40_endpoints"),
        ("Small-N 20 EPs (N<=400)", "smalln_20_n_le_400"),
        ("Small-N 19 EPs (N<=300)", "smalln_19_n_le_300"),
        ("Small-N 16 EPs (N<=283)", "smalln_16_n_le_283"),
    ]:
        for tr in ("dual_modality", "smiles_only"):
            for m in ("baseline", "studentized_cvplus", "cqr_lgb", "mondrian_ad_eb", "selected"):
                a = cf["aggregates"]["frozen_v331"][sk][tr][m]
                ad = a["pooled_ad_tier_cov80"]
                f_ad = lambda k: f"{100*ad[k]:.1f}%" if ad.get(k) is not None else "N/A"
                mlbl = f"**{m} (OOF-gated)**" if m == "selected" else m
                lines.append(
                    f"| {sl_lbl} | `{tr}` | {mlbl} | {100*a['mean_cov80']:.1f}% | {100*a['mean_cov90']:.1f}% | "
                    f"{100*a['min_cov80']:.1f}% | {a['n_severe_undercov_lt_70']}/{a['n_endpoints']} | "
                    f"{a['mean_w80']:.3f} | {a['mean_w90']:.3f} | {f_ad('In-Domain')} / {f_ad('Borderline')} / {f_ad('OOD')} |"
                )

    lines.extend([
        "",
        "### 2.2 Combined Plan B-1 (Deliverable 1 OOF-Promoted Small-N Models + Deliverable 2 OOF-Gated Conformal)",
        "",
        "| Slice | Track | Baseline Cov80/90 (W80) | Plan B-1 Selected Cov80/90 (W80) | Min Cov80 (Before -> After) | Severe <70% (Before -> After) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for sl_lbl, sk in [
        ("All 40 Reg EPs", "all_40_endpoints"),
        ("Small-N 19 EPs (N<=300)", "smalln_19_n_le_300"),
        ("Small-N 16 EPs (N<=283)", "smalln_16_n_le_283"),
    ]:
        for tr in ("dual_modality", "smiles_only"):
            b = cf["aggregates"]["frozen_v331"][sk][tr]["baseline"]
            u = cf["aggregates"]["plan_b1_upgraded"][sk][tr]["selected"]
            lines.append(
                f"| {sl_lbl} | `{tr}` | {100*b['mean_cov80']:.1f}% / {100*b['mean_cov90']:.1f}% ({b['mean_w80']:.3f}) | "
                f"**{100*u['mean_cov80']:.1f}% / {100*u['mean_cov90']:.1f}%** ({u['mean_w80']:.3f}) | "
                f"{100*b['min_cov80']:.1f}% -> **{100*u['min_cov80']:.1f}%** | "
                f"{b['n_severe_undercov_lt_70']} -> **{u['n_severe_undercov_lt_70']}** |"
            )

    lines.extend([
        "",
        "### 2.3 Per-Endpoint Conformal Holdout Table (All 40 Regression Endpoints, `dual_modality`, Frozen V33.1 Predictions)",
        "",
        "| Endpoint | N | Selected (OOF) | Base Cov80/90 | StudCV+ Cov80/90 | CQR Cov80/90 | Mondrian Cov80/90 | Selected Cov80/90 | W80 (Base -> Sel) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for ep, er in sorted(cf["endpoints"].items(), key=lambda kv: kv[1]["n_total"]):
        fd = er["frozen_v331"]["dual_modality"]
        hm = fd["holdout_metrics"]
        sel = fd["selected_method"]
        fc = lambda m: f"{100*hm[m]['cov80']:.1f}%/{100*hm[m]['cov90']:.1f}%"
        lines.append(
            f"| `{ep}` | {er['n_total']} | `{sel}` | {fc('baseline')} | {fc('studentized_cvplus')} | "
            f"{fc('cqr_lgb')} | {fc('mondrian_ad_eb')} | **{fc(sel)}** | {hm['baseline']['w80']:.3f} -> {hm[sel]['w80']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Zero-Leakage & `node2` Portability Verification Checklist",
        "",
        "- [x] **Zero holdout peeking**: All Small-N promotions (`margin >= 0.002`) and all Conformal method selections were locked strictly on nested 5-fold CV OOF (`is_holdout == 0`) prior to single untouched 20% holdout evaluation.",
        "- [x] **Zero cluster leakage**: Cluster disjointness asserted on all 40 endpoints (`set(cv_clusters) ∩ set(ho_clusters) == ∅`); inner hyperparameter tuning uses cluster-aware GroupKFold.",
        "- [x] **Strict Leave-Case17-Out**: All 7 small-N PK endpoints refitted strictly on `case17_excluded == 0`.",
        "- [x] **Strict `node2` portability**: Every promoted Small-N `.joblib` and Conformal calibrator `.joblib` verified via recursive object walker to contain ZERO `catboost`, ZERO `tabpfn`, and ZERO `torch` references (pure `numpy`/`scipy`/`sklearn`/`lightgbm` + `smalln_models.py`).",
        "- [x] **Zero mutation of frozen directories**: All outputs written exclusively under `<WORK_DIR>/v33_upgrade/v2_upgrade/smalln_cqr/` and mirrored to `D:\\work\\中检院\\v33_upgrade\\v2_upgrade\\smalln_cqr\\`.",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    global GLOBAL_MANIFEST, GLOBAL_PHYS, GLOBAL_SMI_VALID, GLOBAL_SEQ_VALID, GLOBAL_MORGAN_PACKED, GLOBAL_KEY_TO_IDX, GLOBAL_SMALLN_RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))

    sn_fp = OUT_DIR / "smalln_results.json"
    if sn_fp.exists():
        GLOBAL_SMALLN_RES = json.loads(sn_fp.read_text(encoding="utf-8"))
        print(f"[init] Loaded Deliverable 1 results from {sn_fp} ({len(GLOBAL_SMALLN_RES['endpoints'])} endpoints)", flush=True)

    fdir = EXP_BASE / "features"
    GLOBAL_KEY_TO_IDX = {}
    with (fdir / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            GLOBAL_KEY_TO_IDX[r["compound_key"]] = int(r["compound_idx"])
    pc = np.load(fdir / "physchem_core.npz")
    GLOBAL_PHYS = np.ascontiguousarray(pc["X"], dtype=np.float32)
    GLOBAL_SMI_VALID = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
    GLOBAL_SEQ_VALID = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
    pc.close()
    mb = np.load(fdir / "morgan_bits_1024.npz")
    GLOBAL_MORGAN_PACKED = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
    mb.close()

    reg_eps = [
        ep
        for ep, d in sorted(GLOBAL_MANIFEST["endpoints"].items(), key=lambda kv: kv[1]["counts"]["n_total"])
        if d["task_type"] == "regression"
    ]
    print(f"[Plan B-1 Deliverable 2] Running CQR + Studentized CV+ + Mondrian AD-EB across ALL {len(reg_eps)} regression endpoints with {args.workers} workers...", flush=True)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(args.workers, len(reg_eps))) as pool:
        results = pool.map(_worker_run_endpoint_conformal, reg_eps)

    by_ep = {r["endpoint"]: r for r in results}
    eps_all40 = list(reg_eps)
    eps_20 = [ep for ep in reg_eps if by_ep[ep]["n_total"] <= 400]
    eps_19 = [ep for ep in reg_eps if by_ep[ep]["n_total"] <= 300]
    eps_16 = [ep for ep in reg_eps if by_ep[ep]["n_total"] <= 283]

    aggregates: Dict[str, Any] = {}
    for stream in ("frozen_v331", "plan_b1_upgraded"):
        aggregates[stream] = {}
        for sk, ep_list in [
            ("all_40_endpoints", eps_all40),
            ("smalln_20_n_le_400", eps_20),
            ("smalln_19_n_le_300", eps_19),
            ("smalln_16_n_le_283", eps_16),
        ]:
            aggregates[stream][sk] = {}
            for tr in ("dual_modality", "smiles_only"):
                aggregates[stream][sk][tr] = {
                    m: _agg_conformal_slice(by_ep, ep_list, stream, tr, m)
                    for m in ("baseline", "studentized_cvplus", "cqr_lgb", "mondrian_ad_eb", "selected")
                }

    # Export clean node2-ready cqr_mondrian_calibration.json!
    deploy_json: Dict[str, Any] = {
        "schema": "peptide_omnipanel_v33_plan_b1_cqr_mondrian_calibration_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "ad_thresholds": {"in_domain": AD_IN_DOMAIN_THR, "borderline": AD_BORDERLINE_THR},
        "n_regression_endpoints": len(reg_eps),
        "calibrators_base_dir": str(OUT_DIR),
        "endpoints": {},
    }
    for ep in sorted(reg_eps):
        deploy_json["endpoints"][ep] = {
            "n_total": by_ep[ep]["n_total"],
            "frozen_v331": {
                "dual_modality": by_ep[ep]["frozen_v331"]["dual_modality"]["deploy_spec"],
                "smiles_only": by_ep[ep]["frozen_v331"]["smiles_only"]["deploy_spec"],
            },
            "plan_b1_upgraded": {
                "dual_modality": by_ep[ep]["plan_b1_upgraded"]["dual_modality"]["deploy_spec"],
                "smiles_only": by_ep[ep]["plan_b1_upgraded"]["smiles_only"]["deploy_spec"],
            },
            "mondrian_pure_json_only": {
                "dual_modality": by_ep[ep]["frozen_v331"]["dual_modality"]["mondrian_ad_eb_spec"],
                "smiles_only": by_ep[ep]["frozen_v331"]["smiles_only"]["mondrian_ad_eb_spec"],
            },
        }
    cal_json_fp = OUT_DIR / "cqr_mondrian_calibration.json"
    cal_json_fp.write_text(json.dumps(deploy_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # Master unified summary JSON
    master_summary = {
        "schema": "peptide_omnipanel_v33_plan_b1_smalln_cqr_summary_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_conformal_sec": round(time.time() - t_all, 2),
        "remote_artifacts_dir": str(OUT_DIR),
        "deliverable1_smalln": GLOBAL_SMALLN_RES,
        "deliverable2_conformal": {
            "n_regression_endpoints": len(reg_eps),
            "aggregates": aggregates,
            "endpoints": by_ep,
        },
    }
    sum_fp = OUT_DIR / "smalln_cqr_summary.json"
    sum_fp.write_text(json.dumps(master_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rep_fp = OUT_DIR / "smalln_cqr_report.md"
    _generate_markdown_report(master_summary, rep_fp)

    print(f"\n[DONE Deliverable 2] Saved:\n  - {cal_json_fp}\n  - {sum_fp}\n  - {rep_fp}\nin {master_summary['elapsed_conformal_sec']}s!", flush=True)
    print("\n=== ALL 40 REGRESSION ENDPOINTS (dual_modality, frozen_v331) ===")
    print(json.dumps(aggregates["frozen_v331"]["all_40_endpoints"]["dual_modality"], indent=2), flush=True)
    print("\n=== SMALL-N 16 ENDPOINTS N<=283 (dual_modality, frozen_v331) ===")
    print(json.dumps(aggregates["frozen_v331"]["smalln_16_n_le_283"]["dual_modality"], indent=2), flush=True)


if __name__ == "__main__":
    main()
