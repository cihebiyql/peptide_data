#!/usr/bin/env python3
"""Evaluation metrics, censored bound concordance, out-of-fold probability calibration
(Isotonic & Platt), CV+/locally-adaptive conformal prediction intervals, and AD stratification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """10-bin Expected Calibration Error."""
    y_t = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = len(y_t)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y_t[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    censor_type: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute exact-row metrics AND censored bound concordance / violation metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(yt) & np.isfinite(yp)

    if censor_type is None:
        exact_mask = finite
        cens_mask = np.zeros_like(finite, dtype=bool)
    else:
        ct = np.asarray(censor_type)
        exact_mask = finite & (ct == "exact")
        cens_mask = finite & (ct != "exact")

    res: Dict[str, Any] = {
        "n_total": int(finite.sum()),
        "n_exact": int(exact_mask.sum()),
        "n_censored": int(cens_mask.sum()),
    }

    if exact_mask.sum() >= 2:
        ye = yt[exact_mask]
        pe = yp[exact_mask]
        mae = float(mean_absolute_error(ye, pe))
        rmse = float(np.sqrt(mean_squared_error(ye, pe)))
        r2 = float(r2_score(ye, pe)) if np.std(ye) > 1e-12 else float("nan")
        sp = float(spearmanr(ye, pe).statistic) if (np.std(ye) > 1e-12 and np.std(pe) > 1e-12) else 0.0
        pr = float(pearsonr(ye, pe).statistic) if (np.std(ye) > 1e-12 and np.std(pe) > 1e-12) else 0.0
        res.update({
            "mae": round(mae, 5),
            "rmse": round(rmse, 5),
            "r2": round(r2, 5) if np.isfinite(r2) else None,
            "spearman": round(sp, 5) if np.isfinite(sp) else 0.0,
            "pearson": round(pr, 5) if np.isfinite(pr) else 0.0,
        })
    elif exact_mask.sum() == 1:
        err = float(abs(yt[exact_mask][0] - yp[exact_mask][0]))
        res.update({"mae": round(err, 5), "rmse": round(err, 5), "r2": None, "spearman": None, "pearson": None})
    else:
        res.update({"mae": None, "rmse": None, "r2": None, "spearman": None, "pearson": None})

    # Censored metrics
    if cens_mask.any() and y_lower is not None and y_upper is not None and censor_type is not None:
        yl = np.asarray(y_lower, dtype=np.float64)
        yu = np.asarray(y_upper, dtype=np.float64)
        ct = np.asarray(censor_type)
        concordant = 0
        violations: List[float] = []
        tol = 1e-9
        for i in np.where(cens_mask)[0]:
            c = ct[i]
            p = yp[i]
            if c == "right":
                ok = p >= yl[i] - tol
                viol = max(0.0, yl[i] - p)
            elif c == "left":
                ok = p <= yu[i] + tol
                viol = max(0.0, p - yu[i])
            elif c == "interval":
                ok = (p >= yl[i] - tol) and (p <= yu[i] + tol)
                viol = max(0.0, yl[i] - p, p - yu[i])
            else:
                ok = True
                viol = 0.0
            if ok:
                concordant += 1
            violations.append(float(viol))
        n_c = int(cens_mask.sum())
        res["censored_concordance"] = round(concordant / n_c, 5)
        res["censored_violation_mae"] = round(float(np.mean(violations)), 5)

        # Unified hinge MAE across ALL rows (exact abs error + censored bound violation)
        all_errs = []
        if exact_mask.any():
            all_errs.extend(np.abs(yt[exact_mask] - yp[exact_mask]).tolist())
        all_errs.extend(violations)
        res["all_row_hinge_mae"] = round(float(np.mean(all_errs)), 5)

        # Generalized Harrell C-index over comparable pairs (sample if huge)
        res["censored_c_index"] = round(
            compute_censored_c_index(yt[finite], yp[finite], yl[finite], yu[finite], ct[finite]), 5
        )
    else:
        res["censored_concordance"] = None
        res["censored_violation_mae"] = None
        res["all_row_hinge_mae"] = res.get("mae")
        res["censored_c_index"] = None

    return res


def compute_censored_c_index(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    censor_type: np.ndarray,
    max_n: int = 2500,
) -> float:
    """Generalized concordance index respecting exact, right (>), left (<), and interval bounds."""
    n = len(y_val)
    if n < 2:
        return float("nan")
    if n > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_n, replace=False)
        y_val, y_pred, y_lo, y_hi, censor_type = y_val[idx], y_pred[idx], y_lo[idx], y_hi[idx], censor_type[idx]
        n = max_n

    # Effective guaranteed lower/upper intervals:
    # exact: [y, y]; right: [y_lo, +inf]; left: [-inf, y_hi]; interval: [y_lo, y_hi]
    eff_lo = y_val.copy()
    eff_hi = y_val.copy()
    is_r = censor_type == "right"
    is_l = censor_type == "left"
    is_i = censor_type == "interval"
    eff_lo[is_r] = y_lo[is_r]
    eff_hi[is_r] = np.inf
    eff_lo[is_l] = -np.inf
    eff_hi[is_l] = y_hi[is_l]
    eff_lo[is_i] = y_lo[is_i]
    eff_hi[is_i] = y_hi[is_i]

    # Pair (i, j) is strictly ordered y_i < y_j iff eff_hi[i] < eff_lo[j]
    # Vectorized in blocks of 256 to keep memory tiny
    concord = 0.0
    total = 0.0
    blk = 256
    for s in range(0, n, blk):
        e = min(n, s + blk)
        strictly_less = eff_hi[s:e, None] < eff_lo[None, :]  # (b, n)
        cnt = int(strictly_less.sum())
        if cnt == 0:
            continue
        p_diff = y_pred[None, :] - y_pred[s:e, None]
        concord += float(((p_diff > 1e-12) & strictly_less).sum())
        concord += 0.5 * float(((np.abs(p_diff) <= 1e-12) & strictly_less).sum())
        total += float(cnt)

    return float(concord / total) if total > 0 else float("nan")


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.int32)
    p = np.asarray(y_prob, dtype=np.float64)
    finite = np.isfinite(p)
    yt = yt[finite]
    p = np.clip(p[finite], 1e-6, 1.0 - 1e-6)
    res: Dict[str, Any] = {
        "n_total": int(len(yt)),
        "n_pos": int((yt == 1).sum()),
        "n_neg": int((yt == 0).sum()),
    }
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        res.update({"roc_auc": None, "pr_auc": None, "mcc": None, "brier": None, "ece": None, "log_loss": None})
        return res
    y_hat = (p >= threshold).astype(np.int32)
    res.update({
        "roc_auc": round(float(roc_auc_score(yt, p)), 5),
        "pr_auc": round(float(average_precision_score(yt, p)), 5),
        "mcc": round(float(matthews_corrcoef(yt, y_hat)), 5),
        "brier": round(float(brier_score_loss(yt, p)), 5),
        "ece": round(float(compute_ece(yt, p, n_bins=10)), 5),
        "log_loss": round(float(log_loss(yt, p)), 5),
    })
    return res


class ProbabilityCalibrator:
    """Out-of-fold Isotonic vs Platt probability calibrator selected strictly on nested CV Brier."""

    def __init__(self):
        self.method_: str = "none"
        self.calibrator_: Any = None
        self.cv_comparison_: Dict[str, Any] = {}

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(pc / (1.0 - pc))[:, None]

    def fit(self, p_oof: np.ndarray, y_cv: np.ndarray, fold_ids: np.ndarray) -> "ProbabilityCalibrator":
        valid = np.isfinite(p_oof)
        p = np.clip(p_oof[valid], 1e-6, 1.0 - 1e-6)
        y = np.asarray(y_cv[valid], dtype=np.int32)
        fids = np.asarray(fold_ids[valid], dtype=np.int32)
        uniq_folds = sorted(set(fids.tolist()))

        # Nested OOF evaluation of none, platt, isotonic
        nested_preds = {
            "none": p.copy(),
            "platt": np.zeros_like(p),
            "isotonic": np.zeros_like(p),
        }
        for f in uniq_folds:
            va = fids == f
            tr = ~va
            if len(np.unique(y[tr])) < 2:
                nested_preds["platt"][va] = p[va]
                nested_preds["isotonic"][va] = p[va]
                continue
            lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
            lr.fit(self._logit(p[tr]), y[tr])
            nested_preds["platt"][va] = lr.predict_proba(self._logit(p[va]))[:, 1]

            iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
            iso.fit(p[tr], y[tr])
            nested_preds["isotonic"][va] = iso.predict(p[va])

        comp: Dict[str, Any] = {}
        best_method = "none"
        best_brier = float("inf")
        for m in ("none", "platt", "isotonic"):
            brier = float(brier_score_loss(y, nested_preds[m]))
            ece = float(compute_ece(y, nested_preds[m], n_bins=10))
            ll = float(log_loss(y, np.clip(nested_preds[m], 1e-6, 1.0 - 1e-6)))
            comp[m] = {"cv_brier": round(brier, 5), "cv_ece": round(ece, 5), "cv_logloss": round(ll, 5)}
            # Require at least tiny improvement over 'none' to adopt calibrator
            tol = 0.0 if m == "none" else 1e-5
            if brier + tol < best_brier:
                best_brier = brier
                best_method = m

        self.method_ = best_method
        self.cv_comparison_ = comp

        # Fit final winning calibrator on full CV OOF
        if self.method_ == "platt":
            lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
            lr.fit(self._logit(p), y)
            self.calibrator_ = lr
        elif self.method_ == "isotonic":
            iso = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
            iso.fit(p, y)
            self.calibrator_ = iso
        else:
            self.calibrator_ = None
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p_raw, dtype=np.float64)
        out = np.full_like(p_arr, np.nan)
        ok = np.isfinite(p_arr)
        if not ok.any():
            return out
        pc = np.clip(p_arr[ok], 1e-6, 1.0 - 1e-6)
        if self.method_ == "platt" and self.calibrator_ is not None:
            out[ok] = self.calibrator_.predict_proba(self._logit(pc))[:, 1]
        elif self.method_ == "isotonic" and self.calibrator_ is not None:
            out[ok] = self.calibrator_.predict(pc)
        else:
            out[ok] = pc
        return np.clip(out, 1e-5, 1.0 - 1e-5)


class ConformalIntervalCalibrator:
    """CV+ / Split Conformal prediction intervals (80% & 90%) with optional normalized scaling.

    Supports:
      - 'standard': constant half-width q_80, q_90 from CV OOF exact absolute residuals
      - 'scaled_sd': normalized nonconformity score |r_i| / (sd_i + eps) when informative
        OOF uncertainty `sd_i` (e.g. GP posterior SD or distance-adaptive residual scale)
        is available with strictly matching computation on OOF and test!
    """

    def __init__(self, bounds: Optional[Tuple[float, float]] = None):
        self.bounds = bounds
        self.mode_: str = "standard"
        self.q80_: float = 0.0
        self.q90_: float = 0.0
        self.eps_: float = 1e-3
        self.dist_slope_: float = 0.0
        self.dist_intercept_: float = 1.0
        self.min_scale_: float = 0.25
        self.scale_source_: str = "constant"
        self.cv_diagnostics_: Dict[str, Any] = {}

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        s = np.sort(np.asarray(scores, dtype=np.float64))
        n = len(s)
        if n == 0:
            return 0.0
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        k = min(max(k, 1), n)
        return float(s[k - 1])

    def _fit_dist_scale(self, abs_res: np.ndarray, max_tanimoto: np.ndarray) -> Tuple[float, float, float]:
        ok = np.isfinite(max_tanimoto) & np.isfinite(abs_res)
        med_r = float(np.median(abs_res)) if len(abs_res) > 0 else 1.0
        floor_s = max(0.30 * med_r, 1e-3)
        if ok.sum() < 25:
            return 0.0, max(med_r, 1e-3), floor_s
        dist = 1.0 - np.clip(max_tanimoto[ok], 0.0, 1.0)
        # Positive correlation between distance and |residual|?
        sp = float(spearmanr(dist, abs_res[ok]).statistic)
        if not np.isfinite(sp) or sp < 0.08:
            return 0.0, max(med_r, 1e-3), floor_s
        try:
            reg = HuberRegressor(alpha=1.0).fit(dist[:, None], abs_res[ok])
            slope = max(0.0, float(reg.coef_[0]))
            intercept = max(floor_s, float(reg.intercept_))
            return slope, intercept, floor_s
        except Exception:
            return 0.0, max(med_r, 1e-3), floor_s

    def _eval_scale(
        self,
        n: int,
        model_sd: Optional[np.ndarray],
        max_tanimoto: Optional[np.ndarray],
        source: str,
        slope: float,
        intercept: float,
        floor_s: float,
    ) -> np.ndarray:
        if source == "model_sd" and model_sd is not None:
            sd = np.asarray(model_sd, dtype=np.float64)
            med = float(np.nanmedian(sd)) if np.isfinite(sd).any() else 1.0
            sd = np.where(np.isfinite(sd), sd, med)
            return np.maximum(sd, floor_s)
        if source == "tanimoto_dist" and max_tanimoto is not None:
            d = 1.0 - np.clip(np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.5), 0.0, 1.0)
            return np.maximum(intercept + slope * d, floor_s)
        return np.ones(n, dtype=np.float64)

    def fit(
        self,
        y_cv: np.ndarray,
        pred_oof: np.ndarray,
        censor_cv: np.ndarray,
        oof_sd: Optional[np.ndarray] = None,
        oof_max_tanimoto: Optional[np.ndarray] = None,
    ) -> "ConformalIntervalCalibrator":
        mask = (censor_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof)
        if mask.sum() < 5:
            mask = np.isfinite(y_cv) & np.isfinite(pred_oof)
        abs_res = np.abs(y_cv[mask] - pred_oof[mask])
        med_r = float(np.median(abs_res)) if len(abs_res) > 0 else 1.0
        self.min_scale_ = max(0.25 * med_r, 1e-3)

        # Standard conformal quantiles
        std_q80 = self._conformal_quantile(abs_res, 0.20)
        std_q90 = self._conformal_quantile(abs_res, 0.10)
        self.cv_diagnostics_["standard"] = {
            "q80": round(std_q80, 5),
            "q90": round(std_q90, 5),
            "mean_width_80": round(2.0 * std_q80, 5),
            "mean_width_90": round(2.0 * std_q90, 5),
        }

        chosen_source = "constant"
        best_q80, best_q90 = std_q80, std_q90
        best_width90 = 2.0 * std_q90

        # Check model_sd scaling (e.g. exact GP posterior SD or sub-ensemble SD)
        if oof_sd is not None:
            sd_sub = oof_sd[mask]
            if np.isfinite(sd_sub).sum() >= 20 and np.nanstd(sd_sub) > 1e-5:
                sp = float(spearmanr(sd_sub[np.isfinite(sd_sub)], abs_res[np.isfinite(sd_sub)]).statistic)
                if np.isfinite(sp) and sp >= 0.15:
                    s_vec = np.maximum(np.where(np.isfinite(sd_sub), sd_sub, np.nanmedian(sd_sub)), self.min_scale_)
                    norm_scores = abs_res / s_vec
                    q80_s = self._conformal_quantile(norm_scores, 0.20)
                    q90_s = self._conformal_quantile(norm_scores, 0.10)
                    w90_s = float(np.mean(2.0 * q90_s * s_vec))
                    self.cv_diagnostics_["model_sd"] = {
                        "spearman_sd_err": round(sp, 4),
                        "q80_norm": round(q80_s, 5),
                        "q90_norm": round(q90_s, 5),
                        "mean_width_90": round(w90_s, 5),
                    }
                    if w90_s < best_width90 * 0.98:
                        chosen_source = "model_sd"
                        best_q80, best_q90 = q80_s, q90_s
                        best_width90 = w90_s

        # Check Tanimoto-distance adaptive scaling
        if oof_max_tanimoto is not None:
            slope, intercept, floor_s = self._fit_dist_scale(abs_res, oof_max_tanimoto[mask])
            if slope > 1e-4:
                s_vec = self._eval_scale(
                    len(abs_res), None, oof_max_tanimoto[mask], "tanimoto_dist", slope, intercept, floor_s
                )
                norm_scores = abs_res / s_vec
                q80_d = self._conformal_quantile(norm_scores, 0.20)
                q90_d = self._conformal_quantile(norm_scores, 0.10)
                w90_d = float(np.mean(2.0 * q90_d * s_vec))
                self.cv_diagnostics_["tanimoto_dist"] = {
                    "slope": round(slope, 5),
                    "intercept": round(intercept, 5),
                    "q80_norm": round(q80_d, 5),
                    "q90_norm": round(q90_d, 5),
                    "mean_width_90": round(w90_d, 5),
                }
                if w90_d < best_width90 * 0.98:
                    chosen_source = "tanimoto_dist"
                    best_q80, best_q90 = q80_d, q90_d
                    best_width90 = w90_d
                    self.dist_slope_ = slope
                    self.dist_intercept_ = intercept
                    self.min_scale_ = floor_s

        self.scale_source_ = chosen_source
        self.mode_ = "standard" if chosen_source == "constant" else f"adaptive_{chosen_source}"
        self.q80_ = best_q80
        self.q90_ = best_q90
        return self

    def predict_intervals(
        self,
        pred: np.ndarray,
        model_sd: Optional[np.ndarray] = None,
        max_tanimoto: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(pred, dtype=np.float64)
        s = self._eval_scale(
            len(p), model_sd, max_tanimoto, self.scale_source_, self.dist_slope_, self.dist_intercept_, self.min_scale_
        )
        hw80 = self.q80_ * s
        hw90 = self.q90_ * s
        lo80, hi80 = p - hw80, p + hw80
        lo90, hi90 = p - hw90, p + hw90
        if self.bounds is not None:
            lo80 = np.clip(lo80, self.bounds[0], self.bounds[1])
            hi80 = np.clip(hi80, self.bounds[0], self.bounds[1])
            lo90 = np.clip(lo90, self.bounds[0], self.bounds[1])
            hi90 = np.clip(hi90, self.bounds[0], self.bounds[1])
        return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        pred: np.ndarray,
        censor_type: np.ndarray,
        model_sd: Optional[np.ndarray] = None,
        max_tanimoto: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        ivs = self.predict_intervals(pred, model_sd=model_sd, max_tanimoto=max_tanimoto)
        mask = (censor_type == "exact") & np.isfinite(y_true) & np.isfinite(pred)
        if not mask.any():
            return {"mode": self.mode_, "n_eval": 0, "cov80": None, "cov90": None}
        yt = y_true[mask]
        c80 = float(np.mean((yt >= ivs["lo80"][mask]) & (yt <= ivs["hi80"][mask])))
        c90 = float(np.mean((yt >= ivs["lo90"][mask]) & (yt <= ivs["hi90"][mask])))
        w80 = float(np.mean(ivs["hi80"][mask] - ivs["lo80"][mask]))
        w90 = float(np.mean(ivs["hi90"][mask] - ivs["lo90"][mask]))
        return {
            "mode": self.mode_,
            "n_eval": int(mask.sum()),
            "cov80": round(c80, 4),
            "cov90": round(c90, 4),
            "mean_width_80": round(w80, 4),
            "mean_width_90": round(w90, 4),
        }


def compute_ad_stratification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    censor_type: np.ndarray,
    max_tanimoto: np.ndarray,
    tiers: List[str],
    task_type: str,
) -> Dict[str, Any]:
    """Empirical holdout error vs Tanimoto AD tier stratification."""
    tier_arr = np.array(tiers)
    summary: Dict[str, Any] = {}
    for tier_name in ("In-Domain", "Borderline", "OOD", "SeqOnly"):
        m = (tier_arr == tier_name) & np.isfinite(y_pred) & np.isfinite(y_true)
        if task_type == "regression":
            m = m & (censor_type == "exact")
            cnt = int(m.sum())
            if cnt == 0:
                continue
            err = np.abs(y_true[m] - y_pred[m])
            summary[tier_name] = {
                "n": cnt,
                "mae": round(float(np.mean(err)), 5),
                "median_ae": round(float(np.median(err)), 5),
                "rmse": round(float(np.sqrt(np.mean(err ** 2))), 5),
                "mean_max_tanimoto": round(float(np.nanmean(max_tanimoto[m])), 4) if tier_name != "SeqOnly" else None,
            }
        else:
            cnt = int(m.sum())
            if cnt == 0:
                continue
            yt = y_true[m].astype(np.int32)
            p = np.clip(y_pred[m], 1e-6, 1.0 - 1e-6)
            entry: Dict[str, Any] = {
                "n": cnt,
                "brier": round(float(brier_score_loss(yt, p)), 5),
                "err_rate": round(float(np.mean((p >= 0.5) != yt)), 5),
                "mean_max_tanimoto": round(float(np.nanmean(max_tanimoto[m])), 4) if tier_name != "SeqOnly" else None,
            }
            if len(np.unique(yt)) == 2:
                entry["roc_auc"] = round(float(roc_auc_score(yt, p)), 5)
            summary[tier_name] = entry

    # Overall Spearman correlation between max_tanimoto and |error|
    ok = np.isfinite(max_tanimoto) & np.isfinite(y_pred) & np.isfinite(y_true)
    if task_type == "regression":
        ok = ok & (censor_type == "exact")
    if ok.sum() >= 10:
        abs_err = np.abs(y_true[ok] - y_pred[ok])
        sp = float(spearmanr(max_tanimoto[ok], abs_err).statistic)
        summary["spearman_tanimoto_vs_abs_err"] = round(sp, 4) if np.isfinite(sp) else None
    else:
        summary["spearman_tanimoto_vs_abs_err"] = None
    return summary
