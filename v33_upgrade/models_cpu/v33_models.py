#!/usr/bin/env python3
"""Anti-(p >> n) model families for V33 CPU modeling:
  - Exact Tanimoto + RBF + Condition Gaussian Process Regressor (GP-MLL)
  - Exact Tanimoto + RBF + Condition Kernel Ridge Regressor & Kernel Logistic Classifier
  - Censored-Aware XGBoost AFT Regressor (exact/right/left/interval)
  - Regularized Linear, Huber, SVR, Shallow/Medium LightGBM, XGBoost, ExtraTrees
  - Winsorized & Bounded [0, 1] wrappers
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore", category=UserWarning)


def tanimoto_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact Jaccard/Tanimoto kernel matrix on binary/non-negative float32 arrays."""
    A64 = np.asarray(A, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    dot = A64 @ B64.T
    na = (A64 * A64).sum(axis=1)[:, None]
    nb = (B64 * B64).sum(axis=1)[None, :]
    denom = na + nb - dot
    K = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 1e-12)
    return K


def rbf_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return cdist(np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64), metric="sqeuclidean")


def cond_kernel_matrix(A: np.ndarray, B: np.ndarray, n_cond_cols: int) -> np.ndarray:
    if A.shape[1] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    return (np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T) / float(max(1, n_cond_cols))


class TanimotoRBFGPR(BaseEstimator, RegressorMixin):
    """Exact Gaussian Process Regression with composite Tanimoto + RBF + Condition kernel.

    Kernel:
        K(i, j) = w_tan * K_Tanimoto(bits_i, bits_j)
                + w_rbf * exp(-gamma * ||z_i - z_j||^2)
                + w_cond * K_cond(c_i, c_j)
    Hyperparameters (w_tan, w_rbf, w_cond, gamma, noise_var) are optimized strictly on the
    training fold by maximizing exact GP Log Marginal Likelihood via multi-start L-BFGS-B.
    """

    def __init__(self, n_restarts: int = 3, random_state: int = 42, jitter: float = 1e-6):
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.jitter = jitter

    def fit(self, X_bundle: Dict[str, Any], y: np.ndarray) -> "TanimotoRBFGPR":
        X_phys = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        X_bits = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        X_cond = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        n_cond_cols = int(X_bundle.get("n_cond_cols", 1))
        has_cond = X_cond.shape[1] > 0

        y_arr = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr))
        if self.y_std_ < 1e-8:
            self.y_std_ = 1.0
        y_norm = (y_arr - self.y_mean_) / self.y_std_

        n, d = X_phys.shape
        K_tan = tanimoto_matrix(X_bits, X_bits)
        D2_phys = rbf_sqdist(X_phys, X_phys)
        K_cond = cond_kernel_matrix(X_cond, X_cond, n_cond_cols) if has_cond else None

        # Median heuristic for RBF lengthscale
        tri = D2_phys[np.triu_indices(n, k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
        base_gamma = 1.0 / max(med_d2, 1.0)

        eye = np.eye(n, dtype=np.float64)

        def neg_mll(theta: np.ndarray) -> float:
            w_tan = np.exp(theta[0])
            w_rbf = np.exp(theta[1])
            gamma = np.exp(theta[2])
            noise = np.exp(theta[3])
            K = w_tan * K_tan + w_rbf * np.exp(-gamma * D2_phys)
            if has_cond:
                w_cond = np.exp(theta[4])
                K = K + w_cond * K_cond
            K = K + (noise + self.jitter) * eye
            try:
                L, lower = cho_factor(K, lower=True, check_finite=False)
                alpha = cho_solve((L, lower), y_norm, check_finite=False)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                nll = 0.5 * (float(y_norm @ alpha) + logdet + n * np.log(2.0 * np.pi))
                # Weak log-normal hyperprior to prevent degenerate extremes on tiny n
                reg = 0.02 * float(np.sum((theta[:2]) ** 2)) + 0.02 * float((theta[3] + 1.5) ** 2)
                return nll + reg
            except np.linalg.LinAlgError:
                return 1e12

        bounds = [
            (-5.0, 2.5),  # log w_tan
            (-5.0, 2.5),  # log w_rbf
            (np.log(base_gamma) - 4.0, np.log(base_gamma) + 4.0),  # log gamma
            (-6.0, 1.5),  # log noise
        ]
        if has_cond:
            bounds.append((-6.0, 2.0))  # log w_cond

        inits = [
            [np.log(0.45), np.log(0.45), np.log(base_gamma), np.log(0.15)],
            [np.log(0.75), np.log(0.20), np.log(base_gamma * 0.5), np.log(0.25)],
            [np.log(0.20), np.log(0.75), np.log(base_gamma * 2.0), np.log(0.10)],
        ]
        if has_cond:
            inits = [v + [np.log(0.15)] for v in inits]

        best_val = np.inf
        best_theta = np.array(inits[0], dtype=np.float64)
        for init in inits[: max(1, self.n_restarts)]:
            res = minimize(
                neg_mll,
                x0=np.array(init, dtype=np.float64),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 80},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta = res.x.copy()

        self.w_tan_ = float(np.exp(best_theta[0]))
        self.w_rbf_ = float(np.exp(best_theta[1]))
        self.gamma_ = float(np.exp(best_theta[2]))
        self.noise_ = float(np.exp(best_theta[3]))
        self.w_cond_ = float(np.exp(best_theta[4])) if has_cond else 0.0
        self.has_cond_ = has_cond
        self.n_cond_cols_ = n_cond_cols

        K_final = self.w_tan_ * K_tan + self.w_rbf_ * np.exp(-self.gamma_ * D2_phys)
        if has_cond:
            K_final = K_final + self.w_cond_ * K_cond
        K_final = K_final + (self.noise_ + self.jitter) * eye

        L, lower = cho_factor(K_final, lower=True, check_finite=False)
        self.L_ = L
        self.lower_ = lower
        self.alpha_ = cho_solve((L, lower), y_norm, check_finite=False)
        self.X_phys_ = X_phys.astype(np.float32)
        self.X_bits_ = X_bits.astype(np.float32)
        self.X_cond_ = X_cond.astype(np.float32)
        return self

    def _cross_kernel(self, X_bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        Xp = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        Xb = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        Xc = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        K_tan = tanimoto_matrix(Xb, self.X_bits_)
        D2 = rbf_sqdist(Xp, self.X_phys_)
        K_star = self.w_tan_ * K_tan + self.w_rbf_ * np.exp(-self.gamma_ * D2)
        k_diag = np.full(Xp.shape[0], self.w_tan_ + self.w_rbf_, dtype=np.float64)
        if self.has_cond_:
            Kc = cond_kernel_matrix(Xc, self.X_cond_, self.n_cond_cols_)
            K_star = K_star + self.w_cond_ * Kc
            cond_diag = (Xc * Xc).sum(axis=1) / float(max(1, self.n_cond_cols_))
            k_diag = k_diag + self.w_cond_ * cond_diag
        return K_star, k_diag

    def predict(self, X_bundle: Dict[str, Any]) -> np.ndarray:
        K_star, _ = self._cross_kernel(X_bundle)
        pred_norm = K_star @ self.alpha_
        return (pred_norm * self.y_std_ + self.y_mean_).astype(np.float64)

    def predict_with_std(self, X_bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        K_star, k_diag = self._cross_kernel(X_bundle)
        pred_norm = K_star @ self.alpha_
        # Solve L v = K_star^T -> predictive variance = k_diag + noise - ||v||^2
        from scipy.linalg import solve_triangular

        v = solve_triangular(self.L_, K_star.T, lower=self.lower_, check_finite=False)
        var_norm = np.maximum(k_diag + self.noise_ - np.sum(v * v, axis=0), 1e-6)
        mean = pred_norm * self.y_std_ + self.y_mean_
        std = np.sqrt(var_norm) * self.y_std_
        return mean.astype(np.float64), std.astype(np.float64)


class TanimotoRBFKRR(BaseEstimator, RegressorMixin):
    """Exact Tanimoto + RBF + Condition Kernel Ridge Regressor."""

    def __init__(
        self,
        alpha: float = 0.2,
        w_tan: float = 0.45,
        w_cond: float = 0.15,
        gamma_mult: float = 1.0,
    ):
        self.alpha = alpha
        self.w_tan = w_tan
        self.w_cond = w_cond
        self.gamma_mult = gamma_mult

    def fit(self, X_bundle: Dict[str, Any], y: np.ndarray) -> "TanimotoRBFKRR":
        X_phys = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        X_bits = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        X_cond = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        n_cond_cols = int(X_bundle.get("n_cond_cols", 1))
        has_cond = X_cond.shape[1] > 0

        y_arr = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(np.mean(y_arr))
        y_c = y_arr - self.y_mean_

        n, d = X_phys.shape
        D2 = rbf_sqdist(X_phys, X_phys)
        tri = D2[np.triu_indices(n, k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
        self.gamma_ = float(self.gamma_mult) / max(med_d2, 1.0)

        w_c = self.w_cond if has_cond else 0.0
        w_t = self.w_tan * (1.0 - w_c)
        w_r = (1.0 - self.w_tan) * (1.0 - w_c)
        self.w_t_, self.w_r_, self.w_c_ = w_t, w_r, w_c
        self.has_cond_ = has_cond
        self.n_cond_cols_ = n_cond_cols

        K = w_t * tanimoto_matrix(X_bits, X_bits) + w_r * np.exp(-self.gamma_ * D2)
        if has_cond:
            K = K + w_c * cond_kernel_matrix(X_cond, X_cond, n_cond_cols)
        K[np.diag_indices(n)] += float(self.alpha)

        L, lower = cho_factor(K, lower=True, check_finite=False)
        self.dual_coef_ = cho_solve((L, lower), y_c, check_finite=False)
        self.X_phys_ = X_phys.astype(np.float32)
        self.X_bits_ = X_bits.astype(np.float32)
        self.X_cond_ = X_cond.astype(np.float32)
        return self

    def predict(self, X_bundle: Dict[str, Any]) -> np.ndarray:
        Xp = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        Xb = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        Xc = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        K_star = self.w_t_ * tanimoto_matrix(Xb, self.X_bits_) + self.w_r_ * np.exp(
            -self.gamma_ * rbf_sqdist(Xp, self.X_phys_)
        )
        if self.has_cond_:
            K_star = K_star + self.w_c_ * cond_kernel_matrix(Xc, self.X_cond_, self.n_cond_cols_)
        return (K_star @ self.dual_coef_ + self.y_mean_).astype(np.float64)


class TanimotoRBFKLR(BaseEstimator, ClassifierMixin):
    """Exact Tanimoto + RBF + Condition Kernel Logistic Regression for classification (n <= 3500)."""

    def __init__(
        self,
        C: float = 2.0,
        w_tan: float = 0.5,
        w_cond: float = 0.15,
        gamma_mult: float = 1.0,
    ):
        self.C = C
        self.w_tan = w_tan
        self.w_cond = w_cond
        self.gamma_mult = gamma_mult

    def fit(self, X_bundle: Dict[str, Any], y: np.ndarray) -> "TanimotoRBFKLR":
        X_phys = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        X_bits = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        X_cond = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        n_cond_cols = int(X_bundle.get("n_cond_cols", 1))
        has_cond = X_cond.shape[1] > 0

        y_arr = np.asarray(y, dtype=np.float64)
        n, d = X_phys.shape
        D2 = rbf_sqdist(X_phys, X_phys)
        tri = D2[np.triu_indices(n, k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
        self.gamma_ = float(self.gamma_mult) / max(med_d2, 1.0)

        w_c = self.w_cond if has_cond else 0.0
        w_t = self.w_tan * (1.0 - w_c)
        w_r = (1.0 - self.w_tan) * (1.0 - w_c)
        self.w_t_, self.w_r_, self.w_c_ = w_t, w_r, w_c
        self.has_cond_ = has_cond
        self.n_cond_cols_ = n_cond_cols

        K = w_t * tanimoto_matrix(X_bits, X_bits) + w_r * np.exp(-self.gamma_ * D2)
        if has_cond:
            K = K + w_c * cond_kernel_matrix(X_cond, X_cond, n_cond_cols)

        lam = 1.0 / max(float(self.C), 1e-4)

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            alpha = params[:-1]
            b = params[-1]
            Ka = K @ alpha
            z = Ka + b
            # Stable log(1 + exp(z)) - y*z
            pos_z = np.maximum(z, 0.0)
            loss_data = np.sum(pos_z - y_arr * z + np.log1p(np.exp(-np.abs(z))))
            loss = float(loss_data + 0.5 * lam * float(alpha @ Ka))
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
            diff = p - y_arr
            grad_alpha = K @ (diff + lam * alpha)
            grad_b = float(np.sum(diff))
            grad = np.empty_like(params)
            grad[:-1] = grad_alpha
            grad[-1] = grad_b
            return loss, grad

        x0 = np.zeros(n + 1, dtype=np.float64)
        p_mean = float(np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4))
        x0[-1] = np.log(p_mean / (1.0 - p_mean))
        res = minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 120})
        self.alpha_ = res.x[:-1].copy()
        self.bias_ = float(res.x[-1])
        self.X_phys_ = X_phys.astype(np.float32)
        self.X_bits_ = X_bits.astype(np.float32)
        self.X_cond_ = X_cond.astype(np.float32)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X_bundle: Dict[str, Any]) -> np.ndarray:
        Xp = np.asarray(X_bundle["X_phys"], dtype=np.float64)
        Xb = np.asarray(X_bundle["X_bits"], dtype=np.float64)
        Xc = np.asarray(X_bundle["X_cond"], dtype=np.float64)
        K_star = self.w_t_ * tanimoto_matrix(Xb, self.X_bits_) + self.w_r_ * np.exp(
            -self.gamma_ * rbf_sqdist(Xp, self.X_phys_)
        )
        if self.has_cond_:
            K_star = K_star + self.w_c_ * cond_kernel_matrix(Xc, self.X_cond_, self.n_cond_cols_)
        z = np.clip(K_star @ self.alpha_ + self.bias_, -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p1, p1], axis=1)


class XGBoostAFTRegressor:
    """Censored-aware XGBoost `survival:aft` operating natively in linear y-space.

    Maps y -> T = exp(y - y_ref) > 0:
      - exact:    [exp(y - y_ref), exp(y - y_ref)]
      - right:    [exp(y_lower - y_ref), +inf]
      - left:     [0.0, exp(y_upper - y_ref)]
      - interval: [exp(y_lower - y_ref), exp(y_upper - y_ref)]
    Prediction via `output_margin=True` directly recovers y_hat = margin + y_ref!
    """

    def __init__(
        self,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 0.8,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        min_child_weight: float = 5.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        reg_lambda: float = 5.0,
        reg_alpha: float = 0.1,
        nthread: int = 4,
        seed: int = 42,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = aft_loss_distribution_scale
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.nthread = nthread
        self.seed = seed
        self.bounds = bounds
        self.y_ref_: float = 0.0
        self.booster_: Optional[xgb.Booster] = None

    def fit(
        self,
        X: np.ndarray,
        y_value: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        censor_type: np.ndarray,
    ) -> "XGBoostAFTRegressor":
        exact_mask = censor_type == "exact"
        self.y_ref_ = float(np.median(y_value[exact_mask])) if exact_mask.any() else float(np.median(y_value))

        # Clamp exponent shift to [-20, 20] for float32 safety
        lo_t = np.exp(np.clip(y_lower - self.y_ref_, -20.0, 20.0)).astype(np.float32)
        hi_t = np.exp(np.clip(y_upper - self.y_ref_, -20.0, 20.0)).astype(np.float32)

        right_mask = censor_type == "right"
        left_mask = censor_type == "left"
        hi_t[right_mask] = np.inf
        lo_t[left_mask] = 0.0
        # Exact rows enforce exact equality
        lo_t[exact_mask] = np.exp(np.clip(y_value[exact_mask] - self.y_ref_, -20.0, 20.0)).astype(np.float32)
        hi_t[exact_mask] = lo_t[exact_mask]
        # Safeguard float32 rounding on tight interval/left bounds
        non_exact_fin = (~exact_mask) & (~right_mask)
        tight = non_exact_fin & (hi_t <= lo_t)
        if tight.any():
            hi_t[tight] = lo_t[tight] * 1.001 + 1e-6

        dtrain = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        dtrain.set_float_info("label_lower_bound", lo_t)
        dtrain.set_float_info("label_upper_bound", hi_t)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.aft_loss_distribution,
            "aft_loss_distribution_scale": float(self.aft_loss_distribution_scale),
            "tree_method": "hist",
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "min_child_weight": float(self.min_child_weight),
            "subsample": float(self.subsample),
            "colsample_bytree": float(self.colsample_bytree),
            "reg_lambda": float(self.reg_lambda),
            "reg_alpha": float(self.reg_alpha),
            "base_score": 1.0,
            "nthread": int(self.nthread),
            "seed": int(self.seed),
            "verbosity": 0,
        }
        self.booster_ = xgb.train(params, dtrain, num_boost_round=int(self.n_estimators), verbose_eval=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.booster_ is not None
        dtest = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        margin = self.booster_.predict(dtest, output_margin=True).astype(np.float64)
        pred = margin + self.y_ref_
        if self.bounds is not None:
            pred = np.clip(pred, self.bounds[0], self.bounds[1])
        return pred


class WinsorizedRegressor(BaseEstimator, RegressorMixin):
    """Clips training targets ONLY (never validation targets) for outlier robustness."""

    def __init__(self, base_estimator: Any, upper_cap: float = 5.0):
        self.base_estimator = base_estimator
        self.upper_cap = upper_cap

    def fit(self, X: Any, y: np.ndarray) -> "WinsorizedRegressor":
        y_clip = np.clip(np.asarray(y, dtype=np.float64), None, self.upper_cap)
        self.base_estimator.fit(X, y_clip)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.base_estimator.predict(X), dtype=np.float64)


class BoundedRegressor(BaseEstimator, RegressorMixin):
    """Wraps any regressor and enforces physical bounds [lo, hi] (e.g. [0, 1] for PPB/F)."""

    def __init__(self, base_estimator: Any, lo: float = 0.0, hi: float = 1.0):
        self.base_estimator = base_estimator
        self.lo = lo
        self.hi = hi

    def fit(self, X: Any, y: np.ndarray) -> "BoundedRegressor":
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(np.asarray(self.base_estimator.predict(X), dtype=np.float64), self.lo, self.hi)

    def predict_with_std(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.base_estimator, "predict_with_std"):
            m, s = self.base_estimator.predict_with_std(X)
            return np.clip(m, self.lo, self.hi), s
        return self.predict(X), np.zeros(len(X["X_phys"]) if isinstance(X, dict) else len(X), dtype=np.float64)


def make_estimator(
    family: str,
    params: Dict[str, Any],
    task_type: str,
    n_jobs: int = 4,
    bounds: Optional[Tuple[float, float]] = None,
    winsor_cap: Optional[float] = None,
    seed: int = 42,
) -> Any:
    """Instantiate a clean configured model estimator."""
    p = dict(params)

    if task_type == "regression":
        if family == "TanimotoRBF_GPR":
            est = TanimotoRBFGPR(n_restarts=p.get("n_restarts", 3), random_state=seed)
        elif family == "TanimotoRBF_KRR":
            est = TanimotoRBFKRR(
                alpha=p["alpha"],
                w_tan=p["w_tan"],
                w_cond=p.get("w_cond", 0.15),
                gamma_mult=p.get("gamma_mult", 1.0),
            )
        elif family == "Ridge":
            est = Ridge(alpha=p["alpha"])
        elif family == "ElasticNet":
            est = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], max_iter=3000, random_state=seed)
        elif family == "Huber":
            est = HuberRegressor(alpha=p["alpha"], epsilon=p.get("epsilon", 1.35), max_iter=1000)
        elif family == "SVR_RBF":
            est = SVR(C=p["C"], epsilon=p.get("epsilon", 0.1), gamma=p.get("gamma", "scale"))
        elif family == "LightGBM":
            est = lgb.LGBMRegressor(
                objective=p.get("objective", "regression"),
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                num_leaves=p["num_leaves"],
                min_child_samples=p["min_child_samples"],
                subsample=p.get("subsample", 0.8),
                subsample_freq=1,
                colsample_bytree=p.get("colsample_bytree", 0.7),
                reg_lambda=p.get("reg_lambda", 5.0),
                reg_alpha=p.get("reg_alpha", 0.1),
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
            )
        elif family == "XGBoost":
            est = xgb.XGBRegressor(
                objective=p.get("objective", "reg:squarederror"),
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                min_child_weight=p.get("min_child_weight", 5.0),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.7),
                reg_lambda=p.get("reg_lambda", 5.0),
                reg_alpha=p.get("reg_alpha", 0.1),
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=seed,
                verbosity=0,
            )
        elif family == "ExtraTrees":
            est = ExtraTreesRegressor(
                n_estimators=p.get("n_estimators", 300),
                max_depth=p.get("max_depth", 12),
                min_samples_leaf=p.get("min_samples_leaf", 2),
                max_features=p.get("max_features", 0.3),
                n_jobs=n_jobs,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown regression family: {family}")

        if winsor_cap is not None:
            est = WinsorizedRegressor(est, upper_cap=winsor_cap)
        if bounds is not None:
            est = BoundedRegressor(est, lo=bounds[0], hi=bounds[1])
        return est

    # Classification
    if family == "TanimotoRBF_KLR":
        return TanimotoRBFKLR(
            C=p["C"],
            w_tan=p["w_tan"],
            w_cond=p.get("w_cond", 0.15),
            gamma_mult=p.get("gamma_mult", 1.0),
        )
    if family == "Logistic":
        return LogisticRegression(C=p["C"], max_iter=500, solver="lbfgs")
    if family == "SVC_RBF":
        return SVC(C=p["C"], gamma=p.get("gamma", "scale"), probability=True, random_state=seed)
    if family == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            num_leaves=p["num_leaves"],
            min_child_samples=p["min_child_samples"],
            subsample=p.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.7),
            reg_lambda=p.get("reg_lambda", 2.0),
            reg_alpha=p.get("reg_alpha", 0.1),
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
    if family == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            min_child_weight=p.get("min_child_weight", 3.0),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.7),
            reg_lambda=p.get("reg_lambda", 2.0),
            reg_alpha=p.get("reg_alpha", 0.1),
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )
    if family == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 18),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            max_features=p.get("max_features", 0.2),
            n_jobs=n_jobs,
            random_state=seed,
        )
    raise ValueError(f"Unknown classification family: {family}")


def get_candidate_recipes(
    size_tier: str,
    task_type: str,
    track: str,
    n_cv: int,
    outlier_winsor_cap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build curated, anti-(p >> n) candidate recipes for 5-fold grouped CV selection."""
    recipes: List[Dict[str, Any]] = []

    if track == "seq_only":
        # Compact 35D sequence physchem + conds
        if task_type == "regression":
            for alpha in (0.3, 3.0, 30.0):
                recipes.append({"route": "physchem_core", "family": "Ridge", "scale": True, "params": {"alpha": alpha}})
            for C in (1.0, 5.0):
                recipes.append({"route": "physchem_core", "family": "SVR_RBF", "scale": True, "params": {"C": C, "epsilon": 0.1}})
            for cfg in [
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 15, "reg_lambda": 5.0},
                {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 2.0},
            ]:
                recipes.append({"route": "physchem_core", "family": "LightGBM", "scale": False, "params": cfg})
            recipes.append({
                "route": "physchem_core",
                "family": "XGBoost",
                "scale": False,
                "params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5.0, "reg_lambda": 5.0},
            })
            recipes.append({
                "route": "physchem_core",
                "family": "ExtraTrees",
                "scale": False,
                "params": {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 2, "max_features": 0.5},
            })
        else:
            for C in (0.3, 3.0):
                recipes.append({"route": "physchem_core", "family": "Logistic", "scale": True, "params": {"C": C}})
            for cfg in [
                {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 2.0},
                {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 48, "min_child_samples": 15, "reg_lambda": 5.0},
            ]:
                recipes.append({"route": "physchem_core", "family": "LightGBM", "scale": False, "params": cfg})
            recipes.append({
                "route": "physchem_core",
                "family": "XGBoost",
                "scale": False,
                "params": {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3.0, "reg_lambda": 3.0},
            })
            recipes.append({
                "route": "physchem_core",
                "family": "ExtraTrees",
                "scale": False,
                "params": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.4},
            })
        return recipes

    # --- SMALL-N REGRESSION (n < 300): strictly anti-(p >> n) ---
    if size_tier == "small" and task_type == "regression":
        # 1. Exact Tanimoto + RBF + Condition GP & KRR
        recipes.append({
            "route": "tanimoto_hybrid",
            "family": "TanimotoRBF_GPR",
            "scale": True,
            "params": {"n_restarts": 3},
        })
        for w_tan in (0.30, 0.50, 0.70):
            for alpha in (0.05, 0.25, 1.0):
                recipes.append({
                    "route": "tanimoto_hybrid",
                    "family": "TanimotoRBF_KRR",
                    "scale": True,
                    "params": {"alpha": alpha, "w_tan": w_tan, "w_cond": 0.15, "gamma_mult": 1.0},
                })
        # 2. Compact physchem_core and physchem_svd (32D) across diverse regularized learners
        for route in ("physchem_core", "physchem_svd"):
            for alpha in (1.0, 10.0, 50.0, 200.0, 600.0):
                recipes.append({"route": route, "family": "Ridge", "scale": True, "params": {"alpha": alpha}})
            for alpha, l1 in [(0.02, 0.3), (0.08, 0.5), (0.2, 0.2)]:
                recipes.append({"route": route, "family": "ElasticNet", "scale": True, "params": {"alpha": alpha, "l1_ratio": l1}})
            for alpha in (1.0, 10.0, 50.0):
                recipes.append({"route": route, "family": "Huber", "scale": True, "params": {"alpha": alpha, "epsilon": 1.35}})
            for C in (0.5, 2.0, 8.0):
                recipes.append({"route": route, "family": "SVR_RBF", "scale": True, "params": {"C": C, "epsilon": 0.05}})
            for cfg in [
                {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 3, "num_leaves": 6, "min_child_samples": 5, "reg_lambda": 5.0, "colsample_bytree": 0.7},
                {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 10, "min_child_samples": 10, "reg_lambda": 15.0, "colsample_bytree": 0.6},
                {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 6, "min_child_samples": 12, "reg_lambda": 2.0, "colsample_bytree": 0.8},
            ]:
                recipes.append({"route": route, "family": "LightGBM", "scale": False, "params": cfg})
            for cfg in [
                {"n_estimators": 150, "learning_rate": 0.04, "max_depth": 2, "min_child_weight": 3.0, "reg_lambda": 5.0, "colsample_bytree": 0.7},
                {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 5.0, "reg_lambda": 15.0, "colsample_bytree": 0.6},
                {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.7},
            ]:
                recipes.append({"route": route, "family": "XGBoost", "scale": False, "params": cfg})
            for cfg in [
                {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2, "max_features": 0.4},
                {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 3, "max_features": 0.3},
            ]:
                recipes.append({"route": route, "family": "ExtraTrees", "scale": False, "params": cfg})
        return recipes

    # --- MEDIUM / LARGE REGRESSION (n >= 300) ---
    if task_type == "regression":
        routes = ["physchem_core", "physchem_morgan", "physchem_fp_all"]
        for route in routes:
            for alpha in (10.0, 100.0, 500.0):
                recipes.append({"route": route, "family": "Ridge", "scale": True, "params": {"alpha": alpha}})
            for cfg in [
                {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 5.0, "colsample_bytree": 0.6},
                {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 48, "min_child_samples": 20, "reg_lambda": 10.0, "colsample_bytree": 0.5},
                {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "num_leaves": 20, "min_child_samples": 10, "reg_lambda": 2.0, "colsample_bytree": 0.7, "objective": "huber"},
            ]:
                recipes.append({"route": route, "family": "LightGBM", "scale": False, "params": cfg})
            xgb_cfgs = (
                [{"n_estimators": 450, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 6.0, "reg_lambda": 8.0, "colsample_bytree": 0.5}]
                if size_tier == "large"
                else [
                    {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6},
                    {"n_estimators": 550, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.5},
                ]
            )
            for cfg in xgb_cfgs:
                recipes.append({"route": route, "family": "XGBoost", "scale": False, "params": cfg})
            recipes.append({
                "route": route,
                "family": "ExtraTrees",
                "scale": False,
                "params": {"n_estimators": 300, "max_depth": 18, "min_samples_leaf": 2, "max_features": 0.2 if route != "physchem_core" else 0.5},
            })

        # Kernel methods where n_cv <= 3500
        if n_cv <= 3500:
            for w_tan in (0.4, 0.6):
                for alpha in (0.08, 0.3):
                    recipes.append({
                        "route": "tanimoto_hybrid",
                        "family": "TanimotoRBF_KRR",
                        "scale": True,
                        "params": {"alpha": alpha, "w_tan": w_tan, "w_cond": 0.15, "gamma_mult": 1.0},
                    })
            for C in (1.0, 5.0):
                recipes.append({"route": "physchem_core", "family": "SVR_RBF", "scale": True, "params": {"C": C, "epsilon": 0.1}})
                if n_cv <= 1500:
                    recipes.append({"route": "physchem_morgan", "family": "SVR_RBF", "scale": True, "params": {"C": C, "epsilon": 0.1}})

        # Explicit winsorized & Huber outlier ablation for MIC_S_aureus
        if outlier_winsor_cap is not None:
            for route in ("physchem_morgan", "physchem_fp_all"):
                recipes.append({
                    "route": route,
                    "family": "LightGBM",
                    "scale": False,
                    "winsor_cap": outlier_winsor_cap,
                    "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 15, "reg_lambda": 5.0, "colsample_bytree": 0.6, "objective": "huber"},
                })
                recipes.append({
                    "route": route,
                    "family": "XGBoost",
                    "scale": False,
                    "winsor_cap": outlier_winsor_cap,
                    "params": {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6, "objective": "reg:pseudohubererror"},
                })
        return recipes

    # --- CLASSIFICATION (6 endpoints) ---
    routes = ["physchem_core", "physchem_morgan", "physchem_fp_all"]
    for route in routes:
        for C in (0.1, 1.0):
            recipes.append({"route": route, "family": "Logistic", "scale": True, "params": {"C": C}})
        for cfg in [
            {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 3.0, "colsample_bytree": 0.6},
            {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 63, "min_child_samples": 25, "reg_lambda": 8.0, "colsample_bytree": 0.5},
        ]:
            recipes.append({"route": route, "family": "LightGBM", "scale": False, "params": cfg})
        xgb_cls_cfgs = (
            [{"n_estimators": 450, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 4.0, "reg_lambda": 5.0, "colsample_bytree": 0.5}]
            if size_tier == "large"
            else [
                {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 3.0, "reg_lambda": 3.0, "colsample_bytree": 0.6},
                {"n_estimators": 550, "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 5.0, "reg_lambda": 8.0, "colsample_bytree": 0.5},
            ]
        )
        for cfg in xgb_cls_cfgs:
            recipes.append({"route": route, "family": "XGBoost", "scale": False, "params": cfg})
        recipes.append({
            "route": route,
            "family": "ExtraTrees",
            "scale": False,
            "params": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2, "max_features": 0.15 if route != "physchem_core" else 0.4},
        })

    if n_cv <= 3200:
        for w_tan in (0.45, 0.65):
            recipes.append({
                "route": "tanimoto_hybrid",
                "family": "TanimotoRBF_KLR",
                "scale": True,
                "params": {"C": 3.0, "w_tan": w_tan, "w_cond": 0.15, "gamma_mult": 1.0},
            })
    return recipes


def get_aft_candidates(size_tier: str) -> List[Dict[str, Any]]:
    """Curated XGBoost survival:aft candidate grid (tuned inside 5-fold CV)."""
    if size_tier == "small":
        routes = ["physchem_core", "physchem_svd"]
        cfgs = [
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.4, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 3.0, "reg_lambda": 8.0},
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.8, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 5.0, "reg_lambda": 10.0},
            {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 0.5, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 3.0, "reg_lambda": 8.0},
            {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 1.0, "max_depth": 2, "learning_rate": 0.04, "n_estimators": 180, "min_child_weight": 4.0, "reg_lambda": 12.0},
        ]
    else:
        routes = ["physchem_morgan", "physchem_fp_all"]
        cfgs = [
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.5, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 450, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6},
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 550, "min_child_weight": 6.0, "reg_lambda": 8.0, "colsample_bytree": 0.5},
            {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 0.6, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 450, "min_child_weight": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.6},
            {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 1.2, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 550, "min_child_weight": 6.0, "reg_lambda": 8.0, "colsample_bytree": 0.5},
        ]
    out = []
    for r in routes:
        for c in cfgs:
            out.append({"route": r, "params": c})
    return out
