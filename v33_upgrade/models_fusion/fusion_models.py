#!/usr/bin/env python3
"""Anti-(p >> n) multimodal preprocessors, fast kernel/linear/MLP/GBDT/AFT learners,
and strictly leak-free ModalityAwareStacker for V33 Wave-2D Multimodal Fusion.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from fusion_bank import ConditionEncoder, FusionBank

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def compute_sqdist_matrix(A: np.ndarray, B: np.ndarray, use_float32: bool = False) -> np.ndarray:
    """Fast BLAS pairwise squared Euclidean distance matrix (float64 or float32 for huge n)."""
    dtype = np.float32 if use_float32 else np.float64
    A_t = np.asarray(A, dtype=dtype)
    B_t = np.asarray(B, dtype=dtype)
    na = np.sum(A_t * A_t, axis=1)[:, None]
    nb = np.sum(B_t * B_t, axis=1)[None, :]
    d2 = na + nb - dtype(2.0) * (A_t @ B_t.T)
    return np.maximum(d2, dtype(0.0))


class FastPCA:
    """Drop-in fast PCA using exact dual eigh when n <= d and n <= 2200 (50x faster on embeddings!),
    and randomized SVD (n_iter=3) otherwise. Fully picklable with numpy arrays only.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.n_components_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X32 = np.asarray(X, dtype=np.float32)
        n, d = X32.shape
        k = max(1, min(self.n_components, n - 1, d))
        self.n_components_ = k
        mu = X32.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.mean_ = mu
        Xc = (X32 - mu)

        if n <= d and n <= 2200:
            # Exact dual Gram eigendecomposition: O(n^2 d + n^3), blazing fast for n <= 2200!
            Xc64 = Xc.astype(np.float64)
            G = Xc64 @ Xc64.T
            evals, U = eigh(G, subset_by_index=[n - k, n - 1])
            order = np.argsort(evals)[::-1]
            evals = np.maximum(evals[order], 1e-12)
            U = U[:, order]
            s = np.sqrt(evals)
            # Components V^T = diag(1/s) U^T Xc -> shape (k, d)
            Vt = (U.T @ Xc64) / s[:, None]
            self.components_ = Vt.astype(np.float32)
            # Projections = U * s
            return (U * s[None, :]).astype(np.float32)

        pca = PCA(n_components=k, svd_solver="randomized", iterated_power=3, random_state=self.random_state)
        proj = pca.fit_transform(Xc).astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)
        return proj

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        Xc = np.asarray(X, dtype=np.float32) - self.mean_
        return (Xc @ self.components_.T).astype(np.float32)


class EmbeddingFoldPreprocessor:
    """Strictly fold-internal multimodal feature preprocessor supporting:
      - mode='balanced': block-standardized & variance-balanced dense array for Ridge/Logistic/SVR/KRR/SVC/Nystrom
      - mode='pca_gbdt': valid-masked physchem + fold-internal per-block FastPCA + conds for LightGBM/XGBoost/AFT
      - mode='pca_mlp' : standardized fold-internal per-block FastPCA + standardized physchem + conds for MLP
    """

    def __init__(
        self,
        blocks: List[Tuple[str, float]],
        phys_mode: str,
        cond_cols: List[str],
        mode: str = "balanced",
        pca_group_dims: Optional[List[Tuple[str, int]]] = None,
        phys_weight: float = 0.65,
        cond_weight: float = 0.60,
        random_state: int = 42,
    ):
        assert phys_mode in ("smiles48", "dual83")
        assert mode in ("balanced", "pca_gbdt", "pca_mlp")
        self.blocks = list(blocks)
        self.phys_mode = phys_mode
        self.cond_cols = list(cond_cols)
        self.mode = mode
        self.pca_group_dims = list(pca_group_dims) if pca_group_dims else [(b, 64) for b, _ in blocks]
        self.phys_weight = float(phys_weight)
        self.cond_weight = float(cond_weight)
        self.random_state = int(random_state)

        self.cond_enc_: Optional[ConditionEncoder] = None
        self.smi_med_: Optional[np.ndarray] = None
        self.seq_med_: Optional[np.ndarray] = None
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.block_stats_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.block_pcas_: Dict[str, Optional[FastPCA]] = {}
        self.block_pca_scales_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.n_cond_active_: int = 0
        self.out_dim_: int = 0

    def _extract_phys(self, bank: FusionBank, idx: np.ndarray, fit: bool) -> np.ndarray:
        if self.phys_mode == "smiles48":
            X = bank.physchem[idx, :48].copy()
            ok = bank.smiles_valid[idx]
            if fit:
                self.smi_med_ = np.median(X[ok], axis=0).astype(np.float32) if ok.any() else np.zeros(48, dtype=np.float32)
            if (~ok).any():
                X[~ok] = self.smi_med_
            return X

        # dual83: 48 SMILES + 35 Seq + 2 availability flags = 85D
        X_smi = bank.physchem[idx, :48].copy()
        X_seq = bank.physchem[idx, 48:83].copy()
        smi_ok = bank.smiles_valid[idx]
        seq_ok = bank.seq_valid[idx]
        if fit:
            self.smi_med_ = (
                np.median(X_smi[smi_ok], axis=0).astype(np.float32) if smi_ok.any() else np.zeros(48, dtype=np.float32)
            )
            self.seq_med_ = (
                np.median(X_seq[seq_ok], axis=0).astype(np.float32) if seq_ok.any() else np.zeros(35, dtype=np.float32)
            )
        if (~smi_ok).any():
            X_smi[~smi_ok] = self.smi_med_
        if (~seq_ok).any():
            X_seq[~seq_ok] = self.seq_med_
        flags = np.stack([smi_ok.astype(np.float32), seq_ok.astype(np.float32)], axis=1)
        return np.concatenate([X_smi, X_seq, flags], axis=1)

    def _scale_pca_dim(self, target_k: int, n_valid_tr: int, d_block: int, n_total_blocks: int) -> int:
        """Anti-(p >> n) adaptive cap on PCA components inside each training fold."""
        # Scale down cleanly on small-n folds so total PCA dims stay compact relative to n_valid_tr
        if n_valid_tr < 80:
            cap = max(4, min(16, n_valid_tr // max(2, n_total_blocks + 1)))
        elif n_valid_tr < 250:
            cap = max(6, min(24, n_valid_tr // max(3, n_total_blocks + 2)))
        elif n_valid_tr < 800:
            cap = min(target_k, max(12, n_valid_tr // max(4, n_total_blocks + 1)))
        else:
            cap = target_k
        return int(max(2, min(cap, target_k, n_valid_tr - 2, d_block - 1)))

    def fit_transform(
        self,
        bank: FusionBank,
        idx: np.ndarray,
        cond_rows: List[Dict[str, str]],
    ) -> np.ndarray:
        self.cond_enc_ = ConditionEncoder(self.cond_cols, min_count=3).fit(cond_rows)
        X_cond = self.cond_enc_.transform(cond_rows)
        self.n_cond_active_ = len(self.cond_enc_.active_cols_)

        X_phys = self._extract_phys(bank, idx, fit=True)
        self.phys_mean_ = X_phys.mean(axis=0).astype(np.float32)
        sd_p = X_phys.std(axis=0).astype(np.float32)
        sd_p[sd_p < 1e-6] = 1.0
        self.phys_std_ = sd_p

        parts: List[np.ndarray] = []

        if self.mode == "balanced":
            for bname, bw in self.blocks:
                E, v = bank.get_emb_f32(bname, idx)
                if v.any():
                    mu = E[v].mean(axis=0).astype(np.float32)
                    sd = E[v].std(axis=0).astype(np.float32)
                    sd[sd < 1e-6] = 1.0
                else:
                    mu = np.zeros(E.shape[1], dtype=np.float32)
                    sd = np.ones(E.shape[1], dtype=np.float32)
                self.block_stats_[bname] = (mu, sd)
                Z = np.zeros_like(E, dtype=np.float32)
                if v.any():
                    Z[v] = np.clip((E[v] - mu) / sd, -8.0, 8.0)
                scale = float(bw) / np.sqrt(float(E.shape[1]))
                parts.append(Z * np.float32(scale))

            Xp_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0)
            parts.append(Xp_s * np.float32(self.phys_weight / np.sqrt(float(X_phys.shape[1]))))

            if X_cond.shape[1] > 0:
                parts.append(X_cond * np.float32(self.cond_weight / np.sqrt(float(max(1, self.n_cond_active_)))))
        else:
            # pca_gbdt or pca_mlp
            if self.mode == "pca_mlp":
                Xp_out = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)
            else:
                Xp_out = X_phys.astype(np.float32)
            parts.append(Xp_out)

            n_blk = len(self.pca_group_dims)
            for j, (bname, target_k) in enumerate(self.pca_group_dims):
                E, v = bank.get_emb_f32(bname, idx)
                n_val = int(v.sum())
                if n_val >= 8:
                    mu = E[v].mean(axis=0).astype(np.float32)
                    sd = E[v].std(axis=0).astype(np.float32)
                    sd[sd < 1e-6] = 1.0
                    self.block_stats_[bname] = (mu, sd)
                    Z_v = np.clip((E[v] - mu) / sd, -8.0, 8.0)
                    k_use = self._scale_pca_dim(target_k, n_val, E.shape[1], n_blk)
                    pca = FastPCA(n_components=k_use, random_state=self.random_state + 17 * j)
                    P_v = pca.fit_transform(Z_v).astype(np.float32)
                    self.block_pcas_[bname] = pca
                    P_full = np.zeros((len(idx), k_use), dtype=np.float32)
                    P_full[v] = P_v
                    if self.mode == "pca_mlp":
                        pmu = P_full[v].mean(axis=0).astype(np.float32)
                        psd = P_full[v].std(axis=0).astype(np.float32)
                        psd[psd < 1e-6] = 1.0
                        self.block_pca_scales_[bname] = (pmu, psd)
                        P_full[v] = np.clip((P_full[v] - pmu) / psd, -8.0, 8.0)
                    parts.append(P_full)
                else:
                    self.block_stats_[bname] = (np.zeros(E.shape[1], dtype=np.float32), np.ones(E.shape[1], dtype=np.float32))
                    self.block_pcas_[bname] = None

            if X_cond.shape[1] > 0:
                parts.append(X_cond.astype(np.float32))

        X_out = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
        self.out_dim_ = X_out.shape[1]
        return X_out

    def transform(
        self,
        bank: FusionBank,
        idx: np.ndarray,
        cond_rows: Optional[List[Dict[str, str]]] = None,
        default_cond: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        assert self.cond_enc_ is not None
        if cond_rows is not None:
            X_cond = self.cond_enc_.transform(cond_rows)
        elif default_cond is not None:
            X_cond = self.cond_enc_.transform_constant(len(idx), default_cond)
        else:
            X_cond = self.cond_enc_.transform_constant(len(idx), {})

        X_phys = self._extract_phys(bank, idx, fit=False)
        parts: List[np.ndarray] = []

        if self.mode == "balanced":
            for bname, bw in self.blocks:
                E, v = bank.get_emb_f32(bname, idx)
                mu, sd = self.block_stats_[bname]
                Z = np.zeros_like(E, dtype=np.float32)
                if v.any():
                    Z[v] = np.clip((E[v] - mu) / sd, -8.0, 8.0)
                scale = float(bw) / np.sqrt(float(E.shape[1]))
                parts.append(Z * np.float32(scale))

            Xp_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0)
            parts.append(Xp_s * np.float32(self.phys_weight / np.sqrt(float(X_phys.shape[1]))))

            if X_cond.shape[1] > 0:
                parts.append(X_cond * np.float32(self.cond_weight / np.sqrt(float(max(1, self.n_cond_active_)))))
        else:
            if self.mode == "pca_mlp":
                Xp_out = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)
            else:
                Xp_out = X_phys.astype(np.float32)
            parts.append(Xp_out)

            for bname, _ in self.pca_group_dims:
                pca = self.block_pcas_.get(bname)
                if pca is None:
                    continue
                E, v = bank.get_emb_f32(bname, idx)
                mu, sd = self.block_stats_[bname]
                P_full = np.zeros((len(idx), pca.n_components_), dtype=np.float32)
                if v.any():
                    Z_v = np.clip((E[v] - mu) / sd, -8.0, 8.0)
                    P_v = pca.transform(Z_v).astype(np.float32)
                    if self.mode == "pca_mlp":
                        pmu, psd = self.block_pca_scales_[bname]
                        P_v = np.clip((P_v - pmu) / psd, -8.0, 8.0)
                    P_full[v] = P_v
                parts.append(P_full)

            if X_cond.shape[1] > 0:
                parts.append(X_cond.astype(np.float32))

        return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


class BoundedWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, est: Any, bounds: Optional[Tuple[float, float]] = None, winsor_cap: Optional[float] = None):
        self.est = est
        self.bounds = bounds
        self.winsor_cap = winsor_cap

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoundedWrapper":
        y_use = np.clip(np.asarray(y, dtype=np.float64), None, self.winsor_cap) if self.winsor_cap is not None else y
        self.est.fit(X, y_use)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(self.est.predict(X), dtype=np.float64)
        if self.bounds is not None:
            p = np.clip(p, self.bounds[0], self.bounds[1])
        return p


class ExactRBFKRR(BaseEstimator, RegressorMixin):
    """Exact Cholesky RBF Kernel Ridge Regressor (supports precomputed D2 cache for fast tuning)."""

    def __init__(self, alpha: float = 0.2, gamma_mult: float = 1.0):
        self.alpha = float(alpha)
        self.gamma_mult = float(gamma_mult)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D2_pre: Optional[np.ndarray] = None,
        base_gamma: Optional[float] = None,
    ) -> "ExactRBFKRR":
        X64 = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        self.y_mean_ = float(np.mean(y_arr))
        y_c = y_arr - self.y_mean_
        n, d = X64.shape
        D2 = D2_pre if D2_pre is not None else compute_sqdist_matrix(X64, X64)
        if base_gamma is None:
            tri = D2[np.triu_indices(n, k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
            base_gamma = 1.0 / max(med_d2, 1e-6)
        self.gamma_ = self.gamma_mult * base_gamma
        K = np.exp(-self.gamma_ * D2)
        K[np.diag_indices(n)] += self.alpha
        L, lower = cho_factor(K, lower=True, check_finite=False)
        self.dual_coef_ = cho_solve((L, lower), y_c, check_finite=False).astype(np.float64)
        self.X_tr_ = X.astype(np.float32)
        return self

    def predict(self, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
        D2 = D2_star if D2_star is not None else compute_sqdist_matrix(X, self.X_tr_)
        K_star = np.exp(-self.gamma_ * D2)
        return (K_star @ self.dual_coef_ + self.y_mean_).astype(np.float64)


class ExactRBFKLR(BaseEstimator, ClassifierMixin):
    """Exact L-BFGS RBF Kernel Logistic Regression (fast & well-calibrated on n <= 3500)."""

    def __init__(self, C: float = 2.0, gamma_mult: float = 1.0):
        self.C = float(C)
        self.gamma_mult = float(gamma_mult)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D2_pre: Optional[np.ndarray] = None,
        base_gamma: Optional[float] = None,
    ) -> "ExactRBFKLR":
        X64 = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n, d = X64.shape
        D2 = D2_pre if D2_pre is not None else compute_sqdist_matrix(X64, X64)
        if base_gamma is None:
            tri = D2[np.triu_indices(n, k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
            base_gamma = 1.0 / max(med_d2, 1e-6)
        self.gamma_ = self.gamma_mult * base_gamma
        K = np.exp(-self.gamma_ * D2)
        lam = 1.0 / max(self.C, 1e-4)

        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            alpha = params[:-1]
            b = params[-1]
            Ka = K @ alpha
            z = Ka + b
            pos_z = np.maximum(z, 0.0)
            loss_data = np.sum(pos_z - y_arr * z + np.log1p(np.exp(-np.abs(z))))
            loss = float(loss_data + 0.5 * lam * float(alpha @ Ka))
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
            diff = p - y_arr
            grad = np.empty_like(params)
            grad[:-1] = K @ (diff + lam * alpha)
            grad[-1] = float(np.sum(diff))
            return loss, grad

        x0 = np.zeros(n + 1, dtype=np.float64)
        p_mean = float(np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4))
        x0[-1] = np.log(p_mean / (1.0 - p_mean))
        res = minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 100})
        self.alpha_ = res.x[:-1].copy()
        self.bias_ = float(res.x[-1])
        self.X_tr_ = X.astype(np.float32)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
        D2 = D2_star if D2_star is not None else compute_sqdist_matrix(X, self.X_tr_)
        K_star = np.exp(-self.gamma_ * D2)
        z = np.clip(K_star @ self.alpha_ + self.bias_, -40.0, 40.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p1, p1], axis=1)


class PrecomputedRBFSVR(BaseEstimator, RegressorMixin):
    """Exact RBF SVR storing float32 support vectors for compact serialization & fast precomputed CV."""

    def __init__(self, C: float = 2.0, epsilon: float = 0.08, gamma_mult: float = 1.0):
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.gamma_mult = float(gamma_mult)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D2_pre: Optional[np.ndarray] = None,
        base_gamma: Optional[float] = None,
    ) -> "PrecomputedRBFSVR":
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        D2 = D2_pre if D2_pre is not None else compute_sqdist_matrix(X64, X64)
        if base_gamma is None:
            tri = D2[np.triu_indices(n, k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
            base_gamma = 1.0 / max(med_d2, 1e-6)
        self.gamma_ = self.gamma_mult * base_gamma
        K = np.exp(-self.gamma_ * D2)
        svr = SVR(kernel="precomputed", C=self.C, epsilon=self.epsilon)
        svr.fit(K, y)
        self.support_idx_ = svr.support_.copy()
        self.dual_coef_ = svr.dual_coef_[0].astype(np.float64).copy()
        self.intercept_ = float(svr.intercept_[0])
        self.X_sv_ = X[self.support_idx_].astype(np.float32)
        return self

    def predict(self, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
        if D2_star is not None:
            K_sv = np.exp(-self.gamma_ * D2_star[:, self.support_idx_])
        else:
            D2 = compute_sqdist_matrix(X, self.X_sv_)
            K_sv = np.exp(-self.gamma_ * D2)
        return (K_sv @ self.dual_coef_ + self.intercept_).astype(np.float64)


class PrecomputedRBFSVC(BaseEstimator, ClassifierMixin):
    """Exact RBF SVC using precomputed Gram matrix + internal Platt calibration."""

    def __init__(self, C: float = 2.0, gamma_mult: float = 1.0, random_state: int = 42):
        self.C = float(C)
        self.gamma_mult = float(gamma_mult)
        self.random_state = int(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D2_pre: Optional[np.ndarray] = None,
        base_gamma: Optional[float] = None,
    ) -> "PrecomputedRBFSVC":
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        D2 = D2_pre if D2_pre is not None else compute_sqdist_matrix(X64, X64)
        if base_gamma is None:
            tri = D2[np.triu_indices(n, k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
            base_gamma = 1.0 / max(med_d2, 1e-6)
        self.gamma_ = self.gamma_mult * base_gamma
        K = np.exp(-self.gamma_ * D2)
        self.svc_ = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.random_state)
        self.svc_.fit(K, y)
        self.X_tr_ = X.astype(np.float32)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
        D2 = D2_star if D2_star is not None else compute_sqdist_matrix(X, self.X_tr_)
        K_star = np.exp(-self.gamma_ * D2)
        return self.svc_.predict_proba(K_star)


class FastPlattRBFSVC(BaseEstimator, ClassifierMixin):
    """Exact full-rank RBF SVC using precomputed Gram + 15% internal stratified Platt calibration
    and compact support-vector-only storage (10x faster and 4x smaller than 5-fold probability=True!).
    """

    def __init__(self, C: float = 4.0, gamma_mult: float = 2.0, random_state: int = 42):
        self.C = float(C)
        self.gamma_mult = float(gamma_mult)
        self.random_state = int(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        D2_pre: Optional[np.ndarray] = None,
        base_gamma: Optional[float] = None,
    ) -> "FastPlattRBFSVC":
        from sklearn.model_selection import StratifiedShuffleSplit

        n, d = X.shape
        y_arr = np.asarray(y, dtype=np.int32)
        use_f32 = n > 7500
        D2 = D2_pre if D2_pre is not None else compute_sqdist_matrix(X, X, use_float32=use_f32)
        if base_gamma is None:
            rng = np.random.default_rng(self.random_state)
            sub = rng.choice(n, size=min(n, 800), replace=False)
            dsub = D2[np.ix_(sub, sub)]
            tri = dsub[np.triu_indices(len(sub), k=1)]
            med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, d))
            base_gamma = 1.0 / max(med_d2, 1e-6)
        self.gamma_ = float(self.gamma_mult * base_gamma)
        K = np.exp(-np.asarray(self.gamma_, dtype=D2.dtype) * D2)

        # Internal 85/15 stratified split to calibrate decision_function -> probability without leakage
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
        core_idx, cal_idx = next(sss.split(np.zeros(n), y_arr))
        svc_cal = SVC(kernel="precomputed", C=self.C, probability=False)
        svc_cal.fit(K[np.ix_(core_idx, core_idx)], y_arr[core_idx])
        dec_cal = svc_cal.decision_function(K[np.ix_(cal_idx, core_idx)])
        platt = LogisticRegression(C=10.0, solver="lbfgs", max_iter=200)
        platt.fit(dec_cal[:, None], y_arr[cal_idx])
        self.platt_a_ = float(platt.coef_[0, 0])
        self.platt_b_ = float(platt.intercept_[0])

        # Refit final SVC on 100% of training fold and store ONLY support vectors!
        svc_full = SVC(kernel="precomputed", C=self.C, probability=False)
        svc_full.fit(K, y_arr)
        self.support_idx_ = svc_full.support_.copy()
        self.dual_coef_ = svc_full.dual_coef_[0].astype(np.float64).copy()
        self.intercept_ = float(svc_full.intercept_[0])
        # Verify exact decision_function sign against sklearn (handles class ordering safely)
        chk_i = cal_idx[: min(32, len(cal_idx))]
        dec_sk = svc_full.decision_function(K[chk_i])
        dec_man = K[np.ix_(chk_i, self.support_idx_)].astype(np.float64) @ self.dual_coef_ + self.intercept_
        self.sign_ = 1.0 if np.mean(np.abs(dec_sk - dec_man)) <= np.mean(np.abs(dec_sk + dec_man)) else -1.0
        self.X_sv_ = X[self.support_idx_].astype(np.float32)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
        if D2_star is not None:
            K_sv = np.exp(-self.gamma_ * D2_star[:, self.support_idx_].astype(np.float64))
        else:
            D2 = compute_sqdist_matrix(X, self.X_sv_, use_float32=False)
            K_sv = np.exp(-self.gamma_ * D2)
        dec = self.sign_ * (K_sv @ self.dual_coef_ + self.intercept_)
        z = np.clip(self.platt_a_ * dec + self.platt_b_, -35.0, 35.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p1, p1], axis=1)


class FullEmbMLPWrapper:
    """Standardizes each feature to unit std before MLP so full-dimensional embeddings train fast."""

    def __init__(self, task_type: str, hidden: Tuple[int, ...] = (256, 64), alpha: float = 1.0, max_iter: int = 55, seed: int = 42):
        self.task_type = task_type
        self.hidden = tuple(hidden)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FullEmbMLPWrapper":
        X32 = np.asarray(X, dtype=np.float32)
        self.mu_ = X32.mean(axis=0).astype(np.float32)
        sd = X32.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        self.sd_ = sd
        Xn = np.clip((X32 - self.mu_) / self.sd_, -8.0, 8.0)
        if self.task_type == "regression":
            self.mlp_ = MLPRegressor(
                hidden_layer_sizes=self.hidden, alpha=self.alpha, batch_size=128,
                learning_rate_init=1e-3, max_iter=self.max_iter, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=7, random_state=self.seed,
            )
        else:
            self.mlp_ = MLPClassifier(
                hidden_layer_sizes=self.hidden, alpha=self.alpha, batch_size=128,
                learning_rate_init=1e-3, max_iter=self.max_iter, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=7, random_state=self.seed,
            )
        self.mlp_.fit(Xn, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = np.clip((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_, -8.0, 8.0)
        return self.mlp_.predict(Xn).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = np.clip((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_, -8.0, 8.0)
        return self.mlp_.predict_proba(Xn)


class NystromRBFModel:
    """Scalable Nystrom RBF kernel approximation + Ridge / Logistic for large endpoints (n > 3500)."""

    def __init__(
        self,
        task_type: str,
        n_components: int = 800,
        reg_param: float = 1.0,
        gamma_mult: float = 1.0,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.n_components = int(n_components)
        self.reg_param = float(reg_param)
        self.gamma_mult = float(gamma_mult)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NystromRBFModel":
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        sub = rng.choice(n, size=min(n, 600), replace=False)
        D2 = compute_sqdist_matrix(X[sub], X[sub])
        tri = D2[np.triu_indices(len(sub), k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X.shape[1]))
        gamma = self.gamma_mult / max(med_d2, 1e-6)
        n_comp = min(self.n_components, n)
        self.nys_ = Nystroem(kernel="rbf", gamma=gamma, n_components=n_comp, random_state=self.random_state)
        Z = self.nys_.fit_transform(X).astype(np.float32)
        if self.task_type == "regression":
            self.head_ = Ridge(alpha=self.reg_param)
            self.head_.fit(Z, y)
        else:
            self.head_ = LogisticRegression(C=self.reg_param, max_iter=500, solver="lbfgs")
            self.head_.fit(Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.nys_.transform(X).astype(np.float32)
        return self.head_.predict(Z).astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.nys_.transform(X).astype(np.float32)
        return self.head_.predict_proba(Z)


class XGBoostAFTRegressor:
    """Censored-aware XGBoost `survival:aft` operating natively in linear y-space (exact match to Wave-2C)."""

    def __init__(
        self,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 0.6,
        max_depth: int = 5,
        learning_rate: float = 0.04,
        n_estimators: int = 450,
        min_child_weight: float = 5.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.6,
        reg_lambda: float = 8.0,
        reg_alpha: float = 0.1,
        nthread: int = 4,
        seed: int = 42,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.aft_loss_distribution = aft_loss_distribution
        self.aft_loss_distribution_scale = float(aft_loss_distribution_scale)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.nthread = int(nthread)
        self.seed = int(seed)
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

        lo_t = np.exp(np.clip(y_lower - self.y_ref_, -20.0, 20.0)).astype(np.float32)
        hi_t = np.exp(np.clip(y_upper - self.y_ref_, -20.0, 20.0)).astype(np.float32)

        right_mask = censor_type == "right"
        left_mask = censor_type == "left"
        hi_t[right_mask] = np.inf
        lo_t[left_mask] = 0.0
        lo_t[exact_mask] = np.exp(np.clip(y_value[exact_mask] - self.y_ref_, -20.0, 20.0)).astype(np.float32)
        hi_t[exact_mask] = lo_t[exact_mask]
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
            "aft_loss_distribution_scale": self.aft_loss_distribution_scale,
            "tree_method": "hist",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "base_score": 1.0,
            "nthread": self.nthread,
            "seed": self.seed,
            "verbosity": 0,
        }
        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.booster_ is not None
        dtest = xgb.DMatrix(np.asarray(X, dtype=np.float32))
        margin = self.booster_.predict(dtest, output_margin=True).astype(np.float64)
        pred = margin + self.y_ref_
        if self.bounds is not None:
            pred = np.clip(pred, self.bounds[0], self.bounds[1])
        return pred


def make_fusion_estimator(
    family: str,
    params: Dict[str, Any],
    task_type: str,
    n_jobs: int = 4,
    bounds: Optional[Tuple[float, float]] = None,
    winsor_cap: Optional[float] = None,
    seed: int = 42,
) -> Any:
    p = dict(params)
    if task_type == "regression":
        if family == "Ridge":
            est: Any = Ridge(alpha=float(p["alpha"]))
        elif family == "KRR_RBF":
            est = ExactRBFKRR(alpha=float(p["alpha"]), gamma_mult=float(p.get("gamma_mult", 1.0)))
        elif family == "SVR_RBF":
            est = PrecomputedRBFSVR(
                C=float(p["C"]), epsilon=float(p.get("epsilon", 0.08)), gamma_mult=float(p.get("gamma_mult", 1.0))
            )
        elif family == "Nystrom_RBF":
            est = NystromRBFModel(
                task_type="regression",
                n_components=int(p.get("n_components", 800)),
                reg_param=float(p["alpha"]),
                gamma_mult=float(p.get("gamma_mult", 1.0)),
                random_state=seed,
            )
        elif family == "MLP":
            est = MLPRegressor(
                hidden_layer_sizes=tuple(p.get("hidden", (192, 64))),
                alpha=float(p.get("alpha", 0.3)),
                learning_rate_init=float(p.get("lr", 1e-3)),
                batch_size=int(p.get("batch_size", 128)),
                max_iter=int(p.get("max_iter", 60)),
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=seed,
            )
        elif family == "FullEmb_MLP":
            est = FullEmbMLPWrapper(
                task_type="regression",
                hidden=tuple(p.get("hidden", (256, 64))),
                alpha=float(p.get("alpha", 1.0)),
                max_iter=int(p.get("max_iter", 55)),
                seed=seed,
            )
        elif family == "LightGBM":
            est = lgb.LGBMRegressor(
                objective=p.get("objective", "regression"),
                n_estimators=int(p["n_estimators"]),
                learning_rate=float(p["learning_rate"]),
                max_depth=int(p["max_depth"]),
                num_leaves=int(p["num_leaves"]),
                min_child_samples=int(p["min_child_samples"]),
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=float(p.get("colsample_bytree", 0.6)),
                reg_lambda=float(p.get("reg_lambda", 8.0)),
                reg_alpha=float(p.get("reg_alpha", 0.1)),
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
            )
        elif family == "XGBoost":
            est = xgb.XGBRegressor(
                objective=p.get("objective", "reg:squarederror"),
                n_estimators=int(p["n_estimators"]),
                learning_rate=float(p["learning_rate"]),
                max_depth=int(p["max_depth"]),
                min_child_weight=float(p.get("min_child_weight", 5.0)),
                subsample=0.8,
                colsample_bytree=float(p.get("colsample_bytree", 0.6)),
                reg_lambda=float(p.get("reg_lambda", 8.0)),
                reg_alpha=float(p.get("reg_alpha", 0.1)),
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=seed,
                verbosity=0,
            )
        else:
            raise ValueError(f"Unknown regression family: {family}")

        if bounds is not None or winsor_cap is not None:
            return BoundedWrapper(est, bounds=bounds, winsor_cap=winsor_cap)
        return est

    # Classification
    if family == "Logistic":
        return LogisticRegression(C=float(p["C"]), max_iter=250, tol=1e-3, solver="lbfgs")
    if family == "KLR_RBF":
        return ExactRBFKLR(C=float(p["C"]), gamma_mult=float(p.get("gamma_mult", 1.0)))
    if family == "SVC_RBF":
        return PrecomputedRBFSVC(C=float(p["C"]), gamma_mult=float(p.get("gamma_mult", 1.0)), random_state=seed)
    if family == "FastSVC_RBF":
        return FastPlattRBFSVC(C=float(p["C"]), gamma_mult=float(p.get("gamma_mult", 2.0)), random_state=seed)
    if family == "FullEmb_MLP":
        return FullEmbMLPWrapper(
            task_type="classification",
            hidden=tuple(p.get("hidden", (256, 64))),
            alpha=float(p.get("alpha", 1.0)),
            max_iter=int(p.get("max_iter", 55)),
            seed=seed,
        )
    if family == "Nystrom_RBF":
        return NystromRBFModel(
            task_type="classification",
            n_components=int(p.get("n_components", 800)),
            reg_param=float(p["C"]),
            gamma_mult=float(p.get("gamma_mult", 1.0)),
            random_state=seed,
        )
    if family == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden", (256, 64))),
            alpha=float(p.get("alpha", 0.3)),
            learning_rate_init=float(p.get("lr", 1e-3)),
            batch_size=int(p.get("batch_size", 128)),
            max_iter=int(p.get("max_iter", 60)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
    if family == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            num_leaves=int(p["num_leaves"]),
            min_child_samples=int(p["min_child_samples"]),
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=float(p.get("colsample_bytree", 0.6)),
            reg_lambda=float(p.get("reg_lambda", 5.0)),
            reg_alpha=float(p.get("reg_alpha", 0.1)),
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
    if family == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            max_depth=int(p["max_depth"]),
            min_child_weight=float(p.get("min_child_weight", 4.0)),
            subsample=0.8,
            colsample_bytree=float(p.get("colsample_bytree", 0.6)),
            reg_lambda=float(p.get("reg_lambda", 5.0)),
            reg_alpha=float(p.get("reg_alpha", 0.1)),
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(f"Unknown classification family: {family}")


def get_rep_candidates(
    size_tier: str,
    task_type: str,
    n_cv: int,
    is_direct_fusion: bool = False,
    winsor_cap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Curated, fast, anti-(p >> n) candidate recipes for an embedding representation."""
    cands: List[Dict[str, Any]] = []

    if task_type == "regression":
        # 1. Balanced Ridge (CV-tuned L2 alpha across wide range — fast via dual/primal Gram)
        for alpha in (0.1, 0.5, 2.0, 10.0, 50.0, 200.0):
            cands.append({"mode": "balanced", "family": "Ridge", "params": {"alpha": alpha}, "winsor_cap": winsor_cap})

        # 2. Nonlinear Exact Kernel (up to n_cv <= 8000!) + FullEmb MLP on large n
        if n_cv <= 3600:
            g_mults = (0.5, 1.0, 2.0) if n_cv <= 1500 else (0.5, 1.0)
            for gm in g_mults:
                for alpha in (0.03, 0.15, 0.6, 2.5):
                    cands.append({"mode": "balanced", "family": "KRR_RBF", "params": {"alpha": alpha, "gamma_mult": gm}, "winsor_cap": winsor_cap})
            svr_cs = (0.5, 2.0, 8.0) if n_cv <= 2600 else (1.0, 4.0)
            for C in svr_cs:
                cands.append({"mode": "balanced", "family": "SVR_RBF", "params": {"C": C, "epsilon": 0.08, "gamma_mult": 1.0}, "winsor_cap": winsor_cap})
        else:
            # Large regression (hemolysis_HC50, PAMPA): exact RBF KRR + SVR + Full-Embedding MLP!
            for gm in (1.0, 2.0):
                for alpha in (0.08, 0.3):
                    cands.append({"mode": "balanced", "family": "KRR_RBF", "params": {"alpha": alpha, "gamma_mult": gm}, "winsor_cap": winsor_cap})
                for C in (1.0, 3.0):
                    cands.append({"mode": "balanced", "family": "SVR_RBF", "params": {"C": C, "epsilon": 0.06, "gamma_mult": gm}, "winsor_cap": winsor_cap})
            cands.append({"mode": "balanced", "family": "FullEmb_MLP", "params": {"hidden": (256, 64), "alpha": 1.0, "max_iter": 55}, "winsor_cap": winsor_cap})
            cands.append({"mode": "pca_mlp", "family": "MLP", "params": {"hidden": (256, 64), "alpha": 0.3, "lr": 1e-3, "max_iter": 60}, "winsor_cap": winsor_cap})

        # Extra MLP on direct fusion for medium/large endpoints
        if is_direct_fusion and 800 <= n_cv <= 3600:
            cands.append({"mode": "pca_mlp", "family": "MLP", "params": {"hidden": (192, 64), "alpha": 0.3, "lr": 1e-3, "max_iter": 80}, "winsor_cap": winsor_cap})

        # 3. Compact Fold-Internal PCA + LightGBM & XGBoost
        if size_tier == "small":
            lgb_cfgs = [
                {"n_estimators": 160, "learning_rate": 0.04, "max_depth": 3, "num_leaves": 7, "min_child_samples": 6, "reg_lambda": 10.0, "colsample_bytree": 0.6},
                {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 12, "min_child_samples": 10, "reg_lambda": 15.0, "colsample_bytree": 0.5},
            ]
            xgb_cfgs = [
                {"n_estimators": 160, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 4.0, "reg_lambda": 10.0, "colsample_bytree": 0.6},
            ]
        elif size_tier == "medium":
            lgb_cfgs = [
                {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 5, "num_leaves": 24, "min_child_samples": 15, "reg_lambda": 8.0, "colsample_bytree": 0.55},
                {"n_estimators": 550, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 12.0, "colsample_bytree": 0.45, "objective": "huber"},
            ]
            xgb_cfgs = [
                {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 6.0, "reg_lambda": 10.0, "colsample_bytree": 0.55},
            ]
        else:
            lgb_cfgs = [
                {"n_estimators": 550, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 8.0, "colsample_bytree": 0.5},
                {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 7, "num_leaves": 48, "min_child_samples": 25, "reg_lambda": 12.0, "colsample_bytree": 0.45},
            ]
            xgb_cfgs = [
                {"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.5},
            ]
        for cfg in lgb_cfgs:
            cands.append({"mode": "pca_gbdt", "family": "LightGBM", "params": cfg, "winsor_cap": winsor_cap})
        for cfg in xgb_cfgs:
            cands.append({"mode": "pca_gbdt", "family": "XGBoost", "params": cfg, "winsor_cap": winsor_cap})
        return cands

    # --- Classification ---
    log_cs = (0.1, 0.5, 2.5, 10.0) if n_cv > 3600 else (0.05, 0.2, 1.0, 5.0, 20.0)
    for C in log_cs:
        cands.append({"mode": "balanced", "family": "Logistic", "params": {"C": C}})

    if n_cv <= 3600:
        g_mults = (0.5, 1.0, 2.0) if n_cv <= 1500 else (0.5, 1.0)
        klr_cs = (0.5, 2.0, 8.0, 25.0) if n_cv <= 1800 else (0.5, 2.5, 12.0)
        for gm in g_mults:
            for C in klr_cs:
                cands.append({"mode": "balanced", "family": "KLR_RBF", "params": {"C": C, "gamma_mult": gm}})
        svc_cs = (0.5, 2.0, 8.0) if n_cv <= 1500 else (1.0, 4.0)
        for C in svc_cs:
            cands.append({"mode": "balanced", "family": "SVC_RBF", "params": {"C": C, "gamma_mult": 1.0}})
        if is_direct_fusion or n_cv >= 600:
            cands.append({"mode": "pca_mlp", "family": "MLP", "params": {"hidden": (256, 64), "alpha": 0.3, "lr": 1e-3, "max_iter": 60}})
    else:
        # Large classification (overall_peptide_toxicity, hemolysis_cls): exact FastSVC_RBF + Full-Embedding MLP!
        gms = (1.5, 3.0) if n_cv > 10000 else (1.0, 2.0, 4.0)
        cs = (2.0, 6.0) if n_cv > 10000 else (2.0, 6.0, 16.0)
        for gm in gms:
            for C in cs:
                cands.append({"mode": "balanced", "family": "FastSVC_RBF", "params": {"C": C, "gamma_mult": gm}})
        for alpha in (0.3, 1.5):
            cands.append({"mode": "balanced", "family": "FullEmb_MLP", "params": {"hidden": (256, 64), "alpha": alpha, "max_iter": 55}})
        cands.append({"mode": "pca_mlp", "family": "MLP", "params": {"hidden": (256, 64), "alpha": 0.3, "lr": 1e-3, "max_iter": 60}})

    lgb_cls_cfgs = [
        {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 5.0, "colsample_bytree": 0.55},
        {"n_estimators": 650, "learning_rate": 0.03, "max_depth": 8, "num_leaves": 48, "min_child_samples": 25, "reg_lambda": 10.0, "colsample_bytree": 0.45},
    ]
    xgb_cls_cfgs = [
        {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 5.0, "reg_lambda": 8.0, "colsample_bytree": 0.5},
    ]
    for cfg in lgb_cls_cfgs:
        cands.append({"mode": "pca_gbdt", "family": "LightGBM", "params": cfg})
    for cfg in xgb_cls_cfgs:
        cands.append({"mode": "pca_gbdt", "family": "XGBoost", "params": cfg})
    return cands


def get_multimodal_aft_candidates(size_tier: str) -> List[Dict[str, Any]]:
    """Curated XGBoost survival:aft hyperparameter grid on compact multimodal PCA + physchem features."""
    if size_tier == "small":
        return [
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.4, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 3.0, "reg_lambda": 10.0, "colsample_bytree": 0.6},
            {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.8, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 5.0, "reg_lambda": 12.0, "colsample_bytree": 0.6},
            {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 0.5, "max_depth": 3, "learning_rate": 0.04, "n_estimators": 160, "min_child_weight": 4.0, "reg_lambda": 10.0, "colsample_bytree": 0.6},
        ]
    return [
        {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 0.5, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 450, "min_child_weight": 6.0, "reg_lambda": 8.0, "colsample_bytree": 0.55},
        {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 550, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.5},
        {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 0.6, "max_depth": 5, "learning_rate": 0.04, "n_estimators": 450, "min_child_weight": 6.0, "reg_lambda": 8.0, "colsample_bytree": 0.55},
        {"aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 1.0, "max_depth": 6, "learning_rate": 0.03, "n_estimators": 550, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.5},
    ]


class ModalityAwareStacker:
    """Anti-(p >> n), modality-missingness-safe meta-stacker supporting:
      - Regression: 'best_single', 'topk_blend', 'convex_simplex' (L1/MAE + L2 shrinkage),
                    'convex_hinge' (censored hinge loss), 'ridge_nonneg', 'ridge_l2'
      - Classification: 'best_single', 'topk_blend', 'convex_simplex' (prob space),
                        'logit_convex' (non-negative simplex in logit space + bias/temp),
                        'logistic_l2' (L2 logistic on logits)

    When any base prediction is NaN on a row (e.g., SMILES-only head on a Seq-only row in Track B),
    available weights are cleanly renormalized so missing heads never distort predictions!
    """

    def __init__(
        self,
        task_type: str,
        method: str = "convex_simplex",
        params: Optional[Dict[str, Any]] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.task_type = task_type
        self.method = method
        self.params = dict(params) if params else {}
        self.bounds = bounds

        self.head_names_: List[str] = []
        self.weights_: np.ndarray = np.array([], dtype=np.float64)
        self.bias_: float = 0.0
        self.temp_: float = 1.0
        self.col_means_: np.ndarray = np.array([], dtype=np.float64)
        self.model_: Any = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        pc = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        return np.log(pc / (1.0 - pc))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    @staticmethod
    def _weighted_nan_combine(M: np.ndarray, w: np.ndarray, fallback_val: float = 0.0) -> np.ndarray:
        """Row-wise weighted average over finite entries only (renormalizing available weights)."""
        fin = np.isfinite(M)
        M0 = np.where(fin, M, 0.0)
        W = np.where(fin, w[None, :], 0.0)
        wsum = W.sum(axis=1)
        num = (M0 * W).sum(axis=1)
        out = np.divide(num, wsum, out=np.full(M.shape[0], fallback_val, dtype=np.float64), where=wsum > 1e-12)
        # If all weighted heads are NaN on some row, fall back to mean of ANY finite head
        still_missing = wsum <= 1e-12
        if still_missing.any():
            any_cnt = fin[still_missing].sum(axis=1)
            any_sum = M0[still_missing].sum(axis=1)
            out[still_missing] = np.divide(
                any_sum, any_cnt, out=np.full(int(still_missing.sum()), fallback_val), where=any_cnt > 0
            )
        return out

    def fit(
        self,
        P: np.ndarray,
        y: np.ndarray,
        head_names: List[str],
        y_lower: Optional[np.ndarray] = None,
        y_upper: Optional[np.ndarray] = None,
        censor_type: Optional[np.ndarray] = None,
    ) -> "ModalityAwareStacker":
        P = np.asarray(P, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, m = P.shape
        self.head_names_ = list(head_names)
        self.fallback_val_ = float(np.nanmean(y)) if len(y) > 0 else 0.0

        # Column means for fallback imputation in linear/logistic meta-models
        self.col_means_ = np.array(
            [float(np.nanmean(P[:, j])) if np.isfinite(P[:, j]).any() else self.fallback_val_ for j in range(m)],
            dtype=np.float64,
        )

        # Compute individual head scores on rows where each head is finite
        head_scores = np.full(m, -1e9, dtype=np.float64)
        for j in range(m):
            ok = np.isfinite(P[:, j]) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            if self.task_type == "regression":
                if self.method == "convex_hinge" and censor_type is not None and y_lower is not None and y_upper is not None:
                    ct = censor_type[ok]
                    yp = P[ok, j]
                    yt = y[ok]
                    yl = y_lower[ok]
                    yu = y_upper[ok]
                    errs = np.where(
                        ct == "exact", np.abs(yt - yp),
                        np.where(ct == "right", np.maximum(0.0, yl - yp),
                        np.where(ct == "left", np.maximum(0.0, yp - yu),
                        np.maximum(0.0, np.maximum(yl - yp, yp - yu))))
                    )
                    head_scores[j] = -float(np.mean(errs))
                else:
                    head_scores[j] = -float(np.mean(np.abs(y[ok] - P[ok, j])))
            else:
                yt = y[ok]
                if len(np.unique(yt)) < 2:
                    continue
                pc = np.clip(P[ok, j], 1e-6, 1.0 - 1e-6)
                ll = -float(np.mean(yt * np.log(pc) + (1.0 - yt) * np.log(1.0 - pc)))
                head_scores[j] = -ll

        if self.method == "best_single":
            best_j = int(np.argmax(head_scores))
            w = np.zeros(m, dtype=np.float64)
            w[best_j] = 1.0
            self.weights_ = w
            return self

        if self.method == "topk_blend":
            k = min(int(self.params.get("k", 3)), m)
            order = np.argsort(-head_scores)
            w = np.zeros(m, dtype=np.float64)
            chosen = [j for j in order[:k] if head_scores[j] > -1e8]
            if not chosen:
                chosen = [0]
            for j in chosen:
                w[j] = 1.0 / len(chosen)
            self.weights_ = w
            return self

        if self.method in ("convex_simplex", "convex_hinge"):
            reg = float(self.params.get("reg", 0.01))
            unif = np.ones(m, dtype=np.float64) / float(m)

            if self.task_type == "regression":
                eps_smooth = 1e-3

                def obj_reg(logits_w: np.ndarray) -> float:
                    lw = logits_w - np.max(logits_w)
                    w = np.exp(lw)
                    w = w / np.sum(w)
                    pred = self._weighted_nan_combine(P, w, self.fallback_val_)
                    if self.method == "convex_hinge" and censor_type is not None and y_lower is not None and y_upper is not None:
                        ct = censor_type
                        e_ex = np.sqrt((y - pred) ** 2 + eps_smooth ** 2) - eps_smooth
                        e_r = np.maximum(0.0, y_lower - pred)
                        e_l = np.maximum(0.0, pred - y_upper)
                        e_i = np.maximum(0.0, np.maximum(y_lower - pred, pred - y_upper))
                        loss = float(np.mean(np.where(ct == "exact", e_ex, np.where(ct == "right", e_r, np.where(ct == "left", e_l, e_i)))))
                    else:
                        loss = float(np.mean(np.sqrt((y - pred) ** 2 + eps_smooth ** 2)))
                    return loss + reg * float(np.sum((w - unif) ** 2))

                # Multi-start: uniform + top-1 biased
                best_j = int(np.argmax(head_scores))
                init_top = np.full(m, -1.5, dtype=np.float64)
                init_top[best_j] = 1.5
                best_res = None
                for x0 in (np.zeros(m, dtype=np.float64), init_top):
                    res = minimize(obj_reg, x0, method="BFGS", options={"maxiter": 120})
                    if best_res is None or res.fun < best_res.fun:
                        best_res = res
                lw = best_res.x - np.max(best_res.x)
                w = np.exp(lw)
                w = w / np.sum(w)
                w[w < 1e-4] = 0.0
                if w.sum() > 0:
                    w = w / w.sum()
                self.weights_ = w
                return self

            # Classification probability-space convex simplex minimizing log-loss
            def obj_cls_prob(logits_w: np.ndarray) -> float:
                lw = logits_w - np.max(logits_w)
                w = np.exp(lw)
                w = w / np.sum(w)
                p = np.clip(self._weighted_nan_combine(P, w, self.fallback_val_), 1e-6, 1.0 - 1e-6)
                ll = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
                return ll + reg * float(np.sum((w - unif) ** 2))

            best_j = int(np.argmax(head_scores))
            init_top = np.full(m, -1.5, dtype=np.float64)
            init_top[best_j] = 1.5
            best_res = None
            for x0 in (np.zeros(m, dtype=np.float64), init_top):
                res = minimize(obj_cls_prob, x0, method="BFGS", options={"maxiter": 120})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            lw = best_res.x - np.max(best_res.x)
            w = np.exp(lw)
            w = w / np.sum(w)
            w[w < 1e-4] = 0.0
            if w.sum() > 0:
                w = w / w.sum()
            self.weights_ = w
            return self

        if self.method == "logit_convex":
            assert self.task_type == "classification"
            reg = float(self.params.get("reg", 0.02))
            unif = np.ones(m, dtype=np.float64) / float(m)
            Z = np.where(np.isfinite(P), self._logit(np.where(np.isfinite(P), P, 0.5)), np.nan)
            fb_z = float(self._logit(np.array([np.clip(self.fallback_val_, 1e-4, 1.0 - 1e-4)]))[0])

            def obj_logit(theta: np.ndarray) -> float:
                lw = theta[:m]
                lw = lw - np.max(lw)
                w = np.exp(lw)
                w = w / np.sum(w)
                temp = float(np.exp(np.clip(theta[m], -2.0, 2.0)))
                bias = float(theta[m + 1])
                z_comb = temp * self._weighted_nan_combine(Z, w, fb_z) + bias
                pos_z = np.maximum(z_comb, 0.0)
                ll = float(np.mean(pos_z - y * z_comb + np.log1p(np.exp(-np.abs(z_comb)))))
                return ll + reg * float(np.sum((w - unif) ** 2)) + 0.005 * float(theta[m] ** 2 + bias ** 2)

            best_j = int(np.argmax(head_scores))
            x0_unif = np.zeros(m + 2, dtype=np.float64)
            x0_top = np.zeros(m + 2, dtype=np.float64)
            x0_top[:m] = -1.5
            x0_top[best_j] = 1.5
            best_res = None
            for x0 in (x0_unif, x0_top):
                res = minimize(obj_logit, x0, method="BFGS", options={"maxiter": 140})
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
            lw = best_res.x[:m] - np.max(best_res.x[:m])
            w = np.exp(lw)
            w = w / np.sum(w)
            w[w < 1e-4] = 0.0
            if w.sum() > 0:
                w = w / w.sum()
            self.weights_ = w
            self.temp_ = float(np.exp(np.clip(best_res.x[m], -2.0, 2.0)))
            self.bias_ = float(best_res.x[m + 1])
            return self

        if self.method in ("ridge_nonneg", "ridge_l2"):
            assert self.task_type == "regression"
            P_imp = np.where(np.isfinite(P), P, self.col_means_[None, :])
            alpha = float(self.params.get("alpha", 10.0))
            pos = self.method == "ridge_nonneg"
            reg_m = Ridge(alpha=alpha, positive=pos)
            reg_m.fit(P_imp, y)
            self.model_ = reg_m
            coef = np.asarray(reg_m.coef_, dtype=np.float64)
            s = float(np.sum(np.abs(coef)))
            self.weights_ = coef / s if s > 1e-12 else np.ones(m) / m
            return self

        if self.method == "logistic_l2":
            assert self.task_type == "classification"
            P_imp = np.where(np.isfinite(P), P, self.col_means_[None, :])
            Z = self._logit(P_imp)
            C = float(self.params.get("C", 0.1))
            lr = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
            lr.fit(Z, y.astype(np.int32))
            self.model_ = lr
            coef = np.asarray(lr.coef_[0], dtype=np.float64)
            s = float(np.sum(np.abs(coef)))
            self.weights_ = coef / s if s > 1e-12 else np.ones(m) / m
            return self

        raise ValueError(f"Unknown stacker method: {self.method}")

    def predict(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64)
        if self.method in ("best_single", "topk_blend", "convex_simplex", "convex_hinge"):
            out = self._weighted_nan_combine(P, self.weights_, self.fallback_val_)
        elif self.method == "logit_convex":
            Z = np.where(np.isfinite(P), self._logit(np.where(np.isfinite(P), P, 0.5)), np.nan)
            fb_z = float(self._logit(np.array([np.clip(self.fallback_val_, 1e-4, 1.0 - 1e-4)]))[0])
            z_comb = self.temp_ * self._weighted_nan_combine(Z, self.weights_, fb_z) + self.bias_
            out = self._sigmoid(z_comb)
        elif self.method in ("ridge_nonneg", "ridge_l2"):
            P_imp = np.where(np.isfinite(P), P, self.col_means_[None, :])
            out = self.model_.predict(P_imp).astype(np.float64)
        elif self.method == "logistic_l2":
            P_imp = np.where(np.isfinite(P), P, self.col_means_[None, :])
            Z = self._logit(P_imp)
            out = self.model_.predict_proba(Z)[:, 1].astype(np.float64)
        else:
            raise ValueError(f"Unknown stacker method: {self.method}")

        if self.task_type == "classification":
            return np.clip(out, 1e-6, 1.0 - 1e-6)
        if self.bounds is not None:
            return np.clip(out, self.bounds[0], self.bounds[1])
        return out

    def weight_dict(self) -> Dict[str, float]:
        return {name: round(float(w), 5) for name, w in zip(self.head_names_, self.weights_)}

    def active_heads(self, tol: float = 1e-4) -> List[str]:
        if self.method in ("ridge_nonneg", "ridge_l2", "logistic_l2"):
            # All heads participate in linear meta-learner unless coef ~ 0
            return [n for n, w in zip(self.head_names_, self.weights_) if abs(float(w)) > tol]
        return [n for n, w in zip(self.head_names_, self.weights_) if float(w) > tol]
