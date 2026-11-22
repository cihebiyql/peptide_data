#!/usr/bin/env python3
"""Plan B-1: 100% node2-portable Small-N Foundation/Shrinkage Models & Conformal Calibration Classes.

Strictly depends ONLY on: numpy, scipy, sklearn, lightgbm (ZERO catboost, ZERO tabpfn, ZERO torch).
Compatible with BOTH offline UnifiedSmallNBank (50,995 compounds) and live v33_live_encoders.LiveMiniBank!
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

AD_IN_DOMAIN_THR = 0.65
AD_BORDERLINE_THR = 0.40

FM_SMILES_BLOCKS = [
    "ncaa_smiles_repr",
    "ncaa_fusion_hf",
    "ncaa_protein_proj",
    "peptideclm2",
    "chemberta_mtr",
]

FM_DUAL_EXTRA_GROUPS = [
    ("esmc_core", ("esmc_mean", "esmc_max")),
    ("esmc_term", ("esmc_nterm5", "esmc_cterm5")),
]


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A64 = np.asarray(A, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    na = np.sum(A64 * A64, axis=1)[:, None]
    nb = np.sum(B64 * B64, axis=1)[None, :]
    return np.maximum(na + nb - 2.0 * (A64 @ B64.T), 0.0)


def _tanimoto_kernel(A_bits: np.ndarray, B_bits: np.ndarray) -> np.ndarray:
    """Exact strictly positive-definite Tanimoto / Jaccard kernel on binary Morgan bits."""
    A = np.asarray(A_bits, dtype=np.float64)
    B = np.asarray(B_bits, dtype=np.float64)
    pa = np.sum(A, axis=1)[:, None]
    pb = np.sum(B, axis=1)[None, :]
    inter = A @ B.T
    union = pa + pb - inter
    K = np.divide(inter, union, out=np.zeros_like(inter), where=union > 1e-12)
    return np.clip(K, 0.0, 1.0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return 0.0
    k = int(math.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def _finite_sample_cluster_quantile(
    scores: np.ndarray, alpha: float, n_eff: Optional[int] = None
) -> float:
    """Exact finite-sample quantile using effective sample size n_eff (cluster-aware safeguard when n_eff < n)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return 0.0
    if n_eff is None or n_eff >= n:
        k = int(math.ceil((1.0 - alpha) * (n + 1)))
    else:
        m = max(2, int(n_eff))
        # Exact finite-sample level at effective independent cluster count m: ceil((m+1)(1-alpha))/m
        q_lvl = min(1.0, math.ceil((1.0 - alpha) * (m + 1)) / float(m))
        k = int(math.ceil(q_lvl * n))
    k = min(max(k, 1), n)
    return float(s[k - 1])


class SmallNConditionEncoder:
    """Self-contained fold-internal categorical encoder for `cond_*` columns (node2-portable)."""

    def __init__(self, cond_cols: Sequence[str], min_count: int = 3):
        self.cond_cols = list(cond_cols)
        self.min_count = int(min_count)
        self.active_cols_: List[str] = []
        self.categories_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []

    @staticmethod
    def _clean(val: Any) -> str:
        if val is None:
            return "unspecified"
        s = str(val).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return "unspecified"
        return s

    def fit(self, cond_rows: Sequence[Dict[str, Any]]) -> "SmallNConditionEncoder":
        self.active_cols_ = []
        self.categories_ = {}
        self.feature_names_ = []
        if not self.cond_cols or len(cond_rows) == 0:
            return self
        for col in self.cond_cols:
            vals = [self._clean(r.get(col, "unspecified")) for r in cond_rows]
            counts: Dict[str, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            collapsed = [v if counts[v] >= self.min_count else "other" for v in vals]
            uniq = sorted(set(collapsed))
            if len(uniq) <= 1:
                continue
            self.active_cols_.append(col)
            self.categories_[col] = uniq
            for cat in uniq:
                self.feature_names_.append(f"{col}={cat}")
        return self

    def transform(self, cond_rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        n = len(cond_rows)
        d = len(self.feature_names_)
        if d == 0:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.zeros((n, d), dtype=np.float32)
        offset = 0
        for col in self.active_cols_:
            cats = self.categories_[col]
            cat_to_j = {c: j for j, c in enumerate(cats)}
            has_other = "other" in cat_to_j
            other_j = cat_to_j.get("other", -1)
            for i, r in enumerate(cond_rows):
                v = self._clean(r.get(col, "unspecified"))
                j = cat_to_j.get(v, other_j if has_other else -1)
                if j >= 0:
                    out[i, offset + j] = 1.0
            offset += len(cats)
        return out

    def transform_constant(self, n: int, default_cond: Dict[str, Any]) -> np.ndarray:
        if len(self.feature_names_) == 0:
            return np.zeros((n, 0), dtype=np.float32)
        row = self.transform([default_cond])[0]
        return np.repeat(row[None, :], n, axis=0)


class FastDualPCA:
    """Exact dual PCA for small-n high-dim embeddings (pure numpy/scipy, node2-portable)."""

    def __init__(self, n_components: int):
        self.n_components = int(n_components)
        self.n_components_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X64 = np.asarray(X, dtype=np.float64)
        n, d = X64.shape
        k = max(1, min(self.n_components, n - 1, d))
        self.n_components_ = k
        mu = X64.mean(axis=0)
        self.mean_ = mu.astype(np.float32)
        Xc = X64 - mu
        if n <= d:
            G = Xc @ Xc.T
            evals, U = eigh(G, subset_by_index=[max(0, n - k), n - 1])
            order = np.argsort(evals)[::-1]
            evals = np.maximum(evals[order], 1e-12)
            U = U[:, order]
            s = np.sqrt(evals)
            Vt = (U.T @ Xc) / s[:, None]
            self.components_ = Vt.astype(np.float32)
            return (U * s[None, :]).astype(np.float32)
        C = Xc.T @ Xc
        evals, V = eigh(C, subset_by_index=[max(0, d - k), d - 1])
        order = np.argsort(evals)[::-1]
        V = V[:, order]
        self.components_ = V.T.astype(np.float32)
        return (Xc @ V).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.components_ is not None
        Xc = np.asarray(X, dtype=np.float32) - self.mean_
        return (Xc @ self.components_.T).astype(np.float32)


class MultiBlockKernelPreprocessor:
    """Strictly fold-internal multi-block preprocessor for zero-PCA-loss Composite GPR & Kernel models.

    Produces a structured dict of centered L2-normalized FM blocks, valid masks, binary Morgan bits,
    standardized physchem, and scaled condition one-hot vectors.
    Compatible with BOTH UnifiedSmallNBank and v33_live_encoders.LiveMiniBank.
    """

    def __init__(self, track: str, cond_cols: Sequence[str]):
        assert track in ("dual_modality", "smiles_only")
        self.track = track
        self.cond_cols = list(cond_cols)
        self.cond_enc_: Optional[SmallNConditionEncoder] = None
        self.n_cond_active_: int = 0
        self.smi_med_: Optional[np.ndarray] = None
        self.seq_med_: Optional[np.ndarray] = None
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.fm_means_: Dict[str, np.ndarray] = {}
        self.fm_names_: List[str] = []

    def _extract_phys(self, bank: Any, idx: np.ndarray, fit: bool) -> np.ndarray:
        if self.track == "smiles_only":
            X = bank.physchem[idx, :48].astype(np.float32).copy()
            ok = bank.smiles_valid[idx]
            if fit:
                self.smi_med_ = np.median(X[ok], axis=0).astype(np.float32) if ok.any() else np.zeros(48, dtype=np.float32)
            if (~ok).any():
                X[~ok] = self.smi_med_
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_smi = bank.physchem[idx, :48].astype(np.float32).copy()
        X_seq = bank.physchem[idx, 48:83].astype(np.float32).copy()
        smi_ok = bank.smiles_valid[idx]
        seq_ok = bank.seq_valid[idx]
        if fit:
            self.smi_med_ = np.median(X_smi[smi_ok], axis=0).astype(np.float32) if smi_ok.any() else np.zeros(48, dtype=np.float32)
            self.seq_med_ = np.median(X_seq[seq_ok], axis=0).astype(np.float32) if seq_ok.any() else np.zeros(35, dtype=np.float32)
        if (~smi_ok).any():
            X_smi[~smi_ok] = self.smi_med_
        if (~seq_ok).any():
            X_seq[~seq_ok] = self.seq_med_
        flags = np.stack([smi_ok.astype(np.float32), seq_ok.astype(np.float32)], axis=1)
        out = np.concatenate([X_smi, X_seq, flags], axis=1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _center_l2(self, E: np.ndarray, valid: np.ndarray, name: str, fit: bool) -> Tuple[np.ndarray, np.ndarray]:
        E32 = np.nan_to_num(np.asarray(E, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        v = np.asarray(valid, dtype=bool)
        if fit:
            mu = E32[v].mean(axis=0, dtype=np.float64).astype(np.float32) if v.any() else np.zeros(E32.shape[1], dtype=np.float32)
            self.fm_means_[name] = mu
        else:
            mu = self.fm_means_[name]
        U = np.zeros_like(E32, dtype=np.float32)
        if v.any():
            C = E32[v] - mu
            norms = np.linalg.norm(C, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            U[v] = C / norms
        return U, v

    def fit_transform(self, bank: Any, idx: np.ndarray, cond_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        self.cond_enc_ = SmallNConditionEncoder(self.cond_cols, min_count=3).fit(cond_rows)
        X_cond = self.cond_enc_.transform(cond_rows)
        self.n_cond_active_ = len(self.cond_enc_.active_cols_)
        if X_cond.shape[1] > 0:
            X_cond = X_cond / np.float32(math.sqrt(max(1, self.n_cond_active_)))

        X_phys = self._extract_phys(bank, idx, fit=True)
        self.phys_mean_ = X_phys.mean(axis=0).astype(np.float32)
        sd = X_phys.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        self.phys_std_ = sd
        X_phys_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)

        X_bits = bank.morgan_bits[idx].astype(np.float32).copy()
        smi_ok = bank.smiles_valid[idx].copy()
        if (~smi_ok).any():
            X_bits[~smi_ok] = 0.0

        fm_vecs: Dict[str, np.ndarray] = {}
        fm_valid: Dict[str, np.ndarray] = {}
        self.fm_names_ = []

        for bname in FM_SMILES_BLOCKS:
            E, v = bank.get_emb_f32(bname, idx)
            U, vm = self._center_l2(E, v, bname, fit=True)
            fm_vecs[bname] = U
            fm_valid[bname] = vm
            self.fm_names_.append(bname)

        if self.track == "dual_modality":
            for gname, sub_blocks in FM_DUAL_EXTRA_GROUPS:
                sub_us = []
                vm_comb = None
                for sb in sub_blocks:
                    E, v = bank.get_emb_f32(sb, idx)
                    U, vm = self._center_l2(E, v, sb, fit=True)
                    sub_us.append(U)
                    vm_comb = vm if vm_comb is None else (vm_comb & vm)
                G = np.concatenate(sub_us, axis=1) / np.float32(math.sqrt(len(sub_us)))
                if vm_comb is not None and (~vm_comb).any():
                    G[~vm_comb] = 0.0
                fm_vecs[gname] = G.astype(np.float32)
                fm_valid[gname] = vm_comb if vm_comb is not None else np.zeros(len(idx), dtype=bool)
                self.fm_names_.append(gname)

        return {
            "fm_vecs": fm_vecs,
            "fm_valid": fm_valid,
            "fm_names": list(self.fm_names_),
            "morgan_bits": X_bits,
            "smiles_valid": smi_ok,
            "physchem": X_phys_s,
            "cond": X_cond.astype(np.float32),
        }

    def transform(
        self,
        bank: Any,
        idx: np.ndarray,
        cond_rows: Optional[Sequence[Dict[str, Any]]] = None,
        default_cond: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self.cond_enc_ is not None
        if cond_rows is not None:
            X_cond = self.cond_enc_.transform(cond_rows)
        elif default_cond is not None:
            X_cond = self.cond_enc_.transform_constant(len(idx), default_cond)
        else:
            X_cond = self.cond_enc_.transform_constant(len(idx), {})
        if X_cond.shape[1] > 0:
            X_cond = X_cond / np.float32(math.sqrt(max(1, self.n_cond_active_)))

        X_phys = self._extract_phys(bank, idx, fit=False)
        X_phys_s = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)

        X_bits = bank.morgan_bits[idx].astype(np.float32).copy()
        smi_ok = bank.smiles_valid[idx].copy()
        if (~smi_ok).any():
            X_bits[~smi_ok] = 0.0

        fm_vecs: Dict[str, np.ndarray] = {}
        fm_valid: Dict[str, np.ndarray] = {}
        for bname in FM_SMILES_BLOCKS:
            E, v = bank.get_emb_f32(bname, idx)
            U, vm = self._center_l2(E, v, bname, fit=False)
            fm_vecs[bname] = U
            fm_valid[bname] = vm

        if self.track == "dual_modality":
            for gname, sub_blocks in FM_DUAL_EXTRA_GROUPS:
                sub_us = []
                vm_comb = None
                for sb in sub_blocks:
                    E, v = bank.get_emb_f32(sb, idx)
                    U, vm = self._center_l2(E, v, sb, fit=False)
                    sub_us.append(U)
                    vm_comb = vm if vm_comb is None else (vm_comb & vm)
                G = np.concatenate(sub_us, axis=1) / np.float32(math.sqrt(len(sub_us)))
                if vm_comb is not None and (~vm_comb).any():
                    G[~vm_comb] = 0.0
                fm_vecs[gname] = G.astype(np.float32)
                fm_valid[gname] = vm_comb if vm_comb is not None else np.zeros(len(idx), dtype=bool)

        return {
            "fm_vecs": fm_vecs,
            "fm_valid": fm_valid,
            "fm_names": list(self.fm_names_),
            "morgan_bits": X_bits,
            "smiles_valid": smi_ok,
            "physchem": X_phys_s,
            "cond": X_cond.astype(np.float32),
        }


def compute_multiblock_grams(
    X_a: Dict[str, Any],
    X_b: Dict[str, Any],
    fm_kernel: str = "cosine",
    exp_gamma: float = 1.25,
) -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    """Compute per-block Gram matrices between X_a and X_b plus squared physchem distance D2_phys."""
    names: List[str] = []
    grams: Dict[str, np.ndarray] = {}

    for b in X_a["fm_names"]:
        Ua = X_a["fm_vecs"][b].astype(np.float64)
        Ub = X_b["fm_vecs"][b].astype(np.float64)
        va = X_a["fm_valid"][b].astype(np.float64)[:, None]
        vb = X_b["fm_valid"][b].astype(np.float64)[None, :]
        vmask = va * vb
        cos = np.clip(Ua @ Ub.T, -1.0, 1.0)
        if fm_kernel == "cosine":
            # Strictly PSD centered cosine kernel (shifted positive or pure cosine; pure cosine is strictly PSD!)
            K = cos * vmask
        elif fm_kernel == "exp_cosine":
            # Strictly PSD Gaussian/Exponential kernel on the unit hypersphere: exp(-gamma * ||u-v||^2 / 2) = exp(-gamma*(1-cos))
            K = np.exp(-exp_gamma * np.maximum(1.0 - cos, 0.0)) * vmask
        elif fm_kernel == "matern32_cosine":
            r = np.sqrt(np.maximum(2.0 * (1.0 - cos), 0.0)) * math.sqrt(3.0) * math.sqrt(exp_gamma)
            K = (1.0 + r) * np.exp(-r) * vmask
        else:
            raise ValueError(f"Unknown fm_kernel: {fm_kernel}")
        grams[b] = K
        names.append(b)

    # Exact Morgan Tanimoto kernel (strictly PSD)
    K_tan = _tanimoto_kernel(X_a["morgan_bits"], X_b["morgan_bits"])
    grams["morgan_tanimoto"] = K_tan
    names.append("morgan_tanimoto")

    # Physchem D2 (used for RBF kernel with learnable lengthscale)
    D2_phys = _sqdist(X_a["physchem"], X_b["physchem"])

    # Condition linear kernel (if active)
    if X_a["cond"].shape[1] > 0:
        Ca = X_a["cond"].astype(np.float64)
        Cb = X_b["cond"].astype(np.float64)
        grams["cond"] = Ca @ Cb.T
        names.append("cond")

    return names, grams, D2_phys


class MultiBlockCompositeGPR(BaseEstimator, RegressorMixin):
    """Exact MAP Marginal-Likelihood Multi-Block Cosine + Tanimoto + RBF Composite GPR.

    Zero PCA information loss across all 4 foundation models + Morgan Tanimoto + Physchem RBF + Conditions.
    Exact analytical gradients -> converges in <40ms per fold via L-BFGS-B.
    Returns exact Bayesian posterior mean AND exact predictive standard deviation sigma(x)!
    """

    def __init__(
        self,
        fm_kernel: str = "cosine",
        exp_gamma: float = 1.25,
        prior_var_reg: float = 0.15,
        prior_noise_reg: float = 0.10,
        min_noise_var: float = 0.04,
        bounds: Optional[Tuple[float, float]] = None,
        winsor_cap: Optional[float] = None,
    ):
        self.fm_kernel = fm_kernel
        self.exp_gamma = float(exp_gamma)
        self.prior_var_reg = float(prior_var_reg)
        self.prior_noise_reg = float(prior_noise_reg)
        self.min_noise_var = float(min_noise_var)
        self.bounds = bounds
        self.winsor_cap = winsor_cap

    def fit(self, X_dict: Dict[str, Any], y: np.ndarray) -> "MultiBlockCompositeGPR":
        y_arr = np.asarray(y, dtype=np.float64)
        if self.winsor_cap is not None:
            y_arr = np.clip(y_arr, None, self.winsor_cap)
        n = len(y_arr)
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr))
        if self.y_std_ < 1e-6:
            self.y_std_ = 1.0
        y_s = (y_arr - self.y_mean_) / self.y_std_

        fixed_names, grams, D2_phys = compute_multiblock_grams(
            X_dict, X_dict, fm_kernel=self.fm_kernel, exp_gamma=self.exp_gamma
        )
        tri = D2_phys[np.triu_indices(n, k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_dict["physchem"].shape[1]))
        self.phys_d2_scale_ = max(med_d2, 1e-4)
        D2_norm = D2_phys / self.phys_d2_scale_

        K_list = [grams[nm] for nm in fixed_names]
        n_fixed = len(K_list)
        # Parameter vector: [log_s2_fixed (n_fixed), log_s2_phys, log_ell_phys, log_noise]
        n_var = n_fixed + 1
        mu_var = math.log(0.8 / max(1, n_var))
        mu_noise = math.log(0.25)
        min_log_n = math.log(max(self.min_noise_var, 1e-3))

        opt_bounds = (
            [(-7.5, 2.5)] * n_var
            + [(-1.5, 2.5)]
            + [(min_log_n, 1.8)]
        )
        eye_n = np.eye(n, dtype=np.float64)

        def nlml_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            log_vars = theta[:n_var]
            log_ell = theta[n_var]
            log_sn = theta[n_var + 1]

            vars_ = np.exp(log_vars)
            ell2 = math.exp(2.0 * log_ell)
            sn2 = math.exp(log_sn)

            K_phys = np.exp(-0.5 * D2_norm / ell2)
            K = sn2 * eye_n
            for j in range(n_fixed):
                K = K + vars_[j] * K_list[j]
            K = K + vars_[n_fixed] * K_phys

            try:
                L, lower = cho_factor(K, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e12, np.zeros_like(theta)

            alpha = cho_solve((L, lower), y_s, check_finite=False)
            logdet = 2.0 * float(np.sum(np.log(np.maximum(np.diag(L), 1e-12))))
            nlml = 0.5 * float(y_s @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2.0 * math.pi)

            # MAP hyperprior (keeps small-n variance estimates well-regularized)
            prior = (
                0.5 * self.prior_var_reg * float(np.sum((log_vars - mu_var) ** 2))
                + 0.5 * 0.25 * (log_ell ** 2)
                + 0.5 * self.prior_noise_reg * ((log_sn - mu_noise) ** 2)
            )

            K_inv = cho_solve((L, lower), eye_n, check_finite=False)
            W = K_inv - np.outer(alpha, alpha)

            grad = np.zeros_like(theta)
            for j in range(n_fixed):
                grad[j] = 0.5 * vars_[j] * float(np.sum(W * K_list[j])) + self.prior_var_reg * (log_vars[j] - mu_var)
            grad[n_fixed] = 0.5 * vars_[n_fixed] * float(np.sum(W * K_phys)) + self.prior_var_reg * (log_vars[n_fixed] - mu_var)
            dK_dell = vars_[n_fixed] * K_phys * (D2_norm / ell2)
            grad[n_var] = 0.5 * float(np.sum(W * dK_dell)) + 0.25 * log_ell
            grad[n_var + 1] = 0.5 * sn2 * float(np.trace(W)) + self.prior_noise_reg * (log_sn - mu_noise)
            return nlml + prior, grad

        # 3 diverse initializations: balanced, FM-heavy, Chem/Physchem-heavy
        inits = [
            np.array([mu_var] * n_var + [0.0, mu_noise], dtype=np.float64),
        ]
        # FM-heavy init
        v_fm = np.full(n_var, mu_var - 0.8, dtype=np.float64)
        for j, nm in enumerate(fixed_names):
            if nm != "morgan_tanimoto" and nm != "cond":
                v_fm[j] = mu_var + 0.4
        inits.append(np.concatenate([v_fm, [0.2, math.log(0.18)]]))
        # Chem/Physchem-heavy init
        v_ch = np.full(n_var, mu_var - 0.8, dtype=np.float64)
        for j, nm in enumerate(fixed_names):
            if nm in ("morgan_tanimoto", "cond"):
                v_ch[j] = mu_var + 0.5
        v_ch[n_fixed] = mu_var + 0.6
        inits.append(np.concatenate([v_ch, [-0.2, math.log(0.22)]]))

        best_res = None
        for x0 in inits:
            res = minimize(
                nlml_and_grad, x0, jac=True, method="L-BFGS-B",
                bounds=opt_bounds, options={"maxiter": 120, "ftol": 1e-7},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        theta_opt = best_res.x
        self.fixed_names_ = fixed_names
        self.block_names_ = fixed_names + ["physchem_rbf"]
        self.block_vars_ = np.exp(theta_opt[:n_var]).astype(np.float64)
        self.ell_phys_ = float(math.exp(theta_opt[n_var]))
        self.noise_var_ = float(math.exp(theta_opt[n_var + 1]))
        self.nlml_ = float(best_res.fun)

        # Final Cholesky & alpha
        K_phys = np.exp(-0.5 * D2_norm / (self.ell_phys_ ** 2))
        K = (self.noise_var_ + 1e-6) * eye_n
        for j in range(n_fixed):
            K = K + self.block_vars_[j] * K_list[j]
        K = K + self.block_vars_[n_fixed] * K_phys
        L, _ = cho_factor(K, lower=True, check_finite=False)
        self.L_ = L.astype(np.float64)
        self.alpha_ = cho_solve((L, True), y_s, check_finite=False).astype(np.float64)

        # Store compact training state for exact test-time kernel evaluation (float32 for tiny joblib!)
        self.X_tr_ = {
            "fm_vecs": {k: v.astype(np.float32) for k, v in X_dict["fm_vecs"].items()},
            "fm_valid": {k: v.astype(bool) for k, v in X_dict["fm_valid"].items()},
            "fm_names": list(X_dict["fm_names"]),
            "morgan_bits": X_dict["morgan_bits"].astype(np.uint8),
            "smiles_valid": X_dict["smiles_valid"].astype(bool),
            "physchem": X_dict["physchem"].astype(np.float32),
            "cond": X_dict["cond"].astype(np.float32),
        }
        return self

    def predict(
        self, X_dict: Dict[str, Any], return_std: bool = False
    ) -> Any:
        fixed_names, grams, D2_phys = compute_multiblock_grams(
            X_dict, self.X_tr_, fm_kernel=self.fm_kernel, exp_gamma=self.exp_gamma
        )
        n_q = len(X_dict["physchem"])
        n_fixed = len(self.fixed_names_)
        K_star = np.zeros((n_q, len(self.alpha_)), dtype=np.float64)
        for j, nm in enumerate(self.fixed_names_):
            K_star += self.block_vars_[j] * grams[nm]
        K_phys_star = np.exp(-0.5 * (D2_phys / self.phys_d2_scale_) / (self.ell_phys_ ** 2))
        K_star += self.block_vars_[n_fixed] * K_phys_star

        mu = self.y_mean_ + self.y_std_ * (K_star @ self.alpha_)
        if self.bounds is not None:
            mu = np.clip(mu, self.bounds[0], self.bounds[1])

        if not return_std:
            return mu.astype(np.float64)

        # Exact diagonal prior variance K(x*, x*) + noise_var
        k_diag = np.full(n_q, self.noise_var_ + self.block_vars_[n_fixed], dtype=np.float64)
        for j, nm in enumerate(self.fixed_names_):
            if nm == "morgan_tanimoto":
                has_bits = (np.sum(X_dict["morgan_bits"], axis=1) > 0).astype(np.float64)
                k_diag += self.block_vars_[j] * has_bits
            elif nm == "cond":
                k_diag += self.block_vars_[j] * np.sum(X_dict["cond"].astype(np.float64) ** 2, axis=1)
            else:
                v_q = X_dict["fm_valid"][nm].astype(np.float64)
                k_diag += self.block_vars_[j] * v_q

        V = solve_triangular(self.L_, K_star.T, lower=True, check_finite=False)
        pred_var = np.maximum(k_diag - np.sum(V * V, axis=0), 0.25 * self.noise_var_)
        sigma = self.y_std_ * np.sqrt(pred_var)
        return mu.astype(np.float64), sigma.astype(np.float64)

    def block_weight_summary(self) -> Dict[str, float]:
        tot = float(np.sum(self.block_vars_) + self.noise_var_)
        out = {nm: round(float(v / tot), 4) for nm, v in zip(self.block_names_, self.block_vars_)}
        out["noise"] = round(float(self.noise_var_ / tot), 4)
        return out


class CompactTabularPreprocessor:
    """Strictly fold-internal <=96D dense feature builder for TabPFN v2 & EB-Huber (100% node2-portable)."""

    # 12 core interpretable SMILES/peptide physchem indices in physchem_core (0..47 SMILES, 48..82 seq)
    # Using first 12 SMILES descriptors + first 4 seq descriptors (always safe & deterministic)
    def __init__(self, track: str, cond_cols: Sequence[str], max_total_dims: int = 94):
        assert track in ("dual_modality", "smiles_only")
        self.track = track
        self.cond_cols = list(cond_cols)
        self.max_total_dims = int(max_total_dims)
        self.cond_enc_: Optional[SmallNConditionEncoder] = None
        self.smi_med_: Optional[np.ndarray] = None
        self.seq_med_: Optional[np.ndarray] = None
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.phys_pca_: Optional[FastDualPCA] = None
        self.morgan_svd_: Optional[TruncatedSVD] = None
        self.fm_stats_: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.fm_pcas_: Dict[str, Optional[FastDualPCA]] = {}
        self.fm_order_: List[str] = []
        self.core_phys_idx_: List[int] = []
        self.out_mean_: Optional[np.ndarray] = None
        self.out_std_: Optional[np.ndarray] = None

    def _extract_phys(self, bank: Any, idx: np.ndarray, fit: bool) -> np.ndarray:
        if self.track == "smiles_only":
            X = bank.physchem[idx, :48].astype(np.float32).copy()
            ok = bank.smiles_valid[idx]
            if fit:
                self.smi_med_ = np.median(X[ok], axis=0).astype(np.float32) if ok.any() else np.zeros(48, dtype=np.float32)
            if (~ok).any():
                X[~ok] = self.smi_med_
            return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_smi = bank.physchem[idx, :48].astype(np.float32).copy()
        X_seq = bank.physchem[idx, 48:83].astype(np.float32).copy()
        smi_ok = bank.smiles_valid[idx]
        seq_ok = bank.seq_valid[idx]
        if fit:
            self.smi_med_ = np.median(X_smi[smi_ok], axis=0).astype(np.float32) if smi_ok.any() else np.zeros(48, dtype=np.float32)
            self.seq_med_ = np.median(X_seq[seq_ok], axis=0).astype(np.float32) if seq_ok.any() else np.zeros(35, dtype=np.float32)
        if (~smi_ok).any():
            X_smi[~smi_ok] = self.smi_med_
        if (~seq_ok).any():
            X_seq[~seq_ok] = self.seq_med_
        flags = np.stack([smi_ok.astype(np.float32), seq_ok.astype(np.float32)], axis=1)
        return np.nan_to_num(np.concatenate([X_smi, X_seq, flags], axis=1), nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, bank: Any, idx: np.ndarray, cond_rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        n = len(idx)
        self.cond_enc_ = SmallNConditionEncoder(self.cond_cols, min_count=3).fit(cond_rows)
        X_cond = self.cond_enc_.transform(cond_rows)
        if X_cond.shape[1] > 12:
            X_cond = X_cond[:, :12]
        n_cond = X_cond.shape[1]

        X_phys = self._extract_phys(bank, idx, fit=True)
        self.phys_mean_ = X_phys.mean(axis=0).astype(np.float32)
        sd = X_phys.std(axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        self.phys_std_ = sd
        Xp = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)

        # Keep top highest-variance non-constant physchem descriptors (up to 10) + PCA(up to 12)
        self.core_phys_idx_ = list(range(min(10, Xp.shape[1])))
        k_phys_pca = max(2, min(12, n // 6, Xp.shape[1] - 1))
        self.phys_pca_ = FastDualPCA(k_phys_pca)
        P_phys = self.phys_pca_.fit_transform(Xp)

        # Compact Morgan SVD (up to 8D)
        X_bits = bank.morgan_bits[idx].astype(np.float32).copy()
        if (~bank.smiles_valid[idx]).any():
            X_bits[~bank.smiles_valid[idx]] = 0.0
        k_svd = max(2, min(8, n // 8, X_bits.shape[1] - 1))
        if np.any(X_bits > 0) and n >= 10:
            self.morgan_svd_ = TruncatedSVD(n_components=k_svd, random_state=42)
            P_morg = self.morgan_svd_.fit_transform(X_bits).astype(np.float32)
        else:
            self.morgan_svd_ = None
            P_morg = np.zeros((n, 0), dtype=np.float32)

        # Remaining budget for 4-FM PCA blocks (guaranteed total <= max_total_dims <= 94!)
        used = len(self.core_phys_idx_) + P_phys.shape[1] + P_morg.shape[1] + n_cond
        rem = max(12, self.max_total_dims - used)

        if self.track == "smiles_only":
            fm_spec = [
                ("ncaa_smiles_repr", 10),
                ("ncaa_fusion_hf", 12),
                ("ncaa_protein_proj", 8),
                ("peptideclm2", 12),
                ("chemberta_mtr", 12),
            ]
        else:
            fm_spec = [
                ("ncaa_smiles_repr", 8),
                ("ncaa_fusion_hf", 10),
                ("peptideclm2", 10),
                ("chemberta_mtr", 10),
                ("esmc_mean", 10),
                ("esmc_max", 6),
            ]

        total_req = sum(k for _, k in fm_spec)
        scale = min(1.0, rem / float(max(1, total_req)))
        parts: List[np.ndarray] = [Xp[:, self.core_phys_idx_], P_phys]
        if P_morg.shape[1] > 0:
            parts.append(P_morg)

        self.fm_order_ = []
        self.fm_stats_ = {}
        self.fm_pcas_ = {}
        for bname, target_k in fm_spec:
            E, v = bank.get_emb_f32(bname, idx)
            n_val = int(v.sum())
            k_use = int(max(2, min(round(target_k * scale), max(2, n_val // 6), n_val - 2, E.shape[1] - 1)))
            if n_val >= 8 and k_use >= 2:
                mu = E[v].mean(axis=0, dtype=np.float64).astype(np.float32)
                sd_e = E[v].std(axis=0, dtype=np.float64).astype(np.float32)
                sd_e[sd_e < 1e-6] = 1.0
                self.fm_stats_[bname] = (mu, sd_e)
                Z = np.clip((E[v] - mu) / sd_e, -8.0, 8.0)
                pca = FastDualPCA(k_use)
                Pv = pca.fit_transform(Z)
                self.fm_pcas_[bname] = pca
                P_full = np.zeros((n, pca.n_components_), dtype=np.float32)
                P_full[v] = Pv
                parts.append(P_full)
                self.fm_order_.append(bname)

        if n_cond > 0:
            parts.append(X_cond.astype(np.float32))

        X_out = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
        self.out_mean_ = X_out.mean(axis=0).astype(np.float32)
        osd = X_out.std(axis=0).astype(np.float32)
        osd[osd < 1e-6] = 1.0
        self.out_std_ = osd
        return np.clip((X_out - self.out_mean_) / self.out_std_, -8.0, 8.0).astype(np.float32)

    def transform(
        self,
        bank: Any,
        idx: np.ndarray,
        cond_rows: Optional[Sequence[Dict[str, Any]]] = None,
        default_cond: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        assert self.cond_enc_ is not None
        n = len(idx)
        if cond_rows is not None:
            X_cond = self.cond_enc_.transform(cond_rows)
        elif default_cond is not None:
            X_cond = self.cond_enc_.transform_constant(n, default_cond)
        else:
            X_cond = self.cond_enc_.transform_constant(n, {})
        if X_cond.shape[1] > 12:
            X_cond = X_cond[:, :12]

        X_phys = self._extract_phys(bank, idx, fit=False)
        Xp = np.clip((X_phys - self.phys_mean_) / self.phys_std_, -8.0, 8.0).astype(np.float32)
        P_phys = self.phys_pca_.transform(Xp)

        parts: List[np.ndarray] = [Xp[:, self.core_phys_idx_], P_phys]
        if self.morgan_svd_ is not None:
            X_bits = bank.morgan_bits[idx].astype(np.float32).copy()
            if (~bank.smiles_valid[idx]).any():
                X_bits[~bank.smiles_valid[idx]] = 0.0
            parts.append(self.morgan_svd_.transform(X_bits).astype(np.float32))

        for bname in self.fm_order_:
            pca = self.fm_pcas_[bname]
            mu, sd_e = self.fm_stats_[bname]
            E, v = bank.get_emb_f32(bname, idx)
            P_full = np.zeros((n, pca.n_components_), dtype=np.float32)
            if v.any():
                Z = np.clip((E[v] - mu) / sd_e, -8.0, 8.0)
                P_full[v] = pca.transform(Z)
            parts.append(P_full)

        if X_cond.shape[1] > 0:
            parts.append(X_cond.astype(np.float32))

        X_out = np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)
        return np.clip((X_out - self.out_mean_) / self.out_std_, -8.0, 8.0).astype(np.float32)


def _make_cluster_inner_folds(cluster_ids: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic greedy load-balanced cluster GroupKFold inside a training fold (ZERO cluster leakage)."""
    uniq = sorted(set(str(c) for c in cluster_ids))
    k = min(n_splits, len(uniq))
    if k < 2:
        # Fallback simple split if single cluster
        n = len(cluster_ids)
        idx = np.arange(n)
        return [(idx[idx % 2 == 0], idx[idx % 2 == 1]), (idx[idx % 2 == 1], idx[idx % 2 == 0])]
    by_c: Dict[str, List[int]] = {c: [] for c in uniq}
    for i, c in enumerate(cluster_ids):
        by_c[str(c)].append(i)
    sorted_c = sorted(uniq, key=lambda c: (-len(by_c[c]), c))
    bins: List[List[int]] = [[] for _ in range(k)]
    for c in sorted_c:
        j = int(np.argmin([len(b) for b in bins]))
        bins[j].extend(by_c[c])
    splits = []
    all_idx = set(range(len(cluster_ids)))
    for j in range(k):
        va = np.array(sorted(bins[j]), dtype=np.int32)
        tr = np.array(sorted(all_idx - set(bins[j])), dtype=np.int32)
        if len(tr) >= 3 and len(va) >= 1:
            splits.append((tr, va))
    return splits


class MultiBlockShrinkageKRRSVR(BaseEstimator, RegressorMixin):
    """Empirical-Bayes Shrinkage Ridge / Cosine+Tanimoto+RBF KRR / SVR with inner cluster-GroupKFold tuning.

    Uses exact multi-block Gram matrices (ZERO PCA loss), supports `mode` in:
      - 'eb_ridge': Dual linear-cosine + standardized physchem + Tanimoto + cond ridge with shrinkage
      - 'cosine_krr': Composite Cosine+Tanimoto+RBF Kernel Ridge + SVR-L1 selection via inner cluster CV
    100% node2-portable (pure numpy/scipy/sklearn).
    """

    def __init__(
        self,
        mode: str = "cosine_krr",
        bounds: Optional[Tuple[float, float]] = None,
        winsor_cap: Optional[float] = None,
    ):
        assert mode in ("eb_ridge", "cosine_krr")
        self.mode = mode
        self.bounds = bounds
        self.winsor_cap = winsor_cap

    @staticmethod
    def _assemble_gram(
        fixed_names: List[str],
        grams: Dict[str, np.ndarray],
        K_phys: np.ndarray,
        profile: str,
    ) -> np.ndarray:
        fm_blocks = [nm for nm in fixed_names if nm not in ("morgan_tanimoto", "cond")]
        n_fm = max(1, len(fm_blocks))
        if profile == "balanced":
            w_fm, w_tan, w_phys, w_cond = 0.55 / n_fm, 0.20, 0.20, 0.12
        elif profile == "fm_heavy":
            w_fm, w_tan, w_phys, w_cond = 0.78 / n_fm, 0.08, 0.10, 0.10
        else:  # chem_heavy
            w_fm, w_tan, w_phys, w_cond = 0.25 / n_fm, 0.35, 0.35, 0.15
        K = w_phys * K_phys
        for nm in fm_blocks:
            K = K + w_fm * grams[nm]
        if "morgan_tanimoto" in grams:
            K = K + w_tan * grams["morgan_tanimoto"]
        if "cond" in grams:
            K = K + w_cond * grams["cond"]
        return K

    def fit(self, X_dict: Dict[str, Any], y: np.ndarray, cluster_ids: Optional[np.ndarray] = None) -> "MultiBlockShrinkageKRRSVR":
        y_arr = np.asarray(y, dtype=np.float64)
        if self.winsor_cap is not None:
            y_arr = np.clip(y_arr, None, self.winsor_cap)
        n = len(y_arr)
        self.y_mean_ = float(np.mean(y_arr))
        y_c = y_arr - self.y_mean_
        cids = np.arange(n) if cluster_ids is None else np.asarray(cluster_ids)
        inner_splits = _make_cluster_inner_folds(cids, n_splits=5)

        fm_k = "cosine" if self.mode == "eb_ridge" else "exp_cosine"
        fixed_names, grams, D2_phys = compute_multiblock_grams(X_dict, X_dict, fm_kernel=fm_k, exp_gamma=1.25)
        tri = D2_phys[np.triu_indices(n, k=1)]
        med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_dict["physchem"].shape[1]))
        self.phys_d2_scale_ = max(med_d2, 1e-4)
        K_phys = np.exp(-0.5 * D2_phys / self.phys_d2_scale_)
        self.fm_kernel_ = fm_k
        self.fixed_names_ = fixed_names

        profiles = ("balanced", "fm_heavy", "chem_heavy")
        alphas = (0.05, 0.15, 0.5, 1.5, 5.0, 15.0)
        shrinks = (1.0, 0.92, 0.82) if self.mode == "eb_ridge" else (1.0,)

        best_score = float("inf")
        best_cfg: Dict[str, Any] = {"learner": "krr", "profile": "balanced", "alpha": 1.0, "shrink": 1.0}

        for prof in profiles:
            K_all = self._assemble_gram(fixed_names, grams, K_phys, prof)
            # KRR / EB-Ridge sweep
            oof_krr = {(a, sh): np.zeros(n, dtype=np.float64) for a in alphas for sh in shrinks}
            for tr, va in inner_splits:
                K_tr = K_all[np.ix_(tr, tr)]
                K_va = K_all[np.ix_(va, tr)]
                y_tr = y_arr[tr]
                ym_tr = float(np.mean(y_tr))
                yc_tr = y_tr - ym_tr
                evals, U = eigh(K_tr)
                evals = np.maximum(evals, 0.0)
                Uty = U.T @ yc_tr
                for a in alphas:
                    dual = U @ (Uty / (evals + a))
                    pred_raw = K_va @ dual
                    for sh in shrinks:
                        p = ym_tr + sh * pred_raw
                        if self.bounds is not None:
                            p = np.clip(p, self.bounds[0], self.bounds[1])
                        oof_krr[(a, sh)][va] = p
            for (a, sh), p_oof in oof_krr.items():
                mae = float(np.mean(np.abs(y_arr - p_oof)))
                if mae < best_score:
                    best_score = mae
                    best_cfg = {"learner": "krr", "profile": prof, "alpha": a, "shrink": sh}

            # In cosine_krr mode, also test exact L1-aligned precomputed-kernel SVR!
            if self.mode == "cosine_krr":
                y_sd = max(float(np.std(y_arr)), 1e-4)
                for C in (0.3, 1.0, 3.0, 10.0):
                    for eps_frac in (0.05, 0.15):
                        eps = eps_frac * y_sd
                        p_oof = np.zeros(n, dtype=np.float64)
                        for tr, va in inner_splits:
                            svr = SVR(kernel="precomputed", C=C, epsilon=eps)
                            svr.fit(K_all[np.ix_(tr, tr)], y_arr[tr])
                            p = svr.predict(K_all[np.ix_(va, tr)])
                            if self.bounds is not None:
                                p = np.clip(p, self.bounds[0], self.bounds[1])
                            p_oof[va] = p
                        mae = float(np.mean(np.abs(y_arr - p_oof)))
                        if mae < best_score:
                            best_score = mae
                            best_cfg = {"learner": "svr", "profile": prof, "C": C, "epsilon": eps}

        self.best_cfg_ = best_cfg
        self.inner_mae_ = best_score
        K_final = self._assemble_gram(fixed_names, grams, K_phys, best_cfg["profile"])
        if best_cfg["learner"] == "krr":
            K_reg = K_final + (float(best_cfg["alpha"]) + 1e-6) * np.eye(n, dtype=np.float64)
            L, lower = cho_factor(K_reg, lower=True, check_finite=False)
            self.dual_coef_ = cho_solve((L, lower), y_c, check_finite=False).astype(np.float64)
            self.shrink_ = float(best_cfg["shrink"])
            self.svr_ = None
        else:
            self.svr_ = SVR(kernel="precomputed", C=float(best_cfg["C"]), epsilon=float(best_cfg["epsilon"]))
            self.svr_.fit(K_final, y_arr)
            self.dual_coef_ = None
            self.shrink_ = 1.0

        self.X_tr_ = {
            "fm_vecs": {k: v.astype(np.float32) for k, v in X_dict["fm_vecs"].items()},
            "fm_valid": {k: v.astype(bool) for k, v in X_dict["fm_valid"].items()},
            "fm_names": list(X_dict["fm_names"]),
            "morgan_bits": X_dict["morgan_bits"].astype(np.uint8),
            "smiles_valid": X_dict["smiles_valid"].astype(bool),
            "physchem": X_dict["physchem"].astype(np.float32),
            "cond": X_dict["cond"].astype(np.float32),
        }
        return self

    def predict(self, X_dict: Dict[str, Any]) -> np.ndarray:
        fixed_names, grams, D2_phys = compute_multiblock_grams(
            X_dict, self.X_tr_, fm_kernel=self.fm_kernel_, exp_gamma=1.25
        )
        K_phys = np.exp(-0.5 * D2_phys / self.phys_d2_scale_)
        K_star = self._assemble_gram(self.fixed_names_, grams, K_phys, self.best_cfg_["profile"])
        if self.svr_ is not None:
            p = self.svr_.predict(K_star)
        else:
            p = self.y_mean_ + self.shrink_ * (K_star @ self.dual_coef_)
        if self.bounds is not None:
            p = np.clip(p, self.bounds[0], self.bounds[1])
        return p.astype(np.float64)


class EBHuberRegressor(BaseEstimator, RegressorMixin):
    """Empirical-Bayes Shrinkage Huber Regressor tuned via inner cluster-GroupKFold (100% node2-portable)."""

    def __init__(
        self,
        bounds: Optional[Tuple[float, float]] = None,
        winsor_cap: Optional[float] = None,
    ):
        self.bounds = bounds
        self.winsor_cap = winsor_cap

    def fit(self, X: np.ndarray, y: np.ndarray, cluster_ids: Optional[np.ndarray] = None) -> "EBHuberRegressor":
        X64 = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if self.winsor_cap is not None:
            y_arr = np.clip(y_arr, None, self.winsor_cap)
        n = len(y_arr)
        cids = np.arange(n) if cluster_ids is None else np.asarray(cluster_ids)
        inner_splits = _make_cluster_inner_folds(cids, n_splits=5)

        best_mae = float("inf")
        best_params = (10.0, 1.35)
        for alpha in (1.0, 5.0, 20.0, 60.0, 150.0):
            for eps in (1.15, 1.35, 1.75):
                p_oof = np.zeros(n, dtype=np.float64)
                ok = True
                for tr, va in inner_splits:
                    try:
                        est = HuberRegressor(alpha=alpha, epsilon=eps, max_iter=300)
                        est.fit(X64[tr], y_arr[tr])
                        p = est.predict(X64[va])
                    except Exception:
                        ok = False
                        break
                    if self.bounds is not None:
                        p = np.clip(p, self.bounds[0], self.bounds[1])
                    p_oof[va] = p
                if not ok:
                    continue
                mae = float(np.mean(np.abs(y_arr - p_oof)))
                if mae < best_mae:
                    best_mae = mae
                    best_params = (alpha, eps)

        self.best_alpha_, self.best_epsilon_ = best_params
        self.inner_mae_ = best_mae
        try:
            self.est_ = HuberRegressor(alpha=self.best_alpha_, epsilon=self.best_epsilon_, max_iter=400)
            self.est_.fit(X64, y_arr)
        except Exception:
            self.est_ = Ridge(alpha=self.best_alpha_).fit(X64, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = np.asarray(self.est_.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64)
        if self.bounds is not None:
            p = np.clip(p, self.bounds[0], self.bounds[1])
        return p


class PortableSmallNModelBundle:
    """Self-contained 5-fold node2-portable Small-N serving bundle (ZERO catboost/tabpfn/torch).

    Can contain one or more 5-fold small-N learners ('mkgpr_cos', 'mkgpr_exp', 'eb_ridge', 'eb_huber', 'cosine_krr')
    plus frozen convex blend weights with existing V33.1 streams ('cpu', 'fusion', 'deep').
    """

    def __init__(
        self,
        endpoint: str,
        track: str,
        bounds: Optional[Tuple[float, float]],
        cond_cols: Sequence[str],
        canonical_conditions: Dict[str, Any],
        blend_weights: Dict[str, float],
        smalln_folds: Dict[str, List[Tuple[int, Any, Any]]],
    ):
        self.schema = "v33_plan_b1_smalln_portable_v1"
        self.endpoint = endpoint
        self.track = track
        self.bounds = bounds
        self.cond_cols = list(cond_cols)
        self.canonical_conditions = dict(canonical_conditions)
        self.blend_weights = dict(blend_weights)
        self.smalln_folds = smalln_folds

    def predict_smalln_streams(
        self,
        bank: Any,
        idx: np.ndarray,
        cond_rows: Optional[Sequence[Dict[str, Any]]] = None,
        default_cond: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        dc = default_cond if default_cond is not None else self.canonical_conditions
        out: Dict[str, np.ndarray] = {}
        gp_sigmas: List[np.ndarray] = []
        for mname, folds in self.smalln_folds.items():
            fold_preds = []
            fold_sigs = []
            for _, pre, est in folds:
                X = pre.transform(bank, idx, cond_rows=cond_rows, default_cond=dc)
                if isinstance(est, MultiBlockCompositeGPR):
                    mu, sig = est.predict(X, return_std=True)
                    fold_preds.append(mu)
                    fold_sigs.append(sig)
                else:
                    fold_preds.append(est.predict(X))
            out[mname] = np.mean(np.stack(fold_preds, axis=0), axis=0)
            if fold_sigs:
                # Total predictive std: sqrt(mean(sigma_f^2) + var(mu_f))
                mean_var = np.mean(np.stack([s ** 2 for s in fold_sigs], axis=0), axis=0)
                epist_var = np.var(np.stack(fold_preds, axis=0), axis=0)
                gp_sigmas.append(np.sqrt(mean_var + epist_var))
        gp_std = np.mean(np.stack(gp_sigmas, axis=0), axis=0) if gp_sigmas else None
        return out, gp_std

    def predict_combined(
        self,
        bank: Any,
        idx: np.ndarray,
        existing_streams: Dict[str, np.ndarray],
        cond_rows: Optional[Sequence[Dict[str, Any]]] = None,
        default_cond: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        sn_streams, gp_std = self.predict_smalln_streams(bank, idx, cond_rows=cond_rows, default_cond=default_cond)
        merged = dict(existing_streams)
        merged.update(sn_streams)
        active = [k for k, w in self.blend_weights.items() if float(w) > 0.0 and k in merged]
        w = np.array([float(self.blend_weights[k]) for k in active], dtype=np.float64)
        w = w / np.sum(w)
        M = np.column_stack([merged[k] for k in active])
        fin = np.isfinite(M)
        M0 = np.where(fin, M, 0.0)
        W = np.where(fin, w[None, :], 0.0)
        wsum = W.sum(axis=1)
        y_out = np.divide((M0 * W).sum(axis=1), wsum, out=np.zeros(len(idx)), where=wsum > 1e-12)
        if self.bounds is not None:
            y_out = np.clip(y_out, self.bounds[0], self.bounds[1])
        return y_out.astype(np.float64), gp_std


# =============================================================================
# Deliverable 2: Node2-Portable Conformal Calibration Models (Methods b, c, d)
# =============================================================================

class StudentizedResidualScaleCalibrator:
    """Method (b): Finite-Sample Corrected Studentized Residual Conformal (100% node2-portable).

    Predicts local residual scale sigma_hat(x) via inner-CV shrinkage-guarded Ridge on log(|r| + eps) using:
      [ad_dist = 1 - max_tanimoto, top5_dist = 1 - top5_tanimoto, compact physchem PCA, optional gp_sigma].
    Computes exact finite-sample quantile ceil((n+1)(1-alpha))/n on |y - mu| / sigma_hat(x).
    """

    def __init__(
        self,
        bounds: Optional[Tuple[float, float]] = None,
        k_phys_pca: int = 4,
    ):
        self.bounds = bounds
        self.k_phys_pca = int(k_phys_pca)
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.phys_pca_: Optional[FastDualPCA] = None
        self.z_mean_: Optional[np.ndarray] = None
        self.z_std_: Optional[np.ndarray] = None
        self.ridge_: Optional[Ridge] = None
        self.use_gp_sigma_: bool = False
        self.use_constant_scale_: bool = False
        self.med_r_: float = 0.5
        self.eps_log_: float = 0.05
        self.q80_: float = 1.0
        self.q90_: float = 1.5

    def _build_features(
        self,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray],
        gp_sigma: Optional[np.ndarray],
        fit: bool,
    ) -> np.ndarray:
        n = len(max_tanimoto)
        t1 = np.where(np.isfinite(max_tanimoto), np.clip(max_tanimoto, 0.0, 1.0), 0.35)
        d1 = (1.0 - t1)[:, None]
        if top5_tanimoto is not None:
            t5 = np.where(np.isfinite(top5_tanimoto), np.clip(top5_tanimoto, 0.0, 1.0), t1)
            d5 = (1.0 - t5)[:, None]
        else:
            d5 = d1.copy()

        Xp = np.nan_to_num(np.asarray(physchem, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.phys_mean_ = Xp.mean(axis=0).astype(np.float32)
            sd = Xp.std(axis=0).astype(np.float32)
            sd[sd < 1e-6] = 1.0
            self.phys_std_ = sd
        Xp_s = np.clip((Xp - self.phys_mean_) / self.phys_std_, -6.0, 6.0)
        if fit:
            k = max(1, min(self.k_phys_pca, max(1, n // 35), Xp_s.shape[1] - 1))
            self.phys_pca_ = FastDualPCA(k)
            Pp = self.phys_pca_.fit_transform(Xp_s)
        else:
            Pp = self.phys_pca_.transform(Xp_s)

        cols = [d1, d5, Pp]
        if fit:
            self.use_gp_sigma_ = gp_sigma is not None and np.isfinite(gp_sigma).mean() > 0.8
        if self.use_gp_sigma_:
            gs = np.where(np.isfinite(gp_sigma), np.maximum(gp_sigma, 1e-4), self.med_r_) if gp_sigma is not None else np.full(n, self.med_r_)
            cols.append(np.log(gs)[:, None])
        Z = np.concatenate(cols, axis=1).astype(np.float64)
        if fit:
            self.z_mean_ = Z.mean(axis=0)
            zsd = Z.std(axis=0)
            zsd[zsd < 1e-6] = 1.0
            self.z_std_ = zsd
        return np.clip((Z - self.z_mean_) / self.z_std_, -5.0, 5.0)

    def fit(
        self,
        y: np.ndarray,
        pred: np.ndarray,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray] = None,
        gp_sigma: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
    ) -> "StudentizedResidualScaleCalibrator":
        abs_r = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))
        n = len(abs_r)
        self.med_r_ = max(float(np.median(abs_r)), 1e-3)
        self.eps_log_ = max(0.20 * self.med_r_, 1e-3)

        Z = self._build_features(physchem, max_tanimoto, top5_tanimoto, gp_sigma, fit=True)
        target = np.log(abs_r + self.eps_log_)

        # Choose strong L2 alpha via nested fold CV; guard against noisy scale predictors via constant-mean baseline!
        self.use_constant_scale_ = False
        if fold_ids is not None and len(set(fold_ids.tolist())) >= 2:
            uniq_f = sorted(set(fold_ids.tolist()))
            const_errs = []
            for f in uniq_f:
                va = fold_ids == f
                tr = ~va
                if tr.sum() < 5 or va.sum() == 0:
                    continue
                const_errs.append(np.mean((target[va] - float(np.mean(target[tr]))) ** 2))
            const_mse = float(np.mean(const_errs)) if const_errs else float("inf")

            best_a = 100.0
            best_loss = float("inf")
            for a in (10.0, 30.0, 100.0, 300.0, 1000.0):
                errs = []
                for f in uniq_f:
                    va = fold_ids == f
                    tr = ~va
                    if tr.sum() < 5 or va.sum() == 0:
                        continue
                    reg = Ridge(alpha=a).fit(Z[tr], target[tr])
                    errs.append(np.mean((target[va] - reg.predict(Z[va])) ** 2))
                if errs and float(np.mean(errs)) < best_loss:
                    best_loss = float(np.mean(errs))
                    best_a = a
            if best_loss >= const_mse * 0.995:
                self.use_constant_scale_ = True
        else:
            best_a = 100.0

        self.best_alpha_ = float(best_a)

        # Compute STRICTLY OUT-OF-FOLD local scales sigma_hat_oof so studentized scores are never optimistic!
        if self.use_constant_scale_:
            sigma_oof = np.full(n, self.med_r_, dtype=np.float64)
            self.ridge_ = None
        else:
            sigma_oof = np.full(n, self.med_r_, dtype=np.float64)
            if fold_ids is not None and len(set(fold_ids.tolist())) >= 2:
                for f in sorted(set(fold_ids.tolist())):
                    va = fold_ids == f
                    tr = ~va
                    if tr.sum() < 5 or va.sum() == 0:
                        continue
                    reg_f = Ridge(alpha=self.best_alpha_).fit(Z[tr], target[tr])
                    s_va = np.exp(reg_f.predict(Z[va])) - self.eps_log_
                    sigma_oof[va] = np.clip(s_va, 0.45 * self.med_r_, 2.8 * self.med_r_)
            else:
                reg_tmp = Ridge(alpha=self.best_alpha_).fit(Z, target)
                sigma_oof = np.clip(np.exp(reg_tmp.predict(Z)) - self.eps_log_, 0.45 * self.med_r_, 2.8 * self.med_r_)
            self.ridge_ = Ridge(alpha=self.best_alpha_).fit(Z, target)

        scores = abs_r / sigma_oof
        self.n_cal_ = int(n)
        self.n_eff_clusters_ = int(len(set(str(c) for c in cluster_ids))) if cluster_ids is not None else int(n)
        # Exact finite-sample quantile ceil((n+1)(1-alpha))/n
        self.q80_ = _conformal_quantile(scores, 0.20)
        self.q90_ = _conformal_quantile(scores, 0.10)
        return self

    def predict_scale(
        self,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray] = None,
        gp_sigma: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.use_constant_scale_ or self.ridge_ is None:
            return np.full(len(max_tanimoto), self.med_r_, dtype=np.float64)
        Z = self._build_features(physchem, max_tanimoto, top5_tanimoto, gp_sigma, fit=False)
        s = np.exp(self.ridge_.predict(Z)) - self.eps_log_
        return np.clip(s, 0.45 * self.med_r_, 2.8 * self.med_r_).astype(np.float64)

    def predict_intervals(
        self,
        pred: np.ndarray,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray] = None,
        gp_sigma: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(pred, dtype=np.float64)
        s = self.predict_scale(physchem, max_tanimoto, top5_tanimoto, gp_sigma)
        lo80, hi80 = p - self.q80_ * s, p + self.q80_ * s
        lo90, hi90 = p - self.q90_ * s, p + self.q90_ * s
        if self.bounds is not None:
            lo80, hi80 = np.clip(lo80, self.bounds[0], self.bounds[1]), np.clip(hi80, self.bounds[0], self.bounds[1])
            lo90, hi90 = np.clip(lo90, self.bounds[0], self.bounds[1]), np.clip(hi90, self.bounds[0], self.bounds[1])
        return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90, "scale": s}


class ResidualCQRCalibrator:
    """Method (c): Residual-Centered Conformalized Quantile Regression (Romano et al., 2019).

    Trains fold-internal regularized LightGBM Quantile Regressors (alpha=0.10/0.90 & 0.05/0.95) on
    signed residuals e = y - y_hat using [ad_dist, top5_dist, physchem_compact, y_hat], then applies
    exact out-of-fold finite-sample signed conformity adjustment E_i = max(q_lo - y, y - q_hi).
    100% node2-portable (pure numpy + lightgbm)!
    """

    def __init__(self, bounds: Optional[Tuple[float, float]] = None, k_phys_pca: int = 12):
        self.bounds = bounds
        self.k_phys_pca = int(k_phys_pca)
        self.phys_mean_: Optional[np.ndarray] = None
        self.phys_std_: Optional[np.ndarray] = None
        self.phys_pca_: Optional[FastDualPCA] = None
        self.fold_models_: List[Dict[str, Any]] = []
        self.Q80_: float = 0.0
        self.Q90_: float = 0.0

    def _build_X(
        self,
        pred: np.ndarray,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray],
        fit: bool,
    ) -> np.ndarray:
        n = len(pred)
        t1 = np.where(np.isfinite(max_tanimoto), np.clip(max_tanimoto, 0.0, 1.0), 0.35)
        d1 = (1.0 - t1)[:, None]
        if top5_tanimoto is not None:
            t5 = np.where(np.isfinite(top5_tanimoto), np.clip(top5_tanimoto, 0.0, 1.0), t1)
            d5 = (1.0 - t5)[:, None]
        else:
            d5 = d1.copy()
        Xp = np.nan_to_num(np.asarray(physchem, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.phys_mean_ = Xp.mean(axis=0).astype(np.float32)
            sd = Xp.std(axis=0).astype(np.float32)
            sd[sd < 1e-6] = 1.0
            self.phys_std_ = sd
        Xp_s = np.clip((Xp - self.phys_mean_) / self.phys_std_, -6.0, 6.0)
        if fit:
            k = max(2, min(self.k_phys_pca, max(2, n // 12), Xp_s.shape[1] - 1))
            self.phys_pca_ = FastDualPCA(k)
            Pp = self.phys_pca_.fit_transform(Xp_s)
        else:
            Pp = self.phys_pca_.transform(Xp_s)
        p_col = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0)[:, None]
        return np.concatenate([d1.astype(np.float32), d5.astype(np.float32), p_col, Pp.astype(np.float32)], axis=1)

    @staticmethod
    def _make_lgb_q(alpha: float, n_tr: int, seed: int) -> lgb.LGBMRegressor:
        min_child = max(5, min(25, n_tr // 10))
        n_est = 35 if n_tr < 200 else (60 if n_tr < 800 else 90)
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(alpha),
            n_estimators=n_est,
            learning_rate=0.05,
            num_leaves=7 if n_tr < 300 else 15,
            max_depth=3 if n_tr < 300 else 4,
            min_child_samples=min_child,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )

    def fit(
        self,
        y: np.ndarray,
        pred: np.ndarray,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
        cluster_ids: Optional[np.ndarray] = None,
    ) -> "ResidualCQRCalibrator":
        y_arr = np.asarray(y, dtype=np.float64)
        p_arr = np.asarray(pred, dtype=np.float64)
        res = y_arr - p_arr
        n = len(y_arr)
        X = self._build_X(p_arr, physchem, max_tanimoto, top5_tanimoto, fit=True)

        if fold_ids is None or len(set(fold_ids.tolist())) < 2:
            fids = np.arange(n) % 5
        else:
            fids = np.asarray(fold_ids, dtype=np.int32)
        uniq_f = sorted(set(fids.tolist()))

        q10_oof = np.zeros(n, dtype=np.float64)
        q90_oof = np.zeros(n, dtype=np.float64)
        q05_oof = np.zeros(n, dtype=np.float64)
        q95_oof = np.zeros(n, dtype=np.float64)
        self.fold_models_ = []

        for f in uniq_f:
            va = fids == f
            tr = ~va
            n_tr = int(tr.sum())
            if n_tr < 8 or va.sum() == 0:
                q10_oof[va] = float(np.quantile(res[tr], 0.10)) if n_tr > 0 else -0.5
                q90_oof[va] = float(np.quantile(res[tr], 0.90)) if n_tr > 0 else 0.5
                q05_oof[va] = float(np.quantile(res[tr], 0.05)) if n_tr > 0 else -0.8
                q95_oof[va] = float(np.quantile(res[tr], 0.95)) if n_tr > 0 else 0.8
                continue
            m10 = self._make_lgb_q(0.10, n_tr, 101 + f).fit(X[tr], res[tr])
            m90 = self._make_lgb_q(0.90, n_tr, 201 + f).fit(X[tr], res[tr])
            m05 = self._make_lgb_q(0.05, n_tr, 301 + f).fit(X[tr], res[tr])
            m95 = self._make_lgb_q(0.95, n_tr, 401 + f).fit(X[tr], res[tr])

            d10 = m10.predict(X[va])
            d90 = m90.predict(X[va])
            d05 = m05.predict(X[va])
            d95 = m95.predict(X[va])

            # Non-crossing quantile projection
            lo80_d = np.minimum(d10, d90)
            hi80_d = np.maximum(d10, d90)
            lo90_d = np.minimum(np.minimum(d05, d95), lo80_d)
            hi90_d = np.maximum(np.maximum(d05, d95), hi80_d)

            q10_oof[va] = p_arr[va] + lo80_d
            q90_oof[va] = p_arr[va] + hi80_d
            q05_oof[va] = p_arr[va] + lo90_d
            q95_oof[va] = p_arr[va] + hi90_d
            self.fold_models_.append({"fold": int(f), "m10": m10, "m90": m90, "m05": m05, "m95": m95})

        # Exact Romano et al. (2019) CQR signed nonconformity scores on strictly OOF predictions!
        E80 = np.maximum(q10_oof - y_arr, y_arr - q90_oof)
        E90 = np.maximum(q05_oof - y_arr, y_arr - q95_oof)
        self.Q80_ = _conformal_quantile(E80, 0.20)
        self.Q90_ = _conformal_quantile(E90, 0.10)
        self.n_cal_ = int(n)
        self.n_eff_clusters_ = int(len(set(str(c) for c in cluster_ids))) if cluster_ids is not None else int(n)
        return self

    def predict_intervals(
        self,
        pred: np.ndarray,
        physchem: np.ndarray,
        max_tanimoto: np.ndarray,
        top5_tanimoto: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(pred, dtype=np.float64)
        X = self._build_X(p, physchem, max_tanimoto, top5_tanimoto, fit=False)
        if not self.fold_models_:
            lo80, hi80 = p - abs(self.Q80_), p + abs(self.Q80_)
            lo90, hi90 = p - abs(self.Q90_), p + abs(self.Q90_)
        else:
            d10 = np.mean([fm["m10"].predict(X) for fm in self.fold_models_], axis=0)
            d90 = np.mean([fm["m90"].predict(X) for fm in self.fold_models_], axis=0)
            d05 = np.mean([fm["m05"].predict(X) for fm in self.fold_models_], axis=0)
            d95 = np.mean([fm["m95"].predict(X) for fm in self.fold_models_], axis=0)
            lo80_d = np.minimum(d10, d90)
            hi80_d = np.maximum(d10, d90)
            lo90_d = np.minimum(np.minimum(d05, d95), lo80_d)
            hi90_d = np.maximum(np.maximum(d05, d95), hi80_d)
            lo80 = p + lo80_d - self.Q80_
            hi80 = p + hi80_d + self.Q80_
            lo90 = p + lo90_d - self.Q90_
            hi90 = p + hi90_d + self.Q90_
            # Ensure 90% CI always encloses 80% CI
            lo90 = np.minimum(lo90, lo80)
            hi90 = np.maximum(hi90, hi80)
        if self.bounds is not None:
            lo80, hi80 = np.clip(lo80, self.bounds[0], self.bounds[1]), np.clip(hi80, self.bounds[0], self.bounds[1])
            lo90, hi90 = np.clip(lo90, self.bounds[0], self.bounds[1]), np.clip(hi90, self.bounds[0], self.bounds[1])
        return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}


def assign_ad_tiers(max_tanimoto: np.ndarray) -> np.ndarray:
    """Assign exact V33.1 AD tiers: In-Domain >= 0.65, Borderline [0.40, 0.65), OOD < 0.40 (or NaN)."""
    t = np.asarray(max_tanimoto, dtype=np.float64)
    tiers = np.full(len(t), "OOD", dtype=object)
    ok = np.isfinite(t)
    tiers[ok & (t >= AD_IN_DOMAIN_THR)] = "In-Domain"
    tiers[ok & (t >= AD_BORDERLINE_THR) & (t < AD_IN_DOMAIN_THR)] = "Borderline"
    return tiers


def _weighted_pava_nondecreasing(vals: List[float], weights: List[float]) -> List[float]:
    """Exact weighted Pool-Adjacent-Violators (unbiased isotonic non-decreasing projection)."""
    blocks: List[List[float]] = []  # [sum_wv, sum_w, count]
    for v, w in zip(vals, weights):
        ww = max(float(w), 1e-6)
        blocks.append([v * ww, ww, 1])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            b2 = blocks.pop()
            blocks[-1][0] += b2[0]
            blocks[-1][1] += b2[1]
            blocks[-1][2] += b2[2]
    out: List[float] = []
    for swv, sw, cnt in blocks:
        m = swv / sw
        out.extend([m] * int(cnt))
    return out


def fit_mondrian_ad_eb(
    y: np.ndarray,
    pred: np.ndarray,
    max_tanimoto: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    cluster_ids: Optional[np.ndarray] = None,
    min_stratum_n: int = 25,
    enforce_monotonic: bool = True,
) -> Dict[str, Any]:
    """Method (d): Mondrian AD-Stratified Empirical-Bayes Shrinkage Conformal (pure JSON spec!).

    Stratifies residuals by AD tier (In-Domain >= 0.65, Borderline 0.40-0.65, OOD < 0.40).
    Uses exact finite-sample quantile ceil((n+1)(1-alpha))/n; when stratum count n_g < 25, applies
    smooth Empirical-Bayes shrinkage toward global quantile so small-N bins never collapse,
    plus unbiased weighted-PAVA monotonic AD-risk ordering (q_InDomain <= q_Borderline <= q_OOD).
    """
    abs_r = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))
    tiers = assign_ad_tiers(max_tanimoto)
    n_tot = len(abs_r)
    n_eff_tot = len(set(str(c) for c in cluster_ids)) if cluster_ids is not None else n_tot

    q80_glob = _conformal_quantile(abs_r, 0.20)
    q90_glob = _conformal_quantile(abs_r, 0.10)

    tier_order = ("In-Domain", "Borderline", "OOD")
    q80_tier: Dict[str, float] = {}
    q90_tier: Dict[str, float] = {}
    tier_counts: Dict[str, int] = {}
    tier_shrink: Dict[str, float] = {}

    for tname in tier_order:
        m = tiers == tname
        ng = int(m.sum())
        tier_counts[tname] = ng
        if ng == 0:
            q80_tier[tname] = q80_glob
            q90_tier[tname] = q90_glob
            tier_shrink[tname] = 0.0
            continue
        q80_raw = _conformal_quantile(abs_r[m], 0.20)
        q90_raw = _conformal_quantile(abs_r[m], 0.10)
        # Smooth Empirical-Bayes shrinkage toward global when stratum count < min_stratum_n (25)
        lam = 1.0 if ng >= min_stratum_n else (float(ng) / float(min_stratum_n))
        tier_shrink[tname] = round(lam, 4)
        q80_tier[tname] = lam * q80_raw + (1.0 - lam) * q80_glob
        q90_tier[tname] = lam * q90_raw + (1.0 - lam) * q90_glob

    if enforce_monotonic:
        ws = [max(float(tier_counts[t]), 1.0) for t in tier_order]
        p80 = _weighted_pava_nondecreasing([q80_tier[t] for t in tier_order], ws)
        p90 = _weighted_pava_nondecreasing([q90_tier[t] for t in tier_order], ws)
        for i, tname in enumerate(tier_order):
            q80_tier[tname] = p80[i]
            q90_tier[tname] = max(p90[i], p80[i])

    return {
        "mode": "mondrian_ad_eb",
        "ad_thresholds": {"in_domain": AD_IN_DOMAIN_THR, "borderline": AD_BORDERLINE_THR},
        "min_stratum_n": int(min_stratum_n),
        "n_cal_total": int(n_tot),
        "n_eff_clusters": int(n_eff_tot),
        "q80_global": round(float(q80_glob), 5),
        "q90_global": round(float(q90_glob), 5),
        "q80_by_tier": {k: round(float(v), 5) for k, v in q80_tier.items()},
        "q90_by_tier": {k: round(float(v), 5) for k, v in q90_tier.items()},
        "tier_counts": tier_counts,
        "tier_eb_weights": tier_shrink,
        "bounds": list(bounds) if bounds is not None else None,
    }


def apply_mondrian_ad_eb(
    pred: np.ndarray,
    spec: Dict[str, Any],
    max_tanimoto: np.ndarray,
) -> Dict[str, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    tiers = assign_ad_tiers(max_tanimoto)
    q80 = np.array([float(spec["q80_by_tier"][t]) for t in tiers], dtype=np.float64)
    q90 = np.array([float(spec["q90_by_tier"][t]) for t in tiers], dtype=np.float64)
    lo80, hi80 = p - q80, p + q80
    lo90, hi90 = p - q90, p + q90
    bounds = spec.get("bounds")
    if bounds is not None:
        lo80, hi80 = np.clip(lo80, bounds[0], bounds[1]), np.clip(hi80, bounds[0], bounds[1])
        lo90, hi90 = np.clip(lo90, bounds[0], bounds[1]), np.clip(hi90, bounds[0], bounds[1])
    return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}
