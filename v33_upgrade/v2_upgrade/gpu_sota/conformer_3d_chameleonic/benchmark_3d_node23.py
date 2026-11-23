#!/usr/bin/env python3
"""Plan B-2 Upgrade 2 — 11-Endpoint 3D Conformer-Aware Chameleonic Benchmark on node23 (64 CPU cores, 125 GB RAM).

Strictly evaluates frozen 5-fold cluster CV + untouched 20% holdout + Leave-17-Cases-Out across all 11
flagship Tier-1 PK/Permeability endpoints:
  PAMPA, Caco2, MDCK, RRCK, efflux_ratio, F, PPB, LogD7.4, half_life_reg, CL_per_kg, Vd_per_kg

Compares:
  1. Controlled Direct 4-FM Fusion Ablation:
     direct_dual_base vs direct_dual_plus_3d (and direct_smiles_base vs direct_smiles_plus_3d)
  2. End-to-End Nested OOF Stacked Champion:
     Official Frozen 4-FM Fusion vs 4-FM Fusion + 29D RDKit ETKDGv3 3D Chameleonic Descriptors (node2-CPU-portable!)
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import gc
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
EXP_BASE = V33_BASE / "expansion_v331"
OUT_3D_DIR = V33_BASE / "v2_upgrade" / "gpu_sota" / "conformer_3d_chameleonic"

for p in (str(V33_BASE / "models_fusion"), str(V33_BASE / "models_cpu"), str(EXP_BASE / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion_config import DIRECT_FUSION_REPS, ENDPOINT_SPECS as CORE24_SPECS  # noqa: E402
from fusion_metrics import regression_metrics  # noqa: E402
from fusion_models import (  # noqa: E402
    BoundedWrapper,
    EmbeddingFoldPreprocessor,
    ExactRBFKRR,
    ModalityAwareStacker,
    PrecomputedRBFSVR,
    compute_sqdist_matrix,
    make_fusion_estimator,
)
from v331_config import V331_ENDPOINT_SPECS  # noqa: E402

csv.field_size_limit(10_000_000)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

TIER1_11_ENDPOINTS: List[str] = [
    "PAMPA",
    "Caco2",
    "MDCK",
    "RRCK",
    "efflux_ratio",
    "F",
    "PPB",
    "LogD7.4",
    "half_life_reg",
    "CL_per_kg",
    "Vd_per_kg",
]

ALL_SPECS: Dict[str, Dict[str, Any]] = {}
for _k, _v in CORE24_SPECS.items():
    ALL_SPECS[_k] = dict(_v)
for _k, _v in V331_ENDPOINT_SPECS.items():
    ALL_SPECS[_k] = dict(_v)

GLOBAL_BANK: Optional["Node23FusionBank3D"] = None


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _col_arr(rows: List[Dict[str, str]], col: str) -> Optional[np.ndarray]:
    if not rows or col not in rows[0]:
        return None
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        v = r.get(col, "")
        if v != "":
            out[i] = float(v)
    return out


class Node23FusionBank3D:
    """Read-only shared 50,995-compound 4-FM + 29D 3D Chameleonic bank (CoW-shared across forked workers)."""

    def __init__(self) -> None:
        feat_dir = EXP_BASE / "features"
        emb_dir = EXP_BASE / "embeddings"
        manifest = json.loads((feat_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.case17_indices: Dict[str, int] = manifest["case17_indices"]
        self.n_total = int(manifest["n_compounds_total"])

        self.key_to_idx: Dict[str, int] = {}
        with (feat_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.key_to_idx[r["compound_key"]] = int(r["compound_idx"])

        pc = np.load(feat_dir / "physchem_core.npz")
        self.physchem = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        self.emb_blocks: Dict[str, np.ndarray] = {}
        self.emb_valid: Dict[str, np.ndarray] = {}

        d_ncaa = np.load(emb_dir / "ncaa_repdistill_v2.npz")
        v_ncaa = np.ascontiguousarray(d_ncaa["valid"], dtype=bool)
        for k, bk in [("smiles_repr", "ncaa_smiles_repr"), ("fusion_hf", "ncaa_fusion_hf"), ("protein_proj", "ncaa_protein_proj")]:
            self.emb_blocks[bk] = np.ascontiguousarray(d_ncaa[k], dtype=np.float16)
            self.emb_valid[bk] = v_ncaa
        d_ncaa.close()

        d_pclm = np.load(emb_dir / "peptideclm2.npz")
        self.emb_blocks["peptideclm2"] = np.ascontiguousarray(d_pclm["embeddings"], dtype=np.float16)
        self.emb_valid["peptideclm2"] = np.ascontiguousarray(d_pclm["valid"], dtype=bool)
        d_pclm.close()

        d_cb = np.load(emb_dir / "chemberta_mtr.npz")
        self.emb_blocks["chemberta_mtr"] = np.ascontiguousarray(d_cb["embeddings"], dtype=np.float16)
        self.emb_valid["chemberta_mtr"] = np.ascontiguousarray(d_cb["valid"], dtype=bool)
        d_cb.close()

        d_esmc = np.load(emb_dir / "esmc_600m_multiregion.npz")
        v_esmc = np.ascontiguousarray(d_esmc["valid"], dtype=bool)
        for k, bk in [("mean", "esmc_mean"), ("max", "esmc_max"), ("nterm5", "esmc_nterm5"), ("cterm5", "esmc_cterm5")]:
            self.emb_blocks[bk] = np.ascontiguousarray(d_esmc[k], dtype=np.float16)
            self.emb_valid[bk] = v_esmc
        d_esmc.close()

        # Load deterministic 29D RDKit ETKDGv3 3D Chameleonic descriptors
        d3 = np.load(OUT_3D_DIR / "chameleonic_3d_tier1.npz")
        self.X_3d = np.ascontiguousarray(d3["X_3d"], dtype=np.float32)
        self.valid_3d = np.ascontiguousarray(d3["valid_3d"], dtype=bool)
        d3.close()
        assert self.X_3d.shape[0] == self.n_total
        self.emb_blocks["chameleonic_3d"] = self.X_3d
        self.emb_valid["chameleonic_3d"] = self.valid_3d

        self.case17_queries = _read_tsv(V33_BASE / "data" / "case17_queries.tsv")
        for r in self.case17_queries:
            r["compound_idx"] = self.case17_indices[r["drug_name"]]

    def get_emb_f32(self, block_name: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.emb_blocks[block_name][idx].astype(np.float32),
            self.emb_valid[block_name][idx].copy(),
        )

    def load_endpoint_with_baselines(self, endpoint: str) -> Dict[str, Any]:
        is_v331 = (endpoint == "efflux_ratio")
        ep_path = (EXP_BASE / "data" / "endpoints" / f"{endpoint}.tsv") if is_v331 else (V33_BASE / "data" / "endpoints" / f"{endpoint}.tsv")
        cpu_dir = (EXP_BASE / "models_cpu" / endpoint) if is_v331 else (V33_BASE / "models_cpu" / endpoint)
        fus_dir = (EXP_BASE / "models_fusion" / endpoint) if is_v331 else (V33_BASE / "models_fusion" / endpoint)

        rows = _read_tsv(ep_path)
        cols = list(rows[0].keys())
        cond_cols = [c for c in cols if c.startswith("cond_")]
        record_ids = [r["record_id"] for r in rows]
        compound_keys = [r["compound_key"] for r in rows]
        compound_idx = np.array([self.key_to_idx[k] for k in compound_keys], dtype=np.int32)
        cluster_ids = np.array([r["cluster_id"] for r in rows], dtype=object)
        is_holdout = np.array([int(r["is_holdout"]) for r in rows], dtype=np.int32)
        fold_id = np.array([int(r["fold_id"]) for r in rows], dtype=np.int32)
        case17_ex = np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32)
        censor_type = np.array([r["censor_type"] for r in rows], dtype=object)
        y_value = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)
        y_lower = np.array([float(r["y_lower"]) for r in rows], dtype=np.float64)
        y_upper = np.array([float(r["y_upper"]) for r in rows], dtype=np.float64)
        cond_rows = [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows]

        cv_idx = np.where(is_holdout == 0)[0]
        ho_idx = np.where(is_holdout == 1)[0]
        assert set(cluster_ids[cv_idx]).isdisjoint(set(cluster_ids[ho_idx])), f"{endpoint}: cluster leak!"

        cpu_oof = _read_tsv(cpu_dir / "oof_predictions.tsv")
        cpu_ho = _read_tsv(cpu_dir / "holdout_predictions.tsv")
        cpu_c17 = _read_tsv(cpu_dir / "case17_predictions.tsv")
        fus_oof = _read_tsv(fus_dir / "oof_predictions.tsv")
        fus_ho = _read_tsv(fus_dir / "holdout_predictions.tsv")
        fus_c17 = _read_tsv(fus_dir / "case17_predictions.tsv")

        assert len(cpu_oof) == len(fus_oof) == len(cv_idx)
        assert len(cpu_ho) == len(fus_ho) == len(ho_idx)
        for li, oi in enumerate(cv_idx):
            assert cpu_oof[li]["record_id"] == fus_oof[li]["record_id"] == record_ids[oi]
        for li, oi in enumerate(ho_idx):
            assert cpu_ho[li]["record_id"] == fus_ho[li]["record_id"] == record_ids[oi]

        return {
            "endpoint": endpoint,
            "n": len(rows),
            "cond_cols": cond_cols,
            "record_ids": record_ids,
            "compound_keys": compound_keys,
            "compound_idx": compound_idx,
            "cluster_ids": cluster_ids,
            "is_holdout": is_holdout,
            "fold_id": fold_id,
            "case17_excluded": case17_ex,
            "censor_type": censor_type,
            "y_value": y_value,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "cond_rows": cond_rows,
            "cv_idx": cv_idx,
            "ho_idx": ho_idx,
            "official": {
                "cpu_oof_dual": _col_arr(cpu_oof, "pred_dual"),
                "cpu_oof_smi": _col_arr(cpu_oof, "pred_smiles_only"),
                "cpu_ho_dual": _col_arr(cpu_ho, "pred_dual"),
                "cpu_ho_smi": _col_arr(cpu_ho, "pred_smiles_only"),
                "cpu_c17_dual": _col_arr(cpu_c17, "pred_dual"),
                "cpu_c17_smi": _col_arr(cpu_c17, "pred_smiles_only"),
                "fus_oof_dual": _col_arr(fus_oof, "pred_dual"),
                "fus_oof_smi": _col_arr(fus_oof, "pred_smiles_only"),
                "fus_ho_dual": _col_arr(fus_ho, "pred_dual"),
                "fus_ho_smi": _col_arr(fus_ho, "pred_smiles_only"),
                "fus_c17_dual": _col_arr(fus_c17, "pred_dual"),
                "fus_c17_smi": _col_arr(fus_c17, "pred_smiles_only"),
            },
        }


class AugmentedFoldPreprocessor(EmbeddingFoldPreprocessor):
    """Exact EmbeddingFoldPreprocessor keeping 29D `chameleonic_3d` uncompressed inside pca_gbdt."""

    def fit_transform(self, bank: Node23FusionBank3D, idx: np.ndarray, cond_rows: List[Dict[str, str]]) -> np.ndarray:
        if self.mode == "balanced":
            return super().fit_transform(bank, idx, cond_rows)
        orig_pca = list(self.pca_group_dims)
        has_3d = any(b == "chameleonic_3d" for b, _ in orig_pca)
        self.pca_group_dims = [(b, k) for b, k in orig_pca if b != "chameleonic_3d"]
        X_base = super().fit_transform(bank, idx, cond_rows)
        self.pca_group_dims = orig_pca
        if not has_3d:
            return X_base
        E3, v3 = bank.get_emb_f32("chameleonic_3d", idx)
        self._has_raw_3d = True
        self._med_3d = np.median(E3[v3], axis=0).astype(np.float32) if v3.any() else np.zeros(E3.shape[1], dtype=np.float32)
        E3_out = E3.copy()
        if (~v3).any():
            E3_out[~v3] = self._med_3d
            E3_out[~v3, -1] = 0.0
        if self.mode == "pca_mlp":
            self._mu_3d = E3_out.mean(axis=0).astype(np.float32)
            sd3 = E3_out.std(axis=0).astype(np.float32)
            sd3[sd3 < 1e-6] = 1.0
            self._sd_3d = sd3
            E3_out = np.clip((E3_out - self._mu_3d) / self._sd_3d, -8.0, 8.0).astype(np.float32)
        return np.ascontiguousarray(np.concatenate([X_base, E3_out], axis=1), dtype=np.float32)

    def transform(
        self,
        bank: Node23FusionBank3D,
        idx: np.ndarray,
        cond_rows: Optional[List[Dict[str, str]]] = None,
        default_cond: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        if self.mode == "balanced":
            return super().transform(bank, idx, cond_rows=cond_rows, default_cond=default_cond)
        orig_pca = list(self.pca_group_dims)
        has_3d = getattr(self, "_has_raw_3d", False)
        self.pca_group_dims = [(b, k) for b, k in orig_pca if b != "chameleonic_3d"]
        X_base = super().transform(bank, idx, cond_rows=cond_rows, default_cond=default_cond)
        self.pca_group_dims = orig_pca
        if not has_3d:
            return X_base
        E3, v3 = bank.get_emb_f32("chameleonic_3d", idx)
        E3_out = E3.copy()
        if (~v3).any():
            E3_out[~v3] = self._med_3d
            E3_out[~v3, -1] = 0.0
        if self.mode == "pca_mlp":
            E3_out = np.clip((E3_out - self._mu_3d) / self._sd_3d, -8.0, 8.0).astype(np.float32)
        return np.ascontiguousarray(np.concatenate([X_base, E3_out], axis=1), dtype=np.float32)


def _score_reg(m: Dict[str, Any]) -> Tuple[float, float]:
    return (-(m["mae"] if m["mae"] is not None else 1e9), m["spearman"] if m["spearman"] is not None else -1.0)


def _predict_single(est: Any, X: np.ndarray, D2_star: Optional[np.ndarray] = None) -> np.ndarray:
    if D2_star is not None:
        inner = est.est if isinstance(est, BoundedWrapper) else est
        if isinstance(inner, (ExactRBFKRR, PrecomputedRBFSVR)):
            p = inner.predict(X, D2_star=D2_star)
            if isinstance(est, BoundedWrapper) and est.bounds is not None:
                p = np.clip(p, est.bounds[0], est.bounds[1])
            return p.astype(np.float64)
    return np.asarray(est.predict(X), dtype=np.float64)


def _get_candidates(size_tier: str, n_cv: int, winsor_cap: Optional[float] = None) -> List[Dict[str, Any]]:
    cands = []
    for alpha in (0.1, 0.5, 2.0, 10.0, 50.0, 200.0):
        cands.append({"mode": "balanced", "family": "Ridge", "params": {"alpha": alpha}, "winsor_cap": winsor_cap})
    if n_cv <= 3600:
        g_mults = (0.5, 1.0, 2.0) if n_cv <= 1500 else (0.5, 1.0)
        for gm in g_mults:
            for alpha in (0.03, 0.15, 0.6, 2.5):
                cands.append({"mode": "balanced", "family": "KRR_RBF", "params": {"alpha": alpha, "gamma_mult": gm}, "winsor_cap": winsor_cap})
        for C in ((0.5, 2.0, 8.0) if n_cv <= 2600 else (1.0, 4.0)):
            cands.append({"mode": "balanced", "family": "SVR_RBF", "params": {"C": C, "epsilon": 0.08, "gamma_mult": 1.0}, "winsor_cap": winsor_cap})
    else:
        for gm in (0.5, 1.0, 2.0):
            for alpha in (0.2, 1.0, 5.0):
                cands.append({"mode": "balanced", "family": "Nystrom_RBF", "params": {"n_components": 1000, "alpha": alpha, "gamma_mult": gm}, "winsor_cap": winsor_cap})
        cands.append({"mode": "pca_mlp", "family": "MLP", "params": {"hidden": (256, 64), "alpha": 0.3, "lr": 1e-3, "max_iter": 55}, "winsor_cap": winsor_cap})

    if size_tier == "small":
        lgb_cfgs = [
            {"n_estimators": 160, "learning_rate": 0.04, "max_depth": 3, "num_leaves": 7, "min_child_samples": 6, "reg_lambda": 10.0, "colsample_bytree": 0.6},
            {"n_estimators": 220, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 12, "min_child_samples": 10, "reg_lambda": 15.0, "colsample_bytree": 0.5},
        ]
        xgb_cfgs = [{"n_estimators": 160, "learning_rate": 0.04, "max_depth": 3, "min_child_weight": 4.0, "reg_lambda": 10.0, "colsample_bytree": 0.6}]
    elif size_tier == "medium":
        lgb_cfgs = [
            {"n_estimators": 450, "learning_rate": 0.04, "max_depth": 5, "num_leaves": 24, "min_child_samples": 15, "reg_lambda": 8.0, "colsample_bytree": 0.55},
            {"n_estimators": 550, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 12.0, "colsample_bytree": 0.45, "objective": "huber"},
        ]
        xgb_cfgs = [{"n_estimators": 450, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 6.0, "reg_lambda": 10.0, "colsample_bytree": 0.55}]
    else:
        lgb_cfgs = [
            {"n_estimators": 550, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 8.0, "colsample_bytree": 0.5},
            {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 7, "num_leaves": 48, "min_child_samples": 25, "reg_lambda": 12.0, "colsample_bytree": 0.45},
        ]
        xgb_cfgs = [{"n_estimators": 500, "learning_rate": 0.04, "max_depth": 6, "min_child_weight": 8.0, "reg_lambda": 10.0, "colsample_bytree": 0.5}]

    for cfg in lgb_cfgs:
        cands.append({"mode": "pca_gbdt", "family": "LightGBM", "params": cfg, "winsor_cap": winsor_cap})
    for cfg in xgb_cfgs:
        cands.append({"mode": "pca_gbdt", "family": "XGBoost", "params": cfg, "winsor_cap": winsor_cap})
    return cands


def benchmark_rep_controlled(
    bank: Node23FusionBank3D,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    rep_cfg: Dict[str, Any],
    n_jobs: int = 4,
) -> Dict[str, Any]:
    cv_rows, ho_rows = ep_data["cv_idx"], ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)
    cidx_cv, cidx_ho = ep_data["compound_idx"][cv_rows], ep_data["compound_idx"][ho_rows]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
    y_ho, yl_ho, yu_ho, ct_ho = ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_rows]
    cond_ho = [ep_data["cond_rows"][i] for i in ho_rows]
    bounds = spec.get("bounds")
    winsor_cap = spec.get("outlier_winsor_cap")

    valid_cv = bank.smiles_valid[cidx_cv].copy() if rep_cfg["track"] == "smiles_only" else np.ones(n_cv, dtype=bool)
    valid_ho = bank.smiles_valid[cidx_ho].copy() if rep_cfg["track"] == "smiles_only" else np.ones(n_ho, dtype=bool)

    cands = _get_candidates(spec["size_tier"], n_cv, winsor_cap=winsor_cap)
    bal_ids = [i for i, c in enumerate(cands) if c["mode"] == "balanced"]
    pca_ids = [i for i, c in enumerate(cands) if c["mode"] in ("pca_gbdt", "pca_mlp")]
    oofs = [np.full(n_cv, np.nan, dtype=np.float64) for _ in cands]
    errors: Dict[int, str] = {}
    needs_d2 = any(cands[i]["family"] in ("KRR_RBF", "SVR_RBF") for i in bal_ids)

    for f in range(5):
        va_m = (fids_cv == f) & valid_cv
        tr_m = (fids_cv != f) & valid_cv & (ct_cv == "exact")
        if tr_m.sum() < 4 or va_m.sum() == 0:
            continue
        tr_pos, va_pos = np.where(tr_m)[0], np.where(va_m)[0]
        y_tr = y_cv[tr_pos]

        if bal_ids:
            pre_b = AugmentedFoldPreprocessor(
                blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
                mode="balanced", pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f,
            )
            X_tr = pre_b.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            X_va = pre_b.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])

            D2_tr = D2_va = base_gamma = None
            if needs_d2:
                use_f32 = len(tr_pos) > 3500
                D2_tr = compute_sqdist_matrix(X_tr, X_tr, use_float32=use_f32)
                D2_va = compute_sqdist_matrix(X_va, X_tr, use_float32=use_f32)
                if len(tr_pos) > 1500:
                    sub = np.random.default_rng(42 + f).choice(len(tr_pos), size=1000, replace=False)
                    tri = D2_tr[np.ix_(sub, sub)][np.triu_indices(1000, k=1)]
                else:
                    tri = D2_tr[np.triu_indices(len(tr_pos), k=1)]
                med_d2 = float(np.median(tri[tri > 0])) if (tri > 0).any() else float(max(1, X_tr.shape[1]))
                base_gamma = 1.0 / max(med_d2, 1e-6)

            for i in bal_ids:
                c = cands[i]
                try:
                    est = make_fusion_estimator(c["family"], c["params"], "regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=c.get("winsor_cap"), seed=42 + f)
                    inner = est.est if isinstance(est, BoundedWrapper) else est
                    if D2_tr is not None and isinstance(inner, (ExactRBFKRR, PrecomputedRBFSVR)):
                        y_fit = np.clip(y_tr, None, est.winsor_cap) if (isinstance(est, BoundedWrapper) and est.winsor_cap is not None) else y_tr
                        inner.fit(X_tr, y_fit, D2_pre=D2_tr, base_gamma=base_gamma)
                        oofs[i][va_pos] = _predict_single(est, X_va, D2_star=D2_va)
                    else:
                        est.fit(X_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, X_va)
                except Exception as exc:
                    errors[i] = str(exc)
            del X_tr, X_va, D2_tr, D2_va

        if pca_ids:
            pre_p = AugmentedFoldPreprocessor(
                blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
                mode="pca_gbdt", pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f,
            )
            Xp_tr = pre_p.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            Xp_va = pre_p.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])
            Xm_tr = Xm_va = None
            if any(cands[i]["mode"] == "pca_mlp" for i in pca_ids):
                mu = Xp_tr.mean(axis=0)
                sd = Xp_tr.std(axis=0)
                sd[sd < 1e-6] = 1.0
                Xm_tr = np.clip((Xp_tr - mu) / sd, -8.0, 8.0).astype(np.float32)
                Xm_va = np.clip((Xp_va - mu) / sd, -8.0, 8.0).astype(np.float32)
            for i in pca_ids:
                c = cands[i]
                try:
                    est = make_fusion_estimator(c["family"], c["params"], "regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=c.get("winsor_cap"), seed=42 + f)
                    if c["mode"] == "pca_mlp":
                        est.fit(Xm_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xm_va)
                    else:
                        est.fit(Xp_tr, y_tr)
                        oofs[i][va_pos] = _predict_single(est, Xp_va)
                except Exception as exc:
                    errors[i] = str(exc)
            del Xp_tr, Xp_va, Xm_tr, Xm_va

    valid_res = []
    for i, c in enumerate(cands):
        if i in errors:
            continue
        m = regression_metrics(y_cv[valid_cv], oofs[i][valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv])
        valid_res.append({"recipe": c, "metrics": m, "score": _score_reg(m), "oof_pred": oofs[i]})
    valid_res.sort(key=lambda x: x["score"], reverse=True)

    def _refit_ho(recipe: Dict[str, Any]) -> Dict[str, Any]:
        oof_p = np.full(n_cv, np.nan, dtype=np.float64)
        ho_vpos = np.where(valid_ho)[0]
        ho_cond_v = [cond_ho[i] for i in ho_vpos]
        ho_folds = []
        for f in range(5):
            va_m = (fids_cv == f) & valid_cv
            tr_m = (fids_cv != f) & valid_cv & (ct_cv == "exact")
            if tr_m.sum() < 4 or va_m.sum() == 0:
                continue
            tr_pos, va_pos = np.where(tr_m)[0], np.where(va_m)[0]
            pre = AugmentedFoldPreprocessor(
                blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
                mode=recipe["mode"], pca_group_dims=rep_cfg["pca_group_dims"], random_state=42 + f,
            )
            X_tr = pre.fit_transform(bank, cidx_cv[tr_pos], [cond_cv[i] for i in tr_pos])
            X_va = pre.transform(bank, cidx_cv[va_pos], [cond_cv[i] for i in va_pos])
            est = make_fusion_estimator(recipe["family"], recipe["params"], "regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=recipe.get("winsor_cap"), seed=42 + f)
            est.fit(X_tr, y_cv[tr_pos])
            oof_p[va_pos] = _predict_single(est, X_va)
            if len(ho_vpos) > 0:
                ho_folds.append(_predict_single(est, pre.transform(bank, cidx_ho[ho_vpos], cond_rows=ho_cond_v)))
        ho_p = np.full(n_ho, np.nan, dtype=np.float64)
        if ho_folds:
            ho_p[ho_vpos] = np.mean(np.stack(ho_folds, axis=0), axis=0)
        return {
            "recipe": recipe,
            "rep_cfg": rep_cfg,
            "oof_pred": oof_p,
            "ho_pred": ho_p,
            "cv_metrics": regression_metrics(y_cv[valid_cv], oof_p[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv]),
            "ho_metrics": regression_metrics(y_ho[valid_ho], ho_p[valid_ho], yl_ho[valid_ho], yu_ho[valid_ho], ct_ho[valid_ho]),
        }

    best_rec = valid_res[0]["recipe"]
    best_all = _refit_ho(best_rec)
    sm = [r for r in valid_res if r["recipe"]["family"] not in ("LightGBM", "XGBoost")]
    tr = [r for r in valid_res if r["recipe"]["family"] in ("LightGBM", "XGBoost")]
    return {
        "best_overall": best_all,
        "best_smooth": (best_all if sm[0]["recipe"] == best_rec else _refit_ho(sm[0]["recipe"])) if sm else None,
        "best_tree": (best_all if tr[0]["recipe"] == best_rec else _refit_ho(tr[0]["recipe"])) if tr else None,
    }


def refit_leave_case17_out(
    bank: Node23FusionBank3D,
    ep_data: Dict[str, Any],
    spec: Dict[str, Any],
    head_obj: Dict[str, Any],
    c17_idx: np.ndarray,
    n_jobs: int = 4,
) -> np.ndarray:
    keep_rows = np.where(ep_data["case17_excluded"] == 0)[0]
    cidx = ep_data["compound_idx"][keep_rows]
    c17_set = set(int(x) for x in c17_idx)
    assert all(int(x) not in c17_set for x in cidx), "Case17 leakage!"
    clids = ep_data["cluster_ids"][keep_rows]
    y, ct = ep_data["y_value"][keep_rows], ep_data["censor_type"][keep_rows]
    conds = [ep_data["cond_rows"][i] for i in keep_rows]
    rep_cfg, recipe = head_obj["rep_cfg"], head_obj["recipe"]
    valid = bank.smiles_valid[cidx].copy() if rep_cfg["track"] == "smiles_only" else np.ones(len(cidx), dtype=bool)
    bounds = spec.get("bounds")

    cl_to_pos: Dict[str, List[int]] = {}
    for i, c in enumerate(clids):
        cl_to_pos.setdefault(str(c), []).append(i)
    fids = np.zeros(len(clids), dtype=np.int32)
    fcnt = [0] * 5
    for _, plist in sorted(cl_to_pos.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        bf = int(np.argmin(fcnt))
        for p in plist:
            fids[p] = bf
        fcnt[bf] += len(plist)

    canon = spec["canonical_conditions"]
    preds = []
    for f in range(5):
        tr_m = (fids != f) & valid & (ct == "exact")
        if tr_m.sum() < 4:
            continue
        tr_pos = np.where(tr_m)[0]
        pre = AugmentedFoldPreprocessor(
            blocks=rep_cfg["blocks"], phys_mode=rep_cfg["phys_mode"], cond_cols=ep_data["cond_cols"],
            mode=recipe["mode"], pca_group_dims=rep_cfg["pca_group_dims"], random_state=142 + f,
        )
        X_tr = pre.fit_transform(bank, cidx[tr_pos], [conds[i] for i in tr_pos])
        est = make_fusion_estimator(recipe["family"], recipe["params"], "regression", n_jobs=n_jobs, bounds=bounds, winsor_cap=recipe.get("winsor_cap"), seed=142 + f)
        est.fit(X_tr, y[tr_pos])
        preds.append(_predict_single(est, pre.transform(bank, c17_idx, default_cond=canon)))
    return np.mean(np.stack(preds, axis=0), axis=0)


def run_nested_stack(
    head_names: List[str],
    oof_map: Dict[str, np.ndarray],
    ho_map: Dict[str, np.ndarray],
    y_cv: np.ndarray, yl_cv: np.ndarray, yu_cv: np.ndarray, ct_cv: np.ndarray, fids_cv: np.ndarray,
    y_ho: np.ndarray, yl_ho: np.ndarray, yu_ho: np.ndarray, ct_ho: np.ndarray,
    bounds: Optional[Tuple[float, float]],
    eval_cv: np.ndarray, eval_ho: np.ndarray,
) -> Dict[str, Any]:
    n_cv, n_ho = len(y_cv), len(y_ho)
    P_oof = np.column_stack([oof_map[h] for h in head_names])
    P_ho = np.column_stack([ho_map[h] for h in head_names])
    meta_cands = [
        {"method": "best_single", "params": {}},
        {"method": "topk_blend", "params": {"k": 2}},
        {"method": "topk_blend", "params": {"k": 3}},
        {"method": "convex_simplex", "params": {"reg": 0.0}},
        {"method": "convex_simplex", "params": {"reg": 0.01}},
        {"method": "convex_simplex", "params": {"reg": 0.05}},
        {"method": "convex_simplex", "params": {"reg": 0.15}},
    ]
    best_cfg, best_sc, best_nested = None, (-1e12, -1e12), None
    for mc in meta_cands:
        nested = np.full(n_cv, np.nan, dtype=np.float64)
        for f in range(5):
            va = (fids_cv == f) & eval_cv
            tr = (fids_cv != f) & eval_cv & (ct_cv == "exact")
            if tr.sum() < 4 or va.sum() == 0:
                continue
            st = ModalityAwareStacker("regression", mc["method"], mc["params"], bounds=bounds)
            st.fit(P_oof[tr], y_cv[tr], head_names=head_names)
            nested[va] = st.predict(P_oof[va])
        m = regression_metrics(y_cv[eval_cv], nested[eval_cv], yl_cv[eval_cv], yu_cv[eval_cv], ct_cv[eval_cv])
        sc = _score_reg(m)
        if sc > best_sc:
            best_sc, best_cfg, best_nested = sc, mc, nested

    fit_m = eval_cv & (ct_cv == "exact")
    final_st = ModalityAwareStacker("regression", best_cfg["method"], best_cfg["params"], bounds=bounds)
    final_st.fit(P_oof[fit_m], y_cv[fit_m], head_names=head_names)
    ho_pred = np.full(n_ho, np.nan, dtype=np.float64)
    if eval_ho.any():
        ho_pred[eval_ho] = final_st.predict(P_ho[eval_ho])
    return {
        "best_cfg": best_cfg,
        "stacker": final_st,
        "weights": final_st.weight_dict(),
        "active_heads": final_st.active_heads(),
        "nested_oof": best_nested,
        "ho_pred": ho_pred,
        "cv_metrics": regression_metrics(y_cv[eval_cv], best_nested[eval_cv], yl_cv[eval_cv], yu_cv[eval_cv], ct_cv[eval_cv]),
        "ho_metrics": regression_metrics(y_ho[eval_ho], ho_pred[eval_ho], yl_ho[eval_ho], yu_ho[eval_ho], ct_ho[eval_ho]),
    }


def run_one_endpoint(args: Tuple[str, int]) -> Tuple[str, Dict[str, Any]]:
    ep, n_jobs = args
    te = time.time()
    bank = GLOBAL_BANK
    assert bank is not None
    spec = ALL_SPECS[ep]
    bounds = spec.get("bounds")
    ep_data = bank.load_endpoint_with_baselines(ep)
    cv_rows, ho_rows = ep_data["cv_idx"], ep_data["ho_idx"]
    y_cv, yl_cv, yu_cv, ct_cv, fids_cv = (
        ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows],
        ep_data["censor_type"][cv_rows], ep_data["fold_id"][cv_rows],
    )
    y_ho, yl_ho, yu_ho, ct_ho = (
        ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows],
    )
    cidx_cv, cidx_ho = ep_data["compound_idx"][cv_rows], ep_data["compound_idx"][ho_rows]
    smi_cv, smi_ho = bank.smiles_valid[cidx_cv], bank.smiles_valid[cidx_ho]
    dual_cv, dual_ho = np.ones(len(cv_rows), dtype=bool), np.ones(len(ho_rows), dtype=bool)
    cov_3d = round(100.0 * float(bank.valid_3d[ep_data["compound_idx"]].mean()), 1)
    c17_idx = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)

    ds_base = DIRECT_FUSION_REPS["direct_smiles"]
    dd_base = DIRECT_FUSION_REPS["direct_dual"]
    rep_cfgs = {
        "chameleonic_3d_only": {
            "name": "chameleonic_3d_only",
            "track": "smiles_only",
            "blocks": [("chameleonic_3d", 0.90)],
            "phys_mode": "smiles48",
            "pca_group_dims": [("chameleonic_3d", 29)],
        },
        "direct_smiles_base": dict(ds_base),
        "direct_smiles_plus_3d": {
            "name": "direct_smiles_plus_3d",
            "track": "smiles_only",
            "blocks": list(ds_base["blocks"]) + [("chameleonic_3d", 0.55)],
            "phys_mode": "smiles48",
            "pca_group_dims": list(ds_base["pca_group_dims"]) + [("chameleonic_3d", 29)],
        },
        "direct_dual_base": dict(dd_base),
        "direct_dual_plus_3d": {
            "name": "direct_dual_plus_3d",
            "track": "dual_modality",
            "blocks": list(dd_base["blocks"]) + [("chameleonic_3d", 0.50)],
            "phys_mode": "dual83",
            "pca_group_dims": list(dd_base["pca_group_dims"]) + [("chameleonic_3d", 29)],
        },
    }

    trained: Dict[str, Dict[str, Any]] = {}
    ablation: Dict[str, Any] = {}
    for rname, rcfg in rep_cfgs.items():
        res = benchmark_rep_controlled(bank, ep_data, spec, rcfg, n_jobs=n_jobs)
        bo = res["best_overall"]
        trained[rname] = bo
        if res["best_smooth"] is not None:
            trained[f"{rname}_smooth"] = res["best_smooth"]
        if res["best_tree"] is not None:
            trained[f"{rname}_tree"] = res["best_tree"]
        ablation[rname] = {"winning_family": bo["recipe"]["family"], "cv_metrics": bo["cv_metrics"], "ho_metrics": bo["ho_metrics"]}

    off = ep_data["official"]
    base_fus_cv_dual = regression_metrics(y_cv, off["fus_oof_dual"], yl_cv, yu_cv, ct_cv)
    base_fus_ho_dual = regression_metrics(y_ho, off["fus_ho_dual"], yl_ho, yu_ho, ct_ho)
    base_fus_cv_smi = regression_metrics(y_cv[smi_cv], off["fus_oof_smi"][smi_cv], yl_cv[smi_cv], yu_cv[smi_cv], ct_cv[smi_cv])
    base_fus_ho_smi = regression_metrics(y_ho[smi_ho], off["fus_ho_smi"][smi_ho], yl_ho[smi_ho], yu_ho[smi_ho], ct_ho[smi_ho])

    oof_map = {"official_fusion_dual": off["fus_oof_dual"], "official_fusion_smi": off["fus_oof_smi"]}
    ho_map = {"official_fusion_dual": off["fus_ho_dual"], "official_fusion_smi": off["fus_ho_smi"]}
    for k, obj in trained.items():
        oof_map[k], ho_map[k] = obj["oof_pred"], obj["ho_pred"]

    heads_dual_3d = [h for h in ("official_fusion_dual", "chameleonic_3d_only", "direct_dual_plus_3d_smooth", "direct_dual_plus_3d_tree", "direct_smiles_plus_3d_smooth") if h in oof_map]
    st_dual_3d = run_nested_stack(heads_dual_3d, oof_map, ho_map, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, dual_cv, dual_ho)

    heads_smi_3d = [h for h in ("official_fusion_smi", "chameleonic_3d_only", "direct_smiles_plus_3d_smooth", "direct_smiles_plus_3d_tree") if h in oof_map]
    st_smi_3d = run_nested_stack(heads_smi_3d, oof_map, ho_map, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, y_ho, yl_ho, yu_ho, ct_ho, bounds, smi_cv, smi_ho)

    # Strict Leave-17-Cases-Out (inactive heads cleanly padded with NaN, ignored by ModalityAwareStacker!)
    c17_map = {"official_fusion_dual": off["fus_c17_dual"], "official_fusion_smi": off["fus_c17_smi"]}
    needed_c17 = (set(st_dual_3d["active_heads"]) | set(st_smi_3d["active_heads"])) - set(c17_map.keys())
    for h in sorted(needed_c17):
        if h in trained:
            c17_map[h] = refit_leave_case17_out(bank, ep_data, spec, trained[h], c17_idx, n_jobs=n_jobs)
    nan_c17 = np.full(len(c17_idx), np.nan, dtype=np.float64)
    c17_dual_3d = st_dual_3d["stacker"].predict(np.column_stack([c17_map.get(h, nan_c17) for h in heads_dual_3d]))
    c17_smi_3d = st_smi_3d["stacker"].predict(np.column_stack([c17_map.get(h, nan_c17) for h in heads_smi_3d]))

    c17_records = []
    for i, q in enumerate(bank.case17_queries):
        c17_records.append({
            "drug_name": q["drug_name"],
            "valid_3d": bool(bank.valid_3d[q["compound_idx"]]),
            "baseline_fusion_dual": round(float(off["fus_c17_dual"][i]), 5) if np.isfinite(off["fus_c17_dual"][i]) else None,
            "fusion_plus_3d_dual": round(float(c17_dual_3d[i]), 5) if np.isfinite(c17_dual_3d[i]) else None,
            "baseline_fusion_smi": round(float(off["fus_c17_smi"][i]), 5) if np.isfinite(off["fus_c17_smi"][i]) else None,
            "fusion_plus_3d_smi": round(float(c17_smi_3d[i]), 5) if np.isfinite(c17_smi_3d[i]) else None,
        })

    dd_base_ho = ablation["direct_dual_base"]["ho_metrics"]["mae"]
    dd_3d_ho = ablation["direct_dual_plus_3d"]["ho_metrics"]["mae"]
    ds_base_ho = ablation["direct_smiles_base"]["ho_metrics"]["mae"]
    ds_3d_ho = ablation["direct_smiles_plus_3d"]["ho_metrics"]["mae"]

    doc = {
        "endpoint": ep,
        "n_cv": len(cv_rows),
        "n_holdout": len(ho_rows),
        "coverage_3d_pct": cov_3d,
        "direct_ablation": ablation,
        "direct_delta_dual_ho_mae": round(dd_3d_ho - dd_base_ho, 5),
        "direct_delta_smi_ho_mae": round(ds_3d_ho - ds_base_ho, 5),
        "baseline_4fm_fusion": {"dual_cv": base_fus_cv_dual, "dual_ho": base_fus_ho_dual, "smi_cv": base_fus_cv_smi, "smi_ho": base_fus_ho_smi},
        "fusion_plus_3d_chameleonic": {
            "dual_weights": st_dual_3d["weights"],
            "dual_cv": st_dual_3d["cv_metrics"],
            "dual_ho": st_dual_3d["ho_metrics"],
            "delta_dual_cv_mae": round(st_dual_3d["cv_metrics"]["mae"] - base_fus_cv_dual["mae"], 5),
            "delta_dual_ho_mae": round(st_dual_3d["ho_metrics"]["mae"] - base_fus_ho_dual["mae"], 5),
            "delta_dual_ho_spearman": round((st_dual_3d["ho_metrics"]["spearman"] or 0.0) - (base_fus_ho_dual["spearman"] or 0.0), 5),
            "smi_weights": st_smi_3d["weights"],
            "smi_cv": st_smi_3d["cv_metrics"],
            "smi_ho": st_smi_3d["ho_metrics"],
            "delta_smi_ho_mae": round(st_smi_3d["ho_metrics"]["mae"] - base_fus_ho_smi["mae"], 5),
        },
        "fusion_plus_3d_and_unimol2": None,
        "case17_leaveout": c17_records,
        "elapsed_sec": round(time.time() - te, 1),
    }
    print(
        f"[DONE] {ep:<15} (cov={cov_3d:5.1f}%, {doc['elapsed_sec']:5.1f}s) | "
        f"Direct HO {dd_base_ho:.4f}->{dd_3d_ho:.4f} ({dd_3d_ho - dd_base_ho:+.4f}) | "
        f"Stack Base {base_fus_ho_dual['mae']:.4f}->+3D {st_dual_3d['ho_metrics']['mae']:.4f} "
        f"({st_dual_3d['ho_metrics']['mae'] - base_fus_ho_dual['mae']:+.4f}) "
        f"Sp {base_fus_ho_dual['spearman']:.3f}->{st_dual_3d['ho_metrics']['spearman']:.3f}",
        flush=True,
    )
    return ep, doc


def main() -> None:
    global GLOBAL_BANK
    t_all = time.time()
    ext_meta = json.loads((OUT_3D_DIR / "extraction_meta.json").read_text(encoding="utf-8"))
    GLOBAL_BANK = Node23FusionBank3D()
    print(f"[NODE23 3D BENCHMARK] Loaded 50,995-compound bank + 3D Chameleonic ({int(GLOBAL_BANK.valid_3d.sum()):,} valid). Launching 11 endpoints in parallel...", flush=True)

    from concurrent.futures import as_completed
    jobs = [(ep, 8 if ep == "PAMPA" else 4) for ep in TIER1_11_ENDPOINTS]
    ep_results: Dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=11) as pool:
        futs = [pool.submit(run_one_endpoint, j) for j in jobs]
        for fut in as_completed(futs):
            ep, doc = fut.result()
            ep_results[ep] = doc

    def _m(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return round(float(np.mean(v)), 5) if v else None

    agg = {
        "n_endpoints": len(TIER1_11_ENDPOINTS),
        "direct_dual_base_mean_ho_mae": _m([ep_results[e]["direct_ablation"]["direct_dual_base"]["ho_metrics"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "direct_dual_plus_3d_mean_ho_mae": _m([ep_results[e]["direct_ablation"]["direct_dual_plus_3d"]["ho_metrics"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "direct_dual_3d_improved_count": sum(1 for e in TIER1_11_ENDPOINTS if ep_results[e]["direct_delta_dual_ho_mae"] < -1e-6),
        "direct_smi_base_mean_ho_mae": _m([ep_results[e]["direct_ablation"]["direct_smiles_base"]["ho_metrics"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "direct_smi_plus_3d_mean_ho_mae": _m([ep_results[e]["direct_ablation"]["direct_smiles_plus_3d"]["ho_metrics"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "direct_smi_3d_improved_count": sum(1 for e in TIER1_11_ENDPOINTS if ep_results[e]["direct_delta_smi_ho_mae"] < -1e-6),
        "stack_baseline_4fm_mean_ho_mae": _m([ep_results[e]["baseline_4fm_fusion"]["dual_ho"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "stack_plus_3d_mean_ho_mae": _m([ep_results[e]["fusion_plus_3d_chameleonic"]["dual_ho"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "stack_baseline_4fm_mean_ho_spearman": _m([ep_results[e]["baseline_4fm_fusion"]["dual_ho"]["spearman"] for e in TIER1_11_ENDPOINTS]),
        "stack_plus_3d_mean_ho_spearman": _m([ep_results[e]["fusion_plus_3d_chameleonic"]["dual_ho"]["spearman"] for e in TIER1_11_ENDPOINTS]),
        "stack_plus_3d_improved_count": sum(1 for e in TIER1_11_ENDPOINTS if ep_results[e]["fusion_plus_3d_chameleonic"]["delta_dual_ho_mae"] < -1e-6),
        "stack_plus_3d_smi_base_mean_ho_mae": _m([ep_results[e]["baseline_4fm_fusion"]["smi_ho"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "stack_plus_3d_smi_new_mean_ho_mae": _m([ep_results[e]["fusion_plus_3d_chameleonic"]["smi_ho"]["mae"] for e in TIER1_11_ENDPOINTS]),
        "stack_plus_3d_uni_mean_ho_mae": None,
    }
    summary = {
        "schema": "peptide_omnipanel_3d_chameleonic_summary_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "compute_host": "node23 (64-core CPU, 125 GB RAM — 0% local Windows load)",
        "endpoints_evaluated": TIER1_11_ENDPOINTS,
        "extraction_meta": ext_meta,
        "aggregate": agg,
        "endpoints": ep_results,
        "total_elapsed_sec": round(time.time() - t_all, 1),
    }
    sp = OUT_3D_DIR / "chameleonic_3d_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== NODE23 UPGRADE 2 BENCHMARK COMPLETE -> {sp} ({summary['total_elapsed_sec']}s) ===", flush=True)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
