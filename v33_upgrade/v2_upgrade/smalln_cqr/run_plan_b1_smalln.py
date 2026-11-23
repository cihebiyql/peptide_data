#!/usr/bin/env python3
"""Plan B-1 Deliverable 1 (runs on node23 64-core CPU ONLY):
Small-N Foundation/Shrinkage Modeling Suite (TabPFN v2 + Multi-Block Cosine/Tanimoto/RBF Composite GPR
+ Empirical-Bayes Shrinkage Ridge / Huber / Cosine-KRR-SVR) across ALL N <= 400 regression endpoints.

STRICT ZERO-LEAKAGE & ZERO-HOLDOUT-PEEKING GUARANTEES:
1. Read-only access to /data/qlyu/v33_upgrade/{data,features,embeddings,models_cpu,models_fusion,models_gpu_deep,expansion_v331,bundle}/.
2. Writes ONLY under /data/qlyu/v33_upgrade/v2_upgrade/smalln_cqr/.
3. Exact frozen 5-fold cluster splits (is_holdout == 0 CV, is_holdout == 1 untouched 20% holdout).
4. Promotion decision made STRICTLY on nested 5-fold CV OOF MAE (margin >= 0.002 over frozen V33.1 champion).
5. Untouched 20% holdout and Leave-Case17-Out evaluated ONCE after freezing all selections & weights.
6. Every promoted node2-portable .joblib verified to contain ZERO catboost / tabpfn / torch references.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
from scipy.optimize import minimize

V33_BASE = Path("/data/qlyu/v33_upgrade")
EXP_BASE = V33_BASE / "expansion_v331"
BUNDLE_DIR = V33_BASE / "bundle"
OUT_DIR = V33_BASE / "v2_upgrade" / "smalln_cqr"
TABPFN_CKPT = Path("/data/qlyu/.cache/tabpfn/tabpfn-v2-regressor.ckpt")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))

from v33_config import CASE17_CLINICAL_ROUTE, CASE17_FDA_NUMERIC, ENDPOINT_SPECS as CORE_SPECS  # noqa: E402
from v33_metrics import regression_metrics  # noqa: E402
from smalln_models import (  # noqa: E402
    CompactTabularPreprocessor,
    EBHuberRegressor,
    MultiBlockCompositeGPR,
    MultiBlockKernelPreprocessor,
    MultiBlockShrinkageKRRSVR,
    PortableSmallNModelBundle,
)

csv.field_size_limit(10_000_000)

PK_SMALLN_ENDPOINTS = {"CL_absolute", "Vd_absolute", "Cmax", "F", "PPB", "Vd_per_kg", "CL_per_kg"}
ALT_C17_CONDS: Dict[str, Dict[str, Dict[str, str]]] = {
    "CL_absolute": {"SC_CL_over_F": {"cond_species": "human", "cond_route": "SC", "cond_estimand": "CL_over_F"}},
    "F": {"SC": {"cond_species": "human", "cond_route": "SC"}},
    "Vd_absolute": {"SC_V_F": {"cond_species": "human", "cond_route": "SC", "cond_estimand": "V_F"}},
}

GLOBAL_BANK: Optional["UnifiedSmallNBank"] = None
GLOBAL_MANIFEST: Optional[Dict[str, Any]] = None
TABPFN_AVAILABLE: bool = False
TABPFN_STATUS_MSG: str = ""


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


class UnifiedSmallNBank:
    """Read-only unified 50,995-compound feature + 4-FM embedding bank (CoW-shared across forked workers)."""

    def __init__(self):
        t0 = time.time()
        fdir = EXP_BASE / "features"
        edir = EXP_BASE / "embeddings"
        fman = json.loads((fdir / "feature_manifest.json").read_text(encoding="utf-8"))
        self.n_total = int(fman["n_compounds_total"])
        self.case17_indices: Dict[str, int] = fman["case17_indices"]
        self.key_to_idx: Dict[str, int] = {}
        with (fdir / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.key_to_idx[r["compound_key"]] = int(r["compound_idx"])

        pc = np.load(fdir / "physchem_core.npz")
        self.physchem = np.ascontiguousarray(pc["X"], dtype=np.float32)
        self.smiles_valid = np.ascontiguousarray(pc["smiles_valid_mask"], dtype=bool)
        self.seq_valid = np.ascontiguousarray(pc["seq_valid_mask"], dtype=bool)
        pc.close()

        mb = np.load(fdir / "morgan_bits_1024.npz")
        self.morgan_bits = np.ascontiguousarray(mb["morgan_bits_1024"], dtype=np.uint8)
        self.morgan_packed = np.ascontiguousarray(mb["morgan_bits_packed"], dtype=np.uint8)
        mb.close()

        self.emb_blocks: Dict[str, np.ndarray] = {}
        self.emb_valid: Dict[str, np.ndarray] = {}

        d_nc = np.load(edir / "ncaa_repdistill_v2.npz")
        v_nc = np.ascontiguousarray(d_nc["valid"], dtype=bool)
        for k, bk in [("smiles_repr", "ncaa_smiles_repr"), ("fusion_hf", "ncaa_fusion_hf"), ("protein_proj", "ncaa_protein_proj")]:
            self.emb_blocks[bk] = np.ascontiguousarray(d_nc[k], dtype=np.float16)
            self.emb_valid[bk] = v_nc
        d_nc.close()

        d_pc = np.load(edir / "peptideclm2.npz")
        v_pc = np.ascontiguousarray(d_pc["valid"], dtype=bool)
        self.emb_blocks["peptideclm2"] = np.ascontiguousarray(d_pc["embeddings"], dtype=np.float16)
        self.emb_valid["peptideclm2"] = v_pc
        d_pc.close()

        d_cb = np.load(edir / "chemberta_mtr.npz")
        v_cb = np.ascontiguousarray(d_cb["valid"], dtype=bool)
        self.emb_blocks["chemberta_mtr"] = np.ascontiguousarray(d_cb["embeddings"], dtype=np.float16)
        self.emb_valid["chemberta_mtr"] = v_cb
        d_cb.close()

        d_es = np.load(edir / "esmc_600m_multiregion.npz")
        v_es = np.ascontiguousarray(d_es["valid"], dtype=bool)
        for k, bk in [("mean", "esmc_mean"), ("max", "esmc_max"), ("nterm5", "esmc_nterm5"), ("cterm5", "esmc_cterm5")]:
            self.emb_blocks[bk] = np.ascontiguousarray(d_es[k], dtype=np.float16)
            self.emb_valid[bk] = v_es
        d_es.close()

        self.case17_queries = _read_tsv(V33_BASE / "data" / "case17_queries.tsv")
        for q in self.case17_queries:
            q["compound_idx"] = self.case17_indices[q["drug_name"]]
        self.case17_idx = np.array([q["compound_idx"] for q in self.case17_queries], dtype=np.int32)
        print(f"[bank] Loaded unified {self.n_total:,}-compound 4-FM bank in {time.time()-t0:.2f}s", flush=True)

    def map_keys(self, keys: List[str]) -> np.ndarray:
        return np.array([self.key_to_idx[k] for k in keys], dtype=np.int32)

    def get_emb_f32(self, block_name: str, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.emb_blocks[block_name][idx].astype(np.float32),
            self.emb_valid[block_name][idx].copy(),
        )


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
        out[miss] = np.divide(M0[miss].sum(axis=1), cnt, out=np.full(int(miss.sum()), fallback), where=cnt > 0)
    return out


def _apply_simplex_blend(
    names: List[str],
    streams: Dict[str, np.ndarray],
    weights: Dict[str, float],
    bounds: Optional[Tuple[float, float]] = None,
    fallback: float = 0.0,
) -> np.ndarray:
    active = [s for s in names if float(weights.get(s, 0.0)) > 0.0 and s in streams]
    if not active:
        active = [names[0]]
    w = np.array([float(weights[s]) for s in active], dtype=np.float64)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(active)) / len(active)
    M = np.column_stack([streams[s] for s in active])
    y_out = _weighted_nan_combine(M, w, fallback=fallback)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


def _fit_simplex_weights_l1(
    P: np.ndarray,
    y: np.ndarray,
    reg: float = 0.02,
    prune_tol: float = 0.03,
) -> np.ndarray:
    """Fit strongly regularized simplex weights minimizing smooth L1 MAE (anti-overfit for small-N)."""
    n, m = P.shape
    if m == 1:
        return np.array([1.0], dtype=np.float64)
    unif = np.ones(m, dtype=np.float64) / float(m)
    fallback = float(np.nanmean(y)) if len(y) > 0 else 0.0
    eps_s = 1e-3

    def loss_fn(theta: np.ndarray) -> float:
        lw = theta - np.max(theta)
        w = np.exp(lw)
        w = w / np.sum(w)
        pred = _weighted_nan_combine(P, w, fallback=fallback)
        mae_s = float(np.mean(np.sqrt((y - pred) ** 2 + eps_s ** 2)))
        return mae_s + reg * float(np.sum((w - unif) ** 2))

    inits = [np.zeros(m, dtype=np.float64)]
    for j in range(m):
        v = np.full(m, -2.5, dtype=np.float64)
        v[j] = 2.5
        inits.append(v)

    best_res = None
    for x0 in inits:
        res = minimize(loss_fn, x0, method="BFGS", options={"maxiter": 120})
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


def _fit_2way_alpha_l1(p_base: np.ndarray, p_new: np.ndarray, y: np.ndarray, reg: float = 0.005) -> float:
    """1D grid + refine convex weight w_new in [0, 1] minimizing MAE + tiny ridge toward 0."""
    grid = np.linspace(0.0, 1.0, 41)
    best_w = 0.0
    best_v = float("inf")
    fin = np.isfinite(p_base) & np.isfinite(p_new) & np.isfinite(y)
    if fin.sum() < 4:
        return 0.0
    yb, pb, pn = y[fin], p_base[fin], p_new[fin]
    for w in grid:
        pred = (1.0 - w) * pb + w * pn
        v = float(np.mean(np.abs(yb - pred))) + reg * (w ** 2)
        if v < best_v:
            best_v = v
            best_w = float(w)
    return best_w


def _load_endpoint_frozen(ep: str, ep_doc: Dict[str, Any], bank: UnifiedSmallNBank) -> Dict[str, Any]:
    """Load exact frozen TSV + aligned CPU/Fusion/(Deep) OOF/HO/Case17 streams & verify manifest parity."""
    wave = ep_doc.get("expansion_wave", "v33.0")
    ep_dir = EXP_BASE / "data/endpoints" if wave == "v33.1" else V33_BASE / "data/endpoints"
    cpu_dir = EXP_BASE / "models_cpu" if wave == "v33.1" else V33_BASE / "models_cpu"
    fus_dir = EXP_BASE / "models_fusion" if wave == "v33.1" else V33_BASE / "models_fusion"

    rows = _read_tsv(ep_dir / f"{ep}.tsv")
    cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
    cv_rows_i = [i for i, r in enumerate(rows) if int(r["is_holdout"]) == 0]
    ho_rows_i = [i for i, r in enumerate(rows) if int(r["is_holdout"]) == 1]

    cids_all = np.array([r["cluster_id"] for r in rows], dtype=object)
    assert set(cids_all[cv_rows_i]).isdisjoint(set(cids_all[ho_rows_i])), f"{ep}: cluster leakage!"

    cpu_oof = _read_tsv(cpu_dir / ep / "oof_predictions.tsv")
    cpu_ho = _read_tsv(cpu_dir / ep / "holdout_predictions.tsv")
    cpu_c17 = _read_tsv(cpu_dir / ep / "case17_predictions.tsv")
    fus_oof = _read_tsv(fus_dir / ep / "oof_predictions.tsv")
    fus_ho = _read_tsv(fus_dir / ep / "holdout_predictions.tsv")
    fus_c17 = _read_tsv(fus_dir / ep / "case17_predictions.tsv")
    assert len(cpu_oof) == len(fus_oof) == len(cv_rows_i)
    assert len(cpu_ho) == len(fus_ho) == len(ho_rows_i)
    for k, orig_i in enumerate(cv_rows_i):
        assert cpu_oof[k]["record_id"] == fus_oof[k]["record_id"] == rows[orig_i]["record_id"]
    for k, orig_i in enumerate(ho_rows_i):
        assert cpu_ho[k]["record_id"] == fus_ho[k]["record_id"] == rows[orig_i]["record_id"]

    existing_oof = {
        "dual_modality": {"cpu": _col_arr(cpu_oof, "pred_dual"), "fusion": _col_arr(fus_oof, "pred_dual")},
        "smiles_only": {"cpu": _col_arr(cpu_oof, "pred_smiles_only"), "fusion": _col_arr(fus_oof, "pred_smiles_only")},
    }
    existing_ho = {
        "dual_modality": {"cpu": _col_arr(cpu_ho, "pred_dual"), "fusion": _col_arr(fus_ho, "pred_dual")},
        "smiles_only": {"cpu": _col_arr(cpu_ho, "pred_smiles_only"), "fusion": _col_arr(fus_ho, "pred_smiles_only")},
    }
    existing_c17 = {
        "dual_modality": {"cpu": _col_arr(cpu_c17, "pred_dual"), "fusion": _col_arr(fus_c17, "pred_dual")},
        "smiles_only": {"cpu": _col_arr(cpu_c17, "pred_smiles_only"), "fusion": _col_arr(fus_c17, "pred_smiles_only")},
    }
    existing_c17_alt: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    dep_oof = dep_ho = dep_c17 = None
    if wave == "v33.0":
        dep_dir = V33_BASE / "models_gpu_deep" / ep
        dep_oof = _read_tsv(dep_dir / "oof_predictions.tsv")
        dep_ho = _read_tsv(dep_dir / "holdout_predictions.tsv")
        dep_c17 = _read_tsv(dep_dir / "case17_predictions.tsv")
        assert len(dep_oof) == len(cv_rows_i) and len(dep_ho) == len(ho_rows_i)
        existing_oof["dual_modality"]["deep"] = _col_arr(dep_oof, "pred_dual")
        existing_oof["smiles_only"]["deep"] = _col_arr(dep_oof, "pred_smiles_only")
        existing_ho["dual_modality"]["deep"] = _col_arr(dep_ho, "pred_dual")
        existing_ho["smiles_only"]["deep"] = _col_arr(dep_ho, "pred_smiles_only")
        existing_c17["dual_modality"]["deep"] = _col_arr(dep_c17, "pred_dual")
        existing_c17["smiles_only"]["deep"] = _col_arr(dep_c17, "pred_smiles_only")

    for alt_k in ALT_C17_CONDS.get(ep, {}):
        existing_c17_alt[alt_k] = {
            "dual_modality": {
                "cpu": _col_arr(cpu_c17, f"pred_dual_{alt_k}") if _col_arr(cpu_c17, f"pred_dual_{alt_k}") is not None else existing_c17["dual_modality"]["cpu"],
                "fusion": _col_arr(fus_c17, f"pred_dual_{alt_k}") if _col_arr(fus_c17, f"pred_dual_{alt_k}") is not None else existing_c17["dual_modality"]["fusion"],
            },
            "smiles_only": {
                "cpu": _col_arr(cpu_c17, f"pred_smiles_only_{alt_k}") if _col_arr(cpu_c17, f"pred_smiles_only_{alt_k}") is not None else existing_c17["smiles_only"]["cpu"],
                "fusion": _col_arr(fus_c17, f"pred_smiles_only_{alt_k}") if _col_arr(fus_c17, f"pred_smiles_only_{alt_k}") is not None else existing_c17["smiles_only"]["fusion"],
            },
        }
        if dep_c17 is not None:
            existing_c17_alt[alt_k]["dual_modality"]["deep"] = (
                _col_arr(dep_c17, f"pred_dual_{alt_k}") if _col_arr(dep_c17, f"pred_dual_{alt_k}") is not None else existing_c17["dual_modality"]["deep"]
            )
            existing_c17_alt[alt_k]["smiles_only"]["deep"] = (
                _col_arr(dep_c17, f"pred_smiles_only_{alt_k}") if _col_arr(dep_c17, f"pred_smiles_only_{alt_k}") is not None else existing_c17["smiles_only"]["deep"]
            )

    # Exact frozen champion OOF & HO (using bundle/champion_predictions for v33.0, manifest weights for v33.1)
    bounds_raw = ep_doc.get("bounds")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1])) if bounds_raw is not None else None
    y_all = np.array([float(r["y_value"]) for r in rows], dtype=np.float64)
    y_cv = y_all[cv_rows_i]
    y_ho = y_all[ho_rows_i]
    yl_all = np.array([float(r["y_lower"]) for r in rows], dtype=np.float64)
    yu_all = np.array([float(r["y_upper"]) for r in rows], dtype=np.float64)
    ct_all = np.array([r["censor_type"] for r in rows], dtype=object)
    cidx_all = bank.map_keys([r["compound_key"] for r in rows])

    base_champ_oof: Dict[str, np.ndarray] = {}
    base_champ_ho: Dict[str, np.ndarray] = {}
    base_champ_c17: Dict[str, np.ndarray] = {}
    base_champ_c17_alt: Dict[str, Dict[str, np.ndarray]] = {}

    if wave == "v33.0":
        coof = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "oof_champion.tsv")
        cho = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "holdout_champion.tsv")
        cc17 = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "case17_champion.tsv")
        base_champ_oof["dual_modality"] = _col_arr(coof, "pred_dual")
        base_champ_oof["smiles_only"] = _col_arr(coof, "pred_smiles_only")
        base_champ_ho["dual_modality"] = _col_arr(cho, "pred_dual")
        base_champ_ho["smiles_only"] = _col_arr(cho, "pred_smiles_only")
        base_champ_c17["dual_modality"] = _col_arr(cc17, "pred_dual")
        base_champ_c17["smiles_only"] = _col_arr(cc17, "pred_smiles_only")
        for alt_k in ALT_C17_CONDS.get(ep, {}):
            base_champ_c17_alt[alt_k] = {
                "dual_modality": _col_arr(cc17, f"pred_dual_{alt_k}") if _col_arr(cc17, f"pred_dual_{alt_k}") is not None else base_champ_c17["dual_modality"],
                "smiles_only": _col_arr(cc17, f"pred_smiles_only_{alt_k}") if _col_arr(cc17, f"pred_smiles_only_{alt_k}") is not None else base_champ_c17["smiles_only"],
            }
    else:
        for tr in ("dual_modality", "smiles_only"):
            w = ep_doc[tr]["weights"]
            fb = float(np.nanmean(y_cv))
            # For v33.1 single-family winners, OOF is exact; for super_ensemble, reconstruct via frozen weights (and verify!)
            base_champ_oof[tr] = _apply_simplex_blend(list(w.keys()), existing_oof[tr], w, bounds=bounds, fallback=fb)
            base_champ_ho[tr] = _apply_simplex_blend(list(w.keys()), existing_ho[tr], w, bounds=bounds, fallback=fb)
            base_champ_c17[tr] = _apply_simplex_blend(list(w.keys()), existing_c17[tr], w, bounds=bounds, fallback=fb)
        for alt_k in ALT_C17_CONDS.get(ep, {}):
            base_champ_c17_alt[alt_k] = {}
            for tr in ("dual_modality", "smiles_only"):
                w = ep_doc[tr]["weights"]
                base_champ_c17_alt[alt_k][tr] = _apply_simplex_blend(
                    list(w.keys()), existing_c17_alt[alt_k][tr], w, bounds=bounds, fallback=float(np.nanmean(y_cv))
                )

    return {
        "endpoint": ep,
        "wave": wave,
        "bounds": bounds,
        "cond_cols": cond_cols,
        "canonical_conditions": ep_doc["canonical_conditions"],
        "rows": rows,
        "cv_idx": np.array(cv_rows_i, dtype=np.int32),
        "ho_idx": np.array(ho_rows_i, dtype=np.int32),
        "compound_idx": cidx_all,
        "cluster_id": cids_all,
        "fold_id": np.array([int(r["fold_id"]) for r in rows], dtype=np.int32),
        "case17_excluded": np.array([int(r["case17_excluded"]) for r in rows], dtype=np.int32),
        "censor_type": ct_all,
        "y_value": y_all,
        "y_lower": yl_all,
        "y_upper": yu_all,
        "cond_rows": [{c: r.get(c, "unspecified") for c in cond_cols} for r in rows],
        "existing_oof": existing_oof,
        "existing_ho": existing_ho,
        "existing_c17": existing_c17,
        "existing_c17_alt": existing_c17_alt,
        "base_champ_oof": base_champ_oof,
        "base_champ_ho": base_champ_ho,
        "base_champ_c17": base_champ_c17,
        "base_champ_c17_alt": base_champ_c17_alt,
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
        assert forbidden not in mods, f"FORBIDDEN package '{forbidden}' found in {path}: {sorted(mods)}"
    return sorted(mods)


def _fit_predict_tabpfn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_eval_list: List[np.ndarray],
    bounds: Optional[Tuple[float, float]] = None,
    n_estimators: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run TabPFN v2 on CPU inside isolated tabpfn_venv using offline cached checkpoint."""
    import torch
    from tabpfn import TabPFNRegressor

    torch.set_num_threads(2)
    reg = TabPFNRegressor(
        model_path=str(TABPFN_CKPT),
        device="cpu",
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
        random_state=42,
    )
    reg.fit(X_tr.astype(np.float32), y_tr.astype(np.float32))
    preds: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    for Xq in X_eval_list:
        if len(Xq) == 0:
            preds.append(np.zeros(0, dtype=np.float64))
            stds.append(np.zeros(0, dtype=np.float64))
            continue
        Xq32 = Xq.astype(np.float32)
        p = np.asarray(reg.predict(Xq32), dtype=np.float64)
        if bounds is not None:
            p = np.clip(p, bounds[0], bounds[1])
        try:
            qs = reg.predict(Xq32, output_type="quantiles", quantiles=[0.10, 0.90])
            s = np.maximum((np.asarray(qs[1], dtype=np.float64) - np.asarray(qs[0], dtype=np.float64)) / 2.5631, 1e-4)
        except Exception:
            s = np.full(len(Xq32), np.nan, dtype=np.float64)
        preds.append(p)
        stds.append(s)
    return preds, stds


def _run_one_endpoint_track(
    ep_data: Dict[str, Any],
    ep_doc: Dict[str, Any],
    track: str,
    bank: UnifiedSmallNBank,
    run_tabpfn: bool,
    tabpfn_n_est: int = 8,
) -> Dict[str, Any]:
    """Run all 6 small-N learners + strict nested-OOF selection + single holdout evaluation for ONE endpoint x track."""
    ep = ep_data["endpoint"]
    bounds = ep_data["bounds"]
    cv_idx = ep_data["cv_idx"]
    ho_idx = ep_data["ho_idx"]
    n_cv, n_ho = len(cv_idx), len(ho_idx)

    cidx_cv = ep_data["compound_idx"][cv_idx]
    cidx_ho = ep_data["compound_idx"][ho_idx]
    fids_cv = ep_data["fold_id"][cv_idx]
    cids_cv = ep_data["cluster_id"][cv_idx]
    y_cv = ep_data["y_value"][cv_idx]
    yl_cv = ep_data["y_lower"][cv_idx]
    yu_cv = ep_data["y_upper"][cv_idx]
    ct_cv = ep_data["censor_type"][cv_idx]
    cond_cv = [ep_data["cond_rows"][i] for i in cv_idx]

    y_ho = ep_data["y_value"][ho_idx]
    yl_ho = ep_data["y_lower"][ho_idx]
    yu_ho = ep_data["y_upper"][ho_idx]
    ct_ho = ep_data["censor_type"][ho_idx]
    cond_ho = [ep_data["cond_rows"][i] for i in ho_idx]

    valid_cv = bank.smiles_valid[cidx_cv].copy() if track == "smiles_only" else np.ones(n_cv, dtype=bool)
    valid_ho = bank.smiles_valid[cidx_ho].copy() if track == "smiles_only" else np.ones(n_ho, dtype=bool)
    ho_vpos = np.where(valid_ho)[0]
    ho_cond_v = [cond_ho[i] for i in ho_vpos]

    portable_names = ["mkgpr_cos", "mkgpr_exp", "eb_ridge", "cosine_krr", "eb_huber"]
    all_new_names = list(portable_names) + (["tabpfn_v2"] if run_tabpfn else [])

    oof_preds: Dict[str, np.ndarray] = {m: np.full(n_cv, np.nan, dtype=np.float64) for m in all_new_names}
    ho_fold_preds: Dict[str, List[np.ndarray]] = {m: [] for m in all_new_names}
    oof_sigmas: Dict[str, np.ndarray] = {m: np.full(n_cv, np.nan, dtype=np.float64) for m in ("mkgpr_cos", "mkgpr_exp", "tabpfn_v2")}
    ho_fold_sigmas: Dict[str, List[np.ndarray]] = {m: [] for m in ("mkgpr_cos", "mkgpr_exp", "tabpfn_v2")}
    fitted_folds: Dict[str, List[Tuple[int, Any, Any]]] = {m: [] for m in portable_names}
    gpr_block_weights: Dict[str, List[Dict[str, float]]] = {"mkgpr_cos": [], "mkgpr_exp": []}
    tabpfn_dim_used: List[int] = []
    tabpfn_err: Optional[str] = None

    for f in range(5):
        va_mask = (fids_cv == f) & valid_cv
        tr_mask = (fids_cv != f) & valid_cv & (ct_cv == "exact")
        if tr_mask.sum() < 6 or va_mask.sum() == 0:
            continue
        tr_pos = np.where(tr_mask)[0]
        va_pos = np.where(va_mask)[0]
        tr_cond = [cond_cv[i] for i in tr_pos]
        va_cond = [cond_cv[i] for i in va_pos]
        y_tr = y_cv[tr_pos]
        cid_tr = cids_cv[tr_pos]

        # 1. Zero-PCA-loss Multi-Block Kernel Preprocessor (for GPR, EB-Ridge, Cosine-KRR/SVR)
        kpre = MultiBlockKernelPreprocessor(track=track, cond_cols=ep_data["cond_cols"])
        Xk_tr = kpre.fit_transform(bank, cidx_cv[tr_pos], tr_cond)
        Xk_va = kpre.transform(bank, cidx_cv[va_pos], va_cond)
        Xk_ho = kpre.transform(bank, cidx_ho[ho_vpos], ho_cond_v) if len(ho_vpos) > 0 else None

        # (1a) Multi-Block Cosine+Tanimoto+RBF Composite GPR
        gpr_c = MultiBlockCompositeGPR(fm_kernel="cosine", bounds=bounds).fit(Xk_tr, y_tr)
        mu_va, sig_va = gpr_c.predict(Xk_va, return_std=True)
        oof_preds["mkgpr_cos"][va_pos] = mu_va
        oof_sigmas["mkgpr_cos"][va_pos] = sig_va
        gpr_block_weights["mkgpr_cos"].append(gpr_c.block_weight_summary())
        if Xk_ho is not None:
            mu_h, sig_h = gpr_c.predict(Xk_ho, return_std=True)
            ho_fold_preds["mkgpr_cos"].append(mu_h)
            ho_fold_sigmas["mkgpr_cos"].append(sig_h)
        fitted_folds["mkgpr_cos"].append((f, kpre, gpr_c))

        # (1b) Multi-Block Exp-Cosine+Tanimoto+RBF Composite GPR
        gpr_e = MultiBlockCompositeGPR(fm_kernel="exp_cosine", exp_gamma=1.25, bounds=bounds).fit(Xk_tr, y_tr)
        mu_va, sig_va = gpr_e.predict(Xk_va, return_std=True)
        oof_preds["mkgpr_exp"][va_pos] = mu_va
        oof_sigmas["mkgpr_exp"][va_pos] = sig_va
        gpr_block_weights["mkgpr_exp"].append(gpr_e.block_weight_summary())
        if Xk_ho is not None:
            mu_h, sig_h = gpr_e.predict(Xk_ho, return_std=True)
            ho_fold_preds["mkgpr_exp"].append(mu_h)
            ho_fold_sigmas["mkgpr_exp"].append(sig_h)
        fitted_folds["mkgpr_exp"].append((f, kpre, gpr_e))

        # (1c) Empirical-Bayes Shrinkage Ridge (cluster-GroupKFold inner CV)
        ebr = MultiBlockShrinkageKRRSVR(mode="eb_ridge", bounds=bounds).fit(Xk_tr, y_tr, cluster_ids=cid_tr)
        oof_preds["eb_ridge"][va_pos] = ebr.predict(Xk_va)
        if Xk_ho is not None:
            ho_fold_preds["eb_ridge"].append(ebr.predict(Xk_ho))
        fitted_folds["eb_ridge"].append((f, kpre, ebr))

        # (1d) Multi-Block Cosine+Tanimoto+RBF KRR / SVR (cluster-GroupKFold inner CV)
        ckrr = MultiBlockShrinkageKRRSVR(mode="cosine_krr", bounds=bounds).fit(Xk_tr, y_tr, cluster_ids=cid_tr)
        oof_preds["cosine_krr"][va_pos] = ckrr.predict(Xk_va)
        if Xk_ho is not None:
            ho_fold_preds["cosine_krr"].append(ckrr.predict(Xk_ho))
        fitted_folds["cosine_krr"].append((f, kpre, ckrr))

        # 2. Compact <=94D Tabular Preprocessor (for EB-Huber & TabPFN v2)
        tpre = CompactTabularPreprocessor(track=track, cond_cols=ep_data["cond_cols"], max_total_dims=94)
        Xt_tr = tpre.fit_transform(bank, cidx_cv[tr_pos], tr_cond)
        Xt_va = tpre.transform(bank, cidx_cv[va_pos], va_cond)
        Xt_ho = tpre.transform(bank, cidx_ho[ho_vpos], ho_cond_v) if len(ho_vpos) > 0 else np.zeros((0, Xt_tr.shape[1]), dtype=np.float32)

        # (2a) Empirical-Bayes Shrinkage Huber
        ebh = EBHuberRegressor(bounds=bounds).fit(Xt_tr, y_tr, cluster_ids=cid_tr)
        oof_preds["eb_huber"][va_pos] = ebh.predict(Xt_va)
        if len(ho_vpos) > 0:
            ho_fold_preds["eb_huber"].append(ebh.predict(Xt_ho))
        fitted_folds["eb_huber"].append((f, tpre, ebh))

        # (2b) TabPFN v2 Regressor (isolated venv, CPU, <=94D fold-internal features!)
        if run_tabpfn:
            tabpfn_dim_used.append(int(Xt_tr.shape[1]))
            try:
                p_list, s_list = _fit_predict_tabpfn(Xt_tr, y_tr, [Xt_va, Xt_ho], bounds=bounds, n_estimators=tabpfn_n_est)
                oof_preds["tabpfn_v2"][va_pos] = p_list[0]
                oof_sigmas["tabpfn_v2"][va_pos] = s_list[0]
                if len(ho_vpos) > 0:
                    ho_fold_preds["tabpfn_v2"].append(p_list[1])
                    ho_fold_sigmas["tabpfn_v2"].append(s_list[1])
            except Exception as exc:
                tabpfn_err = f"{type(exc).__name__}: {exc}"

    # Aggregate 5-fold Holdout predictions & exact predictive std
    ho_preds: Dict[str, np.ndarray] = {}
    ho_sigmas: Dict[str, np.ndarray] = {}
    for m in all_new_names:
        arr = np.full(n_ho, np.nan, dtype=np.float64)
        if len(ho_vpos) > 0 and len(ho_fold_preds[m]) > 0:
            P_f = np.stack(ho_fold_preds[m], axis=0)
            arr[ho_vpos] = np.mean(P_f, axis=0)
        ho_preds[m] = arr
    for m in ("mkgpr_cos", "mkgpr_exp", "tabpfn_v2"):
        s_arr = np.full(n_ho, np.nan, dtype=np.float64)
        if len(ho_vpos) > 0 and len(ho_fold_sigmas[m]) > 0 and len(ho_fold_preds[m]) > 0:
            S_f = np.stack(ho_fold_sigmas[m], axis=0)
            P_f = np.stack(ho_fold_preds[m], axis=0)
            mean_v = np.nanmean(S_f ** 2, axis=0)
            epist_v = np.var(P_f, axis=0)
            s_arr[ho_vpos] = np.sqrt(np.maximum(mean_v + epist_v, 1e-8))
        ho_sigmas[m] = s_arr

    # =========================================================================
    # STRICT ZERO-HOLDOUT-PEEKING SELECTION GATE (ONLY OOF CV MAE!)
    # =========================================================================
    base_oof = ep_data["base_champ_oof"][track]
    base_ho = ep_data["base_champ_ho"][track]
    exist_oof = ep_data["existing_oof"][track]
    exist_ho = ep_data["existing_ho"][track]
    base_cv_m = regression_metrics(y_cv[valid_cv], base_oof[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv])
    base_cv_mae = float(base_cv_m["mae"])
    # Also use manifest reported CV MAE for exact parity reference (conservative baseline!)
    manifest_cv_mae = float(ep_doc[track]["cv_metrics"]["mae"])
    manifest_ho_mae = float(ep_doc[track]["holdout_metrics"]["mae"])
    ref_cv_mae = min(base_cv_mae, manifest_cv_mae)
    fallback_y = float(np.nanmean(y_cv[valid_cv])) if valid_cv.any() else 0.0

    # 1. Individual learner OOF metrics
    indiv_cv_metrics: Dict[str, Dict[str, Any]] = {}
    for m in all_new_names:
        indiv_cv_metrics[m] = regression_metrics(y_cv[valid_cv], oof_preds[m][valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv])

    # Best GPR (for exact GP posterior sigma export & multi-way blend)
    best_gpr_name = min(("mkgpr_cos", "mkgpr_exp"), key=lambda m: indiv_cv_metrics[m]["mae"] if indiv_cv_metrics[m]["mae"] is not None else 1e9)
    best_port_single = min(portable_names, key=lambda m: indiv_cv_metrics[m]["mae"] if indiv_cv_metrics[m]["mae"] is not None else 1e9)
    best_shrink_name = min(("eb_ridge", "cosine_krr", "eb_huber"), key=lambda m: indiv_cv_metrics[m]["mae"] if indiv_cv_metrics[m]["mae"] is not None else 1e9)

    def _eval_nested_2way(p_new_oof: np.ndarray, p_new_ho: np.ndarray, new_key: str) -> Dict[str, Any]:
        nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
        ws = []
        for f in range(5):
            va = (fids_cv == f) & valid_cv
            tr = (fids_cv != f) & valid_cv & (ct_cv == "exact")
            w_f = _fit_2way_alpha_l1(base_oof[tr], p_new_oof[tr], y_cv[tr], reg=0.005)
            ws.append(w_f)
            p_va = (1.0 - w_f) * base_oof[va] + w_f * p_new_oof[va]
            if bounds is not None:
                p_va = np.clip(p_va, bounds[0], bounds[1])
            nested_oof[va] = p_va
        fit_m = valid_cv & (ct_cv == "exact")
        w_full = _fit_2way_alpha_l1(base_oof[fit_m], p_new_oof[fit_m], y_cv[fit_m], reg=0.005)
        # Decompose base_champ weights into explicit existing family weights for clean serving!
        base_w = ep_doc[track]["weights"]
        full_weights: Dict[str, float] = {}
        if 1.0 - w_full >= 0.03:
            for k, wk in base_w.items():
                if float(wk) > 0:
                    full_weights[k] = round((1.0 - w_full) * float(wk), 5)
        if w_full >= 0.03:
            full_weights[new_key] = round(w_full, 5)
        if not full_weights:
            full_weights = {k: float(v) for k, v in base_w.items()}
        s_tot = sum(full_weights.values())
        full_weights = {k: round(v / s_tot, 5) for k, v in full_weights.items()}
        ho_streams = dict(exist_ho)
        ho_streams[new_key] = p_new_ho
        ho_pred = _apply_simplex_blend(list(full_weights.keys()), ho_streams, full_weights, bounds=bounds, fallback=fallback_y)
        cv_m = regression_metrics(y_cv[valid_cv], nested_oof[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv])
        return {"cv_metrics": cv_m, "nested_oof": nested_oof, "ho_pred": ho_pred, "weights": full_weights, "fold_w_new": [round(x, 3) for x in ws]}

    def _eval_nested_multi(cand_keys: List[str], oof_map: Dict[str, np.ndarray], ho_map: Dict[str, np.ndarray]) -> Dict[str, Any]:
        P = np.column_stack([oof_map[k] for k in cand_keys])
        nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
        for f in range(5):
            va = (fids_cv == f) & valid_cv
            tr = (fids_cv != f) & valid_cv & (ct_cv == "exact")
            if tr.sum() < 6 or va.sum() == 0:
                nested_oof[va] = base_oof[va]
                continue
            wf = _fit_simplex_weights_l1(P[tr], y_cv[tr], reg=0.025, prune_tol=0.04)
            wd = {k: float(wf[i]) for i, k in enumerate(cand_keys)}
            nested_oof[va] = _apply_simplex_blend(cand_keys, {k: oof_map[k][va] for k in cand_keys}, wd, bounds=bounds, fallback=fallback_y)
        fit_m = valid_cv & (ct_cv == "exact")
        w_full = _fit_simplex_weights_l1(P[fit_m], y_cv[fit_m], reg=0.025, prune_tol=0.04)
        weights = {k: round(float(w_full[i]), 5) for i, k in enumerate(cand_keys) if w_full[i] > 0.0}
        s_tot = sum(weights.values())
        weights = {k: round(v / s_tot, 5) for k, v in weights.items()}
        ho_pred = _apply_simplex_blend(list(weights.keys()), ho_map, weights, bounds=bounds, fallback=fallback_y)
        cv_m = regression_metrics(y_cv[valid_cv], nested_oof[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv])
        return {"cv_metrics": cv_m, "nested_oof": nested_oof, "ho_pred": ho_pred, "weights": weights}

    # Build Portable Candidate Pool (ZERO TabPFN dependency!)
    port_candidates: Dict[str, Dict[str, Any]] = {}
    for m in portable_names:
        port_candidates[f"single:{m}"] = {
            "cv_metrics": indiv_cv_metrics[m],
            "nested_oof": oof_preds[m],
            "ho_pred": ho_preds[m],
            "weights": {m: 1.0},
        }
    port_candidates[f"blend2:{best_port_single}"] = _eval_nested_2way(oof_preds[best_port_single], ho_preds[best_port_single], best_port_single)
    if best_gpr_name != best_port_single:
        port_candidates[f"blend2:{best_gpr_name}"] = _eval_nested_2way(oof_preds[best_gpr_name], ho_preds[best_gpr_name], best_gpr_name)

    # Multi-way portable blend: existing streams + best_gpr + best_shrinkage
    multi_port_keys = list(exist_oof.keys()) + list(dict.fromkeys([best_gpr_name, best_shrink_name]))
    multi_port_oof = dict(exist_oof)
    multi_port_ho = dict(exist_ho)
    for k in (best_gpr_name, best_shrink_name):
        multi_port_oof[k] = oof_preds[k]
        multi_port_ho[k] = ho_preds[k]
    port_candidates["blend_multi:portable"] = _eval_nested_multi(multi_port_keys, multi_port_oof, multi_port_ho)

    # Strictly OOF-only selection for Portable Suite (margin >= 0.002!)
    best_port_cand = min(
        port_candidates.keys(),
        key=lambda k: port_candidates[k]["cv_metrics"]["mae"] if port_candidates[k]["cv_metrics"]["mae"] is not None else 1e9,
    )
    best_port_cv_mae = float(port_candidates[best_port_cand]["cv_metrics"]["mae"])
    port_cv_gain = ref_cv_mae - best_port_cv_mae
    port_promoted = bool(port_cv_gain >= 0.002)

    if port_promoted:
        chosen_port = port_candidates[best_port_cand]
        port_win_name = best_port_cand
        port_weights = chosen_port["weights"]
        port_oof_final = chosen_port["nested_oof"]
        port_ho_final = chosen_port["ho_pred"]
    else:
        port_win_name = "baseline_unchanged"
        port_weights = dict(ep_doc[track]["weights"])
        port_oof_final = base_oof
        port_ho_final = base_ho

    # Also evaluate Full Suite (incl. TabPFN v2) strictly on OOF!
    full_candidates = dict(port_candidates)
    if run_tabpfn and np.isfinite(oof_preds["tabpfn_v2"][valid_cv]).sum() >= 4:
        full_candidates["single:tabpfn_v2"] = {
            "cv_metrics": indiv_cv_metrics["tabpfn_v2"],
            "nested_oof": oof_preds["tabpfn_v2"],
            "ho_pred": ho_preds["tabpfn_v2"],
            "weights": {"tabpfn_v2": 1.0},
        }
        full_candidates["blend2:tabpfn_v2"] = _eval_nested_2way(oof_preds["tabpfn_v2"], ho_preds["tabpfn_v2"], "tabpfn_v2")
        multi_full_keys = list(dict.fromkeys(multi_port_keys + ["tabpfn_v2"]))
        multi_full_oof = dict(multi_port_oof)
        multi_full_ho = dict(multi_port_ho)
        multi_full_oof["tabpfn_v2"] = oof_preds["tabpfn_v2"]
        multi_full_ho["tabpfn_v2"] = ho_preds["tabpfn_v2"]
        full_candidates["blend_multi:with_tabpfn"] = _eval_nested_multi(multi_full_keys, multi_full_oof, multi_full_ho)

    best_full_cand = min(
        full_candidates.keys(),
        key=lambda k: full_candidates[k]["cv_metrics"]["mae"] if full_candidates[k]["cv_metrics"]["mae"] is not None else 1e9,
    )
    best_full_cv_mae = float(full_candidates[best_full_cand]["cv_metrics"]["mae"])
    full_cv_gain = ref_cv_mae - best_full_cv_mae
    full_promoted = bool(full_cv_gain >= 0.002)

    if full_promoted:
        chosen_full = full_candidates[best_full_cand]
        full_win_name = best_full_cand
        full_weights = chosen_full["weights"]
        full_oof_final = chosen_full["nested_oof"]
        full_ho_final = chosen_full["ho_pred"]
    else:
        full_win_name = "baseline_unchanged"
        full_weights = dict(ep_doc[track]["weights"])
        full_oof_final = base_oof
        full_ho_final = base_ho

    # =========================================================================
    # SINGLE FROZEN EVALUATION ON UNTOUCHED 20% HOLDOUT!
    # =========================================================================
    base_ho_m = regression_metrics(y_ho[valid_ho], base_ho[valid_ho], yl_ho[valid_ho], yu_ho[valid_ho], ct_ho[valid_ho])
    port_ho_m = regression_metrics(y_ho[valid_ho], port_ho_final[valid_ho], yl_ho[valid_ho], yu_ho[valid_ho], ct_ho[valid_ho])
    full_ho_m = regression_metrics(y_ho[valid_ho], full_ho_final[valid_ho], yl_ho[valid_ho], yu_ho[valid_ho], ct_ho[valid_ho])
    indiv_ho_metrics = {
        m: regression_metrics(y_ho[valid_ho], ho_preds[m][valid_ho], yl_ho[valid_ho], yu_ho[valid_ho], ct_ho[valid_ho])
        for m in all_new_names
    }

    # Keep ONLY active portable small-N folds needed for serving
    active_port_models = [m for m in portable_names if float(port_weights.get(m, 0.0)) > 0.0]
    portable_bundle = None
    if port_promoted and active_port_models:
        portable_bundle = PortableSmallNModelBundle(
            endpoint=ep,
            track=track,
            bounds=bounds,
            cond_cols=ep_data["cond_cols"],
            canonical_conditions=ep_data["canonical_conditions"],
            blend_weights=port_weights,
            smalln_folds={m: fitted_folds[m] for m in active_port_models},
        )

    # Mean learned GPR block weights across 5 folds (for scientific interpretability!)
    mean_gpr_w: Dict[str, float] = {}
    if gpr_block_weights[best_gpr_name]:
        keys_w = list(gpr_block_weights[best_gpr_name][0].keys())
        mean_gpr_w = {k: round(float(np.mean([d[k] for d in gpr_block_weights[best_gpr_name]])), 4) for k in keys_w}

    return {
        "track": track,
        "manifest_baseline_cv_mae": manifest_cv_mae,
        "manifest_baseline_ho_mae": manifest_ho_mae,
        "baseline_cv_metrics": base_cv_m,
        "baseline_ho_metrics": base_ho_m,
        "indiv_cv_metrics": indiv_cv_metrics,
        "indiv_ho_metrics": indiv_ho_metrics,
        "best_gpr_name": best_gpr_name,
        "best_gpr_block_weights": mean_gpr_w,
        "tabpfn_dims": sorted(set(tabpfn_dim_used)),
        "tabpfn_error": tabpfn_err,
        "portable": {
            "promoted": port_promoted,
            "winner_name": port_win_name,
            "weights": port_weights,
            "cv_mae_gain": round(float(port_cv_gain), 5),
            "cv_metrics": regression_metrics(y_cv[valid_cv], port_oof_final[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv]),
            "ho_metrics": port_ho_m,
            "ho_mae_delta": round(float((base_ho_m["mae"] or 0.0) - (port_ho_m["mae"] or 0.0)), 5),
            "active_new_models": active_port_models,
        },
        "full_with_tabpfn": {
            "promoted": full_promoted,
            "winner_name": full_win_name,
            "weights": full_weights,
            "cv_mae_gain": round(float(full_cv_gain), 5),
            "cv_metrics": regression_metrics(y_cv[valid_cv], full_oof_final[valid_cv], yl_cv[valid_cv], yu_cv[valid_cv], ct_cv[valid_cv]),
            "ho_metrics": full_ho_m,
            "ho_mae_delta": round(float((base_ho_m["mae"] or 0.0) - (full_ho_m["mae"] or 0.0)), 5),
            "uses_tabpfn": bool(float(full_weights.get("tabpfn_v2", 0.0)) > 0.0),
        },
        "arrays": {
            "port_oof": port_oof_final,
            "port_ho": port_ho_final,
            "full_oof": full_oof_final,
            "full_ho": full_ho_final,
            "gp_oof_sigma": oof_sigmas[best_gpr_name],
            "gp_ho_sigma": ho_sigmas[best_gpr_name],
        },
        "portable_bundle": portable_bundle,
    }


def _run_leave_case17_out_pk(
    ep_data: Dict[str, Any],
    track_res: Dict[str, Any],
    track: str,
    bank: UnifiedSmallNBank,
    run_tabpfn: bool,
    tabpfn_n_est: int = 8,
) -> Dict[str, Any]:
    """Strict Zero-Leakage Leave-Case17-Out refit (case17_excluded == 0 ONLY!) for small-N PK endpoints."""
    ep = ep_data["endpoint"]
    bounds = ep_data["bounds"]
    c17_idx = bank.case17_idx
    keep = (ep_data["case17_excluded"] == 0) & (ep_data["censor_type"] == "exact")
    if track == "smiles_only":
        keep = keep & bank.smiles_valid[ep_data["compound_idx"]]
    keep_pos = np.where(keep)[0]
    if len(keep_pos) < 8:
        return {"skipped": True, "reason": f"only {len(keep_pos)} non-case17 rows"}

    cidx_tr = ep_data["compound_idx"][keep_pos]
    y_tr = ep_data["y_value"][keep_pos]
    cid_tr = ep_data["cluster_id"][keep_pos]
    cond_tr = [ep_data["cond_rows"][i] for i in keep_pos]
    canon_cond = ep_data["canonical_conditions"]
    alt_conds = ALT_C17_CONDS.get(ep, {})

    # Determine which new models are needed for portable & full_with_tabpfn winners
    port_w = track_res["portable"]["weights"]
    full_w = track_res["full_with_tabpfn"]["weights"]
    needed_new = {k for k, w in {**port_w, **full_w}.items() if float(w) > 0 and k not in ("cpu", "fusion", "deep")}

    # Refit needed new models strictly on case17_excluded == 0!
    new_c17_canon: Dict[str, np.ndarray] = {}
    new_c17_alt: Dict[str, Dict[str, np.ndarray]] = {ak: {} for ak in alt_conds}

    if needed_new:
        kpre = MultiBlockKernelPreprocessor(track=track, cond_cols=ep_data["cond_cols"])
        Xk_tr = kpre.fit_transform(bank, cidx_tr, cond_tr)
        Xk_c17 = kpre.transform(bank, c17_idx, default_cond=canon_cond)
        Xk_alt = {ak: kpre.transform(bank, c17_idx, default_cond=ac) for ak, ac in alt_conds.items()}

        if "mkgpr_cos" in needed_new:
            m = MultiBlockCompositeGPR(fm_kernel="cosine", bounds=bounds).fit(Xk_tr, y_tr)
            new_c17_canon["mkgpr_cos"] = m.predict(Xk_c17)
            for ak in alt_conds:
                new_c17_alt[ak]["mkgpr_cos"] = m.predict(Xk_alt[ak])
        if "mkgpr_exp" in needed_new:
            m = MultiBlockCompositeGPR(fm_kernel="exp_cosine", exp_gamma=1.25, bounds=bounds).fit(Xk_tr, y_tr)
            new_c17_canon["mkgpr_exp"] = m.predict(Xk_c17)
            for ak in alt_conds:
                new_c17_alt[ak]["mkgpr_exp"] = m.predict(Xk_alt[ak])
        if "eb_ridge" in needed_new:
            m = MultiBlockShrinkageKRRSVR(mode="eb_ridge", bounds=bounds).fit(Xk_tr, y_tr, cluster_ids=cid_tr)
            new_c17_canon["eb_ridge"] = m.predict(Xk_c17)
            for ak in alt_conds:
                new_c17_alt[ak]["eb_ridge"] = m.predict(Xk_alt[ak])
        if "cosine_krr" in needed_new:
            m = MultiBlockShrinkageKRRSVR(mode="cosine_krr", bounds=bounds).fit(Xk_tr, y_tr, cluster_ids=cid_tr)
            new_c17_canon["cosine_krr"] = m.predict(Xk_c17)
            for ak in alt_conds:
                new_c17_alt[ak]["cosine_krr"] = m.predict(Xk_alt[ak])

        if "eb_huber" in needed_new or ("tabpfn_v2" in needed_new and run_tabpfn):
            tpre = CompactTabularPreprocessor(track=track, cond_cols=ep_data["cond_cols"], max_total_dims=94)
            Xt_tr = tpre.fit_transform(bank, cidx_tr, cond_tr)
            Xt_c17 = tpre.transform(bank, c17_idx, default_cond=canon_cond)
            Xt_alt = {ak: tpre.transform(bank, c17_idx, default_cond=ac) for ak, ac in alt_conds.items()}
            if "eb_huber" in needed_new:
                m = EBHuberRegressor(bounds=bounds).fit(Xt_tr, y_tr, cluster_ids=cid_tr)
                new_c17_canon["eb_huber"] = m.predict(Xt_c17)
                for ak in alt_conds:
                    new_c17_alt[ak]["eb_huber"] = m.predict(Xt_alt[ak])
            if "tabpfn_v2" in needed_new and run_tabpfn:
                eval_list = [Xt_c17] + [Xt_alt[ak] for ak in alt_conds]
                p_list, _ = _fit_predict_tabpfn(Xt_tr, y_tr, eval_list, bounds=bounds, n_estimators=tabpfn_n_est)
                new_c17_canon["tabpfn_v2"] = p_list[0]
                for idx_a, ak in enumerate(alt_conds):
                    new_c17_alt[ak]["tabpfn_v2"] = p_list[1 + idx_a]

    fb = float(np.mean(y_tr))
    base_c17 = ep_data["base_champ_c17"][track]
    merged_canon = dict(ep_data["existing_c17"][track])
    merged_canon.update(new_c17_canon)

    port_c17 = (
        _apply_simplex_blend(list(port_w.keys()), merged_canon, port_w, bounds=bounds, fallback=fb)
        if track_res["portable"]["promoted"]
        else base_c17
    )
    full_c17 = (
        _apply_simplex_blend(list(full_w.keys()), merged_canon, full_w, bounds=bounds, fallback=fb)
        if track_res["full_with_tabpfn"]["promoted"]
        else base_c17
    )

    out: Dict[str, Any] = {
        "n_train_non_case17": int(len(keep_pos)),
        "drugs": [],
    }
    for i, q in enumerate(bank.case17_queries):
        dn = q["drug_name"]
        rec: Dict[str, Any] = {
            "drug_name": dn,
            "clinical_route": CASE17_CLINICAL_ROUTE.get(dn, "IV"),
            "baseline_pred": round(float(base_c17[i]), 5) if np.isfinite(base_c17[i]) else None,
            "portable_pred": round(float(port_c17[i]), 5) if np.isfinite(port_c17[i]) else None,
            "full_pred": round(float(full_c17[i]), 5) if np.isfinite(full_c17[i]) else None,
        }
        if ep == "PPB":
            rec["fda_ppb_ref"] = CASE17_FDA_NUMERIC.get(dn, {}).get("ppb")
        out["drugs"].append(rec)

    if ep == "PPB":
        # Compute audited Case17 FDA PPB benchmark MAE & Spearman (14 FDA numeric anchors!)
        idx_ref = [i for i, q in enumerate(bank.case17_queries) if CASE17_FDA_NUMERIC.get(q["drug_name"], {}).get("ppb") is not None]
        y_ref = np.array([CASE17_FDA_NUMERIC[bank.case17_queries[i]["drug_name"]]["ppb"] for i in idx_ref], dtype=np.float64)
        out["fda_ppb_benchmark"] = {
            "n_drugs": len(idx_ref),
            "baseline": regression_metrics(y_ref, base_c17[idx_ref]),
            "portable": regression_metrics(y_ref, port_c17[idx_ref]),
            "full_with_tabpfn": regression_metrics(y_ref, full_c17[idx_ref]),
        }
    return out


def _worker_run_endpoint(args: Tuple[str, bool, int]) -> Dict[str, Any]:
    ep, run_tabpfn, tabpfn_n_est = args
    t0 = time.time()
    assert GLOBAL_BANK is not None and GLOBAL_MANIFEST is not None
    ep_doc = GLOBAL_MANIFEST["endpoints"][ep]
    ep_data = _load_endpoint_frozen(ep, ep_doc, GLOBAL_BANK)

    res_dual = _run_one_endpoint_track(ep_data, ep_doc, "dual_modality", GLOBAL_BANK, run_tabpfn=run_tabpfn, tabpfn_n_est=tabpfn_n_est)
    res_smi = _run_one_endpoint_track(ep_data, ep_doc, "smiles_only", GLOBAL_BANK, run_tabpfn=run_tabpfn, tabpfn_n_est=tabpfn_n_est)

    # Save verified node2-portable .joblib if either track promoted!
    promoted_dir = OUT_DIR / "promoted_models"
    promoted_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = None
    joblib_modules: List[str] = []
    if res_dual["portable_bundle"] is not None or res_smi["portable_bundle"] is not None:
        payload = {
            "schema": "v33_plan_b1_smalln_endpoint_v1",
            "endpoint": ep,
            "wave": ep_data["wave"],
            "dual_modality": res_dual["portable_bundle"],
            "smiles_only": res_smi["portable_bundle"],
        }
        jp = promoted_dir / f"{ep}_smalln_portable.joblib"
        joblib.dump(payload, jp, compress=1)
        joblib_modules = _check_joblib_node2_portable(jp)
        joblib_path = str(jp)

    # Save OOF / HO predictions & exact GP posterior sigmas for Deliverable 2!
    cache_dir = OUT_DIR / "smalln_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{ep}_smalln_preds.npz",
        dual_port_oof=res_dual["arrays"]["port_oof"],
        dual_port_ho=res_dual["arrays"]["port_ho"],
        dual_full_oof=res_dual["arrays"]["full_oof"],
        dual_full_ho=res_dual["arrays"]["full_ho"],
        dual_gp_oof_sigma=res_dual["arrays"]["gp_oof_sigma"],
        dual_gp_ho_sigma=res_dual["arrays"]["gp_ho_sigma"],
        smi_port_oof=res_smi["arrays"]["port_oof"],
        smi_port_ho=res_smi["arrays"]["port_ho"],
        smi_full_oof=res_smi["arrays"]["full_oof"],
        smi_full_ho=res_smi["arrays"]["full_ho"],
        smi_gp_oof_sigma=res_smi["arrays"]["gp_oof_sigma"],
        smi_gp_ho_sigma=res_smi["arrays"]["gp_ho_sigma"],
    )

    # Strict Leave-Case17-Out for all PK endpoints in small-N!
    c17_doc = None
    if ep in PK_SMALLN_ENDPOINTS:
        c17_doc = {
            "dual_modality": _run_leave_case17_out_pk(ep_data, res_dual, "dual_modality", GLOBAL_BANK, run_tabpfn=run_tabpfn, tabpfn_n_est=tabpfn_n_est),
            "smiles_only": _run_leave_case17_out_pk(ep_data, res_smi, "smiles_only", GLOBAL_BANK, run_tabpfn=run_tabpfn, tabpfn_n_est=tabpfn_n_est),
        }

    # Strip heavy arrays & bundle objects before returning IPC summary
    for tr_r in (res_dual, res_smi):
        tr_r.pop("arrays", None)
        tr_r.pop("portable_bundle", None)

    elapsed = round(time.time() - t0, 2)
    dp = res_dual["portable"]
    df = res_dual["full_with_tabpfn"]
    print(
        f"  [{ep:<20}] N={ep_doc['counts']['n_total']:3d} | Base CV/HO={res_dual['baseline_cv_metrics']['mae']:.4f}/{res_dual['baseline_ho_metrics']['mae']:.4f} "
        f"| Port({dp['winner_name'][:18]:<18}) CV/HO={dp['cv_metrics']['mae']:.4f}/{dp['ho_metrics']['mae']:.4f} ({'+' if dp['ho_mae_delta']>=0 else ''}{dp['ho_mae_delta']:.4f}) "
        f"| Full({df['winner_name'][:18]:<18}) CV/HO={df['cv_metrics']['mae']:.4f}/{df['ho_metrics']['mae']:.4f} ({elapsed:.1f}s)",
        flush=True,
    )
    return {
        "endpoint": ep,
        "wave": ep_data["wave"],
        "n_total": int(ep_doc["counts"]["n_total"]),
        "n_cv": int(ep_doc["counts"]["n_cv"]),
        "n_holdout": int(ep_doc["counts"]["n_holdout"]),
        "n_exact_cv": int((ep_data["censor_type"][ep_data["cv_idx"]] == "exact").sum()),
        "n_exact_ho": int((ep_data["censor_type"][ep_data["ho_idx"]] == "exact").sum()),
        "elapsed_sec": elapsed,
        "promoted_joblib": joblib_path,
        "joblib_modules": joblib_modules,
        "dual_modality": res_dual,
        "smiles_only": res_smi,
        "case17": c17_doc,
    }


def main():
    global GLOBAL_BANK, GLOBAL_MANIFEST, TABPFN_AVAILABLE, TABPFN_STATUS_MSG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-n", type=int, default=400)
    ap.add_argument("--tabpfn-est", type=int, default=8)
    ap.add_argument("--no-tabpfn", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))

    # Check TabPFN availability WITHOUT importing torch in parent before fork()!
    import importlib.util

    if not args.no_tabpfn and TABPFN_CKPT.exists():
        if importlib.util.find_spec("tabpfn") is not None:
            TABPFN_AVAILABLE = True
            TABPFN_STATUS_MSG = f"tabpfn available in isolated tabpfn_venv (deferred post-fork import) using offline {TABPFN_CKPT}"
        else:
            TABPFN_AVAILABLE = False
            TABPFN_STATUS_MSG = "tabpfn package not found in current Python environment"
    else:
        TABPFN_AVAILABLE = False
        TABPFN_STATUS_MSG = "TabPFN disabled or checkpoint missing"
    print(f"[TabPFN] {TABPFN_STATUS_MSG}", flush=True)

    GLOBAL_BANK = UnifiedSmallNBank()

    target_eps = [
        ep
        for ep, d in sorted(GLOBAL_MANIFEST["endpoints"].items(), key=lambda kv: kv[1]["counts"]["n_total"])
        if d["task_type"] == "regression" and int(d["counts"]["n_total"]) <= args.max_n
    ]
    print(f"[Plan B-1 Deliverable 1] Running Small-N Suite on {len(target_eps)} endpoints (N <= {args.max_n}) with {args.workers} workers...", flush=True)

    jobs = [(ep, TABPFN_AVAILABLE, args.tabpfn_est) for ep in target_eps]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        results = pool.map(_worker_run_endpoint, jobs)

    by_ep = {r["endpoint"]: r for r in results}

    def _agg(ep_subset: List[str], track: str, mode: str) -> Dict[str, Any]:
        b_cv, b_ho, n_cv, n_ho, b_sp, n_sp, b_r2, n_r2 = [], [], [], [], [], [], [], []
        n_prom = 0
        n_ho_imp = 0
        for ep in ep_subset:
            tr = by_ep[ep][track]
            bm_cv, bm_ho = tr["baseline_cv_metrics"], tr["baseline_ho_metrics"]
            nm = tr[mode]
            if nm["promoted"]:
                n_prom += 1
                if nm["ho_mae_delta"] > 0:
                    n_ho_imp += 1
            b_cv.append(bm_cv["mae"])
            b_ho.append(bm_ho["mae"])
            n_cv.append(nm["cv_metrics"]["mae"])
            n_ho.append(nm["ho_metrics"]["mae"])
            if bm_ho.get("spearman") is not None and nm["ho_metrics"].get("spearman") is not None:
                b_sp.append(bm_ho["spearman"])
                n_sp.append(nm["ho_metrics"]["spearman"])
            if bm_ho.get("r2") is not None and nm["ho_metrics"].get("r2") is not None:
                b_r2.append(bm_ho["r2"])
                n_r2.append(nm["ho_metrics"]["r2"])
        return {
            "n_endpoints": len(ep_subset),
            "n_oof_promoted": n_prom,
            "n_promoted_improved_holdout": n_ho_imp,
            "mean_cv_mae_before": round(float(np.mean(b_cv)), 5),
            "mean_cv_mae_after": round(float(np.mean(n_cv)), 5),
            "mean_cv_mae_delta": round(float(np.mean(b_cv) - np.mean(n_cv)), 5),
            "mean_ho_mae_before": round(float(np.mean(b_ho)), 5),
            "mean_ho_mae_after": round(float(np.mean(n_ho)), 5),
            "mean_ho_mae_delta": round(float(np.mean(b_ho) - np.mean(n_ho)), 5),
            "mean_ho_spearman_before": round(float(np.mean(b_sp)), 5) if b_sp else None,
            "mean_ho_spearman_after": round(float(np.mean(n_sp)), 5) if n_sp else None,
            "mean_ho_r2_before": round(float(np.mean(b_r2)), 5) if b_r2 else None,
            "mean_ho_r2_after": round(float(np.mean(n_r2)), 5) if n_r2 else None,
        }

    eps_le_300 = [ep for ep in target_eps if by_ep[ep]["n_total"] <= 300]
    eps_le_283 = [ep for ep in target_eps if by_ep[ep]["n_total"] <= 283]  # 16 endpoints slice
    eps_weak_9 = [ep for ep in ["Vd_absolute", "MBEC_biofilm", "CL_absolute", "PPB", "MDCK", "RRCK", "F", "Cmax", "efflux_ratio"] if ep in by_ep]

    summary = {
        "schema": "v33_plan_b1_smalln_results_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_sec": round(time.time() - t_all, 2),
        "tabpfn_status": TABPFN_STATUS_MSG,
        "selection_gate": "strictly_inner_5fold_cv_oof_mae_gain >= 0.002 (zero holdout peeking)",
        "slices": {
            "all_n_le_400_20ep": {
                "endpoints": target_eps,
                "dual_portable": _agg(target_eps, "dual_modality", "portable"),
                "dual_full_with_tabpfn": _agg(target_eps, "dual_modality", "full_with_tabpfn"),
                "smiles_portable": _agg(target_eps, "smiles_only", "portable"),
                "smiles_full_with_tabpfn": _agg(target_eps, "smiles_only", "full_with_tabpfn"),
            },
            "all_n_le_300_19ep": {
                "endpoints": eps_le_300,
                "dual_portable": _agg(eps_le_300, "dual_modality", "portable"),
                "dual_full_with_tabpfn": _agg(eps_le_300, "dual_modality", "full_with_tabpfn"),
                "smiles_portable": _agg(eps_le_300, "smiles_only", "portable"),
                "smiles_full_with_tabpfn": _agg(eps_le_300, "smiles_only", "full_with_tabpfn"),
            },
            "smalln_16ep_n_le_283": {
                "endpoints": eps_le_283,
                "dual_portable": _agg(eps_le_283, "dual_modality", "portable"),
                "dual_full_with_tabpfn": _agg(eps_le_283, "dual_modality", "full_with_tabpfn"),
                "smiles_portable": _agg(eps_le_283, "smiles_only", "portable"),
                "smiles_full_with_tabpfn": _agg(eps_le_283, "smiles_only", "full_with_tabpfn"),
            },
            "priority_weak_9ep": {
                "endpoints": eps_weak_9,
                "dual_portable": _agg(eps_weak_9, "dual_modality", "portable"),
                "dual_full_with_tabpfn": _agg(eps_weak_9, "dual_modality", "full_with_tabpfn"),
                "smiles_portable": _agg(eps_weak_9, "smiles_only", "portable"),
                "smiles_full_with_tabpfn": _agg(eps_weak_9, "smiles_only", "full_with_tabpfn"),
            },
        },
        "endpoints": by_ep,
    }
    out_fp = OUT_DIR / "smalln_results.json"
    out_fp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE Deliverable 1] Saved {out_fp} in {summary['elapsed_sec']}s!", flush=True)
    print(json.dumps(summary["slices"]["all_n_le_300_19ep"], indent=2), flush=True)


if __name__ == "__main__":
    main()
