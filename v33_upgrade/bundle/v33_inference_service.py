#!/usr/bin/env python3
"""Peptide OmniPanel — Self-Contained Live Super-Ensemble Inference HTTP Service.

Runs 100% self-contained on CPU (`http.server.ThreadingHTTPServer`, 0 MiB GPU VRAM),
binding `127.0.0.1:8902` (or configured `--host/--port`).

Loads ONLY the pruned champion models referenced by `v33_manifest.json`, keeps
ncAA-RepDistill v2 + PeptideCLM-2 + ChemBERTa-MTR + unified 46-EP MultiTaskPeptideNet46
warm in RAM, and communicates with a persistent background `esmc_worker.py` over
localhost TCP for warm ESM-C 600M multi-region sequence encoding.

HTTP Endpoints:
  - GET  /health
  - GET  /endpoints
  - POST /predict
  - POST /batch
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

BASE_DIR = Path(os.environ.get("V33_BASE_DIR", str(Path(__file__).resolve().parent.parent)))
BUNDLE_DIR = BASE_DIR / "bundle"
SERVING_DIR = BUNDLE_DIR / "serving_models"

# Read-only model class definitions for joblib unpickling & deep checkpoint loading
sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(BASE_DIR / "models_cpu"))
sys.path.insert(0, str(BASE_DIR / "models_fusion"))
sys.path.insert(0, str(BASE_DIR / "models_gpu_deep"))

from fusion_models import (  # noqa: E402
    BoundedWrapper,
    ExactRBFKLR,
    ExactRBFKRR,
    FastPlattRBFSVC,
    PrecomputedRBFSVC,
    PrecomputedRBFSVR,
)
from inference_api import load_deep_checkpoint, predict_from_numpy  # noqa: E402
try:
    from mt46_inference import load_mt46_checkpoint, predict_mt46_from_numpy  # noqa: E402
except Exception:  # pragma: no cover
    load_mt46_checkpoint = None  # type: ignore[assignment]
    predict_mt46_from_numpy = None  # type: ignore[assignment]
from v33_config import AUX_ENDPOINTS, ENDPOINT_SPECS, compute_halflife_mechanism_anchor  # noqa: E402
from v33_live_encoders import (  # noqa: E402
    AA20,
    LiveMiniBank,
    V33EncodingHub,
)

AA20_IDX = {a: i for i, a in enumerate(AA20)}
ESMC_PYTHON = Path(os.environ.get("V33_ESMC_PYTHON", sys.executable))
ESMC_WORKER_SCRIPT = BUNDLE_DIR / "esmc_worker.py"
_DEFAULT_ESMC_MODEL = BASE_DIR.parent / "pretrained" / "esmc_600m"
ESMC_MODEL_DIR = os.environ.get(
    "V33_ESMC_MODEL_DIR",
    str(_DEFAULT_ESMC_MODEL) if _DEFAULT_ESMC_MODEL.exists() else "",
).strip()
EP_WORKERS = max(4, int(os.environ.get("V33_EP_WORKERS", "16")))
ESMC_THREADS = max(4, int(os.environ.get("V33_ESMC_THREADS", "24")))


def _resolve_deep_ckpt_path(raw_path: str) -> Path:
    """Resolve deep checkpoint paths portably relative to BASE_DIR."""
    p = Path(raw_path)
    if p.exists():
        return p
    if not p.is_absolute():
        rel = BASE_DIR / p
        if rel.exists():
            return rel
    # Fallback: match models_gpu_deep/... suffix if present
    marker = "models_gpu_deep/"
    if marker in raw_path:
        remapped = BASE_DIR / raw_path[raw_path.index(marker) :]
        if remapped.exists():
            return remapped
    return p


PK_ENDPOINTS = {"half_life_reg", "half_life_cls", "PPB", "CL_per_kg", "CL_absolute", "Vd_per_kg", "Vd_absolute", "F", "Cmax"}
HEMOL_ENDPOINTS = {"hemolysis_cls", "hemolysis_HC50", "hemolysis_MHC"}


def _get_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return round(int(ln.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return 0.0


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -35.0, 35.0)))


def _predict_cpu_est(est: Any, X: Any, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1].astype(np.float64)
        dec = est.decision_function(X).astype(np.float64)
        return _sigmoid(dec)
    return np.asarray(est.predict(X), dtype=np.float64)


def _predict_fusion_est(est: Any, X: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1].astype(np.float64)
        dec = est.decision_function(X).astype(np.float64)
        return _sigmoid(dec)
    return np.asarray(est.predict(X), dtype=np.float64)


def _apply_frozen_calibrator(p_raw: np.ndarray, cal_spec: Optional[Dict[str, Any]]) -> np.ndarray:
    p = np.asarray(p_raw, dtype=np.float64)
    if cal_spec is None:
        return np.clip(p, 1e-5, 1.0 - 1e-5)
    method = cal_spec.get("method", "none")
    params = cal_spec.get("params", {})
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pc = np.clip(p[ok], 1e-6, 1.0 - 1e-6)
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


def _apply_frozen_conformal(
    pred: np.ndarray,
    conf_spec: Dict[str, Any],
    max_tanimoto: np.ndarray,
) -> Dict[str, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    mode = conf_spec.get("mode", "standard")
    if mode == "mondrian_ad_eb":
        thr = conf_spec.get("ad_thresholds") or {"in_domain": 0.65, "borderline": 0.40}
        t_in = float(thr.get("in_domain", 0.65))
        t_bo = float(thr.get("borderline", 0.40))
        q80_map = conf_spec.get("q80_by_tier") or {}
        q90_map = conf_spec.get("q90_by_tier") or {}
        q80_g = float(conf_spec.get("q80_global", conf_spec.get("q80", 0.5)))
        q90_g = float(conf_spec.get("q90_global", conf_spec.get("q90", 0.75)))
        t = np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.0)
        hw80 = np.where(
            t >= t_in,
            float(q80_map.get("In-Domain", q80_g)),
            np.where(t >= t_bo, float(q80_map.get("Borderline", q80_g)), float(q80_map.get("OOD", q80_g))),
        )
        hw90 = np.where(
            t >= t_in,
            float(q90_map.get("In-Domain", q90_g)),
            np.where(t >= t_bo, float(q90_map.get("Borderline", q90_g)), float(q90_map.get("OOD", q90_g))),
        )
    else:
        if mode == "adaptive_tanimoto_dist":
            t = np.where(np.isfinite(max_tanimoto), max_tanimoto, 0.5)
            d = 1.0 - np.clip(t, 0.0, 1.0)
            s = np.maximum(
                float(conf_spec["dist_intercept"]) + float(conf_spec["dist_slope"]) * d,
                float(conf_spec["min_scale"]),
            )
        else:
            s = np.ones(len(p), dtype=np.float64)
        hw80 = float(conf_spec["q80"]) * s
        hw90 = float(conf_spec["q90"]) * s
    lo80, hi80 = p - hw80, p + hw80
    lo90, hi90 = p - hw90, p + hw90
    bounds = conf_spec.get("bounds")
    if bounds is not None:
        lo80 = np.clip(lo80, bounds[0], bounds[1])
        hi80 = np.clip(hi80, bounds[0], bounds[1])
        lo90 = np.clip(lo90, bounds[0], bounds[1])
        hi90 = np.clip(hi90, bounds[0], bounds[1])
    return {"lo80": lo80, "hi80": hi80, "lo90": lo90, "hi90": hi90}


def _weighted_blend(
    streams: Dict[str, np.ndarray],
    weights: Dict[str, float],
    task_type: str,
    bounds: Optional[List[float]] = None,
) -> np.ndarray:
    names = [k for k, w in weights.items() if float(w) > 0.0 and k in streams]
    w_arr = np.array([float(weights[k]) for k in names], dtype=np.float64)
    w_arr = w_arr / np.sum(w_arr)
    M = np.column_stack([streams[k] for k in names])
    fin = np.isfinite(M)
    if task_type == "classification":
        Z = np.where(fin, _logit(np.where(fin, M, 0.5)), 0.0)
        W = np.where(fin, w_arr[None, :], 0.0)
        wsum = W.sum(axis=1)
        z_out = np.divide((Z * W).sum(axis=1), wsum, out=np.zeros(M.shape[0]), where=wsum > 1e-12)
        return np.clip(_sigmoid(z_out), 1e-6, 1.0 - 1e-6)
    M0 = np.where(fin, M, 0.0)
    W = np.where(fin, w_arr[None, :], 0.0)
    wsum = W.sum(axis=1)
    y_out = np.divide((M0 * W).sum(axis=1), wsum, out=np.zeros(M.shape[0]), where=wsum > 1e-12)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


class FastADAnalogEngine:
    """<15ms global exact Tanimoto + sequence 3-mer Jaccard AD & Top-3 Analog lookup across all 24 endpoints."""

    def __init__(self, ad_npz_path: Path, analog_gz_path: Path):
        t0 = time.time()
        d = np.load(ad_npz_path, allow_pickle=False)
        ref_packed = np.ascontiguousarray(d["ref_morgan_packed"], dtype=np.uint8)
        self.n_ref = int(ref_packed.shape[0])
        self.ref_bits_f32 = np.ascontiguousarray(np.unpackbits(ref_packed, axis=1).astype(np.float32))
        self.ref_pop = self.ref_bits_f32.sum(axis=1).astype(np.float32)
        self.ref_smi_valid = np.ascontiguousarray(d["ref_smi_valid"], dtype=bool)
        self.ref_seq_valid = np.ascontiguousarray(d["ref_seq_valid"], dtype=bool)
        self.in_domain_thr = float(d["ad_in_domain_thr"][0])
        self.borderline_thr = float(d["ad_borderline_thr"][0])

        # Sparse 3-mer matrix (n_ref x 8000) for sequence-only fallback
        indptr = d["seq_3mer_indptr"].astype(np.int32)
        indices = d["seq_3mer_indices"].astype(np.int32)
        data = np.ones(len(indices), dtype=np.float32)
        self.seq_3mer_csr = csr_matrix((data, indices, indptr), shape=(self.n_ref, 8000), dtype=np.float32)
        self.seq_3mer_pop = d["seq_3mer_pop"].astype(np.float32)

        self.ep_ref_idx: Dict[str, np.ndarray] = {}
        for k in d.files:
            if k.startswith("ep_ref_idx__"):
                ep = k[len("ep_ref_idx__") :]
                self.ep_ref_idx[ep] = np.ascontiguousarray(d[k], dtype=np.int32)
        d.close()

        with gzip.open(analog_gz_path, "rt", encoding="utf-8") as gz:
            lookup = json.load(gz)
        self.ep_records: Dict[str, List[Dict[str, Any]]] = {
            ep: v["records"] for ep, v in lookup["endpoints"].items()
        }
        print(
            f"[ad] FastADAnalogEngine loaded ({self.n_ref:,} refs x {len(self.ep_ref_idx)} endpoints) in {time.time()-t0:.2f}s",
            flush=True,
        )

    @staticmethod
    def _seq_to_3mer_vec(seq: str) -> Tuple[csr_matrix, float]:
        clean = [c for c in (seq or "").upper() if c in AA20_IDX]
        if len(clean) < 3:
            return csr_matrix((1, 8000), dtype=np.float32), 0.0
        codes = sorted({AA20_IDX[clean[i]] * 400 + AA20_IDX[clean[i + 1]] * 20 + AA20_IDX[clean[i + 2]] for i in range(len(clean) - 2)})
        vec = csr_matrix((np.ones(len(codes), dtype=np.float32), (np.zeros(len(codes), dtype=np.int32), codes)), shape=(1, 8000), dtype=np.float32)
        return vec, float(len(codes))

    def query_batch(
        self,
        morgan_bits: np.ndarray,
        smiles_valid: np.ndarray,
        sequences: List[str],
        endpoints: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Compute exact AD metrics & Top-3 nearest training analogs for every item x endpoint."""
        B = len(smiles_valid)
        # 1. Single vectorized Tanimoto BLAS matmul for all SMILES-valid queries
        sim_smi_all = np.zeros((B, self.n_ref), dtype=np.float32)
        smi_pos = np.where(smiles_valid)[0]
        if len(smi_pos) > 0:
            q_bits = morgan_bits[smi_pos].astype(np.float32)
            q_pop = q_bits.sum(axis=1)
            inter = q_bits @ self.ref_bits_f32.T  # (b, n_ref)
            union = q_pop[:, None] + self.ref_pop[None, :] - inter
            sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            sim[:, ~self.ref_smi_valid] = -1.0
            sim[q_pop <= 0] = 0.0
            sim_smi_all[smi_pos] = sim

        # 2. Sequence 3-mer fallback ONLY for items without valid SMILES or for seq-only reference rows
        sim_seq_cache: Dict[int, np.ndarray] = {}
        for i in range(B):
            if not smiles_valid[i] and sequences[i]:
                q_vec, q_p = self._seq_to_3mer_vec(sequences[i])
                if q_p > 0:
                    inter_q = (self.seq_3mer_csr @ q_vec.T).toarray().ravel()
                    union_q = q_p + self.seq_3mer_pop - inter_q
                    s_q = np.divide(inter_q, union_q, out=np.zeros_like(inter_q), where=union_q > 0)
                    s_q[~self.ref_seq_valid] = -1.0
                    sim_seq_cache[i] = s_q

        out_by_ep: Dict[str, List[Dict[str, Any]]] = {}
        for ep in endpoints:
            g_idx = self.ep_ref_idx[ep]
            recs = self.ep_records[ep]
            n_r = len(g_idx)
            k5 = min(5, n_r)
            k3 = min(3, n_r)

            ep_list: List[Dict[str, Any]] = []
            for i in range(B):
                if smiles_valid[i]:
                    s_ep = sim_smi_all[i, g_idx]
                    metric_type = "tanimoto"
                elif i in sim_seq_cache:
                    s_ep = sim_seq_cache[i][g_idx]
                    metric_type = "seq_3mer_jaccard"
                else:
                    s_ep = np.zeros(n_r, dtype=np.float32)
                    metric_type = "none"

                # Guard against all -1 rows (e.g., if endpoint has only seq-only refs and query is SMILES)
                valid_s = np.maximum(s_ep, 0.0)
                if n_r == 0:
                    ep_list.append({
                        "max_tanimoto": 0.0, "top5_mean_tanimoto": 0.0, "ad_tier": "OOD",
                        "similarity_metric": metric_type, "top3_analogs": [],
                    })
                    continue

                if k5 >= n_r:
                    top_idx = np.argsort(-valid_s)
                else:
                    part_idx = np.argpartition(-valid_s, k5 - 1)[:k5]
                    top_idx = part_idx[np.argsort(-valid_s[part_idx])]

                max_t = float(valid_s[top_idx[0]])
                top5_t = float(np.mean(valid_s[top_idx[:k5]]))
                if max_t >= self.in_domain_thr:
                    tier = "In-Domain"
                elif max_t >= self.borderline_thr:
                    tier = "Borderline"
                else:
                    tier = "OOD"

                analogs = []
                for loc_j in top_idx[:k3]:
                    r = recs[int(loc_j)]
                    analogs.append({
                        "compound_key": r["compound_key"],
                        "smiles": r["smiles"],
                        "sequence": r["sequence"],
                        "similarity": round(float(valid_s[int(loc_j)]), 4),
                        "experimental_value": r["experimental_value"],
                        "censor_type": r["censor_type"],
                        "unit": r["unit"],
                    })

                ep_list.append({
                    "max_tanimoto": round(max_t, 4),
                    "top5_mean_tanimoto": round(top5_t, 4),
                    "ad_tier": tier,
                    "similarity_metric": metric_type,
                    "top3_analogs": analogs,
                })
            out_by_ep[ep] = ep_list
        return out_by_ep


def _resolve_conditions(ep: str, canonical: Dict[str, str], user_cond: Dict[str, Any]) -> Dict[str, str]:
    """Map optional request-level assay conditions cleanly onto endpoint `cond_*` columns."""
    c = dict(canonical)
    rbc = user_cond.get("rbc_species")
    pk_route = user_cond.get("pk_route")
    pk_species = user_cond.get("pk_species")
    pk_matrix = user_cond.get("pk_matrix")

    if ep in HEMOL_ENDPOINTS and rbc:
        c["cond_species"] = str(rbc).strip().lower()

    if ep in PK_ENDPOINTS:
        if pk_species and "cond_species" in c:
            c["cond_species"] = str(pk_species).strip().lower()
        if pk_route and "cond_route" in c:
            r_clean = str(pk_route).strip().upper()
            if r_clean == "ORAL":
                r_clean = "PO"
            c["cond_route"] = r_clean
        if pk_matrix and "cond_matrix" in c:
            m_clean = str(pk_matrix).strip().lower()
            if ep == "PPB" and m_clean == "in_vivo_plasma":
                m_clean = "plasma"
            c["cond_matrix"] = m_clean
    return c


def _format_display_value(raw_val: float, ep_doc: Dict[str, Any]) -> Tuple[float, str]:
    """Convert raw model output into interpretable linear display value & string."""
    task_type = ep_doc["task_type"]
    log10_space = bool(ep_doc["log10_space"])
    lin_unit = ep_doc["linear_unit"]
    if task_type == "classification":
        dv = round(float(raw_val), 5)
        return dv, f"{dv * 100.0:.1f}%"
    if log10_space:
        lin = float(10.0 ** np.clip(raw_val, -15.0, 15.0))
        if lin_unit == "cm/s":
            return lin, f"{lin:.2e} cm/s"
        if lin >= 100.0:
            return round(lin, 2), f"{lin:.1f} {lin_unit}"
        if lin >= 10.0:
            return round(lin, 3), f"{lin:.2f} {lin_unit}"
        return round(lin, 4), f"{lin:.3g} {lin_unit}"
    dv = round(float(raw_val), 4)
    if lin_unit in ("fraction", "fraction_bound"):
        return dv, f"{dv * 100.0:.1f}%"
    return dv, f"{dv:+.2f}" if dv >= 0 else f"{dv:.2f}"


class V33InferenceEngine:
    """Authoritative V33 Champion Inference Engine."""

    def __init__(self, esmc_port: int = 18902, torch_threads: int = 16):
        self.t_boot = time.time()
        self.manifest = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
        self.endpoints_order = sorted(self.manifest["endpoints"].keys())
        self.tier1_endpoints = [e for e in self.endpoints_order if self.manifest["endpoints"][e]["tier"] == 1]
        self.tier2_endpoints = [e for e in self.endpoints_order if self.manifest["endpoints"][e]["tier"] == 2]

        # 1. Start / verify warm ESM-C 600M worker + load live encoders & 46,598-compound cache
        self.esmc_proc: Optional[subprocess.Popen] = None
        self._ensure_esmc_worker(esmc_port)
        self.hub = V33EncodingHub(esmc_port=esmc_port, torch_threads=torch_threads)

        # 2. Load Fast AD + Top-3 Analog Engine
        self.ad_engine = FastADAnalogEngine(
            BUNDLE_DIR / "ad_analog_index.npz",
            BUNDLE_DIR / "analog_lookup.json.gz",
        )

        # 3. Load ONLY pruned champion serving models referenced by v33_manifest.json (parallelized across CPU cores)
        from concurrent.futures import ThreadPoolExecutor

        t_m = time.time()
        inv = self.manifest["serving_inventory"]
        self.cpu_models: Dict[str, Any] = {}
        self.fusion_models: Dict[str, Any] = {}
        load_jobs = (
            [("cpu", ep, SERVING_DIR / f"{ep}_cpu.joblib") for ep in inv["cpu_endpoints_bundled"]]
            + [("fusion", ep, SERVING_DIR / f"{ep}_fusion.joblib") for ep in inv["fusion_endpoints_bundled"]]
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [(kind, ep, pool.submit(joblib.load, path)) for kind, ep, path in load_jobs]
            for kind, ep, fut in futs:
                if kind == "cpu":
                    self.cpu_models[ep] = fut.result()
                else:
                    self.fusion_models[ep] = fut.result()

        # Load referenced deep checkpoints into CPU eval mode
        self.deep_models: Dict[str, Dict[str, List[Tuple[Any, Dict[str, Any]]]]] = {}
        deep_refs = inv["deep_checkpoint_refs"]
        n_ckpts = 0
        for ep, kind_map in deep_refs.items():
            self.deep_models[ep] = {}
            for kind, ck_paths in kind_map.items():
                loaded = []
                for p in ck_paths:
                    loaded.append(load_deep_checkpoint(_resolve_deep_ckpt_path(p), device="cpu"))
                    n_ckpts += 1
                self.deep_models[ep][kind] = loaded

        # Load shared 46-endpoint unified MultiTaskPeptideNet46 folds (5 compact fp16 checkpoints)
        self.mt46_models: List[Tuple[Any, Dict[str, Any]]] = []
        mt46_refs = inv.get("mt46_checkpoint_refs") or []
        if mt46_refs and load_mt46_checkpoint is not None:
            for p in mt46_refs:
                rp = _resolve_deep_ckpt_path(p)
                if rp.exists():
                    self.mt46_models.append(load_mt46_checkpoint(rp, device="cpu"))

        self.deep_lock = threading.Lock()
        self._tune_estimator_threads(n_jobs=2)
        self.ep_pool = ThreadPoolExecutor(max_workers=EP_WORKERS)

        self.startup_sec = round(time.time() - self.t_boot, 2)
        print(
            f"[engine] READY in {self.startup_sec}s (models={time.time()-t_m:.1f}s | cpu={len(self.cpu_models)} "
            f"| fusion={len(self.fusion_models)} | deep_ckpts={n_ckpts} | mt46_folds={len(self.mt46_models)} | RSS={_get_rss_mb():.1f} MB)",
            flush=True,
        )

    def _tune_estimator_threads(self, n_jobs: int = 2) -> None:
        """Cap per-estimator OpenMP threads so 12 concurrent endpoint workers run contention-free on 64-core node23."""
        visited = set()

        def _walk(obj: Any) -> None:
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)
            if hasattr(obj, "n_jobs") and isinstance(getattr(obj, "n_jobs", None), int):
                try:
                    setattr(obj, "n_jobs", n_jobs)
                except Exception:
                    pass
            if hasattr(obj, "nthread") and isinstance(getattr(obj, "nthread", None), int):
                try:
                    setattr(obj, "nthread", n_jobs)
                except Exception:
                    pass
            if isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _walk(v)
            elif hasattr(obj, "estimators_"):
                _walk(getattr(obj, "estimators_"))
            elif hasattr(obj, "models"):
                _walk(getattr(obj, "models"))
            elif hasattr(obj, "base_estimator"):
                _walk(getattr(obj, "base_estimator"))

        _walk(self.cpu_models)
        _walk(self.fusion_models)

    def _ensure_esmc_worker(self, port: int) -> None:
        from v33_live_encoders import ESMCWorkerClient

        cli = ESMCWorkerClient(host="127.0.0.1", port=port, timeout=5.0)
        if cli.health().get("ok"):
            print(f"[engine] Attached to existing warm esmc_worker on 127.0.0.1:{port}", flush=True)
            return

        log_fp = BUNDLE_DIR / "esmc_worker.log"
        print(f"[engine] Starting persistent esmc_worker on 127.0.0.1:{port} (log={log_fp})...", flush=True)
        log_fh = open(log_fp, "a", encoding="utf-8")
        cmd = [
            str(ESMC_PYTHON),
            str(ESMC_WORKER_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--threads",
            str(ESMC_THREADS),
        ]
        if ESMC_MODEL_DIR:
            cmd.extend(["--model-dir", ESMC_MODEL_DIR])
        self.esmc_proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        t_wait = time.time()
        while time.time() - t_wait < 480.0:
            if self.esmc_proc.poll() is not None:
                raise RuntimeError(f"esmc_worker exited prematurely with code {self.esmc_proc.returncode}! Check {log_fp}")
            if cli.health().get("ok"):
                print(f"[engine] esmc_worker ready on 127.0.0.1:{port} in {time.time()-t_wait:.1f}s", flush=True)
                return
            time.sleep(0.5)
        raise TimeoutError(f"esmc_worker failed to become ready within 480s! Check {log_fp}")

    def shutdown(self) -> None:
        if self.esmc_proc is not None and self.esmc_proc.poll() is None:
            try:
                self.esmc_proc.terminate()
                self.esmc_proc.wait(timeout=5)
            except Exception:
                pass

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ok": True,
            "version": self.manifest.get("version", "v33.1"),
            "mode": "v33_live",
            "endpoints": len(self.endpoints_order),
            "tier1_endpoints": len(self.tier1_endpoints),
            "tier2_endpoints": len(self.tier2_endpoints),
            "encoders_ready": self.hub.encoders_status(),
            "models_loaded": {
                "cpu_bundles": len(self.cpu_models),
                "fusion_bundles": len(self.fusion_models),
                "deep_endpoints": len(self.deep_models),
                "mt46_folds": len(self.mt46_models),
            },
            "rss_mb": _get_rss_mb(),
            "startup_sec": self.startup_sec,
            "uptime_s": round(time.time() - self.t_boot, 1),
        }

    def _run_cpu_folds(
        self,
        folds: List[Tuple[int, Any, Any]],
        bank: LiveMiniBank,
        idx: np.ndarray,
        task_type: str,
        cond_dict: Dict[str, str],
        meta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        preds = []
        for _, pre, est in folds:
            X = pre.transform(bank, idx, default_cond=cond_dict, meta=meta)
            preds.append(_predict_cpu_est(est, X, task_type))
        return np.mean(np.stack(preds, axis=0), axis=0)

    def _compute_cpu_aux_meta(
        self,
        bank: LiveMiniBank,
        idx: np.ndarray,
        track: str,
        target_ep: str,
        aux_cache: Dict[Tuple[str, str], np.ndarray],
    ) -> np.ndarray:
        aux_list = [a for a in AUX_ENDPOINTS if a != target_ep]
        cols = []
        for a in aux_list:
            ckey = (a, track)
            if ckey not in aux_cache:
                cb = self.cpu_models[a]
                tb = cb[track]
                p = self._run_cpu_folds(
                    tb["folds"], bank, idx, cb["task_type"], cb["canonical_default_conditions"], meta=None
                )
                if cb["task_type"] == "classification" and tb.get("calibrator") is not None:
                    p = tb["calibrator"].transform(p)
                aux_cache[ckey] = p.astype(np.float32)
            cols.append(aux_cache[ckey])
        return np.column_stack(cols).astype(np.float32)

    def _run_cpu_stream(
        self,
        ep: str,
        track: str,
        bank: LiveMiniBank,
        idx: np.ndarray,
        cond_dict: Dict[str, str],
        aux_cache: Dict[Tuple[str, str], np.ndarray],
        stream_kind: str = "main",
    ) -> np.ndarray:
        cb = self.cpu_models[ep]
        if stream_kind == "seq_only":
            if cb.get("seq_only") is None or not bank.seq_valid[idx].any():
                return np.full(len(idx), np.nan, dtype=np.float64)
            out = np.full(len(idx), np.nan, dtype=np.float64)
            sv = np.where(bank.seq_valid[idx])[0]
            out[sv] = self._run_cpu_folds(cb["seq_only"]["folds"], bank, idx[sv], cb["task_type"], cond_dict, meta=None)
            return out

        if stream_kind == "aft":
            aft_key = "folds_dual" if track == "dual_modality" else "folds_smiles_only"
            folds = cb["aft"][aft_key]
            needs_meta = any(bool(getattr(pre, "has_meta_", False)) for _, pre, _ in folds)
            meta = self._compute_cpu_aux_meta(bank, idx, track, ep, aux_cache) if needs_meta else None
            return self._run_cpu_folds(folds, bank, idx, "regression", cond_dict, meta=meta)

        tb_ref = cb[track] if cb.get(track) is not None else cb.get("dual_modality")
        folds = tb_ref["folds"]
        needs_meta = any(bool(getattr(pre, "has_meta_", False)) for _, pre, _ in folds)
        meta = self._compute_cpu_aux_meta(bank, idx, track, ep, aux_cache) if needs_meta else None
        return self._run_cpu_folds(folds, bank, idx, cb["task_type"], cond_dict, meta=meta)

    def _run_fusion_stream(
        self,
        ep: str,
        track: str,
        bank: LiveMiniBank,
        idx: np.ndarray,
        cond_dict: Dict[str, str],
        aux_cache: Dict[Tuple[str, str], np.ndarray],
        cpu_pred_cache: Dict[Tuple[str, str, str], np.ndarray],
        is_aft: bool = False,
    ) -> np.ndarray:
        fb = self.fusion_models[ep]
        task_type = "regression" if is_aft else fb["task_type"]
        if is_aft:
            aft_blk = fb.get("aft") or {}
            stacker = aft_blk.get("stacker_dual") if track == "dual_modality" else aft_blk.get("stacker_smiles_only")
            if stacker is None:
                return np.full(len(idx), np.nan, dtype=np.float64)
            cand_heads = list(stacker.head_names_)
            active_heads = set(stacker.active_heads(tol=1e-9))
        else:
            tb = fb.get(track)
            if tb is None:
                return np.full(len(idx), np.nan, dtype=np.float64)
            stacker = tb["stacker"]
            cand_heads = tb["candidate_heads"]
            active_heads = set(tb["active_heads"])

        head_preds: Dict[str, np.ndarray] = {}
        for h in active_heads:
            if h == "cpu_dual":
                k = (ep, "dual_modality", "main")
                if k not in cpu_pred_cache:
                    cpu_pred_cache[k] = self._run_cpu_stream(ep, "dual_modality", bank, idx, cond_dict, aux_cache, "main")
                head_preds[h] = cpu_pred_cache[k]
            elif h == "cpu_smiles":
                k = (ep, "smiles_only", "main")
                if k not in cpu_pred_cache:
                    cpu_pred_cache[k] = self._run_cpu_stream(ep, "smiles_only", bank, idx, cond_dict, aux_cache, "main")
                head_preds[h] = cpu_pred_cache[k]
            elif h == "cpu_aft_dual":
                k = (ep, "dual_modality", "aft")
                if k not in cpu_pred_cache:
                    cpu_pred_cache[k] = self._run_cpu_stream(ep, "dual_modality", bank, idx, cond_dict, aux_cache, "aft")
                head_preds[h] = cpu_pred_cache[k]
            elif h == "cpu_aft_smiles":
                k = (ep, "smiles_only", "aft")
                if k not in cpu_pred_cache:
                    cpu_pred_cache[k] = self._run_cpu_stream(ep, "smiles_only", bank, idx, cond_dict, aux_cache, "aft")
                head_preds[h] = cpu_pred_cache[k]
            elif h == "cpu_seq_only":
                k = (ep, "seq_only", "seq_only")
                if k not in cpu_pred_cache:
                    cpu_pred_cache[k] = self._run_cpu_stream(ep, "dual_modality", bank, idx, cond_dict, aux_cache, "seq_only")
                head_preds[h] = cpu_pred_cache[k]
            elif h in fb["pretrained_base_heads"]:
                hobj = fb["pretrained_base_heads"][h]
                h_track = hobj["rep_cfg"]["track"]
                h_is_aft = bool(hobj.get("is_aft", False))
                valid_mask = bank.smiles_valid[idx] if h_track == "smiles_only" else np.ones(len(idx), dtype=bool)
                out_h = np.full(len(idx), np.nan, dtype=np.float64)
                v_pos = np.where(valid_mask)[0]
                if len(v_pos) > 0:
                    f_preds = []
                    for _, pre, est in hobj["folds"]:
                        X = pre.transform(bank, idx[v_pos], default_cond=cond_dict)
                        p = est.predict(X) if h_is_aft else _predict_fusion_est(est, X, task_type)
                        f_preds.append(p)
                    out_h[v_pos] = np.mean(np.stack(f_preds, axis=0), axis=0)
                head_preds[h] = out_h

        P_cols = [
            head_preds[h] if h in head_preds else np.full(len(idx), np.nan, dtype=np.float64)
            for h in cand_heads
        ]
        return stacker.predict(np.column_stack(P_cols))

    def _run_deep_stream(
        self,
        ep: str,
        track: str,
        kind: str,
        bank: LiveMiniBank,
        idx: np.ndarray,
        cond_dict: Dict[str, str],
        deep_cache: Dict[Tuple[str, str, str], np.ndarray],
    ) -> np.ndarray:
        ckey = (ep, track, kind)
        if ckey in deep_cache:
            return deep_cache[ckey]

        if kind == "deep":
            if ep in ("half_life_reg", "half_life_cls") and "halflife_specialist" in self.deep_models.get(ep, {}):
                p_mt = self._run_deep_stream(ep, track, "multitask", bank, idx, cond_dict, deep_cache)
                p_sp = self._run_deep_stream(ep, track, "halflife_specialist", bank, idx, cond_dict, deep_cache)
                out = 0.5 * p_mt + 0.5 * p_sp
                deep_cache[ckey] = out
                return out
            out = self._run_deep_stream(ep, track, "multitask", bank, idx, cond_dict, deep_cache)
            deep_cache[ckey] = out
            return out

        models_5 = self.deep_models[ep][kind]
        phys = bank.physchem[idx]
        sv = bank.smiles_valid[idx]
        qv = bank.seq_valid[idx]
        nc_s = bank.emb_blocks["ncaa_smiles_repr"][idx]
        nc_f = bank.emb_blocks["ncaa_fusion_hf"][idx]
        nc_p = bank.emb_blocks["ncaa_protein_proj"][idx]
        pc = bank.emb_blocks["peptideclm2"][idx]
        cb = bank.emb_blocks["chemberta_mtr"][idx]
        esmc_4 = np.stack(
            [
                bank.emb_blocks["esmc_mean"][idx],
                bank.emb_blocks["esmc_max"][idx],
                bank.emb_blocks["esmc_nterm5"][idx],
                bank.emb_blocks["esmc_cterm5"][idx],
            ],
            axis=1,
        )
        cond_rows = [cond_dict] * len(idx)

        preds = []
        with self.deep_lock:
            for model, ckpt in models_5:
                mu, _ = predict_from_numpy(
                    model, ckpt, ep, phys, nc_s, nc_f, nc_p, pc, cb, esmc_4, sv, qv,
                    cond_rows=cond_rows, eval_mode=track, device="cpu",
                )
                preds.append(mu)
        out = np.mean(np.stack(preds, axis=0), axis=0)
        deep_cache[ckey] = out
        return out

    def _run_mt46_stream(
        self,
        ep: str,
        track: str,
        bank: LiveMiniBank,
        idx: np.ndarray,
        cond_dict: Dict[str, str],
        bounds: Optional[List[float]],
        deep_cache: Dict[Tuple[str, str, str], np.ndarray],
    ) -> np.ndarray:
        ckey = (ep, track, "mt46")
        if ckey in deep_cache:
            return deep_cache[ckey]
        if not self.mt46_models or predict_mt46_from_numpy is None:
            return np.full(len(idx), np.nan, dtype=np.float64)

        phys = bank.physchem[idx]
        sv = bank.smiles_valid[idx]
        qv = bank.seq_valid[idx]
        nc_s = bank.emb_blocks["ncaa_smiles_repr"][idx]
        nc_f = bank.emb_blocks["ncaa_fusion_hf"][idx]
        nc_p = bank.emb_blocks["ncaa_protein_proj"][idx]
        pc = bank.emb_blocks["peptideclm2"][idx]
        cb = bank.emb_blocks["chemberta_mtr"][idx]
        esmc_4 = np.stack(
            [
                bank.emb_blocks["esmc_mean"][idx],
                bank.emb_blocks["esmc_max"][idx],
                bank.emb_blocks["esmc_nterm5"][idx],
                bank.emb_blocks["esmc_cterm5"][idx],
            ],
            axis=1,
        )
        cond_rows = [cond_dict] * len(idx)

        preds = []
        with self.deep_lock:
            for model, ckpt in self.mt46_models:
                mu, _ = predict_mt46_from_numpy(
                    model, ckpt, ep, phys, nc_s, nc_f, nc_p, pc, cb, esmc_4, sv, qv,
                    cond_rows=cond_rows, canonical_conditions=cond_dict, bounds=bounds,
                    eval_mode=track, device="cpu",
                )
                preds.append(mu)
        out = np.mean(np.stack(preds, axis=0), axis=0)
        deep_cache[ckey] = out
        return out

    def predict_batch_items(
        self,
        raw_items: List[Dict[str, Any]],
        user_cond: Dict[str, Any],
        include_tier2: bool = True,
        force_live_encode: bool = False,
        return_embeddings_debug: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        t_start = time.time()
        B = len(raw_items)
        if B == 0:
            return [], {"encode_ms": 0.0, "predict_ms": 0.0, "ad_ms": 0.0, "total_ms": 0.0}

        # 1. Resolve inputs (track, canonical SMILES, canonical sequence)
        resolved: List[Dict[str, Any]] = []
        item_errors: Dict[int, str] = {}
        valid_positions: List[int] = []
        for i, it in enumerate(raw_items):
            try:
                r = self.hub.resolve_item(it.get("smiles"), it.get("sequence"))
                r["name"] = it.get("name") or f"peptide_{i+1}"
                resolved.append(r)
                valid_positions.append(i)
            except Exception as exc:
                resolved.append({"name": it.get("name") or f"peptide_{i+1}", "error": str(exc)})
                item_errors[i] = str(exc)

        if not valid_positions:
            return [
                {"name": resolved[i]["name"], "status": "error", "error": item_errors[i]}
                for i in range(B)
            ], {"encode_ms": 0.0, "predict_ms": 0.0, "ad_ms": 0.0, "total_ms": round((time.time() - t_start) * 1000.0, 2)}

        valid_resolved = [resolved[i] for i in valid_positions]
        V = len(valid_resolved)

        # 2. Encode valid batch (smart bank/LRU cache or true live encoding)
        mini_bank, cache_hits, enc_timings = self.hub.encode_resolved_batch(
            valid_resolved, force_live_encode=force_live_encode
        )

        target_eps = self.endpoints_order if include_tier2 else self.tier1_endpoints

        # 3. Compute fast AD + Top-3 training analogs across all target endpoints
        t_ad0 = time.time()
        ad_by_ep = self.ad_engine.query_batch(
            mini_bank.morgan_bits,
            mini_bank.smiles_valid,
            [r["canonical_sequence"] for r in valid_resolved],
            target_eps,
        )
        ad_ms = (time.time() - t_ad0) * 1000.0

        # 4. Run champion inference grouped by track (`dual_modality` vs `smiles_only`)
        t_pred0 = time.time()
        track_groups: Dict[str, np.ndarray] = {}
        for tr in ("dual_modality", "smiles_only"):
            pos = np.array([k for k, r in enumerate(valid_resolved) if r["track"] == tr], dtype=np.int32)
            if len(pos) > 0:
                track_groups[tr] = pos

        # Store per-endpoint predictions aligned to valid_resolved [0..V-1]
        ep_outputs: Dict[str, List[Optional[Dict[str, Any]]]] = {ep: [None] * V for ep in target_eps}

        for track, pos_idx in track_groups.items():
            # Precompute read-only CPU aux meta features in parallel before fanning out across all 24 endpoints
            aux_cache: Dict[Tuple[str, str], np.ndarray] = {}

            def _precompute_aux(aux_ep: str, aux_tr: str) -> Tuple[Tuple[str, str], Optional[np.ndarray]]:
                cb = self.cpu_models.get(aux_ep)
                if cb is None or cb.get(aux_tr) is None:
                    return (aux_ep, aux_tr), None
                tb = cb[aux_tr]
                p = self._run_cpu_folds(
                    tb["folds"], mini_bank, pos_idx, cb["task_type"], cb["canonical_default_conditions"], meta=None
                )
                if cb["task_type"] == "classification" and tb.get("calibrator") is not None:
                    p = tb["calibrator"].transform(p)
                return (aux_ep, aux_tr), p.astype(np.float32)

            aux_tracks = ("dual_modality", "smiles_only") if track == "dual_modality" else ("smiles_only",)
            aux_futs = [
                self.ep_pool.submit(_precompute_aux, a_ep, a_tr)
                for a_ep in AUX_ENDPOINTS
                for a_tr in aux_tracks
            ]
            for fut in aux_futs:
                k_aux, val_aux = fut.result()
                if val_aux is not None:
                    aux_cache[k_aux] = val_aux

            def _eval_one_endpoint(ep: str) -> Tuple[str, List[Tuple[int, Dict[str, Any]]]]:
                cpu_pred_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
                deep_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
                ep_doc = self.manifest["endpoints"][ep]
                tr_doc = ep_doc[track]
                task_type = ep_doc["task_type"]
                bounds = ep_doc.get("bounds")
                weights: Dict[str, float] = tr_doc["weights"]
                cond_dict = _resolve_conditions(ep, ep_doc["canonical_conditions"], user_cond)

                # Run only active champion streams!
                streams: Dict[str, np.ndarray] = {}
                for fam, w in weights.items():
                    if float(w) <= 0.0:
                        continue
                    if fam == "cpu":
                        k = (ep, track, "main")
                        if k not in cpu_pred_cache:
                            cpu_pred_cache[k] = self._run_cpu_stream(ep, track, mini_bank, pos_idx, cond_dict, aux_cache, "main")
                        streams["cpu"] = cpu_pred_cache[k]
                    elif fam == "fusion":
                        streams["fusion"] = self._run_fusion_stream(
                            ep, track, mini_bank, pos_idx, cond_dict, aux_cache, cpu_pred_cache, is_aft=False
                        )
                    elif fam == "deep":
                        streams["deep"] = self._run_deep_stream(ep, track, "deep", mini_bank, pos_idx, cond_dict, deep_cache)
                    elif fam == "deep_specialist":
                        streams["deep_specialist"] = self._run_deep_stream(
                            ep, track, "halflife_specialist", mini_bank, pos_idx, cond_dict, deep_cache
                        )
                    elif fam == "deep_multitask":
                        streams["deep_multitask"] = self._run_deep_stream(
                            ep, track, "multitask", mini_bank, pos_idx, cond_dict, deep_cache
                        )
                    elif fam == "mt46":
                        streams["mt46"] = self._run_mt46_stream(
                            ep, track, mini_bank, pos_idx, cond_dict, bounds, deep_cache
                        )

                raw_pred = _weighted_blend(streams, weights, task_type=task_type, bounds=bounds)

                # Calibration (cls) or Conformal intervals (reg)
                cal_prob = None
                ivs = None
                if task_type == "classification":
                    cal_prob = _apply_frozen_calibrator(raw_pred, tr_doc.get("calibrator"))
                    final_val = cal_prob
                else:
                    final_val = raw_pred
                    max_t_vec = np.array([ad_by_ep[ep][int(p)]["max_tanimoto"] for p in pos_idx], dtype=np.float64)
                    ivs = _apply_frozen_conformal(final_val, tr_doc["conformal"], max_tanimoto=max_t_vec)

                # Censored-aware diagnostic prediction on key censored endpoints
                cens_pred_vec = None
                cens_note_base = None
                if ep_doc.get("censored_head") is not None and (
                    ep in (
                        "hemolysis_HC50", "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans",
                        "half_life_reg", "PPB", "solubility_reg", "F",
                    )
                    or ep_doc.get("expansion_wave") == "v33.1"
                ):
                    ch_tr = ep_doc["censored_head"][track]
                    ch_w = ch_tr["weights"]
                    ch_streams: Dict[str, np.ndarray] = {"champion_exact": final_val}
                    for c_fam, cw in ch_w.items():
                        if float(cw) <= 0.0 or c_fam == "champion_exact":
                            continue
                        if c_fam == "cpu_aft" and ep in self.cpu_models and self.cpu_models[ep].get("aft") is not None:
                            k = (ep, track, "aft")
                            if k not in cpu_pred_cache:
                                cpu_pred_cache[k] = self._run_cpu_stream(ep, track, mini_bank, pos_idx, cond_dict, aux_cache, "aft")
                            ch_streams["cpu_aft"] = cpu_pred_cache[k]
                        elif c_fam == "fusion_aft" and ep in self.fusion_models and self.fusion_models[ep].get("aft") is not None:
                            ch_streams["fusion_aft"] = self._run_fusion_stream(
                                ep, track, mini_bank, pos_idx, cond_dict, aux_cache, cpu_pred_cache, is_aft=True
                            )
                        elif c_fam == "deep" and ep in self.deep_models:
                            ch_streams["deep"] = self._run_deep_stream(ep, track, "deep", mini_bank, pos_idx, cond_dict, deep_cache)
                    if any(k in ch_streams for k in ch_w):
                        cens_pred_vec = _weighted_blend(ch_streams, ch_w, task_type="regression", bounds=bounds)
                        n_c = ep_doc["counts"]["n_censored_total"]
                        cens_note_base = f"Trained with {n_c} censored records; censored-aware AFT/Tobit diagnostic head available."

                # Populate formatted endpoint prediction entries
                ep_rows: List[Tuple[int, Dict[str, Any]]] = []
                for loc_j, v_i in enumerate(pos_idx):
                    v_i_int = int(v_i)
                    fv = float(final_val[loc_j])
                    dv, dstr = _format_display_value(fv, ep_doc)
                    ad_info = ad_by_ep[ep][v_i_int]
                    entry: Dict[str, Any] = {
                        "endpoint": ep,
                        "display_en": ep_doc["display_en"],
                        "display_zh": ep_doc["display_zh"],
                        "tier": ep_doc["tier"],
                        "domain": ep_doc["domain"],
                        "task_type": task_type,
                        "raw_value": round(fv, 5),
                        "display_value": dv,
                        "display_text": dstr,
                        "unit": ep_doc["unit"],
                        "linear_unit": ep_doc["linear_unit"],
                        "log10_space": ep_doc["log10_space"],
                        "winning_family": tr_doc["winning_family"],
                        "weights": weights,
                        "conditions_used": cond_dict,
                        "ad_tier": ad_info["ad_tier"],
                        "max_tanimoto": ad_info["max_tanimoto"],
                        "top5_tanimoto": ad_info["top5_mean_tanimoto"],
                        "top3_analogs": ad_info["top3_analogs"],
                    }
                    if task_type == "classification":
                        entry["calibrated_prob"] = round(float(cal_prob[loc_j]), 5)
                        entry["raw_uncalibrated_prob"] = round(float(raw_pred[loc_j]), 5)
                        entry["ci80_low"] = None
                        entry["ci80_high"] = None
                        entry["ci90_low"] = None
                        entry["ci90_high"] = None
                    else:
                        lo80, hi80 = float(ivs["lo80"][loc_j]), float(ivs["hi80"][loc_j])
                        lo90, hi90 = float(ivs["lo90"][loc_j]), float(ivs["hi90"][loc_j])
                        entry["calibrated_prob"] = None
                        entry["ci80_low"] = round(lo80, 5)
                        entry["ci80_high"] = round(hi80, 5)
                        entry["ci90_low"] = round(lo90, 5)
                        entry["ci90_high"] = round(hi90, 5)
                        if ep_doc["log10_space"]:
                            entry["ci80_linear"] = [round(10.0 ** np.clip(lo80, -15, 15), 4), round(10.0 ** np.clip(hi80, -15, 15), 4)]
                            entry["ci90_linear"] = [round(10.0 ** np.clip(lo90, -15, 15), 4), round(10.0 ** np.clip(hi90, -15, 15), 4)]
                    if cens_pred_vec is not None:
                        cv = float(cens_pred_vec[loc_j])
                        cdv, cdstr = _format_display_value(cv, ep_doc)
                        entry["censored_note"] = cens_note_base
                        entry["censored_diagnostic_raw"] = round(cv, 5)
                        entry["censored_diagnostic_display"] = cdv
                        entry["censored_diagnostic_text"] = cdstr
                    ep_rows.append((v_i_int, entry))
                return ep, ep_rows

            ep_futs = [self.ep_pool.submit(_eval_one_endpoint, ep) for ep in target_eps]
            for fut in ep_futs:
                ep_name, ep_rows = fut.result()
                for v_i_int, entry in ep_rows:
                    ep_outputs[ep_name][v_i_int] = entry

        predict_ms = (time.time() - t_pred0) * 1000.0
        total_ms = (time.time() - t_start) * 1000.0

        # Assemble final response list aligned to original raw_items order
        valid_pos_to_v = {orig_i: v_i for v_i, orig_i in enumerate(valid_positions)}
        results: List[Dict[str, Any]] = []
        for orig_i in range(B):
            if orig_i in item_errors:
                results.append({
                    "name": resolved[orig_i]["name"],
                    "status": "error",
                    "error": item_errors[orig_i],
                })
                continue
            v_i = valid_pos_to_v[orig_i]
            r = valid_resolved[v_i]
            csmi = r["canonical_smiles"]
            anch = compute_halflife_mechanism_anchor(csmi) if csmi else {
                "anchor_class": "non_acylated", "diacid_internal_c": 0, "monoacyl_linear_c": 0,
                "anchor_mid_h": None, "anchor_range_h": None,
            }
            ep_dict = {ep: ep_outputs[ep][v_i] for ep in target_eps}
            ep_list = [ep_outputs[ep][v_i] for ep in target_eps]

            res_obj: Dict[str, Any] = {
                "status": "ok",
                "version": self.manifest.get("version", "v33.1"),
                "name": r["name"],
                "canonical_smiles": csmi or None,
                "canonical_sequence": r["canonical_sequence"] or None,
                "smiles_generated_from_sequence": bool(r["from_canonical_seq"]),
                "modality_track": r["track"],
                "cache_hit": bool(cache_hits[v_i]),
                "force_live_encode": bool(force_live_encode),
                "mechanism_anchor": anch,
                "endpoints": ep_dict,
                "predictions": ep_list,
                "timing_ms": {
                    "encode_ms": enc_timings["encode_total_ms"],
                    "smiles_live_ms": enc_timings["smiles_live_ms"],
                    "esmc_live_ms": enc_timings["esmc_live_ms"],
                    "predict_ms": round(predict_ms, 2),
                    "ad_ms": round(ad_ms, 2),
                    "total_ms": round(total_ms, 2),
                },
            }
            if return_embeddings_debug:
                res_obj["embeddings_debug"] = {
                    "physchem_83": mini_bank.physchem[v_i].tolist(),
                    "ncaa_smiles_repr": mini_bank.emb_blocks["ncaa_smiles_repr"][v_i].tolist(),
                    "ncaa_fusion_hf": mini_bank.emb_blocks["ncaa_fusion_hf"][v_i].tolist(),
                    "ncaa_protein_proj": mini_bank.emb_blocks["ncaa_protein_proj"][v_i].tolist(),
                    "peptideclm2": mini_bank.emb_blocks["peptideclm2"][v_i].tolist(),
                    "chemberta_mtr": mini_bank.emb_blocks["chemberta_mtr"][v_i].tolist(),
                    "esmc_mean": mini_bank.emb_blocks["esmc_mean"][v_i].tolist(),
                    "esmc_max": mini_bank.emb_blocks["esmc_max"][v_i].tolist(),
                    "esmc_nterm5": mini_bank.emb_blocks["esmc_nterm5"][v_i].tolist(),
                    "esmc_cterm5": mini_bank.emb_blocks["esmc_cterm5"][v_i].tolist(),
                }
            results.append(res_obj)

        batch_timings = {
            "encode_ms": enc_timings["encode_total_ms"],
            "predict_ms": round(predict_ms, 2),
            "ad_ms": round(ad_ms, 2),
            "total_ms": round(total_ms, 2),
        }
        return results, batch_timings


ENGINE: Optional[V33InferenceEngine] = None
MANIFEST_BYTES: bytes = b"{}"


class V33RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"[http] {self.address_string()} - {fmt % args}\n")
        sys.stdout.flush()

    def _send_json_bytes(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_json_bytes(status, body)

    def _read_json(self) -> Dict[str, Any]:
        clen = int(self.headers.get("Content-Length", "0"))
        if clen <= 0:
            return {}
        if clen > 32 * 1024 * 1024:
            raise ValueError("Request body exceeds 32 MB limit.")
        raw = self.rfile.read(clen)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/health":
            assert ENGINE is not None
            self._send_json(200, ENGINE.health())
            return
        if path == "/endpoints":
            self._send_json_bytes(200, MANIFEST_BYTES)
            return
        self._send_json(404, {"status": "error", "error": f"Unknown GET path: {path}"})

    def do_POST(self) -> None:
        assert ENGINE is not None
        path = self.path.split("?")[0]
        try:
            body = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "error", "error": f"Invalid JSON body: {exc}"})
            return

        # Merge flat condition keys & optional nested `conditions` dict
        cond_src = dict(body.get("conditions") or {}) if isinstance(body.get("conditions"), dict) else {}
        for ck in ("rbc_species", "pk_route", "pk_species", "pk_matrix"):
            if body.get(ck) is not None:
                cond_src[ck] = body[ck]

        include_tier2 = bool(body.get("include_tier2", True))
        force_live = bool(body.get("force_live_encode", False))
        ret_dbg = bool(body.get("return_embeddings_debug", False))

        try:
            if path == "/predict":
                results, _ = ENGINE.predict_batch_items(
                    [body], cond_src, include_tier2=include_tier2,
                    force_live_encode=force_live, return_embeddings_debug=ret_dbg,
                )
                res = results[0]
                code = 200 if res.get("status") == "ok" else 400
                self._send_json(code, res)
                return

            if path == "/batch":
                items = body.get("items")
                if not isinstance(items, list) or len(items) == 0:
                    self._send_json(400, {"status": "error", "error": "Field 'items' must be a non-empty list."})
                    return
                if len(items) > 500:
                    self._send_json(400, {"status": "error", "error": f"Batch size {len(items)} exceeds max 500."})
                    return
                results, timings = ENGINE.predict_batch_items(
                    items, cond_src, include_tier2=include_tier2,
                    force_live_encode=force_live, return_embeddings_debug=False,
                )
                n_ok = sum(1 for r in results if r.get("status") == "ok")
                self._send_json(200, {
                    "status": "ok",
                    "version": ENGINE.manifest.get("version", "v33.1"),
                    "total": len(results),
                    "succeeded": n_ok,
                    "failed": len(results) - n_ok,
                    "timing_ms": timings,
                    "results": results,
                })
                return

            self._send_json(404, {"status": "error", "error": f"Unknown POST path: {path}"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"status": "error", "error": str(exc)})


def main() -> None:
    global ENGINE, MANIFEST_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--esmc-port", type=int, default=18902)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    print(f"=== Starting Peptide OmniPanel V33 Live Inference Service on {args.host}:{args.port} ===", flush=True)
    ENGINE = V33InferenceEngine(esmc_port=args.esmc_port, torch_threads=args.threads)
    MANIFEST_BYTES = (BUNDLE_DIR / "v33_manifest.json").read_bytes()

    server = ThreadingHTTPServer((args.host, args.port), V33RequestHandler)
    server.daemon_threads = True

    def _handle_sig(signum, frame):
        print(f"[server] Signal {signum} received, shutting down cleanly...", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    print(f"[server] Listening on http://{args.host}:{args.port} (V33 live mode, {len(ENGINE.endpoints_order)} endpoints)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if ENGINE is not None:
            ENGINE.shutdown()


if __name__ == "__main__":
    main()
