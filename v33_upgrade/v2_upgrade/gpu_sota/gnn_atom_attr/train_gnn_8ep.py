#!/usr/bin/env python3
"""Plan B-2 Upgrade 3 — Train Pure-PyTorch Molecular E-GAT + Task-Specific WeightAndSum
Multi-Task Network across the 8 Flagship Web Endpoints on Local Windows RTX 5080 (bf16).

8 Flagship Web Endpoints:
  PAMPA, Caco2, hemolysis_cls, hemolysis_HC50, overall_peptide_toxicity,
  cytotoxicity_cls, cytotoxicity_CC50, half_life_reg

Strict Zero-Leakage Protocol:
  1. Per-endpoint 5-fold CV + Untouched 20% Holdout:
     For each target endpoint E and fold k in 0..4, masks out
     forbidden_clusters = HoldoutClusters(E) | ValFoldClusters(E, k) AND forbidden_cidx
     across ALL 8 endpoints (`assert len(train_clusters & forbidden_clusters) == 0`).
  2. Compact Unified 8-Head 5-Fold Deployment Ensemble (`gnn_8ep_5fold_fp16.pt`, < 25 MB):
     Strictly excludes ALL Holdout clusters across all 8 endpoints AND all Case17 clusters
     so it has ZERO holdout or Case17 contamination, and serves all 8 endpoints + per-atom
     attributions in < 200 ms on CPU!
  3. Comprehensive unit test on Semaglutide, Cyclosporine A, Micafungin, and Magainin-2!
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V33_ROOT = Path(r"D:\work\中检院\v33_upgrade")
GPU_SOTA_ROOT = V33_ROOT / "v2_upgrade" / "gpu_sota"
OUT_GNN_DIR = GPU_SOTA_ROOT / "gnn_atom_attr"
CACHE_V331_DIR = GPU_SOTA_ROOT / "_cache" / "expansion_v331"
CORE24_DATA_DIR = V33_ROOT / "models_gpu_deep" / "_cache" / "data"

if str(GPU_SOTA_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_SOTA_ROOT))
if str(V33_ROOT / "models_cpu") not in sys.path:
    sys.path.insert(0, str(V33_ROOT / "models_cpu"))

from gnn_atom_attribution import (  # noqa: E402
    GNN_8_ENDPOINTS,
    GNN_8_TASK_META,
    GNNAtomAttributionEngine,
    PurePyTorchEGATNet,
    collate_graphs,
    mol_to_graph_numpy,
    parse_peptide_input,
    render_atom_attribution_svg,
)
from v33_metrics import classification_metrics, regression_metrics  # noqa: E402

csv.field_size_limit(10_000_000)
CENSOR_TO_INT = {"exact": 0, "right": 1, "left": 2, "interval": 3}

MODEL_CFG = {
    "d_node": 160,
    "d_edge": 48,
    "n_layers": 4,
    "n_heads": 4,
    "dropout": 0.18,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _featurize_chunk(items: List[Tuple[int, str]]) -> List[Tuple[int, Optional[Dict[str, np.ndarray]]]]:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for cidx, smi in items:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                out.append((cidx, None))
            else:
                out.append((cidx, mol_to_graph_numpy(mol)))
        except Exception:
            out.append((cidx, None))
    return out


def tobit_huber_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    y_lo: torch.Tensor,
    y_hi: torch.Tensor,
    censor: torch.Tensor,
    has_censor: bool = True,
) -> torch.Tensor:
    mu = pred[:, 0].float()
    sigma = F.softplus(pred[:, 1].float() + 0.5413).clamp(min=0.10, max=4.0)
    y = y.float()
    nll_ex = 0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)
    hub_ex = F.huber_loss(mu, y, reduction="none", delta=1.0)
    loss_ex = 0.5 * nll_ex + 0.5 * hub_ex
    if not has_censor:
        return loss_ex.mean()

    y_lo, y_hi = y_lo.float(), y_hi.float()
    z_r = ((mu - y_lo) / sigma).clamp(-15.0, 15.0)
    loss_r = 0.5 * (-torch.special.log_ndtr(z_r)) + 0.5 * F.huber_loss(F.relu(y_lo - mu), torch.zeros_like(mu), reduction="none", delta=1.0)
    z_l = ((y_hi - mu) / sigma).clamp(-15.0, 15.0)
    loss_l = 0.5 * (-torch.special.log_ndtr(z_l)) + 0.5 * F.huber_loss(F.relu(mu - y_hi), torch.zeros_like(mu), reduction="none", delta=1.0)
    u = ((y_hi - mu) / sigma).clamp(-15.0, 15.0)
    l = ((y_lo - mu) / sigma).clamp(-15.0, 15.0)
    u = torch.maximum(u, l + 1e-4)
    flip = u > 0.0
    a = torch.where(flip, -l, u)
    b = torch.where(flip, -u, l)
    la = torch.special.log_ndtr(a)
    lb = torch.special.log_ndtr(b)
    log_p = la + torch.log1p(-torch.exp((lb - la).clamp(max=-1e-6)))
    loss_i = 0.5 * (-log_p.clamp(min=-25.0)) + 0.5 * F.huber_loss(F.relu(y_lo - mu) + F.relu(mu - y_hi), torch.zeros_like(mu), reduction="none", delta=1.0)

    return torch.where(censor == 0, loss_ex, torch.where(censor == 1, loss_r, torch.where(censor == 2, loss_l, loss_i))).mean()


def cls_bce_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_sm = y.float() * 0.98 + 0.01
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), y_sm)


class GraphBank8:
    """Loads all 8 flagship web endpoints + pre-featurizes all unique valid SMILES graphs."""

    def __init__(self) -> None:
        t0 = time.time()
        feat_dir = CACHE_V331_DIR / "features"
        self.key_to_idx: Dict[str, int] = {}
        with (feat_dir / "compound_index.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                self.key_to_idx[r["compound_key"]] = int(r["compound_idx"])

        # Map compound_idx -> canonical_smiles from merged V33.1 compound table
        self.idx_to_smiles: Dict[int, str] = {}
        with (CACHE_V331_DIR / "data" / "compounds_v331_merged.tsv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ck = r["compound_key"]
                if ck in self.key_to_idx:
                    smi = (r.get("canonical_smiles") or "").strip()
                    if smi:
                        self.idx_to_smiles[self.key_to_idx[ck]] = smi

        self.endpoints: Dict[str, Dict[str, Any]] = {}
        self.all_case17_clusters: Set[str] = set()
        self.all_holdout_clusters: Set[str] = set()
        needed_cidx: Set[int] = set()

        for ep in GNN_8_ENDPOINTS:
            ep_path = (
                (CACHE_V331_DIR / "data" / "endpoints" / f"{ep}.tsv")
                if ep == "cytotoxicity_CC50"
                else (CORE24_DATA_DIR / "endpoints" / f"{ep}.tsv")
            )
            with ep_path.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))

            # Keep ONLY rows with valid SMILES (molecular GNN operates on SMILES graphs!)
            valid_rows = [r for r in rows if self.key_to_idx[r["compound_key"]] in self.idx_to_smiles]
            cidx = np.array([self.key_to_idx[r["compound_key"]] for r in valid_rows], dtype=np.int64)
            needed_cidx.update(cidx.tolist())
            cluster_ids = np.array([r["cluster_id"] for r in valid_rows], dtype=object)
            is_holdout = np.array([int(r["is_holdout"]) for r in valid_rows], dtype=np.int32)
            fold_id = np.array([int(r["fold_id"]) for r in valid_rows], dtype=np.int32)
            case17_ex = np.array([int(r["case17_excluded"]) for r in valid_rows], dtype=np.int32)
            censor_str = np.array([r["censor_type"] for r in valid_rows], dtype=object)
            censor_int = np.array([CENSOR_TO_INT[c] for c in censor_str], dtype=np.int64)
            y_val = np.array([float(r["y_value"]) for r in valid_rows], dtype=np.float64)
            y_lo = np.array([float(r["y_lower"]) for r in valid_rows], dtype=np.float64)
            y_hi = np.array([float(r["y_upper"]) for r in valid_rows], dtype=np.float64)

            cv_idx = np.where(is_holdout == 0)[0]
            ho_idx = np.where(is_holdout == 1)[0]
            assert set(cluster_ids[cv_idx]).isdisjoint(set(cluster_ids[ho_idx])), f"{ep}: HO/CV cluster leak!"
            self.all_case17_clusters.update(cluster_ids[case17_ex == 1].tolist())
            self.all_holdout_clusters.update(cluster_ids[ho_idx].tolist())

            self.endpoints[ep] = {
                "endpoint": ep,
                "record_ids": [r["record_id"] for r in valid_rows],
                "compound_keys": [r["compound_key"] for r in valid_rows],
                "compound_idx": cidx,
                "cluster_ids": cluster_ids,
                "is_holdout": is_holdout,
                "fold_id": fold_id,
                "case17_excluded": case17_ex,
                "censor_str": censor_str,
                "censor_int": censor_int,
                "y_value": y_val,
                "y_lower": y_lo,
                "y_upper": y_hi,
                "cv_idx": cv_idx,
                "ho_idx": ho_idx,
            }

        # Parallel RDKit molecular graph featurization across unique SMILES
        items = [(ci, self.idx_to_smiles[ci]) for ci in sorted(needed_cidx)]
        n_workers = min(16, max(1, (os.cpu_count() or 8) - 2))
        chunk_sz = max(128, len(items) // (n_workers * 4))
        chunks = [items[i : i + chunk_sz] for i in range(0, len(items), chunk_sz)]
        self.graphs: Dict[int, Dict[str, np.ndarray]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for res_chunk in pool.map(_featurize_chunk, chunks):
                for ci, g in res_chunk:
                    if g is not None:
                        self.graphs[ci] = g

        # Filter out any rare SMILES that failed RDKit parsing (should be 0)
        for ep, d in self.endpoints.items():
            ok = np.array([int(ci) in self.graphs for ci in d["compound_idx"]], dtype=bool)
            if not ok.all():
                for k in ("compound_idx", "cluster_ids", "is_holdout", "fold_id", "case17_excluded", "censor_str", "censor_int", "y_value", "y_lower", "y_upper"):
                    d[k] = d[k][ok]
                d["record_ids"] = [r for r, m in zip(d["record_ids"], ok) if m]
                d["compound_keys"] = [r for r, m in zip(d["compound_keys"], ok) if m]
                d["cv_idx"] = np.where(d["is_holdout"] == 0)[0]
                d["ho_idx"] = np.where(d["is_holdout"] == 1)[0]

        print(
            f"[GRAPH BANK READY] {len(self.graphs):,} unique valid peptide graphs across 8 flagship endpoints in {time.time() - t0:.1f}s",
            flush=True,
        )


@torch.no_grad()
def predict_gnn_endpoint(
    model: PurePyTorchEGATNet,
    graphs: Dict[int, Dict[str, np.ndarray]],
    cidx_arr: np.ndarray,
    ep: str,
    y_scaler: Tuple[float, float],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    meta = GNN_8_TASK_META[ep]
    ym, ys = y_scaler
    n = len(cidx_arr)
    preds = np.full(n, np.nan, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        g_list = [graphs[int(ci)] for ci in cidx_arr[s:e]]
        batch = collate_graphs(g_list, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw, _, _ = model.forward_task(batch, ep)
        if meta["task_type"] == "regression":
            val = raw[:, 0].float().cpu().numpy().astype(np.float64) * ys + ym
            if meta["bounds"] is not None:
                val = np.clip(val, meta["bounds"][0], meta["bounds"][1])
            preds[s:e] = val
        else:
            preds[s:e] = np.clip(torch.sigmoid(raw[:, 0].float()).cpu().numpy().astype(np.float64), 1e-6, 1.0 - 1e-6)
    return preds


def _eval_ep_score(ep: str, y_raw: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray, c_int: np.ndarray, pred: np.ndarray) -> float:
    if GNN_8_TASK_META[ep]["task_type"] == "regression":
        ex = (c_int == 0) & np.isfinite(pred)
        mae = float(np.mean(np.abs(y_raw[ex] - pred[ex]))) if ex.any() else 10.0
        r_m, l_m = c_int == 1, c_int == 2
        viols = []
        if r_m.any():
            viols.append(np.maximum(0.0, y_lo[r_m] - pred[r_m]))
        if l_m.any():
            viols.append(np.maximum(0.0, pred[l_m] - y_hi[l_m]))
        v = float(np.mean(np.concatenate(viols))) if viols else 0.0
        return -(mae + 0.35 * v)
    from sklearn.metrics import brier_score_loss, roc_auc_score
    yt = y_raw.astype(np.int32)
    if len(np.unique(yt)) < 2:
        return -1.0
    pp = np.clip(pred, 1e-6, 1.0 - 1e-6)
    return float(roc_auc_score(yt, pp) - 0.20 * brier_score_loss(yt, pp))


def train_per_endpoint_zero_leak_gnn(bank: GraphBank8, device: torch.device) -> Dict[str, Any]:
    """Strict per-endpoint 5-fold zero-cross-endpoint-cluster-leakage CV + Holdout benchmark for all 8 endpoints."""
    ep_summaries: Dict[str, Any] = {}
    total_leaked_cl = 0
    total_leaked_cidx = 0

    for ep_i, target_ep in enumerate(GNN_8_ENDPOINTS, 1):
        t0 = time.time()
        d_t = bank.endpoints[target_ep]
        meta_t = GNN_8_TASK_META[target_ep]
        cv_rows = d_t["cv_idx"]
        ho_rows = d_t["ho_idx"]
        n_cv, n_ho = len(cv_rows), len(ho_rows)
        orig_to_cv = {int(oi): pos for pos, oi in enumerate(cv_rows)}

        oof_pred = np.full(n_cv, np.nan, dtype=np.float64)
        ho_folds = []
        fold_audits = []

        for f in range(5):
            set_seed(42 + ep_i * 101 + f * 17)
            va_rows = cv_rows[d_t["fold_id"][cv_rows] == f]
            tr_rows = cv_rows[d_t["fold_id"][cv_rows] != f]

            forbidden_clusters = set(d_t["cluster_ids"][ho_rows].tolist()) | set(d_t["cluster_ids"][va_rows].tolist())
            forbidden_cidx = set(d_t["compound_idx"][ho_rows].tolist()) | set(d_t["compound_idx"][va_rows].tolist())

            task_train_rows: Dict[str, np.ndarray] = {target_ep: tr_rows}
            train_clusters_all: Set[str] = set(d_t["cluster_ids"][tr_rows].tolist())
            train_cidx_all: Set[int] = set(d_t["compound_idx"][tr_rows].tolist())

            for aux_ep in GNN_8_ENDPOINTS:
                if aux_ep == target_ep:
                    continue
                d_a = bank.endpoints[aux_ep]
                cand = d_a["cv_idx"]
                keep = np.array(
                    [
                        (d_a["cluster_ids"][i] not in forbidden_clusters) and (int(d_a["compound_idx"][i]) not in forbidden_cidx)
                        for i in cand
                    ],
                    dtype=bool,
                )
                kept = cand[keep]
                if len(kept) >= 8:
                    task_train_rows[aux_ep] = kept
                    train_clusters_all.update(d_a["cluster_ids"][kept].tolist())
                    train_cidx_all.update(d_a["compound_idx"][kept].tolist())

            # HARD ZERO-LEAKAGE ASSERTION
            leaked_cl = len(train_clusters_all & forbidden_clusters)
            leaked_ci = len(train_cidx_all & forbidden_cidx)
            assert leaked_cl == 0 and leaked_ci == 0, f"GNN LEAK in {target_ep} fold {f}: cl={leaked_cl}, ci={leaked_ci}"
            total_leaked_cl += leaked_cl
            total_leaked_cidx += leaked_ci
            fold_audits.append(
                {
                    "fold": f,
                    "n_target_train": int(len(tr_rows)),
                    "n_target_val": int(len(va_rows)),
                    "n_train_clusters_all_8": len(train_clusters_all),
                    "n_forbidden_clusters": len(forbidden_clusters),
                    "n_leaked_clusters": 0,
                    "n_leaked_compounds": 0,
                }
            )

            # Fit per-task y scalers strictly on training rows
            y_scalers: Dict[str, Tuple[float, float]] = {}
            task_y_scaled: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = {}
            for ep_k, r_idx in task_train_rows.items():
                dk = bank.endpoints[ep_k]
                yv = dk["y_value"][r_idx].copy()
                yl = dk["y_lower"][r_idx].copy()
                yu = dk["y_upper"][r_idx].copy()
                ct = dk["censor_int"][r_idx].copy()
                if GNN_8_TASK_META[ep_k]["task_type"] == "regression":
                    ex = ct == 0
                    ref = yv[ex] if ex.sum() >= 3 else yv
                    ym, ys = float(np.mean(ref)), float(np.std(ref))
                    if not np.isfinite(ys) or ys < 1e-4:
                        ys = 1.0
                    y_scalers[ep_k] = (ym, ys)
                    yv, yl, yu = (yv - ym) / ys, (yl - ym) / ys, (yu - ym) / ys
                else:
                    y_scalers[ep_k] = (0.0, 1.0)
                task_y_scaled[ep_k] = (yv.astype(np.float32), yl.astype(np.float32), yu.astype(np.float32), ct, bool((ct != 0).any()))

            model = PurePyTorchEGATNet(endpoints=list(task_train_rows.keys()), **MODEL_CFG).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2, fused=True)

            epochs = 26 if n_cv < 3000 else 22
            bt = min(192, max(32, len(tr_rows) // 6))
            steps_per_ep = max(14, min(65, math.ceil(len(tr_rows) / bt)))
            total_steps = epochs * steps_per_ep
            warmup = max(1, int(0.10 * total_steps))
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lambda st: (st + 1) / warmup if st < warmup else 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * (st - warmup) / max(1, total_steps - warmup))),
            )

            aux_eps = [e for e in task_train_rows if e != target_ep]
            rng = np.random.default_rng(42 + ep_i * 31 + f)

            best_sc = -1e9
            best_state = None
            va_cidx = d_t["compound_idx"][va_rows]
            ho_cidx = d_t["compound_idx"][ho_rows]

            for ep_num in range(epochs):
                model.train()
                aux_w = 0.50 if ep_num < int(0.55 * epochs) else 0.15
                perm = rng.permutation(len(tr_rows))
                ptr = 0

                for _ in range(steps_per_ep):
                    if ptr + bt > len(tr_rows):
                        perm = rng.permutation(len(tr_rows))
                        ptr = 0
                    loc_t = perm[ptr : ptr + bt]
                    ptr += bt

                    # Build combined graph batch (target + 2 aux tasks)
                    g_list = [bank.graphs[int(d_t["compound_idx"][tr_rows[i]])] for i in loc_t]
                    slices = [(target_ep, 0, len(loc_t), loc_t)]
                    cur = len(loc_t)

                    if aux_eps:
                        chosen = rng.choice(aux_eps, size=min(2, len(aux_eps)), replace=False)
                        for a_ep in chosen:
                            ar = task_train_rows[a_ep]
                            ba = min(96, len(ar))
                            loc_a = rng.integers(0, len(ar), size=ba)
                            da = bank.endpoints[a_ep]
                            g_list.extend(bank.graphs[int(da["compound_idx"][ar[i]])] for i in loc_a)
                            slices.append((a_ep, cur, cur + ba, loc_a))
                            cur += ba

                    batch = collate_graphs(g_list, device)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        h_atom = model.encode_atoms(batch)
                        bidx = batch["batch_idx"]
                        gctx = batch["global_ctx"]
                        ng = int(batch["num_graphs"])
                        # Evaluate only needed task heads (full graph slice then slice output)
                        outs = {}
                        for ep_k, s0, e0, loc_k in slices:
                            pred_all, _, _ = model.heads[ep_k](h_atom, bidx, gctx, ng)
                            outs[ep_k] = (pred_all[s0:e0], loc_k)

                    # Losses in float32
                    losses = []
                    for i_sl, (ep_k, _, _, loc_k) in enumerate(slices):
                        pred_k, _ = outs[ep_k]
                        yv, yl, yu, ct, has_c = task_y_scaled[ep_k]
                        yt = torch.from_numpy(yv[loc_k]).to(device)
                        if GNN_8_TASK_META[ep_k]["task_type"] == "regression":
                            lk = tobit_huber_loss(
                                pred_k, yt,
                                torch.from_numpy(yl[loc_k]).to(device),
                                torch.from_numpy(yu[loc_k]).to(device),
                                torch.from_numpy(ct[loc_k]).to(device),
                                has_censor=has_c,
                            )
                        else:
                            lk = cls_bce_loss(pred_k, yt)
                        losses.append(lk if i_sl == 0 else (aux_w / max(1, len(slices) - 1)) * lk)
                    loss = sum(losses)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                    opt.step()
                    sched.step()

                if (ep_num >= 6 and ep_num % 2 == 0) or ep_num == epochs - 1:
                    p_va = predict_gnn_endpoint(model, bank.graphs, va_cidx, target_ep, y_scalers[target_ep], device)
                    sc = _eval_ep_score(
                        target_ep, d_t["y_value"][va_rows], d_t["y_lower"][va_rows], d_t["y_upper"][va_rows], d_t["censor_int"][va_rows], p_va
                    )
                    if sc > best_sc:
                        best_sc = sc
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            va_pos = np.array([orig_to_cv[int(oi)] for oi in va_rows], dtype=np.int64)
            oof_pred[va_pos] = predict_gnn_endpoint(model, bank.graphs, va_cidx, target_ep, y_scalers[target_ep], device)
            ho_folds.append(predict_gnn_endpoint(model, bank.graphs, ho_cidx, target_ep, y_scalers[target_ep], device))

        ho_pred = np.mean(np.stack(ho_folds, axis=0), axis=0)
        if meta_t["task_type"] == "regression":
            cv_m = regression_metrics(d_t["y_value"][cv_rows], oof_pred, d_t["y_lower"][cv_rows], d_t["y_upper"][cv_rows], d_t["censor_str"][cv_rows])
            ho_m = regression_metrics(d_t["y_value"][ho_rows], ho_pred, d_t["y_lower"][ho_rows], d_t["y_upper"][ho_rows], d_t["censor_str"][ho_rows])
            msg = f"CV MAE={cv_m['mae']:.4f} Sp={cv_m['spearman']:.4f} | HO MAE={ho_m['mae']:.4f} Sp={ho_m['spearman']:.4f} R2={ho_m['r2']:.4f}"
        else:
            cv_m = classification_metrics(d_t["y_value"][cv_rows], oof_pred)
            ho_m = classification_metrics(d_t["y_value"][ho_rows], ho_pred)
            msg = f"CV ROC={cv_m['roc_auc']:.4f} PR={cv_m['pr_auc']:.4f} | HO ROC={ho_m['roc_auc']:.4f} PR={ho_m['pr_auc']:.4f} MCC={ho_m['mcc']:.4f}"

        ep_dir = OUT_GNN_DIR / target_ep
        ep_dir.mkdir(parents=True, exist_ok=True)
        with (ep_dir / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["record_id", "compound_key", "fold_id", "y_value", "censor_type", "pred_gnn_smiles"])
            for loc_i, oi in enumerate(cv_rows):
                w.writerow([d_t["record_ids"][oi], d_t["compound_keys"][oi], int(d_t["fold_id"][oi]), f"{d_t['y_value'][oi]:.6f}", d_t["censor_str"][oi], f"{oof_pred[loc_i]:.6f}"])
        with (ep_dir / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["record_id", "compound_key", "y_value", "censor_type", "pred_gnn_smiles"])
            for loc_i, oi in enumerate(ho_rows):
                w.writerow([d_t["record_ids"][oi], d_t["compound_keys"][oi], f"{d_t['y_value'][oi]:.6f}", d_t["censor_str"][oi], f"{ho_pred[loc_i]:.6f}"])

        dt = round(time.time() - t0, 1)
        ep_summaries[target_ep] = {
            "endpoint": target_ep,
            "task_type": meta_t["task_type"],
            "n_cv_smiles": n_cv,
            "n_ho_smiles": n_ho,
            "cv_metrics": cv_m,
            "holdout_metrics": ho_m,
            "fold_audits": fold_audits,
            "total_leaked_clusters": 0,
            "elapsed_sec": dt,
        }
        print(f"  [GNN {ep_i}/8] {target_ep:<26} | {msg} ({dt}s)", flush=True)

    return {
        "endpoints": ep_summaries,
        "total_cv_folds_verified": 40,
        "total_leaked_clusters": total_leaked_cl,
        "total_leaked_compounds": total_leaked_cidx,
    }


def train_unified_5fold_deployment_checkpoint(bank: GraphBank8, device: torch.device, ckpt_path: Path) -> Dict[str, Any]:
    """Train ONE compact 5-fold unified 8-head PurePyTorchEGATNet ensemble (< 25 MB fp16!).

    Strictly excludes ALL Holdout clusters across all 8 endpoints AND ALL Case17 clusters
    so ZERO holdout or Case17 compounds ever enter training!
    """
    t0 = time.time()
    forbidden_global = bank.all_holdout_clusters | bank.all_case17_clusters
    cv_clusters: Set[str] = set()
    for ep in GNN_8_ENDPOINTS:
        d = bank.endpoints[ep]
        for i in d["cv_idx"]:
            if d["case17_excluded"][i] == 0 and d["cluster_ids"][i] not in forbidden_global:
                cv_clusters.add(d["cluster_ids"][i])
    assert len(cv_clusters & forbidden_global) == 0

    sorted_cl = sorted(cv_clusters)
    cl_to_fold = {c: i % 5 for i, c in enumerate(sorted_cl)}

    fold_sds_fp16 = []
    y_scalers_per_fold = []
    fold_ho_preds: Dict[str, List[np.ndarray]] = {ep: [] for ep in GNN_8_ENDPOINTS}

    for f in range(5):
        set_seed(2026 + f * 37)
        val_cl = {c for c, fk in cl_to_fold.items() if fk == f}
        forb_f = forbidden_global | val_cl

        task_rows: Dict[str, np.ndarray] = {}
        y_scalers: Dict[str, Tuple[float, float]] = {}
        task_y: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = {}
        tr_cl_check: Set[str] = set()

        for ep in GNN_8_ENDPOINTS:
            d = bank.endpoints[ep]
            keep = [
                i for i in d["cv_idx"]
                if d["case17_excluded"][i] == 0 and d["cluster_ids"][i] not in forb_f
            ]
            r_idx = np.array(keep, dtype=np.int64)
            task_rows[ep] = r_idx
            tr_cl_check.update(d["cluster_ids"][r_idx].tolist())

            yv = d["y_value"][r_idx].copy()
            yl = d["y_lower"][r_idx].copy()
            yu = d["y_upper"][r_idx].copy()
            ct = d["censor_int"][r_idx].copy()
            if GNN_8_TASK_META[ep]["task_type"] == "regression":
                ex = ct == 0
                ref = yv[ex] if ex.sum() >= 3 else yv
                ym, ys = float(np.mean(ref)), float(np.std(ref))
                if not np.isfinite(ys) or ys < 1e-4:
                    ys = 1.0
                y_scalers[ep] = (ym, ys)
                yv, yl, yu = (yv - ym) / ys, (yl - ym) / ys, (yu - ym) / ys
            else:
                y_scalers[ep] = (0.0, 1.0)
            task_y[ep] = (yv.astype(np.float32), yl.astype(np.float32), yu.astype(np.float32), ct, bool((ct != 0).any()))

        assert len(tr_cl_check & forb_f) == 0, "Unified GNN cluster leakage!"

        model = PurePyTorchEGATNet(endpoints=GNN_8_ENDPOINTS, **MODEL_CFG).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2, fused=True)
        epochs = 30
        steps_per_ep = 65
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=3e-5)

        weights = np.array([len(task_rows[e]) ** 0.35 for e in GNN_8_ENDPOINTS], dtype=np.float64)
        probs = weights / weights.sum()
        rng = np.random.default_rng(2026 + f)

        model.train()
        for _ in range(epochs):
            for _ in range(steps_per_ep):
                chosen = rng.choice(GNN_8_ENDPOINTS, size=4, replace=False, p=probs)
                g_list, slices = [], []
                cur = 0
                for ep_k in chosen:
                    ar = task_rows[ep_k]
                    ba = min(96, len(ar))
                    loc = rng.integers(0, len(ar), size=ba)
                    dk = bank.endpoints[ep_k]
                    g_list.extend(bank.graphs[int(dk["compound_idx"][ar[i]])] for i in loc)
                    slices.append((ep_k, cur, cur + ba, loc))
                    cur += ba
                batch = collate_graphs(g_list, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h_atom = model.encode_atoms(batch)
                    bidx, gctx, ng = batch["batch_idx"], batch["global_ctx"], int(batch["num_graphs"])
                    outs = []
                    for ep_k, s0, e0, loc in slices:
                        pred_all, _, _ = model.heads[ep_k](h_atom, bidx, gctx, ng)
                        outs.append((ep_k, pred_all[s0:e0], loc))
                losses = []
                for ep_k, pred_k, loc in outs:
                    yv, yl, yu, ct, has_c = task_y[ep_k]
                    yt = torch.from_numpy(yv[loc]).to(device)
                    if GNN_8_TASK_META[ep_k]["task_type"] == "regression":
                        losses.append(
                            tobit_huber_loss(
                                pred_k, yt,
                                torch.from_numpy(yl[loc]).to(device),
                                torch.from_numpy(yu[loc]).to(device),
                                torch.from_numpy(ct[loc]).to(device),
                                has_censor=has_c,
                            )
                        )
                    else:
                        losses.append(cls_bce_loss(pred_k, yt))
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                opt.step()
                sched.step()

        # Evaluate on untouched 20% holdout for all 8 endpoints
        for ep in GNN_8_ENDPOINTS:
            d = bank.endpoints[ep]
            p_ho = predict_gnn_endpoint(model, bank.graphs, d["compound_idx"][d["ho_idx"]], ep, y_scalers[ep], device)
            fold_ho_preds[ep].append(p_ho)

        sd_fp16 = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
        fold_sds_fp16.append(sd_fp16)
        y_scalers_per_fold.append(y_scalers)
        print(f"  [UNIFIED GNN] fold {f}/5 trained (0 leaked clusters)", flush=True)

    n_params_single = sum(p.numel() for p in PurePyTorchEGATNet(endpoints=GNN_8_ENDPOINTS, **MODEL_CFG).parameters())
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "schema": "peptide_omnipanel_gnn_atom_attr_v1",
        "endpoints": GNN_8_ENDPOINTS,
        "model_config": MODEL_CFG,
        "n_folds": 5,
        "n_params_per_fold": n_params_single,
        "y_scalers_per_fold": y_scalers_per_fold,
        "fold_state_dicts_fp16": fold_sds_fp16,
    }
    torch.save(ckpt, ckpt_path)
    size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
    assert size_mb < 25.0, f"Checkpoint too large: {size_mb} MB (>= 25 MB)!"

    unified_ho_metrics = {}
    for ep in GNN_8_ENDPOINTS:
        d = bank.endpoints[ep]
        ho = d["ho_idx"]
        p_ho = np.mean(np.stack(fold_ho_preds[ep], axis=0), axis=0)
        if GNN_8_TASK_META[ep]["task_type"] == "regression":
            unified_ho_metrics[ep] = regression_metrics(d["y_value"][ho], p_ho, d["y_lower"][ho], d["y_upper"][ho], d["censor_str"][ho])
        else:
            unified_ho_metrics[ep] = classification_metrics(d["y_value"][ho], p_ho)

    print(f"[UNIFIED GNN CKPT SAVED] {ckpt_path} ({size_mb} MB, {n_params_single:,} params/fold, {time.time() - t0:.1f}s)", flush=True)
    return {
        "checkpoint_file": str(ckpt_path),
        "checkpoint_size_mb": size_mb,
        "n_params_per_fold": n_params_single,
        "n_folds": 5,
        "n_leaked_clusters": 0,
        "unified_holdout_metrics": unified_ho_metrics,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def run_unit_tests_on_4_peptides(ckpt_path: Path) -> Dict[str, Any]:
    """Battle-test `gnn_atom_attribution.py` on Semaglutide, Cyclosporine A, Micafungin, and Magainin-2!"""
    print("\n=== Unit-Testing gnn_atom_attribution.py on CPU ===", flush=True)

    # 1. Verify ZERO forbidden external imports in gnn_atom_attribution.py via AST
    src_path = GPU_SOTA_ROOT / "gnn_atom_attribution.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported_roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imported_roots.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    allowed = {"__future__", "math", "time", "pathlib", "typing", "numpy", "torch", "rdkit"}
    unexpected = imported_roots - allowed
    assert not unexpected, f"Unexpected non-core imports in gnn_atom_attribution.py: {unexpected}"
    assert "torch_geometric" not in imported_roots and "dgl" not in imported_roots
    print(f"  [IMPORT AUDIT PASS] Imports strictly limited to stdlib + {sorted(imported_roots & {'numpy', 'torch', 'rdkit'})}", flush=True)

    # Pull canonical SMILES for Semaglutide and Cyclosporine A from case17_queries.tsv
    c17_smi: Dict[str, str] = {}
    with (CORE24_DATA_DIR / "case17_queries.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c17_smi[r["drug_name"]] = r["canonical_smiles"]

    # Verified canonical SMILES for Micafungin (FDA echinocandin antifungal cyclic lipohexapeptide)
    micafungin_smi = (
        "CCCCCOc1ccc(-c2cc(-c3ccc(C(=O)N[C@H]4C[C@@H](O)[C@@H](O)NC(=O)[C@@H]5[C@@H](O)[C@@H](C)CN5"
        "C(=O)[C@H]([C@@H](C)O)NC(=O)[C@H]([C@H](O)[C@@H](O)c5ccc(O)c(OS(=O)(=O)O)c5)NC(=O)[C@@H]5C[C@@H](O)CN5"
        "C(=O)[C@H]([C@@H](C)O)NC4=O)cc3)no2)cc1"
    )
    # Magainin-2 unmodified antimicrobial peptide sequence (23-mer, tested via RDKit MolFromSequence!)
    magainin2_seq = "GIGKFLHSAKKFGKAFVGEIMNS"

    test_cases = [
        {"name": "Semaglutide", "input": c17_smi["Semaglutide"], "is_sequence": False, "svg_ep": "half_life_reg"},
        {"name": "Cyclosporine_A", "input": c17_smi["Cyclosporine_A"], "is_sequence": False, "svg_ep": "PAMPA"},
        {"name": "Micafungin", "input": micafungin_smi, "is_sequence": False, "svg_ep": "hemolysis_cls"},
        {"name": "Magainin_2", "input": magainin2_seq, "is_sequence": True, "svg_ep": "hemolysis_HC50"},
    ]

    engine = GNNAtomAttributionEngine(ckpt_path=ckpt_path, device="cpu")
    # Warm up PyTorch CPU thread pool once (1 tiny dipeptide)
    _ = engine.predict_and_attribute("CC(N)C(=O)NC(C)C(=O)O")

    svg_dir = OUT_GNN_DIR / "sample_svgs"
    svg_dir.mkdir(parents=True, exist_ok=True)
    case_reports = []

    for tc in test_cases:
        # Measure 3-run median CPU latency
        latencies = []
        res = None
        for _ in range(3):
            res = engine.predict_and_attribute(tc["input"], is_sequence=tc["is_sequence"])
            latencies.append(res["latency_ms"])
        med_ms = round(float(np.median(latencies)), 2)
        assert med_ms < 200.0, f"{tc['name']} CPU latency {med_ms} ms exceeded 200 ms budget!"

        n_atoms = res["n_atoms"]
        assert set(res["atom_weights"].keys()) == set(GNN_8_ENDPOINTS)
        for ep, w in res["atom_weights"].items():
            assert len(w) == n_atoms, f"{tc['name']} {ep} weight len mismatch"
            arr = np.asarray(w, dtype=np.float64)
            assert np.all(np.isfinite(arr)) and arr.min() >= 0.0 and arr.max() <= 1.0, f"Invalid weights for {tc['name']} {ep}"

        # Verify task differentiation (PAMPA vs hemolysis_cls weights should not be identical!)
        w_pampa = np.asarray(res["atom_weights"]["PAMPA"])
        w_hemo = np.asarray(res["atom_weights"]["hemolysis_cls"])
        task_corr = float(np.corrcoef(w_pampa, w_hemo)[0, 1]) if np.std(w_pampa) > 1e-6 and np.std(w_hemo) > 1e-6 else 0.0

        # Render SVG & verify valid well-formed XML/SVG
        mol = parse_peptide_input(tc["input"], is_sequence=tc["is_sequence"])
        ep_vis = tc["svg_ep"]
        pred_info = res["predictions"][ep_vis]
        pred_txt = f"Pred={pred_info['value']:.3f} {pred_info['unit']}"
        svg_str = render_atom_attribution_svg(
            mol, res["atom_weights"][ep_vis], endpoint_name=ep_vis, title=f"{tc['name']} — {ep_vis}", prediction_text=pred_txt
        )
        root_xml = ET.fromstring(svg_str)
        assert root_xml.tag.endswith("svg"), f"Invalid SVG root tag: {root_xml.tag}"
        svg_file = svg_dir / f"{tc['name']}_{ep_vis}.svg"
        svg_file.write_text(svg_str, encoding="utf-8")

        rec = {
            "name": tc["name"],
            "input_mode": "sequence" if tc["is_sequence"] else "smiles",
            "n_atoms": n_atoms,
            "cpu_latency_ms_median": med_ms,
            "cpu_latency_ms_runs": [round(x, 2) for x in latencies],
            "under_200ms": bool(med_ms < 200.0),
            "all_weights_finite_0_1": True,
            "pampa_vs_hemolysis_weight_corr": round(task_corr, 4),
            "svg_endpoint": ep_vis,
            "svg_valid_xml": True,
            "svg_bytes": len(svg_str.encode("utf-8")),
            "svg_file": str(svg_file),
            "predictions": res["predictions"],
        }
        case_reports.append(rec)
        print(
            f"  [PASS] {tc['name']:<16} ({rec['input_mode']:<8}, {n_atoms:3d} atoms) | "
            f"CPU={med_ms:5.1f} ms (<200ms) | SVG={rec['svg_bytes']/1024:.1f} KB (valid XML) | "
            f"PAMPA/Hemo corr={task_corr:.3f}",
            flush=True,
        )

    return {
        "import_audit_passed": True,
        "external_deps": sorted(imported_roots & {"numpy", "torch", "rdkit"}),
        "all_cases_passed": True,
        "cases": case_reports,
    }


def main() -> None:
    t_all = time.time()
    OUT_GNN_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA RTX 5080 required for training!"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")

    bank = GraphBank8()

    print("\n=== Step 1: Per-Endpoint Strict Zero-Leakage 5-Fold CV + Untouched Holdout (8 Flagship EPs) ===", flush=True)
    cv_ho_res = train_per_endpoint_zero_leak_gnn(bank, device)

    print("\n=== Step 2: Compact Unified 8-Head 5-Fold fp16 Deployment Checkpoint (< 25 MB) ===", flush=True)
    ckpt_path = OUT_GNN_DIR / "checkpoints" / "gnn_8ep_5fold_fp16.pt"
    unified_res = train_unified_5fold_deployment_checkpoint(bank, device, ckpt_path)

    print("\n=== Step 3: CPU Unit Tests & SVG Verification on Semaglutide, Cyclosporine A, Micafungin, Magainin-2 ===", flush=True)
    unit_res = run_unit_tests_on_4_peptides(ckpt_path)

    summary = {
        "schema": "peptide_omnipanel_gnn_atom_attr_summary_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "architecture": "Pure-PyTorch 4-Layer Edge-Conditioned Multi-Head GATv2 + Multi-Scale JK + Task WeightAndSum",
        "endpoints": GNN_8_ENDPOINTS,
        "model_config": MODEL_CFG,
        "total_elapsed_sec": round(time.time() - t_all, 1),
        "per_endpoint_zero_leak_benchmark": cv_ho_res,
        "unified_5fold_checkpoint": unified_res,
        "unit_tests_4peptides": unit_res,
    }
    sp = OUT_GNN_DIR / "gnn_atom_attr_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== UPGRADE 3 COMPLETE -> {sp} ({summary['total_elapsed_sec']}s) ===", flush=True)


if __name__ == "__main__":
    main()
