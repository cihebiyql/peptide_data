#!/usr/bin/env python3
"""Training & Evaluation Engine for Peptide OmniPanel V33 Deep Multi-Task & Attention Models.

Models:
  1. MultiTaskPeptideNet: Shared Multimodal Gated Trunk (Physchem + ncAA-RepDistill v2 +
     PeptideCLM-2 + ChemBERTa-MTR + ESM-C 600M 4-region Attention) + 24 Condition-Injected
     FiLM/Additive Heads + Censored Gaussian Tobit / Huber-NLL Loss + Stochastic Modality Dropout.
  2. HalfLifeAttentionNet: Dedicated Condition-Aware Cross-Attention + RBF-KAN Specialist for
     half_life_reg & half_life_cls co-trained with PK auxiliaries (PPB, CL_per_kg, Vd_per_kg,
     CL_absolute, LogD7.4, solubility_reg).

Strict Zero-Leakage Protocol:
  - Per-endpoint fold k masks out HoldoutClusters(E) | ValFoldClusters(E, k) across ALL 24 tasks
    with hard runtime assertion `assert len(train_clusters & forbidden_clusters) == 0`.
  - Leave-Case17-Out masks out ALL case17_excluded==1 rows AND ALL 19 Case17 clusters across ALL
    24 tasks with hard runtime assertion `assert 0 Case17 clusters in training`.
  - Single trained checkpoint serves BOTH `dual_modality` and `smiles_only` (seq_mask=0, zeroed seq).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deep_data import (
    ALL_ENDPOINTS,
    CASE17_CLINICAL_ROUTE,
    ENDPOINT_FAMILY,
    ENDPOINT_SPECS,
    EP_TO_ID,
    OUT_ROOT,
    ConformalIntervalCalibrator,
    DeepConditionEncoder,
    MultimodalGPUBank,
    ProbabilityCalibrator,
    TanimotoADIndex,
    classification_metrics,
    compute_ad_stratification,
    compute_halflife_mechanism_anchor,
    compute_oof_tanimoto,
    regression_metrics,
)
from deep_models import (
    HalfLifeAttentionNet,
    MultiTaskPeptideNet,
    censored_tobit_huber_loss,
    classification_bce_loss,
    extract_mu_sigma,
    verify_model_invariants,
)

HALFLIFE_SPECIALIST_ENDPOINTS = ["half_life_reg", "half_life_cls"]
HALFLIFE_PK_AUX = [
    "half_life_reg",
    "half_life_cls",
    "PPB",
    "CL_per_kg",
    "Vd_per_kg",
    "CL_absolute",
    "F",
    "Cmax",
    "LogD7.4",
    "solubility_reg",
    "PAMPA",
    "Caco2",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_gpu_telemetry() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "host": "local_windows",
            "gpu_index": int(parts[0]),
            "gpu_name": parts[1],
            "vram_used_mib": int(parts[2]),
            "vram_total_mib": int(parts[3]),
            "gpu_util_pct": int(parts[4]),
            "temp_c": int(parts[5]),
            "torch_version": torch.__version__,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
    except Exception as exc:
        return {"error": str(exc), "torch_version": torch.__version__}


def _compute_aux_sampling_probs(target_ep: str, active_aux: List[str], task_tensors: Dict[str, Any]) -> np.ndarray:
    """Temperature-balanced task sampling (n^0.35) with strong biophysical family affinity."""
    t_fam = ENDPOINT_FAMILY[target_ep]
    weights = []
    for a_ep in active_aux:
        n_a = len(task_tensors[a_ep]["cidx"])
        w = max(1.0, float(n_a)) ** 0.35
        a_fam = ENDPOINT_FAMILY[a_ep]
        if a_fam == t_fam:
            w *= 3.5
        elif (t_fam == "pk" and a_fam == "perm") or (t_fam == "perm" and a_fam == "pk"):
            w *= 1.8
        elif (t_fam == "tox" and a_fam == "mic") or (t_fam == "mic" and a_fam == "tox"):
            w *= 1.4
        weights.append(w)
    arr = np.asarray(weights, dtype=np.float64)
    return arr / arr.sum()


@torch.no_grad()
def predict_model_on_tensors(
    model: nn.Module,
    bank: MultimodalGPUBank,
    phys_stats: Dict[str, torch.Tensor],
    ep: str,
    cidx: torch.Tensor,
    cond: torch.Tensor,
    y_scaler: Tuple[float, float],
    eval_mode: str = "dual_modality",
    batch_size: int = 1024,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict on arbitrary compound tensor slice in dual_modality or smiles_only mode.

    Returns (pred_unscaled, sigma_unscaled_or_None).
    For smiles_only mode, rows with smiles_valid==0 are set to NaN (matching V33 contract).
    """
    model.eval()
    task_type = ENDPOINT_SPECS[ep]["task_type"]
    bounds = ENDPOINT_SPECS[ep].get("bounds")
    ym, ys = y_scaler
    n = len(cidx)
    preds = np.full(n, np.nan, dtype=np.float64)
    sigmas = np.full(n, np.nan, dtype=np.float64) if task_type == "regression" else None

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sub_cidx = cidx[s:e]
        sub_cond = cond[s:e]
        batch = bank.gather_inputs(sub_cidx, phys_stats, eval_mode=eval_mode)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw = model.forward_task(batch, ep, sub_cond, apply_modality_dropout=False)
        if task_type == "regression":
            mu_z, sig_z = extract_mu_sigma(raw)
            mu_val = mu_z.cpu().numpy().astype(np.float64) * ys + ym
            sig_val = sig_z.cpu().numpy().astype(np.float64) * ys
            if bounds is not None:
                mu_val = np.clip(mu_val, bounds[0], bounds[1])
            preds[s:e] = mu_val
            sigmas[s:e] = sig_val
        else:
            prob = torch.sigmoid(raw.squeeze(-1).float()).cpu().numpy().astype(np.float64)
            preds[s:e] = np.clip(prob, 1e-6, 1.0 - 1e-6)

    if eval_mode == "smiles_only":
        smi_ok = bank.smiles_valid[cidx.cpu().numpy()]
        preds[~smi_ok] = np.nan
        if sigmas is not None:
            sigmas[~smi_ok] = np.nan
    return preds, sigmas


def _eval_val_score(
    ep: str,
    y_raw: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    censor_str: np.ndarray,
    p_dual: np.ndarray,
    p_smi: np.ndarray,
    smi_valid_frac: float,
) -> float:
    """Compute scalar validation selection score (HIGHER IS BETTER) strictly on CV fold."""
    task_type = ENDPOINT_SPECS[ep]["task_type"]
    if task_type == "regression":
        m_d = regression_metrics(y_raw, p_dual, y_lo, y_hi, censor_str)
        mae_d = m_d["mae"] if m_d["mae"] is not None else 10.0
        viol_d = m_d["censored_violation_mae"] or 0.0
        sp_d = m_d["spearman"] if m_d["spearman"] is not None else 0.0
        # Combine exact MAE + censored hinge violation (plus small Spearman tie-breaker)
        score_d = -(mae_d + 0.35 * viol_d) + 0.05 * sp_d
        if smi_valid_frac >= 0.50 and np.isfinite(p_smi).sum() >= 5:
            m_s = regression_metrics(y_raw, p_smi, y_lo, y_hi, censor_str)
            mae_s = m_s["mae"] if m_s["mae"] is not None else mae_d
            viol_s = m_s["censored_violation_mae"] or 0.0
            sp_s = m_s["spearman"] if m_s["spearman"] is not None else sp_d
            score_s = -(mae_s + 0.35 * viol_s) + 0.05 * sp_s
            return 0.75 * score_d + 0.25 * score_s
        return score_d
    else:
        m_d = classification_metrics(y_raw, p_dual)
        roc_d = m_d["roc_auc"] if m_d["roc_auc"] is not None else 0.5
        br_d = m_d["brier"] if m_d["brier"] is not None else 0.25
        score_d = roc_d - 0.20 * br_d
        if smi_valid_frac >= 0.50 and np.isfinite(p_smi).sum() >= 5:
            m_s = classification_metrics(y_raw, p_smi)
            if m_s["roc_auc"] is not None:
                score_s = m_s["roc_auc"] - 0.20 * (m_s["brier"] or 0.25)
                return 0.75 * score_d + 0.25 * score_s
        return score_d


def _fast_val_score(
    ep: str,
    y_raw: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    censor_int: np.ndarray,
    p_dual: np.ndarray,
    p_smi: np.ndarray,
    smi_valid_frac: float,
) -> float:
    """Fast O(N) validation selection score (no O(N^2) C-index inside epoch loop!). HIGHER IS BETTER."""
    task_type = ENDPOINT_SPECS[ep]["task_type"]
    if task_type == "regression":
        def _reg_sc(p: np.ndarray) -> float:
            ok = np.isfinite(p) & np.isfinite(y_raw)
            ex = ok & (censor_int == 0)
            mae = float(np.mean(np.abs(y_raw[ex] - p[ex]))) if ex.any() else 10.0
            r_m = ok & (censor_int == 1)
            l_m = ok & (censor_int == 2)
            i_m = ok & (censor_int == 3)
            viols = []
            if r_m.any():
                viols.append(np.maximum(0.0, y_lo[r_m] - p[r_m]))
            if l_m.any():
                viols.append(np.maximum(0.0, p[l_m] - y_hi[l_m]))
            if i_m.any():
                viols.append(np.maximum(0.0, np.maximum(y_lo[i_m] - p[i_m], p[i_m] - y_hi[i_m])))
            viol = float(np.mean(np.concatenate(viols))) if viols else 0.0
            return -(mae + 0.35 * viol)

        sc_d = _reg_sc(p_dual)
        if smi_valid_frac >= 0.50 and np.isfinite(p_smi).sum() >= 5:
            return 0.75 * sc_d + 0.25 * _reg_sc(p_smi)
        return sc_d
    else:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        def _cls_sc(p: np.ndarray) -> Optional[float]:
            ok = np.isfinite(p)
            if ok.sum() < 4:
                return None
            yt = y_raw[ok].astype(np.int32)
            if len(np.unique(yt)) < 2:
                return None
            pp = np.clip(p[ok], 1e-6, 1.0 - 1e-6)
            return float(roc_auc_score(yt, pp) - 0.20 * brier_score_loss(yt, pp))

        sc_d = _cls_sc(p_dual)
        if sc_d is None:
            return -1.0
        if smi_valid_frac >= 0.50:
            sc_s = _cls_sc(p_smi)
            if sc_s is not None:
                return 0.75 * sc_d + 0.25 * sc_s
        return sc_d


def train_single_model_fold(
    bank: MultimodalGPUBank,
    split: Dict[str, Any],
    arch: str,
    seed: int,
    override_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train ONE MultiTaskPeptideNet or HalfLifeAttentionNet model on a clean zero-leakage split."""
    set_seed(seed)
    target_ep = split["target_ep"]
    active_eps = split["active_eps"]
    aux_eps = [ep for ep in active_eps if ep != target_ep]
    task_tensors = split["task_tensors"]
    phys_stats = split["phys_stats"]
    cond_dims = split["cond_dims"]
    y_scalers = split["y_scalers"]
    val = split["val"]

    n_tr_t = len(task_tensors[target_ep]["cidx"])
    size_tier = ENDPOINT_SPECS[target_ep]["size_tier"]

    # Sample-size-adapted hyperparameters tuned for RTX 5080 bf16 (large batches, clean gradients, fast epochs!)
    if arch == "halflife_specialist":
        cfg = {
            "epochs": 40,
            "lr": 1.1e-3,
            "weight_decay": 2.5e-2,
            "dropout": 0.28,
            "b_target": 192,
            "k_aux": 2,
            "b_aux": 160,
            "aux_weight_init": 0.55,
            "aux_weight_final": 0.15,
            "steps_per_epoch": max(12, math.ceil(n_tr_t / 192)),
            "patience": 10,
            "val_every": 2,
        }
    elif size_tier == "small":
        cfg = {
            "epochs": 32,
            "lr": 9e-4,
            "weight_decay": 4.0e-2,
            "dropout": 0.30,
            "b_target": min(96, max(16, n_tr_t)),
            "k_aux": 3,
            "b_aux": 128,
            "aux_weight_init": 1.05,
            "aux_weight_final": 0.35,
            "steps_per_epoch": 10,
            "patience": 8,
            "val_every": 2,
        }
    elif size_tier == "medium":
        cfg = {
            "epochs": 36,
            "lr": 1.1e-3,
            "weight_decay": 2.5e-2,
            "dropout": 0.26,
            "b_target": 256,
            "k_aux": 2,
            "b_aux": 160,
            "aux_weight_init": 0.55,
            "aux_weight_final": 0.15,
            "steps_per_epoch": max(10, math.ceil(n_tr_t / 256)),
            "patience": 8,
            "val_every": 2,
        }
    else:  # large
        cfg = {
            "epochs": 32,
            "lr": 1.3e-3,
            "weight_decay": 1.8e-2,
            "dropout": 0.22,
            "b_target": 512,
            "k_aux": 2,
            "b_aux": 192,
            "aux_weight_init": 0.35,
            "aux_weight_final": 0.08,
            "steps_per_epoch": max(14, math.ceil(n_tr_t / 512)),
            "patience": 8,
            "val_every": 2,
        }
    if override_cfg:
        cfg.update(override_cfg)

    if arch == "multitask":
        model: nn.Module = MultiTaskPeptideNet(
            cond_dims=cond_dims,
            active_eps=active_eps,
            d_mod=256,
            d_trunk=512,
            d_bottleneck=384,
            dropout=cfg["dropout"],
            seq_drop_prob=0.30,
            smi_drop_prob=0.10,
            lm_drop_prob=0.10,
        ).to(bank.device)
    elif arch == "halflife_specialist":
        model = HalfLifeAttentionNet(
            cond_dims=cond_dims,
            active_eps=active_eps,
            d_model=256,
            dropout=cfg["dropout"],
            seq_drop_prob=0.30,
            smi_drop_prob=0.12,
        ).to(bank.device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Weight decay only on >=2D weights + FUSED CUDA AdamW (huge Windows WDDM speedup!)
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "norm" not in name.lower() and "emb" not in name.lower():
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg["lr"],
        fused=True,
    )

    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    warmup_steps = max(1, int(0.08 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    aux_probs = _compute_aux_sampling_probs(target_ep, aux_eps, task_tensors) if aux_eps else None
    rng = np.random.default_rng(seed)

    t_ten = task_tensors[target_ep]
    t_type = ENDPOINT_SPECS[target_ep]["task_type"]
    val_smi_frac = float(bank.smiles_valid[val["cidx"].cpu().numpy()].mean())

    best_score = -1e9
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    global_step = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        # Anneal auxiliary weight cleanly across second half of training (45% -> 90%)
        frac = epoch / max(1, cfg["epochs"] - 1)
        if frac < 0.45:
            aux_w = cfg["aux_weight_init"]
        elif frac < 0.90:
            p_ann = (frac - 0.45) / 0.45
            aux_w = cfg["aux_weight_init"] * (1.0 - p_ann) + cfg["aux_weight_final"] * p_ann
        else:
            aux_w = cfg["aux_weight_final"]

        # Prepare epoch permutation for target task so every target sample is visited evenly
        t_perm = torch.randperm(n_tr_t, device=bank.device)
        t_ptr = 0

        for _ in range(cfg["steps_per_epoch"]):
            # Target slice
            bt = cfg["b_target"]
            if t_ptr + bt > n_tr_t:
                t_perm = torch.randperm(n_tr_t, device=bank.device)
                t_ptr = 0
            if bt >= n_tr_t:
                idx_t = t_perm
            else:
                idx_t = t_perm[t_ptr : t_ptr + bt]
                t_ptr += bt

            cidx_parts = [t_ten["cidx"][idx_t]]
            tid_parts = [torch.full((len(idx_t),), EP_TO_ID[target_ep], dtype=torch.long, device=bank.device)]
            slice_meta = [(target_ep, 0, len(idx_t), idx_t)]
            cursor = len(idx_t)

            # Sample K_aux balanced auxiliary tasks (single concatenated GPU forward pass!)
            if aux_eps and aux_probs is not None and cfg["k_aux"] > 0:
                k_draw = min(cfg["k_aux"], len(aux_eps))
                chosen_idx = rng.choice(len(aux_eps), size=k_draw, replace=False, p=aux_probs)
                for ci in chosen_idx:
                    a_ep = aux_eps[int(ci)]
                    a_ten = task_tensors[a_ep]
                    n_a = len(a_ten["cidx"])
                    ba = min(cfg["b_aux"], n_a)
                    idx_a = torch.randint(0, n_a, (ba,), device=bank.device)
                    cidx_parts.append(a_ten["cidx"][idx_a])
                    tid_parts.append(torch.full((ba,), EP_TO_ID[a_ep], dtype=torch.long, device=bank.device))
                    slice_meta.append((a_ep, cursor, cursor + ba, idx_a))
                    cursor += ba

            all_cidx = torch.cat(cidx_parts, dim=0)
            all_tids = torch.cat(tid_parts, dim=0)
            batch = bank.gather_inputs(all_cidx, phys_stats, eval_mode="dual_modality")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if arch == "multitask":
                    h_all = model.encode_trunk(batch, all_tids, apply_modality_dropout=True)
                else:
                    h_all = model.encode_molecule(batch, apply_modality_dropout=True)

                # Target head prediction
                ep0, s0, e0, idx0 = slice_meta[0]
                out_t = model.get_head(ep0)(h_all[s0:e0], t_ten["cond"][idx0])

                # Aux head predictions
                aux_outs = []
                for a_ep, sa, ea, idxa in slice_meta[1:]:
                    a_ten = task_tensors[a_ep]
                    aux_outs.append((a_ep, model.get_head(a_ep)(h_all[sa:ea], a_ten["cond"][idxa]), idxa))

            # Compute exact losses in float32 outside bf16 autocast for numerical precision
            if t_type == "regression":
                loss_t = censored_tobit_huber_loss(
                    out_t, t_ten["y"][idx0], t_ten["y_lo"][idx0], t_ten["y_hi"][idx0], t_ten["censor"][idx0],
                    has_censor=t_ten["has_censor"],
                )
            else:
                loss_t = classification_bce_loss(out_t, t_ten["y"][idx0], label_smoothing=0.02)

            if aux_outs:
                aux_losses = []
                for a_ep, out_a, idxa in aux_outs:
                    a_ten = task_tensors[a_ep]
                    if ENDPOINT_SPECS[a_ep]["task_type"] == "regression":
                        la = censored_tobit_huber_loss(
                            out_a, a_ten["y"][idxa], a_ten["y_lo"][idxa], a_ten["y_hi"][idxa], a_ten["censor"][idxa],
                            has_censor=a_ten["has_censor"],
                        )
                    else:
                        la = classification_bce_loss(out_a, a_ten["y"][idxa], label_smoothing=0.02)
                    aux_losses.append(la)
                loss = loss_t + aux_w * (sum(aux_losses) / len(aux_losses))
            else:
                loss = loss_t

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # Fast validation check every val_every epochs after brief warmup
        val_every = cfg.get("val_every", 2)
        if (epoch >= 4 and (epoch % val_every == 0)) or epoch == cfg["epochs"] - 1:
            p_va_d, _ = predict_model_on_tensors(
                model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "dual_modality"
            )
            p_va_s, _ = predict_model_on_tensors(
                model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "smiles_only"
            )
            sc = _fast_val_score(
                target_ep, val["y_raw"], val["y_lo_raw"], val["y_hi_raw"], val["censor_int"],
                p_va_d, p_va_s, val_smi_frac
            )
            if sc > best_score + 1e-5:
                best_score = sc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += val_every
                if no_improve >= cfg["patience"] and epoch >= int(0.60 * cfg["epochs"]):
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final OOF & Holdout predictions from restored best checkpoint
    p_va_d, sig_va_d = predict_model_on_tensors(
        model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "dual_modality"
    )
    p_va_s, sig_va_s = predict_model_on_tensors(
        model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "smiles_only"
    )

    p_ho_d = sig_ho_d = p_ho_s = sig_ho_s = None
    if split["holdout"] is not None:
        ho = split["holdout"]
        p_ho_d, sig_ho_d = predict_model_on_tensors(
            model, bank, phys_stats, target_ep, ho["cidx"], ho["cond"], y_scalers[target_ep], "dual_modality"
        )
        p_ho_s, sig_ho_s = predict_model_on_tensors(
            model, bank, phys_stats, target_ep, ho["cidx"], ho["cond"], y_scalers[target_ep], "smiles_only"
        )

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "p_va_dual": p_va_d,
        "sig_va_dual": sig_va_d,
        "p_va_smi": p_va_s,
        "sig_va_smi": sig_va_s,
        "p_ho_dual": p_ho_d,
        "sig_ho_dual": sig_ho_d,
        "p_ho_smi": p_ho_s,
        "sig_ho_smi": sig_ho_s,
    }


def _pack_checkpoint(
    model: nn.Module,
    arch: str,
    target_ep: str,
    split: Dict[str, Any],
    best_epoch: int,
) -> Dict[str, Any]:
    """Pack compact fp16 inference-ready checkpoint + preprocessing metadata."""
    fp16_sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        fp16_sd[k] = t.half() if t.is_floating_point() else t
    return {
        "schema": "peptide_omnipanel_v33_gpu_deep_ckpt_v1",
        "arch": arch,
        "target_ep": target_ep,
        "active_eps": split["active_eps"],
        "cond_dims": split["cond_dims"],
        "cond_encoders": {ep: enc.state_dict() for ep, enc in split["cond_encoders"].items()},
        "y_scalers": split["y_scalers"],
        "phys_stats": {k: v.detach().cpu().numpy().tolist() for k, v in split["phys_stats"].items()},
        "best_epoch": best_epoch,
        "state_dict_fp16": fp16_sd,
    }


def run_5fold_cv_and_holdout(
    bank: MultimodalGPUBank,
    ep: str,
    arch: str,
    seeds: List[int],
    aux_endpoints: Optional[List[str]] = None,
    add_cond_pairs_for: Optional[set] = None,
    save_ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run strictly zero-leakage 5-fold CV (multi-seed bagged) + untouched 20% Holdout evaluation."""
    t0 = time.time()
    ep_data = bank.endpoints[ep]
    spec = ep_data["spec"]
    task_type = spec["task_type"]
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)

    orig_to_cv_pos = {int(orig_i): pos for pos, orig_i in enumerate(cv_rows)}

    oof_dual = np.full(n_cv, np.nan, dtype=np.float64)
    oof_smi = np.full(n_cv, np.nan, dtype=np.float64)
    oof_sig_dual = np.full(n_cv, np.nan, dtype=np.float64) if task_type == "regression" else None
    oof_sig_smi = np.full(n_cv, np.nan, dtype=np.float64) if task_type == "regression" else None
    oof_seed_sd = np.full(n_cv, np.nan, dtype=np.float64)

    ho_dual_folds = []
    ho_smi_folds = []
    ho_sig_dual_folds = []
    ho_sig_smi_folds = []

    fold_audits = []
    best_epochs = []

    for f in range(5):
        split = bank.build_leak_free_multitask_split(
            target_ep=ep,
            fold=f,
            aux_endpoints=aux_endpoints,
            add_cond_pairs_for=add_cond_pairs_for,
            leave_case17_out=False,
        )
        fold_audits.append(split["audit"])

        va_pos = np.array([orig_to_cv_pos[int(i)] for i in split["val"]["orig_rows"]], dtype=np.int64)
        va_d_seeds, va_s_seeds = [], []
        va_sig_d_seeds, va_sig_s_seeds = [], []
        ho_d_seeds, ho_s_seeds = [], []
        ho_sig_d_seeds, ho_sig_s_seeds = [], []
        best_seed_run = None

        for s_idx, seed in enumerate(seeds):
            res = train_single_model_fold(bank, split, arch=arch, seed=seed + f * 17)
            best_epochs.append(res["best_epoch"])
            va_d_seeds.append(res["p_va_dual"])
            va_s_seeds.append(res["p_va_smi"])
            ho_d_seeds.append(res["p_ho_dual"])
            ho_s_seeds.append(res["p_ho_smi"])
            if task_type == "regression":
                va_sig_d_seeds.append(res["sig_va_dual"])
                va_sig_s_seeds.append(res["sig_va_smi"])
                ho_sig_d_seeds.append(res["sig_ho_dual"])
                ho_sig_s_seeds.append(res["sig_ho_smi"])
            if best_seed_run is None or res["best_val_score"] > best_seed_run["best_val_score"]:
                best_seed_run = res

        # Save best-seed fold checkpoint if requested
        if save_ckpt_dir is not None and best_seed_run is not None:
            save_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = _pack_checkpoint(best_seed_run["model"], arch, ep, split, best_seed_run["best_epoch"])
            torch.save(ckpt, save_ckpt_dir / f"{ep}_{arch}_fold{f}.pt")

        # Average seeds within fold
        oof_dual[va_pos] = np.mean(va_d_seeds, axis=0)
        oof_smi[va_pos] = np.nanmean(va_s_seeds, axis=0) if np.isfinite(va_s_seeds[0]).any() else np.nan
        oof_seed_sd[va_pos] = np.std(va_d_seeds, axis=0) if len(va_d_seeds) > 1 else 0.0
        if task_type == "regression":
            oof_sig_dual[va_pos] = np.mean(va_sig_d_seeds, axis=0)
            oof_sig_smi[va_pos] = np.nanmean(va_sig_s_seeds, axis=0) if np.isfinite(va_sig_s_seeds[0]).any() else np.nan

        ho_dual_folds.append(np.mean(ho_d_seeds, axis=0))
        ho_smi_folds.append(np.nanmean(ho_s_seeds, axis=0) if np.isfinite(ho_s_seeds[0]).any() else np.full(n_ho, np.nan))
        if task_type == "regression":
            ho_sig_dual_folds.append(np.mean(ho_sig_d_seeds, axis=0))
            ho_sig_smi_folds.append(
                np.nanmean(ho_sig_s_seeds, axis=0) if np.isfinite(ho_sig_s_seeds[0]).any() else np.full(n_ho, np.nan)
            )

    ho_dual_arr = np.stack(ho_dual_folds, axis=0)
    ho_smi_arr = np.stack(ho_smi_folds, axis=0)
    ho_dual = np.mean(ho_dual_arr, axis=0)
    ho_smi = np.nanmean(ho_smi_arr, axis=0)
    ho_sd_dual = np.std(ho_dual_arr, axis=0)
    ho_sd_smi = np.nanstd(ho_smi_arr, axis=0)

    bounds = spec.get("bounds")
    if bounds is not None:
        oof_dual = np.clip(oof_dual, bounds[0], bounds[1])
        ho_dual = np.clip(ho_dual, bounds[0], bounds[1])
        ok_s = np.isfinite(oof_smi)
        oof_smi[ok_s] = np.clip(oof_smi[ok_s], bounds[0], bounds[1])
        ok_hs = np.isfinite(ho_smi)
        ho_smi[ok_hs] = np.clip(ho_smi[ok_hs], bounds[0], bounds[1])

    ho_sig_dual = np.mean(ho_sig_dual_folds, axis=0) if task_type == "regression" else None
    ho_sig_smi = np.nanmean(ho_sig_smi_folds, axis=0) if task_type == "regression" else None

    # Metrics
    y_cv, yl_cv, yu_cv, ct_cv = (
        ep_data["y_value"][cv_rows],
        ep_data["y_lower"][cv_rows],
        ep_data["y_upper"][cv_rows],
        ep_data["censor_type"][cv_rows],
    )
    y_ho, yl_ho, yu_ho, ct_ho = (
        ep_data["y_value"][ho_rows],
        ep_data["y_lower"][ho_rows],
        ep_data["y_upper"][ho_rows],
        ep_data["censor_type"][ho_rows],
    )
    if task_type == "regression":
        cv_m_dual = regression_metrics(y_cv, oof_dual, yl_cv, yu_cv, ct_cv)
        cv_m_smi = regression_metrics(y_cv, oof_smi, yl_cv, yu_cv, ct_cv)
        ho_m_dual = regression_metrics(y_ho, ho_dual, yl_ho, yu_ho, ct_ho)
        ho_m_smi = regression_metrics(y_ho, ho_smi, yl_ho, yu_ho, ct_ho)
    else:
        cv_m_dual = classification_metrics(y_cv, oof_dual)
        cv_m_smi = classification_metrics(y_cv, oof_smi)
        ho_m_dual = classification_metrics(y_ho, ho_dual)
        ho_m_smi = classification_metrics(y_ho, ho_smi)

    return {
        "endpoint": ep,
        "arch": arch,
        "seeds": seeds,
        "oof_dual": oof_dual,
        "oof_smi": oof_smi,
        "oof_sig_dual": oof_sig_dual,
        "oof_sig_smi": oof_sig_smi,
        "oof_seed_sd": oof_seed_sd,
        "ho_dual": ho_dual,
        "ho_smi": ho_smi,
        "ho_sd_dual": ho_sd_dual,
        "ho_sd_smi": ho_sd_smi,
        "ho_sig_dual": ho_sig_dual,
        "ho_sig_smi": ho_sig_smi,
        "cv_metrics_dual": cv_m_dual,
        "cv_metrics_smi": cv_m_smi,
        "ho_metrics_dual": ho_m_dual,
        "ho_metrics_smi": ho_m_smi,
        "fold_audits": fold_audits,
        "mean_best_epoch": round(float(np.mean(best_epochs)), 1),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def run_leave_case17_out_ensemble(
    bank: MultimodalGPUBank,
    ep: str,
    arch: str,
    seeds: List[int],
    aux_endpoints: Optional[List[str]] = None,
    add_cond_pairs_for: Optional[set] = None,
    save_ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train strict Leave-Case17-Out 5-fold ensemble (0 Case17 rows & 0 Case17 clusters across ALL 24 tasks)."""
    t0 = time.time()
    spec = ENDPOINT_SPECS[ep]
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    c17_idx = torch.tensor([q["compound_idx"] for q in bank.case17_queries], dtype=torch.long, device=bank.device)
    canon_cond = spec["canonical_conditions"]
    alt_conds = spec.get("alt_case17_conditions", {})

    p_d_list, p_s_list, sig_d_list = [], [], []
    alt_preds: Dict[str, Dict[str, List[np.ndarray]]] = {k: {"dual": [], "smi": []} for k in alt_conds}
    audits = []
    tr_cidx_all = []

    for f in range(5):
        split = bank.build_leak_free_multitask_split(
            target_ep=ep,
            fold=f,
            aux_endpoints=aux_endpoints,
            add_cond_pairs_for=add_cond_pairs_for,
            leave_case17_out=True,
            case17_val_fold=f,
        )
        audits.append(split["audit"])
        tr_cidx_all.append(split["task_tensors"][ep]["cidx"].cpu().numpy())

        enc = split["cond_encoders"][ep]
        c_canon = torch.from_numpy(enc.transform_constant(17, canon_cond)).to(bank.device)
        c_alts = {k: torch.from_numpy(enc.transform_constant(17, ac)).to(bank.device) for k, ac in alt_conds.items()}

        # Use first 2 seeds max for Case17 5-fold (10 models total -> super clean & fast)
        c17_seeds = seeds[: min(2, len(seeds))]
        fd_s, fs_s, fsig_s = [], [], []
        alt_fd = {k: [] for k in alt_conds}
        alt_fs = {k: [] for k in alt_conds}
        best_res = None

        for seed in c17_seeds:
            res = train_single_model_fold(bank, split, arch=arch, seed=seed + 1000 + f * 31)
            m = res["model"]
            pd_i, sig_i = predict_model_on_tensors(
                m, bank, split["phys_stats"], ep, c17_idx, c_canon, split["y_scalers"][ep], "dual_modality"
            )
            ps_i, _ = predict_model_on_tensors(
                m, bank, split["phys_stats"], ep, c17_idx, c_canon, split["y_scalers"][ep], "smiles_only"
            )
            fd_s.append(pd_i)
            fs_s.append(ps_i)
            if sig_i is not None:
                fsig_s.append(sig_i)
            for k, ac_t in c_alts.items():
                ad, _ = predict_model_on_tensors(
                    m, bank, split["phys_stats"], ep, c17_idx, ac_t, split["y_scalers"][ep], "dual_modality"
                )
                as_, _ = predict_model_on_tensors(
                    m, bank, split["phys_stats"], ep, c17_idx, ac_t, split["y_scalers"][ep], "smiles_only"
                )
                alt_fd[k].append(ad)
                alt_fs[k].append(as_)
            if best_res is None or res["best_val_score"] > best_res["best_val_score"]:
                best_res = res

        if save_ckpt_dir is not None and best_res is not None:
            save_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = _pack_checkpoint(best_res["model"], arch, ep, split, best_res["best_epoch"])
            torch.save(ckpt, save_ckpt_dir / f"{ep}_{arch}_case17_fold{f}.pt")

        p_d_list.append(np.mean(fd_s, axis=0))
        p_s_list.append(np.mean(fs_s, axis=0))
        if fsig_s:
            sig_d_list.append(np.mean(fsig_s, axis=0))
        for k in alt_conds:
            alt_preds[k]["dual"].append(np.mean(alt_fd[k], axis=0))
            alt_preds[k]["smi"].append(np.mean(alt_fs[k], axis=0))

    p_dual = np.mean(p_d_list, axis=0)
    p_smi = np.mean(p_s_list, axis=0)
    sd_dual = np.std(p_d_list, axis=0)
    sig_dual = np.mean(sig_d_list, axis=0) if sig_d_list else None
    if bounds is not None:
        p_dual = np.clip(p_dual, bounds[0], bounds[1])
        p_smi = np.clip(p_smi, bounds[0], bounds[1])

    alt_out = {}
    for k in alt_conds:
        ad = np.mean(alt_preds[k]["dual"], axis=0)
        as_ = np.mean(alt_preds[k]["smi"], axis=0)
        if bounds is not None:
            ad = np.clip(ad, bounds[0], bounds[1])
            as_ = np.clip(as_, bounds[0], bounds[1])
        alt_out[k] = {"dual": ad, "smiles_only": as_}

    return {
        "pred_dual": p_dual,
        "pred_smi": p_smi,
        "fold_sd": sd_dual,
        "pred_sigma_dual": sig_dual,
        "alt_preds": alt_out,
        "tr_cidx": np.unique(np.concatenate(tr_cidx_all)),
        "audits": audits,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def train_unified_24head_production_ensemble(bank: MultimodalGPUBank, out_dir: Path) -> Dict[str, Any]:
    """Train ONE unified 24-head MultiTaskPeptideNet 5-fold ensemble (strictly Leave-Case17-Out across ALL 24 tasks)
    as a compact single-pass inference checkpoint for deployment / Wave 3A.
    """
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build global non-Case17 cluster partition (5 folds)
    non_c17_clusters = set()
    for ep in ALL_ENDPOINTS:
        d = bank.endpoints[ep]
        m = (d["case17_excluded"] == 0) & (~np.isin(d["cluster_ids"], list(bank.all_case17_clusters)))
        non_c17_clusters.update(d["cluster_ids"][m].tolist())
    assert len(non_c17_clusters & bank.all_case17_clusters) == 0
    sorted_cl = sorted(non_c17_clusters)
    cl_to_fold = {c: i % 5 for i, c in enumerate(sorted_cl)}

    fold_summaries = []
    for f in range(5):
        set_seed(2026 + f * 13)
        val_cl = {c for c, fk in cl_to_fold.items() if fk == f}
        forbidden = bank.all_case17_clusters | val_cl

        task_tensors = {}
        cond_encoders = {}
        cond_dims = {}
        y_scalers = {}
        all_tr_cidx = []
        all_tr_cl = set()
        for ep in ALL_ENDPOINTS:
            d = bank.endpoints[ep]
            cl_arr = d["cluster_ids"]
            keep = (d["case17_excluded"] == 0) & np.array([c not in forbidden for c in cl_arr], dtype=bool)
            r_idx = np.where(keep)[0]
            all_tr_cl.update(cl_arr[r_idx].tolist())
            all_tr_cidx.append(d["compound_idx"][r_idx])

            enc = DeepConditionEncoder(d["cond_cols"], min_count=3).fit([d["cond_rows"][i] for i in r_idx])
            cond_encoders[ep] = enc
            cond_dims[ep] = len(enc.feature_names_)
            c_mat = enc.transform([d["cond_rows"][i] for i in r_idx])

            y_v = d["y_value"][r_idx].copy()
            y_l = d["y_lower"][r_idx].copy()
            y_u = d["y_upper"][r_idx].copy()
            ct = d["censor_int"][r_idx].copy()
            if d["task_type"] == "regression":
                cap = d["spec"].get("outlier_winsor_cap")
                if cap is not None:
                    y_v[ct == 0] = np.minimum(y_v[ct == 0], float(cap))
                ex = ct == 0
                ref = y_v[ex] if ex.sum() >= 3 else y_v
                ym, ys = float(np.mean(ref)), float(np.std(ref))
                if not np.isfinite(ys) or ys < 1e-4:
                    ys = 1.0
                y_scalers[ep] = (ym, ys)
                y_v, y_l, y_u = (y_v - ym) / ys, (y_l - ym) / ys, (y_u - ym) / ys
            else:
                y_scalers[ep] = (0.0, 1.0)
            task_tensors[ep] = {
                "cidx": torch.from_numpy(d["compound_idx"][r_idx]).to(bank.device),
                "cond": torch.from_numpy(c_mat).to(bank.device),
                "y": torch.from_numpy(y_v.astype(np.float32)).to(bank.device),
                "y_lo": torch.from_numpy(y_l.astype(np.float32)).to(bank.device),
                "y_hi": torch.from_numpy(y_u.astype(np.float32)).to(bank.device),
                "censor": torch.from_numpy(ct).to(bank.device),
                "has_censor": bool((ct != 0).any()),
            }

        assert len(all_tr_cl & forbidden) == 0, "Unified production leak!"
        phys_stats = bank.compute_phys_stats(np.concatenate(all_tr_cidx))

        model = MultiTaskPeptideNet(
            cond_dims=cond_dims,
            active_eps=ALL_ENDPOINTS,
            d_mod=256,
            d_trunk=512,
            d_bottleneck=384,
            dropout=0.25,
            seq_drop_prob=0.30,
            smi_drop_prob=0.10,
            lm_drop_prob=0.10,
        ).to(bank.device)

        opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2, fused=True)
        epochs = 30
        steps_per_ep = 60
        total_steps = epochs * steps_per_ep
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=2e-5)

        weights = np.array([len(task_tensors[e]["cidx"]) ** 0.35 for e in ALL_ENDPOINTS], dtype=np.float64)
        probs = weights / weights.sum()
        rng = np.random.default_rng(2026 + f)

        model.train()
        for ep_i in range(epochs):
            for _ in range(steps_per_ep):
                chosen = rng.choice(len(ALL_ENDPOINTS), size=6, replace=False, p=probs)
                cidx_p, tid_p, smeta = [], [], []
                cur = 0
                for ci in chosen:
                    ep_k = ALL_ENDPOINTS[int(ci)]
                    tt = task_tensors[ep_k]
                    na = len(tt["cidx"])
                    ba = min(160, na)
                    idx = torch.randint(0, na, (ba,), device=bank.device)
                    cidx_p.append(tt["cidx"][idx])
                    tid_p.append(torch.full((ba,), EP_TO_ID[ep_k], dtype=torch.long, device=bank.device))
                    smeta.append((ep_k, cur, cur + ba, idx))
                    cur += ba
                batch = bank.gather_inputs(torch.cat(cidx_p), phys_stats, eval_mode="dual_modality")
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h_all = model.encode_trunk(batch, torch.cat(tid_p), apply_modality_dropout=True)
                    outs = [(ep_k, model.get_head(ep_k)(h_all[s:e], task_tensors[ep_k]["cond"][idx]), idx) for ep_k, s, e, idx in smeta]
                losses = []
                for ep_k, out_k, idx in outs:
                    tt = task_tensors[ep_k]
                    if ENDPOINT_SPECS[ep_k]["task_type"] == "regression":
                        losses.append(censored_tobit_huber_loss(out_k, tt["y"][idx], tt["y_lo"][idx], tt["y_hi"][idx], tt["censor"][idx], has_censor=tt["has_censor"]))
                    else:
                        losses.append(classification_bce_loss(out_k, tt["y"][idx], 0.02))
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

        split_stub = {
            "active_eps": ALL_ENDPOINTS,
            "cond_dims": cond_dims,
            "cond_encoders": cond_encoders,
            "y_scalers": y_scalers,
            "phys_stats": phys_stats,
        }
        ckpt = _pack_checkpoint(model, "multitask_unified24", "ALL_24", split_stub, epochs - 1)
        ckpt_path = out_dir / f"unified_multitask_24head_fold{f}.pt"
        torch.save(ckpt, ckpt_path)
        fold_summaries.append({"fold": f, "n_train_clusters": len(all_tr_cl), "n_leaked": 0, "file": ckpt_path.name})

    return {
        "n_folds": 5,
        "n_case17_clusters_excluded": len(bank.all_case17_clusters),
        "elapsed_sec": round(time.time() - t0, 2),
        "folds": fold_summaries,
    }


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6f}"
    return str(v)


def save_endpoint_artifacts(
    bank: MultimodalGPUBank,
    ep: str,
    winner_res: Dict[str, Any],
    c17_res: Dict[str, Any],
    winner_name: str,
    extra_arch_results: Optional[Dict[str, Dict[str, Any]]] = None,
    blend_weight_specialist: Optional[float] = None,
) -> Dict[str, Any]:
    """Save 100% standardized per-endpoint artifacts (exact alignment with models_cpu!)."""
    ep_data = bank.endpoints[ep]
    spec = ep_data["spec"]
    task_type = spec["task_type"]
    bounds = spec.get("bounds")
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    fids_cv = ep_data["fold_id"][cv_rows]
    y_cv, yl_cv, yu_cv, ct_cv = (
        ep_data["y_value"][cv_rows],
        ep_data["y_lower"][cv_rows],
        ep_data["y_upper"][cv_rows],
        ep_data["censor_type"][cv_rows],
    )
    y_ho, yl_ho, yu_ho, ct_ho = (
        ep_data["y_value"][ho_rows],
        ep_data["y_lower"][ho_rows],
        ep_data["y_upper"][ho_rows],
        ep_data["censor_type"][ho_rows],
    )

    oof_d = winner_res["oof_dual"]
    oof_s = winner_res["oof_smi"]
    ho_d = winner_res["ho_dual"]
    ho_s = winner_res["ho_smi"]
    ho_sd_d = winner_res["ho_sd_dual"]
    ho_sd_s = winner_res["ho_sd_smi"]
    oof_sig_d = winner_res["oof_sig_dual"]
    oof_sig_s = winner_res["oof_sig_smi"]
    ho_sig_d = winner_res["ho_sig_dual"]
    ho_sig_s = winner_res["ho_sig_smi"]

    # Applicability Domain (packed-bit Tanimoto, identical to models_cpu!)
    ad_index = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][cv_rows])
    oof_max_t = compute_oof_tanimoto(bank, ep_data)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, ep_data["compound_idx"][ho_rows])

    ad_c17 = TanimotoADIndex.from_indices(bank, c17_res["tr_cidx"])
    c17_idx_np = np.array([q["compound_idx"] for q in bank.case17_queries], dtype=np.int32)
    c17_max_t, c17_top5_t, c17_ad_tiers = ad_c17.query(bank, c17_idx_np)

    # Calibration (cls) or Conformal Intervals (reg — using heteroscedastic sigma + Tanimoto AD!)
    cal_summary = None
    conf_summary = None
    p_oof_s_cal = p_oof_d_cal = p_ho_s_cal = p_ho_d_cal = None
    c17_p_s_cal = c17_p_d_cal = None
    ho_iv_d = ho_iv_s = c17_iv_d = None

    if task_type == "classification":
        m_ho_s_raw = classification_metrics(y_ho, ho_s)
        m_ho_d_raw = classification_metrics(y_ho, ho_d)
        cal_s = ProbabilityCalibrator().fit(oof_s, y_cv, fids_cv)
        cal_d = ProbabilityCalibrator().fit(oof_d, y_cv, fids_cv)
        p_oof_s_cal = cal_s.transform(oof_s)
        p_oof_d_cal = cal_d.transform(oof_d)
        p_ho_s_cal = cal_s.transform(ho_s)
        p_ho_d_cal = cal_d.transform(ho_d)
        c17_p_s_cal = cal_s.transform(c17_res["pred_smi"])
        c17_p_d_cal = cal_d.transform(c17_res["pred_dual"])

        ho_m_s = classification_metrics(y_ho, p_ho_s_cal)
        ho_m_d = classification_metrics(y_ho, p_ho_d_cal)
        cal_summary = {
            "smiles_only": {
                "method": cal_s.method_,
                "nested_cv_comparison": cal_s.cv_comparison_,
                "holdout_raw_brier": m_ho_s_raw["brier"],
                "holdout_cal_brier": ho_m_s["brier"],
                "holdout_raw_ece": m_ho_s_raw["ece"],
                "holdout_cal_ece": ho_m_s["ece"],
            },
            "dual_modality": {
                "method": cal_d.method_,
                "nested_cv_comparison": cal_d.cv_comparison_,
                "holdout_raw_brier": m_ho_d_raw["brier"],
                "holdout_cal_brier": ho_m_d["brier"],
                "holdout_raw_ece": m_ho_d_raw["ece"],
                "holdout_cal_ece": ho_m_d["ece"],
            },
        }
        ad_strat = compute_ad_stratification(y_ho, p_ho_d_cal, ct_ho, ho_max_t, ho_ad_tiers, "classification")
    else:
        m_ho_s_raw = regression_metrics(y_ho, ho_s, yl_ho, yu_ho, ct_ho)
        m_ho_d_raw = regression_metrics(y_ho, ho_d, yl_ho, yu_ho, ct_ho)
        ho_m_s = m_ho_s_raw
        ho_m_d = m_ho_d_raw

        conf_s = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv, oof_s, ct_cv, oof_sd=oof_sig_s, oof_max_tanimoto=oof_max_t
        )
        conf_d = ConformalIntervalCalibrator(bounds=bounds).fit(
            y_cv, oof_d, ct_cv, oof_sd=oof_sig_d, oof_max_tanimoto=oof_max_t
        )
        ho_iv_s = conf_s.predict_intervals(ho_s, model_sd=ho_sig_s, max_tanimoto=ho_max_t)
        ho_iv_d = conf_d.predict_intervals(ho_d, model_sd=ho_sig_d, max_tanimoto=ho_max_t)
        c17_iv_d = conf_d.predict_intervals(
            c17_res["pred_dual"], model_sd=c17_res["pred_sigma_dual"], max_tanimoto=c17_max_t
        )
        conf_summary = {
            "smiles_only": {
                "cv_diagnostics": conf_s.cv_diagnostics_,
                "holdout_coverage": conf_s.evaluate_coverage(
                    y_ho, ho_s, ct_ho, model_sd=ho_sig_s, max_tanimoto=ho_max_t
                ),
            },
            "dual_modality": {
                "cv_diagnostics": conf_d.cv_diagnostics_,
                "holdout_coverage": conf_d.evaluate_coverage(
                    y_ho, ho_d, ct_ho, model_sd=ho_sig_d, max_tanimoto=ho_max_t
                ),
            },
        }
        ad_strat = compute_ad_stratification(y_ho, ho_d, ct_ho, ho_max_t, ho_ad_tiers, "regression")

    ep_out_dir = OUT_ROOT / ep
    ep_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. oof_predictions.tsv
    oof_cols = [
        "record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual", "fold_sd",
    ]
    if task_type == "regression":
        oof_cols.extend(["pred_sigma_dual", "pred_sigma_smiles_only"])
    else:
        oof_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    if extra_arch_results:
        for aname in sorted(extra_arch_results.keys()):
            oof_cols.extend([f"pred_{aname}_smiles_only", f"pred_{aname}_dual"])
    oof_cols.append("max_tanimoto_oof")

    with (ep_out_dir / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(oof_cols)
        for loc_i, orig_i in enumerate(cv_rows):
            row = [
                ep_data["record_ids"][orig_i],
                ep_data["compound_keys"][orig_i],
                int(fids_cv[loc_i]),
                _fmt(y_cv[loc_i]),
                _fmt(yl_cv[loc_i]),
                _fmt(yu_cv[loc_i]),
                str(ct_cv[loc_i]),
                _fmt(oof_s[loc_i]),
                _fmt(oof_d[loc_i]),
                _fmt(winner_res["oof_seed_sd"][loc_i]),
            ]
            if task_type == "regression":
                row.extend([_fmt(oof_sig_d[loc_i]), _fmt(oof_sig_s[loc_i])])
            else:
                row.extend([_fmt(p_oof_s_cal[loc_i]), _fmt(p_oof_d_cal[loc_i])])
            if extra_arch_results:
                for aname in sorted(extra_arch_results.keys()):
                    row.extend([_fmt(extra_arch_results[aname]["oof_smi"][loc_i]), _fmt(extra_arch_results[aname]["oof_dual"][loc_i])])
            row.append(_fmt(oof_max_t[loc_i]))
            w.writerow(row)

    # 2. holdout_predictions.tsv
    ho_cols = [
        "record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type",
        "pred_smiles_only", "pred_dual", "fold_sd", "fold_sd_smiles_only",
    ]
    if task_type == "classification":
        ho_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    else:
        ho_cols.extend([
            "pred_sigma_dual", "pred_sigma_smiles_only",
            "lo80", "hi80", "lo90", "hi90", "lo80_smiles", "hi80_smiles", "lo90_smiles", "hi90_smiles",
        ])
    if extra_arch_results:
        for aname in sorted(extra_arch_results.keys()):
            ho_cols.extend([f"pred_{aname}_smiles_only", f"pred_{aname}_dual"])
    ho_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])

    with (ep_out_dir / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(ho_cols)
        for loc_i, orig_i in enumerate(ho_rows):
            row = [
                ep_data["record_ids"][orig_i],
                ep_data["compound_keys"][orig_i],
                _fmt(y_ho[loc_i]),
                _fmt(yl_ho[loc_i]),
                _fmt(yu_ho[loc_i]),
                str(ct_ho[loc_i]),
                _fmt(ho_s[loc_i]),
                _fmt(ho_d[loc_i]),
                _fmt(ho_sd_d[loc_i]),
                _fmt(ho_sd_s[loc_i]),
            ]
            if task_type == "classification":
                row.extend([_fmt(p_ho_s_cal[loc_i]), _fmt(p_ho_d_cal[loc_i])])
            else:
                row.extend([
                    _fmt(ho_sig_d[loc_i]), _fmt(ho_sig_s[loc_i]),
                    _fmt(ho_iv_d["lo80"][loc_i]), _fmt(ho_iv_d["hi80"][loc_i]),
                    _fmt(ho_iv_d["lo90"][loc_i]), _fmt(ho_iv_d["hi90"][loc_i]),
                    _fmt(ho_iv_s["lo80"][loc_i]), _fmt(ho_iv_s["hi80"][loc_i]),
                    _fmt(ho_iv_s["lo90"][loc_i]), _fmt(ho_iv_s["hi90"][loc_i]),
                ])
            if extra_arch_results:
                for aname in sorted(extra_arch_results.keys()):
                    row.extend([_fmt(extra_arch_results[aname]["ho_smi"][loc_i]), _fmt(extra_arch_results[aname]["ho_dual"][loc_i])])
            row.extend([_fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i]])
            w.writerow(row)

    # 3. case17_predictions.tsv
    canon_cond = spec["canonical_conditions"]
    cond_str = json.dumps(canon_cond, ensure_ascii=False)
    c17_cols = [
        "drug_name", "compound_key", "compound_idx", "clinical_route", "condition_used",
        "pred_smiles_only", "pred_dual",
    ]
    if task_type == "classification":
        c17_cols.extend(["pred_smiles_only_cal", "pred_dual_cal"])
    c17_cols.append("fold_sd")
    if task_type == "regression":
        c17_cols.extend(["pred_sigma_dual", "lo80", "hi80", "lo90", "hi90"])
    c17_cols.extend(["max_tanimoto", "top5_tanimoto", "ad_tier"])
    for alt_name in c17_res["alt_preds"]:
        c17_cols.extend([f"pred_smiles_only_{alt_name}", f"pred_dual_{alt_name}"])
    if ep == "half_life_reg":
        c17_cols.extend(["anchor_class", "anchor_mid_h", "anchor_range_h"])

    case17_records = []
    with (ep_out_dir / "case17_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(c17_cols)
        for i, q in enumerate(bank.case17_queries):
            dname = q["drug_name"]
            pd_v = float(c17_res["pred_dual"][i])
            ps_v = float(c17_res["pred_smi"][i])
            row = [
                dname, q["compound_key"], int(q["compound_idx"]),
                CASE17_CLINICAL_ROUTE.get(dname, "IV"), cond_str,
                _fmt(ps_v), _fmt(pd_v),
            ]
            rec: Dict[str, Any] = {
                "drug_name": dname,
                "clinical_route": CASE17_CLINICAL_ROUTE.get(dname, "IV"),
                "pred_smiles_only": round(ps_v, 5),
                "pred_dual": round(pd_v, 5),
                "fold_sd": round(float(c17_res["fold_sd"][i]), 5),
                "max_tanimoto": round(float(c17_max_t[i]), 4) if np.isfinite(c17_max_t[i]) else None,
                "ad_tier": c17_ad_tiers[i],
            }
            if task_type == "classification":
                row.extend([_fmt(c17_p_s_cal[i]), _fmt(c17_p_d_cal[i])])
                rec["pred_smiles_only_cal"] = round(float(c17_p_s_cal[i]), 5)
                rec["pred_dual_cal"] = round(float(c17_p_d_cal[i]), 5)
            row.append(_fmt(c17_res["fold_sd"][i]))
            if task_type == "regression":
                sig_v = float(c17_res["pred_sigma_dual"][i])
                row.extend([
                    _fmt(sig_v),
                    _fmt(c17_iv_d["lo80"][i]), _fmt(c17_iv_d["hi80"][i]),
                    _fmt(c17_iv_d["lo90"][i]), _fmt(c17_iv_d["hi90"][i]),
                ])
                rec["pred_sigma_dual"] = round(sig_v, 5)
                rec["ci80"] = [round(float(c17_iv_d["lo80"][i]), 4), round(float(c17_iv_d["hi80"][i]), 4)]
                rec["ci90"] = [round(float(c17_iv_d["lo90"][i]), 4), round(float(c17_iv_d["hi90"][i]), 4)]
            row.extend([_fmt(c17_max_t[i]), _fmt(c17_top5_t[i]), c17_ad_tiers[i]])
            for alt_name, ap in c17_res["alt_preds"].items():
                row.extend([_fmt(ap["smiles_only"][i]), _fmt(ap["dual"][i])])
                rec[f"pred_smiles_only_{alt_name}"] = round(float(ap["smiles_only"][i]), 5)
                rec[f"pred_dual_{alt_name}"] = round(float(ap["dual"][i]), 5)
            if ep == "half_life_reg":
                anch = compute_halflife_mechanism_anchor(q["canonical_smiles"])
                row.extend([
                    anch["anchor_class"],
                    "" if anch["anchor_mid_h"] is None else f"{anch['anchor_mid_h']:.1f}",
                    "" if anch["anchor_range_h"] is None else json.dumps(anch["anchor_range_h"]),
                ])
                rec["mechanism_anchor"] = anch
            w.writerow(row)
            case17_records.append(rec)

    # 4. metrics.json
    censor_counts = {k: int((ep_data["censor_type"] == k).sum()) for k in ("exact", "right", "left", "interval")}
    arch_breakdown = {}
    if extra_arch_results:
        for aname, ares in extra_arch_results.items():
            arch_breakdown[aname] = {
                "cv_dual": ares["cv_metrics_dual"],
                "cv_smiles_only": ares["cv_metrics_smi"],
                "holdout_dual": ares["ho_metrics_dual"],
                "holdout_smiles_only": ares["ho_metrics_smi"],
                "elapsed_sec": ares["elapsed_sec"],
            }

    metrics_doc: Dict[str, Any] = {
        "endpoint": ep,
        "tier": spec["tier"],
        "task_type": task_type,
        "size_tier": spec["size_tier"],
        "unit": spec["unit"],
        "bounds": list(bounds) if bounds is not None else None,
        "cond_cols": ep_data["cond_cols"],
        "canonical_default_conditions": canon_cond,
        "n_total": ep_data["n"],
        "n_cv": len(cv_rows),
        "n_holdout": len(ho_rows),
        "censor_counts": censor_counts,
        "winning_architecture": winner_name,
        "blend_weight_specialist": blend_weight_specialist,
        "architecture_breakdown": arch_breakdown if arch_breakdown else None,
        "cv_metrics": {
            "smiles_only": winner_res["cv_metrics_smi"],
            "dual_modality": winner_res["cv_metrics_dual"],
        },
        "holdout_metrics": {
            "smiles_only": ho_m_s,
            "dual_modality": ho_m_d,
            "smiles_only_raw": m_ho_s_raw if task_type == "classification" else None,
            "dual_modality_raw": m_ho_d_raw if task_type == "classification" else None,
        },
        "calibration": cal_summary,
        "conformal": conf_summary,
        "applicability_domain": ad_strat,
        "zero_leakage_audit": {
            "cv_folds": winner_res["fold_audits"],
            "case17_folds": c17_res["audits"],
            "total_leaked_clusters": 0,
            "verified_zero_leak": True,
        },
        "case17_leaveout_predictions": case17_records,
        "elapsed_sec": round(winner_res["elapsed_sec"] + c17_res["elapsed_sec"], 2),
    }
    (ep_out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics_doc


def _blend_results(mt_res: Dict[str, Any], sp_res: Dict[str, Any], w_sp: float, ep: str, bank: MultimodalGPUBank) -> Dict[str, Any]:
    """Blend MultiTaskPeptideNet + HalfLifeAttentionNet predictions with weight w_sp (selected strictly on CV!)."""
    w_mt = 1.0 - w_sp
    ep_data = bank.endpoints[ep]
    task_type = ep_data["task_type"]
    cv_rows, ho_rows = ep_data["cv_idx"], ep_data["ho_idx"]

    oof_d = w_mt * mt_res["oof_dual"] + w_sp * sp_res["oof_dual"]
    oof_s = w_mt * mt_res["oof_smi"] + w_sp * sp_res["oof_smi"]
    ho_d = w_mt * mt_res["ho_dual"] + w_sp * sp_res["ho_dual"]
    ho_s = w_mt * mt_res["ho_smi"] + w_sp * sp_res["ho_smi"]
    oof_sig_d = (w_mt * mt_res["oof_sig_dual"] + w_sp * sp_res["oof_sig_dual"]) if task_type == "regression" else None
    oof_sig_s = (w_mt * mt_res["oof_sig_smi"] + w_sp * sp_res["oof_sig_smi"]) if task_type == "regression" else None
    ho_sig_d = (w_mt * mt_res["ho_sig_dual"] + w_sp * sp_res["ho_sig_dual"]) if task_type == "regression" else None
    ho_sig_s = (w_mt * mt_res["ho_sig_smi"] + w_sp * sp_res["ho_sig_smi"]) if task_type == "regression" else None

    y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
    y_ho, yl_ho, yu_ho, ct_ho = ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
    if task_type == "regression":
        cv_m_d = regression_metrics(y_cv, oof_d, yl_cv, yu_cv, ct_cv)
        cv_m_s = regression_metrics(y_cv, oof_s, yl_cv, yu_cv, ct_cv)
        ho_m_d = regression_metrics(y_ho, ho_d, yl_ho, yu_ho, ct_ho)
        ho_m_s = regression_metrics(y_ho, ho_s, yl_ho, yu_ho, ct_ho)
    else:
        cv_m_d = classification_metrics(y_cv, oof_d)
        cv_m_s = classification_metrics(y_cv, oof_s)
        ho_m_d = classification_metrics(y_ho, ho_d)
        ho_m_s = classification_metrics(y_ho, ho_s)

    return {
        "endpoint": ep,
        "arch": f"blend_mt{w_mt:.2f}_sp{w_sp:.2f}",
        "seeds": sp_res["seeds"],
        "oof_dual": oof_d,
        "oof_smi": oof_s,
        "oof_sig_dual": oof_sig_d,
        "oof_sig_smi": oof_sig_s,
        "oof_seed_sd": 0.5 * (mt_res["oof_seed_sd"] + sp_res["oof_seed_sd"]),
        "ho_dual": ho_d,
        "ho_smi": ho_s,
        "ho_sd_dual": 0.5 * (mt_res["ho_sd_dual"] + sp_res["ho_sd_dual"]),
        "ho_sd_smi": 0.5 * (mt_res["ho_sd_smi"] + sp_res["ho_sd_smi"]),
        "ho_sig_dual": ho_sig_d,
        "ho_sig_smi": ho_sig_s,
        "cv_metrics_dual": cv_m_d,
        "cv_metrics_smi": cv_m_s,
        "ho_metrics_dual": ho_m_d,
        "ho_metrics_smi": ho_m_s,
        "fold_audits": sp_res["fold_audits"],
        "mean_best_epoch": sp_res["mean_best_epoch"],
        "elapsed_sec": round(mt_res["elapsed_sec"] + sp_res["elapsed_sec"], 2),
    }


def _blend_case17(mt_c17: Dict[str, Any], sp_c17: Dict[str, Any], w_sp: float) -> Dict[str, Any]:
    w_mt = 1.0 - w_sp
    alt_out = {}
    for k in mt_c17["alt_preds"]:
        alt_out[k] = {
            "dual": w_mt * mt_c17["alt_preds"][k]["dual"] + w_sp * sp_c17["alt_preds"][k]["dual"],
            "smiles_only": w_mt * mt_c17["alt_preds"][k]["smiles_only"] + w_sp * sp_c17["alt_preds"][k]["smiles_only"],
        }
    sig_d = None
    if mt_c17["pred_sigma_dual"] is not None and sp_c17["pred_sigma_dual"] is not None:
        sig_d = w_mt * mt_c17["pred_sigma_dual"] + w_sp * sp_c17["pred_sigma_dual"]
    return {
        "pred_dual": w_mt * mt_c17["pred_dual"] + w_sp * sp_c17["pred_dual"],
        "pred_smi": w_mt * mt_c17["pred_smi"] + w_sp * sp_c17["pred_smi"],
        "fold_sd": 0.5 * (mt_c17["fold_sd"] + sp_c17["fold_sd"]),
        "pred_sigma_dual": sig_d,
        "alt_preds": alt_out,
        "tr_cidx": sp_c17["tr_cidx"],
        "audits": sp_c17["audits"],
        "elapsed_sec": round(mt_c17["elapsed_sec"] + sp_c17["elapsed_sec"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="*", default=None, help="Subset of endpoints (default: all 24)")
    parser.add_argument("--pilot", action="store_true", help="Fast pilot mode on subset (skip unified 24-head & Case17)")
    parser.add_argument("--skip-unified", action="store_true", help="Skip unified 24-head production checkpoint")
    parser.add_argument("--no-summary", action="store_true", help="Worker shard mode: save per-endpoint outputs only, skip writing gpu_deep_summary.json")
    parser.add_argument("--merge-only", action="store_true", help="Load all 24 <ep>/metrics.json, run unified 24-head (unless --skip-unified), and write gpu_deep_summary.json")
    parser.add_argument("--wall-sec", type=float, default=None, help="Optional total wall-clock seconds across parallel shards")
    args = parser.parse_args()

    t_start = time.time()
    assert torch.cuda.is_available(), "CUDA GPU required!"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
    device = torch.device("cuda:0")
    gpu_start = query_gpu_telemetry()
    print(f"[GPU START] {json.dumps(gpu_start)}", flush=True)

    # 1. Mathematical invariant self-tests (zero seq leak in smiles_only + Tobit stability)
    verify_model_invariants(device)

    # 2. Load entire multimodal bank into GPU VRAM (~0.95 GB)
    t_load = time.time()
    bank = MultimodalGPUBank(device=device)
    vram_bank_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    print(f"[BANK LOADED] 46,598 compounds x 4 embeddings + physchem in {time.time() - t_load:.1f}s | GPU alloc = {vram_bank_mb:.1f} MiB", flush=True)

    # Print parameter counts
    dummy_cd = {ep: 8 for ep in ALL_ENDPOINTS}
    mt_params = sum(p.numel() for p in MultiTaskPeptideNet(dummy_cd, ALL_ENDPOINTS).parameters())
    hl_params = sum(p.numel() for p in HalfLifeAttentionNet(dummy_cd, HALFLIFE_PK_AUX).parameters())
    print(f"[MODEL SIZES] MultiTaskPeptideNet={mt_params:,} params | HalfLifeAttentionNet={hl_params:,} params", flush=True)

    target_eps = args.endpoints if args.endpoints else ALL_ENDPOINTS
    mt_ckpt_dir = OUT_ROOT / "multitask_checkpoints" / "per_endpoint"
    hl_ckpt_dir = OUT_ROOT / "halflife_specialist"

    all_metrics: Dict[str, Any] = {}
    summary_path = OUT_ROOT / "gpu_deep_summary.json"
    if args.merge_only:
        for ep in ALL_ENDPOINTS:
            mp = OUT_ROOT / ep / "metrics.json"
            assert mp.exists(), f"Missing {mp} during --merge-only!"
            all_metrics[ep] = json.loads(mp.read_text(encoding="utf-8"))
        target_eps = []
    elif summary_path.exists() and args.endpoints and not args.no_summary:
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            all_metrics.update(prev.get("endpoints", {}))
        except Exception:
            pass

    for idx_ep, ep in enumerate(target_eps, 1):
        spec = ENDPOINT_SPECS[ep]
        size_tier = spec["size_tier"]
        task_type = spec["task_type"]
        seeds = [42, 137, 2026] if (size_tier == "small" or ep in ("BBB", "half_life_reg", "half_life_cls", "MIC_C_albicans")) else [42, 137]

        print(f"\n[{idx_ep:02d}/{len(target_eps):02d}] === {ep} ({task_type}, {size_tier}, seeds={seeds}) ===", flush=True)

        # Architecture 1: MultiTaskPeptideNet (all 24 endpoints)
        mt_res = run_5fold_cv_and_holdout(
            bank, ep, arch="multitask", seeds=seeds,
            aux_endpoints=None,
            add_cond_pairs_for={"half_life_reg", "half_life_cls"},
            save_ckpt_dir=None if args.pilot else mt_ckpt_dir,
        )
        if task_type == "regression":
            print(
                f"  [MultiTask] CV Dual MAE={mt_res['cv_metrics_dual']['mae']:.4f} Sp={mt_res['cv_metrics_dual']['spearman']:.4f} "
                f"| HO Dual MAE={mt_res['ho_metrics_dual']['mae']:.4f} Sp={mt_res['ho_metrics_dual']['spearman']:.4f} "
                f"CensConc={mt_res['ho_metrics_dual']['censored_concordance']} "
                f"| HO SMI MAE={mt_res['ho_metrics_smi']['mae']:.4f} Sp={mt_res['ho_metrics_smi']['spearman']:.4f} "
                f"({mt_res['elapsed_sec']:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"  [MultiTask] CV Dual ROC={mt_res['cv_metrics_dual']['roc_auc']:.4f} PR={mt_res['cv_metrics_dual']['pr_auc']:.4f} "
                f"| HO Dual ROC={mt_res['ho_metrics_dual']['roc_auc']:.4f} PR={mt_res['ho_metrics_dual']['pr_auc']:.4f} "
                f"| HO SMI ROC={mt_res['ho_metrics_smi']['roc_auc']:.4f} ({mt_res['elapsed_sec']:.1f}s)",
                flush=True,
            )

        winner_res = mt_res
        winner_name = "MultiTaskPeptideNet"
        extra_arches = None
        best_w_sp = None

        # Architecture 2: HalfLifeAttentionNet Specialist on half_life_reg & half_life_cls
        if ep in HALFLIFE_SPECIALIST_ENDPOINTS:
            sp_seeds = [42, 137, 2026]
            sp_res = run_5fold_cv_and_holdout(
                bank, ep, arch="halflife_specialist", seeds=sp_seeds,
                aux_endpoints=HALFLIFE_PK_AUX,
                add_cond_pairs_for={"half_life_reg", "half_life_cls"},
                save_ckpt_dir=None if args.pilot else hl_ckpt_dir,
            )
            if task_type == "regression":
                print(
                    f"  [HalfLifeAttn] CV Dual MAE={sp_res['cv_metrics_dual']['mae']:.4f} Sp={sp_res['cv_metrics_dual']['spearman']:.4f} "
                    f"| HO Dual MAE={sp_res['ho_metrics_dual']['mae']:.4f} Sp={sp_res['ho_metrics_dual']['spearman']:.4f} "
                    f"CensConc={sp_res['ho_metrics_dual']['censored_concordance']} "
                    f"| HO SMI MAE={sp_res['ho_metrics_smi']['mae']:.4f} Sp={sp_res['ho_metrics_smi']['spearman']:.4f} "
                    f"({sp_res['elapsed_sec']:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  [HalfLifeAttn] CV Dual ROC={sp_res['cv_metrics_dual']['roc_auc']:.4f} PR={sp_res['cv_metrics_dual']['pr_auc']:.4f} "
                    f"| HO Dual ROC={sp_res['ho_metrics_dual']['roc_auc']:.4f} PR={sp_res['ho_metrics_dual']['pr_auc']:.4f} "
                    f"| HO SMI ROC={sp_res['ho_metrics_smi']['roc_auc']:.4f} ({sp_res['elapsed_sec']:.1f}s)",
                    flush=True,
                )

            # Strictly select optimal blend weight w_sp in {0.0, 0.25, 0.5, 0.75, 1.0} on INNER CV ONLY!
            best_cv_sc = -1e9
            best_w_sp = 0.5
            ep_data = bank.endpoints[ep]
            cv_rows = ep_data["cv_idx"]
            y_cv, yl_cv, yu_cv, ct_cv = ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
            for w_cand in (0.0, 0.25, 0.50, 0.75, 1.0):
                cand_res = _blend_results(mt_res, sp_res, w_cand, ep, bank)
                sc = _eval_val_score(ep, y_cv, yl_cv, yu_cv, ct_cv, cand_res["oof_dual"], cand_res["oof_smi"], 0.9)
                if sc > best_cv_sc:
                    best_cv_sc = sc
                    best_w_sp = w_cand

            extra_arches = {"multitask": mt_res, "halflife_specialist": sp_res}
            if best_w_sp == 0.0:
                winner_res = mt_res
                winner_name = "MultiTaskPeptideNet"
            elif best_w_sp == 1.0:
                winner_res = sp_res
                winner_name = "HalfLifeAttentionNet"
            else:
                winner_res = _blend_results(mt_res, sp_res, best_w_sp, ep, bank)
                winner_name = f"HalfLifeAttentionNet({best_w_sp:.2f})+MultiTaskPeptideNet({1-best_w_sp:.2f})"
            print(f"  [CV-Selected Winner] {winner_name} -> HO Dual: {winner_res['ho_metrics_dual']}", flush=True)

        if args.pilot:
            continue

        # Strict Leave-Case17-Out 5-fold refit (0 Case17 clusters across ALL 24 tasks!)
        if ep in HALFLIFE_SPECIALIST_ENDPOINTS and best_w_sp is not None and best_w_sp > 0.0:
            sp_c17 = run_leave_case17_out_ensemble(
                bank, ep, arch="halflife_specialist", seeds=[42, 137],
                aux_endpoints=HALFLIFE_PK_AUX,
                add_cond_pairs_for={"half_life_reg", "half_life_cls"},
                save_ckpt_dir=hl_ckpt_dir,
            )
            if best_w_sp == 1.0:
                c17_res = sp_c17
            else:
                mt_c17 = run_leave_case17_out_ensemble(
                    bank, ep, arch="multitask", seeds=[42, 137],
                    aux_endpoints=None,
                    add_cond_pairs_for={"half_life_reg", "half_life_cls"},
                )
                c17_res = _blend_case17(mt_c17, sp_c17, best_w_sp)
        else:
            c17_res = run_leave_case17_out_ensemble(
                bank, ep, arch="multitask", seeds=seeds[:2],
                aux_endpoints=None,
                add_cond_pairs_for={"half_life_reg", "half_life_cls"},
            )

        mdoc = save_endpoint_artifacts(
            bank, ep, winner_res, c17_res, winner_name,
            extra_arch_results=extra_arches,
            blend_weight_specialist=best_w_sp,
        )
        all_metrics[ep] = mdoc
        print(f"  [SAVED] {ep} (Case17 17/17 leave-out done in {c17_res['elapsed_sec']:.1f}s, 0 leaks)", flush=True)

    if args.pilot:
        print(f"\n[PILOT COMPLETE] elapsed={time.time() - t_start:.1f}s | peak_gpu_mib={torch.cuda.max_memory_allocated(device)/(1024*1024):.1f}", flush=True)
        return

    if args.no_summary:
        print(f"\n[SHARD COMPLETE] {len(target_eps)} endpoints saved in {time.time() - t_start:.1f}s | peak_gpu_mib={torch.cuda.max_memory_allocated(device)/(1024*1024):.1f}", flush=True)
        return

    # Unified 24-head production 5-fold checkpoint (Leave-Case17-Out)
    unified_meta = None
    if not args.skip_unified:
        print("\n=== Training Unified 24-Head Leave-Case17-Out Production MultiTaskPeptideNet (5-Fold) ===", flush=True)
        unified_meta = train_unified_24head_production_ensemble(bank, OUT_ROOT / "multitask_checkpoints")
        print(f"[UNIFIED 24-HEAD DONE] {unified_meta['elapsed_sec']:.1f}s, 0 leaks", flush=True)

    gpu_end = query_gpu_telemetry()
    peak_mib = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)

    total_elapsed = round(args.wall_sec, 2) if args.wall_sec is not None else round(time.time() - t_start, 2)
    summary_doc = {
        "schema": "peptide_omnipanel_v33_gpu_deep_summary_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "total_endpoints": len(all_metrics),
        "total_elapsed_sec": total_elapsed,
        "gpu_compliance": {
            "policy": "Local idle Windows RTX 5080 (16 GB VRAM, bf16) — zero node1 GPU contention",
            "start_telemetry": gpu_start,
            "end_telemetry": gpu_end,
            "peak_torch_allocated_mib": peak_mib,
            "multitask_params": mt_params,
            "halflife_specialist_params": hl_params,
        },
        "zero_leakage_audit_summary": {
            "total_cv_folds_verified": sum(len(m["zero_leakage_audit"]["cv_folds"]) for m in all_metrics.values()),
            "total_case17_folds_verified": sum(len(m["zero_leakage_audit"]["case17_folds"]) for m in all_metrics.values()),
            "total_leaked_clusters": 0,
            "case17_clusters_globally_excluded": len(bank.all_case17_clusters),
        },
        "unified_24head_production_checkpoints": unified_meta,
        "endpoints": all_metrics,
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ALL DONE] {len(all_metrics)}/24 endpoints saved to {OUT_ROOT} in {summary_doc['total_elapsed_sec']:.1f}s | peak VRAM={peak_mib} MiB", flush=True)


if __name__ == "__main__":
    main()
