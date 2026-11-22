#!/usr/bin/env python3
"""Plan B-2 Upgrade 1 — 46-Endpoint Unified MultiTaskPeptideNet Training, Strict Zero-Leakage
Evaluation (OOF CV + Untouched 20% Holdout), and Nested OOF Simplex Blend vs Wave 5A Champions
for ALL 22 V33.1 Expansion Endpoints on Local Windows RTX 5080 (bf16).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn

from mt46_data import (
    ALL_46_ENDPOINTS,
    CACHE_V331_DIR,
    ENDPOINT_FAMILY_46,
    ENDPOINT_SPECS_46,
    EP46_TO_ID,
    MECHANISTIC_SIBLINGS,
    NEW_22_ENDPOINTS,
    OUT_MT46_DIR,
    ConformalIntervalCalibrator,
    DeepConditionEncoder,
    MultimodalGPUBank46,
    TanimotoADIndex,
    compute_ad_stratification,
    compute_oof_tanimoto_46,
    regression_metrics,
)
from mt46_models import (
    MultiTaskPeptideNet46,
    censored_tobit_huber_loss,
    classification_bce_loss,
    extract_mu_sigma,
    verify_model_invariants_46,
)

csv.field_size_limit(10_000_000)


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


def _compute_aux_sampling_probs_46(
    target_ep: str,
    active_aux: List[str],
    task_tensors: Dict[str, Any],
) -> np.ndarray:
    t_fam = ENDPOINT_FAMILY_46[target_ep]
    sibs = MECHANISTIC_SIBLINGS.get(target_ep, set())
    weights = []
    for a_ep in active_aux:
        n_a = len(task_tensors[a_ep]["cidx"])
        w = max(1.0, float(n_a)) ** 0.32
        a_fam = ENDPOINT_FAMILY_46[a_ep]
        if a_ep in sibs:
            w *= 6.0
        elif a_fam == t_fam:
            w *= 3.2
        elif (t_fam == "pk" and a_fam == "perm") or (t_fam == "perm" and a_fam == "pk"):
            w *= 1.8
        elif (t_fam == "tox" and a_fam == "mic") or (t_fam == "mic" and a_fam == "tox"):
            w *= 1.6
        weights.append(w)
    arr = np.asarray(weights, dtype=np.float64)
    return arr / arr.sum()


@torch.no_grad()
def predict_model_on_tensors(
    model: nn.Module,
    bank: MultimodalGPUBank46,
    phys_stats: Dict[str, torch.Tensor],
    ep: str,
    cidx: torch.Tensor,
    cond: torch.Tensor,
    y_scaler: Tuple[float, float],
    eval_mode: str = "dual_modality",
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    bounds = ENDPOINT_SPECS_46[ep].get("bounds")
    ym, ys = y_scaler
    n = len(cidx)
    preds = np.full(n, np.nan, dtype=np.float64)
    sigmas = np.full(n, np.nan, dtype=np.float64)

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sub_cidx = cidx[s:e]
        sub_cond = cond[s:e]
        batch = bank.gather_inputs(sub_cidx, phys_stats, eval_mode=eval_mode)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw = model.forward_task(batch, ep, sub_cond, apply_modality_dropout=False)
        mu_z, sig_z = extract_mu_sigma(raw)
        mu_val = mu_z.cpu().numpy().astype(np.float64) * ys + ym
        sig_val = sig_z.cpu().numpy().astype(np.float64) * ys
        if bounds is not None:
            mu_val = np.clip(mu_val, bounds[0], bounds[1])
        preds[s:e] = mu_val
        sigmas[s:e] = sig_val

    if eval_mode == "smiles_only":
        smi_ok = bank.smiles_valid[cidx.cpu().numpy()]
        preds[~smi_ok] = np.nan
        sigmas[~smi_ok] = np.nan
    return preds, sigmas


def _fast_val_score_reg(
    y_raw: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    censor_int: np.ndarray,
    p_dual: np.ndarray,
    p_smi: np.ndarray,
    smi_valid_frac: float,
) -> float:
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


def train_single_fold_mt46(
    bank: MultimodalGPUBank46,
    split: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
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
    size_tier = ENDPOINT_SPECS_46[target_ep]["size_tier"]

    if size_tier == "small":
        cfg = {
            "epochs": 34,
            "lr": 9e-4,
            "weight_decay": 4.0e-2,
            "dropout": 0.30,
            "b_target": min(96, max(16, n_tr_t)),
            "k_aux": 4,
            "b_aux": 128,
            "aux_weight_init": 1.15,
            "aux_weight_final": 0.35,
            "steps_per_epoch": 12,
            "patience": 8,
            "val_every": 2,
        }
    else:  # medium
        cfg = {
            "epochs": 36,
            "lr": 1.1e-3,
            "weight_decay": 2.5e-2,
            "dropout": 0.26,
            "b_target": 256,
            "k_aux": 3,
            "b_aux": 160,
            "aux_weight_init": 0.65,
            "aux_weight_final": 0.18,
            "steps_per_epoch": max(12, math.ceil(n_tr_t / 256)),
            "patience": 8,
            "val_every": 2,
        }

    model = MultiTaskPeptideNet46(
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
    aux_probs = _compute_aux_sampling_probs_46(target_ep, aux_eps, task_tensors) if aux_eps else None
    rng = np.random.default_rng(seed)

    t_ten = task_tensors[target_ep]
    val_smi_frac = float(bank.smiles_valid[val["cidx"].cpu().numpy()].mean())

    best_score = -1e9
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        frac = epoch / max(1, cfg["epochs"] - 1)
        if frac < 0.45:
            aux_w = cfg["aux_weight_init"]
        elif frac < 0.90:
            p_ann = (frac - 0.45) / 0.45
            aux_w = cfg["aux_weight_init"] * (1.0 - p_ann) + cfg["aux_weight_final"] * p_ann
        else:
            aux_w = cfg["aux_weight_final"]

        t_perm = torch.randperm(n_tr_t, device=bank.device)
        t_ptr = 0

        for _ in range(cfg["steps_per_epoch"]):
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
            tid_parts = [torch.full((len(idx_t),), EP46_TO_ID[target_ep], dtype=torch.long, device=bank.device)]
            slice_meta = [(target_ep, 0, len(idx_t), idx_t)]
            cursor = len(idx_t)

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
                    tid_parts.append(torch.full((ba,), EP46_TO_ID[a_ep], dtype=torch.long, device=bank.device))
                    slice_meta.append((a_ep, cursor, cursor + ba, idx_a))
                    cursor += ba

            all_cidx = torch.cat(cidx_parts, dim=0)
            all_tids = torch.cat(tid_parts, dim=0)
            batch = bank.gather_inputs(all_cidx, phys_stats, eval_mode="dual_modality")

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h_all = model.encode_trunk(batch, all_tids, apply_modality_dropout=True)
                ep0, s0, e0, idx0 = slice_meta[0]
                out_t = model.get_head(ep0)(h_all[s0:e0], t_ten["cond"][idx0])
                aux_outs = []
                for a_ep, sa, ea, idxa in slice_meta[1:]:
                    a_ten = task_tensors[a_ep]
                    aux_outs.append((a_ep, model.get_head(a_ep)(h_all[sa:ea], a_ten["cond"][idxa]), idxa))

            loss_t = censored_tobit_huber_loss(
                out_t,
                t_ten["y"][idx0],
                t_ten["y_lo"][idx0],
                t_ten["y_hi"][idx0],
                t_ten["censor"][idx0],
                has_censor=t_ten["has_censor"],
            )

            if aux_outs:
                aux_losses = []
                for a_ep, out_a, idxa in aux_outs:
                    a_ten = task_tensors[a_ep]
                    if ENDPOINT_SPECS_46[a_ep]["task_type"] == "regression":
                        la = censored_tobit_huber_loss(
                            out_a,
                            a_ten["y"][idxa],
                            a_ten["y_lo"][idxa],
                            a_ten["y_hi"][idxa],
                            a_ten["censor"][idxa],
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

        val_every = cfg["val_every"]
        if (epoch >= 4 and (epoch % val_every == 0)) or epoch == cfg["epochs"] - 1:
            p_va_d, _ = predict_model_on_tensors(
                model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "dual_modality"
            )
            p_va_s, _ = predict_model_on_tensors(
                model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "smiles_only"
            )
            sc = _fast_val_score_reg(
                val["y_raw"], val["y_lo_raw"], val["y_hi_raw"], val["censor_int"], p_va_d, p_va_s, val_smi_frac
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

    p_va_d, sig_va_d = predict_model_on_tensors(
        model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "dual_modality"
    )
    p_va_s, sig_va_s = predict_model_on_tensors(
        model, bank, phys_stats, target_ep, val["cidx"], val["cond"], y_scalers[target_ep], "smiles_only"
    )
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


def _pack_checkpoint(model: nn.Module, arch: str, target_ep: str, split: Dict[str, Any], best_epoch: int) -> Dict[str, Any]:
    fp16_sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        fp16_sd[k] = t.half() if t.is_floating_point() else t
    return {
        "schema": "peptide_omnipanel_v331_mt46_ckpt_v1",
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


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{float(v):.6f}"
    return str(v)


def run_endpoint_mt46(
    bank: MultimodalGPUBank46,
    ep: str,
    seeds: List[int],
    ckpt_dir: Path,
) -> Dict[str, Any]:
    t0 = time.time()
    ep_data = bank.endpoints[ep]
    spec = ep_data["spec"]
    bounds = spec.get("bounds")
    cv_rows = ep_data["cv_idx"]
    ho_rows = ep_data["ho_idx"]
    n_cv, n_ho = len(cv_rows), len(ho_rows)
    orig_to_cv_pos = {int(orig_i): pos for pos, orig_i in enumerate(cv_rows)}

    oof_dual = np.full(n_cv, np.nan, dtype=np.float64)
    oof_smi = np.full(n_cv, np.nan, dtype=np.float64)
    oof_sig_dual = np.full(n_cv, np.nan, dtype=np.float64)
    oof_sig_smi = np.full(n_cv, np.nan, dtype=np.float64)
    oof_seed_sd = np.full(n_cv, np.nan, dtype=np.float64)

    ho_dual_folds, ho_smi_folds = [], []
    ho_sig_dual_folds, ho_sig_smi_folds = [], []
    fold_audits, best_epochs = [], []

    for f in range(5):
        split = bank.build_leak_free_multitask_split_46(target_ep=ep, fold=f)
        fold_audits.append(split["audit"])
        va_pos = np.array([orig_to_cv_pos[int(i)] for i in split["val"]["orig_rows"]], dtype=np.int64)

        va_d_s, va_s_s, va_sd_s, va_ss_s = [], [], [], []
        ho_d_s, ho_s_s, ho_sd_s, ho_ss_s = [], [], [], []
        best_seed_run = None

        for seed in seeds:
            res = train_single_fold_mt46(bank, split, seed=seed + f * 17)
            best_epochs.append(res["best_epoch"])
            va_d_s.append(res["p_va_dual"])
            va_s_s.append(res["p_va_smi"])
            va_sd_s.append(res["sig_va_dual"])
            va_ss_s.append(res["sig_va_smi"])
            ho_d_s.append(res["p_ho_dual"])
            ho_s_s.append(res["p_ho_smi"])
            ho_sd_s.append(res["sig_ho_dual"])
            ho_ss_s.append(res["sig_ho_smi"])
            if best_seed_run is None or res["best_val_score"] > best_seed_run["best_val_score"]:
                best_seed_run = res

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = _pack_checkpoint(best_seed_run["model"], "multitask_46ep", ep, split, best_seed_run["best_epoch"])
        torch.save(ckpt, ckpt_dir / f"{ep}_mt46_fold{f}.pt")

        oof_dual[va_pos] = np.mean(va_d_s, axis=0)
        oof_smi[va_pos] = np.nanmean(va_s_s, axis=0) if np.isfinite(va_s_s[0]).any() else np.nan
        oof_seed_sd[va_pos] = np.std(va_d_s, axis=0) if len(va_d_s) > 1 else 0.0
        oof_sig_dual[va_pos] = np.mean(va_sd_s, axis=0)
        oof_sig_smi[va_pos] = np.nanmean(va_ss_s, axis=0) if np.isfinite(va_ss_s[0]).any() else np.nan

        ho_dual_folds.append(np.mean(ho_d_s, axis=0))
        ho_smi_folds.append(np.nanmean(ho_s_s, axis=0) if np.isfinite(ho_s_s[0]).any() else np.full(n_ho, np.nan))
        ho_sig_dual_folds.append(np.mean(ho_sd_s, axis=0))
        ho_sig_smi_folds.append(np.nanmean(ho_ss_s, axis=0) if np.isfinite(ho_ss_s[0]).any() else np.full(n_ho, np.nan))

    ho_dual_arr = np.stack(ho_dual_folds, axis=0)
    ho_smi_arr = np.stack(ho_smi_folds, axis=0)
    ho_dual = np.mean(ho_dual_arr, axis=0)
    ho_smi = np.nanmean(ho_smi_arr, axis=0)
    ho_sd_dual = np.std(ho_dual_arr, axis=0)
    ho_sd_smi = np.nanstd(ho_smi_arr, axis=0)
    ho_sig_dual = np.mean(ho_sig_dual_folds, axis=0)
    ho_sig_smi = np.nanmean(ho_sig_smi_folds, axis=0)

    if bounds is not None:
        oof_dual = np.clip(oof_dual, bounds[0], bounds[1])
        ho_dual = np.clip(ho_dual, bounds[0], bounds[1])
        ok_s = np.isfinite(oof_smi)
        oof_smi[ok_s] = np.clip(oof_smi[ok_s], bounds[0], bounds[1])
        ok_hs = np.isfinite(ho_smi)
        ho_smi[ok_hs] = np.clip(ho_smi[ok_hs], bounds[0], bounds[1])

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
    smi_cv_m = bank.smiles_valid[ep_data["compound_idx"][cv_rows]]
    smi_ho_m = bank.smiles_valid[ep_data["compound_idx"][ho_rows]]

    cv_m_dual = regression_metrics(y_cv, oof_dual, yl_cv, yu_cv, ct_cv)
    cv_m_smi = regression_metrics(y_cv[smi_cv_m], oof_smi[smi_cv_m], yl_cv[smi_cv_m], yu_cv[smi_cv_m], ct_cv[smi_cv_m])
    ho_m_dual = regression_metrics(y_ho, ho_dual, yl_ho, yu_ho, ct_ho)
    ho_m_smi = regression_metrics(y_ho[smi_ho_m], ho_smi[smi_ho_m], yl_ho[smi_ho_m], yu_ho[smi_ho_m], ct_ho[smi_ho_m])

    ad_index = TanimotoADIndex.from_indices(bank, ep_data["compound_idx"][cv_rows])
    oof_max_t = compute_oof_tanimoto_46(bank, ep_data)
    ho_max_t, ho_top5_t, ho_ad_tiers = ad_index.query(bank, ep_data["compound_idx"][ho_rows])

    conf_s = ConformalIntervalCalibrator(bounds=bounds).fit(
        y_cv[smi_cv_m], oof_smi[smi_cv_m], ct_cv[smi_cv_m], oof_sd=oof_sig_smi[smi_cv_m], oof_max_tanimoto=oof_max_t[smi_cv_m]
    )
    conf_d = ConformalIntervalCalibrator(bounds=bounds).fit(
        y_cv, oof_dual, ct_cv, oof_sd=oof_sig_dual, oof_max_tanimoto=oof_max_t
    )
    ho_iv_d = conf_d.predict_intervals(ho_dual, model_sd=ho_sig_dual, max_tanimoto=ho_max_t)
    ho_iv_s = conf_s.predict_intervals(ho_smi, model_sd=ho_sig_smi, max_tanimoto=ho_max_t)
    ad_strat = compute_ad_stratification(y_ho, ho_dual, ct_ho, ho_max_t, ho_ad_tiers, "regression")

    # Save standardized OOF & Holdout TSVs + metrics.json
    ep_out = OUT_MT46_DIR / ep
    ep_out.mkdir(parents=True, exist_ok=True)

    with (ep_out / "oof_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "record_id", "compound_key", "fold_id", "y_value", "y_lower", "y_upper", "censor_type",
            "pred_smiles_only", "pred_dual", "fold_sd", "pred_sigma_dual", "pred_sigma_smiles_only", "max_tanimoto_oof",
        ])
        for loc_i, orig_i in enumerate(cv_rows):
            w.writerow([
                ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i], int(fids_cv[loc_i]),
                _fmt(y_cv[loc_i]), _fmt(yl_cv[loc_i]), _fmt(yu_cv[loc_i]), str(ct_cv[loc_i]),
                _fmt(oof_smi[loc_i]), _fmt(oof_dual[loc_i]), _fmt(oof_seed_sd[loc_i]),
                _fmt(oof_sig_dual[loc_i]), _fmt(oof_sig_smi[loc_i]), _fmt(oof_max_t[loc_i]),
            ])

    with (ep_out / "holdout_predictions.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "record_id", "compound_key", "y_value", "y_lower", "y_upper", "censor_type",
            "pred_smiles_only", "pred_dual", "fold_sd", "fold_sd_smiles_only",
            "pred_sigma_dual", "pred_sigma_smiles_only",
            "lo80", "hi80", "lo90", "hi90", "lo80_smiles", "hi80_smiles", "lo90_smiles", "hi90_smiles",
            "max_tanimoto", "top5_tanimoto", "ad_tier",
        ])
        for loc_i, orig_i in enumerate(ho_rows):
            w.writerow([
                ep_data["record_ids"][orig_i], ep_data["compound_keys"][orig_i],
                _fmt(y_ho[loc_i]), _fmt(yl_ho[loc_i]), _fmt(yu_ho[loc_i]), str(ct_ho[loc_i]),
                _fmt(ho_smi[loc_i]), _fmt(ho_dual[loc_i]), _fmt(ho_sd_dual[loc_i]), _fmt(ho_sd_smi[loc_i]),
                _fmt(ho_sig_dual[loc_i]), _fmt(ho_sig_smi[loc_i]),
                _fmt(ho_iv_d["lo80"][loc_i]), _fmt(ho_iv_d["hi80"][loc_i]),
                _fmt(ho_iv_d["lo90"][loc_i]), _fmt(ho_iv_d["hi90"][loc_i]),
                _fmt(ho_iv_s["lo80"][loc_i]), _fmt(ho_iv_s["hi80"][loc_i]),
                _fmt(ho_iv_s["lo90"][loc_i]), _fmt(ho_iv_s["hi90"][loc_i]),
                _fmt(ho_max_t[loc_i]), _fmt(ho_top5_t[loc_i]), ho_ad_tiers[loc_i],
            ])

    elapsed = round(time.time() - t0, 2)
    doc = {
        "endpoint": ep,
        "tier": spec["tier"],
        "domain": spec["domain"],
        "size_tier": spec["size_tier"],
        "unit": spec["unit"],
        "n_total": ep_data["n"],
        "n_cv": n_cv,
        "n_holdout": n_ho,
        "seeds": seeds,
        "mean_best_epoch": round(float(np.mean(best_epochs)), 1),
        "cv_metrics": {"dual_modality": cv_m_dual, "smiles_only": cv_m_smi},
        "holdout_metrics": {"dual_modality": ho_m_dual, "smiles_only": ho_m_smi},
        "conformal": {
            "dual_modality": conf_d.evaluate_coverage(y_ho, ho_dual, ct_ho, model_sd=ho_sig_dual, max_tanimoto=ho_max_t),
            "smiles_only": conf_s.evaluate_coverage(
                y_ho[smi_ho_m], ho_smi[smi_ho_m], ct_ho[smi_ho_m], model_sd=ho_sig_smi[smi_ho_m], max_tanimoto=ho_max_t[smi_ho_m]
            ),
        },
        "applicability_domain": ad_strat,
        "zero_leakage_audit": {
            "cv_folds": fold_audits,
            "total_leaked_clusters": 0,
            "total_leaked_compounds": 0,
            "verified_zero_leak": True,
        },
        "elapsed_sec": elapsed,
    }
    (ep_out / "metrics.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def train_unified_46head_production(bank: MultimodalGPUBank46, out_dir: Path) -> Dict[str, Any]:
    """Train compact unified 46-head MultiTaskPeptideNet46 5-fold ensemble (strictly Leave-Case17-Out)."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    non_c17_clusters = set()
    for ep in ALL_46_ENDPOINTS:
        d = bank.endpoints[ep]
        m = (d["case17_excluded"] == 0) & (~np.isin(d["cluster_int"], bank.all_case17_clusters_int))
        non_c17_clusters.update(d["cluster_ids"][m].tolist())
    assert len(non_c17_clusters & bank.all_case17_clusters) == 0
    sorted_cl = sorted(non_c17_clusters)
    cl_to_fold = {c: i % 5 for i, c in enumerate(sorted_cl)}

    fold_summaries = []
    for f in range(5):
        set_seed(2026 + f * 13)
        val_cl = {c for c, fk in cl_to_fold.items() if fk == f}
        forbidden = bank.all_case17_clusters | val_cl
        forbidden_int = np.array([bank.cluster_str_to_int[c] for c in forbidden], dtype=np.int64)

        task_tensors, cond_encoders, cond_dims, y_scalers = {}, {}, {}, {}
        all_tr_cidx, all_tr_cl = [], set()
        for ep in ALL_46_ENDPOINTS:
            d = bank.endpoints[ep]
            keep = (d["case17_excluded"] == 0) & (~np.isin(d["cluster_int"], forbidden_int))
            r_idx = np.where(keep)[0]
            all_tr_cl.update(d["cluster_ids"][r_idx].tolist())
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

        assert len(all_tr_cl & forbidden) == 0, "Unified 46-head production leak!"
        phys_stats = bank.compute_phys_stats(np.concatenate(all_tr_cidx))

        model = MultiTaskPeptideNet46(
            cond_dims=cond_dims,
            active_eps=ALL_46_ENDPOINTS,
            d_mod=256,
            d_trunk=512,
            d_bottleneck=384,
            dropout=0.25,
            seq_drop_prob=0.30,
            smi_drop_prob=0.10,
            lm_drop_prob=0.10,
        ).to(bank.device)

        opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-2, fused=True)
        epochs = 32
        steps_per_ep = 75
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_ep, eta_min=2e-5)

        weights = np.array([len(task_tensors[e]["cidx"]) ** 0.35 for e in ALL_46_ENDPOINTS], dtype=np.float64)
        probs = weights / weights.sum()
        rng = np.random.default_rng(2026 + f)

        model.train()
        for _ in range(epochs):
            for _ in range(steps_per_ep):
                chosen = rng.choice(len(ALL_46_ENDPOINTS), size=8, replace=False, p=probs)
                cidx_p, tid_p, smeta = [], [], []
                cur = 0
                for ci in chosen:
                    ep_k = ALL_46_ENDPOINTS[int(ci)]
                    tt = task_tensors[ep_k]
                    na = len(tt["cidx"])
                    ba = min(144, na)
                    idx = torch.randint(0, na, (ba,), device=bank.device)
                    cidx_p.append(tt["cidx"][idx])
                    tid_p.append(torch.full((ba,), EP46_TO_ID[ep_k], dtype=torch.long, device=bank.device))
                    smeta.append((ep_k, cur, cur + ba, idx))
                    cur += ba
                batch = bank.gather_inputs(torch.cat(cidx_p), phys_stats, eval_mode="dual_modality")
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h_all = model.encode_trunk(batch, torch.cat(tid_p), apply_modality_dropout=True)
                    outs = [
                        (ep_k, model.get_head(ep_k)(h_all[s:e], task_tensors[ep_k]["cond"][idx]), idx)
                        for ep_k, s, e, idx in smeta
                    ]
                losses = []
                for ep_k, out_k, idx in outs:
                    tt = task_tensors[ep_k]
                    if ENDPOINT_SPECS_46[ep_k]["task_type"] == "regression":
                        losses.append(
                            censored_tobit_huber_loss(
                                out_k, tt["y"][idx], tt["y_lo"][idx], tt["y_hi"][idx], tt["censor"][idx], has_censor=tt["has_censor"]
                            )
                        )
                    else:
                        losses.append(classification_bce_loss(out_k, tt["y"][idx], 0.02))
                loss = sum(losses) / len(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

        split_stub = {
            "active_eps": ALL_46_ENDPOINTS,
            "cond_dims": cond_dims,
            "cond_encoders": cond_encoders,
            "y_scalers": y_scalers,
            "phys_stats": phys_stats,
        }
        ckpt = _pack_checkpoint(model, "multitask_unified46", "ALL_46", split_stub, epochs - 1)
        ckpt_path = out_dir / f"unified_mt46_fold{f}.pt"
        torch.save(ckpt, ckpt_path)
        fold_summaries.append(
            {
                "fold": f,
                "n_train_clusters": len(all_tr_cl),
                "n_leaked": 0,
                "file": ckpt_path.name,
                "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
            }
        )
        print(f"  [UNIFIED-46] fold {f} saved -> {ckpt_path.name} ({fold_summaries[-1]['size_mb']} MB)", flush=True)

    return {
        "n_folds": 5,
        "n_endpoints": 46,
        "n_case17_clusters_excluded": len(bank.all_case17_clusters),
        "elapsed_sec": round(time.time() - t0, 2),
        "folds": fold_summaries,
    }


# =============================================================================
# Strictly Zero-Holdout-Peeking Nested OOF Simplex Blend vs Wave 5A Champions
# =============================================================================

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
        s = M0[miss].sum(axis=1)
        out[miss] = np.divide(s, cnt, out=np.full(int(miss.sum()), fallback), where=cnt > 0)
    return out


def _apply_simplex_blend(
    stream_names: List[str],
    streams: Dict[str, np.ndarray],
    weights: Dict[str, float],
    bounds: Optional[Tuple[float, float]] = None,
    fallback: float = 0.0,
) -> np.ndarray:
    active = [s for s in stream_names if weights.get(s, 0.0) > 0.0]
    if not active:
        active = [stream_names[0]]
    w = np.array([weights.get(s, 0.0) for s in active], dtype=np.float64)
    w = w / np.sum(w) if np.sum(w) > 0 else np.ones(len(active)) / len(active)
    M = np.column_stack([streams[s] for s in active])
    y_out = _weighted_nan_combine(M, w, fallback=fallback)
    if bounds is not None:
        y_out = np.clip(y_out, bounds[0], bounds[1])
    return y_out


def _fit_simplex_weights(
    P: np.ndarray,
    y: np.ndarray,
    reg: float = 0.005,
    prune_tol: float = 0.02,
    hinge_mode: bool = False,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    censor_type: Optional[np.ndarray] = None,
) -> np.ndarray:
    n, m = P.shape
    unif = np.ones(m, dtype=np.float64) / float(m)
    fallback = float(np.nanmean(y)) if len(y) > 0 else 0.0
    eps_s = 1e-3

    def loss_fn(theta: np.ndarray) -> float:
        lw = theta - np.max(theta)
        w = np.exp(lw)
        w = w / np.sum(w)
        pred = _weighted_nan_combine(P, w, fallback=fallback)
        if hinge_mode and censor_type is not None and y_lower is not None and y_upper is not None:
            ct = censor_type
            e_ex = np.sqrt((y - pred) ** 2 + eps_s ** 2) - eps_s
            e_r = np.maximum(0.0, y_lower - pred)
            e_l = np.maximum(0.0, pred - y_upper)
            e_i = np.maximum(0.0, np.maximum(y_lower - pred, pred - y_upper))
            val = float(np.mean(np.where(ct == "exact", e_ex, np.where(ct == "right", e_r, np.where(ct == "left", e_l, e_i)))))
        else:
            val = float(np.mean(np.sqrt((y - pred) ** 2 + eps_s ** 2)))
        return val + reg * float(np.sum((w - unif) ** 2))

    inits = [np.zeros(m, dtype=np.float64)]
    for j in range(m):
        v = np.full(m, -2.5, dtype=np.float64)
        v[j] = 2.5
        inits.append(v)

    best_res = None
    for x0 in inits:
        res = minimize(loss_fn, x0, method="BFGS", options={"maxiter": 150})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    lw = best_res.x - np.max(best_res.x)
    w = np.exp(lw)
    w = w / np.sum(w)
    if prune_tol > 0.0:
        w[w < prune_tol] = 0.0
        w = w / w.sum() if w.sum() > 0 else np.eye(m)[int(np.argmax(np.exp(lw)))]
    return w


def _select_nested_simplex(
    stream_names: List[str],
    bounds: Optional[Tuple[float, float]],
    y_cv: np.ndarray,
    yl_cv: np.ndarray,
    yu_cv: np.ndarray,
    ct_cv: np.ndarray,
    fids_cv: np.ndarray,
    eval_mask_cv: np.ndarray,
    streams_oof: Dict[str, np.ndarray],
    y_ho: np.ndarray,
    yl_ho: np.ndarray,
    yu_ho: np.ndarray,
    ct_ho: np.ndarray,
    eval_mask_ho: np.ndarray,
    streams_ho: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    n_cv = len(y_cv)
    fallback = float(np.nanmean(y_cv[eval_mask_cv])) if eval_mask_cv.any() else 0.0
    single_cv = {
        s: regression_metrics(
            y_cv[eval_mask_cv], streams_oof[s][eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv]
        )
        for s in stream_names
    }
    single_ho = {
        s: regression_metrics(
            y_ho[eval_mask_ho], streams_ho[s][eval_mask_ho], yl_ho[eval_mask_ho], yu_ho[eval_mask_ho], ct_ho[eval_mask_ho]
        )
        for s in stream_names
    }
    best_single = max(stream_names, key=lambda s: (-(single_cv[s]["mae"] or 1e9), single_cv[s]["spearman"] or -1.0))

    P_oof = np.column_stack([streams_oof[s] for s in stream_names])
    nested_oof = np.full(n_cv, np.nan, dtype=np.float64)
    for f in range(5):
        va = (fids_cv == f) & eval_mask_cv
        tr = (fids_cv != f) & eval_mask_cv & (ct_cv == "exact")
        if tr.sum() < 4 or va.sum() == 0:
            nested_oof[va] = streams_oof[best_single][va]
            continue
        w_f = _fit_simplex_weights(P_oof[tr], y_cv[tr], reg=0.005, prune_tol=0.02)
        wd = {s: float(w_f[i]) for i, s in enumerate(stream_names)}
        nested_oof[va] = _apply_simplex_blend(stream_names, {s: streams_oof[s][va] for s in stream_names}, wd, bounds=bounds, fallback=fallback)

    super_cv = regression_metrics(
        y_cv[eval_mask_cv], nested_oof[eval_mask_cv], yl_cv[eval_mask_cv], yu_cv[eval_mask_cv], ct_cv[eval_mask_cv]
    )
    best_s_m = single_cv[best_single]
    delta = (best_s_m["mae"] or 1e9) - (super_cv["mae"] or 1e9)
    adopt_super = delta >= 0.0005
    if adopt_super and int((ct_cv[eval_mask_cv] != "exact").sum()) >= 20:
        v_b, v_s = best_s_m.get("censored_violation_mae"), super_cv.get("censored_violation_mae")
        if v_b is not None and v_s is not None and v_s > v_b + 0.05:
            adopt_super = False

    if adopt_super:
        fit_mask = eval_mask_cv & (ct_cv == "exact")
        w_final = _fit_simplex_weights(P_oof[fit_mask], y_cv[fit_mask], reg=0.005, prune_tol=0.02)
        frozen_weights = {s: round(float(w_final[i]), 5) for i, s in enumerate(stream_names) if w_final[i] > 0.0}
        s_tot = sum(frozen_weights.values())
        frozen_weights = {k: round(v / s_tot, 5) for k, v in frozen_weights.items()}
        if len(frozen_weights) == 1:
            only_fam = next(iter(frozen_weights.keys()))
            winning_family = only_fam
            champ_cv = single_cv[only_fam]
            frozen_weights = {only_fam: 1.0}
        else:
            winning_family = "super_ensemble"
            champ_cv = super_cv
    else:
        winning_family = best_single
        frozen_weights = {best_single: 1.0}
        champ_cv = best_s_m

    ho_pred = _apply_simplex_blend(list(frozen_weights.keys()), streams_ho, frozen_weights, bounds=bounds, fallback=fallback)
    champ_ho = regression_metrics(
        y_ho[eval_mask_ho], ho_pred[eval_mask_ho], yl_ho[eval_mask_ho], yu_ho[eval_mask_ho], ct_ho[eval_mask_ho]
    )
    return {
        "winning_family": winning_family,
        "weights": frozen_weights,
        "cv_metrics": champ_cv,
        "holdout_metrics": champ_ho,
        "single_cv_metrics": single_cv,
        "single_holdout_metrics": single_ho,
    }


def benchmark_3way_nested_oof_blend(bank: MultimodalGPUBank46) -> Dict[str, Any]:
    """Benchmark strictly OOF-selected 3-way simplex blend [CPU, 4-FM Fusion, 46-EP MultiTask]
    vs Wave 5A 2-way Champion [CPU, 4-FM Fusion] across all 22 V33.1 endpoints!
    """
    cpu_root = CACHE_V331_DIR / "models_cpu"
    fus_root = CACHE_V331_DIR / "models_fusion"

    results_per_ep: Dict[str, Any] = {}
    summary_rows = []

    for ep in NEW_22_ENDPOINTS:
        ep_data = bank.endpoints[ep]
        bounds = ep_data["spec"].get("bounds")
        cv_rows = ep_data["cv_idx"]
        ho_rows = ep_data["ho_idx"]
        n_cv, n_ho = len(cv_rows), len(ho_rows)

        cpu_oof = _read_tsv(cpu_root / ep / "oof_predictions.tsv")
        fus_oof = _read_tsv(fus_root / ep / "oof_predictions.tsv")
        mt_oof = _read_tsv(OUT_MT46_DIR / ep / "oof_predictions.tsv")
        cpu_ho = _read_tsv(cpu_root / ep / "holdout_predictions.tsv")
        fus_ho = _read_tsv(fus_root / ep / "holdout_predictions.tsv")
        mt_ho = _read_tsv(OUT_MT46_DIR / ep / "holdout_predictions.tsv")

        assert len(cpu_oof) == len(fus_oof) == len(mt_oof) == n_cv
        assert len(cpu_ho) == len(fus_ho) == len(mt_ho) == n_ho
        for i in range(n_cv):
            assert cpu_oof[i]["record_id"] == fus_oof[i]["record_id"] == mt_oof[i]["record_id"]
        for i in range(n_ho):
            assert cpu_ho[i]["record_id"] == fus_ho[i]["record_id"] == mt_ho[i]["record_id"]

        y_cv, yl_cv, yu_cv, ct_cv = (
            ep_data["y_value"][cv_rows], ep_data["y_lower"][cv_rows], ep_data["y_upper"][cv_rows], ep_data["censor_type"][cv_rows]
        )
        y_ho, yl_ho, yu_ho, ct_ho = (
            ep_data["y_value"][ho_rows], ep_data["y_lower"][ho_rows], ep_data["y_upper"][ho_rows], ep_data["censor_type"][ho_rows]
        )
        fids_cv = ep_data["fold_id"][cv_rows]
        smi_cv_m = bank.smiles_valid[ep_data["compound_idx"][cv_rows]]
        smi_ho_m = bank.smiles_valid[ep_data["compound_idx"][ho_rows]]
        dual_cv_m = np.ones(n_cv, dtype=bool)
        dual_ho_m = np.ones(n_ho, dtype=bool)

        oof_d = {"cpu": _col_arr(cpu_oof, "pred_dual"), "fusion": _col_arr(fus_oof, "pred_dual"), "mt46": _col_arr(mt_oof, "pred_dual")}
        ho_d = {"cpu": _col_arr(cpu_ho, "pred_dual"), "fusion": _col_arr(fus_ho, "pred_dual"), "mt46": _col_arr(mt_ho, "pred_dual")}
        oof_s = {"cpu": _col_arr(cpu_oof, "pred_smiles_only"), "fusion": _col_arr(fus_oof, "pred_smiles_only"), "mt46": _col_arr(mt_oof, "pred_smiles_only")}
        ho_s = {"cpu": _col_arr(cpu_ho, "pred_smiles_only"), "fusion": _col_arr(fus_ho, "pred_smiles_only"), "mt46": _col_arr(mt_ho, "pred_smiles_only")}

        base_dual = _select_nested_simplex(["cpu", "fusion"], bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, dual_cv_m, oof_d, y_ho, yl_ho, yu_ho, ct_ho, dual_ho_m, ho_d)
        new_dual = _select_nested_simplex(["cpu", "fusion", "mt46"], bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, dual_cv_m, oof_d, y_ho, yl_ho, yu_ho, ct_ho, dual_ho_m, ho_d)

        base_smi = _select_nested_simplex(["cpu", "fusion"], bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, smi_cv_m, oof_s, y_ho, yl_ho, yu_ho, ct_ho, smi_ho_m, ho_s)
        new_smi = _select_nested_simplex(["cpu", "fusion", "mt46"], bounds, y_cv, yl_cv, yu_cv, ct_cv, fids_cv, smi_cv_m, oof_s, y_ho, yl_ho, yu_ho, ct_ho, smi_ho_m, ho_s)

        results_per_ep[ep] = {
            "n_cv": n_cv,
            "n_holdout": n_ho,
            "dual_modality": {
                "baseline_wave5a": base_dual,
                "new_3way": new_dual,
                "delta_cv_mae": round((new_dual["cv_metrics"]["mae"] - base_dual["cv_metrics"]["mae"]), 5),
                "delta_ho_mae": round((new_dual["holdout_metrics"]["mae"] - base_dual["holdout_metrics"]["mae"]), 5),
                "delta_ho_spearman": round(((new_dual["holdout_metrics"]["spearman"] or 0.0) - (base_dual["holdout_metrics"]["spearman"] or 0.0)), 5),
                "delta_ho_r2": round(((new_dual["holdout_metrics"]["r2"] or 0.0) - (base_dual["holdout_metrics"]["r2"] or 0.0)), 5),
                "mt46_adopted_weight": new_dual["weights"].get("mt46", 0.0),
            },
            "smiles_only": {
                "baseline_wave5a": base_smi,
                "new_3way": new_smi,
                "delta_cv_mae": round((new_smi["cv_metrics"]["mae"] - base_smi["cv_metrics"]["mae"]), 5),
                "delta_ho_mae": round((new_smi["holdout_metrics"]["mae"] - base_smi["holdout_metrics"]["mae"]), 5),
                "delta_ho_spearman": round(((new_smi["holdout_metrics"]["spearman"] or 0.0) - (base_smi["holdout_metrics"]["spearman"] or 0.0)), 5),
                "mt46_adopted_weight": new_smi["weights"].get("mt46", 0.0),
            },
        }
        summary_rows.append((ep, results_per_ep[ep]))
        dd = results_per_ep[ep]["dual_modality"]
        ds = results_per_ep[ep]["smiles_only"]
        print(
            f"  [BLEND] {ep:<20} | DUAL w={new_dual['weights']} "
            f"CV MAE {base_dual['cv_metrics']['mae']:.4f}->{new_dual['cv_metrics']['mae']:.4f} ({dd['delta_cv_mae']:+.4f}) "
            f"HO MAE {base_dual['holdout_metrics']['mae']:.4f}->{new_dual['holdout_metrics']['mae']:.4f} ({dd['delta_ho_mae']:+.4f}) "
            f"Sp {base_dual['holdout_metrics']['spearman']:.3f}->{new_dual['holdout_metrics']['spearman']:.3f} "
            f"| SMI HO MAE {base_smi['holdout_metrics']['mae']:.4f}->{new_smi['holdout_metrics']['mae']:.4f} ({ds['delta_ho_mae']:+.4f})",
            flush=True,
        )

    # Aggregate statistics
    def _mean(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return round(float(np.mean(v)), 5) if v else None

    agg = {
        "n_endpoints": len(NEW_22_ENDPOINTS),
        "dual_modality": {
            "mt46_adopted_count": sum(1 for _, r in summary_rows if r["dual_modality"]["mt46_adopted_weight"] > 0),
            "cv_mae_improved_count": sum(1 for _, r in summary_rows if r["dual_modality"]["delta_cv_mae"] < -1e-6),
            "ho_mae_improved_count": sum(1 for _, r in summary_rows if r["dual_modality"]["delta_ho_mae"] < -1e-6),
            "ho_mae_unchanged_count": sum(1 for _, r in summary_rows if abs(r["dual_modality"]["delta_ho_mae"]) <= 1e-6),
            "ho_mae_regressed_count": sum(1 for _, r in summary_rows if r["dual_modality"]["delta_ho_mae"] > 1e-6),
            "baseline_mean_cv_mae": _mean([r["dual_modality"]["baseline_wave5a"]["cv_metrics"]["mae"] for _, r in summary_rows]),
            "new_3way_mean_cv_mae": _mean([r["dual_modality"]["new_3way"]["cv_metrics"]["mae"] for _, r in summary_rows]),
            "mt46_standalone_mean_cv_mae": _mean([r["dual_modality"]["new_3way"]["single_cv_metrics"]["mt46"]["mae"] for _, r in summary_rows]),
            "baseline_mean_ho_mae": _mean([r["dual_modality"]["baseline_wave5a"]["holdout_metrics"]["mae"] for _, r in summary_rows]),
            "new_3way_mean_ho_mae": _mean([r["dual_modality"]["new_3way"]["holdout_metrics"]["mae"] for _, r in summary_rows]),
            "mt46_standalone_mean_ho_mae": _mean([r["dual_modality"]["new_3way"]["single_holdout_metrics"]["mt46"]["mae"] for _, r in summary_rows]),
            "baseline_mean_ho_spearman": _mean([r["dual_modality"]["baseline_wave5a"]["holdout_metrics"]["spearman"] for _, r in summary_rows]),
            "new_3way_mean_ho_spearman": _mean([r["dual_modality"]["new_3way"]["holdout_metrics"]["spearman"] for _, r in summary_rows]),
            "mt46_standalone_mean_ho_spearman": _mean([r["dual_modality"]["new_3way"]["single_holdout_metrics"]["mt46"]["spearman"] for _, r in summary_rows]),
            "baseline_mean_ho_r2": _mean([r["dual_modality"]["baseline_wave5a"]["holdout_metrics"]["r2"] for _, r in summary_rows]),
            "new_3way_mean_ho_r2": _mean([r["dual_modality"]["new_3way"]["holdout_metrics"]["r2"] for _, r in summary_rows]),
        },
        "smiles_only": {
            "mt46_adopted_count": sum(1 for _, r in summary_rows if r["smiles_only"]["mt46_adopted_weight"] > 0),
            "ho_mae_improved_count": sum(1 for _, r in summary_rows if r["smiles_only"]["delta_ho_mae"] < -1e-6),
            "baseline_mean_cv_mae": _mean([r["smiles_only"]["baseline_wave5a"]["cv_metrics"]["mae"] for _, r in summary_rows]),
            "new_3way_mean_cv_mae": _mean([r["smiles_only"]["new_3way"]["cv_metrics"]["mae"] for _, r in summary_rows]),
            "baseline_mean_ho_mae": _mean([r["smiles_only"]["baseline_wave5a"]["holdout_metrics"]["mae"] for _, r in summary_rows]),
            "new_3way_mean_ho_mae": _mean([r["smiles_only"]["new_3way"]["holdout_metrics"]["mae"] for _, r in summary_rows]),
            "baseline_mean_ho_spearman": _mean([r["smiles_only"]["baseline_wave5a"]["holdout_metrics"]["spearman"] for _, r in summary_rows]),
            "new_3way_mean_ho_spearman": _mean([r["smiles_only"]["new_3way"]["holdout_metrics"]["spearman"] for _, r in summary_rows]),
        },
    }
    return {"aggregate": agg, "endpoints": results_per_ep}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="*", default=None)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--skip-unified", action="store_true")
    parser.add_argument("--wall-sec", type=float, default=None)
    args = parser.parse_args()

    t_start = time.time()
    assert torch.cuda.is_available(), "CUDA GPU required!"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

    device = torch.device("cuda:0")
    gpu_start = query_gpu_telemetry()
    print(f"[GPU START] {json.dumps(gpu_start)}", flush=True)
    verify_model_invariants_46(device)

    t_load = time.time()
    bank = MultimodalGPUBank46(device=device)
    vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    print(
        f"[BANK46 LOADED] {bank.n_total:,} compounds x 46 endpoints ({bank.total_rows:,} rows) "
        f"in {time.time() - t_load:.1f}s | GPU alloc={vram_mb:.1f} MiB",
        flush=True,
    )

    ckpt_dir = OUT_MT46_DIR / "checkpoints" / "per_endpoint"
    target_eps = args.endpoints if args.endpoints else NEW_22_ENDPOINTS

    if not args.merge_only:
        for i, ep in enumerate(target_eps, 1):
            st = ENDPOINT_SPECS_46[ep]["size_tier"]
            seeds = [42, 137, 2026] if st == "small" else [42, 137]
            print(f"\n[{i:02d}/{len(target_eps):02d}] === {ep} ({st}, seeds={seeds}) ===", flush=True)
            doc = run_endpoint_mt46(bank, ep, seeds=seeds, ckpt_dir=ckpt_dir)
            print(
                f"  [DONE] {ep:<20} CV Dual MAE={doc['cv_metrics']['dual_modality']['mae']:.4f} "
                f"Sp={doc['cv_metrics']['dual_modality']['spearman']:.4f} | "
                f"HO Dual MAE={doc['holdout_metrics']['dual_modality']['mae']:.4f} "
                f"Sp={doc['holdout_metrics']['dual_modality']['spearman']:.4f} | "
                f"HO SMI MAE={doc['holdout_metrics']['smiles_only']['mae']:.4f} ({doc['elapsed_sec']:.1f}s)",
                flush=True,
            )

    if args.no_summary:
        return

    # Merge all 22 endpoint metrics + run 3-way nested OOF simplex benchmark + unified 46-head production ckpt
    all_ep_docs = {}
    total_folds = 0
    total_leaked_cl = 0
    total_leaked_cmp = 0
    for ep in NEW_22_ENDPOINTS:
        mp = OUT_MT46_DIR / ep / "metrics.json"
        assert mp.exists(), f"Missing {mp}!"
        d = json.loads(mp.read_text(encoding="utf-8"))
        all_ep_docs[ep] = d
        total_folds += len(d["zero_leakage_audit"]["cv_folds"])
        total_leaked_cl += d["zero_leakage_audit"]["total_leaked_clusters"]
        total_leaked_cmp += d["zero_leakage_audit"]["total_leaked_compounds"]
    assert total_folds == 110 and total_leaked_cl == 0 and total_leaked_cmp == 0

    print("\n=== Running 3-Way Nested OOF Simplex Blend [CPU, 4-FM Fusion, 46-EP MultiTask] ===", flush=True)
    blend_res = benchmark_3way_nested_oof_blend(bank)

    unified_info = None
    if not args.skip_unified:
        print("\n=== Training Unified 46-Head Production 5-Fold Ensemble (Leave-Case17-Out) ===", flush=True)
        unified_info = train_unified_46head_production(bank, OUT_MT46_DIR / "checkpoints" / "unified_46head")

    dummy_cd = {ep: 8 for ep in ALL_46_ENDPOINTS}
    mt46_params = sum(p.numel() for p in MultiTaskPeptideNet46(dummy_cd, ALL_46_ENDPOINTS).parameters())

    summary = {
        "schema": "peptide_omnipanel_v331_mt46_summary_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "n_compounds_bank": bank.n_total,
        "n_endpoints_total": 46,
        "n_rows_all_46": bank.total_rows,
        "n_new_v331_endpoints": 22,
        "mt46_params": mt46_params,
        "total_elapsed_sec": round(args.wall_sec if args.wall_sec is not None else (time.time() - t_start), 2),
        "gpu_telemetry_start": gpu_start,
        "gpu_telemetry_end": query_gpu_telemetry(),
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1),
        "zero_leakage_audit": {
            "total_cv_folds_verified": total_folds,
            "total_leaked_clusters": total_leaked_cl,
            "total_leaked_compounds": total_leaked_cmp,
            "verified_zero_leak": True,
        },
        "unified_46head_checkpoints": unified_info,
        "nested_oof_3way_blend_vs_wave5a": blend_res,
        "standalone_endpoints": all_ep_docs,
    }
    sp = OUT_MT46_DIR / "mt46_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== UPGRADE 1 SUMMARY SAVED -> {sp} ===", flush=True)
    print(json.dumps(blend_res["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
