#!/usr/bin/env python3
"""Inference helper for Peptide OmniPanel V33 Deep Multi-Task & HalfLife Attention checkpoints.

Supports loading any saved `.pt` checkpoint from:
  - models_gpu_deep/multitask_checkpoints/unified_multitask_24head_fold{0..4}.pt
  - models_gpu_deep/multitask_checkpoints/per_endpoint/<ep>_multitask_fold{0..4}.pt
  - models_gpu_deep/halflife_specialist/<ep>_halflife_specialist_fold{0..4}.pt
for both `dual_modality` and `smiles_only` inference (zero sequence features in smiles_only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from deep_data import DeepConditionEncoder, ENDPOINT_SPECS
from deep_models import HalfLifeAttentionNet, MultiTaskPeptideNet, extract_mu_sigma


def load_deep_checkpoint(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load an inference-ready fp16 checkpoint back into float32/bf16 eval mode."""
    dev = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    arch = ckpt["arch"]
    if arch in ("multitask", "multitask_unified24"):
        model: torch.nn.Module = MultiTaskPeptideNet(
            cond_dims=ckpt["cond_dims"],
            active_eps=ckpt["active_eps"],
        )
    elif arch == "halflife_specialist":
        model = HalfLifeAttentionNet(
            cond_dims=ckpt["cond_dims"],
            active_eps=ckpt["active_eps"],
        )
    else:
        raise ValueError(f"Unsupported checkpoint arch: {arch}")

    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict_fp16"].items()}
    model.load_state_dict(sd)
    model.to(dev).eval()
    return model, ckpt


@torch.no_grad()
def predict_from_numpy(
    model: torch.nn.Module,
    ckpt: Dict[str, Any],
    endpoint: str,
    phys_83: np.ndarray,
    ncaa_smi: np.ndarray,
    ncaa_fus: np.ndarray,
    ncaa_prot: np.ndarray,
    pclm2: np.ndarray,
    chemberta: np.ndarray,
    esmc_4reg: np.ndarray,
    smi_valid: np.ndarray,
    seq_valid: np.ndarray,
    cond_rows: Optional[List[Dict[str, str]]] = None,
    eval_mode: str = "dual_modality",
    device: torch.device | str = "cpu",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference on raw numpy multimodal arrays for arbitrary new peptides."""
    dev = torch.device(device)
    n = len(phys_83)
    pstats = ckpt["phys_stats"]
    pc_mean = np.asarray(pstats["pc_mean"], dtype=np.float32)
    pc_std = np.asarray(pstats["pc_std"], dtype=np.float32)
    ps_mean = np.asarray(pstats["ps_mean"], dtype=np.float32)
    ps_std = np.asarray(pstats["ps_std"], dtype=np.float32)

    smi_m = np.asarray(smi_valid, dtype=bool).copy()
    seq_m = np.asarray(seq_valid, dtype=bool).copy()
    if eval_mode == "smiles_only":
        seq_m[:] = False
    elif eval_mode == "seq_only":
        smi_m[:] = False

    p_chem = np.clip((phys_83[:, :48] - pc_mean) / pc_std, -8.0, 8.0).astype(np.float32)
    p_seq = np.clip((phys_83[:, 48:83] - ps_mean) / ps_std, -8.0, 8.0).astype(np.float32)
    p_chem[~smi_m] = 0.0
    p_seq[~seq_m] = 0.0

    def _masked(arr: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        out = np.asarray(arr, dtype=np.float32).copy()
        out[~mask] = 0.0
        return torch.from_numpy(out).to(dev)

    batch = {
        "phys_chem": torch.from_numpy(p_chem).to(dev),
        "phys_seq": torch.from_numpy(p_seq).to(dev),
        "ncaa_smi": _masked(ncaa_smi, smi_m),
        "ncaa_fus": _masked(ncaa_fus, smi_m),
        "ncaa_prot": _masked(ncaa_prot, smi_m),
        "pclm2": _masked(pclm2, smi_m),
        "chemberta": _masked(chemberta, smi_m),
        "esmc_4reg": _masked(esmc_4reg, seq_m),
        "smi_mask": torch.from_numpy(smi_m).to(dev),
        "seq_mask": torch.from_numpy(seq_m).to(dev),
    }

    enc = DeepConditionEncoder.from_state_dict(ckpt["cond_encoders"][endpoint])
    if cond_rows is None:
        c_np = enc.transform_constant(n, ENDPOINT_SPECS[endpoint]["canonical_conditions"])
    else:
        c_np = enc.transform(cond_rows)
    cond_t = torch.from_numpy(c_np).to(dev)

    raw = model.forward_task(batch, endpoint, cond_t, apply_modality_dropout=False)
    spec = ENDPOINT_SPECS[endpoint]
    if spec["task_type"] == "regression":
        mu_z, sig_z = extract_mu_sigma(raw)
        ym, ys = ckpt["y_scalers"][endpoint]
        mu = mu_z.cpu().numpy().astype(np.float64) * ys + ym
        sig = sig_z.cpu().numpy().astype(np.float64) * ys
        if spec.get("bounds") is not None:
            mu = np.clip(mu, spec["bounds"][0], spec["bounds"][1])
        return mu, sig
    prob = torch.sigmoid(raw.squeeze(-1).float()).cpu().numpy().astype(np.float64)
    return np.clip(prob, 1e-6, 1.0 - 1e-6), None
