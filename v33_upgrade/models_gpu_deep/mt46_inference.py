#!/usr/bin/env python3
"""Portable CPU/GPU inference helper for 46-Endpoint MultiTaskPeptideNet46 checkpoints.

100% self-contained & portable (ZERO hardcoded Windows or node23 paths).
Loads `unified_mt46_fold{0..4}.pt` on CPU (0 MiB GPU VRAM) in eval mode while
strictly preserving exact `smiles_only` mathematical invariance (`0.00e+00` seq leak).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_data import DeepConditionEncoder


def _safe_head_key(ep: str) -> str:
    return ep.replace(".", "__dot__")


class ResidualSwiGLU46(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.20):
        super().__init__()
        h = hidden_dim or int(dim * 1.5)
        self.norm = nn.LayerNorm(dim)
        self.w_gate_val = nn.Linear(dim, h * 2)
        self.w_out = nn.Linear(h, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        u, v = self.w_gate_val(h).chunk(2, dim=-1)
        h = F.silu(u) * v
        h = self.drop(h)
        return x + self.drop(self.w_out(h))


class EmbeddingProjector46(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, in_drop: float = 0.20, out_drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(in_drop),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(out_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiRegionESMCBranch46(nn.Module):
    def __init__(self, in_dim: int = 1152, d_model: int = 256, n_heads: int = 4, dropout: float = 0.20):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_drop = dropout
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.region_pos = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ffn = ResidualSwiGLU46(d_model, hidden_dim=d_model * 2, dropout=dropout)
        self.pool_score = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
        )

    def forward(self, esmc_4reg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = esmc_4reg.shape[0]
        h = self.proj(esmc_4reg) + self.region_pos
        hn = self.attn_norm(h)
        qkv = self.qkv(hn).view(bsz, 4, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        attn_out = self.attn_out(attn_out.transpose(1, 2).reshape(bsz, 4, self.d_model))
        h = self.ffn(h + attn_out)
        w = torch.softmax(self.pool_score(h), dim=1)
        pooled = self.out_proj(torch.cat([(h * w).sum(dim=1), h.amax(dim=1)], dim=-1))
        return pooled, h


class ConditionInjectedTaskHead46(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        task_type: str,
        head_hidden: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.task_type = task_type
        out_dim = 2 if task_type == "regression" else 1

        if self.cond_dim > 0:
            self.cond_enc = nn.Sequential(
                nn.Linear(self.cond_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.GELU(),
            )
            self.film = nn.Linear(32, in_dim * 2)
            self.cond_baseline = nn.Linear(32, 1)
            self.mol_inter = nn.Linear(in_dim, 16, bias=False)
            self.cond_inter = nn.Linear(32, 16, bias=False)
        else:
            self.cond_enc = None

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, h_shared: torch.Tensor, cond_vec: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cond_enc is not None and cond_vec is not None and cond_vec.shape[1] > 0:
            c = self.cond_enc(cond_vec)
            gamma, beta = self.film(c).chunk(2, dim=-1)
            h_mod = h_shared * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)
            raw_out = self.mlp(h_mod)
            c_base = self.cond_baseline(c)
            inter = (self.mol_inter(h_shared) * self.cond_inter(c)).sum(dim=-1, keepdim=True) / 4.0
            delta = c_base + 0.15 * inter
            if self.task_type == "regression":
                return torch.cat([raw_out[:, 0:1] + delta, raw_out[:, 1:2]], dim=-1)
            return raw_out + delta
        return self.mlp(h_shared)


class MultiTaskPeptideNet46Portable(nn.Module):
    """Exact state-dict compatible portable 46-endpoint MultiTaskPeptideNet46."""

    def __init__(
        self,
        cond_dims: Dict[str, int],
        active_eps: List[str],
        task_types: Dict[str, str],
        d_mod: int = 256,
        d_trunk: int = 512,
        d_bottleneck: int = 384,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.active_eps = sorted(list(active_eps))
        self.ep_to_id = {ep: i for i, ep in enumerate(self.active_eps)}
        self.d_mod = d_mod

        self.enc_phys_chem = nn.Sequential(
            nn.Linear(48, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            ResidualSwiGLU46(d_mod, hidden_dim=d_mod, dropout=dropout * 0.6),
        )
        self.enc_phys_seq = nn.Sequential(
            nn.Linear(35, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            ResidualSwiGLU46(d_mod, hidden_dim=d_mod, dropout=dropout * 0.6),
        )

        self.enc_ncaa_smi = EmbeddingProjector46(768, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_ncaa_fus = EmbeddingProjector46(2048, d_mod, in_drop=0.25, out_drop=dropout * 0.7)
        self.enc_ncaa_prot = EmbeddingProjector46(1280, d_mod, in_drop=0.20, out_drop=dropout * 0.7)
        self.enc_pclm2 = EmbeddingProjector46(512, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_chemberta = EmbeddingProjector46(384, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_esmc = MultiRegionESMCBranch46(in_dim=1152, d_model=d_mod, n_heads=4, dropout=dropout * 0.8)

        self.n_modalities = 8
        self.mod_type_emb = nn.Parameter(torch.randn(1, self.n_modalities, d_mod) * 0.02)
        self.task_emb = nn.Embedding(len(self.active_eps), 32)
        self.gate_scorer = nn.Sequential(
            nn.Linear(d_mod + 32, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.task_branch_gate = nn.Linear(32, 2)

        self.fuse_proj = nn.Sequential(
            nn.Linear(d_mod * 3 + 32 + 2, d_trunk),
            nn.LayerNorm(d_trunk),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trunk_block1 = ResidualSwiGLU46(d_trunk, hidden_dim=int(d_trunk * 1.5), dropout=dropout)
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(d_trunk, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
        self.trunk_block2 = ResidualSwiGLU46(d_bottleneck, hidden_dim=int(d_bottleneck * 1.5), dropout=dropout)

        self.heads = nn.ModuleDict()
        for ep in self.active_eps:
            self.heads[_safe_head_key(ep)] = ConditionInjectedTaskHead46(
                in_dim=d_bottleneck,
                cond_dim=cond_dims.get(ep, 0),
                task_type=task_types[ep],
                head_hidden=128,
                dropout=dropout * 0.8,
            )

    def encode_trunk(self, batch: Dict[str, torch.Tensor], task_ids: torch.Tensor) -> torch.Tensor:
        smi_m = batch["smi_mask"]
        seq_m = batch["seq_mask"]

        t0 = self.enc_phys_chem(batch["phys_chem"])
        t1 = self.enc_ncaa_smi(batch["ncaa_smi"])
        t2 = self.enc_ncaa_fus(batch["ncaa_fus"])
        t3 = self.enc_ncaa_prot(batch["ncaa_prot"])
        t4 = self.enc_pclm2(batch["pclm2"])
        t5 = self.enc_chemberta(batch["chemberta"])
        t6 = self.enc_phys_seq(batch["phys_seq"])
        t7, _ = self.enc_esmc(batch["esmc_4reg"])

        tokens = torch.stack([t0, t1, t2, t3, t4, t5, t6, t7], dim=1)
        tok_mask = torch.stack([smi_m, smi_m, smi_m, smi_m, smi_m, smi_m, seq_m, seq_m], dim=1)
        tokens = tokens * tok_mask.unsqueeze(-1).to(tokens.dtype)

        t_emb = self.task_emb(task_ids)
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, self.n_modalities, -1)
        gate_in = torch.cat([tokens + self.mod_type_emb, t_emb_exp], dim=-1)
        logits = self.gate_scorer(gate_in).squeeze(-1).masked_fill(~tok_mask, -1e4)
        weights = torch.softmax(logits, dim=-1) * tok_mask.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_attn = (tokens * weights.unsqueeze(-1)).sum(dim=1)

        chem_m = tok_mask[:, :6].to(tokens.dtype)
        seq_m_2 = tok_mask[:, 6:].to(tokens.dtype)
        z_chem = (tokens[:, :6] * chem_m.unsqueeze(-1)).sum(dim=1) / chem_m.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_seq = (tokens[:, 6:] * seq_m_2.unsqueeze(-1)).sum(dim=1) / seq_m_2.sum(dim=1, keepdim=True).clamp(min=1.0)
        b_gate = torch.sigmoid(self.task_branch_gate(t_emb))
        z_chem = z_chem * (smi_m.unsqueeze(-1).to(tokens.dtype) * b_gate[:, 0:1].to(tokens.dtype))
        z_seq = z_seq * (seq_m.unsqueeze(-1).to(tokens.dtype) * b_gate[:, 1:2].to(tokens.dtype))

        flags = torch.stack([smi_m.to(tokens.dtype), seq_m.to(tokens.dtype)], dim=-1)
        fused = self.fuse_proj(torch.cat([z_attn, z_chem, z_seq, t_emb.to(tokens.dtype), flags], dim=-1))
        h = self.trunk_block1(fused)
        h = self.bottleneck_proj(h)
        h = self.trunk_block2(h)
        return h

    def forward_task(
        self,
        batch: Dict[str, torch.Tensor],
        ep: str,
        cond_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = batch["smi_mask"].shape[0]
        task_ids = torch.full((bsz,), self.ep_to_id[ep], dtype=torch.long, device=batch["smi_mask"].device)
        h = self.encode_trunk(batch, task_ids)
        return self.heads[_safe_head_key(ep)](h, cond_vec)


def load_mt46_checkpoint(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> Tuple[MultiTaskPeptideNet46Portable, Dict[str, Any]]:
    """Load an fp16 `unified_mt46_fold*.pt` checkpoint into eval mode with zero hardcoded paths."""
    dev = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    sd_fp16 = ckpt["state_dict_fp16"]
    active_eps = sorted(list(ckpt["active_eps"]))
    # Infer task_type directly from each head's final linear layer out_features (2 = regression, 1 = classification)
    task_types: Dict[str, str] = {}
    for ep in active_eps:
        w_key = f"heads.{_safe_head_key(ep)}.mlp.6.weight"
        out_dim = int(sd_fp16[w_key].shape[0])
        task_types[ep] = "regression" if out_dim == 2 else "classification"

    model = MultiTaskPeptideNet46Portable(
        cond_dims=ckpt["cond_dims"],
        active_eps=active_eps,
        task_types=task_types,
    )
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd_fp16.items()}
    model.load_state_dict(sd, strict=True)
    model.to(dev).eval()
    ckpt["_task_types"] = task_types
    return model, ckpt


@torch.no_grad()
def predict_mt46_from_numpy(
    model: MultiTaskPeptideNet46Portable,
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
    canonical_conditions: Optional[Dict[str, str]] = None,
    bounds: Optional[List[float]] = None,
    eval_mode: str = "dual_modality",
    device: torch.device | str = "cpu",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run exact inference for any of the 46 endpoints in `MultiTaskPeptideNet46Portable`."""
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
        c_np = enc.transform_constant(n, canonical_conditions or {})
    else:
        c_np = enc.transform(cond_rows)
    cond_t = torch.from_numpy(c_np).to(dev)

    raw = model.forward_task(batch, endpoint, cond_t)
    task_type = ckpt["_task_types"][endpoint]
    if task_type == "regression":
        mu_z = raw[:, 0].float()
        sig_z = F.softplus(raw[:, 1].float() + 0.5413).clamp(min=0.10, max=4.0)
        ym, ys = ckpt["y_scalers"][endpoint]
        mu = mu_z.cpu().numpy().astype(np.float64) * float(ys) + float(ym)
        sig = sig_z.cpu().numpy().astype(np.float64) * float(ys)
        if bounds is not None:
            mu = np.clip(mu, float(bounds[0]), float(bounds[1]))
        return mu, sig
    prob = torch.sigmoid(raw.squeeze(-1).float()).cpu().numpy().astype(np.float64)
    return np.clip(prob, 1e-6, 1.0 - 1e-6), None
