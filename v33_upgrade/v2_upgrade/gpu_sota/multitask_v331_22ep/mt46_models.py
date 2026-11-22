#!/usr/bin/env python3
"""46-Endpoint MultiTaskPeptideNet Architecture (Residual SwiGLU + ESM-C 4-Region Fused SDPA +
30% Stochastic Sequence Dropout + Mask-Aware Modality Gate + Residual FiLM + Exact Tobit Loss).

Scaled cleanly from the proven V33 24-endpoint MultiTaskPeptideNet to all 46 V33.1 endpoints
(24 core + 22 expansion), strictly preserving exact smiles_only mathematical invariance.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mt46_data import ALL_46_ENDPOINTS, ENDPOINT_SPECS_46, EP46_TO_ID


def safe_head_key(ep: str) -> str:
    return ep.replace(".", "__dot__")


class ResidualSwiGLU(nn.Module):
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


class EmbeddingProjector(nn.Module):
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


class MultiRegionESMCBranch(nn.Module):
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
        self.ffn = ResidualSwiGLU(d_model, hidden_dim=d_model * 2, dropout=dropout)
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


class ConditionInjectedTaskHead(nn.Module):
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
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
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


def censored_tobit_huber_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    y_lo: torch.Tensor,
    y_hi: torch.Tensor,
    censor: torch.Tensor,
    huber_weight: float = 0.50,
    nll_weight: float = 0.50,
    has_censor: bool = True,
) -> torch.Tensor:
    mu = pred[:, 0].float()
    raw_s = pred[:, 1].float()
    sigma = F.softplus(raw_s + 0.5413).clamp(min=0.10, max=4.0)
    y = y.float()

    r = y - mu
    nll_ex = 0.5 * (r / sigma) ** 2 + torch.log(sigma)
    hub_ex = F.huber_loss(mu, y, reduction="none", delta=1.0)
    loss_ex = nll_weight * nll_ex + huber_weight * hub_ex
    if not has_censor:
        return loss_ex.mean()

    y_lo = y_lo.float()
    y_hi = y_hi.float()

    z_r = ((mu - y_lo) / sigma).clamp(-15.0, 15.0)
    nll_r = -torch.special.log_ndtr(z_r)
    viol_r = F.relu(y_lo - mu)
    hub_r = F.huber_loss(viol_r, torch.zeros_like(viol_r), reduction="none", delta=1.0)
    loss_r = nll_weight * nll_r + huber_weight * hub_r

    z_l = ((y_hi - mu) / sigma).clamp(-15.0, 15.0)
    nll_l = -torch.special.log_ndtr(z_l)
    viol_l = F.relu(mu - y_hi)
    hub_l = F.huber_loss(viol_l, torch.zeros_like(viol_l), reduction="none", delta=1.0)
    loss_l = nll_weight * nll_l + huber_weight * hub_l

    u = ((y_hi - mu) / sigma).clamp(-15.0, 15.0)
    l = ((y_lo - mu) / sigma).clamp(-15.0, 15.0)
    u = torch.maximum(u, l + 1e-4)
    flip = u > 0.0
    a = torch.where(flip, -l, u)
    b = torch.where(flip, -u, l)
    la = torch.special.log_ndtr(a)
    lb = torch.special.log_ndtr(b)
    diff = (lb - la).clamp(max=-1e-6)
    log_prob = la + torch.log1p(-torch.exp(diff))
    nll_i = -log_prob.clamp(min=-25.0)
    viol_i = F.relu(y_lo - mu) + F.relu(mu - y_hi)
    hub_i = F.huber_loss(viol_i, torch.zeros_like(viol_i), reduction="none", delta=1.0)
    loss_i = nll_weight * nll_i + huber_weight * hub_i

    loss = torch.where(
        censor == 0,
        loss_ex,
        torch.where(censor == 1, loss_r, torch.where(censor == 2, loss_l, loss_i)),
    )
    return loss.mean()


def classification_bce_loss(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.02) -> torch.Tensor:
    y_sm = y.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), y_sm)


def extract_mu_sigma(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = pred[:, 0].float()
    sigma = F.softplus(pred[:, 1].float() + 0.5413).clamp(min=0.10, max=4.0)
    return mu, sigma


class MultiTaskPeptideNet46(nn.Module):
    """46-Endpoint Unified Multimodal Gated Trunk + Residual FiLM + Exact Tobit Loss."""

    def __init__(
        self,
        cond_dims: Dict[str, int],
        active_eps: List[str],
        d_mod: int = 256,
        d_trunk: int = 512,
        d_bottleneck: int = 384,
        dropout: float = 0.25,
        seq_drop_prob: float = 0.30,
        smi_drop_prob: float = 0.10,
        lm_drop_prob: float = 0.10,
    ):
        super().__init__()
        self.active_eps = list(active_eps)
        self.d_mod = d_mod
        self.seq_drop_prob = seq_drop_prob
        self.smi_drop_prob = smi_drop_prob
        self.lm_drop_prob = lm_drop_prob

        self.enc_phys_chem = nn.Sequential(
            nn.Linear(48, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            ResidualSwiGLU(d_mod, hidden_dim=d_mod, dropout=dropout * 0.6),
        )
        self.enc_phys_seq = nn.Sequential(
            nn.Linear(35, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
            ResidualSwiGLU(d_mod, hidden_dim=d_mod, dropout=dropout * 0.6),
        )

        self.enc_ncaa_smi = EmbeddingProjector(768, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_ncaa_fus = EmbeddingProjector(2048, d_mod, in_drop=0.25, out_drop=dropout * 0.7)
        self.enc_ncaa_prot = EmbeddingProjector(1280, d_mod, in_drop=0.20, out_drop=dropout * 0.7)
        self.enc_pclm2 = EmbeddingProjector(512, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_chemberta = EmbeddingProjector(384, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_esmc = MultiRegionESMCBranch(in_dim=1152, d_model=d_mod, n_heads=4, dropout=dropout * 0.8)

        self.n_modalities = 8
        self.mod_type_emb = nn.Parameter(torch.randn(1, self.n_modalities, d_mod) * 0.02)
        self.task_emb = nn.Embedding(len(ALL_46_ENDPOINTS), 32)
        self.gate_scorer = nn.Sequential(
            nn.Linear(d_mod + 32, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.task_branch_gate = nn.Linear(32, 2)
        nn.init.zeros_(self.task_branch_gate.weight)
        nn.init.constant_(self.task_branch_gate.bias, 1.0)

        self.fuse_proj = nn.Sequential(
            nn.Linear(d_mod * 3 + 32 + 2, d_trunk),
            nn.LayerNorm(d_trunk),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trunk_block1 = ResidualSwiGLU(d_trunk, hidden_dim=int(d_trunk * 1.5), dropout=dropout)
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(d_trunk, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
        self.trunk_block2 = ResidualSwiGLU(d_bottleneck, hidden_dim=int(d_bottleneck * 1.5), dropout=dropout)

        self.heads = nn.ModuleDict()
        for ep in self.active_eps:
            self.heads[safe_head_key(ep)] = ConditionInjectedTaskHead(
                in_dim=d_bottleneck,
                cond_dim=cond_dims.get(ep, 0),
                task_type=ENDPOINT_SPECS_46[ep]["task_type"],
                head_hidden=128,
                dropout=dropout * 0.8,
            )

    def get_head(self, ep: str) -> ConditionInjectedTaskHead:
        return self.heads[safe_head_key(ep)]

    def encode_trunk(
        self,
        batch: Dict[str, torch.Tensor],
        task_ids: torch.Tensor,
        apply_modality_dropout: bool = True,
    ) -> torch.Tensor:
        smi_m = batch["smi_mask"].clone()
        seq_m = batch["seq_mask"].clone()
        bsz = smi_m.shape[0]
        device = smi_m.device

        if self.training and apply_modality_dropout:
            both = smi_m & seq_m
            u = torch.rand(bsz, device=device)
            drop_seq = both & (u < self.seq_drop_prob)
            drop_smi = both & (u >= self.seq_drop_prob) & (u < self.seq_drop_prob + self.smi_drop_prob)
            seq_m = seq_m & (~drop_seq)
            smi_m = smi_m & (~drop_smi)

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

        if self.training and apply_modality_dropout and self.lm_drop_prob > 0:
            lm_keep = torch.rand((bsz, 5), device=device) >= self.lm_drop_prob
            tok_mask[:, 1:6] = tok_mask[:, 1:6] & lm_keep

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
        apply_modality_dropout: bool = True,
    ) -> torch.Tensor:
        bsz = batch["smi_mask"].shape[0]
        task_ids = torch.full((bsz,), EP46_TO_ID[ep], dtype=torch.long, device=batch["smi_mask"].device)
        h = self.encode_trunk(batch, task_ids, apply_modality_dropout=apply_modality_dropout)
        return self.get_head(ep)(h, cond_vec)


def verify_model_invariants_46(device: torch.device) -> None:
    torch.manual_seed(0)
    cond_dims = {ep: 4 for ep in ALL_46_ENDPOINTS}
    m = MultiTaskPeptideNet46(cond_dims, ALL_46_ENDPOINTS).to(device).eval()

    bsz = 8
    base = {
        "phys_chem": torch.randn(bsz, 48, device=device),
        "phys_seq": torch.randn(bsz, 35, device=device),
        "ncaa_smi": torch.randn(bsz, 768, device=device),
        "ncaa_fus": torch.randn(bsz, 2048, device=device),
        "ncaa_prot": torch.randn(bsz, 1280, device=device),
        "pclm2": torch.randn(bsz, 512, device=device),
        "chemberta": torch.randn(bsz, 384, device=device),
        "esmc_4reg": torch.randn(bsz, 4, 1152, device=device),
        "smi_mask": torch.ones(bsz, dtype=torch.bool, device=device),
        "seq_mask": torch.zeros(bsz, dtype=torch.bool, device=device),
    }
    perturbed = dict(base)
    perturbed["phys_seq"] = torch.randn(bsz, 35, device=device) * 50.0
    perturbed["esmc_4reg"] = torch.randn(bsz, 4, 1152, device=device) * 50.0
    cvec = torch.randn(bsz, 4, device=device)

    with torch.no_grad():
        diff = (
            m.forward_task(base, "cytotoxicity_CC50", cvec, False)
            - m.forward_task(perturbed, "cytotoxicity_CC50", cvec, False)
        ).abs().max().item()
    assert diff < 1e-6, f"MultiTaskPeptideNet46 smiles_only sequence leak: {diff}"

    pred = torch.tensor(
        [[-5.0, -3.0], [5.0, 2.0], [-10.0, -2.0], [10.0, -2.0], [0.0, 0.0], [12.0, -1.0]],
        device=device,
        requires_grad=True,
    )
    y = torch.tensor([0.0, 0.0, 2.0, -2.0, 0.5, 0.5], device=device)
    y_lo = torch.tensor([0.0, 0.0, 2.0, -5.0, -1.0, -1.0], device=device)
    y_hi = torch.tensor([0.0, 0.0, 5.0, -2.0, 1.0, 1.0], device=device)
    censor = torch.tensor([0, 0, 1, 2, 3, 3], dtype=torch.long, device=device)
    loss = censored_tobit_huber_loss(pred, y, y_lo, y_hi, censor)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all(), "Non-finite Tobit loss/grad!"
    print(f"[UNIT CHECK PASS] MT46 smiles_only invariance={diff:.2e} | Tobit extreme loss={loss.item():.4f}", flush=True)
