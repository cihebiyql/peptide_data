#!/usr/bin/env python3
"""PyTorch Deep Multi-Task & Attention Neural Architectures for Peptide OmniPanel V33:

1. MultiTaskPeptideNet:
   - Modality-specific encoders (Chem/Mod Physchem 48D, Seq Physchem 35D, ncAA-RepDistill v2
     smiles_repr 768D + fusion_hf 2048D + protein_proj 1280D, PeptideCLM-2 512D, ChemBERTa-MTR 384D)
   - Multi-Region ESM-C 600M Attention Branch ([mean, max, nterm5, cterm5] 4x1152D -> Multi-Head
     Self-Attention + Learned Attention Pooling)
   - Stochastic Modality Dropout (30% seq-drop, 10% smiles-drop) + Mask-Aware Learned Modality
     Attention Gate so the EXACT SAME checkpoint serves SMILES-only & Dual-modality cleanly!
   - Shared Residual SwiGLU Trunk (512D -> 384D)
   - 24 Condition-Injected Task Heads (Residual FiLM + PepLife-style Additive/Low-Rank Condition
     Decomposition) outputting (mu, log_sigma) for regression and logit for classification.
   - Exact Gaussian Tobit / Censored Log-Likelihood (right/left/interval) + Huber-NLL loss.

2. HalfLifeAttentionNet:
   - Dedicated Condition-Aware Attention + RBF-KAN Specialist for half_life_reg & half_life_cls
     with co-trained PK auxiliaries (PPB, CL_per_kg, Vd_per_kg, CL_absolute, LogD7.4).
   - Explicit RBF-KAN nonlinear encoder on peptide modification & PK-critical descriptors
     (fatty_diacid_len, fatty_monoacyl_len, is_cyclic, n_disulfide, n_nmethyl, n_d_aa_approx, has_peg).
   - Cross-Attention from Chemical/Modification query onto ESM-C 600M 4-region tokens + Self-Attention.
   - Explicit PepLife I5a-v2 Additive + Low-Rank Bilinear + Residual FiLM PK Condition Head + Tobit Loss.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_data import ALL_ENDPOINTS, ENDPOINT_SPECS, EP_TO_ID, PK_CORE_INDICES


def safe_head_key(ep: str) -> str:
    return ep.replace(".", "__dot__")


# =============================================================================
# Core Building Blocks
# =============================================================================

class ResidualSwiGLU(nn.Module):
    """Pre-LN Residual SwiGLU block with stable residual scaling."""

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
    """Projects a pretrained frozen embedding cleanly via LayerNorm -> Dropout -> Linear -> GELU -> LN."""

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
    """PepLife-inspired Multi-Region ESM-C 600M Attention Branch over [mean, max, nterm5, cterm5].
    Uses fused QKV + F.scaled_dot_product_attention for 5x lower kernel launch latency on RTX 5080.
    """

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
        """Returns (pooled_256, region_tokens_4x256)."""
        bsz = esmc_4reg.shape[0]
        h = self.proj(esmc_4reg) + self.region_pos  # (B, 4, D)
        hn = self.attn_norm(h)
        qkv = self.qkv(hn).view(bsz, 4, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, 4, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        attn_out = self.attn_out(attn_out.transpose(1, 2).reshape(bsz, 4, self.d_model))
        h = h + attn_out
        h = self.ffn(h)  # (B, 4, D)

        w = torch.softmax(self.pool_score(h), dim=1)  # (B, 4, 1)
        attn_pool = (h * w).sum(dim=1)  # (B, D)
        max_pool = h.amax(dim=1)  # (B, D)
        pooled = self.out_proj(torch.cat([attn_pool, max_pool], dim=-1))  # (B, D)
        return pooled, h


class RBFBasisKANLayer(nn.Module):
    """Smooth Gaussian-RBF / B-spline-inspired KAN residual layer for compact physical & modification features.

    Expands each scalar feature x_j onto K=8 smooth spline/RBF knots in [-3.5, 3.5] plus a linear SiLU skip,
    capturing nonlinear lipidation chain-length / charge / cyclization thresholds cleanly!
    """

    def __init__(self, in_dim: int, out_dim: int, n_knots: int = 8, dropout: float = 0.15):
        super().__init__()
        self.in_dim = in_dim
        self.n_knots = n_knots
        grid = torch.linspace(-3.5, 3.5, n_knots)
        self.register_buffer("grid", grid)
        self.inv_sigma = nn.Parameter(torch.full((in_dim, n_knots), 1.0))
        self.spline_linear = nn.Linear(in_dim * n_knots, out_dim)
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        xc = x.clamp(-5.0, 5.0).unsqueeze(-1)  # (B, D, 1)
        diff = (xc - self.grid.view(1, 1, -1)) * self.inv_sigma.unsqueeze(0)
        rbf = torch.exp(-0.5 * diff * diff)  # (B, D, K)
        spline_out = self.spline_linear(rbf.reshape(x.shape[0], -1))
        base_out = self.base_linear(F.silu(x))
        return self.drop(self.norm(spline_out + base_out))


class ConditionInjectedTaskHead(nn.Module):
    """Condition-Injected Task Head with bounded Residual FiLM + PepLife Additive/Low-Rank Condition Term."""

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
            # Zero-init FiLM so training starts smoothly at identity modulation
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

            # PepLife-style additive baseline + low-rank bilinear interaction
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

            c_base = self.cond_baseline(c)  # (B, 1)
            inter = (self.mol_inter(h_shared) * self.cond_inter(c)).sum(dim=-1, keepdim=True) / 4.0
            delta = c_base + 0.15 * inter
            if self.task_type == "regression":
                mu = raw_out[:, 0:1] + delta
                log_s = raw_out[:, 1:2]
                return torch.cat([mu, log_s], dim=-1)
            return raw_out + delta
        return self.mlp(h_shared)


# =============================================================================
# Exact Censored Gaussian Tobit + Hybrid Huber/NLL Loss (float32 stable!)
# =============================================================================

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
    """Exact Gaussian Tobit / Censored Log-Likelihood + Huber robust point loss in float32.

    censor codes:
      0 = exact:    y_true == y
      1 = right:    y_true >= y_lo   -> -log(1 - Phi((y_lo - mu)/sigma)) = -log_ndtr((mu - y_lo)/sigma)
      2 = left:     y_true <= y_hi   -> -log(Phi((y_hi - mu)/sigma))     = -log_ndtr((y_hi - mu)/sigma)
      3 = interval: y_lo <= y <= y_hi -> -log(Phi((y_hi - mu)/sigma) - Phi((y_lo - mu)/sigma))
    """
    mu = pred[:, 0].float()
    raw_s = pred[:, 1].float()
    # Smooth positive sigma in [0.10, 4.0], initialized near ~1.0 when raw_s=0
    sigma = F.softplus(raw_s + 0.5413).clamp(min=0.10, max=4.0)

    y = y.float()

    # 1. Exact rows (censor == 0)
    r = y - mu
    nll_ex = 0.5 * (r / sigma) ** 2 + torch.log(sigma)
    hub_ex = F.huber_loss(mu, y, reduction="none", delta=1.0)
    loss_ex = nll_weight * nll_ex + huber_weight * hub_ex
    if not has_censor:
        return loss_ex.mean()

    y_lo = y_lo.float()
    y_hi = y_hi.float()

    # 2. Right-censored (censor == 1): true y >= y_lo
    z_r = ((mu - y_lo) / sigma).clamp(-15.0, 15.0)
    nll_r = -torch.special.log_ndtr(z_r)
    viol_r = F.relu(y_lo - mu)
    hub_r = F.huber_loss(viol_r, torch.zeros_like(viol_r), reduction="none", delta=1.0)
    loss_r = nll_weight * nll_r + huber_weight * hub_r

    # 3. Left-censored (censor == 2): true y <= y_hi
    z_l = ((y_hi - mu) / sigma).clamp(-15.0, 15.0)
    nll_l = -torch.special.log_ndtr(z_l)
    viol_l = F.relu(mu - y_hi)
    hub_l = F.huber_loss(viol_l, torch.zeros_like(viol_l), reduction="none", delta=1.0)
    loss_l = nll_weight * nll_l + huber_weight * hub_l

    # 4. Interval-censored (censor == 3): y_lo <= y <= y_hi
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
        torch.where(
            censor == 1,
            loss_r,
            torch.where(censor == 2, loss_l, loss_i),
        ),
    )
    return loss.mean()


def classification_bce_loss(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.02) -> torch.Tensor:
    y_sm = y.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1).float(), y_sm)


def extract_mu_sigma(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = pred[:, 0].float()
    sigma = F.softplus(pred[:, 1].float() + 0.5413).clamp(min=0.10, max=4.0)
    return mu, sigma


# =============================================================================
# Architecture 1: MultiTaskPeptideNet
# =============================================================================

class MultiTaskPeptideNet(nn.Module):
    """Shared Multimodal Gated Trunk + Masked Multi-Task + Censored Tobit Loss."""

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

        # Branch 1: Chem/Mod Physchem (48D) & Seq Physchem (35D)
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

        # Branch 2: Modified-Peptide Chemical LMs (ncAA-RepDistill v2 + PeptideCLM-2 + ChemBERTa-MTR)
        self.enc_ncaa_smi = EmbeddingProjector(768, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_ncaa_fus = EmbeddingProjector(2048, d_mod, in_drop=0.25, out_drop=dropout * 0.7)
        self.enc_ncaa_prot = EmbeddingProjector(1280, d_mod, in_drop=0.20, out_drop=dropout * 0.7)
        self.enc_pclm2 = EmbeddingProjector(512, d_mod, in_drop=0.15, out_drop=dropout * 0.6)
        self.enc_chemberta = EmbeddingProjector(384, d_mod, in_drop=0.15, out_drop=dropout * 0.6)

        # Branch 3: Multi-Region PLM (ESM-C 600M 4x1152D -> Self-Attention + Attention Pool)
        self.enc_esmc = MultiRegionESMCBranch(in_dim=1152, d_model=d_mod, n_heads=4, dropout=dropout * 0.8)

        # Learned Modality Attention Gate (8 modality tokens: 6 chem + 2 seq)
        self.n_modalities = 8
        self.mod_type_emb = nn.Parameter(torch.randn(1, self.n_modalities, d_mod) * 0.02)
        self.task_emb = nn.Embedding(len(ALL_ENDPOINTS), 32)
        self.gate_scorer = nn.Sequential(
            nn.Linear(d_mod + 32, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Task-conditioned branch gate so chemical/modification tasks (e.g. PPB, LogD, PAMPA)
        # can dynamically down-weight canonical-only sequence embeddings while sequence-heavy
        # tasks (e.g. BBB, hemolysis, MIC) keep sequence branch fully open!
        self.task_branch_gate = nn.Linear(32, 2)
        nn.init.zeros_(self.task_branch_gate.weight)
        nn.init.constant_(self.task_branch_gate.bias, 1.0)

        # Fuse attention-gated token + gated chem mean + gated seq mean + task_emb + availability flags -> d_trunk
        self.fuse_proj = nn.Sequential(
            nn.Linear(d_mod * 3 + 32 + 2, d_trunk),
            nn.LayerNorm(d_trunk),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared Trunk: 512D -> 384D residual bottleneck
        self.trunk_block1 = ResidualSwiGLU(d_trunk, hidden_dim=int(d_trunk * 1.5), dropout=dropout)
        self.bottleneck_proj = nn.Sequential(
            nn.Linear(d_trunk, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
        self.trunk_block2 = ResidualSwiGLU(d_bottleneck, hidden_dim=int(d_bottleneck * 1.5), dropout=dropout)

        # Condition-Injected Task Heads (up to 24 heads)
        self.heads = nn.ModuleDict()
        for ep in self.active_eps:
            self.heads[safe_head_key(ep)] = ConditionInjectedTaskHead(
                in_dim=d_bottleneck,
                cond_dim=cond_dims.get(ep, 0),
                task_type=ENDPOINT_SPECS[ep]["task_type"],
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

        # Stochastic Modality Dropout during training (branchless, zero GPU-CPU sync!)
        if self.training and apply_modality_dropout:
            both = smi_m & seq_m
            u = torch.rand(bsz, device=device)
            drop_seq = both & (u < self.seq_drop_prob)
            drop_smi = both & (u >= self.seq_drop_prob) & (u < self.seq_drop_prob + self.smi_drop_prob)
            seq_m = seq_m & (~drop_seq)
            smi_m = smi_m & (~drop_smi)

        # Encode all 8 modality tokens
        t0 = self.enc_phys_chem(batch["phys_chem"])
        t1 = self.enc_ncaa_smi(batch["ncaa_smi"])
        t2 = self.enc_ncaa_fus(batch["ncaa_fus"])
        t3 = self.enc_ncaa_prot(batch["ncaa_prot"])
        t4 = self.enc_pclm2(batch["pclm2"])
        t5 = self.enc_chemberta(batch["chemberta"])
        t6 = self.enc_phys_seq(batch["phys_seq"])
        t7, _ = self.enc_esmc(batch["esmc_4reg"])

        tokens = torch.stack([t0, t1, t2, t3, t4, t5, t6, t7], dim=1)  # (B, 8, D)

        # Build exact token availability mask (B, 8)
        tok_mask = torch.stack([smi_m, smi_m, smi_m, smi_m, smi_m, smi_m, seq_m, seq_m], dim=1)

        # Optional gentle individual chemical LM dropout during training (never drops phys_chem t0)
        if self.training and apply_modality_dropout and self.lm_drop_prob > 0:
            lm_keep = torch.rand((bsz, 5), device=device) >= self.lm_drop_prob
            # Ensure at least one LM stays if smi_m is True
            tok_mask[:, 1:6] = tok_mask[:, 1:6] & lm_keep

        # Zero unavailable tokens BEFORE pooling (guarantees exact mathematical invariance!)
        mask_f = tok_mask.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens * mask_f

        # Task-conditioned Learned Modality Attention Gate
        t_emb = self.task_emb(task_ids)  # (B, 32)
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, self.n_modalities, -1)
        gate_in = torch.cat([tokens + self.mod_type_emb, t_emb_exp], dim=-1)
        logits = self.gate_scorer(gate_in).squeeze(-1)  # (B, 8)
        logits = logits.masked_fill(~tok_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        # Re-mask weights explicitly so unavailable modalities have exact 0.0 weight
        weights = weights * tok_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_attn = (tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # Masked branch means (chem 0..5, seq 6..7)
        chem_m = tok_mask[:, :6].to(tokens.dtype)
        seq_m_2 = tok_mask[:, 6:].to(tokens.dtype)
        z_chem = (tokens[:, :6] * chem_m.unsqueeze(-1)).sum(dim=1) / chem_m.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_seq = (tokens[:, 6:] * seq_m_2.unsqueeze(-1)).sum(dim=1) / seq_m_2.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Explicitly zero z_chem / z_seq when branch absent, then apply task-conditioned branch gate
        b_gate = torch.sigmoid(self.task_branch_gate(t_emb))  # (B, 2)
        z_chem = z_chem * (smi_m.unsqueeze(-1).to(tokens.dtype) * b_gate[:, 0:1].to(tokens.dtype))
        z_seq = z_seq * (seq_m.unsqueeze(-1).to(tokens.dtype) * b_gate[:, 1:2].to(tokens.dtype))

        flags = torch.stack([smi_m.to(tokens.dtype), seq_m.to(tokens.dtype)], dim=-1)  # (B, 2)
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
        task_ids = torch.full((bsz,), EP_TO_ID[ep], dtype=torch.long, device=batch["smi_mask"].device)
        h = self.encode_trunk(batch, task_ids, apply_modality_dropout=apply_modality_dropout)
        return self.get_head(ep)(h, cond_vec)


# =============================================================================
# Architecture 2: HalfLifeAttentionNet (Dedicated Specialist!)
# =============================================================================

class HalfLifeAttentionNet(nn.Module):
    """Dedicated Condition-Aware Cross-Attention + RBF-KAN Specialist for half_life_reg & half_life_cls
    with co-trained PK auxiliaries (PPB, CL_per_kg, Vd_per_kg, CL_absolute, LogD7.4).

    Combines:
      1. RBF-KAN nonlinear encoder over explicit peptide modification & PK descriptors
         (fatty_diacid_len, fatty_monoacyl_len, is_cyclic, n_disulfide, n_nmethyl, n_d_aa_approx, has_peg).
      2. Chemical LM fusion (ncAA-RepDistill v2 + PeptideCLM-2 + ChemBERTa-MTR).
      3. Multi-Region ESM-C 600M Self-Attention + Chem-to-ESMC Cross-Attention (explicitly querying
         N-term/C-term proteolytic vulnerability conditioned on chemical modification state!).
      4. PepLife I5a-v2 Explicit Intrinsic + Condition Baseline + Low-Rank Bilinear + Residual FiLM Head.
    """

    def __init__(
        self,
        cond_dims: Dict[str, int],
        active_eps: List[str],
        d_model: int = 256,
        dropout: float = 0.25,
        seq_drop_prob: float = 0.30,
        smi_drop_prob: float = 0.12,
    ):
        super().__init__()
        self.active_eps = list(active_eps)
        self.d_model = d_model
        self.seq_drop_prob = seq_drop_prob
        self.smi_drop_prob = smi_drop_prob
        self.register_buffer("pk_core_idx", torch.tensor(PK_CORE_INDICES, dtype=torch.long))

        # 1. Explicit Modification + PK-Core RBF-KAN branch
        self.mod_kan = nn.Sequential(
            RBFBasisKANLayer(len(PK_CORE_INDICES), d_model, n_knots=8, dropout=dropout * 0.6),
            ResidualSwiGLU(d_model, hidden_dim=d_model, dropout=dropout * 0.6),
        )
        self.phys_chem_enc = nn.Sequential(
            nn.Linear(48, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
        )
        self.phys_seq_enc = nn.Sequential(
            nn.Linear(35, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout * 0.6),
        )

        # 2. Chemical LMs
        self.ncaa_smi_enc = EmbeddingProjector(768, d_model, in_drop=0.15, out_drop=dropout * 0.6)
        self.ncaa_fus_enc = EmbeddingProjector(2048, d_model, in_drop=0.25, out_drop=dropout * 0.7)
        self.ncaa_prot_enc = EmbeddingProjector(1280, d_model, in_drop=0.20, out_drop=dropout * 0.7)
        self.pclm2_enc = EmbeddingProjector(512, d_model, in_drop=0.15, out_drop=dropout * 0.6)
        self.cb_enc = EmbeddingProjector(384, d_model, in_drop=0.15, out_drop=dropout * 0.6)

        # 3. ESM-C 600M 4-Region Self-Attention + Fused Chem-to-Sequence Cross-Attention
        self.esmc_branch = MultiRegionESMCBranch(in_dim=1152, d_model=d_model, n_heads=4, dropout=dropout * 0.8)
        self.cross_heads = 4
        self.cross_head_dim = d_model // 4
        self.cross_drop = dropout * 0.8
        self.chem_query_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, d_model * 2)
        self.cross_out_proj = nn.Linear(d_model, d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        # Modification-aware gate on canonical sequence branch (lets lipidated/cyclic peptides
        # trust chemical modification state while unmodified peptides leverage ESM-C!)
        self.mod_seq_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.mod_seq_gate.weight)
        nn.init.constant_(self.mod_seq_gate.bias, 0.5)

        # Learned Modality Gate over 9 tokens: [mod_kan, phys_chem, ncaa_smi, ncaa_fus, ncaa_prot, pclm2, cb, phys_seq, esmc_cross]
        self.n_toks = 9
        self.mod_pos = nn.Parameter(torch.randn(1, self.n_toks, d_model) * 0.02)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.mol_fusion = nn.Sequential(
            nn.Linear(d_model * 3 + 2, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualSwiGLU(384, hidden_dim=512, dropout=dropout),
            ResidualSwiGLU(384, hidden_dim=512, dropout=dropout),
        )

        # Explicit PepLife I5a-v2 + Residual FiLM Heads per active PK/half-life endpoint
        self.heads = nn.ModuleDict()
        for ep in self.active_eps:
            self.heads[safe_head_key(ep)] = ConditionInjectedTaskHead(
                in_dim=384,
                cond_dim=cond_dims.get(ep, 0),
                task_type=ENDPOINT_SPECS[ep]["task_type"],
                head_hidden=128,
                dropout=dropout * 0.8,
            )

    def get_head(self, ep: str) -> ConditionInjectedTaskHead:
        return self.heads[safe_head_key(ep)]

    def encode_molecule(self, batch: Dict[str, torch.Tensor], apply_modality_dropout: bool = True) -> torch.Tensor:
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

        pk_feats = batch["phys_chem"].index_select(dim=1, index=self.pk_core_idx)
        t_mod = self.mod_kan(pk_feats)
        t_pc = self.phys_chem_enc(batch["phys_chem"])
        t_ns = self.ncaa_smi_enc(batch["ncaa_smi"])
        t_nf = self.ncaa_fus_enc(batch["ncaa_fus"])
        t_np = self.ncaa_prot_enc(batch["ncaa_prot"])
        t_p2 = self.pclm2_enc(batch["pclm2"])
        t_cb = self.cb_enc(batch["chemberta"])
        t_ps = self.phys_seq_enc(batch["phys_seq"])

        esmc_pool, esmc_4toks = self.esmc_branch(batch["esmc_4reg"])
        # Chem/Mod-conditioned fused cross-attention onto [mean, max, nterm5, cterm5]
        # When smi_m is False, query is 0 -> falls back cleanly to esmc_pool!
        chem_q = self.chem_query_proj(torch.cat([t_mod, t_ns], dim=-1)) * smi_m.unsqueeze(-1).to(t_mod.dtype)
        q = self.cross_q(chem_q).view(bsz, 1, self.cross_heads, self.cross_head_dim).transpose(1, 2)
        kv = self.cross_kv(esmc_4toks).view(bsz, 4, 2, self.cross_heads, self.cross_head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cross_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.cross_drop if self.training else 0.0
        )
        cross_out = self.cross_out_proj(cross_out.transpose(1, 2).reshape(bsz, self.d_model))
        t_esmc = self.cross_norm(esmc_pool + 0.5 * cross_out * smi_m.unsqueeze(-1).to(esmc_pool.dtype))

        tokens = torch.stack([t_mod, t_pc, t_ns, t_nf, t_np, t_p2, t_cb, t_ps, t_esmc], dim=1)  # (B, 9, D)
        tok_mask = torch.stack([smi_m, smi_m, smi_m, smi_m, smi_m, smi_m, smi_m, seq_m, seq_m], dim=1)

        tokens = tokens * tok_mask.unsqueeze(-1).to(tokens.dtype)
        logits = self.gate(tokens + self.mod_pos).squeeze(-1).masked_fill(~tok_mask, -1e4)
        w = torch.softmax(logits, dim=-1) * tok_mask.to(logits.dtype)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_attn = (tokens * w.unsqueeze(-1)).sum(dim=1)

        chem_m = tok_mask[:, :7].to(tokens.dtype)
        seq_m2 = tok_mask[:, 7:].to(tokens.dtype)
        z_chem = (tokens[:, :7] * chem_m.unsqueeze(-1)).sum(dim=1) / chem_m.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_seq = (tokens[:, 7:] * seq_m2.unsqueeze(-1)).sum(dim=1) / seq_m2.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Modification-aware sequence gate (active when both modalities exist; exact 0 when seq_m=0)
        seq_mod_g = torch.where(
            smi_m.unsqueeze(-1),
            torch.sigmoid(self.mod_seq_gate(t_mod)).to(tokens.dtype),
            torch.ones((bsz, 1), device=device, dtype=tokens.dtype),
        )
        z_chem = z_chem * smi_m.unsqueeze(-1).to(tokens.dtype)
        z_seq = z_seq * (seq_m.unsqueeze(-1).to(tokens.dtype) * seq_mod_g)

        flags = torch.stack([smi_m.to(tokens.dtype), seq_m.to(tokens.dtype)], dim=-1)
        return self.mol_fusion(torch.cat([z_attn, z_chem, z_seq, flags], dim=-1))

    def forward_task(
        self,
        batch: Dict[str, torch.Tensor],
        ep: str,
        cond_vec: Optional[torch.Tensor],
        apply_modality_dropout: bool = True,
    ) -> torch.Tensor:
        h = self.encode_molecule(batch, apply_modality_dropout=apply_modality_dropout)
        return self.get_head(ep)(h, cond_vec)


# =============================================================================
# Unit Self-Checks (SMILES-Only Mathematical Invariance & Tobit Stability)
# =============================================================================

def verify_model_invariants(device: torch.device) -> None:
    """Verify exact mathematical zero-sequence-leakage in smiles_only mode & Tobit stability."""
    torch.manual_seed(0)
    cond_dims = {ep: 4 for ep in ALL_ENDPOINTS}
    m1 = MultiTaskPeptideNet(cond_dims, ALL_ENDPOINTS).to(device).eval()
    m2 = HalfLifeAttentionNet(cond_dims, ["half_life_reg", "half_life_cls", "PPB"]).to(device).eval()

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
        d1 = (m1.forward_task(base, "half_life_reg", cvec, False) - m1.forward_task(perturbed, "half_life_reg", cvec, False)).abs().max().item()
        d2 = (m2.forward_task(base, "half_life_reg", cvec, False) - m2.forward_task(perturbed, "half_life_reg", cvec, False)).abs().max().item()
    assert d1 < 1e-6, f"MultiTaskPeptideNet smiles_only sequence leak: {d1}"
    assert d2 < 1e-6, f"HalfLifeAttentionNet smiles_only sequence leak: {d2}"

    # Verify Tobit loss & gradient finiteness across extreme exact/right/left/interval cases
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
    print(f"[UNIT CHECK PASS] smiles_only invariance: MT={d1:.2e}, HL={d2:.2e} | Tobit extreme loss={loss.item():.4f}, grad finite=True")
