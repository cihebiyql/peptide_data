#!/usr/bin/env python3
"""Portable Pure-PyTorch Molecular E-GAT + Task-Specific WeightAndSum Live Per-Atom Attribution Engine.

100% Pure-PyTorch + RDKit + NumPy — ZERO torch_geometric, ZERO dgl, ZERO external runtime deps!
Designed to load and run natively on `node2`'s torch 2.6.0 CPU in < 200 ms per peptide.

Public API:
  - GNN_8_ENDPOINTS: the 8 flagship web endpoints
  - mol_to_graph_tensors(mol) -> dict of atom/edge tensors
  - PurePyTorchEGATNet: Edge-Conditioned Multi-Head GATv2 + Task-Specific WeightAndSum Readout
  - GNNAtomAttributionEngine(ckpt_path, device='cpu'):
      * predict_and_attribute(smiles_or_seq, is_sequence=False, endpoints=None)
        -> {
            "smiles": ..., "n_atoms": int, "latency_ms": float,
            "predictions": dict[str, dict],
            "atom_weights": dict[str, list[float]],  # strictly in [0, 1], length == n_atoms
            "atom_symbols": list[str],
          }
  - render_atom_attribution_svg(mol, weights, endpoint_name, width=860, height=480, ...) -> str (valid SVG)
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

GNN_8_ENDPOINTS: List[str] = [
    "PAMPA",
    "Caco2",
    "hemolysis_cls",
    "hemolysis_HC50",
    "overall_peptide_toxicity",
    "cytotoxicity_cls",
    "cytotoxicity_CC50",
    "half_life_reg",
]

GNN_8_TASK_META: Dict[str, Dict[str, Any]] = {
    "PAMPA": {"task_type": "regression", "unit": "log10(cm/s)", "bounds": None, "label": "PAMPA Permeability"},
    "Caco2": {"task_type": "regression", "unit": "log10(cm/s)", "bounds": None, "label": "Caco-2 Permeability"},
    "hemolysis_cls": {"task_type": "classification", "unit": "prob [0,1]", "bounds": (0.0, 1.0), "label": "Hemolysis Risk"},
    "hemolysis_HC50": {"task_type": "regression", "unit": "log10(uM)", "bounds": None, "label": "Hemolysis HC50"},
    "overall_peptide_toxicity": {"task_type": "classification", "unit": "prob [0,1]", "bounds": (0.0, 1.0), "label": "Overall Peptide Toxicity"},
    "cytotoxicity_cls": {"task_type": "classification", "unit": "prob [0,1]", "bounds": (0.0, 1.0), "label": "Cytotoxicity Risk"},
    "cytotoxicity_CC50": {"task_type": "regression", "unit": "log10(uM)", "bounds": (-4.0, 5.5), "label": "Cytotoxicity CC50"},
    "half_life_reg": {"task_type": "regression", "unit": "log10(h)", "bounds": None, "label": "Plasma Half-Life (t1/2)"},
}

ATOM_FEAT_DIM = 50
EDGE_FEAT_DIM = 14
GLOBAL_CTX_DIM = 6

_ATOMIC_NUMS = [6, 7, 8, 16, 9, 17, 35, 53, 15, 5]
_HYB_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
]

DEFAULT_CKPT_PATH = Path(__file__).resolve().parent / "gnn_atom_attr" / "checkpoints" / "gnn_8ep_5fold_fp16.pt"


# =============================================================================
# 1. Pure RDKit Peptide-Aware Molecular Graph Featurizer (NO torch_geometric!)
# =============================================================================

def parse_peptide_input(smiles_or_seq: str, is_sequence: bool = False) -> Chem.Mol:
    """Parse SMILES or unmodified 1-letter amino acid sequence via RDKit MolFromSequence."""
    s = (smiles_or_seq or "").strip()
    if not s:
        raise ValueError("Empty input string")
    if is_sequence:
        clean = "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        if not clean:
            raise ValueError(f"No valid canonical amino acids found in sequence: {s!r}")
        mol = Chem.MolFromSequence(clean)
        if mol is None:
            raise ValueError(f"RDKit MolFromSequence failed for sequence length {len(clean)}")
        return mol
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return mol
    # Automatic fallback: if pure 1-letter canonical peptide sequence
    clean = "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(clean) == len(s) and len(clean) >= 2:
        mol = Chem.MolFromSequence(clean)
        if mol is not None:
            return mol
    raise ValueError(f"Could not parse peptide SMILES/sequence: {s[:80]!r}")


def mol_to_graph_numpy(mol: Chem.Mol) -> Dict[str, np.ndarray]:
    """Convert RDKit Mol to compact pure-NumPy graph arrays (50D atom feats, 14D edge feats, 6D global ctx)."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        raise ValueError("Molecule has 0 atoms")

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    atom_ring_sizes: List[List[int]] = [[] for _ in range(n_atoms)]
    for ring in atom_rings:
        rlen = len(ring)
        for aidx in ring:
            atom_ring_sizes[aidx].append(rlen)

    # Identify carbonyl carbons/oxygens and peptide amide bonds quickly without SMARTS overhead
    is_carbonyl_c = [False] * n_atoms
    is_carbonyl_o = [False] * n_atoms
    is_amide_n = [False] * n_atoms
    amide_bond_set = set()

    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        zi, zj = ai.GetAtomicNum(), aj.GetAtomicNum()
        if bt == Chem.rdchem.BondType.DOUBLE:
            if zi == 6 and zj == 8:
                is_carbonyl_c[i] = True
                is_carbonyl_o[j] = True
            elif zi == 8 and zj == 6:
                is_carbonyl_c[j] = True
                is_carbonyl_o[i] = True

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
            if is_carbonyl_c[i] and aj.GetAtomicNum() == 7:
                is_amide_n[j] = True
                amide_bond_set.add((i, j))
                amide_bond_set.add((j, i))
            elif is_carbonyl_c[j] and ai.GetAtomicNum() == 7:
                is_amide_n[i] = True
                amide_bond_set.add((i, j))
                amide_bond_set.add((j, i))

    x = np.zeros((n_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    n_ring_atoms = 0
    n_hetero = 0
    n_macro_atoms = 0
    n_amide_n = 0
    n_hbd_atoms = 0

    for idx, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        # 0..10: atomic num one-hot (11)
        try:
            x[idx, _ATOMIC_NUMS.index(z)] = 1.0
        except ValueError:
            x[idx, 10] = 1.0
        if z != 6 and z != 1:
            n_hetero += 1

        # 11..16: degree (6)
        deg = min(atom.GetTotalDegree(), 5)
        x[idx, 11 + deg] = 1.0

        # 17..22: formal charge (6)
        fc = atom.GetFormalCharge()
        if -2 <= fc <= 2:
            x[idx, 17 + (fc + 2)] = 1.0
        else:
            x[idx, 22] = 1.0

        # 23..27: total Hs (5)
        nhs = min(atom.GetTotalNumHs(), 4)
        x[idx, 23 + nhs] = 1.0

        # 28..33: hybridization (6)
        hyb = atom.GetHybridization()
        try:
            x[idx, 28 + _HYB_TYPES.index(hyb)] = 1.0
        except ValueError:
            x[idx, 33] = 1.0

        # 34..35: aromatic, in_ring (2)
        if atom.GetIsAromatic():
            x[idx, 34] = 1.0
        in_ring = atom.IsInRing()
        if in_ring:
            x[idx, 35] = 1.0
            n_ring_atoms += 1

        # 36..41: ring sizes 3,4,5,6,7,>=8 macrocycle (6)
        rsizes = atom_ring_sizes[idx]
        if rsizes:
            has_macro = False
            for rs in rsizes:
                if rs == 3:
                    x[idx, 36] = 1.0
                elif rs == 4:
                    x[idx, 37] = 1.0
                elif rs == 5:
                    x[idx, 38] = 1.0
                elif rs == 6:
                    x[idx, 39] = 1.0
                elif rs == 7:
                    x[idx, 40] = 1.0
                elif rs >= 8:
                    x[idx, 41] = 1.0
                    has_macro = True
            if has_macro:
                n_macro_atoms += 1

        # 42..43: chirality CW/CCW (2)
        ctag = atom.GetChiralTag()
        if ctag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
            x[idx, 42] = 1.0
        elif ctag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
            x[idx, 43] = 1.0

        # 44..49: peptide pharmacophore & mass flags (6)
        if z in (7, 8, 16) and atom.GetTotalNumHs() > 0:
            x[idx, 44] = 1.0  # HBD
            n_hbd_atoms += 1
        if z in (7, 8):
            x[idx, 45] = 1.0  # HBA
        if is_amide_n[idx]:
            x[idx, 46] = 1.0
            n_amide_n += 1
        if is_carbonyl_c[idx] or is_carbonyl_o[idx]:
            x[idx, 47] = 1.0
        # Basic / acidic ionizable flag
        is_ion = False
        if z == 7 and not is_amide_n[idx] and (fc > 0 or not atom.GetIsAromatic()):
            is_ion = True
        elif z == 8 and (fc < 0 or (atom.GetTotalNumHs() > 0 and any(is_carbonyl_c[nb.GetIdx()] for nb in atom.GetNeighbors()))):
            is_ion = True
        if is_ion:
            x[idx, 48] = 1.0
        x[idx, 49] = min(atom.GetMass() / 32.0, 4.0)

    # Directed edges + self-loops
    n_bonds = mol.GetNumBonds()
    n_edges = n_bonds * 2 + n_atoms
    edge_src = np.empty(n_edges, dtype=np.int64)
    edge_dst = np.empty(n_edges, dtype=np.int64)
    edge_attr = np.zeros((n_edges, EDGE_FEAT_DIM), dtype=np.float32)

    bond_rings = ring_info.BondRings()
    bond_macro = [False] * n_bonds
    for ring in bond_rings:
        if len(ring) >= 8:
            for bidx in ring:
                bond_macro[bidx] = True

    e_pos = 0
    for bidx, bond in enumerate(mol.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
        bt = bond.GetBondType()
        try:
            feat[_BOND_TYPES.index(bt)] = 1.0
        except ValueError:
            feat[0] = 1.0
        if bond.GetIsConjugated():
            feat[4] = 1.0
        if bond.IsInRing():
            feat[5] = 1.0
        if bond_macro[bidx]:
            feat[6] = 1.0
        st = bond.GetStereo()
        try:
            feat[7 + _BOND_STEREOS.index(st)] = 1.0
        except ValueError:
            feat[7] = 1.0
        if (i, j) in amide_bond_set:
            feat[11] = 1.0
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 16 and mol.GetAtomWithIdx(j).GetAtomicNum() == 16:
            feat[12] = 1.0

        edge_src[e_pos] = i
        edge_dst[e_pos] = j
        edge_attr[e_pos] = feat
        e_pos += 1
        edge_src[e_pos] = j
        edge_dst[e_pos] = i
        edge_attr[e_pos] = feat
        e_pos += 1

    # Self-loops guarantee every node has >= 1 incoming edge for scatter_softmax!
    for idx in range(n_atoms):
        edge_src[e_pos] = idx
        edge_dst[e_pos] = idx
        edge_attr[e_pos, 13] = 1.0
        e_pos += 1

    inv_n = 1.0 / float(max(1, n_atoms))
    global_ctx = np.array(
        [
            math.log1p(n_atoms) / 5.5,
            n_ring_atoms * inv_n,
            n_hetero * inv_n,
            n_macro_atoms * inv_n,
            n_amide_n * inv_n,
            n_hbd_atoms * inv_n,
        ],
        dtype=np.float32,
    )
    return {
        "x": x,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_attr": edge_attr,
        "global_ctx": global_ctx,
        "n_atoms": n_atoms,
    }


def collate_graphs(graphs: List[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Collate list of numpy graph dicts into a single disjoint-union PyTorch batch."""
    xs, srcs, dsts, eattrs, bidxs, gctxs = [], [], [], [], [], []
    offset = 0
    for g_i, g in enumerate(graphs):
        na = g["x"].shape[0]
        xs.append(g["x"])
        srcs.append(g["edge_src"] + offset)
        dsts.append(g["edge_dst"] + offset)
        eattrs.append(g["edge_attr"])
        bidxs.append(np.full(na, g_i, dtype=np.int64))
        gctxs.append(g["global_ctx"])
        offset += na
    return {
        "x": torch.from_numpy(np.concatenate(xs, axis=0)).to(device),
        "edge_src": torch.from_numpy(np.concatenate(srcs, axis=0)).to(device),
        "edge_dst": torch.from_numpy(np.concatenate(dsts, axis=0)).to(device),
        "edge_attr": torch.from_numpy(np.concatenate(eattrs, axis=0)).to(device),
        "batch_idx": torch.from_numpy(np.concatenate(bidxs, axis=0)).to(device),
        "global_ctx": torch.from_numpy(np.stack(gctxs, axis=0)).to(device),
        "num_graphs": len(graphs),
    }


# =============================================================================
# 2. Pure-PyTorch Edge-Conditioned Multi-Head GATv2 + WeightAndSum Architecture
# =============================================================================

def _scatter_softmax_2d(logits: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Numerically stable pure-PyTorch grouped softmax over edges grouped by destination node (fp32)."""
    orig_dtype = logits.dtype
    lf = logits.float()
    H = lf.shape[1]
    max_per_node = torch.full((num_nodes, H), -1e9, device=lf.device, dtype=torch.float32)
    max_per_node.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, H), lf, reduce="amax", include_self=False)
    shifted = (lf - max_per_node[dst]).clamp(-20.0, 0.0)
    exp_l = torch.exp(shifted)
    denom = torch.zeros((num_nodes, H), device=lf.device, dtype=torch.float32)
    denom.index_add_(0, dst, exp_l)
    return (exp_l / denom[dst].clamp_min(1e-8)).to(orig_dtype)


def _scatter_softmax_1d(logits: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Pure-PyTorch grouped softmax over atoms grouped by molecule (fp32)."""
    orig_dtype = logits.dtype
    lf = logits.float()
    max_g = torch.full((num_graphs,), -1e9, device=lf.device, dtype=torch.float32)
    max_g.scatter_reduce_(0, batch_idx, lf, reduce="amax", include_self=False)
    exp_l = torch.exp((lf - max_g[batch_idx]).clamp(-20.0, 0.0))
    denom = torch.zeros((num_graphs,), device=lf.device, dtype=torch.float32)
    denom.index_add_(0, batch_idx, exp_l)
    return (exp_l / denom[batch_idx].clamp_min(1e-8)).to(orig_dtype)


def _scatter_sum_2d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros((dim_size, src.shape[1]), device=src.device, dtype=torch.float32)
    out.index_add_(0, index, src.float())
    return out.to(src.dtype)


def _scatter_max_2d(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    sf = src.float()
    out = torch.full((dim_size, sf.shape[1]), -1e4, device=sf.device, dtype=torch.float32)
    out.scatter_reduce_(0, index.unsqueeze(1).expand(-1, sf.shape[1]), sf, reduce="amax", include_self=False)
    return out.to(src.dtype)


class EGATv2Layer(nn.Module):
    """Pure-PyTorch Pre-LN Residual Edge-Conditioned Multi-Head GATv2 Layer + SwiGLU FFN."""

    def __init__(self, d_node: int, d_edge: int, n_heads: int = 4, dropout: float = 0.15, update_edge: bool = True):
        super().__init__()
        assert d_node % n_heads == 0
        self.d_node = d_node
        self.n_heads = n_heads
        self.d_head = d_node // n_heads
        self.update_edge = update_edge

        self.norm_n = nn.LayerNorm(d_node)
        self.norm_e = nn.LayerNorm(d_edge)

        self.w_src = nn.Linear(d_node, d_node, bias=False)
        self.w_dst = nn.Linear(d_node, d_node, bias=False)
        self.w_edge_k = nn.Linear(d_edge, d_node, bias=False)
        self.w_val = nn.Linear(d_node, d_node, bias=False)
        self.w_edge_v = nn.Linear(d_edge, d_node, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(1, n_heads, self.d_head) * 0.1)
        self.w_out = nn.Linear(d_node, d_node)
        self.drop = nn.Dropout(dropout)

        # Edge residual update (omitted on final layer so 100% of params receive gradients)
        if self.update_edge:
            self.edge_upd = nn.Sequential(
                nn.Linear(d_node * 2 + d_edge, d_edge),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_edge, d_edge),
            )
        else:
            self.edge_upd = None

        # Node SwiGLU FFN
        self.ffn_norm = nn.LayerNorm(d_node)
        h_ffn = int(d_node * 1.5)
        self.ffn_gv = nn.Linear(d_node, h_ffn * 2)
        self.ffn_out = nn.Linear(h_ffn, d_node)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = h.shape[0]
        hn = self.norm_n(h)
        en = self.norm_e(e)

        q_src = self.w_src(hn).view(N, self.n_heads, self.d_head)
        k_dst = self.w_dst(hn).view(N, self.n_heads, self.d_head)
        ek = self.w_edge_k(en).view(-1, self.n_heads, self.d_head)

        attn_h = F.leaky_relu(q_src[edge_src] + k_dst[edge_dst] + ek, negative_slope=0.2)
        logits = (attn_h * self.attn_vec).sum(dim=-1)  # (E, H)
        alpha = _scatter_softmax_2d(logits, edge_dst, N)
        alpha = self.drop(alpha)

        v_src = self.w_val(hn).view(N, self.n_heads, self.d_head)
        ev = self.w_edge_v(en).view(-1, self.n_heads, self.d_head)
        msg = (v_src[edge_src] + ev) * alpha.unsqueeze(-1)  # (E, H, Dh)

        agg = torch.zeros((N, self.n_heads, self.d_head), device=h.device, dtype=torch.float32)
        agg.index_add_(0, edge_dst, msg.float())
        h = h + self.drop(self.w_out(agg.to(h.dtype).reshape(N, self.d_node)))

        # SwiGLU FFN residual
        hf = self.ffn_norm(h)
        u, v = self.ffn_gv(hf).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_out(self.drop(F.silu(u) * v)))

        # Edge update residual
        if self.edge_upd is not None:
            e_in = torch.cat([hn[edge_src], hn[edge_dst], en], dim=-1)
            e = e + self.drop(self.edge_upd(e_in))
        return h, e


class TaskWeightAndSumHead(nn.Module):
    """Task-specific WeightAndSum + Softmax Attention Readout with exact additive atom importance."""

    def __init__(self, d_node: int, task_type: str, dropout: float = 0.20):
        super().__init__()
        self.task_type = task_type
        out_dim = 2 if task_type == "regression" else 1

        # Task-specific per-atom gate MLP (WeightAndSum)
        self.atom_gate = nn.Sequential(
            nn.Linear(d_node, d_node // 2),
            nn.GELU(),
            nn.Linear(d_node // 2, 1),
        )
        # Direct additive per-atom contribution head (interpretable local pharmacophore score!)
        self.atom_contrib = nn.Linear(d_node, 1)

        # Graph-level prediction head over [WeightAndSum, SoftmaxPool, MaxPool, global_ctx]
        self.graph_mlp = nn.Sequential(
            nn.LayerNorm(d_node * 3 + GLOBAL_CTX_DIM),
            nn.Dropout(dropout),
            nn.Linear(d_node * 3 + GLOBAL_CTX_DIM, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_node, out_dim),
        )

    def forward(
        self,
        h_atom: torch.Tensor,
        batch_idx: torch.Tensor,
        global_ctx: torch.Tensor,
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pred, atom_gate_sigmoid, atom_local_contrib)."""
        gate_logits = self.atom_gate(h_atom).squeeze(-1)  # (N_atoms,)
        atom_sig = torch.sigmoid(gate_logits)  # WeightAndSum gate in (0, 1)
        atom_soft = _scatter_softmax_1d(gate_logits, batch_idx, num_graphs)

        # Weighted sum pools
        ws_num = _scatter_sum_2d(h_atom * atom_sig.unsqueeze(-1), batch_idx, num_graphs)
        ws_den = torch.zeros((num_graphs,), device=h_atom.device, dtype=atom_sig.dtype)
        ws_den.index_add_(0, batch_idx, atom_sig)
        g_ws = ws_num / ws_den.clamp_min(1e-6).unsqueeze(-1)

        g_att = _scatter_sum_2d(h_atom * atom_soft.unsqueeze(-1), batch_idx, num_graphs)
        g_max = _scatter_max_2d(h_atom, batch_idx, num_graphs)

        # Direct additive WeightAndSum scalar residual (grounds attribution directly in prediction!)
        local_c = self.atom_contrib(h_atom).squeeze(-1)  # (N_atoms,)
        add_num = torch.zeros((num_graphs,), device=h_atom.device, dtype=local_c.dtype)
        add_num.index_add_(0, batch_idx, local_c * atom_sig)
        y_add = (add_num / ws_den.clamp_min(1e-6)).unsqueeze(-1)  # (B, 1)

        g_feat = torch.cat([g_ws, g_att, g_max, global_ctx.to(g_ws.dtype)], dim=-1)
        raw = self.graph_mlp(g_feat)
        if self.task_type == "regression":
            pred = torch.cat([raw[:, 0:1] + 0.35 * y_add, raw[:, 1:2]], dim=-1)
        else:
            pred = raw + 0.35 * y_add
        return pred, atom_sig, local_c


class PurePyTorchEGATNet(nn.Module):
    """100% Pure-PyTorch 4-layer Multi-Head E-GATv2 + Multi-Scale JK + Task WeightAndSum Network."""

    def __init__(
        self,
        endpoints: Sequence[str] = GNN_8_ENDPOINTS,
        d_node: int = 160,
        d_edge: int = 48,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.18,
    ):
        super().__init__()
        self.endpoints = list(endpoints)
        self.d_node = d_node
        self.d_edge = d_edge
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.atom_in = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
        )
        self.edge_in = nn.Sequential(
            nn.Linear(EDGE_FEAT_DIM, d_edge),
            nn.LayerNorm(d_edge),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                EGATv2Layer(d_node=d_node, d_edge=d_edge, n_heads=n_heads, dropout=dropout, update_edge=(li < n_layers - 1))
                for li in range(n_layers)
            ]
        )
        self.jk_proj = nn.Sequential(
            nn.Linear(d_node * 2, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict()
        for ep in self.endpoints:
            self.heads[ep] = TaskWeightAndSumHead(
                d_node=d_node,
                task_type=GNN_8_TASK_META[ep]["task_type"],
                dropout=dropout,
            )

    def encode_atoms(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.atom_in(batch["x"])
        e = self.edge_in(batch["edge_attr"])
        src, dst = batch["edge_src"], batch["edge_dst"]
        h_mid = h
        for li, layer in enumerate(self.layers):
            h, e = layer(h, e, src, dst)
            if li == (self.n_layers // 2) - 1:
                h_mid = h
        return self.jk_proj(torch.cat([h_mid, h], dim=-1))

    def forward_task(
        self,
        batch: Dict[str, torch.Tensor],
        ep: str,
        h_atom: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_atom is None:
            h_atom = self.encode_atoms(batch)
        return self.heads[ep](h_atom, batch["batch_idx"], batch["global_ctx"], int(batch["num_graphs"]))

    def forward_all(
        self,
        batch: Dict[str, torch.Tensor],
        endpoints: Optional[Sequence[str]] = None,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_atom = self.encode_atoms(batch)
        eps = list(endpoints) if endpoints is not None else self.endpoints
        return {ep: self.heads[ep](h_atom, batch["batch_idx"], batch["global_ctx"], int(batch["num_graphs"])) for ep in eps}


def _normalize_atom_scores(sig: np.ndarray, local_c: np.ndarray) -> np.ndarray:
    """Signed, magnitude-calibrated per-atom contribution score in [0.0, 1.0] centered at 0.50 (neutral).

    Preserves the sign of the grounded additive WeightAndSum contribution `sig * local_c`:
      - < 0.50 (blue): negative / suppressing contribution
      - ~ 0.50 (neutral): weak / negligible contribution (scale floor 0.35 prevents false crimson hotspots on low-signal peptides)
      - > 0.50 (amber -> crimson): positive / enhancing contribution
    """
    sig_a = np.asarray(sig, dtype=np.float64)
    lc_a = np.asarray(local_c, dtype=np.float64)
    if len(sig_a) == 0:
        return np.array([], dtype=np.float64)
    signed = sig_a * lc_a
    # Robust magnitude scale with a 0.35 floor so flat/low-signal peptides remain muted near neutral 0.50
    p95 = float(np.percentile(np.abs(signed), 95)) if len(signed) > 1 else float(abs(signed[0]))
    scale = max(p95, 0.35)
    norm_signed = np.clip(signed / scale, -1.0, 1.0)
    return np.clip(0.5 + 0.5 * norm_signed, 0.0, 1.0)


# =============================================================================
# 3. Battle-Tested CPU Inference & Attribution Engine (< 200 ms on node2 CPU!)
# =============================================================================

class GNNAtomAttributionEngine:
    """Loads compact fp16 5-fold PurePyTorchEGATNet ensemble and runs deterministic CPU attribution."""

    def __init__(self, ckpt_path: Union[str, Path] = DEFAULT_CKPT_PATH, device: str = "cpu"):
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device)
        ckpt = torch.load(str(self.ckpt_path), map_location=self.device, weights_only=False)
        self.cfg: Dict[str, Any] = ckpt["model_config"]
        self.endpoints: List[str] = list(ckpt["endpoints"])
        self.y_scalers_per_fold: List[Dict[str, Tuple[float, float]]] = ckpt["y_scalers_per_fold"]

        self.models: List[PurePyTorchEGATNet] = []
        for sd_fp16 in ckpt["fold_state_dicts_fp16"]:
            m = PurePyTorchEGATNet(
                endpoints=self.endpoints,
                d_node=int(self.cfg["d_node"]),
                d_edge=int(self.cfg["d_edge"]),
                n_layers=int(self.cfg["n_layers"]),
                n_heads=int(self.cfg["n_heads"]),
                dropout=0.0,
            )
            sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd_fp16.items()}
            m.load_state_dict(sd)
            m.to(self.device).eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_and_attribute(
        self,
        smiles_or_seq: str,
        is_sequence: bool = False,
        endpoints: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run 5-fold pure-PyTorch E-GAT inference & return per-endpoint predictions + [0,1] atom weights."""
        t0 = time.perf_counter()
        mol = parse_peptide_input(smiles_or_seq, is_sequence=is_sequence)
        g = mol_to_graph_numpy(mol)
        batch = collate_graphs([g], self.device)
        target_eps = list(endpoints) if endpoints is not None else self.endpoints

        preds_accum: Dict[str, List[float]] = {ep: [] for ep in target_eps}
        sigmas_accum: Dict[str, List[float]] = {ep: [] for ep in target_eps}
        weights_accum: Dict[str, List[np.ndarray]] = {ep: [] for ep in target_eps}

        for f_idx, model in enumerate(self.models):
            y_scalers = self.y_scalers_per_fold[f_idx]
            outs = model.forward_all(batch, target_eps)
            for ep in target_eps:
                raw, atom_sig, local_c = outs[ep]
                meta = GNN_8_TASK_META[ep]
                if meta["task_type"] == "regression":
                    ym, ys = y_scalers[ep]
                    mu_val = float(raw[0, 0].item()) * ys + ym
                    sig_val = float(F.softplus(raw[0, 1] + 0.5413).clamp(0.10, 4.0).item()) * ys
                    if meta["bounds"] is not None:
                        mu_val = float(np.clip(mu_val, meta["bounds"][0], meta["bounds"][1]))
                    preds_accum[ep].append(mu_val)
                    sigmas_accum[ep].append(sig_val)
                else:
                    prob = float(torch.sigmoid(raw[0, 0]).clamp(1e-6, 1.0 - 1e-6).item())
                    preds_accum[ep].append(prob)

                w_norm = _normalize_atom_scores(atom_sig.cpu().numpy(), local_c.cpu().numpy())
                weights_accum[ep].append(w_norm)

        predictions: Dict[str, Dict[str, Any]] = {}
        atom_weights: Dict[str, List[float]] = {}
        for ep in target_eps:
            meta = GNN_8_TASK_META[ep]
            p_arr = np.asarray(preds_accum[ep], dtype=np.float64)
            # Keep signed neutral-centered [0, 1] ensemble mean WITHOUT forcible min-max rescaling
            w_mean = np.clip(np.mean(weights_accum[ep], axis=0), 0.0, 1.0)
            atom_weights[ep] = [round(float(v), 4) for v in w_mean]
            rec: Dict[str, Any] = {
                "value": round(float(p_arr.mean()), 5),
                "ensemble_sd": round(float(p_arr.std()), 5),
                "unit": meta["unit"],
                "task_type": meta["task_type"],
            }
            if meta["task_type"] == "regression":
                rec["aleatoric_sigma"] = round(float(np.mean(sigmas_accum[ep])), 5)
            predictions[ep] = rec

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "n_atoms": int(mol.GetNumAtoms()),
            "atom_symbols": [a.GetSymbol() for a in mol.GetAtoms()],
            "latency_ms": round(dt_ms, 2),
            "predictions": predictions,
            "atom_weights": atom_weights,
        }


# =============================================================================
# 4. Publication-Grade RDKit rdMolDraw2D SVG Attribution Heatmap Renderer
# =============================================================================

def _palette_cool_to_warm(t: float) -> Tuple[float, float, float]:
    """Perceptual diverging/sequential palette:
    0.00 -> cool muted blue-teal (0.20, 0.56, 0.72)
    0.45 -> soft neutral cream/ice (0.94, 0.96, 0.97)
    0.75 -> warm amber-gold (0.96, 0.65, 0.24)
    1.00 -> rich coral-crimson (0.86, 0.24, 0.20)
    """
    t = float(max(0.0, min(1.0, t)))
    stops = [
        (0.00, (0.20, 0.56, 0.72)),
        (0.35, (0.58, 0.80, 0.86)),
        (0.50, (0.95, 0.96, 0.96)),
        (0.75, (0.97, 0.66, 0.24)),
        (1.00, (0.86, 0.24, 0.20)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = (t - t0) / max(1e-8, t1 - t0)
            return (
                c0[0] + u * (c1[0] - c0[0]),
                c0[1] + u * (c1[1] - c0[1]),
                c0[2] + u * (c1[2] - c0[2]),
            )
    return stops[-1][1]


def _xml_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_atom_attribution_svg(
    mol: Union[Chem.Mol, str],
    weights: Sequence[float],
    endpoint_name: str,
    width: int = 860,
    height: int = 500,
    title: Optional[str] = None,
    prediction_text: Optional[str] = None,
) -> str:
    """Render a crisp, publication-grade RDKit rdMolDraw2D SVG heatmap with perceptual palette & legend.

    Args:
        mol: RDKit Mol or SMILES string.
        weights: Per-atom normalized importance weights in [0, 1] (len == mol.GetNumAtoms()).
        endpoint_name: Endpoint identifier (e.g. 'PAMPA', 'hemolysis_cls', 'half_life_reg').
    Returns:
        Well-formed standalone SVG string.
    """
    if isinstance(mol, str):
        mol_obj = Chem.Mol(parse_peptide_input(mol))
    else:
        mol_obj = Chem.Mol(mol)

    n_atoms = mol_obj.GetNumAtoms()
    w_arr = np.asarray(list(weights), dtype=np.float64)
    if len(w_arr) != n_atoms:
        raise ValueError(f"weights length ({len(w_arr)}) != mol.GetNumAtoms() ({n_atoms})")
    if not np.all(np.isfinite(w_arr)):
        raise ValueError("weights contain non-finite values")
    w_arr = np.clip(w_arr, 0.0, 1.0)

    # Generate crisp 2D coordinates (CoordGen preferred for <= 160 atoms, fast RDKit fallback for giant peptides)
    if mol_obj.GetNumConformers() == 0:
        try:
            rdDepictor.SetPreferCoordGen(n_atoms <= 140)
            rdDepictor.Compute2DCoords(mol_obj)
        except Exception:
            rdDepictor.SetPreferCoordGen(False)
            AllChem.Compute2DCoords(mol_obj)

    header_h = 54
    footer_h = 54
    draw_h = max(240, height - header_h - footer_h)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, draw_h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.06
    opts.bondLineWidth = 1.6 if n_atoms <= 120 else 1.2
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.continuousHighlight = True
    opts.highlightBondWidthMultiplier = 10 if n_atoms <= 120 else 7

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    highlight_atoms: List[int] = []
    for i in range(n_atoms):
        wi = float(w_arr[i])
        atom_colors[i] = _palette_cool_to_warm(wi)
        atom_radii[i] = 0.22 + 0.30 * wi
        highlight_atoms.append(i)

    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_bonds: List[int] = []
    for bond in mol_obj.GetBonds():
        bidx = bond.GetIdx()
        wb = 0.5 * (float(w_arr[bond.GetBeginAtomIdx()]) + float(w_arr[bond.GetEndAtomIdx()]))
        bond_colors[bidx] = _palette_cool_to_warm(wb)
        highlight_bonds.append(bidx)

    drawer.DrawMolecule(
        mol_obj,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    raw_svg = drawer.GetDrawingText()

    # Extract inner SVG body cleanly so we can wrap it with a crisp header + gradient legend bar
    start_tag_end = raw_svg.find(">", raw_svg.find("<svg"))
    end_tag_start = raw_svg.rfind("</svg>")
    inner_mol_svg = raw_svg[start_tag_end + 1 : end_tag_start] if (start_tag_end != -1 and end_tag_start != -1) else raw_svg

    ep_meta = GNN_8_TASK_META.get(endpoint_name, {})
    ep_label = ep_meta.get("label", endpoint_name)
    hdr_title = title or f"Per-Atom Attribution — {ep_label} ({endpoint_name})"
    pred_badge = f" | {prediction_text}" if prediction_text else ""
    grad_id = f"attrGrad_{abs(hash(endpoint_name)) % 100000}"

    legend_w = min(360, width - 220)
    legend_x = (width - legend_w) // 2
    legend_y = header_h + draw_h + 14

    svg_doc = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="{grad_id}" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#338FB8"/>
      <stop offset="35%" stop-color="#94CCDB"/>
      <stop offset="50%" stop-color="#F2F5F5"/>
      <stop offset="75%" stop-color="#F7A83D"/>
      <stop offset="100%" stop-color="#DB3D33"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2"/>
  <rect x="0" y="0" width="{width}" height="{header_h}" rx="10" fill="#F8FAFC"/>
  <line x1="0" y1="{header_h}" x2="{width}" y2="{header_h}" stroke="#E2E8F0" stroke-width="1"/>
  <text x="18" y="23" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="14.5" font-weight="600" fill="#0F172A">{_xml_escape(hdr_title)}</text>
  <text x="18" y="42" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11.5" font-weight="500" fill="#475569">2D Graph Surrogate (E-GAT, n={n_atoms}){_xml_escape(pred_badge)}</text>
  <g transform="translate(0, {header_h})">
    {inner_mol_svg}
  </g>
  <line x1="16" y1="{header_h + draw_h + 2}" x2="{width - 16}" y2="{header_h + draw_h + 2}" stroke="#F1F5F9" stroke-width="1"/>
  <rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="12" rx="6" fill="url(#{grad_id})" stroke="#CBD5E1" stroke-width="0.8"/>
  <text x="{legend_x}" y="{legend_y + 27}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" fill="#1E40AF">Negative (−)</text>
  <text x="{legend_x + legend_w // 2}" y="{legend_y + 27}" text-anchor="middle" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" fill="#475569">Neutral (0)</text>
  <text x="{legend_x + legend_w}" y="{legend_y + 27}" text-anchor="end" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" fill="#991B1B">Positive (+)</text>
</svg>
"""
    return svg_doc
