#!/usr/bin/env python3
"""Persistent background ESM-C 600M Multi-Region sequence encoder worker.

Self-contained pure-PyTorch ESM-C 600M inference (exact match to BioHub / HuggingFace
`EsmcModel` + `EsmcTokenizer`, loaded directly from `config.json` + `model.safetensors`
with zero `transformers>=5` or `esm` package dependency). Runs natively inside Python 3.10+
with PyTorch + safetensors and serves localhost TCP length-prefixed JSON requests so
ESM-C 600M stays permanently warm in RAM with 0 GPU VRAM usage.

Exact pooling contract (identical to Wave-2B extract_esmc_600m.py):
  - 1 token per residue with BOS (<cls>, idx 0) and EOS (<eos>, idx L-1) excluded
  - Returns float32 [B, 4, 1152] corresponding to [mean, max, nterm5, cterm5].
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file


def _default_esmc_dir() -> Path:
    env_dir = os.environ.get("V33_ESMC_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    # Portable layout: <v33_next>/v33_runtime/bundle/esmc_worker.py -> <v33_next>/pretrained/esmc_600m
    portable = Path(__file__).resolve().parents[2] / "pretrained" / "esmc_600m"
    if portable.exists():
        return portable
    return Path(
        "<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36"
    )


HF_ESMC_DIR = _default_esmc_dir()
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 1022

# Canonical amino acid vocabulary identical to BioHub/HF EsmcTokenizer
ESMC_VOCAB = [
    "<cls>",  # 0
    "<pad>",  # 1
    "<eos>",  # 2
    "<unk>",  # 3
    "L",      # 4
    "A",      # 5
    "G",      # 6
    "V",      # 7
    "S",      # 8
    "E",      # 9
    "R",      # 10
    "T",      # 11
    "I",      # 12
    "D",      # 13
    "P",      # 14
    "K",      # 15
    "Q",      # 16
    "N",      # 17
    "F",      # 18
    "Y",      # 19
    "M",      # 20
    "H",      # 21
    "W",      # 22
    "C",      # 23
    "X",      # 24
    "B",      # 25
    "U",      # 26
    "Z",      # 27
    "O",      # 28
    ".",      # 29
    "-",      # 30
    "|",      # 31
    "<mask>", # 32
]
ESMC_TOKEN_TO_ID = {tok: idx for idx, tok in enumerate(ESMC_VOCAB)}
CLS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (1, 1, T, D)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class _EsmcLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden_states).to(hidden_states.dtype)


class _EsmcAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)
        self.q_norm = _EsmcLayerNorm(hidden_size, bias=False)
        self.k_norm = _EsmcLayerNorm(hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states)).to(hidden_states.dtype)
        k = self.k_norm(self.k_proj(hidden_states)).to(hidden_states.dtype)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(out)


class _EsmcMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _EsmcLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_hidden_layers: int,
        attention_bias: bool = False,
        mlp_bias: bool = False,
    ):
        super().__init__()
        self.self_attn = _EsmcAttention(hidden_size, num_heads, head_dim, bias=attention_bias)
        self.mlp = _EsmcMLP(hidden_size, intermediate_size, bias=mlp_bias)
        self.input_layernorm = _EsmcLayerNorm(hidden_size)
        self.post_attention_layernorm = _EsmcLayerNorm(hidden_size)
        self.scaling_factor = math.sqrt(num_hidden_layers / 36.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        attn_out = self.self_attn(self.input_layernorm(hidden_states), cos, sin, attn_mask)
        hidden_states = residual + attn_out / self.scaling_factor
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states)) / self.scaling_factor
        return hidden_states


class StandaloneEsmcModel(nn.Module):
    """Exact pure-PyTorch implementation of BioHub ESM-C (`EsmcModel`)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        hidden_size = int(cfg["hidden_size"])
        num_layers = int(cfg["num_hidden_layers"])
        num_heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg.get("head_dim") or (hidden_size // num_heads))
        intermediate_size = int(cfg["intermediate_size"])
        vocab_size = int(cfg["vocab_size"])
        attn_bias = bool(cfg.get("attention_bias", False))
        mlp_bias = bool(cfg.get("mlp_bias", False))
        rope_theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.layers = nn.ModuleList(
            [
                _EsmcLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    num_hidden_layers=num_layers,
                    attention_bias=attn_bias,
                    mlp_bias=mlp_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = _EsmcLayerNorm(hidden_size, bias=False)

    @classmethod
    def from_local_dir(cls, model_dir: Path, dtype: torch.dtype = torch.float32) -> "StandaloneEsmcModel":
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        model = cls(cfg)
        raw_sd = safetensors_load_file(str(model_dir / "model.safetensors"), device="cpu")
        sd = {}
        for k, v in raw_sd.items():
            if k.startswith("esmc."):
                sd[k[len("esmc.") :]] = v.to(dtype=dtype) if v.is_floating_point() else v
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, f"ESMC state_dict mismatch: missing={missing}, unexpected={unexpected}"
        model.to(dtype=dtype).eval()
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]
        pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.float32).unsqueeze(0)
        freqs = pos[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=hidden_states.dtype)
        sin = emb.sin().to(dtype=hidden_states.dtype)

        if bool(attention_mask.all()):
            attn_mask = None
        else:
            # Bidirectional key padding mask: True = attend, False = masked padding
            attn_mask = attention_mask[:, None, None, :].bool()

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attn_mask)
        return self.norm(hidden_states)


def _tokenize_esmc_batch(seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    encoded: List[List[int]] = []
    for s in seqs:
        s_trunc = s[:MAX_SEQ_LEN]
        ids = [CLS_ID] + [ESMC_TOKEN_TO_ID.get(ch, UNK_ID) for ch in s_trunc] + [EOS_ID]
        encoded.append(ids)
    max_l = max(len(x) for x in encoded)
    bsz = len(encoded)
    input_ids = torch.full((bsz, max_l), PAD_ID, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_l), dtype=torch.long)
    lengths: List[int] = []
    for bi, ids in enumerate(encoded):
        ln = len(ids)
        input_ids[bi, :ln] = torch.tensor(ids, dtype=torch.long)
        attn_mask[bi, :ln] = 1
        lengths.append(ln)
    return input_ids, attn_mask, lengths


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def _send_msg(conn: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def _recv_msg(conn: socket.socket) -> dict:
    hdr = _recv_exact(conn, 4)
    length = struct.unpack("!I", hdr)[0]
    if length > 64 * 1024 * 1024:
        raise ValueError(f"Message too large: {length}")
    payload = _recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))


class ESMCWorker:
    def __init__(self, model_dir: Path, num_threads: int = 16, use_bf16: bool = False):
        self.t_start = time.time()
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

        t0 = time.time()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.dtype_str = "bfloat16" if use_bf16 else "float32"
        self.model_dir = Path(model_dir)
        self.model = StandaloneEsmcModel.from_local_dir(self.model_dir, dtype=dtype)
        self.lock = threading.Lock()
        self.total_encoded = 0

        # Warmup pass so first user request is immediately warm!
        _ = self.encode_sequences(["ACDEFGHIKLM"])
        self.total_encoded = 0
        self.load_sec = round(time.time() - t0, 2)
        print(
            f"[esmc_worker] Ready in {self.load_sec}s (standalone_pt, dtype={self.dtype_str}, "
            f"threads={num_threads}, torch={torch.__version__}, model_dir={self.model_dir})",
            flush=True,
        )

    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        n = len(sequences)
        out_arr = np.zeros((n, 4, HIDDEN_DIM), dtype=np.float32)
        if n == 0:
            return out_arr

        # Deduplicate sequences inside batch
        uniq_map: Dict[str, List[int]] = {}
        for i, s in enumerate(sequences):
            s_clean = (s or "").strip().upper()[:MAX_SEQ_LEN]
            if len(s_clean) >= 2:
                uniq_map.setdefault(s_clean, []).append(i)
        if not uniq_map:
            return out_arr

        uniq_seqs = sorted(uniq_map.keys(), key=len)
        with self.lock, torch.inference_mode():
            bs = 16
            for pos in range(0, len(uniq_seqs), bs):
                batch_seqs = uniq_seqs[pos : pos + bs]
                input_ids, attention_mask, lengths = _tokenize_esmc_batch(batch_seqs)
                hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).float()  # [B, T, 1152]

                for bi, tot_len in enumerate(lengths):
                    if tot_len > 2:
                        res_h = hidden[bi, 1 : tot_len - 1, :]
                    else:
                        res_h = hidden[bi, : max(1, tot_len), :]
                    R = res_h.size(0)
                    k = min(5, R)
                    v_mean = res_h.mean(dim=0).numpy()
                    v_max = res_h.max(dim=0).values.numpy()
                    v_nt5 = res_h[:k].mean(dim=0).numpy()
                    v_ct5 = res_h[-k:].mean(dim=0).numpy()
                    stacked = np.stack([v_mean, v_max, v_nt5, v_ct5], axis=0).astype(np.float32)
                    for orig_i in uniq_map[batch_seqs[bi]]:
                        out_arr[orig_i] = stacked

        self.total_encoded += len(uniq_seqs)
        return out_arr


def handle_client(conn: socket.socket, worker: ESMCWorker) -> None:
    try:
        while True:
            req = _recv_msg(conn)
            cmd = req.get("cmd", "encode")
            if cmd == "health":
                _send_msg(
                    conn,
                    {
                        "ok": True,
                        "backend": "standalone_pytorch",
                        "dtype": worker.dtype_str,
                        "load_sec": worker.load_sec,
                        "uptime_s": round(time.time() - worker.t_start, 1),
                        "total_encoded": worker.total_encoded,
                    },
                )
            elif cmd == "encode":
                t0 = time.time()
                seqs = req.get("sequences", [])
                arr = worker.encode_sequences(seqs)
                raw_b64 = base64.b64encode(arr.tobytes()).decode("ascii")
                _send_msg(
                    conn,
                    {
                        "ok": True,
                        "shape": list(arr.shape),
                        "dtype": "float32",
                        "data_b64": raw_b64,
                        "elapsed_ms": round((time.time() - t0) * 1000.0, 2),
                    },
                )
            elif cmd == "ping":
                _send_msg(conn, {"ok": True, "pong": True})
            else:
                _send_msg(conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})
    except EOFError:
        pass
    except Exception as exc:
        try:
            _send_msg(conn, {"ok": False, "error": str(exc)})
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18902)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--model-dir", type=str, default=str(HF_ESMC_DIR))
    args = ap.parse_args()

    worker = ESMCWorker(Path(args.model_dir), num_threads=args.threads, use_bf16=args.bf16)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(32)
    print(f"[esmc_worker] Listening on {args.host}:{args.port}", flush=True)

    def _stop(signum, frame):
        print(f"[esmc_worker] Signal {signum} received, exiting.", flush=True)
        try:
            srv.close()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while True:
        conn, _ = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, worker), daemon=True)
        t.start()


if __name__ == "__main__":
    main()
