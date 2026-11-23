#!/usr/bin/env python3
"""Extract PeptideCLM-2 (hybrid-small, 512D, float16) embeddings for all 46,598 V33 compounds.

Model: <WORK_DIR>/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/models/peptideclm-2-hybrid-small
Architecture: 14-block RoPE SwiGLU Transformer (ChemPepMTR.MLM_model, 31M params, embed_dim=512, max_seq_len=2048).
Features:
  - Runs L/D-Ala & Ser/pSer chirality/PTM canary check before full extraction.
  - Native mean_pool (excludes [PAD], matches audited benchmark contract).
  - Weighted chunk_mean pooling for any SMILES exceeding 2046 tokens (zero silent truncation!).
  - Supports GPU (bf16/fp32) or multi-process CPU on node23.
Output: <WORK_DIR>/v33_upgrade/embeddings/peptideclm2.npz
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_index import EMB_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

MODEL_DIR = Path(
    "<WORK_DIR>/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/models/peptideclm-2-hybrid-small"
)
MAX_LEN = 2048


def load_peptideclm2(device: torch.device, use_bf16: bool = False, model_dir: Path = MODEL_DIR):
    from transformers import PreTrainedTokenizerFast

    pkg_name = "v33_peptideclm2_hybrid_small"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(model_dir.resolve())]
        sys.modules[pkg_name] = pkg

    chem_mod = importlib.import_module(pkg_name + ".ChemPepMTR")
    cfg_cls = importlib.import_module(pkg_name + ".config").model_config
    cfg_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = cfg_cls(**cfg_dict)
    assert cfg.embed_dim == 512, f"Expected 512D, got {cfg.embed_dim}"

    from safetensors.torch import load_file as safetensors_load_file

    encoder = chem_mod.MLM_model(cfg)
    sd = safetensors_load_file(str(model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    # Only non-persistent RoPE buffers (if any) could be absent; trainable weights must match strictly
    non_buf_missing = [k for k in missing if not (k.endswith(".theta") or k.endswith(".cache"))]
    assert not non_buf_missing and not unexpected, (
        f"PeptideCLM-2 load mismatch: missing={missing}, unexpected={unexpected}"
    )

    # Drop unused MLM vocab head to save VRAM and compute
    encoder.model.sequence_head = nn.Identity()
    dtype = torch.bfloat16 if (use_bf16 and device.type == "cuda") else torch.float32
    encoder.eval().to(device=device, dtype=dtype)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_LEN,
    )
    assert tokenizer.pad_token_id == 0, f"Expected pad_token_id=0, got {tokenizer.pad_token_id}"
    return encoder, tokenizer


def run_chirality_ptm_canary(encoder, tokenizer, device: torch.device) -> dict:
    """Verify L/D-Ala stereochemical sensitivity and Ser/pSer PTM sensitivity."""
    pairs = {
        "L_vs_D_Ala": (
            "N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",  # L-Ala-L-Ala
            "N[C@H](C)C(=O)N[C@@H](C)C(=O)O",   # D-Ala-L-Ala
        ),
        "Ser_vs_pSer": (
            "N[C@@H](CO)C(=O)O",                       # L-Ser
            "N[C@@H](COP(=O)(O)O)C(=O)O",              # L-pSer
        ),
    }
    res = {}
    for name, (s1, s2) in pairs.items():
        enc = tokenizer([s1, s2], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = encoder(input_ids=ids, attention_mask=mask).mean_pool.float().cpu().numpy()
        l2 = float(np.linalg.norm(out[0] - out[1]))
        cos = float(np.dot(out[0], out[1]) / (np.linalg.norm(out[0]) * np.linalg.norm(out[1]) + 1e-12))
        assert np.isfinite(l2) and l2 > 1e-3, f"Canary {name} failed: l2={l2}"
        res[name] = {"l2_diff": round(l2, 4), "cosine_sim": round(cos, 6)}
    return res


def _forward_mean_pool(encoder, ids: torch.Tensor, mask: torch.Tensor, use_bf16: bool) -> np.ndarray:
    """Forward pass handling SDPA attn_mask dtype cleanly under bf16 or fp32."""
    # Note: ChemPepMTR MultiHeadAttention converts mask to float32 before SDPA.
    # Running under autocast(bfloat16) matches SDPA dtypes automatically on PyTorch 2.5+.
    with torch.inference_mode():
        if use_bf16 and ids.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = encoder(input_ids=ids, attention_mask=mask).mean_pool
        else:
            out = encoder(input_ids=ids, attention_mask=mask).mean_pool
    return out.float().cpu().numpy()


def _extract_worker(args: Tuple[int, List[Tuple[int, str]], str, int, int, bool, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    worker_id, items, device_str, batch_size, threads_per_worker, use_bf16, model_dir_str = args
    if device_str == "cpu":
        torch.set_num_threads(threads_per_worker)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    device = torch.device(device_str)
    # Note: Keep weights float32 and let autocast cast matmuls on CUDA, avoiding SDPA float32 mask mismatch!
    encoder, tokenizer = load_peptideclm2(device, use_bf16=False, model_dir=Path(model_dir_str))

    # Pre-tokenize all items to sort by EXACT token length (massive speedup for variable-length peptides!)
    t0 = time.time()
    tokenized = []
    unk_count = 0
    unk_id = tokenizer.unk_token_id
    for cidx, smi in items:
        ids = tokenizer(smi, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        if unk_id is not None and unk_id in ids:
            unk_count += 1
        tokenized.append((cidx, ids))

    tokenized.sort(key=lambda x: len(x[1]))
    n = len(tokenized)
    out_idx = np.empty(n, dtype=np.int32)
    out_emb = np.empty((n, 512), dtype=np.float16)
    out_chunked = np.zeros(n, dtype=np.bool_)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    payload = MAX_LEN - 2

    # Separate standard (<=2046 payload tokens) vs ultra-long (>2046 tokens)
    std_items = [x for x in tokenized if len(x[1]) <= payload]
    long_items = [x for x in tokenized if len(x[1]) > payload]

    cursor = 0
    pos = 0
    while pos < len(std_items):
        # Adaptive polite batch size for long sequences so VRAM stays very low (<2 GiB)
        max_tok = len(std_items[min(pos + batch_size - 1, len(std_items) - 1)][1]) + 2
        if device.type == "cuda":
            eff_bs = batch_size
            if max_tok > 1024:
                eff_bs = min(eff_bs, 8)
            elif max_tok > 512:
                eff_bs = min(eff_bs, 24)
            elif max_tok > 256:
                eff_bs = min(eff_bs, 64)
        else:
            eff_bs = batch_size

        batch = std_items[pos : pos + eff_bs]
        seqs = [[cls_id] + it[1] + [sep_id] for it in batch]
        L = max(len(s) for s in seqs)
        ids_t = torch.full((len(batch), L), pad_id, dtype=torch.long, device=device)
        mask_t = torch.zeros((len(batch), L), dtype=torch.long, device=device)
        for bi, s in enumerate(seqs):
            ids_t[bi, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            mask_t[bi, : len(s)] = 1

        vecs = _forward_mean_pool(encoder, ids_t, mask_t, use_bf16=use_bf16)
        b_len = len(batch)
        out_idx[cursor : cursor + b_len] = [it[0] for it in batch]
        out_emb[cursor : cursor + b_len] = vecs.astype(np.float16)
        cursor += b_len
        pos += b_len

        if worker_id == 0 and (pos % 4096 < b_len or pos >= len(std_items)):
            print(f"[PeptideCLM-2 w0] std {pos:,}/{len(std_items):,} (maxL={L}, {time.time()-t0:.1f}s)", flush=True)

    # Ultra-long SMILES: weighted chunk_mean pooling over 2046-token windows
    for cidx, full_ids in long_items:
        windows = []
        weights = []
        for start in range(0, len(full_ids), payload):
            w = full_ids[start : start + payload]
            if w:
                windows.append([cls_id] + w + [sep_id])
                weights.append(len(w))
        win_vecs = []
        for ws in range(0, len(windows), 4):
            sub_w = windows[ws : ws + 4]
            L = max(len(w) for w in sub_w)
            ids_t = torch.full((len(sub_w), L), pad_id, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(sub_w), L), dtype=torch.long, device=device)
            for bi, w in enumerate(sub_w):
                ids_t[bi, : len(w)] = torch.tensor(w, dtype=torch.long, device=device)
                mask_t[bi, : len(w)] = 1
            win_vecs.append(_forward_mean_pool(encoder, ids_t, mask_t, use_bf16=use_bf16))
        wmat = np.concatenate(win_vecs, axis=0)
        warr = np.asarray(weights, dtype=np.float32)
        vec = (wmat * (warr[:, None] / warr.sum())).sum(axis=0)
        out_idx[cursor] = cidx
        out_emb[cursor] = vec.astype(np.float16)
        out_chunked[cursor] = True
        cursor += 1

    assert cursor == n
    return out_idx, out_emb, out_chunked, unk_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads-per-worker", type=int, default=6)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    ap.add_argument("--out", type=str, default=str(EMB_DIR / "peptideclm2.npz"))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    t0 = time.time()
    print(f"=== PeptideCLM-2 (hybrid-small 512D) Extraction (device={args.device}, bs={args.batch_size}, workers={args.workers}, bf16={args.bf16}) ===", flush=True)

    # Canary verification first on target device (or CPU if multi-worker CPU)
    canary_dev = torch.device("cpu" if args.device == "cpu" else args.device)
    enc_c, tok_c = load_peptideclm2(canary_dev, use_bf16=False, model_dir=model_dir)
    canary_res = run_chirality_ptm_canary(enc_c, tok_c, canary_dev)
    print(f"[CANARY PASS] {json.dumps(canary_res)}", flush=True)
    del enc_c
    if canary_dev.type == "cuda":
        torch.cuda.empty_cache()

    rows, case17_indices, _ = load_v33_compound_table()
    N = len(rows)
    valid_items = [(r["compound_idx"], r["canonical_smiles"]) for r in rows if r["smiles_valid"]]
    valid_mask = np.array([r["smiles_valid"] for r in rows], dtype=np.bool_)
    compound_keys = np.array([r["compound_key"] for r in rows])

    embeddings = np.zeros((N, 512), dtype=np.float16)
    chunked_mask = np.zeros(N, dtype=np.bool_)
    total_unk = 0

    if args.device == "cpu" and args.workers > 1:
        valid_sorted = sorted(valid_items, key=lambda x: len(x[1]))
        shards = [valid_sorted[w :: args.workers] for w in range(args.workers)]
        tasks = [(w, shards[w], "cpu", args.batch_size, args.threads_per_worker, False, str(model_dir)) for w in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for idx_arr, emb_arr, chk_arr, ucnt in ex.map(_extract_worker, tasks):
                embeddings[idx_arr] = emb_arr
                chunked_mask[idx_arr] = chk_arr
                total_unk += ucnt
    else:
        idx_arr, emb_arr, chk_arr, total_unk = _extract_worker(
            (0, valid_items, args.device, args.batch_size, args.threads_per_worker, args.bf16, str(model_dir))
        )
        embeddings[idx_arr] = emb_arr
        chunked_mask[idx_arr] = chk_arr

    stats = verify_embedding_arrays({"embeddings": embeddings}, valid_mask, case17_indices)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        valid=valid_mask,
        chunked=chunked_mask,
        compound_keys=compound_keys,
    )
    elapsed = round(time.time() - t0, 2)
    meta = {
        "name": "peptideclm2",
        "variant": "peptideclm-2-hybrid-small",
        "file": str(out_path),
        "sha256": sha256_file(out_path),
        "size_bytes": out_path.stat().st_size,
        "model_dir": str(MODEL_DIR),
        "device": args.device,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "bf16": args.bf16,
        "max_seq_len": MAX_LEN,
        "n_chunked": int(chunked_mask.sum()),
        "n_smiles_with_unk": int(total_unk),
        "canary_check": canary_res,
        "elapsed_sec": elapsed,
        "arrays": stats,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE PeptideCLM-2 in {elapsed}s -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB), chunked={meta['n_chunked']}, unk={total_unk}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
