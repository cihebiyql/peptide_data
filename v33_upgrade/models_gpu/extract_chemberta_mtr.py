#!/usr/bin/env python3
"""Extract ChemBERTa-77M-MTR embeddings (384D, float16) for all 46,598 V33 compounds.

Supports BOTH:
  - GPU inference (bf16/fp16, polite batch size, ~30s on RTX 4090)
  - Fast multi-process CPU inference on node23 (64 cores, ~2 min)

Model: <WORK_DIR>/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca
Output: <WORK_DIR>/v33_upgrade/embeddings/chemberta_mtr.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_index import EMB_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

MODEL_PATH = Path(
    "<WORK_DIR>/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca"
)
MAX_LEN = 512


def _extract_chunk_worker(args: Tuple[int, List[Tuple[int, str]], str, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker function for CPU multiprocessing or single GPU execution."""
    worker_id, items, device_str, batch_size, threads_per_worker = args
    if device_str == "cpu":
        torch.set_num_threads(threads_per_worker)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    device = torch.device(device_str)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True, torch_dtype=dtype)
    model.eval().to(device)

    # Sort by SMILES length for minimal dynamic padding waste
    items_sorted = sorted(items, key=lambda x: len(x[1]))
    n = len(items_sorted)
    out_idx = np.empty(n, dtype=np.int32)
    out_emb = np.empty((n, 384), dtype=np.float16)
    out_trunc = np.zeros(n, dtype=np.bool_)

    # Split into standard (<=510 tokens) and long (>510 tokens) using chunk_mean pooling
    t0 = time.time()
    pos = 0
    while pos < n:
        batch = items_sorted[pos : pos + batch_size]
        b_indices = [it[0] for it in batch]
        b_smiles = [it[1] for it in batch]

        # Pre-check raw tokenized lengths to handle long SMILES cleanly via chunk_mean
        raw_ids = tok(b_smiles, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        short_pos = []
        long_items = []
        for j, ids in enumerate(raw_ids):
            if len(ids) <= MAX_LEN - 2:
                short_pos.append(j)
            else:
                long_items.append((j, ids))

        batch_vecs = np.zeros((len(batch), 384), dtype=np.float32)

        if short_pos:
            sub_smi = [b_smiles[j] for j in short_pos]
            enc = tok(
                sub_smi,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            ).to(device)
            with torch.inference_mode():
                hidden = model(**enc).last_hidden_state.float()  # [B, T, 384]
            attn = enc["attention_mask"].bool()
            # Exclude BOS (pos 0) and EOS (last valid pos) for clean chemical token mean pooling
            lengths = attn.sum(dim=1)
            tok_mask = attn.clone()
            tok_mask[:, 0] = False
            for bi in range(len(short_pos)):
                L = int(lengths[bi].item())
                if L > 2:
                    tok_mask[bi, L - 1] = False
                else:
                    tok_mask[bi] = attn[bi]
            denom = tok_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden * tok_mask.unsqueeze(-1)).sum(dim=1) / denom
            pooled_np = pooled.cpu().numpy()
            for bi, orig_j in enumerate(short_pos):
                batch_vecs[orig_j] = pooled_np[bi]

        if long_items:
            payload = MAX_LEN - 2
            cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
            sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
            for orig_j, full_ids in long_items:
                out_trunc[pos + orig_j] = True
                windows = []
                weights = []
                for start in range(0, len(full_ids), payload):
                    w = full_ids[start : start + payload]
                    if w:
                        windows.append([cls_id] + w + [sep_id])
                        weights.append(len(w))
                max_wlen = max(len(w) for w in windows)
                win_ids = torch.full((len(windows), max_wlen), pad_id, dtype=torch.long, device=device)
                win_attn = torch.zeros((len(windows), max_wlen), dtype=torch.long, device=device)
                for wi, w in enumerate(windows):
                    win_ids[wi, : len(w)] = torch.tensor(w, dtype=torch.long, device=device)
                    win_attn[wi, : len(w)] = 1
                win_vecs = []
                for ws in range(0, len(windows), batch_size):
                    with torch.inference_mode():
                        h = model(input_ids=win_ids[ws : ws + batch_size], attention_mask=win_attn[ws : ws + batch_size]).last_hidden_state.float()
                    am = win_attn[ws : ws + batch_size].bool()
                    tm = am.clone()
                    tm[:, 0] = False
                    lens = am.sum(dim=1)
                    for k in range(tm.size(0)):
                        Lk = int(lens[k].item())
                        if Lk > 2:
                            tm[k, Lk - 1] = False
                        else:
                            tm[k] = am[k]
                    d = tm.sum(dim=1, keepdim=True).clamp(min=1)
                    win_vecs.append(((h * tm.unsqueeze(-1)).sum(dim=1) / d).cpu().numpy())
                wmat = np.concatenate(win_vecs, axis=0)
                warr = np.asarray(weights, dtype=np.float32)
                batch_vecs[orig_j] = (wmat * (warr[:, None] / warr.sum())).sum(axis=0)

        out_idx[pos : pos + len(batch)] = b_indices
        out_emb[pos : pos + len(batch)] = batch_vecs.astype(np.float16)
        pos += len(batch)

        if worker_id == 0 and (pos % (batch_size * 50) == 0 or pos >= n):
            print(f"[ChemBERTa w0] {pos:,}/{n:,} ({time.time()-t0:.1f}s)", flush=True)

    return out_idx, out_emb, out_trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=1, help="CPU worker processes (only when --device cpu)")
    ap.add_argument("--threads-per-worker", type=int, default=6)
    ap.add_argument("--out", type=str, default=str(EMB_DIR / "chemberta_mtr.npz"))
    args = ap.parse_args()

    t0 = time.time()
    print(f"=== ChemBERTa-77M-MTR Extraction (device={args.device}, bs={args.batch_size}, workers={args.workers}) ===", flush=True)
    rows, case17_indices, _ = load_v33_compound_table()
    N = len(rows)

    valid_items = [(r["compound_idx"], r["canonical_smiles"]) for r in rows if r["smiles_valid"]]
    valid_mask = np.array([r["smiles_valid"] for r in rows], dtype=np.bool_)
    compound_keys = np.array([r["compound_key"] for r in rows])

    embeddings = np.zeros((N, 384), dtype=np.float16)
    chunked_mask = np.zeros(N, dtype=np.bool_)

    if args.device == "cpu" and args.workers > 1:
        # Interleave sorted items across workers for balanced lengths
        valid_sorted = sorted(valid_items, key=lambda x: len(x[1]))
        shards = [valid_sorted[w :: args.workers] for w in range(args.workers)]
        tasks = [(w, shards[w], "cpu", args.batch_size, args.threads_per_worker) for w in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for idx_arr, emb_arr, trunc_arr in ex.map(_extract_chunk_worker, tasks):
                embeddings[idx_arr] = emb_arr
                chunked_mask[idx_arr] = trunc_arr
    else:
        idx_arr, emb_arr, trunc_arr = _extract_chunk_worker(
            (0, valid_items, args.device, args.batch_size, args.threads_per_worker)
        )
        embeddings[idx_arr] = emb_arr
        chunked_mask[idx_arr] = trunc_arr

    # Verify strict invariants
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
        "name": "chemberta_mtr",
        "file": str(out_path),
        "sha256": sha256_file(out_path),
        "size_bytes": out_path.stat().st_size,
        "model_path": str(MODEL_PATH),
        "device": args.device,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "long_smiles_strategy": "weighted_chunk_mean_510tok",
        "n_chunked": int(chunked_mask.sum()),
        "elapsed_sec": elapsed,
        "arrays": stats,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE ChemBERTa-77M-MTR in {elapsed}s -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB), chunked={meta['n_chunked']}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
