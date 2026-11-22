#!/usr/bin/env python3
"""Extract ESM-C 600M multi-region pooled embeddings (mean, max, nterm5, cterm5; each 1152D float16)
for all 46,598 V33 compounds.

Strategy:
  - Deduplicates valid parent_sequence strings (44,773 valid rows -> 31,787 unique sequences, ~29% savings).
  - Sorts unique sequences by length for tight dynamic batching (median peptide len = 18 aa!).
  - Computes 4 complementary residue-level pooling views per sequence (excluding BOS/EOS):
      * mean   (1152D): global sequence mean
      * max    (1152D): global residue max (captures localized motifs/anchors)
      * nterm5 (1152D): mean of first min(5, L) residues (N-terminal proteolytic/DPP-4/signal stability)
      * cterm5 (1152D): mean of last  min(5, L) residues (C-terminal carboxypeptidase/amidation region)
  - Maps deduplicated embeddings back to all 46,598 compound_index.tsv rows.

Python: /data/qlyu/esmfold2-env312/bin/python (transformers 5.22 + esm 3.4.2 + torch 2.11+cu130)
Model weights:
  - Primary (batched HF safetensors): /data/qlyu/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36
  - Mirror (.pth): /data/qlyu/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/snapshots/d11cc14d44078eaecbc6a843d5eb20f4eecc1e7e
Output: /data/qlyu/v33_upgrade/embeddings/esmc_600m_multiregion.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common_index import EMB_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

HF_ESMC_DIR = Path(
    "/data/qlyu/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36"
)
PTH_ESMC_SNAP = Path(
    "/data/qlyu/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/snapshots/d11cc14d44078eaecbc6a843d5eb20f4eecc1e7e"
)
HIDDEN_DIM = 1152
MAX_SEQ_LEN = 1022


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--model-dir", type=str, default=str(HF_ESMC_DIR))
    ap.add_argument("--out", type=str, default=str(EMB_DIR / "esmc_600m_multiregion.npz"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    print(f"=== ESM-C 600M Multi-Region Extraction (device={device}, bs={args.batch_size}, bf16={args.bf16}) ===", flush=True)

    rows, case17_indices, _ = load_v33_compound_table()
    N = len(rows)
    valid_mask = np.array([r["seq_valid"] for r in rows], dtype=np.bool_)
    compound_keys = np.array([r["compound_key"] for r in rows])

    # Deduplicate unique valid sequences
    seq_to_rows: Dict[str, List[int]] = {}
    for r in rows:
        if r["seq_valid"]:
            s = r["parent_sequence"]
            seq_to_rows.setdefault(s, []).append(r["compound_idx"])

    unique_seqs = sorted(seq_to_rows.keys(), key=len)
    n_unique = len(unique_seqs)
    print(f"[ESM-C] Valid sequence rows={int(valid_mask.sum()):,}, unique sequences={n_unique:,} (minL={len(unique_seqs[0])}, maxL={len(unique_seqs[-1])})", flush=True)

    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True, dtype=dtype)
    model.eval().to(device)

    # Sanity check 1-token-per-residue + BOS/EOS framing on short peptide
    probe_enc = tok(["ACDEFG", "MKL"], return_tensors="pt", padding=True)
    assert probe_enc["input_ids"].shape == (2, 8), f"Unexpected ESM-C tokenization shape: {probe_enc['input_ids'].shape}"
    assert int(probe_enc["input_ids"][0, 0]) == tok.cls_token_id or int(probe_enc["input_ids"][0, 0]) == 0
    assert int(probe_enc["input_ids"][0, 7]) == tok.eos_token_id or int(probe_enc["input_ids"][0, 7]) == 2
    with torch.inference_mode():
        probe_out = model(**{k: v.to(device) for k, v in probe_enc.items()}).last_hidden_state
    assert probe_out.shape == (2, 8, HIDDEN_DIM), f"Unexpected ESM-C output shape: {probe_out.shape}"
    assert torch.isfinite(probe_out).all().item(), "ESM-C probe produced non-finite values!"
    print(f"[ESM-C] Model loaded & probe PASSED (shape={tuple(probe_out.shape)}, dtype={probe_out.dtype}, {time.time()-t0:.1f}s)", flush=True)

    u_mean = np.empty((n_unique, HIDDEN_DIM), dtype=np.float16)
    u_max = np.empty((n_unique, HIDDEN_DIM), dtype=np.float16)
    u_nterm5 = np.empty((n_unique, HIDDEN_DIM), dtype=np.float16)
    u_cterm5 = np.empty((n_unique, HIDDEN_DIM), dtype=np.float16)

    t_inf = time.time()
    pos = 0
    step = 0
    with torch.inference_mode():
        while pos < n_unique:
            # Adaptive polite batch size so peak VRAM stays well under 5 GiB even on long sequences
            next_max_len = len(unique_seqs[min(pos + args.batch_size - 1, n_unique - 1)])
            eff_bs = args.batch_size
            if next_max_len > 256:
                eff_bs = min(eff_bs, 16)
            elif next_max_len > 128:
                eff_bs = min(eff_bs, 32)
            elif next_max_len > 64:
                eff_bs = min(eff_bs, 64)

            batch_seqs = unique_seqs[pos : pos + eff_bs]
            enc = tok(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN + 2,
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)

            out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()  # [B, T, 1152]
            lengths = attention_mask.sum(dim=1).tolist()

            b_mean = np.empty((len(batch_seqs), HIDDEN_DIM), dtype=np.float16)
            b_max = np.empty((len(batch_seqs), HIDDEN_DIM), dtype=np.float16)
            b_nt5 = np.empty((len(batch_seqs), HIDDEN_DIM), dtype=np.float16)
            b_ct5 = np.empty((len(batch_seqs), HIDDEN_DIM), dtype=np.float16)

            for bi, tot_len in enumerate(lengths):
                # Exclude BOS (idx 0) and EOS (idx tot_len - 1)
                if tot_len > 2:
                    res_h = out[bi, 1 : tot_len - 1, :]  # [R, 1152]
                else:
                    res_h = out[bi, : max(1, tot_len), :]
                R = res_h.size(0)
                k = min(5, R)
                b_mean[bi] = res_h.mean(dim=0).cpu().numpy().astype(np.float16)
                b_max[bi] = res_h.max(dim=0).values.cpu().numpy().astype(np.float16)
                b_nt5[bi] = res_h[:k].mean(dim=0).cpu().numpy().astype(np.float16)
                b_ct5[bi] = res_h[-k:].mean(dim=0).cpu().numpy().astype(np.float16)

            u_mean[pos : pos + len(batch_seqs)] = b_mean
            u_max[pos : pos + len(batch_seqs)] = b_max
            u_nterm5[pos : pos + len(batch_seqs)] = b_nt5
            u_cterm5[pos : pos + len(batch_seqs)] = b_ct5

            pos += len(batch_seqs)
            step += 1
            if step % 50 == 0 or pos >= n_unique:
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
                print(
                    f"[ESM-C] {pos:,}/{n_unique:,} unique seqs (batchMaxL={len(batch_seqs[-1])}, peak_alloc={vram_mb:.0f}MiB, {time.time()-t_inf:.1f}s)",
                    flush=True,
                )

    # Map deduplicated embeddings back to all 46,598 compound indices
    arr_mean = np.zeros((N, HIDDEN_DIM), dtype=np.float16)
    arr_max = np.zeros((N, HIDDEN_DIM), dtype=np.float16)
    arr_nt5 = np.zeros((N, HIDDEN_DIM), dtype=np.float16)
    arr_ct5 = np.zeros((N, HIDDEN_DIM), dtype=np.float16)

    for ui, seq in enumerate(unique_seqs):
        idxs = seq_to_rows[seq]
        arr_mean[idxs] = u_mean[ui]
        arr_max[idxs] = u_max[ui]
        arr_nt5[idxs] = u_nterm5[ui]
        arr_ct5[idxs] = u_cterm5[ui]

    stats = verify_embedding_arrays(
        {"mean": arr_mean, "max": arr_max, "nterm5": arr_nt5, "cterm5": arr_ct5},
        valid_mask,
        case17_indices,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mean=arr_mean,
        max=arr_max,
        nterm5=arr_nt5,
        cterm5=arr_ct5,
        valid=valid_mask,
        compound_keys=compound_keys,
    )
    elapsed = round(time.time() - t0, 2)
    peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if device.type == "cuda" else 0.0
    meta = {
        "name": "esmc_600m_multiregion",
        "file": str(out_path),
        "sha256": sha256_file(out_path),
        "size_bytes": out_path.stat().st_size,
        "model_dir": str(model_dir),
        "hf_hub_mirror": str(PTH_ESMC_SNAP),
        "device": str(device),
        "bf16": args.bf16,
        "batch_size": args.batch_size,
        "n_unique_sequences": n_unique,
        "n_valid_rows": int(valid_mask.sum()),
        "peak_torch_alloc_mib": peak_vram_mib,
        "elapsed_sec": elapsed,
        "arrays": stats,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE ESM-C 600M in {elapsed}s -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB)", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
