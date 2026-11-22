#!/usr/bin/env python3
"""Extract ncAA-RepDistill v2 embeddings (smiles_repr 768D + fusion_hf 2048D + protein_proj 1280D)
for all 46,598 V33 compounds.

Exact verified training contract (per asset_and_method_scout.md):
  - Python: /usr/bin/python3
  - Repo: /data/qlyu/ncaadistill
  - Config: /data/qlyu/ncaadistill/config/phase2/phase2_repro_sweep24_e0_b0.yaml
  - Checkpoint: /data/qlyu/ncaadistill/artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt
  - Weights: ckpt["ema_state_dict"] (asserting only non-trainable rope.inv_freq missing)
  - Vocab: /data/qlyu/ncaadistill/data/raw/adaptation/vocab_smiles.txt
           (asserting exact SHA-256 7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92)
  - Architecture: graph_attention (max_distance=8, per_head) + fusion_cfg (residual_distill)
  - Preprocessing: SingleInputDataset + dynamic batch trimming (collate_and_trim_smiles_batch)
                   + window-length-weighted chunk_mean for SMILES > 766 chars
  - Gate: Verifies MLM token recovery sanity test BEFORE extraction.

Output: /data/qlyu/v33_upgrade/embeddings/ncaa_repdistill_v2.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common_index import EMB_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

NCAA_REPO = Path(os.environ.get("NCAA_REPO_DIR", "/data/qlyu/ncaadistill"))
CONFIG_PATH = Path(os.environ.get("NCAA_CONFIG_PATH", str(NCAA_REPO / "config/phase2/phase2_repro_sweep24_e0_b0.yaml")))
CKPT_PATH = Path(os.environ.get("NCAA_CKPT_PATH", str(NCAA_REPO / "artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt")))
VOCAB_PATH = Path(os.environ.get("NCAA_VOCAB_PATH", str(NCAA_REPO / "data/raw/adaptation/vocab_smiles.txt")))
EXPECTED_VOCAB_SHA256 = "7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92"

sys.path.insert(0, str(NCAA_REPO))


def load_ncaa_model(device: torch.device):
    from data.vocab_manager import VocabManager
    from model.single_input_dual_teacher import SingleInputDualTeacher

    vocab_sha = sha256_file(VOCAB_PATH)
    assert vocab_sha == EXPECTED_VOCAB_SHA256, (
        f"Vocab SHA-256 mismatch! Got {vocab_sha}, expected {EXPECTED_VOCAB_SHA256}"
    )
    vocab_mgr = VocabManager(str(VOCAB_PATH))
    assert len(vocab_mgr) == 37, f"Expected vocab size 37, got {len(vocab_mgr)}"

    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    smiles_cfg = cfg["student"]["smiles_encoder"]
    proj_cfg = cfg["student"]["projection"]
    graph_attention_cfg = dict(cfg.get("student", {}).get("graph_attention") or {})
    # Every valid SMILES in V33 is unique; disable 50k-entry (768x768 int16 = 59 GB!) LRU cache bloat!
    graph_attention_cfg["cache_size"] = 1
    contrastive_proj_cfg = cfg.get("student", {}).get("contrastive_proj") or {}
    fusion_cfg = cfg.get("fusion") or {}

    assert graph_attention_cfg.get("enabled") is True, "graph_attention must be enabled!"
    assert fusion_cfg.get("enabled") is True, "fusion_cfg must be enabled!"

    model = SingleInputDualTeacher(
        smiles_vocab_size=int(smiles_cfg["vocab_size"]),
        smiles_embed_dim=int(smiles_cfg["embed_dim"]),
        smiles_num_layers=int(smiles_cfg["num_layers"]),
        smiles_num_heads=int(smiles_cfg["num_heads"]),
        smiles_max_length=int(smiles_cfg["max_length"]),
        esm2_dim=int(proj_cfg["esm2_dim"]),
        chemberta_dim=int(proj_cfg["chemberta_dim"]),
        graph_attention=graph_attention_cfg,
        contrastive_proj_cfg=contrastive_proj_cfg,
        fusion_cfg=fusion_cfg,
        use_smiles_mlm=bool(cfg["student"].get("use_smiles_mlm", True)),
        dropout=float(smiles_cfg.get("dropout", 0.1)),
        use_gradient_checkpointing=False,
    )

    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    assert "ema_state_dict" in ckpt, "ema_state_dict missing from checkpoint!"
    missing, unexpected = model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    print(f"[ncAA] Loaded EMA weights: missing={missing}, unexpected={unexpected}", flush=True)
    # Only non-trainable RoPE inv_freq buffer may be missing from EMA state_dict
    non_rope_missing = [k for k in missing if "rope.inv_freq" not in k]
    assert not non_rope_missing, f"Unexpected missing keys in EMA load: {non_rope_missing}"

    model.to(device)
    model.eval()
    return model, vocab_mgr, int(smiles_cfg["max_length"]), graph_attention_cfg, {
        "vocab_sha256": vocab_sha,
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_global_step": int(ckpt.get("global_step", -1)),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


@torch.inference_mode()
def run_mlm_sanity_test(model, vocab_mgr, max_length: int, graph_attention_cfg: dict, device: torch.device) -> dict:
    """Verify 100% single-position MLM recovery on Phe-Leu dipeptide SMILES."""
    from data.dataset import SingleInputDataset

    test_smiles = "CC(C)CC(NC(=O)C(N)Cc1ccccc1)C(=O)O"  # Phe-Leu
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write("canonical_smiles,dummy\n")
        tf.write(f'"{test_smiles}",0.0\n')
        tmp_csv = tf.name

    try:
        ds = SingleInputDataset(
            data_path=tmp_csv,
            smiles_vocab_manager=vocab_mgr,
            smiles_max_length=max_length,
            apply_smiles_masking=False,
            mask_prob=0.0,
            random_smiles_config=None,
            consistency_config=None,
            graph_attention_config=graph_attention_cfg,
            label_column=None,
            normalization_config=None,
            split_name="mlm_probe",
        )
        ds.smiles_col = "canonical_smiles"
        item = ds[0]
    finally:
        os.unlink(tmp_csv)

    mask = item["smiles_attention_mask"]
    L = int(mask.sum().item())
    ids_base = item["smiles_input_ids"][:L].clone()
    attn_base = mask[:L].clone()
    gd_base = item["graph_distances"][:L, :L].clone()

    # Probe every non-special token position (1 .. L-2)
    mask_id = vocab_mgr.mask_token_id
    probe_positions = list(range(1, L - 1))
    batch_ids = ids_base.unsqueeze(0).repeat(len(probe_positions), 1)
    batch_attn = attn_base.unsqueeze(0).repeat(len(probe_positions), 1)
    batch_gd = gd_base.unsqueeze(0).repeat(len(probe_positions), 1, 1)
    targets = []
    for bi, pos in enumerate(probe_positions):
        targets.append(int(ids_base[pos].item()))
        batch_ids[bi, pos] = mask_id

    out = model(
        batch_ids.to(device),
        batch_attn.to(device),
        graph_distances=batch_gd.to(device),
        return_all=True,
    )
    logits = out["smiles_mlm_logits"].float().cpu()
    correct = 0
    for bi, (pos, tgt) in enumerate(zip(probe_positions, targets)):
        pred = int(logits[bi, pos].argmax().item())
        if pred == tgt:
            correct += 1

    acc = correct / len(probe_positions)
    print(f"[MLM SANITY] Phe-Leu single-token recovery: {correct}/{len(probe_positions)} ({100*acc:.1f}%)", flush=True)
    assert acc >= 0.95, f"MLM sanity check failed! Only {correct}/{len(probe_positions)} ({100*acc:.1f}%)"
    return {
        "test_smiles": test_smiles,
        "correct": correct,
        "total_positions": len(probe_positions),
        "accuracy": round(acc, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--out", type=str, default=str(EMB_DIR / "ncaa_repdistill_v2.npz"))
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    print(f"=== ncAA-RepDistill v2 Extraction (device={device}, bs={args.batch_size}, workers={args.num_workers}, bf16={args.bf16}) ===", flush=True)

    model, vocab_mgr, max_length, graph_attention_cfg, load_info = load_ncaa_model(device)
    mlm_receipt = run_mlm_sanity_test(model, vocab_mgr, max_length, graph_attention_cfg, device)

    rows, case17_indices, _ = load_v33_compound_table()
    N = len(rows)
    valid_mask = np.array([r["smiles_valid"] for r in rows], dtype=np.bool_)
    compound_keys = np.array([r["compound_key"] for r in rows])

    payload_limit = max_length - 2  # 766 chars
    std_rows = []
    long_rows = []
    for r in rows:
        if not r["smiles_valid"]:
            continue
        smi = r["canonical_smiles"]
        if len(smi) <= payload_limit:
            std_rows.append((r["compound_idx"], smi))
        else:
            long_rows.append((r["compound_idx"], smi))

    # Sort standard SMILES by length so collate_and_trim_smiles_batch trims every batch tightly!
    std_rows.sort(key=lambda x: len(x[1]))
    print(f"[ncAA] Standard SMILES (<= {payload_limit} chars): {len(std_rows):,} | Long SMILES (chunk_mean): {len(long_rows):,}", flush=True)

    smiles_repr = np.zeros((N, 768), dtype=np.float16)
    fusion_hf = np.zeros((N, 2048), dtype=np.float16)
    protein_proj = np.zeros((N, 1280), dtype=np.float16)
    chunked_mask = np.zeros(N, dtype=np.bool_)

    # --- Part A: Standard SMILES via official SingleInputDataset + graph_distances + dynamic trimming ---
    from data.dataloader import collate_and_trim_smiles_batch
    from data.dataset import SingleInputDataset

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
        tmp_df = pd.DataFrame({
            "compound_idx": [x[0] for x in std_rows],
            "canonical_smiles": [x[1] for x in std_rows],
        })
        tmp_df.to_csv(tf.name, index=False)
        tmp_csv_path = tf.name

    try:
        ds = SingleInputDataset(
            data_path=tmp_csv_path,
            smiles_vocab_manager=vocab_mgr,
            smiles_max_length=max_length,
            apply_smiles_masking=False,
            mask_prob=0.0,
            random_smiles_config=None,
            consistency_config=None,
            graph_attention_config=graph_attention_cfg,
            label_column=None,
            normalization_config=None,
            split_name="v33_std",
        )
        ds.smiles_col = "canonical_smiles"

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=bool(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
            collate_fn=collate_and_trim_smiles_batch,
        )

        processed = 0
        t_std = time.time()
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                ids = batch["smiles_input_ids"].to(device, non_blocking=True)
                attn = batch["smiles_attention_mask"].to(device, non_blocking=True)
                gd = batch.get("graph_distances")
                if gd is not None:
                    gd = gd.to(device, non_blocking=True)

                bsz = ids.size(0)
                # Sub-batch if sequence length is near 768 to keep peak VRAM polite (<4 GiB)
                sub_bs = bsz
                if ids.size(1) > 512 and bsz > 24:
                    sub_bs = 24

                sr_chunks, fh_chunks, pp_chunks = [], [], []
                for s0 in range(0, bsz, sub_bs):
                    s1 = min(s0 + sub_bs, bsz)
                    gd_sub = gd[s0:s1] if gd is not None else None
                    if args.bf16 and device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out = model(ids[s0:s1], attn[s0:s1], graph_distances=gd_sub, return_all=False)
                    else:
                        out = model(ids[s0:s1], attn[s0:s1], graph_distances=gd_sub, return_all=False)
                    sr_chunks.append(out["smiles_repr"].float().cpu().numpy())
                    fh_chunks.append(out["fusion_hf"].float().cpu().numpy())
                    pp_chunks.append(out["protein_proj"].float().cpu().numpy())

                sr_np = np.concatenate(sr_chunks, axis=0)
                fh_np = np.concatenate(fh_chunks, axis=0)
                pp_np = np.concatenate(pp_chunks, axis=0)

                orig_indices = [std_rows[processed + k][0] for k in range(bsz)]
                smiles_repr[orig_indices] = sr_np.astype(np.float16)
                fusion_hf[orig_indices] = fh_np.astype(np.float16)
                protein_proj[orig_indices] = pp_np.astype(np.float16)
                processed += bsz

                if (step + 1) % 100 == 0 or processed >= len(std_rows):
                    vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
                    print(
                        f"[ncAA std] {processed:,}/{len(std_rows):,} (L={ids.size(1)}, peak_alloc={vram_mb:.0f}MiB, {time.time()-t_std:.1f}s)",
                        flush=True,
                    )
        assert processed == len(std_rows)
    finally:
        if os.path.exists(tmp_csv_path):
            os.unlink(tmp_csv_path)

    # --- Part B: Long SMILES (> 766 chars) via window-length-weighted chunk_mean (no graph bias on split windows) ---
    if long_rows:
        t_long = time.time()
        pad_id = int(vocab_mgr.pad_token_id)
        all_windows: List[str] = []
        mol_spans: List[Tuple[int, int, int, List[int]]] = []
        for cidx, smi in long_rows:
            start_w = len(all_windows)
            weights = []
            for st in range(0, len(smi), payload_limit):
                w = smi[st : st + payload_limit]
                if w:
                    all_windows.append(w)
                    weights.append(len(w))
            mol_spans.append((cidx, start_w, len(all_windows), weights))

        print(f"[ncAA long] Encoding {len(long_rows):,} long SMILES -> {len(all_windows):,} windows...", flush=True)
        win_sr = np.empty((len(all_windows), 768), dtype=np.float32)
        win_fh = np.empty((len(all_windows), 2048), dtype=np.float32)
        win_pp = np.empty((len(all_windows), 1280), dtype=np.float32)

        # Match official extract_phase3_features.py chunk_mean contract for graph_attention:
        # select neutral graph bias index (minimum L2 norm among {1..K-1}) for valid tokens, 0 for padding.
        graph_attn_enabled = bool(getattr(model.smiles_encoder, "graph_attention_enabled", False))
        neutral_graph_idx = None
        if graph_attn_enabled:
            w_gb = model.smiles_encoder.graph_bias_embed.weight.detach()
            neutral_graph_idx = int(w_gb[1:].norm(dim=1).argmin().item()) + 1
            print(f"[ncAA long] graph_attention enabled -> using official neutral_graph_idx={neutral_graph_idx}", flush=True)

        win_bs = min(args.batch_size, 24)
        with torch.inference_mode():
            for ws in range(0, len(all_windows), win_bs):
                sub = all_windows[ws : ws + win_bs]
                encoded = [
                    vocab_mgr.encode(list(w), add_special_tokens=True, max_length=max_length, padding=False, truncation=True)
                    for w in sub
                ]
                L = max(len(e) for e in encoded)
                ids_t = torch.full((len(sub), L), pad_id, dtype=torch.long, device=device)
                attn_t = torch.zeros((len(sub), L), dtype=torch.long, device=device)
                for bi, e in enumerate(encoded):
                    ids_t[bi, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
                    attn_t[bi, : len(e)] = 1
                gd_t = None
                if graph_attn_enabled:
                    gd_t = torch.zeros((len(sub), L, L), dtype=torch.long, device=device)
                    for bi in range(len(sub)):
                        lv = int(attn_t[bi].sum().item())
                        if lv > 0:
                            gd_t[bi, :lv, :lv] = neutral_graph_idx
                if args.bf16 and device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model(ids_t, attn_t, graph_distances=gd_t, return_all=False)
                else:
                    out = model(ids_t, attn_t, graph_distances=gd_t, return_all=False)
                win_sr[ws : ws + len(sub)] = out["smiles_repr"].float().cpu().numpy()
                win_fh[ws : ws + len(sub)] = out["fusion_hf"].float().cpu().numpy()
                win_pp[ws : ws + len(sub)] = out["protein_proj"].float().cpu().numpy()

        for cidx, w0, w1, weights in mol_spans:
            warr = np.asarray(weights, dtype=np.float32)
            w_norm = warr[:, None] / warr.sum()
            smiles_repr[cidx] = (win_sr[w0:w1] * w_norm).sum(axis=0).astype(np.float16)
            fusion_hf[cidx] = (win_fh[w0:w1] * w_norm).sum(axis=0).astype(np.float16)
            protein_proj[cidx] = (win_pp[w0:w1] * w_norm).sum(axis=0).astype(np.float16)
            chunked_mask[cidx] = True

        print(f"[ncAA long] Done {len(long_rows):,} long SMILES in {time.time()-t_long:.1f}s", flush=True)

    # Verify strict numerical & Case17 invariants
    stats = verify_embedding_arrays(
        {"smiles_repr": smiles_repr, "fusion_hf": fusion_hf, "protein_proj": protein_proj},
        valid_mask,
        case17_indices,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        smiles_repr=smiles_repr,
        fusion_hf=fusion_hf,
        protein_proj=protein_proj,
        valid=valid_mask,
        chunked=chunked_mask,
        compound_keys=compound_keys,
    )

    elapsed = round(time.time() - t0, 2)
    peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if device.type == "cuda" else 0.0
    meta = {
        "name": "ncaa_repdistill_v2",
        "file": str(out_path),
        "sha256": sha256_file(out_path),
        "size_bytes": out_path.stat().st_size,
        "checkpoint": str(CKPT_PATH),
        "config": str(CONFIG_PATH),
        "vocab": str(VOCAB_PATH),
        "load_info": load_info,
        "mlm_sanity_test": mlm_receipt,
        "device": str(device),
        "bf16": args.bf16,
        "batch_size": args.batch_size,
        "peak_torch_alloc_mib": peak_vram_mib,
        "long_smiles_strategy": "window_len_weighted_chunk_mean_766",
        "n_standard_smiles": len(std_rows),
        "n_chunked_smiles": int(chunked_mask.sum()),
        "elapsed_sec": elapsed,
        "arrays": stats,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE ncAA-RepDistill v2 in {elapsed}s -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB)", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
