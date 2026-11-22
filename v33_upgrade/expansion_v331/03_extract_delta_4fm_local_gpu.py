#!/usr/bin/env python3
"""
Wave 5A — Step 03 (runs on LOCAL Windows RTX 5080 16GB GPU via _local_cache/venv/Scripts/python.exe):
Extracts ALL 4 Foundation Model embeddings for:
  - 5 existing V33 anchor compounds (verifies cosine > 0.999 against frozen V33 embeddings!)
  - K genuinely NEW delta compounds from expansion_v331/data/delta_compounds.tsv

Exact contracts matched 100% to Wave-2 extract_*.py:
  1. ncAA-RepDistill v2 (smiles_repr 768D + fusion_hf 2048D + protein_proj 1280D, float16)
     - EMA weights, SHA-verified 37-token vocab, graph_attention + SingleInputDataset (<=766 chars)
     - Window-length-weighted chunk_mean + neutral_graph_idx (>766 chars)
  2. PeptideCLM-2 hybrid-small (512D, float16)
     - RoPE SwiGLU ChemPepMTR.MLM_model, native mean_pool, weighted chunk_mean (>2046 tokens)
  3. ChemBERTa-77M-MTR (384D, float16)
     - Exclude BOS/EOS mean pool, weighted chunk_mean (>510 tokens)
  4. ESM-C 600M Multi-Region (mean, max, nterm5, cterm5; 4x1152D, float16)
     - Deduplicated parent_sequence, exclude BOS/EOS, mean/max/nterm5/cterm5 pooling

Outputs (in D:/work/中检院/v33_upgrade/expansion_v331/embeddings_delta/):
  - delta_ncaa_repdistill_v2.npz
  - delta_peptideclm2.npz
  - delta_chemberta_mtr.npz
  - delta_esmc_600m_multiregion.npz
  - delta_4fm_anchor_audit.json
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

LOCAL_CACHE = Path(r"D:\work\中检院\v33_upgrade\models_gpu\_local_cache")
REF_EMB_DIR = LOCAL_CACHE / "embeddings"
EXP_DATA_DIR = Path(r"D:\work\中检院\v33_upgrade\expansion_v331\data")
OUT_DIR = Path(r"D:\work\中检院\v33_upgrade\expansion_v331\embeddings_delta")

NCAA_REPO = LOCAL_CACHE / "ncaadistill"
NCAA_CFG = NCAA_REPO / "config" / "phase2" / "phase2_repro_sweep24_e0_b0.yaml"
NCAA_CKPT = NCAA_REPO / "artifacts" / "best.pt"
NCAA_VOCAB = NCAA_REPO / "data" / "raw" / "adaptation" / "vocab_smiles.txt"
EXPECTED_VOCAB_SHA = "7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92"

PEPTIDECLM2_DIR = LOCAL_CACHE / "peptideclm2"
CHEMBERTA_DIR = LOCAL_CACHE / "chemberta_mtr"
ESMC_DIR = LOCAL_CACHE / "esmc_600m"


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float64)
    b32 = b.astype(np.float64)
    na = float(np.linalg.norm(a32))
    nb = float(np.linalg.norm(b32))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a32, b32) / (na * nb))


def load_tsv(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append({
                "compound_idx": int(r["compound_idx"]),
                "compound_key": r["compound_key"],
                "canonical_smiles": r["canonical_smiles"],
                "parent_sequence": r["parent_sequence"],
                "smiles_valid": bool(int(r["smiles_valid"])),
                "seq_valid": bool(int(r["seq_valid"])),
            })
    return rows


# ---------------------------------------------------------------------------
# 1. ncAA-RepDistill v2
# ---------------------------------------------------------------------------
def run_ncaa(all_rows: List[dict], device: torch.device) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    t0 = time.time()
    if str(NCAA_REPO) not in sys.path:
        sys.path.insert(0, str(NCAA_REPO))
    from data.dataloader import collate_and_trim_smiles_batch
    from data.dataset import SingleInputDataset
    from data.vocab_manager import VocabManager
    from model.single_input_dual_teacher import SingleInputDualTeacher

    assert sha256_file(NCAA_VOCAB) == EXPECTED_VOCAB_SHA, "ncAA vocab SHA mismatch!"
    vocab_mgr = VocabManager(str(NCAA_VOCAB))
    with NCAA_CFG.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    smiles_cfg = cfg["student"]["smiles_encoder"]
    proj_cfg = cfg["student"]["projection"]
    ga_cfg = dict(cfg.get("student", {}).get("graph_attention") or {})
    ga_cfg["cache_size"] = 1
    cp_cfg = cfg.get("student", {}).get("contrastive_proj") or {}
    fus_cfg = cfg.get("fusion") or {}
    max_length = int(smiles_cfg["max_length"])

    model = SingleInputDualTeacher(
        smiles_vocab_size=int(smiles_cfg["vocab_size"]),
        smiles_embed_dim=int(smiles_cfg["embed_dim"]),
        smiles_num_layers=int(smiles_cfg["num_layers"]),
        smiles_num_heads=int(smiles_cfg["num_heads"]),
        smiles_max_length=max_length,
        esm2_dim=int(proj_cfg["esm2_dim"]),
        chemberta_dim=int(proj_cfg["chemberta_dim"]),
        graph_attention=ga_cfg,
        contrastive_proj_cfg=cp_cfg,
        fusion_cfg=fus_cfg,
        use_smiles_mlm=bool(cfg["student"].get("use_smiles_mlm", True)),
        dropout=float(smiles_cfg.get("dropout", 0.1)),
        use_gradient_checkpointing=False,
    )
    ckpt = torch.load(str(NCAA_CKPT), map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    assert not [k for k in missing if "rope.inv_freq" not in k], f"ncAA unexpected missing: {missing}"
    model.to(device).eval()

    N = len(all_rows)
    sr = np.zeros((N, 768), dtype=np.float16)
    fh = np.zeros((N, 2048), dtype=np.float16)
    pp = np.zeros((N, 1280), dtype=np.float16)
    chunked = np.zeros(N, dtype=np.bool_)

    payload_limit = max_length - 2  # 766
    std_rows, long_rows = [], []
    for li, r in enumerate(all_rows):
        if not r["smiles_valid"]:
            continue
        smi = r["canonical_smiles"]
        if len(smi) <= payload_limit:
            std_rows.append((li, smi))
        else:
            long_rows.append((li, smi))
    std_rows.sort(key=lambda x: len(x[1]))
    print(f"[ncAA] std={len(std_rows):,} | long={len(long_rows):,}", flush=True)

    if std_rows:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
            pd.DataFrame({"loc_i": [x[0] for x in std_rows], "canonical_smiles": [x[1] for x in std_rows]}).to_csv(tf.name, index=False)
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
                graph_attention_config=ga_cfg,
                label_column=None,
                normalization_config=None,
                split_name="v331_delta",
            )
            ds.smiles_col = "canonical_smiles"
            loader = DataLoader(ds, batch_size=48, shuffle=False, num_workers=0, collate_fn=collate_and_trim_smiles_batch)
            processed = 0
            with torch.inference_mode():
                for batch in loader:
                    ids = batch["smiles_input_ids"].to(device, non_blocking=True)
                    attn = batch["smiles_attention_mask"].to(device, non_blocking=True)
                    gd = batch.get("graph_distances")
                    if gd is not None:
                        gd = gd.to(device, non_blocking=True)
                    bsz = ids.size(0)
                    sub_bs = 24 if (ids.size(1) > 512 and bsz > 24) else bsz
                    sr_c, fh_c, pp_c = [], [], []
                    for s0 in range(0, bsz, sub_bs):
                        s1 = min(s0 + sub_bs, bsz)
                        gd_s = gd[s0:s1] if gd is not None else None
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            out = model(ids[s0:s1], attn[s0:s1], graph_distances=gd_s, return_all=False)
                        sr_c.append(out["smiles_repr"].float().cpu().numpy())
                        fh_c.append(out["fusion_hf"].float().cpu().numpy())
                        pp_c.append(out["protein_proj"].float().cpu().numpy())
                    locs = [std_rows[processed + k][0] for k in range(bsz)]
                    sr[locs] = np.concatenate(sr_c, axis=0).astype(np.float16)
                    fh[locs] = np.concatenate(fh_c, axis=0).astype(np.float16)
                    pp[locs] = np.concatenate(pp_c, axis=0).astype(np.float16)
                    processed += bsz
            assert processed == len(std_rows)
        finally:
            if os.path.exists(tmp_csv):
                os.unlink(tmp_csv)

    if long_rows:
        pad_id = int(vocab_mgr.pad_token_id)
        all_windows: List[str] = []
        mol_spans: List[Tuple[int, int, int, List[int]]] = []
        for li, smi in long_rows:
            st_w = len(all_windows)
            wts = []
            for st in range(0, len(smi), payload_limit):
                w = smi[st : st + payload_limit]
                if w:
                    all_windows.append(w)
                    wts.append(len(w))
            mol_spans.append((li, st_w, len(all_windows), wts))
        win_sr = np.empty((len(all_windows), 768), dtype=np.float32)
        win_fh = np.empty((len(all_windows), 2048), dtype=np.float32)
        win_pp = np.empty((len(all_windows), 1280), dtype=np.float32)
        ga_en = bool(getattr(model.smiles_encoder, "graph_attention_enabled", False))
        neutral_idx = None
        if ga_en:
            w_gb = model.smiles_encoder.graph_bias_embed.weight.detach()
            neutral_idx = int(w_gb[1:].norm(dim=1).argmin().item()) + 1
        with torch.inference_mode():
            for ws in range(0, len(all_windows), 24):
                sub = all_windows[ws : ws + 24]
                encoded = [vocab_mgr.encode(list(w), add_special_tokens=True, max_length=max_length, padding=False, truncation=True) for w in sub]
                L = max(len(e) for e in encoded)
                ids_t = torch.full((len(sub), L), pad_id, dtype=torch.long, device=device)
                attn_t = torch.zeros((len(sub), L), dtype=torch.long, device=device)
                for bi, e in enumerate(encoded):
                    ids_t[bi, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
                    attn_t[bi, : len(e)] = 1
                gd_t = None
                if ga_en:
                    gd_t = torch.zeros((len(sub), L, L), dtype=torch.long, device=device)
                    for bi in range(len(sub)):
                        lv = int(attn_t[bi].sum().item())
                        if lv > 0:
                            gd_t[bi, :lv, :lv] = neutral_idx
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids_t, attn_t, graph_distances=gd_t, return_all=False)
                win_sr[ws : ws + len(sub)] = out["smiles_repr"].float().cpu().numpy()
                win_fh[ws : ws + len(sub)] = out["fusion_hf"].float().cpu().numpy()
                win_pp[ws : ws + len(sub)] = out["protein_proj"].float().cpu().numpy()
        for li, w0, w1, wts in mol_spans:
            warr = np.asarray(wts, dtype=np.float32)
            wn = warr[:, None] / warr.sum()
            sr[li] = (win_sr[w0:w1] * wn).sum(axis=0).astype(np.float16)
            fh[li] = (win_fh[w0:w1] * wn).sum(axis=0).astype(np.float16)
            pp[li] = (win_pp[w0:w1] * wn).sum(axis=0).astype(np.float16)
            chunked[li] = True

    del model
    torch.cuda.empty_cache()
    print(f"[ncAA] Finished in {time.time()-t0:.1f}s", flush=True)
    return {"smiles_repr": sr, "fusion_hf": fh, "protein_proj": pp}, chunked


# ---------------------------------------------------------------------------
# 2. PeptideCLM-2 (hybrid-small 512D)
# ---------------------------------------------------------------------------
def run_peptideclm2(all_rows: List[dict], device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    from safetensors.torch import load_file as st_load
    from transformers import PreTrainedTokenizerFast

    pkg_name = "v331_peptideclm2_local"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(PEPTIDECLM2_DIR.resolve())]
        sys.modules[pkg_name] = pkg
    chem_mod = importlib.import_module(pkg_name + ".ChemPepMTR")
    cfg_cls = importlib.import_module(pkg_name + ".config").model_config
    cfg = cfg_cls(**json.loads((PEPTIDECLM2_DIR / "config.json").read_text(encoding="utf-8")))
    encoder = chem_mod.MLM_model(cfg)
    sd = st_load(str(PEPTIDECLM2_DIR / "model.safetensors"), device="cpu")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not (k.endswith(".theta") or k.endswith(".cache"))] and not unexpected
    encoder.model.sequence_head = nn.Identity()
    encoder.eval().to(device=device, dtype=torch.float32)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(PEPTIDECLM2_DIR / "tokenizer.json"),
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=2048,
    )
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    payload = 2046

    N = len(all_rows)
    emb = np.zeros((N, 512), dtype=np.float16)
    chunked = np.zeros(N, dtype=np.bool_)

    tokenized = []
    for li, r in enumerate(all_rows):
        if r["smiles_valid"]:
            ids = tokenizer(r["canonical_smiles"], add_special_tokens=False, truncation=False, padding=False)["input_ids"]
            tokenized.append((li, ids))
    tokenized.sort(key=lambda x: len(x[1]))
    std_items = [x for x in tokenized if len(x[1]) <= payload]
    long_items = [x for x in tokenized if len(x[1]) > payload]
    print(f"[PeptideCLM-2] std={len(std_items):,} | long={len(long_items):,}", flush=True)

    pos = 0
    with torch.inference_mode():
        while pos < len(std_items):
            max_tok = len(std_items[min(pos + 64 - 1, len(std_items) - 1)][1]) + 2
            eff_bs = 8 if max_tok > 1024 else (24 if max_tok > 512 else 64)
            batch = std_items[pos : pos + eff_bs]
            seqs = [[cls_id] + it[1] + [sep_id] for it in batch]
            L = max(len(s) for s in seqs)
            ids_t = torch.full((len(batch), L), pad_id, dtype=torch.long, device=device)
            mask_t = torch.zeros((len(batch), L), dtype=torch.long, device=device)
            for bi, s in enumerate(seqs):
                ids_t[bi, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                mask_t[bi, : len(s)] = 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = encoder(input_ids=ids_t, attention_mask=mask_t).mean_pool
            vecs = out.float().cpu().numpy().astype(np.float16)
            for bi, it in enumerate(batch):
                emb[it[0]] = vecs[bi]
            pos += len(batch)

        for li, full_ids in long_items:
            windows, wts = [], []
            for st in range(0, len(full_ids), payload):
                w = full_ids[st : st + payload]
                if w:
                    windows.append([cls_id] + w + [sep_id])
                    wts.append(len(w))
            wvecs = []
            for ws in range(0, len(windows), 4):
                sub_w = windows[ws : ws + 4]
                L = max(len(w) for w in sub_w)
                ids_t = torch.full((len(sub_w), L), pad_id, dtype=torch.long, device=device)
                mask_t = torch.zeros((len(sub_w), L), dtype=torch.long, device=device)
                for bi, w in enumerate(sub_w):
                    ids_t[bi, : len(w)] = torch.tensor(w, dtype=torch.long, device=device)
                    mask_t[bi, : len(w)] = 1
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = encoder(input_ids=ids_t, attention_mask=mask_t).mean_pool
                wvecs.append(out.float().cpu().numpy())
            wmat = np.concatenate(wvecs, axis=0)
            warr = np.asarray(wts, dtype=np.float32)
            emb[li] = (wmat * (warr[:, None] / warr.sum())).sum(axis=0).astype(np.float16)
            chunked[li] = True

    del encoder
    torch.cuda.empty_cache()
    print(f"[PeptideCLM-2] Finished in {time.time()-t0:.1f}s", flush=True)
    return emb, chunked


# ---------------------------------------------------------------------------
# 3. ChemBERTa-77M-MTR (384D)
# ---------------------------------------------------------------------------
def run_chemberta(all_rows: List[dict], device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CHEMBERTA_DIR), local_files_only=True)
    model = AutoModel.from_pretrained(str(CHEMBERTA_DIR), local_files_only=True, dtype=torch.bfloat16).eval().to(device)
    MAX_LEN = 512
    payload = MAX_LEN - 2
    cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    N = len(all_rows)
    emb = np.zeros((N, 384), dtype=np.float16)
    chunked = np.zeros(N, dtype=np.bool_)

    valid_items = [(li, r["canonical_smiles"]) for li, r in enumerate(all_rows) if r["smiles_valid"]]
    valid_items.sort(key=lambda x: len(x[1]))
    print(f"[ChemBERTa-MTR] valid={len(valid_items):,}", flush=True)

    with torch.inference_mode():
        for pos in range(0, len(valid_items), 128):
            batch = valid_items[pos : pos + 128]
            b_smis = [it[1] for it in batch]
            raw_ids = tok(b_smis, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
            short_j, long_items = [], []
            for j, ids in enumerate(raw_ids):
                if len(ids) <= payload:
                    short_j.append(j)
                else:
                    long_items.append((j, ids))
            b_vecs = np.zeros((len(batch), 384), dtype=np.float32)
            if short_j:
                enc = tok([b_smis[j] for j in short_j], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
                hidden = model(**enc).last_hidden_state.float()
                attn = enc["attention_mask"].bool()
                lens = attn.sum(dim=1)
                tm = attn.clone()
                tm[:, 0] = False
                for bi in range(len(short_j)):
                    L = int(lens[bi].item())
                    if L > 2:
                        tm[bi, L - 1] = False
                    else:
                        tm[bi] = attn[bi]
                pooled = ((hidden * tm.unsqueeze(-1)).sum(dim=1) / tm.sum(dim=1, keepdim=True).clamp(min=1)).cpu().numpy()
                for bi, oj in enumerate(short_j):
                    b_vecs[oj] = pooled[bi]
            for oj, full_ids in long_items:
                chunked[batch[oj][0]] = True
                windows, wts = [], []
                for st in range(0, len(full_ids), payload):
                    w = full_ids[st : st + payload]
                    if w:
                        windows.append([cls_id] + w + [sep_id])
                        wts.append(len(w))
                max_wl = max(len(w) for w in windows)
                w_ids = torch.full((len(windows), max_wl), pad_id, dtype=torch.long, device=device)
                w_attn = torch.zeros((len(windows), max_wl), dtype=torch.long, device=device)
                for wi, w in enumerate(windows):
                    w_ids[wi, : len(w)] = torch.tensor(w, dtype=torch.long, device=device)
                    w_attn[wi, : len(w)] = 1
                h = model(input_ids=w_ids, attention_mask=w_attn).last_hidden_state.float()
                am = w_attn.bool()
                tm = am.clone()
                tm[:, 0] = False
                lens = am.sum(dim=1)
                for k in range(tm.size(0)):
                    Lk = int(lens[k].item())
                    if Lk > 2:
                        tm[k, Lk - 1] = False
                    else:
                        tm[k] = am[k]
                wmat = ((h * tm.unsqueeze(-1)).sum(dim=1) / tm.sum(dim=1, keepdim=True).clamp(min=1)).cpu().numpy()
                warr = np.asarray(wts, dtype=np.float32)
                b_vecs[oj] = (wmat * (warr[:, None] / warr.sum())).sum(axis=0)
            for j, it in enumerate(batch):
                emb[it[0]] = b_vecs[j].astype(np.float16)

    del model
    torch.cuda.empty_cache()
    print(f"[ChemBERTa-MTR] Finished in {time.time()-t0:.1f}s", flush=True)
    return emb, chunked


# ---------------------------------------------------------------------------
# 4. ESM-C 600M Multi-Region (4 x 1152D)
# ---------------------------------------------------------------------------
def run_esmc(all_rows: List[dict], device: torch.device) -> Dict[str, np.ndarray]:
    t0 = time.time()
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ESMC_DIR), local_files_only=True)
    model = AutoModel.from_pretrained(str(ESMC_DIR), local_files_only=True, dtype=torch.bfloat16).eval().to(device)

    N = len(all_rows)
    seq_to_locs: Dict[str, List[int]] = {}
    for li, r in enumerate(all_rows):
        if r["seq_valid"]:
            seq_to_locs.setdefault(r["parent_sequence"], []).append(li)
    unique_seqs = sorted(seq_to_locs.keys(), key=len)
    print(f"[ESM-C 600M] unique valid seqs={len(unique_seqs):,}", flush=True)

    arr_mean = np.zeros((N, 1152), dtype=np.float16)
    arr_max = np.zeros((N, 1152), dtype=np.float16)
    arr_nt5 = np.zeros((N, 1152), dtype=np.float16)
    arr_ct5 = np.zeros((N, 1152), dtype=np.float16)

    pos = 0
    with torch.inference_mode():
        while pos < len(unique_seqs):
            next_max = len(unique_seqs[min(pos + 96 - 1, len(unique_seqs) - 1)])
            eff_bs = 16 if next_max > 256 else (32 if next_max > 128 else (64 if next_max > 64 else 96))
            b_seqs = unique_seqs[pos : pos + eff_bs]
            enc = tok(b_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            ids = enc["input_ids"].to(device, non_blocking=True)
            attn = enc["attention_mask"].to(device, non_blocking=True)
            out = model(input_ids=ids, attention_mask=attn).last_hidden_state.float()
            lens = attn.sum(dim=1).tolist()
            for bi, tot_len in enumerate(lens):
                res_h = out[bi, 1 : tot_len - 1, :] if tot_len > 2 else out[bi, : max(1, tot_len), :]
                k = min(5, res_h.size(0))
                vm = res_h.mean(dim=0).cpu().numpy().astype(np.float16)
                vx = res_h.max(dim=0).values.cpu().numpy().astype(np.float16)
                vn = res_h[:k].mean(dim=0).cpu().numpy().astype(np.float16)
                vc = res_h[-k:].mean(dim=0).cpu().numpy().astype(np.float16)
                locs = seq_to_locs[b_seqs[bi]]
                arr_mean[locs] = vm
                arr_max[locs] = vx
                arr_nt5[locs] = vn
                arr_ct5[locs] = vc
            pos += len(b_seqs)

    del model
    torch.cuda.empty_cache()
    print(f"[ESM-C 600M] Finished in {time.time()-t0:.1f}s", flush=True)
    return {"mean": arr_mean, "max": arr_max, "nterm5": arr_nt5, "cterm5": arr_ct5}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Local CUDA GPU is required!"
    device = torch.device("cuda:0")
    print(f"=== V33.1 Delta 4-FM Extraction on LOCAL GPU: {torch.cuda.get_device_name(0)} ===", flush=True)

    anchors = load_tsv(EXP_DATA_DIR / "anchor_5_compounds.tsv")
    deltas = load_tsv(EXP_DATA_DIR / "delta_compounds.tsv")
    n_anc = len(anchors)
    K = len(deltas)
    all_rows = anchors + deltas
    print(f"Loaded {n_anc} V33 anchors + {K:,} delta compounds (total={len(all_rows):,})", flush=True)

    # Extract all 4 FMs
    ncaa_out, ncaa_chk = run_ncaa(all_rows, device)
    pclm_emb, pclm_chk = run_peptideclm2(all_rows, device)
    cberta_emb, cberta_chk = run_chemberta(all_rows, device)
    esmc_out = run_esmc(all_rows, device)

    # STRICT 5-Anchor Cosine Verification (> 0.999) against frozen V33 reference embeddings!
    print("\n=== Verifying 5 Existing V33 Anchor Compounds vs Frozen V33 Embeddings ===", flush=True)
    ref_ncaa = np.load(REF_EMB_DIR / "ncaa_repdistill_v2.npz", allow_pickle=False)
    ref_pclm = np.load(REF_EMB_DIR / "peptideclm2.npz", allow_pickle=False)
    ref_cb = np.load(REF_EMB_DIR / "chemberta_mtr.npz", allow_pickle=False)
    ref_esmc = np.load(REF_EMB_DIR / "esmc_600m_multiregion.npz", allow_pickle=False)

    anchor_report = []
    min_cos_all = 1.0
    for ai, a in enumerate(anchors):
        cidx = a["compound_idx"]
        ck = a["compound_key"]
        assert str(ref_ncaa["compound_keys"][cidx]) == ck, f"Anchor compound_key mismatch at {cidx}"
        rec = {"compound_idx": cidx, "compound_key": ck, "smiles_len": len(a["canonical_smiles"]), "seq_len": len(a["parent_sequence"]), "cosines": {}}
        if a["smiles_valid"]:
            for k in ("smiles_repr", "fusion_hf", "protein_proj"):
                c = cos_sim(ncaa_out[k][ai], ref_ncaa[k][cidx])
                rec["cosines"][f"ncaa_{k}"] = round(c, 6)
                min_cos_all = min(min_cos_all, c)
                assert c > 0.999, f"Anchor {ck} ncaa_{k} cos={c:.6f} <= 0.999!"
            c_p = cos_sim(pclm_emb[ai], ref_pclm["embeddings"][cidx])
            rec["cosines"]["peptideclm2"] = round(c_p, 6)
            min_cos_all = min(min_cos_all, c_p)
            assert c_p > 0.999, f"Anchor {ck} peptideclm2 cos={c_p:.6f} <= 0.999!"

            c_cb = cos_sim(cberta_emb[ai], ref_cb["embeddings"][cidx])
            rec["cosines"]["chemberta_mtr"] = round(c_cb, 6)
            min_cos_all = min(min_cos_all, c_cb)
            assert c_cb > 0.999, f"Anchor {ck} chemberta_mtr cos={c_cb:.6f} <= 0.999!"
        if a["seq_valid"]:
            for k in ("mean", "max", "nterm5", "cterm5"):
                c_e = cos_sim(esmc_out[k][ai], ref_esmc[k][cidx])
                rec["cosines"][f"esmc_{k}"] = round(c_e, 6)
                min_cos_all = min(min_cos_all, c_e)
                assert c_e > 0.999, f"Anchor {ck} esmc_{k} cos={c_e:.6f} <= 0.999!"
        anchor_report.append(rec)
        print(f"  [ANCHOR PASS] idx={cidx:5d} {ck} | min_cos={min(rec['cosines'].values()):.6f} | {rec['cosines']}", flush=True)

    print(f"ALL 5 ANCHORS PASSED! Global min cosine = {min_cos_all:.6f} (> 0.999)!", flush=True)

    # Slice off the K delta rows and verify zero NaN/Inf and exact validity masks
    d_keys = np.array([r["compound_key"] for r in deltas])
    d_smi_v = np.array([r["smiles_valid"] for r in deltas], dtype=np.bool_)
    d_seq_v = np.array([r["seq_valid"] for r in deltas], dtype=np.bool_)

    def _check_finite(name: str, arr: np.ndarray, mask: np.ndarray):
        assert arr.shape[0] == K
        assert np.all(np.isfinite(arr)), f"Non-finite in {name}!"
        if mask.any():
            norms = np.linalg.norm(arr[mask].astype(np.float32), axis=1)
            assert float(norms.min()) > 1e-4, f"Zero vector in valid row of {name}!"
        if (~mask).any():
            assert float(np.abs(arr[~mask]).max()) == 0.0, f"Non-zero in invalid row of {name}!"

    d_ncaa = {k: v[n_anc:] for k, v in ncaa_out.items()}
    d_ncaa_chk = ncaa_chk[n_anc:]
    for k, v in d_ncaa.items():
        _check_finite(f"ncaa_{k}", v, d_smi_v)
    np.savez_compressed(
        OUT_DIR / "delta_ncaa_repdistill_v2.npz",
        smiles_repr=d_ncaa["smiles_repr"],
        fusion_hf=d_ncaa["fusion_hf"],
        protein_proj=d_ncaa["protein_proj"],
        valid=d_smi_v,
        chunked=d_ncaa_chk,
        compound_keys=d_keys,
    )

    d_pclm = pclm_emb[n_anc:]
    d_pclm_chk = pclm_chk[n_anc:]
    _check_finite("peptideclm2", d_pclm, d_smi_v)
    np.savez_compressed(
        OUT_DIR / "delta_peptideclm2.npz",
        embeddings=d_pclm,
        valid=d_smi_v,
        chunked=d_pclm_chk,
        compound_keys=d_keys,
    )

    d_cb = cberta_emb[n_anc:]
    d_cb_chk = cberta_chk[n_anc:]
    _check_finite("chemberta_mtr", d_cb, d_smi_v)
    np.savez_compressed(
        OUT_DIR / "delta_chemberta_mtr.npz",
        embeddings=d_cb,
        valid=d_smi_v,
        chunked=d_cb_chk,
        compound_keys=d_keys,
    )

    d_esmc = {k: v[n_anc:] for k, v in esmc_out.items()}
    for k, v in d_esmc.items():
        _check_finite(f"esmc_{k}", v, d_seq_v)
    np.savez_compressed(
        OUT_DIR / "delta_esmc_600m_multiregion.npz",
        mean=d_esmc["mean"],
        max=d_esmc["max"],
        nterm5=d_esmc["nterm5"],
        cterm5=d_esmc["cterm5"],
        valid=d_seq_v,
        compound_keys=d_keys,
    )

    audit = {
        "schema": "v331_delta_4fm_anchor_audit_v1",
        "gpu_name": torch.cuda.get_device_name(0),
        "n_anchors": n_anc,
        "n_delta_compounds": K,
        "n_delta_smiles_valid": int(d_smi_v.sum()),
        "n_delta_seq_valid": int(d_seq_v.sum()),
        "global_min_anchor_cosine": round(min_cos_all, 6),
        "anchor_verification_pass": True,
        "anchors": anchor_report,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (OUT_DIR / "delta_4fm_anchor_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"\nDONE Delta 4-FM extraction in {audit['elapsed_sec']}s! Saved 4 delta .npz files to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
