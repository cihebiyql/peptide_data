#!/usr/bin/env python3
"""Benchmark (and optionally extract Tier-1 PK/permeability subset for) Uni-Mol2-84M.

Python: <WORK_DIR>/anaconda3/envs/cuda12.1/bin/python
Weights: <WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2/modelzoo/84M/checkpoint.pt
  (via UNIMOL_WEIGHT_DIR=<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2)

Modes:
  --mode benchmark  : Benchmark 500 cyclic/modified peptides (+ Case17), record exact
                      conformer+GPU throughput, peak VRAM, and projected time for Tier-1 (~10k) & full (43.9k).
                      Saves <WORK_DIR>/v33_upgrade/embeddings/unimol2_benchmark.json.
  --mode tier1      : Extract Uni-Mol2-84M embeddings for all unique valid SMILES across Tier-1 PK/permeability
                      endpoints (PAMPA, Caco2, MDCK, RRCK, PPB, CL_per_kg, CL_absolute, Vd_per_kg, F, Cmax,
                      LogD7.4, BBB, half_life_reg, half_life_cls, solubility_reg) + all 17 Case17 drugs,
                      aligned to all 46,598 compound_index.tsv rows with explicit valid mask!
                      Saves <WORK_DIR>/v33_upgrade/embeddings/unimol2_84m_tier1.npz.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set

os.environ.setdefault("UNIMOL_WEIGHT_DIR", "<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common_index import DATA_DIR, EMB_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

TIER1_PK_PERM_ENDPOINTS = [
    "PAMPA", "Caco2", "MDCK", "RRCK", "BBB", "LogD7.4",
    "PPB", "CL_per_kg", "CL_absolute", "Vd_per_kg", "F", "Cmax",
    "half_life_reg", "half_life_cls", "solubility_reg",
]


def collect_tier1_pk_indices(rows: List[dict], case17_indices: dict) -> List[int]:
    key_to_idx = {r["compound_key"]: r["compound_idx"] for r in rows if r["smiles_valid"]}
    selected: Set[int] = set(case17_indices.values())
    ep_dir = DATA_DIR / "endpoints"
    for ep in TIER1_PK_PERM_ENDPOINTS:
        p = ep_dir / f"{ep}.tsv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                ck = r.get("compound_key", "")
                if ck in key_to_idx:
                    selected.add(key_to_idx[ck])
    return sorted(selected)


def run_unimol2_batched(smiles_list: List[str], batch_size: int = 32, max_atoms: int = 256) -> np.ndarray:
    from unimol_tools import UniMolRepr

    clf = UniMolRepr(
        data_type="molecule",
        remove_hs=True,
        model_name="unimolv2",
        model_size="84m",
        batch_size=batch_size,
        use_cuda=torch.cuda.is_available(),
        use_gpu="0",
    )
    # Process in chunks of 1000 so RAM stays tiny and progress prints cleanly
    out_vecs = []
    chunk_sz = 1000
    t0 = time.time()
    for s0 in range(0, len(smiles_list), chunk_sz):
        sub = smiles_list[s0 : s0 + chunk_sz]
        rep = clf.get_repr(sub, return_atomic_reprs=False)
        cls_repr = np.asarray(rep["cls_repr"], dtype=np.float32)
        out_vecs.append(cls_repr)
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            vram_mb = 0.0
        print(
            f"[Uni-Mol2-84M] {min(s0 + chunk_sz, len(smiles_list)):,}/{len(smiles_list):,} "
            f"(dim={cls_repr.shape[1]}, peak_alloc={vram_mb:.0f}MiB, {time.time()-t0:.1f}s)",
            flush=True,
        )
    return np.concatenate(out_vecs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["benchmark", "tier1"], default="benchmark")
    ap.add_argument("--n-bench", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    rows, case17_indices, _ = load_v33_compound_table()
    N = len(rows)
    tier1_indices = collect_tier1_pk_indices(rows, case17_indices)
    print(f"[Uni-Mol2] Tier-1 PK/permeability unique valid SMILES (+ Case17): {len(tier1_indices):,}", flush=True)

    if args.mode == "benchmark":
        # Pick 500 cyclic/modified peptides (all 17 Case17 + 483 from Tier-1 PK/PAMPA)
        c17_set = set(case17_indices.values())
        bench_idx = sorted(c17_set) + [i for i in tier1_indices if i not in c17_set][: max(0, args.n_bench - len(c17_set))]
        bench_smiles = [rows[i]["canonical_smiles"] for i in bench_idx]
        t_run = time.time()
        vecs = run_unimol2_batched(bench_smiles, batch_size=args.batch_size)
        dt = time.time() - t_run
        rate = len(bench_smiles) / max(dt, 1e-6)
        est_tier1_min = (len(tier1_indices) / rate) / 60.0
        est_full_min = (43943 / rate) / 60.0
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0
        finite_ok = bool(np.all(np.isfinite(vecs)))
        col_std = float(vecs.std(axis=0).mean())
        report = {
            "model": "Uni-Mol2-84M",
            "checkpoint": "<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2/modelzoo/84M/checkpoint.pt",
            "n_bench": len(bench_smiles),
            "embed_dim": int(vecs.shape[1]),
            "elapsed_sec": round(dt, 2),
            "throughput_mol_per_sec": round(rate, 2),
            "n_tier1_pk_perm": len(tier1_indices),
            "est_tier1_minutes": round(est_tier1_min, 2),
            "est_full_43943_minutes": round(est_full_min, 2),
            "peak_torch_alloc_mib": peak_vram,
            "finite_ok": finite_ok,
            "mean_dim_std": round(col_std, 6),
            "recommended_tier1_extract": bool(finite_ok and est_tier1_min < 15.0 and peak_vram < 10000),
        }
        out_json = EMB_DIR / "unimol2_benchmark.json"
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"=== Uni-Mol2-84M Benchmark Result ===\n{json.dumps(report, indent=2)}", flush=True)
    else:
        smiles_list = [rows[i]["canonical_smiles"] for i in tier1_indices]
        vecs = run_unimol2_batched(smiles_list, batch_size=args.batch_size)
        D = vecs.shape[1]
        embeddings = np.zeros((N, D), dtype=np.float16)
        valid_mask = np.zeros(N, dtype=np.bool_)
        compound_keys = np.array([r["compound_key"] for r in rows])

        embeddings[tier1_indices] = vecs.astype(np.float16)
        valid_mask[tier1_indices] = True

        stats = verify_embedding_arrays({"embeddings": embeddings}, valid_mask, case17_indices)
        out_path = EMB_DIR / "unimol2_84m_tier1.npz"
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            valid=valid_mask,
            compound_keys=compound_keys,
        )
        elapsed = round(time.time() - t0, 2)
        peak_vram = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0
        meta = {
            "name": "unimol2_84m_tier1",
            "file": str(out_path),
            "sha256": sha256_file(out_path),
            "size_bytes": out_path.stat().st_size,
            "checkpoint": "<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2/modelzoo/84M/checkpoint.pt",
            "scope_endpoints": TIER1_PK_PERM_ENDPOINTS + ["Case17"],
            "n_valid": int(valid_mask.sum()),
            "peak_torch_alloc_mib": peak_vram,
            "elapsed_sec": elapsed,
            "arrays": stats,
        }
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"DONE Uni-Mol2-84M Tier-1 in {elapsed}s -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
