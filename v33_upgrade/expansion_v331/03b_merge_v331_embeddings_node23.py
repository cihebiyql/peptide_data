#!/usr/bin/env python3
"""
Wave 5A — Step 03b (runs on node23):
Concatenates READ-ONLY frozen V33 46,598-row embeddings with K-row local RTX 5080 delta embeddings
into <WORK_DIR>/v33_upgrade/expansion_v331/embeddings/{ncaa_repdistill_v2,peptideclm2,chemberta_mtr,esmc_600m_multiregion}.npz.
Asserts rows 0..46597 are BIT-FOR-BIT identical to V33 and compound_keys/valid masks match expansion_v331/features!
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

V33_EMB = Path("<WORK_DIR>/v33_upgrade/embeddings")
EXP_BASE = Path("<WORK_DIR>/v33_upgrade/expansion_v331")
DELTA_DIR = EXP_BASE / "embeddings_delta"
OUT_EMB = EXP_BASE / "embeddings"
FEAT_DIR = EXP_BASE / "features"


def main():
    t0 = time.time()
    OUT_EMB.mkdir(parents=True, exist_ok=True)

    pc = np.load(FEAT_DIR / "physchem_core.npz", allow_pickle=False)
    exp_keys = pc["compound_keys"]
    exp_smi_v = pc["smiles_valid_mask"].astype(np.bool_)
    exp_seq_v = pc["seq_valid_mask"].astype(np.bool_)
    N_TOTAL = len(exp_keys)
    N_V33 = 46598
    K = N_TOTAL - N_V33
    print(f"Merging V33 ({N_V33:,}) + Delta ({K:,}) -> Total ({N_TOTAL:,}) embeddings...", flush=True)

    specs = [
        ("ncaa_repdistill_v2.npz", "delta_ncaa_repdistill_v2.npz", ["smiles_repr", "fusion_hf", "protein_proj", "valid", "chunked", "compound_keys"], exp_smi_v),
        ("peptideclm2.npz", "delta_peptideclm2.npz", ["embeddings", "valid", "chunked", "compound_keys"], exp_smi_v),
        ("chemberta_mtr.npz", "delta_chemberta_mtr.npz", ["embeddings", "valid", "chunked", "compound_keys"], exp_smi_v),
        ("esmc_600m_multiregion.npz", "delta_esmc_600m_multiregion.npz", ["mean", "max", "nterm5", "cterm5", "valid", "compound_keys"], exp_seq_v),
    ]

    summary = {}
    for base_name, delta_name, fields, expected_valid in specs:
        v33_d = np.load(V33_EMB / base_name, allow_pickle=False)
        del_d = np.load(DELTA_DIR / delta_name, allow_pickle=False)
        merged = {}
        for f in fields:
            arr = np.concatenate([v33_d[f], del_d[f]], axis=0)
            assert np.array_equal(arr[:N_V33], v33_d[f]), f"Prefix mismatch on {base_name}:{f}!"
            assert arr.shape[0] == N_TOTAL, f"Shape mismatch on {base_name}:{f}: {arr.shape}"
            merged[f] = arr
        assert np.array_equal(merged["compound_keys"], exp_keys), f"compound_keys mismatch in {base_name}!"
        assert np.array_equal(merged["valid"].astype(np.bool_), expected_valid), f"valid mask mismatch in {base_name}!"
        out_fp = OUT_EMB / base_name
        np.savez_compressed(out_fp, **merged)
        summary[base_name] = {
            "shape_first": list(merged[fields[0]].shape),
            "n_valid": int(merged["valid"].sum()),
            "size_mb": round(out_fp.stat().st_size / 1e6, 2),
        }
        print(f"  [OK] {base_name}: N={N_TOTAL:,}, valid={summary[base_name]['n_valid']:,}, size={summary[base_name]['size_mb']} MB", flush=True)

    (OUT_EMB / "merge_summary.json").write_text(json.dumps({"elapsed_sec": round(time.time() - t0, 2), "files": summary}, indent=2), encoding="utf-8")
    print(f"DONE merging 4-FM embeddings in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
