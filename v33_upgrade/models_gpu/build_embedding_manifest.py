#!/usr/bin/env python3
"""Independent verification & manifest builder for all V33 pretrained embedding caches.

Re-loads every extracted .npz archive from <WORK_DIR>/v33_upgrade/embeddings/,
independently asserts:
  1. Exact row count == 46,598 and exact 1-to-1 compound_keys match with compound_index.tsv
  2. Zero NaN/Inf globally and on valid==True rows
  3. Non-zero variance across molecules and non-zero L2 norms on all valid rows
  4. 100% coverage of all 17 Case17 reference peptide drugs
  5. Computes fresh SHA-256 hashes and consolidates .meta.json + gpu_compliance_log.jsonl

Writes: <WORK_DIR>/v33_upgrade/embeddings/embedding_manifest.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common_index import EMB_DIR, FEAT_DIR, load_v33_compound_table, sha256_file, verify_embedding_arrays

EXPECTED_CORE_CACHES = [
    ("ncaa_repdistill_v2", ["smiles_repr", "fusion_hf", "protein_proj"]),
    ("peptideclm2", ["embeddings"]),
    ("esmc_600m_multiregion", ["mean", "max", "nterm5", "cterm5"]),
    ("chemberta_mtr", ["embeddings"]),
]
OPTIONAL_CACHES = [
    ("unimol2_84m_tier1", ["embeddings"]),
    ("unimol2_84m", ["embeddings"]),
]


def main():
    rows, case17_indices, feat_manifest = load_v33_compound_table()
    expected_keys = np.array([r["compound_key"] for r in rows])

    caches_manifest = {}
    for name, array_keys in EXPECTED_CORE_CACHES + OPTIONAL_CACHES:
        npz_path = EMB_DIR / f"{name}.npz"
        if not npz_path.exists():
            if any(name == c[0] for c in EXPECTED_CORE_CACHES):
                raise FileNotFoundError(f"Required core embedding cache missing: {npz_path}")
            continue

        print(f"[VERIFY] Loading {npz_path} ...", flush=True)
        data = np.load(npz_path, allow_pickle=False)
        assert "compound_keys" in data, f"{name}: missing compound_keys"
        assert "valid" in data, f"{name}: missing valid mask"
        assert np.array_equal(data["compound_keys"], expected_keys), (
            f"{name}: compound_keys do NOT match compound_index.tsv 1-to-1!"
        )
        valid_mask = data["valid"]
        arrays = {k: data[k] for k in array_keys}
        stats = verify_embedding_arrays(arrays, valid_mask, case17_indices)

        meta_path = npz_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        caches_manifest[name] = {
            "file": str(npz_path),
            "sha256": sha256_file(npz_path),
            "size_bytes": npz_path.stat().st_size,
            "size_mb": round(npz_path.stat().st_size / (1024 * 1024), 2),
            "n_rows": int(len(expected_keys)),
            "n_valid": int(valid_mask.sum()),
            "valid_pct": round(100.0 * float(valid_mask.sum()) / len(expected_keys), 2),
            "case17_covered": 17,
            "arrays": stats,
            "meta": meta,
        }
        print(
            f"  [PASS] {name}: {npz_path.stat().st_size/1e6:.2f} MB, valid={int(valid_mask.sum()):,}/46,598, "
            f"arrays={ {k: v['shape'] for k, v in stats.items()} }",
            flush=True,
        )

    # Load GPU compliance log
    gpu_log_path = EMB_DIR / "gpu_compliance_log.jsonl"
    gpu_checks = []
    if gpu_log_path.exists():
        for line in gpu_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                gpu_checks.append(json.loads(line))

    # Load optional Uni-Mol2 benchmark report if present
    unimol_bench_path = EMB_DIR / "unimol2_benchmark.json"
    unimol_bench = json.loads(unimol_bench_path.read_text(encoding="utf-8")) if unimol_bench_path.exists() else None

    manifest = {
        "schema": "peptide_omnipanel_v33_pretrained_embeddings_manifest_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "compound_index_file": str(FEAT_DIR / "compound_index.tsv"),
        "compound_index_sha256": feat_manifest["compound_index"]["sha256"],
        "n_compounds_total": 46598,
        "n_smiles_valid": 43943,
        "n_seq_valid": 44773,
        "case17_indices": case17_indices,
        "caches": caches_manifest,
        "unimol2_benchmark": unimol_bench,
        "gpu_compliance": {
            "rule": "max_2_gpus, 2 consecutive nvidia-smi samples (5-10s apart) confirming VRAM < 50% (<12,282 MiB) AND Util < 50%",
            "log_file": str(gpu_log_path),
            "n_checks": len(gpu_checks),
            "checks": gpu_checks,
        },
        "verification_status": "ALL_CHECKS_PASSED",
    }

    out_manifest = EMB_DIR / "embedding_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Wrote verified manifest -> {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
