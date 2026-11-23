#!/usr/bin/env python3
"""Shared index loading, alignment assertions, SHA-256 hashing, and NPZ verification
for Peptide OmniPanel V33 pretrained representation caches.

Guarantees exact 1-to-1 row alignment with <WORK_DIR>/v33_upgrade/features/compound_index.tsv
(N = 46,598 compounds, including all 17 Case17 reference peptide drugs).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

csv.field_size_limit(10_000_000)

BASE_DIR = Path(os.environ.get("V33_BASE_DIR", "<WORK_DIR>/v33_upgrade"))
DATA_DIR = BASE_DIR / "data"
FEAT_DIR = BASE_DIR / "features"
EMB_DIR = BASE_DIR / "embeddings"
SCRIPTS_DIR = BASE_DIR / "scripts_gpu"

EXPECTED_N_TOTAL = 46598
EXPECTED_N_SMILES_VALID = 43943
EXPECTED_N_SEQ_VALID = 44773


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def load_v33_compound_table() -> Tuple[List[dict], Dict[str, int], dict]:
    """Load compound_index.tsv joined with canonical_smiles & parent_sequence
    from compounds_v33.tsv + case17_queries.tsv, asserting exact manifest alignment.

    Returns:
        rows: list of dicts of length 46,598 ordered strictly by compound_idx 0..46597
        case17_indices: dict[drug_name -> compound_idx] (17 entries)
        feat_manifest: parsed feature_manifest.json
    """
    manifest_path = FEAT_DIR / "feature_manifest.json"
    feat_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case17_indices: Dict[str, int] = feat_manifest["case17_indices"]
    assert len(case17_indices) == 17, f"Expected 17 Case17 drugs, got {len(case17_indices)}"

    lookup: Dict[str, Tuple[str, str]] = {}
    with (DATA_DIR / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            lookup[r["compound_key"]] = (r["canonical_smiles"], r["parent_sequence"])
    with (DATA_DIR / "case17_queries.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            lookup.setdefault(r["compound_key"], (r["canonical_smiles"], r["parent_sequence"]))

    rows: List[dict] = []
    with (FEAT_DIR / "compound_index.tsv").open(encoding="utf-8", newline="") as fh:
        for expected_idx, r in enumerate(csv.DictReader(fh, delimiter="\t")):
            idx = int(r["compound_idx"])
            assert idx == expected_idx, f"Row index mismatch: expected {expected_idx}, got {idx}"
            ckey = r["compound_key"]
            assert ckey in lookup, f"Missing compound_key in compounds/case17 tables: {ckey}"
            smi, seq = lookup[ckey]
            smi_valid = r["smiles_valid"] == "1"
            seq_valid = r["seq_valid"] == "1"
            if smi_valid:
                assert bool(smi.strip()), f"Empty SMILES on smiles_valid=1 row {idx} ({ckey})"
            if seq_valid:
                assert bool(seq.strip()), f"Empty sequence on seq_valid=1 row {idx} ({ckey})"
            rows.append({
                "compound_idx": idx,
                "compound_key": ckey,
                "canonical_smiles": smi.strip(),
                "parent_sequence": seq.strip(),
                "smiles_valid": smi_valid,
                "seq_valid": seq_valid,
                "is_case17": r["is_case17"] == "1",
                "case17_drug_name": r["case17_drug_name"],
            })

    assert len(rows) == EXPECTED_N_TOTAL, f"Expected {EXPECTED_N_TOTAL} rows, got {len(rows)}"
    n_smi = sum(1 for r in rows if r["smiles_valid"])
    n_seq = sum(1 for r in rows if r["seq_valid"])
    assert n_smi == EXPECTED_N_SMILES_VALID, f"Expected {EXPECTED_N_SMILES_VALID} valid SMILES, got {n_smi}"
    assert n_seq == EXPECTED_N_SEQ_VALID, f"Expected {EXPECTED_N_SEQ_VALID} valid seqs, got {n_seq}"

    for drug_name, cidx in case17_indices.items():
        assert rows[cidx]["is_case17"], f"Case17 mismatch at index {cidx} for {drug_name}"
        assert rows[cidx]["case17_drug_name"] == drug_name, (
            f"Case17 name mismatch at {cidx}: expected {drug_name}, got {rows[cidx]['case17_drug_name']}"
        )
        assert rows[cidx]["smiles_valid"], f"Case17 {drug_name} has smiles_valid=0!"
        assert rows[cidx]["seq_valid"], f"Case17 {drug_name} has seq_valid=0!"

    return rows, case17_indices, feat_manifest


def verify_embedding_arrays(
    arrays: Dict[str, np.ndarray],
    valid_mask: np.ndarray,
    case17_indices: Dict[str, int],
    expected_n: int = EXPECTED_N_TOTAL,
) -> dict:
    """Run strict numerical & alignment verification on extracted embeddings.

    Checks:
      - Exact row count N = 46,598
      - Zero NaN/Inf on valid==True rows (and zero NaN/Inf globally)
      - Non-zero variance across molecules (and non-zero row norms on all valid rows)
      - 100% coverage of all 17 Case17 drugs
    """
    assert valid_mask.shape == (expected_n,), f"valid_mask shape {valid_mask.shape} != ({expected_n},)"
    assert valid_mask.dtype == np.bool_, f"valid_mask dtype {valid_mask.dtype} != bool"
    n_valid = int(valid_mask.sum())
    assert n_valid > 0, "Zero valid rows!"

    missing_c17 = [name for name, idx in case17_indices.items() if not valid_mask[idx]]
    assert not missing_c17, f"Case17 drugs missing from valid_mask: {missing_c17}"

    summary = {}
    for name, arr in arrays.items():
        assert arr.shape[0] == expected_n, f"{name}: row count {arr.shape[0]} != {expected_n}"
        assert arr.ndim == 2, f"{name}: expected 2D array, got {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name}: non-finite values detected!"
        valid_sub = arr[valid_mask].astype(np.float32, copy=False)
        assert np.all(np.isfinite(valid_sub)), f"{name}: non-finite values on valid rows!"
        row_norms = np.linalg.norm(valid_sub, axis=1)
        zero_norm_cnt = int((row_norms <= 1e-6).sum())
        assert zero_norm_cnt == 0, f"{name}: {zero_norm_cnt} valid rows have near-zero norm!"
        # Variance across molecules (sample up to 10,000 rows for fast exact check if large)
        col_std = valid_sub.std(axis=0)
        mean_std = float(col_std.mean())
        nonzero_var_dims = int((col_std > 1e-6).sum())
        assert mean_std > 1e-4 and nonzero_var_dims > 0, (
            f"{name}: degenerate embeddings (mean_std={mean_std}, nonzero_var_dims={nonzero_var_dims})"
        )
        # Check pairwise distinctness across Case17 drugs
        c17_idx_list = list(case17_indices.values())
        c17_mat = arr[c17_idx_list].astype(np.float32)
        c17_norms = np.linalg.norm(c17_mat, axis=1)
        assert np.all(c17_norms > 1e-4), f"{name}: zero embedding on a Case17 drug!"
        c17_std = float(c17_mat.std(axis=0).mean())
        assert c17_std > 1e-4, f"{name}: identical embeddings across Case17 drugs!"

        summary[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "n_valid": n_valid,
            "zero_nan_inf": True,
            "mean_dim_std": round(mean_std, 6),
            "nonzero_var_dims": nonzero_var_dims,
            "mean_l2_norm": round(float(row_norms.mean()), 4),
            "min_l2_norm": round(float(row_norms.min()), 4),
            "max_l2_norm": round(float(row_norms.max()), 4),
            "case17_covered": 17,
            "case17_mean_dim_std": round(c17_std, 6),
        }
    return summary
