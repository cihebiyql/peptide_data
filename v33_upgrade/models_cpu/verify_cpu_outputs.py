#!/usr/bin/env python3
"""Independent verification gate for all 24 V33 CPU endpoint artifacts on node23."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np

from v33_config import ENDPOINT_SPECS, ENDPOINTS_DIR, OUT_DIR, SEQ_ONLY_ENDPOINTS
from v33_features import FeatureBank

csv.field_size_limit(10_000_000)


def main() -> None:
    bank = FeatureBank()
    summary_path = OUT_DIR / "cpu_benchmark_summary.json"
    assert summary_path.exists(), "Missing cpu_benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_endpoints"] == 24, f"Expected 24 endpoints, got {summary['total_endpoints']}"

    total_oof = 0
    total_ho = 0
    total_joblib_mb = 0.0
    failures = []

    for ep, spec in sorted(ENDPOINT_SPECS.items()):
        ep_dir = OUT_DIR / ep
        req_files = [
            "oof_predictions.tsv",
            "holdout_predictions.tsv",
            "case17_predictions.tsv",
            "metrics.json",
            "models_5fold.joblib",
        ]
        for rf in req_files:
            assert (ep_dir / rf).exists(), f"{ep}: missing {rf}"

        # Load ground-truth endpoint TSV
        with (ENDPOINTS_DIR / f"{ep}.tsv").open(encoding="utf-8") as fh:
            gt_rows = list(csv.DictReader(fh, delimiter="\t"))
        gt_cv = [r for r in gt_rows if r["is_holdout"] == "0"]
        gt_ho = [r for r in gt_rows if r["is_holdout"] == "1"]

        # Verify exact OOF row alignment
        with (ep_dir / "oof_predictions.tsv").open(encoding="utf-8") as fh:
            oof_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(oof_rows) == len(gt_cv), f"{ep}: OOF rows {len(oof_rows)} != {len(gt_cv)}"
        for i, (o, g) in enumerate(zip(oof_rows, gt_cv)):
            assert o["record_id"] == g["record_id"], f"{ep}: OOF record_id mismatch at {i}"
            assert o["compound_key"] == g["compound_key"], f"{ep}: OOF compound_key mismatch at {i}"
            assert o["pred_dual"] != "" and np.isfinite(float(o["pred_dual"])), f"{ep}: OOF pred_dual NaN at {i}"
            cidx = bank.key_to_idx[g["compound_key"]]
            if bank.smiles_valid[cidx]:
                assert o["pred_smiles_only"] != "" and np.isfinite(float(o["pred_smiles_only"])), (
                    f"{ep}: OOF pred_smiles_only NaN on smiles_valid=1 row {i}"
                )

        # Verify exact Holdout row alignment
        with (ep_dir / "holdout_predictions.tsv").open(encoding="utf-8") as fh:
            ho_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(ho_rows) == len(gt_ho), f"{ep}: HO rows {len(ho_rows)} != {len(gt_ho)}"
        for i, (h, g) in enumerate(zip(ho_rows, gt_ho)):
            assert h["record_id"] == g["record_id"], f"{ep}: HO record_id mismatch at {i}"
            assert h["compound_key"] == g["compound_key"], f"{ep}: HO compound_key mismatch at {i}"
            pd_val = float(h["pred_dual"])
            assert np.isfinite(pd_val), f"{ep}: HO pred_dual NaN at {i}"
            if spec["bounds"] is not None:
                lo_b, hi_b = spec["bounds"]
                assert lo_b - 1e-6 <= pd_val <= hi_b + 1e-6, f"{ep}: HO pred_dual {pd_val} OOB {spec['bounds']}"
            cidx = bank.key_to_idx[g["compound_key"]]
            if bank.smiles_valid[cidx]:
                ps_val = float(h["pred_smiles_only"])
                assert np.isfinite(ps_val), f"{ep}: HO pred_smiles_only NaN at {i}"

        # Verify Case17 17/17 finite predictions on BOTH tracks
        with (ep_dir / "case17_predictions.tsv").open(encoding="utf-8") as fh:
            c17_rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(c17_rows) == 17, f"{ep}: Case17 rows {len(c17_rows)} != 17"
        for r in c17_rows:
            assert r["pred_smiles_only"] != "" and np.isfinite(float(r["pred_smiles_only"])), (
                f"{ep}: Case17 {r['drug_name']} missing smiles_only pred"
            )
            assert r["pred_dual"] != "" and np.isfinite(float(r["pred_dual"])), (
                f"{ep}: Case17 {r['drug_name']} missing dual pred"
            )

        # Verify joblib loads cleanly & check size
        jb_path = ep_dir / "models_5fold.joblib"
        total_joblib_mb += jb_path.stat().st_size / (1024 * 1024)
        bundle = joblib.load(jb_path)
        assert len(bundle["smiles_only"]["folds"]) == 5, f"{ep}: smiles_only folds != 5"
        assert len(bundle["dual_modality"]["folds"]) == 5, f"{ep}: dual folds != 5"
        if ep in SEQ_ONLY_ENDPOINTS:
            assert bundle["seq_only"] is not None and len(bundle["seq_only"]["folds"]) == 5

        total_oof += len(oof_rows)
        total_ho += len(ho_rows)

    print(
        f"VERIFICATION PASSED: 24/24 endpoints | OOF={total_oof} + HO={total_ho} = {total_oof+total_ho} rows "
        f"(expected 69,111) | 24x17=408 Case17 leave-out predictions | Total joblib size={total_joblib_mb:.1f} MiB"
    )
    assert total_oof + total_ho == 69111


if __name__ == "__main__":
    main()
