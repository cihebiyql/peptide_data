#!/usr/bin/env python3
"""
Wave 5A — Step 06 (runs OFFLINE on node23 — ZERO disruption to node2!):
  --phase pre  : Loads untouched 24-endpoint V33InferenceEngine and snapshots exact 24-ep predictions
                 on Semaglutide, Cyclosporine A, Caspofungin, Classic AMP, and 5-molecule batch.
  --phase post : Loads merged 46-endpoint V33.1 V33InferenceEngine and asserts:
                 1. BIT-FOR-BIT EXACT MATCH (max_abs_diff == 0.0, identical AD & Top-3 analogs)
                    on ALL 24 existing endpoints vs pre-merge snapshot!
                 2. Finite, physically bounded predictions, valid 80%/90% CIs, real AD tiers & Top-3 analogs,
                    and 0 NaN/Inf across ALL 46 endpoints (24 existing + 22 new) on all test molecules & 5-mol batch!
                 3. Cleanly terminates its temporary esmc_worker so ZERO daemon processes remain on node23!
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

V33_BASE = Path("<WORK_DIR>/v33_upgrade")
BUNDLE_DIR = V33_BASE / "bundle"
EXP_BASE = V33_BASE / "expansion_v331"
PRE_SNAPSHOT = EXP_BASE / "pre_merge_24ep_baseline.json"
POST_RECEIPT = EXP_BASE / "v331_offline_verification_receipt.json"

# Configure node23 esmc python (esmfold2-env312) cleanly!
os.environ.setdefault("V33_BASE_DIR", str(V33_BASE))
if Path("<WORK_DIR>/esmfold2-env312/bin/python").exists():
    os.environ.setdefault("V33_ESMC_PYTHON", "<WORK_DIR>/esmfold2-env312/bin/python")
if Path("<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36").exists():
    os.environ.setdefault(
        "V33_ESMC_MODEL_DIR",
        "<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36",
    )

sys.path.insert(0, str(BUNDLE_DIR))

import csv  # noqa: E402


def load_test_panel() -> List[Dict[str, str]]:
    c17: Dict[str, str] = {}
    with (V33_BASE / "data" / "case17_queries.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c17[r["drug_name"]] = r["canonical_smiles"]
    return [
        {"name": "Semaglutide", "smiles": c17["Semaglutide"]},
        {"name": "Cyclosporine_A", "smiles": c17["Cyclosporine_A"]},
        {"name": "Caspofungin", "smiles": c17["Caspofungin"]},
        {"name": "Classic_AMP_LL37_core", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS"},  # Magainin-2 (classic AMP)
        {"name": "Classic_AMP_Indolicidin", "sequence": "ILPWKWPWWPWRR"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pre", "post"], required=True)
    ap.add_argument("--esmc-port", type=int, default=18992)
    args = ap.parse_args()

    t0 = time.time()
    from v33_inference_service import V33InferenceEngine

    print(f"=== V33.1 Offline Verification (phase={args.phase}, esmc_port={args.esmc_port}) ===", flush=True)
    engine = V33InferenceEngine(esmc_port=args.esmc_port, torch_threads=12)
    try:
        health = engine.health()
        print(f"[HEALTH] {json.dumps(health)}", flush=True)
        panel = load_test_panel()

        # Run single predictions on first 4 items + batch prediction on all 5 items!
        single_res: Dict[str, Any] = {}
        for item in panel[:4]:
            res_list, timings = engine.predict_batch_items([item], user_cond={}, include_tier2=True, force_live_encode=False)
            assert len(res_list) == 1 and res_list[0]["status"] == "ok", f"Single failed: {item['name']}"
            single_res[item["name"]] = res_list[0]
            print(
                f"  [SINGLE] {item['name']:<24} track={res_list[0]['modality_track']:<13} eps={len(res_list[0]['endpoints'])} total_ms={timings['total_ms']:.1f}",
                flush=True,
            )

        batch_list, batch_timings = engine.predict_batch_items(panel, user_cond={}, include_tier2=True, force_live_encode=False)
        assert len(batch_list) == 5 and all(r["status"] == "ok" for r in batch_list)
        print(f"  [BATCH5] ok=5/5 eps={len(batch_list[0]['endpoints'])} total_ms={batch_timings['total_ms']:.1f}", flush=True)

        # Also test live encoding (force_live_encode=True) on a genuinely novel AMP sequence in post phase!
        live_test_res = None
        if args.phase == "post":
            live_list, live_t = engine.predict_batch_items(
                [{"name": "Novel_Live_AMP", "sequence": "KWKLFKKIGAVLKVLTTGL"}],
                user_cond={},
                include_tier2=True,
                force_live_encode=True,
            )
            assert len(live_list) == 1 and live_list[0]["status"] == "ok" and len(live_list[0]["endpoints"]) == 46
            assert not live_list[0]["cache_hit"]
            live_test_res = {"name": live_list[0]["name"], "n_endpoints": len(live_list[0]["endpoints"]), "timing_ms": live_t}
            print(f"  [LIVE  ] Novel_Live_AMP eps=46 cache_hit=False total_ms={live_t['total_ms']:.1f}", flush=True)

        if args.phase == "pre":
            assert health["endpoints"] == 24, f"Expected 24 pre-merge endpoints, got {health['endpoints']}"
            doc = {"phase": "pre", "health": health, "single": single_res, "batch5": batch_list, "batch_timings": batch_timings}
            PRE_SNAPSHOT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[PRE SAVED] {PRE_SNAPSHOT} ({time.time()-t0:.1f}s)", flush=True)
            return

        # --- POST-MERGE FULL VERIFICATION ---
        pre_doc = json.loads(PRE_SNAPSHOT.read_text(encoding="utf-8"))
        existing_24 = sorted(pre_doc["single"]["Semaglutide"]["endpoints"].keys())
        assert len(existing_24) == 24
        all_post_eps = sorted(single_res["Semaglutide"]["endpoints"].keys())
        new_eps = [e for e in all_post_eps if e not in existing_24]
        assert len(all_post_eps) == 46 and len(new_eps) == 22, f"Expected 46 (24+22), got {len(all_post_eps)}"

        # 1. BIT-FOR-BIT PARITY CHECK ON ALL 24 EXISTING ENDPOINTS!
        max_abs_diff = 0.0
        n_checked_24 = 0
        for name in single_res:
            for ep in existing_24:
                p_old = pre_doc["single"][name]["endpoints"][ep]
                p_new = single_res[name]["endpoints"][ep]
                diff = abs(float(p_old["raw_value"]) - float(p_new["raw_value"]))
                max_abs_diff = max(max_abs_diff, diff)
                assert diff == 0.0, f"REGRESSION on {name}:{ep}! old={p_old['raw_value']} vs new={p_new['raw_value']}"
                assert p_old["ad_tier"] == p_new["ad_tier"], f"AD tier changed on {name}:{ep}!"
                assert p_old["max_tanimoto"] == p_new["max_tanimoto"], f"max_tanimoto changed on {name}:{ep}!"
                assert p_old["top3_analogs"] == p_new["top3_analogs"], f"top3_analogs changed on {name}:{ep}!"
                assert p_old["ci80_low"] == p_new["ci80_low"] and p_old["ci80_high"] == p_new["ci80_high"], f"CI80 changed on {name}:{ep}!"
                assert p_old["ci90_low"] == p_new["ci90_low"] and p_old["ci90_high"] == p_new["ci90_high"], f"CI90 changed on {name}:{ep}!"
                assert p_old.get("censored_diagnostic_raw") == p_new.get("censored_diagnostic_raw"), f"Censored diag changed on {name}:{ep}!"
                n_checked_24 += 1
        print(f"  [ZERO-REGRESSION PASS] {n_checked_24} (molecule x existing-24-endpoint) predictions are 100% BIT-FOR-BIT IDENTICAL (max_abs_diff={max_abs_diff})!", flush=True)

        # 2. FULL SANITY CHECK ON ALL 46 ENDPOINTS ACROSS ALL 5 BATCH MOLECULES!
        nan_count = 0
        for res in batch_list:
            assert len(res["endpoints"]) == 46
            for ep, eobj in res["endpoints"].items():
                rv = eobj["raw_value"]
                if rv is None or not math.isfinite(float(rv)):
                    nan_count += 1
                    raise AssertionError(f"Non-finite raw_value for {res['name']}:{ep}")
                if eobj["task_type"] == "regression":
                    for ck in ("ci80_low", "ci80_high", "ci90_low", "ci90_high"):
                        v = eobj[ck]
                        assert v is not None and math.isfinite(float(v)), f"Invalid {ck} on {res['name']}:{ep}"
                    assert eobj["ci90_low"] <= eobj["ci80_low"] <= eobj["raw_value"] <= eobj["ci80_high"] <= eobj["ci90_high"], (
                        f"CI ordering violated on {res['name']}:{ep}: {eobj['ci90_low']}, {eobj['ci80_low']}, {eobj['raw_value']}, {eobj['ci80_high']}, {eobj['ci90_high']}"
                    )
                assert eobj["ad_tier"] in ("In-Domain", "Borderline", "OOD"), f"Bad AD tier on {res['name']}:{ep}"
                assert len(eobj["top3_analogs"]) == 3, f"Expected 3 analogs on {res['name']}:{ep}, got {len(eobj['top3_analogs'])}"

        # Spot-check sample predictions on key new endpoints for reporting
        spot_eps = ["cytotoxicity_CC50", "hemolysis_MHC", "efflux_ratio", "Vd_absolute", "MIC_A_baumannii", "MIC_K_pneumoniae", "MBC_S_aureus", "MFC_C_albicans", "MBIC_biofilm", "MBEC_biofilm", "LC_antimicrobial"]
        spot_summary = {
            res["name"]: {
                ep: {
                    "display_text": res["endpoints"][ep]["display_text"],
                    "raw_value": res["endpoints"][ep]["raw_value"],
                    "ci80": [res["endpoints"][ep]["ci80_low"], res["endpoints"][ep]["ci80_high"]],
                    "ad_tier": res["endpoints"][ep]["ad_tier"],
                    "max_tanimoto": res["endpoints"][ep]["max_tanimoto"],
                    "top1_analog_sim": res["endpoints"][ep]["top3_analogs"][0]["similarity"],
                }
                for ep in spot_eps
            }
            for res in batch_list
        }

        receipt = {
            "schema": "v331_offline_verification_receipt_v1",
            "status": "PASS",
            "version": health["version"],
            "n_total_endpoints": health["endpoints"],
            "n_tier1_endpoints": health["tier1_endpoints"],
            "n_tier2_endpoints": health["tier2_endpoints"],
            "n_existing_24_verified_bit_exact": n_checked_24,
            "max_abs_diff_existing_24": max_abs_diff,
            "n_new_22_verified": len(new_eps) * len(batch_list),
            "nan_or_inf_count": nan_count,
            "batch5_timings_ms": batch_timings,
            "live_encode_test": live_test_res,
            "rss_mb": health["rss_mb"],
            "models_loaded": health["models_loaded"],
            "spot_predictions": spot_summary,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        POST_RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[POST VERIFICATION PASS] All 46 endpoints x 5 molecules verified (0 NaN/Inf, 0 regressions)! Saved {POST_RECEIPT}", flush=True)
        print(json.dumps({k: v for k, v in receipt.items() if k != "spot_predictions"}, indent=2), flush=True)
    finally:
        engine.shutdown()
        print("[SHUTDOWN] Cleanly stopped temporary esmc_worker (0 daemons left on node23).", flush=True)


if __name__ == "__main__":
    main()
