#!/usr/bin/env python3
"""Promote verified v33.3 scientific upgrades into authoritative `v33_manifest.json`.

Strictly promotes ONLY verified, zero-holdout-peeking upgrades:
  1. Step 1 (L1 Fix): Full 19-cluster-excluded + clean-stacker-OOF Case17 scorecard
     (replaces exact-only Case17 numbers; preserves old values in `v332_exact_only_reference`).
  2. Step 2B (Cluster-Weighted Conformal Calibration): Upgrades all 40 regression endpoints x 2 tracks
     (80 tracks) to Cluster-Weighted Mondrian AD-EB (29 tracks, pure JSON) + Cluster-Weighted
     Residual-Centered CQR (51 tracks, 12.06 MB portable LightGBM `.joblib`), recomputing exact
     Wilson 95% CIs and 3-tier `uq_reliability` (`low` / `moderate` / `calibrated`).
  3. Embeds auditable `v333_scientific_upgrade_summary` recording Steps 1, 2A, 2B, 3A, 3B
     (both positive breakthroughs and honest negative results).
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

V33_BASE = Path("${V33_BASE_DIR}")
BUNDLE_DIR = V33_BASE / "bundle"
SCI_DIR = V33_BASE / "v333_science"


def _wilson_ci95(k: int, n: int) -> Optional[Tuple[float, float]]:
    if n <= 0:
        return None
    z = 1.959963984540054
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _classify_uq_reliability(n_eval: int, cov80: float, cov90: float) -> Tuple[str, Optional[str]]:
    if n_eval < 25 or cov80 < 0.65 or cov90 < 0.78:
        reasons = []
        if n_eval < 25:
            reasons.append(f"small holdout calibration n={n_eval}")
        if cov80 < 0.65:
            reasons.append(f"holdout Cov80={100*cov80:.1f}% < 65%")
        if cov90 < 0.78:
            reasons.append(f"holdout Cov90={100*cov90:.1f}% < 78%")
        return "low", f"Low UQ reliability ({'; '.join(reasons)}); interpret CI bounds cautiously."
    if n_eval < 50 or cov80 < 0.72 or cov90 < 0.84:
        return "moderate", f"Moderate UQ reliability (n={n_eval}, Cov80={100*cov80:.1f}%, Cov90={100*cov90:.1f}%)."
    return "calibrated", None


def main() -> None:
    manifest_path = BUNDLE_DIR / "v33_manifest.json"
    backup_path = BUNDLE_DIR / "v33_manifest_v332_backup.json"
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
        print(f"[backup] Saved {backup_path}")

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    s1 = json.loads((SCI_DIR / "step1_case17_strict_results.json").read_text(encoding="utf-8"))
    s2b = json.loads((SCI_DIR / "v333_conformal_calibration.json").read_text(encoding="utf-8"))
    s2a = json.loads((SCI_DIR / "step2_censored_logit_cqr_results.json").read_text(encoding="utf-8"))
    s3 = json.loads((SCI_DIR / "step3_multitask_smalln_pk_results.json").read_text(encoding="utf-8"))

    # 1. Copy verified CQR calibrators into bundle/conformal_calibrators/ & bundle/v333_models.py
    src_cal_dir = SCI_DIR / "promoted_cqr_cw_calibrators"
    dst_cal_dir = BUNDLE_DIR / "conformal_calibrators"
    if dst_cal_dir.exists():
        shutil.rmtree(dst_cal_dir)
    shutil.copytree(src_cal_dir, dst_cal_dir)
    shutil.copy2(SCI_DIR / "v333_models.py", BUNDLE_DIR / "v333_models.py")
    n_cals = len(list(dst_cal_dir.glob("*.joblib")))
    cal_mb = sum(p.stat().st_size for p in dst_cal_dir.glob("*.joblib")) / (1024.0 * 1024.0)
    print(f"[conformal] Copied {n_cals} portable CQR calibrators ({cal_mb:.2f} MB) + v333_models.py into {BUNDLE_DIR}")

    # 2. Upgrade all 40 regression endpoints x 2 tracks conformal calibration & UQ reliability metadata
    rel_counts: Dict[str, int] = {"calibrated": 0, "moderate": 0, "low": 0}
    mode_counts: Dict[str, int] = {}
    for ep, ep_s2b in s2b["endpoints"].items():
        if ep not in man["endpoints"]:
            continue
        ed = man["endpoints"][ep]
        for tr in ("dual_modality", "smiles_only"):
            t_s2b = ep_s2b["tracks"][tr]
            spec = t_s2b["production_conformal_spec"]
            ho = t_s2b["selected_ho"]
            n_eval = int(ho["n_eval"])
            cov80 = float(ho["cov80"])
            cov90 = float(ho["cov90"])
            k80 = int(round(cov80 * n_eval))
            k90 = int(round(cov90 * n_eval))
            ci80 = _wilson_ci95(k80, n_eval)
            ci90 = _wilson_ci95(k90, n_eval)
            rel, note = _classify_uq_reliability(n_eval, cov80, cov90)
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
            mode_counts[spec["mode"]] = mode_counts.get(spec["mode"], 0) + 1

            ed[tr]["conformal"] = spec
            ed[tr]["conformal_holdout_coverage"] = {
                "n_eval": n_eval,
                "cov80": round(cov80, 4),
                "cov90": round(cov90, 4),
                "mean_width80": round(float(ho["w80"]), 4),
                "mean_width90": round(float(ho["w90"]), 4),
                "cov80_ci95_wilson": list(ci80) if ci80 else None,
                "cov90_ci95_wilson": list(ci90) if ci90 else None,
                "by_tier": ho.get("by_tier", {}),
                "uq_reliability": rel,
                "uq_note": note,
                "v332_baseline_cov80": round(float(t_s2b["ho_metrics"]["baseline"]["cov80"]), 4),
                "v332_baseline_cov90": round(float(t_s2b["ho_metrics"]["baseline"]["cov90"]), 4),
            }

    # 3. Update Case17 Scorecard with strict 19-cluster-excluded + clean-stacker-OOF results (Step 1)
    old_c17 = man.get("case17_scorecard", {})
    sc_cmp = s1["scorecard_comparison"]
    new_drugs = []
    old_drugs_by_name = {d["drug_name"]: d for d in old_c17.get("drugs", [])}
    s1_eps = s1["endpoints"]
    for i, drow in enumerate(s1["drugs"]):
        dn = drow["drug_name"]
        base_d = dict(old_drugs_by_name.get(dn, {}))
        base_d["v33_half_life_route_matched_h"] = drow["strict_hl_rm_h"]
        base_d["v33_half_life_iv_h"] = round(10.0 ** s1_eps["half_life_reg"]["strict_dual"][i], 3)
        if "SC" in s1_eps["half_life_reg"]["strict_alt_dual"]:
            base_d["v33_half_life_sc_h"] = round(10.0 ** s1_eps["half_life_reg"]["strict_alt_dual"]["SC"][i], 3)
        anch = base_d.get("v33_half_life_mechanism_anchor") or {}
        base_d["v33_half_life_hybrid_h"] = anch.get("anchor_mid_h") if anch.get("anchor_mid_h") is not None else drow["strict_hl_rm_h"]
        base_d["v33_ppb_dual"] = round(s1_eps["PPB"]["strict_dual"][i], 5)
        base_d["v33_ppb_smiles_only"] = round(s1_eps["PPB"]["strict_smi"][i], 5)
        if s1_eps["PPB"].get("strict_cens"):
            base_d["v33_ppb_censored"] = round(s1_eps["PPB"]["strict_cens"][i], 5)
        base_d["v33_hemolysis_prob_cal"] = round(s1_eps["hemolysis_cls"]["strict_dual"][i], 5)
        base_d["v33_hemolysis_hc50_uM"] = round(10.0 ** s1_eps["hemolysis_HC50"]["strict_dual"][i], 1)
        if s1_eps["hemolysis_HC50"].get("strict_cens"):
            base_d["v33_hemolysis_hc50_censored_uM"] = round(10.0 ** s1_eps["hemolysis_HC50"]["strict_cens"][i], 1)
        base_d["v33_toxicity_prob"] = round(s1_eps["overall_peptide_toxicity"]["strict_dual"][i], 5)
        base_d["v33_bbb_prob"] = round(s1_eps["BBB"]["strict_dual"][i], 5)
        base_d["v33_pampa_log10"] = round(s1_eps["PAMPA"]["strict_dual"][i], 5)
        base_d["v33_caco2_log10"] = round(s1_eps["Caco2"]["strict_dual"][i], 5)
        base_d["v33_logd74"] = round(s1_eps["LogD7.4"]["strict_dual"][i], 5)
        base_d["v33_cl_per_kg"] = round(10.0 ** s1_eps["CL_per_kg"]["strict_dual"][i], 3)
        base_d["v33_vd_per_kg"] = round(10.0 ** s1_eps["Vd_per_kg"]["strict_dual"][i], 3)
        base_d["v33_f_po"] = round(s1_eps["F"]["strict_dual"][i], 5)
        if "SC" in s1_eps["F"]["strict_alt_dual"]:
            base_d["v33_f_sc"] = round(s1_eps["F"]["strict_alt_dual"]["SC"][i], 5)
        base_d["v33_solubility_prob"] = round(s1_eps["solubility_cls"]["strict_dual"][i], 5)
        new_drugs.append(base_d)

    man["case17_scorecard"] = {
        "protocol": (
            "Strict Full-Cluster Leave-Case17-Out (v33.3 L1 Fix): ALL 19 Case17 clusters (911 rows, "
            "including 472 same-cluster analogs) excluded across CPU, Fusion, AUX meta-features, "
            "and Deep; Fusion c17_stacker fit strictly on non-Case17-cluster OOF."
        ),
        "half_life_pure_ml_route_matched_log10": sc_cmp["half_life_pure_ml_log10"]["strict_full_cluster"],
        "half_life_with_mechanism_anchor_log10": sc_cmp["half_life_with_mechanism_anchor_log10"]["strict_full_cluster"],
        "ppb_dual": sc_cmp["ppb_dual"]["strict_full_cluster"],
        "ppb_censored_head": sc_cmp["ppb_censored"]["strict_full_cluster"],
        "hemolysis_false_positive_check": {
            "max_prob_across_17_fda_drugs": sc_cmp["hemolysis_false_positive_check"]["strict_max_prob"],
            "n_drugs_gt_0_50": sc_cmp["hemolysis_false_positive_check"]["strict_n_gt_0_50"],
            "v32_n_drugs_gt_0_60": 8,
        },
        "v332_exact_only_reference": {
            "note": "Prior v33.0-v33.2 scorecard excluded only exact matches (439 rows) in CPU/Fusion.",
            "half_life_pure_ml_log10": sc_cmp["half_life_pure_ml_log10"]["old_exact_only"],
            "half_life_with_mechanism_anchor_log10": sc_cmp["half_life_with_mechanism_anchor_log10"]["old_exact_only"],
            "ppb_dual": sc_cmp["ppb_dual"]["old_exact_only"],
            "hemolysis_max_prob": sc_cmp["hemolysis_false_positive_check"]["old_max_prob"],
        },
        "drugs": new_drugs,
    }

    # 4. Update version & embed transparent v33.3 scientific summary
    man["version"] = "v33.3"
    man["v333_upgraded_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    man["serving_inventory"]["conformal_calibrators_dir"] = "conformal_calibrators"
    man["serving_inventory"]["conformal_calibrators_count"] = n_cals
    man["serving_inventory"]["conformal_calibrators_mb"] = round(cal_mb, 3)

    man["v333_scientific_upgrade_summary"] = {
        "step1_case17_strict_full_cluster_fix": {
            "clusters_excluded": s1["n_case17_clusters_excluded"],
            "scorecard_comparison": sc_cmp,
        },
        "step2b_cluster_weighted_conformal": {
            "summary": s2b["summary"],
            "mode_counts_80_tracks": mode_counts,
            "uq_reliability_counts_80_tracks": rel_counts,
        },
        "step2a_censored_logit_diagnostic": {
            "n_endpoints_evaluated": len(s2a["part2a_censored_logit"]),
            "ppb_case17_strict_logit_tobit": s2a["part2a_censored_logit"].get("PPB", {}).get("ppb_case17_strict_logit_tobit"),
            "finding": (
                "TobitKernelEM / Logit-Tobit breaks the ~0.72 linear PPB ceiling on lipidated GLP-1s "
                "and reduces censored Hinge MAE on cytotoxicity_CC50/MIC_E_faecalis/MIC_S_aureus/MBC_S_aureus, "
                "with minor exact-subset MAE trade-off; kept diagnostic while main models stay exact-optimal."
            ),
        },
        "step3_multitask_and_smalln_pk_diagnostic": {
            "lineage_mt_krr_wins_over_st_krr": sum(
                1 for r in s3["part3a_lineage_transfer"].values() if r["mt_gain_over_st_cv"] > 0
            ),
            "lineage_total_endpoints": len(s3["part3a_lineage_transfer"]),
            "strict_oof_promoted_over_wave5a": [
                ep for ep, r in s3["part3a_lineage_transfer"].items() if r["promoted_strict_oof"]
            ],
            "smalln_pk_zero_winners_curse_highlights": {
                "Vd_absolute": {
                    "baseline_ho_mae": s3["part3b_smalln_pk"]["Vd_absolute"]["baseline_ho"]["mae"],
                    "baseline_ho_r2": s3["part3b_smalln_pk"]["Vd_absolute"]["baseline_ho"]["r2"],
                    "allo_ens_cv_mae": s3["part3b_smalln_pk"]["Vd_absolute"]["candidates_cv"]["zero_tune_allo_ens"]["mae"],
                    "allo_ens_ho_mae": s3["part3b_smalln_pk"]["Vd_absolute"]["candidates_ho"]["zero_tune_allo_ens"]["mae"],
                    "allo_ens_ho_r2": s3["part3b_smalln_pk"]["Vd_absolute"]["candidates_ho"]["zero_tune_allo_ens"]["r2"],
                },
                "PPB": {
                    "baseline_ho_mae": s3["part3b_smalln_pk"]["PPB"]["baseline_ho"]["mae"],
                    "baseline_ho_r2": s3["part3b_smalln_pk"]["PPB"]["baseline_ho"]["r2"],
                    "logit_gpr_ho_mae": s3["part3b_smalln_pk"]["PPB"]["candidates_ho"]["gpr_native"]["mae"],
                    "logit_gpr_ho_spearman": s3["part3b_smalln_pk"]["PPB"]["candidates_ho"]["gpr_native"]["spearman"],
                    "logit_gpr_ho_r2": s3["part3b_smalln_pk"]["PPB"]["candidates_ho"]["gpr_native"]["r2"],
                },
                "Cmax": {
                    "baseline_ho_mae": s3["part3b_smalln_pk"]["Cmax"]["baseline_ho"]["mae"],
                    "tabpfn_ho_mae": s3["part3b_smalln_pk"]["Cmax"]["candidates_ho"]["tabpfn_native"]["mae"],
                    "tabpfn_ho_spearman": s3["part3b_smalln_pk"]["Cmax"]["candidates_ho"]["tabpfn_native"]["spearman"],
                },
            },
            "halflife_prot20_ablation": {
                "lgb_no_prot_ho_mae": s3["part3b_halflife_prot20"]["lgb_no_prot_ho"]["mae"],
                "lgb_with_prot20_ho_mae": s3["part3b_halflife_prot20"]["lgb_with_prot20_ho"]["mae"],
                "blend_weight_on_champion": s3["part3b_halflife_prot20"]["blend_weight"],
            },
        },
    }

    manifest_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Upgraded {manifest_path} to v33.3! Modes={mode_counts}, UQ_rel={rel_counts}")


if __name__ == "__main__":
    main()
