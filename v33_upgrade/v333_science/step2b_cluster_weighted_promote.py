#!/usr/bin/env python3
"""v33.3 Step 2B Production Conformal Selector & Exporter (runs on compute-node CPU).

Fixes the OOF-selection mismatch discovered in Step 2:
  - Previously, `_oof_coverage_efficiency_score` evaluated nested OOF coverage row-unweighted,
    which biased selection toward dense-cluster-dominated unweighted `mondrian_ad_eb` (HO 77.33%).
  - Here, nested 5-fold OOF coverage and AD-tier coverage are evaluated CLUSTER-WEIGHTED
    (`w_i = 1 / m_{c(i)}`), exactly matching the cluster-disjoint holdout / new-scaffold deployment
    distribution — STRICTLY ZERO HOLDOUT PEEKING!
  - Compares pure-JSON `mondrian_ad_eb_cw` vs compact `cqr_lgb_cw` (plus baseline & unweighted
    references) on strictly nested 5-fold cluster-weighted OOF, exports prod-node-ready JSON spec +
    compact verified `.joblib` calibrators (<5 MB total), and evaluates ONCE on untouched 20% Holdout.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

V33_BASE = Path("${V33_BASE_DIR}")
EXP_BASE = V33_BASE / "expansion_v331"
BUNDLE_DIR = V33_BASE / "bundle"
SMALLN_DIR = V33_BASE / "v2_upgrade" / "smalln_cqr"
OUT_DIR = V33_BASE / "v333_science"
CAL_OUT_DIR = OUT_DIR / "promoted_cqr_cw_calibrators"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SMALLN_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))

import run_plan_b1_conformal as b1_conf  # noqa: E402
from run_plan_b1_conformal import (  # noqa: E402
    _apply_baseline_manifest_conformal,
    _compute_oof_and_ho_tanimoto,
    _eval_interval_metrics,
    _extract_phys_block,
    _fit_baseline_conformal_on_subset,
)
from run_plan_b1_smalln import UnifiedSmallNBank  # noqa: E402
from smalln_models import apply_mondrian_ad_eb, assign_ad_tiers  # noqa: E402
from v333_models import ClusterWeightedCQRCalibrator, fit_mondrian_ad_eb_cw  # noqa: E402

csv.field_size_limit(10_000_000)

G_MANIFEST: Optional[Dict[str, Any]] = None


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _eval_cluster_weighted_interval_metrics(
    y: np.ndarray,
    ivs: Dict[str, np.ndarray],
    max_tanimoto: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate interval coverage & width weighted by 1 / m_{c(i)} (each cluster gets weight 1.0)."""
    ok = (
        np.isfinite(y)
        & np.isfinite(ivs["lo80"])
        & np.isfinite(ivs["hi80"])
        & np.isfinite(ivs["lo90"])
        & np.isfinite(ivs["hi90"])
    )
    if not ok.any():
        return {"n_eval": 0, "n_clusters": 0, "cov80": None, "cov90": None, "w80": None, "w90": None, "by_tier": {}}

    yt = y[ok]
    cids = np.asarray(cluster_ids, dtype=object)[ok]
    counts: Dict[str, int] = {}
    for c in cids:
        cs = str(c)
        counts[cs] = counts.get(cs, 0) + 1
    w = np.array([1.0 / float(counts[str(c)]) for c in cids], dtype=np.float64)
    w_tot = float(np.sum(w))

    c80 = ((yt >= ivs["lo80"][ok]) & (yt <= ivs["hi80"][ok])).astype(np.float64)
    c90 = ((yt >= ivs["lo90"][ok]) & (yt <= ivs["hi90"][ok])).astype(np.float64)
    w80 = ivs["hi80"][ok] - ivs["lo80"][ok]
    w90 = ivs["hi90"][ok] - ivs["lo90"][ok]
    tiers = assign_ad_tiers(max_tanimoto[ok])

    by_tier = {}
    for tn in ("In-Domain", "Borderline", "OOD"):
        m = tiers == tn
        if m.any():
            wt = w[m]
            wsum = float(np.sum(wt))
            by_tier[tn] = {
                "n": int(m.sum()),
                "n_clusters_mass": round(wsum, 2),
                "cov80": round(float(np.sum(wt * c80[m]) / wsum), 4),
                "cov90": round(float(np.sum(wt * c90[m]) / wsum), 4),
                "w80": round(float(np.sum(wt * w80[m]) / wsum), 4),
                "w90": round(float(np.sum(wt * w90[m]) / wsum), 4),
            }

    return {
        "n_eval": int(ok.sum()),
        "n_clusters": len(counts),
        "cov80": round(float(np.sum(w * c80) / w_tot), 4),
        "cov90": round(float(np.sum(w * c90) / w_tot), 4),
        "w80": round(float(np.sum(w * w80) / w_tot), 4),
        "w90": round(float(np.sum(w * w90) / w_tot), 4),
        "by_tier": by_tier,
    }


def _cw_oof_score(m: Dict[str, Any], ref_w80: float, ref_w90: float) -> Tuple[float, Dict[str, float]]:
    """Cluster-weighted OOF coverage-efficiency score (lower is better; penalizes under-coverage strongly)."""
    c80 = float(m["cov80"]) if m["cov80"] is not None else 0.0
    c90 = float(m["cov90"]) if m["cov90"] is not None else 0.0
    w80 = float(m["w80"]) if m["w80"] is not None else ref_w80 * 2.0
    w90 = float(m["w90"]) if m["w90"] is not None else ref_w90 * 2.0

    pen_marg = (
        3.0 * max(0.0, 0.79 - c80)
        + 0.8 * max(0.0, c80 - 0.86)
        + 3.0 * max(0.0, 0.89 - c90)
        + 0.8 * max(0.0, c90 - 0.95)
    )
    k_tot = max(1.0, float(m.get("n_clusters", 1)))
    pen_ad = 0.0
    for tn, td in m.get("by_tier", {}).items():
        cmass = float(td.get("n_clusters_mass", 0.0))
        if cmass >= 4.0:
            wt = cmass / k_tot
            mult = 1.5 if tn in ("Borderline", "OOD") else 0.7
            pen_ad += mult * wt * (max(0.0, 0.77 - td["cov80"]) + max(0.0, 0.87 - td["cov90"]))

    rel_w = 0.5 * (w80 / max(ref_w80, 1e-4) + w90 / max(ref_w90, 1e-4))
    total = 14.0 * pen_marg + 10.0 * pen_ad + rel_w
    return float(total), {
        "pen_marg": round(pen_marg, 5),
        "pen_ad": round(pen_ad, 5),
        "rel_width": round(rel_w, 5),
        "total_score": round(total, 5),
    }


def _check_portable(path: Path) -> List[str]:
    obj = joblib.load(path)
    mods: Set[str] = set()
    seen: Set[int] = set()

    def walk(x: Any):
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        mod = getattr(type(x), "__module__", "")
        if mod:
            mods.add(mod.split(".")[0])
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
        elif hasattr(x, "__dict__"):
            for v in x.__dict__.values():
                walk(v)

    walk(obj)
    for forbidden in ("catboost", "tabpfn", "torch"):
        assert forbidden not in mods, f"FORBIDDEN {forbidden} in {path}: {sorted(mods)}"
    return sorted(mods)


def _run_one_ep(ep: str) -> Dict[str, Any]:
    assert G_MANIFEST is not None and b1_conf.GLOBAL_KEY_TO_IDX is not None
    ep_doc = G_MANIFEST["endpoints"][ep]
    wave = ep_doc.get("expansion_wave", "v33.0")
    ep_dir = EXP_BASE / "data/endpoints" if wave == "v33.1" else V33_BASE / "data/endpoints"
    rows = _read_tsv(ep_dir / f"{ep}.tsv")
    cv_rows = [r for r in rows if int(r["is_holdout"]) == 0]
    ho_rows = [r for r in rows if int(r["is_holdout"]) == 1]

    cidx_cv = np.array([b1_conf.GLOBAL_KEY_TO_IDX[r["compound_key"]] for r in cv_rows], dtype=np.int32)
    cidx_ho = np.array([b1_conf.GLOBAL_KEY_TO_IDX[r["compound_key"]] for r in ho_rows], dtype=np.int32)
    fids_cv = np.array([int(r["fold_id"]) for r in cv_rows], dtype=np.int32)
    cids_cv = np.array([r["cluster_id"] for r in cv_rows], dtype=object)
    y_cv = np.array([float(r["y_value"]) for r in cv_rows], dtype=np.float64)
    y_ho = np.array([float(r["y_value"]) for r in ho_rows], dtype=np.float64)
    ct_cv = np.array([r["censor_type"] for r in cv_rows], dtype=object)
    ct_ho = np.array([r["censor_type"] for r in ho_rows], dtype=object)
    bounds_raw = ep_doc.get("bounds")
    bounds = (float(bounds_raw[0]), float(bounds_raw[1])) if bounds_raw is not None else None

    oof_max_t, oof_top5_t, ho_max_t, ho_top5_t = _compute_oof_and_ho_tanimoto(cidx_cv, fids_cv, cidx_ho)

    if wave == "v33.0":
        coof = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "oof_champion.tsv")
        cho = _read_tsv(BUNDLE_DIR / "champion_predictions" / ep / "holdout_champion.tsv")
        pred_oof_map = {
            "dual_modality": np.array([float(r["pred_dual"]) if r["pred_dual"] != "" else np.nan for r in coof]),
            "smiles_only": np.array([float(r["pred_smiles_only"]) if r["pred_smiles_only"] != "" else np.nan for r in coof]),
        }
        pred_ho_map = {
            "dual_modality": np.array([float(r["pred_dual"]) if r["pred_dual"] != "" else np.nan for r in cho]),
            "smiles_only": np.array([float(r["pred_smiles_only"]) if r["pred_smiles_only"] != "" else np.nan for r in cho]),
        }
    else:
        cpu_oof = _read_tsv(EXP_BASE / "models_cpu" / ep / "oof_predictions.tsv")
        cpu_ho = _read_tsv(EXP_BASE / "models_cpu" / ep / "holdout_predictions.tsv")
        fus_oof = _read_tsv(EXP_BASE / "models_fusion" / ep / "oof_predictions.tsv")
        fus_ho = _read_tsv(EXP_BASE / "models_fusion" / ep / "holdout_predictions.tsv")
        pred_oof_map, pred_ho_map = {}, {}
        for tr, col in (("dual_modality", "pred_dual"), ("smiles_only", "pred_smiles_only")):
            w = ep_doc[tr]["weights"]
            c_o = np.array([float(r[col]) if r[col] != "" else np.nan for r in cpu_oof])
            f_o = np.array([float(r[col]) if r[col] != "" else np.nan for r in fus_oof])
            c_h = np.array([float(r[col]) if r[col] != "" else np.nan for r in cpu_ho])
            f_h = np.array([float(r[col]) if r[col] != "" else np.nan for r in fus_ho])
            wc, wf = float(w.get("cpu", 0.0)), float(w.get("fusion", 0.0))
            tot = max(wc + wf, 1e-12)
            po = (wc * np.nan_to_num(c_o) + wf * np.nan_to_num(f_o)) / tot
            ph = (wc * np.nan_to_num(c_h) + wf * np.nan_to_num(f_h)) / tot
            if bounds is not None:
                po, ph = np.clip(po, bounds[0], bounds[1]), np.clip(ph, bounds[0], bounds[1])
            pred_oof_map[tr], pred_ho_map[tr] = po, ph

    CAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"endpoint": ep, "wave": wave, "tracks": {}}

    for track in ("dual_modality", "smiles_only"):
        valid_cv = b1_conf.GLOBAL_SMI_VALID[cidx_cv] if track == "smiles_only" else np.ones(len(cv_rows), dtype=bool)
        valid_ho = b1_conf.GLOBAL_SMI_VALID[cidx_ho] if track == "smiles_only" else np.ones(len(ho_rows), dtype=bool)
        phys_cv = _extract_phys_block(cidx_cv, track)
        phys_ho = _extract_phys_block(cidx_ho, track)

        m_cv = valid_cv & (ct_cv == "exact") & np.isfinite(y_cv) & np.isfinite(pred_oof_map[track])
        m_ho = valid_ho & (ct_ho == "exact") & np.isfinite(y_ho) & np.isfinite(pred_ho_map[track])
        ye, pe, fe, ce = y_cv[m_cv], pred_oof_map[track][m_cv], fids_cv[m_cv], cids_cv[m_cv]
        ph_e, t1_e, t5_e = phys_cv[m_cv], oof_max_t[m_cv], oof_top5_t[m_cv]
        yh, ph, ph_h, t1_h, t5_h = y_ho[m_ho], pred_ho_map[track][m_ho], phys_ho[m_ho], ho_max_t[m_ho], ho_top5_t[m_ho]

        base_conf = ep_doc[track]["conformal"]
        base_mode = base_conf.get("mode", "standard")
        methods = ("baseline", "mondrian_ad_eb_cw", "cqr_lgb_cw")
        nested_ivs = {k: {b: np.full(len(ye), np.nan) for b in ("lo80", "hi80", "lo90", "hi90")} for k in methods}

        for f in sorted(set(fe.tolist())):
            va = fe == f
            tr = ~va
            if tr.sum() < 6 or va.sum() == 0:
                continue
            b_sp = _fit_baseline_conformal_on_subset(ye[tr], pe[tr], t1_e[tr], mode=base_mode, bounds=bounds)
            iv_b = _apply_baseline_manifest_conformal(pe[va], b_sp, t1_e[va])
            mcw_sp = fit_mondrian_ad_eb_cw(ye[tr], pe[tr], t1_e[tr], bounds=bounds, cluster_ids=ce[tr])
            iv_m = apply_mondrian_ad_eb(pe[va], mcw_sp, t1_e[va])
            cqr_cw = ClusterWeightedCQRCalibrator(bounds=bounds, cluster_weighted=True).fit(
                ye[tr], pe[tr], ph_e[tr], t1_e[tr], t5_e[tr], fold_ids=fe[tr], cluster_ids=ce[tr]
            )
            iv_c = cqr_cw.predict_intervals(pe[va], ph_e[va], t1_e[va], t5_e[va])
            for k, iv in (("baseline", iv_b), ("mondrian_ad_eb_cw", iv_m), ("cqr_lgb_cw", iv_c)):
                for b in ("lo80", "hi80", "lo90", "hi90"):
                    nested_ivs[k][b][va] = iv[b]

        # Strictly CLUSTER-WEIGHTED nested 5-fold OOF selection (ZERO holdout peeking!)
        oof_cw = {k: _eval_cluster_weighted_interval_metrics(ye, nested_ivs[k], t1_e, ce) for k in methods}
        rw80 = float(oof_cw["baseline"]["w80"] or 1.0)
        rw90 = float(oof_cw["baseline"]["w90"] or 1.5)
        scores = {k: _cw_oof_score(oof_cw[k], rw80, rw90)[1] for k in methods}

        # Prior structural rule + strictly CW-OOF selection (ZERO holdout peeking!):
        # - Bounded [0, 1] endpoints (PPB, F) have inherently asymmetric clipped residuals -> use asymmetric cqr_lgb_cw.
        # - Unbounded log10 endpoints -> prefer pure-JSON mondrian_ad_eb_cw unless cqr_lgb_cw beats CW-OOF by >= 0.04.
        if bounds is not None:
            sel = "cqr_lgb_cw"
        elif scores["cqr_lgb_cw"]["total_score"] + 0.04 < scores["mondrian_ad_eb_cw"]["total_score"]:
            sel = "cqr_lgb_cw"
        else:
            sel = "mondrian_ad_eb_cw"

        # Fit final production calibrator on full exact CV & evaluate ONCE on untouched 20% Holdout!
        spec_mcw = fit_mondrian_ad_eb_cw(ye, pe, t1_e, bounds=bounds, cluster_ids=ce)
        iv_ho_base = _apply_baseline_manifest_conformal(ph, base_conf, t1_h)
        iv_ho_mcw = apply_mondrian_ad_eb(ph, spec_mcw, t1_h)
        cal_cw = ClusterWeightedCQRCalibrator(bounds=bounds, cluster_weighted=True).fit(
            ye, pe, ph_e, t1_e, t5_e, fold_ids=fe, cluster_ids=ce
        )
        iv_ho_cqr = cal_cw.predict_intervals(ph, ph_h, t1_h, t5_h)

        ho_m = {
            "baseline": _eval_interval_metrics(yh, iv_ho_base, t1_h),
            "mondrian_ad_eb_cw": _eval_interval_metrics(yh, iv_ho_mcw, t1_h),
            "cqr_lgb_cw": _eval_interval_metrics(yh, iv_ho_cqr, t1_h),
        }

        cal_rel = None
        cal_kb = 0.0
        if sel == "cqr_lgb_cw":
            fname = f"{ep}__{track}__cqr_lgb_cw.joblib"
            fp = CAL_OUT_DIR / fname
            joblib.dump(cal_cw, fp, compress=3)
            _check_portable(fp)
            cal_rel = f"conformal_calibrators/{fname}"
            cal_kb = round(fp.stat().st_size / 1024.0, 2)
            final_spec = {
                "mode": "cqr_lgb_cw",
                "calibrator_file": cal_rel,
                "n_cal_total": int(len(ye)),
                "n_eff_clusters": int(len(set(str(c) for c in ce))),
                "Q80": round(float(cal_cw.Q80_), 5),
                "Q90": round(float(cal_cw.Q90_), 5),
                "fallback_mondrian_cw": spec_mcw,
                "bounds": list(bounds) if bounds is not None else None,
            }
        else:
            final_spec = spec_mcw

        out["tracks"][track] = {
            "selected_method": sel,
            "oof_cw_metrics": oof_cw,
            "oof_cw_scores": scores,
            "ho_metrics": ho_m,
            "selected_ho": ho_m[sel],
            "production_conformal_spec": final_spec,
            "calibrator_kb": cal_kb,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    t0 = time.time()

    global G_MANIFEST
    G_MANIFEST = json.loads((BUNDLE_DIR / "v33_manifest.json").read_text(encoding="utf-8"))
    bank = UnifiedSmallNBank()
    b1_conf.GLOBAL_MANIFEST = G_MANIFEST
    b1_conf.GLOBAL_PHYS = bank.physchem
    b1_conf.GLOBAL_SMI_VALID = bank.smiles_valid
    b1_conf.GLOBAL_SEQ_VALID = bank.seq_valid
    b1_conf.GLOBAL_MORGAN_PACKED = bank.morgan_packed
    b1_conf.GLOBAL_KEY_TO_IDX = bank.key_to_idx

    reg_eps = sorted(ep for ep, ed in G_MANIFEST["endpoints"].items() if ed["task_type"] == "regression")
    ctx = mp.get_context("fork")
    ep_res = {}
    with ctx.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_run_one_ep, reg_eps):
            ep_res[r["endpoint"]] = r

    summary = {}
    for mname in ("baseline", "mondrian_ad_eb_cw", "cqr_lgb_cw", "selected_cw"):
        c80s, c90s, w80s, ood80s = [], [], [], []
        for ep, ed in ep_res.items():
            for tr in ("dual_modality", "smiles_only"):
                hm = ed["tracks"][tr]["selected_ho"] if mname == "selected_cw" else ed["tracks"][tr]["ho_metrics"][mname]
                if hm["cov80"] is not None:
                    c80s.append(hm["cov80"])
                    c90s.append(hm["cov90"])
                    w80s.append(hm["w80"])
                if "OOD" in hm.get("by_tier", {}) and hm["by_tier"]["OOD"]["n"] >= 3:
                    ood80s.append(hm["by_tier"]["OOD"]["cov80"])
        summary[mname] = {
            "n_tracks": len(c80s),
            "mean_cov80": round(float(np.mean(c80s)), 4),
            "mean_cov90": round(float(np.mean(c90s)), 4),
            "min_cov80": round(float(np.min(c80s)), 4),
            "n_tracks_cov80_lt_70pct": int(sum(1 for x in c80s if x < 0.70)),
            "mean_w80": round(float(np.mean(w80s)), 4),
            "mean_ood_cov80": round(float(np.mean(ood80s)), 4) if ood80s else None,
        }
    sel_counts = {}
    tot_kb = 0.0
    for ep, ed in ep_res.items():
        for tr in ("dual_modality", "smiles_only"):
            sm = ed["tracks"][tr]["selected_method"]
            sel_counts[sm] = sel_counts.get(sm, 0) + 1
            tot_kb += float(ed["tracks"][tr]["calibrator_kb"])
    summary["selected_counts"] = sel_counts
    summary["total_cqr_joblib_mb"] = round(tot_kb / 1024.0, 3)

    out_doc = {
        "schema": "v333_step2b_cw_conformal_production_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": summary,
        "endpoints": ep_res,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path = OUT_DIR / "v333_conformal_calibration.json"
    out_path.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Step 2B-CW DONE] {out_path} in {out_doc['elapsed_sec']}s", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
