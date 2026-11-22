#!/usr/bin/env python3
"""Extract 100% real audited V32 reference metrics and Case17 FDA/V32 references into
v32_reference_snapshot.json (NEVER invents any V32 number; uses null where V32 was absent/deferred).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

BASE = Path(r"D:\work\中检院")
OUT_DIR = BASE / "v33_upgrade" / "bundle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V32_REP_PATH = BASE / ".publish" / "peptide_data_v31" / "omnigrid" / "configs" / "endpoint_representatives.tsv"
V32_C17_PATH = BASE / ".publish" / "peptide_data_v31" / "benchmark" / "case17" / "data" / "case17_comparison_base.tsv"
V32_REF_PATH = BASE / ".publish" / "peptide_data_v31" / "benchmark" / "case17" / "data" / "case17_reference_values.tsv"

V32_MAP = {
    "BBB": ("BBB", "classification"),
    "CL_absolute": ("CL_absolute", "regression"),
    "CL_per_kg": ("CL_per_kg", "regression"),
    "Caco2": ("Caco2", "regression"),
    "Cmax": ("Cmax", "regression"),
    "F": ("F", "regression"),
    "LogD7.4": ("LogD7.4", "regression"),
    "MDCK": ("MDCK", "regression"),
    "MIC_E_coli": ("MIC", "regression"),
    "PAMPA": ("PAMPA", "regression"),
    "PPB": ("PPB", "regression"),
    "RRCK": ("RRCK", "regression"),
    "Vd_per_kg": ("Vdss_per_kg", "regression"),
    "half_life_cls": ("half_life", "classification"),
    "half_life_reg": ("half_life", "regression"),
    "hemolysis_HC50": ("hemolysis_HC50", "regression"),
    "hemolysis_cls": ("hemolysis", "classification"),
    "overall_peptide_toxicity": ("overall_peptide_toxicity", "classification"),
    "solubility_cls": ("solubility", "classification"),
    "solubility_reg": ("solubility", "regression"),
}

ALL_24 = [
    "BBB", "CL_absolute", "CL_per_kg", "Caco2", "Cmax", "F", "LogD7.4", "MDCK",
    "MIC_C_albicans", "MIC_E_coli", "MIC_P_aeruginosa", "MIC_S_aureus", "PAMPA",
    "PPB", "RRCK", "Vd_per_kg", "cytotoxicity_cls", "half_life_cls", "half_life_reg",
    "hemolysis_HC50", "hemolysis_cls", "overall_peptide_toxicity", "solubility_cls", "solubility_reg",
]


def _f(v):
    if v is None or str(v).strip() == "":
        return None
    x = float(v)
    return round(x, 5) if math.isfinite(x) else None


def main() -> None:
    rep_rows = {}
    with V32_REP_PATH.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rep_rows[(r["endpoint"], r["task_kind"])] = r

    v32_endpoints = {}
    for ep in ALL_24:
        if ep in V32_MAP and V32_MAP[ep] in rep_rows:
            r = rep_rows[V32_MAP[ep]]
            v32_endpoints[ep] = {
                "v32_endpoint": r["endpoint"],
                "v32_task_kind": r["task_kind"],
                "v32_protocol": r["protocol"],
                "v32_records": int(r["records"]) if r["records"] else None,
                "v32_model": r["recommended_combination"],
                "cv_mae": _f(r.get("cv_mae_mean")),
                "cv_rmse": _f(r.get("cv_rmse_mean")),
                "cv_spearman": _f(r.get("cv_spearman_mean")),
                "cv_roc_auc": _f(r.get("cv_roc_auc_mean")),
                "cv_pr_auc": _f(r.get("cv_pr_auc_mean")),
                "cv_brier": _f(r.get("cv_brier_mean")),
                "holdout_mae": _f(r.get("holdout_mae_mean")),
                "holdout_rmse": _f(r.get("holdout_rmse_mean")),
                "holdout_spearman": _f(r.get("holdout_spearman_mean")),
                "holdout_roc_auc": _f(r.get("holdout_roc_auc_mean")),
                "holdout_pr_auc": _f(r.get("holdout_pr_auc_mean")),
                "holdout_mcc": _f(r.get("holdout_mcc_mean")),
                "holdout_brier": _f(r.get("holdout_brier_mean")),
            }
        else:
            v32_endpoints[ep] = None

    c17_v32 = {}
    with V32_C17_PATH.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c17_v32[r["name"]] = dict(r)

    c17_fda = {}
    with V32_REF_PATH.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            c17_fda[r["name"]] = dict(r)

    snap = {
        "schema": "peptide_omnipanel_v32_audited_reference_snapshot_v1",
        "source_files": {
            "endpoint_representatives": str(V32_REP_PATH),
            "case17_comparison_base": str(V32_C17_PATH),
            "case17_reference_values": str(V32_REF_PATH),
        },
        "endpoints": v32_endpoints,
        "case17_v32": c17_v32,
        "case17_fda_refs": c17_fda,
    }
    out_path = OUT_DIR / "v32_reference_snapshot.json"
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    mapped = sum(1 for v in v32_endpoints.values() if v is not None)
    print(f"Wrote {out_path} ({mapped}/24 V32 mapped endpoints, {24 - mapped} null/new)")


if __name__ == "__main__":
    main()
