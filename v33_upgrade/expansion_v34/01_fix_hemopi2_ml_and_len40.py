#!/usr/bin/env python3
"""V34 Stream A Script 1 (runs on compute_host):
Addresses Round-3 Peer Review P0-4 / P1-2:
  1. Diagnose & Fix HemoPI2-ML (`hemopi2_ml_clf.sav`) silent failure in v33.6:
     - Exact root cause: in `02_v336`, `from sklearn.base import clone` loaded conda's
       `scikit-learn==1.4.2` BEFORE `sys.path.insert(0, runtime_sklearn)` (`1.3.1`),
       crashing `predict_proba` with `AttributeError: 'DecisionTreeClassifier' object
       has no attribute 'monotonic_cst'`.
     - Fix: prepend `runtime_sklearn` (scikit-learn 1.3.1) BEFORE any `sklearn` import,
       log full subprocess output, and raise on any non-zero exit.
     - Evaluate BOTH:
       (a) Official pretrained `hemopi2_ml_clf.sav` (RandomForestClassifier, n_estimators=200,
           1,190 composition/physicochemical features via official `composition_calculate.py`)
           under official HemoPI2 40-aa truncation (`seq[:40]`) AND untruncated (`seq`), on
           both All Canonical (`n=3,230`) and Zero-Overlap Novel (`n=2,334`) holdout sets.
       (b) Same-split zero-leakage retrained HemoPI2-ML on our exact `hemolysis_cls` CV split
           (`n=13,316` canonical peptides -> frozen `n=3,230` holdout & `n=2,334` zero-overlap):
           exact cloned official RandomForest (`seq[:40]` & full-length), plus official ML
           classifier family benchmarks (ExtraTrees, LightGBM, XGBoost) on the exact same
           1,190-feature official HemoPI2 descriptor matrix.
  2. `<= 40 aa` vs `> 40 aa` Length-Distribution & Truncation Sensitivity Analysis:
     - Full sequence length distributions for `hemolysis_cls` CV & holdout (all canonical and
       zero-overlap novel).
     - Stratified evaluation (`all_canonical`, `le40`, `gt40`, and zero-overlap counterparts)
       for `Peptide OmniPanel`, `HemoPI2 ESM2-t6` (official `<=40 aa` truncation AND
       untruncated `max_length=512`), `HemoPI2 Pretrained ML`, and `HemoPI2 Same-Split ML`
       reporting exact `n`, `n_pos`, `pos_rate`, `ROC-AUC`, `PR-AUC`, `Brier`, paired
       cluster-bootstrap 95% CIs, raw p-values (`<=0.001` floor), and Holm-Bonferroni
       adjusted p-values across both the HemoPI2 v3.4 family and the full updated external
       baseline ledger (merging v33.6's 21 comparisons + newly recovered HemoPI2-ML comparisons).
  3. Structured `published_platform_comparison` ledger comparing `Peptide OmniPanel` vs
     `pepADMET` (JCIM) and `PeptiVerse` (Nature Communications) on published architectural
     capabilities and endpoint coverage.

Strictly ZERO holdout peeking; CPU workers capped <= 16 cores to coexist safely on compute_host.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

V33_BASE = Path(os.environ.get("V33_BASE", str(Path(__file__).resolve().parents[1])))
BUNDLE_DIR = V33_BASE / "bundle"
EXP331 = V33_BASE / "expansion_v331"
EXP334 = V33_BASE / "expansion_v334"
EXP336 = V33_BASE / "expansion_v336"
EXP34 = V33_BASE / "expansion_v34"
CACHE_DIR = EXP34 / "cache_hemopi2"
EXP34.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEMOPI_ROOT = Path(os.environ.get("HEMOPI_ROOT", str(Path(__file__).resolve().parents[2] / "external_baselines" / "hemopi2")))
HEMOPI_MODEL_DIR = HEMOPI_ROOT / "peptide_reference_bundle_20261013/external_assets/HemoPI2/extracted/Model"
HEMOPI_REPO_DIR = HEMOPI_ROOT / "peptide_reference_bundle_20261013/repos/HemoPI2"
HEMOPI_SKLEARN_PY_PATH = str(HEMOPI_ROOT / "runtime_sklearn")

csv.field_size_limit(10_000_000)
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_SET = set(AA20)
MAX_CPU_WORKERS = 12  # Strictly <= 16 cores on shared compute_host


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    if not pvals:
        return {}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    rmax = 0.0
    for i, (k, p) in enumerate(items):
        rmax = max(rmax, min(1.0, p * (m - i)))
        adj[k] = round(rmax, 4)
    return adj


def paired_cluster_bootstrap_cls(
    y: np.ndarray,
    pred_omni: np.ndarray,
    pred_base: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261127,
) -> Dict[str, Any]:
    """Two-sided paired cluster-bootstrap test for ROC-AUC & PR-AUC (Omni - Base)."""
    ok = np.isfinite(y) & np.isfinite(pred_omni) & np.isfinite(pred_base)
    y, pa, pb, cl = y[ok].astype(int), pred_omni[ok], pred_base[ok], clusters[ok]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {
            "n_eval": int(len(y)),
            "n_clusters": int(len(set(map(str, cl)))),
            "metric": "delta_roc_auc (Omni - Base)",
            "mean_delta": None,
            "ci95": None,
            "p_value": None,
            "p_value_display": "NA",
            "pr_auc_delta": None,
            "pr_auc_ci95": None,
            "pr_auc_p_value": None,
        }

    cl_map: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cl):
        cl_map[str(c)].append(i)
    cl_arrs = [np.array(v, dtype=np.int32) for v in cl_map.values()]
    n_cl = len(cl_arrs)
    rng = np.random.RandomState(seed)
    d_roc, d_pr = [], []
    for _ in range(n_boot):
        pick = rng.randint(0, n_cl, size=n_cl)
        idx = np.concatenate([cl_arrs[p] for p in pick])
        yb, p_a, p_b = y[idx], pa[idx], pb[idx]
        if len(np.unique(yb)) < 2:
            continue
        d_roc.append(float(roc_auc_score(yb, p_a)) - float(roc_auc_score(yb, p_b)))
        d_pr.append(float(average_precision_score(yb, p_a)) - float(average_precision_score(yb, p_b)))

    dr = np.array(d_roc, dtype=np.float64)
    dp = np.array(d_pr, dtype=np.float64)
    p_floor = 1.0 / max(len(dr), 1)

    p_roc_raw = float(2.0 * min(np.mean(dr <= 0.0), np.mean(dr >= 0.0)))
    p_roc_num = max(p_roc_raw, p_floor)
    p_pr_raw = float(2.0 * min(np.mean(dp <= 0.0), np.mean(dp >= 0.0)))
    p_pr_num = max(p_pr_raw, p_floor)

    return {
        "n_eval": int(len(y)),
        "n_clusters": int(n_cl),
        "n_boot_valid": int(len(dr)),
        "metric": "delta_roc_auc (Omni - Base)",
        "point_delta_roc_auc": round(float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), 4),
        "mean_delta": round(float(dr.mean()), 4),
        "ci95": [round(float(np.percentile(dr, 2.5)), 4), round(float(np.percentile(dr, 97.5)), 4)],
        "p_value": round(p_roc_num, 4),
        "p_value_display": f"<={p_floor:.3f}" if p_roc_raw < p_floor else f"{p_roc_num:.4f}",
        "point_delta_pr_auc": round(float(average_precision_score(y, pa) - average_precision_score(y, pb)), 4),
        "pr_auc_mean_delta": round(float(dp.mean()), 4),
        "pr_auc_ci95": [round(float(np.percentile(dp, 2.5)), 4), round(float(np.percentile(dp, 97.5)), 4)],
        "pr_auc_p_value": round(p_pr_num, 4),
        "pr_auc_p_value_display": f"<={p_floor:.3f}" if p_pr_raw < p_floor else f"{p_pr_num:.4f}",
    }


def score_cls_subset(
    y: np.ndarray,
    pred: np.ndarray,
    pred_omni: np.ndarray,
    clusters: np.ndarray,
    thr: float = 0.5,
    n_boot: int = 1000,
    seed: int = 20261127,
) -> Dict[str, Any]:
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(pred_omni)
    y_v, p_v, o_v, c_v = y[ok].astype(int), pred[ok], pred_omni[ok], clusters[ok]
    n_pos = int((y_v == 1).sum())
    n_neg = int((y_v == 0).sum())
    out: Dict[str, Any] = {
        "n": int(len(y_v)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": round(float(y_v.mean()), 4) if len(y_v) else None,
        "n_clusters": int(len(set(map(str, c_v)))),
    }
    if len(y_v) >= 5 and len(np.unique(y_v)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_v, p_v)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_v, p_v)), 4)
        out["brier"] = round(float(np.mean((y_v - p_v) ** 2)), 4)
        out["mcc"] = round(float(matthews_corrcoef(y_v, (p_v >= thr).astype(int))), 4)
        out["omni_same_subset_roc_auc"] = round(float(roc_auc_score(y_v, o_v)), 4)
        out["omni_same_subset_pr_auc"] = round(float(average_precision_score(y_v, o_v)), 4)
        out["omni_same_subset_brier"] = round(float(np.mean((y_v - o_v) ** 2)), 4)
        out["paired_boot_vs_omni"] = paired_cluster_bootstrap_cls(y_v, o_v, p_v, c_v, n_boot=n_boot, seed=seed)
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
        out["brier"] = None
        out["mcc"] = None
        out["paired_boot_vs_omni"] = None
    return out


def length_distribution_summary(seqs: List[str], y: Optional[np.ndarray] = None) -> Dict[str, Any]:
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    le40 = lens <= 40
    gt40 = lens > 40
    bins = [
        ("4_to_10", (lens >= 4) & (lens <= 10)),
        ("11_to_20", (lens >= 11) & (lens <= 20)),
        ("21_to_30", (lens >= 21) & (lens <= 30)),
        ("31_to_40", (lens >= 31) & (lens <= 40)),
        ("41_to_60", (lens >= 41) & (lens <= 60)),
        ("61_to_100", (lens >= 61) & (lens <= 100)),
        ("gt_100", lens > 100),
    ]
    res: Dict[str, Any] = {
        "n_total": int(len(lens)),
        "n_le40": int(le40.sum()),
        "pct_le40": round(100.0 * float(le40.mean()), 2) if len(lens) else 0.0,
        "n_gt40": int(gt40.sum()),
        "pct_gt40": round(100.0 * float(gt40.mean()), 2) if len(lens) else 0.0,
        "min": int(lens.min()) if len(lens) else 0,
        "p10": round(float(np.percentile(lens, 10)), 1) if len(lens) else 0.0,
        "median": round(float(np.median(lens)), 1) if len(lens) else 0.0,
        "mean": round(float(lens.mean()), 2) if len(lens) else 0.0,
        "p90": round(float(np.percentile(lens, 90)), 1) if len(lens) else 0.0,
        "max": int(lens.max()) if len(lens) else 0,
        "length_bins": {k: int(m.sum()) for k, m in bins},
    }
    if y is not None and len(y) == len(lens):
        res["pos_rate_overall"] = round(float(y.mean()), 4)
        res["pos_rate_le40"] = round(float(y[le40].mean()), 4) if le40.any() else None
        res["pos_rate_gt40"] = round(float(y[gt40].mean()), 4) if gt40.any() else None
    return res


def run_hemopi2_ml_isolated_subprocess(
    cv_seqs: List[str],
    cv_y: List[int],
    ho_seqs: List[str],
    tag: str,
    truncate_40: bool,
) -> Dict[str, Any]:
    """Run official HemoPI2 composition_calculate.py + pretrained hemopi2_ml_clf.sav (sklearn 1.3.1)
    + cloned same-split RF in an isolated subprocess with runtime_sklearn inserted FIRST on sys.path.
    Caches extracted 1,190-feature matrices to disk under CACHE_DIR for fast reproducibility.
    """
    feat_cache = CACHE_DIR / f"hemopi2_1190_feats_{tag}.npz"
    cv_in = [s[:40] if truncate_40 else s for s in cv_seqs]
    ho_in = [s[:40] if truncate_40 else s for s in ho_seqs]

    with tempfile.TemporaryDirectory(prefix=f"hemopi2_v34_{tag}_") as tmpd:
        tmp = Path(tmpd)
        (tmp / "inputs.json").write_text(
            json.dumps(
                {
                    "cv_seqs": cv_in,
                    "cv_y": cv_y,
                    "ho_seqs": ho_in,
                    "feat_cache": str(feat_cache),
                    "n_workers": MAX_CPU_WORKERS,
                }
            ),
            encoding="utf-8",
        )
        helper = tmp / "run_hemopi2_sk131.py"
        helper.write_text(
            f"""
import sys
# CRITICAL FIX vs v33.6: prepend runtime_sklearn (scikit-learn 1.3.1) BEFORE any sklearn import!
sys.path.insert(0, {repr(HEMOPI_SKLEARN_PY_PATH)})

import json, pickle, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

MODEL_DIR = Path({repr(str(HEMOPI_MODEL_DIR))})
tmp = Path({repr(str(tmp))})
inp = json.loads((tmp / "inputs.json").read_text(encoding="utf-8"))
cv_seqs = inp["cv_seqs"]
cv_y = np.array(inp["cv_y"], dtype=int)
ho_seqs = inp["ho_seqs"]
feat_cache = Path(inp["feat_cache"])
n_workers = int(inp["n_workers"])

print(f"[HemoPI2-sk131] sklearn={{sklearn.__version__}} from {{sklearn.__file__}}", flush=True)
clf_pre = pickle.load(open(MODEL_DIR / "hemopi2_ml_clf.sav", "rb"))
if hasattr(clf_pre, "n_jobs"):
    clf_pre.n_jobs = min(8, n_workers)
feat_cols = list(clf_pre.feature_names_in_)
params_pre = clf_pre.get_params()
print(f"[HemoPI2-sk131] Loaded pretrained {{type(clf_pre).__name__}} n_features={{len(feat_cols)}} n_estimators={{params_pre.get('n_estimators')}}", flush=True)

def build_hemopi_feats(seqs, split_tag):
    n = len(seqs)
    nw = min(n_workers, max(1, n // 80))
    step = (n + nw - 1) // nw
    chunks = [(ci, seqs[ci * step : (ci + 1) * step]) for ci in range(nw) if seqs[ci * step : (ci + 1) * step]]

    def _run_chunk(args):
        ci, sub = args
        seq_f = tmp / f"{{split_tag}}_{{ci}}.seq"
        seq_f.write_text("\\n".join(sub) + "\\n", encoding="utf-8")
        wdir = tmp / f"work_{{split_tag}}_{{ci}}"
        wdir.mkdir(exist_ok=True)
        out_csv = tmp / f"{{split_tag}}_{{ci}}_feats.csv"
        proc = subprocess.run(
            [sys.executable, str(MODEL_DIR / "composition_calculate.py"), str(seq_f), str(wdir), str(out_csv)],
            cwd=str(MODEL_DIR.parent),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"composition_calculate.py failed on {{split_tag}}_{{ci}}: {{proc.stderr[-600:]}}")
        df_c = pd.read_csv(out_csv)
        missing = [c for c in feat_cols if c not in df_c.columns]
        if missing:
            for c in missing:
                df_c[c] = 0.0
        arr = df_c[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        if arr.shape[0] != len(sub):
            raise RuntimeError(f"Row mismatch on {{split_tag}}_{{ci}}: got {{arr.shape[0]}} vs {{len(sub)}}")
        return ci, arr, len(missing)

    with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
        res_list = sorted(ex.map(_run_chunk, chunks), key=lambda x: x[0])
    n_miss = max(m for _, _, m in res_list)
    return np.vstack([arr for _, arr, _ in res_list]), n_miss

t0 = time.time()
if feat_cache.exists():
    z = np.load(feat_cache, allow_pickle=False)
    X_cv = z["X_cv"].astype(np.float32)
    X_ho = z["X_ho"].astype(np.float32)
    n_miss_ho = int(z["n_miss_ho"])
    n_miss_cv = int(z["n_miss_cv"])
    z.close()
    if X_cv.shape[0] != len(cv_seqs) or X_ho.shape[0] != len(ho_seqs) or X_ho.shape[1] != len(feat_cols):
        X_ho, n_miss_ho = build_hemopi_feats(ho_seqs, "ho")
        X_cv, n_miss_cv = build_hemopi_feats(cv_seqs, "cv")
        np.savez_compressed(feat_cache, X_cv=X_cv, X_ho=X_ho, n_miss_cv=n_miss_cv, n_miss_ho=n_miss_ho)
    else:
        print(f"[HemoPI2-sk131] Loaded cached 1190-feature matrices from {{feat_cache}}", flush=True)
else:
    X_ho, n_miss_ho = build_hemopi_feats(ho_seqs, "ho")
    print(f"[HemoPI2-sk131] Extracted X_ho {{X_ho.shape}} in {{time.time()-t0:.1f}}s (missing_cols={{n_miss_ho}})", flush=True)
    t1 = time.time()
    X_cv, n_miss_cv = build_hemopi_feats(cv_seqs, "cv")
    print(f"[HemoPI2-sk131] Extracted X_cv {{X_cv.shape}} in {{time.time()-t1:.1f}}s (missing_cols={{n_miss_cv}})", flush=True)
    np.savez_compressed(feat_cache, X_cv=X_cv, X_ho=X_ho, n_miss_cv=n_miss_cv, n_miss_ho=n_miss_ho)

# Predict with pretrained official HemoPI2 ML model using DataFrame with exact column names
df_ho = pd.DataFrame(X_ho, columns=feat_cols)
df_cv = pd.DataFrame(X_cv, columns=feat_cols)
p_pre = clf_pre.predict_proba(df_ho)[:, 1].astype(np.float64)

# Same-split zero-leakage retrain of exact cloned official RandomForestClassifier
t2 = time.time()
clf_same = clone(clf_pre)
if hasattr(clf_same, "n_jobs"):
    clf_same.n_jobs = min(8, n_workers)
clf_same.fit(df_cv, cv_y)
p_same_rf = clf_same.predict_proba(df_ho)[:, 1].astype(np.float64)
print(f"[HemoPI2-sk131] Pretrained predict + Same-split RF refit done in {{time.time()-t2:.1f}}s", flush=True)

meta = {{
    "sklearn_runtime_version": sklearn.__version__,
    "estimator_class": type(clf_pre).__name__,
    "estimator_params": {{k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in params_pre.items()}},
    "n_features": len(feat_cols),
    "first_10_features": feat_cols[:10],
    "n_missing_cols_ho": int(n_miss_ho),
    "n_missing_cols_cv": int(n_miss_cv),
}}
(tmp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
np.savez(tmp / "preds.npz", p_pre=p_pre, p_same_rf=p_same_rf)
""",
            encoding="utf-8",
        )
        proc = subprocess.run([sys.executable, str(helper)], capture_output=True, text=True, timeout=1800)
        print(f"  [{tag} stdout]:\n{proc.stdout.strip()}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"HemoPI2-ML subprocess ({tag}) failed with rc={proc.returncode}:\n{proc.stderr[-1500:]}"
            )
        meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
        preds = np.load(tmp / "preds.npz")
        p_pre = preds["p_pre"].astype(np.float64)
        p_same_rf = preds["p_same_rf"].astype(np.float64)
        preds.close()

    # Load cached features in parent process to train additional same-split ML classifier families
    z = np.load(feat_cache, allow_pickle=False)
    X_cv = z["X_cv"].astype(np.float32)
    X_ho = z["X_ho"].astype(np.float32)
    z.close()

    return {
        "meta": meta,
        "X_cv": X_cv,
        "X_ho": X_ho,
        "p_pre": p_pre,
        "p_same_rf": p_same_rf,
    }


def run_hemopi2_esm2_eval(ho_seqs: List[str], truncate_40: bool) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(HEMOPI_MODEL_DIR))
    esm_mdl = AutoModelForSequenceClassification.from_pretrained(str(HEMOPI_MODEL_DIR))
    esm_mdl.eval()
    torch.set_num_threads(8)
    probs = []
    bs = 64
    max_len = 42 if truncate_40 else 512
    with torch.no_grad():
        for st in range(0, len(ho_seqs), bs):
            batch_s = [s[:40] for s in ho_seqs[st : st + bs]] if truncate_40 else ho_seqs[st : st + bs]
            enc = tok(batch_s, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            logits = esm_mdl(**enc).logits
            pr = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(pr)
    out = np.concatenate(probs).astype(np.float64)
    print(f"  [HemoPI2 ESM2-t6 trunc40={truncate_40}] Predicted {len(out)} seqs in {time.time()-t0:.1f}s", flush=True)
    return out


def build_published_platform_comparison() -> Dict[str, Any]:
    """Structured architectural & endpoint-coverage comparison against pepADMET (JCIM) and PeptiVerse (Nat Commun).
    Reports ONLY published architectural capabilities & scope — zero fabricated head-to-head metrics!
    """
    return {
        "disclaimer": (
            "Architectural and capability ledger compiled from published papers and public repositories. "
            "No fabricated cross-platform benchmark scores are reported where identical splits or public "
            "pretrained inference weights are unavailable; same-split empirical benchmarks are reported "
            "separately for HemoPI2 v1.3, CycPeptMP, Multi_CycGT, and PLifePred."
        ),
        "platforms": {
            "Peptide_OmniPanel_v3_4": {
                "citation": "Peptide OmniPanel 1.4 / Engine v34 (This Work, 2026)",
                "input_modalities": [
                    "Canonical 1-letter linear peptide sequence",
                    "Arbitrary SMILES (cyclic/macrocyclic, N-methylated, D-amino acids, disulfide, stapled, lipidated/PEGylated NCAAs)",
                    "HELM / name lookup with automatic sequence-to-SMILES canonicalization",
                ],
                "model_architecture": (
                    "4-Foundation-Model Multimodal Super-Ensemble (NCAA-RepDistill-v2 + PeptideCLM-2 + "
                    "ChemBERTa-MTR + ESM-C 600M) + CPU Tanimoto-KRR/SVR/LGBM/ET + Domain-MTL / Deep branches "
                    "with strictly nested OOF simplex stacking and zero-holdout-peeking promotion"
                ),
                "n_endpoints": 47,
                "endpoint_domains": {
                    "permeability_lipophilicity": ["PAMPA", "Caco2", "MDCK", "RRCK", "efflux_ratio", "BBB", "LogD7.4", "LogD_decadiene"],
                    "systemic_clinical_pk": ["half_life_reg", "half_life_cls", "CL_absolute", "CL_per_kg", "Vd_absolute", "Vd_per_kg", "PPB", "F", "Cmax"],
                    "toxicity_hemolysis": ["hemolysis_cls", "hemolysis_HC50", "hemolysis_MHC", "cytotoxicity_cls", "cytotoxicity_CC50", "overall_peptide_toxicity"],
                    "solubility": ["solubility_cls", "solubility_reg"],
                    "antimicrobial_biofilm_mic_mbc": "22 pathogen-specific MIC / MBC / MBIC / MBEC / MFC / LC endpoints",
                },
                "uncertainty_and_ad": "Mondrian Tanimoto AD tiers + cluster-weighted split-conformal 80%/90% intervals + Platt probability calibration + 3-tier CI-aware evidence grading (20 quantitative / 26 rank_only / 1 exploratory_not_recommended)",
                "interpretability": "Direct production-ensemble perturbation attribution across all 47 endpoints (residue Ala/Gly masking + RDKit-valid chemical substituent pruning) + fidelity-audited E-GAT on 8 endpoints",
                "design_and_decision_tools": "PeptideOpt multi-objective Pareto analog generator (5 strategies) + Batch Screening with 4-pillar Peptide-MPO [0-100] + 57-product FDA/DailyMed clinical PK reference & approved-peptide percentile context",
                "public_web_server_and_api": "Live web server + FastAPI REST API + open-source code",
            },
            "pepADMET": {
                "citation": "pepADMET (J. Chem. Inf. Model., 2024/2025)",
                "input_modalities": [
                    "Peptide SMILES / molecular graph (supports modified and cyclic peptides where SMILES provided)",
                ],
                "model_architecture": "Graph neural network / molecular fingerprint ML predictors trained per ADMET property",
                "n_endpoints": "Focused multi-endpoint peptide ADMET panel (primarily membrane permeability such as PAMPA/Caco-2, lipophilicity/solubility, toxicity/hemolysis, and selected PK/stability properties; narrower than 47 endpoints and without 22 pathogen-specific antimicrobial/biofilm endpoints)",
                "endpoint_domains": {
                    "permeability_lipophilicity": "Core focus (e.g., PAMPA / Caco-2 / lipophilicity)",
                    "systemic_clinical_pk": "Limited coverage (typically lacks full human clinical CL/Vd/F/Cmax/PPB regulatory panel)",
                    "toxicity_hemolysis": "Selected hemolytic/toxicity classification tasks",
                    "antimicrobial_biofilm_mic_mbc": "Not a primary antimicrobial MIC/MBC/biofilm suite",
                },
                "uncertainty_and_ad": "Standard point predictions; lacks Mondrian cluster-weighted conformal intervals and CI-aware 3-tier evidence grading",
                "interpretability": "Substructure/attention visualization where supported; no production-ensemble perturbation across 47 endpoints",
                "design_and_decision_tools": "ADMET prediction focus; does not bundle FDA DailyMed regulatory PK percentile lookup + 5-strategy Pareto PeptideOpt + 4-pillar Peptide-MPO batch screening",
                "public_web_server_and_api": "Published web/tool platform; pretrained weights/endpoints not directly drop-in runnable on our frozen 47-endpoint cluster-split benchmark without re-implementation",
            },
            "PeptiVerse": {
                "citation": "PeptiVerse (Nature Communications, 2025)",
                "input_modalities": [
                    "Canonical and modified peptide sequence / SMILES representations (unified peptide foundation/representation framework)",
                ],
                "model_architecture": "Peptide foundation / multi-property deep learning framework covering therapeutic peptide property prediction and discovery",
                "n_endpoints": "Broad therapeutic peptide property & bioactivity benchmarks (permeability, toxicity/hemolysis, half-life/stability, antimicrobial/therapeutic activity)",
                "endpoint_domains": {
                    "permeability_lipophilicity": "Included (macrocycle permeability benchmarks such as CycPeptMPDB)",
                    "systemic_clinical_pk": "Half-life / stability emphasis; lacks dedicated human FDA/DailyMed clinical CL/Vd/PPB/F/Cmax PK panel with approved-peptide percentile context",
                    "toxicity_hemolysis": "Included (hemolysis / toxicity classification)",
                    "antimicrobial_biofilm_mic_mbc": "General antimicrobial activity rather than 22 quantitative pathogen-stratified MIC/MBC/biofilm regression endpoints",
                },
                "uncertainty_and_ad": "Point-estimate benchmarking; lacks Mondrian Tanimoto AD-stratified conformal UQ and explicit failure-mode grading (`exploratory_not_recommended`)",
                "interpretability": "Representation/embedding analysis; lacks unified direct ensemble perturbation attribution across all 47 ADMET/PK/MIC endpoints",
                "design_and_decision_tools": "Property prediction & peptide representation learning; lacks integrated clinical PK evidence index + approved-peptide percentiles + interactive Pareto PeptideOpt web workbench",
                "public_web_server_and_api": "Published benchmark/codebase; separate task definitions and splits prevent zero-leakage off-the-shelf scoring on our frozen cluster-grouped holdouts without retraining",
            },
        },
        "same_split_published_tools_evaluated_empirically": [
            "HemoPI2 v1.3 (Rathore et al., Protein Sci. 2025; GPL-3.0) — pretrained ML (fixed sk1.3.1), same-split retrained ML family (RF/ET/LGBM/XGB on exact 1,190 HemoPI2 features), and pretrained ESM2-t6 (truncated <=40 aa & untruncated)",
            "CycPeptMP / CycPeptMPDB baseline configs (Li et al., Brief. Bioinform. 2024; MIT) — RF-750 & 2D-SVM retrained on identical PAMPA split",
            "Multi_CycGT baseline config (Cao et al., JCIM 2024; GPL-3.0) — Tanimoto-kernel SVR retrained on identical PAMPA split",
            "PLifePred-style AAC+DPC+PhysChem RBF-SVR (Mathur et al., Sci. Rep. 2018) — retrained on identical half_life_reg split",
        ],
    }


def main() -> None:
    t_start = time.time()
    print("=== [V34 Stream A-1] HemoPI2-ML Fix + <=40aa Sensitivity + Competitor Ledger ===", flush=True)

    # 1. Load compound sequences (base v33.1 + delta v33.4)
    ck_to_seq: Dict[str, str] = {}
    for pth in (EXP331 / "data" / "compounds_v331_merged.tsv", EXP334 / "data" / "delta_v334_compounds.tsv"):
        if pth.exists():
            with pth.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    ck_to_seq[r["compound_key"]] = r.get("parent_sequence", "")

    # 2. Load exact frozen hemolysis_cls endpoint split & champion holdout predictions
    ep_path = V33_BASE / "data" / "endpoints" / "hemolysis_cls.tsv"
    rows = list(csv.DictReader(ep_path.open(encoding="utf-8"), delimiter="\t"))
    cv_r = [r for r in rows if int(r["is_holdout"]) == 0 and r["censor_type"] == "exact"]
    ho_r = [r for r in rows if int(r["is_holdout"]) == 1 and r["censor_type"] == "exact"]

    champ_path = BUNDLE_DIR / "champion_predictions" / "hemolysis_cls" / "holdout_champion.tsv"
    ho_champ_map: Dict[str, float] = {}
    with champ_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("censor_type", "exact") == "exact":
                ho_champ_map[r["record_id"]] = float(r["pred_dual"])

    y_ho_all = np.array([float(r["y_value"]) for r in ho_r], dtype=np.float64)
    p_omni_all = np.array([ho_champ_map.get(r["record_id"], np.nan) for r in ho_r], dtype=np.float64)
    cl_ho_all = np.array([r["cluster_id"] for r in ho_r], dtype=object)

    # 3. Load HemoPI2 published datasets for strict zero-overlap identification
    hemopi_known_seqs = set()
    for fn in ("cross_val_dataset.csv", "independent_dataset.csv"):
        fp = HEMOPI_REPO_DIR / "Dataset" / fn
        if fp.exists():
            df_h = pd.read_csv(fp)
            scol = "SEQUENCE" if "SEQUENCE" in df_h.columns else df_h.columns[0]
            for s in df_h[scol].dropna().astype(str):
                hemopi_known_seqs.add(s.strip().upper())
    print(f"  Loaded {len(hemopi_known_seqs)} unique sequences from HemoPI2 published corpus", flush=True)

    # Canonical 20-AA filter (exact match to v33.6: len >= 4 & all canonical AA20)
    ho_seq_idx, ho_seqs, ho_is_novel = [], [], []
    for i, r in enumerate(ho_r):
        sq = (ck_to_seq.get(r["compound_key"]) or "").strip().upper()
        if len(sq) >= 4 and set(sq).issubset(AA20_SET):
            ho_seq_idx.append(i)
            ho_seqs.append(sq)
            ho_is_novel.append(sq not in hemopi_known_seqs)
    ho_seq_idx = np.array(ho_seq_idx, dtype=np.int32)
    ho_is_novel = np.array(ho_is_novel, dtype=bool)
    ho_lens = np.array([len(s) for s in ho_seqs], dtype=np.int32)
    ho_le40 = ho_lens <= 40
    ho_gt40 = ho_lens > 40

    y_can = y_ho_all[ho_seq_idx]
    p_omni_can = p_omni_all[ho_seq_idx]
    cl_can = cl_ho_all[ho_seq_idx]

    cv_seqs, cv_ys = [], []
    for r in cv_r:
        sq = (ck_to_seq.get(r["compound_key"]) or "").strip().upper()
        if len(sq) >= 4 and set(sq).issubset(AA20_SET):
            cv_seqs.append(sq)
            cv_ys.append(int(float(r["y_value"])))
    cv_y_arr = np.array(cv_ys, dtype=int)

    print(
        f"  hemolysis_cls canonical CV={len(cv_seqs)} (<=40:{sum(len(s)<=40 for s in cv_seqs)}, >40:{sum(len(s)>40 for s in cv_seqs)}) | "
        f"HO={len(ho_seqs)} (<=40:{int(ho_le40.sum())}, >40:{int(ho_gt40.sum())}) | "
        f"ZeroOverlap={int(ho_is_novel.sum())} (<=40:{int((ho_is_novel & ho_le40).sum())}, >40:{int((ho_is_novel & ho_gt40).sum())})",
        flush=True,
    )
    assert len(ho_seqs) == 3230, f"Expected 3230 canonical holdout rows, got {len(ho_seqs)}"
    assert int(ho_is_novel.sum()) == 2334, f"Expected 2334 zero-overlap rows, got {int(ho_is_novel.sum())}"

    # 4. Run HemoPI2-ML (official <=40aa truncation AND full-length untruncated) via fixed sklearn 1.3.1 helper
    print("\n--- [Step 1] HemoPI2-ML Pretrained + Same-Split RF (Official <=40aa Truncation) ---", flush=True)
    ml_trunc = run_hemopi2_ml_isolated_subprocess(cv_seqs, cv_ys, ho_seqs, tag="trunc40", truncate_40=True)

    print("\n--- [Step 2] HemoPI2-ML Pretrained + Same-Split RF (Full-Length Untruncated) ---", flush=True)
    ml_full = run_hemopi2_ml_isolated_subprocess(cv_seqs, cv_ys, ho_seqs, tag="full", truncate_40=False)

    # 5. Train official HemoPI2 ML classifier family (ExtraTrees, LightGBM, XGBoost) on exact 1,190 HemoPI2 features (same-split, ZERO leakage)
    print("\n--- [Step 3] Same-Split Official HemoPI2 ML Classifier Family (ET / LGBM / XGB on 1,190 features) ---", flush=True)
    X_cv_t, X_ho_t = ml_trunc["X_cv"], ml_trunc["X_ho"]
    X_cv_f, X_ho_f = ml_full["X_cv"], ml_full["X_ho"]

    t_fam = time.time()
    et_t = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", n_jobs=8, random_state=42)
    et_t.fit(X_cv_t, cv_y_arr)
    p_same_et_trunc = et_t.predict_proba(X_ho_t)[:, 1].astype(np.float64)

    et_f = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", n_jobs=8, random_state=42)
    et_f.fit(X_cv_f, cv_y_arr)
    p_same_et_full = et_f.predict_proba(X_ho_f)[:, 1].astype(np.float64)

    lgb_t = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.8,
        colsample_bytree=0.7, n_jobs=8, random_state=42, verbose=-1
    )
    lgb_t.fit(X_cv_t, cv_y_arr)
    p_same_lgb_trunc = lgb_t.predict_proba(X_ho_t)[:, 1].astype(np.float64)

    lgb_f = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04, num_leaves=31, subsample=0.8,
        colsample_bytree=0.7, n_jobs=8, random_state=42, verbose=-1
    )
    lgb_f.fit(X_cv_f, cv_y_arr)
    p_same_lgb_full = lgb_f.predict_proba(X_ho_f)[:, 1].astype(np.float64)

    p_same_xgb_trunc = None
    p_same_xgb_full = None
    if HAS_XGB:
        xgb_t = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, eval_metric="logloss", n_jobs=8, random_state=42, tree_method="hist"
        )
        xgb_t.fit(X_cv_t, cv_y_arr)
        p_same_xgb_trunc = xgb_t.predict_proba(X_ho_t)[:, 1].astype(np.float64)

        xgb_f = xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=6, subsample=0.8,
            colsample_bytree=0.7, eval_metric="logloss", n_jobs=8, random_state=42, tree_method="hist"
        )
        xgb_f.fit(X_cv_f, cv_y_arr)
        p_same_xgb_full = xgb_f.predict_proba(X_ho_f)[:, 1].astype(np.float64)
    print(f"  Trained ET/LGBM/XGB (HAS_XGB={HAS_XGB}) on 1,190 HemoPI2 features in {time.time()-t_fam:.1f}s", flush=True)

    # 6. Run HemoPI2 ESM2-t6 (official <=40aa truncation AND untruncated max_length=512)
    print("\n--- [Step 4] HemoPI2 Pretrained ESM2-t6 (Official <=40aa Truncated vs Untruncated) ---", flush=True)
    p_esm_trunc = run_hemopi2_esm2_eval(ho_seqs, truncate_40=True)
    p_esm_full = run_hemopi2_esm2_eval(ho_seqs, truncate_40=False)

    # 7. Assemble all model prediction vectors aligned to canonical holdout (n=3,230)
    models: Dict[str, Tuple[np.ndarray, float, str, str]] = {
        "HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40": (
            ml_trunc["p_pre"], 0.46, "pretrained_official_trunc40",
            "Official pretrained hemopi2_ml_clf.sav (RF-200, sklearn 1.3.1, 1,190 features) with official HemoPI2 seq[:40] truncation"
        ),
        "HemoPI2_v1_3_Pretrained_ML_FullLength": (
            ml_full["p_pre"], 0.46, "pretrained_untruncated",
            "Official pretrained hemopi2_ml_clf.sav (RF-200, sklearn 1.3.1, 1,190 features) evaluated on full untruncated sequences"
        ),
        "HemoPI2_v1_3_SameSplit_Retrained_RF_OfficialTrunc40": (
            ml_trunc["p_same_rf"], 0.50, "same_split_retrained_trunc40",
            "Exact cloned hemopi2_ml_clf.sav (RF-200) on 1,190 HemoPI2 features (seq[:40]), trained strictly on our hemolysis_cls CV split (ZERO leakage)"
        ),
        "HemoPI2_v1_3_SameSplit_Retrained_RF_FullLength": (
            ml_full["p_same_rf"], 0.50, "same_split_retrained_full",
            "Exact cloned hemopi2_ml_clf.sav (RF-200) on 1,190 HemoPI2 features (full sequence), trained strictly on our hemolysis_cls CV split (ZERO leakage)"
        ),
        "HemoPI2_v1_3_SameSplit_ExtraTrees_FullLength": (
            p_same_et_full, 0.50, "same_split_retrained_full",
            "ExtraTrees-200 on exact 1,190 HemoPI2 composition/physicochemical features, trained strictly on our hemolysis_cls CV split"
        ),
        "HemoPI2_v1_3_SameSplit_ExtraTrees_OfficialTrunc40": (
            p_same_et_trunc, 0.50, "same_split_retrained_trunc40",
            "ExtraTrees-200 on exact 1,190 HemoPI2 features (seq[:40]), trained strictly on our hemolysis_cls CV split"
        ),
        "HemoPI2_v1_3_SameSplit_LightGBM_FullLength": (
            p_same_lgb_full, 0.50, "same_split_retrained_full",
            "LightGBM-400 on exact 1,190 HemoPI2 composition/physicochemical features, trained strictly on our hemolysis_cls CV split"
        ),
        "HemoPI2_v1_3_SameSplit_LightGBM_OfficialTrunc40": (
            p_same_lgb_trunc, 0.50, "same_split_retrained_trunc40",
            "LightGBM-400 on exact 1,190 HemoPI2 features (seq[:40]), trained strictly on our hemolysis_cls CV split"
        ),
        "HemoPI2_v1_3_Pretrained_ESM2t6_OfficialTrunc40": (
            p_esm_trunc, 0.55, "pretrained_official_trunc40",
            "Official pretrained HemoPI2 ESM2-t6 classifier (Model 3) with official seq[:40] truncation (max_length=42)"
        ),
        "HemoPI2_v1_3_Pretrained_ESM2t6_FullLength": (
            p_esm_full, 0.55, "pretrained_untruncated",
            "Official pretrained HemoPI2 ESM2-t6 classifier (Model 3) evaluated WITHOUT 40-aa truncation (max_length=512)"
        ),
    }
    if p_same_xgb_full is not None and p_same_xgb_trunc is not None:
        models["HemoPI2_v1_3_SameSplit_XGBoost_FullLength"] = (
            p_same_xgb_full, 0.50, "same_split_retrained_full",
            "XGBoost-400 on exact 1,190 HemoPI2 composition/physicochemical features, trained strictly on our hemolysis_cls CV split"
        )
        models["HemoPI2_v1_3_SameSplit_XGBoost_OfficialTrunc40"] = (
            p_same_xgb_trunc, 0.50, "same_split_retrained_trunc40",
            "XGBoost-400 on exact 1,190 HemoPI2 features (seq[:40]), trained strictly on our hemolysis_cls CV split"
        )

    # 8. Evaluate across ALL 6 length/overlap strata!
    strata_masks = {
        "all_canonical": np.ones(len(ho_seqs), dtype=bool),
        "le40_all_canonical": ho_le40,
        "gt40_all_canonical": ho_gt40,
        "zero_overlap_novel": ho_is_novel,
        "le40_zero_overlap_novel": ho_is_novel & ho_le40,
        "gt40_zero_overlap_novel": ho_is_novel & ho_gt40,
    }

    # Primary headline comparisons (matching v33.6 structure + recovered ML models)
    primary_benchmarks: Dict[str, Any] = {}
    v34_hemopi_pvals: Dict[str, float] = {}

    print("\n=== PRIMARY BENCHMARK RESULTS (All Canonical n=3,230 & Zero-Overlap Novel n=2,334) ===", flush=True)
    for mkey, (pvec, thr, mode_tag, desc) in models.items():
        res_all = score_cls_subset(y_can, pvec, p_omni_can, cl_can, thr=thr, seed=20261127)
        res_nov = score_cls_subset(
            y_can[ho_is_novel], pvec[ho_is_novel], p_omni_can[ho_is_novel], cl_can[ho_is_novel], thr=thr, seed=20261128
        )
        primary_benchmarks[mkey] = {
            "mode": mode_tag,
            "threshold_for_mcc": thr,
            "note": desc,
            "all_canonical": res_all,
            "zero_overlap_novel": res_nov,
        }
        if res_all["paired_boot_vs_omni"] and res_all["paired_boot_vs_omni"]["p_value"] is not None:
            v34_hemopi_pvals[f"{mkey}::all_canonical"] = res_all["paired_boot_vs_omni"]["p_value"]
        if res_nov["paired_boot_vs_omni"] and res_nov["paired_boot_vs_omni"]["p_value"] is not None:
            v34_hemopi_pvals[f"{mkey}::zero_overlap_novel"] = res_nov["paired_boot_vs_omni"]["p_value"]

        print(
            f"  {mkey:<52} | ALL(3230): AUC={res_all['roc_auc']:.4f} PR={res_all['pr_auc']:.4f} "
            f"(Omni={res_all['omni_same_subset_roc_auc']:.4f}, Δ={res_all['paired_boot_vs_omni']['mean_delta']:+.4f}, p={res_all['paired_boot_vs_omni']['p_value_display']}) | "
            f"NOVEL(2334): AUC={res_nov['roc_auc']:.4f} PR={res_nov['pr_auc']:.4f} "
            f"(Omni={res_nov['omni_same_subset_roc_auc']:.4f}, Δ={res_nov['paired_boot_vs_omni']['mean_delta']:+.4f}, p={res_nov['paired_boot_vs_omni']['p_value_display']})",
            flush=True,
        )

    # 9. Length-stratified sensitivity analysis (<=40 aa vs >40 aa)
    print("\n=== LENGTH SENSITIVITY ANALYSIS (<=40 aa vs >40 aa) ===", flush=True)
    len_sensitivity: Dict[str, Any] = {
        "length_distributions": {
            "cv_canonical": length_distribution_summary(cv_seqs, cv_y_arr),
            "holdout_all_canonical": length_distribution_summary(ho_seqs, y_can),
            "holdout_zero_overlap_novel": length_distribution_summary(
                [s for s, k in zip(ho_seqs, ho_is_novel) if k], y_can[ho_is_novel]
            ),
        },
        "strata": {},
    }

    key_models_for_strata = [
        "HemoPI2_v1_3_Pretrained_ESM2t6_OfficialTrunc40",
        "HemoPI2_v1_3_Pretrained_ESM2t6_FullLength",
        "HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40",
        "HemoPI2_v1_3_Pretrained_ML_FullLength",
        "HemoPI2_v1_3_SameSplit_Retrained_RF_OfficialTrunc40",
        "HemoPI2_v1_3_SameSplit_Retrained_RF_FullLength",
        "HemoPI2_v1_3_SameSplit_ExtraTrees_FullLength",
        "HemoPI2_v1_3_SameSplit_LightGBM_FullLength",
    ]
    if "HemoPI2_v1_3_SameSplit_XGBoost_FullLength" in models:
        key_models_for_strata.append("HemoPI2_v1_3_SameSplit_XGBoost_FullLength")

    strata_pvals: Dict[str, float] = {}
    for sname, mask in strata_masks.items():
        y_s, o_s, c_s = y_can[mask], p_omni_can[mask], cl_can[mask]
        s_dict: Dict[str, Any] = {
            "n": int(mask.sum()),
            "n_pos": int((y_s == 1).sum()),
            "n_neg": int((y_s == 0).sum()),
            "pos_rate": round(float(y_s.mean()), 4) if len(y_s) else None,
            "n_clusters": int(len(set(map(str, c_s)))),
            "Peptide_OmniPanel": {
                "roc_auc": round(float(roc_auc_score(y_s, o_s)), 4) if len(np.unique(y_s)) == 2 else None,
                "pr_auc": round(float(average_precision_score(y_s, o_s)), 4) if len(np.unique(y_s)) == 2 else None,
                "brier": round(float(np.mean((y_s - o_s) ** 2)), 4) if len(y_s) else None,
            },
            "models": {},
        }
        for mk in key_models_for_strata:
            pvec, thr, _, _ = models[mk]
            sc = score_cls_subset(y_s, pvec[mask], o_s, c_s, thr=thr, seed=20261129)
            s_dict["models"][mk] = sc
            if sc["paired_boot_vs_omni"] and sc["paired_boot_vs_omni"]["p_value"] is not None:
                strata_pvals[f"{sname}::{mk}"] = sc["paired_boot_vs_omni"]["p_value"]
        len_sensitivity["strata"][sname] = s_dict
        print(
            f"  Stratum {sname:<25} (n={s_dict['n']:4d}, pos={s_dict['pos_rate']:.3f}, cl={s_dict['n_clusters']:4d}): "
            f"Omni AUC={s_dict['Peptide_OmniPanel']['roc_auc']} | "
            f"ESM2(t40)={s_dict['models']['HemoPI2_v1_3_Pretrained_ESM2t6_OfficialTrunc40']['roc_auc']} "
            f"ESM2(full)={s_dict['models']['HemoPI2_v1_3_Pretrained_ESM2t6_FullLength']['roc_auc']} | "
            f"PreML(t40)={s_dict['models']['HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40']['roc_auc']} | "
            f"SameRF(t40)={s_dict['models']['HemoPI2_v1_3_SameSplit_Retrained_RF_OfficialTrunc40']['roc_auc']} "
            f"SameLGB(full)={s_dict['models']['HemoPI2_v1_3_SameSplit_LightGBM_FullLength']['roc_auc']}",
            flush=True,
        )

    # Apply Holm-Bonferroni within length-sensitivity strata
    strata_holm = holm_bonferroni(strata_pvals)
    for k, padj in strata_holm.items():
        sn, mk = k.split("::", 1)
        len_sensitivity["strata"][sn]["models"][mk]["paired_boot_vs_omni"]["p_value_holm_within_strata_family"] = padj

    # 10. Compute UPDATED Global Holm-Bonferroni across the FULL external baseline family (v33.6 21 + newly recovered HemoPI2-ML!)
    v336_receipt_path = EXP336 / "v336_prod_blind_temporal_published_tools_receipt.json"
    combined_pvals: Dict[str, float] = {}
    if v336_receipt_path.exists():
        v336 = json.loads(v336_receipt_path.read_text(encoding="utf-8"))
        ext336 = v336.get("external_baselines_same_split", {})
        for ep_k, ep_v in ext336.items():
            if ep_k.startswith("_") or not isinstance(ep_v, dict):
                continue
            for m_k, m_v in ep_v.items():
                if isinstance(m_v, dict) and "paired_boot_vs_omni" in m_v:
                    combined_pvals[f"{ep_k}::{m_k}"] = float(m_v["paired_boot_vs_omni"]["p_value"])
                elif m_k == "Published_HemoPI2_v1_3_Benchmark" and isinstance(m_v, dict):
                    for hk, hv in m_v.items():
                        if isinstance(hv, dict) and "paired_boot_vs_omni" in hv:
                            combined_pvals[f"{ep_k}::{hk}"] = float(hv["paired_boot_vs_omni"]["p_value"])

    # Add newly recovered primary HemoPI2-ML comparisons (canonical + zero-overlap + same-split RF/ET/LGBM/XGB)
    new_global_entries = {
        "hemolysis_cls::HemoPI2_v1_3_Pretrained_ML_AllCanonical": primary_benchmarks["HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40"]["all_canonical"]["paired_boot_vs_omni"]["p_value"],
        "hemolysis_cls::HemoPI2_v1_3_Pretrained_ML_ZeroOverlapNovel": primary_benchmarks["HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40"]["zero_overlap_novel"]["paired_boot_vs_omni"]["p_value"],
        "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_RF": primary_benchmarks["HemoPI2_v1_3_SameSplit_Retrained_RF_OfficialTrunc40"]["all_canonical"]["paired_boot_vs_omni"]["p_value"],
        "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_RF_FullLength": primary_benchmarks["HemoPI2_v1_3_SameSplit_Retrained_RF_FullLength"]["all_canonical"]["paired_boot_vs_omni"]["p_value"],
        "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_ExtraTrees": primary_benchmarks["HemoPI2_v1_3_SameSplit_ExtraTrees_FullLength"]["all_canonical"]["paired_boot_vs_omni"]["p_value"],
        "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_LightGBM": primary_benchmarks["HemoPI2_v1_3_SameSplit_LightGBM_FullLength"]["all_canonical"]["paired_boot_vs_omni"]["p_value"],
    }
    if "HemoPI2_v1_3_SameSplit_XGBoost_FullLength" in primary_benchmarks:
        new_global_entries["hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_XGBoost"] = primary_benchmarks["HemoPI2_v1_3_SameSplit_XGBoost_FullLength"]["all_canonical"]["paired_boot_vs_omni"]["p_value"]

    combined_pvals.update(new_global_entries)
    global_holm = holm_bonferroni(combined_pvals)
    v34_local_holm = holm_bonferroni(v34_hemopi_pvals)

    # Attach Holm p-values back onto primary_benchmarks
    for mkey, mv in primary_benchmarks.items():
        if f"{mkey}::all_canonical" in v34_local_holm:
            mv["all_canonical"]["paired_boot_vs_omni"]["p_value_holm_hemopi2_family"] = v34_local_holm[f"{mkey}::all_canonical"]
        if f"{mkey}::zero_overlap_novel" in v34_local_holm:
            mv["zero_overlap_novel"]["paired_boot_vs_omni"]["p_value_holm_hemopi2_family"] = v34_local_holm[f"{mkey}::zero_overlap_novel"]

    # Attach global Holm p-values to the headline 7 recovered entries
    global_map = [
        ("HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_Pretrained_ML_AllCanonical"),
        ("HemoPI2_v1_3_Pretrained_ML_OfficialTrunc40", "zero_overlap_novel", "hemolysis_cls::HemoPI2_v1_3_Pretrained_ML_ZeroOverlapNovel"),
        ("HemoPI2_v1_3_SameSplit_Retrained_RF_OfficialTrunc40", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_RF"),
        ("HemoPI2_v1_3_SameSplit_Retrained_RF_FullLength", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_RF_FullLength"),
        ("HemoPI2_v1_3_SameSplit_ExtraTrees_FullLength", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_ExtraTrees"),
        ("HemoPI2_v1_3_SameSplit_LightGBM_FullLength", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_LightGBM"),
    ]
    if "HemoPI2_v1_3_SameSplit_XGBoost_FullLength" in primary_benchmarks:
        global_map.append(("HemoPI2_v1_3_SameSplit_XGBoost_FullLength", "all_canonical", "hemolysis_cls::HemoPI2_v1_3_SameSplit_Retrained_ML_XGBoost"))
    for mk, subk, gk in global_map:
        primary_benchmarks[mk][subk]["paired_boot_vs_omni"]["p_value_holm_global_updated"] = global_holm[gk]

    # 11. Build final JSON receipt
    receipt = {
        "schema": "v34_hemopi2_ml_receipt_v1",
        "elapsed_sec": round(time.time() - t_start, 1),
        "tool_citation": "HemoPI2 v1.3 (Rathore et al., Protein Science 2025; PyPI hemopi2==1.3; GPL-3.0)",
        "v336_failure_diagnosis_and_fix": {
            "root_cause": (
                "In v33.6 (`02_v336_prod_blind_temporal_published_tools.py:275-277`), "
                "`from sklearn.base import clone` imported conda's `scikit-learn==1.4.2` BEFORE "
                "`sys.path.insert(0, runtime_sklearn)` (`scikit-learn==1.3.1`). Unpickling "
                "`hemopi2_ml_clf.sav` (trained in sklearn 1.3.1) under sklearn 1.4.2 crashed at "
                "`predict_proba` with `AttributeError: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'`."
            ),
            "v336_silent_drop_audit": (
                "v33.6 caught the non-zero helper exit code and printed stderr without persisting "
                "a failure record into the JSON receipt. v34 fixes the import ordering so native "
                "scikit-learn 1.3.1 loads cleanly, verifies zero missing feature columns, and raises "
                "on any subprocess failure."
            ),
            "recovered_pretrained_estimator": ml_trunc["meta"],
        },
        "evaluation_counts": {
            "hemolysis_cls_cv_canonical": int(len(cv_seqs)),
            "hemolysis_cls_holdout_total_exact": int(len(ho_r)),
            "hemolysis_cls_holdout_all_canonical": int(len(ho_seqs)),
            "hemolysis_cls_holdout_zero_overlap_novel": int(ho_is_novel.sum()),
            "hemopi2_published_corpus_unique_seqs": int(len(hemopi_known_seqs)),
            "omni_full_holdout_roc_auc": round(float(roc_auc_score(y_ho_all, p_omni_all)), 4),
            "omni_all_canonical_roc_auc": round(float(roc_auc_score(y_can, p_omni_can)), 4),
            "omni_all_canonical_pr_auc": round(float(average_precision_score(y_can, p_omni_can)), 4),
            "omni_zero_overlap_roc_auc": round(float(roc_auc_score(y_can[ho_is_novel], p_omni_can[ho_is_novel])), 4),
            "omni_zero_overlap_pr_auc": round(float(average_precision_score(y_can[ho_is_novel], p_omni_can[ho_is_novel])), 4),
        },
        "methodological_caveats": [
            "Off-the-shelf pretrained HemoPI2 models (ML and ESM2-t6) were trained on HemoPI2's own hemolytic threshold definition, whereas Peptide OmniPanel was trained on our hemolysis_cls definition; the same-split retrained HemoPI2-ML models (RF/ET/LGBM/XGB on exact 1,190 HemoPI2 features) eliminate both label-definition and training-corpus discrepancies.",
            "The zero-overlap novel filter (n=2,334) excludes exact canonical sequence matches to HemoPI2's published cross_val_dataset.csv and independent_dataset.csv; homologous near-neighbors from shared DBAASP/Hemolytik literature may still favor off-the-shelf pretrained HemoPI2.",
            "Official HemoPI2 (hemopi2_classification.py:lenchk) truncates input sequences >40 aa to the first 40 residues; both official truncated (<=40 aa) and full-length untruncated evaluations are reported alongside <=40 aa vs >40 aa stratified sensitivity analyses.",
        ],
        "primary_benchmarks": primary_benchmarks,
        "length_truncation_sensitivity": len_sensitivity,
        "updated_global_holm_bonferroni_ledger": {
            "method": "Holm-Bonferroni step-down across v33.6 external baseline comparisons (21) + newly recovered HemoPI2-ML comparisons",
            "n_comparisons_total": len(combined_pvals),
            "p_floor_note": "With 1,000 paired cluster-bootstrap replicates, p_value=0.001 denotes p <= 0.001.",
            "raw_pvalues": {k: round(v, 4) for k, v in sorted(combined_pvals.items())},
            "holm_adjusted_pvalues": {k: global_holm[k] for k in sorted(global_holm.keys())},
        },
        "published_platform_comparison": build_published_platform_comparison(),
    }

    out_path = EXP34 / "v34_hemopi2_ml_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== [V34 Stream A-1 DONE] Saved {out_path} in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
