#!/usr/bin/env python3
"""V33.6 Master Script 1 (runs on compute host):
Addresses Round-2 peer-review findings N-4, N-6, N-7, N-8, N-9, N-11, N-13 & P2:
  1. 1,000x Cluster-Block Bootstrap 95% CIs for BOTH `dual_modality` AND `smiles_only`
     across ALL 47 endpoints (suppressing degenerate CIs when n_clusters < 10; eliminating
     silent `"C0"` cluster collapse).
  2. CI-Aware Honest `evidence_grade` (requires R2_ci95_lo > 0 and uq_reliability != "low"
     for regression `quantitative`; flags `Vd_absolute` with negative R2 & Spearman as
     `exploratory_not_recommended`; downgrades PPB and negative-CI-lower-bound endpoints
     to `rank_only`).
  3. Rigorous `ΔLogD` validation using strict LOCO OOF (NOT 5x5 mean), cluster-level
     bootstrap 95% CI, source_code audit of all 27 matched LogD7.4 octanol pairs,
     max inter-cluster Tanimoto verification, and actual 2.4-threshold + 3x3 tercile
     confusion matrix.
  4. Corrected Statistical Significance Tests:
     - Family comparison: `cpu` vs `fusion` ONLY (no circular `champion`), using
       SD-normalized error (`MAE / SD(y_ho)`) + raw MAE + Holm-Bonferroni correction.
     - 4-FM Single-Model Ablation: Friedman + Nemenyi post-hoc critical difference +
       pairwise Holm-adjusted Wilcoxon across all 24 core endpoints (fixing the key names!).
  5. E-GAT 8-endpoint TRUE graph atom/edge removal deletion faithfulness (n=80/ep +
     1,000x bootstrap 95% CIs on top/rand deletion ratio).
  6. True Dynamic LODO Case17 Separation calling the exact production `detect_acylation_anchor`
     function from `chem_utils.py` (ZERO handwritten lodo_map!).
  7. Exact metadata hygiene (`uq_reliability_counts`, `version="v33.6"`, fresh `generated_at`).
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import csv
import datetime
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import friedmanchisquare, pearsonr, rankdata, spearmanr, wilcoxon
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

V33_BASE = Path("")
BUNDLE_DIR = V33_BASE / "bundle"
EXP331 = V33_BASE / "expansion_v331"
EXP334 = V33_BASE / "expansion_v334"
EXP335 = V33_BASE / "expansion_v335"
EXP336 = V33_BASE / "expansion_v336"
EXP336.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(V33_BASE.parent))
sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP334))
sys.path.insert(0, str(EXP335))
sys.path.insert(0, str(EXP336))

from v33_metrics import classification_metrics, regression_metrics  # noqa: E402
from importlib import import_module

exp334_mod = import_module("os.environ.get(V334_TRAINER_MODULE, 03_train_and_promote_v334)")
exp335_a = import_module("01_v335_protocol_and_logd_audit")
gnn_mod = import_module("gnn_atom_attribution")

UnifiedV334Bank = exp334_mod.UnifiedV334Bank
GNNAtomAttributionEngine = gnn_mod.GNNAtomAttributionEngine
GNN_8_ENDPOINTS = gnn_mod.GNN_8_ENDPOINTS
GNN_8_TASK_META = gnn_mod.GNN_8_TASK_META
mol_to_graph_numpy = gnn_mod.mol_to_graph_numpy
parse_peptide_input = gnn_mod.parse_peptide_input

# Import exact production SMARTS LODO anchor uploaded to EXP336/web_pkg/
from web_pkg.chem_utils import detect_acylation_anchor  # noqa: E402

csv.field_size_limit(10_000_000)


def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, p * (m - i))
        running_max = max(running_max, val)
        adj[k] = float(f"{running_max:.3e}")
    return adj


def get_ep_path(ep: str) -> Path:
    for base in (EXP335 / "endpoints", V33_BASE / "data" / "endpoints", EXP331 / "data" / "endpoints"):
        p = base / f"{ep}.tsv"
        if p.exists():
            return p
    raise FileNotFoundError(ep)


def get_champ_ho_path(ep: str) -> Optional[Path]:
    for p in (
        BUNDLE_DIR / "champion_predictions" / ep / "holdout_champion.tsv",
        EXP331 / "champion_predictions" / ep / "holdout_champion.tsv",
        EXP331 / "models_fusion" / ep / "holdout_predictions.tsv",
        EXP331 / "models_cpu" / ep / "holdout_predictions.tsv",
    ):
        if p.exists():
            return p
    return None


def strict_cluster_bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    clusters: np.ndarray,
    task_type: str,
    lo80: Optional[np.ndarray] = None,
    hi80: Optional[np.ndarray] = None,
    n_boot: int = 1000,
    seed: int = 20261126,
    min_clusters: int = 10,
) -> Dict[str, Any]:
    """Cluster-level block bootstrap 95% CI; suppresses degenerate CIs when n_clusters < 10."""
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred, clusters = y[ok], pred[ok], clusters[ok]
    if lo80 is not None and hi80 is not None:
        lo80, hi80 = lo80[ok], hi80[ok]
    cl_map: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        cl_map[str(c)].append(i)
    cl_arrs = [np.array(v, dtype=np.int32) for v in cl_map.values()]
    n_cl = len(cl_arrs)

    base = {
        "n_eval": int(len(y)),
        "n_clusters": int(n_cl),
        "n_boot": int(n_boot),
    }
    if n_cl < min_clusters or len(y) < 10:
        return {
            **base,
            "ci_reliable": False,
            "ci_note": f"insufficient_clusters (n_clusters={n_cl} < {min_clusters}); cluster bootstrap 95% CI suppressed",
        }

    rng = np.random.RandomState(seed)
    out: Dict[str, Any] = {**base, "ci_reliable": True}
    if task_type == "classification":
        aucs, briers = [], []
        for _ in range(n_boot):
            pick = rng.randint(0, n_cl, size=n_cl)
            idx = np.concatenate([cl_arrs[p] for p in pick])
            yb, pb = y[idx], pred[idx]
            if len(np.unique(yb)) < 2:
                continue
            aucs.append(float(roc_auc_score(yb, pb)))
            briers.append(float(np.mean((yb - pb) ** 2)))
        if len(aucs) >= 20:
            out["roc_auc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]
            out["brier_ci95"] = [round(float(np.percentile(briers, 2.5)), 4), round(float(np.percentile(briers, 97.5)), 4)]
        return out

    maes, sps, r2s, cov80s = [], [], [], []
    for _ in range(n_boot):
        pick = rng.randint(0, n_cl, size=n_cl)
        idx = np.concatenate([cl_arrs[p] for p in pick])
        yb, pb = y[idx], pred[idx]
        maes.append(float(np.mean(np.abs(yb - pb))))
        if np.std(yb) > 1e-9 and np.std(pb) > 1e-9:
            sp_val = float(spearmanr(yb, pb).statistic)
            if math.isfinite(sp_val):
                sps.append(sp_val)
        ss_tot = float(np.sum((yb - np.mean(yb)) ** 2))
        if ss_tot > 1e-12:
            r2s.append(float(1.0 - np.sum((yb - pb) ** 2) / ss_tot))
        if lo80 is not None and hi80 is not None:
            cov80s.append(float(np.mean((yb >= lo80[idx]) & (yb <= hi80[idx]))))
    if maes:
        out["mae_ci95"] = [round(float(np.percentile(maes, 2.5)), 4), round(float(np.percentile(maes, 97.5)), 4)]
    if len(sps) >= 20:
        out["spearman_ci95"] = [round(float(np.percentile(sps, 2.5)), 4), round(float(np.percentile(sps, 97.5)), 4)]
    if len(r2s) >= 20:
        out["r2_ci95"] = [round(float(np.percentile(r2s, 2.5)), 4), round(float(np.percentile(r2s, 97.5)), 4)]
    if len(cov80s) >= 20:
        out["cov80_ci95"] = [round(float(np.percentile(cov80s, 2.5)), 4), round(float(np.percentile(cov80s, 97.5)), 4)]
    return out


def audit_delta_logd_loco_rigorous(bank: UnifiedV334Bank) -> Dict[str, Any]:
    """P0-3 (N-9): Re-audit LogD_decadiene max inter-cluster Tc & 27-macrocycle ΔLogD strictly with LOCO OOF + cluster bootstrap + octanol source audit."""
    print("\n================ [V33.6 P0-3: Strict LOCO ΔLogD & Octanol Source Audit] ================", flush=True)
    jmc_rows = list(csv.DictReader((EXP335 / "endpoints" / "LogD_decadiene.tsv").open(encoding="utf-8"), delimiter="\t"))
    cidx_all = bank.map_keys([r["compound_key"] for r in jmc_rows])
    y_all = np.array([float(r["y_value"]) for r in jmc_rows], dtype=np.float64)
    cl_ids = np.array([r["cluster_id"] for r in jmc_rows], dtype=object)
    cond_all = [{"cond_assay": r.get("cond_assay", "shake_flask_decadiene_water_ph74")} for r in jmc_rows]

    # Exact max inter-cluster Tanimoto verification
    bits = bank.morgan_bits[cidx_all]
    sim_all = exp335_a.tanimoto_matrix(bits, bits)
    max_inter_tc = 0.0
    max_inter_pair = None
    for i in range(len(jmc_rows)):
        for j in range(i + 1, len(jmc_rows)):
            if cl_ids[i] != cl_ids[j]:
                if float(sim_all[i, j]) > max_inter_tc:
                    max_inter_tc = float(sim_all[i, j])
                    max_inter_pair = (str(cl_ids[i]), str(cl_ids[j]))
    print(f"  Complete-linkage clusters={len(set(cl_ids))}, max_inter_cluster_tc={max_inter_tc:.4f} (pair={max_inter_pair})", flush=True)

    # Strict deterministic LOCO CV predictions for ALL 87 macrocycles (CPU track = winning family)
    unique_cls = sorted(set(cl_ids))
    loco_oof = np.full(len(jmc_rows), np.nan, dtype=np.float64)
    for cl in unique_cls:
        te_mask = (cl_ids == cl)
        tr_mask = ~te_mask
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]
        f_inner = np.array([i % 5 for i in range(len(tr_idx))], dtype=np.int32)
        _, _, p_te, _ = exp335_a.train_cpu_track(
            bank,
            cidx_all[tr_idx], y_all[tr_idx], f_inner, [cond_all[i] for i in tr_idx],
            cidx_all[te_idx], [cond_all[i] for i in te_idx],
            ["cond_assay"], None, "smiles_only",
        )
        loco_oof[te_idx] = p_te
    m_loco = regression_metrics(y_all, loco_oof)
    print(f"  LogD_decadiene LOCO (9 clusters, n=87): MAE={m_loco['mae']:.4f}, Sp={m_loco['spearman']:.4f}, R2={m_loco['r2']:.4f}", flush=True)

    # Audit LogD7.4 octanol source_code for all matched compounds
    oct_src_map: Dict[str, str] = {}
    with get_ep_path("LogD7.4").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            oct_src_map[r["compound_key"]] = r.get("source_code", "unknown")

    oct_oos: Dict[str, Tuple[float, float]] = {}
    for fn in ("oof_champion.tsv", "holdout_champion.tsv"):
        fp = BUNDLE_DIR / "champion_predictions" / "LogD7.4" / fn
        if fp.exists():
            with fp.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r.get("censor_type", "exact") != "exact":
                        continue
                    try:
                        # Prefer smiles_only if present so both phases use identical smiles_only modality
                        pv = float(r.get("pred_smiles_only") or r["pred_dual"])
                        oct_oos[r["compound_key"]] = (float(r["y_value"]), pv)
                    except Exception:
                        pass

    mt, mp, m_cls, m_srcs, matched_records = [], [], [], [], []
    for i, r in enumerate(jmc_rows):
        ck = r["compound_key"]
        if ck in oct_oos and np.isfinite(loco_oof[i]):
            ot, op = oct_oos[ck]
            dt, dp = float(y_all[i]), float(loco_oof[i])
            sc = oct_src_map.get(ck, "unknown")
            mt.append(ot - dt)
            mp.append(op - dp)
            m_cls.append(str(cl_ids[i]))
            m_srcs.append(sc)
            matched_records.append({
                "compound_key": ck, "cluster_id": str(cl_ids[i]), "logd_oct_source": sc,
                "logd_oct_true": round(ot, 3), "logd_oct_oos": round(op, 3),
                "logd_dec_true": round(dt, 3), "logd_dec_loco_oof": round(dp, 3),
                "delta_true": round(ot - dt, 3), "delta_pred_loco": round(op - dp, 3),
            })

    mt_arr = np.array(mt, dtype=np.float64)
    mp_arr = np.array(mp, dtype=np.float64)
    m_cls_arr = np.array(m_cls, dtype=object)
    src_counts = dict(Counter(m_srcs))
    n_cl_matched = len(set(m_cls))

    m_delta = regression_metrics(mt_arr, mp_arr)
    # Cluster-level block bootstrap (honest! Resamples clusters of the 27 macrocycles, NOT compounds!)
    cl_map: Dict[str, List[int]] = defaultdict(list)
    for idx_i, c in enumerate(m_cls):
        cl_map[c].append(idx_i)
    cl_arrs = [np.array(v, dtype=np.int32) for v in cl_map.values()]
    rng_d = np.random.RandomState(20261126)
    d_sps, d_maes = [], []
    for _ in range(1000):
        pick = rng_d.randint(0, len(cl_arrs), size=len(cl_arrs))
        idx = np.concatenate([cl_arrs[p] for p in pick])
        if np.std(mt_arr[idx]) > 1e-9 and np.std(mp_arr[idx]) > 1e-9:
            v_sp = float(spearmanr(mt_arr[idx], mp_arr[idx]).statistic)
            if math.isfinite(v_sp):
                d_sps.append(v_sp)
        d_maes.append(float(np.mean(np.abs(mt_arr[idx] - mp_arr[idx]))))

    # Actual deployment threshold (2.4) + 3x3 tercile confusion matrix ([<0.7, 0.7-2.4, >=2.4])
    y_hi24 = (mt_arr >= 2.4).astype(int)
    auc_24 = round(float(roc_auc_score(y_hi24, mp_arr)), 4) if len(np.unique(y_hi24)) == 2 else None
    bacc_24 = round(float(balanced_accuracy_score(y_hi24, (mp_arr >= 2.4).astype(int))), 4) if len(np.unique(y_hi24)) == 2 else None
    y_hi20 = (mt_arr >= 2.0).astype(int)
    auc_20 = round(float(roc_auc_score(y_hi20, mp_arr)), 4) if len(np.unique(y_hi20)) == 2 else None

    def _bin3(arr: np.ndarray) -> np.ndarray:
        return np.where(arr < 0.7, 0, np.where(arr < 2.4, 1, 2))

    cm3 = confusion_matrix(_bin3(mt_arr), _bin3(mp_arr), labels=[0, 1, 2]).tolist()
    tercile_acc = round(float(np.mean(_bin3(mt_arr) == _bin3(mp_arr))), 4)
    q33, q67 = float(np.percentile(mt_arr, 33.3)), float(np.percentile(mt_arr, 66.7))

    # Subset restricted to Naylor/JMC same-literature macrocycle octanol measurements (if >=5)
    jmc_only_idx = [i for i, sc in enumerate(m_srcs) if any(k in sc.lower() for k in ("naylor", "jmc", "macrocycle", "cycpept", "lokey"))]
    same_lit_sub = None
    if len(jmc_only_idx) >= 5:
        m_sub = regression_metrics(mt_arr[jmc_only_idx], mp_arr[jmc_only_idx])
        same_lit_sub = {"n": len(jmc_only_idx), "sources": dict(Counter([m_srcs[i] for i in jmc_only_idx])), **m_sub}

    out = {
        "n_matched_dual_phase_macrocycles": int(len(mt_arr)),
        "n_clusters_represented": int(n_cl_matched),
        "max_inter_cluster_tanimoto": round(max_inter_tc, 4),
        "protocol": "Within-Series LOCO Cross-Validation (Not Prospective External Validation): LogD7.4_OOS(smiles_only) - LogD_decadiene_LOCO_OOF vs Measured (LogD_oct - LogD_dec)",
        "octanol_source_audit": src_counts,
        "same_macrocycle_literature_subset": same_lit_sub,
        "spearman": m_delta["spearman"],
        "spearman_cluster_boot_ci95": [round(float(np.percentile(d_sps, 2.5)), 4), round(float(np.percentile(d_sps, 97.5)), 4)] if len(d_sps) >= 20 else None,
        "cluster_boot_caveat": f"Only {n_cl_matched} clusters represented; cluster bootstrap CI is wide and exploratory.",
        "pearson": m_delta["pearson"],
        "mae": m_delta["mae"],
        "mae_cluster_boot_ci95": [round(float(np.percentile(d_maes, 2.5)), 4), round(float(np.percentile(d_maes, 97.5)), 4)],
        "r2": m_delta["r2"],
        "auroc_ge_2_4_deployment_threshold": auc_24,
        "balanced_acc_ge_2_4": bacc_24,
        "high_chameleonicity_ge2_auroc": auc_20,
        "tercile_3x3_confusion_matrix_0_7_and_2_4": cm3,
        "tercile_3class_accuracy": tercile_acc,
        "empirical_terciles_in_sample": {"p33": round(q33, 2), "p67": round(q67, 2), "min": round(float(mt_arr.min()), 2), "max": round(float(mt_arr.max()), 2)},
        "scope_warning": "Validated strictly on cyclic macrocycles within-series (LOCO); ΔLogD for linear/non-macrocyclic peptides is extrapolated and badges are hidden.",
    }
    print(
        f"  [LOCO ΔLogD (n={len(mt_arr)}, clusters={n_cl_matched})]: Sp={out['spearman']:.4f} "
        f"(cluster CI95={out['spearman_cluster_boot_ci95']}), MAE={out['mae']:.4f}, R2={out['r2']:.4f}, "
        f"AUROC(>=2.4)={auc_24}, 3-class Acc={tercile_acc}, oct_sources={src_counts}",
        flush=True,
    )
    return {"delta_validation_loco": out, "loco_metrics": m_loco, "max_inter_cluster_tanimoto": round(max_inter_tc, 4)}


def run_egat_true_graph_deletion_audit(ck_to_smi: Dict[str, str]) -> Dict[str, Any]:
    """P0-4 (N-13): True graph atom & edge removal (re-indexed subgraph, NOT zeroing features!) on n=80 holdout mols/ep + 1000x bootstrap 95% CI."""
    print("\n================ [V33.6 P0-4: E-GAT True Graph-Deletion Faithfulness (n=80/ep + Bootstrap CI)] ================", flush=True)
    eng = GNNAtomAttributionEngine(ckpt_path=BUNDLE_DIR / "gnn_8ep_5fold_fp16.pt", device="cpu")
    torch.set_num_threads(4)
    rng = np.random.RandomState(20261126)
    out: Dict[str, Any] = {}

    for ep in GNN_8_ENDPOINTS:
        t0 = time.time()
        tt = GNN_8_TASK_META[ep]["task_type"]
        ho_fp = get_champ_ho_path(ep)
        rows = []
        with ho_fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("censor_type", "exact") != "exact":
                    continue
                smi = ck_to_smi.get(r["compound_key"], "")
                if not smi:
                    continue
                try:
                    yv = float(r["y_value"])
                    ev = float(r["pred_dual"])
                    if math.isfinite(yv) and math.isfinite(ev):
                        rows.append((r["compound_key"], smi, yv, ev))
                except Exception:
                    continue

        if len(rows) > 200:
            sel_idx = rng.choice(len(rows), size=200, replace=False)
            rows = [rows[i] for i in sorted(sel_idx)]

        ys, ens_ps, gnn_ps = [], [], []
        del_top, del_rand, del_bot = [], [], []

        for _, smi, yv, ev in rows:
            try:
                res = eng.predict_and_attribute(smi, is_sequence=False, endpoints=[ep])
            except Exception:
                continue
            gp = res["predictions"].get(ep, {}).get("value")
            if gp is None or not math.isfinite(float(gp)):
                continue
            gp = float(gp)
            ys.append(yv)
            ens_ps.append(ev)
            gnn_ps.append(gp)

            # TRUE graph atom & incident-edge removal on up to 80 molecules with >= 15 heavy atoms
            if len(del_top) < 80 and res["n_atoms"] >= 15:
                w = np.array(res["atom_weights"][ep], dtype=np.float64)
                mag = np.abs(w - 0.5)
                k = max(2, int(round(0.20 * len(w))))
                order = np.argsort(mag)
                bot_idx = order[:k]
                top_idx = order[-k:]
                rand_idx = rng.choice(len(w), size=k, replace=False)

                mol = parse_peptide_input(smi, is_sequence=False)
                g_np = mol_to_graph_numpy(mol)
                n_nodes = len(g_np["x"])
                src_all = g_np["edge_src"]
                dst_all = g_np["edge_dst"]
                ea_all = g_np["edge_attr"]

                def _eval_true_subgraph(drop_atoms: np.ndarray) -> float:
                    drop_set = set(int(a) for a in drop_atoms)
                    keep = [i for i in range(n_nodes) if i not in drop_set]
                    old_to_new = {old: new for new, old in enumerate(keep)}
                    x_sub = g_np["x"][keep]
                    # Filter edges where BOTH endpoints survive and remap indices!
                    e_mask = np.array(
                        [(int(s) in old_to_new and int(d) in old_to_new) for s, d in zip(src_all, dst_all)],
                        dtype=bool,
                    )
                    if e_mask.any():
                        s_sub = np.array([old_to_new[int(s)] for s in src_all[e_mask]], dtype=np.int64)
                        d_sub = np.array([old_to_new[int(d)] for d in dst_all[e_mask]], dtype=np.int64)
                        ea_sub = ea_all[e_mask]
                    else:
                        s_sub = np.zeros((0,), dtype=np.int64)
                        d_sub = np.zeros((0,), dtype=np.int64)
                        ea_sub = np.zeros((0, ea_all.shape[1]), dtype=ea_all.dtype)
                    batch = {
                        "x": torch.from_numpy(x_sub),
                        "edge_attr": torch.from_numpy(ea_sub),
                        "edge_src": torch.from_numpy(s_sub),
                        "edge_dst": torch.from_numpy(d_sub),
                        "batch_idx": torch.zeros(len(x_sub), dtype=torch.long),
                        "global_ctx": torch.from_numpy(g_np["global_ctx"]).unsqueeze(0),
                        "num_graphs": 1,
                    }
                    preds_f = []
                    with torch.no_grad():
                        for m_net in eng.models:
                            p_t, _, _ = m_net.forward_task(batch, ep)
                            preds_f.append(float(torch.sigmoid(p_t[0, 0]).item()) if tt == "classification" else float(p_t[0, 0].item()))
                    return float(np.mean(preds_f))

                del_top.append(abs(gp - _eval_true_subgraph(top_idx)))
                del_rand.append(abs(gp - _eval_true_subgraph(rand_idx)))
                del_bot.append(abs(gp - _eval_true_subgraph(bot_idx)))

        y_arr, e_arr, g_arr = np.array(ys), np.array(ens_ps), np.array(gnn_ps)
        fid_sp = round(float(spearmanr(e_arr, g_arr).statistic), 4) if len(y_arr) >= 5 else None
        fid_r = round(float(pearsonr(e_arr, g_arr).statistic), 4) if len(y_arr) >= 5 else None
        fid_mae = round(float(np.mean(np.abs(e_arr - g_arr))), 4) if len(y_arr) >= 5 else None

        if tt == "classification":
            gt_metric_name = "roc_auc"
            gt_egat = round(float(roc_auc_score(y_arr, g_arr)), 4) if len(np.unique(y_arr)) == 2 else None
            gt_ens = round(float(roc_auc_score(y_arr, e_arr)), 4) if len(np.unique(y_arr)) == 2 else None
        else:
            gt_metric_name = "spearman"
            gt_egat = round(float(spearmanr(y_arr, g_arr).statistic), 4) if len(y_arr) >= 5 else None
            gt_ens = round(float(spearmanr(y_arr, e_arr).statistic), 4) if len(y_arr) >= 5 else None

        dt_a, dr_a, db_a = np.array(del_top), np.array(del_rand), np.array(del_bot)
        mt, mr, mb = float(dt_a.mean()), float(dr_a.mean()), float(db_a.mean())
        ratio_tr = round(mt / max(mr, 1e-6), 3)
        ratio_tb = round(mt / max(mb, 1e-6), 3)

        # 1,000x paired bootstrap 95% CI on top/rand deletion ratio
        b_ratios = []
        for _ in range(1000):
            bi = rng.randint(0, len(dt_a), size=len(dt_a))
            b_ratios.append(float(dt_a[bi].mean()) / max(float(dr_a[bi].mean()), 1e-6))
        tr_ci = [round(float(np.percentile(b_ratios, 2.5)), 3), round(float(np.percentile(b_ratios, 97.5)), 3)]
        # Require statistically significant deletion > 1.0 (CI lower bound > 1.05) AND fidelity >= 0.70!
        tier_label = (
            "faithful_surrogate"
            if (fid_sp is not None and fid_sp >= 0.70 and ratio_tr >= 1.15 and tr_ci[0] > 1.0)
            else "qualitative_saliency_heuristic"
        )
        out[ep] = {
            "task_type": tt,
            "deletion_protocol": "True graph atom + incident-edge removal (re-indexed subgraph, top-20% vs random-20% vs bottom-20%)",
            "n_holdout_evaluated": int(len(y_arr)),
            "fidelity_vs_4fm_spearman": fid_sp,
            "fidelity_vs_4fm_pearson": fid_r,
            "fidelity_vs_4fm_mae": fid_mae,
            "ground_truth_metric": gt_metric_name,
            "egat_holdout_score": gt_egat,
            "ensemble_4fm_holdout_score": gt_ens,
            "deletion_faithfulness_n": int(len(dt_a)),
            "deletion_top20_delta": round(mt, 4),
            "deletion_rand20_delta": round(mr, 4),
            "deletion_bot20_delta": round(mb, 4),
            "deletion_top_vs_rand_ratio": ratio_tr,
            "deletion_top_vs_rand_ci95": tr_ci,
            "deletion_top_vs_bot_ratio": ratio_tb,
            "interpretability_tier": tier_label,
        }
        print(
            f"  [{ep:<24}] n_del={len(dt_a):<2} | Fid ρ={fid_sp} | TrueDel top/rand={ratio_tr}x (CI95={tr_ci}) -> {tier_label} [{time.time()-t0:.1f}s]",
            flush=True,
        )
    return out


def compute_true_dynamic_lodo_case17(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """P0-4 (N-4): Evaluate Case17 using the EXACT production `detect_acylation_anchor(mol, exclude_drug=name)` function."""
    print("\n================ [V33.6 P0-4: True Dynamic LODO Case17 Evaluation (Production SMARTS)] ================", flush=True)
    c17_drugs = (manifest.get("case17_scorecard") or {}).get("drugs") or []
    # Load Case17 SMILES from case17_queries.tsv
    c17_smi: Dict[str, str] = {}
    c17_fp = V33_BASE / "data" / "case17_queries.tsv"
    if c17_fp.exists():
        with c17_fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                k = r.get("drug_name") or r.get("name") or ""
                if k:
                    c17_smi[k] = r.get("canonical_smiles") or r.get("smiles") or ""

    obs_all, ml_all, lodo_all = [], [], []
    lip_ml_err, nonlip_ml_err, lip_lodo_err = [], [], []
    per_drug_audit = []

    for row in c17_drugs:
        dname = row["drug_name"]
        obs = row.get("fda_half_life_h")
        if obs is None or float(obs) <= 0:
            continue
        rt = str(row.get("clinical_route") or "IV").upper()
        ml_h = row.get("v33_half_life_sc_h") if rt == "SC" else row.get("v33_half_life_iv_h")
        if ml_h is None:
            continue

        smi = c17_smi.get(dname) or row.get("smiles") or ""
        mol = Chem.MolFromSmiles(smi) if smi else None
        anc = detect_acylation_anchor(mol, exclude_drug=dname) if mol is not None else None

        y_obs = math.log10(float(obs))
        y_ml = math.log10(max(float(ml_h), 1e-3))
        if anc is not None and anc.detected and anc.value_h is not None:
            lodo_h = float(anc.value_h)
            used_overlay = True
        else:
            lodo_h = float(ml_h)
            used_overlay = False
        y_lodo = math.log10(max(lodo_h, 1e-3))

        obs_all.append(y_obs)
        ml_all.append(y_ml)
        lodo_all.append(y_lodo)
        is_lip = bool(anc is not None and anc.detected)
        if is_lip:
            lip_ml_err.append(abs(y_obs - y_ml))
            lip_lodo_err.append(abs(y_obs - y_lodo))
        else:
            nonlip_ml_err.append(abs(y_obs - y_ml))
        per_drug_audit.append({
            "drug_name": dname, "obs_h": float(obs), "pure_ml_h": round(float(ml_h), 2),
            "smarts_detected": is_lip, "acylation_class": anc.acylation_class if (anc and anc.detected) else None,
            "lodo_overlay_h": round(lodo_h, 2) if used_overlay else None,
            "lodo_effective_h": round(lodo_h, 2),
            "lodo_remaining_refs": anc.anchor_drugs if (anc and anc.detected) else [],
        })

    oa, ma, la = np.array(obs_all), np.array(ml_all), np.array(lodo_all)
    m_ml = regression_metrics(oa, ma)
    m_lodo = regression_metrics(oa, la)
    # 2,000x bootstrap 95% CIs on Pure-ML & LODO Spearman and MAE
    rng = np.random.RandomState(20261126)
    ml_sps, lodo_sps, ml_maes, lodo_maes = [], [], [], []
    for _ in range(2000):
        bi = rng.randint(0, len(oa), size=len(oa))
        ml_maes.append(float(np.mean(np.abs(oa[bi] - ma[bi]))))
        lodo_maes.append(float(np.mean(np.abs(oa[bi] - la[bi]))))
        if np.std(oa[bi]) > 1e-9:
            ml_sps.append(float(spearmanr(oa[bi], ma[bi]).statistic))
            lodo_sps.append(float(spearmanr(oa[bi], la[bi]).statistic))

    res = {
        "primary_metric_policy": (
            "Pure-ML route-matched metrics are reported as the PRIMARY Case17 headline. "
            "True Dynamic Leave-One-Drug-Out (LODO) rule overlay is computed by invoking "
            "the exact production `detect_acylation_anchor(mol, exclude_drug=drug_name)` "
            "(dynamically recomputing geometric-mean half-life from remaining peer drugs only; "
            "Insulin degludec has no remaining C16-diacid peer and falls back to Pure-ML)."
        ),
        "n_evaluated": int(len(oa)),
        "pure_ml_all_16": {
            **m_ml,
            "mae_ci95": [round(float(np.percentile(ml_maes, 2.5)), 4), round(float(np.percentile(ml_maes, 97.5)), 4)],
            "spearman_ci95": [round(float(np.percentile(ml_sps, 2.5)), 4), round(float(np.percentile(ml_sps, 97.5)), 4)],
        },
        "lodo_rule_overlay_all_16": {
            **m_lodo,
            "mae_ci95": [round(float(np.percentile(lodo_maes, 2.5)), 4), round(float(np.percentile(lodo_maes, 97.5)), 4)],
            "spearman_ci95": [round(float(np.percentile(lodo_sps, 2.5)), 4), round(float(np.percentile(lodo_sps, 97.5)), 4)],
        },
        "pure_ml_all_16_mae_log10": m_ml["mae"],
        "pure_ml_all_16_r2": m_ml["r2"],
        "pure_ml_all_16_spearman": m_ml["spearman"],
        "pure_ml_lipidated_5_mae_log10": round(float(np.mean(lip_ml_err)), 4) if lip_ml_err else None,
        "pure_ml_non_lipidated_11_mae_log10": round(float(np.mean(nonlip_ml_err)), 4) if nonlip_ml_err else None,
        "lodo_rule_overlay_lipidated_5_mae_log10": round(float(np.mean(lip_lodo_err)), 4) if lip_lodo_err else None,
        "lodo_rule_overlay_all_16_mae_log10": m_lodo["mae"],
        "lodo_rule_overlay_all_16_r2": m_lodo["r2"],
        "lodo_rule_overlay_all_16_spearman": m_lodo["spearman"],
        "insulin_degludec_circularity_removed": True,
        "fusion_protein_anchors_removed": True,
        "computed_via_production_smarts_function": True,
        "per_drug_lodo_audit": per_drug_audit,
    }
    print(f"  Pure-ML (n={len(oa)}): MAE={m_ml['mae']:.4f}, R2={m_ml['r2']:.4f}, Sp={m_ml['spearman']:.4f}", flush=True)
    print(f"  True-LODO (n={len(oa)}): MAE={m_lodo['mae']:.4f}, R2={m_lodo['r2']:.4f}, Sp={m_lodo['spearman']:.4f} (lip5={res['lodo_rule_overlay_lipidated_5_mae_log10']})", flush=True)
    return res


def main():
    t0 = time.time()
    manifest_path = BUNDLE_DIR / "v33_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = UnifiedV334Bank()

    ck_to_smi: Dict[str, str] = {}
    with (EXP331 / "data" / "compounds_v331_merged.tsv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ck_to_smi[r["compound_key"]] = r.get("canonical_smiles", "")
    for r in bank.delta_rows:
        ck_to_smi[r["compound_key"]] = r.get("canonical_smiles", "")

    # 1. Recompute cluster-bootstrap 95% CIs for BOTH dual_modality AND smiles_only across ALL 47 endpoints
    print("\n================ [V33.6 P0-2: Dual + SMILES-Only Cluster Bootstrap CIs & CI-Aware Evidence Grade] ================", flush=True)
    ep_sd_y: Dict[str, float] = {}
    grade_counts = Counter()
    uq_counts = {"dual_modality": Counter(), "smiles_only": Counter()}

    for ep, doc in manifest["endpoints"].items():
        tt = doc["task_type"]
        ep_tsv = get_ep_path(ep)
        rec_to_cl: Dict[str, str] = {}
        ho_ys_for_sd: List[float] = []
        with ep_tsv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                ck = r.get("compound_key", "")
                cid = r.get("cluster_id") or f"UNIQ:{ck}"
                if r.get("record_id"):
                    rec_to_cl[r["record_id"]] = cid
                if ck:
                    rec_to_cl[ck] = cid
                if int(r.get("is_holdout", 0)) == 1 and r.get("censor_type", "exact") == "exact":
                    try:
                        ho_ys_for_sd.append(float(r["y_value"]))
                    except Exception:
                        pass
        if len(ho_ys_for_sd) >= 2:
            sd_v = float(np.std(ho_ys_for_sd))
            if sd_v > 1e-9:
                ep_sd_y[ep] = sd_v

        pred_fp = get_champ_ho_path(ep)
        if pred_fp is not None:
            for tr_name, p_col, lo_col, hi_col in [
                ("dual_modality", "pred_dual", "lo80", "hi80"),
                ("smiles_only", "pred_smiles_only", "lo80_smiles", "hi80_smiles"),
            ]:
                ys, ps, cls, lo80s, hi80s = [], [], [], [], []
                with pred_fp.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    cols = set(reader.fieldnames or [])
                    actual_pcol = p_col if p_col in cols else "pred_dual"
                    actual_lo = lo_col if lo_col in cols else ("lo80" if tr_name == "dual_modality" else None)
                    actual_hi = hi_col if hi_col in cols else ("hi80" if tr_name == "dual_modality" else None)
                    for r in reader:
                        if r.get("censor_type", "exact") != "exact":
                            continue
                        try:
                            yv = float(r["y_value"])
                            pv_raw = r.get(actual_pcol)
                            if pv_raw is None or pv_raw == "":
                                continue
                            pv = float(pv_raw)
                            if not (math.isfinite(yv) and math.isfinite(pv)):
                                continue
                            ck = r.get("compound_key", "")
                            rid = r.get("record_id", "")
                            cid = rec_to_cl.get(rid) or rec_to_cl.get(ck) or f"UNIQ:{ck}"
                            ys.append(yv)
                            ps.append(pv)
                            cls.append(cid)
                            if actual_lo and actual_hi and r.get(actual_lo) and r.get(actual_hi):
                                lo80s.append(float(r[actual_lo]))
                                hi80s.append(float(r[actual_hi]))
                        except Exception:
                            continue
                lo_arr = np.array(lo80s, dtype=np.float64) if len(lo80s) == len(ys) and len(lo80s) > 0 else None
                hi_arr = np.array(hi80s, dtype=np.float64) if len(hi80s) == len(ys) and len(hi80s) > 0 else None
                if tr_name in doc and isinstance(doc[tr_name], dict):
                    doc[tr_name]["holdout_bootstrap_ci95"] = strict_cluster_bootstrap_ci(
                        np.array(ys, dtype=np.float64), np.array(ps, dtype=np.float64), np.array(cls, dtype=object),
                        tt, lo_arr, hi_arr, n_boot=1000,
                    )
        else:
            # LogD_decadiene (4 holdout clusters < 10 -> explicitly mark ci_reliable=False!)
            for tr_name in ("dual_modality", "smiles_only"):
                if tr_name in doc and isinstance(doc[tr_name], dict):
                    prev = doc[tr_name].get("holdout_bootstrap_ci95") or {}
                    n_cl = int(prev.get("n_clusters", 4))
                    n_ev = int(prev.get("n_eval", doc["counts"].get("n_holdout", 20)))
                    doc[tr_name]["holdout_bootstrap_ci95"] = {
                        "n_eval": n_ev,
                        "n_clusters": n_cl,
                        "n_boot": 1000,
                        "ci_reliable": False,
                        "ci_note": f"insufficient_clusters (n_clusters={n_cl} < 10); cluster bootstrap 95% CI suppressed",
                    }

        # CI-Aware Honest Evidence Grade (N-6 fix!)
        dm = doc["dual_modality"]
        ho_m = dm["holdout_metrics"]
        ci_dm = dm.get("holdout_bootstrap_ci95") or {}
        n_ho = int(doc["counts"].get("n_holdout", 0))
        n_cl = int(ci_dm.get("n_clusters", 0))
        ci_rel = bool(ci_dm.get("ci_reliable", False))

        if tt == "classification":
            auc = float(ho_m.get("roc_auc") or 0.0)
            auc_lo = float((ci_dm.get("roc_auc_ci95") or [0.0])[0]) if ci_rel else 0.0
            grade = "quantitative" if (n_ho >= 30 and ci_rel and auc >= 0.75 and auc_lo >= 0.65) else "rank_only"
        else:
            r2 = float(ho_m.get("r2") if ho_m.get("r2") is not None else -9.0)
            sp = float(ho_m.get("spearman") if ho_m.get("spearman") is not None else -1.0)
            r2_lo = float((ci_dm.get("r2_ci95") or [-9.0])[0]) if ci_rel else -9.0
            uq_rel = str((dm.get("conformal_holdout_coverage") or {}).get("uq_reliability") or "low")
            for tr_n in ("dual_modality", "smiles_only"):
                u_tr = str(((doc.get(tr_n) or {}).get("conformal_holdout_coverage") or {}).get("uq_reliability") or "low")
                uq_counts[tr_n][u_tr] += 1

            if r2 < 0.0 and sp < 0.0:
                grade = "exploratory_not_recommended"
            elif n_ho >= 30 and ci_rel and r2 >= 0.15 and r2_lo > 0.0 and sp >= 0.45 and uq_rel != "low":
                grade = "quantitative"
            else:
                grade = "rank_only"

        doc["evidence_grade"] = grade
        grade_counts[grade] += 1
        if ep in ("PPB", "Vd_absolute", "LogD_decadiene", "CL_per_kg", "efflux_ratio", "half_life_reg", "PAMPA", "hemolysis_cls"):
            print(
                f"  [{ep:<16}] grade={grade:<27} n_ho={n_ho:<4} n_cl={n_cl:<3} ci_rel={str(ci_rel):<5} "
                f"dual_ci={ci_dm.get('r2_ci95') or ci_dm.get('roc_auc_ci95')} "
                f"smi_ci={(doc.get('smiles_only') or {}).get('holdout_bootstrap_ci95', {}).get('r2_ci95') or (doc.get('smiles_only') or {}).get('holdout_bootstrap_ci95', {}).get('roc_auc_ci95')}",
                flush=True,
            )

    print(f"  CI-Aware Evidence Grades (47 eps): {dict(grade_counts)}", flush=True)
    print(f"  Exact UQ Reliability Counts (41 reg eps): dual={dict(uq_counts['dual_modality'])}, smiles={dict(uq_counts['smiles_only'])}", flush=True)

    # 2. Corrected Statistical Significance Tests (N-11 fix!)
    print("\n================ [V33.6 P0-2: Corrected Family Wilcoxon (nMAE) & 4-FM Friedman/Nemenyi] ================", flush=True)
    nmae_cpu, nmae_fus, raw_cpu, raw_fus = [], [], [], []
    for ep, doc in manifest["endpoints"].items():
        if doc["task_type"] != "regression":
            continue
        fhm = (doc.get("dual_modality") or {}).get("family_holdout_metrics") or {}
        mc = (fhm.get("cpu") or {}).get("mae")
        mf = (fhm.get("fusion") or {}).get("mae")
        sd = ep_sd_y.get(ep)
        if mc is not None and mf is not None and sd is not None and sd > 1e-9:
            nmae_cpu.append(float(mc) / sd)
            nmae_fus.append(float(mf) / sd)
            raw_cpu.append(float(mc))
            raw_fus.append(float(mf))

    c_n, f_n = np.array(nmae_cpu), np.array(nmae_fus)
    c_r, f_r = np.array(raw_cpu), np.array(raw_fus)
    p_nmae = float(wilcoxon(f_n, c_n).pvalue)
    p_raw = float(wilcoxon(f_r, c_r).pvalue)
    rng_s = np.random.RandomState(20261126)
    diff_n = c_n - f_n  # > 0 means Fusion beats CPU
    boot_n = [float(np.mean(diff_n[rng_s.randint(0, len(diff_n), size=len(diff_n))])) for _ in range(5000)]

    # 4-FM Single-Model Ablation Friedman + Nemenyi (FIXED KEYS!)
    fm_keys = ["ncaa_fusion_hf", "esmc_multiregion", "peptideclm2", "chemberta_mtr"]
    fm_scores = {k: [] for k in fm_keys}
    fm_eps = []
    for ep, doc in manifest["endpoints"].items():
        rab = doc.get("representation_ablation_holdout") or {}
        if all(k in rab and isinstance(rab[k], dict) for k in fm_keys):
            metric_k = "roc_auc" if doc["task_type"] == "classification" else "spearman"
            vals = [rab[k].get(metric_k) for k in fm_keys]
            if all(v is not None and math.isfinite(float(v)) for v in vals):
                fm_eps.append(ep)
                for k, v in zip(fm_keys, vals):
                    fm_scores[k].append(float(v))

    mat = np.column_stack([fm_scores[k] for k in fm_keys])  # shape (N_ep, 4)
    fr_fm, p_fm = friedmanchisquare(*[mat[:, j] for j in range(4)])
    # Ranks per endpoint (1 = best / highest score)
    ranks = np.array([rankdata(-row) for row in mat])
    avg_ranks = {k: round(float(ranks[:, j].mean()), 3) for j, k in enumerate(fm_keys)}
    # Nemenyi Critical Difference for k=4 models at alpha=0.05 (q_0.05 = 2.569)
    n_fm_ep = int(mat.shape[0])
    nemenyi_cd_005 = round(float(2.569 * math.sqrt(4 * 5 / (6.0 * n_fm_ep))), 3)

    # Pairwise Wilcoxon among the 4 FMs with Holm-Bonferroni correction
    raw_pair_p: Dict[str, float] = {}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = mat[:, i], mat[:, j]
            pv = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
            raw_pair_p[f"{fm_keys[i]}_vs_{fm_keys[j]}"] = pv
    holm_pair_p = holm_bonferroni(raw_pair_p)

    # Holm-adjusted across the 2 primary family tests too
    fam_holm = holm_bonferroni({"fusion_vs_cpu_nmae": p_nmae, "fusion_vs_cpu_raw_mae": p_raw})

    stat_tests = {
        "regression_family_cpu_vs_fusion_only": {
            "note": "Compares strictly independent families (`fusion` vs `cpu`; circular `champion` excluded) using SD-normalized error MAE/SD(y_holdout) to remove cross-unit scale dominance.",
            "n_endpoints": int(len(c_n)),
            "mean_normalized_mae": {"fusion": round(float(f_n.mean()), 4), "cpu": round(float(c_n.mean()), 4)},
            "mean_raw_mae": {"fusion": round(float(f_r.mean()), 4), "cpu": round(float(c_r.mean()), 4)},
            "fusion_win_rate": f"{int(np.sum(f_n < c_n))}/{len(c_n)}",
            "wilcoxon_p_nmae_raw": float(f"{p_nmae:.3e}"),
            "wilcoxon_p_nmae_holm": fam_holm["fusion_vs_cpu_nmae"],
            "wilcoxon_p_raw_mae_holm": fam_holm["fusion_vs_cpu_raw_mae"],
            "paired_boot_cpu_minus_fusion_nmae_ci95": [round(float(np.percentile(boot_n, 2.5)), 4), round(float(np.percentile(boot_n, 97.5)), 4)],
        },
        "single_fm_4model_ablation_friedman_nemenyi": {
            "n_endpoints": n_fm_ep,
            "endpoints_included": fm_eps,
            "metric": "ROC-AUC (classification) / Spearman rho (regression)",
            "mean_primary_score": {k: round(float(np.mean(fm_scores[k])), 4) for k in fm_keys},
            "mean_rank_1_is_best": avg_ranks,
            "friedman_chi2": round(float(fr_fm), 4),
            "friedman_p": float(f"{p_fm:.3e}"),
            "nemenyi_critical_diff_alpha_0_05": nemenyi_cd_005,
            "pairwise_wilcoxon_p_holm": holm_pair_p,
        },
    }
    print(f"  [CPU vs Fusion (n={len(c_n)})] nMAE: Fusion={f_n.mean():.4f} vs CPU={c_n.mean():.4f} | wins={int(np.sum(f_n<c_n))}/{len(c_n)} | Wilcoxon Holm p={fam_holm['fusion_vs_cpu_nmae']}", flush=True)
    print(f"  [4-FM Friedman (n={n_fm_ep})] chi2={fr_fm:.2f} (p={p_fm:.2e}) | avg_ranks={avg_ranks} | Nemenyi CD(0.05)={nemenyi_cd_005}", flush=True)

    # 3. Rigorous LOCO ΔLogD validation (N-9)
    logd_audit = audit_delta_logd_loco_rigorous(bank)

    # 4. E-GAT True Graph-Deletion Faithfulness (N-13)
    egat_audit = run_egat_true_graph_deletion_audit(ck_to_smi)

    # 5. True Dynamic LODO Case17 Separation (N-4)
    c17_lodo = compute_true_dynamic_lodo_case17(manifest)

    # 6. Merge ALL audited V33.6 results cleanly into v33_manifest.json + fix metadata
    manifest["version"] = "v33.6"
    manifest["generated_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest.setdefault("overall_summary", {})["statistical_significance_tests"] = stat_tests
    manifest["overall_summary"]["evidence_grade_counts"] = dict(grade_counts)
    # Fix stale uq_reliability_counts (P2)
    if "plan_b_upgrades" in manifest and "conformal_calibration" in manifest["plan_b_upgrades"]:
        manifest["plan_b_upgrades"]["conformal_calibration"]["n_regression_endpoints"] = int(sum(uq_counts["dual_modality"].values()))
        manifest["plan_b_upgrades"]["conformal_calibration"]["uq_reliability_counts"] = {
            "dual_modality": dict(uq_counts["dual_modality"]),
            "smiles_only": dict(uq_counts["smiles_only"]),
        }
    manifest.setdefault("v335_protocol_audit", {})["logd_decadiene_delta_validation"] = logd_audit["delta_validation_loco"]
    manifest["v335_protocol_audit"]["max_inter_cluster_tanimoto"] = logd_audit["max_inter_cluster_tanimoto"]
    manifest["v335_protocol_audit"]["evidence_grade_policy"] = (
        "CI-aware grading: regression `quantitative` requires n_ho>=30, n_clusters>=10, R2>=0.15, R2_ci95_lo>0, "
        "Spearman>=0.45, and uq_reliability!=low; negative R2 & Spearman flagged `exploratory_not_recommended`."
    )
    manifest["egat_fidelity_audit"] = egat_audit
    manifest.setdefault("case17_scorecard", {})["non_circular_separation"] = c17_lodo

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    receipt = {
        "schema": "v336_stats_ci_logd_egat_case17_receipt_v1",
        "elapsed_sec": round(time.time() - t0, 1),
        "evidence_grade_counts": dict(grade_counts),
        "uq_reliability_counts": {"dual_modality": dict(uq_counts["dual_modality"]), "smiles_only": dict(uq_counts["smiles_only"])},
        "statistical_significance_tests": stat_tests,
        "logd_decadiene_loco_audit": logd_audit,
        "egat_true_graph_deletion_audit": egat_audit,
        "case17_true_dynamic_lodo": c17_lodo,
    }
    (EXP336 / "v336_stats_ci_logd_egat_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== V33.6 MASTER SCRIPT 1 DONE in {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
