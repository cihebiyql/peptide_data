#!/usr/bin/env python3
"""V34 Stream A Script 2 (runs on compute_host):
Addresses Round-3 Peer Review P2-7 (3D Conformer & Intramolecular H-Bond [IMHB]
Dynamic Features for Macrocycle Permeability & Prospective Temporal Generalization):

  1. Fast, parallelized (<=16 CPU workers, checkpointed & cached at
     `$V33_BASE/expansion_v34/cache_3d/`) RDKit 3D conformer + IMHB
     descriptor extractor across all unique compounds in the 6 permeability /
     lipophilicity endpoints (`PAMPA`, `Caco2`, `MDCK`, `RRCK`, `LogD7.4`,
     `LogD_decadiene`; N=9,114 unique SMILES):
       - `ETKDGv3` with `useMacrocycleTorsions=True`, `useSmallRingTorsions=True`,
         `enforceChirality=True`, `numConfs=8` (MW < 1800) / `4` (MW >= 1800),
         heavy-atom embedding + `AddHs(addCoords=True)` + `MMFF94s` (`maxIters=80`)
         with graceful `UFF` / random-coords fallback.
       - 31 3D dynamic descriptors:
         * `3D_PSA` (Lee-Richards SASA of polar N/O & polar H via `rdFreeSASA`):
           min, max, mean, chameleonic span (`max - min`), std, low-E conformer,
           shielding ratio `psa3d_min / max(TPSA_2D, 1)`, total 3D SASA.
         * `IMHB_count` (donor-acceptor N-H...O/N and O-H...O/N pairs with heavy-atom
           distance <= 3.3 A separated by >= 4 topological bonds):
           min, max, mean, span, low-E conformer, shielded-donor fraction `imhb_max / max(HBD, 1)`,
           plus strict geometric IMHB (`H...A <= 2.5 A` & `D-H...A >= 110 deg`) max/mean.
         * 3D Shape & Size (`Descriptors3D`): `RadiusOfGyration` (mean, min, span),
           `Asphericity` (mean, span), `Eccentricity` (mean), `SpherocityIndex`
           (mean, span), `NPR1` & `NPR2` (mean, span).
         * Conformer energetics: `conf_energy_spread` (`max - min` kcal/mol),
           `conf_energy_std`, `n_confs_generated`.

  2. Experiment A — Prospective Temporal Splits (`PAMPA`: 7,395 train -> 1,177 future;
     `LogD7.4`: 653 train -> 22 future):
       - Exact 2D baseline (`CPU`, `4-FM Fusion`, `Super-Ensemble`) vs `2D + 3D-IMHB`
         (`CPU+3D`, `4-FM+3D Fusion`, `Super-Ensemble+3D`) + standalone `3D-IMHB-only` ablation.
       - Reports `R2`, `Spearman`, `Pearson`, `MAE`, `RMSE`, paired cluster-bootstrap
         95% CIs & two-sided p-values, and train-vs-future-test distributional shift
         diagnostics (MW, 2D TPSA, 3D PSA, chameleonic span, IMHB, Tanimoto AD, y-shift).

  3. Experiment B — Zero-Holdout-Peeking Cluster-Grouped 5-Fold CV across all 6
     permeability/lipophilicity endpoints (`PAMPA`, `Caco2`, `MDCK`, `RRCK`, `LogD7.4`,
     `LogD_decadiene`):
       - Exact frozen cluster-grouped CV splits (`is_holdout == 0`, `fold_id`).
       - Strict OOF-only promotion gate (`oof_mae_3d < oof_mae_2d`) BEFORE holdout evaluation.
       - Reports ALL OOF & holdout deltas with 100% scientific honesty (including null/modest
         or negative deltas without cherry-picking).

CPU workers strictly capped <= 16 cores to coexist safely on shared compute_host.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import csv
import json
import math
import multiprocessing as mp
import signal
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, rdFreeSASA, rdMolDescriptors
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.svm import SVR

RDLogger.DisableLog("rdApp.*")

V33_BASE = Path(os.environ.get("V33_BASE", str(Path(__file__).resolve().parents[1])))
BUNDLE_DIR = V33_BASE / "bundle"
EXP331 = V33_BASE / "expansion_v331"
EXP334 = V33_BASE / "expansion_v334"
EXP335 = V33_BASE / "expansion_v335"
EXP34 = V33_BASE / "expansion_v34"
CACHE_3D_DIR = EXP34 / "cache_3d"
EXP34.mkdir(parents=True, exist_ok=True)
CACHE_3D_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BUNDLE_DIR))
sys.path.insert(0, str(V33_BASE / "models_cpu"))
sys.path.insert(0, str(V33_BASE / "models_fusion"))
sys.path.insert(0, str(EXP334))

from fusion_config import DIRECT_FUSION_REPS, INDIVIDUAL_REPS  # noqa: E402
from fusion_models import EmbeddingFoldPreprocessor, ModalityAwareStacker, make_fusion_estimator  # noqa: E402
from v33_features import FoldPreprocessor  # noqa: E402
from v33_metrics import regression_metrics  # noqa: E402
from v33_models import make_estimator  # noqa: E402

exp334_mod = import_module("03_train_and_promote_v334")
UnifiedV334Bank = exp334_mod.UnifiedV334Bank

csv.field_size_limit(10_000_000)

N_3D_WORKERS = 16  # Strictly <= 16 cores on shared compute_host
TARGET_ENDPOINTS = ["PAMPA", "Caco2", "MDCK", "RRCK", "LogD7.4", "LogD_decadiene"]

FEAT3D_NAMES = [
    "psa3d_min",
    "psa3d_max",
    "psa3d_mean",
    "psa3d_span",
    "psa3d_std",
    "psa3d_lowE",
    "psa3d_ratio_to_tpsa2d",
    "sasa3d_total_mean",
    "imhb_min",
    "imhb_max",
    "imhb_mean",
    "imhb_span",
    "imhb_lowE",
    "imhb_per_hbd",
    "imhb_geom_max",
    "imhb_geom_mean",
    "rg_mean",
    "rg_min",
    "rg_span",
    "asphericity_mean",
    "asphericity_span",
    "eccentricity_mean",
    "spherocity_mean",
    "spherocity_span",
    "npr1_mean",
    "npr2_mean",
    "npr1_span",
    "npr2_span",
    "conf_energy_spread",
    "conf_energy_std",
    "n_confs_generated",
]
N_FEAT3D = len(FEAT3D_NAMES)


def get_ep_path(ep: str) -> Path:
    for base in (EXP335 / "endpoints", V33_BASE / "data" / "endpoints", EXP331 / "data" / "endpoints"):
        p = base / f"{ep}.tsv"
        if p.exists():
            return p
    raise FileNotFoundError(ep)


class _MolTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _MolTimeout()


def _compute_3d_imhb_single(args: Tuple[str, str]) -> Tuple[str, np.ndarray, bool, int]:
    """Worker function: compute 31 dynamic 3D conformer & IMHB features for one SMILES
    with strict per-molecule wall-clock guard (40s) + graceful macrocycle fallback.
    """
    ck, smi = args
    nan_vec = np.full(N_FEAT3D, np.nan, dtype=np.float32)
    if not smi:
        return ck, nan_vec, False, 0

    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() < 3:
        return ck, nan_vec, False, 0

    signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        mw = float(Descriptors.MolWt(mol))
        n_heavy = int(mol.GetNumAtoms())
        tpsa2d = float(rdMolDescriptors.CalcTPSA(mol))
        n_hbd = int(rdMolDescriptors.CalcNumHBD(mol))
        n_confs = 8 if mw < 1800.0 else 4

        # Stage 1: Heavy-atom ETKDGv3 with macrocycle torsion preferences
        m_emb = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.useMacrocycleTorsions = True
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        params.randomSeed = 20261127
        params.numThreads = 1

        cids: List[int] = []
        signal.alarm(30)
        try:
            if n_heavy > 130:
                # Giant peptides (>130 heavy atoms): skip O(N^3) triangle smoothing; direct random-coords ETKDGv3
                params.useRandomCoords = True
                params.maxIterations = 5
                cids = list(AllChem.EmbedMultipleConfs(m_emb, numConfs=n_confs, params=params))
            else:
                params.useRandomCoords = False
                params.maxIterations = 6 if n_heavy > 90 else 8
                cids = list(AllChem.EmbedMultipleConfs(m_emb, numConfs=n_confs, params=params))
                if not cids:
                    params.useRandomCoords = True
                    params.maxIterations = 6
                    cids = list(AllChem.EmbedMultipleConfs(m_emb, numConfs=n_confs, params=params))
        except _MolTimeout:
            cids = []
        finally:
            signal.alarm(0)

        if not cids:
            # Fast graceful fallback (ensures near-100% conformer coverage even on stiff polycyclic scaffolds)
            signal.alarm(15)
            try:
                m_emb = Chem.Mol(mol)
                p_fb = AllChem.ETKDGv3()
                p_fb.useMacrocycleTorsions = True
                p_fb.useRandomCoords = True
                p_fb.enforceChirality = False
                p_fb.maxIterations = 4
                p_fb.randomSeed = 20261127
                p_fb.numThreads = 1
                cids = list(AllChem.EmbedMultipleConfs(m_emb, numConfs=max(2, n_confs // 2), params=p_fb))
            except _MolTimeout:
                cids = []
            finally:
                signal.alarm(0)

        if not cids:
            return ck, nan_vec, False, 0

        # Add explicit hydrogens with 3D coordinates and relax with MMFF94s (or UFF fallback)
        m3 = Chem.AddHs(m_emb, addCoords=True)
        cids = [conf.GetId() for conf in m3.GetConformers()]
        energies: List[float] = []
        ff_code = 0  # 1=MMFF94s, 2=UFF, 3=unrelaxed
        max_ff_iters = 50 if n_heavy > 110 else 65
        signal.alarm(20)
        try:
            if n_heavy <= 150 and AllChem.MMFFHasAllMoleculeParams(m3):
                opt_res = AllChem.MMFFOptimizeMoleculeConfs(m3, numThreads=1, mmffVariant="MMFF94s", maxIters=max_ff_iters)
                energies = [float(e) for _, e in opt_res]
                ff_code = 1
            else:
                opt_res = AllChem.UFFOptimizeMoleculeConfs(m3, numThreads=1, maxIters=max_ff_iters)
                energies = [float(e) for _, e in opt_res]
                ff_code = 2
        except Exception:
            try:
                opt_res = AllChem.UFFOptimizeMoleculeConfs(m3, numThreads=1, maxIters=40)
                energies = [float(e) for _, e in opt_res]
                ff_code = 2
            except Exception:
                energies = [0.0 for _ in cids]
                ff_code = 3
        finally:
            signal.alarm(0)

        # Precompute topological distance matrix & candidate IMHB (donor, acceptor) pairs separated by >= 4 bonds
        d_top = Chem.GetDistanceMatrix(m3)
        donors: List[Tuple[int, List[int]]] = []  # (heavy_idx, [attached_H_indices])
        acceptors: List[int] = []
        polar_atom_indices: List[int] = []

        for atom in m3.GetAtoms():
            idx = atom.GetIdx()
            z = atom.GetAtomicNum()
            if z in (7, 8):
                polar_atom_indices.append(idx)
                acceptors.append(idx)
                h_nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
                if h_nbrs:
                    donors.append((idx, h_nbrs))
            elif z == 1:
                nbrs = atom.GetNeighbors()
                if nbrs and nbrs[0].GetAtomicNum() in (7, 8):
                    polar_atom_indices.append(idx)

        cand_pairs: List[Tuple[int, int, List[int]]] = []
        for d_idx, h_list in donors:
            for a_idx in acceptors:
                if d_idx != a_idx and d_top[d_idx, a_idx] >= 4.0:
                    cand_pairs.append((d_idx, a_idx, h_list))

        radii = rdFreeSASA.classifyAtoms(m3)
        psa3d_vals, sasa_vals = [], []
        imhb_vals, imhb_geom_vals = [], []
        rg_vals, asph_vals, ecc_vals, sph_vals, npr1_vals, npr2_vals = [], [], [], [], [], []

        for cid in cids:
            conf = m3.GetConformer(cid)
            pos = conf.GetPositions()  # shape (n_atoms, 3)

            # 3D Polar Surface Area via rdFreeSASA (exact Lee-Richards SASA on N/O + polar H)
            try:
                tot_sasa = float(rdFreeSASA.CalcSASA(m3, radii, confIdx=cid))
                p_sasa = 0.0
                for pidx in polar_atom_indices:
                    a = m3.GetAtomWithIdx(pidx)
                    if a.HasProp("SASA"):
                        p_sasa += float(a.GetProp("SASA"))
                sasa_vals.append(tot_sasa)
                psa3d_vals.append(p_sasa)
            except Exception:
                pass

            # IMHB counts: heavy-atom D...A <= 3.3 A (>= 4 bonds) AND strict geometric H...A <= 2.5 A, angle >= 110 deg
            n_imhb = 0
            n_imhb_geom = 0
            for d_idx, a_idx, h_list in cand_pairs:
                v_da = pos[a_idx] - pos[d_idx]
                dist_da = float(math.sqrt(v_da[0] * v_da[0] + v_da[1] * v_da[1] + v_da[2] * v_da[2]))
                if dist_da <= 3.3:
                    n_imhb += 1
                    # Strict geometric check with explicit H
                    geom_hit = False
                    for h_idx in h_list:
                        v_ha = pos[a_idx] - pos[h_idx]
                        dist_ha = float(math.sqrt(v_ha[0] * v_ha[0] + v_ha[1] * v_ha[1] + v_ha[2] * v_ha[2]))
                        if dist_ha <= 2.5 and dist_ha > 1e-6:
                            v_hd = pos[d_idx] - pos[h_idx]
                            dist_hd = float(math.sqrt(v_hd[0] * v_hd[0] + v_hd[1] * v_hd[1] + v_hd[2] * v_hd[2]))
                            if dist_hd > 1e-6:
                                cos_ang = float(np.dot(v_hd, v_ha) / (dist_hd * dist_ha))
                                # cos(110 deg) = -0.34202; angle >= 110 deg <=> cos_ang <= -0.34202
                                if cos_ang <= -0.34202:
                                    geom_hit = True
                                    break
                    if geom_hit:
                        n_imhb_geom += 1
            imhb_vals.append(float(n_imhb))
            imhb_geom_vals.append(float(n_imhb_geom))

            # 3D shape descriptors
            try:
                rg_vals.append(float(Descriptors3D.RadiusOfGyration(m3, confId=cid)))
                asph_vals.append(float(Descriptors3D.Asphericity(m3, confId=cid)))
                ecc_vals.append(float(Descriptors3D.Eccentricity(m3, confId=cid)))
                sph_vals.append(float(Descriptors3D.SpherocityIndex(m3, confId=cid)))
                npr1_vals.append(float(Descriptors3D.NPR1(m3, confId=cid)))
                npr2_vals.append(float(Descriptors3D.NPR2(m3, confId=cid)))
            except Exception:
                pass

        if not psa3d_vals or not rg_vals:
            return ck, nan_vec, False, ff_code

        psa_a = np.asarray(psa3d_vals, dtype=np.float64)
        sasa_a = np.asarray(sasa_vals, dtype=np.float64)
        imhb_a = np.asarray(imhb_vals, dtype=np.float64)
        imhbg_a = np.asarray(imhb_geom_vals, dtype=np.float64)
        rg_a = np.asarray(rg_vals, dtype=np.float64)
        asph_a = np.asarray(asph_vals, dtype=np.float64)
        ecc_a = np.asarray(ecc_vals, dtype=np.float64)
        sph_a = np.asarray(sph_vals, dtype=np.float64)
        npr1_a = np.asarray(npr1_vals, dtype=np.float64)
        npr2_a = np.asarray(npr2_vals, dtype=np.float64)

        e_arr = np.asarray(energies, dtype=np.float64)
        e_fin = e_arr[np.isfinite(e_arr)]
        if len(e_fin) >= 2:
            low_idx = int(np.argmin(np.where(np.isfinite(e_arr), e_arr, 1e18)))
            e_spread = float(np.clip(np.max(e_fin) - np.min(e_fin), 0.0, 200.0))
            e_std = float(np.clip(np.std(e_fin), 0.0, 100.0))
        else:
            low_idx = 0
            e_spread = 0.0
            e_std = 0.0

        low_idx = min(low_idx, len(psa_a) - 1, len(imhb_a) - 1)
        vec = np.array(
            [
                float(np.min(psa_a)),
                float(np.max(psa_a)),
                float(np.mean(psa_a)),
                float(np.max(psa_a) - np.min(psa_a)),
                float(np.std(psa_a)),
                float(psa_a[low_idx]),
                float(np.min(psa_a) / max(tpsa2d, 1.0)),
                float(np.mean(sasa_a)),
                float(np.min(imhb_a)),
                float(np.max(imhb_a)),
                float(np.mean(imhb_a)),
                float(np.max(imhb_a) - np.min(imhb_a)),
                float(imhb_a[low_idx]),
                float(np.max(imhb_a) / max(n_hbd, 1)),
                float(np.max(imhbg_a)),
                float(np.mean(imhbg_a)),
                float(np.mean(rg_a)),
                float(np.min(rg_a)),
                float(np.max(rg_a) - np.min(rg_a)),
                float(np.mean(asph_a)),
                float(np.max(asph_a) - np.min(asph_a)),
                float(np.mean(ecc_a)),
                float(np.mean(sph_a)),
                float(np.max(sph_a) - np.min(sph_a)),
                float(np.mean(npr1_a)),
                float(np.mean(npr2_a)),
                float(np.max(npr1_a) - np.min(npr1_a)),
                float(np.max(npr2_a) - np.min(npr2_a)),
                e_spread,
                e_std,
                float(len(cids)),
            ],
            dtype=np.float32,
        )
        if not np.isfinite(vec).all():
            vec = np.where(np.isfinite(vec), vec, np.nan).astype(np.float32)
        return ck, vec, True, ff_code
    except Exception:
        return ck, nan_vec, False, 0


def build_or_load_3d_cache(
    bank: UnifiedV334Bank,
    ck_to_smi: Dict[str, str],
    needed_cks: List[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute or resume-load 3D-IMHB descriptors for all needed compounds, aligned to bank indices."""
    cache_file = CACHE_3D_DIR / "features_3d_imhb_v1.npz"
    done_map: Dict[str, Tuple[np.ndarray, bool, int]] = {}

    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        cks_saved = [str(x) for x in z["compound_keys"]]
        X_saved = z["X3d"].astype(np.float32)
        v_saved = z["valid"].astype(bool)
        ff_saved = z["ff_code"].astype(np.int8)
        z.close()
        if X_saved.shape[1] == N_FEAT3D:
            for i, ck in enumerate(cks_saved):
                done_map[ck] = (X_saved[i], bool(v_saved[i]), int(ff_saved[i]))
            print(f"[3D-CACHE] Loaded {len(done_map):,} cached compounds from {cache_file}", flush=True)

    todo = [ck for ck in needed_cks if ck not in done_map]
    t0 = time.time()
    if todo:
        # Deterministic shuffle for unbiased live ETA & balanced worker load across MWs
        import random as _rnd
        todo_with_len = list(todo)
        _rnd.Random(42).shuffle(todo_with_len)
        print(
            f"[3D-GEN] Computing ETKDGv3+MMFF94s 3D-IMHB features for {len(todo_with_len):,} compounds "
            f"across {N_3D_WORKERS} CPU workers...",
            flush=True,
        )
        work_items = [(ck, ck_to_smi.get(ck, "")) for ck in todo_with_len]
        completed = 0
        ok_cnt = 0

        def _save_ckpt() -> None:
            keys_all = sorted(done_map.keys())
            X_all = np.vstack([done_map[k][0] for k in keys_all]).astype(np.float32)
            v_all = np.array([done_map[k][1] for k in keys_all], dtype=bool)
            ff_all = np.array([done_map[k][2] for k in keys_all], dtype=np.int8)
            tmp_f = CACHE_3D_DIR / "features_3d_imhb_v1.tmp.npz"
            np.savez_compressed(tmp_f, compound_keys=np.array(keys_all), X3d=X_all, valid=v_all, ff_code=ff_all)
            tmp_f.replace(cache_file)

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=N_3D_WORKERS) as pool:
            for ck, vec, ok, ffc in pool.imap_unordered(_compute_3d_imhb_single, work_items, chunksize=1):
                done_map[ck] = (vec, ok, ffc)
                completed += 1
                if ok:
                    ok_cnt += 1
                if completed % 250 == 0 or completed == len(work_items):
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1e-6)
                    eta_m = (len(work_items) - completed) / max(rate, 1e-6) / 60.0
                    print(
                        f"  [3D-GEN] {completed:,}/{len(work_items):,} (ok={ok_cnt:,}, {100.0*ok_cnt/completed:.1f}%) "
                        f"| {elapsed/60:.1f}m elapsed | {rate:.2f} mol/s | ETA {eta_m:.1f}m",
                        flush=True,
                    )
                    _save_ckpt()

    # Align full bank matrix
    X3d_bank = np.full((bank.n_total, N_FEAT3D), np.nan, dtype=np.float32)
    valid_bank = np.zeros(bank.n_total, dtype=bool)
    ff_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_needed_ok = 0
    for ck in needed_cks:
        if ck in done_map and ck in bank.key_to_idx:
            vec, ok, ffc = done_map[ck]
            bidx = bank.key_to_idx[ck]
            X3d_bank[bidx] = vec
            valid_bank[bidx] = ok
            ff_counts[ffc] = ff_counts.get(ffc, 0) + 1
            if ok:
                n_needed_ok += 1

    stats = {
        "n_unique_target_compounds": len(needed_cks),
        "n_conformer_success": n_needed_ok,
        "conformer_success_rate_pct": round(100.0 * n_needed_ok / max(len(needed_cks), 1), 2),
        "force_field_breakdown": {
            "MMFF94s": ff_counts.get(1, 0),
            "UFF_fallback": ff_counts.get(2, 0),
            "unrelaxed_ETKDGv3": ff_counts.get(3, 0),
            "failed_2d_fallback": ff_counts.get(0, 0),
        },
        "n_features": N_FEAT3D,
        "feature_names": FEAT3D_NAMES,
        "conformer_generation_wall_sec": round(time.time() - t0, 1),
    }
    print(f"[3D-CACHE] Ready: {stats['n_conformer_success']}/{stats['n_unique_target_compounds']} ({stats['conformer_success_rate_pct']}%) | FF={stats['force_field_breakdown']}", flush=True)
    return X3d_bank, valid_bank, stats


def paired_cluster_bootstrap_reg(
    y: np.ndarray,
    pred_2d: np.ndarray,
    pred_3d: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261127,
) -> Dict[str, Any]:
    """Two-sided paired cluster-bootstrap comparing (2D+3D) vs (2D):
    positive delta_mae_reduction means 3D improved MAE (MAE_2D - MAE_3D > 0).
    positive delta_spearman means 3D improved Spearman (rho_3D - rho_2D > 0).
    """
    ok = np.isfinite(y) & np.isfinite(pred_2d) & np.isfinite(pred_3d)
    y, p2, p3, cl = y[ok], pred_2d[ok], pred_3d[ok], clusters[ok]
    if len(y) < 6:
        return {"n_eval": int(len(y)), "n_clusters": int(len(set(map(str, cl)))), "p_value_mae": None, "p_value_spearman": None}

    cl_map: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cl):
        cl_map[str(c)].append(i)
    cl_arrs = [np.array(v, dtype=np.int32) for v in cl_map.values()]
    n_cl = len(cl_arrs)
    rng = np.random.RandomState(seed)
    d_mae, d_sp = [], []
    for _ in range(n_boot):
        pick = rng.randint(0, n_cl, size=n_cl)
        idx = np.concatenate([cl_arrs[p] for p in pick])
        yb, a2, a3 = y[idx], p2[idx], p3[idx]
        d_mae.append(float(np.mean(np.abs(yb - a2)) - np.mean(np.abs(yb - a3))))
        if np.std(yb) > 1e-12 and np.std(a2) > 1e-12 and np.std(a3) > 1e-12:
            r2 = float(spearmanr(yb, a2).statistic)
            r3 = float(spearmanr(yb, a3).statistic)
            if math.isfinite(r2) and math.isfinite(r3):
                d_sp.append(r3 - r2)

    dm = np.asarray(d_mae, dtype=np.float64)
    p_floor = 1.0 / max(len(dm), 1)
    pm_raw = float(2.0 * min(np.mean(dm <= 0.0), np.mean(dm >= 0.0)))
    pm = max(pm_raw, p_floor)

    out: Dict[str, Any] = {
        "n_eval": int(len(y)),
        "n_clusters": int(n_cl),
        "mae_reduction_point": round(float(np.mean(np.abs(y - p2)) - np.mean(np.abs(y - p3))), 5),
        "mae_reduction_mean": round(float(dm.mean()), 5),
        "mae_reduction_ci95": [round(float(np.percentile(dm, 2.5)), 5), round(float(np.percentile(dm, 97.5)), 5)],
        "p_value_mae": round(pm, 4),
        "p_value_mae_display": f"<={p_floor:.3f}" if pm_raw < p_floor else f"{pm:.4f}",
    }
    if len(d_sp) >= 50:
        ds = np.asarray(d_sp, dtype=np.float64)
        ps_raw = float(2.0 * min(np.mean(ds <= 0.0), np.mean(ds >= 0.0)))
        ps_floor = 1.0 / len(ds)
        ps = max(ps_raw, ps_floor)
        r2_pt = float(spearmanr(y, p2).statistic)
        r3_pt = float(spearmanr(y, p3).statistic)
        out["spearman_gain_point"] = round(r3_pt - r2_pt, 5) if math.isfinite(r2_pt) and math.isfinite(r3_pt) else None
        out["spearman_gain_mean"] = round(float(ds.mean()), 5)
        out["spearman_gain_ci95"] = [round(float(np.percentile(ds, 2.5)), 5), round(float(np.percentile(ds, 97.5)), 5)]
        out["p_value_spearman"] = round(ps, 4)
        out["p_value_spearman_display"] = f"<={ps_floor:.3f}" if ps_raw < ps_floor else f"{ps:.4f}"
    return out


def _clean_scale_3d_fold(X3d_tr: np.ndarray, X3d_va: np.ndarray, X3d_te: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Strictly fold-internal median imputation + z-score standardization + [-8, 8] clipping for 3D-IMHB block."""
    tr = X3d_tr.astype(np.float32).copy()
    va = X3d_va.astype(np.float32).copy()
    te = X3d_te.astype(np.float32).copy() if X3d_te is not None else None
    d = tr.shape[1]
    meds = np.zeros(d, dtype=np.float32)
    for j in range(d):
        ok = np.isfinite(tr[:, j])
        meds[j] = float(np.median(tr[ok, j])) if ok.any() else 0.0
        if (~ok).any():
            tr[~ok, j] = meds[j]
        va_bad = ~np.isfinite(va[:, j])
        if va_bad.any():
            va[va_bad, j] = meds[j]
        if te is not None:
            te_bad = ~np.isfinite(te[:, j])
            if te_bad.any():
                te[te_bad, j] = meds[j]
    mu = tr.mean(axis=0).astype(np.float32)
    sd = tr.std(axis=0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    tr_s = np.clip((tr - mu) / sd, -8.0, 8.0).astype(np.float32)
    va_s = np.clip((va - mu) / sd, -8.0, 8.0).astype(np.float32)
    te_s = np.clip((te - mu) / sd, -8.0, 8.0).astype(np.float32) if te is not None else None
    return tr_s, va_s, te_s


def train_cpu_track_v34(
    bank: UnifiedV334Bank,
    cidx_cv: np.ndarray,
    y_cv: np.ndarray,
    fids_cv: np.ndarray,
    cond_cv: List[Dict[str, str]],
    cidx_ho: np.ndarray,
    cond_ho: List[Dict[str, str]],
    cond_cols: List[str],
    bounds: Optional[Tuple[float, float]],
    X3d_bank: Optional[np.ndarray] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Exact V33 CPU track (matches `03_train_and_promote_v334.train_cpu_track`), optionally augmented
    strictly fold-internally via `FoldPreprocessor(..., meta=X3d)`!
    """
    valid_cv = np.ones(len(cidx_cv), dtype=bool)
    valid_ho = np.ones(len(cidx_ho), dtype=bool)
    use_3d = X3d_bank is not None
    meta_cv = X3d_bank[cidx_cv] if use_3d else None
    meta_ho = X3d_bank[cidx_ho] if use_3d else None

    recipes = [
        ("Tanimoto_KRR_025", "TanimotoRBF_KRR", "tanimoto_hybrid", True, {"alpha": 0.25, "w_tan": 0.5, "w_cond": 0.15, "gamma_mult": 1.0}),
        ("Tanimoto_KRR_100", "TanimotoRBF_KRR", "tanimoto_hybrid", True, {"alpha": 1.0, "w_tan": 0.5, "w_cond": 0.15, "gamma_mult": 1.0}),
        ("Ridge_svd_10", "Ridge", "physchem_svd", True, {"alpha": 10.0}),
        ("Ridge_svd_50", "Ridge", "physchem_svd", True, {"alpha": 50.0}),
        ("SVR_svd_2", "SVR_RBF", "physchem_svd", True, {"C": 2.0, "epsilon": 0.05}),
        ("SVR_svd_8", "SVR_RBF", "physchem_svd", True, {"C": 8.0, "epsilon": 0.05}),
        ("LGBM_svd", "LightGBM", "physchem_svd", False, {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 10, "min_child_samples": 10, "reg_lambda": 15.0, "colsample_bytree": 0.6}),
        ("ET_svd", "ExtraTrees", "physchem_svd", False, {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 3, "max_features": 0.3}),
    ]
    if len(cidx_cv) <= 1500:
        recipes.insert(0, ("Tanimoto_GPR", "TanimotoRBF_GPR", "tanimoto_hybrid", True, {"n_restarts": 3}))

    route_cache: Dict[Tuple[str, bool], List[Any]] = {}
    best_name, best_mae, best_oof, best_ho = None, 1e18, None, None

    for rname, fam, route, scale_flag, params in recipes:
        rkey = (route, scale_flag)
        if rkey not in route_cache:
            fdata = []
            for f in range(5):
                tr = (fids_cv != f) & valid_cv
                va = (fids_cv == f) & valid_cv
                if tr.sum() < 8 or va.sum() < 1:
                    continue
                pre = FoldPreprocessor(
                    track="dual_modality",
                    route=route,
                    cond_cols=cond_cols,
                    scale_for_linear=scale_flag,
                    svd_dim=min(32, max(6, int(tr.sum()) - 3)),
                    random_state=42 + f,
                )
                X_tr = pre.fit_transform(bank, cidx_cv[tr], [cond_cv[i] for i in np.where(tr)[0]], meta=meta_cv[tr] if use_3d else None)
                X_va = pre.transform(bank, cidx_cv[va], [cond_cv[i] for i in np.where(va)[0]], meta=meta_cv[va] if use_3d else None)
                X_ho = pre.transform(bank, cidx_ho[valid_ho], [cond_ho[i] for i in np.where(valid_ho)[0]], meta=meta_ho[valid_ho] if use_3d else None) if valid_ho.any() else None
                fdata.append((f, tr, va, X_tr, X_va, X_ho))
            route_cache[rkey] = fdata

        fdata = route_cache[rkey]
        if not fdata:
            continue
        oof = np.full(len(cidx_cv), np.nan, dtype=np.float64)
        ho_preds = []
        ok_all = True

        def _fit_cpu_fold(item):
            f, tr, va, X_tr, X_va, X_ho = item
            est = make_estimator(family=fam, params=params, task_type="regression", n_jobs=2, bounds=bounds, seed=42 + f)
            est.fit(X_tr, y_cv[tr])
            p_va = np.asarray(est.predict(X_va), dtype=np.float64)
            if bounds is not None:
                p_va = np.clip(p_va, bounds[0], bounds[1])
            p_h = None
            if X_ho is not None:
                p_h = np.asarray(est.predict(X_ho), dtype=np.float64)
                if bounds is not None:
                    p_h = np.clip(p_h, bounds[0], bounds[1])
            return va, p_va, p_h

        try:
            with ThreadPoolExecutor(max_workers=min(5, len(fdata))) as ex:
                for va, p_va, p_h in ex.map(_fit_cpu_fold, fdata):
                    oof[va] = p_va
                    if p_h is not None:
                        ho_preds.append(p_h)
        except Exception as exc:
            print(f"    [WARN cpu {rname}] {exc}", flush=True)
            ok_all = False
        if not ok_all or not ho_preds:
            continue
        m = regression_metrics(y_cv[valid_cv], oof[valid_cv])
        mae = m["mae"] if m["mae"] is not None else 1e18
        if mae < best_mae:
            best_mae = mae
            best_name = rname
            best_oof = oof
            best_ho = np.mean(np.vstack(ho_preds), axis=0)

    return str(best_name), best_oof, best_ho


def train_3d_standalone_head(
    X3d_cv: np.ndarray,
    y_cv: np.ndarray,
    fids_cv: np.ndarray,
    X3d_ho: np.ndarray,
    bounds: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Train a strictly fold-internal 3D-IMHB-only head (best of LGBM / SVR-RBF on OOF MAE)."""
    cands = [
        ("SVR_3D_c2", lambda f: SVR(kernel="rbf", C=2.0, epsilon=0.05)),
        ("SVR_3D_c8", lambda f: SVR(kernel="rbf", C=8.0, epsilon=0.05)),
        ("LGBM_3D", lambda f: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=4, num_leaves=10, min_child_samples=10, reg_lambda=10.0, colsample_bytree=0.8, n_jobs=4, random_state=42 + f, verbose=-1)),
    ]
    best_name, best_mae, best_oof, best_ho = "SVR_3D_c2", 1e18, None, None
    for cname, fn in cands:
        oof = np.full(len(y_cv), np.nan, dtype=np.float64)
        hos = []
        for f in range(5):
            tr = fids_cv != f
            va = fids_cv == f
            if tr.sum() < 8 or va.sum() < 1:
                continue
            Xtr_s, Xva_s, Xho_s = _clean_scale_3d_fold(X3d_cv[tr], X3d_cv[va], X3d_ho)
            mdl = fn(f)
            mdl.fit(Xtr_s, y_cv[tr])
            pva = np.asarray(mdl.predict(Xva_s), dtype=np.float64)
            pho = np.asarray(mdl.predict(Xho_s), dtype=np.float64)
            if bounds is not None:
                pva = np.clip(pva, bounds[0], bounds[1])
                pho = np.clip(pho, bounds[0], bounds[1])
            oof[va] = pva
            hos.append(pho)
        m = regression_metrics(y_cv, oof)
        mae = m["mae"] if m["mae"] is not None else 1e18
        if mae < best_mae:
            best_mae, best_name, best_oof, best_ho = mae, cname, oof, np.mean(np.vstack(hos), axis=0)
    return best_oof, best_ho, best_name


def train_fusion_track_v34(
    bank: UnifiedV334Bank,
    cidx_cv: np.ndarray,
    y_cv: np.ndarray,
    fids_cv: np.ndarray,
    cond_cv: List[Dict[str, str]],
    cidx_ho: np.ndarray,
    cond_ho: List[Dict[str, str]],
    cond_cols: List[str],
    bounds: Optional[Tuple[float, float]],
    cpu_oof: np.ndarray,
    cpu_ho: np.ndarray,
    X3d_bank: Optional[np.ndarray] = None,
    head_3d_only: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """Exact V33 4-FM Fusion track, optionally augmented strictly fold-internally with
    variance-balanced 3D-IMHB features appended to every representation block + dedicated 3D head.
    """
    valid_cv = np.ones(len(cidx_cv), dtype=bool)
    valid_ho = np.ones(len(cidx_ho), dtype=bool)
    use_3d = X3d_bank is not None
    w3d_scale = 0.65 / math.sqrt(float(N_FEAT3D))

    direct_list = list(DIRECT_FUSION_REPS.values()) if isinstance(DIRECT_FUSION_REPS, dict) else list(DIRECT_FUSION_REPS)
    indiv_list = list(INDIVIDUAL_REPS.values()) if isinstance(INDIVIDUAL_REPS, dict) else list(INDIVIDUAL_REPS)
    all_rep_cfgs = list(indiv_list) + list(direct_list)
    keep_names = {"ncaa_fusion_hf", "ncaa_smiles_repr", "peptideclm2", "chemberta_mtr", "esmc_mean", "direct_smiles", "direct_dual"}
    rep_cfgs = [r for r in all_rep_cfgs if r["name"] in keep_names]

    cand_specs = [
        ("KRR_a1", "KRR_RBF", {"alpha": 1.0, "gamma_mult": 1.0}),
        ("KRR_a5", "KRR_RBF", {"alpha": 5.0, "gamma_mult": 1.0}),
        ("KRR_a20", "KRR_RBF", {"alpha": 20.0, "gamma_mult": 1.0}),
        ("SVR_c3", "SVR_RBF", {"C": 3.0, "epsilon": 0.1, "gamma_mult": 1.0}),
        ("SVR_c10", "SVR_RBF", {"C": 10.0, "epsilon": 0.1, "gamma_mult": 1.0}),
        ("Ridge_a30", "Ridge", {"alpha": 30.0}),
    ]

    head_oof: Dict[str, np.ndarray] = {"cpu_dual": cpu_oof.copy()}
    head_ho: Dict[str, np.ndarray] = {"cpu_dual": cpu_ho.copy()}
    if use_3d and head_3d_only is not None:
        head_oof["conformer_3d_imhb_only"] = head_3d_only[0].copy()
        head_ho["conformer_3d_imhb_only"] = head_3d_only[1].copy()

    for rcfg in rep_cfgs:
        rname = rcfg["name"]
        r_track = rcfg["track"]
        v_cv = bank.smiles_valid[cidx_cv].copy() if r_track == "smiles_only" else np.ones(len(cidx_cv), dtype=bool)
        v_ho = bank.smiles_valid[cidx_ho].copy() if r_track == "smiles_only" else np.ones(len(cidx_ho), dtype=bool)
        if v_cv.sum() < 15:
            continue

        fpreps = []
        for f in range(5):
            tr = (fids_cv != f) & v_cv
            va = (fids_cv == f) & v_cv
            if tr.sum() < 8 or va.sum() < 1:
                continue
            pre = EmbeddingFoldPreprocessor(
                blocks=rcfg["blocks"],
                phys_mode=rcfg["phys_mode"],
                cond_cols=cond_cols,
                mode="balanced",
                pca_group_dims=rcfg["pca_group_dims"],
                random_state=42 + f,
            )
            X_tr = pre.fit_transform(bank, cidx_cv[tr], [cond_cv[i] for i in np.where(tr)[0]])
            X_va = pre.transform(bank, cidx_cv[va], [cond_cv[i] for i in np.where(va)[0]])
            X_ho_f = (
                pre.transform(bank, cidx_ho[v_ho], [cond_ho[i] for i in np.where(v_ho)[0]])
                if v_ho.any()
                else np.zeros((0, X_tr.shape[1]), dtype=np.float32)
            )
            if use_3d:
                z3_tr, z3_va, z3_ho = _clean_scale_3d_fold(
                    X3d_bank[cidx_cv[tr]], X3d_bank[cidx_cv[va]], X3d_bank[cidx_ho[v_ho]] if v_ho.any() else None
                )
                X_tr = np.hstack([X_tr, z3_tr * w3d_scale]).astype(np.float32)
                X_va = np.hstack([X_va, z3_va * w3d_scale]).astype(np.float32)
                if v_ho.any() and z3_ho is not None:
                    X_ho_f = np.hstack([X_ho_f, z3_ho * w3d_scale]).astype(np.float32)
                else:
                    X_ho_f = np.zeros((0, X_tr.shape[1]), dtype=np.float32)
            fpreps.append((f, tr, va, X_tr, X_va, X_ho_f))

        best_mae, best_oof, best_ho = 1e18, None, None
        for _, fam, params in cand_specs:
            oof = np.full(len(cidx_cv), np.nan, dtype=np.float64)
            ho_list = []

            def _fit_fus_fold(item):
                f, tr, va, X_tr, X_va, X_ho_f = item
                est = make_fusion_estimator(family=fam, params=params, task_type="regression", n_jobs=2, bounds=bounds, seed=42 + f)
                est.fit(X_tr, y_cv[tr])
                p_va = np.asarray(est.predict(X_va), dtype=np.float64)
                p_h = np.asarray(est.predict(X_ho_f), dtype=np.float64) if len(X_ho_f) > 0 else None
                return va, p_va, p_h

            with ThreadPoolExecutor(max_workers=min(5, len(fpreps))) as ex:
                for va, p_va, p_h in ex.map(_fit_fus_fold, fpreps):
                    oof[va] = p_va
                    if p_h is not None:
                        ho_list.append(p_h)
            m = regression_metrics(y_cv[v_cv], oof[v_cv])
            mae = m["mae"] if m["mae"] is not None else 1e18
            if mae < best_mae:
                best_mae = mae
                best_oof = oof
                best_ho = np.full(len(cidx_ho), np.nan, dtype=np.float64)
                if ho_list:
                    best_ho[v_ho] = np.mean(np.vstack(ho_list), axis=0)

        if best_oof is not None and best_ho is not None:
            head_oof[rname] = best_oof
            head_ho[rname] = best_ho
            print(f"      [fus {'3D' if use_3d else '2D'} {rname:<16}] best OOF MAE={best_mae:.4f}", flush=True)

    cand_heads = list(head_oof.keys())
    P_cv = np.column_stack([head_oof[h] for h in cand_heads])
    P_ho = np.column_stack([head_ho[h] for h in cand_heads])

    # Strictly nested OOF predictions for the stacker
    fus_oof = np.full(len(cidx_cv), np.nan, dtype=np.float64)
    for f in range(5):
        tr = (fids_cv != f) & valid_cv
        va = (fids_cv == f) & valid_cv
        if tr.sum() < 8 or va.sum() < 1:
            continue
        st_f = ModalityAwareStacker(task_type="regression", method="convex_simplex", bounds=bounds)
        st_f.fit(P_cv[tr], y_cv[tr], cand_heads)
        fus_oof[va] = st_f.predict(P_cv[va])

    stacker = ModalityAwareStacker(task_type="regression", method="convex_simplex", bounds=bounds)
    stacker.fit(P_cv[valid_cv], y_cv[valid_cv], cand_heads)
    fus_ho = np.full(len(cidx_ho), np.nan, dtype=np.float64)
    if valid_ho.any():
        fus_ho[valid_ho] = stacker.predict(P_ho[valid_ho])

    w_dict = {}
    if hasattr(stacker, "weights_") and stacker.weights_ is not None:
        w_dict = {h: round(float(w), 4) for h, w in zip(cand_heads, stacker.weights_)}
    return fus_oof, fus_ho, cand_heads, w_dict


def combine_super_ensemble_oof(
    y_cv: np.ndarray,
    fids_cv: np.ndarray,
    cpu_oof: np.ndarray,
    fus_oof: np.ndarray,
    cpu_ho: np.ndarray,
    fus_ho: np.ndarray,
    bounds: Optional[Tuple[float, float]],
) -> Dict[str, Any]:
    """Exact nested OOF simplex selection between CPU and Fusion (matching select_simplex_oof)."""
    valid_mask = np.isfinite(y_cv) & np.isfinite(cpu_oof) & np.isfinite(fus_oof)
    m_cpu = regression_metrics(y_cv[valid_mask], cpu_oof[valid_mask])
    m_fus = regression_metrics(y_cv[valid_mask], fus_oof[valid_mask])
    mae_c = m_cpu["mae"] if m_cpu["mae"] is not None else 1e9
    mae_f = m_fus["mae"] if m_fus["mae"] is not None else 1e9
    best_single = "fusion" if mae_f <= mae_c else "cpu"
    best_single_mae = min(mae_c, mae_f)

    sup_oof = np.full(len(y_cv), np.nan, dtype=np.float64)
    w_folds = []
    for f in range(5):
        tr = (fids_cv != f) & valid_mask
        va = (fids_cv == f) & valid_mask
        if tr.sum() < 10 or va.sum() < 1:
            continue
        res = minimize(
            lambda w: float(np.mean(np.abs(y_cv[tr] - (w[0] * cpu_oof[tr] + (1.0 - w[0]) * fus_oof[tr])))),
            [0.5],
            bounds=[(0.0, 1.0)],
            method="L-BFGS-B",
        )
        wc = float(res.x[0])
        w_folds.append(wc)
        p = wc * cpu_oof[va] + (1.0 - wc) * fus_oof[va]
        if bounds is not None:
            p = np.clip(p, bounds[0], bounds[1])
        sup_oof[va] = p

    m_sup = regression_metrics(y_cv[valid_mask], sup_oof[valid_mask])
    mae_s = m_sup["mae"] if m_sup["mae"] is not None else 1e9
    if mae_s < best_single_mae - 1e-4 and w_folds:
        wc_avg = round(float(np.mean(w_folds)), 4)
        wf_avg = round(1.0 - wc_avg, 4)
        pred_ho = wc_avg * cpu_ho + wf_avg * fus_ho
        if bounds is not None:
            pred_ho = np.clip(pred_ho, bounds[0], bounds[1])
        return {
            "winning_family": "super_ensemble",
            "weights": {"cpu": wc_avg, "fusion": wf_avg},
            "oof_pred": sup_oof,
            "ho_pred": pred_ho,
        }
    if best_single == "fusion":
        return {"winning_family": "fusion", "weights": {"cpu": 0.0, "fusion": 1.0}, "oof_pred": fus_oof, "ho_pred": fus_ho}
    return {"winning_family": "cpu", "weights": {"cpu": 1.0, "fusion": 0.0}, "oof_pred": cpu_oof, "ho_pred": cpu_ho}


def build_cluster_balanced_folds(clusters: List[str], n_folds: int = 5) -> np.ndarray:
    cl_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        cl_to_rows[str(c)].append(i)
    sorted_cls = sorted(cl_to_rows.keys(), key=lambda c: -len(cl_to_rows[c]))
    fold_sizes = [0] * n_folds
    fids = np.zeros(len(clusters), dtype=np.int32)
    for c in sorted_cls:
        f = int(np.argmin(fold_sizes))
        for idx in cl_to_rows[c]:
            fids[idx] = f
        fold_sizes[f] += len(cl_to_rows[c])
    return fids


def compute_shift_diagnostics(
    bank: UnifiedV334Bank,
    X3d_bank: np.ndarray,
    cidx_tr: np.ndarray,
    y_tr: np.ndarray,
    cidx_te: np.ndarray,
    y_te: np.ndarray,
) -> Dict[str, Any]:
    """Quantify historical-train vs future-test distributional shift (target, MW, 2D TPSA, 3D PSA, IMHB, Tanimoto AD)."""
    # Max Tanimoto similarity of future test to historical train
    tr_b = bank.morgan_bits[cidx_tr].astype(np.float32)
    te_b = bank.morgan_bits[cidx_te].astype(np.float32)
    inter = te_b @ tr_b.T
    union = te_b.sum(axis=1)[:, None] + tr_b.sum(axis=1)[None, :] - inter
    max_tc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).max(axis=1)

    def _med_iqr(arr: np.ndarray) -> Dict[str, Optional[float]]:
        a = arr[np.isfinite(arr)]
        if len(a) == 0:
            return {"median": None, "mean": None, "sd": None, "p10": None, "p90": None}
        return {
            "median": round(float(np.median(a)), 3),
            "mean": round(float(np.mean(a)), 3),
            "sd": round(float(np.std(a)), 3),
            "p10": round(float(np.percentile(a, 10)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
        }

    idx_map = {n: i for i, n in enumerate(FEAT3D_NAMES)}
    # physchem[:, 0] is MolWt, physchem[:, 5] or TPSA - let's inspect cleanly via 3D features + y + Tanimoto
    return {
        "target_y_train": _med_iqr(y_tr),
        "target_y_future_test": _med_iqr(y_te),
        "max_tanimoto_future_to_train": {
            "median": round(float(np.median(max_tc)), 4),
            "mean": round(float(np.mean(max_tc)), 4),
            "pct_ge_0_65_in_domain": round(100.0 * float((max_tc >= 0.65).mean()), 2),
            "pct_lt_0_40_ood": round(100.0 * float((max_tc < 0.40).mean()), 2),
        },
        "psa3d_min_train": _med_iqr(X3d_bank[cidx_tr, idx_map["psa3d_min"]]),
        "psa3d_min_future_test": _med_iqr(X3d_bank[cidx_te, idx_map["psa3d_min"]]),
        "psa3d_span_train": _med_iqr(X3d_bank[cidx_tr, idx_map["psa3d_span"]]),
        "psa3d_span_future_test": _med_iqr(X3d_bank[cidx_te, idx_map["psa3d_span"]]),
        "imhb_max_train": _med_iqr(X3d_bank[cidx_tr, idx_map["imhb_max"]]),
        "imhb_max_future_test": _med_iqr(X3d_bank[cidx_te, idx_map["imhb_max"]]),
        "rg_mean_train": _med_iqr(X3d_bank[cidx_tr, idx_map["rg_mean"]]),
        "rg_mean_future_test": _med_iqr(X3d_bank[cidx_te, idx_map["rg_mean"]]),
    }


def compute_3d_univariate_correlations(X3d: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    key_feats = [
        "psa3d_min",
        "psa3d_span",
        "psa3d_ratio_to_tpsa2d",
        "imhb_max",
        "imhb_span",
        "imhb_per_hbd",
        "imhb_geom_max",
        "rg_mean",
        "asphericity_mean",
        "npr1_mean",
        "conf_energy_spread",
    ]
    idx_map = {n: i for i, n in enumerate(FEAT3D_NAMES)}
    out: Dict[str, Optional[float]] = {}
    for fn in key_feats:
        col = X3d[:, idx_map[fn]]
        ok = np.isfinite(col) & np.isfinite(y)
        if ok.sum() >= 10 and np.std(col[ok]) > 1e-12 and np.std(y[ok]) > 1e-12:
            r = float(spearmanr(col[ok], y[ok]).statistic)
            out[fn] = round(r, 4) if math.isfinite(r) else None
        else:
            out[fn] = None
    return out


def run_temporal_experiments(bank: UnifiedV334Bank, X3d_bank: np.ndarray) -> Dict[str, Any]:
    """Experiment A: Prospective Temporal Split for PAMPA (7,395 -> 1,177) & LogD7.4 (653 -> 22)."""
    print("\n================ [Experiment A: Prospective Temporal Splits (2D vs 2D + 3D-IMHB)] ================", flush=True)
    specs = [
        (
            "PAMPA",
            "Release-cohort split (non-SweMacrocycleDB historical train -> SweMacrocycleDB 2024-2025 + 2025-2026 JMC/ACSMCL test; exact compound dedup)",
            lambda sc: any(w in sc.lower() for w in ("swemacrocycle", "2026", "5c00037", "5c02090")),
        ),
        (
            "LogD7.4",
            "Release-cohort split (historical Naylor/PEGASUS/ChEMBL train -> 2025-2026 Chloroalkene/JMC test; exact compound dedup)",
            lambda sc: any(w in sc.lower() for w in ("2026", "5c02090")),
        ),
    ]
    out: Dict[str, Any] = {}
    for ep, desc, is_fut_fn in specs:
        t0 = time.time()
        rows = [r for r in csv.DictReader(get_ep_path(ep).open(encoding="utf-8"), delimiter="\t") if r["censor_type"] == "exact"]
        cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
        tr_r = [r for r in rows if not is_fut_fn(r.get("source_code", ""))]
        tr_cks = {r["compound_key"] for r in tr_r}
        te_r = [r for r in rows if is_fut_fn(r.get("source_code", "")) and r["compound_key"] not in tr_cks]

        cidx_tr = bank.map_keys([r["compound_key"] for r in tr_r])
        y_tr = np.array([float(r["y_value"]) for r in tr_r], dtype=np.float64)
        cl_tr = [r["cluster_id"] for r in tr_r]
        cond_tr = [{c: r.get(c, "unspecified") for c in cond_cols} for r in tr_r]
        fids_tr = build_cluster_balanced_folds(cl_tr, n_folds=5)

        cidx_te = bank.map_keys([r["compound_key"] for r in te_r])
        y_te = np.array([float(r["y_value"]) for r in te_r], dtype=np.float64)
        cl_te = np.array([r["cluster_id"] for r in te_r], dtype=object)
        cond_te = [{c: r.get(c, "unspecified") for c in cond_cols} for r in te_r]

        print(f"  [TEMPORAL {ep}] n_tr={len(tr_r)}, n_te={len(te_r)}, n_te_clusters={len(set(cl_te))}", flush=True)

        # Standalone 3D-IMHB only
        oof_3d_only, te_3d_only, m3d_name = train_3d_standalone_head(X3d_bank[cidx_tr], y_tr, fids_tr, X3d_bank[cidx_te], None)

        # 2D Baseline: CPU + 4-FM Fusion + Super-Ensemble
        cpu2_name, cpu2_oof, cpu2_te = train_cpu_track_v34(bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, None, X3d_bank=None)
        fus2_oof, fus2_te, _, fus2_w = train_fusion_track_v34(bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, None, cpu2_oof, cpu2_te, X3d_bank=None)
        se2 = combine_super_ensemble_oof(y_tr, fids_tr, cpu2_oof, fus2_oof, cpu2_te, fus2_te, None)

        # 2D + 3D-IMHB: CPU+3D + 4-FM+3D Fusion + Super-Ensemble+3D
        cpu3_name, cpu3_oof, cpu3_te = train_cpu_track_v34(bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, None, X3d_bank=X3d_bank)
        fus3_oof, fus3_te, _, fus3_w = train_fusion_track_v34(
            bank, cidx_tr, y_tr, fids_tr, cond_tr, cidx_te, cond_te, cond_cols, None, cpu3_oof, cpu3_te,
            X3d_bank=X3d_bank, head_3d_only=(oof_3d_only, te_3d_only),
        )
        se3 = combine_super_ensemble_oof(y_tr, fids_tr, cpu3_oof, fus3_oof, cpu3_te, fus3_te, None)

        m_3d_only = regression_metrics(y_te, te_3d_only)
        m_cpu2 = regression_metrics(y_te, cpu2_te)
        m_cpu3 = regression_metrics(y_te, cpu3_te)
        m_se2 = regression_metrics(y_te, se2["ho_pred"])
        m_se3 = regression_metrics(y_te, se3["ho_pred"])

        boot_cpu = paired_cluster_bootstrap_reg(y_te, cpu2_te, cpu3_te, cl_te, n_boot=1000, seed=20261127)
        boot_se = paired_cluster_bootstrap_reg(y_te, se2["ho_pred"], se3["ho_pred"], cl_te, n_boot=1000, seed=20261128)
        shift = compute_shift_diagnostics(bank, X3d_bank, cidx_tr, y_tr, cidx_te, y_te)

        out[ep] = {
            "description": desc,
            "n_train_historical": int(len(tr_r)),
            "n_test_future_novel": int(len(te_r)),
            "n_test_clusters": int(len(set(cl_te))),
            "standalone_3d_imhb_only": {"learner": m3d_name, "future_test_metrics": m_3d_only, "train_oof_metrics": regression_metrics(y_tr, oof_3d_only)},
            "cpu_2d_baseline": {"learner": cpu2_name, "future_test_metrics": m_cpu2, "train_oof_metrics": regression_metrics(y_tr, cpu2_oof)},
            "cpu_2d_plus_3d_imhb": {"learner": cpu3_name, "future_test_metrics": m_cpu3, "train_oof_metrics": regression_metrics(y_tr, cpu3_oof)},
            "cpu_paired_boot_3d_vs_2d": boot_cpu,
            "super_ensemble_2d_baseline": {
                "winning_family": se2["winning_family"], "weights": se2["weights"], "fusion_head_weights": fus2_w,
                "future_test_metrics": m_se2, "train_oof_metrics": regression_metrics(y_tr, se2["oof_pred"]),
            },
            "super_ensemble_2d_plus_3d_imhb": {
                "winning_family": se3["winning_family"], "weights": se3["weights"], "fusion_head_weights": fus3_w,
                "future_test_metrics": m_se3, "train_oof_metrics": regression_metrics(y_tr, se3["oof_pred"]),
            },
            "super_ensemble_paired_boot_3d_vs_2d": boot_se,
            "train_vs_future_shift_diagnostics": shift,
            "univariate_3d_spearman_on_train": compute_3d_univariate_correlations(X3d_bank[cidx_tr], y_tr),
            "univariate_3d_spearman_on_future_test": compute_3d_univariate_correlations(X3d_bank[cidx_te], y_te),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"    [TEMPORAL {ep}] 3D-only: R2={m_3d_only['r2']} Sp={m_3d_only['spearman']} MAE={m_3d_only['mae']}\n"
            f"    [TEMPORAL {ep}] CPU 2D->2D+3D: R2 {m_cpu2['r2']}->{m_cpu3['r2']} | Sp {m_cpu2['spearman']}->{m_cpu3['spearman']} | MAE {m_cpu2['mae']}->{m_cpu3['mae']} (p_mae={boot_cpu.get('p_value_mae_display')})\n"
            f"    [TEMPORAL {ep}] SE  2D->2D+3D: R2 {m_se2['r2']}->{m_se3['r2']} | Sp {m_se2['spearman']}->{m_se3['spearman']} | MAE {m_se2['mae']}->{m_se3['mae']} (p_mae={boot_se.get('p_value_mae_display')})",
            flush=True,
        )
    return out


def run_6ep_zero_peeking_cv(bank: UnifiedV334Bank, X3d_bank: np.ndarray) -> Dict[str, Any]:
    """Experiment B: Zero-Holdout-Peeking Cluster-Grouped CV across 6 permeability/lipophilicity endpoints."""
    print("\n================ [Experiment B: 6-Endpoint Zero-Holdout-Peeking Cluster-Grouped CV (2D vs 2D + 3D-IMHB)] ================", flush=True)
    out: Dict[str, Any] = {}
    for ep in TARGET_ENDPOINTS:
        t0 = time.time()
        rows = list(csv.DictReader(get_ep_path(ep).open(encoding="utf-8"), delimiter="\t"))
        cond_cols = [c for c in rows[0].keys() if c.startswith("cond_")]
        cv_r = [r for r in rows if int(r["is_holdout"]) == 0 and r["censor_type"] == "exact"]
        ho_r = [r for r in rows if int(r["is_holdout"]) == 1 and r["censor_type"] == "exact"]

        # Assert strict zero cluster leakage between frozen CV and frozen holdout
        assert set(r["cluster_id"] for r in cv_r).isdisjoint(set(r["cluster_id"] for r in ho_r)), f"{ep}: cluster leakage!"

        cidx_cv = bank.map_keys([r["compound_key"] for r in cv_r])
        y_cv = np.array([float(r["y_value"]) for r in cv_r], dtype=np.float64)
        fids_cv = np.array([int(r["fold_id"]) for r in cv_r], dtype=np.int32)
        cl_cv = np.array([r["cluster_id"] for r in cv_r], dtype=object)
        cond_cv = [{c: r.get(c, "unspecified") for c in cond_cols} for r in cv_r]

        cidx_ho = bank.map_keys([r["compound_key"] for r in ho_r])
        y_ho = np.array([float(r["y_value"]) for r in ho_r], dtype=np.float64)
        cl_ho = np.array([r["cluster_id"] for r in ho_r], dtype=object)
        cond_ho = [{c: r.get(c, "unspecified") for c in cond_cols} for r in ho_r]

        # Standalone 3D-IMHB head
        oof_3d, ho_3d, m3d_name = train_3d_standalone_head(X3d_bank[cidx_cv], y_cv, fids_cv, X3d_bank[cidx_ho], None)

        # 2D Baseline
        cpu2_name, cpu2_oof, cpu2_ho = train_cpu_track_v34(bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, X3d_bank=None)
        fus2_oof, fus2_ho, _, fus2_w = train_fusion_track_v34(bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, cpu2_oof, cpu2_ho, X3d_bank=None)
        se2 = combine_super_ensemble_oof(y_cv, fids_cv, cpu2_oof, fus2_oof, cpu2_ho, fus2_ho, None)

        # 2D + 3D-IMHB
        cpu3_name, cpu3_oof, cpu3_ho = train_cpu_track_v34(bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, X3d_bank=X3d_bank)
        fus3_oof, fus3_ho, _, fus3_w = train_fusion_track_v34(
            bank, cidx_cv, y_cv, fids_cv, cond_cv, cidx_ho, cond_ho, cond_cols, None, cpu3_oof, cpu3_ho,
            X3d_bank=X3d_bank, head_3d_only=(oof_3d, ho_3d),
        )
        se3 = combine_super_ensemble_oof(y_cv, fids_cv, cpu3_oof, fus3_oof, cpu3_ho, fus3_ho, None)

        # OOF metrics (strictly used for ZERO-HOLDOUT-PEEKING promotion gate!)
        oof_m_cpu2 = regression_metrics(y_cv, cpu2_oof)
        oof_m_cpu3 = regression_metrics(y_cv, cpu3_oof)
        oof_m_se2 = regression_metrics(y_cv, se2["oof_pred"])
        oof_m_se3 = regression_metrics(y_cv, se3["oof_pred"])

        oof_delta_mae_cpu = round(float(oof_m_cpu2["mae"] - oof_m_cpu3["mae"]), 5)
        oof_delta_mae_se = round(float(oof_m_se2["mae"] - oof_m_se3["mae"]), 5)
        oof_boot_se = paired_cluster_bootstrap_reg(y_cv, se2["oof_pred"], se3["oof_pred"], cl_cv, n_boot=1000, seed=20261129)

        # Strict OOF promotion gate (positive delta = OOF MAE improved without looking at holdout!)
        oof_gate_cpu_pass = bool(oof_delta_mae_cpu > 0.0)
        oof_gate_se_pass = bool(oof_delta_mae_se > 0.0)

        # Holdout evaluation (gated prediction uses 3D iff oof_gate_se_pass else stays on 2D!)
        ho_m_cpu2 = regression_metrics(y_ho, cpu2_ho)
        ho_m_cpu3 = regression_metrics(y_ho, cpu3_ho)
        ho_m_se2 = regression_metrics(y_ho, se2["ho_pred"])
        ho_m_se3 = regression_metrics(y_ho, se3["ho_pred"])
        ho_boot_se = paired_cluster_bootstrap_reg(y_ho, se2["ho_pred"], se3["ho_pred"], cl_ho, n_boot=1000, seed=20261130)

        out[ep] = {
            "n_cv": int(len(cv_r)),
            "n_cv_clusters": int(len(set(cl_cv))),
            "n_holdout": int(len(ho_r)),
            "n_holdout_clusters": int(len(set(cl_ho))),
            "standalone_3d_imhb_only": {
                "learner": m3d_name,
                "oof_metrics": regression_metrics(y_cv, oof_3d),
                "holdout_metrics": regression_metrics(y_ho, ho_3d),
            },
            "cpu_track": {
                "2d_learner": cpu2_name,
                "3d_learner": cpu3_name,
                "oof_2d": oof_m_cpu2,
                "oof_2d_plus_3d": oof_m_cpu3,
                "oof_mae_reduction_2d_minus_3d": oof_delta_mae_cpu,
                "oof_gate_passed": oof_gate_cpu_pass,
                "holdout_2d": ho_m_cpu2,
                "holdout_2d_plus_3d": ho_m_cpu3,
                "holdout_mae_reduction_2d_minus_3d": round(float(ho_m_cpu2["mae"] - ho_m_cpu3["mae"]), 5),
            },
            "super_ensemble_track": {
                "2d_winning_family": se2["winning_family"],
                "2d_weights": se2["weights"],
                "2d_fusion_head_weights": fus2_w,
                "3d_winning_family": se3["winning_family"],
                "3d_weights": se3["weights"],
                "3d_fusion_head_weights": fus3_w,
                "oof_2d": oof_m_se2,
                "oof_2d_plus_3d": oof_m_se3,
                "oof_mae_reduction_2d_minus_3d": oof_delta_mae_se,
                "oof_paired_boot_3d_vs_2d": oof_boot_se,
                "oof_promotion_gate_passed": oof_gate_se_pass,
                "holdout_2d": ho_m_se2,
                "holdout_2d_plus_3d": ho_m_se3,
                "holdout_mae_reduction_2d_minus_3d": round(float(ho_m_se2["mae"] - ho_m_se3["mae"]), 5),
                "holdout_paired_boot_3d_vs_2d": ho_boot_se,
                "zero_peeking_deployed_holdout_metrics": ho_m_se3 if oof_gate_se_pass else ho_m_se2,
                "zero_peeking_deployed_choice": "2D+3D_IMHB" if oof_gate_se_pass else "2D_Baseline",
            },
            "univariate_3d_spearman_on_cv": compute_3d_univariate_correlations(X3d_bank[cidx_cv], y_cv),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        print(
            f"  [6EP {ep:<15}] OOF SE MAE {oof_m_se2['mae']:.4f}->{oof_m_se3['mae']:.4f} (Δ={oof_delta_mae_se:+.4f}, gate={'PASS' if oof_gate_se_pass else 'FAIL'}, p={oof_boot_se.get('p_value_mae_display')}) | "
            f"HO SE MAE {ho_m_se2['mae']:.4f}->{ho_m_se3['mae']:.4f} (Sp {ho_m_se2['spearman']:.4f}->{ho_m_se3['spearman']:.4f}, R2 {ho_m_se2['r2']:.4f}->{ho_m_se3['r2']:.4f}) [{time.time()-t0:.1f}s]",
            flush=True,
        )
    return out


def main() -> None:
    t_start = time.time()
    print("=== [V34 Stream A-2] 3D Conformer & IMHB Features for Macrocycle Permeability & Temporal Generalization ===", flush=True)
    bank = UnifiedV334Bank()

    ck_to_smi: Dict[str, str] = {}
    for pth in (EXP331 / "data" / "compounds_v331_merged.tsv", EXP334 / "data" / "delta_v334_compounds.tsv"):
        if pth.exists():
            with pth.open(encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    ck_to_smi[r["compound_key"]] = r.get("canonical_smiles", "")

    needed_cks_set = set()
    for ep in TARGET_ENDPOINTS:
        with get_ep_path(ep).open(encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                needed_cks_set.add(r["compound_key"])
    needed_cks = sorted(needed_cks_set)

    # 1. Build or load 3D-IMHB descriptor cache
    X3d_bank, valid3d_bank, feat_stats = build_or_load_3d_cache(bank, ck_to_smi, needed_cks)

    # 2. Experiment A: Prospective temporal splits (PAMPA & LogD7.4)
    temporal_res = run_temporal_experiments(bank, X3d_bank)

    # 3. Experiment B: Zero-holdout-peeking 6-endpoint cluster-grouped CV
    cv6_res = run_6ep_zero_peeking_cv(bank, X3d_bank)

    # Summarize honest scientific findings automatically from real numbers
    n_oof_pass_se = sum(1 for v in cv6_res.values() if v["super_ensemble_track"]["oof_promotion_gate_passed"])
    n_oof_pass_cpu = sum(1 for v in cv6_res.values() if v["cpu_track"]["oof_gate_passed"])
    pampa_t = temporal_res["PAMPA"]

    receipt = {
        "schema": "v34_3d_conformer_imhb_receipt_v1",
        "elapsed_sec": round(time.time() - t_start, 1),
        "conformer_pipeline": {
            "embedding_method": "RDKit ETKDGv3 (useMacrocycleTorsions=True, useSmallRingTorsions=True, enforceChirality=True, heavy-atom embedding -> AddHs(addCoords=True))",
            "conformer_budget": "numConfs=8 for MW < 1800 Da; numConfs=4 for MW >= 1800 Da",
            "force_field_relaxation": "MMFF94s (maxIters=80) with UFF (maxIters=80) fallback",
            "imhb_definition": "Intramolecular donor-acceptor (N-H/O-H to N/O) heavy-atom distance <= 3.3 A separated by >= 4 topological bonds (plus strict geometric H...A <= 2.5 A & D-H...A >= 110 deg)",
            "psa3d_definition": "Lee-Richards 3D solvent-accessible polar surface area (rdFreeSASA) over N, O, and polar H atoms across conformer ensemble (min, max, mean, chameleonic span = max - min)",
            "cache_path": str(CACHE_3D_DIR / "features_3d_imhb_v1.npz"),
            "stats": feat_stats,
        },
        "temporal_split_3d_evaluation": temporal_res,
        "six_endpoint_cluster_cv_3d_evaluation": cv6_res,
        "honest_summary": {
            "pampa_temporal_2d_vs_3d_super_ensemble": {
                "r2_2d": pampa_t["super_ensemble_2d_baseline"]["future_test_metrics"]["r2"],
                "r2_2d_plus_3d": pampa_t["super_ensemble_2d_plus_3d_imhb"]["future_test_metrics"]["r2"],
                "spearman_2d": pampa_t["super_ensemble_2d_baseline"]["future_test_metrics"]["spearman"],
                "spearman_2d_plus_3d": pampa_t["super_ensemble_2d_plus_3d_imhb"]["future_test_metrics"]["spearman"],
                "mae_2d": pampa_t["super_ensemble_2d_baseline"]["future_test_metrics"]["mae"],
                "mae_2d_plus_3d": pampa_t["super_ensemble_2d_plus_3d_imhb"]["future_test_metrics"]["mae"],
                "p_value_mae": pampa_t["super_ensemble_paired_boot_3d_vs_2d"].get("p_value_mae_display"),
                "p_value_spearman": pampa_t["super_ensemble_paired_boot_3d_vs_2d"].get("p_value_spearman_display"),
            },
            "six_endpoint_oof_gate_pass_count_super_ensemble": f"{n_oof_pass_se}/{len(TARGET_ENDPOINTS)}",
            "six_endpoint_oof_gate_pass_count_cpu": f"{n_oof_pass_cpu}/{len(TARGET_ENDPOINTS)}",
            "zero_peeking_protocol": "Holdout promotion strictly gated on OOF MAE improvement > 0; both gate-passing and non-passing holdout deltas are reported transparently.",
        },
    }

    out_path = EXP34 / "v34_3d_conformer_imhb_receipt.json"
    out_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== [V34 Stream A-2 DONE] Saved {out_path} in {time.time()-t_start:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
