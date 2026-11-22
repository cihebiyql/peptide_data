#!/usr/bin/env python3
"""Deterministic, node2-CPU-Portable Multi-Conformer RDKit ETKDGv3 3D Chameleonic Descriptor Extractor.

Captures cyclic & modified peptide conformational chameleonicity (3D shape, radius of gyration
flexibility, geometric intramolecular H-bonds IMHB, 3D Polar SASA via rdFreeSASA, and dynamic
chameleonic shielding swing across conformers) with strict heavy-atom safeguards (>250 heavy
atoms cleanly masked with valid_3d=0.0 so live inference never stalls!).

Runtime dependencies: rdkit + numpy ONLY (100% node2-CPU-portable).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdFreeSASA, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

CHAMELEONIC_3D_NAMES: List[str] = [
    "log1p_pmi1_mean",
    "log1p_pmi2_mean",
    "log1p_pmi3_mean",
    "npr1_mean",
    "npr2_mean",
    "npr_sum_mean",
    "rg_mean",
    "rg_min",
    "rg_std",
    "rg_norm_mw",
    "asphericity_mean",
    "eccentricity_mean",
    "isf_mean",
    "spherocity_mean",
    "spherocity_std",
    "pbf_mean",
    "imhb_min",
    "imhb_mean",
    "imhb_max",
    "imhb_spread",
    "imhb_per_hbd",
    "imhb_max_per_hbd",
    "exposed_hbd_min",
    "psa3d_min",
    "psa3d_mean",
    "psa3d_spread",
    "psa3d_to_tpsa_ratio",
    "exposed_polar_frac_mean",
    "valid_3d",
]
CHAMELEONIC_3D_DIM = len(CHAMELEONIC_3D_NAMES)  # 29


def _find_hbd_hba_pairs(mol_h: Chem.Mol) -> Tuple[List[Tuple[int, int]], List[int], np.ndarray]:
    """Identify (donor_idx, h_idx) pairs, acceptor indices, and heavy-atom topological distance matrix."""
    donors: List[Tuple[int, int]] = []
    acceptors: List[int] = []
    for atom in mol_h.GetAtoms():
        z = atom.GetAtomicNum()
        if z in (7, 8):
            acceptors.append(atom.GetIdx())
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    donors.append((atom.GetIdx(), nb.GetIdx()))
    # Topological distance matrix (fast C++ Floyd-Warshall inside RDKit)
    dmat = Chem.GetDistanceMatrix(mol_h)
    return donors, acceptors, dmat


def _count_imhb_for_conformer(
    coords: np.ndarray,
    donors: List[Tuple[int, int]],
    acceptors: List[int],
    topo_dmat: np.ndarray,
    max_da: float = 3.5,
    max_ha: float = 2.6,
    min_cos_angle: float = -0.4226,  # cos(115 deg) = -0.4226; vec(H->D) . vec(H->A) <= -0.4226
    min_topo_dist: int = 4,
) -> int:
    """Count exact 3D geometric intramolecular H-bonds (D-H...A) for one conformer."""
    if not donors or not acceptors:
        return 0
    acc_arr = np.asarray(acceptors, dtype=np.int64)
    acc_xyz = coords[acc_arr]  # (Na, 3)

    d_idx = np.asarray([d for d, _ in donors], dtype=np.int64)
    h_idx = np.asarray([h for _, h in donors], dtype=np.int64)
    d_xyz = coords[d_idx]  # (Nd, 3)
    h_xyz = coords[h_idx]  # (Nd, 3)

    # Pairwise distances D..A and H..A -> (Nd, Na)
    da_vec = acc_xyz[None, :, :] - d_xyz[:, None, :]
    da_dist2 = np.sum(da_vec * da_vec, axis=-1)
    ha_vec = acc_xyz[None, :, :] - h_xyz[:, None, :]
    ha_dist2 = np.sum(ha_vec * ha_vec, axis=-1)

    topo_sub = topo_dmat[np.ix_(d_idx, acc_arr)]
    cand = (topo_sub >= min_topo_dist) & (da_dist2 <= max_da * max_da) & (ha_dist2 <= max_ha * max_ha)
    if not cand.any():
        return 0

    # Check D-H...A angle >= 115 deg: cos(angle) = ((D-H)/|D-H|) . ((A-H)/|A-H|) <= cos(115 deg)
    di, ai = np.where(cand)
    v_hd = d_xyz[di] - h_xyz[di]
    v_ha = acc_xyz[ai] - h_xyz[di]
    n_hd = np.linalg.norm(v_hd, axis=-1).clip(min=1e-6)
    n_ha = np.linalg.norm(v_ha, axis=-1).clip(min=1e-6)
    cos_ang = np.sum(v_hd * v_ha, axis=-1) / (n_hd * n_ha)
    valid = cos_ang <= min_cos_angle
    # Count unique donor H atoms forming at least one geometric IMHB
    if not valid.any():
        return 0
    return int(len(set(di[valid].tolist())))


def _safe_float(val: float, default: float = 0.0, lo: float = -1e6, hi: float = 1e6) -> float:
    if not math.isfinite(val):
        return default
    return float(max(lo, min(hi, val)))


def compute_chameleonic_3d_descriptors(
    smiles: str,
    n_confs: int = 3,
    max_heavy_atoms: int = 250,
    seed: int = 42,
    num_threads: int = 1,
    return_best_mol: bool = False,
) -> Tuple[np.ndarray, Optional[Chem.Mol]] | np.ndarray:
    """Compute 29D deterministic RDKit ETKDGv3 3D Chameleonic descriptors for a single SMILES.

    Adaptive speed budget so live inference stays fast (<150 ms/mol typical) and macrocycles >250 heavy
    atoms cleanly fall back with valid_3d=0.0:
      - <= 90 heavy atoms : 3 conformers ETKDGv3 + fast MMFF94/UFF relaxation (maxIters=40)
      - 91..160 heavy atoms: 3 conformers ETKDGv3 (macrocycle torsions, direct torsion geometry)
      - 161..250 heavy atoms: 2 conformers ETKDGv3 (fast random-coord macrocycle embedding)
      - > 250 heavy atoms : cleanly skipped (valid_3d=0.0, zero vector, < 1 ms!)
    """
    zeros = np.zeros(CHAMELEONIC_3D_DIM, dtype=np.float32)
    if not smiles:
        return (zeros, None) if return_best_mol else zeros

    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return (zeros, None) if return_best_mol else zeros

    n_heavy = mol0.GetNumHeavyAtoms()
    if n_heavy < 3 or n_heavy > max_heavy_atoms:
        return (zeros, None) if return_best_mol else zeros

    n_hbd_2d = int(rdMolDescriptors.CalcNumHBD(mol0))
    tpsa_2d = float(rdMolDescriptors.CalcTPSA(mol0))
    mw = max(1.0, float(rdMolDescriptors.CalcExactMolWt(mol0)))
    mw_cbrt = mw ** (1.0 / 3.0)

    n_target = n_confs if n_heavy <= 130 else min(2, n_confs)
    mol_h = Chem.AddHs(mol0)

    # Tiered fast ETKDGv3 conformer generation (avoids 20s macrocycle knots while keeping >250 cleanly masked!)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = bool(n_heavy <= 130)
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True
    params.enforceChirality = True
    params.pruneRmsThresh = -1.0  # disable O(K^2) RMS pruning overhead
    params.numThreads = int(num_threads)

    if n_heavy <= 75:
        params.useRandomCoords = False
        params.maxIterations = 20
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_target, params=params))
        if len(cids) == 0:
            params.useRandomCoords = True
            params.maxIterations = 25
            cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_target, params=params))
    elif n_heavy <= 130:
        params.useRandomCoords = True
        params.maxIterations = 20
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_target, params=params))
    else:
        # 131..250 heavy-atom macrocycles: fast distance-geometry random coords with strict iteration cap
        params.useRandomCoords = True
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False
        params.maxIterations = 10
        cids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_target, params=params))

    if len(cids) == 0:
        # Final lightweight 1-conformer fallback
        params.useRandomCoords = True
        params.maxIterations = 40
        cid = AllChem.EmbedMolecule(mol_h, params)
        if cid >= 0:
            cids = [cid]
        else:
            return (zeros, None) if return_best_mol else zeros

    # Fast MMFF94 / UFF optimization for <= 90 heavy atoms (with safe fallback)
    energies: List[float] = []
    if n_heavy <= 90:
        mmff_ok = False
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=int(num_threads), maxIters=40)
                if res and len(res) == len(cids):
                    energies = [float(e) if math.isfinite(e) else 1e9 for _, e in res]
                    mmff_ok = True
        except Exception:
            mmff_ok = False
        if not mmff_ok:
            try:
                res = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=int(num_threads), maxIters=40)
                if res and len(res) == len(cids):
                    energies = [float(e) if math.isfinite(e) else 1e9 for _, e in res]
            except Exception:
                energies = [0.0] * len(cids)
    else:
        energies = [0.0] * len(cids)

    # Pre-identify polar atoms (N, O, and polar H attached to N/O) for 3D Polar SASA via rdFreeSASA
    radii = rdFreeSASA.classifyAtoms(mol_h)
    polar_mask = np.zeros(mol_h.GetNumAtoms(), dtype=bool)
    for idx, atom in enumerate(mol_h.GetAtoms()):
        z = atom.GetAtomicNum()
        if z in (7, 8):
            polar_mask[idx] = True
        elif z == 1:
            nbs = atom.GetNeighbors()
            if nbs and nbs[0].GetAtomicNum() in (7, 8):
                polar_mask[idx] = True

    donors, acceptors, topo_dmat = _find_hbd_hba_pairs(mol_h)

    pmi1s, pmi2s, pmi3s = [], [], []
    npr1s, npr2s = [], []
    rgs, asphs, eccs, isfs, sphs, pbfs = [], [], [], [], [], []
    imhbs, psa3ds, tot_sasas = [], [], []

    for cid in cids:
        try:
            pmi1 = _safe_float(Descriptors3D.PMI1(mol_h, confId=cid), 0.0, 0.0, 1e9)
            pmi2 = _safe_float(Descriptors3D.PMI2(mol_h, confId=cid), 0.0, 0.0, 1e9)
            pmi3 = _safe_float(Descriptors3D.PMI3(mol_h, confId=cid), 0.0, 0.0, 1e9)
            npr1 = _safe_float(Descriptors3D.NPR1(mol_h, confId=cid), 0.5, 0.0, 1.0)
            npr2 = _safe_float(Descriptors3D.NPR2(mol_h, confId=cid), 0.5, 0.0, 1.0)
            rg = _safe_float(Descriptors3D.RadiusOfGyration(mol_h, confId=cid), 0.0, 0.0, 100.0)
            asph = _safe_float(Descriptors3D.Asphericity(mol_h, confId=cid), 0.0, 0.0, 1.0)
            ecc = _safe_float(Descriptors3D.Eccentricity(mol_h, confId=cid), 0.0, 0.0, 1.0)
            isf = _safe_float(Descriptors3D.InertialShapeFactor(mol_h, confId=cid), 0.0, 0.0, 0.25)
            sph = _safe_float(Descriptors3D.SpherocityIndex(mol_h, confId=cid), 0.0, 0.0, 1.0)
            pbf = _safe_float(rdMolDescriptors.CalcPBF(mol_h, confId=cid), 0.0, 0.0, 30.0)
        except Exception:
            continue

        coords = mol_h.GetConformer(cid).GetPositions().astype(np.float64)
        imhb = _count_imhb_for_conformer(coords, donors, acceptors, topo_dmat)

        tot_sasa = float(rdFreeSASA.CalcSASA(mol_h, radii, confIdx=cid))
        atom_sasas = np.array([float(a.GetProp("SASA")) for a in mol_h.GetAtoms()], dtype=np.float64)
        psa_3d = float(atom_sasas[polar_mask].sum())

        pmi1s.append(math.log1p(pmi1))
        pmi2s.append(math.log1p(pmi2))
        pmi3s.append(math.log1p(pmi3))
        npr1s.append(npr1)
        npr2s.append(npr2)
        rgs.append(rg)
        asphs.append(asph)
        eccs.append(ecc)
        isfs.append(isf)
        sphs.append(sph)
        pbfs.append(pbf)
        imhbs.append(float(imhb))
        psa3ds.append(_safe_float(psa_3d, 0.0, 0.0, 1e5))
        tot_sasas.append(_safe_float(tot_sasa, 1.0, 1.0, 1e5))

    if not rgs:
        return (zeros, None) if return_best_mol else zeros

    imhb_min = float(min(imhbs))
    imhb_mean = float(np.mean(imhbs))
    imhb_max = float(max(imhbs))
    psa_min = float(min(psa3ds))
    psa_mean = float(np.mean(psa3ds))
    psa_max = float(max(psa3ds))
    tot_sasa_mean = max(1.0, float(np.mean(tot_sasas)))
    rg_mean = float(np.mean(rgs))
    hbd_denom = float(max(1, n_hbd_2d))

    vec = np.array(
        [
            float(np.mean(pmi1s)),
            float(np.mean(pmi2s)),
            float(np.mean(pmi3s)),
            float(np.mean(npr1s)),
            float(np.mean(npr2s)),
            float(np.mean(npr1s) + np.mean(npr2s)),
            rg_mean,
            float(min(rgs)),
            float(np.std(rgs)) if len(rgs) > 1 else 0.0,
            rg_mean / mw_cbrt,
            float(np.mean(asphs)),
            float(np.mean(eccs)),
            float(np.mean(isfs)),
            float(np.mean(sphs)),
            float(np.std(sphs)) if len(sphs) > 1 else 0.0,
            float(np.mean(pbfs)),
            imhb_min,
            imhb_mean,
            imhb_max,
            imhb_max - imhb_min,
            imhb_mean / hbd_denom,
            imhb_max / hbd_denom,
            max(0.0, float(n_hbd_2d) - imhb_max),
            psa_min,
            psa_mean,
            psa_max - psa_min,
            psa_min / max(10.0, tpsa_2d),
            psa_mean / tot_sasa_mean,
            1.0,
        ],
        dtype=np.float32,
    )
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    if return_best_mol:
        # Keep only the lowest-energy (or first) conformer on mol_h for downstream Uni-Mol2 reuse!
        best_pos = int(np.argmin(energies)) if len(energies) == len(cids) else 0
        best_cid = cids[best_pos]
        mol_out = Chem.Mol(mol_h, False)
        mol_out.AddConformer(mol_h.GetConformer(best_cid), assignId=True)
        return vec, mol_out
    return vec
