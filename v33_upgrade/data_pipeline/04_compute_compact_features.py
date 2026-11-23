#!/usr/bin/env python3
"""
Step 04 (runs on node23, 64 CPU cores, RDKit 2026.09.1):
Computes the Compact Anti-(p >> n) Feature Bank ONCE for ALL unique compounds in
compounds_v33.tsv + case17_queries.tsv, saving to <WORK_DIR>/v33_upgrade/features/:
  - compound_index.tsv (unified 0..N-1 mapping for all V33 + Case17 compounds)
  - physchem_core.npz (~83 curated, interpretable, zero-NaN/Inf peptide/macrocycle + sequence descriptors)
  - fp_compact.npz (chiral Morgan Count r2 1024D, chiral AtomPair 1024D, MACCS 167D, chiral Torsion 512D + valid mask)
  - morgan_bits_1024.npz (chiral Morgan Bit r2 1024D + packed 128-byte array for fast AD Tanimoto lookup)
  - feature_manifest.json (exact feature names, shapes, dtypes, SHA-256, RDKit version, Case17 indices)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, MACCSkeys, QED, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10_000_000)

BASE_DIR = Path("<WORK_DIR>/v33_upgrade")
DATA_DIR = BASE_DIR / "data"
FEAT_DIR = BASE_DIR / "features"
N_WORKERS = min(64, os.cpu_count() or 32)

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy scale
KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg consensus hydrophobicity scale (for hydrophobic moment muH)
EISENBERG_SCALE = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19,
    "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06,
    "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53,
    "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26,
}

# Boman protein-binding potential (kcal/mol)
BOMAN_SCALE = {
    "L": -4.92, "I": -4.92, "V": -4.04, "F": -2.98, "M": -2.35,
    "W": -2.33, "A": -1.81, "C": -1.28, "G": -0.94, "Y": 0.14,
    "T": 2.57, "S": 3.40, "H": 4.66, "Q": 5.54, "K": 5.55,
    "N": 6.64, "E": 6.81, "D": 8.72, "R": 14.92, "P": 0.0,
}

PKA_POS = {"K": 10.54, "R": 12.48, "H": 6.04}
PKA_NEG = {"D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46}

RDKIT_DESC_NAMES = [
    "MolWt", "Log10_MolWt", "ExactMolWt", "MolLogP", "MolMR",
    "TPSA", "TPSA_per_MW", "LabuteASA", "NumHDonors", "NumHAcceptors",
    "HBD_per_HBA", "NumRotatableBonds", "RotB_per_Heavy", "FractionCSP3",
    "HeavyAtomCount", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeteroatoms", "Hetero_per_Heavy", "NHOHCount",
    "NOCount", "FormalCharge", "AbsFormalCharge", "QED", "BertzCT",
    "Kappa1", "Kappa2", "HallKierAlpha", "Chi0v", "Chi1v",
    "NumAmideBonds", "ChiralCenterCount", "PosIonizableCount",
    "NegIonizableCount", "EstNetCharge_pH74", "LogP_per_TPSA",
]

MOD_DESC_NAMES = [
    "is_cyclic", "macrocycle_ring_size", "n_disulfide", "n_nmethyl",
    "n_d_aa_approx", "fatty_diacid_len", "fatty_monoacyl_len",
    "has_peg", "has_c_amidation", "has_n_acetylation",
]

SEQ_DESC_NAMES = (
    [f"aa_frac_{a}" for a in AA20]
    + [
        "seq_len", "log10_seq_len", "seq_net_charge_pH74", "seq_charge_density",
        "seq_hydrophobicity_kd", "seq_hydrophobicity_eisenberg",
        "seq_muH_alpha_100", "seq_muH_beta_160",
        "seq_aromatic_frac", "seq_basic_frac", "seq_acidic_frac",
        "seq_polar_frac", "seq_aliphatic_frac", "seq_boman_index", "seq_isoelectric_point",
    ]
)

ALL_PHYSCHEM_NAMES = RDKIT_DESC_NAMES + MOD_DESC_NAMES + SEQ_DESC_NAMES  # 38 + 10 + 35 = 83 dims!

# Worker globals
_GENS = {}
_ION_SMARTS = {}


def _init_worker():
    global _GENS, _ION_SMARTS
    _GENS = {
        "morgan_cnt": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
        "morgan_bit": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True),
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024, includeChirality=True),
        "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=512, includeChirality=True),
    }
    _ION_SMARTS = {
        # Basic protonatable amines/guanidines/imidazoles at pH 7.4 (not amides!)
        "pos_amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O);!$(NS=O);!$([n])][CX4]"),
        "pos_guanidine": Chem.MolFromSmarts("[NX3;H1,H2]C(=[NX2;H1,H0])[NX3;H1,H2]"),
        "neg_carboxyl": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
        "neg_sulfate_phos": Chem.MolFromSmarts("[SX4,PX4](=O)(=O)[OX2H1,OX1-]"),
    }


def _safe_float(x, lo=-1e6, hi=1e6) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return 0.0
        return max(lo, min(hi, v))
    except Exception:
        return 0.0


def _seq_charge_at_ph(seq: str, ph: float) -> float:
    q = 1.0 / (1.0 + 10.0 ** (ph - 8.0))  # N-term
    q -= 1.0 / (1.0 + 10.0 ** (3.1 - ph))  # C-term
    for ch in seq:
        if ch in PKA_POS:
            q += 1.0 / (1.0 + 10.0 ** (ph - PKA_POS[ch]))
        elif ch in PKA_NEG:
            q -= 1.0 / (1.0 + 10.0 ** (PKA_NEG[ch] - ph))
    return q


def _compute_seq_descriptors(seq: str) -> tuple[bool, list[float]]:
    clean = [c for c in (seq or "").upper() if c in KD_SCALE]
    if len(clean) < 2:
        return False, [0.0] * len(SEQ_DESC_NAMES)
    L = len(clean)
    cnt = {a: 0 for a in AA20}
    for c in clean:
        cnt[c] += 1
    aa_fracs = [cnt[a] / L for a in AA20]
    net_q = _seq_charge_at_ph("".join(clean), 7.4)
    q_dens = net_q / L
    kd_mean = sum(KD_SCALE[c] for c in clean) / L
    eis_vals = [EISENBERG_SCALE[c] for c in clean]
    eis_mean = sum(eis_vals) / L

    # Hydrophobic moment for alpha-helix (100 deg) and beta-strand (160 deg)
    def mu_h(angle_deg: float) -> float:
        rad = math.radians(angle_deg)
        sx = sum(v * math.sin(i * rad) for i, v in enumerate(eis_vals))
        cx = sum(v * math.cos(i * rad) for i, v in enumerate(eis_vals))
        return math.sqrt(sx * sx + cx * cx) / L

    arom = (cnt["F"] + cnt["W"] + cnt["Y"]) / L
    basic = (cnt["K"] + cnt["R"] + cnt["H"]) / L
    acidic = (cnt["D"] + cnt["E"]) / L
    polar = (cnt["S"] + cnt["T"] + cnt["N"] + cnt["Q"] + cnt["C"]) / L
    aliph = (cnt["A"] + cnt["V"] + cnt["I"] + cnt["L"] + cnt["M"]) / L
    boman = -sum(BOMAN_SCALE[c] for c in clean) / L

    # Bisection for isoelectric point (pI) in [1.0, 14.0]
    lo, hi = 1.0, 14.0
    s_clean = "".join(clean)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if _seq_charge_at_ph(s_clean, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    pi_val = 0.5 * (lo + hi)

    feats = aa_fracs + [
        float(L), math.log10(float(L)), net_q, q_dens,
        kd_mean, eis_mean, mu_h(100.0), mu_h(160.0),
        arom, basic, acidic, polar, aliph, boman, pi_val,
    ]
    return True, [_safe_float(x) for x in feats]


def _compute_one(item):
    idx, smi, seq, mod_vals = item
    seq_valid, seq_feats = _compute_seq_descriptors(seq)

    if not smi:
        return (
            idx, False, seq_valid,
            np.array([0.0] * len(RDKIT_DESC_NAMES) + mod_vals + seq_feats, dtype=np.float32),
            None, None, None, None, None,
        )

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (
            idx, False, seq_valid,
            np.array([0.0] * len(RDKIT_DESC_NAMES) + mod_vals + seq_feats, dtype=np.float32),
            None, None, None, None, None,
        )

    mw = _safe_float(Descriptors.MolWt(mol), 1.0, 1e6)
    exact_mw = _safe_float(Descriptors.ExactMolWt(mol), 1.0, 1e6)
    logp = _safe_float(Crippen.MolLogP(mol), -50.0, 50.0)
    mr = _safe_float(Crippen.MolMR(mol), 0.0, 1e5)
    tpsa = _safe_float(rdMolDescriptors.CalcTPSA(mol), 0.0, 1e5)
    labute = _safe_float(rdMolDescriptors.CalcLabuteASA(mol), 0.0, 1e5)
    hbd = float(Lipinski.NumHDonors(mol))
    hba = float(Lipinski.NumHAcceptors(mol))
    rotb = float(Lipinski.NumRotatableBonds(mol))
    fcsp3 = _safe_float(rdMolDescriptors.CalcFractionCSP3(mol), 0.0, 1.0)
    heavy = float(max(1, mol.GetNumHeavyAtoms()))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    arom_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    aliph_rings = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    sat_rings = float(rdMolDescriptors.CalcNumSaturatedRings(mol))
    hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    nhoh = float(Lipinski.NHOHCount(mol))
    no_cnt = float(Lipinski.NOCount(mol))
    fchg = float(Chem.GetFormalCharge(mol))
    try:
        qed_val = _safe_float(QED.qed(mol), 0.0, 1.0)
    except Exception:
        qed_val = 0.0
    try:
        bertz = _safe_float(Descriptors.BertzCT(mol), 0.0, 1e6) if heavy <= 350 else _safe_float(heavy * math.log2(max(2.0, heavy)) * 10.0, 0.0, 1e6)
    except Exception:
        bertz = 0.0
    k1 = _safe_float(rdMolDescriptors.CalcKappa1(mol), 0.0, 1e5)
    k2 = _safe_float(rdMolDescriptors.CalcKappa2(mol), 0.0, 1e5)
    hka = _safe_float(rdMolDescriptors.CalcHallKierAlpha(mol), -1e4, 1e4)
    chi0v = _safe_float(rdMolDescriptors.CalcChi0v(mol), 0.0, 1e5)
    chi1v = _safe_float(rdMolDescriptors.CalcChi1v(mol), 0.0, 1e5)
    amide = float(rdMolDescriptors.CalcNumAmideBonds(mol))
    chiral = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=True)))
    pos_ion = float(
        len(mol.GetSubstructMatches(_ION_SMARTS["pos_amine"]))
        + len(mol.GetSubstructMatches(_ION_SMARTS["pos_guanidine"]))
    )
    neg_ion = float(
        len(mol.GetSubstructMatches(_ION_SMARTS["neg_carboxyl"]))
        + len(mol.GetSubstructMatches(_ION_SMARTS["neg_sulfate_phos"]))
    )
    est_q74 = pos_ion - neg_ion

    rd_feats = [
        mw, math.log10(max(1.0, mw)), exact_mw, logp, mr,
        tpsa, tpsa / max(1.0, mw), labute, hbd, hba,
        hbd / max(1.0, hba), rotb, rotb / heavy, fcsp3,
        heavy, rings, arom_rings, aliph_rings,
        sat_rings, hetero, hetero / heavy, nhoh,
        no_cnt, fchg, abs(fchg), qed_val, bertz,
        k1, k2, hka, chi0v, chi1v,
        amide, chiral, pos_ion, neg_ion, est_q74, logp / max(1.0, tpsa),
    ]
    physchem = np.array([_safe_float(x) for x in rd_feats] + mod_vals + seq_feats, dtype=np.float32)

    # Compact chiral fingerprints
    m_cnt = np.clip(_GENS["morgan_cnt"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    m_bit = _GENS["morgan_bit"].GetFingerprintAsNumPy(mol).astype(np.uint8)
    ap_cnt = np.clip(_GENS["atompair"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    tt_cnt = np.clip(_GENS["torsion"].GetCountFingerprintAsNumPy(mol), 0, 65535).astype(np.uint16)
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint8)  # 167D

    return (idx, True, seq_valid, physchem, m_cnt, ap_cnt, maccs, tt_cnt, m_bit)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading compounds_v33.tsv + case17_queries.tsv (workers={N_WORKERS})...", flush=True)
    unified = []
    seen_keys = {}

    with (DATA_DIR / "compounds_v33.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ckey = row["compound_key"]
            seen_keys[ckey] = len(unified)
            unified.append({
                "compound_idx": len(unified),
                "compound_key": ckey,
                "canonical_smiles": row["canonical_smiles"],
                "smiles_origin": row["smiles_origin"],
                "parent_sequence": row["parent_sequence"],
                "seq_origin": row["seq_origin"],
                "is_case17": 0,
                "case17_drug_name": "",
                "mods": [float(row[k]) for k in MOD_DESC_NAMES],
            })

    case17_indices = {}
    with (DATA_DIR / "case17_queries.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ckey = row["compound_key"]
            dname = row["drug_name"]
            if ckey in seen_keys:
                cidx = seen_keys[ckey]
                unified[cidx]["is_case17"] = 1
                unified[cidx]["case17_drug_name"] = dname
                case17_indices[dname] = cidx
            else:
                cidx = len(unified)
                seen_keys[ckey] = cidx
                unified.append({
                    "compound_idx": cidx,
                    "compound_key": ckey,
                    "canonical_smiles": row["canonical_smiles"],
                    "smiles_origin": row["smiles_origin"],
                    "parent_sequence": row["parent_sequence"],
                    "seq_origin": row["seq_origin"],
                    "is_case17": 1,
                    "case17_drug_name": dname,
                    "mods": [float(row[k]) for k in MOD_DESC_NAMES],
                })
                case17_indices[dname] = cidx

    N = len(unified)
    print(f"  Total unified compounds (V33 + Case17): {N:,} (Case17 mapped: {len(case17_indices)}/17)", flush=True)

    # Pre-allocate compact arrays
    X_phys = np.zeros((N, len(ALL_PHYSCHEM_NAMES)), dtype=np.float32)
    fp_morgan_cnt = np.zeros((N, 1024), dtype=np.uint16)
    fp_atompair = np.zeros((N, 1024), dtype=np.uint16)
    fp_maccs = np.zeros((N, 167), dtype=np.uint8)
    fp_torsion = np.zeros((N, 512), dtype=np.uint16)
    fp_morgan_bit = np.zeros((N, 1024), dtype=np.uint8)
    smiles_valid_mask = np.zeros(N, dtype=bool)
    seq_valid_mask = np.zeros(N, dtype=bool)

    print("[2/4] Computing physchem_core + compact chiral fingerprints across 64 cores...", flush=True)
    tasks = [(u["compound_idx"], u["canonical_smiles"], u["parent_sequence"], u["mods"]) for u in unified]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
        for idx, smi_ok, seq_ok, phys, m_cnt, ap_cnt, maccs, tt_cnt, m_bit in ex.map(_compute_one, tasks, chunksize=128):
            smiles_valid_mask[idx] = smi_ok
            seq_valid_mask[idx] = seq_ok
            X_phys[idx] = phys
            if smi_ok:
                fp_morgan_cnt[idx] = m_cnt
                fp_atompair[idx] = ap_cnt
                fp_maccs[idx] = maccs
                fp_torsion[idx] = tt_cnt
                fp_morgan_bit[idx] = m_bit

    # Strict finite checks
    assert np.all(np.isfinite(X_phys)), "Non-finite value detected in physchem_core!"
    assert all(smiles_valid_mask[cidx] for cidx in case17_indices.values()), "A Case17 drug failed SMILES featurization!"
    fp_morgan_packed = np.packbits(fp_morgan_bit, axis=1)  # (N, 128) uint8

    print(
        f"  Feature computation finished in {time.time()-t0:.1f}s | "
        f"smiles_valid={int(smiles_valid_mask.sum()):,}/{N:,} | seq_valid={int(seq_valid_mask.sum()):,}/{N:,}",
        flush=True,
    )

    print("[3/4] Saving compressed .npz archives & compound_index.tsv...", flush=True)
    idx_path = FEAT_DIR / "compound_index.tsv"
    with idx_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["compound_idx", "compound_key", "is_case17", "case17_drug_name", "smiles_origin", "seq_origin", "smiles_valid", "seq_valid"],
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        for u in unified:
            i = u["compound_idx"]
            w.writerow({
                "compound_idx": i,
                "compound_key": u["compound_key"],
                "is_case17": u["is_case17"],
                "case17_drug_name": u["case17_drug_name"],
                "smiles_origin": u["smiles_origin"],
                "seq_origin": u["seq_origin"],
                "smiles_valid": int(smiles_valid_mask[i]),
                "seq_valid": int(seq_valid_mask[i]),
            })

    phys_path = FEAT_DIR / "physchem_core.npz"
    np.savez_compressed(
        phys_path,
        X=X_phys,
        feature_names=np.array(ALL_PHYSCHEM_NAMES),
        smiles_valid_mask=smiles_valid_mask,
        seq_valid_mask=seq_valid_mask,
        compound_keys=np.array([u["compound_key"] for u in unified]),
    )

    fp_path = FEAT_DIR / "fp_compact.npz"
    np.savez_compressed(
        fp_path,
        morgan_count_r2_1024=fp_morgan_cnt,
        atompair_1024=fp_atompair,
        maccs_167=fp_maccs,
        torsion_512=fp_torsion,
        smiles_valid_mask=smiles_valid_mask,
        compound_keys=np.array([u["compound_key"] for u in unified]),
    )

    ad_path = FEAT_DIR / "morgan_bits_1024.npz"
    np.savez_compressed(
        ad_path,
        morgan_bits_1024=fp_morgan_bit,
        morgan_bits_packed=fp_morgan_packed,
        smiles_valid_mask=smiles_valid_mask,
        compound_keys=np.array([u["compound_key"] for u in unified]),
    )

    print("[4/4] Writing feature_manifest.json with exact SHA-256 hashes...", flush=True)
    manifest = {
        "schema": "peptide_omnipanel_v33_compact_feature_bank_v1",
        "rdkit_version": rdkit.__version__,
        "numpy_version": np.__version__,
        "n_compounds_total": N,
        "n_smiles_valid": int(smiles_valid_mask.sum()),
        "n_seq_valid": int(seq_valid_mask.sum()),
        "case17_indices": case17_indices,
        "physchem_core": {
            "file": str(phys_path),
            "sha256": sha256_file(phys_path),
            "size_bytes": phys_path.stat().st_size,
            "shape": list(X_phys.shape),
            "dtype": str(X_phys.dtype),
            "n_rdkit_desc": len(RDKIT_DESC_NAMES),
            "n_mod_desc": len(MOD_DESC_NAMES),
            "n_seq_desc": len(SEQ_DESC_NAMES),
            "feature_names": ALL_PHYSCHEM_NAMES,
        },
        "fp_compact": {
            "file": str(fp_path),
            "sha256": sha256_file(fp_path),
            "size_bytes": fp_path.stat().st_size,
            "arrays": {
                "morgan_count_r2_1024": {"shape": list(fp_morgan_cnt.shape), "dtype": str(fp_morgan_cnt.dtype), "params": "radius=2,fpSize=1024,includeChirality=True,count"},
                "atompair_1024": {"shape": list(fp_atompair.shape), "dtype": str(fp_atompair.dtype), "params": "fpSize=1024,includeChirality=True,count"},
                "maccs_167": {"shape": list(fp_maccs.shape), "dtype": str(fp_maccs.dtype), "params": "MACCSkeys 167 bits"},
                "torsion_512": {"shape": list(fp_torsion.shape), "dtype": str(fp_torsion.dtype), "params": "fpSize=512,includeChirality=True,count"},
            },
        },
        "morgan_bits_1024": {
            "file": str(ad_path),
            "sha256": sha256_file(ad_path),
            "size_bytes": ad_path.stat().st_size,
            "arrays": {
                "morgan_bits_1024": {"shape": list(fp_morgan_bit.shape), "dtype": str(fp_morgan_bit.dtype)},
                "morgan_bits_packed": {"shape": list(fp_morgan_packed.shape), "dtype": str(fp_morgan_packed.dtype)},
            },
        },
        "compound_index": {
            "file": str(idx_path),
            "sha256": sha256_file(idx_path),
            "size_bytes": idx_path.stat().st_size,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (FEAT_DIR / "feature_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DONE Step 04 in {manifest['elapsed_sec']}s! "
        f"physchem={phys_path.stat().st_size/1e6:.2f}MB, fp_compact={fp_path.stat().st_size/1e6:.2f}MB, morgan_bits={ad_path.stat().st_size/1e6:.2f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
