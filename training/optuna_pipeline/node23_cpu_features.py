#!/usr/bin/env python3
"""node23: expanded CPU features (full RDKit desc + atom-pair + topo-torsion + DPC)."""
import csv, sys, time, math
from pathlib import Path
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors, Descriptors

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)
RDLogger.DisableLog("rdApp.*")

DATA = Path("<WORK_DIR>/s3")
ANNO = DATA / "compound_sequence_annotation.tsv"
SM = DATA / "structures_master.tsv"
OUT = DATA / "expanded_cpu_features_fp16.npz"
OUT_IDX = DATA / "expanded_cpu_features_index.tsv"

AA = "ACDEFGHIKLMNPQRSTVWY"
DPC_PAIRS = [a+b for a in AA for b in AA]

def dpc_features(seq):
    if len(seq) < 2: return [0.0]*400
    n = len(seq) - 1
    dpc = [0.0]*400
    for i in range(n):
        pair = seq[i:i+2]
        if pair[0] in AA and pair[1] in AA:
            dpc[AA.index(pair[0])*20 + AA.index(pair[1])] += 1.0/n
    return dpc

t0 = time.time()
print("Loading...", flush=True)
seq_data = {}
with open(ANNO, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        seq_data[r["compound_id"]] = r["residue_letters"]
smi_map = {}
with open(SM, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        smi_map[r["compound_id"]] = r["smiles_canonical"] or r["smiles_input"]

all_cids = sorted(set(seq_data) & set(smi_map))
n = len(all_cids)
desc_funcs = Descriptors._descList
desc_names = [name for name, _ in desc_funcs]
N_DESC = len(desc_names)
N_DPC = 400
N_AP = 2048
N_TT = 2048
TOTAL = N_DESC + N_DPC + N_AP + N_TT
print(f"Compounds: {n:,} | RDKit desc: {N_DESC} | Total dim: {TOTAL}", flush=True)

feat = np.zeros((n, TOTAL), dtype=np.float16)
for idx, cid in enumerate(all_cids):
    row = feat[idx]
    smi = smi_map[cid]
    mol = Chem.MolFromSmiles(smi)
    if mol:
        try:
            all_desc = Descriptors.CalcMolDescriptors(mol)
            for j, name in enumerate(desc_names):
                v = all_desc.get(name, 0.0)
                row[j] = v if math.isfinite(v) else 0.0
        except: pass
        try:
            ap = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=N_AP)
            arr = np.zeros(N_AP, dtype=np.float16)
            DataStructs.ConvertToNumpyArray(ap, arr)
            row[N_DESC:N_DESC+N_AP] = arr
        except: pass
        try:
            tt = rdMolDescriptors.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=N_TT)
            arr = np.zeros(N_TT, dtype=np.float16)
            DataStructs.ConvertToNumpyArray(tt, arr)
            row[N_DESC+N_AP:N_DESC+N_AP+N_TT] = arr
        except: pass
    seq = seq_data[cid]
    row[N_DESC+N_AP+N_TT:] = dpc_features(seq)
    if idx % 5000 == 0:
        print(f"  {idx:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

np.savez_compressed(OUT, embeddings=feat)
with open(OUT_IDX, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["row_idx","compound_id"] + desc_names +
               [f"dpc_{p}" for p in DPC_PAIRS] +
               [f"ap_{i}" for i in range(N_AP)] +
               [f"tt_{i}" for i in range(N_TT)])
    for i, cid in enumerate(all_cids):
        w.writerow([i, cid])
print(f"Done! {n:,}×{TOTAL} ({time.time()-t0:.0f}s)")
