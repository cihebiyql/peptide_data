#!/usr/bin/env python3
"""STEP 1 ONLY: Pre-compute all feature matrices and save to local disk cache."""
import csv, sys, time, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)

DATA = Path('<WORK_DIR>/s3')
PER = Path('<WORK_DIR>/s3_per')
CACHE = Path('<WORK_DIR>/s3/feat_cache')  # local dir, NOT NFS
CACHE.mkdir(exist_ok=True)

t0 = time.time()
print("Pre-computing ALL feature matrices...", flush=True)

# Build master compound list
all_cids = set()
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            if cid: all_cids.add(cid)
all_cids = sorted(all_cids)
n = len(all_cids)
print(f"Compounds: {n:,}", flush=True)

def fill(idx_dict, src_mat, dim):
    mat = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, c in enumerate(all_cids):
        j = idx_dict.get(c, -1)
        if j >= 0:
            mat[i] = src_mat[j]
            valid[i] = True
    return mat, valid

# 1. Morgan
print("[1/13] Morgan...", flush=True)
zm = np.load(DATA / 'morgan_fp16.npz')
src = zm['embeddings'].astype(np.float32)
lookup = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup[r['compound_id']] = int(r['row_idx'])
mat, val = fill(lookup, src, src.shape[1])
np.savez(CACHE / 'morgan.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape}, valid={val.sum():,} ({time.time()-t0:.0f}s)", flush=True)

# 2. ESM2
print("[2/13] ESM2...", flush=True)
ze = np.load(DATA / 'esm2_fp16.npz')
src = ze['embeddings'].astype(np.float32)
esm_lookup = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): esm_lookup[r['sequence']] = int(r['row_idx'])
# seq -> cid mapping
seq_to_cid = {}
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            if cid and seq and seq not in seq_to_cid: seq_to_cid[seq] = cid
mat = np.zeros((n, src.shape[1]), dtype=np.float32)
val = np.zeros(n, dtype=bool)
for seq, cid in seq_to_cid.items():
    pos = all_cids.index(cid) if cid in set(all_cids) else -1
    if pos >= 0:
        j = esm_lookup.get(seq, -1)
        if j >= 0: mat[pos] = src[j]; val[pos] = True
np.savez(CACHE / 'esm2.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape}, valid={val.sum():,} ({time.time()-t0:.0f}s)", flush=True)

# 3. ncAA
print("[3/13] ncAA...", flush=True)
zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
src = np.vstack([zn1['embeddings'], zn2['embeddings']]).astype(np.float32)
lookup = {}
with open(DATA / 'ncaa_set1_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup[r['compound_id']] = int(r['row_idx'])
with open(DATA / 'ncaa_mod_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup.setdefault(r['compound_id'], int(r['row_idx']) + zn1['embeddings'].shape[0])
mat, val = fill(lookup, src, src.shape[1])
np.savez(CACHE / 'ncaa.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape}, valid={val.sum():,} ({time.time()-t0:.0f}s)", flush=True)
del src, zn1, zn2

# 4. desc (always available)
print("[4/13] desc...", flush=True)
desc_lookup = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_lookup[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in
            ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']],
            dtype=np.float32)
mat = np.stack([desc_lookup.get(c, np.zeros(11, dtype=np.float32)) for c in all_cids])
val = np.ones(n, dtype=bool)
np.savez(CACHE / 'desc.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape} ({time.time()-t0:.0f}s)", flush=True)

# 5. ChemBERTa
print("[5/13] ChemBERTa...", flush=True)
zcb = np.load(DATA / 'chemberta_fp16.npz')
src = zcb['embeddings'].astype(np.float32)
lookup = {}
with open(DATA / 'chemberta_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup[r['compound_id']] = int(r['row_idx'])
mat, val = fill(lookup, src, src.shape[1])
np.savez(CACHE / 'chemberta.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape}, valid={val.sum():,} ({time.time()-t0:.0f}s)", flush=True)

# 6. ESM2-3B
print("[6/13] ESM2-3B...", flush=True)
ze3 = np.load(DATA / 'esm3b_fp16.npz')
src = ze3['embeddings'].astype(np.float32)
e3b_lookup = {}
with open(DATA / 'esm3b_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): e3b_lookup[r['sequence']] = int(r['row_idx'])
mat = np.zeros((n, src.shape[1]), dtype=np.float32)
val = np.zeros(n, dtype=bool)
for seq, cid in seq_to_cid.items():
    pos = all_cids.index(cid) if cid in set(all_cids) else -1
    if pos >= 0:
        j = e3b_lookup.get(seq, -1)
        if j >= 0: mat[pos] = src[j]; val[pos] = True
np.savez(CACHE / 'esm3b.npz', mat=mat.astype(np.float16), valid=val)
print(f"  {mat.shape}, valid={val.sum():,} ({time.time()-t0:.0f}s)", flush=True)
del src, ze3

# 7-9: RDKit full, DPC, AtomPair
print("[7-9] RDKit/DPC/AtomPair...", flush=True)
zx = np.load(DATA / 'expanded_cpu_features_fp16.npz')
src = zx['embeddings'].astype(np.float32)
lookup = {}
with open(DATA / 'expanded_cpu_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup[r['compound_id']] = int(r['row_idx'])
full_mat, exp_val = fill(lookup, src, src.shape[1])
np.savez(CACHE / 'rdkit_full.npz', mat=full_mat[:, 0:217].astype(np.float16), valid=exp_val)
np.savez(CACHE / 'dpc.npz', mat=full_mat[:, 217:617].astype(np.float16), valid=exp_val)
np.savez(CACHE / 'atompair.npz', mat=full_mat[:, 617:2665].astype(np.float16), valid=exp_val)
print(f"  rdkit {full_mat[:,0:217].shape}, valid={exp_val.sum():,} ({time.time()-t0:.0f}s)", flush=True)
del src, full_mat

# 10-13: AAC, PhysChem, DBAASP, Mod
print("[10-13] AAC/PhysChem/DBAASP/Mod...", flush=True)
za = np.load(DATA / 'all_features_fp16.npz')
src = za['embeddings'].astype(np.float32)
lookup = {}
with open(DATA / 'all_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): lookup[r['compound_id']] = int(r['row_idx'])
full_mat, allf_val = fill(lookup, src, src.shape[1])
np.savez(CACHE / 'aac.npz', mat=full_mat[:, 0:26].astype(np.float16), valid=allf_val)
np.savez(CACHE / 'physchem.npz', mat=full_mat[:, 26:36].astype(np.float16), valid=allf_val)
np.savez(CACHE / 'dbaasp.npz', mat=full_mat[:, 36:48].astype(np.float16), valid=allf_val)
np.savez(CACHE / 'mod.npz', mat=full_mat[:, 48:57].astype(np.float16), valid=allf_val)
print(f"  aac {full_mat[:,0:26].shape}, valid={allf_val.sum():,} ({time.time()-t0:.0f}s)", flush=True)

# Save compound list
with open(CACHE / 'compound_ids.json', 'w') as f:
    json.dump(all_cids, f)

print(f"\nALL DONE ({time.time()-t0:.0f}s)", flush=True)
print(f"Cache dir: {CACHE}")
for f in sorted(CACHE.glob('*.npz')):
    print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")
