#!/usr/bin/env python3
"""STEP 1 (v2): Pre-compute features with minimal memory footprint."""
import csv, sys, time, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)
DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_per')
CACHE = DATA / 'feat_cache'
CACHE.mkdir(exist_ok=True)

t0 = time.time()

# Master compound list
all_cids = set()
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            if cid: all_cids.add(cid)
all_cids = sorted(all_cids)
n = len(all_cids)
cid_set = set(all_cids)
print(f"Compounds: {n:,}", flush=True)

def fill_from_dict(idx_dict, src_mat, dim):
    mat = np.zeros((n, dim), dtype=np.float16)  # keep float16 to save memory
    valid = np.zeros(n, dtype=bool)
    for i, c in enumerate(all_cids):
        j = idx_dict.get(c, -1)
        if j >= 0:
            mat[i] = src_mat[j]
            valid[i] = True
    return mat, valid

def load_idx(tsv_path, key_col='compound_id'):
    d = {}
    with open(tsv_path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            d[r[key_col]] = int(r['row_idx'])
    return d

# 1. Morgan
print("[1] Morgan...", flush=True)
z = np.load(DATA / 'morgan_fp16.npz'); src = z['embeddings']
lookup = load_idx(DATA / 'morgan_index.tsv')
mat, val = fill_from_dict(lookup, src, src.shape[1])
np.savez(CACHE / 'morgan.npz', mat=mat, valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 2. ESM2 (sequence-based)
print("[2] ESM2...", flush=True)
z = np.load(DATA / 'esm2_fp16.npz'); src = z['embeddings']
esm_idx = load_idx(DATA / 'esm2_index.tsv', 'sequence')
seq_map = {}
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            if cid in cid_set and seq and seq not in seq_map:
                seq_map[seq] = cid
mat = np.zeros((n, src.shape[1]), dtype=np.float16)
val = np.zeros(n, dtype=bool)
cid_to_pos = {c: i for i, c in enumerate(all_cids)}
for seq, cid in seq_map.items():
    p = cid_to_pos.get(cid, -1)
    j = esm_idx.get(seq, -1)
    if p >= 0 and j >= 0: mat[p] = src[j]; val[p] = True
np.savez(CACHE / 'esm2.npz', mat=mat, valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 3. ncAA (avoid vstack - write directly to preallocated matrix)
print("[3] ncAA...", flush=True)
z1 = np.load(DATA / 'ncaa_set1_fp16.npz'); s1 = z1['embeddings']
dim = s1.shape[1]
mat = np.zeros((n, dim), dtype=np.float16)
val = np.zeros(n, dtype=bool)
# set1
for r in csv.DictReader(open(DATA / 'ncaa_set1_index.tsv'), delimiter='\t'):
    p = cid_to_pos.get(r['compound_id'], -1)
    if p >= 0: mat[p] = s1[int(r['row_idx'])]; val[p] = True
del s1, z1
# set2 (offset indices)
z2 = np.load(DATA / 'ncaa_mod_fp16.npz'); s2 = z2['embeddings']
for r in csv.DictReader(open(DATA / 'ncaa_mod_index.tsv'), delimiter='\t'):
    cid = r['compound_id']
    if cid not in cid_to_pos: continue
    p = cid_to_pos[cid]
    if not val[p]:  # don't overwrite set1
        mat[p] = s2[int(r['row_idx'])]; val[p] = True
del s2, z2
np.savez(CACHE / 'ncaa.npz', mat=mat, valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 4. desc
print("[4] desc...", flush=True)
d = {}
with open(DATA / 'descriptors.tsv') as f:
    cols = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']
    for r in csv.DictReader(f, delimiter='\t'):
        d[r['compound_id']] = [float(r[c]) if r[c] else 0.0 for c in cols]
mat = np.array([[d.get(c, [0.0]*11)[j] for j in range(11)] for c in all_cids], dtype=np.float16)
np.savez(CACHE / 'desc.npz', mat=mat, valid=np.ones(n, dtype=bool))
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 5. ChemBERTa
print("[5] ChemBERTa...", flush=True)
z = np.load(DATA / 'chemberta_fp16.npz'); src = z['embeddings']
lookup = load_idx(DATA / 'chemberta_index.tsv')
mat, val = fill_from_dict(lookup, src, src.shape[1])
np.savez(CACHE / 'chemberta.npz', mat=mat, valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 6. ESM2-3B
print("[6] ESM2-3B...", flush=True)
z = np.load(DATA / 'esm3b_fp16.npz'); src = z['embeddings']
e3b_idx = load_idx(DATA / 'esm3b_index.tsv', 'sequence')
mat = np.zeros((n, src.shape[1]), dtype=np.float16)
val = np.zeros(n, dtype=bool)
for seq, cid in seq_map.items():
    p = cid_to_pos.get(cid, -1)
    j = e3b_idx.get(seq, -1)
    if p >= 0 and j >= 0: mat[p] = src[j]; val[p] = True
np.savez(CACHE / 'esm3b.npz', mat=mat, valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# 7-9: rdkit_full, dpc, atompair
print("[7-9] rdkit/dpc/atompair...", flush=True)
z = np.load(DATA / 'expanded_cpu_features_fp16.npz'); src = z['embeddings']
lookup = load_idx(DATA / 'expanded_cpu_features_index.tsv')
full, val = fill_from_dict(lookup, src, src.shape[1])
np.savez(CACHE / 'rdkit_full.npz', mat=full[:, 0:217], valid=val)
np.savez(CACHE / 'dpc.npz', mat=full[:, 217:617], valid=val)
np.savez(CACHE / 'atompair.npz', mat=full[:, 617:2665], valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)
del full, src, z

# 10-13: aac, physchem, dbaasp, mod
print("[10-13] aac/physchem/dbaasp/mod...", flush=True)
z = np.load(DATA / 'all_features_fp16.npz'); src = z['embeddings']
lookup = load_idx(DATA / 'all_features_index.tsv')
full, val = fill_from_dict(lookup, src, src.shape[1])
np.savez(CACHE / 'aac.npz', mat=full[:, 0:26], valid=val)
np.savez(CACHE / 'physchem.npz', mat=full[:, 26:36], valid=val)
np.savez(CACHE / 'dbaasp.npz', mat=full[:, 36:48], valid=val)
np.savez(CACHE / 'mod.npz', mat=full[:, 48:57], valid=val)
print(f"  ok ({time.time()-t0:.0f}s)", flush=True)

# Save compound list
with open(CACHE / 'cids.json', 'w') as f:
    json.dump(all_cids, f)

print(f"\nDONE ({time.time()-t0:.0f}s)", flush=True)
