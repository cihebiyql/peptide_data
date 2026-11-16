#!/usr/bin/env python3
"""Expand feature cache: compute ESM2 embeddings for OLD data sequences.
Adds OLD compounds to the cache so the parallel workers can use them."""
import csv, sys, time, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)
INTEG = Path('/data/qlyu/s3_integrated')
CACHE = Path('/data/qlyu/s3/feat_cache')
DATA = Path('/data/qlyu/s3')

t0 = time.time()

# Load current cache
with open(CACHE / 'cids.json') as f:
    existing_cids = json.load(f)
cid_pos = {c: i for i, c in enumerate(existing_cids)}
n_existing = len(existing_cids)

# Find OLD data sequences not in cache
old_records = {}  # record_id -> sequence
for fp in sorted(INTEG.glob('*__clean.tsv')):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('source', '') == 'OLD' or (not r.get('new_compound_id', '')):
                rid = r.get('record_id', '')
                seq = (r.get('sequence_notation') or '').strip().upper()
                if rid and seq and rid not in old_records:
                    old_records[rid] = seq

print(f"Existing cache: {n_existing:,} compounds")
print(f"OLD data unique records: {len(old_records):,}")

# Check which sequences already have ESM2 embeddings
z = np.load(CACHE / 'esm2.npz')
esm_mat = z['mat']
esm_valid = z['valid']

# Load ESM2 model to compute new embeddings
import torch
from transformers import AutoTokenizer, AutoModel

SNAP = '/data/qlyu/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
model = AutoModel.from_pretrained(SNAP, local_files_only=True, torch_dtype=torch.bfloat16)
model.eval().to(device)

# Get ESM2 index to check which sequences already have embeddings
esm2_index = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        esm2_index[r['sequence']] = int(r['row_idx'])

# Load raw ESM2 embeddings
ze = np.load(DATA / 'esm2_fp16.npz')
raw_esm = ze['embeddings']

# Compute embeddings for OLD sequences
old_seqs = list(old_records.values())
unique_old_seqs = list(set(old_seqs))
print(f"Unique OLD sequences to embed: {len(unique_old_seqs):,}")

new_embeddings = {}
BATCH = 16
for i in range(0, len(unique_old_seqs), BATCH):
    batch = unique_old_seqs[i:i+BATCH]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022).to(device)
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].bool()
    for b, seq in enumerate(batch):
        m = mask[b]
        L = int(m.sum().item())
        mm = m.clone(); mm[0] = False; mm[max(L-1,0)] = False
        new_embeddings[seq] = out[b][mm].float().mean(dim=0).cpu().numpy().astype(np.float16)
    if i % (BATCH * 20) == 0:
        print(f"  {i+len(batch):,}/{len(unique_old_seqs):,} ({time.time()-t0:.0f}s)", flush=True)

print(f"Computed {len(new_embeddings):,} new ESM2 embeddings ({time.time()-t0:.0f}s)")

# Expand cache: add OLD record IDs as new compound IDs
new_cids = list(existing_cids)
new_esm_mat = np.zeros((n_existing + len(old_records), esm_mat.shape[1]), dtype=np.float16)
new_esm_valid = np.zeros(n_existing + len(old_records), dtype=bool)

# Copy existing
new_esm_mat[:n_existing] = esm_mat
new_esm_valid[:n_existing] = esm_valid

# Add OLD records
for rid, seq in old_records.items():
    new_cids.append(rid)  # use record_id as compound_id for OLD data
    pos = len(new_cids) - 1
    emb = new_embeddings.get(seq)
    if emb is not None:
        new_esm_mat[pos] = emb
        new_esm_valid[pos] = True

# Save expanded cache
np.savez(CACHE / 'esm2.npz', mat=new_esm_mat, valid=new_esm_valid)

# Expand other feature groups with zeros for OLD compounds (they only have ESM2)
for gname in ['morgan', 'ncaa', 'desc', 'chemberta', 'esm3b', 'rdkit_full', 'dpc', 'atompair', 'aac', 'physchem', 'dbaasp', 'mod']:
    zg = np.load(CACHE / f'{gname}.npz')
    old_mat = zg['mat']
    old_valid = zg['valid']
    dim = old_mat.shape[1]
    new_mat = np.zeros((n_existing + len(old_records), dim), dtype=np.float16)
    new_valid = np.zeros(n_existing + len(old_records), dtype=bool)
    new_mat[:n_existing] = old_mat
    new_valid[:n_existing] = old_valid
    # desc is always valid (zeros for missing)
    if gname == 'desc':
        new_valid[n_existing:] = True
    np.savez(CACHE / f'{gname}.npz', mat=new_mat, valid=new_valid)

# Save updated compound list
with open(CACHE / 'cids.json', 'w') as f:
    json.dump(new_cids, f)

print(f"\nCache expanded: {n_existing:,} → {len(new_cids):,} compounds ({time.time()-t0:.0f}s)")
print(f"New ESM2 valid: {new_esm_valid[n_existing:].sum():,}/{len(old_records):,}")
