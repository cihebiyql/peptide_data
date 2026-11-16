#!/usr/bin/env python3
"""补录渗透性端点：PAMPA / Caco2 / RRCK（原始数据中有但从未进入端点表）。
1) 从 observations_annotated.tsv 提取 3 个端点表（20 列 schema 与现有表一致）
2) 用现有源特征文件扩展 feat_cache（无需 GPU：morgan/esm2/ncaa/rdkit_full 等已覆盖 95-100%）
"""
import csv, sys, time, json
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
csv.field_size_limit(10_000_000)
DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_integrated')
CACHE = DATA / 'feat_cache'
FAMS = {
    'PAMPA_log10Papp': 'PAMPA',
    'Caco2_log10Papp': 'Caco2',
    'RRCK_log10Papp': 'RRCK',
}
t0 = time.time()

# ---------- 1. 提取观察 ----------
obs = {fam: {} for fam in FAMS}  # fam -> {(cid): best row}
with open(DATA / 'observations_annotated.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        fam = r.get('endpoint_family', '')
        if fam not in FAMS: continue
        cid = r.get('compound_id', '')
        if not cid: continue
        v = r.get('value_numeric', '') or r.get('value', '')
        try: val = float(v)
        except: continue
        # 同一化合物多次测量：保留所有 (cid, value) 对，写表时展开
        obs[fam].setdefault(cid, set()).add(round(val, 6))

# ---------- 2. 注释 + 结构 ----------
annot = {}
with open(DATA / 'compound_sequence_annotation.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        annot[r['compound_id']] = r
master = {}
with open(DATA / 'structures_master.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        master[r['compound_id']] = r

HEADER = ['unified_endpoint','task_kind','source','value','unit_target','relation',
          'sequence_notation','identity_key','is_canonical','inchikey','subsource',
          'endpoint_detail','record_id','reference','partition','cross_dup',
          'new_compound_id','new_mw','homology_cluster_90','cluster_size']

new_cids = set()
for fam, ep in FAMS.items():
    n_rows = 0
    with open(PER / f'{ep}__clean.tsv', 'w', encoding='utf-8', newline='') as out:
        w = csv.writer(out, delimiter='\t', lineterminator='\n')
        w.writerow(HEADER)
        for cid, vals in sorted(obs[fam].items()):
            a = annot.get(cid, {})
            m = master.get(cid, {})
            seq = (a.get('residue_letters') or '').strip()
            for val in sorted(vals):
                w.writerow([ep, 'regression', 'NEW', f'{val:g}', 'log10(cm/s)', '=',
                            seq, seq or (m.get('smiles_canonical') or ''),
                            a.get('is_canonical', ''), m.get('inchikey_computed', ''),
                            fam, ep, '', cid, '', '0',
                            cid, a.get('mw_numeric', ''), '', ''])
                n_rows += 1
                new_cids.add(cid)
    print(f"[{ep}] {len(obs[fam]):,} 化合物 / {n_rows:,} 行 已写入", flush=True)

# ---------- 3. 扩展特征缓存 ----------
with open(CACHE / 'cids.json') as f:
    all_cids = json.load(f)
cid_pos = {c: i for i, c in enumerate(all_cids)}
new_cids = sorted(c for c in new_cids if c not in cid_pos)
n_new = len(new_cids)
n_old = len(all_cids)
print(f"\n缓存扩展: {n_old:,} -> {n_old+n_new:,} (+{n_new:,})", flush=True)
new_pos = {c: i for i, c in enumerate(new_cids)}  # 填充函数写入 add_mat（局部 0..n_new-1）

def load_idx(path, key_col='compound_id'):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            d[r[key_col]] = int(r['row_idx'])
    return d

def expand_group(gname, fill_fn, always_valid=False):
    z = np.load(CACHE / f'{gname}.npz')
    mat, val = z['mat'], z['valid']
    dim = mat.shape[1]
    add_mat = np.zeros((n_new, dim), dtype=np.float16)
    add_val = np.zeros(n_new, dtype=bool)
    hit = fill_fn(add_mat, add_val)
    if always_valid: add_val[:] = True
    np.savez(CACHE / f'{gname}.npz',
             mat=np.concatenate([mat, add_mat]),
             valid=np.concatenate([val, add_val]))
    print(f"  {gname}: +{hit}/{n_new}", flush=True)

# cid -> row 索引类
z = np.load(DATA / 'morgan_fp16.npz'); morgan_src = z['embeddings']; morgan_idx = load_idx(DATA / 'morgan_index.tsv')
def fill_morgan(am, av):
    h = 0
    for c, p in new_pos.items():
        j = morgan_idx.get(c, -1)
        if j >= 0: am[p] = morgan_src[j]; av[p] = True; h += 1
    return h
expand_group('morgan', fill_morgan)
del morgan_src

ncaa_idx = {}
for src_name in ['ncaa_set1', 'ncaa_mod']:
    z2 = np.load(DATA / f'{src_name}_fp16.npz')
    src2 = z2['embeddings']
    idx2 = load_idx(DATA / f'{src_name}_index.tsv')
    for c, p in new_pos.items():
        j = idx2.get(c, -1)
        if j >= 0 and c not in ncaa_idx:
            ncaa_idx[c] = (src2, j)
def fill_ncaa(am, av):
    h = 0
    for c, p in new_pos.items():
        if c in ncaa_idx:
            src2, j = ncaa_idx[c]
            am[p] = src2[j]; av[p] = True; h += 1
    return h
expand_group('ncaa', fill_ncaa)
del ncaa_idx

desc_d = {}
with open(DATA / 'descriptors.tsv') as f:
    cols = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']
    for r in csv.DictReader(f, delimiter='\t'):
        if r['compound_id'] in new_pos:
            desc_d[r['compound_id']] = [float(r[c]) if r[c] else 0.0 for c in cols]
def fill_desc(am, av):
    h = 0
    for c, p in new_pos.items():
        if c in desc_d: am[p] = desc_d[c]; h += 1
    return h
expand_group('desc', fill_desc, always_valid=True)

z = np.load(DATA / 'chemberta_fp16.npz'); cb_src = z['embeddings']; cb_idx = load_idx(DATA / 'chemberta_index.tsv')
def fill_cb(am, av):
    h = 0
    for c, p in new_pos.items():
        j = cb_idx.get(c, -1)
        if j >= 0: am[p] = cb_src[j]; av[p] = True; h += 1
    return h
expand_group('chemberta', fill_cb)
del cb_src

# 序列键类（esm2 / esm3b）
cid_seq = {}
for c in new_pos:
    a = annot.get(c)
    if a:
        s = (a.get('residue_letters') or '').strip().upper()
        if s: cid_seq[c] = s
for gname, npz_name in [('esm2', 'esm2_fp16'), ('esm3b', 'esm3b_fp16')]:
    z = np.load(DATA / f'{npz_name}.npz'); src = z['embeddings']
    sidx = load_idx(DATA / f'{gname}_index.tsv', 'sequence')
    def fill_seq(am, av, src=src, sidx=sidx):
        h = 0
        for c, p in new_pos.items():
            s = cid_seq.get(c)
            j = sidx.get(s, -1) if s else -1
            if j >= 0: am[p] = src[j]; av[p] = True; h += 1
        return h
    expand_group(gname, fill_seq)
    del src

# expanded_cpu (rdkit_full/dpc/atompair) + all_features (aac/physchem/dbaasp/mod)
z = np.load(DATA / 'expanded_cpu_features_fp16.npz'); ec_src = z['embeddings']; ec_idx = load_idx(DATA / 'expanded_cpu_features_index.tsv')
ec_full = np.zeros((n_new, ec_src.shape[1]), dtype=np.float16)
ec_val = np.zeros(n_new, dtype=bool)
ec_hit = 0
for c, p in new_pos.items():
    j = ec_idx.get(c, -1)
    if j >= 0: ec_full[p] = ec_src[j]; ec_val[p] = True; ec_hit += 1
for gname, sl in [('rdkit_full', slice(0, 217)), ('dpc', slice(217, 617)), ('atompair', slice(617, 2665))]:
    zg = np.load(CACHE / f'{gname}.npz')
    np.savez(CACHE / f'{gname}.npz',
             mat=np.concatenate([zg['mat'], ec_full[:, sl]]),
             valid=np.concatenate([zg['valid'], ec_val]))
    print(f"  {gname}: +{ec_hit}/{n_new}", flush=True)
del ec_src, ec_full

z = np.load(DATA / 'all_features_fp16.npz'); af_src = z['embeddings']; af_idx = load_idx(DATA / 'all_features_index.tsv')
af_full = np.zeros((n_new, af_src.shape[1]), dtype=np.float16)
af_val = np.zeros(n_new, dtype=bool)
af_hit = 0
for c, p in new_pos.items():
    j = af_idx.get(c, -1)
    if j >= 0: af_full[p] = af_src[j]; af_val[p] = True; af_hit += 1
for gname, sl in [('aac', slice(0, 26)), ('physchem', slice(26, 36)), ('dbaasp', slice(36, 48)), ('mod', slice(48, 57))]:
    zg = np.load(CACHE / f'{gname}.npz')
    np.savez(CACHE / f'{gname}.npz',
             mat=np.concatenate([zg['mat'], af_full[:, sl]]),
             valid=np.concatenate([zg['valid'], af_val]))
    print(f"  {gname}: +{af_hit}/{n_new}", flush=True)
del af_src, af_full

# 更新 cids.json
all_cids.extend(new_cids)
with open(CACHE / 'cids.json', 'w') as f:
    json.dump(all_cids, f)
print(f"\nDONE ({time.time()-t0:.0f}s) — 可启动 PAMPA/Caco2/RRCK worker", flush=True)
