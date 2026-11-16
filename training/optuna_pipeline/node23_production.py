#!/usr/bin/env python3
"""Node23: production model training with full grid search (64 cores).

Reads observations_annotated.tsv directly (self-contained).
Trains per-endpoint, excludes 20 demo drugs, predicts for them.
"""
import csv, sys, time, joblib, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

csv.field_size_limit(10_000_000)
DATA = Path('/data/qlyu/s3')
OUT = DATA / 'production'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
DESC_COLS = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']

DEMO = {
    'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽',
    'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽',
}
EXCLUDE = set(DEMO.keys())

def tune(X, y, groups):
    best_sp, best_p = -1, None
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))
    for ne in [500, 800]:
        for lr in [0.03, 0.05, 0.1]:
            for lv in [31, 63, 127]:
                for mc in [10, 20]:
                    pr = np.zeros_like(y)
                    for tr, te in splits:
                        m = lgb.LGBMRegressor(n_estimators=ne, learning_rate=lr,
                                              num_leaves=lv, min_child_samples=mc,
                                              subsample=0.9, colsample_bytree=0.9,
                                              random_state=0, verbose=-1, n_jobs=32)
                        m.fit(X[tr], y[tr]); pr[te] = m.predict(X[te])
                    sp = float(spearmanr(y, pr).statistic)
                    if sp > best_sp:
                        best_sp = sp
                        best_p = dict(n_estimators=ne, learning_rate=lr,
                                      num_leaves=lv, min_child_samples=mc,
                                      subsample=0.9, colsample_bytree=0.9,
                                      random_state=42, verbose=-1, n_jobs=32)
    return best_p, best_sp

t0 = time.time()
print('Loading features...', flush=True)

# Morgan
zm = np.load(DATA / 'morgan_fp16.npz')
mor = zm['embeddings'].astype(np.float32)
mor_idx = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        mor_idx[r['compound_id']] = int(r['row_idx'])

# ESM2
ze = np.load(DATA / 'esm2_fp16.npz')
esm = ze['embeddings'].astype(np.float32)
esm_idx = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        esm_idx[r['sequence']] = int(r['row_idx'])

# Descriptors
desc = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in DESC_COLS], dtype=np.float32)

print(f'Features loaded ({time.time()-t0:.1f}s)', flush=True)

# Demo drug features (from structures_master SMILES)
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.QED import qed
RDLogger.DisableLog('rdApp.*')

demo_feat = {}
with open(DATA / 'structures_master.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r['compound_id'] not in EXCLUDE: continue
        smi = r['smiles_canonical'] or r['smiles_input']
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        fp = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, fp)
        d = np.array([Descriptors.MolWt(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
                      Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
                      Lipinski.NumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
                      m.GetNumHeavyAtoms(), rdMolDescriptors.CalcFractionCSP3(m),
                      Chem.GetFormalCharge(m), qed(m)], dtype=np.float32)
        demo_feat[r['compound_id']] = {'morgan_desc': np.concatenate([fp, d]), 'desc': d}
print(f'Demo features: {len(demo_feat)}', flush=True)

# Demo actual values
demo_actual = {}
with open(DATA / 'observations_annotated.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        cid = r.get('compound_id', '')
        if cid in EXCLUDE and r.get('value_numeric'):
            try: v = float(r['value_numeric'])
            except: continue
            demo_actual.setdefault(cid, {}).setdefault(r['endpoint_family'], []).append(
                (v, r.get('unit_norm', ''), r.get('endpoint_raw', '')[:30]))

# Read observations and group by endpoint
ENDPOINTS = {
    'half_life': 'esm',      # ESM2 for all (OLD+NEW coverage)
    'EC50': 'morgan',         # Morgan for NEW
    'IC50': 'morgan',
    'hemolysis': 'esm',       # ESM2 strong for hemolysis
    'MIC': 'morgan',
    'MBC': 'morgan',
    'Ki': 'morgan',
    'half_life_terminal': 'esm',
    'PPB': 'morgan',
    'CL': 'morgan',
    'Vd': 'morgan',
}

ep_data = {}
with open(DATA / 'observations_annotated.tsv', encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if not r.get('value_numeric'): continue
        ep = r['endpoint_family']
        if ep not in ENDPOINTS: continue
        cid = r.get('compound_id', '')
        if not cid: continue
        try: y = float(r['value_numeric'])
        except: continue
        seq = (r.get('sequence') or '').strip().upper()
        ep_data.setdefault(ep, []).append({
            'cid': cid, 'y': y, 'seq': seq,
            'is_demo': cid in EXCLUDE,
            'ep_raw': r.get('endpoint_raw', ''),
            'unit': r.get('unit_norm', ''),
            'qualifier': r.get('qualifier', ''),
        })

print(f'Endpoint data loaded ({time.time()-t0:.1f}s)', flush=True)
for ep in sorted(ep_data, key=lambda x: -len(ep_data[x])):
    n_demo = sum(1 for r in ep_data[ep] if r['is_demo'])
    print(f'  {ep}: {len(ep_data[ep]):,} rows ({n_demo} demo)', flush=True)

# Train + predict per endpoint
results = []
for ep_name, feat_mode in ENDPOINTS.items():
    all_rows = ep_data.get(ep_name, [])
    if len(all_rows) < 50: continue

    # Split: training vs demo
    train_rows = []
    demo_rows = []
    seen = set()
    for r in all_rows:
        # 去重 (compound + value)
        k = (r['cid'], round(r['y'], 3))
        if k in seen: continue
        seen.add(k)
        if r['is_demo']:
            demo_rows.append(r)
        else:
            train_rows.append(r)

    if len(train_rows) < 50: continue

    # Build features
    if feat_mode == 'esm':
        # ESM2 + desc (where available)
        feats, ys, gs = [], [], []
        for r in train_rows:
            if r['seq'] not in esm_idx: continue
            e = esm[esm_idx[r['seq']]]
            d = desc.get(r['cid'], np.zeros(11, dtype=np.float32))
            feats.append(np.concatenate([e, d]))
            ys.append(r['y'])
            gs.append(r['cid'])
        demo_input = {}
        for r in demo_rows:
            if r['seq'] and r['seq'] in esm_idx:
                e = esm[esm_idx[r['seq']]]
                d = demo_feat.get(r['cid'], {}).get('desc', np.zeros(11, dtype=np.float32))
                demo_input[r['cid']] = np.concatenate([e, d])
    else:
        # Morgan + desc
        feats, ys, gs = [], [], []
        for r in train_rows:
            if r['cid'] not in mor_idx: continue
            fp = mor[mor_idx[r['cid']]]
            d = desc.get(r['cid'], np.zeros(11, dtype=np.float32))
            feats.append(np.concatenate([fp, d]))
            ys.append(r['y'])
            gs.append(r['cid'])
        demo_input = {}
        for r in demo_rows:
            if r['cid'] in demo_feat:
                demo_input[r['cid']] = demo_feat[r['cid']]['morgan_desc']

    if len(feats) < 50: continue
    X = np.stack(feats)
    y = np.array(ys)
    groups = np.array(gs)
    n_comp = len(set(gs))
    dim = X.shape[1]
    print(f'\n[{ep_name}] {len(X):,} rows / {n_comp:,} compounds | {feat_mode} ({dim}d) | ({time.time()-t0:.0f}s)', flush=True)

    # Grid search
    print(f'  Grid search ({time.time()-t0:.0f}s)...', flush=True)
    best_p, best_sp = tune(X, y, groups)
    print(f'  Best: ne={best_p["n_estimators"]} lr={best_p["learning_rate"]} '
          f'lv={best_p["num_leaves"]} mc={best_p["min_child_samples"]} → Sp={best_sp:.3f}', flush=True)

    # Train production model
    model = lgb.LGBMRegressor(**best_p)
    model.fit(X, y)
    joblib.dump({'model': model, 'endpoint': ep_name, 'feat_mode': feat_mode,
                 'params': best_p, 'cv_sp': best_sp, 'n_train': len(X)},
                OUT / 'models' / f'{ep_name}.joblib')

    # Predict demo drugs
    for cid, name in DEMO.items():
        if cid not in demo_input: continue
        pred = model.predict(demo_input[cid].reshape(1, -1))[0]
        actuals = demo_actual.get(cid, {}).get(ep_name, [])
        if not actuals and ep_name == 'half_life':
            actuals = demo_actual.get(cid, {}).get('Peptide half-life', []) or \
                      demo_actual.get(cid, {}).get('T1/2', [])
        act_str, act_val = '—', None
        if actuals:
            vals = [a[0] for a in actuals if a[0] is not None]
            if vals:
                act_val = float(np.median(vals))
                act_str = f'{act_val:.3g}'
        results.append([name, ep_name, act_str, f'{pred:.3g}',
                        actuals[0][1] if actuals else '', f'{best_sp:.3f}', len(X)])
        print(f'  → {name}: pred={pred:.3g} | actual={act_str}', flush=True)

# Output
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(results)

with open(OUT / 'report.md', 'w', encoding='utf-8') as f:
    f.write('# V32 生产模型（node23 调参版）\n\n```\n')
    f.write('\n'.join(f'{r[0]} | {r[1]} | actual={r[2]} | pred={r[3]} | cv_sp={r[5]}' for r in results))
    f.write('\n```\n')

print(f'\n{"="*60}')
print(f'[done] {len(results)} predictions, {time.time()-t0:.0f}s total')
print(f'[out] {OUT}')
