#!/usr/bin/env python3
"""Node23: production model using per-endpoint tables (log-transformed values)."""
import csv, sys, time, joblib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.QED import qed

csv.field_size_limit(10_000_000)
RDLogger.DisableLog('rdApp.*')
DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_per')
OUT = DATA / 'production_v3'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
DESC_COLS = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']

DEMO = {
    'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽',
    'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽',
}
EXCLUDE = set(DEMO.keys())

t0 = time.time()

# Load features
zm = np.load(DATA / 'morgan_fp16.npz')
mor = zm['embeddings'].astype(np.float32)
mor_idx = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        mor_idx[r['compound_id']] = int(r['row_idx'])

ze = np.load(DATA / 'esm2_fp16.npz')
esm = ze['embeddings'].astype(np.float32)
esm_idx = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        esm_idx[r['sequence']] = int(r['row_idx'])

desc = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in DESC_COLS], dtype=np.float32)

# Demo drug features
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

# Demo actual values (from per-endpoint tables)
demo_actual = {}
for ep in ['half_life', 'EC50', 'IC50', 'PPB', 'CL', 'Vd', 'hemolysis']:
    fp = PER / f'{ep}__clean.tsv'
    if not fp.exists(): continue
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            if cid in EXCLUDE and r.get('value', ''):
                demo_actual.setdefault(cid, {}).setdefault(ep, []).append(
                    (float(r['value']), r.get('unit_target', '')))

print(f'Features loaded ({time.time()-t0:.1f}s), demo_feat={len(demo_feat)}', flush=True)

# Endpoints to train
ENDPOINTS = {
    'half_life': 'esm+desc',
    'EC50': 'morgan+desc',
    'IC50': 'morgan+desc',
    'hemolysis': 'esm+desc',
    'MIC': 'morgan+desc',
    'MBC': 'morgan+desc',
    'Ki': 'morgan+desc',
    'PPB': 'morgan+desc',
    'CL': 'morgan+desc',
    'Vd': 'morgan+desc',
}

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                   min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                   random_state=42, verbose=-1, n_jobs=32)

results = []
for ep_name, feat_mode in ENDPOINTS.items():
    fp = PER / f'{ep_name}__clean.tsv'
    if not fp.exists():
        print(f'\n[{ep_name}] no file, skip', flush=True)
        continue

    # Read data
    train_rows, demo_rows = [], []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: y = float(r['value'])
            except: continue

            if cid in EXCLUDE:
                demo_rows.append((cid, y, seq, r))
            else:
                train_rows.append((cid, y, seq, r))

    # Build training features
    if 'esm' in feat_mode:
        feats, ys, gs = [], [], []
        for cid, y, seq, r in train_rows:
            if seq and seq in esm_idx:
                e = esm[esm_idx[seq]]
                d = desc.get(cid, np.zeros(11, dtype=np.float32))
                feats.append(np.concatenate([e, d]))
                ys.append(y)
                gs.append(cid)
        # demo features
        demo_input = {}
        for cid, y, seq, r in demo_rows:
            if seq and seq in esm_idx:
                e = esm[esm_idx[seq]]
                d = demo_feat.get(cid, {}).get('desc', np.zeros(11))
                demo_input[cid] = np.concatenate([e, d])
    else:
        feats, ys, gs = [], [], []
        for cid, y, seq, r in train_rows:
            if cid in mor_idx:
                fp_arr = mor[mor_idx[cid]]
                d = desc.get(cid, np.zeros(11, dtype=np.float32))
                feats.append(np.concatenate([fp_arr, d]))
                ys.append(y)
                gs.append(cid)
        demo_input = {}
        for cid, y, seq, r in demo_rows:
            if cid in demo_feat:
                demo_input[cid] = demo_feat[cid]['morgan_desc']

    if len(feats) < 50:
        print(f'\n[{ep_name}] {len(feats)} rows < 50, skip', flush=True)
        continue

    X = np.stack(feats)
    y = np.array(ys)
    groups = np.array(gs)
    unit = train_rows[0][3].get('unit_target', '')
    n_comp = len(set(gs))
    print(f'\n[{ep_name}] {len(X):,} rows / {n_comp:,} compounds | {feat_mode} ({X.shape[1]}d) | {unit}', flush=True)

    # CV evaluation
    gkf = GroupKFold(n_splits=5)
    cv_pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**LGBM_PARAMS)
        m.fit(X[tr], y[tr])
        cv_pred[te] = m.predict(X[te])
    cv_sp = float(spearmanr(y, cv_pred).statistic)
    cv_mae = float(np.mean(np.abs(y - cv_pred)))
    print(f'  CV: Sp={cv_sp:.3f} MAE={cv_mae:.3f}', flush=True)

    # Train production model
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    joblib.dump({'model': model, 'endpoint': ep_name, 'feat_mode': feat_mode,
                 'unit': unit, 'cv_sp': cv_sp, 'cv_mae': cv_mae,
                 'n_train': len(X), 'params': LGBM_PARAMS},
                OUT / 'models' / f'{ep_name}.joblib')

    # Predict demo drugs
    for cid, name in DEMO.items():
        if cid not in demo_input: continue
        pred = model.predict(demo_input[cid].reshape(1, -1))[0]
        actuals = demo_actual.get(cid, {}).get(ep_name, [])
        act_str, act_val = '—', None
        if actuals:
            act_val = float(np.median([a[0] for a in actuals]))
            act_str = f'{act_val:.3g}'
        results.append([name, ep_name, act_str, f'{pred:.3g}', unit,
                        f'{cv_sp:.3f}', len(X)])
        print(f'  → {name}: pred={pred:.3g} {unit} | actual={act_str}', flush=True)

# Output
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(results)

with open(OUT / 'report.md', 'w', encoding='utf-8') as f:
    f.write('# V32 生产模型 v3 — 肽药样本外预测（调参后）\n\n')
    f.write(f'> 训练时排除 {len(EXCLUDE)} 种演示药物 | 特征+参数已调优\n\n| 药物 | 端点 | 实测 | 预测 | 单位 | CV Sp | 训练行 |\n|---|---|---|---|---|---|---|\n')
    for r in results:
        f.write(f'| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[4]} | {r[5]} | {r[6]:,} |\n')

print(f'\n{"="*60}')
print(f'[done] {len(results)} predictions, {time.time()-t0:.0f}s')
