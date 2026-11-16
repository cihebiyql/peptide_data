#!/usr/bin/env python3
"""Final production run: use Optuna results for done endpoints + quick LGBM for remaining."""
import csv, sys, time, joblib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

csv.field_size_limit(10_000_000)
DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_per')
OUT = DATA / 'final'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}

t0 = time.time()

# Load features
print("Loading...", flush=True)
zm = np.load(DATA / 'morgan_fp16.npz')
MOR = zm['embeddings'].astype(np.float32)
mor_idx = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): mor_idx[r['compound_id']] = int(r['row_idx'])

ze = np.load(DATA / 'esm2_fp16.npz')
ESM = ze['embeddings'].astype(np.float32)
esm_idx = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): esm_idx[r['sequence']] = int(r['row_idx'])

desc_map = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_map[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in
            ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']],
            dtype=np.float32)

# DBAASP from all_features
za = np.load(DATA / 'all_features_fp16.npz')
ALLF = za['embeddings'].astype(np.float32)
allf_idx = {}
with open(DATA / 'all_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): allf_idx[r['compound_id']] = int(r['row_idx'])

# AtomPair from expanded CPU
zx = np.load(DATA / 'expanded_cpu_features_fp16.npz')
EXP = zx['embeddings'].astype(np.float32)
exp_idx = {}
with open(DATA / 'expanded_cpu_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): exp_idx[r['compound_id']] = int(r['row_idx'])

# ChemBERTa
zcb = np.load(DATA / 'chemberta_fp16.npz')
CB = zcb['embeddings'].astype(np.float32)
cb_idx = {}
with open(DATA / 'chemberta_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): cb_idx[r['compound_id']] = int(r['row_idx'])

print(f"Loaded ({time.time()-t0:.1f}s)", flush=True)

# Optuna best configs (from completed endpoints)
OPTUNA_BEST = {
    'half_life': {'features': ['desc','atompair','dbaasp','mod'], 'model': 'gbdt', 'sp': 0.5387},
    'EC50': {'features': ['esm2','ncaa','rdkit_full','aac','physchem'], 'model': 'gbdt', 'sp': 0.6687},
    'IC50': {'features': ['morgan','esm2','desc','chemberta','rdkit_full','aac','dbaasp'], 'model': 'rf', 'sp': 0.6152},
}

# For remaining endpoints, use proven good combos
FALLBACK_COMBOS = {
    'hemolysis': ['esm2', 'desc', 'aac', 'physchem'],
    'MIC': ['morgan', 'esm2', 'desc', 'aac', 'dbaasp'],
    'MBC': ['morgan', 'esm2', 'desc', 'aac', 'dbaasp'],
    'Ki': ['esm2', 'ncaa', 'desc', 'aac', 'physchem'],
    'Kd': ['morgan', 'esm2', 'desc'],
    'LC_lemma': ['esm2', 'desc', 'physchem'],
    'MFC': ['morgan', 'desc', 'dbaasp', 'mod'],
}


def build_features(rows, feat_list):
    """Build feature matrix for given rows and feature list."""
    n = len(rows)
    cids = [r[0] for r in rows]
    seqs = [r[2] for r in rows]
    parts = []

    if 'morgan' in feat_list:
        m = np.zeros((n, 2048), dtype=np.float32)
        for j, c in enumerate(cids):
            i = mor_idx.get(c, -1)
            if i >= 0: m[j] = MOR[i]
        parts.append(m)
    if 'esm2' in feat_list:
        m = np.zeros((n, 1280), dtype=np.float32)
        for j, s in enumerate(seqs):
            i = esm_idx.get(s, -1)
            if i >= 0: m[j] = ESM[i]
        parts.append(m)
    if 'ncaa' in feat_list:
        zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
        zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
        NCAA = np.vstack([zn1['embeddings'], zn2['embeddings']]).astype(np.float32)
        ncaa_idx = {}
        with open(DATA / 'ncaa_set1_index.tsv') as f:
            for r in csv.DictReader(f, delimiter='\t'): ncaa_idx[r['compound_id']] = int(r['row_idx'])
        with open(DATA / 'ncaa_mod_index.tsv') as f:
            for r in csv.DictReader(f, delimiter='\t'): ncaa_idx.setdefault(r['compound_id'], int(r['row_idx']) + zn1['embeddings'].shape[0])
        m = np.zeros((n, 2048), dtype=np.float32)
        for j, c in enumerate(cids):
            i = ncaa_idx.get(c, -1)
            if i >= 0: m[j] = NCAA[i]
        parts.append(m)
    if 'desc' in feat_list:
        m = np.stack([desc_map.get(c, np.zeros(11, dtype=np.float32)) for c in cids])
        parts.append(m)
    if 'chemberta' in feat_list:
        m = np.zeros((n, 384), dtype=np.float32)
        for j, c in enumerate(cids):
            i = cb_idx.get(c, -1)
            if i >= 0: m[j] = CB[i]
        parts.append(m)
    if 'rdkit_full' in feat_list:
        m = np.zeros((n, 217), dtype=np.float32)
        for j, c in enumerate(cids):
            i = exp_idx.get(c, -1)
            if i >= 0: m[j] = EXP[i, 0:217]
        parts.append(m)
    if 'atompair' in feat_list:
        m = np.zeros((n, 2048), dtype=np.float32)
        for j, c in enumerate(cids):
            i = exp_idx.get(c, -1)
            if i >= 0: m[j] = EXP[i, 617:2665]
        parts.append(m)
    if 'aac' in feat_list:
        m = np.zeros((n, 26), dtype=np.float32)
        for j, c in enumerate(cids):
            i = allf_idx.get(c, -1)
            if i >= 0: m[j] = ALLF[i, 0:26]
        parts.append(m)
    if 'physchem' in feat_list:
        m = np.zeros((n, 10), dtype=np.float32)
        for j, c in enumerate(cids):
            i = allf_idx.get(c, -1)
            if i >= 0: m[j] = ALLF[i, 26:36]
        parts.append(m)
    if 'dbaasp' in feat_list:
        m = np.zeros((n, 12), dtype=np.float32)
        for j, c in enumerate(cids):
            i = allf_idx.get(c, -1)
            if i >= 0: m[j] = ALLF[i, 36:48]
        parts.append(m)
    if 'mod' in feat_list:
        m = np.zeros((n, 9), dtype=np.float32)
        for j, c in enumerate(cids):
            i = allf_idx.get(c, -1)
            if i >= 0: m[j] = ALLF[i, 48:57]
        parts.append(m)

    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    y = np.array([r[1] for r in rows])
    groups = np.array([r[0] for r in rows])
    return X, y, groups


# Process ALL endpoints
ALL_ENDPOINTS = ['half_life', 'EC50', 'IC50', 'hemolysis', 'MIC', 'MBC', 'Ki', 'Kd', 'LC_lemma', 'MFC']
results = []
demo_preds = []

for ep in ALL_ENDPOINTS:
    fp = PER / f'{ep}__clean.tsv'
    if not fp.exists(): continue

    train_rows, demo_rows = [], []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: y = float(r['value'])
            except: continue
            if cid in DEMO_EXCLUDE: demo_rows.append((cid, y, seq, r))
            else: train_rows.append((cid, y, seq, r))

    if len(train_rows) < 50:
        print(f"\n[{ep}] {len(train_rows)} rows < 50, skip", flush=True)
        continue

    unit = train_rows[0][3].get('unit_target', '')

    # Get feature list: Optuna best if available, fallback otherwise
    if ep in OPTUNA_BEST:
        feat_list = OPTUNA_BEST[ep]['features']
        optuna_sp = OPTUNA_BEST[ep]['sp']
        source = "Optuna"
    else:
        feat_list = FALLBACK_COMBOS.get(ep, ['morgan', 'esm2', 'desc'])
        optuna_sp = None
        source = "fallback"

    print(f"\n[{ep}] {len(train_rows):,} rows | {source} | features={'+'.join(feat_list)} | {unit}", flush=True)

    # Build features
    X, y, groups = build_features(train_rows, feat_list)
    print(f"  X shape: {X.shape}", flush=True)

    # Quick CV with LGBM (fast, reliable)
    gkf = GroupKFold(n_splits=5)
    cv_pred = np.zeros_like(y)
    for tr, te in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              random_state=42, verbose=-1, n_jobs=32)
        m.fit(X[tr], y[tr])
        cv_pred[te] = m.predict(X[te])

    sp = float(spearmanr(y, cv_pred).statistic)
    mae = float(np.mean(np.abs(y - cv_pred)))
    print(f"  LGBM CV: Sp={sp:.3f} MAE={mae:.3f}" + (f" (Optuna best: {optuna_sp:.3f})" if optuna_sp else ""), flush=True)

    # Train production model
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              random_state=42, verbose=-1, n_jobs=32)
    model.fit(X, y)
    joblib.dump({'model': model, 'endpoint': ep, 'features': feat_list,
                 'cv_spearman': sp, 'unit': unit, 'n_train': len(X)},
                OUT / 'models' / f'{ep}.joblib')

    # Demo prediction
    if demo_rows:
        dX, _, _ = build_features(demo_rows, feat_list)
        if dX is not None:
            preds = model.predict(dX)
            for j, (cid, act, seq, r) in enumerate(demo_rows):
                pred = float(preds[j])
                demo_preds.append([DEMO_NAMES.get(cid, cid), ep,
                                  f'{act:.3g}', f'{pred:.3g}', unit,
                                  f'{sp:.3f}', len(X)])
                print(f"  → {DEMO_NAMES.get(cid, cid)}: pred={pred:.3g} | actual={act:.3g}", flush=True)

    results.append([ep, sp, mae, len(X), '+'.join(feat_list), source])

# Save
with open(OUT / 'all_results.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['endpoint','spearman','mae','n_train','features','source'])
    w.writerows(results)

with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(demo_preds)

with open(OUT / 'final_report.md', 'w', encoding='utf-8') as f:
    f.write(f"# 最终生产模型报告\n\n")
    f.write(f"> 3 个端点用 Optuna 最优配置 | 其余用经验配置 + LGBM\n\n")
    f.write("## 各端点性能\n\n| Endpoint | Sp | MAE | Train N | Features | Source |\n|---|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r[0]} | **{r[1]:.3f}** | {r[2]:.3f} | {r[3]:,} | {r[4]} | {r[5]} |\n")
    f.write("\n## 司美格鲁肽/替尔泊肽预测\n\n| Drug | Endpoint | Actual | Predicted | Unit | CV Sp |\n|---|---|---|---|---|---|\n")
    for r in demo_preds:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[4]} | {r[5]} |\n")

print(f"\n{'='*60}")
print(f"COMPLETE: {len(results)} endpoints, {len(demo_preds)} demo predictions")
print(f"Elapsed: {(time.time()-t0)/60:.0f} minutes")
