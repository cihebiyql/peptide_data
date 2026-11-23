#!/usr/bin/env python3
"""Full systematic optimization sweep on node23 (64 cores).

For each endpoint × feature_set × model: GroupKFold-5, report Sp/MAE.
Feature sets: morgan+desc, esm+desc, ncaa+desc, morgan+esm+desc, ncaa+morgan+desc
Models: Ridge, LGBM (4 param combos)
Output: optimization_matrix.tsv + best_config_per_endpoint.tsv + production models
"""
import csv, sys, time, joblib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

csv.field_size_limit(10_000_000)
DATA = Path('<WORK_DIR>/s3')
PER = Path('<WORK_DIR>/s3_per')
OUT = DATA / 'optimization'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
DESC_COLS = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']
MIN_ROWS = 50

t0 = time.time()

# ---- Load all features ----
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

# ncAA (both sets)
zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
ncaa1 = zn1['embeddings'].astype(np.float32)
ncaa1_idx = {}
with open(DATA / 'ncaa_set1_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        ncaa1_idx[r['compound_id']] = int(r['row_idx'])

zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
ncaa2 = zn2['embeddings'].astype(np.float32)
ncaa2_idx = {}
with open(DATA / 'ncaa_mod_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        ncaa2_idx[r['compound_id']] = int(r['row_idx'])

# Descriptors
desc_map = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_map[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in DESC_COLS], dtype=np.float32)

print(f'Features: morgan={mor.shape}, esm={esm.shape}, ncaa1={ncaa1.shape}, ncaa2={ncaa2.shape} ({time.time()-t0:.1f}s)', flush=True)

DEMO_EXCLUDE = {
    'cmp_bccb3404f692c7bf44452c96',  # semaglutide
    'cmp_c80b5e5a1682bb35cd28ec5b',  # tirzepatide
}

def get_ncaa(cid):
    if cid in ncaa2_idx: return ncaa2[ncaa2_idx[cid]]
    if cid in ncaa1_idx: return ncaa1[ncaa1_idx[cid]]
    return None

# ---- Define feature builders ----
def build_features(train_rows):
    """Build 5 feature sets for training rows."""
    sets = {}
    zero_desc = np.zeros(11, dtype=np.float32)

    # 1. morgan+desc
    ids_m = [(r[0], mor_idx[r[0]]) for r in train_rows if r[0] in mor_idx]
    if len(ids_m) >= MIN_ROWS:
        d = np.stack([desc_map.get(train_rows[i][0], zero_desc) for i in range(len(train_rows)) if train_rows[i][0] in mor_idx])
        sets['morgan+desc'] = (np.hstack([mor[[i for _, i in ids_m]], d]),
                               np.array([train_rows[i][1] for i in range(len(train_rows)) if train_rows[i][0] in mor_idx]),
                               np.array([train_rows[i][0] for i in range(len(train_rows)) if train_rows[i][0] in mor_idx]))

    # 2. esm+desc
    ids_e = [(i, esm_idx[r[2]]) for i, r in enumerate(train_rows) if r[2] and r[2] in esm_idx]
    if len(ids_e) >= MIN_ROWS:
        e = esm[[i for _, i in ids_e]]
        d = np.stack([desc_map.get(train_rows[i][0], zero_desc) for i, _ in ids_e])
        sets['esm+desc'] = (np.hstack([e, d]),
                           np.array([train_rows[i][1] for i, _ in ids_e]),
                           np.array([train_rows[i][0] for i, _ in ids_e]))

    # 3. ncaa+desc
    ids_n = [(i, r[0]) for i, r in enumerate(train_rows) if get_ncaa(r[0]) is not None]
    if len(ids_n) >= MIN_ROWS:
        n = np.stack([get_ncaa(train_rows[i][0]) for i, _ in ids_n])
        d = np.stack([desc_map.get(train_rows[i][0], zero_desc) for i, _ in ids_n])
        sets['ncaa+desc'] = (np.hstack([n, d]),
                            np.array([train_rows[i][1] for i, _ in ids_n]),
                            np.array([train_rows[i][0] for i, _ in ids_n]))

    # 4. morgan+esm+desc (rows with BOTH morgan AND esm)
    ids_me = []
    for i, r in enumerate(train_rows):
        if r[0] in mor_idx and r[2] and r[2] in esm_idx:
            ids_me.append(i)
    if len(ids_me) >= MIN_ROWS:
        m = mor[[mor_idx[train_rows[i][0]] for i in ids_me]]
        e = esm[[esm_idx[train_rows[i][2]] for i in ids_me]]
        d = np.stack([desc_map.get(train_rows[i][0], zero_desc) for i in ids_me])
        sets['morgan+esm+desc'] = (np.hstack([m, e, d]),
                                  np.array([train_rows[i][1] for i in ids_me]),
                                  np.array([train_rows[i][0] for i in ids_me]))

    # 5. ncaa+morgan+desc
    ids_nm = []
    for i, r in enumerate(train_rows):
        if r[0] in mor_idx and get_ncaa(r[0]) is not None:
            ids_nm.append(i)
    if len(ids_nm) >= MIN_ROWS:
        n = np.stack([get_ncaa(train_rows[i][0]) for i in ids_nm])
        m = mor[[mor_idx[train_rows[i][0]] for i in ids_nm]]
        d = np.stack([desc_map.get(train_rows[i][0], zero_desc) for i in ids_nm])
        sets['ncaa+morgan+desc'] = (np.hstack([n, m, d]),
                                   np.array([train_rows[i][1] for i in ids_nm]),
                                   np.array([train_rows[i][0] for i in ids_nm]))
    return sets


# ---- Hyperparameter grids ----
LGBM_GRID = [
    dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=20),
    dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20),
    dict(n_estimators=600, learning_rate=0.08, num_leaves=63, min_child_samples=15),
    dict(n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=10),
]
RIDGE_GRID = [dict(alpha=1.0), dict(alpha=10.0), dict(alpha=50.0)]


def evaluate(X, y, groups):
    """Run all models, return list of (model_name, params, sp, mae, dir)."""
    gkf = GroupKFold(n_splits=5)
    try:
        splits = list(gkf.split(X, y, groups))
    except ValueError:
        return []

    results = []
    dy = y - y.mean()

    # Ridge variants
    for params in RIDGE_GRID:
        pred = np.zeros_like(y)
        for tr, te in splits:
            sc = StandardScaler(with_mean=False).fit(X[tr])
            pred[te] = Ridge(**params).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        sp = float(spearmanr(y, pred).statistic)
        mae = float(np.mean(np.abs(y - pred)))
        dp = pred - pred.mean()
        mask = (np.abs(dy) > 1e-9) & (np.abs(dp) > 1e-9)
        dirr = float(np.mean(np.sign(dy[mask]) == np.sign(dp[mask]))) if mask.sum() > 10 else float('nan')
        results.append((f"ridge(a={params['alpha']})", json.dumps(params), sp, mae, dirr))

    # LGBM variants
    for params in LGBM_GRID:
        full = dict(**params, subsample=0.9, colsample_bytree=0.9,
                    random_state=42, verbose=-1, n_jobs=32)
        pred = np.zeros_like(y)
        for tr, te in splits:
            m = lgb.LGBMRegressor(**full)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        sp = float(spearmanr(y, pred).statistic)
        mae = float(np.mean(np.abs(y - pred)))
        dp = pred - pred.mean()
        mask = (np.abs(dy) > 1e-9) & (np.abs(dp) > 1e-9)
        dirr = float(np.mean(np.sign(dy[mask]) == np.sign(dp[mask]))) if mask.sum() > 10 else float('nan')
        results.append((f"lgbm(ne={params['n_estimators']},lr={params['learning_rate']},lv={params['num_leaves']})",
                        json.dumps(full), sp, mae, dirr))
    return results


# ---- Main sweep ----
all_results = []
best_configs = {}

for fp in sorted(PER.glob('*__clean.tsv')):
    ep = fp.stem.replace('__clean', '')
    train_rows = []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            if cid in DEMO_EXCLUDE: continue
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: val = float(r['value'])
            except: continue
            train_rows.append((cid, val, seq))
    if len(train_rows) < MIN_ROWS:
        continue

    unit = ''
    with open(fp, encoding='utf-8') as f:
        rd = csv.DictReader(f, delimiter='\t')
        unit = next(iter(rd), {}).get('unit_target', '')

    feat_sets = build_features(train_rows)
    if not feat_sets:
        continue

    ep_best_sp, ep_best = -1, None
    print(f'\n[{ep}] {len(train_rows):,} rows | {len(feat_sets)} feature sets | {unit}', flush=True)

    for fs_name, (X, y, groups) in feat_sets.items():
        model_results = evaluate(X, y, groups)
        if not model_results:
            continue
        for model_name, params_json, sp, mae, dirr in model_results:
            all_results.append([ep, fs_name, model_name, params_json, len(X),
                               f'{sp:.4f}', f'{mae:.4f}', f'{dirr:.4f}'])
            if sp > ep_best_sp:
                ep_best_sp = sp
                ep_best = (fs_name, model_name, params_json, X, y, groups)
        best_in_fs = max(model_results, key=lambda x: x[2])
        print(f'  {fs_name:<22} {best_in_fs[0]:<35} Sp={best_in_fs[2]:.3f} MAE={best_in_fs[3]:.3f} ({len(X)} rows)', flush=True)

    # Train production model with best config
    if ep_best:
        fs_name, model_name, params_json, X, y, groups = ep_best
        params = json.loads(params_json)
        if 'ridge' in model_name:
            model = Ridge(**params)
            scaler = StandardScaler(with_mean=False).fit(X)
            model.fit(scaler.transform(X), y)
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            scaler = None
        joblib.dump({'model': model, 'scaler': scaler, 'endpoint': ep,
                     'feature_set': fs_name, 'model_name': model_name,
                     'params': params, 'cv_spearman': ep_best_sp,
                     'n_train': len(X), 'unit': unit},
                    OUT / 'models' / f'{ep}.joblib')
        best_configs[ep] = {
            'feature_set': fs_name, 'model': model_name,
            'cv_spearman': round(ep_best_sp, 4), 'n_train': len(X),
            'unit': unit, 'dim': X.shape[1]
        }

# ---- Output ----
with open(OUT / 'optimization_matrix.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['endpoint', 'feature_set', 'model', 'params', 'rows', 'Spearman', 'MAE', 'DirAgree'])
    w.writerows(all_results)

with open(OUT / 'best_configs.json', 'w', encoding='utf-8') as f:
    json.dump(best_configs, f, ensure_ascii=False, indent=2)

with open(OUT / 'optimization_report.md', 'w', encoding='utf-8') as f:
    f.write('# 全端点系统优化扫描\n\n')
    f.write(f'特征集: morgan+desc / esm+desc / ncaa+desc / morgan+esm+desc / ncaa+morgan+desc\n')
    f.write(f'模型: Ridge(3α) + LGBM(4组参数) = 7 配置 × 5 特征集 = 35 组/端点\n\n')
    f.write('| 端点 | 最优特征 | 最优模型 | CV Sp | MAE | 维度 | 训练行 |\n')
    f.write('|---|---|---|---|---|---|---|\n')
    for ep, cfg in sorted(best_configs.items(), key=lambda x: -x[1]['cv_spearman']):
        f.write(f"| {ep} | {cfg['feature_set']} | {cfg['model']} | **{cfg['cv_spearman']}** | — | {cfg['dim']} | {cfg['n_train']:,} |\n")

print(f'\n{"="*70}')
print(f'[done] {len(all_results)} total configs evaluated across {len(best_configs)} endpoints')
print(f'[elapsed] {time.time()-t0:.0f}s')
print(f'[out] {OUT}')
print(f'\nBest configs:')
for ep, cfg in sorted(best_configs.items(), key=lambda x: -x[1]['cv_spearman']):
    print(f'  {ep:<22} {cfg["feature_set"]:<22} {cfg["model"]:<35} Sp={cfg["cv_spearman"]}')
