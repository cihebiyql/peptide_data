#!/usr/bin/env python3
"""Expanded Optuna: 10 feature groups (original 4 + new 6 from peptide binding project).

Original: morgan / esm / ncaa / desc
New:      aac(26) / physchem(10) / dbaasp(12) / mod(9) / maccs(166) / chembl(6)
Run after current pipeline finishes.
"""
import csv, sys, time, joblib, json, warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
csv.field_size_limit(10_000_000)

DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_per')
OUT = DATA / 'optuna_expanded'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 50
N_TRIALS = 100
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}

t0 = time.time()

# ---- Load ALL features ----
print('Loading features...', flush=True)

# Original features
zm = np.load(DATA / 'morgan_fp16.npz')
MOR = zm['embeddings'].astype(np.float32)
mor_idx = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        mor_idx[r['compound_id']] = int(r['row_idx'])

ze = np.load(DATA / 'esm2_fp16.npz')
ESM = ze['embeddings'].astype(np.float32)
esm_idx = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        esm_idx[r['sequence']] = int(r['row_idx'])

zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
NC1 = zn1['embeddings'].astype(np.float32)
ncaa_idx = {}
with open(DATA / 'ncaa_set1_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        ncaa_idx[r['compound_id']] = int(r['row_idx'])
zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
NC2 = zn2['embeddings'].astype(np.float32)
with open(DATA / 'ncaa_mod_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        ncaa_idx.setdefault(r['compound_id'], int(r['row_idx']) + NC1.shape[0])
NCAA = np.vstack([NC1, NC2])

desc_map = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_map[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in
            ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']],
            dtype=np.float32)

# NEW: all_features (229-dim: aac 26 + physchem 10 + dbaasp 12 + mod 9 + maccs 166 + chembl 6)
za = np.load(DATA / 'all_features_fp16.npz')
ALLF = za['embeddings'].astype(np.float32)
allf_idx = {}
with open(DATA / 'all_features_index.tsv') as f:
    rd = csv.DictReader(f, delimiter='\t')
    for r in rd:
        allf_idx[r['compound_id']] = int(r['row_idx'])

# Feature group slices within ALLF (229 dims)
SLICES = {
    'aac':      (0, 26),     # amino acid composition + grouped
    'physchem': (26, 36),    # GRAVY, pI, charge, hydrophobic moment, etc.
    'dbaasp':   (36, 48),    # DBAASP 12 peptide-specific properties
    'mod':      (48, 57),    # modification counts
    'maccs':    (57, 223),   # MACCS keys 166-bit
    'chembl':   (223, 229),  # ChEMBL properties 6
}

print(f'All loaded: morgan={MOR.shape}, esm={ESM.shape}, ncaa={NCAA.shape}, allfeat={ALLF.shape} ({time.time()-t0:.1f}s)', flush=True)


def assemble(train_rows, config):
    """Build feature matrix from 10 modality flags."""
    valid = []
    for i, (cid, y, seq, r) in enumerate(train_rows):
        parts = []
        ok = True
        if config.get('use_morgan'):
            if cid in mor_idx: parts.append(MOR[mor_idx[cid]])
            else: ok = False
        if config.get('use_esm'):
            if seq and seq in esm_idx: parts.append(ESM[esm_idx[seq]])
            else: ok = False
        if config.get('use_ncaa'):
            ni = ncaa_idx.get(cid)
            if ni is not None: parts.append(NCAA[ni])
            else: ok = False
        if config.get('use_desc'):
            parts.append(desc_map.get(cid, np.zeros(11, dtype=np.float32)))
        # NEW feature groups from all_features
        ai = allf_idx.get(cid)
        if ai is not None:
            if config.get('use_aac'):
                s, e = SLICES['aac']; parts.append(ALLF[ai, s:e])
            if config.get('use_physchem'):
                s, e = SLICES['physchem']; parts.append(ALLF[ai, s:e])
            if config.get('use_dbaasp'):
                s, e = SLICES['dbaasp']; parts.append(ALLF[ai, s:e])
            if config.get('use_mod'):
                s, e = SLICES['mod']; parts.append(ALLF[ai, s:e])
            if config.get('use_maccs'):
                s, e = SLICES['maccs']; parts.append(ALLF[ai, s:e])
            if config.get('use_chembl'):
                s, e = SLICES['chembl']; parts.append(ALLF[ai, s:e])
        else:
            # If any new feature is requested but compound not in all_features
            if any(config.get(f'use_{k}') for k in SLICES):
                ok = False
        if ok and parts:
            valid.append((i, np.concatenate(parts)))
    if len(valid) < MIN_ROWS:
        return None, None, None
    X = np.stack([v[1] for v in valid])
    y = np.array([train_rows[v[0]][1] for v in valid])
    groups = np.array([train_rows[v[0]][0] for v in valid])
    return X, y, groups


def make_objective(train_rows):
    def objective(trial):
        config = {
            'use_morgan': trial.suggest_categorical('morgan', [True, False]),
            'use_esm': trial.suggest_categorical('esm', [True, False]),
            'use_ncaa': trial.suggest_categorical('ncaa', [True, False]),
            'use_desc': trial.suggest_categorical('desc', [True, False]),
            'use_aac': trial.suggest_categorical('aac', [True, False]),
            'use_physchem': trial.suggest_categorical('physchem', [True, False]),
            'use_dbaasp': trial.suggest_categorical('dbaasp', [True, False]),
            'use_mod': trial.suggest_categorical('mod', [True, False]),
            'use_maccs': trial.suggest_categorical('maccs', [True, False]),
            'use_chembl': trial.suggest_categorical('chembl', [True, False]),
        }
        if not any(config.values()):
            return -1.0

        model_type = trial.suggest_categorical('model', ['ridge', 'lgbm'])
        if model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('lgbm_ne', 100, 800),
                'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('lgbm_lv', 15, 200),
                'min_child_samples': trial.suggest_int('lgbm_mc', 5, 40),
                'subsample': 0.9,
                'colsample_bytree': trial.suggest_float('lgbm_ct', 0.5, 1.0),
                'random_state': 42, 'verbose': -1, 'n_jobs': 32,
            }
        else:
            params = {'alpha': trial.suggest_float('ridge_alpha', 0.1, 100, log=True)}

        X, y, groups = assemble(train_rows, config)
        if X is None:
            return -1.0

        gkf = GroupKFold(n_splits=5)
        try:
            splits = list(gkf.split(X, y, groups))
        except ValueError:
            return -1.0

        preds = np.zeros_like(y)
        for tr, te in splits:
            if model_type == 'lgbm':
                m = lgb.LGBMRegressor(**params)
                m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            else:
                sc = StandardScaler(with_mean=False).fit(X[tr])
                m = Ridge(**params)
                m.fit(sc.transform(X[tr]), y[tr])
                preds[te] = m.predict(sc.transform(X[te]))

        sp = float(spearmanr(y, preds).statistic)
        return sp if not np.isnan(sp) else -1.0
    return objective


# ---- Main ----
KEY_ENDPOINTS = ['half_life', 'EC50', 'IC50', 'hemolysis', 'MIC', 'MBC', 'Ki', 'Kd', 'LC_lemma']
results = []
demo_predictions = []

for ep_name in KEY_ENDPOINTS:
    fp = PER / f'{ep_name}__clean.tsv'
    if not fp.exists(): continue
    train_rows, demo_rows = [], []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: y = float(r['value'])
            except: continue
            if cid in DEMO_EXCLUDE:
                demo_rows.append((cid, y, seq, r))
            else:
                train_rows.append((cid, y, seq, r))
    if len(train_rows) < MIN_ROWS: continue

    unit = train_rows[0][3].get('unit_target', '') if train_rows else ''
    print(f'\n{"="*60}')
    print(f'[{ep_name}] {len(train_rows):,} rows | unit={unit}', flush=True)

    objective = make_objective(train_rows)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=600)

    best = study.best_trial
    print(f'  Best Sp={best.value:.4f} (trial {best.number}, {time.time()-t0:.0f}s)', flush=True)
    cfg = best.params
    feat_used = [k for k in ['morgan','esm','ncaa','desc','aac','physchem','dbaasp','mod','maccs','chembl'] if cfg.get(k)]
    print(f'  Features: {"+".join(feat_used)}', flush=True)
    print(f'  Model: {cfg["model"]}', flush=True)

    # Save
    joblib.dump({'endpoint': ep_name, 'best_params': cfg,
                 'best_spearman': best.value, 'n_trials': len(study.trials),
                 'features_used': feat_used},
                OUT / 'models' / f'{ep_name}.joblib')

    # Demo prediction
    feat_config = {f'use_{k}': cfg.get(k, False) for k in
                   ['morgan','esm','ncaa','desc','aac','physchem','dbaasp','mod','maccs','chembl']}
    X, y, groups = assemble(train_rows, feat_config)
    if X is not None:
        if cfg['model'] == 'lgbm':
            mp = {'n_estimators': int(cfg['lgbm_ne']), 'learning_rate': float(cfg['lgbm_lr']),
                  'num_leaves': int(cfg['lgbm_lv']), 'min_child_samples': int(cfg['lgbm_mc']),
                  'subsample': 0.9, 'colsample_bytree': float(cfg['lgbm_ct']),
                  'random_state': 42, 'verbose': -1, 'n_jobs': 32}
            model = lgb.LGBMRegressor(**mp); model.fit(X, y)
        else:
            mp = {'alpha': float(cfg['ridge_alpha'])}
            sc = StandardScaler(with_mean=False).fit(X)
            model = Ridge(**mp); model.fit(sc.transform(X), y)

        for cid, act, seq, r in demo_rows:
            parts = []
            ok = True
            if feat_config.get('use_morgan') and cid in mor_idx: parts.append(MOR[mor_idx[cid]])
            elif feat_config.get('use_morgan'): ok = False
            if feat_config.get('use_esm') and seq in esm_idx: parts.append(ESM[esm_idx[seq]])
            elif feat_config.get('use_esm'): ok = False
            if feat_config.get('use_ncaa') and cid in ncaa_idx: parts.append(NCAA[ncaa_idx[cid]])
            elif feat_config.get('use_ncaa'): ok = False
            if feat_config.get('use_desc'): parts.append(desc_map.get(cid, np.zeros(11, dtype=np.float32)))
            ai = allf_idx.get(cid)
            if ai is not None:
                for gname in SLICES:
                    if feat_config.get(f'use_{gname}'):
                        s, e = SLICES[gname]; parts.append(ALLF[ai, s:e])
            if ok and parts:
                feat = np.concatenate(parts).reshape(1, -1)
                pred = float(model.predict(feat)[0])
                demo_predictions.append([DEMO_NAMES.get(cid, cid), ep_name,
                                        f'{act:.3g}', f'{pred:.3g}', unit, f'{best.value:.4f}', len(X)])
                print(f'  → {DEMO_NAMES.get(cid, cid)}: pred={pred:.3g} | actual={act:.3g}', flush=True)

with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(demo_predictions)

print(f'\n{"="*60}')
print(f'[DONE] {len(demo_predictions)} predictions, {time.time()-t0:.0f}s total')
