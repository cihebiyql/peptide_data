#!/usr/bin/env python3
"""Optuna AutoML pipeline: automated feature selection + model selection + hyperparameter tuning.

For each endpoint: Optuna searches over (modality subsets × model type × hyperparams).
Features: Morgan / ESM2 / ncAA-RepDistill / RDKit descriptors / modification features
Models: Ridge / LightGBM
Evaluation: GroupKFold-5 (by compound_id for leakage control)
Output: best model per endpoint + full search history + demo drug predictions
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

DATA = Path('<WORK_DIR>/s3')
PER = Path('<WORK_DIR>/s3_per')
OUT = DATA / 'optuna_results'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
DESC_COLS = ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']
MIN_ROWS = 50
N_TRIALS = 150
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}

t0 = time.time()

# ---- Load features ----
print('Loading features...', flush=True)

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
        desc_map[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in DESC_COLS], dtype=np.float32)

print(f'Features: morgan={MOR.shape}, esm={ESM.shape}, ncaa={NCAA.shape} ({time.time()-t0:.1f}s)', flush=True)


def get_ncaa(cid):
    i = ncaa_idx.get(cid)
    return NCAA[i] if i is not None else None


def build_endpoint_data(fp):
    """Read per-endpoint table, return train rows and demo rows."""
    train_rows, demo_rows = [], []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: val = float(r['value'])
            except: continue
            if cid in DEMO_EXCLUDE:
                demo_rows.append((cid, val, seq, r))
            else:
                train_rows.append((cid, val, seq, r))
    return train_rows, demo_rows


def assemble_features(train_rows, config):
    """Build feature matrix based on Optuna config (which modalities to use)."""
    parts = []
    zero_d = np.zeros(11, dtype=np.float32)
    valid = []

    for i, (cid, y, seq, r) in enumerate(train_rows):
        row_parts = []
        ok = True
        if config['use_morgan']:
            if cid in mor_idx:
                row_parts.append(MOR[mor_idx[cid]])
            else:
                ok = False
        if config['use_esm']:
            if seq and seq in esm_idx:
                row_parts.append(ESM[esm_idx[seq]])
            else:
                ok = False
        if config['use_ncaa']:
            n = get_ncaa(cid)
            if n is not None:
                row_parts.append(n)
            else:
                ok = False
        if config['use_desc']:
            row_parts.append(desc_map.get(cid, zero_d))
        if ok:
            valid.append((i, np.concatenate(row_parts)))

    if len(valid) < MIN_ROWS:
        return None, None, None
    X = np.stack([v[1] for v in valid])
    y = np.array([train_rows[v[0]][1] for v in valid])
    groups = np.array([train_rows[v[0]][0] for v in valid])
    return X, y, groups


def make_objective(train_rows):
    """Create Optuna objective function for this endpoint."""

    def objective(trial):
        # Feature modality selection
        config = {
            'use_morgan': trial.suggest_categorical('use_morgan', [True, False]),
            'use_esm': trial.suggest_categorical('use_esm', [True, False]),
            'use_ncaa': trial.suggest_categorical('use_ncaa', [True, False]),
            'use_desc': trial.suggest_categorical('use_desc', [True, False]),
        }
        # At least one modality
        if not any([config['use_morgan'], config['use_esm'], config['use_ncaa'], config['use_desc']]):
            return -1.0

        # Model selection
        model_type = trial.suggest_categorical('model', ['ridge', 'lgbm'])

        if model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('lgbm_ne', 100, 1000),
                'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('lgbm_lv', 15, 255),
                'min_child_samples': trial.suggest_int('lgbm_mc', 5, 50),
                'reg_alpha': trial.suggest_float('lgbm_la', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('lgbm_lb', 1e-3, 10, log=True),
                'subsample': 0.9,
                'colsample_bytree': trial.suggest_float('lgbm_ct', 0.5, 1.0),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': 32,
            }
        else:
            params = {'alpha': trial.suggest_float('ridge_alpha', 0.1, 100, log=True)}

        # Build features
        X, y, groups = assemble_features(train_rows, config)
        if X is None:
            return -1.0

        # GroupKFold CV
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
        if np.isnan(sp):
            return -1.0
        return sp

    return objective


def predict_demo(model_type, params, X, y, demo_rows, config):
    """Train final model and predict for demo drugs."""
    if model_type == 'lgbm':
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        scaler = None
    else:
        scaler = StandardScaler(with_mean=False).fit(X)
        model = Ridge(**params)
        model.fit(scaler.transform(X), y)

    preds = {}
    zero_d = np.zeros(11, dtype=np.float32)
    for cid, val, seq, r in demo_rows:
        parts = []
        ok = True
        if config['use_morgan']:
            if cid in mor_idx:
                parts.append(MOR[mor_idx[cid]])
            else:
                ok = False
        if config['use_esm']:
            if seq and seq in esm_idx:
                parts.append(ESM[esm_idx[seq]])
            else:
                ok = False
        if config['use_ncaa']:
            n = get_ncaa(cid)
            if n is not None:
                parts.append(n)
            else:
                ok = False
        if config['use_desc']:
            parts.append(desc_map.get(cid, zero_d))
        if ok and parts:
            feat = np.concatenate(parts).reshape(1, -1)
            if scaler:
                feat = scaler.transform(feat)
            preds[cid] = float(model.predict(feat)[0])
    return preds


# ---- Main pipeline ----
all_results = []
demo_predictions = []

SKIP_DONE = {'BBB', 'CC50', 'CL', 'EC50', 'IC50'}  # 已完成的端点跳过

for fp in sorted(PER.glob('*__clean.tsv')):
    ep = fp.stem.replace('__clean', '')
    if ep in SKIP_DONE:
        continue
    train_rows, demo_rows = read_data = build_endpoint_data(fp)
    if len(train_rows) < MIN_ROWS:
        continue

    unit = train_rows[0][3].get('unit_target', '') if train_rows else ''
    print(f'\n{"="*60}')
    print(f'[{ep}] {len(train_rows):,} train rows | {len(demo_rows)} demo | unit={unit}', flush=True)

    # Optuna search
    objective = make_objective(train_rows)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, timeout=600)  # max 10 min per endpoint

    best = study.best_trial
    best_sp = best.value
    print(f'  Best Sp={best_sp:.4f} (trial {best.number}, {time.time()-t0:.0f}s)', flush=True)

    # Extract best config
    cfg = best.params
    model_type = cfg['model']
    feat_config = {
        'use_morgan': cfg.get('use_morgan', False),
        'use_esm': cfg.get('use_esm', False),
        'use_ncaa': cfg.get('use_ncaa', False),
        'use_desc': cfg.get('use_desc', False),
    }
    feat_str = '+'.join(k.replace('use_', '') for k, v in feat_config.items() if v)
    print(f'  Features: {feat_str}', flush=True)
    print(f'  Model: {model_type} | Params: { {k:v for k,v in cfg.items() if k not in ("model","use_morgan","use_esm","use_ncaa","use_desc")} }', flush=True)

    # Rebuild features with best config
    X, y, groups = assemble_features(train_rows, feat_config)
    if X is None:
        continue

    # Extract model params with CORRECT LightGBM parameter names
    if model_type == 'lgbm':
        mp = {}
        for k, v in cfg.items():
            if k == 'lgbm_ne': mp['n_estimators'] = int(v)
            elif k == 'lgbm_lr': mp['learning_rate'] = float(v)
            elif k == 'lgbm_lv': mp['num_leaves'] = int(v)
            elif k == 'lgbm_mc': mp['min_child_samples'] = int(v)
            elif k == 'lgbm_la': mp['reg_alpha'] = float(v)
            elif k == 'lgbm_lb': mp['reg_lambda'] = float(v)
            elif k == 'lgbm_ct': mp['colsample_bytree'] = float(v)
        mp.update(subsample=0.9, random_state=42, verbose=-1, n_jobs=32)
    else:
        mp = {'alpha': cfg.get('ridge_alpha', 10.0)}

    # Train production model
    if model_type == 'lgbm':
        model = lgb.LGBMRegressor(**mp)
        model.fit(X, y)
        scaler = None
    else:
        scaler = StandardScaler(with_mean=False).fit(X)
        model = Ridge(**mp)
        model.fit(scaler.transform(X), y)

    joblib.dump({'model': model, 'scaler': scaler, 'endpoint': ep,
                 'feature_config': feat_config, 'model_type': model_type,
                 'model_params': mp, 'cv_spearman': best_sp,
                 'n_train': len(X), 'unit': unit, 'dim': X.shape[1],
                 'n_trials': len(study.trials)},
                OUT / 'models' / f'{ep}.joblib')

    # Predict demo drugs (with error protection)
    try:
        demo_preds = predict_demo(model_type, mp, X, y, demo_rows, feat_config)
    except Exception as e:
        print(f'  ⚠ Demo prediction failed: {e}', flush=True)
        demo_preds = {}
    for cid, pred in demo_preds.items():
        actuals = [r[1] for r in demo_rows if r[0] == cid]
        act = float(np.median(actuals)) if actuals else None
        demo_predictions.append([DEMO_NAMES.get(cid, cid), ep,
                                f'{act:.3g}' if act is not None else '—',
                                f'{pred:.3g}', unit, f'{best_sp:.4f}', len(X)])
        print(f'  → {DEMO_NAMES.get(cid, cid)}: pred={pred:.3g} | actual={act:.3g}' if act else
              f'  → {DEMO_NAMES.get(cid, cid)}: pred={pred:.3g} | no actual', flush=True)

    # Record top 10 trials
    for tr in study.trials[:10]:
        if tr.state == optuna.trial.TrialState.COMPLETE and tr.value is not None:
            all_results.append([ep, tr.number, tr.value,
                                json.dumps({k: str(v) for k, v in tr.params.items()}),
                                len(study.trials)])

# ---- Final output ----
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug', 'endpoint', 'actual', 'predicted', 'unit', 'cv_spearman', 'n_train'])
    w.writerows(demo_predictions)

with open(OUT / 'search_history.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['endpoint', 'trial', 'spearman', 'params', 'total_trials'])
    w.writerows(all_results)

with open(OUT / 'optuna_report.md', 'w', encoding='utf-8') as f:
    f.write('# Optuna AutoML 全端点优化报告\n\n')
    f.write(f'> {N_TRIALS} trials/endpoint | TPE采样 | 中位数剪枝 | GroupKFold-5\n\n')
    f.write('## Demo Drug Predictions\n\n| Drug | Endpoint | Actual | Predicted | Unit | CV Sp | Train N |\n|---|---|---|---|---|---|---|\n')
    for r in demo_predictions:
        f.write(f'| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[4]} | {r[5]} | {r[6]:,} |\n')

print(f'\n{"="*60}')
print(f'[DONE] {len(demo_predictions)} predictions across all endpoints')
print(f'[elapsed] {time.time()-t0:.0f}s total')
print(f'[out] {OUT}')
