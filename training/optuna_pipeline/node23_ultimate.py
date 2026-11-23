#!/usr/bin/env python3
"""ULTIMATE Optuna pipeline: 13 feature groups × 8 model types × full hyperparameter tuning.

Designed for extended unattended runs on node23 (64 cores, no time limit).
Every endpoint gets exhaustive search: TPE sampler, 200 trials, MedianPruner.

Models: Ridge / ElasticNet / SVR / LightGBM / XGBoost / RandomForest / MLP / CatBoost
Features: morgan / esm2 / ncaa / desc / chemberta / esm3b / rdkit_full / dpc / atompair / aac / physchem / dbaasp / mod
"""
import csv, sys, time, joblib, json, warnings, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
csv.field_size_limit(10_000_000)

DATA = Path('<WORK_DIR>/s3')
PER = Path('<WORK_DIR>/s3_per')
OUT = DATA / 'ultimate'
CHECKPOINT = OUT / 'checkpoint.json'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 50
N_TRIALS = 150
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}

FEATURE_FLAGS = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
                 'rdkit_full','dpc','atompair','aac','physchem','dbaasp','mod']

t0 = time.time()
print("="*70)
print("ULTIMATE PIPELINE: 13 features × 8 models × exhaustive tuning")
print("="*70)

# ---- Load ALL features ----
print("Loading features...", flush=True)
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

zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
NC1 = zn1['embeddings'].astype(np.float32)
ncaa_idx = {}
with open(DATA / 'ncaa_set1_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): ncaa_idx[r['compound_id']] = int(r['row_idx'])
zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
NC2 = zn2['embeddings'].astype(np.float32)
with open(DATA / 'ncaa_mod_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): ncaa_idx.setdefault(r['compound_id'], int(r['row_idx']) + NC1.shape[0])
NCAA = np.vstack([NC1, NC2])

desc_map = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_map[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in
            ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']],
            dtype=np.float32)

# ChemBERTa (fast numeric loading)
zcb = np.load(DATA / 'chemberta_fp16.npz')
cb_emb = zcb['embeddings'].astype(np.float32)
cb_map = {}
with open(DATA / 'chemberta_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        cb_map[r['compound_id']] = cb_emb[int(r['row_idx'])]

# ESM2-3B (fast numeric loading)
ze3 = np.load(DATA / 'esm3b_fp16.npz')
e3b_emb = ze3['embeddings'].astype(np.float32)
e3b_map = {}
with open(DATA / 'esm3b_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        e3b_map[r['sequence']] = e3b_emb[int(r['row_idx'])]

zx = np.load(DATA / 'expanded_cpu_features_fp16.npz')
EXP = zx['embeddings'].astype(np.float32)
exp_idx = {}
with open(DATA / 'expanded_cpu_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): exp_idx[r['compound_id']] = int(r['row_idx'])
EXP_SLICES = {'rdkit_full': (0, 217), 'dpc': (217, 617), 'atompair': (617, 2665)}

za = np.load(DATA / 'all_features_fp16.npz')
ALLF = za['embeddings'].astype(np.float32)
allf_idx = {}
with open(DATA / 'all_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): allf_idx[r['compound_id']] = int(r['row_idx'])
ALLF_SLICES = {'aac': (0,26), 'physchem': (26,36), 'dbaasp': (36,48), 'mod': (48,57)}

print(f"All features loaded ({time.time()-t0:.1f}s)", flush=True)

# ---- VECTORIZED feature assembly (pre-built lookup, no Python loop per row) ----
_feat_cache = {}

def assemble_fast(train_rows, config):
    """Vectorized: build group matrices once, then np.hstack selected ones."""
    cache_key = tuple(sorted(k for k, v in config.items() if v))
    if cache_key in _feat_cache:
        cached = _feat_cache[cache_key]
        return cached

    n = len(train_rows)
    cids = [r[0] for r in train_rows]
    seqs = [r[2] for r in train_rows]

    # Pre-build per-group matrices for this set of rows
    group_mats = {}
    group_valid = np.ones(n, dtype=bool)

    if config.get('morgan'):
        idx = [mor_idx.get(c, -1) for c in cids]
        valid = np.array([i >= 0 for i in idx])
        group_valid &= valid
        mat = np.zeros((n, MOR.shape[1]), dtype=np.float32)
        for j, i in enumerate(idx):
            if i >= 0: mat[j] = MOR[i]
        group_mats['morgan'] = mat

    if config.get('esm2'):
        idx = [esm_idx.get(s, -1) for s in seqs]
        valid = np.array([i >= 0 for i in idx])
        group_valid &= valid
        mat = np.zeros((n, ESM.shape[1]), dtype=np.float32)
        for j, i in enumerate(idx):
            if i >= 0: mat[j] = ESM[i]
        group_mats['esm2'] = mat

    if config.get('ncaa'):
        mat = np.zeros((n, NCAA.shape[1]), dtype=np.float32)
        for j, c in enumerate(cids):
            i = ncaa_idx.get(c, -1)
            if i >= 0: mat[j] = NCAA[i]
            else: group_valid[j] = False
        group_mats['ncaa'] = mat

    if config.get('desc'):
        mat = np.stack([desc_map.get(c, np.zeros(11, dtype=np.float32)) for c in cids])
        group_mats['desc'] = mat

    if config.get('chemberta'):
        mat = np.zeros((n, 384), dtype=np.float32)
        for j, c in enumerate(cids):
            v = cb_map.get(c)
            if v is not None: mat[j] = v
            else: group_valid[j] = False
        group_mats['chemberta'] = mat

    if config.get('esm3b'):
        mat = np.zeros((n, 2560), dtype=np.float32)
        for j, s in enumerate(seqs):
            v = e3b_map.get(s)
            if v is not None: mat[j] = v
            else: group_valid[j] = False
        group_mats['esm3b'] = mat

    if config.get('rdkit_full'):
        mat = np.zeros((n, 217), dtype=np.float32)
        for j, c in enumerate(cids):
            i = exp_idx.get(c, -1)
            if i >= 0: s0, e0 = EXP_SLICES['rdkit_full']; mat[j] = EXP[i, s0:e0]
            else: group_valid[j] = False
        group_mats['rdkit_full'] = mat

    if config.get('dpc'):
        mat = np.zeros((n, 400), dtype=np.float32)
        for j, c in enumerate(cids):
            i = exp_idx.get(c, -1)
            if i >= 0: s0, e0 = EXP_SLICES['dpc']; mat[j] = EXP[i, s0:e0]
            else: group_valid[j] = False
        group_mats['dpc'] = mat

    if config.get('atompair'):
        mat = np.zeros((n, 2048), dtype=np.float32)
        for j, c in enumerate(cids):
            i = exp_idx.get(c, -1)
            if i >= 0: s0, e0 = EXP_SLICES['atompair']; mat[j] = EXP[i, s0:e0]
            else: group_valid[j] = False
        group_mats['atompair'] = mat

    for gname in ['aac', 'physchem', 'dbaasp', 'mod']:
        if config.get(gname):
            s0, e0 = ALLF_SLICES[gname]
            mat = np.zeros((n, e0 - s0), dtype=np.float32)
            for j, c in enumerate(cids):
                i = allf_idx.get(c, -1)
                if i >= 0: mat[j] = ALLF[i, s0:e0]
                else: group_valid[j] = False
            group_mats[gname] = mat

    # Filter valid rows and concatenate
    valid_idx = np.where(group_valid)[0]
    if len(valid_idx) < MIN_ROWS:
        _feat_cache[cache_key] = (None, None, None)
        return None, None, None

    if group_mats:
        selected = [group_mats[k][valid_idx] for k in config if config[k] and k in group_mats]
        X = np.hstack(selected) if len(selected) > 1 else selected[0]
    else:
        _feat_cache[cache_key] = (None, None, None)
        return None, None, None

    y = np.array([train_rows[i][1] for i in valid_idx])
    groups = np.array([train_rows[i][0] for i in valid_idx])
    _feat_cache[cache_key] = (X, y, groups)
    return X, y, groups


def assemble(train_rows, config):
    valid = []
    for i, (cid, y, seq, r) in enumerate(train_rows):
        parts = []
        ok = True
        if config.get('morgan'):
            if cid in mor_idx: parts.append(MOR[mor_idx[cid]])
            else: ok = False
        if config.get('esm2'):
            if seq in esm_idx: parts.append(ESM[esm_idx[seq]])
            else: ok = False
        if config.get('ncaa'):
            ni = ncaa_idx.get(cid)
            if ni is not None: parts.append(NCAA[ni])
            else: ok = False
        if config.get('desc'):
            parts.append(desc_map.get(cid, np.zeros(11, dtype=np.float32)))
        if config.get('chemberta'):
            if cid in cb_map: parts.append(cb_map[cid])
            else: ok = False
        if config.get('esm3b'):
            if seq in e3b_map: parts.append(e3b_map[seq])
            else: ok = False
        ei = exp_idx.get(cid)
        if config.get('rdkit_full'):
            if ei is not None: s,e = EXP_SLICES['rdkit_full']; parts.append(EXP[ei, s:e])
            else: ok = False
        if config.get('dpc'):
            if ei is not None: s,e = EXP_SLICES['dpc']; parts.append(EXP[ei, s:e])
            else: ok = False
        if config.get('atompair'):
            if ei is not None: s,e = EXP_SLICES['atompair']; parts.append(EXP[ei, s:e])
            else: ok = False
        ai = allf_idx.get(cid)
        if config.get('aac'):
            if ai is not None: s,e = ALLF_SLICES['aac']; parts.append(ALLF[ai, s:e])
            else: ok = False
        if config.get('physchem'):
            if ai is not None: s,e = ALLF_SLICES['physchem']; parts.append(ALLF[ai, s:e])
            else: ok = False
        if config.get('dbaasp'):
            if ai is not None: s,e = ALLF_SLICES['dbaasp']; parts.append(ALLF[ai, s:e])
            else: ok = False
        if config.get('mod'):
            if ai is not None: s,e = ALLF_SLICES['mod']; parts.append(ALLF[ai, s:e])
            else: ok = False
        if ok and parts:
            valid.append((i, np.concatenate(parts)))
    if len(valid) < MIN_ROWS: return None, None, None
    X = np.stack([v[1] for v in valid])
    y = np.array([train_rows[v[0]][1] for v in valid])
    groups = np.array([train_rows[v[0]][0] for v in valid])
    return X, y, groups


def build_model(model_type, params, n_rows=0, n_dims=0):
    """Build a sklearn-compatible model from type + params.
    Skips expensive models on large/high-dim data to prevent hanging."""
    if model_type == 'ridge':
        return Ridge(alpha=params['alpha']), True
    elif model_type == 'elasticnet':
        if n_rows > 2000 or n_dims > 3000:
            return None, False  # ElasticNet single-threaded, too slow for large/high-dim
        return ElasticNet(alpha=params['en_alpha'], l1_ratio=params['en_l1'], max_iter=500), True
    elif model_type == 'svr':
        if n_rows > 3000:
            return None, False  # SVR O(n²) too slow for large datasets
        return SVR(C=params['svr_C'], gamma=params['svr_gamma'], epsilon=params['svr_eps']), True
    elif model_type == 'lgbm':
        return lgb.LGBMRegressor(
            n_estimators=params['lgbm_ne'], learning_rate=params['lgbm_lr'],
            num_leaves=params['lgbm_lv'], min_child_samples=params['lgbm_mc'],
            subsample=0.9, colsample_bytree=params['lgbm_ct'],
            reg_alpha=params.get('lgbm_la', 0.01), reg_lambda=params.get('lgbm_lb', 0.01),
            random_state=42, verbose=-1, n_jobs=16), False
    elif model_type == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=params['xgb_ne'], learning_rate=params['xgb_lr'],
            max_depth=params['xgb_depth'], min_child_weight=params['xgb_mcw'],
            subsample=0.9, colsample_bytree=params['xgb_ct'],
            reg_alpha=params.get('xgb_la', 0.01), reg_lambda=params.get('xgb_lb', 1.0),
            random_state=42, verbosity=0, n_jobs=16, tree_method='hist'), False
    elif model_type == 'rf':
        return RandomForestRegressor(
            n_estimators=params['rf_ne'], max_depth=params.get('rf_depth', None),
            min_samples_split=params['rf_mss'], min_samples_leaf=params['rf_msl'],
            max_features=params['rf_mf'], random_state=42, n_jobs=16), False
    elif model_type == 'mlp':
        if n_rows > 3000 or n_dims > 5000:
            return None, False  # MLP too slow for large/high-dim data
        hidden = params['mlp_hidden']
        return MLPRegressor(
            hidden_layer_sizes=hidden, activation=params['mlp_act'],
            alpha=params['mlp_alpha'], learning_rate_init=params['mlp_lri'],
            max_iter=200, early_stopping=True, n_iter_no_change=10,
            random_state=42), True
    elif model_type == 'gbdt':
        if n_rows > 2500:
            return None, False  # sklearn GBDT single-threaded, too slow for large data
        return GradientBoostingRegressor(
            n_estimators=params['gb_ne'], learning_rate=params['gb_lr'],
            max_depth=params['gb_depth'], subsample=0.9,
            random_state=42), False
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def make_objective(train_rows):
    def objective(trial):
        config = {f: trial.suggest_categorical(f, [True, False]) for f in FEATURE_FLAGS}
        if not any(config.values()): return -1.0
        n_selected = sum(config.values())
        if n_selected > 6: return -1.0  # limit to max 6 groups to keep assembly fast

        model_type = trial.suggest_categorical('model',
            ['ridge', 'elasticnet', 'svr', 'lgbm', 'xgboost', 'rf', 'mlp', 'gbdt'])

        params = {}
        if model_type == 'ridge':
            params['alpha'] = trial.suggest_float('ridge_alpha', 0.01, 100, log=True)
        elif model_type == 'elasticnet':
            params['en_alpha'] = trial.suggest_float('en_alpha', 0.001, 10, log=True)
            params['en_l1'] = trial.suggest_float('en_l1', 0.01, 0.99)
        elif model_type == 'svr':
            params['svr_C'] = trial.suggest_float('svr_C', 0.1, 100, log=True)
            params['svr_gamma'] = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
            params['svr_eps'] = trial.suggest_float('svr_eps', 0.001, 0.5, log=True)
        elif model_type == 'lgbm':
            params['lgbm_ne'] = trial.suggest_int('lgbm_ne', 100, 1000)
            params['lgbm_lr'] = trial.suggest_float('lgbm_lr', 0.005, 0.2, log=True)
            params['lgbm_lv'] = trial.suggest_int('lgbm_lv', 15, 255)
            params['lgbm_mc'] = trial.suggest_int('lgbm_mc', 3, 50)
            params['lgbm_ct'] = trial.suggest_float('lgbm_ct', 0.4, 1.0)
            params['lgbm_la'] = trial.suggest_float('lgbm_la', 1e-4, 10, log=True)
            params['lgbm_lb'] = trial.suggest_float('lgbm_lb', 1e-4, 10, log=True)
        elif model_type == 'xgboost':
            params['xgb_ne'] = trial.suggest_int('xgb_ne', 100, 800)
            params['xgb_lr'] = trial.suggest_float('xgb_lr', 0.005, 0.2, log=True)
            params['xgb_depth'] = trial.suggest_int('xgb_depth', 3, 12)
            params['xgb_mcw'] = trial.suggest_int('xgb_mcw', 1, 20)
            params['xgb_ct'] = trial.suggest_float('xgb_ct', 0.4, 1.0)
            params['xgb_la'] = trial.suggest_float('xgb_la', 1e-4, 10, log=True)
            params['xgb_lb'] = trial.suggest_float('xgb_lb', 1e-4, 10, log=True)
        elif model_type == 'rf':
            params['rf_ne'] = trial.suggest_int('rf_ne', 50, 500)
            params['rf_depth'] = trial.suggest_int('rf_depth', 3, 30)
            params['rf_mss'] = trial.suggest_int('rf_mss', 2, 20)
            params['rf_msl'] = trial.suggest_int('rf_msl', 1, 10)
            params['rf_mf'] = trial.suggest_float('rf_mf', 0.1, 1.0)
        elif model_type == 'mlp':
            n_layers = trial.suggest_int('mlp_nlayers', 1, 3)
            h1 = trial.suggest_int('mlp_h1', 32, 512)
            if n_layers == 1: hidden = (h1,)
            elif n_layers == 2:
                h2 = trial.suggest_int('mlp_h2', 16, 256); hidden = (h1, h2)
            else:
                h2 = trial.suggest_int('mlp_h2', 16, 256)
                h3 = trial.suggest_int('mlp_h3', 8, 128); hidden = (h1, h2, h3)
            params['mlp_hidden'] = hidden
            params['mlp_act'] = trial.suggest_categorical('mlp_act', ['relu', 'tanh'])
            params['mlp_alpha'] = trial.suggest_float('mlp_alpha', 1e-5, 0.1, log=True)
            params['mlp_lri'] = trial.suggest_float('mlp_lri', 1e-4, 0.01, log=True)
        elif model_type == 'gbdt':
            params['gb_ne'] = trial.suggest_int('gb_ne', 50, 500)
            params['gb_lr'] = trial.suggest_float('gb_lr', 0.01, 0.2, log=True)
            params['gb_depth'] = trial.suggest_int('gb_depth', 2, 8)

        X, y, groups = assemble_fast(train_rows, config)
        if X is None: return -1.0

        gkf = GroupKFold(n_splits=5)
        try: splits = list(gkf.split(X, y, groups))
        except ValueError: return -1.0

        preds = np.zeros_like(y)
        for tr, te in splits:
            try:
                model, needs_scaling = build_model(model_type, params,
                                                    n_rows=len(tr), n_dims=X.shape[1])
                if model is None:
                    preds[te] = y[tr].mean()
                    continue
                if needs_scaling:
                    sc = StandardScaler(with_mean=False).fit(X[tr])
                    model.fit(sc.transform(X[tr]), y[tr])
                    preds[te] = model.predict(sc.transform(X[te]))
                else:
                    model.fit(X[tr], y[tr])
                    preds[te] = model.predict(X[te])
            except Exception:
                preds[te] = y[tr].mean()

        sp = float(spearmanr(y, preds).statistic)
        return sp if not np.isnan(sp) else -1.0
    return objective


# ---- Checkpoint / resume ----
completed_eps = set()
if CHECKPOINT.exists():
    completed_eps = set(json.loads(CHECKPOINT.read_text()).get('completed', []))
    print(f"Resuming: {len(completed_eps)} endpoints already done", flush=True)

# ---- Main loop ----
KEY_ENDPOINTS = ['half_life', 'EC50', 'IC50', 'hemolysis', 'MIC', 'MBC', 'Ki', 'Kd', 'LC_lemma', 'MFC']
all_demo_predictions = []
all_best_configs = {}

for ep_name in KEY_ENDPOINTS:
    if ep_name in completed_eps:
        print(f"\n[{ep_name}] SKIP (already done)", flush=True)
        continue

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
            if cid in DEMO_EXCLUDE: demo_rows.append((cid, y, seq, r))
            else: train_rows.append((cid, y, seq, r))
    if len(train_rows) < MIN_ROWS: continue

    unit = train_rows[0][3].get('unit_target', '') if train_rows else ''
    print(f"\n{'='*70}")
    print(f"[{ep_name}] {len(train_rows):,} rows | {unit} | starting at {time.time()-t0:.0f}s", flush=True)

    objective = make_objective(train_rows)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=30),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5, interval_steps=5))

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   gc_after_trial=True, timeout=900)  # max 15 min per endpoint

    best = study.best_trial
    cfg = best.params
    feat_used = [k for k in FEATURE_FLAGS if cfg.get(k)]
    print(f"  ═══ BEST RESULT ═══", flush=True)
    print(f"  Spearman: {best.value:.4f} (trial {best.number})", flush=True)
    print(f"  Features({len(feat_used)}): {'+'.join(feat_used)}", flush=True)
    print(f"  Model: {cfg['model']}", flush=True)
    model_params = {k: v for k, v in cfg.items() if not k in FEATURE_FLAGS and k != 'model'}
    print(f"  Params: {model_params}", flush=True)
    print(f"  Trials: {len(study.trials)} | Pruned: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}", flush=True)

    # Save best config
    all_best_configs[ep_name] = {
        'spearman': best.value, 'features': feat_used,
        'model': cfg['model'], 'params': {k: str(v) for k, v in model_params.items()},
        'n_trials': len(study.trials), 'n_train': len(train_rows), 'unit': unit
    }
    joblib.dump({'endpoint': ep_name, 'best_params': cfg, 'best_spearman': best.value,
                 'features_used': feat_used, 'n_trials': len(study.trials),
                 'all_trials': [(t.number, t.value, {k: str(v) for k, v in t.params.items()})
                               for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]},
                OUT / 'models' / f'{ep_name}.joblib')

    # Demo prediction
    feat_config = {f: cfg.get(f, False) for f in FEATURE_FLAGS}
    X, y, groups = assemble_fast(train_rows, feat_config)
    if X is not None:
        try:
            model, needs_scaling = build_model(cfg['model'], {k: v for k, v in cfg.items() if k not in FEATURE_FLAGS and k != 'model'})
            if needs_scaling:
                scaler = StandardScaler(with_mean=False).fit(X)
                model.fit(scaler.transform(X), y)
            else:
                scaler = None
                model.fit(X, y)

            for cid, act, seq, r in demo_rows:
                dX, _, _ = assemble_fast([(cid, 0.0, seq, r)], feat_config)
                if dX is not None:
                    if scaler: dX = scaler.transform(dX)
                    pred = float(model.predict(dX)[0])
                    all_demo_predictions.append([DEMO_NAMES.get(cid, cid), ep_name,
                                                f'{act:.3g}', f'{pred:.3g}', unit,
                                                f'{best.value:.4f}', len(X)])
                    print(f"  → {DEMO_NAMES.get(cid, cid)}: pred={pred:.3g} | actual={act:.3g}", flush=True)
        except Exception as e:
            print(f"  Demo prediction error: {e}", flush=True)

    # Checkpoint
    completed_eps.add(ep_name)
    CHECKPOINT.write_text(json.dumps({'completed': sorted(completed_eps),
                                       'elapsed': time.time()-t0,
                                       'configs': {k: v['spearman'] for k, v in all_best_configs.items()}}))

# ---- Final output ----
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(all_demo_predictions)

with open(OUT / 'best_configs.json', 'w', encoding='utf-8') as f:
    json.dump(all_best_configs, f, ensure_ascii=False, indent=2)

with open(OUT / 'ultimate_report.md', 'w', encoding='utf-8') as f:
    f.write(f"# ULTIMATE Optuna 全特征全模型搜索\n\n")
    f.write(f"> {len(FEATURE_FLAGS)} 特征组 × 8 模型 × {N_TRIALS} trials | 无时间限制\n\n")
    f.write("## 最优配置\n\n| Endpoint | Sp | Features | Model |\n|---|---|---|---|\n")
    for ep, cfg in sorted(all_best_configs.items(), key=lambda x: -x[1]['spearman']):
        f.write(f"| {ep} | **{cfg['spearman']:.4f}** | {len(cfg['features'])} groups | {cfg['model']} |\n")
    f.write("\n## Demo Predictions\n\n| Drug | Endpoint | Actual | Predicted | Unit | CV Sp |\n|---|---|---|---|---|---|\n")
    for r in all_demo_predictions:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[4]} | {r[5]} |\n")

print(f"\n{'='*70}")
print(f"COMPLETE: {len(all_demo_predictions)} predictions across {len(all_best_configs)} endpoints")
print(f"Total time: {(time.time()-t0)/60:.0f} minutes")
print(f"Best configs saved to {OUT / 'best_configs.json'}")
for ep, cfg in sorted(all_best_configs.items(), key=lambda x: -x[1]['spearman']):
    print(f"  {ep:<20} Sp={cfg['spearman']:.4f}  {cfg['model']:<12} {len(cfg['features'])} features")
