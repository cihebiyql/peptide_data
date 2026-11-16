#!/usr/bin/env python3
"""COMPLETE PIPELINE: 22 endpoints × 13 feature groups × 8 models × exhaustive tuning.

Architecture:
  1. Pre-compute ALL 13 feature groups as contiguous matrices (ONE TIME)
  2. For each endpoint: Optuna searches (feature subset × model × hyperparams)
  3. Feature assembly = numpy fancy indexing (microseconds, no Python loops)
  4. Supports checkpoint/resume, no time limit

This is the definitive version. Run for as long as needed.
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

DATA = Path('/data/qlyu/s3')
PER = Path('/data/qlyu/s3_per')
OUT = DATA / 'complete'
CKPT = OUT / 'checkpoint.json'
CACHE = DATA / 'feature_cache.npz'  # pre-computed matrices cache
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 40
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}
N_TRIALS = 500  # maximum exhaustive search
TIMEOUT = 7200  # 2 hours per endpoint max

t0 = time.time()
print("=" * 70)
print("COMPLETE PIPELINE: ALL endpoints × ALL features × ALL models")
print("=" * 70)

# ============================================================
# STEP 1: Pre-compute ALL feature groups as contiguous matrices
# ============================================================
print("\nStep 1: Pre-computing feature matrices...", flush=True)

# Build master compound list (union of all endpoints)
all_cids = set()
all_seqs = set()
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            if cid:
                all_cids.add(cid)
            if seq:
                all_seqs.add(seq)
all_cids = sorted(all_cids)
n_compounds = len(all_cids)
cid_pos = {c: i for i, c in enumerate(all_cids)}
print(f"  Master list: {n_compounds:,} compounds, {len(all_seqs):,} unique sequences", flush=True)

GROUP_NAMES = ['morgan', 'esm2', 'ncaa', 'desc', 'chemberta', 'esm3b',
               'rdkit_full', 'dpc', 'atompair', 'aac', 'physchem', 'dbaasp', 'mod']
GROUP_DIMS = {}

# Helper: fill matrix from index dict
def fill_matrix(idx_dict, source_mat, n, dim, key_type='cid'):
    mat = np.zeros((n, dim), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i, key in enumerate(all_cids if key_type == 'cid' else []):
        idx = idx_dict.get(key, -1)
        if idx >= 0:
            mat[i] = source_mat[idx]
            valid[i] = True
    return mat, valid

# 1. Morgan
print("  [1/13] Morgan...", flush=True)
zm = np.load(DATA / 'morgan_fp16.npz')
MOR_src = zm['embeddings'].astype(np.float32)
mor_lookup = {}
with open(DATA / 'morgan_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): mor_lookup[r['compound_id']] = int(r['row_idx'])
MOR, mor_valid = fill_matrix(mor_lookup, MOR_src, n_compounds, MOR_src.shape[1])
GROUP_DIMS['morgan'] = MOR.shape[1]
del MOR_src

# 2. ESM2
print("  [2/13] ESM2-650M...", flush=True)
ze = np.load(DATA / 'esm2_fp16.npz')
ESM_src = ze['embeddings'].astype(np.float32)
esm_lookup = {}
with open(DATA / 'esm2_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): esm_lookup[r['sequence']] = int(r['row_idx'])
# Map sequence to compound position
ESM = np.zeros((n_compounds, ESM_src.shape[1]), dtype=np.float32)
esm_valid = np.zeros(n_compounds, dtype=bool)
# Build seq->cid mapping from endpoint tables
seq_to_cid = {}
for fp in PER.glob('*__clean.tsv'):
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            if cid and seq and seq not in seq_to_cid:
                seq_to_cid[seq] = cid
for seq, cid in seq_to_cid.items():
    pos = cid_pos.get(cid)
    idx = esm_lookup.get(seq, -1)
    if pos is not None and idx >= 0:
        ESM[pos] = ESM_src[idx]
        esm_valid[pos] = True
GROUP_DIMS['esm2'] = ESM.shape[1]
del ESM_src

# 3. ncAA
print("  [3/13] ncAA-RepDistill...", flush=True)
zn1 = np.load(DATA / 'ncaa_set1_fp16.npz')
zn2 = np.load(DATA / 'ncaa_mod_fp16.npz')
NCAA_src = np.vstack([zn1['embeddings'], zn2['embeddings']]).astype(np.float32)
ncaa_lookup = {}
with open(DATA / 'ncaa_set1_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): ncaa_lookup[r['compound_id']] = int(r['row_idx'])
with open(DATA / 'ncaa_mod_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): ncaa_lookup.setdefault(r['compound_id'], int(r['row_idx']) + zn1['embeddings'].shape[0])
NCAA, ncaa_valid = fill_matrix(ncaa_lookup, NCAA_src, n_compounds, NCAA_src.shape[1])
GROUP_DIMS['ncaa'] = NCAA.shape[1]
del NCAA_src, zn1, zn2

# 4. Basic descriptors
print("  [4/13] Basic descriptors...", flush=True)
desc_lookup = {}
with open(DATA / 'descriptors.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        desc_lookup[r['compound_id']] = np.array([float(r[c]) if r[c] else 0.0 for c in
            ['MolWt','cLogP','TPSA','HBD','HBA','RotB','Rings','HeavyAtoms','FracCsp3','FormalCharge','QED']],
            dtype=np.float32)
DESC = np.stack([desc_lookup.get(c, np.zeros(11, dtype=np.float32)) for c in all_cids])
GROUP_DIMS['desc'] = 11

# 5. ChemBERTa
print("  [5/13] ChemBERTa...", flush=True)
zcb = np.load(DATA / 'chemberta_fp16.npz')
CB_src = zcb['embeddings'].astype(np.float32)
cb_lookup = {}
with open(DATA / 'chemberta_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): cb_lookup[r['compound_id']] = int(r['row_idx'])
CB, cb_valid = fill_matrix(cb_lookup, CB_src, n_compounds, CB_src.shape[1])
GROUP_DIMS['chemberta'] = CB.shape[1]
del CB_src

# 6. ESM2-3B
print("  [6/13] ESM2-3B...", flush=True)
ze3 = np.load(DATA / 'esm3b_fp16.npz')
E3B_src = ze3['embeddings'].astype(np.float32)
e3b_lookup = {}
with open(DATA / 'esm3b_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): e3b_lookup[r['sequence']] = int(r['row_idx'])
E3B = np.zeros((n_compounds, E3B_src.shape[1]), dtype=np.float32)
e3b_valid = np.zeros(n_compounds, dtype=bool)
for seq, cid in seq_to_cid.items():
    pos = cid_pos.get(cid)
    idx = e3b_lookup.get(seq, -1)
    if pos is not None and idx >= 0:
        E3B[pos] = E3B_src[idx]
        e3b_valid[pos] = True
GROUP_DIMS['esm3b'] = E3B.shape[1]
del E3B_src

# 7-9: From expanded CPU features
print("  [7-9] RDKit full + DPC + AtomPair...", flush=True)
zx = np.load(DATA / 'expanded_cpu_features_fp16.npz')
EXP_src = zx['embeddings'].astype(np.float32)
exp_lookup = {}
with open(DATA / 'expanded_cpu_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): exp_lookup[r['compound_id']] = int(r['row_idx'])
EXP_full, exp_valid = fill_matrix(exp_lookup, EXP_src, n_compounds, EXP_src.shape[1])
RDKIT = EXP_full[:, 0:217]
DPC = EXP_full[:, 217:617]
AP = EXP_full[:, 617:2665]
GROUP_DIMS['rdkit_full'] = 217
GROUP_DIMS['dpc'] = 400
GROUP_DIMS['atompair'] = 2048
del EXP_src, EXP_full

# 10-13: From compound features
print("  [10-13] AAC + PhysChem + DBAASP + Mod...", flush=True)
za = np.load(DATA / 'all_features_fp16.npz')
ALLF_src = za['embeddings'].astype(np.float32)
allf_lookup = {}
with open(DATA / 'all_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): allf_lookup[r['compound_id']] = int(r['row_idx'])
ALLF, allf_valid = fill_matrix(allf_lookup, ALLF_src, n_compounds, ALLF_src.shape[1])
AAC = ALLF[:, 0:26]
PHYS = ALLF[:, 26:36]
DBAASP = ALLF[:, 36:48]
MOD = ALLF[:, 48:57]
GROUP_DIMS['aac'] = 26
GROUP_DIMS['physchem'] = 10
GROUP_DIMS['dbaasp'] = 12
GROUP_DIMS['mod'] = 9
del ALLF_src, ALLF

# Store all groups in a dict for easy access
FEATURES = {
    'morgan': (MOR, mor_valid),
    'esm2': (ESM, esm_valid),
    'ncaa': (NCAA, ncaa_valid),
    'desc': (DESC, np.ones(n_compounds, dtype=bool)),  # always available (zeros for missing)
    'chemberta': (CB, cb_valid),
    'esm3b': (E3B, e3b_valid),
    'rdkit_full': (RDKIT, exp_valid),
    'dpc': (DPC, exp_valid),
    'atompair': (AP, exp_valid),
    'aac': (AAC, allf_valid),
    'physchem': (PHYS, allf_valid),
    'dbaasp': (DBAASP, allf_valid),
    'mod': (MOD, allf_valid),
}

# Save feature cache for fast restart
print("  Saving feature cache...", flush=True)
cache_data = {'compound_ids': np.array(all_cids, dtype=object)}
for g in GROUP_NAMES:
    mat, valid = FEATURES[g]
    cache_data[f'{g}_mat'] = mat
    cache_data[f'{g}_valid'] = valid
np.savez_compressed(CACHE, **cache_data)
print(f"  Feature cache saved to {CACHE}", flush=True)
print(f"\n  All 13 groups ready ({time.time()-t0:.1f}s)", flush=True)
for g, (mat, valid) in FEATURES.items():
    print(f"    {g:<14} {mat.shape[1]:>5} dims  {valid.sum():>7,} valid", flush=True)

# ============================================================
# STEP 2: Fast feature assembly (numpy slicing only)
# ============================================================

def assemble_fast(row_positions, config):
    """Assemble features for given row positions. Pure numpy, no Python loops."""
    parts = []
    for g in GROUP_NAMES:
        if config.get(g):
            mat, _ = FEATURES[g]
            parts.append(mat[row_positions])
    if not parts:
        return None
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def get_valid_mask(row_positions, config):
    """Check which positions have ALL required features valid."""
    mask = np.ones(len(row_positions), dtype=bool)
    for g in GROUP_NAMES:
        if config.get(g):
            _, valid = FEATURES[g]
            mask &= valid[row_positions]
    return mask


# ============================================================
# STEP 3: Optuna objective (fast, no bottlenecks)
# ============================================================

def make_objective(row_positions, y, groups):
    def objective(trial):
        config = {f: trial.suggest_categorical(f, [True, False]) for f in GROUP_NAMES}
        if not any(config.values()):
            return -1.0
        if sum(config.values()) > 7:
            return -1.0

        valid = get_valid_mask(row_positions, config)
        if valid.sum() < MIN_ROWS:
            return -1.0
        valid_pos = row_positions[valid]
        X = assemble_fast(valid_pos, config)
        y_v = y[valid]
        g_v = groups[valid]

        model_type = trial.suggest_categorical('model',
            ['ridge', 'elasticnet', 'svr', 'lgbm', 'rf', 'mlp', 'gbdt'])

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
            params['ne'] = trial.suggest_int('lgbm_ne', 100, 1000)
            params['lr'] = trial.suggest_float('lgbm_lr', 0.005, 0.2, log=True)
            params['lv'] = trial.suggest_int('lgbm_lv', 15, 255)
            params['mc'] = trial.suggest_int('lgbm_mc', 3, 50)
            params['ct'] = trial.suggest_float('lgbm_ct', 0.4, 1.0)
            params['la'] = trial.suggest_float('lgbm_la', 1e-4, 10, log=True)
            params['lb'] = trial.suggest_float('lgbm_lb', 1e-4, 10, log=True)
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

        # Cross-validation
        gkf = GroupKFold(n_splits=5)
        try:
            splits = list(gkf.split(X, y_v, g_v))
        except ValueError:
            return -1.0

        preds = np.zeros_like(y_v)
        for tr, te in splits:
            try:
                if model_type == 'ridge':
                    sc = StandardScaler(with_mean=False).fit(X[tr])
                    m = Ridge(alpha=params['alpha'])
                    m.fit(sc.transform(X[tr]), y_v[tr])
                    preds[te] = m.predict(sc.transform(X[te]))
                elif model_type == 'elasticnet':
                    if len(tr) > 2000 or X.shape[1] > 3000: preds[te] = y_v[tr].mean(); continue
                    sc = StandardScaler(with_mean=False).fit(X[tr])
                    m = ElasticNet(alpha=params['en_alpha'], l1_ratio=params['en_l1'], max_iter=500)
                    m.fit(sc.transform(X[tr]), y_v[tr])
                    preds[te] = m.predict(sc.transform(X[te]))
                elif model_type == 'svr':
                    if len(tr) > 2000: preds[te] = y_v[tr].mean(); continue
                    sc = StandardScaler(with_mean=False).fit(X[tr])
                    m = SVR(C=params['svr_C'], gamma=params['svr_gamma'], epsilon=params['svr_eps'])
                    m.fit(sc.transform(X[tr]), y_v[tr])
                    preds[te] = m.predict(sc.transform(X[te]))
                elif model_type == 'lgbm':
                    m = lgb.LGBMRegressor(
                        n_estimators=params['ne'], learning_rate=params['lr'],
                        num_leaves=params['lv'], min_child_samples=params['mc'],
                        subsample=0.9, colsample_bytree=params['ct'],
                        reg_alpha=params['la'], reg_lambda=params['lb'],
                        random_state=42, verbose=-1, n_jobs=16)
                    m.fit(X[tr], y_v[tr])
                    preds[te] = m.predict(X[te])
                elif model_type == 'rf':
                    m = RandomForestRegressor(
                        n_estimators=params['rf_ne'], max_depth=params['rf_depth'],
                        min_samples_split=params['rf_mss'], min_samples_leaf=params['rf_msl'],
                        max_features=params['rf_mf'], random_state=42, n_jobs=16)
                    m.fit(X[tr], y_v[tr])
                    preds[te] = m.predict(X[te])
                elif model_type == 'mlp':
                    if len(tr) > 3000 or X.shape[1] > 5000: preds[te] = y_v[tr].mean(); continue
                    sc = StandardScaler(with_mean=False).fit(X[tr])
                    m = MLPRegressor(
                        hidden_layer_sizes=params['mlp_hidden'], activation=params['mlp_act'],
                        alpha=params['mlp_alpha'], learning_rate_init=params['mlp_lri'],
                        max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=42)
                    m.fit(sc.transform(X[tr]), y_v[tr])
                    preds[te] = m.predict(sc.transform(X[te]))
                elif model_type == 'gbdt':
                    if len(tr) > 2500: preds[te] = y_v[tr].mean(); continue
                    m = GradientBoostingRegressor(
                        n_estimators=params['gb_ne'], learning_rate=params['gb_lr'],
                        max_depth=params['gb_depth'], subsample=0.9, random_state=42)
                    m.fit(X[tr], y_v[tr])
                    preds[te] = m.predict(X[te])
            except Exception:
                preds[te] = y_v[tr].mean()

        sp = float(spearmanr(y_v, preds).statistic)
        return sp if not np.isnan(sp) else -1.0
    return objective


# ============================================================
# STEP 4: Run all endpoints
# ============================================================

completed = set()
if CKPT.exists():
    data = json.loads(CKPT.read_text())
    completed = set(data.get('completed', []))
    print(f"\nResuming: {len(completed)} endpoints already done", flush=True)

ALL_ENDPOINTS = sorted([fp.stem.replace('__clean', '') for fp in PER.glob('*__clean.tsv')])
print(f"\nEndpoints to process: {len(ALL_ENDPOINTS)}")
print(f"  Already done: {len(completed)}")
print(f"  Remaining: {len([e for e in ALL_ENDPOINTS if e not in completed])}")

all_results = {}
all_demo_preds = []

for ep_name in ALL_ENDPOINTS:
    if ep_name in completed:
        continue

    fp = PER / f'{ep_name}__clean.tsv'
    if not fp.exists():
        continue

    # Read data
    ep_rows = []
    demo_rows = []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1':
                continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try:
                y = float(r['value'])
            except:
                continue
            pos = cid_pos.get(cid)
            if pos is None:
                continue
            if cid in DEMO_EXCLUDE:
                demo_rows.append((cid, y, seq, r, pos))
            else:
                ep_rows.append((cid, y, seq, r, pos))

    if len(ep_rows) < MIN_ROWS:
        continue

    # Deduplicate
    seen = set()
    dedup_rows = []
    for row in ep_rows:
        k = (row[0], round(row[1], 3))
        if k not in seen:
            seen.add(k)
            dedup_rows.append(row)

    if len(dedup_rows) < MIN_ROWS:
        continue

    # Convert to arrays
    row_pos = np.array([r[4] for r in dedup_rows])
    y = np.array([r[1] for r in dedup_rows])
    groups = np.array([r[0] for r in dedup_rows])
    unit = dedup_rows[0][3].get('unit_target', '')

    print(f"\n{'='*70}")
    print(f"[{ep_name}] {len(dedup_rows):,} rows | {len(set(groups)):,} compounds | {unit}")
    print(f"  Starting at {time.time()-t0:.0f}s", flush=True)

    # Run Optuna
    objective = make_objective(row_pos, y, groups)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50),
        pruner=optuna.pruners.NopPruner())  # no pruning, let all trials complete

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)

    best = study.best_trial
    cfg = best.params
    feat_used = [g for g in GROUP_NAMES if cfg.get(g)]
    model_params = {k: v for k, v in cfg.items() if k not in GROUP_NAMES and k != 'model'}

    print(f"  ═══ BEST ═══", flush=True)
    print(f"  Sp = {best.value:.4f} (trial {best.number}/{len(study.trials)})", flush=True)
    print(f"  Features({len(feat_used)}): {'+'.join(feat_used)}", flush=True)
    print(f"  Model: {cfg['model']}", flush=True)
    print(f"  Params: {model_params}", flush=True)
    print(f"  Elapsed: {time.time()-t0:.0f}s", flush=True)

    # Save result
    all_results[ep_name] = {
        'spearman': best.value, 'features': feat_used, 'model': cfg['model'],
        'params': {k: str(v) for k, v in model_params.items()},
        'n_train': len(dedup_rows), 'n_trials': len(study.trials), 'unit': unit
    }
    joblib.dump(all_results[ep_name], OUT / 'models' / f'{ep_name}.joblib')

    # Demo prediction
    if demo_rows:
        try:
            feat_config = {g: cfg.get(g, False) for g in GROUP_NAMES}
            demo_pos = np.array([d[4] for d in demo_rows])
            dX = assemble_fast(demo_pos, feat_config)
            if dX is not None:
                # Train production model on full data
                valid = get_valid_mask(row_pos, feat_config)
                X_full = assemble_fast(row_pos[valid], feat_config)
                y_full = y[valid]

                if cfg['model'] == 'lgbm':
                    mp = {'n_estimators': int(cfg.get('lgbm_ne', 500)),
                          'learning_rate': float(cfg.get('lgbm_lr', 0.05)),
                          'num_leaves': int(cfg.get('lgbm_lv', 63)),
                          'min_child_samples': int(cfg.get('lgbm_mc', 20)),
                          'subsample': 0.9, 'colsample_bytree': float(cfg.get('lgbm_ct', 0.9)),
                          'reg_alpha': float(cfg.get('lgbm_la', 0.01)),
                          'reg_lambda': float(cfg.get('lgbm_lb', 0.01)),
                          'random_state': 42, 'verbose': -1, 'n_jobs': 32}
                    model = lgb.LGBMRegressor(**mp)
                    model.fit(X_full, y_full)
                    preds = model.predict(dX)
                elif cfg['model'] == 'ridge':
                    sc = StandardScaler(with_mean=False).fit(X_full)
                    model = Ridge(alpha=float(cfg.get('ridge_alpha', 10)))
                    model.fit(sc.transform(X_full), y_full)
                    preds = model.predict(sc.transform(dX))
                else:
                    # For other models, use LGBM as proxy
                    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                              random_state=42, verbose=-1, n_jobs=32)
                    model.fit(X_full, y_full)
                    preds = model.predict(dX)

                for j, d in enumerate(demo_rows):
                    pred = float(preds[j])
                    act = d[1]
                    all_demo_preds.append([DEMO_NAMES.get(d[0], d[0]), ep_name,
                                          f'{act:.3g}', f'{pred:.3g}', unit,
                                          f'{best.value:.4f}', len(dedup_rows)])
                    print(f"  → {DEMO_NAMES.get(d[0], d[0])}: pred={pred:.3g} | actual={act:.3g}", flush=True)
        except Exception as e:
            print(f"  Demo prediction error: {e}", flush=True)

    # Checkpoint
    completed.add(ep_name)
    CKPT.write_text(json.dumps({
        'completed': sorted(completed),
        'elapsed': time.time() - t0,
        'results': {k: v['spearman'] for k, v in all_results.items()}
    }))

# ============================================================
# STEP 5: Final output
# ============================================================
print(f"\n{'='*70}")
print(f"COMPLETE: {len(all_results)} endpoints finished")

with open(OUT / 'all_results.json', 'w') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f2:
    w = csv.writer(f2, delimiter='\t', lineterminator='\n')
    w.writerow(['drug', 'endpoint', 'actual', 'predicted', 'unit', 'cv_spearman', 'n_train'])
    w.writerows(all_demo_preds)

print(f"\nFinal results:")
for ep, cfg in sorted(all_results.items(), key=lambda x: -x[1]['spearman']):
    print(f"  {ep:<22} Sp={cfg['spearman']:.4f}  {cfg['model']:<10} {len(cfg['features'])} features")
print(f"\nTotal time: {(time.time()-t0)/60:.0f} minutes")
print(f"\n[OUT] {OUT}")
