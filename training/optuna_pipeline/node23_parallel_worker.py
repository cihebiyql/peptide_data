#!/usr/bin/env python3
"""Single-endpoint worker for parallel execution. Each process handles ONE endpoint."""
import csv, sys, time, joblib, json, warnings
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

EP = sys.argv[1]  # endpoint name from command line
DATA = Path('/data/qlyu/s3')
CACHE = DATA / 'feat_cache'
PER = Path('/data/qlyu/s3_integrated')  # integrated OLD+NEW data
OUT = DATA / 'complete'
CKPT_DIR = OUT / 'ckpt'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 40
N_TRIALS = 500
TIMEOUT = 7200
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}
GROUP_NAMES = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
               'rdkit_full','dpc','atompair','aac','physchem','dbaasp','mod']

t0 = time.time()

# Check if already done
ep_ckpt = CKPT_DIR / f'{EP}.json'
if ep_ckpt.exists():
    print(f"[{EP}] already done, skipping", flush=True)
    sys.exit(0)

# Load cached features
with open(CACHE / 'cids.json') as f:
    all_cids = json.load(f)
cid_pos = {c: i for i, c in enumerate(all_cids)}
FEATURES = {}
for g in GROUP_NAMES:
    z = np.load(CACHE / f'{g}.npz')
    FEATURES[g] = (z['mat'].astype(np.float32), z['valid'])
print(f"[{EP}] features loaded ({time.time()-t0:.1f}s)", flush=True)


def assemble(pos, config):
    parts = [FEATURES[g][0][pos] for g in GROUP_NAMES if config.get(g)]
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def valid_mask(pos, config):
    m = np.ones(len(pos), dtype=bool)
    for g in GROUP_NAMES:
        if config.get(g):
            m &= FEATURES[g][1][pos]
    return m


def objective(trial):
    config = {f: trial.suggest_categorical(f, [True, False] if VIABLE[f] else [False]) for f in GROUP_NAMES}
    # ALWAYS suggest model so best_trial.params has 'model' even if config fails below
    mt = trial.suggest_categorical('model', ['ridge','elasticnet','svr','lgbm','rf','mlp','gbdt'])
    if not any(config.values()) or sum(config.values()) > 7:
        return -1.0
    vm = valid_mask(pos, config)
    if vm.sum() < MIN_ROWS: return -1.0
    vp = pos[vm]
    X = assemble(vp, config)
    y_v = y[vm]; g_v = groups[vm]

    p = {}
    if mt == 'ridge': p['a'] = trial.suggest_float('ra', 0.01, 100, log=True)
    elif mt == 'elasticnet': p['a'] = trial.suggest_float('ea', 0.001, 10, log=True); p['l1'] = trial.suggest_float('el', 0.01, 0.99)
    elif mt == 'svr': p['C'] = trial.suggest_float('sC', 0.1, 100, log=True); p['g'] = trial.suggest_categorical('sg', ['scale','auto']); p['e'] = trial.suggest_float('se', 0.001, 0.5, log=True)
    elif mt == 'lgbm':
        p['ne'] = trial.suggest_int('lne', 100, 1000); p['lr'] = trial.suggest_float('llr', 0.005, 0.2, log=True)
        p['lv'] = trial.suggest_int('llv', 15, 255); p['mc'] = trial.suggest_int('lmc', 3, 50)
        p['ct'] = trial.suggest_float('lct', 0.4, 1.0); p['la'] = trial.suggest_float('lla', 1e-4, 10, log=True)
        p['lb'] = trial.suggest_float('llb', 1e-4, 10, log=True)
    elif mt == 'rf':
        p['ne'] = trial.suggest_int('rne', 50, 500); p['dp'] = trial.suggest_int('rdp', 3, 30)
        p['ss'] = trial.suggest_int('rss', 2, 20); p['sl'] = trial.suggest_int('rsl', 1, 10); p['mf'] = trial.suggest_float('rmf', 0.1, 1.0)
    elif mt == 'mlp':
        nl = trial.suggest_int('mnl', 1, 3); h1 = trial.suggest_int('mh1', 32, 512)
        hd = (h1,) if nl==1 else (h1, trial.suggest_int('mh2',16,256)) if nl==2 else (h1, trial.suggest_int('mh2',16,256), trial.suggest_int('mh3',8,128))
        p['hd'] = hd; p['act'] = trial.suggest_categorical('ma', ['relu','tanh'])
        p['al'] = trial.suggest_float('mal', 1e-5, 0.1, log=True); p['lri'] = trial.suggest_float('mlri', 1e-4, 0.01, log=True)
    elif mt == 'gbdt':
        p['ne'] = trial.suggest_int('gne', 50, 500); p['lr'] = trial.suggest_float('glr', 0.01, 0.2, log=True); p['dp'] = trial.suggest_int('gdp', 2, 8)

    gkf = GroupKFold(n_splits=5)
    try: splits = list(gkf.split(X, y_v, g_v))
    except ValueError: return -1.0

    preds = np.zeros_like(y_v)
    for tr, te in splits:
        try:
            if mt == 'ridge':
                sc = StandardScaler(with_mean=False).fit(X[tr])
                preds[te] = Ridge(alpha=p['a']).fit(sc.transform(X[tr]), y_v[tr]).predict(sc.transform(X[te]))
            elif mt == 'elasticnet':
                if len(tr) > 2000 or X.shape[1] > 3000: preds[te] = y_v[tr].mean(); continue
                sc = StandardScaler(with_mean=False).fit(X[tr])
                preds[te] = ElasticNet(alpha=p['a'], l1_ratio=p['l1'], max_iter=500).fit(sc.transform(X[tr]), y_v[tr]).predict(sc.transform(X[te]))
            elif mt == 'svr':
                if len(tr) > 2000: preds[te] = y_v[tr].mean(); continue
                sc = StandardScaler(with_mean=False).fit(X[tr])
                preds[te] = SVR(C=p['C'], gamma=p['g'], epsilon=p['e']).fit(sc.transform(X[tr]), y_v[tr]).predict(sc.transform(X[te]))
            elif mt == 'lgbm':
                m = lgb.LGBMRegressor(n_estimators=p['ne'], learning_rate=p['lr'], num_leaves=p['lv'],
                    min_child_samples=p['mc'], subsample=0.9, colsample_bytree=p['ct'],
                    reg_alpha=p['la'], reg_lambda=p['lb'], random_state=42, verbose=-1, n_jobs=4)
                m.fit(X[tr], y_v[tr]); preds[te] = m.predict(X[te])
            elif mt == 'rf':
                m = RandomForestRegressor(n_estimators=p['ne'], max_depth=p['dp'],
                    min_samples_split=p['ss'], min_samples_leaf=p['sl'], max_features=p['mf'],
                    random_state=42, n_jobs=4)
                m.fit(X[tr], y_v[tr]); preds[te] = m.predict(X[te])
            elif mt == 'mlp':
                if len(tr) > 3000 or X.shape[1] > 5000: preds[te] = y_v[tr].mean(); continue
                sc = StandardScaler(with_mean=False).fit(X[tr])
                m = MLPRegressor(hidden_layer_sizes=p['hd'], activation=p['act'], alpha=p['al'],
                    learning_rate_init=p['lri'], max_iter=200, early_stopping=True,
                    n_iter_no_change=10, random_state=42)
                m.fit(sc.transform(X[tr]), y_v[tr]); preds[te] = m.predict(sc.transform(X[te]))
            elif mt == 'gbdt':
                if len(tr) > 2500: preds[te] = y_v[tr].mean(); continue
                m = GradientBoostingRegressor(n_estimators=p['ne'], learning_rate=p['lr'],
                    max_depth=p['dp'], subsample=0.9, random_state=42)
                m.fit(X[tr], y_v[tr]); preds[te] = m.predict(X[te])
        except: preds[te] = y_v[tr].mean()

    sp = float(spearmanr(y_v, preds).statistic)
    return sp if not np.isnan(sp) else -1.0


# Read endpoint data
fp = PER / f'{EP}__clean.tsv'
if not fp.exists():
    print(f"[{EP}] no file", flush=True); sys.exit(0)

rows, demo = [], []
with open(fp, encoding='utf-8') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        if r.get('struct_dup') == '1': continue
        cid = r.get('new_compound_id', '') or r.get('record_id', '')
        seq = (r.get('sequence_notation') or '').strip().upper()
        try: yv = float(r['value'])
        except: continue
        p = cid_pos.get(cid)
        if p is None: continue
        if cid in DEMO_EXCLUDE: demo.append((cid, yv, seq, r, p))
        else: rows.append((cid, yv, seq, r, p))

seen = set(); dd = []
for r in rows:
    k = (r[0], round(r[1], 3))
    if k not in seen: seen.add(k); dd.append(r)
if len(dd) < MIN_ROWS:
    print(f"[{EP}] {len(dd)} < {MIN_ROWS}, skip", flush=True); sys.exit(0)

pos = np.array([r[4] for r in dd])
y = np.array([r[1] for r in dd])
groups = np.array([r[0] for r in dd])
unit = dd[0][3].get('unit_target', '')

# Per-endpoint viable feature groups: only offer groups valid for >= MIN_ROWS rows.
# Prevents OLD-only endpoints (only esm2+desc valid) from having a near-empty feasible
# space (3/8192 configs), which caused all-trial failures and KeyError('model').
VIABLE = {g: bool(FEATURES[g][1][pos].sum() >= MIN_ROWS) for g in GROUP_NAMES}
print(f"[{EP}] {len(dd):,} rows | {unit} | viable: {[g for g in GROUP_NAMES if VIABLE[g]]}", flush=True)

# Run Optuna
study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)

best = study.best_trial
cfg = best.params
feats = [g for g in GROUP_NAMES if cfg.get(g)]
mp = {k: v for k, v in cfg.items() if k not in GROUP_NAMES and k != 'model'}

result = {'spearman': best.value, 'features': feats, 'model': cfg['model'],
    'params': {k: str(v) for k, v in mp.items()}, 'n_train': len(dd),
    'n_trials': len(study.trials), 'unit': unit}

# Persist top-20 diverse trials so the ensemble phase can reuse them for free
try:
    done = [t for t in study.trials if t.value is not None and t.value > 0 and 'model' in t.params]
    done.sort(key=lambda t: -t.value)
    seen_sig, top = set(), []
    for t in done:
        sig = (t.params['model'], tuple(sorted(g for g in GROUP_NAMES if t.params.get(g))))
        if sig in seen_sig: continue
        seen_sig.add(sig)
        top.append({'value': t.value, 'model': t.params['model'],
                    'features': [g for g in GROUP_NAMES if t.params.get(g)],
                    'params': {k: str(v) for k, v in t.params.items()
                               if k not in GROUP_NAMES and k != 'model'}})
        if len(top) >= 20: break
    result['top_trials'] = top
except Exception as e:
    print(f"[{EP}] top_trials dump failed: {e}", flush=True)

print(f"[{EP}] ═══ BEST ═══ Sp={best.value:.4f} | {cfg['model']} | {'+'.join(feats)}", flush=True)
print(f"[{EP}] Params: {mp}", flush=True)
print(f"[{EP}] Trials: {len(study.trials)} | Time: {(time.time()-t0)/60:.0f}min", flush=True)

# Demo prediction
if demo:
    try:
        fc = {g: cfg.get(g, False) for g in GROUP_NAMES}
        dm = valid_mask(pos, fc)
        X_full = assemble(pos[dm], fc); y_full = y[dm]
        dX = assemble(np.array([d[4] for d in demo]), fc)
        if dX is not None:
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                random_state=42, verbose=-1, n_jobs=4)
            model.fit(X_full, y_full)
            preds = model.predict(dX)
            result['demo'] = []
            for j, d in enumerate(demo):
                result['demo'].append({'drug': DEMO_NAMES.get(d[0], d[0]),
                    'actual': f'{d[1]:.3g}', 'predicted': f'{float(preds[j]):.3g}'})
                print(f"[{EP}] → {DEMO_NAMES.get(d[0],d[0])}: {preds[j]:.3g} | actual={d[1]:.3g}", flush=True)
    except Exception as e:
        print(f"[{EP}] demo error: {e}", flush=True)

# Save
joblib.dump(result, OUT / 'models' / f'{EP}.joblib')
ep_ckpt.write_text(json.dumps(result))
print(f"[{EP}] DONE ({(time.time()-t0)/60:.0f}min)", flush=True)
