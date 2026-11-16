#!/usr/bin/env python3
"""STEP 2: Load cached features, run Optuna search for ALL endpoints.
Loads from feat_cache/ (pre-computed in step1, loads in seconds)."""
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
DATA = Path('/data/qlyu/s3')
CACHE = DATA / 'feat_cache'
PER = Path('/data/qlyu/s3_per')
OUT = DATA / 'complete'
CKPT = OUT / 'checkpoint.json'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 40
N_TRIALS = 500
TIMEOUT = 7200
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}
GROUP_NAMES = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
               'rdkit_full','dpc','atompair','aac','physchem','dbaasp','mod']

t0 = time.time()
print("Loading cached features...", flush=True)
with open(CACHE / 'cids.json') as f:
    all_cids = json.load(f)
cid_pos = {c: i for i, c in enumerate(all_cids)}
n = len(all_cids)

FEATURES = {}
for g in GROUP_NAMES:
    z = np.load(CACHE / f'{g}.npz')
    FEATURES[g] = (z['mat'].astype(np.float32), z['valid'])
print(f"Loaded {len(FEATURES)} groups, {n:,} compounds ({time.time()-t0:.1f}s)", flush=True)


def assemble(pos, config):
    parts = [FEATURES[g][0][pos] for g in GROUP_NAMES if config.get(g)]
    return np.hstack(parts) if len(parts) > 1 else parts[0]


def valid_mask(pos, config):
    m = np.ones(len(pos), dtype=bool)
    for g in GROUP_NAMES:
        if config.get(g):
            m &= FEATURES[g][1][pos]
    return m


def make_objective(pos, y, groups):
    def objective(trial):
        config = {f: trial.suggest_categorical(f, [True, False]) for f in GROUP_NAMES}
        if not any(config.values()) or sum(config.values()) > 7:
            return -1.0
        vm = valid_mask(pos, config)
        if vm.sum() < MIN_ROWS: return -1.0
        vp = pos[vm]
        X = assemble(vp, config)
        y_v = y[vm]; g_v = groups[vm]

        mt = trial.suggest_categorical('model', ['ridge','elasticnet','svr','lgbm','rf','mlp','gbdt'])
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
                        reg_alpha=p['la'], reg_lambda=p['lb'], random_state=42, verbose=-1, n_jobs=16)
                    m.fit(X[tr], y_v[tr]); preds[te] = m.predict(X[te])
                elif mt == 'rf':
                    m = RandomForestRegressor(n_estimators=p['ne'], max_depth=p['dp'],
                        min_samples_split=p['ss'], min_samples_leaf=p['sl'], max_features=p['mf'],
                        random_state=42, n_jobs=16)
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
    return objective


# ---- Main ----
completed = set()
if CKPT.exists():
    completed = set(json.loads(CKPT.read_text()).get('completed', []))
    print(f"Resuming: {len(completed)} done", flush=True)

ALL_EP = sorted([fp.stem.replace('__clean','') for fp in PER.glob('*__clean.tsv')])
print(f"Endpoints: {len(ALL_EP)} total, {len([e for e in ALL_EP if e not in completed])} remaining", flush=True)

all_results = {}
all_demo = []

for ep in ALL_EP:
    if ep in completed: continue
    fp = PER / f'{ep}__clean.tsv'
    if not fp.exists(): continue

    rows, demo = [], []
    with open(fp, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r.get('struct_dup') == '1': continue
            cid = r.get('new_compound_id', '')
            seq = (r.get('sequence_notation') or '').strip().upper()
            try: y = float(r['value'])
            except: continue
            p = cid_pos.get(cid)
            if p is None: continue
            if cid in DEMO_EXCLUDE: demo.append((cid, y, seq, r, p))
            else: rows.append((cid, y, seq, r, p))

    seen = set(); dd = []
    for r in rows:
        k = (r[0], round(r[1], 3))
        if k not in seen: seen.add(k); dd.append(r)
    if len(dd) < MIN_ROWS: continue

    pos = np.array([r[4] for r in dd])
    y = np.array([r[1] for r in dd])
    groups = np.array([r[0] for r in dd])
    unit = dd[0][3].get('unit_target', '')

    print(f"\n{'='*70}\n[{ep}] {len(dd):,} rows | {unit} | t={time.time()-t0:.0f}s", flush=True)

    obj = make_objective(pos, y, groups)
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=50))
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT, gc_after_trial=True)

    best = study.best_trial
    cfg = best.params
    feats = [g for g in GROUP_NAMES if cfg.get(g)]
    mp = {k: v for k, v in cfg.items() if k not in GROUP_NAMES and k != 'model'}

    print(f"  ═══ BEST ═══ Sp={best.value:.4f} (trial {best.number}/{len(study.trials)})", flush=True)
    print(f"  Features({len(feats)}): {'+'.join(feats)}", flush=True)
    print(f"  Model: {cfg['model']}", flush=True)
    print(f"  Params: {mp}", flush=True)

    all_results[ep] = {'spearman': best.value, 'features': feats, 'model': cfg['model'],
        'params': {k: str(v) for k, v in mp.items()}, 'n_train': len(dd),
        'n_trials': len(study.trials), 'unit': unit}
    joblib.dump(all_results[ep], OUT / 'models' / f'{ep}.joblib')

    # Demo prediction
    if demo:
        try:
            fc = {g: cfg.get(g, False) for g in GROUP_NAMES}
            dm = valid_mask(pos, fc)
            X_full = assemble(pos[dm], fc); y_full = y[dm]
            dX = assemble(np.array([d[4] for d in demo]), fc)
            if dX is not None:
                if cfg['model'] == 'lgbm':
                    model = lgb.LGBMRegressor(
                        n_estimators=int(mp.get('lne',500)), learning_rate=float(mp.get('llr',0.05)),
                        num_leaves=int(mp.get('llv',63)), min_child_samples=int(mp.get('lmc',20)),
                        subsample=0.9, colsample_bytree=float(mp.get('lct',0.9)),
                        reg_alpha=float(mp.get('lla',0.01)), reg_lambda=float(mp.get('llb',0.01)),
                        random_state=42, verbose=-1, n_jobs=32)
                else:
                    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                        random_state=42, verbose=-1, n_jobs=32)
                model.fit(X_full, y_full)
                preds = model.predict(dX)
                for j, d in enumerate(demo):
                    all_demo.append([DEMO_NAMES.get(d[0], d[0]), ep, f'{d[1]:.3g}',
                                   f'{float(preds[j]):.3g}', unit, f'{best.value:.4f}', len(dd)])
                    print(f"  → {DEMO_NAMES.get(d[0],d[0])}: {preds[j]:.3g} | actual={d[1]:.3g}", flush=True)
        except Exception as e:
            print(f"  Demo error: {e}", flush=True)

    completed.add(ep)
    CKPT.write_text(json.dumps({'completed': sorted(completed), 'elapsed': time.time()-t0,
        'results': {k: v['spearman'] for k, v in all_results.items()}}))

# Final output
with open(OUT / 'all_results.json', 'w') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(all_demo)

print(f"\n{'='*70}\nCOMPLETE: {len(all_results)} endpoints, {len(all_demo)} demo predictions")
print(f"Total: {(time.time()-t0)/60:.0f} minutes")
for ep, cfg in sorted(all_results.items(), key=lambda x: -x[1]['spearman']):
    print(f"  {ep:<22} Sp={cfg['spearman']:.4f}  {cfg['model']:<10} {len(cfg['features'])}f")
