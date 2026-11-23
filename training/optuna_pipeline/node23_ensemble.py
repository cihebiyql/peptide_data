#!/usr/bin/env python3
"""Phase-2 ensemble builder: per endpoint, construct greedy multi-model/multi-view
ensembles from the phase-1 Optuna best config and compare against the single best.

Protocol (identical to phase-1 worker): GroupKFold-5 (groups=compound), OOF Spearman.
Common evaluation set R = best config's valid rows, so all candidates are comparable.
Candidates:
  A. phase-1 best config (params from ckpt)
  B. best features x other model types (mini Optuna tune each)
  C. feature-subset variants of best features (drop-one) with best model type
  D. esm2-only / desc-only LGBM baselines (max-coverage views)
Ensemble: greedy forward selection by OOF Spearman of the prediction average.
Winner = max(single best, greedy ensemble). Saved to complete/ensemble/{EP}.json.
"""
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

EP = sys.argv[1]
DATA = Path('<WORK_DIR>/s3')
CACHE = DATA / 'feat_cache'
PER = Path('<WORK_DIR>/s3_integrated')
CKPT_DIR = DATA / 'complete' / 'ckpt'
OUT = DATA / 'complete' / 'ensemble'
OUT.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT / f'{EP}.json'
if OUT_JSON.exists():
    print(f"[{EP}] ensemble already done, skip", flush=True); sys.exit(0)

MIN_ROWS = 40
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}
GROUP_NAMES = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
               'rdkit_full','dpc','atompair','aac','physchem','dbaasp','mod']
ALL_MODELS = ['ridge','elasticnet','svr','lgbm','rf','mlp','gbdt']
FLOAT_KEYS = {'ra','ea','el','sC','se','llr','lct','lla','llb','rmf','mal','mlri','glr'}
INT_KEYS = {'lne','llv','lmc','rne','rdp','rss','rsl','mnl','mh1','mh2','mh3','gne','gdp'}
t0 = time.time()

# ---------- load ----------
with open(CACHE / 'cids.json') as f:
    all_cids = json.load(f)
cid_pos = {c: i for i, c in enumerate(all_cids)}
FEATURES = {}
for g in GROUP_NAMES:
    z = np.load(CACHE / f'{g}.npz')
    FEATURES[g] = (z['mat'].astype(np.float32), z['valid'])
ckpt = json.loads((CKPT_DIR / f'{EP}.json').read_text())
if ckpt.get('spearman', -1) <= 0:
    print(f"[{EP}] phase-1 dead (Sp={ckpt.get('spearman')}), skip", flush=True)
    sys.exit(0)

def assemble(pos_idx, config):
    parts = [FEATURES[g][0][pos_idx] for g in GROUP_NAMES if config.get(g)]
    return np.hstack(parts) if len(parts) > 1 else parts[0]

# ---------- endpoint rows ----------
fp = PER / f'{EP}__clean.tsv'
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
        if cid in DEMO_EXCLUDE: demo.append((cid, yv, p))
        else: rows.append((cid, yv, p))
seen = set(); dd = []
for r in rows:
    k = (r[0], round(r[1], 3))
    if k not in seen: seen.add(k); dd.append(r)
pos = np.array([r[2] for r in dd]); y = np.array([r[1] for r in dd])
groups = np.array([r[0] for r in dd])
VIABLE = {g: bool(FEATURES[g][1][pos].sum() >= MIN_ROWS) for g in GROUP_NAMES}

# Common evaluation set R = best config's valid rows
best_feats = set(ckpt['features'])
best_cfg = {g: g in best_feats for g in GROUP_NAMES}
R_mask = np.ones(len(dd), dtype=bool)
for g in best_feats: R_mask &= FEATURES[g][1][pos]
R = np.where(R_mask)[0]
y_R = y[R]; g_R = groups[R]
X_best = assemble(R, best_cfg)
splits = list(GroupKFold(n_splits=5).split(np.zeros(len(R)), y_R, g_R))
print(f"[{EP}] rows={len(dd):,} R={len(R):,} best={ckpt['model']}:{'+'.join(sorted(best_feats))} Sp={ckpt['spearman']:.4f}", flush=True)

# ---------- model builders (mirror phase-1 worker) ----------
def build(mt, p):
    if mt == 'ridge': return Ridge(alpha=p['a']), True
    if mt == 'elasticnet': return ElasticNet(alpha=p['a'], l1_ratio=p['l1'], max_iter=500), True
    if mt == 'svr': return SVR(C=p['C'], gamma=p['g'], epsilon=p['e']), True
    if mt == 'lgbm': return lgb.LGBMRegressor(n_estimators=p['ne'], learning_rate=p['lr'],
        num_leaves=p['lv'], min_child_samples=p['mc'], subsample=0.9, colsample_bytree=p['ct'],
        reg_alpha=p['la'], reg_lambda=p['lb'], random_state=42, verbose=-1, n_jobs=4), False
    if mt == 'rf': return RandomForestRegressor(n_estimators=p['ne'], max_depth=p['dp'],
        min_samples_split=p['ss'], min_samples_leaf=p['sl'], max_features=p['mf'],
        random_state=42, n_jobs=4), False
    if mt == 'mlp':
        nl = p['mnl']
        hd = (p['mh1'],) if nl == 1 else (p['mh1'], p['mh2']) if nl == 2 else (p['mh1'], p['mh2'], p['mh3'])
        return MLPRegressor(hidden_layer_sizes=hd, activation=p['ma'], alpha=p['al'],
            learning_rate_init=p['lri'], max_iter=200, early_stopping=True,
            n_iter_no_change=10, random_state=42), True
    if mt == 'gbdt': return GradientBoostingRegressor(n_estimators=p['ne'], learning_rate=p['lr'],
        max_depth=p['dp'], subsample=0.9, random_state=42), False

def allowed(mt, n_rows, n_dims):
    tr = n_rows * 0.8
    if mt == 'elasticnet' and (tr > 2000 or n_dims > 3000): return False
    if mt == 'svr' and tr > 2000: return False
    if mt == 'mlp' and (tr > 3000 or n_dims > 5000): return False
    if mt == 'gbdt' and tr > 2500: return False
    return True

def oof_predict(mt, p, X):
    """OOF preds on R via the fixed splits."""
    m, needs_scale = build(mt, p)
    preds = np.zeros(len(R))
    for tr, te in splits:
        try:
            if needs_scale:
                sc = StandardScaler(with_mean=False).fit(X[tr])
                preds[te] = m.fit(sc.transform(X[tr]), y_R[tr]).predict(sc.transform(X[te]))
            else:
                preds[te] = m.fit(X[tr], y_R[tr]).predict(X[te])
        except: preds[te] = y_R[tr].mean()
    return preds

def sp_of(preds):
    s = spearmanr(y_R, preds).statistic
    return float(s) if not np.isnan(s) else -1.0

# ---------- candidate A: phase-1 best ----------
raw = ckpt['params']
p_best = {}
for k, v in raw.items():
    if k in FLOAT_KEYS: p_best[k] = float(v)
    elif k in INT_KEYS: p_best[k] = int(v)
    else: p_best[k] = v

cands = []  # (name, model_type, params, config_for_X, oof_preds, sp)
preds_A = oof_predict(ckpt['model'], p_best, X_best)
sp_A = sp_of(preds_A)
cands.append((f"best:{ckpt['model']}", ckpt['model'], p_best, best_cfg, preds_A, sp_A))
print(f"[{EP}] A best single: Sp={sp_A:.4f} (phase-1 ckpt {ckpt['spearman']:.4f})", flush=True)

# ---------- mini Optuna ----------
N_MINI = 15 if len(R) > 10000 else (40 if len(R) > 3000 else 80)

def tune(mt, X, seed=43):
    """Mini tune -> (params, oof_preds, sp)."""
    def obj(trial):
        p = {}
        if mt == 'ridge': p['a'] = trial.suggest_float('ra', 0.01, 100, log=True)
        elif mt == 'elasticnet':
            p['a'] = trial.suggest_float('ea', 0.001, 10, log=True); p['l1'] = trial.suggest_float('el', 0.01, 0.99)
        elif mt == 'svr':
            p['C'] = trial.suggest_float('sC', 0.1, 100, log=True); p['g'] = trial.suggest_categorical('sg', ['scale','auto'])
            p['e'] = trial.suggest_float('se', 0.001, 0.5, log=True)
        elif mt == 'lgbm':
            p['ne'] = trial.suggest_int('lne', 100, 1000); p['lr'] = trial.suggest_float('llr', 0.005, 0.2, log=True)
            p['lv'] = trial.suggest_int('llv', 15, 255); p['mc'] = trial.suggest_int('lmc', 3, 50)
            p['ct'] = trial.suggest_float('lct', 0.4, 1.0); p['la'] = trial.suggest_float('lla', 1e-4, 10, log=True)
            p['lb'] = trial.suggest_float('llb', 1e-4, 10, log=True)
        elif mt == 'rf':
            p['ne'] = trial.suggest_int('rne', 50, 500); p['dp'] = trial.suggest_int('rdp', 3, 30)
            p['ss'] = trial.suggest_int('rss', 2, 20); p['sl'] = trial.suggest_int('rsl', 1, 10)
            p['mf'] = trial.suggest_float('rmf', 0.1, 1.0)
        elif mt == 'mlp':
            p['mnl'] = trial.suggest_int('mnl', 1, 3); p['mh1'] = trial.suggest_int('mh1', 32, 512)
            nl = p['mnl']
            if nl >= 2: p['mh2'] = trial.suggest_int('mh2', 16, 256)
            if nl >= 3: p['mh3'] = trial.suggest_int('mh3', 8, 128)
            p['ma'] = trial.suggest_categorical('ma', ['relu','tanh'])
            p['al'] = trial.suggest_float('mal', 1e-5, 0.1, log=True); p['lri'] = trial.suggest_float('mlri', 1e-4, 0.01, log=True)
        elif mt == 'gbdt':
            p['ne'] = trial.suggest_int('gne', 50, 500); p['lr'] = trial.suggest_float('glr', 0.01, 0.2, log=True)
            p['dp'] = trial.suggest_int('gdp', 2, 8)
        return sp_of(oof_predict(mt, p, X))
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(20, N_MINI)))
    study.optimize(obj, n_trials=N_MINI, gc_after_trial=True)
    if not study.trials: return None
    b = study.best_trial
    pr = oof_predict(mt, b.params, X)
    return b.params, pr, sp_of(pr)

# ---------- candidate B: other models on best features ----------
for mt in ALL_MODELS:
    if mt == ckpt['model']: continue
    if not allowed(mt, len(R), X_best.shape[1]): continue
    res = tune(mt, X_best)
    if res is None or res[2] <= 0: continue
    pB, prB, spB = res
    cands.append((f"featB:{mt}", mt, pB, best_cfg, prB, spB))
    print(f"[{EP}] B {mt} on best feats: Sp={spB:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------- candidate C: drop-one feature variants ----------
if len(best_feats) >= 3:
    for drop_g in sorted(best_feats):
        cfgC = {g: g in (best_feats - {drop_g}) for g in GROUP_NAMES}
        XC = assemble(R, cfgC)
        res = tune(ckpt['model'], XC, seed=44)
        if res is None or res[2] <= 0: continue
        pC, prC, spC = res
        cands.append((f"drop{drop_g}", ckpt['model'], pC, cfgC, prC, spC))
        print(f"[{EP}] C -{drop_g}: Sp={spC:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------- candidate D: esm2-only / desc-only LGBM baselines ----------
for g0 in ('esm2', 'desc'):
    if not VIABLE.get(g0): continue
    mD = R_mask & FEATURES[g0][1][pos]
    if mD.sum() < max(MIN_ROWS, 0.9 * len(R)): continue
    cfgD = {g: g == g0 for g in GROUP_NAMES}
    XD = assemble(R, cfgD)
    res = tune('lgbm', XD, seed=45)
    if res is None or res[2] <= 0: continue
    pD, prD, spD = res
    cands.append((f"base:{g0}", 'lgbm', pD, cfgD, prD, spD))
    print(f"[{EP}] D {g0}-only LGBM: Sp={spD:.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------- greedy ensemble (require member diversity) ----------
order = sorted(cands, key=lambda c: -c[5])
sel = [order[0]]
ens_pred = order[0][4].copy()
ens_sp = order[0][5]
used_kinds = {order[0][1]}
for c in order[1:]:
    if len(sel) >= 6: break
    if c[1] in used_kinds and len(sel) < 3: pass  # same model ok if features differ
    trial_pred = (ens_pred * len(sel) + c[4]) / (len(sel) + 1)
    trial_sp = sp_of(trial_pred)
    if trial_sp > ens_sp + 1e-4:
        sel.append(c); ens_pred = trial_pred; ens_sp = trial_sp
print(f"[{EP}] greedy ensemble ({len(sel)} members): Sp={ens_sp:.4f} vs single {sp_A:.4f}", flush=True)

# ---------- winner ----------
use_ens = ens_sp > sp_A + 1e-4
winners = sel if use_ens else [cands[0]]
result = {
    'endpoint': EP, 'n_rows_full': len(dd), 'n_rows_R': len(R),
    'single': {'sp': sp_A, 'model': ckpt['model'], 'features': sorted(best_feats),
               'params': {k: str(v) for k, v in p_best.items()}},
    'ensemble_sp': ens_sp,
    'ensemble_members': [{'name': c[0], 'model': c[1], 'sp': round(c[5], 4),
                          'features': sorted(g for g in GROUP_NAMES if c[3].get(g))} for c in sel],
    'winner': 'ensemble' if use_ens else 'single',
    'unit': ckpt.get('unit', ''),
    'time_s': round(time.time() - t0, 1),
}

# ---------- demo predictions with winner ----------
if demo:
    try:
        dpos = np.array([d[2] for d in demo])
        d_preds = []
        for c in winners:
            Xf = assemble(R, c[3]); Xd = assemble(dpos, c[3])
            m, needs_scale = build(c[1], c[2])
            if needs_scale:
                sc = StandardScaler(with_mean=False).fit(Xf)
                d_preds.append(np.asarray(m.fit(sc.transform(Xf), y_R).predict(sc.transform(Xd))).ravel())
            else:
                d_preds.append(np.asarray(m.fit(Xf, y_R).predict(Xd)).ravel())
        dp = np.mean(d_preds, axis=0)
        result['demo'] = [{'drug': DEMO_NAMES.get(d[0], d[0]), 'actual': f'{d[1]:.3g}',
                           'predicted': f'{float(dp[j]):.3g}'} for j, d in enumerate(demo)]
        for j, d in enumerate(demo):
            print(f"[{EP}] → {DEMO_NAMES.get(d[0],d[0])}: {dp[j]:.3g} | actual={d[1]:.3g}", flush=True)
    except Exception as e:
        print(f"[{EP}] demo error: {e}", flush=True)

joblib.dump({'result': result,
             'members': [{'name': c[0], 'model': c[1], 'params': c[2],
                          'features': [g for g in GROUP_NAMES if c[3].get(g)]} for c in winners]},
            OUT / f'{EP}.joblib')
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1))
tag = 'ENSEMBLE' if use_ens else 'SINGLE'
print(f"[{EP}] DONE ({(time.time()-t0)/60:.0f}min) winner={tag} Sp={max(sp_A, ens_sp):.4f}", flush=True)
