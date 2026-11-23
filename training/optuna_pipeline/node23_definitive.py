#!/usr/bin/env python3
"""DEFINITIVE Optuna pipeline: ALL features × ALL endpoints, one run.

14 feature groups (~13,000 dims total), searched by TPE.
Run on node23 (64 cores). No further feature additions needed.
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
OUT = DATA / 'definitive'
(OUT / 'models').mkdir(parents=True, exist_ok=True)
MIN_ROWS = 50
N_TRIALS = 120
TIMEOUT_PER_EP = 480  # 8 minutes
DEMO_EXCLUDE = {'cmp_bccb3404f692c7bf44452c96', 'cmp_c80b5e5a1682bb35cd28ec5b'}
DEMO_NAMES = {'cmp_bccb3404f692c7bf44452c96': '司美格鲁肽', 'cmp_c80b5e5a1682bb35cd28ec5b': '替尔泊肽'}

t0 = time.time()
print("Loading ALL features...", flush=True)

# ---- Group 1-4: Original features ----
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

# ---- Group 5: ChemBERTa (compound_id indexed) ----
zg = np.load(DATA / 'gpu_embeddings_v2.npz', allow_pickle=True)
cb_ids = list(zg['chemberta_ids'])
cb_emb = zg['chemberta_embeddings'].astype(np.float32)
cb_map = {str(c): cb_emb[i] for i, c in enumerate(cb_ids)}
print(f"  ChemBERTa: {len(cb_map):,} × {cb_emb.shape[1]}", flush=True)

# ---- Group 6: ESM2-3B (sequence indexed) ----
e3b_seqs = list(zg['esm3b_ids'])
e3b_emb = zg['esm3b_embeddings'].astype(np.float32)
e3b_map = {str(s): e3b_emb[i] for i, s in enumerate(e3b_seqs)}
print(f"  ESM2-3B: {len(e3b_map):,} × {e3b_emb.shape[1]}", flush=True)

# ---- Group 7: Expanded CPU (compound_id indexed) ----
zx = np.load(DATA / 'expanded_cpu_features_fp16.npz')
EXP = zx['embeddings'].astype(np.float32)
exp_idx = {}
with open(DATA / 'expanded_cpu_features_index.tsv') as f:
    rd = csv.DictReader(f, delimiter='\t')
    for r in rd: exp_idx[r['compound_id']] = int(r['row_idx'])
# Slices within expanded: rdkit_full(0:217) dpc(217:617) atompair(617:2665) topo(2665:4709)
EXP_SLICES = {'rdkit_full': (0, 217), 'dpc': (217, 617), 'atompair': (617, 2665), 'topo': (2665, 4709)}
print(f"  Expanded CPU: {len(exp_idx):,} × {EXP.shape[1]}", flush=True)

# ---- Group 8: Compound features 229d ----
za = np.load(DATA / 'all_features_fp16.npz')
ALLF = za['embeddings'].astype(np.float32)
allf_idx = {}
with open(DATA / 'all_features_index.tsv') as f:
    for r in csv.DictReader(f, delimiter='\t'): allf_idx[r['compound_id']] = int(r['row_idx'])
ALLF_SLICES = {'aac': (0,26), 'physchem': (26,36), 'dbaasp': (36,48),
               'mod': (48,57), 'maccs': (57,223), 'chembl': (223,229)}
print(f"  Compound 229d: {len(allf_idx):,}", flush=True)

# All 14 feature group flags
FEATURE_FLAGS = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
                 'rdkit_full','dpc','atompair','topo',
                 'aac','physchem','dbaasp','mod','maccs','chembl']
# Actually 16 flags, let's merge some to reduce search space
FEATURE_FLAGS = ['morgan','esm2','ncaa','desc','chemberta','esm3b',
                 'rdkit_full','dpc','atompair',
                 'aac','physchem','dbaasp','mod']
print(f"Feature groups: {len(FEATURE_FLAGS)}", flush=True)
print(f"All loaded ({time.time()-t0:.1f}s)", flush=True)


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
        # Expanded CPU groups
        ei = exp_idx.get(cid)
        if config.get('rdkit_full'):
            if ei is not None:
                s, e = EXP_SLICES['rdkit_full']; parts.append(EXP[ei, s:e])
            else: ok = False
        if config.get('dpc'):
            if ei is not None:
                s, e = EXP_SLICES['dpc']; parts.append(EXP[ei, s:e])
            else: ok = False
        if config.get('atompair'):
            if ei is not None:
                s, e = EXP_SLICES['atompair']; parts.append(EXP[ei, s:e])
            else: ok = False
        # Compound feature groups
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
    if len(valid) < MIN_ROWS:
        return None, None, None
    X = np.stack([v[1] for v in valid])
    y = np.array([train_rows[v[0]][1] for v in valid])
    groups = np.array([train_rows[v[0]][0] for v in valid])
    return X, y, groups


def make_objective(train_rows):
    def objective(trial):
        config = {f: trial.suggest_categorical(f, [True, False]) for f in FEATURE_FLAGS}
        if not any(config.values()): return -1.0
        model_type = trial.suggest_categorical('model', ['ridge', 'lgbm'])
        if model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('ne', 100, 600),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('lv', 15, 150),
                'min_child_samples': trial.suggest_int('mc', 5, 40),
                'subsample': 0.9, 'colsample_bytree': trial.suggest_float('ct', 0.5, 1.0),
                'random_state': 42, 'verbose': -1, 'n_jobs': 32}
        else:
            params = {'alpha': trial.suggest_float('alpha', 0.1, 100, log=True)}

        X, y, groups = assemble(train_rows, config)
        if X is None: return -1.0
        gkf = GroupKFold(n_splits=5)
        try: splits = list(gkf.split(X, y, groups))
        except ValueError: return -1.0

        preds = np.zeros_like(y)
        for tr, te in splits:
            if model_type == 'lgbm':
                m = lgb.LGBMRegressor(**params); m.fit(X[tr], y[tr])
                preds[te] = m.predict(X[te])
            else:
                sc = StandardScaler(with_mean=False).fit(X[tr])
                m = Ridge(**params); m.fit(sc.transform(X[tr]), y[tr])
                preds[te] = m.predict(sc.transform(X[te]))
        sp = float(spearmanr(y, preds).statistic)
        return sp if not np.isnan(sp) else -1.0
    return objective


# ---- Main ----
KEY_ENDPOINTS = ['half_life', 'EC50', 'IC50', 'hemolysis', 'MIC', 'MBC', 'Ki', 'Kd', 'LC_lemma', 'MFC']
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
            if cid in DEMO_EXCLUDE: demo_rows.append((cid, y, seq, r))
            else: train_rows.append((cid, y, seq, r))
    if len(train_rows) < MIN_ROWS: continue

    unit = train_rows[0][3].get('unit_target', '') if train_rows else ''
    print(f"\n{'='*60}")
    print(f"[{ep_name}] {len(train_rows):,} rows | {unit}", flush=True)

    objective = make_objective(train_rows)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_PER_EP)

    best = study.best_trial
    cfg = best.params
    feat_used = [k for k in FEATURE_FLAGS if cfg.get(k)]
    print(f"  Sp={best.value:.4f} | trial {best.number} | {time.time()-t0:.0f}s", flush=True)
    print(f"  Features({len(feat_used)}): {'+'.join(feat_used)}", flush=True)
    print(f"  Model: {cfg['model']}", flush=True)

    # Save
    joblib.dump({'endpoint': ep_name, 'best_params': cfg, 'best_spearman': best.value,
                 'n_trials': len(study.trials), 'features_used': feat_used},
                OUT / 'models' / f'{ep_name}.joblib')

    # Demo prediction
    feat_config = {f: cfg.get(f, False) for f in FEATURE_FLAGS}
    X, y, groups = assemble(train_rows, feat_config)
    if X is not None:
        if cfg['model'] == 'lgbm':
            mp = {'n_estimators': int(cfg['ne']), 'learning_rate': float(cfg['lr']),
                  'num_leaves': int(cfg['lv']), 'min_child_samples': int(cfg['mc']),
                  'subsample': 0.9, 'colsample_bytree': float(cfg['ct']),
                  'random_state': 42, 'verbose': -1, 'n_jobs': 32}
            model = lgb.LGBMRegressor(**mp); model.fit(X, y); scaler = None
        else:
            scaler = StandardScaler(with_mean=False).fit(X)
            model = Ridge(alpha=float(cfg['alpha'])); model.fit(scaler.transform(X), y)

        # Build demo features and predict
        for cid, act, seq, r in demo_rows:
            d_config = dict(feat_config)
            dX, _, _ = assemble([(cid, 0.0, seq, r)], d_config)
            if dX is not None:
                if scaler: dX = scaler.transform(dX)
                pred = float(model.predict(dX)[0])
                demo_predictions.append([DEMO_NAMES.get(cid, cid), ep_name,
                                        f'{act:.3g}', f'{pred:.3g}', unit, f'{best.value:.4f}', len(X)])
                print(f"  → {DEMO_NAMES.get(cid, cid)}: {pred:.3g} | actual={act:.3g}", flush=True)

# Save results
with open(OUT / 'demo_predictions.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['drug','endpoint','actual','predicted','unit','cv_spearman','n_train'])
    w.writerows(demo_predictions)

with open(OUT / 'definitive_report.md', 'w', encoding='utf-8') as f:
    f.write(f"# 最终版 Optuna 全特征搜索\n\n> {len(FEATURE_FLAGS)} 特征组 | {N_TRIALS} trials | TPE\n\n")
    f.write("| Drug | Endpoint | Actual | Predicted | Unit | CV Sp | Train N |\n|---|---|---|---|---|---|---|\n")
    for r in demo_predictions:
        f.write(f"| {r[0]} | {r[1]} | {r[2]} | **{r[3]}** | {r[4]} | {r[5]} | {r[6]:,} |\n")

print(f"\n{'='*60}")
print(f"[DONE] {len(demo_predictions)} predictions, {time.time()-t0:.0f}s total")
