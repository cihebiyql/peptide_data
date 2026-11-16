#!/usr/bin/env python3
"""长效肽半衰期锚定模型：以 10 个已发表 t1/2 的合成肽训练（FDA 标签值），
留一法验证，外推司美格鲁肽/替尔泊肽（不参与训练）。
特征 = Codex 冻结引擎的 descriptors+graph_modifications+estate+erg（SMILES 全信息）。"""
import sys, json, time
import numpy as np
from pathlib import Path

sys.path.insert(0, '/data/qlyu/s3/omnigrid/engine')
from peptide_modeling_features import extract_structure
from peptide_modeling_add_smiles_methods import extract_smiles_methods
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

ANCHORS = {  # name -> (t1/2 hours, SMILES)
    'GLP-1(7-36)': (0.033, None), 'tesamorelin': (0.43, None), 'exenatide': (2.4, None),
    'teduglutide': (2.0, None), 'lixisenatide': (3.0, None), 'detemir': (6.0, None),
    'glargine': (12.0, None), 'setmelanotide': (11.0, None), 'liraglutide': (13.0, None),
    'degludec': (25.0, None),
}
smiles_map = json.load(open('/data/qlyu/s3/omnigrid/anchor_smiles.json'))
NAME2PUBCHEM = {'GLP-1(7-36)': 'glucagon-like peptide 1', 'tesamorelin': 'tesamorelin',
                'exenatide': 'exenatide', 'teduglutide': 'teduglutide',
                'lixisenatide': 'lixisenatide', 'detemir': 'insulin detemir',
                'glargine': 'insulin glargine', 'setmelanotide': 'setmelanotide',
                'liraglutide': 'liraglutide', 'degludec': 'insulin degludec'}
qf = json.load(open('/data/qlyu/s3/omnigrid/query_features.json'))
DRUGS = {'semaglutide': (168.0, qf['query'][0]['canonical_smiles']),
         'tirzepatide': (120.0, qf['query'][1]['canonical_smiles'])}

def featurize(smiles):
    s = extract_structure(smiles)
    m = extract_smiles_methods(smiles)
    if not (s['valid'] and m['valid'].get('estate') and m['valid'].get('erg')):
        return None
    return np.concatenate([s['descriptors'], s['graph_modifications'],
                           np.asarray(m['values']['estate'], np.float32),
                           np.asarray(m['values']['erg'], np.float32)])

X, y, names = [], [], []
for name, (th, _) in ANCHORS.items():
    smi = smiles_map[NAME2PUBCHEM[name]]['smiles']
    f = featurize(smi)
    if f is None: print(f'!! {name} featurize failed'); continue
    X.append(f); y.append(np.log10(th)); names.append(name)
X = np.vstack(X); y = np.array(y)
print(f'anchors: {len(y)} | dim {X.shape[1]}')

# LOO model selection over alpha and feature blocks
blocks = {'desc': (0, 14), 'graph': (14, 19), 'estate': (19, 177), 'erg': (177, 492)}
combos = [('desc+estate+erg', list(range(0,14)) + list(range(19,492))),
          ('desc+graph+estate+erg', list(range(0,492))),
          ('desc+estate', list(range(0,14)) + list(range(19,177))),
          ('desc', list(range(0,14)))]
best = None
for cname, cols in combos:
    Xc = X[:, cols]
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    Xz = (Xc - mu) / sd
    for alpha in (0.1, 0.3, 1, 3, 10, 30, 100, 300):
        preds = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(Xz):
            m = Ridge(alpha=alpha).fit(Xz[tr], y[tr])
            preds[te] = m.predict(Xz[te])
        mae = float(np.abs(preds - y).mean())
        r2 = 1 - ((preds - y)**2).sum() / ((y - y.mean())**2).sum()
        if best is None or mae < best[0]:
            best = (mae, r2, cname, alpha, cols, mu, sd, preds)
mae, r2, cname, alpha, cols, mu, sd, loo_preds = best
print(f'BEST LOO: {cname} alpha={alpha} | MAE={mae:.3f} log10 | R2={r2:.3f}')
print('LOO 预测 vs 实际 (h):')
for n, lp, yv in zip(names, loo_preds, y):
    print(f'  {n:<15} pred {10**lp:8.2f} h | actual {10**yv:8.2f} h')

# full fit -> predict held-out drugs
m = Ridge(alpha=alpha).fit((X[:, cols] - mu) / sd, y)
print('\n=== 留出外推（未参与训练） ===')
drug_preds = {}
for dn, (actual, smi) in DRUGS.items():
    f = featurize(smi)
    p = float(m.predict(((f[cols] - mu) / sd).reshape(1, -1))[0])
    drug_preds[dn] = {'pred_h': 10**p, 'actual_h': actual, 'pred_log10': p}
    print(f'  {dn:<12} pred {10**p:8.1f} h | actual {actual:.0f} h | ratio {actual/10**p:.1f}x')

# save model artifacts for the service
import joblib
art = {'cols': cols, 'mu': mu, 'sd': sd, 'alpha': alpha, 'block_name': cname,
       'loo_mae': mae, 'loo_r2': r2, 'anchors': names,
       'anchor_actual_h': [10**v for v in y], 'anchor_loo_pred_h': [10**v for v in loo_preds],
       'drug_holdout': drug_preds,
       'coef': m.coef_.tolist(), 'intercept': float(m.intercept_)}
joblib.dump(art, '/data/qlyu/s3/omnigrid/halflife_anchor.joblib')
print('\nsaved /data/qlyu/s3/omnigrid/halflife_anchor.joblib')
