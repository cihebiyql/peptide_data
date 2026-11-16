#!/usr/bin/env python3
"""v2: 机制特征版锚定模型。特征=最长脂肪链/末端羧酸数/醚键(PEG)数/MW/LogP/RotB。
10 个已发表锚点 LOO 验证，司美格鲁肽/替尔泊肽留出外推。"""
import sys, json, re
import numpy as np
sys.path.insert(0, '/data/qlyu/s3/omnigrid/engine')
from peptide_modeling_features import extract_structure
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
import joblib

smiles_map = json.load(open('/data/qlyu/s3/omnigrid/anchor_smiles.json'))
qf = json.load(open('/data/qlyu/s3/omnigrid/query_features.json'))
ANCHORS = [  # (name, pubchem_name, t1/2 h)
    ('GLP-1', 'glucagon-like peptide 1', 0.033), ('tesamorelin', 'tesamorelin', 0.43),
    ('exenatide', 'exenatide', 2.4), ('teduglutide', 'teduglutide', 2.0),
    ('lixisenatide', 'lixisenatide', 3.0), ('detemir', 'insulin detemir', 6.0),
    ('glargine', 'insulin glargine', 12.0), ('setmelanotide', 'setmelanotide', 11.0),
    ('liraglutide', 'liraglutide', 13.0), ('degludec', 'insulin degludec', 25.0)]
DRUGS = {'semaglutide': (168.0, qf['query'][0]['canonical_smiles']),
         'tirzepatide': (120.0, qf['query'][1]['canonical_smiles'])}

def mech_features(smiles):
    s = extract_structure(smiles)
    if not s['valid']: return None
    desc = s['descriptors']  # MolWt ExactMolWt MolLogP TPSA HBD HBA RotB HeavyAtoms Atoms Rings AromRings AliphRings FracCSP3 Charge
    mol_logp, mw, rotb, tpsa = float(desc[2]), float(desc[0]), float(desc[6]), float(desc[3])
    # longest linear aliphatic carbon run in canonical SMILES
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    can = Chem.MolToSmiles(mol)
    runs = re.findall(r'C{4,}', can)
    max_chain = max((len(r) for r in runs), default=0)
    # terminal carboxylic acids (free COOH)
    cooh = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[OX2H1]')))
    # ether oxygens (PEG signature)
    ether = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][OX2][#6]')))
    return np.array([min(max_chain, 22) / 22.0, cooh > 0, ether / 10.0,
                     np.log10(mw) / 4.0, mol_logp / 10.0, rotb / 60.0, tpsa / 800.0],
                    dtype=np.float64)

X, y, names = [], [], []
for name, pub, th in ANCHORS:
    f = mech_features(smiles_map[pub]['smiles'])
    if f is None: continue
    X.append(f); y.append(np.log10(th)); names.append(name)
X, y = np.vstack(X), np.array(y)
print('features: [chain, cooh>0, ether, logMW, logP, rotB, tpsa]')
for n, f in zip(names, X): print(f'  {n:<14}', np.round(f, 3))

best = None
for alpha in (0.03, 0.1, 0.3, 1, 3, 10, 30, 100):
    preds = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    mae = float(np.abs(preds - y).mean())
    r2 = 1 - ((preds - y)**2).sum() / ((y - y.mean())**2).sum()
    spear = float(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(preds, y).statistic)
    print(f'alpha={alpha:<6} LOO MAE={mae:.3f} R2={r2:.3f} Sp={spear:.3f}')
    if best is None or mae < best[0]: best = (mae, r2, spear, alpha, preds.copy())
mae, r2, spear, alpha, loo = best
print(f'\nBEST alpha={alpha}')
print('LOO 预测 vs 实际:')
for n, lp, yv in zip(names, loo, y):
    print(f'  {n:<14} {10**lp:8.2f} h vs {10**yv:8.2f} h')
m = Ridge(alpha=alpha).fit(X, y)
print('\n=== 留出外推 ===')
drug_preds = {}
for dn, (actual, smi) in DRUGS.items():
    f = mech_features(smi)
    p = float(m.predict(f.reshape(1, -1))[0])
    drug_preds[dn] = {'pred_h': 10**p, 'actual_h': actual}
    print(f'  {dn:<12} {10**p:8.1f} h vs actual {actual:.0f} h')

joblib.dump({'coef': m.coef_.tolist(), 'intercept': float(m.intercept_), 'alpha': alpha,
             'feature_names': ['max_chain22', 'has_cooh', 'ether10', 'logMW4', 'logP10', 'rotB60', 'tpsa800'],
             'loo_mae': mae, 'loo_r2': r2, 'loo_spearman': spear,
             'anchors': names, 'anchor_actual_h': [10**v for v in y],
             'anchor_loo_pred_h': [10**v for v in loo], 'drug_holdout': drug_preds},
            '/data/qlyu/s3/omnigrid/halflife_anchor.joblib')
print('saved')
