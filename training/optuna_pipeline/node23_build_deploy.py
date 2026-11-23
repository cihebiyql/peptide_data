#!/usr/bin/env python3
"""Build the deployment manifest for the omnigrid inference service on node23. v2

Fixes vs v1: normalize Windows backslash paths; estate/erg recognized as feature
groups; do not trust TSV refit_status (read result JSON instead); compute
training-mode defaults for one-hot context covariate columns.
Output: <WORK_DIR>/s3/omnigrid/deploy_manifest.json
"""
import csv, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter

csv.field_size_limit(10_000_000)
SRC = Path('<WORK_DIR>/s3/representative_seed_models.tsv')
OUT = Path('<WORK_DIR>/s3/omnigrid')
OUT.mkdir(parents=True, exist_ok=True)
KNOWN_GROUPS = ['morgan','descriptors','graph_modifications','sequence','esm2','morgan_r3',
                'atompair','torsion','maccs','rdkit_full','dpc','sequence_extra',
                'rdkit_path','pattern','layered','estate','erg']
TMP_BASE = '/tmp/peptide_modeling_work_'
DATA_BASE = '<WORK_DIR>/peptide_modeling_multiseed20h_v1/'

def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

def norm(p):
    return p.replace('\\','/').strip() if p else p

def persistent(p):
    p = norm(p)
    if p.startswith(TMP_BASE):
        rest = p[len(TMP_BASE):]
        parts = rest.split('/', 1)
        if len(parts) == 2:
            mapped = DATA_BASE + parts[0] + '/' + parts[1]
            if Path(mapped).exists(): return mapped
    return p

rows = list(csv.DictReader(open(SRC, encoding='utf-8'), delimiter='\t'))
groups = defaultdict(list)
for r in rows:
    groups[(r['endpoint'], r['task_id'])].append(r)
print(f'rows: {len(rows)} | endpoint-task groups: {len(groups)} | endpoints: {len(set(k[0] for k in groups))}')

manifest, errors = [], []
for (ep, task_id), seeds in sorted(groups.items()):
    entry = {'endpoint': ep, 'task_id': task_id, 'kind': seeds[0]['task_kind'],
             'unit': seeds[0].get('unit',''), 'transform': seeds[0].get('target_transform',''),
             'task_definition': (seeds[0].get('task_definition') or '')[:300], 'seeds': []}
    for r in sorted(seeds, key=lambda x: int(x['seed'])):
        seed = r['seed']
        rp = persistent(r['result_path'])
        try:
            res = json.loads(Path(rp).read_text())
        except Exception as e:
            errors.append(f'{ep}/{seed}: read {rp}: {type(e).__name__}'); continue
        st = res.get('refit_status')
        if st is None: st = res.get('status')
        if st != 'completed':
            errors.append(f'{ep}/{seed}: status {st}'); continue
        art = res.get('artifacts', {})
        model_p = persistent(art.get('refit_model','') or art.get('model','') or '')
        if not model_p or not Path(model_p).exists():
            errors.append(f'{ep}/{seed}: model missing'); continue
        contract_p = persistent(art.get('model_contract','') or '')
        contract = {}
        if contract_p and Path(contract_p).exists():
            try: contract = json.loads(Path(contract_p).read_text())
            except Exception: pass
        fns = contract.get('feature_names') or []
        cov = [n for n in fns if not any(n.startswith(g+':') for g in KNOWN_GROUPS)]
        entry['seeds'].append({
            'seed': int(seed), 'model_path': model_p,
            'route': res.get('selected_route') or r.get('selected_route',''),
            'model': res.get('selected_model') or r.get('selected_model',''),
            'params': res.get('selected_params') or {},
            'feature_names': fns, 'n_features': len(fns),
            'transform': contract.get('target_transform',''),
            'unit': contract.get('target_unit',''),
            'inverse': contract.get('task',{}).get('inverse_transform',''),
            'cv': res.get('inner_cv'),
            'holdout': res.get('holdout_metrics_display_only'),
            'record_count': contract.get('task',{}).get('n_records'),
            'covariates': cov,
            'table_path': norm(contract.get('task',{}).get('table_path','')),
        })
    if entry['seeds']:
        manifest.append(entry)

# covariate mode defaults from task tables (one-hot ctx columns -> training mode)
print('\ncomputing covariate defaults...')
for e in manifest:
    covs = set()
    for s in e['seeds']: covs.update(s['covariates'])
    if not covs: continue
    s0 = e['seeds'][0]
    tp = s0['table_path']
    if not tp: e['covariate_defaults'] = {}; continue
    # walk ancestors of the model path to locate the run root containing the table
    table = None
    mp = Path(s0['model_path'])
    for anc in [mp] + list(mp.parents):
        for cand in (anc / tp, anc.parent / tp, anc.parent.parent / tp):
            if cand.exists(): table = cand; break
            mapped = Path(persistent(str(cand)))
            if mapped.exists(): table = mapped; break
        if table: break
    defaults = {}
    if table and table.exists():
        fam_cols = defaultdict(Counter)
        with open(table, encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                for c in covs:
                    if c in r:
                        fam_cols[c][r[c]] += 1
        for c in covs:
            if fam_cols[c]:
                defaults[c] = fam_cols[c].most_common(1)[0][0]
    e['covariate_defaults'] = defaults
    if covs:
        print(f"  {e['endpoint']}: {len(covs)} cov cols, defaults={defaults}")

n_models = sum(len(e['seeds']) for e in manifest)
print(f'\nmanifest entries: {len(manifest)} | endpoints: {len(set(e["endpoint"] for e in manifest))} | seed models: {n_models}')
print(f'errors: {len(errors)}')
for er in errors[:8]: print(' ', er)

(OUT / 'deploy_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
print('saved:', OUT / 'deploy_manifest.json')
