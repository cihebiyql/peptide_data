#!/usr/bin/env python3
"""OmniGrid inference service on node23 (port 8901).

Serves ALL trained endpoint models (Codex peptide_modeling campaign) for a query
molecule given (amino acid sequence, SMILES, optional precomputed ESM2 vector).
Feature computation reuses the frozen extractor modules verbatim (engine/) so
every model contract (feature name/order) matches training exactly.

POST /predict  {"sequence": "...", "smiles": "...", "esm2": [...1280 floats...],
                "covariates": {"rbc_species_context:human": 1}, "endpoints": ["MIC", ...]}
GET  /health   GET /endpoints   POST /debug {"route": "...", ...} -> X sha256
"""
import json, sys, time, hashlib, threading, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import joblib

BASE = Path('<WORK_DIR>/s3/omnigrid')
ENGINE = BASE / 'engine'
sys.path.insert(0, str(ENGINE))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

MANIFEST = json.loads((BASE / 'deploy_manifest.json').read_text())
FEAT_MANIFEST = json.loads(Path('/tmp/peptide_modeling_work_multiseed20h_campaign_v1/round_000_seed_42/features/manifest.json').read_text())
GROUP_NAMES = {g: d['feature_names'] for g, d in FEAT_MANIFEST['groups'].items()}
# estate/erg live only in the later overlay manifest; derive canonical names from the engine
if 'estate' not in GROUP_NAMES:
    from peptide_modeling_add_smiles_methods import rdkit_tools as _sm_tools
    _types = _sm_tools()['atom_types']
    GROUP_NAMES['estate'] = [f'count_{t}' for t in _types] + [f'estate_sum_{t}' for t in _types]
    GROUP_NAMES['erg'] = [f'erg_{i:03d}' for i in range(315)]

# Context-covariate defaults for the endpoints shipped with default contexts.
COV_DEFAULTS = {
    'MHC10': {'rbc_species_context:human': 1.0, 'rbc_species_context:mouse': 0.0},
    'hemolysis_HC50': {'rbc_species_context:human': 1.0, 'rbc_species_context:horse': 0.0},
    'CL_renal_per_kg': {'renal_kg_annotation_context:reported_in_vivo_organism_human': 1.0,
                        'renal_kg_annotation_context:reported_in_vivo_organism_rat': 0.0,
                        'renal_kg_annotation_context:reported_kg_basis_unspecified_mask': 0.0,
                        'renal_kg_annotation_context:source_reported_body_weight_annotation': 0.0},
}
MAX_COV = 6  # endpoints with more context columns than this are deferred

MODEL_CACHE, MODEL_LOCK = {}, threading.Lock()
_t0 = time.time()

# ---------------- feature extraction ----------------
def compute_groups(sequence, smiles, esm2_vec):
    groups = {}
    kinds = {}
    if smiles:
        from peptide_modeling_features import extract_structure
        from peptide_modeling_expand_features import extract_expanded_structure
        from peptide_modeling_add_methods import extract_graph_methods
        from peptide_modeling_add_smiles_methods import extract_smiles_methods
        s = extract_structure(smiles)
        if s['group_valid']['morgan']: groups['morgan'] = s['morgan'].astype(np.float32)
        if s['group_valid']['descriptors']: groups['descriptors'] = s['descriptors'].astype(np.float32)
        if s['group_valid']['graph_modifications']: groups['graph_modifications'] = s['graph_modifications'].astype(np.float32)
        e = extract_expanded_structure(smiles)
        for g in ('morgan_r3', 'atompair', 'torsion', 'maccs'):
            if e['valid'].get(g): groups[g] = np.asarray(e['values'][g], dtype=np.float32)
        if e['valid'].get('rdkit_full'):
            full = np.asarray(e['values']['rdkit_full'], dtype=np.float32)
            # training eligibility allows <=10% missing; pipelines carry SimpleImputer,
            # and LightGBM handles NaN natively — pass NaN through.
            if np.isnan(full).mean() <= 0.1: groups['rdkit_full'] = full
        gm = extract_graph_methods(smiles)
        for g in ('rdkit_path', 'pattern', 'layered'):
            if gm['valid'].get(g): groups[g] = np.asarray(gm['values'][g], dtype=np.float32)
        sm = extract_smiles_methods(smiles)
        for g in ('estate', 'erg'):
            if sm['valid'].get(g): groups[g] = np.asarray(sm['values'][g], dtype=np.float32)
        kinds['structure'] = bool(groups)
    if sequence:
        from peptide_modeling_features import extract_sequence
        from peptide_modeling_expand_features import extract_expanded_sequence
        sq = extract_sequence(sequence.strip().upper())
        if sq['valid']: groups['sequence'] = sq['sequence'].astype(np.float32)
        ex = extract_expanded_sequence(sequence.strip().upper())
        if ex.get('valid'):
            groups['dpc'] = np.asarray(ex['dpc'], dtype=np.float32)
            groups['sequence_extra'] = np.asarray(ex['sequence_extra'], dtype=np.float32)
        if esm2_vec is not None and len(esm2_vec) == 1280:
            groups['esm2'] = np.asarray(esm2_vec, dtype=np.float32)
        kinds['sequence'] = bool(groups)
    return groups

def assemble(groups, feature_names, cov_values):
    """Build X (1, n) in exact contract order; returns None if any feature missing."""
    vals = np.zeros(len(feature_names), dtype=np.float32)
    # per-group column index maps
    idx_maps = {}
    for g, names in GROUP_NAMES.items():
        idx_maps[g] = {n: i for i, n in enumerate(names)}
    missing = []
    for j, fn in enumerate(feature_names):
        gv = None
        for g, imap in idx_maps.items():
            if fn.startswith(g + ':') and fn[len(g)+1:] in imap:
                vec = groups.get(g)
                if vec is None or len(vec) != len(imap):
                    missing.append(fn); gv = None; break
                gv = vec[imap[fn[len(g)+1:]]]
                break
        if gv is None:
            if fn in cov_values: gv = np.float32(cov_values[fn])
            elif missing and missing[-1] == fn: continue
            else: missing.append(fn); continue
        vals[j] = gv
    if missing:
        return None, missing[:5]
    return vals.reshape(1, -1), []

def get_model(path):
    with MODEL_LOCK:
        m = MODEL_CACHE.get(path)
        if m is None:
            m = joblib.load(path)
            MODEL_CACHE[path] = m
        return m

def predict_one(model_obj, model_kind_tag, X):
    """Return scalar prediction; classifiers -> positive-class probability."""
    if hasattr(model_obj, 'predict_proba'):
        try: return float(np.asarray(model_obj.predict_proba(X))[:, 1][0])
        except Exception: pass
    return float(np.asarray(model_obj.predict(X))[0])

ESM2_SNAPSHOTS = [
    Path.home() / '.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots',
    Path('<WORK_DIR>/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots'),
]
def compute_esm2_cpu(sequence):
    import torch
    from transformers import AutoTokenizer, AutoModel
    snap_root = None
    for c in ESM2_SNAPSHOTS:
        if c.exists(): snap_root = c; break
    if snap_root is None: return None
    snap = sorted(snap_root.iterdir())[0]
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    model = AutoModel.from_pretrained(str(snap), local_files_only=True)
    model.eval()
    with torch.no_grad():
        enc = tok(sequence, return_tensors='pt', truncation=True, max_length=1022)
        out = model(**enc).last_hidden_state[0].float()
        mask = enc['attention_mask'][0].bool()
        # residue mean excluding special tokens (cls/eos/pad)
        mm = mask.clone(); mm[0] = False
        L = int(mask.sum())
        if L > 1: mm[L-1] = False
        return out[mm].mean(dim=0).numpy().astype(np.float32)

def acylation_anchor(smiles):
    """Literature-anchored half-life expert layer: detect fatty acylation class from SMILES.
    Weekly estimate anchored ONLY on dulaglutide/efpeglenatide/albiglutide labels (120/160/168h);
    the demo drugs are held out of anchoring."""
    import re
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        diacid = 0
        for L in range(22, 10, -1):
            if mol.HasSubstructMatch(Chem.MolFromSmarts('C(=O)' + 'C' * L + 'C(=O)')):
                diacid = L; break
        can = Chem.MolToSmiles(mol)
        mono = max((len(r) for r in re.findall(r'C{10,}', can)), default=0)
        if diacid >= 16:
            return {'class': 'weekly_diacid', 'value': 144.0, 'range': [120, 168],
                    'note': 'C≥16 脂肪二羧酸脂化 → 周给药长效类；锚定: dulaglutide 120h / efpeglenatide 160h / albiglutide 168h (FDA 标签，不含留出药物)'}
        if diacid >= 12:
            return {'class': 'depot_diacid', 'value': 25.0, 'range': [25, 25],
                    'note': 'C12-15 二羧酸 depot 类；锚定: insulin degludec 25h (FDA 标签)'}
        if mono >= 14:
            return {'class': 'daily_acyl', 'value': 9.0, 'range': [5, 13],
                    'note': 'C14-16 单酰化日给药类；锚定: detemir 5-7h / liraglutide 13h (FDA 标签)'}
        return None
    except Exception:
        return None

def run_predict(payload):
    sequence = (payload.get('sequence') or '').strip()
    smiles = (payload.get('smiles') or '').strip()
    esm2_vec = payload.get('esm2')
    cov_over = payload.get('covariates') or {}
    want = set(payload.get('endpoints') or [])
    if not sequence and not smiles:
        return {'error': 'need sequence or smiles'}
    if esm2_vec is None and sequence:
        try: esm2_vec = compute_esm2_cpu(sequence).tolist()
        except Exception: esm2_vec = None
    groups = compute_groups(sequence, smiles, esm2_vec)
    results, skipped = [], []
    for e in MANIFEST:
        ep = e['endpoint']
        if want and ep not in want: continue
        covs = sorted(set(c for s in e['seeds'] for c in s['covariates']))
        cov_values = dict(COV_DEFAULTS.get(ep, {}))
        cov_values.update({k: float(v) for k, v in cov_over.items()})
        if len(covs) > MAX_COV and not all(c in cov_values for c in covs):
            skipped.append({'endpoint': ep, 'reason': 'conditional_endpoint_pending'})
            continue
        seed_preds, seed_infos, miss = [], [], None
        for s in e['seeds']:
            X, missing = assemble(groups, s['feature_names'], cov_values)
            if X is None:
                miss = missing; continue
            try:
                model_obj = get_model(s['model_path'])
                # classification models report probability; regressors raw (maybe log10)
                is_cls = e['kind'] == 'classification'
                v = predict_one(model_obj, s['model'], X)
                seed_preds.append(v)
                seed_infos.append({'seed': s['seed'], 'route': s['route'], 'model': s['model'],
                                   'value_model_space': v})
            except Exception as ex:
                seed_infos.append({'seed': s['seed'], 'error': str(ex)[:120]})
        if not seed_preds:
            skipped.append({'endpoint': ep, 'reason': 'features_missing', 'detail': miss})
            continue
        arr = np.array(seed_preds)
        mean = float(arr.mean())
        # inverse transform: log10 -> original scale for regressors
        s0 = e['seeds'][0]
        tr = s0.get('transform') or e.get('transform') or ''
        inv, unit = mean, s0.get('unit') or e.get('unit') or ''
        if e['kind'] == 'regression' and tr == 'log10':
            try:
                inv = float(10 ** mean)
            except OverflowError:
                inv = None  # extreme out-of-domain extrapolation
        results.append({
            'endpoint': ep, 'kind': e['kind'],
            'task_definition': e.get('task_definition', ''),
            'value': (float(f'{inv:.4g}') if inv is not None else None),
            'value_model_space': float(f'{mean:.4g}'),
            'sd_seeds': round(float(arr.std()), 4) if len(arr) > 1 else 0.0,
            'n_seeds': len(arr), 'unit': unit if e['kind'] == 'regression' else 'probability',
            'transform': tr, 'n_train_records': s0.get('record_count'),
            'best_cv': (s0.get('cv') or {}).get('spearman') or (s0.get('cv') or {}).get('mae'),
            'seeds_detail': seed_infos,
        })
    results.sort(key=lambda r: (r['kind'], r['endpoint']))
    # mechanism-anchored half-life expert layer (SMILES-based, literature-anchored)
    if smiles:
        anch = acylation_anchor(smiles)
        if anch:
            results.insert(0, {
                'endpoint': 'half_life_anchor', 'kind': 'mechanism',
                'task_definition': '机制锚定半衰期（脂肪酰化类 → 文献锚定范围）',
                'value': float(f"{anch['value']:.4g}"),
                'range_h': anch['range'], 'acylation_class': anch['class'],
                'note': anch['note'],
                'unit': 'h (文献锚定范围)', 'n_seeds': 0, 'sd_seeds': 0.0,
                'n_train_records': '文献类锚定', 'best_cv': None,
                'seeds_detail': [],
            })
    return {'n_ok': len(results), 'predictions': results, 'skipped': skipped,
            'groups_computed': sorted(groups.keys()), 'service_time_s': round(time.time()-_t0, 1)}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == '/health':
            self._json(200, {'ok': True, 'models_cached': len(MODEL_CACHE),
                             'entries': len(MANIFEST),
                             'endpoints': len(set(e['endpoint'] for e in MANIFEST))})
        elif self.path == '/endpoints':
            eps = []
            for e in MANIFEST:
                covs = sorted(set(c for s in e['seeds'] for c in s['covariates']))
                eps.append({'endpoint': e['endpoint'], 'kind': e['kind'], 'unit': e.get('unit'),
                            'n_seeds': len(e['seeds']), 'n_cov': len(covs),
                            'task': (e.get('task_definition') or '')[:80]})
            self._json(200, eps)
        else:
            self._json(404, {'error': 'not found'})
    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(n) or b'{}')
        except Exception as ex:
            return self._json(400, {'error': f'bad json: {ex}'})
        if self.path == '/predict':
            return self._json(200, run_predict(payload))
        if self.path == '/debug':
            groups = compute_groups((payload.get('sequence') or '').strip(),
                                    (payload.get('smiles') or '').strip(), payload.get('esm2'))
            route = payload.get('route', '')
            gnames = FEAT_MANIFEST['routes'].get(route)
            if not gnames: return self._json(404, {'error': 'unknown route'})
            names, parts = [], []
            for g in gnames:
                names.extend(f'{g}:{n}' for n in GROUP_NAMES[g])
                v = groups.get(g)
                parts.append(v if v is not None else np.full(len(GROUP_NAMES[g]), np.nan, np.float32))
            X = np.hstack(parts).astype(np.float32)
            return self._json(200, {'route': route, 'n_features': X.shape[0],
                'matrix_sha256': hashlib.sha256(X.tobytes()).hexdigest(),
                'n_nan': int(np.isnan(X).sum())})
        self._json(404, {'error': 'not found'})

if __name__ == '__main__':
    print(f'omnigrid service: {len(MANIFEST)} entries, '
          f'{len(set(e["endpoint"] for e in MANIFEST))} endpoints', flush=True)
    ThreadingHTTPServer(('0.0.0.0', 8901), Handler).serve_forever()
