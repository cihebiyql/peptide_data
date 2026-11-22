import sqlite3, json, re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def clean(value):
    value = str(value or "").strip()
    return "" if value.lower() in {"", "\\n", "none", "unknown", "not reported"} else value

def scalar(value, qualifier=""):
    value = clean(value).replace("≤", "<=").replace("≥", ">=").replace("≈", "~")
    m = re.fullmatch(r"\s*(<=|>=|<|>|=|~)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", value)
    if not m:
        return None
    prefix, num = m.groups()
    q = clean(qualifier)
    if q and q not in {"<", ">", "<=", ">=", "=", "~"}:
        return None
    if prefix and q and prefix != q:
        return None
    try:
        return (prefix or q or "=", Decimal(num).normalize())
    except InvalidOperation:
        return None

def classify_species(name):
    if not name:
        return None
    s = name.lower()
    if "escherichia coli" in s or "e. coli" in s or "e.coli" in s:
        return "MIC_E_coli"
    if "staphylococcus aureus" in s or "s. aureus" in s or "s.aureus" in s:
        return "MIC_S_aureus"
    if "pseudomonas aeruginosa" in s or "p. aeruginosa" in s or "p.aeruginosa" in s:
        return "MIC_P_aeruginosa"
    if "candida albicans" in s or "c. albicans" in s or "c.albicans" in s:
        return "MIC_C_albicans"
    return None

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. CHEMBL MIC species from condition_json
    print("=== 1. CHEMBL MIC SPECIES ===")
    chembl_sp = Counter()
    chembl_lane = Counter()
    for r in cur.execute("""
        SELECT lane, condition_json, structure_model_eligible, sequence_kind
        FROM records_with_structure_qc
        WHERE endpoint='MIC' AND source_code='CHEMBL' AND lane IN ('strict_numeric','censored_numeric')
    """):
        cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        assay = cj.get('semantics', {}).get('assay', {})
        org = assay.get('system_description') or assay.get('description') or ''
        sp = classify_species(org)
        if sp:
            chembl_sp[sp] += 1
            chembl_lane[(sp, r['lane'])] += 1
    print("  CHEMBL species totals:", dict(chembl_sp))
    print("  CHEMBL by lane:", dict(chembl_lane))

    # 2. DBAASP MIC exact multiset bridge (including censored_numeric!)
    print("\n=== 2. DBAASP MIC EXACT MULTISET BRIDGE ===")
    # Load all DBAASP observations (all lanes for scope completeness!)
    scopes = defaultdict(list)
    for r in cur.execute("""
        SELECT record_id, endpoint, lane, compound_id, assay_id, raw_value, raw_relation, raw_unit, value, unit, relation, raw_json
        FROM records
        WHERE source_code='DBAASP' AND origin='NEW'
    """):
        rj = json.loads(r['raw_json']) if r['raw_json'] else {}
        ov = rj.get('original_value') or r['raw_value'] or r['value']
        oq = rj.get('qualifier') or r['raw_relation'] or r['relation']
        ou = rj.get('original_unit') or r['raw_unit'] or r['unit']
        scopes[(r['assay_id'], r['compound_id'])].append({
            'record_id': r['record_id'],
            'endpoint': r['endpoint'],
            'raw_endpoint': rj.get('endpoint', ''),
            'lane': r['lane'],
            'sig': scalar(ov, oq),
            'unit': clean(ou).lower().replace('µ','u').replace('μ','u').replace(' ',''),
            'value': r['value'],
            'relation': r['relation'],
        })
    print(f"  Loaded {len(scopes):,} DBAASP (assay_id, compound_id) scopes")

    indexed = defaultdict(lambda: defaultdict(dict))
    for compound, assay, param, val in cur.execute("""
        SELECT compound_id, assay_id, parameter, value
        FROM condition_evidence
        WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT'
    """):
        m = re.fullmatch(r"\[(\d+)\]\.activity\.(concentration|targetSpecies\.name|medium\.name)", param)
        if not m:
            continue
        idx, field = m.groups()
        if (assay, compound) in scopes:
            indexed[(assay, compound)][idx][field] = val

    dbaasp_sp = Counter()
    dbaasp_lane = Counter()
    dbaasp_ambig_same_sp = Counter()
    complete_scopes = 0
    for key, obs_list in scopes.items():
        entries = indexed.get(key)
        if not entries:
            continue
        orig_cnt = Counter(o['sig'] for o in obs_list)
        ctx_sigs = [(idx, scalar(e.get('concentration')), e) for idx, e in entries.items()]
        ctx_cnt = Counter(s for _, s, _ in ctx_sigs)
        eps = {o['raw_endpoint'] for o in obs_list}
        units = {o['unit'] for o in obs_list}
        if None in orig_cnt or None in ctx_cnt or orig_cnt != ctx_cnt:
            continue
        if len(eps) != 1 or len(units) != 1 or '' in units:
            continue
        complete_scopes += 1
        # Map signature -> list of context entries
        sig_to_ctx = defaultdict(list)
        for idx, s, e in ctx_sigs:
            sig_to_ctx[s].append(e)
        for o in obs_list:
            if o['endpoint'] != 'MIC' or o['lane'] not in ('strict_numeric', 'censored_numeric'):
                continue
            cands = sig_to_ctx[o['sig']]
            if len(cands) == 1:
                sp = classify_species(cands[0].get('targetSpecies.name', ''))
                if sp:
                    dbaasp_sp[sp] += 1
                    dbaasp_lane[(sp, o['lane'])] += 1
            else:
                # Check if ALL candidates with same signature have the SAME species!
                sps = {classify_species(c.get('targetSpecies.name', '')) for c in cands}
                if len(sps) == 1 and None not in sps:
                    sp = next(iter(sps))
                    dbaasp_ambig_same_sp[(sp, o['lane'])] += 1

    print(f"  Complete single-endpoint/unit scopes: {complete_scopes:,}")
    print("  DBAASP unique-sig species:", dict(dbaasp_sp))
    print("  DBAASP unique-sig by lane:", dict(dbaasp_lane))
    print("  DBAASP multi-sig SAME-species bonus:", dict(dbaasp_ambig_same_sp))

    # 3. Check OLD DBAASP_v4.0.1 / Hemolytik2_2026 / hemopi2_2025 / dramp_v4 hemolysis condition_json structure
    print("\n=== 3. HEMOLYSIS REVIEW STRUCTURE AUDIT ===")
    hemo_stats = Counter()
    for r in cur.execute("""
        SELECT endpoint, origin, source_code, lane, reason, unit, relation, value, condition_json, structure_model_eligible, sequence_kind
        FROM records_with_structure_qc
        WHERE endpoint IN ('hemolysis','hemolysis_percent','hemolysis_HC50','HC50','hemolysis_EC50','hemolysis_LC50','hemolysis_MHC','hemolysis_MHC10','MHC','MHC10','HC10','hemolysis_IC50','hemolysis_SOURCE_DEFINED_BINARY','Hemolysis:Source Training Class')
          AND lane IN ('review','censored_numeric','strict_numeric','binary_evidence')
    """):
        cj = {}
        try:
            cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        except Exception:
            pass
        sem = cj.get('semantics', {})
        ed = sem.get('endpoint_detail', '')
        dose = sem.get('dose', '')
        has_dose = bool(dose)
        hemo_stats[(r['endpoint'], r['source_code'], r['lane'], r['unit'], r['relation'], ed, has_dose)] += 1
    for k, v in hemo_stats.most_common(45):
        print(f"  n={v:6d} | ep={k[0]:24s} src={str(k[1]):28s} lane={k[2]:15s} u={str(k[3]):8s} rel={str(k[4]):5s} detail={str(k[5]):28s} has_dose={k[6]}")

    # 4. Check BBB & Solubility overlap between review and binary_evidence
    print("\n=== 4. BBB & SOLUBILITY OVERLAP ===")
    bbb_bin_keys = set(r[0] for r in cur.execute("SELECT identity_key FROM records WHERE endpoint='BBB' AND lane='binary_evidence'"))
    bbb_rev_keys = set(r[0] for r in cur.execute("SELECT identity_key FROM records WHERE endpoint='BBB' AND source_code='BBB-PEP-Prediction'"))
    print(f"  BBB binary_evidence unique id_keys: {len(bbb_bin_keys)}, BBB-PEP-Prediction: {len(bbb_rev_keys)}, intersection: {len(bbb_bin_keys & bbb_rev_keys)}")

    sol_bin_keys = set(r[0] for r in cur.execute("SELECT identity_key FROM records WHERE endpoint='solubility' AND lane='binary_evidence'"))
    sol_rev_keys = set(r[0] for r in cur.execute("SELECT identity_key FROM records WHERE endpoint='solubility' AND source_code='SolPepBench/PepSol2000'"))
    print(f"  Sol binary_evidence unique id_keys: {len(sol_bin_keys)}, SolPepBench/PepSol2000: {len(sol_rev_keys)}, intersection: {len(sol_bin_keys & sol_rev_keys)}")

if __name__ == "__main__":
    main()
