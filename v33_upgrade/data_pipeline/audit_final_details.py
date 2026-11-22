import sqlite3, json, re
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. LogD7.4 / LogD assay description audit
    print("=== 1. LOGD / LOGD7.4 ASSAY DESCRIPTIONS & pH ===")
    logd_c = Counter()
    for r in cur.execute("""
        SELECT endpoint, origin, source_code, lane, reason, structure_model_eligible, condition_json, quality_flags
        FROM records_with_structure_qc
        WHERE endpoint IN ('LogD7.4', 'LogD') AND lane IN ('strict_numeric', 'review')
    """):
        cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        sem = cj.get('semantics', {})
        assay = sem.get('assay', '')
        if isinstance(assay, dict):
            desc = assay.get('description', '')
        else:
            desc = str(assay)
        ed = str(sem.get('endpoint_detail', ''))
        mat = str(sem.get('matrix', ''))
        text = f"{desc} | {ed} | {mat}".lower()
        is_calc = any(w in text for w in ['calculated', 'clogp', 'mlogp', 'alogp', 'predicted'])
        has_74 = ('7.4' in text) or ('74' in ed) or ('ph7.4' in text) or ('ph 7.4' in text)
        is_decadiene = 'decadiene' in text
        is_chrom = 'chromlogd' in text or 'hplc' in text or 'chrom' in text
        logd_c[(r['endpoint'], r['origin'], r['source_code'], r['structure_model_eligible'], is_calc, has_74, is_decadiene, is_chrom)] += 1
    for k, v in logd_c.most_common(40):
        print(f"  n={v:4d} | ep={k[0]:8s} {k[1]:3s} src={k[2]:38s} smi={k[3]:5s} calc={str(k[4]):5s} pH74={str(k[5]):5s} dec={str(k[6]):5s} chrom={str(k[7]):5s}")

    # 2. Papp & permeability assay classification
    print("\n=== 2. PAPP / PERMEABILITY ASSAY TYPES ===")
    papp_c = Counter()
    for r in cur.execute("""
        SELECT endpoint, origin, source_code, lane, unit, structure_model_eligible, condition_json
        FROM records_with_structure_qc
        WHERE endpoint IN ('Papp', 'permeability') AND lane IN ('strict_numeric', 'censored_numeric', 'review')
          AND structure_model_eligible='true'
    """):
        cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        sem = cj.get('semantics', {})
        assay = sem.get('assay', '')
        desc = (assay.get('description', '') if isinstance(assay, dict) else str(assay)).lower()
        atype = 'other'
        if 'caco-2' in desc or 'caco2' in desc: atype = 'Caco2'
        elif 'mdck' in desc: atype = 'MDCK'
        elif 'pampa' in desc: atype = 'PAMPA'
        elif 'rrck' in desc: atype = 'RRCK'
        papp_c[(r['endpoint'], r['lane'], r['unit'], atype)] += 1
    for k, v in papp_c.most_common(30):
        print(f"  n={v:4d} | {k}")

    # 3. Check OLD vs NEW structure overlap for PAMPA, Caco2, MDCK, RRCK
    print("\n=== 3. PERMEABILITY OLD REVIEW vs NEW STRICT OVERLAP ===")
    for ep in ['PAMPA', 'Caco2', 'MDCK', 'RRCK']:
        new_mols = set(r[0] for r in cur.execute("""
            SELECT canonical_identity_group FROM records_with_structure_qc
            WHERE endpoint=? AND origin='NEW' AND lane='strict_numeric' AND canonical_identity_group<>''
        """, (ep,)))
        for r in cur.execute("""
            SELECT source_code, lane, reason, COUNT(*) as n,
                   COUNT(DISTINCT canonical_identity_group) as n_mol
            FROM records_with_structure_qc
            WHERE endpoint=? AND origin='OLD' AND structure_model_eligible='true' AND unit IN ('cm/s','source_log10_permeability')
            GROUP BY source_code, lane, reason
            ORDER BY n DESC LIMIT 8
        """, (ep,)):
            mols = set(x[0] for x in cur.execute("""
                SELECT canonical_identity_group FROM records_with_structure_qc
                WHERE endpoint=? AND origin='OLD' AND source_code=? AND lane=? AND reason=? AND canonical_identity_group<>''
            """, (ep, r['source_code'], r['lane'], r['reason'])))
            net_new = len(mols - new_mols)
            print(f"  [{ep:5s}] {r['source_code']:38s} lane={r['lane']:14s} n={r['n']:5d} mols={len(mols):5d} net_new_vs_NEW={net_new:5d} | {r['reason']}")

    # 4. Inspect Pepti-DrugPhAI, PEPlife2 semantic_review, openFDA/DailyMed, ema_smpc, primary_pk_literature_v14, primary_ppb_kp_v15
    print("\n=== 4. PK REVIEW QUALITY FLAGS & ESTIMANDS ===")
    for ep in ['PPB', 'half_life', 'half_life_terminal', 'CL_per_kg', 'CL_absolute', 'Vd_per_kg', 'Vdss_per_kg', 'F']:
        rows = cur.execute("""
            SELECT origin, source_code, lane, reason, relation, unit, structure_model_eligible,
                   has_modifications, sequence_kind, condition_json, quality_flags
            FROM records_with_structure_qc
            WHERE endpoint=? AND lane IN ('strict_numeric','censored_numeric','review')
        """, (ep,)).fetchall()
        c = Counter()
        for r in rows:
            cj = {}
            try: cj = json.loads(r['condition_json']) if r['condition_json'] else {}
            except: pass
            sem = cj.get('semantics', {})
            ed = str(sem.get('endpoint_detail', ''))
            mat = str(sem.get('matrix', ''))
            sp = str(sem.get('species', ''))
            rt = str(sem.get('route', ''))
            c[(r['origin'], r['source_code'], r['lane'], r['structure_model_eligible'], ed[:30], sp[:15], mat[:15], rt[:10])] += 1
        print(f"\n  [{ep}] top condition buckets:")
        for k, v in c.most_common(12):
            print(f"    n={v:4d} | {k[0]} {k[1]:34s} {k[2]:15s} smi={k[3]:5s} ed={k[4]:30s} sp={k[5]:15s} mat={k[6]:15s} rt={k[7]}")

    # 5. Why did DBAASP scopes fail multiset match? Inspect 5 failed scopes!
    print("\n=== 5. DBAASP FAILED SCOPE DIAGNOSTICS ===")
    sample_assays = cur.execute("""
        SELECT DISTINCT assay_id, compound_id FROM records
        WHERE source_code='DBAASP' AND origin='NEW' AND endpoint='MIC' AND lane='strict_numeric'
        LIMIT 8
    """).fetchall()
    for a, cid in sample_assays:
        obs = cur.execute("""
            SELECT endpoint, lane, raw_value, raw_relation, raw_unit, value, unit, relation, raw_json
            FROM records WHERE source_code='DBAASP' AND assay_id=? AND compound_id=?
        """, (a, cid)).fetchall()
        ce = cur.execute("""
            SELECT parameter, value FROM condition_evidence
            WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT' AND assay_id=? AND compound_id=?
              AND (parameter LIKE '%.concentration' OR parameter LIKE '%.targetSpecies.name')
        """, (a, cid)).fetchall()
        obs_s = [(r['endpoint'], r['lane'], json.loads(r['raw_json']).get('original_value'), json.loads(r['raw_json']).get('qualifier'), r['raw_unit']) for r in obs]
        ce_s = [(r['parameter'], r['value']) for r in ce]
        print(f"  assay={a} cmp={cid} | n_obs={len(obs_s)} n_ce={len(ce_s)}")
        print(f"    obs[:6]={obs_s[:6]}")
        print(f"    ce[:8]={ce_s[:8]}")

    # 6. Check sequence_kind & has_modifications for rows WITHOUT SMILES across target endpoints
    print("\n=== 6. NO-SMILES SEQUENCE ELIGIBILITY AUDIT ===")
    for ep in ['half_life', 'hemolysis', 'overall_peptide_toxicity', 'BBB', 'solubility', 'cytotoxicity', 'MIC']:
        rows = cur.execute("""
            SELECT sequence_kind, has_modifications,
                   COUNT(*) as n,
                   SUM(CASE WHEN sequence GLOB '[ACDEFGHIKLMNPQRSTVWY]*' AND length(sequence) BETWEEN 2 AND 120 THEN 1 ELSE 0 END) as n_valid20
            FROM records_with_structure_qc
            WHERE endpoint=? AND structure_model_eligible='false' AND lane IN ('strict_numeric','binary_evidence','censored_numeric','review')
            GROUP BY sequence_kind, has_modifications
            ORDER BY n DESC LIMIT 8
        """, (ep,)).fetchall()
        print(f"  [{ep}] no-SMILES breakdown:")
        for r in rows:
            print("   ", dict(r))

if __name__ == "__main__":
    main()
