import sqlite3, json
from collections import Counter
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Inspect DBAASP & CHEMBL MIC linkage to condition_evidence
    print("=== MIC DBAASP SAMPLE ===")
    for r in cur.execute("""
        SELECT record_id, source_record_id, compound_id, source_code, source_entry_id, assay_id, study_id, lane, value, unit, relation, condition_key, condition_status, condition_json, raw_json
        FROM records WHERE endpoint='MIC' AND source_code='DBAASP' AND lane IN ('strict_numeric','censored_numeric') LIMIT 5
    """):
        d = dict(r)
        print(json.dumps(d, ensure_ascii=False))
        # Check condition_evidence for this observation_id / source_entry_id / compound_id / assay_id
        obs_id = json.loads(d['raw_json']).get('observation_id') if d['raw_json'] else None
        ce1 = cur.execute("SELECT scope, parameter, value, unit FROM condition_evidence WHERE observation_id=? LIMIT 10", (obs_id,)).fetchall()
        ce2 = cur.execute("SELECT scope, parameter, value, unit FROM condition_evidence WHERE compound_id=? AND source_code='DBAASP' LIMIT 15", (d['compound_id'],)).fetchall()
        print("  CE by obs_id:", [dict(x) for x in ce1])
        print("  CE by compound_id:", [dict(x) for x in ce2[:8]])

    print("\n=== MIC CHEMBL SAMPLE ===")
    for r in cur.execute("""
        SELECT record_id, source_record_id, compound_id, source_code, source_entry_id, assay_id, study_id, lane, value, unit, relation, condition_key, condition_status, condition_json, raw_json
        FROM records WHERE endpoint='MIC' AND source_code='CHEMBL' AND lane IN ('strict_numeric','censored_numeric') LIMIT 3
    """):
        d = dict(r)
        print(json.dumps(d, ensure_ascii=False))
        ce = cur.execute("SELECT scope, parameter, value, unit FROM condition_evidence WHERE assay_id=? LIMIT 10", (d['assay_id'],)).fetchall()
        print("  CE by assay_id:", [dict(x) for x in ce])

    # 2. Check OLD vs NEW overlap and detailed breakdown for key endpoints
    print("\n=== OLD vs NEW KEY ENDPOINTS SUMMARY ===")
    for ep in ['LogD7.4', 'LogD', 'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'solubility', 'BBB',
               'PPB', 'half_life', 'half_life_terminal', 'CL_per_kg', 'CL_absolute', 'CL',
               'Vd_per_kg', 'Vdss_per_kg', 'Vd_absolute', 'F', 'Cmax']:
        rows = cur.execute("""
            SELECT origin, source_code, lane, reason, COUNT(*) as n,
                   SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
                   SUM(CASE WHEN sequence_kind='canonical_20aa' THEN 1 ELSE 0 END) as n_seq20
            FROM records_with_structure_qc
            WHERE endpoint=?
            GROUP BY origin, source_code, lane, reason
            ORDER BY n DESC LIMIT 12
        """, (ep,)).fetchall()
        print(f"\n[{ep}]")
        for r in rows:
            print(f"  n={r['n']:5d} smi={r['n_smi']:5d} seq20={r['n_seq20']:5d} | {r['origin']:3s} | {r['lane']:16s} | {str(r['source_code']):38s} | {r['reason']}")

    # 3. Check Hemolysis & Cytotoxicity review structures (what is inside condition_json / raw_json for % hemolysis @ conc?)
    print("\n=== HEMOLYSIS % REVIEW SAMPLES (DBAASP_v4.0.1, Hemolytik2_2026, hemopi2_2025, dramp_v4) ===")
    for src in ['DBAASP_v4.0.1', 'Hemolytik2_2026', 'hemopi2_2025', 'dramp_v4', 'HEMOPIMOD']:
        r = cur.execute("""
            SELECT endpoint, origin, source_code, lane, reason, raw_value, raw_unit, value, unit, relation, condition_json, raw_json
            FROM records_with_structure_qc
            WHERE endpoint IN ('hemolysis', 'hemolysis_percent', 'Hemolysis:Source Training Class', 'hemolysis_HC50', 'hemolysis_SOURCE_DEFINED_BINARY')
              AND source_code=? AND lane='review'
            LIMIT 2
        """, (src,)).fetchall()
        for row in r:
            d = dict(row)
            if d['raw_json'] and len(d['raw_json']) > 500: d['raw_json'] = d['raw_json'][:500] + '...'
            print(f"  [{src}]", json.dumps(d, ensure_ascii=False))

    # 4. Check executed_sources or scripts for how DBAASP MIC raw observations were constructed
    print("\n=== CHECK DBAASP source_entry_id PATTERNS FOR MIC ===")
    for r in cur.execute("""
        SELECT source_entry_id, COUNT(*) as n FROM records
        WHERE endpoint='MIC' AND source_code='DBAASP'
        GROUP BY substr(source_entry_id, 1, 30) LIMIT 15
    """):
        print(" ", dict(r))

if __name__ == "__main__":
    main()
