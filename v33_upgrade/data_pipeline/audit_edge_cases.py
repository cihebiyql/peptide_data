import sqlite3, json
from collections import Counter
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Solubility binary_evidence: why has_modifications='true'? What are the sequences & solvents?
    print("=== 1. SOLUBILITY binary_evidence SAMPLE ===")
    sol_solvents = Counter()
    for i, r in enumerate(cur.execute("""
        SELECT source_code, value, sequence, sequence_kind, has_modifications, quality_flags, condition_json, raw_json
        FROM records_with_structure_qc
        WHERE endpoint='solubility' AND lane='binary_evidence'
    """)):
        cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        sol_solvents[(r['has_modifications'], cj.get('semantics', {}).get('matrix', ''), cj.get('semantics', {}).get('endpoint_detail', ''))] += 1
        if i < 4:
            print(" ", dict(r))
    print("  Solvent/mod breakdown:", sol_solvents.most_common(15))

    # 2. Naylor LPE 2018 LogD dec=False (529 rows) — what matrix/solvent is it?
    print("\n=== 2. NAYLOR LPE 2018 LogD MATRIX BREAKDOWN ===")
    nay = Counter()
    for r in cur.execute("""
        SELECT condition_json, value FROM records WHERE endpoint='LogD' AND source_code='Naylor LPE 2018 primary SI'
    """):
        cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        sem = cj.get('semantics', {})
        nay[(sem.get('matrix',''), sem.get('endpoint_detail',''), sem.get('assay',''))] += 1
    for k, v in nay.most_common(10):
        print(f"  n={v:4d} | {k}")

    # 3. Hemolysis binary_evidence: why did classification_v9/hemolysis_human_erythrocyte_20pct_at_50uM have smi=0, seq=0?!
    print("\n=== 3. HEMOLYSIS binary_evidence IDENTITY CHECK ===")
    for src in [
        'ENDL_HEMOLYT',
        'classification_v9/hemolysis_HLPpredFuse_weak_benchmark',
        'classification_v9/hemolysis_human_erythrocyte_20pct_at_50uM',
        'hemopi2_2025',
        'Hemolytik2_2026',
        'DBAASP_v4.0.1',
    ]:
        r = cur.execute("""
            SELECT source_code, value, sequence, sequence_kind, has_modifications, smiles, canonical_smiles,
                   structure_model_eligible, identity_key, raw_json
            FROM records_with_structure_qc
            WHERE endpoint='hemolysis' AND lane='binary_evidence' AND source_code=?
            LIMIT 2
        """, (src,)).fetchall()
        for row in r:
            d = dict(row)
            if d['raw_json'] and len(d['raw_json']) > 350: d['raw_json'] = d['raw_json'][:350] + '...'
            print(f"  [{src}]", json.dumps(d, ensure_ascii=False))

    # 4. Check half_life binary_evidence (classification_v9/T1/2_stability_ge_1h) identity check!
    print("\n=== 4. HALF_LIFE binary_evidence IDENTITY CHECK ===")
    for r in cur.execute("""
        SELECT source_code, value, sequence, sequence_kind, has_modifications, smiles, structure_model_eligible, identity_key, raw_json
        FROM records_with_structure_qc
        WHERE endpoint='half_life' AND lane='binary_evidence' LIMIT 4
    """):
        d = dict(r)
        if d['raw_json'] and len(d['raw_json']) > 350: d['raw_json'] = d['raw_json'][:350] + '...'
        print(" ", json.dumps(d, ensure_ascii=False))

    # 5. Check Cytotoxicity review condition_json & detail breakdown
    print("\n=== 5. CYTOTOXICITY REVIEW STRUCTURE BREAKDOWN ===")
    cyto = Counter()
    for r in cur.execute("""
        SELECT endpoint, source_code, lane, unit, relation, structure_model_eligible, condition_json
        FROM records_with_structure_qc
        WHERE endpoint IN ('cytotoxicity','cytotoxicity_SOURCE_DEFINED_BINARY','CC50','cytotoxicity_CC50','cytotoxicity_IC50','cytotoxicity_EC50','cytotoxicity_LC50','50% Cytotoxicity','50% Cell death')
          AND lane IN ('binary_evidence','strict_numeric','censored_numeric','review')
    """):
        cj = {}
        try: cj = json.loads(r['condition_json']) if r['condition_json'] else {}
        except: pass
        sem = cj.get('semantics', {})
        ed = str(sem.get('endpoint_detail', ''))
        has_dose = bool(sem.get('dose', ''))
        cyto[(r['endpoint'], r['source_code'], r['lane'], r['unit'], r['relation'], r['structure_model_eligible'], ed[:32], has_dose)] += 1
    for k, v in cyto.most_common(30):
        print(f"  n={v:4d} | {k}")

    # 6. Check if any OLD MIC records exist in catalogue.sqlite
    print("\n=== 6. MIC BY ORIGIN & SOURCE_CODE ===")
    for r in cur.execute("""
        SELECT origin, source_code, lane, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi
        FROM records_with_structure_qc
        WHERE endpoint='MIC'
        GROUP BY origin, source_code, lane
        ORDER BY n DESC
    """):
        print(" ", dict(r))

    # 7. Check OLD PEPlife2 vs NEW PEPLIFE2 overlap in half_life
    print("\n=== 7. HALF_LIFE PEPLIFE2 OVERLAP & CONDITIONS ===")
    for r in cur.execute("""
        SELECT origin, source_code, lane, reason, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
               SUM(CASE WHEN sequence<>'' AND has_modifications='false' THEN 1 ELSE 0 END) as n_unmod_seq
        FROM records_with_structure_qc
        WHERE endpoint='half_life' AND source_code IN ('PEPlife2','PEPLIFE2','CHEMBL','Pepti-DrugPhAI','peptherdia_live_2026_07_14','PLIFEPRED')
        GROUP BY origin, source_code, lane, reason
        ORDER BY n DESC LIMIT 20
    """):
        print(" ", dict(r))

    # Sample NEW PEPLIFE2 condition_evidence!
    print("\n=== 8. NEW PEPLIFE2 condition_evidence SAMPLE ===")
    for r in cur.execute("""
        SELECT parameter, COUNT(*) as n FROM condition_evidence WHERE source_code='PEPLIFE2' GROUP BY parameter ORDER BY n DESC LIMIT 20
    """):
        print(" ", dict(r))

    # Sample CHEMBL PK condition_json for half_life, CL_per_kg, Vd_per_kg, F, Cmax, PPB
    print("\n=== 9. NEW CHEMBL PK condition_json SAMPLE ===")
    for ep in ['PPB', 'half_life', 'CL_per_kg', 'CL_absolute', 'Vd_per_kg', 'Vdss_per_kg', 'F', 'Cmax']:
        r = cur.execute("""
            SELECT condition_json FROM records WHERE endpoint=? AND origin='NEW' AND source_code='CHEMBL' AND lane='strict_numeric' LIMIT 1
        """, (ep,)).fetchone()
        if r:
            print(f"  [{ep}] {r[0][:400]}")

if __name__ == "__main__":
    main()
