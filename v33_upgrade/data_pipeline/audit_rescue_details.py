import sqlite3
import json
from collections import Counter
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def sample(sql, params=(), title="", limit=5):
        print(f"\n================ {title} ================")
        rows = cur.execute(sql, params).fetchall()
        print(f"Total matched rows in query: {len(rows)}")
        for r in rows[:limit]:
            d = dict(r)
            if 'raw_json' in d and d['raw_json'] and len(d['raw_json']) > 400:
                d['raw_json'] = d['raw_json'][:400] + "..."
            print(" ", json.dumps(d, ensure_ascii=False))

    # 1. LogD7.4 and LogD review rows
    sample("""
        SELECT origin, source_code, lane, reason, endpoint, raw_value, raw_unit, value, unit, relation, condition_json, upstream_eligible, quality_flags, structure_model_eligible, sequence_kind, raw_json
        FROM records_with_structure_qc
        WHERE endpoint IN ('LogD7.4', 'LogD') AND lane='review'
        LIMIT 12
    """, title="LogD7.4 / LogD review samples", limit=12)

    # Check LogD7.4 and LogD breakdown by source_code + upstream_eligible + structure_model_eligible
    print("\n--- LogD7.4 & LogD breakdown ---")
    for r in cur.execute("""
        SELECT endpoint, origin, source_code, lane, reason, upstream_eligible, structure_model_eligible, COUNT(*) as n
        FROM records_with_structure_qc
        WHERE endpoint IN ('LogD7.4', 'LogD')
        GROUP BY endpoint, origin, source_code, lane, reason, upstream_eligible, structure_model_eligible
        ORDER BY n DESC
    """):
        print(" ", dict(r))

    # 2. PAMPA / Caco2 / MDCK / RRCK OLD review rows (why review? overlap with NEW CYCPEPTMPDB?)
    print("\n--- PAMPA / Caco2 / MDCK / RRCK OLD vs NEW overlap check ---")
    for ep in ['PAMPA', 'Caco2', 'MDCK', 'RRCK']:
        rows = cur.execute("""
            SELECT origin, source_code, lane, reason, upstream_eligible, structure_model_eligible, COUNT(*) as n, COUNT(DISTINCT canonical_identity_group) as n_mol
            FROM records_with_structure_qc
            WHERE endpoint=?
            GROUP BY origin, source_code, lane, reason, upstream_eligible, structure_model_eligible
            ORDER BY n DESC LIMIT 12
        """, (ep,)).fetchall()
        print(f" [{ep}]")
        for r in rows:
            print("  ", dict(r))

    # 3. PK endpoints: PPB, half_life, half_life_terminal, CL_per_kg, CL_absolute, CL, Vd_per_kg, Vdss_per_kg, Vd_absolute, F, Cmax
    print("\n--- PK endpoints condition_json & review breakdown ---")
    for ep in ['PPB', 'half_life', 'half_life_terminal', 'CL_per_kg', 'CL_absolute', 'CL', 'Vd_per_kg', 'Vdss_per_kg', 'Vd_absolute', 'F', 'Cmax']:
        rows = cur.execute("""
            SELECT origin, source_code, lane, reason, raw_unit, unit, relation, upstream_eligible, structure_model_eligible, COUNT(*) as n
            FROM records_with_structure_qc
            WHERE endpoint=?
            GROUP BY origin, source_code, lane, reason, raw_unit, unit, relation, upstream_eligible, structure_model_eligible
            ORDER BY n DESC LIMIT 12
        """, (ep,)).fetchall()
        print(f" [{ep}]")
        for r in rows:
            print("  ", dict(r))

    # Sample some PK review rows with condition_json and raw_json
    for ep in ['PPB', 'half_life', 'CL_per_kg', 'CL_absolute', 'CL', 'Vd_per_kg', 'F']:
        sample("""
            SELECT origin, source_code, lane, reason, raw_value, raw_unit, value, unit, relation, condition_json, upstream_eligible, quality_flags, structure_model_eligible, sequence, raw_json
            FROM records_with_structure_qc
            WHERE endpoint=? AND lane='review'
            LIMIT 4
        """, (ep,), title=f"{ep} review samples", limit=4)

    # 4. Hemolysis percent / review samples
    sample("""
        SELECT origin, source_code, lane, reason, endpoint, raw_value, raw_unit, value, unit, relation, condition_json, structure_model_eligible, sequence, raw_json
        FROM records_with_structure_qc
        WHERE endpoint IN ('hemolysis', 'hemolysis_percent', 'hemolysis_HC50', 'HC50', 'Hemolysis:Source Training Class', 'hemolysis_SOURCE_DEFINED_BINARY')
          AND lane IN ('review', 'censored_numeric')
        LIMIT 12
    """, title="Hemolysis review/censored samples", limit=12)

    # 5. Cytotoxicity samples
    sample("""
        SELECT origin, source_code, lane, reason, endpoint, raw_value, raw_unit, value, unit, relation, condition_json, structure_model_eligible, sequence, raw_json
        FROM records_with_structure_qc
        WHERE endpoint IN ('cytotoxicity', 'CC50', 'cytotoxicity_CC50', 'cytotoxicity_IC50', '50% Cytotoxicity', 'cytotoxicity_SOURCE_DEFINED_BINARY')
          AND lane IN ('review', 'censored_numeric', 'binary_evidence')
        LIMIT 10
    """, title="Cytotoxicity samples", limit=10)

    # 6. MIC organism / strain inspection (records.condition_json, raw_json, condition_evidence)
    sample("""
        SELECT record_id, origin, source_code, source_record_id, compound_id, assay_id, study_id, lane, value, unit, relation, condition_key, condition_status, condition_json, structure_model_eligible, sequence_kind, raw_json
        FROM records_with_structure_qc
        WHERE endpoint='MIC' AND lane IN ('strict_numeric', 'censored_numeric')
        LIMIT 8
    """, title="MIC strict/censored samples", limit=8)

    # Inspect condition_evidence parameters for MIC/DBAASP/CHEMBL/half_life/PK
    print("\n--- condition_evidence parameter counts (top 40) ---")
    for r in cur.execute("""
        SELECT source_code, scope, parameter, COUNT(*) as n
        FROM condition_evidence
        GROUP BY source_code, scope, parameter
        ORDER BY n DESC LIMIT 40
    """):
        print(" ", dict(r))

if __name__ == "__main__":
    main()
