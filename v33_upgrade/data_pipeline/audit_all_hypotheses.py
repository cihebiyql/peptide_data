import sqlite3, json, re
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== DISTINCT sequence_kind & structure_qc_status ===")
    for r in cur.execute("SELECT sequence_kind, COUNT(*) as n FROM records GROUP BY sequence_kind ORDER BY n DESC"):
        print("  seq_kind:", dict(r))
    for r in cur.execute("SELECT structure_qc_status, structure_model_eligible, COUNT(*) as n FROM records_with_structure_qc GROUP BY structure_qc_status, structure_model_eligible ORDER BY n DESC"):
        print("  struct_qc:", dict(r))

    # Helper to inspect buckets cleanly
    def inspect_ep(eps, label):
        print(f"\n==================== {label} ====================")
        ph = ",".join("?" for _ in eps)
        rows = cur.execute(f"""
            SELECT endpoint, origin, source_code, lane, reason, unit, relation,
                   COUNT(*) as n,
                   SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
                   SUM(CASE WHEN sequence IS NOT NULL AND sequence<>'' THEN 1 ELSE 0 END) as n_seq,
                   SUM(CASE WHEN sequence_kind LIKE '%canonical%' THEN 1 ELSE 0 END) as n_canon_seq,
                   MIN(CAST(value AS REAL)) as vmin, MAX(CAST(value AS REAL)) as vmax
            FROM records_with_structure_qc
            WHERE endpoint IN ({ph})
            GROUP BY endpoint, origin, source_code, lane, reason, unit, relation
            ORDER BY n DESC
        """, eps).fetchall()
        for r in rows[:35]:
            print(f"  n={r['n']:5d} smi={r['n_smi']:5d} seq={r['n_seq']:5d} cseq={r['n_canon_seq']:5d} | {r['endpoint']:24s} | {r['origin']:3s} | {r['lane']:15s} | {str(r['source_code']):34s} | u={str(r['unit']):16s} rel={str(r['relation']):5s} | [{r['vmin']}, {r['vmax']}] | {r['reason']}")
        if len(rows) > 35:
            print(f"  ... +{len(rows)-35} more buckets")

    inspect_ep(['LogD7.4', 'LogD'], 'LOGD / LOGD7.4')
    inspect_ep(['PAMPA', 'Caco2', 'MDCK', 'RRCK', 'Papp', 'permeability'], 'PERMEABILITY')
    inspect_ep(['solubility'], 'SOLUBILITY')
    inspect_ep(['BBB'], 'BBB')
    inspect_ep(['PPB'], 'PPB')
    inspect_ep(['half_life', 'half_life_terminal'], 'HALF_LIFE')
    inspect_ep(['CL_per_kg', 'CL_absolute', 'CL', 'CL_renal_per_kg', 'CL_renal_absolute'], 'CLEARANCE')
    inspect_ep(['Vd_per_kg', 'Vdss_per_kg', 'Vd_absolute', 'Vd'], 'VOLUME OF DISTRIBUTION')
    inspect_ep(['F', 'Cmax'], 'BIOAVAILABILITY & CMAX')
    inspect_ep(['hemolysis', 'hemolysis_SOURCE_DEFINED_BINARY', 'Hemolysis:Source Training Class', 'hemolysis_percent',
                'hemolysis_HC50', 'HC50', 'hemolysis_EC50', 'hemolysis_LC50', 'hemolysis_MHC', 'hemolysis_MHC10',
                'MHC', 'MHC10', 'HC10', 'hemolysis_IC50', 'hemolysis_HD50', 'hemolysis_HC10'], 'HEMOLYSIS')
    inspect_ep(['overall_peptide_toxicity', 'cytotoxicity', 'cytotoxicity_SOURCE_DEFINED_BINARY', 'CC50',
                'cytotoxicity_CC50', 'cytotoxicity_IC50', 'cytotoxicity_EC50', 'cytotoxicity_LC50',
                'cytotoxicity_PERCENT', '50% Cytotoxicity', '50% Cell death'], 'TOXICITY & CYTOTOXICITY')

if __name__ == "__main__":
    main()
