import sqlite3
import json
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Check all endpoints matching our target domains/names
    all_eps = [r[0] for r in cur.execute("SELECT DISTINCT endpoint FROM records").fetchall()]
    exact_targets = [
        'PAMPA', 'Caco2', 'MDCK', 'RRCK', 'Papp', 'permeability', 'logPapp',
        'BBB', 'B3Pdb source-defined response',
        'LogD7.4', 'LogD', 'LogP',
        'solubility',
        'PPB', 'half_life', 'half_life_terminal',
        'CL_per_kg', 'CL_absolute', 'CL', 'CL_renal_per_kg', 'CL_renal_absolute', 'CL_per_mg_protein',
        'Vd_per_kg', 'Vdss_per_kg', 'Vd_absolute', 'Vd',
        'F', 'Cmax',
        'hemolysis', 'hemolysis_SOURCE_DEFINED_BINARY', 'Hemolysis:Source Training Class',
        'hemolysis_percent', 'hemolysis_HC50', 'HC50', 'hemolysis_EC50', 'hemolysis_LC50',
        'hemolysis_MHC', 'hemolysis_MHC10', 'MHC', 'MHC10', 'HC10', 'hemolysis_IC50',
        'overall_peptide_toxicity',
        'cytotoxicity', 'cytotoxicity_SOURCE_DEFINED_BINARY', 'CC50', 'cytotoxicity_CC50',
        'cytotoxicity_IC50', 'cytotoxicity_EC50', 'cytotoxicity_LC50', 'cytotoxicity_PERCENT',
        '50% Cytotoxicity', '50% Cell death',
        'MIC'
    ]
    # Also find any other endpoint containing solubility, half_life, PPB, Vd, CL_, LogD, bioavail, BBB, hemol, cytotox
    extra = []
    for ep in all_eps:
        epl = (ep or '').lower()
        if ep in exact_targets:
            continue
        if any(k in epl for k in ['solub', 'logd', 'ppb', 'half_life', 'half-life', 't1/2', 'vdss', 'vd_', 'cl_', 'bioavail', 'hemol', 'cytotox', 'bbb', 'pampa', 'caco', 'mdck', 'rrck', 'cmax']):
            if '%' not in epl:
                extra.append(ep)
    print("=== EXTRA MATCHING ENDPOINTS ===")
    for ep in sorted(extra):
        cnt = cur.execute("SELECT lane, COUNT(*) FROM records WHERE endpoint=? GROUP BY lane", (ep,)).fetchall()
        print(f"  {ep!r:45s} {[(r[0], r[1]) for r in cnt]}")

    print("\n=== DETAILED BREAKDOWN BY TARGET ENDPOINT ===")
    for ep in exact_targets:
        rows = cur.execute("""
            SELECT lane, reason, origin, source_code, raw_unit, unit, relation, normalization_reason, COUNT(*) as n
            FROM records WHERE endpoint=?
            GROUP BY lane, reason, origin, source_code, raw_unit, unit, relation, normalization_reason
            ORDER BY n DESC
        """, (ep,)).fetchall()
        if not rows:
            continue
        tot = sum(r['n'] for r in rows)
        print(f"\n--- {ep} (total={tot:,}) ---")
        for r in rows[:20]:
            print(f"  n={r['n']:6d} | lane={r['lane']:16s} | orig={r['origin']:3s} | src={str(r['source_code']):22s} | raw_u={str(r['raw_unit']):15s} -> u={str(r['unit']):15s} | rel={str(r['relation']):4s} | reason={r['reason']}")
        if len(rows) > 20:
            print(f"  ... (+{len(rows)-20} more groups)")

if __name__ == "__main__":
    main()
