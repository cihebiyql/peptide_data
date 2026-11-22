import sqlite3, json, re
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")

def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def show_samples(sql, params, title, n=3):
        print(f"\n--- {title} ---")
        rows = cur.execute(sql, params).fetchall()
        for r in rows[:n]:
            d = dict(r)
            cj = d.get('condition_json')
            if cj:
                try:
                    cobj = json.loads(cj)
                    # trim huge source_context
                    if isinstance(cobj.get('source_context'), str) and len(cobj['source_context']) > 450:
                        cobj['source_context'] = cobj['source_context'][:450] + '...'
                    d['condition_json'] = cobj
                except Exception:
                    d['condition_json'] = cj[:400]
            if d.get('raw_json') and len(d['raw_json']) > 400:
                d['raw_json'] = d['raw_json'][:400] + '...'
            print(" ", json.dumps(d, ensure_ascii=False))

    # 1. LogD7.4 & LogD source details (especially Naylor, ChEMBL, PMC10539501, Helicogenic, Chloroalkene, acs_jmedchem_5c02090)
    for ep, src in [
        ('LogD7.4', 'ChEMBL peptide endpoint extractor'),
        ('LogD7.4', 'ChEMBL'),
        ('LogD7.4', 'PMC10539501 primary article source data'),
        ('LogD', 'Naylor LPE 2018 primary SI'),
        ('LogD', 'CHEMBL'),
        ('LogD', 'Helicogenic macrocycles 2026 primary SI'),
        ('LogD', 'Chloroalkene peptide 2026 primary SI'),
        ('LogD', 'acs_jmedchem_5c02090'),
    ]:
        show_samples("""
            SELECT endpoint, origin, source_code, lane, reason, raw_value, raw_unit, value, unit, relation,
                   structure_model_eligible, sequence, sequence_kind, quality_flags, condition_json, raw_json
            FROM records_with_structure_qc
            WHERE endpoint=? AND source_code=? LIMIT 2
        """, (ep, src), f"{ep} | {src}", n=2)

    # 2. PAMPA / Caco2 / MDCK / RRCK / Papp rescue candidates
    for ep, src in [
        ('PAMPA', 'swemacrocycledb_figshare_v1_ar_gt_0_3'),
        ('PAMPA', 'Naylor LPE 2018 primary SI'),
        ('PAMPA', 'ChEMBL'),
        ('Caco2', 'swemacrocycledb_figshare_v1_ar_gt_0_3'),
        ('Caco2', 'ChEMBL'),
        ('MDCK', 'PEGASUS supplementary dataset'),
        ('MDCK', 'ChEMBL'),
        ('Papp', 'CHEMBL'),
    ]:
        show_samples("""
            SELECT endpoint, origin, source_code, lane, reason, raw_value, raw_unit, value, unit, relation,
                   structure_model_eligible, quality_flags, condition_json
            FROM records_with_structure_qc
            WHERE endpoint=? AND source_code=? AND lane<>'text_evidence' LIMIT 3
        """, (ep, src), f"{ep} | {src}", n=3)

    # Check CycPeptMPDB -10 sentinel in NEW CYCPEPTMPDB
    print("\n--- CYCPEPTMPDB sentinel check (value <= 1e-9 or raw_value <= -9.5) ---")
    for r in cur.execute("""
        SELECT endpoint, lane, COUNT(*) as n, MIN(CAST(raw_value AS REAL)) as rmin, MAX(CAST(raw_value AS REAL)) as rmax,
               SUM(CASE WHEN CAST(raw_value AS REAL) <= -9.99 THEN 1 ELSE 0 END) as n_le_neg10,
               SUM(CASE WHEN CAST(raw_value AS REAL) <= -9.0 THEN 1 ELSE 0 END) as n_le_neg9
        FROM records WHERE source_code IN ('CYCPEPTMPDB','CycPeptMPDB') AND endpoint IN ('PAMPA','Caco2','MDCK','RRCK')
        GROUP BY endpoint, lane
    """):
        print(" ", dict(r))

    # 3. PK review details: PPB, half_life, CL, Vd, F
    for ep, src in [
        ('PPB', 'primary_ppb_kp_v15'),
        ('PPB', 'PMC10539501 primary article source data'),
        ('PPB', 'Icotrokinra primary article'),
        ('PPB', 'openFDA/DailyMed label'),
        ('PPB', 'peptherdia_live_2026_07_14'),
        ('half_life', 'PEPlife2'),
        ('half_life', 'Pepti-DrugPhAI'),
        ('half_life', 'PLIFEPRED'),
        ('half_life_terminal', 'PEPTHERDIA'),
        ('CL_per_kg', 'Pepti-DrugPhAI'),
        ('CL_per_kg', 'primary_pk_literature_v14'),
        ('CL_per_kg', 'ema_smpc_product_information_2026_07_13'),
        ('CL_absolute', 'openFDA/DailyMed label'),
        ('Vd_per_kg', 'Pepti-DrugPhAI'),
        ('F', 'peptherdia_live_2026_07_14'),
        ('F', 'openFDA/DailyMed label'),
    ]:
        show_samples("""
            SELECT endpoint, origin, source_code, lane, reason, raw_value, raw_unit, value, unit, relation,
                   structure_model_eligible, sequence, sequence_kind, quality_flags, condition_json
            FROM records_with_structure_qc
            WHERE endpoint=? AND source_code=? AND lane='review' LIMIT 2
        """, (ep, src), f"{ep} | {src} (review)", n=2)

if __name__ == "__main__":
    main()
