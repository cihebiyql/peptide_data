#!/usr/bin/env python3
"""Deep inspection of DBAASP condition_evidence fields, dramp_v4 sequences, LogP, Vd_absolute, efflux_ratio."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")


def clean_str(val) -> str:
    s = str(val or "").strip()
    return "" if s.lower() in {"", "\\n", "none", "unknown", "not reported", "n.a.", "na"} else s


def parse_json_safe(s) -> dict:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main():
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== 1. Distinct DBAASP condition_evidence parameter suffixes ===")
    pat_cnt = Counter()
    for (p,) in cur.execute("SELECT parameter FROM condition_evidence WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT' LIMIT 200000"):
        p_norm = re.sub(r"\[\d+\]", "[i]", p)
        pat_cnt[p_norm] += 1
    for k, v in pat_cnt.most_common(40):
        print(f"  {k:55s}: {v}")

    print("\n=== 2. Sample DBAASP MBIC, MBEC, MBC, MFC records + their condition_evidence ===")
    for ep in ("MBIC", "MBEC", "MBC", "MFC", "LC_lemma"):
        rows = list(cur.execute("""
            SELECT record_id, endpoint, lane, compound_id, assay_id, raw_value, raw_relation, raw_unit, value, unit, relation, raw_json
            FROM records WHERE source_code='DBAASP' AND origin='NEW' AND endpoint=? LIMIT 2
        """, (ep,)))
        for r in rows:
            print(f"  [{ep}] rec={r['record_id']} lane={r['lane']} cid={r['compound_id']} aid={r['assay_id']} val={r['value']} {r['unit']} rel={r['relation']} raw={r['raw_json'][:280]}")
            ce = list(cur.execute("""
                SELECT parameter, value FROM condition_evidence
                WHERE source_code='DBAASP' AND compound_id=? AND assay_id=? LIMIT 40
            """, (r["compound_id"], r["assay_id"])))
            print("    CE:", [(x["parameter"], x["value"]) for x in ce[:20]])

    print("\n=== 3. Inspect dramp_v4 raw_json for sequence & modifications ===")
    for ep in ("cytotoxicity_CC50", "cytotoxicity_IC50", "hemolysis_MHC", "hemolysis_MHC10"):
        n_tot = n_has_seq_rj = n_pure_seq = 0
        sample = None
        for r in cur.execute("""
            SELECT sequence, sequence_kind, has_modifications, canonical_smiles, structure_model_eligible, raw_json, condition_json
            FROM records_with_structure_qc WHERE source_code='dramp_v4' AND endpoint=?
        """, (ep,)):
            n_tot += 1
            rj = parse_json_safe(r["raw_json"])
            s_rj = clean_str(rj.get("sequence"))
            if s_rj:
                n_has_seq_rj += 1
                if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{2,120}", s_rj.upper()):
                    n_pure_seq += 1
            if sample is None:
                sample = (r["sequence"], r["sequence_kind"], r["has_modifications"], list(rj.keys()), rj.get("sequence"), rj.get("smiles"), rj.get("context_json", "")[:250])
        print(f"  dramp_v4 {ep:20s}: total={n_tot}, rj_seq={n_has_seq_rj}, pure20={n_pure_seq} | sample={sample}")

    print("\n=== 4. Inspect CHEMBL & B3Pdb LogP records ===")
    for r in cur.execute("""
        SELECT source_code, origin, lane, value, unit, relation, canonical_smiles, sequence, structure_model_eligible, condition_json, raw_json
        FROM records_with_structure_qc WHERE endpoint='LogP' LIMIT 15
    """):
        cj = parse_json_safe(r["condition_json"])
        print(" ", r["source_code"], r["lane"], r["value"], r["unit"], r["relation"], "smi_ok=", r["structure_model_eligible"], "seq=", (r["sequence"] or "")[:20], "cj=", str(cj)[:220])
    # Count unique SMILES/sequences in LogP
    logp_smis = set()
    logp_seqs = set()
    for r in cur.execute("""
        SELECT source_code, canonical_smiles, sequence, structure_model_eligible, condition_json, value
        FROM records_with_structure_qc WHERE endpoint='LogP' AND lane IN ('strict_numeric','censored_numeric','review')
    """):
        if r["structure_model_eligible"] == "true" and r["canonical_smiles"]:
            logp_smis.add(r["canonical_smiles"])
        elif r["sequence"]:
            logp_seqs.add(r["sequence"])
    print(f"  LogP unique valid SMILES={len(logp_smis)}, seq-only={len(logp_seqs)}")

    print("\n=== 5. Inspect Vd_absolute unique compounds ===")
    vd_smis = set()
    vd_seqs = set()
    vd_rows = 0
    for r in cur.execute("""
        SELECT endpoint, source_code, lane, value, unit, relation, canonical_smiles, sequence, structure_model_eligible, has_modifications, sequence_kind
        FROM records_with_structure_qc WHERE endpoint IN ('Vd_absolute','Vd') AND lane IN ('strict_numeric','censored_numeric','review')
    """):
        if r["unit"] not in ("L", "l", "mL", "ml"):
            continue
        vd_rows += 1
        if r["structure_model_eligible"] == "true" and r["canonical_smiles"]:
            vd_smis.add(r["canonical_smiles"])
        elif r["sequence"]:
            vd_seqs.add(r["sequence"])
    print(f"  Vd_absolute+Vd(L/mL): rows={vd_rows}, unique valid SMILES={len(vd_smis)}, unique seq-only={len(vd_seqs)}")

    print("\n=== 6. Inspect efflux_ratio assays & value ranges ===")
    er_assays = Counter()
    er_vals = []
    er_smis = set()
    for r in cur.execute("""
        SELECT source_code, lane, value, relation, canonical_smiles, condition_json
        FROM records_with_structure_qc WHERE endpoint='efflux_ratio' AND lane IN ('strict_numeric','censored_numeric','review')
    """):
        cj = parse_json_safe(r["condition_json"])
        sem = cj.get("semantics", {})
        er_assays[(r["source_code"], str(sem.get("assay", "")), str(sem.get("endpoint_detail", "")), r["relation"])] += 1
        er_smis.add(r["canonical_smiles"])
        try:
            er_vals.append(float(r["value"]))
        except Exception:
            pass
    print("  efflux_ratio unique SMILES:", len(er_smis), "val min/max:", min(er_vals), max(er_vals))
    for k, v in er_assays.most_common():
        print("   ", k, "->", v)

    print("\n=== 7. Inspect cyclicpepedia_pubchem / cybase / acs_jmedchem cytotoxicity ===")
    for src in ("cyclicpepedia_pubchem", "cybase", "acs_jmedchem_5c00544", "DBAASP_v4.0.1"):
        cnt = Counter()
        for r in cur.execute("""
            SELECT unit, relation, structure_model_eligible, sequence_kind, has_modifications, condition_json
            FROM records_with_structure_qc WHERE endpoint='cytotoxicity' AND source_code=? AND unit IN ('uM','ug/mL','nM')
            LIMIT 500
        """, (src,)):
            cj = parse_json_safe(r["condition_json"])
            sem = cj.get("semantics", {})
            ed = str(sem.get("endpoint_detail", ""))
            cnt[(r["unit"], r["relation"], r["structure_model_eligible"], r["sequence_kind"], r["has_modifications"], ed[:30])] += 1
        print(f"  cytotoxicity {src}:", cnt.most_common(8))

    con.close()


if __name__ == "__main__":
    main()
