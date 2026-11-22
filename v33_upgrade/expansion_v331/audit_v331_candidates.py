#!/usr/bin/env python3
"""Deep read-only audit of all Group A + Group B candidate new endpoints in catalogue.sqlite."""
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
    if isinstance(s, dict):
        return s
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def parse_scalar_dec(value, qualifier=""):
    value = clean_str(value).replace("≤", "<=").replace("≥", ">=").replace("≈", "~")
    m = re.fullmatch(r"\s*(<=|>=|<|>|=|~)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", value)
    if not m:
        return None
    prefix, num = m.groups()
    q = clean_str(qualifier)
    if q and q not in {"<", ">", "<=", ">=", "=", "~"}:
        return None
    if prefix and q and prefix != q:
        return None
    try:
        return (prefix or q or "=", Decimal(num).normalize())
    except InvalidOperation:
        return None


def main():
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== 1. ALL distinct endpoints matching target keywords ===")
    for r in cur.execute("""
        SELECT endpoint, lane, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
               SUM(CASE WHEN sequence IS NOT NULL AND sequence != '' THEN 1 ELSE 0 END) as n_seq
        FROM records_with_structure_qc
        WHERE lower(endpoint) LIKE '%mhc%' OR lower(endpoint) LIKE '%hc10%' OR lower(endpoint) LIKE '%logp%'
           OR lower(endpoint) LIKE '%efflux%' OR lower(endpoint) LIKE '%mbic%' OR lower(endpoint) LIKE '%mbec%'
           OR lower(endpoint) LIKE '%mbc%' OR lower(endpoint) LIKE '%mfc%' OR lower(endpoint) LIKE '%biofilm%'
           OR lower(endpoint) LIKE '%cytotox%' OR lower(endpoint) LIKE '%cc50%' OR lower(endpoint) LIKE '%cell death%'
           OR endpoint IN ('Vd','Vd_absolute','LC_lemma','LC','LogD','MIC','hemolysis')
        GROUP BY endpoint, lane
        ORDER BY endpoint, n DESC
    """):
        print(f"  {r['endpoint']:38s} | {r['lane']:16s} | n={r['n']:6d} | smi={r['n_smi']:6d} | seq={r['n_seq']:6d}")

    print("\n=== 2. Detailed Group A breakdown (endpoint x lane x unit x relation x source) ===")
    ga_eps = [
        "efflux_ratio",
        "CC50", "cytotoxicity_CC50", "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50",
        "50% Cytotoxicity", "50% Cell death", "cytotoxicity",
        "hemolysis_MHC", "MHC", "hemolysis_MHC10", "HC10", "hemolysis_HC10", "hemolysis",
        "Vd_absolute", "Vd", "LogP", "LogD",
    ]
    ph = ",".join("?" for _ in ga_eps)
    for r in cur.execute(f"""
        SELECT endpoint, lane, unit, relation, source_code, origin, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
               SUM(CASE WHEN sequence IS NOT NULL AND sequence != '' THEN 1 ELSE 0 END) as n_seq
        FROM records_with_structure_qc
        WHERE endpoint IN ({ph}) AND lane IN ('strict_numeric','censored_numeric','review')
        GROUP BY endpoint, lane, unit, relation, source_code, origin
        ORDER BY endpoint, n DESC
    """, ga_eps):
        if r["endpoint"] in ("hemolysis", "cytotoxicity") and r["n"] < 15:
            continue
        if r["endpoint"] == "LogD" and r["n"] < 5:
            continue
        print(f"  {r['endpoint']:20s} | {r['lane']:16s} | {str(r['unit']):14s} | {str(r['relation']):8s} | {str(r['origin']):3s} | {str(r['source_code']):40s} | n={r['n']:5d} smi={r['n_smi']:5d} seq={r['n_seq']:5d}")

    print("\n=== 3. Check efflux_ratio raw samples & relations ===")
    for r in cur.execute("""
        SELECT lane, source_code, origin, raw_value, raw_unit, raw_relation, value, unit, relation, condition_json, raw_json
        FROM records_with_structure_qc
        WHERE endpoint='efflux_ratio'
        LIMIT 12
    """):
        print(" ", dict(r))

    print("\n=== 4. Check LogP & LogD (non-7.4) details ===")
    for ep in ("LogP", "LogD"):
        c_det = Counter()
        samples = []
        for r in cur.execute("""
            SELECT lane, source_code, origin, value, unit, relation, condition_json, structure_model_eligible
            FROM records_with_structure_qc
            WHERE endpoint=? AND lane IN ('strict_numeric','censored_numeric','review')
        """, (ep,)):
            cj = parse_json_safe(r["condition_json"])
            sem = cj.get("semantics", {}) if isinstance(cj, dict) else {}
            ed = str(sem.get("endpoint_detail", "") if isinstance(sem, dict) else "")
            assay = str(sem.get("assay", "") if isinstance(sem, dict) else "")[:80]
            c_det[(r["source_code"], r["lane"], r["unit"], ed[:40], r["structure_model_eligible"])] += 1
            if len(samples) < 6:
                samples.append((r["source_code"], r["lane"], r["value"], r["unit"], r["relation"], ed, assay))
        print(f"  [{ep}] top combos:")
        for k, v in c_det.most_common(20):
            print("   ", k, "->", v)
        print(f"  [{ep}] samples:", samples[:4])

    print("\n=== 5. Check Vd_absolute & Vd details ===")
    for r in cur.execute("""
        SELECT endpoint, lane, unit, raw_unit, relation, source_code, origin, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi
        FROM records_with_structure_qc
        WHERE endpoint IN ('Vd_absolute','Vd') AND lane IN ('strict_numeric','censored_numeric','review')
        GROUP BY endpoint, lane, unit, raw_unit, relation, source_code, origin
        ORDER BY endpoint, n DESC
    """):
        print(" ", dict(r))

    print("\n=== 6. Check hemolysis endpoint_detail for MHC / HC10 / MHC10 ===")
    det_hem = Counter()
    for r in cur.execute("""
        SELECT endpoint, lane, unit, source_code, condition_json, structure_model_eligible, sequence
        FROM records_with_structure_qc
        WHERE endpoint IN ('hemolysis','hemolysis_MHC','MHC','hemolysis_MHC10','HC10','hemolysis_HC10')
          AND lane IN ('strict_numeric','censored_numeric','review')
    """):
        cj = parse_json_safe(r["condition_json"])
        sem = cj.get("semantics", {}) if isinstance(cj, dict) else {}
        ed = str(sem.get("endpoint_detail", "") if isinstance(sem, dict) else "")
        hc = parse_json_safe(cj.get("historical_condition", ""))
        md = str(hc.get("metric_detail", ""))
        if r["endpoint"] != "hemolysis" or any(k in (ed + " " + md).lower() for k in ("mhc", "hc10", "hc 10", "10%", "minimum hem")):
            det_hem[(r["endpoint"], r["source_code"], r["lane"], r["unit"], ed[:40], md[:40], r["structure_model_eligible"], bool(r["sequence"]))] += 1
    for k, v in det_hem.most_common(40):
        print(" ", k, "->", v)

    print("\n=== 7. Check cytotoxicity quantitative & dramp_v4 empty-sequence check ===")
    cyto_det = Counter()
    dramp_samples = []
    for r in cur.execute("""
        SELECT endpoint, lane, unit, source_code, origin, condition_json, raw_json, structure_model_eligible, sequence
        FROM records_with_structure_qc
        WHERE endpoint IN ('CC50','cytotoxicity_CC50','cytotoxicity_IC50','cytotoxicity_EC50','cytotoxicity_LC50','50% Cytotoxicity','50% Cell death','cytotoxicity')
          AND lane IN ('strict_numeric','censored_numeric','review')
    """):
        cj = parse_json_safe(r["condition_json"])
        sem = cj.get("semantics", {}) if isinstance(cj, dict) else {}
        ed = str(sem.get("endpoint_detail", "") if isinstance(sem, dict) else "")
        if r["endpoint"] == "cytotoxicity" and not any(k in ed.lower() for k in ("cc50", "ic50", "ec50", "lc50", "ld50", "gi50")):
            continue
        if r["unit"] not in ("uM", "ug/mL", "nM", "mM", "mg/mL"):
            continue
        cyto_det[(r["endpoint"], r["source_code"], r["lane"], r["unit"], ed[:35], r["structure_model_eligible"], bool(r["sequence"]))] += 1
        if r["source_code"] == "dramp_v4" and not r["sequence"] and len(dramp_samples) < 3:
            dramp_samples.append((r["endpoint"], r["raw_json"][:300], r["condition_json"][:300]))
    for k, v in cyto_det.most_common(45):
        print(" ", k, "->", v)
    if dramp_samples:
        print("  dramp_v4 empty-seq samples:", dramp_samples[:2])

    print("\n=== 8. Check Group B endpoints (DBAASP multiset bridge + ChEMBL + other sources) ===")
    gb_eps = ("MIC", "MBC", "MFC", "MBIC", "MBIC50", "MBIC90", "MBEC", "LC_lemma", "LC")
    ph_b = ",".join("?" for _ in gb_eps)
    for r in cur.execute(f"""
        SELECT endpoint, lane, unit, source_code, origin, COUNT(*) as n,
               SUM(CASE WHEN structure_model_eligible='true' THEN 1 ELSE 0 END) as n_smi,
               SUM(CASE WHEN sequence IS NOT NULL AND sequence != '' THEN 1 ELSE 0 END) as n_seq
        FROM records_with_structure_qc
        WHERE endpoint IN ({ph_b}) AND lane IN ('strict_numeric','censored_numeric','review')
        GROUP BY endpoint, lane, unit, source_code, origin
        ORDER BY endpoint, n DESC
    """, gb_eps):
        if r["n"] < 5:
            continue
        print(f"  {r['endpoint']:12s} | {r['lane']:16s} | {str(r['unit']):10s} | {str(r['origin']):3s} | {str(r['source_code']):30s} | n={r['n']:6d} smi={r['n_smi']:6d} seq={r['n_seq']:6d}")

    # Run exact DBAASP multiset bridge across ALL DBAASP NEW records to see exact linkable counts for MIC, MBC, MFC, MBIC, MBEC, LC_lemma!
    print("\n=== 9. Exact DBAASP multiset bridge audit for MIC, MBC, MFC, MBIC, MBEC, LC_lemma ===")
    dbaasp_scopes = defaultdict(list)
    for r in cur.execute("""
        SELECT record_id, endpoint, lane, compound_id, assay_id, raw_value, raw_relation, raw_unit, value, unit, relation, raw_json
        FROM records
        WHERE source_code='DBAASP' AND origin='NEW'
    """):
        rj = parse_json_safe(r["raw_json"])
        ov = rj.get("original_value") or r["raw_value"] or r["value"]
        oq = rj.get("qualifier") or r["raw_relation"] or r["relation"]
        ou = rj.get("original_unit") or r["raw_unit"] or r["unit"]
        dbaasp_scopes[(r["assay_id"], r["compound_id"])].append({
            "record_id": r["record_id"],
            "endpoint": r["endpoint"],
            "raw_endpoint": clean_str(rj.get("endpoint")),
            "lane": r["lane"],
            "unit_norm": clean_str(r["unit"]),
            "sig": parse_scalar_dec(ov, oq),
            "unit": clean_str(ou).lower().replace("µ", "u").replace("μ", "u").replace(" ", ""),
        })

    dbaasp_indexed = defaultdict(lambda: defaultdict(dict))
    for cid, aid, param, val in cur.execute("""
        SELECT compound_id, assay_id, parameter, value
        FROM condition_evidence
        WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT'
    """):
        m = re.fullmatch(r"\[(\d+)\]\.activity\.(concentration|targetSpecies\.name|medium\.name|measureGroup\.name|activityMeasure\.name)", param)
        if not m:
            continue
        idx, field = m.groups()
        if (aid, cid) in dbaasp_scopes:
            dbaasp_indexed[(aid, cid)][idx][field] = val

    # Check both strict single-endpoint scopes AND multi-endpoint scopes (using activityMeasure.name if available!)
    linked_by_ep_sp = defaultdict(Counter)
    linked_counts = Counter()
    unlinked_reasons = Counter()
    lc_measures = Counter()

    def species_genus(sp_name: str) -> str:
        s = (sp_name or "").strip().lower()
        if not s:
            return "unspecified"
        parts = s.split()
        return " ".join(parts[:2]) if len(parts) >= 2 else s

    for key, obs_list in dbaasp_scopes.items():
        entries = dbaasp_indexed.get(key)
        if not entries:
            for o in obs_list:
                if o["endpoint"] in gb_eps:
                    unlinked_reasons[(o["endpoint"], "no_ctx")] += 1
            continue
        orig_cnt = Counter(o["sig"] for o in obs_list)
        ctx_sigs = [(idx, parse_scalar_dec(e.get("concentration")), e) for idx, e in entries.items()]
        ctx_cnt = Counter(s for _, s, _ in ctx_sigs)
        eps = {o["raw_endpoint"] for o in obs_list}
        units = {o["unit"] for o in obs_list}
        if None in orig_cnt or None in ctx_cnt or orig_cnt != ctx_cnt:
            for o in obs_list:
                if o["endpoint"] in gb_eps:
                    unlinked_reasons[(o["endpoint"], "multiset_mismatch")] += 1
            continue
        if len(eps) != 1 or len(units) != 1 or "" in units:
            for o in obs_list:
                if o["endpoint"] in gb_eps:
                    unlinked_reasons[(o["endpoint"], f"multi_ep_or_unit:{len(eps)}_{len(units)}")] += 1
            continue
        sig_to_ctx = defaultdict(list)
        for idx, s, e in ctx_sigs:
            sig_to_ctx[s].append(e)
        for o in obs_list:
            ep = o["endpoint"]
            if ep not in gb_eps or o["lane"] not in ("strict_numeric", "censored_numeric", "review"):
                continue
            cands = sig_to_ctx[o["sig"]]
            sp_g = [species_genus(c.get("targetSpecies.name", "")) for c in cands]
            if len(cands) == 1 or len(set(sp_g)) == 1:
                g = sp_g[0]
                linked_counts[(ep, o["lane"], o["unit_norm"])] += 1
                linked_by_ep_sp[ep][g] += 1
                if ep == "LC_lemma":
                    lc_measures[(o["raw_endpoint"], cands[0].get("activityMeasure.name", ""))] += 1

    print("  Linked DBAASP counts (ep, lane, unit):")
    for k, v in sorted(linked_counts.items()):
        print("   ", k, "->", v)
    print("  Unlinked reasons (top 15):")
    for k, v in unlinked_reasons.most_common(15):
        print("   ", k, "->", v)
    print("  LC_lemma raw_endpoint / activityMeasure:", lc_measures.most_common(10))
    for ep in ("MIC", "MBC", "MFC", "MBIC", "MBEC", "LC_lemma"):
        print(f"  [{ep}] Top 20 species:")
        for sp, cnt in linked_by_ep_sp[ep].most_common(20):
            print(f"    {sp:35s}: {cnt:5d}")

    print("\n=== 10. ChEMBL species counts for MIC, MBC, MFC, MBIC, MBEC ===")
    chembl_sp = defaultdict(Counter)
    for r in cur.execute("""
        SELECT endpoint, lane, unit, condition_json
        FROM records_with_structure_qc
        WHERE source_code='CHEMBL' AND endpoint IN ('MIC','MBC','MFC','MBIC','MBEC')
          AND lane IN ('strict_numeric','censored_numeric','review')
          AND unit IN ('uM','ug/mL')
    """):
        cj = parse_json_safe(r["condition_json"])
        sem = cj.get("semantics", {}) if isinstance(cj, dict) else {}
        sp = str(sem.get("species", "") or "")
        assay = sem.get("assay", "")
        if not sp and isinstance(assay, dict):
            sp = str(assay.get("system_description", "") or assay.get("description", "")[:60])
        chembl_sp[(r["endpoint"], r["unit"])][species_genus(sp)] += 1
    for (ep, u), cnts in sorted(chembl_sp.items()):
        tot = sum(cnts.values())
        print(f"  ChEMBL {ep} ({u}) total={tot}: top10={cnts.most_common(10)}")

    con.close()


if __name__ == "__main__":
    main()
