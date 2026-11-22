#!/usr/bin/env python3
"""Verify exact extended DBAASP multiset bridge (handling multi-endpoint scopes + range sigs),
DBAASP OBSERVATION scope, ChEMBL LogP assay descriptions, and dramp_v4 pure-20AA counts."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")
NON_MOD = {"", "none", "n.a.", "na", "n/a", "free", "null", "unmodified", "no", "false", "\\n"}
AA20_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,120}$")


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


def parse_sig(value, qualifier="", lower="", upper=""):
    v = clean_str(value).replace("≤", "<=").replace("≥", ">=").replace("≈", "~").replace("–", "-")
    # Check range in value string e.g. '0.0625-0.5' or '16 - 32'
    mr = re.fullmatch(r"\s*([+]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+]?(?:\d+(?:\.\d*)?|\.\d+))\s*", v)
    if mr:
        try:
            return ("range", Decimal(mr.group(1)).normalize(), Decimal(mr.group(2)).normalize())
        except InvalidOperation:
            return None
    if not v and clean_str(lower) and clean_str(upper):
        try:
            return ("range", Decimal(clean_str(lower)).normalize(), Decimal(clean_str(upper)).normalize())
        except InvalidOperation:
            return None
    m = re.fullmatch(r"\s*(<=|>=|<|>|=|~)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", v)
    if not m:
        return None
    prefix, num = m.groups()
    q = clean_str(qualifier)
    if q and q not in {"<", ">", "<=", ">=", "=", "~"}:
        return None
    if prefix and q and prefix != q:
        return None
    try:
        return ("scalar", prefix or q or "=", Decimal(num).normalize())
    except InvalidOperation:
        return None


def species_genus(sp_name: str) -> str:
    s = (sp_name or "").strip().lower()
    if not s:
        return ""
    parts = s.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else s


def main():
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("=== 1. DBAASP OBSERVATION scope parameter distribution ===")
    obs_p = Counter()
    for (p,) in cur.execute("SELECT parameter FROM condition_evidence WHERE source_code='DBAASP' AND scope='OBSERVATION' LIMIT 100000"):
        obs_p[p] += 1
    print(" ", obs_p.most_common(15))

    print("\n=== 2. ChEMBL LogP full assay descriptions ===")
    chembl_logp_assays = Counter()
    for r in cur.execute("""
        SELECT r.value, r.relation, r.canonical_smiles, r.condition_json, r.assay_id, r.compound_id
        FROM records_with_structure_qc r
        WHERE r.endpoint='LogP' AND r.source_code='CHEMBL' AND r.lane IN ('strict_numeric','censored_numeric','review')
    """):
        cj = parse_json_safe(r["condition_json"])
        sem = cj.get("semantics", {})
        assay = sem.get("assay", "")
        desc = assay.get("description", "") if isinstance(assay, dict) else str(assay)
        chembl_logp_assays[desc[:100]] += 1
    for k, v in chembl_logp_assays.most_common(20):
        print(f"  n={v:3d} | {k}")

    print("\n=== 3. dramp_v4 provably unmodified pure 20AA check ===")
    for ep in ("cytotoxicity_CC50", "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50", "hemolysis_MHC", "hemolysis_MHC10"):
        n_tot = n_pure_unmod = n_seq_mod = 0
        for r in cur.execute("""
            SELECT raw_json, condition_json, unit, lane
            FROM records_with_structure_qc
            WHERE source_code='dramp_v4' AND endpoint=? AND lane IN ('strict_numeric','censored_numeric','review')
        """, (ep,)):
            n_tot += 1
            rj = parse_json_safe(r["raw_json"])
            s = clean_str(rj.get("sequence")).upper()
            if not AA20_RE.match(s):
                continue
            ctx = parse_json_safe(rj.get("context_json"))
            nmod = str(ctx.get("N-terminal_Modification", "") or "").strip().lower()
            cmod = str(ctx.get("C-terminal_Modification", "") or "").strip().lower()
            omod = str(ctx.get("Other_Modifications", "") or "").strip().lower()
            lin = str(ctx.get("Linear/Cyclic/Branched", "") or "").strip().lower()
            if nmod in NON_MOD and cmod in NON_MOD and omod in NON_MOD and lin in ("linear", "n.a.", "na", "none", ""):
                n_pure_unmod += 1
            else:
                n_seq_mod += 1
        print(f"  dramp_v4 {ep:20s}: total={n_tot:4d} | pure_unmod_20aa={n_pure_unmod:4d} | seq_mod_only={n_seq_mod:4d}")

    print("\n=== 4. Extended Exact DBAASP Multiset Bridge (exact bijection including multi-endpoint scopes & range sigs) ===")
    dbaasp_scopes = defaultdict(list)
    for r in cur.execute("""
        SELECT record_id, endpoint, lane, compound_id, assay_id, raw_value, raw_relation, raw_unit, value, lower, upper, unit, relation, raw_json
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
            "sig": parse_sig(ov, oq, r["lower"], r["upper"]),
            "unit": clean_str(ou).lower().replace("µ", "u").replace("μ", "u").replace(" ", ""),
        })

    dbaasp_indexed = defaultdict(lambda: defaultdict(dict))
    for cid, aid, param, val in cur.execute("""
        SELECT compound_id, assay_id, parameter, value
        FROM condition_evidence
        WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT'
    """):
        m = re.fullmatch(r"\[(\d+)\]\.activity\.(concentration|targetSpecies\.name|targetCell\.name|medium\.name)", param)
        if not m:
            continue
        idx, field = m.groups()
        if (aid, cid) in dbaasp_scopes:
            dbaasp_indexed[(aid, cid)][idx][field] = val

    target_eps = {"MIC", "MBC", "MFC", "MBIC", "MBIC50", "MBIC90", "MBEC", "MBEC50", "MBEC90", "LC_lemma"}
    linked_by_ep = defaultdict(Counter)
    linked_totals = Counter()
    linked_mode = Counter()

    for key, obs_list in dbaasp_scopes.items():
        entries = dbaasp_indexed.get(key)
        if not entries:
            continue
        orig_cnt = Counter(o["sig"] for o in obs_list)
        ctx_sigs = [(idx, parse_sig(e.get("concentration")), e) for idx, e in entries.items()]
        ctx_cnt = Counter(s for _, s, _ in ctx_sigs)
        units = {o["unit"] for o in obs_list}
        # Strict mathematical requirement: exact 1-to-1 multiset equality AND single consistent unit across the scope!
        if None in orig_cnt or None in ctx_cnt or orig_cnt != ctx_cnt:
            continue
        if len(units) != 1 or "" in units:
            continue
        sig_to_ctx = defaultdict(list)
        for idx, s, e in ctx_sigs:
            sig_to_ctx[s].append(e)
        for o in obs_list:
            ep = o["endpoint"]
            if ep not in target_eps or o["lane"] not in ("strict_numeric", "censored_numeric", "review"):
                continue
            if o["sig"][0] != "scalar":
                continue
            if o["unit_norm"] not in ("uM", "ug/mL"):
                continue
            cands = sig_to_ctx[o["sig"]]
            # Require ALL candidates for this sig to be microbial targetSpecies (NOT mammalian targetCell!)
            if any(c.get("targetCell.name") and not c.get("targetSpecies.name") for c in cands):
                continue
            sp_list = [species_genus(c.get("targetSpecies.name", "")) for c in cands]
            if any(not sp for sp in sp_list):
                continue
            if len(cands) == 1:
                linked_by_ep[ep][sp_list[0]] += 1
                linked_totals[(ep, o["unit_norm"])] += 1
                linked_mode[(ep, "unique_sig")] += 1
            elif len(set(sp_list)) == 1:
                linked_by_ep[ep][sp_list[0]] += 1
                linked_totals[(ep, o["unit_norm"])] += 1
                linked_mode[(ep, "unanimous_species")] += 1

    print("  Total linked by (ep, unit):")
    for k, v in sorted(linked_totals.items()):
        print(f"    {str(k):28s}: {v:6d}")
    for ep in ("MIC", "MBC", "MFC", "MBIC", "MBIC50", "MBIC90", "MBEC", "MBEC50", "MBEC90", "LC_lemma"):
        if not linked_by_ep[ep]:
            continue
        tot = sum(linked_by_ep[ep].values())
        print(f"  [{ep}] total={tot:,} | top15={linked_by_ep[ep].most_common(15)}")

    con.close()


if __name__ == "__main__":
    main()
