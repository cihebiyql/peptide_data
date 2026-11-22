#!/usr/bin/env python3
"""
Step 01: Extract all raw candidate observations for Peptide OmniPanel V33 (Data Cleaning 2.0)
from the frozen read-only catalogue.sqlite + Case17 benchmark files.

Strict rules:
- Opens catalogue.sqlite strictly read-only (?mode=ro).
- Never fabricates SMILES or modifies source files.
- Every rescued row carries an explicit rescue_reason plus origin, source_code, source_record_id.
- Preserves exact + right/left/interval censored records without dropping inequalities.
- Links DBAASP MIC species ONLY via the mathematically proven, order-independent multiset bridge.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")
CASE17_RAW = Path(r"D:\work\中检院\data\case17_20261119\case17_raw_responses.json")
CASE17_REF = Path(r"D:\work\中检院\data\case17_20261119\case17_reference_values.tsv")
OUT_DIR = Path(r"D:\work\中检院\v33_upgrade\data_pipeline\staging")

AA20_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,120}$")
HELM_PURE20_RE = re.compile(r"^PEPTIDE1\{[ACDEFGHIKLMNPQRSTVWY](\.[ACDEFGHIKLMNPQRSTVWY]){1,119}\}\$\$\$\$$")
RANGE_PCT_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*%")
DOSE_RE = re.compile(r"(?:<=|>=|<|>|~)?\s*(\d+(?:\.\d+)?)\s*(uM|µM|μM|mM|nM|ug/mL|µg/mL|μg/mL|ug/ml|µg/ml|μg/ml|mg/mL|mg/ml|g/L)", re.I)

NON_MOD_STRINGS = {"", "none", "n.a.", "na", "n/a", "free", "null", "unmodified", "no", "false", "\\n"}


def clean_str(val) -> str:
    s = str(val or "").strip()
    return "" if s.lower() in {"", "\\n", "none", "unknown", "not reported", "n.a.", "na"} else s


def parse_float(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "inf", "-inf", "none", "\\n"}:
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


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


def norm_species(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unspecified"
    if any(k in t for k in ("homo sapiens", "human", "patient", "volunteer", "adult", "infant", "clinical")):
        return "human"
    if any(k in t for k in ("rattus", "rat", "sprague", "wistar")):
        return "rat"
    if any(k in t for k in ("mus musculus", "mouse", "mice", "murine", "balb", "icr", "c57bl", "cd-1")):
        return "mouse"
    if any(k in t for k in ("canis", "dog", "beagle")):
        return "dog"
    if any(k in t for k in ("macaca", "monkey", "cynomolgus", "rhesus", "primate")):
        return "monkey"
    if any(k in t for k in ("oryctolagus", "rabbit")):
        return "rabbit"
    if any(k in t for k in ("ovis", "sheep", "ovine")):
        return "sheep"
    if any(k in t for k in ("bos taurus", "bovine", "calf", "cow")):
        return "bovine"
    if any(k in t for k in ("pig", "porcine", "sus scrofa", "guinea pig", "hamster", "horse", "equine")):
        return "other"
    return "unspecified"


def norm_route(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unspecified"
    if "in vitro" in t or "ex vivo" in t or "microsom" in t or "protease" in t or "homogenate" in t:
        return "in_vitro"
    # Check specific tokens
    if re.search(r"\b(iv|i\.v\.|intravenous|intravenouly|bolus|infusion)\b", t):
        return "IV"
    if re.search(r"\b(sc|s\.c\.|subcutaneous)\b", t):
        return "SC"
    if re.search(r"\b(im|i\.m\.|intramuscular)\b", t):
        return "IM"
    if re.search(r"\b(po|p\.o\.|oral|orally|intragastric)\b", t):
        return "PO"
    if re.search(r"\b(in|intranasal|nasal)\b", t):
        return "IN"
    if re.search(r"\b(ip|i\.p\.|intraperitoneal)\b", t):
        return "IP"
    return "unspecified"


def norm_hl_matrix(text: str, route: str = "") -> str:
    t = (text or "").lower()
    if any(k in t for k in ("intestin", "trypsin", "chymotrypsin", "pancreatin", "pepsin", "sif", "sgf", "gastric", "protease", "elastase", "cathepsin", "dpp-4", "dpp4", "neprilysin")):
        if any(k in t for k in ("gastric", "pepsin", "sgf", "stomach")):
            return "in_vitro_gastric_protease"
        return "in_vitro_intestinal_protease"
    if any(k in t for k in ("microsom", "hepatocyte", "liver", "s9", "kidney", "renal homogenate")):
        return "in_vitro_liver_kidney"
    if any(k in t for k in ("in vivo", "terminal", "elimination", "patient", "volunteer", "pharmacokinetic", "administered", "bolus", "infusion", "dosed", "single dose")) or route in ("IV", "SC", "IM", "PO", "IN", "IP"):
        if "blood" in t and "plasma" not in t:
            return "in_vivo_blood"
        return "in_vivo_plasma"
    if "serum" in t:
        return "in_vitro_serum"
    if "plasma" in t:
        return "in_vitro_plasma"
    if "blood" in t:
        return "in_vitro_blood"
    if "buffer" in t or "pbs" in t or "water" in t or "solution" in t:
        return "in_vitro_buffer"
    return "unspecified"


def norm_ppb_matrix(text: str) -> str:
    t = (text or "").lower()
    if "serum" in t:
        return "serum"
    if "albumin" in t or "hsa" in t or "bsa" in t:
        return "albumin"
    if "plasma" in t:
        return "plasma"
    return "unspecified"


def norm_cl_estimand(text: str, route: str = "") -> str:
    t = (text or "").lower()
    if "renal" in t:
        return "CL_renal"
    if any(k in t for k in ("cl/f", "cl_over_f", "apparent_clearance", "apparent clearance", "oral clearance", "cl/f")):
        return "CL_over_F"
    if route in ("PO", "SC", "IM", "IN") and "systemic" not in t and "total_body" not in t:
        return "CL_over_F"
    if any(k in t for k in ("systemic", "total", "plasma clearance", "observed_clearance", "iv")) or route == "IV":
        return "CL_total"
    return "CL_unspecified"


def norm_vd_estimand(endpoint: str, text: str, route: str = "") -> str:
    t = (text or "").lower()
    if endpoint == "Vdss_per_kg" or any(k in t for k in ("vdss", "vss", "steady state", "steady-state")):
        return "Vss"
    if any(k in t for k in ("vz/f", "vd/f", "v/f", "apparent volume", "v_over_f")) or route in ("PO", "SC", "IM"):
        if "steady" in t:
            return "Vss"
        if any(k in t for k in ("/f", "apparent")) and route != "IV":
            return "V_F"
    if any(k in t for k in ("vz", "terminal", "beta")):
        return "Vz"
    if any(k in t for k in ("vd", "volume of distribution")):
        return "Vd"
    return "Vd_unspecified"


def norm_solvent(text: str) -> str:
    t = (text or "").lower()
    if "dmso" in t:
        return "dmso"
    if "ultrapure" in t or t.strip() == "water":
        return "ultrapure_water"
    if "dpbs" in t or "pbs" in t or "ph 7.4" in t or "ph7.4" in t or "phosphate" in t:
        return "pbs_ph74"
    if "fassif" in t or "fessif" in t or "sgf" in t or "sif" in t:
        return "biorelevant_fluid"
    if any(k in t for k in ("hcooh", "acetic", "ammonia", "acid", "hcl", "naoh")):
        return "acid_base"
    if any(k in t for k in ("aqueous", "buffer", "water")):
        return "aqueous_other"
    return "unspecified"


def parse_conc_dose(dose_str: str):
    """Parse concentration string like '100 uM', '250 ug/mL', '<=100 ug/mL' -> (val, norm_unit)."""
    if not dose_str:
        return (None, "")
    s = dose_str.replace("micro g/ ml", "ug/mL").replace("micro g/ml", "ug/mL").replace("µ", "u").replace("μ", "u")
    m = DOSE_RE.search(s)
    if not m:
        return (None, "")
    v = parse_float(m.group(1))
    u = m.group(2).lower().replace("µ", "u").replace("μ", "u")
    if v is None or v <= 0:
        return (None, "")
    if u == "mm":
        return (v * 1000.0, "uM")
    if u == "nm":
        return (v / 1000.0, "uM")
    if u == "um":
        return (v, "uM")
    if u in ("mg/ml", "g/l"):
        return (v * 1000.0, "ug/mL")
    if u == "ug/ml":
        return (v, "ug/mL")
    return (None, "")


def parse_cmax_dose_bucket(desc: str) -> str:
    t = (desc or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg/kg|ug/kg|µg/kg)", t)
    if m:
        val = float(m.group(1))
        if "ug" in m.group(2) or "µg" in m.group(2):
            val /= 1000.0
        if val < 1.0:
            return "lt1_mg_kg"
        if val <= 10.0:
            return "1_to_10_mg_kg"
        return "gt10_mg_kg"
    if re.search(r"\b\d+(?:\.\d+)?\s*(mg|ug|µg)\b", t):
        return "fixed_dose"
    return "unspecified"


def classify_mic_species(name: str):
    if not name:
        return (None, "unspecified")
    s = name.lower()
    sp = None
    if "escherichia coli" in s or "e. coli" in s or "e.coli" in s:
        sp = "MIC_E_coli"
    elif "staphylococcus aureus" in s or "s. aureus" in s or "s.aureus" in s:
        sp = "MIC_S_aureus"
    elif "pseudomonas aeruginosa" in s or "p. aeruginosa" in s or "p.aeruginosa" in s:
        sp = "MIC_P_aeruginosa"
    elif "candida albicans" in s or "c. albicans" in s or "c.albicans" in s:
        sp = "MIC_C_albicans"
    if not sp:
        return (None, "unspecified")
    # Strain category
    if "25922" in s:
        st = "ATCC_25922"
    elif "25923" in s:
        st = "ATCC_25923"
    elif "29213" in s:
        st = "ATCC_29213"
    elif "27853" in s:
        st = "ATCC_27853"
    elif "90028" in s or "10231" in s:
        st = "ATCC_ref_candida"
    elif any(k in s for k in ("mrsa", "usa300", "usa 300", "43300", "methicillin-resistant", "methicillin resistant", " mr")):
        st = "MRSA"
    elif "atcc" in s or "nctc" in s or "dsm" in s or "mtcc" in s:
        st = "standard_other"
    else:
        st = "unspecified"
    return (sp, st)


def norm_mic_medium(m: str) -> str:
    t = (m or "").lower()
    if not t:
        return "unspecified"
    if "camhb" in t or "cation-adjusted" in t:
        return "CAMHB"
    if "mhb" in t or "mueller" in t:
        return "MHB"
    if "rpmi" in t:
        return "RPMI"
    if "lb" in t or "luria" in t:
        return "LB"
    if "tsb" in t or "tryptic soy" in t:
        return "TSB"
    if "ypd" in t or "sabouraud" in t:
        return "fungal_broth"
    return "other"


def check_pure_unmodified_20aa(row, rj: dict, cj: dict) -> tuple[str, int]:
    """
    Strictly verify whether a record's sequence is a provably unmodified pure 20AA peptide (len 2..120)
    eligible for deterministic RDKit Chem.MolFromSequence(seq) when source SMILES is missing.
    NEVER returns 1 for modified peptides, D-amino acids, cyclized peptides, or residue projections!
    """
    seq = clean_str(row["sequence"]).upper()
    if not AA20_RE.match(seq):
        return ("", 0)

    # 1. Special verified case: SolPepBench/PepSol2000 has pure unmodified HELM PEPTIDE1{A.C...}$$$$
    helm = clean_str(rj.get("helm", ""))
    if helm and HELM_PURE20_RE.match(helm):
        return (seq, 1)

    # 2. General strict contract: sequence_kind must be bare_standard AND has_modifications == 'false'
    if row["sequence_kind"] != "bare_standard" or str(row["has_modifications"]).lower() != "false":
        return (seq, 0)

    qf = (row["quality_flags"] or "").lower()
    if any(bad in qf for bad in ("residue_projection_only", "preserves_d_amino_acids", "n_methylation", "cyclization", "thioamide", "modified_sequence")):
        return (seq, 0)

    # Check source_context / context_json for chemical_modifications, n/c-term mods
    ctx_raw = cj.get("source_context") or rj.get("context_json") or ""
    ctx = parse_json_safe(ctx_raw) if isinstance(ctx_raw, str) else (ctx_raw if isinstance(ctx_raw, dict) else {})
    for k in ("chemical_modifications", "N-terminal_Modification", "C-terminal_Modification", "Other_Modifications", "n_terminus", "c_terminus"):
        v = str(ctx.get(k, "") or "").strip().lower()
        if v not in NON_MOD_STRINGS:
            return (seq, 0)
    lin = str(ctx.get("Linear/Cyclic/Branched", "") or ctx.get("linearity_cyclicity", "") or "").strip().lower()
    if lin and lin not in ("linear", "n.a.", "na", "none"):
        return (seq, 0)

    return (seq, 1)


def extract_context_text(cj: dict) -> tuple[dict, str, str, str, str, str]:
    """Return (sem, assay_desc, species_raw, matrix_raw, route_raw, endpoint_detail)."""
    sem = cj.get("semantics", {}) if isinstance(cj, dict) else {}
    if not isinstance(sem, dict):
        sem = {}
    assay = sem.get("assay", "")
    if isinstance(assay, dict):
        assay_desc = " ".join(
            str(assay.get(k, "") or "")
            for k in ("description", "system_description", "cell_line", "strain", "tissue")
        ).strip()
        sp_a = str(assay.get("system_description", "") or "")
        mat_a = str(assay.get("tissue", "") or assay.get("cell_line", "") or "")
    else:
        assay_desc = str(assay or "")
        sp_a = ""
        mat_a = ""
    hc = parse_json_safe(cj.get("historical_condition", "")) if isinstance(cj, dict) else {}
    sp = clean_str(sem.get("species")) or clean_str(hc.get("species")) or sp_a
    mat = clean_str(sem.get("matrix")) or clean_str(hc.get("matrix")) or mat_a
    rt = clean_str(sem.get("route")) or clean_str(hc.get("route"))
    ed = clean_str(sem.get("endpoint_detail")) or clean_str(hc.get("metric_detail"))
    return sem, assay_desc, sp, mat, rt, ed


def is_enzyme_or_non_pk_halflife(desc: str) -> bool:
    t = (desc or "").lower()
    bad = [
        "loose half of its activity", "lose half of its activity", "inactivation half",
        "dissociation half", "residence time", "off-rate", "koff", "thermal",
        "denaturation", "radionuclide", "radioactive decay", "beta-amyloid",
        "fibril", "aggregation half", "fluorescence half", "photobleach",
        "enzyme half-life", "catalytic", "calpain", "proteasome inhibition",
        "time required to", "binding half-life", "complex half-life"
    ]
    return any(b in t for b in bad)


def is_permeability_cm_s_convertible(raw_u: str, norm_u: str, val: float):
    """Convert unambiguous linear permeability units to cm/s. Never guesses ambiguous sign typos like 10^6cm/s."""
    if val is None or val <= 0:
        return None
    if norm_u == "cm/s":
        return val
    u = (raw_u or norm_u or "").strip().lower().replace(" ", "")
    if u in ("cm/s", "cms-1", "cm.s-1"):
        return val
    if u in ("10^-6cm/s", "10'-6cm/s", "1e-6cm/s", "ucm/s", "10e-6cms-1", "10^-6cms-1"):
        return val * 1e-6
    if u in ("10^-5cm/s", "10'-5cm/s", "1e-5cm/s"):
        return val * 1e-5
    if u in ("10^-7cm/s", "10'-7cm/s", "1e-7cm/s"):
        return val * 1e-7
    if u in ("10^-8cm/s", "10'-8cm/s", "1e-8cm/s"):
        return val * 1e-8
    if u in ("10^-9cm/s", "10'-9cm/s", "1e-9cm/s"):
        return val * 1e-9
    if u in ("nm/s", "nm.s-1"):
        return val * 1e-7
    return None


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Load PEPLIFE2 condition_evidence map (assay_id, compound_id) -> {assay, dose_route}
    print("[1/5] Loading PEPLIFE2 condition_evidence...", flush=True)
    peplife_ce = defaultdict(dict)
    for cid, aid, param, val in cur.execute("""
        SELECT compound_id, assay_id, parameter, value
        FROM condition_evidence
        WHERE source_code='PEPLIFE2' AND parameter IN ('[0].assay', '[0].dose_route', 'assay', 'dose_route')
    """):
        k = param.split(".")[-1]
        if clean_str(val):
            peplife_ce[(aid, cid)][k] = clean_str(val)

    # Pre-compute DBAASP MIC exact order-independent multiset bridge
    print("[2/5] Running exact order-independent DBAASP MIC multiset bridge...", flush=True)
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
            "sig": parse_scalar_dec(ov, oq),
            "unit": clean_str(ou).lower().replace("µ", "u").replace("μ", "u").replace(" ", ""),
        })

    dbaasp_indexed = defaultdict(lambda: defaultdict(dict))
    for cid, aid, param, val in cur.execute("""
        SELECT compound_id, assay_id, parameter, value
        FROM condition_evidence
        WHERE source_code='DBAASP' AND scope='ASSAY_CONTEXT'
    """):
        m = re.fullmatch(r"\[(\d+)\]\.activity\.(concentration|targetSpecies\.name|medium\.name)", param)
        if not m:
            continue
        idx, field = m.groups()
        if (aid, cid) in dbaasp_scopes:
            dbaasp_indexed[(aid, cid)][idx][field] = val

    # record_id -> (mic_endpoint, strain_cat, medium, bridge_mode)
    dbaasp_mic_linked = {}
    for key, obs_list in dbaasp_scopes.items():
        entries = dbaasp_indexed.get(key)
        if not entries:
            continue
        orig_cnt = Counter(o["sig"] for o in obs_list)
        ctx_sigs = [(idx, parse_scalar_dec(e.get("concentration")), e) for idx, e in entries.items()]
        ctx_cnt = Counter(s for _, s, _ in ctx_sigs)
        eps = {o["raw_endpoint"] for o in obs_list}
        units = {o["unit"] for o in obs_list}
        if None in orig_cnt or None in ctx_cnt or orig_cnt != ctx_cnt:
            continue
        if len(eps) != 1 or len(units) != 1 or "" in units:
            continue
        sig_to_ctx = defaultdict(list)
        for idx, s, e in ctx_sigs:
            sig_to_ctx[s].append(e)
        for o in obs_list:
            if o["endpoint"] != "MIC" or o["lane"] not in ("strict_numeric", "censored_numeric"):
                continue
            cands = sig_to_ctx[o["sig"]]
            if len(cands) == 1:
                sp, st = classify_mic_species(cands[0].get("targetSpecies.name", ""))
                if sp:
                    med = norm_mic_medium(cands[0].get("medium.name", ""))
                    dbaasp_mic_linked[o["record_id"]] = (sp, st, med, "dbaasp_exact_multiset_unique_sig")
            else:
                sp_list = [classify_mic_species(c.get("targetSpecies.name", "")) for c in cands]
                sp_set = {x[0] for x in sp_list}
                if len(sp_set) == 1 and None not in sp_set:
                    sp = next(iter(sp_set))
                    st_set = {x[1] for x in sp_list}
                    st = next(iter(st_set)) if len(st_set) == 1 else "unspecified"
                    med_set = {norm_mic_medium(c.get("medium.name", "")) for c in cands}
                    med = next(iter(med_set)) if len(med_set) == 1 else "unspecified"
                    dbaasp_mic_linked[o["record_id"]] = (sp, st, med, "dbaasp_exact_multiset_same_species")
    print(f"  Linked {len(dbaasp_mic_linked):,} DBAASP MIC observations cleanly.", flush=True)

    # Free bridge structures
    del dbaasp_scopes, dbaasp_indexed

    # Target raw endpoints to scan in a single SQL query
    scan_endpoints = [
        "PAMPA", "Caco2", "MDCK", "RRCK", "Papp", "permeability",
        "BBB",
        "LogD7.4", "LogD",
        "solubility",
        "PPB", "half_life", "half_life_terminal",
        "CL_per_kg", "CL_absolute", "CL",
        "Vd_per_kg", "Vdss_per_kg", "Vd_absolute",
        "F", "Cmax",
        "hemolysis", "hemolysis_SOURCE_DEFINED_BINARY", "Hemolysis:Source Training Class",
        "hemolysis_percent", "hemolysis_HC50", "HC50",
        "overall_peptide_toxicity",
        "cytotoxicity", "cytotoxicity_SOURCE_DEFINED_BINARY", "CC50", "cytotoxicity_CC50",
        "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50",
        "50% Cytotoxicity", "50% Cell death",
        "MIC",
    ]

    fields = [
        "endpoint_v33", "task_tier", "task_type",
        "record_id", "origin", "source_code", "source_record_id", "endpoint_raw",
        "lane_orig", "reason_orig", "rescue_reason", "evidence_priority",
        "canonical_smiles_src", "sequence_raw", "sequence_kind", "has_modifications", "is_pure_unmodified_20aa",
        "val_num", "val_unit", "val_relation", "val_lower", "val_upper",
        "aux_conc_val", "aux_conc_unit", "aux_pct_val",
        "cond_species", "cond_matrix", "cond_route", "cond_estimand",
        "cond_assay", "cond_solvent", "cond_strain", "cond_dose",
    ]

    out_tsv_gz = OUT_DIR / "raw_candidates.tsv.gz"
    stats = Counter()
    rescue_stats = Counter()
    drop_stats = Counter()

    def make_row(
        ep_v33, tier, ttype, r, seq_raw, is_pure20,
        val_num, val_unit, val_rel, rescue_reason, priority,
        val_lower="", val_upper="", aux_conc_val="", aux_conc_unit="", aux_pct_val="",
        cond_species="unspecified", cond_matrix="unspecified", cond_route="unspecified",
        cond_estimand="unspecified", cond_assay="unspecified", cond_solvent="unspecified",
        cond_strain="unspecified", cond_dose="unspecified",
    ):
        smi = clean_str(r["canonical_smiles"]) if r["structure_model_eligible"] == "true" else ""
        if not smi and not seq_raw:
            drop_stats[(ep_v33, "no_smiles_and_no_seq")] += 1
            return None
        stats[ep_v33] += 1
        rescue_stats[(ep_v33, rescue_reason)] += 1
        return {
            "endpoint_v33": ep_v33,
            "task_tier": tier,
            "task_type": ttype,
            "record_id": r["record_id"],
            "origin": r["origin"],
            "source_code": r["source_code"],
            "source_record_id": r["source_record_id"],
            "endpoint_raw": r["endpoint"],
            "lane_orig": r["lane"],
            "reason_orig": r["reason"],
            "rescue_reason": rescue_reason,
            "evidence_priority": priority,
            "canonical_smiles_src": smi,
            "sequence_raw": seq_raw,
            "sequence_kind": r["sequence_kind"],
            "has_modifications": r["has_modifications"],
            "is_pure_unmodified_20aa": is_pure20,
            "val_num": val_num,
            "val_unit": val_unit,
            "val_relation": val_rel,
            "val_lower": val_lower,
            "val_upper": val_upper,
            "aux_conc_val": aux_conc_val,
            "aux_conc_unit": aux_conc_unit,
            "aux_pct_val": aux_pct_val,
            "cond_species": cond_species,
            "cond_matrix": cond_matrix,
            "cond_route": cond_route,
            "cond_estimand": cond_estimand,
            "cond_assay": cond_assay,
            "cond_solvent": cond_solvent,
            "cond_strain": cond_strain,
            "cond_dose": cond_dose,
        }

    print("[3/5] Scanning records_with_structure_qc and extracting cleaned candidates...", flush=True)
    ph = ",".join("?" for _ in scan_endpoints)
    sql = f"""
        SELECT record_id, origin, source_code, source_record_id, compound_id, assay_id,
               endpoint, lane, reason, raw_value, raw_unit, value, lower, upper, relation, unit,
               condition_json, sequence, sequence_kind, has_modifications, quality_flags,
               structure_model_eligible, canonical_smiles, raw_json
        FROM records_with_structure_qc
        WHERE endpoint IN ({ph})
          AND lane IN ('strict_numeric', 'binary_evidence', 'censored_numeric', 'review')
    """

    n_written = 0
    with gzip.open(out_tsv_gz, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for r in cur.execute(sql, scan_endpoints):
            ep = r["endpoint"]
            lane = r["lane"]
            src = r["source_code"] or ""
            orig = r["origin"] or ""
            rel = clean_str(r["relation"])
            unit = clean_str(r["unit"])
            raw_unit = clean_str(r["raw_unit"])
            val = parse_float(r["value"])
            raw_val = parse_float(r["raw_value"])
            smi_ok = r["structure_model_eligible"] == "true" and bool(clean_str(r["canonical_smiles"]))

            # Fast pre-skip for MIC unlinked DBAASP rows to save JSON parsing time
            if ep == "MIC":
                if lane not in ("strict_numeric", "censored_numeric") or unit != "uM" or val is None or val <= 0:
                    continue
                if src == "DBAASP":
                    link = dbaasp_mic_linked.get(r["record_id"])
                    if not link:
                        continue
                    sp_ep, strain_cat, med, bridge_mode = link
                    rj = parse_json_safe(r["raw_json"])
                    cj = parse_json_safe(r["condition_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    rr = "strict_native" if lane == "strict_numeric" else "censored_inequality_preserved"
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, "uM", rel if rel else "=", f"{rr}:{bridge_mode}", 1 if lane == "strict_numeric" else 2,
                        cond_assay=med, cond_strain=strain_cat,
                    )
                    if out:
                        writer.writerow(out)
                        n_written += 1
                elif src == "CHEMBL":
                    cj = parse_json_safe(r["condition_json"])
                    sem, assay_desc, sp_raw, mat_raw, rt_raw, ed = extract_context_text(cj)
                    sp_ep, strain_cat = classify_mic_species(sp_raw or assay_desc)
                    if not sp_ep:
                        continue
                    rj = parse_json_safe(r["raw_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    rr = "strict_native" if lane == "strict_numeric" else "censored_inequality_preserved"
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, "uM", rel if rel else "=", f"{rr}:chembl_assay_organism", 1 if lane == "strict_numeric" else 2,
                        cond_assay=norm_mic_medium(assay_desc), cond_strain=strain_cat,
                    )
                    if out:
                        writer.writerow(out)
                        n_written += 1
                continue

            cj = parse_json_safe(r["condition_json"])
            rj = parse_json_safe(r["raw_json"])
            seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
            sem, assay_desc, sp_raw, mat_raw, rt_raw, ed = extract_context_text(cj)
            full_text = f"{assay_desc} {ed} {mat_raw} {sp_raw}".lower()

            # ==================== A. PERMEABILITY ====================
            if ep in ("PAMPA", "Caco2", "MDCK", "RRCK", "Papp", "permeability"):
                if not smi_ok:
                    continue
                target_ep = ep
                if ep in ("Papp", "permeability"):
                    if "caco-2" in full_text or "caco2" in full_text:
                        target_ep = "Caco2"
                    elif "mdck" in full_text:
                        target_ep = "MDCK"
                    elif "pampa" in full_text:
                        target_ep = "PAMPA"
                    elif "rrck" in full_text:
                        target_ep = "RRCK"
                    else:
                        continue

                # Determine direction / subtype
                direction = "unspecified"
                if any(k in full_text for k in ("b to a", "b->a", "basolateral to apical", "_ba")):
                    direction = "BA"
                elif any(k in full_text for k in ("a to b", "a->b", "apical to basolateral", "_ab", "donor-to-acceptor")):
                    direction = "AB"

                assay_tag = "unspecified"
                if target_ep == "PAMPA":
                    if "lecithin" in full_text or "dodecane" in full_text:
                        assay_tag = "pampa_lecithin_dodecane"
                    elif "git" in full_text:
                        assay_tag = "pampa_git"
                    else:
                        assay_tag = "pampa_generic"
                elif target_ep == "MDCK":
                    if "mdr1" in full_text or "pgp" in full_text or "p-gp" in full_text:
                        assay_tag = "mdck_mdr1"
                    elif "low-efflux" in full_text or "mdckii-le" in full_text or "le " in full_text:
                        assay_tag = "mdck_le"
                    else:
                        assay_tag = "mdck_wt"

                # Handle special PEGASUS MDCK below_LOQ -> left censored < 3e-7 cm/s
                if target_ep == "MDCK" and src == "PEGASUS supplementary dataset" and rel == "below_LOQ":
                    out = make_row(
                        "MDCK", "tier1", "regression", r, seq_raw, is_pure20,
                        3.0e-7, "cm/s", "<", "pegasus_mdck_below_loq_left_censored", 2,
                        cond_estimand=direction, cond_assay="mdck_mdr1",
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                    continue

                # Handle OLD CycPeptMPDB log10 permeability (NEVER -10 sentinel!)
                if src == "CycPeptMPDB" and orig == "OLD" and unit == "source_log10_permeability":
                    if rel != "=" or raw_val is None or raw_val <= -9.99 or raw_val > -2.0:
                        continue
                    cm_s = 10.0 ** raw_val
                    out = make_row(
                        target_ep, "tier1", "regression", r, seq_raw, is_pure20,
                        cm_s, "cm/s", "=", "old_cycpeptmpdb_non_sentinel_log10", 2,
                        cond_estimand=direction, cond_assay=assay_tag,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                    continue

                # Handle exact / censored linear cm/s or convertible units
                if rel in ("label", "censored_or_imputed", "between"):
                    continue
                if rel == "" and src not in ("swemacrocycledb_figshare_v1_ar_gt_0_3", "acs_jmedchem_5c02090", "acs_jmedchem_5c01901", "acsmedchemlett_5c00037"):
                    continue
                if rel not in ("=", "<", "<=", ">", ">=", ""):
                    continue
                # Skip CycPeptMPDB -10 sentinel in NEW if any
                if src == "CYCPEPTMPDB" and raw_val is not None and raw_val <= -9.99:
                    continue
                cm_s = is_permeability_cm_s_convertible(raw_unit, unit, val)
                if cm_s is None or cm_s <= 1e-10 or cm_s > 1e-2:
                    continue
                if lane == "strict_numeric":
                    rr = "strict_native" if ep == target_ep else f"strict_papp_mapped_{ep}"
                    pri = 1
                elif lane == "censored_numeric":
                    rr = "censored_inequality_preserved" if ep == target_ep else f"censored_papp_mapped_{ep}"
                    pri = 2
                else:
                    rr = f"review_verified_cms:{src}"
                    pri = 2
                out = make_row(
                    target_ep, "tier1", "regression", r, seq_raw, is_pure20,
                    cm_s, "cm/s", rel if rel else "=", rr, pri,
                    cond_estimand=direction, cond_assay=assay_tag,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== B. BBB ====================
            if ep == "BBB":
                if lane != "binary_evidence":
                    continue
                if val not in (0.0, 1.0):
                    continue
                out = make_row(
                    "BBB", "tier1", "classification", r, seq_raw, is_pure20,
                    int(val), "label01", "label", "binary_evidence_native", 2,
                    cond_assay="source_binary",
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== C. LogD7.4 ====================
            if ep in ("LogD7.4", "LogD"):
                if not smi_ok:
                    continue
                if rel in ("label", "between"):
                    continue
                if val is None or val < -6.0 or val > 10.0:
                    continue
                # Exclude calculated CLogP / MlogP / ALogP
                if any(w in full_text for w in ("calculated", "clogp", "mlogp", "alogp", "predicted")):
                    continue
                # Exclude 1,9-decadiene hydrocarbon/water partition (distinct physical quantity!)
                if "decadiene" in full_text:
                    continue
                # Must confirm pH 7.4 (explicitly in endpoint_detail, assay_desc, matrix, or endpoint=='LogD7.4' with no conflicting pH)
                has_other_ph = bool(re.search(r"ph\s*[1-689]|logd\s*[1-689]\.?[0-9]|ph\s*7\.[012356789]", full_text))
                if has_other_ph:
                    continue
                has_74 = ("7.4" in full_text) or ("ph7_4" in full_text) or ("ph74" in full_text) or (ep == "LogD7.4" and src == "PEGASUS supplementary dataset")
                if not has_74:
                    continue
                if rel == "" and src not in ("acs_jmedchem_5c02090", "CHEMBL"):
                    continue
                if rel not in ("=", "<", "<=", ">", ">=", ""):
                    continue
                if "chromlogd" in full_text or "hplc" in full_text or "chrom" in full_text:
                    method = "chromlogd_ph74"
                elif "octanol" in full_text or "shake" in full_text:
                    method = "shake_flask_octanol_ph74"
                else:
                    method = "unspecified_ph74"
                if lane == "strict_numeric":
                    rr = "strict_native"
                    pri = 1
                elif "missing_unit" in (r["reason"] or ""):
                    rr = "dimensionless_logd_ph74_unit_rescue"
                    pri = 1
                elif src == "Naylor LPE 2018 primary SI":
                    rr = "naylor_2018_octanol_pbs_ph74"
                    pri = 1
                else:
                    rr = f"verified_ph74_logd_rescue:{src}"
                    pri = 2
                out = make_row(
                    "LogD7.4", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "dimensionless_log10", rel if rel else "=", rr, pri,
                    cond_assay=method,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== D. SOLUBILITY (cls & reg) ====================
            if ep == "solubility":
                if lane == "binary_evidence" and val in (0.0, 1.0):
                    ctx = parse_json_safe(cj.get("source_context", ""))
                    solv = norm_solvent(ctx.get("matrix", "") or mat_raw)
                    out = make_row(
                        "solubility_cls", "tier1", "classification", r, seq_raw, is_pure20,
                        int(val), "label01", "label", "binary_evidence_solpepbench", 2,
                        cond_solvent=solv, cond_assay="source_binary",
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                    continue

                if lane == "review" and src == "classification_v9/solubility_qualitative_source_binary" and val in (0.0, 1.0):
                    out = make_row(
                        "solubility_cls", "tier1", "classification", r, seq_raw, is_pure20,
                        int(val), "label01", "label", "review_qualitative_solubility_binary", 3,
                        cond_solvent=norm_solvent(full_text), cond_assay="source_binary",
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                    continue

                # Quantitative solubility (uM or ug/mL with validated SMILES!)
                if not smi_ok:
                    continue
                if val is None or val <= 0:
                    continue
                if rel not in ("=", ">", ">=", "<", "<="):
                    continue
                solv = norm_solvent(full_text)
                if solv == "unspecified":
                    solv = "aqueous_buffer" if any(k in full_text for k in ("water", "buffer", "pbs", "ph", "aqueous", "kinetic", "thermodynamic", "pegasus", "naylor")) else "unspecified"
                atype = "kinetic" if "kinetic" in full_text else ("thermodynamic" if "thermo" in full_text else "unspecified")

                if unit == "uM":
                    if not (1e-4 <= val <= 1e6):
                        continue
                    rr = "strict_native" if lane == "strict_numeric" else ("censored_inequality_preserved" if lane == "censored_numeric" else f"verified_uM_solubility_rescue:{src}")
                    pri = 1 if lane == "strict_numeric" else 2
                    out = make_row(
                        "solubility_reg", "tier1", "regression", r, seq_raw, is_pure20,
                        val, "uM", rel, rr, pri,
                        cond_solvent=solv, cond_assay=atype,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                    # Also emit unambiguous aqueous binary solubility_cls (>=100 uM soluble=1, <10 uM insoluble=0)
                    if solv in ("pbs_ph74", "ultrapure_water", "aqueous_buffer", "aqueous_other", "unspecified"):
                        bval = None
                        brr = ""
                        if rel in ("=", ">", ">=") and val >= 100.0:
                            bval, brr = 1, "quant_solubility_ge100uM_soluble"
                        elif rel in ("=", "<", "<=") and val < 10.0:
                            bval, brr = 0, "quant_solubility_lt10uM_insoluble"
                        if bval is not None:
                            out2 = make_row(
                                "solubility_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                bval, "label01", "label", brr, 1,
                                cond_solvent="pbs_ph74" if solv == "pbs_ph74" else "aqueous_buffer",
                                cond_assay="quantitative_threshold",
                            )
                            if out2:
                                writer.writerow(out2); n_written += 1
                elif unit == "ug/mL" and src == "CHEMBL" and orig == "NEW":
                    # Validated SMILES -> convert on node23 with exact RDKit MW!
                    out = make_row(
                        "solubility_reg", "tier1", "regression", r, seq_raw, is_pure20,
                        val, "ug/mL_needs_mw", rel, "rdkit_mw_verified_ug_ml_to_uM", 2,
                        cond_solvent=solv, cond_assay=atype,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                continue

            # ==================== E. PPB ====================
            if ep == "PPB":
                # Exclude ChEMBL peptide extractor IC50/Inhibition false positives and nM units
                if unit in ("nM", "uM", "label01", "s", ""):
                    # Check PepTherDia review interval ranges like "43-49%" or "96-99%"
                    if src == "peptherdia_live_2026_07_14" and smi_ok and lane == "review":
                        ctx = parse_json_safe(cj.get("source_context", ""))
                        raw_txt = str(ctx.get("raw_endpoint_text", "") or "")
                        m = RANGE_PCT_RE.search(raw_txt)
                        if m:
                            lo, hi = float(m.group(1)) / 100.0, float(m.group(2)) / 100.0
                            if 0.0 <= lo <= hi <= 1.0 and (hi - lo) <= 0.35:
                                mid = 0.5 * (lo + hi)
                                out = make_row(
                                    "PPB", "tier1", "regression", r, seq_raw, is_pure20,
                                    mid, "fraction_bound", "interval", "peptherdia_range_interval_rescue", 2,
                                    val_lower=lo, val_upper=hi,
                                    cond_species=norm_species(sp_raw or full_text),
                                    cond_matrix=norm_ppb_matrix(mat_raw or full_text),
                                )
                                if out:
                                    writer.writerow(out); n_written += 1
                    continue
                if "herpes" in full_text or ed.lower() in ("ic50", "inhibition"):
                    continue
                if not smi_ok and not is_pure20:
                    continue
                frac = None
                rr_tag = ""
                if unit == "fraction_bound":
                    if val is not None and 0.0 <= val <= 1.0:
                        frac = val
                elif unit == "%bound":
                    if val is not None and 0.0 <= val <= 100.0:
                        frac = val / 100.0
                        rr_tag = "percent_bound_to_fraction_rescue"
                elif unit == "%unbound":
                    if val is not None and 0.0 <= val <= 100.0:
                        frac = 1.0 - (val / 100.0)
                        rr_tag = "percent_unbound_to_fraction_rescue"
                if frac is None or not (0.0 <= frac <= 1.0):
                    continue
                if rel not in ("=", ">", ">=", "<", "<=", "~", ""):
                    continue
                if rel == "" and src not in ("CHEMBL",):
                    continue
                if lane == "strict_numeric":
                    rr, pri = "strict_native", 1
                elif lane == "censored_numeric":
                    rr, pri = "censored_inequality_preserved", 2
                else:
                    rr = rr_tag or f"verified_ppb_review_rescue:{src}"
                    pri = 2 if rel in ("=", ">", ">=", "<", "<=") else 3
                out = make_row(
                    "PPB", "tier1", "regression", r, seq_raw, is_pure20,
                    frac, "fraction_bound", "=" if rel in ("", "~") else rel, rr, pri,
                    cond_species=norm_species(sp_raw or full_text),
                    cond_matrix=norm_ppb_matrix(mat_raw or full_text),
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== F. HALF_LIFE (reg & cls) ====================
            if ep in ("half_life", "half_life_terminal"):
                # 1. Binary half_life_cls native & ChEMBL candidates
                if ep == "half_life" and unit == "label01" and rel == "label":
                    if src == "PLIFEPRED" and val is not None and val > 1.0:
                        # PLIFEPRED values are in SECONDS! Convert to hours for half_life_reg!
                        h_val = val / 3600.0
                        if 1e-4 <= h_val <= 5000.0 and (smi_ok or is_pure20):
                            out = make_row(
                                "half_life_reg", "tier1", "regression", r, seq_raw, is_pure20,
                                h_val, "h", "=", "plifepred_seconds_to_hours_rescue", 2,
                                cond_species="unspecified", cond_matrix="in_vitro_blood",
                                cond_route="in_vitro", cond_estimand="t1_2_stability",
                            )
                            if out:
                                writer.writerow(out); n_written += 1
                            out_c = make_row(
                                "half_life_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                1 if h_val >= 1.0 else 0, "label01", "label", "derived_from_plifepred_ge1h", 2,
                                cond_species="unspecified", cond_matrix="in_vitro_blood", cond_route="in_vitro",
                            )
                            if out_c:
                                writer.writerow(out_c); n_written += 1
                        continue
                    if val in (0.0, 1.0) and (
                        lane == "binary_evidence"
                        or src in (
                            "classification_v9/plasma_T1/2_ge_1h_ChEMBL_candidate",
                            "classification_v9/serum_T1/2_ge_1h_ChEMBL_candidate",
                        )
                    ):
                        if not smi_ok and not seq_raw:
                            continue
                        mat_c = "in_vitro_plasma" if "plasma" in src.lower() else ("in_vitro_serum" if "serum" in src.lower() else norm_hl_matrix(full_text))
                        out_c = make_row(
                            "half_life_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            int(val), "label01", "label",
                            "binary_evidence_native" if lane == "binary_evidence" else f"chembl_ge1h_binary_rescue:{src}",
                            2 if lane == "binary_evidence" else 3,
                            cond_species=norm_species(sp_raw or full_text),
                            cond_matrix=mat_c,
                            cond_route=norm_route(rt_raw or full_text),
                        )
                        if out_c:
                            writer.writerow(out_c); n_written += 1
                    continue

                # 2. Numeric half_life_reg (hours)
                if unit != "h" or val is None or not (1e-4 <= val <= 5000.0):
                    continue
                if rel not in ("=", ">", ">=", "<", "<=", "~"):
                    continue
                if not smi_ok and not is_pure20:
                    # Also keep sequence-only PEPlife2 strict rows ONLY if they have clean canonical sequence; wait:
                    # if modified peptide has no SMILES, we can still keep it with seq_raw for sequence-only models,
                    # BUT never fabricate SMILES! Let's check: keeping rows with smi_ok OR seq_raw is great if
                    # modified-without-SMILES are clearly tagged smiles_origin='missing'!
                    if not seq_raw:
                        continue
                # Exclude non-PK enzyme inactivation / dissociation half-lives in ChEMBL
                if src in ("CHEMBL", "ChEMBL", "ChEMBL peptide endpoint extractor") and is_enzyme_or_non_pk_halflife(assay_desc):
                    drop_stats[("half_life_reg", "chembl_non_pk_enzyme_halflife")] += 1
                    continue
                if src == "ChEMBL peptide endpoint extractor" and not smi_ok:
                    # Covered by NEW CHEMBL with validated SMILES
                    continue

                # Enrich NEW PEPLIFE2 condition from condition_evidence
                extra_txt = ""
                if src == "PEPLIFE2" and orig == "NEW":
                    ce_info = peplife_ce.get((r["assay_id"], r["compound_id"]), {})
                    if ce_info:
                        rt_raw = rt_raw or ce_info.get("dose_route", "")
                        extra_txt = f" {ce_info.get('assay', '')} {ce_info.get('dose_route', '')}"

                comb_txt = f"{full_text} {extra_txt}"
                c_route = norm_route(rt_raw or comb_txt)
                c_species = norm_species(sp_raw or comb_txt)
                c_matrix = norm_hl_matrix(mat_raw or comb_txt, c_route)
                if ep == "half_life_terminal" or "terminal" in comb_txt:
                    c_est = "t1_2_terminal"
                elif c_matrix.startswith("in_vivo"):
                    c_est = "t1_2_in_vivo"
                elif c_matrix.startswith("in_vitro"):
                    c_est = "t1_2_in_vitro_stability"
                else:
                    c_est = "t1_2_unspecified"

                if lane == "strict_numeric":
                    rr, pri = "strict_native", 1
                elif lane == "censored_numeric":
                    rr, pri = "censored_inequality_preserved", 2
                elif ep == "half_life_terminal" and src == "PEPTHERDIA":
                    rr, pri = "peptherdia_terminal_half_life_rescue", 2
                elif src == "Pepti-DrugPhAI":
                    rr, pri = "pepti_drugphai_half_life_rescue", 2
                elif rel == "~":
                    rr, pri = f"approx_relation_half_life_rescue:{src}", 2
                elif src == "PEPlife2" and r["reason"] == "historical_semantic_review":
                    rr, pri = "peplife2_semantic_review_rescue", 2
                elif src in ("openFDA/DailyMed label", "openfda_dailymed_local_snapshots", "ema_smpc_product_information_2026_07_13", "primary_pk_literature_v14", "B3Pdb", "peptherdia_live_2026_07_14"):
                    rr, pri = f"verified_pk_half_life_rescue:{src}", 2
                else:
                    continue

                norm_rel = "=" if rel == "~" else rel
                out = make_row(
                    "half_life_reg", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "h", norm_rel, rr, pri,
                    cond_species=c_species, cond_matrix=c_matrix, cond_route=c_route, cond_estimand=c_est,
                )
                if out:
                    writer.writerow(out); n_written += 1

                # Derive clean binary half_life_cls (>= 1.0 h) whenever unambiguous
                bval = None
                if norm_rel == "=":
                    bval = 1 if val >= 1.0 else 0
                elif norm_rel in (">", ">=") and val >= 1.0:
                    bval = 1
                elif norm_rel in ("<", "<=") and val <= 1.0:
                    bval = 0
                if bval is not None:
                    out_c = make_row(
                        "half_life_cls", "tier1", "classification", r, seq_raw, is_pure20,
                        bval, "label01", "label", f"derived_from_half_life_reg_ge1h:{rr}", 1 if pri == 1 else 2,
                        cond_species=c_species, cond_matrix=c_matrix, cond_route=c_route,
                    )
                    if out_c:
                        writer.writerow(out_c); n_written += 1
                continue

            # ==================== G. CL_per_kg & CL_absolute ====================
            if ep in ("CL_per_kg", "CL_absolute", "CL"):
                if not smi_ok and not is_pure20:
                    continue
                if rel not in ("=", "<", "<=", ">", ">=", "~"):
                    continue
                if val is None or val <= 0:
                    continue
                c_route = norm_route(rt_raw or full_text)
                c_species = norm_species(sp_raw or full_text)
                c_est = norm_cl_estimand(full_text, c_route)
                if c_est == "CL_renal":
                    drop_stats[("CL", "excluded_renal_clearance")] += 1
                    continue

                if ep == "CL_per_kg" and unit == "mL/min/kg":
                    if not (1e-4 <= val <= 1000.0):
                        continue
                    if lane == "strict_numeric":
                        rr, pri = "strict_native", 1
                    elif lane == "censored_numeric":
                        rr, pri = "censored_inequality_preserved", 2
                    elif src in ("Pepti-DrugPhAI", "primary_pk_literature_v14", "ema_smpc_product_information_2026_07_13", "openFDA/DailyMed label", "openfda_dailymed_local_snapshots", "Helicogenic macrocycles 2026 primary SI", "ChEMBL"):
                        rr, pri = f"verified_cl_per_kg_rescue:{src}", 2
                    else:
                        continue
                    out = make_row(
                        "CL_per_kg", "tier1", "regression", r, seq_raw, is_pure20,
                        val, "mL/min/kg", "=" if rel == "~" else rel, rr, pri,
                        cond_species=c_species, cond_route=c_route, cond_estimand=c_est,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                elif ep == "CL_absolute" and unit == "L/h":
                    if not (1e-4 <= val <= 5000.0):
                        continue
                    if lane == "strict_numeric":
                        rr, pri = "strict_native", 1
                    elif lane == "censored_numeric":
                        rr, pri = "censored_inequality_preserved", 2
                    elif src in ("openFDA/DailyMed label", "openfda_dailymed_local_snapshots", "Pepti-DrugPhAI", "primary_pk_literature_v14", "ema_smpc_product_information_2026_07_13"):
                        rr, pri = f"verified_cl_abs_rescue:{src}", 2
                    else:
                        continue
                    out = make_row(
                        "CL_absolute", "tier1", "regression", r, seq_raw, is_pure20,
                        val, "L/h", "=" if rel == "~" else rel, rr, pri,
                        cond_species=c_species, cond_route=c_route, cond_estimand=c_est,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                elif ep == "CL" and unit in ("L/min", "ml/min", "mL/min"):
                    lh = val * 60.0 if unit == "L/min" else val * 0.06
                    if not (1e-4 <= lh <= 5000.0):
                        continue
                    out = make_row(
                        "CL_absolute", "tier1", "regression", r, seq_raw, is_pure20,
                        lh, "L/h", "=" if rel == "~" else rel, f"cl_unit_convert_{unit}_to_Lh:{src}", 2,
                        cond_species=c_species, cond_route=c_route, cond_estimand=c_est,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                continue

            # ==================== H. Vd_per_kg (unified with Vdss_per_kg) ====================
            if ep in ("Vd_per_kg", "Vdss_per_kg"):
                if not smi_ok and not is_pure20:
                    continue
                if unit != "L/kg" or val is None or not (1e-3 <= val <= 500.0):
                    continue
                if rel not in ("=", "<", "<=", ">", ">=", "~", ""):
                    continue
                if rel == "" and src != "acs_jmedchem_5c00544":
                    continue
                c_route = norm_route(rt_raw or full_text)
                c_species = norm_species(sp_raw or full_text)
                c_est = norm_vd_estimand(ep, full_text, c_route)
                if lane == "strict_numeric":
                    rr = "strict_native" if ep == "Vd_per_kg" else "unified_vdss_per_kg_strict"
                    pri = 1
                elif lane == "censored_numeric":
                    rr, pri = "censored_inequality_preserved", 2
                elif src in ("Pepti-DrugPhAI", "primary_pk_literature_v14", "ChEMBL", "openFDA/DailyMed label", "openfda_dailymed_local_snapshots", "ema_smpc_product_information_2026_07_13", "acs_jmedchem_5c00544"):
                    rr, pri = f"verified_vd_per_kg_rescue:{src}", 2
                else:
                    continue
                out = make_row(
                    "Vd_per_kg", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "L/kg", "=" if rel in ("", "~") else rel, rr, pri,
                    cond_species=c_species, cond_route=c_route, cond_estimand=c_est,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== I. Bioavailability F ====================
            if ep == "F":
                if not smi_ok and not is_pure20:
                    continue
                if unit != "fraction" or val is None or not (0.0 <= val <= 1.0):
                    continue
                if rel not in ("=", "<", "<=", ">", ">=", "~"):
                    continue
                c_route = norm_route(rt_raw or full_text)
                if c_route == "IV":
                    # IV bioavailability is 1.0 by definition (or route misparsed)
                    drop_stats[("F", "excluded_iv_route")] += 1
                    continue
                c_species = norm_species(sp_raw or full_text)
                qf = (r["quality_flags"] or "").lower()
                if "multiple_endpoint_values_in_sentence_context_mapping_needed" in qf:
                    drop_stats[("F", "excluded_ambiguous_multi_value_sentence")] += 1
                    continue
                if lane == "strict_numeric":
                    rr, pri = "strict_native", 1
                elif lane == "censored_numeric":
                    rr, pri = "censored_inequality_preserved", 2
                elif src in ("peptherdia_live_2026_07_14", "ChEMBL", "openFDA/DailyMed label", "openfda_dailymed_local_snapshots", "Helicogenic macrocycles 2026 primary SI", "PMC12990038 primary article tables"):
                    rr, pri = f"verified_bioavailability_rescue:{src}", 2
                else:
                    continue
                out = make_row(
                    "F", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "fraction", "=" if rel == "~" else rel, rr, pri,
                    cond_species=c_species, cond_route=c_route,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== J. Cmax ====================
            if ep == "Cmax":
                if lane not in ("strict_numeric", "censored_numeric"):
                    continue
                if not smi_ok or unit != "uM" or val is None or not (1e-5 <= val <= 1e6):
                    continue
                out = make_row(
                    "Cmax", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "uM", rel if rel else "=", "strict_native" if lane == "strict_numeric" else "censored_inequality_preserved",
                    1 if lane == "strict_numeric" else 2,
                    cond_species=norm_species(sp_raw or full_text),
                    cond_route=norm_route(rt_raw or full_text),
                    cond_dose=parse_cmax_dose_bucket(assay_desc),
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== K. OVERALL PEPTIDE TOXICITY ====================
            if ep == "overall_peptide_toxicity":
                if lane != "binary_evidence" or val not in (0.0, 1.0):
                    continue
                out = make_row(
                    "overall_peptide_toxicity", "tier1", "classification", r, seq_raw, is_pure20,
                    int(val), "label01", "label", "binary_evidence_native", 2,
                    cond_assay="source_binary",
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== L. HEMOLYSIS (cls & HC50) ====================
            if ep in ("hemolysis", "hemolysis_SOURCE_DEFINED_BINARY", "Hemolysis:Source Training Class", "hemolysis_percent", "hemolysis_HC50", "HC50"):
                rbc_sp = norm_species(sp_raw or mat_raw or full_text)

                # L1. Binary hemolysis sources
                if unit == "label01" and rel == "label" and val in (0.0, 1.0):
                    if lane == "binary_evidence":
                        pri = 2 if src in ("DBAASP_v4.0.1", "Hemolytik2_2026", "hemopi2_2025", "dramp_v4") else 3
                        out = make_row(
                            "hemolysis_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            int(val), "label01", "label", f"binary_evidence_native:{src}", pri,
                            cond_species=rbc_sp, cond_assay="source_binary",
                        )
                        if out:
                            writer.writerow(out); n_written += 1
                    elif src in ("HEMOPIMOD", "classification_v9/hemolysis_ChEMBL_source_binary_candidate"):
                        out = make_row(
                            "hemolysis_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            int(val), "label01", "label", f"binary_review_rescue:{src}", 3,
                            cond_species=rbc_sp, cond_assay="source_binary",
                        )
                        if out:
                            writer.writerow(out); n_written += 1
                    continue

                # L2. Pure HC50 quantitative + censored (for hemolysis_HC50 AND thresholded hemolysis_cls!)
                is_hc50 = (
                    ep in ("hemolysis_HC50", "HC50")
                    or (ep == "hemolysis" and ed in ("HC50_hemolytic_concentration", "HC50_uM_regression"))
                )
                if is_hc50 and val is not None and val > 0 and rel in ("=", ">", ">=", "<", "<="):
                    if unit == "uM" and (1e-3 <= val <= 1e5):
                        if not smi_ok and not is_pure20 and not seq_raw:
                            continue
                        if lane == "strict_numeric":
                            rr, pri = "strict_native", 1
                        elif lane == "censored_numeric":
                            rr, pri = "censored_inequality_preserved", 2
                        else:
                            rr = f"hc50_uM_review_rescue:{src}"
                            pri = 1 if rel == "=" else 2
                        out = make_row(
                            "hemolysis_HC50", "tier1", "regression", r, seq_raw, is_pure20,
                            val, "uM", rel, rr, pri,
                            cond_species=rbc_sp, cond_estimand="HC50",
                        )
                        if out:
                            writer.writerow(out); n_written += 1
                        # Unambiguous thresholded binary hemolysis_cls
                        bval, brr = None, ""
                        if rel in ("=", "<", "<=") and val <= 50.0:
                            bval, brr = 1, f"hc50_le50uM_hemolytic:{src}"
                        elif rel in ("=", ">", ">=") and val >= 100.0:
                            bval, brr = 0, f"hc50_ge100uM_or_censored_safe:{src}"
                        if bval is not None:
                            out_c = make_row(
                                "hemolysis_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                bval, "label01", "label", brr, 1,
                                cond_species=rbc_sp, cond_assay="quantitative_threshold",
                            )
                            if out_c:
                                writer.writerow(out_c); n_written += 1
                    elif unit == "ug/mL" and (smi_ok or is_pure20):
                        # Defer exact MW conversion to node23 RDKit!
                        out = make_row(
                            "hemolysis_HC50", "tier1", "regression", r, seq_raw, is_pure20,
                            val, "ug/mL_needs_mw", rel, f"hc50_ug_ml_rdkit_mw_rescue:{src}", 2,
                            cond_species=rbc_sp, cond_estimand="HC50",
                        )
                        if out:
                            writer.writerow(out); n_written += 1
                    continue

                # L3. % hemolysis at known exposure concentration (DBAASP_v4.0.1, Hemolytik2_2026) -> hemolysis_cls!
                if ep == "hemolysis" and unit == "percent" and ed == "hemolysis_percent_at_exposure_concentration":
                    if not (smi_ok or is_pure20):
                        continue
                    if val is None or not (0.0 <= val <= 105.0) or rel not in ("=", "<", "<=", ">", ">=", "~"):
                        continue
                    dose_v, dose_u = parse_conc_dose(clean_str(sem.get("dose")))
                    if dose_v is None or not dose_u:
                        continue
                    if dose_u == "uM":
                        bval, brr = None, ""
                        if rel in ("=", ">", ">=", "~") and val >= 20.0 and dose_v <= 50.0:
                            bval, brr = 1, f"pct_hemolysis_ge20_at_le50uM:{src}"
                        elif rel in ("=", ">", ">=", "~") and val >= 50.0 and dose_v <= 100.0:
                            bval, brr = 1, f"pct_hemolysis_ge50_at_le100uM:{src}"
                        elif rel in ("=", "<", "<=", "~") and val <= 10.0 and dose_v >= 50.0:
                            bval, brr = 0, f"pct_hemolysis_le10_at_ge50uM:{src}"
                        elif rel in ("=", "<", "<=", "~") and val < 20.0 and dose_v >= 100.0:
                            bval, brr = 0, f"pct_hemolysis_lt20_at_ge100uM:{src}"
                        if bval is not None:
                            out_c = make_row(
                                "hemolysis_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                bval, "label01", "label", brr, 1,
                                cond_species=rbc_sp, cond_assay="quantitative_threshold",
                            )
                            if out_c:
                                writer.writerow(out_c); n_written += 1
                    elif dose_u == "ug/mL":
                        # Defer exact MW conversion & thresholding to node23 RDKit!
                        out_c = make_row(
                            "hemolysis_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            -1, "pct_at_ug_mL_needs_mw", rel, f"pct_hemolysis_ug_ml_rdkit_mw_rescue:{src}", 1,
                            aux_conc_val=dose_v, aux_conc_unit="ug/mL", aux_pct_val=val,
                            cond_species=rbc_sp, cond_assay="quantitative_threshold",
                        )
                        if out_c:
                            writer.writerow(out_c); n_written += 1
                continue

            # ==================== M. CYTOTOXICITY_CLS ====================
            if ep in ("cytotoxicity", "cytotoxicity_SOURCE_DEFINED_BINARY", "CC50", "cytotoxicity_CC50",
                      "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50", "50% Cytotoxicity", "50% Cell death"):
                # Exclude non-cellular zebrafish embryo assays
                if "zebrafish" in full_text or "embryo" in full_text:
                    drop_stats[("cytotoxicity_cls", "excluded_zebrafish_embryo")] += 1
                    continue
                cyto_sp = norm_species(sp_raw or mat_raw or full_text)
                if any(k in full_text for k in ("hela", "mcf-7", "mcf7", "a549", "hepg2", "jurkat", "ht-29", "ht29", "k562", "cancer", "carcinoma", "leukemia", "melanoma", "tumor", "tumour", "hl-60", "pc-3", "u87", "b16", "hepa 1-6", "mt-4", "mt4")):
                    cell_grp = "cancer_cell_line"
                elif any(k in full_text for k in ("hek", "hacat", "pbmc", "fibroblast", "keratinocyte", "vero", "cos-7", "cho", "3t3", "l929", "macrophage", "raw 264", "raw264", "mrc-5", "huvec", "normal", "primary", "epithelial", "renal", "kidney")):
                    cell_grp = "normal_mammalian"
                else:
                    cell_grp = "unspecified"

                # M1. Binary sources
                if unit == "label01" and rel == "label" and val in (0.0, 1.0):
                    if lane == "binary_evidence" or src == "classification_v9/cell_line_cytotoxicity_ChEMBL_source_binary_candidate":
                        if not smi_ok and not is_pure20:
                            continue
                        out_c = make_row(
                            "cytotoxicity_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            int(val), "label01", "label",
                            f"binary_evidence_native:{src}" if lane == "binary_evidence" else f"binary_review_rescue:{src}",
                            2,
                            cond_species=cyto_sp, cond_matrix=cell_grp, cond_assay="source_binary",
                        )
                        if out_c:
                            writer.writerow(out_c); n_written += 1
                    continue

                # M2. Quantitative CC50 / IC50 / EC50 / LC50 / 50% Cytotoxicity / 50% Cell death
                is_ic50_like = (
                    ep in ("CC50", "cytotoxicity_CC50", "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50", "50% Cytotoxicity", "50% Cell death")
                    or (ep == "cytotoxicity" and any(k in ed.lower() for k in ("cc50", "ic50", "ec50", "lc50", "ld50", "gi50")))
                )
                if is_ic50_like and val is not None and val > 0 and rel in ("=", ">", ">=", "<", "<="):
                    if not (smi_ok or is_pure20):
                        continue
                    if unit == "uM" and (1e-4 <= val <= 1e5):
                        bval, brr = None, ""
                        if rel in ("=", "<", "<=") and val <= 25.0:
                            bval, brr = 1, f"cyto_cc50_ic50_le25uM_toxic:{src}"
                        elif rel in ("=", ">", ">=") and val >= 100.0:
                            bval, brr = 0, f"cyto_cc50_ic50_ge100uM_safe:{src}"
                        if bval is not None:
                            out_c = make_row(
                                "cytotoxicity_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                bval, "label01", "label", brr, 1,
                                cond_species=cyto_sp, cond_matrix=cell_grp, cond_assay="quantitative_threshold",
                            )
                            if out_c:
                                writer.writerow(out_c); n_written += 1
                    elif unit == "ug/mL":
                        out_c = make_row(
                            "cytotoxicity_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            val, "cyto_conc_ug_mL_needs_mw", rel, f"cyto_ug_ml_rdkit_mw_rescue:{src}", 1,
                            cond_species=cyto_sp, cond_matrix=cell_grp, cond_assay="quantitative_threshold",
                        )
                        if out_c:
                            writer.writerow(out_c); n_written += 1
                    continue

                # M3. % cytotoxicity at exposure concentration
                if ep == "cytotoxicity" and unit == "percent" and ed == "cytotoxicity_percent_at_exposure_concentration":
                    if not (smi_ok or is_pure20):
                        continue
                    if val is None or not (0.0 <= val <= 105.0) or rel not in ("=", "<", "<=", ">", ">=", "~"):
                        continue
                    dose_v, dose_u = parse_conc_dose(clean_str(sem.get("dose")))
                    if dose_v is None or not dose_u:
                        continue
                    if dose_u == "uM":
                        bval, brr = None, ""
                        if rel in ("=", ">", ">=", "~") and val >= 50.0 and dose_v <= 50.0:
                            bval, brr = 1, f"cyto_pct_ge50_at_le50uM:{src}"
                        elif rel in ("=", "<", "<=", "~") and val <= 15.0 and dose_v >= 50.0:
                            bval, brr = 0, f"cyto_pct_le15_at_ge50uM:{src}"
                        if bval is not None:
                            out_c = make_row(
                                "cytotoxicity_cls", "tier1", "classification", r, seq_raw, is_pure20,
                                bval, "label01", "label", brr, 1,
                                cond_species=cyto_sp, cond_matrix=cell_grp, cond_assay="quantitative_threshold",
                            )
                            if out_c:
                                writer.writerow(out_c); n_written += 1
                    elif dose_u == "ug/mL":
                        out_c = make_row(
                            "cytotoxicity_cls", "tier1", "classification", r, seq_raw, is_pure20,
                            -1, "cyto_pct_at_ug_mL_needs_mw", rel, f"cyto_pct_ug_ml_rdkit_mw_rescue:{src}", 1,
                            aux_conc_val=dose_v, aux_conc_unit="ug/mL", aux_pct_val=val,
                            cond_species=cyto_sp, cond_matrix=cell_grp, cond_assay="quantitative_threshold",
                        )
                        if out_c:
                            writer.writerow(out_c); n_written += 1
                continue

    con.close()

    # Extract Case17 raw queries
    print("[4/5] Extracting Case17 queries from case17_raw_responses.json & case17_reference_values.tsv...", flush=True)
    c17_json = json.loads(CASE17_RAW.read_text(encoding="utf-8"))
    ref_map = {}
    with CASE17_REF.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ref_map[row["name"]] = row
    c17_out = OUT_DIR / "case17_raw.tsv"
    with c17_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drug_name", "source_smiles", "half_life_ref", "ppb_ref"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        for name, smi in c17_json["input"]:
            ref = ref_map.get(name, {})
            w.writerow({
                "drug_name": name,
                "source_smiles": smi,
                "half_life_ref": ref.get("half_life_ref", ""),
                "ppb_ref": ref.get("ppb_ref", ""),
            })

    # Write extraction audit summary
    audit = {
        "elapsed_sec": round(time.time() - t0, 2),
        "total_candidate_rows": n_written,
        "by_endpoint": dict(sorted(stats.items())),
        "by_rescue_reason": {f"{k[0]}|{k[1]}": v for k, v in sorted(rescue_stats.items())},
        "drops": {f"{k[0]}|{k[1]}": v for k, v in sorted(drop_stats.items())},
    }
    (OUT_DIR / "extraction_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[5/5] DONE in {audit['elapsed_sec']}s! Total candidate rows written: {n_written:,}")
    print("Per-endpoint candidate row counts:")
    for ep, cnt in sorted(stats.items()):
        print(f"  {ep:28s}: {cnt:6,d}")
    print("Drops:", dict(drop_stats))


if __name__ == "__main__":
    main()
