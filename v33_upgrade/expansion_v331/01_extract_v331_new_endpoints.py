#!/usr/bin/env python3
"""
Wave 5A — Step 01: Extract all Group A (Tier 1) + Group B (Tier 2) candidate new endpoints
for Peptide OmniPanel V33.1 from frozen read-only catalogue.sqlite.

Strict scientific & zero-leakage rules:
- Opens catalogue.sqlite strictly read-only (?mode=ro).
- Leaves ALL 24 existing V33 endpoints 100% untouched (does NOT re-extract them).
- NEVER fabricates SMILES for modified peptides; ONLY flags `is_pure_unmodified_20aa = 1`
  for provably unmodified canonical 20-AA linear peptides (including verified dramp_v4 rows
  whose sequence & modification metadata reside in raw_json).
- Links ALL DBAASP NEW microbial/biofilm/lethal records (MIC, MBC, MFC, MBIC*, MBEC*, LC_lemma)
  AND DBAASP NEW mammalian cytotoxicity cells ONLY via the mathematically proven,
  order-independent exact multiset bijection (`orig_cnt == ctx_cnt` & consistent unit).
- Excludes zebrafish/embryo from cytotoxicity_CC50; excludes calculated CLogP/ALogP/MLogP,
  pH-buffered LogD, and non-octanol solvents from LogP.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DB_PATH = Path(r"D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite")
OUT_DIR = Path(r"D:\work\中检院\v33_upgrade\expansion_v331\staging")

AA20_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{2,120}$")
HELM_PURE20_RE = re.compile(r"^PEPTIDE1\{[ACDEFGHIKLMNPQRSTVWY](\.[ACDEFGHIKLMNPQRSTVWY]){1,119}\}\$\$\$\$$")
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


def parse_sig(value, qualifier="", lower="", upper=""):
    """Exact order-independent signature for DBAASP multiset bijection (scalar or range)."""
    v = clean_str(value).replace("≤", "<=").replace("≥", ">=").replace("≈", "~").replace("–", "-")
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


def norm_species(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unspecified"
    if any(k in t for k in ("homo sapiens", "human", "patient", "volunteer", "adult", "infant", "clinical", "hela", "hek", "hacat", "pbmc", "hepg2", "mcf-7", "mcf7", "a549", "jurkat", "ht-29", "ht29", "k562", "hl-60", "mrc-5", "huvec", "u87", "pc-3", "hff", "hsf", "sf188", "d283", "h460")):
        return "human"
    if any(k in t for k in ("rattus", "rat", "sprague", "wistar")):
        return "rat"
    if any(k in t for k in ("mus musculus", "mouse", "mice", "murine", "balb", "icr", "c57bl", "cd-1", "3t3", "l929", "raw 264", "raw264", "b16", "hepa 1-6")):
        return "mouse"
    if any(k in t for k in ("canis", "dog", "beagle", "mdck")):
        return "dog"
    if any(k in t for k in ("macaca", "monkey", "cynomolgus", "rhesus", "primate", "vero", "cos-7", "cos7")):
        return "monkey"
    if any(k in t for k in ("oryctolagus", "rabbit")):
        return "rabbit"
    if any(k in t for k in ("ovis", "sheep", "ovine")):
        return "sheep"
    if any(k in t for k in ("bos taurus", "bovine", "calf", "cow")):
        return "bovine"
    if any(k in t for k in ("pig", "porcine", "sus scrofa", "guinea pig", "hamster", "cho", "horse", "equine", "feline", "fcwf")):
        return "other"
    return "unspecified"


def norm_route(text: str) -> str:
    t = (text or "").lower()
    if not t:
        return "unspecified"
    if re.search(r"\b(iv|i\.v\.|intravenous|bolus|infusion)\b", t):
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


def norm_vd_estimand(endpoint: str, text: str, route: str = "") -> str:
    t = (text or "").lower()
    if any(k in t for k in ("vdss", "vss", "steady state", "steady-state")):
        return "Vss"
    if any(k in t for k in ("vz/f", "vd/f", "v/f", "apparent volume", "v_over_f")) or (route in ("PO", "SC", "IM") and "steady" not in t):
        if route != "IV":
            return "V_F"
    if any(k in t for k in ("vz", "terminal", "beta")):
        return "Vz"
    if any(k in t for k in ("vd", "volume of distribution")):
        return "Vd"
    return "Vd_unspecified"


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
    if "tsb" in t or "tryptic soy" in t or "tryptone soya" in t:
        return "TSB"
    if "ypd" in t or "sabouraud" in t or "sda" in t or "sdb" in t:
        return "fungal_broth"
    if "bhi" in t or "brain heart" in t:
        return "BHI"
    return "other"


def extract_context_text(cj: dict) -> tuple[dict, str, str, str, str, str]:
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


def check_pure_unmodified_20aa(row, rj: dict, cj: dict) -> tuple[str, int]:
    """
    Strictly verify whether a record's sequence is a provably unmodified pure 20-AA linear peptide
    eligible for deterministic RDKit Chem.MolFromSequence(seq) when source SMILES is missing.
    Also handles dramp_v4 rows where `records.sequence` was blank but `raw_json["sequence"]`
    and `raw_json["context_json"]` explicitly specify Free N/C-termini, Other_Modifications=None, Linear.
    NEVER returns 1 for modified, cyclic, branched, D-AA, or 'Not included yet' peptides!
    """
    seq = clean_str(row["sequence"]).upper()
    from_dramp_rj = False
    if not seq and row["source_code"] == "dramp_v4":
        seq = clean_str(rj.get("sequence")).upper()
        from_dramp_rj = bool(seq)

    if not AA20_RE.match(seq):
        return ("", 0)

    helm = clean_str(rj.get("helm", ""))
    if helm and HELM_PURE20_RE.match(helm):
        return (seq, 1)

    if from_dramp_rj:
        ctx = parse_json_safe(rj.get("context_json"))
        nmod = str(ctx.get("N-terminal_Modification", "") or "").strip().lower()
        cmod = str(ctx.get("C-terminal_Modification", "") or "").strip().lower()
        omod = str(ctx.get("Other_Modifications", "") or "").strip().lower()
        lin = str(ctx.get("Linear/Cyclic/Branched", "") or "").strip().lower()
        # Require explicit Free/None on N-term, C-term, Other, and Linear (rejects 'Not included yet', Amidation, Disulfide, etc.)
        if nmod in NON_MOD_STRINGS and cmod in NON_MOD_STRINGS and omod in NON_MOD_STRINGS and lin == "linear":
            return (seq, 1)
        return (seq, 0)

    if row["sequence_kind"] != "bare_standard" or str(row["has_modifications"]).lower() != "false":
        return (seq, 0)

    qf = (row["quality_flags"] or "").lower()
    if any(bad in qf for bad in ("residue_projection_only", "preserves_d_amino_acids", "n_methylation", "cyclization", "thioamide", "modified_sequence")):
        return (seq, 0)

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


# --- Species Classifiers for Group B ---
def classify_new_mic_species(name: str) -> tuple[str | None, str]:
    """Classify ONLY NEW species-specific MIC targets (excluding the 4 existing V33 MICs!)."""
    if not name:
        return (None, "unspecified")
    s = name.lower()
    sp = None
    if "bacillus subtilis" in s or "b. subtilis" in s or "b.subtilis" in s:
        sp = "MIC_B_subtilis"
    elif "acinetobacter baumannii" in s or "a. baumannii" in s or "a.baumannii" in s:
        sp = "MIC_A_baumannii"
    elif "klebsiella pneumoniae" in s or "k. pneumoniae" in s or "k.pneumoniae" in s:
        sp = "MIC_K_pneumoniae"
    elif "staphylococcus epidermidis" in s or "s. epidermidis" in s or "s.epidermidis" in s:
        sp = "MIC_S_epidermidis"
    elif "salmonella enterica" in s or "salmonella typhimurium" in s or "s. typhimurium" in s or "s. enterica" in s:
        sp = "MIC_Salmonella"
    elif "enterococcus faecium" in s or "e. faecium" in s or "e.faecium" in s:
        sp = "MIC_E_faecium"
    elif "enterococcus faecalis" in s or "e. faecalis" in s or "e.faecalis" in s:
        sp = "MIC_E_faecalis"
    elif "listeria monocytogenes" in s or "l. monocytogenes" in s or "l.monocytogenes" in s:
        sp = "MIC_L_monocytogenes"
    elif "cryptococcus neoformans" in s or "c. neoformans" in s or "c.neoformans" in s:
        sp = "MIC_C_neoformans"
    if not sp:
        return (None, "unspecified")
    if any(k in s for k in ("vre", "vancomycin-resistant", "vancomycin resistant", "mdr", "multidrug", "carbapenem", "crkp", "crab")):
        st = "MDR_resistant"
    elif "atcc" in s or "nctc" in s or "dsm" in s or "mtcc" in s or "kctc" in s or "cctcc" in s:
        st = "ATCC_ref"
    else:
        st = "unspecified"
    return (sp, st)


def classify_mbc_species(name: str) -> tuple[str | None, str]:
    if not name:
        return (None, "unspecified")
    s = name.lower()
    sp = None
    if "escherichia coli" in s or "e. coli" in s or "e.coli" in s:
        sp = "MBC_E_coli"
    elif "staphylococcus aureus" in s or "s. aureus" in s or "s.aureus" in s:
        sp = "MBC_S_aureus"
    elif "pseudomonas aeruginosa" in s or "p. aeruginosa" in s or "p.aeruginosa" in s:
        sp = "MBC_P_aeruginosa"
    elif "acinetobacter baumannii" in s or "a. baumannii" in s or "a.baumannii" in s:
        sp = "MBC_A_baumannii"
    elif "klebsiella pneumoniae" in s or "k. pneumoniae" in s or "k.pneumoniae" in s:
        sp = "MBC_K_pneumoniae"
    elif "bacillus subtilis" in s or "b. subtilis" in s or "b.subtilis" in s:
        sp = "MBC_B_subtilis"
    elif "staphylococcus epidermidis" in s or "s. epidermidis" in s or "s.epidermidis" in s:
        sp = "MBC_S_epidermidis"
    elif "enterococcus faecalis" in s or "e. faecalis" in s or "e.faecalis" in s:
        sp = "MBC_E_faecalis"
    if not sp:
        return (None, "unspecified")
    if any(k in s for k in ("mrsa", "usa300", "43300", "methicillin-resistant", "mdr", "vre")):
        st = "MDR_MRSA"
    elif "atcc" in s or "nctc" in s or "dsm" in s:
        st = "ATCC_ref"
    else:
        st = "unspecified"
    return (sp, st)


FUNGAL_PATTERNS = [
    ("candida_albicans", ("candida albicans", "c. albicans", "c.albicans")),
    ("candida_krusei", ("candida krusei", "pichia kudriavzevii", "issatchenkia orientalis")),
    ("candida_glabrata", ("candida glabrata", "nakaseomyces glabratus")),
    ("candida_tropicalis", ("candida tropicalis",)),
    ("candida_dubliniensis", ("candida dubliniensis",)),
    ("candida_parapsilosis", ("candida parapsilosis",)),
    ("candida_other", ("candida auris", "candida boidinii", "candida guilliermondii", "candida lusitaniae", "candida kefyr", "candida utilis", "candida sp")),
    ("cryptococcus_neoformans", ("cryptococcus neoformans", "c. neoformans")),
    ("cryptococcus_other", ("cryptococcus gattii", "cryptococcus albidus", "cryptococcus sp")),
    ("saccharomyces_cerevisiae", ("saccharomyces cerevisiae", "s. cerevisiae")),
    ("aspergillus", ("aspergillus fumigatus", "aspergillus niger", "aspergillus flavus", "aspergillus terreus", "aspergillus nidulans", "aspergillus sp")),
    ("other_fungus", ("fusarium", "trichophyton", "rhizopus", "curvularia", "microsporum", "epidermophyton", "malassezia", "penicillium", "botrytis", "alternaria", "neurospora", "schizosaccharomyces", "kluyveromyces")),
]


def classify_fungal_species(name: str) -> str | None:
    """Strictly return canonical fungal species tag, or None for any non-fungal organism!"""
    s = (name or "").lower()
    if not s:
        return None
    for tag, keys in FUNGAL_PATTERNS:
        if any(k in s for k in keys):
            return tag
    return None


def norm_microbe_species_tag(name: str) -> str:
    """Canonical species tag for pooled multi-species biofilm/LC endpoints."""
    s = (name or "").lower()
    if not s:
        return "unspecified"
    checks = [
        ("staphylococcus_aureus", ("staphylococcus aureus", "s. aureus", "s.aureus")),
        ("pseudomonas_aeruginosa", ("pseudomonas aeruginosa", "p. aeruginosa", "p.aeruginosa")),
        ("escherichia_coli", ("escherichia coli", "e. coli", "e.coli")),
        ("acinetobacter_baumannii", ("acinetobacter baumannii", "a. baumannii")),
        ("klebsiella_pneumoniae", ("klebsiella pneumoniae", "k. pneumoniae")),
        ("staphylococcus_epidermidis", ("staphylococcus epidermidis", "s. epidermidis")),
        ("enterococcus_faecalis", ("enterococcus faecalis", "e. faecalis")),
        ("enterococcus_faecium", ("enterococcus faecium", "e. faecium")),
        ("bacillus_subtilis", ("bacillus subtilis", "b. subtilis")),
        ("bacillus_megaterium", ("bacillus megaterium",)),
        ("salmonella", ("salmonella enterica", "salmonella typhimurium", "s. typhimurium", "s. enterica")),
        ("candida_albicans", ("candida albicans", "c. albicans")),
        ("streptococcus_mutans", ("streptococcus mutans",)),
        ("micrococcus_luteus", ("micrococcus luteus",)),
        ("listeria_monocytogenes", ("listeria monocytogenes",)),
        ("serratia_marcescens", ("serratia marcescens",)),
    ]
    for tag, keys in checks:
        if any(k in s for k in keys):
            return tag
    return "other"


def norm_cell_group(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in (
        "hela", "mcf-7", "mcf7", "a549", "hepg2", "jurkat", "ht-29", "ht29", "k562", "cancer",
        "carcinoma", "leukemia", "leukaemia", "melanoma", "tumor", "tumour", "hl-60", "hl60",
        "pc-3", "pc3", "u87", "u251", "b16", "hepa 1-6", "mt-4", "mt4", "d283", "sf188", "h460",
        "mda-mb", "sk-br", "hct116", "hct-116", "caco-2", "caco2", "sw480", "lncap", "du145",
        "a375", "u937", "thp-1", "thp1", "molt-4", "raji", "sp2/0", "p388", "l1210", "glioma",
        "sarcoma", "myeloma", "lymphoma", "neuroblastoma", "osteosarcoma", "ovcar", "skov",
    )):
        return "cancer_cell_line"
    if any(k in t for k in (
        "hek", "hacat", "pbmc", "fibroblast", "keratinocyte", "vero", "cos-7", "cos7", "cho",
        "3t3", "l929", "macrophage", "raw 264", "raw264", "mrc-5", "mrc5", "huvec", "normal",
        "primary", "epithelial", "renal", "kidney", "hff", "hsf", "wi-38", "wi38", "bj ", "mdck",
        "monocyte", "lymphocyte", "splenocyte", "hepatocyte", "erythrocyte", "astrocyte", "fcwf",
    )):
        return "normal_mammalian"
    return "unspecified"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Build mathematically proven exact DBAASP multiset bijection (covers MIC, MBC, MFC, MBIC*, MBEC*, LC_lemma, AND targetCell for CC50/50% Cytotoxicity!)
    print("[1/3] Running mathematically proven exact DBAASP multiset bijection...", flush=True)
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

    # record_id -> (target_species_name, target_cell_name, medium_name, bridge_mode)
    dbaasp_microbe_linked = {}
    dbaasp_cell_linked = {}

    microbe_eps = {"MIC", "MBC", "MFC", "MBIC", "MBIC50", "MBIC90", "MBEC", "MBEC50", "MBEC90", "LC_lemma"}
    cyto_dbaasp_eps = {"CC50", "50% Cytotoxicity", "50% Cell death"}

    for key, obs_list in dbaasp_scopes.items():
        entries = dbaasp_indexed.get(key)
        if not entries:
            continue
        orig_cnt = Counter(o["sig"] for o in obs_list)
        ctx_sigs = [(idx, parse_sig(e.get("concentration")), e) for idx, e in entries.items()]
        ctx_cnt = Counter(s for _, s, _ in ctx_sigs)
        units = {o["unit"] for o in obs_list}
        if None in orig_cnt or None in ctx_cnt or orig_cnt != ctx_cnt:
            continue
        if len(units) != 1 or "" in units:
            continue
        sig_to_ctx = defaultdict(list)
        for idx, s, e in ctx_sigs:
            sig_to_ctx[s].append(e)

        for o in obs_list:
            ep = o["endpoint"]
            if o["sig"][0] != "scalar":
                continue
            cands = sig_to_ctx[o["sig"]]
            if ep in microbe_eps:
                if any(c.get("targetCell.name") and not c.get("targetSpecies.name") for c in cands):
                    continue
                sp_names = [clean_str(c.get("targetSpecies.name", "")) for c in cands]
                if any(not x for x in sp_names):
                    continue
                if len(cands) == 1:
                    dbaasp_microbe_linked[o["record_id"]] = (
                        sp_names[0],
                        norm_mic_medium(cands[0].get("medium.name", "")),
                        "dbaasp_exact_multiset_unique_sig",
                    )
                else:
                    # Check unanimous canonical species across all tied candidates!
                    genus_set = {" ".join(x.lower().split()[:2]) for x in sp_names}
                    if len(genus_set) == 1:
                        meds = {norm_mic_medium(c.get("medium.name", "")) for c in cands}
                        med = next(iter(meds)) if len(meds) == 1 else "unspecified"
                        dbaasp_microbe_linked[o["record_id"]] = (
                            sp_names[0],
                            med,
                            "dbaasp_exact_multiset_same_species",
                        )
            elif ep in cyto_dbaasp_eps:
                cell_names = [clean_str(c.get("targetCell.name", "")) for c in cands]
                if cell_names and all(cell_names):
                    if len(set(cell_names)) == 1:
                        dbaasp_cell_linked[o["record_id"]] = cell_names[0]

    print(f"  Linked {len(dbaasp_microbe_linked):,} DBAASP microbial/biofilm/LC records + {len(dbaasp_cell_linked):,} DBAASP cell-line tags.", flush=True)
    del dbaasp_scopes, dbaasp_indexed

    # 2. Scan all candidate raw endpoints in one SQL query
    scan_endpoints = [
        "efflux_ratio",
        "CC50", "cytotoxicity_CC50", "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50",
        "50% Cytotoxicity", "50% Cell death", "cytotoxicity",
        "hemolysis_MHC", "MHC", "hemolysis_MHC10", "MHC10", "HC10", "hemolysis_HC10", "hemolysis",
        "Vd_absolute", "Vd",
        "LogP",
        "MIC", "MBC", "MFC", "MBIC", "MBIC50", "MBIC90", "MBEC", "MBEC50", "MBEC90", "LC_lemma",
    ]
    fields = [
        "endpoint_v331", "task_tier", "task_type",
        "record_id", "origin", "source_code", "source_record_id", "endpoint_raw",
        "lane_orig", "reason_orig", "rescue_reason", "evidence_priority",
        "canonical_smiles_src", "sequence_raw", "sequence_kind", "has_modifications", "is_pure_unmodified_20aa",
        "val_num", "val_unit", "val_relation", "val_lower", "val_upper",
        "cond_species", "cond_matrix", "cond_route", "cond_estimand", "cond_assay", "cond_strain",
    ]

    stats = Counter()
    rescue_stats = Counter()
    drop_stats = Counter()

    def make_row(
        ep_v331, tier, ttype, r, seq_raw, is_pure20,
        val_num, val_unit, val_rel, rescue_reason, priority,
        val_lower="", val_upper="",
        cond_species="unspecified", cond_matrix="unspecified", cond_route="unspecified",
        cond_estimand="unspecified", cond_assay="unspecified", cond_strain="unspecified",
    ):
        smi = clean_str(r["canonical_smiles"]) if r["structure_model_eligible"] == "true" else ""
        if not smi and not seq_raw:
            drop_stats[(ep_v331, "no_smiles_and_no_seq")] += 1
            return None
        stats[ep_v331] += 1
        rescue_stats[(ep_v331, rescue_reason)] += 1
        return {
            "endpoint_v331": ep_v331,
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
            "cond_species": cond_species,
            "cond_matrix": cond_matrix,
            "cond_route": cond_route,
            "cond_estimand": cond_estimand,
            "cond_assay": cond_assay,
            "cond_strain": cond_strain,
        }

    print("[2/3] Scanning records_with_structure_qc for V33.1 candidate endpoints...", flush=True)
    ph = ",".join("?" for _ in scan_endpoints)
    sql = f"""
        SELECT record_id, origin, source_code, source_record_id, compound_id, assay_id,
               endpoint, lane, reason, raw_value, raw_unit, value, lower, upper, relation, unit,
               condition_json, sequence, sequence_kind, has_modifications, quality_flags,
               structure_model_eligible, canonical_smiles, raw_json
        FROM records_with_structure_qc
        WHERE endpoint IN ({ph})
          AND lane IN ('strict_numeric', 'censored_numeric', 'review')
    """

    out_tsv_gz = OUT_DIR / "raw_candidates_v331.tsv.gz"
    n_written = 0

    with gzip.open(out_tsv_gz, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()

        for r in cur.execute(sql, scan_endpoints):
            ep = r["endpoint"]
            lane = r["lane"]
            src = r["source_code"] or ""
            orig = r["origin"] or ""
            rel = clean_str(r["relation"]).replace("≥", ">=").replace("≤", "<=").replace(">>", ">").replace("<<", "<")
            unit = clean_str(r["unit"])
            val = parse_float(r["value"])
            lo_v = parse_float(r["lower"])
            hi_v = parse_float(r["upper"])
            smi_ok = r["structure_model_eligible"] == "true" and bool(clean_str(r["canonical_smiles"]))

            # ==================== GROUP B: MIC (NEW SPECIES ONLY!) ====================
            if ep == "MIC":
                if val is None or val <= 0:
                    continue
                if rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                if src == "DBAASP":
                    if lane not in ("strict_numeric", "censored_numeric") or unit != "uM":
                        continue
                    link = dbaasp_microbe_linked.get(r["record_id"])
                    if not link:
                        continue
                    sp_name, med, bmode = link
                    sp_ep, st_cat = classify_new_mic_species(sp_name)
                    if not sp_ep:
                        continue
                    rj = parse_json_safe(r["raw_json"])
                    cj = parse_json_safe(r["condition_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    rr = "strict_native" if lane == "strict_numeric" else "censored_inequality_preserved"
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, "uM", rel or "=", f"{rr}:{bmode}", 1 if lane == "strict_numeric" else 2,
                        cond_assay=med, cond_strain=st_cat,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                elif src == "CHEMBL":
                    if unit not in ("uM", "ug/mL"):
                        continue
                    if unit == "ug/mL" and not smi_ok:
                        continue
                    cj = parse_json_safe(r["condition_json"])
                    sem, assay_desc, sp_raw, mat_raw, rt_raw, ed = extract_context_text(cj)
                    sp_ep, st_cat = classify_new_mic_species(sp_raw or assay_desc)
                    if not sp_ep:
                        continue
                    rj = parse_json_safe(r["raw_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    u_out = "uM" if unit == "uM" else "ug/mL_needs_mw"
                    if u_out == "uM" and not (1e-4 <= val <= 1e5):
                        continue
                    rr = "strict_native:chembl" if (lane == "strict_numeric" and unit == "uM") else (
                        "censored_inequality:chembl" if (lane == "censored_numeric" and unit == "uM") else "chembl_ug_ml_rdkit_mw"
                    )
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, u_out, rel or "=", rr, 1 if lane == "strict_numeric" else 2,
                        cond_assay=norm_mic_medium(assay_desc), cond_strain=st_cat,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                continue

            # ==================== GROUP B: MBC (SPECIES-SPECIFIC) ====================
            if ep == "MBC":
                if val is None or val <= 0 or rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                if src == "DBAASP":
                    if lane not in ("strict_numeric", "censored_numeric") or unit != "uM":
                        continue
                    link = dbaasp_microbe_linked.get(r["record_id"])
                    if not link:
                        continue
                    sp_name, med, bmode = link
                    sp_ep, st_cat = classify_mbc_species(sp_name)
                    if not sp_ep:
                        continue
                    rj = parse_json_safe(r["raw_json"])
                    cj = parse_json_safe(r["condition_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    rr = "strict_native" if lane == "strict_numeric" else "censored_inequality_preserved"
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, "uM", rel or "=", f"{rr}:{bmode}", 1 if lane == "strict_numeric" else 2,
                        cond_assay=med, cond_strain=st_cat,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                elif src == "CHEMBL":
                    if unit not in ("uM", "ug/mL"):
                        continue
                    if unit == "ug/mL" and not smi_ok:
                        continue
                    cj = parse_json_safe(r["condition_json"])
                    sem, assay_desc, sp_raw, mat_raw, rt_raw, ed = extract_context_text(cj)
                    sp_ep, st_cat = classify_mbc_species(sp_raw or assay_desc)
                    if not sp_ep:
                        continue
                    rj = parse_json_safe(r["raw_json"])
                    seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                    u_out = "uM" if unit == "uM" else "ug/mL_needs_mw"
                    if u_out == "uM" and not (1e-4 <= val <= 1e5):
                        continue
                    rr = "strict_native:chembl" if (lane == "strict_numeric" and unit == "uM") else (
                        "censored_inequality:chembl" if (lane == "censored_numeric" and unit == "uM") else "chembl_ug_ml_rdkit_mw"
                    )
                    out = make_row(
                        sp_ep, "tier2", "regression", r, seq_raw, is_pure20,
                        val, u_out, rel or "=", rr, 1 if lane == "strict_numeric" else 2,
                        cond_assay=norm_mic_medium(assay_desc), cond_strain=st_cat,
                    )
                    if out:
                        writer.writerow(out); n_written += 1
                continue

            # ==================== GROUP B: MFC (FUNGAL ONLY!) ====================
            if ep == "MFC":
                if val is None or val <= 0 or rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                sp_name = ""
                med = "unspecified"
                bmode = ""
                if src == "DBAASP":
                    if lane not in ("strict_numeric", "censored_numeric") or unit != "uM":
                        continue
                    link = dbaasp_microbe_linked.get(r["record_id"])
                    if not link:
                        continue
                    sp_name, med, bmode = link
                elif src == "CHEMBL":
                    if unit not in ("uM", "ug/mL"):
                        continue
                    if unit == "ug/mL" and not smi_ok:
                        continue
                    cj_tmp = parse_json_safe(r["condition_json"])
                    _, assay_desc, sp_raw, _, _, _ = extract_context_text(cj_tmp)
                    sp_name = sp_raw or assay_desc
                    med = norm_mic_medium(assay_desc)
                    bmode = "chembl_assay_organism"
                else:
                    continue

                f_tag = classify_fungal_species(sp_name)
                if not f_tag:
                    drop_stats[("MFC", "excluded_non_fungal")] += 1
                    continue

                rj = parse_json_safe(r["raw_json"])
                cj = parse_json_safe(r["condition_json"])
                seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                u_out = "uM" if unit == "uM" else "ug/mL_needs_mw"
                if u_out == "uM" and not (1e-4 <= val <= 1e5):
                    continue
                rr = f"{lane}:{bmode}"
                pri = 1 if (lane == "strict_numeric" and unit == "uM") else 2
                # Emit BOTH MFC_antifungal (unified fungal-only with cond_species) AND MFC_C_albicans (for exact gate decision in Step 02!)
                out_f = make_row(
                    "MFC_antifungal", "tier2", "regression", r, seq_raw, is_pure20,
                    val, u_out, rel or "=", rr, pri,
                    cond_species=f_tag, cond_assay=med,
                )
                if out_f:
                    writer.writerow(out_f); n_written += 1
                if f_tag == "candida_albicans":
                    out_ca = make_row(
                        "MFC_C_albicans", "tier2", "regression", r, seq_raw, is_pure20,
                        val, u_out, rel or "=", rr, pri,
                        cond_assay=med,
                    )
                    if out_ca:
                        writer.writerow(out_ca); n_written += 1
                continue

            # ==================== GROUP B: ANTI-BIOFILM MBIC & MBEC ====================
            if ep in ("MBIC", "MBIC50", "MBIC90", "MBEC", "MBEC50", "MBEC90"):
                if val is None or val <= 0 or rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                if unit not in ("uM", "ug/mL"):
                    continue
                is_mbic = ep.startswith("MBIC")
                target_ep = "MBIC_biofilm" if is_mbic else "MBEC_biofilm"
                est_tag = ep  # MBIC / MBIC50 / MBIC90 or MBEC / MBEC50 / MBEC90
                sp_name = ""
                med = "unspecified"
                bmode = ""
                if src == "DBAASP":
                    link = dbaasp_microbe_linked.get(r["record_id"])
                    if not link:
                        continue
                    sp_name, med, bmode = link
                elif src == "CHEMBL":
                    cj_tmp = parse_json_safe(r["condition_json"])
                    _, assay_desc, sp_raw, _, _, _ = extract_context_text(cj_tmp)
                    sp_name = sp_raw or assay_desc
                    med = norm_mic_medium(assay_desc)
                    bmode = "chembl_assay_organism"
                else:
                    continue

                rj = parse_json_safe(r["raw_json"])
                cj = parse_json_safe(r["condition_json"])
                seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                if unit == "ug/mL" and not (smi_ok or is_pure20):
                    continue
                u_out = "uM" if unit == "uM" else "ug/mL_needs_mw"
                if u_out == "uM" and not (1e-4 <= val <= 1e5):
                    continue
                sp_tag = norm_microbe_species_tag(sp_name)
                rr = f"{ep}_{unit}:{bmode}"
                pri = 1 if (rel in ("=", "") and unit == "uM") else 2
                out = make_row(
                    target_ep, "tier2", "regression", r, seq_raw, is_pure20,
                    val, u_out, rel or "=", rr, pri,
                    cond_species=sp_tag, cond_estimand=est_tag, cond_assay=med,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== GROUP B: ANTIMICROBIAL LETHAL CONCENTRATION LC ====================
            if ep == "LC_lemma":
                if src != "DBAASP" or lane not in ("strict_numeric", "censored_numeric") or unit != "uM":
                    continue
                if val is None or not (1e-4 <= val <= 1e5) or rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                rj = parse_json_safe(r["raw_json"])
                if clean_str(rj.get("endpoint")) != "DBAASP LC":
                    drop_stats[("LC_antimicrobial", "non_dbaasp_lc")] += 1
                    continue
                link = dbaasp_microbe_linked.get(r["record_id"])
                if not link:
                    continue
                sp_name, med, bmode = link
                cj = parse_json_safe(r["condition_json"])
                seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
                sp_tag = norm_microbe_species_tag(sp_name)
                rr = ("strict_native" if lane == "strict_numeric" else "censored_inequality_preserved") + f":{bmode}"
                out = make_row(
                    "LC_antimicrobial", "tier2", "regression", r, seq_raw, is_pure20,
                    val, "uM", rel or "=", rr, 1 if lane == "strict_numeric" else 2,
                    cond_species=sp_tag, cond_assay=med,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ==================== GROUP A (TIER 1) ====================
            cj = parse_json_safe(r["condition_json"])
            rj = parse_json_safe(r["raw_json"])
            seq_raw, is_pure20 = check_pure_unmodified_20aa(r, rj, cj)
            sem, assay_desc, sp_raw, mat_raw, rt_raw, ed = extract_context_text(cj)
            full_text = f"{assay_desc} {ed} {mat_raw} {sp_raw}".lower()

            # ---------- A1. EFFLUX_RATIO ----------
            if ep == "efflux_ratio":
                if not smi_ok or unit != "ratio" or val is None or not (0.01 <= val <= 500.0):
                    continue
                if rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                # Distinguish uninhibited Caco-2 vs inhibitor-conditioned Caco-2 vs MDCK vs other!
                if "inhibitor" in full_text:
                    a_tag = "caco2_with_inhibitor"
                elif "caco" in full_text:
                    a_tag = "caco2"
                elif "mdck" in full_text:
                    a_tag = "mdck"
                else:
                    a_tag = "unspecified"
                rr = "censored_inequality_preserved" if lane == "censored_numeric" else f"verified_efflux_ratio:{src}"
                out = make_row(
                    "efflux_ratio", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "ratio", rel or "=", rr, 1 if rel in ("=", "") else 2,
                    cond_assay=a_tag,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ---------- A2. CYTOTOXICITY_CC50 (QUANTITATIVE MAMMALIAN) ----------
            if ep in ("CC50", "cytotoxicity_CC50", "cytotoxicity_IC50", "cytotoxicity_EC50", "cytotoxicity_LC50",
                      "50% Cytotoxicity", "50% Cell death", "cytotoxicity"):
                # Attach DBAASP linked cell-line name if available
                cell_linked = dbaasp_cell_linked.get(r["record_id"], "")
                ft_cyto = f"{full_text} {cell_linked}".lower()

                # Strictly exclude non-cellular zebrafish embryo assays
                if "zebrafish" in ft_cyto or "embryo" in ft_cyto:
                    drop_stats[("cytotoxicity_CC50", "excluded_zebrafish_embryo")] += 1
                    continue

                if ep == "cytotoxicity":
                    ed_l = ed.lower()
                    # Strictly require quantitative 50% potency estimand (exclude percent, pubchem_potency, tgi, binary!)
                    if not any(k in ed_l for k in ("cc50", "ic50", "ec50", "lc50", "ld50", "gi50", "tc50")):
                        continue
                    if "percent" in ed_l or unit in ("percent", "label01", ""):
                        continue

                if val is None or val <= 0 or rel not in ("=", ">", ">=", "<", "<=", "~", ""):
                    continue
                if rel == "" and src not in ("acs_jmedchem_5c00544", "cyclicpepedia_pubchem", "CHEMBL", "DBAASP"):
                    continue

                # Determine estimand tag
                ed_l = f"{ep} {ed}".lower()
                if "cc50" in ed_l or "50% cytotoxicity" in ed_l or "50% cell death" in ed_l or "tc50" in ed_l:
                    est_tag = "CC50"
                elif "ic50" in ed_l or "gi50" in ed_l:
                    est_tag = "IC50"
                elif "ec50" in ed_l:
                    est_tag = "EC50"
                elif "lc50" in ed_l or "ld50" in ed_l:
                    est_tag = "LC50"
                else:
                    est_tag = "CC50"

                # Normalize concentration unit cleanly
                u_low = unit.lower()
                val_out = None
                unit_out = ""
                if u_low == "um":
                    if 1e-4 <= val <= 1e5:
                        val_out, unit_out = val, "uM"
                elif u_low == "nm":
                    v_um = val / 1000.0
                    if 1e-4 <= v_um <= 1e5:
                        val_out, unit_out = v_um, "uM"
                elif u_low == "mm":
                    v_um = val * 1000.0
                    if 1e-4 <= v_um <= 1e5:
                        val_out, unit_out = v_um, "uM"
                elif u_low in ("ug/ml", "µg/ml", "μg/ml"):
                    if smi_ok or is_pure20:
                        val_out, unit_out = val, "ug/mL_needs_mw"
                elif u_low in ("mg/ml", "g/l"):
                    if smi_ok or is_pure20:
                        val_out, unit_out = val * 1000.0, "ug/mL_needs_mw"

                if val_out is None:
                    continue
                # For uM rows without SMILES, still require valid canonical sequence!
                if not smi_ok and not seq_raw:
                    continue

                cyto_sp = norm_species(sp_raw or mat_raw or ft_cyto)
                cell_grp = norm_cell_group(ft_cyto)
                norm_rel = "=" if rel in ("", "~") else rel
                if lane == "strict_numeric" and unit_out == "uM":
                    rr, pri = f"strict_native:{src}", 1
                elif lane == "censored_numeric" and unit_out == "uM":
                    rr, pri = f"censored_inequality:{src}", 2
                elif unit_out == "ug/mL_needs_mw":
                    rr, pri = f"cyto_ug_ml_rdkit_mw:{src}", 2
                else:
                    rr, pri = f"cyto_quant_review_rescue:{src}", 1 if norm_rel == "=" else 2

                out = make_row(
                    "cytotoxicity_CC50", "tier1", "regression", r, seq_raw, is_pure20,
                    val_out, unit_out, norm_rel, rr, pri,
                    cond_species=cyto_sp, cond_matrix=cell_grp, cond_estimand=est_tag,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ---------- A3. HEMOLYSIS_MHC (QUANTITATIVE MHC / MHC10 / HC10) ----------
            if ep in ("hemolysis_MHC", "MHC", "hemolysis_MHC10", "MHC10", "HC10", "hemolysis_HC10", "hemolysis"):
                ed_l = ed.lower()
                is_mhc_like = (
                    ep in ("hemolysis_MHC", "MHC", "hemolysis_MHC10", "MHC10", "HC10", "hemolysis_HC10")
                    or (ep == "hemolysis" and any(k in ed_l for k in ("mhc_", "hc10_")))
                )
                if not is_mhc_like:
                    continue
                if val is None or val <= 0 or rel not in ("=", ">", ">=", "<", "<=", "~", ""):
                    continue

                if "hc10" in f"{ep} {ed}".lower():
                    est_tag = "HC10"
                elif "mhc10" in f"{ep} {ed}".lower():
                    est_tag = "MHC10"
                else:
                    est_tag = "MHC"

                u_low = unit.lower()
                val_out = None
                unit_out = ""
                if u_low == "um":
                    if 1e-3 <= val <= 1e5:
                        val_out, unit_out = val, "uM"
                elif u_low == "nm":
                    v_um = val / 1000.0
                    if 1e-3 <= v_um <= 1e5:
                        val_out, unit_out = v_um, "uM"
                elif u_low == "mm":
                    v_um = val * 1000.0
                    if 1e-3 <= v_um <= 1e5:
                        val_out, unit_out = v_um, "uM"
                elif u_low in ("ug/ml", "µg/ml", "μg/ml"):
                    if smi_ok or is_pure20:
                        val_out, unit_out = val, "ug/mL_needs_mw"
                elif u_low in ("mg/ml", "g/l"):
                    if smi_ok or is_pure20:
                        val_out, unit_out = val * 1000.0, "ug/mL_needs_mw"

                if val_out is None:
                    continue
                if not smi_ok and not seq_raw:
                    continue

                rbc_sp = norm_species(sp_raw or mat_raw or full_text)
                norm_rel = "=" if rel in ("", "~") else rel
                if lane == "strict_numeric" and unit_out == "uM":
                    rr, pri = f"strict_native:{src}", 1
                elif lane == "censored_numeric" and unit_out == "uM":
                    rr, pri = f"censored_inequality:{src}", 2
                elif unit_out == "ug/mL_needs_mw":
                    rr, pri = f"mhc_ug_ml_rdkit_mw:{src}", 2
                else:
                    rr, pri = f"mhc_quant_review_rescue:{src}", 1 if norm_rel == "=" else 2

                out = make_row(
                    "hemolysis_MHC", "tier1", "regression", r, seq_raw, is_pure20,
                    val_out, unit_out, norm_rel, rr, pri,
                    cond_species=rbc_sp, cond_estimand=est_tag,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ---------- A4. VD_ABSOLUTE (VOLUME OF DISTRIBUTION IN LITERS) ----------
            if ep in ("Vd_absolute", "Vd"):
                u_low = unit.lower()
                if u_low not in ("l", "ml"):
                    continue
                if not smi_ok and not is_pure20 and not seq_raw:
                    continue
                scale = 1.0 if u_low == "l" else 0.001
                c_route = norm_route(rt_raw or full_text)
                c_sp = norm_species(sp_raw or full_text)
                c_est = norm_vd_estimand(ep, full_text, c_route)

                if rel == "between":
                    if lo_v is not None and hi_v is not None and lo_v > 0 and hi_v >= lo_v:
                        lo_l, hi_l = lo_v * scale, hi_v * scale
                        if 1e-3 <= lo_l <= hi_l <= 1e5:
                            mid_l = math.sqrt(lo_l * hi_l)
                            out = make_row(
                                "Vd_absolute", "tier1", "regression", r, seq_raw, is_pure20,
                                mid_l, "L", "interval", f"vd_abs_interval_rescue:{src}", 2,
                                val_lower=math.log10(lo_l), val_upper=math.log10(hi_l),
                                cond_species=c_sp, cond_route=c_route, cond_estimand=c_est,
                            )
                            if out:
                                writer.writerow(out); n_written += 1
                    continue

                if val is None or val <= 0 or rel not in ("=", "~", ">", ">=", "<", "<="):
                    continue
                val_l = val * scale
                if not (1e-3 <= val_l <= 1e5):
                    continue
                norm_rel = "=" if rel == "~" else rel
                rr = "strict_native" if lane == "strict_numeric" else f"verified_vd_abs_rescue:{src}"
                pri = 1 if lane == "strict_numeric" else 2
                out = make_row(
                    "Vd_absolute", "tier1", "regression", r, seq_raw, is_pure20,
                    val_l, "L", norm_rel, rr, pri,
                    cond_species=c_sp, cond_route=c_route, cond_estimand=c_est,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

            # ---------- A5. LOGP (EXPERIMENTAL OCTANOL/WATER ONLY!) ----------
            if ep == "LogP":
                if not smi_ok and not is_pure20:
                    continue
                if val is None or not (-6.0 <= val <= 10.0):
                    continue
                if rel not in ("=", ">", ">=", "<", "<=", ""):
                    continue
                # Strictly exclude calculated, pH-buffered LogD, and non-octanol solvents!
                if any(w in full_text for w in ("calculated", "clogp", "mlogp", "alogp", "predicted", "heptane", "decadiene", "chloroform", "cyclohexane", "hexane")):
                    drop_stats[("LogP", "excluded_calc_or_non_octanol")] += 1
                    continue
                if re.search(r"\bph\s*\d", full_text) or "7.4" in full_text:
                    drop_stats[("LogP", "excluded_ph_buffered_logd")] += 1
                    continue
                a_tag = "shake_flask" if "shake" in full_text else ("hplc" if "hplc" in full_text or "chrom" in full_text else "unspecified")
                out = make_row(
                    "LogP", "tier1", "regression", r, seq_raw, is_pure20,
                    val, "dimensionless_log10", rel or "=", f"verified_exp_logp:{src}", 1 if rel in ("=", "") else 2,
                    cond_assay=a_tag,
                )
                if out:
                    writer.writerow(out); n_written += 1
                continue

    con.close()

    audit = {
        "elapsed_sec": round(time.time() - t0, 2),
        "total_candidate_rows": n_written,
        "by_endpoint": dict(sorted(stats.items())),
        "by_rescue_reason": {f"{k[0]}|{k[1]}": v for k, v in sorted(rescue_stats.items())},
        "drops": {f"{k[0]}|{k[1]}": v for k, v in sorted(drop_stats.items())},
    }
    (OUT_DIR / "extraction_audit_v331.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[3/3] DONE Step 01 in {audit['elapsed_sec']}s! Total candidate rows written: {n_written:,}")
    for ep_name, cnt in sorted(stats.items()):
        print(f"  {ep_name:28s}: {cnt:6,d}")
    print("Drops:", dict(drop_stats))


if __name__ == "__main__":
    main()
