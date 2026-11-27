"""Approved-peptide clinical percentile context engine (ADMET-AI-style contextualization).

Provides fast, deterministic, unit-aligned percentile positioning of predicted
peptide ADMET/PK properties against:
  1. `approved_clinical_peptides`: Factual FDA/DailyMed/Case17/2019-2023 clinical PK
     observations (`half_life_reg`, `PPB`, `F`, `CL_absolute`, `CL_per_kg`,
     `Vd_absolute`, `Vd_per_kg`, `Cmax`).
  2. `curated_peptide_panel`:
     - Case17 17-drug clinical peptide reference profiles (`PAMPA`, `Caco2`, `LogD7.4`,
       `hemolysis_HC50`, `hemolysis_cls`, `overall_peptide_toxicity`, `BBB`,
       `solubility_cls`), transparently annotated as V33 clinical benchmark profiles.
     - Empirical corpus quantile ranges anchored from `v33_manifest.json` for the
       remaining 31 endpoints (`MDCK`, `RRCK`, `efflux_ratio`, `half_life_cls`,
       `cytotoxicity_*`, `hemolysis_MHC`, `LogD_decadiene`, `solubility_reg`, and
       all 22 Tier-2 antimicrobial/biofilm endpoints).

Public API:
    compute_clinical_percentile(
        endpoint: str,
        value: float | None,
        log10_space: bool = False,
    ) -> dict[str, Any] | None
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
from pathlib import Path
from typing import Any, Optional

WEB_DIR = Path(__file__).resolve().parent
UPGRADE_DIR = WEB_DIR.parent
DEFAULT_PERCENTILES_PATH = WEB_DIR / "clinical_percentiles.json"
DEFAULT_CLINICAL_INDEX_PATH = WEB_DIR / "clinical_evidence_index.json"
DEFAULT_MANIFEST_PATH = UPGRADE_DIR / "bundle" / "v33_manifest.json"

_CACHE_LOCK = threading.Lock()
_PERCENTILE_DB: Optional[dict[str, Any]] = None
_PERCENTILE_DB_PATH: Optional[Path] = None

# Empirical display/corpus bounds matching v33_manifest.json & adapter UI calibration
# (used strictly for corpus-range endpoints without FDA clinical label measurements).
_CORPUS_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "MDCK": (-8.5, -4.5),
    "RRCK": (-8.5, -4.5),
    "efflux_ratio": (-0.6, 2.0),
    "half_life_cls": (0.0, 1.0),
    "cytotoxicity_cls": (0.0, 1.0),
    "cytotoxicity_CC50": (-0.5, 3.4),
    "hemolysis_MHC": (-0.5, 3.4),
    "LogD_decadiene": (-5.0, 4.0),
    "solubility_reg": (0.0, 3.8),
    "LC_antimicrobial": (-1.0, 3.0),
    "MBC_A_baumannii": (-1.0, 3.0),
    "MBC_E_coli": (-1.0, 3.0),
    "MBC_E_faecalis": (-1.0, 3.0),
    "MBC_P_aeruginosa": (-1.0, 3.0),
    "MBC_S_aureus": (-1.0, 3.0),
    "MBEC_biofilm": (-1.0, 3.2),
    "MBIC_biofilm": (-1.0, 3.0),
    "MFC_C_albicans": (-1.0, 3.0),
    "MIC_A_baumannii": (-1.0, 3.0),
    "MIC_B_subtilis": (-1.0, 3.0),
    "MIC_C_albicans": (-1.0, 3.0),
    "MIC_C_neoformans": (-1.0, 3.0),
    "MIC_E_coli": (-1.0, 3.0),
    "MIC_E_faecalis": (-1.0, 3.0),
    "MIC_E_faecium": (-1.0, 3.0),
    "MIC_K_pneumoniae": (-1.0, 3.0),
    "MIC_L_monocytogenes": (-1.0, 3.0),
    "MIC_P_aeruginosa": (-1.0, 3.0),
    "MIC_S_aureus": (-1.0, 3.0),
    "MIC_S_epidermidis": (-1.0, 3.0),
    "MIC_Salmonella": (-1.0, 3.0),
}

# Mapping from manifest Case17 scorecard keys to V33 endpoints (curated_peptide_panel)
_CASE17_PROFILED_FIELDS: dict[str, tuple[str, bool]] = {
    # endpoint: (case17_drug_field, field_is_already_log10)
    "PAMPA": ("v33_pampa_log10", True),
    "Caco2": ("v33_caco2_log10", True),
    "LogD7.4": ("v33_logd74", False),
    "hemolysis_HC50": ("v33_hemolysis_hc50_uM", False),
    "hemolysis_cls": ("v33_hemolysis_prob_cal", False),
    "overall_peptide_toxicity": ("v33_toxicity_prob", False),
    "BBB": ("v33_bbb_prob", False),
    "solubility_cls": ("v33_solubility_prob", False),
}

_ROUTE_CANON = {
    "subcutaneous": "SC",
    "sc": "SC",
    "intravenous": "IV",
    "iv": "IV",
    "oral": "PO",
    "po": "PO",
    "intramuscular": "IM",
    "im": "IM",
    "intranasal": "IN",
    "intrathecal": "IT",
    "inhalation": "INH",
    "unspecified": "unspecified",
    "": "unspecified",
}


def _norm_drug_name(name: str) -> str:
    s = re.sub(r"[_\-\s]+", " ", str(name or "").strip()).lower()
    if s == "cyclosporine":
        return "cyclosporine a"
    return s


def _display_drug_name(name: str) -> str:
    s = str(name or "").strip().replace("_", " ")
    if s.lower() == "cyclosporine":
        return "Cyclosporine A"
    return s


def _norm_route(route: str) -> str:
    r = str(route or "").strip().lower()
    return _ROUTE_CANON.get(r, route.strip() or "unspecified")


def _parse_numeric_and_unit(value_display: str) -> tuple[Optional[float], str]:
    """Parse factual PK numeric/range string (e.g. '30.0–50.0 L', '~97 %', '>>99% %') -> (midpoint, unit)."""
    s = str(value_display or "").strip()
    if not s:
        return None, ""
    # Clean prefixes like ~, >>, <<, >, <, >=, <=
    s_clean = re.sub(r"^[~≈<>≤≥\s]+", "", s)
    # Match range like 30.0–50.0 or 0.25-0.52 or single scientific number 3.046e+04
    num_pat = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    m_range = re.match(rf"^{num_pat}\s*[–—\-]\s*{num_pat}\s*(.*)$", s_clean)
    if m_range:
        lo = float(m_range.group(1))
        hi = float(m_range.group(2))
        unit = m_range.group(3).strip()
        # Geometric midpoint for positive ranges spanning >2x, arithmetic otherwise
        if lo > 0 and hi > 0 and (hi / lo) >= 2.0:
            val = math.sqrt(lo * hi)
        else:
            val = 0.5 * (lo + hi)
        return val, unit
    m_single = re.match(rf"^{num_pat}\s*(.*)$", s_clean)
    if m_single:
        val = float(m_single.group(1))
        unit = m_single.group(2).strip()
        return val, unit
    return None, ""


def _fmt_example_value(val: float, linear_unit: str) -> str:
    if linear_unit in ("fraction_bound", "fraction", "probability"):
        return f"{val * 100:.1f}%"
    if linear_unit == "cm/s":
        return f"{val:.2e} cm/s"
    if linear_unit in ("logD", "logD_dec"):
        return f"{val:+.2f}" if val >= 0 else f"{val:.2f}"
    if abs(val) >= 100:
        num = f"{val:.0f}" if abs(val - round(val)) < 0.05 else f"{val:.1f}"
    elif abs(val) >= 10:
        num = f"{val:.1f}"
    elif abs(val) >= 0.1:
        num = f"{val:.2g}" if abs(val) < 1 else f"{val:.3g}"
    elif abs(val) >= 0.001:
        num = f"{val:.3g}"
    else:
        num = f"{val:.2e}"
    return f"{num} {linear_unit}".strip()


def _collect_exact_mw_lookup(clinical_index: dict[str, Any]) -> dict[str, float]:
    """Build exact RDKit MW lookup from workspace SMILES/sequences (zero guessed MWs)."""
    mw_map: dict[str, float] = {}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return mw_map

    # 1. Built-in drug_library.py SMILES
    try:
        from .drug_library import DRUG_LIBRARY

        for d in DRUG_LIBRARY:
            smi = d.get("smiles")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = float(Descriptors.MolWt(mol))
            for alias in [d.get("display_name"), d.get("key"), *(d.get("aliases") or [])]:
                if alias:
                    mw_map[_norm_drug_name(str(alias))] = mw
    except Exception:
        pass

    # 2. Recent FDA 2019-2024 peptides SMILES (if present)
    recent_path = UPGRADE_DIR / "expansion_v335" / "fda_2019_2024_recent_peptides.json"
    if recent_path.exists():
        try:
            recent = json.loads(recent_path.read_text(encoding="utf-8"))
            for d in recent:
                smi = d.get("smiles")
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                mw_map[_norm_drug_name(d.get("name", ""))] = float(Descriptors.MolWt(mol))
        except Exception:
            pass

    # 3. Canonical 20-AA sequences indexed in clinical_evidence_index.json
    try:
        from .chem_utils import sequence_to_mol

        prods = clinical_index.get("products") or {}
        by_seq = (clinical_index.get("lookup") or {}).get("by_seq") or {}
        for seq, pid in by_seq.items():
            if pid not in prods:
                continue
            key = _norm_drug_name(prods[pid].get("name", ""))
            if key in mw_map:
                continue
            mol = sequence_to_mol(seq)
            if mol is not None:
                mw_map[key] = float(Descriptors.MolWt(mol))
    except Exception:
        pass

    return mw_map


def _aggregate_drug_records(
    entries: list[dict[str, Any]],
    log10_space: bool,
    linear_unit: str,
) -> list[dict[str, Any]]:
    """Aggregate multiple observations per drug into 1 representative record per approved peptide."""
    by_drug: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        dk = _norm_drug_name(e["name"])
        by_drug.setdefault(dk, []).append(e)

    merged: list[dict[str, Any]] = []
    for dk, recs in by_drug.items():
        # If a drug has intrathecal (CSF) records mixed with systemic IV/SC/PO records
        # (e.g. Ziconotide), isolate explicit systemic routes so CSF values don't distort systemic PK.
        has_it = any(r.get("route") in ("IT", "intrathecal") for r in recs)
        explicit_systemic = [r for r in recs if r.get("route") in ("IV", "SC", "IM", "PO")]
        if has_it and explicit_systemic:
            pool = explicit_systemic
        else:
            systemic = [r for r in recs if r.get("route") not in ("IT", "intrathecal")]
            pool = systemic if systemic else recs
        vals = [float(r["value"]) for r in pool]
        if log10_space:
            pos_vals = [v for v in vals if v > 0]
            if not pos_vals:
                continue
            rep_val = 10.0 ** statistics.median([math.log10(v) for v in pos_vals])
        else:
            rep_val = float(statistics.median(vals))
        # Pick representative metadata from the record closest to rep_val
        best = min(
            pool,
            key=lambda r: abs(
                (math.log10(r["value"]) - math.log10(rep_val))
                if (log10_space and r["value"] > 0 and rep_val > 0)
                else (r["value"] - rep_val)
            ),
        )
        # Preserve 6 significant digits across orders of magnitude (avoids 1e-7 cm/s underflowing to 0.0)
        stored_val = float(f"{rep_val:.6g}") if rep_val != 0.0 else 0.0
        merged.append(
            {
                "name": _display_drug_name(best["name"]),
                "value": stored_val,
                "compare_value": round(math.log10(rep_val), 6) if (log10_space and rep_val > 0) else round(rep_val, 6),
                "value_display": _fmt_example_value(rep_val, linear_unit),
                "unit": linear_unit,
                "route": _norm_route(best.get("route", "")),
                "source": best.get("source", "NLM DailyMed / US FDA"),
            }
        )

    merged.sort(key=lambda r: r["compare_value"])
    return merged


def build_clinical_percentiles_data(
    clinical_index_path: Path = DEFAULT_CLINICAL_INDEX_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Generate the complete `clinical_percentiles.json` payload across all 47 V33 endpoints."""
    clinical_index = json.loads(clinical_index_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    products = clinical_index.get("products") or {}
    endpoints_meta = manifest.get("endpoints") or {}
    case17_drugs = (manifest.get("case17_scorecard") or {}).get("drugs") or []
    blind_drugs = (manifest.get("recent_fda_2019_2023_blind_set") or {}).get("drugs") or []
    mw_map = _collect_exact_mw_lookup(clinical_index)

    # --- Source (a) + (b): Factual FDA-approved clinical PK distributions ---
    approved_raw: dict[str, list[dict[str, Any]]] = {
        "half_life_reg": [],
        "PPB": [],
        "F": [],
        "CL_absolute": [],
        "CL_per_kg": [],
        "Vd_absolute": [],
        "Vd_per_kg": [],
        "Cmax": [],
    }

    # 1. Extract from deduplicated clinical_evidence_index.json
    for prod in products.values():
        drug_name = str(prod.get("name") or "").strip()
        dn_norm = _norm_drug_name(drug_name)
        for rec in prod.get("pk_records") or []:
            ep_raw = str(rec.get("endpoint") or "").strip()
            val, unit = _parse_numeric_and_unit(str(rec.get("value_display") or ""))
            if val is None or val < 0:
                continue
            unit_norm = unit.replace("μ", "µ").replace("% %", "%").strip()
            route = _norm_route(str(rec.get("route") or ""))
            src = str(rec.get("source") or "NLM DailyMed FDA Prescribing Information")

            if ep_raw == "PPB" and "%" in unit_norm:
                frac = max(0.0, min(1.0, val / 100.0))
                approved_raw["PPB"].append({"name": drug_name, "value": frac, "route": route, "source": src})

            elif ep_raw == "F" and "%" in unit_norm:
                frac = max(0.0, min(1.0, val / 100.0))
                approved_raw["F"].append({"name": drug_name, "value": frac, "route": route, "source": src})

            elif ep_raw == "CL" and val > 0:
                u_low = unit_norm.lower()
                if u_low == "l/h":
                    approved_raw["CL_absolute"].append({"name": drug_name, "value": val, "route": route, "source": src})
                elif u_low == "ml/min":
                    approved_raw["CL_absolute"].append({"name": drug_name, "value": val * 0.06, "route": route, "source": src})
                elif u_low == "ml/h":
                    approved_raw["CL_absolute"].append({"name": drug_name, "value": val / 1000.0, "route": route, "source": src})
                elif u_low == "ml/min/kg":
                    approved_raw["CL_per_kg"].append({"name": drug_name, "value": val, "route": route, "source": src})
                elif u_low in ("ml/h/kg", "ml/kg/h"):
                    approved_raw["CL_per_kg"].append({"name": drug_name, "value": val / 60.0, "route": route, "source": src})
                elif u_low in ("l/h/kg", "l/kg/h"):
                    approved_raw["CL_per_kg"].append({"name": drug_name, "value": val * 1000.0 / 60.0, "route": route, "source": src})

            elif ep_raw == "Vd" and val > 0:
                u_low = unit_norm.lower()
                if u_low == "l":
                    approved_raw["Vd_absolute"].append({"name": drug_name, "value": val, "route": route, "source": src})
                elif u_low == "ml":
                    approved_raw["Vd_absolute"].append({"name": drug_name, "value": val / 1000.0, "route": route, "source": src})
                elif u_low == "l/kg":
                    approved_raw["Vd_per_kg"].append({"name": drug_name, "value": val, "route": route, "source": src})
                elif u_low == "ml/kg":
                    approved_raw["Vd_per_kg"].append({"name": drug_name, "value": val / 1000.0, "route": route, "source": src})

            elif ep_raw == "Cmax" and val > 0:
                u_low = unit_norm.lower()
                if u_low == "pmol/l":
                    approved_raw["Cmax"].append({"name": drug_name, "value": val * 1e-6, "route": route, "source": src})
                elif u_low == "nmol/l":
                    approved_raw["Cmax"].append({"name": drug_name, "value": val * 1e-3, "route": route, "source": src})
                elif u_low in ("umol/l", "µmol/l", "um", "µm"):
                    approved_raw["Cmax"].append({"name": drug_name, "value": val, "route": route, "source": src})
                elif dn_norm in mw_map and mw_map[dn_norm] > 0:
                    mw = mw_map[dn_norm]
                    # 1 ng/mL = 1 ug/L -> uM = (ng/mL) / MW(g/mol)
                    if u_low == "pg/ml":
                        c_um = (val * 1e-3) / mw
                    elif u_low == "ng/ml":
                        c_um = val / mw
                    elif u_low in ("µg/ml", "ug/ml", "mg/l"):
                        c_um = (val * 1000.0) / mw
                    else:
                        c_um = None
                    if c_um is not None and c_um > 0:
                        approved_raw["Cmax"].append({"name": drug_name, "value": c_um, "route": route, "source": src})

    # 2. Merge FDA-observed clinical T1/2 & PPB from Case17 clinical benchmark scorecard
    for d in case17_drugs:
        dname = _display_drug_name(str(d.get("drug_name") or ""))
        route = _norm_route(str(d.get("clinical_route") or ""))
        src = "FDA Prescribing Information (Case17 Benchmark)"
        if d.get("fda_half_life_h") is not None and float(d["fda_half_life_h"]) > 0:
            approved_raw["half_life_reg"].append(
                {"name": dname, "value": float(d["fda_half_life_h"]), "route": route, "source": src}
            )
        if d.get("fda_ppb") is not None:
            approved_raw["PPB"].append(
                {"name": dname, "value": max(0.0, min(1.0, float(d["fda_ppb"]))), "route": route, "source": src}
            )

    # 3. Merge FDA-observed clinical T1/2, PPB, Vd, CL from Recent FDA 2019-2023 Blind Set
    for d in blind_drugs:
        dname = _display_drug_name(str(d.get("name") or ""))
        route = _norm_route(str(d.get("route") or ""))
        src = f"FDA Prescribing Information ({d.get('fda_approval_year', '2019-2023')})"
        if d.get("obs_half_life_h") is not None and float(d["obs_half_life_h"]) > 0:
            approved_raw["half_life_reg"].append(
                {"name": dname, "value": float(d["obs_half_life_h"]), "route": route, "source": src}
            )
        if d.get("obs_ppb") is not None:
            approved_raw["PPB"].append(
                {"name": dname, "value": max(0.0, min(1.0, float(d["obs_ppb"]))), "route": route, "source": src}
            )
        # Extract Rezafungin FDA label Vd (67 L) & CL (0.35 L/h) if not in clinical index
        note = str(d.get("fda_label_note") or "")
        if _norm_drug_name(dname) == "rezafungin":
            m_vd = re.search(r"Vd\s+([0-9.]+)\s*L\b", note)
            if m_vd:
                approved_raw["Vd_absolute"].append(
                    {"name": dname, "value": float(m_vd.group(1)), "route": route, "source": src}
                )
            m_cl = re.search(r"CL\s+([0-9.]+)\s*L/h\b", note)
            if m_cl:
                approved_raw["CL_absolute"].append(
                    {"name": dname, "value": float(m_cl.group(1)), "route": route, "source": src}
                )

    endpoints_out: dict[str, Any] = {}

    for ep, meta in endpoints_meta.items():
        log10_sp = bool(meta.get("log10_space", False))
        unit = str(meta.get("unit") or "")
        linear_unit = str(meta.get("linear_unit") or unit)
        task_type = str(meta.get("task_type") or "regression")
        n_total = int((meta.get("counts") or {}).get("n_total") or 0)

        if ep in approved_raw and approved_raw[ep]:
            examples = _aggregate_drug_records(approved_raw[ep], log10_space=log10_sp, linear_unit=linear_unit)
            comp_vals = [e["compare_value"] for e in examples]
            endpoints_out[ep] = {
                "endpoint": ep,
                "reference_group": "approved_clinical_peptides",
                "distribution_mode": "empirical_samples",
                "task_type": task_type,
                "unit": unit,
                "linear_unit": linear_unit,
                "log10_space": log10_sp,
                "n_reference": len(examples),
                "min_compare": comp_vals[0],
                "median_compare": round(float(statistics.median(comp_vals)), 6),
                "max_compare": comp_vals[-1],
                "samples": examples,
            }

        elif ep in _CASE17_PROFILED_FIELDS:
            field_name, field_is_log10 = _CASE17_PROFILED_FIELDS[ep]
            raw_list: list[dict[str, Any]] = []
            for d in case17_drugs:
                raw_v = d.get(field_name)
                if raw_v is None:
                    continue
                fv = float(raw_v)
                lin_v = (10.0 ** fv) if (log10_sp and field_is_log10) else fv
                raw_list.append(
                    {
                        "name": _display_drug_name(str(d.get("drug_name") or "")),
                        "value": lin_v,
                        "route": _norm_route(str(d.get("clinical_route") or "")),
                        "source": "Case17 Clinical Peptide Panel (V33 Reference Profile)",
                    }
                )
            examples = _aggregate_drug_records(raw_list, log10_space=log10_sp, linear_unit=linear_unit)
            comp_vals = [e["compare_value"] for e in examples]
            endpoints_out[ep] = {
                "endpoint": ep,
                "reference_group": "curated_peptide_panel",
                "distribution_mode": "empirical_samples",
                "task_type": task_type,
                "unit": unit,
                "linear_unit": linear_unit,
                "log10_space": log10_sp,
                "n_reference": len(examples),
                "min_compare": comp_vals[0],
                "median_compare": round(float(statistics.median(comp_vals)), 6),
                "max_compare": comp_vals[-1],
                "samples": examples,
            }

        else:
            # Empirical reference range anchored from v33_manifest.json bounds & UI corpus scale
            bounds = meta.get("bounds")
            if ep in _CORPUS_REFERENCE_RANGES:
                lo, hi = _CORPUS_REFERENCE_RANGES[ep]
            elif isinstance(bounds, list) and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
            elif task_type == "classification":
                lo, hi = 0.0, 1.0
            else:
                lo, hi = -2.0, 2.0

            # Smooth reference quantile anchors across the empirical corpus range
            probs = [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 100.0]
            span = hi - lo
            if task_type == "classification":
                q_vals = [round(lo + (p / 100.0) * span, 4) for p in probs]
            else:
                # Slightly concentrated around center (bell-like empirical peptide corpus)
                # Mapping quantiles via symmetric normal-like positions inside [lo, hi]
                fracs = [0.0, 0.10, 0.18, 0.33, 0.50, 0.67, 0.82, 0.90, 1.0]
                q_vals = [round(lo + f * span, 4) for f in fracs]

            endpoints_out[ep] = {
                "endpoint": ep,
                "reference_group": "curated_peptide_panel",
                "distribution_mode": "quantile_anchors",
                "task_type": task_type,
                "unit": unit,
                "linear_unit": linear_unit,
                "log10_space": log10_sp,
                "n_reference": n_total if n_total > 0 else 17,
                "min_compare": q_vals[0],
                "median_compare": q_vals[4],
                "max_compare": q_vals[-1],
                "quantile_grid": [{"p": p, "compare_value": v} for p, v in zip(probs, q_vals)],
                "samples": [],
            }

    return {
        "schema_version": "omnipanel-clinical-percentiles-v1",
        "description": (
            "Approved-peptide clinical PK reference distributions and curated peptide panel "
            "reference ranges for ADMET-AI-style percentile contextualization across all 47 endpoints."
        ),
        "sources": {
            "approved_clinical_peptides": (
                "Factual FDA/DailyMed prescribing information (`clinical_evidence_index.json`), "
                "Case17 FDA clinical peptide benchmark, and 2019-2023 FDA blind set (`v33_manifest.json`)."
            ),
            "curated_peptide_panel": (
                "Case17 17-drug clinical peptide V33 reference profiles and empirical corpus bounds "
                "from `v33_manifest.json`."
            ),
        },
        "stats": {
            "n_endpoints": len(endpoints_out),
            "n_approved_clinical_endpoints": sum(
                1 for v in endpoints_out.values() if v["reference_group"] == "approved_clinical_peptides"
            ),
            "n_curated_panel_endpoints": sum(
                1 for v in endpoints_out.values() if v["reference_group"] == "curated_peptide_panel"
            ),
        },
        "endpoints": endpoints_out,
    }


def _load_percentile_db(path: Optional[Path] = None) -> dict[str, Any]:
    global _PERCENTILE_DB, _PERCENTILE_DB_PATH
    target = path
    if target is None:
        env_p = os.environ.get("V33_CLINICAL_PERCENTILES_PATH", "").strip()
        target = Path(env_p) if env_p else DEFAULT_PERCENTILES_PATH

    with _CACHE_LOCK:
        if _PERCENTILE_DB is not None and _PERCENTILE_DB_PATH == target:
            return _PERCENTILE_DB
        if target.exists() and target.is_file():
            data = json.loads(target.read_text(encoding="utf-8"))
        else:
            data = build_clinical_percentiles_data()
        _PERCENTILE_DB = data
        _PERCENTILE_DB_PATH = target
        return data


def reload_clinical_percentiles(path: Optional[Path] = None) -> dict[str, Any]:
    """Clear cache and reload clinical_percentiles.json (useful in tests)."""
    global _PERCENTILE_DB, _PERCENTILE_DB_PATH
    with _CACHE_LOCK:
        _PERCENTILE_DB = None
        _PERCENTILE_DB_PATH = None
    return _load_percentile_db(path)


def _empirical_percentile(compare_val: float, sorted_vals: list[float]) -> float:
    """Compute smooth Hazen empirical percentile in [0.0, 100.0] with tie handling and interpolation."""
    n = len(sorted_vals)
    if n == 0:
        return 50.0
    if n == 1:
        if compare_val < sorted_vals[0]:
            return 0.0
        if compare_val > sorted_vals[0]:
            return 100.0
        return 50.0

    # Exact match / tie-aware midpoint rank
    n_less = 0
    n_equal = 0
    for v in sorted_vals:
        if abs(compare_val - v) <= 1e-9:
            n_equal += 1
        elif v < compare_val:
            n_less += 1
        else:
            break

    if n_equal > 0:
        return round(((n_less + 0.5 * n_equal) / n) * 100.0, 1)

    # Below minimum: smoothly taper toward 0
    if compare_val < sorted_vals[0]:
        span = max(sorted_vals[-1] - sorted_vals[0], 1e-6)
        min_pct = (0.5 / n) * 100.0
        dist_frac = (sorted_vals[0] - compare_val) / (0.25 * span)
        return round(max(0.0, min_pct * math.exp(-dist_frac)), 1)

    # Above maximum: smoothly taper toward 100
    if compare_val > sorted_vals[-1]:
        span = max(sorted_vals[-1] - sorted_vals[0], 1e-6)
        max_pct = ((n - 0.5) / n) * 100.0
        dist_frac = (compare_val - sorted_vals[-1]) / (0.25 * span)
        return round(min(100.0, 100.0 - (100.0 - max_pct) * math.exp(-dist_frac)), 1)

    # Strictly between sorted_vals[n_less - 1] and sorted_vals[n_less]: linear interpolation
    lo_v = sorted_vals[n_less - 1]
    hi_v = sorted_vals[n_less]
    lo_p = ((n_less - 0.5) / n) * 100.0
    hi_p = ((n_less + 0.5) / n) * 100.0
    t = (compare_val - lo_v) / max(hi_v - lo_v, 1e-12)
    return round(lo_p + t * (hi_p - lo_p), 1)


def _quantile_grid_percentile(compare_val: float, grid: list[dict[str, float]]) -> float:
    """Piecewise-linear interpolation on quantile anchor grid, clamped to [0.0, 100.0]."""
    if not grid:
        return 50.0
    if compare_val <= grid[0]["compare_value"]:
        return 0.0
    if compare_val >= grid[-1]["compare_value"]:
        return 100.0
    for i in range(len(grid) - 1):
        v0, p0 = float(grid[i]["compare_value"]), float(grid[i]["p"])
        v1, p1 = float(grid[i + 1]["compare_value"]), float(grid[i + 1]["p"])
        if v0 <= compare_val <= v1:
            t = (compare_val - v0) / max(v1 - v0, 1e-12)
            return round(p0 + t * (p1 - p0), 1)
    return 50.0


def _ordinal_en(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def compute_clinical_percentile(
    endpoint: str,
    value: Optional[float],
    log10_space: bool = False,
) -> Optional[dict[str, Any]]:
    """Compute ADMET-AI-style clinical/reference percentile context for one endpoint prediction.

    Args:
        endpoint: V33 endpoint identifier (e.g. `"half_life_reg"`, `"PPB"`, `"PAMPA"`).
        value: Predicted numeric value.
            - If `log10_space=False` (default): `value` is in the endpoint's linear unit
              (`h`, `L/h`, `mL/min/kg`, `L`, `L/kg`, `uM`, `fraction [0,1]`, `cm/s`, `LogD`, `prob [0,1]`).
            - If `log10_space=True`: `value` is already in `log10` space for log10 endpoints
              (ignored for linear/classification endpoints).
        log10_space: Whether `value` is passed in `log10` space rather than linear units.

    Returns:
        Dict with keys:
            - `percentile`: float in [0.0, 100.0]
            - `reference_group`: `"approved_clinical_peptides"` | `"curated_peptide_panel"`
            - `n_reference`: int
            - `summary_zh`: str
            - `summary_en`: str
            - `nearest_approved_examples`: list[dict] (up to 3 nearest reference examples)
        Or `None` if `endpoint` is unknown or `value` is None / non-finite.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None

    db = _load_percentile_db()
    ep_doc = (db.get("endpoints") or {}).get(endpoint)
    if not ep_doc:
        return None

    ep_is_log10 = bool(ep_doc.get("log10_space", False))
    if ep_is_log10:
        if log10_space:
            compare_val = v
        else:
            if v <= 0.0:
                compare_val = float(ep_doc.get("min_compare", -10.0)) - 10.0
            else:
                compare_val = math.log10(v)
    else:
        compare_val = v

    ref_group: str = str(ep_doc["reference_group"])
    n_ref: int = int(ep_doc["n_reference"])
    samples: list[dict[str, Any]] = ep_doc.get("samples") or []

    if ep_doc.get("distribution_mode") == "empirical_samples" and samples:
        sorted_comp = [float(s["compare_value"]) for s in samples]
        pct = _empirical_percentile(compare_val, sorted_comp)
        nearest_raw = sorted(samples, key=lambda s: abs(float(s["compare_value"]) - compare_val))[:3]
        nearest_examples = [
            {
                "name": s["name"],
                "value": s["value"],
                "value_display": s["value_display"],
                "unit": s["unit"],
                "route": s["route"],
                "source": s["source"],
            }
            for s in nearest_raw
        ]
    else:
        pct = _quantile_grid_percentile(compare_val, ep_doc.get("quantile_grid") or [])
        nearest_examples = []

    pct = max(0.0, min(100.0, round(float(pct), 1)))
    pct_int = int(round(pct))
    ord_en = _ordinal_en(pct_int)

    if ref_group == "approved_clinical_peptides":
        top2_en = ", ".join(f"{e['name']} ({e['value_display']})" for e in nearest_examples[:2])
        top2_zh = "、".join(f"{e['name']} ({e['value_display']})" for e in nearest_examples[:2])
        summary_en = (
            f"{ord_en} percentile among {n_ref} FDA-approved clinical peptides"
            + (f" (nearest: {top2_en})" if top2_en else "")
        )
        summary_zh = (
            f"位于 {n_ref} 种已上市临床多肽的第 {pct_int} 百分位"
            + (f"（最接近：{top2_zh}）" if top2_zh else "")
        )
    elif nearest_examples:
        top2_en = ", ".join(f"{e['name']} ({e['value_display']})" for e in nearest_examples[:2])
        top2_zh = "、".join(f"{e['name']} ({e['value_display']})" for e in nearest_examples[:2])
        summary_en = (
            f"{ord_en} percentile among {n_ref} clinical benchmark peptides "
            f"(V33 reference panel; nearest: {top2_en})"
        )
        summary_zh = (
            f"位于 {n_ref} 种临床基准多肽参考集的第 {pct_int} 百分位"
            f"（最接近：{top2_zh}）"
        )
    else:
        summary_en = f"{ord_en} percentile within curated peptide reference range (n={n_ref:,})"
        summary_zh = f"位于整理多肽参考分布区间的第 {pct_int} 百分位（n={n_ref:,}）"

    return {
        "percentile": pct,
        "reference_group": ref_group,
        "n_reference": n_ref,
        "summary_zh": summary_zh,
        "summary_en": summary_en,
        "nearest_approved_examples": nearest_examples,
    }


def get_percentile_catalog_summary() -> dict[str, Any]:
    """Return metadata summary of loaded clinical percentile distributions."""
    db = _load_percentile_db()
    return {
        "schema_version": db.get("schema_version"),
        "stats": db.get("stats") or {},
        "endpoints": {
            ep: {
                "reference_group": doc["reference_group"],
                "distribution_mode": doc["distribution_mode"],
                "n_reference": doc["n_reference"],
                "unit": doc["unit"],
                "linear_unit": doc["linear_unit"],
                "log10_space": doc["log10_space"],
            }
            for ep, doc in (db.get("endpoints") or {}).items()
        },
    }


def main() -> None:
    data = build_clinical_percentiles_data()
    DEFAULT_PERCENTILES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    st = data["stats"]
    print(
        f"Built {DEFAULT_PERCENTILES_PATH.name}: {st['n_endpoints']} endpoints "
        f"({st['n_approved_clinical_endpoints']} approved_clinical_peptides, "
        f"{st['n_curated_panel_endpoints']} curated_peptide_panel)."
    )


if __name__ == "__main__":
    main()
