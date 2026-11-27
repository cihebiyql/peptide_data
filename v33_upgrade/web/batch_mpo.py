"""Batch Peptide Screening & Multi-Parameter Optimization (Peptide-MPO) Engine (P1-6).

Evaluates up to 100 peptides through OmniPanelAdapter, computes smooth medicinal-chemistry
desirability curves [0, 1] across 4 core ADMET pillars (Permeability, Half-Life, Low Hemolysis,
Low Toxicity), applies evidence-grade confidence weighting and applicability-domain OOD penalties,
ranks compounds on a composite Peptide-MPO [0, 100] score, and generates formula-injection-safe CSV.
"""

from __future__ import annotations

import csv
import io
import math
import time
from typing import Any, Optional

from .adapter import OmniPanelAdapter
from .batch_parser import parse_batch_text, validate_parsed_entry
from .chem_utils import InputValidationError
from .drug_library import lookup_drug
from .schemas import (
    AssayConditions,
    BatchMoleculeResult,
    BatchParsedRow,
    BatchScreenItemInput,
    BatchScreenRequest,
    BatchScreenResponse,
    BatchScreenResult,
    EndpointPrediction,
    MechanismAnchor,
    MpoWeights,
)

MAX_SCREEN_ITEMS = 100

EVIDENCE_GRADE_FACTORS: dict[str, float] = {
    "quantitative": 1.0,
    "rank_only": 0.75,
    "exploratory_not_recommended": 0.25,
}

AD_PENALTY_BY_STATUS: dict[str, float] = {
    "in_domain": 1.0,
    "mechanism": 1.0,
    "borderline": 0.92,
    "ood": 0.72,
}


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(val)))


def _piecewise_linear(x: float, knots: list[tuple[float, float]]) -> float:
    """Evaluate a monotonic/piecewise-linear desirability curve in [0, 1]."""
    if x <= knots[0][0]:
        return _clamp(knots[0][1])
    for i in range(len(knots) - 1):
        x0, y0 = knots[i]
        x1, y1 = knots[i + 1]
        if x <= x1:
            t = (x - x0) / max(1e-12, x1 - x0)
            return _clamp(y0 + t * (y1 - y0))
    return _clamp(knots[-1][1])


def _desirability_permeability(log10_papp: float) -> float:
    """Desirability for passive/cellular permeability in log10(cm/s): -8 -> 0.05, -6 -> 0.65, -5 -> 0.95."""
    return _piecewise_linear(
        log10_papp,
        [(-8.5, 0.02), (-8.0, 0.06), (-7.0, 0.28), (-6.0, 0.68), (-5.0, 0.95), (-4.5, 1.0)],
    )


def _desirability_half_life(hours: float) -> float:
    """Desirability for plasma half-life in hours (log10-scaled): 0.1h -> 0.05, 1h -> 0.35, 24h -> 0.85, >=120h -> 1.0."""
    log_h = math.log10(max(hours, 1e-3))
    return _piecewise_linear(
        log_h,
        [(-1.3, 0.02), (-1.0, 0.08), (0.0, 0.36), (0.78, 0.66), (1.38, 0.86), (2.08, 1.0)],
    )


def _desirability_low_risk_prob(prob: float) -> float:
    """Desirability for safety risk probability (lower prob = safer): 0.0 -> 1.0, 0.25 -> 0.80, 0.50 -> 0.38, 0.85 -> 0.04."""
    return _piecewise_linear(
        _clamp(prob, 0.0, 1.0),
        [(0.0, 1.0), (0.10, 0.94), (0.25, 0.80), (0.50, 0.38), (0.75, 0.12), (0.95, 0.02)],
    )


def _desirability_safety_ic50(uM: float) -> float:
    """Desirability for HC50 / CC50 safety concentration (µM; higher = safer): 3 µM -> 0.08, 30 µM -> 0.48, 100 µM -> 0.80, 500 µM -> 1.0."""
    log_c = math.log10(max(uM, 0.1))
    return _piecewise_linear(
        log_c,
        [(0.0, 0.02), (0.5, 0.08), (1.0, 0.24), (1.5, 0.50), (2.0, 0.80), (2.7, 1.0)],
    )


def _grade_factor(pred: Optional[EndpointPrediction]) -> float:
    if pred is None or pred.evidence_grade is None:
        return 0.75
    return EVIDENCE_GRADE_FACTORS.get(pred.evidence_grade, 0.75)


def _ad_factor(pred: Optional[EndpointPrediction]) -> float:
    if pred is None:
        return 0.92
    return AD_PENALTY_BY_STATUS.get(pred.ad_status, 0.92)


def _worst_ad(statuses: list[str]) -> str:
    if not statuses:
        return "borderline"
    if "ood" in statuses:
        return "ood"
    if "borderline" in statuses:
        return "borderline"
    return "in_domain"


def _normalize_weights(weights: MpoWeights) -> dict[str, float]:
    raw = {
        "permeability": max(0.0, float(weights.permeability)),
        "half_life": max(0.0, float(weights.half_life)),
        "low_hemolysis": max(0.0, float(weights.low_hemolysis)),
        "low_toxicity": max(0.0, float(weights.low_toxicity)),
    }
    total = sum(raw.values())
    if total <= 1e-9:
        return {"permeability": 0.25, "half_life": 0.25, "low_hemolysis": 0.25, "low_toxicity": 0.25}
    return {k: v / total for k, v in raw.items()}


def compute_peptide_mpo(
    predictions: list[EndpointPrediction],
    weights: Optional[MpoWeights] = None,
    mechanism_anchor: Optional[MechanismAnchor] = None,
) -> dict[str, Any]:
    """Compute Peptide-MPO [0, 100] score, per-pillar desirability [0, 1], evidence factors, and OOD penalty."""
    w_cfg = weights or MpoWeights()
    norm_w = _normalize_weights(w_cfg)
    by_ep = {p.endpoint: p for p in predictions}

    pillar_scores: dict[str, float] = {}
    pillar_details: dict[str, dict[str, Any]] = {}
    pillar_grade_factors: dict[str, float] = {}
    pillar_ad_factors: dict[str, float] = {}

    # 1. Permeability pillar (PAMPA 60% + Caco2 40%, whichever available)
    perm_parts: list[tuple[float, float, EndpointPrediction]] = []
    for ep_name, sub_w in (("PAMPA", 0.6), ("Caco2", 0.4)):
        p = by_ep.get(ep_name)
        if p is not None and p.value is not None and p.value > 0:
            log_v = float(p.raw_value) if p.raw_value is not None else math.log10(max(p.value, 1e-12))
            perm_parts.append((sub_w, _desirability_permeability(log_v), p))
    if perm_parts:
        sw_sum = sum(x[0] for x in perm_parts)
        d_perm = sum(x[0] * x[1] for x in perm_parts) / sw_sum
        gf_perm = sum(x[0] * _grade_factor(x[2]) for x in perm_parts) / sw_sum
        af_perm = sum(x[0] * _ad_factor(x[2]) for x in perm_parts) / sw_sum
        primary_p = perm_parts[0][2]
        ad_perm = _worst_ad([x[2].ad_status for x in perm_parts])
        raw_disp = " · ".join(f"{x[2].endpoint} {x[2].value_display}" for x in perm_parts)
    else:
        d_perm, gf_perm, af_perm, primary_p, ad_perm, raw_disp = 0.35, 0.75, 0.92, None, "borderline", "N/A"
    pillar_scores["permeability"] = round(d_perm, 4)
    pillar_grade_factors["permeability"] = gf_perm
    pillar_ad_factors["permeability"] = af_perm
    pillar_details["permeability"] = {
        "desirability": round(d_perm, 4),
        "raw_display": raw_disp,
        "endpoint": primary_p.endpoint if primary_p else "PAMPA",
        "evidence_grade": primary_p.evidence_grade if primary_p else None,
        "evidence_factor": round(gf_perm, 3),
        "ad_status": ad_perm,
    }

    # 2. Half-Life pillar (prefer fatty-acylation mechanism anchor if detected, else half_life_reg)
    hl_p = by_ep.get("half_life_reg")
    if mechanism_anchor is not None and mechanism_anchor.detected and mechanism_anchor.value_h is not None:
        hl_hours = float(mechanism_anchor.value_h)
        d_hl = _desirability_half_life(hl_hours)
        gf_hl = 1.0
        af_hl = 1.0
        ad_hl = "in_domain"
        hl_disp = (
            f"{mechanism_anchor.range_h[0]:.0f}–{mechanism_anchor.range_h[1]:.0f} h (anchor)"
            if mechanism_anchor.range_h
            else f"{hl_hours:.1f} h (anchor)"
        )
        hl_grade = "quantitative"
    elif hl_p is not None and hl_p.value is not None and hl_p.value > 0:
        d_hl = _desirability_half_life(float(hl_p.value))
        gf_hl = _grade_factor(hl_p)
        af_hl = _ad_factor(hl_p)
        ad_hl = hl_p.ad_status
        hl_disp = f"{hl_p.value_display} h"
        hl_grade = hl_p.evidence_grade
    else:
        d_hl, gf_hl, af_hl, ad_hl, hl_disp, hl_grade = 0.35, 0.75, 0.92, "borderline", "N/A", None
    pillar_scores["half_life"] = round(d_hl, 4)
    pillar_grade_factors["half_life"] = gf_hl
    pillar_ad_factors["half_life"] = af_hl
    pillar_details["half_life"] = {
        "desirability": round(d_hl, 4),
        "raw_display": hl_disp,
        "endpoint": "half_life_reg",
        "evidence_grade": hl_grade,
        "evidence_factor": round(gf_hl, 3),
        "ad_status": ad_hl,
    }

    # 3. Low Hemolysis pillar (hemolysis_cls 55% + hemolysis_HC50 45%)
    hem_parts: list[tuple[float, float, EndpointPrediction]] = []
    h_cls = by_ep.get("hemolysis_cls")
    if h_cls is not None and h_cls.value is not None:
        hem_parts.append((0.55, _desirability_low_risk_prob(float(h_cls.value)), h_cls))
    h_hc50 = by_ep.get("hemolysis_HC50")
    if h_hc50 is not None and h_hc50.value is not None and h_hc50.value > 0:
        hem_parts.append((0.45, _desirability_safety_ic50(float(h_hc50.value)), h_hc50))
    if hem_parts:
        sw_sum = sum(x[0] for x in hem_parts)
        d_hem = sum(x[0] * x[1] for x in hem_parts) / sw_sum
        gf_hem = sum(x[0] * _grade_factor(x[2]) for x in hem_parts) / sw_sum
        af_hem = sum(x[0] * _ad_factor(x[2]) for x in hem_parts) / sw_sum
        primary_h = hem_parts[0][2]
        ad_hem = _worst_ad([x[2].ad_status for x in hem_parts])
        hem_disp = " · ".join(
            f"Risk {x[2].value_display}" if x[2].endpoint == "hemolysis_cls" else f"HC50 {x[2].value_display} µM"
            for x in hem_parts
        )
    else:
        d_hem, gf_hem, af_hem, primary_h, ad_hem, hem_disp = 0.65, 0.75, 0.92, None, "borderline", "N/A"
    pillar_scores["low_hemolysis"] = round(d_hem, 4)
    pillar_grade_factors["low_hemolysis"] = gf_hem
    pillar_ad_factors["low_hemolysis"] = af_hem
    pillar_details["low_hemolysis"] = {
        "desirability": round(d_hem, 4),
        "raw_display": hem_disp,
        "endpoint": primary_h.endpoint if primary_h else "hemolysis_cls",
        "evidence_grade": primary_h.evidence_grade if primary_h else None,
        "evidence_factor": round(gf_hem, 3),
        "ad_status": ad_hem,
    }

    # 4. Low Toxicity pillar (overall_peptide_toxicity 40% + cytotoxicity_cls 35% + cytotoxicity_CC50 25%)
    tox_parts: list[tuple[float, float, EndpointPrediction]] = []
    t_ov = by_ep.get("overall_peptide_toxicity")
    if t_ov is not None and t_ov.value is not None:
        tox_parts.append((0.40, _desirability_low_risk_prob(float(t_ov.value)), t_ov))
    t_cy = by_ep.get("cytotoxicity_cls")
    if t_cy is not None and t_cy.value is not None:
        tox_parts.append((0.35, _desirability_low_risk_prob(float(t_cy.value)), t_cy))
    t_cc50 = by_ep.get("cytotoxicity_CC50")
    if t_cc50 is not None and t_cc50.value is not None and t_cc50.value > 0:
        tox_parts.append((0.25, _desirability_safety_ic50(float(t_cc50.value)), t_cc50))
    if tox_parts:
        sw_sum = sum(x[0] for x in tox_parts)
        d_tox = sum(x[0] * x[1] for x in tox_parts) / sw_sum
        gf_tox = sum(x[0] * _grade_factor(x[2]) for x in tox_parts) / sw_sum
        af_tox = sum(x[0] * _ad_factor(x[2]) for x in tox_parts) / sw_sum
        primary_t = tox_parts[0][2]
        ad_tox = _worst_ad([x[2].ad_status for x in tox_parts])
        tox_disp = " · ".join(
            f"CC50 {x[2].value_display} µM" if x[2].endpoint == "cytotoxicity_CC50" else f"{x[2].endpoint.split('_')[0].title()} {x[2].value_display}"
            for x in tox_parts[:2]
        )
    else:
        d_tox, gf_tox, af_tox, primary_t, ad_tox, tox_disp = 0.65, 0.75, 0.92, None, "borderline", "N/A"
    pillar_scores["low_toxicity"] = round(d_tox, 4)
    pillar_grade_factors["low_toxicity"] = gf_tox
    pillar_ad_factors["low_toxicity"] = af_tox
    pillar_details["low_toxicity"] = {
        "desirability": round(d_tox, 4),
        "raw_display": tox_disp,
        "endpoint": primary_t.endpoint if primary_t else "overall_peptide_toxicity",
        "evidence_grade": primary_t.evidence_grade if primary_t else None,
        "evidence_factor": round(gf_tox, 3),
        "ad_status": ad_tox,
    }

    # Combine pillars with optional evidence-grade confidence weighting
    eff_weights: dict[str, float] = {}
    for k in ("permeability", "half_life", "low_hemolysis", "low_toxicity"):
        gf = pillar_grade_factors[k] if w_cfg.weight_by_evidence_grade else 1.0
        eff_weights[k] = norm_w[k] * gf
    eff_total = sum(eff_weights.values())
    if eff_total <= 1e-9:
        base_mpo_01 = sum(norm_w[k] * pillar_scores[k] for k in norm_w)
    else:
        base_mpo_01 = sum(eff_weights[k] * pillar_scores[k] for k in eff_weights) / eff_total

    # OOD penalty across the 4 MPO pillars
    if w_cfg.penalize_ood:
        ood_penalty = sum(norm_w[k] * pillar_ad_factors[k] for k in norm_w)
    else:
        ood_penalty = 1.0
    ood_penalty = round(_clamp(ood_penalty, 0.5, 1.0), 4)

    # AD counts across all predicted Tier-1 endpoints
    tier1_preds = [p for p in predictions if p.tier == 1] or predictions
    n_in = sum(1 for p in tier1_preds if p.ad_status in ("in_domain", "mechanism"))
    n_border = sum(1 for p in tier1_preds if p.ad_status == "borderline")
    n_ood = sum(1 for p in tier1_preds if p.ad_status == "ood")
    total_ad = max(1, n_in + n_border + n_ood)
    if n_ood / total_ad >= 0.50:
        ad_overall = "ood"
    elif n_in / total_ad >= 0.45 and n_ood / total_ad < 0.35:
        ad_overall = "in_domain"
    else:
        ad_overall = "borderline"

    mpo_score = round(_clamp(base_mpo_01 * ood_penalty, 0.0, 1.0) * 100.0, 2)
    mpo_tier = "top" if mpo_score >= 70.0 else ("moderate" if mpo_score >= 45.0 else "caution")

    return {
        "peptide_mpo_score": mpo_score,
        "mpo_tier": mpo_tier,
        "pillar_scores": pillar_scores,
        "pillar_details": pillar_details,
        "ood_penalty": ood_penalty,
        "ad_overall": ad_overall,
        "ad_counts": {"in_domain": n_in, "borderline": n_border, "ood": n_ood},
    }


def _csv_safe_cell(val: Any) -> str:
    """Escape leading spreadsheet formula characters (=, +, -, @, tab, CR) to prevent CSV injection."""
    if val is None:
        return ""
    s = str(val)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        # Allow pure negative numeric literals like -5.82
        try:
            float(s)
            return s
        except ValueError:
            return "'" + s
    return s


def _build_csv_export(results: list[BatchScreenResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    headers = [
        "rank",
        "id",
        "name",
        "status",
        "peptide_mpo_score",
        "mpo_tier",
        "ad_overall",
        "ood_penalty",
        "score_permeability",
        "score_half_life",
        "score_low_hemolysis",
        "score_low_toxicity",
        "half_life",
        "pampa",
        "caco2",
        "hemolysis_risk",
        "hc50",
        "toxicity",
        "cyto_risk",
        "mw",
        "logp",
        "net_charge_ph74",
        "sequence",
        "canonical_smiles",
        "prediction_source",
        "error",
    ]
    writer.writerow(headers)
    for r in results:
        km = r.key_metrics or {}
        ps = r.pillar_scores or {}
        writer.writerow(
            [
                r.rank if r.rank is not None else "",
                _csv_safe_cell(r.id),
                _csv_safe_cell(r.name),
                r.status,
                r.peptide_mpo_score if r.peptide_mpo_score is not None else "",
                r.mpo_tier,
                r.ad_overall,
                r.ood_penalty,
                ps.get("permeability", ""),
                ps.get("half_life", ""),
                ps.get("low_hemolysis", ""),
                ps.get("low_toxicity", ""),
                _csv_safe_cell(km.get("half_life", "")),
                _csv_safe_cell(km.get("pampa", "")),
                _csv_safe_cell(km.get("caco2", "")),
                _csv_safe_cell(km.get("hemolysis_risk", "")),
                _csv_safe_cell(km.get("hc50", "")),
                _csv_safe_cell(km.get("toxicity", "")),
                _csv_safe_cell(km.get("cyto_risk", "")),
                r.mw if r.mw is not None else "",
                r.logp if r.logp is not None else "",
                r.net_charge_ph74 if r.net_charge_ph74 is not None else "",
                _csv_safe_cell(r.sequence or ""),
                _csv_safe_cell(r.canonical_smiles or r.smiles or ""),
                r.prediction_source,
                _csv_safe_cell(r.error or ""),
            ]
        )
    return buf.getvalue()


def _parse_screen_items(req: BatchScreenRequest) -> tuple[list[tuple[str, BatchParsedRow]], bool]:
    """Convert structured items[] or raw_text into validated (compound_id, BatchParsedRow) pairs (max 100)."""
    truncated = False
    paired: list[tuple[str, BatchParsedRow]] = []

    if req.items is not None:
        if len(req.items) == 0:
            raise InputValidationError("Batch screen 'items' list cannot be empty.")
        if len(req.items) > MAX_SCREEN_ITEMS:
            truncated = True
        for idx, it in enumerate(req.items[:MAX_SCREEN_ITEMS], start=1):
            cid = (it.id or "").strip() or f"CMPD-{idx:03d}"
            cname = (it.name or "").strip() or cid
            row = validate_parsed_entry(
                index=idx,
                name=cname,
                query=it.query,
                sequence=it.sequence,
                smiles=it.smiles,
                drug_lookup_fn=lookup_drug,
            )
            paired.append((cid, row))
        return paired, truncated

    if req.raw_text and req.raw_text.strip():
        parsed = parse_batch_text(req.raw_text, filename=req.filename, drug_lookup_fn=lookup_drug)
        if parsed.total == 0:
            raise InputValidationError("No peptides could be parsed from raw_text.")
        truncated = parsed.truncated or (len(parsed.rows) > MAX_SCREEN_ITEMS)
        for idx, row in enumerate(parsed.rows[:MAX_SCREEN_ITEMS], start=1):
            cid = f"CMPD-{idx:03d}"
            paired.append((cid, row))
        return paired, truncated

    raise InputValidationError("Provide either 'items' (up to 100 compounds) or 'raw_text' for batch screening.")


def run_batch_screen(
    adapter: OmniPanelAdapter,
    req: BatchScreenRequest,
) -> BatchScreenResponse:
    """Execute Peptide-MPO batch screening, ranking, and CSV export."""
    t0 = time.perf_counter()
    paired_rows, truncated = _parse_screen_items(req)
    parsed_rows = [r for _, r in paired_rows]
    ids_by_pos = [cid for cid, _ in paired_rows]

    cond = req.conditions or AssayConditions()
    batch_results: list[BatchMoleculeResult] = adapter.predict_batch(
        parsed_rows,
        include_tier2=req.include_tier2,
        conditions=cond,
    )

    screen_rows: list[BatchScreenResult] = []
    for cid, bm in zip(ids_by_pos, batch_results):
        if bm.status != "ok":
            screen_rows.append(
                BatchScreenResult(
                    rank=None,
                    id=cid,
                    name=bm.name,
                    status="error",
                    error=bm.error or "Prediction failed",
                    mpo_tier="error",
                    ad_overall="ood",
                    prediction_source="error",
                )
            )
            continue

        mpo = compute_peptide_mpo(
            predictions=bm.predictions,
            weights=req.weights,
            mechanism_anchor=bm.mechanism_anchor,
        )
        screen_rows.append(
            BatchScreenResult(
                rank=None,
                id=cid,
                name=bm.name,
                status="ok",
                resolved_type=bm.resolved_type,
                sequence=bm.sequence,
                smiles=bm.smiles,
                canonical_smiles=bm.canonical_smiles,
                peptide_mpo_score=mpo["peptide_mpo_score"],
                mpo_tier=mpo["mpo_tier"],  # type: ignore[arg-type]
                pillar_scores=mpo["pillar_scores"],
                pillar_details=mpo["pillar_details"],
                ood_penalty=mpo["ood_penalty"],
                ad_overall=mpo["ad_overall"],  # type: ignore[arg-type]
                ad_counts=mpo["ad_counts"],
                mw=bm.mw,
                logp=bm.logp,
                net_charge_ph74=bm.net_charge_ph74,
                key_metrics=bm.key_metrics,
                prediction_source=bm.prediction_source,
            )
        )

    # Rank ok compounds descending by Peptide-MPO score, errors last
    ok_rows = [r for r in screen_rows if r.status == "ok"]
    err_rows = [r for r in screen_rows if r.status != "ok"]
    ok_rows.sort(
        key=lambda r: (
            -(r.peptide_mpo_score if r.peptide_mpo_score is not None else -1.0),
            -(r.ad_counts.get("in_domain", 0)),
            r.name.lower(),
        )
    )
    for rank_idx, r in enumerate(ok_rows, start=1):
        r.rank = rank_idx
    sorted_results = ok_rows + err_rows

    scores = [float(r.peptide_mpo_score) for r in ok_rows if r.peptide_mpo_score is not None]
    summary = {
        "mean_mpo": round(sum(scores) / len(scores), 2) if scores else None,
        "top_mpo": round(max(scores), 2) if scores else None,
        "n_top_tier": sum(1 for r in ok_rows if r.mpo_tier == "top"),
        "n_in_domain_majority": sum(1 for r in ok_rows if r.ad_overall == "in_domain"),
        "n_ood_flagged": sum(1 for r in ok_rows if r.ad_overall == "ood"),
    }

    csv_export = _build_csv_export(sorted_results)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return BatchScreenResponse(
        total=len(sorted_results),
        succeeded=len(ok_rows),
        failed=len(err_rows),
        truncated=truncated,
        max_items=MAX_SCREEN_ITEMS,
        weights_used=req.weights,
        results=sorted_results,
        summary=summary,
        csv_export=csv_export,
        runtime_ms=elapsed_ms,
        adapter_mode=adapter.mode,
    )
