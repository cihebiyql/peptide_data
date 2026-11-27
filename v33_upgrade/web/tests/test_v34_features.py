"""Comprehensive pytest suite for Peptide OmniPanel V3.4 backend engines & APIs.

Covers:
  - P0-2 / P0-3: v32_cv_note provenance tag, clinical_percentile wiring, v34_upgrades catalog field
  - P1-5: POST /api/v1/peptide-opt (all 5 modification strategies, RDKit sanitization, Pareto optimality)
  - P1-6: POST /api/v1/batch-screen (Peptide-MPO desirability, evidence weighting, OOD penalty, CSV export)
  - P2-8: POST /api/v1/direct-attribution (residue masking + valid substituent pruning across all 47 endpoints)
"""

from __future__ import annotations

import csv
import io
import os

import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

os.environ.setdefault("V33_OFFLINE_ONLY", "1")

from v33_upgrade.web.app import app
from v33_upgrade.web.chem_utils import detect_motifs_and_highlights
from v33_upgrade.web.drug_library import lookup_drug

client = TestClient(app)


def _assert_valid_single_frag_smiles(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"RDKit failed to parse SMILES: {smi}"
    Chem.SanitizeMol(mol)
    assert len(Chem.GetMolFrags(mol)) == 1, f"SMILES produced disconnected fragments: {smi}"
    return mol


def test_p0_v32_cv_note_and_clinical_percentile_wiring():
    """Verify v32_cv_note on /endpoints and /predict, v34_upgrades dict, and clinical_percentile field."""
    expected_note = "deprecated_legacy_unclustered_cv (historical reference only; do not use for model selection)"

    r_ep = client.get("/api/v1/endpoints")
    assert r_ep.status_code == 200
    ep_data = r_ep.json()
    assert "v34_upgrades" in ep_data and isinstance(ep_data["v34_upgrades"], dict)
    assert len(ep_data["endpoints"]) == 47
    for item in ep_data["endpoints"]:
        assert item["v32_cv_note"] == expected_note

    r_pred = client.post("/api/v1/predict", json={"query": "Semaglutide", "include_svg": False})
    assert r_pred.status_code == 200
    pred_data = r_pred.json()
    assert len(pred_data["predictions"]) == 47
    by_ep = {p["endpoint"]: p for p in pred_data["predictions"]}
    for p in pred_data["predictions"]:
        assert p["v32_cv_note"] == expected_note
        cp = p["clinical_percentile"]
        assert cp is not None, f"Missing clinical_percentile for {p['endpoint']}"
        assert 0.0 <= cp["percentile"] <= 100.0
        assert cp["reference_group"] in ("approved_clinical_peptides", "curated_peptide_panel")
        assert cp["n_reference"] >= 1
        assert isinstance(cp["summary_en"], str) and cp["summary_en"]
        assert isinstance(cp["summary_zh"], str) and cp["summary_zh"]
        assert isinstance(cp["nearest_approved_examples"], list)

    # Spot-check clinical PK endpoint uses approved_clinical_peptides with nearest approved examples
    hl_cp = by_ep["half_life_reg"]["clinical_percentile"]
    assert hl_cp["reference_group"] == "approved_clinical_peptides"
    assert len(hl_cp["nearest_approved_examples"]) >= 1


def test_p1_5_peptide_opt_linear_all_5_strategies_and_pareto():
    """Test PeptideOpt on a canonical linear peptide with all 5 modification strategies."""
    seq = "KWFKPLKGI"
    res = client.post(
        "/api/v1/peptide-opt",
        json={
            "sequence": seq,
            "strategies": [
                "ala_scan",
                "point_mutation",
                "n_methylation",
                "cyclization",
                "lipidation_c16_c18",
            ],
            "max_variants": 24,
            "objective_x": {"endpoint": "PAMPA", "direction": "max"},
            "objective_y": {"endpoint": "half_life_reg", "direction": "max"},
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()

    parent = data["parent"]
    assert parent["is_parent"] is True
    assert parent["strategy"] == "parent"
    parent_mol = _assert_valid_single_frag_smiles(parent["smiles"])
    parent_can = Chem.MolToSmiles(parent_mol, canonical=True)

    variants = data["variants"]
    assert 5 <= len(variants) <= 24
    assert data["total_evaluated"] == len(variants)
    assert set(data["strategies_used"]) == {
        "ala_scan",
        "point_mutation",
        "n_methylation",
        "cyclization",
        "lipidation_c16_c18",
    }

    # Verify every variant is chemically valid, single-fragment, and deduplicated by canonical SMILES
    seen_can = {parent_can}
    strats_seen: set[str] = set()
    for v in variants:
        mol = _assert_valid_single_frag_smiles(v["smiles"])
        can = Chem.MolToSmiles(mol, canonical=True)
        assert can not in seen_can, f"Duplicate canonical SMILES in PeptideOpt: {v['mutation_label']}"
        seen_can.add(can)
        strats_seen.add(v["strategy"])
        assert 0.0 <= v["mpo_score"] <= 100.0
        assert v["ad_status"] in ("in_domain", "borderline", "ood")
        assert isinstance(v["safety_guardrail_pass"], bool)
        assert v["delta_x"] is not None and v["delta_y"] is not None

        if v["strategy"] == "cyclization":
            has_macro = any(len(r) >= 12 for r in mol.GetRingInfo().AtomRings())
            badges, _ = detect_motifs_and_highlights(mol)
            badge_keys = {b.key for b in badges}
            assert has_macro or ("disulfide" in badge_keys), f"Cyclization failed to create ring/disulfide: {v}"

        if v["strategy"] == "n_methylation":
            badges, _ = detect_motifs_and_highlights(mol)
            assert any(b.key == "n_methyl" for b in badges), f"N-methylation missing n_methyl motif: {v}"
            assert v["hbd"] < parent["hbd"]

        if v["strategy"] == "lipidation_c16_c18":
            badges, _ = detect_motifs_and_highlights(mol)
            badge_keys = {b.key for b in badges}
            assert "fatty_diacid" in badge_keys or "fatty_monoacyl" in badge_keys, f"Lipidation missing lipid motif: {v}"

    assert len(strats_seen) == 5

    # Verify exact 2D Pareto dominance math over {parent} U variants
    all_pts = [parent] + variants
    for cand in all_pts:
        dominated = any(
            other["id"] != cand["id"]
            and other["x_raw"] >= cand["x_raw"] - 1e-9
            and other["y_raw"] >= cand["y_raw"] - 1e-9
            and (other["x_raw"] > cand["x_raw"] + 1e-9 or other["y_raw"] > cand["y_raw"] + 1e-9)
            for other in all_pts
        )
        assert cand["is_pareto_optimal"] == (not dominated)

    frontier = data["pareto_frontier"]
    assert len(frontier) == data["pareto_count"] >= 1
    assert all(f["is_pareto_optimal"] for f in frontier)
    # Frontier sorted ascending by x_raw for SVG polyline
    xs = [f["x_raw"] for f in frontier]
    assert xs == sorted(xs)


def test_p1_5_peptide_opt_cyclic_smiles_and_validation():
    """Test PeptideOpt on Cyclosporine A (cyclic SMILES) and input validation edge cases."""
    csa = lookup_drug("Cyclosporine A")
    assert csa is not None and csa["smiles"]

    res = client.post(
        "/api/v1/peptide-opt",
        json={
            "smiles": csa["smiles"],
            "strategies": ["ala_scan", "n_methylation", "cyclization"],
            "max_variants": 12,
            "objective_x": {"endpoint": "LogD7.4", "direction": "max"},
            "objective_y": {"endpoint": "hemolysis_cls", "direction": "min"},
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert "n_methylation" in data["strategies_used"]
    assert len(data["variants"]) >= 1
    for v in data["variants"]:
        _assert_valid_single_frag_smiles(v["smiles"])
    skipped_strats = {s["strategy"] for s in data["strategies_skipped"]}
    assert "ala_scan" in skipped_strats
    assert "cyclization" in skipped_strats

    # Cap > 32 rejected by Pydantic (422)
    r_cap = client.post("/api/v1/peptide-opt", json={"sequence": "KWFKPLKGI", "max_variants": 40})
    assert r_cap.status_code == 422

    # Unknown objective endpoint rejected (400)
    r_bad_ep = client.post(
        "/api/v1/peptide-opt",
        json={"sequence": "KWFKPLKGI", "objective_x": {"endpoint": "FAKE_EP", "direction": "max"}},
    )
    assert r_bad_ep.status_code == 400


def test_p1_6_batch_screen_mpo_ranking_weights_ood_and_csv():
    """Test Batch Screening + Peptide-MPO desirability, OOD penalty, evidence weighting, and CSV export."""
    items = [
        {"id": "S1", "name": "Semaglutide", "query": "Semaglutide"},
        {"id": "S2", "name": "Cyclosporine A", "query": "Cyclosporine A"},
        {"id": "S3", "name": "Magainin-2", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS"},
        {"id": "S4", "name": "=SUM(A1:A5)", "smiles": "CC(=O)NC(C)C(=O)NC(CC1=CC=CC=C1)C(=O)O"},
        {"id": "S5", "name": "BadEntry", "smiles": "(((NOT_VALID)))"},
    ]

    r_pen = client.post(
        "/api/v1/batch-screen",
        json={
            "items": items,
            "weights": {
                "permeability": 0.25,
                "half_life": 0.25,
                "low_hemolysis": 0.25,
                "low_toxicity": 0.25,
                "penalize_ood": True,
                "weight_by_evidence_grade": True,
            },
        },
    )
    assert r_pen.status_code == 200, r_pen.text
    d_pen = r_pen.json()
    assert d_pen["total"] == 5
    assert d_pen["succeeded"] == 4
    assert d_pen["failed"] == 1
    assert d_pen["truncated"] is False

    ok_rows = [r for r in d_pen["results"] if r["status"] == "ok"]
    err_rows = [r for r in d_pen["results"] if r["status"] == "error"]
    assert len(ok_rows) == 4 and len(err_rows) == 1
    assert [r["rank"] for r in ok_rows] == [1, 2, 3, 4]
    assert err_rows[0]["rank"] is None

    # Sorted descending by peptide_mpo_score
    scores = [r["peptide_mpo_score"] for r in ok_rows]
    assert scores == sorted(scores, reverse=True)
    for r in ok_rows:
        assert 0.0 <= r["peptide_mpo_score"] <= 100.0
        for pk in ("permeability", "half_life", "low_hemolysis", "low_toxicity"):
            assert 0.0 <= r["pillar_scores"][pk] <= 1.0
        assert 0.5 <= r["ood_penalty"] <= 1.0

    # Verify CSV export parses cleanly and escapes Excel formula injection (=SUM...)
    csv_text = d_pen["csv_export"]
    rows = list(csv.reader(io.StringIO(csv_text)))
    assert len(rows) == 6  # header + 5 items
    assert rows[0][0] == "rank" and "peptide_mpo_score" in rows[0]
    assert any("'=SUM(A1:A5)" in row for row in rows)

    # Compare with penalize_ood=False -> ood_penalty must be 1.0 and MPO >= penalized MPO
    r_nopen = client.post(
        "/api/v1/batch-screen",
        json={
            "items": items[:4],
            "weights": {
                "permeability": 0.25,
                "half_life": 0.25,
                "low_hemolysis": 0.25,
                "low_toxicity": 0.25,
                "penalize_ood": False,
                "weight_by_evidence_grade": False,
            },
        },
    )
    assert r_nopen.status_code == 200
    d_nopen = r_nopen.json()
    for r in d_nopen["results"]:
        assert r["ood_penalty"] == 1.0

    # Half-life-only weight should rank once-weekly Semaglutide #1
    r_hl = client.post(
        "/api/v1/batch-screen",
        json={
            "items": items[:3],
            "weights": {
                "permeability": 0.0,
                "half_life": 1.0,
                "low_hemolysis": 0.0,
                "low_toxicity": 0.0,
                "penalize_ood": True,
                "weight_by_evidence_grade": True,
            },
        },
    )
    assert r_hl.status_code == 200
    assert r_hl.json()["results"][0]["name"] == "Semaglutide"

    # Raw FASTA text input also supported
    r_raw = client.post(
        "/api/v1/batch-screen",
        json={"raw_text": ">P1\nKWFKPLKGI\n>P2\nGIGKFLHSAKKFGKAFVGEIMNS\n", "filename": "peps.fasta"},
    )
    assert r_raw.status_code == 200
    assert r_raw.json()["succeeded"] == 2

    # >100 items rejected with 422
    too_many = [{"id": f"X{i}", "sequence": "KWFKPLKGI"} for i in range(101)]
    assert client.post("/api/v1/batch-screen", json={"items": too_many}).status_code == 422


def test_p2_8_direct_attribution_linear_cyclic_lipidated_and_all_endpoints():
    """Test model-faithful Direct Ensemble Perturbation Attribution on linear, cyclic, and lipidated peptides."""
    # 1. Canonical linear peptide -> residue Ala/Gly masking, works on non-EGAT Tier-2 endpoint (MIC_E_coli) & PAMPA
    for ep in ("PAMPA", "hemolysis_HC50", "MIC_E_coli"):
        res = client.post(
            "/api/v1/direct-attribution",
            json={"sequence": "KWFKPLKGI", "endpoint": ep, "max_perturbations": 12},
        )
        assert res.status_code == 200, res.text
        d = res.json()
        assert d["interpretability_tier"] == "direct_ensemble_faithful"
        assert d["method"] == "residue_alanine_glycine_masking"
        assert d["endpoint"] == ep
        assert len(d["available_endpoints"]) == 47
        assert d["n_perturbations"] == 9  # 9 residues
        assert len(d["atom_weights"]) == d["n_atoms"]
        assert d["svg_2d"] == d["svg"] and "<svg" in d["svg_2d"]
        assert "Direct Ensemble Perturbation" in d["svg_2d"]
        for aw in d["atom_weights"]:
            assert -1.0 <= aw["weight"] <= 1.0
            assert 0.0 <= aw["weight_01"] <= 1.0
        for imp in d["residue_or_group_impacts"]:
            _assert_valid_single_frag_smiles(imp["perturbed_smiles"])
            assert -1.0 <= imp["normalized_weight"] <= 1.0
            assert imp["direction"] in ("increases", "decreases", "neutral")

    # Cache hit on identical request
    res_cached = client.post(
        "/api/v1/direct-attribution",
        json={"sequence": "KWFKPLKGI", "endpoint": "PAMPA", "max_perturbations": 12},
    )
    assert res_cached.status_code == 200
    assert res_cached.json()["cache_hit"] is True

    # 2. Cyclic N-methylated macrocycle (Cyclosporine A) -> chemical substituent pruning (ZERO broken valences!)
    r_csa = client.post(
        "/api/v1/direct-attribution",
        json={"query": "Cyclosporine A", "endpoint": "PAMPA", "max_perturbations": 16},
    )
    assert r_csa.status_code == 200, r_csa.text
    d_csa = r_csa.json()
    assert d_csa["method"] == "chemical_substituent_pruning"
    assert d_csa["n_atoms"] == 85
    assert d_csa["n_perturbations"] >= 8
    kinds_csa = {imp["kind"] for imp in d_csa["residue_or_group_impacts"]}
    assert "n_methyl" in kinds_csa
    for imp in d_csa["residue_or_group_impacts"]:
        _assert_valid_single_frag_smiles(imp["perturbed_smiles"])
        assert len(imp["atom_indices"]) >= 1

    # 3. Lipidated clinical peptide (Semaglutide) & cyclic disulfide (Octreotide)
    r_sema = client.post(
        "/api/v1/direct-attribution",
        json={"query": "Semaglutide", "endpoint": "PPB", "max_perturbations": 12},
    )
    assert r_sema.status_code == 200, r_sema.text
    d_sema = r_sema.json()
    assert d_sema["method"] == "chemical_substituent_pruning"
    assert d_sema["n_perturbations"] >= 5
    kinds_sema = {imp["kind"] for imp in d_sema["residue_or_group_impacts"]}
    assert "lipid_tail" in kinds_sema
    for imp in d_sema["residue_or_group_impacts"]:
        _assert_valid_single_frag_smiles(imp["perturbed_smiles"])

    r_oct = client.post(
        "/api/v1/direct-attribution",
        json={"query": "Octreotide", "endpoint": "cytotoxicity_CC50", "max_perturbations": 12},
    )
    assert r_oct.status_code == 200, r_oct.text
    d_oct = r_oct.json()
    assert d_oct["method"] == "chemical_substituent_pruning"
    assert d_oct["n_perturbations"] >= 4
    for imp in d_oct["residue_or_group_impacts"]:
        _assert_valid_single_frag_smiles(imp["perturbed_smiles"])

    # 4. Unknown endpoint rejected with 400
    r_bad = client.post(
        "/api/v1/direct-attribution",
        json={"query": "Cyclosporine A", "endpoint": "NON_EXISTENT_ENDPOINT"},
    )
    assert r_bad.status_code == 400


def test_p1_5_peptide_opt_disulfide_cyclization_and_lipidated_parent():
    """Verify Cys-Cys oxidative disulfide cyclization and lipidation on linear/clinical peptides."""
    r_cys = client.post(
        "/api/v1/peptide-opt",
        json={
            "sequence": "CKWFKPLKGC",
            "strategies": ["cyclization", "lipidation_c16_c18"],
            "max_variants": 10,
            "objective_x": {"endpoint": "PAMPA", "direction": "max"},
            "objective_y": {"endpoint": "half_life_reg", "direction": "max"},
        },
    )
    assert r_cys.status_code == 200, r_cys.text
    d_cys = r_cys.json()
    labels = [v["mutation_label"] for v in d_cys["variants"]]
    assert any("Disulfide" in lbl for lbl in labels)
    assert any("Head-to-Tail" in lbl for lbl in labels)
    for v in d_cys["variants"]:
        _assert_valid_single_frag_smiles(v["smiles"])

    # Lipidated clinical peptide parent (Liraglutide)
    r_lira = client.post(
        "/api/v1/peptide-opt",
        json={
            "query": "Liraglutide",
            "strategies": ["n_methylation", "lipidation_c16_c18"],
            "max_variants": 6,
            "objective_x": {"endpoint": "PAMPA", "direction": "max"},
            "objective_y": {"endpoint": "PPB", "direction": "max"},
        },
    )
    assert r_lira.status_code == 200, r_lira.text
    d_lira = r_lira.json()
    assert len(d_lira["variants"]) >= 1
    for v in d_lira["variants"]:
        _assert_valid_single_frag_smiles(v["smiles"])
