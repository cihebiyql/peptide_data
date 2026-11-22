"""Comprehensive pytest API test suite for Peptide OmniPanel V33 (24 unified endpoints)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Force offline manifest mode during unit tests unless explicitly overridden
os.environ.setdefault("V33_OFFLINE_ONLY", "1")

from v33_upgrade.web.adapter import OmniPanelAdapter, V33_ENDPOINT_ORDER
from v33_upgrade.web.app import app
from v33_upgrade.web.drug_library import lookup_drug

client = TestClient(app)

EXPECTED_TIER1 = {
    "half_life_reg",
    "half_life_cls",
    "PPB",
    "CL_per_kg",
    "CL_absolute",
    "Vd_per_kg",
    "F",
    "Cmax",
    "PAMPA",
    "Caco2",
    "MDCK",
    "RRCK",
    "BBB",
    "hemolysis_cls",
    "hemolysis_HC50",
    "overall_peptide_toxicity",
    "cytotoxicity_cls",
    "LogD7.4",
    "solubility_reg",
    "solubility_cls",
}
EXPECTED_TIER2 = {
    "MIC_E_coli",
    "MIC_S_aureus",
    "MIC_P_aeruginosa",
    "MIC_C_albicans",
}


def test_health_endpoint_real_manifest_kpis():
    manifest_path = Path(__file__).resolve().parents[2] / "bundle" / "v33_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp_eps = manifest["endpoints"]
    exp_t1 = sum(1 for v in exp_eps.values() if int(v.get("tier", 1)) == 1)
    exp_t2 = sum(1 for v in exp_eps.values() if int(v.get("tier", 1)) == 2)

    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    assert data["scope"] == "internal_research_only"
    assert data["adapter_mode"] in ("v33_remote", "v33_offline_fallback")
    assert data["manifest_loaded"] is True
    assert data["endpoints_total"] == len(exp_eps)
    assert data["tier1_endpoints"] == exp_t1
    assert data["tier2_endpoints"] == exp_t2
    assert data["case17_drugs_loaded"] == 17

    ds = data["dataset_summary"]
    assert ds["n_unique_compounds"] >= 46597
    assert ds["n_endpoints"] == len(exp_eps)
    assert ds["n_total_rows"] >= 69111
    assert ds["n_censored_rows"] >= 2523


def test_endpoints_catalog_zero_fabricated_metrics():
    manifest_path = Path(__file__).resolve().parents[2] / "bundle" / "v33_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp_eps = manifest["endpoints"]

    res = client.get("/api/v1/endpoints")
    assert res.status_code == 200
    data = res.json()
    assert data["scope"] == "internal_research_only"
    assert len(data["endpoints"]) == len(exp_eps)

    eps = {e["endpoint"]: e for e in data["endpoints"]}
    assert EXPECTED_TIER1.issubset({e["endpoint"] for e in data["endpoints"] if e["tier"] == 1})
    assert EXPECTED_TIER2.issubset({e["endpoint"] for e in data["endpoints"] if e["tier"] == 2})
    assert set(eps.keys()) == set(exp_eps.keys())

    # Spot-check real audited v33_manifest.json metrics (never synthetic V32 counts or fake lifts)
    for ep_name in ("half_life_reg", "PAMPA", "hemolysis_cls", "hemolysis_HC50", "MIC_E_coli", "cytotoxicity_cls"):
        ep = eps[ep_name]
        m_ep = manifest["endpoints"][ep_name]
        assert ep["n_train"] == m_ep["counts"]["n_cv"]
        assert ep["n_holdout"] == m_ep["counts"]["n_holdout"]
        assert ep["n_censored"] == m_ep["counts"]["n_censored_total"]
        assert ep["winning_family"] == m_ep["dual_modality"]["winning_family"]

        mkey = "roc_auc" if m_ep["task_type"] == "classification" else "spearman"
        assert ep["cv_score_v33"] == pytest.approx(m_ep["dual_modality"]["cv_metrics"][mkey], abs=1e-4)
        assert ep["holdout_score_v33"] == pytest.approx(m_ep["dual_modality"]["holdout_metrics"][mkey], abs=1e-4)

    # New V33 endpoints with no V32 baseline must explicitly report None (never fabricated!)
    assert eps["cytotoxicity_cls"]["cv_score_v32"] is None
    assert eps["MIC_S_aureus"]["cv_score_v32"] is None

    # Verify Case17 clinical benchmark and 4-FM dominance summary are present and real
    c17 = data["case17_scorecard"]
    assert c17["half_life_pure_ml_route_matched_log10"]["spearman"] == pytest.approx(0.9029, abs=1e-3)
    assert c17["half_life_with_mechanism_anchor_log10"]["mae"] == pytest.approx(0.2987, abs=1e-3)
    assert c17["ppb_dual"]["spearman"] == pytest.approx(0.7093, abs=1e-3)
    assert c17["hemolysis_false_positive_check"]["n_drugs_gt_0_50"] == 0

    ex_res = client.get("/api/v1/examples")
    assert ex_res.status_code == 200
    ex = ex_res.json()
    assert len(ex["quick_examples"]) == 6
    assert len(ex["example_categories"]) == 5
    assert len(ex["drugs"]) == 25
    assert set(ex["batch_presets"].keys()) >= {"demo_7", "case17_17", "echinocandin_6"}


def test_predict_canonical_sequence_amp():
    seq = "HISEAVKIFGCKWKENWKWNSNC"
    res = client.post("/api/v1/predict", json={"query": seq, "include_tier2": True})
    assert res.status_code == 200
    data = res.json()
    assert data["resolved_type"] == "sequence"
    assert data["sequence"] == seq
    assert data["physchem"]["length_aa"] == len(seq)
    assert data["physchem"]["mw"] > 2000
    assert data["svg"] and "<svg" in data["svg"]
    assert data["mechanism_anchor"]["detected"] is False
    assert len(data["radar"]) == 6
    assert len(data["predictions"]) >= 24

    eps = {p["endpoint"]: p for p in data["predictions"]}
    assert EXPECTED_TIER1.issubset(set(eps.keys()))
    assert EXPECTED_TIER2.issubset(set(eps.keys()))
    assert "MIC_E_coli" in eps
    assert "MIC_S_aureus" in eps
    assert "MIC_P_aeruginosa" in eps
    assert "MIC_C_albicans" in eps
    assert "hemolysis_HC50" in eps
    assert eps["PAMPA"]["ci80_low"] is not None and eps["PAMPA"]["ci80_low"] > 0
    assert eps["PAMPA"]["ci80_high"] > eps["PAMPA"]["ci80_low"]
    assert "0.00e+00" not in (eps["PAMPA"]["ci80_display"] or "")
    assert eps["hemolysis_cls"]["calibrated_prob"] is not None
    assert len(eps["PAMPA"]["nearest_analogs"]) == 3


def test_predict_semaglutide_smiles_and_name():
    sema = lookup_drug("Semaglutide")
    assert sema is not None and sema["smiles"]

    res = client.post("/api/v1/predict", json={"smiles": sema["smiles"], "conditions": {"pk_context": "human_sc"}})
    assert res.status_code == 200
    data = res.json()
    assert data["query_name"] == "Semaglutide"
    assert data["mechanism_anchor"]["detected"] is True
    assert data["mechanism_anchor"]["acylation_class"] == "weekly_diacid"
    assert data["mechanism_anchor"]["range_h"] == [120.0, 168.0]
    assert data["mechanism_anchor"]["ml_baseline_h"] is not None
    assert any(b["key"] == "fatty_diacid" for b in data["modifications"])
    assert any(h["key"] == "fatty_acyl" for h in data["svg_highlights"])

    by_ep = {p["endpoint"]: p for p in data["predictions"]}
    # V33 fixes V32's false-positive hemolysis on Semaglutide (must be < 0.50!)
    assert by_ep["hemolysis_cls"]["value"] < 0.50
    assert by_ep["PPB"]["value"] > 0.65

    # Chinese drug name lookup
    res_zh = client.post("/api/v1/predict", json={"query": "司美格鲁肽", "include_svg": False})
    assert res_zh.status_code == 200
    assert res_zh.json()["query_name"] == "Semaglutide"


def test_predict_tirzepatide_and_cyclosporine_a():
    tirz = lookup_drug("Tirzepatide")
    assert tirz is not None
    r_t = client.post("/api/v1/predict", json={"smiles": tirz["smiles"], "include_svg": False})
    assert r_t.status_code == 200
    dt = r_t.json()
    assert dt["mechanism_anchor"]["detected"] is True
    assert dt["mechanism_anchor"]["acylation_class"] == "weekly_diacid"
    assert dt["mechanism_anchor"]["chain_carbons"] == 20
    assert any(b["key"] == "aib" for b in dt["modifications"])
    assert dt["physchem"]["mw"] > 4700
    assert next(p for p in dt["predictions"] if p["endpoint"] == "hemolysis_cls")["value"] < 0.50

    csa = lookup_drug("Cyclosporine A")
    assert csa is not None
    r_c = client.post("/api/v1/predict", json={"smiles": csa["smiles"]})
    assert r_c.status_code == 200
    dc = r_c.json()
    assert dc["mechanism_anchor"]["detected"] is False
    mod_keys = {b["key"] for b in dc["modifications"]}
    assert "macrocycle" in mod_keys
    assert "n_methyl" in mod_keys
    hl_keys = {h["key"] for h in dc["svg_highlights"]}
    assert "n_methyl" in hl_keys
    assert "macrocycle" in hl_keys


def test_predict_dual_input_and_condition_routing():
    exen = lookup_drug("Exenatide")
    assert exen is not None
    res_human = client.post(
        "/api/v1/predict",
        json={
            "sequence": exen["sequence"],
            "smiles": exen["smiles"],
            "conditions": {"rbc_species": "human", "pk_context": "human_plasma"},
            "include_svg": False,
        },
    )
    assert res_human.status_code == 200
    d_h = res_human.json()
    assert d_h["resolved_type"] == "dual"
    hc50_h = next(p for p in d_h["predictions"] if p["endpoint"] == "hemolysis_HC50")
    assert "human" in (hc50_h["assay_context"] or "")

    res_mouse = client.post(
        "/api/v1/predict",
        json={
            "sequence": exen["sequence"],
            "smiles": exen["smiles"],
            "conditions": {"rbc_species": "mouse", "pk_context": "human_sc"},
            "include_svg": False,
        },
    )
    assert res_mouse.status_code == 200
    hc50_m = next(p for p in res_mouse.json()["predictions"] if p["endpoint"] == "hemolysis_HC50")
    assert "mouse" in (hc50_m["assay_context"] or "")
    assert hc50_m["value"] != hc50_h["value"]


def test_predict_garbage_input_returns_clean_400():
    for bad in ("???not_a_molecule###", "123@@@", "1234567890", "XYZ_INVALID_AA_SEQ"):
        res = client.post("/api/v1/predict", json={"query": bad})
        assert res.status_code == 400
        assert "detail" in res.json()

    assert client.post("/api/v1/predict", json={}).status_code == 400
    assert client.post("/api/v1/predict", json={"smiles": "C(C(C(((("}).status_code == 400


def test_render_svg_smiles_and_sequence():
    r_seq = client.post("/api/v1/render-svg", json={"sequence": "GIGKFLHSAKKFGKAFVGEIMNS"})
    assert r_seq.status_code == 200
    assert "<svg" in r_seq.json()["svg"]
    assert r_seq.json()["atom_count"] > 100

    r_oxy = client.post("/api/v1/render-svg", json={"query": "Oxytocin"})
    assert r_oxy.status_code == 200
    assert any(h["key"] == "disulfide" for h in r_oxy.json()["highlights"])

    assert client.post("/api/v1/render-svg", json={"query": "%%%invalid%%%"}).status_code == 400


def test_batch_fasta_and_csv_and_upload():
    fasta = ">AMP_1\nHISEAVKIFGCKWKENWKWNSNC\n>Magainin2\nGIGKFLHSAKKFGKAFVGEIMNS\n"
    r_parse = client.post("/api/v1/batch/parse", json={"raw_text": fasta, "filename": "amps.fasta"})
    assert r_parse.status_code == 200
    assert r_parse.json()["valid_count"] == 2

    r_batch = client.post("/api/v1/batch", json={"raw_text": fasta, "filename": "amps.fasta", "include_tier2": True})
    assert r_batch.status_code == 200
    bd = r_batch.json()
    assert bd["total"] == 2
    assert bd["succeeded"] == 2
    assert bd["failed"] == 0
    assert len(bd["results"][0]["predictions"]) >= 24
    assert len(bd["results"][0]["radar"]) == 6

    csv_text = "name,query\nSemaglutide,Semaglutide\nCyclosporine A,Cyclosporine A\nBadRow,???garbage###\n"
    r_csv = client.post("/api/v1/batch", json={"raw_text": csv_text, "filename": "mix.csv"})
    assert r_csv.status_code == 200
    cd = r_csv.json()
    assert cd["total"] == 3
    assert cd["succeeded"] == 2
    assert cd["failed"] == 1
    assert cd["results"][0]["mechanism_anchor"]["detected"] is True
    assert cd["results"][0]["key_metrics"]["half_life"].endswith("★")

    r_up = client.post(
        "/api/v1/batch/upload",
        files={"file": ("peptides.csv", csv_text.encode("utf-8"), "text/csv")},
        data={"include_tier2": "false", "rbc_species": "human"},
    )
    assert r_up.status_code == 200
    assert r_up.json()["succeeded"] == 2


def test_spa_root_serves_subpath_safe_html_without_hardcoded_v32_counts():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "OmniPanel" in html
    # Subpath-safe relative asset references (no leading slash)
    assert 'href="static/styles.css"' in html
    assert 'src="static/app.js"' in html
    assert 'href="docs"' in html
    # Zero hardcoded stale V32 counts anywhere in HTML
    assert "58,932" not in html
    assert "213,800" not in html


def test_adapter_manifest_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_src = Path(__file__).resolve().parents[2] / "bundle" / "v33_manifest.json"
    copied = tmp_path / "custom_v33_manifest.json"
    copied.write_text(manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("V33_MANIFEST_PATH", str(copied))
    monkeypatch.setenv("V33_OFFLINE_ONLY", "1")

    ad = OmniPanelAdapter()
    assert ad.manifest_path == copied
    assert ad.mode == "v33_offline_fallback"
    assert len(ad.get_catalog()) >= 24
    assert ad.get_dataset_summary()["n_unique_compounds"] >= 46597


def test_all_25_curated_examples_and_echinocandins():
    """Verify all 17 Case17 drugs + 3 added FDA Echinocandins + 5 AMPs resolve & predict cleanly."""
    from v33_upgrade.web.drug_library import DRUG_LIBRARY, BATCH_PRESETS

    assert len(DRUG_LIBRARY) == 25
    for name in ("Micafungin", "Anidulafungin", "Rezafungin", "Caspofungin", "米卡芬净", "阿尼芬净", "瑞扎芬净"):
        res = client.post("/api/v1/predict", json={"query": name, "include_svg": False})
        assert res.status_code == 200, f"Failed on {name}: {res.text}"
        d = res.json()
        assert d["physchem"]["mw"] > 1000
        assert any(b["key"] == "macrocycle" for b in d["modifications"])

    # Verify Case17 17-drug batch preset parses 17/17 valid
    r_c17 = client.post("/api/v1/batch/parse", json={"raw_text": BATCH_PRESETS["case17_17"]["csv"], "filename": "case17.csv"})
    assert r_c17.status_code == 200
    assert r_c17.json()["valid_count"] == 17
    assert r_c17.json()["invalid_count"] == 0

    # Verify Echinocandins 6-drug batch preset parses 6/6 valid
    r_ec = client.post("/api/v1/batch/parse", json={"raw_text": BATCH_PRESETS["echinocandin_6"]["csv"], "filename": "echino.csv"})
    assert r_ec.status_code == 200
    assert r_ec.json()["valid_count"] == 6
    assert r_ec.json()["invalid_count"] == 0
