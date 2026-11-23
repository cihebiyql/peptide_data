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


def test_spa_root_serves_subpath_safe_html_without_internal_codenames():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    assert "OmniPanel" in html
    assert 'href="static/styles.css?v=1.2' in html
    assert 'src="static/app.js?v=1.2' in html
    assert 'href="docs"' in html
    # Zero hardcoded stale counts, zero internal codenames, zero institution naming
    for forbidden in ("58,932", "213,800", "V32", "V33", "v33_remote", "中检院", "叶老师", "NIFDC", "Prof. Ye"):
        assert forbidden not in html, f"Found forbidden UI string: {forbidden}"
    # Bilingual toggle, Dark/Light theme toggle, Tasks Center tab, Return Context Bar & v1.1 visual showcase hooks
    assert 'data-tab="tasks"' in html
    assert 'id="btn-theme-toggle"' in html
    assert 'data-lang="zh"' in html and 'data-lang="en"' in html
    assert 'id="detail-context-bar"' in html
    assert 'id="hero-mol-canvas"' in html
    assert 'id="admet-map-host"' in html
    assert 'id="arch-flow-host"' in html
    assert 'id="landing-case17-chart"' in html
    assert 'id="svg-view-toggle"' in html
    assert 'card_semaglutide.png?v=1.1' in html


def test_v11_scientific_static_assets_served():
    for png_name in ("card_semaglutide.png", "card_cyclosporine.png", "card_micafungin.png", "card_magainin.png"):
        r = client.get(f"/static/assets/{png_name}")
        assert r.status_code == 200, f"Missing static asset {png_name}"
        assert r.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(r.content) < 300 * 1024, f"{png_name} exceeds 300 KB ({len(r.content)} bytes)"

    r_map = client.get("/static/assets/admet_human_map.svg")
    assert r_map.status_code == 200
    assert 'viewBox="0 0 640 520"' in r_map.text
    for hs in ("bbb", "gut", "pk", "clearance", "safety", "micro"):
        assert f'data-hotspot="{hs}"' in r_map.text

    r_arch = client.get("/static/assets/arch_4fm_flow.svg")
    assert r_arch.status_code == 200
    assert 'viewBox="0 0 960 230"' in r_arch.text
    assert "flow-pulse" in r_arch.text


def test_async_task_center_lifecycle_and_exports():
    """Verify POST /api/v1/jobs -> polling -> GET detail -> export summary/full CSV & JSON -> delete."""
    import time

    headers = {"X-Client-Id": "pytest_task_session_01"}
    r_create = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "title": "17 Cases Mini Test",
            "raw_text": "name,query\nSemaglutide,Semaglutide\nMicafungin,Micafungin\nCyclosporine A,Cyclosporine A\n",
            "include_tier2": True,
        },
    )
    assert r_create.status_code == 200
    job = r_create.json()
    job_id = job["job_id"]
    assert job["total"] == 3
    assert job["title"] == "17 Cases Mini Test"

    # Wait briefly for background worker thread to complete 3 offline-fallback items
    detail = None
    for _ in range(40):
        r_det = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert r_det.status_code == 200
        detail = r_det.json()
        if detail["status"] in ("completed", "partial", "failed"):
            break
        time.sleep(0.1)

    assert detail is not None and detail["status"] == "completed"
    assert detail["succeeded"] == 3 and detail["failed"] == 0
    assert len(detail["results"]) == 3
    # Verify rich per-molecule fields required for instant 0ms Batch/Task -> Detail drill-down
    first = detail["results"][0]
    assert first["physchem"] is not None and first["physchem"]["mw"] > 4000
    assert first["drug_reference"] is not None
    assert len(first["predictions"]) >= 46
    assert detail["summary"]["n_acylation_anchors"] == 1

    # Client isolation: another client must NOT see this job
    r_other = client.get(f"/api/v1/jobs/{job_id}", headers={"X-Client-Id": "other_client_99"})
    assert r_other.status_code == 404

    # List endpoint returns our job
    r_list = client.get("/api/v1/jobs", headers=headers)
    assert r_list.status_code == 200
    assert any(j["job_id"] == job_id for j in r_list.json()["jobs"])

    # Export summary CSV, full 46-endpoint wide CSV, and JSON
    r_sum_csv = client.get(f"/api/v1/jobs/{job_id}/export?format=summary_csv", headers=headers)
    assert r_sum_csv.status_code == 200 and "Semaglutide" in r_sum_csv.text
    r_full_csv = client.get(f"/api/v1/jobs/{job_id}/export?format=full_csv", headers=headers)
    assert r_full_csv.status_code == 200 and "PAMPA_value" in r_full_csv.text and "MIC_E_coli_value" in r_full_csv.text
    r_json = client.get(f"/api/v1/jobs/{job_id}/export?format=json", headers=headers)
    assert r_json.status_code == 200 and r_json.json()["job_id"] == job_id

    # Clean up test task
    r_del = client.delete(f"/api/v1/jobs/{job_id}", headers=headers)
    assert r_del.status_code == 200 and r_del.json()["deleted"] is True


def test_examples_have_no_institution_or_internal_version_labels():
    res = client.get("/api/v1/examples")
    assert res.status_code == 200
    raw = res.text
    for forbidden in ("中检院", "叶老师", "NIFDC", "Prof. Ye", "V32", "V33"):
        assert forbidden not in raw, f"Found forbidden label in /api/v1/examples: {forbidden}"


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


def test_mondrian_conformal_manifest_and_live_atom_attribution():
    """Verify Route A zero-leakage Mondrian AD-EB conformal manifest + synced overall_summary + 2D Graph Surrogate E-GAT API."""
    from collections import Counter

    manifest_path = Path(__file__).resolve().parents[2] / "bundle" / "v33_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "v33.2"
    reg_eps = [ep for ep, v in manifest["endpoints"].items() if v["task_type"] == "regression"]
    assert len(reg_eps) == 40
    for ep in reg_eps:
        for tr in ("dual_modality", "smiles_only"):
            conf = manifest["endpoints"][ep][tr]["conformal"]
            assert conf["mode"] == "mondrian_ad_eb"
            assert set(conf["q80_by_tier"].keys()) == {"In-Domain", "Borderline", "OOD"}
    # Route A strict zero-holdout-peeking & zero train-serve mismatch: no unified_mt46 loaded
    assert manifest["serving_inventory"].get("mt46_checkpoint_refs") == []
    assert manifest["serving_inventory"].get("mt46_promoted_endpoints") == []
    # Top-level overall_summary winning-family counts must match 46 endpoints 100%
    actual_dual = dict(Counter(v["dual_modality"]["winning_family"] for v in manifest["endpoints"].values()))
    assert manifest["overall_summary"]["dual_modality_winning_family_counts"] == actual_dual
    # Verify MBC_E_faecalis UI metadata is present (no English fallback)
    ep_res = client.get("/api/v1/endpoints").json()
    mbc_ef = next(e for e in ep_res["endpoints"] if e["endpoint"] == "MBC_E_faecalis")
    assert "粪肠球菌" in mbc_ef["display_zh"]

    # Test POST /api/v1/atom-attribution if checkpoint is available in workspace
    res = client.post(
        "/api/v1/atom-attribution",
        json={"query": "Cyclosporine A", "endpoint": "PAMPA", "width": 760, "height": 460},
    )
    if res.status_code == 200:
        d = res.json()
        assert "<svg" in d["svg"]
        assert "2D Graph Surrogate" in d["svg"]
        assert "Surrogate:" in d["svg"]
        assert d["endpoint"] == "PAMPA"
        assert d["n_atoms"] == 85
        assert len(d["top_atoms"]) >= 5
        assert len(d["available_endpoints"]) == 8
        # Also verify /predict auto-attaches atom_attribution_svg when include_svg=True (never unconditional half_life_reg default)
        rp = client.post("/api/v1/predict", json={"query": "Semaglutide", "include_svg": True})
        assert rp.status_code == 200
        dp = rp.json()
        assert dp["atom_attribution_svg"] and "2D Graph Surrogate" in dp["atom_attribution_svg"]
        assert dp["atom_attribution_endpoint"] == "PAMPA"
        assert len(dp["atom_attribution_endpoints"]) == 8
    else:
        assert res.status_code == 503


def test_audit_regressions_uq_reliability_strict_remote_422_413_and_csv_source(monkeypatch: pytest.MonkeyPatch):
    """Regression suite covering independent audit findings N3, N5, N8, N9, N10, N15."""
    from concurrent.futures import ThreadPoolExecutor

    # N8: Unknown atom-attribution endpoint must return 422 (never silently fall back to PAMPA with 200)
    r_bad_ep = client.post(
        "/api/v1/atom-attribution",
        json={"query": "Semaglutide", "endpoint": "NOT_A_REAL_ENDPOINT"},
    )
    assert r_bad_ep.status_code == 422

    # N9: /api/v1/batch/upload must reject payloads > 2 MB with HTTP 413
    oversized = b"name,query\n" + (b"Semaglutide,Semaglutide\n" * 100_000)
    assert len(oversized) > 2 * 1024 * 1024
    r_big = client.post(
        "/api/v1/batch/upload",
        files={"file": ("huge.csv", oversized, "text/csv")},
        data={"include_tier2": "false"},
    )
    assert r_big.status_code == 413

    # N3: Manifest & /api/v1/endpoints expose 3-tier uq_reliability (low / moderate / calibrated)
    ep_data = client.get("/api/v1/endpoints").json()
    by_ep = {e["endpoint"]: e for e in ep_data["endpoints"]}
    assert by_ep["PPB"]["uq_reliability"] == "low"
    assert by_ep["Cmax"]["uq_reliability"] == "moderate"
    assert by_ep["PAMPA"]["uq_reliability"] == "calibrated"
    assert by_ep["hemolysis_cls"]["uq_reliability"] is None

    # N5 + CSV formula injection + error row prediction_source + anonymous session isolation
    headers = {"X-Client-Id": "pytest_audit_n5_01"}
    r_job = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "title": "CSV Source Check",
            "raw_text": "name,query\n=SUM(A1:A2),Semaglutide\nBadMol,???invalid###\n",
            "include_tier2": False,
        },
    )
    assert r_job.status_code == 200
    jid = r_job.json()["job_id"]
    import time
    for _ in range(30):
        det = client.get(f"/api/v1/jobs/{jid}", headers=headers).json()
        if det["status"] in ("completed", "partial", "failed"):
            break
        time.sleep(0.1)
    r_sum = client.get(f"/api/v1/jobs/{jid}/export?format=summary_csv", headers=headers)
    assert r_sum.status_code == 200 and "prediction_source" in r_sum.text.splitlines()[0]
    assert "'=SUM(A1:A2)" in r_sum.text
    assert ",error,error," in r_sum.text
    r_wide = client.get(f"/api/v1/jobs/{jid}/export?format=full_csv", headers=headers)
    wide_hdr = r_wide.text.splitlines()[0]
    assert r_wide.status_code == 200 and "prediction_source" in wide_hdr and "PAMPA_ci80_low" in wide_hdr and "PAMPA_uq_reliability" in wide_hdr
    client.delete(f"/api/v1/jobs/{jid}", headers=headers)

    # Anonymous requests without X-Client-Id must NOT see each other's jobs
    r_anon1 = client.post("/api/v1/jobs", json={"title": "Anon Secret", "raw_text": "Semaglutide\n", "include_tier2": False})
    assert r_anon1.status_code == 200
    r_anon2 = client.get("/api/v1/jobs")
    assert r_anon2.status_code == 200 and r_anon2.json()["total"] == 0

    # N5: When strict_remote=True and remote engine is offline, /predict must return 503 (never fabricated values)
    from v33_upgrade.web import app as web_app_mod
    monkeypatch.setattr(web_app_mod.adapter, "strict_remote", True)
    monkeypatch.setattr(web_app_mod.adapter, "mode", "v33_offline_fallback")
    monkeypatch.setattr(web_app_mod.adapter, "remote_url", None)
    r_503 = client.post("/api/v1/predict", json={"query": "Semaglutide", "include_svg": False})
    assert r_503.status_code == 503

    # N15: Redacted health must hide filesystem paths and loopback URL while preserving booleans
    monkeypatch.setattr(web_app_mod, "REDACT_HEALTH_PATHS", True)
    h_red = client.get("/api/v1/health").json()
    assert h_red["manifest_loaded"] is True
    assert h_red["bundle_path"] is None and h_red["manifest_path"] is None and h_red["v33_remote_url"] is None

    # Thread-safety check: 8 concurrent atom-attribution calls return bit-identical values
    def _one_attr(_i: int):
        rr = client.post("/api/v1/atom-attribution", json={"query": "Cyclosporine A", "endpoint": "PAMPA"})
        return rr.status_code, (rr.json()["gnn_prediction"]["value"] if rr.status_code == 200 else None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(_one_attr, range(8)))
    if outs[0][0] == 200:
        assert all(code == 200 for code, _ in outs)
        assert len({val for _, val in outs}) == 1
