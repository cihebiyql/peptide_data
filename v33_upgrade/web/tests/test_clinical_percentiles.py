"""Unit tests for clinical PK index hygiene (P0-3) and approved-peptide clinical percentiles (P0-2)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from v33_upgrade.expansion_v335.build_public_domain_clinical_index import (
    FORBIDDEN_SUBSTRINGS,
    build_clean_index,
)
from v33_upgrade.web.clinical_percentile_engine import (
    build_clinical_percentiles_data,
    compute_clinical_percentile,
    get_percentile_catalog_summary,
    reload_clinical_percentiles,
)

WEB_DIR = Path(__file__).resolve().parents[1]
UPGRADE_DIR = WEB_DIR.parent
CLINICAL_INDEX_PATH = WEB_DIR / "clinical_evidence_index.json"
CLINICAL_PERCENTILES_PATH = WEB_DIR / "clinical_percentiles.json"
MANIFEST_PATH = UPGRADE_DIR / "bundle" / "v33_manifest.json"


# ---------------------------------------------------------------------------
# P0-3: Clinical Evidence Index Deduplication, Exact SetID URLs & License Hygiene
# ---------------------------------------------------------------------------


def test_clinical_index_deduplication_and_merged_aliases():
    data = json.loads(CLINICAL_INDEX_PATH.read_text(encoding="utf-8"))
    products = data["products"]
    lookup = data["lookup"]

    # Duplicate keys must be gone
    assert "fda_glucagon_recombinant" not in products
    assert "fda_vasopressin_or_argipressin" not in products
    assert "fda_glucagon" in products
    assert "fda_vasopressin" in products

    # Merged glucagon retains verified setid, all 13 unique PK records, and combined aliases
    gluc = products["fda_glucagon"]
    assert gluc["dailymed_setid"] == "3d8c92cf-9263-4126-8a5b-4cd761e31bc1"
    assert gluc["n_pk_records"] == len(gluc["pk_records"]) == 13
    gluc_aliases_low = {a.lower() for a in gluc["aliases"]}
    for expected_alias in ("glucagon", "glucagon recombinant", "baqsimi", "glucagen", "gvoke", "ogluo"):
        assert expected_alias in gluc_aliases_low
        assert lookup["by_name"][expected_alias] == "fda_glucagon"

    # Merged vasopressin retains verified setid, both unique PK records, and argipressin aliases
    vaso = products["fda_vasopressin"]
    assert vaso["dailymed_setid"] == "b5b2c367-6b7e-40be-a296-c4632f3e783f"
    assert vaso["n_pk_records"] == len(vaso["pk_records"]) == 2
    vaso_aliases_low = {a.lower() for a in vaso["aliases"]}
    for expected_alias in ("vasopressin", "pitressin", "argipressin", "vasopressin (or argipressin)"):
        assert expected_alias in vaso_aliases_low
        assert lookup["by_name"][expected_alias] == "fda_vasopressin"

    # Zero dangling lookup keys anywhere
    for map_name in ("by_name", "by_seq", "by_ik14"):
        for k, pid in lookup[map_name].items():
            assert pid in products, f"Dangling {map_name}[{k!r}] -> {pid!r}"


def test_clinical_index_exact_stats_and_idempotent_builder():
    data = json.loads(CLINICAL_INDEX_PATH.read_text(encoding="utf-8"))
    products = data["products"]
    lookup = data["lookup"]
    stats = data["stats"]

    assert stats["n_products"] == len(products) == 57
    assert stats["n_products_with_pk"] == sum(1 for p in products.values() if p["pk_records"]) == 57
    assert stats["n_dailymed_setid_verified"] == sum(1 for p in products.values() if p.get("dailymed_setid")) == 53
    assert stats["n_pk_records_indexed"] == sum(len(p["pk_records"]) for p in products.values()) == 413
    assert stats["n_seq_keys"] == len(lookup["by_seq"])
    assert stats["n_ik14_keys"] == len(lookup["by_ik14"])
    assert stats["n_name_keys"] == len(lookup["by_name"])

    for pid, prod in products.items():
        assert prod["product_id"] == pid
        assert prod["n_pk_records"] == len(prod["pk_records"])

    # Builder is deterministic and idempotent
    rebuilt = build_clean_index(data, refresh_setids=False)
    assert rebuilt == data


def test_clinical_index_exact_setid_urls_and_honest_url_types():
    data = json.loads(CLINICAL_INDEX_PATH.read_text(encoding="utf-8"))
    products = data["products"]
    valid_url_types = {"dailymed_spl_setid", "fda_drugs_portal_search", "dailymed_search"}

    n_setid_urls = 0
    n_fda_portal = 0
    n_dm_search = 0

    for pid, prod in products.items():
        setid = prod.get("dailymed_setid")
        for rec in prod["pk_records"]:
            url = rec["source_url"]
            utype = rec.get("source_url_type")
            assert utype in valid_url_types, f"{pid} invalid source_url_type={utype!r}"

            if setid:
                expected_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
                assert url == expected_url, f"{pid} record still has non-setid URL: {url}"
                assert "search.cfm" not in url
                assert utype == "dailymed_spl_setid"
                n_setid_urls += 1
            else:
                if "Review" in rec["source"] or "Drugs@FDA" in rec["source"]:
                    assert utype == "fda_drugs_portal_search"
                    assert "accessdata.fda.gov" in url
                    n_fda_portal += 1
                else:
                    assert utype == "dailymed_search"
                    assert "dailymed.nlm.nih.gov/dailymed/search.cfm" in url
                    n_dm_search += 1

    assert n_setid_urls == data["stats"]["n_pk_records_exact_setid_url"] == 399
    assert n_fda_portal == data["stats"]["n_pk_records_fda_portal_url"] == 4
    assert n_dm_search == data["stats"]["n_pk_records_dailymed_search_url"] == 10


def test_zero_forbidden_strings_in_clinical_json_assets():
    for path in (CLINICAL_INDEX_PATH, CLINICAL_PERCENTILES_PATH):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            assert not re.search(re.escape(forbidden), text, re.IGNORECASE), (
                f"Forbidden string {forbidden!r} found in {path.name}"
            )


# ---------------------------------------------------------------------------
# P0-2: Approved-Peptide Clinical Percentiles & Engine
# ---------------------------------------------------------------------------


def test_clinical_percentiles_json_covers_all_47_manifest_endpoints():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pct_db = json.loads(CLINICAL_PERCENTILES_PATH.read_text(encoding="utf-8"))
    m_eps = manifest["endpoints"]
    p_eps = pct_db["endpoints"]

    assert set(p_eps.keys()) == set(m_eps.keys())
    assert len(p_eps) == 47

    expected_approved = {
        "half_life_reg",
        "PPB",
        "F",
        "CL_absolute",
        "CL_per_kg",
        "Vd_absolute",
        "Vd_per_kg",
        "Cmax",
    }
    for ep, m_info in m_eps.items():
        p_info = p_eps[ep]
        assert p_info["unit"] == m_info["unit"]
        assert p_info["linear_unit"] == m_info["linear_unit"]
        assert p_info["log10_space"] == bool(m_info.get("log10_space", False))
        if ep in expected_approved:
            assert p_info["reference_group"] == "approved_clinical_peptides"
            assert p_info["n_reference"] >= 14
            samples = p_info["samples"]
            assert len(samples) == p_info["n_reference"]
            # Must be strictly sorted by compare_value
            comp = [s["compare_value"] for s in samples]
            assert comp == sorted(comp)
        else:
            assert p_info["reference_group"] == "curated_peptide_panel"
            assert p_info["n_reference"] > 0

        # Regression check: preserve significant digits for ALL log10_space empirical samples (no cm/s underflow)
        if p_info["log10_space"] and p_info.get("samples"):
            for s in p_info["samples"]:
                assert s["value"] > 0.0, f"{ep} sample {s['name']} underflowed to {s['value']}"
                assert abs(math.log10(s["value"]) - s["compare_value"]) < 1e-4, (
                    f"{ep} sample {s['name']} precision drift: log10({s['value']}) != {s['compare_value']}"
                )

    # Verify Ziconotide systemic IV clearance (16.2 L/h) & Vd (30.46 L) rather than CSF values
    zico_cl = next(s for s in p_eps["CL_absolute"]["samples"] if s["name"] == "Ziconotide")
    assert zico_cl["value"] == pytest.approx(16.2, rel=0.05)
    zico_vd = next(s for s in p_eps["Vd_absolute"]["samples"] if s["name"] == "Ziconotide")
    assert zico_vd["value"] == pytest.approx(30.46, rel=0.05)

    # Reproducibility check
    rebuilt = build_clinical_percentiles_data(CLINICAL_INDEX_PATH, MANIFEST_PATH)
    assert rebuilt == pct_db


def test_compute_clinical_percentile_half_life_clinical_anchors():
    reload_clinical_percentiles(CLINICAL_PERCENTILES_PATH)

    # Semaglutide (~168 h): near top of approved peptides
    sema = compute_clinical_percentile("half_life_reg", 168.0)
    assert sema is not None
    assert sema["reference_group"] == "approved_clinical_peptides"
    assert sema["n_reference"] == 27
    assert 90.0 <= sema["percentile"] <= 99.0
    assert any(ex["name"] == "Semaglutide" for ex in sema["nearest_approved_examples"])
    assert "百分位" in sema["summary_zh"]
    assert "percentile" in sema["summary_en"]

    # Octreotide (~1.8 h): near middle of approved peptides
    octr = compute_clinical_percentile("half_life_reg", 1.8)
    assert octr is not None
    assert 20.0 <= octr["percentile"] <= 55.0
    assert any(ex["name"] == "Octreotide" for ex in octr["nearest_approved_examples"])

    # Oxytocin (~0.067 h): near bottom of approved peptides
    oxy = compute_clinical_percentile("half_life_reg", 0.067)
    assert oxy is not None
    assert 0.0 <= oxy["percentile"] <= 10.0
    assert any(ex["name"] == "Oxytocin" for ex in oxy["nearest_approved_examples"])

    # Strict monotonicity across the clinical half-life spectrum
    vals = [0.02, 0.067, 0.5, 1.8, 4.6, 13.0, 25.0, 168.0, 250.0]
    pcts = [compute_clinical_percentile("half_life_reg", v)["percentile"] for v in vals]
    assert pcts == sorted(pcts)
    assert pcts[0] < pcts[3] < pcts[5] < pcts[7] <= pcts[8]


def test_compute_clinical_percentile_log10_and_linear_equivalence():
    # log10_space=True must match linear log10_space=False for all log10 endpoints
    pairs = [
        ("half_life_reg", 168.0),
        ("half_life_reg", 1.8),
        ("CL_absolute", 0.045),
        ("CL_per_kg", 1.28),
        ("Vd_absolute", 12.5),
        ("Vd_per_kg", 0.25),
        ("Cmax", 0.5),
        ("PAMPA", 1e-6),
        ("Caco2", 2e-7),
        ("hemolysis_HC50", 120.0),
        ("MIC_E_coli", 8.0),
    ]
    for ep, lin_val in pairs:
        r_lin = compute_clinical_percentile(ep, lin_val, log10_space=False)
        r_log = compute_clinical_percentile(ep, math.log10(lin_val), log10_space=True)
        assert r_lin is not None and r_log is not None
        assert r_lin["percentile"] == pytest.approx(r_log["percentile"], abs=0.1)
        assert r_lin["reference_group"] == r_log["reference_group"]
        assert r_lin["n_reference"] == r_log["n_reference"]


def test_compute_clinical_percentile_all_pk_and_panel_endpoints_schema():
    required_keys = {
        "percentile",
        "reference_group",
        "n_reference",
        "summary_zh",
        "summary_en",
        "nearest_approved_examples",
    }
    cases = [
        ("PPB", 0.99, "approved_clinical_peptides", 85.0, 100.0),
        ("PPB", 0.0, "approved_clinical_peptides", 0.0, 10.0),
        ("F", 0.89, "approved_clinical_peptides", 55.0, 90.0),
        ("CL_absolute", 0.045, "approved_clinical_peptides", 0.0, 10.0),
        ("CL_absolute", 170.0, "approved_clinical_peptides", 90.0, 100.0),
        ("Vd_absolute", 12.5, "approved_clinical_peptides", 10.0, 40.0),
        ("Cmax", 1.0, "approved_clinical_peptides", 65.0, 95.0),
        ("PAMPA", 1e-6, "curated_peptide_panel", 70.0, 100.0),
        ("LogD7.4", 1.2, "curated_peptide_panel", 30.0, 80.0),
        ("hemolysis_cls", 0.90, "curated_peptide_panel", 95.0, 100.0),
        ("MIC_E_coli", 10.0, "curated_peptide_panel", 40.0, 60.0),
    ]
    for ep, val, exp_grp, lo, hi in cases:
        res = compute_clinical_percentile(ep, val)
        assert res is not None
        assert set(res.keys()) == required_keys
        assert res["reference_group"] == exp_grp
        assert isinstance(res["percentile"], float)
        assert lo <= res["percentile"] <= hi, f"{ep}({val}) got P{res['percentile']} not in [{lo}, {hi}]"
        assert isinstance(res["n_reference"], int) and res["n_reference"] > 0
        assert isinstance(res["summary_zh"], str) and len(res["summary_zh"]) > 5
        assert isinstance(res["summary_en"], str) and len(res["summary_en"]) > 5
        assert isinstance(res["nearest_approved_examples"], list)
        for ex in res["nearest_approved_examples"]:
            assert {"name", "value", "value_display", "unit", "route", "source"}.issubset(ex.keys())


def test_compute_clinical_percentile_edge_cases_and_summary():
    assert compute_clinical_percentile("half_life_reg", None) is None
    assert compute_clinical_percentile("half_life_reg", float("nan")) is None
    assert compute_clinical_percentile("half_life_reg", float("inf")) is None
    assert compute_clinical_percentile("UNKNOWN_ENDPOINT_XYZ", 1.0) is None

    # Non-positive linear value on log10 endpoint must not raise ValueError
    r_zero = compute_clinical_percentile("half_life_reg", 0.0, log10_space=False)
    assert r_zero is not None and r_zero["percentile"] == 0.0
    r_neg = compute_clinical_percentile("CL_absolute", -1.0, log10_space=False)
    assert r_neg is not None and r_neg["percentile"] == 0.0

    # Catalog summary reports all 47 endpoints cleanly
    cat = get_percentile_catalog_summary()
    assert cat["stats"]["n_endpoints"] == 47
    assert cat["stats"]["n_approved_clinical_endpoints"] == 8
    assert cat["stats"]["n_curated_panel_endpoints"] == 39
