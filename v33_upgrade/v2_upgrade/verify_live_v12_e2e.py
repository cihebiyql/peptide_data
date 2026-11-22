#!/usr/bin/env python3
"""Live Public HTTPS Playwright + API End-to-End Verification for Peptide OmniPanel v1.2 / v33.2."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "https://lmmd.ecust.edu.cn/peptide-omnipanel/"
OUT_DIR = Path(r"D:\work\中检院\v33_upgrade\reports\screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FORBIDDEN = ("V32", "V33", "V33.1", "v33_remote", "中检院", "叶老师", "NIFDC", "Prof. Ye")


def api_json(path: str, payload: dict | None = None) -> dict:
    url = BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200, f"{url} -> HTTP {resp.status}"
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    # 1. Live HTTPS API verification
    h = api_json("api/v1/health")
    print("[API] /health:", h["version"], h["adapter_mode"], "remote_healthy=", h["v33_remote_healthy"], "eps=", h["endpoints_total"])
    assert h["ok"] is True
    assert h["version"] == "1.2.0"
    assert h["adapter_mode"] == "v33_remote"
    assert h["v33_remote_healthy"] is True
    assert h["endpoints_total"] == 46

    # Check live Semaglutide prediction (smiles_only track: efflux_ratio + MBC_S_aureus MT46 + Mondrian + Atom Attribution SVG)
    p_sema = api_json("api/v1/predict", {"query": "Semaglutide", "include_tier2": True, "include_svg": True})
    assert p_sema["adapter_mode"] == "v33_remote"
    assert p_sema["atom_attribution_svg"] and "<svg" in p_sema["atom_attribution_svg"]
    assert len(p_sema["atom_attribution_endpoints"]) == 8
    by_ep = {p["endpoint"]: p for p in p_sema["predictions"]}
    assert "mt46" in (by_ep["efflux_ratio"]["weights"] or {}), f"MT46 missing on efflux_ratio (smiles_only): {by_ep['efflux_ratio']['weights']}"
    assert "mt46" in (by_ep["MBC_S_aureus"]["weights"] or {}), f"MT46 missing on MBC_S_aureus (smiles_only): {by_ep['MBC_S_aureus']['weights']}"
    print(
        f"[API] Semaglutide ({p_sema['modality_track']}) PASS: efflux_ratio={by_ep['efflux_ratio']['value_display']} "
        f"(weights={by_ep['efflux_ratio']['weights']}), "
        f"MBC_S_aureus weights={by_ep['MBC_S_aureus']['weights']}, "
        f"attr_ep={p_sema['atom_attribution_endpoint']}"
    )

    # Check live Exenatide prediction (dual_modality track: efflux_ratio + cytotoxicity_CC50 + hemolysis_MHC + MIC_Salmonella + MIC_B_subtilis MT46!)
    p_exen = api_json("api/v1/predict", {"query": "Exenatide", "include_tier2": True, "include_svg": False})
    ex_ep = {p["endpoint"]: p for p in p_exen["predictions"]}
    for mt_ep in ("efflux_ratio", "cytotoxicity_CC50", "hemolysis_MHC", "MIC_Salmonella", "MIC_B_subtilis"):
        assert "mt46" in (ex_ep[mt_ep]["weights"] or {}), f"MT46 missing on {mt_ep} (dual_modality): {ex_ep[mt_ep]['weights']}"
    print(
        f"[API] Exenatide ({p_exen['modality_track']}) PASS: 5/5 dual_modality MT46 heads verified! "
        f"(CC50={ex_ep['cytotoxicity_CC50']['value_display']}, MHC={ex_ep['hemolysis_MHC']['value_display']}, "
        f"efflux={ex_ep['efflux_ratio']['value_display']})"
    )

    # Check live POST /api/v1/atom-attribution across multiple endpoints (1st cold + 2nd cached)
    a1 = api_json("api/v1/atom-attribution", {"query": "Cyclosporine A", "endpoint": "PAMPA"})
    a2 = api_json("api/v1/atom-attribution", {"query": "Cyclosporine A", "endpoint": "hemolysis_cls"})
    assert "<svg" in a1["svg"] and a1["n_atoms"] == 85
    assert "<svg" in a2["svg"] and a2["endpoint"] == "hemolysis_cls"
    print(f"[API] /atom-attribution PASS: PAMPA={a1['latency_ms']}ms (cache={a1['cache_hit']}), hemolysis_cls={a2['latency_ms']}ms (cache={a2['cache_hit']})")

    # Check Mondrian OOD widening on a novel non-peptide-like OOD query
    p_ood = api_json("api/v1/predict", {"smiles": "CC(=O)NC(CC1=CC=C(F)C(F)=C1)C(=O)NC(C(F)(F)F)C(=O)O", "include_tier2": False, "include_svg": False})
    ood_by_ep = {p["endpoint"]: p for p in p_ood["predictions"]}
    print(
        f"[API] OOD Mondrian check: PAMPA ad={ood_by_ep['PAMPA']['ad_status']} "
        f"(max_tc={ood_by_ep['PAMPA']['max_tanimoto']}), CI80={ood_by_ep['PAMPA']['ci80_display']}"
    )

    # 2. Live Browser Visual & Functional Verification (Playwright Chromium 1440x960)
    console_errs: list[str] = []
    page_errs: list[str] = []
    http_errs: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 960}, locale="zh-CN")
        page = ctx.new_page()
        page.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errs.append(str(e)))
        page.on("response", lambda r: http_errs.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        page.goto(BASE_URL, wait_until="networkidle", timeout=45000)
        page.wait_for_selector(".brand-ver", timeout=15000)
        ver_txt = page.locator(".brand-ver").inner_text().strip()
        assert ver_txt == "v1.2", f"Expected brand-ver v1.2, got {ver_txt}"

        # Check forbidden visible strings in ZH
        vis_zh = page.evaluate("document.body.innerText")
        for f in FORBIDDEN:
            assert f not in vis_zh, f"Forbidden string {f!r} found in ZH landing!"

        # Run Semaglutide via modality showcase card -> click Atom Attribution toggle!
        page.locator(".modality-card").first.click()
        page.wait_for_selector("#predict-results:not(.hidden)", timeout=30000)
        page.wait_for_function("!document.getElementById('btn-svg-attr').disabled", timeout=20000)
        page.locator("#btn-svg-attr").click()
        page.wait_for_selector("#svg-attr-bar:not(.hidden)", timeout=10000)
        page.wait_for_selector("#mol-svg-stage svg", timeout=10000)
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT_DIR / "v12_01_atom_attr_semaglutide.png"), full_page=False)
        print("[UI] Captured v12_01_atom_attr_semaglutide.png")

        # Switch attribution endpoint pill to hemolysis_cls on Semaglutide
        page.locator("#svg-attr-bar .svg-attr-pill[data-attr-ep='hemolysis_cls']").click()
        page.wait_for_timeout(600)
        assert page.locator("#svg-attr-bar .svg-attr-pill[data-attr-ep='hemolysis_cls']").evaluate("el => el.classList.contains('active')")

        # Predict Cyclosporine A -> toggle Atom Attribution (PAMPA) + verify SVG!
        page.fill("#smart-query-input", "Cyclosporine A")
        page.click("#btn-predict")
        page.wait_for_function("document.getElementById('res-mol-name').innerText.includes('Cyclosporine A')", timeout=30000)
        page.wait_for_selector("#svg-attr-bar:not(.hidden)", timeout=10000)
        page.locator("#svg-attr-bar .svg-attr-pill[data-attr-ep='PAMPA']").click()
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT_DIR / "v12_02_atom_attr_cyclosporine.png"), full_page=False)
        print("[UI] Captured v12_02_atom_attr_cyclosporine.png")

        # Switch to EN + Dark mode and verify zero forbidden strings & clean dark atom attribution
        page.locator(".lang-btn[data-lang='en']").click()
        page.locator("#btn-theme-toggle").click()
        page.wait_for_timeout(500)
        vis_en = page.evaluate("document.body.innerText")
        for f in FORBIDDEN:
            assert f not in vis_en, f"Forbidden string {f!r} found in EN detail!"
        page.screenshot(path=str(OUT_DIR / "v12_03_atom_attr_en_dark.png"), full_page=False)
        print("[UI] Captured v12_03_atom_attr_en_dark.png")

        browser.close()

    print(f"[UI] Errors: console={len(console_errs)}, page={len(page_errs)}, http>=400={len(http_errs)}")
    if console_errs:
        print("  console:", console_errs)
    if page_errs:
        print("  page:", page_errs)
    if http_errs:
        print("  http:", http_errs)
    assert len(console_errs) == 0 and len(page_errs) == 0 and len(http_errs) == 0
    print("[PASS] ALL LIVE HTTPS v1.2 / v33.2 END-TO-END CHECKS PASSED!")


if __name__ == "__main__":
    main()
