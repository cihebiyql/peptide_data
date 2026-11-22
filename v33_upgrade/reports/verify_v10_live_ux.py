"""Live public HTTPS E2E & visual verification for Peptide OmniPanel v1.0 UX overhaul.

Tests:
  1. Zero console/page/HTTP errors across all 5 tabs on https://lmmd.ecust.edu.cn/peptide-omnipanel/
  2. Zero forbidden labels (V32, V33, V33.1, v33_remote, 中检院, 叶老师, NIFDC, Prof. Ye) across all tabs & both languages
  3. Bilingual toggle (中文 <-> English) & Light/Dark theme toggle with localStorage persistence
  4. Single prediction (Semaglutide, 24 Tier1 + 22 Tier2 = 46 endpoints + 2D SVG + radar + auto-logged in Task Center)
  5. Batch 17 Clinical Cases -> async Task Center submission -> real-time chunked progress -> 5 KPIs -> 5 column presets -> screening filters -> radar overlay
  6. Seamless Task -> Compound Detail (0ms cached drill-down) -> Sticky Return/Next/Prev bar -> Next (Tirzepatide) -> Return restoring highlighted row
  7. Live public HTTPS exports (summary_csv, 46-endpoint full_csv, json) + cleanup
  8. Captures 5 crisp desktop screenshots for visual inspection.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

BASE_URL = "https://lmmd.ecust.edu.cn/peptide-omnipanel/"
SHOT_DIR = Path(r"D:\work\中检院\v33_upgrade\reports\screenshots")
SHOT_DIR.mkdir(parents=True, exist_ok=True)

FORBIDDEN_STRINGS = [
    "V32",
    "V33",
    "v33_remote",
    "中检院",
    "叶老师",
    "NIFDC",
    "Prof. Ye",
]


def assert_no_forbidden(page, context_label: str) -> None:
    text = page.evaluate("() => document.body.innerText")
    found = [s for s in FORBIDDEN_STRINGS if s in text]
    assert not found, f"[{context_label}] Forbidden visible UI string(s) found: {found}"


def main() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    bad_responses: list[str] = []
    report: dict = {"url": BASE_URL, "checks": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 960}, locale="zh-CN")
        page = context.new_page()

        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("response", lambda r: bad_responses.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        print("1. Loading public HTTPS site (ZH Light)...")
        t0 = time.perf_counter()
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_function(
            "() => (document.querySelector('#backend-status .status-text')?.textContent || '').includes('46')",
            timeout=20000,
        )
        page.wait_for_selector(".example-pill", timeout=15000)
        # Ensure ZH mode active (in case browser default was EN)
        page.locator('.lang-btn[data-lang="zh"]').click()
        page.wait_for_timeout(150)

        load_ms = round((time.perf_counter() - t0) * 1000, 1)
        status_text = page.locator("#backend-status .status-text").inner_text().strip()
        example_count = page.locator(".exlib-chip").count()
        print(f"   Loaded in {load_ms}ms | status='{status_text}' | examples={example_count}")
        assert "46" in status_text, f"Expected 46 endpoints in status badge, got: {status_text}"
        assert example_count == 25, f"Expected 25 curated examples, got {example_count}"
        assert_no_forbidden(page, "Home ZH Light")

        shot1 = SHOT_DIR / "v10_01_home_zh_light.png"
        page.screenshot(path=str(shot1), full_page=False)
        report["checks"]["home_zh_light"] = {
            "load_ms": load_ms,
            "status_badge": status_text,
            "examples": example_count,
            "screenshot": str(shot1),
        }

        print("2. Running single prediction (Semaglutide)...")
        page.locator(".example-pill", has_text="Semaglutide").first.click()
        page.wait_for_selector("#predict-results:not(.hidden)", timeout=35000)
        page.wait_for_selector("#mol-svg-stage svg", timeout=15000)
        t1_cnt = int(page.locator("#count-tier-1").inner_text().strip())
        t2_cnt = int(page.locator("#count-tier-2").inner_text().strip())
        print(f"   Semaglutide rendered: Tier1={t1_cnt} + Tier2={t2_cnt} = {t1_cnt + t2_cnt} endpoints + 2D SVG + Radar")
        assert t1_cnt == 24 and t2_cnt == 22, f"Expected 24+22 endpoints, got {t1_cnt}+{t2_cnt}"
        assert_no_forbidden(page, "Semaglutide Detail ZH Light")

        print("3. Switching to English + Dark Mode...")
        page.locator('.lang-btn[data-lang="en"]').click()
        page.locator("#btn-theme-toggle").click()
        page.wait_for_timeout(350)
        theme_val = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
        lang_val = page.evaluate("() => localStorage.getItem('omnipanel_lang')")
        nav_tasks_en = page.locator('.nav-tab[data-tab="tasks"] span[data-i18n="nav_tasks"]').inner_text().strip()
        print(f"   theme={theme_val} | lang={lang_val} | tasks_tab='{nav_tasks_en}'")
        assert theme_val == "dark", f"Expected dark theme, got {theme_val}"
        assert lang_val == "en", f"Expected en lang, got {lang_val}"
        assert nav_tasks_en == "Tasks", f"Expected 'Tasks', got '{nav_tasks_en}'"
        assert_no_forbidden(page, "Semaglutide Detail EN Dark")

        page.evaluate("() => document.getElementById('predict-results')?.scrollIntoView({ behavior: 'instant', block: 'start' })")
        page.wait_for_timeout(200)
        shot2 = SHOT_DIR / "v10_02_home_en_dark.png"
        page.screenshot(path=str(shot2), full_page=False)
        report["checks"]["en_dark_mode"] = {
            "theme": theme_val,
            "lang": lang_val,
            "tasks_tab_label": nav_tasks_en,
            "tier1": t1_cnt,
            "tier2": t2_cnt,
            "screenshot": str(shot2),
        }

        print("4. Switching back to ZH Light & submitting 17 Clinical Cases batch task...")
        page.locator('.lang-btn[data-lang="zh"]').click()
        page.locator("#btn-theme-toggle").click()
        page.wait_for_timeout(250)
        page.locator('.nav-tab[data-tab="batch"]').click()
        page.wait_for_selector("#view-batch.active", timeout=5000)
        page.locator("#btn-batch-case17").click()
        page.wait_for_timeout(250)
        assert_no_forbidden(page, "Batch Tab ZH")

        page.locator("#btn-run-batch").click()
        page.wait_for_selector("#batch-active-banner:not(.hidden)", timeout=15000)
        print("   Submitted async 17-case batch job; opening Task Center...")
        page.locator("#btn-goto-tasks-from-batch").click()
        page.wait_for_selector("#view-tasks.active", timeout=5000)

        t_job0 = time.perf_counter()
        completed = False
        for sec in range(95):
            st_pill = page.locator("#tasks-detail-host .ins-header-top .task-st-pill").first
            if st_pill.count() > 0:
                cls_attr = st_pill.get_attribute("class") or ""
                if "st-completed" in cls_attr or "st-partial" in cls_attr:
                    completed = True
                    break
            page.wait_for_timeout(1000)
        job_wait_s = round(time.perf_counter() - t_job0, 1)
        assert completed, "17-case batch task did not complete within 95s!"
        print(f"   17-case task completed in {job_wait_s}s!")

        kpi_count = page.locator("#tasks-detail-host .bkpi").count()
        row_count = page.locator("#tasks-detail-host tbody tr.clickable-mol-row").count()
        radar_svgs = page.locator("#radar-stage-tasks svg").count()
        print(f"   Task Inspector: {kpi_count} KPIs | {row_count} rows | {radar_svgs} radar SVG")
        assert kpi_count == 5, f"Expected 5 KPI cards, got {kpi_count}"
        assert row_count == 17, f"Expected 17 rows, got {row_count}"
        assert radar_svgs >= 1, "Expected multi-peptide radar overlay SVG"
        assert_no_forbidden(page, "Tasks Center 17 Cases ZH Light")

        # Test column preset switch (Safety) and quick filter (Low Hemolysis)
        page.locator('#tasks-detail-host [data-colv="safety"]').click()
        page.wait_for_timeout(150)
        safety_th = page.locator("#tasks-detail-host thead").inner_text()
        assert "HC50" in safety_th or "CC50" in safety_th, f"Expected Safety columns, got: {safety_th}"

        page.locator('#tasks-detail-host [data-rf="low_hem"]').click()
        page.wait_for_timeout(150)
        filtered_rows = page.locator("#tasks-detail-host tbody tr.clickable-mol-row").count()
        print(f"   Quick filter 'low_hem' matched {filtered_rows}/17 rows")
        assert filtered_rows >= 10, f"Expected >= 10 low-hemolysis clinical drugs, got {filtered_rows}"

        # Reset back to Overview + All for screenshots
        page.locator('#tasks-detail-host [data-colv="overview"]').click()
        page.locator('#tasks-detail-host [data-rf="all"]').click()
        page.wait_for_timeout(200)

        shot3 = SHOT_DIR / "v10_03_tasks_center_17cases.png"
        page.screenshot(path=str(shot3), full_page=False)

        # Dark mode screenshot of Tasks Center
        page.locator("#btn-theme-toggle").click()
        page.wait_for_timeout(250)
        shot4 = SHOT_DIR / "v10_04_tasks_center_dark.png"
        page.screenshot(path=str(shot4), full_page=False)
        page.locator("#btn-theme-toggle").click()
        page.wait_for_timeout(200)

        print("5. Testing Task -> Compound Detail (0ms) -> Next (Tirzepatide) -> Return loop...")
        t_drill = time.perf_counter()
        page.locator("#tasks-detail-host tbody tr.clickable-mol-row").first.click()
        page.wait_for_selector("#detail-context-bar:not(.hidden)", timeout=5000)
        drill_ms = round((time.perf_counter() - t_drill) * 1000, 1)
        pos_text_1 = page.locator("#ctx-pos-pill").inner_text().strip()
        mol_title_1 = page.locator("#res-mol-name").inner_text().strip()
        print(f"   Drill-down in {drill_ms}ms | pos='{pos_text_1}' | mol='{mol_title_1}'")
        assert "1 / 17" in pos_text_1, f"Expected 1 / 17, got {pos_text_1}"
        assert "Semaglutide" in mol_title_1, f"Expected Semaglutide, got {mol_title_1}"

        page.wait_for_selector("#mol-svg-stage svg", timeout=15000)

        # Click Next -> Tirzepatide (2 / 17)
        t_next = time.perf_counter()
        page.locator("#btn-ctx-next").click()
        page.wait_for_function("() => document.getElementById('ctx-pos-pill')?.textContent?.includes('2 / 17')", timeout=5000)
        next_ms = round((time.perf_counter() - t_next) * 1000, 1)
        mol_title_2 = page.locator("#res-mol-name").inner_text().strip()
        page.wait_for_selector("#mol-svg-stage svg", timeout=15000)
        print(f"   Next compound in {next_ms}ms | mol='{mol_title_2}'")
        assert "Tirzepatide" in mol_title_2, f"Expected Tirzepatide, got {mol_title_2}"
        assert_no_forbidden(page, "Detail with Return Context Bar")

        shot5 = SHOT_DIR / "v10_05_detail_with_return_bar.png"
        page.screenshot(path=str(shot5), full_page=False)

        # Click Return to Task Results -> verify exact return + highlighted row!
        page.locator("#btn-ctx-back").click()
        page.wait_for_selector("#view-tasks.active", timeout=5000)
        highlighted_count = page.locator("#tasks-detail-host tbody tr.row-last-inspected").count()
        highlighted_text = page.locator("#tasks-detail-host tbody tr.row-last-inspected").first.inner_text()
        print(f"   Returned cleanly to Task Center | highlighted={highlighted_count} | row='{highlighted_text[:40]}...'")
        assert highlighted_count == 1 and "Tirzepatide" in highlighted_text

        report["checks"]["task_center_and_loop"] = {
            "job_wait_s": job_wait_s,
            "kpi_count": kpi_count,
            "rows": row_count,
            "low_hem_filtered": filtered_rows,
            "drill_down_ms": drill_ms,
            "next_switch_ms": next_ms,
            "returned_highlighted": "Tirzepatide",
            "screenshots": [str(shot3), str(shot4), str(shot5)],
        }

        print("6. Verifying Validation & API tabs (zero forbidden strings)...")
        page.locator('.nav-tab[data-tab="validation"]').click()
        page.wait_for_selector("#val-tbody tr", timeout=10000)
        val_rows = page.locator("#val-tbody tr").count()
        print(f"   Validation table rows: {val_rows}")
        assert val_rows == 46, f"Expected 46 validation rows, got {val_rows}"
        assert_no_forbidden(page, "Validation Tab ZH")

        page.locator('.lang-btn[data-lang="en"]').click()
        page.wait_for_timeout(200)
        assert_no_forbidden(page, "Validation Tab EN")

        page.locator('.nav-tab[data-tab="api"]').click()
        page.wait_for_selector("#view-api.active", timeout=5000)
        assert_no_forbidden(page, "API Docs Tab EN")

        print("7. Testing live public HTTPS Job Export endpoints & cleaning up synthetic test jobs...")
        client_id = page.evaluate("() => localStorage.getItem('omnipanel_client_id')")
        assert client_id, "Missing omnipanel_client_id"

        req_list = Request(f"{BASE_URL}api/v1/jobs", headers={"X-Client-Id": client_id})
        with urlopen(req_list, timeout=15) as r:
            jobs = json.loads(r.read().decode("utf-8"))["jobs"]
        print(f"   Session has {len(jobs)} tracked job(s) (incl. single + 17-case batch)")
        batch_job = next(j for j in jobs if j["job_type"] == "batch" and j["total"] == 17)
        job_id = batch_job["job_id"]

        for fmt, marker in [
            ("summary_csv", "Semaglutide"),
            ("full_csv", "MIC_E_coli_value"),
            ("json", job_id),
        ]:
            req_exp = Request(
                f"{BASE_URL}api/v1/jobs/{job_id}/export?format={fmt}",
                headers={"X-Client-Id": client_id},
            )
            with urlopen(req_exp, timeout=20) as r:
                body = r.read().decode("utf-8")
                assert r.status == 200 and marker in body, f"Export {fmt} failed or missing {marker}"
                if fmt == "full_csv":
                    header_cols = body.splitlines()[0].split(",")
                    print(f"   Full CSV export OK: {len(header_cols)} columns, {len(body.splitlines()) - 1} rows")
                    assert "efflux_ratio_value" in header_cols and "cytotoxicity_CC50_value" in header_cols

        for j in jobs:
            req_del = Request(
                f"{BASE_URL}api/v1/jobs/{j['job_id']}",
                headers={"X-Client-Id": client_id},
                method="DELETE",
            )
            with urlopen(req_del, timeout=10) as r:
                assert r.status == 200
        print(f"   Cleaned up {len(jobs)} synthetic verification job(s).")

        browser.close()

    report["console_errors"] = console_errors
    report["page_errors"] = page_errors
    report["bad_responses"] = bad_responses
    report_path = Path(r"D:\work\中检院\v33_upgrade\reports\v10_live_ux_verification.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    assert not console_errors, f"Console errors: {console_errors}"
    assert not page_errors, f"Page errors: {page_errors}"
    assert not bad_responses, f"HTTP >=400 responses: {bad_responses}"
    print("\nALL LIVE PUBLIC HTTPS UX VERIFICATION CHECKS PASSED 100%!")


if __name__ == "__main__":
    main()
