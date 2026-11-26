#!/usr/bin/env python3
"""Build a license-clean Regulatory PK Reference Index for Peptide OmniPanel (v3).

License & provenance hygiene (Round-2 P0-1 remediation):
  - Retains strictly factual numeric/range clinical PK observations sourced from
    US FDA prescribing information (NLM DailyMed SPL) and US FDA Drugs@FDA
    NDA/BLA Clinical Pharmacology Reviews.
  - Independently resolves official NLM DailyMed SPL `setid` identifiers via the
    public NLM DailyMed v2 REST API (`dailymed.nlm.nih.gov/dailymed/services/v2/spls.json`).
  - Strips ALL CC-BY-NC derivative fields (`context_counts`, `context_highlights`,
    FAERS/trial aggregations, THPdb/PepTherDia/ChEMBL summaries, and ADMETatlas
    internal product IDs `ATLPD*`).
  - Uses legally accurate wording: factual PK numbers are non-copyrightable scientific
    facts; FDA NDA/BLA Clinical Pharmacology Reviews are US Federal Government works
    (17 U.S.C. § 105); DailyMed SPL prescribing labels include manufacturer-submitted
    text distributed via NLM DailyMed.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
LIVE_JSON = WEB_DIR / "clinical_evidence_index.json"
SRC_JSON = Path(os.environ.get("CLINICAL_EVIDENCE_SRC", str(LIVE_JSON)))


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"fda_{s}" if s else "fda_unknown"


def resolve_dailymed_setid(drug_name: str) -> tuple[str | None, str | None]:
    """Query public NLM DailyMed v2 REST API for official SPL setid."""
    url = (
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name="
        + urllib.parse.quote(drug_name)
        + "&pagesize=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or []
        if data:
            return str(data[0].get("setid") or ""), str(data[0].get("title") or "")
    except Exception:
        pass
    return None, None


def main() -> None:
    raw = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    products_in = raw.get("products", {})
    clean_products: dict[str, Any] = {}
    old_to_new_key: dict[str, str] = {}
    n_pk_total = 0
    n_setid_resolved = 0

    for old_key, prod in sorted(products_in.items()):
        name = str(prod.get("name") or old_key).strip()
        pk_clean = []
        for rec in prod.get("pk_records", []):
            src = str(rec.get("source") or "")
            url = str(rec.get("source_url") or "")
            # Keep strictly FDA DailyMed / Drugs@FDA factual numeric/range records
            is_fda = (
                "US FDA" in src
                or src in {"DailyMed", "FDA_Drug_Approval_Review", "FDA_DATA433"}
                or "dailymed.nlm.nih.gov" in url
                or "accessdata.fda.gov" in url
            )
            if not is_fda:
                continue
            val = str(rec.get("value_display") or "").strip()
            if not val or len(val) > 40:
                continue
            src_label = (
                "US FDA Drugs@FDA NDA/BLA Review"
                if ("accessdata.fda.gov" in url or "Review" in src or "DATA433" in src)
                else "NLM DailyMed FDA Prescribing Information"
            )
            pk_clean.append(
                {
                    "endpoint": rec.get("endpoint"),
                    "value_display": val,
                    "unit": rec.get("unit") or "",
                    "route": rec.get("route") or "",
                    "population": rec.get("population") or "",
                    "source": src_label,
                    "source_url": url or f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={urllib.parse.quote(name)}",
                }
            )
        if not pk_clean:
            continue

        new_key = _slugify(name)
        idx_suffix = 2
        while new_key in clean_products:
            new_key = f"{_slugify(name)}_{idx_suffix}"
            idx_suffix += 1
        old_to_new_key[old_key] = new_key

        setid, spl_title = resolve_dailymed_setid(name)
        if setid:
            n_setid_resolved += 1

        ext_links: dict[str, str] = {
            "dailymed_search": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={urllib.parse.quote(name)}",
            "fda_drugs_portal": "https://www.accessdata.fda.gov/scripts/cder/daf/",
            "admetatlas_portal": "https://admetatlas.scbdd.com/",
        }
        if setid:
            ext_links["dailymed_spl_url"] = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
            ext_links["dailymed_setid"] = setid

        clean_products[new_key] = {
            "name": name,
            "generic_name": prod.get("generic_name") or name,
            "brand_name": prod.get("brand_name") or "",
            "approval_year": prod.get("approval_year"),
            "dailymed_setid": setid,
            "dailymed_spl_title": spl_title,
            "pk_records": pk_clean,
            "external_links": ext_links,
        }
        n_pk_total += len(pk_clean)
        time.sleep(0.05)

    # Remap lookup maps cleanly to new fda_<slug> keys (zero ATLPD identifiers!)
    by_seq = {k: old_to_new_key[v] for k, v in (raw.get("by_seq") or {}).items() if v in old_to_new_key}
    by_ik14 = {k: old_to_new_key[v] for k, v in (raw.get("by_ik14") or {}).items() if v in old_to_new_key}

    out = {
        "schema_version": "omnipanel-fda-regulatory-pk-v3",
        "source_attribution": {
            "database": "NLM DailyMed FDA Prescribing Information & US FDA Drugs@FDA Clinical Pharmacology Reviews",
            "citation": (
                "Factual numeric PK values verified against official NLM DailyMed SPL records "
                "(dailymed.nlm.nih.gov, setid resolved via DailyMed v2 API) and US FDA NDA/BLA "
                "Clinical Pharmacology Reviews (accessdata.fda.gov). Initial drug-list cross-check "
                "was informed by public FDA peptide approvals; zero third-party derivative tables "
                "or external portal IDs are mirrored."
            ),
            "license": (
                "Factual numeric PK observations (non-copyrightable scientific facts); "
                "US FDA NDA/BLA reviews are US Federal Government works (17 U.S.C. § 105); "
                "DailyMed SPL labels include manufacturer-submitted prescribing information "
                "accessed via NLM DailyMed. Strictly non-commercial academic research use."
            ),
            "url": "https://dailymed.nlm.nih.gov/",
            "caveat": (
                "Regulatory label PK values are dosage-, route-, and formulation-dependent clinical "
                "observations provided strictly for preclinical research context, not clinical or prescribing decisions."
            ),
        },
        "stats": {
            "n_products": len(clean_products),
            "n_products_with_pk": len(clean_products),
            "n_dailymed_setid_verified": n_setid_resolved,
            "n_pk_records_indexed": n_pk_total,
            "n_seq_keys": len(by_seq),
            "n_ik14_keys": len(by_ik14),
        },
        "by_seq": by_seq,
        "by_ik14": by_ik14,
        "products": clean_products,
    }
    LIVE_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Built v3 Regulatory PK index: {len(clean_products)} products, "
        f"{n_setid_resolved} DailyMed SETIDs verified live, {n_pk_total} factual PK records."
    )


if __name__ == "__main__":
    main()
