#!/usr/bin/env python3
"""Build a 100% license-clean Public-Domain Regulatory PK Index for Peptide OmniPanel.

License hygiene (P0-1):
  - Retains ONLY factual PK parameters sourced directly from US Federal Government
    Public-Domain regulatory documents (FDA DailyMed SPL XMLs & FDA Drugs@FDA NDA/BLA
    Clinical Pharmacology Reviews; 17 U.S.C. § 105 — zero CC-BY-NC restriction).
  - Strips ALL CC-BY-NC 4.0 derivative annotations (ADMETatlas context_counts,
    context_highlights, FAERS/trial aggregations, THPdb, PepTherDia, ChEMBL summaries).
  - Provides clean outbound reference links (`admetatlas_url`) for users who wish to
    inspect external clinical/pharmacovigilance evidence directly on ADMETatlas.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
LIVE_JSON = WEB_DIR / "clinical_evidence_index.json"
SRC_JSON = Path(os.environ.get("CLINICAL_EVIDENCE_SRC", str(LIVE_JSON)))

PUBLIC_DOMAIN_SOURCES = {
    "DailyMed",
    "FDA_Drug_Approval_Review",
    "FDA_DATA433",
}


def main() -> None:
    if not SRC_JSON.exists():
        SRC_JSON.write_text(LIVE_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    raw = json.loads(SRC_JSON.read_text(encoding="utf-8"))

    clean_products: dict[str, Any] = {}
    valid_pids: set[str] = set()
    n_pd_records = 0

    for pid, p in (raw.get("products") or {}).items():
        pd_pks = []
        seen = set()
        for r in p.get("pk_records") or []:
            src = str(r.get("source") or "").strip()
            url = str(r.get("reference_url") or "").strip()
            if src not in PUBLIC_DOMAIN_SOURCES:
                continue
            if not (url.startswith("https://dailymed.nlm.nih.gov/") or url.startswith("https://www.accessdata.fda.gov/")):
                continue
            # Keep clean numeric/range regulatory PK facts only (exclude truncated prose snippets)
            val_disp = str(r.get("value_display") or "").strip()
            if len(val_disp) > 36:
                continue
            dedup_key = (r.get("endpoint"), val_disp, r.get("route"), r.get("species"), src)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            pd_pks.append(
                {
                    "endpoint": r.get("endpoint"),
                    "endpoint_id": r.get("endpoint_id"),
                    "value_display": val_disp,
                    "species": r.get("species") or "human",
                    "route": r.get("route") or "unspecified",
                    "matrix": r.get("matrix") or "",
                    "dose": r.get("dose") or "",
                    "source": "US FDA DailyMed SPL" if src == "DailyMed" else "US FDA NDA/BLA Review",
                    "reference_url": url,
                }
            )
        if not pd_pks:
            continue
        valid_pids.add(pid)
        n_pd_records += len(pd_pks)
        clean_products[pid] = {
            "product_id": pid,
            "name": p["name"],
            "aliases": p.get("aliases") or [p["name"]],
            "route": p.get("route") or "",
            "atc": p.get("atc") or [],
            "n_pk_records": len(pd_pks),
            "pk_records": pd_pks,
            "external_links": {
                "dailymed_search": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={p['name'].replace(' ', '+')}",
                "admetatlas_portal": "https://admetatlas.scbdd.com/",
                "admetatlas_product_id": pid,
            },
        }

    # Filter lookups strictly to products with public-domain US FDA/DailyMed records
    old_lk = raw.get("lookup") or {}
    clean_lk = {
        "by_name": {k: v for k, v in (old_lk.get("by_name") or {}).items() if v in valid_pids},
        "by_seq": {k: v for k, v in (old_lk.get("by_seq") or {}).items() if v in valid_pids},
        "by_ik14": {k: v for k, v in (old_lk.get("by_ik14") or {}).items() if v in valid_pids},
    }

    out = {
        "schema_version": "omnipanel-fda-public-domain-pk-v2",
        "source_attribution": {
            "database": "US FDA DailyMed SPL & Drugs@FDA Clinical Pharmacology Reviews (Public Domain, 17 U.S.C. § 105)",
            "citation": "Primary US FDA prescribing information (dailymed.nlm.nih.gov) & NDA/BLA reviews (accessdata.fda.gov); product cross-reference identifiers compatible with ADMETatlas V1.0 (Yi et al., NAR 2026) via outbound link only.",
            "license": "Public Domain (US Federal Government Works, 17 U.S.C. § 105) — zero CC-BY-NC derivative redistribution.",
            "url": "https://dailymed.nlm.nih.gov/",
            "caveat": "Regulatory label PK values are dosage- and formulation-dependent clinical observations provided strictly for preclinical research context, not clinical or prescribing decisions.",
        },
        "stats": {
            "n_products": len(clean_products),
            "n_products_with_pk": len(clean_products),
            "n_pk_records_indexed": n_pd_records,
            "n_seq_keys": len(clean_lk["by_seq"]),
            "n_ik14_keys": len(clean_lk["by_ik14"]),
        },
        "lookup": clean_lk,
        "products": clean_products,
    }

    LIVE_JSON.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"CLEAN_PUBLIC_DOMAIN_INDEX_BUILT: {out['stats']} | bytes={LIVE_JSON.stat().st_size:,}")


if __name__ == "__main__":
    main()
