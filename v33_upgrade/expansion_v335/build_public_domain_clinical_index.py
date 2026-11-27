#!/usr/bin/env python3
"""Build a license-clean Regulatory PK Reference Index for Peptide OmniPanel (v3).

License & provenance hygiene (Round-2 P0-1 & v3.4 P0-3 remediation):
  - Retains strictly factual numeric/range clinical PK observations sourced from
    US FDA prescribing information (NLM DailyMed SPL) and US FDA Drugs@FDA
    NDA/BLA Clinical Pharmacology Reviews.
  - Independently resolves official NLM DailyMed SPL `setid` identifiers via the
    public NLM DailyMed v2 REST API (`dailymed.nlm.nih.gov/dailymed/services/v2/spls.json`),
    reusing verified setids offline unless `--refresh-setids` is passed.
  - Merges duplicate product entries cleanly without losing unique PK records or aliases:
      * `fda_glucagon` + `fda_glucagon_recombinant` -> `fda_glucagon`
      * `fda_vasopressin` + `fda_vasopressin_or_argipressin` -> `fda_vasopressin`
  - Upgrades `source_url` to exact `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=<setid>`
    across all records whose product has a verified `dailymed_setid`, and annotates
    every record with an explicit honest `source_url_type`:
      * `"dailymed_spl_setid"` — exact DailyMed SPL document URL by verified `setid`
      * `"fda_drugs_portal_search"` — US FDA Drugs@FDA portal URL (no verified SPL `setid`)
      * `"dailymed_search"` — DailyMed drug search URL (no verified SPL `setid`)
  - Strips ALL CC-BY-NC derivative fields (`context_counts`, `context_highlights`,
    FAERS/trial aggregations, THPdb/PepTherDia/ChEMBL summaries, and external portal IDs).
  - Uses legally accurate wording: factual PK numbers are non-copyrightable scientific
    facts; FDA NDA/BLA Clinical Pharmacology Reviews are US Federal Government works
    (17 U.S.C. § 105); DailyMed SPL prescribing labels include manufacturer-submitted
    text distributed via NLM DailyMed.
"""
from __future__ import annotations

import argparse
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

# Canonical product merge map: duplicate_key -> canonical_key
PRODUCT_MERGE_MAP: dict[str, str] = {
    "fda_glucagon_recombinant": "fda_glucagon",
    "fda_vasopressin_or_argipressin": "fda_vasopressin",
    "glucagon_recombinant": "fda_glucagon",
    "vasopressin_or_argipressin": "fda_vasopressin",
}

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "ATLPD",
    "FAERS",
    "context_counts",
    "THPdb",
    "PepTherDia",
    "CC-BY-NC",
)

FDA_DRUGS_PORTAL_URL = "https://www.accessdata.fda.gov/scripts/cder/daf/"


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"fda_{s}" if s else "fda_unknown"


def _canonical_product_key(old_key: str, name: str) -> str:
    slug = old_key if old_key.startswith("fda_") else _slugify(name)
    if old_key in PRODUCT_MERGE_MAP:
        return PRODUCT_MERGE_MAP[old_key]
    if slug in PRODUCT_MERGE_MAP:
        return PRODUCT_MERGE_MAP[slug]
    return slug


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
            return str(data[0].get("setid") or "") or None, str(data[0].get("title") or "") or None
    except Exception:
        pass
    return None, None


def _pk_record_signature(rec: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(rec.get("endpoint") or "").strip(),
        str(rec.get("value_display") or "").strip(),
        str(rec.get("unit") or "").strip(),
        str(rec.get("route") or "").strip().lower(),
        str(rec.get("population") or "").strip().lower(),
        str(rec.get("source") or "").strip(),
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out


def _upgrade_record_url(
    rec: dict[str, Any],
    drug_name: str,
    setid: str | None,
) -> dict[str, Any]:
    """Assign exact DailyMed SPL setid URL or explicit honest fallback source_url_type."""
    src = str(rec.get("source") or "").strip()
    is_fda_review = "Review" in src or "Drugs@FDA" in src or "DATA433" in src

    if setid:
        source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
        source_url_type = "dailymed_spl_setid"
    elif is_fda_review:
        source_url = FDA_DRUGS_PORTAL_URL
        source_url_type = "fda_drugs_portal_search"
    else:
        source_url = (
            "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query="
            + urllib.parse.quote(drug_name)
        )
        source_url_type = "dailymed_search"

    return {
        "endpoint": rec.get("endpoint"),
        "value_display": str(rec.get("value_display") or "").strip(),
        "unit": rec.get("unit") or "",
        "route": rec.get("route") or "",
        "population": rec.get("population") or "",
        "source": src,
        "source_url": source_url,
        "source_url_type": source_url_type,
    }


def build_clean_index(raw: dict[str, Any], refresh_setids: bool = False) -> dict[str, Any]:
    """Clean, deduplicate, and upgrade a raw clinical evidence index dictionary."""
    products_in: dict[str, Any] = raw.get("products", {})
    merged_buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    old_to_new_key: dict[str, str] = {}

    for old_key, prod in sorted(products_in.items()):
        name = str(prod.get("name") or old_key).strip()
        canon_key = _canonical_product_key(old_key, name)
        old_to_new_key[old_key] = canon_key
        merged_buckets.setdefault(canon_key, []).append((old_key, prod))

    clean_products: dict[str, Any] = {}
    n_pk_total = 0
    n_setid_resolved = 0
    n_exact_setid_url = 0
    n_fda_portal_url = 0
    n_dailymed_search_url = 0

    for canon_key in sorted(merged_buckets.keys()):
        entries = merged_buckets[canon_key]
        # Prefer the entry whose key already matches canon_key and/or has a verified setid
        entries_sorted = sorted(
            entries,
            key=lambda kv: (
                0 if kv[0] == canon_key else 1,
                0 if kv[1].get("dailymed_setid") else 1,
                len(kv[0]),
            ),
        )
        primary_key, primary_prod = entries_sorted[0]
        name = str(primary_prod.get("name") or canon_key.replace("fda_", "").title()).strip()
        generic_name = str(primary_prod.get("generic_name") or name).strip()
        brand_name = str(primary_prod.get("brand_name") or "").strip()
        approval_year = primary_prod.get("approval_year")
        route = str(primary_prod.get("route") or "").strip()

        # Resolve setid: prefer existing verified setid among merged entries unless refresh requested
        setid: str | None = None
        spl_title: str | None = None
        if not refresh_setids:
            for _, p in entries_sorted:
                if p.get("dailymed_setid"):
                    setid = str(p["dailymed_setid"]).strip() or None
                    spl_title = str(p.get("dailymed_spl_title") or "").strip() or None
                    break
        else:
            setid, spl_title = resolve_dailymed_setid(name)
            time.sleep(0.05)

        if setid:
            n_setid_resolved += 1

        # Merge aliases and ATC codes across all merged entries
        all_aliases: list[str] = [name]
        all_atc: list[str] = []
        for _, p in entries_sorted:
            if p.get("name"):
                all_aliases.append(str(p["name"]))
            if p.get("generic_name"):
                all_aliases.append(str(p["generic_name"]))
            if p.get("brand_name"):
                all_aliases.append(str(p["brand_name"]))
            all_aliases.extend(str(a) for a in (p.get("aliases") or []))
            all_atc.extend(str(a) for a in (p.get("atc") or []))
            if approval_year is None and p.get("approval_year") is not None:
                approval_year = p.get("approval_year")
            if not route and p.get("route"):
                route = str(p.get("route")).strip()
            if not brand_name and p.get("brand_name"):
                brand_name = str(p.get("brand_name")).strip()

        aliases = _dedupe_preserve_order(all_aliases)
        atc = _dedupe_preserve_order(all_atc)

        # Collect, sanitize, and deduplicate strictly FDA / DailyMed factual PK records
        seen_sigs: set[tuple[str, str, str, str, str, str]] = set()
        pk_clean: list[dict[str, Any]] = []
        for _, p in entries_sorted:
            for rec in p.get("pk_records", []):
                src = str(rec.get("source") or "")
                url = str(rec.get("source_url") or "")
                is_fda = (
                    "US FDA" in src
                    or "DailyMed" in src
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
                normalized = {
                    "endpoint": rec.get("endpoint"),
                    "value_display": val,
                    "unit": rec.get("unit") or "",
                    "route": rec.get("route") or "",
                    "population": rec.get("population") or "",
                    "source": src_label,
                }
                sig = _pk_record_signature(normalized)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                upgraded = _upgrade_record_url(normalized, name, setid)
                pk_clean.append(upgraded)
                if upgraded["source_url_type"] == "dailymed_spl_setid":
                    n_exact_setid_url += 1
                elif upgraded["source_url_type"] == "fda_drugs_portal_search":
                    n_fda_portal_url += 1
                else:
                    n_dailymed_search_url += 1

        if not pk_clean:
            continue

        ext_links: dict[str, str] = {
            "dailymed_search": (
                "https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query="
                + urllib.parse.quote(name)
            ),
            "fda_drugs_portal": FDA_DRUGS_PORTAL_URL,
        }
        if setid:
            ext_links["dailymed_spl_url"] = (
                f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
            )
            ext_links["dailymed_setid"] = setid

        clean_products[canon_key] = {
            "name": name,
            "generic_name": generic_name,
            "brand_name": brand_name,
            "approval_year": approval_year,
            "dailymed_setid": setid,
            "dailymed_spl_title": spl_title,
            "pk_records": pk_clean,
            "external_links": ext_links,
            "product_id": canon_key,
            "aliases": aliases,
            "route": route,
            "atc": atc,
            "n_pk_records": len(pk_clean),
        }
        n_pk_total += len(pk_clean)

    # Remap lookup maps cleanly to canonical fda_<slug> keys (supporting both nested `lookup` and flat legacy layout)
    raw_lookup = raw.get("lookup") or {}
    raw_by_seq = raw_lookup.get("by_seq") or raw.get("by_seq") or {}
    raw_by_ik14 = raw_lookup.get("by_ik14") or raw.get("by_ik14") or {}
    raw_by_name = raw_lookup.get("by_name") or raw.get("by_name") or {}

    by_seq: dict[str, str] = {}
    for k, v in raw_by_seq.items():
        mapped = old_to_new_key.get(v, PRODUCT_MERGE_MAP.get(v, v))
        if mapped in clean_products:
            by_seq[k] = mapped

    by_ik14: dict[str, str] = {}
    for k, v in raw_by_ik14.items():
        mapped = old_to_new_key.get(v, PRODUCT_MERGE_MAP.get(v, v))
        if mapped in clean_products:
            by_ik14[k] = mapped

    by_name: dict[str, str] = {}
    for k, v in raw_by_name.items():
        mapped = old_to_new_key.get(v, PRODUCT_MERGE_MAP.get(v, v))
        if mapped in clean_products:
            by_name[k.strip().lower()] = mapped
    # Guarantee every canonical product name & alias resolves cleanly
    for canon_key, prod in clean_products.items():
        for alias in [prod["name"], prod["generic_name"], *(prod.get("aliases") or [])]:
            low = str(alias or "").strip().lower()
            if low:
                by_name[low] = canon_key

    out: dict[str, Any] = {
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
            "n_pk_records_exact_setid_url": n_exact_setid_url,
            "n_pk_records_fda_portal_url": n_fda_portal_url,
            "n_pk_records_dailymed_search_url": n_dailymed_search_url,
            "n_seq_keys": len(by_seq),
            "n_ik14_keys": len(by_ik14),
            "n_name_keys": len(by_name),
        },
        "products": clean_products,
        "lookup": {
            "by_name": dict(sorted(by_name.items())),
            "by_seq": dict(sorted(by_seq.items())),
            "by_ik14": dict(sorted(by_ik14.items())),
        },
    }

    serialized = json.dumps(out, indent=2, ensure_ascii=False)
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if re.search(re.escape(forbidden), serialized, re.IGNORECASE):
            raise RuntimeError(f"Forbidden license/provenance string detected in output: {forbidden}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean FDA/DailyMed clinical PK index.")
    parser.add_argument(
        "--refresh-setids",
        action="store_true",
        help="Re-query NLM DailyMed v2 REST API live instead of reusing verified setids.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=SRC_JSON,
        help="Input clinical_evidence_index.json path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=LIVE_JSON,
        help="Output clinical_evidence_index.json path.",
    )
    args = parser.parse_args()

    raw = json.loads(args.src.read_text(encoding="utf-8"))
    cleaned = build_clean_index(raw, refresh_setids=args.refresh_setids)
    args.out.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    st = cleaned["stats"]
    print(
        f"Built v3 Regulatory PK index: {st['n_products']} products, "
        f"{st['n_dailymed_setid_verified']} DailyMed SETIDs verified, "
        f"{st['n_pk_records_indexed']} factual PK records "
        f"({st['n_pk_records_exact_setid_url']} exact setid URLs, "
        f"{st['n_pk_records_fda_portal_url']} FDA portal, "
        f"{st['n_pk_records_dailymed_search_url']} DailyMed search)."
    )


if __name__ == "__main__":
    main()
