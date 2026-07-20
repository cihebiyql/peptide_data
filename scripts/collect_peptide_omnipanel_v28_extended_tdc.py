"""Freeze raw TDC small-molecule labels for V2.8 extended research heads.

This collector deliberately performs *no* peptide-domain claim.  It creates a
hash-addressed, row-level source table for project-trained transfer baselines.
Every output is marked research-only because TDC's ADME/Tox datasets are
small-molecule data and component licence review has not been completed for a
redistributable release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "peptide_omnipanel_v28_extended_tdc_20260719_run1"
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
TDC_ADME_URL = "https://tdcommons.ai/single_pred_tasks/adme/"
TDC_TOX_URL = "https://tdcommons.ai/single_pred_tasks/tox/"
USER_AGENT = "peptide-omnipanel-v28-research/1.0"


@dataclass(frozen=True)
class Dataset:
    endpoint_id: str
    source_dataset: str
    file_id: int
    task_kind: str
    unit: str
    positive_class_semantics: str | None
    value_semantics: str
    transform: str = "identity"

    @property
    def source_url(self) -> str:
        return DATAVERSE_URL.format(file_id=self.file_id)


# File identifiers are pinned from the TDC upstream metadata table.  The
# pLD50 conversion is intentional: V2.8's registry exposes log10(mol/kg),
# while Zhu's source target is log10(1/(mol/kg)).
DATASETS = (
    Dataset(
        "HIA",
        "hia_hou",
        4259591,
        "classification",
        "probability",
        "TDC HIA_Hou Y=1",
        "source_defined_human_intestinal_absorption",
    ),
    Dataset(
        "Pgp_inhibition",
        "pgp_broccatelli",
        4259597,
        "classification",
        "probability",
        "TDC Pgp_Broccatelli Y=1 (P-gp inhibitor)",
        "source_defined_Pgp_inhibition",
    ),
    Dataset(
        "CYP1A2_inhibition",
        "cyp1a2_veith",
        4259573,
        "classification",
        "probability",
        "TDC CYP1A2_Veith Y=1 (inhibitor)",
        "source_defined_CYP1A2_inhibition",
    ),
    Dataset(
        "CYP2C19_inhibition",
        "cyp2c19_veith",
        4259576,
        "classification",
        "probability",
        "TDC CYP2C19_Veith Y=1 (inhibitor)",
        "source_defined_CYP2C19_inhibition",
    ),
    Dataset(
        "CYP2C9_inhibition",
        "cyp2c9_veith",
        4259577,
        "classification",
        "probability",
        "TDC CYP2C9_Veith Y=1 (inhibitor)",
        "source_defined_CYP2C9_inhibition",
    ),
    Dataset(
        "CYP2D6_inhibition",
        "cyp2d6_veith",
        4259580,
        "classification",
        "probability",
        "TDC CYP2D6_Veith Y=1 (inhibitor)",
        "source_defined_CYP2D6_inhibition",
    ),
    Dataset(
        "CYP3A4_inhibition",
        "cyp3a4_veith",
        4259582,
        "classification",
        "probability",
        "TDC CYP3A4_Veith Y=1 (inhibitor)",
        "source_defined_CYP3A4_inhibition",
    ),
    Dataset(
        "CYP2C9_substrate",
        "cyp2c9_substrate_carbonmangels",
        4259584,
        "classification",
        "probability",
        "TDC CYP2C9_Substrate_CarbonMangels Y=1 (substrate)",
        "source_defined_CYP2C9_substrate",
    ),
    Dataset(
        "CYP2D6_substrate",
        "cyp2d6_substrate_carbonmangels",
        4259578,
        "classification",
        "probability",
        "TDC CYP2D6_Substrate_CarbonMangels Y=1 (substrate)",
        "source_defined_CYP2D6_substrate",
    ),
    Dataset(
        "CYP3A4_substrate",
        "cyp3a4_substrate_carbonmangels",
        4259581,
        "classification",
        "probability",
        "TDC CYP3A4_Substrate_CarbonMangels Y=1 (substrate)",
        "source_defined_CYP3A4_substrate",
    ),
    Dataset(
        "hERG",
        "herg_karim",
        6822246,
        "classification",
        "probability",
        "TDC hERG_Karim Y=1 (hERG block below 10 uM)",
        "source_defined_hERG_block",
    ),
    Dataset(
        "AMES",
        "ames",
        4259564,
        "classification",
        "probability",
        "TDC AMES Y=1 (mutagenic)",
        "source_defined_Ames_mutagenicity",
    ),
    Dataset(
        "DILI",
        "dili",
        4259585,
        "classification",
        "probability",
        "TDC DILI Y=1 (drug-induced liver injury)",
        "source_defined_drug_induced_liver_injury",
    ),
    Dataset(
        "ClinTox",
        "clintox",
        4259572,
        "classification",
        "probability",
        "TDC ClinTox Y=1 (clinical toxicity label)",
        "source_defined_clinical_toxicity",
    ),
    Dataset(
        "carcinogenicity",
        "carcinogens_lagunin",
        4259570,
        "classification",
        "probability",
        "TDC Carcinogens_Lagunin Y=1 (carcinogenic)",
        "source_defined_carcinogenicity",
    ),
    Dataset(
        "skin_reaction",
        "skin_reaction",
        4259609,
        "classification",
        "probability",
        "TDC Skin_Reaction Y=1 (skin sensitization/reaction)",
        "source_defined_skin_reaction",
    ),
    Dataset(
        "LD50",
        "ld50_zhu",
        4267146,
        "regression",
        "log10(mol/kg)",
        None,
        "rat_oral_log10_median_lethal_dose",
        "negate_pLD50",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch_bytes(url: str, timeout: int, retries: int) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to download {url}: {error}")


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def numeric(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_rows(dataset: Dataset, raw_path: Path) -> tuple[list[dict[str, Any]], int]:
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows: list[dict[str, Any]] = []
        rejected = 0
        for index, source in enumerate(reader, start=1):
            smiles = first(source, "Drug", "X")
            target = numeric(first(source, "Y"))
            if not smiles or target is None:
                rejected += 1
                continue
            if dataset.task_kind == "classification" and target not in {0.0, 1.0}:
                rejected += 1
                continue
            transformed = -target if dataset.transform == "negate_pLD50" else target
            rows.append(
                {
                    "record_id": f"TDC:{dataset.source_dataset}:{index}",
                    "endpoint_id": dataset.endpoint_id,
                    "task_kind": dataset.task_kind,
                    "source_id": "tdc_extended_public_raw_research_only",
                    "source_dataset": dataset.source_dataset,
                    "source_file_id": dataset.file_id,
                    "source_row_index": index,
                    "source_record_id": first(source, "Drug_ID", "ID") or str(index),
                    "molecule_name": first(source, "Drug_ID", "ID"),
                    "smiles": smiles,
                    "raw_value": target,
                    "value": transformed,
                    "unit": dataset.unit,
                    "value_transform": dataset.transform,
                    "positive_class_semantics": dataset.positive_class_semantics or "",
                    "value_semantics": dataset.value_semantics,
                    "evidence_tier": (
                        "L2_source_defined_binary"
                        if dataset.task_kind == "classification"
                        else "L0_source_numeric_reexpressed"
                    ),
                    "training_domain": "small_molecule_transfer_only",
                    "research_only": "true",
                    "license_status": "component_license_pending_release_clearance",
                    "source_url": dataset.source_url,
                    "tdc_task_page": TDC_ADME_URL
                    if dataset.endpoint_id in {"HIA", "Pgp_inhibition"}
                    or dataset.endpoint_id.startswith("CYP")
                    else TDC_TOX_URL,
                }
            )
    return rows, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh", action="store_true", help="redownload raw files even if frozen"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        raw_path = raw_dir / f"{dataset.source_dataset}.tsv"
        if args.refresh or not raw_path.is_file() or not raw_path.stat().st_size:
            raw_path.write_bytes(fetch_bytes(dataset.source_url, args.timeout, args.retries))
        normalized, rejected = normalize_rows(dataset, raw_path)
        all_rows.extend(normalized)
        source_rows.append(
            {
                "endpoint_id": dataset.endpoint_id,
                "source_dataset": dataset.source_dataset,
                "file_id": dataset.file_id,
                "source_url": dataset.source_url,
                "raw_path": str(raw_path.relative_to(ROOT)),
                "raw_sha256": sha256_file(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "accepted_rows": len(normalized),
                "rejected_rows": rejected,
                "task_kind": dataset.task_kind,
                "value_transform": dataset.transform,
                "license_status": "component_license_pending_release_clearance",
                "training_domain": "small_molecule_transfer_only",
            }
        )
        print(f"frozen {dataset.endpoint_id}: {len(normalized)} accepted, {rejected} rejected")

    fields = [
        "record_id",
        "endpoint_id",
        "task_kind",
        "source_id",
        "source_dataset",
        "source_file_id",
        "source_row_index",
        "source_record_id",
        "molecule_name",
        "smiles",
        "raw_value",
        "value",
        "unit",
        "value_transform",
        "positive_class_semantics",
        "value_semantics",
        "evidence_tier",
        "training_domain",
        "research_only",
        "license_status",
        "source_url",
        "tdc_task_page",
    ]
    write_tsv(out_dir / "observations.tsv", all_rows, fields)
    write_tsv(
        out_dir / "source_manifest.tsv",
        source_rows,
        [
            "endpoint_id",
            "source_dataset",
            "file_id",
            "source_url",
            "raw_path",
            "raw_sha256",
            "raw_bytes",
            "accepted_rows",
            "rejected_rows",
            "task_kind",
            "value_transform",
            "license_status",
            "training_domain",
        ],
    )
    observations = out_dir / "observations.tsv"
    manifest = {
        "schema_version": "peptide-omnipanel-v28-extended-tdc-1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "scope": "internal_research_only_small_molecule_transfer_training",
        "source_bundle": {
            "tdc_dataverse_doi": "10.7910/DVN/21LKWG",
            "tdc_metadata_url": "https://raw.githubusercontent.com/mims-harvard/TDC/main/tdc/metadata.py",
            "component_license_status": "pending_release_clearance",
        },
        "counts": {
            "endpoints": len({row["endpoint_id"] for row in all_rows}),
            "observations": len(all_rows),
            "classification_rows": sum(row["task_kind"] == "classification" for row in all_rows),
            "regression_rows": sum(row["task_kind"] == "regression" for row in all_rows),
        },
        "outputs": {
            "observations": str(observations.relative_to(ROOT)),
            "observations_sha256": sha256_file(observations),
            "source_manifest": str((out_dir / "source_manifest.tsv").relative_to(ROOT)),
            "source_manifest_sha256": sha256_file(out_dir / "source_manifest.tsv"),
        },
        "semantic_invariants": [
            "All TDC rows are small-molecule transfer labels, never measured peptide labels.",
            "LD50_Zhu pLD50 is negated to the registry's log10(mol/kg) convention.",
            "No third-party prediction API or model weight is used.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
