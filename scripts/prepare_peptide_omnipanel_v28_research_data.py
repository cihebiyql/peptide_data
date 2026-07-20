"""Create a provenance-preserving V2.8 research dataset from frozen local data.

The output is intentionally a *staging* dataset.  It never upgrades weak,
positive-unlabeled, small-molecule transfer, or insufficient rows to measured
peptide evidence.  Each observation retains the original file, row identifier,
licence, condition, relation and endpoint-specific target-space definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "peptide_omnipanel_v28_research_data_20260719_run1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_sequence(value: str) -> str | None:
    sequence = "".join(str(value or "").split()).upper()
    return sequence if sequence and set(sequence) <= AA else None


def finite(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def local_row(
    *,
    endpoint_id: str,
    task_kind: str,
    source_path: Path,
    source_row: dict[str, str],
    sequence: str | None,
    target: float | None,
    target_space: str,
    evidence_tier: str,
    eligibility: str,
    status: str = "available",
    note: str = "",
) -> dict[str, Any]:
    stable_record_id = (
        source_row.get("record_id")
        or source_row.get("clean_row_id")
        or source_row.get("source_row_number")
        or ""
    )
    return {
        "observation_id": f"{endpoint_id}:{source_path.name}:{stable_record_id}",
        "endpoint_id": endpoint_id,
        "task_kind": task_kind,
        "source_path": str(source_path.relative_to(ROOT)),
        "source_record_id": source_row.get("record_id")
        or source_row.get("clean_row_id")
        or source_row.get("source_record_id")
        or "",
        "source_id": source_row.get("source_id", ""),
        "source_url": source_row.get("source_url", ""),
        "license": source_row.get("license", ""),
        "sequence": sequence or "",
        "smiles": source_row.get("smiles", "") or source_row.get("structure", ""),
        "target": "" if target is None else target,
        "target_space": target_space,
        "relation": source_row.get("normalized_relation") or source_row.get("relation") or "",
        "censoring_type": source_row.get("censoring_type", ""),
        "assay": source_row.get("raw_assay") or source_row.get("assay") or "",
        "species": source_row.get("raw_species") or source_row.get("species") or "",
        "matrix": source_row.get("raw_matrix") or source_row.get("matrix") or "",
        "source_group": source_row.get("source_group_ids") or source_row.get("doi_pmid") or "",
        "evidence_tier": evidence_tier,
        "eligibility": eligibility,
        "status": status,
        "note": note,
    }


def strict_numeric_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    selectors = {
        "human_plasma_stability": lambda text: "human plasma" in text,
        "mouse_plasma_stability": lambda text: "mouse plasma" in text,
        "intestinal_stability": lambda text: "intestinal" in text,
        "membrane_retention": lambda text: "membrane_retention" in text,
        "CL": lambda row, text: (
            row.get("endpoint_family") == "CL"
            and row.get("parameter_semantics") == "systemic_clearance"
            and row.get("parameter_basis") == "absolute"
        ),
    }
    for row in read_tsv(path):
        context = " ".join(
            row.get(field, "")
            for field in (
                "raw_endpoint",
                "raw_endpoint_detail",
                "raw_assay",
                "raw_matrix",
                "condition_json",
                "parameter_semantics",
            )
        ).lower()
        for endpoint_id, selector in selectors.items():
            selected = selector(row, context) if endpoint_id == "CL" else selector(context)
            if not selected:
                continue
            sequence = canonical_sequence(row.get("sequence", ""))
            relation = row.get("normalized_relation", "")
            target = finite(row.get("normalized_value", ""))
            if endpoint_id == "membrane_retention":
                target_space = "fraction_retained"
                task_kind = "regression"
            elif endpoint_id == "CL":
                target_space = "mL_per_min_per_kg"
                task_kind = "regression"
            else:
                target_space = "log10_hours"
                task_kind = "censored_regression"
            record = local_row(
                endpoint_id=endpoint_id,
                task_kind=task_kind,
                source_path=path,
                source_row=row,
                sequence=sequence,
                target=target,
                target_space=target_space,
                evidence_tier="L0_local_frozen_numeric",
                eligibility="research_candidate",
                status="available"
                if sequence and target is not None
                else "not_sequence_model_eligible",
                note="exact_only_for_initial_sequence_model"
                if relation == "="
                else "censored_preserved_not_initial_fit",
            )
            if relation != "=" or target is None:
                quarantine.append(record)
            else:
                accepted.append(record)
    return accepted, quarantine


def binary_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {
        "hemolysis_human_erythrocyte_20pct_at_50uM": (
            "hemolysis",
            "L2_source_defined_binary",
            "human_erythrocyte_ge20pct_at50uM",
        ),
        "general_peptide_toxicity_ToxinPred3_weak_benchmark": (
            "overall_peptide_toxicity",
            "L5_weak_benchmark_label",
            "ToxinPred3_source_defined_toxicity",
        ),
        "cell_line_cytotoxicity_ChEMBL_source_binary_candidate": (
            "cytotoxicity",
            "L2_source_defined_binary",
            "mixed_cell_line_source_defined_cytotoxicity",
        ),
    }
    for row in read_tsv(path):
        config = mapping.get(row.get("task_id", ""))
        if config is None:
            continue
        endpoint_id, tier, definition = config
        sequence = canonical_sequence(row.get("sequence", ""))
        label = finite(row.get("label", ""))
        if label not in {0.0, 1.0}:
            continue
        rows.append(
            local_row(
                endpoint_id=endpoint_id,
                task_kind="classification",
                source_path=path,
                source_row=row,
                sequence=sequence,
                target=label,
                target_space="binary_label",
                evidence_tier=tier,
                eligibility="research_candidate",
                status="available" if sequence else "not_sequence_model_eligible",
                note=definition,
            )
        )
    return rows


def hc50_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_tsv(path):
        if row.get("endpoint_detail") != "HC50_uM_regression" or row.get("relation") != "=":
            continue
        value_um = finite(row.get("value", ""))
        sequence = canonical_sequence(row.get("sequence", ""))
        if value_um is None or value_um <= 0:
            continue
        rows.append(
            local_row(
                endpoint_id="HC50",
                task_kind="censored_regression",
                source_path=path,
                source_row=row,
                sequence=sequence,
                target=math.log10(value_um * 1e-6),
                target_space="log10_mol_per_L",
                evidence_tier="L0_direct_numeric_curated_dataset",
                eligibility="research_candidate",
                status="available" if sequence else "not_sequence_model_eligible",
                note=(
                    "HemoPI2 pooled mammalian RBC species; source averages/ranges may be "
                    "pre-aggregated"
                ),
            )
        )
    return rows


def pu_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_tsv(path):
        if row.get("endpoint_family") != "cell_penetration":
            continue
        sequence = canonical_sequence(row.get("sequence", ""))
        rows.append(
            local_row(
                endpoint_id="cell_penetration",
                task_kind="positive_unlabeled",
                source_path=path,
                source_row=row,
                sequence=sequence,
                target=1.0,
                target_space="positive_membership_only",
                evidence_tier="L3_positive_unlabeled",
                eligibility="PU_ranking_only",
                status="available" if sequence else "not_sequence_model_eligible",
                note="positive examples only; absence must never be treated as a negative label",
            )
        )
    return rows


def kp_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_tsv(path):
        sequence = canonical_sequence(row.get("linear_sequence", ""))
        target = finite(row.get("value", ""))
        rows.append(
            local_row(
                endpoint_id="Kp",
                task_kind="mechanistic_tissue_vector",
                source_path=path,
                source_row=row,
                sequence=sequence,
                target=target,
                target_space="tissue_to_plasma_ratio",
                evidence_tier="L0_review_only_not_training",
                eligibility="not_trainable",
                status="review_only",
                note="Kp ratio basis/stereochemistry insufficient for supervised training",
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    inputs = {
        "strict_numeric": ROOT / "data/peptide_ml_cleaning_v1/strict_numeric.tsv",
        "binary_catalog": ROOT
        / "data/peptide_admet_ml_classification_v9/peptide_admet_classification_catalog_v9.tsv",
        "hc50": ROOT
        / "data"
        / "external_downloads"
        / "peptide_property_expansion_v13"
        / "sequence_pk"
        / "hemopi2_2025"
        / "normalized.tsv",
        "positive_unlabeled": ROOT / "data/peptide_ml_cleaning_v1/positive_unlabeled.tsv",
        "kp_review": ROOT / "data/peptide_kp_endpoints/peptide_kp_observations.tsv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen source missing: " + ", ".join(missing))

    rows, quarantine = strict_numeric_rows(inputs["strict_numeric"])
    rows.extend(binary_rows(inputs["binary_catalog"]))
    rows.extend(hc50_rows(inputs["hc50"]))
    rows.extend(pu_rows(inputs["positive_unlabeled"]))
    rows.extend(kp_rows(inputs["kp_review"]))
    # Explicit gaps make the 39-endpoint display auditable instead of silently
    # treating a fallback prior as a trained model.
    gaps = [
        ("protease_stability", "blocked: 375 observations overlap intestinal_stability"),
        ("degradation_site_probability", "blocked: no residue-level peptide labels"),
        ("immunogenicity_risk", "blocked: no therapeutic-peptide immunogenicity labels"),
    ]
    fields = [
        "observation_id",
        "endpoint_id",
        "task_kind",
        "source_path",
        "source_record_id",
        "source_id",
        "source_url",
        "license",
        "sequence",
        "smiles",
        "target",
        "target_space",
        "relation",
        "censoring_type",
        "assay",
        "species",
        "matrix",
        "source_group",
        "evidence_tier",
        "eligibility",
        "status",
        "note",
    ]
    observations_path = out_dir / "observations.tsv"
    quarantine_path = out_dir / "quarantine.tsv"
    write_tsv(observations_path, rows, fields)
    write_tsv(quarantine_path, quarantine, fields)
    counts = Counter(row["endpoint_id"] for row in rows)
    summary = [
        {
            "endpoint_id": endpoint,
            "rows": counts.get(endpoint, 0),
            "sequence_eligible_rows": sum(
                row["endpoint_id"] == endpoint and row["status"] == "available" for row in rows
            ),
            "status": "data_gap" if endpoint in dict(gaps) else "staged",
            "note": dict(gaps).get(endpoint, ""),
        }
        for endpoint in sorted(set(counts).union(dict(gaps)))
    ]
    summary_path = out_dir / "endpoint_summary.tsv"
    write_tsv(
        summary_path, summary, ["endpoint_id", "rows", "sequence_eligible_rows", "status", "note"]
    )
    manifest = {
        "schema_version": "peptide-omnipanel-v28-research-data-1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "scope": "internal_research_only",
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "outputs": {
            "observations": str(observations_path.relative_to(ROOT)),
            "observations_sha256": sha256_file(observations_path),
            "quarantine": str(quarantine_path.relative_to(ROOT)),
            "quarantine_sha256": sha256_file(quarantine_path),
            "endpoint_summary": str(summary_path.relative_to(ROOT)),
            "endpoint_summary_sha256": sha256_file(summary_path),
        },
        "counts": {
            "observations": len(rows),
            "quarantine": len(quarantine),
            "endpoints_with_rows": len(counts),
            "endpoint_counts": dict(sorted(counts.items())),
        },
        "unresolved_endpoint_gaps": [
            {"endpoint_id": endpoint, "reason": reason} for endpoint, reason in gaps
        ],
        "invariants": [
            "Censored numeric labels remain in quarantine until a censored learner is implemented.",
            "Cell penetration is positive-unlabeled and cannot be evaluated as a binary "
            "classifier.",
            "Protease and intestinal stability rows are not duplicated into two independent "
            "datasets.",
            "Kp review rows are not training rows and cannot be substituted with permeability "
            "or BBB.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
