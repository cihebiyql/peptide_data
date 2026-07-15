from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_pk_semantic_tasks_v15 as pk  # noqa: E402
from scripts import run_assay_conditioned_baselines_v15 as conditioned  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "data"
    / "peptide_property_expansion_v15"
    / "peptide_property_training_deduplicated_v15.tsv"
)
DEFAULT_LITERATURE = (
    ROOT
    / "data"
    / "literature_data_v15_integration_audit_v1"
    / "normalized_candidates.tsv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "peptide_ml_cleaning_v1"
SCHEMA_VERSION = "peptide_ml_cleaning_v1"
CONVERSION_VERSION = "ml_unit_normalization_v1"

RAW_TO_FAMILY = {
    "LogD": "LogD",
    "LogD7.4": "LogD",
    "logd": "LogD",
    "solubility": "solubility",
    "F": "F",
    "T1/2": "T1/2",
    "half_life": "T1/2",
    "stability_half_life": "T1/2",
    "PPB": "PPB",
    "CL": "CL",
    "clearance": "CL",
    "Vd": "Vd",
    "volume_distribution": "Vd",
    "BBB": "BBB",
    "bbb_penetration": "BBB",
    "Kp": "Kp",
    "permeability": "permeability",
    "efflux_ratio": "permeability",
    "membrane_retention": "permeability",
    "cell_penetration": "cell_penetration",
}

BLANK_RELATION_EXACT_SOURCES = {
    "acs_jmedchem_5c00544",
    "acs_jmedchem_5c02090",
    "acs_jmedchem_5c01901",
    "swemacrocycledb_figshare_v1_ar_gt_0_3",
}

SENTINEL_COMPACT = {
    "na",
    "none",
    "null",
    "unknown",
    "notavailable",
    "notapplicable",
    "notreported",
}
VALID_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
STANDARD_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

OUTPUT_FIELDS = [
    "clean_row_id",
    "record_id",
    "member_record_ids",
    "member_count",
    "source_id",
    "source_ids",
    "source_record_id",
    "source_record_ids",
    "doi_pmid",
    "source_url",
    "license",
    "input_dataset",
    "member_lineage_json",
    "raw_endpoint",
    "raw_endpoint_detail",
    "raw_task_type",
    "raw_evidence_tier",
    "endpoint_family",
    "task_id",
    "task_kind",
    "task_semantic_signature",
    "parameter_semantics",
    "parameter_basis",
    "assay_family",
    "species_group",
    "matrix_group",
    "route_group",
    "raw_species",
    "raw_matrix",
    "raw_assay",
    "raw_route",
    "raw_dose",
    "raw_timepoint",
    "condition_json",
    "condition_group_id",
    "raw_value",
    "raw_relation",
    "raw_unit",
    "raw_label",
    "normalized_value",
    "normalized_lower",
    "normalized_upper",
    "normalized_relation",
    "relation_basis",
    "normalized_unit",
    "target_transform",
    "conversion_formula",
    "conversion_version",
    "censoring_type",
    "normalization_status",
    "normalization_reason",
    "binary_evidence_class",
    "binary_use_tier",
    "positive_class",
    "negative_class",
    "label_rule",
    "research_train_eligible",
    "production_train_eligible",
    "sequence",
    "helm",
    "smiles",
    "inchikey",
    "identity_type",
    "upstream_identity_key",
    "identity_alias_ids",
    "identity_group_id",
    "representation_type",
    "representation_text",
    "sequence_model_eligible",
    "structure_model_eligible",
    "source_group_ids",
    "upstream_observation_key",
    "measurement_condition_key",
    "source_condition_key",
    "normalized_observation_key",
    "condition_variability",
    "hard_condition_conflict",
    "partition",
    "review_reasons",
    "quality_flags",
    "context_json",
    "upstream_eligible_after_qc",
    "upstream_eligibility_reason",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def slug(value: Any, max_length: int = 56) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", clean(value).casefold()).strip("_")
    if not result:
        return "not_reported"
    if len(result) <= max_length:
        return result
    return result[: max_length - 9].rstrip("_") + "_" + sha256_text(result)[:8]


def is_sentinel(value: Any) -> bool:
    normalized = clean(value).casefold()
    if not normalized:
        return False
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    return compact in SENTINEL_COMPACT


def parse_context(value: Any) -> tuple[dict[str, Any], str]:
    raw = clean(value)
    if not raw:
        return {}, ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}, "invalid_context_json"
    if not isinstance(parsed, dict):
        return {}, "context_json_not_object"
    return parsed, ""


def iter_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Missing TSV header: {path}")
        for row in reader:
            yield {key: "" if value is None else value for key, value in row.items()}


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def parse_number(value: Any) -> float | None:
    try:
        number = float(clean(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def number_text(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def safe_identity_aliases(row: dict[str, str]) -> tuple[list[str], str, str, str, str]:
    aliases: set[str] = set()
    sequence = clean(row.get("sequence")).upper()
    helm = clean(row.get("helm"))
    smiles = clean(row.get("smiles"))
    inchikey = clean(row.get("inchikey")).upper()
    identity_type = clean(row.get("identity_type"))
    quality = clean(row.get("quality_flags"))

    valid_sequence = bool(STANDARD_SEQUENCE.fullmatch(sequence)) and not is_sentinel(sequence)
    valid_helm = bool(helm and "{" in helm and "}" in helm and not is_sentinel(helm))
    valid_smiles = bool(smiles and not is_sentinel(smiles))
    valid_inchikey = bool(VALID_INCHIKEY.fullmatch(inchikey))
    modified_projection = any(
        token in quality
        for token in (
            "residue_projection_only",
            "modified_or_noncanonical",
            "chirality_or_modification",
            "modified_sequence_preserved",
        )
    )
    safe_sequence = valid_sequence and not modified_projection

    if valid_inchikey:
        aliases.add(f"INCHIKEY:{inchikey}")
    if valid_helm:
        aliases.add(f"HELM_SHA256:{sha256_text(helm)}")
    if valid_smiles:
        aliases.add(f"SMILES_SHA256:{sha256_text(smiles)}")
    if safe_sequence and not (valid_helm or valid_smiles or valid_inchikey):
        aliases.add(f"SEQUENCE_SHA256:{sha256_text(sequence)}")

    if valid_helm:
        representation_type, representation_text = "helm", f"HELM:{helm}"
    elif valid_smiles:
        representation_type, representation_text = "smiles", f"SMILES:{smiles}"
    elif safe_sequence:
        representation_type, representation_text = "sequence", f"SEQ:{sequence}"
    else:
        representation_type, representation_text = "", ""

    sequence_eligible = str(
        representation_type == "sequence" and safe_sequence
    ).lower()
    structure_eligible = str(valid_helm or valid_smiles).lower()
    return (
        sorted(aliases),
        representation_type,
        representation_text,
        sequence_eligible,
        structure_eligible,
    )


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def assign_identity_groups(rows: list[dict[str, Any]]) -> None:
    union = UnionFind()
    for row in rows:
        aliases = row["_identity_aliases"]
        if not aliases:
            continue
        union.find(aliases[0])
        for alias in aliases[1:]:
            union.union(aliases[0], alias)
    members: dict[str, list[str]] = defaultdict(list)
    for alias in union.parent:
        members[union.find(alias)].append(alias)
    ids = {
        root: "MLIDENTITY1:" + sha256_text(stable_json(sorted(values)))[:24]
        for root, values in members.items()
    }
    for row in rows:
        aliases = row["_identity_aliases"]
        row["identity_group_id"] = ids[union.find(aliases[0])] if aliases else ""


def source_groups(row: dict[str, str]) -> list[str]:
    return pk.normalize_provenance_groups(row)


def relation_semantics(row: dict[str, str]) -> tuple[str, str]:
    relation = clean(row.get("relation"))
    if relation in {"=", "<", "<=", ">", ">=", "~"}:
        return relation, "reported_relation"
    if relation:
        return relation, "unsupported_relation"
    source = clean(row.get("source_id")).casefold()
    if source in BLANK_RELATION_EXACT_SOURCES:
        return "=", "implicit_exact_from_source_contract"
    return "", "blank_relation_not_assumed_exact"


def bounds_from_relation(value: float, relation: str) -> tuple[str, float | None, float | None]:
    if relation == "=":
        return "exact", value, value
    if relation in {"<", "<="}:
        return "upper_bound", None, value
    if relation in {">", ">="}:
        return "lower_bound", value, None
    if relation == "~":
        return "approximate", value, value
    return "unsupported", None, None


def transform_bounds(
    lower: float | None, upper: float | None, transform: Any
) -> tuple[float | None, float | None]:
    return (
        transform(lower) if lower is not None else None,
        transform(upper) if upper is not None else None,
    )


def canonical_common_condition(row: dict[str, str]) -> dict[str, str]:
    return {
        "species": conditioned.canonical_species(row),
        "matrix": conditioned.canonical_matrix(row),
        "route": slug(row.get("route")),
        "assay": slug(row.get("assay"), max_length=48),
        "dose": slug(row.get("dose"), max_length=48),
        "timepoint": slug(row.get("timepoint"), max_length=48),
    }


def task_id(family: str, signature: dict[str, Any]) -> tuple[str, str]:
    serialized = stable_json(signature)
    readable = ".".join(
        [
            "mlv1",
            slug(family, 24),
            slug(signature.get("target_kind"), 20),
            slug(signature.get("parameter_semantics"), 32),
        ]
    )
    return f"{readable}.{sha256_text(serialized)[:16]}", serialized


def normalized_result() -> dict[str, Any]:
    return {
        "task_kind": "",
        "parameter_semantics": "",
        "parameter_basis": "",
        "assay_family": "",
        "normalized_value": "",
        "normalized_lower": "",
        "normalized_upper": "",
        "normalized_relation": "",
        "relation_basis": "",
        "normalized_unit": "",
        "target_transform": "",
        "conversion_formula": "",
        "conversion_version": CONVERSION_VERSION,
        "censoring_type": "",
        "normalization_status": "review",
        "normalization_reason": "",
        "binary_evidence_class": "",
        "binary_use_tier": "",
        "positive_class": "",
        "negative_class": "",
        "research_train_eligible": "false",
        "production_train_eligible": "not_assessed_step7",
        "_condition": {},
        "_semantic_train": True,
    }


def normalize_time(row: dict[str, str], result: dict[str, Any]) -> None:
    value = parse_number(row.get("value"))
    unit = pk.normalize_unit_key(row.get("unit", ""))
    factors = {
        "s": 1 / 3600,
        "sec": 1 / 3600,
        "seconds": 1 / 3600,
        "min": 1 / 60,
        "h": 1.0,
        "d": 24.0,
        "day": 24.0,
        "days": 24.0,
    }
    if value is None or value <= 0:
        result["normalization_reason"] = "missing_nonfinite_or_nonpositive_time"
        return
    if unit not in factors:
        result["normalization_reason"] = "unsupported_or_ambiguous_time_unit"
        return
    relation, relation_basis = relation_semantics(row)
    censoring, lower, upper = bounds_from_relation(value * factors[unit], relation)
    if censoring == "unsupported":
        result["normalization_reason"] = relation_basis
        return
    lower, upper = transform_bounds(lower, upper, math.log10)
    value_h = value * factors[unit]
    noncomparable = conditioned.t1_noncomparable_reason(row)
    condition = conditioned.t1_condition(row)
    result.update(
        task_kind="regression",
        parameter_semantics=condition["assay_family"],
        parameter_basis="elapsed_time",
        assay_family=condition["assay_family"],
        normalized_value=number_text(math.log10(value_h)),
        normalized_lower=number_text(lower),
        normalized_upper=number_text(upper),
        normalized_relation=relation,
        relation_basis=relation_basis,
        normalized_unit="log10(h)",
        target_transform="log10",
        conversion_formula=f"log10(raw_value*{factors[unit]:.12g} h/{unit})",
        censoring_type=censoring,
        normalization_status="normalized" if not noncomparable else "review",
        normalization_reason=(f"noncomparable_t1_semantics:{noncomparable}" if noncomparable else ""),
        _condition={
            "species": condition["species_context"],
            "matrix": condition["matrix_context"],
            "route": condition["route_context"],
            "assay": condition["assay_context"],
            "protease": condition["protease_context"],
        },
    )


def normalize_pk(row: dict[str, str], result: dict[str, Any]) -> None:
    semantic = pk.build_semantic_row(row)
    relation, relation_basis = relation_semantics(row)
    relation_supported = relation in {"=", "<", "<=", ">", ">=", "~"}
    semantic_reasons = list(filter(None, clean(semantic["review_reasons"]).split(";")))
    if not relation_supported:
        semantic_reasons.append(relation_basis)
    result.update(
        task_kind=semantic["target_kind"],
        parameter_semantics=semantic["parameter_semantics"],
        parameter_basis=semantic["parameter_basis"],
        assay_family=semantic["analysis_method"],
        normalized_value=semantic["target_value"],
        normalized_relation=relation,
        relation_basis=relation_basis,
        normalized_unit=semantic["target_unit"],
        target_transform="identity",
        conversion_formula=semantic["normalization_basis"],
        censoring_type=(
            bounds_from_relation(parse_number(semantic["target_value"]) or 0.0, relation)[0]
            if relation_supported
            else "unsupported"
        ),
        normalization_status=(
            "normalized" if semantic["target_value"] and relation_supported else "review"
        ),
        normalization_reason=";".join(sorted(set(semantic_reasons))),
        _semantic_train=semantic["layer"] == "train",
        _condition={
            "species": semantic["species_group"],
            "matrix": slug(semantic["matrix"]),
            "route": semantic["route_group"],
            "dose": slug(row.get("dose"), max_length=48),
            "timepoint": slug(row.get("timepoint"), max_length=48),
            "formulation": semantic["formulation_group"],
            "method": semantic["analysis_method"],
            "analyte": semantic["analyte_scope"],
            "volume_subtype": semantic["volume_subtype"],
        },
    )
    value = parse_number(semantic["target_value"])
    if value is not None and relation_supported:
        censoring, lower, upper = bounds_from_relation(value, relation)
        result["censoring_type"] = censoring
        result["normalized_lower"] = number_text(lower)
        result["normalized_upper"] = number_text(upper)


def normalize_ppb(row: dict[str, str], result: dict[str, Any]) -> None:
    value = parse_number(row.get("value"))
    unit = pk.normalize_unit_key(row.get("unit", ""))
    if value is None:
        result["normalization_reason"] = "missing_numeric_ppb"
        return
    if unit in {"%", "%bound", "percent", "percentbound"}:
        factor, formula = 0.01, "percent_bound/100"
    elif unit == "fraction":
        factor, formula = 1.0, "fraction_bound_identity"
    else:
        result["normalization_reason"] = "unsupported_or_direction_ambiguous_ppb_unit"
        return
    normalized = value * factor
    if not 0 <= normalized <= 1:
        result["normalization_reason"] = "ppb_fraction_out_of_range"
        return
    relation, basis = relation_semantics(row)
    censoring, lower, upper = bounds_from_relation(normalized, relation)
    if censoring == "unsupported":
        result["normalization_reason"] = basis
        return
    common = canonical_common_condition(row)
    result.update(
        task_kind="regression",
        parameter_semantics="fraction_bound",
        parameter_basis="plasma_or_serum_protein_binding",
        assay_family="ppb",
        normalized_value=number_text(normalized),
        normalized_lower=number_text(lower),
        normalized_upper=number_text(upper),
        normalized_relation=relation,
        relation_basis=basis,
        normalized_unit="fraction_bound",
        target_transform="identity",
        conversion_formula=formula,
        censoring_type=censoring,
        normalization_status="normalized",
        normalization_reason="",
        _condition=common,
    )


def normalize_logd(row: dict[str, str], result: dict[str, Any]) -> None:
    value = parse_number(row.get("value"))
    unit = clean(row.get("unit")).casefold()
    if value is None or unit != "dimensionless":
        result["normalization_reason"] = "logd_requires_dimensionless_numeric_value"
        return
    relation, basis = relation_semantics(row)
    censoring, lower, upper = bounds_from_relation(value, relation)
    if censoring == "unsupported":
        result["normalization_reason"] = basis
        return
    text = " ".join((clean(row.get("endpoint")), clean(row.get("endpoint_detail")))).casefold()
    ph = "7.4" if "7.4" in text or "7_4" in text else "not_reported"
    assay_text = " ".join((clean(row.get("assay")), clean(row.get("endpoint_detail")), clean(row.get("source_id")))).casefold()
    method = "chi_hplc" if "chromlog" in assay_text or "pegasus" in assay_text else "shake_flask" if "shake" in assay_text else "source_reported_method_unknown"
    result.update(
        task_kind="regression",
        parameter_semantics="distribution_coefficient",
        parameter_basis=f"pH_{ph}",
        assay_family=method,
        normalized_value=number_text(value),
        normalized_lower=number_text(lower),
        normalized_upper=number_text(upper),
        normalized_relation=relation,
        relation_basis=basis,
        normalized_unit="dimensionless_log10",
        target_transform="identity",
        conversion_formula="dimensionless_logd_identity",
        censoring_type=censoring,
        normalization_status="normalized",
        normalization_reason="",
        _condition={**canonical_common_condition(row), "pH": ph, "method": method},
    )


def normalize_solubility(row: dict[str, str], context: dict[str, Any], result: dict[str, Any]) -> None:
    value = parse_number(row.get("value"))
    unit_raw = clean(row.get("unit"))
    if value is None or value <= 0:
        result["normalization_reason"] = "missing_or_nonpositive_solubility_value"
        return
    relation, basis = relation_semantics(row)
    censoring, lower, upper = bounds_from_relation(value, relation)
    if censoring == "unsupported":
        result["normalization_reason"] = basis
        return
    unit_key = pk.normalize_unit_key(unit_raw)
    if unit_key in {"um", "umol/l", "umolar"}:
        factor = 1e-6
        target_unit = "log10(mol/L)"
        formula = "log10(raw_uM*1e-6)"
    elif unit_key in {"m", "mol/l"}:
        factor = 1.0
        target_unit = "log10(mol/L)"
        formula = "log10(raw_mol_per_l)"
    else:
        factor = 1.0
        target_unit = f"log10({slug(unit_raw, 32)})"
        formula = "log10(raw_value_in_source_dimension);no_cross_unit_conversion"
    normalized = math.log10(value * factor)
    lower, upper = transform_bounds(
        lower,
        upper,
        lambda item: math.log10(item * factor),
    )
    solvent = clean(context.get("matrix")) or clean(row.get("matrix")) or "not_reported"
    raw_text = clean(context.get("source_raw_text"))
    temperature = "not_reported"
    match = re.search(r"Temperature,? K[^0-9]*(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)
    if match:
        temperature = match.group(1)
    result.update(
        task_kind="regression",
        parameter_semantics="solubility",
        parameter_basis=slug(unit_raw, 32),
        assay_family=slug(row.get("assay"), 36),
        normalized_value=number_text(normalized),
        normalized_lower=number_text(lower),
        normalized_upper=number_text(upper),
        normalized_relation=relation,
        relation_basis=basis,
        normalized_unit=target_unit,
        target_transform="log10",
        conversion_formula=formula,
        censoring_type=censoring,
        normalization_status="normalized",
        normalization_reason="",
        _condition={
            **canonical_common_condition(row),
            "solvent": slug(solvent, 48),
            "temperature_k": temperature,
            "source_dimension": slug(unit_raw, 32),
        },
    )


def permeability_assay(row: dict[str, str]) -> str:
    text = " ".join((clean(row.get("endpoint_detail")), clean(row.get("assay")))).casefold().replace("-", "")
    for pattern, label in (
        ("caco2", "caco2"),
        ("pampa", "pampa"),
        ("mdck", "mdck"),
        ("rrck", "rrck"),
    ):
        if pattern in text:
            return label
    return slug(row.get("endpoint_detail"), 40)


def normalize_permeability(row: dict[str, str], result: dict[str, Any]) -> None:
    value = parse_number(row.get("value"))
    if value is None:
        result["normalization_reason"] = "missing_numeric_permeability_target"
        return
    raw_endpoint = clean(row.get("endpoint"))
    unit = pk.normalize_unit_key(row.get("unit", ""))
    if raw_endpoint == "efflux_ratio":
        if value <= 0:
            result["normalization_reason"] = "nonpositive_efflux_ratio"
            return
        transform = math.log10
        normalized = transform(value)
        target_unit = "log10(ratio)"
        formula = "log10(efflux_ratio)"
        semantics = "efflux_ratio"
    elif raw_endpoint == "membrane_retention":
        normalized = value / 100.0
        transform = lambda item: item / 100.0
        target_unit = "fraction_retained"
        formula = "percent/100"
        semantics = "membrane_retention"
    elif unit in {"source_log10_permeability", "log10(cm/s)"}:
        normalized = value
        transform = lambda item: item
        target_unit = unit
        formula = "already_log10_no_retransform"
        semantics = "apparent_permeability"
    elif unit == "10^-6cm/s":
        if value <= 0:
            result["normalization_reason"] = "nonpositive_linear_permeability"
            return
        normalized = math.log10(value * 1e-6)
        transform = lambda item: math.log10(item * 1e-6)
        target_unit = "log10(cm/s)"
        formula = "log10(raw_value*1e-6 cm/s)"
        semantics = "apparent_permeability"
    else:
        result["normalization_reason"] = "unsupported_permeability_unit"
        return
    relation, basis = relation_semantics(row)
    censoring, lower, upper = bounds_from_relation(value, relation)
    if censoring == "unsupported":
        result["normalization_reason"] = basis
        return
    lower, upper = transform_bounds(lower, upper, transform)
    assay_family = permeability_assay(row)
    result.update(
        task_kind="regression",
        parameter_semantics=semantics,
        parameter_basis=slug(row.get("endpoint_detail"), 48),
        assay_family=assay_family,
        normalized_value=number_text(normalized),
        normalized_lower=number_text(lower),
        normalized_upper=number_text(upper),
        normalized_relation=relation,
        relation_basis=basis,
        normalized_unit=target_unit,
        target_transform="log10" if "log10" in target_unit else "identity",
        conversion_formula=formula,
        censoring_type=censoring,
        normalization_status="normalized",
        normalization_reason="",
        _condition={
            **canonical_common_condition(row),
            "assay_family": assay_family,
            "metric_detail": slug(row.get("endpoint_detail"), 48),
        },
    )


def binary_evidence(row: dict[str, str], context: dict[str, Any], result: dict[str, Any]) -> None:
    label = clean(row.get("label"))
    endpoint = clean(row.get("endpoint"))
    origin = clean(context.get("label_origin"))
    if label not in {"0", "1"}:
        result["normalization_reason"] = "binary_label_not_0_or_1"
        return
    condition = canonical_common_condition(row)
    if endpoint == "T1/2" and "threshold_1h" in clean(row.get("endpoint_detail")):
        evidence = "threshold_derived_from_numeric"
        use = "derived_binary_research"
        positive = "half-life >= 1 h"
        negative = "half-life < 1 h"
        research = True
        semantics = "stability_threshold_1h"
    elif endpoint == "solubility":
        evidence = "source_defined_binary"
        use = "rough_solvent_conditioned_benchmark"
        positive = "source-defined soluble"
        negative = "source-defined insoluble"
        research = True
        semantics = "source_defined_solubility"
        condition["solvent"] = slug(context.get("matrix") or row.get("matrix"), 48)
    elif endpoint == "BBB":
        if label == "0" or "rule_constructed" in origin or "rule_constructed" in clean(row.get("quality_flags")):
            evidence = "rule_constructed_negative"
        else:
            evidence = "database_positive_membership"
        use = "weak_benchmark_reproduction_only"
        positive = "compiled BBB-positive peptide"
        negative = "rule-constructed weak negative"
        research = True
        semantics = "BBB_compiled_positive_rule_negative"
    else:
        evidence = "source_defined_binary"
        use = "source_defined_binary_review"
        positive = "source-defined positive"
        negative = "source-defined negative"
        research = False
        semantics = slug(row.get("endpoint_detail"), 48)
    result.update(
        task_kind="binary_classification",
        parameter_semantics=semantics,
        parameter_basis="binary_label",
        assay_family=slug(row.get("assay"), 36),
        normalized_value=label,
        normalized_lower="",
        normalized_upper="",
        normalized_relation="label",
        relation_basis="explicit_binary_label",
        normalized_unit="binary_label",
        target_transform="identity",
        conversion_formula="explicit_label_mapping",
        censoring_type="not_applicable",
        normalization_status="normalized",
        normalization_reason="",
        binary_evidence_class=evidence,
        binary_use_tier=use,
        positive_class=positive,
        negative_class=negative,
        research_train_eligible=str(research).lower(),
        _condition=condition,
    )


def positive_only(row: dict[str, str], result: dict[str, Any]) -> None:
    result.update(
        task_kind="positive_unlabeled",
        parameter_semantics=slug(row.get("endpoint_detail"), 48),
        parameter_basis="database_function_membership",
        assay_family=slug(row.get("assay"), 36),
        normalized_value="1",
        normalized_relation="label",
        relation_basis="positive_membership_only",
        normalized_unit="positive_membership",
        target_transform="identity",
        conversion_formula="positive_membership_only_no_negative_inference",
        censoring_type="not_applicable",
        normalization_status="normalized",
        normalization_reason="positive_only_unlabeled",
        binary_evidence_class="positive_only_unlabeled",
        binary_use_tier="positive_unlabeled_only",
        positive_class="database-curated positive membership",
        negative_class="not available; absence is not negative",
        research_train_eligible="false",
        _condition=canonical_common_condition(row),
    )


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    family = RAW_TO_FAMILY[row["endpoint"]]
    context, context_error = parse_context(row.get("context_json"))
    aliases, representation_type, representation_text, sequence_ok, structure_ok = safe_identity_aliases(row)
    result = normalized_result()

    if row.get("label"):
        if row["endpoint"] in {"bbb_penetration", "cell_penetration"}:
            positive_only(row, result)
        else:
            binary_evidence(row, context, result)
    elif family == "T1/2":
        normalize_time(row, result)
    elif family in {"F", "CL", "Vd"}:
        normalize_pk(row, result)
    elif family == "PPB":
        normalize_ppb(row, result)
    elif family == "LogD":
        normalize_logd(row, result)
    elif family == "solubility":
        normalize_solubility(row, context, result)
    elif family == "permeability":
        normalize_permeability(row, result)
    elif family in {"BBB", "Kp"}:
        value = parse_number(row.get("value"))
        relation, basis = relation_semantics(row)
        if value is not None and relation in {"=", "<", "<=", ">", ">=", "~"}:
            censoring, lower, upper = bounds_from_relation(value, relation)
            result.update(
                task_kind="regression",
                parameter_semantics=slug(row.get("endpoint_detail"), 48),
                parameter_basis="source_defined_metric",
                assay_family=slug(row.get("assay"), 36),
                normalized_value=number_text(value),
                normalized_lower=number_text(lower),
                normalized_upper=number_text(upper),
                normalized_relation=relation,
                relation_basis=basis,
                normalized_unit=clean(row.get("unit")) or "source_unit_not_reported",
                target_transform="identity",
                conversion_formula="source_value_identity_no_semantic_pooling",
                censoring_type=censoring,
                normalization_status="review",
                normalization_reason="BBB_or_Kp_metric_requires_semantic_review",
                _condition=canonical_common_condition(row),
            )
        else:
            result["normalization_reason"] = basis or "missing_target"
    else:
        result["normalization_reason"] = "unsupported_endpoint_family"

    condition = result.pop("_condition") or canonical_common_condition(row)
    signature = {
        "endpoint_family": family,
        "target_kind": result["task_kind"] or "review",
        "parameter_semantics": result["parameter_semantics"] or "unresolved",
        "parameter_basis": result["parameter_basis"] or "unresolved",
        "assay_family": result["assay_family"] or "not_reported",
        "normalized_unit": result["normalized_unit"] or "not_normalized",
        "target_transform": result["target_transform"] or "none",
        "condition": condition,
        "binary_evidence_lane": result["binary_use_tier"] or "not_binary",
    }
    task, signature_text = task_id(family, signature)
    source_group_list = source_groups(row)
    reasons = [reason for reason in (context_error, result["normalization_reason"]) if reason]
    if not aliases:
        reasons.append("missing_valid_exact_identity")
    if not representation_text:
        reasons.append("no_valid_model_representation")
    semantic_train = result.pop("_semantic_train")
    if not semantic_train:
        reasons.append("semantic_or_source_evidence_review")
    upstream_eligible = clean(row.get("eligible_after_qc")) == "true"
    if not upstream_eligible:
        upstream_reason = slug(row.get("eligibility_reason"), 72)
        reasons.append(f"upstream_qc_ineligible:{upstream_reason}")

    output = {field: "" for field in OUTPUT_FIELDS}
    output.update(
        clean_row_id="MLCLEAN1:" + sha256_text(row["record_id"] + task)[:24],
        record_id=row["record_id"],
        member_record_ids=row["record_id"],
        member_count=1,
        source_id=row["source_id"],
        source_ids=row["source_id"],
        source_record_id=row["source_record_id"],
        source_record_ids=row["source_record_id"],
        doi_pmid=row["doi_pmid"],
        source_url=row["source_url"],
        license=row["license"],
        input_dataset=row["input_dataset"],
        raw_endpoint=row["endpoint"],
        raw_endpoint_detail=clean(row.get("endpoint_detail")),
        raw_task_type=clean(row.get("task_type")),
        raw_evidence_tier=clean(row.get("evidence_tier")),
        endpoint_family=family,
        task_id=task,
        task_semantic_signature=signature_text,
        species_group=condition.get("species", "not_reported"),
        matrix_group=condition.get("matrix", "not_reported"),
        route_group=condition.get("route", "not_reported"),
        raw_species=clean(row.get("species")),
        raw_matrix=clean(row.get("matrix")),
        raw_assay=clean(row.get("assay")),
        raw_route=clean(row.get("route")),
        raw_dose=clean(row.get("dose")),
        raw_timepoint=clean(row.get("timepoint")),
        condition_json=stable_json(condition),
        condition_group_id="MLCOND1:" + sha256_text(stable_json([task, condition]))[:24],
        raw_value=clean(row.get("value")),
        raw_relation=clean(row.get("relation")),
        raw_unit=clean(row.get("unit")),
        raw_label=clean(row.get("label")),
        label_rule=clean(row.get("label_rule")),
        sequence=clean(row.get("sequence")),
        helm=clean(row.get("helm")),
        smiles=clean(row.get("smiles")),
        inchikey=clean(row.get("inchikey")),
        identity_type=clean(row.get("identity_type")),
        upstream_identity_key=clean(row.get("identity_key")),
        identity_alias_ids=";".join(aliases),
        representation_type=representation_type,
        representation_text=representation_text,
        sequence_model_eligible=sequence_ok,
        structure_model_eligible=structure_ok,
        source_group_ids=";".join(source_group_list),
        upstream_observation_key=clean(row.get("observation_key")),
        review_reasons=";".join(sorted(set(filter(None, reasons)))),
        quality_flags=clean(row.get("quality_flags")),
        context_json=stable_json(context),
        upstream_eligible_after_qc=str(upstream_eligible).lower(),
        upstream_eligibility_reason=clean(row.get("eligibility_reason")),
        **{key: value for key, value in result.items() if not key.startswith("_")},
    )
    output["_identity_aliases"] = aliases
    output["_source_groups"] = source_group_list
    output["_upstream_eligible"] = upstream_eligible
    output["member_lineage_json"] = stable_json(
        [
            {
                "context_json": output["context_json"],
                "doi_pmid": output["doi_pmid"],
                "evidence_tier": output["raw_evidence_tier"],
                "input_dataset": output["input_dataset"],
                "license": output["license"],
                "quality_flags": output["quality_flags"],
                "record_id": output["record_id"],
                "review_reasons": output["review_reasons"],
                "source_group_ids": source_group_list,
                "source_id": output["source_id"],
                "source_record_id": output["source_record_id"],
                "source_url": output["source_url"],
                "upstream_observation_key": output["upstream_observation_key"],
            }
        ]
    )
    return output


def finalize_keys(rows: list[dict[str, Any]]) -> None:
    assign_identity_groups(rows)
    for row in rows:
        base = [row["identity_group_id"], row["task_id"], row["condition_group_id"]]
        row["measurement_condition_key"] = (
            "MLMEASURE1:" + sha256_text(stable_json(base))[:24]
            if row["identity_group_id"]
            else ""
        )
        source_group_list = row["_source_groups"] or [
            f"SOURCE:{slug(row['source_id'])}"
        ]
        source_condition_keys = (
            [
                "MLSOURCECOND1:"
                + sha256_text(stable_json(base + [source_group]))[:24]
                for source_group in source_group_list
            ]
            if row["identity_group_id"]
            else []
        )
        row["_source_condition_keys"] = source_condition_keys
        row["source_condition_key"] = ";".join(source_condition_keys)
        target = row["normalized_value"] or row["raw_label"]
        normalized_payload = base + [
            row["task_kind"],
            row["normalized_relation"],
            target,
            row["normalized_lower"],
            row["normalized_upper"],
        ]
        row["normalized_observation_key"] = (
            "MLOBS1:" + sha256_text(stable_json(normalized_payload))[:24]
            if row["identity_group_id"] and target
            else ""
        )


def collapse_normalized_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    for row in rows:
        key = row["normalized_observation_key"]
        if key:
            grouped[key].append(row)
        else:
            unresolved.append(row)
    representatives: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = grouped[key]
        representative = dict(
            min(
                members,
                key=lambda row: (
                    not row["_upstream_eligible"],
                    bool(row["review_reasons"]),
                    row["record_id"],
                ),
            )
        )
        representative["member_record_ids"] = ";".join(sorted({row["record_id"] for row in members}))
        representative["source_record_ids"] = ";".join(sorted({row["source_record_id"] for row in members if row["source_record_id"]}))
        representative["source_ids"] = ";".join(sorted({row["source_id"] for row in members if row["source_id"]}))
        representative["doi_pmid"] = " | ".join(
            sorted({row["doi_pmid"] for row in members if row["doi_pmid"]})
        )
        representative["source_url"] = " | ".join(
            sorted({row["source_url"] for row in members if row["source_url"]})
        )
        representative["license"] = " | ".join(
            sorted({row["license"] for row in members if row["license"]})
        )
        representative["input_dataset"] = " | ".join(
            sorted({row["input_dataset"] for row in members if row["input_dataset"]})
        )
        all_source_groups = sorted(
            {group for row in members for group in row["_source_groups"]}
        )
        representative["source_group_ids"] = ";".join(all_source_groups)
        representative["_source_groups"] = all_source_groups
        all_identity_aliases = sorted(
            {alias for row in members for alias in row["_identity_aliases"]}
        )
        representative["identity_alias_ids"] = ";".join(all_identity_aliases)
        representative["_identity_aliases"] = all_identity_aliases
        representative["upstream_observation_key"] = " | ".join(
            sorted(
                {
                    row["upstream_observation_key"]
                    for row in members
                    if row["upstream_observation_key"]
                }
            )
        )
        member_lineage = sorted(
            [json.loads(row["member_lineage_json"])[0] for row in members],
            key=lambda item: (item["record_id"], item["source_id"], item["source_record_id"]),
        )
        representative["member_lineage_json"] = stable_json(member_lineage)
        representative["member_count"] = len(members)
        representatives.append(representative)
        if len(members) > 1:
            duplicate_rows.append(
                {
                    "normalized_observation_key": key,
                    "representative_record_id": representative["record_id"],
                    "member_count": len(members),
                    "member_record_ids": representative["member_record_ids"],
                    "source_ids": representative["source_ids"],
                    "raw_values_json": stable_json(
                        sorted(
                            {
                                (row["raw_value"], row["raw_relation"], row["raw_unit"], row["raw_label"])
                                for row in members
                            }
                        )
                    ),
                    "member_lineage_json": representative["member_lineage_json"],
                    "scope": "unit_and_task_normalized_exact_observation_duplicate",
                }
            )
    representatives.extend(unresolved)
    representatives.sort(key=lambda row: row["record_id"])
    return representatives, duplicate_rows


def flag_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["measurement_condition_key"]:
            by_condition[row["measurement_condition_key"]].append(row)
        for source_condition_key in row["_source_condition_keys"]:
            by_source_condition[source_condition_key].append(row)

    variable = {
        key
        for key, members in by_condition.items()
        if len(
            {
                (row["normalized_value"], row["normalized_relation"], row["raw_label"])
                for row in members
            }
        )
        > 1
    }
    hard = {
        key
        for key, members in by_source_condition.items()
        if len(
            {
                (row["normalized_value"], row["normalized_relation"], row["raw_label"])
                for row in members
            }
        )
        > 1
    }
    conflicts: list[dict[str, Any]] = []
    for row in rows:
        row["condition_variability"] = str(row["measurement_condition_key"] in variable).lower()
        row["hard_condition_conflict"] = str(
            any(key in hard for key in row["_source_condition_keys"])
        ).lower()
    for key in sorted(variable):
        members = by_condition[key]
        conflicts.append(
            {
                "measurement_condition_key": key,
                "hard_within_source_conflict": str(
                    any(
                        key in hard
                        for row in members
                        for key in row["_source_condition_keys"]
                    )
                ).lower(),
                "rows": len(members),
                "record_ids": ";".join(sorted(row["record_id"] for row in members)),
                "source_ids": ";".join(sorted({row["source_id"] for row in members})),
                "targets_json": stable_json(
                    sorted(
                        {
                            (row["normalized_value"], row["normalized_relation"], row["raw_label"])
                            for row in members
                        }
                    )
                ),
                "interpretation": "replicate_variability_or_condition_conflict_review",
            }
        )
    return conflicts


def assign_partition(row: dict[str, Any]) -> None:
    reasons = set(filter(None, row["review_reasons"].split(";")))
    if row["hard_condition_conflict"] == "true":
        reasons.add("hard_condition_conflict")
    if not row["identity_group_id"] or not row["representation_text"]:
        partition = "identity_review"
    elif row["normalization_status"] != "normalized":
        partition = "normalization_review"
    elif row["task_kind"] == "regression" and row["censoring_type"] in {
        "upper_bound",
        "lower_bound",
        "approximate",
    }:
        partition = "censored_numeric"
    elif row["task_kind"] == "regression" and row["censoring_type"] == "exact":
        partition = (
            "strict_numeric"
            if row["_upstream_eligible"] and not reasons
            else "semantic_review"
        )
    elif row["task_kind"] == "binary_classification":
        partition = "binary_evidence_catalog"
    elif row["task_kind"] == "positive_unlabeled":
        partition = "positive_unlabeled"
    else:
        partition = "semantic_review"
    row["partition"] = partition
    row["review_reasons"] = ";".join(sorted(reasons))
    if partition == "strict_numeric":
        row["research_train_eligible"] = "true"
    elif partition != "binary_evidence_catalog" or (
        not row["_upstream_eligible"]
        or not row["identity_group_id"]
        or not row["representation_text"]
        or row["hard_condition_conflict"] == "true"
    ):
        row["research_train_eligible"] = "false"


def literature_cleaned(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in iter_tsv(path):
        sequence = clean(source.get("sequence")) or clean(source.get("sequence_raw"))
        mock = {
            "sequence": sequence,
            "helm": "",
            "smiles": clean(source.get("smiles")),
            "inchikey": "",
            "identity_type": clean(source.get("identity_type")),
            "quality_flags": clean(source.get("decision_reason")),
        }
        aliases, rep_type, rep_text, seq_ok, structure_ok = safe_identity_aliases(mock)
        value = parse_number(source.get("value"))
        normalized = ""
        if clean(source.get("endpoint")) == "T1/2" and value is not None and clean(source.get("unit")).casefold() in {"h", "hour", "hours"} and value > 0:
            normalized = number_text(math.log10(value))
        context, _ = parse_context(source.get("context_json"))
        rows.append(
            {
                "candidate_id": source["candidate_id"],
                "source_family": source["source_family"],
                "source_record_id": source["source_record_id"],
                "endpoint": source["endpoint"],
                "endpoint_detail": source["endpoint_detail"],
                "raw_value": source["value"],
                "raw_unit": source["unit"],
                "raw_label": source["label"],
                "normalized_value": normalized,
                "normalized_unit": "log10(h)" if normalized else "",
                "identity_alias_ids": ";".join(aliases),
                "representation_type": rep_type,
                "representation_text": rep_text,
                "sequence_model_eligible": seq_ok,
                "structure_model_eligible": structure_ok,
                "context_json": stable_json(context),
                "v15_relation": source["v15_relation"],
                "decision": source["decision"],
                "partition": "literature_review",
                "research_train_eligible": "false",
                "production_train_eligible": "false",
                "review_reason": "literature_staging_not_promoted_by_steps_1_to_6",
            }
        )
    return rows


def artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": display_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def build(
    input_path: Path = DEFAULT_INPUT,
    literature_path: Path = DEFAULT_LITERATURE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    literature_path = literature_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {ROOT.resolve(), input_path, literature_path}:
        raise ValueError("Unsafe output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_sha_before = sha256_file(input_path)

    source_rows = [
        row for row in iter_tsv(input_path) if row.get("endpoint") in RAW_TO_FAMILY
    ]
    normalized = [normalize_row(row) for row in source_rows]
    finalize_keys(normalized)
    representatives, duplicate_rows = collapse_normalized_duplicates(normalized)
    # Recompute provenance-aware keys after duplicate members merge their source groups.
    finalize_keys(representatives)
    representative_by_record = {row["record_id"]: row for row in representatives}
    for duplicate in duplicate_rows:
        representative = representative_by_record[duplicate["representative_record_id"]]
        duplicate["normalized_observation_key"] = representative[
            "normalized_observation_key"
        ]
    conflicts = flag_conflicts(representatives)
    for row in representatives:
        assign_partition(row)

    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        partitions[row["partition"]].append(row)
    for rows in partitions.values():
        rows.sort(key=lambda row: (row["task_id"], row["identity_group_id"], row["record_id"]))

    registry: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        by_task[row["task_id"]].append(row)
    for task in sorted(by_task):
        rows = by_task[task]
        signatures = {row["task_semantic_signature"] for row in rows}
        if len(signatures) != 1:
            raise ValueError(f"Task semantic collision: {task}")
        strict_numeric_count = sum(
            row["partition"] == "strict_numeric" for row in rows
        )
        positive_count = sum(row["raw_label"] == "1" for row in rows)
        negative_count = sum(row["raw_label"] == "0" for row in rows)
        binary_classes = {
            row["binary_evidence_class"]
            for row in rows
            if row["binary_evidence_class"]
        }
        strict_experimental_binary = binary_classes == {
            "experimental_positive",
            "experimental_negative",
        }
        strict_modelable = strict_numeric_count >= 30 or (
            strict_experimental_binary
            and positive_count >= 20
            and negative_count >= 20
        )
        research_modelable = strict_modelable or (
            rows[0]["task_kind"] == "binary_classification"
            and positive_count >= 20
            and negative_count >= 20
        )
        if strict_modelable:
            modelability_reason = "strict_numeric_or_experimental_binary_size_gate"
        elif research_modelable:
            modelability_reason = "derived_or_weak_binary_research_benchmark_only"
        else:
            modelability_reason = "below_current_per_task_size_gate"
        registry.append(
            {
                "task_id": task,
                "endpoint_family": rows[0]["endpoint_family"],
                "task_kind": rows[0]["task_kind"],
                "parameter_semantics": rows[0]["parameter_semantics"],
                "normalized_unit": rows[0]["normalized_unit"],
                "target_transform": rows[0]["target_transform"],
                "binary_evidence_classes": ";".join(
                    sorted(
                        {
                            row["binary_evidence_class"]
                            for row in rows
                            if row["binary_evidence_class"]
                        }
                    )
                ),
                "binary_use_tiers": ";".join(
                    sorted(
                        {
                            row["binary_use_tier"]
                            for row in rows
                            if row["binary_use_tier"]
                        }
                    )
                ),
                "semantic_signature": rows[0]["task_semantic_signature"],
                "rows": len(rows),
                "strict_numeric_rows": strict_numeric_count,
                "censored_rows": sum(row["partition"] == "censored_numeric" for row in rows),
                "positive_rows": positive_count,
                "negative_rows": negative_count,
                "unique_identities": len({row["identity_group_id"] for row in rows if row["identity_group_id"]}),
                "source_groups": len({group for row in rows for group in row["_source_groups"]}),
                "strict_modelable": str(strict_modelable).lower(),
                "research_modelable": str(research_modelable).lower(),
                "modelability_reason": modelability_reason,
            }
        )

    literature = literature_cleaned(literature_path)
    outputs: dict[str, dict[str, Any]] = {}

    def emit(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
        path = output_dir / name
        count = write_tsv(path, rows, fields)
        outputs[name] = artifact(path, count)

    representatives.sort(key=lambda row: (row["partition"], row["task_id"], row["record_id"]))
    emit("normalized_observations.tsv", representatives, OUTPUT_FIELDS)
    emit("strict_numeric.tsv", partitions["strict_numeric"], OUTPUT_FIELDS)
    emit("censored_observations.tsv", partitions["censored_numeric"], OUTPUT_FIELDS)
    emit("binary_evidence_catalog.tsv", partitions["binary_evidence_catalog"], OUTPUT_FIELDS)
    emit("positive_unlabeled.tsv", partitions["positive_unlabeled"], OUTPUT_FIELDS)
    review_rows = [
        row
        for partition in ("identity_review", "normalization_review", "semantic_review")
        for row in partitions[partition]
    ]
    review_rows.sort(key=lambda row: (row["partition"], row["task_id"], row["record_id"]))
    emit("review_observations.tsv", review_rows, OUTPUT_FIELDS)
    emit(
        "exact_duplicates.tsv",
        duplicate_rows,
        [
            "normalized_observation_key",
            "representative_record_id",
            "member_count",
            "member_record_ids",
            "source_ids",
            "raw_values_json",
            "member_lineage_json",
            "scope",
        ],
    )
    emit(
        "condition_conflicts.tsv",
        conflicts,
        [
            "measurement_condition_key",
            "hard_within_source_conflict",
            "rows",
            "record_ids",
            "source_ids",
            "targets_json",
            "interpretation",
        ],
    )
    emit(
        "task_registry.tsv",
        registry,
        [
            "task_id",
            "endpoint_family",
            "task_kind",
            "parameter_semantics",
            "normalized_unit",
            "target_transform",
            "binary_evidence_classes",
            "binary_use_tiers",
            "semantic_signature",
            "rows",
            "strict_numeric_rows",
            "censored_rows",
            "positive_rows",
            "negative_rows",
            "unique_identities",
            "source_groups",
            "strict_modelable",
            "research_modelable",
            "modelability_reason",
        ],
    )
    emit(
        "literature_review_cleaned.tsv",
        literature,
        [
            "candidate_id",
            "source_family",
            "source_record_id",
            "endpoint",
            "endpoint_detail",
            "raw_value",
            "raw_unit",
            "raw_label",
            "normalized_value",
            "normalized_unit",
            "identity_alias_ids",
            "representation_type",
            "representation_text",
            "sequence_model_eligible",
            "structure_model_eligible",
            "context_json",
            "v15_relation",
            "decision",
            "partition",
            "research_train_eligible",
            "production_train_eligible",
            "review_reason",
        ],
    )

    exclusion_counts = Counter(
        reason
        for row in review_rows
        for reason in row["review_reasons"].split(";")
        if reason
    )
    exclusion_summary = [
        {"reason": reason, "rows": count}
        for reason, count in sorted(exclusion_counts.items())
    ]
    emit("exclusion_summary.tsv", exclusion_summary, ["reason", "rows"])

    strict = partitions["strict_numeric"]
    binary = partitions["binary_evidence_catalog"]
    assertions = {
        "input_hash_unchanged": sha256_file(input_path) == input_sha_before,
        "row_conservation": sum(int(row["member_count"]) for row in representatives)
        == len(source_rows),
        "all_context_json_objects": all(
            isinstance(json.loads(row["context_json"]), dict) for row in representatives
        ),
        "all_rows_have_task": all(row["task_id"] for row in representatives),
        "task_semantic_signature_one_to_one": all(
            len({row["task_semantic_signature"] for row in by_task[task]}) == 1
            for task in by_task
        ),
        "normalized_keys_unique": len(
            [row["normalized_observation_key"] for row in representatives if row["normalized_observation_key"]]
        )
        == len(
            {
                row["normalized_observation_key"]
                for row in representatives
                if row["normalized_observation_key"]
            }
        ),
        "duplicate_member_lineage_complete": all(
            len(json.loads(row["member_lineage_json"])) == int(row["member_count"])
            for row in representatives
        ),
        "duplicate_keys_reference_final_observations": {
            row["normalized_observation_key"] for row in duplicate_rows
        }.issubset(
            {
                row["normalized_observation_key"]
                for row in representatives
                if row["normalized_observation_key"]
            }
        ),
        "no_selected_identity_sentinels": all(
            not is_sentinel(row["representation_text"].split(":", 1)[-1])
            for row in representatives
            if row["representation_text"]
        ),
        "strict_numeric_exact_only": all(
            row["normalized_relation"] == "=" and row["censoring_type"] == "exact"
            for row in strict
        ),
        "strict_numeric_has_valid_representation": all(
            row["identity_group_id"] and row["representation_text"] for row in strict
        ),
        "strict_numeric_upstream_qc_eligible": all(
            row["upstream_eligible_after_qc"] == "true" for row in strict
        ),
        "modified_projection_never_uses_bare_sequence": all(
            not (
                any(
                    token in row["quality_flags"]
                    for token in (
                        "residue_projection_only",
                        "modified_or_noncanonical",
                        "chirality_or_modification",
                        "modified_sequence_preserved",
                    )
                )
                and row["representation_type"] == "sequence"
            )
            for row in representatives
        ),
        "pk_conditions_preserve_dose_and_timepoint": all(
            "dose" in json.loads(row["condition_json"])
            and "timepoint" in json.loads(row["condition_json"])
            for row in representatives
            if row["endpoint_family"] in {"F", "CL", "Vd"}
        ),
        "censored_rows_not_point_train_eligible": all(
            row["research_train_eligible"] == "false"
            for row in partitions["censored_numeric"]
        ),
        "cycpept_minus_10_sentinel_not_strict": all(
            not (
                row["source_id"] == "CycPeptMPDB" and row["raw_value"] == "-10"
            )
            for row in strict
        ),
        "binary_research_rows_have_identity_and_qc": all(
            row["identity_group_id"]
            and row["representation_text"]
            and row["upstream_eligible_after_qc"] == "true"
            and row["hard_condition_conflict"] != "true"
            for row in binary
            if row["research_train_eligible"] == "true"
        ),
        "bbb_strict_binary_rows_zero": sum(
            row["endpoint_family"] == "BBB"
            and row["binary_evidence_class"]
            in {"experimental_positive", "experimental_negative"}
            for row in binary
        )
        == 0,
        "literature_direct_training_zero": all(
            row["research_train_eligible"] == "false" for row in literature
        ),
        "literature_rows_1560": len(literature) == 1560,
        "peplife_na_identity_blocked": sum(
            row["source_id"] == "PEPlife2"
            and is_sentinel(row["smiles"])
            and row["partition"] == "identity_review"
            for row in representatives
        )
        == 896,
        "peplife_missing_identity_total_898": sum(
            row["source_id"] == "PEPlife2"
            and row["partition"] == "identity_review"
            for row in representatives
        )
        == 898,
        "bbb_weak_task_contains_both_classes": any(
            row["endpoint_family"] == "BBB"
            and row["task_kind"] == "binary_classification"
            and row["positive_rows"] > 0
            and row["negative_rows"] > 0
            for row in registry
        ),
        "solubility_binary_has_seven_solvent_tasks_without_conflict": len(
            {
                row["task_id"]
                for row in binary
                if row["endpoint_family"] == "solubility"
            }
        )
        == 7
        and all(
            row["hard_condition_conflict"] != "true"
            for row in binary
            if row["endpoint_family"] == "solubility"
        ),
        "weak_binary_tasks_never_strict_modelable": all(
            row["strict_modelable"] == "false"
            for row in registry
            if row["binary_use_tiers"]
            in {
                "derived_binary_research",
                "rough_solvent_conditioned_benchmark",
                "weak_benchmark_reproduction_only",
            }
        ),
    }
    if not all(assertions.values()):
        failed = sorted(key for key, value in assertions.items() if not value)
        raise ValueError("ML cleaning assertions failed: " + ", ".join(failed))

    readme = f"""# Peptide ML Cleaning v1

This release implements cleaning steps 1-6 over the V15 training-deduplicated candidate table: semantic tasks, normalized observation deduplication, safe identity/model representations, unit normalization, censoring sidecars, and binary evidence lanes. It does not create splits and does not promote literature staging.

## Counts

- Input in-scope rows: {len(source_rows):,}
- Normalized representative rows: {len(representatives):,}
- Strict numeric research candidates: {len(strict):,}
- Censored numeric sidecar: {len(partitions['censored_numeric']):,}
- Binary evidence catalog: {len(binary):,}
- Positive-unlabeled: {len(partitions['positive_unlabeled']):,}
- Identity/normalization/semantic review: {len(review_rows):,}
- Literature review retained: {len(literature):,}; direct training rows: 0

`strict_numeric.tsv` is the only point-regression candidate table. Binary rows remain physically separated by evidence lane. `task_registry.tsv` declares one semantic signature per task. Future split work must use identity and source groups; no split is fabricated here.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    outputs["README.md"] = artifact(readme_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "v15_training_deduplicated": artifact(input_path, len(list(iter_tsv(input_path)))),
            "literature_staging": artifact(literature_path, len(literature)),
        },
        "scope": {
            "raw_endpoint_to_family": RAW_TO_FAMILY,
            "steps_implemented": [
                "semantic_task_definition",
                "normalized_observation_deduplication",
                "identity_and_model_representation_qc",
                "unit_normalization",
                "censoring_preservation",
                "binary_evidence_lanes",
            ],
            "splits_created": False,
            "provenance_license_gate_completed": False,
        },
        "counts": {
            "input_scope_rows": len(source_rows),
            "representative_rows": len(representatives),
            "collapsed_duplicate_members": len(source_rows) - len(representatives),
            "strict_numeric_rows": len(strict),
            "censored_rows": len(partitions["censored_numeric"]),
            "binary_catalog_rows": len(binary),
            "positive_unlabeled_rows": len(partitions["positive_unlabeled"]),
            "review_rows": len(review_rows),
            "tasks": len(registry),
            "condition_variability_groups": len(conflicts),
            "hard_condition_conflict_groups": sum(
                row["hard_within_source_conflict"] == "true" for row in conflicts
            ),
            "partition_rows": dict(
                sorted(Counter(row["partition"] for row in representatives).items())
            ),
            "representative_by_endpoint": dict(
                sorted(Counter(row["endpoint_family"] for row in representatives).items())
            ),
            "partition_by_endpoint": {
                endpoint: dict(
                    sorted(
                        Counter(
                            row["partition"]
                            for row in representatives
                            if row["endpoint_family"] == endpoint
                        ).items()
                    )
                )
                for endpoint in sorted({row["endpoint_family"] for row in representatives})
            },
            "strict_numeric_by_endpoint": dict(
                sorted(Counter(row["endpoint_family"] for row in strict).items())
            ),
            "censored_by_endpoint": dict(
                sorted(
                    Counter(
                        row["endpoint_family"]
                        for row in partitions["censored_numeric"]
                    ).items()
                )
            ),
            "binary_by_endpoint": dict(
                sorted(Counter(row["endpoint_family"] for row in binary).items())
            ),
            "binary_by_evidence": dict(
                sorted(Counter(row["binary_evidence_class"] for row in binary).items())
            ),
            "binary_by_use_tier": dict(
                sorted(Counter(row["binary_use_tier"] for row in binary).items())
            ),
            "positive_unlabeled_by_endpoint": dict(
                sorted(
                    Counter(
                        row["endpoint_family"]
                        for row in partitions["positive_unlabeled"]
                    ).items()
                )
            ),
            "tasks_by_endpoint": dict(
                sorted(Counter(row["endpoint_family"] for row in registry).items())
            ),
            "strict_modelable_tasks": sum(
                row["strict_modelable"] == "true" for row in registry
            ),
            "research_modelable_tasks": sum(
                row["research_modelable"] == "true" for row in registry
            ),
            "review_reasons": dict(sorted(exclusion_counts.items())),
        },
        "assertions": assertions,
        "outputs": outputs,
        "implementation": {
            "script": display_path(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
            "conversion_version": CONVERSION_VERSION,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args.input, args.literature, args.output_dir)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
