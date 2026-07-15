from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V15_INPUT = (
    ROOT
    / "data"
    / "peptide_property_expansion_v15"
    / "peptide_property_training_deduplicated_v15.tsv"
)
DEFAULT_CLEAN_DIR = ROOT / "data" / "peptide_ml_cleaning_v1"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "peptide_ml_cleaning_v1_statistics_v1"
SCHEMA_VERSION = "peptide_ml_cleaning_v1_statistics_v1"

CLEAN_ENDPOINT_FAMILIES = [
    "LogD",
    "solubility",
    "F",
    "T1/2",
    "PPB",
    "CL",
    "Vd",
    "BBB",
    "Kp",
    "permeability",
    "cell_penetration",
]
PARTITIONS = [
    "strict_numeric",
    "censored_numeric",
    "binary_evidence_catalog",
    "positive_unlabeled",
    "identity_review",
    "normalization_review",
    "semantic_review",
]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
SENTINELS = {
    "",
    "na",
    "none",
    "null",
    "unknown",
    "notavailable",
    "notapplicable",
    "notreported",
}
STAT_KEYS = [
    "n",
    "unique_values",
    "min",
    "p01",
    "p05",
    "p25",
    "median",
    "p75",
    "p95",
    "p99",
    "max",
    "mean",
    "sd",
    "iqr",
    "mad",
    "top_value_fraction",
    "zero_count",
    "negative_count",
    "positive_count",
    "tukey_outlier_count",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def is_sentinel(value: Any) -> bool:
    return compact(value) in SENTINELS


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


def parse_number(value: Any) -> float | None:
    try:
        number = float(clean(value))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def number_text(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def ratio_text(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return ""
    return number_text(float(numerator) / float(denominator))


def quantile(sorted_values: list[float], probability: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def describe(values: Iterable[float]) -> dict[str, Any]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {key: 0 if key in {"n", "unique_values"} else "" for key in STAT_KEYS}
    q01 = quantile(finite, 0.01)
    q05 = quantile(finite, 0.05)
    q25 = quantile(finite, 0.25)
    median = quantile(finite, 0.50)
    q75 = quantile(finite, 0.75)
    q95 = quantile(finite, 0.95)
    q99 = quantile(finite, 0.99)
    assert q25 is not None and median is not None and q75 is not None
    deviations = sorted(abs(value - median) for value in finite)
    mad = quantile(deviations, 0.5)
    iqr = q75 - q25
    lower_fence = q25 - 1.5 * iqr
    upper_fence = q75 + 1.5 * iqr
    top_count = max(Counter(finite).values())
    result: dict[str, Any] = {
        "n": len(finite),
        "unique_values": len(set(finite)),
        "min": number_text(finite[0]),
        "p01": number_text(q01),
        "p05": number_text(q05),
        "p25": number_text(q25),
        "median": number_text(median),
        "p75": number_text(q75),
        "p95": number_text(q95),
        "p99": number_text(q99),
        "max": number_text(finite[-1]),
        "mean": number_text(statistics.fmean(finite)),
        "sd": number_text(statistics.stdev(finite) if len(finite) > 1 else 0.0),
        "iqr": number_text(iqr),
        "mad": number_text(mad),
        "top_value_fraction": ratio_text(top_count, len(finite)),
        "zero_count": sum(value == 0 for value in finite),
        "negative_count": sum(value < 0 for value in finite),
        "positive_count": sum(value > 0 for value in finite),
        "tukey_outlier_count": sum(
            value < lower_fence or value > upper_fence for value in finite
        ),
    }
    return result


def prefixed_stats(prefix: str, values: Iterable[float]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in describe(values).items()}


def histogram(values: list[float]) -> list[tuple[float, float, int]]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return []
    minimum, maximum = finite[0], finite[-1]
    if minimum == maximum:
        return [(minimum, maximum, len(finite))]
    q25 = quantile(finite, 0.25)
    q75 = quantile(finite, 0.75)
    assert q25 is not None and q75 is not None
    width = 2 * (q75 - q25) * (len(finite) ** (-1 / 3))
    if width > 0:
        bins = math.ceil((maximum - minimum) / width)
    else:
        bins = math.ceil(math.sqrt(len(finite)))
    bins = min(50, max(5, bins))
    span = (maximum - minimum) / bins
    counts = [0] * bins
    for value in finite:
        index = min(bins - 1, int((value - minimum) / span))
        counts[index] += 1
    return [
        (minimum + index * span, minimum + (index + 1) * span, count)
        for index, count in enumerate(counts)
    ]


def iter_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Missing TSV header: {path}")
        for row in reader:
            yield {key: "" if value is None else value for key, value in row.items()}


def write_tsv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = {key for row in rows for key in row}
    fields = [field for field in preferred if field in available]
    fields.extend(sorted(available - set(fields)))
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
    return len(rows)


def sequence_class(value: Any) -> tuple[str, str]:
    raw = clean(value)
    if is_sentinel(raw):
        return "missing_or_sentinel", ""
    # Lowercase residues can encode D-amino acids, so raw projections are not uppercased.
    normalized = re.sub(r"\s+", "", raw)
    if STANDARD_SEQUENCE.fullmatch(normalized):
        return "standard_20aa", normalized
    return "modified_or_nonstandard", raw


def helm_monomer_count(value: str) -> int | None:
    blocks = re.findall(r"\{([^{}]*)\}", value)
    if not blocks:
        return None
    tokens = [token for block in blocks for token in block.split(".") if token]
    return len(tokens) if tokens else None


def model_residue_count(row: dict[str, str]) -> int | None:
    representation_type = row.get("representation_type", "")
    text = row.get("representation_text", "")
    if representation_type == "sequence" and text.startswith("SEQ:"):
        payload = text[4:]
        return len(payload) if STANDARD_SEQUENCE.fullmatch(payload) else None
    if representation_type == "helm" and text.startswith("HELM:"):
        return helm_monomer_count(text[5:])
    return None


def sequence_accumulator() -> dict[str, Any]:
    return {
        "rows": 0,
        "member_rows": 0,
        "standard_rows": 0,
        "exact_standard_rows": 0,
        "standard_member_rows": 0,
        "nonstandard_rows": 0,
        "missing_rows": 0,
        "raw_sequence_present_rows": 0,
        "raw_sequence_nonblank_rows": 0,
        "sequence_model_eligible_rows": 0,
        "structure_model_eligible_rows": 0,
        "projection_or_modified_rows": 0,
        "nonascii_representation_rows": 0,
        "representation_prefix_error_rows": 0,
        "unique_raw_sequences": set(),
        "unique_standard_sequences": set(),
        "unique_identities": set(),
        "unique_representations": set(),
        "raw_standard_lengths": [],
        "raw_standard_member_lengths": [],
        "raw_nonstandard_char_lengths": [],
        "representation_char_lengths": [],
        "model_residue_counts": [],
        "length_counter": Counter(),
        "length_member_counter": Counter(),
        "model_residue_counter": Counter(),
        "model_residue_member_counter": Counter(),
        "aa_observation": Counter(),
        "aa_member": Counter(),
        "aa_rows_containing": Counter(),
        "aa_member_rows_containing": Counter(),
    }


def add_sequence(
    accumulator: dict[str, Any],
    sequence: str,
    member_weight: int = 1,
    identity_group_id: str = "",
    representation_type: str = "",
    representation_text: str = "",
    sequence_model_eligible: str = "",
    structure_model_eligible: str = "",
    quality_flags: str = "",
) -> None:
    accumulator["rows"] += 1
    accumulator["member_rows"] += member_weight
    raw_sequence = clean(sequence)
    accumulator["raw_sequence_nonblank_rows"] += bool(raw_sequence)
    accumulator["exact_standard_rows"] += bool(
        STANDARD_SEQUENCE.fullmatch(raw_sequence)
    )
    category, normalized = sequence_class(sequence)
    if category == "missing_or_sentinel":
        accumulator["missing_rows"] += 1
    elif category == "standard_20aa":
        length = len(normalized)
        accumulator["raw_sequence_present_rows"] += 1
        accumulator["standard_rows"] += 1
        accumulator["standard_member_rows"] += member_weight
        accumulator["unique_raw_sequences"].add(normalized)
        accumulator["unique_standard_sequences"].add(normalized)
        accumulator["raw_standard_lengths"].append(length)
        accumulator["raw_standard_member_lengths"].extend([length] * member_weight)
        accumulator["length_counter"][length] += 1
        accumulator["length_member_counter"][length] += member_weight
        residues = Counter(normalized)
        accumulator["aa_observation"].update(residues)
        accumulator["aa_member"].update(
            {residue: count * member_weight for residue, count in residues.items()}
        )
        for residue in residues:
            accumulator["aa_rows_containing"][residue] += 1
            accumulator["aa_member_rows_containing"][residue] += member_weight
    else:
        accumulator["raw_sequence_present_rows"] += 1
        accumulator["nonstandard_rows"] += 1
        accumulator["unique_raw_sequences"].add(normalized)
        accumulator["raw_nonstandard_char_lengths"].append(len(normalized))
    if identity_group_id:
        accumulator["unique_identities"].add(identity_group_id)
    if representation_text:
        accumulator["unique_representations"].add(representation_text)
        accumulator["representation_char_lengths"].append(len(representation_text))
        expected_prefix = {
            "sequence": "SEQ:",
            "helm": "HELM:",
            "smiles": "SMILES:",
        }.get(representation_type)
        if expected_prefix and not representation_text.startswith(expected_prefix):
            accumulator["representation_prefix_error_rows"] += 1
        if not representation_text.isascii():
            accumulator["nonascii_representation_rows"] += 1
    residue_count = model_residue_count(
        {
            "representation_type": representation_type,
            "representation_text": representation_text,
        }
    )
    if residue_count is not None:
        accumulator["model_residue_counts"].append(residue_count)
        accumulator["model_residue_counter"][residue_count] += 1
        accumulator["model_residue_member_counter"][residue_count] += member_weight
    accumulator["sequence_model_eligible_rows"] += sequence_model_eligible == "true"
    accumulator["structure_model_eligible_rows"] += structure_model_eligible == "true"
    accumulator["projection_or_modified_rows"] += any(
        token in quality_flags
        for token in (
            "residue_projection_only",
            "modified_or_noncanonical",
            "modified_sequence_preserved",
            "chirality_or_modification",
        )
    )


def finalize_sequence_summary(
    key: tuple[str, str, str], accumulator: dict[str, Any]
) -> dict[str, Any]:
    data_layer, endpoint, partition = key
    row = {
        "data_layer": data_layer,
        "endpoint": endpoint,
        "partition": partition,
        "rows": accumulator["rows"],
        "member_rows": accumulator["member_rows"],
        "raw_sequence_present_rows": accumulator["raw_sequence_present_rows"],
        "raw_sequence_nonblank_rows": accumulator["raw_sequence_nonblank_rows"],
        "raw_exact_standard_20aa_rows": accumulator["exact_standard_rows"],
        "raw_standard_20aa_rows": accumulator["standard_rows"],
        "raw_standard_20aa_member_rows": accumulator["standard_member_rows"],
        "raw_modified_or_nonstandard_rows": accumulator["nonstandard_rows"],
        "raw_sequence_missing_rows": accumulator["missing_rows"],
        "unique_raw_sequences": len(accumulator["unique_raw_sequences"]),
        "unique_standard_sequences": len(accumulator["unique_standard_sequences"]),
        "unique_identity_groups": len(accumulator["unique_identities"]),
        "unique_selected_representations": len(accumulator["unique_representations"]),
        "sequence_model_eligible_rows": accumulator["sequence_model_eligible_rows"],
        "structure_model_eligible_rows": accumulator["structure_model_eligible_rows"],
        "projection_or_modified_rows": accumulator["projection_or_modified_rows"],
        "nonascii_representation_rows": accumulator["nonascii_representation_rows"],
        "representation_prefix_error_rows": accumulator[
            "representation_prefix_error_rows"
        ],
        **prefixed_stats(
            "raw_standard_length", accumulator["raw_standard_lengths"]
        ),
        **prefixed_stats(
            "raw_nonstandard_char_length",
            accumulator["raw_nonstandard_char_lengths"],
        ),
        **prefixed_stats(
            "representation_char_length",
            accumulator["representation_char_lengths"],
        ),
        **prefixed_stats("model_residue_count", accumulator["model_residue_counts"]),
    }
    return row


def split_values(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def size_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 29:
        return "10-29"
    if value <= 99:
        return "30-99"
    if value <= 999:
        return "100-999"
    return ">=1000"


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
    v15_input: Path = DEFAULT_V15_INPUT,
    clean_dir: Path = DEFAULT_CLEAN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    v15_input = v15_input.resolve()
    clean_dir = clean_dir.resolve()
    output_dir = output_dir.resolve()
    normalized_path = clean_dir / "normalized_observations.tsv"
    registry_path = clean_dir / "task_registry.tsv"
    literature_path = clean_dir / "literature_review_cleaned.tsv"
    cleaning_manifest_path = clean_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    v15_endpoint: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "numeric_rows": 0,
            "label_rows": 0,
            "sources": set(),
            "units": set(),
            "relations": Counter(),
            "task_types": Counter(),
            "labels": Counter(),
            "identity_keys": set(),
            "sequences": set(),
            "standard_sequences": set(),
            "raw_sequence_nonblank_rows": 0,
            "exact_standard_sequence_rows": 0,
            "normalized_standard_sequence_rows": 0,
            "missing_sequence_rows": 0,
            "nonstandard_sequence_rows": 0,
        }
    )
    raw_numeric: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    label_counts: Counter[tuple[str, str, str]] = Counter()
    sequence_groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        sequence_accumulator
    )
    v15_rows = 0

    for row in iter_tsv(v15_input):
        v15_rows += 1
        endpoint = clean(row.get("endpoint")) or "not_reported"
        summary = v15_endpoint[endpoint]
        summary["rows"] += 1
        summary["sources"].add(clean(row.get("source_id")))
        summary["units"].add(clean(row.get("unit")) or "not_reported")
        summary["relations"][clean(row.get("relation")) or "blank"] += 1
        summary["task_types"][clean(row.get("task_type")) or "not_reported"] += 1
        identity_key = clean(row.get("identity_key"))
        if identity_key:
            summary["identity_keys"].add(identity_key)
        category, normalized_sequence = sequence_class(row.get("sequence"))
        raw_sequence = clean(row.get("sequence"))
        summary["raw_sequence_nonblank_rows"] += bool(raw_sequence)
        summary["exact_standard_sequence_rows"] += bool(
            STANDARD_SEQUENCE.fullmatch(raw_sequence)
        )
        if category == "missing_or_sentinel":
            summary["missing_sequence_rows"] += 1
        else:
            summary["sequences"].add(normalized_sequence)
            if category == "standard_20aa":
                summary["normalized_standard_sequence_rows"] += 1
                summary["standard_sequences"].add(normalized_sequence)
            else:
                summary["nonstandard_sequence_rows"] += 1
        value = parse_number(row.get("value"))
        if value is not None:
            summary["numeric_rows"] += 1
            raw_numeric[
                (
                    endpoint,
                    clean(row.get("task_type")) or "not_reported",
                    clean(row.get("unit")) or "not_reported",
                    clean(row.get("relation")) or "blank",
                )
            ].append(value)
        label = clean(row.get("label"))
        if label:
            summary["label_rows"] += 1
            summary["labels"][label] += 1
            label_counts[(endpoint, clean(row.get("task_type")), label)] += 1
        for key in (
            ("v15_training_deduplicated", "__ALL__", "all"),
            ("v15_training_deduplicated", endpoint, "all"),
        ):
            add_sequence(
                sequence_groups[key],
                row.get("sequence", ""),
                identity_group_id=identity_key,
                quality_flags=row.get("quality_flags", ""),
            )

    v15_endpoint_rows = []
    for endpoint in sorted(v15_endpoint):
        item = v15_endpoint[endpoint]
        v15_endpoint_rows.append(
            {
                "raw_endpoint": endpoint,
                "rows": item["rows"],
                "numeric_rows": item["numeric_rows"],
                "label_rows": item["label_rows"],
                "numeric_fraction": ratio_text(item["numeric_rows"], item["rows"]),
                "label_fraction": ratio_text(item["label_rows"], item["rows"]),
                "unique_sources": len(item["sources"] - {""}),
                "unique_identity_keys": len(item["identity_keys"]),
                "unique_raw_sequences": len(item["sequences"]),
                "unique_standard_sequences": len(item["standard_sequences"]),
                "raw_sequence_nonblank_rows": item["raw_sequence_nonblank_rows"],
                "raw_exact_standard_20aa_rows": item[
                    "exact_standard_sequence_rows"
                ],
                "whitespace_normalized_standard_20aa_rows": item[
                    "normalized_standard_sequence_rows"
                ],
                "missing_sequence_rows": item["missing_sequence_rows"],
                "nonstandard_sequence_rows": item["nonstandard_sequence_rows"],
                "units_json": stable_json(sorted(item["units"])),
                "relations_json": stable_json(dict(sorted(item["relations"].items()))),
                "task_types_json": stable_json(dict(sorted(item["task_types"].items()))),
                "labels_json": stable_json(dict(sorted(item["labels"].items()))),
            }
        )

    raw_numeric_rows = []
    for key in sorted(raw_numeric):
        endpoint, task_type, unit, relation = key
        raw_numeric_rows.append(
            {
                "raw_endpoint": endpoint,
                "task_type": task_type,
                "raw_unit": unit,
                "raw_relation": relation,
                "descriptive_only": "true",
                **describe(raw_numeric[key]),
            }
        )

    v15_label_rows = [
        {
            "raw_endpoint": endpoint,
            "task_type": task_type,
            "label": label,
            "rows": count,
        }
        for (endpoint, task_type, label), count in sorted(label_counts.items())
    ]

    registry = {row["task_id"]: row for row in iter_tsv(registry_path)}
    clean_rows = list(iter_tsv(normalized_path))
    endpoint_acc: dict[str, dict[str, Any]] = {
        endpoint: {
            "rows": 0,
            "member_rows": 0,
            "partitions": Counter(),
            "identities": set(),
            "representations": set(),
            "tasks": set(),
            "sources": set(),
            "source_groups": set(),
            "raw_endpoints": set(),
            "representation_types": Counter(),
            "standard_sequence_rows": 0,
            "nonstandard_sequence_rows": 0,
            "missing_sequence_rows": 0,
            "sequence_model_rows": 0,
            "structure_model_rows": 0,
            "missing_identity_rows": 0,
            "hard_conflict_rows": 0,
            "research_train_rows": 0,
        }
        for endpoint in CLEAN_ENDPOINT_FAMILIES
    }
    strict_rows: list[dict[str, str]] = []
    censored_rows: list[dict[str, str]] = []
    binary_rows: list[dict[str, str]] = []
    pu_rows: list[dict[str, str]] = []
    representation_groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "member_rows": 0,
            "identities": set(),
            "representations": set(),
            "char_lengths": [],
            "residue_counts": [],
            "nonascii_rows": 0,
            "prefix_error_rows": 0,
        }
    )

    for row in clean_rows:
        endpoint = row["endpoint_family"]
        item = endpoint_acc.setdefault(
            endpoint,
            {
                "rows": 0,
                "member_rows": 0,
                "partitions": Counter(),
                "identities": set(),
                "representations": set(),
                "tasks": set(),
                "sources": set(),
                "source_groups": set(),
                "raw_endpoints": set(),
                "representation_types": Counter(),
                "standard_sequence_rows": 0,
                "nonstandard_sequence_rows": 0,
                "missing_sequence_rows": 0,
                "sequence_model_rows": 0,
                "structure_model_rows": 0,
                "missing_identity_rows": 0,
                "hard_conflict_rows": 0,
                "research_train_rows": 0,
            },
        )
        member_weight = int(row["member_count"])
        item["rows"] += 1
        item["member_rows"] += member_weight
        item["partitions"][row["partition"]] += 1
        if row["identity_group_id"]:
            item["identities"].add(row["identity_group_id"])
        else:
            item["missing_identity_rows"] += 1
        if row["representation_text"]:
            item["representations"].add(row["representation_text"])
        item["tasks"].add(row["task_id"])
        item["sources"].update(split_values(row["source_ids"]) or [row["source_id"]])
        item["source_groups"].update(split_values(row["source_group_ids"]))
        item["raw_endpoints"].add(row["raw_endpoint"])
        item["representation_types"][row["representation_type"] or "missing"] += 1
        category, _ = sequence_class(row["sequence"])
        if category == "standard_20aa":
            item["standard_sequence_rows"] += 1
        elif category == "modified_or_nonstandard":
            item["nonstandard_sequence_rows"] += 1
        else:
            item["missing_sequence_rows"] += 1
        item["sequence_model_rows"] += row["sequence_model_eligible"] == "true"
        item["structure_model_rows"] += row["structure_model_eligible"] == "true"
        item["hard_conflict_rows"] += row["hard_condition_conflict"] == "true"
        item["research_train_rows"] += row["research_train_eligible"] == "true"
        for key in (
            ("ml_clean_representatives", "__ALL__", "all"),
            ("ml_clean_representatives", endpoint, "all"),
            ("ml_clean_partitions", endpoint, row["partition"]),
        ):
            add_sequence(
                sequence_groups[key],
                row["sequence"],
                member_weight=member_weight,
                identity_group_id=row["identity_group_id"],
                representation_type=row["representation_type"],
                representation_text=row["representation_text"],
                sequence_model_eligible=row["sequence_model_eligible"],
                structure_model_eligible=row["structure_model_eligible"],
                quality_flags=row["quality_flags"],
            )
        for representation_key in (
            ("__ALL__", "all", row["representation_type"] or "missing"),
            (endpoint, "all", row["representation_type"] or "missing"),
            (
                endpoint,
                row["partition"],
                row["representation_type"] or "missing",
            ),
        ):
            representation_item = representation_groups[representation_key]
            representation_item["rows"] += 1
            representation_item["member_rows"] += member_weight
            if row["identity_group_id"]:
                representation_item["identities"].add(row["identity_group_id"])
            if row["representation_text"]:
                representation_item["representations"].add(
                    row["representation_text"]
                )
                representation_item["char_lengths"].append(
                    len(row["representation_text"])
                )
                representation_item["nonascii_rows"] += not row[
                    "representation_text"
                ].isascii()
                expected_prefix = {
                    "sequence": "SEQ:",
                    "helm": "HELM:",
                    "smiles": "SMILES:",
                }.get(row["representation_type"])
                representation_item["prefix_error_rows"] += bool(
                    expected_prefix
                    and not row["representation_text"].startswith(expected_prefix)
                )
            residue_count = model_residue_count(row)
            if residue_count is not None:
                representation_item["residue_counts"].append(residue_count)
        if row["partition"] == "strict_numeric":
            strict_rows.append(row)
        elif row["partition"] == "censored_numeric":
            censored_rows.append(row)
        elif row["partition"] == "binary_evidence_catalog":
            binary_rows.append(row)
        elif row["partition"] == "positive_unlabeled":
            pu_rows.append(row)

    selected_representation_rows = []
    for key in sorted(representation_groups):
        endpoint, partition, representation_type = key
        item = representation_groups[key]
        selected_representation_rows.append(
            {
                "endpoint_family": endpoint,
                "partition": partition,
                "representation_type": representation_type,
                "rows": item["rows"],
                "member_rows": item["member_rows"],
                "unique_identity_groups": len(item["identities"]),
                "unique_representations": len(item["representations"]),
                "nonascii_rows": item["nonascii_rows"],
                "prefix_error_rows": item["prefix_error_rows"],
                **prefixed_stats("character_length", item["char_lengths"]),
                **prefixed_stats("model_residue_count", item["residue_counts"]),
            }
        )

    literature_rows = list(iter_tsv(literature_path))
    literature_counts: Counter[tuple[str, str]] = Counter(
        (row["endpoint"], row["endpoint_detail"]) for row in literature_rows
    )
    literature_distribution = [
        {
            "data_layer": "literature_review",
            "endpoint": endpoint,
            "endpoint_detail": detail,
            "rows": count,
            "training_rows": 0,
        }
        for (endpoint, detail), count in sorted(literature_counts.items())
    ]

    clean_endpoint_rows = []
    for endpoint in CLEAN_ENDPOINT_FAMILIES:
        item = endpoint_acc[endpoint]
        endpoint_tasks = [
            task for task in registry.values() if task["endpoint_family"] == endpoint
        ]
        row = {
            "endpoint_family": endpoint,
            "declared_in_cleaning_scope": "true",
            "input_member_rows": item["member_rows"],
            "representative_rows": item["rows"],
            "unique_identity_groups": len(item["identities"]),
            "missing_identity_rows": item["missing_identity_rows"],
            "unique_selected_representations": len(item["representations"]),
            "tasks": len(item["tasks"]),
            "strict_modelable_tasks": sum(
                task["strict_modelable"] == "true" for task in endpoint_tasks
            ),
            "research_modelable_tasks": sum(
                task["research_modelable"] == "true" for task in endpoint_tasks
            ),
            "unique_sources": len(item["sources"] - {""}),
            "unique_source_groups": len(item["source_groups"]),
            "raw_endpoints_json": stable_json(sorted(item["raw_endpoints"])),
            "raw_standard_20aa_rows": item["standard_sequence_rows"],
            "raw_modified_or_nonstandard_rows": item["nonstandard_sequence_rows"],
            "raw_sequence_missing_rows": item["missing_sequence_rows"],
            "sequence_model_eligible_rows": item["sequence_model_rows"],
            "structure_model_eligible_rows": item["structure_model_rows"],
            "representation_types_json": stable_json(
                dict(sorted(item["representation_types"].items()))
            ),
            "hard_condition_conflict_rows": item["hard_conflict_rows"],
            "research_train_eligible_rows": item["research_train_rows"],
            "production_train_status": "not_assessed_step7",
        }
        for partition in PARTITIONS:
            row[f"partition_{partition}"] = item["partitions"][partition]
        clean_endpoint_rows.append(row)

    sequence_summary_rows = [
        finalize_sequence_summary(key, sequence_groups[key])
        for key in sorted(sequence_groups)
    ]
    sequence_histogram_rows: list[dict[str, Any]] = []
    amino_acid_rows: list[dict[str, Any]] = []
    for key in sorted(sequence_groups):
        data_layer, endpoint, partition = key
        item = sequence_groups[key]
        for length, count in sorted(item["length_counter"].items()):
            sequence_histogram_rows.append(
                {
                    "data_layer": data_layer,
                    "endpoint": endpoint,
                    "partition": partition,
                    "length_metric": "raw_standard_20aa_residue_count",
                    "length": length,
                    "representative_rows": count,
                    "member_rows": item["length_member_counter"][length],
                }
            )
        for length, count in sorted(item["model_residue_counter"].items()):
            sequence_histogram_rows.append(
                {
                    "data_layer": data_layer,
                    "endpoint": endpoint,
                    "partition": partition,
                    "length_metric": "selected_model_residue_count_sequence_or_helm",
                    "length": length,
                    "representative_rows": count,
                    "member_rows": item["model_residue_member_counter"][length],
                }
            )
        if partition != "all":
            continue
        unique_counter: Counter[str] = Counter()
        unique_containing: Counter[str] = Counter()
        for sequence in item["unique_standard_sequences"]:
            residues = Counter(sequence)
            unique_counter.update(residues)
            for residue in residues:
                unique_containing[residue] += 1
        obs_total = sum(item["aa_observation"].values())
        member_total = sum(item["aa_member"].values())
        unique_total = sum(unique_counter.values())
        for residue in AMINO_ACIDS:
            amino_acid_rows.append(
                {
                    "data_layer": data_layer,
                    "endpoint": endpoint,
                    "residue": residue,
                    "observation_residue_count": item["aa_observation"][residue],
                    "observation_residue_fraction": ratio_text(
                        item["aa_observation"][residue], obs_total
                    ),
                    "member_weighted_residue_count": item["aa_member"][residue],
                    "member_weighted_residue_fraction": ratio_text(
                        item["aa_member"][residue], member_total
                    ),
                    "unique_sequence_residue_count": unique_counter[residue],
                    "unique_sequence_residue_fraction": ratio_text(
                        unique_counter[residue], unique_total
                    ),
                    "sequence_rows_containing": item["aa_rows_containing"][residue],
                    "member_rows_containing": item["aa_member_rows_containing"][residue],
                    "unique_sequences_containing": unique_containing[residue],
                }
            )

    endpoint_unit_values: dict[tuple[str, ...], list[float]] = defaultdict(list)
    task_values: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for row in strict_rows:
        value = parse_number(row["normalized_value"])
        if value is None:
            continue
        endpoint_key = (
            row["endpoint_family"],
            row["parameter_semantics"],
            row["parameter_basis"],
            row["assay_family"],
            row["normalized_unit"],
            row["target_transform"],
        )
        endpoint_unit_values[endpoint_key].append(value)
        task_values[row["task_id"]].append((row, value))

    endpoint_unit_rows = []
    for key in sorted(endpoint_unit_values):
        endpoint, semantics, basis, assay, unit, transform = key
        rows_for_group = [
            row
            for row in strict_rows
            if (
                row["endpoint_family"],
                row["parameter_semantics"],
                row["parameter_basis"],
                row["assay_family"],
                row["normalized_unit"],
                row["target_transform"],
            )
            == key
        ]
        endpoint_unit_rows.append(
            {
                "endpoint_family": endpoint,
                "parameter_semantics": semantics,
                "parameter_basis": basis,
                "assay_family": assay,
                "normalized_unit": unit,
                "target_transform": transform,
                "descriptive_only": "true",
                "tasks": len({row["task_id"] for row in rows_for_group}),
                "unique_identities": len(
                    {row["identity_group_id"] for row in rows_for_group if row["identity_group_id"]}
                ),
                "unique_sources": len({row["source_id"] for row in rows_for_group}),
                **describe(endpoint_unit_values[key]),
            }
        )

    task_numeric_rows = []
    histogram_rows: list[dict[str, Any]] = []
    for task in sorted(task_values):
        pairs = task_values[task]
        rows_for_task = [pair[0] for pair in pairs]
        values = [pair[1] for pair in pairs]
        registry_row = registry[task]
        source_counts = Counter(row["source_id"] for row in rows_for_task)
        task_numeric_rows.append(
            {
                "task_id": task,
                "endpoint_family": rows_for_task[0]["endpoint_family"],
                "parameter_semantics": rows_for_task[0]["parameter_semantics"],
                "parameter_basis": rows_for_task[0]["parameter_basis"],
                "assay_family": rows_for_task[0]["assay_family"],
                "normalized_unit": rows_for_task[0]["normalized_unit"],
                "target_transform": rows_for_task[0]["target_transform"],
                "semantic_signature": rows_for_task[0]["task_semantic_signature"],
                "strict_modelable": registry_row["strict_modelable"],
                "research_modelable": registry_row["research_modelable"],
                "size_bin": size_bin(len(rows_for_task)),
                "unique_identities": len(
                    {row["identity_group_id"] for row in rows_for_task if row["identity_group_id"]}
                ),
                "unique_sources": len(source_counts),
                "largest_source_fraction": ratio_text(
                    max(source_counts.values()), len(rows_for_task)
                ),
                "sequence_representation_rows": sum(
                    row["representation_type"] == "sequence" for row in rows_for_task
                ),
                "helm_representation_rows": sum(
                    row["representation_type"] == "helm" for row in rows_for_task
                ),
                "smiles_representation_rows": sum(
                    row["representation_type"] == "smiles" for row in rows_for_task
                ),
                "hard_conflict_rows": sum(
                    row["hard_condition_conflict"] == "true" for row in rows_for_task
                ),
                **describe(values),
            }
        )
        if len(values) >= 10:
            bins = histogram(values)
            for index, (lower, upper, count) in enumerate(bins):
                histogram_rows.append(
                    {
                        "scope": "task",
                        "group_id": task,
                        "endpoint_family": rows_for_task[0]["endpoint_family"],
                        "normalized_unit": rows_for_task[0]["normalized_unit"],
                        "n": len(values),
                        "bin_index": index,
                        "lower_inclusive": number_text(lower),
                        "upper": number_text(upper),
                        "upper_inclusive": str(index == len(bins) - 1).lower(),
                        "count": count,
                    }
                )
    for key in sorted(endpoint_unit_values):
        values = endpoint_unit_values[key]
        endpoint, semantics, basis, assay, unit, transform = key
        group_id = "ENDPOINTUNIT1:" + sha256_text(stable_json(key))[:20]
        bins = histogram(values)
        for index, (lower, upper, count) in enumerate(bins):
            histogram_rows.append(
                {
                    "scope": "endpoint_semantics_unit_descriptive",
                    "group_id": group_id,
                    "endpoint_family": endpoint,
                    "parameter_semantics": semantics,
                    "parameter_basis": basis,
                    "assay_family": assay,
                    "normalized_unit": unit,
                    "target_transform": transform,
                    "n": len(values),
                    "bin_index": index,
                    "lower_inclusive": number_text(lower),
                    "upper": number_text(upper),
                    "upper_inclusive": str(index == len(bins) - 1).lower(),
                    "count": count,
                }
            )

    censor_groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in censored_rows:
        if row["censoring_type"] == "upper_bound":
            bound = parse_number(row["normalized_upper"])
        elif row["censoring_type"] == "lower_bound":
            bound = parse_number(row["normalized_lower"])
        else:
            bound = parse_number(row["normalized_value"])
        if bound is not None:
            censor_groups[
                (
                    row["task_id"],
                    row["endpoint_family"],
                    row["normalized_unit"],
                    row["normalized_relation"],
                    row["censoring_type"],
                )
            ].append(bound)
    censoring_distribution = [
        {
            "task_id": key[0],
            "endpoint_family": key[1],
            "normalized_unit": key[2],
            "normalized_relation": key[3],
            "censoring_type": key[4],
            "point_regression_eligible": "false",
            **describe(values),
        }
        for key, values in sorted(censor_groups.items())
    ]

    classification_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in binary_rows + pu_rows:
        classification_groups[row["task_id"]].append(row)
    binary_distribution = []
    for task in sorted(classification_groups):
        rows_for_task = classification_groups[task]
        labels_by_identity: dict[str, set[str]] = defaultdict(set)
        for row in rows_for_task:
            if row["identity_group_id"] and row["raw_label"]:
                labels_by_identity[row["identity_group_id"]].add(row["raw_label"])
        source_counts = Counter(row["source_id"] for row in rows_for_task)
        positives = sum(row["raw_label"] == "1" for row in rows_for_task)
        negatives = sum(row["raw_label"] == "0" for row in rows_for_task)
        binary_distribution.append(
            {
                "task_id": task,
                "endpoint_family": rows_for_task[0]["endpoint_family"],
                "task_kind": rows_for_task[0]["task_kind"],
                "binary_use_tiers": ";".join(
                    sorted({row["binary_use_tier"] for row in rows_for_task if row["binary_use_tier"]})
                ),
                "binary_evidence_classes": ";".join(
                    sorted(
                        {
                            row["binary_evidence_class"]
                            for row in rows_for_task
                            if row["binary_evidence_class"]
                        }
                    )
                ),
                "rows": len(rows_for_task),
                "positive_rows": positives,
                "negative_rows": negatives,
                "positive_rate_among_labeled": ratio_text(
                    positives, positives + negatives
                ),
                "is_positive_unlabeled": str(
                    rows_for_task[0]["task_kind"] == "positive_unlabeled"
                ).lower(),
                "unique_identities": len(labels_by_identity),
                "identity_label_conflicts": sum(
                    len(labels) > 1 for labels in labels_by_identity.values()
                ),
                "unique_sources": len(source_counts),
                "largest_source_fraction": ratio_text(
                    max(source_counts.values()), len(rows_for_task)
                ),
                "positive_class": rows_for_task[0]["positive_class"],
                "negative_class": rows_for_task[0]["negative_class"],
                "strict_modelable": registry[task]["strict_modelable"],
                "research_modelable": registry[task]["research_modelable"],
            }
        )

    rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        rows_by_task[row["task_id"]].append(row)
    task_viability = []
    for task in sorted(registry):
        task_rows = rows_by_task.get(task, [])
        reg = registry[task]
        source_counts = Counter(row["source_id"] for row in task_rows)
        strict_count = int(reg["strict_numeric_rows"])
        positive_count = int(reg["positive_rows"])
        negative_count = int(reg["negative_rows"])
        if reg["strict_modelable"] == "true":
            failure_reason = ""
        elif strict_count:
            failure_reason = "strict_numeric_below_30_rows"
        elif positive_count and negative_count:
            failure_reason = "non_strict_binary_or_below_class_size_gate"
        elif positive_count or negative_count:
            failure_reason = "single_class_or_positive_unlabeled"
        else:
            failure_reason = "review_only_or_no_trainable_target"
        task_viability.append(
            {
                **reg,
                "size_bin_all_rows": size_bin(len(task_rows)),
                "size_bin_strict_numeric": size_bin(strict_count),
                "largest_source_fraction": ratio_text(
                    max(source_counts.values()) if source_counts else 0,
                    len(task_rows),
                ),
                "strict_modelability_failure_reason": failure_reason,
            }
        )

    identity_groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    identity_missing: Counter[tuple[str, str]] = Counter()
    for row in clean_rows:
        for key in (
            ("__ALL__", "all"),
            (row["endpoint_family"], "all"),
            (row["endpoint_family"], row["partition"]),
        ):
            if row["identity_group_id"]:
                identity_groups[key][row["identity_group_id"]] += 1
            else:
                identity_missing[key] += 1
    identity_multiplicity = []
    for key in sorted(set(identity_groups) | set(identity_missing)):
        counts = list(identity_groups[key].values())
        identity_multiplicity.append(
            {
                "endpoint_family": key[0],
                "partition": key[1],
                "identity_groups": len(counts),
                "missing_identity_rows": identity_missing[key],
                "singleton_identities": sum(count == 1 for count in counts),
                "identities_with_2_to_4_rows": sum(2 <= count <= 4 for count in counts),
                "identities_with_5_or_more_rows": sum(count >= 5 for count in counts),
                **prefixed_stats("observations_per_identity", counts),
            }
        )

    critical_fields = [
        "task_id",
        "identity_group_id",
        "representation_text",
        "sequence",
        "source_id",
        "source_group_ids",
        "raw_endpoint_detail",
        "raw_unit",
        "normalized_value",
        "normalized_unit",
        "raw_species",
        "raw_matrix",
        "raw_assay",
        "raw_route",
        "raw_dose",
        "raw_timepoint",
    ]
    missingness_rows = []
    missingness_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        missingness_groups[(row["endpoint_family"], row["partition"])].append(row)
    for key in sorted(missingness_groups):
        rows_for_group = missingness_groups[key]
        for field in critical_fields:
            missing = sum(not clean(row.get(field)) for row in rows_for_group)
            missingness_rows.append(
                {
                    "endpoint_family": key[0],
                    "partition": key[1],
                    "field": field,
                    "rows": len(rows_for_group),
                    "missing_rows": missing,
                    "missing_fraction": ratio_text(missing, len(rows_for_group)),
                }
            )

    anomaly_rows: list[dict[str, Any]] = []

    def add_anomaly(row: dict[str, str], severity: str, rule: str, reason: str) -> None:
        anomaly_rows.append(
            {
                "severity": severity,
                "rule": rule,
                "reason": reason,
                "record_id": row["record_id"],
                "task_id": row["task_id"],
                "endpoint_family": row["endpoint_family"],
                "parameter_semantics": row["parameter_semantics"],
                "raw_endpoint_detail": row["raw_endpoint_detail"],
                "raw_value": row["raw_value"],
                "raw_unit": row["raw_unit"],
                "normalized_value": row["normalized_value"],
                "normalized_unit": row["normalized_unit"],
                "source_id": row["source_id"],
                "doi_pmid": row["doi_pmid"],
                "identity_group_id": row["identity_group_id"],
                "raw_sequence": row["sequence"],
                "representation_type": row["representation_type"],
                "representation_char_length": len(row["representation_text"]),
                "condition_json": row["condition_json"],
                "member_lineage_json": row["member_lineage_json"],
            }
        )

    for row in strict_rows:
        value = parse_number(row["normalized_value"])
        category, normalized_sequence = sequence_class(row["sequence"])
        if category == "standard_20aa" and len(normalized_sequence) > 100:
            add_anomaly(
                row,
                "high",
                "standard_sequence_gt_100_aa_in_strict_numeric",
                f"Standard residue sequence length {len(normalized_sequence)} exceeds peptide scope threshold 100.",
            )
        if (
            row["endpoint_family"] == "T1/2"
            and row["parameter_semantics"] == "protease_intestinal_stability"
            and value is not None
            and value < math.log10(1 / 3600)
        ):
            add_anomaly(
                row,
                "high",
                "subsecond_protease_half_life",
                "Normalized half-life is below one second; verify source parameter semantics and unit against the primary record.",
            )
        if (
            row["endpoint_family"] == "CL"
            and row["normalized_unit"] == "L/h"
            and value is not None
            and value > 100
        ):
            add_anomaly(
                row,
                "high",
                "extreme_total_body_clearance_value_gt_100_L_h",
                "L/h-scale clearance value exceeds 100; verify CL versus CL/F semantics, unit, scale, and extraction.",
            )
        if row["endpoint_family"] in {"F", "PPB"} and value is not None and not 0 <= value <= 1:
            add_anomaly(
                row,
                "critical",
                "fraction_target_out_of_domain",
                "Normalized fraction target is outside [0,1].",
            )
        if len(row["representation_text"]) > 5000:
            add_anomaly(
                row,
                "medium",
                "selected_representation_gt_5000_chars",
                "Selected structural representation is unusually long for peptide modeling.",
            )
    anomaly_rows.sort(key=lambda row: (row["severity"], row["rule"], row["record_id"]))

    outputs: dict[str, dict[str, Any]] = {}

    def emit(name: str, rows: list[dict[str, Any]], preferred: list[str]) -> None:
        path = output_dir / name
        count = write_tsv(path, rows, preferred)
        outputs[name] = artifact(path, count)

    emit(
        "v15_raw_endpoint_inventory.tsv",
        v15_endpoint_rows,
        ["raw_endpoint", "rows", "numeric_rows", "label_rows"],
    )
    emit(
        "v15_raw_numeric_distribution.tsv",
        raw_numeric_rows,
        ["raw_endpoint", "task_type", "raw_unit", "raw_relation", "descriptive_only", *STAT_KEYS],
    )
    emit(
        "v15_label_distribution.tsv",
        v15_label_rows,
        ["raw_endpoint", "task_type", "label", "rows"],
    )
    emit(
        "clean_endpoint_inventory.tsv",
        clean_endpoint_rows,
        ["endpoint_family", "input_member_rows", "representative_rows", "unique_identity_groups", "tasks"],
    )
    emit(
        "literature_review_distribution.tsv",
        literature_distribution,
        ["data_layer", "endpoint", "endpoint_detail", "rows", "training_rows"],
    )
    emit(
        "sequence_representation_summary.tsv",
        sequence_summary_rows,
        ["data_layer", "endpoint", "partition", "rows", "member_rows"],
    )
    emit(
        "selected_representation_distribution.tsv",
        selected_representation_rows,
        [
            "endpoint_family",
            "partition",
            "representation_type",
            "rows",
            "member_rows",
            "unique_identity_groups",
            "unique_representations",
        ],
    )
    emit(
        "sequence_length_histogram.tsv",
        sequence_histogram_rows,
        ["data_layer", "endpoint", "partition", "length_metric", "length", "representative_rows", "member_rows"],
    )
    emit(
        "amino_acid_composition.tsv",
        amino_acid_rows,
        ["data_layer", "endpoint", "residue"],
    )
    emit(
        "strict_numeric_endpoint_unit_distribution.tsv",
        endpoint_unit_rows,
        ["endpoint_family", "parameter_semantics", "parameter_basis", "assay_family", "normalized_unit", "target_transform", "descriptive_only", *STAT_KEYS],
    )
    emit(
        "strict_numeric_task_distribution.tsv",
        task_numeric_rows,
        ["task_id", "endpoint_family", "parameter_semantics", "normalized_unit", "strict_modelable", "size_bin", *STAT_KEYS],
    )
    emit(
        "strict_numeric_histograms.tsv",
        histogram_rows,
        ["scope", "group_id", "endpoint_family", "normalized_unit", "n", "bin_index", "lower_inclusive", "upper", "upper_inclusive", "count"],
    )
    emit(
        "censoring_distribution.tsv",
        censoring_distribution,
        ["task_id", "endpoint_family", "normalized_unit", "normalized_relation", "censoring_type", "point_regression_eligible", *STAT_KEYS],
    )
    emit(
        "binary_task_distribution.tsv",
        binary_distribution,
        ["task_id", "endpoint_family", "task_kind", "binary_use_tiers", "rows", "positive_rows", "negative_rows", "positive_rate_among_labeled"],
    )
    emit(
        "task_size_viability.tsv",
        task_viability,
        ["task_id", "endpoint_family", "task_kind", "rows", "strict_numeric_rows", "positive_rows", "negative_rows", "strict_modelable", "research_modelable"],
    )
    emit(
        "identity_multiplicity_distribution.tsv",
        identity_multiplicity,
        ["endpoint_family", "partition", "identity_groups", "missing_identity_rows", "singleton_identities"],
    )
    emit(
        "field_missingness.tsv",
        missingness_rows,
        ["endpoint_family", "partition", "field", "rows", "missing_rows", "missing_fraction"],
    )
    emit(
        "row_anomalies.tsv",
        anomaly_rows,
        ["severity", "rule", "reason", "record_id", "task_id", "endpoint_family", "raw_value", "raw_unit", "normalized_value", "normalized_unit", "source_id"],
    )

    partition_counts = Counter(row["partition"] for row in clean_rows)
    strict_endpoint_counts = Counter(row["endpoint_family"] for row in strict_rows)
    task_size_counts = Counter(size_bin(len(values)) for values in task_values.values())
    representation_counts = Counter(
        row["representation_type"] or "missing" for row in clean_rows
    )
    anomaly_counts = Counter(row["rule"] for row in anomaly_rows)
    assertions = {
        "v15_rows_215568": v15_rows == 215568,
        "v15_raw_endpoints_50": len(v15_endpoint_rows) == 50,
        "clean_representatives_19979": len(clean_rows) == 19979,
        "clean_partition_conservation": sum(partition_counts.values()) == len(clean_rows),
        "strict_numeric_11591": len(strict_rows) == 11591,
        "strict_numeric_all_finite": all(
            parse_number(row["normalized_value"]) is not None for row in strict_rows
        ),
        "strict_numeric_no_hard_conflict": all(
            row["hard_condition_conflict"] == "false" for row in strict_rows
        ),
        "strict_task_unit_transform_invariant": all(
            len(
                {
                    (row["normalized_unit"], row["target_transform"])
                    for row, _ in pairs
                }
            )
            == 1
            for pairs in task_values.values()
        ),
        "binary_not_mixed_into_numeric": all(
            row["task_kind"] == "regression" for row in strict_rows
        ),
        "kp_zero_rows_explicit": next(
            row for row in clean_endpoint_rows if row["endpoint_family"] == "Kp"
        )["representative_rows"]
        == 0,
        "missing_identity_not_counted_as_identity": next(
            row
            for row in identity_multiplicity
            if row["endpoint_family"] == "__ALL__" and row["partition"] == "all"
        )["identity_groups"]
        == 14374,
        "literature_rows_1560_review_only": len(literature_rows) == 1560
        and all(row["research_train_eligible"] == "false" for row in literature_rows),
        "sequence_representation_prefixes_valid": all(
            not row["representation_text"]
            or row["representation_text"].startswith(
                {
                    "sequence": "SEQ:",
                    "helm": "HELM:",
                    "smiles": "SMILES:",
                }[row["representation_type"]]
            )
            for row in clean_rows
        ),
    }
    if not all(assertions.values()):
        failed = sorted(key for key, value in assertions.items() if not value)
        raise ValueError("Statistical summary assertions failed: " + ", ".join(failed))

    readme = f"""# Peptide ML Cleaning v1 Statistics

This deterministic statistical release covers all {len(v15_endpoint_rows)} raw endpoints in the frozen V15 training-deduplicated table ({v15_rows:,} rows) and all declared ML-clean endpoint families, including the zero-row Kp family.

- ML-clean representatives: {len(clean_rows):,}
- Strict numeric: {len(strict_rows):,}
- Binary evidence catalog: {len(binary_rows):,}
- Positive-unlabeled: {len(pu_rows):,}
- Censored: {len(censored_rows):,}
- Literature review is reported separately: {len(literature_rows):,}

Never pool `normalized_value` across task kinds or incompatible units. Use `strict_numeric_task_distribution.tsv` for model-facing numeric diagnostics. Endpoint-unit summaries are descriptive only. Sequence projections, selected model representations, and identity groups are reported separately.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    outputs["README.md"] = artifact(readme_path)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "v15_training_deduplicated": artifact(v15_input, v15_rows),
            "ml_clean_normalized": artifact(normalized_path, len(clean_rows)),
            "ml_clean_task_registry": artifact(registry_path, len(registry)),
            "literature_review": artifact(literature_path, len(literature_rows)),
            "cleaning_manifest": artifact(cleaning_manifest_path),
        },
        "counts": {
            "v15_rows": v15_rows,
            "v15_raw_endpoints": len(v15_endpoint_rows),
            "clean_declared_endpoint_families": len(CLEAN_ENDPOINT_FAMILIES),
            "clean_nonzero_endpoint_families": sum(
                row["representative_rows"] > 0 for row in clean_endpoint_rows
            ),
            "clean_representatives": len(clean_rows),
            "partitions": dict(sorted(partition_counts.items())),
            "strict_numeric_by_endpoint": dict(sorted(strict_endpoint_counts.items())),
            "tasks": len(registry),
            "strict_numeric_tasks": len(task_values),
            "strict_task_size_bins": dict(sorted(task_size_counts.items())),
            "representation_types": dict(sorted(representation_counts.items())),
            "unique_identity_groups": len(
                {row["identity_group_id"] for row in clean_rows if row["identity_group_id"]}
            ),
            "unique_selected_representations": len(
                {row["representation_text"] for row in clean_rows if row["representation_text"]}
            ),
            "sequence_model_eligible_rows": sum(
                row["sequence_model_eligible"] == "true" for row in clean_rows
            ),
            "structure_model_eligible_rows": sum(
                row["structure_model_eligible"] == "true" for row in clean_rows
            ),
            "literature_review_rows": len(literature_rows),
            "anomaly_rows": len(anomaly_rows),
            "anomaly_unique_records": len(
                {row["record_id"] for row in anomaly_rows}
            ),
            "anomaly_rules": dict(sorted(anomaly_counts.items())),
        },
        "scope_notes": {
            "numeric": "Strict values are summarized by task and compatible endpoint/semantics/unit/transform groups only.",
            "sequence": "Raw sequence projections, selected model representations, and identity groups are distinct denominators.",
            "binary": "Weak, derived, and positive-unlabeled tasks are never mixed with continuous numeric statistics.",
            "literature": "The 1,560 literature rows are a separate review layer and are not added to ML-clean training counts.",
        },
        "assertions": assertions,
        "outputs": outputs,
        "implementation": {
            "script": display_path(Path(__file__)),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build endpoint, sequence, label, and numeric distributions for peptide ML cleaning v1."
    )
    parser.add_argument("--v15-input", type=Path, default=DEFAULT_V15_INPUT)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args.v15_input, args.clean_dir, args.output_dir)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
