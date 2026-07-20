"""Fail-closed dataset adapters for Peptide-OmniPanel observations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .representations import is_sentinel


class DatasetValidationError(ValueError):
    """Raised when a dataset row cannot be interpreted without guessing."""


KNOWN_PARTITIONS = frozenset(
    {
        "strict_numeric",
        "binary_evidence_catalog",
        "censored_numeric",
        "positive_unlabeled",
    }
)
KNOWN_SPLITS = frozenset({"development", "calibration", "sealed"})
KNOWN_TRANSFORMS = frozenset({"identity", "log10"})
USE_CASES = frozenset({"internal_research", "redistribution", "production"})

ACTION_ALLOW = "allow"
ACTION_ALLOW_WITH_ATTRIBUTION = "allow_with_attribution"
ACTION_ALLOW_NONCOMMERCIAL = "allow_noncommercial"
ACTION_INTERNAL_ONLY = "internal_only"
ACTION_REVIEW_REQUIRED = "review_required"
ACTION_DENY = "deny"

COMMON_REQUIRED_COLUMNS = frozenset(
    {
        "clean_row_id",
        "record_id",
        "endpoint_family",
        "task_id",
        "task_kind",
        "normalized_value",
        "normalized_lower",
        "normalized_upper",
        "normalized_relation",
        "normalized_unit",
        "target_transform",
        "censoring_type",
        "raw_label",
        "research_train_eligible",
        "production_train_eligible",
        "license",
        "identity_group_id",
        "partition",
        "sequence",
        "helm",
        "smiles",
        "representation_type",
        "representation_text",
        "source_group_ids",
    }
)


@dataclass(frozen=True)
class LicenseActions:
    license_text: str
    internal_research: str
    redistribution: str
    production: str
    reasons: tuple[str, ...] = ()

    def action_for(self, use_case: str) -> str:
        if use_case not in USE_CASES:
            raise DatasetValidationError(f"unknown use case: {use_case!r}")
        return str(getattr(self, use_case))


@dataclass(frozen=True)
class Observation:
    observation_id: str
    record_id: str
    partition: str
    target_kind: str
    endpoint_family: str
    task_id: str
    task_kind: str
    value: float | None
    label: int | None
    lower_bound: float | None
    upper_bound: float | None
    relation: str
    unit: str | None
    target_transform: str
    censoring_type: str
    identity_group_id: str
    sequence: str | None
    helm: str | None
    smiles: str | None
    representation_type: str | None
    representation_text: str | None
    source_group_ids: tuple[str, ...]
    license_text: str
    research_train_eligible: bool
    production_train_eligible: str
    license_actions: LicenseActions
    split_id: str | None = None
    component_id: str | None = None
    sequence_model_eligible: bool | None = None
    structure_model_eligible: bool | None = None
    representation_equivalence_verified: bool | None = None
    projection_is_lossy: bool | None = None

    @property
    def model_target(self) -> float | int | None:
        return self.label if self.label is not None else self.value


def read_tsv(
    path: str | Path,
    *,
    required_columns: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Read a UTF-8 TSV and fail on missing/duplicate/ragged columns."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DatasetValidationError(f"TSV is empty: {input_path}") from exc
        if not header or any(not column.strip() for column in header):
            raise DatasetValidationError(f"TSV contains an empty header column: {input_path}")
        duplicates = sorted(column for column, count in Counter(header).items() if count > 1)
        if duplicates:
            raise DatasetValidationError(f"TSV has duplicate columns: {duplicates}")
        required = set(required_columns)
        missing = sorted(required - set(header))
        if missing:
            raise DatasetValidationError(f"TSV misses required columns: {missing}")

        rows: list[dict[str, str]] = []
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise DatasetValidationError(
                    f"TSV row {line_number} has {len(fields)} fields; expected {len(header)}"
                )
            rows.append(dict(zip(header, fields)))
    return rows


def parse_bool(value: object, *, field: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise DatasetValidationError(f"{field} must be exactly true or false, got {value!r}")


def parse_optional_bool(value: object, *, field: str) -> bool | None:
    if is_sentinel(value):
        return None
    return parse_bool(value, field=field)


def parse_optional_float(value: object, *, field: str) -> float | None:
    if is_sentinel(value):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{field} must be a finite number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise DatasetValidationError(f"{field} must be finite, got {value!r}")
    return parsed


def parse_required_float(value: object, *, field: str) -> float:
    parsed = parse_optional_float(value, field=field)
    if parsed is None:
        raise DatasetValidationError(f"{field} is required")
    return parsed


def transform_value(value: float, transform: str) -> float:
    normalized = str(transform).strip().casefold()
    if normalized == "identity":
        return float(value)
    if normalized == "log10":
        if value <= 0:
            raise DatasetValidationError("log10 target transform requires a positive value")
        return math.log10(value)
    raise DatasetValidationError(f"unsupported target transform: {transform!r}")


def inverse_transform_value(value: float, transform: str) -> float:
    normalized = str(transform).strip().casefold()
    if normalized == "identity":
        return float(value)
    if normalized == "log10":
        return 10.0 ** float(value)
    raise DatasetValidationError(f"unsupported target transform: {transform!r}")


def _technical_eligibility(value: object, *, field: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "eligible", "allow", "allowed"}:
        return "eligible"
    if normalized in {"false", "ineligible", "deny", "denied"}:
        return "ineligible"
    if normalized.startswith("not_assessed") or normalized in {"review", "review_required"}:
        return "not_assessed"
    raise DatasetValidationError(f"unsupported {field} state: {value!r}")


def resolve_license_actions(
    license_text: object,
    *,
    research_train_eligible: bool,
    production_train_eligible: object,
) -> LicenseActions:
    """Resolve conservative actions without treating technical eligibility as a license."""

    rendered = "" if license_text is None else str(license_text).strip()
    normalized = rendered.casefold()
    production_state = _technical_eligibility(
        production_train_eligible,
        field="production_train_eligible",
    )
    reasons: list[str] = []

    explicit_internal_only = (
        "internal_review_only" in normalized
        or "sequence_license_excluded" in normalized
    )
    explicit_no_reuse = "no cc reuse license" in normalized
    unidentified = (
        not rendered
        or "not identified" in normalized
        or "verify before use" in normalized
        or "no explicit" in normalized
        or "underlying source terms apply" in normalized
    )
    is_cc0 = "cc0" in normalized and not explicit_internal_only
    is_cc_by_nc = "cc by-nc" in normalized or "cc by nc" in normalized
    is_cc_by = "cc by" in normalized and not is_cc_by_nc
    is_nist_open = "nist open access" in normalized

    if explicit_internal_only:
        base_internal = ACTION_INTERNAL_ONLY
        base_redistribution = ACTION_DENY
        base_production = ACTION_DENY
        reasons.append("license_text_restricts_row_release")
    elif explicit_no_reuse:
        base_internal = ACTION_REVIEW_REQUIRED
        base_redistribution = ACTION_DENY
        base_production = ACTION_DENY
        reasons.append("publisher_copyright_no_reuse_license")
    elif unidentified:
        base_internal = ACTION_REVIEW_REQUIRED
        base_redistribution = ACTION_REVIEW_REQUIRED
        base_production = ACTION_REVIEW_REQUIRED
        reasons.append("license_or_upstream_terms_not_cleared")
    elif is_cc0:
        base_internal = ACTION_ALLOW
        base_redistribution = ACTION_ALLOW
        base_production = ACTION_ALLOW
    elif is_cc_by_nc:
        base_internal = ACTION_ALLOW_NONCOMMERCIAL
        base_redistribution = ACTION_ALLOW_NONCOMMERCIAL
        base_production = ACTION_DENY
        reasons.append("noncommercial_restriction")
    elif is_cc_by or is_nist_open:
        base_internal = ACTION_ALLOW_WITH_ATTRIBUTION
        base_redistribution = ACTION_ALLOW_WITH_ATTRIBUTION
        base_production = ACTION_ALLOW_WITH_ATTRIBUTION
    else:
        base_internal = ACTION_REVIEW_REQUIRED
        base_redistribution = ACTION_REVIEW_REQUIRED
        base_production = ACTION_REVIEW_REQUIRED
        reasons.append("unrecognized_license_text")

    if research_train_eligible:
        internal_action = base_internal
    else:
        internal_action = ACTION_DENY
        reasons.append("research_train_ineligible")

    if production_state == "eligible":
        production_action = base_production
    elif production_state == "ineligible":
        production_action = ACTION_DENY
        reasons.append("production_train_ineligible")
    else:
        production_action = (
            ACTION_DENY if base_production == ACTION_DENY else ACTION_REVIEW_REQUIRED
        )
        reasons.append("production_train_not_assessed")

    return LicenseActions(
        license_text=rendered,
        internal_research=internal_action,
        redistribution=base_redistribution,
        production=production_action,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def action_allows_use(action: str, use_case: str) -> bool:
    if use_case not in USE_CASES:
        raise DatasetValidationError(f"unknown use case: {use_case!r}")
    if use_case == "internal_research":
        return action in {
            ACTION_ALLOW,
            ACTION_ALLOW_WITH_ATTRIBUTION,
            ACTION_ALLOW_NONCOMMERCIAL,
            ACTION_INTERNAL_ONLY,
        }
    if use_case == "redistribution":
        return action in {
            ACTION_ALLOW,
            ACTION_ALLOW_WITH_ATTRIBUTION,
            ACTION_ALLOW_NONCOMMERCIAL,
        }
    return action in {ACTION_ALLOW, ACTION_ALLOW_WITH_ATTRIBUTION}


def _optional_text(value: object) -> str | None:
    return None if is_sentinel(value) else str(value).strip()


def _parse_source_groups(value: object) -> tuple[str, ...]:
    if is_sentinel(value):
        return ()
    rendered = str(value).strip()
    if rendered.startswith("["):
        try:
            parsed = json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError("source_group_ids contains invalid JSON") from exc
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise DatasetValidationError("source_group_ids JSON must be a list of strings")
        return tuple(dict.fromkeys(item.strip() for item in parsed if item.strip()))
    separator = "|" if "|" in rendered else ";" if ";" in rendered else None
    values = [rendered] if separator is None else rendered.split(separator)
    return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))


def _require_text(row: Mapping[str, str], field: str) -> str:
    value = row.get(field)
    if is_sentinel(value):
        raise DatasetValidationError(f"{field} is required")
    return str(value).strip()


def _validate_transform(value: object) -> str:
    transform = str(value).strip().casefold()
    if transform not in KNOWN_TRANSFORMS:
        raise DatasetValidationError(f"unsupported target_transform: {value!r}")
    return transform


def _same_numeric_value(left: float, right: float) -> bool:
    # Exported decimal strings can have different display precision.
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def observation_from_row(row: Mapping[str, str]) -> Observation:
    missing_columns = sorted(COMMON_REQUIRED_COLUMNS - set(row))
    if missing_columns:
        raise DatasetValidationError(f"observation row misses columns: {missing_columns}")

    partition = _require_text(row, "partition")
    if partition not in KNOWN_PARTITIONS:
        raise DatasetValidationError(f"unsupported partition: {partition!r}")
    task_kind = _require_text(row, "task_kind")
    transform = _validate_transform(row["target_transform"])
    research_eligible = parse_bool(
        row["research_train_eligible"], field="research_train_eligible"
    )
    production_eligible = _require_text(row, "production_train_eligible")
    actions = resolve_license_actions(
        row["license"],
        research_train_eligible=research_eligible,
        production_train_eligible=production_eligible,
    )

    value = parse_optional_float(row["normalized_value"], field="normalized_value")
    lower = parse_optional_float(row["normalized_lower"], field="normalized_lower")
    upper = parse_optional_float(row["normalized_upper"], field="normalized_upper")
    relation = _require_text(row, "normalized_relation")
    censoring_type = _require_text(row, "censoring_type")
    label: int | None = None

    if partition == "strict_numeric":
        if task_kind != "regression" or relation != "=" or censoring_type != "exact":
            raise DatasetValidationError("strict_numeric row must be exact regression with '=' relation")
        if value is None:
            raise DatasetValidationError("strict_numeric normalized_value is required")
        if not is_sentinel(row["raw_label"]):
            raise DatasetValidationError("strict_numeric raw_label must be empty")
        if lower is not None and not _same_numeric_value(lower, value):
            raise DatasetValidationError("strict_numeric lower bound must equal normalized_value")
        if upper is not None and not _same_numeric_value(upper, value):
            raise DatasetValidationError("strict_numeric upper bound must equal normalized_value")
        target_kind = "strict_regression"
    elif partition == "binary_evidence_catalog":
        if task_kind != "binary_classification" or relation != "label":
            raise DatasetValidationError("binary row must have binary_classification/label semantics")
        if censoring_type != "not_applicable" or transform != "identity":
            raise DatasetValidationError("binary row requires identity transform and no censoring")
        raw_label = str(row["raw_label"]).strip()
        if raw_label not in {"0", "1"}:
            raise DatasetValidationError(f"binary raw_label must be 0 or 1, got {raw_label!r}")
        label = int(raw_label)
        if value is None or value != float(label):
            raise DatasetValidationError("binary normalized_value must match raw_label")
        if lower is not None or upper is not None:
            raise DatasetValidationError("binary rows must not contain numeric bounds")
        target_kind = "binary"
    elif partition == "positive_unlabeled":
        if task_kind != "positive_unlabeled" or relation != "label":
            raise DatasetValidationError("PU row must have positive_unlabeled/label semantics")
        if censoring_type != "not_applicable" or transform != "identity":
            raise DatasetValidationError("PU row requires identity transform and no censoring")
        if str(row["raw_label"]).strip() != "1" or value != 1.0:
            raise DatasetValidationError("PU catalog rows must be observed positives with label 1")
        label = 1
        if lower is not None or upper is not None:
            raise DatasetValidationError("PU rows must not contain numeric bounds")
        target_kind = "positive_unlabeled"
    else:
        allowed_relations = {"<", "<=", ">", ">=", "~"}
        if task_kind != "regression" or relation not in allowed_relations:
            raise DatasetValidationError("censored row has unsupported task/relation semantics")
        if value is None:
            raise DatasetValidationError("censored normalized_value is required")
        if not is_sentinel(row["raw_label"]):
            raise DatasetValidationError("censored raw_label must be empty")
        if censoring_type == "lower_bound":
            if relation not in {">", ">="} or lower is None or upper is not None:
                raise DatasetValidationError("lower_bound row requires >/>=, lower value, and no upper")
            target_kind = "censored_regression"
        elif censoring_type == "upper_bound":
            if relation not in {"<", "<="} or upper is None or lower is not None:
                raise DatasetValidationError("upper_bound row requires </<=, upper value, and no lower")
            target_kind = "censored_regression"
        elif censoring_type == "approximate":
            if relation != "~":
                raise DatasetValidationError("approximate row requires '~' relation")
            target_kind = "approximate_regression"
        else:
            raise DatasetValidationError(f"unsupported censored type: {censoring_type!r}")

    split_id = _optional_text(row.get("split_id"))
    if split_id is not None and split_id not in KNOWN_SPLITS:
        raise DatasetValidationError(f"unsupported split_id: {split_id!r}")

    return Observation(
        observation_id=_require_text(row, "clean_row_id"),
        record_id=_require_text(row, "record_id"),
        partition=partition,
        target_kind=target_kind,
        endpoint_family=_require_text(row, "endpoint_family"),
        task_id=_require_text(row, "task_id"),
        task_kind=task_kind,
        value=value,
        label=label,
        lower_bound=lower,
        upper_bound=upper,
        relation=relation,
        unit=_optional_text(row["normalized_unit"]),
        target_transform=transform,
        censoring_type=censoring_type,
        identity_group_id=_require_text(row, "identity_group_id"),
        sequence=_optional_text(row["sequence"]),
        helm=_optional_text(row["helm"]),
        smiles=_optional_text(row["smiles"]),
        representation_type=_optional_text(row["representation_type"]),
        representation_text=_optional_text(row["representation_text"]),
        source_group_ids=_parse_source_groups(row["source_group_ids"]),
        license_text=str(row["license"]).strip(),
        research_train_eligible=research_eligible,
        production_train_eligible=production_eligible,
        license_actions=actions,
        split_id=split_id,
        component_id=_optional_text(row.get("component_id")),
        sequence_model_eligible=parse_optional_bool(
            row.get("sequence_model_eligible"),
            field="sequence_model_eligible",
        ),
        structure_model_eligible=parse_optional_bool(
            row.get("structure_model_eligible"),
            field="structure_model_eligible",
        ),
        representation_equivalence_verified=parse_optional_bool(
            row.get("representation_equivalence_verified"),
            field="representation_equivalence_verified",
        ),
        projection_is_lossy=parse_optional_bool(
            row.get("projection_is_lossy"),
            field="projection_is_lossy",
        ),
    )


def load_observations(path: str | Path) -> list[Observation]:
    rows = read_tsv(path, required_columns=COMMON_REQUIRED_COLUMNS)
    observations: list[Observation] = []
    seen_ids: set[str] = set()
    for line_offset, row in enumerate(rows, start=2):
        try:
            observation = observation_from_row(row)
        except DatasetValidationError as exc:
            raise DatasetValidationError(f"{Path(path)} row {line_offset}: {exc}") from exc
        if observation.observation_id in seen_ids:
            raise DatasetValidationError(
                f"{Path(path)} has duplicate clean_row_id: {observation.observation_id}"
            )
        seen_ids.add(observation.observation_id)
        observations.append(observation)
    return observations


def read_split_assignments(
    path: str | Path,
    *,
    key_column: str = "clean_row_id",
    split_column: str = "split_id",
) -> dict[str, str]:
    rows = read_tsv(path, required_columns={key_column, split_column})
    assignments: dict[str, str] = {}
    for line_offset, row in enumerate(rows, start=2):
        key = _require_text(row, key_column)
        split_id = _require_text(row, split_column)
        if split_id not in KNOWN_SPLITS:
            raise DatasetValidationError(
                f"{Path(path)} row {line_offset}: unsupported split {split_id!r}"
            )
        if key in assignments:
            raise DatasetValidationError(f"duplicate split assignment for {key!r}")
        assignments[key] = split_id
    return assignments


def merge_p0_splits(
    observations: Sequence[Observation],
    split_assignments: Mapping[str, str],
    *,
    observation_key: str = "observation_id",
    component_assignments: Mapping[str, str] | None = None,
) -> list[Observation]:
    if observation_key not in {"observation_id", "record_id", "identity_group_id"}:
        raise DatasetValidationError(f"unsupported observation split key: {observation_key!r}")
    merged: list[Observation] = []
    for observation in observations:
        key = str(getattr(observation, observation_key))
        if key not in split_assignments:
            raise DatasetValidationError(f"missing P0 split assignment for {observation_key}={key!r}")
        split_id = str(split_assignments[key]).strip()
        if split_id not in KNOWN_SPLITS:
            raise DatasetValidationError(f"unsupported P0 split: {split_id!r}")
        if observation.split_id is not None and observation.split_id != split_id:
            raise DatasetValidationError(
                f"conflicting split for {observation.observation_id}: "
                f"{observation.split_id!r} != {split_id!r}"
            )
        component_id = observation.component_id
        if component_assignments is not None:
            if key not in component_assignments:
                raise DatasetValidationError(
                    f"missing P0 component assignment for {observation_key}={key!r}"
                )
            assigned_component = str(component_assignments[key]).strip()
            if not assigned_component:
                raise DatasetValidationError(f"empty P0 component assignment for {key!r}")
            if component_id is not None and component_id != assigned_component:
                raise DatasetValidationError(
                    f"conflicting component for {observation.observation_id}: "
                    f"{component_id!r} != {assigned_component!r}"
                )
            component_id = assigned_component
        merged.append(replace(observation, split_id=split_id, component_id=component_id))

    for field in ("record_id", "identity_group_id"):
        splits_by_group: dict[str, set[str]] = defaultdict(set)
        for observation in merged:
            assert observation.split_id is not None
            splits_by_group[str(getattr(observation, field))].add(observation.split_id)
        crossing = sorted(group for group, values in splits_by_group.items() if len(values) > 1)
        if crossing:
            raise DatasetValidationError(
                f"P0 split assignments cross {field} groups: {crossing[:5]}"
            )
    if component_assignments is not None:
        splits_by_component: dict[str, set[str]] = defaultdict(set)
        for observation in merged:
            assert observation.split_id is not None
            assert observation.component_id is not None
            splits_by_component[observation.component_id].add(observation.split_id)
        crossing = sorted(
            component
            for component, values in splits_by_component.items()
            if len(values) > 1
        )
        if crossing:
            raise DatasetValidationError(
                f"P0 split assignments cross component_id groups: {crossing[:5]}"
            )
    return merged


def load_observations_with_splits(
    observation_path: str | Path,
    split_path: str | Path,
    *,
    split_key_column: str = "clean_row_id",
    split_column: str = "split_id",
    observation_key: str = "observation_id",
    component_column: str | None = "component_id",
) -> list[Observation]:
    observations = load_observations(observation_path)
    assignments = read_split_assignments(
        split_path,
        key_column=split_key_column,
        split_column=split_column,
    )
    component_assignments: dict[str, str] | None = None
    if component_column is not None:
        split_rows = read_tsv(split_path, required_columns={split_key_column})
        if split_rows and component_column in split_rows[0]:
            component_assignments = {}
            for row in split_rows:
                key = _require_text(row, split_key_column)
                component = _require_text(row, component_column)
                if key in component_assignments:
                    raise DatasetValidationError(f"duplicate component assignment for {key!r}")
                component_assignments[key] = component
    return merge_p0_splits(
        observations,
        assignments,
        observation_key=observation_key,
        component_assignments=component_assignments,
    )


def select_model_tasks(
    observations: Iterable[Observation],
    *,
    use_case: str = "internal_research",
    endpoint_families: Iterable[str] | None = None,
    target_kinds: Iterable[str] | None = None,
    task_ids: Iterable[str] | None = None,
    min_observations: int = 1,
    min_identities: int = 1,
    require_split: bool = True,
) -> dict[str, list[Observation]]:
    """Select task rows only when use, labels, and split contracts are satisfied."""

    if use_case not in USE_CASES:
        raise DatasetValidationError(f"unknown use case: {use_case!r}")
    if min_observations < 1 or min_identities < 1:
        raise DatasetValidationError("minimum observation and identity counts must be positive")
    endpoint_filter = None if endpoint_families is None else set(endpoint_families)
    target_filter = None if target_kinds is None else set(target_kinds)
    task_filter = None if task_ids is None else set(task_ids)

    grouped: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        if endpoint_filter is not None and observation.endpoint_family not in endpoint_filter:
            continue
        if target_filter is not None and observation.target_kind not in target_filter:
            continue
        if task_filter is not None and observation.task_id not in task_filter:
            continue
        if require_split and observation.split_id is None:
            continue
        action = observation.license_actions.action_for(use_case)
        if not action_allows_use(action, use_case):
            continue
        grouped[observation.task_id].append(observation)

    selected: dict[str, list[Observation]] = {}
    for task_id in sorted(grouped):
        rows = sorted(grouped[task_id], key=lambda row: row.observation_id)
        identity_count = len({row.identity_group_id for row in rows})
        if len(rows) >= min_observations and identity_count >= min_identities:
            selected[task_id] = rows
    return selected


def build_license_action_matrix(
    observations: Iterable[Observation],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], int] = Counter()
    reasons_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for observation in observations:
        actions = observation.license_actions
        key = (
            actions.license_text,
            actions.internal_research,
            actions.redistribution,
            actions.production,
        )
        grouped[key] += 1
        reasons_by_key[key].update(actions.reasons)
    return [
        {
            "license": key[0],
            "internal_research": key[1],
            "redistribution": key[2],
            "production": key[3],
            "observation_count": grouped[key],
            "reasons": sorted(reasons_by_key[key]),
        }
        for key in sorted(grouped)
    ]


# Short aliases for script-oriented callers.
load_tsv = read_tsv
merge_splits = merge_p0_splits
load_with_splits = load_observations_with_splits
select_tasks = select_model_tasks
