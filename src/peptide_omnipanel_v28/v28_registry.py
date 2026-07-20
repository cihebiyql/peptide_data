"""Strict readers for the V2.8 self-trained endpoint and source registries.

The registries are intentionally data-only.  They describe which locally
trained or locally implemented route is allowed for an endpoint; they never
resolve a third-party model, execute a network call, or deserialize a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_TIERS = frozenset({"A", "B", "C", "D", "E"})
FORBIDDEN_SOURCE_KINDS = frozenset({"third_party_model_weight", "prediction_api"})


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array of non-empty strings")
    items = tuple(_text(item, field) for item in value)
    if not items:
        raise ValueError(f"{field} must not be empty")
    if len(items) != len(set(items)):
        raise ValueError(f"{field} must not contain duplicates")
    return items


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


@dataclass(frozen=True)
class EndpointSpec:
    """One semantic endpoint with its legal locally owned fallback chain."""

    endpoint_id: str
    panel: str
    task_kind: str
    primary_output: str
    unit: str
    required_representation: tuple[str, ...]
    fallback_policy: tuple[str, ...]
    allowed_tiers: tuple[str, ...]
    semantic_constraints: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EndpointSpec:
        required_representation = _text_list(
            value.get("required_representation"), "required_representation"
        )
        fallback_policy = _text_list(value.get("fallback_policy"), "fallback_policy")
        allowed_tiers = _text_list(value.get("allowed_tiers"), "allowed_tiers")
        invalid_tiers = set(allowed_tiers).difference(ALLOWED_TIERS)
        if invalid_tiers:
            raise ValueError(f"allowed_tiers contains unsupported tiers: {sorted(invalid_tiers)}")
        if fallback_policy[-1] != "E":
            raise ValueError("fallback_policy must end with the E-tier project prior")
        if any(route not in ALLOWED_TIERS for route in fallback_policy):
            raise ValueError("fallback_policy may contain only A/B/C/D/E tiers")
        if not set(fallback_policy).issubset(allowed_tiers):
            raise ValueError("fallback_policy tiers must be declared in allowed_tiers")
        return cls(
            endpoint_id=_text(value.get("endpoint_id"), "endpoint_id"),
            panel=_text(value.get("panel"), "panel"),
            task_kind=_text(value.get("task_kind"), "task_kind"),
            primary_output=_text(value.get("primary_output"), "primary_output"),
            unit=_text(value.get("unit"), "unit"),
            required_representation=required_representation,
            fallback_policy=fallback_policy,
            allowed_tiers=allowed_tiers,
            semantic_constraints=_object(
                value.get("semantic_constraints", {}), "semantic_constraints"
            ),
        )


@dataclass(frozen=True)
class EndpointRegistry:
    schema_version: str
    ownership_contract: Mapping[str, Any]
    endpoints: tuple[EndpointSpec, ...]

    def __post_init__(self) -> None:
        if self.ownership_contract.get("third_party_prediction_api_allowed") is not False:
            raise ValueError("registry must explicitly forbid third-party prediction APIs")
        if self.ownership_contract.get("third_party_model_weights_allowed_in_release") is not False:
            raise ValueError("registry must explicitly forbid third-party model weights")
        endpoint_ids = [item.endpoint_id for item in self.endpoints]
        if not endpoint_ids:
            raise ValueError("endpoint registry must contain at least one endpoint")
        if len(endpoint_ids) != len(set(endpoint_ids)):
            raise ValueError("endpoint registry endpoint_id values must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EndpointRegistry:
        endpoints = value.get("endpoints")
        if not isinstance(endpoints, Sequence) or isinstance(endpoints, (str, bytes, bytearray)):
            raise ValueError("endpoints must be an array")
        return cls(
            schema_version=_text(value.get("schema_version"), "schema_version"),
            ownership_contract=_object(value.get("ownership_contract"), "ownership_contract"),
            endpoints=tuple(
                EndpointSpec.from_mapping(_object(item, "endpoint")) for item in endpoints
            ),
        )

    def endpoint(self, endpoint_id: str) -> EndpointSpec:
        for item in self.endpoints:
            if item.endpoint_id == endpoint_id:
                return item
        raise KeyError(endpoint_id)


@dataclass(frozen=True)
class SourceSpec:
    """One raw-data source.  It may never introduce a model-weight dependency."""

    source_id: str
    source_kind: str
    raw_data_allowed: bool
    license_status: str
    use_policy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceSpec:
        source_kind = _text(value.get("source_kind"), "source_kind")
        if source_kind in FORBIDDEN_SOURCE_KINDS:
            raise ValueError(f"forbidden source_kind in self-trained plan: {source_kind}")
        allowed = value.get("raw_data_allowed")
        if not isinstance(allowed, bool):
            raise ValueError("raw_data_allowed must be a boolean")
        return cls(
            source_id=_text(value.get("source_id"), "source_id"),
            source_kind=source_kind,
            raw_data_allowed=allowed,
            license_status=_text(value.get("license_status"), "license_status"),
            use_policy=_text(value.get("use_policy"), "use_policy"),
        )


@dataclass(frozen=True)
class SourceRegistry:
    schema_version: str
    sources: tuple[SourceSpec, ...]

    def __post_init__(self) -> None:
        source_ids = [item.source_id for item in self.sources]
        if not source_ids:
            raise ValueError("source registry must contain at least one source")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source registry source_id values must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceRegistry:
        sources = value.get("sources")
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
            raise ValueError("sources must be an array")
        return cls(
            schema_version=_text(value.get("schema_version"), "schema_version"),
            sources=tuple(SourceSpec.from_mapping(_object(item, "source")) for item in sources),
        )


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON registry: {resolved}") from exc
    return _object(raw, "registry root")


def load_endpoint_registry(path: str | Path) -> EndpointRegistry:
    """Load and validate an endpoint registry without touching model artifacts."""

    return EndpointRegistry.from_mapping(_read_json(path))


def load_source_registry(path: str | Path) -> SourceRegistry:
    """Load and validate a raw-data source registry without network access."""

    return SourceRegistry.from_mapping(_read_json(path))
