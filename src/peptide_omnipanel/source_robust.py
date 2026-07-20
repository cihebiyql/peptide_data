"""Fail-closed source/component-balanced row weights for V2.5 training."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SMALL_SOURCE_DOMAIN = "TRAIN_OTHER_SMALL"


@dataclass(frozen=True)
class SourceBalancedWeights:
    """Row-aligned weights, training domains, and JSON-serializable diagnostics."""

    weights: tuple[float, ...]
    domains: tuple[str, ...]
    diagnostics: dict[str, Any]


def source_component_balanced_weights(
    source_clusters: Sequence[Any],
    components: Sequence[Any],
    *,
    minimum_domain_components: int = 20,
) -> SourceBalancedWeights:
    """Build the preregistered V2.5 train-only source/component row weights.

    Source clusters with fewer than ``minimum_domain_components`` distinct
    molecular components are pooled into :data:`SMALL_SOURCE_DOMAIN`. Each
    resulting domain receives equal total weight, each component within a
    domain receives equal total weight, and repeated rows within a component
    split that component's weight equally. The returned row weights have mean
    one.

    The function fails closed on malformed identities, component/source graph
    inconsistencies, or any failed numerical invariant.
    """

    threshold = _validate_threshold(minimum_domain_components)
    sources = _normalized_identifiers(source_clusters, name="source_clusters")
    component_ids = _normalized_identifiers(components, name="components")
    if len(sources) != len(component_ids):
        raise ValueError("source_clusters and components must have the same length")
    if not sources:
        raise ValueError("source-component balancing requires at least one row")
    if SMALL_SOURCE_DOMAIN in sources:
        raise ValueError(f"source cluster identity {SMALL_SOURCE_DOMAIN!r} is reserved")

    source_components: dict[str, set[str]] = defaultdict(set)
    component_source: dict[str, str] = {}
    for source, component in zip(sources, component_ids, strict=True):
        previous = component_source.setdefault(component, source)
        if previous != source:
            raise ValueError(
                f"component {component!r} occurs in multiple source clusters: "
                f"{previous!r}, {source!r}"
            )
        source_components[source].add(component)

    small_sources = {
        source
        for source, source_component_ids in source_components.items()
        if len(source_component_ids) < threshold
    }
    domains = tuple(
        SMALL_SOURCE_DOMAIN if source in small_sources else source for source in sources
    )
    domain_components: dict[str, set[str]] = defaultdict(set)
    domain_component_rows: dict[tuple[str, str], int] = defaultdict(int)
    for domain, component in zip(domains, component_ids, strict=True):
        domain_components[domain].add(component)
        domain_component_rows[(domain, component)] += 1

    domain_count = len(domain_components)
    if domain_count == 0:
        raise RuntimeError("source-component balancing produced no training domains")

    raw_weights = tuple(
        1.0
        / (
            domain_count
            * len(domain_components[domain])
            * domain_component_rows[(domain, component)]
        )
        for domain, component in zip(domains, component_ids, strict=True)
    )
    raw_total = math.fsum(raw_weights)
    if not math.isfinite(raw_total) or raw_total <= 0.0:
        raise RuntimeError("source-component raw weight total must be finite and positive")

    row_count = len(sources)
    weights = tuple(row_count * weight / raw_total for weight in raw_weights)
    diagnostics = _weight_diagnostics(
        weights=weights,
        domains=domains,
        components=component_ids,
        source_components=source_components,
        small_sources=small_sources,
        threshold=threshold,
    )
    required = (
        "finite_positive",
        "mean_equals_one",
        "equal_domain_total",
        "equal_component_total_within_domain",
    )
    failed = [name for name in required if diagnostics[name] is not True]
    if failed:
        raise RuntimeError(
            "source-component weight invariant failure: " + ", ".join(failed)
        )
    return SourceBalancedWeights(weights=weights, domains=domains, diagnostics=diagnostics)


def _weight_diagnostics(
    *,
    weights: tuple[float, ...],
    domains: tuple[str, ...],
    components: tuple[str, ...],
    source_components: dict[str, set[str]],
    small_sources: set[str],
    threshold: int,
) -> dict[str, Any]:
    domain_values: dict[str, list[float]] = defaultdict(list)
    component_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for weight, domain, component in zip(weights, domains, components, strict=True):
        domain_values[domain].append(weight)
        component_values[domain][component].append(weight)

    domain_totals = {
        domain: math.fsum(domain_values[domain]) for domain in sorted(domain_values)
    }
    component_totals = {
        domain: {
            component: math.fsum(component_values[domain][component])
            for component in sorted(component_values[domain])
        }
        for domain in sorted(component_values)
    }
    domain_component_counts = {
        domain: len(component_totals[domain]) for domain in sorted(component_totals)
    }
    source_component_counts = {
        source: len(source_components[source]) for source in sorted(source_components)
    }

    total = math.fsum(weights)
    square_total = math.fsum(weight * weight for weight in weights)
    mean = total / len(weights)
    finite = all(math.isfinite(weight) for weight in weights)
    positive = all(weight > 0.0 for weight in weights)
    expected_domain_total = len(weights) / len(domain_totals)
    equal_domain_total = all(
        _close(total_value, expected_domain_total) for total_value in domain_totals.values()
    )
    equal_component_total = all(
        all(
            _close(total_value, expected_domain_total / len(totals))
            for total_value in totals.values()
        )
        for totals in component_totals.values()
    )
    kish = total * total / square_total if square_total > 0.0 else float("nan")

    diagnostics: dict[str, Any] = {
        "finite": finite,
        "positive": positive,
        "finite_positive": finite and positive,
        "mean": mean,
        "mean_equals_one": _close(mean, 1.0),
        "domain_totals": domain_totals,
        "equal_domain_total": equal_domain_total,
        "component_totals_within_domain": component_totals,
        "equal_component_total_within_domain": equal_component_total,
        "min": min(weights),
        "max": max(weights),
        "kish_effective_sample_size": kish,
        "row_count": len(weights),
        "source_cluster_count": len(source_components),
        "domain_count": len(domain_totals),
        "small_source_cluster_count": len(small_sources),
        "minimum_domain_components": threshold,
        "source_cluster_component_counts": source_component_counts,
        "domain_component_counts": domain_component_counts,
    }
    if not math.isfinite(kish) or kish <= 0.0 or kish > len(weights) + 1e-9:
        raise RuntimeError("Kish effective sample size is outside its valid range")
    return diagnostics


def _normalized_identifiers(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a row-aligned sequence, not a string")
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a row-aligned sequence") from exc
    normalized: list[str] = []
    for index, value in enumerate(rows):
        if not isinstance(value, str):
            raise ValueError(f"{name}[{index}] must be a string")
        identifier = value.strip()
        if not identifier:
            raise ValueError(f"{name}[{index}] must be non-empty")
        normalized.append(identifier)
    return tuple(normalized)


def _validate_threshold(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("minimum_domain_components must be a positive integer")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
