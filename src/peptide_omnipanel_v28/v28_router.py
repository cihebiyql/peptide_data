"""Always-return routing for V2.8 self-trained endpoint experts.

This module is deliberately separate from the V2.6 fail-closed panel.  It
does not change the active service.  Instead, it turns a registry plus only
locally controlled candidate functions into a transparent tiered response:
``A -> B -> C -> D -> E``.  The final E-tier prior is mandatory for each
registered endpoint, so an otherwise valid sequence never becomes an
unexplained ``null`` prediction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from peptide_omnipanel.contracts import PeptideInput, ValidatedSequence

from .v28_registry import EndpointRegistry, EndpointSpec

CandidatePredictor = Callable[["V28PredictionContext"], Mapping[str, Any]]


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def _interval(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("prediction_interval must be [lower, upper]")
    if len(value) != 2:
        raise ValueError("prediction_interval must contain exactly two values")
    lower, upper = _finite(value[0], "interval lower"), _finite(value[1], "interval upper")
    if lower > upper:
        raise ValueError("prediction_interval must satisfy lower <= upper")
    return [lower, upper]


def _flag(value: Any, field: str, default: bool) -> bool:
    """Read an explicit boolean response flag without truthiness coercion."""

    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _optional_text(value: Any, field: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_value(endpoint: EndpointSpec, raw: Mapping[str, Any], sequence: str) -> Any:
    """Validate scalar and structured endpoint values without conflating them.

    Most ADMET endpoints return a scalar.  Two V2.8 endpoints are explicitly
    structured: ``Kp`` is a tissue-to-plasma vector and degradation is a
    probability for every residue.  The original router only accepted floats,
    which would either force a lossy average or make the all-endpoint contract
    impossible.  Keep their native shapes and validate them here instead.
    """

    value = raw.get("value")
    if endpoint.task_kind == "mechanistic_tissue_vector":
        if not isinstance(value, Mapping) or not value:
            raise ValueError("mechanistic_tissue_vector value must be a non-empty mapping")
        normalized: dict[str, float] = {}
        for tissue, measurement in value.items():
            if not isinstance(tissue, str) or not tissue.strip():
                raise ValueError("tissue vector keys must be non-empty strings")
            normalized[tissue.strip()] = _finite(measurement, f"Kp[{tissue}]")
        return normalized
    if endpoint.task_kind == "sequence_labeling":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("sequence_labeling value must be a probability array")
        if len(value) != len(sequence):
            raise ValueError("sequence_labeling value length must match input sequence length")
        normalized = [_finite(item, "per-residue probability") for item in value]
        if any(item < 0.0 or item > 1.0 for item in normalized):
            raise ValueError("per-residue probabilities must be between 0 and 1")
        return normalized
    return _finite(value, "value")


@dataclass(frozen=True)
class V28PredictionContext:
    """Input passed to a self-owned endpoint predictor."""

    validated: ValidatedSequence
    endpoint: EndpointSpec

    @property
    def sequence(self) -> str:
        return self.validated.sequence


@dataclass(frozen=True)
class EndpointCandidate:
    """A registered local model, mechanism, neighbor, or project prior."""

    endpoint_id: str
    tier: str
    model_id: str
    model_source: str
    predictor: CandidatePredictor
    input_representation: str = "sequence"
    representation_assumptions: tuple[str, ...] = ()
    peptide_validated: bool = False
    locally_owned: bool = True
    uses_network: bool = False
    default_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id must be non-empty")
        if self.tier not in {"A", "B", "C", "D", "E"}:
            raise ValueError("tier must be A/B/C/D/E")
        if not self.model_id.strip() or not self.model_source.strip():
            raise ValueError("model_id and model_source must be non-empty")
        if not callable(self.predictor):
            raise ValueError("predictor must be callable")
        if not self.locally_owned:
            raise ValueError("V2.8 candidates must be locally owned")
        if self.uses_network:
            raise ValueError("V2.8 candidates may not invoke a network service")
        if not self.input_representation.strip():
            raise ValueError("input_representation must be non-empty")


@dataclass(frozen=True)
class V28PanelResult:
    """A full, non-null-for-valid-input V2.8 endpoint response."""

    validated: ValidatedSequence
    endpoints: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.validated.to_dict(),
            "endpoints": {key: dict(value) for key, value in self.endpoints.items()},
        }


@dataclass
class AlwaysReturnRouter:
    """Select the highest allowed local candidate that yields a valid number."""

    registry: EndpointRegistry
    candidates: Sequence[EndpointCandidate]
    _by_endpoint: dict[str, tuple[EndpointCandidate, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        grouped: dict[str, list[EndpointCandidate]] = {
            endpoint.endpoint_id: [] for endpoint in self.registry.endpoints
        }
        for candidate in self.candidates:
            if candidate.endpoint_id not in grouped:
                raise ValueError(f"candidate references unknown endpoint: {candidate.endpoint_id}")
            endpoint = self.registry.endpoint(candidate.endpoint_id)
            if candidate.tier not in endpoint.allowed_tiers:
                raise ValueError(
                    f"candidate tier {candidate.tier} is not allowed for {candidate.endpoint_id}"
                )
            grouped[candidate.endpoint_id].append(candidate)
        for endpoint in self.registry.endpoints:
            endpoint_candidates = grouped[endpoint.endpoint_id]
            if not any(candidate.tier == "E" for candidate in endpoint_candidates):
                raise ValueError(
                    f"endpoint {endpoint.endpoint_id} requires a locally owned E-tier prior"
                )
        self._by_endpoint = {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _ordered_candidates(
        endpoint: EndpointSpec, candidates: Sequence[EndpointCandidate]
    ) -> tuple[EndpointCandidate, ...]:
        rank = {tier: index for index, tier in enumerate(endpoint.fallback_policy)}
        return tuple(sorted(candidates, key=lambda candidate: rank[candidate.tier]))

    @staticmethod
    def _normalize(
        endpoint: EndpointSpec,
        candidate: EndpointCandidate,
        raw: Mapping[str, Any],
        failures: Sequence[str],
        sequence: str,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate prediction must be an object")
        value = _normalized_value(endpoint, raw, sequence)
        probability = raw.get("probability")
        if probability is not None:
            probability = _finite(probability, "probability")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probability must be between 0 and 1")
        if endpoint.task_kind in {"mechanistic_tissue_vector", "sequence_labeling"}:
            if probability is not None:
                raise ValueError("structured endpoint predictions may not also set probability")
            if raw.get("prediction_interval") is not None:
                raise ValueError("structured endpoint predictions may not set prediction_interval")
        warnings = [str(item) for item in candidate.default_warnings]
        warnings.extend(str(item) for item in raw.get("warnings", ()))
        if failures:
            warnings.append("higher_tier_candidate_unavailable")
        research_only = _flag(
            raw.get("research_only"), "research_only", candidate.tier not in {"A", "B"}
        )
        low_confidence = _flag(
            raw.get("low_confidence"),
            "low_confidence",
            candidate.tier not in {"A", "B"} or not candidate.peptide_validated,
        )
        evidence_tier = _optional_text(
            raw.get("evidence_tier"),
            "evidence_tier",
            "L5_self_model_pseudo_label" if candidate.tier == "C" else "L4_mechanism_computed",
        )
        confidence = _optional_text(
            raw.get("confidence"), "confidence", "low" if low_confidence else "high"
        )
        status = (
            "predicted"
            if candidate.tier in {"A", "B"} and not research_only and not low_confidence
            else "predicted_low_evidence"
        )
        return {
            "endpoint_id": endpoint.endpoint_id,
            "primary_output": endpoint.primary_output,
            "value": value,
            "probability": probability,
            "unit": str(raw.get("unit", endpoint.unit)),
            "status": status,
            "prediction_tier": candidate.tier,
            "model_id": candidate.model_id,
            "model_source": candidate.model_source,
            "input_representation": candidate.input_representation,
            "representation_assumptions": list(candidate.representation_assumptions),
            "peptide_validated": candidate.peptide_validated,
            "applicability_domain": str(raw.get("applicability_domain", "unknown")),
            "nearest_training_similarity": raw.get("nearest_training_similarity"),
            "prediction_interval": (
                None
                if endpoint.task_kind in {"mechanistic_tissue_vector", "sequence_labeling"}
                else _interval(raw.get("prediction_interval"))
            ),
            "coverage_guaranteed": bool(raw.get("coverage_guaranteed", False)),
            "research_only": research_only,
            "low_confidence": low_confidence,
            "confidence": confidence,
            "evidence_tier": evidence_tier,
            "warnings": warnings,
        }

    def predict(
        self, value: str | PeptideInput | ValidatedSequence, input_format: str = "auto"
    ) -> V28PanelResult:
        if isinstance(value, ValidatedSequence):
            validated = value
        elif isinstance(value, PeptideInput):
            validated = value.validate()
        else:
            validated = PeptideInput(sequence=value, format=input_format).validate()

        results: dict[str, dict[str, Any]] = {}
        for endpoint in self.registry.endpoints:
            context = V28PredictionContext(validated=validated, endpoint=endpoint)
            failures: list[str] = []
            for candidate in self._ordered_candidates(
                endpoint, self._by_endpoint[endpoint.endpoint_id]
            ):
                try:
                    raw = candidate.predictor(context)
                    results[endpoint.endpoint_id] = self._normalize(
                        endpoint, candidate, raw, failures, context.sequence
                    )
                    break
                except Exception as exc:
                    failures.append(f"{candidate.tier}:{type(exc).__name__}")
            else:  # Defensive: construction requires E, but a broken E is visible.
                raise RuntimeError(
                    f"all self-owned candidates failed for {endpoint.endpoint_id}: {failures}"
                )
        return V28PanelResult(validated=validated, endpoints=results)

    __call__ = predict
