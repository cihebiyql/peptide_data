"""Research-only assembly of a complete local V2.8 endpoint panel.

This module is intentionally not imported by ``peptide_omnipanel.service``.
It loads only SHA-registered project-trained sequence models and supplies a
transparent local E-tier prior for every remaining registered endpoint.  The
result is suitable for internal exploration, never a promoted clinical or
release inference service.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from peptide_omnipanel.features import hashed_character_ngrams, sequence_physicochemical_features

from .v28_model_artifacts import load_model_artifact
from .v28_registry import EndpointSpec, load_endpoint_registry
from .v28_router import AlwaysReturnRouter, EndpointCandidate, V28PredictionContext


class ResearchPanelError(ValueError):
    """Raised when a research bundle fails its local provenance contract."""


def sequence_feature_vector(sequence: str) -> np.ndarray:
    """Return the frozen, sequence-only feature schema used by research heads."""

    physical = sequence_physicochemical_features(sequence).values
    ngrams = hashed_character_ngrams(
        sequence, modality="sequence", n_features=64, ngram_range=(1, 3)
    ).values
    return np.asarray([*physical, *ngrams], dtype=float)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


def _prior_value(endpoint: EndpointSpec, sequence: str) -> Any:
    """Small, explicit E-tier priors; they are not fitted endpoint models."""

    length = len(sequence)
    charge = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
    hydrophobic = sum(residue in "AVILMFWY" for residue in sequence) / length
    cationic = _sigmoid((charge / length) * 4.0)
    probability_endpoints = {
        "solubility",
        "F",
        "BBB",
        "HIA",
        "Pgp_inhibition",
        "CYP1A2_inhibition",
        "CYP2C19_inhibition",
        "CYP2C9_inhibition",
        "CYP2D6_inhibition",
        "CYP3A4_inhibition",
        "CYP2C9_substrate",
        "CYP2D6_substrate",
        "CYP3A4_substrate",
        "hERG",
        "AMES",
        "DILI",
        "ClinTox",
        "carcinogenicity",
        "skin_reaction",
        "overall_peptide_toxicity",
        "cytotoxicity",
        "hemolysis",
        "immunogenicity_risk",
    }
    if endpoint.endpoint_id in probability_endpoints:
        return min(max(0.5 + 0.12 * (hydrophobic - 0.5) + 0.08 * (cationic - 0.5), 0.01), 0.99)
    if endpoint.endpoint_id == "Kp":
        base = max(0.05, 0.6 + 0.4 * hydrophobic - 0.1 * abs(charge) / length)
        return {
            "brain": base * 0.7,
            "liver": base,
            "kidney": base * 1.1,
            "muscle": base * 0.8,
            "adipose": base * 0.9,
        }
    if endpoint.endpoint_id == "degradation_site_probability":
        susceptible = {"K": 0.7, "R": 0.7, "F": 0.58, "W": 0.58, "Y": 0.58, "L": 0.52, "P": 0.2}
        return [susceptible.get(residue, 0.4) for residue in sequence]
    defaults = {
        "LogD7.4": 0.0,
        "T1/2": 1.0,
        "PPB": 0.5,
        "CL": 10.0,
        "Vd": 0.5,
        "permeability": -6.0,
        "LD50": -3.0,
        "HC50": -5.0,
        "cell_penetration": cationic,
        "membrane_retention": min(max(hydrophobic, 0.0), 1.0),
        "human_plasma_stability": 1.0,
        "mouse_plasma_stability": 1.0,
        "intestinal_stability": 1.0,
        "protease_stability": 1.0,
    }
    return defaults.get(endpoint.endpoint_id, 0.5)


def _prior_candidate(endpoint: EndpointSpec) -> EndpointCandidate:
    def predictor(context: V28PredictionContext) -> Mapping[str, Any]:
        value = _prior_value(endpoint, context.sequence)
        probability = (
            value
            if endpoint.endpoint_id
            in {
                "solubility",
                "F",
                "BBB",
                "HIA",
                "Pgp_inhibition",
                "CYP1A2_inhibition",
                "CYP2C19_inhibition",
                "CYP2C9_inhibition",
                "CYP2D6_inhibition",
                "CYP3A4_inhibition",
                "CYP2C9_substrate",
                "CYP2D6_substrate",
                "CYP3A4_substrate",
                "hERG",
                "AMES",
                "DILI",
                "ClinTox",
                "carcinogenicity",
                "skin_reaction",
                "overall_peptide_toxicity",
                "cytotoxicity",
                "hemolysis",
                "immunogenicity_risk",
            }
            else None
        )
        return {
            "value": value,
            "probability": probability,
            "unit": endpoint.unit,
            "research_only": True,
            "low_confidence": True,
            "confidence": "low",
            "evidence_tier": "L4_mechanism_computed",
            "applicability_domain": "not_applicable_local_prior",
            "coverage_guaranteed": True,
            "warnings": ["no_endpoint_specific_sequence_model", "not_peptide_validated"],
        }

    return EndpointCandidate(
        endpoint_id=endpoint.endpoint_id,
        tier="E",
        model_id=f"v28_e_prior_{endpoint.endpoint_id}",
        model_source="project_local_transparent_prior",
        predictor=predictor,
        input_representation="sequence",
        representation_assumptions=("natural_L_linear_free_termini",),
        peptide_validated=False,
        default_warnings=("research_only", "low_confidence"),
    )


def _as_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchPanelError(f"{field} must be an object")
    return dict(value)


def _bundle(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResearchPanelError(f"cannot read research bundle: {path}") from exc
    bundle = _as_mapping(value, "research bundle")
    if bundle.get("scope") != "internal_research_only":
        raise ResearchPanelError("research bundle must be internal_research_only")
    if not isinstance(bundle.get("registered_sha256"), Mapping):
        raise ResearchPanelError("research bundle requires registered_sha256")
    if not isinstance(bundle.get("models"), Sequence):
        raise ResearchPanelError("research bundle requires a models array")
    return bundle


def _model_candidate(
    model_spec: Mapping[str, Any],
    *,
    bundle_path: Path,
    registered_sha256: Mapping[str, str],
) -> EndpointCandidate:
    endpoint_id = model_spec.get("endpoint_id")
    manifest_name = model_spec.get("artifact_manifest")
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise ResearchPanelError("model endpoint_id must be a non-empty string")
    if not isinstance(manifest_name, str) or not manifest_name:
        raise ResearchPanelError("model artifact_manifest must be a non-empty string")
    artifact_path = (bundle_path.parent / manifest_name).resolve()
    try:
        artifact_path.relative_to(bundle_path.parent.resolve())
    except ValueError as exc:
        raise ResearchPanelError("model artifact manifest escapes bundle directory") from exc
    loaded = load_model_artifact(artifact_path, registered_sha256s=registered_sha256)
    payload = _as_mapping(loaded.payload.get("research_model"), "artifact research_model")
    output_kind = payload.get("output_kind")
    inverse_transform = payload.get("inverse_transform", "identity")
    endpoint_semantics = loaded.specification.endpoint_semantics
    if endpoint_semantics.endpoint_id != endpoint_id:
        raise ResearchPanelError("bundle endpoint_id differs from artifact endpoint semantics")
    if output_kind not in {
        "classification",
        "probability_regression",
        "regression",
        "positive_unlabeled",
    }:
        raise ResearchPanelError("unsupported research model output_kind")
    center = np.asarray(payload.get("feature_center", []), dtype=float)
    radius = float(payload.get("feature_radius", 0.0))
    if center.ndim != 1 or not len(center) or not math.isfinite(radius) or radius <= 0:
        raise ResearchPanelError(
            "artifact requires a finite feature_center and positive feature_radius"
        )

    def predictor(context: V28PredictionContext) -> Mapping[str, Any]:
        feature = sequence_feature_vector(context.sequence).reshape(1, -1)
        if feature.shape[1] != center.shape[0]:
            raise RuntimeError("sequence feature schema differs from research artifact")
        distance = float(np.linalg.norm(feature[0] - center) / radius)
        if output_kind == "classification":
            classes = list(getattr(loaded.model, "classes_", ()))
            if 1 not in classes:
                raise RuntimeError("classification artifact has no positive class")
            probability = float(loaded.model.predict_proba(feature)[0][classes.index(1)])
            value = probability
        elif output_kind == "probability_regression":
            probability = min(max(float(loaded.model.predict(feature)[0]), 0.0), 1.0)
            value = probability
        elif output_kind == "positive_unlabeled":
            value = _sigmoid(float(loaded.model.decision_function(feature)[0]))
            probability = None
        else:
            value = float(loaded.model.predict(feature)[0])
            if inverse_transform == "pow10":
                value = 10.0**value
            probability = None
        domain = "in_domain" if distance <= 1.0 else "out_of_domain"
        return {
            "value": value,
            "probability": probability,
            "unit": endpoint_semantics.unit,
            "research_only": True,
            "low_confidence": True,
            "confidence": "low",
            "evidence_tier": payload.get("evidence_tier", "L5_self_model_pseudo_label"),
            "applicability_domain": domain,
            "nearest_training_similarity": max(0.0, 1.0 - distance),
            "coverage_guaranteed": False,
            "warnings": payload.get("warnings", []),
        }

    return EndpointCandidate(
        endpoint_id=endpoint_id,
        tier="C",
        model_id=str(model_spec.get("model_id", f"v28_c_{endpoint_id}")),
        model_source="project_trained_local_research_model",
        predictor=predictor,
        input_representation="sequence",
        representation_assumptions=("natural_L_linear_free_termini",),
        peptide_validated=bool(payload.get("peptide_validated", False)),
        default_warnings=("research_only", "low_confidence"),
    )


def build_research_only_router(
    *,
    endpoint_registry_path: str | Path,
    research_bundle_path: str | Path,
) -> AlwaysReturnRouter:
    """Build the isolated research panel without modifying the active service."""

    registry = load_endpoint_registry(endpoint_registry_path)
    bundle_path = Path(research_bundle_path).resolve()
    bundle = _bundle(bundle_path)
    registered_sha256 = _as_mapping(bundle["registered_sha256"], "registered_sha256")
    candidates: list[EndpointCandidate] = [
        _prior_candidate(endpoint) for endpoint in registry.endpoints
    ]
    known_endpoints = {endpoint.endpoint_id for endpoint in registry.endpoints}
    seen: set[str] = set()
    for item in bundle["models"]:
        spec = _as_mapping(item, "research model")
        endpoint_id = str(spec.get("endpoint_id", ""))
        if endpoint_id not in known_endpoints:
            raise ResearchPanelError(f"research model references unknown endpoint: {endpoint_id}")
        if endpoint_id in seen:
            raise ResearchPanelError(f"research bundle has duplicate endpoint model: {endpoint_id}")
        candidates.append(
            _model_candidate(spec, bundle_path=bundle_path, registered_sha256=registered_sha256)
        )
        seen.add(endpoint_id)
    return AlwaysReturnRouter(registry=registry, candidates=candidates)
