"""Isolated V2.8 self-trained Peptide-OmniPanel implementation.

This package deliberately remains outside ``peptide_omnipanel`` so changes to
V2.8 cannot alter historical V2.5 frozen-source fingerprints.
"""

from .v28_model_artifacts import load_model_artifact, validate_model_artifact_manifest
from .v28_registry import (
    EndpointRegistry,
    EndpointSpec,
    SourceRegistry,
    SourceSpec,
    load_endpoint_registry,
    load_source_registry,
)
from .v28_router import (
    AlwaysReturnRouter,
    EndpointCandidate,
    V28PanelResult,
    V28PredictionContext,
)

__all__ = [
    "AlwaysReturnRouter",
    "EndpointCandidate",
    "EndpointRegistry",
    "EndpointSpec",
    "SourceRegistry",
    "SourceSpec",
    "V28PanelResult",
    "V28PredictionContext",
    "load_endpoint_registry",
    "load_model_artifact",
    "load_source_registry",
    "validate_model_artifact_manifest",
]
