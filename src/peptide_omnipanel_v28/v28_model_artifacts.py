"""Fail-closed validation and loading for V2.8 project-trained model artifacts.

``joblib`` deserialization can execute code.  This module therefore treats a
trusted, release-owned SHA256 allowlist as a mandatory input and performs all
manifest, path, provenance, and digest checks before deserializing anything.
It deliberately does not support remote URLs, prediction APIs, or third-party
weights.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_TRAINED_ORIGIN = "project_trained"
JOBLIB_FORMAT = "joblib"
SUPPORTED_TASK_KINDS = frozenset(
    {
        "regression",
        "censored_regression",
        "classification",
        "binary_classification",
        "positive_unlabeled",
    }
)
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "artifact_origin",
        "format",
        "model_path",
        "sha256",
        "feature_schema",
        "seed",
        "training_input_hash",
        "endpoint_semantics",
    }
)
REQUIRED_ENDPOINT_FIELDS = frozenset(
    {
        "endpoint_id",
        "task_kind",
        "unit",
        "parameter_basis",
        "assay",
        "species",
        "matrix",
        "value_semantics",
    }
)
CLASSIFICATION_TASK_KINDS = frozenset(
    {"classification", "binary_classification", "positive_unlabeled"}
)


class ArtifactManifestError(ValueError):
    """Raised before deserialization when an artifact contract is not proven."""


@dataclass(frozen=True)
class EndpointSemantics:
    """Endpoint meaning required to prevent assay or parameter conflation."""

    endpoint_id: str
    task_kind: str
    unit: str
    parameter_basis: str
    assay: str
    species: str
    matrix: str
    value_semantics: str
    positive_class_semantics: str | None


@dataclass(frozen=True)
class ValidatedModelArtifact:
    """A local artifact whose manifest, registration, path, and SHA all passed."""

    manifest_path: Path | None
    model_path: Path
    sha256: str
    feature_schema: str | dict[str, Any]
    seed: int
    training_input_hash: str
    endpoint_semantics: EndpointSemantics


@dataclass(frozen=True)
class LoadedModelArtifact:
    """A validated artifact and the estimator extracted from its payload."""

    specification: ValidatedModelArtifact
    model: Any
    payload: Mapping[str, Any]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    digest = _required_text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactManifestError(f"{field} must be a lowercase SHA256 digest")
    return digest


def _validate_feature_schema(value: Any) -> str | dict[str, Any]:
    if isinstance(value, str):
        return _required_text(value, "feature_schema")
    if isinstance(value, Mapping) and value:
        # Copy the schema so callers cannot mutate the validated view afterwards.
        return dict(value)
    raise ArtifactManifestError("feature_schema must be a non-empty string or mapping")


def _validate_endpoint_semantics(value: Any) -> EndpointSemantics:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError("endpoint_semantics must be a mapping")
    missing = sorted(REQUIRED_ENDPOINT_FIELDS - set(value))
    if missing:
        raise ArtifactManifestError(
            "endpoint_semantics is missing required fields: " + ", ".join(missing)
        )
    task_kind = _required_text(value["task_kind"], "endpoint_semantics.task_kind")
    if task_kind not in SUPPORTED_TASK_KINDS:
        raise ArtifactManifestError(f"unsupported endpoint task_kind: {task_kind}")
    positive_class = value.get("positive_class_semantics")
    if task_kind in CLASSIFICATION_TASK_KINDS:
        positive_class = _required_text(
            positive_class,
            "endpoint_semantics.positive_class_semantics",
        )
    elif positive_class is not None:
        positive_class = _required_text(
            positive_class,
            "endpoint_semantics.positive_class_semantics",
        )
    return EndpointSemantics(
        endpoint_id=_required_text(value["endpoint_id"], "endpoint_semantics.endpoint_id"),
        task_kind=task_kind,
        unit=_required_text(value["unit"], "endpoint_semantics.unit"),
        parameter_basis=_required_text(
            value["parameter_basis"], "endpoint_semantics.parameter_basis"
        ),
        assay=_required_text(value["assay"], "endpoint_semantics.assay"),
        species=_required_text(value["species"], "endpoint_semantics.species"),
        matrix=_required_text(value["matrix"], "endpoint_semantics.matrix"),
        value_semantics=_required_text(
            value["value_semantics"], "endpoint_semantics.value_semantics"
        ),
        positive_class_semantics=positive_class,
    )


def _registered_hashes(registered_sha256s: Collection[str] | Mapping[str, str]) -> set[str]:
    raw_hashes = (
        registered_sha256s.values()
        if isinstance(registered_sha256s, Mapping)
        else registered_sha256s
    )
    hashes = {_sha256(value, "registered_sha256s entry") for value in raw_hashes}
    if not hashes:
        raise ArtifactManifestError("registered_sha256s must contain a trusted weight digest")
    return hashes


def _resolve_local_model_path(model_path: Any, manifest_dir: Path) -> Path:
    rendered = _required_text(model_path, "model_path")
    if "://" in rendered or rendered.startswith(("api:", "http:", "https:")):
        raise ArtifactManifestError("remote/API model paths are forbidden")
    relative = Path(rendered)
    if relative.is_absolute():
        raise ArtifactManifestError("model_path must be relative to the artifact manifest")
    root = manifest_dir.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactManifestError("model_path escapes the artifact manifest directory") from exc
    if not resolved.is_file():
        raise ArtifactManifestError(f"model artifact is not a regular file: {resolved}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_dir: str | Path,
    registered_sha256s: Collection[str] | Mapping[str, str],
    manifest_path: str | Path | None = None,
) -> ValidatedModelArtifact:
    """Validate one V2.8 local model manifest and verify its artifact SHA.

    ``registered_sha256s`` must come from a trusted, frozen release registry;
    passing the manifest's own digest back as the registry would defeat the
    independent-registration boundary.
    """

    if not isinstance(manifest, Mapping):
        raise ArtifactManifestError("artifact manifest must be a mapping")
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ArtifactManifestError(
            "artifact manifest is missing required fields: " + ", ".join(missing)
        )
    origin = _required_text(manifest["artifact_origin"], "artifact_origin")
    if origin != PROJECT_TRAINED_ORIGIN:
        raise ArtifactManifestError(
            "only project_trained artifacts are allowed; third-party/API weights are forbidden"
        )
    artifact_format = _required_text(manifest["format"], "format")
    if artifact_format != JOBLIB_FORMAT:
        raise ArtifactManifestError("only local joblib model artifacts are supported")
    digest = _sha256(manifest["sha256"], "sha256")
    if digest not in _registered_hashes(registered_sha256s):
        raise ArtifactManifestError("model weight SHA256 is not present in the trusted registry")
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactManifestError("seed must be a non-negative integer")
    root = Path(manifest_dir)
    artifact = _resolve_local_model_path(manifest["model_path"], root)
    observed_digest = _file_sha256(artifact)
    if observed_digest != digest:
        raise ArtifactManifestError(
            f"model artifact SHA256 mismatch: {observed_digest} != {digest}"
        )
    recorded_manifest_path = None
    if manifest_path is not None:
        recorded_manifest_path = Path(manifest_path).resolve()
    return ValidatedModelArtifact(
        manifest_path=recorded_manifest_path,
        model_path=artifact,
        sha256=digest,
        feature_schema=_validate_feature_schema(manifest["feature_schema"]),
        seed=seed,
        training_input_hash=_sha256(
            manifest["training_input_hash"], "training_input_hash"
        ),
        endpoint_semantics=_validate_endpoint_semantics(manifest["endpoint_semantics"]),
    )


def load_model_artifact(
    manifest_path: str | Path,
    *,
    registered_sha256s: Collection[str] | Mapping[str, str],
) -> LoadedModelArtifact:
    """Load a SHA-registered, project-trained joblib artifact without network access."""

    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ArtifactManifestError(f"artifact manifest is not a regular file: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"cannot read artifact manifest: {path}") from exc
    specification = validate_model_artifact_manifest(
        manifest,
        manifest_dir=path.parent,
        registered_sha256s=registered_sha256s,
        manifest_path=path,
    )

    # Read and re-hash the exact bytes passed to joblib.  This closes the gap
    # between path validation and deserialization if a file changes in place.
    artifact_bytes = specification.model_path.read_bytes()
    observed_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if observed_digest != specification.sha256:
        raise ArtifactManifestError("model artifact changed after manifest validation")
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - deployment dependency contract.
        raise RuntimeError("joblib is required to load V2.8 model artifacts") from exc
    payload = joblib.load(io.BytesIO(artifact_bytes))
    if not isinstance(payload, Mapping):
        raise ArtifactManifestError("model artifact payload must be a mapping")
    model_keys = [key for key in ("model", "estimator") if key in payload]
    if len(model_keys) != 1:
        raise ArtifactManifestError(
            "model artifact payload must contain exactly one of 'model' or 'estimator'"
        )
    return LoadedModelArtifact(
        specification=specification,
        model=payload[model_keys[0]],
        payload=dict(payload),
    )


# Short aliases for registry code that treats manifests as generic artifacts.
validate_artifact_manifest = validate_model_artifact_manifest
load_registered_model_artifact = load_model_artifact
