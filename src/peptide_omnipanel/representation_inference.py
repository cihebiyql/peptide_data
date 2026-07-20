"""Fail-closed, representation-aware inference for V2.6 frozen direct bundles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .contracts import PeptideInput
from .optimization import representation_feature_vector
from .pir import pir_from_helm
from .representations import audit_smiles_representation, is_sentinel

SUPPORTED_REPRESENTATIONS = frozenset({"sequence", "helm", "structure"})


@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    endpoint_family: str
    task_kind: str
    representation: str
    feature_schema: str
    candidate_id: str
    candidate_name: str
    framework: str
    artifact_path: str
    artifact_sha256: str
    development_rows: int
    development_components: int


@dataclass(frozen=True)
class RepresentationInputs:
    sequence: str | None
    sequence_format: str
    helm: str | None
    structure: str | None
    ambiguity: Mapping[str, str]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RepresentationInputs:
        sequence = None if is_sentinel(payload.get("sequence")) else str(payload["sequence"])
        helm = None if is_sentinel(payload.get("helm")) else str(payload["helm"])
        smiles = None if is_sentinel(payload.get("smiles")) else str(payload["smiles"])
        structure = None if is_sentinel(payload.get("structure")) else str(payload["structure"])
        ambiguity: dict[str, str] = {}
        if smiles is not None and structure is not None and smiles.strip() != structure.strip():
            ambiguity["structure"] = "conflicting_structure_and_smiles_fields"
        selected_structure = structure if structure is not None else smiles
        return cls(
            sequence=sequence,
            sequence_format=str(payload.get("sequence_format", "auto")),
            helm=helm,
            structure=selected_structure,
            ambiguity=ambiguity,
        )

    def normalized_text(self, representation: str) -> str | None:
        if representation == "sequence":
            if self.sequence is None:
                return None
            return (
                PeptideInput(sequence=self.sequence, format=self.sequence_format)
                .validate()
                .sequence
            )
        if representation == "helm":
            if self.helm is None:
                return None
            text = self.helm.strip()
            pir_from_helm(text)
            return text
        if representation == "structure":
            if self.structure is None:
                return None
            text = self.structure.strip()
            audit = audit_smiles_representation(text)
            if audit["status"] != "passed":
                raise ValueError(f"SMILES audit failed: {audit['status']}")
            return text
        raise ValueError(f"unsupported representation: {representation}")


class FrozenDirectBundle:
    """Loaded frozen estimators keyed by their exact V2.1 representation lane."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        tasks: tuple[FrozenTask, ...],
        estimators: Mapping[str, Any],
    ) -> None:
        self.manifest = dict(manifest)
        self.tasks = tasks
        self.estimators = dict(estimators)

    @classmethod
    def load(cls, manifest_path: str | Path) -> FrozenDirectBundle:
        import json

        path = Path(manifest_path).resolve()
        root = path.parent
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "v26-frozen-direct-bundle-1"
            or manifest.get("status") != "pass"
            or manifest.get("training_data") != "development_only"
            or manifest.get("calibration_loaded") is not False
            or manifest.get("sealed_labels_loaded") is not False
        ):
            raise RuntimeError("bundle manifest does not prove a passed development-only fit")
        raw_tasks = manifest.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) != int(manifest.get("task_count", -1)):
            raise RuntimeError("bundle manifest task count is invalid")
        tasks: list[FrozenTask] = []
        estimators: dict[str, Any] = {}
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                raise RuntimeError("bundle task entries must be mappings")
            task = FrozenTask(
                task_id=str(raw["task_id"]),
                endpoint_family=str(raw["endpoint_family"]),
                task_kind=str(raw["task_kind"]),
                representation=str(raw["representation"]),
                feature_schema=str(raw["feature_schema"]),
                candidate_id=str(raw["candidate_id"]),
                candidate_name=str(raw["candidate_name"]),
                framework=str(raw["framework"]),
                artifact_path=str(raw["artifact_path"]),
                artifact_sha256=str(raw["artifact_sha256"]),
                development_rows=int(raw["development_rows"]),
                development_components=int(raw["development_components"]),
            )
            if task.representation not in SUPPORTED_REPRESENTATIONS:
                raise RuntimeError(f"bundle task has unsupported representation: {task.task_id}")
            if task.task_kind not in {"regression", "binary_classification"}:
                raise RuntimeError(f"bundle task has unsupported kind: {task.task_id}")
            if task.task_id in estimators:
                raise RuntimeError(f"duplicate bundle task ID: {task.task_id}")
            artifact = (root / task.artifact_path).resolve()
            try:
                artifact.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"model artifact escapes bundle root: {artifact}") from exc
            observed_sha = sha256_file(artifact)
            if observed_sha != task.artifact_sha256:
                raise RuntimeError(
                    f"artifact SHA256 mismatch for {task.task_id}: "
                    f"{observed_sha} != {task.artifact_sha256}"
                )
            try:
                import joblib
            except ImportError as exc:  # pragma: no cover - deployment contract.
                raise RuntimeError("joblib is required for frozen-direct inference") from exc
            payload = joblib.load(artifact)
            if not isinstance(payload, dict) or "model" not in payload:
                raise RuntimeError(f"invalid model artifact payload: {artifact}")
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                for field in ("task_id", "representation", "candidate_id"):
                    if str(metadata.get(field)) != str(getattr(task, field)):
                        raise RuntimeError(
                            f"artifact metadata mismatch for {task.task_id}: {field}"
                        )
            tasks.append(task)
            estimators[task.task_id] = payload["model"]
        return cls(manifest=manifest, tasks=tuple(tasks), estimators=estimators)

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        inputs = RepresentationInputs.from_mapping(payload)
        results: dict[str, dict[str, Any]] = {}
        feature_cache: dict[tuple[str, str], Any] = {}
        for task in self.tasks:
            base = {
                "task_id": task.task_id,
                "endpoint_family": task.endpoint_family,
                "task_kind": task.task_kind,
                "required_representation": task.representation,
                "candidate_id": task.candidate_id,
                "value": None,
                "probability": None,
            }
            ambiguity = inputs.ambiguity.get(task.representation)
            if ambiguity is not None:
                results[task.task_id] = {
                    **base,
                    "status": "input_ambiguous",
                    "reason": ambiguity,
                }
                continue
            try:
                text = inputs.normalized_text(task.representation)
            except Exception as exc:
                results[task.task_id] = {
                    **base,
                    "status": "representation_failed",
                    "reason": (
                        f"invalid_required_representation:{task.representation}:"
                        f"{type(exc).__name__}"
                    ),
                }
                continue
            if text is None:
                results[task.task_id] = {
                    **base,
                    "status": "representation_failed",
                    "reason": f"missing_required_representation:{task.representation}",
                }
                continue
            try:
                import numpy as np

                cache_key = task.representation, text
                if cache_key not in feature_cache:
                    feature_cache[cache_key] = representation_feature_vector(
                        text, task.representation
                    )
                matrix = np.asarray([feature_cache[cache_key]], dtype=np.float32)
                estimator = self.estimators[task.task_id]
                if task.task_kind == "regression":
                    value = float(estimator.predict(matrix)[0])
                    probability = None
                    scalar = value
                else:
                    probabilities = estimator.predict_proba(matrix)
                    classes = list(estimator.classes_)
                    probability = float(probabilities[0, classes.index(1)]) if 1 in classes else 0.0
                    value = None
                    scalar = probability
                if not math.isfinite(scalar):
                    raise ValueError("model prediction is not finite")
                results[task.task_id] = {
                    **base,
                    "value": value,
                    "probability": probability,
                    "status": "predicted",
                    "reason": None,
                }
            except Exception as exc:
                results[task.task_id] = {
                    **base,
                    "status": "prediction_failed",
                    "reason": f"prediction_failed:{type(exc).__name__}",
                }
        return {
            "bundle_schema_version": self.manifest["schema_version"],
            "available_representations": {
                "sequence": inputs.sequence is not None,
                "helm": inputs.helm is not None,
                "structure": inputs.structure is not None,
            },
            "tasks": results,
        }
