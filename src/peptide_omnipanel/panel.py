"""Unified inference panel for sequence-first peptide property prediction.

The panel deliberately keeps endpoint families separate from assay-specific
tasks.  A family therefore never invents an aggregate scalar: model outputs
live under ``tasks`` and unavailable families fail closed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

from .contracts import PeptideInput, ValidatedSequence
from .features import PeptideFeatureVector, build_peptide_features
from .pir import PeptideIntermediateRepresentation, pir_from_sequence

ENDPOINT_FAMILIES = (
    "permeability",
    "LogD",
    "solubility",
    "F",
    "T1/2",
    "PPB",
    "CL",
    "Vd",
    "BBB",
    "Kp",
)

ALLOWED_STATUSES = frozenset(
    {
        "predicted",
        "low_confidence",
        "insufficient_data",
        "not_applicable",
        "input_ambiguous",
        "representation_failed",
    }
)
VALUE_STATUSES = frozenset({"predicted", "low_confidence"})


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ModelEntry:
    """Manifest metadata for one assay- or condition-specific predictor."""

    endpoint_family: str
    task_id: str
    predictor_id: str
    model_version: str
    unit: str | None = None
    support_tier: str | None = None
    condition: Mapping[str, Any] = field(default_factory=dict)
    artifact_path: str | None = None
    task_kind: str | None = None
    threshold: float | None = None
    interval_half_width: float | None = None
    primary_for_family: bool = False
    development_rows: int | None = None
    calibration_rows: int | None = None
    sequence_length_min: int | None = None
    sequence_length_max: int | None = None

    def __post_init__(self) -> None:
        if self.endpoint_family not in ENDPOINT_FAMILIES:
            raise ValueError(
                f"Unknown endpoint family {self.endpoint_family!r}; "
                f"expected one of {', '.join(ENDPOINT_FAMILIES)}"
            )
        for name in ("task_id", "predictor_id", "model_version"):
            object.__setattr__(self, name, _non_empty_text(getattr(self, name), name))
        if self.unit is not None and not isinstance(self.unit, str):
            raise ValueError("unit must be a string or null")
        if self.support_tier is not None and not isinstance(self.support_tier, str):
            raise ValueError("support_tier must be a string or null")
        if not isinstance(self.condition, Mapping):
            raise ValueError("condition must be a mapping")
        object.__setattr__(self, "condition", dict(self.condition))
        if self.artifact_path is not None and not isinstance(self.artifact_path, str):
            raise ValueError("artifact_path must be a string or null")
        if self.task_kind not in {None, "regression", "binary_classification"}:
            raise ValueError("task_kind must be regression, binary_classification or null")
        for name in ("threshold", "interval_half_width"):
            number = getattr(self, name)
            if number is not None and (not isinstance(number, Real) or not math.isfinite(number)):
                raise ValueError(f"{name} must be a finite number or null")
        if self.threshold is not None and not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be between zero and one")
        if self.interval_half_width is not None and float(self.interval_half_width) < 0.0:
            raise ValueError("interval_half_width must be non-negative")
        for name in (
            "development_rows",
            "calibration_rows",
            "sequence_length_min",
            "sequence_length_max",
        ):
            count = getattr(self, name)
            if count is not None and (not isinstance(count, int) or count < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if (
            self.sequence_length_min is not None
            and self.sequence_length_max is not None
            and self.sequence_length_min > self.sequence_length_max
        ):
            raise ValueError("sequence length bounds are reversed")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelEntry:
        if not isinstance(value, Mapping):
            raise ValueError("Each model manifest entry must be an object")
        task_id = value.get("task_id")
        return cls(
            endpoint_family=value.get("endpoint_family"),
            task_id=task_id,
            predictor_id=value.get("predictor_id", task_id),
            model_version=value.get("model_version", "unversioned"),
            unit=value.get("unit"),
            support_tier=value.get("support_tier"),
            condition=value.get("condition", {}),
            artifact_path=value.get("artifact_path"),
            task_kind=value.get("task_kind"),
            threshold=value.get("threshold"),
            interval_half_width=value.get("interval_half_width"),
            primary_for_family=bool(value.get("primary_for_family", False)),
            development_rows=value.get("development_rows"),
            calibration_rows=value.get("calibration_rows"),
            sequence_length_min=value.get("sequence_length_min"),
            sequence_length_max=value.get("sequence_length_max"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_family": self.endpoint_family,
            "task_id": self.task_id,
            "predictor_id": self.predictor_id,
            "model_version": self.model_version,
            "unit": self.unit,
            "support_tier": self.support_tier,
            "condition": dict(self.condition),
            "artifact_path": self.artifact_path,
            "task_kind": self.task_kind,
            "threshold": self.threshold,
            "interval_half_width": self.interval_half_width,
            "primary_for_family": self.primary_for_family,
            "development_rows": self.development_rows,
            "calibration_rows": self.calibration_rows,
            "sequence_length_min": self.sequence_length_min,
            "sequence_length_max": self.sequence_length_max,
        }


PredictorLoader = Callable[[ModelEntry], Any]


@dataclass(frozen=True)
class ModelBundle:
    """A safe manifest plus explicitly injected in-memory predictors.

    The manifest is data only.  It never imports code or deserializes a model
    path, so loading executable predictors remains an explicit caller action.
    """

    version: str = "peptide-omnipanel-unversioned"
    entries: tuple[ModelEntry, ...] = ()
    predictors: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_empty_text(self.version, "version"))
        entries = tuple(self.entries)
        task_ids = [entry.task_id for entry in entries]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Model manifest task_id values must be unique")
        if not isinstance(self.predictors, Mapping):
            raise ValueError("predictors must be a mapping")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "predictors", dict(self.predictors))

    @classmethod
    def empty(cls, version: str = "peptide-omnipanel-unversioned") -> ModelBundle:
        return cls(version=version)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any] | str | Path,
        *,
        predictors: Mapping[str, Any] | None = None,
        loader: PredictorLoader | None = None,
    ) -> ModelBundle:
        """Build a bundle from JSON-compatible metadata and explicit predictors."""

        if isinstance(manifest, (str, Path)):
            path = Path(manifest)
            parsed = json.loads(path.read_text(encoding="utf-8"))
        else:
            parsed = manifest
        if not isinstance(parsed, Mapping):
            raise ValueError("Model manifest must contain a JSON object")

        raw_entries = parsed.get("models", parsed.get("entries", ()))
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            raise ValueError("Model manifest 'models' must be an array")
        entries = tuple(ModelEntry.from_mapping(item) for item in raw_entries)
        version = parsed.get(
            "bundle_version",
            parsed.get("model_version", parsed.get("version", "unversioned")),
        )

        supplied = {} if predictors is None else dict(predictors)
        resolved: dict[str, Any] = {}
        for entry in entries:
            predictor = supplied.get(entry.predictor_id)
            if predictor is None:
                predictor = supplied.get(entry.task_id)
            if predictor is None and loader is not None:
                predictor = loader(entry)
            if predictor is not None:
                resolved[entry.predictor_id] = predictor
        return cls(version=version, entries=entries, predictors=resolved)

    @property
    def loaded_model_count(self) -> int:
        return sum(
            1 for entry in self.entries if self.predictor_for(entry) is not None
        )

    def entries_for(self, endpoint_family: str) -> tuple[ModelEntry, ...]:
        if endpoint_family not in ENDPOINT_FAMILIES:
            raise ValueError(f"Unknown endpoint family: {endpoint_family!r}")
        return tuple(
            entry for entry in self.entries if entry.endpoint_family == endpoint_family
        )

    def predictor_for(self, entry: ModelEntry) -> Any | None:
        return self.predictors.get(entry.predictor_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.version,
            "loaded_model_count": self.loaded_model_count,
            "models": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class PredictionContext:
    """Validated representations passed to one explicitly registered predictor."""

    validated: ValidatedSequence
    pir: PeptideIntermediateRepresentation
    features: PeptideFeatureVector
    model_entry: ModelEntry

    @property
    def sequence(self) -> str:
        return self.validated.sequence


@dataclass(frozen=True)
class PanelResult:
    """One normalized input and its fixed all-endpoint response panel."""

    validated: ValidatedSequence
    pir: PeptideIntermediateRepresentation
    features: PeptideFeatureVector
    endpoints: Mapping[str, Mapping[str, Any]]
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        input_payload = self.validated.to_dict()
        input_payload["normalized_notation"] = self.validated.sequence
        return {
            "input": input_payload,
            "endpoints": {
                family: dict(self.endpoints[family]) for family in ENDPOINT_FAMILIES
            },
            "model_version": self.model_version,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )


def _finite_number(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _prediction_interval(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("prediction_interval must contain [lower, upper]")
    if len(value) != 2:
        raise ValueError("prediction_interval must contain exactly two values")
    lower = _finite_number(value[0], "prediction_interval lower bound")
    upper = _finite_number(value[1], "prediction_interval upper bound")
    if lower is None or upper is None or lower > upper:
        raise ValueError("prediction_interval must satisfy finite lower <= upper")
    return [lower, upper]


def _task_stub(entry: ModelEntry, reason: str) -> dict[str, Any]:
    return {
        "task_id": entry.task_id,
        "predictor_id": entry.predictor_id,
        "value": None,
        "probability": None,
        "unit": entry.unit,
        "condition": dict(entry.condition),
        "prediction_interval": None,
        "status": "insufficient_data",
        "support_tier": entry.support_tier,
        "applicability": None,
        "model_version": entry.model_version,
        "reason": reason,
    }


def _normalize_task_output(
    entry: ModelEntry, raw_output: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        if not isinstance(raw_output, Mapping):
            raise ValueError("predictor output must be an object")
        value = _finite_number(raw_output.get("value"), "value")
        probability = _finite_number(raw_output.get("probability"), "probability")
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        interval = _prediction_interval(raw_output.get("prediction_interval"))
        inferred_status = (
            "predicted" if value is not None or probability is not None else "insufficient_data"
        )
        status = raw_output.get("status", inferred_status)
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported status {status!r}")
        if status in VALUE_STATUSES and value is None and probability is None:
            raise ValueError(f"status {status!r} requires value or probability")
        if status not in VALUE_STATUSES:
            value = None
            probability = None
            interval = None
        return {
            "task_id": entry.task_id,
            "predictor_id": entry.predictor_id,
            "value": value,
            "probability": probability,
            "unit": entry.unit,
            "condition": dict(entry.condition),
            "prediction_interval": interval,
            "status": status,
            "support_tier": entry.support_tier,
            "applicability": raw_output.get("applicability"),
            "model_version": entry.model_version,
            "reason": raw_output.get("reason"),
        }
    except (TypeError, ValueError) as exc:
        return _task_stub(entry, f"invalid_model_output:{exc}")


def _family_status(tasks: Sequence[Mapping[str, Any]]) -> str:
    statuses = [task["status"] for task in tasks]
    if "predicted" in statuses:
        return "predicted"
    if "low_confidence" in statuses:
        return "low_confidence"
    for status in (
        "input_ambiguous",
        "representation_failed",
        "not_applicable",
    ):
        if statuses and all(item == status for item in statuses):
            return status
    return "insufficient_data"


class PanelPredictor:
    """Run the validated sequence -> PIR -> features -> endpoint panel chain."""

    def __init__(self, model_bundle: ModelBundle | None = None) -> None:
        self.model_bundle = model_bundle or ModelBundle.empty()

    def predict(
        self,
        value: str | PeptideInput | ValidatedSequence,
        input_format: str = "auto",
    ) -> PanelResult:
        if isinstance(value, ValidatedSequence):
            validated = value
        elif isinstance(value, PeptideInput):
            validated = value.validate()
        else:
            validated = PeptideInput(sequence=value, format=input_format).validate()

        pir = pir_from_sequence(validated)
        features = build_peptide_features(sequence=validated.sequence)
        endpoints: dict[str, dict[str, Any]] = {}
        for family in ENDPOINT_FAMILIES:
            entries = self.model_bundle.entries_for(family)
            if not entries:
                endpoints[family] = {
                    "status": "insufficient_data",
                    "value": None,
                    "probability": None,
                    "unit": None,
                    "prediction_interval": None,
                    "primary_task_id": None,
                    "reason": "no_model_loaded",
                    "tasks": [],
                }
                continue

            tasks: list[dict[str, Any]] = []
            for entry in entries:
                predictor = self.model_bundle.predictor_for(entry)
                if predictor is None:
                    tasks.append(_task_stub(entry, "no_predictor_loaded"))
                    continue
                context = PredictionContext(
                    validated=validated,
                    pir=pir,
                    features=features,
                    model_entry=entry,
                )
                try:
                    if callable(predictor):
                        raw_output = predictor(context)
                    elif callable(getattr(predictor, "predict", None)):
                        raw_output = predictor.predict(context)
                    else:
                        raise TypeError("registered predictor is not callable")
                except Exception as exc:  # Predictor failures must not leak pseudo-values.
                    tasks.append(
                        _task_stub(
                            entry,
                            f"prediction_failed:{type(exc).__name__}",
                        )
                    )
                    continue
                tasks.append(_normalize_task_output(entry, raw_output))

            status = _family_status(tasks)
            primary_task = next(
                (
                    task
                    for entry, task in zip(entries, tasks, strict=True)
                    if entry.primary_for_family and task["status"] in VALUE_STATUSES
                ),
                None,
            )
            endpoints[family] = {
                "status": status,
                "value": None if primary_task is None else primary_task["value"],
                "probability": (
                    None if primary_task is None else primary_task["probability"]
                ),
                "unit": None if primary_task is None else primary_task["unit"],
                "prediction_interval": (
                    None if primary_task is None else primary_task["prediction_interval"]
                ),
                "primary_task_id": (
                    None if primary_task is None else primary_task["task_id"]
                ),
                "reason": None if status in VALUE_STATUSES else "no_valid_prediction",
                "tasks": tasks,
            }
        return PanelResult(
            validated=validated,
            pir=pir,
            features=features,
            endpoints=endpoints,
            model_version=self.model_bundle.version,
        )

    __call__ = predict
