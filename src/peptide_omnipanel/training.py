"""Training-set selection and feature helpers for V2-P1/P2."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .datasets import Observation, action_allows_use
from .features import (
    FEATURE_SCHEMA_VERSION,
    combine_feature_blocks,
    hashed_character_ngrams,
    sequence_physicochemical_features,
)
from .representations import is_sentinel


LICENSE_POLICIES = frozenset({"governed", "research_candidate"})


@dataclass(frozen=True)
class TaskTrainingSpec:
    task_id: str
    endpoint_family: str
    task_kind: str
    unit: str | None
    target_transform: str
    support_tier: str
    development_rows: int
    calibration_rows: int
    sealed_rows: int
    development_identities: int
    calibration_identities: int
    label_counts_development: tuple[tuple[int, int], ...] = ()
    label_counts_calibration: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "endpoint_family": self.endpoint_family,
            "task_kind": self.task_kind,
            "unit": self.unit,
            "target_transform": self.target_transform,
            "support_tier": self.support_tier,
            "development_rows": self.development_rows,
            "calibration_rows": self.calibration_rows,
            "sealed_rows": self.sealed_rows,
            "development_identities": self.development_identities,
            "calibration_identities": self.calibration_identities,
            "label_counts_development": dict(self.label_counts_development),
            "label_counts_calibration": dict(self.label_counts_calibration),
        }


def sequence_text_for_model(observation: Observation) -> str | None:
    """Keep explicit peptide notation, but never invent sequence from a structure."""

    if is_sentinel(observation.sequence):
        return None
    text = "".join(str(observation.sequence).split())
    return text or None


def observation_allowed(observation: Observation, license_policy: str) -> bool:
    if license_policy not in LICENSE_POLICIES:
        raise ValueError(f"unsupported license policy: {license_policy!r}")
    if not observation.research_train_eligible:
        return False
    if license_policy == "research_candidate":
        return True
    action = observation.license_actions.internal_research
    return action_allows_use(action, "internal_research")


def _binary_labels(rows: Sequence[Observation]) -> Counter[int]:
    return Counter(int(row.label) for row in rows if row.label is not None)


def select_task_training_specs(
    observations: Iterable[Observation],
    *,
    license_policy: str,
    primary_min_development: int = 30,
    primary_min_calibration: int = 10,
    fewshot_min_development: int = 5,
    fewshot_min_calibration: int = 2,
) -> tuple[list[TaskTrainingSpec], list[dict[str, Any]]]:
    """Select sequence-addressable tasks without opening the sealed split."""

    if primary_min_development < fewshot_min_development:
        raise ValueError("primary development minimum cannot be below few-shot minimum")
    if primary_min_calibration < fewshot_min_calibration:
        raise ValueError("primary calibration minimum cannot be below few-shot minimum")

    grouped: dict[str, list[Observation]] = defaultdict(list)
    excluded_counts: Counter[str] = Counter()
    for observation in observations:
        if observation.target_kind not in {"strict_regression", "binary"}:
            excluded_counts["unsupported_target_kind"] += 1
            continue
        if observation.split_id not in {"development", "calibration", "sealed"}:
            excluded_counts["missing_or_invalid_split"] += 1
            continue
        if not observation_allowed(observation, license_policy):
            excluded_counts["license_or_research_gate"] += 1
            continue
        if sequence_text_for_model(observation) is None:
            excluded_counts["sequence_missing"] += 1
            continue
        grouped[observation.task_id].append(observation)

    specs: list[TaskTrainingSpec] = []
    inventory: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        rows = grouped[task_id]
        development = [row for row in rows if row.split_id == "development"]
        calibration = [row for row in rows if row.split_id == "calibration"]
        sealed = [row for row in rows if row.split_id == "sealed"]
        first = rows[0]
        incompatible = any(
            (row.endpoint_family, row.task_kind, row.unit, row.target_transform)
            != (first.endpoint_family, first.task_kind, first.unit, first.target_transform)
            for row in rows[1:]
        )
        reasons: list[str] = []
        if incompatible:
            reasons.append("task_contract_inconsistent")
        development_labels = _binary_labels(development)
        calibration_labels = _binary_labels(calibration)
        if first.task_kind == "binary_classification":
            if set(development_labels) != {0, 1}:
                reasons.append("development_missing_binary_class")
            if set(calibration_labels) != {0, 1}:
                reasons.append("calibration_missing_binary_class")

        support_tier: str | None = None
        if not reasons:
            if (
                len(development) >= primary_min_development
                and len(calibration) >= primary_min_calibration
            ):
                support_tier = "primary"
            elif (
                len(development) >= fewshot_min_development
                and len(calibration) >= fewshot_min_calibration
            ):
                support_tier = "few_shot"
            else:
                reasons.append("below_fewshot_split_minimum")

        inventory.append(
            {
                "task_id": task_id,
                "endpoint_family": first.endpoint_family,
                "task_kind": first.task_kind,
                "development_rows": len(development),
                "calibration_rows": len(calibration),
                "sealed_rows_untouched": len(sealed),
                "support_tier": support_tier or "not_trainable",
                "eligible": support_tier is not None,
                "reasons": reasons,
            }
        )
        if support_tier is None:
            continue
        specs.append(
            TaskTrainingSpec(
                task_id=task_id,
                endpoint_family=first.endpoint_family,
                task_kind=first.task_kind,
                unit=first.unit,
                target_transform=first.target_transform,
                support_tier=support_tier,
                development_rows=len(development),
                calibration_rows=len(calibration),
                sealed_rows=len(sealed),
                development_identities=len(
                    {row.identity_group_id for row in development}
                ),
                calibration_identities=len(
                    {row.identity_group_id for row in calibration}
                ),
                label_counts_development=tuple(sorted(development_labels.items())),
                label_counts_calibration=tuple(sorted(calibration_labels.items())),
            )
        )

    inventory.insert(
        0,
        {
            "task_id": "__selection_summary__",
            "endpoint_family": "all",
            "task_kind": "summary",
            "development_rows": 0,
            "calibration_rows": 0,
            "sealed_rows_untouched": 0,
            "support_tier": "summary",
            "eligible": True,
            "reasons": [
                f"excluded_{reason}={count}"
                for reason, count in sorted(excluded_counts.items())
            ],
        },
    )
    return specs, inventory


def rows_for_spec(
    observations: Iterable[Observation],
    spec: TaskTrainingSpec,
    split: str,
    *,
    license_policy: str,
) -> list[Observation]:
    if split not in {"development", "calibration"}:
        raise ValueError("training helpers never expose the sealed split")
    return sorted(
        (
            observation
            for observation in observations
            if observation.task_id == spec.task_id
            and observation.split_id == split
            and observation_allowed(observation, license_policy)
            and sequence_text_for_model(observation) is not None
        ),
        key=lambda row: row.observation_id,
    )


def feature_vector_for_sequence(
    sequence: str,
    cache: dict[str, tuple[float, ...]] | None = None,
) -> tuple[float, ...]:
    if cache is not None and sequence in cache:
        return cache[sequence]
    vector = combine_feature_blocks(
        [
            hashed_character_ngrams(
                sequence,
                modality="sequence",
                n_features=256,
                ngram_range=(1, 3),
            ),
            sequence_physicochemical_features(sequence),
        ]
    ).values
    if cache is not None:
        cache[sequence] = vector
    return vector


def feature_matrix_and_targets(
    rows: Sequence[Observation],
    *,
    cache: dict[str, tuple[float, ...]] | None = None,
) -> tuple[Any, Any, tuple[str, ...]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - training host contract.
        raise RuntimeError("NumPy is required for feature matrices") from exc
    if not rows:
        raise ValueError("cannot build a matrix from zero observations")
    vectors: list[tuple[float, ...]] = []
    targets: list[float] = []
    sequences: list[str] = []
    for row in rows:
        sequence = sequence_text_for_model(row)
        if sequence is None:
            raise ValueError(f"row {row.observation_id} has no sequence representation")
        target = row.label if row.label is not None else row.value
        if target is None:
            raise ValueError(f"row {row.observation_id} has no supervised target")
        vectors.append(feature_vector_for_sequence(sequence, cache))
        targets.append(float(target))
        sequences.append(sequence)
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
        tuple(sequences),
    )


def feature_schema_name() -> str:
    return f"{FEATURE_SCHEMA_VERSION}:sequence_char256_physchem_compact_v1"
