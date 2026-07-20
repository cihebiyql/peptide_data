"""Eligibility gates and machine-readable experiment records for V2 P3/P4."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import InputValidationError, validate_sequence_input
from .datasets import Observation, action_allows_use


class ExperimentValidationError(ValueError):
    """Raised when an experiment request would violate an evidence contract."""


@dataclass(frozen=True)
class Eligibility:
    stage: str
    branch: str
    status: str
    reasons: tuple[str, ...]
    counts: Mapping[str, Any]
    safeguards: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"eligible", "not_eligible"}:
            raise ExperimentValidationError(f"invalid eligibility status: {self.status!r}")
        if self.status == "not_eligible" and not self.reasons:
            raise ExperimentValidationError("not_eligible records require at least one reason")

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "branch": self.branch,
            "status": self.status,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "counts": dict(self.counts),
            "safeguards": list(self.safeguards),
            "evidence": [dict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class ExperimentResult:
    branch: str
    eligibility_status: str
    execution_status: str
    trained: bool
    successful: bool
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.eligibility_status not in {"eligible", "not_eligible"}:
            raise ExperimentValidationError("result has an invalid eligibility status")
        if self.execution_status not in {"not_run", "planned", "trained", "failed"}:
            raise ExperimentValidationError("result has an invalid execution status")
        if self.execution_status == "trained" and not self.trained:
            raise ExperimentValidationError("trained execution must set trained=true")
        if self.trained and self.execution_status != "trained":
            raise ExperimentValidationError("trained=true requires execution_status='trained'")
        if self.successful and not self.trained:
            raise ExperimentValidationError("an untrained experiment cannot be successful")
        if self.metrics and not self.trained:
            raise ExperimentValidationError("an untrained experiment cannot report metrics")
        if self.eligibility_status == "not_eligible" and self.execution_status != "not_run":
            raise ExperimentValidationError("an ineligible experiment must remain not_run")

    @classmethod
    def from_eligibility(cls, eligibility: Eligibility) -> "ExperimentResult":
        return cls(
            branch=eligibility.branch,
            eligibility_status=eligibility.status,
            execution_status="planned" if eligibility.eligible else "not_run",
            trained=False,
            successful=False,
            reasons=eligibility.reasons,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "eligibility_status": self.eligibility_status,
            "execution_status": self.execution_status,
            "trained": self.trained,
            "successful": self.successful,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True)
class P3Pair:
    record_id: str
    identity_group_id: str
    task_id: str
    split_id: str
    equivalence_key: str
    sequence_observation_ids: tuple[str, ...]
    structure_observation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "identity_group_id": self.identity_group_id,
            "task_id": self.task_id,
            "split_id": self.split_id,
            "equivalence_key": self.equivalence_key,
            "sequence_observation_ids": list(self.sequence_observation_ids),
            "structure_observation_ids": list(self.structure_observation_ids),
        }


@dataclass(frozen=True)
class PseudoLabelCandidate:
    label_id: str
    identity_group_id: str
    task_id: str
    split_id: str

    def __post_init__(self) -> None:
        if not self.label_id or not self.identity_group_id or not self.task_id or not self.split_id:
            raise ExperimentValidationError("pseudo-label candidate fields must be non-empty")


@dataclass(frozen=True)
class TemporalChallengeCandidate:
    observation_id: str
    source_id: str
    is_post_cutoff: bool
    overlaps_training: bool
    license_cleared: bool = False
    semantic_aligned: bool = False

    def __post_init__(self) -> None:
        if not self.observation_id or not self.source_id:
            raise ExperimentValidationError("temporal candidate identifiers must be non-empty")


def _allowed_internal(observation: Observation) -> bool:
    action = observation.license_actions.internal_research
    return action_allows_use(action, "internal_research")


def _has_exact_sequence(observation: Observation) -> bool:
    if (observation.representation_type or "").casefold() != "sequence":
        return False
    if not observation.sequence:
        return False
    try:
        validated = validate_sequence_input(observation.sequence, "plain")
    except InputValidationError:
        return False
    return validated.sequence == observation.sequence.strip().upper()


def evaluate_p3_fusion_eligibility(
    observations: Sequence[Observation],
    verified_equivalence: Mapping[str, str],
    *,
    min_development_pairs: int = 1,
    min_evaluation_pairs: int = 0,
) -> Eligibility:
    """Gate fusion on explicit record/identity/task/equivalence matches only."""

    if min_development_pairs < 1 or min_evaluation_pairs < 0:
        raise ExperimentValidationError("P3 pair thresholds are invalid")

    sequence_by_key: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    structure_by_key: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
    verified_count = 0
    for observation in observations:
        equivalence_key = str(verified_equivalence.get(observation.observation_id, "")).strip()
        if not equivalence_key:
            continue
        verified_count += 1
        if observation.split_id is None or not _allowed_internal(observation):
            continue
        key = (
            observation.record_id,
            observation.identity_group_id,
            observation.task_id,
            observation.split_id,
            equivalence_key,
        )
        representation = (observation.representation_type or "").casefold()
        if _has_exact_sequence(observation):
            sequence_by_key[key].append(observation.observation_id)
        if representation in {"smiles", "helm", "structure", "atom_graph"} and (
            observation.smiles or observation.helm
        ):
            structure_by_key[key].append(observation.observation_id)

    pairs: list[P3Pair] = []
    for key in sorted(set(sequence_by_key) & set(structure_by_key)):
        record_id, identity_group_id, task_id, split_id, equivalence_key = key
        pairs.append(
            P3Pair(
                record_id=record_id,
                identity_group_id=identity_group_id,
                task_id=task_id,
                split_id=split_id,
                equivalence_key=equivalence_key,
                sequence_observation_ids=tuple(sorted(sequence_by_key[key])),
                structure_observation_ids=tuple(sorted(structure_by_key[key])),
            )
        )

    by_split = Counter(pair.split_id for pair in pairs)
    evaluation_count = by_split["calibration"] + by_split["sealed"]
    reasons: list[str] = []
    if verified_count == 0:
        reasons.append("no_verified_representation_equivalence_evidence")
    if not sequence_by_key:
        reasons.append("no_verified_exact_sequence_candidates")
    if not structure_by_key:
        reasons.append("no_verified_structure_candidates")
    if not pairs:
        reasons.append("zero_record_identity_assay_equivalent_pairs")
    if by_split["development"] < min_development_pairs:
        reasons.append("insufficient_development_pairs")
    if evaluation_count < min_evaluation_pairs:
        reasons.append("insufficient_evaluation_pairs")

    counts = {
        "input_observations": len(observations),
        "verified_observations": verified_count,
        "sequence_candidate_keys": len(sequence_by_key),
        "structure_candidate_keys": len(structure_by_key),
        "equivalent_pair_keys": len(pairs),
        "development_pairs": by_split["development"],
        "calibration_pairs": by_split["calibration"],
        "sealed_pairs": by_split["sealed"],
    }
    return Eligibility(
        stage="V2-P3",
        branch="gated_multimodal_fusion",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(dict.fromkeys(reasons)),
        counts=counts,
        safeguards=(
            "requires_explicit_equivalence_key",
            "record_id_identity_group_id_task_id_split_id_must_match",
            "task_id_is_the_assay_semantic_contract",
            "no_lossy_sequence_projection_pairing",
        ),
        evidence=tuple(pair.to_dict() for pair in pairs),
    )


def evaluate_censored_branch(observations: Sequence[Observation]) -> Eligibility:
    rows = [
        observation
        for observation in observations
        if observation.target_kind in {"censored_regression", "approximate_regression"}
    ]
    eligible_rows = [
        observation
        for observation in rows
        if observation.split_id == "development" and _allowed_internal(observation)
    ]
    relation_counts = Counter(observation.relation for observation in rows)
    evidence = tuple(
        {
            "observation_id": observation.observation_id,
            "task_id": observation.task_id,
            "relation": observation.relation,
            "lower_bound": observation.lower_bound,
            "upper_bound": observation.upper_bound,
            "value": observation.value,
            "eligible_for_development": observation in eligible_rows,
        }
        for observation in rows
    )
    reasons: list[str] = []
    if not rows:
        reasons.append("no_censored_or_approximate_observations")
    if not eligible_rows:
        reasons.append("no_license_and_split_eligible_censored_development_rows")
    return Eligibility(
        stage="V2-P4",
        branch="censored",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(reasons),
        counts={
            "total_rows": len(rows),
            "eligible_development_rows": len(eligible_rows),
            "relation_counts": dict(sorted(relation_counts.items())),
            "explicit_lower_bound_rows": sum(
                row.target_kind == "censored_regression" and row.relation in {">", ">="}
                for row in rows
            ),
            "explicit_upper_bound_rows": sum(
                row.target_kind == "censored_regression" and row.relation in {"<", "<="}
                for row in rows
            ),
            "lower_value_present_rows": sum(row.lower_bound is not None for row in rows),
            "upper_value_present_rows": sum(row.upper_bound is not None for row in rows),
            "approximate_rows": sum(row.target_kind == "approximate_regression" for row in rows),
        },
        safeguards=(
            "relation_and_bounds_are_preserved",
            "approximate_rows_do_not_enter_bounded_adapter",
            "calibration_and_sealed_rows_are_evaluation_only",
        ),
        evidence=evidence,
    )


def evaluate_pu_branch(
    observations: Sequence[Observation],
    *,
    unlabeled_count: int,
    class_prior: float | None,
    treat_unlabeled_as_negative: bool = False,
) -> Eligibility:
    if treat_unlabeled_as_negative:
        raise ExperimentValidationError("PU unlabeled examples must never be treated as negatives")
    if unlabeled_count < 0:
        raise ExperimentValidationError("unlabeled_count must be non-negative")
    if class_prior is not None and not 0.0 < class_prior < 1.0:
        raise ExperimentValidationError("PU class_prior must be in (0, 1)")

    positives = [row for row in observations if row.target_kind == "positive_unlabeled"]
    invalid_labels = [row.observation_id for row in positives if row.label != 1]
    eligible_positives = [
        row for row in positives if row.split_id == "development" and _allowed_internal(row)
    ]
    reasons: list[str] = []
    if invalid_labels:
        reasons.append("positive_unlabeled_catalog_contains_nonpositive_labels")
    if not eligible_positives:
        reasons.append("no_license_and_split_eligible_positive_rows")
    if unlabeled_count == 0:
        reasons.append("no_unlabeled_pool")
    if class_prior is None:
        reasons.append("class_prior_or_estimation_protocol_missing")
    return Eligibility(
        stage="V2-P4",
        branch="positive_unlabeled",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(reasons),
        counts={
            "positive_rows": len(positives),
            "eligible_development_positives": len(eligible_positives),
            "unlabeled_rows": unlabeled_count,
            "invalid_positive_labels": len(invalid_labels),
            "class_prior": class_prior,
        },
        safeguards=(
            "unlabeled_is_not_negative",
            "nnpu_or_positive_confidence_loss_only",
            "weak_binary_benchmark_is_not_strict_pu_ground_truth",
        ),
    )


def evaluate_pseudo_branch(
    candidates: Sequence[PseudoLabelCandidate],
    reference_observations: Sequence[Observation],
) -> Eligibility:
    forbidden = [row for row in candidates if row.split_id in {"calibration", "sealed"}]
    unknown_splits = [row for row in candidates if row.split_id not in {"development", "calibration", "sealed"}]
    protected_identities = {
        row.identity_group_id
        for row in reference_observations
        if row.split_id in {"calibration", "sealed"}
    }
    identity_leaks = [row for row in candidates if row.identity_group_id in protected_identities]
    development = [row for row in candidates if row.split_id == "development"]
    reasons: list[str] = []
    if not candidates:
        reasons.append("no_pseudo_label_candidates")
    if forbidden:
        reasons.append("pseudo_labels_assigned_to_calibration_or_sealed")
    if unknown_splits:
        reasons.append("pseudo_labels_have_unknown_split")
    if identity_leaks:
        reasons.append("pseudo_identity_overlaps_calibration_or_sealed_truth")
    if not development:
        reasons.append("no_development_pseudo_labels")
    return Eligibility(
        stage="V2-P4",
        branch="pseudo_labels",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(reasons),
        counts={
            "candidate_rows": len(candidates),
            "development_rows": len(development),
            "forbidden_calibration_or_sealed_rows": len(forbidden),
            "unknown_split_rows": len(unknown_splits),
            "protected_identity_overlap_rows": len(identity_leaks),
        },
        safeguards=(
            "pseudo_labels_never_enter_calibration_or_sealed",
            "pseudo_labels_never_supply_validation_metrics",
            "identity_split_inheritance_is_required",
        ),
        evidence=tuple(
            {
                "label_id": row.label_id,
                "identity_group_id": row.identity_group_id,
                "task_id": row.task_id,
                "split_id": row.split_id,
            }
            for row in candidates
        ),
    )


def evaluate_pk_fewshot_branch(
    observations: Sequence[Observation],
    *,
    min_identities_per_task: int = 3,
) -> Eligibility:
    if min_identities_per_task < 1:
        raise ExperimentValidationError("min_identities_per_task must be positive")
    endpoint_names = {"F", "CL", "Vd", "Kp"}
    rows = [
        row
        for row in observations
        if row.endpoint_family in endpoint_names and row.target_kind == "strict_regression"
    ]
    development_rows = [
        row for row in rows if row.split_id == "development" and _allowed_internal(row)
    ]
    by_task: dict[str, list[Observation]] = defaultdict(list)
    for row in development_rows:
        by_task[row.task_id].append(row)
    task_counts = {
        task_id: {
            "rows": len(task_rows),
            "identities": len({row.identity_group_id for row in task_rows}),
            "source_groups": len({source for row in task_rows for source in row.source_group_ids}),
        }
        for task_id, task_rows in sorted(by_task.items())
    }
    eligible_tasks = [
        task_id
        for task_id, counts in task_counts.items()
        if counts["identities"] >= min_identities_per_task
    ]
    reasons: list[str] = []
    if not rows:
        reasons.append("no_strict_pk_endpoint_rows")
    if not development_rows:
        reasons.append("no_license_and_split_eligible_pk_development_rows")
    if not eligible_tasks:
        reasons.append("no_pk_task_meets_minimum_identity_gate")
    return Eligibility(
        stage="V2-P4",
        branch="pk_fewshot",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(reasons),
        counts={
            "total_strict_pk_rows": len(rows),
            "eligible_development_rows": len(development_rows),
            "tasks": task_counts,
            "eligible_task_ids": eligible_tasks,
            "min_identities_per_task": min_identities_per_task,
        },
        safeguards=(
            "fewshot_ranking_not_pbpk",
            "no_holdout_r2_below_30_independent_identities",
            "parameter_subtype_species_route_matrix_remain_separate",
        ),
    )


def evaluate_temporal_challenge_branch(
    candidates: Sequence[TemporalChallengeCandidate],
) -> Eligibility:
    post_cutoff = [row for row in candidates if row.is_post_cutoff]
    overlaps = [row for row in candidates if row.overlaps_training]
    uncleared = [row for row in candidates if not row.license_cleared]
    semantic_mismatch = [row for row in candidates if not row.semantic_aligned]
    usable = [
        row
        for row in post_cutoff
        if not row.overlaps_training and row.license_cleared and row.semantic_aligned
    ]
    reasons: list[str] = []
    if not candidates:
        reasons.append("no_temporal_challenge_manifest")
    if not post_cutoff:
        reasons.append("no_post_cutoff_observations")
    if overlaps:
        reasons.append("temporal_candidates_overlap_training")
    if uncleared:
        reasons.append("temporal_candidates_not_license_cleared")
    if semantic_mismatch:
        reasons.append("temporal_candidates_not_semantically_aligned")
    if not usable:
        reasons.append("no_nonoverlapping_post_cutoff_observations")
    return Eligibility(
        stage="V2-P4",
        branch="temporal_challenge",
        status="not_eligible" if reasons else "eligible",
        reasons=tuple(reasons),
        counts={
            "candidate_rows": len(candidates),
            "post_cutoff_rows": len(post_cutoff),
            "training_overlap_rows": len(overlaps),
            "license_uncleared_rows": len(uncleared),
            "semantic_mismatch_rows": len(semantic_mismatch),
            "usable_rows": len(usable),
            "unique_sources": len({row.source_id for row in usable}),
        },
        safeguards=(
            "cutoff_is_frozen_before_training",
            "identity_structure_source_overlap_audit_required",
            "challenge_is_never_used_for_model_selection",
        ),
        evidence=tuple(
            {
                "observation_id": row.observation_id,
                "source_id": row.source_id,
                "is_post_cutoff": row.is_post_cutoff,
                "overlaps_training": row.overlaps_training,
                "license_cleared": row.license_cleared,
                "semantic_aligned": row.semantic_aligned,
            }
            for row in candidates
        ),
    )


def build_p3_p4_experiment_ledger(
    eligibility_records: Iterable[Eligibility],
) -> dict[str, Any]:
    records = list(eligibility_records)
    if len({record.branch for record in records}) != len(records):
        raise ExperimentValidationError("experiment branches must be unique")
    results = [ExperimentResult.from_eligibility(record) for record in records]
    return {
        "schema_version": "peptide_omnipanel_v2_p3_p4_experiments_v1",
        "training_performed": False,
        "successful_training_branches": 0,
        "eligibility": [record.to_dict() for record in records],
        "results": [result.to_dict() for result in results],
    }
