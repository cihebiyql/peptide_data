"""Materialize V2.1 direct champions on the isolated development view only."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import artifact_entry, now_iso, sha256_file, write_json
from .datasets import Observation, load_observations
from .optimization import (
    ModelCandidate,
    build_estimator,
    feature_schema,
    model_candidates,
    representation_feature_vector,
    representation_text,
)
from .training import observation_allowed

V21_FROZEN_CHAMPIONS_SHA256 = "6f1a7de942f6083cd16a62c66fab0540e45135005b281d83468429c0d6ab4070"
V21_FROZEN_TASK_COUNT = 11
SUPPORTED_REPRESENTATIONS = frozenset({"sequence", "helm", "structure"})
SUPPORTED_TASK_KINDS = frozenset({"regression", "binary_classification"})
DEVELOPMENT_FILENAMES = ("strict_numeric.tsv", "binary_evidence_catalog.tsv")


@dataclass(frozen=True)
class FrozenChampion:
    task_id: str
    endpoint_family: str
    task_kind: str
    representation: str
    candidate_id: str
    candidate_name: str
    framework: str
    parameters: dict[str, Any]
    development_rows: int
    development_components: int
    cohort_sha256: str
    run_name: str

    def candidate(self) -> ModelCandidate:
        return ModelCandidate(
            name=self.candidate_name,
            framework=self.framework,
            parameters=dict(self.parameters),
        )


def _required_text(row: dict[str, str], field: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise RuntimeError(f"frozen champion row is missing {field}")
    return value


def _parse_champion(row: dict[str, str]) -> FrozenChampion:
    if row.get("rank", "").strip() != "1" or row.get("is_champion", "").strip() != "true":
        raise RuntimeError("frozen champion input contains a non-champion row")
    if row.get("run_kind", "").strip() != "classical":
        raise RuntimeError("V2.6 Stage 0 accepts only classical V2.1 direct champions")
    task_kind = _required_text(row, "task_kind")
    representation = _required_text(row, "representation")
    if task_kind not in SUPPORTED_TASK_KINDS:
        raise RuntimeError(f"unsupported frozen champion task kind: {task_kind}")
    if representation not in SUPPORTED_REPRESENTATIONS:
        raise RuntimeError(f"unsupported frozen champion representation: {representation}")
    try:
        parameters = json.loads(_required_text(row, "parameters"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("frozen champion parameters are not valid JSON") from exc
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen champion parameters must contain a JSON object")
    try:
        development_rows = int(_required_text(row, "rows"))
        development_components = int(_required_text(row, "components"))
    except ValueError as exc:
        raise RuntimeError("frozen champion row/component counts must be integers") from exc
    if development_rows < 1 or development_components < 1:
        raise RuntimeError("frozen champion row/component counts must be positive")
    cohort = _required_text(row, "cohort_sha256")
    if len(cohort) != 64 or any(character not in "0123456789abcdef" for character in cohort):
        raise RuntimeError("frozen champion cohort_sha256 is invalid")
    return FrozenChampion(
        task_id=_required_text(row, "task_id"),
        endpoint_family=_required_text(row, "endpoint_family"),
        task_kind=task_kind,
        representation=representation,
        candidate_id=_required_text(row, "candidate_id"),
        candidate_name=_required_text(row, "candidate_name"),
        framework=_required_text(row, "framework"),
        parameters=parameters,
        development_rows=development_rows,
        development_components=development_components,
        cohort_sha256=cohort,
        run_name=_required_text(row, "run_name"),
    )


def load_frozen_champions(
    path: Path,
    *,
    expected_sha256: str = V21_FROZEN_CHAMPIONS_SHA256,
    expected_task_count: int = V21_FROZEN_TASK_COUNT,
) -> list[FrozenChampion]:
    """Load the SHA-pinned V2.1 cohort champion table."""

    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"frozen champion table SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    champions = [_parse_champion(row) for row in rows]
    if len(champions) != expected_task_count:
        raise RuntimeError(
            f"frozen champion task count mismatch: {len(champions)} != {expected_task_count}"
        )
    task_ids = [champion.task_id for champion in champions]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("frozen champion table contains duplicate task IDs")
    candidate_ids = [champion.candidate_id for champion in champions]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("frozen champion table contains duplicate candidate IDs")
    return sorted(champions, key=lambda champion: champion.task_id)


def validate_champion_registry(champion: FrozenChampion) -> None:
    """Prove the serialized configuration is still an exact registered V2.1 candidate."""

    matches = [
        candidate
        for candidate in model_candidates(champion.task_kind)
        if candidate.name == champion.candidate_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"registered V2.1 candidate is unavailable: {champion.candidate_name}")
    registered = matches[0]
    if registered.framework != champion.framework or registered.parameters != champion.parameters:
        raise RuntimeError(
            f"frozen champion configuration differs from registry: {champion.task_id}"
        )


def cohort_sha256(rows: Iterable[Observation]) -> str:
    """Reproduce the V2.1 OOF cohort identity hash exactly."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.observation_id):
        target = row.model_target
        if target is None or row.component_id is None:
            raise RuntimeError(f"incomplete cohort identity for {row.observation_id}")
        digest.update(
            "\0".join((row.observation_id, row.component_id, str(float(target)))).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def select_champion_rows(
    observations: Sequence[Observation],
    champion: FrozenChampion,
    *,
    license_policy: str,
) -> list[Observation]:
    """Select exactly the development cohort used by a V2.1 champion."""

    if any(row.split_id != "development" for row in observations):
        raise RuntimeError("V2.6 Stage 0 received a non-development observation")
    rows = sorted(
        (
            row
            for row in observations
            if row.task_id == champion.task_id
            and observation_allowed(row, license_policy)
            and representation_text(row, champion.representation) is not None
        ),
        key=lambda row: row.observation_id,
    )
    if len(rows) != champion.development_rows:
        raise RuntimeError(
            f"development row count mismatch for {champion.task_id}: "
            f"{len(rows)} != {champion.development_rows}"
        )
    components = {row.component_id for row in rows}
    if None in components or len(components) != champion.development_components:
        raise RuntimeError(
            f"development component count mismatch for {champion.task_id}: "
            f"{len(components - {None})} != {champion.development_components}"
        )
    observed_cohort = cohort_sha256(rows)
    if observed_cohort != champion.cohort_sha256:
        raise RuntimeError(
            f"development cohort SHA mismatch for {champion.task_id}: "
            f"{observed_cohort} != {champion.cohort_sha256}"
        )
    contract = {(row.endpoint_family, row.task_kind) for row in rows}
    if contract != {(champion.endpoint_family, champion.task_kind)}:
        raise RuntimeError(f"task contract mismatch for {champion.task_id}")
    return rows


def training_matrix(rows: Sequence[Observation], representation: str) -> tuple[Any, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - training-host contract.
        raise RuntimeError("NumPy is required for frozen-direct materialization") from exc
    vectors: list[tuple[float, ...]] = []
    targets: list[float] = []
    for row in rows:
        text = representation_text(row, representation)
        if text is None:
            raise RuntimeError(
                f"required {representation} representation disappeared for {row.observation_id}"
            )
        target = row.model_target
        if target is None:
            raise RuntimeError(f"missing target for {row.observation_id}")
        vectors.append(representation_feature_vector(text, representation))
        targets.append(float(target))
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
    )


def fit_frozen_champion(
    champion: FrozenChampion,
    matrix: Any,
    targets: Any,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    validate_champion_registry(champion)
    estimator = build_estimator(champion.candidate(), champion.task_kind, seed=seed, n_jobs=n_jobs)
    fit_kwargs: dict[str, Any] = {}
    if (
        champion.task_kind == "binary_classification"
        and champion.parameters.get("auto_class_weight") is True
    ):
        import numpy as np

        labels = targets.astype(int)
        counts = np.bincount(labels, minlength=2)
        if not np.all(counts > 0):
            raise RuntimeError(f"binary development cohort lacks a class: {champion.task_id}")
        class_weights = len(labels) / (2.0 * counts)
        fit_kwargs["sample_weight"] = class_weights[labels]
    estimator.fit(matrix, targets, **fit_kwargs)
    return estimator


def published_artifact_entry(path: Path, root: Path) -> dict[str, Any]:
    """Describe an artifact by its stable bundle-relative path."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"artifact is outside the pending bundle root: {path}") from exc
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _registered_development_artifact(manifest: dict[str, Any], filename: str) -> dict[str, Any]:
    matches = [
        artifact
        for artifact in manifest.get("output_artifacts", [])
        if Path(str(artifact.get("path", ""))).as_posix().endswith(f"/development/{filename}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"views manifest does not uniquely register development/{filename}")
    return dict(matches[0])


def validate_development_view(
    development_dir: Path, views_manifest_path: Path
) -> tuple[list[Observation], dict[str, Any]]:
    """Load only registered development files; calibration/sealed paths are never opened."""

    manifest = json.loads(views_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "pass"
        or manifest.get("sealed_inventory_contains_targets") is not False
        or manifest.get("development_views_contain_only_development") is not True
    ):
        raise RuntimeError("views manifest does not prove development-only label isolation")
    observations: list[Observation] = []
    inputs: list[dict[str, Any]] = []
    for filename in DEVELOPMENT_FILENAMES:
        path = development_dir / filename
        registered = _registered_development_artifact(manifest, filename)
        observed_sha = sha256_file(path)
        if observed_sha != registered.get("sha256"):
            raise RuntimeError(f"development view SHA256 mismatch: {path}")
        loaded = load_observations(path)
        if len(loaded) != int(registered.get("rows", -1)):
            raise RuntimeError(f"development view row count mismatch: {path}")
        observations.extend(loaded)
        inputs.append(artifact_entry(path, rows=len(loaded)))
    if any(row.split_id != "development" for row in observations):
        raise RuntimeError("development view contains a non-development observation")
    return observations, {
        "views_manifest_sha256": sha256_file(views_manifest_path),
        "development_artifacts": inputs,
        "development_observations": len(observations),
    }


def materialize_frozen_direct(
    *,
    development_dir: Path,
    views_manifest_path: Path,
    champions_path: Path,
    output_dir: Path,
    expected_champions_sha256: str = V21_FROZEN_CHAMPIONS_SHA256,
    expected_task_count: int = V21_FROZEN_TASK_COUNT,
    license_policy: str = "research_candidate",
    seed: int = 20260716,
    n_jobs: int = 8,
    execution_files: Sequence[Path] = (),
) -> dict[str, Any]:
    """Fit all frozen direct champions and atomically publish their audited bundle."""

    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing frozen bundle: {output_dir}")
    champions = load_frozen_champions(
        champions_path,
        expected_sha256=expected_champions_sha256,
        expected_task_count=expected_task_count,
    )
    observations, development_evidence = validate_development_view(
        development_dir, views_manifest_path
    )
    pending = output_dir.with_name(f".{output_dir.name}.pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    task_entries: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    try:
        import joblib

        for index, champion in enumerate(champions, start=1):
            rows = select_champion_rows(observations, champion, license_policy=license_policy)
            matrix, targets = training_matrix(rows, champion.representation)
            estimator = fit_frozen_champion(
                champion,
                matrix,
                targets,
                seed=seed,
                n_jobs=n_jobs,
            )
            task_digest = hashlib.sha256(champion.task_id.encode("utf-8")).hexdigest()[:16]
            artifact_path = pending / f"task_{index:02d}_{task_digest}.joblib"
            metadata = {
                **asdict(champion),
                "feature_schema": feature_schema(champion.representation),
                "training_data": "development_only",
                "calibration_loaded": False,
                "sealed_labels_loaded": False,
                "seed": seed,
            }
            joblib.dump({"model": estimator, "metadata": metadata}, artifact_path)
            task_entry = {
                **metadata,
                "artifact_path": artifact_path.name,
                "artifact_sha256": sha256_file(artifact_path),
                "artifact_bytes": artifact_path.stat().st_size,
            }
            task_entries.append(task_entry)
            artifact_rows.append(published_artifact_entry(artifact_path, pending))

        representation_counts = Counter(champion.representation for champion in champions)
        source_files = {str(path): sha256_file(path) for path in sorted(execution_files)}
        manifest = {
            "schema_version": "v26-frozen-direct-bundle-1",
            "stage": "V2.6-full-development-only-direct-materialization",
            "created_at": now_iso(),
            "status": "pass",
            "selection_source": "frozen_v21_development_component_oof_champions",
            "training_data": "development_only",
            "calibration_loaded": False,
            "sealed_labels_loaded": False,
            "active_service_changed": False,
            "license_policy": license_policy,
            "seed": seed,
            "n_jobs": n_jobs,
            "task_count": len(task_entries),
            "representation_task_counts": dict(sorted(representation_counts.items())),
            "champion_table": artifact_entry(champions_path, rows=len(champions)),
            "development_evidence": development_evidence,
            "source_files": source_files,
            "artifacts": artifact_rows,
            "tasks": task_entries,
        }
        manifest_path = pending / "manifest.json"
        write_json(manifest_path, manifest)
        (pending / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        pending.rename(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
