"""Train local research-only sequence heads and emit a SHA-registered bundle.

The script has two separate routes:

* TDC small-molecule labels train a local structure teacher and are distilled
  onto project peptide structures before a sequence student is fitted.  These
  students are L5 pseudo-label transfer models, never peptide measurements.
* Frozen peptide observations train narrow sequence models only when their
  endpoint semantics are already separated in the staging dataset.

All outputs remain research-only and are consumed only by the isolated V2.8
research router.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import OneClassSVM

from peptide_omnipanel_v28.v28_registry import load_endpoint_registry
from peptide_omnipanel_v28.v28_research_panel import sequence_feature_vector

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TDC = ROOT / "data/peptide_omnipanel_v28_extended_tdc_20260719_run1"
DEFAULT_LOCAL = ROOT / "data/peptide_omnipanel_v28_research_data_20260719_run1"
DEFAULT_MASTER = ROOT / "data/chembl_peptide_master/chembl_peptide_master.tsv"
DEFAULT_REGISTRY = ROOT / "configs/peptide_omnipanel_v28_endpoint_registry.json"
DEFAULT_OUT_DIR = ROOT / "data/peptide_omnipanel_v28_research_bundle_20260719_run1"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
FEATURE_SCHEMA = {"schema_version": "v28_sequence_physchem_char64_v1", "dimensions": 93}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset(directory: Path) -> tuple[Path, str]:
    manifest = read_json(directory / "dataset_manifest.json")
    observations_ref = manifest["outputs"]["observations"]
    observations = ROOT / observations_ref
    if not observations.is_file():
        raise FileNotFoundError(f"dataset manifest points to missing observations: {observations}")
    digest = sha256_file(observations)
    if digest != manifest["outputs"]["observations_sha256"]:
        raise ValueError(f"dataset observations hash mismatch: {observations}")
    return observations, digest


def canonical_sequence(value: str) -> str | None:
    sequence = "".join(str(value or "").split()).upper()
    return sequence if sequence and set(sequence) <= AA else None


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def morgan_matrix(smiles_values: Iterable[str], *, bits: int = 256) -> tuple[np.ndarray, list[int]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    vectors: list[np.ndarray] = []
    retained: list[int] = []
    for index, value in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        vectors.append(array)
        retained.append(index)
    if not vectors:
        return np.empty((0, bits), dtype=np.uint8), retained
    return np.vstack(vectors), retained


def sequence_matrix(sequences: list[str]) -> np.ndarray:
    return np.vstack([sequence_feature_vector(sequence) for sequence in sequences])


def aggregate_targets(rows: Iterable[dict[str, str]]) -> tuple[list[str], np.ndarray]:
    targets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "available":
            continue
        sequence = canonical_sequence(row.get("sequence", ""))
        target = finite(row.get("target"))
        if sequence is not None and target is not None:
            targets[sequence].append(target)
    sequences = sorted(targets)
    return sequences, np.asarray([float(np.median(targets[sequence])) for sequence in sequences])


def endpoint_semantics(
    endpoint_id: str,
    task_kind: str,
    unit: str,
    *,
    assay: str,
    species: str,
    matrix: str,
    value_semantics: str,
    positive_class_semantics: str | None = None,
) -> dict[str, str]:
    value = {
        "endpoint_id": endpoint_id,
        "task_kind": task_kind,
        "unit": unit,
        "parameter_basis": "endpoint_specific_registered_basis",
        "assay": assay,
        "species": species,
        "matrix": matrix,
        "value_semantics": value_semantics,
    }
    if positive_class_semantics is not None:
        value["positive_class_semantics"] = positive_class_semantics
    return value


def feature_domain(features: np.ndarray) -> tuple[list[float], float]:
    center = features.mean(axis=0)
    distances = np.linalg.norm(features - center, axis=1)
    return center.tolist(), max(float(np.quantile(distances, 0.95)), 1e-6)


def write_artifact(
    *,
    out_dir: Path,
    endpoint_id: str,
    estimator: Any,
    training_hash: str,
    semantics: Mapping[str, str],
    research_model: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = endpoint_id.lower().replace("/", "_").replace(".", "_")
    artifact = model_dir / f"{slug}.joblib"
    payload = {"estimator": estimator, "research_model": dict(research_model)}
    joblib.dump(payload, artifact, compress=3)
    digest = sha256_file(artifact)
    manifest_path = model_dir / f"{slug}.manifest.json"
    manifest = {
        "artifact_origin": "project_trained",
        "format": "joblib",
        "model_path": artifact.name,
        "sha256": digest,
        "feature_schema": FEATURE_SCHEMA,
        "seed": seed,
        "training_input_hash": training_hash,
        "endpoint_semantics": dict(semantics),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "endpoint_id": endpoint_id,
        "model_id": f"v28_research_{slug}",
        "artifact_manifest": str(manifest_path.relative_to(out_dir)),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_sha256": digest,
    }


def local_models(
    rows: list[dict[str, str]],
    *,
    registry: Mapping[str, Any],
    out_dir: Path,
    training_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["endpoint_id"]].append(row)
    artifacts: list[dict[str, Any]] = []
    classification = {"hemolysis", "overall_peptide_toxicity", "cytotoxicity"}
    regression = {
        "HC50",
        "human_plasma_stability",
        "mouse_plasma_stability",
        "intestinal_stability",
    }
    for endpoint_id in sorted(classification.union(regression)):
        sequences, targets = aggregate_targets(grouped.get(endpoint_id, []))
        if len(sequences) < 25:
            continue
        features = sequence_matrix(sequences)
        center, radius = feature_domain(features)
        spec = registry[endpoint_id]
        if endpoint_id in classification:
            labels = targets.astype(int)
            if len(set(labels)) < 2:
                continue
            estimator: Any = ExtraTreesClassifier(
                n_estimators=128, min_samples_leaf=2, max_features=0.7, n_jobs=-1, random_state=seed
            ).fit(features, labels)
            model_kind = "classification"
            task_kind = "classification"
            inverse_transform = "identity"
            positive = f"staged {endpoint_id} source-defined positive class"
        else:
            estimator = ExtraTreesRegressor(
                n_estimators=128, min_samples_leaf=2, max_features=0.7, n_jobs=-1, random_state=seed
            ).fit(features, targets)
            model_kind = "regression"
            task_kind = "censored_regression" if endpoint_id != "HC50" else "censored_regression"
            inverse_transform = "pow10" if endpoint_id.endswith("stability") else "identity"
            positive = None
        source_tiers = sorted({row["evidence_tier"] for row in grouped[endpoint_id]})
        evidence_tier = source_tiers[0] if len(source_tiers) == 1 else "mixed_research_source_tiers"
        semantics = endpoint_semantics(
            endpoint_id,
            task_kind,
            spec.unit,
            assay=f"local_frozen_{endpoint_id}_research_dataset",
            species="endpoint_conditioned_or_source_mixed",
            matrix="endpoint_conditioned_or_source_mixed",
            value_semantics=spec.primary_output,
            positive_class_semantics=positive,
        )
        artifacts.append(
            write_artifact(
                out_dir=out_dir,
                endpoint_id=endpoint_id,
                estimator=estimator,
                training_hash=training_hash,
                semantics=semantics,
                research_model={
                    "output_kind": model_kind,
                    "inverse_transform": inverse_transform,
                    "feature_center": center,
                    "feature_radius": radius,
                    "evidence_tier": evidence_tier,
                    "peptide_validated": endpoint_id in {"hemolysis", "HC50"},
                    "warnings": ["research_only", "low_confidence", "narrow_source_conditions"],
                    "training_rows": len(sequences),
                },
                seed=seed,
            )
        )
    pu_sequences, _ = aggregate_targets(grouped.get("cell_penetration", []))
    if len(pu_sequences) >= 25:
        features = sequence_matrix(pu_sequences)
        center, radius = feature_domain(features)
        estimator = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(features)
        spec = registry["cell_penetration"]
        artifacts.append(
            write_artifact(
                out_dir=out_dir,
                endpoint_id="cell_penetration",
                estimator=estimator,
                training_hash=training_hash,
                semantics=endpoint_semantics(
                    "cell_penetration",
                    "positive_unlabeled",
                    spec.unit,
                    assay="local_frozen_positive_membership",
                    species="source_mixed",
                    matrix="cell_uptake",
                    value_semantics="positive_unlabeled_cell_penetration_ranking",
                    positive_class_semantics=(
                        "source-curated cell penetrating peptide membership only"
                    ),
                ),
                research_model={
                    "output_kind": "positive_unlabeled",
                    "inverse_transform": "identity",
                    "feature_center": center,
                    "feature_radius": radius,
                    "evidence_tier": "L3_positive_unlabeled",
                    "peptide_validated": False,
                    "warnings": [
                        "research_only",
                        "low_confidence",
                        "positive_unlabeled_not_probability",
                    ],
                    "training_rows": len(pu_sequences),
                },
                seed=seed,
            )
        )
    return artifacts


def master_peptides(path: Path, maximum: int, seed: int) -> tuple[list[str], list[str]]:
    rows: list[tuple[str, str]] = []
    for row in read_tsv(path):
        sequence = canonical_sequence(row.get("linear_sequence", ""))
        smiles = row.get("canonical_smiles", "")
        if sequence is not None and smiles:
            rows.append((sequence, smiles))
    rng = np.random.default_rng(seed)
    if len(rows) > maximum:
        rows = [rows[index] for index in sorted(rng.choice(len(rows), size=maximum, replace=False))]
    return [row[0] for row in rows], [row[1] for row in rows]


def tdc_transfer_models(
    rows: list[dict[str, str]],
    *,
    master_sequences: list[str],
    master_smiles: list[str],
    registry: Mapping[str, Any],
    out_dir: Path,
    training_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["endpoint_id"]].append(row)
    master_morgan, master_indices = morgan_matrix(master_smiles)
    selected_sequences = [master_sequences[index] for index in master_indices]
    selected_features = sequence_matrix(selected_sequences)
    artifacts: list[dict[str, Any]] = []
    for offset, endpoint_id in enumerate(sorted(grouped)):
        endpoint_rows = grouped[endpoint_id]
        targets = [finite(row["value"]) for row in endpoint_rows]
        structure, indices = morgan_matrix([row["smiles"] for row in endpoint_rows])
        labels = np.asarray([targets[index] for index in indices], dtype=float)
        if not len(structure) or not len(master_morgan):
            continue
        task_kind = endpoint_rows[0]["task_kind"]
        current_seed = seed + offset
        if task_kind == "classification":
            teacher: Any = ExtraTreesClassifier(
                n_estimators=64,
                min_samples_leaf=2,
                max_features=0.7,
                n_jobs=-1,
                random_state=current_seed,
            ).fit(structure, labels.astype(int))
            classes = list(teacher.classes_)
            pseudo = teacher.predict_proba(master_morgan)[:, classes.index(1)]
            semantics_kind = "classification"
            output_kind = "probability_regression"
            positive = endpoint_rows[0]["positive_class_semantics"]
        else:
            teacher = ExtraTreesRegressor(
                n_estimators=64,
                min_samples_leaf=2,
                max_features=0.7,
                n_jobs=-1,
                random_state=current_seed,
            ).fit(structure, labels)
            pseudo = teacher.predict(master_morgan)
            semantics_kind = "regression"
            output_kind = "regression"
            positive = None
        student = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=3,
            max_features=0.7,
            n_jobs=-1,
            random_state=current_seed,
        ).fit(selected_features, pseudo)
        center, radius = feature_domain(selected_features)
        spec = registry[endpoint_id]
        artifacts.append(
            write_artifact(
                out_dir=out_dir,
                endpoint_id=endpoint_id,
                estimator=student,
                training_hash=training_hash,
                semantics=endpoint_semantics(
                    endpoint_id,
                    semantics_kind,
                    spec.unit,
                    assay=endpoint_rows[0]["source_dataset"],
                    species="small_molecule_source_not_peptide_validated",
                    matrix="source_defined",
                    value_semantics=endpoint_rows[0]["value_semantics"],
                    positive_class_semantics=positive or None,
                ),
                research_model={
                    "output_kind": output_kind,
                    "inverse_transform": "identity",
                    "feature_center": center,
                    "feature_radius": radius,
                    "evidence_tier": "L5_self_model_pseudo_label",
                    "peptide_validated": False,
                    "warnings": [
                        "research_only",
                        "low_confidence",
                        "small_molecule_transfer_teacher",
                        "sequence_student_trained_on_pseudo_labels",
                        "not_peptide_validated",
                    ],
                    "teacher_training_rows": len(labels),
                    "student_training_rows": len(selected_sequences),
                },
                seed=current_seed,
            )
        )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdc-dir", default=str(DEFAULT_TDC))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL))
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-master-peptides", type=int, default=6000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tdc_observations, tdc_hash = verify_dataset(Path(args.tdc_dir).resolve())
    local_observations, local_hash = verify_dataset(Path(args.local_dir).resolve())
    master = Path(args.master).resolve()
    if not master.is_file():
        raise FileNotFoundError(master)
    registry_data = load_endpoint_registry(Path(args.registry).resolve())
    registry = {endpoint.endpoint_id: endpoint for endpoint in registry_data.endpoints}
    out_dir = Path(args.out_dir).resolve()
    master_sequences, master_smiles = master_peptides(master, args.max_master_peptides, args.seed)
    tdc_artifacts = tdc_transfer_models(
        list(read_tsv(tdc_observations)),
        master_sequences=master_sequences,
        master_smiles=master_smiles,
        registry=registry,
        out_dir=out_dir,
        training_hash=tdc_hash,
        seed=args.seed,
    )
    local_artifacts = local_models(
        list(read_tsv(local_observations)),
        registry=registry,
        out_dir=out_dir,
        training_hash=local_hash,
        seed=args.seed,
    )
    artifacts = sorted(tdc_artifacts + local_artifacts, key=lambda item: item["endpoint_id"])
    endpoint_ids = [item["endpoint_id"] for item in artifacts]
    if len(endpoint_ids) != len(set(endpoint_ids)):
        raise RuntimeError("multiple research artifacts were generated for one endpoint")
    bundle = {
        "schema_version": "peptide-omnipanel-v28-research-bundle-1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "scope": "internal_research_only",
        "active_service_changed": False,
        "inputs": {
            "tdc_observations": str(tdc_observations.relative_to(ROOT)),
            "tdc_observations_sha256": tdc_hash,
            "local_observations": str(local_observations.relative_to(ROOT)),
            "local_observations_sha256": local_hash,
            "peptide_master": str(master.relative_to(ROOT)),
            "peptide_master_sha256": sha256_file(master),
        },
        "registered_sha256": {item["endpoint_id"]: item["artifact_sha256"] for item in artifacts},
        "models": [
            {key: value for key, value in item.items() if key != "artifact_sha256"}
            for item in artifacts
        ],
        "counts": {
            "sequence_models": len(artifacts),
            "tdc_transfer_models": len(tdc_artifacts),
            "local_peptide_models": len(local_artifacts),
            "fallback_only_endpoints": sorted(set(registry).difference(endpoint_ids)),
            "master_peptides_for_distillation": len(master_sequences),
        },
        "invariants": [
            "All models are project-trained local artifacts; no third-party prediction API or "
            "weights are used.",
            "TDC transfer students are L5 pseudo-label models and not measured peptide endpoint "
            "models.",
            "This bundle is research-only and must not replace the active V2.6 service.",
        ],
    }
    bundle_path = out_dir / "research_bundle_manifest.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bundle["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
