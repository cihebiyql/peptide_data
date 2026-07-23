"""Peptide-only V31 multimodal features and strict 18-endpoint inference.

The runtime accepts a canonical peptide sequence and predicts the hypothetical
natural-L, unmodified, linear peptide with free N/C termini.  It does not infer
the real chemistry of a modified or cyclic peptide from a bare sequence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from peptide_omnipanel.features import (
    hashed_character_ngrams,
    rdkit_morgan_descriptor_features,
)
from peptide_omnipanel.plm import (
    compute_transformers_esm2_mean_max_embeddings,
    traditional_sequence_features,
)

ENDPOINT_ORDER = (
    "LogD7.4",
    "solubility",
    "F",
    "T1/2",
    "PPB",
    "CL",
    "Vd",
    "BBB",
    "permeability",
    "overall_peptide_toxicity",
    "cytotoxicity",
    "hemolysis",
    "HC50",
    "cell_penetration",
    "membrane_retention",
    "human_plasma_stability",
    "mouse_plasma_stability",
    "intestinal_stability",
)
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
BARE_SEQUENCE_ASSUMPTION = "natural_L_unmodified_linear_free_N_and_C_termini"
FEATURE_SCHEMA_VERSION = "peptide_omnipanel_v31_peptide18_multimodal_features_v1"
FEATURE_WIDTHS = {
    "traditional": 93,
    "esm_mean": 1280,
    "helm_char": 64,
    "smiles_char": 64,
    "morgan": 256,
    "descriptors": 10,
    "availability": 5,
}
FEATURE_ROUTES = {
    "traditional": (0, 93),
    "esm_sequence": (0, 1373),
    "structure": (1373, 1772),
    "multimodal": (0, 1772),
}


class V31Peptide18Error(ValueError):
    """Raised when a V31 feature, bundle, or inference contract fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sequence(value: str) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence or any(residue not in CANONICAL_AA for residue in sequence):
        raise V31Peptide18Error(
            "input must contain one non-empty sequence using only the canonical "
            "20 amino-acid letters"
        )
    if len(sequence) > 1024:
        raise V31Peptide18Error("sequence exceeds the frozen 1024-residue ESM2 limit")
    return sequence


def linear_peptide_helm(sequence: str) -> str:
    canonical = canonical_sequence(sequence)
    return "PEPTIDE1{" + ".".join(canonical) + "}$$$$"


def linear_peptide_smiles(sequence: str) -> str:
    canonical = canonical_sequence(sequence)
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - local runtime includes RDKit
        raise V31Peptide18Error("RDKit is required for deterministic peptide 2D chemistry") from exc
    molecule = Chem.MolFromSequence(canonical)
    if molecule is None:
        raise V31Peptide18Error("RDKit could not generate the assumed linear peptide")
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not smiles:
        raise V31Peptide18Error("RDKit returned an empty peptide SMILES")
    return smiles


def feature_names() -> tuple[str, ...]:
    names = [f"traditional_{index:04d}" for index in range(93)]
    names += [f"esm2_mean_{index:04d}" for index in range(1280)]
    names += [f"helm_char_{index:04d}" for index in range(64)]
    names += [f"smiles_char_{index:04d}" for index in range(64)]
    names += [f"morgan_r2_{index:04d}" for index in range(256)]
    names += [f"rdkit_descriptor_{index:02d}" for index in range(10)]
    names += [
        "availability_sequence",
        "availability_esm",
        "availability_helm",
        "availability_smiles",
        "availability_rdkit",
    ]
    return tuple(names)


def feature_schema_sha256() -> str:
    payload = json.dumps(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_widths": FEATURE_WIDTHS,
            "feature_routes": FEATURE_ROUTES,
            "feature_names": feature_names(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def build_multimodal_feature_row(
    *,
    sequence: Any = "",
    helm: Any = "",
    smiles: Any = "",
    esm_mean: np.ndarray | None = None,
    traditional: np.ndarray | None = None,
) -> np.ndarray:
    """Build one fixed-width row with explicit modality availability masks."""

    sequence_text = _optional_text(sequence)
    helm_text = _optional_text(helm)
    smiles_text = _optional_text(smiles)
    sequence_available = bool(sequence_text)
    if sequence_available:
        sequence_text = canonical_sequence(sequence_text)
    if traditional is None:
        traditional_row = (
            traditional_sequence_features([sequence_text])[0]
            if sequence_available
            else np.zeros(93, dtype=np.float32)
        )
    else:
        traditional_row = np.asarray(traditional, dtype=np.float32).reshape(-1)
    if traditional_row.shape != (93,):
        raise V31Peptide18Error("traditional feature row must have width 93")
    if esm_mean is None:
        esm_row = np.zeros(1280, dtype=np.float32)
        esm_available = False
    else:
        esm_row = np.asarray(esm_mean, dtype=np.float32).reshape(-1)
        if esm_row.shape != (1280,) or not np.isfinite(esm_row).all():
            raise V31Peptide18Error("ESM mean feature row must be finite width 1280")
        esm_available = True
    helm_block = hashed_character_ngrams(
        helm_text or None,
        modality="helm",
        n_features=64,
        ngram_range=(1, 3),
    )
    smiles_block = hashed_character_ngrams(
        smiles_text or None,
        modality="smiles",
        n_features=64,
        ngram_range=(1, 3),
    )
    morgan, descriptors = rdkit_morgan_descriptor_features(
        smiles_text or None,
        radius=2,
        n_bits=256,
        use_rdkit=True,
    )
    availability = np.asarray(
        [
            float(sequence_available),
            float(esm_available),
            float(helm_block.available),
            float(smiles_block.available),
            float(morgan.available and descriptors.available),
        ],
        dtype=np.float32,
    )
    result = np.concatenate(
        [
            traditional_row,
            esm_row,
            np.asarray(helm_block.values, dtype=np.float32),
            np.asarray(smiles_block.values, dtype=np.float32),
            np.asarray(morgan.values, dtype=np.float32),
            np.asarray(descriptors.values, dtype=np.float32),
            availability,
        ]
    ).astype(np.float32, copy=False)
    if result.shape != (1772,) or not np.isfinite(result).all():
        raise V31Peptide18Error("multimodal feature row fails width/finiteness contract")
    return result


def select_feature_route(matrix: np.ndarray, route: str) -> np.ndarray:
    if route not in FEATURE_ROUTES:
        raise V31Peptide18Error(f"unsupported feature route: {route}")
    start, stop = FEATURE_ROUTES[route]
    selected = np.asarray(matrix[:, start:stop], dtype=np.float32)
    if selected.ndim != 2 or selected.shape[1] != stop - start:
        raise V31Peptide18Error("feature-route shape contract failed")
    return selected


def dynamic_sequence_feature_row(
    sequence: str,
    *,
    model_name: str,
    revision: str,
    model_source: Path,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, Any]]:
    canonical = canonical_sequence(sequence)
    mean, _, mean_identity, _, execution = compute_transformers_esm2_mean_max_embeddings(
        [canonical],
        model_name=model_name,
        revision=revision,
        model_source=model_source,
        batch_size=1,
        min_batch_size=1,
        max_tokens_per_batch=2048,
        device=device,
        precision="bf16" if device.startswith("cuda") else "fp32",
        local_files_only=True,
    )
    traditional = traditional_sequence_features([canonical])[0]
    helm = linear_peptide_helm(canonical)
    smiles = linear_peptide_smiles(canonical)
    row = build_multimodal_feature_row(
        sequence=canonical,
        helm=helm,
        smiles=smiles,
        esm_mean=mean[0],
        traditional=traditional,
    )
    metadata = {
        "canonical_sequence": canonical,
        "sequence_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "assumed_helm": helm,
        "assumed_smiles": smiles,
        "assumption": BARE_SEQUENCE_ASSUMPTION,
        "feature_schema_sha256": feature_schema_sha256(),
        "feature_sha256": hashlib.sha256(row.tobytes()).hexdigest(),
        "esm_identity": mean_identity,
        "esm_execution": execution,
    }
    return row.reshape(1, -1), metadata


def _prediction(estimator: Any, task_kind: str, matrix: np.ndarray) -> float:
    if task_kind in {"classification", "positive_unlabeled"}:
        if hasattr(estimator, "predict_proba"):
            probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
            return float(probabilities[0, -1])
        decision = float(np.asarray(estimator.decision_function(matrix)).reshape(-1)[0])
        return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, decision)))))
    return float(np.asarray(estimator.predict(matrix), dtype=float).reshape(-1)[0])


def predict_panel(
    sequence: str,
    *,
    bundle_dir: Path,
    model_source: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise V31Peptide18Error(f"bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("endpoint_order") != list(ENDPOINT_ORDER):
        raise V31Peptide18Error("bundle does not contain the exact ordered 18-endpoint panel")
    if manifest.get("feature_schema_sha256") != feature_schema_sha256():
        raise V31Peptide18Error("feature schema drift detected")
    matrix, input_metadata = dynamic_sequence_feature_row(
        sequence,
        model_name=manifest["esm_model_name"],
        revision=manifest["esm_revision"],
        model_source=model_source,
        device=device,
    )
    records: list[dict[str, Any]] = []
    dynamic_count = 0
    validated_count = 0
    for endpoint in ENDPOINT_ORDER:
        descriptor = manifest["endpoints"][endpoint]
        model_path = bundle_dir / descriptor["model_path"]
        if sha256_file(model_path) != descriptor["model_sha256"]:
            raise V31Peptide18Error(f"model SHA mismatch for {endpoint}")
        estimator = joblib.load(model_path)
        features = select_feature_route(matrix, descriptor["feature_route"])
        value = _prediction(estimator, descriptor["task_kind"], features)
        raw_value = value
        transform = descriptor["output_transform"]
        if transform == "clip_0_1":
            value = float(np.clip(value, 0.0, 1.0))
        elif transform == "clip_nonnegative":
            value = max(0.0, value)
        if not math.isfinite(value):
            raise V31Peptide18Error(f"non-finite dynamic prediction for {endpoint}")
        dynamic_count += 1
        validation_status = descriptor["validation_status"]
        if validation_status == "validated":
            validated_count += 1
        warnings = list(descriptor["warnings"])
        if descriptor["modification_sensitive"]:
            warnings.append("bare_sequence_predicts_hypothetical_unmodified_linear_analogue")
        records.append(
            {
                "endpoint_id": endpoint,
                "primary_output": descriptor["primary_output"],
                "value": value,
                "raw_model_value": raw_value,
                "value_kind": (
                    "ranking_score"
                    if descriptor["task_kind"] == "positive_unlabeled"
                    else "probability"
                    if descriptor["task_kind"] == "classification"
                    else "regression"
                ),
                "probability": (value if descriptor["task_kind"] == "classification" else None),
                "unit": descriptor["unit"],
                "condition_contract": descriptor["condition"],
                "status": "predicted_dynamic_local_model",
                "model_id": descriptor["model_id"],
                "model_sha256": descriptor["model_sha256"],
                "model_family": descriptor["model_family"],
                "feature_route": descriptor["feature_route"],
                "training_row_count": descriptor["training_row_count"],
                "oof_metrics": descriptor["oof_metrics"],
                "evidence_lane": descriptor["evidence_lane"],
                "validation_status": validation_status,
                "research_only": True,
                "low_confidence": bool(descriptor["low_confidence"]),
                "insufficient_validation": validation_status != "validated",
                "modification_sensitive": bool(descriptor["modification_sensitive"]),
                "representation_assumptions": [BARE_SEQUENCE_ASSUMPTION],
                "prediction_interval": descriptor.get("prediction_interval"),
                "warnings": list(dict.fromkeys(warnings)),
            }
        )
    return {
        "schema_version": "peptide_omnipanel_v31_peptide18_prediction_v1",
        "scope": "internal_research_only",
        "input": input_metadata,
        "model_bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "endpoint_record_count": len(records),
        "dynamic_prediction_count": dynamic_count,
        "validated_prediction_count": validated_count,
        "status": "pass" if dynamic_count == 18 else "partial",
        "endpoints": records,
        "warnings": [
            "internal_research_only",
            "bare_sequence_cannot_encode_real_cyclization_or_modifications",
            "low_evidence_endpoints_are_not_clinically_validated",
        ],
    }
