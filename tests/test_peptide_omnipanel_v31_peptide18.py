from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from peptide_omnipanel.v31_peptide18 import (
    ENDPOINT_ORDER,
    FEATURE_ROUTES,
    V31Peptide18Error,
    build_multimodal_feature_row,
    canonical_sequence,
    feature_schema_sha256,
    linear_peptide_helm,
    linear_peptide_smiles,
    select_feature_route,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/peptide_omnipanel_v31_peptide18_endpoint_registry.json"
BUNDLE = ROOT / "models/peptide_omnipanel_v31/release_peptide18_a3_20260722"


def test_registry_is_exact_peptide18_and_forbids_small_molecule_teacher() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert registry["endpoint_count"] == 18
    assert registry["small_molecule_teacher_allowed"] is False
    assert tuple(row["endpoint_id"] for row in registry["endpoints"]) == ENDPOINT_ORDER


def test_bare_sequence_normalization_and_rejection() -> None:
    assert canonical_sequence(" acd ef ") == "ACDEF"
    with pytest.raises(V31Peptide18Error):
        canonical_sequence("")
    with pytest.raises(V31Peptide18Error):
        canonical_sequence("ACDX")
    with pytest.raises(V31Peptide18Error):
        canonical_sequence("A" * 1025)


def test_multimodal_feature_routes_are_finite_and_fixed_width() -> None:
    sequence = "ACDEFGHIK"
    row = build_multimodal_feature_row(
        sequence=sequence,
        helm=linear_peptide_helm(sequence),
        smiles=linear_peptide_smiles(sequence),
        esm_mean=np.arange(1280, dtype=np.float32) / 1280,
    )
    assert row.shape == (1772,)
    assert np.isfinite(row).all()
    assert row[-5:].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    for route, (start, stop) in FEATURE_ROUTES.items():
        selected = select_feature_route(row.reshape(1, -1), route)
        assert selected.shape == (1, stop - start)


def test_bundle_manifest_and_all_model_hashes_are_frozen() -> None:
    manifest_path = BUNDLE / "bundle_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local trained V31 bundle is not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["endpoint_count"] == 18
    assert manifest["dynamic_model_count"] == 18
    assert manifest["validated_model_count"] == 0
    assert manifest["small_molecule_teacher_rows"] == 0
    assert manifest["endpoint_order"] == list(ENDPOINT_ORDER)
    assert manifest["feature_schema_sha256"] == feature_schema_sha256()
    assert list(manifest["endpoints"]) == sorted(ENDPOINT_ORDER)
    for endpoint in ENDPOINT_ORDER:
        descriptor = manifest["endpoints"][endpoint]
        model = BUNDLE / descriptor["model_path"]
        oof = BUNDLE / descriptor["oof_path"]
        assert model.is_file() and sha256_file(model) == descriptor["model_sha256"]
        assert oof.is_file() and sha256_file(oof) == descriptor["oof_sha256"]
        assert descriptor["training_row_count"] > 0
        assert descriptor["validation_status"] == "not_validated"


def test_verified_prediction_artifacts_report_18_dynamic_not_validated() -> None:
    verification_path = ROOT / "docs/peptide_omnipanel_v31_peptide18_verification.json"
    example_path = ROOT / "docs/peptide_omnipanel_v31_peptide18_example_prediction.json"
    if not verification_path.is_file() or not example_path.is_file():
        pytest.skip("local V31 inference artifacts are not present")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert verification["status"] == "pass"
    assert verification["dynamic_model_count"] == 18
    assert verification["validated_model_count"] == 0
    assert verification["probe_training_exact_overlap"] == []
    assert verification["all_endpoints_dynamic_across_probes"] is True
    assert example["status"] == "pass"
    assert example["endpoint_record_count"] == 18
    assert example["dynamic_prediction_count"] == 18
    assert example["validated_prediction_count"] == 0
    assert tuple(row["endpoint_id"] for row in example["endpoints"]) == ENDPOINT_ORDER
    assert all(math.isfinite(float(row["value"])) for row in example["endpoints"])
    assert all(row["research_only"] is True for row in example["endpoints"])
