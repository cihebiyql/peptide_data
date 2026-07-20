from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from peptide_omnipanel_v28.v28_research_panel import (
    build_research_only_router,
    sequence_feature_vector,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/peptide_omnipanel_v28_endpoint_registry.json"
PREDICT_SCRIPT = ROOT / "scripts" / "predict_peptide_omnipanel_v28_research.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path) -> Path:
    model_dir = root / "models"
    model_dir.mkdir()
    features = np.vstack([sequence_feature_vector("ACDE"), sequence_feature_vector("KKKK")])
    estimator = DummyClassifier(strategy="prior").fit(features, np.asarray([0, 1]))
    artifact = model_dir / "hia.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "research_model": {
                "output_kind": "classification",
                "inverse_transform": "identity",
                "feature_center": features.mean(axis=0).tolist(),
                "feature_radius": 1.0,
                "evidence_tier": "L2_source_defined_binary",
                "peptide_validated": False,
                "warnings": ["research_only", "low_confidence"],
            },
        },
        artifact,
    )
    artifact_sha = _sha256(artifact)
    manifest = {
        "artifact_origin": "project_trained",
        "format": "joblib",
        "model_path": "hia.joblib",
        "sha256": artifact_sha,
        "feature_schema": {"schema_version": "v28_sequence_physchem_char64_v1"},
        "seed": 20260719,
        "training_input_hash": "1" * 64,
        "endpoint_semantics": {
            "endpoint_id": "HIA",
            "task_kind": "classification",
            "unit": "probability",
            "parameter_basis": "human_intestinal_absorption",
            "assay": "fixture",
            "species": "source_defined",
            "matrix": "source_defined",
            "value_semantics": "source_defined_HIA",
            "positive_class_semantics": "source label 1",
        },
    }
    artifact_manifest = model_dir / "hia.manifest.json"
    artifact_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = {
        "scope": "internal_research_only",
        "registered_sha256": {"HIA": artifact_sha},
        "models": [
            {
                "endpoint_id": "HIA",
                "model_id": "fixture_hia",
                "artifact_manifest": "models/hia.manifest.json",
            }
        ],
    }
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    return bundle_path


def _prediction_script_module():
    specification = importlib.util.spec_from_file_location("v28_research_predict", PREDICT_SCRIPT)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_research_panel_returns_all_39_endpoints_with_explicit_low_evidence(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    router = build_research_only_router(
        endpoint_registry_path=REGISTRY,
        research_bundle_path=bundle,
    )

    result = router.predict("ACDE").to_dict()["endpoints"]

    assert len(result) == 39
    assert all(value["research_only"] for value in result.values())
    assert all(value["low_confidence"] for value in result.values())
    assert result["HIA"]["prediction_tier"] == "C"
    assert result["Kp"]["prediction_tier"] == "E"
    assert set(result["Kp"]["value"]) == {"brain", "liver", "kidney", "muscle", "adipose"}
    assert len(result["degradation_site_probability"]["value"]) == 4


def test_research_cli_response_keeps_explicit_panel_boundary(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    script = _prediction_script_module()

    result = script.research_response("ACDE", input_format="auto", registry=REGISTRY, bundle=bundle)

    assert result["panel_metadata"] == {
        "architecture_id": "peptide_omnipanel_v28_research_only",
        "scope": "internal_research_only",
        "active_service_changed": False,
        "endpoint_count": 39,
        "research_only": True,
        "low_confidence": True,
        "warnings": [
            "not_a_clinical_or_release_prediction",
            "all_endpoint_outputs_require_research_only_interpretation",
        ],
    }
    assert all(
        item["research_only"] and item["low_confidence"] for item in result["endpoints"].values()
    )
