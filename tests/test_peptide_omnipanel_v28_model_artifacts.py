from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from peptide_omnipanel_v28.v28_model_artifacts import (
    ArtifactManifestError,
    load_model_artifact,
    validate_model_artifact_manifest,
)


def endpoint_semantics(task_kind: str = "regression") -> dict[str, str]:
    semantics = {
        "endpoint_id": "LogD7.4",
        "task_kind": task_kind,
        "unit": "log10_ratio",
        "parameter_basis": "distribution_coefficient_pH7.4",
        "assay": "shake_flask",
        "species": "not_applicable",
        "matrix": "octanol_buffer",
        "value_semantics": "continuous_LogD_at_pH7.4",
    }
    if task_kind in {"classification", "binary_classification", "positive_unlabeled"}:
        semantics["positive_class_semantics"] = "endpoint_specific_positive_class"
    return semantics


def write_fixture(root: Path) -> tuple[Path, dict[str, object], str]:
    artifact = root / "models" / "logd.joblib"
    artifact.parent.mkdir()
    joblib.dump({"estimator": {"name": "fixture-model"}}, artifact, compress=3)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "artifact_origin": "project_trained",
        "format": "joblib",
        "model_path": "models/logd.joblib",
        "sha256": digest,
        "feature_schema": {
            "schema_version": "morgan_v1",
            "n_bits": 256,
        },
        "seed": 20260719,
        "training_input_hash": "1" * 64,
        "endpoint_semantics": endpoint_semantics(),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest, digest


class V28ModelArtifactTests(unittest.TestCase):
    def test_validates_registered_local_artifact_and_loads_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, manifest, digest = write_fixture(root)
            specification = validate_model_artifact_manifest(
                manifest,
                manifest_dir=root,
                registered_sha256s={digest},
            )
            loaded = load_model_artifact(
                manifest_path,
                registered_sha256s={digest},
            )

        self.assertEqual(specification.sha256, digest)
        self.assertEqual(specification.endpoint_semantics.endpoint_id, "LogD7.4")
        self.assertEqual(loaded.model, {"name": "fixture-model"})
        self.assertEqual(loaded.specification.manifest_path, manifest_path.resolve())

    def test_rejects_missing_contract_fields_and_incomplete_endpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, manifest, digest = write_fixture(root)
            for field in (
                "model_path",
                "sha256",
                "feature_schema",
                "seed",
                "training_input_hash",
                "endpoint_semantics",
            ):
                invalid = dict(manifest)
                invalid.pop(field)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ArtifactManifestError, "missing required fields"
                ):
                    validate_model_artifact_manifest(
                        invalid,
                        manifest_dir=root,
                        registered_sha256s={digest},
                    )

            invalid = dict(manifest)
            invalid["endpoint_semantics"] = {"endpoint_id": "LogD7.4"}
            with self.assertRaisesRegex(ArtifactManifestError, "endpoint_semantics is missing"):
                validate_model_artifact_manifest(
                    invalid,
                    manifest_dir=root,
                    registered_sha256s={digest},
                )

    def test_rejects_third_party_api_and_unregistered_weights_before_joblib(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, manifest, digest = write_fixture(root)
            for origin in ("third_party", "api", "external_pretrained"):
                invalid = dict(manifest)
                invalid["artifact_origin"] = origin
                with self.subTest(origin=origin), self.assertRaisesRegex(
                    ArtifactManifestError, "only project_trained"
                ):
                    validate_model_artifact_manifest(
                        invalid,
                        manifest_dir=root,
                        registered_sha256s={digest},
                    )

            with patch("joblib.load") as mocked_load, self.assertRaisesRegex(
                ArtifactManifestError, "not present in the trusted registry"
            ):
                load_model_artifact(manifest_path, registered_sha256s={"2" * 64})
            mocked_load.assert_not_called()

    def test_rejects_tampered_weight_before_joblib_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path, manifest, digest = write_fixture(root)
            (root / str(manifest["model_path"])).write_bytes(b"tampered")
            with patch("joblib.load") as mocked_load, self.assertRaisesRegex(
                ArtifactManifestError, "SHA256 mismatch"
            ):
                load_model_artifact(manifest_path, registered_sha256s={digest})
            mocked_load.assert_not_called()

    def test_rejects_absolute_escape_and_remote_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, manifest, digest = write_fixture(root)
            invalid_paths = (
                str((root / "models/logd.joblib").resolve()),
                "../outside.joblib",
                "https://models.example/logd.joblib",
                "api://prediction/logd",
            )
            for model_path in invalid_paths:
                invalid = dict(manifest)
                invalid["model_path"] = model_path
                with self.subTest(model_path=model_path), self.assertRaises(
                    ArtifactManifestError
                ):
                    validate_model_artifact_manifest(
                        invalid,
                        manifest_dir=root,
                        registered_sha256s={digest},
                    )

    def test_classification_requires_positive_class_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, manifest, digest = write_fixture(root)
            invalid = dict(manifest)
            invalid["endpoint_semantics"] = endpoint_semantics("binary_classification")
            invalid["endpoint_semantics"].pop("positive_class_semantics")
            with self.assertRaisesRegex(ArtifactManifestError, "positive_class_semantics"):
                validate_model_artifact_manifest(
                    invalid,
                    manifest_dir=root,
                    registered_sha256s={digest},
                )

    def test_payload_must_contain_exactly_one_registered_estimator_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact = root / "bad.joblib"
            joblib.dump({"model": object(), "estimator": object()}, artifact)
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = {
                "artifact_origin": "project_trained",
                "format": "joblib",
                "model_path": artifact.name,
                "sha256": digest,
                "feature_schema": "fixture_v1",
                "seed": 1,
                "training_input_hash": "3" * 64,
                "endpoint_semantics": endpoint_semantics(),
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactManifestError, "exactly one"):
                load_model_artifact(manifest_path, registered_sha256s={digest})


if __name__ == "__main__":
    unittest.main()
