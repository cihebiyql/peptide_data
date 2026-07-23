"""Persistent, single-process V31 inference runtime for the local web app.

The command-line V31 contract deliberately reloads every artifact on every
invocation.  A web server needs the opposite lifecycle: verify and load the
frozen ESM2 encoder and all endpoint heads once, then serialize GPU forwards.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from peptide_omnipanel.plm import (
    PLM_POOLING,
    _model_artifact_hashes,
    _pool_transformers_esm2_batch,
    _resolve_snapshot_path,
    _resolved_device,
    _resolved_precision,
    _resolved_snapshot_revision,
    _torch_dtype_name,
    build_cache_identity,
    traditional_sequence_features,
)
from peptide_omnipanel.v31_peptide18 import (
    BARE_SEQUENCE_ASSUMPTION,
    ENDPOINT_ORDER,
    V31Peptide18Error,
    _prediction,
    build_multimodal_feature_row,
    canonical_sequence,
    feature_schema_sha256,
    linear_peptide_helm,
    linear_peptide_smiles,
    select_feature_route,
    sha256_file,
)


class PersistentV31Predictor:
    """Load V31 once and return the strict ordered peptide-only 18-panel."""

    def __init__(
        self,
        *,
        bundle_dir: Path,
        model_source: Path,
        device: str = "cuda",
    ) -> None:
        started = time.perf_counter()
        self.bundle_dir = Path(bundle_dir).resolve()
        self.model_source = Path(model_source).resolve()
        self.manifest_path = self.bundle_dir / "bundle_manifest.json"
        if not self.manifest_path.is_file():
            raise V31Peptide18Error(f"bundle manifest is missing: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("endpoint_order") != list(ENDPOINT_ORDER):
            raise V31Peptide18Error("bundle does not contain the exact ordered 18-endpoint panel")
        if self.manifest.get("feature_schema_sha256") != feature_schema_sha256():
            raise V31Peptide18Error("feature schema drift detected")
        if self.manifest.get("validated_model_count") != 0:
            raise V31Peptide18Error("web disclosure contract expects zero validated V31 heads")

        self.estimators: dict[str, Any] = {}
        for endpoint in ENDPOINT_ORDER:
            descriptor = self.manifest["endpoints"][endpoint]
            model_path = self.bundle_dir / descriptor["model_path"]
            if sha256_file(model_path) != descriptor["model_sha256"]:
                raise V31Peptide18Error(f"model SHA mismatch for {endpoint}")
            self.estimators[endpoint] = joblib.load(model_path)

        try:
            import torch
            import transformers
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:  # pragma: no cover - deployment dependency
            raise RuntimeError("persistent V31 runtime requires torch and transformers") from error

        self.torch = torch
        self.transformers_version = str(transformers.__version__)
        self.resolved_device = _resolved_device(device)
        self.resolved_precision = _resolved_precision("auto", self.resolved_device)
        self.torch_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.resolved_precision]
        self.snapshot = _resolve_snapshot_path(
            model_name=self.manifest["esm_model_name"],
            revision=self.manifest["esm_revision"],
            model_source=self.model_source,
            local_files_only=True,
        )
        self.resolved_revision = _resolved_snapshot_revision(
            self.snapshot, self.manifest["esm_revision"]
        )
        self.artifact_hashes = _model_artifact_hashes(self.snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            str(self.snapshot),
            local_files_only=True,
            dtype=self.torch_dtype,
            add_pooling_layer=False,
        )
        self.encoder.eval().to(self.resolved_device)
        config_payload = json.dumps(
            self.encoder.config.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.config_sha256 = hashlib.sha256(config_payload).hexdigest()
        self.max_positions = int(getattr(self.encoder.config, "max_position_embeddings", 0))
        self.hidden_size = int(self.encoder.config.hidden_size)
        if self.hidden_size != 1280:
            raise V31Peptide18Error(f"ESM hidden width drift: {self.hidden_size}")
        self.device_name = (
            torch.cuda.get_device_name(self.resolved_device)
            if self.resolved_device.startswith("cuda")
            else "CPU"
        )
        self._lock = threading.Lock()
        self.loaded_seconds = time.perf_counter() - started

    def runtime_info(self) -> dict[str, Any]:
        return {
            "bundle_id": self.manifest["bundle_id"],
            "endpoint_count": len(ENDPOINT_ORDER),
            "validated_model_count": 0,
            "device": self.resolved_device,
            "device_name": self.device_name,
            "precision": self.resolved_precision,
            "model_name": self.manifest["esm_model_name"],
            "model_revision": self.resolved_revision,
            "model_load_seconds": self.loaded_seconds,
        }

    def _esm_identity(self, sequence: str, observed_tokens: int) -> dict[str, Any]:
        return build_cache_identity(
            model_name=self.manifest["esm_model_name"],
            model_revision=self.resolved_revision,
            requested_model_revision=self.manifest["esm_revision"],
            encoder_config_sha256=self.config_sha256,
            embedding_dim=self.hidden_size,
            sequences=[sequence],
            backend_version=self.transformers_version,
            torch_version=str(self.torch.__version__),
            compute_dtype=_torch_dtype_name(self.resolved_precision),
            max_position_embeddings=self.max_positions,
            max_observed_tokens=observed_tokens,
            **self.artifact_hashes,
            pooling=PLM_POOLING,
        )

    def predict(self, sequence: str) -> dict[str, Any]:
        request_started = time.perf_counter()
        canonical = canonical_sequence(sequence)
        with self._lock:
            mean, _maximum, observed_tokens = _pool_transformers_esm2_batch(
                sequences=[canonical],
                tokenizer=self.tokenizer,
                model=self.encoder,
                torch_module=self.torch,
                resolved_device=self.resolved_device,
                torch_dtype=self.torch_dtype,
                resolved_precision=self.resolved_precision,
                max_positions=self.max_positions,
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
        matrix = row.reshape(1, -1)
        records: list[dict[str, Any]] = []
        for endpoint in ENDPOINT_ORDER:
            descriptor = self.manifest["endpoints"][endpoint]
            features = select_feature_route(matrix, descriptor["feature_route"])
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                raw_value = _prediction(
                    self.estimators[endpoint], descriptor["task_kind"], features
                )
            value = raw_value
            if descriptor["output_transform"] == "clip_0_1":
                value = float(np.clip(value, 0.0, 1.0))
            elif descriptor["output_transform"] == "clip_nonnegative":
                value = max(0.0, value)
            if not math.isfinite(value):
                raise V31Peptide18Error(f"non-finite dynamic prediction for {endpoint}")
            endpoint_warnings = list(descriptor["warnings"])
            if descriptor["modification_sensitive"]:
                endpoint_warnings.append(
                    "bare_sequence_predicts_hypothetical_unmodified_linear_analogue"
                )
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
                    "probability": (
                        value if descriptor["task_kind"] == "classification" else None
                    ),
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
                    "validation_status": descriptor["validation_status"],
                    "research_only": True,
                    "low_confidence": bool(descriptor["low_confidence"]),
                    "insufficient_validation": True,
                    "modification_sensitive": bool(descriptor["modification_sensitive"]),
                    "representation_assumptions": [BARE_SEQUENCE_ASSUMPTION],
                    "prediction_interval": descriptor.get("prediction_interval"),
                    "warnings": list(dict.fromkeys(endpoint_warnings)),
                }
            )
        elapsed = time.perf_counter() - request_started
        return {
            "schema_version": "peptide_omnipanel_v31_peptide18_prediction_v1",
            "scope": "internal_research_only",
            "input": {
                "canonical_sequence": canonical,
                "sequence_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "sequence_length": len(canonical),
                "assumed_helm": helm,
                "assumed_smiles": smiles,
                "assumption": BARE_SEQUENCE_ASSUMPTION,
                "feature_schema_sha256": feature_schema_sha256(),
                "feature_sha256": hashlib.sha256(row.tobytes()).hexdigest(),
                "esm_identity": self._esm_identity(canonical, observed_tokens),
                "esm_execution": {
                    "device": self.resolved_device,
                    "device_name": self.device_name,
                    "precision": self.resolved_precision,
                    "persistent_model": True,
                    "observed_tokens": observed_tokens,
                },
            },
            "model_bundle_id": self.manifest["bundle_id"],
            "bundle_manifest_sha256": sha256_file(self.manifest_path),
            "endpoint_record_count": len(records),
            "dynamic_prediction_count": len(records),
            "validated_prediction_count": 0,
            "status": "pass" if len(records) == 18 else "partial",
            "runtime_seconds": elapsed,
            "endpoints": records,
            "warnings": [
                "internal_research_only",
                "bare_sequence_cannot_encode_real_cyclization_or_modifications",
                "low_evidence_endpoints_are_not_clinically_validated",
            ],
        }
