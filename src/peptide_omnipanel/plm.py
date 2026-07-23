"""Strict, reusable frozen-protein-language-model feature utilities.

The functions in this module deliberately stop at *frozen* ESM2 embeddings.
They make cache provenance and sequence inventory explicit so a development
or hold-out observation cannot silently receive an embedding from a different
encoder revision.  Supervised heads live in the experiment script, where their
split contract can be audited separately.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .features import hashed_character_ngrams, sequence_physicochemical_features

PLM_CACHE_SCHEMA_VERSION = "peptide_omnipanel_plm_cache_v2"
PLM_LEGACY_CACHE_SCHEMA_VERSION = "peptide_omnipanel_plm_cache_v1"
PLM_POOLING = "mean_residue"
PLM_MAX_POOLING = "max_residue"
PLM_SUPPORTED_POOLINGS = frozenset({PLM_POOLING, PLM_MAX_POOLING})
PLM_PRECISION_POLICY_VERSION = "esm2_frozen_precision_v1"
PLM_SEQUENCE_NORMALIZATION_VERSION = "uppercase_trim_v1"
TRADITIONAL_FEATURE_SCHEMA_VERSION = "sequence_physchem29_charhash64_v1"


class PLMCacheError(ValueError):
    """Raised when an embedding cache cannot prove its exact provenance."""


def _normalise_sequences(sequences: Sequence[str]) -> list[str]:
    normalized = [str(sequence).strip().upper() for sequence in sequences]
    if any(not sequence for sequence in normalized):
        raise PLMCacheError("PLM cache sequences must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise PLMCacheError("PLM cache sequences must be unique")
    if normalized != sorted(normalized):
        raise PLMCacheError("PLM cache sequences must be lexicographically sorted")
    return normalized


def sequence_inventory_sha256(sequences: Sequence[str]) -> str:
    """Hash a sorted unique sequence inventory without an ambiguous delimiter."""

    normalized = _normalise_sequences(sequences)
    digest = hashlib.sha256()
    for sequence in normalized:
        encoded = sequence.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def traditional_sequence_features(sequences: Sequence[str]) -> np.ndarray:
    """Return the frozen 29 physicochemical + 64 signed n-gram feature block."""

    rows: list[np.ndarray] = []
    for raw_sequence in sequences:
        sequence = str(raw_sequence).strip().upper()
        physicochemical = sequence_physicochemical_features(sequence)
        ngrams = hashed_character_ngrams(
            sequence,
            modality="sequence",
            n_features=64,
            ngram_range=(1, 3),
        )
        if not physicochemical.available or not ngrams.available:
            raise ValueError(f"traditional feature input is unavailable: {raw_sequence!r}")
        rows.append(np.asarray([*physicochemical.values, *ngrams.values], dtype=np.float32))
    if not rows:
        return np.empty((0, 93), dtype=np.float32)
    matrix = np.stack(rows).astype(np.float32, copy=False)
    if matrix.shape[1] != 93 or not np.isfinite(matrix).all():
        raise RuntimeError("traditional sequence feature matrix contract failed")
    return matrix


def build_cache_identity(
    *,
    model_name: str,
    model_revision: str,
    requested_model_revision: str | None = None,
    encoder_config_sha256: str,
    embedding_dim: int,
    sequences: Sequence[str],
    backend: str = "transformers",
    pooling: str = PLM_POOLING,
    backend_version: str = "unknown",
    torch_version: str = "unknown",
    compute_dtype: str = "float32",
    model_weight_manifest_sha256: str = "unknown",
    tokenizer_sha256: str = "unknown",
    special_tokens_map_sha256: str = "unknown",
    max_position_embeddings: int = 0,
    max_observed_tokens: int = 0,
    truncation_policy: str = "forbidden",
    precision_policy_version: str = PLM_PRECISION_POLICY_VERSION,
    sequence_normalization_version: str = PLM_SEQUENCE_NORMALIZATION_VERSION,
    accepted_alphabet_policy: str = "esm2_tokenizer_v1",
) -> dict[str, Any]:
    """Build the immutable identity checked before a cache can be reused."""

    if not model_name or not model_revision or not encoder_config_sha256:
        raise PLMCacheError("model name, revision, and config SHA256 are required")
    if embedding_dim < 1:
        raise PLMCacheError("embedding_dim must be positive")
    if backend != "transformers":
        raise PLMCacheError(f"unsupported PLM backend: {backend!r}")
    if pooling not in PLM_SUPPORTED_POOLINGS:
        raise PLMCacheError(f"unsupported PLM pooling: {pooling!r}")
    if compute_dtype not in {"float32", "float16", "bfloat16"}:
        raise PLMCacheError(f"unsupported compute dtype: {compute_dtype!r}")
    if truncation_policy != "forbidden":
        raise PLMCacheError("PLM cache contract forbids sequence truncation")
    if max_position_embeddings < 0 or max_observed_tokens < 0:
        raise PLMCacheError("position counts cannot be negative")
    return {
        "schema_version": PLM_CACHE_SCHEMA_VERSION,
        "model_name": str(model_name),
        "model_revision": str(model_revision),
        "requested_model_revision": str(requested_model_revision or model_revision),
        "encoder_config_sha256": str(encoder_config_sha256),
        "embedding_dim": int(embedding_dim),
        "dtype": "float32",
        "compute_dtype": compute_dtype,
        "model_weight_manifest_sha256": str(model_weight_manifest_sha256),
        "tokenizer_sha256": str(tokenizer_sha256),
        "special_tokens_map_sha256": str(special_tokens_map_sha256),
        "backend": backend,
        "backend_version": str(backend_version),
        "torch_version": str(torch_version),
        "python_version": sys.version.split()[0],
        "pooling": pooling,
        "precision_policy_version": str(precision_policy_version),
        "sequence_normalization_version": str(sequence_normalization_version),
        "accepted_alphabet_policy": str(accepted_alphabet_policy),
        "truncation_policy": truncation_policy,
        "max_position_embeddings": int(max_position_embeddings),
        "max_observed_tokens": int(max_observed_tokens),
        "sequence_count": len(_normalise_sequences(sequences)),
        "sequence_inventory_sha256": sequence_inventory_sha256(sequences),
    }


def _cache_metadata(npz: Mapping[str, Any]) -> dict[str, Any]:
    if "identity_json" not in npz:
        raise PLMCacheError("PLM cache does not contain identity_json")
    try:
        payload = str(npz["identity_json"].item())
        parsed = json.loads(payload)
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PLMCacheError("PLM cache has invalid identity_json") from error
    if not isinstance(parsed, dict):
        raise PLMCacheError("PLM cache identity must be an object")
    return parsed


def validate_embedding_cache(
    *,
    sequences: Sequence[str],
    embeddings: np.ndarray,
    identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on malformed arrays or a provenance mismatch."""

    normalized = _normalise_sequences(sequences)
    required_v1 = {
        "schema_version",
        "model_name",
        "model_revision",
        "encoder_config_sha256",
        "embedding_dim",
        "dtype",
        "backend",
        "backend_version",
        "torch_version",
        "python_version",
        "pooling",
        "sequence_count",
        "sequence_inventory_sha256",
    }
    required_v2 = required_v1 | {
        "requested_model_revision",
        "compute_dtype",
        "model_weight_manifest_sha256",
        "tokenizer_sha256",
        "special_tokens_map_sha256",
        "precision_policy_version",
        "sequence_normalization_version",
        "accepted_alphabet_policy",
        "truncation_policy",
        "max_position_embeddings",
        "max_observed_tokens",
    }
    schema_version = identity.get("schema_version")
    if schema_version not in {PLM_CACHE_SCHEMA_VERSION, PLM_LEGACY_CACHE_SCHEMA_VERSION}:
        raise PLMCacheError("PLM cache schema version mismatch")
    required = required_v2 if schema_version == PLM_CACHE_SCHEMA_VERSION else required_v1
    missing = sorted(required - set(identity))
    if missing:
        raise PLMCacheError("PLM cache identity misses: " + ", ".join(missing))
    if identity.get("dtype") != "float32":
        raise PLMCacheError("PLM cache dtype provenance must be float32")
    if identity.get("pooling") not in PLM_SUPPORTED_POOLINGS:
        raise PLMCacheError("PLM cache pooling provenance mismatch")
    if schema_version == PLM_CACHE_SCHEMA_VERSION:
        if identity.get("compute_dtype") not in {"float32", "float16", "bfloat16"}:
            raise PLMCacheError("PLM cache compute dtype provenance mismatch")
        if identity.get("truncation_policy") != "forbidden":
            raise PLMCacheError("PLM cache truncation provenance mismatch")
    if int(identity.get("sequence_count", -1)) != len(normalized):
        raise PLMCacheError("PLM cache sequence count mismatch")
    if identity.get("sequence_inventory_sha256") != sequence_inventory_sha256(normalized):
        raise PLMCacheError("PLM cache sequence inventory mismatch")
    if (
        not isinstance(embeddings, np.ndarray)
        or embeddings.dtype != np.float32
        or embeddings.ndim != 2
        or embeddings.shape != (len(normalized), int(identity["embedding_dim"]))
        or not np.isfinite(embeddings).all()
    ):
        raise PLMCacheError("PLM cache embeddings fail shape/dtype/finiteness contract")
    if expected_identity is not None:
        mismatches = [key for key, value in expected_identity.items() if identity.get(key) != value]
        if mismatches:
            raise PLMCacheError("PLM cache provenance mismatch: " + ", ".join(sorted(mismatches)))


def write_embedding_cache(
    path: Path,
    *,
    sequences: Sequence[str],
    embeddings: np.ndarray,
    identity: Mapping[str, Any],
) -> None:
    """Write one self-describing, validated, compressed embedding cache."""

    normalized = _normalise_sequences(sequences)
    matrix = np.asarray(embeddings)
    validate_embedding_cache(
        sequences=normalized,
        embeddings=matrix,
        identity=identity,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sequences=np.asarray(normalized),
        embeddings=matrix,
        identity_json=np.asarray(
            json.dumps(dict(identity), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        ),
    )


def load_embedding_cache(
    path: Path,
    *,
    sequences: Sequence[str],
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only an exact inventory/provenance match; never subset a cache."""

    normalized = _normalise_sequences(sequences)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "sequences" not in archive or "embeddings" not in archive:
                raise PLMCacheError("PLM cache lacks sequences or embeddings")
            observed = [str(value) for value in archive["sequences"].tolist()]
            if observed != normalized:
                raise PLMCacheError("PLM cache sequence order/inventory mismatch")
            embeddings = np.asarray(archive["embeddings"])
            identity = _cache_metadata(archive)
    except OSError as error:
        raise PLMCacheError(f"cannot read PLM cache: {path}") from error
    validate_embedding_cache(
        sequences=normalized,
        embeddings=embeddings,
        identity=identity,
        expected_identity=expected_identity,
    )
    return embeddings, identity


def _resolved_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolved_precision(requested: str, resolved_device: str) -> str:
    import torch

    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported precision: {requested!r}")
    if requested == "auto":
        if resolved_device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp32"
    if requested in {"bf16", "fp16"} and not resolved_device.startswith("cuda"):
        raise ValueError(f"{requested} precision requires a CUDA device")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("requested bf16 precision is not supported by this CUDA device")
    return requested


def _torch_dtype_name(precision: str) -> str:
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_file_manifest_sha256(paths: Sequence[Path], *, root: Path) -> str:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_snapshot_path(
    *,
    model_name: str,
    revision: str,
    model_source: str | Path | None,
    local_files_only: bool,
) -> Path:
    if model_source is not None:
        path = Path(model_source).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"PLM model source is not a directory: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - transformers dependency supplies it
        raise RuntimeError("huggingface_hub is required to resolve encoder artifacts") from error
    return Path(
        snapshot_download(
            repo_id=model_name,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "*.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.txt",
                "vocabulary.txt",
                "merges.txt",
                "tokenizer.model",
                "spiece.model",
            ],
        )
    ).resolve()


def _resolved_snapshot_revision(snapshot: Path, requested_revision: str) -> str:
    """Return the immutable Hub commit encoded in a snapshot directory."""

    candidate = snapshot.name.lower()
    if len(candidate) == 40 and all(character in "0123456789abcdef" for character in candidate):
        return candidate
    # A user-supplied directory may not use the Hub cache layout.  Its exact
    # byte manifests still make the cache immutable, but a mutable symbolic
    # revision is not an acceptable release identity.
    requested = str(requested_revision).lower()
    if len(requested) == 40 and all(character in "0123456789abcdef" for character in requested):
        return requested
    raise PLMCacheError(
        "PLM snapshot must resolve to an immutable 40-character commit SHA; "
        f"got requested revision {requested_revision!r} at {snapshot}"
    )


def _validate_weight_index(snapshot: Path) -> None:
    for index_path in sorted(snapshot.rglob("*.index.json")):
        if index_path.name not in {
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }:
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise PLMCacheError(f"invalid encoder weight index: {index_path}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise PLMCacheError(f"empty encoder weight index: {index_path}")
        missing = sorted(
            {
                shard
                for shard in weight_map.values()
                if not isinstance(shard, str) or not (index_path.parent / shard).is_file()
            }
        )
        if missing:
            raise PLMCacheError(
                f"encoder weight index references missing shards: {', '.join(map(str, missing))}"
            )


def _model_artifact_hashes(snapshot: Path) -> dict[str, str]:
    _validate_weight_index(snapshot)
    weight_names = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    weight_files = [
        path
        for path in snapshot.rglob("*")
        if path.is_file()
        and (
            path.name in weight_names
            or path.name.endswith(".safetensors")
            or path.name.startswith("pytorch_model")
            and path.name.endswith(".bin")
        )
    ]
    if not weight_files:
        raise PLMCacheError(f"no encoder weight files found under {snapshot}")
    tokenizer_names = {
        "added_tokens.json",
        "tokenizer.model",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocabulary.txt",
        "merges.txt",
    }
    tokenizer_files = [
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name in tokenizer_names
    ]
    if not tokenizer_files:
        raise PLMCacheError(f"no tokenizer artifacts found under {snapshot}")
    special_files = [
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name == "special_tokens_map.json"
    ]
    return {
        "model_weight_manifest_sha256": _canonical_file_manifest_sha256(
            weight_files, root=snapshot
        ),
        "tokenizer_sha256": _canonical_file_manifest_sha256(tokenizer_files, root=snapshot),
        "special_tokens_map_sha256": (
            _canonical_file_manifest_sha256(special_files, root=snapshot)
            if special_files
            else hashlib.sha256(b"absent").hexdigest()
        ),
    }


def _length_bucket_batches(
    sequences: Sequence[str],
    *,
    batch_size: int,
    max_tokens_per_batch: int,
    special_tokens: int = 2,
) -> list[list[int]]:
    """Create deterministic length buckets while preserving recoverable indices."""

    if batch_size < 1 or max_tokens_per_batch < 1 or special_tokens < 0:
        raise ValueError("batch_size/token budget must be positive and special_tokens nonnegative")
    indexed = sorted(enumerate(sequences), key=lambda item: (len(item[1]), item[1], item[0]))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max_tokens = 0
    for index, sequence in indexed:
        token_count = len(sequence) + special_tokens
        if token_count > max_tokens_per_batch:
            raise ValueError(
                f"one sequence requires {token_count} tokens, exceeding batch budget "
                f"{max_tokens_per_batch}"
            )
        prospective_max = max(current_max_tokens, token_count)
        if current and (
            len(current) >= batch_size
            or prospective_max * (len(current) + 1) > max_tokens_per_batch
        ):
            batches.append(current)
            current = []
            current_max_tokens = 0
        current.append(index)
        current_max_tokens = max(current_max_tokens, token_count)
    if current:
        batches.append(current)
    return batches


def inspect_transformers_esm2_identity(
    sequences: Sequence[str],
    *,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    revision: str = "main",
    model_source: str | Path | None = None,
    local_files_only: bool = True,
    device: str = "auto",
    precision: str = "auto",
) -> dict[str, Any]:
    """Read the local encoder config to validate a cache before model loading."""

    normalized = _normalise_sequences(sequences)
    try:
        import torch
        import transformers
        from transformers import AutoConfig
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PLM cache inspection requires the optional dependency group: pip install .[plm]"
        ) from error
    resolved_device = _resolved_device(device)
    resolved_precision = _resolved_precision(precision, resolved_device)
    snapshot = _resolve_snapshot_path(
        model_name=model_name,
        revision=revision,
        model_source=model_source,
        local_files_only=local_files_only,
    )
    resolved_revision = _resolved_snapshot_revision(snapshot, revision)
    config = AutoConfig.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    config_payload = json.dumps(
        config.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    artifact_hashes = _model_artifact_hashes(snapshot)
    return build_cache_identity(
        model_name=model_name,
        model_revision=resolved_revision,
        requested_model_revision=revision,
        encoder_config_sha256=hashlib.sha256(config_payload).hexdigest(),
        embedding_dim=int(config.hidden_size),
        sequences=normalized,
        backend_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
        compute_dtype=_torch_dtype_name(resolved_precision),
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 0)),
        max_observed_tokens=max((len(sequence) + 2 for sequence in normalized), default=0),
        **artifact_hashes,
    )


def _pool_transformers_esm2_batch(
    *,
    sequences: Sequence[str],
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    resolved_device: str,
    torch_dtype: Any,
    resolved_precision: str,
    max_positions: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run one batch in an isolated scope so OOM tensors are promptly released."""

    encoded = tokenizer(
        list(sequences),
        return_tensors="pt",
        padding=True,
        truncation=False,
        return_special_tokens_mask=True,
    )
    observed_tokens = int(encoded["input_ids"].shape[1])
    if max_positions and observed_tokens > max_positions:
        raise PLMCacheError(
            f"tokenized sequence length {observed_tokens} exceeds encoder limit "
            f"{max_positions}; truncation is forbidden"
        )
    special_tokens_mask = encoded.pop("special_tokens_mask").to(resolved_device)
    encoded = {name: value.to(resolved_device) for name, value in encoded.items()}
    autocast_context = (
        torch_module.autocast(device_type="cuda", dtype=torch_dtype)
        if resolved_device.startswith("cuda") and resolved_precision in {"bf16", "fp16"}
        else nullcontext()
    )
    with torch_module.inference_mode(), autocast_context:
        hidden = model(**encoded).last_hidden_state
    residue_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
    denominator = residue_mask.sum(dim=1, keepdim=True)
    if bool((denominator == 0).any().item()):
        raise PLMCacheError("at least one sequence has an empty residue mask")
    mean_vectors = (
        (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / denominator
    ).float().cpu().numpy().astype(np.float32, copy=False)
    masked_hidden = hidden.masked_fill(~residue_mask.unsqueeze(-1), float("-inf"))
    max_vectors = masked_hidden.max(dim=1).values.float().cpu().numpy().astype(
        np.float32, copy=False
    )
    if not np.isfinite(mean_vectors).all() or not np.isfinite(max_vectors).all():
        raise PLMCacheError("ESM2 pooling produced non-finite values")
    return mean_vectors, max_vectors, observed_tokens


def compute_transformers_esm2_mean_max_embeddings(
    sequences: Sequence[str],
    *,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    revision: str = "main",
    model_source: str | Path | None = None,
    batch_size: int = 64,
    min_batch_size: int = 8,
    max_tokens_per_batch: int = 8192,
    device: str = "auto",
    precision: str = "auto",
    local_files_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute residue mean and max frozen ESM2 embeddings from one snapshot.

    Importing ``torch``/``transformers`` is intentionally delayed.  The base
    project can therefore inspect provenance and test cache validation without
    installing optional GPU dependencies.
    """

    normalized = _normalise_sequences(sequences)
    if not normalized:
        raise PLMCacheError("PLM embedding requires at least one sequence")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if min_batch_size < 1 or min_batch_size > batch_size:
        raise ValueError("min_batch_size must be in [1, batch_size]")
    if max_tokens_per_batch < 1:
        raise ValueError("max_tokens_per_batch must be positive")
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PLM embedding requires the optional dependency group: pip install .[plm]"
        ) from error

    resolved_device = _resolved_device(device)
    resolved_precision = _resolved_precision(precision, resolved_device)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[resolved_precision]
    snapshot = _resolve_snapshot_path(
        model_name=model_name,
        revision=revision,
        model_source=model_source,
        local_files_only=local_files_only,
    )
    resolved_revision = _resolved_snapshot_revision(snapshot, revision)
    artifact_hashes = _model_artifact_hashes(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        dtype=torch_dtype,
        add_pooling_layer=False,
    )
    device_memory_before: dict[str, int] = {}
    if resolved_device.startswith("cuda"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved_device)
        device_memory_before = {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }
    try:
        model.eval().to(resolved_device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
            raise RuntimeError(
                "ESM2 model loading exhausted CUDA memory before the embedding preflight"
            ) from error
        raise
    config_payload = json.dumps(
        model.config.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    config_sha256 = hashlib.sha256(config_payload).hexdigest()
    max_positions = int(getattr(model.config, "max_position_embeddings", 0))
    batches = _length_bucket_batches(
        normalized,
        batch_size=batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
    )
    pending = list(batches)
    mean_matrix = np.empty((len(normalized), int(model.config.hidden_size)), dtype=np.float32)
    max_matrix = np.empty((len(normalized), int(model.config.hidden_size)), dtype=np.float32)
    completed = np.zeros(len(normalized), dtype=bool)
    fallback_events: list[dict[str, Any]] = []
    effective_batch_sizes: list[int] = []
    max_observed_tokens = 0
    if resolved_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(resolved_device)
    # A real one-sequence forward proves that model loading, tokenization,
    # precision and both pooling paths work before the full inventory starts.
    preflight_mean, preflight_max, preflight_tokens = _pool_transformers_esm2_batch(
        sequences=[normalized[0]],
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        resolved_device=resolved_device,
        torch_dtype=torch_dtype,
        resolved_precision=resolved_precision,
        max_positions=max_positions,
    )
    if preflight_mean.shape != preflight_max.shape or preflight_mean.shape[0] != 1:
        raise PLMCacheError("ESM2 preflight pooling shape contract failed")
    del preflight_mean, preflight_max
    max_observed_tokens = max(max_observed_tokens, preflight_tokens)
    while pending:
        indices = pending.pop(0)
        batch = [normalized[index] for index in indices]
        try:
            mean_vectors, max_vectors, observed_tokens = _pool_transformers_esm2_batch(
                sequences=batch,
                tokenizer=tokenizer,
                model=model,
                torch_module=torch,
                resolved_device=resolved_device,
                torch_dtype=torch_dtype,
                resolved_precision=resolved_precision,
                max_positions=max_positions,
            )
            max_observed_tokens = max(max_observed_tokens, observed_tokens)
            mean_matrix[np.asarray(indices, dtype=int)] = mean_vectors
            max_matrix[np.asarray(indices, dtype=int)] = max_vectors
            completed[np.asarray(indices, dtype=int)] = True
            effective_batch_sizes.append(len(indices))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_oom = (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
            )
            if not is_oom:
                raise
            if len(indices) <= min_batch_size:
                raise RuntimeError(
                    f"ESM2 embedding exhausted CUDA memory at minimum batch size "
                    f"{min_batch_size}"
                ) from error
            gc.collect()
            if resolved_device.startswith("cuda"):
                with torch.cuda.device(resolved_device):
                    torch.cuda.empty_cache()
            midpoint = len(indices) // 2
            left = indices[:midpoint]
            right = indices[midpoint:]
            if not left or not right:
                raise RuntimeError("cannot split OOM batch further") from error
            fallback_events.append(
                {
                    "failed_batch_size": len(indices),
                    "retry_batch_sizes": [len(left), len(right)],
                    "max_sequence_length": max(len(normalized[index]) for index in indices),
                }
            )
            pending = [left, right, *pending]
    if not bool(completed.all()):
        missing = np.flatnonzero(~completed).tolist()
        raise RuntimeError(f"PLM embedding failed to populate sequence indices: {missing[:10]}")
    identity_common = {
        "model_name": model_name,
        "model_revision": resolved_revision,
        "requested_model_revision": revision,
        "encoder_config_sha256": config_sha256,
        "embedding_dim": int(mean_matrix.shape[1]),
        "sequences": normalized,
        "backend_version": str(transformers.__version__),
        "torch_version": str(torch.__version__),
        "compute_dtype": _torch_dtype_name(resolved_precision),
        "max_position_embeddings": max_positions,
        "max_observed_tokens": max_observed_tokens,
        **artifact_hashes,
    }
    mean_identity = build_cache_identity(
        pooling=PLM_POOLING,
        **identity_common,
    )
    max_identity = build_cache_identity(
        pooling=PLM_MAX_POOLING,
        **identity_common,
    )
    validate_embedding_cache(
        sequences=normalized, embeddings=mean_matrix, identity=mean_identity
    )
    validate_embedding_cache(
        sequences=normalized, embeddings=max_matrix, identity=max_identity
    )
    return (
        mean_matrix,
        max_matrix,
        mean_identity,
        max_identity,
        {
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "snapshot_path": str(snapshot),
            "pooling_outputs": [PLM_POOLING, PLM_MAX_POOLING],
            "preflight_passed": True,
            "preflight_tokens": int(preflight_tokens),
            "device_memory_before": device_memory_before,
            "device": resolved_device,
            "device_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.startswith("cuda")
                else "cpu"
            ),
            "local_files_only": bool(local_files_only),
            "model_max_positions": max_positions,
            "max_observed_tokens": max_observed_tokens,
            "precision": resolved_precision,
            "compute_dtype": _torch_dtype_name(resolved_precision),
            "output_dtype": "float32",
            "requested_batch_size": batch_size,
            "minimum_batch_size": min_batch_size,
            "max_tokens_per_batch": max_tokens_per_batch,
            "effective_batch_sizes": effective_batch_sizes,
            "oom_fallback_count": len(fallback_events),
            "oom_fallback_events": fallback_events,
            "peak_cuda_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(resolved_device))
                if resolved_device.startswith("cuda")
                else 0
            ),
            "peak_cuda_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(resolved_device))
                if resolved_device.startswith("cuda")
                else 0
            ),
        },
    )


def compute_transformers_esm2_mean_embeddings(
    sequences: Sequence[str],
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Backward-compatible mean-only view over the audited paired cache path."""

    mean_matrix, _max_matrix, mean_identity, _max_identity, execution = (
        compute_transformers_esm2_mean_max_embeddings(sequences, **kwargs)
    )
    return mean_matrix, mean_identity, execution


def valid_probability_vector(values: Sequence[float]) -> bool:
    """Small validation helper shared by experiment-level model cards."""

    return all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in values)
