"""Auditable, resumable ESM2 embeddings; no downloads or silent truncation.

Extraction requires an existing PyTorch/Transformers environment. Legacy caches
can be audited using NumPy alone, but remain excluded from primary comparisons
when their model/pooling provenance has not been established.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

SCHEMA = "peptide_modeling_esm_v1"
POOLING = "last_hidden_state_residue_mean_excluding_all_special_and_padding_tokens"
ESM_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
INVALID_KINDS = {
    "multichain",
    "ambiguous_sequences",
    "partial_parent_projection",
    "unparsed_or_missing",
    "missing",
    "unparsed_notation",
}


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_hash(contract):
    return hashlib.sha256(stable(contract).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_tsv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sequence_error(sequence, kind="", max_length=1022):
    if not sequence:
        return "empty_sequence"
    if kind in INVALID_KINDS:
        return "sequence_kind_not_encodable"
    if len(sequence) > max_length:
        return "sequence_exceeds_max_length"
    if not set(sequence) <= ESM_LETTERS:
        return "unsupported_sequence_characters_no_automatic_projection"
    return ""


def prepare_sequences(records, max_length=1022):
    """Cache exact sequence strings once; retain every input-to-cache mapping."""
    members = []
    explicit_keys = {}
    sequence_kinds = {}
    for index, record in enumerate(records, 2):
        seq = str(record.get("sequence", record.get("key", "")) or "").strip()
        kind = str(record.get("sequence_kind", "") or "").strip()
        key = str(record.get("key", "") or "").strip()
        if key:
            if key in explicit_keys and explicit_keys[key] != seq:
                raise ValueError("conflicting_input_key:" + key)
            explicit_keys[key] = seq
        if seq:
            sequence_kinds.setdefault(seq, set()).add(kind)
        members.append(
            {
                "input_row": index,
                "record_id": record.get("record_id", ""),
                "key": seq,
                "sequence_kind": kind,
            }
        )
    sequences = sorted(sequence_kinds)
    if not sequences:
        raise ValueError("no_nonempty_sequences")
    lookup = {seq: i for i, seq in enumerate(sequences)}
    errors = {}
    for seq, kinds in sequence_kinds.items():
        # A sound single-chain source representation can support an identical
        # sequence that is also present in a review-only source row. The review
        # row itself retains its non-encodable status in input_members.tsv.
        encodable_kinds = sorted(kinds - INVALID_KINDS)
        kind = encodable_kinds[0] if encodable_kinds else sorted(kinds)[0]
        error = sequence_error(seq, kind, max_length)
        if error:
            errors[lookup[seq]] = error
    for row in members:
        row["row_idx"] = lookup.get(row["key"], -1)
        row["input_error"] = sequence_error(row["key"], row["sequence_kind"], max_length)
    return {"sequences": sequences, "members": members, "errors": errors}


def read_input(path, max_length=1022):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not ({"sequence", "key"} & set(reader.fieldnames)):
            raise ValueError("input_requires_sequence_or_key_tsv_column")
        return prepare_sequences(reader, max_length)


def mean_pool_numpy(hidden, attention_mask, special_tokens_mask):
    hidden = np.asarray(hidden, dtype=np.float32)
    valid = np.asarray(attention_mask, dtype=bool) & ~np.asarray(special_tokens_mask, dtype=bool)
    if hidden.ndim != 3 or hidden.shape[:2] != valid.shape:
        raise ValueError("pooling_shape_mismatch")
    counts = valid.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("no_residue_tokens")
    pooled = (hidden * valid[..., None]).sum(axis=1) / counts[:, None]
    if not np.isfinite(pooled).all():
        raise ValueError("nonfinite_embedding")
    return pooled.astype(np.float32)


def check_resume_contract(output_dir, contract, resume):
    output_dir = Path(output_dir)
    checkpoint = output_dir / "checkpoint.json"
    if checkpoint.exists():
        if not resume:
            raise ValueError("output_exists_use_resume_or_new_directory")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("contract_hash") != contract_hash(contract):
            raise ValueError("checkpoint_contract_mismatch")
        return state
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("nonempty_output_without_checkpoint")
    return None


def _asset_candidate(payload):
    if not isinstance(payload, dict):
        raise ValueError("asset_receipt_requires_json_object")
    candidate = payload.get("model_assets", payload)
    if not isinstance(candidate, dict):
        raise ValueError("asset_receipt_model_assets_requires_object")
    return candidate


def _asset_receipt_error(candidate, snapshot, metadata):
    required = {"path", "revision", "metadata", "sha256"}
    if not required <= set(candidate):
        return "asset_receipt_missing_required_keys"
    if candidate["path"] != str(snapshot):
        return "asset_receipt_path_mismatch"
    if candidate["revision"] != snapshot.name:
        return "asset_receipt_revision_mismatch"
    if not isinstance(candidate["metadata"], dict) or candidate["metadata"] != metadata:
        return "asset_receipt_metadata_mismatch"
    hashes = candidate["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(metadata):
        return "asset_receipt_hash_keys_mismatch"
    if any(
        not isinstance(v, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", v) for v in hashes.values()
    ):
        return "asset_receipt_invalid_sha256"
    return ""


def snapshot_receipt(snapshot, previous=None, asset_receipt=None):
    """Reuse verified asset hashes only when every observed file still matches.

    ``asset_receipt`` accepts a prior manifest/checkpoint path or a raw assets
    object. Invalid or changed receipts trigger fresh hashes and leave an audit
    reason. This does not reuse embeddings, sequence membership, or run contracts.
    """
    snapshot = Path(snapshot).resolve()
    files = sorted(p for p in snapshot.iterdir() if p.is_file())
    if not files or not (snapshot / "config.json").is_file():
        raise ValueError("snapshot_missing_config")
    if not any(p.name.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("snapshot_missing_weights")
    if not (snapshot / "vocab.txt").is_file():
        raise ValueError("snapshot_missing_vocab")
    metadata = {
        p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files
    }
    candidates = []
    rejected = []
    if previous is not None:
        try:
            candidates.append(
                (
                    _asset_candidate(previous),
                    "checkpoint.model_assets",
                    hashlib.sha256(stable(previous).encode()).hexdigest(),
                )
            )
        except ValueError as exc:
            rejected.append({"source": "checkpoint.model_assets", "reason": str(exc)})
    if asset_receipt is not None:
        source = (
            str(asset_receipt) if not isinstance(asset_receipt, dict) else "asset_receipt_object"
        )
        try:
            if isinstance(asset_receipt, dict):
                payload = asset_receipt
                source_sha = hashlib.sha256(stable(payload).encode()).hexdigest()
            else:
                receipt_path = Path(asset_receipt)
                receipt_bytes = receipt_path.read_bytes()
                payload = json.loads(receipt_bytes.decode("utf-8"))
                source_sha = hashlib.sha256(receipt_bytes).hexdigest()
            candidates.append((_asset_candidate(payload), source, source_sha))
        except (OSError, ValueError, TypeError) as exc:
            rejected.append(
                {"source": source, "reason": f"asset_receipt_unreadable:{type(exc).__name__}"}
            )
    reusable = []
    for candidate, source, source_sha in candidates:
        error = _asset_receipt_error(candidate, snapshot, metadata)
        if error:
            rejected.append({"source": source, "reason": error})
        else:
            reusable.append((candidate, source, source_sha))
    if reusable:
        candidate, source, source_sha = reusable[0]
        hashes = {name: value.lower() for name, value in candidate["sha256"].items()}
        mode = (
            "own_checkpoint_receipt"
            if source == "checkpoint.model_assets"
            else "cached_asset_receipt"
        )
    else:
        hashes = {p.name: sha_file(p) for p in files}
        source, source_sha, mode = "", "", "fresh_file_hashes"
        after = {
            p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files
        }
        if after != metadata:
            raise ValueError("snapshot_assets_changed_during_hash")
    return {
        "path": str(snapshot),
        "revision": snapshot.name,
        "metadata": metadata,
        "sha256": hashes,
        "hash_mode": mode,
        "cached_asset_source": source,
        "cached_asset_source_sha": source_sha,
        "observedstat": metadata,
        "asset_receipt_rejections": rejected,
    }


def group_manifest(shape):
    return {
        "matrix": "esm2.npy",
        "valid": "esm2_valid.npy",
        "index": "esm2_index.tsv",
        "key_kind": "sequence",
        "feature_names": [f"esm2_{i}" for i in range(shape[1])],
        "dtype": "float16",
        "shape": list(shape),
    }


def routes():
    return {
        "esm_sequence": ["sequence", "esm2"],
        "fusion_esm": ["morgan", "descriptors", "graph_modifications", "sequence", "esm2"],
    }


def write_index(output_dir, sequences):
    write_tsv(
        Path(output_dir) / "esm2_index.tsv",
        ["key", "row_idx"],
        ({"key": seq, "row_idx": i} for i, seq in enumerate(sequences)),
    )


def import_legacy_cache(npz_path, index_path, output_dir, expected_dim=1280, max_length=1022):
    """Audit a historical numeric cache without upgrading its unknown provenance."""
    with np.load(npz_path, allow_pickle=False) as archive:
        if "embeddings" not in archive.files:
            raise ValueError("legacy_missing_embeddings")
        matrix = archive["embeddings"]
    if matrix.ndim != 2 or matrix.shape[1] != expected_dim:
        raise ValueError("legacy_dimension_mismatch")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("legacy_nonfinite_or_nonnumeric")
    with Path(index_path).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    if len(records) != len(matrix):
        raise ValueError("legacy_index_row_count_mismatch")
    sequences = [None] * len(matrix)
    seen_keys = set()
    for row in records:
        key = str(row.get("sequence", row.get("key", ""))).strip()
        idx = int(row["row_idx"])
        if key in seen_keys:
            raise ValueError("legacy_duplicate_sequence_key")
        if not key or idx < 0 or idx >= len(matrix) or sequences[idx] is not None:
            raise ValueError("legacy_invalid_index")
        sequences[idx] = key
        seen_keys.add(key)
    converted = matrix.astype(np.float16)
    if not np.isfinite(converted).all():
        raise ValueError("legacy_float16_overflow")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("legacy_output_must_be_new")
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = np.array([not sequence_error(seq, max_length=max_length) for seq in sequences])
    valid &= np.any(converted != 0, axis=1)
    np.save(output_dir / "esm2.npy", converted)
    np.save(output_dir / "esm2_valid.npy", valid)
    write_index(output_dir, sequences)
    manifest = {
        "schema": SCHEMA,
        "status": "legacy_audited",
        "provenance_status": "legacy_unverified_model_provenance",
        "eligible_for_primary_comparison": False,
        "model": "historically_labelled_esm2_650m_not_reestablished_by_numeric_audit",
        "pooling": "historically_labelled_mean_pool_not_independently_reestablished",
        "inputs": {str(npz_path): sha_file(npz_path), str(index_path): sha_file(index_path)},
        "counts": {"sequences": len(sequences), "valid": int(valid.sum())},
        "groups": {"esm2": group_manifest(converted.shape)},
        "routes": routes(),
    }
    write_json(output_dir / "esm2_manifest.json", manifest)
    return manifest


def extract_embeddings(
    input_sequences,
    output_dir,
    model_snapshot,
    device="cuda:0",
    batch_tokens=4096,
    max_length=1022,
    expected_dim=1280,
    resume=False,
    max_attempts=2,
    asset_receipt=None,
):
    """Extract only from local weights, with explicit review and failure masks."""
    if batch_tokens < 1 or max_attempts < 1:
        raise ValueError("invalid_batch_or_attempt_budget")
    output_dir = Path(output_dir)
    prepared = read_input(input_sequences, max_length)
    sequences = prepared["sequences"]
    previous_checkpoint = output_dir / "checkpoint.json"
    previous = (
        json.loads(previous_checkpoint.read_text(encoding="utf-8"))
        if (resume and previous_checkpoint.exists())
        else {}
    )
    receipt = snapshot_receipt(model_snapshot, previous.get("model_assets"), asset_receipt)
    cfg = json.loads((Path(model_snapshot) / "config.json").read_text(encoding="utf-8"))
    if int(cfg.get("hidden_size", 0)) != expected_dim:
        raise ValueError("model_hidden_size_mismatch")
    contract = {
        "schema": SCHEMA,
        "input_sha256": sha_file(input_sequences),
        "model_assets": receipt["sha256"],
        "pooling": POOLING,
        "max_length": max_length,
        "expected_dim": expected_dim,
        "dtype": "float16",
        "script_sha256": sha_file(__file__),
        "max_attempts": max_attempts,
    }
    existing = check_resume_contract(output_dir, contract, resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (len(sequences), expected_dim)
    mode = "r+" if existing else "w+"
    matrix = np.lib.format.open_memmap(
        output_dir / "esm2.npy", mode=mode, dtype=np.float16, shape=shape
    )
    valid = np.lib.format.open_memmap(
        output_dir / "esm2_valid.npy", mode=mode, dtype=bool, shape=(len(sequences),)
    )
    states = np.lib.format.open_memmap(
        output_dir / "esm2_state.npy", mode=mode, dtype=np.uint8, shape=(len(sequences),)
    )
    attempts = np.lib.format.open_memmap(
        output_dir / "esm2_attempts.npy", mode=mode, dtype=np.uint16, shape=(len(sequences),)
    )
    try:
        errors = dict(existing.get("errors", {})) if existing else {}
        if not existing:
            matrix[:] = 0
            valid[:] = False
            states[:] = 0
            attempts[:] = 0
            write_index(output_dir, sequences)
            write_tsv(
                output_dir / "input_members.tsv",
                ["input_row", "record_id", "key", "sequence_kind", "row_idx", "input_error"],
                prepared["members"],
            )
            for idx, error in prepared["errors"].items():
                states[idx] = 2
                errors[str(idx)] = error
        if matrix.shape != shape or matrix.dtype != np.float16 or valid.shape != (len(sequences),):
            raise ValueError("checkpoint_array_shape_mismatch")
        if valid.dtype != bool or states.dtype != np.uint8 or attempts.dtype != np.uint16:
            raise ValueError("checkpoint_array_dtype_mismatch")
        if not np.array_equal(valid, states == 1):
            raise ValueError("checkpoint_state_validity_mismatch")
        if not np.isfinite(matrix[valid]).all():
            raise ValueError("checkpoint_completed_embedding_nonfinite")
        started = time.time()

        def checkpoint():
            for array in (matrix, valid, states, attempts):
                array.flush()
            state = {
                "schema": SCHEMA,
                "contract": contract,
                "contract_hash": contract_hash(contract),
                "model_assets": receipt,
                "counts": {
                    "sequences": len(sequences),
                    "valid": int(valid.sum()),
                    "review": int((states == 2).sum()),
                    "failed": int((states == 3).sum()),
                    "pending": int((states == 0).sum()),
                },
                "errors": errors,
                "last_saved_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_json(previous_checkpoint, state)
            return state

        checkpoint()
        pending = [
            i for i in range(len(sequences)) if states[i] in (0, 3) and attempts[i] < max_attempts
        ]
        if pending:
            # Optional imports stay inside the actual extraction path.
            import torch
            from transformers import AutoModel, AutoTokenizer

            target_device = torch.device(device)
            if target_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("requested_cuda_unavailable_no_cpu_fallback")
            dtype = torch.float16 if target_device.type == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True)
            model = AutoModel.from_pretrained(
                str(model_snapshot),
                local_files_only=True,
                torch_dtype=dtype,
                trust_remote_code=False,
            )
            model.eval().to(target_device)

            def infer(indices):
                try:
                    encoded = tokenizer(
                        [sequences[i] for i in indices],
                        return_tensors="pt",
                        padding=True,
                        truncation=False,
                        return_special_tokens_mask=True,
                    )
                    specials = encoded.pop("special_tokens_mask").to(target_device).bool()
                    encoded = {k: v.to(target_device) for k, v in encoded.items()}
                    residue_mask = encoded["attention_mask"].bool() & ~specials
                    lengths = residue_mask.sum(dim=1)
                    if [int(x) for x in lengths.cpu().tolist()] != [
                        len(sequences[i]) for i in indices
                    ]:
                        raise ValueError("tokenizer_residue_count_mismatch")
                    with torch.inference_mode():
                        hidden = model(**encoded).last_hidden_state
                        pooled = (hidden.float() * residue_mask[..., None]).sum(dim=1)
                        pooled /= lengths[:, None]
                        vectors = pooled.cpu().numpy().astype(np.float16)
                    if not np.isfinite(vectors).all() or np.any(~np.any(vectors != 0, axis=1)):
                        raise ValueError("nonfinite_or_allzero_embedding")
                    for idx, vector in zip(indices, vectors, strict=True):
                        attempts[idx] += 1
                        matrix[idx] = vector
                        valid[idx] = True
                        states[idx] = 1
                        errors.pop(str(idx), None)
                except Exception as exc:
                    if target_device.type == "cuda":
                        torch.cuda.empty_cache()
                    if len(indices) > 1:
                        middle = len(indices) // 2
                        infer(indices[:middle])
                        infer(indices[middle:])
                        return
                    idx = indices[0]
                    attempts[idx] += 1
                    states[idx] = 3
                    valid[idx] = False
                    errors[str(idx)] = f"{type(exc).__name__}:{str(exc)[:500]}"
                    if attempts[idx] < max_attempts:
                        infer(indices)
                checkpoint()

            pending.sort(key=lambda i: (len(sequences[i]), sequences[i]))
            batch = []
            largest = 0
            for idx in pending:
                proposed = max(largest, len(sequences[idx]) + 2)
                if batch and (proposed * (len(batch) + 1) > batch_tokens or len(batch) >= 32):
                    infer(batch)
                    print(
                        f"ESM2 {int(valid.sum())}/{len(sequences)} valid "
                        f"({time.time() - started:.1f}s)",
                        flush=True,
                    )
                    batch, largest = [], 0
                batch.append(idx)
                largest = max(largest, len(sequences[idx]) + 2)
            if batch:
                infer(batch)
            del model
        final = checkpoint()
        manifest = {
            "schema": SCHEMA,
            "status": "complete_with_failures" if final["counts"]["failed"] else "complete",
            "provenance_status": "local_snapshot_hashed_fresh_extraction",
            "eligible_for_primary_comparison": bool(valid.any()),
            "contract": contract,
            "model_assets": receipt,
            "counts": final["counts"],
            "device": device,
            "groups": {"esm2": group_manifest(shape)},
            "routes": routes(),
            "elapsed_seconds_this_invocation": round(time.time() - started, 3),
            "interpretation": "parent sequence embeddings do not encode chemical modifications",
        }
        manifest["outputs"] = {
            name: {
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha_file(output_dir / name),
            }
            for name in (
                "esm2.npy",
                "esm2_valid.npy",
                "esm2_index.tsv",
                "input_members.tsv",
                "checkpoint.json",
            )
        }
        write_json(output_dir / "esm2_manifest.json", manifest)
        return manifest
    finally:
        for array in (matrix, valid, states, attempts):
            array.flush()
            array._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-sequences", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--expected-dim", type=int, default=1280)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--asset-receipt",
        type=Path,
        help="Prior fresh manifest/checkpoint model_assets; stat changes force rehash",
    )
    parser.add_argument("--import-legacy-npz", type=Path)
    parser.add_argument("--legacy-index", type=Path)
    args = parser.parse_args()
    if args.import_legacy_npz:
        if not args.legacy_index or args.model_snapshot or args.asset_receipt:
            parser.error("legacy import requires --legacy-index and excludes model/asset options")
        result = import_legacy_cache(
            args.import_legacy_npz,
            args.legacy_index,
            args.output_dir,
            args.expected_dim,
            args.max_length,
        )
    else:
        if not args.input_sequences or not args.model_snapshot:
            parser.error("extraction requires --input-sequences and --model-snapshot")
        result = extract_embeddings(
            args.input_sequences,
            args.output_dir,
            args.model_snapshot,
            args.device,
            args.batch_tokens,
            args.max_length,
            args.expected_dim,
            args.resume,
            args.max_attempts,
            args.asset_receipt,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    csv.field_size_limit(10000000)
    main()
