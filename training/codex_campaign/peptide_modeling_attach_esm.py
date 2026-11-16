"""Attach verified full-universe ESM embeddings without changing CPU feature banks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
from peptide_modeling_esm import INVALID_KINDS, sequence_error, sha_file, write_json
from peptide_modeling_esm import contract_hash as signature
from peptide_modeling_esm import write_tsv as write_tsv

VERSION = "peptide_full_esm_attachment_v1"
REQUIRED_OUTPUTS = {
    "esm2.npy",
    "esm2_valid.npy",
    "esm2_index.tsv",
    "input_members.tsv",
    "checkpoint.json",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def safe_path(root, relative):
    # Manifests move between Linux and Windows; only relative artifact paths travel.
    relative = str(relative).replace("\\", "/")
    if not relative or relative.startswith("/") or ":" in relative:
        raise ValueError("Invalid relative artifact path: " + relative)
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes cache: " + relative)
    return result


def verify_artifacts(cache, hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing bank artifact hashes")
    for relative, expected in hashes.items():
        path = safe_path(cache, relative)
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError("Artifact hash mismatch: " + relative)


def index_keys(path, count=None):
    data = list(rows(path))
    if count is None:
        count = len(data)
    keys = [None] * count
    seen = set()
    for row in data:
        key = row.get("key", "")
        index = int(row["row_idx"])
        if not key or key in seen or not 0 <= index < count or keys[index] is not None:
            raise ValueError("Duplicate key or invalid feature index")
        keys[index] = key
        seen.add(key)
    if len(data) != count or any(key is None for key in keys):
        raise ValueError("Incomplete feature index")
    return keys


def source_rows_receipt(run_dir, cache, original):
    formal = run_dir / "feature_rows_index.tsv"
    source = formal if formal.is_file() else run_dir / "source_feature_rows_index.tsv"
    if not source.is_file():
        raise ValueError("Missing source feature_rows_index.tsv")
    labels = [row["record_id"] for row in rows(source)]
    record_path = original["record_index"]
    if record_path not in original["artifact_hashes"]:
        raise ValueError("Record index lacks artifact hash")
    records = [row["record_id"] for row in rows(safe_path(cache, record_path))]
    if len(set(labels)) != len(labels) or len(set(records)) != len(records):
        raise ValueError("Duplicate label or bank record IDs")
    if set(labels) != set(records):
        raise ValueError("Source feature rows differ from bank record universe")
    return source, sha_file(source), len(labels)


def validate_fresh(esm_dir, sequence_keys, expected_dim):
    manifest_path = esm_dir / "esm2_manifest.json"
    fresh = read_json(manifest_path)
    if fresh.get("status") != "complete":
        raise ValueError("Fresh ESM extraction is not complete")
    if fresh.get("provenance_status") != "local_snapshot_hashed_fresh_extraction":
        raise ValueError("Unverified ESM provenance")
    outputs = fresh.get("outputs", {})
    if not REQUIRED_OUTPUTS <= set(outputs):
        raise ValueError("Missing hashed ESM outputs")
    for relative, record in outputs.items():
        path = safe_path(esm_dir, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("ESM artifact size/hash mismatch: " + relative)
        if sha_file(path) != record["sha256"]:
            raise ValueError("ESM artifact hash mismatch: " + relative)
    contract = fresh["contract"]
    assets = fresh.get("model_assets", {}).get("sha256", {})
    if assets != contract.get("model_assets") or not {"config.json", "vocab.txt"} <= set(assets):
        raise ValueError("ESM model asset contract mismatch")
    if not any(name.endswith((".safetensors", ".bin")) for name in assets):
        raise ValueError("ESM model weight hashes missing")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
        for value in assets.values()
    ):
        raise ValueError("Invalid ESM model asset SHA256")
    checkpoint = read_json(esm_dir / "checkpoint.json")
    if checkpoint.get("contract") != contract or checkpoint.get("contract_hash") != signature(
        contract
    ):
        raise ValueError("ESM checkpoint contract hash mismatch")
    if "model_assets" in checkpoint and checkpoint["model_assets"].get("sha256") != assets:
        raise ValueError("ESM checkpoint model assets mismatch")
    group = fresh["groups"]["esm2"]
    if any(
        group[field] != name
        for field, name in (
            ("matrix", "esm2.npy"),
            ("valid", "esm2_valid.npy"),
            ("index", "esm2_index.tsv"),
        )
    ):
        raise ValueError("Unexpected ESM group artifact paths")
    if group.get("key_kind") != "sequence" or contract.get("expected_dim") != expected_dim:
        raise ValueError("ESM group dimension/key contract mismatch")
    matrix = np.load(esm_dir / group["matrix"], mmap_mode="r", allow_pickle=False)
    valid = np.load(esm_dir / group["valid"], mmap_mode="r", allow_pickle=False)
    try:
        if matrix.ndim != 2 or matrix.shape != (len(sequence_keys), expected_dim):
            raise ValueError("ESM shape differs from sequence universe/dimension")
        if list(matrix.shape) != group["shape"] or str(matrix.dtype) != group["dtype"]:
            raise ValueError("ESM matrix shape/dtype differs from manifest")
        if matrix.dtype != np.float16 or contract.get("dtype") != "float16":
            raise ValueError("ESM matrix dtype must be float16")
        if len(group.get("feature_names", [])) != expected_dim:
            raise ValueError("ESM feature names dimension mismatch")
        if valid.dtype != bool or valid.shape != (len(sequence_keys),):
            raise ValueError("ESM validity mask shape/dtype mismatch")
        keys = index_keys(esm_dir / group["index"], len(matrix))
        if set(keys) != set(sequence_keys):
            raise ValueError("Fresh ESM sequence key universe mismatch")
        if not np.isfinite(matrix[valid]).all() or np.any(~np.any(matrix[valid] != 0, axis=1)):
            raise ValueError("Valid ESM contains nonfinite/allzero embeddings")
        counts = {
            "sequences": len(keys),
            "valid": int(valid.sum()),
            "review": int((~valid).sum()),
            "failed": 0,
            "pending": 0,
        }
        if fresh.get("counts") != counts or checkpoint.get("counts") != counts:
            raise ValueError("ESM mask/checkpoint/manifest count mismatch")
        if fresh.get("eligible_for_primary_comparison") != bool(valid.any()):
            raise ValueError("ESM primary comparison eligibility mismatch")
        max_length = contract.get("max_length")
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("Invalid ESM maximum sequence length")
        member_errors = {key: [] for key in keys}
        member_kinds = {key: set() for key in keys}
        for member in rows(esm_dir / "input_members.tsv"):
            key = member.get("key", "")
            index = int(member["row_idx"])
            if key not in member_errors or not 0 <= index < len(keys) or keys[index] != key:
                raise ValueError("ESM input member sequence universe/index mismatch")
            error = sequence_error(key, member.get("sequence_kind", ""), max_length)
            if member.get("input_error", "") != error:
                raise ValueError("Unjustified ESM review member error")
            member_errors[key].append(error)
            member_kinds[key].add(member.get("sequence_kind", ""))
        expected_errors = {}
        long_review = 0
        for index, key in enumerate(keys):
            errors = member_errors[key]
            if not errors:
                raise ValueError("Missing ESM input member evidence")
            reviewed = all(errors)
            if bool(valid[index]) == reviewed:
                raise ValueError("ESM review mask contradicts sequence/member evidence")
            if reviewed:
                # Match the extractor's exact choice across duplicate source members.
                kinds = member_kinds[key]
                sound_kinds = sorted(kinds - INVALID_KINDS)
                kind = sound_kinds[0] if sound_kinds else sorted(kinds)[0]
                error = sequence_error(key, kind, max_length)
                expected_errors[str(index)] = error
                long_review += error == "sequence_exceeds_max_length"
        if checkpoint.get("errors", {}) != expected_errors:
            raise ValueError("ESM review errors disagree with hashed checkpoint")
        return fresh, counts, long_review, sha_file(manifest_path)
    finally:
        matrix._mmap.close()
        valid._mmap.close()


def attach_full_esm(run_dir, esm_dir):
    run_dir, esm_dir = Path(run_dir).resolve(), Path(esm_dir).resolve()
    pointer = run_dir / "features/manifest.json"
    pointer_sha = sha_file(pointer)
    original = read_json(pointer)
    # Resolve the local bank by basename so a copied manifest remains portable.
    cache_name = str(original["cache_dir"]).replace("\\", "/").split("/")[-1]
    old_cache = safe_path(run_dir / "features", cache_name)
    verify_artifacts(old_cache, original.get("artifact_hashes"))
    for group in original["groups"].values():
        if not {group[k] for k in ("matrix", "valid", "index")} <= set(original["artifact_hashes"]):
            raise ValueError("Unhashed original feature group artifacts")
    source, source_sha, record_count = source_rows_receipt(run_dir, old_cache, original)
    sequence_group = original["groups"]["sequence"]
    sequences = index_keys(
        safe_path(old_cache, sequence_group["index"]), sequence_group["shape"][0]
    )
    fresh, counts, long_review, fresh_sha = validate_fresh(
        esm_dir, sequences, original["groups"]["esm2"]["shape"][1]
    )
    previous_attachment = original.get("configuration", {}).get("full_esm_attachment", {})
    if (
        previous_attachment.get("esm_manifest_sha256") == fresh_sha
        and previous_attachment.get("source_feature_rows_sha256") == source_sha
        and previous_attachment.get("implementation_sha256") == sha_file(__file__)
    ):
        return original
    config = {
        "schema": VERSION,
        "old_bank_fingerprint": original["input_fingerprint"],
        "old_bank_pointer_sha256": pointer_sha,
        "old_bank_artifact_hashes": original["artifact_hashes"],
        "esm_manifest_sha256": fresh_sha,
        "esm_outputs": fresh["outputs"],
        "esm_contract": fresh["contract"],
        "source_feature_rows_sha256": source_sha,
        "source_feature_rows_path": str(source),
        "run_dir": str(run_dir),
        "implementation_sha256": sha_file(__file__),
    }
    fingerprint = signature(config)
    destination = run_dir / "features" / fingerprint[:16]
    lock = run_dir / "features/.full_esm_attach.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(fingerprint)
    staging = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex)
    try:
        if sha_file(pointer) != pointer_sha:
            raise ValueError("Feature pointer changed during ESM validation; retry")
        if destination.exists():
            result = read_json(destination / "manifest.json")
            if result.get("input_fingerprint") != fingerprint:
                raise ValueError("Content-addressed ESM bank fingerprint collision")
            verify_artifacts(destination, result["artifact_hashes"])
        else:
            staging.mkdir()
            for relative in original["artifact_hashes"]:
                target = safe_path(staging, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(old_cache, relative), target)
            fresh_relative = "esm2_full/" + fresh_sha[:16]
            fresh_root = staging / fresh_relative
            fresh_root.mkdir(parents=True, exist_ok=True)
            for relative in [*fresh["outputs"], "esm2_manifest.json"]:
                target = safe_path(fresh_root, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(esm_dir, relative), target)
            verify_artifacts(staging, original["artifact_hashes"])
            verify_artifacts(fresh_root, {k: v["sha256"] for k, v in fresh["outputs"].items()})
            if sha_file(fresh_root / "esm2_manifest.json") != fresh_sha:
                raise ValueError("Fresh ESM manifest changed during copy")
            result = copy.deepcopy(original)
            result.update(
                input_fingerprint=fingerprint,
                cache_dir=str(destination),
                manifest_path=str(destination / "manifest.json"),
                cache_reused=False,
            )
            result.setdefault("configuration", {})["full_esm_attachment"] = config
            group = copy.deepcopy(fresh["groups"]["esm2"])
            for field in ("matrix", "valid", "index"):
                group[field] = fresh_relative + "/" + group[field]
            result["groups"]["esm2"] = group
            result.setdefault("coverage", {})["esm2"] = {
                **counts,
                "representations": counts["sequences"],
                "invalid": counts["review"],
                "long_review": long_review,
                "expanded_sequence_universe": len(sequences),
                "missing_expanded_sequences": 0,
                "provenance_status": fresh["provenance_status"],
                "interpretation": "ESM review mask preserves CPU record mappings",
            }
            receipt = {
                "schema": VERSION,
                "status": "complete",
                "configuration": config,
                "old_cache": str(old_cache),
                "new_cache": str(destination),
                "esm_source": str(esm_dir),
                "labeled_records": record_count,
                "coverage": result["coverage"]["esm2"],
                "cpu_artifacts_unchanged": True,
                "routes_unchanged": True,
            }
            write_json(fresh_root / "attachment_receipt.json", receipt)
            result["artifact_hashes"] = {
                path.relative_to(staging).as_posix(): sha_file(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file() and path != staging / "manifest.json"
            }
            write_json(staging / "manifest.json", result)
            staging.rename(destination)
        # Final compare-and-swap check: never overwrite a newer feature publisher.
        if sha_file(pointer) != pointer_sha or sha_file(source) != source_sha:
            raise ValueError("Feature pointer/source rows changed before ESM publication; retry")
        verify_artifacts(old_cache, original["artifact_hashes"])
        write_json(pointer, result)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--esm-dir", required=True, type=Path)
    args = parser.parse_args()
    result = attach_full_esm(args.run_dir, args.esm_dir)
    print(
        json.dumps(
            {
                "cache_dir": result["cache_dir"],
                "input_fingerprint": result["input_fingerprint"],
                "esm_coverage": result["coverage"]["esm2"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    csv.field_size_limit(10000000)
    main()
