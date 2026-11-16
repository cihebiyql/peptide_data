"""Create an independent ESM comparison experiment with frozen CPU outer splits."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRESH_PROVENANCE = "local_snapshot_hashed_fresh_extraction"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_member(root, relative):
    normalized = PurePosixPath(str(relative).replace("\\", "/"))
    if (
        normalized.is_absolute()
        or ".." in normalized.parts
        or any(":" in p for p in normalized.parts)
    ):
        raise ValueError("Unsafe artifact path: " + str(relative))
    result = Path(root).joinpath(*normalized.parts).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes source directory")
    if not result.is_file():
        raise ValueError("Missing artifact: " + str(result))
    return result


def read_index(path, n):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    keys = [r["key"] for r in rows]
    positions = [int(r["row_idx"]) for r in rows]
    if (
        len(keys) != n
        or len(set(keys)) != n
        or not all(keys)
        or sorted(positions) != list(range(n))
    ):
        raise ValueError("Feature index keys/positions are not one-to-one")
    return dict(zip(keys, positions, strict=True))


def verify_cpu_cache(base_run, manifest):
    cache_name = str(manifest["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    cache = (base_run / "features" / cache_name).resolve()
    if not cache.is_dir() or not cache.is_relative_to((base_run / "features").resolve()):
        raise ValueError("CPU feature cache is not inside base_run/features")
    if "esm2" in manifest["groups"]:
        raise ValueError("Base run already contains an ESM group; use the frozen CPU base")
    if not manifest.get("artifact_hashes"):
        raise ValueError("CPU cache has no artifact hashes")
    for relative, expected in manifest["artifact_hashes"].items():
        if sha_file(safe_member(cache, relative)) != expected:
            raise ValueError("CPU feature artifact hash mismatch: " + relative)
    for group in manifest["groups"].values():
        for field in ("matrix", "valid", "index"):
            safe_member(cache, group[field])
    return cache


def verify_esm(esm_dir, manifest):
    if manifest.get("status") != "complete":
        raise ValueError("ESM extraction must be complete without failed/pending rows")
    if manifest.get("provenance_status") != FRESH_PROVENANCE:
        raise ValueError("Legacy or unverified ESM provenance is not admitted")
    if manifest.get("eligible_for_primary_comparison") is not True:
        raise ValueError("ESM is not eligible for the primary comparison")
    contract = manifest.get("contract", {})
    assets = manifest.get("model_assets", {}).get("sha256", {})
    if not assets or assets != contract.get("model_assets"):
        raise ValueError("Fresh extraction model asset hashes are missing or inconsistent")
    if not any(name.endswith((".safetensors", ".bin")) for name in assets):
        raise ValueError("Model weight hash is required")
    if not {"config.json", "vocab.txt"}.issubset(assets):
        raise ValueError("Model configuration and vocabulary hashes are required")
    for asset_hash in assets.values():
        if len(asset_hash) != 64 or any(ch not in "0123456789abcdef" for ch in asset_hash):
            raise ValueError("Malformed model asset SHA256")
    outputs = manifest.get("outputs", {})
    group = manifest.get("groups", {}).get("esm2")
    if not group or group.get("key_kind") != "sequence":
        raise ValueError("ESM sequence feature group is required")
    for field in ("matrix", "valid", "index"):
        if group[field] not in outputs:
            raise ValueError("ESM group artifact is not bound to an output hash")
    for relative, receipt in outputs.items():
        path = safe_member(esm_dir, relative)
        if path.stat().st_size != receipt["bytes"] or sha_file(path) != receipt["sha256"]:
            raise ValueError("ESM output byte/hash mismatch: " + relative)
    matrix = np.load(safe_member(esm_dir, group["matrix"]), mmap_mode="r")
    valid = np.load(safe_member(esm_dir, group["valid"]), mmap_mode="r")
    shape = tuple(group["shape"])
    if matrix.ndim != 2 or matrix.shape != shape or shape[1] != contract["expected_dim"]:
        raise ValueError("ESM shape does not match its extraction contract")
    if matrix.dtype != np.dtype(group["dtype"]) or valid.dtype != np.dtype(bool):
        raise ValueError("ESM dtype/valid mask contract mismatch")
    if valid.shape != (shape[0],) or not valid.all() or shape[0] == 0:
        raise ValueError("Fresh primary ESM comparison requires all source sequences valid")
    if len(group["feature_names"]) != shape[1] or len(set(group["feature_names"])) != shape[1]:
        raise ValueError("ESM feature names do not match the matrix columns")
    for start in range(0, shape[0], 2048):
        block = matrix[start : start + 2048]
        if not np.isfinite(block).all() or np.any(~np.any(block != 0, axis=1)):
            raise ValueError("ESM contains a nonfinite or all-zero valid embedding")
    index = read_index(safe_member(esm_dir, group["index"]), shape[0])
    counts = manifest.get("counts", {})
    if counts.get("sequences") != shape[0] or counts.get("valid") != shape[0]:
        raise ValueError("ESM count receipt does not match arrays/index")
    if any(counts.get(k, 0) for k in ("failed", "pending", "review")):
        raise ValueError("ESM receipt still contains unresolved source sequences")
    return index


def merge_experiment(base_run, esm_dir, output_run):
    """Validate, copy and bind a new experiment; never overwrite the CPU baseline."""
    base_run, esm_dir, output_run = map(
        lambda p: Path(p).resolve(), (base_run, esm_dir, output_run)
    )
    if output_run.exists():
        raise FileExistsError("Experiment output already exists: " + str(output_run))
    cpu_path = base_run / "features/manifest.json"
    esm_path = esm_dir / "esm2_manifest.json"
    cpu, esm = read_json(cpu_path), read_json(esm_path)
    cpu_cache = verify_cpu_cache(base_run, cpu)
    if output_run.is_relative_to(cpu_cache):
        raise ValueError("Experiment cannot be nested inside the frozen CPU cache")
    esm_index = verify_esm(esm_dir, esm)
    sequence = cpu["groups"]["sequence"]
    cpu_sequence_index = read_index(cpu_cache / sequence["index"], sequence["shape"][0])
    if set(cpu_sequence_index) != set(esm_index):
        raise ValueError("ESM sequence coverage differs from the frozen CPU sequence universe")
    task_manifest = read_json(base_run / "task_manifest.json")
    eligible = [t for t in task_manifest["tasks"] if t["status"] == "eligible"]
    if not eligible:
        raise ValueError("No eligible task tables to copy")
    source_files = [
        base_run / name
        for name in (
            "task_manifest.json",
            "task_summary.tsv",
            "deferred_records.tsv",
            "execution_config.json",
        )
    ]
    split_files = []
    for task in eligible:
        source_files.append(safe_member(base_run, task["table_path"]))
        split = safe_member(base_run, f"splits/{task['task_id']}/group_assignment.json")
        split_data = read_json(split)
        if split_data.get("protocol") != "full_task_group_assignment_before_feature_mask":
            raise ValueError("Base outer split was not frozen before feature masking")
        if (
            not split_data.get("input_hash")
            or not split_data.get("train_records")
            or not split_data.get("test_records")
        ):
            raise ValueError("Frozen split receipt is incomplete")
        if set(split_data["train_records"]) & set(split_data["test_records"]):
            raise ValueError("Frozen split has train/test record overlap")
        split_files.append(split)
    original_hashes = {
        str(p.relative_to(base_run)): sha_file(p) for p in source_files + split_files
    }
    cpu_original_hashes = {
        str(p.relative_to(cpu_cache)): sha_file(p) for p in cpu_cache.rglob("*") if p.is_file()
    }
    provenance = {
        "schema": "peptide_esm_merge_v1",
        "cpu_input_fingerprint": cpu["input_fingerprint"],
        "cpu_manifest_sha256": sha_file(cpu_path),
        "esm_manifest_sha256": sha_file(esm_path),
        "esm_contract": esm["contract"],
        "esm_outputs": esm["outputs"],
        "merge_implementation_sha256": sha_file(Path(__file__)),
    }
    fingerprint = signature(provenance)
    staging = output_run.with_name(output_run.name + ".merging")
    if staging.exists():
        raise FileExistsError("Interrupted merge output exists: " + str(staging))
    staging.mkdir(parents=True)
    for path in source_files + split_files:
        destination = staging / path.relative_to(base_run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha_file(destination) != original_hashes[str(path.relative_to(base_run))]:
            raise ValueError("Copied task/split artifact changed")
    cache_relative = Path("features") / fingerprint[:16]
    cache_stage = staging / cache_relative
    shutil.copytree(cpu_cache, cache_stage)
    shutil.copy2(cpu_cache / "manifest.json", cache_stage / "cpu_source_manifest.json")
    esm_stage = cache_stage / "esm2"
    esm_stage.mkdir()
    for relative in [*esm["outputs"], "esm2_manifest.json"]:
        source = safe_member(esm_dir, relative)
        destination = esm_stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha_file(destination) != sha_file(source):
            raise ValueError("Copied ESM artifact changed")
    merged = copy.deepcopy(cpu)
    merged["input_fingerprint"] = fingerprint
    merged["cache_dir"] = str(output_run / cache_relative)
    merged["manifest_path"] = str(output_run / cache_relative / "manifest.json")
    merged["configuration"].update(
        {"cpu_input_fingerprint": cpu["input_fingerprint"], "esm2_merge": provenance}
    )
    group = copy.deepcopy(esm["groups"]["esm2"])
    for field in ("matrix", "valid", "index"):
        group[field] = "esm2/" + group[field]
    merged["groups"]["esm2"] = group
    merged["routes"].update(
        {"esm_sequence": ["sequence", "esm2"], "fusion_esm": [*cpu["routes"]["fusion"], "esm2"]}
    )
    merged.setdefault("coverage", {})["esm2"] = {
        "representations": len(esm_index),
        "valid": len(esm_index),
        "invalid": 0,
        "all_zero_rows": 0,
    }
    merged["cache_reused"] = False
    merged["artifact_hashes"] = {
        str(p.relative_to(cache_stage)).replace("\\", "/"): sha_file(p)
        for p in sorted(cache_stage.rglob("*"))
        if p.is_file() and p != cache_stage / "manifest.json"
    }
    write_json(cache_stage / "manifest.json", merged)
    write_json(staging / "features/manifest.json", merged)
    config = read_json(base_run / "execution_config.json")
    config["run_dir"] = str(output_run)
    config["training"]["routes"] = ["sequence", "structure", "fusion", "esm_sequence", "fusion_esm"]
    config["training"]["n_trials"] = 20
    config.setdefault("resources", {})["max_parallel_tasks"] = 2
    config["resources"]["remote_durable_root"] = (
        "/data/qlyu/peptide_modeling_20261113_v1/experiments/esm2"
    )
    config["resources"]["remote_training_scratch"] = "/tmp/peptide_modeling_qlyu_20261113_v1_es2"
    config["include_full_coverage"] = False
    config["experiment"] = {
        "feature_comparison": "cpu_plus_fresh_esm2",
        "scopes": ["common"],
        "base_run": str(base_run),
        "frozen_outer_splits_reused": True,
    }
    write_json(staging / "config.json", config)
    write_json(staging / "execution_config.json", config)
    unchanged = all(
        sha_file(base_run / rel) == expected for rel, expected in original_hashes.items()
    )
    unchanged &= all(
        sha_file(cpu_cache / rel) == expected for rel, expected in cpu_original_hashes.items()
    )
    if not unchanged:
        raise ValueError("Frozen CPU source changed during merge")
    receipt = {
        "schema": "peptide_esm_experiment_merge_v1",
        "status": "complete",
        "base_run": str(base_run),
        "output_run": str(output_run),
        "provenance": provenance,
        "esm_model_weight_hashes": esm["model_assets"]["sha256"],
        "copied_eligible_tasks": len(eligible),
        "copied_source_hashes": original_hashes,
        "frozen_split_sha256": {str(p.relative_to(base_run)): sha_file(p) for p in split_files},
        "assertions": {
            "cpu_sources_unchanged": unchanged,
            "outer_splits_byte_identical": all(
                sha_file(staging / p.relative_to(base_run)) == sha_file(p) for p in split_files
            ),
            "esm_sequence_universe_equal": set(cpu_sequence_index) == set(esm_index),
        },
        "training_command_argv": [
            "python3",
            "scripts/peptide_modeling_pipeline.py",
            "--config",
            str(output_run / "config.json"),
            "--run-dir",
            str(output_run),
            "--stage",
            "train",
            "--profile",
            "full",
        ],
    }
    write_json(staging / "merge_receipt.json", receipt)
    staging.rename(output_run)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--esm-dir", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    result = merge_experiment(args.base_run, args.esm_dir, args.output_run)
    print(
        json.dumps(
            {"status": "complete", "cache_dir": result["cache_dir"], "routes": result["routes"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
