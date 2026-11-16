"""Join frozen feature banks without changing source arrays or inferred chemistry.

The current arrays contain the cytotoxicity bank's sequences and old graph
prefix, twelve newly computed EState/ErG rows, and seven source-exact CLint
graphs. Historical extraction state is retained only under source_snapshots/.
This module does not change labels, scientific task definitions or split groups.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import numpy as np
from peptide_modeling_add_methods import ordered_index
from peptide_modeling_add_smiles_methods import (
    ERG_OPTIONS,
    extract_smiles_methods,
    rdkit_tools,
)
from peptide_modeling_expand_features import (
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts, canonical_key

VERSION = "peptide_unified_frozen_features_v2"
ADDITIONAL = {"estate", "erg"}
SEQUENCE = {"sequence", "sequence_extra", "dpc", "esm2"}
ESM_FIELDS = ("model_assets", "pooling", "max_length", "expected_dim", "dtype", "script_sha256")


def prefix_equal(before, after):
    """Compare numerical payload bytes, including NaN bits, in bounded chunks."""
    if before.dtype != after.dtype or before.shape[1:] != after.shape[1:]:
        return False
    if len(before) > len(after):
        return False
    for start in range(0, len(before), 512):
        stop = min(start + 512, len(before))
        a = np.ascontiguousarray(before[start:stop]).view(np.uint8)
        b = np.ascontiguousarray(after[start:stop]).view(np.uint8)
        if not np.array_equal(a, b):
            return False
    return True


def link_verified(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if sha(source) != expected:
        raise ValueError("Source artifact changed before immutable copy")
    try:
        os.link(source, destination)
    except OSError:
        import shutil

        shutil.copy2(source, destination)
    if sha(destination) != expected:
        raise ValueError("Immutable copied artifact hash differs")


def unique_rows(path, field):
    result = list(rows(path))
    if not result or any(not item.get(field) for item in result):
        raise ValueError("An index needs nonempty identities")
    if len({item[field] for item in result}) != len(result):
        raise ValueError("Duplicate identities in a frozen source index")
    return result


def graph_map(cache, manifest):
    relative = "structure_source_smiles.tsv"
    if relative not in manifest["artifact_hashes"]:
        raise ValueError("Every graph requires a hashed full-source SMILES mapping")
    graph_group = next(
        group
        for group in manifest["groups"].values()
        if group["key_kind"] == "canonical_identity_group"
    )
    indexed = ordered_index(cache / graph_group["index"])
    mapped = unique_rows(cache / relative, "key")
    if len(indexed) != len(mapped):
        raise ValueError("Graph universe and source SMILES differ")
    by_key = {item["key"]: item for item in mapped}
    result = []
    for item in indexed:
        source = by_key.get(item["key"])
        if source is None or (
            "row_idx" in source and int(source["row_idx"]) != int(item["row_idx"])
        ):
            raise ValueError("Source SMILES row positions differ from their graph index")
        if canonical_key(source["canonical_isomeric_smiles"]) != item["key"]:
            raise ValueError("Full isomeric SMILES hash differs from its identity")
        result.append(
            {
                "key": item["key"],
                "row_idx": int(item["row_idx"]),
                "canonical_isomeric_smiles": source["canonical_isomeric_smiles"],
            }
        )
    return result


def source_bank(run):
    run = Path(run).resolve()
    pointer = run / "features/manifest.json"
    manifest = read_json(pointer)
    cache = Path(manifest["cache_dir"])
    _verify_artifacts(manifest)
    if not manifest.get("artifact_hashes"):
        raise ValueError("A source bank must have artifact hashes")
    graphs = graph_map(cache, manifest)
    sequence_index = None
    arrays = {}
    for name, group in manifest["groups"].items():
        kind = group["key_kind"]
        index = ordered_index(safe_path(cache, group["index"]))
        if kind == "canonical_identity_group":
            expected = [item["key"] for item in graphs]
        elif kind == "sequence":
            if sequence_index is None:
                sequence_index = index
            expected = [item["key"] for item in sequence_index]
        else:
            raise ValueError("Unsupported source feature identity kind")
        if [item["key"] for item in index] != expected:
            raise ValueError("Feature index orders differ within one identity universe")
        matrix = np.load(safe_path(cache, group["matrix"]), mmap_mode="r", allow_pickle=False)
        valid = np.load(safe_path(cache, group["valid"]), mmap_mode="r", allow_pickle=False)
        if (
            matrix.ndim != 2
            or list(matrix.shape) != group["shape"]
            or matrix.shape != (len(index), len(group["feature_names"]))
            or str(matrix.dtype) != group["dtype"]
            or valid.dtype != np.bool_
            or valid.shape != (len(index),)
        ):
            raise ValueError("Source matrix, mask and declared schema differ")
        validate_values(matrix, valid, group)
        arrays[name] = (matrix, valid)
    aliases = unique_rows(cache / manifest["record_index"], "record_id")
    if manifest.get("requested_records") != len(aliases):
        raise ValueError("Source record count contradicts its alias table")
    graph_keys = {item["key"] for item in graphs}
    sequence_keys = {item["key"] for item in sequence_index or []}
    if any(
        (item.get("structure_key") and item["structure_key"] not in graph_keys)
        or (item.get("sequence_key") and item["sequence_key"] not in sequence_keys)
        for item in aliases
    ):
        raise ValueError("Source record alias points outside its declared identity universe")
    return {
        "run": run,
        "pointer": pointer,
        "pointer_sha256": sha(pointer),
        "manifest": manifest,
        "cache": cache,
        "graphs": graphs,
        "sequences": sequence_index or [],
        "aliases": aliases,
        "arrays": arrays,
    }


def validate_values(matrix, valid, group):
    for start in range(0, len(matrix), 512):
        block = np.asarray(matrix[start : start + 512])[np.asarray(valid[start : start + 512])]
        if not len(block):
            continue
        if np.isinf(block).any() or np.isnan(block).all(axis=1).any():
            raise ValueError("A valid source row contains infinite or wholly missing features")
        if group.get("allow_partial_missing", False):
            if np.any(np.isnan(block).mean(axis=1) > group.get("max_missing_fraction", 0)):
                raise ValueError("A valid source row exceeds declared missingness")
        elif not np.isfinite(block).all():
            raise ValueError("A valid source row contains undeclared missing features")


def version_values(value, key):
    result = set()
    if isinstance(value, dict):
        if key in value:
            result.add(str(value[key]))
        for child in value.values():
            result.update(version_values(child, key))
    elif isinstance(value, list):
        for child in value:
            result.update(version_values(child, key))
    return result


def group_schema(group):
    return {
        key: value
        for key, value in group.items()
        if key not in {"shape", "matrix", "valid", "index"}
    } | {"width": group["shape"][1]}


def esm_contract(bank):
    group = bank["manifest"]["groups"]["esm2"]
    relative = (Path(group["matrix"]).parent / "esm2_manifest.json").as_posix()
    if relative not in bank["manifest"]["artifact_hashes"]:
        raise ValueError("A sequence bank needs its hashed ESM extraction contract")
    contract = read_json(bank["cache"] / relative)["contract"]
    if any(field not in contract for field in ESM_FIELDS):
        raise ValueError("ESM contract lacks a required model or pooling field")
    if contract["expected_dim"] != group["shape"][1] or contract["dtype"] != group["dtype"]:
        raise ValueError("ESM shape or dtype contradicts the producer contract")
    return {field: contract[field] for field in ESM_FIELDS}


def validate_components(base, cyto, clint):
    b, c, clinical = (bank["manifest"] for bank in (base, cyto, clint))
    structural = {
        name
        for name, group in b["groups"].items()
        if group["key_kind"] == "canonical_identity_group"
    }
    if (
        len(b["groups"]) != 17
        or len(structural) != 13
        or set(b["groups"]) - structural != SEQUENCE
        or set(c["groups"]) != set(b["groups"]) - ADDITIONAL
        or set(clinical["groups"]) != structural
    ):
        raise ValueError("The components do not provide seventeen compatible declared groups")
    for name in c["groups"]:
        if group_schema(b["groups"][name]) != group_schema(c["groups"][name]):
            raise ValueError("Cytotoxicity feature names, dtype or transforms differ")
        for before, after in zip(base["arrays"][name], cyto["arrays"][name], strict=True):
            if not prefix_equal(before, after):
                raise ValueError("A cytotoxicity source changed an old feature prefix")
    for name in structural:
        if group_schema(b["groups"][name]) != group_schema(clinical["groups"][name]):
            raise ValueError("CLint feature names, dtype or transforms differ")
    for key in ("rdkit_version", "numpy_version"):
        versions = [version_values(m["configuration"], key) for m in (b, c, clinical)]
        if any(len(value) != 1 for value in versions):
            raise ValueError("Component algorithm versions are inconsistent")
        if not all(value == versions[0] for value in versions):
            raise ValueError("Component algorithm versions are inconsistent")
    additions = b["configuration"]["additional_smiles_methods"]
    if (
        additions["rdkit_version"] != rdkit_tools()["version"]
        or additions["implementation_sha256"]
        != sha(Path(__file__).with_name("peptide_modeling_add_smiles_methods.py"))
        or additions["erg_options"] != ERG_OPTIONS
    ):
        raise ValueError("The additional graph algorithm differs from its frozen source")
    if esm_contract(base) != esm_contract(cyto):
        raise ValueError("ESM model, tokenizer, pooling or producer implementation differs")
    if (
        clinical["configuration"]["source_feature_fingerprint"] != b["input_fingerprint"]
        or clinical["configuration"]["inherited_algorithm_configuration"] != b["configuration"]
        or c["configuration"]["cytotoxicity_overlay"]["base_fingerprint"]
        != additions["source_fingerprint"]
    ):
        raise ValueError("Component lineage does not bind to the same frozen parent")
    for before, after, label in (
        (base["graphs"], cyto["graphs"], "graph"),
        (base["sequences"], cyto["sequences"], "sequence"),
    ):
        if before != after[: len(before)]:
            raise ValueError(f"The cytotoxicity {label} identity prefix changed")
    if [row["record_id"] for row in base["aliases"]] != [
        row["record_id"] for row in cyto["aliases"]
    ]:
        raise ValueError("The existing labelled record IDs or their order changed")
    changed = {
        before["record_id"]
        for before, after in zip(base["aliases"], cyto["aliases"], strict=True)
        if before != after
    }
    overlay = "cytotoxicity_representation_overlay.tsv"
    if overlay not in c["artifact_hashes"] or changed != {
        row["record_id"] for row in rows(cyto["cache"] / overlay)
    }:
        raise ValueError("A changed record alias is not bound to the source-proven overlay")
    if set(row["key"] for row in cyto["graphs"]) & {row["key"] for row in clint["graphs"]}:
        raise ValueError("New CLint full graph identities overlap existing structures")
    if set(row["record_id"] for row in cyto["aliases"]) & {
        row["record_id"] for row in clint["aliases"]
    }:
        raise ValueError("New CLint records overlap existing labelled IDs")
    if any(not row.get("structure_key") or row.get("sequence_key") for row in clint["aliases"]):
        raise ValueError("CLint records must remain complete-graph-only without parent sequences")
    if len(clint["aliases"]) != len(clint["graphs"]) or {
        row["structure_key"] for row in clint["aliases"]
    } != {row["key"] for row in clint["graphs"]}:
        raise ValueError("Each newly admitted CLint record needs its own full source graph")
    return structural, changed


def append_arrays(path, arrays):
    shape = (sum(len(array) for array in arrays), *arrays[0].shape[1:])
    if any(array.dtype != arrays[0].dtype or array.shape[1:] != shape[1:] for array in arrays):
        raise ValueError("Array append schemas differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(path, mode="w+", dtype=arrays[0].dtype, shape=shape)
    offset = 0
    for array in arrays:
        for start in range(0, len(array), 512):
            stop = min(start + 512, len(array))
            output[offset + start : offset + stop] = array[start:stop]
        if not prefix_equal(array, output[offset : offset + len(array)]):
            raise ValueError("Numerical payload bytes changed during the append")
        offset += len(array)
    output.flush()
    del output


def coverage(cache, group):
    matrix = np.load(cache / group["matrix"], mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / group["valid"], mmap_mode="r", allow_pickle=False)
    validate_values(matrix, valid, group)
    return {
        "representations": len(valid),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "all_zero_rows": int(np.sum(valid & ~np.any(matrix != 0, axis=1))),
    }


def snapshot(bank, cache, label):
    prefix = "source_snapshots/" + label
    source_manifest = cache / prefix / "manifest.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    link_verified(bank["pointer"], source_manifest, bank["pointer_sha256"])
    hashes = {prefix + "/manifest.json": bank["pointer_sha256"]}
    for relative, expected in bank["manifest"]["artifact_hashes"].items():
        target = safe_path(cache, prefix + "/artifacts/" + relative)
        link_verified(safe_path(bank["cache"], relative), target, expected)
        hashes[prefix + "/artifacts/" + relative] = expected
    return hashes


def verify_sources(banks):
    for bank in banks:
        _verify_artifacts(bank["manifest"])
        if sha(bank["pointer"]) != bank["pointer_sha256"]:
            raise ValueError("A frozen source feature pointer changed during unification")


def verify_unified(manifest, include_snapshots=True):
    """Validate runtime assets; historical snapshots are checked once at delivery.

    Runtime route loading uses the existing artifact verifier's small LRU. Keeping
    historical files in a separate signed inventory prevents repeated full-bank
    hashing on every feature-route load.
    """
    _verify_artifacts(manifest)
    if include_snapshots:
        cache = Path(manifest["cache_dir"])
        for relative, expected in manifest.get("source_snapshot_hashes", {}).items():
            if sha(safe_path(cache, relative)) != expected:
                raise ValueError("Historical source snapshot hash mismatch")


def unify_features(base_run, cyto_run, clint_run, run_dir, ancestor_runs=()):
    run_dir = Path(run_dir).resolve()
    sources = [source_bank(run) for run in (base_run, cyto_run, clint_run)]
    ancestors = [source_bank(run) for run in ancestor_runs]
    if any(
        run_dir == bank["run"] or run_dir in bank["run"].parents or bank["run"] in run_dir.parents
        for bank in [*sources, *ancestors]
    ):
        raise ValueError("Unified output must be independent of every immutable source")
    base, cyto, clint = sources
    structural, changed = validate_components(base, cyto, clint)
    registry = read_json(base["run"] / "representation_methods.json")
    registry_artifact = base["manifest"].get("representation_registry_file")
    if (
        registry != base["manifest"].get("representation_methods")
        or registry_artifact not in base["manifest"]["artifact_hashes"]
        or read_json(base["cache"] / registry_artifact) != registry
    ):
        raise ValueError("The current method registry differs from its hashed source artifact")
    methods = [item for item in registry["methods"] if item.get("independent") is True]
    if (
        len(methods) != 13
        or len({item["algorithm_family_id"] for item in methods}) != 13
        or sum(item["source_kind"] == "structure" for item in methods) != 10
    ):
        raise ValueError("The current top-level registry must declare thirteen independent methods")
    for method in methods:
        if (
            method["route"] not in base["manifest"]["routes"]
            or base["manifest"]["routes"][method["route"]] != method["group_names"]
            or sum(base["manifest"]["groups"][name]["shape"][1] for name in method["group_names"])
            != method["dimensions"]
        ):
            raise ValueError("A registered independent method contradicts its feature route")
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "additional_graph_extractor_sha256": sha(
            Path(__file__).with_name("peptide_modeling_add_smiles_methods.py")
        ),
        "sources": {
            name: {
                "fingerprint": bank["manifest"]["input_fingerprint"],
                "pointer_sha256": bank["pointer_sha256"],
            }
            for name, bank in zip(("base13", "cytotoxicity", "clint"), sources, strict=True)
        },
        "ancestor_pointer_sha256": [bank["pointer_sha256"] for bank in ancestors],
        "registry_sha256": sha(base["run"] / "representation_methods.json"),
        "esm_contract": esm_contract(cyto),
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "structure_order": "base prefix, cytotoxicity graph tail, CLint graph tail",
        "sequence_policy": "cytotoxicity complete sequence bank unchanged; CLint sequence blank",
        "label_and_group_policy": "feature only; no target or split definition is inferred",
    }
    fingerprint = signature(configuration)
    pointer = run_dir / "features/manifest.json"
    if pointer.exists():
        previous = read_json(pointer)
        if previous["input_fingerprint"] != fingerprint:
            raise ValueError("Refusing to overwrite an unrelated unified feature pointer")
        verify_unified(previous)
        verify_sources([*sources, *ancestors])
        return previous
    cache = run_dir / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(base["manifest"])
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache),
        manifest_path=str(cache / "manifest.json"),
        configuration=configuration,
        cache_reused=False,
        scope="Indexed feature representations only; scientific label eligibility remains separate",
    )
    manifest["coverage"] = {}
    manifest["artifact_hashes"] = {}
    manifest["source_snapshot_hashes"] = {}
    manifest["representation_registry_file"] = "representation_methods.json"
    for label, bank in zip(("base13", "cytotoxicity", "clint"), sources, strict=True):
        manifest["source_snapshot_hashes"].update(snapshot(bank, cache, label))
    for index, bank in enumerate(ancestors):
        manifest["source_snapshot_hashes"].update(snapshot(bank, cache, f"ancestor_{index}"))
    new_graphs = cyto["graphs"][len(base["graphs"]) :]
    extracted = [
        extract_smiles_methods(row["canonical_isomeric_smiles"], expected_key=row["key"])
        for row in new_graphs
    ]
    for name, group in manifest["groups"].items():
        if name in structural:
            if name in ADDITIONAL:
                values = np.asarray(
                    [row["values"][name] for row in extracted], dtype=np.dtype(group["dtype"])
                )
                masks = np.asarray([row["valid"][name] for row in extracted], dtype=bool)
                arrays = [base["arrays"][name][0], values, clint["arrays"][name][0]]
                valid = [base["arrays"][name][1], masks, clint["arrays"][name][1]]
            else:
                arrays = [cyto["arrays"][name][0], clint["arrays"][name][0]]
                valid = [cyto["arrays"][name][1], clint["arrays"][name][1]]
            append_arrays(safe_path(cache, group["matrix"]), arrays)
            append_arrays(safe_path(cache, group["valid"]), valid)
            group["shape"][0] = len(cyto["graphs"]) + len(clint["graphs"])
        else:
            source_group = cyto["manifest"]["groups"][name]
            for field in ("matrix", "valid"):
                relative = source_group[field]
                link_verified(
                    cyto["cache"] / relative,
                    safe_path(cache, group[field]),
                    cyto["manifest"]["artifact_hashes"][relative],
                )
            group["shape"][0] = len(cyto["sequences"])
        manifest["coverage"][name] = coverage(cache, group)
    graphs = [*cyto["graphs"], *clint["graphs"]]
    graphs = [{**row, "row_idx": i} for i, row in enumerate(graphs)]
    for relative, kind in {
        group["index"]: group["key_kind"] for group in manifest["groups"].values()
    }.items():
        destination = safe_path(cache, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind == "canonical_identity_group":
            write_tsv(
                destination,
                ["key", "row_idx"],
                [{"key": row["key"], "row_idx": i} for i, row in enumerate(graphs)],
            )
        else:
            source_group = next(
                group
                for name, group in cyto["manifest"]["groups"].items()
                if manifest["groups"][name]["index"] == relative
            )
            source_relative = source_group["index"]
            link_verified(
                cyto["cache"] / source_relative,
                destination,
                cyto["manifest"]["artifact_hashes"][source_relative],
            )
    write_tsv(cache / "structure_source_smiles.tsv", list(graphs[0]), graphs)
    aliases = [*cyto["aliases"], *clint["aliases"]]
    write_tsv(cache / manifest["record_index"], list(aliases[0]), aliases)
    manifest["requested_records"] = len(aliases)
    manifest["structure_keys"] = len(graphs)
    manifest["sequence_keys"] = len(cyto["sequences"])
    manifest["representation_methods"] = registry
    write_json(cache / "representation_methods.json", registry)
    esm_folder = Path(manifest["groups"]["esm2"]["matrix"]).parent
    source_esm_folder = Path(cyto["manifest"]["groups"]["esm2"]["matrix"]).parent
    for filename in ("esm2_manifest.json", "checkpoint.json", "input_members.tsv"):
        relative = (esm_folder / filename).as_posix()
        source_relative = (source_esm_folder / filename).as_posix()
        if source_relative in cyto["manifest"]["artifact_hashes"]:
            link_verified(
                cyto["cache"] / source_relative,
                safe_path(cache, relative),
                cyto["manifest"]["artifact_hashes"][source_relative],
            )
    np.save(cache / "structure_rows_done.npy", np.ones(len(graphs), dtype=bool), allow_pickle=False)
    np.save(
        cache / "sequence_rows_done.npy",
        np.ones(len(cyto["sequences"]), dtype=bool),
        allow_pickle=False,
    )
    write_json(
        cache / "unified_new_graph_audit.json",
        [
            {"graph_key": row["key"], "errors": result["errors"], "valid": result["valid"]}
            for row, result in zip(new_graphs, extracted, strict=True)
        ],
    )
    write_json(cache / "source_snapshot_inventory.json", manifest["source_snapshot_hashes"])
    current_files = [
        path
        for path in cache.rglob("*")
        if path.is_file() and "source_snapshots" not in path.relative_to(cache).parts
    ]
    for path in current_files:
        manifest["artifact_hashes"][path.relative_to(cache).as_posix()] = sha(path)
    verify_sources([*sources, *ancestors])
    receipt = {
        "status": "verified",
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "records": len(aliases),
        "structures": len(graphs),
        "sequences": len(cyto["sequences"]),
        "feature_groups": len(manifest["groups"]),
        "feature_dimensions": sum(group["shape"][1] for group in manifest["groups"].values()),
        "independent_methods": len(methods),
        "independent_structure_methods": 10,
        "computed_graphs_for_estate_and_erg_only": len(extracted),
        "changed_source_proven_cytotoxicity_aliases": len(changed),
        "new_clint_structure_only_records": len(clint["aliases"]),
        "source_artifacts_revalidated_unchanged": {
            label: len(bank["manifest"]["artifact_hashes"])
            for label, bank in zip(("base13", "cytotoxicity", "clint"), sources, strict=True)
        },
        "ancestor_artifacts_revalidated_unchanged": [
            len(bank["manifest"]["artifact_hashes"]) for bank in ancestors
        ],
        "all_old_numerical_prefixes_byte_equal": True,
        "all_sequence_and_esm_files_byte_identical_to_cytotoxicity_bank": True,
        "all_sources_and_pointers_unchanged": True,
        "no_labels_or_split_groups_changed": True,
        "historical_state_policy": (
            "preserved under source_snapshots only; new done masks match current universes"
        ),
        "model_training_performed": False,
        "runtime_artifact_hash_count": len(manifest["artifact_hashes"]) + 1,
        "historical_snapshot_hash_count": len(manifest["source_snapshot_hashes"]),
        "runtime_history_hash_policy": (
            "runtime validates current assets and signed history inventory; "
            "full delivery validation also checks every historical snapshot"
        ),
    }
    write_json(cache / "unified_feature_receipt.json", receipt)
    manifest["artifact_hashes"]["unified_feature_receipt.json"] = sha(
        cache / "unified_feature_receipt.json"
    )
    write_json(cache / "manifest.json", manifest)
    verify_unified(manifest)
    receipt["published_runtime_artifact_hashes_verified"] = len(manifest["artifact_hashes"])
    receipt["published_source_snapshot_hashes_verified"] = len(manifest["source_snapshot_hashes"])
    receipt["published_artifact_hashes_verified"] = len(manifest["artifact_hashes"]) + len(
        manifest["source_snapshot_hashes"]
    )
    write_json(run_dir / "unified_feature_verification.json", receipt)
    write_json(run_dir / "representation_methods.json", registry)
    write_json(pointer, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cytotoxicity-run", type=Path, required=True)
    parser.add_argument("--clint-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ancestor-run", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = unify_features(
        args.base_run, args.cytotoxicity_run, args.clint_run, args.run_dir, args.ancestor_run
    )
    print(
        {
            "status": "verified",
            "cache": manifest["cache_dir"],
            "fingerprint": manifest["input_fingerprint"],
            "records": manifest["requested_records"],
        }
    )


if __name__ == "__main__":
    main()
