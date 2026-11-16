"""Add independently named graph methods without changing an existing feature bank."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from peptide_modeling_expand_features import (
    copy_verified,
    flush,
    open_array,
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _resolve_structures, _verify_artifacts

VERSION = "peptide_independent_methods_v1"
NEW_GROUPS = ("rdkit_path", "pattern", "layered")
FP_SIZE = 2048
RDKIT_DOC = "https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html"


@lru_cache(maxsize=1)
def rdkit_tools():
    from rdkit import Chem, DataStructs, rdBase

    return {"Chem": Chem, "DataStructs": DataStructs, "version": rdBase.rdkitVersion}


def method_registry(groups):
    """Count algorithm families, never radii, descriptor subsets, or concatenations."""
    specifications = (
        ("morgan", "circular_atom_environment", "morgan", "structure", True),
        ("atom_pair", "distance_labelled_atom_pairs", "atompair", "structure", True),
        ("topological_torsion", "four_atom_topological_torsions", "torsion", "structure", True),
        ("maccs", "maccs_structural_keys", "maccs", "structure", False),
        ("rdkit_descriptors", "named_molecular_descriptors", "rdkit_full", "structure", "mixed"),
        (
            "sequence_composition",
            "residue_composition_physicochemical",
            "sequence",
            "sequence",
            False,
        ),
        ("dipeptide_composition", "ordered_adjacent_residue_frequency", "dpc", "sequence", False),
        ("esm2", "pretrained_transformer_residue_mean", "esm2", "sequence", False),
        ("rdkit_path", "hashed_labelled_subgraph_paths", "rdkit_path", "structure", False),
        ("pattern", "rdkit_smarts_pattern_screen", "pattern", "structure", False),
        ("layered", "layered_topological_atom_bond_ring_paths", "layered", "structure", False),
    )
    methods = []
    for method, family, group, source_kind, stereo in specifications:
        methods.append(
            {
                "method_id": method,
                "algorithm_family_id": family,
                "group_names": [group],
                "route": "method_" + method,
                "independent": True,
                "source_kind": source_kind,
                "stereochemistry_encoded": stereo,
                "dimensions": int(groups[group]["shape"][1]),
            }
        )
    return {
        "schema": "peptide_independent_representation_methods_v1",
        "independent_method_count": len(methods),
        "methods": methods,
        "non_counted_variants": {
            "morgan_r3": "same circular algorithm family as Morgan radius2",
            "descriptors": "subset of named full RDKit descriptors",
            "graph_modifications": "supplementary graph motif measurements",
            "sequence_extra": "supplementary sequence measurements",
            "fusion_routes": "concatenations are not independent methods",
        },
        "new_graph_methods_reference": RDKIT_DOC,
        "structure_policy": (
            "Use the full source-validated canonical isomeric SMILES. New path/pattern/layered "
            "fingerprints are stereo-insensitive; chirality-sensitive Morgan/AtomPair/Torsion "
            "remain separate methods. Never infer a missing modification or replace it with "
            "an unmodified standard residue."
        ),
    }


def invalid_result(reason):
    return {
        "values": {name: [0] * FP_SIZE for name in NEW_GROUPS},
        "valid": dict.fromkeys(NEW_GROUPS, False),
        "errors": [reason],
    }


def extract_graph_methods(smiles, max_atoms=1000):
    rd = rdkit_tools()
    try:
        molecule = rd["Chem"].MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None:
        return invalid_result("parse_error_review")
    if molecule.GetNumAtoms() > max_atoms:
        return invalid_result("atom_limit_review")
    result = invalid_result("uncomputed")
    result["errors"] = []
    functions = {
        "rdkit_path": lambda mol: rd["Chem"].RDKFingerprint(
            mol,
            minPath=1,
            maxPath=7,
            fpSize=FP_SIZE,
            nBitsPerHash=2,
            useHs=True,
            tgtDensity=0.0,
            branchedPaths=True,
            useBondOrder=True,
        ),
        "pattern": lambda mol: rd["Chem"].PatternFingerprint(
            mol,
            fpSize=FP_SIZE,
            tautomerFingerprints=False,
        ),
        "layered": lambda mol: rd["Chem"].LayeredFingerprint(
            mol,
            layerFlags=0xFFFFFFFF,
            minPath=1,
            maxPath=7,
            fpSize=FP_SIZE,
            branchedPaths=True,
        ),
    }
    for name, function in functions.items():
        try:
            values = np.zeros(FP_SIZE, dtype=np.uint8)
            rd["DataStructs"].ConvertToNumpyArray(function(molecule), values)
            if not np.isin(values, [0, 1]).all():
                raise ValueError("Fingerprint is not binary")
            result["values"][name] = values.tolist()
            result["valid"][name] = True
        except Exception as exc:
            result["errors"].append(name + ":" + type(exc).__name__)
    return result


def isolated_batch(payload, timeout=120):
    if not payload:
        return []
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    reason = "native_failure_review"
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--fingerprint-worker"],
            input=json.dumps(payload),
            encoding="utf-8",
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
        if process.returncode == 0:
            result = json.loads(process.stdout)
            if len(result) != len(payload):
                raise ValueError("Worker result count mismatch")
            for item in result:
                for name in NEW_GROUPS:
                    vector = np.asarray(item["values"][name])
                    if vector.shape != (FP_SIZE,) or not np.isin(vector, [0, 1]).all():
                        raise ValueError("Worker returned an invalid fingerprint")
            return result
    except subprocess.TimeoutExpired:
        reason = "feature_timeout_review"
    except (OSError, ValueError):
        pass
    if len(payload) == 1:
        return [invalid_result(reason)]
    middle = len(payload) // 2
    return isolated_batch(payload[:middle], timeout) + isolated_batch(payload[middle:], timeout)


def ordered_index(path):
    result = sorted(rows(path), key=lambda item: int(item["row_idx"]))
    if [int(item["row_idx"]) for item in result] != list(range(len(result))):
        raise ValueError("Index row positions must be unique and contiguous")
    if len({item["key"] for item in result}) != len(result):
        raise ValueError("Duplicate representation keys")
    return result


def add_methods(base_run, dataset_dir, output_run, workers=4, batch_size=64, timeout=120):
    base_run, dataset_dir, output_run = map(Path, (base_run, dataset_dir, output_run))
    if (
        base_run.resolve() == output_run.resolve()
        or base_run.resolve() in output_run.resolve().parents
    ):
        raise ValueError("New methods require an independent sibling run directory")
    if not 1 <= workers <= 16 or not 1 <= batch_size <= 256 or timeout <= 0:
        raise ValueError("Invalid bounded extraction resources")
    pointer = base_run / "features/manifest.json"
    original = read_json(pointer)
    _verify_artifacts(original)
    source_cache = Path(original["cache_dir"])
    structure_group = original["groups"]["morgan"]
    indexed = ordered_index(source_cache / structure_group["index"])
    requested = [{"canonical_identity_group": item["key"]} for item in indexed]
    structures, _ = _resolve_structures(dataset_dir, requested)
    if set(structures) != {item["key"] for item in indexed}:
        raise ValueError("All inherited structural identities need source-validated SMILES")
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "base_manifest_sha256": sha(pointer),
        "base_input_fingerprint": original["input_fingerprint"],
        "structure_qc_sha256": sha(dataset_dir / "structure_qc/structure_validation.tsv"),
        "dataset_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "structure_index_sha256": sha(source_cache / structure_group["index"]),
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "new_fingerprint_size": FP_SIZE,
        "max_structure_atoms": 1000,
        "rdkit_path": {
            "minPath": 1,
            "maxPath": 7,
            "nBitsPerHash": 2,
            "useHs": True,
            "tgtDensity": 0,
            "branchedPaths": True,
            "useBondOrder": True,
        },
        "pattern": {"tautomerFingerprints": False, "experimental_api_version_pinned": True},
        "layered": {
            "layerFlags": 0xFFFFFFFF,
            "minPath": 1,
            "maxPath": 7,
            "branchedPaths": True,
            "experimental_api_version_pinned": True,
        },
    }
    fingerprint = signature(configuration)
    cache = output_run / "features" / fingerprint[:16]
    output_pointer = output_run / "features/manifest.json"
    if output_pointer.exists() and read_json(output_pointer)["input_fingerprint"] != fingerprint:
        raise ValueError("Refusing to overwrite an output pointer for different inputs")
    if (cache / "manifest.json").exists():
        manifest = read_json(cache / "manifest.json")
        _verify_artifacts(manifest)
        write_json(output_run / "representation_methods.json", method_registry(manifest["groups"]))
        copy_verified(
            cache / "independent_methods_receipt.json",
            output_run / "independent_methods_receipt.json",
        )
        write_json(output_pointer, manifest)
        return manifest
    cache.mkdir(parents=True, exist_ok=True)
    for relative in original["artifact_hashes"]:
        copy_verified(safe_path(source_cache, relative), safe_path(cache, relative))
    copy_verified(pointer, cache / "inherited_feature_manifest.json")
    matrices, masks = {}, {}
    for name in NEW_GROUPS:
        matrices[name] = open_array(cache / (name + ".npy"), (len(indexed), FP_SIZE), "uint8")
        masks[name] = open_array(cache / (name + "_valid.npy"), (len(indexed),), "bool")
    done = open_array(cache / "additional_methods_done.npy", (len(indexed),), "bool")
    pending = [item for item in indexed if not done[int(item["row_idx"])]]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    with (cache / "additional_methods_errors.jsonl").open("a", encoding="utf-8") as errors:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    isolated_batch,
                    [{"smiles": structures[item["key"]], "max_atoms": 1000} for item in batch],
                    timeout,
                ): batch
                for batch in batches
            }
            completed = 0
            for future in as_completed(futures):
                batch = futures[future]
                extracted = future.result()
                for item, result in zip(batch, extracted, strict=True):
                    index = int(item["row_idx"])
                    for name in NEW_GROUPS:
                        matrices[name][index] = result["values"][name]
                        masks[name][index] = result["valid"][name]
                    done[index] = True
                    if result["errors"]:
                        errors.write(
                            json.dumps({"key": item["key"], "errors": result["errors"]}) + "\n"
                        )
                flush([*matrices.values(), *masks.values(), done])
                errors.flush()
                completed += len(batch)
                if completed % 512 < batch_size:
                    print(
                        json.dumps(
                            {"structures_done": completed, "pending_at_start": len(pending)}
                        ),
                        flush=True,
                    )
    if not done.all():
        raise ValueError("Not all structural methods reached a terminal extraction state")
    manifest = copy.deepcopy(original)
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        cache_reused=False,
    )
    manifest["configuration"] = {
        "source_configuration": original["configuration"],
        "independent_method_expansion": configuration,
    }
    for name in NEW_GROUPS:
        valid = np.asarray(masks[name])
        manifest["groups"][name] = {
            "matrix": name + ".npy",
            "valid": name + "_valid.npy",
            "index": structure_group["index"],
            "key_kind": "canonical_identity_group",
            "dtype": "uint8",
            "shape": [len(indexed), FP_SIZE],
            "feature_names": [f"bit_{i}" for i in range(FP_SIZE)],
            "stereochemistry_encoded": False,
        }
        manifest["coverage"][name] = {
            "representations": len(valid),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "all_zero_rows": int(np.sum(valid & ~np.any(matrices[name] != 0, axis=1))),
        }
    registry = method_registry(manifest["groups"])
    for method in registry["methods"]:
        manifest["routes"][method["route"]] = method["group_names"]
    manifest["routes"]["structure_methods_all"] = [
        *original["routes"]["structure_extended"],
        *NEW_GROUPS,
    ]
    manifest["routes"]["fusion_methods_all"] = [
        *original["routes"]["fusion_esm_extended"],
        *NEW_GROUPS,
    ]
    manifest["representation_methods"] = registry
    write_json(cache / "representation_methods.json", registry)
    inherited = original["artifact_hashes"]
    manifest["artifact_hashes"] = {
        relative: sha(safe_path(cache, relative)) for relative in inherited
    }
    if manifest["artifact_hashes"] != inherited:
        raise ValueError("Inherited feature artifacts changed")
    extra = [
        "representation_methods.json",
        "inherited_feature_manifest.json",
        "additional_methods_done.npy",
        "additional_methods_errors.jsonl",
    ]
    extra.extend(name + suffix for name in NEW_GROUPS for suffix in (".npy", "_valid.npy"))
    manifest["artifact_hashes"].update({name: sha(cache / name) for name in extra})
    if sha(pointer) != configuration["base_manifest_sha256"]:
        raise ValueError("Source pointer changed during independent extraction")
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "row_idx", "canonical_isomeric_smiles"],
        ({**item, "canonical_isomeric_smiles": structures[item["key"]]} for item in indexed),
    )
    # Freeze the graph mapping as an artifact too; it includes every reported modification.
    manifest["artifact_hashes"]["structure_source_smiles.tsv"] = sha(
        cache / "structure_source_smiles.tsv"
    )
    receipt = {
        "schema": VERSION,
        "status": "completed",
        "input_fingerprint": fingerprint,
        "source_pointer_sha256": configuration["base_manifest_sha256"],
        "source_pointer_unchanged": True,
        "inherited_artifacts_byte_identical": len(inherited),
        "independent_method_count": registry["independent_method_count"],
        "feature_groups": len(manifest["groups"]),
        "unique_feature_dimensions": sum(
            group["shape"][1] for group in manifest["groups"].values()
        ),
        "requested_records": manifest["requested_records"],
        "structure_keys": len(indexed),
        "coverage": {name: manifest["coverage"][name] for name in NEW_GROUPS},
        "source_smiles_mapping_sha256": manifest["artifact_hashes"]["structure_source_smiles.tsv"],
        "modified_structure_policy": registry["structure_policy"],
        "script_sha256": configuration["implementation_sha256"],
        "reference": RDKIT_DOC,
    }
    write_json(cache / "independent_methods_receipt.json", receipt)
    manifest["artifact_hashes"]["independent_methods_receipt.json"] = sha(
        cache / "independent_methods_receipt.json"
    )
    write_json(cache / "manifest.json", manifest)
    write_json(output_run / "representation_methods.json", registry)
    write_json(output_run / "independent_methods_receipt.json", receipt)
    write_json(output_pointer, manifest)
    return manifest


def main():
    if sys.argv[1:] == ["--fingerprint-worker"]:
        payload = json.load(sys.stdin)
        json.dump(
            [
                extract_graph_methods(item["smiles"], item.get("max_atoms", 1000))
                for item in payload
            ],
            sys.stdout,
            allow_nan=False,
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    manifest = add_methods(
        args.base_run, args.dataset, args.run_dir, workers=args.workers, batch_size=args.batch_size
    )
    print(
        json.dumps(
            {
                "input_fingerprint": manifest["input_fingerprint"],
                "independent_methods": manifest["representation_methods"][
                    "independent_method_count"
                ],
                "cache_dir": manifest["cache_dir"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
