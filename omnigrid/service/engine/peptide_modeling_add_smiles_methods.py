"""Add EState and ErG methods to an independent, immutable SMILES feature bank."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from peptide_modeling_add_methods import ordered_index
from peptide_modeling_expand_features import (
    flush,
    open_array,
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
)
from peptide_modeling_features import _verify_artifacts, canonical_key

VERSION = "peptide_estate_erg_smiles_methods_v1"
GROUPS = {"estate": 158, "erg": 315}
REFERENCES = {
    "estate": "https://www.rdkit.org/docs/source/rdkit.Chem.EState.Fingerprinter.html",
    "erg": "https://www.rdkit.org/docs/source/rdkit.Chem.rdReducedGraphs.html",
}
ERG_OPTIONS = {"atomTypes": 0, "fuzzIncrement": 0.3, "minPath": 1, "maxPath": 15}


@lru_cache(maxsize=1)
def rdkit_tools():
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdReducedGraphs
    from rdkit.Chem.EState import AtomTypes, Fingerprinter

    AtomTypes.BuildPatts()
    names = [entry[0] if entry else "" for entry in AtomTypes.esPatterns]
    if len(names) != 79 or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("The pinned EState schema requires 79 unique named atom types")
    probe = Chem.MolFromSmiles("NCC(=O)O")
    estate = Fingerprinter.FingerprintMol(probe)
    erg = rdReducedGraphs.GetErGFingerprint(probe, **ERG_OPTIONS)
    if [len(values) for values in estate] != [79, 79] or np.asarray(erg).shape != (315,):
        raise ValueError("Installed RDKit fingerprint dimensions differ from the frozen schema")
    return {
        "Chem": Chem,
        "estate": Fingerprinter.FingerprintMol,
        "erg": rdReducedGraphs.GetErGFingerprint,
        "atom_types": names,
        "version": rdBase.rdkitVersion,
    }


def invalid_result(reason):
    # JSON contains finite placeholders, but their masks are false. Published
    # matrices use NaN for these unavailable rows rather than valid fake zeroes.
    return {
        "values": {name: [0.0] * size for name, size in GROUPS.items()},
        "valid": dict.fromkeys(GROUPS, False),
        "errors": [reason],
    }


def extract_smiles_methods(smiles, max_atoms=1000, expected_key=None):
    rd = rdkit_tools()
    try:
        molecule = rd["Chem"].MolFromSmiles(smiles)
    except Exception:
        molecule = None
    if molecule is None:
        return invalid_result("parse_error_review")
    if molecule.GetNumAtoms() > max_atoms:
        return invalid_result("atom_limit_review")
    # Standardize traversal order for stable floating-point aggregation. The
    # source identity itself remains the complete canonical isomeric SMILES.
    canonical = rd["Chem"].MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if expected_key is not None and canonical_key(canonical) != expected_key:
        return invalid_result("canonical_identity_review")
    molecule = rd["Chem"].MolFromSmiles(canonical)
    result = invalid_result("uncomputed")
    result["errors"] = []
    functions = {
        "estate": lambda: np.concatenate(rd["estate"](molecule)),
        "erg": lambda: rd["erg"](molecule, **ERG_OPTIONS),
    }
    for name, function in functions.items():
        try:
            values = np.asarray(function(), dtype=np.float64)
            if values.shape != (GROUPS[name],) or not np.isfinite(values).all():
                raise ValueError("Fingerprint has an invalid dimension or nonfinite entry")
            if np.any(np.abs(values) > np.finfo(np.float32).max):
                raise ValueError("Fingerprint does not fit float32")
            if name == "estate" and (
                np.any(values[:79] < 0) or not np.equal(values[:79], np.floor(values[:79])).all()
            ):
                raise ValueError("EState atom-type counts are not nonnegative integers")
            if name == "erg" and np.any(values < 0):
                raise ValueError("ErG fingerprint contains negative fuzzy counts")
            result["values"][name] = values.astype(np.float32).tolist()
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
            [sys.executable, str(Path(__file__).resolve()), "--smiles-worker"],
            input=json.dumps(payload),
            encoding="utf-8",
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
        if process.returncode == 0:
            results = json.loads(process.stdout)
            if len(results) != len(payload):
                raise ValueError("Worker output cardinality differs from its input")
            for result in results:
                for name, size in GROUPS.items():
                    values = np.asarray(result["values"][name])
                    if values.shape != (size,) or not np.isfinite(values).all():
                        raise ValueError("Worker returned invalid fingerprint values")
            return results
    except subprocess.TimeoutExpired:
        reason = "feature_timeout_review"
    except (OSError, ValueError):
        pass
    if len(payload) == 1:
        return [invalid_result(reason)]
    middle = len(payload) // 2
    return isolated_batch(payload[:middle], timeout) + isolated_batch(payload[middle:], timeout)


def source_graphs(manifest):
    """Bind every inherited graph to the frozen full-source SMILES mapping."""
    cache = Path(manifest["cache_dir"])
    relative = "structure_source_smiles.tsv"
    if relative not in manifest.get("artifact_hashes", {}):
        raise ValueError("The source SMILES mapping must be a hashed source artifact")
    if sha(cache / relative) != manifest["artifact_hashes"][relative]:
        raise ValueError("Source SMILES mapping hash mismatch")
    indexed = ordered_index(cache / manifest["groups"]["morgan"]["index"])
    mapping = list(rows(cache / relative))
    if len(mapping) != len(indexed):
        raise ValueError("Source graph and full SMILES universes differ")
    by_key = {}
    for item in mapping:
        smiles = item["canonical_isomeric_smiles"]
        if item["key"] in by_key or canonical_key(smiles) != item["key"]:
            raise ValueError("Source graph identity collides or differs from its full SMILES SHA")
        by_key[item["key"]] = item
    for item in indexed:
        mapped = by_key.get(item["key"])
        if mapped is None or int(mapped["row_idx"]) != int(item["row_idx"]):
            raise ValueError("Source graph row positions differ from the frozen SMILES map")
    return indexed, {key: item["canonical_isomeric_smiles"] for key, item in by_key.items()}


def registry_with_new_methods(original):
    registry = copy.deepcopy(original["representation_methods"])
    additions = [
        {
            "method_id": "estate_fingerprint",
            "algorithm_family_id": "electrotopological_atom_types",
            "group_names": ["estate"],
            "route": "method_estate_fingerprint",
            "independent": True,
            "source_kind": "structure",
            "stereochemistry_encoded": False,
            "dimensions": 158,
        },
        {
            "method_id": "erg",
            "algorithm_family_id": "extended_reduced_graph_pharmacophores",
            "group_names": ["erg"],
            "route": "method_erg",
            "independent": True,
            "source_kind": "structure",
            "stereochemistry_encoded": False,
            "dimensions": 315,
        },
    ]
    registry["methods"].extend(additions)
    methods = registry["methods"]
    for field in ("method_id", "algorithm_family_id", "route"):
        if len({method[field] for method in methods}) != len(methods):
            raise ValueError("New algorithms duplicate a source method or algorithm family")
    registry["independent_method_count"] = len(methods)
    registry["additional_smiles_methods_references"] = REFERENCES
    registry["additional_smiles_methods_policy"] = (
        "EState uses 79 atom-type counts and 79 summed electrotopological indices. "
        "ErG uses the pinned reduced-graph pharmacophore fingerprint. Neither encodes "
        "chirality; source identities and chirality-sensitive methods stay unchanged."
    )
    return registry


def copy_immutable(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != expected:
            raise ValueError("Existing inherited artifact changed: " + str(destination))
        return
    if sha(source) != expected:
        raise ValueError("Inherited source artifact changed")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha(destination) != expected:
        raise ValueError("Inherited copy hash mismatch")


def add_smiles_methods(base_run, output_run, workers=4, batch_size=32, timeout=120):
    base_run, output_run = Path(base_run).resolve(), Path(output_run).resolve()
    if base_run == output_run or base_run in output_run.parents or output_run in base_run.parents:
        raise ValueError("New methods require an independent sibling output directory")
    if not 1 <= workers <= 4 or not 1 <= batch_size <= 64 or not 0 < timeout <= 300:
        raise ValueError("Extraction resources must remain within the bounded local CPU budget")
    pointer = base_run / "features/manifest.json"
    original = read_json(pointer)
    _verify_artifacts(original)
    indexed, structures = source_graphs(original)
    source = Path(original["cache_dir"])
    registry = registry_with_new_methods(original)
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "source_manifest_sha256": sha(pointer),
        "source_fingerprint": original["input_fingerprint"],
        "source_smiles_sha256": original["artifact_hashes"]["structure_source_smiles.tsv"],
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "max_structure_atoms": 1000,
        "group_dimensions": GROUPS,
        "estate_atom_type_names": rdkit_tools()["atom_types"],
        "erg_options": ERG_OPTIONS,
        "references": REFERENCES,
    }
    fingerprint = signature(configuration)
    cache = output_run / "features" / fingerprint[:16]
    output_pointer = output_run / "features/manifest.json"
    if output_pointer.exists() and read_json(output_pointer)["input_fingerprint"] != fingerprint:
        raise ValueError("Refusing to replace a feature pointer for different inputs")
    if (cache / "manifest.json").exists():
        result = read_json(cache / "manifest.json")
        _verify_artifacts(result)
        return result
    cache.mkdir(parents=True, exist_ok=True)
    reserved = {
        "additional_smiles_registry.json",
        "additional_smiles_source_manifest.json",
        "additional_smiles_done.npy",
        "additional_smiles_errors.jsonl",
        "additional_smiles_receipt.json",
        *[name + suffix for name in GROUPS for suffix in (".npy", "_valid.npy")],
    }
    if reserved & set(original["artifact_hashes"]):
        raise ValueError("New artifact names collide with inherited immutable files")
    for relative, expected in original["artifact_hashes"].items():
        copy_immutable(safe_path(source, relative), safe_path(cache, relative), expected)
    shutil.copy2(pointer, cache / "additional_smiles_source_manifest.json")
    write_json(cache / "additional_smiles_configuration.json", configuration)
    matrices, masks = {}, {}
    for name, size in GROUPS.items():
        matrices[name] = open_array(cache / (name + ".npy"), (len(indexed), size), "float32")
        masks[name] = open_array(cache / (name + "_valid.npy"), (len(indexed),), "bool")
    done = open_array(cache / "additional_smiles_done.npy", (len(indexed),), "bool")
    pending = [item for item in indexed if not done[int(item["row_idx"])]]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    errors_path = cache / "additional_smiles_errors.jsonl"
    with errors_path.open("a", encoding="utf-8") as errors, ThreadPoolExecutor(workers) as pool:
        futures = {
            pool.submit(
                isolated_batch,
                [
                    {
                        "smiles": structures[item["key"]],
                        "max_atoms": 1000,
                        "expected_key": item["key"],
                    }
                    for item in batch
                ],
                timeout,
            ): batch
            for batch in batches
        }
        completed = len(indexed) - len(pending)
        for future in as_completed(futures):
            batch, results = futures[future], future.result()
            for item, result in zip(batch, results, strict=True):
                position = int(item["row_idx"])
                for name, size in GROUPS.items():
                    values = np.asarray(result["values"][name], dtype=np.float32)
                    valid = bool(result["valid"][name])
                    if values.shape != (size,) or (valid and not np.isfinite(values).all()):
                        raise ValueError("Worker vector violates its named schema")
                    matrices[name][position] = values if valid else np.full(size, np.nan)
                    masks[name][position] = valid
                done[position] = True
                if result["errors"]:
                    errors.write(
                        json.dumps({"key": item["key"], "errors": result["errors"]}) + "\n"
                    )
            flush([*matrices.values(), *masks.values(), done])
            errors.flush()
            completed += len(batch)
            if completed % 512 < batch_size:
                print(
                    json.dumps({"structures_done": completed, "structures_total": len(indexed)}),
                    flush=True,
                )
    if not done.all():
        raise ValueError("Not all source structures have a terminal extraction state")
    manifest = copy.deepcopy(original)
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache),
        manifest_path=str(cache / "manifest.json"),
        cache_reused=False,
    )
    manifest["configuration"] = {
        "source_configuration": original["configuration"],
        "additional_smiles_methods": configuration,
    }
    type_names = rdkit_tools()["atom_types"]
    names = {
        "estate": ["count_" + name for name in type_names]
        + ["estate_sum_" + name for name in type_names],
        "erg": [f"erg_{index:03d}" for index in range(315)],
    }
    for name, size in GROUPS.items():
        valid = np.asarray(masks[name])
        if not np.isfinite(matrices[name][valid]).all():
            raise ValueError("A published valid feature row contains a nonfinite value")
        if not np.isnan(matrices[name][~valid]).all():
            raise ValueError("Unavailable feature rows must retain explicit missing values")
        manifest["groups"][name] = {
            "matrix": name + ".npy",
            "valid": name + "_valid.npy",
            "index": original["groups"]["morgan"]["index"],
            "key_kind": "canonical_identity_group",
            "dtype": "float32",
            "shape": [len(indexed), size],
            "feature_names": names[name],
            "stereochemistry_encoded": False,
        }
        manifest["coverage"][name] = {
            "representations": len(indexed),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "valid_zero_rows": int(np.sum(valid & ~np.any(matrices[name] != 0, axis=1))),
        }
    for method in registry["methods"][-2:]:
        manifest["routes"][method["route"]] = method["group_names"]
    for route in ("structure_methods_all", "fusion_methods_all"):
        manifest["routes"][route] = [*original["routes"][route], *GROUPS]
    manifest["representation_methods"] = registry
    manifest["representation_registry_file"] = "additional_smiles_registry.json"
    write_json(cache / "additional_smiles_registry.json", registry)
    inherited = original["artifact_hashes"]
    for relative, expected in inherited.items():
        if (
            sha(safe_path(source, relative)) != expected
            or sha(safe_path(cache, relative)) != expected
        ):
            raise ValueError("An inherited artifact changed during extraction")
    aliases = list(rows(cache / original["record_index"]))
    ids = [item["record_id"] for item in aliases]
    if len(ids) != len(set(ids)) or len(ids) != original["requested_records"]:
        raise ValueError("Record aliases must preserve every unique source label ID")
    receipt = {
        "schema": VERSION,
        "status": "completed",
        "input_fingerprint": fingerprint,
        "source_pointer_sha256": configuration["source_manifest_sha256"],
        "source_artifacts_unchanged": len(inherited),
        "source_record_aliases_unchanged": True,
        "source_smiles_mapping_sha256": configuration["source_smiles_sha256"],
        "requested_records": len(ids),
        "structure_keys": len(indexed),
        "independent_method_count": len(registry["methods"]),
        "independent_structure_methods": sum(
            m["source_kind"] == "structure" for m in registry["methods"]
        ),
        "feature_groups": len(manifest["groups"]),
        "unique_feature_dimensions": sum(
            group["shape"][1] for group in manifest["groups"].values()
        ),
        "coverage": {name: manifest["coverage"][name] for name in GROUPS},
        "script_sha256": configuration["implementation_sha256"],
        "references": REFERENCES,
    }
    write_json(cache / "additional_smiles_receipt.json", receipt)
    extras = reserved | {"additional_smiles_configuration.json"}
    manifest["artifact_hashes"] = {
        **inherited,
        **{name: sha(cache / name) for name in extras},
    }
    if sha(pointer) != configuration["source_manifest_sha256"]:
        raise ValueError("Source feature pointer changed during extraction")
    _verify_artifacts(manifest)
    write_json(cache / "manifest.json", manifest)
    write_json(output_run / "representation_methods.json", registry)
    write_json(output_run / "additional_smiles_receipt.json", receipt)
    write_json(output_pointer, manifest)
    return manifest


def main():
    if sys.argv[1:] == ["--smiles-worker"]:
        payload = json.load(sys.stdin)
        json.dump(
            [
                extract_smiles_methods(
                    item["smiles"], item.get("max_atoms", 1000), item.get("expected_key")
                )
                for item in payload
            ],
            sys.stdout,
            allow_nan=False,
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = add_smiles_methods(args.base_run, args.run_dir, args.workers, args.batch_size)
    print(
        json.dumps(
            {
                "input_fingerprint": result["input_fingerprint"],
                "cache_dir": result["cache_dir"],
                "methods": len(result["representation_methods"]["methods"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
