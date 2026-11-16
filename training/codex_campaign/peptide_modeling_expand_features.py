"""Add audited CPU feature types to an independent fresh-ESM experiment.

The original arrays and frozen outer splits are copied, never overwritten. Full
RDKit columns retain explicit missing values; their imputation belongs inside CV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np

VERSION = "peptide_expanded_features_v1"
AA = "ACDEFGHIKLMNPQRSTVWY"
STRUCTURE_GROUPS = ("morgan_r3", "atompair", "torsion", "maccs", "rdkit_full")
BASE_STRUCTURE_GROUPS = ("morgan", "descriptors", "graph_modifications")
SEQUENCE_EXTRA_NAMES = [
    "shannon_entropy_bits",
    "entropy_over_log2_20",
    "distinct_aa_fraction_of_20",
    "polar_STNQ_fraction",
    "aliphatic_ILV_fraction",
    "sulfur_CM_fraction",
    "small_AGST_fraction",
    "turn_GPSND_fraction",
    "glycine_proline_fraction",
    "longest_identical_run_fraction",
    "adjacent_repeat_fraction",
    "n_terminal_basic_KRH",
    "c_terminal_acidic_DE",
    "n_terminal_parent_hydropathy",
    "c_terminal_parent_hydropathy",
    "hydrophobic_moment_ideal_100deg",
    "hydrophobic_moment_ideal_180deg",
    "adjacent_aromatic_pair_fraction",
    "opposite_charge_group_neighbor_fraction",
    "mean_adjacent_parent_hydropathy_change",
]
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
DESCRIPTOR_TRANSFORMS = {"Ipc": "signed_log1p", "AvgIpc": "signed_log1p"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rows(path):
    csv.field_size_limit(10_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, fields, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def safe_path(root, relative):
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError("Unsafe relative source artifact")
    resolved = Path(root).joinpath(*path.parts).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes source root")
    return resolved


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError("Existing copied input changed: " + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha(source) != sha(destination):
        raise ValueError("Copied input changed")


@lru_cache(maxsize=1)
def rdkit_tools():
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

    functions = list(Descriptors._descList)
    names = [name for name, _ in functions]
    if not functions or len(names) != len(set(names)):
        raise ValueError("RDKit descriptor names must be nonempty and unique")
    return {
        "Chem": Chem,
        "DataStructs": DataStructs,
        "version": rdBase.rdkitVersion,
        "descriptor_functions": functions,
        "maccs": MACCSkeys.GenMACCSKeys,
        "morgan_r3": rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=2048, includeChirality=True
        ),
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=2048, includeChirality=True, countSimulation=False
        ),
        "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=2048, includeChirality=True
        ),
    }


def definitions():
    descriptor_names = [name for name, _ in rdkit_tools()["descriptor_functions"]]
    return {
        "morgan_r3": (2048, "uint8", [f"bit_{i}" for i in range(2048)]),
        "atompair": (2048, "uint8", [f"bit_{i}" for i in range(2048)]),
        "torsion": (2048, "uint8", [f"bit_{i}" for i in range(2048)]),
        "maccs": (167, "uint8", [f"key_{i}" for i in range(167)]),
        "rdkit_full": (
            len(descriptor_names),
            "float32",
            [
                "signed_log1p_" + name if name in DESCRIPTOR_TRANSFORMS else name
                for name in descriptor_names
            ],
        ),
        "dpc": (400, "float32", ["dpc_" + a + b for a in AA for b in AA]),
        "sequence_extra": (len(SEQUENCE_EXTRA_NAMES), "float32", SEQUENCE_EXTRA_NAMES),
    }


def invalid_structure(error, include_base=False):
    defs = definitions()
    result = {
        "values": {
            k: np.full(defs[k][0], np.nan, dtype=np.float32)
            if k == "rdkit_full"
            else np.zeros(defs[k][0], dtype=np.uint8)
            for k in STRUCTURE_GROUPS
        },
        "valid": dict.fromkeys(STRUCTURE_GROUPS, False),
        "errors": [error],
    }
    if include_base:
        from peptide_modeling_features import DESCRIPTOR_NAMES, GRAPH_MOD_NAMES

        for name, dim in (
            ("morgan", 2048),
            ("descriptors", len(DESCRIPTOR_NAMES)),
            ("graph_modifications", len(GRAPH_MOD_NAMES)),
        ):
            result["values"][name] = np.zeros(dim, dtype="uint8" if name == "morgan" else "float32")
            result["valid"][name] = False
    return result


def extract_expanded_structure(smiles, max_atoms=1000, include_base=False):
    rd = rdkit_tools()
    result = invalid_structure("uncomputed", include_base)
    try:
        mol = rd["Chem"].MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return invalid_structure("parse_error", include_base)
    atoms = mol.GetNumAtoms()
    if atoms > max_atoms:
        return invalid_structure("atom_limit_review", include_base)
    result["errors"] = []
    for name in ("morgan_r3", "atompair", "torsion", "maccs"):
        try:
            vector = rd["maccs"](mol) if name == "maccs" else rd[name].GetFingerprint(mol)
            rd["DataStructs"].ConvertToNumpyArray(vector, result["values"][name])
            result["valid"][name] = True
        except Exception as exc:
            result["errors"].append(name + ":" + type(exc).__name__)
    full = result["values"]["rdkit_full"]
    for i, (name, function) in enumerate(rd["descriptor_functions"]):
        if (name in {"Ipc", "AvgIpc"} and atoms > 128) or (
            name.startswith("BCUT2D_") and atoms > 256
        ):
            result["errors"].append("descriptor:" + name + ":size_guard")
            continue
        try:
            value = float(function(mol))
            if not math.isfinite(value):
                raise ValueError("nonfinite")
            if name in DESCRIPTOR_TRANSFORMS:
                value = math.copysign(math.log1p(abs(value)), value)
            if abs(value) > np.finfo(np.float32).max:
                raise ValueError("outside_float32")
            full[i] = value
        except Exception as exc:
            result["errors"].append("descriptor:" + name + ":" + type(exc).__name__)
    result["valid"]["rdkit_full"] = bool(np.isfinite(full).any() and np.isnan(full).mean() <= 0.1)
    if include_base:
        from peptide_modeling_features import extract_structure

        baseline = extract_structure(smiles, max_atoms=max_atoms)
        for name in BASE_STRUCTURE_GROUPS:
            result["values"][name] = baseline[name]
            result["valid"][name] = baseline["group_valid"][name]
        if baseline["error"]:
            result["errors"].append("base:" + baseline["error"])
    return result


def extract_expanded_sequence(sequence):
    dpc = np.zeros(400, dtype=np.float32)
    extra = np.zeros(len(SEQUENCE_EXTRA_NAMES), dtype=np.float32)
    if not sequence or set(sequence) - set(AA):
        return {"valid": False, "dpc": dpc, "sequence_extra": extra}
    n = len(sequence)
    counts = Counter(sequence)
    probabilities = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)

    def fraction(letters):
        return sum(counts[a] for a in letters) / n

    runs, longest = 1, 1
    repeats = aromatic = opposite = 0
    hydro_change = 0.0
    for a, b in zip(sequence[:-1], sequence[1:], strict=True):
        dpc[AA.index(a) * 20 + AA.index(b)] += 1 / (n - 1)
        runs = runs + 1 if a == b else 1
        longest = max(longest, runs)
        repeats += a == b
        aromatic += a in "FWY" and b in "FWY"
        opposite += (a in "DE" and b in "KR") or (a in "KR" and b in "DE")
        hydro_change += abs(HYDROPATHY[a] - HYDROPATHY[b])
    moments = []
    for angle in (100, 180):
        real = sum(
            HYDROPATHY[a] * math.cos(math.radians(angle * i)) for i, a in enumerate(sequence)
        )
        imag = sum(
            HYDROPATHY[a] * math.sin(math.radians(angle * i)) for i, a in enumerate(sequence)
        )
        moments.append(math.hypot(real, imag) / n)
    pairs = max(n - 1, 1)
    extra[:] = [
        entropy,
        entropy / math.log2(20),
        len(counts) / 20,
        fraction("STNQ"),
        fraction("ILV"),
        fraction("CM"),
        fraction("AGST"),
        fraction("GPSND"),
        fraction("GP"),
        longest / n,
        repeats / pairs,
        float(sequence[0] in "KRH"),
        float(sequence[-1] in "DE"),
        HYDROPATHY[sequence[0]],
        HYDROPATHY[sequence[-1]],
        *moments,
        aromatic / pairs,
        opposite / pairs,
        hydro_change / pairs,
    ]
    return {"valid": bool(np.isfinite(extra).all()), "dpc": dpc, "sequence_extra": extra}


def isolated_batch(smiles, timeout=90):
    if not smiles:
        return []
    reason = "native_failure_review"
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(smiles),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        if process.returncode == 0:
            result = json.loads(process.stdout)
            if len(result) != len(smiles):
                raise ValueError("worker result length mismatch")
            for r in result:
                r["values"] = {
                    k: np.asarray(
                        v,
                        dtype="float32"
                        if k in {"rdkit_full", "descriptors", "graph_modifications"}
                        else "uint8",
                    )
                    for k, v in r["values"].items()
                }
            return result
    except subprocess.TimeoutExpired:
        reason = "feature_timeout_review"
    except (OSError, ValueError):
        pass
    if len(smiles) == 1:
        source = smiles[0]
        return [
            invalid_structure(
                reason, isinstance(source, dict) and source.get("include_base", False)
            )
        ]
    middle = len(smiles) // 2
    return isolated_batch(smiles[:middle], timeout) + isolated_batch(smiles[middle:], timeout)


def open_array(path, shape, dtype):
    path = Path(path)
    if path.exists():
        result = np.load(path, mmap_mode="r+")
        if result.shape != tuple(shape) or result.dtype != np.dtype(dtype):
            raise ValueError("Expanded cache shape/dtype mismatch: " + str(path))
        return result
    if not shape[0]:
        result = np.zeros(shape, dtype=dtype)
        np.save(path, result, allow_pickle=False)
        return result
    result = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    result[:] = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    result.flush()
    return result


def flush(arrays):
    for array in arrays:
        if hasattr(array, "flush"):
            array.flush()


def labeled_rows(dataset_dir, enriched_rows=()):
    """Retain every labelled row; representations, not targets, control masks."""
    fields = (
        "record_id",
        "origin",
        "compound_id",
        "source_record_id",
        "endpoint",
        "sequence",
        "sequence_kind",
        "source_entry_id",
    )
    records = {}
    for lane in ("strict_numeric", "binary_evidence"):
        for row in rows(Path(dataset_dir) / "lanes" / (lane + ".tsv")):
            rid = row["record_id"]
            if rid in records:
                raise ValueError("Duplicate labelled record ID")
            records[rid] = {k: row.get(k, "") for k in fields}
            records[rid]["label_lane"] = lane
    for gate in rows(Path(dataset_dir) / "representation_gates.tsv"):
        record = records.get(gate["record_id"])
        if record is not None:
            record.update(
                sequence_eligible=gate["sequence_model_eligible"],
                structure_eligible=gate["structure_model_eligible"],
                canonical_identity_group=gate["canonical_identity_group"],
            )
    old_ids = {r["source_record_id"] for r in records.values() if r["origin"] == "OLD"}
    new_ids = {r["compound_id"] for r in records.values() if r["origin"] == "NEW"}
    members = {}
    for row in rows(Path(dataset_dir) / "structure_qc/structure_members.tsv"):
        key = row["source_record_id"] if row["origin"] == "OLD" else row["compound_id"]
        if key in (old_ids if row["origin"] == "OLD" else new_ids):
            members[row["origin"], key] = row["smiles_sha256"]
    for record in records.values():
        key = record["source_record_id"] if record["origin"] == "OLD" else record["compound_id"]
        record["structure_smiles_sha256"] = members.get((record["origin"], key), "")
        record.setdefault("sequence_eligible", "false")
        record.setdefault("structure_eligible", "false")
        record.setdefault("canonical_identity_group", "")
    for rich in enriched_rows:
        record = records.get(rich["record_id"])
        if record is None:
            raise ValueError("Enriched row is not in frozen labelled data")
        if (
            record["sequence_eligible"] == "true"
            and rich.get("sequence_eligible") == "true"
            and rich.get("sequence")
            and record["sequence"] != rich["sequence"]
        ):
            raise ValueError("Conflicting enriched parent sequence")
        if (
            record["sequence_eligible"] != "true"
            and str(rich.get("sequence_eligible", "")).lower() == "true"
            and rich.get("sequence")
            and not set(rich["sequence"]) - set(AA)
        ):
            record.update(
                sequence=rich["sequence"],
                sequence_kind="parent_projection",
                sequence_eligible="true",
                sequence_recovery="compiled_source_contract",
            )
    return list(records.values())


def expand_features(base_run, dataset_dir, output_run, rows_index=None):
    from peptide_modeling_features import _resolve_structures

    base_run, dataset_dir, output_run = (
        Path(x).resolve() for x in (base_run, dataset_dir, output_run)
    )
    if output_run == base_run or output_run.is_relative_to(base_run / "features"):
        raise ValueError("Expanded output must be independent of source cache")
    original = read_json(base_run / "features/manifest.json")
    source_cache = (
        base_run / "features" / str(original["cache_dir"]).replace("\\", "/").split("/")[-1]
    )
    source_hashes = {}
    if not original.get("artifact_hashes") or "esm2" not in original["groups"]:
        raise ValueError("Verified fresh ESM feature base is required")
    for relative, expected in original["artifact_hashes"].items():
        source = safe_path(source_cache, relative)
        if sha(source) != expected:
            raise ValueError("Source cache artifact changed: " + relative)
        source_hashes[relative] = expected
    task_manifest = read_json(base_run / "task_manifest.json")
    tasks = [t for t in task_manifest["tasks"] if t["status"] == "eligible"]
    enriched_rows = {}
    source_files = ["task_manifest.json", "task_summary.tsv", "deferred_records.tsv"]
    for task in tasks:
        source_files.append(task["table_path"])
        source_files.append(f"splits/{task['task_id']}/group_assignment.json")
        for row in rows(base_run / task["table_path"]):
            enriched_rows[row["record_id"]] = row
    if rows_index is None:
        task_rows = labeled_rows(dataset_dir, enriched_rows.values())
    else:
        supplied = list(rows_index.values()) if isinstance(rows_index, dict) else list(rows_index)
        task_rows = labeled_rows(dataset_dir, supplied)
    output_run.mkdir(parents=True, exist_ok=True)
    for relative in source_files:
        destination = (
            (output_run / "source_outer_splits" / relative.removeprefix("splits/"))
            if relative.startswith("splits/")
            else (output_run / "source_priority_tasks" / relative)
        )
        copy_verified(safe_path(base_run, relative), destination)
    structure_records = list(rows(source_cache / original["groups"]["morgan"]["index"]))
    sequence_records = list(rows(source_cache / original["groups"]["sequence"]["index"]))
    structure_records.sort(key=lambda r: int(r["row_idx"]))
    sequence_records.sort(key=lambda r: int(r["row_idx"]))
    structures, structure_aliases = _resolve_structures(dataset_dir, task_rows)
    keys = sorted(structures)
    old_structure_index = {r["key"]: int(r["row_idx"]) for r in structure_records}
    if not set(old_structure_index) <= set(keys):
        raise ValueError("Expanded labelled structure universe lost original task structures")
    sequences = sorted(
        {
            r["sequence"]
            for r in task_rows
            if r["sequence"]
            and r["sequence_eligible"] == "true"
            and not set(r["sequence"]) - set(AA)
        }
        | {r["key"] for r in sequence_records}
    )
    print(
        json.dumps(
            {
                "phase": "label_bank",
                "records": len(task_rows),
                "endpoints": len({r["endpoint"] for r in task_rows}),
                "structures": len(keys),
                "sequences": len(sequences),
            }
        ),
        flush=True,
    )
    defs = definitions()
    configuration = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "base_manifest_sha256": sha(base_run / "features/manifest.json"),
        "base_input_fingerprint": original["input_fingerprint"],
        "dataset_manifest_sha256": sha(dataset_dir / "manifest.json"),
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "feature_names": {k: v[2] for k, v in defs.items()},
        "descriptor_transforms": DESCRIPTOR_TRANSFORMS,
        "descriptor_cost_guards": {"Ipc_AvgIpc_max_atoms": 128, "BCUT2D_max_atoms": 256},
        "rdkit_max_missing_fraction": 0.1,
        "max_structure_atoms": 1000,
        "fingerprints": {
            "includeChirality": True,
            "morgan_radius": 3,
            "fpSize": 2048,
            "atompair_use2D": True,
            "atompair_countSimulation": False,
            "torsion_atom_count": 4,
        },
        "sequence_features_scope": "explicit_parent_sequence_proxies_not_chemical_properties",
        "labeled_record_count": len(task_rows),
        "record_representation_sha256": signature(
            sorted(
                (
                    r["record_id"],
                    r["sequence"],
                    r["sequence_kind"],
                    r["sequence_eligible"],
                    r["canonical_identity_group"],
                    r["structure_smiles_sha256"],
                    r["structure_eligible"],
                )
                for r in task_rows
            )
        ),
        "sequence_hydropathy_correction": {"V": 4.2, "W": -0.9, "Y": -1.3},
        "legacy_hydropathy_mapping_preserved_in_source": {"V": -0.9, "W": -1.3, "Y": 4.2},
    }
    fingerprint = signature(configuration)
    cache = output_run / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists() and (
        output_run / "feature_expansion_receipt.json"
    ).exists():
        manifest = read_json(cache / "manifest.json")
        for relative, expected in manifest["artifact_hashes"].items():
            if sha(cache / relative) != expected:
                raise ValueError("Expanded cached artifact changed")
        write_json(output_run / "features/manifest.json", manifest)
        return manifest
    for source in source_cache.rglob("*"):
        if source.is_file() and source.name != "manifest.json":
            copy_verified(source, cache / source.relative_to(source_cache))
    copy_verified(source_cache / "manifest.json", cache / "source_feature_manifest.json")
    write_json(cache / "expansion_pending.json", configuration)
    matrices, valids = {}, {}
    expanded_groups = {}
    new_structure_index = {key: i for i, key in enumerate(keys)}
    for name in BASE_STRUCTURE_GROUPS:
        group = copy.deepcopy(original["groups"][name])
        dim, dtype = group["shape"][1], group["dtype"]
        matrices[name] = open_array(cache / f"bank_{name}.npy", (len(keys), dim), dtype)
        valids[name] = open_array(cache / f"bank_{name}_valid.npy", (len(keys),), "bool")
        source_mat = np.load(source_cache / group["matrix"], mmap_mode="r")
        source_valid = np.load(source_cache / group["valid"], mmap_mode="r")
        dest_positions = [new_structure_index[key] for key in old_structure_index]
        source_positions = list(old_structure_index.values())
        matrices[name][dest_positions] = source_mat[source_positions]
        valids[name][dest_positions] = source_valid[source_positions]
        group.update(
            matrix=f"bank_{name}.npy",
            valid=f"bank_{name}_valid.npy",
            index="structure_bank_index.tsv",
            shape=[len(keys), dim],
        )
        expanded_groups[name] = group
    sequence_group = copy.deepcopy(original["groups"]["sequence"])
    dim = sequence_group["shape"][1]
    matrices["sequence"] = open_array(cache / "bank_sequence.npy", (len(sequences), dim), "float32")
    valids["sequence"] = open_array(cache / "bank_sequence_valid.npy", (len(sequences),), "bool")
    sequence_group.update(
        matrix="bank_sequence.npy",
        valid="bank_sequence_valid.npy",
        index="sequence_bank_index.tsv",
        shape=[len(sequences), dim],
        hydropathy_correction={"V": 4.2, "W": -0.9, "Y": -1.3},
    )
    expanded_groups["sequence"] = sequence_group
    for name, (dim, dtype, names) in defs.items():
        is_sequence = name in {"dpc", "sequence_extra"}
        count = len(sequences) if is_sequence else len(keys)
        matrices[name] = open_array(cache / f"{name}.npy", (count, dim), dtype)
        valids[name] = open_array(cache / f"{name}_valid.npy", (count,), "bool")
        expanded_groups[name] = {
            "matrix": f"{name}.npy",
            "valid": f"{name}_valid.npy",
            "index": "sequence_bank_index.tsv" if is_sequence else "structure_bank_index.tsv",
            "key_kind": "sequence" if is_sequence else "canonical_identity_group",
            "feature_names": names,
            "dtype": dtype,
            "shape": [count, dim],
        }
    expanded_groups["rdkit_full"].update(
        allow_partial_missing=True,
        max_missing_fraction=0.1,
        raw_descriptor_names=[n for n, _ in rdkit_tools()["descriptor_functions"]],
        column_transforms=DESCRIPTOR_TRANSFORMS,
    )
    done = open_array(cache / "expanded_structure_done.npy", (len(keys),), "bool")
    pending = [
        (i, {"smiles": structures[key], "include_base": key not in old_structure_index})
        for i, key in enumerate(keys)
        if not done[i]
    ]
    batches = [pending[i : i + 32] for i in range(0, len(pending), 32)]
    with (cache / "expanded_structure_errors.jsonl").open("a", encoding="utf-8") as errors:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(isolated_batch, [s for _, s in b]): b for b in batches}
            completed = 0
            for future in as_completed(futures):
                batch = futures[future]
                extracted = future.result()
                for (index, _), result in zip(batch, extracted, strict=True):
                    for name in result["values"]:
                        matrices[name][index] = result["values"][name]
                        valids[name][index] = result["valid"][name]
                    done[index] = True
                    if result["errors"]:
                        errors.write(
                            json.dumps({"key": keys[index], "errors": result["errors"]}) + "\n"
                        )
                completed += len(batch)
                flush([*matrices.values(), *valids.values(), done])
                errors.flush()
                if completed % 512 < 32:
                    print(f"Expanded structures {completed}/{len(pending)}", flush=True)
    for index, sequence in enumerate(sequences):
        result = extract_expanded_sequence(sequence)
        for name in ("dpc", "sequence_extra"):
            matrices[name][index] = result[name]
            valids[name][index] = result["valid"]
        from peptide_modeling_features import extract_sequence

        basic = extract_sequence(sequence)
        basic["sequence"][sequence_group["feature_names"].index("mean_hydropathy")] = sum(
            HYDROPATHY[a] for a in sequence
        ) / len(sequence)
        matrices["sequence"][index] = basic["sequence"]
        valids["sequence"][index] = basic["valid"]
    flush([*matrices.values(), *valids.values()])
    manifest = copy.deepcopy(original)
    manifest.update(
        schema=VERSION,
        input_fingerprint=fingerprint,
        cache_dir=str(cache),
        manifest_path=str(cache / "manifest.json"),
        cache_reused=False,
        requested_records=len(task_rows),
        configuration={
            "source_configuration": original["configuration"],
            "expansion": configuration,
        },
    )
    manifest["groups"].update(expanded_groups)
    manifest["coverage"]["esm2"]["expanded_sequence_universe"] = len(sequences)
    manifest["coverage"]["esm2"]["missing_expanded_sequences"] = len(
        set(sequences) - {r["key"] for r in sequence_records}
    )
    manifest["record_index"] = "record_bank_index.tsv"
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        ({"key": key, "row_idx": i} for i, key in enumerate(keys)),
    )
    write_tsv(
        cache / "sequence_bank_index.tsv",
        ["key", "row_idx"],
        ({"key": key, "row_idx": i} for i, key in enumerate(sequences)),
    )
    sequence_set, structure_set = set(sequences), set(keys)
    mapped = []
    for row in task_rows:
        structure = row["canonical_identity_group"] or structure_aliases.get(
            row["structure_smiles_sha256"], ""
        )
        if row["structure_eligible"] != "true" or structure not in structure_set:
            structure = ""
        sequence = (
            row["sequence"]
            if row["sequence_eligible"] == "true" and row["sequence"] in sequence_set
            else ""
        )
        mapped.append(
            {"record_id": row["record_id"], "structure_key": structure, "sequence_key": sequence}
        )
    write_tsv(
        cache / "record_bank_index.tsv", ["record_id", "structure_key", "sequence_key"], mapped
    )
    write_tsv(
        output_run / "labeled_record_index.tsv",
        ["record_id", "endpoint", "label_lane", "structure_eligible", "sequence_eligible"],
        (
            {
                k: row[k]
                for k in (
                    "record_id",
                    "endpoint",
                    "label_lane",
                    "structure_eligible",
                    "sequence_eligible",
                )
            }
            for row in task_rows
        ),
    )
    structural = [*original["routes"]["structure"], *STRUCTURE_GROUPS]
    sequential = ["sequence", "dpc", "sequence_extra"]
    manifest["routes"].update(
        structure_extended=structural,
        sequence_extended=sequential,
        fusion_extended=[*structural, *sequential],
        esm_sequence_extended=[*sequential, "esm2"],
        fusion_esm_extended=[*structural, *sequential, "esm2"],
    )
    for name, values in matrices.items():
        valid = np.asarray(valids[name])
        if np.isinf(values).any() or (name != "rdkit_full" and not np.isfinite(values).all()):
            raise ValueError("Expanded features contain invalid numeric values: " + name)
        manifest["coverage"][name] = {
            "representations": len(valid),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "all_zero_rows": int(np.sum(valid & ~np.any(values != 0, axis=1))),
        }
    full = matrices["rdkit_full"]
    write_tsv(
        cache / "rdkit_column_audit.tsv",
        ["feature", "finite", "missing", "transform"],
        (
            {
                "feature": name,
                "finite": int(np.isfinite(full[:, i]).sum()),
                "missing": int(np.isnan(full[:, i]).sum()),
                "transform": DESCRIPTOR_TRANSFORMS.get(raw, "identity"),
            }
            for i, ((raw, _), name) in enumerate(
                zip(rdkit_tools()["descriptor_functions"], defs["rdkit_full"][2], strict=True)
            )
        ),
    )
    manifest["artifact_hashes"] = {
        f.relative_to(cache).as_posix(): sha(f)
        for f in sorted(cache.rglob("*"))
        if f.is_file() and f.name != "manifest.json"
    }
    write_json(cache / "manifest.json", manifest)
    write_json(output_run / "features/manifest.json", manifest)
    unchanged = all(sha(source_cache / rel) == expected for rel, expected in source_hashes.items())
    split_hashes = {r: sha(base_run / r) for r in source_files if r.startswith("splits/")}
    assertions = {
        "source_feature_artifacts_unchanged": unchanged,
        "outer_splits_byte_identical": all(
            sha(output_run / "source_outer_splits" / r.removeprefix("splits/")) == h
            for r, h in split_hashes.items()
        ),
        "all_structure_rows_terminal": bool(done.all()),
        "no_inf_values": all(not np.isinf(a).any() for a in matrices.values()),
    }
    if not all(assertions.values()):
        raise AssertionError(assertions)
    receipt = {
        "schema": VERSION,
        "status": "complete",
        "configuration": configuration,
        "source_base_run": str(base_run),
        "cache_dir": str(cache),
        "assertions": assertions,
        "frozen_split_sha256": split_hashes,
        "group_dimensions": {k: v["shape"][1] for k, v in manifest["groups"].items()},
        "total_feature_dimensions": sum(v["shape"][1] for v in manifest["groups"].values()),
        "coverage": manifest["coverage"],
        "labeled_records": len(task_rows),
        "labeled_endpoints": len({r["endpoint"] for r in task_rows}),
        "unique_structure_universe": len(keys),
        "unique_sequence_universe": len(sequences),
        "fresh_esm_sequences_available": original["groups"]["esm2"]["shape"][0],
        "mean_hydropathy_corrected_only_in_new_cache": True,
    }
    write_json(output_run / "feature_expansion_receipt.json", receipt)
    config_path = base_run / "config.json"
    if not config_path.exists():
        config_path = base_run / "execution_config.json"
    config = read_json(config_path)
    config["run_dir"] = str(output_run)
    config["training"].update(inner_folds=5, routes=list(manifest["routes"]))
    write_json(output_run / "feature_expansion_config.json", config)
    print(
        json.dumps(
            {
                "status": "complete",
                "group_dimensions": receipt["group_dimensions"],
                "total_feature_dimensions": receipt["total_feature_dimensions"],
            }
        ),
        flush=True,
    )
    return manifest


def main():
    if sys.argv[1:] == ["--worker"]:
        outputs = []
        for source in json.load(sys.stdin):
            result = (
                extract_expanded_structure(
                    source["smiles"], include_base=source.get("include_base", False)
                )
                if isinstance(source, dict)
                else extract_expanded_structure(source)
            )
            result["values"] = {
                k: [float(x) if np.isfinite(x) else None for x in v]
                for k, v in result["values"].items()
            }
            outputs.append(result)
        json.dump(outputs, sys.stdout, allow_nan=False)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    expand_features(args.base_run, args.dataset, args.output_run)


if __name__ == "__main__":
    main()
