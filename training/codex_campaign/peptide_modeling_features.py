"""Audited CPU features with content-addressed caches and explicit missingness.

Source structures must already have passed the independent structure QC. This
module never creates SMILES from a sequence or guesses a missing modification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

VERSION = "peptide_cpu_features_v2"
AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = dict(
    zip(
        AA,
        [
            1.8,
            2.5,
            -3.5,
            -3.5,
            2.8,
            -0.4,
            -3.2,
            4.5,
            -3.9,
            3.8,
            1.9,
            -3.5,
            -1.6,
            -3.5,
            -4.5,
            -0.8,
            -0.7,
            4.2,
            -0.9,
            -1.3,
        ],
        strict=True,
    )
)
# Explicit residue names protect against positional dictionary mistakes. These
# corrected values only affect newly built caches; frozen v1 artifacts stay intact.
HYDROPATHY.update({"V": 4.2, "W": -0.9, "Y": -1.3})
DESCRIPTOR_NAMES = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "HeavyAtoms",
    "Atoms",
    "RingCount",
    "AromaticRings",
    "AliphaticRings",
    "FractionCSP3",
    "FormalCharge",
]
GRAPH_MOD_NAMES = [
    "n_methyl_amide_nitrogens",
    "disulfide_bonds",
    "ester_bonds",
    "assigned_stereocenters",
    "unassigned_stereocenters",
]
SEQUENCE_NAMES = (
    ["length", "log1p_length"]
    + ["aac_" + aa for aa in AA]
    + [
        "basic_residue_fraction",
        "acidic_residue_fraction",
        "aromatic_residue_fraction",
        "hydrophobic_residue_fraction",
        "basic_minus_acidic_fraction",
        "mean_hydropathy",
    ]
)
ERROR_CODES = {
    "": 0,
    "parse_error": 1,
    "atom_limit_review": 2,
    "morgan_error": 3,
    "descriptor_nonfinite_or_error": 4,
    "graph_modification_error": 5,
    "invalid_sequence": 6,
    "native_failure_review": 7,
    "feature_timeout_review": 8,
}
INVALID_SEQUENCE_KINDS = {
    "multichain",
    "ambiguous_sequences",
    "partial_parent_projection",
    "unparsed_or_missing",
    "missing",
    "unparsed_notation",
}


@lru_cache(maxsize=1)
def _rdkit():
    # The training host only loads arrays and must not import its old RDKit build.
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

    descriptors = [
        Descriptors.MolWt,
        Descriptors.ExactMolWt,
        Crippen.MolLogP,
        rdMolDescriptors.CalcTPSA,
        Lipinski.NumHDonors,
        Lipinski.NumHAcceptors,
        Lipinski.NumRotatableBonds,
        lambda mol: mol.GetNumHeavyAtoms(),
        lambda mol: mol.GetNumAtoms(),
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumAromaticRings,
        rdMolDescriptors.CalcNumAliphaticRings,
        rdMolDescriptors.CalcFractionCSP3,
        Chem.GetFormalCharge,
    ]
    patterns = [
        Chem.MolFromSmarts(s)
        for s in (
            "[N;X3]([CH3])[C](=O)",
            "[S;X2]-[S;X2]",
            "[CX3](=O)[OX2][#6]",
        )
    ]
    return {
        "Chem": Chem,
        "DataStructs": DataStructs,
        "version": rdBase.rdkitVersion,
        "descriptors": dict(zip(DESCRIPTOR_NAMES, descriptors, strict=True)),
        "morgan": rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True
        ),
        "patterns": patterns,
    }


def _descriptor(name, mol):
    return _rdkit()["descriptors"][name](mol)


DESCRIPTOR_FUNCTIONS = [
    (name, lambda mol, name=name: _descriptor(name, mol)) for name in DESCRIPTOR_NAMES
]


def _sha(path):
    if not Path(path).is_file():
        return ""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def canonical_key(smiles):
    return "CANON_SMILES:" + hashlib.sha256(smiles.encode()).hexdigest()


def _true(value, default=True):
    if value is None or value == "":
        return default
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _rows(path):
    csv.field_size_limit(10000000)
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def _write(path, fields, values):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(values)


def extract_structure(smiles, max_atoms=1000):
    rdkit = _rdkit()
    Chem, DataStructs = rdkit["Chem"], rdkit["DataStructs"]
    morgan, patterns = rdkit["morgan"], rdkit["patterns"]
    result = {
        "morgan": np.zeros(2048, dtype=np.uint8),
        "descriptors": np.zeros(len(DESCRIPTOR_FUNCTIONS), dtype=np.float32),
        "graph_modifications": np.zeros(len(GRAPH_MOD_NAMES), dtype=np.float32),
        "group_valid": dict.fromkeys(("morgan", "descriptors", "graph_modifications"), False),
        "valid": False,
        "error": "",
    }
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        result["error"] = "parse_error"
        return result
    if mol.GetNumAtoms() > max_atoms:
        result["error"] = "atom_limit_review"
        return result
    errors = []
    try:
        DataStructs.ConvertToNumpyArray(morgan.GetFingerprint(mol), result["morgan"])
        result["group_valid"]["morgan"] = bool(np.any(result["morgan"]))
        if not result["group_valid"]["morgan"]:
            errors.append("morgan_error")
    except Exception:
        errors.append("morgan_error")
    try:
        descriptor = np.array([func(mol) for _, func in DESCRIPTOR_FUNCTIONS], dtype=np.float32)
        if not np.isfinite(descriptor).all():
            raise ValueError("nonfinite_descriptor")
        result["descriptors"] = descriptor
        result["group_valid"]["descriptors"] = True
    except Exception:
        errors.append("descriptor_nonfinite_or_error")
    try:
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        n_methyl = len({match[0] for match in mol.GetSubstructMatches(patterns[0])})
        motifs = [n_methyl] + [len(mol.GetSubstructMatches(p)) for p in patterns[1:]]
        motifs += [
            sum(label != "?" for _, label in centers),
            sum(label == "?" for _, label in centers),
        ]
        result["graph_modifications"] = np.asarray(motifs, dtype=np.float32)
        result["group_valid"]["graph_modifications"] = True
    except Exception:
        errors.append("graph_modification_error")
    result["valid"] = all(result["group_valid"].values())
    result["error"] = ";".join(errors)
    return result


def extract_sequence(sequence):
    if not sequence or set(sequence) - set(AA):
        return {
            "valid": False,
            "error": "invalid_sequence",
            "sequence": np.zeros(len(SEQUENCE_NAMES), dtype=np.float32),
        }
    n = len(sequence)
    counts = Counter(sequence)

    def fraction(letters):
        return sum(counts[a] for a in letters) / n

    basic, acidic = fraction("KRH"), fraction("DE")
    values = (
        [n, np.log1p(n)]
        + [counts[a] / n for a in AA]
        + [
            basic,
            acidic,
            fraction("FWY"),
            fraction("AILMFWV"),
            basic - acidic,
            sum(HYDROPATHY[a] for a in sequence) / n,
        ]
    )
    vector = np.asarray(values, dtype=np.float32)
    return {"valid": bool(np.isfinite(vector).all()), "error": "", "sequence": vector}


def _invalid_structure(error):
    return {
        "morgan": np.zeros(2048, dtype=np.uint8),
        "descriptors": np.zeros(len(DESCRIPTOR_FUNCTIONS), dtype=np.float32),
        "graph_modifications": np.zeros(len(GRAPH_MOD_NAMES), dtype=np.float32),
        "group_valid": dict.fromkeys(("morgan", "descriptors", "graph_modifications"), False),
        "valid": False,
        "error": error,
    }


def _isolated_structure_batch(smiles, timeout=90):
    """Bisect failed native batches so one molecule cannot abort a whole cache."""
    if not smiles:
        return []
    error = "native_failure_review"
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--structure-worker"],
            input=json.dumps(smiles),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if process.returncode == 0:
            extracted = json.loads(process.stdout)
            if len(extracted) != len(smiles):
                raise ValueError("worker_result_count_mismatch")
            for result in extracted:
                for name in ("morgan", "descriptors", "graph_modifications"):
                    result[name] = np.asarray(
                        result[name], dtype="uint8" if name == "morgan" else "float32"
                    )
            return extracted
    except subprocess.TimeoutExpired:
        error = "feature_timeout_review"
    except (OSError, ValueError):
        pass
    if len(smiles) == 1:
        return [_invalid_structure(error)]
    middle = len(smiles) // 2
    return _isolated_structure_batch(smiles[:middle], timeout) + _isolated_structure_batch(
        smiles[middle:], timeout
    )


@lru_cache(maxsize=256)
def _checked_artifact(path, expected_hash, size, modified_ns):
    del size, modified_ns
    if _sha(path) != expected_hash:
        raise ValueError("Feature cache artifact hash mismatch: " + path)


def _verify_artifacts(manifest):
    base = Path(manifest["cache_dir"])
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        path = base / relative
        stat = path.stat()
        _checked_artifact(str(path), expected, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=128)
def _index_cached(path, modified_ns, size):
    del modified_ns, size
    return {r["key"]: int(r["row_idx"]) for r in _rows(path)}


@lru_cache(maxsize=16)
def _records_cached(path, modified_ns, size):
    del modified_ns, size
    return {r["record_id"]: r for r in _rows(path)}


def _task_rows(record_task_paths, rows_index, run_dir):
    if rows_index is not None:
        source = rows_index.values() if isinstance(rows_index, dict) else rows_index
    else:
        paths = record_task_paths or sorted((Path(run_dir) / "task_tables").glob("*.tsv"))
        if isinstance(paths, (str, Path)):
            paths = [paths]
        source = (row for path in paths for row in _rows(path))
    result = {}
    for row in source:
        record = str(row["record_id"])
        if record in result:
            for field in ("sequence", "canonical_identity_group", "structure_smiles_sha256"):
                if result[record].get(field, "") != row.get(field, ""):
                    raise ValueError(f"Conflicting representation for {record}: {field}")
        result[record] = dict(row)
    return list(result.values())


def _resolve_structures(dataset_dir, task_rows):
    requested_hashes = {r.get("structure_smiles_sha256", "") for r in task_rows} - {""}
    requested_keys = {r.get("canonical_identity_group", "") for r in task_rows} - {""}
    structures, aliases = {}, {}
    path = Path(dataset_dir) / "structure_qc/structure_validation.tsv"
    if not path.exists() and not (requested_hashes or requested_keys):
        return structures, aliases
    if not path.exists():
        raise ValueError("Independent structure_validation.tsv is required for structure features")
    for row in _rows(path):
        if not row["status"].startswith("parsed") or not row["canonical_smiles"]:
            continue
        key = canonical_key(row["canonical_smiles"])
        if row["smiles_sha256"] in requested_hashes or key in requested_keys:
            structures[key] = row["canonical_smiles"]
            aliases[row["smiles_sha256"]] = key
    return structures, aliases


def _create_or_load(path, shape, dtype):
    if path.exists():
        result = np.load(path, mmap_mode="r+")
        if result.shape != tuple(shape) or result.dtype != np.dtype(dtype):
            raise ValueError("Cache array shape/dtype mismatch: " + str(path))
        return result
    if not shape[0]:
        np.save(path, np.zeros(shape, dtype=dtype), allow_pickle=False)
        return np.load(path)
    result = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    result[:] = 0
    result.flush()
    return result


def _flush(arrays):
    for array in arrays:
        if hasattr(array, "flush"):
            array.flush()


def build_features(dataset_dir, run_dir, record_task_paths=None, rows_index=None):
    """Compute only task-requested representations; resume a matching cache safely."""
    dataset_dir, run_dir = Path(dataset_dir), Path(run_dir)
    task_rows = _task_rows(record_task_paths, rows_index, run_dir)
    structures, structure_aliases = _resolve_structures(dataset_dir, task_rows)
    sequences = sorted(
        {
            r.get("sequence", "")
            for r in task_rows
            if r.get("sequence")
            and not (set(r["sequence"]) - set(AA))
            and r.get("sequence_kind", "") not in INVALID_SEQUENCE_KINDS
            and _true(r.get("sequence_eligible"))
        }
    )
    structure_keys = sorted(structures)
    configuration = {
        "version": VERSION,
        "rdkit_version": _rdkit()["version"],
        "numpy_version": np.__version__,
        "implementation_sha256": _sha(Path(__file__)),
        "dataset_manifest_sha256": _sha(dataset_dir / "manifest.json"),
        "structure_qc_sha256": _sha(dataset_dir / "structure_qc/structure_validation.tsv"),
        "morgan": {"radius": 2, "fpSize": 2048, "includeChirality": True},
        "descriptor_names": [name for name, _ in DESCRIPTOR_FUNCTIONS],
        "graph_modification_names": GRAPH_MOD_NAMES,
        "sequence_names": SEQUENCE_NAMES,
        "max_structure_atoms": 1000,
        "representations_sha256": _hash([[(k, structures[k]) for k in structure_keys], sequences]),
        "records_sha256": _hash(
            sorted(
                (
                    str(r["record_id"]),
                    r.get("sequence", ""),
                    r.get("sequence_kind", ""),
                    r.get("canonical_identity_group", ""),
                    r.get("structure_smiles_sha256", ""),
                    str(r.get("sequence_eligible", "")),
                    str(r.get("structure_eligible", "")),
                )
                for r in task_rows
            )
        ),
    }
    fingerprint = _hash(configuration)
    cache_root = run_dir / "features"
    cache = cache_root / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        if result["input_fingerprint"] != fingerprint:
            raise ValueError("Cache fingerprint collision")
        _verify_artifacts(result)
        result["cache_reused"] = True
        (cache_root / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    (cache / "pending.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    structure_index = {key: i for i, key in enumerate(structure_keys)}
    sequence_index = {key: i for i, key in enumerate(sequences)}
    groups = {}
    matrices, valid_arrays = {}, {}
    definitions = {
        "morgan": (2048, "uint8", [f"bit_{i}" for i in range(2048)]),
        "descriptors": (len(DESCRIPTOR_FUNCTIONS), "float32", [n for n, _ in DESCRIPTOR_FUNCTIONS]),
        "graph_modifications": (len(GRAPH_MOD_NAMES), "float32", GRAPH_MOD_NAMES),
        "sequence": (len(SEQUENCE_NAMES), "float32", SEQUENCE_NAMES),
    }
    for name, (dim, dtype, names) in definitions.items():
        is_sequence = name == "sequence"
        n = len(sequences) if is_sequence else len(structure_keys)
        matrices[name] = _create_or_load(cache / f"{name}.npy", (n, dim), dtype)
        valid_arrays[name] = _create_or_load(cache / f"{name}_valid.npy", (n,), "bool")
        groups[name] = {
            "matrix": f"{name}.npy",
            "valid": f"{name}_valid.npy",
            "index": "sequence_index.tsv" if is_sequence else "structure_index.tsv",
            "key_kind": "sequence" if is_sequence else "canonical_identity_group",
            "feature_names": names,
            "dtype": dtype,
            "shape": [n, dim],
        }
    done_structures = _create_or_load(cache / "structure_done.npy", (len(structure_keys),), "bool")
    error_codes = _create_or_load(cache / "structure_errors.npy", (len(structure_keys),), "int16")
    errors = []
    pending = [(key, i) for key, i in structure_index.items() if not done_structures[i]]
    for start in range(0, len(pending), 128):
        batch = pending[start : start + 128]
        extracted_batch = _isolated_structure_batch([structures[key] for key, _ in batch])
        for (_, i), extracted in zip(batch, extracted_batch, strict=True):
            for name in ("morgan", "descriptors", "graph_modifications"):
                matrices[name][i] = extracted[name]
                valid_arrays[name][i] = extracted["group_valid"][name]
            error = extracted["error"].split(";")[0]
            error_codes[i] = ERROR_CODES.get(error, 99)
            done_structures[i] = True
        _flush(
            list(matrices.values()) + list(valid_arrays.values()) + [error_codes, done_structures]
        )
    for sequence, i in sequence_index.items():
        extracted = extract_sequence(sequence)
        matrices["sequence"][i] = extracted["sequence"]
        valid_arrays["sequence"][i] = extracted["valid"]
    _flush(list(matrices.values()) + list(valid_arrays.values()) + [error_codes, done_structures])
    error_names = {v: k for k, v in ERROR_CODES.items()}
    for key, i in structure_index.items():
        if error_codes[i]:
            errors.append(
                {"key": key, "error": error_names.get(int(error_codes[i]), "unknown_error")}
            )
    _write(cache / "structure_errors.tsv", ["key", "error"], errors)
    _write(
        cache / "structure_index.tsv",
        ["key", "row_idx"],
        ({"key": key, "row_idx": i} for key, i in structure_index.items()),
    )
    _write(
        cache / "sequence_index.tsv",
        ["key", "row_idx"],
        ({"key": key, "row_idx": i} for key, i in sequence_index.items()),
    )
    record_map = []
    for row in task_rows:
        key = row.get("canonical_identity_group", "")
        if key not in structure_index:
            key = structure_aliases.get(row.get("structure_smiles_sha256", ""), "")
        if not _true(row.get("structure_eligible")):
            key = ""
        sequence = row.get("sequence", "")
        if row.get("sequence_kind", "") in INVALID_SEQUENCE_KINDS or not _true(
            row.get("sequence_eligible")
        ):
            sequence = ""
        record_map.append(
            {
                "record_id": row["record_id"],
                "structure_key": key,
                "sequence_key": sequence if sequence in sequence_index else "",
            }
        )
    _write(cache / "record_index.tsv", ["record_id", "structure_key", "sequence_key"], record_map)
    coverage = {}
    for name, values in matrices.items():
        valid = valid_arrays[name]
        if np.any(valid) and not np.isfinite(values[valid]).all():
            raise ValueError("Nonfinite values marked valid: " + name)
        if (
            name in {"morgan", "descriptors", "sequence"}
            and np.any(valid)
            and not np.any(values[valid])
        ):
            raise ValueError("An entire required feature group is zero: " + name)
        coverage[name] = {
            "representations": len(valid),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "all_zero_rows": int(np.sum(valid & ~np.any(values != 0, axis=1))),
        }
    routes = {
        "sequence": ["sequence"],
        "structure": ["morgan", "descriptors", "graph_modifications"],
        "fusion": ["morgan", "descriptors", "graph_modifications", "sequence"],
    }
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "configuration": configuration,
        "groups": groups,
        "routes": routes,
        "coverage": coverage,
        "record_index": "record_index.tsv",
        "requested_records": len(task_rows),
        "cache_reused": False,
        "scope": (
            "CPU feature extraction only; source annotations are not guessed; "
            "graph motifs do not infer D/L residues"
        ),
    }
    manifest["artifact_hashes"] = {
        path.name: _sha(path) for path in sorted(cache.iterdir()) if path.suffix in {".npy", ".tsv"}
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _valid_feature_values(values, group):
    finite = np.isfinite(values)
    if np.isinf(values).any() or not finite.any():
        return False
    if not group.get("allow_partial_missing", False):
        return bool(finite.all())
    limit = float(group.get("max_missing_fraction", 0.1))
    if not 0 <= limit < 1:
        raise ValueError("Invalid feature missingness contract")
    return bool(1.0 - float(finite.mean()) <= limit + 1e-12)


def load_feature_route(task_rows, cache, route):
    """Return all input rows in order; callers must use the explicit valid mask."""
    if isinstance(cache, dict):
        manifest = cache
    else:
        path = Path(cache)
        if path.is_dir():
            path = path / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
    base = Path(manifest["cache_dir"])
    _verify_artifacts(manifest)
    group_names = manifest["routes"][route] if isinstance(route, str) else list(route)
    records = list(task_rows)
    record_path = base / manifest["record_index"]
    stat = record_path.stat()
    record_index = _records_cached(str(record_path), stat.st_mtime_ns, stat.st_size)
    result_valid = np.ones(len(records), dtype=bool)
    errors = [{} for _ in records]
    parts, names = [], []
    for group_name in group_names:
        group = manifest["groups"][group_name]
        index_path = base / group["index"]
        stat = index_path.stat()
        index = _index_cached(str(index_path), stat.st_mtime_ns, stat.st_size)
        mat = np.load(base / group["matrix"], mmap_mode="r")
        group_valid = np.load(base / group["valid"], mmap_mode="r")
        if mat.shape != tuple(group["shape"]) or mat.shape[1] != len(group["feature_names"]):
            raise ValueError("Feature schema mismatch: " + group_name)
        values = np.full((len(records), mat.shape[1]), np.nan, dtype=np.float32)
        valid = np.zeros(len(records), dtype=bool)
        for i, record in enumerate(records):
            mapped = record_index.get(str(record["record_id"]), {})
            is_sequence = group["key_kind"] == "sequence"
            if group["key_kind"] == "record_id":
                key = str(record["record_id"])
                allowed = _true(
                    record.get(group.get("eligibility_field", "record_context_eligible")),
                    default=False,
                )
            else:
                key = mapped.get("sequence_key" if is_sequence else "structure_key")
                if key is None:
                    key = record.get("sequence" if is_sequence else "canonical_identity_group", "")
                allowed = _true(
                    record.get("sequence_eligible" if is_sequence else "structure_eligible")
                )
            if is_sequence and record.get("sequence_kind", "") in INVALID_SEQUENCE_KINDS:
                allowed = False
            pos = index.get(key)
            if (
                allowed
                and pos is not None
                and group_valid[pos]
                and _valid_feature_values(mat[pos], group)
            ):
                values[i] = mat[pos]
                valid[i] = True
            else:
                errors[i][group_name] = "missing_ineligible_or_failed_representation"
        result_valid &= valid
        parts.append(values)
        names.extend(group_name + ":" + name for name in group["feature_names"])
    if not parts:
        raise ValueError("At least one feature group is required")
    return {
        "X": np.hstack(parts),
        "valid": result_valid,
        "feature_names": names,
        "record_ids": [str(r["record_id"]) for r in records],
        "errors": errors,
    }


if __name__ == "__main__":
    if sys.argv[1:] != ["--structure-worker"]:
        raise SystemExit("This module is imported by the modeling pipeline")
    inputs = json.load(sys.stdin)
    outputs = []
    for source in inputs:
        result = extract_structure(source)
        outputs.append(
            {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in result.items()
            }
        )
    json.dump(outputs, sys.stdout, allow_nan=False)
