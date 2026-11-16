"""Precompute explicit DRAMP database-reference structures; never admit training."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import shutil
import sys
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
from peptide_modeling_add_smiles_methods import rdkit_tools  # noqa: E402
from peptide_modeling_clint_features import (  # noqa: E402
    STRUCTURE_GROUPS,
    atomic_array,
    extract_all,
)
from peptide_modeling_expand_features import (  # noqa: E402
    definitions,
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import (  # noqa: E402
    _verify_artifacts,
    canonical_key,
    load_feature_route,
)
from peptide_modeling_qualitative_cyto_features import check_matrix  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERSION = "dramp_literal_binary_reference_features_v1"
IDENTITY_TIER = "database_declared_literal_reference_only"
EVIDENCE_FILE = "fifty_two_source_objects_and_reference_graphs.json"
EVIDENCE_SHA = "b8d162ae695a991a60160e3c633bcf1ea09ca35cc67e92783b14333e7082ceeb"
BASELINE_SHA = "fef174451e147a5c1e7388bd7dcb8869974743a655437b65cbda462d787e12f7"
ALGORITHMS = (
    "peptide_modeling_clint_features.py",
    "peptide_modeling_expand_features.py",
    "peptide_modeling_add_methods.py",
    "peptide_modeling_add_smiles_methods.py",
)
SNAPSHOT_FILES = (
    EVIDENCE_FILE,
    "fifty_two_dispositions.json",
    "collection_receipt.json",
    "species_and_assay_material_role_check.json",
    "compatibility_assessment.json",
    "specific_PACAP_quantitative_mapping_opportunity.json",
    "delivery_receipt.json",
)


def plan_records(objects):
    """Validate literal chemistry independently of flags, then deduplicate graphs only."""
    records, graphs, seen = [], {}, set()
    chemistry = {
        "N-terminal_Modification": "Free",
        "C-terminal_Modification": "Free",
        "Other_Modifications": "None",
        "Linear/Cyclic/Branched": "Linear",
        "Stereochemistry": "L",
    }
    chem = rdkit_tools()["Chem"]
    for obj in objects:
        source, raw, ref = obj["native_source_object"], obj["record"], obj["reference_graph"]
        rid = raw["record_id"]
        if rid in seen or raw["value"] != "0" or raw["unit"] != "label01":
            raise ValueError("Repeated record or changed original zero label/unit")
        seen.add(rid)
        sequence = str(source.get("Sequence") or "")
        eligible = (
            all(source.get(k) == v for k, v in chemistry.items())
            and bool(sequence)
            and not set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
            and "C" not in sequence
            and source.get("Sequence_Length") == len(sequence)
        )
        if bool(ref) != eligible:
            raise ValueError("Unresolved chemistry cannot acquire a reference graph")
        key = ""
        if eligible:
            molecule = chem.MolFromFASTA(sequence)
            canonical = chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            key = canonical_key(canonical)
            if (
                ref["canonical_isomeric_smiles"] != canonical
                or ref["graph_key"] != key
                or ref["sample_purity_or_mature_assay_form_independently_verified"] is not False
            ):
                raise ValueError("Source reference graph/caps/stereo or identity tier changed")
            graphs[key] = {"key": key, "canonical_isomeric_smiles": canonical}
        records.append(
            {
                "record_id": rid,
                "raw_endpoint": raw["endpoint"],
                "source_entry_id": raw["source_entry_id"],
                "raw_label": "0",
                "raw_label_unit": "label01",
                "structure_key": key,
                "canonical_identity_group": key,
                "representation_eligible": str(eligible).lower(),
                "structure_eligible": str(eligible).lower(),
                "sequence_eligible": "false",
                "sequence": "",
                "training_admitted": "false",
                "identity_tier": IDENTITY_TIER if eligible else "unresolved_no_representation",
                "primary_assay_sample_identity_verified": "false",
                "raw_family_placeholder": obj["original_frozen_feature_identity"]["split_group"],
                "raw_family_placeholder_validated": "false",
                "source_xlsx_file": obj["source_xlsx_file"],
                "source_xlsx_row": obj["source_xlsx_row"],
                "identity_reason_codes": json.dumps(obj["identity_blockers"], ensure_ascii=False),
            }
        )
    return records, [graphs[k] for k in sorted(graphs)]


def read_source_plan(evidence_dir):
    evidence_dir = Path(evidence_dir)
    if sha(evidence_dir / EVIDENCE_FILE) != EVIDENCE_SHA:
        raise ValueError("Accepted 52-record source evidence SHA changed")
    receipt = read_json(evidence_dir / "collection_receipt.json")
    for relative, expected in receipt["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Pinned original source changed: " + relative)
    objects = read_json(evidence_dir / EVIDENCE_FILE)
    records, graphs = plan_records(objects)
    if len(records) != 52 or len(graphs) != 38:
        raise ValueError("Expected 52 ledger rows and 38 unique literal reference graphs")
    return objects, records, graphs


def without_loader_ast(path):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    module.body = [
        n
        for n in module.body
        if not (isinstance(n, ast.FunctionDef) and n.name == "load_feature_route")
    ]
    return signature(ast.dump(module, include_attributes=False))


def algorithm_contract(base, frozen_baseline):
    """Reject extractor drift; explicitly isolate the known record-context loader change."""
    if base["configuration"]["rdkit_version"] != rdkit_tools()["version"]:
        raise ValueError("RDKit extractor version differs from parent")
    names = definitions()["rdkit_full"][2]
    if len(names) != 217 or names != base["groups"]["rdkit_full"]["feature_names"]:
        raise ValueError("All 217 RDKit descriptor names/order must match parent")
    cache = Path(base["cache_dir"])
    relative = "source_snapshots/clint/manifest.json"
    snapshot = safe_path(cache, relative)
    if sha(snapshot) != base["source_snapshot_hashes"][relative]:
        raise ValueError("Pinned parent algorithm snapshot changed")
    pins = read_json(snapshot)["configuration"]["implementation_sha256"]
    current = {name: sha(ROOT / "scripts" / name) for name in ALGORITHMS}
    if any(current[name] != pins[name] for name in ALGORITHMS):
        raise ValueError("Structural extraction algorithm source SHA differs from parent")
    if sha(frozen_baseline) != BASELINE_SHA or pins["peptide_modeling_features.py"] != BASELINE_SHA:
        raise ValueError("Frozen baseline source snapshot does not match parent")
    current_baseline = ROOT / "scripts/peptide_modeling_features.py"
    old_ast, new_ast = without_loader_ast(frozen_baseline), without_loader_ast(current_baseline)
    if old_ast != new_ast:
        raise ValueError("Baseline extraction source changed beyond the loader")
    return {
        "rdkit_version": rdkit_tools()["version"],
        "descriptor_columns_checked": 217,
        "extractor_source_sha256": current,
        "frozen_baseline_sha256": BASELINE_SHA,
        "current_baseline_sha256": sha(current_baseline),
        "extractor_module_ast_without_loader_sha256": old_ast,
        "only_allowed_source_container_difference": (
            "record_id-context loader; extraction AST identical"
        ),
    }


def reuse_partition(graphs, parent_graphs, parent_index):
    reused, missing = [], []
    for row in graphs:
        key = row["key"]
        if canonical_key(row["canonical_isomeric_smiles"]) != key:
            raise ValueError("Graph key does not identify its exact canonical SMILES")
        if key in parent_index:
            if (
                parent_graphs.get(key, {}).get("canonical_isomeric_smiles")
                != row["canonical_isomeric_smiles"]
            ):
                raise ValueError("Same parent key has different complete SMILES")
            reused.append(row)
        else:
            missing.append(row)
    return reused, missing


def checked_vector(value, valid, metadata, name):
    raw = np.asarray(value)
    if not valid or raw.shape != (metadata["shape"][1],):
        raise ValueError("Method failed or wrong feature dimension: " + name)
    if np.isinf(raw).any() or np.isnan(raw).all():
        raise ValueError("Nonfinite/all-missing structural row: " + name)
    dtype = np.dtype(metadata["dtype"])
    if np.issubdtype(dtype, np.integer) and (
        not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all()
    ):
        raise ValueError("Cannot silently cast missing/fractional fingerprint to zeros")
    vector = np.asarray(raw, dtype=dtype)
    check_matrix(vector[None, :], np.asarray([True]), metadata, name)
    return vector


def canary_check(base, reused, parent_index, extractor=extract_all):
    chem = rdkit_tools()["Chem"]
    ordered = sorted(
        reused, key=lambda r: chem.MolFromSmiles(r["canonical_isomeric_smiles"]).GetNumAtoms()
    )
    if not ordered:
        raise ValueError("At least one exact parent graph is needed for algorithm canaries")
    selected = {ordered[0]["key"]: ordered[0], ordered[-1]["key"]: ordered[-1]}
    middle = min(
        ordered,
        key=lambda r: abs(chem.MolFromSmiles(r["canonical_isomeric_smiles"]).GetNumAtoms() - 180),
    )
    selected[middle["key"]] = middle
    arrays = {
        n: np.load(safe_path(base["cache_dir"], base["groups"][n]["matrix"]), mmap_mode="r")
        for n in STRUCTURE_GROUPS
    }
    proof = []
    for row in selected.values():
        first, masks, _ = extractor(row["canonical_isomeric_smiles"], row["key"])
        second, masks2, _ = extractor(row["canonical_isomeric_smiles"], row["key"])
        for name in STRUCTURE_GROUPS:
            meta = base["groups"][name]
            one = checked_vector(first[name], masks[name], meta, name)
            two = checked_vector(second[name], masks2[name], meta, name)
            old = arrays[name][parent_index[row["key"]]]
            if one.tobytes() != two.tobytes() or one.tobytes() != old.tobytes():
                raise ValueError("Deterministic parent byte canary mismatch: " + name)
        proof.append(
            {
                "key": row["key"],
                "descriptor_columns_compared": 217,
                "groups_byte_identical": 13,
                "repeated_local_byte_identical": True,
            }
        )
    return proof


def verify_loader(records, manifest):
    expected = np.asarray([r["representation_eligible"] == "true" for r in records])
    for route in manifest["routes"]:
        loaded = load_feature_route(records, manifest, route)
        if not np.array_equal(loaded["valid"], expected):
            raise ValueError("Record-level reference mask changed on load")
        if not np.isnan(loaded["X"][~expected]).all():
            raise ValueError("Unresolved records acquired zeros or fabricated representations")
    return {
        "records": len(records),
        "valid": int(expected.sum()),
        "explicit_invalid_NaN": int((~expected).sum()),
        "routes": len(manifest["routes"]),
    }


def publish_receipt(base_run, base, manifest, records, frozen_baseline, cache_reused):
    cache = Path(manifest["cache_dir"])
    run_dir = cache.parents[1]
    configuration = manifest["configuration"]
    loaded = verify_loader(records, manifest)
    _verify_artifacts(base)
    for name, expected in base["artifact_hashes"].items():
        if sha(safe_path(base["cache_dir"], name)) != expected:
            raise ValueError("Immutable parent artifact changed")
    if (
        sha(base_run / "features/manifest.json") != configuration["parent_pointer_sha256"]
        or sha(base_run / "representation_methods.json") != configuration["parent_registry_sha256"]
    ):
        raise ValueError("Immutable parent pointer or registry changed")
    if algorithm_contract(base, frozen_baseline) != configuration["algorithm_contract"]:
        raise ValueError("Extractor source changed while feature bank was built")
    write_json(run_dir / "features/manifest.json", manifest)
    write_json(
        run_dir / "representation_methods.json", read_json(cache / "representation_methods.json")
    )
    shutil.copy2(cache / "source_feature_rows_index.tsv", run_dir / "source_feature_rows_index.tsv")
    write_json(
        run_dir / "feature_verification.json",
        {
            "status": "prepared_feature_only_no_training",
            "records": 52,
            "valid_reference_records": 38,
            "explicit_unrepresented_records": 14,
            "unique_reference_graphs": 38,
            "exact_parent_graphs_reused": 20,
            "new_unique_graphs_extracted": 18,
            "groups": 13,
            "dimensions": 15212,
            "independent_methods": 10,
            "algorithm_canaries": read_json(cache / "algorithm_canaries.json"),
            "loader": loaded,
            "parent_active_artifacts_unchanged": len(base["artifact_hashes"]),
            "parent_pointer_byte_unchanged": True,
            "training_admitted_records": 0,
            "source_models_created": 0,
            "sequence_or_context_dimensions": 0,
            "cache": str(cache),
            "artifact_hashes_verified": len(manifest["artifact_hashes"]),
            "cache_reused_on_this_invocation": cache_reused,
        },
    )
    return manifest


def build_features(base_run, evidence_dir, run_dir, frozen_baseline):
    base_run, run_dir = Path(base_run).resolve(), Path(run_dir).resolve()
    evidence_dir, frozen_baseline = Path(evidence_dir).resolve(), Path(frozen_baseline).resolve()
    if any(
        a == b or a in b.parents or b in a.parents
        for a, b in (
            (base_run, run_dir),
            (evidence_dir, run_dir),
        )
    ):
        raise ValueError("Output must be isolated from all immutable source roots")
    objects, records, graphs = read_source_plan(evidence_dir)
    pointer = base_run / "features/manifest.json"
    before = sha(pointer)
    base = read_json(pointer)
    _verify_artifacts(base)
    registry_path = base_run / "representation_methods.json"
    registry_before = sha(registry_path)
    original_registry = read_json(registry_path)
    methods = [
        m
        for m in original_registry["methods"]
        if m["source_kind"] == "structure" and m["independent"] is True
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Exactly ten independent parent SMILES methods required")
    groups = {n: copy.deepcopy(base["groups"][n]) for n in STRUCTURE_GROUPS}
    if sum(g["shape"][1] for g in groups.values()) != 15212:
        raise ValueError("Parent 13-group chemistry schema changed")
    contract = algorithm_contract(base, frozen_baseline)
    source_hashes = {n: sha(evidence_dir / n) for n in SNAPSHOT_FILES}
    configuration = {
        "version": VERSION,
        "parent_fingerprint": base["input_fingerprint"],
        "parent_pointer_sha256": before,
        "parent_registry_sha256": registry_before,
        "source_evidence_sha256": source_hashes,
        "algorithm_contract": contract,
        "implementation_sha256": sha(__file__),
        "identity_tier": IDENTITY_TIER,
        "record_count": 52,
        "unique_reference_graphs": 38,
        "training_admitted": False,
        "context_dimensions": 0,
        "blas_threads": 1,
        "graph_workers": 1,
        "numpy_version": np.__version__,
        "python_version": list(sys.version_info[:3]),
    }
    fingerprint = signature(configuration)
    cache = run_dir / "features" / fingerprint[:16]
    if (cache / "manifest.json").is_file():
        manifest = read_json(cache / "manifest.json")
        _verify_artifacts(manifest)
        if manifest["input_fingerprint"] != fingerprint:
            raise ValueError("Feature fingerprint collision")
        return publish_receipt(base_run, base, manifest, records, frozen_baseline, True)
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "pending.json", configuration)
    parent_cache = Path(base["cache_dir"])
    parent_graphs = {r["key"]: r for r in rows(parent_cache / "structure_source_smiles.tsv")}
    parent_index = {
        r["key"]: int(r["row_idx"]) for r in rows(parent_cache / "structure_bank_index.tsv")
    }
    reused, missing = reuse_partition(graphs, parent_graphs, parent_index)
    if (len(reused), len(missing)) != (20, 18):
        raise ValueError("Expected exact parent reuse20/new18; do not widen reuse matching")
    with threadpool_limits(limits=1):
        canaries = canary_check(base, reused, parent_index)
        computed = {
            r["key"]: extract_all(r["canonical_isomeric_smiles"], r["key"]) for r in missing
        }
        runtime_threads = threadpool_info()
    if any(t["num_threads"] != 1 for t in runtime_threads):
        raise ValueError("BLAS thread budget was not one during extraction")
    reused_keys = {r["key"] for r in reused}
    for name, meta in groups.items():
        old = np.load(safe_path(parent_cache, meta["matrix"]), mmap_mode="r")
        valid = np.load(safe_path(parent_cache, meta["valid"]), mmap_mode="r")
        if old.dtype != np.dtype(meta["dtype"]):
            raise ValueError("Parent matrix dtype does not match metadata")
        matrix = np.empty((len(graphs), meta["shape"][1]), dtype=old.dtype)
        for i, graph in enumerate(graphs):
            key = graph["key"]
            if key in reused_keys:
                vector = checked_vector(
                    old[parent_index[key]], valid[parent_index[key]], meta, name
                )
            else:
                result = computed[key]
                vector = checked_vector(result[0][name], result[1][name], meta, name)
            matrix[i] = vector
            if key in reused_keys and matrix[i].tobytes() != old[parent_index[key]].tobytes():
                raise ValueError("Exact reused vector bytes changed")
        meta["shape"][0] = len(graphs)
        meta["index"] = "structure_bank_index.tsv"
        check_matrix(matrix, np.ones(len(graphs), dtype=bool), meta, name)
        atomic_array(safe_path(cache, meta["matrix"]), matrix)
        atomic_array(safe_path(cache, meta["valid"]), np.ones(len(graphs), dtype=bool))
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": r["key"], "row_idx": i} for i, r in enumerate(graphs)],
    )
    write_tsv(cache / "structure_source_smiles.tsv", ["key", "canonical_isomeric_smiles"], graphs)
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {"record_id": r["record_id"], "structure_key": r["structure_key"], "sequence_key": ""}
            for r in records
        ],
    )
    write_tsv(cache / "source_feature_rows_index.tsv", list(records[0]), records)
    write_json(cache / "source_objects_snapshot.json", objects)
    write_json(cache / "source_parent_manifest.json", base)
    write_json(cache / "algorithm_canaries.json", canaries)
    write_json(
        cache / "new_graph_extraction_audit.json",
        [{"key": key, "errors": r[2]} for key, r in computed.items()],
    )
    write_json(
        cache / "reuse_receipt.json",
        {
            "reused_graph_keys": sorted(reused_keys),
            "new_graph_keys": sorted(computed),
            "reused_all13_groups_byte_identical": True,
            "runtime_threadpools_during_extraction": runtime_threads,
        },
    )
    snapshot_dir = cache / "source_evidence"
    snapshot_dir.mkdir()
    for name in SNAPSHOT_FILES:
        if sha(evidence_dir / name) != source_hashes[name]:
            raise ValueError("Source evidence changed while computing features")
        shutil.copy2(evidence_dir / name, snapshot_dir / name)
    implementation_dir = cache / "source_implementation"
    implementation_dir.mkdir()
    for name in (*ALGORITHMS, "peptide_modeling_features.py", Path(__file__).name):
        shutil.copy2(ROOT / "scripts" / name, implementation_dir / name)
    shutil.copy2(frozen_baseline, implementation_dir / "frozen_baseline_features.py")
    registry = {
        **copy.deepcopy(original_registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": "literal_database_reference_feature_only",
        "context_is_additional_chemical_method": False,
    }
    write_json(cache / "representation_methods.json", registry)
    routes = {m["route"]: m["group_names"] for m in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": configuration,
        "record_index": "record_bank_index.tsv",
        "requested_records": 52,
        "structure_keys": 38,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "training_admitted": False,
        "status": "prepared_feature_only_no_training",
        "coverage": {
            n: {
                "representations": 38,
                "valid": 38,
                "invalid": 0,
                "record_valid": 38,
                "record_invalid": 14,
            }
            for n in groups
        },
    }
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p)
        for p in cache.rglob("*")
        if p.is_file() and p.name != "pending.json"
    }
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    return publish_receipt(base_run, base, manifest, records, frozen_baseline, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-run", type=Path, default=ROOT / "data/peptide_modeling_unified_features_20261114_v1"
    )
    parser.add_argument(
        "--evidence-dir", type=Path, default=ROOT / "data/dramp_binary_source_recovery_20261115_v1"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "data/peptide_modeling_dramp_binary_reference_features_20261115_v1",
    )
    parser.add_argument(
        "--frozen-baseline",
        type=Path,
        default=ROOT
        / "data/peptide_modeling_multiseed20h_overlay_v1/scripts/peptide_modeling_features.py",
    )
    args = parser.parse_args()
    manifest = build_features(args.base_run, args.evidence_dir, args.run_dir, args.frozen_baseline)
    print(json.dumps({"status": manifest["status"], "cache": manifest["cache_dir"]}))


if __name__ == "__main__":
    main()
