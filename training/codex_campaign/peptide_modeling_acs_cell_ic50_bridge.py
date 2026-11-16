"""Prepare conservative author-SI cell-line IC50 estimates and full-graph features.

Author code/graph conflicts and unspecified stereochemistry are source-specific
quarantines. Formatted central estimates are not converted into invented exact
measurement equalities, and their typed spreads remain statistically undefined.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from peptide_modeling_add_smiles_methods import rdkit_tools
from peptide_modeling_clint_features import STRUCTURE_GROUPS, atomic_array, extract_all
from peptide_modeling_features import _verify_artifacts, load_feature_route
from peptide_modeling_mhc_mass_bridge import check_matrix, safe_path, write_tsv
from peptide_modeling_train import _inner_splits, freeze_outer_split, validate_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/acs_cell_ic50_source_audit_20261114_v1"
BASE = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_acs_cell_ic50_followup_20261114_v1"
RECEIPT_SHA = "5c2184b18e9f3fb2c6f64101d01a150fc906a6eae8c2ec808b28c78319ed2260"
CSV_SHA = "f74711637f133163a3b3e045a12e8906753f361cfe489a90d7d45ae076be496c"
DB_SHA = "9944192e916f8c0e82464e4169c8d19d834c8a1bf5da67085f91b9bae09d5bcf"
DOI = "10.1021/acs.jmedchem.5c00544"
MEASURE = "source_reported_mammalian_cell_line_IC50_readout_not_independently_verified"
DISPERSION = "not_independently_defined_for_cellular_columns"
VERSION = "peptide_acs_cellular_ic50_conservative_v1"
CELLS = {2: "D283-med", 3: "SF188", 4: "H460", 5: "HFF-1"}
CANCER_CELLS = ("D283-med", "SF188", "H460")
CELL_TYPES = {"D283-med": "medulloblastoma", "SF188": "glioblastoma", "H460": "lung carcinoma"}
SHARED_CODES = {"15", "16", "19", "30"}
STEREO_CODES = {"23", "37", "38"}
QUARANTINED = SHARED_CODES | STEREO_CODES
SEEDS = (42, 2026, 3407)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def project_path(name):
    path = Path(str(name).replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def verify_sources(source):
    source = Path(source)
    receipt_path = source / "delivery_receipt.json"
    if sha(receipt_path) != RECEIPT_SHA:
        raise ValueError("Pinned current ACS audit receipt changed")
    receipt = read(receipt_path)
    protected = {receipt_path: RECEIPT_SHA}
    for section in ("files", "historical_intermediate_artifacts_not_execution_inputs"):
        for name, expected in receipt[section].items():
            relative = Path(name.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Source artifact path escapes the audit directory")
            path = source / relative
            if path.is_symlink() or not path.is_file() or sha(path) != expected:
                raise ValueError("Pinned ACS source artifact changed: " + name)
            protected[path] = expected
    audit = read(source / "audit.json")
    for name, expected in audit["source_sha256"].items():
        path = project_path(name)
        if sha(path) != expected:
            raise ValueError("Original ACS input changed: " + name)
        protected[path] = expected
    db = ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite"
    if sha(db) != DB_SHA:
        raise ValueError("Frozen source catalogue changed")
    protected[db] = DB_SHA
    return audit, protected


def parse_cell(raw):
    """Parse only this author's numeric formatting; keep the original text verbatim."""
    text = str(raw).strip()
    if text.startswith((">", "<", "≥", "≤")):
        return {"kind": "censored", "raw": text}
    parts = [part.strip() for part in text.split("±")]
    if len(parts) > 2:
        raise ValueError("Multiple unknown spread components")
    numbers = []
    for part in parts:
        if not re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", part):
            raise ValueError("Unsupported author central-estimate formatting")
        try:
            value = Decimal(part.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("Invalid author decimal") from exc
        if not value.is_finite() or value < 0:
            raise ValueError("Nonfinite or negative author estimate")
        numbers.append(value)
    if numbers[0] <= 0:
        raise ValueError("A log-target central estimate must be positive")
    return {
        "kind": "central_estimate",
        "raw": text,
        "central": numbers[0],
        "spread": numbers[1] if len(numbers) == 2 else None,
        "dispersion_kind": DISPERSION,
    }


def inspect_graph(smiles):
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Author full graph is unparseable")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    undefined = sum(
        value == "?" for _, value in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    )
    return canonical, "CANON_SMILES:" + hashlib.sha256(canonical.encode()).hexdigest(), undefined


def author_source(path):
    if sha(path) != CSV_SHA:
        raise ValueError("Author CSV does not match the pinned raw bytes")
    with Path(path).open(encoding="cp1252", newline="") as stream:
        raw_rows = list(csv.reader(stream))
    if any(raw_rows[0][column] != cell + " IC50 (nM)" for column, cell in CELLS.items()):
        raise ValueError("Cellular source header or unit differs")
    compounds, pairs, by_graph = {}, {}, defaultdict(list)
    for number, row in enumerate(raw_rows[2:], 3):
        code = row[0].strip()
        if not code or not row[1].strip():
            continue
        if code in compounds:
            raise ValueError("Author compound code is duplicated")
        canonical, key, undefined = inspect_graph(row[1])
        compounds[code] = {
            "code": code,
            "source_row": number,
            "source_smiles": row[1],
            "canonical": canonical,
            "key": key,
            "undefined": undefined,
        }
        by_graph[key].append(code)
        for column, cell in CELLS.items():
            pairs[code, cell] = {"raw": row[column], "source_column": column + 1}
    conflicts = {code for codes in by_graph.values() if len(codes) > 1 for code in codes}
    unspecified = {code for code, item in compounds.items() if item["undefined"]}
    if conflicts != SHARED_CODES or unspecified != STEREO_CODES:
        raise ValueError("Source-specific identity/stereochemistry quarantine changed")
    return compounds, pairs


def validate_record(candidate, frozen, compound, source_cell):
    code, cell = candidate["compound"], candidate["cell"]
    if code in QUARANTINED:
        raise ValueError("Source code is quarantined, not a globally invalid chemical graph")
    parsed = parse_cell(source_cell["raw"])
    if parsed["kind"] != "central_estimate":
        raise ValueError("A censored source cell cannot become a point estimate")
    raw = json.loads(frozen["raw_json"])
    context = json.loads(raw["context_json"])
    detail = {
        "D283-med": "d283_med_ic50",
        "SF188": "sf188_ic50",
        "H460": "h460_ic50",
        "HFF-1": "hff_1_ic50",
    }[cell]
    if (
        frozen["record_id"] != candidate["record_id"]
        or frozen["source_code"] != "acs_jmedchem_5c00544"
        or frozen["endpoint"] != "cytotoxicity"
        or frozen["source_entry_id"] != "compound_" + code + ":" + cell + "_ic50"
        or raw["source_record_id"] != frozen["source_entry_id"]
        or raw["endpoint_detail"].lower().replace("-", "_") != detail
        or raw["matrix"] != cell + " cell line"
        or raw["doi_pmid"] != DOI
        or frozen["raw_unit"] != "nM"
        or raw["unit"] != "nM"
        or frozen["unit"] != "nM"
        or frozen["raw_relation"] != ""
        or raw["relation"] != ""
        or candidate["raw_relation"] != ""
        or context["compound"] != code
        or context["source_raw_value"] != parsed["raw"]
        or candidate["raw_cell"] != parsed["raw"]
        or int(candidate["source_csv_row"]) != compound["source_row"]
        or int(candidate["source_column"]) != source_cell["source_column"]
        or candidate["dispersion_kind"] != DISPERSION
        or candidate["relation_evidence"] != "point_estimate"
    ):
        raise ValueError("Source author code/cell/header/point-estimate lineage differs")
    central = parsed["central"]
    if any(
        Decimal(str(value)) != central
        for value in (frozen["raw_value"], frozen["value"], candidate["source_value_nM"])
    ):
        raise ValueError("Source central estimate changed")
    if (
        parsed["spread"] is None
        or Decimal(context["reported_sd"]) != parsed["spread"]
        or Decimal(candidate["plus_minus_component"]) != parsed["spread"]
    ):
        raise ValueError("Source typed spread differs; its statistic must remain unknown")
    canonical, key, undefined = inspect_graph(frozen["smiles"])
    if (
        undefined != 0
        or compound["undefined"] != 0
        or canonical != compound["canonical"]
        or key != compound["key"]
        or canonical != frozen["canonical_smiles"]
        or key != frozen["canonical_identity_group"]
        or key != candidate["graph"]
        or canonical != candidate["canonical_isomeric_smiles"]
        or frozen["structure_model_eligible"] != "true"
        or candidate["disposition"] != "unambiguous_source_point_estimate_candidate"
        or candidate["source_graph_identity_ambiguity"] != "False"
        or json.loads(candidate["shared_source_compound_ids"]) != []
    ):
        raise ValueError("Full author structure, identity association or stereo evidence differs")
    normalized = central / Decimal(1000)
    if Decimal(candidate["proposed_value_uM"]) != normalized:
        raise ValueError("Cellular nM-to-uM normalization differs")
    target = math.log10(float(normalized))
    if not math.isclose(float(candidate["target"]), target, rel_tol=1e-13, abs_tol=1e-13):
        raise ValueError("Cellular log10 target differs")
    return central, normalized, target, key


def reconcile_groups(rows, families):
    wanted = {r["graph"] for r in rows}
    known = {key: set() for key in wanted}
    for row in families:
        if (
            row["canonical_identity_group"] in wanted
            and row["structure_eligible"].lower() == "true"
        ):
            known[row["canonical_identity_group"]].add(row["split_group"])
    for cell in CELLS.values():
        selected = [r for r in rows if r["cell"] == cell]
        parent = {}

        def find(key, parent=parent):
            parent.setdefault(key, key)
            if parent[key] != key:
                parent[key] = find(parent[key])
            return parent[key]

        for row in selected:
            key = row["graph"]
            if sorted(known[key]) != json.loads(row["existing_family_groups"]):
                raise ValueError("Known same-graph FAMILY associations changed")
            find(key)
            for family in known[key]:
                a, b = find(key), find(family)
                if a != b:
                    parent[b] = a
        roots = defaultdict(list)
        for key in list(parent):
            roots[find(key)].append(key)
        groups = {
            node: "IDENTITY:" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:24]
            for members in roots.values()
            for node in members
        }
        for row in selected:
            if row["split_group"] != groups[row["graph"]]:
                raise ValueError("Conservative graph/FAMILY split component changed")


def source_rows(source=SOURCE):
    audit, protected = verify_sources(source)
    source = Path(source)
    selected = table(source / "conservative_candidate_records.tsv")
    if len(selected) != 90 or len({r["record_id"] for r in selected}) != 90:
        raise ValueError("Exactly ninety conservative source estimates are required")
    source_csv = project_path(
        next(p for p in audit["source_sha256"] if p.endswith("jm5c00544_si_002.csv"))
    )
    compounds, cells = author_source(source_csv)
    db = ROOT / "data/dataset_reclean_20261113_v1/catalogue.sqlite"
    connection = sqlite3.connect("file:" + db.as_posix() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    marks = ",".join("?" for _ in selected)
    frozen = {
        r["record_id"]: dict(r)
        for r in connection.execute(
            f"SELECT * FROM records_with_structure_qc WHERE record_id IN ({marks})",
            [r["record_id"] for r in selected],
        )
    }
    connection.close()
    if len(frozen) != 90:
        raise ValueError("Conservative source catalogue membership differs")
    family_path = project_path(
        next(p for p in audit["source_sha256"] if p.endswith("feature_rows_index.tsv"))
    )
    reconcile_groups(selected, table(family_path))
    rows, proofs, seen = [], [], set()
    for candidate in selected:
        pair = candidate["compound"], candidate["cell"]
        if pair in seen:
            raise ValueError("A source code/cell point cannot be duplicated")
        seen.add(pair)
        raw = frozen[candidate["record_id"]]
        central, normalized, target, key = validate_record(
            candidate, raw, compounds[pair[0]], cells[pair]
        )
        rows.append(
            {
                **candidate,
                "target": format(target, ".17g"),
                "value": str(normalized),
                "unit": "uM",
                "target_transform": "log10",
                "structure_key": key,
                "canonical_identity_group": key,
                "structure_eligible": "true",
                "sequence_eligible": "false",
                "sequence": "",
                "source_raw_relation": raw["raw_relation"],
                "relation": "",
                "source_unit": "nM",
                "source_central_estimate_nM": str(central),
                "source_typed_spread_nM": candidate["plus_minus_component"],
                "source_doi": DOI,
                "point_evidence": (
                    "author_formatted_central_estimate_not_invented_measurement_equality"
                ),
                "canonical_measure": MEASURE,
                "cellular_readout_and_exposure_verified": "false",
                "dispersion_statistic_verified": "false",
                "normal_cell_safety_verified": "false",
                "training_task_eligible": "true" if pair[1] in CANCER_CELLS else "false",
            }
        )
        proofs.append(
            {
                "record_id": candidate["record_id"],
                "compound": pair[0],
                "cell": pair[1],
                "source_csv_row": compounds[pair[0]]["source_row"],
                "source_column": cells[pair]["source_column"],
                "raw_source_cell": candidate["raw_cell"],
                "raw_relation_preserved": raw["raw_relation"],
                "central_nM": str(central),
                "typed_spread_nM": candidate["plus_minus_component"],
                "dispersion_kind": DISPERSION,
                "model_value_uM": str(normalized),
                "structure_key": key,
                "source_raw_json_sha256": signature(json.loads(raw["raw_json"])),
            }
        )
    if {cell: sum(r["cell"] == cell for r in rows) for cell in CELLS.values()} != {
        "D283-med": 29,
        "SF188": 29,
        "H460": 29,
        "HFF-1": 3,
    }:
        raise ValueError("Per-cell conservative point counts changed")
    if len({r["structure_key"] for r in rows}) != 30:
        raise ValueError("Thirty conservative full graphs must be retained")
    return rows, proofs, protected, audit


def build_features(task_rows, source_proof, protected, output, base_run=BASE):
    output, base_run = Path(output).resolve(), Path(base_run).resolve()
    if output == base_run or output in base_run.parents or base_run in output.parents:
        raise ValueError("Feature output must be separate from immutable parent bank")
    base_path = base_run / "features/manifest.json"
    pointer_sha = sha(base_path)
    base = read(base_path)
    _verify_artifacts(base)
    registry_path = base_run / "representation_methods.json"
    registry = read(registry_path)
    methods = [
        m for m in registry["methods"] if m["source_kind"] == "structure" and m["independent"]
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Ten independent complete-graph methods are required")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if base["configuration"]["rdkit_version"] != rdkit_tools()["version"]:
        raise ValueError("RDKit differs from frozen parent algorithms")
    cache_source = Path(base["cache_dir"])
    old_index = {
        r["key"]: int(r["row_idx"]) for r in table(cache_source / groups["morgan"]["index"])
    }
    old_graphs = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in table(cache_source / "structure_source_smiles.tsv")
    }
    graph_map = {}
    for row in task_rows:
        key, smiles = row["structure_key"], row["canonical_isomeric_smiles"]
        if key in graph_map and graph_map[key] != smiles:
            raise ValueError("Source graph key has conflicting full structures")
        graph_map[key] = smiles
    graphs = [{"key": key, "smiles": graph_map[key]} for key in sorted(graph_map)]
    if len(graphs) != 30 or len(task_rows) != 90:
        raise ValueError("Expected ninety source rows and thirty full graphs")
    reused = {r["key"] for r in graphs if r["key"] in old_index}
    new = [r for r in graphs if r["key"] not in reused]
    if len(reused) != 6 or len(new) != 24:
        raise ValueError("Expected six reusable and 24 new author graphs")
    if any(
        old_graphs[key] != next(r["smiles"] for r in graphs if r["key"] == key) for key in reused
    ):
        raise ValueError("Parent full graph differs from original archive")
    config = {
        "version": VERSION,
        "implementation_sha256": sha(__file__),
        "implementation_dependencies_sha256": {
            name: sha(Path(__file__).with_name(name))
            for name in (
                "peptide_modeling_mhc_mass_bridge.py",
                "peptide_modeling_clint_features.py",
                "peptide_modeling_expand_features.py",
                "peptide_modeling_add_methods.py",
                "peptide_modeling_add_smiles_methods.py",
                "peptide_modeling_features.py",
            )
        },
        "parent_fingerprint": base["input_fingerprint"],
        "parent_pointer_sha256": pointer_sha,
        "registry_sha256": sha(registry_path),
        "source_sha256": {str(p): h for p, h in protected.items()},
        "source_proof": source_proof,
        "rdkit_version": rdkit_tools()["version"],
        "numpy_version": np.__version__,
        "inherited_algorithm_configuration": base["configuration"],
        "extracted_graphs": len(new),
        "reused_graphs": len(reused),
        "sequence_policy": "No sequence projection or ESM generated",
        "max_structure_atoms": 1000,
    }
    fingerprint = signature(config)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "manifest.json").exists():
        old = read(cache / "manifest.json")
        _verify_artifacts(old)
        return old
    extracted = {r["key"]: extract_all(r["smiles"], r["key"]) for r in new}
    compatibility = {
        r["key"]: extract_all(r["smiles"], r["key"]) for r in graphs if r["key"] in reused
    }
    for name, metadata in groups.items():
        before = np.load(safe_path(cache_source, metadata["matrix"]), mmap_mode="r")
        before_valid = np.load(safe_path(cache_source, metadata["valid"]), mmap_mode="r")
        for key, values in compatibility.items():
            recomputed = np.asarray(values[0][name], dtype=metadata["dtype"])
            if not values[1][name] or not np.array_equal(
                before[old_index[key]], recomputed, equal_nan=True
            ):
                raise ValueError("Current extractor differs from frozen reused vectors: " + name)
        matrix = np.asarray(
            [
                before[old_index[r["key"]]] if r["key"] in reused else extracted[r["key"]][0][name]
                for r in graphs
            ],
            dtype=metadata["dtype"],
        )
        valid = np.asarray(
            [
                before_valid[old_index[r["key"]]]
                if r["key"] in reused
                else extracted[r["key"]][1][name]
                for r in graphs
            ],
            dtype=bool,
        )
        metadata["shape"] = [len(graphs), metadata["shape"][1]]
        metadata["index"] = "structure_bank_index.tsv"
        if matrix.shape != tuple(metadata["shape"]):
            raise ValueError("Full graph feature shape differs: " + name)
        check_matrix(matrix, valid, metadata, name)
        for index, row in enumerate(graphs):
            if (
                row["key"] in reused
                and before[old_index[row["key"]]].tobytes() != matrix[index].tobytes()
            ):
                raise ValueError("Reused graph feature bytes changed: " + name)
        atomic_array(safe_path(cache, metadata["matrix"]), matrix)
        atomic_array(safe_path(cache, metadata["valid"]), valid)
    write_tsv(
        cache / "structure_bank_index.tsv",
        ["key", "row_idx"],
        [{"key": r["key"], "row_idx": i} for i, r in enumerate(graphs)],
    )
    write_tsv(
        cache / "structure_source_smiles.tsv",
        ["key", "canonical_isomeric_smiles"],
        [{"key": r["key"], "canonical_isomeric_smiles": r["smiles"]} for r in graphs],
    )
    write_tsv(
        cache / "record_bank_index.tsv",
        ["record_id", "structure_key", "sequence_key"],
        [
            {"record_id": r["record_id"], "structure_key": r["structure_key"], "sequence_key": ""}
            for r in task_rows
        ],
    )
    registry = {
        **copy.deepcopy(registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": MEASURE,
    }
    write(cache / "representation_methods.json", registry)
    write(cache / "source_parent_manifest.json", base)
    write(cache / "source_evidence_receipt.json", config)
    write(
        cache / "new_graph_extraction_audit.json",
        [{"key": key, "errors": value[2]} for key, value in extracted.items()],
    )
    routes = {m["route"]: m["group_names"] for m in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = {
        "schema": VERSION,
        "input_fingerprint": fingerprint,
        "cache_dir": str(cache),
        "manifest_path": str(cache / "manifest.json"),
        "configuration": config,
        "record_index": "record_bank_index.tsv",
        "requested_records": 90,
        "structure_keys": 30,
        "groups": groups,
        "routes": routes,
        "representation_methods": registry,
        "coverage": {name: {"representations": 30, "valid": 30, "invalid": 0} for name in groups},
        "scope": MEASURE,
    }
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob("*") if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    _verify_artifacts(base)
    if sha(base_path) != pointer_sha:
        raise ValueError("Parent feature pointer changed")
    if any(
        sha(Path(__file__).with_name(name)) != expected
        for name, expected in config["implementation_dependencies_sha256"].items()
    ):
        raise ValueError("Feature implementation changed during extraction")
    for method in methods:
        payload = load_feature_route(task_rows, manifest, method["route"])
        if not payload["valid"].all() or payload["X"].shape != (90, method["dimensions"]):
            raise ValueError("A method did not load all source observations")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    write(
        output / "feature_verification.json",
        {
            "status": "verified",
            "records": 90,
            "graphs": 30,
            "reused_graphs": 6,
            "new_graphs": 24,
            "feature_groups": len(groups),
            "independent_methods": 10,
            "fusion_dimensions": sum(m["shape"][1] for m in groups.values()),
            "artifact_hashes_verified": len(manifest["artifact_hashes"]),
            "parent_artifacts_unchanged": len(base["artifact_hashes"]),
            "parent_pointer_unchanged": True,
            "reused_rows_byte_identical": True,
            "reused_algorithm_canaries_verified": len(compatibility),
            "all_records_all_methods_loaded": True,
            "no_sequence_or_esm_generated": True,
            "fingerprint": fingerprint,
        },
    )
    return manifest


def split_prechecks(rows, output, config, audit):
    results = []
    source_cohorts = {item["cell"]: item for item in audit["cohorts"]}
    for cell in CANCER_CELLS:
        cohort = [row for row in rows if row["cell"] == cell]
        y, groups = validate_rows(cohort, "regression")
        if len(cohort) != 29 or len(set(groups)) != 29:
            raise ValueError("A cancer-cell task must retain 29 complete identity groups")
        source_plans = {
            item["seed"]: item
            for item in source_cohorts[cell]["actual_three_seed_outer_holdout_and_inner5"]
        }
        seed_results = []
        for seed in SEEDS:
            settings = {**config, "seed": seed}
            split = freeze_outer_split(
                cohort, "regression", Path(output) / f"split_prechecks/{cell}_{seed}.json", settings
            )
            train = np.flatnonzero(np.isin(groups, split["train_groups"]))
            test = np.flatnonzero(np.isin(groups, split["test_groups"]))
            inner = _inner_splits(y[train], groups[train], "regression", settings)
            plan = {
                "seed": seed,
                "train_rows": len(train),
                "holdout_rows": len(test),
                "actual_inner_folds": len(inner),
                "train_record_ids": [cohort[i]["record_id"] for i in train],
                "holdout_record_ids": [cohort[i]["record_id"] for i in test],
                "inner_members": [
                    {
                        "train_record_ids": [cohort[train[i]]["record_id"] for i in a],
                        "validation_record_ids": [cohort[train[i]]["record_id"] for i in b],
                    }
                    for a, b in inner
                ],
            }
            if plan != source_plans[seed] or len(inner) != 5 or (len(train), len(test)) != (20, 9):
                raise ValueError("Actual holdout/fivefold members differ from frozen source review")
            for a, b in inner:
                if set(groups[train[a]]) & set(groups[train[b]]):
                    raise ValueError("Known molecular group crosses an inner fold")
            seed_results.append(plan)
        results.append({"cell": cell, "seeds": seed_results})
    return {
        "status": "verified_partitions_not_fitted",
        "cohorts": results,
        "source_review_members_exactly_reproduced": True,
        "independent_external_validation": False,
        "source_doi": DOI,
    }


def prepare(source=SOURCE, output=OUTPUT, base_run=BASE):
    source, output = Path(source), Path(output).resolve()
    if (output / "delivery_receipt.json").exists():
        raise FileExistsError("Frozen ACS derived admission exists; use a new version")
    rows, proofs, protected, audit = source_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    features = build_features(rows, proofs, protected, output, base_run)
    config = {
        "resources": {"training_host": "node23", "n_jobs": 1, "nice": 10, "cpu_budget": 1},
        "training": {
            "routes": list(features["routes"]),
            "models": [
                "linear",
                "elasticnet",
                "random_forest",
                "extra_trees",
                "histgb",
                "lightgbm",
                "xgboost",
                "svm",
            ],
            "inner_folds": 5,
            "test_size": 0.3,
            "seed": 42,
            "min_rows": 10,
            "min_groups": 8,
            "search_backend": "optuna",
            "search_budget": 128,
            "n_jobs": 1,
            "refit_full": True,
            "svm_max_features": 16384,
            "histgb_max_features": 16384,
            "elasticnet_max_features": 16384,
        },
        "required_seeds": list(SEEDS),
        "validation_protocol": "fixed_outer_group_holdout_with_fivefold_inner_selection",
    }
    candidates = read(source / "candidate_task_manifest.json")["tasks"]
    tasks = []
    (output / "task_tables").mkdir(exist_ok=True)
    for candidate in candidates:
        definition = copy.deepcopy(candidate["task_definition"])
        cell = definition["cell_line"]
        if cell not in CANCER_CELLS or definition["quantity"] != MEASURE:
            raise ValueError("Unexpected source task definition or non-training cell")
        cohort = [row for row in rows if row["cell"] == cell]
        source_table = table(source / candidate["table_path"].replace("\\", "/"))
        if [r["record_id"] for r in source_table] != [r["record_id"] for r in cohort]:
            raise ValueError("Source task membership differs from all-cell conservative ledger")
        task_id = candidate["task_id"].removeprefix("CANDIDATE_")
        path = "task_tables/" + task_id + ".tsv"
        write_tsv(output / path, list(cohort[0]), cohort)
        definition.update(
            {
                "point_evidence_type": "author_formatted_central_estimate",
                "raw_missing_relation_preserved": True,
                "invented_equality": False,
                "dispersion_statistic": DISPERSION,
                "cell_type_in_source_SI_figure": CELL_TYPES[cell],
                "normal_cell_safety_verified": False,
                "enzyme_assay_conditions_transferred": False,
            }
        )
        tasks.append(
            {
                "task_id": task_id,
                "endpoint": "cytotoxicity_IC50",
                "task_kind": "regression",
                "status": "eligible",
                "table_path": path,
                "n_records": 29,
                "n_groups": 29,
                "unit": "uM",
                "target_transform": "log10",
                "inverse_transform": "pow10",
                "task_definition": definition,
                "requested_inner_folds": 5,
                "source_endpoint_lineage": "cytotoxicity",
                "source_doi": DOI,
                "validation_protocol": config["validation_protocol"],
                "source_evidence_tier": (
                    "pinned_author_supporting_csv_central_estimate_graph_"
                    "cell_scoped_readout_unverified"
                ),
                "scientific_warning": (
                    "Same-paper cancer-cell pharmacology estimate benchmark; cellular readout, "
                    "exposure and dispersion statistic remain unverified. No normal-cell safety, "
                    "clinical, or independent-source validation."
                ),
            }
        )
    if len(tasks) != 3:
        raise ValueError("Exactly three cancer-cell training tasks are permitted")
    manifest = {
        "schema": VERSION,
        "tasks": tasks,
        "models_fitted": 0,
        "clear_label_feature_rows": 90,
        "training_rows": 87,
        "HFF1_records": 3,
        "HFF1_training_task_created": False,
    }
    write(output / "task_manifest.json", manifest)
    write(output / "targeted_task_manifest.json", manifest)
    write(output / "recommended_training_config.json", config)
    write_tsv(output / "qualified_source_points.tsv", list(rows[0]), rows)
    write(
        output / "fivefold_prechecks.json", split_prechecks(rows, output, config["training"], audit)
    )
    write(
        output / "source_verification.json",
        {
            "status": "verified_conservative_source_estimates",
            "records": proofs,
            "source_doi": DOI,
            "original_source_points": 108,
            "original_censors": 18,
            "shared_code_graph_quarantined_points": 9,
            "unspecified_stereo_quarantined_points": 9,
            "source_graph_quarantine_is_global_blacklist": False,
            "qualified_points": 90,
            "training_points": 87,
            "feature_only_HFF1_points": 3,
            "source_raw_relations_invented": False,
            "source_typed_spread_statistic_verified": False,
            "normal_cell_safety_verified": False,
            "enzyme_conditions_borrowed": False,
            "protected_files": {str(p): h for p, h in protected.items()},
            "source_files_unchanged": True,
            "models_fitted": 0,
            "remote_operations": False,
        },
    )
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("Protected author or frozen source changed during preparation")
    write(
        output / "admission_receipt.json",
        {
            "status": "prepared_not_trained",
            "tasks": 3,
            "training_points": 87,
            "clear_label_feature_points": 90,
            "unique_graphs": 30,
            "unit": "uM",
            "source_estimates_not_exact_measurement_equalities": True,
            "dispersion_statistic_unverified": True,
            "source_code_graph_and_stereo_quarantines_retained": True,
            "source_files_unchanged": len(protected),
            "feature_fingerprint": features["input_fingerprint"],
            "models_fitted": 0,
            "remote_operations": False,
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-run", type=Path, default=BASE)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.base_run)
    print(
        json.dumps(
            {"status": "prepared_not_trained", "tasks": len(result["tasks"]), "training_points": 87}
        )
    )


if __name__ == "__main__":
    main()
