"""Derive reference covalent graphs for pinned, existing safety observations.

This performs chemical-format conversion, not sequence design or training. An X
is interpreted only within the three explicitly pinned source scopes. Two
independent graph assemblies, stereochemistry and elemental formula must agree.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from peptide_modeling_clint_features import STRUCTURE_GROUPS, atomic_array, extract_all
from peptide_modeling_expand_features import (
    read_json,
    rows,
    safe_path,
    sha,
    signature,
    write_json,
    write_tsv,
)
from peptide_modeling_features import _verify_artifacts, canonical_key, load_feature_route
from peptide_modeling_qualitative_cyto_features import check_matrix
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/remaining_safety_source_audit_20261114_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_safety_spec_features_20261114_v1"
SOURCE_RECEIPT_SHA = "a0d03ebcb86f93393f05b6834086cf858f65e0d26401d9b71dcd83b2077e1ec0"
VERSION = "source_safety_chemical_spec_structure_bank_v1"
STANDARD = frozenset("ACDEFGHIKLMNPQRSTVWY")
COMPONENTS = {
    "AIB": {
        "cid": 6119,
        "formula": "C4H9NO2",
        "inchikey": "FUOOLUPWFVMBKG-UHFFFAOYSA-N",
        "sha256": "f2047ca473558137fd88d315962b2e8be95a15ec202c789e5ce73194bbd92c10",
        "canonical": "CC(C)(N)C(=O)O",
    },
    "L1NAL": {
        "cid": 2724883,
        "formula": "C13H13NO2",
        "inchikey": "OFYAYGJCPXRNBL-LBPRGKRZSA-N",
        "sha256": "2064ff2ed0e3288c94f8d6083ff8444efeefb0ee9d65d99db8ae973c384d6bc4",
        "canonical": "N[C@@H](Cc1cccc2ccccc12)C(=O)O",
    },
}
SCOPES = {
    "LC50_AIB": ("cytotoxicity_LC50_AIB_PMID17767978.tsv", 11, "PMID:17767978", "L929"),
    "CC50_COPIES": ("cytotoxicity_CC50_PMID29290802_deduplicated.tsv", 5, "PMID:29290802", "Vero"),
    "CC50_NAL": (
        "cytotoxicity_CC50_PMID20718719_Nal_specifications.tsv",
        6,
        "PMID:20718719",
        "Vero",
    ),
}


def canonical(molecule):
    mol = Chem.Mol(molecule)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def registered_components(directory):
    result = {}
    for key, expected in COMPONENTS.items():
        path = Path(directory) / (key + ".response")
        if sha(path) != expected["sha256"]:
            raise ValueError("Official component response changed: " + key)
        receipt = read_json(Path(directory) / (key + ".receipt.json"))
        if receipt["status"] != 200 or receipt["sha256"] != expected["sha256"]:
            raise ValueError("Official component receipt differs")
        if not receipt["url"].startswith("https://pubchem.ncbi.nlm.nih.gov/rest/pug/"):
            raise ValueError("Component is not the pinned official reference")
        item = read_json(path)["PropertyTable"]["Properties"]
        if len(item) != 1:
            raise ValueError("A registered component must be unambiguous")
        item = item[0]
        if any(
            item[name] != expected[field]
            for name, field in (
                ("CID", "cid"),
                ("MolecularFormula", "formula"),
                ("InChIKey", "inchikey"),
            )
        ):
            raise ValueError("Registered component identity differs")
        mol = Chem.MolFromSmiles(item["SMILES"])
        if mol is None or canonical(mol) != expected["canonical"]:
            raise ValueError("Registered component full stereochemical graph differs")
        if rdMolDescriptors.CalcMolFormula(mol) != expected["formula"]:
            raise ValueError("Registered component elemental formula differs")
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if (key == "AIB" and centers) or (key == "L1NAL" and [x[1] for x in centers] != ["S"]):
            raise ValueError("AIB achirality or L-1-Nal stereochemistry differs")
        result[key] = mol
    return result


def tokens_from_spec(spec, paper):
    if set(spec) != {"chemistry", "sequence"}:
        raise ValueError("Unexpected source chemical specification fields")
    fields, sequence = spec["chemistry"], spec["sequence"]
    if paper not in {item[2] for item in SCOPES.values()} or set(fields) != {
        "N-terminal_Modification",
        "C-terminal_Modification",
        "Other_Modifications",
        "Linear/Cyclic/Branched",
        "Stereochemistry",
    }:
        raise ValueError("Source paper or chemistry fields are outside the pinned scopes")
    if (
        fields.get("N-terminal_Modification") != "Free"
        or fields.get("C-terminal_Modification") != "Amidation"
        or fields.get("Linear/Cyclic/Branched") != "Linear"
        or fields.get("Stereochemistry") != "L"
    ):
        raise ValueError(
            "Only explicitly free-N, amidated-C, linear L source chemistry is supported"
        )
    if not isinstance(sequence, str) or not sequence or sequence != sequence.upper():
        raise ValueError("Source sequence must retain its exact uppercase residue spelling")
    if set(sequence) - (STANDARD | {"X"}) or "C" in sequence:
        raise ValueError("Unknown residue or unresolved cysteine/covalent chemistry")
    other = fields.get("Other_Modifications")
    if not isinstance(other, str):
        raise ValueError("Modification declaration must retain its exact source string")
    tokens = list(sequence)
    if paper == "PMID:17767978":
        if other != "X=AIB (Aminoisobutyric acid)" or "X" not in sequence:
            raise ValueError("AIB mapping requires its exact source-scoped declaration")
        tokens = ["AIB" if item == "X" else item for item in tokens]
    elif other == "None":
        if "X" in sequence:
            raise ValueError("An undeclared X cannot become a standard amino acid")
    elif paper == "PMID:20718719":
        match = re.fullmatch(
            r"The 'X' at position ([0-9]+) indicates Nal \(3-\[1-naphthyl\]alanine\)\.", other
        )
        if match is None or sequence.count("X") != 1:
            raise ValueError("Nal requires the exact 1-naphthyl component and unique position")
        position = int(match[1]) - 1
        if position < 0 or position >= len(sequence) or sequence[position] != "X":
            raise ValueError("Reported Nal position differs from source sequence")
        tokens[position] = "L1NAL"
    else:
        raise ValueError("Unregistered source-scoped chemical modification")
    if any(item not in STANDARD | {"AIB", "L1NAL"} for item in tokens):
        raise ValueError("Incomplete source residue interpretation")
    return tokens


def acid_component(token, components):
    mol = Chem.Mol(components[token]) if token in components else Chem.MolFromSequence(token)
    if mol is None:
        raise ValueError("Residue component cannot be constructed")
    pattern = Chem.MolFromSmarts("[N;!$(N-C=O)]-[CX4]-[CX3](=[OX1])-[OX2H1]")
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) != 1:
        raise ValueError("Amino/carboxyl attachment sites are not unique")
    amino, alpha, carbonyl, _, hydroxyl = matches[0]
    return mol, (amino, alpha, carbonyl, hydroxyl)


def formula_counts(formula):
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if "".join(element + n for element, n in parts) != formula:
        raise ValueError("Unsupported charged/isotopic formula")
    return Counter({element: int(n or 1) for element, n in parts})


def expected_formula(tokens, components):
    result = Counter()
    for token in tokens:
        result.update(
            formula_counts(rdMolDescriptors.CalcMolFormula(acid_component(token, components)[0]))
        )
    result.subtract({"H": 2 * (len(tokens) - 1), "O": len(tokens) - 1})
    result.update({"N": 1, "H": 1})
    result.subtract({"O": 1})
    return +result


def assemble_components(tokens, components):
    combined, sites, expected_chirality = None, [], {}
    for residue, token in enumerate(tokens, 1):
        mol, atoms = acid_component(token, components)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        for atom in mol.GetAtoms():
            atom.SetIntProp("_source_atom_key", residue * 100 + atom.GetIdx() + 1)
            if atom.HasProp("_CIPCode"):
                expected_chirality[atom.GetIntProp("_source_atom_key")] = atom.GetProp("_CIPCode")
        offset = 0 if combined is None else combined.GetNumAtoms()
        sites.append(tuple(index + offset for index in atoms))
        combined = mol if combined is None else Chem.CombineMols(combined, mol)
    rw = Chem.RWMol(combined)
    for left, right in zip(sites, sites[1:], strict=False):
        rw.AddBond(left[2], right[0], Chem.BondType.SINGLE)
    terminal = rw.GetAtomWithIdx(sites[-1][3])
    terminal.SetAtomicNum(7)
    terminal.SetNumExplicitHs(0)
    terminal.SetNoImplicit(False)
    for index in sorted((item[3] for item in sites[:-1]), reverse=True):
        rw.RemoveAtom(index)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    actual = {
        a.GetIntProp("_source_atom_key"): a.GetProp("_CIPCode")
        for a in mol.GetAtoms()
        if a.HasProp("_CIPCode")
    }
    if actual != expected_chirality:
        raise ValueError("Component assembly changed source L or side-chain stereochemistry")
    return mol


def _pdb_atom(mol, residue, name):
    found = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetPDBResidueInfo() is not None
        and a.GetPDBResidueInfo().GetResidueNumber() == residue
        and a.GetPDBResidueInfo().GetName().strip() == name
    ]
    if len(found) != 1:
        raise ValueError("Template residue attachment site is not unique")
    return found[0]


def assemble_whole_template(tokens):
    # The transient A/F residues are construction scaffolds, never feature inputs.
    sequence = "".join("A" if t == "AIB" else "F" if t == "L1NAL" else t for t in tokens)
    mol = Chem.MolFromSequence(sequence)
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetName().strip() == "CB":
            atom.SetIntProp("_source_beta_key", info.GetResidueNumber() * 100 + 99)
    rw = Chem.RWMol(mol)
    remove = []
    for residue, token in enumerate(tokens, 1):
        if token == "AIB":
            alpha = rw.GetAtomWithIdx(_pdb_atom(rw, residue, "CA"))
            alpha.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            alpha.SetNumExplicitHs(0)
            alpha.SetNoImplicit(True)
            methyl = rw.AddAtom(Chem.Atom(6))
            rw.AddBond(alpha.GetIdx(), methyl, Chem.BondType.SINGLE)
        elif token == "L1NAL":
            remove.extend(
                _pdb_atom(rw, residue, name) for name in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
            )
    terminal = rw.GetAtomWithIdx(_pdb_atom(rw, len(tokens), "OXT"))
    terminal.SetAtomicNum(7)
    terminal.SetNumExplicitHs(0)
    terminal.SetNoImplicit(False)
    for index in sorted(remove, reverse=True):
        rw.RemoveAtom(index)
    for residue, token in enumerate(tokens, 1):
        if token != "L1NAL":
            continue
        beta = next(
            a.GetIdx()
            for a in rw.GetAtoms()
            if a.HasProp("_source_beta_key")
            and a.GetIntProp("_source_beta_key") == residue * 100 + 99
        )
        naphthyl = Chem.MolFromSmiles("c1cccc2ccccc12")
        offset = rw.GetNumAtoms()
        rw = Chem.RWMol(Chem.CombineMols(rw.GetMol(), naphthyl))
        rw.AddBond(beta, offset, Chem.BondType.SINGLE)
    result = rw.GetMol()
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
    return result


def derive_graph(spec, paper, components):
    tokens = tokens_from_spec(spec, paper)
    left, right = assemble_components(tokens, components), assemble_whole_template(tokens)
    first, second = canonical(left), canonical(right)
    if first != second:
        raise ValueError("Independent full graph assemblies disagree")
    mol = Chem.MolFromSmiles(first)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if formula_counts(formula) != expected_formula(tokens, components):
        raise ValueError("Condensation and terminal-amidation elemental balance differs")
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if any(code == "?" for _, code in centers) or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Derived source graph has unspecified stereochemistry or extra fragments")
    if Chem.GetFormalCharge(mol) or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("Only neutral, closed-shell covalent reference graphs are declared")
    return dict(
        canonical_isomeric_smiles=first,
        structure_key=canonical_key(first),
        tokens=tokens,
        formula=formula,
        heavy_atoms=mol.GetNumHeavyAtoms(),
        stereocenters=len(centers),
        independent_assemblies_equal=True,
        source_measured_smiles=False,
        protonation_or_salt_state_verified=False,
    )


def validate_sources(source):
    source = Path(source)
    if sha(source / "delivery_receipt.json") != SOURCE_RECEIPT_SHA:
        raise ValueError("Pinned safety source audit receipt changed")
    receipt, audit = read_json(source / "delivery_receipt.json"), read_json(source / "audit.json")
    protected = {str(source / "delivery_receipt.json"): SOURCE_RECEIPT_SHA}
    prefix = SOURCE.relative_to(ROOT)
    for name, expected in receipt["files"].items():
        path = source / Path(name).relative_to(prefix)
        if sha(path) != expected:
            raise ValueError("Protected source changed: " + name)
        protected[str(path)] = expected
    for name, expected in audit["source_sha256"].items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError("Protected source changed: " + name)
        protected[str(path)] = expected
    selected, members = validate_selected_tables(source, audit)
    return selected, members, protected, audit


def validate_selected_tables(source, audit):
    """Bind every representative and copy member to its exact reviewed source point."""
    source = Path(source)
    checks = {r["record_id"]: r for r in audit["record_checks"]}
    if len(checks) != len(audit["record_checks"]):
        raise ValueError("Source audit repeats an original record identity")
    selected, members = [], []
    for scope, (name, count, paper, cell) in SCOPES.items():
        table = list(rows(source / "candidate_tables" / name))
        if len(table) != count or len({r["record_id"] for r in table}) != count:
            raise ValueError("Scoped representative count or identity differs")
        for row in table:
            if row["source_paper"] != paper or row["cell"] != cell or row["raw_relation"] != "=":
                raise ValueError("Scoped source paper, cell or relation changed")
            value = float(row["model_value"])
            if (
                not math.isfinite(value)
                or value <= 0
                or not math.isclose(
                    float(row["target"]), math.log10(value), rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise ValueError("Target must remain the log10 of the exact positive source point")
            ids = (
                json.loads(row["merged_record_ids"])
                if scope == "CC50_COPIES"
                else [row["record_id"]]
            )
            if len(ids) != (2 if scope == "CC50_COPIES" else 1) or row["record_id"] not in ids:
                raise ValueError("Source-copy representative/member mapping differs")
            if row["record_id"] != min(ids):
                raise ValueError(
                    "Source-copy representative must remain the deterministic first ID"
                )
            spec = json.loads(row["chemical_specification"])
            tokens_from_spec(spec, paper)
            for record_id in ids:
                item = checks[record_id]
                point = item.get("resolved_point", {})
                if not item.get("original_sequence_identity_matches") or not item.get(
                    "original_annotation_matches_pinned"
                ):
                    raise ValueError("Original sequence or annotation evidence is incomplete")
                if (
                    item.get("source_status")
                    != "unique_exact_original_scalar_unit_sequence_annotation"
                ):
                    raise ValueError("Source point/sequence/annotation proof is not unique")
                if spec != {
                    "sequence": item["chemistry"]["sequence"],
                    "chemistry": item["chemistry"]["fields"],
                }:
                    raise ValueError("Source chemistry differs from reviewed observation")
                if (
                    point.get("source_paper") != paper
                    or point.get("cell") != cell
                    or point.get("relation") != "="
                ):
                    raise ValueError("Source paper, cell or exact relation differs")
                if point.get("unit") != row["model_unit"] or Decimal(point["scalar"]) != Decimal(
                    row["model_value"]
                ):
                    raise ValueError("Source point scalar or unconverted unit differs")
                members.append(
                    dict(
                        record_id=record_id,
                        representative_record_id=row["record_id"],
                        scope=scope,
                        source_entry_id=item["source_entry_id"],
                        existing_families=sorted(
                            set(item.get("existing_entry_families", []))
                            | set(item.get("existing_graph_families", []))
                        ),
                    )
                )
            selected.append({**row, "scope": scope, "members": ids, "spec": spec})
    if len(selected) != 22 or len(members) != 27 or len({r["record_id"] for r in members}) != 27:
        raise ValueError("The source universe is exactly 22 representatives / 27 original members")
    return selected, members


def reconcile_groups(ledger, members, audit):
    family_path = next(
        ROOT / name for name in audit["source_sha256"] if name.endswith("feature_rows_index.tsv")
    )
    graph_families, sequence_families = defaultdict(set), defaultdict(set)
    for row in rows(family_path):
        family = row.get("split_group", "")
        if not family.startswith("FAMILY:"):
            continue
        if row.get("canonical_identity_group"):
            graph_families[row["canonical_identity_group"]].add(family)
        if row.get("sequence_eligible") == "true" and row.get("sequence"):
            sequence_families[row["sequence"]].add(family)
    lookup = {item["record_id"]: item for item in ledger}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in members:
        item = lookup[row["representative_record_id"]]
        known = set(row["existing_families"]) | graph_families[item["structure_key"]]
        if "X" not in item["source_sequence"]:
            known |= sequence_families[item["source_sequence"]]
        row["existing_families"] = sorted(known)
        row["structure_key"] = item["structure_key"]
        for family in known:
            union(item["structure_key"], family)
        find(item["structure_key"])
    components = defaultdict(list)
    for key in list(parent):
        components[find(key)].append(key)
    names = {
        root: "IDENTITY:" + signature(sorted(values))[:24] for root, values in components.items()
    }
    for row in members:
        row["split_group"] = names[find(row["structure_key"])]
    member_lookup = {r["record_id"]: r for r in members}
    for item in ledger:
        item["split_group"] = member_lookup[item["record_id"]]["split_group"]
        item["existing_families"] = member_lookup[item["record_id"]]["existing_families"]
    return components


def build_bank(ledger, members, parent_run, run_dir, protected):
    parent_run, run_dir = Path(parent_run).resolve(), Path(run_dir).resolve()
    if parent_run == run_dir or parent_run in run_dir.parents or run_dir in parent_run.parents:
        raise ValueError("Derived output cannot overlap the immutable parent feature bank")
    base_pointer = parent_run / "features/manifest.json"
    base, original_registry = (
        read_json(base_pointer),
        read_json(parent_run / "representation_methods.json"),
    )
    _verify_artifacts(base)
    parent_sha = sha(base_pointer)
    source_cache = Path(base["cache_dir"])
    methods = [
        m
        for m in original_registry["methods"]
        if m["source_kind"] == "structure" and m["independent"]
    ]
    if len(methods) != 10 or len({m["algorithm_family_id"] for m in methods}) != 10:
        raise ValueError("Exactly ten independent structural methods must be reused")
    if base["configuration"]["rdkit_version"] != rdBase.rdkitVersion:
        raise ValueError("RDKit differs from immutable parent algorithms")
    groups = {name: copy.deepcopy(base["groups"][name]) for name in STRUCTURE_GROUPS}
    if any(meta["key_kind"] != "canonical_identity_group" for meta in groups.values()):
        raise ValueError("Sequence features cannot enter this structure-only bank")
    by_key = {r["structure_key"]: r["canonical_isomeric_smiles"] for r in ledger}
    graphs = [{"key": k, "canonical_isomeric_smiles": v} for k, v in sorted(by_key.items())]
    old_index = {
        r["key"]: int(r["row_idx"]) for r in rows(source_cache / groups["morgan"]["index"])
    }
    old_graphs = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in rows(source_cache / "structure_source_smiles.tsv")
    }
    reused = {r["key"] for r in graphs if r["key"] in old_index}
    if any(old_graphs[k] != by_key[k] for k in reused):
        raise ValueError("Parent structure key collides with a different complete graph")
    configuration = dict(
        version=VERSION,
        implementation_sha256=sha(__file__),
        parent_manifest_sha256=parent_sha,
        parent_fingerprint=base["input_fingerprint"],
        protected_source_files=protected,
        graphs=graphs,
        member_mapping=members,
        rdkit_version=rdBase.rdkitVersion,
        numpy_version=np.__version__,
        source_measured_smiles=False,
        sequence_policy="No projected sequence or ESM features constructed",
    )
    fp = signature(configuration)
    cache = run_dir / "features" / fp[:16]
    cache.mkdir(parents=True, exist_ok=True)
    extracted = {
        r["key"]: extract_all(r["canonical_isomeric_smiles"], r["key"])
        for r in graphs
        if r["key"] not in reused
    }
    for name, meta in groups.items():
        old = np.load(safe_path(source_cache, meta["matrix"]), mmap_mode="r")
        valid = np.load(safe_path(source_cache, meta["valid"]), mmap_mode="r")
        values = np.asarray(
            [
                old[old_index[r["key"]]] if r["key"] in reused else extracted[r["key"]][0][name]
                for r in graphs
            ],
            dtype=meta["dtype"],
        )
        masks = np.asarray(
            [
                valid[old_index[r["key"]]] if r["key"] in reused else extracted[r["key"]][1][name]
                for r in graphs
            ],
            dtype=bool,
        )
        meta.update(shape=[len(graphs), meta["shape"][1]], index="structure_bank_index.tsv")
        check_matrix(values, masks, meta, name)
        atomic_array(safe_path(cache, meta["matrix"]), values)
        atomic_array(safe_path(cache, meta["valid"]), masks)
        for i, item in enumerate(graphs):
            if item["key"] in reused and not np.array_equal(
                np.ascontiguousarray(old[old_index[item["key"]]]).view(np.uint8),
                np.ascontiguousarray(values[i]).view(np.uint8),
            ):
                raise ValueError("Reused parent vector bytes changed")
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
            for r in members
        ],
    )
    registry = {
        **copy.deepcopy(original_registry),
        "methods": methods,
        "independent_method_count": 10,
        "bank_scope": "derived_source_chemical_specification_structures_only",
    }
    write_json(cache / "representation_methods.json", registry)
    write_json(cache / "derivation_ledger.json", ledger)
    write_json(cache / "member_mapping.json", members)
    write_json(
        cache / "component_registry.json",
        {
            key: {
                **expected,
                "reference": read_json(run_dir / "component_references" / (key + ".receipt.json")),
            }
            for key, expected in COMPONENTS.items()
        },
    )
    write_json(cache / "source_parent_manifest.json", base)
    write_json(cache / "extraction_audit.json", {k: v[2] for k, v in extracted.items()})
    routes = {m["route"]: m["group_names"] for m in methods}
    routes["structure_methods_all"] = list(STRUCTURE_GROUPS)
    manifest = dict(
        schema=VERSION,
        input_fingerprint=fp,
        cache_dir=str(cache),
        manifest_path=str(cache / "manifest.json"),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=len(members),
        structure_keys=len(graphs),
        groups=groups,
        routes=routes,
        representation_methods=registry,
        scope="Source chemical-format derivation only; not admission or model training",
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p)
        for p in cache.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    task_rows = [
        {
            "record_id": r["record_id"],
            "structure_eligible": "true",
            "sequence_eligible": "false",
            "canonical_identity_group": r["structure_key"],
        }
        for r in members
    ]
    for method in methods:
        loaded = load_feature_route(task_rows, manifest, method["route"])
        if not loaded["valid"].all() or loaded["X"].shape != (len(members), method["dimensions"]):
            raise ValueError("Published method did not load every original member")
    old_records = {r["record_id"] for r in rows(source_cache / base["record_index"])}
    _verify_artifacts(base)
    if sha(base_pointer) != parent_sha:
        raise ValueError("Parent feature pointer changed")
    write_json(run_dir / "features/manifest.json", manifest)
    write_json(run_dir / "representation_methods.json", registry)
    proof = dict(
        status="verified",
        records=len(members),
        representatives=len(ledger),
        graphs=len(graphs),
        reused_graphs=len(reused),
        extracted_graphs=len(extracted),
        source_id_overlap_with_parent=len(old_records & {r["record_id"] for r in members}),
        independent_structure_methods=10,
        groups=13,
        dimensions=sum(m["shape"][1] for m in groups.values()),
        artifact_hashes_verified=len(manifest["artifact_hashes"]),
        parent_artifacts_unchanged=len(base["artifact_hashes"]),
        parent_pointer_unchanged=True,
        reused_vectors_byte_identical=True,
        all_members_all_methods_loaded=True,
        projected_sequence_features_constructed=False,
        model_training=False,
        source_measured_smiles=False,
        feature_fingerprint=fp,
    )
    write_json(run_dir / "feature_verification.json", proof)
    return proof


def prepare(source=SOURCE, parent_run=PARENT, run_dir=OUTPUT):
    source, run_dir = Path(source), Path(run_dir)
    selected, members, protected, audit = validate_sources(source)
    components = registered_components(run_dir / "component_references")
    ledger = []
    for row in selected:
        graph = derive_graph(row["spec"], row["source_paper"], components)
        ledger.append(
            dict(
                record_id=row["record_id"],
                scope=row["scope"],
                members=row["members"],
                source_sequence=row["spec"]["sequence"],
                source_chemical_specification=row["spec"],
                source_paper=row["source_paper"],
                cell=row["cell"],
                model_value=row["model_value"],
                model_unit=row["model_unit"],
                target=row["target"],
                **graph,
            )
        )
    reconcile_groups(ledger, members, audit)
    write_json(run_dir / "derivation_ledger.json", ledger)
    write_json(run_dir / "member_mapping.json", members)
    graph_fields = [
        "record_id",
        "scope",
        "structure_key",
        "canonical_isomeric_smiles",
        "split_group",
    ]
    write_tsv(
        run_dir / "derived_graph_rows.tsv",
        graph_fields,
        [{name: item[name] for name in graph_fields} for item in ledger],
    )
    counts = {
        scope: dict(
            representatives=sum(r["scope"] == scope for r in ledger),
            original_members=sum(r["scope"] == scope for r in members),
            full_graphs=len({r["structure_key"] for r in ledger if r["scope"] == scope}),
            reconciled_groups=len({r["split_group"] for r in ledger if r["scope"] == scope}),
        )
        for scope in SCOPES
    }
    write_json(run_dir / "scope_identity_counts.json", counts)
    proof = build_bank(ledger, members, parent_run, run_dir, protected)
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("Protected original source changed during graph derivation")
    write_json(
        run_dir / "derivation_receipt.json",
        dict(
            status="derived_graphs_and_features_verified_not_admitted",
            chemical_derivation=True,
            source_measured_smiles=False,
            independent_assemblies_equal=True,
            scope_identity_counts=counts,
            protected_source_files_unchanged=len(protected),
            feature_verification=proof,
            model_training=False,
            training_admitted=False,
            canonical_sequence_or_esm_projection=False,
            limitations=[
                "Neutral covalent graphs derived from source chemistry; not measured source SMILES",
                "Source biological protocol and article tables are not independently validated",
                "Existing families are conservatively joined; no scaffold/homology claim",
            ],
        ),
    )
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.parent, args.output)))


if __name__ == "__main__":
    main()
