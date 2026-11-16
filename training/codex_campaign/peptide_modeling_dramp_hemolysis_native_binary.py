"""DRAMP native hemolytic presence annotations, with complete reference graphs.

The admitted source contract is29 observations (7 affirmative/22 absence), not a
uniform physiological threshold.  Preparation and reporting are separate from
the explicitly named diagnostic-learning phase; reporting never fits/unpickles.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import hashlib
import inspect
import json
import math
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ADMISSION = ROOT / "data/dramp_native_binary_admission_review_20261115_v1"
STRUCTURE = ROOT / "data/dramp_native_positive_structure_review_20261115_v1"
NEGATIVE = ROOT / "data/dramp_binary_source_recovery_20261115_v1"
NEGATIVE_FEATURES = ROOT / "data/peptide_modeling_dramp_binary_reference_features_20261115_v1"
PARENT = ROOT / "data/peptide_modeling_unified_features_20261114_v1"
OUTPUT = ROOT / "data/peptide_modeling_dramp_hemolysis_native_binary_followup_20261115_v1"
ADMISSION_SHA = "ce205a474834d69406abff84bd9e47d3490b7f1b329125f2e7060f88c6ecf74d"
ADMISSION_DELIVERY_SHA = "3766da4228112013b8b52f36ffcc1930de90fb886fc54620ec9fc0aba53c7768"
STRUCTURE_DELIVERY_SHA = "46690e5e478573e905e203b73bb90a2f1a273f727e828e800ad04dc7252d7d14"
NEGATIVE_OBJECT_SHA = "b8d162ae695a991a60160e3c633bcf1ea09ca35cc67e92783b14333e7082ceeb"
PARENT_POINTER_SHA = "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
NEGATIVE_FINGERPRINT = "7299d652414e8dca81c7265bb3ed6319788bcb397511d700f11a5c85cacee475"
TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
SCHEMA = "DRAMP_native_hemolytic_annotation_graph_only_v1"
QUANTITY = "DRAMP_source_native_hemolytic_presence_judgment_given_complete_reference_graph"
QUERY_MODE = "source_native_annotation_registered_reference_only"
REPORT_SCHEMA = "DRAMP_native_binary_graph_only_calibrated_diagnostics_v1"
SEEDS = (42, 2026, 3407)
MODELS = (
    "linear",
    "elasticnet",
    "random_forest",
    "extra_trees",
    "histgb",
    "lightgbm",
    "xgboost",
    "svm",
)
CHEMICAL_GROUPS = (
    "morgan",
    "descriptors",
    "graph_modifications",
    "morgan_r3",
    "atompair",
    "torsion",
    "maccs",
    "rdkit_full",
    "rdkit_path",
    "pattern",
    "layered",
    "estate",
    "erg",
)
ALGORITHMS = (
    "peptide_modeling_clint_features.py",
    "peptide_modeling_expand_features.py",
    "peptide_modeling_add_methods.py",
    "peptide_modeling_add_smiles_methods.py",
)
POLICY = dict(
    schema="native_binary_diagnostic_learning_policy_v1",
    input_feature_scope="complete_chemical_graph_only_no_context_covariates",
    fixed_controls=["molecule_only"],
    context_control_omission="no_nonlabel_context_features_exist",
    fixed_logistic=dict(
        C=1.0,
        solver="lbfgs",
        tol=1e-8,
        max_iter=5000,
        class_weight=None,
        route="structure_methods_all",
        own_route_search=False,
        imputer="median_add_indicator_keep_empty",
        scaler_with_mean=False,
        X_dtype="float64_exact_upcast_of_stored_float32",
        coefficient_dtype="float64",
        identical_input_probability_atol=1e-12,
        identical_input_probability_rtol=1e-10,
        folds_per_seed=6,
        fits_three_seeds=18,
    ),
    platt=dict(
        input="logit_raw_probability",
        clip=1e-6,
        C=1.0,
        solver="lbfgs",
        tol=1e-10,
        max_iter=5000,
        sample_weights="inverse_known_parent_size_normalized_mean_one",
        coefficient_dtype="float64",
        fit_count_three_seeds=6,
    ),
    threshold=dict(
        candidate_rule="sorted_unique_0_0.5_1_probabilities_and_adjacent_midpoints",
        objective="known_parent_equal_MCC",
        tie_score_tolerance=1e-12,
        tie_break="closest_to_0.5_then_lower_threshold",
    ),
    calibration_bins=5,
    evaluation_calibration_scope="outer_train_inner_OOF_only_never_holdout_labels",
    evaluation_calibrated_inner_metrics_role="calibration_and_threshold_fit_diagnostic_not_independent",
    deployment_calibration_scope="full29_fixed_selected_params_grouped_OOF5_no_new_search",
    deployment_selected_parameter_CV_fits_three_seeds=15,
    deployment_OOF_metrics_role="deployment_calibration_fit_diagnostic_not_independent_evaluation",
    all_diagnostic_learning_fits_three_seeds=39,
    diagnostic_report_stage_model_fits=0,
    diagnostic_report_stage_model_loads=0,
)
WARNING = (
    "Source-native affirmative/absence hemolytic annotations under their recorded source contexts, "
    "given registered or explicit literal complete reference graphs. This is not a common "
    "RBC assay, "
    "uniform physiological cutoff, clinical toxicity probability, tolerated dose, verified primary "
    "sample identity, redox/salt/tautomer/purity or cross-species effect. Unknown materials, "
    "species, "
    "doses and exposure remain unknown and are provenance only. SwissProt function-only, low/weak, "
    "significance, mixed, collection, crab-haemocyte and numerical-threshold constructions are not "
    "labels. Original32 negatives are preserved:22 participate, PACAP remains a scoped-only card, "
    "four graphs are unresolved, two mixed, one crab and two collection records stay outside. "
    "Known-parent union crosses labels (Stripe) and includes CRAMP18, Temporin-La, APOC1, TT and "
    "mini-ChBac pairs; it does not prove scaffold or homology isolation. Outer holdout evaluates "
    "only the outer-train model plus calibration/threshold learned from its innerOOF. Calibrated "
    "OOF fit diagnostics are not independent accuracy. Separate all29 refits and their full29 OOF "
    "deployment calibration have no independent scores on their included records."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def table(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(records), "An empty evidence table is not admissible")
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)


def contained(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    path = (p if p.is_absolute() else root / p).resolve()
    require(path != root and path.is_relative_to(root), "Evidence path escapes its frozen root")
    return path


def array_hash(value):
    value = np.ascontiguousarray(value)
    return dict(
        dtype=str(value.dtype),
        shape=list(value.shape),
        bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


TASK = (
    "TASK_DRAMP_NATIVE_BINARY_"
    + object_hash(
        dict(quantity=QUANTITY, admission=ADMISSION_SHA, structure=STRUCTURE_DELIVERY_SHA, scope=29)
    )[:24]
)


def source_frame(admission=ADMISSION, structure=STRUCTURE, negative=NEGATIVE):
    """Freeze all29 approved native observations; preserve the original32 disposition ledger."""
    admission, structure, negative = map(Path, (admission, structure, negative))
    require(
        sha(admission / "delivery_receipt.json") == ADMISSION_DELIVERY_SHA,
        "Independent source-admission delivery changed",
    )
    for name, pin in read(admission / "delivery_receipt.json")["files"].items():
        require(
            sha(contained(admission, name)) == pin,
            "An independent source-admission artifact changed",
        )
    require(sha(admission / "admission_review.json") == ADMISSION_SHA, "Admission contract changed")
    require(
        sha(structure / "delivery_receipt.json") == STRUCTURE_DELIVERY_SHA,
        "Complete-reference structure delivery changed",
    )
    for name, pin in read(structure / "delivery_receipt.json")["files"].items():
        require(
            sha(contained(structure, name)) == pin, "A frozen complete-reference artifact changed"
        )
    negative_path = negative / "fifty_two_source_objects_and_reference_graphs.json"
    require(sha(negative_path) == NEGATIVE_OBJECT_SHA, "Original52 negative source objects changed")
    review = read(admission / "admission_review.json")
    contract = review["hemolysis_field_aware_native_inactivity_extension"]
    require(
        (
            contract["total_rows"],
            contract["positive_rows"],
            contract["negative_rows"],
            contract["total_components"],
            contract["class_group_counts"],
        )
        == (29, 7, 22, 19, {"0": 13, "1": 7}),
        "The29-row extension was narrowed or enlarged",
    )
    components = contract["components"]
    ids = {i for members in components.values() for i in members}
    reviewed = read(admission / "hemolysis_native_candidate_observations_review_only.json")
    extension = read(admission / "six_original_native_inactivity_field_extension.json")
    extension_ids = {r["source_id"] for r in extension}
    reviewed += extension
    selected = {r["source_id"]: r for r in reviewed if r["source_id"] in ids}
    require(len(selected) == len(ids) == 29, "Every approved extended source object is required")
    positives = {r["source_id"]: r for r in read(structure / "source_structure_cards.json")}
    old = {
        r["record"]["source_entry_id"]: r
        for r in read(negative_path)
        if r["record"]["endpoint"] == "hemolysis_SOURCE_DEFINED_BINARY"
    }
    require(len(old) == 32, "Original32 hemolysis negatives are not preserved")
    context = {r["source_id"]: r for r in read(admission / "per_source_context_role_review.json")}
    grouping = {
        i: "DRAMP_NATIVE_GROUP:" + hashlib.sha256("||".join(sorted(m)).encode()).hexdigest()[:24]
        for m in components.values()
        for i in m
    }
    records, evidence = [], []
    for source_id in sorted(ids):
        item = selected[source_id]
        label = int(item["reviewed_native_polarity"])
        require(label in {0, 1} and item["endpoint"] == "hemolysis", "Non-native binary judgment")
        for span in item["polarity_spans"]:
            require(
                item["annotation"][span["start"] : span["end"]] == span["exact_text"],
                "Native polarity span changed",
            )
        if label:
            source_card = positives[source_id]
            graph = source_card["candidate_graph"]
            require(
                graph is not None
                and source_card["raw_affirmative_evidence"]["source_reference_tier"]
                == "DRAMP_secondary_curated_annotation_not_primary_verified",
                "SwissProt-only or unclosed positive graph entered the task",
            )
            smiles, key = graph["canonical_isomeric_smiles"], graph["whole_graph_key"]
            native = source_card["raw_source_object"]
            memberships = json.loads(
                source_card["raw_affirmative_evidence"]["exact_catalogue_memberships"]
            )
            original_ids = [r["record_id"] for r in memberships]
            tier = graph["identity_tier"]
        else:
            source_card = old[source_id]
            graph = source_card["reference_graph"]
            require(
                graph is not None and source_card["record"]["value"] == "0",
                "Negative graph/label changed",
            )
            smiles, key = graph["canonical_isomeric_smiles"], graph["graph_key"]
            native = source_card["native_source_object"]
            original_ids = [source_card["record"]["record_id"]]
            tier = graph["identity_tier"]
        require(
            key
            == item["whole_graph_key"]
            == "CANON_SMILES:" + hashlib.sha256(smiles.encode()).hexdigest(),
            "Source-native label no longer binds its exact complete reference graph",
        )
        rid = "DRAMP_NATIVE:" + hashlib.sha256(item["source_entry_key"].encode()).hexdigest()[:24]
        material = (
            " | ".join(s["exact_text"] for s in context[source_id]["material_spans"])
            or "not_reported"
        )
        species = (
            " | ".join(s["exact_text"] for s in context[source_id]["assay_species_word_spans"])
            or "not_reported"
        )
        row = dict(
            record_id=rid,
            target=str(label),
            value=str(label),
            unit="label01",
            target_transform="identity",
            inverse_transform="identity",
            source_entry_id=source_id,
            source_entry_key=item["source_entry_key"],
            canonical_smiles=smiles,
            smiles=smiles,
            canonical_identity_group=key,
            structure_smiles_sha256=hashlib.sha256(smiles.encode()).hexdigest(),
            split_group=grouping[source_id],
            structure_eligible="true",
            sequence_eligible="false",
            sequence="",
            sequence_kind="complete_SMILES_only_no_standard_projection",
            canonical_quantity_id=QUANTITY,
            identity_tier=tier,
            original_catalogue_record_ids=json.dumps(original_ids),
            assay_material_literal=material,
            assay_species_literal=species,
            native_annotation=item["annotation"],
            polarity_spans_json=json.dumps(item["polarity_spans"], ensure_ascii=False),
            context_roles_json=json.dumps(context[source_id], ensure_ascii=False, sort_keys=True),
            original_negative_label_unchanged="true" if label == 0 else "not_an_original_negative",
            source_subset="field_aware_explicit_inactivity_extension"
            if source_id in extension_ids
            else "literal_native_judgment_core",
            primary_sample_identity_verified="false",
            annotation_only="true",
        )
        records.append(row)
        evidence.append(
            dict(
                record_id=rid,
                source_id=source_id,
                native_judgment=item,
                source_object=native,
                complete_reference=graph,
                original_catalogue_record_ids=original_ids,
                context_roles=context[source_id],
            )
        )
    validate_rows(records)
    dispositions = read(admission / "original_32_negative_compatibility_and_coverage.json")
    require(
        len(dispositions) == 32 and len({r["record_id"] for r in dispositions}) == 32,
        "Original32 source disposition ledger changed",
    )
    included = {r["source_entry_id"] for r in records if r["target"] == "0"}
    for item in dispositions:
        item["included_in_this29_source_annotation_task"] = item["source_id"] in included
        require(
            item["old_label"] == "0" and item["old_label_unchanged"] is True,
            "An original negative was changed",
        )
    require(
        sum(r["included_in_this29_source_annotation_task"] for r in dispositions) == 22,
        "Original32 participation cannot be claimed as32/32",
    )
    return records, evidence, dispositions, review


def validate_rows(records):
    require(
        len(records) == len({r["record_id"] for r in records}) == 29,
        "The complete29 observations are mandatory",
    )
    require(
        Counter(r["target"] for r in records) == {"0": 22, "1": 7}
        and len({r["split_group"] for r in records}) == 19,
        "Native class/known-parent counts changed",
    )
    require(
        {k: len({r["split_group"] for r in records if r["target"] == k}) for k in ("0", "1")}
        == {"0": 13, "1": 7},
        "Native class-group support changed",
    )
    by_graph = defaultdict(set)
    for r in records:
        require(
            r["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(r["canonical_smiles"].encode()).hexdigest()
            and r["structure_eligible"] == "true"
            and r["sequence_eligible"] == "false"
            and not r["sequence"]
            and r["canonical_quantity_id"] == QUANTITY,
            "Complete source graphs or graph-only inputs changed",
        )
        by_graph[r["canonical_identity_group"]].add((r["target"], r["split_group"]))
    require(all(len(v) == 1 for v in by_graph.values()), "Same-wholegraph label/group conflict")


def training_config():
    return dict(
        routes=[
            "method_" + x
            for x in (
                "morgan",
                "atom_pair",
                "topological_torsion",
                "maccs",
                "rdkit_descriptors",
                "rdkit_path",
                "pattern",
                "layered",
                "estate_fingerprint",
                "erg",
            )
        ]
        + ["structure_methods_all"],
        models=list(MODELS),
        inner_folds=5,
        test_size=0.2,
        min_rows=10,
        min_groups=8,
        min_train_rows=6,
        min_class_count=5,
        seed=42,
        search_backend="optuna",
        search_budget=128,
        n_jobs=1,
        refit_full=True,
        svm_max_features=16384,
        histgb_max_features=16384,
        elasticnet_max_features=16384,
    )


def split_plans(records, settings, engine=None):
    """The real stock splitter; no estimator is constructed or fitted here."""
    if engine is None:
        import peptide_modeling_train as engine
    y, groups = engine.validate_rows(records, "classification")
    output = []
    with tempfile.TemporaryDirectory(prefix="native_binary_nofit_split_") as scratch:
        for seed in SEEDS:
            cfg = {**settings, "seed": seed}
            split = engine.freeze_outer_split(
                records, "classification", Path(scratch) / f"{seed}.json", cfg
            )
            train = [i for i, r in enumerate(records) if r["record_id"] in split["train_records"]]
            held = [i for i, r in enumerate(records) if r["record_id"] in split["test_records"]]
            inner = engine._inner_splits(y[train], groups[train], "classification", cfg)
            full = engine._inner_splits(y, groups, "classification", cfg)
            require(
                len(inner) == len(full) == 5
                and min(Counter(y[train]).values()) >= 5
                and min(Counter(y[held]).values()) >= 2,
                "Stock holdout/class constraints failed",
            )

            def encode(splits, indices):
                return [
                    dict(
                        fold=f,
                        train_record_ids=[records[indices[i]]["record_id"] for i in a],
                        validation_record_ids=[records[indices[i]]["record_id"] for i in b],
                    )
                    for f, (a, b) in enumerate(splits)
                ]

            output.append(
                dict(
                    seed=seed,
                    actual_outer_seed=split["seed"],
                    outer_split=split,
                    outer_train_record_ids=[records[i]["record_id"] for i in train],
                    outer_holdout_record_ids=[records[i]["record_id"] for i in held],
                    inner_folds=encode(inner, train),
                    full29_oof_folds=encode(full, list(range(29))),
                    outer_train_class_rows={
                        str(int(k)): int(v) for k, v in Counter(y[train]).items()
                    },
                    outer_holdout_class_rows={
                        str(int(k)): int(v) for k, v in Counter(y[held]).items()
                    },
                    selected_model_fit_scope="outer_train_only",
                    all29_refit_required=True,
                )
            )
    return dict(
        protocol="stock_group_holdout_inner5_with_separate_full29_selected_parameter_OOF5",
        seeds=output,
        requested_seeds=list(SEEDS),
        performance_used_for_split_choice=False,
        model_fits=0,
        full29_selected_parameter_OOF_is_deployment_calibration_not_holdout=True,
    )


def node_split_plans(records, settings):
    """Ephemeral stdin-only no-fit execution of the pinned stock splitter on node23."""
    trainer = ROOT / "scripts/peptide_modeling_train.py"
    require(sha(trainer) == TRAIN_SHA, "Frozen stock trainer changed")
    source = inspect.getsource(split_plans)
    payload = dict(
        records=records,
        settings=settings,
        helper=source,
        trainer=base64.b64encode(trainer.read_bytes()).decode(),
        trainer_sha256=TRAIN_SHA,
    )
    script = (
        "import json,base64,types,hashlib,sys,socket,tempfile,platform\n"
        "from pathlib import Path\nfrom collections import Counter\n"
        "import numpy as np\nimport sklearn\n"
        "p=" + repr(payload) + "\n"
        "assert socket.gethostname()=='node23' and sys.flags.dont_write_bytecode\n"
        "src=base64.b64decode(p['trainer']);"
        "assert hashlib.sha256(src).hexdigest()==p['trainer_sha256']\n"
        "m=types.ModuleType('native_binary_frozen_trainer');m.__file__='<stdin_frozen_37bd_trainer>';"
        "sys.modules[m.__name__]=m;exec(compile(src,m.__file__,'exec'),m.__dict__)\n"
        "def forbid(*a,**k):raise AssertionError('Fitting/model loading forbidden')\n"
        "m._fit_model=forbid;m._estimator=forbid;m.joblib.load=forbid\n"
        "SEEDS=(42,2026,3407)\n"
        "def require(v,s):\n if not v:raise ValueError(s)\n"
        "exec(compile(p['helper'],'<split_plan_helper>','exec'))\n"
        "result=split_plans(p['records'],p['settings'],m)\n"
        "result['runtime']={'host':socket.gethostname(),'python':platform.python_version(),"
        "'numpy':np.__version__,'sklearn':sklearn.__version__,'trainer_sha256':p['trainer_sha256'],"
        "'production_fits':0,'model_loads':0,'no_bytecode':True}\n"
        "print(json.dumps(result,allow_nan=False))\n"
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "node23",
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            "OPENBLAS_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "/usr/bin/python3",
            "-B",
            "-",
        ],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=55,
    )
    require(result.returncode == 0, "Actual node split preflight failed: " + result.stderr[-3000:])
    return json.loads(result.stdout)


def source_snapshots(output, admission=ADMISSION, structure=STRUCTURE, negative=NEGATIVE):
    root = Path(output) / "task_tables/source_snapshots"
    for name, folder, receipt_sha in (
        ("admission", Path(admission), ADMISSION_DELIVERY_SHA),
        ("structure", Path(structure), STRUCTURE_DELIVERY_SHA),
    ):
        receipt = folder / "delivery_receipt.json"
        require(sha(receipt) == receipt_sha, "Source snapshot delivery changed")
        destination = root / name
        destination.mkdir(parents=True, exist_ok=False)
        for relative, pin in read(receipt)["files"].items():
            source = contained(folder, relative)
            target = contained(destination, relative)
            require(sha(source) == pin, "Source artifact changed during snapshot")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(receipt, destination / "delivery_receipt.json")
    destination = root / "negative"
    destination.mkdir()
    source = Path(negative) / "fifty_two_source_objects_and_reference_graphs.json"
    require(sha(source) == NEGATIVE_OBJECT_SHA, "Original source52 changed during snapshot")
    shutil.copy2(source, destination / source.name)
    return root


def build_feature_bank(records, output, negative_features=NEGATIVE_FEATURES, parent=PARENT):
    """Copy all22 negative reference rows; compute exactly seven complete positive graphs."""
    from peptide_modeling_clint_features import atomic_array, extract_all
    from peptide_modeling_dramp_binary_reference_features import canary_check, checked_vector
    from peptide_modeling_features import _verify_artifacts, load_feature_route
    from threadpoolctl import threadpool_info, threadpool_limits

    output, parent, negative_features = map(Path, (output, parent, negative_features))
    pointer = parent / "features/manifest.json"
    require(sha(pointer) == PARENT_POINTER_SHA, "Frozen parent pointer changed")
    original = read(pointer)
    require(len(original["artifact_hashes"]) == 48, "Parent active artifact scope changed")
    for relative, pin in original["artifact_hashes"].items():
        require(
            sha(contained(original["cache_dir"], relative)) == pin,
            "Parent artifact changed before extraction",
        )
    negative_pointer = negative_features / "features/manifest.json"
    negative_pointer_sha = sha(negative_pointer)
    base = read(negative_pointer)
    require(
        base["input_fingerprint"] == NEGATIVE_FINGERPRINT,
        "Frozen negative reference feature bank changed",
    )
    _verify_artifacts(base)
    algorithms = base["configuration"]["algorithm_contract"]
    for name, pin in algorithms["extractor_source_sha256"].items():
        require(
            sha(ROOT / "scripts" / name) == pin, "Frozen structural extraction algorithm changed"
        )
    from peptide_modeling_add_smiles_methods import rdkit_tools

    require(
        rdkit_tools()["version"] == algorithms["rdkit_version"], "RDKit feature version changed"
    )
    registry = copy.deepcopy(base["representation_methods"])
    methods = registry["methods"]
    require(
        len(methods) == registry["independent_method_count"] == 10
        and len({m["algorithm_family_id"] for m in methods}) == 10
        and all(m["source_kind"] == "structure" and m["independent"] is True for m in methods),
        "Ten independent chemical methods are mandatory",
    )
    configuration = dict(
        schema=SCHEMA,
        implementation_sha256=sha(__file__),
        implementation_dependencies_sha256=algorithms["extractor_source_sha256"],
        admission_review_sha256=ADMISSION_SHA,
        structure_delivery_sha256=STRUCTURE_DELIVERY_SHA,
        parent_pointer_sha256=PARENT_POINTER_SHA,
        parent_fingerprint=original["input_fingerprint"],
        negative_pointer_sha256=negative_pointer_sha,
        negative_fingerprint=NEGATIVE_FINGERPRINT,
        record_ids=[r["record_id"] for r in records],
        graphs={r["record_id"]: r["canonical_identity_group"] for r in records},
        contexts_used_as_covariates=False,
        standard_sequence_projection_used=False,
        positive_new_extraction_graphs=7,
        negative_reused_graphs=22,
        chemical_dimensions=15212,
        algorithm_contract=algorithms,
        diagnostic_policy_sha256=object_hash(POLICY),
    )
    fingerprint = object_hash(configuration)
    cache = output / "features" / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=False)
    groups = {name: copy.deepcopy(base["groups"][name]) for name in CHEMICAL_GROUPS}
    require(
        sum(g["shape"][1] for g in groups.values()) == 15212, "Chemical schema dimension changed"
    )
    base_cache = Path(base["cache_dir"])
    index = {r["key"]: int(r["row_idx"]) for r in rows(base_cache / "structure_bank_index.tsv")}
    source_graphs = {
        r["key"]: r["canonical_isomeric_smiles"]
        for r in rows(base_cache / "structure_source_smiles.tsv")
    }
    graph_map = {r["canonical_identity_group"]: r["canonical_smiles"] for r in records}
    require(len(graph_map) == 29, "Source29 graphs must remain distinct covalent references")
    negative_graphs = [
        dict(key=r["canonical_identity_group"], canonical_isomeric_smiles=r["canonical_smiles"])
        for r in records
        if r["target"] == "0"
    ]
    positives = [r for r in records if r["target"] == "1"]
    require(
        len(negative_graphs) == 22
        and all(
            source_graphs.get(r["key"]) == r["canonical_isomeric_smiles"] for r in negative_graphs
        ),
        "An original negative is not bound to its exact cached graph",
    )
    with threadpool_limits(limits=1):
        canaries = canary_check(base, negative_graphs, index)
        computed = {
            r["canonical_identity_group"]: extract_all(
                r["canonical_smiles"], r["canonical_identity_group"]
            )
            for r in positives
        }
        thread_info = threadpool_info()
    require(
        len(computed) == 7 and all(t["num_threads"] == 1 for t in thread_info),
        "Positive extraction or BLAS thread scope changed",
    )
    keys = sorted(graph_map)
    negative_keys = {r["key"] for r in negative_graphs}
    reuse_rows = defaultdict(dict)
    for name, meta in groups.items():
        old = np.load(base_cache / meta["matrix"], mmap_mode="r", allow_pickle=False)
        masks = np.load(base_cache / meta["valid"], allow_pickle=False)
        matrix = np.empty((29, meta["shape"][1]), dtype=np.dtype(meta["dtype"]))
        for i, key in enumerate(keys):
            if key in negative_keys:
                vector = checked_vector(old[index[key]], masks[index[key]], meta, name)
                require(
                    vector.tobytes() == old[index[key]].tobytes(),
                    "A copied negative vector changed bytes",
                )
                reuse_rows[key][name] = hashlib.sha256(vector.tobytes()).hexdigest()
            else:
                values, valid, _ = computed[key]
                vector = checked_vector(values[name], valid[name], meta, name)
            matrix[i] = vector
        meta["shape"][0], meta["index"] = 29, "structure_bank_index.tsv"
        atomic_array(cache / meta["matrix"], matrix)
        atomic_array(cache / meta["valid"], np.ones(29, dtype=bool))
    table(cache / "structure_bank_index.tsv", [dict(key=k, row_idx=i) for i, k in enumerate(keys)])
    table(
        cache / "structure_source_smiles.tsv",
        [dict(key=k, canonical_isomeric_smiles=graph_map[k]) for k in keys],
    )
    table(
        cache / "record_bank_index.tsv",
        [
            dict(
                record_id=r["record_id"],
                structure_key=r["canonical_identity_group"],
                sequence_key="",
            )
            for r in records
        ],
    )
    registry.update(
        bank_scope="native_source_hemolysis_binary_graph_only",
        context_is_additional_chemical_method=False,
    )
    write(cache / "representation_methods.json", registry)
    write(cache / "source_negative_manifest.json", base)
    write(cache / "source_parent_manifest.json", original)
    write(
        cache / "negative_exact_vector_reuse.json",
        dict(
            rows=reuse_rows,
            copied_graphs=22,
            all13_groups_byte_identical=True,
            negative_source_fingerprint=NEGATIVE_FINGERPRINT,
        ),
    )
    write(
        cache / "positive_extraction_receipt.json",
        dict(
            graphs=sorted(computed),
            graphs_extracted=7,
            errors={k: v[2] for k, v in computed.items()},
            chemistry_canaries=canaries,
            runtime_threadpools=thread_info,
            sequence_projection_used=False,
        ),
    )
    manifest = dict(
        schema=SCHEMA,
        input_fingerprint=fingerprint,
        cache_dir=str(cache.resolve()),
        manifest_path=str((cache / "manifest.json").resolve()),
        configuration=configuration,
        record_index="record_bank_index.tsv",
        requested_records=29,
        structure_keys=29,
        groups=groups,
        representation_methods=registry,
        routes={
            **{m["route"]: m["group_names"] for m in methods},
            "structure_methods_all": list(CHEMICAL_GROUPS),
        },
    )
    manifest["artifact_hashes"] = {
        p.relative_to(cache).as_posix(): sha(p) for p in sorted(cache.rglob("*")) if p.is_file()
    }
    write(cache / "manifest.json", manifest)
    _verify_artifacts(manifest)
    for route in manifest["routes"]:
        loaded = load_feature_route(records, manifest, route)
        require(
            loaded["record_ids"] == [r["record_id"] for r in records]
            and loaded["valid"].all()
            and loaded["X"].shape[0] == 29,
            "All ten routes must use identical29 rows and masks",
        )
    # Retain the complete small38-graph negative cache as portable byte-reuse evidence.
    witness = output / "task_tables/source_snapshots/negative_features"
    witness.mkdir(parents=True, exist_ok=False)
    for relative, pin in base["artifact_hashes"].items():
        source = contained(base_cache, relative)
        destination = contained(witness, relative)
        require(sha(source) == pin, "Negative feature witness changed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(base_cache / "manifest.json", witness / "manifest.json")
    write(output / "features/manifest.json", manifest)
    write(output / "representation_methods.json", registry)
    for relative, pin in original["artifact_hashes"].items():
        require(
            sha(contained(original["cache_dir"], relative)) == pin,
            "A parent artifact changed during extraction",
        )
    require(
        sha(pointer) == PARENT_POINTER_SHA and sha(negative_pointer) == negative_pointer_sha,
        "Parent/negative feature pointer changed",
    )
    write(
        output / "feature_verification.json",
        dict(
            status="verified_not_trained",
            records=29,
            negative_graphs_copied=22,
            positive_graphs_extracted=7,
            independent_methods=10,
            chemical_groups=13,
            chemical_dimensions=15212,
            context_dimensions=0,
            all_route_rows_and_masks_identical=True,
            parent48_artifacts_unchanged=True,
            parent_pointer_unchanged=True,
            negative_reference_bank_unchanged=True,
            numeric_signature_nan_safe=True,
            model_fits=0,
            canaries=canaries,
        ),
    )
    return manifest


def script_closure():
    root, seen = ROOT / "scripts", set()
    pending = [
        Path(__file__).stem,
        "peptide_modeling_followup_seeds",
        "peptide_modeling_dispatch",
        "peptide_modeling_supervise",
        "peptide_modeling_pipeline",
    ]
    while pending:
        name = pending.pop()
        path = root / (name + ".py")
        if name in seen or not path.is_file():
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = (
                [v.name.split(".")[0] for v in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0]]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            pending += [m for m in modules if (root / (m + ".py")).is_file()]
    return sorted(seen)


def portable_features(root):
    from peptide_modeling_pipeline import portable_features as load

    return load(Path(root))


def verify_source_proof(root, task, records):
    """Portable source/label/parent proof: no original XLSX, RDKit, fitting or unpickling."""
    root = Path(root)
    validate_rows(records)
    require(
        task["task_id"] == TASK
        and task["task_kind"] == "classification"
        and task["endpoint"] == "hemolysis_SOURCE_DEFINED_BINARY"
        and task["canonical_quantity_id"] == QUANTITY,
        "Source-native task/quantity changed",
    )
    descriptor = task["source_native_binary_observation_coverage"]
    path = contained(root, descriptor["proof_path"])
    require(sha(path) == descriptor["proof_sha256"], "Source-native binary proof changed")
    proof = read(path)
    require(
        proof["schema"] == SCHEMA
        and proof["task_id"] == TASK
        and proof["implementation_sha256"] == sha(__file__)
        and proof["admission_review_sha256"] == ADMISSION_SHA
        and proof["structure_delivery_sha256"] == STRUCTURE_DELIVERY_SHA
        and proof["records"] == 29
        and proof["positive_records"] == 7
        and proof["negative_records"] == 22
        and proof["known_parent_components"] == 19
        and proof["diagnostic_policy"] == POLICY,
        "Source contract29/provenance/diagnostic policy changed",
    )
    for item in proof["artifacts"].values():
        require(
            sha(contained(root, item["path"])) == item["sha256"],
            "Source-native binary artifact changed",
        )
    expected, _, dispositions, _ = source_frame(
        root / proof["admission_snapshot"],
        root / proof["structure_snapshot"],
        root / proof["negative_snapshot"],
    )
    require(
        records == expected
        and records == rows(root / task["table_path"])
        and read(root / proof["artifacts"]["original32_dispositions"]["path"]) == dispositions,
        "Original29 source labels, complete graphs, parent union or32 disposition ledger changed",
    )
    require(
        descriptor["original_labelled_records"] == 32
        and descriptor["included_original_records"] == 22
        and descriptor["new_affirmative_source_records"] == 7
        and descriptor["all_original_records_covered"] is False,
        "Original32 observation participation was promoted to32/32",
    )
    require(
        task["scientific_definition"] == task["task_definition"]
        and task["scientific_definition"]["diagnostic_policy"] == POLICY
        and task["scientific_warning"] == WARNING,
        "Native annotation query/report scope changed",
    )
    return proof


def prepare(output=OUTPUT, *, freeze_node=True):
    output = Path(output).resolve()
    require(not (output / "delivery_receipt.json").exists(), "Prepared delivery already exists")
    output.mkdir(parents=True, exist_ok=True)
    require(
        not any(p.name != "WORKING.md" for p in output.iterdir()),
        "Preparation namespace is not empty",
    )
    records, evidence, dispositions, _ = source_frame()
    table_path = f"task_tables/{TASK}.tsv"
    table(output / table_path, records)
    table(output / "feature_rows_index.tsv", records)
    source_snapshots(output)
    write(output / "task_tables/observation_evidence.json", evidence)
    write(output / "task_tables/original32_dispositions.json", dispositions)
    write(output / "diagnostic_policy.json", POLICY)
    cfg = training_config()
    local = split_plans(records, cfg)
    write(output / "task_tables/local_prefit_split_plan.json", local)
    plan = node_split_plans(records, cfg) if freeze_node else local
    write(output / "task_tables/node_prefit_split_plan.json", plan)
    feature_manifest = build_feature_bank(records, output)
    artifacts = dict(
        source_table=dict(path=table_path, sha256=sha(output / table_path)),
        observation_evidence=dict(
            path="task_tables/observation_evidence.json",
            sha256=sha(output / "task_tables/observation_evidence.json"),
        ),
        original32_dispositions=dict(
            path="task_tables/original32_dispositions.json",
            sha256=sha(output / "task_tables/original32_dispositions.json"),
        ),
        node_prefit_plan=dict(
            path="task_tables/node_prefit_split_plan.json",
            sha256=sha(output / "task_tables/node_prefit_split_plan.json"),
        ),
        diagnostic_policy=dict(
            path="diagnostic_policy.json", sha256=sha(output / "diagnostic_policy.json")
        ),
    )
    proof = dict(
        schema=SCHEMA,
        task_id=TASK,
        quantity_id=QUANTITY,
        implementation_sha256=sha(__file__),
        admission_review_sha256=ADMISSION_SHA,
        structure_delivery_sha256=STRUCTURE_DELIVERY_SHA,
        records=29,
        positive_records=7,
        negative_records=22,
        known_parent_components=19,
        class_parent_groups={"0": 13, "1": 7},
        diagnostic_policy=POLICY,
        artifacts=artifacts,
        admission_snapshot="task_tables/source_snapshots/admission",
        structure_snapshot="task_tables/source_snapshots/structure",
        negative_snapshot="task_tables/source_snapshots/negative",
        original32_participation=22,
        all32_covered=False,
        chemical_feature_fingerprint=feature_manifest["input_fingerprint"],
        query_mode=QUERY_MODE,
        physiological_threshold_verified=False,
        primary_sample_identity_verified=False,
        warning=WARNING,
    )
    proof_path = "task_tables/native_binary_source_proof.json"
    write(output / proof_path, proof)
    coverage = dict(
        original_endpoint="hemolysis_SOURCE_DEFINED_BINARY",
        original_labelled_records=32,
        included_original_records=22,
        new_affirmative_source_records=7,
        pooled_records=29,
        all_original_records_covered=False,
        original_negative_labels_unchanged=True,
        source_annotation_only=True,
        physiological_threshold_verified=False,
        primary_sample_identity_verified=False,
        full29_refit_required=True,
        original_record_ids=[
            r["record_id"] for r in dispositions if r["included_in_this29_source_annotation_task"]
        ],
        proof_path=proof_path,
        proof_sha256=sha(output / proof_path),
    )
    definition = dict(
        endpoint="hemolysis_SOURCE_DEFINED_BINARY",
        canonical_quantity_id=QUANTITY,
        query_mode=QUERY_MODE,
        source_native_annotation_only=True,
        physical_shared_threshold=False,
        prediction_covariates="complete_graph_only",
        context_features=False,
        clinical_inference_supported=False,
        original_negative_observation_coverage="22/32",
        separate_all_common_refit_required=True,
        molecular_equal_diagnostics_required=True,
        molecular_reporting_schema=REPORT_SCHEMA,
        implementation_module=Path(__file__).stem,
        implementation_sha256=sha(__file__),
        diagnostic_policy=POLICY,
        scientific_warning=WARNING,
    )
    task = dict(
        task_id=TASK,
        endpoint="hemolysis_SOURCE_DEFINED_BINARY",
        task_kind="classification",
        status="eligible",
        training_admitted=True,
        table_path=table_path,
        n_records=29,
        n_groups=19,
        unit="label01",
        target_transform="identity",
        inverse_transform="identity",
        requested_inner_folds=5,
        required_seeds=list(SEEDS),
        canonical_quantity_id=QUANTITY,
        validation_protocol="group_holdout_inner5_then_full29_refit_and_fixed_parameter_OOF5",
        scientific_definition=definition,
        task_definition=definition,
        source_native_binary_observation_coverage=coverage,
        scientific_warning=WARNING,
    )
    write(output / "task_manifest.json", dict(schema=SCHEMA, tasks=[task], production_model_fits=0))
    configuration = dict(
        dataset="data/dataset_reclean_20261113_v1",
        require_complete_feature_bank=False,
        resources=dict(training_host="node23", n_jobs=1, cpu_budget=1, nice=10),
        required_seeds=list(SEEDS),
        training=cfg,
        diagnostic_policy=POLICY,
        mandatory_postprocessing="separate_learning_phase_then_report_without_fit_or_model_load",
    )
    write(output / "config.json", configuration)
    write(output / "recommended_training_config.json", configuration)
    table(
        output / "task_summary.tsv",
        [dict(task_id=TASK, records=29, positive=7, negative=22, known_parent_groups=19)],
    )
    verify_source_proof(output, task, records)
    scripts = output / "scripts"
    scripts.mkdir()
    for name in script_closure():
        shutil.copy2(ROOT / "scripts" / (name + ".py"), scripts / (name + ".py"))
    stage = qa_stage_no_fit(output)
    write(output / "stage_report_no_fit_QA.json", stage)
    if freeze_node:
        write(output / "node_preflight_receipt.json", node_payload_canary(output))
    write(
        output / "deployment_requirements.json",
        dict(
            mandatory_wrapper=f"{Path(__file__).name} --run-followup --config <followup_config>",
            dynamic_CLI_entrypoints=[
                "peptide_modeling_dispatch.py",
                "peptide_modeling_supervise.py",
                "peptide_modeling_pipeline.py",
            ],
            full_script_closure=script_closure(),
            freeze_actual_node_plans_before_fit=True,
            report_only_no_fit_or_unpickle=True,
            no_new_twenty_hour_loop=True,
        ),
    )
    write(
        output / "preparation_validation.json",
        dict(
            status="prepared_not_trained" if freeze_node else "local_only_not_trainable",
            records=29,
            positive=7,
            negative=22,
            source_parent_components=19,
            actual_node_fold_plan_frozen=freeze_node,
            actual_inner_folds=5,
            full29_OOF_folds=5,
            original32_negative_records_participating=22,
            production_model_fits=0,
            local_plan=local["protocol"],
            diagnostic_learning_fits_prespecified=39,
            reporting_model_fits=0,
            reporting_model_loads=0,
        ),
    )
    files = {
        p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
        and p.name != "WORKING.md"
        and "__pycache__" not in p.parts
        and "diagnostic_history" not in p.parts
    }
    receipt = dict(
        status="prepared_not_trained" if freeze_node else "local_only_not_trainable",
        task_id=TASK,
        source_records=29,
        positive_records=7,
        negative_records=22,
        known_parent_components=19,
        source_proof_sha256=sha(output / proof_path),
        implementation_sha256=sha(__file__),
        feature_fingerprint=feature_manifest["input_fingerprint"],
        files=files,
    )
    write(output / "delivery_receipt.json", receipt)
    return receipt


def parent_weights(records):
    counts = Counter(r["split_group"] for r in records)
    value = np.asarray([1.0 / counts[r["split_group"]] for r in records], dtype=np.float64)
    return value / value.mean()


def weighted_mcc(target, probability, weights, threshold):
    target, probability, weights = map(np.asarray, (target, probability, weights))
    predicted = probability >= float(threshold)
    positive = target == 1
    tp, tn = (
        float(weights[predicted & positive].sum()),
        float(weights[~predicted & ~positive].sum()),
    )
    fp, fn = (
        float(weights[predicted & ~positive].sum()),
        float(weights[~predicted & positive].sum()),
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def probability_metrics(target, probability, weights, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, p, w = (
        np.asarray(target, dtype=np.float64),
        np.asarray(probability, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    require(
        y.shape == p.shape == w.shape
        and y.ndim == 1
        and len(y)
        and np.isfinite(p).all()
        and np.isfinite(w).all()
        and (w >= 0).all()
        and np.isin(y, [0, 1]).all()
        and ((p >= 0) & (p <= 1)).all()
        and w.sum() > 0,
        "Finite binary probabilities and nonnegative positive-mass weights required",
    )
    both = len(np.unique(y[w > 0])) == 2
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    bin_ids = np.minimum(
        (p * POLICY["calibration_bins"]).astype(int), POLICY["calibration_bins"] - 1
    )
    bins, ece = [], 0.0
    for number in range(POLICY["calibration_bins"]):
        mask = bin_ids == number
        mass = float(w[mask].sum())
        if mass <= 0:
            bins.append(
                dict(
                    bin=number,
                    records=int(mask.sum()),
                    weight=mass,
                    mean_probability=None,
                    observed_fraction=None,
                )
            )
            continue
        mean_p, mean_y = (
            float(np.average(p[mask], weights=w[mask])),
            float(np.average(y[mask], weights=w[mask])),
        )
        ece += mass / float(w.sum()) * abs(mean_p - mean_y)
        bins.append(
            dict(
                bin=number,
                records=int(mask.sum()),
                weight=mass,
                mean_probability=mean_p,
                observed_fraction=mean_y,
            )
        )
    predicted = p >= threshold
    return dict(
        n=len(y),
        positive=int((y == 1).sum()),
        negative=int((y == 0).sum()),
        two_classes_present=both,
        threshold=float(threshold),
        pr_auc=float(average_precision_score(y, p, sample_weight=w)) if both else None,
        roc_auc=float(roc_auc_score(y, p, sample_weight=w)) if both else None,
        mcc=float(weighted_mcc(y, p, w, threshold)),
        brier=float(np.average((p - y) ** 2, weights=w)),
        log_loss=float(np.average(-y * np.log(safe) - (1 - y) * np.log(1 - safe), weights=w)),
        balanced_accuracy=(
            float(np.average(predicted[y == 1], weights=w[y == 1]))
            + float(np.average(~predicted[y == 0], weights=w[y == 0]))
        )
        / 2
        if both
        else None,
        ece_fixed5_bins=float(ece),
        reliability_bins=bins,
    )


def choose_threshold(records, probability, scope="evaluation_outer_train_inner_OOF"):
    y = np.asarray([int(r["target"]) for r in records], dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    require(
        len(p) == len(records) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
        "Threshold selection needs finite source-OOF probabilities",
    )
    unique = sorted(set(map(float, p)))
    candidates = sorted(
        {
            0.0,
            0.5,
            1.0,
            *unique,
            *[(a + b) / 2 for a, b in zip(unique[:-1], unique[1:], strict=True)],
        }
    )
    weights = parent_weights(records)
    scores = [weighted_mcc(y, p, weights, t) for t in candidates]
    best_score = max(scores)
    tied = [
        t
        for t, s in zip(candidates, scores, strict=True)
        if best_score - s <= POLICY["threshold"]["tie_score_tolerance"]
    ]
    selected = min(tied, key=lambda t: (abs(t - 0.5), t))
    return dict(
        threshold=float(selected),
        objective="known_parent_equal_MCC",
        score=float(best_score),
        candidates=candidates,
        policy=POLICY["threshold"],
        source_record_ids=[r["record_id"] for r in records],
        target_float64_sha256=array_hash(y)["bytes_sha256"],
        probability_float64_sha256=array_hash(p)["bytes_sha256"],
        weight_float64_sha256=array_hash(weights)["bytes_sha256"],
        fit_scope=scope,
        labels_from_declared_fit_records_only=True,
        outer_evaluation_holdout_labels_used=False
        if scope != "deployment_full29_fixed_parameter_OOF"
        else None,
        former_holdout_now_in_deployment_training=scope == "deployment_full29_fixed_parameter_OOF",
    )


def platt_features(probability):
    p = np.asarray(probability, dtype=np.float64)
    require(
        p.ndim == 1 and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
        "Raw probabilities must be finite in[0,1]",
    )
    clipped = np.clip(p, POLICY["platt"]["clip"], 1 - POLICY["platt"]["clip"])
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt(records, probability, seed, scope):
    """Explicit learning phase, never called by report-only paths."""
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    X = platt_features(probability)
    y = np.asarray([float(r["target"]) for r in records], dtype=np.float64)
    weights = parent_weights(records)
    require(
        len(X) == len(y) and set(y) == {0.0, 1.0}, "Platt calibration needs both source classes"
    )
    settings = POLICY["platt"]
    model = LogisticRegression(
        C=settings["C"],
        solver=settings["solver"],
        tol=settings["tol"],
        max_iter=settings["max_iter"],
        random_state=seed,
    )
    with threadpool_limits(limits=1):
        model.fit(X, y, sample_weight=weights)
    require(
        model.coef_.dtype == model.intercept_.dtype == np.dtype("float64"),
        "Calibration coefficient dtype must remain float64",
    )
    artifact = dict(
        schema="fixed_Platt_logit_C1_native_source_v1",
        scope=scope,
        seed=seed,
        policy=POLICY["platt"],
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        classes=list(map(float, model.classes_)),
        fit_record_ids=[r["record_id"] for r in records],
        fit_parent_groups=sorted({r["split_group"] for r in records}),
        raw_probability_signature=array_hash(np.asarray(probability, dtype=np.float64)),
        logit_input_signature=array_hash(X),
        target_signature=array_hash(y),
        weight_signature=array_hash(weights),
        fitted_coefficient_dtype=str(model.coef_.dtype),
        labels_from_declared_fit_records_only=True,
        outer_evaluation_holdout_labels_used=False
        if scope == "evaluation_outer_train_inner_OOF"
        else None,
        former_holdout_now_in_deployment_training=scope == "deployment_full29_fixed_parameter_OOF",
        independent_performance_claim=False,
    )
    actual = model.predict_proba(X)[:, list(model.classes_).index(1)]
    require(
        np.allclose(actual, apply_platt(artifact, probability), rtol=1e-12, atol=1e-14),
        "Portable calibration JSON does not reproduce its fitted learner",
    )
    return artifact


def apply_platt(artifact, probability):
    require(
        artifact["schema"] == "fixed_Platt_logit_C1_native_source_v1"
        and artifact["policy"] == POLICY["platt"]
        and artifact["classes"] == [0.0, 1.0],
        "Calibration model or frozen policy changed",
    )
    z = platt_features(probability)[:, 0] * float(artifact["coefficient"]) + float(
        artifact["intercept"]
    )
    require(np.isfinite(z).all(), "Calibration linear predictor is nonfinite")
    output = np.empty(len(z), dtype=np.float64)
    positive = z >= 0
    output[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    output[~positive] = ez / (1 + ez)
    return output


def enriched_predictions(records, probability, folds, role, *, calibration=None):
    p = np.asarray(probability, dtype=np.float64)
    require(
        len(records) == len(p) == len(folds) and np.isfinite(p).all(),
        "Prediction membership/values differ",
    )
    calibrated = apply_platt(calibration, p) if calibration is not None else None
    return [
        dict(
            record_id=r["record_id"],
            split_group=r["split_group"],
            target=float(r["target"]),
            prediction=float(v),
            fold=int(fold),
            evaluation_role=role,
            calibrated_probability=float(calibrated[i]) if calibrated is not None else None,
        )
        for i, (r, v, fold) in enumerate(zip(records, p, folds, strict=True))
    ]


def role_metrics(source, predictions, raw_threshold, calibrated_threshold=None):
    """Pure diagnostics from saved numeric outputs, with explicit fit/evaluation roles."""
    by_id = {r["record_id"]: r for r in source}
    records = [by_id[r["record_id"]] for r in predictions]
    for row, original in zip(predictions, records, strict=True):
        require(
            row["split_group"] == original["split_group"]
            and float(row["target"]) == float(original["target"]),
            "A reported prediction escaped its source target/group",
        )
    if not records:
        return dict(records=0, parent_groups=0, raw=None, calibrated=None)
    y = [float(r["target"]) for r in records]
    raw = [float(r["prediction"]) for r in predictions]

    def calculate(values, threshold):
        return dict(
            record_weighted=probability_metrics(y, values, np.ones(len(y)), threshold),
            parent_equal=probability_metrics(y, values, parent_weights(records), threshold),
        )

    calibrated = [r["calibrated_probability"] for r in predictions]
    return dict(
        records=len(records),
        parent_groups=len({r["split_group"] for r in records}),
        raw=calculate(raw, raw_threshold),
        calibrated=calculate(calibrated, calibrated_threshold)
        if calibrated_threshold is not None
        else None,
    )


def support_diagnostics(source, predictions, plan, role):
    by_id = {r["record_id"]: r for r in source}
    folds = {f["fold"]: f for f in plan["inner_folds"]}
    output = []
    for row in predictions:
        original = by_id[row["record_id"]]
        trained = (
            plan["outer_train_record_ids"]
            if role == "outer_holdout"
            else folds[int(row["fold"])]["train_record_ids"]
        )
        require(
            original["split_group"] not in {by_id[i]["split_group"] for i in trained},
            "A validation source-parent component entered its own fit",
        )
        output.append(
            dict(
                record_id=row["record_id"],
                evaluation_role=role,
                fold=int(row["fold"]),
                unknown_material=original["assay_material_literal"] == "not_reported",
                unknown_species=original["assay_species_literal"] == "not_reported",
                material_literal_seen_in_training=original["assay_material_literal"]
                in {by_id[i]["assay_material_literal"] for i in trained},
                species_literal_seen_in_training=original["assay_species_literal"]
                in {by_id[i]["assay_species_literal"] for i in trained},
                unknown_literal_category_is_not_biological_support=True,
                uniform_physiological_threshold_or_species_effect_validated=False,
                source_context_is_prediction_covariate=False,
            )
        )
    return output


def selected_bundle(run):
    """Verify original selected artifacts and signatures without loading a model."""
    from peptide_modeling_features import load_feature_route

    run = Path(run)
    manifest = read(run / "task_manifest.json")
    require(len(manifest["tasks"]) == 1, "Native binary run must contain exactly its declared task")
    task = manifest["tasks"][0]
    source = rows(run / task["table_path"])
    proof = verify_source_proof(run, task, source)
    result_path = run / "results/full" / (TASK + "_common.json")
    result = read(result_path)
    require(
        result["status"] == result["refit_status"] == "completed"
        and result["deployment_ready"] is True
        and result["actual_inner_folds"] == result["requested_inner_folds"] == 5
        and result["search"]["backend"] == "optuna",
        "Selected task/refit/fivefold Optuna incomplete",
    )
    required = {
        "model",
        "refit_model",
        "oof",
        "holdout",
        "model_contract",
        "input_contract",
        "outer_split",
    }
    require(
        required <= set(result["artifacts"]),
        "Selected evaluation/all29 refit artifacts are missing",
    )
    for name, item in result["artifact_hashes"].items():
        require(
            sha(contained(run, name)) == item["sha256"], "An original selected artifact changed"
        )
    paths = {name: contained(run, value) for name, value in result["artifacts"].items()}
    require(paths["model"] != paths["refit_model"], "Evaluation and all29 models must be separate")
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    require(
        contract["task"] == inputs["task"] == task
        and contract["common_record_ids"] == [r["record_id"] for r in source]
        and contract["model_fit_scope"] == "outer_train_only"
        and contract["holdout_not_used_for_selection"] is True
        and contract["optional_refit_fit_scope"] == "all_common_after_evaluation"
        and inputs["config"]["refit_full"] is True
        and inputs["implementation_sha256"]
        == sha(run / "scripts/peptide_modeling_train.py")
        == TRAIN_SHA,
        "Selected/native source and distinct all29 model contracts changed",
    )
    outer = read(paths["outer_split"])
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=task,
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=inputs["config"],
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=outer["input_hash"],
    )
    actual_signature = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    require(
        actual_signature == inputs["input_hash"] == result["input_hash"] == contract["input_hash"],
        "Selected training input signature changed",
    )
    features = portable_features(run)
    require(
        features["configuration"]["implementation_sha256"] == sha(__file__)
        and features["configuration"]["diagnostic_policy_sha256"] == object_hash(POLICY),
        "Native feature/diagnostic implementation pin changed",
    )
    loaded = load_feature_route(source, features, result["selected_route"])
    recorded = inputs["feature_hashes"][result["selected_route"]]
    require(
        loaded["record_ids"] == [r["record_id"] for r in source]
        and loaded["valid"].all()
        and loaded["feature_names"] == recorded["names"] == contract["feature_names"]
        and array_hash(loaded["X"])["bytes_sha256"] == recorded["matrix_sha256"]
        and array_hash(loaded["valid"])["bytes_sha256"] == recorded["mask_sha256"],
        "Selected complete-SMILES feature matrix/schema changed",
    )
    plans = read(run / proof["artifacts"]["node_prefit_plan"]["path"])
    seed = inputs["config"]["seed"]
    plan = next(p for p in plans["seeds"] if p["seed"] == seed)
    require(
        outer == plan["outer_split"], "Actual selected outer split differs from prefit node plan"
    )
    return dict(
        run=run,
        task=task,
        source=source,
        proof=proof,
        result=result,
        result_path=result_path,
        paths=paths,
        contract=contract,
        inputs=inputs,
        features=features,
        selected_payload=loaded,
        plan=plan,
        seed=seed,
    )


def prediction_values(source, path, expected_ids, fold_map):
    by_id = {r["record_id"]: r for r in source}
    actual = rows(path)
    require(
        len(actual) == len({r["record_id"] for r in actual}) == len(expected_ids)
        and {r["record_id"] for r in actual} == set(expected_ids),
        "Prediction file membership differs",
    )
    values = {}
    for r in actual:
        original = by_id[r["record_id"]]
        p = float(r["prediction"])
        require(
            r["split_group"] == original["split_group"]
            and float(r["target"]) == float(original["target"])
            and int(r["fold"]) == fold_map[r["record_id"]]
            and math.isfinite(p)
            and 0 <= p <= 1,
            "Saved prediction source/class/fold/probability changed",
        )
        values[r["record_id"]] = p
    return np.asarray([values[i] for i in expected_ids], dtype=np.float64)


def fixed_logistic():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    c = POLICY["fixed_logistic"]
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler(with_mean=c["scaler_with_mean"])),
            (
                "model",
                LogisticRegression(
                    C=c["C"],
                    solver=c["solver"],
                    tol=c["tol"],
                    max_iter=c["max_iter"],
                    class_weight=c["class_weight"],
                ),
            ),
        ]
    )


def fit_prediction_artifact(
    source,
    X,
    train_ids,
    predicted_ids,
    estimator,
    folder,
    *,
    scope,
    fold,
    seed,
    model_policy,
    source_binding,
    require_float64=False,
):
    """One named diagnostic-learning operation, checkpointed independently of base training."""
    import joblib
    from peptide_modeling_train import _fit_model, _predict

    by_id = {r["record_id"]: i for i, r in enumerate(source)}
    train = [by_id[i] for i in train_ids]
    predicted = [by_id[i] for i in predicted_ids]
    require(
        set(train_ids).isdisjoint(predicted_ids)
        and {source[i]["split_group"] for i in train}.isdisjoint(
            {source[i]["split_group"] for i in predicted}
        ),
        "Diagnostic prediction has source-parent training leakage",
    )
    y = np.asarray([float(r["target"]) for r in source], dtype=np.float64)
    contract = dict(
        scope=scope,
        fold=fold,
        seed=seed,
        model_policy=model_policy,
        train_record_ids=train_ids,
        predict_record_ids=predicted_ids,
        train_X=array_hash(X[train]),
        predict_X=array_hash(X[predicted]),
        train_y=array_hash(y[train]),
        source_binding=source_binding,
        implementation_sha256=sha(__file__),
        diagnostic_policy_sha256=object_hash(POLICY),
        parameter_search_performed=False,
    )
    signature = object_hash(contract)
    folder = Path(folder)
    receipt_path = folder / "fit_receipt.json"
    if receipt_path.is_file():
        receipt = read(receipt_path)
        require(
            receipt["input_signature"] == signature and receipt["input_contract"] == contract,
            "Completed diagnostic operation has different declared inputs",
        )
        for name, pin in receipt["artifacts"].items():
            require(sha(contained(folder, name)) == pin, "Completed diagnostic artifact changed")
        return np.asarray(
            [float(r["prediction"]) for r in rows(folder / "predictions.tsv")]
        ), receipt
    require(
        not folder.exists(),
        "Partial diagnostic operation requires explicit recovery; do not refit blindly",
    )
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "fit_input_contract.json", contract)
    fitted = _fit_model(estimator, X[train], y[train], 1)
    learner = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    coefficient_dtype = (
        str(np.asarray(learner.coef_).dtype) if hasattr(learner, "coef_") else "not_applicable"
    )
    require(
        not require_float64 or (X.dtype == np.dtype("float64") and coefficient_dtype == "float64"),
        "Fixed Logistic inputs/coefficients must remain float64",
    )
    probability = _predict(fitted, X[predicted], "classification")
    require(
        probability.shape == (len(predicted_ids),)
        and np.isfinite(probability).all()
        and ((probability >= 0) & (probability <= 1)).all(),
        "Diagnostic learner returned invalid binary probabilities",
    )
    if require_float64:
        identical = defaultdict(list)
        for data, p in zip(X[predicted], probability, strict=True):
            identical[data.tobytes()].append(float(p))
        require(
            all(
                all(
                    math.isclose(
                        p,
                        v[0],
                        rel_tol=POLICY["fixed_logistic"]["identical_input_probability_rtol"],
                        abs_tol=POLICY["fixed_logistic"]["identical_input_probability_atol"],
                    )
                    for p in v
                )
                for v in identical.values()
            ),
            "Fixed float64 Logistic identical inputs produced inconsistent probabilities",
        )
    model_path = folder / "model.joblib"
    joblib.dump(fitted, model_path, compress=3)
    reloaded = joblib.load(model_path)
    repeated = _predict(reloaded, X[predicted], "classification")
    require(
        np.allclose(probability, repeated, rtol=1e-10, atol=1e-12),
        "Saved diagnostic learner did not reload reproducibly on training host",
    )
    table(
        folder / "predictions.tsv",
        [
            dict(
                record_id=rid,
                split_group=source[by_id[rid]]["split_group"],
                target=float(y[by_id[rid]]),
                prediction=float(p),
                fold=-1 if fold is None else fold,
            )
            for rid, p in zip(predicted_ids, probability, strict=True)
        ],
    )
    receipt = dict(
        schema="native_binary_named_diagnostic_fit_v1",
        status="completed",
        input_signature=signature,
        input_contract=contract,
        fit_input_dtype=str(X.dtype),
        predict_input_dtype=str(X.dtype),
        fitted_coefficient_dtype=coefficient_dtype,
        prediction_signature=array_hash(probability),
        model_reload_verified_on_training_host=True,
        selected_model_search_repeated=False,
        learning_phase=True,
        host=socket.gethostname(),
        artifacts={
            name: sha(folder / name)
            for name in ("model.joblib", "predictions.tsv", "fit_input_contract.json")
        },
    )
    write(receipt_path, receipt)
    return probability, receipt


def oof_learning(
    source,
    X,
    folds,
    expected_ids,
    factory,
    directory,
    *,
    scope,
    seed,
    policy,
    source_binding,
    require_float64=False,
):
    predictions, traces = {}, []
    by_id = {r["record_id"]: r for r in source}
    require(
        len(folds) == 5 and [f["fold"] for f in folds] == list(range(5)),
        "Exactly predeclared five folds required",
    )
    for fold in folds:
        probability, trace = fit_prediction_artifact(
            source,
            X,
            fold["train_record_ids"],
            fold["validation_record_ids"],
            factory(),
            Path(directory) / f"fold_{fold['fold']}",
            scope=scope,
            fold=fold["fold"],
            seed=seed,
            model_policy=policy,
            source_binding=source_binding,
            require_float64=require_float64,
        )
        traces.append(trace)
        for rid, p in zip(fold["validation_record_ids"], probability, strict=True):
            require(rid not in predictions, "An OOF record was predicted twice")
            predictions[rid] = (float(p), fold["fold"])
    require(set(predictions) == set(expected_ids), "OOF learning omitted declared source records")
    ordered = [by_id[i] for i in expected_ids]
    values = np.asarray([predictions[i][0] for i in expected_ids], dtype=np.float64)
    return ordered, values, [predictions[i][1] for i in expected_ids], traces


def calibration_artifact(path, records, probability, seed, scope):
    """Resume a completed scalar calibrator instead of fitting it again."""
    path = Path(path)
    expected = dict(
        fit_record_ids=[r["record_id"] for r in records],
        scope=scope,
        seed=seed,
        raw_probability_signature=array_hash(np.asarray(probability, dtype=np.float64)),
        target_signature=array_hash(
            np.asarray([float(r["target"]) for r in records], dtype=np.float64)
        ),
        weight_signature=array_hash(parent_weights(records)),
        policy=POLICY["platt"],
    )
    if path.is_file():
        value = read(path)
        require(
            all(value.get(k) == v for k, v in expected.items()), "Cached calibration inputs changed"
        )
        return value
    value = fit_platt(records, probability, seed, scope)
    write(path, value)
    return value


def learn_seed_diagnostics(run):
    """Explicit39-fit learning contract across3 seeds; never a report-only operation."""
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_train import _estimator

    bundle = selected_bundle(run)
    run, source, plan, seed = bundle["run"], bundle["source"], bundle["plan"], bundle["seed"]
    result = bundle["result"]
    by_id = {r["record_id"]: r for r in source}
    folder = run / "learning_diagnostics/native_binary"
    complete = folder / "learning_receipt.json"
    if complete.is_file():
        receipt = read(complete)
        require(
            receipt["source_result_sha256"] == sha(bundle["result_path"])
            and receipt["implementation_sha256"] == sha(__file__),
            "Completed diagnostic learning inputs changed",
        )
        for relative, pin in receipt["artifacts"].items():
            require(
                sha(contained(folder, relative)) == pin,
                "Completed diagnostic learning artifact changed",
            )
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    training_ids, held_ids = plan["outer_train_record_ids"], plan["outer_holdout_record_ids"]
    train_records = [by_id[i] for i in training_ids]
    held_records = [by_id[i] for i in held_ids]
    inner_folds = {i: f["fold"] for f in plan["inner_folds"] for i in f["validation_record_ids"]}
    raw_inner = prediction_values(source, bundle["paths"]["oof"], training_ids, inner_folds)
    raw_held = prediction_values(
        source, bundle["paths"]["holdout"], held_ids, {i: -1 for i in held_ids}
    )
    eval_scope = "evaluation_outer_train_inner_OOF"
    evaluation_calibration = calibration_artifact(
        folder / "evaluation_calibration.json", train_records, raw_inner, seed, eval_scope
    )
    evaluation_threshold = choose_threshold(
        train_records, apply_platt(evaluation_calibration, raw_inner), eval_scope
    )
    raw_threshold = choose_threshold(train_records, raw_inner, eval_scope)
    write(folder / "evaluation_threshold.json", evaluation_threshold)
    write(folder / "evaluation_raw_threshold.json", raw_threshold)
    table(
        folder / "selected_inner_evaluation.tsv",
        enriched_predictions(
            train_records,
            raw_inner,
            [inner_folds[i] for i in training_ids],
            "inner_oof_calibration_fit",
            calibration=evaluation_calibration,
        ),
    )
    table(
        folder / "selected_outer_holdout.tsv",
        enriched_predictions(
            held_records,
            raw_held,
            [-1] * len(held_ids),
            "outer_holdout",
            calibration=evaluation_calibration,
        ),
    )
    X = np.asarray(bundle["selected_payload"]["X"], dtype=np.float32)
    selected_policy = dict(
        model=result["selected_model"],
        params=result["selected_params"],
        route=result["selected_route"],
        new_parameter_search=False,
    )
    binding = dict(
        source_table_sha256=sha(run / bundle["task"]["table_path"]),
        source_result_sha256=sha(bundle["result_path"]),
        selected_input_hash=result["input_hash"],
        node_plan_sha256=sha(run / bundle["proof"]["artifacts"]["node_prefit_plan"]["path"]),
    )

    def factory():
        return _estimator(
            result["selected_model"],
            "classification",
            result["selected_params"],
            bundle["inputs"]["config"],
            X.shape[1],
        )

    all_ids = [r["record_id"] for r in source]
    full_records, full_raw, full_folds, full_trace = oof_learning(
        source,
        X,
        plan["full29_oof_folds"],
        all_ids,
        factory,
        folder / "full29_selected_parameter_oof",
        scope="full29_deployment_selected_parameter_OOF",
        seed=seed,
        policy=selected_policy,
        source_binding=binding,
    )
    deploy_scope = "deployment_full29_fixed_parameter_OOF"
    deployment_calibration = calibration_artifact(
        folder / "deployment_calibration.json", full_records, full_raw, seed, deploy_scope
    )
    deployment_threshold = choose_threshold(
        full_records, apply_platt(deployment_calibration, full_raw), deploy_scope
    )
    write(folder / "deployment_threshold.json", deployment_threshold)
    table(
        folder / "full29_deployment_calibration_oof.tsv",
        enriched_predictions(
            full_records,
            full_raw,
            full_folds,
            "deployment_calibration_fit_not_independent_evaluation",
            calibration=deployment_calibration,
        ),
    )
    fixed_route = POLICY["fixed_logistic"]["route"]
    fixed_payload = load_feature_route(source, bundle["features"], fixed_route)
    fixed_X = np.asarray(fixed_payload["X"], dtype=np.float64)
    fixed_records, fixed_inner, fixed_folds, fixed_trace = oof_learning(
        source,
        fixed_X,
        plan["inner_folds"],
        training_ids,
        fixed_logistic,
        folder / "fixed_Logistic_float64",
        scope="fixed_Logistic_C1_molecule_only_innerOOF",
        seed=seed,
        policy=POLICY["fixed_logistic"],
        source_binding=binding,
        require_float64=True,
    )
    fixed_held, fixed_outer_trace = fit_prediction_artifact(
        source,
        fixed_X,
        training_ids,
        held_ids,
        fixed_logistic(),
        folder / "fixed_Logistic_float64/outer_train",
        scope="fixed_Logistic_C1_outer_holdout",
        fold=None,
        seed=seed,
        model_policy=POLICY["fixed_logistic"],
        source_binding=binding,
        require_float64=True,
    )
    fixed_threshold = choose_threshold(
        fixed_records, fixed_inner, "fixed_Logistic_outer_train_inner_OOF"
    )
    write(folder / "fixed_Logistic_threshold.json", fixed_threshold)
    table(
        folder / "fixed_Logistic_inner.tsv",
        enriched_predictions(fixed_records, fixed_inner, fixed_folds, "inner_oof_fixed_control"),
    )
    table(
        folder / "fixed_Logistic_holdout.tsv",
        enriched_predictions(
            held_records, fixed_held, [-1] * len(held_ids), "outer_holdout_fixed_control"
        ),
    )
    receipt = dict(
        schema="native_binary_diagnostic_learning_receipt_v1",
        status="completed",
        seed=seed,
        implementation_sha256=sha(__file__),
        source_result_sha256=sha(bundle["result_path"]),
        source_binding=binding,
        diagnostic_policy=POLICY,
        selected_params_frozen=selected_policy,
        diagnostic_learning_fits=13,
        full29_fixed_selected_parameter_OOF_fits=5,
        fixed_Logistic_float64_fits=6,
        scalar_Platt_calibrator_fits=2,
        model_selection_repeated=False,
        report_stage_fits=0,
        report_stage_model_loads=0,
        selected_parameter_learning_trace=full_trace,
        fixed_control_learning_trace=fixed_trace + [fixed_outer_trace],
        original_selected_model_and_prediction_files_unchanged=True,
        artifacts={
            p.relative_to(folder).as_posix(): sha(p)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        },
    )
    write(complete, receipt)
    return receipt


def verify_named_fit(
    source,
    X,
    folder,
    *,
    train_ids,
    predicted_ids,
    scope,
    fold,
    seed,
    model_policy,
    source_binding,
    require_float64=False,
):
    """Read saved source/member/dtype/model/prediction receipts; never unpickle/refit."""
    folder = Path(folder)
    trace = read(folder / "fit_receipt.json")
    by_id = {r["record_id"]: i for i, r in enumerate(source)}
    train, predicted = [by_id[i] for i in train_ids], [by_id[i] for i in predicted_ids]
    y = np.asarray([float(r["target"]) for r in source], dtype=np.float64)
    expected = dict(
        scope=scope,
        fold=fold,
        seed=seed,
        model_policy=model_policy,
        train_record_ids=train_ids,
        predict_record_ids=predicted_ids,
        train_X=array_hash(X[train]),
        predict_X=array_hash(X[predicted]),
        train_y=array_hash(y[train]),
        source_binding=source_binding,
        implementation_sha256=sha(__file__),
        diagnostic_policy_sha256=object_hash(POLICY),
        parameter_search_performed=False,
    )
    require(
        trace["schema"] == "native_binary_named_diagnostic_fit_v1"
        and trace["status"] == "completed"
        and trace["input_contract"] == read(folder / "fit_input_contract.json") == expected
        and trace["input_signature"] == object_hash(expected)
        and trace["model_reload_verified_on_training_host"] is True
        and trace["selected_model_search_repeated"] is False
        and trace["learning_phase"] is True,
        "A diagnostic learning operation lost its exact source/parameter/member signature",
    )
    for name, pin in trace["artifacts"].items():
        require(sha(contained(folder, name)) == pin, "A diagnostic learned artifact changed")
    require(
        {"model.joblib", "predictions.tsv", "fit_input_contract.json"} <= set(trace["artifacts"]),
        "A saved diagnostic model/prediction/input witness is absent",
    )
    require(
        trace["fit_input_dtype"] == trace["predict_input_dtype"] == str(X.dtype),
        "A diagnostic fit/predict dtype changed",
    )
    if require_float64:
        require(
            str(X.dtype) == trace["fitted_coefficient_dtype"] == "float64",
            "Fixed Logistic float64 precision contract changed",
        )
    p = prediction_values(
        source,
        folder / "predictions.tsv",
        predicted_ids,
        {i: -1 if fold is None else fold for i in predicted_ids},
    )
    require(
        array_hash(p) == trace["prediction_signature"],
        "Diagnostic probability values differ from their fit receipt",
    )
    require(
        set(train_ids).isdisjoint(predicted_ids)
        and {source[i]["split_group"] for i in train}.isdisjoint(
            {source[i]["split_group"] for i in predicted}
        ),
        "Diagnostic model predicts a parent group included in its own fit",
    )
    if require_float64:
        identical = defaultdict(list)
        for data, value in zip(X[predicted], p, strict=True):
            identical[data.tobytes()].append(float(value))
        require(
            all(
                all(
                    math.isclose(
                        value,
                        v[0],
                        rel_tol=POLICY["fixed_logistic"]["identical_input_probability_rtol"],
                        abs_tol=POLICY["fixed_logistic"]["identical_input_probability_atol"],
                    )
                    for value in v
                )
                for v in identical.values()
            ),
            "Float64 fixed probabilities fail the predeclared identical-input guard",
        )
    return p, trace


def verify_calibration(path, records, probability, seed, scope):
    artifact = read(path)
    expected = dict(
        fit_record_ids=[r["record_id"] for r in records],
        scope=scope,
        seed=seed,
        raw_probability_signature=array_hash(np.asarray(probability, dtype=np.float64)),
        logit_input_signature=array_hash(platt_features(probability)),
        target_signature=array_hash(
            np.asarray([float(r["target"]) for r in records], dtype=np.float64)
        ),
        weight_signature=array_hash(parent_weights(records)),
        policy=POLICY["platt"],
        fitted_coefficient_dtype="float64",
        independent_performance_claim=False,
        labels_from_declared_fit_records_only=True,
        outer_evaluation_holdout_labels_used=False
        if scope == "evaluation_outer_train_inner_OOF"
        else None,
        former_holdout_now_in_deployment_training=scope == "deployment_full29_fixed_parameter_OOF",
    )
    require(
        all(artifact.get(k) == value for k, value in expected.items())
        and math.isfinite(float(artifact["coefficient"]))
        and math.isfinite(float(artifact["intercept"])),
        "Saved scalar calibration input/policy/fit scope or coefficients changed",
    )
    apply_platt(artifact, probability)
    return artifact


def read_enriched(path, source, expected_ids, folds, role, raw, calibration=None):
    actual = rows(path)
    expected = enriched_predictions(source, raw, folds, role, calibration=calibration)
    require(
        len(actual) == len(expected) == len(expected_ids),
        "Saved enriched predictions have wrong membership",
    )
    for observed, wanted in zip(actual, expected, strict=True):
        for key in ("record_id", "split_group", "evaluation_role"):
            require(
                observed[key] == wanted[key], "Saved probability/source evaluation role changed"
            )
        for key in ("target", "prediction", "fold"):
            require(
                float(observed[key]) == float(wanted[key]),
                "Saved raw probability/class/fold changed",
            )
        if wanted["calibrated_probability"] is None:
            require(
                observed["calibrated_probability"] == "",
                "A control acquired an undeclared calibration",
            )
        else:
            require(
                math.isclose(
                    float(observed["calibrated_probability"]),
                    wanted["calibrated_probability"],
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                ),
                "Saved calibrated probability differs from its JSON model",
            )
    return expected


def verify_diagnostics_and_build_report(run):
    """Verify then compute report metadata from existing artifacts only."""
    from peptide_modeling_features import load_feature_route

    bundle = selected_bundle(run)
    run, source, plan, seed = bundle["run"], bundle["source"], bundle["plan"], bundle["seed"]
    result = bundle["result"]
    folder = run / "learning_diagnostics/native_binary"
    learning = read(folder / "learning_receipt.json")
    require(
        learning["schema"] == "native_binary_diagnostic_learning_receipt_v1"
        and learning["status"] == "completed"
        and learning["seed"] == seed
        and learning["implementation_sha256"] == sha(__file__)
        and learning["source_result_sha256"] == sha(bundle["result_path"])
        and learning["diagnostic_policy"] == POLICY
        and (
            learning["diagnostic_learning_fits"],
            learning["full29_fixed_selected_parameter_OOF_fits"],
            learning["fixed_Logistic_float64_fits"],
            learning["scalar_Platt_calibrator_fits"],
        )
        == (13, 5, 6, 2)
        and learning["model_selection_repeated"] is False
        and learning["report_stage_fits"] == learning["report_stage_model_loads"] == 0,
        "Diagnostic learning roles/counts or report-only contract changed",
    )
    for relative, pin in learning["artifacts"].items():
        require(
            sha(contained(folder, relative)) == pin, "A persisted learning-phase artifact changed"
        )
    binding = dict(
        source_table_sha256=sha(run / bundle["task"]["table_path"]),
        source_result_sha256=sha(bundle["result_path"]),
        selected_input_hash=result["input_hash"],
        node_plan_sha256=sha(run / bundle["proof"]["artifacts"]["node_prefit_plan"]["path"]),
    )
    require(
        learning["source_binding"] == binding,
        "Diagnostic learning belongs to different selected/source inputs",
    )
    by_id = {r["record_id"]: r for r in source}
    training_ids, held_ids = plan["outer_train_record_ids"], plan["outer_holdout_record_ids"]
    training, held_source = [by_id[i] for i in training_ids], [by_id[i] for i in held_ids]
    fold_by_id = {i: f["fold"] for f in plan["inner_folds"] for i in f["validation_record_ids"]}
    raw_inner = prediction_values(source, bundle["paths"]["oof"], training_ids, fold_by_id)
    raw_held = prediction_values(
        source, bundle["paths"]["holdout"], held_ids, {i: -1 for i in held_ids}
    )
    scope = "evaluation_outer_train_inner_OOF"
    cal = verify_calibration(
        folder / "evaluation_calibration.json", training, raw_inner, seed, scope
    )
    raw_threshold = choose_threshold(training, raw_inner, scope)
    calibrated_threshold = choose_threshold(training, apply_platt(cal, raw_inner), scope)
    require(
        read(folder / "evaluation_threshold.json") == calibrated_threshold
        and read(folder / "evaluation_raw_threshold.json") == raw_threshold,
        "Evaluation threshold was not selected only from outer-train OOF",
    )
    inner = read_enriched(
        folder / "selected_inner_evaluation.tsv",
        training,
        training_ids,
        [fold_by_id[i] for i in training_ids],
        "inner_oof_calibration_fit",
        raw_inner,
        cal,
    )
    held = read_enriched(
        folder / "selected_outer_holdout.tsv",
        held_source,
        held_ids,
        [-1] * len(held_ids),
        "outer_holdout",
        raw_held,
        cal,
    )
    selected_policy = dict(
        model=result["selected_model"],
        params=result["selected_params"],
        route=result["selected_route"],
        new_parameter_search=False,
    )
    require(
        learning["selected_params_frozen"] == selected_policy,
        "Full29 deployment OOF parameters were reselected",
    )
    full_predictions, full_trace = {}, []
    X = np.asarray(bundle["selected_payload"]["X"], dtype=np.float32)
    for fold in plan["full29_oof_folds"]:
        probability, trace = verify_named_fit(
            source,
            X,
            folder / f"full29_selected_parameter_oof/fold_{fold['fold']}",
            train_ids=fold["train_record_ids"],
            predicted_ids=fold["validation_record_ids"],
            scope="full29_deployment_selected_parameter_OOF",
            fold=fold["fold"],
            seed=seed,
            model_policy=selected_policy,
            source_binding=binding,
        )
        full_trace.append(trace)
        full_predictions.update(
            {
                rid: (float(p), fold["fold"])
                for rid, p in zip(fold["validation_record_ids"], probability, strict=True)
            }
        )
    require(
        learning["selected_parameter_learning_trace"] == full_trace
        and set(full_predictions) == set(by_id),
        "Full29 deployment OOF trace/membership changed",
    )
    full_raw = np.asarray([full_predictions[r["record_id"]][0] for r in source], dtype=np.float64)
    deploy_scope = "deployment_full29_fixed_parameter_OOF"
    deployment_cal = verify_calibration(
        folder / "deployment_calibration.json", source, full_raw, seed, deploy_scope
    )
    deployment_threshold = choose_threshold(
        source, apply_platt(deployment_cal, full_raw), deploy_scope
    )
    require(
        read(folder / "deployment_threshold.json") == deployment_threshold,
        "Deployment threshold differs from full29 fixed-parameter OOF learning",
    )
    full = read_enriched(
        folder / "full29_deployment_calibration_oof.tsv",
        source,
        list(by_id),
        [full_predictions[i][1] for i in by_id],
        "deployment_calibration_fit_not_independent_evaluation",
        full_raw,
        deployment_cal,
    )
    fixed_payload = load_feature_route(
        source, bundle["features"], POLICY["fixed_logistic"]["route"]
    )
    fixed_X = np.asarray(fixed_payload["X"], dtype=np.float64)
    fixed_values, fixed_trace = {}, []
    for fold in plan["inner_folds"]:
        probability, trace = verify_named_fit(
            source,
            fixed_X,
            folder / f"fixed_Logistic_float64/fold_{fold['fold']}",
            train_ids=fold["train_record_ids"],
            predicted_ids=fold["validation_record_ids"],
            scope="fixed_Logistic_C1_molecule_only_innerOOF",
            fold=fold["fold"],
            seed=seed,
            model_policy=POLICY["fixed_logistic"],
            source_binding=binding,
            require_float64=True,
        )
        fixed_trace.append(trace)
        fixed_values.update(
            {
                rid: float(p)
                for rid, p in zip(fold["validation_record_ids"], probability, strict=True)
            }
        )
    fixed_inner = np.asarray([fixed_values[i] for i in training_ids], dtype=np.float64)
    fixed_held, trace = verify_named_fit(
        source,
        fixed_X,
        folder / "fixed_Logistic_float64/outer_train",
        train_ids=training_ids,
        predicted_ids=held_ids,
        scope="fixed_Logistic_C1_outer_holdout",
        fold=None,
        seed=seed,
        model_policy=POLICY["fixed_logistic"],
        source_binding=binding,
        require_float64=True,
    )
    fixed_trace.append(trace)
    require(
        learning["fixed_control_learning_trace"] == fixed_trace,
        "Fixed Logistic six-fit precision/member trace changed",
    )
    fixed_threshold = choose_threshold(
        training, fixed_inner, "fixed_Logistic_outer_train_inner_OOF"
    )
    require(
        read(folder / "fixed_Logistic_threshold.json") == fixed_threshold,
        "Fixed control threshold was selected outside outer-train OOF",
    )
    fixed_oof = read_enriched(
        folder / "fixed_Logistic_inner.tsv",
        training,
        training_ids,
        [fold_by_id[i] for i in training_ids],
        "inner_oof_fixed_control",
        fixed_inner,
    )
    fixed_test = read_enriched(
        folder / "fixed_Logistic_holdout.tsv",
        held_source,
        held_ids,
        [-1] * len(held_ids),
        "outer_holdout_fixed_control",
        fixed_held,
    )
    original_metrics = {}
    for role, role_source, probability, key in (
        ("inner_oof", training, raw_inner, "inner_cv"),
        ("outer_holdout", held_source, raw_held, "holdout"),
    ):
        value = probability_metrics(
            [float(r["target"]) for r in role_source], probability, np.ones(len(role_source)), 0.5
        )
        for metric in ("pr_auc", "roc_auc", "mcc", "brier"):
            require(
                math.isclose(value[metric], result[key][metric], rel_tol=1e-9, abs_tol=1e-10),
                "Original selected score no longer matches unchanged raw probabilities",
            )
        original_metrics[role] = value

    def strata(predictions):
        output = {}
        for field in (
            "source_subset",
            "identity_tier",
            "assay_material_literal",
            "assay_species_literal",
        ):
            output[field] = {
                value: role_metrics(
                    source,
                    [p for p in predictions if by_id[p["record_id"]][field] == value],
                    raw_threshold["threshold"],
                    calibrated_threshold["threshold"],
                )
                for value in sorted({r[field] for r in source})
            }
        return output

    report = dict(
        seed=seed,
        actual_outer_seed=plan["actual_outer_seed"],
        status="verified",
        task_id=TASK,
        quantity_id=QUANTITY,
        source_annotation_only=True,
        primary_sample_identity_verified=False,
        selected_route=result["selected_route"],
        selected_model=result["selected_model"],
        source_result_sha256=sha(bundle["result_path"]),
        selected_input_hash=result["input_hash"],
        source_table_sha256=binding["source_table_sha256"],
        node_plan_sha256=binding["node_plan_sha256"],
        original_selected_model_sha256=sha(bundle["paths"]["model"]),
        original_selected_refit_sha256=sha(bundle["paths"]["refit_model"]),
        original_oof_sha256=sha(bundle["paths"]["oof"]),
        original_holdout_sha256=sha(bundle["paths"]["holdout"]),
        base_probability_calibration_role=bundle["contract"]["probability_calibration"],
        original_selected_metrics_at_default_0_5=original_metrics,
        evaluation_calibration_model=cal,
        evaluation_threshold=calibrated_threshold,
        raw_OOF_threshold=raw_threshold,
        evaluation_by_role=dict(
            inner_oof=dict(
                metrics=role_metrics(
                    source, inner, raw_threshold["threshold"], calibrated_threshold["threshold"]
                ),
                role="model_selection_then_calibration_threshold_fit_diagnostic_not_independent",
                predictions=inner,
                source_support=support_diagnostics(source, inner, plan, "inner_oof"),
            ),
            outer_holdout=dict(
                metrics=role_metrics(
                    source, held, raw_threshold["threshold"], calibrated_threshold["threshold"]
                ),
                role="independent_outer_train_model_plus_innerOOF_calibration_evaluation",
                predictions=held,
                source_support=support_diagnostics(source, held, plan, "outer_holdout"),
                strata=strata(held),
                scores_apply_to_full29_refit=False,
            ),
        ),
        deployment_full29=dict(
            calibration_model=deployment_cal,
            threshold=deployment_threshold,
            metrics=role_metrics(source, full, 0.5, deployment_threshold["threshold"]),
            predictions=full,
            role="deployment_calibration_fit_diagnostic_not_independent_evaluation",
            selected_parameter_OOF_fits=5,
            parameter_search_repeated=False,
            all29_refit_training_record_ids=bundle["contract"]["common_record_ids"],
            original_outer_holdout_now_training_not_independent=True,
        ),
        fixed_molecule_only_Logistic=dict(
            policy=POLICY["fixed_logistic"],
            threshold=fixed_threshold,
            inner_oof=role_metrics(source, fixed_oof, fixed_threshold["threshold"]),
            outer_holdout=role_metrics(source, fixed_test, fixed_threshold["threshold"]),
            predictions=dict(inner_oof=fixed_oof, outer_holdout=fixed_test),
            fits=6,
            input_dtype="float64",
            coefficient_dtype="float64",
            context_designs_exist=False,
            context_control_omission=POLICY["context_control_omission"],
        ),
        diagnostic_learning_fits=13,
        report_stage_fits=0,
        report_stage_model_loads=0,
        warning=WARNING,
    )
    return report


def learning_archive(run, durable):
    import tarfile

    run, durable = Path(run), Path(durable)
    source = run / "learning_diagnostics/native_binary"
    archive = run / "learning_diagnostics/native_binary_learning.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz", compresslevel=1) as target:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    target.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
    destination = durable / "learning_diagnostics" / archive.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(
            sha(destination) == sha(archive), "Existing durable diagnostic-learning archive differs"
        )
    else:
        shutil.copy2(archive, destination)
    require(sha(destination) == sha(archive), "Diagnostic-learning durable archive differs")
    return dict(
        path=str(archive),
        sha256=sha(archive),
        durable_path=str(destination),
        durable_sha256=sha(destination),
    )


def render_report(config_path, journal=None):
    """No fits and no model loading. All probability/threshold evidence is already learned."""
    cfg = read(config_path)
    root, durable = Path(cfg["output_root"]), Path(cfg["durable_root"])
    journal = journal or read(root / "campaign_status.json")
    require(
        len(journal["rounds"]) == 3 and {r["seed"] for r in journal["rounds"]} == set(SEEDS),
        "Three required source-native seed rounds are mandatory",
    )
    reports, participation = [], None
    for entry in journal["rounds"]:
        require(
            entry["status"] == "completed"
            and entry["audit"]["method_model_coverage_met"] is True
            and not entry["audit"]["missing_completed_task_ids"],
            "A source-native selected seed is not complete",
        )
        run = contained(root, entry["run_dir"])
        proof_path = run / "verification.json"
        require(
            sha(proof_path) == entry["audit"]["verification_sha256"]
            and read(proof_path)["status"] == "verified"
            and read(proof_path)["source_task_manifest_sha256"] == sha(run / "task_manifest.json"),
            "The original selected seed reload/manifest verification changed",
        )
        archive = entry["native_binary_diagnostic_learning"]["archive"]
        require(
            sha(contained(run, archive["path"]))
            == archive["sha256"]
            == sha(contained(durable, archive["durable_path"]))
            == archive["durable_sha256"],
            "Saved diagnostic-learning archive or durable copy changed",
        )
        report = verify_diagnostics_and_build_report(run)
        reports.append(report)
        task = read(run / "task_manifest.json")["tasks"][0]
        descriptor = task["source_native_binary_observation_coverage"]
        if participation is None:
            participation = descriptor
        require(descriptor == participation, "Source32 participation differs between seeds")
    payload = dict(
        schema=REPORT_SCHEMA,
        status="verified",
        task_id=TASK,
        quantity_id=QUANTITY,
        query_mode=QUERY_MODE,
        required_seeds=list(SEEDS),
        records=29,
        positive=7,
        negative=22,
        known_parent_groups=19,
        class_parent_groups={"0": 13, "1": 7},
        independent_methods=10,
        model_families=list(MODELS),
        reports=reports,
        diagnostic_policy=POLICY,
        diagnostic_learning_fits=39,
        full29_fixed_selected_parameter_OOF_fits=15,
        fixed_Logistic_float64_fits=18,
        scalar_Platt_calibration_fits=6,
        report_stage_model_fits=0,
        report_stage_model_loads=0,
        original32_observation_participation=participation,
        old32_negative_values_unchanged=True,
        all_original32_records_covered=False,
        source_annotation_only=True,
        calibrated_source_annotation_probability_is_not_clinical_hemolysis_risk=True,
        source_context_covariates=False,
        common_physiological_threshold_validated=False,
        primary_sample_identity_verified=False,
        all29_refits_independently_scored=False,
        source_parent_union_is_not_homology_or_scaffold_isolation=True,
        warning=WARNING,
    )
    path = root / "dramp_native_binary_molecular_diagnostics.json"
    write(path, payload)
    durable.mkdir(parents=True, exist_ok=True)
    destination = durable / path.name
    shutil.copy2(path, destination)
    require(sha(path) == sha(destination), "Durable source-native report differs")
    receipt = dict(
        status="verified",
        path=str(path),
        sha256=sha(path),
        durable_path=str(destination),
        durable_sha256=sha(destination),
        implementation_sha256=sha(__file__),
        diagnostic_policy_sha256=object_hash(POLICY),
        diagnostic_learning_fits=39,
        report_stage_model_fits=0,
        report_stage_model_loads=0,
        source_native_binary_original_observation_participation="22/32",
        separate_full29_refits_verified=3,
    )
    journal["molecular_reporting"], journal["status"] = receipt, "completed"
    write(root / "campaign_status.json", journal)
    write(durable / "campaign_status.json", journal)
    return receipt


def execution_template(config_path):
    cfg = read(config_path)
    template = Path(cfg["template_run"])
    task = read(template / "task_manifest.json")["tasks"][0]
    source = rows(template / task["table_path"])
    proof = verify_source_proof(template, task, source)
    plan = read(template / proof["artifacts"]["node_prefit_plan"]["path"])
    require(
        plan["runtime"]["host"] == "node23"
        and plan["runtime"]["trainer_sha256"] == TRAIN_SHA
        and plan["runtime"]["production_fits"] == plan["runtime"]["model_loads"] == 0
        and read(cfg["template_config"])["training"] == training_config()
        and read(cfg["template_config"])["diagnostic_policy"] == POLICY
        and cfg.get("required_seeds", list(SEEDS)) == list(SEEDS),
        "Execution requires the complete29 actual-node prefit plan and frozen learning policy",
    )
    return cfg


def run_followup(config_path):
    from peptide_modeling_followup_seeds import run_followup as run_seeds

    cfg = execution_template(config_path)
    journal = run_seeds(config_path)
    root, durable = Path(cfg["output_root"]), Path(cfg["durable_root"])
    try:
        require(
            journal["status"] == "completed",
            "Base source-native selected training did not complete",
        )
        journal["status"], journal["stage"] = (
            "running",
            "explicit_post_selection_diagnostic_learning",
        )
        write(root / "campaign_status.json", journal)
        write(durable / "campaign_status.json", journal)
        for entry in journal["rounds"]:
            learning = learn_seed_diagnostics(entry["run_dir"])
            destination = durable / Path(entry["run_dir"]).name
            archive = learning_archive(entry["run_dir"], destination)
            entry["native_binary_diagnostic_learning"] = dict(
                status="completed",
                fit_count=13,
                receipt_path=str(
                    Path(entry["run_dir"])
                    / "learning_diagnostics/native_binary/learning_receipt.json"
                ),
                receipt_sha256=sha(
                    Path(entry["run_dir"])
                    / "learning_diagnostics/native_binary/learning_receipt.json"
                ),
                implementation_sha256=learning["implementation_sha256"],
                archive=archive,
            )
            write(root / "campaign_status.json", journal)
            write(durable / "campaign_status.json", journal)
        return render_report(config_path, journal)
    except BaseException as exc:
        journal["status"] = "requirement_incomplete"
        journal["molecular_reporting"] = dict(status="failed", reason=f"{type(exc).__name__}:{exc}")
        write(root / "campaign_status.json", journal)
        write(durable / "campaign_status.json", journal)
        raise


def predict_from_artifacts(
    result_path,
    X,
    names,
    *,
    run_dir,
    quantity_id,
    reference_mode,
    query_kind="source_native_annotation",
    use_refit=True,
):
    """Explicit model loading belongs to inference, never the report-only workflow."""
    import joblib
    from peptide_modeling_train import _predict

    require(
        quantity_id == QUANTITY
        and reference_mode == QUERY_MODE
        and query_kind == "source_native_annotation",
        "Only native source-annotation/reference queries are supported",
    )
    bundle = selected_bundle(run_dir)
    require(
        Path(result_path).resolve() == bundle["result_path"].resolve(),
        "Inference result belongs to another run",
    )
    matrix = np.asarray(X, dtype=np.float32)
    require(
        matrix.ndim == 2
        and matrix.shape[1] == len(names)
        and names == bundle["contract"]["feature_names"]
        and not np.isinf(matrix).any()
        and not np.isnan(matrix).all(axis=1).any(),
        "Inference requires the complete selected chemical schema",
    )
    folder = Path(run_dir) / "learning_diagnostics/native_binary"
    receipt = read(folder / "learning_receipt.json")
    calibration_name = "deployment_calibration.json" if use_refit else "evaluation_calibration.json"
    threshold_name = "deployment_threshold.json" if use_refit else "evaluation_threshold.json"
    for name in (calibration_name, threshold_name):
        require(
            sha(folder / name) == receipt["artifacts"][name],
            "Inference calibration/threshold artifact changed",
        )
    calibration, threshold = read(folder / calibration_name), read(folder / threshold_name)
    model = joblib.load(bundle["paths"]["refit_model" if use_refit else "model"])
    raw = _predict(model, matrix, "classification")
    calibrated = apply_platt(calibration, raw)
    return [
        dict(
            raw_source_annotation_probability=float(a),
            calibrated_source_annotation_probability=float(b),
            source_native_presence_class=int(b >= threshold["threshold"]),
            threshold=threshold["threshold"],
            mode=QUERY_MODE,
            clinical_risk_or_uniform_RBC_threshold=False,
            primary_sample_identity_verified=False,
            model_scope="all29_refit" if use_refit else "outer_train_only",
            independently_scored_on_included29=False if use_refit else None,
        )
        for a, b in zip(raw, calibrated, strict=True)
    ]


def qa_stage_no_fit(output):
    from peptide_modeling_campaign import stage_round, template_contract
    from peptide_modeling_pipeline import report_stage

    output = Path(output)
    before = {
        p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob("*")
        if p.is_file() and "diagnostic_history" not in p.parts and "__pycache__" not in p.parts
    }
    with tempfile.TemporaryDirectory(prefix="native_binary_stage_nofit_") as scratch:
        template = Path(scratch) / "template"
        shutil.copytree(
            output,
            template,
            ignore=shutil.ignore_patterns("diagnostic_history", "__pycache__", "WORKING.md"),
        )
        cfg = read(template / "config.json")
        cfg["run_dir"] = str(template)
        write(template / "template_config.json", cfg)
        contract = template_contract(
            template, template / "template_config.json", template / "representation_methods.json"
        )
        run = Path(scratch) / "round"
        stage_round(
            template,
            run,
            cfg,
            42,
            Path(scratch) / "durable",
            contract,
            template / "representation_methods.json",
        )
        require(
            report_stage(run) == dict(completed=0, deferred=0, failed=0),
            "Stage/report canary performed model work",
        )
        for name in (
            "peptide_modeling_dispatch.py",
            "peptide_modeling_supervise.py",
            "peptide_modeling_pipeline.py",
        ):
            result = subprocess.run(
                [sys.executable, "-B", str(run / "scripts" / name), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            require(
                result.returncode == 0,
                "Actual staged dynamic CLI failed: " + name + ":" + result.stderr[-1000:],
            )
    require(
        all(sha(output / name) == pin for name, pin in before.items()),
        "No-fit staging changed prepared inputs",
    )
    return dict(
        status="verified_no_fit",
        actual_stage_round=True,
        actual_report_stage=True,
        actual_dynamic_CLI_help=True,
        production_model_fits=0,
        searches=0,
        model_loads=0,
    )


def node_payload_canary(output):
    """Stage the actual complete source closure in an isolated node23 no-fit namespace."""
    import tarfile
    import uuid

    output = Path(output)
    history = output / "diagnostic_history/node_nofit"
    history.mkdir(parents=True, exist_ok=False)
    files = {
        p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
        and "diagnostic_history" not in p.parts
        and "__pycache__" not in p.parts
        and p.name != "WORKING.md"
    }
    archive = history / "prepared_payload.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
        for relative in files:
            stream.add(output / relative, arcname=relative, recursive=False)
    token = uuid.uuid4().hex[:12]
    remote = f"/tmp/peptide_dramp_native_binary_nofit_{token}"
    transferred = subprocess.run(
        ["scp", str(archive), "node23:" + remote + ".tar.gz"],
        text=True,
        capture_output=True,
        timeout=55,
    )
    require(transferred.returncode == 0, "Node no-fit payload transfer failed")
    payload = dict(
        root=remote,
        archive=remote + ".tar.gz",
        archive_sha256=sha(archive),
        files=files,
        module=Path(__file__).stem,
    )
    code = (
        "p="
        + repr(payload)
        + "\n"
        + r"""
import pathlib,hashlib,json,tarfile,os,socket,sys,builtins
root=pathlib.Path(p['root']);archive=pathlib.Path(p['archive'])
assert socket.gethostname()=='node23' and sys.flags.dont_write_bytecode
assert root.parent==pathlib.Path('/tmp') and not root.exists() and os.access('/tmp',os.W_OK)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(archive)==p['archive_sha256']
with tarfile.open(archive) as tar:
 names=tar.getnames();assert len(names)==len(set(names)) and set(names)==set(p['files'])
 assert all(m.isfile() and not pathlib.Path(m.name).is_absolute()
  and '..' not in pathlib.Path(m.name).parts for m in tar.getmembers())
 root.mkdir();tar.extractall(root)
assert all(sha(root/name)==pin for name,pin in p['files'].items())
sys.path.insert(0,str(root/'scripts'))
original_import=builtins.__import__
def without_rdkit(name,*a,**k):
 if name=='rdkit' or name.startswith('rdkit.'):
  raise ImportError('RDKit blocked in actual-node no-fit canary')
 return original_import(name,*a,**k)
builtins.__import__=without_rdkit
import importlib,numpy as np,joblib
h=importlib.import_module(p['module']);engine=importlib.import_module('peptide_modeling_train')
def forbidden(*a,**k):raise AssertionError('Fitting/search/loading forbidden')
engine._fit_model=forbidden;engine._estimator=forbidden;joblib.load=forbidden
task=h.read(root/'task_manifest.json')['tasks'][0];source=h.rows(root/task['table_path'])
proof=h.verify_source_proof(root,task,source)
actual=h.split_plans(source,h.training_config(),engine)
frozen=h.read(root/proof['artifacts']['node_prefit_plan']['path'])
assert actual['seeds']==frozen['seeds']
from peptide_modeling_features import load_feature_route
features=h.portable_features(root)
for route in features['routes']:
 loaded=load_feature_route(source,features,route)
 assert loaded['valid'].all() and len(loaded['record_ids'])==29
 json.dumps(h.array_hash(loaded['X']),allow_nan=False)
nan=np.asarray([[np.nan,1]],dtype=np.float32)
assert h.array_hash(nan)==h.array_hash(nan.copy())
json.dumps(h.array_hash(nan),allow_nan=False)
stage=h.qa_stage_no_fit(root)
assert all(sha(root/name)==pin for name,pin in p['files'].items())
assert not list(root.rglob('*.joblib')) and not list(root.rglob('*.pyc'))
print(json.dumps(dict(status='verified_no_fit',host=socket.gethostname(),root=str(root),
 payload_sha256=p['archive_sha256'],verified_payload_files=len(p['files']),
 actual_outer_and_inner5_memberships_equal=True,full29_OOF5_equal=True,
 records=29,independent_methods=10,routes=11,stage=stage,rdkit_blocked=True,
 production_fits=0,searches=0,model_loads=0,nan_typed_signature=True,
 payload_files_unchanged=True,bytecode_files=0),allow_nan=False))
"""
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "node23",
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            "OPENBLAS_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "/usr/bin/python3",
            "-B",
            "-",
        ],
        input=code,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=55,
    )
    require(
        result.returncode == 0,
        "Actual node payload/source/closure canary failed: " + result.stderr[-3000:],
    )
    receipt = json.loads(result.stdout)
    write(history / "canary_receipt.json", receipt)
    write(history / "payload_receipt.json", payload)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-followup", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--freeze-node-plans", action="store_true")
    args = parser.parse_args()
    if args.run_followup or args.report_only:
        require(args.config is not None, "A reviewed followup config is required")

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal{signum}")

        signal.signal(signal.SIGTERM, interrupted)
        result = render_report(args.config) if args.report_only else run_followup(args.config)
    elif args.freeze_node_plans:
        require(not args.output.exists(), "Use a new no-fit plan output namespace")
        args.output.mkdir(parents=True)
        records, _, _, _ = source_frame()
        table(args.output / "source_rows.tsv", records)
        result = node_split_plans(records, training_config())
        write(args.output / "node_prefit_split_plan.json", result)
        write(args.output / "diagnostic_policy.json", POLICY)
    else:
        value = prepare(args.output)
        result = {k: v for k, v in value.items() if k != "files"}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
