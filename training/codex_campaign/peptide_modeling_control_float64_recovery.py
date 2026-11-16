"""Uniform float64 recomputation of fixed diagnostic controls, never selected models.

Plan mode is read-only and performs no fitting. Apply mode explicitly performs
45 volume and54 native-MHC10 diagnostic Ridge fits on the original frozen splits.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SCHEMA = "fixed_ridge_float64_diagnostic_recovery_v1"
POLICY = "uniform_exact_float32_design_upcast_alpha1_lsqr_v1"
SEEDS = (42, 2026, 3407)
DESIGNS = ("context_only", "molecule_only", "molecule_plus_context")
PRODUCERS = {
    "volume": (
        "peptide_modeling_conditional_volume",
        "88b897bef2e3ae8ac416577147b91a1cbaf2a4c6c4c7db4ee948eb7b05e68b9c",
    ),
    "native_mhc10": (
        "peptide_modeling_mhc10_native_bridge",
        "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9",
    ),
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def object_sha(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def training_signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def contained(root, value):
    root = Path(root).resolve()
    value = Path(value)
    value = (value if value.is_absolute() else root / value).resolve()
    require(value != root and value.is_relative_to(root), "Evidence path escapes its run")
    return value


def unchanged(protected):
    require(
        all(Path(path).is_file() and sha(path) == digest for path, digest in protected.items()),
        "An immutable selected/source/old-control artifact changed",
    )


def matrix_proof(X):
    X = np.asarray(X)
    require(
        X.dtype == np.dtype("float32") and X.ndim == 2 and not np.isinf(X).any(),
        "Control designs must be exact existing float32 values",
    )
    converted = np.asarray(X, dtype=np.float64)
    require(np.array_equal(np.isnan(X), np.isnan(converted)), "Upcast changed the missing mask")
    require(
        np.array_equal(converted[np.isfinite(X)], X[np.isfinite(X)].astype(np.float64)),
        "Upcast changed a finite feature value",
    )
    return dict(
        shape=list(X.shape),
        source_float32_sha256=array_sha(X),
        source_nan_mask_sha256=array_sha(np.isnan(X)),
        upcast_float64_sha256=array_sha(converted),
        finite_values_exactly_preserved=True,
        missing_mask_preserved=True,
        fit_input_dtype="float64",
        predict_input_dtype="float64",
    )


def producer(kind):
    name, expected = PRODUCERS[kind]
    module = importlib.import_module(name)
    require(sha(module.__file__) == expected, "Frozen diagnostic producer changed")
    return module


def operation(role, fold, train, predict):
    return dict(
        fit_role=role,
        fold=fold,
        train_indices=np.asarray(train, dtype=int),
        predict_indices=np.asarray(predict, dtype=int),
    )


def native_operations(source, check):
    by_id = {r["record_id"]: i for i, r in enumerate(source)}
    return [
        operation(
            "inner_oof",
            f["fold"],
            [by_id[r] for r in f["train_record_ids"]],
            [by_id[r] for r in f["validation_record_ids"]],
        )
        for f in check["folds"]
    ] + [
        operation(
            "outer_holdout",
            None,
            [by_id[r] for r in check["train_record_ids"]],
            [by_id[r] for r in check["holdout_record_ids"]],
        )
    ]


def volume_operations(source, predictions):
    by_id = {r["record_id"]: int(r["fold"]) for r in predictions}
    require(
        set(by_id) == {r["record_id"] for r in source} and set(by_id.values()) == set(range(5)),
        "Actual volume OOF folds are incomplete",
    )
    return [
        operation(
            "inner_oof",
            fold,
            [i for i, row in enumerate(source) if by_id[row["record_id"]] != fold],
            [i for i, row in enumerate(source) if by_id[row["record_id"]] == fold],
        )
        for fold in range(5)
    ]


def selected_predictions(rows, predictions, operations):
    by_id = {r["record_id"]: r for r in rows}
    require(
        len(predictions) == len(by_id) and {r["record_id"] for r in predictions} == set(by_id),
        "Selected evaluation record membership changed",
    )
    expected = {
        rows[i]["record_id"]: (-1 if op["fit_role"] == "outer_holdout" else op["fold"])
        for op in operations
        for i in op["predict_indices"]
    }
    for row in predictions:
        source = by_id[row["record_id"]]
        require(
            source["split_group"] == row["split_group"]
            and float(source["target"]) == float(row["target"])
            and int(row["fold"]) == expected[row["record_id"]]
            and math.isfinite(float(row["prediction"])),
            "Selected source/group/target/fold changed",
        )
    for op in operations:
        require(
            not (
                {rows[i]["split_group"] for i in op["train_indices"]}
                & {rows[i]["split_group"] for i in op["predict_indices"]}
            ),
            "A source group crosses a control split",
        )


def prepare_dataset(kind, config_path):
    import peptide_modeling_train as training
    from peptide_modeling_features import load_feature_route
    from peptide_modeling_pipeline import load_rows, portable_features

    module = producer(kind)
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    root, durable = Path(cfg["output_root"]).resolve(), Path(cfg["durable_root"]).resolve()
    journal_path = root / (
        "internal_campaign_status.json" if kind == "volume" else "campaign_status.json"
    )
    journal_bytes = journal_path.read_bytes()
    journal = json.loads(journal_bytes)
    require(journal["config_sha256"] == sha(config_path), "Original controller config changed")
    require(
        journal["status"] == "completed" and journal["molecular_reporting"]["status"] == "verified",
        "Selected models and their original mandatory report must already be complete",
    )
    require(
        len(journal["rounds"]) == 3 and {r["seed"] for r in journal["rounds"]} == set(SEEDS),
        "All three original seeds are required",
    )
    receipt = journal["molecular_reporting"]
    report_path = contained(root, receipt["path"])
    report_bytes = report_path.read_bytes()
    require(
        hashlib.sha256(report_bytes).hexdigest() == receipt["sha256"],
        "Original mandatory report changed",
    )
    old_report = json.loads(report_bytes)
    require(
        "diagnostic_precision_revision" not in old_report,
        "This diagnostic revision is already present",
    )
    require(sha(receipt["durable_path"]) == receipt["sha256"], "Original durable report differs")
    schema = (
        "conditional_volume_estimand_molecular_diagnostics_v1"
        if kind == "volume"
        else "mhc10_native_species_molecular_diagnostics_v1"
    )
    require(old_report["schema"] == schema, "Original selected report schema changed")
    reports = {r["seed"]: r for r in old_report["reports"]}
    require(set(reports) == set(SEEDS), "Original report omits a required seed")
    protected = {
        str(config_path): sha(config_path),
        module.__file__: PRODUCERS[kind][1],
        training.__file__: sha(training.__file__),
    }
    seeds = []
    for entry in journal["rounds"]:
        require(entry["status"] == "completed", "An original selected seed is not complete")
        run = Path(entry["run_dir"]).resolve()
        manifest_path = run / "task_manifest.json"
        manifest = read(manifest_path)
        require(len(manifest["tasks"]) == 1, "Unexpected selected task scope")
        task = manifest["tasks"][0]
        source_path = run / task["table_path"]
        source = load_rows(source_path)
        module.validate_cohort(source, verify_chemistry=False)
        require(len(source) == (20 if kind == "volume" else 27), "Selected source size changed")
        result_path = (
            run
            / ("results/internal_cv" if kind == "volume" else "results/full")
            / (task["task_id"] + ("" if kind == "volume" else "_common") + ".json")
        )
        result = read(result_path)
        require(
            result["status"] == "completed"
            and reports[entry["seed"]]["result_sha256"] == sha(result_path),
            "Original selected result changed",
        )
        paths = {key: contained(run, value) for key, value in result["artifacts"].items()}
        for value, pin in result["artifact_hashes"].items():
            path = contained(run, value)
            require(sha(path) == pin["sha256"], "A selected artifact changed")
            protected[str(path)] = pin["sha256"]
        inputs, contract = read(paths["input_contract"]), read(paths["model_contract"])
        features = portable_features(run)
        for name, digest in features["artifact_hashes"].items():
            path = Path(features["cache_dir"]) / name
            require(sha(path) == digest, "Original source feature bank changed")
            protected[str(path)] = digest
        loaded = load_feature_route(source, features, result["selected_route"])
        require(loaded["valid"].all(), "A selected source feature row is no longer valid")
        require(
            sha(run / "scripts/peptide_modeling_train.py") == sha(training.__file__),
            "Diagnostic runtime differs from original training implementation",
        )
        if kind == "volume":
            external = task["fixed_external_covariates"]
            module.validate_contract(external)
            chemical = np.asarray(loaded["X"])
            context = module.context_matrix(source, external)
            joint = module.append_context(loaded, source, external)
            proof = inputs["features"][result["selected_route"]]
            require(
                proof["matrix_sha256"] == array_sha(joint["X"])
                and proof["valid_sha256"] == array_sha(joint["valid"])
                and proof["names"] == joint["feature_names"] == contract["feature_names"],
                "Selected volume matrix/schema changed",
            )
            signature = dict(
                protocol=inputs["protocol"],
                task=task,
                config=inputs["config"],
                rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
                features=inputs["features"],
                feature_fingerprint=features["input_fingerprint"],
                runtime=result["runtime"],
                external_context=result["external_context"],
            )
            require(
                training_signature(signature)
                == inputs["input_hash"]
                == result["input_hash"]
                == contract["input_hash"],
                "Volume fitted input signature changed",
            )
            oof = load_rows(paths["internal_oof"])
            ops = volume_operations(source, oof)
            selected_predictions(source, oof, ops)
            settings = inputs["config"]
            require(
                settings.get("target_bounds") is None and settings["n_jobs"] == 1,
                "Fixed control settings changed",
            )
            old_controls = read(paths["context_ablations"])
            old_reference = dict(
                path=str(paths["context_ablations"]),
                sha256=sha(paths["context_ablations"]),
                section_object_sha256=object_sha(old_controls),
            )
            designs = dict(
                context_only=context,
                molecule_only=chemical,
                molecule_plus_context=np.column_stack((chemical, context)),
            )
            call = dict(
                contract=external,
                settings=settings,
                splits=[(op["train_indices"], op["predict_indices"]) for op in ops],
            )
        else:
            proof = module.verify_source_proof(run, task, source)
            check_path = module.contained(run, proof["predeclared_splits"]["path"])
            check = next(r for r in read(check_path)["seeds"] if r["seed"] == entry["seed"])
            require(
                loaded["feature_names"][-2:] == module.CONTEXT_NAMES,
                "Native species context is not appended exactly once",
            )
            chemical, context = loaded["X"][:, :-2], loaded["X"][:, -2:]
            require(
                np.array_equal(
                    context, np.vstack([module.species_context(r["rbc_species"]) for r in source])
                ),
                "Original native species inputs changed",
            )
            fp = inputs["feature_hashes"][result["selected_route"]]
            require(
                fp["matrix_sha256"] == array_sha(loaded["X"])
                and fp["mask_sha256"] == array_sha(loaded["valid"])
                and fp["names"] == loaded["feature_names"] == contract["feature_names"],
                "Selected native matrix/schema changed",
            )
            split = read(paths["outer_split"])
            signature = dict(
                implementation_sha256=inputs["implementation_sha256"],
                search_implementation=inputs["search_implementation"],
                task=task,
                rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
                config=inputs["config"],
                features=inputs["feature_hashes"],
                linear_solver_policies=inputs["linear_solver_policies"],
                versions=inputs["versions"],
                outer_split=split["input_hash"],
            )
            require(
                training_signature(signature)
                == inputs["input_hash"]
                == result["input_hash"]
                == contract["input_hash"],
                "Native fitted input signature changed",
            )
            require(
                result["refit_status"] == "completed"
                and result["deployment_ready"] is True
                and contract["common_record_ids"] == [r["record_id"] for r in source]
                and contract["optional_refit_fit_scope"] == "all_common_after_evaluation"
                and "refit_model" in paths,
                "Original all27 refit is incomplete",
            )
            ops = native_operations(source, check)
            selected_predictions(source, load_rows(paths["oof"]) + load_rows(paths["holdout"]), ops)
            old_controls = reports[entry["seed"]]["fixed_ridge_controls"]
            old_reference = dict(
                path=str(report_path),
                sha256=sha(report_path),
                section_object_sha256=object_sha(old_controls),
            )
            designs = dict(
                molecule_only=np.asarray(chemical, dtype=np.float32),
                context_only=context,
                molecule_plus_context=np.column_stack((chemical, context)),
            )
            call = dict(check=check)
            protected[str(check_path)] = sha(check_path)
        require(
            set(designs) == set(DESIGNS)
            and all(np.asarray(X).dtype == np.dtype("float32") for X in designs.values()),
            "Original designs are not float32",
        )
        y = np.asarray([float(r["target"]) for r in source], dtype=np.float64)
        require(np.isfinite(y).all(), "Source transformed target is nonfinite")
        design_proofs = {
            name: dict(**matrix_proof(X), source_record_ids=[r["record_id"] for r in source])
            for name, X in designs.items()
        }
        seeds.append(
            dict(
                kind=kind,
                seed=entry["seed"],
                source=source,
                y=y,
                groups=np.asarray([r["split_group"] for r in source]),
                chemical=chemical,
                designs=designs,
                design_proofs=design_proofs,
                operations=ops,
                call=call,
                source_table_sha256=sha(source_path),
                result_sha256=sha(result_path),
                old_control_reference=old_reference,
                old_controls=old_controls,
                source_module=module,
                training_module=training,
            )
        )
        for path in (
            manifest_path,
            source_path,
            result_path,
            run / "features/manifest.json",
            run / "scripts/peptide_modeling_train.py",
        ):
            protected[str(path)] = sha(path)
    return dict(
        kind=kind,
        config_path=config_path,
        config=cfg,
        root=root,
        durable=durable,
        journal_path=journal_path,
        journal=journal,
        journal_bytes=journal_bytes,
        report_path=report_path,
        report=old_report,
        report_bytes=report_bytes,
        protected=protected,
        seeds=seeds,
    )


def prepare_plan(volume_config, native_config):
    datasets = [
        prepare_dataset("volume", volume_config),
        prepare_dataset("native_mhc10", native_config),
    ]
    require(datasets[0]["root"] != datasets[1]["root"], "The two diagnostic tasks must be distinct")
    for dataset in datasets:
        unchanged(dataset["protected"])
    return dict(
        datasets=datasets,
        diagnostic_fits_by_dataset={"volume": 45, "native_mhc10": 54},
        total_diagnostic_fits=99,
    )


def model_parameters(estimator):
    steps = getattr(estimator, "named_steps", {})
    require(set(steps) == {"impute", "scale", "model"}, "Unexpected fixed-control preprocessing")
    impute, scale, model = steps["impute"], steps["scale"], steps["model"]
    params = dict(
        class_=type(model).__name__,
        alpha=float(model.alpha),
        solver=model.solver,
        imputer_strategy=impute.strategy,
        imputer_add_indicator=impute.add_indicator,
        imputer_keep_empty_features=impute.keep_empty_features,
        scaler_with_mean=scale.with_mean,
    )
    params["class"] = params.pop("class_")
    require(
        params
        == dict(
            **{"class": "Ridge"},
            alpha=1.0,
            solver="lsqr",
            imputer_strategy="median",
            imputer_add_indicator=True,
            imputer_keep_empty_features=True,
            scaler_with_mean=False,
        ),
        "Fixed Ridge model settings changed",
    )
    return params


class FitObserver:
    def __init__(self, seed_plan):
        self.plan = seed_plan
        self.training = seed_plan["training_module"]
        self.metadata = {}
        self.current = None
        self.estimator_calls = []
        self.progress = seed_plan.get("_fit_progress", {"attempted": 0, "completed": 0})

    def design(self, name, X):
        require(
            name in DESIGNS and name not in self.metadata,
            "A diagnostic design was repeated or changed",
        )
        require(
            matrix_proof(X) == matrix_proof(self.plan["designs"][name]),
            "Producer changed the predeclared float32 design",
        )
        self.current = name
        X64 = np.asarray(X, dtype=np.float64)
        self.metadata[name] = dict(
            **self.plan["design_proofs"][name], fits=[], original_guard_checks=[]
        )
        return X64

    @contextmanager
    def instrument(self):
        training = self.training
        original_estimator, original_fit = training._estimator, training._fit_model
        observer = self

        def estimator(model, kind, params, settings, feature_count=0):
            require(
                model == "linear"
                and kind == "regression"
                and params == {"regularization": 1.0}
                and settings.get("n_jobs") == 1,
                "A diagnostic estimator was selected or reparameterized",
            )
            fitted = original_estimator(model, kind, params, settings, feature_count)
            model_parameters(fitted)
            observer.estimator_calls.append(observer.current)
            return fitted

        def fit(estimator_value, X, y, n_jobs):
            name = observer.current
            require(name in observer.metadata, "A fit ran before its float64 design declaration")
            fits = observer.metadata[name]["fits"]
            require(
                len(fits) < len(observer.plan["operations"]),
                "Extra diagnostic fitting is forbidden",
            )
            op = observer.plan["operations"][len(fits)]
            design64 = np.asarray(observer.plan["designs"][name], dtype=np.float64)
            require(
                np.asarray(X).dtype == np.dtype("float64")
                and np.asarray(y).dtype == np.dtype("float64")
                and n_jobs == 1,
                "Every diagnostic fit input must be float64 and one-thread",
            )
            require(
                array_sha(X) == array_sha(design64[op["train_indices"]])
                and array_sha(y) == array_sha(observer.plan["y"][op["train_indices"]]),
                "Diagnostic fitting used different source rows or targets",
            )
            params = model_parameters(estimator_value)
            observer.progress["attempted"] += 1
            fitted = original_fit(estimator_value, X, y, n_jobs)
            observer.progress["completed"] += 1
            require(
                np.asarray(fitted.named_steps["model"].coef_).dtype == np.dtype("float64"),
                "Fixed-control coefficients are not float64",
            )
            trace = dict(
                fit_index=len(fits),
                fit_role=op["fit_role"],
                fold=op["fold"],
                train_record_ids=[
                    observer.plan["source"][i]["record_id"] for i in op["train_indices"]
                ],
                predict_record_ids=[
                    observer.plan["source"][i]["record_id"] for i in op["predict_indices"]
                ],
                train_X_float64_sha256=array_sha(X),
                train_y_float64_sha256=array_sha(y),
                fit_input_dtype="float64",
                predict_input_dtype="float64",
                fitted_coefficient_dtype="float64",
                model=params,
            )
            fits.append(trace)

            class CheckedPrediction:
                def predict(self, values):
                    require(
                        "prediction_float64_sha256" not in trace, "A diagnostic fit predicted twice"
                    )
                    require(
                        np.asarray(values).dtype == np.dtype("float64")
                        and array_sha(values) == array_sha(design64[op["predict_indices"]]),
                        "Diagnostic prediction input changed or was downcast",
                    )
                    with threadpool_limits(limits=1):
                        p = np.asarray(fitted.predict(values))
                    require(
                        p.dtype == np.dtype("float64") and np.isfinite(p).all(),
                        "A float64 diagnostic prediction is nonfinite or downcast",
                    )
                    trace.update(
                        predict_X_float64_sha256=array_sha(values),
                        prediction_float64_sha256=array_sha(p),
                    )
                    return p

            return CheckedPrediction()

        training._estimator, training._fit_model = estimator, fit
        try:
            yield
        finally:
            training._estimator, training._fit_model = original_estimator, original_fit


def compile_float64_control(kind, observer):
    module = observer.plan["source_module"]
    function = module.contextual_diagnostics if kind == "volume" else module.fixed_ridge_controls
    tree = ast.parse(inspect.getsource(function))
    changed = copy.deepcopy(tree)
    loops = [
        node
        for node in ast.walk(changed)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [getattr(x, "id", None) for x in node.target.elts] == ["name", "X"]
    ]
    require(len(loops) == 1, "Expected exactly one original control design loop")
    statement = ast.parse("X = _float64_design(name, X)").body[0]
    loops[0].body.insert(0, statement)
    reverse = copy.deepcopy(changed)
    reverse_loop = next(
        node
        for node in ast.walk(reverse)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [getattr(x, "id", None) for x in node.target.elts] == ["name", "X"]
    )
    reverse_loop.body.pop(0)
    require(
        ast.dump(reverse, include_attributes=False) == ast.dump(tree, include_attributes=False),
        "Control implementation changed outside the dtype insertion",
    )
    ast.fix_missing_locations(changed)
    namespace = dict(module.__dict__)
    namespace["_float64_design"] = observer.design
    exec(compile(changed, "<private_float64_fixed_controls>", "exec"), namespace)

    def ast_sha(value):
        return hashlib.sha256(
            ast.dump(value.body[0], include_attributes=False).encode()
        ).hexdigest()

    provenance = dict(
        original_function_ast_sha256=ast_sha(tree),
        upcast_function_ast_sha256=ast_sha(changed),
        inserted_design_casts=1,
        outside_cast_ast_identical=True,
        ast_hash_policy="sha256_entire_FunctionDef_ast_dump_without_attributes_utf8",
    )
    return namespace[function.__name__], provenance


def strict_guard_checks(seed_plan, controls, metadata):
    for name in DESIGNS:
        X = np.asarray(seed_plan["designs"][name], dtype=np.float64)
        if seed_plan["kind"] == "volume":
            p = controls["ablations"][name]["predictions"]
            lookup = {
                row["record_id"]: float(value)
                for row, value in zip(seed_plan["source"], p, strict=True)
            }
        else:
            lookup = {
                row["record_id"]: float(row["prediction"])
                for report in controls[name]["evaluation_by_role"].values()
                for row in report["predictions"]
            }
        require(
            set(lookup) == {r["record_id"] for r in seed_plan["source"]}
            and all(math.isfinite(v) for v in lookup.values()),
            "New controls omit records or contain nonfinite predictions",
        )
        for op in seed_plan["operations"]:
            grouped = {}
            for index in op["predict_indices"]:
                grouped.setdefault(X[index].tobytes(), []).append(index)
            for row_bytes, indices in grouped.items():
                if len(indices) < 2:
                    continue
                ids = [seed_plan["source"][i]["record_id"] for i in indices]
                values = [lookup[rid] for rid in ids]
                require(
                    np.allclose(values, values[0], rtol=1e-7, atol=1e-8),
                    "Float64 controls still fail the original strict identical-input guard",
                )
                metadata[name]["original_guard_checks"].append(
                    dict(
                        evaluation_role=op["fit_role"],
                        fold=-1 if op["fold"] is None else op["fold"],
                        source_record_ids=ids,
                        input_row_float64_sha256=hashlib.sha256(row_bytes).hexdigest(),
                        prediction_values=values,
                        observed_absolute_spread=max(values) - min(values),
                        original_allclose_pass=True,
                    )
                )
        for op, fit in zip(seed_plan["operations"], metadata[name]["fits"], strict=True):
            p = np.asarray(
                [lookup[seed_plan["source"][i]["record_id"]] for i in op["predict_indices"]],
                dtype=np.float64,
            )
            require(
                fit["prediction_float64_sha256"] == array_sha(p),
                "Published controls differ from the traced predictions",
            )


def run_seed_controls(seed_plan):
    observer = FitObserver(seed_plan)
    function, ast_provenance = compile_float64_control(seed_plan["kind"], observer)
    original_design_bytes = {name: X.tobytes() for name, X in seed_plan["designs"].items()}
    with observer.instrument(), threadpool_limits(limits=1):
        if seed_plan["kind"] == "volume":
            controls = function(
                seed_plan["source"],
                seed_plan["call"]["contract"],
                seed_plan["chemical"],
                seed_plan["y"],
                seed_plan["groups"],
                seed_plan["call"]["splits"],
                seed_plan["call"]["settings"],
                seed_plan["training_module"],
            )
        else:
            controls = function(
                seed_plan["source"], seed_plan["chemical"], seed_plan["call"]["check"]
            )
    require(
        set(observer.metadata) == set(DESIGNS)
        and len(observer.estimator_calls) == 3
        and set(observer.estimator_calls) == set(DESIGNS),
        "Not all three fixed control designs ran exactly once",
    )
    for name, metadata in observer.metadata.items():
        require(
            len(metadata["fits"]) == len(seed_plan["operations"])
            and all("prediction_float64_sha256" in fit for fit in metadata["fits"]),
            "Diagnostic fit/predict count is incomplete",
        )
        require(
            seed_plan["designs"][name].tobytes() == original_design_bytes[name],
            "Original float32 design values changed",
        )
    strict_guard_checks(seed_plan, controls, observer.metadata)
    return controls, dict(
        schema="fixed_ridge_float64_control_seed_evidence_v1",
        dataset_kind=seed_plan["kind"],
        seed=seed_plan["seed"],
        source_table_sha256=seed_plan["source_table_sha256"],
        result_sha256=seed_plan["result_sha256"],
        old_control_reference=seed_plan["old_control_reference"],
        designs=observer.metadata,
        ast_provenance=ast_provenance,
        diagnostic_fits=3 * len(seed_plan["operations"]),
        selected_model_fits=0,
        parameter_searches=0,
        original_strict_guard_passed=True,
    )


def plan_summary(plan):
    return dict(
        schema=SCHEMA,
        policy=POLICY,
        status="verified_plan_no_fits",
        diagnostic_fits_by_dataset=plan["diagnostic_fits_by_dataset"],
        total_diagnostic_fits=99,
        diagnostic_fits_performed=0,
        selected_model_fits=0,
        parameter_searches=0,
        datasets={
            d["kind"]: dict(
                output_root=str(d["root"]),
                config_path=str(d["config_path"]),
                config_sha256=sha(d["config_path"]),
                old_report_sha256=hashlib.sha256(d["report_bytes"]).hexdigest(),
                old_journal_sha256=hashlib.sha256(d["journal_bytes"]).hexdigest(),
                protected_files=len(d["protected"]),
                seeds=[
                    dict(
                        seed=s["seed"],
                        source_table_sha256=s["source_table_sha256"],
                        result_sha256=s["result_sha256"],
                        designs=s["design_proofs"],
                        planned_fits=3 * len(s["operations"]),
                    )
                    for s in d["seeds"]
                ],
            )
            for d in plan["datasets"]
        },
    )


def runtime_guard():
    require(
        os.name == "posix" and socket.gethostname() == "node23",
        "Diagnostic apply requires its existing node23 runtime",
    )


def active_run_owners(root, process_identity, proc_root=Path("/proc")):
    root = str(Path(root).resolve())
    owners = []
    if proc_root.is_dir():
        for directory in proc_root.iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            try:
                actual = process_identity(int(directory.name))
            except PermissionError:
                continue
            if (
                actual
                and actual["state"] != "Z"
                and any(
                    value == root or value.startswith(root + "/")
                    for value in actual.get("argv", [])
                )
            ):
                owners.append(dict(pid=actual["pid"], start_ticks=actual["start_ticks"]))
    return owners


def apply_recovery(volume_config, native_config):
    from peptide_modeling_campaign import process_identity
    from peptide_modeling_supervise import atomic_json

    runtime_guard()
    plan = prepare_plan(volume_config, native_config)
    for dataset in plan["datasets"]:
        actual = process_identity(dataset["journal"]["pid"])
        old = dataset["journal"]["process_identity"]
        require(
            actual is None or actual["state"] == "Z" or actual["start_ticks"] != old["start_ticks"],
            "An original selected-model controller is still live",
        )
        require(
            not active_run_owners(dataset["root"], process_identity),
            "Another process still owns the selected run namespace",
        )
    recovery_id = f"{time.time_ns()}_{uuid.uuid4().hex[:10]}"
    owner = process_identity(os.getpid())
    require(owner is not None and owner["argv"], "Current diagnostic owner identity is unavailable")
    implementation = sha(__file__)
    for dataset in plan["datasets"]:
        require(
            dataset["journal_path"].read_bytes() == dataset["journal_bytes"]
            and dataset["report_path"].read_bytes() == dataset["report_bytes"],
            "Original report/journal changed before backup",
        )
        recovery = dataset["root"] / "report_recovery" / ("float64_controls_" + recovery_id)
        recovery.mkdir(parents=True, exist_ok=False)
        dataset["recovery"] = recovery
        (recovery / "old_molecular_report.json").write_bytes(dataset["report_bytes"])
        (recovery / "old_campaign_status.json").write_bytes(dataset["journal_bytes"])
        if dataset["kind"] == "native_mhc10":
            for seed in dataset["seeds"]:
                seed["old_control_reference"]["path"] = str(recovery / "old_molecular_report.json")
        journal = copy.deepcopy(dataset["journal"])
        journal.update(
            status="reporting_float64_diagnostic_recovery", pid=os.getpid(), process_identity=owner
        )
        journal["molecular_reporting"] = dict(
            status="diagnostic_recovery_running", recovery_id=recovery_id
        )
        journal.setdefault("recovery_history", []).append(
            dict(
                kind=SCHEMA,
                recovery_id=recovery_id,
                old_journal_sha256=hashlib.sha256(dataset["journal_bytes"]).hexdigest(),
                backup=str(recovery / "old_campaign_status.json"),
                old_report_sha256=hashlib.sha256(dataset["report_bytes"]).hexdigest(),
                old_report_backup=str(recovery / "old_molecular_report.json"),
                tool_sha256=implementation,
                new_owner=owner,
                diagnostic_fits_planned=plan["diagnostic_fits_by_dataset"][dataset["kind"]],
                selected_model_fits=0,
            )
        )
        dataset["active_journal"] = journal
    pre_fit_plan = dict(
        schema="fixed_ridge_float64_pre_fit_plan_v1",
        recovery_id=recovery_id,
        owner=owner,
        implementation_sha256=implementation,
        plan=plan_summary(plan),
        protected_artifact_hashes={d["kind"]: d["protected"] for d in plan["datasets"]},
        selected_model_fits=0,
        diagnostic_fits_performed=0,
    )
    for dataset in plan["datasets"]:
        atomic_json(dataset["recovery"] / "pre_fit_plan.json", pre_fit_plan)
    total = 0
    progress = {"attempted": 0, "completed": 0}
    try:
        for dataset in plan["datasets"]:
            require(
                dataset["journal_path"].read_bytes() == dataset["journal_bytes"]
                and dataset["report_path"].read_bytes() == dataset["report_bytes"],
                "Original report/journal changed before diagnostic ownership adoption",
            )
            unchanged(dataset["protected"])
            atomic_json(dataset["journal_path"], dataset["active_journal"])
            atomic_json(
                dataset["durable"] / dataset["journal_path"].name, dataset["active_journal"]
            )
            dataset["active_journal_sha256"] = sha(dataset["journal_path"])
        for dataset in plan["datasets"]:
            dataset["sidecars"] = []
            for seed in dataset["seeds"]:
                require(sha(__file__) == implementation, "Diagnostic implementation changed")
                seed["_fit_progress"] = progress
                controls, evidence = run_seed_controls(seed)
                controls_path = dataset["recovery"] / f"seed_{seed['seed']}_controls.json"
                evidence_path = dataset["recovery"] / f"seed_{seed['seed']}_evidence.json"
                atomic_json(controls_path, controls)
                evidence.update(
                    controls_path=str(controls_path),
                    controls_sha256=sha(controls_path),
                    controls_object_sha256=object_sha(controls),
                )
                atomic_json(evidence_path, evidence)
                total += evidence["diagnostic_fits"]
                dataset["sidecars"].append(
                    dict(
                        seed=seed["seed"],
                        controls_path=str(controls_path),
                        controls_sha256=sha(controls_path),
                        evidence_path=str(evidence_path),
                        evidence_sha256=sha(evidence_path),
                    )
                )
            unchanged(dataset["protected"])
        require(
            total == progress["attempted"] == progress["completed"] == 99,
            "The uniform diagnostic fit plan was not completed exactly99times",
        )
        joint = dict(
            schema="fixed_ridge_float64_joint_completion_v1",
            status="all99_diagnostic_fits_verified",
            recovery_id=recovery_id,
            policy=POLICY,
            implementation_sha256=implementation,
            owner=owner,
            required_seeds=list(SEEDS),
            designs=list(DESIGNS),
            datasets={
                d["kind"]: dict(
                    output_root=str(d["root"]),
                    config_path=str(d["config_path"]),
                    config_sha256=sha(d["config_path"]),
                    old_report_sha256=hashlib.sha256(d["report_bytes"]).hexdigest(),
                    old_journal_sha256=hashlib.sha256(d["journal_bytes"]).hexdigest(),
                    producer_sha256=PRODUCERS[d["kind"]][1],
                    training_source_sha256=sha(d["seeds"][0]["training_module"].__file__),
                    diagnostic_fits=plan["diagnostic_fits_by_dataset"][d["kind"]],
                    seeds=d["sidecars"],
                    pre_fit_plan=dict(
                        path=str(d["recovery"] / "pre_fit_plan.json"),
                        sha256=sha(d["recovery"] / "pre_fit_plan.json"),
                    ),
                )
                for d in plan["datasets"]
            },
            diagnostic_fits_by_dataset=plan["diagnostic_fits_by_dataset"],
            total_diagnostic_fits=99,
            selected_model_fits=0,
            parameter_searches=0,
            old_selected_sources_models_predictions_unchanged=True,
            old_controls_preserved=True,
            original_strict_guards_passed=True,
        )
        for dataset in plan["datasets"]:
            atomic_json(dataset["recovery"] / "joint_completion.json", joint)
        for dataset in plan["datasets"]:
            recovery = dataset["recovery"]
            first_evidence = read(dataset["sidecars"][0]["evidence_path"])
            receipt = dict(
                schema=SCHEMA,
                status="verified_diagnostic_controls_not_adopted",
                recovery_id=recovery_id,
                dataset_kind=dataset["kind"],
                policy=POLICY,
                implementation_sha256=implementation,
                source_implementations=dict(
                    producer=PRODUCERS[dataset["kind"]][1],
                    training=sha(dataset["seeds"][0]["training_module"].__file__),
                ),
                original_owner=dataset["journal"]["process_identity"],
                new_owner=owner,
                old_report=dict(
                    path=str(dataset["report_path"]),
                    sha256=hashlib.sha256(dataset["report_bytes"]).hexdigest(),
                    backup_path=str(recovery / "old_molecular_report.json"),
                    content_sha256=object_sha(dataset["report"]),
                ),
                old_journal=dict(
                    path=str(dataset["journal_path"]),
                    sha256=hashlib.sha256(dataset["journal_bytes"]).hexdigest(),
                    backup_path=str(recovery / "old_campaign_status.json"),
                ),
                protected_artifact_hashes=dataset["protected"],
                pre_fit_plan=dict(
                    path=str(recovery / "pre_fit_plan.json"),
                    sha256=sha(recovery / "pre_fit_plan.json"),
                ),
                seeds=dataset["sidecars"],
                ast_provenance=first_evidence["ast_provenance"],
                joint_completion=dict(
                    path=str(recovery / "joint_completion.json"),
                    sha256=sha(recovery / "joint_completion.json"),
                ),
                diagnostic_fits=plan["diagnostic_fits_by_dataset"][dataset["kind"]],
                total_diagnostic_fits=99,
                selected_model_fits=0,
                parameter_searches=0,
                old_controls_preserved=True,
                raw_selected_predictions_unchanged=True,
            )
            receipt_path = recovery / "recovery_receipt.json"
            atomic_json(receipt_path, receipt)
            revision = dict(
                schema=SCHEMA,
                policy=POLICY,
                recovery_id=recovery_id,
                dataset_kind=dataset["kind"],
                required_seeds=list(SEEDS),
                designs=list(DESIGNS),
                diagnostic_fits=receipt["diagnostic_fits"],
                total_recovery_diagnostic_fits=99,
                diagnostic_fits_by_dataset=plan["diagnostic_fits_by_dataset"],
                selected_model_fits=0,
                parameter_searches=0,
                precision="float64",
                all_designs_and_seeds_recomputed=True,
                old_controls_preserved=True,
                old_float32_controls_qualified_by_this_revision=False,
                old_selected_predictions_unchanged=True,
                implementation_sha256=implementation,
                selected_report_implementation_sha256=dataset["journal"]["molecular_reporting"][
                    "implementation_sha256"
                ],
                evidence_path=str(receipt_path),
                evidence_sha256=sha(receipt_path),
                old_report_backup=str(recovery / "old_molecular_report.json"),
                old_report_sha256=receipt["old_report"]["sha256"],
                old_journal_backup=str(recovery / "old_campaign_status.json"),
                old_journal_sha256=receipt["old_journal"]["sha256"],
            )
            new_report = copy.deepcopy(dataset["report"])
            by_seed = {r["seed"]: r for r in new_report["reports"]}
            for sidecar in dataset["sidecars"]:
                by_seed[sidecar["seed"]]["float64_fixed_ridge_controls"] = read(
                    sidecar["controls_path"]
                )
                by_seed[sidecar["seed"]]["float64_control_evidence"] = dict(
                    path=sidecar["evidence_path"], sha256=sidecar["evidence_sha256"]
                )
            require(
                strip_new_sections(new_report) == dataset["report"],
                "A selected report field changed during diagnostic appending",
            )
            new_report["diagnostic_precision_revision"] = revision
            dataset["new_report"], dataset["revision"] = new_report, revision
            durable_recovery = dataset["durable"] / "report_recovery" / recovery.name
            shutil.copytree(recovery, durable_recovery)
            require(
                all(
                    sha(p) == sha(durable_recovery / p.relative_to(recovery))
                    for p in recovery.rglob("*")
                    if p.is_file()
                ),
                "Durable float64 evidence differs",
            )
        for dataset in plan["datasets"]:
            unchanged(dataset["protected"])
            require(
                sha(dataset["journal_path"]) == dataset["active_journal_sha256"]
                and dataset["report_path"].read_bytes() == dataset["report_bytes"],
                "Report/journal changed before diagnostic adoption",
            )
        for dataset in plan["datasets"]:
            atomic_json(dataset["report_path"], dataset["new_report"])
            old_receipt = dataset["journal"]["molecular_reporting"]
            durable_report = Path(old_receipt["durable_path"])
            shutil.copy2(dataset["report_path"], durable_report)
            require(
                sha(durable_report) == sha(dataset["report_path"]),
                "Durable new diagnostic report differs",
            )
            journal = dataset["active_journal"]
            journal["recovery_history"][-1]["diagnostic_fits_completed"] = plan[
                "diagnostic_fits_by_dataset"
            ][dataset["kind"]]
            journal["recovery_history"][-1]["total_recovery_diagnostic_fits"] = 99
            journal["molecular_reporting"] = {
                **old_receipt,
                "sha256": sha(dataset["report_path"]),
                "durable_sha256": sha(durable_report),
                "implementation_sha256": implementation,
                "selected_report_implementation_sha256": old_receipt["implementation_sha256"],
                "diagnostic_precision_revision": dataset["revision"],
            }
            journal["status"] = "completed"
            atomic_json(dataset["journal_path"], journal)
            atomic_json(dataset["durable"] / dataset["journal_path"].name, journal)
        for dataset in plan["datasets"]:
            unchanged(dataset["protected"])
        return dict(
            status="completed_float64_diagnostic_recovery",
            recovery_id=recovery_id,
            diagnostic_fits=99,
            selected_model_fits=0,
            parameter_searches=0,
            reports={
                d["kind"]: dict(
                    path=str(d["report_path"]), sha256=sha(d["report_path"]), revision=d["revision"]
                )
                for d in plan["datasets"]
            },
        )
    except BaseException as exc:
        for dataset in plan["datasets"]:
            current = read(dataset["journal_path"])
            if (
                current.get("pid") == os.getpid()
                and current.get("recovery_history", [{}])[-1].get("recovery_id") == recovery_id
            ):
                current["status"] = "requirement_incomplete"
                current["molecular_reporting"] = dict(
                    status="diagnostic_recovery_failed",
                    recovery_id=recovery_id,
                    reason=f"{type(exc).__name__}:{exc}",
                    diagnostic_fits_completed=progress["completed"],
                    diagnostic_fits_attempted=progress["attempted"],
                    selected_model_fits=0,
                    old_report_backup=str(dataset["recovery"] / "old_molecular_report.json"),
                )
                atomic_json(dataset["journal_path"], current)
                atomic_json(dataset["durable"] / dataset["journal_path"].name, current)
        raise


def strip_new_sections(report):
    result = copy.deepcopy(report)
    result.pop("diagnostic_precision_revision", None)
    for item in result["reports"]:
        item.pop("float64_fixed_ridge_controls", None)
        item.pop("float64_control_evidence", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-config", type=Path, required=True)
    parser.add_argument("--native-config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-only", "--verify-only", dest="plan_only", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan_summary(prepare_plan(args.volume_config, args.native_config))))
    else:
        if os.name == "posix" and os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
        print(json.dumps(apply_recovery(args.volume_config, args.native_config)))


if __name__ == "__main__":
    main()
