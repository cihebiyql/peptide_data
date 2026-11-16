"""Extend only the frozen DRAMP29 followup's prefit source-file inventory.

The original169-file delivery and b5 producer/37bd trainer remain byte-identical.
Only followup_seeds.template_contract's in-memory alias is extended; no learning
parameters, estimator, scorer, labels, folds or numerical implementations change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/peptide_modeling_dramp_hemolysis_native_binary_followup_20261115_v1"
OUTPUT = ROOT / "data/peptide_modeling_dramp_native_source_complete_runner_20261115_v1"
WRAPPER = "peptide_modeling_dramp_native_source_complete_runner.py"
PRODUCER = "peptide_modeling_dramp_hemolysis_native_binary.py"
BASE_RECEIPT = "source_base_delivery_receipt.json"
BASE_DELIVERY_SHA = "8e183998467d02a9db3ab01a9904e601669fa6234c5d595d81cdc28569c12ee2"
PRODUCER_SHA = "b5f4323f34c5e784105a83c0355177f62a291de89a65c6495855163e810c7402"
SOURCE_PROOF_SHA = "462aa44df7dd3d6bcd4d6d0b2930734b219aef990d588edd619a6dd93a2ffe9e"
TRAIN_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
TASK = "TASK_DRAMP_NATIVE_BINARY_e4dc11dc410d800c40f828e6"
SCHEMA = "DRAMP_native_binary_prefit_source_closure_v1"
PIN_SCHEMA = "DRAMP_native_binary_source_complete_runtime_pin_v1"
SEEDS = (42, 2026, 3407)
SIDE_PINS = {
    "QA_SIDECAR.json": "d2512ad8cb60186d3f51590e9c6b1c4c046aceaeee7a305f1a06bf1ba1d541a0",
    "RUNBOOK.md": "0753c02bdb24d5a6a044cdf29720160a5c0f55f9b8d20143672258c7ce4e2825",
}
CORE_NAMES = (
    "peptide_modeling_campaign.py",
    "peptide_modeling_followup_seeds.py",
    PRODUCER,
    "peptide_modeling_train.py",
    "peptide_modeling_search.py",
    "peptide_modeling_pipeline.py",
    "peptide_modeling_features.py",
    "peptide_modeling_dispatch.py",
    "peptide_modeling_supervise.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
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


def relative_name(value):
    require(
        isinstance(value, str) and value and "\\" not in value and "\0" not in value,
        "Source-proof paths must use safe relative POSIX names",
    )
    path = PurePosixPath(value)
    require(
        bool(path.parts)
        and not path.is_absolute()
        and all(p not in {"", ".", ".."} and ":" not in p for p in path.parts)
        and path.as_posix() == value,
        "Source-proof path is not safe and canonical",
    )
    return value


def contained(root, value):
    root = Path(root).resolve()
    path = root / relative_name(value)
    require(
        not path.is_symlink() and path.resolve().is_relative_to(root),
        "Source artifact escapes its template or uses a symbolic link",
    )
    return path


def static_delivery(template):
    """Verify the original immutable delivery, including all numerical source files."""
    template = Path(template).resolve()
    receipt = template / BASE_RECEIPT
    require(sha(receipt) == BASE_DELIVERY_SHA, "Original169-file source delivery receipt changed")
    data = read(receipt)
    require(
        len(data["files"]) == 169 and data["implementation_sha256"] == PRODUCER_SHA,
        "The frozen base delivery is not the reviewed b5/169 package",
    )
    for name, expected in data["files"].items():
        require(
            sha(contained(template, name)) == expected, "A frozen base artifact changed: " + name
        )
    for name, expected in SIDE_PINS.items():
        require(
            sha(contained(template, name)) == expected, "The original QA/runbook sidecar changed"
        )
    require(
        data["files"]["scripts/" + PRODUCER] == PRODUCER_SHA
        and data["files"]["scripts/peptide_modeling_train.py"] == TRAIN_SHA,
        "Frozen producer/trainer source changed",
    )
    return data


def native_proof(template):
    manifest = read(Path(template) / "task_manifest.json")
    require(
        len(manifest["tasks"]) == 1 and manifest["tasks"][0]["task_id"] == TASK,
        "Source-complete wrapper supports only the frozen native29 task",
    )
    descriptor = manifest["tasks"][0]["source_native_binary_observation_coverage"]
    name = relative_name(descriptor["proof_path"])
    require(
        sha(contained(template, name)) == descriptor["proof_sha256"] == SOURCE_PROOF_SHA,
        "Frozen native29 source-proof binding changed",
    )
    return name, read(contained(template, name))


def validate_template_config(template, template_config, registry_path):
    base = read(Path(template) / "config.json")
    actual = read(template_config)
    for key, value in base.items():
        if key != "resources":
            require(
                object_sha(actual.get(key)) == object_sha(value),
                "Runtime template changes frozen non-resource configuration: " + key,
            )
    extra = set(actual) - set(base)
    require(extra <= {"run_dir"}, "Unexpected runtime template configuration extension")
    for key, value in base["resources"].items():
        require(
            object_sha(actual["resources"].get(key)) == object_sha(value),
            "Runtime template changes frozen resource setting: " + key,
        )
    require(
        set(actual["resources"]) - set(base["resources"])
        <= {"remote_training_scratch", "remote_durable_root"},
        "Runtime resource extension is not path metadata",
    )
    require(
        Path(registry_path).resolve() == (Path(template) / "representation_methods.json").resolve(),
        "The original source representation registry is required",
    )
    return base, actual


def runtime_pin(template, template_config, registry_path):
    """Path-independent pins suitable for the new deployment's followup configuration."""
    base = static_delivery(template)
    original, _ = validate_template_config(template, template_config, registry_path)
    proof_path, _ = native_proof(template)
    wrapper_path = contained(template, "scripts/" + WRAPPER)
    require(
        sha(wrapper_path) == sha(__file__),
        "Staged source-complete wrapper differs from this implementation",
    )
    return dict(
        schema=PIN_SCHEMA,
        wrapper_module=WRAPPER,
        wrapper_sha256=sha(wrapper_path),
        base_delivery_receipt_sha256=BASE_DELIVERY_SHA,
        producer_sha256=PRODUCER_SHA,
        core_sources_sha256={name: base["files"]["scripts/" + name] for name in CORE_NAMES},
        source_proof_path=proof_path,
        source_proof_sha256=SOURCE_PROOF_SHA,
        template_config_sha256=sha(template_config),
        registry_sha256=sha(registry_path),
        base_original_config_sha256=base["files"]["config.json"],
        numerical_training_config_sha256=object_sha(original["training"]),
        diagnostic_policy_sha256=base["files"]["diagnostic_policy.json"],
        allowed_extension="prefit_source_file_inventory_only",
        alias_target="peptide_modeling_followup_seeds.template_contract",
        stock_search_objective="AP/pr_auc_original37bd_unchanged_ROC_evaluation_only",
    )


def missing_proof_artifacts(template, proof, base_files):
    """Discover omissions from the actual proof, never from a guessed filename list."""
    missing = {}
    for item in proof["artifacts"].values():
        name = relative_name(item["path"])
        require(
            sha(contained(template, name)) == item["sha256"],
            "A pre-bound native source artifact changed",
        )
        if name in base_files:
            require(
                base_files[name] == item["sha256"],
                "Original inventory disagrees with the source proof",
            )
        else:
            require(
                name not in missing or missing[name] == item["sha256"],
                "Conflicting native artifact pins",
            )
            missing[name] = item["sha256"]
    return dict(sorted(missing.items()))


def complete_source_contract(base_contract, template, template_config, registry_path):
    """Return an extended copy of the genuine campaign contract; preserve the original object."""
    pin = runtime_pin(template, template_config, registry_path)
    _, proof = native_proof(template)
    require(
        base_contract["template_config_sha256"] == pin["template_config_sha256"]
        and base_contract["registry_sha256"] == pin["registry_sha256"],
        "Original builder config/registry pin changed",
    )
    wrapper_name = "scripts/" + WRAPPER
    require(
        base_contract["files"].get(wrapper_name) == pin["wrapper_sha256"],
        "Wrapper must be staged before the real INPUT_DIRECTORIES inventory is frozen",
    )
    added = missing_proof_artifacts(template, proof, base_contract["files"])
    metadata = dict(
        schema=SCHEMA,
        wrapper_module=WRAPPER,
        wrapper_sha256=pin["wrapper_sha256"],
        base_builder_module="peptide_modeling_campaign.py",
        base_builder_sha256=pin["core_sources_sha256"]["peptide_modeling_campaign.py"],
        base_followup_module="peptide_modeling_followup_seeds.py",
        base_followup_sha256=pin["core_sources_sha256"]["peptide_modeling_followup_seeds.py"],
        producer_module=PRODUCER,
        producer_sha256=PRODUCER_SHA,
        training_source_sha256=TRAIN_SHA,
        base_delivery_receipt_sha256=BASE_DELIVERY_SHA,
        added_files=added,
        discovery="frozen_native_source_proof_artifacts",
        before_any_fit=True,
        source_proof_path=pin["source_proof_path"],
        source_proof_sha256=SOURCE_PROOF_SHA,
        base_contract_files_sha256=object_sha(base_contract["files"]),
        base_config_sha256=base_contract["template_config_sha256"],
        base_original_config_sha256=pin["base_original_config_sha256"],
        registry_sha256=pin["registry_sha256"],
        numerical_training_config_sha256=pin["numerical_training_config_sha256"],
        numerical_learning_logic_changed=False,
        original_base_files_changed=False,
        stock_search_objective=pin["stock_search_objective"],
    )
    files = {**base_contract["files"], **added}
    require(
        all(files[item["path"]] == item["sha256"] for item in proof["artifacts"].values()),
        "Extended prefit inventory still omits a native source artifact",
    )
    return {**base_contract, "files": files, "runtime_source_closure": metadata}


def load_frozen_modules(template, *, strict_paths=True):
    scripts = (Path(template) / "scripts").resolve()
    sys.path.insert(0, str(scripts))
    modules = {name: importlib.import_module(name[:-3]) for name in CORE_NAMES}
    expected = static_delivery(template)["files"]
    for name, module in modules.items():
        require(
            sha(module.__file__) == expected["scripts/" + name],
            "Imported frozen core source changed: " + name,
        )
        if strict_paths:
            require(
                Path(module.__file__).resolve() == scripts / name,
                "Imported numerical core came from outside the isolated template",
            )
    return modules


def validated_followup(config_path):
    cfg = read(config_path)
    template = Path(cfg["template_run"]).resolve()
    require(
        Path(__file__).resolve() == template / "scripts" / WRAPPER,
        "Execute the staged source-complete wrapper, not an unrelated copy",
    )
    require(
        cfg["runtime_source_closure"]
        == runtime_pin(template, cfg["template_config"], cfg["registry_path"]),
        "Followup configuration does not pin this wrapper and the frozen source closure",
    )
    require(
        cfg.get("task_ids") == [TASK]
        and cfg.get("required_seeds", list(SEEDS)) == list(SEEDS)
        and cfg.get("workers", 1) == 1,
        "Frozen task/seed/worker scope changed",
    )
    return cfg, template


def prefit_contract(cfg, template, modules):
    campaign = modules["peptide_modeling_campaign.py"]
    seeds = modules["peptide_modeling_followup_seeds.py"]
    require(
        seeds.template_contract is campaign.template_contract,
        "Followup template-contract alias was already replaced",
    )
    base = campaign.template_contract(template, cfg["template_config"], Path(cfg["registry_path"]))
    return complete_source_contract(
        base, template, cfg["template_config"], Path(cfg["registry_path"])
    )


def run(config_path):
    """The only runtime patch is the inventory alias; actual owner argv stays this wrapper."""
    cfg, template = validated_followup(config_path)
    require(socket.gethostname() == "node23", "Production followup must run on node23")
    modules = load_frozen_modules(template)
    campaign, seeds, producer = (
        modules[name]
        for name in ("peptide_modeling_campaign.py", "peptide_modeling_followup_seeds.py", PRODUCER)
    )
    producer.execution_template(config_path)
    seeds.validate_followup(cfg)
    contract = prefit_contract(cfg, template, modules)
    root, durable = Path(cfg["output_root"]).resolve(), Path(cfg["durable_root"]).resolve()
    require(
        not (root / "campaign_status.json").exists(),
        "Existing controller requires explicit recovery",
    )
    for destination in (root, durable):
        require(
            not destination.exists() or not any(destination.iterdir()),
            "Followup output namespace is not empty",
        )
        destination.mkdir(parents=True, exist_ok=True)
    identity = campaign.process_identity(os.getpid())
    require(
        identity is not None and identity["state"] != "Z",
        "Actual wrapper process identity unavailable",
    )
    runtime = dict(
        schema=SCHEMA,
        runtime_source_closure=contract["runtime_source_closure"],
        actual_wrapper_owner=identity,
        actual_wrapper_argv=identity["argv"],
        config_path=str(Path(config_path).resolve()),
        config_sha256=sha(config_path),
        closed_contract_sha256=object_sha(contract),
        written_before_producer_call=True,
        recorded_at=time.time(),
        original_entrypoint_not_impersonated=True,
    )
    runtime_path = root / "source_closure_runtime.json"
    write(runtime_path, runtime)
    write(durable / runtime_path.name, runtime)
    old_alias = seeds.template_contract

    def extended_builder(actual_template, actual_config, actual_registry):
        require(
            Path(actual_template).resolve() == template
            and Path(actual_config).resolve() == Path(cfg["template_config"]).resolve()
            and Path(actual_registry).resolve() == Path(cfg["registry_path"]).resolve(),
            "The original followup attempted a different source contract",
        )
        original = campaign.template_contract(actual_template, actual_config, actual_registry)
        closed = complete_source_contract(original, actual_template, actual_config, actual_registry)
        require(
            object_sha(closed) == object_sha(contract),
            "Prefit source closure changed after wrapper startup",
        )
        return closed

    seeds.template_contract = extended_builder
    try:
        return producer.run_followup(config_path)
    finally:
        seeds.template_contract = old_alias
        # This metadata is added only after the frozen producer's last save. While
        # learning is live, the immutable prefit sidecar and template contract carry
        # the same explanation without racing or rewriting in-memory core state.
        journal_path = root / "campaign_status.json"
        if journal_path.exists():
            journal = read(journal_path)
            require(
                journal.get("pid") == identity["pid"]
                and journal.get("process_identity", {}).get("start_ticks")
                == identity["start_ticks"],
                "Actual wrapper/controller ownership changed",
            )
            journal["runtime_source_closure"] = contract["runtime_source_closure"]
            journal["runtime_source_closure_receipt"] = dict(
                path=str(runtime_path),
                sha256=sha(runtime_path),
                durable_path=str(durable / runtime_path.name),
                durable_sha256=sha(durable / runtime_path.name),
            )
            write(journal_path, journal)
            write(durable / journal_path.name, journal)
        static_delivery(template)


def verify_round_no_fit(run_dir, seed, producer, engine, pipeline):
    """Check the native proof after the genuine stage_round, not merely generic reporting."""
    run_dir = Path(run_dir)
    task = producer.read(run_dir / "task_manifest.json")["tasks"][0]
    source = producer.rows(run_dir / task["table_path"])
    proof = producer.verify_source_proof(run_dir, task, source)
    cfg = producer.read(run_dir / "config.json")
    expected = producer.training_config()
    expected["seed"] = seed
    require(
        cfg["training"] == expected
        and producer.read(run_dir / "diagnostic_policy.json") == producer.POLICY,
        "Staging changed training parameters or omitted the exact native diagnostic policy",
    )
    from peptide_modeling_features import load_feature_route

    features = pipeline.portable_features(run_dir)
    for route in features["routes"]:
        loaded = load_feature_route(source, features, route)
        require(
            loaded["record_ids"] == [r["record_id"] for r in source] and loaded["valid"].all(),
            "Staged source rows/masks differ between chemical routes",
        )
    actual = producer.split_plans(source, cfg["training"], engine)
    frozen = producer.read(run_dir / proof["artifacts"]["node_prefit_plan"]["path"])
    # split_plans deliberately emits all three fixed seeds. Compare the actual
    # round's seed here, so all three stage_round configurations are checked.
    a = next(item for item in actual["seeds"] if item["seed"] == seed)
    b = next(item for item in frozen["seeds"] if item["seed"] == seed)
    require(a == b, "Actual staged outer/inner/full29 OOF memberships changed")
    require(
        pipeline.report_stage(run_dir) == dict(completed=0, deferred=0, failed=0),
        "Generic report-stage canary performed unexpected model work",
    )
    return dict(
        seed=seed,
        native_source_proof_verified=True,
        root_policy_sha256=sha(run_dir / "diagnostic_policy.json"),
        rows=29,
        graph_routes=len(features["routes"]),
        outer_inner5_full29OOF5_verified=True,
        numerical_training_params_unchanged=True,
        generic_report_stage_verified=True,
    )


def preflight(config_path):
    cfg, template = validated_followup(config_path)
    modules = load_frozen_modules(template)
    campaign, producer, engine, pipeline = (
        modules[name]
        for name in (
            "peptide_modeling_campaign.py",
            PRODUCER,
            "peptide_modeling_train.py",
            "peptide_modeling_pipeline.py",
        )
    )
    contract = prefit_contract(cfg, template, modules)
    original = read(cfg["template_config"])
    before = {
        p.relative_to(template).as_posix(): sha(p) for p in template.rglob("*") if p.is_file()
    }
    import builtins

    import joblib

    old_import, old_fit, old_estimator, old_load = (
        builtins.__import__,
        engine._fit_model,
        engine._estimator,
        joblib.load,
    )

    def no_rdkit(name, *args, **kwargs):
        if name == "rdkit" or name.startswith("rdkit."):
            raise ImportError("RDKit forbidden in source-complete no-fit preflight")
        return old_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fit/search/unpickle forbidden in source-complete preflight")

    builtins.__import__ = no_rdkit
    engine._fit_model, engine._estimator, joblib.load = forbidden, forbidden, forbidden
    try:
        checks = []
        with tempfile.TemporaryDirectory(prefix="native_source_complete_nofit_") as scratch:
            for seed in SEEDS:
                run_dir = Path(scratch) / f"round_{seed}"
                campaign.stage_round(
                    template,
                    run_dir,
                    original,
                    seed,
                    Path(scratch) / "durable" / str(seed),
                    contract,
                    Path(cfg["registry_path"]),
                )
                check = verify_round_no_fit(run_dir, seed, producer, engine, pipeline)
                persisted = producer.read(run_dir / "campaign_input_contract.json")[
                    "template_contract"
                ]
                require(
                    persisted == contract
                    and persisted["files"].get("diagnostic_policy.json")
                    == producer.sha(run_dir / "diagnostic_policy.json"),
                    "Prefit campaign source closure was not persisted",
                )
                for name in (
                    "peptide_modeling_dispatch.py",
                    "peptide_modeling_supervise.py",
                    "peptide_modeling_pipeline.py",
                ):
                    help_result = subprocess.run(
                        [sys.executable, "-B", str(run_dir / "scripts" / name), "--help"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=25,
                    )
                    require(
                        help_result.returncode == 0,
                        "Staged dynamic CLI failed: " + name + ":" + help_result.stderr[-1000:],
                    )
                check["dynamic_CLI_help_verified"] = True
                checks.append(check)
    finally:
        builtins.__import__ = old_import
        engine._fit_model, engine._estimator, joblib.load = old_fit, old_estimator, old_load
    require(
        all(sha(contained(template, name)) == pin for name, pin in before.items()),
        "No-fit preflight modified the source-complete template",
    )
    static_delivery(template)
    return dict(
        status="verified_source_complete_no_fit",
        schema=SCHEMA,
        host=socket.gethostname(),
        runtime_source_closure=contract["runtime_source_closure"],
        closed_contract_files=len(contract["files"]),
        closed_contract_sha256=object_sha(contract),
        checks=checks,
        production_fits=0,
        searches=0,
        model_loads=0,
        rdkit_blocked=True,
        base169_unchanged=True,
        QA_sidecars_unchanged=True,
        original_learning_logic_changed=False,
        no_production_controller_launched=True,
        actual_entrypoint_argv=list(sys.argv),
        source_complete_wrapper_sha256=sha(__file__),
    )


def main():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--runtime-pin", action="store_true")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--template-config", type=Path)
    parser.add_argument("--registry-path", type=Path)
    args = parser.parse_args()
    if args.runtime_pin:
        require(
            all((args.template, args.template_config, args.registry_path)),
            "Runtime pin requires template/config/registry",
        )
        result = runtime_pin(args.template, args.template_config, args.registry_path)
    else:
        require(args.config is not None, "Source-complete runner requires --config")
        result = preflight(args.config) if args.preflight else run(args.config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
