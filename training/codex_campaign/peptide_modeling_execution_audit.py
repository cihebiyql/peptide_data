"""Read-only execution audit with frozen controller discovery and terminal evidence.

Freeze once: --freeze --output-dir data/peptide_modeling_execution_audit_20261114_v2
Later audits: --registry <output-dir>/controller_registry.json --output-dir <output-dir>
Only the local audit directory is written. Remote sampling uses the same stdlib
script over SSH stdin; it never imports training code or changes a remote run.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import math
import re
import socket
import sqlite3
import struct
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

SEEDS = [42, 2026, 3407]
JOURNALS = {
    "campaign_status.json": "group_holdout_inner5",
    "internal_campaign_status.json": "internal5_no_holdout",
    "qualitative_status.json": "nested5x5_source_qualitative",
}
TERMINAL = {"completed", "requirement_incomplete", "failed"}
ARMS = ("molecule_only", "context_only", "joint")
HC10_CANONICAL_QUANTITY = "source_reported_nominal_human_erythrocyte_10pct_release_concentration"
HC10_CANONICAL_REPORT = "hc10_human_canonical_molecular_diagnostics_v2"
VOLUME_REPORT = "conditional_volume_estimand_molecular_diagnostics_v1"
VOLUME_CONTEXT = "conditional_volume_estimand_context_v1"
CONTROL_TRAINING_SHA = "37bd7ac26bdf029e745bfeb062f27f13379c6ac0f65e21f46d587ca4fd65539f"
CONTROL_PRODUCER_SHAS = {
    "volume": "88b897bef2e3ae8ac416577147b91a1cbaf2a4c6c4c7db4ee948eb7b05e68b9c",
    "native": "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9",
}
CONTROL_VOLUME_ENCODER_SHA = "c0f83e3e0fc7a9cc9b411a3156c84bfb03073118772fbaaaba66fee2b419fc3d"
CONTROL_FLOAT64_SCHEMA = "fixed_ridge_float64_diagnostic_recovery_v1"
CONTROL_FLOAT64_POLICY = "uniform_exact_float32_design_upcast_alpha1_lsqr_v1"
CONTROL_FLOAT64_TOOL_SHA = "a7583f837bb12d3a5086b9c73f52746564872d894f7f1e5e376703a1dc515c12"
MHC10_QUALIFIED_OBSERVATION = "NEW:07:40000000022951890"
HC50_PRIMARY_REPORT = "hc50_primary_corrected_molecular_diagnostics_v1"
HC50_PRIMARY_QUANTITY = "source_reported_nominal_human_erythrocyte_half_lysis_concentration"
HC50_PRIMARY_OBSERVATION = "OLD:PV13:712120ca59fa82219e3e3920"
HC50_PRIMARY_GRAPH = "CANON_SMILES:259c12a291ad369dc5f2df0034afd4ebc3c3450590d813e1fa3a05342d17c887"
HC50_PRIMARY_FAMILY = "FAMILY:d5ae42256f5c2fe6aa4b79f9"
HC50_RAW_FAMILY = "FAMILY:1ec6ec05884fb27225c98e9c"
HC50_OLD_TABLE_SHA = "7cab364576f1eb01c30d7f0d2013bfe3932ed658322fc8cc45c4b84b1ab7a195"
HC50_SOURCE_PINS = {
    "evidence": "0ab13cd339e2bb229b2b36610f0543867eb932b425ad6718d807f120da030ba4",
    "identity": "644f62ec38c1be8e8cfb725377c1d87c9aa539573e2161d6cbc231e5158dbd0f",
    "original_record": "5379777048bc730a50ade1714464e67d0cf3fbe7e826e310b2bda9ccaa230ceb",
    "article_pdf": "292923913137cb2cad6fa969c8f26e5fc7ecb8844dfb664bdf424c87f07a6879",
}
MHC10_NATIVE_TASK = "TASK_MHC10_NATIVE_4acc191c1781762b876225a0"
MHC10_NATIVE_QUANTITY = "native_source_reported_MHC10_concentration_annotation"
MHC10_NATIVE_REPORT = "mhc10_native_species_molecular_diagnostics_v1"
MHC10_NATIVE_SCHEMA = "mhc10_native_species_conditioned_annotation_v1"
MHC10_NATIVE_OLD_SHA = "248ee6fd81f49c3604e152ee81d84fea4807b02bb5feb4ef4228aad22623f478"
MHC10_NATIVE_MEL_GRAPH = (
    "CANON_SMILES:f9420c9f1b030adf9d9fe7626706f6eca5574280a79f1daea103aba9000109c0"
)
MHC10_NATIVE_MEL_GROUP = "FAMILY:e3630b19efc0fb3b36837673"
MHC10_NATIVE_CONTEXT = "rbc_species_context"
MHC10_NATIVE_CONTEXT_NAMES = ["rbc_species_context:human", "rbc_species_context:mouse"]
MHC10_NATIVE_POINTS = {
    "NEW:06:40000000024425451": ("2", "human", "35294199", "6.5"),
    "NEW:06:40000000024425438": ("4", "human", "35294199", "7.4"),
    "NEW:07:40000000022951890": ("0.35", "human", "28002968", ""),
    "NEW:07:40000000019038346": ("1", "human", "30958004", ""),
    "NEW:07:40000000023221359": ("2.26", "mouse", "34180670", ""),
    "NEW:07:40000000022808186": ("2", "mouse", "33422980", ""),
}
MHC10_NATIVE_OBJECT_PINS = {
    "NEW:06:40000000024425438": "0f0e7f111ff6642e733bdfb6d266bba52515fdc5c1a2a707086def2fc3fad019",
    "NEW:06:40000000024425451": "983c21d6122374fd8153fe4b0a6147127c626f6a6ab057dfe5ca2ce6cc1e4d6a",
    "NEW:07:40000000019038346": "02a8a62a41ff5039b32304296ca2dd492161d2456f29409f639fdbc31d0734da",
    "NEW:07:40000000022808186": "caca1218333c3327fa4412425fdb000da4e630526622212059848bd41fdbc121",
    "NEW:07:40000000022951890": "54dabe2c351f70771820945360580e5a8409d80589b5948e9c7cd942e9dc324d",
    "NEW:07:40000000023221359": "8a7c19eb548fb6f1310e102f1332bc7927423462d7011d6cecc8c61fc9ae0e81",
}
MHC10_NATIVE_SOURCE_PINS = {
    "review": "0fce12f02dec174df639c21f321ff0ad136f48e1430da4ebfde656ff4e57f53f",
    "dramp_manifest": "7858b1693112a8ad2241be8d4b415421afe689ef966bf3e403c0295307b8d668",
    "dramp_mapping": "206cbbf610f6dd5d60736b9e6fe30ef7ca863c1840610ae1fa059c94e6cea357",
    "dramp_xlsx": "2196e49100cae4553045873ce640eadb6168555dec5097686842e316ba462f2b",
    "dramp_source_pin": "bba750a61e77eb8387fb238b221e3dd92348f8d9b23ea25f48e82e3af43e9b7c",
}
MHC10_NATIVE_BANK_PINS = {
    "rbc_species_context.npy": "7a754a0c0979a087ebbd124864521778417816830fa81eda88d3c039558396e1",
    "rbc_species_context_valid.npy": (
        "1117241cb138ddf89a6800e4256976248aa25cb70e68433779f5ae734c589168"
    ),
    "rbc_species_index.tsv": "179a8cd553c6b3c3f07ad15af402273d34915682c3ac7403890eee18be8b6f81",
    "record_bank_index.tsv": "4f2bcff7e855eae55f1527b9cbbb3fffa4efc3c23d69aef86f731fdebb5eed87",
    "structure_bank_index.tsv": "4b10cd484138cb2bbf295676d5a2f34374784f4eb799c9fea1372783eeb91484",
    "structure_source_smiles.tsv": (
        "b23c57999fb2ca3ea51c2b8df8fca179ecbbb1ca49daa0c6c9a29e8637bb0127"
    ),
}
GROUP_REPORT_REPAIR_SCHEMA = "peptide_group_followup_report_recovery_v1"
GROUP_REPORT_REPAIR_FAILURES = {
    "KeyError:'counts'",
    "VerificationError:Nonfinite inverse-transformed predictions",
}
GROUP_REPORT_OVERRIDES = {
    "scripts/peptide_modeling_summarize.py",
    "scripts/peptide_modeling_pipeline.py",
}
HC50_PRECISION_SCHEMA = "hc50_precision_report_repair_v1"
HC50_PRECISION_POLICY = "byte_identical_linear_binary32_five_guard_bits_v1"
HC50_PRECISION_TOOL_SHA = "bf83afa1ffb341ad103c723952ac34d7465d6a9646414f6170a70f47046be701"
HC50_PRECISION_LEGACY_SHA = "5bc218673c9cde599c24e7e66a2b9b3e11bd411254419f91b9bbbdfe97e6cb39"
HC50_PRECISION_NUMERIC_SHA = "27d4aa3d737ea88178a6ffb00e3a047ebf3e529af3885bde0fc271dfe4c140f2"
HC50_PRECISION_PAIR = ("NEW:06:120665677696743607", "NEW:06:120846304306691456")
HC50_PRECISION_GRAPH = (
    "CANON_SMILES:f1dfa3c0f579db37febbbd9bb25a3019ffdeb67372f8d49e6dab456b116cad1a"
)
HC50_PRECISION_FAMILY = "FAMILY:f3e1577d66327ec74d73540c"
HC50_PRECISION_FAILURE = (
    "ValueError:One structure-only graph received inconsistent predictions within its role"
)
QUAL_REPORT_SCHEMA = "qualitative_zero_mass_ece_report_recovery_v1"
QUAL_REPORT_POLICY = "zero_effective_weight_calibration_bins_skipped"
QUAL_REPORT_HELPER_SHA = "5ca37bb160875d8875ec32ec0bb84da68991ea68961de06ab39d94b5e291f83c"
QUAL_REPORT_LEARNER_SHA = "a90f6aebfa3ad2c7409540c410c2c227504193d7cf9ad0fd7c43204453abe7ae"
QUAL_REPORT_CONFIG_SHA = "4df9137e67f87215f76a18b4ae5940f41d889d4704c68a2b4e5ecc5dde8b12b9"
QUAL_LEGACY_CONFIG_SHA = "024314e77438617cafbc02b71911234ee4c3bc7ea3fad85d832cebfe3040114d"
QUAL_DIAGNOSIS_SHA = "01a0b55a0b15f2d13cd13756c12fd50d2342a31cbaa47ee96c4ccf909295e70e"
QUAL_FAILED_JOURNAL_SHA = "89f5603cb4c2cbb314e8547a35bdf93e2e41de907c7806d5f44fc92980341f5f"
QUAL_SNAPSHOT_SHA = "7276188e7f0255412f8b4f0c0b2ade3110ed212c4779c57a84e56e19dbc9f197"
QUAL_OLD_OWNER = {"pid": 1978264, "start_ticks": 93195935}
QUAL_CONTEXT = ("cell", "species", "readout_family")
AC50_NATIVE_TASK = "TASK_AC50_NATIVE_6830ccaf49010e62a369534a"
AC50_NATIVE_QUANTITY = "native_source_reported_AC50_concentration_given_semantic_assay_annotation"
AC50_NATIVE_REPORT = "ac50_native_semantic_molecular_diagnostics_v2"
AC50_NATIVE_SCHEMA = "ac50_native_semantic_source_annotation_v1"
AC50_NATIVE_CONTEXT = "ac50_semantic_context"
AC50_NATIVE_TOOL_SHA = "41de6cafe735a16e075525bb38f1880bb73f62bff9a55739c43ffc7b491bc28f"
AC50_NATIVE_HELPER_SHA = "658926a7e1182b2f48e98eaa3bd24b5ad879aacad88a9c692bb267e83a314fa9"
AC50_NATIVE_SOURCE_SHA = "ed344f07c505999cf694fbf992fd3c7a742cf573ca30ee272c90d0ca0131e71b"
AC50_NATIVE_PLAN_SHA = "2226c1b0249ee913de2b492674a1ccf3180c07e26950d8b3b4c8b6861e750fcc"
AC50_NATIVE_POLICY = "uniform_float64_fixed_ridge_from_exact_stored_float32_designs_v2"
AC50_TARGET_TEXT_POLICY = dict(
    schema="source_log10_15g_text_portability_only_v1",
    maximum_binary64_ulp_distance=128,
    maximum_fixed15_significant_digit_decimal_quanta=1,
    all_non_target_source_fields_exact=True,
    frozen_training_target_bytes_unchanged=True,
    applies_to_model_predictions=False,
)
AC50_NATIVE_NR13 = "NEW:08:40000000025164313"
AC50_NATIVE_PDK1 = "NEW:08:40000000003455553"
AC50_NATIVE_P21 = "NEW:08:40000000001291778"
AC50_NATIVE_FIELDS = (
    "native_assay_type",
    "native_readout_family",
    "measured_direction",
    "target_semantic_category",
    "target_resolution",
    "native_target_entity_ids",
    "target_role",
    "species",
    "species_semantic_role",
    "assay_cell_or_expression_host_species",
    "assay_format",
    "detection_method_categories",
    "cell_line",
    "pH",
    "exposure",
    "half_response_normalization_and_fit_rule",
)
AC50_NATIVE_ARTIFACT_PINS = {
    "task_table": "956afa447e7b977801a2752a5575e139fe79265019f510485da500ff4ea5e669",
    "all78_ledger": "df7f75e6034fbc9bfef2f5b5e2306c1dad601e2b122e44b009d6553d0199e71f",
    "ontology": "a8decfb6566bbbf79409652a0666051ec2be4d8b5e908c72d5cfdc75e1b4c67c",
    "predeclared_splits": AC50_NATIVE_PLAN_SHA,
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def contained(root, name):
    root = Path(root).resolve()
    path = Path(name)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Evidence path escapes its frozen run root")
    return path


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def process_identity(pid, proc_root=Path("/proc")):
    try:
        root = Path(proc_root) / str(pid)
        raw = (root / "stat").read_text()
        parts = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": int(pid),
            "start_ticks": int(parts[19]),
            "state": parts[0],
            "argv": (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_state(expected, actual):
    if actual is None or actual.get("state") == "Z":
        return "exited"
    if not expected or actual.get("pid") != expected.get("pid"):
        return "unverified"
    return (
        "same_live_process"
        if actual.get("start_ticks") == expected.get("start_ticks")
        else "pid_reused"
    )


def freeze_entry(root, journal_path, journal, actual):
    """Pin small source/config/table metadata while the owned controller is live."""
    require(
        identity_state(journal.get("process_identity"), actual) == "same_live_process",
        "Controller not live",
    )
    argv = actual["argv"]
    config_path = Path(argv[argv.index("--config") + 1]).resolve()
    cfg = read(config_path)
    require(sha(config_path) == journal.get("config_sha256"), "Journal/config hash mismatch")
    template = Path(cfg["template_run"]).resolve()
    protocol = JOURNALS[journal_path.name]
    manifest_path = template / (
        "source_task_manifest.json" if protocol.startswith("nested") else "task_manifest.json"
    )
    manifest = read(manifest_path)
    if protocol.startswith("nested"):
        tasks = [
            {
                "task_id": manifest["task_id"],
                "endpoint": manifest["raw_endpoint"],
                "table_path": cfg["table_path"],
                "evidence_tier": manifest["evidence_tier"],
                "scientific_protocol": manifest["split_protocol"],
            }
        ]
    else:
        ids = set(cfg["task_ids"])
        tasks = [t for t in manifest["tasks"] if t["task_id"] in ids]
        require({t["task_id"] for t in tasks} == ids, "Declared tasks differ from manifest")
    pinned = []
    for task in tasks:
        path = contained(template, task["table_path"])
        rows = table(path)
        pinned.append(
            {
                "task_id": task["task_id"],
                "endpoint": task["endpoint"],
                "table_relative": path.relative_to(template).as_posix(),
                "table_sha256": sha(path),
                "records": len(rows),
                "evidence_tier": task.get("evidence_tier", "see_pinned_source_manifest"),
                "scientific_definition": task.get("task_definition", {}),
            }
        )
        if task.get("observation_endpoint_coverage") is not None:
            pinned[-1]["observation_endpoint_coverage"] = task["observation_endpoint_coverage"]
        if task.get("native_source_observation_coverage") is not None:
            pinned[-1]["native_source_observation_coverage"] = task[
                "native_source_observation_coverage"
            ]
        if task.get("registered_reference_observation_coverage") is not None:
            pinned[-1]["registered_reference_observation_coverage"] = task[
                "registered_reference_observation_coverage"
            ]
    scopes = ["common"]
    template_config = None
    if cfg.get("template_config"):
        path = Path(cfg["template_config"]).resolve()
        settings = read(path)
        scopes = sorted({"common", *settings.get("coverage_cohorts", {})})
        template_config = {"path": str(path), "sha256": sha(path)}
    require(cfg.get("required_seeds", SEEDS) == SEEDS, "Unexpected required seeds")
    report_repairs = freeze_report_repair_evidence(root, journal)
    precision_pin = freeze_hc50_precision_evidence(root, journal)
    native_pins = {}
    if any(
        task.get("native_source_observation_coverage") is not None
        or task.get("registered_reference_observation_coverage") is not None
        for task in tasks
    ):
        feature_path = template / "features/manifest.json"
        registry_path = template / "representation_methods.json"
        native_pins["native_feature_contract"] = {
            "path": str(feature_path),
            "sha256": sha(feature_path),
            "registry_path": str(registry_path),
            "registry_sha256": sha(registry_path),
        }
    return {
        "controller_id": root.name,
        "output_root": str(root.resolve()),
        "journal_path": str(journal_path.resolve()),
        "protocol": protocol,
        "config_path": str(config_path),
        "config_sha256": sha(config_path),
        "template": str(template),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "template_config": template_config,
        "tasks": pinned,
        "expected_scopes": scopes,
        "required_seeds": SEEDS,
        "owned_process_identity": journal["process_identity"],
        "journal_sha256_at_freeze": sha(journal_path),
        "state_at_freeze": journal["status"],
        **native_pins,
        **({"report_repair_evidence": report_repairs} if report_repairs else {}),
        **({"hc50_precision_evidence": precision_pin} if precision_pin is not None else {}),
    }


def discover(root=Path("/tmp")):
    entries = []
    for directory in sorted(Path(root).glob("peptide_modeling_qlyu_*")):
        if not directory.is_dir():
            continue
        for name in JOURNALS:
            path = directory / name
            if not path.is_file():
                continue
            j = read(path)
            actual = process_identity(j.get("pid"))
            # Only live supplemental controllers are frozen. Main campaign has
            # no task_ids; the qualitative controller uses a separate schema.
            if not j.get("task_ids") and name != "qualitative_status.json":
                continue
            if j.get("status") not in {"waiting_parent", "running"}:
                continue
            entries.append(freeze_entry(directory, path, j, actual))
    require(len({e["controller_id"] for e in entries}) == len(entries), "Repeated controller")
    return {
        "schema": "peptide_controller_registry_v2",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "controllers": entries,
    }


def verify_hashes(root, result, required):
    artifacts, receipts = result.get("artifacts", {}), result.get("artifact_hashes", {})
    require(required <= set(artifacts), "Required model/prediction contracts missing")
    require(all(artifacts[k] in receipts for k in required), "Required artifact not hash covered")
    require(bool(receipts), "No fitted-artifact hashes")
    paths = {}
    for name, receipt in receipts.items():
        path = contained(root, name)
        require(
            path.is_file() and sha(path) == receipt.get("sha256"),
            "Fitted artifact absent or SHA mismatch",
        )
    for key, name in artifacts.items():
        paths[key] = contained(root, name)
    return paths, len(receipts)


def actual_folds(path, fold_column="fold", group_column="split_group", extra_groups=()):
    rows = table(path)
    require(bool(rows), "No saved OOF predictions")
    require(
        {int(r[fold_column]) for r in rows} == set(range(5)),
        "Saved predictions do not contain five actual folds",
    )
    require(len({r["record_id"] for r in rows}) == len(rows), "Repeated OOF record")
    for key in (group_column, *extra_groups):
        seen = defaultdict(set)
        for row in rows:
            seen[row[key]].add(row[fold_column])
        require(
            all(len(values) == 1 for values in seen.values()), "Saved OOF identity crosses folds"
        )
    return rows


def source_membership(predictions, source, group_key="split_group", require_all=False):
    by_id = {r["record_id"]: r for r in source}
    ids = {r["record_id"] for r in predictions}
    require(len(by_id) == len(source) and ids <= set(by_id), "Predictions escape source task")
    require(not require_all or ids == set(by_id), "Predictions omit source records")
    for row in predictions:
        original = by_id[row["record_id"]]
        target = original.get("target", original.get("y"))
        require(float(row["target"]) == float(target), "Prediction target differs from source")
        require(row[group_key] == original[group_key], "Prediction group differs from source")


def verify_standard_result(run, result, task, protocol, proof):
    require(result.get("status") == "completed", "Result is not completed")
    require(result.get("task_id") == task["task_id"], "Result task differs from registry")
    require(
        result.get("actual_inner_folds") == result.get("requested_inner_folds") == 5,
        "Result does not prove five folds",
    )
    require(
        result.get("search", {}).get("backend") == "optuna", "Actual search backend is not Optuna"
    )
    internal = protocol == "internal5_no_holdout"
    required = (
        {"model", "internal_oof", "input_contract", "model_contract"}
        if internal
        else {"model", "oof", "holdout", "outer_split", "model_contract"}
    )
    paths, count = verify_hashes(run, result, required)
    oof = actual_folds(paths["internal_oof" if internal else "oof"])
    source = table(Path(run) / task["table_relative"])
    source_membership(oof, source)
    contract = read(paths["model_contract"])
    require(
        contract.get("input_hash") == result.get("input_hash") and bool(result.get("input_hash")),
        "Model/result input signature mismatch",
    )
    if internal:
        require(
            result.get("protocol") == "internal_grouped_cv_only_no_independent_holdout",
            "Internal protocol changed",
        )
        require(
            result.get("independent_holdout_available") is False and "holdout" not in result,
            "Internal result fabricates holdout",
        )
        require(
            proof.get("actual_folds") == 5
            and proof.get("model_reload_verified_on_training_host") is True
            and proof.get("status") == "verified"
            and proof.get("artifact_hashes_verified", 0) > 0,
            "Internal persisted-model verification missing",
        )
    else:
        require(
            contract.get("holdout_not_used_for_selection") is True,
            "Holdout-selection independence missing",
        )
        held = table(paths["holdout"])
        require(bool(held), "Missing actual holdout predictions")
        source_membership(held, source)
        require(
            not ({r["record_id"] for r in oof} & {r["record_id"] for r in held}),
            "Holdout record leakage",
        )
        require(
            not ({r["split_group"] for r in oof} & {r["split_group"] for r in held}),
            "Holdout group leakage",
        )
    if is_hc50_primary_task(task):
        require(protocol == "group_holdout_inner5", "HC50 primary protocol changed")
        verify_hc50_full_refit(run, result, task, source)
    if is_mhc10_native_task(task):
        require(protocol == "group_holdout_inner5", "Native MHC10 protocol changed")
        try:
            verify_mhc10_native_refit(run, result, task, source)
        except AttributeError as exc:
            raise ValueError("Malformed native MHC10 refit metadata") from exc
    return {
        "status": "completed_verified",
        "artifact_hashes_verified_now": count,
        "actual_inner_folds": 5,
        "actual_backend": "optuna",
    }


def freeze_report_repair_evidence(root, journal):
    """Freeze small adoption receipts while the new controller identity is observable."""
    frozen = {}
    recovery_root = Path(root).resolve() / "report_recovery"
    for round_entry in journal.get("rounds", []):
        descriptor = round_entry.get("followup_report_repair")
        if descriptor is None:
            continue
        require(isinstance(descriptor, dict), "Report-repair descriptor is not an object")
        path = contained(recovery_root, descriptor["receipt_path"])
        require(
            sha(path) == descriptor["sha256"],
            "Report-repair receipt changed before registry freeze",
        )
        receipt = read(path)
        require(
            receipt.get("schema") == GROUP_REPORT_REPAIR_SCHEMA
            and receipt.get("status") == "report_repair_verified_not_adopted",
            "Unsupported immutable report-repair receipt",
        )
        seed = str(round_entry["seed"])
        require(seed not in frozen, "Repeated report-repaired seed")
        frozen[seed] = json.loads(
            json.dumps(
                dict(
                    descriptor=descriptor,
                    failed_journal_sha256=receipt["failed_journal_sha256"],
                    retained_failed_archive=receipt["retained_failed_archive"],
                    runtime_revision=receipt["runtime_revision"],
                )
            )
        )
    return frozen


def verify_report_runtime_revision(revision, contract):
    require(isinstance(revision, dict), "Report-only runtime revision is required")
    manifest_path = Path(revision["manifest_path"])
    require(
        manifest_path.is_absolute()
        and not manifest_path.is_symlink()
        and sha(manifest_path) == revision["manifest_sha256"],
        "Report runtime manifest SHA changed",
    )
    manifest = read(manifest_path)
    overrides = revision.get("overrides")
    require(
        manifest.get("schema") == "peptide_postprocess_override_v1"
        and isinstance(overrides, dict)
        and bool(overrides)
        and not (set(overrides) - GROUP_REPORT_OVERRIDES)
        and manifest.get("overrides") == overrides,
        "Runtime revision changes a training or unsupported module",
    )
    forbidden = {
        "fit",
        "fit_transform",
        "partial_fit",
        "predict",
        "predict_proba",
        "_predict",
        "_fit_model",
        "_refit_selected",
        "train_task",
        "train_internal_task",
        "train_seed",
        "search_candidates",
        "full_stage",
        "run_campaign",
        "supervise",
        "Popen",
    }
    for name, item in overrides.items():
        original = contained(contract["template"], name)
        replacement = Path(item["path"])
        require(
            sha(original) == item["base_sha256"] == contract["files"][name]
            and replacement.is_absolute()
            and not replacement.is_symlink()
            and sha(replacement) == item["sha256"],
            "Reporting replacement/base source SHA changed",
        )
        changed = ast.parse(replacement.read_text(encoding="utf-8-sig"))
        nodes = [changed]
        if name == "scripts/peptide_modeling_pipeline.py":
            baseline = ast.parse(original.read_text(encoding="utf-8-sig"))
            functions = []
            for tree in (baseline, changed):
                matches = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "report_stage"
                ]
                require(
                    len(matches) == 1, "Pipeline override needs exactly one ordinary report_stage"
                )
                functions.append(matches[0])
            nodes = list(functions[1].body)
            for function in functions:
                function.body = []
            require(
                ast.dump(baseline, include_attributes=False)
                == ast.dump(changed, include_attributes=False),
                "Pipeline override changed code/signature/decorators outside report_stage",
            )
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    called = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                    )
                    require(
                        called not in forbidden,
                        "Report-only replacement invokes model training or prediction",
                    )
                    require(
                        not (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "load"
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id in {"joblib", "torch", "pickle"}
                        ),
                        "Report-only replacement reloads a model",
                    )
    return overrides


def report_repair_archive(path, receipt, expected, root_name, protected=(), bound_receipts=None):
    """Stream original/sidecar archives without extracting or deserializing any model."""
    path = Path(path)
    require(
        path.is_file()
        and not path.is_symlink()
        and sha(path) == receipt["sha256"]
        and ("bytes" not in receipt or path.stat().st_size == receipt["bytes"]),
        "Original or repaired report archive changed",
    )
    names = {root_name + "/" + name.replace("\\", "/"): value for name, value in expected.items()}
    found, bound_found = set(), set()
    bound = {root_name + "/" + name: value for name, value in (bound_receipts or {}).items()}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name in bound:
                require(
                    member.isfile() and member.name not in bound_found,
                    "Archived final launch receipt is duplicate or nonregular",
                )
                with archive.extractfile(member) as stream:
                    final = json.load(stream)
                require(
                    all(final.get(k) == value for k, value in bound[member.name].items()),
                    "Empty startup observation conflicts with original archived final receipts",
                )
                bound_found.add(member.name)
            selected = member.name in names
            if not selected and any(
                member.name.startswith(root_name + "/" + prefix) for prefix in protected
            ):
                require(
                    not member.isfile(), "Original archive contains a missing scientific/trial file"
                )
            if not selected:
                continue
            require(
                member.isfile() and member.name not in found,
                "Archived evidence is duplicate or nonregular",
            )
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            require(
                value.hexdigest() == names[member.name],
                "Current scientific bytes differ from original archive",
            )
            found.add(member.name)
    require(found == set(names), "Original archive omits a retained scientific/report artifact")
    require(bound_found == set(bound), "Original archive omits bound final launch receipts")


def report_repair_launch_identity(receipt, historical, supervisor, dispatcher, run, workers):
    """Allow only the observed empty-exec race, without inventing its old argv."""
    identity, command = historical.get("process_identity"), historical.get("argv")
    require(
        isinstance(identity, dict)
        and identity.get("pid") == historical.get("pid")
        and identity.get("start_ticks")
        and isinstance(command, list)
        and len(command) == 8
        and isinstance(workers, int)
        and not isinstance(workers, bool)
        and workers > 0,
        "Original process PID/start and exact configured Popen command are required",
    )
    expected = [
        command[0],
        str(run / "scripts/peptide_modeling_supervise.py"),
        "--run-dir",
        str(run),
        "--config",
        str(run / "config.json"),
        "--workers",
        str(workers),
    ]
    require(command == expected, "Original supervisor Popen command changed run/config/workers")
    observed = identity.get("argv")
    if observed != []:
        require(
            isinstance(observed, list) and bool(observed) and observed == command,
            "Missing or wrong nonempty argv is not an observed empty-exec race",
        )
        return None
    kind = "empty_exec_startup_observation_with_bound_final_receipts"
    records = receipt.get("identity_observations", [])
    require(isinstance(records, list), "Empty startup observation evidence missing")
    matches = [r for r in records if isinstance(r, dict) and r.get("seed") == historical["seed"]]
    require(len(matches) == 1, "Empty startup observation requires one seed-specific launch proof")
    observation = matches[0]
    dispatch = [
        command[0],
        str(run / "scripts/peptide_modeling_dispatch.py"),
        "--run-dir",
        str(run),
        "--config",
        str(run / "config.json"),
        "--profile",
        "full",
        "--workers",
        str(workers),
        "--remote-root",
        str(run),
    ]
    require(
        observation.get("identity_evidence") == kind
        and observation.get("observed_argv") == []
        and observation.get("recorded_process_identity") == identity
        and observation.get("recorded_popen_argv") == command
        and observation.get("final_supervisor_dispatcher_argv")
        == supervisor.get("argv")
        == dispatch
        and isinstance(dispatcher.get("pid"), int)
        and dispatcher["pid"] > 0
        and observation.get("dispatcher_pid")
        == supervisor.get("dispatcher_pid")
        == dispatcher["pid"]
        and supervisor.get("pid") == historical["pid"]
        and observation.get("archived_final_receipts_bound") is True,
        "Empty startup observation lacks exact final command/PID/archive bindings",
    )
    if historical["seed"] == receipt["repaired_seed"]:
        require(
            receipt.get("identity_evidence") == kind,
            "Repaired seed lost its empty startup observation qualifier",
        )
    return {
        "full_supervisor.json": {
            k: supervisor[k]
            for k in ("pid", "config_sha256", "task_manifest_sha256", "argv", "dispatcher_pid")
        },
        "dispatch/latest_full.json": {
            k: dispatcher[k]
            for k in ("pid", "config_sha256", "task_manifest_sha256", "status", "tasks")
        },
    }


def report_repair_owned_process(actual, run):
    if actual is None or actual.get("state") == "Z":
        return False
    argv = actual.get("argv", [])
    return (
        "--run-dir" in argv
        and argv.index("--run-dir") + 1 < len(argv)
        and (argv[argv.index("--run-dir") + 1] == str(run))
    )


def report_repair_argument(identity, flag, expected):
    argv = identity.get("argv", [])
    return (
        flag in argv
        and argv.index(flag) + 1 < len(argv)
        and argv[argv.index(flag) + 1] == str(expected)
    )


def report_repair_process_release(prior, old_round, supervisor, dispatcher, run):
    for expected in (prior["process_identity"], old_round["process_identity"]):
        require(
            isinstance(expected, dict) and expected.get("pid") and expected.get("start_ticks"),
            "Original report-failure process identity missing",
        )
        actual = process_identity(expected["pid"])
        require(
            identity_state(expected, actual) in {"exited", "pid_reused"},
            "An original report-failure process is still live or unverified",
        )
    for value in (supervisor, dispatcher, *dispatcher.get("tasks", {}).values()):
        if value.get("pid"):
            actual = process_identity(value["pid"])
            require(
                not report_repair_owned_process(actual, run),
                "An original seed worker/process is still live",
            )
    proc = Path("/proc")
    for directory in proc.iterdir() if proc.is_dir() else []:
        if directory.name.isdigit():
            require(
                not report_repair_owned_process(process_identity(int(directory.name)), run),
                "An unrecorded process still owns the retained failed seed",
            )


def report_repair_trials(run, result):
    directory = contained(run, result["search"]["search_dir"])
    paths = list((directory / "trials").glob("*.json"))
    require(paths, "Retained model lacks successful Optuna trial evidence")
    rejected = {
        (
            v["route"],
            v["model"],
            json.dumps(v["params"], sort_keys=True, ensure_ascii=False, allow_nan=False),
        )
        for v in result.get("search", {})
        .get("candidate_fit_admission", {})
        .get("rejected_candidates", [])
    }
    selected, hashes = False, {}
    for path in paths:
        record = read(path)
        hashes[str(path.resolve())] = sha(path)
        if "evaluation" in record:
            value = record["evaluation"]
            digest = hashlib.sha256(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            require(
                digest == record.get("record_sha256"), "Retained Optuna trial wrapper SHA changed"
            )
        else:
            value = record
        if value.get("status") != "completed":
            continue
        require(
            all(
                value[k] == 5
                for k in (
                    "actual_folds",
                    "requested_folds",
                    "actual_inner_folds",
                    "requested_inner_folds",
                    "n_inner_folds",
                )
                if k in value
            ),
            "A successful Optuna trial is not fivefold",
        )
        for key, expected in (
            ("input_hash", result.get("input_hash")),
            ("search_input_signature", result["search"].get("namespace")),
        ):
            require(
                expected is None or key not in value or value[key] == expected,
                "A successful Optuna trial belongs to another input/search",
            )
        require(
            math.isfinite(float(value["score"])), "A successful Optuna trial has a nonfinite score"
        )
        key = (
            value["route"],
            value["model"],
            json.dumps(value["params"], sort_keys=True, ensure_ascii=False, allow_nan=False),
        )
        if key not in rejected and key == (
            result["selected_route"],
            result["selected_model"],
            json.dumps(
                result["selected_params"], sort_keys=True, ensure_ascii=False, allow_nan=False
            ),
        ):
            selected = True
    require(selected, "Selected retained model has no successful original Optuna candidate")
    return hashes, directory / "trials"


def verify_group_report_repair(entry, round_entry, seed):
    """Accept a sidecar only after immutable archival proof and explicit CAS adoption."""
    root, run = (
        Path(entry["output_root"]).resolve(),
        contained(entry["output_root"], round_entry["run_dir"]),
    )
    descriptor = round_entry.get("followup_report_repair")
    require(
        isinstance(descriptor, dict), "A sidecar requires an explicit group report-repair receipt"
    )
    pin = entry.get("report_repair_evidence", {}).get(str(seed))
    require(
        isinstance(pin, dict) and pin.get("descriptor") == descriptor,
        "Report-repair adoption evidence was not frozen with the new controller",
    )
    path = contained(root / "report_recovery", descriptor["receipt_path"])
    require(
        path.name == "repair_receipt.json" and sha(path) == descriptor["sha256"],
        "Repair receipt SHA changed",
    )
    recovery = path.parent
    receipt = read(path)
    require(
        receipt.get("schema") == GROUP_REPORT_REPAIR_SCHEMA
        and receipt.get("status") == "report_repair_verified_not_adopted"
        and receipt.get("original_failure") in GROUP_REPORT_REPAIR_FAILURES
        and receipt.get("repaired_seed") == seed
        and Path(receipt["source_run"]).resolve() == run
        and receipt.get("source_config_sha256") == entry["config_sha256"]
        and Path(receipt["source_followup_config"]).resolve()
        == Path(entry["config_path"]).resolve()
        and receipt.get("models_refitted") == receipt.get("productive_time_added") == 0
        and receipt.get("selection_repeated") is False
        and receipt.get("additional_twenty_hour_loop") is False
        and receipt.get("original_models_results_inputs_modified") is False,
        "Repair receipt changes model selection, fitted bytes, time credit or source scope",
    )
    require(
        receipt["failed_journal_sha256"] == pin["failed_journal_sha256"]
        and receipt["retained_failed_archive"] == pin["retained_failed_archive"]
        and receipt["runtime_revision"] == pin["runtime_revision"],
        "Frozen repair/archive/runtime proof changed",
    )
    current = read(entry["journal_path"])
    cfg = read(entry["config_path"])
    require(
        sha(entry["config_path"]) == current.get("config_sha256") == receipt["source_config_sha256"]
        and current.get("process_identity") == entry["owned_process_identity"]
        and current.get("required_seeds") == cfg.get("required_seeds", SEEDS) == SEEDS
        and current.get("additional_twenty_hour_loop") is False,
        "Adopted controller changed its original config/seeds/time policy",
    )
    originals = receipt["original_files"]
    journal_path = root / "campaign_status.json"
    original_journal = originals[str(journal_path)]
    prior_path = contained(recovery / "original", original_journal["backup"])
    require(
        sha(prior_path) == original_journal["sha256"] == receipt["failed_journal_sha256"],
        "Original failed controller journal backup changed",
    )
    prior = read(prior_path)
    require(
        prior.get("schema") == "peptide_bounded_seed_followup_v1"
        and prior.get("status") == "failed"
        and prior.get("pid") == prior["process_identity"]["pid"]
        and prior.get("config_sha256") == receipt["source_config_sha256"]
        and prior.get("required_seeds") == SEEDS
        and prior.get("task_ids") == current.get("task_ids") == cfg.get("task_ids")
        and prior.get("additional_twenty_hour_loop") is False,
        "Original failed journal scope changed",
    )
    require(
        report_repair_argument(prior["process_identity"], "--config", entry["config_path"])
        and report_repair_argument(current["process_identity"], "--config", entry["config_path"]),
        "Original or recovered controller argv belongs to another config",
    )
    old_rounds = prior["rounds"]
    require(
        0 < len(old_rounds) <= 3
        and [r["seed"] for r in old_rounds] == SEEDS[: len(old_rounds)]
        and old_rounds[-1]["seed"] == seed
        and receipt["missing_seeds"] == SEEDS[len(old_rounds) :]
        and len(current["rounds"]) >= len(old_rounds)
        and [r["seed"] for r in current["rounds"]] == SEEDS[: len(current["rounds"])],
        "Report recovery restaged or changed the required seed prefix/suffix",
    )
    old = old_rounds[-1]
    require(
        old.get("status") == "failed"
        and old.get("exit_code")
        == round_entry.get("exit_code")
        == descriptor.get("original_supervisor_exit_code")
        == 1
        and round_entry.get("postprocess_exit_code") == 0
        and round_entry.get("status") == "completed"
        and old.get("pid") == old.get("process_identity", {}).get("pid"),
        "Historical failed supervisor exit/identity was rewritten as successful training",
    )
    revision = receipt["runtime_revision"]
    expected_old = {
        **old,
        "status": "completed",
        "audit": receipt["audit"],
        "postprocess_exit_code": 0,
        "verification_runtime_revision": revision,
        "followup_report_repair": descriptor,
    }
    require(
        round_entry == expected_old
        and current["rounds"][: len(old_rounds) - 1] == old_rounds[:-1]
        and current.get("active_runtime_revision") == revision,
        "Adoption changed historical round fields beyond the explicit report sidecar",
    )
    history = [
        h
        for h in current.get("recovery_history", [])
        if h.get("schema") == GROUP_REPORT_REPAIR_SCHEMA
        and h.get("repair_receipt_path") == str(path)
    ]
    require(len(history) == 1, "Report repair lacks one explicit CAS adoption history event")
    event = history[0]
    require(
        event.get("repair_receipt_sha256") == descriptor["sha256"]
        and event.get("source_failed_journal_sha256") == receipt["failed_journal_sha256"]
        and event.get("previous_controller_identity") == prior["process_identity"]
        and event.get("previous_failure_reason") == prior.get("reason", "")
        and event.get("retained_seeds") == [r["seed"] for r in old_rounds]
        and event.get("missing_seeds") == receipt["missing_seeds"]
        and event.get("configuration_unchanged") is True
        and event.get("models_refitted") == 0
        and event.get("additional_twenty_hour_loop") is False,
        "CAS adoption history does not bind the original failed journal/config/seed prefix",
    )
    for field in ("productive_wall_seconds", "owned_worker_cpu_seconds"):
        require(
            current.get(field) == prior.get(field),
            "Report repair credited downtime or extra compute",
        )
    allowed_originals = {
        journal_path,
        run / "full_supervisor.json",
        run / "dispatch/latest_full.json",
        run / "verification.json",
    }
    require(
        {Path(p).resolve() for p in originals} <= allowed_originals
        and {journal_path, run / "full_supervisor.json", run / "dispatch/latest_full.json"}
        <= {Path(p).resolve() for p in originals},
        "Original report-failure metadata backups missing or out of scope",
    )
    repaired_members = {path.relative_to(recovery).as_posix(): sha(path)}
    for original, item in originals.items():
        original = contained(root, original)
        backup = contained(recovery / "original", item["backup"])
        require(sha(backup) == item["sha256"], "Original failed metadata backup changed")
        if original != journal_path:
            require(
                sha(original) == item["sha256"],
                "Original failed process/verification metadata was overwritten",
            )
        repaired_members[backup.relative_to(recovery).as_posix()] = item["sha256"]
    supervisor, dispatcher = (
        read(run / "full_supervisor.json"),
        read(run / "dispatch/latest_full.json"),
    )
    require(
        supervisor.get("status") == dispatcher.get("status") == "failed"
        and dispatcher.get("reason") == receipt["original_failure"]
        and supervisor.get("archive") == receipt["retained_failed_archive"]
        and supervisor.get("pid") == old["pid"]
        and supervisor.get("config_sha256")
        == dispatcher.get("config_sha256")
        == old["config_sha256"]
        and supervisor.get("task_manifest_sha256")
        == dispatcher.get("task_manifest_sha256")
        == sha(run / "task_manifest.json"),
        "Original report failure/archival/process receipts changed",
    )
    report_repair_process_release(prior, old, supervisor, dispatcher, run)
    report_repair_launch_identity(receipt, old, supervisor, dispatcher, run, cfg.get("workers", 32))
    manifest = read(run / "task_manifest.json")
    eligible = {t["task_id"] for t in manifest["tasks"] if t.get("status") == "eligible"}
    require(
        set(dispatcher.get("tasks", {})) == eligible
        and all(t.get("status") in {"completed", "deferred"} for t in dispatcher["tasks"].values()),
        "An actual training failure or missing worker cannot be repaired as reporting",
    )
    contract = read(root / "template_contract.json")
    require(
        Path(contract["template"]).resolve() == Path(entry["template"]).resolve()
        and sha(cfg["template_config"]) == contract["template_config_sha256"]
        and sha(cfg["registry_path"]) == contract["registry_sha256"],
        "Original template/config/registry changed",
    )
    overrides = verify_report_runtime_revision(revision, contract)
    if receipt["original_failure"] == "KeyError:'counts'":
        require(
            "counts" not in manifest and "scripts/peptide_modeling_pipeline.py" in overrides,
            "Counts report repair does not match the absent-counts defect",
        )
    else:
        require(
            "scripts/peptide_modeling_summarize.py" in overrides
            and read(run / "verification.json").get("reason") == receipt["original_failure"],
            "Numerical report repair does not match its original failure",
        )
    parent = receipt["parent_requirement_evidence"]
    require(
        isinstance(parent, dict)
        and math.isfinite(float(parent.get("productive_wall_seconds", 0)))
        and parent.get("productive_wall_seconds", 0) >= 72000
        and parent.get("verified_required_seeds") == SEEDS
        and parent.get("parent_owned_processes_released") is True,
        "Report recovery lacks the original verified twenty-hour parent requirement",
    )
    verification_path = contained(recovery / "verified_reports", receipt["verification_path"])
    require(
        verification_path.name == "verification.json"
        and contained(root, round_entry["audit"]["verification_path"]) == verification_path
        and contained(root, descriptor["verification_path"]) == verification_path
        and sha(verification_path)
        == descriptor["verification_sha256"]
        == receipt["verification_sha256"]
        == round_entry["audit"]["verification_sha256"],
        "Report verification sidecar is not bound to its receipt and round audit",
    )
    verification = read(verification_path)
    require(
        verification.get("status") == "verified"
        and verification.get("source_run_dir") == str(run)
        and verification.get("source_task_manifest_sha256") == sha(run / "task_manifest.json")
        and verification.get("assertions", {}).get("all_checks_passed") is True,
        "Repaired report does not verify the original seed/source manifest",
    )
    repaired_members[verification_path.relative_to(recovery).as_posix()] = sha(verification_path)
    all_retained = receipt["retained_fitted_files_sha256"]
    require(
        isinstance(all_retained, dict) and bool(all_retained),
        "No original fitted evidence retained",
    )
    for name, expected in all_retained.items():
        require(
            sha(contained(root, name)) == expected,
            "An original fitted/trial result changed after report repair",
        )
    for index, historical in enumerate(old_rounds):
        historical_run = contained(root, historical["run_dir"])
        require(
            historical["index"] == index
            and historical_run == root / f"round_{index:03d}_seed_{SEEDS[index]}",
            "Retained round namespace changed",
        )
        expected_config = json.loads(json.dumps(read(cfg["template_config"])))
        expected_config["training"]["seed"] = SEEDS[index]
        expected_config["run_dir"] = str(historical_run)
        expected_config["resources"].update(
            remote_training_scratch=str(historical_run),
            remote_durable_root=str(Path(cfg["durable_root"]) / historical_run.name),
            max_parallel_tasks=32,
        )
        require(
            read(historical_run / "config.json") == expected_config
            and sha(historical_run / "config.json") == historical["config_sha256"],
            "Retained seed execution settings/configuration changed",
        )
        old_revision = historical.get("runtime_revision")
        expected_files = dict(contract["files"])
        if old_revision is not None:
            expected_files.update(
                {
                    name: item["sha256"]
                    for name, item in verify_report_runtime_revision(old_revision, contract).items()
                }
            )
        scientific = {}
        for relative, expected in contract["files"].items():
            require(
                sha(contained(contract["template"], relative)) == expected,
                "Frozen template input changed",
            )
        for relative, expected in expected_files.items():
            require(
                sha(contained(historical_run, relative)) == expected,
                "Retained seed source/feature input changed",
            )
            scientific[relative] = expected
        provenance = dict(
            seed=SEEDS[index], template_contract=contract, config_sha256=historical["config_sha256"]
        )
        if old_revision is not None:
            provenance["runtime_revision"] = old_revision
        require(
            read(historical_run / "campaign_input_contract.json") == provenance,
            "Retained seed provenance changed",
        )
        for relative in ("config.json", "task_manifest.json", "campaign_input_contract.json"):
            scientific[relative] = sha(historical_run / relative)
        settings = read(historical_run / "config.json")
        scopes = {"common", *settings.get("coverage_cohorts", {})}
        tasks = {
            t["task_id"]: t
            for t in read(historical_run / "task_manifest.json")["tasks"]
            if t["status"] == "eligible"
        }
        expected_results = {f"{t}_{s}.json" for t in tasks for s in scopes}
        result_paths = list((historical_run / "results/full").glob("*.json"))
        require(
            {p.name for p in result_paths} == expected_results,
            "Retained model-scope results missing or added",
        )
        protected = ["models/", "results/"]
        for directory in (historical_run / "models", historical_run / "results"):
            for file in directory.rglob("*"):
                if file.is_file():
                    require(not file.is_symlink(), "Scientific archive input is a symlink")
                    scientific[file.relative_to(historical_run).as_posix()] = sha(file)
        for result_path in result_paths:
            result = read(result_path)
            require(
                result.get("status") in {"completed", "deferred"},
                "A real training failure is not report-recoverable",
            )
            require(
                all_retained.get(str(result_path)) == sha(result_path),
                "Retained result is absent from repair proof",
            )
            if result["status"] == "deferred":
                continue
            t = tasks[result["task_id"]]
            spec = dict(t, table_relative=t["table_path"])
            verify_standard_result(historical_run, result, spec, "group_holdout_inner5", {})
            trial_hashes, trial_dir = report_repair_trials(historical_run, result)
            protected.append(trial_dir.relative_to(historical_run).as_posix() + "/")
            scientific.update(
                {
                    Path(p).relative_to(historical_run).as_posix(): value
                    for p, value in trial_hashes.items()
                }
            )
            for file, expected in result["artifact_hashes"].items():
                require(
                    all_retained.get(str(contained(historical_run, file))) == expected["sha256"],
                    "Retained model/prediction/input artifact omitted from repair receipt",
                )
            inner = actual_folds(contained(historical_run, result["artifacts"]["oof"]))
            held = table(contained(historical_run, result["artifacts"]["holdout"]))
            if t["task_kind"] == "regression":
                for role, values in (("inner_cv", inner), ("holdout", held)):
                    computed = error_metrics(
                        [float(r["target"]) for r in values],
                        [float(r["prediction"]) for r in values],
                    )
                    check_metrics(result[role], computed)
                if historical_run == run:
                    proof = [
                        p
                        for p in verification["results"]
                        if str(p["result_file"]).replace("\\", "/")
                        == result_path.relative_to(run).as_posix()
                    ]
                    require(
                        len(proof) == 1
                        and proof[0].get("source_result_sha256") == sha(result_path),
                        "Sidecar verification source result SHA changed",
                    )
                    check_metrics(
                        proof[0]["holdout"],
                        error_metrics(
                            [float(r["target"]) for r in held],
                            [float(r["prediction"]) for r in held],
                        ),
                    )
        supervisor_receipt = read(historical_run / "full_supervisor.json")
        archive = supervisor_receipt["archive"]
        historical_dispatch = read(historical_run / "dispatch/latest_full.json")
        bound = report_repair_launch_identity(
            receipt,
            historical,
            supervisor_receipt,
            historical_dispatch,
            historical_run,
            cfg.get("workers", 32),
        )
        report_repair_archive(
            archive["path"],
            archive,
            scientific,
            historical_run.name,
            protected,
            bound_receipts=bound,
        )
    report_inputs = recovery / "report_inputs"
    require(
        sha(report_inputs / "task_manifest.json") == sha(run / "task_manifest.json"),
        "Report inputs changed source task manifest",
    )
    for file in (run / "results").rglob("*"):
        if file.is_file():
            require(
                sha(report_inputs / "results" / file.relative_to(run / "results")) == sha(file),
                "Report input result copy changed",
            )
    if receipt["original_failure"] == "KeyError:'counts'":
        require(
            read(report_inputs / "run_summary.json")["tasks"]
            == dict(declared_tasks=len(manifest["tasks"]), eligible_tasks=len(eligible)),
            "Repaired task counts changed declared/eligible scope",
        )
    for file in (report_inputs / "run_summary.json",):
        repaired_members[file.relative_to(recovery).as_posix()] = sha(file)
    repaired_archive = descriptor["repaired_reports_archive"]
    durable_recovery = Path(cfg["durable_root"]) / "report_recovery" / recovery.name
    archive_path = contained(durable_recovery, repaired_archive["path"])
    report_repair_archive(archive_path, repaired_archive, repaired_members, recovery.name)
    for relative, expected in repaired_members.items():
        require(
            sha(contained(durable_recovery, relative)) == expected,
            "Durable report-repair receipt/verification/backup copy changed",
        )
    for name, expected in all_retained.items():
        original = Path(name)
        if original.is_relative_to(run):
            mirror = Path(cfg["durable_root"]) / run.name / "live" / original.relative_to(run)
            require(
                sha(mirror) == expected, "Durable preserved model/prediction/trial copy changed"
            )
    return verification_path


def qualitative_report_recovery_mode(entry, journal):
    return entry.get("protocol") == "nested5x5_source_qualitative" and (
        entry.get("qualitative_report_recovery_evidence") is not None
        or journal.get("report_revision") is not None
        or journal.get("report_recovery") is not None
    )


def qualitative_report_recovery_header(entry, journal):
    """Check small immutable composition/ownership receipts even while learning is active."""
    pin = entry.get("qualitative_report_recovery_evidence")
    require(
        isinstance(pin, dict), "Qualitative report revision requires a frozen composite registry"
    )
    require(
        pin.get("schema") == QUAL_REPORT_SCHEMA
        and pin.get("helper_sha256") == QUAL_REPORT_HELPER_SHA
        and all(pin.get("previous_owner", {}).get(k) == v for k, v in QUAL_OLD_OWNER.items()),
        "Qualitative frozen composition lost the approved helper or original owner",
    )
    root, template = Path(entry["output_root"]).resolve(), Path(entry["template"]).resolve()
    require(
        journal.get("report_revision") == QUAL_REPORT_SCHEMA, "Unknown qualitative report revision"
    )
    config_path = Path(pin["recovery_config_path"])
    require(
        sha(config_path) == pin["recovery_config_sha256"] == QUAL_REPORT_CONFIG_SHA,
        "Qualitative recovery configuration pin changed",
    )
    config = read(config_path)
    require(
        config.get("schema") == QUAL_REPORT_SCHEMA
        and config.get("helper_sha256") == QUAL_REPORT_HELPER_SHA
        and config.get("remaining_seeds") == [2026, 3407]
        and Path(config["template"]).resolve() == template
        and Path(config["run_root"]).resolve() == root
        and config.get("legacy_config_sha256")
        == QUAL_LEGACY_CONFIG_SHA
        == entry["config_sha256"]
        == journal.get("config_sha256")
        == sha(entry["config_path"]),
        "Qualitative recovery changed the original learning configuration or seed suffix",
    )
    old_config = read(entry["config_path"])
    require(
        Path(old_config["template_run"]).resolve() == template
        and Path(old_config["output_root"]).resolve() == root
        and old_config.get("required_seeds") == SEEDS
        and old_config["training"].get("n_jobs") == 1
        and old_config["training"].get("inner_folds") == 5
        and old_config["training"].get("search_backend") == "optuna"
        and len(set(old_config["training"].get("models", []))) >= 7
        and journal.get("additional_twenty_hour_loop") is False,
        "Qualitative original bounded learner/CPU/Optuna scope changed",
    )
    owner = journal.get("process_identity", {})
    require(
        owner.get("pid") == journal.get("pid") and owner == entry["owned_process_identity"],
        "Qualitative recovery controller ownership changed",
    )
    argv = owner.get("argv", [])
    helper = Path(pin["helper_path"])
    require(
        helper.is_file()
        and not helper.is_symlink()
        and sha(helper) == QUAL_REPORT_HELPER_SHA
        and str(helper) in argv
        and report_repair_argument(owner, "--config", str(config_path))
        and argv.count("--run-recovery") == 1
        and "--inspect" not in argv,
        "Qualitative owner is not the approved explicit recovery command",
    )
    evidence_path = Path(config["evidence_path"])
    require(
        sha(evidence_path)
        == config["evidence_sha256"]
        == QUAL_DIAGNOSIS_SHA
        == pin["diagnosis_sha256"],
        "Qualitative original fitted diagnosis changed",
    )
    evidence = read(evidence_path)
    history = contained(root, "recovery_history/" + config["recovery_id"])
    require(
        Path(config["recovery_id"]).name == config["recovery_id"]
        and config["recovery_id"] not in {".", ".."},
        "Qualitative recovery history escapes its namespace",
    )
    proof_path = contained(history, "repair_proof.json")
    require(
        sha(proof_path) == pin["repair_proof_sha256"]
        and read(proof_path) == journal.get("report_recovery"),
        "Qualitative adoption proof changed or differs from the current journal",
    )
    proof = read(proof_path)
    require(
        proof.get("schema") == QUAL_REPORT_SCHEMA
        and proof.get("helper_sha256") == QUAL_REPORT_HELPER_SHA
        and proof.get("config_sha256") == QUAL_REPORT_CONFIG_SHA
        and proof.get("legacy_learning_sha256") == QUAL_REPORT_LEARNER_SHA
        and proof.get("old_failed_pid") == QUAL_OLD_OWNER["pid"]
        and proof.get("old_failure_reason")
        == "ZeroDivisionError:Weights sum to zero, can't be normalized"
        and proof.get("old_journal_sha256")
        == evidence.get("journal_sha256")
        == QUAL_FAILED_JOURNAL_SHA
        and proof.get("seed42_training_calls") == 0
        and proof.get("remaining_seeds") == [2026, 3407]
        and proof.get("runtime_override_only") == "_weighted_metrics"
        and proof.get("empty_support_metric_status") == QUAL_REPORT_POLICY,
        "Qualitative report repair changed old fits, failure scope or metric policy",
    )
    require(
        Path(proof["recovered_seed42_result"]).resolve() == root / "seed_42/result.json",
        "Qualitative report proof points to another recovered seed",
    )
    records = [x for x in journal.get("recovery_history", []) if x == proof]
    require(
        len(records) == 1 and journal["recovery_history"][-1] == proof,
        "Qualitative adoption history is not singular and append-only",
    )
    lock_path, owner_path = root / "qualitative_report_recovery.lock", history / "owner.json"
    require(
        sha(lock_path) == pin["ownership_sha256"] and sha(owner_path) == pin["owner_sha256"],
        "Qualitative exclusive owner marker changed",
    )
    for value in (read(lock_path), read(owner_path)):
        require(
            value.get("token") == proof.get("owner_token")
            and value.get("pid") == journal["pid"]
            and value.get("helper_sha256") == QUAL_REPORT_HELPER_SHA
            and all(
                value["process_identity"].get(k) == owner.get(k)
                for k in ("pid", "start_ticks", "argv")
            ),
            "Qualitative exclusive owner token/start/argv changed",
        )
    snapshot = contained(history, "fitted_snapshot_created_now")
    require(
        Path(proof["pre_repair_snapshot_created_now"]).resolve() == snapshot
        and sha(snapshot / "failed_journal.json") == QUAL_FAILED_JOURNAL_SHA,
        "Qualitative created-now failed-journal snapshot changed",
    )
    old = read(snapshot / "failed_journal.json")
    require(
        old.get("pid") == QUAL_OLD_OWNER["pid"]
        and old.get("process_identity", {}).get("start_ticks") == QUAL_OLD_OWNER["start_ticks"]
        and old.get("status") == "failed"
        and old.get("config_sha256") == QUAL_LEGACY_CONFIG_SHA
        and old.get("reason") == proof["old_failure_reason"]
        and len(old.get("rounds", [])) == 1
        and old["rounds"][0].get("seed") == 42
        and old.get("input_files_sha256") == evidence.get("original_input_files_sha256")
        and journal.get("input_files_sha256") == old["input_files_sha256"]
        and journal.get("recovery_history", [])[:-1] == old.get("recovery_history", []),
        "Qualitative old failed owner/source/one-seed prefix was fabricated",
    )
    parent = journal.get("parent_requirement_evidence", {})
    require(
        float(parent.get("productive_wall_seconds", 0)) >= 20 * 3600
        and parent.get("verified_required_seeds") == SEEDS
        and parent.get("parent_owned_processes_released") is True,
        "Qualitative recovery lacks the existing verified parent requirement",
    )
    return dict(
        root=root,
        template=template,
        config=config,
        old_config=old_config,
        evidence=evidence,
        proof=proof,
        history=history,
        snapshot=snapshot,
        old_journal=old,
        durable=Path(old_config["durable_root"]),
    )


def freeze_qualitative_report_recovery_registry_entry(previous_entry):
    """Pure migration supports an already-exited owner and never rewrites a journal."""
    require(
        previous_entry.get("protocol") == "nested5x5_source_qualitative"
        and "qualitative_report_recovery_evidence" not in previous_entry
        and all(
            previous_entry["owned_process_identity"].get(k) == v for k, v in QUAL_OLD_OWNER.items()
        ),
        "Qualitative migration requires the trusted original owner registry entry",
    )
    journal = read(previous_entry["journal_path"])
    owner = journal["process_identity"]
    argv = owner.get("argv", [])
    require(
        "--config" in argv and len(argv) > argv.index("--config") + 1,
        "Qualitative recovery CLI configuration is absent",
    )
    config_path = Path(argv[argv.index("--config") + 1]).resolve()
    config = read(config_path)
    helper = [
        Path(x) for x in argv if Path(x).name == "peptide_modeling_qualitative_report_recovery.py"
    ]
    require(len(helper) == 1, "Qualitative recovery helper command is missing or ambiguous")
    history = contained(previous_entry["output_root"], "recovery_history/" + config["recovery_id"])
    pin = dict(
        schema=QUAL_REPORT_SCHEMA,
        recovery_config_path=str(config_path),
        recovery_config_sha256=sha(config_path),
        helper_path=str(helper[0]),
        helper_sha256=sha(helper[0]),
        diagnosis_sha256=sha(config["evidence_path"]),
        repair_proof_sha256=sha(history / "repair_proof.json"),
        owner_sha256=sha(history / "owner.json"),
        ownership_sha256=sha(
            Path(previous_entry["output_root"]) / "qualitative_report_recovery.lock"
        ),
        previous_owner=previous_entry["owned_process_identity"],
        previous_entry_sha256=hc50_precision_object_sha(previous_entry),
    )
    entry = json.loads(json.dumps(previous_entry))
    entry.update(
        owned_process_identity=owner,
        state_at_freeze=journal["status"],
        journal_sha256_at_freeze=sha(previous_entry["journal_path"]),
        qualitative_report_recovery_evidence=pin,
    )
    qualitative_report_recovery_header(entry, journal)
    return entry


def qualitative_recovery_snapshot(context):
    """Bind the honest CREATED-NOW public archive and internal snapshot to every old byte."""
    evidence, snapshot = context["evidence"], context["snapshot"]
    prefixes = {
        "seed_42": evidence["file_sha256"],
        "template": evidence["original_input_files_sha256"],
    }
    require(
        evidence.get("status") == "verified_readonly_fitted_report_failure"
        and evidence.get("failure")
        == "zero_weight_calibration_bin_inside_valid_component_bootstrap"
        and evidence.get("models") == 20
        and evidence.get("selected_model_contracts") == 15
        and len(evidence.get("signatures", [])) == 15
        and evidence.get("module_sha256") == QUAL_REPORT_LEARNER_SHA
        and evidence.get("training_module_sha256") == CONTROL_TRAINING_SHA,
        "Qualitative original fitted census/learning signature provenance changed",
    )
    for prefix, expected in prefixes.items():
        directory = snapshot / prefix
        actual = {
            p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*") if p.is_file()
        }
        require(
            actual == expected and not any(p.is_symlink() for p in directory.rglob("*")),
            "Qualitative created-now fitted snapshot changed",
        )
    receipt = read(snapshot / "snapshot_receipt.json")
    require(
        receipt.get("schema") == "qualitative_fitted_snapshot_created_now_v1"
        and receipt.get("created_now_not_prior_archive") is True
        and receipt.get("failed_journal_sha256") == QUAL_FAILED_JOURNAL_SHA
        and receipt.get("seed_files") == len(prefixes["seed_42"])
        and receipt.get("template_files") == len(prefixes["template"]),
        "Qualitative snapshot fabricates a prior published archive or its member census",
    )
    for name, expected in prefixes["template"].items():
        require(
            sha(contained(context["template"], name)) == expected,
            "Qualitative frozen source/feature input changed",
        )
    old_seed = context["root"] / "seed_42"
    current = {
        p.relative_to(old_seed).as_posix(): sha(p) for p in old_seed.rglob("*") if p.is_file()
    }
    require(
        set(current) == set(prefixes["seed_42"]) | {"result.json", "verification.json"}
        and all(current.get(k) == v for k, v in prefixes["seed_42"].items()),
        "Qualitative old model/OOF/fold/search bytes changed or unrecorded fitted files appeared",
    )
    stage = Path(context["config"]["evidence_path"]).parent
    archive_path, public_receipt = (
        stage / "pre_repair_fitted_snapshot_created_now.tar.gz",
        read(stage / "pre_repair_snapshot_receipt.json"),
    )
    require(
        sha(archive_path) == QUAL_SNAPSHOT_SHA == public_receipt.get("archive_sha256")
        and public_receipt.get("created_now_not_prior_archive") is True
        and public_receipt.get("source_observed_diagnosis_sha256") == QUAL_DIAGNOSIS_SHA,
        "Qualitative reviewed created-now public snapshot changed or is missing",
    )
    expected = {
        prefix + "/" + name: value
        for prefix, values in prefixes.items()
        for name, value in values.items()
    }
    expected["qualitative_status_failed.json"] = QUAL_FAILED_JOURNAL_SHA
    found = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            require(
                member.isfile() and member.name not in found,
                "Qualitative snapshot has nonregular/duplicate evidence",
            )
            with archive.extractfile(member) as handle:
                value = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    value.update(block)
            if member.name in expected:
                require(
                    value.hexdigest() == expected[member.name],
                    "Qualitative archived original byte changed",
                )
            else:
                require(
                    member.name == "controller_failed.log",
                    "Qualitative snapshot contains an unexpected namespace",
                )
            found.add(member.name)
    require(set(expected) <= found, "Qualitative public snapshot omits original evidence")


def qualitative_recovery_binding(context):
    template, cfg = context["template"], context["old_config"]
    source = table(contained(template, cfg["table_path"]))
    rows = [
        {
            **r,
            "target": r.get("target", r.get("y")),
            "split_group": r.get("split_group", r.get("group_id")),
        }
        for r in source
    ]
    require(
        rows
        and len({r["record_id"] for r in rows}) == len(rows)
        and len({r["activity_id"] for r in rows}) == len(rows),
        "Qualitative source IDs repeat",
    )
    require({float(r["target"]) for r in rows} == {0.0, 1.0}, "Qualitative source classes changed")
    for field in ("structure_key", "study_id"):
        groups = {}
        for r in rows:
            require(
                groups.setdefault(r[field], r["split_group"]) == r["split_group"],
                "Qualitative graph/study crosses source components",
            )
    manifest = read(template / "features/manifest.json")
    cache = contained(
        template,
        "features/" + str(manifest["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1],
    )
    for name, value in manifest["artifact_hashes"].items():
        require(sha(contained(cache, name)) == value, "Qualitative feature bytes changed")
    aliases = {r["record_id"]: r for r in table(contained(cache, manifest["record_index"]))}
    registry = read(contained(template, cfg["registry_path"]))
    methods = registry["methods"]
    require(
        len(methods) == 10
        and len({m["algorithm_family_id"] for m in methods}) == 10
        and all(
            m.get("independent") is True and m.get("source_kind") == "structure" for m in methods
        ),
        "Qualitative chemical methods were omitted or inflated by context",
    )
    groups = {}
    for name in {g for route in cfg["training"]["routes"] for g in manifest["routes"][route]}:
        meta = manifest["groups"][name]
        require(
            meta["key_kind"] == "canonical_identity_group",
            "Qualitative recovery forbids new record/context chemical features",
        )
        indices = {r["key"]: int(r["row_idx"]) for r in table(contained(cache, meta["index"]))}
        header, raw = hc50_precision_npy(contained(cache, meta["matrix"]))
        valid_header, valid = hc50_precision_npy(contained(cache, meta["valid"]))
        require(
            header["shape"] == tuple(meta["shape"])
            and len(header["shape"]) == 2
            and header["shape"][1] == len(meta["feature_names"])
            and valid_header["shape"] == (header["shape"][0],),
            "Qualitative primitive feature dimensions changed",
        )
        width = header["shape"][1]
        matrix = list(
            struct.unpack(
                "<"
                + ("f" if header["descr"] == "<f4" else "B")
                * (len(raw) // (4 if header["descr"] == "<f4" else 1)),
                raw,
            )
        )
        values = []
        for row in rows:
            key = aliases[row["record_id"]]["structure_key"]
            require(
                key == row["structure_key"] and key in indices,
                "Qualitative feature alias changed its whole graph",
            )
            index = indices[key]
            part = [float(x) for x in matrix[index * width : (index + 1) * width]]
            require(
                valid[index]
                and not any(math.isinf(x) for x in part)
                and not all(math.isnan(x) for x in part),
                "Qualitative feature common cohort changed",
            )
            values.append(part)
        groups[name] = values
    routes = {}
    for route in cfg["training"]["routes"]:
        names = [
            g + ":" + v
            for g in manifest["routes"][route]
            for v in manifest["groups"][g]["feature_names"]
        ]
        values = [
            sum((groups[g][i] for g in manifest["routes"][route]), []) for i in range(len(rows))
        ]
        routes[route] = dict(names=names, values=values)
    learner, training = (
        template / "scripts/peptide_modeling_qualitative_cyto.py",
        template / "scripts/peptide_modeling_train.py",
    )
    require(
        sha(learner) == QUAL_REPORT_LEARNER_SHA and sha(training) == CONTROL_TRAINING_SHA,
        "Qualitative learning code changed",
    )
    tree = ast.parse(training.read_text(encoding="utf-8"))
    defaults = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets)
    ]
    require(len(defaults) == 1, "Qualitative training defaults are ambiguous")
    settings = {**ast.literal_eval(defaults[0].value), **cfg["training"]}
    for name in (
        "n_trials",
        "inner_folds",
        "n_jobs",
        "min_rows",
        "min_groups",
        "min_train_rows",
        "seed",
    ):
        settings[name] = int(settings[name])
    return dict(
        rows=rows,
        routes=routes,
        settings=settings,
        manifest=manifest,
        methods=methods,
        plan=read(contained(template, cfg["split_evidence_path"])),
        warning=ast.literal_eval(
            next(
                n.value
                for n in ast.parse(learner.read_text(encoding="utf-8")).body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "WARNING" for t in n.targets)
            )
        ),
    )


def qualitative_recovery_plan(binding, seed):
    rows = binding["rows"]
    evidence = binding["plan"]
    probes = evidence.get("probes", evidence.get("validation_contract", {}).get("probes", []))
    selected = [p for p in probes if p["seed"] == seed]
    require(
        len(selected) == 1 and len(selected[0]["outer_folds"]) == 5,
        "Qualitative reviewed fivefold plan missing",
    )

    def split(values, test_ids):
        require(
            test_ids <= {r["activity_id"] for r in values},
            "Qualitative plan invents source activities",
        )
        train = [i for i, r in enumerate(values) if r["activity_id"] not in test_ids]
        test = [i for i, r in enumerate(values) if r["activity_id"] in test_ids]
        require(train and test, "Qualitative reviewed fold is empty")
        for key in ("split_group", "structure_key", "study_id"):
            require(
                not {values[i][key] for i in train} & {values[i][key] for i in test},
                "Qualitative graph/study/component leakage",
            )
        require(
            all(
                {float(values[i]["target"]) for i in indices} == {0.0, 1.0}
                for indices in (train, test)
            ),
            "Qualitative fold lost a class",
        )
        return train, test

    seen = []
    plan = []
    for fold in selected[0]["outer_folds"]:
        train, test = split(rows, set(fold["test_activity_ids"]))
        train_rows = [rows[i] for i in train]
        require(len(fold["inner_folds"]) == 5, "Qualitative inner plan is not fivefold")
        inner = [split(train_rows, set(f["test_activity_ids"])) for f in fold["inner_folds"]]
        require(
            sorted(i for _, b in inner for i in b) == list(range(len(train))),
            "Qualitative inner rows repeat or disappear",
        )
        plan.append((train, test, inner))
        seen.extend(test)
    require(sorted(seen) == list(range(len(rows))), "Qualitative outer rows repeat or disappear")
    return plan


def qualitative_matrix_fingerprint(binding, route, indices):
    values = []
    for i in indices:
        chemical = [] if route == "context" else binding["routes"][route]["values"][i]
        values.append([*chemical, *[binding["rows"][i][name] for name in QUAL_CONTEXT]])
    digest = hashlib.sha256()
    for row in values:
        for value in row:
            tag, encoded = (
                (b"s", value.encode())
                if isinstance(value, str)
                else (b"f", float(value).hex().encode("ascii"))
            )
            digest.update(tag + len(encoded).to_bytes(8, "big") + encoded)
    return dict(
        schema="typed_feature_matrix_fingerprint_v1",
        shape=[len(indices), len(values[0])],
        dtype="|O",
        data_sha256=digest.hexdigest(),
    )


def qualitative_weighted_metrics(target, prediction, weights):
    require(
        len(target) == len(prediction) == len(weights) > 0
        and all(math.isfinite(v) for v in [*target, *prediction, *weights])
        and set(target) <= {0.0, 1.0}
        and all(0 <= p <= 1 for p in prediction)
        and all(w >= 0 for w in weights),
        "Qualitative metrics contain invalid weights/targets/probabilities",
    )
    total = math.fsum(weights)
    positive = math.fsum(w * y for w, y in zip(weights, target, strict=True))
    negative = total - positive
    require(
        total > 0 and positive > 0 and negative > 0,
        "Qualitative weighted metrics require both effective classes",
    )
    thresholds = defaultdict(lambda: [0.0, 0.0])
    bins = defaultdict(list)
    for y, p, w in zip(target, prediction, weights, strict=True):
        thresholds[p][int(y)] += w
        bins[min(int(p * 10), 9)].append((y, p, w))
    ap, tp, fp = 0.0, 0.0, 0.0
    for _, (neg, pos) in sorted(thresholds.items(), reverse=True):
        tp += pos
        fp += neg
        if tp + fp:
            ap += (pos / positive) * (tp / (tp + fp))
    area, below = 0.0, 0.0
    for _, (neg, pos) in sorted(thresholds.items()):
        area += pos * (below + neg / 2)
        below += neg
    ece = 0.0
    for records in bins.values():
        mass = math.fsum(v[2] for v in records)
        if mass > 0:
            ece += (mass / total) * abs(
                math.fsum(y * w for y, _, w in records) / mass
                - math.fsum(p * w for _, p, w in records) / mass
            )
    return dict(
        pr_auc=ap,
        roc_auc=area / (positive * negative),
        brier=math.fsum(
            w * (p - y) ** 2 for y, p, w in zip(target, prediction, weights, strict=True)
        )
        / total,
        ece_fixed_10_bins=ece,
    )


def qualitative_performance(report, rows, predictions):
    require(
        set(report)
        == {
            "record_metrics",
            "molecule_equal_weight_metrics",
            "component_bootstrap_95_percent",
            "bootstrap_valid_draws",
            "calibration",
        },
        "Qualitative performance schema changed",
    )
    y = [float(r["target"]) for r in rows]
    counts = Counter(r["structure_key"] for r in rows)
    raw = qualitative_weighted_metrics(y, predictions, [1.0] * len(y))
    weighted = qualitative_weighted_metrics(
        y, predictions, [1 / counts[r["structure_key"]] for r in rows]
    )
    record = report["record_metrics"]
    check_metrics(record, {k: v for k, v in raw.items() if k != "ece_fixed_10_bins"})
    tp = sum(a == 1 and p >= 0.5 for a, p in zip(y, predictions, strict=True))
    tn = sum(a == 0 and p < 0.5 for a, p in zip(y, predictions, strict=True))
    fp = sum(a == 0 and p >= 0.5 for a, p in zip(y, predictions, strict=True))
    fn = len(y) - tp - tn - fp
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    check_metrics(
        record, {"mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0, "threshold": 0.5}
    )
    require(
        record.get("calibration") == "uncalibrated_model_probabilities"
        and report.get("calibration") == "diagnostic_only_no_independent_probability_calibration",
        "Qualitative calibration claim changed",
    )
    check_metrics(report["molecule_equal_weight_metrics"], weighted)
    draws = report["bootstrap_valid_draws"]
    intervals = report["component_bootstrap_95_percent"]
    require(
        isinstance(draws, int)
        and not isinstance(draws, bool)
        and 0 < draws <= 300
        and set(intervals) == set(weighted),
        "Qualitative bootstrap policy or valid-draw count changed",
    )
    for key, pair in intervals.items():
        require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)
            and 0 <= pair[0] <= pair[1] <= 1
            and (key != "pr_auc" or pair[0] > 0),
            "Qualitative bootstrap interval invalid",
        )


def qualitative_recovery_seed(entry, round_entry, seed):
    journal = read(entry["journal_path"])
    context = qualitative_report_recovery_header(entry, journal)
    require(
        identity_state(journal["process_identity"], process_identity(journal["pid"]))
        != "same_live_process",
        "Qualitative recovery controller has not released its process",
    )
    qualitative_recovery_snapshot(context)
    binding = qualitative_recovery_binding(context)
    rows = binding["rows"]
    root = context["root"]
    run = contained(root, f"seed_{seed}")
    result_path, verification_path = run / "result.json", run / "verification.json"
    result, verification = read(result_path), read(verification_path)
    require(
        round_entry.get("seed") == seed
        and round_entry.get("status") == "completed"
        and round_entry.get("result_sha256") == sha(result_path)
        and verification == round_entry.get("verification")
        and verification.get("status") == "verified"
        and verification.get("seed") == seed
        and verification.get("model_reload_verified") is True,
        "Qualitative seed result/verification/reload evidence changed",
    )
    require(
        result.get("status") == "completed"
        and result.get("seed") == seed
        and result.get("actual_outer_folds") == result.get("actual_inner_folds_per_outer") == 5
        and result.get("protocol")
        == verification.get("protocol")
        == "source_qualitative_cell_injury_nested_grouped_5x5_v1"
        and verification.get("outer_folds") == verification.get("inner_folds_per_outer") == 5
        and verification.get("graph_study_overlap") == 0,
        "Qualitative source benchmark protocol is not actual nested5x5",
    )
    require(
        verification.get("records") == len(rows) and result.get("warning") == binding["warning"],
        "Qualitative verification population or weak-benchmark warning changed",
    )
    require(
        len(list(run.rglob("*.joblib"))) == 20
        and len(list(run.rglob("*model_contract.json"))) == 15,
        "Qualitative seed has missing or unrecorded selected models/contracts",
    )
    provenance_keys = dict(
        schema=QUAL_REPORT_SCHEMA,
        report_revision=QUAL_REPORT_SCHEMA,
        empty_support_metric_status=QUAL_REPORT_POLICY,
        learning_implementation_sha256=QUAL_REPORT_LEARNER_SHA,
        reporting_implementation_sha256=QUAL_REPORT_HELPER_SHA,
    )
    if seed == 42:
        provenance = result.get("report_recovery", {})
        require(
            all(provenance.get(k) == v for k, v in provenance_keys.items())
            and verification.get("report_recovery") == provenance
            and provenance.get("training_implementation_sha256") == CONTROL_TRAINING_SHA
            and provenance.get("seed42_reused_all_saved_models") is True
            and provenance.get("production_model_fits") == provenance.get("candidate_searches") == 0
            and provenance.get("outer_oof_rewritten") is False
            and provenance.get("fold_progress_rewritten") is False
            and round_entry.get("report_only_recovery") is True
            and context["proof"].get("result_sha256") == sha(result_path),
            "Qualitative seed42 was refitted, reselected or lost its composite report proof",
        )
    else:
        sidecar = read(run / "report_recovery.json")
        require(
            all(sidecar.get(k) == v for k, v in provenance_keys.items())
            and sidecar.get("seed") == seed
            and sidecar.get("original_learning_executed") is True
            and sidecar.get("runtime_override_only") == "_weighted_metrics"
            and sidecar.get("result_sha256") == sha(result_path)
            and sidecar.get("verification_sha256") == sha(verification_path)
            and "report_recovery" not in result
            and "report_recovery" not in verification,
            "Qualitative fresh seed lacks its exact composed-report runtime sidecar",
        )
    require(
        result.get("source_rows_hash") == hc50_precision_object_sha(rows)
        and result.get("feature_fingerprint") == binding["manifest"]["input_fingerprint"]
        and result.get("n_records") == len(rows)
        and result.get("n_graphs") == len({r["structure_key"] for r in rows})
        and result.get("n_groups") == len({r["split_group"] for r in rows})
        and result.get("outer_scores_used_to_choose_arm") is False
        and result.get("deployment_model_fitted") is False
        and result.get("independent_external_validation") is False
        and result.get("context_counted_as_chemical_method") is False
        and result.get("outer_evaluation_includes_unseen_contexts_for_diagnostics") is True
        and result.get("operational_inference_abstains_on_unseen_context") is True,
        "Qualitative source/feature/cohort/weak-benchmark scope changed",
    )
    require(
        set(result.get("coverage", {})) == set(binding["routes"])
        and all(
            result["coverage"][name].get("valid_rows")
            == result["coverage"][name].get("total_rows")
            == len(rows)
            and result["coverage"][name].get("feature_count") == len(value["names"])
            for name, value in binding["routes"].items()
        ),
        "Qualitative common-row/feature coverage changed",
    )
    oof_path = run / "outer_oof.tsv"
    require(sha(oof_path) == result.get("outer_oof_sha256"), "Qualitative old/new OOF changed")
    predictions = actual_folds(oof_path, "outer_fold", "group_id", ("structure_key", "study_id"))
    require(
        [r["record_id"] for r in predictions] == [r["record_id"] for r in rows],
        "Qualitative OOF order/membership changed",
    )
    folds = read(run / "fold_progress.json")
    require(result.get("folds") == folds and len(folds) == 5, "Qualitative fold progress changed")
    plan = qualitative_recovery_plan(binding, seed)
    settings = {**binding["settings"], "seed": seed}
    signatures = []
    checked = 0
    for number, (train, test, inner) in enumerate(plan):
        fold = folds[number]
        require(
            fold.get("fold") == number
            and fold.get("actual_inner_folds") == 5
            and fold.get("train_record_ids") == [rows[i]["record_id"] for i in train]
            and fold.get("test_record_ids") == [rows[i]["record_id"] for i in test],
            "Qualitative reviewed outer membership changed",
        )
        require(
            set(fold.get("arms", {})) == set((*ARMS, "prevalence")),
            "Qualitative outer arm set changed",
        )
        feature_signatures = {
            route: dict(
                chemical_names=value["names"],
                matrix=qualitative_matrix_fingerprint(binding, route, train),
            )
            for route, value in binding["routes"].items()
        }
        for arm in (*ARMS, "prevalence"):
            receipt = fold["arms"][arm]
            model = contained(run, receipt["model_path"])
            require(sha(model) == receipt["model_sha256"], "Qualitative model bytes changed")
            checked += 1
            if arm == "prevalence":
                expected = sum(float(rows[i]["target"]) for i in train) / len(train)
                require(
                    all(
                        math.isclose(
                            float(predictions[i][arm]), expected, rel_tol=1e-9, abs_tol=1e-10
                        )
                        for i in test
                    ),
                    "Qualitative prevalence used non-training labels",
                )
                continue
            path = contained(run, receipt["contract_path"])
            contract = read(path)
            require(sha(path) == receipt["contract_sha256"], "Qualitative model contract changed")
            actual_settings = (
                {
                    **settings,
                    "search_budget": min(
                        int(settings.get("context_search_budget", 64)),
                        int(settings["search_budget"]),
                    ),
                }
                if arm == "context_only"
                else settings
            )
            features = (
                {
                    "context": dict(
                        chemical_names=[],
                        matrix=qualitative_matrix_fingerprint(binding, "context", train),
                    )
                }
                if arm == "context_only"
                else feature_signatures
            )
            signature = hc50_precision_object_sha(
                dict(
                    protocol=result["protocol"],
                    arm=arm,
                    settings=actual_settings,
                    training_rows=[rows[i] for i in train],
                    features=features,
                    inner=[[a, b] for a, b in inner],
                    implementation=QUAL_REPORT_LEARNER_SHA,
                    training_implementation=CONTROL_TRAINING_SHA,
                )
            )
            require(
                contract.get("input_hash") == signature
                and contract.get("training_record_ids") == [rows[i]["record_id"] for i in train]
                and contract.get("outer_labels_used_for_selection") is False
                and contract.get("arm") == arm
                and contract.get("model_sha256") == receipt["model_sha256"]
                and contract.get("search") == receipt.get("search")
                and contract.get("context_is_additional_chemical_method") is False
                and contract.get("context_fields") == list(QUAL_CONTEXT),
                "Qualitative original source/feature/inner-plan/learning signature changed",
            )
            signatures.append(dict(fold=number, arm=arm, input_hash=signature))
            qualitative_recovery_trials(run, contract, actual_settings, signature, arm)
            selected = contract["selection"]
            require(
                receipt.get("selected_route") == selected["route"]
                and receipt.get("selected_model") == selected["model"],
                "Qualitative selected parameters/model metadata differ",
            )
            if arm != "context_only":
                require(
                    contract.get("chemical_feature_names")
                    == binding["routes"][selected["route"]]["names"],
                    "Qualitative selected chemical schema changed",
                )
            else:
                require(
                    contract.get("chemical_feature_names") == [],
                    "Context is not another chemical feature",
                )
            checked += 1
        for i in test:
            require(
                float(predictions[i]["target"]) == float(rows[i]["target"])
                and int(predictions[i]["outer_fold"]) == number
                and predictions[i]["group_id"] == rows[i]["split_group"]
                and all(
                    predictions[i][f] == rows[i][f]
                    for f in ("activity_id", "structure_key", "study_id")
                ),
                "Qualitative OOF target/source/plan changed",
            )
    require(
        len(signatures) == 15 and set(result.get("performance", {})) == set((*ARMS, "prevalence")),
        "Qualitative arms or signatures incomplete",
    )
    if seed == 42:
        expected = {
            (s["fold"], s["arm"]): s["input_hash"] for s in context["evidence"]["signatures"]
        }
        require(
            all(expected.get((s["fold"], s["arm"])) == s["input_hash"] for s in signatures),
            "Qualitative saved seed signature diagnosis changed",
        )
    for arm in (*ARMS, "prevalence"):
        qualitative_performance(
            result["performance"][arm], rows, [float(p[arm]) for p in predictions]
        )
    coverage = round_entry.get("method_model_evidence", {})
    require(
        coverage.get("requirements_met") is True and len(coverage.get("fold_arms", [])) == 15,
        "Qualitative actual method/model search coverage missing",
    )
    require(
        {(x.get("fold"), x.get("arm")) for x in coverage["fold_arms"]}
        == {(f, a) for f in range(5) for a in ARMS},
        "Qualitative method/model coverage repeats or omits fold arms",
    )
    for item in coverage["fold_arms"]:
        search = folds[item["fold"]]["arms"][item["arm"]]["search"]
        pairs = search["coverage_attempted"]
        methods = sorted(
            m["method_id"] for m in binding["methods"] if m["route"] in {p[0] for p in pairs}
        )
        require(
            item.get("attempted_independent_chemical_methods") == methods
            and item.get("attempted_model_families") == sorted({p[1] for p in pairs})
            and item.get("requirements_met") is True
            and len(item["attempted_model_families"]) >= 7
            and (item["arm"] == "context_only" or len(methods) == 10),
            "Qualitative chemical methods/models were inflated",
        )
    return {
        entry["tasks"][0]["task_id"]: dict(
            status="completed_verified",
            artifact_hashes_verified_now=checked,
            actual_outer_folds=5,
            actual_inner_folds=5,
            actual_backend="optuna",
            report_revision=QUAL_REPORT_SCHEMA,
            independently_recomputed_input_signatures=15,
            bootstrap_verification="input_seed_policy_and_provenance_bound_no_draw_trace_claim",
        )
    }


def qualitative_recovery_trials(run, contract, settings, signature, arm):
    search = contract["search"]
    require(
        search.get("backend") == "optuna" and search.get("status") == "completed",
        "Qualitative search backend/status changed",
    )
    namespace = hashlib.sha256(
        json.dumps(
            search["contract"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    require(
        namespace == search.get("namespace")
        and search["contract"].get("input_signature") == signature,
        "Qualitative candidate/cache namespace changed",
    )
    directory = contained(run, search["search_dir"])
    records = []
    for path in sorted((directory / "trials").glob("*.json")):
        record = read(path)
        value = record["evaluation"]
        require(
            record.get("record_sha256")
            == hashlib.sha256(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            and value.get("search_input_signature") == namespace,
            "Qualitative Optuna trial wrapper/input changed",
        )
        if value.get("status") == "completed":
            require(
                value.get("actual_folds") == value.get("requested_folds") == 5
                and math.isfinite(value["score"]),
                "Qualitative successful trial is not fivefold",
            )
        else:
            require(value.get("status") == "failed", "Qualitative trial is nonterminal")
        records.append(value)
    require(
        len(records) == int(settings["search_budget"]) == search.get("budget", {}).get("used"),
        "Qualitative Optuna budget/trial records incomplete",
    )
    require(
        set(map(tuple, search["coverage_attempted"]))
        == {(v["route"], v["model"]) for v in records},
        "Qualitative claimed attempted pairs differ from actual trials",
    )
    selected = contract["selection"]
    require(
        any(
            all(
                v.get(k) == selected.get(k)
                for k in ("route", "model", "params", "score", "search_trial")
            )
            and v.get("status") == "completed"
            for v in records
        ),
        "Qualitative selected parameters have no successful trial",
    )

    def key(value):
        return value["route"], value["model"], hc50_precision_object_sha(value["params"])

    rejected = set()
    for failure in contract.get("rejected_fit_candidates", []):
        candidate = failure["candidate"]
        require(
            failure.get("reason") and any(v == candidate for v in records),
            "Qualitative outer-fit rejection lacks a candidate trace",
        )
        rejected.add(key(candidate))
    admitted = [v for v in records if v["status"] == "completed" and key(v) not in rejected]
    require(
        admitted
        and key(selected) not in rejected
        and selected["score"] == max(v["score"] for v in admitted),
        "Qualitative selection changed from admitted inner-CV candidate ordering",
    )
    database = directory / "study.sqlite3"
    with sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    ) as connection:
        states = dict(
            connection.execute("SELECT state,COUNT(*) FROM trials GROUP BY state").fetchall()
        )
        stored = connection.execute(
            "SELECT t.number,t.state,a.value_json FROM trials t "
            "LEFT JOIN trial_user_attributes a ON a.trial_id=t.trial_id "
            "AND a.key='evaluation_result' ORDER BY t.number"
        ).fetchall()
    by_number = {value["search_trial"]: value for value in records}
    require(
        len(by_number) == len(records)
        and set(by_number) == set(range(len(records)))
        and len(stored) == len(records)
        and all(
            raw is not None
            and json.loads(raw) == by_number.get(number)
            and state == ("COMPLETE" if by_number[number]["status"] == "completed" else "FAIL")
            for number, state, raw in stored
        ),
        "Qualitative SQLite evaluation/selected parameters differ from trial records",
    )
    counts = Counter("COMPLETE" if v["status"] == "completed" else "FAIL" for v in records)
    require(
        states == dict(counts) and sum(states.values()) == len(records),
        "Qualitative SQLite studies differ from saved candidate trials",
    )


def qualitative_report_recovery_final(entry, journal):
    context = qualitative_report_recovery_header(entry, journal)
    require(
        journal.get("status") == "completed"
        and identity_state(journal["process_identity"], process_identity(journal["pid"]))
        != "same_live_process"
        and [r.get("seed") for r in journal.get("rounds", [])] == SEEDS
        and all(r.get("status") == "completed" for r in journal["rounds"]),
        "Qualitative final recovery has missing seeds, failed reports or a live owner",
    )
    summary = read(context["root"] / "performance_summary.json")
    tree = ast.parse(
        (context["template"] / "scripts/peptide_modeling_qualitative_cyto.py").read_text(
            encoding="utf-8"
        )
    )
    warning = ast.literal_eval(
        next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "WARNING" for t in n.targets)
        )
    )
    require(
        summary == read(context["durable"] / "performance_summary.json")
        and summary.get("report_revision") == QUAL_REPORT_SCHEMA
        and summary.get("report_recovery") == context["proof"]
        and summary.get("seeds") == SEEDS
        and summary.get("protocol") == "source_qualitative_cell_injury_nested_grouped_5x5_v1"
        and summary.get("warning") == warning
        and summary.get("arm_selection_from_outer_scores") is False
        and summary.get("independent_external_validation") is False,
        "Qualitative final performance summary/composed provenance changed",
    )
    for seed in SEEDS:
        run = context["root"] / f"seed_{seed}"
        durable = context["durable"] / run.name
        source = {p.relative_to(run).as_posix(): sha(p) for p in run.rglob("*") if p.is_file()}
        copied = {
            p.relative_to(durable).as_posix(): sha(p) for p in durable.rglob("*") if p.is_file()
        }
        require(source == copied and source, "Qualitative durable seed copy is missing or changed")
        require(
            summary.get("scores", {}).get(str(seed)) == read(run / "result.json")["performance"],
            "Qualitative summary scores differ from actual seed result",
        )
    return dict(
        status="verified",
        report_revision=QUAL_REPORT_SCHEMA,
        seed42_fits=0,
        seed42_searches=0,
        new_learning_seeds=[2026, 3407],
        old_models_preserved=True,
        source_weak_benchmark_only=True,
    )


def check_seed(entry, round_entry, seed):
    """Terminal-only evidence adapter. Never infer fitting from controller status."""
    if any(is_vd_absolute_native_task(t) for t in entry["tasks"]):
        require(
            len(entry["tasks"]) == 1
            and entry.get("controller_execution") == "vd_native_bounded_in_process",
            "Vd native custom execution requires its isolated controller",
        )
        task = entry["tasks"][0]
        try:
            adapter, api = vd_absolute_adapter()
            evidence = adapter.verify_seed(entry, round_entry, seed, task, api)
            return {task["task_id"]: {**evidence, "status": "completed_verified"}}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return {task["task_id"]: {"status": "unverified", "reason": str(exc)}}
    root = Path(entry["output_root"])
    run = contained(root, round_entry.get("run_dir", f"seed_{seed}"))
    if not run.is_dir():
        return {
            t["task_id"]: {"status": "unverified", "reason": "Seed directory absent"}
            for t in entry["tasks"]
        }
    protocol = entry["protocol"]
    if protocol == "nested5x5_source_qualitative":
        task = entry["tasks"][0]
        try:
            result_path = run / "result.json"
            result, proof = read(result_path), read(run / "verification.json")
            if (
                qualitative_report_recovery_mode(entry, read(entry["journal_path"]))
                or result.get("report_recovery") is not None
                or proof.get("report_recovery") is not None
                or (run / "report_recovery.json").exists()
            ):
                try:
                    return qualitative_recovery_seed(entry, round_entry, seed)
                except (
                    sqlite3.Error,
                    tarfile.TarError,
                    EOFError,
                    struct.error,
                    SyntaxError,
                    StopIteration,
                ) as exc:
                    raise ValueError("Malformed qualitative recovery artifact evidence") from exc
            require(
                sha(result_path) == round_entry.get("result_sha256"),
                "Qualitative result receipt mismatch",
            )
            require(
                proof == round_entry.get("verification") and proof.get("status") == "verified",
                "Qualitative verification receipt mismatch",
            )
            require(
                result.get("status") == "completed" and result.get("seed") == seed,
                "Qualitative result incomplete/wrong seed",
            )
            require(
                result.get("actual_outer_folds") == result.get("actual_inner_folds_per_outer") == 5,
                "Qualitative result is not nested fivefold",
            )
            require(
                proof.get("outer_folds") == proof.get("inner_folds_per_outer") == 5
                and proof.get("graph_study_overlap") == 0
                and proof.get("model_reload_verified") is True,
                "Qualitative persisted nested verification missing",
            )
            require(
                result.get("protocol")
                == proof.get("protocol")
                == "source_qualitative_cell_injury_nested_grouped_5x5_v1",
                "Qualitative protocol mismatch",
            )
            oof = run / "outer_oof.tsv"
            require(sha(oof) == result.get("outer_oof_sha256"), "Qualitative OOF hash mismatch")
            predictions = actual_folds(oof, "outer_fold", "group_id", ("structure_key", "study_id"))
            source_membership(
                predictions,
                table(Path(entry["template"]) / task["table_relative"]),
                "group_id",
                require_all=True,
            )
            folds = result.get("folds", [])
            require(
                len(folds) == 5 and {f["fold"] for f in folds} == set(range(5)),
                "Missing outer folds",
            )
            checked = 1
            for fold in folds:
                require(
                    fold.get("actual_inner_folds") == 5,
                    "Outer selection did not use five inner folds",
                )
                for arm in (*ARMS, "prevalence"):
                    evidence = fold["arms"][arm]
                    path = contained(run, evidence["model_path"])
                    require(
                        sha(path) == evidence["model_sha256"],
                        "Qualitative fitted model SHA mismatch",
                    )
                    checked += 1
                    if arm != "prevalence":
                        require(
                            evidence.get("actual_inner_folds") == 5
                            and evidence.get("search", {}).get("backend") == "optuna",
                            "Arm lacks actual Optuna fivefold",
                        )
                        path = contained(run, evidence["contract_path"])
                        require(
                            sha(path) == evidence["contract_sha256"],
                            "Qualitative model contract SHA mismatch",
                        )
                        contract = read(path)
                        require(
                            contract.get("search", {}).get("backend") == "optuna"
                            and contract.get("outer_labels_used_for_selection") is False,
                            "Arm selection contract invalid",
                        )
                        require(
                            contract.get("training_record_ids") == fold.get("train_record_ids"),
                            "Arm training membership changed",
                        )
                        checked += 1
            return {
                task["task_id"]: {
                    "status": "completed_verified",
                    "artifact_hashes_verified_now": checked,
                    "actual_outer_folds": 5,
                    "actual_inner_folds": 5,
                    "actual_backend": "optuna",
                }
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {task["task_id"]: {"status": "unverified", "reason": str(exc)}}
    try:
        require(
            sha(run / "task_manifest.json") == entry["manifest_sha256"],
            "Seed source manifest differs from frozen template",
        )
        require(
            sha(run / "config.json") == round_entry.get("config_sha256"),
            "Seed configuration SHA receipt mismatch",
        )
        require(
            read(run / "config.json").get("training", {}).get("seed") == seed,
            "Seed configuration does not match required seed",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {t["task_id"]: {"status": "unverified", "reason": str(exc)} for t in entry["tasks"]}
    verification, verification_error, repaired_report = {}, None, False
    try:
        default_proof = run / (
            "internal_verification.json" if protocol.startswith("internal") else "verification.json"
        )
        proof_path = default_proof
        declared_proof = round_entry.get("audit", {}).get("verification_path")
        if round_entry.get("followup_report_repair") is not None or (
            declared_proof is not None
            and contained(root, declared_proof) != default_proof.resolve()
        ):
            require(
                protocol == "group_holdout_inner5",
                "Only group holdout report repair supports a sidecar",
            )
            try:
                proof_path = verify_group_report_repair(entry, round_entry, seed)
                repaired_report = True
            except (AttributeError, SyntaxError, tarfile.TarError, EOFError) as exc:
                raise ValueError("Malformed report-repair sidecar evidence") from exc
        if protocol == "group_holdout_inner5":
            require(
                repaired_report or round_entry.get("exit_code", 0) == 0,
                "A failed supervisor exit requires an explicit verified report repair",
            )
        verification = read(proof_path)
        require(verification.get("status") == "verified", "Seed artifact verification missing")
        require(
            sha(proof_path) == round_entry.get("audit", {}).get("verification_sha256"),
            "Seed verification SHA receipt mismatch",
        )
        require(
            sha(run / "task_manifest.json") == verification.get("source_task_manifest_sha256"),
            "Seed manifest does not match verification",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        verification_error = str(exc)
    output = {}
    for task in entry["tasks"]:
        branches = []
        try:
            require(
                sha(run / task["table_relative"]) == task["table_sha256"],
                "Seed source task table changed",
            )
            names = (
                [f"results/internal_cv/{task['task_id']}.json"]
                if protocol.startswith("internal")
                else [
                    f"results/full/{task['task_id']}_{scope}.json"
                    for scope in entry["expected_scopes"]
                ]
            )
            for name in names:
                path = contained(run, name)
                result = read(path)
                require(
                    result.get("task_id") == task["task_id"], "Result task differs from registry"
                )
                if result.get("status") in {"deferred", "failed"}:
                    branches.append(
                        {"status": result["status"], "reason": result.get("reason"), "result": name}
                    )
                    continue
                require(verification_error is None, verification_error or "Verification missing")
                if protocol.startswith("internal"):
                    proofs = [
                        p
                        for p in verification.get("completed_results", [])
                        if p.get("task_id") == task["task_id"]
                    ]
                    require(
                        len(proofs) == 1 and proofs[0].get("result_sha256") == sha(path),
                        "Internal task result verification missing",
                    )
                    proof = proofs[0]
                else:
                    proofs = [
                        p
                        for p in verification.get("results", [])
                        if (
                            str(p.get("result_file")).replace("\\", "/")
                            if repaired_report
                            else p.get("result_file")
                        )
                        == name
                    ]
                    require(
                        len(proofs) == 1 and proofs[0].get("artifact_hashes_verified", 0) > 0,
                        "Holdout result not in verified evidence",
                    )
                    require(
                        verification.get("assertions", {}).get("all_checks_passed") is True,
                        "Holdout prediction verification missing",
                    )
                    proof = proofs[0]
                branches.append(
                    {**verify_standard_result(run, result, task, protocol, proof), "result": name}
                )
            states = {b["status"] for b in branches}
            state = (
                "failed"
                if "failed" in states
                else "completed_verified"
                if "completed_verified" in states
                else "deferred"
                if states == {"deferred"}
                else "unverified"
            )
            output[task["task_id"]] = {"status": state, "branches": branches}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            output[task["task_id"]] = {
                "status": "unverified",
                "reason": str(exc),
                "branches": branches,
            }
    return output


def task_state(journal_state, identity, by_seed, required=SEEDS):
    if journal_state in {"waiting_parent", "running"}:
        return (
            ("queued" if journal_state == "waiting_parent" else "running")
            if identity == "same_live_process"
            else "unverified"
        )
    states = [by_seed.get(str(seed), {}).get("status", "unverified") for seed in required]
    if all(s == "completed_verified" for s in states):
        return "trained_three_seeds"
    if "completed_verified" in states:
        return "partial_verified"
    if all(s == "deferred" for s in states):
        return "deferred"
    return "failed" if "failed" in states else "unverified"


def needs_molecular_reporting(entry, task):
    return (
        entry["protocol"] == "internal5_no_holdout"
        and (
            is_renal_native_task(task)
            or task.get("scientific_definition", {}).get("molecular_equal_diagnostics_required")
            is True
        )
    ) or (
        entry["protocol"] == "group_holdout_inner5"
        and (
            is_hc50_primary_task(task)
            or is_mhc10_native_task(task)
            or is_ac50_native_task(task)
            or is_rbc50_registered_task(task)
            or is_vd_absolute_native_task(task)
        )
    )


def is_rbc50_registered_task(task):
    return task.get("task_id") == "TASK_RBC50_REGISTERED_270d2dbf870da342cf2e85e8"


def is_vd_absolute_native_task(task):
    return task.get("task_id") == "TASK_VD_ABSOLUTE_NATIVE_83c52b8c0afb06f712070e93"


def vd_absolute_adapter_payload():
    """SHA-bound stdlib Vd adapter; carried through SSH stdin without remote writes."""
    payload = globals().get("VD_ABSOLUTE_ADAPTER_PAYLOAD")
    if payload is not None:
        source = base64.b64decode(payload["source_base64"])
        require(
            hashlib.sha256(source).hexdigest() == payload["sha256"],
            "Vd audit adapter transport SHA changed",
        )
        return payload
    source = (
        Path(__file__)
        .with_name("peptide_modeling_vd_absolute_native_execution_audit.py")
        .read_bytes()
    )
    return dict(
        source_base64=base64.b64encode(source).decode(), sha256=hashlib.sha256(source).hexdigest()
    )


def vd_absolute_adapter():
    import types

    payload = vd_absolute_adapter_payload()
    cached = globals().get("_VD_ABSOLUTE_ADAPTER")
    if cached is None or cached[0] != payload["sha256"]:
        namespace = {"__name__": "peptide_modeling_vd_absolute_native_execution_audit"}
        exec(
            compile(
                base64.b64decode(payload["source_base64"]),
                "<frozen_stdlib_vd_absolute_audit_adapter>",
                "exec",
            ),
            namespace,
        )
        cached = (payload["sha256"], types.SimpleNamespace(**namespace))
        globals()["_VD_ABSOLUTE_ADAPTER"] = cached
    return cached[1], types.SimpleNamespace(**globals())


def freeze_vd_absolute_native_post_exit_entry(
    launch_contract_path, launch_receipt_path, preflight_path, config_path
):
    """Pure terminal adoption; ordinary live freeze remains unchanged."""
    adapter, api = vd_absolute_adapter()
    return adapter.freeze_post_exit_entry(
        launch_contract_path, launch_receipt_path, preflight_path, config_path, api
    )


def is_renal_native_task(task):
    return task.get("task_id") == "TASK_RENAL_KG_ANNOTATION_b77ceacb6b28f7f632f34c97"


def renal_adapter_payload():
    """Transport the independent stdlib renal gate through the existing SSH stdin lane."""
    payload = globals().get("RENAL_ADAPTER_PAYLOAD")
    if payload is not None:
        source = base64.b64decode(payload["source_base64"])
        require(
            hashlib.sha256(source).hexdigest() == payload["sha256"],
            "Renal audit adapter transport SHA changed",
        )
        return payload
    source = (
        Path(__file__).with_name("peptide_modeling_renal_kg_native_execution_audit.py").read_bytes()
    )
    return dict(
        source_base64=base64.b64encode(source).decode(), sha256=hashlib.sha256(source).hexdigest()
    )


def renal_adapter():
    import types

    payload = renal_adapter_payload()
    cached = globals().get("_RENAL_ADAPTER")
    if cached is None or cached[0] != payload["sha256"]:
        namespace = {"__name__": "peptide_modeling_renal_kg_native_execution_audit"}
        source = base64.b64decode(payload["source_base64"])
        exec(compile(source, "<frozen_stdlib_renal_audit_adapter>", "exec"), namespace)
        cached = (payload["sha256"], types.SimpleNamespace(**namespace))
        globals()["_RENAL_ADAPTER"] = cached
    return cached[1], types.SimpleNamespace(**globals())


def freeze_renal_native_registry_entry(
    previous_entry, recovery_config_path, review_path, implementation_path
):
    """Pure post-exit adoption; preserve original controller/config/journal ownership."""
    adapter, api = renal_adapter()
    return adapter.freeze_registry_entry(
        previous_entry, recovery_config_path, review_path, implementation_path, api
    )


def verify_renal_native_reporting(entry, journal, task):
    adapter, api = renal_adapter()
    try:
        return adapter.verify_reporting(entry, journal, task, api)
    except (
        sqlite3.Error,
        tarfile.TarError,
        EOFError,
        struct.error,
        OverflowError,
        StopIteration,
        AttributeError,
    ) as exc:
        raise ValueError("Malformed renal source/control/recovery evidence") from exc


def verify_renal_native_coverage(entry, journal, task, state):
    adapter, api = renal_adapter()
    try:
        return adapter.coverage(entry, journal, task, state, api)
    except (
        sqlite3.Error,
        tarfile.TarError,
        EOFError,
        struct.error,
        OverflowError,
        StopIteration,
        AttributeError,
    ) as exc:
        raise ValueError("Malformed renal observation coverage evidence") from exc


def rbc50_adapter_payload():
    """Bundle the audited stdlib adapter through SSH stdin, without remote files."""
    payload = globals().get("RBC50_ADAPTER_PAYLOAD")
    if payload is not None:
        source = base64.b64decode(payload["source_base64"])
        require(
            hashlib.sha256(source).hexdigest() == payload["sha256"],
            "RBC audit adapter transport SHA changed",
        )
        return payload
    source = (
        Path(__file__)
        .with_name("peptide_modeling_rbc50_registered_execution_audit.py")
        .read_bytes()
    )
    return dict(
        source_base64=base64.b64encode(source).decode(), sha256=hashlib.sha256(source).hexdigest()
    )


def rbc50_adapter():
    import types

    payload = rbc50_adapter_payload()
    cached = globals().get("_RBC50_ADAPTER")
    if cached is None or cached[0] != payload["sha256"]:
        namespace = {"__name__": "peptide_modeling_rbc50_registered_execution_audit"}
        source = base64.b64decode(payload["source_base64"])
        exec(compile(source, "<frozen_stdlib_rbc50_audit_adapter>", "exec"), namespace)
        cached = (payload["sha256"], types.SimpleNamespace(**namespace))
        globals()["_RBC50_ADAPTER"] = cached
    return cached[1], types.SimpleNamespace(**globals())


def is_ac50_native_task(task):
    definition = task.get("scientific_definition", {})
    return (
        task.get("endpoint") == "AC50"
        and isinstance(definition, dict)
        and definition.get("molecular_equal_diagnostics_required") is True
        and definition.get("molecular_reporting_schema") == AC50_NATIVE_REPORT
    )


def ac50_object_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def ac50_semantic_context(value):
    require(
        isinstance(value, dict) and set(AC50_NATIVE_FIELDS) <= set(value),
        "AC50 semantic context is incomplete",
    )
    result = {}
    for field in AC50_NATIVE_FIELDS:
        item = value[field]
        if isinstance(item, list):
            require(
                item and all(isinstance(v, str) and v for v in item),
                "AC50 semantic list is empty or invalid",
            )
            item = json.dumps(sorted(item), ensure_ascii=False, separators=(",", ":"))
        require(isinstance(item, str) and item, "AC50 missing state must be explicit")
        result[field] = item
    return result


def verify_ac50_source_target_text(frozen, exact_concentration):
    """Verify the fixed source log label; never apply source rendering tolerance to predictions."""
    concentration = Decimal(exact_concentration)
    require(
        concentration.is_finite() and concentration > 0,
        "AC50 original concentration is not positive finite Decimal",
    )
    value = float(concentration)
    require(math.isfinite(value) and value > 0, "AC50 concentration exceeds finite binary64")
    regenerated = format(math.log10(value), ".15g")
    x, y = float(frozen), float(regenerated)
    a, b = Decimal(frozen), Decimal(regenerated)
    require(
        a.is_finite()
        and b.is_finite()
        and math.isfinite(x)
        and math.isfinite(y)
        and format(x, ".15g") == frozen,
        "AC50 target is not canonical finite log10/.15g text",
    )
    if frozen == regenerated:
        return
    require(
        a and b and math.copysign(1.0, x) == math.copysign(1.0, y),
        "AC50 source text portability cannot cross zero or change sign",
    )
    bits = [struct.unpack("!Q", struct.pack("!d", number))[0] for number in (x, y)]
    ulps = abs(bits[0] - bits[1])  # Both numbers have the same sign; ordering distance is exact.
    quantum = Decimal(10) ** (min(a.adjusted(), b.adjusted()) - 14)
    require(
        abs(a - b) <= quantum and ulps <= 128,
        "AC50 frozen log target exceeds one15g decimal quantum or128binary64 ULP",
    )


def verify_ac50_native_source(entry, task):
    """Independently bind all78 native source objects and the exact77 admitted contexts."""
    require(
        is_ac50_native_task(task) and task["task_id"] == AC50_NATIVE_TASK,
        "Unsupported native AC50 task",
    )
    root = Path(entry["template"])
    require(sha(entry["manifest_path"]) == entry["manifest_sha256"], "AC50 manifest changed")
    matched = [t for t in read(entry["manifest_path"])["tasks"] if t["task_id"] == task["task_id"]]
    require(len(matched) == 1, "AC50 source task missing or repeated")
    frozen, definition = matched[0], task["scientific_definition"]
    require(
        frozen.get("scientific_definition") == frozen.get("task_definition") == definition
        and frozen.get("canonical_quantity_id") == AC50_NATIVE_QUANTITY,
        "AC50 frozen scientific contract changed",
    )
    dependencies = {"peptide_modeling_mhc10_native_bridge.py": AC50_NATIVE_HELPER_SHA}
    for filename, expected in {
        "peptide_modeling_ac50_native_bridge.py": AC50_NATIVE_TOOL_SHA,
        **dependencies,
    }.items():
        require(sha(root / "scripts" / filename) == expected, "AC50 pinned implementation changed")
    require(
        definition.get("implementation_sha256") == AC50_NATIVE_TOOL_SHA
        and definition.get("implementation_dependencies_sha256") == dependencies
        and definition.get("canonical_quantity_id") == AC50_NATIVE_QUANTITY
        and definition.get("query_mode") == "source_annotation"
        and definition.get("context_fields") == list(AC50_NATIVE_FIELDS)
        and definition.get("context_group") == AC50_NATIVE_CONTEXT
        and definition.get("context_columns") == 116
        and definition.get("molecular_reporting_record_count") == 77
        and definition.get("diagnostic_controls_precision") == "float64"
        and definition.get("diagnostic_control_precision_policy") == AC50_NATIVE_POLICY
        and definition.get("target_text_portability_policy") == AC50_TARGET_TEXT_POLICY
        and all(
            definition.get(k) is False
            for k in (
                "biological_potency_equivalence_verified",
                "assay_host_species_known",
                "context_is_chemical_method",
                "source_external_validation_available",
            )
        )
        and all(
            definition.get(k) is True
            for k in (
                "separate_all_common_refit_required",
                "PDK1_biology_unresolved_exact_annotation_only",
                "p21_is_induction_marker_not_HDAC_binding",
                "unresolved_NR13_feature_only",
            )
        ),
        "AC50 source annotation was promoted to uniform biological potency or known host",
    )
    scope = dict(
        original_endpoint="AC50",
        target_endpoint="AC50",
        quantity_id=AC50_NATIVE_QUANTITY,
        original_clear_records=78,
        included_core_records=77,
        original_molecular_groups=17,
        core_molecular_groups=17,
        feature_precomputed_records=78,
        excluded_feature_only_ids=[AC50_NATIVE_NR13],
        query_mode="source_annotation",
        biological_potency_equivalence_verified=False,
        assay_host_species_known=False,
        full_refit_required_for_training_coverage=True,
    )
    descriptor = task.get("native_source_observation_coverage", {})
    require(
        descriptor == frozen.get("native_source_observation_coverage")
        and all(descriptor.get(k) == v for k, v in scope.items()),
        "AC50 native source77/78 coverage changed",
    )
    proof_path = contained(root, descriptor["proof_path"])
    require(sha(proof_path) == descriptor["proof_sha256"], "AC50 proof SHA changed")
    proof = read(proof_path)
    require(
        proof.get("schema") == "ac50_native_core77_original78_source_coverage_v1"
        and proof.get("task_id") == AC50_NATIVE_TASK
        and all(proof.get(k) == v for k, v in scope.items())
        and proof.get("implementation_sha256") == AC50_NATIVE_TOOL_SHA
        and proof.get("implementation_dependencies_sha256") == dependencies
        and proof.get("source_delivery_sha256") == AC50_NATIVE_SOURCE_SHA
        and proof.get("literal10000nM_censored_rows_excluded") == 244
        and proof.get("raw_predictions_may_not_be_mutated") is True
        and proof.get("diagnostic_controls_precision") == "float64"
        and proof.get("diagnostic_control_precision_policy") == AC50_NATIVE_POLICY
        and proof.get("target_text_portability_policy") == AC50_TARGET_TEXT_POLICY,
        "AC50 proof scope, source qualifiers or precision policy changed",
    )
    for name, expected in AC50_NATIVE_ARTIFACT_PINS.items():
        item = proof[name]
        require(
            sha(contained(root, item["path"])) == item["sha256"] == expected,
            "AC50 immutable source/context/node-plan artifact changed: " + name,
        )
    require(
        contained(root, proof["task_table"]["path"]) == contained(root, task["table_relative"])
        and task["table_sha256"] == proof["task_table"]["sha256"],
        "AC50 table proof differs",
    )
    source = table(root / task["table_relative"])
    ledger = table(contained(root, proof["all78_ledger"]["path"]))
    require(
        source
        and frozen.get("scientific_warning")
        == definition.get("scientific_warning")
        == proof.get("warning")
        == source[0].get("scope_warning"),
        "AC50 source-annotation scientific warning changed",
    )
    by_id = {r["record_id"]: r for r in ledger}
    require(
        len(ledger) == len(by_id) == 78
        and len(source) == 77
        and source == [r for r in ledger if r["core_training_eligible"] == "true"]
        and proof["record_ids"] == [r["record_id"] for r in source]
        and proof["all78_record_ids"] == list(by_id)
        and [r["record_id"] for r in ledger if r["core_training_eligible"] != "true"]
        == [AC50_NATIVE_NR13],
        "AC50 NR13 feature encoding became training admission",
    )
    snapshots = contained(root, proof["source_snapshot_directory"])
    require(
        sha(snapshots / "delivery_receipt.json") == AC50_NATIVE_SOURCE_SHA,
        "AC50 independent source delivery changed",
    )
    delivery = read(snapshots / "delivery_receipt.json")
    for relative, expected in delivery["artifact_sha256"].items():
        require(
            sha(contained(snapshots, relative)) == expected, "AC50 exact source snapshot changed"
        )
    candidates = read(snapshots / "candidate_context_records.json")
    joined = read(snapshots / "joined_exact_observation_evidence.json")
    candidate_ids = {r["record_id"]: r for r in candidates}
    require(
        len(candidates) == len(candidate_ids) == len(joined) == 78
        and set(candidate_ids) == set(by_id)
        and {r["record"]["record_id"] for r in joined} == set(by_id),
        "AC50 original source object membership changed",
    )
    graph_families = {}
    for item in joined:
        record, obs, assay, compound, reference, identity = [
            item[k]
            for k in (
                "record",
                "native_observation",
                "native_assay",
                "native_compound",
                "native_reference",
                "feature_identity",
            )
        ]
        row = by_id[record["record_id"]]
        candidate = candidate_ids[row["record_id"]]
        context = json.loads(row["semantic_context_json"])
        selected = ac50_semantic_context(context)
        target = {
            k: selected[k]
            for k in (
                "target_semantic_category",
                "native_target_entity_ids",
                "target_role",
                "species",
                "species_semantic_role",
            )
        }
        require(
            json.loads(record["raw_json"]) == obs
            and obs["endpoint"] == record["endpoint"] == row["native_metric_annotation"] == "AC50"
            and obs["qualifier"] == record["raw_relation"] == row["raw_relation"] == "="
            and obs["unit"] == row["raw_unit"] == "nM"
            and row["unit"] == "uM"
            and Decimal(obs["value"]) / 1000
            == Decimal(row["value"])
            == Decimal(record["value"])
            == Decimal(candidate["value_uM"])
            and row["raw_value"] == obs["value"]
            and obs["original_value"] == row["original_scalar"] == candidate["original_scalar"]
            and obs["original_unit"] == row["original_unit"] == candidate["original_unit"]
            and obs["compound_id"] == compound["compound_id"]
            and obs["assay_id"] == assay["assay_id"] == row["source_assay_id"]
            and obs["study_id"] == assay["study_id"] == row["source_study_id"]
            and obs["reference_id"] == reference["reference_id"] == row["source_reference_id"]
            and row["source_pmid"] == reference["pmid"] == candidate["pmid"]
            and record["smiles"] == compound["smiles"]
            and row["canonical_identity_group"]
            == identity["canonical_identity_group"]
            == candidate["complete_graph"]
            and row["split_group"] == identity["split_group"] == candidate["family"]
            and context == candidate["semantic_context"]
            and context["raw_assay_description_provenance_only"] == assay["description"]
            and selected["native_assay_type"] == assay["assay_type"]
            and row["semantic_context_key"] == ac50_object_hash(selected)
            and row["semantic_target_key"] == ac50_object_hash(target)
            and row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
            and graph_families.setdefault(row["canonical_identity_group"], row["split_group"])
            == row["split_group"]
            and row["source_annotation_only"] == "true"
            and row["uniform_biological_potency_verified"] == "false"
            and row["primary_curve_fit_verified"] == "false"
            and row["assay_host_species"] == "not_reported"
            and selected["assay_cell_or_expression_host_species"] == "not_reported"
            and selected["species"] == "Homo sapiens"
            and selected["species_semantic_role"]
            == "source_target_or_system_organism_annotation_not_assay_host_or_donor_species",
            "AC50 native source scalar, target role, host, complete graph or linkage changed",
        )
        verify_ac50_source_target_text(row["target"], row["value"])
        factor = {"nM": Decimal("0.001"), "uM": Decimal(1)}.get(obs["original_unit"])
        require(
            factor is not None and Decimal(obs["original_value"]) * factor == Decimal(row["value"]),
            "AC50 original-unit Decimal conversion changed",
        )
        links = item["same_activity_binding_target_rows"]
        for link in links:
            require(
                all(
                    link[k] == obs[k]
                    for k in (
                        "compound_id",
                        "substance_id",
                        "assay_id",
                        "reference_id",
                        "source_entry_id",
                    )
                )
                and link["measurement_type"] == "AC50"
                and link["relation"] == "="
                and link["unit"] == "nM"
                and Decimal(link["value_num"]) == Decimal(obs["value"]),
                "AC50 target binding used a different activity or qualifier",
            )
        linked = {r["target_entity_id"] for r in links}
        require(
            sorted(linked) == sorted(context["native_target_entity_ids"])
            if linked
            else context["native_target_entity_ids"] == ["not_reported"],
            "AC50 native target entity linkage changed",
        )
        require(
            {r["bio_entity_id"] for r in item["same_activity_target_entities"]} == linked,
            "AC50 target entity was remapped",
        )
    require(
        len(graph_families) == len(set(graph_families.values())) == 17,
        "AC50 whole-graph/known-FAMILY components changed",
    )
    ontology = read(contained(root, proof["ontology"]["path"]))
    contexts = [ac50_semantic_context(r["semantic_context"]) for r in candidates]
    categories = {k: sorted({r[k] for r in contexts}) for k in AC50_NATIVE_FIELDS}
    require(
        ontology
        == dict(
            schema="fixed_source_semantic_AC50_ontology_v1",
            fields=list(AC50_NATIVE_FIELDS),
            categories=categories,
            feature_names=[k + "=" + v for k in AC50_NATIVE_FIELDS for v in categories[k]],
            dimensions=116,
            construction="unlabelled_source_panel78_only_no_outcome_statistics",
            identifiers_and_outcomes_are_features=False,
            permitted_entity_ID_role="target_concept_annotation_only_not_activity_or_assay_ID",
        ),
        "AC50 ontology learned outcomes or included prohibited identifiers",
    )
    checks = read(contained(root, proof["predeclared_splits"]["path"]))
    require(
        checks.get("protocol") == "group_holdout_inner5_then_separate_all_common_refit"
        and checks.get("refit_full_required") is True
        and checks.get("final_training_records") == 77
        and checks.get("pretraining_runtime", {}).get("host") == "node23"
        and checks["pretraining_runtime"].get("production_fits_before_plan") == 0
        and checks["pretraining_runtime"].get("outer_memberships_changed") is False
        and len(checks["seeds"]) == 3
        and {c["seed"] for c in checks["seeds"]} == set(SEEDS),
        "AC50 actual node plan was not frozen before fits",
    )
    core_ids = [r["record_id"] for r in source]
    for check in checks["seeds"]:
        train, held = check["train_record_ids"], check["holdout_record_ids"]

        def groups(ids):
            return {by_id[i]["split_group"] for i in ids}

        validation = [i for fold in check["folds"] for i in fold["validation_record_ids"]]
        require(
            len(train) == len(set(train))
            and len(held) == len(set(held))
            and set(train).isdisjoint(held)
            and set(train) | set(held) == set(core_ids)
            and groups(train).isdisjoint(groups(held))
            and len(groups(train)) == 13
            and len(groups(held)) == 4
            and check["final_refit_record_ids"] == core_ids
            and check["actual_folds"] == 5
            and len(check["folds"]) == 5
            and {f["fold"] for f in check["folds"]} == set(range(5))
            and len(validation) == len(set(validation)) == len(train)
            and set(validation) == set(train),
            "AC50 outer or inner5 membership changed",
        )
        for fold in check["folds"]:
            require(
                set(fold["train_record_ids"]) == set(train) - set(fold["validation_record_ids"])
                and groups(fold["train_record_ids"]).isdisjoint(
                    groups(fold["validation_record_ids"])
                ),
                "AC50 molecule-family fold leakage",
            )
        require(
            sha(contained(root, check["split_path"])) == check["split_sha256"],
            "AC50 outer split artifact changed",
        )
    verify_ac50_native_features(root, ledger, ontology, entry.get("native_feature_contract"))
    return descriptor, proof, frozen, source, checks


def verify_ac50_native_features(root, ledger, ontology, frozen_pin=None):
    root = Path(root)
    pointer, registry_path = root / "features/manifest.json", root / "representation_methods.json"
    if frozen_pin is not None:
        require(
            sha(pointer) == frozen_pin["sha256"]
            and sha(registry_path) == frozen_pin["registry_sha256"],
            "AC50 frozen feature pointer changed",
        )
    features = read(pointer)
    require(
        features.get("schema") == AC50_NATIVE_SCHEMA
        and features.get("requested_records") == 78
        and features.get("training_records") == 77
        and features.get("feature_only_records") == [AC50_NATIVE_NR13]
        and features.get("structure_keys") == 17
        and features.get("input_fingerprint")
        == "27618f592dfb6bf87f338caffa892641505a8240364eb74fc76690fdffd6e75d"
        and ac50_object_hash(features["configuration"]) == features["input_fingerprint"],
        "AC50 exact78 feature context fingerprint changed",
    )
    for key, expected in {
        "artifact_hashes": "a624f22e210d47ba612f7e1da3cca49c546091e5ae0f159f77eaee417703d571",
        "representation_methods": (
            "691dbc248df5184f39cf61fd02ca819e5212f43cf2d15d6b4ee753a807361711"
        ),
        "groups": "06371a65d4ced2cdd50c6146175a6572c9ebe77ad1bb55c5e92c1fe05c3f9f73",
        "routes": "8efee842c23cb2c7781097ab9e625770cf3b05173257936381eb02e6639e932a",
    }.items():
        require(ac50_object_hash(features[key]) == expected, "AC50 bank/method/schema pin changed")
    require(
        read(registry_path) == features["representation_methods"], "AC50 method registry changed"
    )
    cache_name = str(features["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    require(
        cache_name == features["input_fingerprint"][:16], "AC50 feature cache namespace changed"
    )
    cache = contained(root, "features/" + cache_name)
    for relative, expected in features["artifact_hashes"].items():
        require(sha(contained(cache, relative)) == expected, "AC50 bank artifact changed")
    alias_rows = table(cache / features["record_index"])
    require(
        len(alias_rows) == 78
        and len({r["record_id"] for r in alias_rows}) == 78
        and {r["record_id"]: r["structure_key"] for r in alias_rows}
        == {r["record_id"]: r["canonical_identity_group"] for r in ledger},
        "AC50 fullgraph feature aliases omitted NR13 or invented identities",
    )
    meta = features["groups"][AC50_NATIVE_CONTEXT]
    index_rows = table(cache / meta["index"])
    require(
        index_rows == [dict(key=r["record_id"], row_idx=str(i)) for i, r in enumerate(ledger)],
        "AC50 record-keyed context order changed",
    )
    header, matrix = hc50_precision_npy(cache / meta["matrix"])
    valid_header, valid = hc50_precision_npy(cache / meta["valid"])
    encoded = b"".join(
        struct.pack(
            "<116f",
            *[
                float(
                    ac50_semantic_context(json.loads(row["semantic_context_json"]))[field]
                    == category
                )
                for field in AC50_NATIVE_FIELDS
                for category in ontology["categories"][field]
            ],
        )
        for row in ledger
    )
    require(
        header["shape"] == (78, 116)
        and header["descr"] == "<f4"
        and matrix == encoded
        and valid_header["shape"] == (78,)
        and valid == b"\x01" * 78,
        "AC50 context matrix is not exactly the unlabeled16-field116-channel ontology",
    )
    return features


def ac50_native_designs(root, source, route):
    """Reconstruct original primitive f32 design bytes, then exact f64 upcasts; never fit."""
    root = Path(root)
    features = read(root / "features/manifest.json")
    cache = root / "features" / features["input_fingerprint"][:16]
    require(route in features["routes"], "AC50 selected route is unsupported")
    groups = features["routes"][route]
    require(
        groups.count(AC50_NATIVE_CONTEXT) == 1 and groups[-1] == AC50_NATIVE_CONTEXT,
        "AC50 selected design must contain semantic context exactly once",
    )
    names, parts = [], []
    for name in groups:
        meta = features["groups"][name]
        indices = table(cache / meta["index"])
        index = {r["key"]: int(r["row_idx"]) for r in indices}
        require(
            len(indices) == len(index) and set(index.values()) == set(range(len(index))),
            "AC50 feature bank index repeats or omits rows",
        )
        header, raw = hc50_precision_npy(cache / meta["matrix"])
        mask_header, mask = hc50_precision_npy(cache / meta["valid"])
        width = len(meta["feature_names"])
        require(
            header["shape"] == (len(index), width) == tuple(meta["shape"])
            and mask_header["shape"] == (len(index),)
            and mask_header["descr"] == "|b1",
            "AC50 primitive feature shape changed",
        )
        byte_width = width * (4 if header["descr"] == "<f4" else 1)
        rows = []
        for row in source:
            key = (
                row["record_id"]
                if meta["key_kind"] == "record_id"
                else row["canonical_identity_group"]
            )
            pos = index[key]
            require(mask[pos] == 1, "AC50 admitted input has invalid features")
            data = raw[pos * byte_width : (pos + 1) * byte_width]
            if header["descr"] == "|u1":
                data = struct.pack("<" + "f" * width, *data)
            require(len(data) == 4 * width, "AC50 f32 row width changed")
            values = struct.unpack("<" + "f" * width, data)
            missing = sum(math.isnan(v) for v in values)
            require(
                not any(math.isinf(v) for v in values)
                and missing < width
                and (
                    not missing
                    or meta.get("allow_partial_missing") is True
                    and missing / width <= meta.get("max_missing_fraction", 0)
                ),
                "AC50 feature missingness policy changed",
            )
            rows.append(data)
        parts.append(rows)
        names += [name + ":" + str(n) for n in meta["feature_names"]]
    assembled = [b"".join(part[i] for part in parts) for i in range(len(source))]
    width = len(names)
    designs = {}
    for name, selected_names, values in (
        ("molecule_only", names[:-116], [row[:-464] for row in assembled]),
        ("context_only", names[-116:], [row[-464:] for row in assembled]),
        ("molecule_plus_context", names, assembled),
    ):
        require(selected_names, "AC50 chemical or context design is empty")
        n = len(selected_names)
        upcast = [struct.pack("<" + "d" * n, *struct.unpack("<" + "f" * n, row)) for row in values]
        designs[name] = dict(
            names=selected_names,
            width=n,
            rows={r["record_id"]: value for r, value in zip(source, values, strict=True)},
            float64_rows={r["record_id"]: value for r, value in zip(source, upcast, strict=True)},
            matrix_sha256=hashlib.sha256(b"".join(values)).hexdigest(),
            float64_matrix_sha256=hashlib.sha256(b"".join(upcast)).hexdigest(),
            mask_sha256=hashlib.sha256(b"\x01" * len(source)).hexdigest(),
        )
    require(width == designs["molecule_plus_context"]["width"], "AC50 assembled schema differs")
    return designs


def ac50_native_support(row, fitted):
    target = row["semantic_target_key"] in {r["semantic_target_key"] for r in fitted}
    readout = row["native_readout_family"] in {r["native_readout_family"] for r in fitted}
    joint = row["semantic_context_key"] in {r["semantic_context_key"] for r in fitted}
    return dict(
        target_annotation_seen_in_training=target,
        target_category_seen_in_training=row["target_semantic_category"]
        in {r["target_semantic_category"] for r in fitted},
        readout_seen_in_training=readout,
        joint_context_seen_in_training=joint,
        coarse_target_and_readout_supported=target and readout,
        qualified_joint_query_supported=joint and target and readout,
        source_paper_shared_with_training=row["source_pmid"] in {r["source_pmid"] for r in fitted},
    )


def ac50_native_consistency(values, source, design, context, *, float64=False):
    ids = [r["record_id"] for r in values]
    p = [float(r["prediction"]) for r in values]
    row_bytes = design["float64_rows" if float64 else "rows"]
    require(
        ids
        and len(ids) == len(set(ids))
        and all(math.isfinite(v) for v in p)
        and len({source[i]["canonical_identity_group"] for i in ids}) == 1
        and len({source[i]["split_group"] for i in ids}) == 1
        and len({source[i]["semantic_context_key"] for i in ids}) == 1
        and len({row_bytes[i] for i in ids}) == 1
        and len({(r["evaluation_role"], int(r["fold"])) for r in values}) == 1,
        "AC50 precision comparison lacks exact input/graph/context/fold identity",
    )
    original = all(abs(value - p[0]) <= 1e-8 + 1e-7 * abs(p[0]) for value in p)
    shared = dict(
        full_selected_row_sha256=hashlib.sha256(row_bytes[ids[0]]).hexdigest(),
        raw_predictions_unchanged=True,
        source_record_ids=ids,
    )
    if float64:
        require(original, "AC50 float64 controls fail the unchanged strict identity tolerance")
        return dict(
            policy="same_full_input_graph_context_fold_float64_original_tolerance_v2",
            mode="original_float_tolerance",
            diagnostic_precision="float64",
            binary32_exception_allowed=False,
            **shared,
        )
    ulps = None
    try:
        bits = [hc50_precision_binary32(value) for value in p]
        ulps = max(bits) - min(bits)
    except ValueError:
        require(original, "AC50 precision exception needs exact binary32 predictions")
    mode = "original_float_tolerance"
    if not original:
        policy = context.get("linear_solver_policy", {})
        require(
            context.get("selected_model") == "linear"
            and policy.get("version") == "dimension_based_linear_policy_v2"
            and policy.get("solver") == "lsqr"
            and policy.get("selection_basis") == "raw_route_feature_count_only"
            and policy.get("scaler_with_mean") is False
            and policy.get("feature_count") == design["width"]
            and all(abs(value) >= 2.0**-126 for value in p)
            and len({math.copysign(1.0, value) for value in p}) == 1
            and ulps is not None
            and ulps <= 32,
            "AC50 selected f32 drift lacks the bounded32ULP LSQR identity proof",
        )
        mode = "exact_binary32_LSQR_at_most32ULP"
    return dict(
        policy="same_full_input_graph_context_fold_binary32_LSQR_32ULP_v1",
        mode=mode,
        observed_binary32_ulp_distance=ulps,
        engineering_budget_not_mathematical_BLAS_bound=True,
        maximum_binary32_ulp_budget=32,
        **shared,
    )


def verify_ac50_native_evaluation(
    reported, source, inner, held, check, design, context, *, float64=False
):
    require(
        isinstance(reported, dict)
        and reported.get("quantity_id") == AC50_NATIVE_QUANTITY
        and reported.get("original_clear_records") == 78
        and reported.get("core_records") == 77
        and reported.get("feature_only_ids") == [AC50_NATIVE_NR13]
        and reported.get("warning") == source[0]["scope_warning"]
        and all(
            reported.get(k) is False
            for k in (
                "model_selection_repeated",
                "independently_evaluated_all77_refit",
                "source_external_validation_claimed",
                "biological_potency_equivalence_claimed",
            )
        ),
        "AC50 diagnostic scope promoted source annotations or scored the all77 refit",
    )
    by_id = {r["record_id"]: r for r in source}
    ids = [r["record_id"] for r in inner + held]
    require(
        len(ids) == len(set(ids)) == 77 and set(ids) == set(by_id),
        "AC50 exact77 evaluation predictions missing or repeated",
    )
    folds = {f["fold"]: f for f in check["folds"]}
    fold_by_id = {i: f["fold"] for f in check["folds"] for i in f["validation_record_ids"]}
    require(
        set(reported.get("evaluation_by_role", {})) == {"inner_oof", "outer_holdout"},
        "AC50 diagnostic evaluation roles missing or added",
    )
    all_enriched = []
    for role, predicted, expected in (
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", held, check["holdout_record_ids"]),
    ):
        require(
            [r["record_id"] for r in predicted] == expected,
            "AC50 saved prediction role/order differs from the actual node plan",
        )
        enriched, comparisons = [], defaultdict(list)
        for value in predicted:
            row, fold = by_id[value["record_id"]], int(value["fold"])
            require(
                value["split_group"] == row["split_group"]
                and math.isclose(
                    float(value["target"]), float(row["target"]), rel_tol=0, abs_tol=1e-10
                )
                and math.isfinite(float(value["prediction"]))
                and fold == (fold_by_id[row["record_id"]] if role == "inner_oof" else -1),
                "AC50 OOF/holdout source target, graph group or actual fold changed",
            )
            fitted = [
                by_id[i]
                for i in (
                    folds[fold]["train_record_ids"]
                    if role == "inner_oof"
                    else check["train_record_ids"]
                )
            ]
            require(
                row["split_group"] not in {r["split_group"] for r in fitted},
                "AC50 evaluated molecule entered its own training role",
            )
            enriched_row = {
                **value,
                "evaluation_role": role,
                **{
                    k: row[k]
                    for k in (
                        "native_readout_family",
                        "measured_direction",
                        "target_semantic_category",
                        "target_role",
                        "semantic_target_key",
                        "semantic_context_key",
                        "source_pmid",
                        "annotated_organism_role",
                    )
                },
                "assay_host_species": "not_reported",
                **ac50_native_support(row, fitted),
            }
            enriched.append(enriched_row)
            comparisons[
                (row["canonical_identity_group"], row["semantic_context_key"], fold)
            ].append(enriched_row)
        actual = reported["evaluation_by_role"][role]
        if float64:
            identical_inputs = defaultdict(list)
            for row in enriched:
                identical_inputs[
                    (int(row["fold"]), design["float64_rows"][row["record_id"]])
                ].append(float(row["prediction"]))
            require(
                all(
                    all(abs(p - values[0]) <= 1e-8 + 1e-7 * abs(values[0]) for p in values)
                    for values in identical_inputs.values()
                ),
                "AC50 float64 controls differ for identical full design/role/fold inputs",
            )
        saved_predictions = actual.get("predictions", [])
        require(
            len(saved_predictions) == len(enriched), "AC50 enriched prediction membership differs"
        )
        for saved, row in zip(saved_predictions, enriched, strict=True):
            require(set(saved) == set(row), "AC50 enriched prediction schema changed")
            for key, value in row.items():
                require(
                    (float(saved[key]) == float(value))
                    if key in {"fold", "target", "prediction"}
                    else saved[key] == value,
                    "AC50 source role or derived support flag changed",
                )
        check_mhc10_native_weighted(actual, enriched)
        require(
            actual.get("score_role")
            == ("model_selection" if role == "inner_oof" else "outer_train_model_evaluation")
            and actual.get("score_applies_to_all77_refit") is False,
            "AC50 role score assigned to all77 refit",
        )
        fields = (
            "native_readout_family",
            "measured_direction",
            "target_semantic_category",
            "target_role",
            "semantic_context_key",
            "source_pmid",
        )
        strata = actual.get("strata", {})
        require(set(strata) == set(fields), "AC50 readout/target/joint/source strata missing")
        for field in fields:
            categories = {r[field] for r in enriched}
            require(set(strata[field]) == categories, "AC50 stratum membership changed")
            for category in categories:
                check_mhc10_native_weighted(
                    strata[field][category], [r for r in enriched if r[field] == category]
                )
        flags = {
            "unseen_target_annotation": ("target_annotation_seen_in_training", False),
            "unseen_target_category": ("target_category_seen_in_training", False),
            "unseen_readout": ("readout_seen_in_training", False),
            "unseen_joint_context": ("joint_context_seen_in_training", False),
            "qualified_joint_supported": ("qualified_joint_query_supported", True),
            "coarse_target_readout_supported": ("coarse_target_and_readout_supported", True),
            "shared_source_paper": ("source_paper_shared_with_training", True),
        }
        require(
            actual.get("support_counts")
            == {
                key: sum(r[field] is truth for r in enriched)
                for key, (field, truth) in flags.items()
            },
            "AC50 context counts do not follow actual role-specific fitted members",
        )
        require(
            actual.get("prediction_consistency")
            == [
                ac50_native_consistency(values, by_id, design, context, float64=float64)
                for values in comparisons.values()
            ],
            "AC50 consistency proof differs from exact source inputs/precision/fold",
        )
        all_enriched += enriched
    singleton_ids = [AC50_NATIVE_PDK1, AC50_NATIVE_P21] + [
        r["record_id"] for r in source if r["native_readout_family"] == "agonist_activity"
    ]
    require(
        reported.get("singleton_context_diagnostics")
        == {i: next(r for r in all_enriched if r["record_id"] == i) for i in singleton_ids},
        "AC50 unresolved PDK1/p21/singleton query diagnostics changed",
    )


def verify_ac50_native_refit(run, result, task, source, frozen):
    paths, _ = verify_hashes(
        run,
        result,
        {
            "model",
            "refit_model",
            "oof",
            "holdout",
            "outer_split",
            "input_contract",
            "model_contract",
        },
    )
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    attempts = result.get("refit_attempts", [])
    require(
        result.get("status") == "completed"
        and result.get("scope_label") == "common"
        and result.get("refit_status") == "completed"
        and result.get("deployment_ready") is True
        and paths["model"] != paths["refit_model"]
        and attempts
        and attempts[-1].get("status") == "completed"
        and contract.get("refit_attempts") == attempts
        and contract.get("deployment_ready") is True
        and contract.get("task") == inputs.get("task") == frozen
        and contract.get("common_record_ids") == [r["record_id"] for r in source]
        and contract.get("optional_refit_fit_scope") == "all_common_after_evaluation"
        and contract.get("model_fit_scope") == "outer_train_only"
        and contract.get("holdout_not_used_for_selection") is True
        and contract.get("refit_status") == "completed"
        and contract.get("target_unit") == "uM"
        and contract.get("target_transform") == "log10"
        and contract.get("selected_route") == result.get("selected_route")
        and contract.get("selected_model") == result.get("selected_model"),
        "AC50 distinct all77 final refit lacks exact training membership or ready contract",
    )
    require(
        inputs["config"].get("refit_full") is True
        and inputs["config"].get("inner_folds") == 5
        and inputs["config"].get("search_backend") == "optuna"
        and inputs.get("implementation_sha256")
        == sha(Path(run) / "scripts/peptide_modeling_train.py")
        == CONTROL_TRAINING_SHA
        and inputs.get("input_hash") == result.get("input_hash") == contract.get("input_hash"),
        "AC50 training implementation/5fold Optuna/refit input provenance changed",
    )
    outer = read(paths["outer_split"])
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=frozen,
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=inputs["config"],
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=outer["input_hash"],
    )
    require(
        hashlib.sha256(
            json.dumps(signature, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()
        == result["input_hash"],
        "AC50 signed input changed",
    )
    designs = ac50_native_designs(run, source, result["selected_route"])
    design, recorded = (
        designs["molecule_plus_context"],
        inputs["feature_hashes"][result["selected_route"]],
    )
    require(
        recorded.get("names") == contract.get("feature_names") == design["names"]
        and recorded.get("matrix_sha256") == design["matrix_sha256"]
        and recorded.get("mask_sha256") == design["mask_sha256"],
        "AC50 exact stored f32 chemical/context input differs from signed training bytes",
    )
    if result["selected_model"] == "linear":
        require(
            contract.get("linear_solver_policy")
            == inputs["linear_solver_policies"][result["selected_route"]],
            "AC50 selected linear solver differs from signed training policy",
        )
    return paths, contract, inputs, designs


def verify_ac50_native_reporting(entry, journal, task):
    descriptor, proof, frozen, source, checks = verify_ac50_native_source(entry, task)
    require(
        identity_state(
            entry["owned_process_identity"],
            process_identity(entry["owned_process_identity"]["pid"]),
        )
        != "same_live_process",
        "AC50 reporting owner has not released",
    )
    receipt = journal.get("molecular_reporting", {})
    require(
        isinstance(receipt, dict)
        and receipt.get("status") == "verified"
        and receipt.get("implementation_sha256") == AC50_NATIVE_TOOL_SHA
        and receipt.get("implementation_dependencies_sha256")
        == {"peptide_modeling_mhc10_native_bridge.py": AC50_NATIVE_HELPER_SHA},
        "AC50 mandatory report implementation receipt missing or changed",
    )
    path = contained(entry["output_root"], receipt["path"])
    require(
        path.name == "ac50_native_molecular_diagnostics.json"
        and sha(path) == receipt.get("sha256"),
        "AC50 mandatory report path or SHA changed",
    )
    payload = read(path)
    require(isinstance(payload, dict), "AC50 mandatory report is not an object")
    require(
        payload.get("schema") == AC50_NATIVE_REPORT
        and payload.get("status") == "verified"
        and payload.get("task_id") == AC50_NATIVE_TASK
        and payload.get("quantity_id") == AC50_NATIVE_QUANTITY
        and payload.get("required_seeds") == SEEDS
        and payload.get("original_clear_records") == 78
        and payload.get("core_training_records") == 77
        and payload.get("original_molecular_groups") == 17
        and payload.get("feature_only_ids") == [AC50_NATIVE_NR13]
        and payload.get("warning") == frozen["scientific_warning"]
        and payload.get("raw_predictions_unchanged") is True
        and payload.get("diagnostic_controls_precision") == "float64"
        and payload.get("diagnostic_control_precision_policy") == AC50_NATIVE_POLICY
        and all(
            payload.get(k) is False
            for k in (
                "biological_potency_equivalence_claimed",
                "source_external_validation_claimed",
                "model_selection_repeated",
            )
        ),
        "AC50 report scope or native source annotation77/78 limits changed",
    )
    reports = payload.get("reports", [])
    require(
        isinstance(reports, list)
        and len(reports) == 3
        and all(isinstance(r, dict) for r in reports)
        and {r["seed"] for r in reports} == set(SEEDS)
        and len(journal["rounds"]) == 3
        and {r["seed"] for r in journal["rounds"]} == set(SEEDS),
        "AC50 three required seed reports missing or repeated",
    )
    rounds = {r["seed"]: r for r in journal["rounds"]}
    for report in reports:
        round_entry = rounds[report["seed"]]
        require(
            round_entry.get("status") == "completed"
            and round_entry.get("audit", {}).get("method_model_coverage_met") is True
            and not round_entry["audit"].get("missing_completed_task_ids"),
            "AC50 required seed lacks completed task/method/model evidence",
        )
        run = contained(entry["output_root"], round_entry["run_dir"])
        verification_path = run / "verification.json"
        require(
            sha(verification_path) == round_entry["audit"].get("verification_sha256")
            and read(verification_path).get("status") == "verified"
            and read(verification_path).get("source_task_manifest_sha256")
            == sha(run / "task_manifest.json"),
            "AC50 saved seed verification does not bind the actual source manifest",
        )
        for name, expected in AC50_NATIVE_ARTIFACT_PINS.items():
            require(
                sha(contained(run, proof[name]["path"])) == expected,
                "AC50 seed source/context/node-plan artifact changed: " + name,
            )
        require(
            sha(run / task["table_relative"]) == task["table_sha256"]
            and table(run / task["table_relative"]) == source
            and read(run / "task_manifest.json")["tasks"] == [frozen],
            "AC50 seed source/manifest changed",
        )
        verify_ac50_native_features(
            run,
            table(contained(run, proof["all78_ledger"]["path"])),
            read(contained(run, proof["ontology"]["path"])),
        )
        require(
            sha(contained(run, descriptor["proof_path"])) == descriptor["proof_sha256"]
            and sha(contained(run, proof["predeclared_splits"]["path"])) == AC50_NATIVE_PLAN_SHA,
            "AC50 seed source proof or actual node plan changed",
        )
        result_path = run / "results/full" / (task["task_id"] + "_common.json")
        result = read(result_path)
        paths, contract, inputs, designs = verify_ac50_native_refit(
            run, result, task, source, frozen
        )
        bindings = dict(
            result_sha256=sha(result_path),
            source_table_sha256=task["table_sha256"],
            source_proof_sha256=descriptor["proof_sha256"],
            predeclared_node_plan_sha256=AC50_NATIVE_PLAN_SHA,
            selected_route=result["selected_route"],
            selected_model=result["selected_model"],
            selected_input_hash=result["input_hash"],
            selected_feature_width=designs["molecule_plus_context"]["width"],
            oof_sha256=sha(paths["oof"]),
            holdout_sha256=sha(paths["holdout"]),
            outer_model_sha256=sha(paths["model"]),
            input_contract_sha256=sha(paths["input_contract"]),
            precision_consistency_policy="same_full_input_graph_context_fold_binary32_LSQR_32ULP_v1",
        )
        require(
            all(report.get(k) == v for k, v in bindings.items()),
            "AC50 selected source/model/OOF/holdout/input report binding changed",
        )
        check = next(c for c in checks["seeds"] if c["seed"] == report["seed"])
        outer = read(paths["outer_split"])
        require(
            outer.get("seed") == inputs["config"].get("seed") == report["seed"]
            and outer.get("train_records") == check["train_record_ids"]
            and outer.get("test_records") == check["holdout_record_ids"]
            and outer.get("record_groups") == {r["record_id"]: r["split_group"] for r in source},
            "AC50 saved outer split does not use pre-frozen node memberships",
        )
        inner, held = actual_folds(paths["oof"]), table(paths["holdout"])
        require(
            result.get("n_train") == len(inner) and result.get("n_test") == len(held),
            "AC50 result role counts differ",
        )
        verify_ac50_native_evaluation(
            report,
            source,
            inner,
            held,
            check,
            designs["molecule_plus_context"],
            dict(
                selected_model=result["selected_model"],
                linear_solver_policy=contract.get("linear_solver_policy"),
            ),
        )
        for role, key in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            check_metrics(
                result[key],
                {
                    k: report["evaluation_by_role"][role]["record_weighted"][k]
                    for k in ("mae", "rmse")
                },
            )
        controls = report.get("fixed_ridge_controls", {})
        require(set(controls) == set(designs), "AC50 molecule/context/joint fixed controls missing")
        for name, control in controls.items():
            design = designs[name]
            require(
                control.get("model") == "fixed_ridge_alpha1_LSQR_float64_not_Optuna_selected"
                and control.get("diagnostic_precision") == "float64"
                and control.get("diagnostic_control_precision_policy") == AC50_NATIVE_POLICY
                and control.get("exact_upcast_of_stored_float32_design") is True
                and control.get("stored_float32_design_sha256") == design["matrix_sha256"]
                and control.get("diagnostic_float64_design_sha256")
                == design["float64_matrix_sha256"]
                and control.get("same_frozen_splits") is True,
                "AC50 controls changed fixedalpha1 LSQR, exact f64 upcast or frozen splits",
            )
            verify_ac50_native_evaluation(
                control,
                source,
                control["evaluation_by_role"]["inner_oof"]["predictions"],
                control["evaluation_by_role"]["outer_holdout"]["predictions"],
                check,
                design,
                {},
                float64=True,
            )
        refit = report.get("separate_all_common_refit", {})
        require(
            refit.get("status") == "verified"
            and refit.get("records") == 77
            and refit.get("molecular_groups") == 17
            and refit.get("original_clear_records") == 78
            and refit.get("excluded_feature_only_ids") == [AC50_NATIVE_NR13]
            and refit.get("training_record_ids") == contract["common_record_ids"]
            and contained(run, refit["model_path"]) == paths["refit_model"]
            and refit.get("model_sha256") == sha(paths["refit_model"])
            and refit.get("model_contract_sha256") == sha(paths["model_contract"])
            and refit.get("fit_scope") == "all_common_after_evaluation"
            and refit.get("independently_evaluated_on_these77") is False,
            "AC50 native77/78 participation lacks a separate all77 final training refit",
        )
    durable = read(entry["config_path"])["durable_root"]
    require(
        sha(contained(durable, receipt["durable_path"]))
        == receipt.get("durable_sha256")
        == receipt["sha256"],
        "AC50 mandatory report durable copy changed",
    )
    return dict(
        status="verified",
        path=str(path),
        sha256=receipt["sha256"],
        required_seed_reports=3,
        source_records_per_seed=77,
        molecular_groups_per_seed=17,
        all77_full_refit_verified=3,
        original_clear_records=78,
        excluded_feature_only_ids=[AC50_NATIVE_NR13],
        source_annotation_only=True,
        biological_potency_equivalence_verified=False,
        source_external_validation_claimed=False,
        diagnostic_controls_precision="float64",
        diagnostic_control_fits_bound_by_source_and_membership=54,
        independent_per_fit_control_model_files_present=False,
    )


def verify_ac50_native_coverage(entry, journal, task, state):
    descriptor, _, _, _, _ = verify_ac50_native_source(entry, task)
    evidence = dict(
        **descriptor,
        task_id=task["task_id"],
        source_proof_verified=True,
        coverage_kind="native_AC50_semantic_source_annotation_observation_participation",
        source_annotation_only=True,
        required_seed_full_refit_training_verified=0,
        state="native_source_observation_"
        + {"trained_three_seeds": "unverified"}.get(state, state),
    )
    if state == "trained_three_seeds":
        verify_ac50_native_reporting(entry, journal, task)
        evidence.update(
            state="native_source_observation_trained", required_seed_full_refit_training_verified=3
        )
    return evidence


def is_mhc10_native_task(task):
    definition = task.get("scientific_definition", {})
    return (
        task.get("endpoint") == "MHC10"
        and isinstance(definition, dict)
        and definition.get("molecular_equal_diagnostics_required") is True
        and definition.get("molecular_reporting_schema") == MHC10_NATIVE_REPORT
    )


def is_hc50_primary_task(task):
    definition = task.get("scientific_definition", {})
    return (
        task.get("endpoint") == "hemolysis_HC50"
        and isinstance(definition, dict)
        and definition.get("molecular_equal_diagnostics_required") is True
        and definition.get("molecular_reporting_schema") == HC50_PRIMARY_REPORT
    )


def error_metrics(target, prediction):
    """Recompute log errors without squaring or summing unscaled large values."""
    require(len(target) == len(prediction) > 0, "Diagnostic vectors differ or are empty")
    require(all(math.isfinite(x) for x in [*target, *prediction]), "Nonfinite diagnostic value")
    scale = max(abs(x) for x in [*target, *prediction])
    delta = (
        [p / scale - y / scale for y, p in zip(target, prediction, strict=True)]
        if scale
        else [0.0] * len(target)
    )
    metrics = {
        "mae": math.fsum(abs(x) for x in delta) / len(delta) * scale,
        "rmse": math.sqrt(math.fsum(x * x for x in delta) / len(delta)) * scale,
    }
    require(
        all(math.isfinite(x) for x in metrics.values()), "Diagnostic error exceeds numeric range"
    )
    return metrics


def check_metrics(reported, computed):
    require(isinstance(reported, dict), "Reported diagnostic metrics are not an object")
    for name, expected in computed.items():
        actual = reported.get(name)
        require(
            isinstance(actual, (float, int)) and not isinstance(actual, bool),
            "Reported diagnostic metric is not numeric",
        )
        require(
            math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
            "Reported diagnostic metric differs from source OOF",
        )


def molecular_reporting_profile(task):
    if task["endpoint"] == "HC10":
        definition = task.get("scientific_definition", {})
        schema = definition.get("molecular_reporting_schema")
        if schema is not None:
            require(
                schema == HC10_CANONICAL_REPORT
                and definition.get("molecular_reporting_record_count") == 10
                and definition.get("canonical_quantity_id") == HC10_CANONICAL_QUANTITY,
                "Unsupported HC10 molecular-report version or quantity",
            )
            return {
                "file": "hc10_molecular_diagnostics.json",
                "schema": HC10_CANONICAL_REPORT,
                "records": 10,
                "groups": 5,
                "metrics_space": "log10_uM",
                "constant_molecular_prediction": True,
                "source_fields": (
                    "source_paper",
                    "reported_time_expressions",
                    "graph_identity_tier",
                    "source_original_endpoint",
                    "canonical_quantity_id",
                ),
            }
        return {
            "file": "hc10_molecular_diagnostics.json",
            "records": 9,
            "groups": 5,
            "metrics_space": "log10_uM",
            "constant_molecular_prediction": True,
            "source_fields": ("source_paper", "reported_time_expressions", "graph_identity_tier"),
        }
    if task["endpoint"] == "CL_absolute":
        return {
            "file": "cl_absolute_molecular_diagnostics.json",
            "schema": "conditional_cl_absolute_molecular_diagnostics_v1",
            "records": 7,
            "groups": 5,
            "metrics_space": "log10_L_per_h",
            "constant_molecular_prediction": False,
            "source_fields": ("source_paper", "reported_context", "graph_identity_tier"),
        }
    if task["endpoint"] == "Vd_per_kg":
        definition = task.get("scientific_definition", {})
        require(
            definition.get("molecular_reporting_schema") == VOLUME_REPORT
            and definition.get("molecular_reporting_record_count") == 20
            and definition.get("typed_estimand_multitask") is True
            and definition.get("dose_optional") is True,
            "Unsupported typed volume report or dose scope",
        )
        return {
            "file": "vd_per_kg_molecular_diagnostics.json",
            "schema": VOLUME_REPORT,
            "records": 20,
            "groups": 5,
            "metrics_space": "log10_L_per_kg",
            "constant_molecular_prediction": False,
            "source_fields": ("source_paper", "reported_context", "graph_identity_tier"),
        }
    raise ValueError("Unsupported mandatory molecular-report endpoint")


def mean_molecular_errors(metrics):
    require(bool(metrics), "Molecular errors are empty")
    mae_scale = max(g["mae"] for g in metrics)
    rmse_scale = max(g["rmse"] for g in metrics)
    return {
        "mae": math.fsum(g["mae"] / mae_scale for g in metrics) / len(metrics) * mae_scale
        if mae_scale
        else 0.0,
        "rmse": math.sqrt(math.fsum((g["rmse"] / rmse_scale) ** 2 for g in metrics) / len(metrics))
        * rmse_scale
        if rmse_scale
        else 0.0,
    }


def verify_cl_context_reporting(entry, task, run, result, predictions):
    """Bind CL context diagnostics to the frozen task and actual five-fold predictions."""
    manifest_path = Path(entry["manifest_path"])
    require(sha(manifest_path) == entry["manifest_sha256"], "CL source manifest changed")
    source_tasks = [t for t in read(manifest_path)["tasks"] if t["task_id"] == task["task_id"]]
    require(len(source_tasks) == 1, "CL source task missing or repeated")
    expected = source_tasks[0]["fixed_external_covariates"]
    model_path = contained(run, result["artifacts"]["model_contract"])
    require(
        sha(model_path) == result["artifact_hashes"][str(model_path)]["sha256"],
        "CL contextual model contract SHA mismatch",
    )
    external = read(model_path)["fixed_external_covariates"]
    require(external == expected, "CL context differs from the frozen source contract")
    unsigned = {k: v for k, v in external.items() if k != "sha256"}
    signature = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    require(
        external.get("schema") == "conditional_total_cl_minimum_context_v1"
        and external.get("sha256") == signature
        and external.get("chemical_method_count_increment") == 0,
        "CL context schema/hash/method count changed",
    )
    contexts = external["contexts_by_record"]
    ids = [r["record_id"] for r in predictions]
    require(set(contexts) == set(ids), "CL context record membership differs from source OOF")
    core_keys = {"species", "route", "matrix", "dose_value", "dose_reported_unit"}
    dose_units = external["dose_units"]
    for value in contexts.values():
        require(
            set(value) == core_keys
            and value["species"] == "human"
            and value["route"] == "intravenous"
            and value["matrix"] in {"plasma", "serum"}
            and isinstance(value["dose_value"], (float, int))
            and not isinstance(value["dose_value"], bool)
            and math.isfinite(value["dose_value"])
            and value["dose_value"] > 0
            and value["dose_reported_unit"] in dose_units,
            "CL source context is incomplete or outside human IV scope",
        )
    same_inputs = defaultdict(list)
    for row in predictions:
        context_key = json.dumps(contexts[row["record_id"]], sort_keys=True, separators=(",", ":"))
        same_inputs[(row["split_group"], context_key)].append(float(row["prediction"]))
    require(
        all(
            all(abs(p - values[0]) <= 1e-8 + 1e-7 * abs(values[0]) for p in values)
            for values in same_inputs.values()
        ),
        "CL predictions differ for the same molecule and context",
    )
    path = contained(run, result["artifacts"]["context_ablations"])
    require(
        sha(path) == result["artifact_hashes"][str(path)]["sha256"],
        "CL context diagnostics SHA mismatch",
    )
    diagnostics = read(path)
    require(
        diagnostics.get("record_ids") == ids
        and diagnostics.get("chemical_method_count_increment") == 0
        and diagnostics.get("molecular_signal_established") is False,
        "CL diagnostic record order or claim boundary changed",
    )
    designs = diagnostics.get("ablations", {})
    require(
        set(designs) == {"context_only", "molecule_only", "molecule_plus_context"},
        "CL condition/molecule ablations are missing",
    )
    targets = [float(r["target"]) for r in predictions]
    grouped_indices = defaultdict(list)
    for i, row in enumerate(predictions):
        grouped_indices[row["split_group"]].append(i)
    for name, design in designs.items():
        values = design.get("predictions", [])
        require(len(values) == len(ids), "CL baseline predictions omit source records")
        require(
            design.get("model") == "fixed_ridge_regularization_1_not_optuna_selected",
            "CL diagnostic baseline was selected or changed",
        )
        check_metrics(design.get("metrics", {}), error_metrics(targets, values))
        equal_inputs = defaultdict(list)
        for row, value in zip(predictions, values, strict=True):
            context_key = json.dumps(
                contexts[row["record_id"]], sort_keys=True, separators=(",", ":")
            )
            inputs = (
                (int(row["fold"]), row["split_group"])
                if name == "molecule_only"
                else (int(row["fold"]), context_key)
                if name == "context_only"
                else (int(row["fold"]), row["split_group"], context_key)
            )
            equal_inputs[inputs].append(value)
        require(
            all(
                all(abs(p - same[0]) <= 1e-8 + 1e-7 * abs(same[0]) for p in same)
                for same in equal_inputs.values()
            ),
            "CL diagnostic baseline differs for identical inputs within one fold",
        )
        errors = [
            error_metrics([targets[i] for i in indices], [values[i] for i in indices])
            for indices in grouped_indices.values()
        ]
        check_metrics(
            {
                "mae": design.get("compound_equal_weight_mae"),
                "rmse": design.get("compound_equal_weight_rmse"),
            },
            mean_molecular_errors(errors),
        )
    coverage = diagnostics.get("fold_context_coverage", [])
    require(
        len(coverage) == 5 and {r["fold"] for r in coverage} == set(range(5)),
        "CL context coverage omits or repeats folds",
    )
    for fold in coverage:
        validation = [r for r in predictions if int(r["fold"]) == fold["fold"]]
        training = [r for r in predictions if int(r["fold"]) != fold["fold"]]
        require(
            fold.get("train_groups") == sorted({r["split_group"] for r in training})
            and fold.get("validation_groups") == sorted({r["split_group"] for r in validation}),
            "CL context coverage group membership changed",
        )
        unsupported = {
            key: sorted(
                {contexts[r["record_id"]][key] for r in validation}
                - {contexts[r["record_id"]][key] for r in training}
            )
            for key in ("species", "route", "matrix")
        }
        unsupported["dose_basis"] = sorted(
            {dose_units[contexts[r["record_id"]]["dose_reported_unit"]][0] for r in validation}
            - {dose_units[contexts[r["record_id"]]["dose_reported_unit"]][0] for r in training}
        )
        require(
            fold.get("unsupported_validation_context") == unsupported,
            "CL unsupported fold context differs from the actual source split",
        )


def control_volume_context_bytes(contexts, external):
    """Reproduce only the frozen33-channel encoder, without executing producer code."""
    ontology = {
        "species": ["human", "mouse", "rat", "dog", "monkey", "rabbit"],
        "route": ["oral", "intravenous", "subcutaneous", "intraperitoneal", "intramuscular"],
        "matrix": ["plasma", "serum", "whole_blood"],
    }
    units = {
        "mg": ["mass_absolute_mg", 1],
        "g": ["mass_absolute_mg", 1000],
        "ug": ["mass_absolute_mg", 0.001],
        "mg/kg": ["mass_per_kg_mg", 1],
        "ug/kg": ["mass_per_kg_mg", 0.001],
        "umol": ["molar_absolute_umol", 1],
        "nmol": ["molar_absolute_umol", 0.001],
        "umol/kg": ["molar_per_kg_umol", 1],
        "nmol/kg": ["molar_per_kg_umol", 0.001],
        "mg/m'2": ["mass_per_m2_mg", 1],
        "mg/m2": ["mass_per_m2_mg", 1],
        "mg/kg/day": ["daily_mass_per_kg_mg", 1],
    }
    suffix = {
        "volume_estimand": ["Vz", "Vss", "Vz_over_F"],
        "F_basis": ["not_divided_by_F", "divided_by_unknown_F"],
        "dose_status": ["reported", "not_reported"],
    }
    bases = sorted({value[0] for value in units.values()})
    names = ["context:dose_" + name for name in bases]
    names += ["context:basis_" + name for name in bases]
    names += [f"context:{key}_{value}" for key, values in ontology.items() for value in values]
    names += [f"context:{key}_{value}" for key, values in suffix.items() for value in values]
    require(
        external.get("dose_units") == units
        and external.get("ontology") == {**ontology, **suffix}
        and external.get("feature_names") == names,
        "Diagnostic controls require the exact frozen volume encoder ontology",
    )
    encoded = {}
    for record_id, context in contexts.items():
        require(
            set(context) == set(ontology) | set(suffix) | {"dose_value", "dose_reported_unit"}
            and all(context[key] in values for key, values in {**ontology, **suffix}.items())
            and context["F_basis"]
            == (
                "divided_by_unknown_F"
                if context["volume_estimand"] == "Vz_over_F"
                else "not_divided_by_F"
            ),
            "Diagnostic volume context is missing or physically inconsistent",
        )
        if context["dose_status"] == "not_reported":
            require(
                context["dose_value"] is None and context["dose_reported_unit"] is None,
                "Diagnostic unknown dose must retain None/None",
            )
            values = [0.0] * (2 * len(bases))
        else:
            dose = context["dose_value"]
            require(
                isinstance(dose, (int, float))
                and not isinstance(dose, bool)
                and math.isfinite(dose)
                and dose > 0
                and context["dose_reported_unit"] in units,
                "Diagnostic reported dose is invalid",
            )
            basis, factor = units[context["dose_reported_unit"]]
            values = [dose * factor if value == basis else 0.0 for value in bases]
            values += [float(value == basis) for value in bases]
        values += [
            float(context[key] == value) for key, items in ontology.items() for value in items
        ]
        values += [float(context[key] == value) for key, items in suffix.items() for value in items]
        try:
            data = struct.pack("<33f", *values)
        except (OverflowError, struct.error) as exc:
            raise ValueError("Diagnostic volume context is outside finite binary32") from exc
        require(
            all(math.isfinite(v) for v in struct.unpack("<33f", data)),
            "Diagnostic volume context is outside finite binary32",
        )
        encoded[record_id] = data
    return names, encoded


def control_precision_designs(run, source, result, inputs, contract, kind, external=None):
    """Bind diagnostic designs to original signed binary32 values, including NaNs.

    This is byte reconstruction only. It never loads a model, learns an encoder,
    imputes missing features, predicts, or changes the legacy consistency tolerance.
    """
    require(kind in CONTROL_PRODUCER_SHAS, "Unsupported diagnostic control source")
    run = Path(run)
    features = read(run / "features/manifest.json")
    producer = (
        "peptide_modeling_conditional_volume.py"
        if kind == "volume"
        else "peptide_modeling_mhc10_native_bridge.py"
    )
    configuration = features["configuration"]
    require(
        sha(run / "scripts" / producer) == CONTROL_PRODUCER_SHAS[kind]
        and configuration.get("implementation_sha256") == CONTROL_PRODUCER_SHAS[kind]
        and sha(run / "scripts/peptide_modeling_train.py") == CONTROL_TRAINING_SHA,
        "Diagnostic fixed Ridge/LSQR implementation is not the frozen source",
    )
    native = kind == "native"
    rows_count, graph_count = (27, 22) if native else (20, 5)
    require(
        features.get("requested_records") == rows_count
        and features.get("structure_keys") == graph_count
        and len(source) == rows_count
        and len({r["record_id"] for r in source}) == rows_count
        and len({r["canonical_identity_group"] for r in source}) == graph_count,
        "Diagnostic frozen source membership changed",
    )
    if not native:
        require(
            configuration.get("encoding_dependency_sha256") == CONTROL_VOLUME_ENCODER_SHA
            and sha(run / "scripts/peptide_modeling_conditional_cmax.py")
            == CONTROL_VOLUME_ENCODER_SHA,
            "Diagnostic volume encoder implementation changed",
        )
    fingerprint = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            **({"separators": (",", ":"), "ensure_ascii": False} if native else {}),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    cache_name = str(features["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    require(
        features.get("input_fingerprint") == fingerprint and cache_name == fingerprint[:16],
        "Diagnostic bank configuration/fingerprint changed",
    )
    cache = contained(run, "features/" + cache_name)
    for relative, expected_sha in features["artifact_hashes"].items():
        require(sha(contained(cache, relative)) == expected_sha, "Diagnostic feature bytes changed")
    aliases_list = table(contained(cache, features["record_index"]))
    aliases = {r["record_id"]: r for r in aliases_list}
    require(
        len(aliases_list) == len(aliases) == rows_count
        and set(aliases) == {r["record_id"] for r in source}
        and all(
            aliases[r["record_id"]]["structure_key"] == r["canonical_identity_group"]
            and r.get("structure_eligible") == "true"
            for r in source
        ),
        "Diagnostic aliases no longer prove exact source graphs",
    )
    groups = features["routes"][result["selected_route"]]
    require(groups and len(groups) == len(set(groups)), "Diagnostic route groups changed")
    require(
        (groups[-1:] == [MHC10_NATIVE_CONTEXT] and groups.count(MHC10_NATIVE_CONTEXT) == 1)
        if native
        else MHC10_NATIVE_CONTEXT not in groups,
        "Diagnostic species context was duplicated or added to the volume bank",
    )
    chemical, context, chemical_names = {r["record_id"]: b"" for r in source}, {}, []
    for name in groups:
        meta = features["groups"][name]
        is_context = native and name == MHC10_NATIVE_CONTEXT
        require(
            meta.get("key_kind") == ("record_id" if is_context else "canonical_identity_group"),
            "Diagnostic chemical route contains unsupported record/sequence features",
        )
        index_rows = table(contained(cache, meta["index"]))
        index = {r["key"]: int(r["row_idx"]) for r in index_rows}
        count = rows_count if is_context else graph_count
        require(
            len(index_rows) == len(index) == count and set(index.values()) == set(range(count)),
            "Diagnostic feature index changed",
        )
        header, matrix = hc50_precision_npy(contained(cache, meta["matrix"]))
        valid_header, valid = hc50_precision_npy(contained(cache, meta["valid"]))
        width = len(meta["feature_names"])
        require(
            header["shape"] == tuple(meta["shape"]) == (count, width)
            and valid_header["descr"] == "|b1"
            and valid_header["shape"] == (count,),
            "Diagnostic feature matrix/mask schema changed",
        )
        if is_context:
            require(
                meta["feature_names"] == ["human", "mouse"]
                and meta.get("eligibility_field") == "rbc_species_eligible",
                "Diagnostic species context schema changed",
            )
        else:
            chemical_names += [name + ":" + n for n in meta["feature_names"]]
        for row in source:
            record_id = row["record_id"]
            pos = index[record_id if is_context else row["canonical_identity_group"]]
            require(valid[pos] == 1, "Diagnostic input has an invalid feature mask")
            size = width * (4 if header["descr"] == "<f4" else 1)
            data = matrix[pos * size : (pos + 1) * size]
            if header["descr"] == "|u1":
                data = struct.pack("<" + "f" * width, *data)
            else:
                require(header["descr"] == "<f4", "Diagnostic features are not binary32")
            bits = struct.unpack("<" + "I" * width, data)
            missing = sum((b & 0x7FFFFFFF) > 0x7F800000 for b in bits)
            require(
                not any((b & 0x7FFFFFFF) == 0x7F800000 for b in bits)
                and missing < width
                and (
                    not missing
                    or (
                        meta.get("allow_partial_missing") is True
                        and 0 <= meta.get("max_missing_fraction", 0.1) < 1
                        and missing / width <= meta.get("max_missing_fraction", 0.1) + 1e-12
                    )
                ),
                "Diagnostic feature missingness/finite policy changed",
            )
            if is_context:
                species = row["rbc_species"]
                require(
                    species in {"human", "mouse"}
                    and row.get("rbc_species_eligible") == "true"
                    and data
                    == struct.pack("<2f", float(species == "human"), float(species == "mouse")),
                    "Diagnostic species bytes differ from the frozen source context",
                )
                context[record_id] = data
            else:
                chemical[record_id] += data
    if native:
        context_names = MHC10_NATIVE_CONTEXT_NAMES
        signature = inputs["feature_hashes"][result["selected_route"]]
        mask_key = "mask_sha256"
    else:
        require(isinstance(external, dict), "Diagnostic volume source context is required")
        contexts = external["contexts_by_record"]
        require(set(contexts) == set(chemical), "Diagnostic volume context membership changed")
        context_names, context = control_volume_context_bytes(contexts, external)
        signature = inputs["features"][result["selected_route"]]
        mask_key = "valid_sha256"
    names = chemical_names + context_names
    joint = {record_id: chemical[record_id] + context[record_id] for record_id in chemical}
    require(
        names == contract.get("feature_names") == signature.get("names")
        and hashlib.sha256(b"".join(joint[r["record_id"]] for r in source)).hexdigest()
        == signature.get("matrix_sha256")
        and hashlib.sha256(b"\x01" * rows_count).hexdigest() == signature.get(mask_key),
        "Diagnostic design bytes differ from the original fitted input signature",
    )
    return dict(
        feature_names=names,
        designs=dict(molecule_only=chemical, context_only=context, molecule_plus_context=joint),
    )


def control_precision_matrix_bytes(rows, record_ids):
    """Exactly upcast signed binary32 values; keep missing values and row order."""
    require(
        record_ids and len(record_ids) == len(set(record_ids)) and set(record_ids) <= set(rows),
        "Diagnostic matrix subset repeats or invents a source observation",
    )
    width = len(rows[record_ids[0]]) // 4
    require(
        width > 0 and all(len(rows[i]) == 4 * width for i in record_ids),
        "Diagnostic matrix row width changed",
    )
    binary32 = b"".join(rows[i] for i in record_ids)
    values = struct.unpack("<" + "f" * (len(binary32) // 4), binary32)
    require(not any(math.isinf(value) for value in values), "Diagnostic input is infinite")
    return dict(
        shape=[len(record_ids), width],
        source_float32_sha256=hashlib.sha256(binary32).hexdigest(),
        source_nan_mask_sha256=hashlib.sha256(bytes(math.isnan(v) for v in values)).hexdigest(),
        upcast_float64_sha256=hashlib.sha256(
            struct.pack("<" + "d" * len(values), *values)
        ).hexdigest(),
    )


def verify_control_volume_training_signature(run, source, result, inputs, contract, features):
    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest()

    task = read(Path(run) / "task_manifest.json")["tasks"]
    require(
        len(task) == 1
        and task[0] == inputs["task"] == contract["task"]
        and contract.get("record_ids") == [row["record_id"] for row in source]
        and inputs.get("runtime") == result.get("runtime") == contract.get("runtime")
        and inputs["runtime"].get("training_source_sha256") == CONTROL_TRAINING_SHA
        and inputs["runtime"].get("internal_cv_source_sha256")
        == sha(Path(run) / "scripts/peptide_modeling_internal_cv.py")
        and inputs.get("source_rows_hash") == digest(source),
        "Float64 diagnostic original volume source/model/input scope changed",
    )
    signature = digest(
        dict(
            protocol=inputs["protocol"],
            task=inputs["task"],
            config=inputs["config"],
            rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
            features=inputs["features"],
            feature_fingerprint=features["input_fingerprint"],
            runtime=result["runtime"],
            external_context=result["external_context"],
        )
    )
    require(
        signature
        == inputs.get("input_hash")
        == result.get("input_hash")
        == contract.get("input_hash"),
        "Float64 diagnostic original volume training signature changed",
    )


def control_precision_consistency(predictions, design_rows):
    """Describe legacy failures from actual identical inputs; grant no tolerance exception."""
    identical = defaultdict(list)
    require(
        len(predictions) == len({r["record_id"] for r in predictions})
        and all(r["record_id"] in design_rows for r in predictions),
        "Diagnostic prediction membership repeats or invents a source observation",
    )
    for row in predictions:
        require(math.isfinite(float(row["prediction"])), "Diagnostic prediction is nonfinite")
        identical[(int(row["fold"]), design_rows[row["record_id"]])].append(row)
    failures = []
    for (fold, data), rows in identical.items():
        values = [float(row["prediction"]) for row in rows]
        if all(abs(value - values[0]) <= 1e-8 + 1e-7 * abs(values[0]) for value in values):
            continue
        try:
            positions = [hc50_precision_binary32(value) for value in values]
            ulps = max(positions) - min(positions)
        except ValueError:
            ulps = None
        failures.append(
            dict(
                fold=fold,
                source_record_ids=[row["record_id"] for row in rows],
                input_row_sha256=hashlib.sha256(data).hexdigest(),
                prediction_values=values,
                exact_binary32=ulps is not None,
                binary32_ulp_distance=ulps,
                normal_one_sign=(
                    all(abs(value) >= 2.0**-126 for value in values)
                    and len({math.copysign(1, value) for value in values}) == 1
                ),
                original_legacy_guard_passed=False,
                qualified=False,
            )
        )
    return failures


def verify_control_float64_seed(evidence, binding, source, check, controls, kind):
    """Check every diagnostic fit against original ordered values and frozen folds."""
    native = kind == "native"
    ids = [row["record_id"] for row in source]
    by_id = {row["record_id"]: row for row in source}
    fits_per_design = 6 if native else 5
    names = {"context_only", "molecule_only", "molecule_plus_context"}
    require(
        evidence.get("schema") == "fixed_ridge_float64_control_seed_evidence_v1"
        and evidence.get("dataset_kind") == ("native_mhc10" if native else "volume")
        and evidence.get("seed") == check["seed"]
        and evidence.get("diagnostic_fits") == 3 * fits_per_design
        and evidence.get("selected_model_fits") == evidence.get("parameter_searches") == 0
        and evidence.get("original_strict_guard_passed") is True
        and isinstance(evidence.get("designs"), dict)
        and set(evidence["designs"]) == names
        and set(binding["designs"]) == names,
        "Float64 diagnostic cohort, design scope or fit accounting changed",
    )
    designs = controls if native else controls["ablations"]
    require(set(designs) == names, "Float64 controls omit or add a diagnostic design")
    model = dict(
        **{"class": "Ridge"},
        alpha=1.0,
        solver="lsqr",
        imputer_strategy="median",
        imputer_add_indicator=True,
        imputer_keep_empty_features=True,
        scaler_with_mean=False,
    )
    for name in sorted(names):
        description = evidence["designs"][name]
        rows = binding["designs"][name]
        actual_matrix = control_precision_matrix_bytes(rows, ids)
        require(
            isinstance(description, dict)
            and description.get("source_record_ids") == ids
            and all(description.get(k) == v for k, v in actual_matrix.items())
            and description.get("finite_values_exactly_preserved") is True
            and description.get("missing_mask_preserved") is True
            and description.get("fit_input_dtype") == "float64"
            and description.get("predict_input_dtype") == "float64",
            "Float64 control is not the exact upcast of signed original feature values",
        )
        if native:
            roles = designs[name]["evaluation_by_role"]
            predicted = {
                role: roles[role]["predictions"] for role in ("inner_oof", "outer_holdout")
            }
        else:
            values = designs[name]["predictions"]
            require(len(values) == len(ids), "Float64 volume control omitted a source row")
            folds = {
                record_id: fold["fold"]
                for fold in check["folds"]
                for record_id in fold["validation_record_ids"]
            }
            predicted = {
                "inner_oof": [
                    dict(record_id=record_id, fold=folds[record_id], prediction=value)
                    for record_id, value in zip(ids, values, strict=True)
                ]
            }
        require(
            all(not control_precision_consistency(values, rows) for values in predicted.values()),
            "Float64 diagnostic controls still fail the original identical-input guard",
        )
        guard_checks = []
        guard_scopes = [
            ("inner_oof", fold["fold"], fold["validation_record_ids"]) for fold in check["folds"]
        ]
        if native:
            guard_scopes.append(("outer_holdout", -1, check["holdout_record_ids"]))
        for role, fold, predicted_ids in guard_scopes:
            by_record = {row["record_id"]: row for row in predicted[role]}
            groups = defaultdict(list)
            for record_id in predicted_ids:
                groups[rows[record_id]].append(by_record[record_id])
            for data, same in groups.items():
                if len(same) < 2:
                    continue
                values64 = [float(row["prediction"]) for row in same]
                row_values = struct.unpack("<" + "f" * (len(data) // 4), data)
                guard_checks.append(
                    dict(
                        evaluation_role=role,
                        fold=fold,
                        source_record_ids=[row["record_id"] for row in same],
                        input_row_float64_sha256=hashlib.sha256(
                            struct.pack("<" + "d" * len(row_values), *row_values)
                        ).hexdigest(),
                        prediction_values=values64,
                        observed_absolute_spread=max(values64) - min(values64),
                        original_allclose_pass=True,
                    )
                )
        require(
            description.get("original_guard_checks") == guard_checks,
            "Float64 diagnostic original-guard evidence differs from saved predictions/input bytes",
        )
        expected = [
            dict(
                fit_role="inner_oof",
                fold=fold["fold"],
                train_record_ids=fold["train_record_ids"],
                predict_record_ids=fold["validation_record_ids"],
            )
            for fold in check["folds"]
        ]
        require(
            len(expected) == 5 and [fold["fold"] for fold in check["folds"]] == list(range(5)),
            "Float64 controls need the original actual five folds in order",
        )
        if native:
            expected.append(
                dict(
                    fit_role="outer_holdout",
                    fold=None,
                    train_record_ids=check["train_record_ids"],
                    predict_record_ids=check["holdout_record_ids"],
                )
            )
        fits = description.get("fits", [])
        require(len(fits) == fits_per_design, "Float64 diagnostic fit count changed")
        for index, (fit, frozen) in enumerate(zip(fits, expected, strict=True)):
            train_ids, predicted_ids = frozen["train_record_ids"], frozen["predict_record_ids"]
            actual_predictions = predicted[frozen["fit_role"]]
            by_record = {row["record_id"]: row for row in actual_predictions}
            require(
                len(by_record) == len(actual_predictions)
                and all(record_id in by_record for record_id in predicted_ids)
                and all(
                    int(by_record[i]["fold"])
                    == (-1 if frozen["fit_role"] == "outer_holdout" else frozen["fold"])
                    for i in predicted_ids
                )
                and set(train_ids).isdisjoint(predicted_ids)
                and {by_id[i]["split_group"] for i in train_ids}.isdisjoint(
                    {by_id[i]["split_group"] for i in predicted_ids}
                ),
                "Float64 diagnostic prediction/fitting folds or molecular separation changed",
            )
            targets = [float(by_id[i]["target"]) for i in train_ids]
            values = [float(by_record[i]["prediction"]) for i in predicted_ids]
            require(
                all(math.isfinite(v) for v in targets + values)
                and fit.get("fit_index") == index
                and all(fit.get(k) == v for k, v in frozen.items())
                and fit.get("train_X_float64_sha256")
                == control_precision_matrix_bytes(rows, train_ids)["upcast_float64_sha256"]
                and fit.get("predict_X_float64_sha256")
                == control_precision_matrix_bytes(rows, predicted_ids)["upcast_float64_sha256"]
                and fit.get("train_y_float64_sha256")
                == hashlib.sha256(struct.pack("<" + "d" * len(targets), *targets)).hexdigest()
                and fit.get("prediction_float64_sha256")
                == hashlib.sha256(struct.pack("<" + "d" * len(values), *values)).hexdigest()
                and all(
                    fit.get(k) == "float64"
                    for k in ("fit_input_dtype", "predict_input_dtype", "fitted_coefficient_dtype")
                )
                and fit.get("model") == model,
                "Float64 diagnostic fit input/output dtype, fixed Ridge policy or values changed",
            )
    return dict(
        status="verified",
        diagnostic_fits=3 * fits_per_design,
        selected_model_fits=0,
        parameter_searches=0,
        original_strict_guard_passed=True,
    )


def control_float64_object_sha(value):
    """Match the CONTROL producer wire; HC50's independent spaced JSON policy stays unchanged."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def control_float64_revision_paths(root, revision):
    root = Path(root).resolve()
    evidence = contained(root / "report_recovery", revision["evidence_path"])
    old_report = contained(root / "report_recovery", revision["old_report_backup"])
    old_journal = contained(root / "report_recovery", revision["old_journal_backup"])
    require(
        evidence.name == "recovery_receipt.json"
        and old_report.name == "old_molecular_report.json"
        and old_journal.name == "old_campaign_status.json"
        and evidence.parent == old_report.parent == old_journal.parent
        and evidence.parent.name == "float64_controls_" + revision["recovery_id"]
        and sha(evidence) == revision["evidence_sha256"]
        and sha(old_report) == revision["old_report_sha256"]
        and sha(old_journal) == revision["old_journal_sha256"],
        "Float64 diagnostic recovery namespace or immutable backups changed",
    )
    return evidence, old_report, old_journal


def freeze_control_float64_evidence(root, journal):
    receipt = journal.get("molecular_reporting", {})
    revision = receipt.get("diagnostic_precision_revision")
    if revision is None:
        return None
    require(
        journal.get("status") == "completed"
        and receipt.get("status") == "verified"
        and revision.get("schema") == CONTROL_FLOAT64_SCHEMA
        and revision.get("implementation_sha256") == CONTROL_FLOAT64_TOOL_SHA,
        "Float64 control evidence may freeze only a released completed report",
    )
    path = contained(root, receipt["path"])
    require(
        sha(path) == receipt["sha256"]
        and read(path).get("diagnostic_precision_revision") == revision,
        "Float64 diagnostic report/receipt disagree",
    )
    evidence, _, backup = control_float64_revision_paths(root, revision)
    saved = read(evidence)
    require(
        saved.get("original_owner") == read(backup)["process_identity"]
        and saved.get("new_owner") == journal["process_identity"],
        "Float64 diagnostic old/new controller owner chain changed",
    )
    return dict(
        revision=revision,
        receipt_sha256=hc50_precision_object_sha(receipt),
        report_sha256=receipt["sha256"],
        original_owner=read(backup)["process_identity"],
        report_owner=journal["process_identity"],
    )


def freeze_control_float64_registry_entry(previous_entry):
    """Pure frozen-registry migration; no model, journal or original report writes."""
    require(
        "control_float64_evidence" not in previous_entry
        and len(previous_entry["tasks"]) == 1
        and previous_entry["tasks"][0]["endpoint"] in {"Vd_per_kg", "MHC10"},
        "Float64 migration requires the original scoped controller registry entry",
    )
    journal = read(previous_entry["journal_path"])
    pin = freeze_control_float64_evidence(previous_entry["output_root"], journal)
    require(
        pin is not None and pin["original_owner"] == previous_entry["owned_process_identity"],
        "Float64 diagnostic migration lost the old frozen controller owner",
    )
    entry = json.loads(json.dumps(previous_entry))
    entry.update(
        owned_process_identity=journal["process_identity"],
        state_at_freeze=journal["status"],
        journal_sha256_at_freeze=sha(previous_entry["journal_path"]),
        control_float64_evidence=pin,
        control_float64_registry_migration=dict(
            previous_entry_sha256=hc50_precision_object_sha(previous_entry),
            previous_owned_process_identity=previous_entry["owned_process_identity"],
            previous_config_sha256=previous_entry["config_sha256"],
            previous_manifest_sha256=previous_entry["manifest_sha256"],
        ),
    )
    verify_molecular_reporting(entry, journal, entry["tasks"][0])
    return entry


def control_float64_ast_proof(producer, kind):
    name = "fixed_ridge_controls" if kind == "native" else "contextual_diagnostics"
    functions = [
        node
        for node in ast.parse(Path(producer).read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(functions) == 1, "Float64 diagnostic source function changed")
    function = functions[0]
    original = ast.dump(function, include_attributes=False)
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [getattr(value, "id", None) for value in node.target.elts] == ["name", "X"]
    ]
    require(len(loops) == 1, "Float64 diagnostic design loop is missing or ambiguous")
    loops[0].body.insert(0, ast.parse("X = _float64_design(name, X)").body[0])
    revised = ast.dump(function, include_attributes=False)
    return dict(
        original_function_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
        upcast_function_ast_sha256=hashlib.sha256(revised.encode()).hexdigest(),
        inserted_design_casts=1,
        outside_cast_ast_identical=True,
        ast_hash_policy="sha256_entire_FunctionDef_ast_dump_without_attributes_utf8",
    )


def verify_control_float64_revision(entry, journal, receipt, payload, kind):
    """Admit only a frozen, append-only diagnostic revision; never reinterpret old fits."""
    revision = payload.get("diagnostic_precision_revision")
    if revision is None:
        require(
            receipt.get("diagnostic_precision_revision") is None
            and entry.get("control_float64_evidence") is None,
            "Frozen float64 diagnostic revision was removed",
        )
        return None
    native = kind == "native"
    expected = dict(
        schema=CONTROL_FLOAT64_SCHEMA,
        policy=CONTROL_FLOAT64_POLICY,
        dataset_kind="native_mhc10" if native else "volume",
        required_seeds=SEEDS,
        designs=["context_only", "molecule_only", "molecule_plus_context"],
        diagnostic_fits=54 if native else 45,
        total_recovery_diagnostic_fits=99,
        diagnostic_fits_by_dataset={"volume": 45, "native_mhc10": 54},
        selected_model_fits=0,
        parameter_searches=0,
        precision="float64",
        all_designs_and_seeds_recomputed=True,
        old_controls_preserved=True,
        old_float32_controls_qualified_by_this_revision=False,
        old_selected_predictions_unchanged=True,
        implementation_sha256=CONTROL_FLOAT64_TOOL_SHA,
        selected_report_implementation_sha256=CONTROL_PRODUCER_SHAS[kind],
    )
    require(
        isinstance(revision, dict)
        and receipt.get("diagnostic_precision_revision") == revision
        and all(revision.get(k) == value for k, value in expected.items())
        and receipt.get("implementation_sha256") == CONTROL_FLOAT64_TOOL_SHA
        and receipt.get("selected_report_implementation_sha256") == CONTROL_PRODUCER_SHAS[kind],
        "Float64 diagnostic scope, original-source provenance or99-fit accounting changed",
    )
    pin = entry.get("control_float64_evidence")
    require(
        isinstance(pin, dict)
        and pin.get("revision") == revision
        and pin.get("receipt_sha256") == hc50_precision_object_sha(receipt)
        and pin.get("report_sha256") == receipt["sha256"]
        and pin.get("report_owner")
        == journal["process_identity"]
        == entry["owned_process_identity"]
        and journal.get("status") == "completed"
        and identity_state(
            journal["process_identity"], process_identity(journal["process_identity"]["pid"])
        )
        == "exited",
        "Float64 report has unfrozen evidence, a changed owner or a live diagnostic wrapper",
    )
    evidence_path, old_report_path, old_journal_path = control_float64_revision_paths(
        entry["output_root"], revision
    )
    evidence, old_report, old_journal = (
        read(evidence_path),
        read(old_report_path),
        read(old_journal_path),
    )
    require(
        evidence.get("schema") == CONTROL_FLOAT64_SCHEMA
        and evidence.get("status") == "verified_diagnostic_controls_not_adopted"
        and evidence.get("recovery_id") == revision["recovery_id"]
        and evidence.get("dataset_kind") == expected["dataset_kind"]
        and evidence.get("implementation_sha256") == CONTROL_FLOAT64_TOOL_SHA
        and evidence.get("source_implementations")
        == {"producer": CONTROL_PRODUCER_SHAS[kind], "training": CONTROL_TRAINING_SHA}
        and evidence.get("original_owner")
        == pin["original_owner"]
        == old_journal["process_identity"]
        and evidence.get("new_owner") == journal["process_identity"]
        and identity_state(
            old_journal["process_identity"],
            process_identity(old_journal["process_identity"]["pid"]),
        )
        in {"exited", "pid_reused"}
        and old_journal.get("status") == "completed"
        and old_journal.get("config_sha256") == entry["config_sha256"]
        and old_journal.get("rounds") == journal.get("rounds")
        and old_journal.get("molecular_reporting", {}).get("sha256")
        == revision["old_report_sha256"]
        and evidence.get("diagnostic_fits") == expected["diagnostic_fits"]
        and evidence.get("total_diagnostic_fits") == 99
        and evidence.get("selected_model_fits") == evidence.get("parameter_searches") == 0
        and evidence.get("old_controls_preserved") is True
        and evidence.get("raw_selected_predictions_unchanged") is True,
        "Float64 recovery changed original selected rounds, source ownership or fit accounting",
    )
    old_info, journal_info = evidence.get("old_report", {}), evidence.get("old_journal", {})
    require(
        contained(entry["output_root"], old_info["path"])
        == contained(entry["output_root"], receipt["path"])
        and Path(old_info["backup_path"]).resolve() == old_report_path
        and old_info.get("sha256") == revision["old_report_sha256"]
        and old_info.get("content_sha256") == control_float64_object_sha(old_report)
        and Path(journal_info["path"]).resolve() == Path(entry["journal_path"]).resolve()
        and Path(journal_info["backup_path"]).resolve() == old_journal_path
        and journal_info.get("sha256") == revision["old_journal_sha256"],
        "Float64 original report/journal archive binding changed",
    )
    stripped = json.loads(json.dumps(payload))
    del stripped["diagnostic_precision_revision"]
    for report in stripped.get("reports", []):
        require(
            "float64_fixed_ridge_controls" in report and "float64_control_evidence" in report,
            "Float64 report omitted one prespecified seed",
        )
        del report["float64_fixed_ridge_controls"], report["float64_control_evidence"]
    require(stripped == old_report, "Float64 revision changed an original report field or metric")
    history = [
        item
        for item in journal.get("recovery_history", [])
        if item.get("kind") == CONTROL_FLOAT64_SCHEMA
        and item.get("recovery_id") == revision["recovery_id"]
    ]
    require(
        len(history) == 1
        and journal.get("recovery_history")[:-1] == old_journal.get("recovery_history", [])
        and journal["recovery_history"][-1] == history[0]
        and history[0].get("old_journal_sha256") == revision["old_journal_sha256"]
        and history[0].get("old_report_sha256") == revision["old_report_sha256"]
        and Path(history[0]["backup"]).resolve() == old_journal_path
        and Path(history[0]["old_report_backup"]).resolve() == old_report_path
        and history[0].get("tool_sha256") == CONTROL_FLOAT64_TOOL_SHA
        and history[0].get("new_owner") == journal["process_identity"]
        and history[0].get("diagnostic_fits_planned") == expected["diagnostic_fits"]
        and history[0].get("selected_model_fits") == 0,
        "Float64 diagnostic adoption did not preserve the old journal/source chain",
    )
    producer = (
        Path(entry["template"])
        / "scripts"
        / (
            "peptide_modeling_mhc10_native_bridge.py"
            if native
            else "peptide_modeling_conditional_volume.py"
        )
    )
    require(
        sha(producer) == CONTROL_PRODUCER_SHAS[kind]
        and evidence.get("ast_provenance") == control_float64_ast_proof(producer, kind),
        "Float64 diagnostic changed source logic outside one exact design upcast",
    )
    protected = evidence.get("protected_artifact_hashes")
    require(isinstance(protected, dict) and protected, "Float64 original artifact set is missing")
    normalized = {str(Path(path).resolve()): value for path, value in protected.items()}
    require(
        len(normalized) == len(protected)
        and all(
            isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) and sha(path) == value
            for path, value in normalized.items()
        ),
        "Float64 recovery changed a protected original model/source/prediction/control file",
    )
    seeds = evidence.get("seeds", [])
    require(
        len(seeds) == 3 and {item["seed"] for item in seeds} == set(SEEDS),
        "Float64 recovery omitted or repeated a prescribed seed",
    )
    joint_info = evidence.get("joint_completion", {})
    joint_path = contained(evidence_path.parent, joint_info["path"])
    require(
        joint_path.parent == evidence_path.parent
        and joint_path.name == "joint_completion.json"
        and sha(joint_path) == joint_info.get("sha256"),
        "Float64 joint completion path or SHA changed",
    )
    joint = read(joint_path)
    prefit_info = evidence.get("pre_fit_plan", {})
    prefit_path = contained(evidence_path.parent, prefit_info["path"])
    require(
        prefit_path.parent == evidence_path.parent
        and prefit_path.name == "pre_fit_plan.json"
        and sha(prefit_path) == prefit_info.get("sha256"),
        "Float64 recovery lacks its immutable uniform plan recorded before diagnostic fits",
    )
    prefit = read(prefit_path)
    planned = prefit.get("plan", {})
    require(
        prefit.get("schema") == "fixed_ridge_float64_pre_fit_plan_v1"
        and prefit.get("recovery_id") == revision["recovery_id"]
        and prefit.get("owner") == journal["process_identity"]
        and prefit.get("implementation_sha256") == CONTROL_FLOAT64_TOOL_SHA
        and prefit.get("selected_model_fits") == prefit.get("diagnostic_fits_performed") == 0
        and planned.get("schema") == CONTROL_FLOAT64_SCHEMA
        and planned.get("policy") == CONTROL_FLOAT64_POLICY
        and planned.get("status") == "verified_plan_no_fits"
        and planned.get("diagnostic_fits_by_dataset") == expected["diagnostic_fits_by_dataset"]
        and planned.get("total_diagnostic_fits") == 99
        and planned.get("diagnostic_fits_performed")
        == planned.get("selected_model_fits")
        == planned.get("parameter_searches")
        == 0
        and set(planned.get("datasets", {})) == {"volume", "native_mhc10"}
        and prefit.get("protected_artifact_hashes", {}).get(expected["dataset_kind"]) == protected
        and history[0].get("diagnostic_fits_completed") == expected["diagnostic_fits"]
        and history[0].get("total_recovery_diagnostic_fits") == 99,
        "Float64 uniform pre-fit scope, protected artifacts or completed fit accounting changed",
    )
    required_joint = dict(
        schema="fixed_ridge_float64_joint_completion_v1",
        status="all99_diagnostic_fits_verified",
        recovery_id=revision["recovery_id"],
        policy=CONTROL_FLOAT64_POLICY,
        implementation_sha256=CONTROL_FLOAT64_TOOL_SHA,
        owner=journal["process_identity"],
        required_seeds=SEEDS,
        designs=expected["designs"],
        diagnostic_fits_by_dataset=expected["diagnostic_fits_by_dataset"],
        total_diagnostic_fits=99,
        selected_model_fits=0,
        parameter_searches=0,
        old_selected_sources_models_predictions_unchanged=True,
        old_controls_preserved=True,
        original_strict_guards_passed=True,
    )
    require(
        all(joint.get(k) == value for k, value in required_joint.items())
        and isinstance(joint.get("datasets"), dict)
        and set(joint["datasets"]) == {"volume", "native_mhc10"},
        "Float64 recovery did not verify all99 fixed diagnostic fits across both tasks",
    )
    for dataset_kind, item in joint["datasets"].items():
        dataset_native = dataset_kind == "native_mhc10"
        dataset_root = Path(item["output_root"]).resolve()
        dataset_config = Path(item["config_path"]).resolve()
        configuration = read(dataset_config)
        copied_joint = (
            dataset_root / "report_recovery" / evidence_path.parent.name / joint_path.name
        )
        require(
            sha(dataset_config) == item["config_sha256"]
            and Path(configuration["output_root"]).resolve() == dataset_root
            and item.get("producer_sha256")
            == CONTROL_PRODUCER_SHAS["native" if dataset_native else "volume"]
            and item.get("training_source_sha256") == CONTROL_TRAINING_SHA
            and item.get("diagnostic_fits") == (54 if dataset_native else 45)
            and sha(copied_joint) == joint_info["sha256"]
            and len(item.get("seeds", [])) == 3
            and {row["seed"] for row in item["seeds"]} == set(SEEDS),
            "Float64 joint task/config/source/fit scope or identical joint copies changed",
        )
        planned_dataset = planned["datasets"][dataset_kind]
        peer_plan_path = copied_joint.parent / "pre_fit_plan.json"
        require(
            item.get("pre_fit_plan")
            == {"path": str(peer_plan_path), "sha256": prefit_info["sha256"]}
            and sha(peer_plan_path) == prefit_info["sha256"]
            and all(
                planned_dataset.get(key) == item[key]
                for key in (
                    "output_root",
                    "config_path",
                    "config_sha256",
                    "old_report_sha256",
                    "old_journal_sha256",
                )
            )
            and planned_dataset.get("protected_files")
            == len(prefit["protected_artifact_hashes"][dataset_kind])
            and len(planned_dataset.get("seeds", [])) == 3
            and {row["seed"] for row in planned_dataset["seeds"]} == set(SEEDS)
            and all(
                sha(path) == value
                for path, value in prefit["protected_artifact_hashes"][dataset_kind].items()
            ),
            "Float64 both-task pre-fit plan or original protected files changed",
        )
        if dataset_kind == expected["dataset_kind"]:
            require(
                dataset_root == Path(entry["output_root"]).resolve()
                and dataset_config == Path(entry["config_path"]).resolve()
                and item["config_sha256"] == entry["config_sha256"]
                and item["old_report_sha256"] == revision["old_report_sha256"]
                and item["old_journal_sha256"] == revision["old_journal_sha256"]
                and item["seeds"] == seeds,
                "Float64 joint completion differs from this frozen controller",
            )
        for row in item["seeds"]:
            for field in ("controls", "evidence"):
                file_path = contained(copied_joint.parent, row[field + "_path"])
                require(
                    file_path.parent == copied_joint.parent
                    and file_path.name == f"seed_{row['seed']}_{field}.json"
                    and sha(file_path) == row[field + "_sha256"],
                    "Float64 joint completion contains missing/corrupt seed sidecars",
                )
    argv = journal["process_identity"].get("argv", [])
    tools = [
        Path(value)
        for value in argv
        if Path(value).name == "peptide_modeling_control_float64_recovery.py"
    ]
    require(
        len(tools) == 1
        and sha(tools[0]) == CONTROL_FLOAT64_TOOL_SHA
        and argv.count("--apply") == 1
        and "--plan-only" not in argv
        and "--verify-only" not in argv
        and len(argv[argv.index(str(tools[0])) + 1 :]) == 5
        and all(
            report_repair_argument(
                journal["process_identity"],
                "--native-config" if name == "native_mhc10" else "--volume-config",
                item["config_path"],
            )
            for name, item in joint["datasets"].items()
        ),
        "Float64 report owner is not the released two-task diagnostic recovery command",
    )
    return dict(
        revision=revision,
        evidence=evidence,
        evidence_path=evidence_path,
        backup_paths=[old_report_path, old_journal_path, joint_path, evidence_path, prefit_path],
        protected=normalized,
        required_protected={str(Path(entry["config_path"]).resolve()): entry["config_sha256"]},
        seeds={item["seed"]: item for item in seeds},
        verified_seeds=[],
        old_float32_control_failures=[],
        old_report=old_report,
        prefit=planned["datasets"][expected["dataset_kind"]],
    )


def verify_control_float64_seed_files(
    context, run, source, result, result_path, report, check, kind, external=None
):
    seed = report["seed"]
    item = context["seeds"][seed]
    root = context["evidence_path"].parent
    evidence_path = contained(root, item["evidence_path"])
    controls_path = contained(root, item["controls_path"])
    require(
        evidence_path.parent == controls_path.parent == root
        and evidence_path.name == f"seed_{seed}_evidence.json"
        and controls_path.name == f"seed_{seed}_controls.json"
        and sha(evidence_path) == item["evidence_sha256"]
        and sha(controls_path) == item["controls_sha256"]
        and report.get("float64_control_evidence")
        == {"path": str(evidence_path), "sha256": item["evidence_sha256"]},
        "Float64 per-seed evidence/control path or SHA changed",
    )
    evidence, controls = read(evidence_path), read(controls_path)
    require(
        controls == report.get("float64_fixed_ridge_controls"),
        "Float64 report does not contain its exact saved diagnostic controls",
    )
    input_path = contained(run, result["artifacts"]["input_contract"])
    contract_path = contained(run, result["artifacts"]["model_contract"])
    inputs, contract = read(input_path), read(contract_path)
    if kind == "volume":
        verify_control_volume_training_signature(
            run, source, result, inputs, contract, read(Path(run) / "features/manifest.json")
        )
    frozen_task = inputs["task"]
    source_path = contained(run, frozen_task["table_path"])
    require(
        evidence.get("source_table_sha256") == sha(source_path)
        and evidence.get("result_sha256") == sha(result_path)
        and evidence.get("controls_path") == str(controls_path)
        and evidence.get("controls_sha256") == sha(controls_path)
        and evidence.get("controls_object_sha256") == control_float64_object_sha(controls),
        "Float64 diagnostic source/selected result/control binding changed",
    )
    planned = [item for item in context["prefit"]["seeds"] if item["seed"] == seed]
    require(
        len(planned) == 1
        and planned[0].get("source_table_sha256") == evidence["source_table_sha256"]
        and planned[0].get("result_sha256") == evidence["result_sha256"]
        and planned[0].get("planned_fits") == (18 if kind == "native" else 15)
        and planned[0].get("designs")
        == {
            name: {k: v for k, v in value.items() if k not in {"fits", "original_guard_checks"}}
            for name, value in evidence["designs"].items()
        },
        "Float64 seed/design scope changed after the uniform pre-fit plan",
    )
    original = context["old_report"]["reports"]
    original_report = next(row for row in original if row["seed"] == seed)
    if kind == "native":
        old_path = context["backup_paths"][0]
        original_controls = original_report["fixed_ridge_controls"]
    else:
        old_path = contained(run, result["artifacts"]["context_ablations"])
        original_controls = read(old_path)
    require(
        evidence.get("old_control_reference")
        == dict(
            path=str(old_path),
            sha256=sha(old_path),
            section_object_sha256=control_float64_object_sha(original_controls),
        ),
        "Float64 revision did not preserve the exact original float32 controls",
    )
    binding = control_precision_designs(run, source, result, inputs, contract, kind, external)
    require(
        evidence.get("ast_provenance") == context["evidence"]["ast_provenance"],
        "Float64 seed control AST differs from the independently checked uniform source upcast",
    )
    verified = verify_control_float64_seed(evidence, binding, source, check, controls, kind)
    required = context["required_protected"]
    for path in (
        Path(result_path),
        source_path,
        input_path,
        contract_path,
        Path(run) / "task_manifest.json",
    ):
        required[str(path.resolve())] = sha(path)
    for path, value in result["artifact_hashes"].items():
        required[str(contained(run, path))] = value["sha256"]
    feature_path = Path(run) / "features/manifest.json"
    features = read(feature_path)
    required[str(feature_path.resolve())] = sha(feature_path)
    cache = contained(
        run,
        "features/" + str(features["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1],
    )
    for relative, value in features["artifact_hashes"].items():
        required[str(contained(cache, relative))] = value
    context["verified_seeds"].append(dict(seed=seed, **verified))
    context["backup_paths"].extend([evidence_path, controls_path])
    for name, rows in binding["designs"].items():
        if kind == "native":
            original_roles = {
                role: value["predictions"]
                for role, value in original_controls[name]["evaluation_by_role"].items()
            }
        else:
            folds = {
                record_id: fold["fold"]
                for fold in check["folds"]
                for record_id in fold["validation_record_ids"]
            }
            original_roles = {
                "inner_oof": [
                    dict(record_id=row["record_id"], prediction=value, fold=folds[row["record_id"]])
                    for row, value in zip(
                        source, original_controls["ablations"][name]["predictions"], strict=True
                    )
                ]
            }
        for role, predicted in original_roles.items():
            context["old_float32_control_failures"].extend(
                dict(seed=seed, design=name, evaluation_role=role, **failure)
                for failure in control_precision_consistency(predicted, rows)
            )
    return controls, binding


def finish_control_float64_revision(entry, context):
    require(
        len(context["verified_seeds"]) == 3
        and {item["seed"] for item in context["verified_seeds"]} == set(SEEDS)
        and sum(item["diagnostic_fits"] for item in context["verified_seeds"])
        == context["revision"]["diagnostic_fits"],
        "Float64 diagnostic revision lacks all prespecified actual seed/design fits",
    )
    require(
        all(
            context["protected"].get(path) == value
            for path, value in context["required_protected"].items()
        ),
        "Float64 protected set omitted original selected/source/feature artifacts",
    )
    durable = Path(read(entry["config_path"])["durable_root"])
    root = context["evidence_path"].parent
    for path in context["backup_paths"]:
        require(
            path.parent == root
            and sha(durable / "report_recovery" / root.name / path.name) == sha(path),
            "Float64 recovery durable evidence/archive/control copy changed",
        )
    return dict(
        schema=CONTROL_FLOAT64_SCHEMA,
        policy=CONTROL_FLOAT64_POLICY,
        diagnostic_fits=context["revision"]["diagnostic_fits"],
        total_two_task_diagnostic_fits=99,
        selected_model_fits=0,
        parameter_searches=0,
        input_precision="float64_exact_upcast_of_original_float32",
        original_strict_guard_passed=True,
        old_float32_controls_qualified=False,
        old_float32_control_consistency_failures=context["old_float32_control_failures"],
        original_selected_source_models_predictions_unchanged=True,
    )


def control_volume_metadata(original, revised, selected_route):
    """Normalize only the selected route that the original trainer appends after controls."""
    require(
        original.get("selected_chemical_route") == selected_route
        and (
            "selected_chemical_route" not in revised
            or revised["selected_chemical_route"] == selected_route
        ),
        "Volume old/new control selected route differs from the saved selected result",
    )
    omitted = {"ablations", "selected_chemical_route"}
    require(
        {k: v for k, v in original.items() if k not in omitted}
        == {k: v for k, v in revised.items() if k not in omitted},
        "Volume float64 control changed other source metadata",
    )
    return True


def verify_volume_context_reporting(
    entry, task, run, result, predictions, report, control_revision=None, revised_controls=None
):
    """Keep typed volume outcomes, missing doses and unsupported OOF levels explicit."""
    manifest_path = Path(entry["manifest_path"])
    require(sha(manifest_path) == entry["manifest_sha256"], "Volume source manifest changed")
    source_tasks = [t for t in read(manifest_path)["tasks"] if t["task_id"] == task["task_id"]]
    require(len(source_tasks) == 1, "Volume source task missing or repeated")
    expected = source_tasks[0]["fixed_external_covariates"]
    path = contained(run, result["artifacts"]["model_contract"])
    require(
        sha(path) == result["artifact_hashes"][str(path)]["sha256"],
        "Volume contextual model contract SHA mismatch",
    )
    external = read(path)["fixed_external_covariates"]
    require(external == expected, "Volume context differs from frozen source contract")
    unsigned = {k: v for k, v in external.items() if k != "sha256"}
    signature = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    require(
        external.get("schema") == VOLUME_CONTEXT
        and external.get("sha256") == signature
        and external.get("chemical_method_count_increment") == 0,
        "Volume context schema/hash/method count changed",
    )
    contexts, dose_units = external["contexts_by_record"], external["dose_units"]
    ids = [r["record_id"] for r in predictions]
    require(
        len(ids) == len(set(ids)) == 20 and set(contexts) == set(ids),
        "Volume context membership changed",
    )
    keys = {
        "species",
        "route",
        "matrix",
        "volume_estimand",
        "F_basis",
        "dose_status",
        "dose_value",
        "dose_reported_unit",
    }
    bases, unknown = {}, []
    pairs = {
        "Vz": "not_divided_by_F",
        "Vss": "not_divided_by_F",
        "Vz_over_F": "divided_by_unknown_F",
    }
    for record_id, context in contexts.items():
        require(
            set(context) == keys
            and context["volume_estimand"] in pairs
            and context["F_basis"] == pairs[context["volume_estimand"]]
            and context["species"] in {"human", "rat", "monkey", "mouse"}
            and context["route"] in {"intravenous", "subcutaneous"}
            and context["matrix"] in {"plasma", "whole_blood"},
            "Volume physical quantity, F basis or source context is invalid",
        )
        if context["dose_status"] == "not_reported":
            require(
                context["dose_value"] is None and context["dose_reported_unit"] is None,
                "Unknown volume dose must remain None/None",
            )
            unknown.append(record_id)
            bases[record_id] = "not_reported"
        else:
            dose = context["dose_value"]
            require(
                context["dose_status"] == "reported"
                and isinstance(dose, (float, int))
                and not isinstance(dose, bool)
                and math.isfinite(dose)
                and dose > 0
                and context["dose_reported_unit"] in dose_units,
                "Reported volume dose is missing or invalid",
            )
            bases[record_id] = dose_units[context["dose_reported_unit"]][0]
    require(unknown == ["OLD:PV13:a71fcd2390fa1f00d593a3d7"], "Unknown-dose source scope changed")
    same_inputs = defaultdict(list)
    for row in predictions:
        key = json.dumps(contexts[row["record_id"]], sort_keys=True, separators=(",", ":"))
        same_inputs[(row["split_group"], key)].append(float(row["prediction"]))
    require(
        all(
            all(abs(p - values[0]) <= 1e-8 + 1e-7 * abs(values[0]) for p in values)
            for values in same_inputs.values()
        ),
        "Volume predictions differ for identical molecule/context inputs",
    )
    path = contained(run, result["artifacts"]["context_ablations"])
    require(
        sha(path) == result["artifact_hashes"][str(path)]["sha256"], "Volume ablation SHA mismatch"
    )
    diagnostics = read(path)
    require(
        diagnostics.get("record_ids") == ids
        and diagnostics.get("chemical_method_count_increment") == 0
        and diagnostics.get("molecular_signal_established") is False,
        "Volume diagnostic order or molecular-signal claim changed",
    )
    targets = [float(r["target"]) for r in predictions]
    group_indices = defaultdict(list)
    for i, row in enumerate(predictions):
        group_indices[row["split_group"]].append(i)
    sets = [(diagnostics, True)]
    if control_revision is not None:
        require(
            control_revision["revision"]["schema"] == CONTROL_FLOAT64_SCHEMA
            and any(item["seed"] == report["seed"] for item in control_revision["verified_seeds"])
            and isinstance(revised_controls, dict)
            and control_volume_metadata(diagnostics, revised_controls, result["selected_route"]),
            "Volume old controls need a verified uniform float64 revision to remain unqualified",
        )
        sets = [(diagnostics, False), (revised_controls, True)]
    for control_set, require_consistency in sets:
        designs = control_set.get("ablations", {})
        require(
            set(designs) == {"context_only", "molecule_only", "molecule_plus_context"},
            "Volume controls missing",
        )
        for name, design in designs.items():
            verify_volume_control_design(
                name,
                design,
                predictions,
                contexts,
                ids,
                targets,
                group_indices,
                require_consistency=require_consistency,
            )
    coverage = diagnostics.get("fold_context_coverage", [])
    verify_volume_source_support(report, predictions, contexts, bases, unknown, coverage)


def verify_volume_control_design(
    name, design, predictions, contexts, ids, targets, group_indices, require_consistency=True
):
    values = design.get("predictions", [])
    require(
        len(values) == len(ids)
        and design.get("model") == "fixed_ridge_regularization_1_not_optuna_selected",
        "Volume diagnostic control changed or omitted records",
    )
    check_metrics(design.get("metrics", {}), error_metrics(targets, values))
    errors = [
        error_metrics([targets[i] for i in index], [values[i] for i in index])
        for index in group_indices.values()
    ]
    check_metrics(
        dict(
            mae=design.get("compound_equal_weight_mae"),
            rmse=design.get("compound_equal_weight_rmse"),
        ),
        mean_molecular_errors(errors),
    )
    equal_inputs = defaultdict(list)
    for row, value in zip(predictions, values, strict=True):
        context_key = json.dumps(contexts[row["record_id"]], sort_keys=True, separators=(",", ":"))
        key = (
            (int(row["fold"]), row["split_group"])
            if name == "molecule_only"
            else (int(row["fold"]), context_key)
            if name == "context_only"
            else (int(row["fold"]), row["split_group"], context_key)
        )
        equal_inputs[key].append(value)
    require(
        not require_consistency
        or all(
            all(abs(p - same[0]) <= 1e-8 + 1e-7 * abs(same[0]) for p in same)
            for same in equal_inputs.values()
        ),
        "Volume control predictions differ within identical fold inputs",
    )


def verify_volume_source_support(report, predictions, contexts, bases, unknown, coverage):
    require(
        len(coverage) == 5 and {r["fold"] for r in coverage} == set(range(5)),
        "Volume fold support coverage missing",
    )
    unseen = []
    categorical = ("species", "route", "matrix", "volume_estimand", "F_basis", "dose_status")
    for fold in coverage:
        validation = [r for r in predictions if int(r["fold"]) == fold["fold"]]
        training = [r for r in predictions if int(r["fold"]) != fold["fold"]]
        require(
            fold.get("train_groups") == sorted({r["split_group"] for r in training})
            and fold.get("validation_groups") == sorted({r["split_group"] for r in validation}),
            "Volume support folds differ from saved OOF",
        )
        unsupported = {
            key: sorted(
                {contexts[r["record_id"]][key] for r in validation}
                - {contexts[r["record_id"]][key] for r in training}
            )
            for key in categorical
        }
        unsupported["dose_basis"] = sorted(
            {bases[r["record_id"]] for r in validation} - {bases[r["record_id"]] for r in training}
        )
        require(
            fold.get("unsupported_validation_context") == unsupported,
            "Volume unsupported fold conditions changed",
        )
        unseen.extend(
            r["record_id"]
            for r in validation
            if contexts[r["record_id"]]["volume_estimand"] in unsupported["volume_estimand"]
        )
    require(
        len(unseen) == 6 and report.get("unsupported_estimand_record_ids") == sorted(unseen),
        "Volume unseen-estimand observations hidden or changed",
    )
    require(
        report.get("unknown_dose_record_ids") == sorted(unknown),
        "Volume missing-dose observation hidden or changed",
    )
    summaries = report.get("per_estimand", [])
    counts = {"Vz": (14, 3), "Vss": (1, 1), "Vz_over_F": (5, 1)}
    require(
        len(summaries) == 3 and {x["volume_estimand"] for x in summaries} == set(counts),
        "Volume per-estimand summaries missing or repeated",
    )
    for summary in summaries:
        name = summary["volume_estimand"]
        rows = [r for r in predictions if contexts[r["record_id"]]["volume_estimand"] == name]
        groups = defaultdict(list)
        for row in rows:
            groups[row["split_group"]].append(row)
        require(
            (len(rows), len(groups))
            == counts[name]
            == (summary.get("records"), summary.get("molecules"))
            and summary.get("source_record_ids") == sorted(r["record_id"] for r in rows),
            "Volume per-estimand source membership changed",
        )
        check_metrics(
            summary.get("record_weighted", {}),
            error_metrics(
                [float(r["target"]) for r in rows], [float(r["prediction"]) for r in rows]
            ),
        )
        errors = [
            error_metrics(
                [float(r["target"]) for r in values], [float(r["prediction"]) for r in values]
            )
            for values in groups.values()
        ]
        check_metrics(summary.get("molecule_equal", {}), mean_molecular_errors(errors))
        unsupported_scope = any(r["record_id"] in unseen for r in rows)
        require(
            summary.get("validation_scope")
            == (
                "unsupported_estimand_transfer_only"
                if unsupported_scope
                else "internal_selection_cv_not_independent_validation"
            ),
            "Volume unsupported subtype predictions presented as validated CV",
        )


def verify_mhc10_native_features(root, source, task, frozen_pin=None):
    """Check the frozen chemical bank and exactly one record-keyed species context."""
    root = Path(root)
    pointer = root / "features/manifest.json"
    if frozen_pin is not None:
        require(sha(pointer) == frozen_pin["sha256"], "Native MHC10 feature pointer changed")
    features = read(pointer)
    configuration = features["configuration"]
    expected = {r["record_id"]: r["rbc_species"] for r in source}
    require(
        features.get("schema") == MHC10_NATIVE_SCHEMA
        and features.get("requested_records") == 27
        and features.get("structure_keys") == 22
        and configuration.get("schema") == MHC10_NATIVE_SCHEMA
        and configuration.get("record_ids") == [r["record_id"] for r in source]
        and configuration.get("context_by_record") == expected
        and configuration.get("context_method_increment") == 0
        and configuration.get("chemical_feature_extractions") == 0
        and configuration.get("old21_table_sha256") == MHC10_NATIVE_OLD_SHA
        and configuration.get("source_review_sha256") == MHC10_NATIVE_SOURCE_PINS["review"]
        and configuration.get("parent_pointer_sha256")
        == "333c7d7a7e0048151a7c5f913f19d646dbcfe069e6c3994d4ae9af98ac701e9c"
        and configuration.get("implementation_sha256")
        == task["scientific_definition"]["implementation_sha256"],
        "Native MHC10 feature/species/method provenance changed",
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    require(features.get("input_fingerprint") == fingerprint, "Native feature fingerprint changed")
    cache_name = str(features["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    require(cache_name == fingerprint[:16], "Native feature cache namespace changed")
    cache = contained(root, "features/" + cache_name)
    hashes = features.get("artifact_hashes", {})
    require(len(hashes) == 36, "Native complete feature/context artifact set changed")
    for relative, expected_sha in hashes.items():
        require(
            sha(contained(cache, relative)) == expected_sha, "Native feature artifact SHA changed"
        )
    for relative, expected_sha in MHC10_NATIVE_BANK_PINS.items():
        require(hashes.get(relative) == expected_sha, "Native species/context/index bytes changed")
    registry = read(root / "representation_methods.json")
    if frozen_pin is not None:
        require(
            sha(root / "representation_methods.json") == frozen_pin["registry_sha256"],
            "Native chemical method registry changed",
        )
    require(
        registry == features["representation_methods"]
        and registry.get("independent_method_count") == 10
        and registry.get("context_method_increment") == 0
        and registry.get("context_columns") == 2,
        "Native species context was counted as a chemical method",
    )
    methods = registry["methods"]
    require(
        len(methods) == 10
        and len({m["algorithm_family_id"] for m in methods}) == 10
        and all(
            m.get("independent") is True and m.get("source_kind") == "structure" for m in methods
        ),
        "Native ten independent chemical methods changed",
    )
    groups = features["groups"]
    context = groups[MHC10_NATIVE_CONTEXT]
    require(
        context.get("key_kind") == "record_id"
        and context.get("shape") == [27, 2]
        and context.get("eligibility_field") == "rbc_species_eligible"
        and context.get("feature_names") == ["human", "mouse"]
        and context.get("chemical_method") is False
        and context.get("matrix") == "rbc_species_context.npy"
        and context.get("valid") == "rbc_species_context_valid.npy"
        and context.get("index") == "rbc_species_index.tsv",
        "Native species context schema changed",
    )
    aliases = table(cache / features["record_index"])
    require(
        len(aliases) == 27
        and {r["record_id"]: r["structure_key"] for r in aliases}
        == {r["record_id"]: r["canonical_identity_group"] for r in source},
        "Native feature record/complete-graph membership changed",
    )
    indices = table(cache / context["index"])
    require(
        indices == [dict(key=r["record_id"], row_idx=str(i)) for i, r in enumerate(source)],
        "Native record-keyed species order changed",
    )
    for method in methods:
        require(
            features["routes"].get(method["route"])
            == method["group_names"] + [MHC10_NATIVE_CONTEXT]
            and MHC10_NATIVE_CONTEXT not in method["group_names"],
            "Native chemical route omits or repeats species context",
        )
    require(
        set(features["routes"]) == {m["route"] for m in methods} | {"structure_methods_all"}
        and features["routes"]["structure_methods_all"] == list(groups)
        and all(names.count(MHC10_NATIVE_CONTEXT) == 1 for names in features["routes"].values()),
        "Native fusion/route species schema changed",
    )
    for name, meta in groups.items():
        if name != MHC10_NATIVE_CONTEXT:
            require(
                meta.get("key_kind") == "canonical_identity_group" and meta["shape"][0] == 22,
                "Native chemical features no longer use whole graphs",
            )
    return features


def verify_mhc10_native_source(entry, task):
    require(
        is_mhc10_native_task(task) and task["task_id"] == MHC10_NATIVE_TASK,
        "Unsupported native MHC10 task",
    )
    definition = task["scientific_definition"]
    require(
        definition.get("canonical_quantity_id") == MHC10_NATIVE_QUANTITY
        and definition.get("mandatory_prediction_inputs") == ["rbc_species", "quantity_id"]
        and definition.get("context_group") == MHC10_NATIVE_CONTEXT
        and definition.get("context_columns") == 2
        and definition.get("context_is_chemical_method") is False
        and definition.get("molecular_reporting_record_count") == 27
        and definition.get("separate_all_common_refit_required") is True
        and definition.get("mouse_independent_molecular_groups") == 1
        and all(
            definition.get(k) is False
            for k in (
                "standardized_10percent_quantity_verified",
                "minimum_discrete_vs_interpolated_verified",
                "exposure_time_conditioned",
                "pH_conditioned",
                "mouse_effect_validated",
                "cross_molecule_species_correction_validated",
                "homologous_series_isolation_proven",
            )
        ),
        "Native MHC10 annotation/context limits were promoted",
    )
    manifest_path = Path(entry["manifest_path"])
    require(sha(manifest_path) == entry["manifest_sha256"], "Native source manifest changed")
    matched = [t for t in read(manifest_path)["tasks"] if t["task_id"] == task["task_id"]]
    require(len(matched) == 1, "Native source task missing or repeated")
    frozen = matched[0]
    require(
        frozen.get("scientific_definition") == definition
        and frozen.get("canonical_quantity_id") == MHC10_NATIVE_QUANTITY,
        "Native frozen scientific contract changed",
    )
    implementation = Path(entry["template"]) / "scripts/peptide_modeling_mhc10_native_bridge.py"
    require(
        sha(implementation) == definition.get("implementation_sha256"),
        "Native MHC10 implementation changed",
    )
    scope = dict(
        original_endpoint="MHC10",
        target_endpoint="MHC10",
        quantity_id=MHC10_NATIVE_QUANTITY,
        original_labelled_records=6,
        included_original_records=6,
        original_record_ids=sorted(MHC10_NATIVE_POINTS),
        included_original_molecular_groups=1,
        pooled_records=27,
        pooled_molecular_groups=22,
        species_counts={"human": 25, "mouse": 2},
        global_HC10_alias=False,
        unified_10percent_quantity_verified=False,
        full_refit_required_for_training_coverage=True,
    )
    descriptor = task.get("native_source_observation_coverage", {})
    require(
        isinstance(descriptor, dict)
        and all(descriptor.get(k) == v for k, v in scope.items())
        and descriptor == frozen.get("native_source_observation_coverage"),
        "Native six-of-six participation became a false canonical HC10 mapping",
    )
    proof_path = contained(entry["template"], descriptor["proof_path"])
    require(sha(proof_path) == descriptor["proof_sha256"], "Native source proof SHA changed")
    proof = read(proof_path)
    require(
        isinstance(proof, dict)
        and proof.get("schema") == "mhc10_native_observation_source_coverage_v1"
        and proof.get("task_id") == MHC10_NATIVE_TASK
        and all(proof.get(k) == v for k, v in scope.items())
        and proof.get("implementation_sha256") == definition["implementation_sha256"],
        "Native source proof quantity or scope changed",
    )
    source_path = contained(entry["template"], task["table_relative"])
    require(
        sha(source_path) == task["table_sha256"] == proof["task_table"]["sha256"]
        and contained(entry["template"], proof["task_table"]["path"]) == source_path,
        "Native source table differs from its proof",
    )
    source = table(source_path)
    ids = [r["record_id"] for r in source]
    by_id = {r["record_id"]: r for r in source}
    graph_groups = {}
    require(
        len(source) == len(by_id) == 27 and proof["record_ids"] == ids,
        "Native source27 observation membership changed",
    )
    for row in source:
        graph = row["canonical_smiles"]
        require(
            bool(graph)
            and row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
            and graph_groups.setdefault(graph, row["split_group"]) == row["split_group"]
            and row["canonical_quantity_id"] == MHC10_NATIVE_QUANTITY
            and row["native_metric_annotation"] == "MHC10"
            and row["unit"] == row["raw_unit"] == "uM"
            and row["raw_relation"] == "="
            and row["target_transform"] == "log10"
            and row["rbc_species"] in {"human", "mouse"}
            and row["rbc_species_eligible"] == "true"
            and row["structure_eligible"] == "true"
            and row["rbc_material"] == "erythrocytes"
            and row["minimum_discrete_vs_interpolated_verified"] == "false"
            and row["standardized_10percent_quantity_verified"] == "false"
            and row["explicit_10percent_readout"]
            == str(row["record_id"] == MHC10_QUALIFIED_OBSERVATION).lower()
            and math.isclose(
                float(row["target"]), math.log10(float(row["value"])), rel_tol=0, abs_tol=1e-10
            ),
            "Native source graph, exact annotation or unknown readout scope changed",
        )
        if row["record_id"] in MHC10_NATIVE_POINTS:
            value, species, pmid, ph = MHC10_NATIVE_POINTS[row["record_id"]]
            require(
                Decimal(row["value"]) == Decimal(value)
                and row["rbc_species"] == species
                and row["source_pmid"] == pmid
                and row["reported_pH"] == ph
                and row["reported_exposure_h"] == "1"
                and row["source_original_endpoint"] == "MHC10"
                and row["source_stratum"] == "Melittin_CHEMBL"
                and row["canonical_identity_group"] == MHC10_NATIVE_MEL_GRAPH
                and row["split_group"] == MHC10_NATIVE_MEL_GROUP,
                "An original native MHC10 scalar/species/graph changed",
            )
        else:
            require(
                row["source_original_endpoint"] == "hemolysis_MHC10"
                and row["rbc_species"] == "human"
                and row["source_pmid"] == "23894079"
                and row["reported_pH"] == ""
                and row["reported_exposure_h"] == ""
                and row["source_stratum"] == "DRAMP21_PMID23894079",
                "Native DRAMP annotation gained an invented condition or metric",
            )
    require(
        len(graph_groups) == len(set(graph_groups.values())) == 22
        and Counter(r["rbc_species"] for r in source) == {"human": 25, "mouse": 2}
        and sum(r["split_group"] == MHC10_NATIVE_MEL_GROUP for r in source) == 6,
        "Native27/22 or single-family mouse support changed",
    )
    old_path = contained(entry["template"], proof["old21"]["path"])
    require(
        sha(old_path) == proof["old21"]["sha256"] == MHC10_NATIVE_OLD_SHA,
        "Native old21 table snapshot changed",
    )
    old = {r["record_id"]: r for r in table(old_path)}
    require(
        len(old) == 21
        and set(old) == set(by_id) - set(MHC10_NATIVE_POINTS)
        and all(all(by_id[i].get(k) == v for k, v in prior.items()) for i, prior in old.items()),
        "Native old21 source fields were changed",
    )
    require(
        set(proof["source_files"]) == set(MHC10_NATIVE_SOURCE_PINS),
        "Native source package omitted evidence",
    )
    for name, expected_sha in MHC10_NATIVE_SOURCE_PINS.items():
        item = proof["source_files"][name]
        require(
            sha(contained(entry["template"], item["path"])) == item["sha256"] == expected_sha,
            "Native source archive/DRAMP/review evidence SHA changed",
        )
    require(
        set(proof["observations"]) == set(by_id), "Native per-observation proof membership changed"
    )
    for record_id, item in proof["observations"].items():
        path = contained(entry["template"], item["path"])
        require(sha(path) == item["sha256"], "Native observation snapshot changed")
        obj, row = read(path), by_id[record_id]
        if record_id in MHC10_NATIVE_POINTS:
            require(
                item["sha256"] == MHC10_NATIVE_OBJECT_PINS[record_id],
                "An original six-record raw source object changed",
            )
            observed, assay, reference, compound = (
                obj[k] for k in ("observation", "assay", "reference", "compound")
            )
            frozen_row = obj["frozen_record"]
            require(
                observed["endpoint"] == "MHC10"
                and observed["qualifier"] == "="
                and observed["unit"] == "uM"
                and observed["observation_id"] == record_id.split(":")[-1]
                and Decimal(observed["value"]) == Decimal(row["value"])
                and observed["assay_id"] == assay["assay_id"] == row["source_assay_id"]
                and observed["reference_id"] == reference["reference_id"]
                and reference["pmid"] == row["source_pmid"]
                and observed["compound_id"] == compound["compound_id"]
                and compound["smiles"] == row["canonical_smiles"]
                and assay["description"] == row["reported_readout"]
                and assay["system_description"]
                == ("Homo sapiens" if row["rbc_species"] == "human" else "Mus musculus")
                and frozen_row["record_id"] == record_id
                and frozen_row["endpoint"] == "MHC10"
                and frozen_row["raw_value"] == observed["value"]
                and frozen_row["raw_relation"] == "="
                and frozen_row["raw_unit"] == "uM",
                "Native original archive observation linkage changed",
            )
        else:
            mapping, xlsx, original = (
                obj["verified_mapping"],
                obj["xlsx_source_row"],
                obj["frozen_record"],
            )
            match = re.fullmatch(
                r"\s*\[Ref\.23894079\]\s*MHC10\s*=\s*(\d+(?:\.\d+)?)\s*[µμu]M\s+against\s+human\s+red\s+blood\s+cells\s*",
                str(xlsx["Hemolytic_activity"]),
                re.I,
            )
            require(
                match is not None
                and Decimal(match[1]) == Decimal(row["value"])
                and xlsx["DRAMP_ID"] == row["source_entry_id"]
                and xlsx["Sequence"] == row["sequence"]
                and hashlib.sha256(str(xlsx["Hemolytic_activity"]).strip().encode()).hexdigest()
                == mapping["source_annotation_sha256"]
                and hashlib.sha256(original["raw_json"].encode()).hexdigest()
                == mapping["catalog_raw_json_sha256"]
                and mapping["record_id"] == record_id
                and mapping["raw_relation"] == "="
                and mapping["raw_unit"] == "uM"
                and mapping["species"] == "human"
                and obj["old_task_row"] == old[record_id],
                "Native DRAMP source annotation linkage changed",
            )
    split_path = contained(entry["template"], proof["predeclared_splits"]["path"])
    require(
        sha(split_path) == proof["predeclared_splits"]["sha256"], "Native frozen split plan changed"
    )
    checks = read(split_path)
    require(
        len(checks["seeds"]) == 3
        and {c["seed"] for c in checks["seeds"]} == set(SEEDS)
        and checks.get("refit_full_required") is True
        and checks.get("final_training_records") == 27
        and checks.get("holdout_never_scores_all_common_refit") is True,
        "Native split/refit protocol changed",
    )
    for check in checks["seeds"]:
        train, held = check["train_record_ids"], check["holdout_record_ids"]
        require(
            len(train) == len(set(train))
            and len(held) == len(set(held))
            and set(train).isdisjoint(held)
            and set(train) | set(held) == set(ids)
            and len({by_id[i]["split_group"] for i in train}) == 17
            and len({by_id[i]["split_group"] for i in held}) == 5
            and {by_id[i]["split_group"] for i in train}.isdisjoint(
                {by_id[i]["split_group"] for i in held}
            )
            and check["final_refit_record_ids"] == ids
            and check["actual_folds"] == 5,
            "Native outer molecule/refit membership changed",
        )
        folds = check["folds"]
        validation = [i for f in folds for i in f["validation_record_ids"]]
        require(
            len(folds) == 5
            and {f["fold"] for f in folds} == set(range(5))
            and len(validation) == len(set(validation)) == len(train)
            and set(validation) == set(train),
            "Native actual inner5 membership changed",
        )
        for fold in folds:
            require(
                set(fold["train_record_ids"]) == set(train) - set(fold["validation_record_ids"])
                and {by_id[i]["split_group"] for i in fold["train_record_ids"]}.isdisjoint(
                    {by_id[i]["split_group"] for i in fold["validation_record_ids"]}
                ),
                "Native molecular inner-fold leakage",
            )
        require(
            sha(contained(entry["template"], check["split_path"])) == check["split_sha256"],
            "Native frozen outer split SHA changed",
        )
    verify_mhc10_native_features(entry["template"], source, task, entry["native_feature_contract"])
    return descriptor, proof, frozen, source, checks


def verify_mhc10_native_refit(run, result, task, source):
    paths, _ = verify_hashes(
        run,
        result,
        {
            "model",
            "refit_model",
            "oof",
            "holdout",
            "outer_split",
            "input_contract",
            "model_contract",
        },
    )
    require(
        result.get("status") == "completed"
        and result.get("refit_status") == "completed"
        and result.get("deployment_ready") is True
        and result.get("scope_label") == "common"
        and paths["model"] != paths["refit_model"],
        "Native MHC10 requires a separate all27 refit",
    )
    attempts = result.get("refit_attempts", [])
    require(
        isinstance(attempts, list)
        and attempts
        and all(isinstance(a, dict) for a in attempts)
        and attempts[-1].get("status") == "completed",
        "Native completed-refit attempt missing",
    )
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    require(
        isinstance(contract, dict) and isinstance(inputs, dict),
        "Native refit contracts are not objects",
    )
    matched = [
        t
        for t in read(Path(run) / "task_manifest.json")["tasks"]
        if t["task_id"] == task["task_id"]
    ]
    ids = [r["record_id"] for r in source]
    require(
        len(matched) == 1
        and len(ids) == len(set(ids)) == 27
        and set(MHC10_NATIVE_POINTS) <= set(ids)
        and contract.get("common_record_ids") == ids
        and contract.get("task") == inputs.get("task") == matched[0]
        and contract.get("model_fit_scope") == "outer_train_only"
        and contract.get("optional_refit_fit_scope") == "all_common_after_evaluation"
        and contract.get("refit_status") == "completed"
        and contract.get("deployment_ready") is True
        and contract.get("refit_attempts") == attempts
        and contract.get("holdout_not_used_for_selection") is True
        and contract.get("target_unit") == "uM"
        and contract.get("target_transform") == "log10"
        and contract.get("selected_route") == result.get("selected_route")
        and contract.get("selected_model") == result.get("selected_model"),
        "Native all27/source-six training membership or evaluation scope changed",
    )
    require(
        inputs["config"].get("refit_full") is True
        and inputs["config"].get("inner_folds") == 5
        and inputs["config"].get("search_backend") == "optuna"
        and inputs.get("implementation_sha256")
        == sha(Path(run) / "scripts/peptide_modeling_train.py")
        and inputs.get("input_hash") == contract.get("input_hash") == result.get("input_hash"),
        "Native final training implementation/input changed",
    )
    features = verify_mhc10_native_features(run, source, task)
    route = result["selected_route"]
    require(route in features["routes"], "Native selected feature route is unsupported")
    names = [
        group + ":" + name
        for group in features["routes"][route]
        for name in features["groups"][group]["feature_names"]
    ]
    signature = inputs["feature_hashes"][route]
    require(
        signature.get("names") == contract.get("feature_names") == names
        and names[-2:] == MHC10_NATIVE_CONTEXT_NAMES
        and all(names.count(name) == 1 for name in MHC10_NATIVE_CONTEXT_NAMES)
        and signature.get("mask_sha256") == hashlib.sha256(b"\x01" * 27).hexdigest()
        and re.fullmatch(r"[0-9a-f]{64}", signature.get("matrix_sha256", "")),
        "Native final feature membership or single species context changed",
    )
    outer = read(paths["outer_split"])
    value = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=matched[0],
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=inputs["config"],
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=outer["input_hash"],
    )
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    require(digest == result["input_hash"], "Native all27 final training signature changed")
    return paths, contract, inputs


def check_mhc10_native_weighted(reported, predictions):
    require(isinstance(reported, dict), "Native weighted diagnostic is not an object")
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["split_group"]].append(row)
    require(
        reported.get("records") == len(predictions)
        and reported.get("molecular_groups") == len(grouped),
        "Native diagnostic counts changed",
    )
    if not predictions:
        require(
            reported.get("record_weighted") is None
            and reported.get("molecule_equal") is None
            and not reported.get("per_molecule"),
            "Native empty stratum gained observations or metrics",
        )
        return
    values = error_metrics(
        [float(r["target"]) for r in predictions], [float(r["prediction"]) for r in predictions]
    )
    check_metrics(reported.get("record_weighted"), values)
    require(
        reported["record_weighted"].get("n") == len(predictions),
        "Native metric sample count changed",
    )
    molecules = reported.get("per_molecule", {})
    require(
        isinstance(molecules, dict) and set(molecules) == set(grouped),
        "Native molecular strata changed",
    )
    individual = []
    for group, rows in grouped.items():
        computed = error_metrics(
            [float(r["target"]) for r in rows], [float(r["prediction"]) for r in rows]
        )
        check_metrics(molecules[group], computed)
        require(molecules[group].get("n") == len(rows), "Native molecular sample count changed")
        individual.append(computed)
    check_metrics(reported.get("molecule_equal"), mean_molecular_errors(individual))


def verify_mhc10_native_evaluation(
    report, source, inner, held, check, control=None, unqualified_original_controls=None
):
    """Derive every support flag and stratum from actual frozen molecular folds."""
    if unqualified_original_controls is not None:
        require(
            control in {"molecule_only", "context_only", "molecule_plus_context"}
            and unqualified_original_controls["revision"]["schema"] == CONTROL_FLOAT64_SCHEMA
            and any(
                item["seed"] == check["seed"]
                for item in unqualified_original_controls["verified_seeds"]
            ),
            "Only archived fixed controls may retain explicitly unqualified precision failures",
        )
    require(
        isinstance(report, dict)
        and report.get("independently_evaluated_refit") is False
        and report.get("mouse_effect_validated") is False
        and report.get("native_annotation_not_unified_HC10") is True,
        "Native annotation evaluation became a biophysical or refit performance claim",
    )
    by_id = {r["record_id"]: r for r in source}
    predictions = list(inner) + list(held)
    require(
        len(predictions) == len({r["record_id"] for r in predictions}) == 27,
        "Native evaluation must cover each source observation once",
    )
    source_membership(predictions, source, require_all=True)
    require(
        {r["split_group"] for r in inner}.isdisjoint({r["split_group"] for r in held}),
        "Native evaluation molecule crossed outer roles",
    )
    folds = {f["fold"]: f for f in check["folds"]}
    expected_folds = {i: f["fold"] for f in check["folds"] for i in f["validation_record_ids"]}
    roles = report.get("evaluation_by_role", {})
    require(
        isinstance(roles, dict) and set(roles) == {"inner_oof", "outer_holdout"},
        "Native OOF and holdout diagnostics were pooled or omitted",
    )
    unsupported_all, original_roles = [], {}
    for role, rows, expected_ids in (
        ("inner_oof", inner, check["train_record_ids"]),
        ("outer_holdout", held, check["holdout_record_ids"]),
    ):
        require(
            len(rows) == len(expected_ids) and {r["record_id"] for r in rows} == set(expected_ids),
            "Native evaluation role differs from the frozen split",
        )
        item = roles[role]
        require(
            isinstance(item, dict)
            and item.get("score_role")
            == ("model_selection" if role == "inner_oof" else "outer_train_model_evaluation")
            and item.get("score_applies_to_all27_refit") is False,
            "Native selected-model/refit score role changed",
        )
        saved = item.get("predictions", [])
        require(
            len(saved) == len(rows)
            and len({r["record_id"] for r in saved}) == len(rows)
            and {r["record_id"] for r in saved} == set(expected_ids),
            "Native report prediction membership changed",
        )
        actual = {r["record_id"]: r for r in rows}
        unsupported = []
        equivalent = defaultdict(list)
        for predicted in saved:
            record_id = predicted["record_id"]
            raw, source_row = actual[record_id], by_id[record_id]
            fold = int(raw["fold"])
            require(
                (role == "inner_oof" and expected_folds[record_id] == fold)
                or (role == "outer_holdout" and fold == -1),
                "Native actual inner/holdout fold changed",
            )
            fit_ids = (
                folds[fold]["train_record_ids"]
                if role == "inner_oof"
                else check["train_record_ids"]
            )
            require(
                source_row["split_group"] not in {by_id[i]["split_group"] for i in fit_ids},
                "Native validation molecule entered its own fit",
            )
            known = sorted({by_id[i]["rbc_species"] for i in fit_ids})
            supported = source_row["rbc_species"] in known
            is_unsupported = source_row["rbc_species"] == "mouse" and not supported
            require(
                predicted.get("evaluation_role") == role
                and int(predicted["fold"]) == fold
                and predicted.get("split_group") == source_row["split_group"]
                and float(predicted["target"]) == float(raw["target"])
                and float(predicted["prediction"]) == float(raw["prediction"])
                and all(math.isfinite(float(predicted[k])) for k in ("target", "prediction"))
                and predicted.get("species_seen_in_training") == known
                and predicted.get("species_context_supported") is supported
                and predicted.get("unsupported_mouse_context") is is_unsupported
                and predicted.get("performance_target")
                == "source_native_annotation_not_standardized10percent"
                and predicted.get("original_endpoint") == source_row["source_original_endpoint"]
                and all(
                    predicted.get(k) == source_row[k]
                    for k in (
                        "rbc_species",
                        "source_stratum",
                        "source_pmid",
                        "reported_pH",
                        "reported_exposure_h",
                    )
                ),
                "Native species support, source conditions or saved prediction changed",
            )
            if is_unsupported:
                unsupported.append(record_id)
            if record_id in MHC10_NATIVE_POINTS:
                original_roles[record_id] = role
            key = (
                (source_row["rbc_species"], fold)
                if control == "context_only"
                else (
                    (source_row["split_group"], fold)
                    if control == "molecule_only"
                    else (source_row["split_group"], source_row["rbc_species"], fold)
                )
            )
            equivalent[key].append(float(predicted["prediction"]))
        require(
            item.get("unsupported_mouse_record_ids") == unsupported,
            "Native unsupported mouse observations were omitted",
        )
        for values in equivalent.values():
            require(
                unqualified_original_controls is not None
                or all(abs(p - values[0]) <= 1e-8 + 1e-7 * abs(values[0]) for p in values),
                "Native identical-feature predictions differ within a fit",
            )
        check_mhc10_native_weighted(item, saved)
        strata = item.get("strata", {})
        require(
            isinstance(strata, dict) and set(strata) == {"rbc_species", "source_stratum"},
            "Native species/source diagnostics missing",
        )
        for field, labels in (
            ("rbc_species", ["human", "mouse"]),
            ("source_stratum", ["Melittin_CHEMBL", "DRAMP21_PMID23894079"]),
        ):
            require(set(strata[field]) == set(labels), "Native diagnostic stratum labels changed")
            for label in labels:
                check_mhc10_native_weighted(
                    strata[field][label], [r for r in saved if r[field] == label]
                )
        unsupported_all += unsupported
    require(
        set(unsupported_all) == {i for i, r in by_id.items() if r["rbc_species"] == "mouse"}
        and report.get("original_MHC10_evaluation_roles") == original_roles,
        "Native single-family mouse support or original-six evaluation roles changed",
    )
    contexts = report.get("source_context_by_fold", [])
    require(
        len(contexts) == 5 and {c["fold"] for c in contexts} == set(range(5)),
        "Native source context diagnostics omit actual folds",
    )
    for context in contexts:
        frozen = folds[context["fold"]]
        trained = sorted({by_id[i]["rbc_species"] for i in frozen["train_record_ids"]})
        validation = sorted({by_id[i]["rbc_species"] for i in frozen["validation_record_ids"]})
        require(
            context.get("train_record_ids") == frozen["train_record_ids"]
            and context.get("validation_record_ids") == frozen["validation_record_ids"]
            and context.get("train_species") == trained
            and context.get("validation_species") == validation
            and context.get("unsupported_species") == sorted(set(validation) - set(trained)),
            "Native species support is not derived from actual molecular folds",
        )


def verify_mhc10_native_reporting(entry, journal, task):
    descriptor, proof, frozen, source, checks = verify_mhc10_native_source(entry, task)
    receipt = journal.get("molecular_reporting", {})
    require(
        isinstance(receipt, dict) and receipt.get("status") == "verified",
        "Native MHC10 report missing",
    )
    path = contained(entry["output_root"], receipt["path"])
    require(
        path.name == "mhc10_native_molecular_diagnostics.json"
        and sha(path) == receipt.get("sha256"),
        "Native report path/SHA/implementation changed",
    )
    payload = read(path)
    precision = verify_control_float64_revision(entry, journal, receipt, payload, "native")
    require(
        receipt.get("implementation_sha256")
        == (
            CONTROL_FLOAT64_TOOL_SHA
            if precision is not None
            else task["scientific_definition"]["implementation_sha256"]
        ),
        "Native selected-source and diagnostic-report implementation provenance changed",
    )
    require(
        isinstance(payload, dict)
        and payload.get("schema") == MHC10_NATIVE_REPORT
        and payload.get("status") == "verified"
        and payload.get("task_id") == MHC10_NATIVE_TASK
        and payload.get("quantity_id") == MHC10_NATIVE_QUANTITY
        and payload.get("required_seeds") == SEEDS
        and payload.get("original_MHC10_ids") == sorted(MHC10_NATIVE_POINTS)
        and payload.get("original_MHC10_records") == 6
        and payload.get("original_MHC10_molecular_groups") == 1
        and payload.get("canonical_HC10_mapping_changed") is False
        and payload.get("mouse_effect_validated") is False
        and payload.get("model_selection_repeated") is False,
        "Native participation report became a canonical HC10 or validated-species claim",
    )
    reports = payload.get("reports", [])
    require(
        isinstance(reports, list)
        and len(reports) == 3
        and all(isinstance(r, dict) for r in reports)
        and {r["seed"] for r in reports} == set(SEEDS),
        "Native required seed reports missing or repeated",
    )
    rounds = {r["seed"]: r for r in journal["rounds"]}
    for report in reports:
        run = contained(entry["output_root"], rounds[report["seed"]]["run_dir"])
        result_path = run / "results/full" / (task["task_id"] + "_common.json")
        require(sha(result_path) == report.get("result_sha256"), "Native diagnostic result changed")
        result = read(result_path)
        current = table(run / task["table_relative"])
        require(
            current == source
            and sha(run / task["table_relative"])
            == task["table_sha256"]
            == report.get("source_table_sha256"),
            "Native report source table changed",
        )
        verify_mhc10_native_features(run, current, task, entry["native_feature_contract"])
        paths, contract, inputs = verify_mhc10_native_refit(run, result, task, current)
        require(
            inputs["task"] == frozen
            and report.get("selected_route") == result["selected_route"]
            and report.get("selected_model") == result["selected_model"]
            and report.get("source_proof_sha256") == descriptor["proof_sha256"]
            and report.get("oof_sha256") == sha(paths["oof"])
            and report.get("holdout_sha256") == sha(paths["holdout"])
            and report.get("model_sha256") == sha(paths["model"]),
            "Native selected-model/OOF/source proof binding changed",
        )
        check = next(c for c in checks["seeds"] if c["seed"] == report["seed"])
        inner, held = actual_folds(paths["oof"]), table(paths["holdout"])
        outer = read(paths["outer_split"])
        require(
            outer.get("seed") == inputs["config"].get("seed") == report["seed"]
            and set(outer["train_records"]) == set(check["train_record_ids"])
            and set(outer["test_records"]) == set(check["holdout_record_ids"])
            and outer["record_groups"] == {r["record_id"]: r["split_group"] for r in current}
            and result.get("n_train") == len(inner)
            and result.get("n_test") == len(held),
            "Native selected-model outer/inner scopes changed",
        )
        verify_mhc10_native_evaluation(report, current, inner, held, check)
        revised_controls = None
        if precision is not None:
            revised_controls, _ = verify_control_float64_seed_files(
                precision, run, current, result, result_path, report, check, "native"
            )
        for role, key in (("inner_oof", "inner_cv"), ("outer_holdout", "holdout")):
            check_metrics(
                result[key],
                {
                    name: report["evaluation_by_role"][role]["record_weighted"][name]
                    for name in ("mae", "rmse")
                },
            )
        controls = report.get("fixed_ridge_controls", {})
        require(
            isinstance(controls, dict)
            and set(controls) == {"molecule_only", "context_only", "molecule_plus_context"},
            "Native fixed ridge controls missing or reselected",
        )
        for name, control in controls.items():
            require(
                isinstance(control, dict)
                and control.get("model") == "fixed_ridge_alpha1_not_Optuna_selected"
                and control.get("same_frozen_splits") is True
                and control.get("model_selection_repeated") is False,
                "Native control does not use the prespecified fixed ridge/splits",
            )
            verify_mhc10_native_evaluation(
                control,
                current,
                control["evaluation_by_role"]["inner_oof"]["predictions"],
                control["evaluation_by_role"]["outer_holdout"]["predictions"],
                check,
                control=name,
                unqualified_original_controls=precision,
            )
        if revised_controls is not None:
            for name, control in revised_controls.items():
                require(
                    control.get("model") == "fixed_ridge_alpha1_not_Optuna_selected"
                    and control.get("same_frozen_splits") is True
                    and control.get("model_selection_repeated") is False,
                    "Float64 native diagnostic changed the prespecified control family/splits",
                )
                verify_mhc10_native_evaluation(
                    control,
                    current,
                    control["evaluation_by_role"]["inner_oof"]["predictions"],
                    control["evaluation_by_role"]["outer_holdout"]["predictions"],
                    check,
                    control=name,
                )
        refit = report.get("separate_all_common_refit", {})
        require(
            isinstance(refit, dict)
            and refit.get("status") == "verified"
            and refit.get("records") == 27
            and refit.get("molecular_groups") == 22
            and refit.get("training_record_ids") == contract["common_record_ids"]
            and refit.get("original_MHC10_ids") == sorted(MHC10_NATIVE_POINTS)
            and refit.get("all_six_original_observations_in_training") is True
            and contained(run, refit["model_path"]) == paths["refit_model"]
            and refit.get("model_sha256") == sha(paths["refit_model"])
            and refit.get("model_contract_sha256") == sha(paths["model_contract"])
            and refit.get("fit_scope") == "all_common_after_evaluation"
            and refit.get("independently_evaluated_on_these27") is False,
            "Native source-six participation lacks a separate all27 training refit",
        )
    durable = read(entry["config_path"])["durable_root"]
    require(
        sha(contained(durable, receipt["durable_path"]))
        == receipt.get("durable_sha256")
        == receipt["sha256"],
        "Native mandatory report durable copy changed",
    )
    return dict(
        status="verified",
        path=str(path),
        sha256=receipt["sha256"],
        required_seed_reports=3,
        source_records_per_seed=27,
        molecular_groups_per_seed=22,
        all_six_originals_full_refit_verified=3,
        native_annotation_only=True,
        unified_10percent_quantity_verified=False,
        mouse_effect_validated=False,
        independent_mouse_molecular_groups=1,
        **(
            {"diagnostic_precision_revision": finish_control_float64_revision(entry, precision)}
            if precision is not None
            else {}
        ),
    )


def verify_mhc10_native_coverage(entry, journal, task, state):
    descriptor, _, _, _, _ = verify_mhc10_native_source(entry, task)
    evidence = dict(
        **descriptor,
        task_id=task["task_id"],
        source_proof_verified=True,
        coverage_kind="native_MHC10_annotation_observation_participation",
        native_annotation_only=True,
        canonical_HC10_mapping_changed=False,
        mouse_effect_validated=False,
        independent_mouse_molecular_groups=1,
        required_seed_full_refit_training_verified=0,
        state="native_source_observation_"
        + {"trained_three_seeds": "unverified"}.get(state, state),
    )
    if state == "trained_three_seeds":
        verify_mhc10_native_reporting(entry, journal, task)
        evidence.update(
            state="native_source_observation_trained", required_seed_full_refit_training_verified=3
        )
    return evidence


def verify_hc50_coverage_source(entry, task):
    """Bind one corrected source observation; the uncorrected Free/Free identity stays out."""
    require(is_hc50_primary_task(task), "Unsupported primary-corrected HC50 task")
    definition = task["scientific_definition"]
    require(
        definition.get("canonical_quantity_id") == HC50_PRIMARY_QUANTITY
        and definition.get("molecular_reporting_record_count") == 34
        and definition.get("separate_all_common_refit_required") is True
        and definition.get("all_records_fitted") is False
        and definition.get("corrected_estimate_extrapolated") is True
        and definition.get("corrected_exposure_conflicted") is True
        and definition.get("primary_identity_correction_required") is True
        and definition.get("exposure_time_conditioned") is False
        and definition.get("standardized_protocol_verified") is False,
        "HC50 correction/quality or final-refit contract changed",
    )
    declared = task.get("observation_endpoint_coverage")
    require(
        isinstance(declared, list) and len(declared) == 1, "One HC50 source correction is required"
    )
    descriptor = declared[0]
    scope = dict(
        original_endpoint="hemolysis_IC50",
        canonical_endpoint="hemolysis_HC50",
        canonical_quantity_id=HC50_PRIMARY_QUANTITY,
        record_ids=[HC50_PRIMARY_OBSERVATION],
        original_labelled_records=1,
        included_original_records=1,
        included_original_molecular_groups=1,
        canonical_molecular_groups=30,
        inference_alias_is_global=False,
        raw_uncorrected_identity_covered=False,
        coverage_kind="primary_source_corrected_observation_only",
    )
    require(
        isinstance(descriptor, dict) and all(descriptor.get(k) == v for k, v in scope.items()),
        "HC50 observation mapping scope changed",
    )
    manifest_path = Path(entry["manifest_path"])
    require(sha(manifest_path) == entry["manifest_sha256"], "HC50 source manifest changed")
    matched = [t for t in read(manifest_path)["tasks"] if t["task_id"] == task["task_id"]]
    require(len(matched) == 1, "HC50 source task missing or repeated")
    frozen = matched[0]
    require(
        frozen.get("observation_endpoint_coverage") == declared
        and frozen.get("scientific_definition") == definition,
        "HC50 frozen declaration changed",
    )
    path = contained(entry["template"], descriptor["proof_path"])
    require(sha(path) == descriptor["proof_sha256"], "HC50 source proof SHA mismatch")
    proof = read(path)
    require(isinstance(proof, dict), "HC50 source proof is not an object")
    require(
        proof.get("schema") == "hc50_primary_corrected_observation_coverage_v1"
        and proof.get("canonical_task_id") == task["task_id"]
        and all(proof.get(k) == v for k, v in scope.items())
        and proof.get("full_refit_required_for_each_seed_training_coverage") is True,
        "HC50 source proof scope/refit requirement changed",
    )
    table_path = contained(entry["template"], task["table_relative"])
    require(
        sha(table_path) == task["table_sha256"] == proof["canonical_source_table"]["sha256"]
        and contained(entry["template"], proof["canonical_source_table"]["path"]) == table_path
        and proof["canonical_source_table"].get("records") == 34,
        "HC50 canonical table changed",
    )
    source = table(table_path)
    ids = [r["record_id"] for r in source]
    graph_families = {}
    for row in source:
        graph = row["canonical_smiles"]
        require(
            row["canonical_identity_group"]
            == "CANON_SMILES:" + hashlib.sha256(graph.encode()).hexdigest()
            and graph_families.setdefault(graph, row["split_group"]) == row["split_group"]
            and row["canonical_quantity_id"] == HC50_PRIMARY_QUANTITY
            and row["canonical_species"] == "human"
            and row["canonical_material"] == "erythrocytes"
            and row["unit"] == "uM"
            and row["target_transform"] == "log10"
            and math.isfinite(float(row["target"])),
            "HC50 canonical identity or target changed",
        )
    require(
        len(source) == len(set(ids)) == 34
        and len(graph_families) == 30
        and len(set(graph_families.values())) == 30
        and proof["canonical_record_ids"] == ids,
        "HC50 table must contain34 observations/30 molecules",
    )
    previous = proof["parent_thirtythree"]
    previous_path = contained(entry["template"], previous["path"])
    require(
        previous.get("task_id") == "TASK_e879fa2991add3daba5f1b0e"
        and sha(previous_path) == previous["sha256"] == HC50_OLD_TABLE_SHA,
        "HC50 original33 source snapshot changed",
    )
    old = table(previous_path)
    require(
        len(old) == 33
        and [r["record_id"] for r in old] == ids[:33]
        and all(
            all(row.get(k) == v for k, v in original.items())
            and row.get("estimate_method") == "unknown_in_frozen_contract"
            for row, original in zip(source[:33], old, strict=True)
        ),
        "HC50 original33 source fields or unknown fit status changed",
    )
    objects = proof["source_objects"]
    require(set(objects) == set(HC50_SOURCE_PINS), "HC50 primary/raw source object omitted")
    native = {}
    for name, expected in HC50_SOURCE_PINS.items():
        item = objects[name]
        source_path = contained(entry["template"], item["path"])
        require(
            sha(source_path) == item["sha256"] == expected, "HC50 primary/raw source SHA changed"
        )
        if name != "article_pdf":
            native[name] = read(source_path)
    raw, evidence, identity = (native[k] for k in ("original_record", "evidence", "identity"))
    require(
        raw["record_id"] == HC50_PRIMARY_OBSERVATION
        and raw["endpoint"] == "hemolysis_IC50"
        and raw["source_entry_id"] == "DRAMP32143"
        and raw["reference_ids"] == "PMID:30449002"
        and raw["value"] == raw["raw_value"] == "155"
        and raw["unit"] == raw["raw_unit"] == "uM"
        and raw["raw_relation"] == "="
        and raw["audit_source_row"]["N-terminal_Modification"] == "Free"
        and raw["audit_source_row"]["C-terminal_Modification"] == "Free",
        "HC50 correction lost the original secondary observation",
    )
    corrected = [r for r in source if r["record_id"] == HC50_PRIMARY_OBSERVATION]
    require(len(corrected) == 1, "HC50 corrected observation missing or repeated")
    row = corrected[0]
    expected = dict(
        canonical_identity_group=HC50_PRIMARY_GRAPH,
        split_group=HC50_PRIMARY_FAMILY,
        source_original_endpoint="hemolysis_IC50",
        raw_source_split_group=HC50_RAW_FAMILY,
        value="155",
        reported_sd="13.04",
        reported_statistic="mean_plus_minus_SD",
        reported_donors="5",
        raw_relation="=",
        raw_unit="uM",
        primary_N_terminus="acetylated",
        primary_C_terminus="amidated",
        raw_N_terminus="Free",
        raw_C_terminus="Free",
        primary_identity_correction_applied="true",
        estimate_method="source_reported_estimate",
        extrapolation_status="inferred_beyond_stated_assay_range",
        exposure_evidence_status="conflicted_RBC_method1h_general_caption48h",
        mapping_evidence_status="inferred_from_IC50_table_and_percent_lysis_method",
        independently_defined_HC50="false",
        directly_observed_half_lysis_point="false",
    )
    require(
        all(row.get(k) == v for k, v in expected.items())
        and math.isclose(float(row["target"]), math.log10(155), rel_tol=0, abs_tol=1e-10)
        and proof["corrected_observation"] == row,
        "HC50 corrected scalar/caps/quality evidence changed",
    )
    correction = dict(
        secondary_N_terminus="Free",
        secondary_C_terminus="Free",
        primary_N_terminus="acetylated",
        primary_C_terminus="amidated",
        original_placeholder_family=HC50_RAW_FAMILY,
        corrected_graph=HC50_PRIMARY_GRAPH,
        corrected_family=HC50_PRIMARY_FAMILY,
        identity_tier="L_reference_parent_graph_not_sample_stereopurity_or_salt_state",
    )
    readout = dict(
        status="combined_table_and_percent_lysis_readout_inference",
        independently_defined_HC50=False,
        direct_observed_155uM_half_lysis=False,
        reported_mean="155",
        reported_SD="13.04",
        statistic="mean_plus_minus_SD",
        donors=5,
        observed_range_uM=[1.56, 100],
        maximum_tested_lysis_percent="30-35",
        extrapolation_inferred=True,
        original_curve_refitted=False,
        RBC_method_hours=1,
        general_caption_hours=48,
        exposure_conflicted=True,
    )
    require(
        proof["primary_identity_correction"] == correction
        and proof["primary_readout_mapping"] == readout
        and proof["original_secondary_feature_row"]
        == evidence["identity"]["original_frozen_feature_row"]
        and proof["original_secondary_feature_row"]["split_group"] == HC50_RAW_FAMILY
        and proof["original_secondary_feature_row"]["structure_eligible"] == "false"
        and identity["variants"][-1]["graph_key"] == HC50_PRIMARY_GRAPH
        and identity["variants"][-1]["canonical_isomeric_smiles"] == row["canonical_smiles"],
        "HC50 inferred readout or primary identity correction was promoted",
    )
    split_path = contained(entry["template"], proof["predeclared_splits"]["path"])
    require(
        sha(split_path) == proof["predeclared_splits"]["sha256"], "HC50 predeclared splits changed"
    )
    checks = read(split_path)
    require(
        checks.get("all_common_refit_after_evaluation") is True
        and checks.get("selection_holdout_not_used") is True
        and len(checks["seeds"]) == 3
        and {c["seed"] for c in checks["seeds"]} == set(SEEDS),
        "HC50 predeclared seed/refit protocol changed",
    )
    by_id = {r["record_id"]: r for r in source}
    for check in checks["seeds"]:
        train, held = check["train_record_ids"], check["holdout_record_ids"]
        require(
            len(train) == len(set(train))
            and len(held) == len(set(held))
            and set(train).isdisjoint(held)
            and set(train) | set(held) == set(ids)
            and HC50_PRIMARY_OBSERVATION in held
            and HC50_PRIMARY_OBSERVATION not in train
            and check["corrected_record_evaluation_role"] == "outer_holdout"
            and check["final_refit_record_ids"] == ids
            and check["actual_folds"] == 5
            and len({by_id[i]["split_group"] for i in train}) == 24
            and len({by_id[i]["split_group"] for i in held}) == 6,
            "HC50 predeclared holdout/training roles changed",
        )
        folds = check["folds"]
        validation = [i for f in folds for i in f["validation_record_ids"]]
        require(
            len(folds) == 5
            and {f["fold"] for f in folds} == set(range(5))
            and len(validation) == len(set(validation)) == len(train)
            and set(validation) == set(train),
            "HC50 predeclared fivefold membership changed",
        )
        for fold in folds:
            require(
                set(fold["train_record_ids"]) == set(train) - set(fold["validation_record_ids"])
                and {by_id[i]["split_group"] for i in fold["train_record_ids"]}.isdisjoint(
                    {by_id[i]["split_group"] for i in fold["validation_record_ids"]}
                ),
                "HC50 predeclared molecular fold leakage",
            )
        original_split = contained(entry["template"], check["split_path"])
        require(sha(original_split) == check["split_sha256"], "HC50 frozen outer split SHA changed")
    return descriptor, proof, frozen, row


def verify_hc50_full_refit(run, result, task, source):
    """Verify persisted all34-data training evidence without unpickling an estimator."""
    paths, _ = verify_hashes(
        run,
        result,
        {
            "model",
            "refit_model",
            "oof",
            "holdout",
            "outer_split",
            "input_contract",
            "model_contract",
        },
    )
    require(
        paths["model"] != paths["refit_model"]
        and result.get("refit_status") == "completed"
        and result.get("deployment_ready") is True
        and result.get("scope_label") == "common",
        "HC50 requires a separate completed all34 refit",
    )
    attempts = result.get("refit_attempts")
    require(
        isinstance(attempts, list)
        and attempts
        and all(isinstance(item, dict) for item in attempts)
        and attempts[-1].get("status") == "completed",
        "HC50 completed-refit attempt evidence missing",
    )
    contract, inputs = read(paths["model_contract"]), read(paths["input_contract"])
    require(
        isinstance(contract, dict) and isinstance(inputs, dict),
        "HC50 refit contracts are not objects",
    )
    ids = [r["record_id"] for r in source]
    frozen = [
        t
        for t in read(Path(run) / "task_manifest.json")["tasks"]
        if t["task_id"] == task["task_id"]
    ]
    require(
        len(frozen) == 1
        and len(ids) == len(set(ids)) == 34
        and HC50_PRIMARY_OBSERVATION in ids
        and contract.get("common_record_ids") == ids
        and contract.get("task") == frozen[0]
        and inputs.get("task") == frozen[0]
        and contract.get("refit_attempts") == attempts
        and contract.get("refit_status") == "completed"
        and contract.get("deployment_ready") is True
        and contract.get("optional_refit_fit_scope") == "all_common_after_evaluation"
        and contract.get("model_fit_scope") == "outer_train_only"
        and contract.get("holdout_not_used_for_selection") is True
        and contract.get("target_unit") == "uM"
        and contract.get("target_transform") == "log10"
        and contract.get("selected_route") == result.get("selected_route")
        and contract.get("selected_model") == result.get("selected_model"),
        "HC50 refit common-record membership or evaluation role changed",
    )
    require(
        inputs.get("config", {}).get("refit_full") is True
        and inputs.get("config", {}).get("inner_folds") == 5
        and inputs["config"].get("search_backend") == "optuna"
        and inputs.get("implementation_sha256")
        == sha(Path(run) / "scripts/peptide_modeling_train.py")
        and contract.get("input_hash") == inputs.get("input_hash") == result.get("input_hash"),
        "HC50 refit input or training implementation changed",
    )
    feature = inputs["feature_hashes"][result["selected_route"]]
    require(isinstance(feature, dict), "HC50 refit feature signature is not an object")
    require(
        feature.get("names") == contract.get("feature_names")
        and bool(feature["names"])
        and all(
            re.fullmatch(r"[0-9a-f]{64}", feature.get(k, ""))
            for k in ("matrix_sha256", "mask_sha256")
        ),
        "HC50 all34 refit feature contract missing",
    )
    split = read(paths["outer_split"])
    signature = dict(
        implementation_sha256=inputs["implementation_sha256"],
        search_implementation=inputs["search_implementation"],
        task=frozen[0],
        rows=[(r["record_id"], r["target"], r["split_group"]) for r in source],
        config=inputs["config"],
        features=inputs["feature_hashes"],
        linear_solver_policies=inputs["linear_solver_policies"],
        versions=inputs["versions"],
        outer_split=split["input_hash"],
    )
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    require(digest == result["input_hash"], "HC50 source/common training input signature changed")
    return paths, contract, inputs


def hc50_precision_object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def freeze_hc50_precision_evidence(root, journal):
    receipt = journal.get("molecular_reporting", {})
    if not isinstance(receipt, dict) or receipt.get("precision_reporting_revision") is None:
        return None
    require(
        journal.get("status") == "completed" and receipt.get("status") == "verified",
        "Precision evidence may be frozen only after a verified completed report",
    )
    revision = receipt["precision_reporting_revision"]
    require(
        isinstance(revision, dict)
        and revision.get("schema") == HC50_PRECISION_SCHEMA
        and revision.get("implementation_sha256") == HC50_PRECISION_TOOL_SHA,
        "Unsupported HC50 precision-report implementation",
    )
    path = contained(root, receipt["path"])
    require(
        sha(path) == receipt["sha256"]
        and read(path).get("precision_reporting_revision") == revision,
        "Precision report and receipt revision do not match",
    )
    evidence = contained(Path(root) / "report_recovery", revision["evidence_path"])
    backup = contained(Path(root) / "report_recovery", revision["failed_journal_backup"])
    require(
        evidence.name == "precision_evidence.json"
        and backup.name == "failed_campaign_status.json"
        and evidence.parent == backup.parent
        and evidence.parent.name.startswith("hc50_precision_")
        and sha(evidence) == revision["evidence_sha256"]
        and sha(backup) == revision["failed_journal_sha256"],
        "Precision evidence/failed-journal backup SHA or namespace changed",
    )
    return json.loads(
        json.dumps(
            dict(
                revision=revision,
                receipt_sha256=hc50_precision_object_sha(receipt),
                report_sha256=receipt["sha256"],
                previous_owned_process_identity=read(backup)["process_identity"],
                report_owner_identity=journal["process_identity"],
            )
        )
    )


def freeze_hc50_precision_registry_entry(previous_entry):
    """Pure post-exit migration from a trusted old registry entry; never rewrites a journal."""
    require(
        "hc50_precision_evidence" not in previous_entry
        and len(previous_entry["tasks"]) == 1
        and is_hc50_primary_task(previous_entry["tasks"][0]),
        "Precision registry migration needs the original HC50 entry",
    )
    journal = read(previous_entry["journal_path"])
    pin = freeze_hc50_precision_evidence(previous_entry["output_root"], journal)
    require(
        pin is not None
        and pin["previous_owned_process_identity"] == previous_entry["owned_process_identity"],
        "Precision registry migration is not bound to the old frozen controller",
    )
    entry = json.loads(json.dumps(previous_entry))
    entry.update(
        owned_process_identity=journal["process_identity"],
        state_at_freeze=journal["status"],
        journal_sha256_at_freeze=sha(previous_entry["journal_path"]),
        hc50_precision_evidence=pin,
        precision_registry_migration=dict(
            previous_entry_sha256=hc50_precision_object_sha(previous_entry),
            previous_owned_process_identity=previous_entry["owned_process_identity"],
            previous_config_sha256=previous_entry["config_sha256"],
            previous_manifest_sha256=previous_entry["manifest_sha256"],
        ),
    )
    verify_hc50_molecular_reporting(entry, journal, entry["tasks"][0])
    return entry


def verify_hc50_precision_revision(entry, journal, receipt, payload):
    revision = payload.get("precision_reporting_revision")
    require(
        isinstance(revision, dict) and receipt.get("precision_reporting_revision") == revision,
        "HC50 precision reporting revision is missing or inconsistent",
    )
    required = dict(
        schema=HC50_PRECISION_SCHEMA,
        policy=HC50_PRECISION_POLICY,
        max_allowed_binary32_ulps=32,
        engineering_budget_not_mathematical_BLAS_bound=True,
        implementation_sha256=HC50_PRECISION_TOOL_SHA,
        legacy_delegate_sha256=HC50_PRECISION_LEGACY_SHA,
        model_selection_repeated=False,
        production_model_fits=0,
        source_model_prediction_files_changed=False,
        raw_predictions_modified=False,
    )
    require(
        all(revision.get(k) == v for k, v in required.items()),
        "Unreviewed precision budget, model change or delegate",
    )
    pin = entry.get("hc50_precision_evidence")
    require(
        isinstance(pin, dict)
        and pin.get("revision") == revision
        and pin.get("report_sha256") == receipt["sha256"]
        and pin.get("receipt_sha256") == hc50_precision_object_sha(receipt)
        and pin.get("report_owner_identity")
        == journal["process_identity"]
        == entry["owned_process_identity"],
        "Precision report evidence was not frozen with its new verified owner",
    )
    require(
        receipt.get("implementation_sha256") == HC50_PRECISION_TOOL_SHA
        and receipt.get("implementation_dependencies_sha256")
        == {
            "peptide_modeling_hc10_human_bridge.py": HC50_PRECISION_NUMERIC_SHA,
            "peptide_modeling_hc50_primary_bridge.py": HC50_PRECISION_LEGACY_SHA,
        },
        "Precision report provenance must honestly distinguish the new tool and old delegates",
    )
    implementation = Path(entry["template"]) / "scripts/peptide_modeling_hc50_primary_bridge.py"
    require(
        sha(implementation)
        == HC50_PRECISION_LEGACY_SHA
        == read(Path(entry["template"]) / "features/manifest.json")["configuration"][
            "implementation_sha256"
        ],
        "Precision repair changed the old5bc feature/model reporting implementation",
    )
    tools = [
        Path(a)
        for a in journal["process_identity"].get("argv", [])
        if str(a).replace("\\", "/").endswith("/peptide_modeling_hc50_precision_report.py")
    ]
    require(
        len(tools) == 1
        and sha(tools[0]) == HC50_PRECISION_TOOL_SHA
        and report_repair_argument(journal["process_identity"], "--config", entry["config_path"]),
        "Precision reporting process is not the whitelisted standalone tool/config",
    )
    root = Path(entry["output_root"]).resolve()
    evidence_path = contained(root / "report_recovery", revision["evidence_path"])
    backup = contained(root / "report_recovery", revision["failed_journal_backup"])
    require(
        evidence_path.parent == backup.parent
        and evidence_path.parent.name.startswith("hc50_precision_")
        and sha(evidence_path) == revision["evidence_sha256"]
        and sha(backup) == revision["failed_journal_sha256"],
        "Precision source evidence or old failed journal changed",
    )
    evidence, prior = read(evidence_path), read(backup)
    require(
        evidence.get("schema") == HC50_PRECISION_SCHEMA
        and evidence.get("policy") == HC50_PRECISION_POLICY
        and evidence.get("max_allowed_binary32_ulps") == 32
        and evidence.get("engineering_budget_not_mathematical_BLAS_bound") is True
        and evidence.get("legacy_delegate_sha256") == HC50_PRECISION_LEGACY_SHA
        and evidence.get("implementation_sha256") == HC50_PRECISION_TOOL_SHA
        and evidence.get("config_sha256")
        == sha(entry["config_path"])
        == entry["config_sha256"]
        == journal.get("config_sha256")
        == prior.get("config_sha256")
        and evidence.get("prior_journal_sha256") == revision["failed_journal_sha256"]
        and evidence.get("production_model_fits") == 0
        and evidence.get("model_selection_repeated") is False
        and evidence.get("raw_predictions_modified") is False,
        "Precision evidence changed source/configuration, fitting, predictions or budget",
    )
    require(
        prior.get("status") == "requirement_incomplete"
        and prior.get("molecular_reporting", {}).get("status") == "failed"
        and prior["molecular_reporting"].get("reason") == HC50_PRECISION_FAILURE
        and prior.get("process_identity") == pin["previous_owned_process_identity"]
        and journal.get("status") == "completed"
        and journal["rounds"] == prior["rounds"]
        and all(
            journal.get(k) == v
            for k, v in prior.items()
            if k
            not in {"status", "pid", "process_identity", "molecular_reporting", "recovery_history"}
        ),
        "Precision repair is not the exact completed-training constancy-report failure",
    )
    events = journal.get("recovery_history", [])
    old_events = prior.get("recovery_history", [])
    require(
        len(events) == len(old_events) + 1 and events[:-1] == old_events,
        "Precision repair rewrote its prior history",
    )
    event = events[-1]
    require(
        event.get("kind") == HC50_PRECISION_SCHEMA
        and event.get("prior_journal_sha256") == revision["failed_journal_sha256"]
        and event.get("backup") == revision["failed_journal_backup"]
        and event.get("evidence_path") == revision["evidence_path"]
        and event.get("evidence_sha256") == revision["evidence_sha256"]
        and event.get("source_model_prediction_changes") is False
        and event.get("no_model_fit") is True,
        "Precision registry adoption lacks its exact backup/evidence chain",
    )
    old_actual = process_identity(prior["process_identity"]["pid"])
    require(
        identity_state(prior["process_identity"], old_actual) in {"exited", "pid_reused"},
        "Original HC50 precision-failed controller is still live",
    )
    protected = evidence.get("protected_artifact_hashes")
    require(
        isinstance(protected, dict) and protected,
        "Precision evidence omitted unchanged source/fitted files",
    )
    for path, expected in protected.items():
        require(
            Path(path).is_absolute() and sha(path) == expected,
            "An original precision source/model/prediction changed",
        )
    reports = evidence.get("reports", [])
    require(
        isinstance(reports, list)
        and len(reports) == 3
        and {r["seed"] for r in reports} == set(SEEDS),
        "Precision evidence omits or repeats a fitted seed",
    )
    durable = (
        Path(read(entry["config_path"])["durable_root"])
        / "report_recovery"
        / evidence_path.parent.name
    )
    require(
        sha(durable / evidence_path.name) == revision["evidence_sha256"]
        and sha(durable / backup.name) == revision["failed_journal_sha256"],
        "Durable precision evidence/backup copy differs",
    )
    normalized = {str(Path(p).resolve()): value for p, value in protected.items()}
    require(
        len(normalized) == len(protected), "Precision evidence repeats one file through aliases"
    )
    return dict(
        revision=revision,
        reports={r["seed"]: r for r in reports},
        protected=normalized,
        required_protected={str(Path(entry["config_path"]).resolve()): sha(entry["config_path"])},
        ast_evidence=hc50_precision_ast_evidence(implementation),
    )


def hc50_precision_npy(path):
    """Read only admitted primitive NPY bytes; never import NumPy or unpickle."""
    with Path(path).open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Precision feature artifact is not NPY")
        version = tuple(stream.read(2))
        require(version in {(1, 0), (2, 0), (3, 0)}, "Unsupported primitive NPY version")
        size = 2 if version == (1, 0) else 4
        count = int.from_bytes(stream.read(size), "little")
        require(0 < count <= 100000, "Invalid NPY header size")
        header = ast.literal_eval(
            stream.read(count).decode("utf-8" if version == (3, 0) else "latin1")
        )
        require(
            isinstance(header, dict)
            and header.get("fortran_order") is False
            and header.get("descr") in {"<f4", "|u1", "|b1"}
            and isinstance(header.get("shape"), tuple)
            and all(isinstance(n, int) and n > 0 for n in header["shape"]),
            "Precision input must use primitive C-order binary32/byte arrays",
        )
        data = stream.read()
    width = 4 if header["descr"] == "<f4" else 1
    require(len(data) == math.prod(header["shape"]) * width, "Precision NPY body shape changed")
    return header, data


def hc50_precision_features(run, source, result, inputs, contract):
    """Reconstruct selected float32 row bytes from the exact frozen chemical bank."""
    run = Path(run)
    features = read(run / "features/manifest.json")
    require(
        features["configuration"].get("implementation_sha256") == HC50_PRECISION_LEGACY_SHA,
        "Precision repair must retain the original5bc feature generator",
    )
    cache_name = str(features["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
    cache = contained(run, "features/" + cache_name)
    for relative, expected in features["artifact_hashes"].items():
        require(
            sha(contained(cache, relative)) == expected, "Precision frozen feature artifact changed"
        )
    alias_rows = table(contained(cache, features["record_index"]))
    aliases = {r["record_id"]: r for r in alias_rows}
    ids = [r["record_id"] for r in source]
    require(
        len(alias_rows) == len(aliases) == 34
        and set(aliases) == set(ids)
        and all(
            aliases[r["record_id"]]["structure_key"] == r["canonical_identity_group"]
            and r.get("structure_eligible") == "true"
            for r in source
        ),
        "Precision feature aliases do not bind all34 exact chemical identities",
    )
    groups = features["routes"][result["selected_route"]]
    require(
        groups and len(groups) == len(set(groups)),
        "Precision selected route repeats or lacks a group",
    )
    names, parts = [], []
    for name in groups:
        meta = features["groups"][name]
        require(
            meta.get("key_kind") == "canonical_identity_group",
            "Precision allowance forbids record/context features",
        )
        index_rows = table(contained(cache, meta["index"]))
        index = {r["key"]: int(r["row_idx"]) for r in index_rows}
        require(
            len(index_rows) == len(index) == 30 and set(index.values()) == set(range(30)),
            "Precision chemical index is not the frozen30-graph bank",
        )
        header, matrix = hc50_precision_npy(contained(cache, meta["matrix"]))
        valid_header, valid = hc50_precision_npy(contained(cache, meta["valid"]))
        require(
            header["shape"] == tuple(meta["shape"])
            and header["shape"][0] == 30
            and len(header["shape"]) == 2
            and header["shape"][1] == len(meta["feature_names"])
            and valid_header["descr"] == "|b1"
            and valid_header["shape"] == (30,),
            "Precision chemical matrix/mask shape changed",
        )
        names += [name + ":" + str(value) for value in meta["feature_names"]]
        width = header["shape"][1]
        bytes_per_row = width * (4 if header["descr"] == "<f4" else 1)
        rows = []
        for row in source:
            pos = index[row["canonical_identity_group"]]
            require(valid[pos] != 0, "A precision source row has invalid chemical features")
            data = matrix[pos * bytes_per_row : (pos + 1) * bytes_per_row]
            if header["descr"] == "|u1":
                data = struct.pack("<" + "f" * width, *data)
            else:
                require(header["descr"] == "<f4", "Precision chemical values are not binary32")
                bits = struct.unpack("<" + "I" * width, data)
                missing = sum((b & 0x7FFFFFFF) > 0x7F800000 for b in bits)
                require(
                    not any((b & 0x7FFFFFFF) == 0x7F800000 for b in bits)
                    and missing < width
                    and (
                        not missing
                        or (
                            meta.get("allow_partial_missing") is True
                            and missing / width <= meta.get("max_missing_fraction", 0)
                        )
                    ),
                    "Precision feature missingness/finite-value policy changed",
                )
            rows.append(data)
        parts.append(rows)
    require(
        names
        == contract.get("feature_names")
        == inputs["feature_hashes"][result["selected_route"]]["names"]
        and not any("context:" in n or n.startswith("rbc_species_context:") for n in names),
        "Precision selected input schema contains context or additional features",
    )
    assembled = [b"".join(part[i] for part in parts) for i in range(34)]
    width = len(names)
    require(
        all(len(row) == 4 * width for row in assembled),
        "Precision selected float32 row width changed",
    )
    record_rows = {}
    by_group = {}
    for row, data in zip(source, assembled, strict=True):
        group_value = (row["canonical_identity_group"], data)
        require(
            by_group.setdefault(row["split_group"], group_value) == group_value,
            "Duplicate precision graph inputs are not byte-identical including NaN payloads",
        )
        record_rows[row["record_id"]] = dict(
            split_group=row["split_group"],
            graph=row["canonical_identity_group"],
            row_sha256=hashlib.sha256(data).hexdigest(),
            feature_width=width,
            dtype="float32",
        )
    binding = dict(
        record_rows=record_rows,
        selected_matrix_sha256=hashlib.sha256(b"".join(assembled)).hexdigest(),
        selected_mask_sha256=hashlib.sha256(b"\x01" * 34).hexdigest(),
        feature_names=names,
        feature_width=width,
    )
    expected = inputs["feature_hashes"][result["selected_route"]]
    require(
        binding["selected_matrix_sha256"] == expected["matrix_sha256"]
        and binding["selected_mask_sha256"] == expected["mask_sha256"],
        "Precision selected matrix bytes differ from the fitted input signature",
    )
    return binding


def hc50_precision_binary32(value):
    require(math.isfinite(value), "Precision prediction is nonfinite")
    try:
        packed = struct.pack("!f", value)
    except (OverflowError, struct.error) as exc:
        raise ValueError("Precision prediction exceeds binary32 range") from exc
    require(struct.unpack("!f", packed)[0] == value, "Precision prediction is not exact binary32")
    bits = struct.unpack("!I", packed)[0]
    return 0x80000000 - (bits & 0x7FFFFFFF) if bits & 0x80000000 else 0x80000000 + bits


def hc50_precision_group(source, group, values, binding, context):
    ids = [r["record_id"] for r in values]
    p = [float(r["prediction"]) for r in values]
    require(
        ids and len(ids) == len(set(ids)) and all(math.isfinite(v) for v in p),
        "Precision group repeats records or contains nonfinite predictions",
    )
    bound = [binding["record_rows"][i] for i in ids]
    require(
        len({b["graph"] for b in bound}) == len({b["row_sha256"] for b in bound}) == 1
        and all(
            b["split_group"] == group and b["graph"] == source[i]["canonical_identity_group"]
            for b, i in zip(bound, ids, strict=True)
        )
        and len({int(r["fold"]) for r in values}) == 1,
        "Precision allowance requires the same graph/fold and byte-identical chemical inputs",
    )
    legacy = all(abs(v - p[0]) <= 1e-8 + 1e-7 * abs(p[0]) for v in p)
    ulps = None
    try:
        positions = [hc50_precision_binary32(v) for v in p]
        ulps = max(positions) - min(positions)
    except ValueError:
        if not legacy:
            raise
    mode = "legacy_float_tolerance"
    if not legacy:
        policy = context["model_contract"].get("linear_solver_policy", {})
        require(
            context["seed"] == 2026
            and context["role"] == "inner_oof"
            and set(ids) == set(HC50_PRECISION_PAIR)
            and group == HC50_PRECISION_FAMILY
            and bound[0]["graph"] == HC50_PRECISION_GRAPH
            and context["result"].get("selected_model") == "linear"
            and policy
            == context["inputs"]["linear_solver_policies"][context["result"]["selected_route"]]
            and policy.get("version") == "dimension_based_linear_policy_v2"
            and policy.get("solver") == "lsqr"
            and policy.get("selection_basis") == "raw_route_feature_count_only"
            and policy.get("scaler_with_mean") is False
            and policy.get("feature_count") == binding["feature_width"],
            "Precision exception requires the reviewed2026 pair and frozen linear LSQR policy",
        )
        require(
            all(abs(v) >= 2.0**-126 for v in p)
            and len({math.copysign(1.0, v) for v in p}) == 1
            and ulps is not None
            and ulps <= 32,
            "Precision exception requires normal one-sign binary32 predictions within32ULPs",
        )
        mode = "binary32_five_guard_bits"
    return dict(
        policy=HC50_PRECISION_POLICY,
        mode=mode,
        max_allowed_binary32_ulps=32,
        engineering_budget_not_mathematical_BLAS_bound=True,
        observed_absolute_spread=max(p) - min(p),
        observed_binary32_ulp_distance=ulps,
        raw_predictions_unchanged=True,
        same_graph_and_fold=True,
        byte_identical_selected_chemical_inputs=True,
        selected_input_row_sha256=bound[0]["row_sha256"],
        selected_feature_width=binding["feature_width"],
        source_record_ids=ids,
        canonical_identity_group=bound[0]["graph"],
    )


def hc50_precision_ast_evidence(implementation):
    def evaluator():
        tree = ast.parse(Path(implementation).read_text(encoding="utf-8-sig"))
        selected = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "evaluation_diagnostics"
        ]
        require(len(selected) == 1, "Original HC50 evaluator missing or repeated")
        return ast.Module(body=selected, type_ignores=[])

    old, revised = evaluator(), evaluator()
    expected = ast.dump(
        ast.parse("np.allclose(p, p[0], rtol=1e-7, atol=1e-8)", mode="eval").body,
        include_attributes=False,
    )

    class Replace(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                self.count += 1
                return ast.parse(
                    "_bounded_precision_guard(source, group, values, p)", mode="eval"
                ).body
            return self.generic_visit(node)

    patch = Replace()
    revised = patch.visit(revised)
    require(patch.count == 1, "Precision reporting must replace exactly one legacy constancy guard")
    return dict(
        private_legacy_guard_replacements=1,
        legacy_evaluator_ast_sha256=hc50_precision_object_sha(
            ast.dump(old, include_attributes=False)
        ),
        revised_evaluator_ast_sha256=hc50_precision_object_sha(
            ast.dump(revised, include_attributes=False)
        ),
    )


def verify_hc50_role_report(report, predictions, source, role, precision=None):
    require(isinstance(report, dict), "HC50 evaluation role is not an object")
    by_id = {r["record_id"]: r for r in source}
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["split_group"]].append(row)
    require(
        report.get("records") == len(predictions)
        and report.get("molecular_groups") == len(grouped)
        and report.get("actual_folds") == (5 if role == "inner_oof" else None)
        and report.get("score_role")
        == ("model_selection" if role == "inner_oof" else "outer_train_model_evaluation")
        and report.get("full_refit_performance_claimed") is False,
        "HC50 evaluation/refit score roles changed",
    )
    computed = error_metrics(
        [float(r["target"]) for r in predictions], [float(r["prediction"]) for r in predictions]
    )
    check_metrics(report.get("record_weighted"), computed)
    groups = report.get("per_molecule", [])
    require(
        len(groups) == len(grouped) and {g["split_group"] for g in groups} == set(grouped),
        "HC50 per-molecule evaluation membership changed",
    )
    per_group = []
    for group in groups:
        actual = grouped[group["split_group"]]
        expected = {r["record_id"]: r for r in actual}
        saved = group.get("predictions", [])
        require(
            group.get("observations") == len(actual)
            and group.get("fold") == int(actual[0]["fold"])
            and len(saved) == len(expected)
            and {r["record_id"] for r in saved} == set(expected),
            "HC50 per-molecule count/fold changed",
        )
        predicted = [float(r["prediction"]) for r in actual]
        if precision is None:
            require(
                all(abs(p - predicted[0]) <= 1e-8 + 1e-7 * abs(predicted[0]) for p in predicted),
                "HC50 structure-only molecular predictions differ",
            )
        else:
            context = {**precision, "role": role}
            note = hc50_precision_group(
                by_id, group["split_group"], actual, precision["binding"], context
            )
            require(
                group.get("prediction_consistency") == note,
                "HC50 precision metadata differs from unchanged OOF",
            )
            if note["mode"] == "binary32_five_guard_bits":
                precision["exceptions"].append(
                    dict(evaluation_role=role, split_group=group["split_group"], **note)
                )
        for row in saved:
            original = expected[row["record_id"]]
            require(
                row.get("evaluation_role") == role
                and row.get("split_group") == original["split_group"]
                and row.get("fold") == int(original["fold"])
                and float(row["target"]) == float(original["target"])
                and float(row["prediction"]) == float(original["prediction"])
                and all(
                    row.get(k) == by_id[row["record_id"]][k]
                    for k in (
                        "estimate_method",
                        "extrapolation_status",
                        "exposure_evidence_status",
                        "primary_identity_correction_applied",
                    )
                ),
                "HC50 per-record source quality or prediction changed",
            )
        values = error_metrics([float(r["target"]) for r in actual], predicted)
        check_metrics(group, values)
        per_group.append(values)
    check_metrics(report.get("molecule_equal"), mean_molecular_errors(per_group))
    excluded = report.get("excluding_corrected_observation_score_only", {})
    require(isinstance(excluded, dict), "HC50 score-only exclusion is not an object")
    kept = [r for r in predictions if r["record_id"] != HC50_PRIMARY_OBSERVATION]
    require(
        excluded.get("records") == len(kept)
        and excluded.get("model_retrained") is False
        and excluded.get("ablation_experiment") is False,
        "HC50 score-only exclusion became a refit claim",
    )
    check_metrics(
        excluded.get("metrics"),
        error_metrics([float(r["target"]) for r in kept], [float(r["prediction"]) for r in kept]),
    )
    return computed


def verify_hc50_molecular_reporting(entry, journal, task):
    descriptor, proof, frozen, source_row = verify_hc50_coverage_source(entry, task)
    receipt = journal.get("molecular_reporting", {})
    require(
        isinstance(receipt, dict) and receipt.get("status") == "verified",
        "HC50 mandatory molecular report missing",
    )
    path = contained(entry["output_root"], receipt["path"])
    require(
        path.name == "hc50_primary_molecular_diagnostics.json"
        and sha(path) == receipt.get("sha256"),
        "HC50 mandatory report path/SHA changed",
    )
    payload = read(path)
    require(isinstance(payload, dict), "HC50 mandatory report is not an object")
    require(
        payload.get("schema") == HC50_PRIMARY_REPORT
        and payload.get("status") == "verified"
        and payload.get("task_id") == task["task_id"]
        and payload.get("canonical_quantity_id") == HC50_PRIMARY_QUANTITY
        and payload.get("source_corrected_observation") == HC50_PRIMARY_OBSERVATION
        and payload.get("model_selection_repeated") is False
        and payload.get("raw_uncorrected_identity_covered") is False
        and len(payload.get("seeds", [])) == 3
        and set(payload["seeds"]) == set(SEEDS),
        "HC50 mandatory report scope or source correction changed",
    )
    implementation = Path(entry["template"]) / "scripts/peptide_modeling_hc50_primary_bridge.py"
    precision = None
    if (
        payload.get("precision_reporting_revision") is not None
        or receipt.get("precision_reporting_revision") is not None
    ):
        try:
            precision = verify_hc50_precision_revision(entry, journal, receipt, payload)
        except (AttributeError, SyntaxError, struct.error, UnicodeError, EOFError) as exc:
            raise ValueError("Malformed HC50 precision evidence") from exc
    else:
        require(
            not any(
                "precision_consistency" in r
                for r in payload.get("reports", [])
                if isinstance(r, dict)
            ),
            "Precision metadata requires an explicit frozen reporting revision",
        )
        require(
            receipt.get("implementation_sha256")
            == sha(implementation)
            == read(Path(entry["template"]) / "features/manifest.json")["configuration"][
                "implementation_sha256"
            ]
            and receipt.get("implementation_dependencies_sha256")
            == {"peptide_modeling_hc10_human_bridge.py": HC50_PRECISION_NUMERIC_SHA},
            "HC50 report implementation provenance changed",
        )
    reports = payload.get("reports", [])
    rounds = {r["seed"]: r for r in journal["rounds"]}
    require(
        isinstance(reports, list)
        and len(reports) == 3
        and all(isinstance(report, dict) for report in reports)
        and {r["seed"] for r in reports} == set(SEEDS)
        and set(SEEDS) <= set(rounds),
        "HC50 required seed reports missing or repeated",
    )
    checks = read(contained(entry["template"], proof["predeclared_splits"]["path"]))
    for report in reports:
        require(
            report.get("status") == "verified"
            and report.get("records") == 34
            and report.get("molecules") == 30
            and report.get("metrics_space") == "log10_uM"
            and report.get("extra_model_fit") is False
            and report.get("new_model_selection") is False,
            "HC50 report cohort or protocol changed",
        )
        run = contained(entry["output_root"], rounds[report["seed"]]["run_dir"])
        result_path = run / "results/full" / (task["task_id"] + "_common.json")
        require(sha(result_path) == report.get("result_sha256"), "HC50 source result changed")
        result = read(result_path)
        source = table(run / task["table_relative"])
        require(
            sha(run / task["table_relative"])
            == report.get("source_table_sha256")
            == task["table_sha256"],
            "HC50 reported source table changed",
        )
        paths, contract, inputs = verify_hc50_full_refit(run, result, task, source)
        precision_context = None
        if precision is not None:
            binding = hc50_precision_features(run, source, result, inputs, contract)
            item = precision["reports"][report["seed"]]
            verified_path = run / "verification.json"
            verified = read(verified_path)
            result_proof = [
                p
                for p in verified.get("results", [])
                if str(p.get("result_file")).replace("\\", "/")
                == result_path.relative_to(run).as_posix()
            ]
            require(
                sha(verified_path) == rounds[report["seed"]]["audit"]["verification_sha256"]
                and verified.get("status") == "verified"
                and verified.get("source_task_manifest_sha256") == sha(run / "task_manifest.json")
                and len(result_proof) == 1
                and result_proof[0].get("source_result_sha256") == sha(result_path),
                "Precision result no longer matches its original completed-training verification",
            )
            require(
                item.get("selected_route") == result["selected_route"]
                and item.get("selected_model") == result["selected_model"]
                and item.get("numerical_context")
                == dict(
                    selected_model=result["selected_model"],
                    linear_solver_policy=contract.get("linear_solver_policy"),
                )
                and item.get("result_sha256") == sha(result_path)
                and item.get("source_table_sha256") == sha(run / task["table_relative"])
                and item.get("input_contract_sha256") == sha(paths["input_contract"])
                and item.get("input_hash") == result["input_hash"]
                and item.get("oof_sha256") == sha(paths["oof"])
                and item.get("holdout_sha256") == sha(paths["holdout"])
                and item.get("refit_model_sha256") == sha(paths["refit_model"])
                and item.get("feature_evidence") == binding,
                "Precision evidence differs from original selected inputs/model/predictions",
            )
            required = precision["required_protected"]
            for name, expected in result["artifact_hashes"].items():
                required[str(contained(run, name))] = expected["sha256"]
            pointer = run / "features/manifest.json"
            bank = read(pointer)
            cache_name = str(bank["cache_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
            cache = contained(run, "features/" + cache_name)
            for name, expected in bank["artifact_hashes"].items():
                required[str(contained(cache, name))] = expected
            for path in (
                result_path,
                run / "task_manifest.json",
                run / task["table_relative"],
                contained(run, proof["predeclared_splits"]["path"]),
                contained(run, descriptor["proof_path"]),
                run / "verification.json",
                run / "scripts/peptide_modeling_train.py",
                pointer,
            ):
                required[str(path.resolve())] = sha(path)
            precision_context = dict(
                seed=report["seed"],
                binding=binding,
                result=result,
                inputs=inputs,
                model_contract=contract,
                exceptions=[],
            )
        require(
            inputs["task"] == frozen
            and report.get("selected_route") == result["selected_route"]
            and report.get("selected_model") == result["selected_model"]
            and report.get("source_proof_sha256") == descriptor["proof_sha256"]
            and sha(paths["oof"]) == report.get("oof_sha256")
            and sha(paths["holdout"]) == report.get("holdout_sha256"),
            "HC50 report source/model binding changed",
        )
        inner, held = actual_folds(paths["oof"]), table(paths["holdout"])
        source_membership(inner + held, source, require_all=True)
        require(
            len(inner) + len(held) == len({r["record_id"] for r in inner + held}) == 34
            and {r["split_group"] for r in inner}.isdisjoint({r["split_group"] for r in held})
            and all(int(r["fold"]) == -1 for r in held),
            "HC50 OOF/holdout roles overlap or omit records",
        )
        expected = next(c for c in checks["seeds"] if c["seed"] == report["seed"])
        expected_folds = {
            i: f["fold"] for f in expected["folds"] for i in f["validation_record_ids"]
        }
        require(
            {r["record_id"] for r in inner} == set(expected["train_record_ids"])
            and {r["record_id"] for r in held} == set(expected["holdout_record_ids"])
            and {r["record_id"]: int(r["fold"]) for r in inner} == expected_folds,
            "HC50 actual predictions differ from predeclared evaluation roles",
        )
        split = read(paths["outer_split"])
        require(
            split.get("seed") == report["seed"]
            and inputs["config"].get("seed") == report["seed"]
            and set(split["train_records"]) == set(expected["train_record_ids"])
            and set(split["test_records"]) == set(expected["holdout_record_ids"])
            and split["record_groups"] == {r["record_id"]: r["split_group"] for r in source}
            and result.get("n_train") == len(inner)
            and result.get("n_test") == len(held),
            "HC50 selected-model frozen split or evaluation counts changed",
        )
        roles = report.get("evaluation_by_role", {})
        require(
            set(roles) == {"inner_oof", "outer_holdout"}, "HC50 evaluation roles missing or pooled"
        )
        for role, values, result_key in (
            ("inner_oof", inner, "inner_cv"),
            ("outer_holdout", held, "holdout"),
        ):
            check_metrics(
                result[result_key],
                verify_hc50_role_report(
                    roles[role], values, source, role, precision=precision_context
                ),
            )
        if precision_context is not None:
            expected = dict(
                policy=HC50_PRECISION_POLICY,
                max_allowed_binary32_ulps=32,
                engineering_budget_not_mathematical_BLAS_bound=True,
                groups_using_allowance=len(precision_context["exceptions"]),
                exceptions=precision_context["exceptions"],
                selected_matrix_sha256=binding["selected_matrix_sha256"],
                selected_mask_sha256=binding["selected_mask_sha256"],
                feature_width=binding["feature_width"],
                raw_predictions_unchanged=True,
                **precision["ast_evidence"],
            )
            require(
                report.get("precision_consistency")
                == expected
                == precision["reports"][report["seed"]].get("precision_consistency"),
                "Precision budget/AST/evidence exceptions do not reproduce saved OOF",
            )
            precision.setdefault("exceptions", []).extend(precision_context["exceptions"])
        corrected = report.get("corrected_observation", {})
        require(isinstance(corrected, dict), "HC50 corrected-observation report is not an object")
        actual = [r for r in held if r["record_id"] == HC50_PRIMARY_OBSERVATION]
        require(
            len(actual) == 1
            and corrected.get("record_id") == HC50_PRIMARY_OBSERVATION
            and corrected.get("original_endpoint") == "hemolysis_IC50"
            and corrected.get("evaluation_role") == "outer_holdout"
            and corrected.get("fold") == -1
            and corrected.get("split_group") == HC50_PRIMARY_FAMILY
            and corrected.get("primary_corrected_graph") == HC50_PRIMARY_GRAPH
            and corrected.get("raw_uncorrected_identity_covered") is False
            and corrected.get("reported_mean_uM") == "155"
            and corrected.get("reported_SD_uM") == "13.04"
            and corrected.get("donors") == 5
            and corrected.get("molecules") == 1
            and corrected.get("independently_defined_HC50") is False
            and corrected.get("directly_observed_half_lysis_point") is False
            and float(corrected["target"]) == float(actual[0]["target"])
            and float(corrected["prediction"]) == float(actual[0]["prediction"])
            and all(
                corrected.get(k) == source_row[k]
                for k in (
                    "estimate_method",
                    "extrapolation_status",
                    "exposure_evidence_status",
                    "primary_identity_correction_applied",
                )
            ),
            "HC50 corrected point was mistaken for inner-training or direct measurement",
        )
        check_metrics(
            corrected, error_metrics([float(actual[0]["target"])], [float(actual[0]["prediction"])])
        )
        refit = report.get("separate_all_common_refit", {})
        require(isinstance(refit, dict), "HC50 all34 refit report is not an object")
        require(
            refit.get("status") == "verified"
            and refit.get("records") == 34
            and refit.get("molecules") == 30
            and refit.get("training_record_ids") == contract["common_record_ids"]
            and refit.get("corrected_observation_in_training") is True
            and contained(run, refit["model_path"]) == paths["refit_model"]
            and refit.get("model_sha256") == sha(paths["refit_model"])
            and refit.get("model_contract_sha256") == sha(paths["model_contract"])
            and refit.get("refit_attempts") == result["refit_attempts"]
            and refit.get("fit_scope") == "all_common_after_evaluation"
            and refit.get("independently_evaluated_on_these34") is False,
            "HC50 all34 refit training membership or evaluation claim changed",
        )
    durable = read(entry["config_path"])["durable_root"]
    durable_path = contained(durable, receipt["durable_path"])
    require(
        sha(durable_path) == receipt.get("durable_sha256") == receipt["sha256"],
        "HC50 durable report differs",
    )
    if precision is not None:
        require(
            len(precision.get("exceptions", [])) == 1,
            "Precision repair must contain only the reviewed2026 pair exception",
        )
        protected, required = precision["protected"], precision["required_protected"]
        extra = set(protected) - set(required)
        require(
            len(extra) == 1
            and Path(next(iter(extra))).name == "peptide_modeling_hc50_primary_bridge.py"
            and protected[next(iter(extra))] == HC50_PRECISION_LEGACY_SHA
            and all(protected.get(p) == value for p, value in required.items()),
            "Precision evidence omitted or added source/fitted provenance files",
        )
    return dict(
        status="verified",
        path=str(path),
        sha256=receipt["sha256"],
        required_seed_reports=3,
        source_records_per_seed=34,
        molecular_groups_per_seed=30,
        full_refit_training_seeds=3,
        corrected_observation_evaluation_role="outer_holdout",
        independently_evaluated_full_refit=False,
        raw_uncorrected_identity_covered=False,
        **(
            dict(
                precision_policy=HC50_PRECISION_POLICY,
                precision_exception_groups_verified=1,
                max_binary32_ulps=32,
                raw_predictions_modified=False,
            )
            if precision is not None
            else {}
        ),
    )


def verify_molecular_reporting(entry, journal, task):
    """Verify mandatory source-bound reports after the base fitted-model checks."""
    if is_vd_absolute_native_task(task):
        adapter, api = vd_absolute_adapter()
        return adapter.verify_reporting(entry, journal, task, api)
    if is_renal_native_task(task):
        return verify_renal_native_reporting(entry, journal, task)
    if is_rbc50_registered_task(task):
        adapter, api = rbc50_adapter()
        return adapter.verify_reporting(entry, journal, task, api)
    if is_ac50_native_task(task):
        try:
            return verify_ac50_native_reporting(entry, journal, task)
        except AttributeError as exc:
            raise ValueError("Malformed native AC50 report metadata") from exc
    if is_mhc10_native_task(task):
        try:
            return verify_mhc10_native_reporting(entry, journal, task)
        except AttributeError as exc:
            raise ValueError("Malformed native MHC10 report metadata") from exc
    if is_hc50_primary_task(task):
        return verify_hc50_molecular_reporting(entry, journal, task)
    profile = molecular_reporting_profile(task)
    receipt = journal.get("molecular_reporting", {})
    require(isinstance(receipt, dict), "Mandatory molecular receipt is not an object")
    require(receipt.get("status") == "verified", "Mandatory molecular report is not verified")
    path = contained(entry["output_root"], receipt["path"])
    require(path.name == profile["file"], "Unexpected molecular report file")
    require(sha(path) == receipt.get("sha256"), "Molecular report SHA mismatch")
    payload = read(path)
    require(isinstance(payload, dict), "Mandatory molecular report is not an object")
    control_precision = (
        verify_control_float64_revision(entry, journal, receipt, payload, "volume")
        if task["endpoint"] == "Vd_per_kg"
        else None
    )
    if "schema" in profile:
        require(payload.get("schema") == profile["schema"], "Molecular report schema changed")
    require(
        payload.get("status") == "verified" and payload.get("task_id") == task["task_id"],
        "Molecular report task/status mismatch",
    )
    seeds = payload.get("seeds", [])
    require(
        len(seeds) == len(set(seeds)) == 3 and set(seeds) == set(entry["required_seeds"]),
        "Molecular report must contain three distinct required seeds",
    )
    require(
        payload.get("model_selection_repeated") is False
        and payload.get("independent_holdout") is False,
        "Molecular report declares extra selection or an independent holdout",
    )
    reports = payload.get("reports", [])
    require(
        len(reports) == 3 and {r["seed"] for r in reports} == set(seeds),
        "Molecular reports omit or duplicate required seeds",
    )
    rounds = {r["seed"]: r for r in journal["rounds"]}
    for report in reports:
        require(
            report.get("status") == "verified"
            and report.get("extra_model_fit") is False
            and report.get("selection_criterion_changed") is False
            and report.get("independent_holdout") is False,
            "Molecular report is incomplete or changes the training protocol",
        )
        require(
            report.get("records") == profile["records"]
            and report.get("molecules") == profile["groups"]
            and report.get("metrics_space") == profile["metrics_space"]
            and report.get("distinct_molecular_partitions") == 1,
            "Molecular report cohort or metric space differs",
        )
        run = contained(entry["output_root"], rounds[report["seed"]]["run_dir"])
        result_path = run / "results/internal_cv" / (task["task_id"] + ".json")
        require(sha(result_path) == report.get("result_sha256"), "Diagnostic source result changed")
        result = read(result_path)
        require(
            report.get("selected_route") == result.get("selected_route")
            and report.get("selected_model") == result.get("selected_model"),
            "Diagnostic selected route/model differs from saved result",
        )
        oof_name = result["artifacts"]["internal_oof"]
        oof_path = contained(run, oof_name)
        require(
            sha(oof_path)
            == report.get("oof_sha256")
            == result["artifact_hashes"][oof_name]["sha256"],
            "Diagnostic source OOF changed",
        )
        source = table(run / task["table_relative"])
        predictions = actual_folds(oof_path)
        source_membership(predictions, source, require_all=True)
        grouped = defaultdict(list)
        source_by_id = {r["record_id"]: r for r in source}
        for row in predictions:
            grouped[row["split_group"]].append(row)
        require(
            len(predictions) == profile["records"] and len(grouped) == profile["groups"],
            "Diagnostic source row/group counts differ from its endpoint contract",
        )
        record_metrics = error_metrics(
            [float(r["target"]) for r in predictions], [float(r["prediction"]) for r in predictions]
        )
        check_metrics(report.get("record_weighted", {}), record_metrics)
        check_metrics(result.get("internal_cv_selection_metrics", {}), record_metrics)
        groups = report.get("per_molecule", [])
        require(
            len(groups) == profile["groups"] and {g["split_group"] for g in groups} == set(grouped),
            "Per-molecule report omits or repeats groups",
        )
        metrics = []
        for group in groups:
            rows = grouped[group["split_group"]]
            target, predicted = (
                [float(r["target"]) for r in rows],
                [float(r["prediction"]) for r in rows],
            )
            require(
                group.get("observations") == len(rows)
                and group.get("fold") == int(rows[0]["fold"]),
                "Molecule record count/fold differs",
            )
            if profile["constant_molecular_prediction"]:
                require(
                    all(
                        abs(p - predicted[0]) <= 1e-8 + 1e-7 * abs(predicted[0]) for p in predicted
                    ),
                    "Structure-only predictions differ within a molecule",
                )
            require(
                group.get("target_min") == min(target) and group.get("target_max") == max(target),
                "Molecule target range differs",
            )
            expected = {r["record_id"]: r for r in rows}
            saved = group.get("predictions", [])
            require(
                len(saved) == len(expected) and {r["record_id"] for r in saved} == set(expected),
                "Per-molecule record membership differs",
            )
            for row in saved:
                original, old_source = expected[row["record_id"]], source_by_id[row["record_id"]]
                require(
                    float(row["target"]) == float(original["target"])
                    and float(row["prediction"]) == float(original["prediction"]),
                    "Per-molecule saved prediction differs from source OOF",
                )
                require(
                    all(
                        row.get(k) == old_source[k]
                        for k in profile["source_fields"]
                        if k in old_source
                    ),
                    "Per-molecule source context changed",
                )
            value = error_metrics(target, predicted)
            check_metrics(group, value)
            metrics.append(value)
        equal = mean_molecular_errors(metrics)
        check_metrics(report.get("molecule_equal", {}), equal)
        if task["endpoint"] == "CL_absolute":
            verify_cl_context_reporting(entry, task, run, result, predictions)
        elif task["endpoint"] == "Vd_per_kg":
            revised_controls = None
            if control_precision is not None:
                by_record = {row["record_id"]: row for row in predictions}
                check = dict(
                    seed=report["seed"],
                    folds=[
                        dict(
                            fold=fold,
                            train_record_ids=[
                                row["record_id"]
                                for row in source
                                if int(by_record[row["record_id"]]["fold"]) != fold
                            ],
                            validation_record_ids=[
                                row["record_id"]
                                for row in source
                                if int(by_record[row["record_id"]]["fold"]) == fold
                            ],
                        )
                        for fold in range(5)
                    ],
                )
                frozen = read(entry["manifest_path"])["tasks"]
                external = next(
                    item["fixed_external_covariates"]
                    for item in frozen
                    if item["task_id"] == task["task_id"]
                )
                revised_controls, _ = verify_control_float64_seed_files(
                    control_precision,
                    run,
                    source,
                    result,
                    result_path,
                    report,
                    check,
                    "volume",
                    external,
                )
            verify_volume_context_reporting(
                entry,
                task,
                run,
                result,
                predictions,
                report,
                control_revision=control_precision,
                revised_controls=revised_controls,
            )
    if task["endpoint"] in {"CL_absolute", "Vd_per_kg"}:
        durable = read(entry["config_path"])["durable_root"]
        durable_path = contained(durable, receipt.get("durable_path", profile["file"]))
        require(
            sha(durable_path)
            == receipt.get("durable_sha256", receipt["sha256"])
            == receipt["sha256"],
            "Durable molecular report SHA mismatch",
        )
    return {
        "status": "verified",
        "path": str(path),
        "sha256": receipt["sha256"],
        "required_seed_reports": 3,
        "source_records_per_seed": profile["records"],
        "molecular_groups_per_seed": profile["groups"],
        "extra_model_fit": control_precision is not None,
        "independent_holdout": False,
        **(
            {
                "diagnostic_precision_revision": finish_control_float64_revision(
                    entry, control_precision
                )
            }
            if control_precision is not None
            else {}
        ),
    }


def verify_observation_coverage_source(entry, task):
    """Validate the single reviewed observation mapping, never a global label alias."""
    if is_hc50_primary_task(task):
        return verify_hc50_coverage_source(entry, task)
    declared = task.get("observation_endpoint_coverage", [])
    require(
        isinstance(declared, list) and len(declared) == 1, "Unexpected observation coverage scope"
    )
    descriptor = declared[0]
    require(isinstance(descriptor, dict), "Observation coverage declaration is not an object")
    require(
        task["endpoint"] == descriptor.get("canonical_endpoint") == "HC10"
        and descriptor.get("original_endpoint") == "MHC10"
        and descriptor.get("canonical_quantity_id") == HC10_CANONICAL_QUANTITY
        and descriptor.get("record_ids") == [MHC10_QUALIFIED_OBSERVATION]
        and descriptor.get("original_labelled_records") == 6
        and descriptor.get("included_original_records") == 1
        and descriptor.get("included_original_molecular_groups") == 1
        and descriptor.get("canonical_molecular_groups") == 5
        and descriptor.get("inference_alias_is_global") is False
        and descriptor.get("all_original_records_covered") is False,
        "Observation coverage is not the reviewed one-of-six MHC10 mapping",
    )
    require(
        task.get("scientific_definition", {}).get("molecular_reporting_schema")
        == HC10_CANONICAL_REPORT,
        "Observation coverage lacks the ten-row molecular report contract",
    )
    manifest_path = Path(entry["manifest_path"])
    require(sha(manifest_path) == entry["manifest_sha256"], "Observation coverage manifest changed")
    matched = [t for t in read(manifest_path)["tasks"] if t["task_id"] == task["task_id"]]
    require(len(matched) == 1, "Observation coverage source task missing or repeated")
    frozen_task = matched[0]
    require(
        frozen_task.get("observation_endpoint_coverage") == declared,
        "Observation mapping differs from the frozen task",
    )
    proof_path = contained(entry["template"], descriptor["proof_path"])
    require(sha(proof_path) == descriptor["proof_sha256"], "Observation mapping proof SHA mismatch")
    proof = read(proof_path)
    require(isinstance(proof, dict), "Observation coverage proof is not an object")
    require(
        proof.get("schema") == "hc10_observation_endpoint_coverage_v1"
        and proof.get("canonical_task_id") == task["task_id"],
        "Observation mapping proof task/schema changed",
    )
    for key in (
        "canonical_endpoint",
        "original_endpoint",
        "canonical_quantity_id",
        "record_ids",
        "original_labelled_records",
        "included_original_records",
        "included_original_molecular_groups",
        "canonical_molecular_groups",
        "inference_alias_is_global",
        "all_original_records_covered",
    ):
        require(proof.get(key) == descriptor[key], "Observation mapping proof declaration changed")
    table_path = contained(entry["template"], task["table_relative"])
    require(sha(table_path) == task["table_sha256"], "Observation mapping source table changed")
    source = table(table_path)
    ids = [r["record_id"] for r in source]
    table_proof = proof["canonical_source_table"]
    require(
        len(source) == len(set(ids)) == 10
        and len({r["split_group"] for r in source}) == 5
        and len(proof["canonical_record_ids"]) == 10
        and set(proof["canonical_record_ids"]) == set(ids)
        and contained(entry["template"], table_proof["path"]) == table_path
        and table_proof["sha256"] == task["table_sha256"],
        "Observation mapping does not bind the actual ten-row/five-group table",
    )
    previous = proof["parent_nine"]
    previous_path = contained(entry["template"], previous["table_path"])
    require(
        previous.get("task_id") == "TASK_HC10_HUMAN_SOURCE_799fd3266d92284a"
        and previous.get("sha256")
        == sha(previous_path)
        == "c2ae46d5b1d072aac26d2811feceb3013f1b5b906a479911cf9e366245319e3f",
        "Previous nine-observation source snapshot changed",
    )
    previous_rows = table(previous_path)
    by_id = {r["record_id"]: r for r in source}
    require(
        len(previous_rows) == 9
        and {r["record_id"] for r in previous_rows} == set(ids) - {MHC10_QUALIFIED_OBSERVATION},
        "Canonical data version dropped or replaced previous observations",
    )
    for old in previous_rows:
        updated = by_id[old["record_id"]]
        require(
            all(updated.get(key) == value for key, value in old.items())
            and updated["source_original_endpoint"] == old["source_endpoint"],
            "Previous source cells or original endpoint were rewritten",
        )
    rows = [r for r in source if r["record_id"] == MHC10_QUALIFIED_OBSERVATION]
    observations = proof.get("included_observations", [])
    require(len(rows) == len(observations) == 1, "Mapped original observation missing or repeated")
    row, observation = rows[0], observations[0]
    require(isinstance(observation, dict), "Mapped observation is not an object")
    require(
        observation.get("record_id") == row["record_id"]
        and observation.get("original_endpoint") == row["source_original_endpoint"] == "MHC10"
        and observation.get("canonical_endpoint") == "HC10"
        and row["canonical_quantity_id"] == HC10_CANONICAL_QUANTITY
        and row["unit"] == observation.get("unit") == "uM"
        and row["target_transform"] == "log10"
        and observation.get("relation") == row["raw_relation"] == "="
        and Decimal(str(observation["value"])) == Decimal(row["value"]) == Decimal("0.35")
        and Decimal(str(observation["raw_scalar"])) == Decimal(row["raw_scalar"]) == Decimal("0.35")
        and row["raw_unit"] == "uM"
        and math.isclose(float(row["target"]), math.log10(0.35), rel_tol=1e-12, abs_tol=1e-12),
        "Mapped observation endpoint, scalar, relation or target changed",
    )
    graph_key = "CANON_SMILES:f9420c9f1b030adf9d9fe7626706f6eca5574280a79f1daea103aba9000109c0"
    family = "FAMILY:e3630b19efc0fb3b36837673"
    require(
        observation.get("canonical_identity_group") == row["canonical_identity_group"] == graph_key
        and observation.get("split_group") == row["split_group"] == family
        and observation.get("canonical_smiles") == row["canonical_smiles"]
        and hashlib.sha256(row["canonical_smiles"].encode()).hexdigest()
        == graph_key.split(":", 1)[1],
        "Mapped observation complete graph or molecular family changed",
    )
    objects = {}
    for name in ("observation", "assay", "reference", "compound"):
        witness = observation["source_objects"][name]
        path = contained(entry["template"], witness["path"])
        require(sha(path) == witness["sha256"], "Original observation source snapshot SHA changed")
        objects[name] = read(path)
        require(isinstance(objects[name], dict), "Original source snapshot is not an object")
    raw, assay, reference, compound = (
        objects[k] for k in ("observation", "assay", "reference", "compound")
    )
    require(
        raw["endpoint"] == "MHC10"
        and raw["qualifier"] == "="
        and raw["observation_id"] == "40000000022951890"
        and raw["source_entry_id"] == "activity:22951890"
        and Decimal(raw["value"]) == Decimal("0.35")
        and raw["unit"] == "uM"
        and raw["assay_id"] == assay["assay_id"] == row["source_assay_id"] == "38000000002085257"
        and raw["reference_id"] == reference["reference_id"]
        and str(reference["pmid"]) == "28002968"
        and reference["doi"].lower() == "10.1021/acs.jmedchem.6b00922"
        and row["source_paper"] == "PMID:28002968"
        and raw["compound_id"] == compound["compound_id"]
        and compound["smiles"] == row["canonical_smiles"],
        "Original observation/assay/reference/compound linkage changed",
    )
    description = assay["description"]
    readout = observation["readout_evidence"]
    require(
        readout.get("threshold_percent") == 10
        and readout.get("species") == "human"
        and readout.get("matrix") == "erythrocytes"
        and readout.get("assay_description") == description
        and readout.get("minimum_discrete_vs_interpolated_verified") is False
        and readout.get("standardized_protocol_verified") is False
        and readout.get("pH") is None
        and re.search(r"10\s*%", description)
        and re.search(r"ha?emoglobin\s+release", description, re.I)
        and re.search(r"human\s+(?:red\s+blood\s+cells?|RBCs?)", description, re.I)
        and re.search(r"\b1\s*(?:hrs?|hours?)\b", description, re.I),
        "Original observation does not prove the reviewed nominal human 10% readout",
    )
    excluded = proof.get("excluded_original_records", [])
    excluded_ids = [r["record_id"] if isinstance(r, dict) else r for r in excluded]
    require(
        len(excluded_ids) == len(set(excluded_ids)) == 5 and row["record_id"] not in excluded_ids,
        "Observation mapping hides the five uncovered original records",
    )
    return descriptor, proof, frozen_task, row


def verify_hc50_observation_endpoint_coverage(entry, journal, task, state):
    descriptor, _, _, _ = verify_hc50_coverage_source(entry, task)
    evidence = {
        **descriptor,
        "canonical_task_id": task["task_id"],
        "source_proof_verified": True,
        "required_seed_observations_verified": 0,
        "required_seed_final_refit_training_verified": 0,
        "required_seed_outer_train_observations": 0,
        "corrected_observation_evaluation_role": "outer_holdout",
        "state": {
            "queued": "canonical_observation_queued",
            "running": "canonical_observation_running",
            "partial_verified": "canonical_observation_partial_verified",
        }.get(state, "canonical_observation_unverified"),
    }
    if state == "trained_three_seeds":
        verified = verify_hc50_molecular_reporting(entry, journal, task)
        evidence.update(
            state="canonical_observation_trained",
            required_seed_observations_verified=verified["full_refit_training_seeds"],
            required_seed_final_refit_training_verified=verified["full_refit_training_seeds"],
            required_seed_outer_holdout_evaluations_verified=3,
            full_refit_independently_evaluated_on_included_records=False,
        )
    return [evidence]


def verify_observation_endpoint_coverage(entry, journal, task, state):
    if is_hc50_primary_task(task):
        return verify_hc50_observation_endpoint_coverage(entry, journal, task, state)
    descriptor, proof, frozen_task, row = verify_observation_coverage_source(entry, task)
    evidence = {
        **descriptor,
        "coverage_kind": "verified_observation_lineage_only",
        "canonical_task_id": task["task_id"],
        "source_proof_verified": True,
        "required_seed_observations_verified": 0,
        "state": {
            "queued": "canonical_observation_queued",
            "running": "canonical_observation_running",
            "partial_verified": "canonical_observation_partial_verified",
        }.get(state, "canonical_observation_unverified"),
    }
    if state != "trained_three_seeds":
        return [evidence]
    rounds = {r["seed"]: r for r in journal["rounds"]}
    for seed in entry["required_seeds"]:
        run = contained(entry["output_root"], rounds[seed]["run_dir"])
        result = read(run / "results/internal_cv" / (task["task_id"] + ".json"))
        input_path = contained(run, result["artifacts"]["input_contract"])
        require(
            sha(input_path) == result["artifact_hashes"][str(input_path)]["sha256"],
            "Observation model-input SHA changed",
        )
        require(
            read(input_path)["task"] == frozen_task,
            "Observation mapping was added after model fitting",
        )
        oof_path = contained(run, result["artifacts"]["internal_oof"])
        require(
            sha(oof_path) == result["artifact_hashes"][str(oof_path)]["sha256"],
            "Mapped source OOF SHA changed",
        )
        selected = [r for r in actual_folds(oof_path) if r["record_id"] == row["record_id"]]
        require(
            len(selected) == 1
            and selected[0]["split_group"] == row["split_group"]
            and float(selected[0]["target"]) == float(row["target"]),
            "Mapped original observation was not selected exactly once in every required OOF",
        )
        evidence["required_seed_observations_verified"] += 1
    evidence["state"] = "canonical_observation_trained"
    return [evidence]


def sample_entry(entry):
    observed = {"controller_id": entry["controller_id"], "protocol": entry["protocol"], "tasks": {}}
    try:
        j = read(entry["journal_path"])
        if qualitative_report_recovery_mode(entry, j):
            qualitative_report_recovery_header(entry, j)
        require(
            sha(entry["config_path"]) == entry["config_sha256"] == j.get("config_sha256"),
            "Pinned controller configuration changed",
        )
        require(
            sha(entry["manifest_path"]) == entry["manifest_sha256"],
            "Pinned source manifest changed",
        )
        require(
            j.get("pid") == entry["owned_process_identity"]["pid"]
            and j.get("process_identity", {}).get("start_ticks")
            == entry["owned_process_identity"]["start_ticks"],
            "Journal controller ownership changed",
        )
        for task in entry["tasks"]:
            require(
                sha(contained(entry["template"], task["table_relative"])) == task["table_sha256"],
                "Pinned source table changed",
            )
        if entry.get("template_config"):
            item = entry["template_config"]
            require(sha(item["path"]) == item["sha256"], "Pinned training template changed")
        actual = process_identity(entry["owned_process_identity"]["pid"])
        identity = identity_state(entry["owned_process_identity"], actual)
        observed.update(
            journal_state=j.get("status"),
            process_state=identity,
            actual_process=actual,
            journal_sha256=sha(entry["journal_path"]),
            rounds_recorded=len(j.get("rounds", [])),
        )
        by_task = {task["task_id"]: {} for task in entry["tasks"]}
        if j.get("status") in TERMINAL:
            # No fitted models are read while a controller is still running.
            seen = set()
            for r in j.get("rounds", []):
                seed = r.get("seed")
                if seed not in entry["required_seeds"]:
                    continue
                require(seed not in seen, "Repeated required seed in terminal journal")
                seen.add(seed)
                ready_tasks = [
                    t
                    for t in entry["tasks"]
                    if not (needs_molecular_reporting(entry, t) and identity == "same_live_process")
                ]
                if not ready_tasks:
                    continue
                for task_id, evidence in check_seed(
                    {**entry, "tasks": ready_tasks}, r, seed
                ).items():
                    by_task[task_id][str(seed)] = evidence
        for task in entry["tasks"]:
            evidence = by_task[task["task_id"]]
            state = task_state(j.get("status"), identity, evidence, entry["required_seeds"])
            if qualitative_report_recovery_mode(entry, j) and state == "trained_three_seeds":
                observed["qualitative_report_recovery"] = qualitative_report_recovery_final(
                    entry, j
                )
            reporting = None
            if needs_molecular_reporting(entry, task) and j.get("status") in TERMINAL:
                try:
                    require(
                        identity != "same_live_process",
                        "Mandatory postreport controller has not exited",
                    )
                    require(
                        state == "trained_three_seeds",
                        "Base models lack three verified seed results",
                    )
                    reporting = verify_molecular_reporting(entry, j, task)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    reporting = {"status": "unverified", "reason": str(exc)}
                    if state == "trained_three_seeds" or identity == "same_live_process":
                        state = "unverified"
            observed["tasks"][task["task_id"]] = {
                "endpoint": task["endpoint"],
                "state": state,
                "seed_evidence": evidence,
            }
            if reporting is not None:
                observed["tasks"][task["task_id"]]["molecular_reporting"] = reporting
            if task.get("registered_reference_observation_coverage") is not None:
                try:
                    require(is_rbc50_registered_task(task), "Unknown registered-reference task")
                    adapter, api = rbc50_adapter()
                    registered = adapter.coverage(entry, j, task, state, api)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    observed["tasks"][task["task_id"]].update(
                        state="unverified",
                        reason="Registered reference coverage invalid: " + str(exc),
                    )
                    registered = [
                        dict(
                            original_endpoint=item.get("original_endpoint"),
                            canonical_task_id=task["task_id"],
                            source_proof_verified=False,
                            state="registered_reference_observation_unverified",
                            reason=str(exc),
                        )
                        for item in task["registered_reference_observation_coverage"]
                    ]
                observed["tasks"][task["task_id"]][
                    "validated_registered_reference_observation_coverage"
                ] = registered
            if task.get("observation_endpoint_coverage") is not None:
                try:
                    mapped = verify_observation_endpoint_coverage(entry, j, task, state)
                except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
                    observed["tasks"][task["task_id"]].update(
                        state="unverified", reason="Observation coverage invalid: " + str(exc)
                    )
                    mapped = [
                        {
                            "original_endpoint": item.get("original_endpoint"),
                            "canonical_endpoint": task["endpoint"],
                            "canonical_task_id": task["task_id"],
                            "source_proof_verified": False,
                            "state": "canonical_observation_unverified",
                            "reason": str(exc),
                        }
                        for item in (
                            task["observation_endpoint_coverage"]
                            if isinstance(task["observation_endpoint_coverage"], list)
                            else [{}]
                        )
                        if isinstance(item, dict)
                    ]
                observed["tasks"][task["task_id"]]["validated_observation_endpoint_coverage"] = (
                    mapped
                )
            if task.get("native_source_observation_coverage") is not None:
                try:
                    native = (
                        vd_absolute_adapter()[0].coverage(
                            entry, j, task, state, vd_absolute_adapter()[1]
                        )
                        if is_vd_absolute_native_task(task)
                        else verify_renal_native_coverage(entry, j, task, state)
                        if is_renal_native_task(task)
                        else verify_ac50_native_coverage(entry, j, task, state)
                        if is_ac50_native_task(task)
                        else verify_mhc10_native_coverage(entry, j, task, state)
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    InvalidOperation,
                ) as exc:
                    observed["tasks"][task["task_id"]].update(
                        state="unverified", reason="Native annotation coverage invalid: " + str(exc)
                    )
                    native = dict(
                        original_endpoint=(
                            "CL_renal_per_kg" if is_renal_native_task(task) else task["endpoint"]
                        ),
                        quantity_id=(
                            "native_source_reported_renal_clearance_per_reported_kg_annotation"
                            if is_renal_native_task(task)
                            else "native_source_absolute_volume_annotation_given_reported_estimand"
                            if is_vd_absolute_native_task(task)
                            else AC50_NATIVE_QUANTITY
                            if is_ac50_native_task(task)
                            else MHC10_NATIVE_QUANTITY
                        ),
                        task_id=task["task_id"],
                        source_proof_verified=False,
                        coverage_kind=(
                            "native_renal_perkg_source_annotation_observation_participation"
                            if is_renal_native_task(task)
                            else "native_typed_volume_L_source_annotation_observation_participation"
                            if is_vd_absolute_native_task(task)
                            else "native_AC50_semantic_source_annotation_observation_participation"
                            if is_ac50_native_task(task)
                            else "native_MHC10_annotation_observation_participation"
                        ),
                        native_annotation_only=True,
                        state="native_source_observation_unverified",
                        reason=str(exc),
                    )
                observed["tasks"][task["task_id"]][
                    "validated_native_source_observation_coverage"
                ] = native
    except (OSError, ValueError, KeyError, TypeError) as exc:
        observed.update(journal_state="unverified", reason=str(exc))
        observed["tasks"] = {
            t["task_id"]: {"endpoint": t["endpoint"], "state": "unverified", "reason": str(exc)}
            for t in entry["tasks"]
        }
    return observed


def remote_call(registry=None):
    payload = "" if registry is None else base64.b64encode(json.dumps(registry).encode()).decode()
    adapter_payload = rbc50_adapter_payload()
    renal_payload = renal_adapter_payload()
    vd_payload = vd_absolute_adapter_payload()
    source = (
        Path(__file__)
        .read_text(encoding="utf-8")
        .replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\nREGISTRY_PAYLOAD = "
            + repr(payload)
            + "\n"
            + "RBC50_ADAPTER_PAYLOAD = "
            + repr(adapter_payload)
            + "\nRENAL_ADAPTER_PAYLOAD = "
            + repr(renal_payload)
            + "\nVD_ABSOLUTE_ADAPTER_PAYLOAD = "
            + repr(vd_payload)
            + "\n",
            1,
        )
    )
    result = subprocess.run(
        [
            "ssh.exe",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "node23",
            "python3",
            "-",
            "--remote-sample",
        ],
        input=source,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError("Read-only remote audit failed: " + result.stderr[-4000:])
    return json.loads(result.stdout)


def build_report(registry, observations, main_snapshot, frozen_inventory, output):
    output = Path(output)
    main = main_snapshot["campaign_status.json"]
    rounds = {
        r["seed"]: r
        for r in main["rounds"]
        if r.get("status") == "completed" and r.get("audit", {}).get("status") == "verified"
    }
    main_endpoints = (
        set.intersection(*(set(rounds[s]["audit"]["completed_endpoints"]) for s in SEEDS))
        if set(SEEDS) <= set(rounds)
        else set()
    )
    original = {
        r["endpoint"]: r
        for r in frozen_inventory
        if int(r["strict_numeric"]) + int(r["binary_evidence"]) > 0
    }
    require(len(original) == 61, "Frozen labelled endpoint scope changed")
    tasks = []
    main_ids = {
        item["task_id"]
        for r in rounds.values()
        for items in r["audit"].get("endpoint_completed_scopes", {}).values()
        for item in items
    }
    lookup = {o["controller_id"]: o for o in observations["controllers"]}
    for controller in registry["controllers"]:
        obs = lookup.get(controller["controller_id"], {})
        for task in controller["tasks"]:
            evidence = obs.get("tasks", {}).get(
                task["task_id"], {"state": "unverified", "reason": "Controller observation missing"}
            )
            tasks.append(
                {
                    **task,
                    **evidence,
                    "controller_id": controller["controller_id"],
                    "protocol": controller["protocol"],
                    "controller_state": obs.get("journal_state", "unverified"),
                    "process_state": obs.get("process_state", "unverified"),
                    "source_revision_of_main_task": task["task_id"] in main_ids,
                }
            )
    require(
        len(tasks) == len({t["task_id"] for t in tasks}),
        "Supplement registry repeats a task across controllers",
    )
    aliases = defaultdict(list)
    native_participation = defaultdict(list)
    registered_participation = defaultdict(list)
    ac50_participation = []
    renal_participation = []
    vd_participation = []
    for task in tasks:
        for item in task.get("validated_registered_reference_observation_coverage", []):
            name = item.get("original_endpoint")
            require(
                name in {"cytotoxicity_HC50", "hemolysis_HD50"},
                "Unknown registered-reference original endpoint",
            )
            if item.get("source_proof_verified") is True:
                require(
                    item.get("original_labelled_records")
                    == int(original[name]["strict_numeric"])
                    + int(original[name]["binary_evidence"])
                    and item.get("registered_reference_only") is True
                    and item.get("primary_sample_graph_verified") is False
                    and item.get("independent_raw_endpoint_predictor") is False,
                    "RBC reference participation promoted to complete sample/endpoint predictor",
                )
            registered_participation[name].append({**item, "controller_id": task["controller_id"]})
        item = task.get("validated_native_source_observation_coverage")
        if isinstance(item, dict) and item.get("original_endpoint") == "Vd_absolute":
            if item.get("source_proof_verified") is True:
                require(
                    is_vd_absolute_native_task(task)
                    and item.get("original_labelled_records")
                    == item.get("included_original_records")
                    == 15
                    == int(original["Vd_absolute"]["strict_numeric"])
                    + int(original["Vd_absolute"]["binary_evidence"])
                    and item.get("training_pool_records") == 27
                    and item.get("training_pool_identity_groups") == 10
                    and item.get("source_estimand_types") == 8
                    and item.get("source_annotation_only") is True
                    and item.get("validated_physiology") is False
                    and item.get("each_estimand_fivefold_validated") is False,
                    "Vd source annotation coverage was promoted to unified "
                    "physiology/subtype validation",
                )
            if item.get("state") == "source_annotation_observation_trained":
                require(
                    task.get("state") == "trained_three_seeds"
                    and item.get("required_seed_final_training_verified") == 3,
                    "Vd qualified coverage lacks all three fully gated seeds",
                )
            vd_participation.append({**item, "controller_id": task["controller_id"]})
            item = None
        if isinstance(item, dict) and item.get("original_endpoint") == "CL_renal_per_kg":
            if item.get("state") == "source_annotation_observation_trained":
                require(
                    task.get("state") == "trained_three_seeds"
                    and item.get("required_seed_final_training_verified") == 3,
                    "Qualified renal coverage lacks a fully gated three-seed task",
                )
            if item.get("source_proof_verified") is True:
                require(
                    item.get("original_labelled_records")
                    == item.get("included_original_records")
                    == 15
                    == int(original["CL_renal_per_kg"]["strict_numeric"])
                    + int(original["CL_renal_per_kg"]["binary_evidence"])
                    and item.get("training_pool_records") == 18
                    and item.get("training_pool_identity_groups") == 5
                    and item.get("basis_unknown_records") == 6
                    and item.get("readout_and_conditions_unverified_records") == 5
                    and item.get("source_corrected_perkg_records") == 2
                    and item.get("new_NAB739_records") == 1
                    and item.get("source_annotation_only") is True
                    and item.get("validated_physiology") is False
                    and item.get("absolute_renal_endpoint_coverage") is False,
                    "Renal15/15 annotation participation was promoted to physiological/absolute "
                    "coverage",
                )
            renal_participation.append({**item, "controller_id": task["controller_id"]})
            item = None
        if isinstance(item, dict) and item.get("original_endpoint") == "AC50":
            if item.get("source_proof_verified") is True:
                require(
                    item.get("quantity_id") == AC50_NATIVE_QUANTITY
                    and item.get("original_clear_records")
                    == 78
                    == (
                        int(original["AC50"]["strict_numeric"])
                        + int(original["AC50"]["binary_evidence"])
                    )
                    and item.get("included_core_records") == 77
                    and item.get("excluded_feature_only_ids") == [AC50_NATIVE_NR13]
                    and item.get("source_annotation_only") is True
                    and item.get("biological_potency_equivalence_verified") is False,
                    "AC50 annotation77/78 coverage was promoted to resolved potency coverage",
                )
            ac50_participation.append({**item, "controller_id": task["controller_id"]})
            item = None
        if item is not None:
            require(
                isinstance(item, dict) and item.get("original_endpoint") == "MHC10",
                "Native source participation has an unsupported original endpoint",
            )
            if item.get("source_proof_verified") is True:
                require(
                    item.get("quantity_id") == MHC10_NATIVE_QUANTITY
                    and item.get("original_labelled_records")
                    == int(original["MHC10"]["strict_numeric"])
                    + int(original["MHC10"]["binary_evidence"])
                    and item.get("native_annotation_only") is True
                    and item.get("unified_10percent_quantity_verified") is False,
                    "Native annotation participation was counted as canonical quantity coverage",
                )
            native_participation["MHC10"].append({**item, "controller_id": task["controller_id"]})
        for item in task.get("validated_observation_endpoint_coverage", []):
            name = item.get("original_endpoint")
            if name not in original:
                continue
            if item.get("source_proof_verified") is True:
                require(
                    item.get("original_labelled_records")
                    == int(original[name]["strict_numeric"])
                    + int(original[name]["binary_evidence"]),
                    "Observation coverage denominator differs from the frozen original endpoint",
                )
            aliases[name].append({**item, "controller_id": task["controller_id"]})
    endpoints = []
    for name, row in sorted(original.items()):
        attached = [t for t in tasks if t["endpoint"] == name]
        states = {t["state"] for t in attached}
        state = (
            "trained"
            if name in main_endpoints or "trained_three_seeds" in states
            else "running"
            if "running" in states
            else "queued"
            if "queued" in states
            else "partial_verified"
            if "partial_verified" in states
            else "unverified"
            if "unverified" in states
            else "failed"
            if "failed" in states
            else "deferred"
            if attached
            else "gap"
        )
        mapped = aliases.get(name, [])
        registered = registered_participation.get(name, [])
        if (
            not attached
            and name not in main_endpoints
            and any(
                item.get("state") == "registered_reference_observation_trained"
                and item.get("all_original_records_covered") is True
                and item.get("included_original_records") == item.get("original_labelled_records")
                and item.get("required_seed_final_refit_training_verified") == 3
                and item.get("source_proof_verified") is True
                for item in registered
            )
        ):
            state = "registered_reference_observation_trained"
        if not attached and name not in main_endpoints and mapped:
            mapped_states = {item["state"] for item in mapped}
            state = next(
                (
                    status
                    for status in (
                        "canonical_observation_trained",
                        "canonical_observation_running",
                        "canonical_observation_queued",
                        "canonical_observation_partial_verified",
                        "canonical_observation_unverified",
                    )
                    if status in mapped_states
                ),
                "canonical_observation_unverified",
            )
        vd_items = vd_participation if name == "Vd_absolute" else []
        if vd_items and name not in main_endpoints:
            state = (
                "source_annotation_observation_trained"
                if any(
                    item.get("state") == "source_annotation_observation_trained"
                    and item.get("source_proof_verified") is True
                    and item.get("required_seed_final_training_verified") == 3
                    for item in vd_items
                )
                else "unverified"
            )
        renal_items = renal_participation if name == "CL_renal_per_kg" else []
        if (
            not attached
            and name not in main_endpoints
            and any(
                item.get("state") == "source_annotation_observation_trained"
                and item.get("source_proof_verified") is True
                and item.get("required_seed_final_training_verified") == 3
                for item in renal_items
            )
        ):
            state = "source_annotation_observation_trained"
        endpoint = {
            "endpoint": name,
            "frozen_strict_numeric": int(row["strict_numeric"]),
            "frozen_binary_evidence": int(row["binary_evidence"]),
            "state": state,
            "main_three_seed_model_evidence": name in main_endpoints,
            "supplement_task_ids": sorted(
                {t["task_id"] for t in attached}
                | {item["canonical_task_id"] for item in mapped}
                | {item["canonical_task_id"] for item in registered}
                | {
                    item["canonical_task_id"]
                    for item in renal_items
                    if item.get("canonical_task_id")
                }
            ),
        }
        if registered:
            endpoint["validated_registered_reference_observation_coverage"] = registered
        if mapped:
            endpoint["canonical_observation_coverage"] = mapped
        if native_participation.get(name):
            endpoint["validated_native_source_observation_coverage"] = native_participation[name]
        if name == "AC50" and ac50_participation:
            endpoint["validated_native_source_observation_coverage"] = ac50_participation
        if renal_items:
            endpoint["validated_native_source_observation_coverage"] = renal_items
        if vd_items:
            endpoint["validated_native_source_observation_coverage"] = vd_items
        endpoints.append(endpoint)
    counts = dict(Counter(r["state"] for r in endpoints))
    protocols = {}
    for protocol in JOURNALS.values():
        members = [t for t in tasks if t["protocol"] == protocol]
        protocols[protocol] = {
            "tasks": len(members),
            "states": dict(Counter(t["state"] for t in members)),
        }
    summary = {
        "original_endpoints": 61,
        "frozen_labelled_records": sum(
            r["frozen_strict_numeric"] + r["frozen_binary_evidence"] for r in endpoints
        ),
        "endpoint_states": counts,
        "supplement_controllers": len(registry["controllers"]),
        "supplement_task_ids": len(tasks),
        "supplement_new_task_ids": sum(not t["source_revision_of_main_task"] for t in tasks),
        "supplement_source_revision_task_ids": sum(
            t["source_revision_of_main_task"] for t in tasks
        ),
        "supplement_protocols": protocols,
        "required_seeds": SEEDS,
        "main_snapshot_observed_at": main_snapshot["observed_at"],
        "remote_observed_at": observations["observed_at"],
        "main_productive_hours_at_snapshot": main["productive_wall_seconds"] / 3600,
        "goal_complete": False,
    }
    if aliases:
        summary["canonical_observation_coverage_note"] = (
            "Observation-level lineage coverage only; not a global label alias, "
            "an independent raw-endpoint predictor, or coverage of all original records."
        )
        summary["canonical_observation_model_tasks"] = len(
            {item["canonical_task_id"] for values in aliases.values() for item in values}
        )
    if native_participation:
        summary["native_source_observation_coverage_tasks"] = len(
            {item["task_id"] for values in native_participation.values() for item in values}
        )
        summary["native_source_observation_coverage_note"] = (
            "Source-native MHC10 annotation participation only; distinct from canonical HC10 "
            "one-of-six quantity mapping. Six original observations are one molecule; "
            "mouse effect, harmonized "
            "ten-percent quantity and cross-source/homology generalization remain unvalidated."
        )
    if ac50_participation:
        summary["ac50_native_source_annotation_tasks"] = len(
            {item["task_id"] for item in ac50_participation}
        )
        summary["ac50_native_source_annotation_coverage"] = dict(
            original_clear_records=78,
            included_core_records=77,
            feature_only_ids=[AC50_NATIVE_NR13],
            molecular_identity_groups=17,
            scope="SOURCE_ANNOTATION_77_OF_78_not_resolved_biophysical_potencies",
            source_external_validation_claimed=False,
            assay_host_species_known=False,
        )
        summary["ac50_native_source_annotation_note"] = (
            "Native AC50 SOURCE ANNOTATION 77/78 only. NR13 is feature/ledger-only; "
            "PDK1 biological identity remains unresolved and p21 is an induction marker. "
            "Human target/system annotation is not assay-host species. All77 final refits "
            "have no independent evaluation on their included records."
        )
    if vd_participation:
        summary["vd_absolute_source_annotation_coverage"] = dict(
            original_labelled_records=15,
            included_original_records=15,
            training_pool_records=27,
            identity_groups=10,
            source_estimand_types=8,
            source_annotation_only=True,
            validated_physiology=False,
            every_estimand_independently_validated=False,
            unseen_estimand_records_per_seed=22,
            unseen_joint_context_records_per_seed=27,
            verified_tasks=sum(
                item.get("state") == "source_annotation_observation_trained"
                for item in vd_participation
            ),
        )
        summary["vd_absolute_source_annotation_note"] = (
            "Original Vd_absolute15/15 observations participate in a27-row/10-identity typed "
            "source-native absolute-volume annotation task. Eight source estimands "
            "remain distinct; "
            "22/27 validation records lack a fitted molecule with the same estimand and27/27 lack "
            "the joint context. All27 final refits are unscored. This is not unified physiological "
            "Vd/Vss, per-subtype fivefold validation, or clinical generalization."
        )
    if renal_participation:
        summary["renal_source_annotation_coverage"] = dict(
            original_labelled_records=15,
            included_original_records=15,
            training_pool_records=18,
            identity_groups=5,
            known_scaffold_groups=4,
            basis_unknown_records=6,
            readout_and_conditions_unverified_records=5,
            source_corrected_perkg_records=2,
            new_NAB739_records=1,
            absolute_renal_endpoint_coverage=False,
            validated_physiology=False,
            independent_holdout=False,
            strict_scaffold_fivefold_possible=False,
            verified_tasks=sum(
                item.get("state") == "source_annotation_observation_trained"
                for item in renal_participation
            ),
            original_observation_training_verified=(
                15
                if any(
                    item.get("state") == "source_annotation_observation_trained"
                    for item in renal_participation
                )
                else 0
            ),
        )
        summary["renal_source_annotation_note"] = (
            "CL_renal_per_kg15/15 original observations participate in an18-row/5-identity "
            "source-annotation task, with2 primary perkg corrections and1 new NAB739 scalar. "
            "Six kg bases and five readout/condition contexts remain unspecified. "
            "This qualified state is distinct from ordinary trained, absolute renal clearance, "
            "validated physiological targets, strict scaffold separation "
            "and independent performance."
        )
    if registered_participation:
        summary["registered_reference_observation_coverage_tasks"] = len(
            {
                item["canonical_task_id"]
                for values in registered_participation.values()
                for item in values
            }
        )
        summary["registered_reference_observation_coverage_note"] = (
            "Registered-reference-only corrected horse RBC cytotoxicity_HC50 1/1 and HD50 3/4 "
            "source observations; not independent nucleated-cell toxicity, complete HD50 coverage, "
            "primary sample graph/redox verification, horse effects or ChaC homology isolation."
        )
    require(
        sum(counts.values()) == 61 and summary["frozen_labelled_records"] == 213800,
        "Endpoint/label conservation failed",
    )
    report = {
        "schema": "peptide_execution_audit_v2",
        "summary": summary,
        "endpoints": endpoints,
        "supplement_tasks": tasks,
        "controllers": observations["controllers"],
        "limitations": [
            "Main 32 endpoint evidence is the retained audited three-seed snapshot; "
            "this is not a new rehash of all main models.",
            "Fitted-model SHA and saved-fold checks run only after a supplemental controller "
            "is terminal. Running/queued states prove process identity, not model completion.",
            "Terminal verification checks hashes and existing same-host reload proofs; "
            "this read-only audit does not reload or refit estimators.",
            "One completed task per endpoint does not qualify every record, assay, "
            "source, or condition.",
            "Fixed outer holdout, internal-only CV, and nested outer5/inner5 weak-source "
            "benchmarks remain separate protocols.",
        ],
    }
    write(output / "execution_audit.json", report)
    fields = [
        "endpoint",
        "frozen_strict_numeric",
        "frozen_binary_evidence",
        "state",
        "main_three_seed_model_evidence",
        "supplement_task_ids",
        "canonical_observation_coverage",
        "validated_native_source_observation_coverage",
        "validated_registered_reference_observation_coverage",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in endpoints:
        writer.writerow(
            {
                **row,
                "supplement_task_ids": json.dumps(row["supplement_task_ids"]),
                "canonical_observation_coverage": json.dumps(
                    row.get("canonical_observation_coverage", []), ensure_ascii=False
                ),
                "validated_native_source_observation_coverage": json.dumps(
                    row.get("validated_native_source_observation_coverage", []), ensure_ascii=False
                ),
                "validated_registered_reference_observation_coverage": json.dumps(
                    row.get("validated_registered_reference_observation_coverage", []),
                    ensure_ascii=False,
                ),
            }
        )
    (output / "endpoint_execution.tsv").write_text(buffer.getvalue(), encoding="utf-8")
    lines = [
        "# 执行审计 v2",
        "",
        f"主批次证据快照：{main_snapshot['observed_at']}；远端只读采样：{observations['observed_at']}。",
        "",
        f"原始61端点状态：{json.dumps(counts, ensure_ascii=False)}。冻结清晰标签213,800条。",
        "",
        f"冻结{len(registry['controllers'])}个补充控制器，{len(tasks)}个不同task ID："
        f"{summary['supplement_new_task_ids']}新增＋"
        f"{summary['supplement_source_revision_task_ids']}来源修订。task ID修订不重复计新增。",
        "",
        f"主批次三个必需种子42/2026/3407的已审计模型覆盖{len(main_endpoints)}个端点。累计有效计算快照为{summary['main_productive_hours_at_snapshot']:.2f}小时，完整目标尚未完成。",
        "",
        "| 补充协议 | 任务数 | 实际状态 |",
        "| --- | ---: | --- |",
    ]
    for protocol, value in protocols.items():
        lines.append(
            f"| {protocol} | {value['tasks']} | {json.dumps(value['states'], ensure_ascii=False)} |"
        )
    if aliases:
        lines += ["", "原始标签的观察级覆盖（不等于独立端点模型或全部记录覆盖）：", ""]
        for name, mappings in sorted(aliases.items()):
            for item in mappings:
                lines.append(
                    f"- {name}：{item.get('included_original_records', 0)}/"
                    f"{item.get('original_labelled_records', '?')} 条原始标签记录，"
                    f"状态 `{item['state']}`；canonical task `{item['canonical_task_id']}`。"
                )
    if native_participation:
        lines += ["", "原生MHC10注释参与（不等于统一HC10量或已验证物种效应）：", ""]
        for name, mappings in sorted(native_participation.items()):
            for item in mappings:
                lines.append(
                    f"- {name}：{item.get('included_original_records', 0)}/"
                    f"{item.get('original_labelled_records', '?')} 条原始注释，"
                    f"全量重拟合已核验 "
                    f"{item.get('required_seed_full_refit_training_verified', 0)}/3 种子；"
                    f"状态 `{item['state']}`。这六条仍为一个分子组，鼠源仅一个分子组。"
                )
    if ac50_participation:
        lines += ["", summary["ac50_native_source_annotation_note"]]
        for item in ac50_participation:
            lines.append(
                f"- AC50: SOURCE ANNOTATION 77/78; NR13 feature-only; "
                f"all77 final refits verified "
                f"{item.get('required_seed_full_refit_training_verified', 0)}/3; "
                f"state `{item['state']}`."
            )
    if registered_participation:
        lines += ["", summary["registered_reference_observation_coverage_note"], ""]
        for name, mappings in sorted(registered_participation.items()):
            for item in mappings:
                lines.append(
                    f"- {name}: registered reference observations "
                    f"{item.get('included_original_records', 0)}/"
                    f"{item.get('original_labelled_records', '?')}; "
                    "all38 refits verified "
                    f"{item.get('required_seed_final_refit_training_verified', 0)}/3; "
                    f"state `{item['state']}`. Primary sample graphs remain unverified."
                )
    lines += [
        "",
        "控制器退出后仍从冻结registry读取journal、config、manifest、任务表和终态结果；PID复用会明确标记，不把旧任务当作新进程。waiting/running、partial、failed、deferred及unverified分别保留。缺文件、缺种子或缺验证证明不会自动降为科学延期，更不会重新训练。",
        "",
        "实际完成要求三个必需种子逐任务均有真实五折、Optuna、既有验证收据及当前模型/预测文件SHA。定性cytotoxicity保留外5内5与来源弱标签范围。运行中只读小文件，避免反复扫描大模型；终态才深入查证。",
        "",
        "本工具不加载或重拟合模型；主32端点沿用保留的三种子审计证据。所有本次写入仅位于本地审计目录；无远端作业变更。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/peptide_modeling_execution_audit_20261114_v2")
    )
    parser.add_argument(
        "--main-snapshot",
        type=Path,
        default=Path("data/peptide_modeling_multiseed20h_overlay_v1/remote_status.json"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("data/dataset_reclean_20261113_v1/endpoint_inventory.tsv"),
    )
    parser.add_argument("--expected-controllers", type=int, default=11)
    parser.add_argument("--remote-sample", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.remote_sample:
        require(socket.gethostname() == "node23", "Read-only probe reached wrong host")
        payload = globals().get("REGISTRY_PAYLOAD", "")
        value = (
            {
                "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "controllers": [
                    sample_entry(e) for e in json.loads(base64.b64decode(payload))["controllers"]
                ],
            }
            if payload
            else discover()
        )
        print(json.dumps(value))
        return
    require(args.freeze != bool(args.registry), "Choose exactly --freeze or --registry")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        path = args.output_dir / "controller_registry.json"
        require(
            not path.exists() and not path.with_suffix(".sha256").exists(),
            "Frozen registry exists; use --registry",
        )
        registry = remote_call()
        require(
            len(registry["controllers"]) == args.expected_controllers,
            "Live controller count differs from explicit registry scope",
        )
        write(path, registry)
        path.with_suffix(".sha256").write_text(sha(path) + "\n", encoding="ascii")
    else:
        path = args.registry
        require(
            sha(path) == path.with_suffix(".sha256").read_text().strip(),
            "Frozen registry SHA changed",
        )
        registry = read(path)
    observations = remote_call(registry)
    write(args.output_dir / "remote_observations.json", observations)
    main_raw = args.main_snapshot.read_bytes()
    (args.output_dir / "main_status_snapshot.json").write_bytes(main_raw)
    report = build_report(
        registry, observations, json.loads(main_raw), table(args.inventory), args.output_dir
    )
    write(
        args.output_dir / "audit_receipt.json",
        {
            "registry_sha256": sha(path),
            "main_snapshot_sha256": hashlib.sha256(main_raw).hexdigest(),
            "inventory_sha256": sha(args.inventory),
            "auditor_sha256": sha(__file__),
            "auditor_dependencies_sha256": {
                "peptide_modeling_rbc50_registered_execution_audit.py": rbc50_adapter_payload()[
                    "sha256"
                ],
                "peptide_modeling_renal_kg_native_execution_audit.py": renal_adapter_payload()[
                    "sha256"
                ],
                "peptide_modeling_vd_absolute_native_execution_audit.py": (
                    vd_absolute_adapter_payload()["sha256"]
                ),
            },
            "summary": report["summary"],
        },
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
