"""Compile scientific tasks without changing the frozen cleaning catalogue.

Source provenance is retained for evaluation; it is not a prediction feature.
Eligibility is a minimum data contract, never a performance validation claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = [
    "record_id",
    "merged_record_ids",
    "origin",
    "compound_id",
    "target",
    "unit",
    "target_transform",
    "inverse_transform",
    "sequence",
    "sequence_kind",
    "sequence_eligible",
    "canonical_identity_group",
    "structure_smiles_sha256",
    "structure_eligible",
    "source_code",
    "source_entry_id",
    "source_group",
    "source_partition",
    "label_rule",
    "chem_modified",
    "measurement_group",
    "split_group",
    "scope_warning",
]
PRIORITY = {"PAMPA": 0, "hemolysis": 1, "half_life": 2, "Caco2": 3, "RRCK": 4, "MDCK": 5}


def text(value):
    value = str(value or "").strip()
    return (
        ""
        if value.lower() in {"", "\\n", "n.a.", "na", "not reported", "not_reported", "none"}
        else value
    )


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hashed(value, prefix=""):
    return prefix + hashlib.sha256(stable(value).encode()).hexdigest()[:24]


def parsed(value):
    if isinstance(value, dict):
        return value
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def contexts(row):
    context = parsed(row.get("condition_json"))
    return (
        context.get("semantics", {}),
        parsed(context.get("historical_condition")),
        parsed(context.get("source_context")),
    )


def canonical_species(value):
    value = text(value).lower()
    if any(word in value for word in ("not_reported", "not reported", "unknown", "unspecified")):
        return ""
    for needles, result in [
        (("human", "homo sapiens"), "human"),
        (("mouse", "mice", "mus musculus"), "mouse"),
        (("rat", "rattus"), "rat"),
        (("rabbit",), "rabbit"),
        (("dog", "canine"), "dog"),
    ]:
        if any(x in value for x in needles):
            return result
    return value


def assay_text(value):
    if isinstance(value, dict):
        return " ".join(
            text(value.get(k))
            for k in ("description", "assay_type", "system_description", "detection_method")
        )
    return text(value)


def define_task(row):
    """Return (scientific definition, deferral reason, scope warning)."""
    ep, kind, unit = row["endpoint"], row["task_kind"], row["unit"]
    sem, hist, ctx = contexts(row)
    definition = {"endpoint": ep, "task_kind": kind, "quantity_unit": unit}
    source = row.get("source_code", "").lower()
    detail = text(sem.get("endpoint_detail")).lower()
    if ep in {"MIC", "MIC50", "MIC90", "MBC", "MFC", "LC_lemma"}:
        return definition, "organism_and_medium_not_observation_aligned", ""
    if ep in {"PAMPA", "Caco2", "RRCK", "MDCK"} and kind == "regression":
        if unit != "cm/s":
            return definition, "permeability_unit_or_source_scale_unresolved", ""
        if "cycpeptmpdb" not in source:
            return definition, "permeability_protocol_not_aligned_across_sources", ""
        definition.update(assay_system=ep, protocol="cycpeptmpdb_source_reported_protocol")
        return (
            definition,
            "",
            "same_protocol_internal_benchmark;cross_protocol_validation_not_established",
        )
    if ep == "hemolysis" and kind == "classification":
        if "20pct_and_50um" in detail or (
            ctx.get("classification_concentration_uM") == 50
            and ctx.get("classification_lysis_percent") == 20
        ):
            definition.update(
                assay_system="human_erythrocytes", label_contract="20_percent_lysis_at_50_uM"
            )
            return (
                definition,
                "",
                "source_measured_bracketing_rule;source_series_validation_required",
            )
        if "hc50_le_100um" in detail:
            definition.update(
                assay_system="mammalian_erythrocytes_species_unresolved",
                label_contract="HC50_le_100_uM",
            )
            return definition, "", "source_defined_threshold_benchmark;RBC_species_not_row_resolved"
        return definition, "hemolysis_threshold_or_assay_definition_unresolved", ""
    if ep in {"overall_peptide_toxicity", "BBB"} and kind == "classification":
        if "toxinpred3" in source:
            definition.update(label_contract="ToxinPred3_source_toxic_vs_nontoxic_benchmark")
            return (
                definition,
                "",
                "weak_benchmark_reproduction_only;not_experimental_general_toxicity",
            )
        if ep == "BBB" and "compiled_positive_rule_negative" in source:
            definition.update(label_contract="compiled_BBB_positive_vs_rule_constructed_negative")
            return (
                definition,
                "",
                "weak_benchmark_reproduction_only;negative_class_not_measured_nonpenetration",
            )
        return definition, "source_label_contract_unresolved", ""
    if ep == "half_life":
        assay = assay_text(sem.get("assay")) or text(ctx.get("assays"))
        matrix = (
            text(sem.get("matrix")) or text(ctx.get("test_samples")) or text(hist.get("matrix"))
        )
        if matrix.startswith("other_") and text(ctx.get("test_samples")):
            matrix = text(ctx.get("test_samples"))
        narrative = " ".join((assay, matrix, detail, text(ctx.get("in_vivo_in_vitro")))).lower()
        if any(
            term in narrative
            for term in ("calpain", "half of its activity", "retention_half_life", "contact lens")
        ):
            return definition, "not_peptide_elimination_or_degradation_half_life", ""
        species = (
            canonical_species(hist.get("species"))
            or canonical_species(sem.get("species"))
            or canonical_species(matrix)
        )
        if species == matrix.lower():
            species = ""
        parameter = text(sem.get("historical_parameter_semantics"))
        route = text(hist.get("route")) or text(sem.get("route"))
        if kind == "classification":
            # Derivation and measurement remain joined by PEPlife source record IDs.
            if "threshold_1h" not in detail and "stability_threshold_1h" not in parameter:
                return definition, "half_life_binary_threshold_unresolved", ""
        if "systemic_terminal_pk" == parameter or "terminal" in detail:
            system = "systemic_terminal_pk"
            matrix_family = "systemic"
            if not species or not route:
                return definition, "systemic_half_life_species_or_route_missing", ""
        elif any(
            term in narrative
            for term in (
                "trypsin",
                "chymotrypsin",
                "protease",
                "pepsin",
                "intestinal",
                "pancreatin",
            )
        ):
            system = "protease_or_intestinal_stability"
            matrix_family = next(
                (
                    term
                    for term in ("chymotrypsin", "trypsin", "pepsin", "pancreatin", "intestinal")
                    if term in narrative
                ),
                "",
            )
            if not matrix_family:
                return definition, "protease_identity_unresolved", ""
            route = "not_applicable"
        elif "serum" in narrative or "plasma" in narrative:
            system = "plasma_serum_stability"
            matrix_family = "serum" if "serum" in narrative else "plasma"
            if "in vivo" in text(ctx.get("in_vivo_in_vitro")).lower() and kind != "classification":
                return definition, "blood_stability_vs_systemic_pk_ambiguous", ""
            route = "not_applicable"
        else:
            return definition, "half_life_assay_system_unresolved", ""
        if not species and system != "protease_or_intestinal_stability":
            return definition, "half_life_species_missing", ""
        if kind == "classification":
            basis = "source_derived_threshold_1h"
        elif any(term in assay.lower() for term in ("hplc", "uplc", "lc-ms", "lcms")):
            basis = "chromatographic_intact_peptide_decay"
        elif "radioimmun" in assay.lower():
            basis = "immunoreactive_peptide_decay"
        elif "radioactiv" in assay.lower():
            basis = "radiolabel_decay"
        else:
            basis = "source_method_unresolved"
        definition.update(
            assay_system=system,
            matrix=matrix_family,
            species=species or "not_reported",
            route=route.lower(),
            measurement_basis=basis,
            pH=hist.get("pH", "not_reported"),
            temperature=hist.get("temperature_k", "not_reported"),
        )
        if kind == "classification":
            definition["label_contract"] = "half_life_ge_1h"
        return definition, "", "assay_specific_exploratory_task;source_and_family_holdout_required"
    if ep in {"IC50", "EC50", "Ki", "Kd", "AC50", "GI50", "CC50", "IC50_inhibitor"}:
        targets = sem.get("target_ids", [])
        if len(targets) != 1:
            return definition, "target_or_assay_not_observation_aligned", ""
        return definition, "target_specific_activity_deferred_until_assay_contract_review", ""
    return definition, "nonpriority_endpoint_requires_semantic_contract", ""


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            previous = self.parent[item]
            self.parent[item] = root
            item = previous
        return root

    def join(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def assign_split_groups(records):
    """Union exact canonical molecules, parent sequences, and measurement lineage."""
    uf = UnionFind()
    for r in records:
        tokens = ["ROW:" + r["record_id"]]
        if r.get("sequence_eligible") == "true" and r.get("sequence"):
            tokens.append("SEQ:" + r["sequence"])
        if r.get("canonical_identity_group"):
            tokens.append("CHEM:" + r["canonical_identity_group"])
        tokens.extend("MEASURE:" + x for x in r.get("measurement_group", "").split(";") if x)
        for token in tokens[1:]:
            uf.join(tokens[0], token)
    for r in records:
        r["split_group"] = hashed(uf.find("ROW:" + r["record_id"]), "FAMILY:")


def admission(records, kind, min_rows, min_groups):
    if len(records) < min_rows:
        return "deferred", "insufficient_records"
    groups = {r["split_group"] for r in records}
    if len(groups) < min_groups:
        return "deferred", "insufficient_independent_groups"
    values = {r["target"] for r in records}
    if len(values) < 2:
        return "deferred", "single_target_value_or_class"
    if kind == "classification":
        if values != {"0", "1"}:
            return "deferred", "binary_encoding_not_0_1"
        if any(
            len({r["split_group"] for r in records if r["target"] == label}) < 5 for label in values
        ):
            return "deferred", "insufficient_groups_per_class"
    return "eligible", "minimum_data_contract_passed_not_performance_validated"


def conflicting_label_ids(records):
    labels = defaultdict(set)
    keys = {}
    for r in records:
        key = r.get("canonical_identity_group", "")
        if (
            not key
            and r.get("sequence_kind") == "bare_standard"
            and r.get("chem_modified") != "true"
        ):
            key = "SEQ:" + r.get("sequence", "")
        if key:
            keys[r["record_id"]] = key
            labels[key].add(r["target"])
    return {record for record, key in keys.items() if len(labels[key]) > 1}


def tsv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_tsv(path, fields, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def source_context(row):
    return contexts(row)[2]


def measurement_groups(row):
    ctx = source_context(row)
    if "peplife2" in row.get("source_code", "").lower():
        number = row.get("source_entry_id", "").split(":")[-1]
        return "PEPLIFE2:" + number
    source = ctx.get("source", {})
    if source.get("database", "").lower() == "peplife2":
        return ";".join(
            "PEPLIFE2:" + s for s in re.split(r"[;,|\s]+", source.get("source_record_ids", "")) if s
        )
    return ""


def compile_tasks(dataset_dir, run_dir, min_rows=40, min_groups=10):
    dataset_dir, run_dir = Path(dataset_dir), Path(run_dir)
    table_dir = run_dir / "task_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    records = []
    for lane in ("strict_numeric", "binary_evidence"):
        records.extend(tsv_rows(dataset_dir / "lanes" / (lane + ".tsv")))
    wanted = {r["record_id"] for r in records}
    gates = {
        r["record_id"]: r
        for r in tsv_rows(dataset_dir / "representation_gates.tsv")
        if r["record_id"] in wanted
    }
    members = {}
    old_ids = {r["source_record_id"] for r in records if r["origin"] == "OLD"}
    new_ids = {r["compound_id"] for r in records if r["origin"] == "NEW"}
    for r in tsv_rows(dataset_dir / "structure_qc" / "structure_members.tsv"):
        key = r["source_record_id"] if r["origin"] == "OLD" else r["compound_id"]
        if key in (old_ids if r["origin"] == "OLD" else new_ids):
            members[(r["origin"], key)] = r["smiles_sha256"]
    workspace = dataset_dir.parent.parent
    pilot = workspace / "data/peptide_hemolysis_binary_pilot/hemolysis_labels_and_splits.tsv"
    hem_pilot = {r["record_id"]: r for r in tsv_rows(pilot)} if pilot.exists() else {}
    # NEW PEPlife copies can inherit conditions only from a matching source record AND value.
    old_pep = defaultdict(list)
    for r in records:
        if (
            r["origin"] == "OLD"
            and r["endpoint"] == "half_life"
            and r["task_kind"] == "regression"
            and r["source_code"].lower() == "peplife2"
        ):
            old_pep[r["source_entry_id"]].append(r)
    enriched = []
    deferred = []
    task_defs = {}
    for raw in records:
        r = dict(raw)
        if r["origin"] == "NEW" and r["source_code"].lower() == "peplife2":
            matches = [
                v for v in old_pep[r["source_entry_id"].split(":")[-1]] if v["value"] == r["value"]
            ]
            definitions = {stable(define_task(v)[0]) for v in matches if not define_task(v)[1]}
            if len(definitions) == 1:
                r["condition_json"] = next(
                    v["condition_json"] for v in matches if not define_task(v)[1]
                )
        definition, reason, warning = define_task(r)
        gate = dict(gates.get(r["record_id"], {}))
        pilot_row = hem_pilot.get(r["source_entry_id"], {})
        pilot_sequence = text(pilot_row.get("sequence"))
        if pilot_sequence and set(pilot_sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
            # This is a declared source parent sequence, not a generated chemical structure.
            r["sequence"] = pilot_sequence
            r["sequence_kind"] = "parent_projection"
            gate["sequence_model_eligible"] = "true"
        if not reason and not (
            gate.get("structure_model_eligible") == "true"
            or gate.get("sequence_model_eligible") == "true"
        ):
            reason = "no_verified_structure_or_encodable_sequence"
        if (
            not reason
            and pilot_row
            and pilot_row.get("strict_source_model_eligible", "").lower() != "true"
        ):
            reason = "hemolysis_pilot_source_gate_rejected"
        if reason:
            deferred.append(
                {"record_id": r["record_id"], "endpoint": r["endpoint"], "reason": reason}
            )
            continue
        tid = hashed(definition, "TASK_")
        task_defs[tid] = definition
        target = float(r["value"])
        if not math.isfinite(target) or (r["task_kind"] == "regression" and target <= 0):
            deferred.append(
                {
                    "record_id": r["record_id"],
                    "endpoint": r["endpoint"],
                    "reason": "nonpositive_or_nonfinite_target",
                }
            )
            continue
        transform = "identity" if r["task_kind"] == "classification" else "log10"
        target_text = (
            str(int(target)) if transform == "identity" else format(math.log10(target), ".12g")
        )
        ctx = source_context(r)
        source_group = (
            pilot_row.get("recommended_split_group")
            or text(r["reference_ids"])
            or "source_series_unresolved"
        )
        source_partition = (
            text(ctx.get("dataset_split"))
            or text(ctx.get("original_split"))
            or text(r.get("source_partition"))
        )
        source_key = r["source_record_id"] if r["origin"] == "OLD" else r["compound_id"]
        compact = {field: "" for field in FIELDS}
        compact.update(
            record_id=r["record_id"],
            merged_record_ids=r["record_id"],
            origin=r["origin"],
            compound_id=r["compound_id"],
            target=target_text,
            unit=r["unit"],
            target_transform=transform,
            inverse_transform="identity" if transform == "identity" else "10**prediction",
            sequence=r["sequence"],
            sequence_kind=r["sequence_kind"],
            sequence_eligible=gate.get("sequence_model_eligible", "false"),
            canonical_identity_group=gate.get("canonical_identity_group", ""),
            structure_smiles_sha256=members.get((r["origin"], source_key), ""),
            structure_eligible=gate.get("structure_model_eligible", "false"),
            source_code=r["source_code"],
            source_entry_id=r["source_entry_id"],
            source_group=source_group,
            source_partition=source_partition,
            label_rule=text(contexts(r)[0].get("label_rule")),
            chem_modified=r["has_modifications"],
            measurement_group=measurement_groups(r),
            scope_warning=warning,
        )
        compact["_task_id"] = tid
        enriched.append(compact)
    assign_split_groups(enriched)
    grouped = defaultdict(list)
    for r in enriched:
        grouped[r.pop("_task_id")].append(r)
    manifest_tasks = []
    collapsed_source_copies = 0
    label_conflict_records = 0
    for tid, task_rows in grouped.items():
        definition = task_defs[tid]
        # Collapse only provably shared PEPlife source measurements copied across OLD/NEW.
        representatives = {}
        for r in task_rows:
            key = (
                (r["measurement_group"], r["target"], r["split_group"])
                if r["measurement_group"] and ";" not in r["measurement_group"]
                else (r["record_id"],)
            )
            if key in representatives:
                existing = representatives[key]
                lineage = existing["merged_record_ids"] + ";" + r["record_id"]

                def score(item):
                    return (
                        item["structure_eligible"] == "true",
                        item["sequence_eligible"] == "true",
                    )

                if score(r) > score(existing):
                    r["merged_record_ids"] = lineage
                    if existing["source_group"] != "source_series_unresolved":
                        r["source_group"] = existing["source_group"]
                    representatives[key] = r
                else:
                    existing["merged_record_ids"] = lineage
                collapsed_source_copies += 1
            else:
                representatives[key] = r
        task_rows = list(representatives.values())
        prototype = task_rows[0]
        if definition["task_kind"] == "classification":
            conflicts = conflicting_label_ids(task_rows)
            label_conflict_records += len(conflicts)
            deferred.extend(
                {
                    "record_id": r["record_id"],
                    "endpoint": definition["endpoint"],
                    "reason": "conflicting_labels_same_exact_representation",
                }
                for r in task_rows
                if r["record_id"] in conflicts
            )
            task_rows = [r for r in task_rows if r["record_id"] not in conflicts]
        status, reason = admission(task_rows, definition["task_kind"], min_rows, min_groups)
        table_path = ""
        if status == "eligible":
            table_path = "task_tables/" + tid + ".tsv"
            write_tsv(run_dir / table_path, FIELDS, sorted(task_rows, key=lambda r: r["record_id"]))
        else:
            deferred.extend(
                {"record_id": r["record_id"], "endpoint": definition["endpoint"], "reason": reason}
                for r in task_rows
            )
        manifest_tasks.append(
            {
                "task_id": tid,
                "endpoint": definition["endpoint"],
                "task_kind": definition["task_kind"],
                "status": status,
                "reason": reason,
                "table_path": table_path,
                "n_records": len(task_rows),
                "n_groups": len({r["split_group"] for r in task_rows}),
                "class_counts": dict(Counter(r["target"] for r in task_rows))
                if definition["task_kind"] == "classification"
                else {},
                "target_transform": prototype["target_transform"],
                "inverse_transform": prototype["inverse_transform"],
                "scope_warning": prototype["scope_warning"],
                "task_definition": definition,
            }
        )
    manifest_tasks.sort(
        key=lambda t: (
            t["status"] != "eligible",
            PRIORITY.get(t["endpoint"], 99),
            t["endpoint"],
            t["task_id"],
        )
    )
    write_tsv(run_dir / "deferred_records.tsv", ["record_id", "endpoint", "reason"], deferred)
    summary = [
        {
            k: t[k]
            for k in (
                "task_id",
                "endpoint",
                "task_kind",
                "status",
                "reason",
                "table_path",
                "n_records",
                "n_groups",
                "scope_warning",
            )
        }
        for t in manifest_tasks
    ]
    if summary:
        write_tsv(run_dir / "task_summary.tsv", list(summary[0]), summary)
    source_hash = hashlib.sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest()
    manifest = {
        "schema": "peptide_modeling_tasks_v1",
        "source_dataset": str(dataset_dir.resolve()),
        "source_manifest_sha256": source_hash,
        "parameters": {"min_rows": min_rows, "min_groups": min_groups, "min_class_groups": 5},
        "tasks": manifest_tasks,
        "counts": {
            "input_records": len(records),
            "eligible_tasks": sum(t["status"] == "eligible" for t in manifest_tasks),
            "eligible_table_records": sum(
                t["n_records"] for t in manifest_tasks if t["status"] == "eligible"
            ),
            "deferred_records": len(deferred),
            "collapsed_source_copies": collapsed_source_copies,
            "label_conflict_records": label_conflict_records,
        },
        "scientific_limits": [
            "eligibility is not model validation",
            "source-specific protocols retain scope warnings",
            "unknown observation-level contexts are deferred",
        ],
    }
    manifest["assertions"] = {
        "input_record_conservation": len(records)
        == manifest["counts"]["eligible_table_records"] + len(deferred) + collapsed_source_copies,
        "all_eligible_tasks_have_groups": all(
            t["n_groups"] >= min_groups for t in manifest_tasks if t["status"] == "eligible"
        ),
    }
    if not all(manifest["assertions"].values()):
        raise AssertionError(stable(manifest["assertions"]))
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = compile_tasks(args.dataset_dir, args.run_dir)
    print(json.dumps(result["counts"], ensure_ascii=False))
