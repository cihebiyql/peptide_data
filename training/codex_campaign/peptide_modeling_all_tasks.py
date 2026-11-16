"""Account for every labelled endpoint and compile conservative scientific tasks.

All labelled rows receive a feature-index row and a terminal data disposition.
Unresolved biological conditions are retained with explicit reasons, not pooled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import peptide_modeling_tasks as baseline

FIELDS = baseline.FIELDS
FEATURE_FIELDS = ["endpoint", "source_record_id", "raw_target"] + FIELDS
ANTIMICROBIAL = {"MIC", "MIC50", "MIC90", "MBC", "MFC", "LC_lemma"}
POTENCY = {"IC50", "EC50", "Ki", "Kd", "AC50", "GI50", "Km", "IC50_inhibitor"}
CELLULAR = {"CC50", "GI50", "LC50", "LD50"}
HEMOLYTIC = {"HC10", "HC50", "MHC", "MHC10"}
PK_PREFIXES = ("CL_", "Vd_", "Vdss_")
IDENTITY_UNITS = {
    "fraction",
    "fraction_bound",
    "fraction_unbound",
    "dimensionless_log10",
    "mole_fraction",
    "mass_fraction",
    "mass_ratio_of_solute_to_solvent",
}
LOG_UNITS = {
    "h",
    "uM",
    "cm/s",
    "L/h",
    "mL/min/kg",
    "uL/min/mg_protein",
    "L",
    "L/kg",
    "mass_concentration_kg_m3",
    "molality_mol_kg",
}
KNOWN_SYSTEMS = ("PAMPA", "Caco2", "RRCK", "MDCK")


def text(value):
    if value is None:
        return ""
    result = str(value).strip()
    return (
        ""
        if result.lower()
        in {
            "",
            "\\n",
            "na",
            "n/a",
            "none",
            "not_reported",
            "not reported",
            "unspecified",
            "unknown",
        }
        else result
    )


def stable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def normal(value):
    return re.sub(r"\s+", " ", text(value).replace("_", " ").strip().lower())


def transform_target(row):
    kind, unit = row["task_kind"], row["unit"]
    value = float(row["value"])
    if not math.isfinite(value):
        raise ValueError("nonfinite_target")
    if kind == "classification":
        if value not in (0.0, 1.0):
            raise ValueError("binary_encoding_not_0_1")
        return {
            "target": str(int(value)),
            "target_transform": "identity",
            "inverse_transform": "identity",
        }
    if kind != "regression":
        raise ValueError("unsupported_supervision_kind")
    if (
        row["endpoint"].lower().startswith("logd")
        or unit in IDENTITY_UNITS
        or unit.startswith("log10(")
    ):
        transform = "identity"
    elif unit in LOG_UNITS:
        if value <= 0:
            raise ValueError("nonpositive_linear_target_requires_source_review")
        transform = "log10"
    else:
        raise ValueError("target_scale_not_defined_for_unit")
    return {
        "target": format(math.log10(value) if transform == "log10" else value, ".12g"),
        "target_transform": transform,
        "inverse_transform": "10**prediction" if transform == "log10" else "identity",
    }


def unique_parameter(sem, names):
    parameters = sem.get("reported_parameters", {})
    found = set()
    for name in names:
        values = parameters.get(name, [])
        if isinstance(values, str):
            values = [[values, ""]]
        for value in values:
            candidate = value[0] if isinstance(value, (list, tuple)) else value
            if text(candidate):
                found.add(normal(candidate))
    return next(iter(found)) if len(found) == 1 else ""


def assay_info(sem):
    assay = sem.get("assay", {})
    return assay if isinstance(assay, dict) else {"description": text(assay)}


def species(value):
    raw = normal(value)
    choices = []
    synonyms = [
        (("human", "homo sapiens"), "human"),
        (("mouse", "mice", "mus musculus"), "mouse"),
        (("rat", "rattus"), "rat"),
        (("rabbit", "oryctolagus"), "rabbit"),
        (("dog", "canine", "canis"), "dog"),
        (("sheep", "ovis aries"), "sheep"),
        (("horse", "equus caballus"), "horse"),
        (("goat", "capra hircus"), "goat"),
        (("pig", "porcine", "sus scrofa"), "pig"),
        (("zebrafish", "danio rerio"), "zebrafish"),
    ]
    for words, label in synonyms:
        if any(re.search(r"\b" + re.escape(word) + r"\b", raw) for word in words):
            choices.append(label)
    if len(choices) > 1:
        return ""
    return choices[0] if choices else ""


def canonical_route(value):
    raw = normal(value).replace("i.v.", "iv").replace("p.o.", "po").replace("s.c.", "sc")
    for pattern, label in [
        (r"\b(iv|intravenous)\b", "intravenous"),
        (r"\b(po|oral)\b", "oral"),
        (r"\b(sc|subcutaneous)\b", "subcutaneous"),
        (r"\b(ip|intraperitoneal)\b", "intraperitoneal"),
    ]:
        if re.search(pattern, raw):
            return label
    return ""


def measurement_method(value):
    raw = normal(value)
    for patterns, label in [
        (("equilibrium dialysis", "rapid equilibrium dialysis"), "equilibrium_dialysis"),
        (("ultrafiltration",), "ultrafiltration"),
        (("ultracentrifugation",), "ultracentrifugation"),
        (("cck8", "cck-8"), "cck8"),
        (("mtt",), "mtt"),
        (("resazurin", "alamar blue"), "resazurin"),
        (("celltiter",), "celltiter"),
        (("radioimmun",), "radioimmunoassay"),
        (("lc-ms", "lcms"), "lc_ms"),
        (("hplc", "uplc"), "chromatography"),
        (("fluorescen",), "fluorescence"),
        (("absorbance", "spectrophot"), "absorbance"),
    ]:
        if any(x in raw for x in patterns):
            return label
    return "not_reported"


def canonical_quantity_basis(endpoint, unit):
    if endpoint == "PPB" and unit == "fraction_bound":
        return "fraction_bound"
    if endpoint == "F":
        return "absolute_bioavailability"
    if endpoint.startswith("CL_renal"):
        return "renal_clearance"
    if endpoint.startswith("CL_per_mg_protein"):
        return "intrinsic_protein_normalized_clearance"
    if endpoint.startswith("CL_"):
        return "systemic_clearance"
    if endpoint.startswith("Vdss_"):
        return "steady_state_distribution_volume"
    if endpoint.startswith("Vd_"):
        return "distribution_volume"
    return endpoint


def canonical_matrix(value):
    value = normal(value)
    if "plasma" in value and "serum" in value:
        return "plasma_or_serum_unresolved"
    for token in ("plasma", "serum", "albumin", "microsomes", "blood"):
        if token in value:
            return token
    return value


def exposure_time(value):
    values = set()
    for number, unit in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|min)\b", normal(value)
    ):
        hours = float(number) / (60 if unit.startswith("m") else 1)
        values.add(format(hours, ".10g"))
    return next(iter(values)) if len(values) == 1 else "not_reported" if not values else "ambiguous"


def scientific_context(row):
    sem, hist, ctx = baseline.contexts(row)
    info = assay_info(sem)
    organism = unique_parameter(
        sem, ("target_species", "target_organism", "assay_organism", "organism")
    )
    sp = (
        species(hist.get("species"))
        or species(sem.get("species"))
        or species(info.get("system_description"))
        or species(organism)
    )
    matrix = normal(hist.get("matrix")) or normal(sem.get("matrix"))
    if row.get("_derived_pk_quality_validated") is True:
        override = sem.get("verified_pk_quality", {})
        if override.get("action") in {"matrix_override", "albumin_source_response"}:
            matrix = override["matrix"]
    return sem, hist, ctx, info, organism, sp, matrix


def define_scientific_task(row):
    if row.get("_derived_pk_quality_validated") is True:
        sem, _, _ = baseline.contexts(row)
        if sem.get("verified_pk_quality", {}).get("action") == "quarantine":
            return (
                {
                    "endpoint": row["endpoint"],
                    "task_kind": row["task_kind"],
                    "quantity_unit": row["unit"],
                },
                "source_quality_quarantine:" + sem["verified_pk_quality"]["reason"],
                "",
            )
    if row.get("_derived_clint_proof_validated") is True:
        from peptide_modeling_clint_bridge import ASSAY_CONTRACT

        sem, _, _ = baseline.contexts(row)
        if (
            row["endpoint"] != "CL_per_mg_protein"
            or row["unit"] != ASSAY_CONTRACT["quantity_unit"]
            or row["task_kind"] != "regression"
            or sem.get("verified_clint_assay_contract") != ASSAY_CONTRACT
        ):
            raise ValueError("Validated CLint admission has a conflicting scientific contract")
        return (
            {
                "endpoint": row["endpoint"],
                "task_kind": "regression",
                "quantity_unit": row["unit"],
                "assay_system": "in_vitro_intrinsic_microsomal_clearance",
                "species": "mouse",
                "matrix": "liver_microsomes",
                "route": "not_applicable_in_vitro",
                "measurement_basis": "intrinsic_protein_normalized_clearance",
                "source_assay_protocol": "acs_jmedchem_5c00544_table6_microsomal",
            },
            "",
            "source_defined_microsomal_benchmark;internal_cv_only_tiny_sample",
        )
    if row.get("_derived_dbaasp_primary_validated") is True:
        sem, _, _ = baseline.contexts(row)
        proof = sem.get("verified_dbaasp_primary_activity", {})
        if (
            row["source_code"] != "DBAASP"
            or row["endpoint"] not in {"MFC", "LD50"}
            or row["unit"] != "uM"
        ):
            raise ValueError("Validated primary activity conflicts with its bounded contract")
        definition = {
            "endpoint": row["endpoint"],
            "task_kind": "regression",
            "quantity_unit": "uM",
            "assay_system": proof["semantic_scope"],
            "organism": proof["organism"],
            "medium": proof["medium"],
            "reported_condition_details": proof["reported_condition_details"],
            "source_assay_protocol": "dbaasp_exact_primary_activity_context",
        }
        warning = "source_defined_microbiology_benchmark;unknown_conditions_remain_unknown"
        if row["endpoint"] == "LD50":
            warning += ";not_mammalian_systemic_ld50;internal_cv_only_tiny_sample"
        return definition, "", warning
    known, reason, warning = baseline.define_task(row)
    if not reason:
        return known, reason, warning
    ep, kind, unit = row["endpoint"], row["task_kind"], row["unit"]
    definition = {"endpoint": ep, "task_kind": kind, "quantity_unit": unit}
    sem, hist, ctx, info, organism, sp, matrix = scientific_context(row)
    narrative = baseline.assay_text(info)
    if ep == "half_life":
        return known, reason, warning
    if ep in ANTIMICROBIAL:
        medium = unique_parameter(sem, ("medium", "culture_medium", "assay_medium", "media"))
        if not organism or not medium:
            return definition, "organism_and_medium_not_observation_aligned", ""
        definition.update(
            assay_system="organism_conditioned_antimicrobial_concentration",
            organism=organism,
            medium=medium,
            strain=normal(info.get("strain"))
            or unique_parameter(sem, ("assay_strain", "strain"))
            or "not_reported",
            exposure_h=exposure_time(narrative),
        )
        # Derived evidence supplies conditions only; its concentration and row
        # provenance are deliberately excluded from the scientific contract.
        if sem.get("verified_condition_details"):
            definition["reported_condition_details"] = sem["verified_condition_details"]
        return definition, "", "organism_and_medium_conditioned;source_validation_required"
    if ep in KNOWN_SYSTEMS or ep in {"Papp", "permeability"}:
        candidate = (
            " ".join((narrative, text(info.get("cell_line")), text(sem.get("endpoint_detail"))))
            .lower()
            .replace("-", "")
        )
        systems = [name for name in KNOWN_SYSTEMS if name.lower() in candidate]
        if "pamp assay" in candidate and "PAMPA" not in systems:
            systems.append("PAMPA")
        if len(systems) != 1:
            return definition, "permeability_assay_system_unresolved", ""
        definition.update(
            assay_system=systems[0],
            protocol="explicit_source_assay_system_method_unharmonized",
            direction=unique_parameter(sem, ("direction", "transport_direction")) or "not_reported",
        )
        return definition, "", "source_assay_specific_exploration;protocol_validation_required"
    if ep.lower().startswith("logd"):
        if unit != "dimensionless_log10":
            return definition, "logd_scale_unresolved", ""
        definition.update(
            assay_system="distribution_coefficient",
            pH="7.4" if "7.4" in ep else hist.get("pH", "not_reported"),
            method=hist.get("method") or measurement_method(narrative),
        )
        return definition, "", "physicochemical_assay_specific;source_validation_required"
    if ep == "solubility":
        solvent = (
            normal(hist.get("solvent")) or normal(sem.get("matrix")) or normal(ctx.get("matrix"))
        )
        if not solvent:
            return definition, "solubility_solvent_unresolved", ""
        definition.update(
            assay_system="solvent_conditioned_solubility",
            solvent=solvent,
            temperature_k=hist.get("temperature_k") or "not_reported",
            pH=hist.get("pH") or unique_parameter(sem, ("ph", "pH")) or "not_reported",
        )
        if kind == "classification":
            positive, negative = text(ctx.get("positive_class")), text(ctx.get("negative_class"))
            if not positive or not negative:
                return definition, "solubility_binary_direction_or_threshold_unresolved", ""
            definition["label_contract"] = {
                "class_1": positive,
                "class_0": negative,
                "kind": "source_defined_solvent_benchmark",
            }
        return definition, "", "solvent_and_dimension_conditioned;binary_labels_source_defined"
    if ep.startswith(PK_PREFIXES) or ep in {"F", "PPB", "Cmax"}:
        if not sp:
            return definition, "pk_species_unresolved", ""
        route = (
            canonical_route(hist.get("route"))
            or canonical_route(sem.get("route"))
            or canonical_route(narrative)
        )
        if ep != "PPB" and not route:
            return definition, "pk_route_unresolved", ""
        if ep == "PPB" and not matrix:
            if "plasma" in normal(narrative):
                matrix = "plasma"
            elif "serum" in normal(narrative):
                matrix = "serum"
            else:
                return definition, "protein_binding_matrix_unresolved", ""
        if ep == "PPB":
            matrix = canonical_matrix(matrix)
            if matrix == "plasma_or_serum_unresolved":
                return definition, "protein_binding_matrix_ambiguous", ""
            # Dose route is provenance, not the definition of an intrinsic
            # fraction-bound assay. Species, matrix, method and pH remain keys.
            route = "not_applicable"
        if ep == "F" and "relative" in (
            normal(narrative) + " " + normal(sem.get("historical_parameter_semantics"))
        ):
            return definition, "relative_bioavailability_reference_unresolved", ""
        if (
            ep.startswith("CL_")
            and "absolute" in ep
            and re.search(r"\bper\s+kg\b", normal(narrative))
        ):
            return definition, "source_unit_and_bodyweight_basis_conflict", ""
        definition.update(
            assay_system="protein_binding" if ep == "PPB" else "systemic_pharmacokinetics",
            species=sp,
            matrix=matrix or "not_reported",
            route=route or "not_applicable",
            measurement_basis=canonical_quantity_basis(ep, unit),
            method=hist.get("method") or measurement_method(narrative),
            pH=hist.get("pH", "not_reported"),
        )
        if unit in {"fraction", "fraction_bound", "fraction_unbound"}:
            definition["target_bounds"] = [0.0, 1.0]
        if row.get("_derived_pk_quality_validated") is True:
            override = sem.get("verified_pk_quality", {})
            if override.get("action") == "albumin_source_response":
                if ep != "PPB" or matrix != "albumin":
                    raise ValueError("Albumin source-quality proof conflicts with its task")
                definition.update(
                    assay_system="source_defined_albumin_binding",
                    measurement_basis="source_reported_albumin_binding_fraction",
                    source_reported_binding_conditions=override["source_conditions"],
                )
                return (
                    definition,
                    "",
                    "source_defined_albumin_assay;not_whole_serum_or_plasma;"
                    "curated_fraction_encoding_retained;source_validation_required",
                )
        if ep == "Cmax":
            dose = normal(sem.get("dose")) or normal(hist.get("dose"))
            if not dose:
                matches = re.findall(
                    r"\b(\d+(?:\.\d+)?)\s*(mg/kg|ug/kg|mg|nmol/kg|umol/kg)\b", normal(narrative)
                )
                dose = " ".join(matches[0]) if len(set(matches)) == 1 else ""
            if not dose:
                return definition, "cmax_dose_unresolved", ""
            definition.update(dose=dose, regimen=normal(hist.get("timepoint")) or "not_reported")
        return definition, "", "species_route_and_parameter_conditioned;source_validation_required"
    hemolytic = ep in HEMOLYTIC or ep.startswith("hemolysis_")
    cell_line = normal(info.get("cell_line")) or unique_parameter(
        sem, ("assay_cell_type", "cell_line")
    )
    method = measurement_method(narrative)
    duration = exposure_time(narrative)
    if kind == "regression" and hemolytic:
        if not sp:
            return definition, "erythrocyte_species_unresolved", ""
        if duration == "ambiguous":
            return definition, "exposure_time_not_unique", ""
        definition.update(
            assay_system="erythrocyte_lysis_concentration",
            species=sp,
            exposure_h=duration,
            method=method,
            response_metric=ep,
        )
        return (
            definition,
            "",
            "erythrocyte_metric_and_species_conditioned;unreported_conditions_retained",
        )
    if kind == "regression" and (ep in POTENCY or ep in CELLULAR or ep.startswith("cytotoxicity_")):
        targets = sem.get("target_ids", [])
        targets = sorted({text(v) for v in targets if text(v)})
        if len(targets) == 1 and ep in POTENCY:
            definition.update(
                target_entity_id=targets[0],
                assay_system="target_conditioned_potency",
                assay_type=normal(info.get("assay_type")) or "not_reported",
                species=sp or "target_entity_specific",
                cell_line=cell_line or "not_reported",
            )
            return definition, "", "target_specific_activity;not_generic_peptide_ADMET"
        cytotoxic = (
            ep in CELLULAR
            or ep.startswith("cytotoxicity_")
            or (
                normal(info.get("assay_type")) == "t"
                or any(
                    v in normal(narrative)
                    for v in ("cytotoxic", "cell viability", "cell growth inhibition")
                )
            )
        )
        if cytotoxic and (cell_line or (ep in {"LC50", "LD50"} and sp)):
            if duration == "ambiguous":
                return definition, "exposure_time_not_unique", ""
            definition.update(
                assay_system="cellular_toxicity"
                if cell_line
                else "organism_lethality_concentration",
                cell_line=cell_line or "not_applicable",
                species=sp or "cell_system_specific",
                exposure_h=duration,
                method=method,
                response_metric=ep,
            )
            return (
                definition,
                "",
                "cell_or_organism_conditioned_toxicity;source_validation_required",
            )
        return definition, "target_or_cell_system_not_observation_aligned", ""
    if kind == "classification":
        return definition, "binary_threshold_or_assay_contract_unresolved", ""
    return definition, "endpoint_quantity_or_assay_contract_unresolved", ""


def write_tsv(path, fields, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _load_records(dataset_dir, additional_records=None, additional_gates=None):
    records = []
    for lane in ("strict_numeric", "binary_evidence"):
        records.extend(baseline.tsv_rows(dataset_dir / "lanes" / (lane + ".tsv")))
    records.extend(additional_records or [])
    additional_ids = {row["record_id"] for row in additional_records or []}
    if set(additional_gates or {}) != additional_ids:
        raise ValueError("Every derived label admission needs its own verified representation gate")
    ids = {r["record_id"] for r in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate labelled record IDs")
    gates = {
        r["record_id"]: r
        for r in baseline.tsv_rows(dataset_dir / "representation_gates.tsv")
        if r["record_id"] in ids and r["record_id"] not in additional_ids
    }
    if additional_gates:
        if set(additional_gates) & set(gates):
            raise ValueError("Derived label admission cannot replace an existing labelled gate")
        gates.update(additional_gates)
    if set(gates) != ids:
        raise ValueError("Labelled rows lack representation gates")
    wanted = {
        (r["origin"], r["source_record_id"] if r["origin"] == "OLD" else r["compound_id"])
        for r in records
    }
    members = {}
    for r in baseline.tsv_rows(dataset_dir / "structure_qc/structure_members.tsv"):
        key = (r["origin"], r["source_record_id"] if r["origin"] == "OLD" else r["compound_id"])
        if key in wanted:
            members[key] = r["smiles_sha256"]
    pilot = (
        dataset_dir.parent.parent
        / "data/peptide_hemolysis_binary_pilot/hemolysis_labels_and_splits.tsv"
    )
    pilots = {r["record_id"]: r for r in baseline.tsv_rows(pilot)} if pilot.exists() else {}
    old_pep = defaultdict(list)
    for r in records:
        if (
            r["origin"] == "OLD"
            and r["endpoint"] == "half_life"
            and r["task_kind"] == "regression"
            and r["source_code"].lower() == "peplife2"
        ):
            old_pep[r["source_entry_id"]].append(r)
    compact = []
    for row in records:
        if row["origin"] == "NEW" and row["source_code"].lower() == "peplife2":
            matches = [
                r
                for r in old_pep[row["source_entry_id"].split(":")[-1]]
                if r["value"] == row["value"]
            ]
            definitions = {
                stable(baseline.define_task(r)[0])
                for r in matches
                if not baseline.define_task(r)[1]
            }
            if len(definitions) == 1:
                row["condition_json"] = next(
                    r["condition_json"] for r in matches if not baseline.define_task(r)[1]
                )
        gate = dict(gates[row["record_id"]])
        p = pilots.get(row["source_entry_id"], {})
        sequence = text(p.get("sequence"))
        if sequence and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
            row["sequence"] = sequence
            row["sequence_kind"] = "parent_projection"
            gate["sequence_model_eligible"] = "true"
        ctx = baseline.source_context(row)
        key = (
            row["origin"],
            row["source_record_id"] if row["origin"] == "OLD" else row["compound_id"],
        )
        out = {field: "" for field in FEATURE_FIELDS}
        try:
            target = transform_target(row)
        except (ValueError, OverflowError) as exc:
            target = {
                "target": "",
                "target_transform": "unresolved",
                "inverse_transform": "unresolved",
            }
            row["_target_reason"] = str(exc)
        out.update(
            endpoint=row["endpoint"],
            source_record_id=row["source_record_id"],
            raw_target=row["value"],
            record_id=row["record_id"],
            merged_record_ids=row["record_id"],
            origin=row["origin"],
            compound_id=row["compound_id"],
            unit=row["unit"],
            sequence=row["sequence"],
            sequence_kind=row["sequence_kind"],
            sequence_eligible=gate.get("sequence_model_eligible", "false"),
            canonical_identity_group=gate.get("canonical_identity_group", ""),
            structure_smiles_sha256=members.get(key, ""),
            structure_eligible=gate.get("structure_model_eligible", "false"),
            source_code=row["source_code"],
            source_entry_id=row["source_entry_id"],
            source_group=p.get("recommended_split_group")
            or text(row.get("reference_ids"))
            or "source_series_unresolved",
            source_partition=text(ctx.get("dataset_split"))
            or text(ctx.get("original_split"))
            or text(row.get("source_partition")),
            label_rule=text(baseline.contexts(row)[0].get("label_rule")),
            chem_modified=row["has_modifications"],
            measurement_group=baseline.measurement_groups(row),
            **target,
        )
        out["_raw"] = row
        out["_pilot"] = p
        compact.append(out)
    return records, compact


def _group_and_reconcile(compact, base_run):
    baseline.assign_split_groups(compact)
    if base_run is None or not (base_run / "task_manifest.json").exists():
        return {"old_groups_reused": 0, "old_components_bridged": 0}
    old = {}
    manifest = json.loads((base_run / "task_manifest.json").read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        if task["status"] == "eligible":
            for row in baseline.tsv_rows(base_run / task["table_path"]):
                for rid in row.get("merged_record_ids", row["record_id"]).split(";"):
                    old[rid] = row["split_group"]
    equivalence = defaultdict(set)
    for row in compact:
        if row["record_id"] in old:
            equivalence[row["split_group"]].add(old[row["record_id"]])
    for row in compact:
        previous = equivalence.get(row["split_group"], set())
        if len(previous) == 1:
            row["split_group"] = next(iter(previous))
    return {
        "old_groups_reused": sum(len(v) == 1 for v in equivalence.values()),
        "old_components_bridged": sum(len(v) > 1 for v in equivalence.values()),
    }


def _restore_split_if_identical(task, run_dir, base_run):
    if base_run is None:
        return "no_base_run"
    manifest_file = base_run / "task_manifest.json"
    if not manifest_file.exists():
        return "base_manifest_missing"
    prior = next(
        (
            t
            for t in json.loads(manifest_file.read_text(encoding="utf-8"))["tasks"]
            if t["task_id"] == task["task_id"] and t["status"] == "eligible"
        ),
        None,
    )
    if prior is None:
        return "new_scientific_task"
    old = {
        r["record_id"]: (r["target"], r["split_group"])
        for r in baseline.tsv_rows(base_run / prior["table_path"])
    }
    current = {
        r["record_id"]: (r["target"], r["split_group"])
        for r in baseline.tsv_rows(run_dir / task["table_path"])
    }
    if current != old:
        return "records_targets_or_connected_groups_changed_new_holdout_required"
    source = base_run / "splits" / task["task_id"] / "group_assignment.json"
    if not source.exists():
        return "base_split_not_present"
    dest = run_dir / "splits" / task["task_id"] / "group_assignment.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha_file(dest) != sha_file(source):
        raise ValueError("Existing frozen split differs from base")
    shutil.copy2(source, dest)
    return "base_outer_split_copied_byte_identical"


def _reconcile_derived_dramp_groups(compact, overlays, base_run, allow_novel_record_ids=()):
    """Join newly verified identities without renaming or merging any existing identity group."""
    if not overlays:
        return {"enabled": False}
    if base_run is None or not (Path(base_run) / "feature_rows_index.tsv").is_file():
        raise ValueError("DRAMP derived representations require a complete prior feature index")
    prior = {
        row["record_id"]: row
        for row in baseline.tsv_rows(Path(base_run) / "feature_rows_index.tsv")
    }
    if set(prior) != {row["record_id"] for row in compact}:
        raise ValueError(
            "DRAMP overlay prior identity index does not match the frozen record universe"
        )
    allowed_novel = set(allow_novel_record_ids)
    if not allowed_novel <= set(overlays):
        raise ValueError("Novel derived identities require a verified overlay record")
    component_groups = defaultdict(set)
    for row in compact:
        if row["record_id"] not in overlays:
            component_groups[row["split_group"]].add(prior[row["record_id"]]["split_group"])
    if any(len(groups) > 1 for groups in component_groups.values()):
        raise ValueError("DRAMP derived identity would bridge distinct existing split groups")
    known_sequences = defaultdict(set)
    for rid, row in prior.items():
        if rid not in overlays and row["sequence_eligible"] == "true" and row["sequence"]:
            known_sequences[row["sequence"]].add(row["split_group"])
    for rid, overlay in overlays.items():
        count = len(known_sequences[overlay["derived_sequence"]])
        if count != 1 and not (count == 0 and rid in allowed_novel):
            raise ValueError(
                "DRAMP source sequence does not resolve to exactly one existing family"
            )
    new_component_groups = {}
    for row in compact:
        if component_groups[row["split_group"]]:
            continue
        if row["record_id"] not in allowed_novel:
            raise ValueError("A novel component lacks a source-proven primary sequence")
        new_component_groups.setdefault(row["split_group"], set()).add(
            prior[row["record_id"]]["split_group"]
        )
    for row in compact:
        groups = component_groups[row["split_group"]]
        if not groups and row["split_group"] in new_component_groups:
            groups = {min(new_component_groups[row["split_group"]])}
        if len(groups) != 1:
            raise ValueError(
                "DRAMP derived component cannot be reconciled with an existing identity"
            )
        row["split_group"] = next(iter(groups))
        if (
            row["record_id"] not in overlays
            and row["split_group"] != prior[row["record_id"]]["split_group"]
        ):
            raise AssertionError("DRAMP overlay changed an unrelated prior split group")
    return {
        "enabled": True,
        "overlaid_records": len(overlays),
        "unrelated_records_retained": len(prior) - len(overlays),
        "old_components_bridged": 0,
        "existing_family_names_preserved": True,
        "new_verified_sequence_components": len(new_component_groups),
    }


def load_condition_evidence_map(path, records):
    """Validate a separately frozen Tier A evidence artifact; never edit inputs."""
    if path is None:
        return {}, {"enabled": False}
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "dbaasp_exact_condition_bridge_v1":
        raise ValueError("Unsupported condition evidence artifact")
    if sha_file(path) != manifest["verified_mapping_sha256"]:
        raise ValueError("Condition evidence artifact hash mismatch")
    originals = {row["record_id"]: row for row in records}
    mapped = {}
    for item in baseline.tsv_rows(path):
        if item["tier"] != "A":
            continue
        if item["record_id"] in mapped:
            raise ValueError("Duplicate derived evidence record")
        raw = originals.get(item["record_id"])
        checks = {
            "source_code": "source_code",
            "source_entry_id": "source_entry_id",
            "endpoint": "endpoint",
            "compound_id": "compound_id",
            "assay_id": "assay_id",
            "observation_id": "source_record_id",
            "frozen_value": "value",
            "frozen_unit": "unit",
        }
        if raw is None or any(item[a] != raw[b] for a, b in checks.items()):
            raise ValueError("Derived evidence does not match frozen source observation")
        if not (
            item["tier_a_training_eligible"] == "true"
            and item["complete_multiset"] == "true"
            and item["single_endpoint_unit"] == "true"
            and item["original_signature_count"] == item["context_signature_count"] == "1"
            and item["scope_observations"] == item["scope_context_entries"]
            and text(item["organism"])
            and (
                text(item["medium"])
                or text(json.loads(item["context_fields_json"]).get("medium.description"))
            )
        ):
            raise ValueError("Derived evidence lacks strict Tier A proof")
        mapped[item["record_id"]] = item
    return mapped, {
        "enabled": True,
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "tier_a_verified_records": len(mapped),
        "applied_records": 0,
        "tier_b_policy": "audit_only_never_admitted",
    }


def enrich_condition_copy(raw, evidence):
    """Return a scientific-classification-only copy with observation conditions."""
    result = dict(raw)
    context = json.loads(raw["condition_json"])
    sem = context.setdefault("semantics", {})
    parameters = sem.setdefault("reported_parameters", {})
    details = json.loads(evidence["context_fields_json"])
    parameters["target_species"] = [[evidence["organism"], ""]]
    # "NA" is the source abbreviation for explicitly described Nutrient agar.
    # Resolve sentinel-looking names only from the paired source description.
    medium = text(evidence["medium"]) or text(details.get("medium.description"))
    parameters["medium"] = [[medium, ""]]
    sem["verified_condition_details"] = {
        key: re.sub(r"\s+", " ", str(value).strip().lower())
        for key, value in details.items()
        if key != "concentration" and str(value).strip()
    }
    result["condition_json"] = stable(context)
    return result


def compile_all_tasks(
    dataset_dir,
    run_dir,
    base_run=None,
    min_rows=40,
    min_groups=10,
    condition_evidence_map=None,
    hemolytik_evidence_map=None,
    dramp_evidence_map=None,
    dramp_representation_overlay=None,
    mhc10_representation_overlay=None,
    cytotoxicity_evidence_map=None,
    cytotoxicity_representation_overlay=None,
    clint_evidence_map=None,
    pk_quality_evidence_map=None,
    dbaasp_primary_evidence_map=None,
    hc10_secondary_evidence_map=None,
):
    dataset_dir, run_dir = Path(dataset_dir), Path(run_dir)
    base_run = Path(base_run) if base_run is not None else None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_tables").mkdir(exist_ok=True)
    csv.field_size_limit(10_000_000)
    source_hash = sha_file(dataset_dir / "manifest.json")
    from peptide_modeling_clint_bridge import load_admissions as load_clint_admissions

    clint_records, clint_gates, clint_summary = load_clint_admissions(
        clint_evidence_map, dataset_dir
    )
    records, compact = _load_records(dataset_dir, clint_records, clint_gates)
    from peptide_modeling_pk_quality_bridge import (
        enrich_quality_copy,
    )
    from peptide_modeling_pk_quality_bridge import (
        load_evidence_map as load_pk_quality_map,
    )

    pk_quality_evidence, pk_quality_summary = load_pk_quality_map(
        pk_quality_evidence_map, records, source_hash
    )
    from peptide_modeling_dbaasp_primary_bridge import (
        enrich_condition_copy as enrich_primary_condition,
    )
    from peptide_modeling_dbaasp_primary_bridge import (
        load_evidence_map as load_primary_condition_map,
    )
    primary_evidence, primary_summary = load_primary_condition_map(
        dbaasp_primary_evidence_map, records, source_hash
    )
    evidence, evidence_summary = load_condition_evidence_map(condition_evidence_map, records)
    from peptide_modeling_hemolytik_bridge import enrich_species_copy, load_evidence_map

    hemolytik_evidence, hemolytik_summary = load_evidence_map(
        hemolytik_evidence_map, records, source_hash
    )
    from peptide_modeling_hc10_secondary_bridge import (
        enrich_species_copy as enrich_hc10_secondary,
    )
    from peptide_modeling_hc10_secondary_bridge import (
        load_evidence_map as load_hc10_secondary,
    )

    hc10_secondary, hc10_secondary_summary = load_hc10_secondary(
        hc10_secondary_evidence_map, records, source_hash
    )
    if set(hc10_secondary) & set(hemolytik_evidence):
        raise ValueError("Distinct Hemolytik evidence tiers cannot overlap")
    from peptide_modeling_dramp_bridge import enrich_cell_copy, load_representation_overlay
    from peptide_modeling_dramp_bridge import load_evidence_map as load_dramp_map

    dramp_evidence, dramp_summary = load_dramp_map(dramp_evidence_map, records, source_hash)
    dramp_overlays, dramp_overlay_summary = load_representation_overlay(
        dramp_representation_overlay, records, source_hash
    )
    if dramp_overlays and set(dramp_overlays) != set(dramp_evidence):
        raise ValueError(
            "DRAMP representation overlay also requires its exact condition evidence map"
        )
    from peptide_modeling_dramp_mhc10_bridge import (
        load_representation_overlay as load_mhc10_overlay,
    )

    mhc10_overlays, mhc10_summary = load_mhc10_overlay(
        mhc10_representation_overlay, records, source_hash
    )
    if set(dramp_overlays) & set(mhc10_overlays):
        raise ValueError("Distinct DRAMP overlay proofs cannot replace the same record")
    combined_overlays = {**dramp_overlays, **mhc10_overlays}
    from peptide_modeling_dramp_cytotoxicity_bridge import (
        enrich_cell_copy as enrich_cytotoxicity_cell,
    )
    from peptide_modeling_dramp_cytotoxicity_bridge import (
        load_evidence_map as load_cytotoxicity_map,
    )
    from peptide_modeling_dramp_cytotoxicity_bridge import (
        load_representation_overlay as load_cytotoxicity_overlay,
    )

    cytotoxicity_evidence, cytotoxicity_summary = load_cytotoxicity_map(
        cytotoxicity_evidence_map, records, source_hash
    )
    cytotoxicity_overlays, cytotoxicity_overlay_summary = load_cytotoxicity_overlay(
        cytotoxicity_representation_overlay, records, source_hash
    )
    if cytotoxicity_overlays and set(cytotoxicity_overlays) != set(cytotoxicity_evidence):
        raise ValueError("Cytotoxicity representation repair requires its exact cell proof map")
    if set(combined_overlays) & set(cytotoxicity_overlays):
        raise ValueError("Separate derived identity proofs overlap")
    combined_overlays = {**combined_overlays, **cytotoxicity_overlays}
    for row in compact:
        if row["record_id"] in combined_overlays:
            overlay = combined_overlays[row["record_id"]]
            if row["record_id"] in cytotoxicity_overlays:
                if (
                    row["sequence_eligible"] == "true"
                    and row["sequence"] != overlay["derived_sequence"]
                ):
                    raise ValueError(
                        "Cytotoxicity proof cannot replace a verified primary sequence"
                    )
                if row["structure_eligible"] == "true" and (
                    overlay["structure_eligible"] != "true"
                    or row["canonical_identity_group"]
                    != overlay["derived_canonical_identity_group"]
                ):
                    raise ValueError("Cytotoxicity proof cannot clear or replace a verified graph")
            row.update(
                sequence=overlay["derived_sequence"],
                sequence_kind=overlay.get("sequence_kind", "bare_standard"),
                sequence_eligible="true",
                canonical_identity_group=overlay["derived_canonical_identity_group"],
                structure_smiles_sha256=overlay["derived_smiles_sha256"],
                structure_eligible=overlay.get("structure_eligible", "true"),
                chem_modified=overlay.get("chem_modified", "false"),
            )
    if evidence and evidence_summary["source_manifest_sha256"] != source_hash:
        raise ValueError("Derived evidence was audited against a different source dataset")
    grouping = _group_and_reconcile(compact, base_run)
    clint_ids = set(clint_gates)
    if clint_ids:
        old_groups = {row["split_group"] for row in compact if row["record_id"] not in clint_ids}
        new_groups = {row["split_group"] for row in compact if row["record_id"] in clint_ids}
        if old_groups & new_groups or len(new_groups) != len(clint_ids):
            raise ValueError("Derived CLint observations do not have independent new identities")
    prior_compact = [row for row in compact if row["record_id"] not in clint_ids]
    dramp_overlay_summary["group_reconciliation"] = _reconcile_derived_dramp_groups(
        prior_compact, combined_overlays, base_run, allow_novel_record_ids=cytotoxicity_overlays
    )
    write_tsv(
        run_dir / "feature_rows_index.tsv",
        FEATURE_FIELDS,
        ({key: row[key] for key in FEATURE_FIELDS} for row in compact),
    )
    groups, definitions = defaultdict(list), {}
    disposition = {}
    placeholders = defaultdict(list)
    for row in compact:
        raw = row["_raw"]
        if row["record_id"] in pk_quality_evidence:
            raw = enrich_quality_copy(raw, pk_quality_evidence[row["record_id"]])
        if row["record_id"] in primary_evidence:
            proof = primary_evidence[row["record_id"]]
            if (
                row["structure_eligible"] != "true"
                or row["canonical_identity_group"] != proof["canonical_identity_group"]
                or row["split_group"] != proof["split_group"]
            ):
                raise ValueError("Primary context cannot replace a graph gate or family assignment")
            raw = enrich_primary_condition(raw, proof)
            primary_summary["applied_records"] += 1
        definition, reason, warning = define_scientific_task(raw)
        if (
            reason in {"", "target_or_cell_system_not_observation_aligned"}
            and row["record_id"] in cytotoxicity_evidence
        ):
            expected_cell = normal(cytotoxicity_evidence[row["record_id"]]["cell_line"])
            if not reason and definition.get("cell_line") not in {None, "", expected_cell}:
                raise ValueError("Cytotoxicity proof conflicts with a resolved source cell")
            raw = enrich_cytotoxicity_cell(raw, cytotoxicity_evidence[row["record_id"]])
            definition, reason, warning = define_scientific_task(raw)
            cytotoxicity_summary["applied_records"] += 1
        if (
            reason == "target_or_cell_system_not_observation_aligned"
            and row["record_id"] in dramp_evidence
        ):
            raw = enrich_cell_copy(raw, dramp_evidence[row["record_id"]])
            definition, reason, warning = define_scientific_task(raw)
            dramp_summary["applied_records"] += 1
        if reason == "erythrocyte_species_unresolved" and row["record_id"] in hemolytik_evidence:
            raw = enrich_species_copy(raw, hemolytik_evidence[row["record_id"]])
            definition, reason, warning = define_scientific_task(raw)
            hemolytik_summary["applied_records"] += 1
        if row["record_id"] in hc10_secondary:
            proof = hc10_secondary[row["record_id"]]
            if reason != "erythrocyte_species_unresolved":
                raise ValueError("HC10 secondary proof must only resolve the missing RBC species")
            if (
                row["structure_eligible"] != "true"
                or row["canonical_identity_group"] != proof["canonical_identity_group"]
            ):
                raise ValueError("HC10 secondary proof cannot replace frozen structure gates")
            raw = enrich_hc10_secondary(raw, proof)
            definition, reason, warning = define_scientific_task(raw)
            hc10_secondary_summary["applied_records"] += 1
        if (
            reason == "organism_and_medium_not_observation_aligned"
            and raw["endpoint"] in ANTIMICROBIAL
            and row["record_id"] in evidence
        ):
            raw = enrich_condition_copy(raw, evidence[row["record_id"]])
            definition, reason, warning = define_scientific_task(raw)
            evidence_summary["applied_records"] += 1
        if (
            not reason
            and row["_pilot"]
            and row["_pilot"].get("strict_source_model_eligible", "").lower() != "true"
        ):
            reason = "hemolysis_pilot_source_gate_rejected"
        if not reason and raw.get("_target_reason"):
            reason = raw["_target_reason"]
        if not reason and not (
            row["structure_eligible"] == "true" or row["sequence_eligible"] == "true"
        ):
            reason = "no_verified_structure_or_encodable_sequence"
        if reason:
            key = (row["endpoint"], raw["task_kind"], row["unit"], reason)
            placeholders[key].append(row)
            continue
        definition = {**definition, "target_transform": row["target_transform"]}
        # Preserve existing task IDs: the baseline definition already implied this transform.
        old_definition, old_reason, _ = baseline.define_task(raw)
        if not old_reason:
            definition = old_definition
        if row["record_id"] in hc10_secondary:
            definition.update(
                evidence_tier=hc10_secondary[row["record_id"]]["evidence_tier"],
                source_reference_pmid="14596922",
                response_percent_lysis=10,
                primary_table_values_verified=False,
            )
            warning = (
                "secondary_curated_quantitative_cells;one_study_one_chemical_family;"
                "within_series_molecular_holdout_not_independent_family_or_source_validation"
            )
        tid = baseline.hashed(definition, "TASK_")
        definitions[tid] = definition
        row["scope_warning"] = warning
        groups[tid].append(row)
    tasks, deferred, collapsed, label_conflicts = [], [], 0, 0
    for tid, members in sorted(groups.items()):
        definition = definitions[tid]
        representatives = {}
        for row in members:
            measurement = row["measurement_group"]
            key = (
                (measurement, row["target"], row["split_group"])
                if measurement and ";" not in measurement
                else (row["record_id"],)
            )
            if key in representatives:
                existing = representatives[key]
                merged = existing["merged_record_ids"] + ";" + row["record_id"]
                if (row["structure_eligible"] == "true", row["sequence_eligible"] == "true") > (
                    existing["structure_eligible"] == "true",
                    existing["sequence_eligible"] == "true",
                ):
                    row["merged_record_ids"] = merged
                    if existing["source_group"] != "source_series_unresolved":
                        row["source_group"] = existing["source_group"]
                    representatives[key] = row
                else:
                    existing["merged_record_ids"] = merged
                collapsed += 1
            else:
                representatives[key] = row
        selected = list(representatives.values())
        prototype = selected[0]
        if definition["task_kind"] == "classification":
            conflicts = baseline.conflicting_label_ids(selected)
            label_conflicts += len(conflicts)
            kept = []
            for row in selected:
                if row["record_id"] in conflicts:
                    for rid in row["merged_record_ids"].split(";"):
                        disposition[rid] = {
                            "status": "deferred",
                            "reason": "conflicting_labels_same_exact_representation",
                            "task_id": tid,
                            "representative": row["record_id"],
                        }
                    deferred.append(
                        {
                            "record_id": row["record_id"],
                            "endpoint": row["endpoint"],
                            "reason": "conflicting_labels_same_exact_representation",
                        }
                    )
                else:
                    kept.append(row)
            selected = kept
        status, reason = baseline.admission(selected, definition["task_kind"], min_rows, min_groups)
        table_path = "task_tables/" + tid + ".tsv" if status == "eligible" else ""
        if table_path:
            write_tsv(
                run_dir / table_path,
                FIELDS,
                ({f: r[f] for f in FIELDS} for r in sorted(selected, key=lambda r: r["record_id"])),
            )
        for row in selected:
            for rid in row["merged_record_ids"].split(";"):
                disposition[rid] = {
                    "status": status,
                    "reason": reason,
                    "task_id": tid,
                    "representative": row["record_id"],
                }
            if status != "eligible":
                deferred.append(
                    {"record_id": row["record_id"], "endpoint": row["endpoint"], "reason": reason}
                )
        item = {
            "task_id": tid,
            "endpoint": definition["endpoint"],
            "task_kind": definition["task_kind"],
            "status": status,
            "reason": reason,
            "table_path": table_path,
            "n_records": len(selected),
            "n_groups": len({r["split_group"] for r in selected}),
            "class_counts": dict(Counter(r["target"] for r in selected))
            if definition["task_kind"] == "classification"
            else {},
            "target_transform": prototype["target_transform"],
            "inverse_transform": prototype["inverse_transform"],
            "scope_warning": prototype["scope_warning"],
            "task_definition": definition,
        }
        item["outer_split_reuse"] = (
            _restore_split_if_identical(item, run_dir, base_run)
            if status == "eligible"
            else "not_eligible"
        )
        tasks.append(item)
    for (ep, kind, unit, reason), members in sorted(placeholders.items()):
        definition = {
            "endpoint": ep,
            "task_kind": kind,
            "quantity_unit": unit,
            "contract_status": "unresolved",
            "unresolved_reason": reason,
        }
        tid = baseline.hashed(definition, "DEFER_")
        for row in members:
            disposition[row["record_id"]] = {
                "status": "deferred",
                "reason": reason,
                "task_id": tid,
                "representative": row["record_id"],
            }
            deferred.append({"record_id": row["record_id"], "endpoint": ep, "reason": reason})
        tasks.append(
            {
                "task_id": tid,
                "endpoint": ep,
                "task_kind": kind,
                "status": "deferred",
                "reason": reason,
                "table_path": "",
                "n_records": len(members),
                "n_groups": len({r["split_group"] for r in members}),
                "class_counts": dict(Counter(r["target"] for r in members))
                if kind == "classification"
                else {},
                "target_transform": members[0]["target_transform"],
                "inverse_transform": members[0]["inverse_transform"],
                "scope_warning": "endpoint_retained_unresolved_contract_not_trainable",
                "task_definition": definition,
                "outer_split_reuse": "not_eligible",
            }
        )
    tasks.sort(
        key=lambda t: (
            t["status"] != "eligible",
            baseline.PRIORITY.get(t["endpoint"], 99),
            t["endpoint"],
            t["task_id"],
        )
    )
    by_endpoint = defaultdict(list)
    for row in compact:
        by_endpoint[row["endpoint"]].append(row)
    endpoint_summary = []
    for ep, members in sorted(by_endpoint.items()):
        ep_tasks = [t for t in tasks if t["endpoint"] == ep]
        eligible_members = [
            r for r in members if disposition[r["record_id"]]["status"] == "eligible"
        ]
        endpoint_summary.append(
            {
                "endpoint": ep,
                "input_records": len(members),
                "eligible_source_records": len(eligible_members),
                "deferred_source_records": len(members) - len(eligible_members),
                "eligible_tasks": sum(t["status"] == "eligible" for t in ep_tasks),
                "deferred_tasks": sum(t["status"] == "deferred" for t in ep_tasks),
                "eligible_table_records": sum(
                    t["n_records"] for t in ep_tasks if t["status"] == "eligible"
                ),
                "reasons": stable(
                    dict(Counter(disposition[r["record_id"]]["reason"] for r in members))
                ),
                "status": "eligible_tasks_present"
                if eligible_members
                else "deferred_with_explicit_reason",
            }
        )
    write_tsv(
        run_dir / "endpoint_admission_summary.tsv", list(endpoint_summary[0]), endpoint_summary
    )
    write_tsv(run_dir / "deferred_records.tsv", ["record_id", "endpoint", "reason"], deferred)
    write_tsv(
        run_dir / "task_summary.tsv",
        [
            "task_id",
            "endpoint",
            "task_kind",
            "status",
            "reason",
            "table_path",
            "n_records",
            "n_groups",
            "scope_warning",
            "outer_split_reuse",
        ],
        (
            {
                key: task[key]
                for key in [
                    "task_id",
                    "endpoint",
                    "task_kind",
                    "status",
                    "reason",
                    "table_path",
                    "n_records",
                    "n_groups",
                    "scope_warning",
                    "outer_split_reuse",
                ]
            }
            for task in tasks
        ),
    )
    write_tsv(
        run_dir / "record_dispositions.tsv",
        ["record_id", "endpoint", "status", "reason", "task_id", "representative"],
        (
            {"record_id": r["record_id"], "endpoint": r["endpoint"], **disposition[r["record_id"]]}
            for r in compact
        ),
    )
    manifest = {
        "schema": "peptide_modeling_all_endpoint_tasks_v1",
        "source_dataset": str(dataset_dir.resolve()),
        "source_manifest_sha256": source_hash,
        "compiler_sha256": sha_file(__file__),
        "parameters": {
            "min_rows": min_rows,
            "min_groups": min_groups,
            "min_class_groups": 5,
            "requested_inner_folds": 5,
            "endpoint_priority_filter": False,
        },
        "feature_rows_index": "feature_rows_index.tsv",
        "tasks": tasks,
        "group_reconciliation": grouping,
        "derived_condition_evidence": evidence_summary,
        "derived_dramp_evidence": dramp_summary,
        "derived_dramp_representation": dramp_overlay_summary,
        "derived_mhc10_representation": mhc10_summary,
        "derived_cytotoxicity_evidence": cytotoxicity_summary,
        "derived_cytotoxicity_representation": cytotoxicity_overlay_summary,
        "derived_clint_admissions": clint_summary,
        "derived_pk_quality_evidence": pk_quality_summary,
        "derived_dbaasp_primary_evidence": primary_summary,
        "derived_hc10_secondary_evidence": hc10_secondary_summary,
        "hemolytik_condition_evidence": hemolytik_summary,
        "counts": {
            "input_records": len(records),
            "feature_index_records": len(compact),
            "endpoints_accounted": len(endpoint_summary),
            "endpoints_with_eligible_tasks": sum(r["eligible_tasks"] > 0 for r in endpoint_summary),
            "eligible_tasks": sum(t["status"] == "eligible" for t in tasks),
            "eligible_table_records": sum(
                t["n_records"] for t in tasks if t["status"] == "eligible"
            ),
            "deferred_records": len(deferred),
            "deferred_source_records": sum(v["status"] == "deferred" for v in disposition.values()),
            "collapsed_source_copies": collapsed,
            "label_conflict_records": label_conflicts,
        },
        "scientific_limits": [
            "all endpoints are accounted; unresolved contexts are not pooled",
            "eligibility is a minimum data contract; "
            "strict fivefold feasibility is checked during training",
            "weak labels remain source-defined benchmarks",
            "grouping does not claim complete90% homology",
        ],
    }
    raw_zero = {
        r["record_id"]
        for r in records
        if r["task_kind"] == "classification" and float(r["value"]) == 0
    }
    indexed_zero = {
        r["record_id"]
        for r in compact
        if r["_raw"]["task_kind"] == "classification" and r["target"] == "0"
    }
    manifest["assertions"] = {
        "all_labelled_source_records_indexed": len(compact) == len(records),
        "all_labelled_ids_have_dispositions": set(disposition) == {r["record_id"] for r in records},
        "all_labelled_endpoints_accounted": set(by_endpoint) == {t["endpoint"] for t in tasks},
        "no_zero_binary_label_lost": raw_zero == indexed_zero,
        "representative_row_conservation": len(records)
        == manifest["counts"]["eligible_table_records"] + len(deferred) + collapsed,
        "source_manifest_unchanged": sha_file(dataset_dir / "manifest.json") == source_hash,
    }
    if not all(manifest["assertions"].values()):
        raise AssertionError(stable(manifest["assertions"]))
    (run_dir / "task_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--min-rows", type=int, default=40)
    parser.add_argument("--min-groups", type=int, default=10)
    parser.add_argument("--condition-evidence-map", type=Path)
    parser.add_argument("--hemolytik-evidence-map", type=Path)
    parser.add_argument("--dramp-evidence-map", type=Path)
    parser.add_argument("--dramp-representation-overlay", type=Path)
    parser.add_argument("--mhc10-representation-overlay", type=Path)
    parser.add_argument("--cytotoxicity-evidence-map", type=Path)
    parser.add_argument("--cytotoxicity-representation-overlay", type=Path)
    parser.add_argument("--clint-evidence-map", type=Path)
    parser.add_argument("--pk-quality-evidence-map", type=Path)
    parser.add_argument("--dbaasp-primary-evidence-map", type=Path)
    parser.add_argument("--hc10-secondary-evidence-map", type=Path)
    args = parser.parse_args()
    result = compile_all_tasks(
        args.dataset_dir,
        args.run_dir,
        args.base_run,
        args.min_rows,
        args.min_groups,
        args.condition_evidence_map,
        args.hemolytik_evidence_map,
        args.dramp_evidence_map,
        args.dramp_representation_overlay,
        args.mhc10_representation_overlay,
        args.cytotoxicity_evidence_map,
        args.cytotoxicity_representation_overlay,
        args.clint_evidence_map,
        args.pk_quality_evidence_map,
        args.dbaasp_primary_evidence_map,
        args.hc10_secondary_evidence_map,
    )
    print(json.dumps(result["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
