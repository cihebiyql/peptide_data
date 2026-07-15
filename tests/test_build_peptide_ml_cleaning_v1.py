from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import build_peptide_ml_cleaning_v1 as builder


ROOT = Path(__file__).resolve().parents[1]
V15_MASTER = (
    ROOT
    / "data"
    / "peptide_property_expansion_v15"
    / "peptide_property_observations_v15.tsv"
)
EXPECTED_V15_MASTER_SHA256 = (
    "9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf"
)


def base_row(**updates: str) -> dict[str, str]:
    row = {
        "record_id": "TEST:1",
        "source_id": "primary_article_test",
        "source_record_id": "row-1",
        "sequence": "ACDEFGHIK",
        "helm": "",
        "smiles": "",
        "inchikey": "",
        "endpoint": "T1/2",
        "endpoint_detail": "terminal_half_life",
        "value": "2",
        "relation": "=",
        "unit": "h",
        "label": "",
        "label_rule": "",
        "task_type": "regression",
        "species": "rat",
        "matrix": "plasma",
        "assay": "LC-MS/MS",
        "route": "IV",
        "dose": "",
        "timepoint": "",
        "context_json": "{}",
        "source_url": "https://example.org/test",
        "doi_pmid": "10.1000/test",
        "license": "test-only",
        "evidence_tier": "primary_article",
        "train_target_eligible": "true",
        "quality_flags": "",
        "input_dataset": "unit_test",
        "identity_type": "sequence",
        "identity_key": "sequence:test",
        "observation_key": "obs:test",
        "duplicate_group_size": "1",
        "baseline_overlap": "false",
        "eligible_after_qc": "true",
        "eligibility_reason": "eligible",
    }
    row.update(updates)
    return row


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class SemanticCleaningUnitTests(unittest.TestCase):
    def test_pampa_and_caco2_are_distinct_tasks(self) -> None:
        pampa = builder.normalize_row(
            base_row(
                endpoint="permeability",
                endpoint_detail="PAMPA",
                value="-8.1",
                unit="source_log10_permeability",
            )
        )
        caco2 = builder.normalize_row(
            base_row(
                endpoint="permeability",
                endpoint_detail="Caco2",
                value="-8.1",
                unit="source_log10_permeability",
            )
        )
        self.assertNotEqual(pampa["task_id"], caco2["task_id"])
        self.assertEqual(pampa["assay_family"], "pampa")
        self.assertEqual(caco2["assay_family"], "caco2")

    def test_systemic_and_plasma_stability_half_life_are_distinct(self) -> None:
        systemic = builder.normalize_row(base_row())
        stability = builder.normalize_row(
            base_row(
                endpoint_detail="human plasma stability half-life",
                route="",
                assay="plasma stability assay",
            )
        )
        self.assertEqual(systemic["parameter_semantics"], "systemic_terminal_pk")
        self.assertEqual(stability["parameter_semantics"], "plasma_serum_stability")
        self.assertNotEqual(systemic["task_id"], stability["task_id"])

    def test_f_routes_are_distinct_tasks(self) -> None:
        oral = builder.normalize_row(
            base_row(
                endpoint="F",
                endpoint_detail="absolute oral bioavailability",
                value="25",
                unit="%",
                route="oral",
            )
        )
        subcutaneous = builder.normalize_row(
            base_row(
                endpoint="F",
                endpoint_detail="absolute bioavailability",
                value="25",
                unit="%",
                route="subcutaneous",
            )
        )
        self.assertNotEqual(oral["task_id"], subcutaneous["task_id"])

    def test_pk_dose_and_timepoint_are_preserved_as_conditions(self) -> None:
        low_dose = builder.normalize_row(
            base_row(
                endpoint="CL",
                endpoint_detail="systemic clearance",
                value="5",
                unit="mL/min/kg",
                dose="1 mg/kg",
                timepoint="0-8 h",
            )
        )
        high_dose = builder.normalize_row(
            base_row(
                endpoint="CL",
                endpoint_detail="systemic clearance",
                value="5",
                unit="mL/min/kg",
                dose="10 mg/kg",
                timepoint="0-24 h",
            )
        )
        self.assertNotEqual(low_dose["task_id"], high_dose["task_id"])
        self.assertIn('"dose":"1_mg_kg"', low_dose["condition_json"])
        self.assertIn('"timepoint":"0_8_h"', low_dose["condition_json"])
        self.assertEqual(low_dose["raw_dose"], "1 mg/kg")
        self.assertEqual(low_dose["raw_timepoint"], "0-8 h")

    def test_clearance_semantics_do_not_mix(self) -> None:
        systemic = builder.normalize_row(
            base_row(
                endpoint="CL",
                endpoint_detail="systemic clearance",
                value="5",
                unit="mL/min/kg",
            )
        )
        apparent = builder.normalize_row(
            base_row(
                endpoint="CL",
                endpoint_detail="apparent clearance CL/F",
                value="5",
                unit="L/h",
            )
        )
        microsomal = builder.normalize_row(
            base_row(
                endpoint="CL",
                endpoint_detail="microsomal intrinsic clearance",
                value="5",
                unit="uL/min/mg protein",
                assay="liver microsomes",
            )
        )
        self.assertEqual(
            len({systemic["task_id"], apparent["task_id"], microsomal["task_id"]}),
            3,
        )

    def test_volume_subtypes_and_apparent_volume_do_not_mix(self) -> None:
        definitions = (
            ("volume of distribution", "L/kg"),
            ("steady-state volume of distribution Vss", "L/kg"),
            ("terminal volume Vz", "L/kg"),
            ("apparent terminal volume Vz_over_F", "L/kg"),
            ("central volume Vc", "L"),
            ("peripheral volume Vp", "L"),
        )
        tasks = {
            builder.normalize_row(
                base_row(endpoint="Vd", endpoint_detail=detail, value="1", unit=unit)
            )["task_id"]
            for detail, unit in definitions
        }
        self.assertEqual(len(tasks), len(definitions))

    def test_equivalent_time_units_collapse_as_one_observation(self) -> None:
        first = builder.normalize_row(base_row(record_id="TEST:120", value="120", unit="min"))
        second = builder.normalize_row(base_row(record_id="TEST:2", value="2", unit="h"))
        rows = [first, second]
        builder.finalize_keys(rows)
        representatives, duplicates = builder.collapse_normalized_duplicates(rows)
        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives[0]["member_count"], 2)
        self.assertEqual(len(duplicates), 1)

    def test_duplicate_collapse_preserves_all_source_lineage(self) -> None:
        first = builder.normalize_row(
            base_row(
                record_id="TEST:source-a",
                source_id="source-a",
                source_record_id="a-1",
                doi_pmid="10.1000/source-a",
                source_url="https://example.org/a",
                license="CC-BY-4.0",
            )
        )
        second = builder.normalize_row(
            base_row(
                record_id="TEST:source-b",
                source_id="source-b",
                source_record_id="b-1",
                doi_pmid="PMID:12345678",
                source_url="https://example.org/b",
                license="source-terms",
            )
        )
        rows = [first, second]
        builder.finalize_keys(rows)
        representatives, duplicates = builder.collapse_normalized_duplicates(rows)
        representative = representatives[0]
        self.assertEqual(len(representatives), 1)
        self.assertIn("DOI:10.1000/source-a", representative["source_group_ids"])
        self.assertIn("PMID:12345678", representative["source_group_ids"])
        self.assertIn("https://example.org/a", representative["source_url"])
        self.assertIn("https://example.org/b", representative["source_url"])
        self.assertIn("CC-BY-4.0", representative["license"])
        self.assertIn("source-terms", representative["license"])
        self.assertIn("TEST:source-a", representative["member_lineage_json"])
        self.assertIn("TEST:source-b", representative["member_lineage_json"])
        self.assertEqual(
            duplicates[0]["member_lineage_json"],
            representative["member_lineage_json"],
        )

    def test_merged_provenance_still_detects_overlapping_source_conflict(self) -> None:
        rows = [
            builder.normalize_row(
                base_row(
                    record_id="TEST:a-two",
                    source_id="source-a",
                    source_record_id="a-two",
                    doi_pmid="10.1000/source-a",
                    value="2",
                )
            ),
            builder.normalize_row(
                base_row(
                    record_id="TEST:b-two",
                    source_id="source-b",
                    source_record_id="b-two",
                    doi_pmid="10.1000/source-b",
                    value="2",
                )
            ),
            builder.normalize_row(
                base_row(
                    record_id="TEST:a-three",
                    source_id="source-a",
                    source_record_id="a-three",
                    doi_pmid="10.1000/source-a",
                    value="3",
                )
            ),
        ]
        builder.finalize_keys(rows)
        representatives, _ = builder.collapse_normalized_duplicates(rows)
        builder.finalize_keys(representatives)
        conflicts = builder.flag_conflicts(representatives)
        self.assertEqual(len(representatives), 2)
        self.assertTrue(all(row["hard_condition_conflict"] == "true" for row in representatives))
        self.assertEqual(conflicts[0]["hard_within_source_conflict"], "true")

    def test_duplicate_collapse_preserves_union_identity_aliases(self) -> None:
        inchikey = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
        rows = [
            builder.normalize_row(
                base_row(
                    record_id="TEST:smiles-alias",
                    identity_type="smiles",
                    smiles="C",
                    inchikey=inchikey,
                )
            ),
            builder.normalize_row(
                base_row(
                    record_id="TEST:helm-alias",
                    identity_type="helm",
                    helm="PEPTIDE1{A.C.D.E.F.G.H.I.K}$$$$",
                    inchikey=inchikey,
                )
            ),
        ]
        builder.finalize_keys(rows)
        original_key = rows[0]["normalized_observation_key"]
        representatives, duplicates = builder.collapse_normalized_duplicates(rows)
        self.assertEqual(len(representatives[0]["_identity_aliases"]), 3)
        builder.finalize_keys(representatives)
        self.assertEqual(representatives[0]["normalized_observation_key"], original_key)
        self.assertEqual(duplicates[0]["normalized_observation_key"], original_key)

    def test_identity_sentinels_and_modified_projection_are_blocked(self) -> None:
        sentinel = builder.safe_identity_aliases(
            {"sequence": "N.A.", "smiles": "N.A.", "identity_type": "smiles"}
        )
        modified = builder.safe_identity_aliases(
            {
                "sequence": "ACDEFGHIK",
                "smiles": "",
                "identity_type": "sequence",
                "quality_flags": "modified_or_noncanonical;residue_projection_only",
            }
        )
        self.assertEqual(sentinel[0], [])
        self.assertEqual(sentinel[2], "")
        self.assertEqual(modified[0], [])
        self.assertEqual(modified[2], "")
        self.assertEqual(modified[3], "false")

    def test_cyclic_structure_does_not_merge_with_linear_sequence(self) -> None:
        linear = builder.safe_identity_aliases(
            {"sequence": "ACDEFGHIK", "identity_type": "sequence"}
        )
        cyclic = builder.safe_identity_aliases(
            {
                "sequence": "ACDEFGHIK",
                "helm": "PEPTIDE1{A.C.D.E.F.G.H.I.K}$PEPTIDE1,PEPTIDE1,1:R1-9:R2$$$",
                "identity_type": "helm",
                "quality_flags": "modified_or_noncanonical",
            }
        )
        self.assertTrue(linear[0])
        self.assertTrue(cyclic[0])
        self.assertTrue(set(linear[0]).isdisjoint(cyclic[0]))
        self.assertEqual(linear[1], "sequence")
        self.assertEqual(cyclic[1], "helm")

    def test_censoring_is_preserved_outside_point_regression(self) -> None:
        upper = builder.normalize_row(base_row(relation="<", value="2"))
        lower = builder.normalize_row(base_row(relation=">", value="2"))
        approximate = builder.normalize_row(base_row(relation="~", value="2"))
        self.assertEqual(upper["censoring_type"], "upper_bound")
        self.assertEqual(lower["censoring_type"], "lower_bound")
        self.assertEqual(approximate["censoring_type"], "approximate")
        self.assertNotEqual(upper["censoring_type"], "exact")

    def test_source_log_permeability_is_not_logged_twice(self) -> None:
        row = builder.normalize_row(
            base_row(
                endpoint="permeability",
                endpoint_detail="PAMPA",
                value="-8.25",
                unit="source_log10_permeability",
            )
        )
        self.assertEqual(row["normalized_value"], "-8.25")
        self.assertEqual(row["conversion_formula"], "already_log10_no_retransform")

    def test_ppb_bound_direction_is_normalized_and_free_is_reviewed(self) -> None:
        percent = builder.normalize_row(
            base_row(endpoint="PPB", value="80", unit="% bound")
        )
        fraction = builder.normalize_row(
            base_row(endpoint="PPB", value="0.8", unit="fraction")
        )
        free = builder.normalize_row(
            base_row(endpoint="PPB", value="20", unit="% free")
        )
        self.assertEqual(percent["normalized_value"], "0.8")
        self.assertEqual(fraction["normalized_value"], "0.8")
        self.assertEqual(free["normalization_status"], "review")

    def test_bbb_weak_classes_share_a_task_but_not_strict_evidence(self) -> None:
        positive = builder.normalize_row(
            base_row(
                endpoint="BBB",
                endpoint_detail="weak BBB benchmark",
                value="",
                unit="",
                label="1",
                task_type="binary",
                context_json='{"label_origin":"compiled_positive"}',
            )
        )
        negative = builder.normalize_row(
            base_row(
                endpoint="BBB",
                endpoint_detail="weak BBB benchmark",
                value="",
                unit="",
                label="0",
                task_type="binary",
                context_json='{"label_origin":"rule_constructed_negative"}',
            )
        )
        self.assertEqual(positive["task_id"], negative["task_id"])
        self.assertEqual(positive["binary_evidence_class"], "database_positive_membership")
        self.assertEqual(negative["binary_evidence_class"], "rule_constructed_negative")
        self.assertEqual(positive["binary_use_tier"], "weak_benchmark_reproduction_only")

    def test_positive_only_membership_does_not_infer_negatives(self) -> None:
        row = builder.normalize_row(
            base_row(
                endpoint="cell_penetration",
                endpoint_detail="cell penetrating peptide",
                value="",
                unit="",
                label="1",
                task_type="binary",
            )
        )
        self.assertEqual(row["task_kind"], "positive_unlabeled")
        self.assertIn("not available", row["negative_class"])
        self.assertEqual(row["research_train_eligible"], "false")


class FullReleaseRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_one = tempfile.TemporaryDirectory(prefix="peptide_ml_clean_one_")
        cls._temp_two = tempfile.TemporaryDirectory(prefix="peptide_ml_clean_two_")
        cls.output_one = Path(cls._temp_one.name)
        cls.output_two = Path(cls._temp_two.name)
        cls.manifest_one = builder.build(output_dir=cls.output_one)
        cls.manifest_two = builder.build(output_dir=cls.output_two)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_one.cleanup()
        cls._temp_two.cleanup()

    def test_release_counts_and_safety_gates(self) -> None:
        counts = self.manifest_one["counts"]
        assertions = self.manifest_one["assertions"]
        self.assertEqual(counts["input_scope_rows"], 20052)
        self.assertEqual(counts["representative_rows"], 19979)
        self.assertEqual(counts["binary_catalog_rows"], 3930)
        self.assertEqual(counts["positive_unlabeled_rows"], 920)
        self.assertGreater(counts["strict_numeric_rows"], 10000)
        self.assertTrue(all(assertions.values()))

    def test_strict_numeric_is_exact_and_model_representable(self) -> None:
        rows = read_tsv(self.output_one / "strict_numeric.tsv")
        self.assertTrue(rows)
        self.assertTrue(
            all(
                row["normalized_relation"] == "="
                and row["censoring_type"] == "exact"
                and row["identity_group_id"]
                and row["representation_text"]
                and row["upstream_eligible_after_qc"] == "true"
                for row in rows
            )
        )

    def test_literature_staging_remains_review_only(self) -> None:
        rows = read_tsv(self.output_one / "literature_review_cleaned.tsv")
        self.assertEqual(len(rows), 1560)
        self.assertTrue(all(row["partition"] == "literature_review" for row in rows))
        self.assertTrue(all(row["research_train_eligible"] == "false" for row in rows))

    def test_peplife_pseudo_identities_and_modified_projections_are_blocked(self) -> None:
        rows = read_tsv(self.output_one / "review_observations.tsv")
        peplife = [
            row
            for row in rows
            if row["partition"] == "identity_review" and row["source_id"] == "PEPlife2"
        ]
        self.assertEqual(sum(row["smiles"] in {"N.A.", "NA"} for row in peplife), 896)
        self.assertEqual(len(peplife), 898)
        self.assertTrue(
            all(not row["identity_group_id"] and not row["representation_text"] for row in peplife)
        )

    def test_binary_solubility_is_conditioned_and_weak_tasks_are_not_strict(self) -> None:
        binary = read_tsv(self.output_one / "binary_evidence_catalog.tsv")
        registry = read_tsv(self.output_one / "task_registry.tsv")
        solubility = [row for row in binary if row["endpoint_family"] == "solubility"]
        self.assertEqual(len({row["task_id"] for row in solubility}), 7)
        self.assertTrue(all(row["hard_condition_conflict"] == "false" for row in solubility))
        weak_tiers = {
            "derived_binary_research",
            "rough_solvent_conditioned_benchmark",
            "weak_benchmark_reproduction_only",
        }
        self.assertTrue(
            all(
                row["strict_modelable"] == "false"
                for row in registry
                if row["binary_use_tiers"] in weak_tiers
            )
        )

    def test_censored_sidecar_preserves_bounds_and_is_ineligible(self) -> None:
        rows = read_tsv(self.output_one / "censored_observations.tsv")
        self.assertTrue(rows)
        self.assertTrue(all(row["research_train_eligible"] == "false" for row in rows))
        for row in rows:
            if row["censoring_type"] == "upper_bound":
                self.assertTrue(row["normalized_upper"])
                self.assertFalse(row["normalized_lower"])
            elif row["censoring_type"] == "lower_bound":
                self.assertTrue(row["normalized_lower"])
                self.assertFalse(row["normalized_upper"])
            else:
                self.assertEqual(row["censoring_type"], "approximate")
                self.assertTrue(row["normalized_lower"])
                self.assertTrue(row["normalized_upper"])

    def test_emitted_partition_conservation_and_artifact_hashes(self) -> None:
        normalized = read_tsv(self.output_one / "normalized_observations.tsv")
        duplicates = read_tsv(self.output_one / "exact_duplicates.tsv")
        partition_names = (
            "strict_numeric.tsv",
            "censored_observations.tsv",
            "binary_evidence_catalog.tsv",
            "positive_unlabeled.tsv",
            "review_observations.tsv",
        )
        self.assertEqual(
            len(normalized),
            sum(len(read_tsv(self.output_one / name)) for name in partition_names),
        )
        self.assertEqual(sum(int(row["member_count"]) for row in normalized), 20052)
        self.assertTrue(
            all(
                len(json.loads(row["member_lineage_json"]))
                == int(row["member_count"])
                for row in normalized
            )
        )
        normalized_keys = {
            row["normalized_observation_key"]
            for row in normalized
            if row["normalized_observation_key"]
        }
        self.assertTrue(
            {row["normalized_observation_key"] for row in duplicates}.issubset(
                normalized_keys
            )
        )
        for metadata in self.manifest_one["outputs"].values():
            path = Path(metadata["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])

    def test_all_tsv_outputs_are_byte_reproducible(self) -> None:
        first = sorted(path.name for path in self.output_one.glob("*.tsv"))
        second = sorted(path.name for path in self.output_two.glob("*.tsv"))
        self.assertEqual(first, second)
        for name in first:
            self.assertEqual(
                (self.output_one / name).read_bytes(),
                (self.output_two / name).read_bytes(),
                name,
            )

    def test_v15_master_hash_is_unchanged(self) -> None:
        digest = hashlib.sha256(V15_MASTER.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_V15_MASTER_SHA256)


if __name__ == "__main__":
    unittest.main()
