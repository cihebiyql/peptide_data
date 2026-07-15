from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_peptide_ml_cleaning_v1 as summary


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class StatisticalUtilityTests(unittest.TestCase):
    def test_describe_uses_stable_linear_quantiles(self) -> None:
        result = summary.describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["p25"], "1.75")
        self.assertEqual(result["median"], "2.5")
        self.assertEqual(result["p75"], "3.25")
        self.assertEqual(result["mean"], "2.5")

    def test_raw_sequence_case_is_not_erased(self) -> None:
        self.assertEqual(summary.sequence_class("ACDE"), ("standard_20aa", "ACDE"))
        self.assertEqual(summary.sequence_class("AcDE")[0], "modified_or_nonstandard")
        self.assertEqual(summary.sequence_class("N.A.")[0], "missing_or_sentinel")

    def test_model_residue_count_separates_sequence_helm_and_smiles(self) -> None:
        self.assertEqual(
            summary.model_residue_count(
                {"representation_type": "sequence", "representation_text": "SEQ:ACDE"}
            ),
            4,
        )
        self.assertEqual(
            summary.model_residue_count(
                {
                    "representation_type": "helm",
                    "representation_text": "HELM:PEPTIDE1{A.C.[Aib].D}$$$$",
                }
            ),
            4,
        )
        self.assertIsNone(
            summary.model_residue_count(
                {"representation_type": "smiles", "representation_text": "SMILES:NCCO"}
            )
        )


class FullStatisticalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_one = tempfile.TemporaryDirectory(prefix="peptide_stats_one_")
        cls._temp_two = tempfile.TemporaryDirectory(prefix="peptide_stats_two_")
        cls.output_one = Path(cls._temp_one.name)
        cls.output_two = Path(cls._temp_two.name)
        cls.summary_one = summary.build(output_dir=cls.output_one)
        cls.summary_two = summary.build(output_dir=cls.output_two)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_one.cleanup()
        cls._temp_two.cleanup()

    def test_all_declared_and_raw_endpoints_are_reported(self) -> None:
        raw = read_tsv(self.output_one / "v15_raw_endpoint_inventory.tsv")
        clean = read_tsv(self.output_one / "clean_endpoint_inventory.tsv")
        self.assertEqual(len(raw), 50)
        self.assertEqual(sum(int(row["rows"]) for row in raw), 215568)
        self.assertEqual(len(clean), 11)
        kp = next(row for row in clean if row["endpoint_family"] == "Kp")
        self.assertEqual(kp["representative_rows"], "0")
        self.assertEqual(kp["input_member_rows"], "0")

    def test_clean_partition_and_endpoint_counts_are_conserved(self) -> None:
        clean = read_tsv(self.output_one / "clean_endpoint_inventory.tsv")
        self.assertEqual(sum(int(row["representative_rows"]) for row in clean), 19979)
        self.assertEqual(
            sum(int(row["partition_strict_numeric"]) for row in clean), 11591
        )
        self.assertEqual(
            sum(int(row["partition_binary_evidence_catalog"]) for row in clean),
            3930,
        )
        self.assertEqual(
            sum(int(row["partition_positive_unlabeled"]) for row in clean), 920
        )

    def test_sequence_projection_and_selected_representation_denominators_are_distinct(self) -> None:
        sequence_rows = read_tsv(self.output_one / "sequence_representation_summary.tsv")
        selected = read_tsv(self.output_one / "selected_representation_distribution.tsv")
        global_sequence = next(
            row
            for row in sequence_rows
            if row["data_layer"] == "ml_clean_representatives"
            and row["endpoint"] == "__ALL__"
        )
        self.assertEqual(global_sequence["raw_sequence_nonblank_rows"], "17972")
        self.assertEqual(global_sequence["raw_sequence_present_rows"], "17908")
        self.assertEqual(global_sequence["raw_exact_standard_20aa_rows"], "8152")
        global_selected = {
            row["representation_type"]: row
            for row in selected
            if row["endpoint_family"] == "__ALL__" and row["partition"] == "all"
        }
        self.assertEqual(global_selected["sequence"]["rows"], "2053")
        self.assertEqual(global_selected["helm"]["rows"], "14435")
        self.assertEqual(global_selected["smiles"]["rows"], "2410")
        self.assertEqual(global_selected["missing"]["rows"], "1081")
        self.assertEqual(global_selected["sequence"]["model_residue_count_max"], "100")

    def test_strict_numeric_statistics_are_task_and_unit_conditioned(self) -> None:
        tasks = read_tsv(self.output_one / "strict_numeric_task_distribution.tsv")
        endpoint_units = read_tsv(
            self.output_one / "strict_numeric_endpoint_unit_distribution.tsv"
        )
        self.assertEqual(len(tasks), 620)
        self.assertEqual(sum(int(row["n"]) for row in tasks), 11591)
        self.assertTrue(all(row["normalized_unit"] for row in tasks))
        self.assertTrue(all(row["descriptive_only"] == "true" for row in endpoint_units))
        keys = {
            (
                row["endpoint_family"],
                row["parameter_semantics"],
                row["parameter_basis"],
                row["assay_family"],
                row["normalized_unit"],
                row["target_transform"],
            )
            for row in endpoint_units
        }
        self.assertEqual(len(keys), len(endpoint_units))

    def test_binary_censored_and_literature_lanes_remain_separate(self) -> None:
        binary = read_tsv(self.output_one / "binary_task_distribution.tsv")
        censored = read_tsv(self.output_one / "censoring_distribution.tsv")
        literature = read_tsv(self.output_one / "literature_review_distribution.tsv")
        self.assertEqual(sum(int(row["rows"]) for row in binary), 4850)
        self.assertEqual(
            sum(int(row["positive_rows"]) for row in binary if row["is_positive_unlabeled"] == "true"),
            920,
        )
        self.assertEqual(sum(int(row["n"]) for row in censored), 230)
        self.assertTrue(all(row["point_regression_eligible"] == "false" for row in censored))
        self.assertEqual(sum(int(row["rows"]) for row in literature), 1560)
        self.assertTrue(all(row["training_rows"] == "0" for row in literature))

    def test_task_fragmentation_and_anomalies_are_explicit(self) -> None:
        viability = read_tsv(self.output_one / "task_size_viability.tsv")
        anomalies = read_tsv(self.output_one / "row_anomalies.tsv")
        self.assertEqual(len(viability), 1067)
        self.assertEqual(
            sum(
                row["size_bin_strict_numeric"] == "1"
                for row in viability
            ),
            413,
        )
        rules = {row["rule"] for row in anomalies}
        self.assertIn("standard_sequence_gt_100_aa_in_strict_numeric", rules)
        self.assertIn("subsecond_protease_half_life", rules)
        self.assertIn("extreme_total_body_clearance_value_gt_100_L_h", rules)
        self.assertEqual(
            sum(row["rule"] == "standard_sequence_gt_100_aa_in_strict_numeric" for row in anomalies),
            94,
        )

    def test_manifest_assertions_hashes_and_reproducibility(self) -> None:
        self.assertTrue(all(self.summary_one["assertions"].values()))
        for metadata in self.summary_one["outputs"].values():
            path = Path(metadata["path"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), metadata["sha256"])
        first = sorted(path.name for path in self.output_one.glob("*.tsv"))
        second = sorted(path.name for path in self.output_two.glob("*.tsv"))
        self.assertEqual(first, second)
        for name in first:
            self.assertEqual(
                (self.output_one / name).read_bytes(),
                (self.output_two / name).read_bytes(),
                name,
            )
        self.assertEqual(
            (self.output_one / "README.md").read_bytes(),
            (self.output_two / "README.md").read_bytes(),
        )
        on_disk = json.loads((self.output_one / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk["counts"], self.summary_one["counts"])


if __name__ == "__main__":
    unittest.main()
