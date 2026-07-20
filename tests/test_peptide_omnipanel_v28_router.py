from __future__ import annotations

import unittest

from peptide_omnipanel_v28.v28_registry import EndpointRegistry
from peptide_omnipanel_v28.v28_router import AlwaysReturnRouter, EndpointCandidate


def registry() -> EndpointRegistry:
    return EndpointRegistry.from_mapping(
        {
            "schema_version": "v28-router-test-1",
            "ownership_contract": {
                "third_party_prediction_api_allowed": False,
                "third_party_model_weights_allowed_in_release": False,
            },
            "endpoints": [
                {
                    "endpoint_id": "LogD7.4",
                    "panel": "core_admet_pbpk",
                    "task_kind": "regression",
                    "primary_output": "logd_7_4",
                    "unit": "dimensionless",
                    "required_representation": ["sequence"],
                    "fallback_policy": ["A", "B", "C", "D", "E"],
                    "allowed_tiers": ["A", "B", "C", "D", "E"],
                    "semantic_constraints": {"pH": "7.4"},
                },
                {
                    "endpoint_id": "F",
                    "panel": "core_admet_pbpk",
                    "task_kind": "binary_classification",
                    "primary_output": "p_high_bioavailability",
                    "unit": "probability",
                    "required_representation": ["sequence"],
                    "fallback_policy": ["C", "E"],
                    "allowed_tiers": ["C", "E"],
                    "semantic_constraints": {"continuous_percent_allowed": False},
                },
                {
                    "endpoint_id": "Kp",
                    "panel": "core_admet_pbpk",
                    "task_kind": "mechanistic_tissue_vector",
                    "primary_output": "kp_tissue_plasma",
                    "unit": "dimensionless_ratio",
                    "required_representation": ["sequence"],
                    "fallback_policy": ["D", "E"],
                    "allowed_tiers": ["D", "E"],
                    "semantic_constraints": {"ratio_orientation": "tissue_to_plasma"},
                },
                {
                    "endpoint_id": "degradation_site_probability",
                    "panel": "peptide_specific",
                    "task_kind": "sequence_labeling",
                    "primary_output": "per_residue_degradation_probability",
                    "unit": "probability_vector",
                    "required_representation": ["sequence"],
                    "fallback_policy": ["D", "E"],
                    "allowed_tiers": ["D", "E"],
                    "semantic_constraints": {"axis": "sequence_position"},
                },
            ],
        }
    )


def returns(value: float, **fields: object):
    def predictor(_context):
        return {"value": value, **fields}

    return predictor


class V28RouterTests(unittest.TestCase):
    def test_highest_working_tier_is_selected_and_all_fields_return(self) -> None:
        router = AlwaysReturnRouter(
            registry(),
            [
                EndpointCandidate(
                    "LogD7.4", "A", "direct", "local", returns(1.2), peptide_validated=True
                ),
                EndpointCandidate("LogD7.4", "E", "prior", "local", returns(0.3)),
                EndpointCandidate(
                    "F", "C", "public_raw_data_model", "local", returns(0.61, probability=0.61)
                ),
                EndpointCandidate("F", "E", "prior", "local", returns(0.5, probability=0.5)),
                EndpointCandidate("Kp", "E", "prior", "local", returns({"brain": 0.2})),
                EndpointCandidate(
                    "degradation_site_probability",
                    "E",
                    "prior",
                    "local",
                    returns([0.1, 0.2, 0.3, 0.4]),
                ),
            ],
        )
        result = router.predict("ACDE")

        self.assertEqual(result.endpoints["LogD7.4"]["prediction_tier"], "A")
        self.assertEqual(result.endpoints["LogD7.4"]["status"], "predicted")
        self.assertEqual(result.endpoints["F"]["prediction_tier"], "C")
        self.assertEqual(result.endpoints["F"]["status"], "predicted_low_evidence")
        self.assertFalse(result.endpoints["F"]["coverage_guaranteed"])
        self.assertEqual(
            set(result.endpoints), {"LogD7.4", "F", "Kp", "degradation_site_probability"}
        )
        self.assertEqual(result.endpoints["Kp"]["value"], {"brain": 0.2})
        self.assertEqual(
            result.endpoints["degradation_site_probability"]["value"], [0.1, 0.2, 0.3, 0.4]
        )
        self.assertTrue(result.endpoints["Kp"]["research_only"])
        self.assertTrue(result.endpoints["Kp"]["low_confidence"])

    def test_failure_falls_through_to_local_prior_with_warning(self) -> None:
        def broken(_context):
            raise RuntimeError("expected")

        router = AlwaysReturnRouter(
            registry(),
            [
                EndpointCandidate("LogD7.4", "A", "broken", "local", broken),
                EndpointCandidate("LogD7.4", "E", "prior", "local", returns(0.3)),
                EndpointCandidate("F", "E", "prior", "local", returns(0.5)),
                EndpointCandidate("Kp", "E", "prior", "local", returns({"brain": 0.2})),
                EndpointCandidate(
                    "degradation_site_probability", "E", "prior", "local", returns([0.1] * 4)
                ),
            ],
        )
        result = router.predict("ACDE")

        prediction = result.endpoints["LogD7.4"]
        self.assertEqual(prediction["prediction_tier"], "E")
        self.assertIn("higher_tier_candidate_unavailable", prediction["warnings"])

    def test_rejects_external_or_missing_prior_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a locally owned E-tier"):
            AlwaysReturnRouter(
                registry(),
                [
                    EndpointCandidate("LogD7.4", "A", "direct", "local", returns(1.0)),
                    EndpointCandidate("F", "E", "prior", "local", returns(0.5)),
                    EndpointCandidate("Kp", "E", "prior", "local", returns({"brain": 0.2})),
                    EndpointCandidate(
                        "degradation_site_probability", "E", "prior", "local", returns([0.1] * 4)
                    ),
                ],
            )

        with self.assertRaisesRegex(ValueError, "locally owned"):
            EndpointCandidate(
                "LogD7.4", "E", "external", "remote", returns(0.0), locally_owned=False
            )

        with self.assertRaisesRegex(ValueError, "network"):
            EndpointCandidate("LogD7.4", "E", "remote", "remote", returns(0.0), uses_network=True)

    def test_rejects_wrong_structured_prediction_shape(self) -> None:
        router = AlwaysReturnRouter(
            registry(),
            [
                EndpointCandidate("LogD7.4", "E", "prior", "local", returns(0.2)),
                EndpointCandidate("F", "E", "prior", "local", returns(0.5)),
                EndpointCandidate("Kp", "E", "prior", "local", returns(0.2)),
                EndpointCandidate(
                    "degradation_site_probability", "E", "prior", "local", returns([0.1, 0.2])
                ),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "all self-owned candidates failed"):
            router.predict("ACDE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
