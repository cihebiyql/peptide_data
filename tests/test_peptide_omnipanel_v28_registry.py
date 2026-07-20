from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_PATH = ROOT / "configs" / "peptide_omnipanel_v28_endpoint_registry.json"
SOURCE_PATH = ROOT / "configs" / "peptide_omnipanel_v28_source_registry.json"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_endpoint_registry_has_unique_complete_panel_contract() -> None:
    registry = _load(ENDPOINT_PATH)
    assert registry["schema_version"] == "peptide-omnipanel-v28-endpoint-registry-1"
    endpoints = registry["endpoints"]
    ids = [item["endpoint_id"] for item in endpoints]
    assert len(ids) == len(set(ids)) == 39

    expected_counts = {
        "core_admet_pbpk": 10,
        "extended_admet_tox": 21,
        "peptide_specific": 8,
    }
    assert registry["panel_counts"] == expected_counts
    assert {
        panel: sum(item["panel"] == panel for item in endpoints)
        for panel in expected_counts
    } == expected_counts

    required = {
        "endpoint_id",
        "panel",
        "task_kind",
        "primary_output",
        "unit",
        "required_representation",
        "fallback_policy",
        "allowed_tiers",
        "semantic_constraints",
    }
    for item in endpoints:
        assert required <= item.keys()
        assert item["fallback_policy"], item["endpoint_id"]
        assert item["allowed_tiers"], item["endpoint_id"]
        assert item["required_representation"]
        assert all(isinstance(value, str) and value for value in item["required_representation"])
        assert item["representation_contract"]["minimum_input"] == "sequence"
        assert item["semantic_constraints"]["rules"]
        assert item["evidence_tier_restrictions"]["never_count_as_measured"]


def test_core_semantics_keep_F_CL_Papp_and_Kp_distinct() -> None:
    endpoints = {
        item["endpoint_id"]: item for item in _load(ENDPOINT_PATH)["endpoints"]
    }
    assert endpoints["F"]["primary_output"] == "p_high_bioavailability"
    assert endpoints["F"]["unit"] == "probability"
    assert endpoints["F"]["semantic_constraints"]["continuous_percent_allowed"] is False

    assert endpoints["CL"]["semantic_constraints"]["distinct_suboutputs"] == [
        "total_clearance",
        "hepatocyte_CLint",
        "microsome_CLint",
    ]
    cl_outputs = {item["field_id"] for item in endpoints["CL"]["output_semantics"]}
    assert cl_outputs == {
        "cl_total_ml_min_kg",
        "cl_hepatocyte_intrinsic_uL_min_1e6_cells",
        "cl_microsome_intrinsic_uL_min_mg_protein",
    }

    assert endpoints["permeability"]["primary_output"] == "assay_specific_log10_papp_cm_s"
    assert endpoints["permeability"]["semantic_constraints"]["must_not_alias"] == ["Kp"]
    assert endpoints["Kp"]["primary_output"] == "kp_tissue_plasma"
    assert endpoints["Kp"]["task_kind"] == "mechanistic_tissue_vector"
    assert endpoints["Kp"]["semantic_constraints"]["must_not_alias"] == [
        "permeability",
        "Papp",
    ]
    assert endpoints["Kp"]["unit"] != endpoints["permeability"]["unit"]


def test_no_third_party_prediction_api_or_weights_are_enabled() -> None:
    endpoint_registry = _load(ENDPOINT_PATH)
    source_registry = _load(SOURCE_PATH)
    for registry in (endpoint_registry, source_registry):
        ownership = registry["ownership_contract"]
        assert ownership["third_party_prediction_api_allowed"] is False
        assert ownership["third_party_model_weights_allowed_in_release"] is False

    policy = source_registry["source_policy"]
    assert policy["third_party_prediction_api_allowed"] is False
    assert policy["third_party_model_weights_allowed"] is False
    assert policy["third_party_predictions_as_measured_labels_allowed"] is False

    forbidden_route_tokens = ("ADMET_AI", "pepADMET", "OpenADMET", "PKSmart")
    for item in endpoint_registry["endpoints"]:
        route_text = " ".join(item["fallback_policy"])
        assert set(item["fallback_policy"]) <= {"A", "B", "C", "D", "E"}
        assert item["fallback_policy"][-1] == "E"
        assert not any(token in route_text for token in forbidden_route_tokens)


def test_source_registry_is_closed_over_endpoints_and_pending_sources_fail_closed() -> None:
    endpoint_ids = {
        item["endpoint_id"] for item in _load(ENDPOINT_PATH)["endpoints"]
    }
    registry = _load(SOURCE_PATH)
    sources = registry["sources"]
    source_ids = [item["source_id"] for item in sources]
    assert len(source_ids) == len(set(source_ids))
    assert any(item["status"] == "local_frozen" for item in sources)
    assert any(item["status"].startswith("pending_") for item in sources)

    for item in sources:
        assert set(item["endpoint_ids"]) <= endpoint_ids
        assert "prediction_api" not in item["source_kind"]
        assert "model_weights" not in item["source_kind"]
        if item["status"].startswith("pending_"):
            assert item["manifest_path"] is None
            assert item["manifest_sha256"] is None
            assert item["license_status"] == "not_cleared"
            assert item["provenance_status"] == "not_frozen"
            assert item["usage_policy"].startswith("blocked_from_training_until_")
        elif item["manifest_path"] is not None:
            manifest = ROOT / item["manifest_path"]
            assert manifest.is_file(), item["source_id"]
            assert _sha256(manifest) == item["manifest_sha256"]


def test_evidence_tiers_never_promote_computed_or_pseudo_labels_to_measured() -> None:
    registry = _load(ENDPOINT_PATH)
    for item in registry["endpoints"]:
        restrictions = item["evidence_tier_restrictions"]
        formal = set(restrictions["formal_evaluation_allowed"])
        assert "L4_mechanism_computed" not in formal
        assert "L5_self_model_pseudo_label" not in formal
        never_measured = set(restrictions["never_count_as_measured"])
        assert "L4_mechanism_computed" in never_measured
        assert "L5_self_model_pseudo_label" in never_measured
