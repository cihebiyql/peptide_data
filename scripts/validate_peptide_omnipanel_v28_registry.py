"""Validate the static V2.8 self-trained endpoint/source registry bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from peptide_omnipanel_v28.v28_registry import (  # noqa: E402
    EndpointRegistry,
    SourceRegistry,
    load_endpoint_registry,
    load_source_registry,
)

CORE_ENDPOINT_IDS = frozenset(
    {
        "LogD7.4",
        "solubility",
        "F",
        "T1/2",
        "PPB",
        "CL",
        "Vd",
        "BBB",
        "permeability",
        "Kp",
    }
)


def validate_semantics(registry: EndpointRegistry) -> dict[str, object]:
    endpoint_ids = {endpoint.endpoint_id for endpoint in registry.endpoints}
    missing_core = sorted(CORE_ENDPOINT_IDS.difference(endpoint_ids))
    if missing_core:
        raise ValueError("core endpoint IDs are missing: " + ", ".join(missing_core))

    f_constraints = registry.endpoint("F").semantic_constraints
    if f_constraints.get("continuous_percent_allowed") is not False:
        raise ValueError("F must forbid converting a class probability into F percent")

    kp_constraints = registry.endpoint("Kp").semantic_constraints
    permeability_constraints = registry.endpoint("permeability").semantic_constraints
    if "Papp" not in kp_constraints.get("must_not_alias", []):
        raise ValueError("Kp must explicitly forbid a Papp alias")
    if "Kp" not in permeability_constraints.get("must_not_alias", []):
        raise ValueError("permeability must explicitly forbid a Kp alias")

    cl_constraints = registry.endpoint("CL").semantic_constraints
    expected_clearance = {"total_clearance", "hepatocyte_CLint", "microsome_CLint"}
    actual_clearance = set(cl_constraints.get("distinct_suboutputs", []))
    if actual_clearance != expected_clearance:
        raise ValueError("CL must retain total/hepatocyte/microsome as distinct suboutputs")
    return {
        "core_endpoint_count": len(CORE_ENDPOINT_IDS),
        "registered_endpoint_count": len(registry.endpoints),
        "panels": sorted({endpoint.panel for endpoint in registry.endpoints}),
    }


def validate_sources(sources: SourceRegistry) -> dict[str, object]:
    forbidden = [
        source.source_id
        for source in sources.sources
        if source.source_kind in {"third_party_model_weight", "prediction_api"}
    ]
    if forbidden:
        raise ValueError("forbidden model sources: " + ", ".join(forbidden))
    raw_sources = sum(source.raw_data_allowed for source in sources.sources)
    if raw_sources == 0:
        raise ValueError("at least one raw-data source must be enabled")
    return {
        "registered_source_count": len(sources.sources),
        "raw_data_source_count": raw_sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-registry",
        type=Path,
        default=ROOT / "configs" / "peptide_omnipanel_v28_endpoint_registry.json",
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=ROOT / "configs" / "peptide_omnipanel_v28_source_registry.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = load_endpoint_registry(args.endpoint_registry)
    sources = load_source_registry(args.source_registry)
    report = {
        "status": "pass",
        "endpoint_registry": str(args.endpoint_registry),
        "source_registry": str(args.source_registry),
        **validate_semantics(endpoints),
        **validate_sources(sources),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
