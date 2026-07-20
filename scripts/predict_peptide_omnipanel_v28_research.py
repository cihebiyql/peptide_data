"""Run the isolated, local, research-only V2.8 39-endpoint panel.

This command intentionally does not import or modify ``peptide_omnipanel.service``.
Every endpoint response is explicitly tagged ``research_only=true`` and
``low_confidence=true``.  It is a reproducible internal display surface, not a
clinical, release, or active-service prediction API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from peptide_omnipanel_v28.v28_research_panel import build_research_only_router

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "peptide_omnipanel_v28_endpoint_registry.json"
DEFAULT_BUNDLE = (
    ROOT
    / "data"
    / "peptide_omnipanel_v28_research_bundle_20260719_run1"
    / "research_bundle_manifest.json"
)


def research_response(
    sequence: str, *, input_format: str, registry: Path, bundle: Path
) -> dict[str, Any]:
    """Create a display-ready response while preserving the research boundary."""

    router = build_research_only_router(
        endpoint_registry_path=registry,
        research_bundle_path=bundle,
    )
    response = router.predict(sequence, input_format=input_format).to_dict()
    response["panel_metadata"] = {
        "architecture_id": "peptide_omnipanel_v28_research_only",
        "scope": "internal_research_only",
        "active_service_changed": False,
        "endpoint_count": len(response["endpoints"]),
        "research_only": True,
        "low_confidence": True,
        "warnings": [
            "not_a_clinical_or_release_prediction",
            "all_endpoint_outputs_require_research_only_interpretation",
        ],
    }
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", required=True, help="one FASTA record or a bare canonical peptide sequence"
    )
    parser.add_argument("--input-format", choices=("auto", "plain", "fasta"), default="auto")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--out", type=Path, help="optional JSON output path; stdout is always written"
    )
    args = parser.parse_args()

    result = research_response(
        args.sequence,
        input_format=args.input_format,
        registry=args.registry,
        bundle=args.bundle,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
