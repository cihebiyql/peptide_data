from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tsv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifacts(
    manifest: dict[str, Any], failures: list[str], label: str
) -> None:
    for name, metadata in manifest["outputs"].items():
        path = ROOT / metadata["path"]
        if not path.is_file():
            failures.append(f"{label}: missing output {name}: {path}")
            continue
        digest = sha256_file(path)
        if digest != metadata["sha256"]:
            failures.append(
                f"{label}: SHA-256 mismatch for {name}: {digest} != {metadata['sha256']}"
            )
        if "rows" in metadata and path.suffix == ".tsv":
            rows = tsv_rows(path)
            if rows != metadata["rows"]:
                failures.append(
                    f"{label}: row mismatch for {name}: {rows} != {metadata['rows']}"
                )


def validate_inputs(
    manifest: dict[str, Any], failures: list[str], label: str, required: bool
) -> None:
    for name, metadata in manifest.get("inputs", {}).items():
        path = ROOT / metadata["path"]
        if not path.exists():
            if required:
                failures.append(f"{label}: missing input {name}: {path}")
            continue
        if path.is_file() and sha256_file(path) != metadata["sha256"]:
            failures.append(f"{label}: input SHA-256 mismatch for {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the published peptide ML-clean release and statistics."
    )
    parser.add_argument(
        "--require-upstream-inputs",
        action="store_true",
        help="Fail when non-redistributed upstream source inputs are absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    cleaning = load_json(ROOT / "data/peptide_ml_cleaning_v1/manifest.json")
    statistics = load_json(
        ROOT / "data/peptide_ml_cleaning_v1_statistics_v1/summary.json"
    )
    validate_artifacts(cleaning, failures, "cleaning")
    validate_artifacts(statistics, failures, "statistics")
    validate_inputs(cleaning, failures, "cleaning", args.require_upstream_inputs)
    validate_inputs(statistics, failures, "statistics", args.require_upstream_inputs)
    if not all(cleaning["assertions"].values()):
        failures.append("cleaning: one or more embedded assertions are false")
    if not all(statistics["assertions"].values()):
        failures.append("statistics: one or more embedded assertions are false")

    normalized = ROOT / "data/peptide_ml_cleaning_v1/normalized_observations.tsv"
    with normalized.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 19_979:
        failures.append(f"normalized representative rows: {len(rows)} != 19979")
    if sum(int(row["member_count"]) for row in rows) != 20_052:
        failures.append("normalized member_count conservation failed")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "cleaning_assertions": len(cleaning["assertions"]),
        "statistics_assertions": len(statistics["assertions"]),
        "representative_rows": len(rows),
        "member_rows": sum(int(row["member_count"]) for row in rows),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(bool(failures))


if __name__ == "__main__":
    main()
