from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_peptide_omnipanel_v28_extended_tdc.py"


def _module():
    specification = importlib.util.spec_from_file_location("v28_tdc_collector", SCRIPT)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_ld50_zhu_is_negated_into_registered_log10_mol_per_kg_space(tmp_path: Path) -> None:
    module = _module()
    raw = tmp_path / "ld50.tsv"
    raw.write_text("ID\tX\tY\ncompound\tCCO\t2.5\n", encoding="utf-8")
    dataset = next(item for item in module.DATASETS if item.endpoint_id == "LD50")

    rows, rejected = module.normalize_rows(dataset, raw)

    assert rejected == 0
    assert rows[0]["raw_value"] == 2.5
    assert rows[0]["value"] == -2.5
    assert rows[0]["unit"] == "log10(mol/kg)"
    assert rows[0]["value_transform"] == "negate_pLD50"


def test_collector_rejects_non_binary_values_for_binary_endpoints(tmp_path: Path) -> None:
    module = _module()
    raw = tmp_path / "hia.tsv"
    raw.write_text("Drug_ID\tDrug\tY\ncompound\tCCO\t0.25\n", encoding="utf-8")
    dataset = next(item for item in module.DATASETS if item.endpoint_id == "HIA")

    rows, rejected = module.normalize_rows(dataset, raw)

    assert rows == []
    assert rejected == 1
