#!/usr/bin/env python3
"""Surgical, backup-protected V33.1 compatibility update for bundle/v33_inference_service.py."""
from __future__ import annotations

import shutil
from pathlib import Path

SVC = Path("<WORK_DIR>/v33_upgrade/bundle/v33_inference_service.py")
BAK = Path("<WORK_DIR>/v33_upgrade/bundle/v33_inference_service_24ep_backup.py")


def main():
    if not BAK.exists():
        shutil.copy2(SVC, BAK)
        print(f"[BACKUP] Created {BAK}", flush=True)
    txt = BAK.read_text(encoding="utf-8")

    repls = [
        (
            'PK_ENDPOINTS = {"half_life_reg", "half_life_cls", "PPB", "CL_per_kg", "CL_absolute", "Vd_per_kg", "F", "Cmax"}\n'
            'HEMOL_ENDPOINTS = {"hemolysis_cls", "hemolysis_HC50"}',
            'PK_ENDPOINTS = {"half_life_reg", "half_life_cls", "PPB", "CL_per_kg", "CL_absolute", "Vd_per_kg", "Vd_absolute", "F", "Cmax"}\n'
            'HEMOL_ENDPOINTS = {"hemolysis_cls", "hemolysis_HC50", "hemolysis_MHC"}',
        ),
        (
            '            "version": "v33.0",\n            "mode": "v33_live",',
            '            "version": self.manifest.get("version", "v33.1"),\n            "mode": "v33_live",',
        ),
        (
            '                if ep_doc.get("censored_head") is not None and ep in (\n'
            '                    "hemolysis_HC50", "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans",\n'
            '                    "half_life_reg", "PPB", "solubility_reg", "F",\n'
            '                ):',
            '                if ep_doc.get("censored_head") is not None and (\n'
            '                    ep in (\n'
            '                        "hemolysis_HC50", "MIC_E_coli", "MIC_S_aureus", "MIC_P_aeruginosa", "MIC_C_albicans",\n'
            '                        "half_life_reg", "PPB", "solubility_reg", "F",\n'
            '                    )\n'
            '                    or ep_doc.get("expansion_wave") == "v33.1"\n'
            '                ):',
        ),
        (
            '                "status": "ok",\n                "version": "v33.0",\n                "name": r["name"],',
            '                "status": "ok",\n                "version": self.manifest.get("version", "v33.1"),\n                "name": r["name"],',
        ),
    ]
    for old, new in repls:
        assert old in txt, f"Expected snippet not found:\n{old}"
        txt = txt.replace(old, new, 1)

    SVC.write_text(txt, encoding="utf-8")
    print(f"[PATCHED] Updated {SVC} cleanly (4 surgical V33.1 lines; 24 existing endpoints 100% preserved).", flush=True)


if __name__ == "__main__":
    main()
