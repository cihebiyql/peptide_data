#!/usr/bin/env python3
"""Read-only binary SSH tar sync of frozen V33.1 data/features/embeddings & existing CPU/Fusion TSVs."""
from __future__ import annotations

import subprocess
import tarfile
import time
from pathlib import Path

CACHE = Path(r"D:\work\中检院\v33_upgrade\v2_upgrade\gpu_sota\_cache")


def pull_tar(remote_cmd: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "node23", remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    n_files = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
        for m in tf:
            tf.extract(m, path=dest_dir)
            if m.isfile():
                n_files += 1
    _, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"SSH failed rc={proc.returncode}: {err.decode('utf-8', errors='replace')}")
    print(f"[PULL OK] {n_files:4d} files -> {dest_dir} ({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    t0 = time.time()
    pull_tar(
        "tar -C <WORK_DIR>/v33_upgrade/expansion_v331 -cf - "
        "data/compounds_v331_merged.tsv data/v331_split_summary.json data/endpoints "
        "features/compound_index.tsv features/feature_manifest.json features/morgan_bits_1024.npz features/physchem_core.npz "
        "embeddings",
        CACHE / "expansion_v331",
    )
    pull_tar(
        "tar -C <WORK_DIR>/v33_upgrade/expansion_v331 --exclude='*.joblib' -cf - models_cpu models_fusion",
        CACHE / "expansion_v331",
    )
    pull_tar(
        "tar -C <WORK_DIR>/v33_upgrade/models_cpu --exclude='*.joblib' -cf - "
        "PAMPA Caco2 MDCK RRCK F PPB LogD7.4 half_life_reg CL_per_kg Vd_per_kg",
        CACHE / "core_models_cpu",
    )
    pull_tar(
        "tar -C <WORK_DIR>/v33_upgrade/models_fusion --exclude='*.joblib' -cf - "
        "PAMPA Caco2 MDCK RRCK F PPB LogD7.4 half_life_reg CL_per_kg Vd_per_kg",
        CACHE / "core_models_fusion",
    )
    print(f"[ALL READ-ONLY CACHE SYNC COMPLETE] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
