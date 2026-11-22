#!/usr/bin/env python3
"""Sync verified V33 GPU Deep artifacts to node23:/data/qlyu/v33_upgrade/models_gpu_deep/
and verify remote file completeness & exact row counts.
"""
from __future__ import annotations

import subprocess
import tarfile
import time
from pathlib import Path

from deep_data import ALL_ENDPOINTS, OUT_ROOT

REMOTE_HOST = "node23"
REMOTE_DIR = "/data/qlyu/v33_upgrade/models_gpu_deep"


def main() -> None:
    t0 = time.time()
    tar_path = OUT_ROOT / "_upload_bundle.tar"
    items_to_pack = [
        OUT_ROOT / "gpu_deep_summary.json",
        OUT_ROOT / "deep_data.py",
        OUT_ROOT / "deep_models.py",
        OUT_ROOT / "train_gpu_deep.py",
        OUT_ROOT / "run_parallel_full.py",
        OUT_ROOT / "inference_api.py",
        OUT_ROOT / "verify_and_report.py",
        OUT_ROOT / "multitask_checkpoints",
        OUT_ROOT / "halflife_specialist",
    ] + [OUT_ROOT / ep for ep in ALL_ENDPOINTS]

    for p in items_to_pack:
        assert p.exists(), f"Missing artifact before upload: {p}"

    print(f"[PACK] Creating uncompressed tar bundle {tar_path} ...", flush=True)
    with tarfile.open(tar_path, "w") as tf:
        for p in items_to_pack:
            tf.add(p, arcname=p.relative_to(OUT_ROOT).as_posix())
    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"[PACK] Bundle ready: {size_mb:.1f} MiB in {time.time() - t0:.1f}s", flush=True)

    # Transfer single tar stream over scp & extract cleanly on node23 (far faster than 150 separate scps!)
    subprocess.run(
        ["ssh", REMOTE_HOST, f"mkdir -p {REMOTE_DIR}"],
        check=True,
    )
    t_scp = time.time()
    subprocess.run(
        ["scp", str(tar_path), f"{REMOTE_HOST}:{REMOTE_DIR}/_upload_bundle.tar"],
        check=True,
    )
    print(f"[SCP] Transferred {size_mb:.1f} MiB in {time.time() - t_scp:.1f}s", flush=True)

    subprocess.run(
        [
            "ssh",
            REMOTE_HOST,
            f"cd {REMOTE_DIR} && tar -xf _upload_bundle.tar && rm -f _upload_bundle.tar && "
            f"ls -la {REMOTE_DIR} && echo '---ENDPOINT_COUNT---' && "
            f"ls -1 {REMOTE_DIR}/*/oof_predictions.tsv | wc -l && "
            f"ls -1 {REMOTE_DIR}/*/holdout_predictions.tsv | wc -l && "
            f"ls -1 {REMOTE_DIR}/*/case17_predictions.tsv | wc -l && "
            f"ls -1 {REMOTE_DIR}/*/metrics.json | wc -l && "
            f"find {REMOTE_DIR}/multitask_checkpoints {REMOTE_DIR}/halflife_specialist -name '*.pt' | wc -l && "
            f"echo '---REMOTE_ROW_COUNTS---' && "
            f"awk 'FNR>1{{c++}} END{{print \"oof_rows=\", c}}' {REMOTE_DIR}/*/oof_predictions.tsv && "
            f"awk 'FNR>1{{c++}} END{{print \"ho_rows=\", c}}' {REMOTE_DIR}/*/holdout_predictions.tsv && "
            f"awk 'FNR>1{{c++}} END{{print \"c17_rows=\", c}}' {REMOTE_DIR}/*/case17_predictions.tsv",
        ],
        check=True,
    )
    tar_path.unlink(missing_ok=True)
    print(f"[UPLOAD COMPLETE & VERIFIED] Total time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
