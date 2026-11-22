#!/usr/bin/env python3
"""Mirror final Plan B-2 GPU SOTA artifacts (checkpoints, predictions, scripts, reports)
from local Windows D:\work\中检院\v33_upgrade\v2_upgrade\gpu_sota\ to
node23:/data/qlyu/v33_upgrade/v2_upgrade/gpu_sota/ via single binary tar stream over SSH,
excluding _cache/ and _logs/!
"""
from __future__ import annotations

import subprocess
import tarfile
import time
from pathlib import Path

LOCAL_ROOT = Path(r"D:\work\中检院\v33_upgrade\v2_upgrade\gpu_sota")
REMOTE_HOST = "node23"
REMOTE_DIR = "/data/qlyu/v33_upgrade/v2_upgrade/gpu_sota"


def main() -> None:
    t0 = time.time()
    items = [
        LOCAL_ROOT / "gpu_sota_report.md",
        LOCAL_ROOT / "gpu_sota_summary.json",
        LOCAL_ROOT / "gnn_atom_attribution.py",
        LOCAL_ROOT / "multitask_v331_22ep",
        LOCAL_ROOT / "conformer_3d_chameleonic",
        LOCAL_ROOT / "gnn_atom_attr",
    ]
    for p in items:
        assert p.exists(), f"Missing artifact before mirror: {p}"

    tar_path = LOCAL_ROOT / "_mirror_bundle.tar"
    print(f"[PACK] Creating tar bundle {tar_path} ...", flush=True)

    def _filter(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if any(x in ("__pycache__", "_logs", "_cache") for x in parts):
            return None
        if ti.name.endswith("test_3d_smoke.py"):
            return None
        return ti

    with tarfile.open(tar_path, "w") as tf:
        for p in items:
            tf.add(p, arcname=p.relative_to(LOCAL_ROOT).as_posix(), filter=_filter)

    size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"[PACK] Bundle ready: {size_mb:.1f} MiB ({time.time() - t0:.1f}s)", flush=True)

    subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, f"mkdir -p {REMOTE_DIR}"], check=True)
    t_scp = time.time()
    subprocess.run(["scp", "-o", "BatchMode=yes", str(tar_path), f"{REMOTE_HOST}:{REMOTE_DIR}/_mirror_bundle.tar"], check=True)
    print(f"[SCP] Uploaded {size_mb:.1f} MiB in {time.time() - t_scp:.1f}s", flush=True)

    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            REMOTE_HOST,
            f"cd {REMOTE_DIR} && tar -xf _mirror_bundle.tar && rm -f _mirror_bundle.tar && "
            f"ls -lh {REMOTE_DIR} && "
            f"echo '---MT46_EPS---' && ls -1 {REMOTE_DIR}/multitask_v331_22ep/*/oof_predictions.tsv | wc -l && "
            f"echo '---MT46_CKPTS---' && find {REMOTE_DIR}/multitask_v331_22ep/checkpoints -name '*.pt' | wc -l && "
            f"echo '---3D_NPZ---' && ls -lh {REMOTE_DIR}/conformer_3d_chameleonic/*.npz {REMOTE_DIR}/conformer_3d_chameleonic/*.json && "
            f"echo '---GNN_CKPT---' && ls -lh {REMOTE_DIR}/gnn_atom_attr/checkpoints/*.pt {REMOTE_DIR}/gnn_atom_attr/*.json",
        ],
        check=True,
    )
    tar_path.unlink(missing_ok=True)
    print(f"[MIRROR COMPLETE & VERIFIED] Total: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
