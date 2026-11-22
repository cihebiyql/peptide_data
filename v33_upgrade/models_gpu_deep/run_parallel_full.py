#!/usr/bin/env python3
"""Parallel 4-worker RTX 5080 launcher for Peptide OmniPanel V33 Deep Multi-Task & Half-Life Attention Suite.

Runs 4 balanced endpoint shards concurrently on the 16 GB RTX 5080 (~2.2 GB VRAM each -> ~8.8 GB total,
lifting GPU utilization from ~15% to ~60% for ~3.5x wall-clock speedup), then runs the unified 24-head
Leave-Case17-Out production checkpoint and merges the authoritative `gpu_deep_summary.json`.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\work\中检院\v33_upgrade\models_gpu_deep")
LOG_DIR = ROOT / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SHARDS = [
    ["half_life_reg", "CL_absolute", "Cmax", "RRCK"],
    ["half_life_cls", "PPB", "F", "MDCK"],
    ["hemolysis_cls", "BBB", "MIC_P_aeruginosa", "CL_per_kg", "Vd_per_kg"],
    ["overall_peptide_toxicity", "PAMPA", "Caco2", "LogD7.4", "solubility_reg"],
    ["hemolysis_HC50", "cytotoxicity_cls", "solubility_cls", "MIC_E_coli", "MIC_S_aureus", "MIC_C_albicans"],
]


def main() -> None:
    t0 = time.time()
    all_eps = [ep for s in SHARDS for ep in s]
    assert len(all_eps) == 24 and len(set(all_eps)) == 24, "Shards must partition all 24 endpoints!"

    procs = []
    fhs = []
    for i, eps in enumerate(SHARDS):
        log_path = LOG_DIR / f"shard_{i}.log"
        fh = log_path.open("w", encoding="utf-8")
        fhs.append(fh)
        cmd = [
            sys.executable,
            str(ROOT / "train_gpu_deep.py"),
            "--no-summary",
            "--endpoints",
            *eps,
        ]
        print(f"[LAUNCH] Shard {i} ({len(eps)} eps: {', '.join(eps)}) -> {log_path.name}", flush=True)
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs.append((i, eps, p, log_path))
        # Stagger launches by 6s so each worker finishes loading npz -> VRAM and frees RAM cleanly
        time.sleep(6.0)

    # Monitor until all 4 shards finish, reporting per-endpoint completions
    done_flags = [False] * len(procs)
    while not all(done_flags):
        time.sleep(15)
        completed_eps = sum(1 for ep in all_eps if (ROOT / ep / "metrics.json").exists())
        statuses = []
        for i, eps, p, lp in procs:
            rc = p.poll()
            if rc is not None:
                done_flags[i] = True
                statuses.append(f"S{i}:DONE(rc={rc})")
            else:
                ep_done = sum(1 for ep in eps if (ROOT / ep / "metrics.json").exists())
                statuses.append(f"S{i}:{ep_done}/{len(eps)}")
        print(f"[PROGRESS {time.time() - t0:6.1f}s] {completed_eps:2d}/24 endpoints saved | " + " | ".join(statuses), flush=True)

    for fh in fhs:
        fh.close()

    failed = [(i, p.returncode, lp) for i, _, p, lp in procs if p.returncode != 0]
    if failed:
        for i, rc, lp in failed:
            print(f"[ERROR] Shard {i} failed with rc={rc}! Tail of {lp}:", flush=True)
            lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-40:]), flush=True)
        sys.exit(1)

    print(f"\n[ALL 4 SHARDS SUCCEEDED] Elapsed={time.time() - t0:.1f}s. Running Unified 24-Head + Summary Merge...", flush=True)
    merge_cmd = [
        sys.executable,
        str(ROOT / "train_gpu_deep.py"),
        "--merge-only",
        "--wall-sec",
        f"{time.time() - t0:.2f}",
    ]
    subprocess.run(merge_cmd, check=True, cwd=str(ROOT))

    # Update total_elapsed_sec in gpu_deep_summary.json to include unified 24-head training too
    import json
    sp = ROOT / "gpu_deep_summary.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    doc["total_elapsed_sec"] = round(time.time() - t0, 2)
    doc["gpu_compliance"]["parallel_workers"] = 5
    sp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[FULL SUITE COMPLETE] Total wall-clock: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
