#!/usr/bin/env python3
"""Parallel 4-worker RTX 5080 launcher for Plan B-2 Upgrade 1 (46-Endpoint MultiTaskPeptideNet
on all 22 V33.1 Expansion Endpoints). Uses ~6.5 GB peak VRAM across 4 workers on 16 GB RTX 5080.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\work\中检院\v33_upgrade\v2_upgrade\gpu_sota\multitask_v331_22ep")
LOG_DIR = ROOT / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 3 balanced shards partitioning all 22 V33.1 endpoints (keeps peak VRAM ~8.5 GB / 16.3 GB on RTX 5080!)
SHARDS = [
    ["cytotoxicity_CC50", "MIC_A_baumannii", "MBC_S_aureus", "efflux_ratio", "MBC_A_baumannii", "MIC_E_faecium", "MFC_C_albicans"],
    ["hemolysis_MHC", "MIC_B_subtilis", "MIC_Salmonella", "Vd_absolute", "MBC_E_faecalis", "MBIC_biofilm", "MIC_L_monocytogenes"],
    ["MIC_K_pneumoniae", "MIC_S_epidermidis", "MBC_E_coli", "MIC_E_faecalis", "MBC_P_aeruginosa", "LC_antimicrobial", "MIC_C_neoformans", "MBEC_biofilm"],
]


def main() -> None:
    t0 = time.time()
    all_eps = [ep for s in SHARDS for ep in s]
    assert len(all_eps) == 22 and len(set(all_eps)) == 22, "Shards must partition all 22 V33.1 endpoints!"

    procs = []
    fhs = []
    for i, eps in enumerate(SHARDS):
        lp = LOG_DIR / f"shard_{i}.log"
        fh = lp.open("w", encoding="utf-8")
        fhs.append(fh)
        cmd = [sys.executable, str(ROOT / "train_mt46_v331.py"), "--no-summary", "--endpoints", *eps]
        print(f"[LAUNCH] Shard {i} ({len(eps)} eps: {', '.join(eps)}) -> {lp.name}", flush=True)
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        procs.append((i, eps, p, lp))
        time.sleep(6.0)

    done_flags = [False] * len(procs)
    while not all(done_flags):
        time.sleep(15)
        completed = sum(1 for ep in all_eps if (ROOT / ep / "metrics.json").exists())
        statuses = []
        for i, eps, p, lp in procs:
            rc = p.poll()
            if rc is not None:
                done_flags[i] = True
                statuses.append(f"S{i}:DONE(rc={rc})")
            else:
                ep_d = sum(1 for ep in eps if (ROOT / ep / "metrics.json").exists())
                statuses.append(f"S{i}:{ep_d}/{len(eps)}")
        print(f"[PROGRESS {time.time() - t0:6.1f}s] {completed:2d}/22 eps saved | " + " | ".join(statuses), flush=True)

    for fh in fhs:
        fh.close()

    failed = [(i, p.returncode, lp) for i, _, p, lp in procs if p.returncode != 0]
    if failed:
        for i, rc, lp in failed:
            print(f"[ERROR] Shard {i} failed rc={rc}! Tail of {lp}:", flush=True)
            print("\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]), flush=True)
        sys.exit(1)

    print(f"\n[ALL 4 SHARDS SUCCEEDED] Elapsed={time.time() - t0:.1f}s. Merging + Unified 46-Head...", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "train_mt46_v331.py"), "--merge-only", "--wall-sec", f"{time.time() - t0:.2f}"],
        check=True,
        cwd=str(ROOT),
    )
    sp = ROOT / "mt46_summary.json"
    doc = json.loads(sp.read_text(encoding="utf-8"))
    doc["total_elapsed_sec"] = round(time.time() - t0, 2)
    sp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[UPGRADE 1 COMPLETE] Total wall-clock: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
