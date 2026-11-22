#!/usr/bin/env python3
"""Strict GPU Compliance Gate for node1 (8x RTX 4090, 24,564 MiB each).

HARD RESOURCE RULE (NON-NEGOTIABLE):
  - Use AT MOST 2 GPUs on node1.
  - ONLY select a GPU if 2 consecutive nvidia-smi samples (5-10s apart) confirm BOTH:
      * memory.used < 50% (< 12,282 MiB / 24,564 MiB), AND
      * utilization.gpu < 50%.
  - Prefer GPUs with generous VRAM headroom (< 45% VRAM, e.g., GPU 5).
  - Record every compliance check into /data/qlyu/v33_upgrade/embeddings/gpu_compliance_log.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

LOG_PATH = Path("/data/qlyu/v33_upgrade/embeddings/gpu_compliance_log.jsonl")


def sample_gpus() -> List[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx, used, total, util, temp = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        rows.append({
            "index": idx,
            "mem_used_mib": used,
            "mem_total_mib": total,
            "mem_pct": round(100.0 * used / total, 2),
            "util_pct": util,
            "temp_c": temp,
        })
    return rows


def check_two_samples(
    interval_sec: float = 6.0,
    max_mem_pct: float = 50.0,
    max_util_pct: float = 50.0,
    max_gpus: int = 1,
    exclude_gpus: Optional[List[int]] = None,
    job_name: str = "unspecified",
) -> dict:
    exclude = set(exclude_gpus or [])
    t1 = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s1 = sample_gpus()
    time.sleep(interval_sec)
    t2 = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s2 = sample_gpus()

    by_idx_1 = {r["index"]: r for r in s1}
    by_idx_2 = {r["index"]: r for r in s2}

    eligible = []
    for idx in sorted(by_idx_1.keys()):
        if idx in exclude or idx not in by_idx_2:
            continue
        r1, r2 = by_idx_1[idx], by_idx_2[idx]
        ok1 = (r1["mem_pct"] < max_mem_pct) and (r1["util_pct"] < max_util_pct)
        ok2 = (r2["mem_pct"] < max_mem_pct) and (r2["util_pct"] < max_util_pct)
        if ok1 and ok2:
            # Rank by max memory usage (prefer generous VRAM headroom <45%), then max util
            eligible.append({
                "index": idx,
                "max_mem_mib": max(r1["mem_used_mib"], r2["mem_used_mib"]),
                "max_mem_pct": max(r1["mem_pct"], r2["mem_pct"]),
                "max_util_pct": max(r1["util_pct"], r2["util_pct"]),
                "s1": r1,
                "s2": r2,
            })

    eligible.sort(key=lambda x: (x["max_mem_pct"], x["max_util_pct"]))
    selected = [e["index"] for e in eligible[: min(max_gpus, 2)]]

    record = {
        "job_name": job_name,
        "sample1_time": t1,
        "sample2_time": t2,
        "interval_sec": interval_sec,
        "thresholds": {"max_mem_pct": max_mem_pct, "max_util_pct": max_util_pct},
        "sample1": s1,
        "sample2": s2,
        "eligible_gpus": [e["index"] for e in eligible],
        "selected_gpus": selected,
        "passed": len(selected) > 0,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser(description="Strict 2-sample GPU compliance gate for node1")
    ap.add_argument("--job", type=str, default="manual_check")
    ap.add_argument("--interval", type=float, default=6.0)
    ap.add_argument("--max-gpus", type=int, default=1)
    ap.add_argument("--max-mem-pct", type=float, default=50.0)
    ap.add_argument("--max-util-pct", type=float, default=50.0)
    ap.add_argument("--wait-timeout", type=float, default=0.0, help="If >0, retry until an eligible GPU appears or timeout (sec)")
    ap.add_argument("--retry-sleep", type=float, default=15.0)
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated GPU IDs to exclude")
    args = ap.parse_args()

    exclude = [int(x) for x in args.exclude.split(",") if x.strip()]
    t0 = time.time()
    while True:
        rec = check_two_samples(
            interval_sec=args.interval,
            max_mem_pct=args.max_mem_pct,
            max_util_pct=args.max_util_pct,
            max_gpus=args.max_gpus,
            exclude_gpus=exclude,
            job_name=args.job,
        )
        if rec["passed"]:
            print(json.dumps(rec, ensure_ascii=False))
            sys.exit(0)
        elapsed = time.time() - t0
        if args.wait_timeout <= 0 or elapsed >= args.wait_timeout:
            print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)
            sys.exit(2)
        print(
            f"[GPU_GATE {rec['sample2_time']}] No eligible GPU yet (elapsed={elapsed:.0f}s/{args.wait_timeout:.0f}s); "
            f"retrying in {args.retry_sleep:.0f}s...",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(args.retry_sleep)


if __name__ == "__main__":
    main()
