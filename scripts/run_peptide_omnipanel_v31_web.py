#!/usr/bin/env python3
"""Launch the local/public Peptide OmniPanel V31 research website."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peptide_omnipanel.v31_web import build_demo
from peptide_omnipanel.v31_web_runtime import PersistentV31Predictor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "models/peptide_omnipanel_v31/release_peptide18_a3_20260722"
DEFAULT_MODEL_SOURCE = Path(
    "/root/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/"
    "snapshots/08e4846e537177426273712802403f7ba8261b6c"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model-source", type=Path, default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--export-dir", type=Path, default=ROOT / ".omx/web_exports/v31")
    parser.add_argument("--history-dir", type=Path, default=ROOT / ".omx/web_history/v31")
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()

    predictor = PersistentV31Predictor(
        bundle_dir=args.bundle_dir,
        model_source=args.model_source,
        device=args.device,
    )
    print(
        "V31_WEB_RUNTIME="
        + json.dumps(predictor.runtime_info(), ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if args.startup_only:
        return 0
    demo = build_demo(
        predictor=predictor,
        export_dir=args.export_dir.resolve(),
        history_dir=args.history_dir.resolve(),
    )
    local_url = f"http://127.0.0.1:{args.port}"
    print(f"V31_WEB_LOCAL_URL={local_url}", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        show_error=True,
        max_threads=4,
        max_file_size="2mb",
        allowed_paths=[str(args.export_dir.resolve())],
        quiet=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
