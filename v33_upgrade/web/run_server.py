"""Local / staging / production launcher for Peptide OmniPanel V33 FastAPI web platform."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

WEB_DIR = Path(__file__).resolve().parent
PARENT_DIR = WEB_DIR.parent
WORKSPACE_ROOT = WEB_DIR.parents[2] if len(WEB_DIR.parents) >= 3 else PARENT_DIR

if (PARENT_DIR / "__init__.py").exists() or PARENT_DIR.name == "v33_upgrade":
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))
    APP_TARGET = "v33_upgrade.web.app:app"
else:
    if str(PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(PARENT_DIR))
    APP_TARGET = "web.app:app"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Peptide OmniPanel V33 FastAPI server")
    parser.add_argument("--host", default=os.environ.get("V33_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("V33_WEB_PORT", "7863")))
    parser.add_argument(
        "--root-path",
        default=os.environ.get("V33_ROOT_PATH", ""),
        help="Optional reverse-proxy prefix e.g. /peptide-next",
    )
    args = parser.parse_args()

    uvicorn.run(
        APP_TARGET,
        host=args.host,
        port=args.port,
        root_path=args.root_path,
        log_level="info",
    )


if __name__ == "__main__":
    main()
