"""Production entrypoint for self-contained Peptide OmniPanel V33 on node2.

Runs via existing systemd unit `peptide-omnigrid.service` (User=mode2, 127.0.0.1:7862),
which Nginx `/peptide-omnipanel/` already proxies. Zero `node23` / `node1` runtime dependency.

On startup (including after any future node2 reboot), ensures the local V33 super-ensemble
inference engine (`127.0.0.1:8902`) is running via `/var/www/peptide-omnipanel/v33_next/run_local_engine.sh`
before launching the FastAPI web app on `127.0.0.1:7862`.
"""
from __future__ import annotations

import fcntl
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

V33_DIR = Path("/var/www/peptide-omnipanel/v33_next")
if str(V33_DIR) not in sys.path:
    sys.path.insert(0, str(V33_DIR))

os.environ["V33_REMOTE_URL"] = os.environ.get("V33_REMOTE_URL", "http://127.0.0.1:8902")
os.environ["V33_MANIFEST_PATH"] = os.environ.get(
    "V33_MANIFEST_PATH",
    str(V33_DIR / "v33_runtime" / "bundle" / "v33_manifest.json"),
)
os.environ["V33_ROOT_PATH"] = os.environ.get("V33_ROOT_PATH", "/peptide-omnipanel")
os.environ["OMNIPANEL_JOBS_DIR"] = os.environ.get("OMNIPANEL_JOBS_DIR", str(V33_DIR / "jobs_store"))
os.environ["OMNIPANEL_GNN_CKPT"] = os.environ.get(
    "OMNIPANEL_GNN_CKPT",
    str(V33_DIR / "v33_runtime" / "bundle" / "gnn_8ep_5fold_fp16.pt"),
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _port_open(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_is_v33_engine(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="ignore")
        return "v33_inference_service.py" in cmdline
    except Exception:
        return False


def ensure_local_v33_engine() -> None:
    """Idempotently start local node2 V33 inference engine if not already listening or booting."""
    if _port_open("127.0.0.1", 8902):
        print("[OmniPanel V33] Local V33 inference engine already listening on 127.0.0.1:8902", flush=True)
        return

    launcher = V33_DIR / "run_local_engine.sh"
    if not launcher.exists():
        print(f"[OmniPanel V33] WARNING: {launcher} not found; skipping auto-launch.", flush=True)
        return

    lock_path = V33_DIR / "v33_engine.lock"
    pid_path = V33_DIR / "v33_engine.pid"
    log_path = V33_DIR / "v33_engine.log"

    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if _port_open("127.0.0.1", 8902):
            return
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text(encoding="utf-8").strip())
                if old_pid > 0 and _pid_is_v33_engine(old_pid):
                    print(
                        f"[OmniPanel V33] Local V33 inference engine (pid={old_pid}) is already warming up.",
                        flush=True,
                    )
                    return
            except Exception:
                pass

        print(f"[OmniPanel V33] Auto-launching local V33 engine via {launcher} (log={log_path})...", flush=True)
        log_fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            ["/bin/bash", str(launcher)],
            cwd=str(V33_DIR),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        print(f"[OmniPanel V33] Local V33 engine started in background (pid={proc.pid}).", flush=True)
        # Brief pause so fast checks see the spawned PID; adapter.py auto-reconnects once /health is 200
        time.sleep(0.5)


import uvicorn  # noqa: E402


def main() -> None:
    ensure_local_v33_engine()
    print(
        f"[OmniPanel V33] Launching FastAPI on 127.0.0.1:7862 "
        f"(root_path={os.environ['V33_ROOT_PATH']}, remote={os.environ['V33_REMOTE_URL']})",
        flush=True,
    )
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=7862,
        root_path=os.environ["V33_ROOT_PATH"],
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info",
    )


if __name__ == "__main__":
    main()
