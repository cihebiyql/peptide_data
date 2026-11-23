"""Production entrypoint for self-contained Peptide OmniPanel V33 on node2.

Runs via existing systemd unit `peptide-omnigrid.service` (User=<DEPLOY_USER>, 127.0.0.1:7862),
which Nginx `/peptide-omnipanel/` proxies. Zero external cluster runtime dependency.

Features:
  - Strict live-inference mode (`V33_STRICT_REMOTE=1`): never silently falls back to hardcoded heuristics.
  - Public health path redaction (`V33_PUBLIC_HEALTH_REDACT=1`).
  - Periodic startup + 12h refresh warmup of all 25 curated examples into 24h TTL cache (`V33_WARMUP_EXAMPLES=1`).
  - Staggered-burst safe concurrency (`V33_WEB_MAX_CONCURRENT=2`, `V33_WEB_QUEUE_TIMEOUT=10`): worst-case
    latency stays safely below the 60s upstream gateway timeout while returning clean HTTP 429 on bursts.
  - Watchdog delegates restarts cleanly to user-level systemd (`v33-inference-engine.service`, Linger=yes)
    with explicit `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` and a 210s grace window (>132s engine startup)
    so it never races or spawns a duplicate 26 GB engine.
"""
from __future__ import annotations

import os
import pwd
import socket
import subprocess
import sys
import threading
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
os.environ["V33_STRICT_REMOTE"] = os.environ.get("V33_STRICT_REMOTE", "1")
os.environ["V33_PUBLIC_HEALTH_REDACT"] = os.environ.get("V33_PUBLIC_HEALTH_REDACT", "1")
os.environ["V33_WARMUP_EXAMPLES"] = os.environ.get("V33_WARMUP_EXAMPLES", "1")
os.environ["V33_WARMUP_REFRESH_SEC"] = os.environ.get("V33_WARMUP_REFRESH_SEC", "43200")
os.environ["V33_CACHE_TTL_SEC"] = os.environ.get("V33_CACHE_TTL_SEC", "86400")
os.environ["V33_CACHE_MAXSIZE"] = os.environ.get("V33_CACHE_MAXSIZE", "1024")
os.environ["V33_WEB_MAX_CONCURRENT"] = os.environ.get("V33_WEB_MAX_CONCURRENT", "2")
os.environ["V33_WEB_QUEUE_TIMEOUT"] = os.environ.get("V33_WEB_QUEUE_TIMEOUT", "10")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _user_systemd_env() -> dict[str, str]:
    uid = os.getuid()
    runtime_dir = f"/run/user/{uid}"
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", runtime_dir)
    env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
    try:
        env.setdefault("HOME", pwd.getpwuid(uid).pw_dir)
    except Exception:
        pass
    return env


def ensure_local_v33_engine() -> None:
    """Ask user-level systemd (`v33-inference-engine.service`) to start :8902 without spawning duplicate processes."""
    if _port_open("127.0.0.1", 8902):
        return
    user_unit = Path.home() / ".config" / "systemd" / "user" / "v33-inference-engine.service"
    if not user_unit.exists():
        print("[OmniPanel Watchdog] :8902 unreachable and user systemd unit missing; manual inspection required.", flush=True)
        return
    try:
        r = subprocess.run(
            ["systemctl", "--user", "start", "v33-inference-engine.service"],
            env=_user_systemd_env(),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            print("[OmniPanel Watchdog] Requested user systemd start for v33-inference-engine.service.", flush=True)
        else:
            print(f"[OmniPanel Watchdog] systemctl --user start failed (rc={r.returncode}): {r.stderr.strip()}", flush=True)
    except Exception as exc:
        print(f"[OmniPanel Watchdog] systemctl --user error: {exc}", flush=True)


def _start_engine_watchdog() -> None:
    """Monitor :8902 and only intervene after >210s continuous downtime (>132s normal cold boot)."""

    def _loop() -> None:
        down_since: float | None = None
        while True:
            time.sleep(30.0)
            try:
                if _port_open("127.0.0.1", 8902, timeout=1.5):
                    down_since = None
                else:
                    now = time.time()
                    if down_since is None:
                        down_since = now
                        print("[OmniPanel Watchdog] :8902 down; waiting 210s grace period for user-systemd recovery.", flush=True)
                    elif now - down_since >= 210.0:
                        ensure_local_v33_engine()
                        down_since = now
            except Exception as exc:
                print(f"[OmniPanel Watchdog] check error: {exc}", flush=True)

    threading.Thread(target=_loop, name="v33-engine-watchdog", daemon=True).start()


import uvicorn  # noqa: E402


def main() -> None:
    if _port_open("127.0.0.1", 8902):
        print("[OmniPanel V33] Local V33 inference engine already listening on 127.0.0.1:8902", flush=True)
    else:
        ensure_local_v33_engine()
    _start_engine_watchdog()
    print(
        f"[OmniPanel V33] Launching FastAPI on 127.0.0.1:7862 "
        f"(root_path={os.environ['V33_ROOT_PATH']}, strict_remote={os.environ['V33_STRICT_REMOTE']}, "
        f"max_conc={os.environ['V33_WEB_MAX_CONCURRENT']}, queue_timeout={os.environ['V33_WEB_QUEUE_TIMEOUT']}s)",
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
