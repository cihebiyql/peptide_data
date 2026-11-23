# Peptide OmniPanel 1.2 / Engine v33.2 — Production Deployment Notes

## Self-Contained Production Architecture (`0 MiB` GPU VRAM, Zero External Cluster Dependency)
- **Public URL**: `https://lmmd.ecust.edu.cn/peptide-omnipanel/` (proxied to `127.0.0.1:7862` via Nginx).
- **Web Application (`127.0.0.1:7862`)**: FastAPI + SPA served by `omnigrid_app_node2.py` under systemd (`Restart=on-failure`) with:
  - `V33_STRICT_REMOTE=1` (disables silent heuristic fallback; returns HTTP 503 if engine is offline)
  - `V33_PUBLIC_HEALTH_REDACT=1` (omits internal filesystem paths & loopback URL from `/api/v1/health`)
  - `V33_WARMUP_EXAMPLES=1` & `V33_CACHE_TTL_SEC=86400` (pre-warms all 25 curated examples into a 24h TTL cache)
  - `V33_WEB_MAX_CONCURRENT=4` (bounded concurrency returning HTTP 429 + `Retry-After` under bursts)
  - Built-in 30s background watchdog supervising `127.0.0.1:8902`.
- **Live Inference Engine (`127.0.0.1:8902`)**: `v33_inference_service.py` (`46` endpoints, 4 foundation models on CPU, `OMP_NUM_THREADS=1`, CatBoost `thread_count=1`, bounded semaphore concurrency, `request_queue_size=64`). Supervised via user-level systemd (`v33-inference-engine.service`) and web watchdog.
- **ESM-C 600M Worker (`127.0.0.1:18902`)**: Pure-PyTorch standalone `esmc_worker.py` on CPU, supervised via user-level systemd (`v33-esmc-worker.service`) or auto-spawned by `:8902`.

## Quick Verification
```bash
curl -sf https://lmmd.ecust.edu.cn/peptide-omnipanel/api/v1/health
```
