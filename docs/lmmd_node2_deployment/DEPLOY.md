# Peptide OmniPanel — Application Server Deployment Template (Sanitized)

> Note: Internal hostnames, private IPs, usernames, SSH key paths, and jump-host recipes have been replaced with placeholders for public repository safety.

- **Public URL**: `https://lmmd.ecust.edu.cn/peptide-omnipanel/`
- **Target Host**: `<APP_HOST>` (Ubuntu 22.04+, Python 3.10+, CPU-only runtime for v1.2 / v33.2 with `0 MiB` GPU VRAM)
- **Web Port**: `127.0.0.1:7862` (FastAPI + SPA via `omnigrid_app_node2.py`, proxied by Nginx `/peptide-omnipanel/`)
- **Local Inference Engine**: `127.0.0.1:8902` (`v33_inference_service.py`) + `127.0.0.1:18902` (`esmc_worker.py`)

## Deployment Outline

1. **Sync Release Bundle**:
   ```bash
   scp peptide-omnipanel.tar.gz <DEPLOY_USER>@<APP_HOST>:/tmp/
   ssh <DEPLOY_USER>@<APP_HOST> 'mkdir -p <APP_ROOT> && tar -xzf /tmp/peptide-omnipanel.tar.gz -C <APP_ROOT>'
   ```

2. **Install Python Environment & Dependencies**:
   ```bash
   cd <APP_ROOT>/app
   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip && pip install -r requirements.txt
   ```

3. **Service Supervision**:
   - Web service managed by systemd (`Restart=on-failure`) with `V33_STRICT_REMOTE=1`, `V33_PUBLIC_HEALTH_REDACT=1`, `V33_WARMUP_EXAMPLES=1`, and a 30s background watchdog for `:8902`.
   - Local inference engine (`:8902`) and ESM-C worker (`:18902`) managed via user-level systemd units (`~/.config/systemd/user/v33-*.service`).

4. **Reverse Proxy Verification**:
   ```bash
   curl -sf http://127.0.0.1:7862/api/v1/health
   curl -sf https://lmmd.ecust.edu.cn/peptide-omnipanel/api/v1/health
   ```

Scope: `internal_research_only`.
