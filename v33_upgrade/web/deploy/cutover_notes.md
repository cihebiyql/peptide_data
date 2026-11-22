# Peptide OmniPanel V33 — Staging Enablement & Production Cutover Guide (`node2`)

## Current Staged Architecture (Zero Impact on Live V31 `:7861` / V32 `:7862`)
- **Inference Engine (`node23`, `10.0.0.3:8902`)**: Live V33 4-foundation-model super-ensemble (24 endpoints, 0 GPU usage).
- **Parallel Reverse Tunnel (`node1`)**: `/data/qlyu/v33_tunnel.sh` forwards `node2:127.0.0.1:8902 -> node23:10.0.0.3:8902` (independent of live V32 `/data/qlyu/omnigrid_tunnel.sh` on `:8901`).
- **Staged FastAPI Web Server (`node2`)**: `/var/www/peptide-omnipanel/v33_next/` running on `127.0.0.1:7863` under user `mode2`.
- **Live Production Untouched**: `:7861` (V31 Gradio), `:7862` (V32 Omnigrid), `:8901` (V32 tunnel), `/etc/nginx/`, and existing systemd units remain 100% untouched.

---

## Step 1 — Enable Public Staging URL (`https://lmmd.ecust.edu.cn/peptide-next/`) [Requires root/sudo]

1. Copy the systemd service unit and switch from manual `nohup` process to systemd:
```bash
# Stop the mode2 nohup process cleanly
pkill -f "uvicorn web.app:app --host 127.0.0.1 --port 7863" || true

# Install and start systemd unit
sudo cp /var/www/peptide-omnipanel/v33_next/deploy/peptide-omnipanel-v33.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now peptide-omnipanel-v33.service
sudo systemctl status peptide-omnipanel-v33.service --no-pager
```

2. Add the Nginx staging location block from `deploy/peptide-next.location.conf` inside the `server {}` block of `/etc/nginx/sites-available/lmmd.ecust.edu.cn`, then test & reload:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

3. Verify public staging endpoint:
```bash
curl -sf https://lmmd.ecust.edu.cn/peptide-next/api/v1/health | jq .
```

---

## Step 2 — Full Production Cutover (When Ready to Replace `/peptide-omnipanel/`)

1. Update `Environment=V33_ROOT_PATH=/peptide-omnipanel` in `/etc/systemd/system/peptide-omnipanel-v33.service` and restart:
```bash
sudo systemctl daemon-reload && sudo systemctl restart peptide-omnipanel-v33.service
```

2. In `/etc/nginx/sites-available/lmmd.ecust.edu.cn`, change `location /peptide-omnipanel/` `proxy_pass` from `http://127.0.0.1:7861/` (or `:7862/`) to `http://127.0.0.1:7863/` and reload:
```bash
sudo nginx -t && sudo systemctl reload nginx
```

3. **Instant Rollback**: Revert `proxy_pass` back to `http://127.0.0.1:7861/` (or `:7862/`) and run `sudo nginx -t && sudo systemctl reload nginx` — zero downtime!
