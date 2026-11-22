#!/bin/bash
set -euo pipefail

DEPLOY_DIR="/var/www/peptide-omnipanel/v33_next/deploy"
NGINX_SNIPPET="/etc/nginx/snippets/peptide-omnipanel.location.conf"
NGINX_BAK="/etc/nginx/snippets/peptide-omnipanel.location.conf.v32.bak"
SVC_DST="/etc/systemd/system/peptide-omnipanel-v33.service"

echo "[1/5] Backing up existing Nginx snippet..."
if [ ! -f "$NGINX_BAK" ]; then
  cp -a "$NGINX_SNIPPET" "$NGINX_BAK"
  echo "  Backed up to $NGINX_BAK"
else
  echo "  Backup $NGINX_BAK already exists (kept intact)"
fi

echo "[2/5] Installing systemd service peptide-omnipanel-v33.service..."
cp "$DEPLOY_DIR/peptide-omnipanel-v33.service" "$SVC_DST"
chmod 644 "$SVC_DST"

echo "[3/5] Switching manual nohup :7863 process to systemd..."
pkill -f "uvicorn web.app:app --host 127.0.0.1 --port 7863" || true
sleep 1
systemctl daemon-reload
systemctl enable --now peptide-omnipanel-v33.service
systemctl restart peptide-omnipanel-v33.service

echo "[4/5] Waiting for 127.0.0.1:7863 health check..."
READY=0
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:7863/api/v1/health > /tmp/v33_cutover_health.json 2>/dev/null; then
    READY=1
    echo "  7863 READY at iter=$i"
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: peptide-omnipanel-v33.service did not become healthy! Aborting Nginx cutover."
  systemctl status peptide-omnipanel-v33.service --no-pager || true
  exit 1
fi

echo "[5/5] Switching /peptide-omnipanel/ Nginx snippet to 127.0.0.1:7863 and reloading Nginx..."
cp "$DEPLOY_DIR/peptide-omnipanel.v33.location.conf" "$NGINX_SNIPPET"
chmod 644 "$NGINX_SNIPPET"

if ! nginx -t; then
  echo "ERROR: nginx -t failed! Rolling back to $NGINX_BAK..."
  cp -a "$NGINX_BAK" "$NGINX_SNIPPET"
  exit 1
fi

systemctl reload nginx
echo "CUTOVER_SUCCESS: /peptide-omnipanel/ is now served by V33 (127.0.0.1:7863)!"
systemctl is-active peptide-omnipanel-v33.service
