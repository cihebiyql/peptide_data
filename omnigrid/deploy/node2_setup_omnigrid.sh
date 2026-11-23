#!/bin/bash
# Deploy OmniGrid V32 app on node2: systemd + nginx location
set -e
mkdir -p /var/www/peptide-omnipanel/omnigrid
cp /tmp/node2_omnigrid_app.py /var/www/peptide-omnipanel/omnigrid/app.py

cat > /etc/systemd/system/peptide-omnigrid.service <<'SV'
[Unit]
Description=Peptide OmniGrid V32 web app (Gradio on 127.0.0.1:7862, ESM2 cuda:1)
After=network.target

[Service]
Type=simple
User=<DEPLOY_USER>
Group=<DEPLOY_USER>
WorkingDirectory=/var/www/peptide-omnipanel/omnigrid
Environment=PYTHONUNBUFFERED=1
Environment=OMNIGRID_DEVICE=cuda:1
ExecStart=/var/www/peptide-omnipanel/app/.venv/bin/python app.py
Restart=on-failure
RestartSec=10
KillSignal=SIGINT
StandardOutput=append:/var/log/peptide-omnipanel/omnigrid.log
StandardError=append:/var/log/peptide-omnipanel/omnigrid.log

[Install]
WantedBy=multi-user.target
SV

cat > /etc/nginx/snippets/peptide-omniv32.location.conf <<'NG'
# -------- Peptide OmniGrid V32 (Gradio 127.0.0.1:7862) --------
location = /peptide-omniv32 { return 301 /peptide-omniv32/; }

location ^~ /peptide-omniv32/ {
    proxy_pass http://127.0.0.1:7862/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
    proxy_read_timeout 300;
    send_timeout 300;
    client_max_body_size 10M;
}
NG

grep -q 'peptide-omniv32.location.conf' /etc/nginx/sites-enabled/lmmd.ecust.edu.cn || \
  sed -i '/peptide-omnipanel.location.conf/a\        include /etc/nginx/snippets/peptide-omniv32.location.conf;' /etc/nginx/sites-enabled/lmmd.ecust.edu.cn

nginx -t && systemctl daemon-reload
systemctl enable peptide-omnigrid
systemctl restart peptide-omnigrid
nginx -s reload
sleep 5
systemctl --no-pager status peptide-omnigrid | head -8
echo '--- log tail ---'
tail -5 /var/log/peptide-omnipanel/omnigrid.log
