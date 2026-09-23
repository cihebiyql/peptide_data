# Peptide OmniPanel V31 — LMMD node2 部署包

按照《LMMD_网站部署与运维指南》标准化流程与《Node2网站部署说明（LMMD）》规范制作。
目标服务器：**应用服务器 172.21.43.16（node2，主机名 node2-R6240A0，2×RTX 4090 24GB）**
最终访问地址：`https://lmmd.ecust.edu.cn/peptide-omnipanel/`

## 服务器事实（2026-09-23 实测）

- Ubuntu 22.04.5，Python 3.10.12，563GB 可用磁盘，251GB 内存
- 2×RTX 4090（GPU0 已被其他应用占用约 17.7GB / GPU1 约 8.8GB，利用率 0%；ESM2-650M bf16 约 1.5GB，放得下）
- 应用端口：**127.0.0.1:7861**（已确认空闲；8004/8009/8025/8039 已被其他应用占用）
- Nginx：单配置 `/etc/nginx/sites-available/lmmd.ecust.edu.cn`（sites-enabled 仅此一个链接）
- SSH：校园网直连会被拒（kex 阶段），需从集群内网（node1 等）跳转登录 `mode2@172.21.43.16`
- 进程托管：**必须 systemd**（服务器规范禁止 nohup/screen/tmux）

## 包内容

```
peptide-omnipanel/
├── app/                     # Python 应用（src/peptide_omnipanel 包 + pyproject + requirements）
├── scripts/run_peptide_omnipanel_v31_web.py   # 支持 --root-path 子路径部署
├── models/release_peptide18_a3_20260722/      # V31 18端点模型 bundle（14MB）
├── esm2_650m/               # facebook/esm2_t33_650M_UR50D 本地快照（2.5GB，离线免下载）
├── nginx/peptide_omnipanel_location.conf      # Nginx location 配置块（含 WebSocket/SSE 支持）
├── peptide-omnipanel.service                  # systemd 服务单元
├── run_server.sh            # systemd ExecStart 启动脚本（127.0.0.1:7861，root_path=/peptide-omnipanel）
└── DEPLOY.md                # 本文件
```

## 部署步骤（已在 2026-09-23 执行）

### 1) 上传（经 node1 跳转）

```bash
# 本机 → node1 → node2
scp peptide-omnipanel.tar.gz node1:/data/qlyu/
ssh node1 'scp -i ~/.ssh/id_ed25519_node2 /data/qlyu/peptide-omnipanel.tar.gz mode2@172.21.43.16:/tmp/'
```

### 2) 解压到 /var/www（需 sudo）

```bash
ssh node1 'ssh -i ~/.ssh/id_ed25519_node2 mode2@172.21.43.16'
# node2 上：
sudo mkdir -p /var/www/peptide-omnipanel
sudo tar -xzf /tmp/peptide-omnipanel.tar.gz -C /var/www/peptide-omnipanel --strip-components=1
sudo chown -R mode2: /var/www/peptide-omnipanel
```

### 3) Python 环境

```bash
cd /var/www/peptide-omnipanel/app
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# GPU 版 torch（服务器有 NVIDIA 驱动）
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 或按驱动版本选
pip install -r requirements.txt
```

### 4) systemd 服务

```bash
sudo mkdir -p /var/log/peptide-omnipanel
sudo cp /var/www/peptide-omnipanel/peptide-omnipanel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now peptide-omnipanel
systemctl status peptide-omnipanel --no-pager
```

### 5) Nginx location（先备份！）

```bash
cp /etc/nginx/sites-available/lmmd.ecust.edu.cn ~/lmmd.ecust.edu.cn.bak.$(date +%Y%m%d)
# 把 nginx/peptide_omnipanel_location.conf 的两个 location 块并入 server{} 内
sudo vim /etc/nginx/sites-available/lmmd.ecust.edu.cn
sudo nginx -t && sudo nginx -s reload
```

### 6) 局域网自测（必须通过后再通知管理员）

```bash
curl -I http://127.0.0.1/peptide-omnipanel/     # 预期 200 OK
# 校园网（经集群节点验证）：
curl -I http://172.21.43.16/peptide-omnipanel/
```

## 阶段二：通知管理员（网关 10.0.0.253）

请管理员在 Apache 反向代理配置追加：

```apache
ProxyPass        /peptide-omnipanel/ http://172.21.43.16/peptide-omnipanel/
ProxyPassReverse /peptide-omnipanel/ http://172.21.43.16/peptide-omnipanel/
```

重载后验证 `https://lmmd.ecust.edu.cn/peptide-omnipanel/`。

## 运维速查

| 操作 | 命令 |
|---|---|
| 启停/重启 | `sudo systemctl restart peptide-omnipanel` |
| 日志 | `tail -f /var/log/peptide-omnipanel/web.log` 或 `journalctl -u peptide-omnipanel -f` |
| 强制 CPU | 编辑 service 的 Environment 加 `V31_DEVICE=cpu` 后 restart |
| 换端口 | Environment `V31_PORT=xxxx` + nginx `proxy_pass` 同步改 |

产品边界：`internal_research_only`，18/18 动态预测、0/18 独立外部验证（页面内已带声明）。
