# OmniGrid V32 — 多肽性质全景预测平台（55 端点）

V32 在 V31（18 端点、裸序列）基础上升级为**全端点、修饰感知**的多肽性质预测平台：

| 项 | V31 | V32 (OmniGrid) |
|---|---|---|
| 端点数 | 18 | **55**（药代 / 渗透 / 活性 / 毒性溶血 / 理化） |
| 输入 | 仅 20 种标准氨基酸裸序列 | **序列 和/或 完整 SMILES（含脂化/环化/D-AA 修饰）** |
| 模型 | 1 seed × 每端点 1 模型 | **3 seeds × 每端点独立 Optuna(128) × 8 模型族**，输出均值 ± 种子波动 |
| 特征 | 4 条路线 | **17 组冻结化学表示**（约 1.3 万维），特征名/顺序与训练合同 SHA256 字节级一致 |
| 半衰期 | 纯 ML（长效药低估 10–20×） | 增设**机制锚定层**（脂肪酸修饰识别 → 同类已上市药物 FDA 标签区间） |
| 部署 | node2 单机 | node2（ESM2-650M @ 第二块 RTX 4090）⇄ node23 推理服务（170 模型） |

## 目录

```
omnigrid/
├── app/node2_omnigrid_app.py        # 网站应用（Gradio，node2，端口 7862）
├── service/
│   ├── node23_omnigrid_service.py   # 推理服务（HTTP :8901，node23，170 个 3-seed 模型）
│   ├── node23_build_deploy.py       # 部署清单构建：模型合同解析 + 协变量默认值
│   ├── node23_anchor_halflife_v3.py # 半衰期机制锚定层（12/12 外部验证）
│   └── node23_add_permeability.py   # 渗透性端点补录（PAMPA/Caco-2/RRCK）
├── deploy/
│   └── node2_setup_omnigrid.sh      # node2 systemd + nginx 一键部署
└── docs/architecture.md             # 三层架构与部署细节
```

## 快速部署

1. **node23 推理服务**：`python3 node23_build_deploy.py`（生成 deploy_manifest.json）→ `python3 node23_omnigrid_service.py`（建议 supervisor 守护）。
   依赖：`rdkit==2026.09.1`（与特征合同一致）、模型工件目录（见 docs）。
2. **node1 隧道**（node2 无法直连 node23 时）：`ssh -N -R 127.0.0.1:8901:<node23>:8901 <node2>`，while 循环守护。
3. **node2 网站**：`bash node2_setup_omnigrid.sh`（systemd `peptide-omnigrid` + nginx location）。
   依赖：项目 venv 内 `gradio / torch / transformers`，ESM2-650M 快照目录。

## 验证记录

- 特征引擎 4 条路由与冻结特征合同 **SHA256 字节级一致**
- 端到端复现冻结对比报告：PPB 司美格鲁肽 **0.9934** / 替尔泊肽 **0.9933**
- 机制锚定层 12/12：detemir→5–13h ✓ liraglutide→5–13h ✓ degludec→25h ✓ semaglutide→120–168h(实际168) ✓ tirzepatide→120–168h(实际120) ✓（两演示药物标签未参与锚定）
- 稳定性：8 条多样序列 + 2 药物 + 非法输入 + FASTA 批量全部通过；单次全端点 3–5 秒

## 边界

- 6 个条件端点（AC50 / 绝对清除率 / 肾清除率 / Cmax / Vd）依赖实验条件协变量，待部署
- `internal_research_only`：全部模型仅供内部科研，不可用于临床或监管决策；开发 OOF 指标不等同独立外部验证
- 本仓库只发布**代码与部署方法**，不含模型权重与训练数据（数据使用见仓库根 `DATA_USE_NOTICE.md`）
