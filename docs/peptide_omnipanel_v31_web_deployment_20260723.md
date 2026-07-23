# Peptide OmniPanel V31 本地科研网站设计与部署说明

> **2026-07-23 历史任务与证据分层升级：** 当前网站已调整为“序列输入、预测面板、历史任务、网站与结果说明”四页面。主预测页不再反复显示低置信度、仅供科研和未独立验证；底层 JSON 不删减证据字段。历史按浏览器隔离并可恢复完整结果。详见 `docs/peptide_omnipanel_v31_web_history_and_evidence_20260723.md`。


> **2026-07-23 中文批量升级：** 网站已支持四页面、提交后自动跳转、实时任务面板、最多 32 条序列、FASTA/CSV/TSV 文件上传及批量 JSON/TSV/ZIP 下载。详见 `docs/peptide_omnipanel_v31_web_batch_chinese_upgrade_20260723.md`。

> 生成日期：2026-07-23
> 当前状态：**本机运行中，公网链接已实测 HTTP 200**
> 产品边界：`internal_research_only`，当前 **18/18 动态预测，0/18 独立外部验证**

## 1. 直接访问

- **公网临时链接**：<https://3ea53ebc88ba6b8cb5.gradio.live>
- **本机链接**：<http://127.0.0.1:7860>
- WSL/Windows 本机通常可直接用 `http://localhost:7860`。

公网地址由 Gradio share tunnel 提供，属于临时链接：当 tmux 会话、本机、WSL 或网络停止时会立即失效；服务商提示最长约 1 周。它不是永久域名。

## 2. 网站的科研交互设计

页面采用深海蓝、青色与高对比白底的科研仪器风格，而不是普通表单。信息层次为：

1. **顶部身份区**：V31、18 endpoints、ESM2-650M、local GPU 与中性的结果说明入口。
2. **运行时条**：显示编码器、RTX 5080、18 个预测头与实际加载时间。
3. **序列输入区**：仅接受 20 种 canonical amino acids，自动去空白和大写，限长 1,024 aa。
4. **表示假设强制提示**：裸序列默认解释为天然 L-氨基酸、未修饰、线性、自由 N/C 端。
5. **18 端点卡片**：按 ADME/PK、Safety、Cell & membrane、Biological stability 分组。
6. **完整结果表**：显示值、单位、输出类型、训练行数、OOF 指标、证据通道、特征路由和条件合同；不在每行重复警示。
7. **分类/排序图**：只对同为 0–1 的 classification/PU 结果作图，不将异量纲回归值混在一根轴上。
8. **历史任务**：按浏览器隔离最近 50 个批次，显示历史序列并可恢复完整预测面板。
9. **网站与结果说明**：集中解释低置信度、仅供科研、未独立验证和 18 端点证据表。
10. **可审计导出**：JSON 保留 model SHA、feature SHA、OOF、`research_only`、`low_confidence`、`validation_status`、warning 和条件；TSV 便于 Excel/R/Python 处理。

## 3. 网站后端架构

```text
Browser
  └─ Gradio 5.50 Blocks UI
       ├─ BrowserState + browser-isolated persistent history
       └─ single GPU queue (concurrency=1, max_size=8)
            └─ PersistentV31Predictor
                 ├─ frozen ESM2-650M, mean-residue pooling, BF16, RTX 5080
                 ├─ 93 traditional peptide features
                 ├─ HELM/SMILES hashed character features
                 ├─ Morgan-256 + 10 RDKit descriptors
                 └─ 18 frozen LightGBM / ExtraTrees endpoint heads
```

与原始 CLI 相比，Web 运行时只在启动时校验 SHA 并加载一次 ESM2 和 18 个头；后续请求复用已加载对象。GPU forward 使用内部锁和 Gradio 单队列，避免多个公网请求同时占用 16 GB 显存。预测事件不暴露为公开 API。

## 4. 当前 18 个端点

> 表内 OOF 是开发集分组 out-of-fold 指标，不是独立外部测试。`LOW` 不会在页面上被伪装成高信心结果。

| Endpoint | Task | Unit | Train n | Feature route | Development OOF | Display evidence |
|---|---:|---|---:|---|---|---|
| `LogD7.4` | regression | dimensionless_log10 | 79 | `multimodal` | mae=0.542, r2=0.741, rmse=0.723 | LOW |
| `solubility` | classification | score_0_1 | 2,168 | `esm_sequence` | average_precision=0.711, balanced_accuracy=0.506, roc_auc=0.604 | LOW |
| `F` | regression | fraction | 12 | `esm_sequence` | mae=0.196, r2=0.024, rmse=0.245 | LOW |
| `T1/2` | classification | probability | 918 | `esm_sequence` | average_precision=0.588, balanced_accuracy=0.695, roc_auc=0.685 | RESEARCH |
| `PPB` | regression | fraction_bound | 103 | `structure` | mae=0.164, r2=0.502, rmse=0.240 | LOW |
| `CL` | regression | mL/min/kg | 19 | `esm_sequence` | mae=21.395, r2=-0.078, rmse=25.147 | LOW |
| `Vd` | regression | L/kg | 15 | `structure` | mae=3.370, r2=-2.610, rmse=3.676 | LOW |
| `BBB` | classification | score_0_1 | 844 | `esm_sequence` | average_precision=0.902, balanced_accuracy=0.816, roc_auc=0.903 | LOW |
| `permeability` | regression | source_log10_permeability | 6,814 | `structure` | mae=0.325, r2=0.676, rmse=0.443 | LOW |
| `overall_peptide_toxicity` | classification | score_0_1 | 11,036 | `esm_sequence` | average_precision=0.954, balanced_accuracy=0.881, roc_auc=0.947 | LOW |
| `cytotoxicity` | classification | score_0_1 | 312 | `multimodal` | average_precision=0.560, balanced_accuracy=0.582, roc_auc=0.789 | LOW |
| `hemolysis` | classification | probability | 2,012 | `multimodal` | average_precision=0.739, balanced_accuracy=0.746, roc_auc=0.851 | RESEARCH |
| `HC50` | regression | log10(mol/L) | 1,926 | `esm_sequence` | mae=0.337, r2=0.509, rmse=0.459 | LOW |
| `cell_penetration` | positive_unlabeled | ranking_score_0_1 | 3,784 | `esm_sequence` | average_precision=0.932, balanced_accuracy=0.897, roc_auc=0.978 | LOW |
| `membrane_retention` | regression | fraction_retained | 12 | `structure` | mae=0.167, r2=-0.050, rmse=0.204 | LOW |
| `human_plasma_stability` | regression | log10(h) | 43 | `traditional` | mae=0.708, r2=0.156, rmse=0.945 | LOW |
| `mouse_plasma_stability` | regression | log10(h) | 17 | `structure` | mae=0.744, r2=0.010, rmse=0.920 | LOW |
| `intestinal_stability` | regression | log10(h) | 393 | `structure` | mae=0.422, r2=0.814, rmse=0.611 | LOW |

## 5. 真实部署与验证证据

- 运行 Python：`/root/.venvs/peptide-omnipanel-v31-web311-20260723/bin/python`
- 关键版本：Python 3.11.15，torch 2.12.1+cu130，transformers 4.57.6，scikit-learn 1.9.0，LightGBM 4.6.0，RDKit 2026.03.4，Gradio 5.50.0。
- 真实 GPU smoke：18/18 dynamic，0/18 validated，特征 SHA 与冻结示例完全一致，18 端点最大绝对数值差为 `0.0`。
- 性能实测：模型常驻后，23 aa 示例约 2.1–3.0 s，后续 6 aa 短肽约 0.36 s；启动加载时间受 Windows/WSL 文件缓存影响。
- V31 相关测试：`19 passed`。
- Ruff：`pass`。
- HTTP：本地 `200`，公网 `200`。
- Gradio config：73 components，11 dependencies，4 个顶级页面，predict API exposed = `false`。
- 真实公网 Chrome：历史列表、序列恢复、主结果降噪与说明页全部通过。
- 视觉验证截图：`docs/assets/peptide_omnipanel_v31_history_page_20260723.png` 和 `docs/assets/peptide_omnipanel_v31_about_page_20260723.png`。

机器可读部署记录：

```text
docs/peptide_omnipanel_v31_web_deployment_20260723.json
```

## 6. 启动、查看与停止

### 前台启动（本地 + 临时公网链接）

```bash
cd /mnt/d/work/中检院
V31_WEB_PYTHON=/root/.venvs/peptide-omnipanel-v31-web311-20260723/bin/python \
  scripts/run_local_peptide_omnipanel_v31_web.sh --share
```

仅需本地访问时删除 `--share`。

### 当前后台会话

```bash
tmux attach -t peptide_v31_web_20260723
tail -f .omx/logs/peptide_v31_web_20260723.log
```

### 停止

```bash
tmux kill-session -t peptide_v31_web_20260723
```

停止后公网链接会立即失效。重启 `--share` 通常会获得一个新 URL，需同步更新部署记录。

## 7. 安全和科研边界

- 公网链接未加账号口令，拿到 URL 的人都能提交序列；已用单队列和最大 8 个等待请求限制 GPU 压力。
- 序列仅在本地模型中计算，不发送到外部推理 API；Gradio 隧道仍负责传输网页请求，敏感或未公开序列不应通过无认证公网链接提交。
- 本站不生成临床建议，不对给药、安全性或实验成败作自动决策。
- 对环肽、D-AA、封端、脂化、二硫键或 PEG 化分子，目前裸序列输出只是假想未修饰线性类似物，不是真实分子预测。

## 8. 下一步可升级项

1. 增加 HELM/修饰肽编辑器，使环化、D-AA、封端和脂化成为显式输入。
2. 增加相似度/适用域、conformal prediction 和 abstention，对远离训练分布的序列直接拒绝过度解读。
3. 对每个端点做 source-held-out 和独立外部测试，将真正通过的头从 `NOT VALIDATED` 晋级。
4. 永久部署时增加 HTTPS 反向代理、身份验证、速率限制、审计日志和稳定域名，不再依赖临时 share URL。
