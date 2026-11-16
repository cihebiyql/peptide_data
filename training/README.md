# 训练脚本与配置

本目录包含两套互补的训练体系，均服务于 V32 全端点平台。

## codex_campaign/（76 个脚本 · 主力训练体系）

数据来源：新旧来源登记 1,289,057 行、213,800 条清晰标注、原始 61 个端点名称。
产物：**55 个端点 × 3 个随机种子（42/2026/3407）的已保存优胜模型**，
即网站 `omnigrid/` 推理服务加载的 170 个模型。

关键脚本（按流程顺序）：

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 特征 | `peptide_modeling_features.py` → `_expand_features.py` → `_add_methods.py` → `_add_smiles_methods.py` | 17 组冻结化学表示（基础指纹 → 扩展指纹/描述符 → 路径/Pattern/Layered → EState/ErG） |
| 特征 | `peptide_modeling_attach_esm.py` | ESM2-650M 嵌入（残基均值、排除全部特殊 token、float16 存储） |
| 训练 | `peptide_modeling_train.py`（引擎）/ `peptide_modeling_all_tasks.py` / `_internal_cv.py` | 每任务 Optuna(128) × 8 模型族 × 内部五折 + 留出；`SimpleImputer(median)` 管道 |
| 调度 | `peptide_modeling_campaign.py` / `_supervise.py` / `_pipeline.py` | 多种子 campaign 调度（累计有效训练 21.8 小时，64 线程） |
| 端点 | `peptide_modeling_conditional_cmax/_cl/_volume.py` 等 | 带实验条件协变量的端点（物种/剂量/估算口径独热列） |
| 审计 | `peptide_modeling_execution_audit.py` | v20-v24 全局审计：52 端点 trained、种子/折/合同/SHA 全链核验 |

训练合同要点：`GroupKFold`（相似分子分组防泄漏）；回归以 MAE 选路、分类以 AP 选路；
全量 refit 模型 + OOF/留出预测均落盘并带 SHA256。

## optuna_pipeline/（28 个脚本 · 我的 Optuna 管线，V32 补充体系）

数据来源：新旧整合端点表（58,932 化合物 / 353,079 条观察 / 25 个可训练端点，
含补录的 PAMPA 6,748 行 / Caco-2 1,160 行）。

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 特征缓存 | `node23_step1b.py` / `node23_expand_cache.py` / `node23_cpu_features.py` | 13 组特征预计算为连续 float16 矩阵（加载 31 秒 vs NFS 15 分钟） |
| 端点补录 | `node23_add_permeability.py` | 从原始观察表补提取 PAMPA/Caco2/RRCK（此前整类漏提取） |
| 训练 | `node23_parallel_worker.py` + `node23_launch_all.sh` | 每端点独立进程 × Optuna 500 trials × 7 模型族 × 13 特征组子集搜索 |
| 集成 | `node23_ensemble.py` + `node23_auto_ensemble.sh` | 贪心多模型/多视图集成（成员加入须提升 OOF Spearman） |
| 部署辅助 | `node23_build_deploy.py` / `node23_omnigrid_service.py` | 部署清单构建与推理服务（见 `omnigrid/`） |

已知设计要点：每端点按“有效行数 ≥ 40”预筛可行特征组（viable-group），
杜绝纯 OLD 数据端点搜索空间近空；模型参数（如 SVR）固定
`C/gamma/epsilon`，Optuna 命名与构造参数在 `build()` 中显式映射。

## 模型权重（不在本仓库）

170 个模型（joblib）合计约 34 GB，超出 GitHub 限制，持久保存在内网
`node23:/data/qlyu/peptide_modeling_multiseed20h_v1/`。运行时由
`omnigrid/configs/deploy_manifest.json` 中的路径清单引用；如需外发，
建议打包为 tar 分卷并附 SHA256（不放入 git）。

## 配置参数（omnigrid/configs/）

| 文件 | 内容 |
|---|---|
| `deploy_manifest.json`（11 MB） | 57 入口 × 3 seeds 的完整部署清单：模型路径、路由、特征名合同、单位/变换、CV 与留出指标 |
| `representative_seed_models.tsv` | 171 行代表模型表（端点/种子/路由/参数/指标/工件 SHA） |
| `endpoint_representatives.tsv` | 每端点代表任务与推荐配置（含逐种子调优参数） |
| `summary.json` / `REPORT.md` | 全端点最优组合与评价值汇总 |
