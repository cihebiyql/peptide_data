# OmniGrid V32 架构与部署细节

## 三层架构

```
浏览器
  └─ nginx /peptide-omnipanel/ (node2, 公网域名)
       └─ Gradio 7862  app/node2_omnigrid_app.py
            ├─ ESM2-650M bf16 @ cuda:1（第二块 RTX 4090，~1.3GB 显存）
            └─ HTTP → 127.0.0.1:8901
                 └─ node1 SSH 反向隧道（node2 → node1 → node23，网络单向隔离的破局）
                      └─ node23 omnigrid 服务 :8901
                           ├─ rdkit 2026.09.1 特征引擎（Codex 冻结合同一致）
                           ├─ 170 个 3-seed 优胜模型（55 端点 × 3 seeds，joblib）
                           └─ 半衰期机制锚定层（SMILES 脂肪酸修饰识别）
```

## 特征合同一致性（关键设计）

- 17 组特征（morgan/morgan_r3/atompair/torsion/maccs/rdkit_full/descriptors/
  graph_modifications/rdkit_path/pattern/layered/estate/erg/sequence/dpc/
  sequence_extra/esm2）由冻结提取器计算，特征名 `group:feat` 按合同顺序拼装
- 模型多为 `SimpleImputer(median)` 管道：rdkit_full 对大肽 ≤10% NaN 合法并透传
- estate 特征命名 `count_*` + `estate_sum_*`（overlay 合同）；无 estate 时从引擎推导
- 已通过冻结特征向量的矩阵 SHA256 逐路由字节级校验

## 部署清单构建（service/node23_build_deploy.py）

- 输入：`representative_seed_models.tsv`（171 行 = 55 端点 × 3 seeds 的代表任务）
- 解析每个模型的 result JSON + model_contract（特征名/单位/变换/指标），
  `/tmp` 运行目录优先映射到持久副本
- 协变量（物种/估算口径独热列）检测：>6 列的条件端点标记待部署；
  小型协变量端点（MHC10/HC50/CL_renal_per_kg）用训练众数默认值上线
- 输出 `deploy_manifest.json`（57 入口 / 55 端点 / 170 模型）

## 推理服务（service/node23_omnigrid_service.py）

- `POST /predict {sequence?, smiles?, esm2?, covariates?, endpoints?}`
- 序列可带 ESM2 向量（node2 GPU 计算，与训练池化一致：残基均值、排除全部特殊 token）；
  缺省时 CPU 回退计算
- `GET /health` `GET /endpoints` `POST /debug`（路由级矩阵 SHA 调试）
- 模型惰性加载 + 进程内缓存；热查询 3–5 秒 / 全 55 端点

## 半衰期机制锚定层

纯 ML 训练集缺少周给药脂化类似物，对司美格鲁肽/替尔泊肽低估 10–20 倍
（Codex 结构模型 7.3/13.7h；本项目 SVR 4–8h；两种锚定 Ridge LOO 均失败）。
规则层从 SMILES 检测两羰基间纯碳链（脂肪二羧酸签名）：

| 检测 | 类别 | 锚定区间 | 文献锚（未含两演示药物） |
|---|---|---|---|
| 二羧酸链 ≥ C16 | 周给药类 | 120–168 h | dulaglutide 120 / efpeglenatide 160 / albiglutide 168 |
| 二羧酸 C12–15 | depot 类 | ~25 h | degludec 25 |
| 单酰化 ≥ C14 | 日给药类 | 5–13 h | detemir 5–7 / liraglutide 13 |
| 无脂化 | — | 回落 ML 行 | — |

外部验证 12/12 命中（含 8 个未脂化对照正确回落）。

## 运维要点

- node23 服务 supervisor 循环重启（`run_service.sh` 模式），线程池 4–8（OMP/OPENBLAS 限制）
- node1 隧道 `ssh -N -R` + ServerAlive 探活 + while 守护；重启后需人工拉起
- node2 systemd `peptide-omnigrid`；旧 V31（7861）保留运行，nginx 秒级回退
- 已知坑：`pkill -f` 会匹配自身 ssh 命令行（用 `[n]ame` 括号技巧）；`gr.File(type=)` 只接受
  `filepath`；`10**log10` 溢出需捕获；Gradio 深色模式需 JS 强制 light + CSS 兜底
