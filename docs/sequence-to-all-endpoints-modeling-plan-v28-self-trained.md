# Peptide-OmniPanel V2.8 自主训练全端点计划

日期：2026-07-18
状态：规划完成，尚未执行
代号：ST-Omni（Self-Trained Omnipanel）

## 1. 目标与边界

### 1.1 目标

构建一个由本项目自主训练、可追溯、无需第三方在线服务或第三方模型权重的肽/环肽性质预测系统：

1. 输入至少一条标准氨基酸序列；可选 HELM、PIR、isomeric SMILES、环化和修饰信息；
2. 核心 ADMET/PBPK、扩展 ADMET/毒性、肽特异性质和可计算物化属性均有显示；
3. 有效输入的已注册端点始终返回预测、机制估计或先验，不返回无解释的空值；
4. 训练代码、模型 artifact、切分、数据 SHA、许可和指标全部归档在本项目；
5. 外部项目只作论文/基线参考，不作为生产推理依赖。

### 1.2 “自主训练”的定义

允许：

- 使用有明确来源和许可的公开原始实验标签；
- 使用 UniProt、ChEMBL、TDC、公开补充材料等原始数据；
- 使用 RDKit、scikit-learn、LightGBM、XGBoost、CatBoost、PyTorch 等通用库；
- 自己实现公开 PBPK/QSPR 方程；
- 自己在公开无标签序列上做自监督预训练。

禁止：

- 生产运行时调用第三方预测 API；
- 将第三方预测结果当成实测标签；
- 未审计许可就打包第三方预训练权重；
- 用外部模型预测值冒充本项目独立模型性能；
- 混合不一致的 assay、species、matrix、unit 或 parameter basis。

关键原则：**可以使用外部数据，但模型、训练、验证和推理必须由本项目控制。**

## 2. 当前事实基础

### 2.1 已冻结强模型

V2.6 已有 11 个 development-only direct champions：4 个 structure、5 个 HELM、2 个 sequence；manifest 明确记录 `task_count=11` 和表示计数（`docs/peptide_omnipanel_v26_frozen_direct_manifest.json:88-107`）。这些模型继续作为现有强端点基线，不重新打开已经失败的 residual、hierarchical sharing 和 ensemble 搜索。

### 2.2 当前严格/研究数据

`data/peptide_ml_cleaning_v1/manifest.json:30-52` 记录 BBB、T1/2、solubility 二分类和删失数据；`:58-113` 记录各端点分区；`:164-186` 记录严格数值量和任务数：

| 端点 | 可用本地证据 |
|---|---:|
| permeability | 9,829 strict numeric + 25 censored |
| T1/2 | 1,392 strict numeric + 918 binary + 195 censored |
| solubility | 89 strict numeric + 2,168 binary |
| BBB | 844 weak binary + 136 positive-unlabeled |
| PPB | 103 strict numeric + 7 censored |
| LogD | 79 strict numeric |
| CL | 49 strict numeric，语义高度分散 |
| Vd | 38 strict numeric，只有少量 identity |
| F | 12 strict numeric + 3 censored |
| Kp | 当前无 training-ready 监督任务 |
| cell penetration | 784 positive-unlabeled |

### 2.3 已存在的自主训练雏形

`scripts/build_peptide_admet_weak_labels.py:65-80` 已定义 9 个端点并在本地训练 RandomForest；`:309-380` 定义模型、scaffold split 和指标；`:462-629` 完成训练、预测、OOD 和 model card；但当前缺少：

- estimator artifact 持久化；
- 可复用 inference API；
- 多 seed/component-aware nested CV；
- hepatocyte CL 与 microsome CL 拆分；
- 肽域校准；
- Kp/PBPK 模块。

现有 CL 配置在 `scripts/build_peptide_admet_weak_labels.py:70-75` 混合两个不同实验体系，且脚本自己在 `:570-575` 和 `:698-702` 标记这一限制；V2.8 必须删除该混合 target。

## 3. 重新规划后的产品范围

不再把“十端点”当最终上限。产品分四个面板、三次发布。

### 3.1 Panel A：核心 ADMET/PBPK，10 个端点族

1. LogD7.4
2. solubility：`p_soluble` + `logS`
3. F：`p_high_bioavailability`；连续 F% 仅在真实连续模型存在时输出
4. T1/2：数值 + 阈值概率
5. PPB/fu
6. CL：total、hepatocyte CLint、microsome CLint 分字段
7. Vd/VDss
8. BBB
9. permeability：PAMPA/Caco-2/RRCK/MDCK 分 assay
10. Kp：brain/liver/kidney/muscle/adipose 组织向量

### 3.2 Panel B：扩展 ADME/代谢/安全性，至少 21 个学习端点

- HIA；
- P-gp inhibition；
- CYP1A2/2C19/2C9/2D6/3A4 inhibition；
- CYP2C9/2D6/3A4 substrate；
- hERG、AMES、DILI、ClinTox、carcinogenicity、LD50、skin reaction；
- overall peptide toxicity、cytotoxicity、hemolysis、HC50。

第一批使用公开原始标签和本项目训练代码构建；不下载或打包 ADMET-AI、pepADMET 等第三方权重。

### 3.3 Panel C：肽特异性质，至少 8 个端点

- cell penetration；
- membrane retention；
- human plasma stability；
- mouse plasma stability；
- intestinal stability；
- protease stability；
- degradation-site probability；
- immunogenicity risk。

positive-unlabeled 任务必须使用 PU-learning 或正例排序，不得伪造负类。

### 3.4 Panel D：自主计算/机制输出，至少 15 个字段

- MW、LogP、pI、pH 7.4 net charge；
- TPSA、HBD、HBA、rotatable bonds、ring count；
- hydrophobic fraction、aromatic fraction、aggregation proxy；
- AUC、Cmax、Tmax、MRT；
- Kp tissue vector 和浓度-时间曲线。

前三类物化字段是确定性计算，不应包装成 ML 预测。PBPK 输出必须注明参数假设和模拟场景。

### 3.5 发布目标

| Release | 范围 | 最低非空输出 |
|---|---|---:|
| R1 自主核心版 | Panel A + 基础物化 | 25 个顶层/派生字段 |
| R2 扩展 ADMET 版 | R1 + Panel B | >=46 个字段 |
| R3 肽/PBPK 完整版 | R2 + Panel C/D 全量 | >=60 个字段 |

字段数不是准确性声明；所有输出必须带 evidence tier。

## 4. 目标架构

```text
PeptideInput
  sequence / HELM / PIR / isomeric SMILES / topology / modifications
                         |
                         v
Representation Builder
  sequence descriptors + peptide descriptors + molecular graph/descriptors
                         |
                         v
Endpoint Registry + Condition Encoder
  assay / species / matrix / route / pH / parameter basis / unit
                         |
                         v
Independent Endpoint Experts
  A direct peptide model
  B locally trained low-resource peptide model
  C self-trained public-data cross-domain model
  D self-implemented PBPK/QSPR/nearest-neighbor
  E stratified prior
                         |
                         v
Always-Return Router + UQ/OOD + Provenance
```

### 4.1 不采用单一大模型

历史 V2.2-V2.5 已否定 residual、hierarchical sharing、nested diversity blend 和 source weighting 的晋级。V2.8 默认使用独立 endpoint experts。共享编码器只能作为预注册 challenger，逐端点证明无负迁移后才能晋级。

### 4.2 表示层

R1 只使用可完全自主复现的表示：

- sequence：氨基酸组成、二肽/字符哈希、长度、净电荷、疏水性、芳香性、复杂度；
- HELM：token/monomer/modification/topology 特征；
- structure：Morgan fingerprint、RDKit descriptors、拓扑/环化/立体特征；
- condition：species、matrix、assay、route、pH、dose、timepoint、parameter basis。

R2 可增加本项目在公开肽序列上自监督训练的 encoder，但不得让该工作阻塞 R1。

## 5. 数据路线

### 5.1 金标准层

继续使用：

- `data/peptide_ml_cleaning_v1/strict_numeric.tsv`；
- `binary_evidence_catalog.tsv`；
- `censored_observations.tsv`；
- `task_registry.tsv`。

必须保持 observation、identity、source、publication、scaffold、component 六级去重/切分边界。

### 5.2 公开原始标签层

按以下顺序引入：

1. TDC 原始 ADMET/Tox 数据：用于 HIA、P-gp、CYP、hERG、AMES、DILI 等自主训练；
2. ChEMBL：按 assay ontology、unit、target/organism、relation 清洗；
3. UniProt/DBAASP/Hemolytik 等：仅提取有许可和可追溯标签的肽毒性/功能记录；
4. 公开论文 SI：只纳入有结构/序列、端点语义、单位和来源的记录。

每个来源生成独立 manifest、license field、source row ID 和原始文件 SHA。

### 5.3 多保真标签

标签等级固定：

```text
L0 measured exact
L1 measured censored
L2 source-defined binary
L3 positive-unlabeled
L4 mechanism-computed
L5 self-model pseudo label
```

L4/L5 永不进入 L0 训练量统计；正式评价只使用 L0/L1/L2 的真实来源标签。

## 6. 模型路线

### 6.1 大数据端点

适用：permeability、T1/2、solubility binary。

候选：ExtraTrees、LightGBM、CatBoost、XGBoost；图/序列网络只作为 challenger。使用固定 component folds、3 seeds、nested selection。

### 6.2 中数据端点

适用：LogD、PPB、numeric solubility、BBB。

候选：强正则树模型、ElasticNet、Gaussian Process、KNN/相似邻居。主指标必须同时报告 observation 和 component-equal 结果。

### 6.3 小数据端点

适用：CL、Vd、F。

并行比较：

- public-data cross-domain base；
- peptide-only direct；
- base + 仅在 nested OOF 内学习的简单 affine/isotonic local calibration；
- nearest-neighbor/conditioned prior。

禁止默认使用复杂 residual network。任何 calibration 必须在 outer-train 内完成，不能复用 outer-test 或 sealed labels。

### 6.4 Kp/PBPK

R1 自主实现组织组成驱动的 Kp/PBPK 模块，并用预测 LogD、pKa、fu/PPB、分子量和组织参数做 Monte Carlo。无监督 Kp 模型时仍返回 D/E 级结果，但不得标为 peptide-validated。

### 6.5 PU 与删失数据

- T1/2/PPB/permeability/F 删失记录使用 censored likelihood、survival model 或区间损失；
- cell penetration/BBB positive-unlabeled 使用 PU ranking；
- 与将未知样本当负例的 baseline 做明确对照。

## 7. 分阶段实施

### Phase 0：冻结注册表与治理

新增：

- `configs/peptide_omnipanel_v28_endpoint_registry.json`；
- `configs/peptide_omnipanel_v28_source_registry.json`；
- `docs/peptide_omnipanel_v28_data_license_audit.tsv`；
- schema、unit、species、assay 和 evidence-tier 验证器。

验收：所有端点都有唯一 `endpoint_id/task_kind/unit/parameter_basis/fallback_policy`；所有来源都有 license/provenance/hash。

### Phase 1：修复并物化自主基线

重构 `scripts/build_peptide_admet_weak_labels.py`：

1. 拆分 hepatocyte CL 与 microsome CL；
2. 保存 estimator、feature schema、training row IDs、fold manifest 和 SHA；
3. 增加加载与单条推理接口；
4. 将 weak-label 输出与 model artifact 分离；
5. 添加 deterministic rebuild 测试。

验收：至少 10 个自主训练 base artifacts 可离线加载；同一输入/seed bitwise 或 tolerance-stable；无第三方网络调用。

### Phase 2：本地肽端点模型

训练并评估：

- LogD；
- numeric solubility；
- PPB；
- BBB；
- T1/2/permeability sequence fallback。

现有 V2.6 champions 作为 frozen reference。只有 component-aware OOF 达到门槛的模型才能成为 A/B 级。

### Phase 3：扩展 ADMET/Tox 自主训练

建立统一 public-data ingestion，训练 Panel B 的 HIA、P-gp、8 个 CYP、hERG、AMES、DILI、ClinTox、carcinogenicity、LD50、skin reaction 及肽毒性任务。

验收：每个模型有 raw data manifest、license、dedup report、fold SHA、3-seed OOF、model card 和 OOD reference distribution。

### Phase 4：自主 Kp/PBPK

新增：

- physicochemical parameter calculator；
- tissue composition registry；
- Kp solver；
- one-/multi-compartment PBPK integrator；
- Monte Carlo uncertainty；
- AUC/Cmax/Tmax/MRT 输出。

验收：单位守恒；质量守恒；极端参数不产生未捕获 NaN/负浓度；基准化合物/肽案例可复现公开方程结果。

### Phase 5：Always-Return Router

按端点选择最高可用等级：

```text
A frozen peptide champion
-> B self-trained peptide low-resource
-> C self-trained public-data cross-domain
-> D self-implemented mechanism/neighbor
-> E stratified prior
```

除 invalid input 外不得返回空值。C/D/E 必须显示低证据和 OOD；不允许跨等级盲目平均。

### Phase 6：完整验证与冻结

- unit/type/schema tests；
- component/source/scaffold leakage audit；
- 3-seed OOF；
- external source-held-out；
- calibration/UQ；
- sequence-only、HELM、cyclic、modified、invalid input E2E；
- CPU/GPU inference smoke；
- artifact/release manifest SHA。

active service 保持不变，直到独立多来源外验、校准和发布审计完成。

## 8. 晋级门

### 8.1 A/B 级监督模型

回归：

- component-equal MAE 为主指标；
- 同时报告 RMSE、median AE、P95 AE；
- 3/3 seeds 同方向；
- 对 reference 改善至少 5%，或在无 reference 时明显优于 stratified median；
- paired component bootstrap 2,000 次；
- 0 component/source/scaffold crossing。

分类：

- AUROC、AUPRC、balanced accuracy、Brier、ECE；
- 每类至少报告 component 数；
- 概率输出必须校准或显式标记 uncalibrated；
- PU 任务单独报告 positive recall、ranking 和合成负例敏感性。

### 8.2 C/D/E 级

不要求伪装成高精度，但必须：

- 始终有来源和模型/公式版本；
- 标记 `peptide_validated=false`；
- 报 OOD/相似度；
- 区间 `coverage_guaranteed=false`，直到在独立肽校准集验证；
- UI 与 A/B 级明显区分。

## 9. 可测试验收标准

1. R1：任意有效标准序列至少返回 Panel A 十端点和 15 个基础/派生字段，主值非空率 100%；
2. R2：至少 46 个注册字段非空；R3 至少 60 个；
3. 生产路径不调用外部预测 API，不加载未登记第三方模型权重；
4. 每个学习模型均有本项目生成的 artifact、training manifest、fold SHA 和 model card；
5. F 概率不写成 F%；Papp 不写成 Kp；三类 CL 永久分开；
6. pseudo/mechanistic labels 不进入 measured-label 计数；
7. sequence-only 的结构依赖预测明确记录 `assumed_linear_unmodified_peptide`；
8. invalid sequence 返回结构化错误；其余有效输入不得静默空值；
9. 所有端点通过 unit、range、NaN、determinism 和 provenance 测试；
10. active service 未经独立 release gate 不切换。

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| “自己的模型”但仍是小分子域 | 固定 C 级，报告 OOD，用肽标签逐步校准，不宣传肽验证 |
| 小样本过拟合 | component/source nested CV、简单模型、3 seeds、bootstrap |
| 端点语义混合 | endpoint registry 强制 assay/species/matrix/basis/unit |
| 伪标签污染真实标签 | L0-L5 分层，训练和统计物理隔离 |
| 多任务负迁移 | 独立 experts 默认；共享模型逐端点过 gate 才晋级 |
| Kp/PBPK 看似精确 | Monte Carlo 宽区间、机制等级 D、列出全部参数假设 |
| 公开数据许可不兼容 | source registry 和 license audit 在下载/训练前 fail closed |
| 只输入 FASTA 丢失修饰 | 显式结构假设和降级；鼓励 HELM/SMILES |

## 11. 验证命令形态

执行阶段应形成以下验证面：

```bash
python scripts/validate_peptide_omnipanel_v28_registry.py
pytest -q tests/test_peptide_omnipanel_v28_*.py
ruff check src scripts tests
python -m compileall src scripts
python scripts/audit_peptide_omnipanel_v28_leakage.py
python scripts/summarize_peptide_omnipanel_v28_benchmarks.py
python scripts/verify_peptide_omnipanel_v28_release.py
```

最终 release 必须含：endpoint/source registries、数据/代码/model SHA、OOF、metrics、OOD、UQ、license audit、router E2E 和 active-service unchanged 证明。

## 12. 决策记录

### 采用

自主训练的独立端点专家 + 自主 PBPK/QSPR + always-return router。

### 未采用

1. 第三方模型权重直接作为产品预测器：速度快，但所有权、版本、许可和肽域验证不可控；
2. 从零训练一个覆盖所有端点的大型基础模型：成本高、端点缺失严重、历史实验已有负迁移证据；
3. 对所有缺失端点只返回统计中位数：能非空，但缺乏序列差异性，只保留作 E 级最后兜底。

### 结果

R1 可以较快实现完全本地、无第三方权重的全核心面板；R2/R3 通过公开原始数据和自主训练扩到至少 60 个字段。准确性按端点逐步晋级，不以“字段多”替代验证质量。
