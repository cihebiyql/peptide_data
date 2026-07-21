# Peptide-OmniPanel：当前模型架构、训练信息与指标审计

- **更新日期**：2026-07-20（Asia/Shanghai）
- **面向用途**：阶段汇报 / 答辩底稿；用来回答“现在到底有哪些模型、各自怎样训练、指标是多少、能否使用”。
- **审计原则**：严格区分 **(i) 当前 39 端点 research display**、**(ii) 有真实 OOF 指标的严格 permeability 开发实验**、以及 **(iii) 历史 V2.6 冻结 direct champions**。三者不是同一个在线服务，更不能把其中一类的分数移花接木给另一类。

> **结论先行**：当前“序列 → 39 端点”显示层是 V2.8 的本地研究系统。它有 **25 个 C 级本地序列模型 + 39 个显式 E 级回退候选**；其中 25 个端点通常由 C 级模型返回，另 14 个端点只能由 E 级透明先验返回。**25 个当前显示层模型尚未完成正式 identity-OOF、source-held-out 或独立外部性能评估，因此没有可诚实引用的 AUROC/MAE。** 已完成严格 OOF 的是另一条 V2.8 Phase2 结构输入 permeability 线：6 个任务均为 `development_only`，全部未获晋级。

---

## 1. 当前到底是哪一个“模型架构”

### 1.1 两条彼此隔离的模型路线

| 路线 | 作用 | 输入 | 覆盖 | 真实状态 | 是否接入当前 39 端点序列显示 |
|---|---|---|---:|---|---|
| **V2.8 research-only display（当前）** | 给有效标准肽序列提供完整、可追溯、低证据显示 | canonical 20-AA 裸序列 / 单条 FASTA | 39/39 有输出；25 C 主模型、14 个 E-only 端点 | `internal_research_only`；所有输出 `research_only=true`、`low_confidence=true` | **是** |
| **V2.8 Phase2 permeability** | 对少数严格肽 permeability 任务做无泄漏 OOF、source-held-out、UQ 审计 | 需要结构（canonical isomeric SMILES） | 6 个独立 permeability assay task | 全部 `reject_development_only` | **否**；不会暗中被替换成 sequence 头 |
| **V2.6 frozen direct champions（历史基线）** | 11 个冻结 direct predictor 的复现实验与外部压力测试 | sequence / HELM / structure，取决于 task | BBB 1、permeability 4、solubility 4、T1/2 2 | `development_only`；active service 未切换 | **否**；表示不全时仍 fail-closed |

这一区分很关键：**“有 39 个显示字段”不等于“39 个经 OOF 验证的监督模型”；“V2.6/V2.8 Phase2 有 MAE”也不等于这些 MAE 属于当前 25 个 sequence heads。**

### 1.2 V2.8 当前序列显示的运行流程

```text
裸序列 / 单条 FASTA
  │  PeptideInput 校验：仅 canonical 20 AA
  │  默认表示假设：natural-L、linear、free termini
  ▼
93 维 sequence feature
  ├─ 29 维理化特征（长度、组成、荷电、疏水等既有项目特征）
  └─ 64 维 hash character n-grams（1–3-gram）
  ▼
AlwaysReturnRouter（端点注册表 + A→B→C→D→E fallback policy）
  ├─ 25 个 C 级 project-trained local sequence artifacts
  ├─ 每一个 39 端点均注册一个 E 级透明本地 prior，保证 C 失败时可见地回退
  └─ 当前无 A/B 已晋级候选
  ▼
39 endpoint response
  value / probability / unit / semantics / tier / model_id
  applicability_domain / warnings / research_only / low_confidence
```

- 对有 C 级 artifact 的 25 个端点，router 优先尝试 C；模型 SHA、manifest、端点语义、特征 schema 与训练输入 SHA 都验证通过后才反序列化。
- 对 C 级 artifact 不存在或预测失败的情况，router 使用该端点的 E prior，并追加 `higher_tier_candidate_unavailable` 警告；不会返回无解释空值。
- E prior 是**透明、确定性的局部规则**（长度、净电荷、疏水残基比例和预先写定默认值），不是训练模型；Kp 返回五组织向量，降解位点返回等长逐残基向量。
- `nearest_training_similarity` 目前是“到训练特征中心的归一化距离”映射，并非真正最近邻序列相似度，也不是经校准的置信度或覆盖率。

### 1.3 输入和化学表示边界

当前 CLI 真实支持 `plain`、`fasta`、`auto`；**尚未实现真实 HELM/PIR/非天然单体/环化拓扑的无损解析**。因此，对含 D-aa、N-methyl、脂化、linker、端基修饰或环化的肽，裸序列会丢失关键化学信息，不能把当前输出解释为完整环肽结构预测。

---

## 2. 当前 25 个 C 级序列模型：训练如何进行

完整逐模型表（端点、算法、训练行、seed、输入 hash、artifact hash、证据级别、指标状态）：

```text
docs/peptide_omnipanel_v28_current_sequence_model_training_inventory_20260720.tsv
```

该表有 **25 行**，是本报告的机器可读主表；其中 `local_canonical_training_sequences` 是同一 canonical sequence 聚合后的实际拟合样本数，**不是**原始 observation 行数。

### 2.1 路线 A：17 个 TDC 小分子 transfer → 肽序列 student

这条路线覆盖 HIA、P-gp、5 个 CYP inhibition、3 个 CYP substrate、hERG、AMES、DILI、ClinTox、carcinogenicity、skin reaction、LD50（总计 17）。

```text
TDC 小分子 observation（17 task，97,342 rows）
  -> RDKit Morgan fingerprint（radius=2, 256 bits）
  -> ExtraTrees teacher
       分类：ExtraTreesClassifier，64 trees，leaf=2，max_features=0.7
       回归：ExtraTreesRegressor，64 trees，leaf=2，max_features=0.7
  -> 对 ChEMBL peptide master 中 4,000 条有效 canonical sequence/SMILES 产生 pseudo label
  -> 93 维 sequence feature 的 ExtraTreesRegressor student
       128 trees，leaf=3，max_features=0.7
  -> C 级 sequence artifact
```

训练规则与规模：

- 每个 teacher 只使用其自身 TDC endpoint 行；teacher 行数从 **280（carcinogenicity）到 13,445（hERG）**，逐端点确切数量在 TSV 中。
- 每个 student 使用同一冻结的 **4,000** 条 ChEMBL peptide master 作为蒸馏承载集合，而非 4,000 条该端点的肽实测标签。
- 分类 teacher 的 pseudo probability 交给 sequence **回归器**拟合；推理时截断到 `[0,1]`。所以该值是小分子 teacher 的跨域伪标签映射，**不是经肽实验校准的发生概率**。
- LD50 是回归 teacher/student；数据摄取时将 Zhu `pLD50` 取负，统一为 registry 的 `log10(mol/kg)`。
- seed 从 `20260719` 起按 endpoint 排序递增；所有 artifact 带本项目 SHA 注册。
- 这 17 个模型统一标注 `L5_self_model_pseudo_label`、`peptide_validated=false`、`research_only` 和 `low_confidence`。

### 2.2 路线 B：7 个局部肽 endpoint-specific ExtraTrees 头

这些模型直接来自冻结本地肽 staging，但仍是窄条件研究训练，不能自动外推到所有肽/所有 assay。

| 端点 | 原始 staging observations | 实际拟合 canonical sequences | 算法 | 输出/变换 | 证据与关键限制 |
|---|---:|---:|---|---|---|
| overall peptide toxicity | 11,036 | 11,036 | ExtraTreesClassifier | probability | `L5_weak_benchmark_label`；不是可替代临床/动物毒性的真值 |
| hemolysis | 2,025 | 1,883 | ExtraTreesClassifier | probability | `L2_source_defined_binary`；artifact 标记 `peptide_validated=true`，但当前仍无 OOF，因此仍 low confidence |
| cytotoxicity | 312 | 141 | ExtraTreesClassifier | probability | `L2_source_defined_binary`；细胞/assay 条件仍窄且混合 |
| HC50 | 1,926 | 1,926 | ExtraTreesRegressor | `log10(mol/L)` | `L0_direct_numeric_curated_dataset`；artifact 标记 `peptide_validated=true`，仍缺本轮严格性能审计 |
| human plasma stability | 65 | 53 | ExtraTreesRegressor | artifact 推理 `pow10` 回转 | `L0_local_frozen_numeric`；小样本、条件混合 |
| mouse plasma stability | 45 | 30 | ExtraTreesRegressor | artifact 推理 `pow10` 回转 | `L0_local_frozen_numeric`；极小样本 |
| intestinal stability | 393 | 388 | ExtraTreesRegressor | artifact 推理 `pow10` 回转 | `L0_local_frozen_numeric`；仅当前肠稳定性上下文 |

所有局部 ExtraTrees 参数相同：`n_estimators=128`、`min_samples_leaf=2`、`max_features=0.7`、`random_state=20260719`、`n_jobs=-1`。同一 canonical sequence 的可用目标会先取 median，再拟合。

### 2.3 路线 C：cell penetration 的 PU 排序器

| 端点 | 原始 staging observations | 实际拟合 sequences | 算法 | 严格语义 |
|---|---:|---:|---|---|
| cell penetration | 784 | 746 | `OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)` | 只有 positive/unlabeled membership；输出为 sigmoid(decision function) 的**研究级 rank score**，不是有真实负类校准后的 probability |

把 PU 数据强行转换成二分类、报告 AUROC 或概率校准都是不合规的；它应使用正样本检索/富集、候选排序和人工复核来评估。

### 2.4 14 个 top-level E-only 端点

在当前 sequence-only 显示中，下列端点没有 C 级主模型：

```text
LogD7.4, solubility, F, T1/2, PPB, CL, Vd, BBB, permeability, Kp,
membrane_retention, protease_stability, degradation_site_probability, immunogenicity_risk
```

它们由透明 E prior 给出可解释的“研究显示值”，不应被称作端点监督预测器。特别是：CL 的 21 条候选混有 total/intrinsic 语义；Kp 只有 3 条 review 记录；protease 与 intestinal stability 有 375 条重叠候选；degradation 缺逐残基标签；immunogenicity 缺治疗肽标签。

---

## 3. 当前 25 个序列头的“训练指标”现状：没有分数不是遗漏，而是边界

### 3.1 已有训练记录 vs 尚无性能证据

| 证据项 | 25 个当前 C 级 sequence heads | 可以如何表述 |
|---|---|---|
| 训练数据行/序列数 | **有**；逐端点写入 artifact payload 与 TSV | 可以报告“训练规模” |
| 算法、参数、随机 seed | **有**；artifact/训练脚本冻结 | 可以报告“模型配置” |
| 输入数据 SHA、artifact SHA、endpoint semantics | **有** | 可以报告“可追溯和可复现” |
| identity-grouped OOF | **未运行** | 不可报告 MAE/AUROC |
| source-held-out | **未运行** | 不可声称跨来源泛化 |
| 独立外部 peptide/cyclic-peptide 测试 | **未运行** | 不可声称可用精度 |
| 概率校准（Brier/ECE）、conformal coverage | **未运行** | 不可把分数当可靠概率/区间 |
| PU 专门的 retrieval/enrichment 评估 | **未运行** | CPP 仅为低证据 rank score |

因此，这 25 个模型的正确状态是 **“已训练、已冻结、可追溯、但尚未经过正式性能验证”**。即使 HC50/hemolysis artifact 自身标有 `peptide_validated=true`，也不能跳过当前同一训练 bundle 缺少 OOF/held-out 的事实。

### 3.2 为什么不能用训练集拟合优度充当指标

训练脚本直接对所有可用训练样本拟合 artifact，并未为当前 25 头写出一份隔离的 OOF prediction 文件。因此若现在回读训练预测得到很高分，那是训练内拟合，不是泛化指标；尤其是 30–53 条的小样本 stability 头、4,000 条伪标签 student 和 11,036 条弱 benchmark toxicity 头，都极易被错误解读。

---

## 4. 有真实 OOF/held-out 分数的 V2.8 Phase2 permeability 线

这 6 个模型**不是当前 sequence-only 39 端点显示头**，但它们是当前项目最完整的、可以用来展示严格训练与评估方法的真实指标。

### 4.1 训练与切分合同

- 数据：`strict_numeric.tsv`，SHA `408edbc4dfae7218068cda49a9cfb7f41cea6459f7310b91b34b8a7e583a3906`；弱标签禁止作为监督。
- 表示：`smiles_char256_morgan512_descriptors_v21`（结构输入）；不能从裸序列推断修饰/环化结构。
- 切分：`identity_group_id` 3-fold；同一 identity 不可同时在 train/test；同 task 所有表示共享 fold。
- seeds：`20260719`、`20260720`、`20260721`；每个 seed 的 OOF prediction 最后取均值。
- 这轮实际 winner：6 个 task 均为 `ExtraTreesRegressor(n_estimators=256, min_samples_leaf=2, max_features=0.5)`。
- 主要指标：identity-component-equal MAE；同时报告 observation MAE、RMSE、median AE、P95 absolute error、R²、Spearman。
- source-held-out：逐来源保留评估；promotion 要求 source/OOF MAE ratio `<=1.25`，并要求独立外部验证。开发 OOF UQ 仅为 90% 对称残差半径，不是独立 coverage 保证。

### 4.2 精确 OOF、source-held-out 与晋级指标

| 端点（assay 语义） | rows / components | OOF component-equal MAE | RMSE | P95 AE | R² | Spearman | source-held MAE | source/OOF | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PAMPA logPapp | 113 / 113 | 0.580593 | 0.750832 | 1.419677 | 0.566791 | 0.754001 | 0.663999 | 1.143656 | reject：缺独立外部验证 |
| Caco-2 AB logPapp | 110 / 109 | 0.419877 | 0.585751 | 1.272918 | 0.597497 | 0.722941 | 0.700112 | 1.667420 | reject：source 泛化差 + 缺外验 |
| Caco-2 efflux ratio | 94 / 94 | 0.353599 | 0.512574 | 0.977596 | 0.594567 | 0.659566 | 0.710254 | 2.008645 | reject：source 泛化差 + 缺外验 |
| Caco-2 BA logPapp | 69 / 69 | 0.286822 | 0.428399 | 0.828485 | 0.034410 | 0.418814 | 0.370157 | 1.290547 | reject：source 泛化差 + 缺外验 |
| Caco-2 logPapp | 65 / 65 | 0.402811 | 0.509245 | 0.876985 | 0.429743 | 0.696018 | 0.616302 | 1.530006 | reject：source 泛化差 + 缺外验 |
| MDCK（source log permeability） | 63 / 63 | 0.545820 | 0.719562 | 1.371748 | 0.096759 | 0.464068 | N/A（单来源） | N/A | reject：source holdout 不足 + 缺外验 |

所有数值的机器可读副本：

```text
docs/peptide_omnipanel_v28_phase2_development_metrics_20260720.tsv
```

该表额外含 observation MAE、median AE、3 个 seed、fold 数、独立外部标志、failure reason 与开发 OOF 的 90% interval radius：

| 端点 | 开发 OOF 90% 对称区间半径 |
|---|---:|
| PAMPA | 1.225285 |
| Caco-2 AB | 0.880633 |
| efflux ratio | 0.834126 |
| Caco-2 BA | 0.536010 |
| Caco-2 logPapp | 0.825243 |
| MDCK | 1.234437 |

这些 radius 只来自同一开发 OOF residual；它们不能作为“在真实未来数据上有 90% coverage”的承诺。

---

## 5. 历史 V2.6 frozen direct champions：训练足迹与已知性能

V2.6 是当前 39 端点 display 之外的冻结直连基线：11 个 task、10,851 development observations、4 structure + 5 HELM + 2 sequence champions，seed `20260716`，未加载 calibration/sealed labels，`active_service_changed=false`。

逐 task 的 framework、参数、rows/components、表示和 artifact SHA：

```text
docs/peptide_omnipanel_v26_frozen_direct_training_inventory_20260720.tsv
```

它包含：

- BBB：sequence ExtraTrees，570 rows / 529 components；
- permeability：4 个 structure task，137–4,585 rows、37–845 components，分别使用 ExtraTrees、HistGradientBoosting 或 CatBoost；
- solubility：4 个 HELM binary task，222–564 rows、221–563 components，使用 LightGBM、XGBoost 或 CatBoost；
- T1/2：sequence binary ExtraTrees（645/505）和 HELM regression CatBoost（244/241）。

V2.6 的一个独立 Helicogenic PAMPA 单来源压力测试曾达到 22 observations / 20 components、component-equal MAE `0.397727`、RMSE `0.573183`，通过了预注册的**单来源**门；但仍缺第二个 >=20 component 独立来源和独立 calibration，所以 `formal_multisource_external_validation=false`，不能提升为 active service。详细见 `docs/peptide_omnipanel_v26_execution_report.md`。

---

## 6. 汇报时怎样正确展示“训练指标”

### 可直接放 PPT 的一句话

> 当前序列显示层已实现 39/39 端点可追溯输出，其中 25 个是本地训练的低证据 sequence heads、14 个是透明 E-tier 先验；这 25 头的正式 OOF/外部指标尚未产生。严格评估已先在 6 个独立 permeability assay 上完成：开发 OOF MAE 为 0.287–0.581，但 source-held-out 证明 5 个任务发生明显跨来源退化，且 6/6 均因外部验证不足未晋级。

### 绝不能写成

- “39 个端点均有验证模型” —— 错；当前为 25 C 主模型 + 14 E-only 端点，且 C 模型均 low confidence。
- “TDC 的 97,342 条就是肽训练标签” —— 错；这是小分子 transfer teacher 标签。
- “CPP score 是细胞穿透概率” —— 错；是 PU rank score。
- “开发 OOF MAE 就证明可以部署” —— 错；source-held-out 与外部验证是独立门。
- “V2.6 的 structure/HELM 模型可以在普通 FASTA 上自动运行” —— 错；缺表示时必须 fail-closed。

---

## 7. 接下来应补哪些指标，才可升级某一个端点

| 模型类型 | 最低需要补的评估 | 主要指标 | 晋级前还需 |
|---|---|---|---|
| 数值回归（HC50、稳定性、LD50 等） | identity OOF + source-held-out + 独立 peptide/cyclic peptide 外测 | component-equal MAE、RMSE、median AE、P95 AE、R²、Spearman | 独立 calibration，90% coverage + selective-risk / abstention |
| 二分类（hemolysis、cytotoxicity、toxicity） | identity/source 分层 split，防止同序列/同来源泄漏 | AUROC、AUPRC、balanced accuracy、sensitivity/specificity | Brier、ECE、阈值固定和独立外测 |
| TDC transfer classification student | teacher 内域指标与 peptide-domain 外测分开报告 | 小分子 teacher 可报 AUROC/AUPRC；肽 student 只能在肽真值上报指标 | 至少一个肽/环肽独立数据源；否则保持 L5 |
| PU（cell penetration） | 仅用正样本检索/候选富集评估 | Precision@K、recall@K、enrichment、人工确认率 | 不可凭假阴性报告普通 AUROC/ECE |
| structured Kp / degradation | endpoint-specific 多组织或逐残基监督集 | tissue-wise MAE/相关、per-residue AUROC/AUPRC | 保持 vector/sequence-label 输出；不能压成假标量 |

在这些指标出现之前，当前最科学的产品行为就是：**继续显示全部 39 个端点，但强制携带 `research_only`、`low_confidence`、证据级别、适用域和 warning。**

---

## 8. 证据文件与冻结哈希

| 对象 | 位置 | SHA-256 |
|---|---|---|
| 25-head research bundle | `data/peptide_omnipanel_v28_research_bundle_20260719_run1/research_bundle_manifest.json` | `4feef813aa5c09a6c28150bb28c18eecac1112db1f616dd3d5cab562beef2d27` |
| TDC training observations | `data/peptide_omnipanel_v28_extended_tdc_20260719_run1/observations.tsv` | `2d5535c648f33853af08489c44b7d19641b6523cd6f58a0a9498dc0daeba2e07` |
| local peptide staging observations | `data/peptide_omnipanel_v28_research_data_20260719_run1/observations.tsv` | `94b4837be4f10ee7cb1a51d38272cc353e9c98ccbabb928235adb73ebee42c9a` |
| current 25-head training inventory | `docs/peptide_omnipanel_v28_current_sequence_model_training_inventory_20260720.tsv` | `c4ebdd34eae46006c45e2c6fcfeaf49eb548c1bd1073d30901496b9e34312d36` |
| Phase2 detailed metrics | `docs/peptide_omnipanel_v28_phase2_development_metrics_20260720.tsv` | `db020ce154bc6f9d5de937f289cb745e0f91328dde3f01fd1167dd3b342a139e` |
| V2.6 frozen training inventory | `docs/peptide_omnipanel_v26_frozen_direct_training_inventory_20260720.tsv` | `4abdf88b6f959eb6a8b138cf6fe55f5be44ec6b7cf8690bffe0e5402e38db932` |
