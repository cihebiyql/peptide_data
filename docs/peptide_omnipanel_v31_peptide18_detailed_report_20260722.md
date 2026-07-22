# Peptide-OmniPanel V31：仅肽/修饰肽 18 端点模型、训练结果与单序列推理说明

> 状态：`internal_research_only`  
> 数据冻结日期：2026-07-22  
> 模型包：`peptide_omnipanel_v31/release_peptide18_a3_20260722`  
> 重要结论：目前已经实现“一条规范氨基酸序列 → 固定返回 18 个端点的本地动态预测”，但这不等于 18 个端点都已被外部验证。当前 `dynamic_model_count=18`，`validated_model_count=0`。

## 1. 为什么从原来的 39 个端点改成 18 个

此前 39 端点显示层中，有 24 个端点依赖小分子 TDC teacher 将小分子性质迁移到肽序列，另有 5 个端点只是透明先验。该方案不符合“只预测肽、修饰肽和环肽性质”的边界，因此 V30 的相关训练结果、OOF、模型和发布包已经停止并隔离。

V31 重新从本地可追溯的肽域来源出发，只保留 18 个确实具有肽、修饰肽或环肽证据的端点：

1. LogD7.4
2. solubility
3. F
4. T1/2
5. PPB
6. CL
7. Vd
8. BBB
9. permeability
10. overall_peptide_toxicity
11. cytotoxicity
12. hemolysis
13. HC50
14. cell_penetration
15. membrane_retention
16. human_plasma_stability
17. mouse_plasma_stability
18. intestinal_stability

以下内容没有被纳入 V31：

- 任何小分子 teacher 标签；
- 旧 V28/V30 的透明先验；
- Kp 的 3 条无序列、无 HELM、无 SMILES 的综述记录；
- 将同一批 intestinal stability 数据重复命名为 protease stability；
- permeability 家族中的 efflux ratio，因为它不是 Papp/permeability 的同一语义；
- CL/F、CLint、表观 V/F 等与注册主输出不相容的 PK 子任务。

因此，V31 的“18 个端点”是肽域端点，不是把小分子任务换一个名称继续使用。

## 2. 数据总量与证据结构

### 2.1 总体统计

| 项目 | 数量 |
|---|---:|
| 端点数 | 18 |
| 训练案例总数 | 30,507 |
| 唯一规范氨基酸序列 | 17,444 |
| 直接测量/数值阈值派生案例 | 12,363 |
| 弱标签、来源定义二分类或 PU 案例 | 18,144 |
| classification 案例 | 17,290 |
| regression 案例 | 9,433 |
| positive-unlabeled 案例 | 3,784 |
| 小分子 teacher 案例 | **0** |
| 禁止来源路径命中 | **0** |

30,507 是“端点-分子-条件案例数”，不是 30,507 个互不重复的肽。一个肽可能在多个端点或多个实验条件中出现；交叉验证使用端点内的来源/同一性/继承分组，避免同一案例随机拆分到训练和测试。

### 2.2 表示覆盖

| 表示 | 案例行数 | 说明 |
|---|---:|---|
| 可计算规范序列与 ESM2 | 23,410 | 使用规范 20 AA 投影；若原分子有修饰，只能视作 base-sequence 辅助表示 |
| HELM/结构记号 | 12,265 | 保留修饰、末端或环化信息的主要来源之一 |
| 非空化学 SMILES | 7,387 | 不等于全部可被 RDKit 成功解析 |
| RDKit 可解析的 2D 化学结构 | 7,060 | 用于 Morgan 指纹和 2D 描述符 |
| 被剔除的 `SEQ:...` 伪 SMILES | 14,036 | 这是序列代理字符串，不是化学结构，V31 明确移出 SMILES 分支 |

剔除 14,036 条伪 SMILES 是本轮清洗的重要修复：如果把 `SEQ:ACD...` 当作 SMILES，模型可能利用“来源是否提供该字段”而不是化学信息，产生虚高结果。

### 2.3 证据层构成

| evidence lane | 案例数 | 可如何解释 |
|---|---:|---|
| weak_binary | 11,880 | 弱基准复现，只能报告弱分数 |
| direct_measured_structure_conditioned | 7,495 | 有直接数值，但通常依赖修饰、物种、基质或 assay 条件 |
| positive_unlabeled | 3,784 | 784 个正成员 + 3,000 个肽域未标注背景；不是标准正负二分类 |
| source_defined_binary | 2,480 | 来源定义二分类，条件和负类规则不一定统一 |
| direct_source_resolved_binary | 2,012 | 来源解析较好的 hemolysis 阈值任务 |
| direct_measured_single_source | 1,926 | HC50 单一数据来源，不能证明跨来源泛化 |
| direct_measured | 918 | T1/2 >= 1 h 的直接数值阈值任务 |
| direct_measured_structure_only | 12 | membrane retention 单系列结构回归 |

## 3. 18 个端点的精确定义、来源和训练量

| 端点 | 本轮训练量 | 本轮主输出 | 主要来源 | 关键限制 |
|---|---:|---|---|---|
| LogD7.4 | 79 | pH 7.4 的无量纲 log10 LogD | `strict_numeric.tsv` 的 LogD 家族 | 修饰肽结构任务；裸序列结果是假想未修饰线性肽代理 |
| solubility | 2,168 | 来源定义的可溶性分数 | `binary_evidence_catalog.tsv` | 7 个来源/溶剂条件任务被视为粗糙 benchmark；不是统一 logS |
| F | 12 | absolute bioavailability fraction | `strict_numeric.tsv` 的 absolute F | 物种、给药途径混合；样本极少 |
| T1/2 | 918 | `P(t1/2 >= 1 h)` | PEPlife2 冻结标签与来源/同源分组 | 这是阈值概率，不是通用连续半衰期小时数 |
| PPB | 103 | fraction bound | `strict_numeric.tsv` 的 PPB | 人/鼠/未报告、plasma/serum 条件混合 |
| CL | 19 | absolute systemic CL, mL/min/kg | `strict_numeric.tsv` | 只保留 absolute systemic + mL/min/kg；CL/F 和 CLint 排除；唯一序列很少 |
| Vd | 15 | absolute Vd, L/kg | `strict_numeric.tsv` | 只保留 absolute volume_of_distribution + L/kg；样本极少 |
| BBB | 844 | weak BBB score | `binary_evidence_catalog.tsv` | compiled-positive/rule-negative 弱基准；不是临床 BBB 概率，也不是 Kp |
| permeability | 6,814 | dominant PAMPA source-log permeability | `strict_numeric.tsv` 的固定 PAMPA task ID | 单来源/单任务尺度；裸序列需生成假想线性肽 2D 结构 |
| overall peptide toxicity | 11,036 | weak toxicity score | ToxinPred3 弱 benchmark 的本地冻结视图 | 负类构造和弱标签不等于直接毒理测量 |
| cytotoxicity | 312 | source-defined cytotoxicity score | 本地 peptide classification catalog 的冻结视图 | 细胞系、浓度、时间混合；来源解析仍不足 |
| hemolysis | 2,012 | `P(>=20% hemolysis at 50 uM)` | DBAASP 派生、来源解析 strict pilot | 末端修饰信息对裸序列输入有损；当前仍无外部 sealed 验证 |
| HC50 | 1,926 | log10(mol/L) | HemoPI2 2025 frozen normalized view | 单来源；RBC 物种未逐行解析；与 hemolysis 二分类不同 |
| cell penetration | 3,784 | PU ranking score | TPDB 784 正成员 + 3,000 肽域未标注背景 | 未标注不等于负类；高 PU AUC 只能理解为正成员/背景可分性 |
| membrane retention | 12 | PAMPA fraction retained | ACS J Med Chem 单一修饰肽系列 | 仅 12 个结构；裸序列是严重外推 |
| human plasma stability | 43 | log10(h) | `strict_numeric.tsv` | 修复旧选择器，排除 22 条 systemic terminal PK；结构依赖强 |
| mouse plasma stability | 17 | log10(h) | `strict_numeric.tsv` | 修复旧选择器，排除 systemic PK、PPB 和 protease 条件污染 |
| intestinal stability | 393 | log10(h) | PEPlife2/protease-intestinal family | 375 条来自一个主要 cohort；不能再重复算 protease stability |

所有案例均保留 `source_path/source_id/source_url/license/source_record_id`、assay、species、matrix、route、condition、exact identity、来源或继承分组等字段。原始案例表仍留在内部研究目录，不随 GitHub 文档发布，以避免许可证和原始数据再分发问题。

### UniProt 在这里扮演什么角色

UniProt 可以提供序列、蛋白身份、功能和部分注释，但通常不能直接提供可统一建模的 LogD、F、CL、Vd、PPB、PAMPA、HC50 等定量药代/ADMET 标签。V31 不再把“从 UniProt 找到序列”误写成“从 UniProt 得到了定量 ADMET 标签”。当前定量标签主要来自原始论文表格、PEPlife2、DBAASP/HemoPI2/TPDB 及本地严格清洗层；UniProt 类资源更多用于身份和序列追踪。

## 4. 当前模型架构

V31 不是一个共享多任务头强行拟合所有端点，而是“共享冻结表示 + 每端点独立选路和预测头”。这是因为 18 个端点的标签语义、证据等级、条件和数据量差异太大。

### 4.1 序列分支

- 编码器：`facebook/esm2_t33_650M_UR50D`
- 冻结 revision：`08e4846e537177426273712802403f7ba8261b6c`
- 池化：residue mean，1,280 维
- 传统序列特征：93 维
  - 29 个长度、组成、疏水、芳香、极性、带电、估算净电荷等特征；
  - 64 个稳定哈希的序列字符 n-gram 特征。
- 本轮 RTX 5080 ESM 缓存：17,444 条唯一序列，约 92.6 秒完成。

这回答了“为什么不用蛋白质语言模型”的问题：V31 已经实际使用 ESM2-650M，不再只是传统手工特征模型。ESM2 当前被冻结，没有端到端微调，原因是小端点很多、标签条件异质，直接微调 650M 参数极易过拟合并造成端点间污染。

### 4.2 修饰/结构分支

- HELM 字符哈希：64 维；
- SMILES 字符哈希：64 维；
- RDKit Morgan radius=2 指纹：256 维；
- RDKit 2D 描述符：10 维；
- 5 个 modality availability mask，防止把缺失表示无声填零；
- 裸序列推理时，RDKit 确定性生成天然 L、未修饰、线性、游离 N/C 端的 2D peptide graph 和 isomeric SMILES。

这里使用的是可靠、确定性的二维图表示，而不是把 ESMFold 或任意单个 3D 构象当作真实修饰化学结构。当前没有直接使用 GNN/MPNN，是因为大量训练记录只有 HELM 或序列投影，小样本 PK 端点也不足以稳定训练端到端图网络。后续在获得更多结构-标签对后，可将 Morgan/2D descriptor 专家替换或增强为 MPNN。

### 4.3 端点头与选路

每个端点在可用的以下路线间做 3 折分组 OOF 比较：

- `traditional`：93D 传统肽特征；
- `esm_sequence`：93D + ESM2 mean；
- `structure`：HELM/SMILES/Morgan/descriptor/mask；
- `multimodal`：全部分支拼接。

最终路线分布：

| 路线 | 端点数 |
|---|---:|
| ESM2 sequence | 8 |
| structure | 6 |
| multimodal | 3 |
| traditional | 1 |

预测头：

- 6 个 `LGBMClassifier`
- 2 个 `LGBMRegressor`
- 1 个 `ExtraTreesClassifier`
- 9 个 `ExtraTreesRegressor`

所有 18 个最终模型合计约 10.5 MB。大样本端点使用 LightGBM，小样本端点使用更保守的 ExtraTrees。cell penetration 的 PU 选路被强制限制为可比的 sequence 路线，避免“正样本有 HELM、未标注背景没有 HELM”的模态可用性泄漏。发现该问题的早期模型包已经作废并隔离。

## 5. 分组 OOF 训练结果

### 5.1 分类与 PU 端点

| 端点 | 训练量 | 路线 | ROC-AUC | AP | Balanced Accuracy | 正确解释 |
|---|---:|---|---:|---:|---:|---|
| solubility | 2,168 | ESM2 | 0.604 | 0.711 | 0.506 | 仅略高于随机，粗糙可用但很弱 |
| T1/2 >=1 h | 918 | ESM2 | 0.685 | 0.588 | 0.695 | 有一定区分力，仍需外部数据验证 |
| BBB weak | 844 | ESM2 | 0.903 | 0.902 | 0.816 | 弱 benchmark 复现，不是生物学校准概率 |
| overall toxicity weak | 11,036 | ESM2 | 0.947 | 0.954 | 0.881 | 弱 benchmark 复现，不能替代直接毒理测量 |
| cytotoxicity | 312 | multimodal | 0.789 | 0.560 | 0.582 | 中等排序能力，条件异质且来源解析不足 |
| hemolysis | 2,012 | multimodal | 0.851 | 0.739 | 0.746 | 当前最有实用性的直接二分类端点之一 |
| cell penetration PU | 3,784 | ESM2 | 0.978 | 0.932 | 0.897 | 仅表示正成员与未标注背景的可分性；输出必须叫 ranking score |

### 5.2 回归端点

| 端点 | 训练量 | 路线 | MAE | RMSE | R² | 正确解释 |
|---|---:|---|---:|---:|---:|---|
| LogD7.4 | 79 | multimodal | 0.542 | 0.723 | 0.741 | 小型结构系列内表现较好，未证明跨来源泛化 |
| F | 12 | ESM2 | 0.196 | 0.245 | 0.024 | 几乎没有可靠解释度，仅最低可用代理 |
| PPB | 103 | structure | 0.164 | 0.240 | 0.502 | 有中等系列内解释度，物种/基质仍混合 |
| CL | 19 | ESM2 | 21.395 | 25.147 | -0.078 | 未优于均值基线，不应据此做决策 |
| Vd | 15 | structure | 3.370 | 3.676 | -2.610 | 明显不可靠，只保留 research-only 动态代理 |
| PAMPA permeability | 6,814 | structure | 0.325 | 0.443 | 0.676 | 单一冻结 PAMPA task 内较好，尺度不能外推到 Caco-2 等 assay |
| HC50 | 1,926 | ESM2 | 0.337 | 0.459 | 0.509 | 单来源内中等表现，未证明跨来源能力 |
| membrane retention | 12 | structure | 0.167 | 0.204 | -0.050 | 仅单一 12 分子系列代理 |
| human plasma stability | 43 | traditional | 0.708 | 0.945 | 0.156 | 较弱；修饰状态和 assay 条件丢失明显 |
| mouse plasma stability | 17 | structure | 0.744 | 0.920 | 0.010 | 样本极少，接近无解释力 |
| intestinal stability | 393 | structure | 0.422 | 0.611 | 0.814 | 系列内较好，但 375 条来自一个主要 cohort |

这些数字是开发数据上的分组 OOF，不是独立外部测试，也不是 sealed/calibration 集结果。尤其是单来源或 analogue-series 数据，即使 R²/AUC 较高，也可能主要反映系列内插值。

## 6. “输入一条序列得到 18 个端点”如何工作

### 6.1 输入合同

当前严格接受：

- 单条 plain sequence；
- 仅规范 20 种氨基酸字母；
- 自动转大写并去空白；
- 长度不超过 1,024 aa；
- 空序列、非规范残基、多个 FASTA 记录会拒绝，而不是猜测。

裸序列统一解释为：

```text
natural_L_unmodified_linear_free_N_and_C_termini
```

这意味着：如果真实分子是环肽、D-AA、N/C 封端、脂化、PEG 化、二硫键特定拓扑或其他修饰，裸序列输出不是该真实分子的预测，而是假想未修饰线性对应物的结果。修饰敏感端点会附加相应 warning。

### 6.2 本地命令

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/predict_peptide_omnipanel_v31_peptide18.py \
  GIGKFLHSAKKFGKAFVGEIMNS \
  --bundle-dir models/peptide_omnipanel_v31/release_peptide18_a3_20260722 \
  --device cuda \
  --output prediction.json
```

每次请求只计算一次 ESM2/传统/假想二维结构特征，然后将相同特征送入 18 个端点头。输出固定含 18 条记录，并为每个端点给出：

- value / value_kind / unit；
- condition contract；
- model ID、model SHA、feature route；
- OOF metrics；
- evidence lane；
- validation status；
- research_only / low_confidence / insufficient_validation；
- 修饰敏感性和警告；
- 回归端点的开发 OOF 残差区间摘要。

示例序列 `GIGKFLHSAKKFGKAFVGEIMNS` 已得到 18/18 动态值；完整 JSON 位于：

```text
docs/peptide_omnipanel_v31_peptide18_example_prediction.json
```

## 7. 验证结果

最终验证不是只检查“JSON 有 18 个字段”，而是检查每个结果确实依赖本次输入：

- 模型文件 SHA 逐个核对；
- OOF 文件 SHA 逐个核对；
- 18 个端点指标从 OOF 重新计算并与 manifest 一致；
- 5 条与训练集无 exact overlap 的 probe 序列；
- 每条 probe 都生成 ESM2、传统特征、HELM、RDKit 线性肽结构；
- 每个端点在 5 条 probe 上都有 5 个不同的数值；
- 每个端点的 probe range 均大于零；
- 同一特征重复预测在数值容差内确定；
- 所有分类/PU 输出在 [0,1]，所有输出为有限数；
- `endpoint_count=18`；
- `dynamic_model_count=18`；
- `validated_model_count=0`；
- `small_molecule_teacher_rows=0`。

验证产物：

```text
docs/peptide_omnipanel_v31_peptide18_verification.json
docs/peptide_omnipanel_v31_peptide18_probe_predictions.tsv
```

关键 SHA-256：

| 产物 | SHA-256 |
|---|---|
| bundle manifest | `f270039d5cf399542fbe976fcb95fb3ecd150813751e625b9ce4d327e910f743` |
| training metrics TSV | `18c8cfa63dfce56468145bd5bc3dc3ce85f9359a79b3f010d34fcf811b03fa3b` |
| verification JSON | `eb5d89901526ed52f8d941dac164e31ead819b8974e0630ce32f9857f4ef7f6c` |
| probe prediction TSV | `4be1141883a14bf67a83d2f28e92b30993d68aa6931d9c592fd0f10606833226` |
| example prediction JSON | `5247545f90129fc8cf889bc993f82e62c951989fd2feb7680fe8a17e211f8af8` |

## 8. 现在能用到什么程度

### 已经做到

- 仅使用肽/修饰肽/环肽来源，完全移除小分子 teacher；
- 18 个端点均有本地、输入依赖的动态预测器；
- 真正使用 ESM2-650M，而不是只有手工序列特征；
- 对修饰依赖端点增加 HELM/SMILES/RDKit 二维结构分支；
- 一条序列固定返回 18 条结果；
- 低证据端点明确标注 research-only、low-confidence 或 not-validated；
- 训练数据、特征缓存、模型、OOF 和推理结果由 SHA 绑定。

### 还没有做到

- 18 个端点均未完成真正独立外部验证；
- CL、Vd、F、membrane retention、mouse plasma stability 样本仍太少；
- BBB、overall toxicity 和 cell penetration 的高分不能按真实概率解释；
- 当前裸序列接口不能表达真实环化、D-AA、端基和侧链修饰；
- 条件异质端点尚未全部实现 species/assay/matrix 条件输入；
- 目前没有经过校准的置信区间和适用域最近邻相似度；
- 没有将模型切换成临床或生产服务。

### 下一步优先级

1. 为 CL、Vd、F、PPB、LogD7.4 和 plasma stability 增加结构明确、条件一致的新肽数据；
2. 增加 HELM/SMILES 输入接口，真实表达环肽和修饰肽；
3. 做 source-held-out、analogue-series-held-out 和独立论文外部验证；
4. 对达到最低数据量的端点比较 frozen ESM、LoRA/adapter、1D-CNN/BiGRU token mixer 和 MPNN；
5. 用 conformal/UQ 和适用域判定决定何时拒绝或降级，而不是对所有序列都给出同等可信度。

## 9. 配套文件

```text
docs/peptide_omnipanel_v31_peptide18_detailed_report_20260722.md
docs/peptide_omnipanel_v31_peptide18_endpoint_inventory.tsv
docs/peptide_omnipanel_v31_peptide18_training_metrics.tsv
docs/peptide_omnipanel_v31_peptide18_endpoint_registry.json
docs/peptide_omnipanel_v31_peptide18_bundle_manifest.json
docs/peptide_omnipanel_v31_peptide18_example_prediction.json
docs/peptide_omnipanel_v31_peptide18_verification.json
docs/peptide_omnipanel_v31_peptide18_probe_predictions.tsv
```

本报告只发布可审计的说明、统计、指标和示例，不在 GitHub 公开原始训练案例或模型权重。
