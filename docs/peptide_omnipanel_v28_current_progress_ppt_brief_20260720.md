# 肽 / 环肽 ADMET-PBPK：当前进展 PPT 详细说明

- **版本/日期**：V2.8-ST-Omni，2026-07-20
- **用途**：这是面向阶段汇报的事实底稿，可直接拆为 PPT。所有数字均来自当前冻结的 manifest、注册表、训练 bundle 或验证输出。
- **一句话结论**：我们已经从“多来源、语义混杂的肽/环肽性质记录”建立出可追溯的清洗数据层，并搭建了一个**序列输入、统一返回 39 个端点**的本地研究原型；但它仍是 `internal_research_only`，所有 39 项均须显示 `research_only=true` 与 `low_confidence=true`，尚无端点可称为临床、注册或正式部署级预测器。

---

## 0. 建议在答辩开场直接说明的边界

### 可以准确地说

1. 已构建肽/环肽 ADMET、PK/PBPK 数据的**可追溯清洗层**：原始来源、记录 ID、URL/DOI/PMID、license、单位、条件、删失关系、identity 和数据分区均保留在相应数据层。
2. 注册表定义了 **39 个互不混淆的端点**：10 个 core ADMET/PBPK、21 个 extended ADMET/tox、8 个肽特异端点。
3. 当前研究显示层对有效的标准肽序列可返回 **39/39 非空输出**；其中 25 个为 C 级本地拟合序列头，14 个为透明 E 级本地先验。
4. 所有训练器均为**本项目本地训练**；没有在线调用第三方预测 API，也没有把第三方模型权重当作正式推理依赖。

### 不能说、也不应在 PPT 中暗示

1. 不能说“39 个端点均有肽实验训练数据”或“39 个端点均已验证”。新增 29 个端点中，只有 25 个存在当前 C 级本地模型；另 4 个缺乏可诚实使用的独立序列监督标签。
2. 不能把 TDC 小分子标签称为肽/环肽测量。它们只用于跨域 transfer，当前证据等级为 `L5_self_model_pseudo_label`。
3. 不能把 `Papp/permeability` 叫作组织分配 `Kp`，不能把 BBB 叫作 Kp，不能把 HC50 和二分类溶血混为同一标签，不能把不同 CL（total / hepatocyte CLint / microsome CLint）合并。
4. 不能把 `F` 的分类概率展示成“连续口服生物利用度百分比”。

---

## 1. 项目目标、输入与输出范围

### 1.1 最终产品目标

输入一条肽/环肽的序列（当前原型真实支持：**标准 20 种氨基酸的裸序列或单条 FASTA**），输出 ADMET/PBPK 相关的 39 个端点，并同时输出：

- 数值或概率；
- 单位与端点语义；
- 模型/回退层级（A/B/C/D/E）；
- `research_only`、`low_confidence`、`evidence_tier`；
- 适用域和警告；
- Kp 的五组织向量、降解位点的逐残基向量等结构化输出。

当前对非天然氨基酸、复杂修饰、真实 HELM/PIR、环化拓扑仍不能由公共输入解析器无损表示；因此序列模式的默认假设是**自然 L 型、线性、游离端基**，并以证据降级显示。这个限制应在汇报中明确。

### 1.2 39 个端点的三层面板

| 面板 | 数量 | 端点 |
|---|---:|---|
| Core ADMET/PBPK | 10 | LogD7.4、solubility、F、T1/2、PPB、CL、Vd、BBB、permeability、Kp |
| Extended ADMET/tox | 21 | HIA、P-gp、5 个 CYP inhibition、3 个 CYP substrate、hERG、AMES、DILI、ClinTox、carcinogenicity、skin reaction、overall toxicity、cytotoxicity、hemolysis、LD50、HC50 |
| 肽特异性质 | 8 | cell penetration、membrane retention、human/mouse plasma stability、intestinal/protease stability、degradation site、immunogenicity |

端点字段、主输出、单位、任务类型、当前显示层、训练/参考数据和关键限制已逐行整理为：

```text
docs/peptide_omnipanel_v28_current_progress_ppt_endpoint_inventory_20260720.tsv
```

该表有 39 行，适合直接导入 Excel / PPT 作为附录。

---

## 2. 数据资产：应如何回答“现在到底有多少数据？”

### 2.1 不要把不同层级简单相加

当前数据来自不同阶段、不同清洗程度的集合。存在子集、重表达、去重前后版本和伪标签，因此**不能**将下表所有数字加成一个“全球唯一总行数”。推荐在 PPT 中按层级展示。

| 数据层 | 行数/规模 | 含义 | 是否可直接当正式训练真值 |
|---|---:|---|---|
| V15 原始扩展 staging | 215,568 行、50 个原始端点名 | 宽召回的多来源候选池 | 否，需语义/许可/质量筛选 |
| ML-cleaning v1 输入范围 | 20,052 行 | 进入清洗管线的记录范围 | 否，仍分 strict/censored/weak/review |
| ML-cleaning v1 代表记录 | 19,979 行 | 去重折叠后的 representative observation | 视分区而定 |
| strict numeric | 11,591 行、620 个任务 | 关系为 `=`、单位与表示通过当前 QC 的数值记录 | 研究候选；仍要按 task/条件/异常清单筛选 |
| censored | 230 行 | `<`、`>`、区间等删失记录 | 仅删失/区间模型，不可静默变成精确点值 |
| binary evidence | 3,930 行 | BBB、T1/2、solubility 等来源定义或阈值标签 | 不等同于普适实验真值 |
| positive-unlabeled | 920 行 | BBB 136、cell penetration 784 | 不能伪造负类 |
| review | 3,308 行 | 身份、单位、语义或条件冲突待复核记录 | 否 |
| **V2.8 TDC 冻结集** | **97,342 行、17 端点** | 小分子 ADME/Tox 原始标签，供 transfer | 仅跨域研究，不是肽测量 |
| **V2.8 本地肽 staging** | **16,622 行、11 端点有记录** | 从本地可追溯肽数据重表达出的研究训练候选 | 仅 endpoint-specific 过滤后研究使用 |

### 2.2 ML-cleaning v1 的数据质量与覆盖度

- 清洗前 20,052 行，去重后 19,979 条代表记录；73 个重复成员被折叠。
- 共有 **14,374 个 identity groups**、14,869 个 selected representations、1,067 个条件化任务（task）。
- 表示形式：14,435 HELM、2,410 SMILES、2,053 plain sequence、1,081 缺失表示。因此，“有序列”与“可用复杂环肽/修饰结构”不是同一概念。
- strict numeric 的主量来自 permeability（9,829 行，占约 84.8%）；T1/2 1,392 行；PPB 103；solubility 89；LogD 79；CL 49；Vd 38；F 12。
- 现存 444 条高优先级异常需在正式建模前再次处理：343 条亚秒级 protease/intestinal half-life、94 条长度 >100 aa 的标准序列、7 条极端 clearance。

### 2.3 V2.8 当前冻结集合的可汇报数字

- 新冻结的 TDC 小分子原始标签：97,342 行（89,957 分类 + 7,385 回归）、17 端点。
- 新冻结的本地肽 research staging：16,622 行，0 行 quarantine；11 个端点有记录。
- 两者相加为 **113,964 条 V2.8 冻结“行引用”**，但本地 staging 与 ML-cleaning v1 存在来源/记录层级关系，**不是**全项目新增的、全局去重后的唯一实验量。
- 研究 bundle 使用 4,000 条 ChEMBL peptide master 的有效结构作为小分子 teacher 到序列 student 的蒸馏承载集合。

关键 SHA：

| 资产 | SHA-256 |
|---|---|
| TDC V2.8 dataset manifest | `aec70c4266d6234f55264772715e471fd63a36ec78882b4e36cc3b14441d5e18` |
| TDC normalized observations | `2d5535c648f33853af08489c44b7d19641b6523cd6f58a0a9498dc0daeba2e07` |
| 本地肽 staging manifest | `f961ddcc3761829552d90d40b4d4df5dd4571a52603763a668de09d2b8223933` |
| 本地肽 staging observations | `94b4837be4f10ee7cb1a51d38272cc353e9c98ccbabb928235adb73ebee42c9a` |
| 25 头研究 bundle manifest | `4feef813aa5c09a6c28150bb28c18eecac1112db1f616dd3d5cab562beef2d27` |

---

## 3. 数据是如何收集和清洗的？

### 3.1 总流程

```text
公开数据库 / 文献补充材料 / 既有本地整理
  -> 原始文件冻结（URL、下载文件、SHA、来源记录 ID、license）
  -> 端点语义拆分（不能混 endpoint）
  -> 单位与数值空间归一化（保留 relation / censoring）
  -> sequence / HELM / SMILES / identity QC
  -> 去重、条件冲突检查、source/identity 分组
  -> strict / censored / binary / PU / review 分层
  -> 仅对符合端点语义的层训练本地研究模型
  -> router 返回数值 + 证据/风险标签
```

### 3.2 已实际使用的主要来源路线

| 路线 | 采集/处理方式 | 当前用途与范围 |
|---|---|---|
| 本地 ML-cleaning v1 | 将既有多来源肽/环肽记录按来源、单位、实验条件、身份表示、relation/censoring 清洗；保留每条记录的 source URL、DOI/PMID、license 和 lineage | 构成 core 数据底座和肽特异 staging 的主要可追溯输入 |
| TDC / Harvard Dataverse | 按 file ID 下载原始 TSV；逐文件保存 URL、原文件 SHA、字节数、标签语义、正类定义和变换 | 17 个 extended ADMET/tox 的小分子 transfer 基线 |
| ChEMBL peptide master | 21,965 条有结构的 peptide master 记录；当前取其中 4,000 条作为蒸馏承载 | 将本地小分子 teacher 的输出蒸馏为序列 student；输出仍是伪标签 |
| 本地 HemoPI2/溶血与 HC50 整理 | 严格区分 binary hemolysis 与数值 HC50；HC50 统一到 `log10(mol/L)` | hemolysis、HC50 C 级序列头 |
| 本地 CPP positive-unlabeled 整理 | 仅保留 784 条 CPP 正样本/未标记条目，不人为生成假阴性 | cell penetration PU 排序头 |
| 血浆/肠稳定性严格数值整理 | 保留物种、matrix、assay、duration 与删失；human/mouse/intestinal 不混池 | 三个窄语义 stability C 级头 |
| ChEMBL Kp 审查记录 | 仅找到 3 条 mouse brain/plasma ratio 审查记录，且不构成完整组织向量监督集 | Kp 语义审查；当前只支持 E 级五组织机制向量 |

### 3.3 已登记但尚未变为当前训练真值的来源

source registry 中仍以 `pending_license_provenance_and_hash_audit` fail-closed 的候选来源包括：TDC/ChEMBL 原始 assay、UniProt 原始记录、DBAASP、Hemolytik、文献 supplementary information、CycPeptMPDB、OpenFDA/DailyMed。它们可用于下一步扩充，但在许可、原文件 hash、端点语义、去重和 source/component split 未完成前，**不能**宣称已进入当前正式训练集。

---

## 4. 17 个 TDC extended 端点：具体来源、行数与语义

所有下列数据均由 TDC Harvard Dataverse 原始文件冻结；原始 URL、file ID、SHA-256 位于：

```text
data/peptide_omnipanel_v28_extended_tdc_20260719_run1/source_manifest.tsv
```

| 端点 | TDC 数据集 | Dataverse file ID | 行数 | 任务 | 汇报时必须说明 |
|---|---|---:|---:|---|---|
| HIA | hia_hou | 4259591 | 578 | 分类 | source-defined human intestinal absorption；不等同肽 F |
| Pgp inhibition | pgp_broccatelli | 4259597 | 1,219 | 分类 | P-gp inhibitor 标签 |
| CYP1A2 inhibition | cyp1a2_veith | 4259573 | 12,579 | 分类 | 小分子 CYP inhibition |
| CYP2C19 inhibition | cyp2c19_veith | 4259576 | 12,665 | 分类 | 同上 |
| CYP2C9 inhibition | cyp2c9_veith | 4259577 | 12,092 | 分类 | 同上 |
| CYP2D6 inhibition | cyp2d6_veith | 4259580 | 13,130 | 分类 | 同上 |
| CYP3A4 inhibition | cyp3a4_veith | 4259582 | 12,328 | 分类 | 同上 |
| CYP2C9 substrate | carbonmangels | 4259584 | 669 | 分类 | 是否 substrate |
| CYP2D6 substrate | carbonmangels | 4259578 | 667 | 分类 | 是否 substrate |
| CYP3A4 substrate | carbonmangels | 4259581 | 670 | 分类 | 是否 substrate |
| hERG | herg_karim | 6822246 | 13,445 | 分类 | source-defined channel blocker 类别 |
| AMES | ames | 4259564 | 7,278 | 分类 | mutagenicity 类别 |
| DILI | dili | 4259585 | 475 | 分类 | drug-induced liver injury 类别 |
| ClinTox | clintox | 4259572 | 1,478 | 分类 | clinical toxicity-related 标签，不能泛化成全部肽毒性 |
| carcinogenicity | carcinogens_lagunin | 4259570 | 280 | 分类 | source-defined carcinogenicity |
| skin reaction | skin_reaction | 4259609 | 404 | 分类 | skin reaction/sensitization；不是 skin permeability |
| LD50 | ld50_zhu | 4267146 | 7,385 | 回归 | Zhu pLD50 已取负，统一为 registry `log10(mol/kg)` |

**重要转换**：Zhu 的 `pLD50 = log10[1/(mol/kg)]`，而 registry 输出为 `log10(mol/kg)`；因此已使用 `log10(mol/kg) = -pLD50`。不做这个转换会使毒性方向完全相反。

**许可证/域限制**：这些行目前是 `component_license_pending_release_clearance` 与 `small_molecule_transfer_only`。对每个端点的模型流程为：小分子 Morgan fingerprint teacher -> 4,000 条 ChEMBL peptide 的本地伪标签 -> 93 维 sequence student。故它们是本地模型，但仍是 `L5_self_model_pseudo_label`，不是肽实测监督。

---

## 5. 本地肽 research staging：逐端点可用情况

| 端点 | 观察数 | 当前路线 | 证据层/限制 |
|---|---:|---|---|
| overall peptide toxicity | 11,036 | C：ExtraTrees 分类 | `L5_weak_benchmark_label`；来源定义弱 benchmark，不能称实验毒性真值 |
| hemolysis | 2,025 | C：ExtraTrees 分类 | `L2_source_defined_binary`；与 HC50 永久分开 |
| HC50 | 1,926 | C：ExtraTrees 回归 | `L0_direct_numeric_curated_dataset`，输出 `log10(mol/L)` |
| cell penetration | 784 | C：OneClassSVM PU 排序 | `L3_positive_unlabeled`，是 score，不是有真阴性的概率分类 |
| intestinal stability | 393 | C：ExtraTrees 回归 | `L0_local_frozen_numeric`；仅当前肠稳定性上下文 |
| cytotoxicity | 312 | C：ExtraTrees 分类 | `L2_source_defined_binary`；细胞/assay 条件仍混合 |
| human plasma stability | 65 | C：ExtraTrees 回归 | `L0_local_frozen_numeric`；样本小、条件混合 |
| mouse plasma stability | 45 | C：ExtraTrees 回归 | `L0_local_frozen_numeric`；样本很小 |
| membrane retention | 12 | E：透明先验 | 12 条均无可用标准 sequence，不能诚实训练 sequence 模型 |
| CL | 21 | E：透明先验 | 有 absolute/intrinsic 等不同语义，不能合成 total CL |
| Kp | 3 | E：透明五组织向量 | review-only brain/plasma 记录，不能当完整 Kp 监督集 |

三项硬性缺口：

- `protease_stability`：375 条候选与 `intestinal_stability` 重叠，不能复制后作为独立训练集；
- `degradation_site_probability`：没有逐残基标注；
- `immunogenicity_risk`：没有治疗肽整体免疫原性标签。

因此后三项也是 E 级先验，不应在 PPT 中伪装成“已有训练数据”。

---

## 6. 当前模型架构：从输入到 39 端点输出

### 6.1 研究显示层架构

```text
Plain sequence / FASTA
  -> 输入校验（canonical 20 aa；默认 natural-L linear free termini）
  -> 93 维序列特征
       = 理化描述符 + 64 维字符 n-gram hash
  -> endpoint router（按注册表 A -> B -> C -> D -> E 排序）
       -> 25 个 C 级本地序列模型
       -> 14 个 E 级透明本地先验
  -> 39 endpoint JSON / 表格显示
       + unit + semantics + model/tier + evidence + AD + warnings
```

### 6.2 25 个 C 级本地模型

| 模型组 | 数量 | 学习器 | 训练输入 | 输出含义 |
|---|---:|---|---|---|
| TDC 跨域 transfer sequence student | 17 | ExtraTrees teacher + ExtraTrees sequence student | 小分子 TDC + ChEMBL peptide pseudo labels | 小分子来源的低证据概率/回归值 |
| 本地肽二分类 | 3 | ExtraTreesClassifier | toxicity、cytotoxicity、hemolysis staging | source-defined probability |
| 本地肽数值回归 | 4 | ExtraTreesRegressor | HC50、human/mouse plasma、intestinal stability | 条件混合的研究级回归 |
| cell penetration | 1 | OneClassSVM | 仅 CPP positive-unlabeled | 非校准 PU ranking score |

> 合计 17 + 3 + 4 + 1 = **25**。所有模型 artifact 均带 manifest、SHA、特征 schema、seed 和 endpoint semantics。

### 6.3 14 个 E 级透明本地先验

- core 10：LogD7.4、solubility、F、T1/2、PPB、CL、Vd、BBB、permeability、Kp；
- peptide 4：membrane retention、protease stability、degradation-site probability、immunogenicity risk。

之所以 core 10 在**本次 sequence-only research display**中仍是 E，不代表历史上没有任何候选模型；而是我们拒绝把以前的 structure/HELM-only weak artifacts 在没有结构/HELM 输入时暗中假装为 sequence 模型。V2.6 曾冻结 11 个 direct development artifacts（4 structure、5 HELM、2 sequence），但它们也尚未激活正式服务。

### 6.4 结构化端点

- `Kp`：返回 `{brain, liver, kidney, muscle, adipose}` 的组织/血浆比值向量；不等同 permeability 或 BBB。
- `degradation_site_probability`：返回长度等于输入序列长度的逐残基概率向量；当前为透明规则，非监督网络。

---

## 7. 当前覆盖度：适合放在 PPT 的核心图

### 7.1 显示覆盖 vs 训练证据覆盖

| 指标 | 数值 | 正确解释 |
|---|---:|---|
| 注册端点 | 39 | 完整字段/语义合同 |
| 有输出端点 | 39 / 39 | 仅研究显示完整性，不等于已验证覆盖 |
| C 级本地拟合序列头 | 25 | 有本地 artifact，但均低证据研究用途 |
| E 级透明先验 | 14 | 提供可审计数值/向量，不应称 endpoint-specific 监督模型 |
| A/B 已晋级端点（本研究显示层） | 0 | 没有把任何低证据路线伪装为已验证结果 |
| 新增 29 端点中 C 级 | 25 / 29 | extended 17 TDC + 4 local；peptide 4 local |
| 新增 29 端点中仍无诚实 sequence 监督集 | 4 / 29 | membrane retention、protease stability、degradation site、immunogenicity |

**推荐图**：环形图（C=25、E=14）+ 堆叠柱状图（3 个 panel：core 0C/10E，extended 21C/0E，peptide 4C/4E）。

### 7.2 必须伴随数值展示的统一安全字段

每个 endpoint 显示：

```json
{
  "status": "predicted_low_evidence",
  "research_only": true,
  "low_confidence": true,
  "confidence": "low",
  "evidence_tier": "L0/L2/L3/L4/L5",
  "peptide_validated": false
}
```

这是产品合同，不是可选备注。特别是 C/TDC route 仍是 `peptide_validated=false`。

---

## 8. 已做的验证与当前性能结论

### 8.1 工程/可复现性验证

- endpoint/source registry 验证：通过；39 endpoints、15 sources。
- V2.8 目标测试：`24 passed, 13 subtests passed`。
- 实际 smoke prediction：标准序列与 FASTA 均验证通过，39/39 端点非空；tier 分布 C=25、E=14。
- 对结构化输出额外验证：Kp 固定有五个 tissue key；degradation vector 长度等于输入 sequence 长度。
- 所有 artifact 加载前检查 SHA-256；source manifest 保留原始 URL、file ID 和 hash。

### 8.2 性能/晋级结论：没有正式晋级

1. **V2.4 NPDB 融合尝试**：4 个 permeability 开发任务中 0/4 通过单任务门；停止继续扩大 ensemble 搜索。
2. **V2.8 Phase2 permeability promotion**：6 个 task 均 `reject_development_only`。主要原因是 source-held-out 泛化差、缺少独立外部验证；例如 Caco-2 AB 的 source/OOF MAE 比为 1.67，efflux ratio 为 2.01，均超过预设 1.25 门。
3. **V2.6 11 个 direct artifacts**：均标为 development-only，未加载 calibration/sealed labels，`active_service_changed=false`。

因此对外汇报时应说：**“已完成数据和研究原型基础设施，正在从开发集性能转向 source-held-out、独立外部验证、UQ 和数据补齐。”** 不能说“模型已经达到可用临床预测性能”。

---

## 9. 当前主要风险、缺口与下一步

| 优先级 | 缺口/风险 | 为什么重要 | 下一步 |
|---|---|---|---|
| P0 | 外部验证不足 | 开发 OOF 不代表跨来源泛化 | endpoint-specific source-held-out + 独立肽/环肽测试集 |
| P0 | TDC 为小分子域 | 17 个端点并非肽实验监督 | 获取肽/环肽原始 ADMET/tox 标签，保留 transfer 仅作基线 |
| P0 | 4 个 endpoint 无诚实 sequence 监督集 | 不能由显示需求倒逼伪标签 | PEPlife2/DRAMP、CPPsite2、TopFIND/MEROPS、IEDB 等逐源 license + 语义审计 |
| P1 | 修饰/环化表示不足 | 裸 sequence 会丢失 D-aa、N-methyl、脂化、环化等关键性质 | 扩展 HELM/PIR/SMILES/topology 输入与模型表示 |
| P1 | 条件混合和删失 | species/matrix/assay/route 不同会制造伪相关 | task-conditioned 模型、删失 likelihood/区间模型、联合 identity/source split |
| P1 | 许可证与公开发布 | 并非所有原始数据允许再分发 | 每个 source 完成 raw SHA、license、provenance、dedup、split 审计后再升级 |
| P2 | 不确定性与 abstention | 低证据输出不能只给点估计 | OOD、conformal interval、abstention / “insufficient evidence” 策略 |

---

## 10. 推荐 PPT 页序与每页要点（12 页）

1. **问题与目标**：输入肽/环肽序列，统一预测 ADMET/PBPK；强调“研究原型，不是临床产品”。
2. **端点全景**：39 = 10 core + 21 extended + 8 peptide-specific；展示三层面板图。
3. **数据全景**：用分层漏斗展示 215,568 V15 staging -> 20,052 cleaning input -> 19,979 representative -> strict/censored/binary/PU/review。
4. **数据收集与可追溯性**：来源冻结、URL/file ID、SHA、license、DOI/PMID、条件/单位/删失、identity split。
5. **TDC 扩充**：17 端点、97,342 行；列出“small molecule transfer only”和 LD50 符号转换。
6. **本地肽数据**：16,622 staging 行、11 个有记录端点；重点展示 toxicity/hemolysis/HC50/CPP/stability。
7. **模型架构**：输入校验 -> 93 维特征 -> local C model / E prior -> 统一 JSON；强调不调用第三方预测 API。
8. **39 端点覆盖图**：C=25、E=14、A/B=0；“39 有显示 != 39 已验证”。
9. **端点案例**：选 HIA（跨域 C）、HC50（本地数值 C）、Kp（E 五组织向量）、degradation（E 逐残基向量）解释差异。
10. **验证与模型性能现状**：工程测试通过；Phase2 6/6 未晋级，强调严格晋级门而非挑选漂亮分数。
11. **当前风险/数据缺口**：4 个关键无监督标签 endpoint、跨域偏移、修饰表示、许可和外部验证。
12. **下一阶段里程碑**：新肽源审计 -> source-held-out -> OOD/UQ -> endpoint-by-endpoint 晋级；展示可量化交付物。

---

## 11. 文件索引（汇报时可追溯）

| 主题 | 事实来源 |
|---|---|
| 39 端点合同 | `configs/peptide_omnipanel_v28_endpoint_registry.json` |
| 来源状态与使用政策 | `configs/peptide_omnipanel_v28_source_registry.json` |
| 17 TDC 的 URL/行数/hash | `data/peptide_omnipanel_v28_extended_tdc_20260719_run1/source_manifest.tsv` |
| TDC 冻结集统计 | `data/peptide_omnipanel_v28_extended_tdc_20260719_run1/dataset_manifest.json` |
| 本地肽 staging 统计 | `data/peptide_omnipanel_v28_research_data_20260719_run1/dataset_manifest.json` |
| 25 个 research artifacts | `data/peptide_omnipanel_v28_research_bundle_20260719_run1/research_bundle_manifest.json` |
| 当前 39 端点覆盖与安全边界 | `docs/peptide_omnipanel_v28_research_coverage_20260719.md` |
| 39 行端点附表 | `docs/peptide_omnipanel_v28_current_progress_ppt_endpoint_inventory_20260720.tsv` |
| 清洗层总体统计 | `data/peptide_ml_cleaning_v1/manifest.json`、`data/peptide_ml_cleaning_v1_statistics_v1/summary.json` |
| 严格性能/晋级决策 | `data/peptide_omnipanel_v28_phase2_promotion_20260719_run1/promotion_manifest.json` |

---

## 12. 可直接口述的结束语

> 当前阶段已经解决了“没有统一语义、没有可追溯来源、不能统一显示”的基础设施问题：我们形成了 39 端点注册表、分层数据资产、25 个本地研究模型和 39/39 的透明显示层。下一阶段的核心不是继续堆模型数量，而是补齐肽/环肽域原始标签、处理修饰/环化表示，并用 source-held-out 与独立外部验证逐端点决定是否晋级。所有当前低证据预测均已被显式标记，避免把研究原型误作已验证结果。
