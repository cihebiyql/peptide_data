# 肽/环肽 ADMET/PBPK 机器学习清洗 v1（步骤 1–6）

## 结论

本轮已将 V15 中当前关注的 LogD、溶解度、F、T1/2、PPB、CL、Vd、BBB、Kp、permeability 和 cell penetration 数据落成可复现的分层清洗发布。输入范围为 20,052 条，归一化观测去重后为 19,979 条。

这不等于“19,979 条都可以直接训练”。数据已物理分层为：

- `strict_numeric.tsv`：11,591 条严格点回归候选，只允许 `relation = "="`，具有有效 identity 和模型表示；
- `censored_observations.tsv`：230 条删失/近似数值，保留上下界，不当普通点回归标签；
- `binary_evidence_catalog.tsv`：3,930 条二分类数据，但是它们分属 threshold-derived、rough source-defined 或 weak benchmark，不是 strict experimental binary；
- `positive_unlabeled.tsv`：920 条只有正类成员资格的数据，不得把“未收录”当阴性；
- `review_observations.tsv`：3,308 条 identity、归一化或语义/冲突待审数据；
- `literature_review_cleaned.tsv`：1,560 条文献包 staging 数据仍全部保持 review，本轮没有任何一条被自动提升为训练数据。

## 六个步骤已做什么

### 1. 端点和任务语义拆分

每个 `task_id` 都由声明式语义签名稳定生成，包含端点族、target kind、参数语义、参数 basis、assay、归一单位、transform 和必要实验条件。共生成 1,067 个保守任务，其中 13 个达到 `strict_modelable` 阈值，19 个达到包含弱/派生二分类的 `research_modelable` 阈值。

已确保：

- PAMPA、Caco-2、MDCK、RRCK 不混为同一 permeability 任务；
- systemic terminal T1/2 与 plasma/serum/protease stability 不混；
- F 按 absolute/relative/unspecified basis、route 和 formulation 拆分；
- systemic CL、CL/F、microsomal/hepatocyte intrinsic CL 不混；
- Vss、Vz、Vc、Vp、Vd/F 和未指定 Vd 不混；
- PK 任务保留 dose 和 timepoint，不同剂量/观察窗不再误判为同条件冲突；
- 溶解度数值回归与 source-defined binary 不混，二分类又按溶剂条件拆分；
- BBB 的 compiled positive 和 rule-constructed negative 在同一弱基准任务中保留正负类，但证据类型仍为行级字段，不会被误标成实验阴性。

### 2. Observation 去重与条件冲突

旧 V15 `observation_key` 基于原始值和原始单位，不能识别 `120 min = 2 h`。本轮在单位和任务语义归一后重建：

- `normalized_observation_key`：identity + task + condition + relation + normalized target；
- `measurement_condition_key`：identity + task + condition，不包含 target，用于发现同条件异值；
- `source_condition_key`：再加上 DOI/PMID/source provenance group，用于区分跨来源变异和同来源冲突。

结果：

- 67 组归一化后完全重复，共折叠 73 个多余 member；
- 743 个同 identity/task/condition 异值组，其中 734 个至少共享一个 provenance group；
- 这些异值没有被静默挑选或平均；严格 exact 点中的同来源冲突保守进入 review，删失值仍留在 sidecar 且不可点回归训练。

折叠后的代表行会合并全部 DOI/PMID、URL、license、source group 和 upstream observation key，并在 `member_lineage_json` 中保留每个 member 的来源、context、evidence tier 和 review reason，避免后续按文献切分时丢失另一来源。

### 3. Identity 和模型表示过滤

Identity 组可以使用验证过的 InChIKey、HELM、SMILES 或无结构修饰风险的标准序列连通，但 InChIKey 不被当成字符模型输入。模型表示只允许 sequence、HELM 或 SMILES。

实际拦截 1,081 条 identity review：

- PEPlife2 共 898 条：890 条 `smiles=N.A.`、6 条 `smiles=NA`、2 条无可用 identity；其中前 896 条就是之前会被 sentinel 伪哈希成共享 SMILES identity 的风险行；
- 183 条修饰/非天然/环化肽只有裸 residue projection，没有足够的精确结构表示；它们不再回退为普通裸 sequence 训练 identity。

严格数值表中的表示构成为 HELM 9,603 条、SMILES 1,911 条、标准 sequence 77 条。

### 4. 数值与单位归一

每条数值保留 `raw_value`、`raw_unit`、`normalized_value`、`normalized_unit`、`conversion_formula` 和 `conversion_version`。

当前主要规则：

- T1/2：second/minute/hour/day 统一为 hour 后取 `log10`；week/month 和 source-specific unit 保守 review；
- Permeability：`10^-6 cm/s` 换算为 cm/s 后取 `log10`；已是 source log10 的值不会重复取 log；
- F：`%` 转 fraction，保留 route/formulation/basis；
- PPB：`% bound` 转 fraction bound；方向不明或 source-specific PPBR 不自动转换；
- CL：分别保留 body-weight normalized、absolute total-body 和 protein-normalized basis；
- Vd：`mL/kg` 可转 `L/kg`，absolute `L` 不伪造体重换算；
- Solubility：可比的 `uM` 转 `log10(mol/L)`；mole fraction、mass ratio、kg/m3 等只在自己的 source dimension/solvent/temperature 任务内取 log，不做无 MW/密度支持的跨量纲合并；
- LogD：数值保持无量纲，pH 和方法是任务语义。

### 5. Censoring 处理

`<`、`<=`、`>`、`>=` 和 `~` 全部保留 relation、lower/upper bound、单位转换和来源，共 230 条。严格点回归表中 11,591 条全部为 `normalized_relation = "="` 且 `censoring_type = "exact"`。空 relation 只有在明确 source contract 中才能补为 exact，其他进 review。

### 6. 二分类证据分层

| 端点 | 行数 | 正/负 | 层级 | 可解释用途 |
|---|---:|---:|---|---|
| BBB | 844 | 421 / 423 | `weak_benchmark_reproduction_only` | compiled positive + rule-constructed negative；严格实验二分类为 0 |
| Solubility | 2,168 | 1,333 / 835 | `rough_solvent_conditioned_benchmark` | source-defined 粗标签，已按 7 类溶剂条件分任务 |
| T1/2 | 918 | 284 / 634 | `derived_binary_research` | 由连续 T1/2 按 1 h 阈值派生，与回归任务分开 |
| BBB positive-only | 136 | 136 / 0 | `positive_unlabeled_only` | 只能用于 PU/corroboration |
| Cell penetration | 784 | 784 / 0 | `positive_unlabeled_only` | 只能用于 PU/corroboration |

二分类 catalog 中没有 strict experimental binary 任务。弱标签数据仍然可以用，但必须按 `binary_use_tier` 明确报告，不能宣称为实验真值。

## 端点级清洗结果

| 端点族 | 归一化代表行 | strict numeric | censored | binary catalog | PU | review |
|---|---:|---:|---:|---:|---:|---:|
| LogD | 151 | 79 | 0 | 0 | 0 | 72 |
| Solubility | 2,488 | 89 | 0 | 2,168 | 0 | 231 |
| F | 44 | 12 | 3 | 0 | 0 | 29 |
| T1/2 | 4,932 | 1,392 | 195 | 918 | 0 | 2,427 |
| PPB | 140 | 103 | 7 | 0 | 0 | 30 |
| CL | 93 | 49 | 0 | 0 | 0 | 44 |
| Vd | 58 | 38 | 0 | 0 | 0 | 20 |
| BBB | 980 | 0 | 0 | 844 | 136 | 0 |
| Permeability | 10,309 | 9,829 | 25 | 0 | 0 | 455 |
| Cell penetration | 784 | 0 | 0 | 0 | 784 | 0 |
| **总计** | **19,979** | **11,591** | **230** | **3,930** | **920** | **3,308** |

11,591 条 strict numeric 来自 9,929 个唯一 identity 和 16 个 source。其中 permeability 为 9,829 条，占 84.8%；T1/2 为 1,392 条。F、CL、Vd 等 PK 端点虽已清洗，但每个严格语义任务仍然很小。

## 数据现在是否“干净”

答案是：**分层后的 strict numeric 已经达到探索性机器学习前的核心清洗标准，但整个 normalized release 不是一张可无条件全量训练的“干净大表”。** 这是有意设计：弱证据、PU、删失和 review 数据被保留但物理分开，不会被安静地混入 strict 训练集。

距离正式模型发布仍有四个主要工作：

1. 按 `identity_group_id` + source/DOI group 做去泄漏 train/validation/test split，本轮明确没有伪造 split；
2. 对 734 个共享 provenance 的条件异值组做重复实验/条件缺失/录入错误审核；
3. 完成 license/reuse 生产发布门；`manifest.json` 明确记录 `provenance_license_gate_completed=false`；
4. 对 F、CL、Vd、LogD、PPB 等小任务继续补数，并建立外部独立验证集。

## 产物和复现

主产物目录：`data/peptide_ml_cleaning_v1/`

- 全部观测和分层：`normalized_observations.tsv`
- 严格点回归：`strict_numeric.tsv`
- 删失 sidecar：`censored_observations.tsv`
- 二分类证据：`binary_evidence_catalog.tsv`
- PU：`positive_unlabeled.tsv`
- 待审：`review_observations.tsv`
- 重复和冲突：`exact_duplicates.tsv`、`condition_conflicts.tsv`
- 任务语义注册：`task_registry.tsv`
- 文献 staging：`literature_review_cleaned.tsv`
- 机器可读统计、断言和 SHA-256：`manifest.json`

构建命令：

```bash
python scripts/build_peptide_ml_cleaning_v1.py
python -m unittest tests.test_build_peptide_ml_cleaning_v1 -v
```

已对两个独立临时输出目录执行重建，所有 TSV 逐字节一致。26 个回归测试全部通过。V15 主表未被修改：

```text
data/peptide_property_expansion_v15/peptide_property_observations_v15.tsv
SHA-256 9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf
```
