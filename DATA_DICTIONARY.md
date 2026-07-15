# Data Dictionary

## Core Observation Files

所有核心观察表使用UTF-8、Tab分隔、LF换行。

### `normalized_observations.tsv`

包含19,979条归一化代表observation。主要字段组如下。

#### Identifiers and Lineage

| Field | Meaning |
|---|---|
| `clean_row_id` | 清洗发布内稳定行ID |
| `record_id` | 被选作代表记录的上游ID |
| `member_record_ids` | 折叠进该observation的全部记录ID |
| `member_count` | 原始member数量；全表求和应为20,052 |
| `member_lineage_json` | 每个member的来源、DOI、URL、license、context、evidence和review reason |
| `source_id/source_ids` | 代表来源和全部来源 |
| `source_record_id/source_record_ids` | 来源内记录ID |
| `doi_pmid` | DOI或PMID，可包含多个来源 |
| `source_group_ids` | 防来源泄漏的DOI/PMID/source分组 |

#### Endpoint and Task Semantics

| Field | Meaning |
|---|---|
| `raw_endpoint` | 上游原始端点名 |
| `endpoint_family` | 归一化端点族 |
| `task_id` | 唯一语义任务ID |
| `task_kind` | regression、binary classification或positive-unlabeled |
| `task_semantic_signature` | task的完整声明式语义JSON |
| `parameter_semantics` | systemic clearance、CL/F、Vss、PAMPA等具体语义 |
| `parameter_basis` | absolute、apparent_over_f、fraction等basis |
| `assay_family` | 归一化assay类别 |
| `condition_json` | species、matrix、route、dose、timepoint等条件 |

#### Raw and Normalized Targets

| Field | Meaning |
|---|---|
| `raw_value/raw_relation/raw_unit/raw_label` | 来源值、关系、单位和标签 |
| `normalized_value` | 任务内规范值；binary/PU中也可能为0/1，不能仅按非空筛连续数值 |
| `normalized_lower/normalized_upper` | 删失或近似值边界 |
| `normalized_relation` | `=`、`<`、`>`、`<=`、`~`或label |
| `normalized_unit` | 规范单位 |
| `target_transform` | identity、log10等 |
| `conversion_formula/version` | 单位转换公式和版本 |
| `censoring_type` | exact、upper_bound、lower_bound、approximate或not_applicable |

#### Identity and Model Representation

| Field | Meaning |
|---|---|
| `identity_alias_ids` | 经过验证的InChIKey/HELM/SMILES/sequence alias |
| `identity_group_id` | alias连通后的身份组；空值不计作identity |
| `representation_type` | sequence、helm、smiles或空 |
| `representation_text` | 带 `SEQ:`、`HELM:`、`SMILES:` 前缀的实际模型payload |
| `sequence_model_eligible` | 可否作为标准序列模型输入 |
| `structure_model_eligible` | 是否有HELM或SMILES结构表示 |

#### Deduplication, Conflict and Partition

| Field | Meaning |
|---|---|
| `measurement_condition_key` | identity + task + condition，不含target |
| `source_condition_key` | 按每个来源组生成的条件key |
| `normalized_observation_key` | identity + task + condition + normalized target |
| `condition_variability` | 同条件存在多个target |
| `hard_condition_conflict` | 至少一个共同来源组内存在不同target |
| `partition` | strict_numeric、censored_numeric、binary、PU或review层 |
| `review_reasons` | 进入review的原因集合 |
| `research_train_eligible` | 当前研究层可用性；不等于生产许可已通过 |
| `production_train_eligible` | 当前为 `not_assessed_step7` |

## Specialized Files

| File | Description |
|---|---|
| `strict_numeric.tsv` | 11,591条exact点回归候选 |
| `censored_observations.tsv` | 230条带上下界或近似关系的数据 |
| `binary_evidence_catalog.tsv` | 3,930条弱、来源定义或阈值派生二分类 |
| `positive_unlabeled.tsv` | 920条positive-only记录 |
| `review_observations.tsv` | 3,308条待审记录 |
| `exact_duplicates.tsv` | 67个单位/任务归一化后的完全重复组 |
| `condition_conflicts.tsv` | 743个同identity/task/condition异值组 |
| `task_registry.tsv` | 1,067个任务的语义、规模和modelability |
| `literature_review_cleaned.tsv` | 1,560条独立文献review staging |

## Statistics Files

| File | Description |
|---|---|
| `clean_endpoint_inventory.tsv` | 11个声明端点族的分区规模，含Kp=0 |
| `v15_raw_endpoint_inventory.tsv` | V15全部50个原始端点 |
| `strict_numeric_task_distribution.tsv` | 620个strict numeric task的分位数、离群和来源占比 |
| `strict_numeric_endpoint_unit_distribution.tsv` | endpoint/semantics/unit描述性分布 |
| `strict_numeric_histograms.tsv` | 大任务和可比组直方分布 |
| `sequence_representation_summary.tsv` | 原始sequence投影、缺失和标准序列统计 |
| `selected_representation_distribution.tsv` | sequence/HELM/SMILES payload长度和近似残基数 |
| `sequence_length_histogram.tsv` | 标准序列长度与HELM/sequence单体数 |
| `amino_acid_composition.tsv` | 观察、member和唯一序列三种权重的氨基酸组成 |
| `binary_task_distribution.tsv` | binary/PU正负类、证据层和来源占比 |
| `censoring_distribution.tsv` | relation和bound分布 |
| `task_size_viability.tsv` | task规模桶、modelability和失败原因 |
| `row_anomalies.tsv` | 444条需优先原文复核的strict记录 |
