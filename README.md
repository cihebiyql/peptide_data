# Peptide / Cyclic-Peptide ADMET-PBPK Data Release

这是一个面向机器学习研究的肽/环肽 ADMET、PK/PBPK 性质数据发布。仓库包含经过任务语义拆分、identity 安全过滤、单位归一、删失值保留、证据分层和 observation 去重后的数据，以及对应的全端点、序列和数值分布统计。

> 重要：本仓库是研究和审计发布，不代表所有行都可以无条件用于训练或商业再分发。请按 `partition`、`binary_use_tier`、来源 license 和 `DATA_USE_NOTICE.md` 使用。

## Release Snapshot

| 指标 | 数量 |
|---|---:|
| ML-clean 输入范围 | 20,052 |
| 归一化代表 observation | 19,979 |
| Strict numeric | 11,591 |
| Censored sidecar | 230 |
| Binary evidence catalog | 3,930 |
| Positive-unlabeled | 920 |
| Review | 3,308 |
| 声明端点族 | 11（10个非零，Kp=0） |
| 总 task | 1,067 |
| Strict numeric task | 620 |
| Strict-modelable task | 13 |
| 唯一 identity group | 14,374 |
| 唯一 selected representation | 14,869 |

### Strict Numeric Endpoints

| Endpoint | Rows |
|---|---:|
| Permeability | 9,829 |
| T1/2 | 1,392 |
| PPB | 103 |
| Solubility | 89 |
| LogD | 79 |
| CL | 49 |
| Vd | 38 |
| F | 12 |
| BBB / Kp / cell penetration | 0 |

Permeability 占 strict numeric 的 84.8%。总行数不能直接等同于可训练规模：620个 strict numeric task 中有413个 singleton task，只有13个达到当前每任务规模门槛。

## Which File Should I Use?

| 目的 | 文件 | 注意事项 |
|---|---|---|
| 查看所有归一化记录 | `data/peptide_ml_cleaning_v1/normalized_observations.tsv` | 包含 strict、weak、PU、censored 和 review，不可整表直接训练 |
| 点回归候选 | `data/peptide_ml_cleaning_v1/strict_numeric.tsv` | 全部为 exact relation；仍需排除 `row_anomalies.tsv` 中的444条高优先级异常 |
| 删失回归 | `data/peptide_ml_cleaning_v1/censored_observations.tsv` | 保存 lower/upper bound；不可作为普通点标签 |
| 弱/派生二分类 | `data/peptide_ml_cleaning_v1/binary_evidence_catalog.tsv` | BBB、solubility、T1/2 标签证据不同，按 `binary_use_tier` 使用 |
| Positive-unlabeled | `data/peptide_ml_cleaning_v1/positive_unlabeled.tsv` | 不得把未命中数据库的样本自动视为负类 |
| 人工复核 | `data/peptide_ml_cleaning_v1/review_observations.tsv` | 不进入普通训练 |
| 任务定义 | `data/peptide_ml_cleaning_v1/task_registry.tsv` | 每个 `task_id` 对应唯一语义签名 |
| 全面统计 | `data/peptide_ml_cleaning_v1_statistics_v1/` | 端点、任务、序列、数值、标签、缺失和异常统计 |

## Data Layers

### Strict Numeric

`strict_numeric.tsv` 只包含：

- `normalized_relation = "="`；
- `censoring_type = "exact"`；
- 有效 `identity_group_id` 和模型表示；
- 上游 QC 通过；
- 无 hard condition conflict。

数值仍必须按 `task_id` 或兼容的 parameter semantics + unit + transform 建模。不要跨以下语义混合：

- systemic T1/2、plasma/serum stability、protease/GI stability；
- CL、CL/F、intrinsic/microsomal CL；
- Vss、Vz、Vc、Vp、V/F；
- PAMPA、Caco-2、MDCK、RRCK permeability；
- 不同溶剂、量纲和温度的 solubility。

### Binary and PU

- BBB：421 compiled positive + 423 rule-constructed negative，只是 weak benchmark；
- T1/2：284 positive + 634 negative，由1小时阈值派生；
- Solubility：1,333 positive + 835 negative，是7个溶剂条件下的 source-defined 标签；
- BBB PU：136 positive-only；
- Cell penetration PU：784 positive-only。

本发布没有 strict experimental binary task。

### Sequence and Structural Representations

| Selected representation | Rows |
|---|---:|
| HELM | 14,435 |
| SMILES | 2,410 |
| Sequence | 2,053 |
| Missing | 1,081 |

Strict numeric 中只有77行使用普通 sequence 表示；9,603行使用HELM，1,911行使用SMILES。原始 `sequence` 字段可能只是修饰肽的 residue projection，不能自动当成标准序列模型输入。

## Known High-Priority Review Items

`data/peptide_ml_cleaning_v1_statistics_v1/row_anomalies.tsv` 包含444条仍在 strict numeric 中、但正式训练前应复核的记录：

- 343条小于1秒的 protease/intestinal T1/2；
- 94条标准序列长度超过100 aa，最长997 aa，可能是长蛋白/酶而非目标肽；
- 7条大于100 L/h的 total-body L/h-scale clearance，其中5条是 CL/F，2条是 systemic CL。

这些记录没有被静默删除，便于追溯；建议在建立正式训练 split 前将其路由到新的 entity-scope / physiological-plausibility review。

## Repository Layout

```text
data/
  peptide_ml_cleaning_v1/               # 分层清洗数据
  peptide_ml_cleaning_v1_statistics_v1/ # 全端点/序列/数值统计
docs/                                    # 中文审计和方法报告
scripts/                                 # 构建、统计和发布验证脚本
tests/                                   # 回归测试
DATA_DICTIONARY.md
DATA_USE_NOTICE.md
RELEASE_NOTES.md
RELEASE_MANIFEST.json
SHA256SUMS
```

## Quick Start

大文件建议流式读取：

```python
import pandas as pd

path = "data/peptide_ml_cleaning_v1/strict_numeric.tsv"
for chunk in pd.read_csv(path, sep="\t", chunksize=10_000):
    # Always group/filter by task_id before modeling.
    print(chunk[["task_id", "endpoint_family", "normalized_value"]].head())
```

只使用标准 sequence 模型表示：

```python
import pandas as pd

df = pd.read_csv(
    "data/peptide_ml_cleaning_v1/strict_numeric.tsv",
    sep="\t",
    low_memory=False,
)
sequence_df = df[df["sequence_model_eligible"].eq(True)].copy()
sequence_df["model_sequence"] = sequence_df["representation_text"].str.removeprefix("SEQ:")
```

排除当前异常清单：

```python
import pandas as pd

strict = pd.read_csv("data/peptide_ml_cleaning_v1/strict_numeric.tsv", sep="\t", low_memory=False)
anomalies = pd.read_csv(
    "data/peptide_ml_cleaning_v1_statistics_v1/row_anomalies.tsv",
    sep="\t",
    low_memory=False,
)
filtered = strict[~strict["record_id"].isin(anomalies["record_id"])].copy()
```

## Validation

本发布使用Python标准库即可验证，不需要安装第三方依赖：

```bash
python scripts/validate_published_release.py
sha256sum -c SHA256SUMS
```

完整重建需要本仓库未再分发的上游 V15 源输入。构建入口保留在：

```bash
python scripts/build_peptide_ml_cleaning_v1.py
python scripts/summarize_peptide_ml_cleaning_v1.py
```

冻结验证结果：

- Cleaning manifest：25个断言全部通过；
- Statistics summary：13个断言全部通过；
- Cleaning测试：26个；统计测试：10个；
- 两次独立临时重建的TSV逐字节一致；
- 上游V15主表未修改，其SHA-256为 `9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf`。

## Provenance and Licensing

每条记录保留 `source_id`、`source_record_id`、DOI/PMID、URL、license、source groups 和 `member_lineage_json`。重复 observation 折叠后仍保留所有成员来源。

本仓库没有为所有来源授予统一的新许可证，也不覆盖原数据库或论文的许可条款。请阅读 `DATA_USE_NOTICE.md`，并在再发布、商业使用或训练公开模型前逐来源确认权限。

## Documentation

- 清洗步骤1-6：`docs/peptide-ml-cleaning-v1.md`
- 全端点/序列/数值分布：`docs/peptide-ml-cleaning-v1-statistical-distributions.md`
- V15总发布审计：`docs/peptide-cyclic-peptide-property-expansion-v15.md`
- 文献包整合审计：`docs/literature-data-v15-integration-audit-v1.md`
- Mozi数据整合审计：`docs/mozi-v15-integration-audit-v1.md`

## Peptide OmniPanel V31 科研网站

`agent/v31-peptide18` 分支同时发布 V31 肽类 18 端点网站的公开安全代码与说明：

- 四页面中文界面：序列输入、预测面板、历史任务、网站与结果说明；
- ESM2-650M 常驻编码器与 18 个本地预测头的运行时接口；
- 单条/多条/FASTA/CSV/TSV 输入与 JSON/TSV/ZIP 导出；
- 浏览器级隔离、最多 50 个批次的本地历史和结果恢复；
- 主结果页证据标记降噪，原始 JSON 仍保留 `research_only`、`low_confidence` 和 `validation_status`；
- V31 相关测试、部署记录和真实浏览器截图。

入口文档：

- `docs/peptide_omnipanel_v31_latest_model_architecture_training_results_20260723.md`（最新模型架构、输入合同、训练方法、18 端点逐项 OOF 结果与候选路线比较）
- `docs/peptide_omnipanel_v31_web_history_and_evidence_20260723.md`
- `docs/peptide_omnipanel_v31_web_deployment_20260723.md`
- `docs/peptide_omnipanel_v31_web_batch_chinese_upgrade_20260723.md`

公开仓库**不包含**训练模型权重、Hugging Face 模型缓存、用户提交序列、浏览器历史、运行日志或临时导出。网站源码需要本地 V31 bundle 才能执行真实预测；公开的 bundle manifest、端点注册表和示例预测用于审计接口与证据合同。

## Citation

如使用本仓库，请同时引用具体数据行中的原始 DOI/PMID/数据库来源，并记录所用commit hash、`task_id`、过滤规则和数据分区。

---

Release repository: <https://github.com/cihebiyql/peptide_data>
