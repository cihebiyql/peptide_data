# 肽/环肽 ADMET/PBPK 全端点、序列与数值分布统计 v1

## 统计范围和口径

本报告同时覆盖两个分母，不将它们混在一起：

1. V15 `peptide_property_training_deduplicated_v15.tsv`：215,568 行、50 个原始端点，用于“全部端点”总盘点；
2. `peptide_ml_cleaning_v1`：19,979 条归一化代表 observation，用于 ADMET/PBPK 分层、序列/结构表示和可比数值分布。

文献 staging 1,560 行是独立 review 层，未加入 19,979 的训练分母。

数值分布遵守以下硬规则：

- binary/PU 的 `0/1` 不会混入连续数值分布；
- 正式数值统计按 `task_id` 进行；
- 端点展示表至少按 endpoint + semantics + unit + transform 分层；
- 下文的 endpoint + unit 合并表只是描述性诊断，不代表可以跨 task 直接合并训练。

## 总体结果

| 指标 | 数量 |
|---|---:|
| V15 总行数 | 215,568 |
| V15 原始端点 | 50 |
| V15 数值行 | 45,233 |
| V15 标签行 | 170,335 |
| ML-clean 声明端点族 | 11（其中 Kp=0） |
| ML-clean 归一化代表行 | 19,979 |
| Strict numeric | 11,591 |
| Binary evidence catalog | 3,930 |
| Positive-unlabeled | 920 |
| Censored | 230 |
| Review | 3,308 |
| 唯一 identity group（不计空值） | 14,374 |
| 唯一 selected representation | 14,869 |
| 总 task | 1,067 |
| 含 strict numeric 的 task | 620 |
| Strict-modelable task | 13 |

V15 任务类型为：145,078 条 source-defined function multilabel、44,215 条 regression、23,270 条 binary、1,926 条 binary classification、795 条 numeric regression、223 条 censored regression/numeric regression、52 条 source-defined binary classification 和 9 条 multiclass classification。

## V15 全部 50 个原始端点

“唯一标准序列”只统计原字段中严格由20种标准大写氨基酸组成的唯一序列，不把小写 D-residue、Aib、修饰符号、HELM 或 SMILES 算成普通序列。

| 原始端点 | 总行 | 数值 | 标签 | 唯一 identity | 唯一标准序列 |
|---|---:|---:|---:|---:|---:|
| therapeutic_function | 141,921 | 0 | 141,921 | 57,778 | 54,241 |
| hemolysis | 33,566 | 24,227 | 9,339 | 19,524 | 9,006 |
| general toxicity | 11,036 | 0 | 11,036 | 11,036 | 11,036 |
| permeability | 10,153 | 10,153 | 0 | 9,441 | 0 |
| cytotoxicity | 5,615 | 4,724 | 891 | 4,107 | 1,221 |
| T1/2 | 4,930 | 4,012 | 918 | 2,230 | 1,222 |
| solubility | 2,488 | 320 | 2,168 | 1,386 | 1,344 |
| drug_delivery | 1,578 | 0 | 1,578 | 1,578 | 1,461 |
| BBB | 844 | 0 | 844 | 844 | 844 |
| cell_penetration | 784 | 0 | 784 | 784 | 737 |
| tumor_homing | 659 | 0 | 659 | 659 | 658 |
| hemolysis_percent | 511 | 511 | 0 | 216 | 216 |
| efflux_ratio | 185 | 185 | 0 | 145 | 0 |
| cytotoxicity_ic50 | 147 | 147 | 0 | 113 | 113 |
| PPB | 140 | 140 | 0 | 82 | 20 |
| bbb_penetration | 136 | 0 | 136 | 136 | 67 |
| LogD7.4 | 129 | 129 | 0 | 74 | 25 |
| CL | 72 | 72 | 0 | 16 | 8 |
| Vd | 53 | 53 | 0 | 9 | 7 |
| hemolysis_hc50 | 52 | 52 | 0 | 52 | 52 |
| plasma_concentration | 47 | 47 | 0 | 4 | 0 |
| F | 44 | 44 | 0 | 31 | 4 |
| cytotoxicity_cc50 | 39 | 39 | 0 | 39 | 39 |
| hemolysis_mhc10 | 39 | 39 | 0 | 39 | 39 |
| cytotoxicity_percent | 38 | 38 | 0 | 30 | 30 |
| cytotoxicity_proxy | 36 | 36 | 0 | 12 | 0 |
| cytotoxicity_ec50 | 32 | 32 | 0 | 31 | 31 |
| hemolysis_mhc | 32 | 32 | 0 | 32 | 32 |
| hemolysis_source_defined_binary | 32 | 0 | 32 | 32 | 32 |
| chromatographic_retention | 22 | 22 | 0 | 22 | 0 |
| logd | 22 | 22 | 0 | 22 | 0 |
| clearance | 21 | 12 | 9 | 9 | 0 |
| stability_half_life | 21 | 21 | 0 | 21 | 21 |
| cytotoxicity_source_defined_binary | 20 | 0 | 20 | 20 | 20 |
| permeability_recovery | 16 | 16 | 0 | 8 | 0 |
| hemolysis_ec50 | 15 | 15 | 0 | 15 | 15 |
| half_life | 13 | 13 | 0 | 9 | 0 |
| lipophilicity_proxy | 12 | 12 | 0 | 12 | 0 |
| membrane_retention | 12 | 12 | 0 | 12 | 0 |
| cytotoxicity_lc50 | 10 | 10 | 0 | 8 | 8 |
| exposure | 10 | 10 | 0 | 3 | 0 |
| plasma_stability | 10 | 10 | 0 | 9 | 0 |
| cell_viability_percent | 7 | 7 | 0 | 7 | 7 |
| hemolysis_lc50 | 6 | 6 | 0 | 6 | 6 |
| volume_distribution | 5 | 5 | 0 | 3 | 0 |
| hemolysis_hd50 | 4 | 4 | 0 | 4 | 4 |
| cytotoxicity_hc50 | 1 | 1 | 0 | 1 | 1 |
| hemolysis_concentration_at_50_percent | 1 | 1 | 0 | 1 | 1 |
| hemolysis_concentration_at_5_percent | 1 | 1 | 0 | 1 | 1 |
| hemolysis_ic50 | 1 | 1 | 0 | 1 | 1 |

## ML-clean 端点分层

Review = identity review + normalization review + semantic review。`input member` 和 `representative` 的差异来自归一化重复折叠。

| 端点族 | Input member | Representative | Strict | Censored | Binary | PU | Review | Tasks | Strict-modelable | Identity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogD | 151 | 151 | 79 | 0 | 0 | 0 | 72 | 9 | 0 | 82 |
| Solubility | 2,488 | 2,488 | 89 | 0 | 2,168 | 0 | 231 | 89 | 0 | 1,386 |
| F | 44 | 44 | 12 | 3 | 0 | 0 | 29 | 18 | 0 | 30 |
| T1/2 | 4,964 | 4,932 | 1,392 | 195 | 918 | 0 | 2,427 | 775 | 2 | 2,225 |
| PPB | 140 | 140 | 103 | 7 | 0 | 0 | 30 | 52 | 0 | 80 |
| CL | 93 | 93 | 49 | 0 | 0 | 0 | 44 | 49 | 0 | 24 |
| Vd | 58 | 58 | 38 | 0 | 0 | 0 | 20 | 45 | 0 | 11 |
| BBB | 980 | 980 | 0 | 0 | 844 | 136 | 0 | 2 | 0 | 980 |
| Kp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Permeability | 10,350 | 10,309 | 9,829 | 25 | 0 | 0 | 455 | 27 | 11 | 8,943 |
| Cell penetration | 784 | 784 | 0 | 0 | 0 | 784 | 0 | 1 | 0 | 784 |

最重要的规模结论：

- Strict numeric 总量 11,591，但 permeability 一项就有 9,829 条，占 84.8%；
- Kp 在声明 scope 中，但当前数据为 0；
- F、CL、Vd 的“总行数”并不等于可用于同一任务的样本数。

## 序列、Identity 和模型表示分布

### 原始 sequence 字段

| 指标 | 行数 |
|---|---:|
| `sequence` 非空 | 17,972 |
| 排除 `None/N.A./NA` 等 sentinel 后有内容 | 17,908 |
| 严格大写标准20AA序列 | 8,152 |
| 修饰/非天然/小写D-residue/列表表示/非标准 | 9,756 |
| 缺失或 sentinel | 2,071 |

原始 `sequence` 字段不是“普通序列模型输入”的同义词。CycPeptMPDB 的单体列表、D-residue、Aib、端基修饰、环化信息和长蛋白都可能出现在这个字段中。

### 最终 selected representation

| 表示 | 行数 | 唯一 identity | 唯一 representation | 模型残基数 min/P25/median/P75/P95/P99/max | 字符长度 median/max |
|---|---:|---:|---:|---|---:|
| HELM | 14,435 | 11,113 | 11,549 | 1 / 7 / 10 / 18 / 39 / 165 / 997 | 86 / 2,012 |
| Sequence | 2,053 | 1,750 | 1,750 | 2 / 8 / 12 / 23 / 40 / 63.44 / 100 | 16 / 104 |
| SMILES | 2,410 | 1,569 | 1,570 | 不声称可靠残基数 | 83 / 24,615 |
| 无有效表示 | 1,081 | 0 | 0 | - | - |

真正 `sequence_model_eligible=true` 只有 2,053 条；`structure_model_eligible=true` 有 16,845 条。Strict numeric 11,591 条中：

- HELM：9,603；
- SMILES：1,911；
- Sequence：77。

所以当前数值数据主要是“结构/HELM建模数据”，不是纯普通氨基酸序列数据。

## Strict numeric 任务规模分布

| 端点 | Strict行 | Tasks | Singleton task | 2–4行 task | Strict-modelable | 最大task |
|---|---:|---:|---:|---:|---:|---:|
| CL | 49 | 38 | 36 | 0 | 0 | 7 |
| F | 12 | 10 | 9 | 1 | 0 | 3 |
| LogD | 79 | 7 | 0 | 0 | 0 | 22 |
| PPB | 103 | 47 | 39 | 1 | 0 | 16 |
| T1/2 | 1,392 | 410 | 253 | 109 | 2 | 375 |
| Vd | 38 | 38 | 38 | 0 | 0 | 1 |
| Permeability | 9,829 | 26 | 1 | 3 | 11 | 6,814 |
| Solubility | 89 | 44 | 37 | 2 | 0 | 17 |

620 个 strict numeric task 中：

- 413 个只有1条；
- 116 个有2–4条；
- 45 个有5–9条；
- 33 个有10–29条；
- 只有13个达到当前 strict-modelable 阈值。

尤其是 Vd：38 条 strict 数据实际是38个 singleton task，不能理解为一个 n=38 的同质数据集。

## Strict numeric 数值分布

下表按 endpoint + normalized unit 做描述性展示。对 CL、Vd、permeability、solubility，即使单位相同也仍需继续按 parameter semantics/assay/task 拆分；完整57个可比组和620个 task 的分布在 TSV 中。

| Endpoint | Unit | n | Min | P25 | Median | P75 | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CL | L/h | 15 | 0.629 | 1.34 | 4.32 | 439.2 | 1,161.63 | 1,416.5 |
| CL | mL/min/kg | 27 | 1.597 | 18.508 | 36.3 | 57.257 | 90.87 | 104 |
| CL | uL/min/mg protein | 7 | 10 | 38 | 160 | 192 | 286.7 | 323 |
| F | fraction | 12 | 0.007 | 0.0515 | 0.137 | 0.493 | 0.635 | 0.77 |
| LogD | dimensionless log10 | 79 | -1.70 | 1.345 | 1.98 | 2.76 | 4.316 | 5.26 |
| PPB | fraction bound | 103 | 0 | 0.52 | 0.89 | 0.9825 | 0.992 | 0.999 |
| T1/2 | log10(h) | 1,392 | -6.653 | -3.396 | -0.431 | 0.603 | 1.575 | 2.723 |
| Vd | L | 15 | 5.87 | 43.9 | 104 | 354.37 | 484.97 | 595.5 |
| Vd | L/kg | 23 | 0.162 | 0.359 | 1.83 | 3.01 | 12.195 | 12.738 |
| Permeability | source log10 permeability | 8,265 | -9.46 | -6.24 | -5.71 | -5.26 | -4.59 | -3.46 |
| Permeability | log10(cm/s) | 1,402 | -8.699 | -7.00 | -6.382 | -5.315 | -4.50 | -3.575 |
| Permeability | log10(efflux ratio) | 150 | -0.620 | 0.003 | 0.477 | 0.889 | 2.322 | 2.580 |
| Permeability | fraction retained | 12 | 0 | 0.0575 | 0.095 | 0.175 | 0.583 | 0.66 |
| Solubility | log10(mol/L) | 41 | -5.553 | -3.385 | -3.169 | -3.120 | -3.059 | -3.029 |
| Solubility | log10(mole fraction) | 36 | -4.500 | -3.488 | -3.202 | -2.918 | -2.686 | -2.562 |
| Solubility | log10(kg/m3) | 5 | -0.222 | 0.176 | 0.720 | 1.859 | 2.215 | 2.303 |
| Solubility | log10(mass ratio) | 6 | -2.409 | -2.246 | -1.484 | -1.193 | -0.781 | -0.643 |
| Solubility | log10(mass fraction) | 1 | -1.775 | -1.775 | -1.775 | -1.775 | -1.775 | -1.775 |

### T1/2 语义子分布

T1/2 全体的 `log10(h)` 分布会混合不同实验意义，必须再拆分：

| 语义 | n | Min | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Systemic terminal PK | 620 | -2.323 | 0.204 | 1.775 | 2.723 |
| Plasma/serum stability | 213 | -2.214 | 0.114 | 1.469 | 2.068 |
| Protease/intestinal stability | 393 | -6.653 | -5.108 | -2.206 | 1.358 |
| Other context | 166 | -2.380 | -0.014 | 1.740 | 2.654 |

Protease/intestinal stability 的中位数 `log10(h)=-5.108`，约为 0.028 秒，与常规肽稳定性直觉明显不一致，已进入异常复核清单。

## Binary、PU 和 Censoring 分布

### Binary/PU

| 端点 | 行数 | 正类 | 负类 | 证据层 | 说明 |
|---|---:|---:|---:|---|---|
| BBB | 844 | 421 | 423 | weak benchmark | compiled positive + rule-constructed negative |
| T1/2 | 918 | 284 | 634 | derived binary | 由1h阈值派生 |
| Solubility | 2,168 | 1,333 | 835 | rough source-defined | 7个溶剂task，只有4个同时有正负类 |
| BBB PU | 136 | 136 | 0 | positive-unlabeled | 不能把未收录样本当阴性 |
| Cell penetration PU | 784 | 784 | 0 | positive-unlabeled | 不能普通二分类 |

所有 weak/derived/PU task 的 `strict_modelable=false`。BBB、T1/2和4个同时有正负类的溶解度task达到 `research_modelable=true`，但仍不是严格实验二分类。

### Censoring

230 条删失/近似观测全部 `point_regression_eligible=false`：

- T1/2：195，包括 `>` 73、`~` 68、`<` 51、`<=` 3；
- Permeability：25，包括 `>` 21、`<` 4；
- PPB：7，包括 `>` 6、`<` 1；
- F：3，`>`、`<`、`~` 各1条。

## 文献 review 独立分布

| 端点 | 子类 | 行数 | 训练行 |
|---|---|---:|---:|
| BBB binary benchmark | B3Pdb positive membership | 269 | 0 |
| BBB evidence | BrainPeps literature membership | 357 | 0 |
| BBB subfunction | Blood-brain barrier | 185 | 0 |
| GI stability | Gastric | 46 | 0 |
| GI stability | Intestinal | 63 | 0 |
| T1/2 | Official PEPlife2 repair | 5 | 0 |
| T1/2 | Blood/serum stability half-life | 635 | 0 |
| **总计** |  | **1,560** | **0** |

## 发现的真实异常

`row_anomalies.tsv` 中共列出444条需回溯的 strict numeric 记录：

1. **343条亚秒级 protease/intestinal T1/2**：normalized half-life < 1 second；最小原始记录为 0.0008 seconds。这不是换算程序错误，但参数语义/原始单位极不寻常，需回原论文。
2. **94条标准序列长度 >100 aa 的 strict numeric**：包含最长997 aa的蛋白/酶和 `Activity Half Life`，明显超出目标肽 ADMET 范围。
3. **7条 total-body L/h-scale clearance >100 L/h**：包括5条 apparent CL/F（由6.81–7.66 L/min转为408.6–459.6 L/h），以及2条1,052.4和1,416.5 L/h的human IV systemic clearance；需分别复核 CL/CL/F 语义、单位、尺度和抽取。

这444条当前仍在 strict 表中，因为本轮任务是统计审计，没有静默修改或删除上一版数据。但在正式肽模型训练前，它们应进入新的 entity-scope/生理合理性门。

## 产物

统计目录：`data/peptide_ml_cleaning_v1_statistics_v1/`

- `v15_raw_endpoint_inventory.tsv`：50个原始端点总表；
- `v15_raw_numeric_distribution.tsv`：原始 endpoint/task/unit/relation 数值分布；
- `v15_label_distribution.tsv`：全标签分布；
- `clean_endpoint_inventory.tsv`：ML-clean 端点分层规模；
- `sequence_representation_summary.tsv`：原始序列投影统计；
- `selected_representation_distribution.tsv`：sequence/HELM/SMILES 最终模型表示分布；
- `sequence_length_histogram.tsv`：序列长度和 HELM/序列单体数直方表；
- `amino_acid_composition.tsv`：观测加权、member加权和唯一序列氨基酸组成；
- `strict_numeric_endpoint_unit_distribution.tsv`：57个端点/语义/单位可比组；
- `strict_numeric_task_distribution.tsv`：620个 strict task 分位数、离群和来源占比；
- `strict_numeric_histograms.tsv`：大任务和可比组直方分布；
- `binary_task_distribution.tsv`：binary/PU 正负类与证据层；
- `censoring_distribution.tsv`：上下界/近似值分布；
- `task_size_viability.tsv`：1,067个 task 的规模和建模失败理由；
- `identity_multiplicity_distribution.tsv`：identity-observation 重复度；
- `field_missingness.tsv`：端点/分区字段缺失；
- `row_anomalies.tsv`：444条数值/实体范围异常；
- `literature_review_distribution.tsv`：1,560条独立文献 review 统计；
- `summary.json`：输入/输出 SHA-256、统计和13个机器断言。

复现命令：

```bash
python scripts/summarize_peptide_ml_cleaning_v1.py
python -m unittest tests.test_summarize_peptide_ml_cleaning_v1 -v
```

两个独立临时目录重建的所有 TSV 和 README 逐字节一致，10个统计回归测试全部通过。

## 对机器学习可用性的最终判断

- **Permeability**：已有多个大任务，是当前最接近可建模的端点；
- **T1/2**：总量较大，但语义破碎，且存在亚秒级值和长蛋白范围污染，必须再做 entity-scope 和原文复核；
- **LogD/PPB/Solubility**：可用于小样本或转移学习，但当前没有达到任务级大样本门；
- **F/CL/Vd**：参数语义、剂量、途径、单位拆分后高度碎片化；CL还有极端值需回溯；
- **BBB/Solubility binary/T1/2 binary**：可作弱标签研究基准，不能称为严格实验二分类；
- **Cell penetration/BBB PU**：只能做 positive-unlabeled 学习；
- **Kp**：当前为0，距离建模最远。
