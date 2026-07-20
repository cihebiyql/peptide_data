# V2.8 研究版：39 端点统一显示与可追溯性说明（2026-07-19）

## 结论与边界

本轮新增的是**隔离的本地研究显示层**，入口为
`scripts/predict_peptide_omnipanel_v28_research.py`。对任意通过输入校验的
单条 FASTA 或标准 20 种氨基酸的裸序列，它都会返回注册表中的 **39/39** 端点；每一项
均显式包含：

```json
{
  "status": "predicted_low_evidence",
  "research_only": true,
  "low_confidence": true,
  "confidence": "low",
  "evidence_tier": "L0... | L3... | L4... | L5..."
}
```

因此这个输出只能用于内部探索和数据检查，**不能**解释为临床、注册、release
或已验证的肽 / 环肽 ADMET/PBPK 结果。它不导入或修改
`peptide_omnipanel.service`；训练 bundle 中的
`active_service_changed=false`。

供程序读取的同一盘点见
`docs/peptide_omnipanel_v28_research_coverage_20260719.json`。

## 覆盖盘点

| 显示路由 | 端点数 | 含义 |
|---|---:|---|
| C：本地拟合的序列头 | 25 | 有冻结训练观测和本项目生成的 joblib artifact；仍全部是 `research_only + low_confidence`。 |
| E：透明本地先验 | 14 | 有值、可审计，但不是有监督端点模型；不可称为“训练数据已经补齐”。 |
| A/B：已验证 / 可晋级模型 | 0 | 本研究面板没有把任何端点伪装成 A/B。 |

用户所称“剩余 29 个端点”对应 21 个 extended ADMET/tox 与 8 个肽特异端点：
其中 **25/29** 已有 C 级本地模型；其余 **4/29**（`membrane_retention`、
`protease_stability`、`degradation_site_probability`、`immunogenicity_risk`）
保持 E 级，而非伪造训练集或虚构模型。原因见“未补齐项”。其余 10 个 core
端点在这个仅序列研究显示层同样返回 E 级透明先验；这里没有把先前的
structure-only 弱标签模型在缺少结构输入时暗中当作序列模型使用。

### C 级：17 个 TDC 小分子迁移序列学生

`HIA`、`Pgp_inhibition`、6 个 CYP inhibition、3 个 CYP substrate、`hERG`、
`AMES`、`DILI`、`ClinTox`、`carcinogenicity`、`skin_reaction`、`LD50`。

训练过程是：冻结 TDC 小分子原始标签 -> 本项目本地 Morgan teacher -> 对冻结
ChEMBL 肽结构生成伪标签 -> 本项目 93 维序列 student。因而模型的
`evidence_tier=L5_self_model_pseudo_label`，**不是肽实测标签**，也不能把
small-molecule 真实性能外推为肽性能。

* 原始冻结观测：97,342 行（89,957 classification；7,385 regression）；17 端点。
* 原始记录、Dataverse file ID/URL、原文件 SHA-256、正类语义及数值转换均在
  `data/peptide_omnipanel_v28_extended_tdc_20260719_run1/source_manifest.tsv`。
* 数据集清单：`dataset_manifest.json`，SHA-256
  `aec70c4266d6234f55264772715e471fd63a36ec78882b4e36cc3b14441d5e18`。
* `LD50_Zhu` 以 `pLD50=log10(1/(mol/kg))` 发布；入库时取负，统一为注册表的
  `log10(mol/kg)`，转换在逐行字段 `value_transform=negate_pLD50` 中留痕。

### C 级：8 个本地肽序列头

| 端点 | 冻结观察数 | 模型 / 标签限制 |
|---|---:|---|
| `overall_peptide_toxicity` | 11,036 | ExtraTrees 分类；来源定义的二分类。 |
| `hemolysis` | 2,025 | ExtraTrees 分类；不等同于 HC50。 |
| `HC50` | 1,926 | ExtraTrees 回归，`log10(mol/L)`。 |
| `cell_penetration` | 784 | OneClassSVM 正-未标记排序；不能当作有真阴性的概率分类器。 |
| `intestinal_stability` | 393 | ExtraTrees 回归；条件混合、仅研究用。 |
| `cytotoxicity` | 312 | ExtraTrees 分类；来源定义标签。 |
| `human_plasma_stability` | 65 | ExtraTrees 回归；样本小。 |
| `mouse_plasma_stability` | 45 | ExtraTrees 回归；样本小。 |

这些观察都具备 `source_path`、`source_record_id`、source URL / license、单位、
条件字段、`evidence_tier` 和逐行状态。冻结 staging 总计 16,622 行，清单：
`data/peptide_omnipanel_v28_research_data_20260719_run1/dataset_manifest.json`，
SHA-256 `f961ddcc3761829552d90d40b4d4df5dd4571a52603763a668de09d2b8223933`。

## 未补齐项：必须保留 E 级

| 端点 | 已找到的证据 | 为什么未训练 | 当前显示 |
|---|---|---|---|
| `membrane_retention` | 12 条 PAMPA retained-fraction | 全部没有可用标准氨基酸序列，不能训练“输入序列”模型。 | 基于序列理化特征的透明 E 先验。 |
| `protease_stability` | 375 条候选行 | 与 `intestinal_stability` 重叠，不能复制后冒充独立 protease 监督集。 | E 先验。 |
| `degradation_site_probability` | 0 条 residue-level 标签 | 没有逐残基监督标签。 | E 规则向量；输出长度严格等于输入序列长度。 |
| `immunogenicity_risk` | 0 条治疗肽免疫原性标签 | 没有可训练的治疗肽源定义标签。 | E 规则概率。 |

`Kp` 另有 3 条 mouse brain/plasma 的 ChEMBL 审查记录，但并非完整组织向量、
且含修饰/非标准残基，不能被伪装为可训练的 sequence Kp 任务；它返回带
brain/liver/kidney/muscle/adipose 键的 E 级机制向量。`CL` 的 21 条记录只对应
不同 intrinsic-clearance 语义，不能混成注册表所要求的 absolute total CL。

## 可复现调用

```bash
.venv/bin/python scripts/predict_peptide_omnipanel_v28_research.py \
  --sequence ACDEFGHIK \
  --out /tmp/peptide_omnipanel_v28_research_prediction.json
```

输出中 Kp 是组织向量，degradation 是逐残基向量；其他端点是标量。调用方应以
`research_only` 和 `low_confidence` 作为硬性展示标记，而不是只看数值。

## 产物链和后续硬门槛

* 训练 bundle：
  `data/peptide_omnipanel_v28_research_bundle_20260719_run1/research_bundle_manifest.json`
  （25 个 artifact 的 path / SHA-256 均已登记）。
* 源注册表：`configs/peptide_omnipanel_v28_source_registry.json`，新增
  `local_v28_extended_tdc_research` 和 `local_v28_peptide_research_staging`，均为
  `local_frozen_research_only`，未解除 release license / active-service 限制。
* 只有在补齐原始、可合法使用的肽 / 环肽标签，完成去重、条件分层、3-seed OOF、
  source-held-out、OOD / UQ 和独立验证后，某个端点才可能从本研究面板晋级；本轮
  没有晋级任何端点。
