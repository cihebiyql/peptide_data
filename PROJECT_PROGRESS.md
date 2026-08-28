# 中检院项目进展总账

> 状态基准：2026-08-28 10:09（Asia/Shanghai）
> 仓库证据最新日期：2026-07-23；当前模型和训练数据冻结日期：2026-07-22
> 维护定位：本文件是项目级唯一可变进度总账；日期化报告、JSON manifest 和 TSV 结果作为冻结证据，不在本文件中重复维护。
> 公开镜像说明：本仓库只发布公开安全子集；总账中部分内部证据路径用于说明审计来源，并不随公开仓库分发。
> 科研边界：当前所有模型均仅供内部科研，不可用于临床、监管或生产决策。

## 1. 一页结论

项目原始需求包含四条业务线，但当前完成度并不均衡：

| 业务线 | 当前证据状态 | 已落地 | 主要缺口 |
| --- | --- | --- | --- |
| 病理切片正常/病变识别 | 未开始；仅有需求 | 需求字段和输入/输出描述 | 没有专用图像数据集、预处理流水线、训练代码、权重、Grad-CAM、API 或国产化测试报告 |
| 亚硝胺杂质致突变风险 | 未开始；仅有需求 | Ames/mutagenicity 可作为未来端点语义参考 | 没有亚硝胺专用数据集、PCA/决策树或 QSAR 模型、训练和外部测试证据 |
| 多肽/环肽 ADMET 与 PBPK 接口 | 开发验证阶段；当前主线 | V31 已完成 18 个端点的冻结数据、分组 OOF、最终模型、动态推理和 Web 原型 | 18/18 均未完成独立外部验证；修饰表示、校准、适用域、PBPK 最终参数表仍不完整 |
| 药物-靶点相互作用与不确定性 | 数据准备阶段 | 已有 ChEMBL/BindingDB/序列及候选 PBPK 数据链 | 没有独立 DTI 模型包、联合 2D/3D/序列架构、UQ 校准、Top-K 或外部独立测试 |

当前唯一可作为“已实际训练并可本地推理”的主模型是：

- 版本：Peptide-OmniPanel V31 peptide18。
- 有效模型包：`models/peptide_omnipanel_v31/release_peptide18_a3_20260722/`。
- 输入：一条由 20 种规范氨基酸组成、长度 1-1,024 aa 的序列；Web 批量层最多 32 条。
- 输出：固定 18 个输入依赖预测值。
- 工程状态：`bundle verified / 18 dynamic / 0 independently validated`。
- 运行状态：截至本总账观察时间，没有项目训练、推理或 Gradio 服务进程；最后的项目日志为 2026-07-22。2026-07-23 的公网部署只应视为历史验证，不能视为当前在线状态。

## 2. 权威证据与版本关系

| 层级 | 当前权威资产 | 用途 |
| --- | --- | --- |
| 原始需求 | `docs/需求描述示例(1)-extracted.md` | 四条业务线的范围基线，不代表实际进度 |
| 全量肽证据库 | `data/peptide_property_expansion_v15/`；`docs/peptide-cyclic-peptide-property-expansion-v15.md` | 多来源收集、清洗、许可和训练门的上游权威 |
| ML 清洗层 | `data/peptide_ml_cleaning_v1/manifest.json`；`docs/peptide-ml-cleaning-v1.md` | 面向建模的代表记录、任务语义、数值/二分类/PU 分层 |
| V31 冻结训练数据 | `data/peptide_omnipanel_v31_peptide18_training_cases_20260722_a2/` | 当前模型实际绑定的数据；无 `_a2` 的同名目录已被替代 |
| V31 端点合同 | `configs/peptide_omnipanel_v31_peptide18_endpoint_registry.json` | 18 个端点的定义、单位、条件和证据层 |
| V31 当前模型 | `models/peptide_omnipanel_v31/release_peptide18_a3_20260722/` | 18 个模型头、OOF、manifest 和 SHA 合同；早期 a2 包已作废 |
| V31 技术总述 | `docs/peptide_omnipanel_v31_latest_model_architecture_training_results_20260723.md` | 当前架构、数据、训练、指标和边界 |
| V31 冻结验证 | `docs/peptide_omnipanel_v31_peptide18_verification.json` | 工程完整性、动态性、重复性及 `validated_model_count=0` |
| 项目文档导航 | `docs/README.md` | Current / Frozen / Superseded / Requirements 分类索引 |

版本演进应按以下口径理解：

```text
V15 全量数据与 ML-cleaning v1
        -> V28 39 端点研究显示层（含迁移模型和透明先验）
        -> V29 ESM2-8M 严格留出复验（0 个模型晋级）
        -> V30 ESM2-650M 搜索（因大量小分子 teacher 迁移而 scope audit 失败并隔离）
        -> V31 a2 仅肽 18 端点（发现 cell-penetration 模态/来源泄漏并作废）
        -> V31 a3 当前有效包（18 个动态模型，0 个独立外验）
```

## 3. 数据集收集与处理

### 3.1 三个不能混用的数据口径

| 数据层 | 数量 | 含义 | 是否直接用于 V31 |
| --- | ---: | --- | --- |
| V15 全量主表 | 273,449 行 | 多来源原始/标准化证据库 | 间接；先经过清洗和任务筛选 |
| V15 training-ready | 219,783 行 | 通过该阶段训练门的记录 | 否，仍不是 V31 cases |
| V15 exact-observation 去重层 | 215,568 行 | 去重后的上游训练记录 | 间接 |
| V15 review | 53,666 行 | 需要复核或不可直接训练的记录 | 否 |
| ML-cleaning 目标范围 | 20,052 行 | 从 V15 去重层筛出的当前任务范围 | 间接 |
| ML-cleaning representative observations | 19,979 条 | 语义拆分和代表记录层 | 部分进入后续汇编 |
| V31 endpoint-molecule-condition cases | 30,507 条 | 端点、分子、实验条件级训练案例；同一肽可有多条 | 是 |
| V31 unique canonical sequences | 17,444 条 | V31 唯一规范序列 | 是 |

因此，273,449、19,979、30,507 和 17,444 不是互相矛盾的“数据集大小”，也不能相加；它们分别描述上游证据、清洗代表记录、训练案例和唯一序列。

### 3.2 上游来源与收集结果

V15 已汇入的主要来源包括：

| 来源 | V15 行数 | 主要角色 |
| --- | ---: | --- |
| TPDB | 145,738 | 肽数据库主来源 |
| DBAASP | 22,686 | 抗菌/毒性/溶血等肽证据 |
| Hemolytik2 | 15,858 | 溶血证据 |
| ToxinPred3 | 11,036 | 弱毒性 benchmark |
| DRAMP | 10,377 | 抗菌肽证据 |
| CycPeptMPDB | 8,880 | 环肽和渗透性等证据 |
| PEPlife2 | 4,455 | 稳定性/T1/2 相关证据 |
| SolPepBench/PepSol2000 | 4,088 | 溶解度相关证据 |
| HemoPI2 | 3,852 | 溶血相关证据 |
| ChEMBL peptide extractor | 2,834 | 肽结构和活性补充 |

V15 还完成了 PPB 49 行、Kp 3 行的论文增量抽取；46 行通过训练门、6 行进入复核。收集层保留 identity、assay、dose/timepoint、DOI/PMID/PMCID、license、evidence tier 和 QC。

许可仍是独立阻塞项：ML-cleaning manifest 中 `provenance_license_gate_completed=false`。DBAASP、PEPlife2、CycPeptMPDB 和 ToxinPred3 等来源不能被默认视为允许整体公开或商业再分发；当前训练可用不等于可发布。

### 3.3 清洗、语义拆分与质量控制

ML-cleaning v1 的 19,979 条代表记录分为：

- strict numeric：11,591；只保留可解释的精确数值目标。
- binary evidence catalog：3,930；保留二分类语义和证据来源。
- positive-unlabeled：920；未标注背景不被错误当作确定负类。
- review：3,308；保留冲突、删失、表示或证据不足记录。

主要处理步骤：

1. 将宽泛来源记录拆成端点和条件明确的 observation。
2. 统一单位、方向和删失值表达；删失值不自动当作 exact point。
3. 生成安全 identity、表示可用性和 exact-observation 去重键。
4. 区分严格数值、来源定义二分类、直接阈值二分类、弱标签和 PU。
5. 保留 review/recovery 队列，不通过删除掩盖冲突。
6. 在训练阶段按 `validation_group` 分组，不做普通随机行切分。

ML-cleaning manifest 的 26 项断言均为 `true`，包括行守恒、严格数值只含 exact、修饰肽不回退为裸序列、弱二分类不进入 strict model 等。

### 3.4 V31 冻结训练集

V31 汇编器只读取六组冻结输入：strict numeric、binary evidence、PU、V28 肽域 observations、PEPlife2 T1/2 和 hemolysis pilot。最终 `_a2` 训练集为：

| 项目 | 数量 |
| --- | ---: |
| 端点 | 18 |
| endpoint-molecule-condition cases | 30,507 |
| 唯一规范序列 | 17,444 |
| 分类案例 | 17,290 |
| 回归案例 | 9,433 |
| PU 案例 | 3,784 |
| 直接测量/数值阈值案例 | 12,363 |
| 弱标签/来源定义/PU 案例 | 18,144 |
| 可计算 sequence/ESM 行 | 23,410 |
| 有 HELM 行 | 12,265 |
| 非空 SMILES 行 | 7,387 |
| RDKit 可解析行 | 7,060 |
| small-molecule teacher 行 | 0 |

V31 的 fail-closed 质量门会拒绝端点顺序不一致、重复 case ID、非肽 scope、禁止来源、非有限 target 和逐端点行数不符；TDC 小分子、teacher distillation、transparent prior、V28 弱回归和 V30 隔离资产均不得进入当前训练集。数据、summary、序列清单和上游输入均由 SHA-256 绑定，并通过 pending 目录原子发布。

### 3.5 与当前 V31 分离的数据资产

仓库还有面向更宽业务需求的跨库候选链，但它们没有进入 V31 的 18 端点训练：

- ChEMBL activity：31,587 行；compound 18,978；assay 17,623。
- BindingDB Ki/Kd/IC50/EC50：905,546 行，覆盖 2,796 个 accession。
- ADMET/PBPK harmonized candidates：82,713 行。
- PBPK manual review queue：5,744 行，其中 high priority 4,581。
- review-ready PBPK 参数候选：2,636 行、336 个化合物。
- B/P 与组织 Kp 的项目级标准化落地仍为 0 行；最终人工复核 PBPK 参数表尚未完成。

这些资产支持后续 DTI/PBPK 工作，但候选、文本抽取或 identifier bridge 不能被表述为训练真值或已完成模型。

## 4. 模型搭建与训练

### 4.1 当前架构

V31 不是一个共享多任务深度网络，而是共享冻结表示与 18 个端点独立树模型头：

```text
规范肽序列
  -> 传统肽特征 93D
  -> 冻结 ESM2-650M residue mean 1,280D
  -> HELM/SMILES 字符特征 64D + 64D
  -> Morgan 256D + RDKit 2D 10D
  -> 模态可用性 5D
  -> 固定宽 1,772D 特征
  -> 端点级路线选择
  -> 7 个分类/PU 头 + 11 个回归头
```

四条候选路线为 `traditional` 93D、`esm_sequence` 1,373D、`structure` 399D 和 `multimodal` 1,772D；最终分别有 1、8、6、3 个端点入选。模型家族为 6 个 LightGBM classifier、2 个 LightGBM regressor、1 个 ExtraTrees classifier 和 9 个 ExtraTrees regressor。

ESM 主干固定为 `facebook/esm2_t33_650M_UR50D` revision `08e4846...`，完全冻结、mean-residue pooling，未做 LoRA、adapter 或端到端微调。17,444 条序列的缓存已在 RTX 5080/BF16 上完成，耗时 92.59 秒，无 OOM fallback。18 个最终树头合计约 9.99 MiB；分组 OOF、路线选择和最终重拟合耗时 153.62 秒。

### 4.2 训练和选模合同

- 固定 seed：`20260722`。
- 分类/PU：`StratifiedGroupKFold`；回归：`GroupKFold`。
- 当前每个端点均为 3 折；优先使用 `validation_group`，缺失时回退 `exact_identity_group`。
- 分类以开发 OOF ROC-AUC 最大选路线；回归以开发 OOF MAE 最小选路线。
- 选路后在该端点全部训练案例上重拟合最终头，OOF 单独保留。
- cell penetration 只比较 sequence 路线，防止“正样本有 HELM、未标注背景无 HELM”的模态可用性泄漏。
- 分类阈值固定 0.5，尚未做独立阈值优化或概率校准。
- 回归 q50/q90 只是开发 OOF 绝对误差分位数，不是覆盖率验证后的置信区间。

当前训练脚本、端点 registry 和 bundle manifest 已落盘；但 seed、选择阈值和主要超参数仍主要硬编码在训练脚本中，尚无独立的 V31 机器可读训练预注册配置。

### 4.3 当前开发指标

以下均为开发分组 OOF，不是独立外部测试：

| 端点 | n | 路线 | 主要指标 | 当前解读 |
| --- | ---: | --- | --- | --- |
| hemolysis | 2,012 | multimodal | ROC-AUC 0.851；AP 0.739 | 有开发信号；无独立外验 |
| PAMPA permeability | 6,814 | structure | MAE 0.325；R2 0.676 | 单一 PAMPA 任务；不可外推 Caco-2/MDCK |
| HC50 | 1,926 | esm_sequence | MAE 0.337；R2 0.509 | 单来源，跨来源泛化未知 |
| intestinal stability | 393 | structure | MAE 0.422；R2 0.814 | 主要 cohort 主导，需 cohort-held-out |
| T1/2 >= 1 h | 918 | esm_sequence | ROC-AUC 0.685；AP 0.588 | 有一定区分力；未外验 |
| solubility | 2,168 | esm_sequence | ROC-AUC 0.604；Bal.Acc 0.506 | 阈值表现接近随机，只宜粗筛 |
| F | 12 | esm_sequence | R2 0.024 | 仅 2 条唯一序列，极低证据 |
| CL | 19 | esm_sequence | R2 -0.078 | 未优于均值基线 |
| Vd | 15 | structure | R2 -2.610 | 明显不可作定量决策 |
| membrane retention | 12 | structure | R2 -0.050 | 单系列、极低证据 |
| mouse plasma stability | 17 | structure | R2 0.010 | 样本和泛化均不足 |

“入选路线”只表示同一开发 OOF 中相对最优，不表示外部验证冠军。

### 4.4 已实现与未实现

已实现：

- 18 端点冻结 registry、数据集和 SHA 合同。
- ESM2-650M 缓存与 1,772D multimodal feature cache。
- 18 端点分组 OOF、路线选择、全量重拟合和模型落盘。
- 单序列 CLI、批量 Web 原型、结果导出和浏览器隔离历史记录。
- 输入依赖、有限值、确定性重复和无 exact-overlap probe 验证。

未实现或未完成：

- 端到端 ESM2 微调、LoRA/adapter、1D-CNN/BiGRU/token mixer。
- GNN/MPNN、真实 3D 构象网络和共享多任务 trunk 的同切分对照。
- 真实 HELM/SMILES 用户输入；裸序列仍无法无损表达环化、D-AA、端基和侧链修饰。
- species/assay/matrix 条件化、独立校准、conformal coverage、OOD/适用域和拒识。
- 临床、监管或生产级发布。

## 5. 测试、验证与外部评估

### 5.1 当前验证矩阵

| 验证层 | 当前状态 | 证据和边界 |
| --- | --- | --- |
| 数据清洗断言 | 通过 | ML-cleaning manifest 26 项断言为 `true` |
| V31 数据 fail-closed 审计 | 通过 | 禁止来源 0；small-molecule teacher 0；数据和上游 SHA 绑定 |
| V31 bundle 工程验证 | 通过 | 18/18 模型及 OOF 存在，SHA 和训练行数可核对 |
| V31 动态性 probe | 通过 | 5 条 probe、无训练集 exact overlap、18/18 有限/可重复/随输入变化 |
| 2026-08-28 V31 定向测试 | 通过 | `19 passed, 59 warnings in 28.08s`；使用隔离、frozen lock 环境复跑三个 V31 测试文件 |
| 2026-08-28 全套 `tests/` | 未全绿 | `707 passed, 6 failed, 1 skipped, 590 warnings, 121 subtests passed`，耗时 289.22 秒；见下方失败分类 |
| 独立外部验证 | 未完成 | V31 `validated_model_count=0`，18/18 均为 `not_validated` |
| 校准/UQ/OOD | 未完成 | 没有独立 calibration、conformal coverage、OOD 或拒识验证 |
| Web 服务 E2E | 历史通过，当前离线 | 2026-07-23 曾做公网 Chrome E2E；当前没有项目服务进程 |

本次全套测试通过隔离、frozen lock 环境运行。测试导入了项目未在依赖中声明的 `openpyxl`，因此验证命令临时使用了 `--with openpyxl`；仓库仍有 139 个 `test_*.py`、约 712 个测试函数，但没有 coverage 配置或阈值，不能用数量替代覆盖率。6 个失败分为：

1. V11 溶血外验的 clean rebuild 未做到字节级 SHA 重现。
2. V15 PPB/Kp manifest 在 Windows 生成反斜杠路径，与测试冻结的 POSIX 路径不一致。
3. V25 node1 launcher 的 `bash -n` 在当前 Windows 调用环境返回 127。
4. V25 preregistration execution fingerprint 与当前 Python 3.12/源码树环境不一致。
5. V30 A7 与 A9 preflight 都绑定 Linux `/root/.cache/...` ESM snapshot，当前 Windows 环境找不到该路径。

因此，“当前 V31 定向合同通过”和“全仓历史测试全绿”必须分开表述。后五项主要暴露跨平台/冻结环境绑定，第一项是实际重建可复现性失败；本轮只做进展审计和文档维护，没有擅自修改历史实验代码或冻结合同。

warning 中有 489 条来自 joblib/NumPy 2.5 兼容性弃用提示；V31 Web 另有 Gradio 6.0 前的 `Blocks(theme/css/js)`、`show_api/api_name` 等迁移提示。当前不影响 V31 定向测试通过，但升级依赖前必须处理。

V31 测试会检查 registry 恰好 18 端点、禁止小分子 teacher、bundle/OOF SHA、训练行数、`not_validated`、probe 无 exact overlap、动态有限输出、输入解析、32 条上限、导出和历史隔离。需要注意：部分 artifact 测试在本地发布物缺失时会 `pytest.skip`，未来 CI 需要显式要求 release artifacts，避免“绿色但未验证包”。

### 5.2 当前切分和泄漏边界

- 30,507 是案例数，不是独立肽数；使用 `validation_group` 分组比随机行切分更严格。
- 路线选择与指标仍来自同一开发 OOF；尚无系统 source-held-out、analogue-series-held-out 或 cohort-held-out。
- 5 条 probe 只能排除 exact overlap，不能排除近同源或 analogue-series 泄漏。
- cell penetration 的早期 a2 包曾有模态可用性泄漏，现已作废；V30 曾混入大量小分子 teacher，已整体隔离。
- PU 中的未标注样本不是确定负类，因此 cell penetration 的 AUC/AP 不能解释为校准临床概率。

### 5.3 历史端点级外验

以下历史结果可以作为方法和风险证据，但不能直接把 V31 的 `validated_model_count` 从 0 提升：

- V11 溶血独立诊断集：91 行、20 个同源簇；冻结 V10 AAC 基线 row AUROC 0.8395、cluster AUROC 0.8788。只有 9 个阳性簇，且 79/91 阴性来自一个系列；exact notation 和当时 ESM2 接近随机。
- V26 Helicogenic：22 observations、20 components，component-equal MAE 0.3977；仅一个有效来源，缺第二来源和正式校准。
- V28 Phase2：6 个渗透性任务均未晋级；多个 source-held-out 结果退化或缺第二来源。
- V29 PEP-FUSE：严格留出实验完成，但 component-bootstrap 95% CI 跨 0，0 个模型晋级。

## 6. 后续工作与验收门

### P0：把开发模型推进到可信外验

1. 冻结每个端点的 source-held-out、analogue/cohort-held-out 和独立论文测试集，测试集不得参与路线选择或阈值调整。
2. 优先验证 hemolysis、PAMPA、HC50、intestinal stability 和 T1/2；分别报告 source/cohort 构成、bootstrap CI 和失败案例。
3. 对 F、CL、Vd、LogD、PPB、human/mouse plasma stability 补充结构明确、条件一致、来源可追溯的新肽数据；样本门未达到前不升级为定量决策模型。
4. 完成逐来源许可/再分发审计；在 `provenance_license_gate_completed=true` 前不发布受限原始行或由其训练的可再分发包。

验收门：每个拟晋级端点必须有独立外部集、预注册指标、source/analogue 隔离、置信区间和明确的失败门；否则继续保持 `not_validated`。

### P1：校准、不确定性和适用域

1. 分类端点在独立校准集上评估 Brier、ECE、calibration curve 和阈值稳定性。
2. 回归端点单独验证 conformal/quantile coverage，禁止把开发 OOF q90 当作置信区间。
3. 建立 sequence/structure 近邻、source/OOD 标记和拒识策略；输出适用域而不仅是点值。
4. 对 PU 任务使用与 PU 语义一致的 ranking/positive-unlabeled 评价，不包装为标准二分类概率。

### P2：输入与架构补全

1. 增加真实 HELM/SMILES 输入和修饰合同，区分裸序列假想线性肽与真实环化/修饰分子。
2. 对条件敏感端点显式加入 species、assay、matrix、route、dose/timepoint。
3. 在完全相同的 group/source-held-out 切分下比较 LoRA/adapter、sequence mixer、MPNN/GNN 或 3D 表征；没有外验增益则不晋级。
4. 将 V31 seed、超参数、选择门和输入 SHA 迁入独立机器可读预注册配置。

### P3：补齐其他业务线

1. 病理线先建立数据授权、器官/物种/染色/扫描仪分层和患者级切分，再决定模型和国产化测试方案。
2. 亚硝胺线建立专用结构与 Ames/致突变数据集、适用域及外部化学系列切分，不直接复用肽模型。
3. DTI/UQ 线从现有 ChEMBL/BindingDB 候选链建立严格 compound/target/scaffold/sequence-family split，再训练独立模型；候选表不能直接算模型完成。
4. PBPK 先完成人工复核参数表、B/P 和 Kp，再定义参数 API 和敏感性传播；V31 弱预测不能替代 PBPK 真值。

## 7. 文档维护规则

1. 只在本文件更新“当前状态、风险、下一步和验证时间”；日期化报告及 JSON/TSV manifest 一旦冻结不回写。
2. 每次数据冻结必须记录资产目录、数据层含义、行数/唯一实体数、来源、许可状态和 SHA；不得只写一个总行数。
3. 每次训练必须记录数据版本、模型包、seed、split、指标性质（OOF/source-held-out/external）、晋级门和作废版本。
4. 每次测试必须记录观察日期、精确命令范围、通过/跳过/警告/失败数；历史测试不自动代表当前环境。
5. `.publish/` 是生成的发布镜像，只维护源文档；`.omx/` 是运行时状态，不能作为对外当前事实。
6. `release`、`pass` 和“18 个动态模型”必须与 `internal_research_only`、`validated_model_count=0` 同时展示。
7. 旧 V28/V29/V30 和 V31 a2 资料通过 `docs/README.md` 标为历史或 superseded，不删除，以保留决策和失败证据。

## 8. 更新记录

| 日期 | 维护内容 | 验证 |
| --- | --- | --- |
| 2026-08-28 | 建立项目级总账；按四条业务线、三层数据口径、V31 当前包、测试矩阵和后续验收门汇总；建立根入口与文档索引 | 交叉核对 manifest、模型包、报告、脚本和当前进程；V31 定向 19 passed；全套 tests 707 passed / 6 failed / 1 skipped |
