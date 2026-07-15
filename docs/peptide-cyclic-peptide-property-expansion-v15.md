# 肽/环肽性质数据 V15：PPB/Kp 原始证据、PK 语义分层与 assay-conditioned 可建模性审计

更新日期：2026-07-14

## 1. 本轮做了什么

V13 已给出全网数据源地图，V14 已完成第一轮原始 PK 增量、source-aware 基线和许可分层。V15 不重复这两份报告，只回答上一轮留下的四个具体问题：

1. 能否继续从原始论文补充 PPB 和严格 tissue/plasma Kp，而不把 bound/unbound、皮肤渗透、BBB proxy 或预测 Kp 混为一个 target？
2. F、CL、Vd 现有数据在拆分 absolute/apparent、subtype、normalization、analysis method 和 source evidence 后，究竟还剩多少严格监督目标？
3. T1/2 和 cytotoxicity 在 assay-conditioned task 下是否真的有可建模信号，还是只是总行数很大？
4. 这些变更能否继续保持科学 training eligibility 与公开/商业再分发许可相互独立？

结论是：

- V15 追加 52 行原始论文证据，包括 PPB 49 行和 Kp 3 行；46 行通过科学训练门，6 行保留在 review。
- PPB 的 QC 可用规模从 V14 的 94 行/75 identities 提高到 140 行/82 identities，但仍属小样本。
- Kp 增加了 3 个直接报告的 parent-specific tissue/plasma ratios，但因 ratio basis 和立体身份不完整，训练可用行仍为 0。
- F/CL/Vd 的 1,138 行记录中，只有 171 行/24 identities 通过严格 semantic task 门；967 行进入 review。
- T1/2 分层后有 2,851 行/593 tasks，但没有任何 task 达到建模门；V15 拒绝用重新混池换取一个表面上的 T1/2 模型。
- cytotoxicity 只有 5 个 task 实际建模，测试集 R2 全为负，所有任务都是描述性结果，没有任何可靠信号声明。
- 最终审计 `data/peptide_property_expansion_v15/audit_report_v15.json` 状态为 `PASS`，`failures=[]`。

## 2. V15 主表实测规模

### 2.1 与 V14 的可对账增量

| 指标 | V14 | V15 | 变化 |
|---|---:|---:|---:|
| observation/label rows | 273,397 | 273,449 | +52 |
| identity keys | 114,768 | 114,775 | +7 |
| exact observation keys | 262,130 | 262,182 | +52 |
| training-ready rows | 219,737 | 219,783 | +46 |
| 去重后 training observations | 215,522 | 215,568 | +46 |
| review rows | 53,660 | 53,666 | +6 |
| duplicate observation groups | 5,185 | 5,185 | 0 |
| core ADMET/PK rows | 35,585 | 35,637 | +52 |
| core ADMET/PK QC rows | 21,484 | 21,530 | +46 |

52 行 staging 现在是 52 个唯一 governed observation keys，46 个 training rows 也是 46 个唯一 governed training observations。这是在修正 liraglutide 的 concentration、donor 和 inner/outer ratio 上下文后得到的最终口径，不再把不同实验条件折叠到同一 observation key。

增量涉及 11 个 identity anchors，但必须再分层：

- 10 个 strict exact identity tuples，共49 行；
- 1 个 GE81112A stereo-ambiguous review anchor，共3 行；
- 所有 46 个训练行都具有 strict exact identity；
- GE81112A 的 PubChem/ChEBI identity 缺失立体层，而原文 Figure 1 报告了修订立体构型，所以不能把它写成严格 exact identity。

### 2.2 不能把 27.34 万行称为 27 万个 PK 实验

V15 仍然保留 V14 的 scope 体系：73 个规范化 raw endpoint strings、34 个 canonical endpoint families 和 7 个 scope classes。真正属于 `core_admet_pk` 的是 35,637 行，其中 21,530 行通过当前 QC。

其余大量记录是 toxicity/safety、therapeutic/function labels、supporting proxies 或 review text。因此正确描述仍是：**一个约 27.34 万行的肽/环肽性质、毒性和功能证据库，其中 core ADMET/PK 约 3.56 万行，当前 QC 可用约 2.15 万行。**

## 3. PPB/Kp 原始论文增量

### 3.1 来源、行数与许可

| compound/source | DOI | PPB | Kp | training | review | license |
|---|---|---:|---:|---:|---:|---|
| liraglutide | `10.1002/jps.23648` | 24 | 0 | 24 | 0 | CC BY-NC 2.5 |
| thanatin and analogs | `10.1126/sciadv.adg3683` | 12 | 0 | 12 | 0 | CC BY 4.0 |
| daptomycin（5 篇原始文献） | multiple | 9 | 0 | 8 | 1 | publisher copyright/PMC access |
| micafungin | `10.1002/bdd.752` | 2 | 0 | 2 | 0 | PMC author manuscript; no CC data license stated |
| semaglutide | `10.1016/j.molmet.2024.102006` | 2 | 0 | 0 | 2 | CC BY-NC-ND 4.0 |
| GE81112A | `10.1128/spectrum.02247-22` | 0 | 3 | 0 | 3 | CC BY 4.0 |
| **合计** |  | **49** | **3** | **46** | **6** |  |

每行保留 identity anchor、endpoint detail、value/relation/unit、species、matrix、assay、dose/timepoint、原文 table/figure locator、DOI/PMID/PMCID、license、evidence tier 和 QC flag。Thanatin 系列的 6 个身份不只指向 PPB 值所在的 Table S4A，Figure 2 的序列、非天然残基和二硫键身份证据也被纳入 direct build inputs 和行级 context。

### 3.2 PPB 方向和单位统一

PPB 最容易出现的错误是把 `% bound`、`% unbound` 和 fraction unbound 当成同一数值尺度。V15 的最终处理为：

- 46 个训练 PPB 全部规范到 `% bound`；
- micafungin 的原始 `fu=0.033` 和 `fu=0.004` 转为 `96.7 % bound` 和 `99.6 % bound`；
- 转换公式是 `percent_bound = (1 - fraction_unbound) * 100`；
- 原始 fu、转换公式、dispersion 和原文也报告 `% bound` 的事实都保留在 `context_json`；
- semaglutide 的 2 个 `% unbound` 仅保留在 review，不进入 `% bound` 回归任务；
- 1 个缺失 in-vivo sampling time 的 daptomycin 记录保留在 review。

因此，“PPB 新增 49 行”不再意味着可以盲目合并 49 个原始数值；真正的统一训练层是 46 个 canonical `% bound` targets。

### 3.3 CycPeptPPB 完整性审计

CycPeptPPB 是本轮必须单独检查的环肽 PPB 来源。结果不是“又找到了几百个可公开 target”，而是：

- Tajimi 公开原始表 16/16 行已在 V14 中逐行存在；sequence、HELM、value、unit、assay、DOI 和 license 对账通过；
- PeptiDream 347 条的结构和 PPB labels 受 NDA 限制，不是可公开训练 target；
- DrugBank external set 17 条是二手 compilation，不能再计为新原始实验；
- 因此 CycPeptPPB 本轮净新增原始公开 target 为 0，但完整性和重复证据已冻结到 manifest。

### 3.4 Kp 的严格含义

GE81112A 原文 Table 2 直接给出三个 parent-specific tissue/plasma ratios：

| tissue/plasma | value | unit | training status |
|---|---:|---|---|
| kidney/plasma | 3.00 | dimensionless | review-only |
| liver/plasma | 20.6 | dimensionless | review-only |
| lung/plasma | 0.515 | dimensionless | review-only |

这三行来自 IV 后 exploratory LC-MS/MS，它们不是 skin permeability、brain/plasma weak proxy、模型预测 Kp 或总放射性。但是原文没有明确说明 ratio basis 是 single-time、AUC-based 还是 equilibrium Kp，且 GE81112A 公开身份缺少完整立体层。因此它们只扩大了 review 证据层，没有扩大严格 Kp 训练集。

Kp 的禁止映射仍为：

- 皮肤 permeability 或 skin `logKp` 不是 PBPK tissue/plasma Kp；
- 没有 tissue/plasma partition 语义的 brain/blood 或 BBB proxy 不是 Kp；
- 模型预测 tissue coefficient 不是实验 target；
- 混合母体与代谢物的总放射性比值不是 parent-specific Kp。

最终 Kp 为 17 行/5 个 identity anchors，但 QC 可用仍为 0 行/0 identities。

## 4. 九个目标端点的 V15 实数

下表直接来自 `data/peptide_property_expansion_v15/target_endpoint_snapshot_v15.tsv`。

| 端点 | raw endpoints | rows | 数值 | 标签 | identity | QC rows | QC identity | QC unique obs | review | sources |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogD | LogD; LogD7.4; logd | 1,584 | 1,580 | 4 | 1,063 | 153 | 96 | 151 | 1,431 | 11 |
| solubility | solubility | 9,051 | 811 | 8,240 | 1,889 | 4,413 | 1,386 | 2,488 | 4,638 | 14 |
| F | F | 272 | 203 | 18 | 154 | 44 | 31 | 44 | 228 | 14 |
| T1/2 | T1/2; half_life; stability_half_life | 8,148 | 6,356 | 1,718 | 3,494 | 5,354 | 2,239 | 4,964 | 2,794 | 28 |
| PPB | PPB | 331 | 287 | 13 | 187 | 140 | 82 | 140 | 191 | 16 |
| CL | CL; clearance | 522 | 478 | 17 | 239 | 94 | 25 | 93 | 428 | 16 |
| Vd | Vd; volume_distribution | 344 | 330 | 3 | 168 | 59 | 12 | 58 | 285 | 12 |
| BBB | BBB; bbb_penetration | 2,407 | 578 | 1,829 | 1,081 | 982 | 980 | 980 | 1,425 | 7 |
| Kp | Kp | 17 | 17 | 0 | 5 | 0 | 0 | 0 | 17 | 3 |

V15 不改变以下基本判断：二分类端点当然可以用，但必须保留标签规则和来源；大量 T1/2 行不等于一个同质 T1/2 任务；CL/Vd 的原始 QC 行不能在不区分 parameter semantics 时直接随机切分训练。

## 5. F/CL/Vd 语义任务层

### 5.1 从 1,138 行到 171 行 strict targets

V15 没有对 F/CL/Vd 虚增新外部数据。`pk_semantic_tasks_v15.tsv` 和 `pk_semantic_review_v15.tsv` 是对冻结 V14 主表的治理投影：

| family | 原始选中 | strict training | review |
|---|---:|---:|---:|
| F | 272 | 20 | 252 |
| CL | 522 | 92 | 430 |
| Vd | 344 | 59 | 285 |
| **合计** | **1,138** | **171** | **967** |

171 行 strict layer 中有 162 个 numeric regression rows 和 9 个 label rows，覆盖 24 个 unique identities、13 个 source groups、25 个 semantic task IDs 和 9 个 global identity-source split groups。所有 strict rows 都属于 primary/direct evidence 类。

原先通过上游 QC 的 197 行中，26 行被更严格的 raw source-evidence gate 降级。一条记录即使 assay text 看起来像 NCA 或 LC-MS/MS，只要它的真实来源是 regulatory summary、compiled secondary、activity database、source-defined ML label 或 unverified source，就仍然必须留在 review。

### 5.2 不可混用规则

V15 将下列规则写入 machine-readable manifest：

1. absolute、relative 和 unspecified-basis F 不可合并。
2. F 必须保留 route 和 formulation；oral、SC、IP、IM、intranasal 不可互换。
3. systemic CL 与 microsomal/hepatocyte intrinsic CL 不可合并。
4. `CL`、`CL/F` 和 apparent clearance 不可合并，不得删除 `/F`。
5. Vss、Vz、Vc、Vp 和 unspecified Vd 必须是不同目标语义。
6. Vd/Vss/Vz 与 `V/F`/`Vz/F` 不可合并。
7. total、body-weight-normalized、protein-normalized 和 source-specific normalization 不可在没有显式转换时合并。
8. NCA、population PK、exposure ratio、in-vitro assay 和 regulatory/compiled summary 不是同一 analysis method。
9. parent/named active moiety 不得与 parent-plus-active-metabolite activity-equivalent estimate 合并。
10. 单任务切分必须按 task identity-source connected component；多任务实验必须使用 global group，不得按行随机切分。

identity grouping 也不再只依赖一个上游 `identity_key`：严格 InChIKey、HELM、canonical/stereochemical SMILES aliases 会被 union 成 connected component；sequence 只在“没有更强结构表示且是 canonical amino-acid sequence”的安全情况下参与联结，避免把相同线性字母但修饰/立体化学不同的分子错误合并。

## 6. T1/2 与 cytotoxicity 的 assay-conditioned 结果

### 6.1 数据形状

conditioned release 只读取冻结的 V14 training-ready 表，因为 V15 的新行是 PPB/Kp，与这两个任务无关。

| endpoint | 上游 endpoint rows seen | conditioned/deduplicated rows | task IDs | modeled tasks |
|---|---:|---:|---:|---:|
| T1/2 | 5,320 | 2,851 | 593 | 0 |
| cytotoxicity | 5,616 | 1,569 | 169 | 5 |
| **合计** |  | **4,420** | **762** | **5** |

T1/2 task ID 包含 assay family、species、matrix、route 和具体 assay/protease context。已明确排除 brain homogenate、contact-lens slow release、vitreous 和 local-retention 等不可比语义。Cytotoxicity task ID 包含 metric family、species、精确 cell context、exposure-context signature 和 source molar unit；nM 与 uM 即使最终都归一到 `log10(uM)`，也不会被静默地放入同一 task ID。

T1/2 的 2,851 行在严格分层后形成 593 个小任务，没有任何一个同时满足最少 75 行、60 identities、8 provenance groups、8 joint components、10 unique targets、20 non-dominant rows 以及 component/target dominance 上限。所以全部 593 个 task 被标记为 `NOT_MODELED`，不再使用混池 fallback。

### 6.2 五个 cytotoxicity 描述性基线

| task 简称 | rows | train/val/test | trained MAE | fit-median MAE | test R2 |
|---|---:|---:|---:|---:|---:|
| CC50 / HEK293T | 123 | 83/20/20 | 0.4578 | 0.4066 | -0.9746 |
| CC50 / HaCaT | 94 | 67/14/13 | 0.4957 | 0.4507 | -0.3638 |
| IC50 / HEK293 | 79 | 55/12/12 | 0.4488 | 0.4471 | -0.0062 |
| IC50 / HaCaT | 93 | 64/14/15 | 0.4615 | 0.4711 | -0.0685 |
| IC50 / HUVEC | 105 | 72/15/18 | 0.3980 | 0.4222 | -0.0218 |

这 5 个 task 共 494 行。虽然两个 IC50 task 的 MAE 比中位数常数基线略好，但五个 R2 全为负，test 只有 12--20 行，每个 task 的测试 identities 也低于 30，并且 exposure duration 大多数未报告。因此五个 task 的 conclusion gate 都是：

```text
descriptive_only_no_reliable_signal_claim
```

V15 不声称这五个 cytotoxicity 任务已学到可靠信号，也不声称 T1/2 已有可用 assay-conditioned 模型。

### 6.3 泄漏与表示门

- 每个已建模 task 内部使用 exact identity 与 normalized DOI/PMID provenance 的联合 connected components 进行 split。
- 5 个 task 各自的 identity crossing、provenance crossing 和 joint-component crossing 都是 0。
- 但 release-global audit 发现跨 task 的 11 个 identity crossings 和 4 个 provenance crossings。因此 `pass_for_multi_task_use=false`；禁止直接把五个 task 合并后沿用原 split。
- `N.A.`、`NA`、`None` 等 structure sentinel 只是缺失值，不能成为 `SMILES:N.A.` 这类模型输入。
- 最终 4,420 行 conditioned release 中 sentinel representation rows = 0，invalid representation rows = 0。
- exact identity split 仍不等于 homology/scaffold split；近同源序列或近邻 scaffold 仍可能跨 split。

## 7. 许可层与科学 eligibility 必须分开

### 7.1 V15 公开层实数

| 层 | rows | sources | 说明 |
|---|---:|---:|---|
| open redistribution | 160,398 | 19 | 明确 CC BY/CC BY-SA/CC0/public domain/NIST OA 且行级追溯通过 |
| public noncommercial | 163,161 | 29 | open 全集 + 明确 CC BY-NC/CC BY-NC-SA |
| noncommercial-only | 2,763 | - | 只出现在第二层 |
| 两层均排除 | 110,288 | - | 许可、身份、来源或上游权利未通过保守门 |

V15 新明确支持 `CC BY-NC 2.5` 许可家族；24 个 liraglutide 行进入 public noncommercial 层，不进入 open-commercial 层。公开层的全部质量门为 true：273,449 个输入行全部对账；包含行具有 identity、endpoint、evidence、source 和 attribution；全局排除行都有理由；36 条 ND 行全部排除。

### 7.2 四种组合都真实存在

- 科学 QC 通过且 open：例如部分 CC BY 的 thanatin PPB。
- 科学 QC 通过但仅 noncommercial：例如 CC BY-NC 2.5 的 liraglutide PPB。
- 科学 QC 通过但不在两个公开再分发层：例如无 CC 数据许可的 daptomycin/micafungin 原始数值。
- 许可 open 但科学上 review-only：例如 CC BY 4.0 的 GE81112A Kp，其 ratio basis 和 stereo identity 不足。

因此：

- `train_target_eligible` 是科学 target 适用性，不是再分发授权；
- `redistribution_cleared` 是许可/追溯视图，不是自动训练集；
- 代码仓库的 MIT/GPL 不能自动授权上游数据；
- 本项目没有对混合来源进行 blanket relicensing，每行继续保留原 license 和 attribution。

## 8. Verifier 发现的问题如何被修复

V15 的最终发布不是第一次构建结果的直接封存。独立 verifier 发现的问题已被修复并加入回归门：

1. **PPB bound/unbound 混用**：micafungin fu 已转为 canonical `% bound`；semaglutide `% unbound` 仅 review；所有 training PPB 方向一致。
2. **Liraglutide observation-key 折叠**：concentration、donor、inner/outer ratio 已进入 assay/dose 字段；52/52 governed observations 现在唯一。
3. **身份口径过度声明**：将 11 个 anchors 明确拆为 10 strict exact + 1 stereo-ambiguous review；Thanatin Figure 2 加入 direct identity evidence。
4. **`N.A.`/`NA` 被当作结构表示**：结构哨兵值在表示选择前被阻断；发布层的 sentinel/invalid representation 都为 0。
5. **T1/2 异质混池**：删除为了达到样本门而构造的 pooled fallback；593 个严格 task 全部如实标记 `NOT_MODELED`。
6. **source gate 被 assay/method 文本掩盖**：raw source/evidence class 现在优先；regulatory、compiled、activity-database、source-defined ML 和 unverified rows 必须 review。
7. **identity alias 未正确 union**：InChIKey、HELM 和结构 aliases 建立 connected components，sequence 只在安全情况参与联结。
8. **输出目录递归删除风险**：conditioned builder 现在会在任何 `rmtree` 之前验证 `--out-dir`，并拒绝 input、input ancestor、workspace 及 workspace ancestor，也拒绝已存在的普通文件路径。

这些修复不是“增加更多结论”，而是缩小可声称范围：Kp 仍为 0 个 strict targets，T1/2 仍为 0 个 modeled tasks，cytotoxicity 仍为 0 个 reliable-signal claims。

## 9. 全网检索回退链

本机已尝试 `agent-reach doctor --json` 和 `agent-reach check-update`，但主机上均为 `command not found`。V15 因此没有把搜索停留在一个不可用的工具上，而是回退到官方接口和原文证据：

- Europe PMC REST 和 PMC full text/supplements；
- PubMed BioC；
- Crossref；
- PubChem PUG REST；
- ChEBI 和 ChemSpider identity cross-checks。

31 份搜索结果已冻结；原文、SI、identity files 和直接 build inputs 均记录 path、bytes 和 SHA-256。搜索摘要、模型仓库 README 或二手数据库不能单独作为新 target 证据。

## 10. 质量控制与最终审计

`data/peptide_property_expansion_v15/audit_report_v15.json` 的最终状态为 `PASS`，`failures=[]`。主要验收包括：

- V14 的 273,397 行冻结记录被保留，只允许 `duplicate_group_size`、`eligible_after_qc`、`eligibility_reason` 三个字段按新全局重算；
- master 273,449 = V14 273,397 + V15 staging 52；
- training 219,783 + review 53,666 = master 273,449，两分区无重叠；
- deduplicated training 为 215,568，exact observation key 无重复；
- primary staging 为 52 = 46 training + 6 review，PPB 49 + Kp 3；
- PPB training 全部为 canonical `% bound`；Kp 全部 review-only；
- PK semantic partition 1,138 = 171 strict + 967 review，所有 quality gates 通过；
- conditioned inventory 762 = 5 modeled + 757 unmodeled，所有 T1/2 task 都未建模；
- per-task leakage gates 通过，release-global multi-task gate 如实为 false；
- redistribution 以 V15 主表为输入，open/noncommercial/excluded 和许可家族完整对账；
- pipeline、summary、primary、PK semantic、conditioned 和 redistribution 声明的 bytes/SHA-256 与磁盘一致。

核心 SHA-256：

| artifact | SHA-256 |
|---|---|
| V15 observations | `9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf` |
| pipeline manifest | `700a1b6dbdbb91f73d218b7c2a3820b6af00723594e91bc7e7b493aafdc12431` |
| primary PPB/Kp staging | `8df19cb424d30979cc47733428f4407053b74896984641fb4a58adc3c18dbb0c` |
| primary PPB/Kp manifest | `568f92b0453631cde38e97290d77c86116773716947e2d6f3347166a735e2edb` |
| PK semantic tasks | `06a33601d304e084d9af6691497dad466e85aabd22c66ddb265498a5733f4602` |
| PK semantic review | `da0fc6ada435fc057fc18c72ca1c78134887b6682666c9c971d51de349d3b706` |
| PK semantic manifest | `14aa0c14922aebe97a1544762706498eb79a653fa62e879c13a6bb7f77b96ae3` |
| assay-conditioned manifest | `8461335fce51b2435f9269cbbefbe24fdee3ba3c0dc4163ce2444d29b1e8e119` |
| redistribution manifest | `8f05ae87d1742f952b9ee41d4ee5e1395b014983609d14d524c267a0de008df4` |
| audit report | `08bc9d0ba16758c3d3fbfe98780dbd91609f41b6b27a5795f0230451ec610d12` |

## 11. 核心路径与复现顺序

### 11.1 机器可读交付

- 主表：`data/peptide_property_expansion_v15/peptide_property_observations_v15.tsv`
- 科学 QC 通过表：`data/peptide_property_expansion_v15/peptide_property_training_ready_v15.tsv`
- exact-observation 去重表：`data/peptide_property_expansion_v15/peptide_property_training_deduplicated_v15.tsv`
- review：`data/peptide_property_expansion_v15/peptide_property_review_v15.tsv`
- PPB/Kp staging：`data/peptide_property_expansion_v15/staging/primary_ppb_kp_v15.tsv`
- PPB/Kp manifest：`data/peptide_property_expansion_v15/manifests/primary_ppb_kp_v15_manifest.json`
- F/CL/Vd strict tasks：`data/peptide_property_expansion_v15/pk_semantic_tasks_v15.tsv`
- F/CL/Vd review：`data/peptide_property_expansion_v15/pk_semantic_review_v15.tsv`
- assay-conditioned release：`data/peptide_property_expansion_v15/assay_conditioned_baselines/`
- 九端点快照：`data/peptide_property_expansion_v15/target_endpoint_snapshot_v15.tsv`
- ML readiness：`data/peptide_property_expansion_v15/ml_readiness_v15.json`
- 许可公开层：`data/peptide_property_expansion_v15/redistribution_cleared/`
- 最终审计：`data/peptide_property_expansion_v15/audit_report_v15.json`

### 11.2 重建

```bash
python scripts/build_primary_ppb_kp_v15.py
python scripts/build_peptide_property_expansion_v15.py
python scripts/summarize_peptide_property_expansion_v15.py
python scripts/build_pk_semantic_tasks_v15.py

uv run --offline --python /usr/bin/python3 --with scikit-learn==1.7.2 \
  python scripts/run_assay_conditioned_baselines_v15.py \
  --input data/peptide_property_expansion_v14/peptide_property_training_ready_v14.tsv \
  --out-dir data/peptide_property_expansion_v15/assay_conditioned_baselines \
  --seed 20260715 --split-trials 5000 --max-features 50000 \
  --min-rows 75 --min-identities 60 --min-provenance-groups 8 \
  --min-joint-components 8 --min-unique-targets 10 \
  --min-non-dominant-rows 20 --max-joint-component-fraction 0.5 \
  --max-target-value-fraction 0.5

python scripts/build_redistribution_cleared_v15.py
python scripts/audit_peptide_property_expansion_v15.py \
  --expected-new-training-rows 46 \
  --expected-new-review-rows 6 \
  --output data/peptide_property_expansion_v15/audit_report_v15.json
```

### 11.3 回归测试与静态检查

```bash
python -m unittest -v \
  tests.test_build_primary_ppb_kp_v15 \
  tests.test_build_peptide_property_expansion_v15 \
  tests.test_summarize_peptide_property_expansion_v15 \
  tests.test_build_pk_semantic_tasks_v15 \
  tests.test_run_assay_conditioned_baselines_v15 \
  tests.test_build_redistribution_cleared_v15 \
  tests.test_audit_peptide_property_expansion_v15

uvx --offline ruff check \
  scripts/build_primary_ppb_kp_v15.py \
  scripts/build_peptide_property_expansion_v15.py \
  scripts/summarize_peptide_property_expansion_v15.py \
  scripts/build_pk_semantic_tasks_v15.py \
  scripts/run_assay_conditioned_baselines_v15.py \
  scripts/build_redistribution_cleared_v15.py \
  scripts/audit_peptide_property_expansion_v15.py \
  tests/test_*v15.py

python -m compileall scripts tests
```

## 12. 距离真正可用的机器学习数据还有多远

V15 的改进主要是“语义和审计变得更真实”，不是“所有稀缺端点都已经够用”。

- **PPB**：140 QC rows/82 identities 比 V14 有明显增长，且 46 个新 training rows 已统一为 `% bound`。但它仍是小样本，并且 species、matrix、assay concentration/time 必须进入任务上下文。
- **Kp**：17 行中 0 行 strict training target。下一步需要原文明确报告 single-time/AUC/equilibrium basis、parent-specific analyte、精确立体 identity 和 tissue/plasma matrix。
- **F/CL/Vd**：171 strict semantic rows 只覆盖 24 identities。这已能支持小样本、分层或迁移学习实验，不支持从零训练通用 PK 大模型，也不支持删除 `/F` 或 subtype 后做一个大回归任务。
- **T1/2**：2,851 conditioned rows 分裂成 593 tasks 且全部未达门，说明真正缺的是同 assay/species/matrix/route 下的标准化重复数据，不是一个更复杂的网络。
- **cytotoxicity**：5 个模型任务的 R2 全为负，需要更完整的 exposure duration、cell context、source-balanced 数据和更大外部验证集。
- **multi-task**：每任务 split-safe 不等于跨任务 split-safe。当前 global identity/provenance crossings 禁止直接联合训练，必须先重建 release-global connected-component assignment。
- **许可**：能做科学训练、能公开分发、能商业使用仍是三个不同问题。下一轮任何扩容仍必须继续保留这三道门。

因此，V15 的最终定位是：**PPB 有真实但仍小的扩容；Kp 仍是主要空缺；F/CL/Vd 首次有不可混用的严格 task 层；T1/2 和 cytotoxicity 的建模不足被显式暴露，而不是被总行数或一个混池模型遮盖。**
