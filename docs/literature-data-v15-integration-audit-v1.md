# 师妹“文献数据”与肽/环肽 V15 的整合审计 v1

## 1. 审计对象与边界

原始文件：

```text
D:\software\xwechat_files\wxid_rl2r3kqj232m22_e40a\msg\file\2026-07\文献数据.zip
```

冻结副本：

```text
data/external_downloads/literature_data_import_v1/archive/文献数据.zip
```

- ZIP 大小：54,071,989 bytes
- ZIP SHA-256：`8f8a38273dfa2b1187536a4c4d85fb386288edb231d71da14b0dad0af24bba56`
- 解压后：245 个文件，334,048,547 bytes
- 路径穿越：0；symlink：0
- 本轮没有执行归档中的 Python/notebook，没有加载 PT/pickle，没有解包 RAR/TGZ；NPZ 仅读取 NPY header、shape 和 dtype。

三条独立子代理检查分别负责：全文件/来源族盘点、清洗代码与许可审查、与 V15 的 identity/observation 重叠复核。主线程重新流式读取 V15 273,449 行，并用脚本固化了所有关键断言。

## 2. 总结结论

这不是一个“整包全新数据集”，而是以下内容的混合包：

1. 已经进入 V15 的数据库原表或旧版副本；
2. 同一原表的镜像目录；
3. train/val/test、特征张量、embedding、split index 等模型派生物；
4. 少量真正值得继续追溯的新增 review 候选；
5. 一个比“新增多少行”更重要的质量发现：现有 PEPlife2 上游 CSV 解析有字段错位污染。

当前可以直接追加到生产训练集的行数是 **0**。这不等于数据没有价值，而是新增价值主要位于 `review/source-recovery` 层；在逐行来源、身份和许可恢复前，不能把 review 数量宣传为 training-ready 数量。

## 3. 文件级重复

- SHA-256 完全相同的文件组：44 组
- 涉及文件：120 个
- 理论可消除的重复副本：126,795,560 bytes

最明显的重复包括：

- `膜通透性/CycPeptMP/` 与 `环肽膜渗透性/CycPeptMP/` 的整树镜像；
- CPMP 各 mmff/uff/ig 配置目录重复携带相同 Caco-2/PAMPA 原表；
- PEPlife 的 `Peplife 2.0 - Basic Search Results.xlsx` 与 `Below 60 Seconds-...xlsx` 内容重复；
- ESM-BBB-Pred 是 BBPpredict 850 行按标签重新拆分后的同一数据；
- CPP_pred_en、CPPCGM、MLCPP2、GraphCPP、PLM4CPP 之间存在大量重新切分或重新编码。

文件重复与 observation 重复必须分开。完整文件级证据见：

```text
data/literature_data_v15_integration_audit_v1/exact_file_duplicate_groups.tsv
```

## 4. 与 V15 的 observation 重叠

### 4.1 确定重复/已覆盖

| 来源族 | 归档规模 | V15 重叠结论 |
|---|---:|---|
| CycPeptMPDB PAMPA/Caco-2/MDCK/RRCK | 8,880 observations | 8,880/8,880 exact；禁止重复追加 |
| PEPlife2 | 4,500 source IDs | 4,455 IDs 已覆盖；其中 4,052 条单位归一后的数值严格相同 |
| HLP 10mer/16mer | 375 | 375/375 `sequence + seconds` 已在 PEPlife2；属于滑窗派生数据 |
| BBPpredict/ESM BBB benchmark | 850 行 | 844 个去冲突 unique `sequence+label` 已在 V15；两条 identity 正负冲突被正确排除 |
| B3Pdb positive | 269 行/265 unique sequences | 260 个 unique positive pair 已在 V15 |
| PepMSND | 635 行 | 23 条 exact identity + T1/2 + matrix 核心重叠，但本包缺逐行来源/许可 |
| THPdb | 852 行/239 ThPP IDs | 当前 V15 已有 82 observations/20 peptide IDs；按既有 V15 筛选口径和前序审计，本包未发现可直接追加记录 |

CycPeptMPDB 的匹配键为：

```text
source_record_id + assay subtype + exact HELM/SMILES + numeric assay-specific value
```

PEPlife2 必须区分两种口径：

- same source record ID：4,455；
- strict unit-normalized numeric duplicate：4,052。

不能把 4,455 全部称为“数值完全相同”，因为其余行包含 censoring、categorical 文本、旧解析异常或非数值端点。

行级证据：

```text
data/literature_data_v15_integration_audit_v1/exact_duplicates.tsv
data/literature_data_v15_integration_audit_v1/source_record_overlaps.tsv
```

### 4.2 最可信的少量新增 review 候选

1. PEPlife2 ID `4046`：84 h，cynomolgus monkey plasma，PMID `38486997`；身份含 Aib、D-serine、C18/linker 和 lactam bridge，只能 review。
2. B3Pdb 新 positive sequences：
   - `RLTRKRGLKLA`
   - `CDIFTNSRGKRA`
   - `YASPKSFRYPNGVLACT`
3. SATPdb same-identity/new-BBB-subfunction：
   - `satpdb18668` / `(Me2)Y-rF-Nle`
   - `satpdb23583` / `(Me2)Y-rFK`
   - `satpdb23712` / `(Me2)YPFF`
   - `satpdb25096` / `(Me2)Y-cit-FK`
   - `satpdb29019` / `(Me2)Y-r-Aba-G`

这些记录仍然需要来源和许可 gate，不是立即可训练的 9 行。

### 4.3 大规模但仍需 source recovery 的候选

#### PepMSND

635 行的结构层分解为：

- 23：identity/value/matrix 核心重叠；
- 133：同 identity、同 T1/2，但 matrix/context 不同或缺失；
- 179：同 identity、不同 T1/2，可能是新条件，也可能是冲突；
- 300：无精确 SMILES/sequence T1/2 identity 匹配。

其中后 140 余列是 RDKit/3D 描述符，不是实验 observation。主表只应暂存前 12 个原始字段。由于 `Dataset.xlsx` 没有逐行 DOI/PMID/source/license，除 23 条核心重叠外的 612 行都必须先做来源恢复；当前 0 行可直接 append。

#### BrainPeps

- 357 rows，337 exact unique sequence representations；
- 166 个 representation-aware potential review additions；
- 表内 publication objects 只有 author/title/year/journal/内部 pbid，不足以直接把整表标成 `BBB=1`。

BrainPeps 的语义可能包括进入、排出、载体、相互作用或仅被 BBB 文献讨论。必须回原论文抽取方向、实验类型和数值/标签。

#### GI stability

- 109 rows，65 peptide names；
- 有 gastric/intestinal、30 min/2 h remaining percentage 和分类标签；
- 没有 sequence、SMILES、HELM、InChIKey、逐行来源或许可。

这些百分比是指定时间点剩余量，不是 T1/2。第一步应做 name-to-structure/sequence mapping，而不是直接塞入半衰期回归任务。

## 5. 关键质量发现：PEPlife2 解析污染

新 Excel 与项目已冻结的官方 PEPlife2 JSON 一致，证明当前上游 CSV 的引号错位导致部分字段右移。

| ID | 官方值 | 当前状态 | 正确动作 |
|---|---|---|---|
| 4001 | 4 Days | V15 已存在，但 value/unit/context 错位 | 修复上游 parser 后重建 |
| 4042 | ~16.8 Hours | V15 已存在，但 value/unit/context 错位 | 修复上游 parser 后重建 |
| 4044 | ~20.5 Hours | V15 已存在，但 value/unit/context 错位 | 修复上游 parser 后重建 |
| 4046 | 84 Hours | 未进入 V15 | 复杂修饰 identity review 后 staging |
| 4047 | N.A. | 未进入 V15 | 正确排除，无数值端点 |

权威本地来源：

```text
data/external_downloads/peptide_admet_expansion_v4/peplife2/api_modified.json
```

修复清单：

```text
data/literature_data_v15_integration_audit_v1/peplife2_repair_candidates.tsv
```

应修复构建链的 PEPlife2 parser，然后全量重建和回归验证；不应直接手改 480 MB 的 V15 TSV。

## 6. 必须排除或隔离的内容

### CycPeptMP/CPMP/Multi

- train/val/test、ECFP `.pt`、NPY split index、RDKit 描述符不是新实验；
- CPMP 普通随机切分没有按 `Structurally_Unique_ID`/same-peptide/source 分组，存在 leakage；
- `-10` 是检测限/哨兵候选，不能无条件作为精确回归值；
- `MultiCycPermea` 覆盖了原始 Source 为 Train/Test/Validation，破坏来源语义。

### BBB negatives

- B3 的 `Negative_CPPs` 是“CPP 但非 B3PP”的构造负类，不等于实验 non-BBB；
- balanced/random negatives 是随机负例；
- `VSRRRRRRGGRRRR` 等出现正负冲突；
- 构造负类可以保留为 weak benchmark，但不能与实验阴性同质量混合。

### CPP benchmark

- 它们是 `cell_penetrating_peptide_binary`，不是 PAMPA/Caco-2/MDCK/RRCK 数值通透率；
- CPPCGM Set2 与 CPP_pred_en/CPP924 对同 916 条序列的标签全部反转；
- PLM4CPP Non-CPP 文件中有 19 行 `label=1`；
- GraphCPP 有跨 split overlap；
- embedding CSV 多数只是 Git-LFS pointer。

必须作为单独 benchmark 家族处理，先恢复论文中 `1` 的定义和 negative generation method。

### pepBERT/CamSol-PTM

- pepBERT NPZ 是 token/feature matrix，没有 tokenizer、原始序列、来源和许可，且有正负 encoding collision；
- CamSol-PTM 补表是合成、纯度和色谱条件，不含溶解度 label/value。

两者都不能生成新的溶解度 observation。

## 7. 清洗与整合规则

推荐清洗键：

```text
exact identity
+ endpoint family/subtype
+ assay/matrix/species/route/in-vivo-vs-vitro/timepoint
+ source record/PMID/DOI
+ relation/value/unit
```

执行顺序：

1. **Artifact gate**：先判定 raw、mirror、split、feature、embedding、source document；派生物不产生 observation。
2. **Identity gate**：结构型修饰肽优先 exact SMILES/HELM/InChIKey；裸 sequence 不能覆盖端基、D/L、非天然残基、环化和 linker。
3. **Endpoint gate**：分开 T1/2、指定时间点 remaining、BBB binary、CPP binary 和物理 permeability；禁止跨语义合并。
4. **Context gate**：保留 assay、matrix、species、route、dose、timepoint、in-vivo/in-vitro；同 identity 不同条件通常是 additive，不是 duplicate。
5. **Censoring gate**：`<`、`>`、`~`、detection limit 和 sentinel 要保留 relation/bounds，不能转成等号点值。
6. **Provenance gate**：每行至少有合法 source record 或 PMID/DOI；模型仓库名不是实验来源。
7. **License gate**：整包没有 LICENSE/COPYING/NOTICE。除项目独立确认的 THPdb Figshare CC BY 4.0 外，其余维持 `NOASSERTION/internal-only`。
8. **Split gate**：按 source/identity cluster 分组切分，避免 same peptide、同 scaffold、同 paper 跨 train/test。

## 8. 本轮生成的可复现产物

```text
scripts/audit_literature_data_v15_integration.py
tests/test_audit_literature_data_v15_integration.py
data/literature_data_v15_integration_audit_v1/README.md
data/literature_data_v15_integration_audit_v1/audit_manifest.json
data/literature_data_v15_integration_audit_v1/file_inventory.tsv
data/literature_data_v15_integration_audit_v1/exact_file_duplicate_groups.tsv
data/literature_data_v15_integration_audit_v1/source_family_catalog.tsv
data/literature_data_v15_integration_audit_v1/normalized_candidates.tsv
data/literature_data_v15_integration_audit_v1/exact_duplicates.tsv
data/literature_data_v15_integration_audit_v1/source_record_overlaps.tsv
data/literature_data_v15_integration_audit_v1/identity_overlap_additive.tsv
data/literature_data_v15_integration_audit_v1/conflict_review.tsv
data/literature_data_v15_integration_audit_v1/excluded_derivatives.tsv
data/literature_data_v15_integration_audit_v1/peplife2_repair_candidates.tsv
data/literature_data_v15_integration_audit_v1/safe_binary_header_inventory.tsv
```

`identity_overlap_additive.tsv` 是 `conflict_review.tsv` 中较有叠加价值的 review 子集，不是互斥分区；142 条 additive review 与 1,156 条 conflict/review 不能相加。

V15 审计前后 SHA-256 均为：

```text
9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf
```

因此本轮确实是 staging/audit，没有静默改动当前 V15 主表。

## 9. 推荐下一步

优先级不是继续把模型仓库的 split/feature 全部堆进主表，而是：

1. 修复 PEPlife2 parser，重建 V15，并确认 4001/4042/4044 的值语义恢复、4046 进入 review；
2. 对 PepMSND 612 条 review 候选做 PMID/DOI/source recovery，优先处理 179 条同 identity 不同值；
3. 对 BrainPeps 166 条 potential additions 回原论文抽取 BBB 方向/数值；
4. 单独建立 BBB/CPP weak benchmark release，显式标记 constructed/random negatives，不与物理通透率混合；
5. 解决 CPPSet2 标签反转和 PLM4CPP 19 条目录/标签异常后，再讨论 benchmark 训练使用。
