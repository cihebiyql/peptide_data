# `mozi-main.zip` 与肽/环肽性质 V15 的整合审计

更新日期：2026-07-14

## 1. 审计目标

本轮检查用户提供的：

```text
C:\Users\ciheb\Downloads\mozi-main.zip
```

目标是回答：

1. 文件中是否存在序列、结构、ADMET/PBPK 端点、数值/标签和来源；
2. 哪些记录与 `data/peptide_property_expansion_v15/` 重复；
3. 哪些记录是同一 identity 的新端点或新实验条件，可以叠加；
4. 哪些记录需要清洗或人工 review；
5. 是否可以安全、合法地并入当前 V15。

三个只读子代理分别检查了文件与数据格式、项目与清洗代码、以及与 V15 的字段和端点重叠。首次子代理调用超时后，任务被收窄并重试；三方最终结论一致，并由本地确定性审计脚本重新验证。

## 2. 原包冻结与安全检查

| 项目 | 结果 |
|---|---|
| 原路径 | `C:\Users\ciheb\Downloads\mozi-main.zip` |
| WSL 路径 | `/mnt/c/Users/ciheb/Downloads/mozi-main.zip` |
| 大小 | 24,452,328 bytes |
| SHA-256 | `e852570ee1268be5d8dfa14708fa7bf15f5b45d9331c12e7f8a65f89d13f915f` |
| ZIP entries | 392 |
| 文件 | 302 |
| 解压后文件 bytes | 58,483,607 |
| 路径穿越 | 0；解压器显式拒绝 |
| 符号链接 | 0；解压器显式拒绝 |

冻结副本和归档清单位于：

- `data/external_downloads/mozi_import_review_v1/archive/mozi-main.zip`
- `data/external_downloads/mozi_import_review_v1/archive_manifest.json`
- `data/external_downloads/mozi_import_review_v1/extracted/mozi-main/`

归档含 3 个 Windows EXE。审计过程没有执行任何 EXE、脚本、安装命令或项目依赖。

## 3. 这个 ZIP 实际是什么

根 `README.md` 将项目描述为 Mozi/Miaobi 多角色协作写作引擎：由 orchestrator 调度研究员、主笔、编辑等角色，生成文章、选题报告和视频脚本。技术栈是 Turborepo/pnpm、Fastify、React、XState、SQLite/Drizzle。

302 个文件的类型为：

| 类型 | 文件数 |
|---|---:|
| code | 204 |
| config/manifest | 57 |
| documentation | 24 |
| assets | 12 |
| binary | 3 |
| other | 2 |

主要扩展名包括 111 个 `.ts`、65 个 `.tsx`、25 个 `.json`、24 个 `.md`、23 个 `.yaml`、10 个 `.py` 和 3 个 `.exe`。25 个 JSON 全部是 package/config/schema/MCP 元数据，不是实验记录。

项目数据库 schema 管理用户、工作流、agent logs、文档、知识库、模型 provider、MCP、审批和产物。少量 `sequence` 字样表示日志块顺序号，不是氨基酸序列；`endpoint` 表示 HTTP/API endpoint，不是 ADMET 端点。

## 4. 肽数据检查结果

以下结构化或常见科学数据文件数量全部为 0：

- CSV、TSV、JSONL；
- XLS、XLSX；
- Parquet、Feather；
- SQLite、SQLite3、DB；
- SDF、FASTA、FA、SMI、SMILES。

强领域词全文检查结果也全部为 0 个命中文件：

- `peptide`、`cyclic peptide`、`amino acid sequence`；
- `SMILES`、`InChIKey`；
- `ADMET`、`PBPK`；
- `plasma protein binding`、`volume of distribution`、`bioavailability`。

因此不存在可以映射为以下最小契约的记录：

```text
sequence_or_structure | endpoint | value_or_label | source
```

`data/mozi_v15_integration_audit_v1/normalized_candidates.tsv` 已保留完整候选表头，但数据行为 0。

## 5. 重复与可叠加结论

### 5.1 数据层

| 类别 | 数量 | 结论 |
|---|---:|---|
| exact observation duplicates | 0 | 没有候选实验记录，无法形成数据重复 |
| identity-only overlaps | 0 | 没有 sequence/HELM/SMILES/InChIKey |
| additive observations | 0 | 没有新端点或新实验条件可以追加 |
| review candidates | 0 | 没有领域数据行需要判定 |
| excluded repository files | 302 | 全部为软件代码、配置、文档、资产或二进制 |

这里的“重复为 0”不是说 ZIP 内容独一无二，而是说其中没有肽性质 observation，因而与 V15 的数据级重复为 0。

### 5.2 软件文件层

归档内部有 3 组完全相同的文件，共 7 个文件：

1. 3 个 WinSW EXE 内容完全相同，每个 18,243,033 bytes；两份额外复制理论上占 36,486,066 bytes。
2. `packages/getnote-mcp/tsconfig.json` 与 `packages/google-mcp/tsconfig.json` 完全相同。
3. `packages/server/tsconfig.json` 与 `packages/shared/tsconfig.json` 完全相同。

这些只能称为“归档内部软件文件重复”，不能计入肽数据重复数，也没有自动删除。

## 6. 为什么不能直接整合

1. **对象类型错误**：这是应用源码，不是实验数据导出。
2. **缺少最小字段**：没有 molecular identity、endpoint、value/label、unit、assay context 或 source。
3. **没有领域清洗逻辑**：已有 Python/TypeScript 脚本用于 PDF vault、成本回填、服务部署和工作流，不是肽数据标准化。
4. **许可不清楚**：全树没有找到 `LICENSE`、`COPYING` 或 `NOTICE`，根 package 也未声明 license；不能默认复用或再分发代码。
5. **安全边界**：包含 EXE、provider/credential 相关代码和 `.env.example`，不能进入科学数据主表，更不能执行。

因此归档级决策是：

```text
EXCLUDE_FROM_V15_WRONG_ARTIFACT_TYPE
```

V15 主表没有被修改，审计前后 SHA-256 都是：

```text
9660cfd4e95a4777a052045864fa38a22dba6a55d7646954cd1d4539bebbf7bf
```

## 7. 如果师妹真正的数据在另一个文件中，应该怎样清洗

正确文件最好是 CSV、TSV 或 XLSX，并至少包含：

```text
sequence / HELM / stereochemical SMILES / InChIKey
endpoint
value or label
relation and unit
species and matrix
assay
route, dose, timepoint or exposure duration
source URL and DOI/PMID/database ID
license or redistribution note
```

拿到正确数据后，按以下规则处理：

1. **冻结原包**：记录原路径、bytes、SHA-256，不在原文件上直接改。
2. **来源与许可**：每一行必须能回到论文、数据库或原始补充文件；代码许可证不能代替数据许可证。
3. **身份规范化**：标准线性肽清理空格和大小写；环化、二硫键、非天然残基、端基修饰、D/L 立体化学必须用 HELM 或结构表示保留，不能只剩裸 sequence。
4. **端点规范化**：拆开 bound/unbound、absolute/apparent、数值/分类以及端点子类型；例如 PPB `% bound` 不能与 `% unbound` 混合，Kp 不能与 skin logKp 或 BBB proxy 混合。
5. **条件规范化**：species、matrix、assay、route、formulation、dose、timepoint/exposure duration 都进入 observation key。
6. **单位规范化**：保留原始值和单位，同时写 canonical value；任何转换都要保存公式和原值。
7. **重叠判定**：
   - identity、端点语义、值/标签、单位和实验条件都一致：exact duplicate；
   - identity 相同，但端点、物种、基质、assay、剂量或时间不同：可叠加的新 observation；
   - observation 语义相同但值冲突：保留双方来源，进入 review，不自动平均；
   - 只有名字、没有可验证结构或序列：review，不进入训练。
8. **科学门与许可门分离**：可训练不等于可公开或可商业再分发。
9. **先 staging 后 append**：先输出 duplicate/additive/review 清单和审计 manifest，全部通过后才追加 V15。

机器可读流程见 `data/mozi_v15_integration_audit_v1/cleaning_plan.tsv`。

## 8. 可复用与不可复用部分

本仓库没有可直接复用的肽数据或肽清洗规则。若获得作者明确授权，最多可以参考其通用思想：

- source JSON 和附件追踪；
- artifact JSON Schema；
- audit logs；
- workflow progress 和 retrospective。

这些是工作流设计思想，不是可以导入 V15 的 observation，也不应在许可未确认时复制代码。

## 9. 交付与验证

核心文件：

- `data/mozi_v15_integration_audit_v1/audit_manifest.json`
- `data/mozi_v15_integration_audit_v1/file_inventory.tsv`
- `data/mozi_v15_integration_audit_v1/exact_file_duplicate_groups.tsv`
- `data/mozi_v15_integration_audit_v1/normalized_candidates.tsv`
- `data/mozi_v15_integration_audit_v1/overlap_summary.tsv`
- `data/mozi_v15_integration_audit_v1/integration_decisions.tsv`
- `data/mozi_v15_integration_audit_v1/domain_keyword_audit.tsv`
- `data/mozi_v15_integration_audit_v1/cleaning_plan.tsv`

重建和测试：

```bash
python scripts/audit_mozi_v15_integration.py
python -m unittest -v tests.test_audit_mozi_v15_integration
uvx --offline ruff check \
  scripts/audit_mozi_v15_integration.py \
  tests/test_audit_mozi_v15_integration.py
python -m compileall -q \
  scripts/audit_mozi_v15_integration.py \
  tests/test_audit_mozi_v15_integration.py
```

所有质量门均为 true，包括归档 SHA、文件数、解压 bytes、安全解压、候选文件计数、V15 未修改和 V15 主表与 pipeline manifest 一致。
