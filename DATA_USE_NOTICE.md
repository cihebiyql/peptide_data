# Data Use and Redistribution Notice

## Scope

本仓库是多来源肽/环肽 ADMET、PK/PBPK 数据的研究性清洗与统计发布。它聚合了公开数据库、论文补充材料、来源定义标签和人工审计结果。

## No Blanket Relicensing

- 本仓库不对所有数据授予统一的新许可证。
- `license` 字段描述的是记录级或来源级已知许可信息；空值或来源条款不明确不等于公共领域。
- 原论文、数据库、补充材料和API的许可、署名、非商业或禁止再分发条款仍然有效。
- GitHub仓库本身公开可访问，不代表其中每条来源记录自动获得商业使用许可。

## Recommended Use

- 科研审计、方法开发、内部模型原型；
- 在保留来源和证据等级的前提下进行任务级建模；
- 对可疑值、弱标签、positive-unlabeled和review层做进一步人工复核。

## Before Redistribution or Commercial Use

1. 按 `source_id`、DOI/PMID、`source_url` 和 `license` 检查来源条款；
2. 保留 `member_lineage_json` 和全部来源归属；
3. 不把 weak、derived 或 rule-constructed 标签描述为实验真值；
4. 对 `row_anomalies.tsv` 中的记录完成原文和单位复核；
5. 建立按 identity + source group 的防泄漏切分；
6. 记录所用仓库commit和数据文件SHA-256。

## Disclaimer

数据按研究审计状态提供，不保证不存在提取错误、单位错误、来源变更或第三方权利限制。使用者负责确认其具体用途所需的权限、伦理和监管要求。
