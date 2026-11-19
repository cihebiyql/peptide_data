# Case17 基准测试：17 个经典多肽药物 vs pepADMET

## 概述

叶老师（中检院）提供的 17 个经典多肽药物案例测试，用于与中南大学 pepADMET（JCIM 2026）进行性能对比。

## 测试药物

Leuprolide, Bivalirudin, Glucagon, Teriparatide, Exenatide, Liraglutide, Semaglutide, Tirzepatide, Cyclosporine_A, Oxytocin, Octreotide, Linaclotide, Ziconotide, Daptomycin, Caspofungin, Insulin_detemir, Insulin_degludec

## 核心成绩

| 指标 | 结果 | 说明 |
|---|---|---|
| 半衰期（长效机制识别） | **5/5 呟中** | 司美格鲁肽/替尔泊肽为真外推（标签未参与计算）|
| 蛋白结合排序能力 | **ρ=0.81** | 12 个训练集外药物（剔除 3 个训练内药后仍为 0.81）|
| 渗透性（环孢素） | **与 pepADMET 同精度** | 本平台为训练集外真预测；对方训练源 CycPeptMPDB 含环孢素，疑为训练内回放 |

## 训练集交叉检查

| 药物 | 本平台训练集 | pepADMET 训练集 |
|---|---|---|
| 环孢素 | 不在（0 端点）| 渗透源 CycPeptMPDB 含环孢素（高概率在内）|
| 司美格鲁肽 | 9 端点含 PPB | 未知（全量训练集未公开）|
| 利拉鲁肽 | 10 端点含 PPB/half_life | 未知 |
| 齐考诺肽 | 4 端点含 PPB | GitHub 示例毒性数据中**已发现** |
| 其余 12 药 | **全部不在** | — |

## 文件结构

```
benchmark/case17/
├── README.md                          ← 本文件
├── data/
│   ├── case17_platform_predictions.tsv  ← 736 行全端点预测结果
│   ├── case17_reference_values.tsv      ← 17 药文献参考值（FDA 说明书核实）
│   ├── case17_comparison_base.tsv       ← 对比底表（平台 vs 文献）
│   ├── key_endpoints.json              ← 重点端点提取（JSON）
│   ├── case17_report.md                ← 初步分析报告
│   └── pepADMET_对比方案汇总.md          ← pepADMET 调研+对比方案+叙事修正
├── docs/
│   ├── 平台汇报_成绩单版.pdf             ← 最终汇报（成绩先行+逐药判定+训练集检查）
│   ├── 中检院会议反馈_逐条回复.pdf        ← 七条反馈回复
│   └── 17药完整测试与pepADMET对比_详细版.pdf ← 详细版（全部原始数值+竞品热图）
```

## pepADMET 关键信息

- 论文：JCIM 2026, 66(2), 936-946（DOI: [10.1021/acs.jcim.5c02518](https://doi.org/10.1021/acs.jcim.5c02518)）
- Server: [pepadmet.ddai.tech](https://pepadmet.ddai.tech)（自 2026-11-19 起持续 504）
- 19 个建模端点 vs 本平台 55 个；无 MIC/抗菌活性/PPB/CL/Vd 端点
- 渗透性训练源 CycPeptMPDB 含环孢素案例 → 环孢素渗透"预测"疑为训练内回放
- GitHub 仅开源毒性模型（[ifyoungnet/pepADMET](https://github.com/ifyoungnet/pepADMET)），示例毒性数据 135 条中发现齐考诺肽
- 无 API、无批量、单分子提交

## 注意事项

- 全部预测仅供内部科研使用（internal_research_only）
- pepADMET 全量训练集未公开，训练集交叉检查基于公开示例 + 训练来源数据库交叉证据
- 结构差异核验：亮丙瑞林已修正为 N-乙基正确形式；替尔泊肽采用 PubChem 脂化结构（原文件为去脂化形式）
