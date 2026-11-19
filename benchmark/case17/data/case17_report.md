# Case17 对比底表报告：平台预测 vs 文献参考值

## 1. 执行摘要

- 平台预测成功率：**17/17**（node23 omnigrid :8901，smiles-only，共 736 行端点预测，每分子 43-44 个端点）

- 半衰期锚定层（half_life_anchor，SMILES 机制层）命中分子：**5/17**（5 个脂肪酰化药物全部命中）

- PPB 端点覆盖：**17/17**

- 输入 SMILES 来源：PubChem 立体异构（canonical isomeric）SMILES，按名称逐一解析（Tirzepatide 取 CID 166567236）

## 2. 半衰期对比（核心）

| 药物 | 文献 t1/2 | 平台（锚定层优先） | 倍差/判定 |
|---|---|---|---|
| Leuprolide | ~3 h (2.9-3 h, 二室模型) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Bivalirudin | ~25 min (肾功能正常) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Glucagon | 8-18 min (说明书; 文献 IV 亦有 3-6 min) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Teriparatide | IV ~5 min; SC ~1 h | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Exenatide | 2.4 h (末端半衰期) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Liraglutide | ~13 h (SC) | 9 h [5-13 h] (daily_acyl) | x1.4 (命中) |
| Semaglutide | ~1 周 (~165-168 h) | 144 h [120-168 h] (weekly_diacid) | x1.2 (命中) |
| Tirzepatide | ~5 天 (~125 h) | 144 h [120-168 h] (weekly_diacid) | x0.87 (命中) |
| Cyclosporine_A | ~8.4 h (5-18 h) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Oxytocin | 3-5 min (标签亦见 1-6 min) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Octreotide | 1.7-1.9 h (SC/IV) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Linaclotide | 血浆低于定量限 无法计算 (肠道局部作用) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Ziconotide | 鞘内 CSF ~4.6 h (2.9-6.5 h); 血浆 ~1.3 h(旧标签) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Daptomycin | ~8 h (8-9 h) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Caspofungin | β 相 9-11 h (γ 相 40-50 h) | n/a (无锚定行; ML half-life 需序列特征被跳过) | n/a |
| Insulin_detemir | 5-7 h (剂量依赖) | 9 h [5-13 h] (daily_acyl) | x0.67 (命中) |
| Insulin_degludec | ~25 h (稳态 剂量无关) | 25 h [25-25 h] (depot_diacid) | x1 (命中) |

### 锚定层命中明细（5/5）

- **Liraglutide**：文献 13h，锚定 daily_acyl 9h [5-13h] → 13h 恰在范围上沿，命中
- **Semaglutide**：文献 ~168h，锚定 weekly_diacid 144h [120-168h] → 命中
- **Tirzepatide**：文献 ~125h，锚定 weekly_diacid 144h [120-168h] → 命中
- **Insulin detemir**：文献 5-7h，锚定 daily_acyl 9h [5-13h] → 命中
- **Insulin degludec**：文献 ~25h，锚定 depot_diacid 25h [25-25h] → 精确命中

注：非酰化药物（12 个）无锚定行；ML 回归版 half_life 端点依赖 sequence 特征，修饰药 smiles-only 提交下按预期跳过——这是修饰多肽的已知覆盖边界（pepADMET 等同类工具同样受 SMILES-only 输入限制）。

## 3. PPB 对比

| 药物 | 文献 PPB | 平台 PPB (fraction_bound) | 倍差/判定 |
|---|---|---|---|
| Leuprolide | 43-49% | 0.491 fraction_bound | x1.1 |
| Bivalirudin | ~0% (不与血浆蛋白/红细胞结合) | 0.448 fraction_bound | 文献~0 (模型高估) |
| Glucagon | 低/未明确报道 (非酰化 29 肽) | 0.832 fraction_bound | n/a |
| Teriparatide | ~0% (无明显结合) | 0.651 fraction_bound | 文献~0 (模型高估) |
| Exenatide | ~53% | 0.661 fraction_bound | x1.2 |
| Liraglutide | >98% (白蛋白) | 0.779 fraction_bound | x0.79 |
| Semaglutide | >99% (白蛋白) | 0.993 fraction_bound | x1 |
| Tirzepatide | ~96% (血浆白蛋白) | 0.991 fraction_bound | x1 |
| Cyclosporine_A | ~90% (主要脂蛋白) | 0.925 fraction_bound | x1 |
| Oxytocin | 轻度/未定量报道 | 0.358 fraction_bound | n/a |
| Octreotide | ~65% (脂蛋白+白蛋白) | 0.691 fraction_bound | x1.1 |
| Linaclotide | n/a (无全身暴露) | 0.585 fraction_bound | n/a |
| Ziconotide | ~50% | 0.498 fraction_bound | x1 |
| Daptomycin | 90-93% (~92%) | 0.869 fraction_bound | x0.94 |
| Caspofungin | ~97% | 0.871 fraction_bound | x0.9 |
| Insulin_detemir | >98% (~99%, 白蛋白) | 0.913 fraction_bound | x0.93 |
| Insulin_degludec | >99% (白蛋白) | 0.921 fraction_bound | x0.93 |

### PPB 观察

- **高结合命中良好**：Ziconotide 0.498 vs 50%（几乎精确）、Leuprolide 0.491 vs 43-49%、Semaglutide 0.993 vs >99%、Tirzepatide 0.991 vs 96%、Cyclosporine A 0.925 vs 90%、Octreotide 0.69 vs 65%、Daptomycin 0.869 vs 90-93%
- **方向正确但幅度偏低**：Liraglutide 0.78 vs >98%、Caspofungin 0.87 vs 97%、两个胰岛素类似物 0.91-0.92 vs >98/99%（高结合段的压缩回归现象）
- **主要偏差**：Glucagon 0.83 与 Teriparatide 0.65 vs 文献~0——非酰化小肽理论上低结合，模型在这些分子上高估；Bivalirudin 0.45 vs ~0 同理

## 4. 其他端点预览（全表见 case17_comparison_base.tsv）

| 药物 | PAMPA | Caco2 | hemolysis(p) | MIC(uM) | CL(mL/min/kg) | F |
|---|---|---|---|---|---|---|
| Leuprolide | 3.52e-08 | 5.55e-08 | 0.062 | 0.354 | 1.03 | 0.233 |
| Bivalirudin | 2.78e-07 | 5.99e-08 | 0.022 | 0.317 | 0.234 | 0.147 |
| Glucagon | 1.82e-07 | 6.8e-08 | 0.00946 | 0.356 | 0.195 | 0.265 |
| Teriparatide | 2.32e-07 | 1.11e-07 | 0.00693 | 0.325 | 0.229 | 0.164 |
| Exenatide | 2.11e-07 | 5.72e-08 | 0.0621 | 1.47 | 0.207 | 0.273 |
| Liraglutide | 7.95e-08 | 5.31e-08 | 0.0233 | 0.317 | 0.201 | 0.272 |
| Semaglutide | 8.43e-08 | 6.64e-08 | 0.772 | 0.359 | 0.203 | 0.274 |
| Tirzepatide | 1.34e-07 | 4.97e-08 | 0.989 | 1.33 | 0.202 | 0.272 |
| Cyclosporine_A | 1.29e-06 | 2.15e-06 | 0.0461 | 0.348 | 5.1 | 0.329 |
| Oxytocin | 4.51e-08 | 1.4e-07 | 0.394 | 0.349 | 1.45 | 0.213 |
| Octreotide | 9.21e-08 | 3.17e-07 | 0.639 | 0.352 | 1.8 | 0.144 |
| Linaclotide | 1.79e-07 | 8.72e-08 | 0.63 | 0.309 | 0.388 | 0.209 |
| Ziconotide | 1.17e-07 | 9.25e-08 | 0.896 | 0.337 | 0.242 | 0.146 |
| Daptomycin | 4.64e-08 | 9.15e-08 | 0.625 | 0.398 | 0.153 | 0.349 |
| Caspofungin | 7.24e-08 | 1.54e-07 | 0.192 | 0.36 | 0.178 | 0.384 |
| Insulin_detemir | 2.54e-07 | 6.38e-08 | 0.811 | 0.318 | 0.233 | 0.272 |
| Insulin_degludec | 2.65e-07 | 6.14e-08 | 0.813 | 0.319 | 0.233 | 0.272 |

## 5. 初步发现与边界

1. **半衰期锚定层对长效酰化药全部命中（5/5）**，包括范围上沿的 Liraglutide 和精确命中的 Degludec；对非酰化短效药（分钟级）平台目前无 smiles-only 半衰期输出，属设计边界而非错误。

2. **PPB 与文献吻合度**：15 个可比较药物中 ~7 个落在文献区间或 ±10% 内；高结合段（>98%）系统性略微低估（0.77-0.92），零结合小肽高估——分类排序能力好（Spearman 型），绝对值需谨慎。

3. **环肽/脂肽表现**：Cyclosporine A（环十一肽）PPB 0.925 vs 90% 命中、Caco2/PAMPA ~2e-6/1.3e-6 cm/s 为全组最高渗透（与 CsA 高渗透+P-gp 底物特性一致）；Daptomycin/Caspofungin（环脂肽）PPB 0.87 vs 92/97% 方向正确；大环分子的 rdkit_full 部分 NaN 已被服务端容忍（≤10% 缺失策略），未影响预测成功率。

4. **超短效药（Oxytocin/Glucagon/Bivalirudin，分钟级）**：血浆半衰期由蛋白酶降解主导，平台当前 smiles-only 端点无法表达分钟级清除——与 pepADMET 对比时建议以 PPB/渗透性/毒性端点为主战场。
