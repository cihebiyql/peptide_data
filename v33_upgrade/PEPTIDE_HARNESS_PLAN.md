# Peptide-Harness (v35 / OmniPanel 1.5) 架构设计与实施规划书（v2.0 红队修订版）

> **版本**: v2.0（经 3 路子代理交叉验证 + Step 0 Claude Sonnet 5.5 独立红队挑刺并闭环全部 3 项 P0 与 17 项 P1）· **修订日期**: 2026-11-27  
> **配套活体追踪文档**: [`PEPTIDE_HARNESS_PROGRESS_LOG.md`](PEPTIDE_HARNESS_PROGRESS_LOG.md)

---

## 1. 修订后的 7 条原则性红线（零容忍，红队一票否决项）

1. **红线 1 — 活性保真与相对原始母体的累计编辑硬预算（修复 P0-1）**
   - **禁止无界漂移**：所有编辑预算均相对**原始母体（Initial Seed）**计算——累计氨基酸替换 `<= 3`，累计化学修饰 `<= 1`（`n_methylation` / `cyclization` / `lipidation_c16_c18` 仅作为**叶子节点**施加在末端，绝不再作为母体继续序列进化，彻底消除 SMILES→残基映射失真，修复 P1-8）。
   - **功能端点或显式降级模式**：
     - 若用户指定功能端点（如 AMP 指定 Tier-A 的 `MIC_E_coli`），自动施加**相对原始母体不恶化硬约束**（回归 `cand <= parent + tol` 或分类不降）；
     - 若用户既未指定功能端点也未设置 `locked_positions`，强制进入 **`developability_only` 模式**：累计氨基酸替换上限自动收紧至 **`<= 2`**，且结果强制携带 `activity_unmodeled = True` 警告标签。
2. **红线 2 — 严格 Pure-ML 奖励，封杀机制锚点作弊**
   - 全程优化与基准评分强制向 `compute_peptide_mpo` 传入 `mechanism_anchor = None`（`use_mechanism_anchor = False`），彻底杜绝 `batch_mpo.py:180-193` 的 C18 脂化免费 `+25 MPO` 与 OOD 豁免作弊。机制锚点仅作事后只读提示展示。
3. **红线 3 — 同轨差分 + 精确元组缓存去重（修复 P1-3, P1-5）**
   - 当启用化学图叶子修饰（`sequence = None`）时，母体同时在 `dual_modality` 与 `smiles_only` 双轨评估；每个变体**严格对照同轨道母体计算 `delta_x / delta_y / delta_mpo`**。
   - `track_switched` 记为**中性标记**（实测 `smiles_only` 在 `half_life_reg` 和 `PPB` 上甚至高于 `dual_modality`，并非“降级”）。
   - `adapter.py` 批内去重键强制使用完整元组 **`(canonical_smiles, sequence, include_tier2, cond_sig)`**，严禁仅用 `canonical_smiles` 合并双轨请求。
4. **红线 4 — 按轨道动态读取端点分级 + 务实双层 AD 门控（修复 P0-3, P1-4, P1-6）**
   - **动态按轨道分级**：从 `manifest["endpoints"][ep]` 实时按当前轨道（`dual_modality` / `smiles_only`）计算，不写死数字：
     - **Tier A（主优化允许）**：`evidence_grade == "quantitative"`；
     - **Tier B（仅报告 + 不恶化软约束，不进 MPO 加权）**：`evidence_grade == "rank_only"` 且当前轨 `spearman >= 0.45`, `r2 >= 0.15`, `n_eff_clusters >= 25`, `uq_reliability != "low"`（自动排除 `uq=low` 的 `CL_absolute` / `F` / `PPB` 和 `n_eff=5` 的 `LogD_decadiene`）；`LogD_decadiene` 仅大环肽诊断展示；
     - **Tier C（禁止作为优化目标，硬拒绝 `ValueError / HTTP 400`）**：`exploratory_not_recommended`（`Vd_absolute`）及一切不满足 Tier A/B 的端点（含 `MDCK`、`Cmax`、`CL_absolute`、`MBEC_biofilm`、`MIC_P_aeruginosa` 等）。
   - **去掉无效“档内常数共形采集”，改用清晰的双层 AD 门控 + 双轨分歧检验**：
     - 对**所有 MPO 支柱端点与主目标**施加逐端点硬底线：`max_tc >= min(0.40, parent_tc - 0.10)`（修复母体本身 OOD 如环孢素 A 时门控空洞的问题）；
     - 保留单层 `ood_penalty`（仅在母体原为 OOD 时生效，不再与档内常数半宽三重叠加）；
     - 对晋升精英池的候选可选计算**双轨认知分歧 `|ŷ_dual − ŷ_smiles|`**，并诚实记录选中增益的 **winner's curse 跨轨缩水幅度**。
5. **红线 5 — 支柱硬否决 + 几何平均 MPO + 相对母体物理护栏（修复 P1-7, P1-9, P1-13）**
   - **公式冻结（`Feas-gMPO`）**：
     1. 先执行**支柱硬否决门控**：`d_hem >= 0.25`（对应 `hemolysis_cls < 0.45` 或相对母体改善）且 `d_tox >= 0.25`（对含二硫键母体采用相对不恶化 `cand <= max(0.55, parent + 0.05)`，并在代码中实算母体基线而非写死数值）；未通过直接淘汰并记入负样本记忆库；
     2. 通过硬否决后，用归一化权重计算加权几何平均：`Feas-gMPO = 100 * m_ad * ∏_k (max(d_k, 0.05))^{w_k}`（`evidence_grade` 在 Tier A 支柱均为 `1.0`；`m_ad` 在 `In-Domain/Borderline` 为 `1.0`，仅母体本为 OOD 时应用平滑罚项）。
   - **精简硬化学护栏（v1 不做过度阈值）**：RDKit 单连通无自由基、元素白名单 `{H,C,N,O,F,P,S,Cl,Br,I}`、`MW ∈ [200, 6500]` 且 `ΔMW <= +800 Da`、脂链 `<= 1`、未配对游离巯基 `<= 1`、相对母体净电荷窗口 `|chg - parent_chg| <= 4.0` 且 `[-max(5.0, |parent_chg|+1), +max(6.5, |parent_chg|+1)]`；疏水斑块 / Eisenberg `μH`（100° 窗口、残基均值）/ GRAVY / `pI` 沉淀区统一作为**软开发性告警指标**（使用相对上限 `max(cap, parent + δ)`，永不误杀 LL-37 等天然母体）。
   - **环肽/二硫键母体边界透明声明**：现有 `peptide_opt.py:581-582` 对 `>=12` 元环（Octreotide / Linaclotide / Oxytocin / Cyclosporine A）不执行序列突变，因此 v1 显式声明这些 SMILES 环肽母体运行在 **`chemical_leaf_only` 模式**（仅执行化学修饰且锁定核心子结构，不伪造序列突变）；修复 `peptide_opt.py:394` 中 `len > 30` 覆盖首尾残基与 `len <= 30` 首位插入 `C` 偏移的两个分支。
6. **红线 6 — 实测延迟定预算 + 纯异步主路由 + 严格防并发饿死（修复 P1-1, P1-2, P1-12, P1-14, P1-15）**
   - Step 1 动工前先实测 `node2`（或同配置引擎）`batch=4` 延迟；默认每子批 `<= 4` 条、批间释放 `_predict_sem`。
   - **生产并发隔离**：`HarnessJobManager`（`max_workers=1`）仅在 `_predict_sem` 拥有全部 2 个空闲槽时才借用 1 槽（保证实时 `/predict` 永不饿死）；**取消易超时的阻塞多轮同步接口**（仅保留 `POST /api/v1/peptide-harness/jobs` `202` 异步主路由 + `GET/DELETE`；若需同步快测路由则严格限定 `rounds=1, max_evals<=6, wall_clock<=15s`）。
   - **预算会计**：`max_total_evals` **把 Critic 评估全额计入**；耗尽时优雅返回当前最优解并标 `budget_exhausted=True`。
   - **防滥用与防静默降级**：Harness 路由强制合法 `X-Client-Id` + 来源 IP 活跃并发上限 `1`；每轮开始断言 `oracle_mode`（生产严格模式下若非 `v33_remote` 立即中止返回部分结果；离线单测必须显式标 `oracle_is_surrogate_heuristic=True`）。
7. **红线 7 — 基准严禁循环论证，必须对决“强简单基线”（修复 P0-2, P1-10, P1-11, P1-16）**
   - **指标严格三分**：
     1. *约束合规率（Constraint Compliance，不作质量结论）*：`DevPass%`、`OOD Drift%`、脂链/疏水斑块合规率；
     2. *Oracle 代理搜索效率（Surrogate Search Metrics）*：冻结定义的 `Feas-gMPO` 与固定参考点 `Feas-cHV`（仅计通过全部硬门控的候选，参考点固定为各领头肽 `r_ref = (parent_x - 0.5, parent_y - 0.5)` 标准化空间）；
     3. *独立反循环核验（Independent Anti-Circularity Audit）*：
        - **双轨跨视图复评**：以 `dual_modality` 优化得到的 Top-K 序列在无 ESM-C 的 `smiles_only` 轨道复评，报告**增益保留率与 winner's curse 缩水率**；
        - **`HemoPI2 v1.3` 外部交叉验证**：仅在纯规范序列子集报告并标明覆盖率（非规范修饰明确标 `unverified_by_hemopi2`，不计入通过）；
        - **母体在库状态显式标注**：检查每个领头肽 InChIKey14 是否在训练集，明确标注 `in_sample` vs `out_of_sample`（优先纳入 FDA 2019–2023 blind 集/Case17 留出簇领头肽）。
   - **对抗强基线（消灭稻草人）**：主基线设为 **`Exhaustive-1-Step + Greedy-Combine`（完全相同门控与相同 `Feas-gMPO`）**，并增加 **`禁用脂化（No-Lipidation）` 对照** 与按策略分解的增益归因，诚实检验多轮 Harness 相比单步贪心组合是否有真实额外收益！

---

## 2. 精简务实的迭代实施路线（按步推进 + 每步独立红队挑刺）

吸收红队 P2 裁剪建议，将 Step 1 拆为 **1a → 1b**，砍掉未经验证的过度设计（`LLMGuidedProposer` 仅保留干净接口桩、默认禁用；`Post-Evolution Polish` 与复杂 Filter Agent 仅在 Step 3 基准证明核心价值后再按需开启）：

- **Step 1a（纯引擎内核 + 确定性门控 + 锁位与算子加固）**：
  - 加固 `web/adapter.py`（精确四元组批内去重 + `tier2=True` 缓存复用 + 2048 缓存）与 `web/peptide_opt.py`（`CoreLockSpec` 锁位过滤 + 修复 `c[C-seq-C]` 两个分支）。
  - 新建 `web/peptide_harness.py`（Part A）：按轨道动态 Tier A/B/C 分类器、`CoreLockSpec`、两阶段 `DeterministicGatekeeper`、支柱硬否决 + 几何 `Feas-gMPO`、相对母体物化软告警、序列优先/化学叶子算子。
  - 使用确定性 `FakeOracle` 跑通核心不变量单测（`web/tests/test_v35_harness.py`） → **更新进度日志 → Claude 红队挑刺 Step 1a**。
- **Step 1b（多轮演化循环 + 无头多端点 Critic + 负样本记忆 + 真实延迟压测）**：
  - 扩展 `web/direct_attribution.py` 无头多端点接口（`skip_svg=True`, `locked_positions` 排除, 零成本复用 Ala-scan）。
  - 完善 `web/peptide_harness.py`（Part B）：多轮精英种群演化、Critic 反思与 `NegativeSampleMemory`、完整预算会计（含 Critic 评估）、双轨母体同轨差分与精英跨轨分歧检查、`oracle_mode` 断言。
  - 在真实 `node2`（只读测试）测量 `batch=4` dual / `smiles_only` 延迟以校准默认参数 → **更新进度日志 → Claude 红队挑刺 Step 1b**。
- **Step 2（独立异步任务管理 `web/harness_jobs.py` + FastAPI 路由与防滥用护栏）**：
  - 实现 `HarnessJobManager`（`max_workers=1`、空闲双槽让渡交互闸门、`202 Accepted`、`GET` 增量进度、`DELETE` 协作取消、强校验 `X-Client-Id` + 来源 IP 并发限流、TTL=3600s 清理）。
  - 集成 `schemas.py` / `app.py`（`/api/v1/peptide-harness/jobs` 主路由 + 受限快测路由），全量 `pytest` 通过 → **更新进度日志 → Claude 红队挑刺 Step 2**。
- **Step 3（等预算消融、强基线对决与反循环基准 `expansion_v35/01_benchmark_peptide_harness.py`）**：
  - 在真实生产引擎上跑通多领头肽基准：对比 `Exhaustive-1-Step + Greedy-Combine`（强基线）、`Random`、`Unconstrained Greedy`（作弊对照）、`w/o Negative Memory`、`w/o AD Gate`、`Peptide-Harness Full` 及 `No-Lipidation` 消融；
  - 严格分列“约束合规率 / Oracle 代理效率 / 双轨缩水与 HemoPI2 独立核验”与 `in_sample` 标注 → **更新进度日志 → Claude 红队挑刺 Step 3**。
- **Step 4（前端 `#view-opt` 双模式交互工作台）**：
  - 仅上线高价值精炼交互：模式切换（Single-Round vs Multi-Round Harness）、可点击残基锁芯片（20-AA 自动解锁，环肽提示 `chemical_leaf_only`）、异步轮次进度条与取消按钮、Critic 负样本与归因卡片、Top-K 推荐卡（含 `activity_unmodeled` 诚实标签与一键 `Load 47-EP`）；守住 `?v=1.4.6`、437+ 对称双语键与零 `0.903` 回归 → **更新进度日志 → Claude 红队挑刺 Step 4**。
- **Step 5（线上零显存部署、按需归档同步与最终红队验收）**：
  - 仅热重启 `node2` `:7862`（`0 MiB` GPU VRAM），完成公网 HTTPS 异步全流程验证；若 Step 3 基准证实净价值，再同步更新 `.publish/peptide_data_v31` 与 `SHA256SUMS` 推送 `origin/main` → **最终红队签字收尾**。
