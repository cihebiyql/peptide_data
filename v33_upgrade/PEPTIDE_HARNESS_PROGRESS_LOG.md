# Peptide-Harness (v35 / OmniPanel 1.5) 活体执行与红队挑刺日志

> **用途**: 贯穿整个 Peptide-Harness 规划、编码、基准测试、前端集成与上线部署全流程的持续维护文档。  
> **规则**: 每完成一个阶段，必须在此记录：① 本步变更与实测结果；② 独立红队 AI（Claude Sonnet 5.5 挑毛病模式）的原话批评与缺陷分级（`P0 / P1 / P2`）；③ 对应修复落地证据与复核结论。

---

## 总进度看板

| 阶段 | 状态 | 红队 AI 审计 | `P0` | `P1` | 关键产出 |
|---|---|---|---:|---:|---|
| **Step 0 · 跨代理交叉验证 + 红队规划挑刺** | ✅ 已闭环修订（v2.0） | Claude Sonnet 5.5 R0 (`BLOCKED → 全部 3 P0 + 17 P1 已修订`) | 3→0 | 17→0 | `PEPTIDE_HARNESS_PLAN.md` (v2.0) |
| **Step 1a · 纯引擎内核、核心锁、动态分级与门控** | ✅ 全部 4 P0 + 10 P1 已闭环修复（9/9 对抗测试通过） | Claude Sonnet 5.5 R1a (`BLOCKED → 4 P0 + 10 P1 全部修复`) | 4→0 | 10→0 | `web/peptide_harness.py` + `adapter/peptide_opt` + `test_v35_harness.py` |
| **Step 1b · 多轮演化、无头 Critic、负记忆与双轨审计** | ✅ 全部 3 P0 + 8 P1 已闭环修复（47/47 测试绿灯） | Claude Sonnet 5.5 R1b (`BLOCKED → 3 P0 + 8 P1 全部修复`) | 3→0 | 8→0 | `web/direct_attribution.py` 无头多端点 + `run_peptide_harness(...)` |
| **Step 2 · 异步任务管理与 API 安全集成** | ✅ R2 全部 P0-C / R1b 遗留 / 6 P1 / P2 已闭环修复（50/50 测试绿灯） | Claude Sonnet 5.5 R2 (`PASS WITH REVISIONS → 全部修复`) | 1→0 | 6→0 | `web/harness_jobs.py` + `schemas/app/adapter/chem_utils` |
| **Step 3 · 强基线对决、消融与反循环基准** | ✅ R3 全部 2 P0 + 4 P1 + 3 P2 及组合堆叠 Bug 已闭环并完成 v35.2（16 肽×7 方法）终审放行 | Gemini 4 Carbon R3-v2 (`PASS`, **0 P0 / 0 P1** / 3 P2 全部顺手清零) | 2→0 | 4→0 | `expansion_v35/01_benchmark_peptide_harness.py` + `v35_peptide_harness_benchmark_receipt.json` (16 肽) |
| **Step 4 · 前端 `#view-opt` 双模式交互工作台** | ✅ R4 全部 8 P1 + 关键 P2 已闭环修复（472==472 对称双语 + 51/51 pytest 绿灯） | Claude Sonnet 5.5 R4 (`PASS WITH REVISIONS → 0 P0 / 8 P1 全部修复`) | 0 | 8→0 | `web/static/{index.html,app.js,styles.css,i18n.js}` (`?v=1.4.7`) |
| **Step 5 · `node2` 零显存上线、归档同步与终审** | ✅ `node2` v1.5.0（0 MiB GPU 显存）公网核验通过，终审 `PASS` 放行并完成 `.publish/peptide_data_v31` 归档与 `origin/main` 推送 | Gemini 4 Carbon R3-v2 (`PASS`, **0 P0 / 0 P1**) | 0 | 0 | 线上 HTTPS `v1.5.0` + `.publish/peptide_data_v31` + `SHA256SUMS` |

---

## Step 0 · 3 路交叉验证 + Claude Sonnet 5.5 独立红队挑刺记录（2026-11-27）

### 1. 红队 R0 结论：`BLOCKED`（针对 v1.0 规划）
- 审稿线程：`Step 0 Red-Team Plan Roast (Claude Sonnet 5.5)`（严格只读对照源码）
- **抓出的 3 项原则性硬伤 (`P0`)**：
  1. **`P0-1` 目标函数缺活性保真与相对原始母体的累计编辑预算**：纯 4 支柱 MPO 会将 AMP/功能肽演化为“低毒但失活的惰性肽”。
  2. **`P0-2` 基准“独立非 Oracle 指标”不独立且存在循环**：`DevPass%` / `OOD Drift%` 是 Gate 自身条件；`HemoPI2` 仅覆盖 `seq[:40]` 规范序列；`|ŷ_CPU − ŷ_Fusion|` 不可用且不独立；`Feas-cHV` / `Feas-gMPO` 公式未冻结。
  3. **`P0-3` 悲观共形采集在 `mondrian_ad_eb_cw` 下是空操作**：档内 `w80` 为常数，不改变档内排序，且与 OOD 门控 + `ood_penalty` 形成三重叠加。
- **抓出的关键 `P1` 缺陷（17 项核心摘录）**：
  - 未实测 `node2` 批延迟且同步接口易超 Nginx 60s；`_predict_sem` 缺交互让渡闸门；批内去重仅用 SMILES 会合并 `dual` 与 `smiles_only` 请求；Tier B/C 计数 `20+12+14=46` 自相矛盾且随轨道变化；`smiles_only` 并非降级轨道；母体 OOD 时门控空洞；绝对物化阈值会误杀 LL-37 等天然母体；化学修饰后再做序列编辑无可靠原子映射；环肽母体在 `peptide_opt.py:581` 无序列算子；缺少 `Exhaustive-1-Step + Greedy-Combine` 强基线与无脂化对照；几何平均 `0.05^0.25=0.47` 不能硬否决致命毒性；`X-Client-Id` 可伪造绕过单人限流；Critic 评估需计入总预算。

### 2. v2.0 规划闭环处置（全部落实进 `PEPTIDE_HARNESS_PLAN.md`）
- **P0-1 闭环**：确立相对原始母体硬预算（AA 编辑 `<=3`、化学修饰 `<=1` 且限定叶子节点）；支持声明功能端点不恶化约束；缺功能端点且缺锁位时强制进入 `developability_only` 模式（AA 上限收紧至 `<=2` + 强制 `activity_unmodeled=True` 标签）。
- **P0-2 闭环**：指标严格三分（约束合规率 / 冻结公式的 `Feas-gMPO` & `Feas-cHV` / 独立双轨复评缩水 + 纯序列 `HemoPI2` 覆盖率标注 + `in_sample` 标注）；新增最强简单基线 `Exhaustive-1-Step + Greedy-Combine` 与 `No-Lipidation` 对照。
- **P0-3 闭环**：删除“共形采集”误导口径，改为全支柱逐端点 AD 底线 `max_tc >= min(0.40, parent_tc - 0.10)` + 精英双轨分歧 `|ŷ_dual − ŷ_smiles|` 检验。
- **P1/P2 闭环**：全部 17 项 P1 写入红线 1–7，按红队 P2 建议将 Step 1 拆为 **1a（内核/门控/锁位）+ 1b（演化循环/无头 Critic/延迟实测）**，裁剪非必要过度设计，稳步推进。


---

## Step 1a · 核心引擎内核、锁位加固与 `node2` 真实批延迟实测（2026-11-27）

### 1. 本步交付与实测结果
- **线上 `node2` 未命中 `batch=4` 实测延迟**（通过公网 `POST /api/v1/batch-screen` 实测）：
  - `dual_modality`（4 条 12-mer 未命中序列）：`wall = 9.41s`，`server_ms = 8,842.8 ms`（均摊 **2.21 s/条**）；
  - `smiles_only`（4 条 12-mer 未命中 SMILES）：`wall = 4.68s`，`server_ms = 4,529.7 ms`（均摊 **1.13 s/条**）。
  - 结论：采用子批 `<= 4` 条、异步 `wall_clock_budget_s = 120s` 可稳妥支撑 3 轮演化（~24–30 次未命中评估约 40–60s）。
- **代码落地**：
  - `web/adapter.py`：默认 `V33_CACHE_MAXSIZE` 增至 `2048`；`_call_v33_remote_batch` 升级为**精确四元组 `(canonical_smiles, sequence, include_tier2, cond)` 批内去重** + `include_tier2=False` 自动复用已缓存的 `include_tier2=True` 超集。
  - `web/peptide_opt.py`：5 类生成算子全部支持可选 `locked_positions` / `locked_atom_indices` / `lock_disulfides`（100% 向后兼容既有 33 项测试）；修复 `c[C-seq-C]` 两个分支（`len > 30` 时若首/尾位点锁定则禁止覆盖；`len <= 30` 标明为末端化学叶子节点）。
  - `web/peptide_harness.py`（新建 Part A）：实现按轨道动态读取 `manifest["endpoints"]` 的 Tier A/B/C 分类器、`CoreLockSpec`（含 `functional_guided` / `developability_only` / `chemical_leaf_only` 与相对初始母体累计编辑上限）、两阶段 `DeterministicGatekeeper`（Pure-ML 几何 `Feas-gMPO` + 支柱硬否决 + 二硫键相对毒性规则 + 逐端点相对 AD 底线 + 功能不恶化门控 + 相对母体物化软告警）、`NegativeSampleMemory` 与序列优先/化学叶子生成器。
  - `web/tests/test_v35_harness.py`（新建）：6 项核心不变量单测全部通过，叠加既有 33 项测试共 **39/39 PASS**。

### 2. Step 1a 红队 AI（Claude Sonnet 5.5 挑毛病模式）审计与修复闭环
- **红队 R1a 结论：`BLOCKED`**（通过实时对抗探针抓出 4 项 P0 与 10 项 P1）：
  - **`P0-1` Stage B 缺数据 fail-open + tier2 缓存污染**：缺 `MIC_E_coli`、缺安全端点或 `max_tanimoto=None` 时均放行；`include_tier2=False` 复用 `tier2=True` 缓存会把 Tier-2 端点泄漏给调用方。
    - **已修复**：Stage B 全面 fail-closed（缺任一必选端点/母体基线/AD 相似度立即拒绝或抛错）；`adapter.py` 复用 `tier2=True` 缓存时做浅拷贝并严格过滤回 Tier-1 端点，且增加结果长度越界保护。
  - **`P0-2` Stage A 轻信调用方标签可伪造**：`sequence=None` 或伪造 `sequence != SMILES` 可绕过全部锁位与预算；`Di-N-Me` / `c[C-seq-C]` 化学编辑计数偏低。
    - **已修复**：强制校验 `canon(sequence_to_mol(seq)) == can`，拒绝未标注非序列分子与母体完全相同候选，按分子图差分独立推导真实化学编辑数（`Di-N-Me` 计 2、`c[C-seq-C]` 计 3）。
  - **`P0-3` 母体 sequence 与 SMILES 不一致（如 C 端酰胺化）时锁位静默清空且剥离酰胺**。
    - **已修复**：`resolve_core_locks` 发现 `canon(sequence_to_mol(seq)) != parent_can` 时，若传入 `locked_positions` 直接抛 `ValueError`，否则强制切换 `chemical_leaf_only` 并置 `parent_sequence=None`。
  - **`P0-4` `peptide_opt.py` 两处锁位旁路**：`c[C-seq-C]` `len<=30` 分支无视末端锁；`lock_disulfides=False` 时显式锁定的游离 Cys 仍可成环。
    - **已修复**：`c[C-seq-C]` 两个长度分支统一受 `not termini_locked` 约束；自由巯基成环无条件排除 `locked_atoms`。
  - **10 项 P1 一并修复**：`NegativeSampleMemory` 覆盖全部 `ala_scan`/`point_mutation` 路径；`validate_optimization_objectives` 仅放行 Tier A 且校验全部活跃轨道；禁止跨轨母体回退；`ad_factor=0.72` 仅作用于母体本身 OOD 场景；二硫键母体采用统一哺乳动物细胞毒 `d_tox` 计算（消除 ad-hoc gMPO 抬高）；收紧坏母体溶血/细胞毒豁免（需改善 `>=0.15` 且 `<=0.70` 且 `HC50/CC50` 不恶化）；Stage A 统一使用 RDKit SMARTS 净电荷估计器（实测 LL-37 叶子候选不再误拒）；区分分类(`0.05`)/回归(`0.15`)功能容差。
- **复核**：新增 9 项针对全部 P0/P1 的对抗性单测，与既有 v34 测试合计 **15/15 PASS**（全套 42 项绿灯）。


---

## Step 1b · 多轮受控演化循环、无头多端点 Critic 与双轨认知审计（2026-11-27）

### 1. 本步交付与实测结果
- **`web/direct_attribution.py`**：新增 `compute_headless_multi_endpoint_attribution(...)`（`skip_svg=True`、预排除 `locked_positions`/`locked_atom_indices`、按 `chunk_size<=4` 分批并支持协作取消，单次评估同时返回多目标 + 安全端点 $\Delta\hat{y}$ 与可零成本复用的 `reusable_evals`）。
- **`web/peptide_harness.py`（Part B）**：实现完整多轮演化引擎 `run_peptide_harness(...)`：
  - 按需自动开启 `include_tier2`（覆盖 Tier-2 功能端点如 `MIC_E_coli`）；
  - 双轨母体基线评估 + 同轨差分 + 精英 `dual_modality → smiles_only` 跨轨认知分歧与 winner's curse 增益保留率审计；
  - 首轮零成本复用 Critic 的 Ala 扫描评估，精英未变自动跳过重复 Critic；第 2 轮起支持最优单点有益编辑的两两组合堆叠（严守初始母体累计编辑上限）；
  - 化学修饰严格作为叶子节点施加在末轮（或 `chemical_leaf_only` 模式）；
  - 全程计入 Critic 评估至 `max_total_evals` 硬预算，支持 `cancel_event` 协作取消与每轮开始 `require_remote_oracle` 断言，冻结 2D `compute_pareto_flags_and_hv`。
- **`web/tests/test_v35_harness.py`**：新增端到端多轮演化、零成本 Critic 复用、跨轨审计、严格远程断言与协作取消测试，10/10 新增测试 + 6/6 v34 测试全部绿灯（**16/16 PASS**）。

### 2. Step 1b 红队 AI（Claude Sonnet 5.5 挑毛病模式）审计与修复闭环
- **红队 R1b 结论：`BLOCKED`**（复核 R1a 4 P0 确认 `P0-1 / P0-3 / P0-4` 闭合，`P0-2` 对化学叶子子结构复查重新打开，并实测抓出 3 项 P0 与 8 项 P1）：
  - **`P0-1` 混合轨道原始空间 Pareto/HV 与跨轨截距作弊**：`smiles_only` 轨道截距高于 `dual_modality` 时，比自己母体差（`delta_gmpo = -3.87`）的叶子节点也能挤进 Pareto 与 Top-K。
    - **已修复**：`compute_pareto_flags_and_hv` 全面改在**同轨差分空间 `(delta_x_raw, delta_y_raw)`** 运作，参考点显式负偏移；精英与 Top-K 严格按同轨 `delta_gmpo` 排序且强制要求 `delta_gmpo > 0`。
  - **`P0-2` Critic 取消时 `base_out[0]` 越界崩溃（`IndexError`）**。
    - **已修复**：`compute_headless_multi_endpoint_attribution` 复用已评估的 `base_result`，对空/异常 `base_out` 返回 `base_ok=False` 并安全退出。
  - **`P0-3` 伪造 `is_chemical_leaf=True`（如将锁定 `K1/W2/F3` 突变为 `R1/Y2/Y3`）可绕过 Stage A 锁位**：MW 增加可通过简单 MW 检验。
    - **已修复**：新增 `_build_peptide_scaffold_query(ref_mol)`（仅去除 C 端 `OXT`，保留芳香 `[nH]` 确保含 Trp/His 肽安全 sanitize），在 Stage A 对化学叶子做**fail-closed 真实分子图子结构匹配**，任何氨基酸突变均立即触发 `CHEMICAL_LEAF_SUBSTRUCTURE_MISMATCH`。
  - **8 项 P1 全部闭环修复**：
    1. `generate_chemical_leaf_candidates` 预过滤超预算 `Di-N-Me` / `c[C-seq-C]` 并改为**跨策略轮询（round-robin）**，末轮在 `run_peptide_harness` 中为化学叶子预留预算并与序列提案交错排布（`lipidation_c16_c18` / `cyclization` 不再被饿死）；
    2. `chemical_leaf_only` 模式与精英未变时跳过无意义 Critic 消耗；
    3. Critic 预排除已突变位点，`A->G` 回收标记为 `gly_mask`，且 `NegativeSampleMemory` 过滤 `DUPLICATE_CANDIDATE` / `IDENTICAL_TO_PARENT` 防止误污染禁忌表；
    4. 跨轨审计预算预留 `audit_reserve = min(3, top_k)`，`loop_eval_cap = max_total_evals - audit_reserve`；
    5. 跨轨认知分歧改为**去截距差分分歧** `|Δ_dual − Δ_smiles|`，并同步输出 `smiles_only_stage_b_passed` 与 `smiles_only_reject_codes`；
    6. `resolve_core_locks` 在 `canon_mol` 规范原子序上计算 `locked_atom_indices`，修复环肽（如 Octreotide）二硫键原子索引错位；
    7. Stage B 新增分类概率突增硬否决（`hemolysis_cls` / `cytotoxicity_cls` 相对母体暴增 `> +0.25` 且超过 `0.65` 即拒绝，堵住 `0.12 → 0.90` 旁路）；
    8. `_run_batch` 对 `EngineBusyError` 自动重试 1 次并在失败时降级返回 `status="partial"` 与 `partial_error`（保留已评估结果不崩）。
- **复核**：新增 4 组 R1b 专项对抗回归单测，全仓 47/47 pytest 全部绿灯（`12.66s`）。

---

## Step 2 · 独立异步任务管理、交互让渡闸门与 API 安全护栏（2026-11-28）

### 1. 本步交付与实测结果
- **`web/adapter.py`**：新增线程安全 `_predict_idle_cond` / `_predict_active_count` / `_predict_pending_interactive` 状态追踪与 `wait_for_interactive_idle(cancel_event, max_wait_s)` 让渡闸门，确保后台 Harness 每批推理前均检查交互槽位空闲且无等待中的实时 `/predict` 请求，绝不饿死交互用户。
- **`web/harness_jobs.py`（新建）**：独立 `HarnessJobManager`（`ThreadPoolExecutor(max_workers=1)`，与批处理任务中心彻底隔离；全局队列上限 `8`；严格 `X-Client-Id` 与来源 IP 双维度活跃并发上限 `1` 防止伪造 ID 刷队列；严格属主隔离防 IDOR；`DELETE` 协作取消并即时唤醒等待中的让渡闸门；`TTL=3600s` & `MAX_RETAINED=64` 自动回收；默认关闭 LLM 钩子 `V35_HARNESS_LLM_ENABLED=0`）。
- **`web/peptide_harness.py`**：新增 `classify_all_endpoints(...)` 与同步前置校验 `preflight_validate_harness_request(...)`（在返回 `202 Accepted` 之前即同步拦截无效 SMILES、Tier-C 目标、同目标 `X==Y`、越界锁位与离线严格模式）。
- **`web/schemas.py` & `web/app.py`**：新增 `PeptideHarnessRequest`（`ConfigDict(extra="forbid")` 封死任意 SSRF / 未声明字段）、`PeptideHarnessSyncRequest`（严格限死 `rounds=1, max_total_evals<=8, wall_clock<=20s` 且单槽互斥），挂载 `POST/GET /api/v1/peptide-harness/jobs`、`GET/DELETE /api/v1/peptide-harness/jobs/{job_id}`、`GET /api/v1/peptide-harness/tiers`、`POST /api/v1/peptide-harness` 并在 `/api/v1/health` 暴露 `harness_llm_enabled`。
- **`web/tests/test_v35_harness.py`**：新增异步全生命周期、IDOR 隔离、per-IP/per-client 限流、队列上限、排队/运行中双态取消、交互让渡闸门、同步限流与防 SSRF 回归测试。

### 2. Step 2 + R1b 复核红队 AI（Claude Sonnet 5.5 挑毛病模式）审计与闭环
- **红队 R2 结论：`PASS WITH REVISIONS`**（审稿线程：`Step 2 + R1b-Recheck Red-Team Roast (Claude Sonnet 5.5)`）：
  - **已证实稳健**：并发计数器 `_predict_active_count` / `_predict_pending_interactive` 在 8 线程×30 轮随机失败与 150 轮提交/取消竞态压测下均回零且无死锁；跨用户 IDOR、恶意 SMARTS ReDoS、额外字段 SSRF 均被可靠拦截。
  - **1 项条件 P0 + 2 项 R1b 残余 + 6 项 Step 2 P1 + 关键 P2 全部闭环修复**：
    1. **`P0-C`（`C*3000` 触发 RDKit `MolToSmiles` 栈溢出崩溃）**：`chem_utils.py` 的 `parse_smiles_safe` 在规范化前增加 `0 < mol.GetNumAtoms() <= 800` 硬上限（覆盖 426 原子 Insulin degludec 及 65-AA 长肽，彻底杜绝超长碳链递归栈溢出）。
    2. **`R1b P0-3`（超图氨基酸替换 `G->A/W, A->L/F, S->T` 绕过纯子图检查）**：升级 `_verify_chemical_leaf_scaffold`，除子图匹配外强制验证所有非 N/非 S/非 C 端羰基 C 原子的重原子度数不变（`m_atom.GetDegree() <= q_atom.GetDegree()`），100% 拦死所有超图侧链伪造且真实 N-Me/脂化/环化叶子零误拒。
    3. **`R1b P1-3`（多轮配置如 `rounds=4, mte=36` 在进入末轮前耗尽预算导致 0 叶子被评估）**：新增跨轮 `leaf_reserve = len(chem_strategies)` 保留槽，前序序列轮受 `seq_eval_cap` 约束，到达末轮或触及 `seq_eval_cap` 即解锁叶子评估。
    4. **`Step 2 P1-1`（`remaining < 12.0` 抛 `EngineBusyError` 被外层 `except Exception` 吞并毒化全局 `mode`）**：在 `_call_v33_remote_single` 与 `_call_v33_remote_batch` 的 `except Exception` 前新增 `except EngineBusyError: raise`。
    5. **`Step 2 P1-2`（全局 `max_retained` 允许单攻击者反复提交取消挤掉受害者结果）**：改为 `max_retained_per_client=8` 按客户端分区淘汰 + `min_protected_ttl_s=300s` 跨客户保护窗口。
    6. **`Step 2 P1-3 / P1-4 / P1-5`（同步路由缺 per-IP/client 保护、批调用超墙钟与交互批端点对让渡闸门不可见）**：同步路由复用 `harness_job_manager.has_active_for(client_id, client_ip)` 且先占锁再 preflight；`predict_batch` 新增 `http_timeout_s`（按剩余墙钟收紧）与 `is_background` 标记（使 `/batch`、`/batch-screen`、`/peptide-opt`、`/direct-attribution` 交互批调用对让渡闸门完全可见）。
    7. **`Step 2 P1-6`（NaN/Inf Oracle 预测绕过 fail-closed Stage B）**：`DeterministicGatekeeper.__init__` 与 `check_stage_b` 强制 `math.isfinite` 校验，任何 NaN/Inf 均触发 `NONFINITE_ENDPOINT:{ep}` 拒绝。
    8. **`P2-1 / P2-2 / P2-3 / P2-4 / P2-6`**：母体取消直接返回干净 `cancelled` 无误导 `error`；批返回长度严格校验；`_extract_client_ip` 用 `ipaddress.ip_address` 校验并识别 `::ffff:127.0.0.1` / `localhost`；限流先于 RDKit preflight、错误消息脱敏内部地址、校验 `direction in ("max","min")`、终态取消不改 `cancel_requested`。

---

## Step 4 · 前端 `#view-opt` 双模式交互工作台与红队 R4 闭环（2026-11-28）

### 1. 本步交付与静态门禁
- **`web/static/{index.html,app.js,styles.css,i18n.js}`**（静态资源版本 `?v=1.4.7`、`i18n.js?v=1.7`）：
  - `#view-opt` 顶部新增单轮扫描（PeptideOpt）vs 多轮受控演化（Peptide-Harness）模式切换；
  - 新增实时解析 20-AA 序列的可点击核心残基锁芯片 + `chemical_leaf_only` / `developability_only` 透明提示；
  - 新增 Tier-A 功能活性保真下拉、演化轮数/预算控制、1500 ms 异步进度轮询与协作取消按钮；
  - 新增 Top-K 推荐卡（含 `activity_unmodeled` / 离线替代警告、双轨认知审计留存率与一键 `Load 47-EP`）、逐轮 Critic 归因反思轨迹与负样本禁忌表；API 文档示例改用高熵 UUID 客户端 ID。
- **静态回归门禁**：`node --check` 零报错、**472 == 472 对称双语键**、零 `0.903`/内部代号、**50/50 pytest 全绿**。

### 2. Step 4 红队 AI（Claude Sonnet 5.5 挑毛病模式）审计与闭环
- **红队 R4 结论：`PASS WITH REVISIONS`**（审稿线程：`Step 4 Frontend Red-Team Roast (Claude Sonnet 5.5)` — **0 P0**，确认 XSS 防护、请求/响应契约与回归红线全部过关，抓出 8 项 P1）：
  1. **`P1-1` 轮询跨模式覆盖结果**：`runPeptideHarness` 与 `runPeptideOpt` 均增加 `state.optMode` 匹配守卫，切到 Single 时隐藏 Harness 进度条且仅缓存结果不覆盖 Single 视图。
  2. **`P1-2` 下拉含 `rank_only` 伪 Tier-A 选项**：静态兜底列表严格收紧至已验证 `quantitative` 端点（移除 `solubility_reg` / `MIC_S_aureus` / `MIC_B_subtilis`），且 `#harness-func-ep` 改为动态按 `/tiers` 过滤。
  3. **`P1-3` 目标端点回退时方向不同步**：`populateOptObjectiveSelects` 在 `X`/`Y` 回退至默认端点时自动同步对应 `dir`。
  4. **`P1-4` `parseCanonicalSeqInput` 与后端判定漂移**：与 `chem_utils.is_canonical_sequence` 100% 对齐（`len>=4` 且排除 `{C,N,S,P,F,I,H}` 仅含 ≤2 种元素的 SMILES 如 `CCC`/`CCCC`），非序列/命名药物统一经 `query` 由后端权威解析。
  5. **`P1-5` 轮询单次抖动即中止与 422 数组显示 `[object Object]`**：新增 `_safeJsonFromText`、`_formatApiError`（展开 FastAPI 422 数组）与连续 3 次失败容错重试。
  6. **`P1-6` 取消按钮可能误发旧 `job_id`**：提交前清零 `state.harnessJobId = null` 并在收到 `202` 前保持取消按钮禁用。
  7. **`P1-7` 未展示 `oracle_is_surrogate_heuristic` / `no_editable_positions` / `audit_skipped_reason`**：全部新增显式警示横幅与母体名称标头。
  8. **`P1-8` & 关键 P2**：`i18n.js` 升级 `?v=1.7`（`styles.css`/`app.js` 升级 `?v=1.4.7`）；语言切换同步恢复模式描述与运行态按钮文案；换序列时清空旧锁位；全部表头与功能选项双语本地化；窄屏卡片防溢出。

---

## Step 3 · 真实 4-FM V33 引擎 + 外部 HemoPI2 v1.3 强基线与反循环消融基准（2026-11-28）

### 1. 本步交付与全量实测数据（零离线替代，`oracle_mode="v33_remote"`, `surrogate=False`）
- **脚本与回执**：`expansion_v35/01_benchmark_peptide_harness.py`、`expansion_v35/v35_peptide_harness_benchmark_receipt.json`（总耗时 `10,095s`，10 领头肽 × 7 组等预算方法 `max_total_evals=20`；含 5 Case17 临床药、3 库内肽、2 库外全新 `out_of_sample_novel` 肽；`0` 项 CoreLock 违规）。
- **引擎实证迭代（v35.1）**：首轮 `Magainin_2` 实测暴露第 2 轮重复全量 Critic 消耗过半预算的问题后，将紧凑预算下 Critic 优化为首轮执行并 100% 回收为 `ala_scan`，释放第 2/3 轮用于 `point_mutation` / top-3 组合堆叠与化学叶子覆盖（`50/50 pytest` 全绿）。
- **10 领头肽全集实测总表（严格三分：约束合规 / Oracle 代理效率 / 独立反循环检验）**：

| 方法 | 均评估数 | 可通行 `ΔgMPO` 均值 (中位) | `Feas-cHV` 均值 | Top-1 可通行率 | CoreLock 违规 | 策略归因 (Top-1) | 外部 `HemoPI2 v1.3` 覆盖率 / `ΔP` / 不恶化率 | 双轨 `SMILES` 复评通过 / 留存中位 |
|---|---:|---:|---:|---:|---:|---|---|---|
| **Peptide-Harness-Full** | 17.8 | **+9.48** (+5.51) | **0.6109** | 90%* (9/9 推荐可行) | 0 | lipidation 7, point 1, N-Me 1 | 11.1% (1/9) / 0.000 / 100% (8/9 标 `unverified`) | 100% (1/1) / 52.1% |
| **Exhaustive-1Step-GreedyCombine** | 17.8 | +2.87 (+2.31) | 0.4263 | 90%* | 0 | greedy_combine 7, N-Me 1, lip 1 | 77.8% (7/9) / **-0.0757** / 100% | 100% (7/7) / 99.3% |
| **Harness-NoLipidation** | 17.2 | +6.07 (+2.90) | 0.3729 | 80%* | 0 | N-Me 5, point 3 | 37.5% (3/8) / **-0.0450** / 100% | 100% (3/3) / 52.1% |
| **Harness-NoMemory** | 17.8 | +9.46 (+4.91) | 0.5673 | 90%* | 0 | lipidation 5, point 3, N-Me 1 | 33.3% (3/9) / -0.0250 / 100% | 100% (3/3) / 79.3% |
| **Harness-NoGate** | 17.8 | +9.83 (+5.51) | 0.6200 | 90% | 0 | lipidation 8, point 1, N-Me 1 | 10.0% (1/10) / 0.000 / 100% | 100% (1/1) / 52.1% |
| **Random-EqualBudget** | 18.1 | +4.31 (+2.67) | 0.5489 | 90%* | 0 | lip 4, point 2, ala 2, N-Me 1 | 44.4% (4/9) / -0.1413 / 100% | 100% (4/4) / 102.1% |
| **Unconstrained-Greedy** | 18.1 | +4.32 (+2.90) | 0.5884 | **80%** (2 不可行) | 0 | lipidation 8, N-Me 2 | 0.0% (全非规范) | 0 |

*\*注：`Melittin`（高溶血强毒母体）在严格门控下无可行解，`Full` / `StrongBase` / `NoLip` / `NoMemory` / `Random` 均诚实返回 `top1=None`（0 假阳性），计入分母后 Top-1 推荐可行率显示为 90%（实为 9/9 非空推荐 100% 可行）；`Unconstrained-Greedy` 则在 `Melittin` 与 `Exenatide` 上选出 2 个不可行解。*

- **配对统计与诚实局限披露**：
  - `Full` vs `Exhaustive-1Step-GreedyCombine`：`ΔgMPO = +6.608`（`7W/3T/0L`, 95% CI `[+0.91, +12.77]`, `p_raw=0.0156`, `p_holm=0.0936`）；`Feas-cHV = +0.1845`（`6W/3T/1L`, `p_raw=0.0312`）；需诚实说明 `Exhaustive-1Step-GreedyCombine` 在 20 预算内 Phase 1 的 18 个序列突变排在化学叶子前，实际几乎纯走序列组合（7/8 线性肽未触及脂化叶子），因此同时须对照采样了脂化的 `Random-EqualBudget`（`+5.17 ΔgMPO`, `p_raw=0.078`）与去脂化的 `Harness-NoLipidation`（线性肽 `+7.56` 仍超 `StrongBase +3.49` 两倍以上）。
  - **消融诚实结论**：脂化（Pure-ML 无机制锚点）对 `Feas-cHV` 带来显著增量（`+0.2379`, `9W/1T/0L`, **`p_holm=0.0234`**）；`NegativeSampleMemory` 在 20 次短预算下 `Feas-cHV` 呈轻微正向趋势（`+0.0436`, `p_raw=0.30`）而 `ΔgMPO` 无显著差异（`+0.02`, `p_raw=0.84`）；`HemoPI2 v1.3` 在全部规范序列推荐上实现 `100%` 不恶化且均值降溶血，但在 `Full` 上覆盖率仅 `11.1%`（8/9 为非规范化学叶子，明确标 `unverified_by_hemopi2`）。

### 2. Step 3 红队 AI（Gemini 4 Carbon 挑毛病模式）首轮终审（R3）与全量闭环
- **红队 R3 结论：`BLOCKED`**（审稿线程：`Step 3 & Step 5 Final Red-Team Audit (Gemini 4 Carbon)`）：
  - **通过项（无需回滚）**：`peptide_harness.py` 经 **864 组边界模糊测试 + 17/17 对抗单测**确认 0 预算越界、0 锁位旁路、0 门控退化，全部脂化均严格 `mechanism_anchor=None` + 同轨 `smiles_only` 母体差分；公网只读核验正常。
  - **拦截项（2 项 `P0` + 4 项 `P1` + 3 项 `P2`，已 100% 修复落地）**：
    1. **`P0-1` 强基线 `Exhaustive-1Step-GreedyCombine` 遭 8/8 线性肽 0 化学叶子 + 7/8 线性肽 0 `point_mutation` 双重饿死**：`generate_sequence_candidates` 改为 `point_mutation` 与 `ala_scan` 跨算子轮询；重构 `run_exhaustive_1step_greedy_combine`，采用与 Harness 对称的免费 RDKit Stage A 预过滤，Phase 1 在全部 5 类算子（`n_methylation` / `cyclization` / `lipidation_c16_c18` 化学叶子 + 均衡 `point_mutation` / `ala_scan`）间公平配比，Phase 2 剩余槽位自动回投未用候选，绝不浪费预算或母体轨道。
    2. **`P0-2` 安全弃权与假阳性指标混淆且漏报 `Harness-NoGate` 在 `Melittin` 上输出不可行解**：新增无歧义指标 `unsafe_top1_rate`（非空且不可行推荐占比）、`abstention_rate`（安全弃权占比）、`safe_recommendation_precision` 与主部署收益 `selected_feasible_delta_gmpo`（选出不可行 Top-1 一律记 `0.0`，不再从落选池事后代捞），并保留 `pool_best_feasible_delta_gmpo` 作次级诊断。
    3. **`P1-1` `ood_drift_rate` 死字符串 `"AD_FLOOR_FAIL"` 且混入母体固有 OOD**：修正为匹配实际拒绝码前缀 `"OOD_DRIFT:"`，并严格拆分 `ad_floor_drift_rate` 与 `raw_ood_status_rate`。
    4. **`P1-2` Stage A 不对称扣费 + 硬编码 `1.0` + `key_endpoints` 键名拼错导致 `top1_hemolysis_cls` 全空**：`parent_record` 补齐 `key_metrics`；修正 `key_endpoints → key_metrics`（现 7 方法全部非空 Top-1 **100%** 填入 `top1_hemolysis_cls` 且母体 **16/16** 填入）；真实计入 `n_stage_a_rejected` 计算 `stage_a_pass_rate`；`Random-EqualBudget` 与 `StrongBase` 对称启用免费 Stage A 预过滤（仅 `Unconstrained-Greedy` 保持无预过滤作反面对照）。
    5. **`P1-3` 反循环验证样本边界与语料重叠口径不透明**：面板由 10 条扩充至 **16 条领头肽**（14 线性 + 2 环肽），其中 **8/16 (50.0%)** 经 `compounds_v33.tsv` 实证核验为真正库外新肽（`in_corpus_inchikey14=False` 且 `in_corpus_sequence=False`），另 **8/16 (50.0%)** 在全量生产语料库内（含全部 5 个 Case17 临床药——明确标为 `case17_in_prod_corpus`，不再笼统写作 `"held-out"`）。
    6. **`P1-4` 版本号与日志/部署看板同步**：`web/__init__.py` 升级至 `1.5.0` 并热部署至 `node2`；本日志补齐 Step 5a 线上实测记录；`.publish/peptide_data_v31` 继续保持 v34（`293d8d6`）直至 R3-v2 复审放行。
    7. **`P2-1 / P2-2 / P2-3` 同步落地**：回执记录完整 `top1_lineage_path` 与 `top1_lineage_sequence`；`peptide_harness.py` 末轮当 `evals_used >= seq_eval_cap` 时优先排满 `min(len(chem_strategies), leaf_reserve)` 个跨策略化学叶子；样本扩至 `n=16` 突破 Holm-Bonferroni 检验力天花板。
  - **额外自检挖出的生产内核隐藏 Bug 修复（v35.2）**：核查 v35.2 首跑逐条遥测时发现 `peptide_harness.py:2449` 在 `use_negative_memory=True` 时提前对 `comb_specs` 调用 `active_mem.record_seen(...)`，导致 Phase C Stage A 将 Round 2/3 的 `combinatorial_stack` 全部误杀为 `DUPLICATE_CANDIDATE`。已删除该误写行、新增回归单测（**51/51 pytest 绿灯**）、热更新 `node2` `:7862`，并在 `node23` 对受影响的 3 组方法（`Peptide-Harness-Full` / `Harness-NoLipidation` / `Harness-NoGate`）完成同引擎增量重跑（`2,666.7s`，其余 4 组确定性方法继承自同版 7 方法 `9,230.7s` 真实 4-FM 基线跑次）。

### 3. Step 3 v35.2 最终 16 领头肽 × 7 方法实测结果（真实 4-FM V33 + 外部 `HemoPI2 v1.3`）
- **脚本与回执**：`expansion_v35/01_benchmark_peptide_harness.py`、`expansion_v35/v35_peptide_harness_benchmark_receipt.json`（`oracle_mode="v33_remote"`, `surrogate=False`, `max_total_evals=20`；16 条领头肽 = 5 `case17_in_prod_corpus` + 3 `in_sample_corpus` + **8 `out_of_sample_novel`**；全部 7 方法 `CoreLock` 违规恒为 `0`，`top1_hemolysis_cls` 与 `top1_lineage_path` 100% 完整）。
- **16 领头肽全集实测总表（严格三分：约束合规 / Oracle 代理效率 / 独立反循环检验）**：

| 方法 | 均评估数 | 主部署 `selΔgMPO` 均值 (中位) | 池最佳 `poolΔgMPO` | `Feas-cHV` 均值 | 真实 Stage A / B | 真实 AD 漂移 / 原始 OOD | **不安全 Top-1 (`unsafe`)** / 弃权 / 精确率 | 外部 `HemoPI2 v1.3` 覆盖 / `ΔP` / 不恶化 | 双轨 `SMILES` 复评 / 留存中位 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| **Peptide-Harness-Full** | 18.6 | **+10.737** (+7.47) | +11.296 | **0.63663** | 96.6% / 81.1% | 2.1% / 29.9% | **0/16 (0.0%)** / 1 / **100.0%** | **0/15 (0.0%)** *(全为化学叶子，未核验)* | 0 *(无规范 Top-1)* |
| **Exhaustive-1Step-GreedyCombine** *(公平全算子强基线)* | 18.8 | +8.932 (+8.99) | +8.932 | 0.60545 | 98.4% / 78.6% | 2.8% / 32.6% | **0/16 (0.0%)** / 1 / **100.0%** | 8/15 (53.3%) / **-0.0844** / 100% | 8/8 (100%) / 96.7% |
| **Harness-NoLipidation** | 18.2 | +6.383 (+4.21) | +8.226 | 0.39300 | 97.2% / 82.3% | 2.1% / 29.5% | **0/16 (0.0%)** / 2 / **100.0%** | 6/14 (42.9%) / -0.0308 / 100% | 6/6 (100%) / 60.3% |
| **Harness-NoMemory** | 18.6 | +11.062 (+7.47) | +11.282 | 0.63740 | 96.4% / 81.1% | 2.1% / 29.9% | **0/16 (0.0%)** / 1 / **100.0%** | 1/15 (6.7%) / -0.2500 / 100% | 1/1 (100%) / 70.9% |
| **Harness-NoGate** | 18.6 | +11.294 (+8.15) | +12.425 | 0.63823 | 96.8% / 79.7% | 2.1% / 29.2% | **1/16 (6.3%)** / 0 / 93.8% | 3/16 (18.8%) / -0.0817 / 100% | 3/3 (100%) / 76.2% |
| **Random-EqualBudget** | 18.8 | +8.104 (+6.44) | +8.104 | 0.58394 | 99.1% / 82.8% | 0.7% / 34.7% | **0/16 (0.0%)** / 1 / **100.0%** | 7/15 (46.7%) / -0.1300 / 100% | 7/7 (100%) / 95.4% |
| **Unconstrained-Greedy** | 18.8 | +4.171 (+1.86) | +7.380 | 0.60691 | 98.3% / 79.0% | 2.8% / 40.6% | **3/16 (18.8%)** / 0 / 81.2% | 2/16 (12.5%) / -0.0575 / 100% | 2/2 (100%) / 97.1% |

- **配对统计检验（16 领头肽全集，双侧 Wilcoxon + 2000 次 Bootstrap 95% CI + Holm-Bonferroni 6 组校正）与诚实局限披露**：
  1. **多目标帕累托前沿超体积（`Feas-cHV`）显著击败公平全算子强基线**：`Peptide-Harness-Full` vs `Exhaustive-1Step-GreedyCombine` 取得 **`ΔFeas-cHV = +0.0312`（`11W/3T/2L`，95% CI `[+0.0148, +0.0506]`，`p_raw = 0.0058`，`p_holm = 0.0290 < 0.05`）**；在 14 条线性肽上同样显著（`Δ = +0.0356`, `11W/1T/2L`, **`p_holm = 0.0290`**）；在 **8 条真正库外全新肽（`out_of_sample_novel`）** 上保持 **`7W/0T/1L`（`Δ = +0.0337`, `p_raw = 0.0234`）**。
  2. **单点部署收益（`selected_feasible_delta_gmpo`）的诚实边界**：在公平补齐强基线的全 5 类算子后，`Full` 均值仍领先强基线（`+10.737` vs `+8.932`，`Δ = +1.806`, `9W/3T/4L`，线性肽 `+12.215` vs `+10.151`），**但单点 `selΔgMPO` 对比强基线未达统计显著（`p_raw = 0.1730`, `p_holm = 0.5190`）**——原因透明：在单步脂化已占主导的部分激素肽上强基线单步即可接近峰值，而多轮“先序列组合演化 → 末轮化学叶子收尾”的核心增量集中在多目标前沿覆盖（`Feas-cHV` `p_holm = 0.0290`）以及功能导向 AMP 复合轨迹（如 `Magainin_2 +29.86 vs +18.12`、`Synth_AMP_OOS2 +26.07 vs +14.93`、`Synth_AMP_OOS5 +23.87 vs +18.32`）。同时，`Full` 在 `selΔgMPO` 上经 Holm 校正后**显著击败** `Unconstrained-Greedy`（`Δ = +6.566`, `13W/1T/2L`, **`p_holm = 0.0186`**）、`Harness-NoLipidation`（`Δ = +4.355`, `12W/2T/2L`, **`p_holm = 0.0215`**）与 `Random-EqualBudget`（`Δ = +2.634`, `11W/4T/1L`, **`p_holm = 0.0384`**）。
  3. **消融诚实结论（绝不包装无效模块）**：
     - **Stage B 门控价值在于安全防越界而非抬高代理分**：`Harness-NoGate` 在 `Melittin` 上输出 1 例违反溶血门控的不可行脂化毒肽（`unsafe=1/16 = 6.3%`，`Unconstrained-Greedy` 达 `3/16 = 18.8%`），而 `Peptide-Harness-Full` 保持 **`0/16 (0.0%)` 假阳性与 `100.0%` 推荐合规精确率**（在 `Melittin` 上诚实弃权 `top1=None`）；两者在其余安全肽上的代理分无统计差异（`p_raw = 0.2249`）。
     - **负样本记忆（`NegativeSampleMemory`）在 20 次短预算下与无记忆打平**：修复 `combinatorial_stack` 提前入表 Bug 后，`Full` 与 `Harness-NoMemory` 在 **15/16 条领头肽上结果完全一致（`0W/15T/1L`, `p_raw = 0.3173`）**，仅在 `Synth_DevOnly_OOS5` 上因帕累托排序选中更优前沿点而相差 `-0.32` 均值；诚实界定其价值主要体现在更长多轮预算（40–60 次评估）防重复踩雷，而非在 20 预算下提升单点峰值。
     - **Pure-ML 脂化算子贡献显著**：去除脂化后 `Feas-cHV` 暴跌 `-0.2436`（`15W/1T/0L`, **`p_holm = 0.0042`**），`selΔgMPO` 下降 `-4.355`（`p_holm = 0.0215`）。
  4. **独立反循环覆盖率的硬性透明警示（红线 7）**：由于 `Peptide-Harness-Full` 的全部 **15/15 个可行 Top-1 均为非规范化学叶子（13 脂化 + 2 N-甲基化）**，外部 `HemoPI2 v1.3`（仅限 20-AA 规范序列）与 `dual→smiles_only` 跨轨复评在 **`Full` Top-1 上的直接覆盖率均为 `0/15 (0.0%)`**——即 `Full` 末轮化学叶子的高 `ΔgMPO` 增益完全依赖 V33 `smiles_only` 单轨预测，尚无外部工具或跨轨第二视图独立验证，须经前瞻湿实验核实。作为补充佐证，在产生规范序列 Top-1 的 `Exhaustive-1Step-GreedyCombine`（`8/15`）、`Harness-NoLipidation`（`6/14`）与 `Random-EqualBudget`（`7/15`）上，外部 `HemoPI2 v1.3` 实现 **`21/21 = 100%` 不恶化**（均降低预测溶血概率 `ΔP = -0.031 ~ -0.130`），且 `dual→smiles_only` 跨轨 Stage B 通过率亦为 **`21/21 = 100%`**（中位增益留存 `60.3%–96.7%`）。

---

## Step 5a · `node2` 零显存热部署与公网 HTTPS 端到端核验（2026-11-28）

- **部署范围**：已将通过全部审计与 `51/51 pytest` 的 `web/{__init__.py,adapter.py,app.py,chem_utils.py,direct_attribution.py,harness_jobs.py,peptide_harness.py,peptide_opt.py,schemas.py}` 及 `web/static/{index.html,app.js,styles.css,i18n.js}`（`?v=1.4.7` / `i18n.js?v=1.7`）部署至 `node2:/var/www/peptide-omnipanel/v33_next/web/`，仅热重启 Web 进程 `:7862`（保持 `:8902` 推理引擎 `pid=162296` 与 `:18902` ESM-C `pid=3782674` 零中断）。
- **零显存红线核验**：经 `node2` 实时 `nvidia-smi --query-compute-apps` 核查，`:7862`、`:8902`、`:18902` 三个 Peptide OmniPanel 进程 **100% 未占用任何 GPU 显存（`0 MiB`）**。
- **公网 HTTPS 实测核验（`https://lmmd.ecust.edu.cn/peptide-omnipanel/`）**：
  - `GET /api/v1/health` → `200 OK`（`version="1.5.0"`, `adapter_mode="v33_remote"`, `endpoints_total=47`, `harness_llm_enabled=false`, 内部路径脱敏为 `null`）；
  - `GET /api/v1/peptide-harness/tiers` → `200 OK`；
  - 安全护栏实测：缺 `X-Client-Id` → `400`、额外 SSRF 字段 → `422`、Tier-C 目标 → `400`、跨用户 IDOR → `404`；
  - 真实异步闭环：`POST /api/v1/peptide-harness/jobs`（`Octreotide`）→ `202 Accepted` → `3.96s` 返回 `completed`（`oracle_is_surrogate_heuristic=false`）。
- **独立红队最终复审（R3-v2）结论：`PASS`（`0 P0 / 0 P1 / 3 P2`）**（审稿线程：`Step 3 v35.2 & Step 5 Final Re-Audit (Gemini 4 Carbon)`）：
  - 只读脚本独立复算 16 肽 / 14 线性肽 / 8 库外新肽 / 2 环肽 4 个子集全部统计量、Wilcoxon / Bootstrap CI / Holm-Bonferroni `p` 值、强基线 14/14 全算子覆盖与 `112/112` 安全互斥不变量，确认与回执及日志 100% 一致；公网 `v1.5.0` 与全套 `51/51 pytest` 均通过，正式放行 `.publish/peptide_data_v31` 归档。
  - **归档前同步清零 3 项非阻塞 `P2`**：① `peptide_harness.py:1260` 与 `:2438` 校验 `"point_mutation" in strategies`（单选 `ala_scan` 不再溢出 `point_mutation` 候选）；② `peptide_harness.py:2523` 将 Phase C `max_round_evals` 与 `round_eval_cap - evals_used` 取最小，防止未评估候选提前记入 `seen_keys`；③ `01_benchmark_peptide_harness.py` 去重 Phase 2 回投 Stage A 拒绝计数，并在 `v35_peptide_harness_benchmark_receipt.json` 补齐 `runtime_provenance`（注明 `base_full_7method_runtime_s=9230.7s` + `incremental_3method_update_runtime_s=2666.7s`）。
- **`.publish/peptide_data_v31` 最终归档与推送**：已将 v1.5 / v35 全部净文件同步至 `.publish/peptide_data_v31`（零原始/受限 TSV、零二进制权重），重算 `SHA256SUMS` 并推送至 `origin/main`。
