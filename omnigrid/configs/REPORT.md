# 当前端点、最优组合与评价值

原始61个端点名称已全部列入；汇总仅只读既有训练结果，训练仍暂停。半衰期与溶解度各有分类/回归两种任务，分别列示。
组合中的“实验条件/物种”是该来源任务实际追加的协变量，没有把它们计为独立化学表示方法。

## 当前已保存模型：固定seed42展示

下表固定展示seed42在五折内部交叉验证中选出的实际已保存赢家，不依据留出集挑选随机种子。代表子任务按覆盖最多记录选择，同一端点可能还有很多靶点/物种/条件不同的任务；不能把这一个任务的成绩当成整个端点所有记录的总体成绩。完整三seed及所有子任务均在附件。

CV列：回归为MAE↓、分类为AP↑。留出列：回归给MAE↓；分类给AP↑ / ROC-AUC↑。回归数值位于表内声明的尺度；log10(uM)的误差不是uM原尺度误差。内部5折无独立留出者明确留空，CV用于调参而非独立外验。

| 端点 | 类型 | 单一代表任务n / 同类型任务数 | 实际已存最优组合(seed42) | 五折CV | 留出评价 | 尺度/协议 |
|---|---|---:|---|---:|---|---|
| AC50 | 回归 | 77 / 1 | EState+实验条件＋HistGB | 0.512 | 0.500 | log10(uM) |
| BBB | 分类 | 844 / 1 | ESM2+序列扩展＋ExtraTrees | 0.913 | 0.873 / 0.879 | label01 |
| CC50 | 回归 | 13 / 1 | Layered＋SVR | 0.153 | 0.336 | log10(uM) |
| CL_absolute | 回归 | 7 / 1 | Layered+实验条件＋XGBoost | 0.537 | — | log10(L/h)；内部5折无留出 |
| CL_per_kg | 回归 | 50 / 3 | 原子对＋XGBoost | 0.368 | 0.507 | log10(mL/min/kg) |
| CL_per_mg_protein | 回归 | 6 / 1 | Morgan＋SVR | 0.586 | — | log10(uL/min/mg_protein)；内部5折无留出 |
| CL_renal_absolute | 回归 | 22 / 1 | Morgan+实验条件＋SVR | 0.459 | — | log10(L/h_as_reported_normalization_annotation)；内部5折无留出；审计暂缓 |
| CL_renal_per_kg | 回归 | 18 / 1 | RDKit描述符+实验条件＋Ridge | 0.205 | — | log10(L/h/kg_reported_basis_annotation)；内部5折无留出 |
| Caco2 | 回归 | 1197 / 1 | 结构特征融合＋LightGBM | 0.316 | 0.305 | log10(cm/s) |
| Cmax | 回归 | 36 / 1 | ErG+实验条件＋XGBoost | 0.337 | — | log10(uM)；内部5折无留出 |
| EC50 | 回归 | 380 / 21 | 原子对＋XGBoost | 0.313 | 0.324 | log10(uM) |
| F | 回归 | 19 / 1 | RDKit路径＋XGBoost | 0.126 | 0.317 | fraction |
| GI50 | 回归 | 7 / 2 | EState＋SVR | 0.255 | — | log10(uM)；内部5折无留出 |
| HC10 | 回归 | 10 / 2 | EState＋XGBoost | 0.984 | — | log10(uM)；内部5折无留出 |
| HC50 | 回归 | 12 / 1 | 原子对＋Ridge | 0.339 | 0.270 | log10(uM) |
| IC50 | 回归 | 1235 / 232 | 结构特征融合＋ExtraTrees | 0.322 | 0.325 | log10(uM) |
| IC50_inhibitor | 回归 | 6 / 3 | Morgan＋Ridge | 0.147 | — | log10(uM)；内部5折无留出 |
| Kd | 回归 | 191 / 19 | 原子对＋Ridge | 0.569 | 0.926 | log10(uM) |
| Ki | 回归 | 426 / 136 | 结构特征融合＋XGBoost | 0.621 | 0.600 | log10(uM) |
| Km | 回归 | 74 / 6 | 拓扑扭转＋HistGB | 0.259 | 0.381 | log10(uM) |
| LC50 | 回归 | 8 / 4 | 结构特征融合＋XGBoost | 0.074 | — | log10(uM)；内部5折无留出 |
| LC_lemma | 回归 | 21 / 2 | 序列组成＋ElasticNet | 0.109 | 0.269 | log10(uM) |
| LD50 | 回归 | 7 / 1 | Pattern＋Ridge | 0.206 | — | log10(uM)；内部5折无留出 |
| LogD7.4 | 回归 | 57 / 1 | Layered＋XGBoost | 0.523 | 0.521 | dimensionless_log10 |
| MBC | 回归 | 17 / 3 | 序列特征融合＋XGBoost | 0.394 | 0.508 | log10(uM) |
| MDCK | 回归 | 55 / 1 | Pattern＋Ridge | 0.383 | 0.694 | log10(cm/s) |
| MFC | 回归 | 31 / 2 | RDKit描述符＋Ridge | 0.245 | 0.416 | log10(uM) |
| MHC | 回归 | 24 / 1 | RDKit描述符＋XGBoost | 0.131 | 0.431 | log10(ug/mL) |
| MHC10 | 回归 | 27 / 1 | RDKit描述符+物种＋XGBoost | 0.378 | 0.391 | log10(uM) |
| MIC | 回归 | 64 / 114 | 拓扑扭转＋XGBoost | 0.051 | 0.113 | log10(uM) |
| MIC50 | 回归 | 24 / 7 | 序列特征融合＋XGBoost | 0.553 | 0.594 | log10(uM) |
| MIC90 | 回归 | 11 / 5 | 原子对＋XGBoost | 0.220 | — | log10(uM)；内部5折无留出 |
| PAMPA | 回归 | 6748 / 1 | 结构特征融合＋LightGBM | 0.281 | 0.268 | log10(cm/s) |
| PPB | 回归 | 52 / 2 | 拓扑扭转＋Ridge | 0.194 | 0.269 | fraction_bound |
| Papp | 回归 | 79 / 2 | 结构特征融合＋XGBoost | 0.470 | 0.406 | log10(cm/s) |
| RRCK | 回归 | 135 / 1 | 原子对＋XGBoost | 0.323 | 0.355 | log10(cm/s) |
| Vd_absolute | 回归 | 27 / 1 | Morgan+实验条件＋随机森林 | 0.672 | 0.688 | log10(L) |
| Vd_per_kg | 回归 | 20 / 1 | EState+实验条件＋XGBoost | 0.397 | — | log10(L/kg)；内部5折无留出 |
| Vdss_per_kg | 回归 | 24 / 1 | 原子对＋XGBoost | 0.239 | 0.269 | log10(L/kg) |
| cytotoxicity_CC50 | 回归 | 6 / 2 | Layered＋Ridge | 0.024 | — | log10(uM)；内部5折无留出 |
| cytotoxicity_EC50 | 回归 | 29 / 1 | Layered＋ExtraTrees | 0.301 | 0.311 | log10(uM) |
| cytotoxicity_IC50 | 回归 | 29 / 5 | 结构特征融合＋ElasticNet | 0.412 | 0.469 | log10(uM) |
| cytotoxicity_LC50 | 回归 | 11 / 1 | RDKit描述符＋ElasticNet | 0.218 | 0.451 | log10(uM) |
| half_life | 分类 | 67 / 4 | 序列组成＋弹性网Logistic | 0.886 | 0.667 / 0.895 | label01 |
| half_life | 回归 | 375 / 23 | ESM2+序列扩展＋Ridge | 0.350 | 0.335 | log10(h) |
| hemolysis | 分类 | 2012 / 2 | 序列特征融合＋LightGBM | 0.852 | 0.834 / 0.934 | label01 |
| hemolysis_EC50 | 回归 | 19 / 1 | 原子对＋XGBoost | 0.592 | 0.860 | log10(uM) |
| hemolysis_HC10 | 回归 | 12 / 1 | Pattern＋Ridge | 0.592 | 0.491 | log10(uM) |
| hemolysis_HC50 | 回归 | 38 / 3 | Pattern+物种＋Ridge | 0.238 | 0.268 | log10(uM) |
| hemolysis_LC50 | 回归 | 38 / 1 | ESM2＋SVR | 0.285 | 0.310 | log10(uM) |
| hemolysis_MHC | 回归 | 18 / 1 | RDKit描述符＋ExtraTrees | 0.317 | 0.880 | log10(uM) |
| hemolysis_MHC10 | 回归 | 21 / 1 | ESM2+序列扩展＋XGBoost | 0.397 | 0.311 | log10(uM) |
| hemolysis_SOURCE_DEFINED_BINARY | 分类 | 29 / 1 | 拓扑扭转＋Logistic | 1.000 | 0.700 / 0.500 | label01 |
| overall_peptide_toxicity | 分类 | 11036 / 1 | ESM2+序列扩展＋LightGBM | 0.956 | 0.958 / 0.955 | label01 |
| permeability | 回归 | 24 / 1 | 原子对＋Ridge | 0.418 | 0.277 | log10(cm/s) |
| solubility | 分类 | 1334 / 5 | 序列组成＋XGBoost | 0.944 | 0.946 / 0.730 | label01 |
| solubility | 回归 | 35 / 1 | MACCS＋ElasticNet | 0.147 | 0.270 | log10(uM) |

## 三seed同口径CV组合建议

每行展示该端点/任务类型中数据覆盖最多的同口径子任务。最优特征—模型组合按该子任务、同一覆盖scope的42/2026/3407三seed最佳调参CV均值排名；超参可因seed而异，未另拟合固定共识模型。不能跨靶点、物种、单位或样本口径比较分数后称全端点唯一最优。

回归CV列为MAE↓，分类CV列为AP/PR-AUC↑，均为三seed均值±样本标准差。回归误差在声明的目标变换尺度上，log10误差不能当作原单位误差。内部五折无独立留出者，CV同时用于调参，数值偏乐观。

独立留出列只在三个实际已存赢家全部等于推荐特征—模型组合时填值；否则见逐seed明细，避免把另一模型的留出成绩贴到推荐组合上。相同配方的超参仍按seed分别优化。

| 端点 | 类型 | 代表样本数 | 三seed建议组合 | 五折CV MAE↓ / AP↑ | 留出MAE↓ / ROC-AUC↑ | 口径 |
|---|---|---:|---|---:|---:|---|
| AC50 | 回归 | 77 | 结构特征融合+实验条件＋SVR | 0.513±0.015 | — | common；uM/log10 |
| BBB | 分类 | 844 | ESM2＋ExtraTrees | 0.895±0.014 | — | common；label01/identity |
| CC50 | 回归 | 13 | 拓扑扭转＋Ridge | 0.167±0.029 | — | common；uM/log10 |
| CL_absolute | 回归 | 7 | Layered+实验条件＋XGBoost | 0.672±0.233 | — | 内部5折，无留出；L/h/log10 |
| CL_per_kg | 回归 | 50 | 原子对＋XGBoost | 0.334±0.080 | — | common；mL/min/kg/log10 |
| CL_per_mg_protein | 回归 | 6 | 原子对＋随机森林 | 0.578±0.050 | — | 内部5折，无留出；uL/min/mg_protein/log10 |
| CL_renal_absolute | 回归 | 22 | 拓扑扭转+实验条件＋LightGBM | 0.470±0.021 | — | 内部5折，无留出；最终审计暂缓；L/h_as_reported_normalization_annotation/log10 |
| CL_renal_per_kg | 回归 | 18 | RDKit描述符+实验条件＋Ridge | 0.205±0.000 | — | 内部5折，无留出；L/h/kg_reported_basis_annotation/log10 |
| Caco2 | 回归 | 1197 | 结构特征融合＋LightGBM | 0.312±0.015 | — | full_structure；cm/s/log10 |
| Cmax | 回归 | 36 | ErG+实验条件＋XGBoost | 0.318±0.036 | — | 内部5折，无留出；uM/log10 |
| EC50 | 回归 | 380 | 结构特征融合＋LightGBM | 0.318±0.004 | — | full_structure；uM/log10 |
| F | 回归 | 19 | 拓扑扭转＋ExtraTrees | 0.132±0.034 | — | common；fraction/identity |
| GI50 | 回归 | 7 | 结构特征融合＋SVR | 0.242±0.019 | — | 内部5折，无留出；uM/log10 |
| HC10 | 回归 | 10 | EState＋XGBoost | 1.020±0.037 | — | 内部5折，无留出；uM/log10 |
| HC50 | 回归 | 12 | 拓扑扭转＋SVR | 0.357±0.019 | — | common；uM/log10 |
| IC50 | 回归 | 1235 | 结构特征融合＋ExtraTrees | 0.318±0.016 | — | full_structure；uM/log10 |
| IC50_inhibitor | 回归 | 6 | Morgan＋Ridge | 0.147±0.000 | — | 内部5折，无留出；uM/log10 |
| Kd | 回归 | 191 | 原子对＋Ridge | 0.671±0.094 | — | full_structure；uM/log10 |
| Ki | 回归 | 426 | Morgan＋XGBoost | 0.644±0.024 | — | full_structure；uM/log10 |
| Km | 回归 | 74 | 拓扑扭转＋HistGB | 0.280±0.027 | — | common；uM/log10 |
| LC50 | 回归 | 8 | 结构特征融合＋XGBoost | 0.078±0.005 | — | 内部5折，无留出；uM/log10 |
| LC_lemma | 回归 | 21 | RDKit描述符＋ElasticNet | 0.135±0.007 | — | common；uM/log10 |
| LD50 | 回归 | 7 | ErG＋XGBoost | 0.242±0.037 | — | 内部5折，无留出；uM/log10 |
| LogD7.4 | 回归 | 57 | 结构特征融合＋Ridge | 0.509±0.067 | — | full_structure；dimensionless_log10/identity |
| MBC | 回归 | 17 | Pattern＋Ridge | 0.499±0.062 | — | common；uM/log10 |
| MDCK | 回归 | 55 | RDKit描述符＋SVR | 0.449±0.068 | — | full_structure；cm/s/log10 |
| MFC | 回归 | 31 | ErG＋ExtraTrees | 0.262±0.031 | — | common；uM/log10 |
| MHC | 回归 | 24 | EState＋XGBoost | 0.245±0.063 | — | common；ug/mL/log10 |
| MHC10 | 回归 | 27 | EState+物种＋XGBoost | 0.452±0.076 | — | common；uM/log10 |
| MIC | 回归 | 64 | 拓扑扭转＋XGBoost | 0.054±0.005 | — | common；uM/log10 |
| MIC50 | 回归 | 24 | 序列组成＋SVR | 0.607±0.071 | — | common；uM/log10 |
| MIC90 | 回归 | 11 | 原子对＋XGBoost | 0.226±0.026 | — | 内部5折，无留出；uM/log10 |
| PAMPA | 回归 | 6748 | 结构特征融合＋LightGBM | 0.280±0.001 | 0.273 | full_structure；cm/s/log10 |
| PPB | 回归 | 52 | 拓扑扭转＋Ridge | 0.232±0.038 | — | common；fraction_bound/identity |
| Papp | 回归 | 79 | 结构特征融合＋XGBoost | 0.441±0.032 | — | full_structure；cm/s/log10 |
| RRCK | 回归 | 135 | RDKit描述符＋XGBoost | 0.309±0.029 | — | full_structure；cm/s/log10 |
| Vd_absolute | 回归 | 27 | Morgan+实验条件＋随机森林 | 0.646±0.035 | — | common；L/log10 |
| Vd_per_kg | 回归 | 20 | RDKit描述符+实验条件＋XGBoost | 0.444±0.035 | — | 内部5折，无留出；L/kg/log10 |
| Vdss_per_kg | 回归 | 24 | 原子对＋XGBoost | 0.234±0.039 | — | common；L/kg/log10 |
| cytotoxicity_CC50 | 回归 | 6 | RDKit描述符＋Ridge | 0.031±0.006 | — | 内部5折，无留出；uM/log10 |
| cytotoxicity_EC50 | 回归 | 29 | MACCS＋XGBoost | 0.315±0.060 | — | common；uM/log10 |
| cytotoxicity_IC50 | 回归 | 29 | ErG＋ExtraTrees | 0.495±0.059 | — | common；uM/log10 |
| cytotoxicity_LC50 | 回归 | 11 | RDKit描述符＋ExtraTrees | 0.254±0.074 | — | common；uM/log10 |
| half_life | 分类 | 67 | 序列组成＋弹性网Logistic | 0.891±0.107 | — | common；label01/identity |
| half_life | 回归 | 375 | 序列特征融合＋SVR | 0.345±0.025 | — | common；h/log10 |
| hemolysis | 分类 | 2012 | 序列特征融合＋LightGBM | 0.840±0.015 | — | common；label01/identity |
| hemolysis_EC50 | 回归 | 19 | Layered＋XGBoost | 0.690±0.077 | — | common；uM/log10 |
| hemolysis_HC10 | 回归 | 12 | Pattern＋ElasticNet | 0.748±0.085 | — | common；uM/log10 |
| hemolysis_HC50 | 回归 | 38 | Pattern+物种＋Ridge | 0.233±0.029 | 0.319 | common；uM/log10 |
| hemolysis_LC50 | 回归 | 38 | ESM2＋SVR | 0.326±0.038 | — | common；uM/log10 |
| hemolysis_MHC | 回归 | 18 | ESM2+序列扩展＋SVR | 0.334±0.064 | — | common；uM/log10 |
| hemolysis_MHC10 | 回归 | 21 | ESM2+序列扩展＋XGBoost | 0.400±0.011 | — | common；uM/log10 |
| hemolysis_SOURCE_DEFINED_BINARY | 分类 | 29 | MACCS＋Logistic | 0.964±0.063 | — | common；label01/identity |
| overall_peptide_toxicity | 分类 | 11036 | ESM2+序列扩展＋LightGBM | 0.955±0.001 | — | common；label01/identity |
| permeability | 回归 | 24 | RDKit路径＋SVR | 0.393±0.096 | — | common；cm/s/log10 |
| solubility | 分类 | 1334 | 序列组成＋随机森林 | 0.941±0.002 | — | common；label01/identity |
| solubility | 回归 | 35 | 结构特征融合＋SVR | 0.156±0.003 | — | common；uM/log10 |

## 定性细胞毒性：嵌套CV

cytotoxicity：150条、45分组，外5内5。joint预声明主臂三seed外层OOF AP=0.464±0.250，ROC-AUC=0.778±0.090。各外折组合不同，没有全局单一最优配方或全量部署模型；没有用外层分数挑选比较臂。各臂及15个外折配方见nested_cytotoxicity_arms.tsv。

## 无独立端点评价或暂缓

| 原端点 | 当前状态 | 说明 |
|---|---|---|
| cytotoxicity_HC50 | registered_reference_observation_trained | Only canonical/registered observation participation; no independent endpoint score |
| cytotoxicity_SOURCE_DEFINED_BINARY | gap | No completed eligible model; deferred / readout unresolved |
| hemolysis_CONCENTRATION_AT_50_PERCENT | gap | No completed eligible model; deferred / readout unresolved |
| hemolysis_HD50 | gap | Only canonical/registered observation participation; no independent endpoint score |
| hemolysis_IC50 | canonical_observation_trained | Only canonical/registered observation participation; no independent endpoint score |

## 完整明细

- 同任务/覆盖口径组合建议：1391行，[全部子任务](all_task_scope_best_combinations.tsv)。
- 实际获选模型：4324行，[逐seed模型、超参、CV与留出指标及来源路径](all_seed_selected_models.tsv)。
- [代表任务三个seed的实际模型与评价](representative_seed_models.tsv)、[固定seed42已存最优模型表](actual_best_seed42_overview.tsv)。
- [端点代表任务表](endpoint_representatives.tsv)、[JSON汇总](summary.json)。
- 原始61名称、不同任务类型、不同实验条件与coverage scope未静默合并；本概览不是61个独立临床验证预测器。所有新训练、扩源与未完成审计继续保持暂停。
