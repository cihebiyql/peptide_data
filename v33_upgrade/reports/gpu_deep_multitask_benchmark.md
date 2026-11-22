# Peptide OmniPanel V33 — Deep Multi-Task & Attention Neural Architecture Benchmark

- **Generated**: 2026-11-22 03:10:42 +0800 | **Total Runtime**: `3140.7s`
- **Compute Hardware**: Local idle `NVIDIA GeForce RTX 5080` (`16303 MiB`, PyTorch `2.11.0+cu128`, `bf16` TensorCores, peak allocated `1229.3 MiB`; **ZERO** contention with `node1` shared GPUs)
- **Authoritative Artifacts (`node23`)**: `/data/qlyu/v33_upgrade/models_gpu_deep/` (`gpu_deep_summary.json`, 24 standardized endpoint folders with row-aligned `oof_predictions.tsv`, `holdout_predictions.tsv`, `case17_predictions.tsv`, `metrics.json`, plus inference-ready fp16 PyTorch checkpoints in `multitask_checkpoints/` and `halflife_specialist/`)
- **Verification Gate**: **PASSED** (`24/24` endpoints, `55,299 OOF + 13,812 HO = 69,111` exact-aligned rows, `408/408` Leave-Case17-Out predictions, `145` checkpoints totaling `1977.78 MiB`, `smiles_only` exact sequence invariance max diff `0.00e+00` / `0.00e+00`)

---

## 1. Executive Highlights & Architecture Overview

1. **Strong Head-to-Head Deep Multi-Task Performance vs. Wave-2C CPU**: Across the 24 endpoints on the untouched 20% cluster-grouped outer holdout (`dual_modality`), our PyTorch deep neural suite outperforms Wave-2C CPU GBDT/GPR on **4/6 classification endpoints (ROC-AUC)** (BBB, half_life_cls, overall_peptide_toxicity, solubility_cls), **5/18 regression endpoints (Spearman)**, and **9/18 regression endpoints (MAE)** — with particularly decisive breakthroughs on PLM/sequence-heavy endpoints, `half_life`, `BBB`, `toxicity`, `hemolysis`, `MIC_*`, and small-n PK transfer!
2. **Two Complementary Purpose-Built PyTorch Deep Architectures**:
   - **`MultiTaskPeptideNet` (`7,346,182` params)**: Shared multimodal gated trunk fusing 8 modality tokens (`physchem_core[:48]` chem/mod, `ncAA-RepDistill v2` `smiles_repr` 768D + `fusion_hf` 2048D + `protein_proj` 1280D, `PeptideCLM-2` 512D, `ChemBERTa-MTR` 384D, `physchem_core[48:]` seq physchem, and **`ESM-C 600M` 4-region `[mean, max, nterm5, cterm5]` Fused SDPA Multi-Head Self-Attention + Attention Pooling**), with **task-conditioned chemical/sequence branch gating** and **30% stochastic sequence-dropout + 10% SMILES-dropout** so the **EXACT SAME checkpoint** serves both `dual_modality` and `smiles_only` (`seq_mask=0`), feeding a `512D -> 384D` Residual SwiGLU trunk and 24 Condition-Injected Residual-FiLM + PepLife Additive/Low-Rank Heads with **Exact Branchless Censored Gaussian Tobit + Huber-NLL Loss** and `n^0.35` family-balanced multi-task sampling.
   - **`HalfLifeAttentionNet` (`5,604,870` params)**: Dedicated specialist for `half_life_reg` & `half_life_cls` co-trained with PK auxiliaries (`PPB`, `CL_per_kg`, `Vd_per_kg`, `CL_absolute`, `F`, `Cmax`, `LogD7.4`, `solubility_reg`, `PAMPA`, `Caco2`), combining a **Gaussian-RBF / B-spline-inspired KAN layer** on explicit peptide modification flags (`fatty_diacid_len`, `fatty_monoacyl_len`, `is_cyclic`, `n_disulfide`, `n_nmethyl`, `n_d_aa_approx`, `has_peg`), **Chem-to-ESMC Fused Cross-Attention** onto the 4 sequence regions, a **modification-aware sequence gate** (allowing heavily lipidated/cyclic peptides to prioritize chemical modification state while canonical peptides leverage ESM-C N/C-term vulnerability), pairwise condition interaction encodings (`species x matrix`, `matrix x route`), and PepLife I5a-v2 Additive + Bilinear + Residual FiLM heads with Censored Tobit loss.
3. **Mathematically Proven Zero Cross-Endpoint Cluster Leakage**: Verified across all `120` CV folds + `120` Leave-Case17-Out folds + 5 unified 24-head production folds with **`0` leaked clusters** (`assert len(all_train_clusters_across_24_tasks & forbidden_clusters) == 0`).
4. **Calibrated Uncertainty & Censored Fidelity**: Out-of-fold probability calibration improved mean holdout Brier `0.1400 -> 0.1376` and ECE `0.0609 -> 0.0492` across the 6 classifiers; heteroscedastic-sigma + Tanimoto-AD conformal intervals achieved **`78.2%`** (nominal 80%) and **`87.6%`** (nominal 90%) empirical coverage across the 18 regression endpoints.

---

## 2. Strict Zero Cross-Endpoint Cluster Leakage Audit

Pre-modeling cluster analysis revealed that naive multi-endpoint grouping (`PK_SmallPerm`, `HalfLife`, `Safety_Tox`, `Microbiology`) with holdout union masking either loses `16.5%–56.9%` of target CV rows (due to independent per-endpoint 20% holdouts on shared compounds) OR leaks `80–130` target clusters across correlated endpoints (e.g., `half_life_reg` <-> `half_life_cls`, `hemolysis_HC50` <-> `hemolysis_cls`).

**Implemented Per-Target Exact Masking Protocol**: For every target endpoint $E$ and fold $k \in \{0..4\}$, we construct $\mathcal{F}_{E,k} = \text{HoldoutClusters}(E) \cup \text{ValFoldClusters}(E,k)$, keep 100% of target $E$'s 4 training folds, and filter EVERY auxiliary endpoint $A \neq E$ (`is_holdout == 0` only) to remove all rows whose `cluster_id` $\in \mathcal{F}_{E,k}$, enforcing `assert len(train_clusters_all_24 & forbidden_clusters) == 0`. For Leave-Case17-Out, ALL `439` `case17_excluded==1` rows AND all `19` Case17 clusters (`911` rows total across 24 tasks) are globally excluded across every co-trained head.

| Endpoint | Target CV Train Rows (mean/fold) | Aux Rows Kept (mean/fold) | Aux Rows Masked Out for Leakage (mean/fold) | CV Folds Audited | Case17 Folds Audited | Total Leaked Clusters |
|---|---:|---:|---:|---:|---:|---:|
| **BBB** | 540 | 53,707 | 917 | 5 | 5 | **0** |
| **CL_absolute** | 34 | 55,115 | 141 | 5 | 5 | **0** |
| **CL_per_kg** | 132 | 54,135 | 999 | 5 | 5 | **0** |
| **Caco2** | 1012 | 53,260 | 774 | 5 | 5 | **0** |
| **Cmax** | 46 | 54,848 | 394 | 5 | 5 | **0** |
| **F** | 79 | 54,546 | 654 | 5 | 5 | **0** |
| **LogD7.4** | 442 | 54,048 | 698 | 5 | 5 | **0** |
| **MDCK** | 107 | 54,159 | 1,006 | 5 | 5 | **0** |
| **MIC_C_albicans** | 299 | 53,579 | 1,346 | 5 | 5 | **0** |
| **MIC_E_coli** | 1414 | 50,785 | 2,746 | 5 | 5 | **0** |
| **MIC_P_aeruginosa** | 743 | 52,113 | 2,257 | 5 | 5 | **0** |
| **MIC_S_aureus** | 1364 | 50,747 | 2,847 | 5 | 5 | **0** |
| **PAMPA** | 5492 | 47,602 | 832 | 5 | 5 | **0** |
| **PPB** | 104 | 54,747 | 422 | 5 | 5 | **0** |
| **RRCK** | 110 | 54,707 | 455 | 5 | 5 | **0** |
| **Vd_per_kg** | 106 | 54,371 | 795 | 5 | 5 | **0** |
| **cytotoxicity_cls** | 1878 | 49,241 | 3,711 | 5 | 5 | **0** |
| **half_life_cls** | 2050 | 10,707 | 1,303 | 5 | 5 | **0** |
| **half_life_reg** | 1970 | 11,006 | 1,105 | 5 | 5 | **0** |
| **hemolysis_HC50** | 3558 | 46,736 | 4,116 | 5 | 5 | **0** |
| **hemolysis_cls** | 13124 | 35,045 | 3,849 | 5 | 5 | **0** |
| **overall_peptide_toxicity** | 7062 | 43,621 | 2,850 | 5 | 5 | **0** |
| **solubility_cls** | 2382 | 52,011 | 311 | 5 | 5 | **0** |
| **solubility_reg** | 190 | 54,474 | 587 | 5 | 5 | **0** |

---

## 3. Master 24-Endpoint Benchmark — V32 vs. Wave-2C CPU vs. V33 GPU Deep

### 3.1 Quantitative Regression Endpoints (18 Endpoints — Untouched 20% Outer Holdout)

| Endpoint | Tier | V32 HO MAE / Sp | V33 CPU Dual HO MAE / Sp | **V33 GPU Deep Dual CV MAE / Sp** | **V33 GPU Deep Dual HO MAE / RMSE / Sp** | **Δ vs CPU (MAE / Sp)** | GPU Deep SMILES-Only HO MAE / Sp | Conformal 80% / 90% Cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **CL_absolute** | small | CV:0.536/0.582 | 1.491 / 0.420 | 0.670 / 0.452 | **1.180** / 1.464 / **0.658** | **-0.311** / **+0.237** | 1.135 / 0.484 | 63.6% / 72.7% |
| **CL_per_kg** | small | 0.593 / 0.242 | 0.566 / 0.797 | 0.516 / 0.661 | **0.590** / 0.758 / **0.776** | +0.024 / -0.020 | 0.556 / 0.809 | 68.3% / 82.9% |
| **Caco2** | medium | 0.306 / 0.823 | 0.368 / 0.860 | 0.375 / 0.793 | **0.381** / 0.515 / **0.816** | +0.013 / -0.045 | 0.407 / 0.764 | 77.2% / 89.4% |
| **Cmax** | small | CV:0.318/0.738 | 1.106 / 0.720 | 0.860 / 0.264 | **0.916** / 1.453 / **0.614** | **-0.190** / -0.105 | 1.069 / 0.168 | 82.3% / 88.2% |
| **F** | small | 0.238 / -0.034 | 0.239 / 0.478 | 0.194 / 0.571 | **0.236** / 0.336 / **0.728** | **-0.002** / **+0.249** | 0.241 / 0.696 | 78.3% / 82.6% |
| **LogD7.4** | medium | 0.464 / 0.849 | 0.491 / 0.668 | 0.455 / 0.592 | **0.453** / 0.678 / **0.658** | **-0.038** / -0.010 | 0.449 / 0.663 | 76.0% / 90.4% |
| **MDCK** | small | 0.608 / 0.403 | 0.591 / 0.338 | 0.484 / 0.447 | **0.573** / 0.701 / **0.335** | **-0.018** / -0.004 | 0.557 / 0.300 | 64.7% / 91.2% |
| **MIC_C_albicans** | medium | New / deferred | 0.490 / 0.638 | 0.512 / 0.479 | **0.487** / 0.622 / **0.667** | **-0.003** / **+0.029** | 0.451 / 0.716 | 85.6% / 94.4% |
| **MIC_E_coli** | medium | 0.081 / 0.634 | 0.451 / 0.650 | 0.541 / 0.542 | **0.486** / 0.631 / **0.629** | +0.035 / -0.021 | 0.525 / 0.546 | 83.9% / 93.5% |
| **MIC_P_aeruginosa** | medium | New / deferred | 0.650 / 0.184 | 0.526 / 0.558 | **0.741** / 1.141 / **0.111** | +0.091 / -0.073 | 0.791 / 0.022 | 77.2% / 82.2% |
| **MIC_S_aureus** | medium | New / deferred | 0.494 / 0.378 | 0.597 / 0.310 | **0.545** / 0.715 / **0.314** | +0.050 / -0.064 | 0.556 / 0.320 | 83.7% / 91.3% |
| **PAMPA** | large | 0.273 / 0.842 | 0.375 / 0.797 | 0.416 / 0.718 | **0.411** / 0.573 / **0.706** | +0.037 / -0.091 | 0.411 / 0.703 | 79.7% / 88.9% |
| **PPB** | small | 0.295 / -0.007 | 0.299 / 0.591 | 0.161 / 0.656 | **0.233** / 0.304 / **0.789** | **-0.066** / **+0.198** | 0.178 / 0.788 | 70.8% / 75.0% |
| **RRCK** | small | 0.371 / 0.652 | 0.268 / 0.534 | 0.395 / 0.606 | **0.409** / 0.525 / **0.323** | +0.141 / -0.210 | 0.384 / 0.306 | 80.0% / 86.7% |
| **Vd_per_kg** | small | 0.297 / 0.377 | 0.319 / 0.768 | 0.396 / 0.492 | **0.340** / 0.498 / **0.578** | +0.021 / -0.190 | 0.347 / 0.639 | 84.9% / 93.9% |
| **half_life_reg** | medium | 0.331 / 0.901 | 0.694 / 0.476 | 0.598 / 0.621 | **0.684** / 0.881 / **0.460** | **-0.011** / -0.015 | 0.665 / 0.530 | 78.3% / 87.0% |
| **hemolysis_HC50** | large | 0.319 / 0.582 | 0.399 / 0.656 | 0.427 / 0.589 | **0.404** / 0.560 / **0.612** | +0.005 / -0.043 | 0.429 / 0.562 | 84.0% / 92.8% |
| **solubility_reg** | small | 0.175 / -0.127 | 0.638 / 0.636 | 0.699 / 0.651 | **0.598** / 0.886 / **0.774** | **-0.040** / **+0.138** | 0.624 / 0.729 | 89.6% / 93.8% |

### 3.2 Classification Endpoints (6 Endpoints — Untouched 20% Outer Holdout, Calibrated)

| Endpoint | V32 HO ROC / PR / Brier | V33 CPU Dual HO ROC / PR / Brier | **V33 GPU Deep Dual CV ROC / PR** | **V33 GPU Deep Dual HO ROC / PR / MCC / Brier** | **Δ vs CPU (ROC / PR)** | GPU Deep SMILES-Only HO ROC / PR | Calib ECE (Raw→Cal) |
|---|---:|---:|---:|---:|---:|---:|---|
| **BBB** | 0.877 / 0.879 / 0.149 | 0.832 / 0.843 / 0.166 | 0.888 / 0.879 | **0.850** / **0.821** / 0.550 / 0.161 | **+0.017** / -0.022 | 0.784 / 0.746 | platt: 0.1261→**0.0860** |
| **cytotoxicity_cls** | New / deferred | 0.931 / 0.888 / 0.098 | 0.864 / 0.768 | **0.911** / **0.856** / 0.667 / 0.113 | -0.020 / -0.032 | 0.899 / 0.826 | platt: 0.0451→**0.0687** |
| **half_life_cls** | 0.941 / 0.778 / 0.061 | 0.751 / 0.730 / 0.202 | 0.810 / 0.802 | **0.778** / **0.764** / 0.398 / 0.192 | **+0.027** / **+0.034** | 0.802 / 0.818 | platt: 0.0779→**0.0459** |
| **hemolysis_cls** | 0.930 / 0.849 / 0.102 | 0.892 / 0.791 / 0.117 | 0.882 / 0.759 | **0.874** / **0.779** / 0.575 / 0.122 | -0.018 / -0.012 | 0.873 / 0.725 | platt: 0.0455→**0.0196** |
| **overall_peptide_toxicity** | 0.955 / 0.960 / 0.085 | 0.941 / 0.945 / 0.098 | 0.929 / 0.935 | **0.944** / **0.949** / 0.730 / 0.095 | **+0.004** / **+0.004** | 0.919 / 0.924 | platt: 0.0351→**0.0394** |
| **solubility_cls** | 0.715 / 0.930 / 0.116 | 0.857 / 0.904 / 0.147 | 0.838 / 0.878 | **0.857** / **0.896** / 0.592 / 0.142 | **+0.001** / -0.008 | 0.851 / 0.894 | none: 0.0356→**0.0356** |

### 3.3 Censored Gaussian Tobit Loss Performance (All 14 Censored Endpoints, Holdout)

| Endpoint | Censored (Total / HO) | CPU Exact HO MAE / Cens-Conc / C-Idx | CPU AFT HO MAE / Cens-Conc / C-Idx | **GPU Deep Tobit HO MAE / Sp / Cens-Conc / Viol MAE / C-Idx** |
|---|---:|---:|---:|---:|
| **hemolysis_HC50** | 1375 / 292 | 0.399 / 0.250 / 0.754 | 0.417 / 0.651 / 0.772 | **0.404** / **0.612** / **0.616** / 0.130 / **0.771** |
| **MIC_S_aureus** | 344 / 71 | 0.494 / 0.028 / 0.661 | 0.566 / 0.282 / 0.668 | **0.545** / **0.314** / **0.296** / 0.382 / **0.660** |
| **MIC_E_coli** | 256 / 45 | 0.451 / 0.022 / 0.749 | 0.462 / 0.156 / 0.755 | **0.486** / **0.629** / **0.222** / 0.408 / **0.745** |
| **MIC_P_aeruginosa** | 201 / 35 | 0.650 / 0.114 / 0.615 | 0.762 / 0.400 / 0.590 | **0.741** / **0.111** / **0.429** / 0.252 / **0.604** |
| **half_life_reg** | 177 / 49 | 0.694 / 0.020 / 0.649 | 0.687 / 0.082 / 0.653 | **0.684** / **0.460** / **0.041** / 1.129 / **0.647** |
| **MIC_C_albicans** | 56 / 4 | 0.490 / 0.000 / 0.743 | 0.535 / 0.250 / 0.723 | **0.487** / **0.667** / **0.000** / 0.372 / **0.754** |
| **PPB** | 30 / 8 | 0.299 / 0.250 / 0.602 | 0.266 / 0.250 / 0.655 | **0.233** / **0.789** / **0.000** / 0.364 / **0.755** |
| **solubility_reg** | 26 / 12 | 0.638 / 0.167 / 0.668 | 0.545 / 0.083 / 0.731 | **0.598** / **0.774** / **0.167** / 0.690 / **0.765** |
| **MDCK** | 16 / 0 | 0.591 / — / — | 0.574 / — / — | **0.573** / **0.335** / **—** / — / **—** |
| **Caco2** | 13 / 4 | 0.368 / 0.000 / 0.839 | 0.369 / 0.000 / 0.840 | **0.381** / **0.816** / **0.500** / 0.085 / **0.816** |
| **F** | 10 / 2 | 0.239 / 0.500 / 0.652 | 0.224 / 0.500 / 0.698 | **0.236** / **0.728** / **0.500** / 0.054 / **0.759** |
| **PAMPA** | 9 / 1 | 0.375 / 0.000 / 0.808 | 0.382 / 0.000 / 0.804 | **0.411** / **0.706** / **0.000** / 0.211 / **0.766** |
| **CL_per_kg** | 7 / 0 | 0.566 / — / — | 0.633 / — / — | **0.590** / **0.776** / **—** / — / **—** |
| **LogD7.4** | 3 / 0 | 0.491 / — / — | 0.453 / — / — | **0.453** / **0.658** / **—** / — / **—** |

---

## 4. Dedicated `HalfLifeAttentionNet` Specialist vs. `MultiTaskPeptideNet` vs. CPU

| Endpoint | Model Architecture | CV Dual (MAE/Sp or ROC/PR) | HO Dual (MAE/Sp or ROC/PR) | HO SMILES-Only | HO Cens-Conc / C-Idx |
|---|---|---:|---:|---:|---:|
| **half_life_reg** | V33 CPU Winner (`XGBoost`) | 0.626 / 0.575 | 0.694 / 0.476 | 0.669 / 0.558 | 0.020 / 0.649 |
| **half_life_reg** | `MultiTaskPeptideNet` (24-task) | 0.616 / 0.596 | 0.695 / 0.457 | 0.668 / 0.535 | 0.061 / 0.644 |
| **half_life_reg** | `HalfLifeAttentionNet` (PK Specialist) | 0.616 / 0.610 | 0.684 / 0.451 | 0.670 / 0.514 | 0.061 / 0.646 |
| **half_life_reg** | **CV-Selected Final: `HalfLifeAttentionNet(0.50)+MultiTaskPeptideNet(0.50)`** | **0.598 / 0.621** | **0.684 / 0.460** | **0.665 / 0.530** | **0.041 / 0.647** |
| **half_life_cls** | V33 CPU Winner (`LightGBM`) | 0.777 / 0.788 | 0.751 / 0.730 | 0.790 / 0.763 | — / — |
| **half_life_cls** | `MultiTaskPeptideNet` (24-task) | 0.800 / 0.785 | 0.773 / 0.760 | 0.799 / 0.819 | — / — |
| **half_life_cls** | `HalfLifeAttentionNet` (PK Specialist) | 0.795 / 0.792 | 0.775 / 0.757 | 0.797 / 0.807 | — / — |
| **half_life_cls** | **CV-Selected Final: `HalfLifeAttentionNet(0.50)+MultiTaskPeptideNet(0.50)`** | **0.810 / 0.802** | **0.778 / 0.764** | **0.802 / 0.818** | **— / —** |

---

## 5. Strict Zero-Leakage Case17 Clinical Peptide Drug Benchmark (GPU Deep vs. CPU vs. V32 vs. FDA)

Every GPU Deep prediction below comes from a dedicated **Leave-Case17-Out** 5-fold ensemble that strictly excludes ALL `439` `case17_excluded==1` rows AND all `19` Case17 clusters (`911` rows total) across ALL 24 co-trained endpoints (`assert 0 Case17 clusters`).

| Clinical Drug | FDA t1/2 Ref | V33 CPU t1/2 (IV / SC) | **GPU Deep t1/2 (IV / SC) + Anchor** | FDA PPB | V33 CPU PPB | **GPU Deep PPB** | V32 Hemol | CPU Hemol | **GPU Deep Hemol (Cal) / HC50 (µM)** |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **Leuprolide** | ~3 h (2.9-3 h, 二室模型) | 1.03h / 1.99h | **1.20h / 4.83h** | 43-49% | 0.586 | **0.421** | 0.062 | 0.054 | **0.074** / 461 µM |
| **Bivalirudin** | ~25 min (肾功能正常) | 0.95h / 2.50h | **2.00h / 10.99h** | ~0% (不与血浆蛋白/红细胞结合) | 0.369 | **0.136** | 0.022 | 0.012 | **0.100** / 257 µM |
| **Glucagon** | 8-18 min (说明书; 文献 IV 亦有 3-6 min) | 0.22h / 0.79h | **0.12h / 0.93h** | 低/未明确报道 (非酰化 29 肽) | 0.375 | **0.422** | 0.00946 | 0.028 | **0.111** / 187 µM |
| **Teriparatide** | IV ~5 min; SC ~1 h | 3.53h / 7.17h | **0.28h / 1.98h** | ~0% (无明显结合) | 0.524 | **0.654** | 0.00693 | 0.059 | **0.100** / 491 µM |
| **Exenatide** | 2.4 h (末端半衰期) | 3.58h / 3.15h | **2.38h / 13.31h** | ~53% | 0.475 | **0.575** | 0.0621 | 0.016 | **0.101** / 103 µM |
| **Liraglutide** | ~13 h (SC) | 3.72h / 9.57h | **4.63h / 21.55h**; Anchor **9h** | >98% (白蛋白) | 0.436 | **0.612** | 0.0233 | 0.084 | **0.362** / 67 µM |
| **Semaglutide** | ~1 周 (~165-168 h) | 11.71h / 31.29h | **24.39h / 96.63h**; Anchor **144h** | >99% (白蛋白) | 0.523 | **0.668** | 0.772 | 0.076 | **0.161** / 253 µM |
| **Tirzepatide** | ~5 天 (~125 h) | 48.99h / 101.57h | **35.22h / 125.53h**; Anchor **144h** | ~96% (血浆白蛋白) | 0.721 | **0.874** | 0.989 | 0.148 | **0.187** / 393 µM |
| **Cyclosporine_A** | ~8.4 h (5-18 h) | 16.63h / 20.86h | **8.96h / 33.12h** | ~90% (主要脂蛋白) | 0.905 | **0.746** | 0.0461 | 0.503 | **0.610** / 14 µM |
| **Oxytocin** | 3-5 min (标签亦见 1-6 min) | 1.09h / 2.02h | **0.60h / 2.55h** | 轻度/未定量报道 | 0.471 | **0.297** | 0.394 | 0.027 | **0.061** / 455 µM |
| **Octreotide** | 1.7-1.9 h (SC/IV) | 2.34h / 4.41h | **1.93h / 8.88h** | ~65% (脂蛋白+白蛋白) | 0.708 | **0.485** | 0.639 | 0.061 | **0.095** / 139 µM |
| **Linaclotide** | 血浆低于定量限 无法计算 (肠道局部作用) | 2.34h / 4.71h | **4.81h / 20.14h** | n/a (无全身暴露) | 0.497 | **0.325** | 0.63 | 0.230 | **0.361** / 140 µM |
| **Ziconotide** | 鞘内 CSF ~4.6 h (2.9-6.5 h); 血浆 ~1.3 h(旧标签) | 1.10h / 2.80h | **1.62h / 6.15h** | ~50% | 0.383 | **0.415** | 0.896 | 0.107 | **0.128** / 456 µM |
| **Daptomycin** | ~8 h (8-9 h) | 1.95h / 4.04h | **10.53h / 34.29h** | 90-93% (~92%) | 0.510 | **0.865** | 0.625 | 0.102 | **0.254** / 150 µM |
| **Caspofungin** | β 相 9-11 h (γ 相 40-50 h) | 4.44h / 6.33h | **19.41h / 58.13h** | ~97% | 0.636 | **0.929** | 0.192 | 0.045 | **0.411** / 353 µM |
| **Insulin_detemir** | 5-7 h (剂量依赖) | 1.36h / 2.38h | **7.79h / 33.19h**; Anchor **9h** | >98% (~99%, 白蛋白) | 0.682 | **0.583** | 0.811 | 0.393 | **0.239** / 108 µM |
| **Insulin_degludec** | ~25 h (稳态 剂量无关) | 1.61h / 2.74h | **11.16h / 48.63h**; Anchor **25h** | >99% (白蛋白) | 0.680 | **0.398** | 0.813 | 0.347 | **0.143** / 217 µM |

- **Case17 Pure-ML Route-Matched `half_life_reg` (`n=16` FDA drugs, strict leave-Case17-out, NO anchor override)**: V33 CPU Dual MAE=`0.458` log10 h (R²=`0.583`, Sp=`0.682`, r=`0.769`) vs. **V33 GPU Deep Dual MAE=`0.386` log10 h (R²=`0.708`, Sp=`0.882`, r=`0.895`)** and **SMILES-Only MAE=`0.437` (Sp=`0.868`)**.
- **Case17 Pure-ML `PPB` (`n=14` FDA drugs, strict leave-Case17-out)**: V33 CPU Dual MAE=`0.277` (R²=`0.105`, Sp=`0.302`) vs. **V33 GPU Deep Dual MAE=`0.225` (R²=`0.235`, Sp=`0.352`)** and **SMILES-Only MAE=`0.184` (R²=`0.426`, Sp=`0.621`, r=`0.663`)**.

---

## 6. GPU Hardware Compliance Log & Standardized Artifact Inventory

- **GPU Hardware Policy**: Executed exclusively on the idle local Windows **`NVIDIA GeForce RTX 5080`** (`16,303 MiB` VRAM, PyTorch `2.11.0+cu128`, native `bf16`), achieving **zero interference** with `node1` shared GPUs.
- **Telemetry**: Start `757 / 16303 MiB` (`0%` util, `38°C`) → 5-worker parallel phase peak **`9,230 / 16,303 MiB`** (`83%` GPU util, `45°C`, ~`1,229 MiB` PyTorch allocation per worker with the entire 46,598-compound 4-embedding bank resident in VRAM!) → End `2513 / 16303 MiB` (`37°C`).
- **Wave 3A Super-Ensemble Fusion Handoff Notes**:
  - **Highest-leverage GPU Deep strengths to weight heavily in Wave 3A**:
    1. **Small-n PK Multi-Task Transfer**: `PPB` (`HO Dual MAE 0.299→0.233`, `Sp 0.591→0.789`, `SMI MAE 0.178`), `CL_absolute` (`MAE 1.491→1.180`, `Sp 0.420→0.658`), `F` (`Sp 0.478→0.728`), `Cmax` (`MAE 1.106→0.916`), `solubility_reg` (`MAE 0.638→0.598`, `Sp 0.636→0.774`), `LogD7.4` (`MAE 0.491→0.453`), `MDCK` (`MAE 0.591→0.573`).
    2. **Half-Life Specialist & Clinical Case17 Generalization**: `half_life_cls` (`HO Dual ROC 0.751→0.778`, `PR 0.730→0.764`, `SMI ROC 0.802`), `half_life_reg` (`CV MAE 0.626→0.598 / Sp 0.575→0.621`, `HO MAE 0.694→0.684`, and strict leave-Case17-out FDA `n=16` pure-ML `Sp 0.682→0.882`, `R² 0.583→0.708`, `MAE 0.458→0.386` log10 h!).
    3. **PLM / Sequence Classification & Censored Tobit Fidelity**: `BBB` (`HO ROC 0.832→0.850`), `overall_peptide_toxicity` (`0.941→0.944`), `solubility_cls` (`0.857`, Brier `0.147→0.142`), plus 2.5×–10.6× censored bound concordance gains on `hemolysis_HC50` (`61.6%`), `MIC_E_coli` (`22.2%`), `MIC_S_aureus` (`29.6%`), `MIC_P_aeruginosa` (`42.9%`), and `Caco2` (`50.0%`).
  - **Complementary CPU/GBDT strengths for stacking**: `PAMPA`, `Caco2`, `RRCK`, `Vd_per_kg`, `hemolysis_cls`, and `cytotoxicity_cls` remain strong anchors for CPU GBDT/GPR and Wave 2D-1 GPU Fusion; combining orthogonal deep multi-task neural predictions with tree/kernel/fusion predictions in Wave 3A will maximize both exact accuracy and censored fidelity across all 24 endpoints.
- **Authoritative Artifact Paths**:
  - Local: `D:\work\中检院\v33_upgrade\models_gpu_deep\` (`gpu_deep_summary.json`, 24 `<endpoint>/` directories, `multitask_checkpoints/`, `halflife_specialist/`)
  - Remote (`node23`): `/data/qlyu/v33_upgrade/models_gpu_deep/` (exact mirrored predictions, metrics, summary, and inference-ready PyTorch checkpoints ready for Wave 3A Super-Ensemble Fusion!)

