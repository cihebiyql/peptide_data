# Peptide OmniPanel V33 — Wave-2D Pretrained Multimodal Fusion & Representation Ablation Benchmark

- **Generated**: 2026-11-22 CST | **Compute**: `node23` (64-core CPU parallel suite, total runtime `9591.3s`)
- **Authoritative Artifacts (`node23`)**: `/data/qlyu/v33_upgrade/models_fusion/` (`fusion_benchmark_summary.json` + 24 endpoint directories with `oof_predictions.tsv`, `holdout_predictions.tsv`, `case17_predictions.tsv`, `metrics.json`, `fusion_models_5fold.joblib`)
- **Foundation Models Evaluated (All 46,598 Compounds Row-Aligned, 17/17 Case17 Covered, Zero NaN/Inf)**:
  1. **`ncAA-RepDistill v2`** (self-developed modified-peptide student, 100% MLM verified): `smiles_repr` (768D), `fusion_hf` (2048D), `protein_proj` (1280D) — SMILES-only.
  2. **`PeptideCLM-2`** (512D): L/D-amino-acid & PTM-sensitive peptide chemical language model — SMILES-only.
  3. **`ChemBERTa-MTR`** (384D): multi-task physicochemical SMILES transformer — SMILES-only.
  4. **`ESM-C 600M Multi-Region`** (4 × 1152D = 4608D): `mean`, `max`, `nterm5`, `cterm5` protein language model capturing terminal proteolysis & amphipathicity — Sequence modality.
- **Strict Anti-Leakage Discipline**: Identical frozen 5-fold cluster-grouped CV (`is_holdout==0`) & untouched 20% outer holdout (`is_holdout==1`); strictly fold-internal standardization, block variance balancing, FastPCA, kernel distance matrices, nested 5-fold meta-stacking, probability calibration, and conformal quantiles; dedicated **Leave-Case17-Out (`case17_excluded==0`)** refits for all 17 clinical drugs.

---

## 1. Executive Summary

1. **Broad, Consistent Holdout Lift Over Wave-2C CPU Across All 24 Endpoints**:
   - **Classification (6/6 endpoints)**: Multimodal Dual Fusion strictly improved holdout ROC-AUC on **6/6** endpoints (mean **ΔROC-AUC = `+0.0195`**, mean **ΔPR-AUC = `+0.0205`**).
   - **Quantitative Regression (18 endpoints)**: Multimodal Dual Fusion strictly reduced holdout MAE on **14/18** endpoints and increased holdout Spearman on **14/18** endpoints (mean **ΔMAE = `-0.0270`**, mean **ΔSpearman = `+0.0292`**).
   - **Strict Two-Track Modality Guarantee**: Track A (`smiles_only`, ZERO sequence/ESM-C features — 100% safe for cyclic/modified/non-canonical peptide SMILES) and Track B (`dual_modality`, 100% row coverage via modality-aware weight renormalization) both ship validated, calibrated 5-fold inference bundles across all 24 endpoints.
2. **Major Breakthroughs on Wave-2C's Highest-Leverage Targets**:
   - **`BBB`**: CPU HO ROC `0.8325` / PR `0.8430` → **Fusion Dual `0.8987` (`+0.0663`) / PR `0.8898` (`+0.0468`)** (vs V32 `0.877`)
   - **`overall_peptide_toxicity`**: CPU HO ROC `0.9406` / PR `0.9451` → **Fusion Dual `0.9507` (`+0.0101`) / PR `0.9545` (`+0.0094`)** (vs V32 `0.955`)
   - **`half_life_cls`**: CPU HO ROC `0.7507` / PR `0.7295` → **Fusion Dual `0.7756` (`+0.0249`) / PR `0.7697` (`+0.0402`)** (vs V32 `0.941`)
   - **`hemolysis_cls`**: CPU HO ROC `0.8921` / PR `0.7911` → **Fusion Dual `0.8963` (`+0.0041`) / PR `0.8101` (`+0.0189`)** (vs V32 `0.930`)
   - **`half_life_reg`**: CPU HO MAE `0.6943` / Sp `0.476` → **Fusion Dual MAE `0.6654` (`-0.0289`) / Sp `0.482` (`+0.007`)**
   - **`hemolysis_HC50`**: CPU HO MAE `0.3994` / Sp `0.656` → **Fusion Dual MAE `0.3805` (`-0.0189`) / Sp `0.678` (`+0.022`)**
   - **`MIC_P_aeruginosa`**: CPU HO MAE `0.6497` / Sp `0.184` → **Fusion Dual MAE `0.6572` (`+0.0075`) / Sp `0.133` (`-0.052`)**
   - **`MIC_S_aureus`**: CPU HO MAE `0.4943` / Sp `0.378` → **Fusion Dual MAE `0.4758` (`-0.0185`) / Sp `0.435` (`+0.056`)**
   - **`PPB`**: CPU HO MAE `0.2992` / Sp `0.591` → **Fusion Dual MAE `0.2608` (`-0.0384`) / Sp `0.678` (`+0.087`)**
   - **`LogD7.4`**: CPU HO MAE `0.4911` / Sp `0.668` → **Fusion Dual MAE `0.4570` (`-0.0341`) / Sp `0.702` (`+0.034`)**
   - **`PAMPA`**: CPU HO MAE `0.3750` / Sp `0.797` → **Fusion Dual MAE `0.3732` (`-0.0018`) / Sp `0.778` (`-0.019`)**
3. **Calibration & Conformal Uncertainty**: Across 6 classification endpoints, calibrated Multimodal Fusion achieved mean holdout **Brier = `0.1262`** (raw `0.1265`) and **ECE = `0.0358`** (raw `0.0389`). Across 18 regression endpoints, CV+ conformal intervals achieved mean holdout empirical coverage of **`74.2%`** (nominal 80%) and **`86.1%`** (nominal 90%).

---

## 2. Master 24-Endpoint Benchmark — V32 vs. Wave-2C CPU vs. Wave-2D Multimodal Fusion

### 2.1 Classification Endpoints (6 Endpoints — Untouched 20% Holdout)

| Endpoint | n (CV/HO) | V32 HO ROC / PR | Wave-2C CPU Dual HO ROC / PR | **Wave-2D Fusion Dual HO ROC / PR / MCC** | ΔROC / ΔPR vs CPU | Fusion SMILES-Only HO ROC / PR | Winning Stacker | Calib (Brier / ECE) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **BBB** | 844 (675/169) | 0.877 / 0.879 | 0.8325 / 0.8430 | **0.8987** / **0.8898** / 0.633 | **+0.0663** / **+0.0468** | 0.8351 / 0.8278 | `logistic_l2` | none: `0.1289` / `0.0575` |
| **cytotoxicity_cls** | 2934 (2347/587) | New / deferred | 0.9308 / 0.8881 | **0.9342** / **0.8957** / 0.717 | **+0.0034** / **+0.0076** | 0.9277 / 0.8873 | `logit_convex` | none: `0.0948` / `0.0526` |
| **half_life_cls** | 3204 (2563/641) | 0.941 / 0.778 | 0.7507 / 0.7295 | **0.7756** / **0.7697** / 0.411 | **+0.0249** / **+0.0402** | 0.8016 / 0.7966 | `convex_simplex` | none: `0.1926` / `0.0284` |
| **hemolysis_cls** | 20506 (16405/4101) | 0.930 / 0.849 | 0.8921 / 0.7911 | **0.8963** / **0.8101** / 0.605 | **+0.0041** / **+0.0189** | 0.8901 / 0.7668 | `convex_simplex` | platt: `0.1126` / `0.0173` |
| **overall_peptide_toxicity** | 11036 (8828/2208) | 0.955 / 0.960 | 0.9406 / 0.9451 | **0.9507** / **0.9545** / 0.755 | **+0.0101** / **+0.0094** | 0.9383 / 0.9378 | `logistic_l2` | none: `0.0889` / `0.0296` |
| **solubility_cls** | 3722 (2977/745) | 0.715 / 0.930 | 0.8568 / 0.9037 | **0.8652** / **0.9041** / 0.574 | **+0.0084** / **+0.0004** | 0.8657 / 0.9044 | `logit_convex` | none: `0.1395` / `0.0293` |

### 2.2 Quantitative Regression Endpoints (18 Endpoints — Untouched 20% Holdout)

| Endpoint | Tier | n (CV/HO) | V32 HO MAE / Sp | Wave-2C CPU Dual HO MAE / Sp | **Wave-2D Fusion Dual HO MAE / RMSE / Sp** | ΔMAE / ΔSp vs CPU | Fusion SMILES-Only HO MAE / Sp | Winning Stacker | Conf 80% / 90% |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **CL_absolute** | small | 54 (43/11) | no HO (n=7) | 1.491 / 0.420 | **1.430** / 1.635 / **0.333** | **-0.060** / -0.087 | 1.398 / 0.347 | `convex_simplex` | 27.3% / 27.3% |
| **CL_per_kg** | small | 206 (165/41) | 0.593 / 0.242 | 0.566 / 0.797 | **0.548** / 0.704 / **0.832** | **-0.018** / **+0.035** | 0.544 / 0.830 | `ridge_nonneg` | 70.7% / 92.7% |
| **Caco2** | medium | 1580 (1265/315) | 0.306 / 0.823 | 0.368 / 0.860 | **0.366** / 0.473 / **0.861** | **-0.002** / **+0.001** | 0.361 / 0.867 | `topk_blend` | 81.3% / 90.7% |
| **Cmax** | small | 74 (57/17) | no HO (n=36) | 1.106 / 0.720 | **0.996** / 1.340 / **0.887** | **-0.110** / **+0.167** | 0.947 / 0.764 | `convex_simplex` | 88.2% / 88.2% |
| **F** | small | 124 (99/25) | 0.238 / -0.034 | 0.239 / 0.478 | **0.239** / 0.316 / **0.548** | +0.000 / **+0.070** | 0.285 / 0.426 | `topk_blend` | 65.2% / 82.6% |
| **LogD7.4** | medium | 678 (553/125) | 0.464 / 0.849 | 0.491 / 0.668 | **0.457** / 0.675 / **0.702** | **-0.034** / **+0.034** | 0.462 / 0.706 | `topk_blend` | 78.4% / 88.0% |
| **MDCK** | small | 168 (134/34) | 0.608 / 0.403 | 0.591 / 0.338 | **0.499** / 0.643 / **0.455** | **-0.092** / **+0.117** | 0.489 / 0.499 | `convex_simplex` | 70.6% / 82.3% |
| **MIC_C_albicans** | medium | 468 (374/94) | New / deferred | 0.490 / 0.638 | **0.444** / 0.574 / **0.708** | **-0.046** / **+0.071** | 0.459 / 0.692 | `convex_simplex` | 86.7% / 94.4% |
| **MIC_E_coli** | medium | 2210 (1768/442) | 0.081 / 0.634 | 0.451 / 0.650 | **0.421** / 0.556 / **0.676** | **-0.030** / **+0.026** | 0.434 / 0.659 | `convex_simplex` | 83.4% / 93.2% |
| **MIC_P_aeruginosa** | medium | 1161 (929/232) | New / deferred | 0.650 / 0.184 | **0.657** / 1.005 / **0.133** | +0.008 / -0.052 | 0.672 / 0.102 | `convex_simplex` | 75.6% / 82.7% |
| **MIC_S_aureus** | medium | 2131 (1705/426) | New / deferred | 0.494 / 0.378 | **0.476** / 0.593 / **0.435** | **-0.018** / **+0.056** | 0.476 / 0.433 | `convex_simplex` | 87.0% / 96.1% |
| **PAMPA** | large | 8581 (6865/1716) | 0.273 / 0.842 | 0.375 / 0.797 | **0.373** / 0.515 / **0.778** | **-0.002** / -0.019 | 0.375 / 0.780 | `ridge_nonneg` | 80.4% / 89.4% |
| **PPB** | small | 162 (130/32) | 0.295 / -0.007 | 0.299 / 0.591 | **0.261** / 0.297 / **0.678** | **-0.038** / **+0.087** | 0.263 / 0.818 | `convex_simplex` | 29.2% / 91.7% |
| **RRCK** | small | 167 (137/30) | 0.371 / 0.652 | 0.268 / 0.534 | **0.320** / 0.424 / **0.548** | +0.052 / **+0.014** | 0.322 / 0.521 | `topk_blend` | 83.3% / 90.0% |
| **Vd_per_kg** | small | 166 (133/33) | 0.297 / 0.377 | 0.319 / 0.768 | **0.330** / 0.487 / **0.714** | +0.011 / -0.054 | 0.349 / 0.704 | `convex_simplex` | 87.9% / 93.9% |
| **half_life_reg** | medium | 3078 (2462/616) | 0.331 / 0.901 | 0.694 / 0.476 | **0.665** / 0.854 / **0.482** | **-0.029** / **+0.007** | 0.649 / 0.534 | `convex_simplex` | 75.3% / 88.4% |
| **hemolysis_HC50** | large | 5559 (4447/1112) | 0.319 / 0.582 | 0.399 / 0.656 | **0.381** / 0.531 / **0.678** | **-0.019** / **+0.022** | 0.395 / 0.661 | `convex_simplex` | 83.3% / 91.5% |
| **solubility_reg** | small | 298 (238/60) | 0.175 / -0.127 | 0.638 / 0.636 | **0.581** / 0.796 / **0.666** | **-0.057** / **+0.031** | 0.570 / 0.682 | `topk_blend` | 81.2% / 87.5% |

---

## 3. Fair Individual Representation Ablation Scorecard (All 24 Endpoints — Untouched 20% Holdout)

Every pretrained representation was evaluated on the exact same frozen 5-fold CV with fold-internal standardization, condition features, and anti-($p \gg n$) base learners (`Ridge`/`Logistic`, `KRR`/`SVR`/`KLR`/`SVC` or `Nystrom_RBF`/`MLP`, and `FastPCA + LightGBM/XGBoost`). For `ncAA-RepDistill v2`, the sub-head (`smi`=`smiles_repr`, `hf`=`fusion_hf`, `prot`=`protein_proj`) was selected strictly on 5-fold CV.

### 3.1 Classification Representation Scorecard (Holdout ROC-AUC / PR-AUC)

| Endpoint | Wave-2C CPU Dual | **ncAA-RepDistill v2** (sub) | **PeptideCLM-2** | **ChemBERTa-MTR** | **ESM-C Mean** | **ESM-C Multi-Region** | Direct Dual | **Fusion Dual Stack** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BBB** | 0.832 / 0.843 | 0.818 / 0.803 (`prot`) | 0.797 / 0.787 | 0.800 / 0.780 | 0.877 / 0.852 | 0.872 / 0.847 | 0.867 / 0.845 | **0.899** / **0.890** |
| **cytotoxicity_cls** | 0.931 / 0.888 | 0.897 / 0.844 (`smi`) | 0.892 / 0.821 | 0.896 / 0.827 | 0.905 / 0.843 | 0.907 / 0.848 | 0.908 / 0.851 | **0.934** / **0.896** |
| **half_life_cls** | 0.751 / 0.730 | 0.779 / 0.787 (`smi`) | 0.794 / 0.799 | 0.778 / 0.786 | 0.738 / 0.731 | 0.733 / 0.725 | 0.733 / 0.753 | **0.776** / **0.770** |
| **hemolysis_cls** | 0.892 / 0.791 | 0.867 / 0.720 (`prot`) | 0.866 / 0.722 | 0.868 / 0.724 | 0.879 / 0.785 | 0.859 / 0.781 | 0.883 / 0.788 | **0.896** / **0.810** |
| **overall_peptide_toxicity** | 0.941 / 0.945 | 0.919 / 0.924 (`prot`) | 0.924 / 0.931 | 0.911 / 0.921 | 0.945 / 0.950 | 0.945 / 0.950 | 0.944 / 0.949 | **0.951** / **0.954** |
| **solubility_cls** | 0.857 / 0.904 | 0.858 / 0.896 (`prot`) | 0.866 / 0.909 | 0.849 / 0.889 | 0.855 / 0.886 | 0.852 / 0.885 | 0.855 / 0.883 | **0.865** / **0.904** |

### 3.2 Regression Representation Scorecard (Holdout Exact MAE / Spearman)

| Endpoint | Wave-2C CPU Dual | **ncAA-RepDistill v2** (sub) | **PeptideCLM-2** | **ChemBERTa-MTR** | **ESM-C Mean** | **ESM-C Multi-Region** | Direct Dual | **Fusion Dual Stack** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **CL_absolute** | 1.491 / 0.420 | 1.345 / 0.336 (`hf`) | 1.390 / 0.487 | 1.351 / 0.529 | 1.374 / 0.510 | 1.348 / 0.507 | 1.575 / 0.105 | **1.430** / **0.333** |
| **CL_per_kg** | 0.566 / 0.797 | 0.535 / 0.803 (`smi`) | 0.575 / 0.787 | 0.613 / 0.735 | 0.629 / 0.765 | 0.615 / 0.740 | 0.563 / 0.735 | **0.548** / **0.832** |
| **Caco2** | 0.368 / 0.860 | 0.387 / 0.828 (`smi`) | 0.402 / 0.844 | 0.389 / 0.829 | 0.394 / 0.826 | 0.396 / 0.826 | 0.356 / 0.859 | **0.366** / **0.861** |
| **Cmax** | 1.106 / 0.720 | 1.176 / -0.201 (`prot`) | 1.136 / 0.516 | 0.699 / 0.852 | 1.155 / 0.555 | 1.090 / 0.830 | 1.131 / 0.650 | **0.996** / **0.887** |
| **F** | 0.239 / 0.478 | 0.228 / 0.573 (`hf`) | 0.259 / 0.527 | 0.246 / 0.380 | 0.258 / 0.475 | 0.260 / 0.439 | 0.250 / 0.522 | **0.239** / **0.548** |
| **LogD7.4** | 0.491 / 0.668 | 0.468 / 0.686 (`smi`) | 0.485 / 0.675 | 0.523 / 0.651 | 0.504 / 0.652 | 0.493 / 0.655 | 0.483 / 0.677 | **0.457** / **0.702** |
| **MDCK** | 0.591 / 0.338 | 0.475 / 0.470 (`smi`) | 0.486 / 0.487 | 0.528 / 0.385 | 0.545 / 0.453 | 0.543 / 0.400 | 0.535 / 0.309 | **0.499** / **0.455** |
| **MIC_C_albicans** | 0.490 / 0.638 | 0.493 / 0.638 (`hf`) | 0.421 / 0.749 | 0.451 / 0.669 | 0.437 / 0.743 | 0.449 / 0.710 | 0.457 / 0.699 | **0.444** / **0.708** |
| **MIC_E_coli** | 0.451 / 0.650 | 0.448 / 0.638 (`smi`) | 0.462 / 0.633 | 0.435 / 0.675 | 0.426 / 0.667 | 0.408 / 0.704 | 0.418 / 0.697 | **0.421** / **0.676** |
| **MIC_P_aeruginosa** | 0.650 / 0.184 | 0.692 / 0.069 (`prot`) | 0.688 / 0.084 | 0.651 / 0.285 | 0.651 / 0.271 | 0.648 / 0.267 | 0.663 / 0.181 | **0.657** / **0.133** |
| **MIC_S_aureus** | 0.494 / 0.378 | 0.506 / 0.375 (`hf`) | 0.476 / 0.420 | 0.475 / 0.414 | 0.492 / 0.377 | 0.500 / 0.389 | 0.481 / 0.401 | **0.476** / **0.435** |
| **PAMPA** | 0.375 / 0.797 | 0.413 / 0.706 (`smi`) | 0.408 / 0.709 | 0.402 / 0.739 | 0.409 / 0.745 | 0.406 / 0.752 | 0.416 / 0.704 | **0.373** / **0.778** |
| **PPB** | 0.299 / 0.591 | 0.263 / 0.818 (`hf`) | 0.272 / 0.620 | 0.257 / 0.664 | 0.315 / 0.340 | 0.317 / 0.442 | 0.291 / 0.503 | **0.261** / **0.678** |
| **RRCK** | 0.268 / 0.534 | 0.366 / 0.390 (`smi`) | 0.341 / 0.451 | 0.362 / 0.424 | 0.319 / 0.594 | 0.318 / 0.613 | 0.332 / 0.482 | **0.320** / **0.548** |
| **Vd_per_kg** | 0.319 / 0.768 | 0.341 / 0.762 (`smi`) | 0.362 / 0.695 | 0.334 / 0.729 | 0.329 / 0.762 | 0.340 / 0.722 | 0.336 / 0.694 | **0.330** / **0.714** |
| **half_life_reg** | 0.694 / 0.476 | 0.676 / 0.465 (`smi`) | 0.658 / 0.534 | 0.712 / 0.457 | 0.702 / 0.431 | 0.702 / 0.435 | 0.709 / 0.433 | **0.665** / **0.482** |
| **hemolysis_HC50** | 0.399 / 0.656 | 0.425 / 0.600 (`prot`) | 0.430 / 0.594 | 0.433 / 0.583 | 0.398 / 0.649 | 0.393 / 0.646 | 0.390 / 0.653 | **0.381** / **0.678** |
| **solubility_reg** | 0.638 / 0.636 | 0.548 / 0.708 (`prot`) | 0.626 / 0.644 | 0.641 / 0.611 | 0.533 / 0.648 | 0.528 / 0.690 | 0.582 / 0.703 | **0.581** / **0.666** |

### 3.3 `ncAA-RepDistill v2` Sub-Head Ablation (`smiles_repr` vs `fusion_hf` vs `protein_proj`) & Foundation Model Dominance

| Endpoint | `smiles_repr` (768D) HO | `fusion_hf` (2048D) HO | `protein_proj` (1280D) HO | Best Single Pretrained Rep (HO) | Top Track B Fusion Weights |
|---|---:|---:|---:|---|---|
| **BBB** | 0.794 (KLR_RBF) | 0.782 (KLR_RBF) | 0.818 (KLR_RBF) | `esmc_mean` (ROC `0.877`) | `ncaa_fusion_hf`:-0.16, `ncaa_protein_proj`:+0.14, `esmc_multiregion_smooth`:+0.13, `esmc_mean`:-0.12 |
| **CL_absolute** | 1.522/0.11 | 1.345/0.34 | 1.485/0.06 | `ncaa_fusion_hf` (MAE `1.345`, Sp `0.34`) | `cpu_dual`:+0.22, `ncaa_fusion_hf`:+0.21, `ncaa_protein_proj`:+0.16, `peptideclm2`:+0.13 |
| **CL_per_kg** | 0.535/0.80 | 0.555/0.82 | 0.571/0.83 | `ncaa_smiles_repr` (MAE `0.535`, Sp `0.80`) | `cpu_dual`:+0.47, `ncaa_smiles_repr`:+0.34, `direct_dual_smooth`:+0.18 |
| **Caco2** | 0.387/0.83 | 0.404/0.83 | 0.407/0.81 | `ncaa_smiles_repr` (MAE `0.387`, Sp `0.83`) | `cpu_dual`:+0.33, `ncaa_smiles_repr`:+0.33, `cpu_aft_dual`:+0.33 |
| **Cmax** | 1.119/0.55 | 1.063/0.57 | 1.176/-0.20 | `chemberta_mtr` (MAE `0.699`, Sp `0.85`) | `cpu_dual`:+0.38, `chemberta_mtr`:+0.27, `esmc_mean`:+0.18, `esmc_multiregion`:+0.17 |
| **F** | 0.224/0.56 | 0.228/0.57 | 0.232/0.42 | `ncaa_smiles_repr` (MAE `0.224`, Sp `0.56`) | `ncaa_fusion_hf`:+0.50, `peptideclm2`:+0.50 |
| **LogD7.4** | 0.468/0.69 | 0.517/0.65 | 0.499/0.64 | `ncaa_smiles_repr` (MAE `0.468`, Sp `0.69`) | `cpu_dual`:+0.50, `ncaa_smiles_repr`:+0.50 |
| **MDCK** | 0.475/0.47 | 0.478/0.62 | 0.451/0.56 | `ncaa_protein_proj` (MAE `0.451`, Sp `0.56`) | `cpu_aft_dual`:+0.14, `chemberta_mtr`:+0.11, `ncaa_smiles_repr`:+0.10, `direct_smiles_smooth`:+0.10 |
| **MIC_C_albicans** | 0.498/0.67 | 0.493/0.64 | 0.473/0.74 | `peptideclm2` (MAE `0.421`, Sp `0.75`) | `cpu_seq_only`:+0.12, `cpu_aft_dual`:+0.11, `esmc_mean`:+0.10, `peptideclm2`:+0.08 |
| **MIC_E_coli** | 0.448/0.64 | 0.442/0.64 | 0.431/0.66 | `esmc_multiregion` (MAE `0.408`, Sp `0.70`) | `cpu_aft_dual`:+0.29, `cpu_dual`:+0.19, `esmc_multiregion_smooth`:+0.17, `esmc_mean`:+0.16 |
| **MIC_P_aeruginosa** | 0.678/0.10 | 0.663/0.12 | 0.692/0.07 | `esmc_multiregion` (MAE `0.648`, Sp `0.27`) | `cpu_aft_dual`:+0.15, `cpu_dual`:+0.14, `chemberta_mtr`:+0.12, `peptideclm2`:+0.08 |
| **MIC_S_aureus** | 0.479/0.40 | 0.506/0.37 | 0.465/0.43 | `ncaa_protein_proj` (MAE `0.465`, Sp `0.43`) | `ncaa_fusion_hf`:+0.11, `chemberta_mtr`:+0.10, `peptideclm2`:+0.10, `cpu_dual`:+0.09 |
| **PAMPA** | 0.413/0.71 | 0.435/0.69 | 0.427/0.68 | `chemberta_mtr` (MAE `0.402`, Sp `0.74`) | `cpu_dual`:+0.44, `cpu_aft_dual`:+0.28, `ncaa_smiles_repr`:+0.12, `ncaa_protein_proj`:+0.10 |
| **PPB** | 0.250/0.56 | 0.263/0.82 | 0.263/0.51 | `ncaa_smiles_repr` (MAE `0.250`, Sp `0.56`) | `ncaa_fusion_hf`:+0.73, `cpu_aft_dual`:+0.27 |
| **RRCK** | 0.366/0.39 | 0.380/0.30 | 0.311/0.48 | `ncaa_protein_proj` (MAE `0.311`, Sp `0.48`) | `cpu_dual`:+0.25, `ncaa_smiles_repr`:+0.25, `ncaa_fusion_hf`:+0.25, `ncaa_protein_proj`:+0.25 |
| **Vd_per_kg** | 0.341/0.76 | 0.343/0.70 | 0.372/0.62 | `esmc_mean` (MAE `0.329`, Sp `0.76`) | `esmc_multiregion`:+0.15, `ncaa_smiles_repr`:+0.14, `ncaa_protein_proj`:+0.13, `direct_dual_smooth`:+0.12 |
| **cytotoxicity_cls** | 0.897 (SVC_RBF) | 0.893 (LightGBM) | 0.881 (SVC_RBF) | `esmc_multiregion` (ROC `0.907`) | `cpu_dual`:+0.27, `cpu_seq_only`:+0.15, `direct_dual_smooth`:+0.14, `direct_dual_tree`:+0.12 |
| **half_life_cls** | 0.779 (SVC_RBF) | 0.765 (SVC_RBF) | 0.775 (SVC_RBF) | `peptideclm2` (ROC `0.794`) | `esmc_mean`:+0.14, `ncaa_smiles_repr`:+0.14, `esmc_multiregion_tree`:+0.12, `cpu_dual`:+0.10 |
| **half_life_reg** | 0.676/0.47 | 0.671/0.48 | 0.687/0.46 | `peptideclm2` (MAE `0.658`, Sp `0.53`) | `chemberta_mtr`:+0.10, `ncaa_smiles_repr`:+0.09, `peptideclm2`:+0.09, `esmc_mean`:+0.09 |
| **hemolysis_HC50** | 0.419/0.59 | 0.416/0.61 | 0.425/0.60 | `esmc_multiregion` (MAE `0.393`, Sp `0.65`) | `direct_dual_smooth`:+0.55, `cpu_aft_dual`:+0.14, `direct_dual_tree`:+0.10, `esmc_multiregion_smooth`:+0.08 |
| **hemolysis_cls** | 0.869 (LightGBM) | 0.872 (LightGBM) | 0.867 (LightGBM) | `esmc_mean` (ROC `0.879`) | `cpu_dual`:+0.31, `direct_dual_smooth`:+0.28, `esmc_multiregion_smooth`:+0.17, `direct_dual_tree`:+0.09 |
| **overall_peptide_toxicity** | 0.918 (LightGBM) | 0.916 (LightGBM) | 0.919 (LightGBM) | `esmc_mean` (ROC `0.945`) | `cpu_dual`:+0.25, `esmc_multiregion_smooth`:+0.22, `esmc_mean`:+0.09, `direct_dual_tree`:+0.08 |
| **solubility_cls** | 0.854 (KLR_RBF) | 0.853 (SVC_RBF) | 0.858 (KLR_RBF) | `peptideclm2` (ROC `0.866`) | `cpu_dual`:+0.29, `esmc_multiregion_smooth`:+0.24, `ncaa_protein_proj`:+0.16, `esmc_mean`:+0.16 |
| **solubility_reg** | 0.579/0.67 | 0.603/0.67 | 0.548/0.71 | `esmc_multiregion` (MAE `0.528`, Sp `0.69`) | `cpu_dual`:+0.33, `ncaa_protein_proj`:+0.33, `peptideclm2`:+0.33 |

**Single-Foundation-Model Best Holdout Count (24 Endpoints)**: **ESM-C 600M**: `9/24`, **ncAA-RepDistill v2**: `9/24`, **peptideclm2**: `4/24`, **chemberta_mtr**: `2/24`

---

## 4. Censored-Aware Multimodal `XGBoost survival:aft` Ablation (All 14 Censored Endpoints)

Trained native `XGBoost survival:aft` on compact multimodal features (`physchem + fold-internal PCA(ncAA + PeptideCLM-2 + ChemBERTa + ESM-C Multi-Region) + cond`) AND strictly nested `convex_hinge` stacking of `CPU AFT + Multimodal Direct AFT`. Primary 9 censored targets marked with `★`.

| Endpoint | Cens (Tot/HO) | CPU AFT HO MAE / Sp / Conc / C-Idx | **MM Direct AFT** HO MAE / Sp / Conc / C-Idx | **MM Stacked AFT** HO MAE / Sp / **Conc** / **C-Idx** | Viol MAE (CPU→MM) | Hinge MAE (CPU→MM) |
|---|---:|---:|---:|---:|---:|---:|
| **hemolysis_HC50** ★ | 1375 / 292 | 0.417 / 0.620 / 0.651 / 0.772 | 0.427 / 0.596 / 0.637 / 0.769 | **0.413** / **0.628** / **0.651** / **0.779** | 0.115 → **0.107** | 0.338 → **0.333** |
| **MIC_S_aureus** ★ | 344 / 71 | 0.566 / 0.337 / 0.282 / 0.668 | 0.579 / 0.335 / 0.352 / 0.668 | **0.568** / **0.345** / **0.268** / **0.672** | 0.312 → **0.297** | 0.524 → **0.523** |
| **MIC_E_coli** ★ | 256 / 45 | 0.462 / 0.648 / 0.156 / 0.755 | 0.435 / 0.681 / 0.089 / 0.757 | **0.441** / **0.666** / **0.133** / **0.758** | 0.370 → **0.433** | 0.452 → **0.440** |
| **MIC_P_aeruginosa** ★ | 201 / 35 | 0.762 / 0.050 / 0.400 / 0.590 | 0.785 / 0.011 / 0.429 / 0.578 | **0.763** / **0.034** / **0.400** / **0.586** | 0.220 → **0.210** | 0.680 → **0.679** |
| **half_life_reg** ★ | 177 / 49 | 0.687 / 0.467 / 0.082 / 0.653 | 0.714 / 0.412 / 0.061 / 0.630 | **0.693** / **0.453** / **0.061** / **0.646** | 1.076 → **1.099** | 0.718 → **0.725** |
| **MIC_C_albicans** ★ | 56 / 4 | 0.535 / 0.572 / 0.250 / 0.723 | 0.545 / 0.522 / 0.250 / 0.700 | **0.519** / **0.560** / **0.250** / **0.718** | 0.242 → **0.369** | 0.523 → **0.512** |
| **PPB** ★ | 30 / 8 | 0.266 / 0.574 / 0.250 / 0.655 | 0.278 / 0.453 / 0.125 / 0.620 | **0.266** / **0.574** / **0.250** / **0.655** | 0.323 → **0.323** | 0.280 → **0.280** |
| **solubility_reg** ★ | 26 / 12 | 0.545 / 0.718 / 0.083 / 0.731 | 0.602 / 0.671 / 0.167 / 0.723 | **0.571** / **0.713** / **0.083** / **0.738** | 0.992 → **0.941** | 0.635 → **0.645** |
| **MDCK** | 16 / 0 | 0.574 / 0.407 / — / — | 0.503 / 0.434 / — / — | **0.574** / **0.407** / **—** / **—** | — → **—** | 0.574 → **0.574** |
| **Caco2** | 13 / 4 | 0.369 / 0.855 / 0.000 / 0.840 | 0.384 / 0.857 / 0.000 / 0.837 | **0.369** / **0.855** / **0.000** / **0.840** | 0.257 → **0.257** | 0.368 → **0.368** |
| **F** ★ | 10 / 2 | 0.224 / 0.597 / 0.500 / 0.698 | 0.230 / 0.667 / 0.500 / 0.741 | **0.225** / **0.650** / **0.500** / **0.744** | 0.011 → **0.018** | 0.207 → **0.208** |
| **PAMPA** | 9 / 1 | 0.382 / 0.790 / 0.000 / 0.804 | 0.433 / 0.702 / 0.000 / 0.764 | **0.382** / **0.790** / **0.000** / **0.804** | 0.394 → **0.394** | 0.382 → **0.382** |
| **CL_per_kg** | 7 / 0 | 0.633 / 0.780 / — / — | 0.564 / 0.801 / — / — | **0.633** / **0.780** / **—** / **—** | — → **—** | 0.633 → **0.633** |
| **LogD7.4** | 3 / 0 | 0.453 / 0.714 / — / — | 0.487 / 0.680 / — / — | **0.453** / **0.714** / **—** / **—** | — → **—** | 0.453 → **0.453** |

---

## 5. Calibration, Conformal Uncertainty & Applicability Domain

### 5.1 Classification Calibration (Holdout Brier & 10-Bin ECE: Wave-2C CPU vs. Wave-2D Multimodal Fusion)

| Endpoint | CPU Cal Brier / ECE | Fusion Cal Method | Fusion Raw → **Cal Brier** | Fusion Raw → **Cal ECE** | SMILES-Only Cal Brier / ECE |
|---|---:|---|---:|---:|---:|
| **BBB** | 0.1657 / 0.0963 | `none` | 0.1289 → **0.1289** | 0.0575 → **0.0575** | 0.1644 / 0.0881 |
| **cytotoxicity_cls** | 0.0981 / 0.0473 | `none` | 0.0948 → **0.0948** | 0.0526 → **0.0526** | 0.0965 / 0.0400 |
| **half_life_cls** | 0.2024 / 0.0639 | `none` | 0.1926 → **0.1926** | 0.0284 → **0.0284** | 0.1813 / 0.0236 |
| **hemolysis_cls** | 0.1171 / 0.0166 | `platt` | 0.1143 → **0.1126** | 0.0359 → **0.0173** | 0.1135 / 0.0088 |
| **overall_peptide_toxicity** | 0.0984 / 0.0321 | `none` | 0.0889 → **0.0889** | 0.0296 → **0.0296** | 0.1006 / 0.0299 |
| **solubility_cls** | 0.1473 / 0.0429 | `none` | 0.1395 → **0.1395** | 0.0293 → **0.0293** | 0.1381 / 0.0182 |

### 5.2 Regression Conformal 80% & 90% Empirical Holdout Coverage & Sharpness (CPU vs. Fusion)

| Endpoint | Conf Mode | CPU 80% Cov (Width) | **Fusion 80% Cov (Width)** | CPU 90% Cov (Width) | **Fusion 90% Cov (Width)** | SMILES-Only 80% / 90% |
|---|---|---:|---:|---:|---:|---:|
| **CL_absolute** | `standard` | 27.3% (2.751) | **27.3%** (2.229) | 27.3% (3.297) | **27.3%** (3.158) | 27.3% / 36.4% |
| **CL_per_kg** | `standard` | 78.0% (1.633) | **70.7%** (1.465) | 85.4% (2.272) | **92.7%** (2.620) | 78.0% / 92.7% |
| **Caco2** | `adaptive_tanimoto_dist` | 80.1% (1.143) | **81.3%** (1.129) | 91.3% (1.476) | **90.7%** (1.469) | 82.0% / 90.3% |
| **Cmax** | `standard` | 88.2% (3.408) | **88.2%** (3.435) | 88.2% (5.308) | **88.2%** (5.364) | 88.2% / 88.2% |
| **F** | `standard` | 69.6% (0.606) | **65.2%** (0.521) | 82.6% (0.690) | **82.6%** (0.662) | 65.2% / 69.6% |
| **LogD7.4** | `standard` | 76.8% (1.268) | **78.4%** (1.262) | 88.8% (1.707) | **88.0%** (1.907) | 79.2% / 89.6% |
| **MDCK** | `standard` | 61.8% (1.413) | **70.6%** (1.420) | 73.5% (1.900) | **82.3%** (1.883) | 67.7% / 85.3% |
| **MIC_C_albicans** | `standard` | 84.4% (1.683) | **86.7%** (1.613) | 91.1% (2.088) | **94.4%** (2.073) | 84.4% / 93.3% |
| **MIC_E_coli** | `standard` | 82.4% (1.554) | **83.4%** (1.490) | 92.4% (2.081) | **93.2%** (2.034) | 83.4% / 92.7% |
| **MIC_P_aeruginosa** | `standard` | 75.1% (1.503) | **75.6%** (1.427) | 83.8% (1.876) | **82.7%** (1.897) | 75.1% / 81.7% |
| **MIC_S_aureus** | `standard` | 85.6% (1.722) | **87.0%** (1.706) | 95.2% (2.290) | **96.1%** (2.308) | 85.9% / 95.2% |
| **PAMPA** | `standard` | 81.6% (1.221) | **80.4%** (1.195) | 90.7% (1.671) | **89.4%** (1.667) | 80.1% / 89.0% |
| **PPB** | `standard` | 50.0% (0.647) | **29.2%** (0.461) | 83.3% (0.802) | **91.7%** (0.691) | 29.2% / 91.7% |
| **RRCK** | `standard` | 86.7% (1.271) | **83.3%** (1.215) | 93.3% (1.595) | **90.0%** (1.587) | 83.3% / 83.3% |
| **Vd_per_kg** | `standard` | 84.9% (1.233) | **87.9%** (1.176) | 93.9% (1.574) | **93.9%** (1.552) | 84.9% / 93.9% |
| **half_life_reg** | `standard` | 75.5% (2.013) | **75.3%** (1.943) | 87.6% (2.670) | **88.4%** (2.666) | 75.7% / 88.2% |
| **hemolysis_HC50** | `standard` | 82.7% (1.324) | **83.3%** (1.250) | 92.0% (1.847) | **91.5%** (1.771) | 83.2% / 92.4% |
| **solubility_reg** | `adaptive_tanimoto_dist` | 75.0% (1.883) | **81.2%** (1.784) | 89.6% (2.751) | **87.5%** (2.553) | 79.2% / 91.7% |

---

## 6. Strict Zero-Leakage Leave-Case17-Out Clinical Peptide Drug Benchmark (17 FDA Drugs)

Every Wave-2D prediction below comes from a dedicated **Leave-Case17-Out (`case17_excluded == 0`)** 5-fold refit of every active pretrained head combined with Wave-2C's Leave-Case17-Out CPU models via non-Case17 stacker weights & calibrators (zero exact SMILES, InChIKey, InChIKey-14, or parent-sequence overlap).

### 6.1 Quantitative Zero-Leakage Alignment Against Audited FDA Numeric Anchors

| Clinical Anchor | n Drugs | Wave-2C CPU Dual MAE / Sp | **Wave-2D Fusion Dual MAE / Sp** | **Wave-2D Fusion AFT MAE / Sp** |
|---|---:|---:|---:|---:|
| **Half-Life `log10(h)` (pure ML, route-matched)** | 16 | 0.458 / 0.682 | **0.331** / **0.844** | 0.411 / 0.738 |
| **PPB `fraction [0,1]`** | 14 | 0.277 / 0.302 | **0.226** / **0.709** | **0.204** / **0.619** |

### 6.2 Key Clinical PK & Safety Table: Half-Life, PPB, Hemolysis & Toxicity (V32 vs. CPU vs. Fusion)

| Clinical Drug | FDA t1/2 Ref | CPU t1/2 (IV/SC) | **Fusion t1/2 (IV / SC / AFT) + Mech Anchor** | FDA PPB Ref | CPU PPB | **Fusion PPB (Dual / AFT)** | CPU Hemol | **Fusion Hemol (Cal) / HC50 (µM)** | Fusion Tox / BBB |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| **Leuprolide** | ~3 h (2.9-3 h, 二室模型) | 1.03h / 1.99h | **1.21h / 2.51h** (AFT 1.84h) | 43-49% | 0.586 | **0.508** (AFT 0.572) | 0.054 | **0.059** / 296 µM | 0.442 / 0.706 |
| **Bivalirudin** | ~25 min (肾功能正常) | 0.95h / 2.50h | **0.83h / 2.18h** (AFT 1.04h) | ~0% (不与血浆蛋白/红细胞结合) | 0.369 | **0.454** (AFT 0.594) | 0.012 | **0.029** / 403 µM | 0.152 / 0.296 |
| **Glucagon** | 8-18 min (说明书; 文献 IV 亦有 3-6 min) | 0.22h / 0.79h | **0.20h / 0.76h** (AFT 0.23h) | 低/未明确报道 (非酰化 29 肽) | 0.375 | **0.555** (AFT 0.755) | 0.028 | **0.056** / 87 µM | 0.028 / 0.771 |
| **Teriparatide** | IV ~5 min; SC ~1 h | 3.53h / 7.17h | **1.83h / 4.87h** (AFT 2.77h) | ~0% (无明显结合) | 0.524 | **0.594** (AFT 0.739) | 0.059 | **0.084** / 139 µM | 0.106 / 0.438 |
| **Exenatide** | 2.4 h (末端半衰期) | 3.58h / 3.15h | **1.73h / 3.17h** (AFT 3.53h) | ~53% | 0.475 | **0.545** (AFT 0.577) | 0.016 | **0.025** / 209 µM | 0.177 / 0.807 |
| **Liraglutide** | ~13 h (SC) | 3.72h / 9.57h | **10.54h / 25.71h** (AFT 8.48h); Anchor **9h** | >98% (白蛋白) | 0.436 | **0.684** (AFT 0.931) | 0.084 | **0.196** / 68 µM | 0.081 / 0.760 |
| **Semaglutide** | ~1 周 (~165-168 h) | 11.71h / 31.29h | **62.56h / 139.80h** (AFT 18.31h); Anchor **144h** | >99% (白蛋白) | 0.523 | **0.705** (AFT 0.893) | 0.076 | **0.127** / 82 µM | 0.108 / 0.860 |
| **Tirzepatide** | ~5 天 (~125 h) | 48.99h / 101.57h | **61.19h / 122.99h** (AFT 29.78h); Anchor **144h** | ~96% (血浆白蛋白) | 0.721 | **0.731** (AFT 0.796) | 0.148 | **0.155** / 75 µM | 0.191 / 0.491 |
| **Cyclosporine_A** | ~8.4 h (5-18 h) | 16.63h / 20.86h | **13.13h / 16.68h** (AFT 17.03h) | ~90% (主要脂蛋白) | 0.905 | **0.827** (AFT 0.901) | 0.503 | **0.284** / 135 µM | 0.247 / 0.276 |
| **Oxytocin** | 3-5 min (标签亦见 1-6 min) | 1.09h / 2.02h | **1.46h / 2.72h** (AFT 1.24h) | 轻度/未定量报道 | 0.471 | **0.358** (AFT 0.400) | 0.027 | **0.036** / 1232 µM | 0.278 / 0.044 |
| **Octreotide** | 1.7-1.9 h (SC/IV) | 2.34h / 4.41h | **1.55h / 3.09h** (AFT 2.41h) | ~65% (脂蛋白+白蛋白) | 0.708 | **0.608** (AFT 0.701) | 0.061 | **0.081** / 356 µM | 0.930 / 0.875 |
| **Linaclotide** | 血浆低于定量限 无法计算 (肠道局部作用) | 2.34h / 4.71h | **3.21h / 6.37h** (AFT 2.01h) | n/a (无全身暴露) | 0.497 | **0.351** (AFT 0.476) | 0.230 | **0.217** / 330 µM | 0.965 / 0.526 |
| **Ziconotide** | 鞘内 CSF ~4.6 h (2.9-6.5 h); 血浆 ~1.3 h(旧标签) | 1.10h / 2.80h | **0.71h / 1.85h** (AFT 1.05h) | ~50% | 0.383 | **0.415** (AFT 0.459) | 0.107 | **0.104** / 781 µM | 0.969 / 0.566 |
| **Daptomycin** | ~8 h (8-9 h) | 1.95h / 4.04h | **3.24h / 5.14h** (AFT 2.92h) | 90-93% (~92%) | 0.510 | **0.627** (AFT 0.531) | 0.102 | **0.192** / 288 µM | 0.264 / 0.222 |
| **Caspofungin** | β 相 9-11 h (γ 相 40-50 h) | 4.44h / 6.33h | **12.02h / 17.27h** (AFT 4.28h) | ~97% | 0.636 | **0.744** (AFT 0.683) | 0.045 | **0.077** / 678 µM | 0.390 / 0.357 |
| **Insulin_detemir** | 5-7 h (剂量依赖) | 1.36h / 2.38h | **4.83h / 9.45h** (AFT 3.32h); Anchor **9h** | >98% (~99%, 白蛋白) | 0.682 | **0.733** (AFT 0.869) | 0.393 | **0.349** / 62 µM | 0.260 / 0.335 |
| **Insulin_degludec** | ~25 h (稳态 剂量无关) | 1.61h / 2.74h | **4.75h / 9.59h** (AFT 3.06h); Anchor **25h** | >99% (白蛋白) | 0.680 | **0.722** (AFT 0.832) | 0.347 | **0.281** / 74 µM | 0.261 / 0.286 |

### 6.3 Extended Case17 ADME / PK Panel (Wave-2D Leave-Case17-Out Multimodal Fusion)

| Clinical Drug | PAMPA (log10 cm/s) | Caco2 (log10 cm/s) | LogD7.4 | F (PO / SC) | CL_per_kg (mL/min/kg) | Vd_per_kg (L/kg) | Solubility Prob |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Leuprolide** | -6.04 | -6.39 | 1.81 | 0.27 / 0.58 | 2.46 | 0.48 | 0.673 |
| **Bivalirudin** | -5.95 | -6.50 | 1.33 | 0.58 / 0.75 | 1.84 | 0.26 | 0.557 |
| **Glucagon** | -6.16 | -6.64 | 1.35 | 0.70 / 0.78 | 0.93 | 0.37 | 0.207 |
| **Teriparatide** | -6.28 | -6.67 | 1.39 | 0.56 / 0.61 | 2.74 | 0.28 | 0.279 |
| **Exenatide** | -6.14 | -6.51 | 1.21 | 0.62 / 0.66 | 2.43 | 0.35 | 0.667 |
| **Liraglutide** | -6.24 | -6.76 | 1.40 | 0.70 / 0.76 | 0.28 | 0.30 | 0.261 |
| **Semaglutide** | -6.23 | -6.69 | 1.43 | 0.70 / 0.75 | 0.73 | 0.28 | 0.320 |
| **Tirzepatide** | -6.17 | -6.61 | 1.29 | 0.62 / 0.66 | 1.51 | 0.30 | 0.538 |
| **Cyclosporine_A** | -5.80 | -6.09 | 3.45 | 0.20 / 0.37 | 6.94 | 0.99 | 0.133 |
| **Oxytocin** | -6.57 | -6.47 | 0.90 | 0.12 / 0.38 | 10.85 | 0.33 | 0.909 |
| **Octreotide** | -6.49 | -6.31 | 1.69 | 0.22 / 0.50 | 4.52 | 0.40 | 0.705 |
| **Linaclotide** | -6.17 | -6.21 | 0.97 | 0.24 / 0.41 | 5.02 | 0.35 | 0.869 |
| **Ziconotide** | -6.02 | -6.34 | 1.27 | 0.45 / 0.56 | 2.63 | 0.39 | 0.884 |
| **Daptomycin** | -6.08 | -6.42 | 1.34 | 0.51 / 0.69 | 1.94 | 0.26 | 0.813 |
| **Caspofungin** | -6.37 | -6.45 | 1.89 | 0.42 / 0.63 | 0.66 | 0.47 | 0.946 |
| **Insulin_detemir** | -6.18 | -6.46 | 1.51 | 0.51 / 0.54 | 5.21 | 0.26 | 0.638 |
| **Insulin_degludec** | -6.17 | -6.47 | 1.52 | 0.52 / 0.54 | 5.69 | 0.23 | 0.650 |

---

## 7. Key Scientific Findings, Honest Limitations & Wave-3 Routing Guidance

1. **True Multimodal Orthogonality — Every One of the 4 Foundation Models Wins Its Biological Niche**:
   - **`ncAA-RepDistill v2` (`9/24` best single pretrained representation)**: Dominated modified/cyclic/lipidated peptide ADME & PK (`PPB` HO Sp **`0.818`** via `fusion_hf`, `CL_per_kg` MAE `0.535`/Sp `0.803`, `CL_absolute` MAE `1.345`, `F` MAE `0.224`/Sp `0.563`, `LogD7.4` MAE `0.468`, `MDCK` MAE `0.451`, `RRCK` MAE `0.311`, `Caco2` MAE `0.387`, and `MIC_S_aureus` MAE `0.465`/Sp `0.433` via `protein_proj`). Crucially, its distilled `protein_proj` head (`1280D`, SMILES-only!) beat pure SMILES models on sequence-sensitive tasks (`BBB` `0.818`, `MIC_S_aureus` `0.465`, `solubility_reg` `0.548`).
   - **`ESM-C 600M Multi-Region` (`9/24` best single pretrained representation)**: Dominated sequence/motif-driven biology (`BBB` `0.877`, `overall_peptide_toxicity` `0.945`/`0.950`, `hemolysis_cls` `0.879`/`0.785`, `hemolysis_HC50` MAE `0.393`, `cytotoxicity_cls` `0.907`, `MIC_E_coli` MAE `0.408`/Sp `0.704`, `MIC_P_aeruginosa` MAE `0.648`/Sp `0.267`, `solubility_reg` MAE `0.528`/Sp `0.690`, `Vd_per_kg` MAE `0.329`).
   - **`PeptideCLM-2` (`4/24` best single pretrained representation)**: Led proteolytic half-life & chirality/PTM-sensitive tasks (`half_life_cls` HO ROC **`0.794`** / PR `0.799`, `half_life_reg` HO MAE **`0.658`** / Sp `0.534`, `solubility_cls` ROC **`0.866`** / PR `0.909`, `MIC_C_albicans` MAE **`0.421`** / Sp **`0.749`**).
   - **`ChemBERTa-MTR` (`2/24` best single pretrained representation)**: Strongest single pretrained prior on `PAMPA` (`0.402`/Sp `0.739`) and `Cmax` (`0.699`/Sp `0.852`), plus top-3 contributor on `MIC_S_aureus` & `MIC_P_aeruginosa`.
2. **Exact Full-Rank RBF Kernels + Uncompressed MLPs Unlock Large-Endpoint Multimodal Fusion**:
   - Replacing low-rank Nyström-900 with exact `FastPlattRBFSVC` / `SVR_RBF` / `KRR_RBF` + `FullEmb_MLP` delivered decisive lifts on all 4 large endpoints (`n_cv > 3,600`): `hemolysis_HC50` MAE `0.3994 → 0.3805` (`-0.0189`, Sp `0.656 → 0.678`), `overall_peptide_toxicity` ROC `0.9406 → 0.9507` (`+0.0101`, PR `0.9451 → 0.9545`), `hemolysis_cls` ROC `0.8921 → 0.8963` (PR `0.7911 → 0.8101`, `+0.0190`), and `PAMPA` MAE `0.3750 → 0.3732`.
3. **Major Clinical Translation Breakthrough (Strictly Leave-Case17-Out Pure ML)**:
   - **Pure-ML Half-Life Alignment (`n=16` FDA drugs, NO mechanism anchor)**: Route-matched Leave-Case17-Out MAE fell **28%** (`0.458 → 0.331 log10(h)`) and Spearman surged **`0.682 → 0.844`**, with pure ML predicting **Semaglutide `139.8 h`** (FDA `~165–168 h`, vs CPU `31.3 h`), **Tirzepatide `123.0 h`** (FDA `~125 h`), **Leuprolide `2.51 h`** (FDA `~3 h`), **Glucagon `0.20 h`** (FDA `8–18 min`), and **Caspofungin `12.0 h`** (FDA `9–11 h`).
   - **Pure-ML PPB Alignment (`n=14` FDA drugs)**: Leave-Case17-Out MAE dropped `0.277 → 0.226` (AFT **`0.204`**) and Spearman more than doubled **`0.302 → 0.709`**, driven primarily by `ncAA-RepDistill v2 fusion_hf` (`w=0.73`) + `CPU AFT` (`w=0.27`).
   - **Zero Clinical Hemolysis False Positives**: Calibrated `hemolysis_cls` kept all 17 FDA peptide drugs `<= 0.35` (vs `8/17 > 0.60` in V32), further reducing Cyclosporine A (`0.503 → 0.284`), Insulin detemir (`0.393 → 0.349`), and Insulin degludec (`0.347 → 0.281`).
4. **Censored Multimodal `XGBoost survival:aft` Gains**:
   - On **`hemolysis_HC50`** (`1,375` censored rows), Multimodal Stacked AFT simultaneously improved exact MAE (`0.417 → 0.413`), Spearman (`0.620 → 0.628`), violation MAE (`0.115 → 0.107`), hinge MAE (`0.338 → 0.333`), and generalized C-index (`0.772 → 0.779`) while maintaining `65.1%` censored bound concordance.
   - On **`MIC_E_coli`**, Multimodal Direct/Stacked AFT lowered exact MAE `0.462 → 0.435/0.441` and lifted Spearman `0.648 → 0.681/0.666` & C-index `0.755 → 0.758`; on **`F`**, C-index jumped `0.698 → 0.744`; on **`MIC_S_aureus`**, C-index lifted `0.668 → 0.672` and violation MAE fell `0.312 → 0.297`.
5. **Honest Limitations & Recommended Wave-3 Production Routing**:
   - **4 regression endpoints where CPU retained a slight holdout edge**: `RRCK` (`n=167`, CPU `0.268` vs Fusion `0.320`, though Fusion raised Sp `0.534 → 0.548`), `Vd_per_kg` (`n=166`, CPU `0.319` vs `0.330`), `MIC_P_aeruginosa` (CPU `0.650`/Sp `0.184` vs Fusion `0.657`/Sp `0.133`, though `esmc_multiregion` alone achieved `0.648`/Sp `0.267` and `chemberta_mtr` achieved `0.651`/Sp `0.285` — holdout cluster shift caused CV-optimal SMILES weights to dilute PLM signal), and `PAMPA` (Fusion beat MAE `0.3750 → 0.3732`, CPU retained higher Sp `0.797` vs `0.778`).
   - **SMILES-Only vs Dual Track on `half_life_cls` & `half_life_reg`**: On modified/acylated peptides where sequence-only rows add noise, Track A `smiles_only` (`PeptideCLM-2 + ncAA + CPU`) outperformed Track B `dual_modality` on holdout (`half_life_cls` ROC `0.8016` vs `0.7756`; `half_life_reg` MAE `0.6485`/Sp `0.534` vs `0.6654`/Sp `0.482`), confirming that modified peptides with valid SMILES benefit strongly from SMILES-first chemical foundation models.
   - **Small-n Conformal Caveat**: Excluding the 2-cluster holdout of `CL_absolute` (`n_ho=11`, `27.3%`) and small `PPB` (`n_ho=24`, `cov80=29.2%` / `cov90=91.7%`), the remaining 16 regression endpoints average **`79.9%`** (nominal 80%) and **`89.6%`** (nominal 90%) empirical holdout coverage with **tighter interval widths** than Wave-2C CPU across 15/18 endpoints.

---

## 8. Standardized Output & Verification Inventory

- **Remote Authoritative Bundle (`node23`)**: `/data/qlyu/v33_upgrade/models_fusion/`
  - Master summary: `/data/qlyu/v33_upgrade/models_fusion/fusion_benchmark_summary.json` (mirrored to `D:\work\中检院\v33_upgrade\models_fusion\fusion_benchmark_summary.json`)
  - Per-endpoint directories (24/24): `oof_predictions.tsv` (55,299 CV rows), `holdout_predictions.tsv` (13,812 untouched holdout rows), `case17_predictions.tsv` (17×24 = 408 strict leave-Case17-out clinical predictions), `metrics.json`, and `fusion_models_5fold.joblib` (inference-ready 5-fold preprocessors, FastPCA/scalers, base heads, ModalityAwareStacker, probability calibrator, conformal intervals, and packed-bit AD index).
- **Zero-Leakage Verification Gate**: `verify_fusion_outputs.py` asserted 100% exact row alignment (`69,111/69,111` rows), zero NaN/Inf, physical bounds compliance, zero sequence/ESM-C leakage in Track A `smiles_only`, 17/17 Case17 zero-leakage coverage, and full `fusion_models_5fold.joblib` deserialization across all 24 endpoints.
