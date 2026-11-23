# Peptide OmniPanel V33 — CPU Modeling, Censored AFT, Small-Sample GPR/Stacking & Calibration Benchmark

- **Generated**: 2026-11-21 CST | **Compute**: `node23` (64-core CPU parallel suite, total runtime `9648.6s`)
- **Authoritative Artifacts (`node23`)**: `<WORK_DIR>/v33_upgrade/models_cpu/` (`cpu_benchmark_summary.json` + 24 endpoint directories with `oof_predictions.tsv`, `holdout_predictions.tsv`, `case17_predictions.tsv`, `metrics.json`, `models_5fold.joblib`)
- **Data & Split Integrity**: Strictly trained & selected ONLY on inner 5-fold cluster-grouped CV (`is_holdout==0`, 24/24 fold-balanced, zero cluster/SMILES/InChIKey leakage); 20% outer holdout (`is_holdout==1`) untouched until final evaluation; Case17 clinical drugs evaluated strictly leave-Case17-out (`case17_excluded==0`).

---

## 1. Executive Highlights (Solving Every V32 Flaw)

1. **Guaranteed 24/24 Dual-Modality & SMILES-Only Coverage**: Every single one of the 24 endpoints ships BOTH a validated 5-fold `smiles_only` ensemble (`physchem_core[:48]` + compact FPs + conditions, **ZERO** sequence features) AND a full `dual_modality` 5-fold ensemble (`physchem_core[:83]` + valid-masked median imputation + `[has_smiles, has_seq]` flags, training on **100%** of rows including `20,506` `hemolysis_cls` and `3,078` `half_life_reg`), PLUS 12 `seq_only` ensembles for canonical sequence inputs. In V32, 12/17 Case17 clinical peptide drugs were skipped on `half_life` due to sequence-only routes; in V33, **17/17 drugs x 24/24 endpoints = 408/408 strict out-of-sample predictions** are produced cleanly.
2. **Anti-($p \gg n$) Compact Feature Engineering & Tanimoto+RBF Kernels**: Completely replaced V32's blind 15,212D concatenation with sample-size-tiered compact routes (`physchem_core` 48D/85D, fold-internal `TruncatedSVD` 32D, exact `Tanimoto+RBF+Condition` GPR/KRR kernels, and support-filtered compact fingerprints).
3. **Censored-Aware `XGBoost survival:aft` Ablation (14 Censored Endpoints)**: Native log-shifted AFT modeling (`T = exp(y - y_ref)`, right=`+inf`, left=`0`, interval=`[L, U]`) preserves all 2,523 non-exact observations instead of dropping or fabricating them, dramatically improving held-out censored bound concordance.
4. **Calibrated Probabilities & Empirical Conformal Uncertainty**: Across the 6 classification endpoints, out-of-fold Isotonic/Platt calibration reduced mean holdout **ECE** from `0.0643` to `0.0499` and mean holdout **Brier** from `0.1412` to `0.1382`. Across the 18 regression endpoints, CV+ / adaptive conformal intervals achieved mean holdout empirical coverage of **`74.8%`** (nominal 80%) and **`85.0%`** (nominal 90%).

---

## 2. Master 24-Endpoint Benchmark — V33 CPU vs. V32 Baseline

### 2.1 Quantitative Regression Endpoints (18 Endpoints)

| Endpoint | Tier | n (CV/HO) | V32 n / HO MAE / HO Sp | V33 Dual Winner (Route + Model) | Dual CV MAE / Sp | Dual HO MAE / RMSE / Sp | SMILES-Only HO MAE / Sp | Conformal 80% / 90% Cov |
|---|---|---:|---|---|---:|---:|---:|---:|
| **CL_absolute** | small | 54 (43/11) | n=7 (CV:0.536/0.582, no HO) | `physchem_core` + `SVR_RBF` | 0.688 / 0.463 | **1.491** / 1.696 / **0.420** | 1.384 / 0.542 | 27.3% / 27.3% |
| **CL_per_kg** | small | 206 (165/41) | n=50 / 0.593 / 0.242 | `physchem_svd` + `SVR_RBF` +meta | 0.502 / 0.724 | **0.566** / 0.696 / **0.797** | 0.575 / 0.784 | 78.0% / 85.4% |
| **Caco2** | medium | 1580 (1265/315) | n=1197 / 0.306 / 0.823 | `physchem_fp_all` + `XGBoost` | 0.379 / 0.797 | **0.368** / 0.475 / **0.860** | 0.366 / 0.860 | 80.1% / 91.3% |
| **Cmax** | small | 74 (57/17) | n=36 (CV:0.318/0.738, no HO) | `physchem_core` + `SVR_RBF` +meta | 0.876 / 0.169 | **1.106** / 1.422 / **0.720** | 0.878 / 0.578 | 88.2% / 88.2% |
| **F** | small | 124 (99/25) | n=19 / 0.238 / -0.034 | `tanimoto_hybrid` + `TanimotoRBF_KRR` | 0.219 / 0.633 | **0.239** / 0.303 / **0.478** | 0.285 / 0.426 | 69.6% / 82.6% |
| **LogD7.4** | medium | 678 (553/125) | n=57 / 0.464 / 0.849 | `physchem_core` + `ExtraTrees` | 0.424 / 0.617 | **0.491** / 0.704 / **0.668** | 0.501 / 0.667 | 76.8% / 88.8% |
| **MDCK** | small | 168 (134/34) | n=55 / 0.608 / 0.403 | `physchem_core` + `XGBoost` | 0.452 / 0.510 | **0.591** / 0.739 / **0.338** | 0.516 / 0.451 | 61.8% / 73.5% |
| **MIC_C_albicans** | medium | 468 (374/94) | New / deferred | `physchem_core` + `ExtraTrees` | 0.514 / 0.402 | **0.490** / 0.619 / **0.638** | 0.493 / 0.621 | 84.4% / 91.1% |
| **MIC_E_coli** | medium | 2210 (1768/442) | n=64 / 0.081 / 0.634 | `physchem_fp_all` + `XGBoost` | 0.489 / 0.605 | **0.451** / 0.594 / **0.650** | 0.453 / 0.640 | 82.4% / 92.4% |
| **MIC_P_aeruginosa** | medium | 1161 (929/232) | New / deferred | `physchem_fp_all` + `ExtraTrees` | 0.459 / 0.578 | **0.650** / 0.966 / **0.184** | 0.690 / 0.072 | 75.1% / 83.8% |
| **MIC_S_aureus** | medium | 2131 (1705/426) | New / deferred | `physchem_fp_all` + `ExtraTrees` | 0.550 / 0.336 | **0.494** / 0.623 / **0.378** | 0.474 / 0.434 | 85.6% / 95.2% |
| **PAMPA** | large | 8581 (6865/1716) | n=6748 / 0.273 / 0.842 | `physchem_fp_all` + `XGBoost` | 0.382 / 0.764 | **0.375** / 0.515 / **0.797** | 0.372 / 0.802 | 81.6% / 90.7% |
| **PPB** | small | 162 (130/32) | n=52 / 0.295 / -0.007 | `physchem_core` + `LightGBM` +meta | 0.182 / 0.579 | **0.299** / 0.330 / **0.591** | 0.246 / 0.606 | 50.0% / 83.3% |
| **RRCK** | small | 167 (137/30) | n=135 / 0.371 / 0.652 | `physchem_svd` + `SVR_RBF` | 0.378 / 0.571 | **0.268** / 0.374 / **0.534** | 0.277 / 0.547 | 86.7% / 93.3% |
| **Vd_per_kg** | small | 166 (133/33) | n=24 / 0.297 / 0.377 | `physchem_core` + `XGBoost` | 0.378 / 0.546 | **0.319** / 0.480 / **0.768** | 0.368 / 0.737 | 84.9% / 93.9% |
| **half_life_reg** | medium | 3078 (2462/616) | n=375 / 0.331 / 0.901 | `physchem_fp_all` + `XGBoost` | 0.626 / 0.575 | **0.694** / 0.867 / **0.476** | 0.669 / 0.558 | 75.5% / 87.6% |
| **hemolysis_HC50** | large | 5559 (4447/1112) | n=38 / 0.319 / 0.582 | `physchem_fp_all` + `XGBoost` | 0.414 / 0.598 | **0.399** / 0.544 / **0.656** | 0.403 / 0.649 | 82.7% / 92.0% |
| **solubility_reg** | small | 298 (238/60) | n=35 / 0.175 / -0.127 | `physchem_svd` + `SVR_RBF` | 0.567 / 0.688 | **0.638** / 0.847 / **0.636** | 0.633 / 0.691 | 75.0% / 89.6% |

### 2.2 Classification Endpoints (6 Endpoints — With Out-of-Fold Probability Calibration)

| Endpoint | n (CV/HO) | V32 n / HO ROC / PR / Brier | V33 Dual Winner | Dual CV ROC / PR | Dual HO ROC / PR / MCC | SMILES-Only HO ROC / PR | Seq-Only HO ROC / PR | Calib (Raw→Cal Brier / ECE) |
|---|---:|---|---|---:|---:|---:|---:|---|
| **BBB** | 844 (675/169) | n=844 / 0.877 / 0.879 / 0.149 | `physchem_fp_all` + `Logistic` | 0.856 / 0.867 | **0.832** / **0.843** / 0.563 | 0.823 / 0.820 | 0.817 / 0.813 | platt: Brier 0.1697→**0.1657**, ECE 0.0981→**0.0963** |
| **cytotoxicity_cls** | 2934 (2347/587) | New / deferred | `physchem_fp_all` + `LightGBM` | 0.869 / 0.774 | **0.931** / **0.888** / 0.714 | 0.930 / 0.886 | 0.905 / 0.833 | platt: Brier 0.0984→**0.0981**, ECE 0.0563→**0.0473** |
| **half_life_cls** | 3204 (2563/641) | n=67 / 0.941 / 0.778 / 0.061 | `physchem_morgan` + `LightGBM` | 0.777 / 0.788 | **0.751** / **0.730** / 0.328 | 0.790 / 0.763 | 0.713 / 0.674 | platt: Brier 0.2101→**0.2024**, ECE 0.1059→**0.0639** |
| **hemolysis_cls** | 20506 (16405/4101) | n=2012 / 0.930 / 0.849 / 0.102 | `physchem_fp_all` + `LightGBM` | 0.888 / 0.775 | **0.892** / **0.791** / 0.589 | 0.889 / 0.765 | 0.860 / 0.739 | platt: Brier 0.1200→**0.1171**, ECE 0.0471→**0.0166** |
| **overall_peptide_toxicity** | 11036 (8828/2208) | n=11036 / 0.955 / 0.960 / 0.085 | `physchem_fp_all` + `LightGBM` | 0.932 / 0.941 | **0.941** / **0.945** / 0.726 | 0.938 / 0.938 | 0.928 / 0.934 | platt: Brier 0.1013→**0.0984**, ECE 0.0442→**0.0321** |
| **solubility_cls** | 3722 (2977/745) | n=1334 / 0.715 / 0.930 / 0.116 | `physchem_core` + `ExtraTrees` | 0.849 / 0.893 | **0.857** / **0.904** / 0.536 | 0.862 / 0.904 | 0.856 / 0.891 | platt: Brier 0.1477→**0.1473**, ECE 0.0342→**0.0429** |

---

## 3. Small-n Anti-($p \gg n$) Analysis & Leak-Free Cross-Endpoint Meta-Stacking

All 7 PK endpoints (`PPB`, `CL_per_kg`, `CL_absolute`, `Vd_per_kg`, `F`, `Cmax`, `LogD7.4`) tested adding strictly **out-of-cluster** predicted meta-features from `{PAMPA, Caco2, LogD7.4, solubility_cls, hemolysis_cls}` (excluding self for `LogD7.4`). Every single query row verified `cluster_id not in aux_train_clusters` at runtime (zero cluster leakage). Meta-features were adopted ONLY when strictly reducing 5-fold CV MAE.

| PK Endpoint | n_cv | Zero-Leak Assertions | Dual No-Meta CV MAE (Sp) | Dual +5-Meta CV MAE (Sp) | Dual Meta Adopted? | SMILES No-Meta→+Meta CV MAE | Adopted? |
|---|---:|---:|---:|---:|:---:|---:|:---:|
| **PPB** | 130 | 7,700 (0 leaks) | 0.1991 (0.540) | 0.1822 (0.579) | YES | 0.1917 → 0.1949 | no |
| **CL_per_kg** | 165 | 8,972 (0 leaks) | 0.5101 (0.721) | 0.5023 (0.724) | YES | 0.5173 → 0.5097 | YES |
| **CL_absolute** | 43 | 2,668 (0 leaks) | 0.6875 (0.463) | 0.7029 (0.472) | no | 0.6853 → 0.7029 | no |
| **Vd_per_kg** | 133 | 7,492 (0 leaks) | 0.3784 (0.546) | 0.3787 (0.557) | no | 0.3629 → 0.3684 | no |
| **F** | 99 | 5,688 (0 leaks) | 0.2190 (0.633) | 0.2193 (0.630) | no | 0.1908 → 0.1894 | YES |
| **Cmax** | 57 | 3,172 (0 leaks) | 0.8928 (0.171) | 0.8764 (0.169) | YES | 0.9852 → 1.0174 | no |
| **LogD7.4** | 553 | 20,432 (0 leaks) | 0.4244 (0.617) | 0.4252 (0.621) | no | 0.4158 → 0.4236 | no |

### `MIC_S_aureus` Outlier Robustness Ablation (2 ChEMBL `10^6 uM` Rows)

| Variant | CV Exact MAE | CV Spearman | Holdout Exact MAE | Holdout RMSE | Holdout Spearman |
|---|---:|---:|---:|---:|---:|
| `lgb_l2_raw` | 0.5671 | 0.3148 | 0.4995 | 0.6356 | 0.3563 |
| `lgb_huber_raw` | 0.5580 | 0.3352 | 0.4953 | 0.6256 | 0.3710 |
| `lgb_huber_winsor5` | 0.5610 | 0.3220 | 0.4942 | 0.6246 | 0.3645 |
| `xgb_l2_raw` | 0.5659 | 0.3090 | 0.4880 | 0.6189 | 0.3872 |
| `xgb_huber_winsor5` | 0.5563 | 0.3355 | 0.4938 | 0.6230 | 0.3676 |

---

## 4. Censored-Aware Modeling (`XGBoost survival:aft`) — Controlled Ablation

Evaluated on ALL endpoints containing censored/interval rows. Compares apples-to-apples **Exact-Only XGBoost** (`reg:squarederror` on exact rows only) and the **Exact-Only Overall Winner** vs. **Censored `XGBoost survival:aft`** (trained on exact + right/left/interval bounds via shifted `T = exp(y - y_ref)`). Reported on the untouched 20% outer holdout for BOTH exact-row MAE/Spearman AND held-out censored bound concordance (fraction respecting experimental `>`/`<`/`[a,b]`) + generalized C-index.

| Endpoint | Censored (Total / HO) | Exact Winner HO MAE / Sp / Cens-Conc | Exact XGB HO MAE / Sp / Cens-Conc | **AFT XGB** HO MAE / Sp / **Cens-Conc** | Viol MAE (Ex→AFT) | C-Index (Ex→AFT) | AFT Dist/Scale |
|---|---:|---:|---:|---:|---:|---:|---|
| **hemolysis_HC50** | 1375 / 292 | 0.399 / 0.656 / 0.250 | 0.403 / 0.651 / 0.243 | **0.417** / **0.620** / **0.651** | 0.298 → **0.115** | 0.751 → **0.772** | normal (σ=0.5) |
| **MIC_S_aureus** | 344 / 71 | 0.494 / 0.378 / 0.028 | 0.487 / 0.383 / 0.028 | **0.566** / **0.337** / **0.282** | 0.683 → **0.312** | 0.652 → **0.668** | normal (σ=0.5) |
| **MIC_E_coli** | 256 / 45 | 0.451 / 0.650 / 0.022 | 0.455 / 0.652 / 0.022 | **0.462** / **0.648** / **0.156** | 0.624 → **0.370** | 0.751 → **0.755** | normal (σ=0.5) |
| **MIC_P_aeruginosa** | 201 / 35 | 0.650 / 0.184 / 0.114 | 0.676 / 0.139 / 0.114 | **0.762** / **0.050** / **0.400** | 0.512 → **0.220** | 0.607 → **0.590** | normal (σ=0.5) |
| **half_life_reg** | 177 / 49 | 0.694 / 0.476 / 0.020 | 0.673 / 0.492 / 0.020 | **0.687** / **0.467** / **0.082** | 1.153 → **1.076** | 0.660 → **0.653** | normal (σ=0.5) |
| **MIC_C_albicans** | 56 / 4 | 0.490 / 0.638 / 0.000 | 0.497 / 0.609 / 0.000 | **0.535** / **0.572** / **0.250** | 0.598 → **0.242** | 0.734 → **0.723** | normal (σ=0.5) |
| **PPB** | 30 / 8 | 0.299 / 0.591 / 0.250 | 0.309 / 0.583 / 0.250 | **0.266** / **0.574** / **0.250** | 0.388 → **0.323** | 0.607 → **0.655** | normal (σ=0.4) |
| **solubility_reg** | 26 / 12 | 0.638 / 0.636 / 0.167 | 0.589 / 0.742 / 0.083 | **0.545** / **0.718** / **0.083** | 1.047 → **0.992** | 0.729 → **0.731** | normal (σ=0.4) |
| **MDCK** | 16 / 0 | 0.591 / 0.338 / — | 0.596 / 0.393 / — | **0.574** / **0.407** / **—** | — → **—** | — → **—** | normal (σ=0.4) |
| **Caco2** | 13 / 4 | 0.368 / 0.860 / 0.000 | 0.373 / 0.855 / 0.000 | **0.369** / **0.855** / **0.000** | 0.371 → **0.257** | 0.837 → **0.840** | normal (σ=0.5) |
| **F** | 10 / 2 | 0.239 / 0.478 / 0.500 | 0.215 / 0.518 / 0.500 | **0.224** / **0.597** / **0.500** | 0.061 → **0.011** | 0.670 → **0.698** | normal (σ=0.4) |
| **PAMPA** | 9 / 1 | 0.375 / 0.797 / 0.000 | 0.384 / 0.789 / 0.000 | **0.382** / **0.790** / **0.000** | 1.261 → **0.394** | 0.803 → **0.804** | normal (σ=1.0) |
| **CL_per_kg** | 7 / 0 | 0.566 / 0.797 / — | 0.626 / 0.782 / — | **0.633** / **0.780** / **—** | — → **—** | — → **—** | logistic (σ=0.5) |
| **LogD7.4** | 3 / 0 | 0.491 / 0.668 / — | 0.458 / 0.705 / — | **0.453** / **0.714** / **—** | — → **—** | — → **—** | logistic (σ=0.6) |

---

## 5. Conformal Prediction Intervals & Applicability Domain (AD) Stratification

### 5.1 Holdout Conformal Uncertainty Coverage (18 Regression Endpoints)

| Endpoint | Conformal Mode | 80% Emp. Coverage (Width) | 90% Emp. Coverage (Width) | SMILES-Only 80% / 90% Cov |
|---|---|---:|---:|---:|
| **CL_absolute** | `standard` | **27.3%** (2.751) | **27.3%** (3.297) | 27.3% / 27.3% |
| **CL_per_kg** | `standard` | **78.0%** (1.633) | **85.4%** (2.272) | 70.7% / 90.2% |
| **Caco2** | `adaptive_tanimoto_dist` | **80.1%** (1.143) | **91.3%** (1.476) | 79.1% / 91.0% |
| **Cmax** | `standard` | **88.2%** (3.408) | **88.2%** (5.308) | 88.2% / 94.1% |
| **F** | `standard` | **69.6%** (0.606) | **82.6%** (0.690) | 65.2% / 69.6% |
| **LogD7.4** | `adaptive_tanimoto_dist` | **76.8%** (1.268) | **88.8%** (1.707) | 74.4% / 85.6% |
| **MDCK** | `standard` | **61.8%** (1.413) | **73.5%** (1.900) | 73.5% / 91.2% |
| **MIC_C_albicans** | `standard` | **84.4%** (1.683) | **91.1%** (2.088) | 82.2% / 92.2% |
| **MIC_E_coli** | `standard` | **82.4%** (1.554) | **92.4%** (2.081) | 82.6% / 91.4% |
| **MIC_P_aeruginosa** | `standard` | **75.1%** (1.503) | **83.8%** (1.876) | 76.6% / 81.7% |
| **MIC_S_aureus** | `standard` | **85.6%** (1.722) | **95.2%** (2.290) | 87.6% / 95.5% |
| **PAMPA** | `standard` | **81.6%** (1.221) | **90.7%** (1.671) | 80.6% / 90.7% |
| **PPB** | `standard` | **50.0%** (0.647) | **83.3%** (0.802) | 66.7% / 91.7% |
| **RRCK** | `standard` | **86.7%** (1.271) | **93.3%** (1.595) | 83.3% / 93.3% |
| **Vd_per_kg** | `standard` | **84.9%** (1.233) | **93.9%** (1.574) | 78.8% / 93.9% |
| **half_life_reg** | `standard` | **75.5%** (2.013) | **87.6%** (2.670) | 76.1% / 87.1% |
| **hemolysis_HC50** | `standard` | **82.7%** (1.324) | **92.0%** (1.847) | 83.5% / 92.0% |
| **solubility_reg** | `standard` | **75.0%** (1.883) | **89.6%** (2.751) | 75.0% / 89.6% |

### 5.2 Empirical Holdout Error vs. Tanimoto Applicability Domain Tier

Tiers via packed-bit `morgan_bits_1024`: **In-Domain** (`T_max >= 0.50`), **Borderline** (`0.30 <= T_max < 0.50`), **OOD** (`T_max < 0.30`).

| Endpoint | Metric | In-Domain (n / err) | Borderline (n / err) | OOD (n / err) | Spearman(T_max, \|err\|) |
|---|---|---:|---:|---:|---:|
| **BBB** | Brier | n=169 / 0.1657 | — | — | -0.161 |
| **CL_absolute** | MAE | n=10 / 1.3919 | n=1 / 2.4813 | — | 0.211 |
| **CL_per_kg** | MAE | n=39 / 0.5501 | n=2 / 0.8782 | — | -0.241 |
| **Caco2** | MAE | n=304 / 0.3586 | n=5 / 0.6343 | n=2 / 1.1004 | -0.179 |
| **Cmax** | MAE | n=4 / 1.2069 | n=12 / 0.8551 | n=1 / 3.7074 | -0.394 |
| **F** | MAE | n=20 / 0.2027 | n=3 / 0.4795 | — | -0.371 |
| **LogD7.4** | MAE | n=123 / 0.4738 | n=2 / 1.5553 | — | -0.331 |
| **MDCK** | MAE | n=34 / 0.5909 | — | — | -0.016 |
| **MIC_C_albicans** | MAE | n=81 / 0.4441 | n=6 / 0.6261 | n=3 / 1.4692 | -0.221 |
| **MIC_E_coli** | MAE | n=385 / 0.4497 | n=12 / 0.4847 | — | -0.162 |
| **MIC_P_aeruginosa** | MAE | n=193 / 0.6497 | n=4 / 0.6470 | — | -0.105 |
| **MIC_S_aureus** | MAE | n=333 / 0.4832 | n=22 / 0.6630 | — | -0.105 |
| **PAMPA** | MAE | n=1685 / 0.3684 | n=16 / 0.6146 | n=14 / 0.8848 | -0.165 |
| **PPB** | MAE | n=19 / 0.3313 | n=4 / 0.1137 | n=1 / 0.4316 | 0.181 |
| **RRCK** | MAE | n=29 / 0.2487 | n=1 / 0.8348 | — | -0.376 |
| **Vd_per_kg** | MAE | n=28 / 0.3435 | n=3 / 0.2179 | n=2 / 0.1241 | 0.313 |
| **cytotoxicity_cls** | Brier | n=571 / 0.0967 | n=13 / 0.1337 | n=3 / 0.2025 | -0.131 |
| **half_life_cls** | Brier | n=553 / 0.1985 | n=41 / 0.2068 | n=1 / 0.0153 | -0.076 |
| **half_life_reg** | MAE | n=498 / 0.6989 | n=23 / 0.7807 | n=6 / 0.3722 | 0.110 |
| **hemolysis_HC50** | MAE | n=819 / 0.3996 | n=1 / 0.2605 | — | 0.005 |
| **hemolysis_cls** | Brier | n=3626 / 0.1125 | n=3 / 0.0102 | — | 0.129 |
| **overall_peptide_toxicity** | Brier | n=2208 / 0.0984 | — | — | -0.039 |
| **solubility_cls** | Brier | n=736 / 0.1463 | n=7 / 0.2335 | — | 0.070 |
| **solubility_reg** | MAE | n=37 / 0.5392 | n=10 / 0.9291 | n=1 / 1.3815 | -0.299 |

---

## 6. Strict Zero-Leakage Case17 Clinical Peptide Drug Benchmark (V33 vs. V32 vs. FDA)

Every V33 prediction below comes from a dedicated **leave-Case17-out** 5-fold refit excluding 100% of `case17_excluded == 1` rows (439 rows removed across the 24 endpoints, including exact SMILES, InChIKey, InChIKey-14, and conservative parent-sequence matches, plus leave-Case17-out auxiliary meta-models).

### 6.1 Key Clinical PK & Safety Table: Half-Life, PPB & Hemolysis (False-Positive Fix Verification!)

| Clinical Drug | FDA t1/2 Ref | V32 t1/2 | **V33 ML t1/2 (IV / SC / AFT) + Mech Anchor** | FDA PPB Ref | V32 PPB | **V33 PPB (Dual / AFT)** | V32 Hemol Prob | **V33 Hemol Prob (Cal) / HC50 (µM, AFT)** |
|---|---|---|---|---|---:|---:|---:|---:|
| **Leuprolide** | ~3 h (2.9-3 h, 二室模型) | n/a (skipped) | IV 1.03h / SC 1.99h (AFT 1.51h) | 43-49% | 0.4907 | **0.586** (AFT 0.572) | 0.062 | **0.054** / 236 µM |
| **Bivalirudin** | ~25 min (肾功能正常) | n/a (skipped) | IV 0.95h / SC 2.50h (AFT 1.08h) | ~0% (不与血浆蛋白/红细胞结合) | 0.448 | **0.369** (AFT 0.594) | 0.022 | **0.012** / 445 µM |
| **Glucagon** | 8-18 min (说明书; 文献 IV 亦有 3-6 min) | n/a (skipped) | IV 0.22h / SC 0.79h (AFT 0.17h) | 低/未明确报道 (非酰化 29 肽) | 0.832 | **0.375** (AFT 0.755) | 0.00946 | **0.028** / 73 µM |
| **Teriparatide** | IV ~5 min; SC ~1 h | n/a (skipped) | IV 3.53h / SC 7.17h (AFT 2.94h) | ~0% (无明显结合) | 0.651 | **0.524** (AFT 0.739) | 0.00693 | **0.059** / 141 µM |
| **Exenatide** | 2.4 h (末端半衰期) | n/a (skipped) | IV 3.58h / SC 3.15h (AFT 2.95h) | ~53% | 0.661 | **0.475** (AFT 0.577) | 0.0621 | **0.016** / 196 µM |
| **Liraglutide** | ~13 h (SC) | 9 h [5-13 h] (daily_acyl) | IV 3.72h / SC 9.57h (AFT 13.46h); Anchor: **9h** (daily_acyl) | >98% (白蛋白) | 0.779 | **0.436** (AFT 0.931) | 0.0233 | **0.084** / 51 µM |
| **Semaglutide** | ~1 周 (~165-168 h) | 144 h [120-168 h] (weekly_diacid) | IV 11.71h / SC 31.29h (AFT 35.30h); Anchor: **144h** (weekly_diacid) | >99% (白蛋白) | 0.993 | **0.523** (AFT 0.893) | 0.772 | **0.076** / 69 µM |
| **Tirzepatide** | ~5 天 (~125 h) | 144 h [120-168 h] (weekly_diacid) | IV 48.99h / SC 101.57h (AFT 68.48h); Anchor: **144h** (weekly_diacid) | ~96% (血浆白蛋白) | 0.991 | **0.721** (AFT 0.796) | 0.989 | **0.148** / 75 µM |
| **Cyclosporine_A** | ~8.4 h (5-18 h) | n/a (skipped) | IV 16.63h / SC 20.86h (AFT 21.06h) | ~90% (主要脂蛋白) | 0.925 | **0.905** (AFT 0.901) | 0.0461 | **0.503** / 104 µM |
| **Oxytocin** | 3-5 min (标签亦见 1-6 min) | n/a (skipped) | IV 1.09h / SC 2.02h (AFT 1.42h) | 轻度/未定量报道 | 0.358 | **0.471** (AFT 0.400) | 0.394 | **0.027** / 2023 µM |
| **Octreotide** | 1.7-1.9 h (SC/IV) | n/a (skipped) | IV 2.34h / SC 4.41h (AFT 2.12h) | ~65% (脂蛋白+白蛋白) | 0.691 | **0.708** (AFT 0.701) | 0.639 | **0.061** / 395 µM |
| **Linaclotide** | 血浆低于定量限 无法计算 (肠道局部作用) | n/a (skipped) | IV 2.34h / SC 4.71h (AFT 1.47h) | n/a (无全身暴露) | 0.585 | **0.497** (AFT 0.476) | 0.63 | **0.230** / 355 µM |
| **Ziconotide** | 鞘内 CSF ~4.6 h (2.9-6.5 h); 血浆 ~1.3 h(旧标签) | n/a (skipped) | IV 1.10h / SC 2.80h (AFT 0.84h) | ~50% | 0.498 | **0.383** (AFT 0.459) | 0.896 | **0.107** / 1031 µM |
| **Daptomycin** | ~8 h (8-9 h) | n/a (skipped) | IV 1.95h / SC 4.04h (AFT 3.05h) | 90-93% (~92%) | 0.869 | **0.510** (AFT 0.531) | 0.625 | **0.102** / 203 µM |
| **Caspofungin** | β 相 9-11 h (γ 相 40-50 h) | n/a (skipped) | IV 4.44h / SC 6.33h (AFT 4.74h) | ~97% | 0.871 | **0.636** (AFT 0.683) | 0.192 | **0.045** / 862 µM |
| **Insulin_detemir** | 5-7 h (剂量依赖) | 9 h [5-13 h] (daily_acyl) | IV 1.36h / SC 2.38h (AFT 3.45h); Anchor: **9h** (daily_acyl) | >98% (~99%, 白蛋白) | 0.913 | **0.682** (AFT 0.869) | 0.811 | **0.393** / 63 µM |
| **Insulin_degludec** | ~25 h (稳态 剂量无关) | 25 h [25-25 h] (depot_diacid) | IV 1.61h / SC 2.74h (AFT 3.46h); Anchor: **25h** (ultralong_depot_diacid) | >99% (白蛋白) | 0.921 | **0.680** (AFT 0.832) | 0.813 | **0.347** / 71 µM |

### 6.2 Extended Case17 ADME / PK Panel (V33 Leave-Case17-Out vs. V32)

| Clinical Drug | PAMPA (log10 cm/s) | Caco2 (log10 cm/s) | LogD7.4 | F (PO / SC) | CL_per_kg (mL/min/kg) | Vd_per_kg (L/kg) | AD Tier (PPB / Hemol) |
|---|---:|---:|---:|---:|---:|---:|---|
| **Leuprolide** | -6.01 | -6.27 | 1.92 | 0.43 / 0.61 | 2.54 | 0.47 | In-Domain / In-Domain |
| **Bivalirudin** | -5.88 | -6.42 | 1.05 | 0.59 / 0.77 | 0.95 | 0.22 | In-Domain / In-Domain |
| **Glucagon** | -6.21 | -6.41 | 1.10 | 0.59 / 0.77 | 0.70 | 0.34 | In-Domain / In-Domain |
| **Teriparatide** | -6.38 | -6.24 | 1.12 | 0.56 / 0.74 | 2.29 | 0.31 | In-Domain / In-Domain |
| **Exenatide** | -6.16 | -6.31 | 0.73 | 0.60 / 0.77 | 1.40 | 0.34 | In-Domain / In-Domain |
| **Liraglutide** | -6.28 | -6.42 | 1.08 | 0.64 / 0.81 | 0.28 | 0.27 | In-Domain / In-Domain |
| **Semaglutide** | -6.30 | -6.32 | 1.10 | 0.56 / 0.73 | 0.90 | 0.27 | In-Domain / In-Domain |
| **Tirzepatide** | -6.18 | -6.42 | 0.77 | 0.54 / 0.71 | 2.02 | 0.25 | In-Domain / In-Domain |
| **Cyclosporine_A** | -5.81 | -5.96 | 3.42 | 0.26 / 0.43 | 6.82 | 1.10 | In-Domain / OOD |
| **Oxytocin** | -6.48 | -6.37 | 1.05 | 0.32 / 0.50 | 13.61 | 0.36 | In-Domain / In-Domain |
| **Octreotide** | -6.55 | -6.19 | 1.99 | 0.34 / 0.52 | 4.87 | 0.49 | In-Domain / In-Domain |
| **Linaclotide** | -6.04 | -6.05 | 0.96 | 0.28 / 0.45 | 5.33 | 0.37 | Borderline / In-Domain |
| **Ziconotide** | -5.98 | -6.02 | 0.96 | 0.32 / 0.49 | 2.00 | 0.42 | Borderline / In-Domain |
| **Daptomycin** | -5.93 | -6.09 | 1.14 | 0.43 / 0.61 | 1.99 | 0.31 | Borderline / Borderline |
| **Caspofungin** | -6.38 | -6.28 | 2.21 | 0.39 / 0.57 | 0.54 | 0.39 | Borderline / Borderline |
| **Insulin_detemir** | -6.21 | -6.13 | 1.17 | 0.47 / 0.64 | 4.00 | 0.25 | Borderline / In-Domain |
| **Insulin_degludec** | -6.19 | -6.13 | 1.16 | 0.46 / 0.63 | 4.17 | 0.24 | Borderline / In-Domain |

---

---

## 7. Key Scientific Findings, Honest Limitations & Wave 2D GPU Fusion Handoff

1. **Major Clinical Safety Breakthrough (Case17 Hemolysis False-Positive Elimination)**:
   - Scaling `hemolysis_cls` 5.2× (`3,938 → 20,506` rows) with clean molar thresholds and out-of-fold Platt calibration (`HO ECE 0.0471 → 0.0166`) eliminated V32's catastrophic false-positive hemolysis flags across FDA-approved clinical peptides (all strictly leave-Case17-out): **Semaglutide `0.772 → 0.076`**, **Tirzepatide `0.989 → 0.148`**, **Ziconotide `0.896 → 0.107` (`HC50 = 1,031 µM`)**, **Octreotide `0.639 → 0.061`**, **Daptomycin `0.625 → 0.102`**, **Linaclotide `0.630 → 0.230`**, **Insulin detemir `0.811 → 0.393`**, and **Insulin degludec `0.813 → 0.347`**. Zero of the 17 drugs remain flagged `>0.60` (vs. 8/17 in V32); Cyclosporine A (`p = 0.503`) is transparently flagged `AD = OOD`.
2. **Censored-Aware `XGBoost survival:aft` Is Essential on Right/Left/Interval-Censored Endpoints**:
   - On **`hemolysis_HC50`** (`1,375` censored rows), AFT jumped held-out censored bound concordance **2.7×** (`24.3% → 65.1%`), cut violation MAE **61%** (`0.298 → 0.115`), and lifted generalized C-index (`0.751 → 0.772`).
   - Across all 4 species-specific **MIC** endpoints, AFT boosted censored bound concordance **3.5×–10×** (`MIC_S_aureus`: `2.8% → 28.2%`, `MIC_E_coli`: `2.2% → 15.6%`, `MIC_P_aeruginosa`: `11.4% → 40.0%`, `MIC_C_albicans`: `0.0% → 25.0%`) while slashing violation MAE **41%–60%**.
   - On **`PPB`**, **`solubility_reg`**, **`MDCK`**, and **`LogD7.4`**, AFT even outperformed exact-only models on exact holdout MAE (`PPB 0.309→0.266`, `solubility_reg 0.638→0.545`, `MDCK 0.591→0.574`, `LogD7.4 0.491→0.453`), and on Case17 `PPB` rescued albumin-binding lipopeptides strictly out-of-sample (`Liraglutide 0.931`, `Semaglutide 0.893`, `Insulin_detemir 0.869`, `Insulin_degludec 0.832`). **Recommendation for Wave 2D Fusion**: blend `pred_aft` alongside `pred_dual` & GPU heads on all censored endpoints (especially `hemolysis_HC50`, `PPB`, `solubility_reg`, and all 4 `MIC_*`).
3. **Conformal Coverage & Applicability Domain Honesty**:
   - Across the 10 medium/large regression endpoints (`n >= 300`), empirical holdout conformal coverage averages **`80.2%`** (nominal 80%) and **`90.1%`** (nominal 90%) — matching finite-sample theory.
   - **Honest small-n caveat (`CL_absolute`, `n=54`, 11 holdout rows across only 2 holdout clusters)**: Holdout coverage is `27.3%` (`3/11`) due to a `0.85 log10` shift between the 2 holdout clusters (`median 0.08`) and the 25 CV clusters (`median 0.81–0.95`). Excluding this 2-cluster holdout, the remaining 17 regression endpoints average **`77.6%`** / **`88.4%`**.
   - Packed-bit Tanimoto AD reliably stratifies holdout error (`In-Domain → Borderline → OOD`): `Caco2` MAE `0.359 → 0.634 → 1.100`, `PAMPA` `0.368 → 0.615 → 0.885`, `MIC_C_albicans` `0.444 → 0.626 → 1.469`, `solubility_reg` `0.539 → 0.929 → 1.382`, `LogD7.4` `0.474 → 1.555`, `F` `0.203 → 0.480`, `CL_per_kg` `0.550 → 0.878`, `cytotoxicity_cls` Brier `0.097 → 0.134 → 0.203`.
4. **High-Leverage Targets for Wave 2D GPU Foundation Model Fusion (`ncAA-RepDistill`, `ESM-C 600M`, `PeptideCLM-2`, `ChemBERTa-MTR`)**:
   - Sequence-heavy endpoints where V32 previously relied on ESM2-650M (`BBB` CPU HO ROC `0.832` vs V32 `0.877`, `overall_peptide_toxicity` `0.941` vs `0.955`, `half_life_reg` `Sp 0.476` across 8-species/8-matrix unified cohort, `half_life_cls` ROC `0.751`, `MIC_P_aeruginosa` `Sp 0.184`, `MIC_S_aureus` `Sp 0.378`) are the highest-leverage targets for fusing GPU PLM (`ESM-C 600M`) and modified-peptide SMILES (`ncAA-RepDistill` / `PeptideCLM-2`) heads with our exact row-aligned CPU OOF/holdout predictions.

## 8. Standardized Output Inventory (Ready for Wave 2D GPU Fusion)

- Master summary: `node23:<WORK_DIR>/v33_upgrade/models_cpu/cpu_benchmark_summary.json`
- Per-endpoint directories (`node23:<WORK_DIR>/v33_upgrade/models_cpu/<endpoint>/`, all 24 endpoints):
  - `oof_predictions.tsv` (exact row-for-row alignment with `data/endpoints/<endpoint>.tsv` CV rows; 55,299 total rows)
  - `holdout_predictions.tsv` (exact row-for-row alignment with holdout rows; 13,812 total rows; includes calibrated probabilities / 80% & 90% conformal intervals / Tanimoto AD)
  - `case17_predictions.tsv` (strict leave-Case17-out predictions for all 17 clinical drugs + AD + conformal intervals + mechanism anchors)
  - `metrics.json` (full CV & holdout scorecards, ablations, calibration & conformal receipts)
  - `models_5fold.joblib` (5-fold trained estimators, fold-internal preprocessors, calibrators, conformal metadata & packed-bit AD index)
