# Peptide OmniPanel V33 — Super-Ensemble Champion Selection, Authoritative Serving Bundle & Live Inference Report

- **Generated**: 2026-11-22 06:15:28 +0800 | **Version**: `v33.0` | **Protocol**: Strictly Zero-Holdout-Peeking Nested 5-Fold OOF Selection
- **Authoritative Bundle (`node23`)**: `/data/qlyu/v33_upgrade/bundle/` (`v33_manifest.json`, `ad_analog_index.npz`, `analog_lookup.json.gz`, `serving_models/`, `v33_inference_service.py`, `esmc_worker.py`)
- **Local Mirror**: `D:\work\中检院\v33_upgrade\bundle\v33_manifest.json`
- **Live Inference Service**: `node23` (`0.0.0.0:8902`, `/usr/bin/python3` + persistent `esmc_worker.py` in `/data/qlyu/esmfold2-env312/bin/python`, 0 GPU usage; V32 `node23:8901` completely untouched & HTTP 200 verified)
- **Frozen Corpus**: `46,597` unique compounds (`46,598` incl. Case17) · `24` endpoints (`20` Tier-1 + `4` Tier-2) · `69,111` rows (`55,299` CV + `13,812` untouched 20% Holdout; `2,523` censored rows)

---

## 1. Executive Summary

1. **Cross-Family Super-Ensemble Champion Selection (Zero Holdout Peeking)**: All weights, single-vs-super-ensemble decisions (`margin >= 0.0005` on nested 5-fold OOF), probability calibrators, and conformal quantiles were frozen strictly on inner 5-fold cluster-grouped CV (`is_holdout == 0`) BEFORE touching the 20% holdout (`is_holdout == 1`) or Leave-Case17-Out predictions.
   - **Track B (`dual_modality`) Champion Mix (24 endpoints)**: {'cpu': 1, 'deep': 3, 'fusion': 7, 'super_ensemble': 13}
   - **Track A (`smiles_only`) Champion Mix (24 endpoints)**: {'cpu': 3, 'deep': 2, 'fusion': 10, 'super_ensemble': 9}
2. **Consistent Untouched-Holdout Gains Across All 3 Model Families**:
   - **6 Classification Endpoints (Mean Holdout ROC-AUC)**: CPU `0.8672` · Fusion `0.8868` · Deep `0.8692` → **V33 Champion `0.8883`** (Mean PR-AUC **`0.8717`**, Mean Calibrated Brier **`0.1259`**, Mean Calibrated ECE **`0.0377`**).
   - **18 Quantitative Regression Endpoints (Mean Holdout Exact MAE / Spearman)**: CPU `0.5516 / 0.5882` · Fusion `0.5246 / 0.6175` · Deep `0.5370 / 0.5859` → **V33 Champion `0.5053 / 0.6235`**.
   - **Conformal Uncertainty Calibration**: Mean holdout empirical coverage across all 18 regression endpoints = **`76.1%`** (nominal 80%) / **`88.9%`** (nominal 90%); across the 10 medium/large endpoints (`n >= 300`), coverage = **`80.8%`** / **`90.6%`**.
3. **Strict Leave-Case17-Out Clinical Breakthrough (17 FDA Peptide Drugs, 0 Leakage)**:
   - **Half-Life (`n=16` FDA drugs, route-matched)**: Pure-ML V33 Champion achieves **MAE = `0.3306` log10 h**, **Spearman = `0.9029`**, **R² = `0.7627`** (with deterministic Fatty-Acylation Mechanism Anchor: **MAE = `0.2987` log10 h**, **Spearman = `0.9028`**, **R² = `0.7813`**).
   - **Plasma Protein Binding (`n=14` FDA drugs)**: V33 Champion achieves **MAE = `0.2264`**, **Spearman = `0.7093`** (Censored-aware head: **MAE = `0.2253`**, **Spearman = `0.3524`**).
   - **Zero Clinical Hemolysis False Positives**: Max calibrated hemolysis probability across all 17 FDA peptide drugs = **`0.4289`** (`0/17 > 0.50`, vs. `8/17 > 0.60` in V32).
4. **Production Live Service (`node23:8902`) Verified**: Startup `215.75s`, pruned champion serving joblibs `10460.0 MB` (`22` CPU + `21` Fusion bundles + `20` Deep endpoints, unused families & zero-weight fusion heads dropped), steady RSS `23193.3 MB`, all 4 foundation encoders live-verified (`min cosine = 0.996829`; all standard-length SMILES & encoders `> 0.999`, chunked Tirzepatide GPU-bf16 vs CPU-fp32 `> 0.996`), true out-of-cache synthetic inference verified, and V32 `:8901` confirmed HTTP 200 untouched.

---

## 2. Master 24-Endpoint Champion Benchmark (V32 Ref vs. CPU vs. Fusion vs. Deep vs. V33 Champion)

### 2.1 Classification Endpoints (6 Endpoints — Calibrated Probabilities)

| Endpoint | n (CV/HO) | V32 HO ROC / PR | CPU HO ROC | Fusion HO ROC | Deep HO ROC | **V33 Champion (Dual)** | Weights | **Dual CV ROC / PR** | **Dual HO ROC / PR / Brier / ECE** | **SMI-Only HO ROC / PR** | Calib |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|
| **BBB** | 844 (675/169) | 0.877 / 0.879 | 0.8325 | 0.8987 | 0.8499 | **`fusion`** | `fusion`:1.00 | 0.9230 / 0.9173 | **0.8987** / **0.8898** / 0.1289 / 0.0575 | 0.8351 / 0.8278 | `none` |
| **cytotoxicity_cls** | 2934 (2347/587) | New / null | 0.9308 | 0.9342 | 0.9111 | **`super_ensemble`** | `fusion`:0.64, `deep`:0.23, `cpu`:0.13 | 0.8857 / 0.8017 | **0.9342** / **0.8950** / 0.0943 / 0.0366 | 0.9277 / 0.8873 | `none` |
| **half_life_cls** | 3204 (2563/641) | 0.941 / 0.778 | 0.7507 | 0.7756 | 0.7780 | **`super_ensemble`** | `fusion`:0.62, `deep_multitask`:0.17, `cpu`:0.13, `deep_specialist`:0.07 | 0.8138 / 0.8152 | **0.7817** / **0.7742** / 0.1913 / 0.0501 | 0.8016 / 0.7966 | `none` |
| **hemolysis_cls** | 20506 (16405/4101) | 0.930 / 0.849 | 0.8921 | 0.8963 | 0.8742 | **`super_ensemble`** | `fusion`:0.61, `deep`:0.24, `cpu`:0.16 | 0.9020 / 0.8043 | **0.8973** / **0.8095** / 0.1126 / 0.0196 | 0.8901 / 0.7668 | `platt` |
| **overall_peptide_toxicity** | 11036 (8828/2208) | 0.955 / 0.960 | 0.9406 | 0.9507 | 0.9445 | **`super_ensemble`** | `fusion`:0.73, `deep`:0.19, `cpu`:0.08 | 0.9437 / 0.9513 | **0.9515** / **0.9549** / 0.0892 / 0.0385 | 0.9390 / 0.9431 | `none` |
| **solubility_cls** | 3722 (2977/745) | 0.715 / 0.930 | 0.8568 | 0.8652 | 0.8574 | **`super_ensemble`** | `fusion`:0.60, `deep`:0.22, `cpu`:0.18 | 0.8596 / 0.9001 | **0.8665** / **0.9067** / 0.1393 / 0.0238 | 0.8653 / 0.9056 | `none` |

### 2.2 Quantitative Regression Endpoints (18 Endpoints — Exact Metrics & Conformal Coverage)

| Endpoint | Tier | n (CV/HO) | V32 HO MAE / Sp | CPU HO MAE / Sp | Fusion HO MAE / Sp | Deep HO MAE / Sp | **V33 Champion (Dual)** | Weights | **Dual CV MAE / Sp** | **Dual HO MAE / RMSE / Sp** | **SMI-Only HO MAE / Sp** | **Conf 80% / 90% Cov** |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| **CL_absolute** | small | 54 (43/11) | CV:0.536 (no HO) | 1.491 / 0.420 | 1.430 / 0.333 | 1.180 / 0.658 | **`deep`** | `deep`:1.00 | 0.670 / 0.452 | **1.180** / 1.464 / **0.658** | 1.216 / 0.580 | 63.6% / 72.7% |
| **CL_per_kg** | small | 206 (165/41) | 0.593 / 0.242 | 0.566 / 0.797 | 0.548 / 0.832 | 0.590 / 0.776 | **`fusion`** | `fusion`:1.00 | 0.500 / 0.675 | **0.548** / 0.704 / **0.832** | 0.547 / 0.794 | 70.7% / 92.7% |
| **Caco2** | medium | 1580 (1265/315) | 0.306 / 0.823 | 0.368 / 0.860 | 0.366 / 0.861 | 0.381 / 0.816 | **`super_ensemble`** | `deep`:0.45, `fusion`:0.33, `cpu`:0.22 | 0.362 / 0.812 | **0.363** / 0.479 / **0.853** | 0.370 / 0.840 | 78.8% / 89.1% |
| **Cmax** | small | 74 (57/17) | CV:0.318 (no HO) | 1.106 / 0.720 | 0.996 / 0.887 | 0.916 / 0.614 | **`deep`** | `deep`:1.00 | 0.860 / 0.264 | **0.916** / 1.453 / **0.614** | 1.069 / 0.168 | 82.3% / 88.2% |
| **F** | small | 124 (99/25) | 0.238 / -0.034 | 0.239 / 0.478 | 0.239 / 0.548 | 0.236 / 0.728 | **`deep`** | `deep`:1.00 | 0.194 / 0.571 | **0.236** / 0.336 / **0.728** | 0.241 / 0.696 | 78.3% / 82.6% |
| **LogD7.4** | medium | 678 (553/125) | 0.464 / 0.849 | 0.491 / 0.668 | 0.457 / 0.702 | 0.453 / 0.658 | **`super_ensemble`** | `fusion`:0.49, `cpu`:0.31, `deep`:0.19 | 0.417 / 0.631 | **0.451** / 0.661 / **0.699** | 0.462 / 0.706 | 80.8% / 90.4% |
| **MDCK** | small | 168 (134/34) | 0.608 / 0.403 | 0.591 / 0.338 | 0.499 / 0.455 | 0.573 / 0.335 | **`super_ensemble`** | `cpu`:0.61, `deep`:0.30, `fusion`:0.08 | 0.447 / 0.505 | **0.543** / 0.682 / **0.359** | 0.497 / 0.484 | 67.7% / 79.4% |
| **MIC_C_albicans** | medium | 468 (374/94) | New / null | 0.490 / 0.638 | 0.444 / 0.708 | 0.487 / 0.667 | **`super_ensemble`** | `deep`:0.47, `fusion`:0.45, `cpu`:0.09 | 0.472 / 0.492 | **0.439** / 0.558 / **0.715** | 0.438 / 0.704 | 86.7% / 94.4% |
| **MIC_E_coli** | medium | 2210 (1768/442) | 0.081 / 0.634 | 0.451 / 0.650 | 0.421 / 0.676 | 0.486 / 0.629 | **`super_ensemble`** | `fusion`:0.54, `cpu`:0.27, `deep`:0.19 | 0.475 / 0.624 | **0.432** / 0.570 / **0.663** | 0.434 / 0.659 | 81.6% / 92.7% |
| **MIC_P_aeruginosa** | medium | 1161 (929/232) | New / null | 0.650 / 0.184 | 0.657 / 0.133 | 0.741 / 0.111 | **`fusion`** | `fusion`:1.00 | 0.451 / 0.603 | **0.657** / 1.005 / **0.133** | 0.672 / 0.102 | 75.6% / 82.7% |
| **MIC_S_aureus** | medium | 2131 (1705/426) | New / null | 0.494 / 0.378 | 0.476 / 0.435 | 0.545 / 0.314 | **`fusion`** | `fusion`:1.00 | 0.540 / 0.382 | **0.476** / 0.593 / **0.435** | 0.476 / 0.433 | 87.0% / 96.1% |
| **PAMPA** | large | 8581 (6865/1716) | 0.273 / 0.842 | 0.375 / 0.797 | 0.373 / 0.778 | 0.411 / 0.706 | **`super_ensemble`** | `cpu`:0.45, `fusion`:0.40, `deep`:0.15 | 0.379 / 0.768 | **0.374** / 0.516 / **0.783** | 0.373 / 0.787 | 80.1% / 90.3% |
| **PPB** | small | 162 (130/32) | 0.295 / -0.007 | 0.299 / 0.591 | 0.261 / 0.678 | 0.233 / 0.789 | **`fusion`** | `fusion`:1.00 | 0.151 / 0.706 | **0.261** / 0.297 / **0.678** | 0.216 / 0.803 | 29.2% / 91.7% |
| **RRCK** | small | 167 (137/30) | 0.371 / 0.652 | 0.268 / 0.534 | 0.320 / 0.548 | 0.409 / 0.323 | **`cpu`** | `cpu`:1.00 | 0.378 / 0.571 | **0.268** / 0.374 / **0.534** | 0.322 / 0.521 | 86.7% / 93.3% |
| **Vd_per_kg** | small | 166 (133/33) | 0.297 / 0.377 | 0.319 / 0.768 | 0.330 / 0.714 | 0.340 / 0.578 | **`fusion`** | `fusion`:1.00 | 0.368 / 0.580 | **0.330** / 0.487 / **0.714** | 0.368 / 0.737 | 87.9% / 93.9% |
| **half_life_reg** | medium | 3078 (2462/616) | 0.331 / 0.900 | 0.694 / 0.476 | 0.665 / 0.482 | 0.684 / 0.460 | **`super_ensemble`** | `fusion`:0.38, `deep_specialist`:0.27, `deep_multitask`:0.26, `cpu`:0.09 | 0.585 / 0.634 | **0.664** / 0.853 / **0.485** | 0.639 / 0.557 | 74.8% / 87.8% |
| **hemolysis_HC50** | large | 5559 (4447/1112) | 0.319 / 0.583 | 0.399 / 0.656 | 0.381 / 0.678 | 0.404 / 0.612 | **`super_ensemble`** | `fusion`:0.53, `deep`:0.30, `cpu`:0.18 | 0.389 / 0.648 | **0.378** / 0.528 / **0.675** | 0.395 / 0.661 | 82.2% / 92.2% |
| **solubility_reg** | small | 298 (238/60) | 0.175 / -0.127 | 0.638 / 0.636 | 0.581 / 0.666 | 0.598 / 0.774 | **`fusion`** | `fusion`:1.00 | 0.545 / 0.728 | **0.581** / 0.796 / **0.666** | 0.633 / 0.691 | 75.0% / 89.6% |

### 2.3 Track A (`smiles_only`) Champion Routing Summary (24 Endpoints)

| Endpoint | Task | SMILES-Only Winner | SMILES-Only Weights | Calib / Conf Mode | SMILES-Only HO Primary Metric | SMILES-Only Conf 80% / 90% or Brier / ECE |
|---|---|---|---|---|---:|---:|
| **BBB** | cla | `fusion` | `fusion`:1.00 | `none` | ROC=0.8351, PR=0.8278 | Brier=0.1644, ECE=0.0881 |
| **CL_absolute** | reg | `super_ensemble` | `deep`:0.59, `cpu`:0.41 | `standard` | MAE=1.216, Sp=0.580 | 63.6% / 72.7% |
| **CL_per_kg** | reg | `super_ensemble` | `deep`:0.54, `cpu`:0.46 | `standard` | MAE=0.547, Sp=0.794 | 68.3% / 85.4% |
| **Caco2** | reg | `super_ensemble` | `deep`:0.44, `cpu`:0.39, `fusion`:0.18 | `adaptive_tanimoto_dist` | MAE=0.370, Sp=0.840 | 77.5% / 86.8% |
| **Cmax** | reg | `deep` | `deep`:1.00 | `standard` | MAE=1.069, Sp=0.168 | 82.3% / 88.2% |
| **F** | reg | `deep` | `deep`:1.00 | `standard` | MAE=0.241, Sp=0.696 | 82.6% / 82.6% |
| **LogD7.4** | reg | `fusion` | `fusion`:1.00 | `adaptive_tanimoto_dist` | MAE=0.462, Sp=0.706 | 79.2% / 89.6% |
| **MDCK** | reg | `super_ensemble` | `cpu`:0.37, `deep`:0.33, `fusion`:0.30 | `standard` | MAE=0.497, Sp=0.484 | 73.5% / 85.3% |
| **MIC_C_albicans** | reg | `super_ensemble` | `cpu`:0.36, `deep`:0.35, `fusion`:0.29 | `standard` | MAE=0.438, Sp=0.704 | 80.0% / 94.4% |
| **MIC_E_coli** | reg | `fusion` | `fusion`:1.00 | `standard` | MAE=0.434, Sp=0.659 | 83.4% / 92.7% |
| **MIC_P_aeruginosa** | reg | `fusion` | `fusion`:1.00 | `standard` | MAE=0.672, Sp=0.102 | 75.1% / 81.7% |
| **MIC_S_aureus** | reg | `fusion` | `fusion`:1.00 | `standard` | MAE=0.476, Sp=0.433 | 85.9% / 95.2% |
| **PAMPA** | reg | `super_ensemble` | `cpu`:0.48, `fusion`:0.37, `deep`:0.15 | `standard` | MAE=0.373, Sp=0.787 | 80.3% / 89.1% |
| **PPB** | reg | `super_ensemble` | `deep`:0.53, `fusion`:0.47 | `standard` | MAE=0.216, Sp=0.803 | 75.0% / 91.7% |
| **RRCK** | reg | `fusion` | `fusion`:1.00 | `standard` | MAE=0.322, Sp=0.521 | 83.3% / 83.3% |
| **Vd_per_kg** | reg | `cpu` | `cpu`:1.00 | `standard` | MAE=0.368, Sp=0.737 | 78.8% / 93.9% |
| **cytotoxicity_cls** | cla | `fusion` | `fusion`:1.00 | `none` | ROC=0.9277, PR=0.8873 | Brier=0.0965, ECE=0.0400 |
| **half_life_cls** | cla | `fusion` | `fusion`:1.00 | `none` | ROC=0.8016, PR=0.7966 | Brier=0.1813, ECE=0.0236 |
| **half_life_reg** | reg | `super_ensemble` | `fusion`:0.45, `deep_specialist`:0.23, `deep_multitask`:0.18, `cpu`:0.14 | `standard` | MAE=0.639, Sp=0.557 | 77.8% / 88.2% |
| **hemolysis_HC50** | reg | `fusion` | `fusion`:1.00 | `standard` | MAE=0.395, Sp=0.661 | 83.2% / 92.4% |
| **hemolysis_cls** | cla | `fusion` | `fusion`:1.00 | `none` | ROC=0.8901, PR=0.7668 | Brier=0.1135, ECE=0.0088 |
| **overall_peptide_toxicity** | cla | `cpu` | `cpu`:1.00 | `platt` | ROC=0.9390, PR=0.9431 | Brier=0.1002, ECE=0.0312 |
| **solubility_cls** | cla | `super_ensemble` | `fusion`:0.56, `deep`:0.31, `cpu`:0.13 | `none` | ROC=0.8653, PR=0.9056 | Brier=0.1385, ECE=0.0275 |
| **solubility_reg** | reg | `cpu` | `cpu`:1.00 | `standard` | MAE=0.633, Sp=0.691 | 75.0% / 89.6% |

### 2.4 Censored-Aware Diagnostic / Blended Head Scorecard (14 Censored Endpoints — Untouched Holdout)

| Endpoint | Cens (Tot/HO) | Champion Exact HO MAE / Cens-Conc / Viol MAE / C-Idx | **Censored Head Weights** | **Censored Head HO MAE / Cens-Conc / Viol MAE / C-Idx** |
|---|---:|---:|---|---:|
| **hemolysis_HC50** | 1375 / 292 | 0.378 / 0.408 / 0.208 / 0.780 | `champion_exact`:0.46, `deep`:0.27, `cpu_aft`:0.19, `fusion_aft`:0.09 | **0.384** / **0.538** / **0.145** / **0.786** |
| **MIC_S_aureus** | 344 / 71 | 0.476 / 0.056 / 0.664 / 0.683 | `fusion_aft`:0.40, `champion_exact`:0.28, `deep`:0.27, `cpu_aft`:0.05 | **0.525** / **0.169** / **0.404** / **0.678** |
| **MIC_E_coli** | 256 / 45 | 0.432 / 0.067 / 0.558 / 0.759 | `champion_exact`:0.41, `deep`:0.21, `cpu_aft`:0.20, `fusion_aft`:0.19 | **0.442** / **0.089** / **0.459** / **0.757** |
| **MIC_P_aeruginosa** | 201 / 35 | 0.657 / 0.114 / 0.476 / 0.611 | `champion_exact`:0.40, `cpu_aft`:0.21, `fusion_aft`:0.20, `deep`:0.18 | **0.700** / **0.257** / **0.305** / **0.598** |
| **half_life_reg** | 177 / 49 | 0.664 / 0.041 / 1.150 / 0.654 | `champion_exact`:1.00 | **0.664** / **0.041** / **1.150** / **0.654** |
| **MIC_C_albicans** | 56 / 4 | 0.439 / 0.000 / 0.577 / 0.776 | `champion_exact`:1.00 | **0.439** / **0.000** / **0.577** / **0.776** |
| **PPB** | 30 / 8 | 0.261 / 0.250 / 0.291 / 0.700 | `deep`:1.00 | **0.233** / **0.000** / **0.364** / **0.755** |
| **solubility_reg** | 26 / 12 | 0.581 / 0.000 / 0.943 / 0.688 | `champion_exact`:1.00 | **0.581** / **0.000** / **0.943** / **0.688** |
| **MDCK** | 16 / 0 | 0.543 / — / — / — | `deep`:0.37, `cpu_aft`:0.27, `fusion_aft`:0.27, `champion_exact`:0.09 | **0.565** / **—** / **—** / **—** |
| **Caco2** | 13 / 4 | 0.363 / 0.000 / 0.237 / 0.837 | `champion_exact`:1.00 | **0.363** / **0.000** / **0.237** / **0.837** |
| **F** | 10 / 2 | 0.236 / 0.500 / 0.054 / 0.759 | `champion_exact`:1.00 | **0.236** / **0.500** / **0.054** / **0.759** |
| **PAMPA** | 9 / 1 | 0.374 / 0.000 / 1.058 / 0.801 | `champion_exact`:1.00 | **0.374** / **0.000** / **1.058** / **0.801** |
| **CL_per_kg** | 7 / 0 | 0.548 / — / — / — | `champion_exact`:0.39, `deep`:0.35, `cpu_aft`:0.13, `fusion_aft`:0.13 | **0.555** / **—** / **—** / **—** |
| **LogD7.4** | 3 / 0 | 0.451 / — / — / — | `champion_exact`:1.00 | **0.451** / **—** / **—** / **—** |

---

## 3. Strict Leave-Case17-Out 17-FDA-Drug Super-Ensemble Scorecard (Zero Leakage)

- **Protocol**: Strict Leave-Case17-Out (case17_excluded==0, 0 overlapping SMILES/InChIKey/InChIKey-14/clusters across all 3 families). Uses the EXACT SAME frozen V33 Champion weights & calibrators selected strictly on inner OOF.
- **Half-Life (`n=16`, log10 h, route-matched)**: Pure ML **MAE=`0.3306`**, **RMSE=`0.4289`**, **R²=`0.7627`**, **Spearman=`0.9029`**, **Pearson=`0.8986`** | Pure ML + Mechanism Anchor **MAE=`0.2987`**, **R²=`0.7813`**, **Spearman=`0.9028`**.
- **PPB (`n=14`)**: V33 Champion **MAE=`0.2264`**, **Spearman=`0.7093`** | Censored Head **MAE=`0.2253`**, **Spearman=`0.3524`**.

| Clinical Drug | Route | FDA t1/2 Ref | V32 t1/2 | **V33 ML t1/2 (IV / SC)** | **Mechanism Anchor** | FDA PPB Ref | **V33 PPB (Champ / Cens)** | V32 Hemol | **V33 Hemol (Cal) / HC50 (µM)** | **Tox / BBB** |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| **Leuprolide** | SC | ~3 h (2.9-3 h, 二室模型) | skipped | **1.18h / 3.48h** | non_acylated | 43-49% | **0.508** / 0.421 | 0.062 | **0.054** / 171 µM | 0.474 / 0.706 |
| **Bivalirudin** | IV | ~25 min (肾功能正常) | skipped | **1.33h / 5.24h** | non_acylated | ~0% (不与血浆蛋白/红细胞结合) | **0.454** / 0.136 | 0.022 | **0.031** / 129 µM | 0.140 / 0.296 |
| **Glucagon** | IV | 8-18 min (说明书; 文献 IV 亦有 3-6 min) | skipped | **0.15h / 0.85h** | non_acylated | 低/未明确报道 (非酰化 29 肽) | **0.555** / 0.422 | 0.009 | **0.054** / 86 µM | 0.025 / 0.771 |
| **Teriparatide** | SC | IV ~5 min; SC ~1 h | skipped | **0.72h / 3.15h** | non_acylated | ~0% (无明显结合) | **0.594** / 0.654 | 0.007 | **0.075** / 155 µM | 0.088 / 0.438 |
| **Exenatide** | SC | 2.4 h (末端半衰期) | skipped | **2.18h / 6.82h** | non_acylated | ~53% | **0.545** / 0.575 | 0.062 | **0.030** / 101 µM | 0.119 / 0.807 |
| **Liraglutide** | SC | ~13 h (SC) | 9 h [5-13 h] (daily_acyl) | **6.14h / 21.33h** | **9h** (`daily_acyl`) | >98% (白蛋白) | **0.684** / 0.612 | 0.023 | **0.207** / 49 µM | 0.058 / 0.760 |
| **Semaglutide** | SC | ~1 周 (~165-168 h) | 144 h [120-168 h] (weekly_diacid) | **32.43h / 100.03h** | **144h** (`weekly_diacid`) | >99% (白蛋白) | **0.705** / 0.668 | 0.772 | **0.117** / 86 µM | 0.074 / 0.860 |
| **Tirzepatide** | SC | ~5 天 (~125 h) | 144 h [120-168 h] (weekly_diacid) | **44.52h / 121.85h** | **144h** (`weekly_diacid`) | ~96% (血浆白蛋白) | **0.731** / 0.874 | 0.989 | **0.155** / 88 µM | 0.122 / 0.491 |
| **Cyclosporine_A** | PO | ~8.4 h (5-18 h) | skipped | **10.98h / 24.63h** | non_acylated | ~90% (主要脂蛋白) | **0.827** / 0.746 | 0.046 | **0.429** / 44 µM | 0.255 / 0.276 |
| **Oxytocin** | IV | 3-5 min (标签亦见 1-6 min) | skipped | **0.88h / 2.55h** | non_acylated | 轻度/未定量报道 | **0.358** / 0.297 | 0.394 | **0.034** / 305 µM | 0.271 / 0.045 |
| **Octreotide** | SC | 1.7-1.9 h (SC/IV) | skipped | **1.80h / 5.58h** | non_acylated | ~65% (脂蛋白+白蛋白) | **0.608** / 0.485 | 0.639 | **0.072** / 128 µM | 0.909 / 0.875 |
| **Linaclotide** | PO | 血浆低于定量限 无法计算 (肠道局部作用) | skipped | **3.84h / 11.38h** | non_acylated | n/a (无全身暴露) | **0.351** / 0.325 | 0.630 | **0.258** / 105 µM | 0.969 / 0.526 |
| **Ziconotide** | IV | 鞘内 CSF ~4.6 h (2.9-6.5 h); 血浆 ~1.3 h(旧标签) | skipped | **1.15h / 3.66h** | non_acylated | ~50% | **0.415** / 0.415 | 0.896 | **0.102** / 590 µM | 0.970 / 0.566 |
| **Daptomycin** | IV | ~8 h (8-9 h) | skipped | **5.80h / 13.85h** | non_acylated | 90-93% (~92%) | **0.627** / 0.865 | 0.625 | **0.183** / 84 µM | 0.208 / 0.222 |
| **Caspofungin** | IV | β 相 9-11 h (γ 相 40-50 h) | skipped | **14.19h / 30.15h** | non_acylated | ~97% | **0.744** / 0.929 | 0.192 | **0.121** / 191 µM | 0.423 / 0.357 |
| **Insulin_detemir** | SC | 5-7 h (剂量依赖) | 9 h [5-13 h] (daily_acyl) | **5.57h / 16.37h** | **9h** (`daily_acyl`) | >98% (~99%, 白蛋白) | **0.733** / 0.583 | 0.811 | **0.320** / 42 µM | 0.217 / 0.335 |
| **Insulin_degludec** | SC | ~25 h (稳态 剂量无关) | 25 h [25-25 h] (depot_diacid) | **6.82h / 20.44h** | **25h** (`ultralong_depot_diacid`) | >99% (白蛋白) | **0.722** / 0.398 | 0.813 | **0.234** / 59 µM | 0.227 / 0.286 |

---

## 4. Live V33 Inference Service Benchmark (`node23:8902`) & Encoder Cosine Parity

- **Runtime**: `/usr/bin/python3` (`http.server.ThreadingHTTPServer` on `0.0.0.0:8902`) + persistent `esmc_worker.py` (`/data/qlyu/esmfold2-env312/bin/python` on `127.0.0.1:18902`, warm ESM-C 600M in RAM).
- **Startup & Memory**: Cold startup **`215.75s`** | Pruned champion serving joblibs **`10460.0 MB`** (`22` CPU + `21` Fusion + `20` Deep endpoints) | Post-verification RSS **`23193.3 MB`** | GPU usage **`0`**.
- **Assay Condition Routing**: Semaglutide `half_life_reg` responds cleanly to route condition (`pk_route='IV'`: `21.57h` vs `pk_route='SC'`: `51.676h`).
- **Batched Inference (`POST /batch`, 5 diverse molecules × 24 endpoints = 120 predictions)**: Succeeded **`5/5`** in **`25482.07 ms`** (encode `0.31 ms`, predict `25387.0 ms`, AD+Top-3 analogs `33.63 ms`).

### 4.1 Cached vs. `force_live_encode=true` Embedding Cosine Parity & Latency

| Test Molecule | Track | Cached Wall (ms) | Live-Encoded Wall (ms) | AD + Top-3 (ms) | Min Cosine Across Encoders | Max 24-EP Pred Diff | Mechanism Anchor | Key Live Predictions (t1/2 / PPB / Hemol / E.coli MIC) |
|---|---|---:|---:|---:|---:|---:|---|---|
| **Semaglutide** | `smiles_only` | 6582.1 | 7296.0 | 8.0 | **0.999042** | 0.010710 | `weekly_diacid` | 21.148h / 0.7991 / 0.097 / 15.871 µM |
| **Tirzepatide** | `smiles_only` | 7183.9 | 8817.5 | 12.3 | **0.996829** | 0.035870 | `weekly_diacid` | 30.866h / 0.8075 / 0.179 / 9.2685 µM |
| **Cyclosporine_A** | `smiles_only` | 6628.2 | 6457.7 | 16.1 | **0.999481** | 0.009500 | `non_acylated` | 8.5915h / 0.8955 / 0.293 / 26.754 µM |
| **Bank_AMP_MKTKAAVLFETHKPFEIVELEL** | `dual_modality` | 18784.4 | 20547.1 | 9.6 | **0.998971** | 0.010890 | `non_acylated` | 0.8675h / 0.5092 / 0.028 / 59.986 µM |
| **AMP_HISEAVKIFGCKWKENWKWNSNC** | `dual_modality` | 20066.3 | 19204.0 | 12.8 | **1.000000** | 0.000000 | `non_acylated` | 3.387h / 0.5491 / 0.122 / 20.309 µM |

### 4.2 Brand-New Unseen Synthetic Peptides (Out-of-Cache Live Encoding)

| Unseen Peptide | Track | `cache_hit` | Wall (ms) | Encode / Predict / AD (ms) | Mechanism Anchor | Key Predictions (t1/2 / PPB / Hemol / E.coli MIC & AD) |
|---|---|:---:|---:|---|---|---|
| **Unseen_AMP_KLLKLLKKLLKLLK** | `dual_modality` | `False` | **19377.7** | 342.1 / 19013.9 / 7.3 | `non_acylated` | 6.1358h / 0.5585 / 0.709 / 6.6504 µM (`In-Domain`, T=0.90) |
| **Unseen_C18Diacid_NMe_Tetrapeptide** | `smiles_only` | `False` | **6646.3** | 203.0 / 6417.9 / 9.3 | `weekly_diacid` | 2.8298h / 0.9406 / 0.336 / 42.397 µM (`Borderline`, T=0.55) |

---

## 5. Verification & Isolation Receipt

- **All Automated Verification Gates**: `PASSED` (`total_verify_sec = 213.26s`, receipt: `D:\work\中检院\v33_upgrade\bundle\verification_receipt.json`)
- **Zero Holdout Peeking & Zero Leakage**: Champion selection, nested simplex fitting, Platt/Isotonic calibration, and Conformal quantiles executed exclusively on inner 5-fold cluster-grouped OOF (`is_holdout == 0`); 20% outer holdout (`is_holdout == 1`) and Leave-Case17-Out (`case17_excluded == 0`) evaluated strictly once with frozen weights & calibrators; `0` NaN/Inf across all 24 endpoints and all SHA-256 checksums verified (`43` pruned serving joblibs + AD index + analog lookup).
- **Live V32 Service (`node23:8901`) Untouched**: `GET http://127.0.0.1:8901/health` -> **HTTP `200`** (`{"ok": true, "models_cached": 152, "entries": 57, "endpoints": 55}`).
- **Zero GPU Contention**: Entire Wave 3A executed strictly on `node23` CPUs (`0` GPU usage on `node1`).
- **Read-Only Upstream Guarantee**: Zero files modified in legacy data/models, `data/`, `features/`, `embeddings/`, `models_cpu/`, `models_fusion/`, or `models_gpu_deep/`.
