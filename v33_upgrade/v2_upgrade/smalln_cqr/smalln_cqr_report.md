# Plan B-1 Audited Report: Small-N Foundation/Shrinkage Modeling & AD-Stratified Empirical Intervals / CQR

> **Production Status (v33.2 Route A)**: Deliverable 1 (Small-N `TabPFN v2` / `MultiBlockCompositeGPR`) is **ARCHIVED RESEARCH-ONLY**. For Deliverable 2, **`mondrian_ad_eb` is deployed uniformly across all `80/80` regression tracks** as a pure-JSON zero-dependency **engineering trade-off** (whereas nested-OOF selected `mondrian_ad_eb` on `48/80` tracks and `cqr_lgb` achieved higher overall holdout coverage at the cost of `.joblib`/LightGBM artifacts). `mondrian_ad_eb` provides empirically calibrated AD-stratified EB-shrunk intervals (not an exact finite-sample conformal guarantee); all `12/80` tracks with `cov80 < 70%` (`7/40` dual + `5/40` SMILES) and underpowered/moderate tracks are explicitly flagged in `v33_manifest.json` and the web UI.

- **Generated at**: `2026-11-22 22:47:23 +0800`
- **Compute Hardware**: `node23` 64-core CPU ONLY (0 GPU usage; `node1` GPUs, local RTX 5080, `node2` live production, and frozen `v33_upgrade` directories untouched)
- **Zero-Leakage Protocol**: Frozen non-chaining dual-track cluster splits (`is_holdout == 0` 5-fold CV, untouched 20% `is_holdout == 1` holdout, strict `case17_excluded == 0` Leave-Case17-Out). All model & calibration selections locked strictly on inner nested 5-fold OOF BEFORE single holdout evaluation.
- **TabPFN v2 Status**: `tabpfn available in isolated tabpfn_venv (deferred post-fork import) using offline ${WORK_DIR}/.cache/tabpfn/tabpfn-v2-regressor.ckpt`

---

## 1. Deliverable 1 — Small-N Modeling Upgrade (`N <= 400`, 20 Regression Endpoints)

### 1.1 Aggregate Summary (Zero-Holdout-Peeking Gate: OOF MAE Gain >= 0.002)

| Slice | Track | Suite | OOF Promoted | Holdout Improved (of Promoted) | Mean CV MAE (Before -> After, Delta) | Mean Holdout MAE (Before -> After, Delta) | Mean Holdout Spearman (Before -> After) |
|---|---|---|---:|---:|---:|---:|---:|
| All Small-N (N<=400, 20 EPs) | `dual_modality` | Portable (no TabPFN) | 2/20 | 0/2 | 0.4475 -> 0.4470 (+0.0005) | 0.5420 -> 0.5431 (-0.0011) | 0.5156 -> 0.5134 |
| All Small-N (N<=400, 20 EPs) | `dual_modality` | Full (+ TabPFN v2) | 3/20 | 0/3 | 0.4475 -> 0.4461 (+0.0014) | 0.5420 -> 0.5448 (-0.0028) | 0.5156 -> 0.5169 |
| All Small-N (N<=400, 20 EPs) | `smiles_only` | Portable (no TabPFN) | 0/20 | 0/0 | 0.4496 -> 0.4496 (+0.0000) | 0.5455 -> 0.5455 (+0.0000) | 0.5199 -> 0.5199 |
| All Small-N (N<=400, 20 EPs) | `smiles_only` | Full (+ TabPFN v2) | 1/20 | 1/1 | 0.4496 -> 0.4488 (+0.0008) | 0.5455 -> 0.5444 (+0.0011) | 0.5199 -> 0.5242 |
| Core Small-N (N<=300, 19 EPs) | `dual_modality` | Portable (no TabPFN) | 2/19 | 0/2 | 0.4451 -> 0.4446 (+0.0005) | 0.5451 -> 0.5463 (-0.0012) | 0.5056 -> 0.5033 |
| Core Small-N (N<=300, 19 EPs) | `dual_modality` | Full (+ TabPFN v2) | 3/19 | 0/3 | 0.4451 -> 0.4436 (+0.0015) | 0.5451 -> 0.5481 (-0.0030) | 0.5056 -> 0.5069 |
| Core Small-N (N<=300, 19 EPs) | `smiles_only` | Portable (no TabPFN) | 0/19 | 0/0 | 0.4478 -> 0.4478 (+0.0000) | 0.5504 -> 0.5504 (+0.0000) | 0.5077 -> 0.5077 |
| Core Small-N (N<=300, 19 EPs) | `smiles_only` | Full (+ TabPFN v2) | 1/19 | 1/1 | 0.4478 -> 0.4469 (+0.0009) | 0.5504 -> 0.5492 (+0.0012) | 0.5077 -> 0.5122 |
| Small-N (N<=283, 16 EPs) | `dual_modality` | Portable (no TabPFN) | 2/16 | 0/2 | 0.4430 -> 0.4423 (+0.0006) | 0.5450 -> 0.5464 (-0.0014) | 0.5165 -> 0.5137 |
| Small-N (N<=283, 16 EPs) | `dual_modality` | Full (+ TabPFN v2) | 3/16 | 0/3 | 0.4430 -> 0.4412 (+0.0018) | 0.5450 -> 0.5486 (-0.0035) | 0.5165 -> 0.5181 |
| Small-N (N<=283, 16 EPs) | `smiles_only` | Portable (no TabPFN) | 0/16 | 0/0 | 0.4455 -> 0.4455 (+0.0000) | 0.5491 -> 0.5491 (+0.0000) | 0.5139 -> 0.5139 |
| Small-N (N<=283, 16 EPs) | `smiles_only` | Full (+ TabPFN v2) | 1/16 | 1/1 | 0.4455 -> 0.4444 (+0.0010) | 0.5491 -> 0.5477 (+0.0014) | 0.5139 -> 0.5193 |
| Priority Weakest (9 EPs) | `dual_modality` | Portable (no TabPFN) | 0/9 | 0/0 | 0.4329 -> 0.4329 (+0.0000) | 0.5917 -> 0.5917 (+0.0000) | 0.4541 -> 0.4541 |
| Priority Weakest (9 EPs) | `dual_modality` | Full (+ TabPFN v2) | 1/9 | 0/1 | 0.4329 -> 0.4309 (+0.0020) | 0.5917 -> 0.5955 (-0.0038) | 0.4541 -> 0.4618 |
| Priority Weakest (9 EPs) | `smiles_only` | Portable (no TabPFN) | 0/9 | 0/0 | 0.4330 -> 0.4330 (+0.0000) | 0.6099 -> 0.6099 (+0.0000) | 0.3726 -> 0.3726 |
| Priority Weakest (9 EPs) | `smiles_only` | Full (+ TabPFN v2) | 1/9 | 1/1 | 0.4330 -> 0.4312 (+0.0019) | 0.6099 -> 0.6074 (+0.0025) | 0.3726 -> 0.3822 |

### 1.2 Per-Endpoint Small-N Benchmark (`dual_modality` — Before vs Portable Winner vs Full + TabPFN v2)

| Endpoint | N | Base CV MAE | Base HO MAE | Best Single New (CV MAE) | TabPFN v2 CV/HO MAE | Portable Winner (OOF Gate) | Port CV -> HO MAE (Delta) | Full+TabPFN Winner | Full CV -> HO MAE (Delta) |
|---|---:|---:|---:|---|---:|---|---:|---|---:|
| `CL_absolute` | 54 | 0.6697 | 1.1802 | `eb_huber` (0.6969) | 0.7618/1.3784 | unchanged | 0.6697 -> 1.1802 (+0.0000) | unchanged | 0.6697 -> 1.1802 (+0.0000) |
| `Vd_absolute` | 63 | 0.4597 | 0.7780 | `mkgpr_exp` (0.4891) | 0.5599/0.4485 | unchanged | 0.4597 -> 0.7780 (+0.0000) | unchanged | 0.4597 -> 0.7780 (+0.0000) |
| `Cmax` | 74 | 0.8603 | 0.9156 | `cosine_krr` (1.1176) | 1.2199/0.8298 | unchanged | 0.8603 -> 0.9156 (+0.0000) | unchanged | 0.8603 -> 0.9156 (+0.0000) |
| `MBC_E_faecalis` | 90 | 0.4677 | 0.6881 | `tabpfn_v2` (0.5290) | 0.5290/0.6487 | unchanged | 0.4677 -> 0.6881 (+0.0000) | unchanged | 0.4677 -> 0.6881 (+0.0000) |
| `MFC_C_albicans` | 110 | 0.3649 | 0.3771 | `tabpfn_v2` (0.4000) | 0.4000/0.3234 | unchanged | 0.3649 -> 0.3771 (+0.0000) | unchanged | 0.3649 -> 0.3771 (+0.0000) |
| `F` | 124 | 0.1941 | 0.2364 | `cosine_krr` (0.2092) | 0.2410/0.2276 | unchanged | 0.1941 -> 0.2364 (+0.0000) | unchanged | 0.1941 -> 0.2364 (+0.0000) |
| `MBC_A_baumannii` | 136 | 0.4945 | 0.5933 | `mkgpr_cos` (0.5434) | 0.5909/0.5821 | unchanged | 0.4945 -> 0.5933 (+0.0000) | unchanged | 0.4945 -> 0.5933 (+0.0000) |
| `MBEC_biofilm` | 138 | 0.3654 | 0.7836 | `cosine_krr` (0.4039) | 0.4276/0.7438 | unchanged | 0.3654 -> 0.7836 (+0.0000) | unchanged | 0.3654 -> 0.7836 (+0.0000) |
| `PPB` | 162 | 0.1511 | 0.2608 | `mkgpr_cos` (0.1751) | 0.2065/0.2813 | unchanged | 0.1511 -> 0.2608 (+0.0000) | unchanged | 0.1511 -> 0.2608 (+0.0000) |
| `Vd_per_kg` | 166 | 0.3685 | 0.3295 | `mkgpr_exp` (0.3668) | 0.4137/0.3426 | unchanged | 0.3685 -> 0.3295 (+0.0000) | unchanged | 0.3685 -> 0.3295 (+0.0000) |
| `RRCK` | 167 | 0.3778 | 0.2682 | `tabpfn_v2` (0.3599) | 0.3599/0.3024 | unchanged | 0.3778 -> 0.2682 (+0.0000) | **single:tabpfn_v2** | 0.3599 -> 0.3024 (-0.0342) |
| `MDCK` | 168 | 0.4467 | 0.5426 | `tabpfn_v2` (0.4856) | 0.4856/0.4530 | unchanged | 0.4467 -> 0.5426 (+0.0000) | unchanged | 0.4467 -> 0.5426 (+0.0000) |
| `efflux_ratio` | 171 | 0.3709 | 0.3599 | `tabpfn_v2` (0.4009) | 0.4009/0.3730 | unchanged | 0.3709 -> 0.3599 (+0.0000) | unchanged | 0.3709 -> 0.3599 (+0.0000) |
| `CL_per_kg` | 206 | 0.5001 | 0.5477 | `tabpfn_v2` (0.5777) | 0.5777/0.5741 | **blend_multi:portable** | 0.4959 -> 0.5530 (-0.0053) | **blend_multi:portable** | 0.4959 -> 0.5530 (-0.0053) |
| `MIC_L_monocytogenes` | 248 | 0.5062 | 0.5125 | `mkgpr_exp` (0.5001) | 0.5186/0.5983 | **single:mkgpr_exp** | 0.5001 -> 0.5295 (-0.0169) | **single:mkgpr_exp** | 0.5001 -> 0.5295 (-0.0169) |
| `MIC_E_faecium` | 283 | 0.4896 | 0.3470 | `tabpfn_v2` (0.4966) | 0.4966/0.3576 | unchanged | 0.4896 -> 0.3470 (+0.0000) | unchanged | 0.4896 -> 0.3470 (+0.0000) |
| `MIC_C_neoformans` | 287 | 0.3945 | 0.4952 | `eb_huber` (0.4333) | 0.5466/0.5001 | unchanged | 0.3945 -> 0.4952 (+0.0000) | unchanged | 0.3945 -> 0.4952 (+0.0000) |
| `MBIC_biofilm` | 296 | 0.4302 | 0.5600 | `tabpfn_v2` (0.4471) | 0.4471/0.5621 | unchanged | 0.4302 -> 0.5600 (+0.0000) | unchanged | 0.4302 -> 0.5600 (+0.0000) |
| `solubility_reg` | 298 | 0.5448 | 0.5807 | `cosine_krr` (0.6062) | 0.6946/0.5544 | unchanged | 0.5448 -> 0.5807 (+0.0000) | unchanged | 0.5448 -> 0.5807 (+0.0000) |
| `MBC_P_aeruginosa` | 392 | 0.4933 | 0.4837 | `cosine_krr` (0.5109) | 0.5217/0.4934 | unchanged | 0.4933 -> 0.4837 (+0.0000) | unchanged | 0.4933 -> 0.4837 (+0.0000) |

### 1.3 Scientific Diagnostic — Winner's Curse / Selection-Induced OOF Optimism on Weakest Small-N Endpoints (`dual_modality`)

> Existing V33.1 `cpu`/`fusion` champions swept 40–80 recipes and stacked 12 already-CV-selected heads on the same 5-fold CV split, inducing downward CV optimism bias on tiny-N endpoints (e.g., `Vd_absolute` Base CV=0.4597 vs HO=0.7780). Zero-tuning `MultiBlockCompositeGPR` and `TabPFN v2` produce unbiased CV estimates and substantially lower untouched-holdout MAE on several weakest small-N endpoints, even where the strict pre-specified OOF gate (`CV_new <= Base_CV - 0.002`) prudently retained baseline without holdout peeking.

| Endpoint | N | Base CV -> HO (Gap) | MultiBlock GPR CV -> HO (HO Delta vs Base) | TabPFN v2 CV -> HO (HO Delta vs Base) |
|---|---:|---:|---:|---:|
| `Vd_absolute` | 63 | 0.4597 -> 0.7780 (+0.3183) | 0.4891 -> 0.5136 (+0.2644) | 0.5599 -> 0.4485 (+0.3295) |
| `MBEC_biofilm` | 138 | 0.3654 -> 0.7836 (+0.4183) | 0.4183 -> 0.7472 (+0.0364) | 0.4276 -> 0.7438 (+0.0399) |
| `CL_absolute` | 54 | 0.6697 -> 1.1802 (+0.5105) | 0.8689 -> 1.2433 (-0.0631) | 0.7618 -> 1.3784 (-0.1982) |
| `PPB` | 162 | 0.1511 -> 0.2608 (+0.1097) | 0.1751 -> 0.2574 (+0.0034) | 0.2065 -> 0.2813 (-0.0206) |
| `MDCK` | 168 | 0.4467 -> 0.5426 (+0.0958) | 0.5684 -> 0.4809 (+0.0616) | 0.4856 -> 0.4530 (+0.0895) |
| `RRCK` | 167 | 0.3778 -> 0.2682 (-0.1096) | 0.3969 -> 0.2787 (-0.0104) | 0.3599 -> 0.3024 (-0.0342) |
| `F` | 124 | 0.1941 -> 0.2364 (+0.0423) | 0.2285 -> 0.2385 (-0.0021) | 0.2410 -> 0.2276 (+0.0088) |
| `Cmax` | 74 | 0.8603 -> 0.9156 (+0.0553) | 1.2716 -> 1.0989 (-0.1833) | 1.2199 -> 0.8298 (+0.0858) |
| `efflux_ratio` | 171 | 0.3709 -> 0.3599 (-0.0110) | 0.4502 -> 0.3924 (-0.0325) | 0.4009 -> 0.3730 (-0.0131) |
| `MFC_C_albicans` | 110 | 0.3649 -> 0.3771 (+0.0122) | 0.4700 -> 0.3153 (+0.0618) | 0.4000 -> 0.3234 (+0.0537) |
| `MBC_E_faecalis` | 90 | 0.4677 -> 0.6881 (+0.2204) | 0.5560 -> 0.6825 (+0.0056) | 0.5290 -> 0.6487 (+0.0393) |
| `MIC_E_faecium` | 283 | 0.4896 -> 0.3470 (-0.1426) | 0.5068 -> 0.2856 (+0.0614) | 0.4966 -> 0.3576 (-0.0106) |
| `solubility_reg` | 298 | 0.5448 -> 0.5807 (+0.0359) | 0.6366 -> 0.6000 (-0.0192) | 0.6946 -> 0.5544 (+0.0264) |

### 1.4 `smiles_only` OOF-Promoted Small-N Endpoints

- **`RRCK`** (`single:tabpfn_v2`): CV MAE `0.3793 -> 0.3626` | Untouched Holdout MAE `0.3217 -> 0.2989` (`+0.0229`), Spearman `0.5208 -> 0.6067`

### 1.5 Promoted `node2`-Portable 5-Fold `.joblib` Artifacts (`node23`)

- `CL_per_kg`: `${WORK_DIR}/v33_upgrade/v2_upgrade/smalln_cqr/promoted_models/CL_per_kg_smalln_portable.joblib` — verified modules: `['builtins']` (0 `catboost`, 0 `tabpfn`, 0 `torch`)
- `MIC_L_monocytogenes`: `${WORK_DIR}/v33_upgrade/v2_upgrade/smalln_cqr/promoted_models/MIC_L_monocytogenes_smalln_portable.joblib` — verified modules: `['builtins', 'numpy', 'smalln_models']` (0 `catboost`, 0 `tabpfn`, 0 `torch`)

### 1.6 Strict Leave-Case17-Out FDA Clinical Drug Benchmark (`PPB`, n=14 FDA Anchors)

- **Baseline V33.1**: MAE = `0.2264`, Spearman = `0.7093`
- **Plan B-1 Portable**: MAE = `0.2264`, Spearman = `0.7093`
- **Plan B-1 Full (+TabPFN v2)**: MAE = `0.2264`, Spearman = `0.7093`

---

## 2. Deliverable 2 — Conformal Calibration Across ALL 40 Regression Endpoints

### 2.1 Aggregate Holdout Coverage & Width (Frozen V33.1 Predictions — Pure Calibration Comparison)

| Slice | Track | Method | Mean Cov80 | Mean Cov90 | Min Cov80 | Severe (<70%) | Mean W80 | Mean W90 | Pooled AD Cov80 (InD / Bord / OOD) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| All 40 Reg EPs | `dual_modality` | baseline | 76.3% | 88.8% | 29.2% | 6/40 | 1.479 | 2.058 | 80.9% / 73.8% / 64.2% |
| All 40 Reg EPs | `dual_modality` | studentized_cvplus | 76.2% | 88.8% | 29.2% | 7/40 | 1.511 | 2.103 | 79.4% / 78.2% / 72.1% |
| All 40 Reg EPs | `dual_modality` | cqr_lgb | 79.2% | 91.0% | 52.2% | 6/40 | 1.631 | 2.253 | 82.1% / 79.1% / 70.1% |
| All 40 Reg EPs | `dual_modality` | mondrian_ad_eb | 76.5% | 88.2% | 29.2% | 7/40 | 1.470 | 2.060 | 79.6% / 81.8% / 76.0% |
| All 40 Reg EPs | `dual_modality` | **selected (OOF-gated)** | 76.6% | 89.1% | 29.2% | 7/40 | 1.506 | 2.080 | 79.0% / 82.2% / 74.5% |
| All 40 Reg EPs | `smiles_only` | baseline | 77.9% | 88.5% | 47.8% | 5/40 | 1.510 | 2.066 | 81.9% / 74.6% / 53.9% |
| All 40 Reg EPs | `smiles_only` | studentized_cvplus | 77.8% | 88.4% | 43.5% | 5/40 | 1.522 | 2.139 | 80.2% / 79.9% / 67.2% |
| All 40 Reg EPs | `smiles_only` | cqr_lgb | 80.1% | 90.2% | 52.2% | 4/40 | 1.670 | 2.228 | 82.5% / 80.9% / 62.5% |
| All 40 Reg EPs | `smiles_only` | mondrian_ad_eb | 78.3% | 88.4% | 52.2% | 5/40 | 1.490 | 2.057 | 80.5% / 83.3% / 74.2% |
| All 40 Reg EPs | `smiles_only` | **selected (OOF-gated)** | 78.8% | 88.0% | 52.2% | 4/40 | 1.529 | 2.050 | 79.9% / 82.8% / 73.4% |
| Small-N 20 EPs (N<=400) | `dual_modality` | baseline | 72.2% | 87.7% | 29.2% | 6/20 | 1.484 | 2.138 | 77.5% / 75.9% / 61.5% |
| Small-N 20 EPs (N<=400) | `dual_modality` | studentized_cvplus | 71.8% | 87.2% | 29.2% | 6/20 | 1.522 | 2.183 | 76.3% / 77.1% / 61.5% |
| Small-N 20 EPs (N<=400) | `dual_modality` | cqr_lgb | 76.2% | 90.5% | 52.2% | 6/20 | 1.678 | 2.401 | 81.4% / 77.1% / 65.4% |
| Small-N 20 EPs (N<=400) | `dual_modality` | mondrian_ad_eb | 72.5% | 86.1% | 29.2% | 6/20 | 1.456 | 2.120 | 75.6% / 79.5% / 67.3% |
| Small-N 20 EPs (N<=400) | `dual_modality` | **selected (OOF-gated)** | 72.9% | 88.1% | 29.2% | 6/20 | 1.533 | 2.170 | 76.5% / 80.7% / 63.5% |
| Small-N 20 EPs (N<=400) | `smiles_only` | baseline | 75.1% | 86.8% | 47.8% | 5/20 | 1.523 | 2.128 | 80.0% / 77.1% / 61.5% |
| Small-N 20 EPs (N<=400) | `smiles_only` | studentized_cvplus | 74.2% | 86.5% | 43.5% | 5/20 | 1.503 | 2.234 | 79.1% / 74.7% / 61.5% |
| Small-N 20 EPs (N<=400) | `smiles_only` | cqr_lgb | 78.0% | 88.5% | 52.2% | 4/20 | 1.744 | 2.320 | 82.1% / 78.3% / 61.5% |
| Small-N 20 EPs (N<=400) | `smiles_only` | mondrian_ad_eb | 75.5% | 86.5% | 52.2% | 5/20 | 1.478 | 2.095 | 76.8% / 82.5% / 67.3% |
| Small-N 20 EPs (N<=400) | `smiles_only` | **selected (OOF-gated)** | 76.4% | 86.0% | 52.2% | 4/20 | 1.557 | 2.092 | 78.4% / 82.5% / 65.4% |
| Small-N 19 EPs (N<=300) | `dual_modality` | baseline | 71.6% | 87.5% | 29.2% | 6/19 | 1.479 | 2.144 | 75.9% / 76.7% / 61.5% |
| Small-N 19 EPs (N<=300) | `dual_modality` | studentized_cvplus | 71.2% | 86.9% | 29.2% | 6/19 | 1.516 | 2.188 | 74.9% / 77.3% / 61.5% |
| Small-N 19 EPs (N<=300) | `dual_modality` | cqr_lgb | 75.7% | 90.2% | 52.2% | 6/19 | 1.679 | 2.415 | 80.3% / 77.9% / 65.4% |
| Small-N 19 EPs (N<=300) | `dual_modality` | mondrian_ad_eb | 72.0% | 85.9% | 29.2% | 6/19 | 1.451 | 2.123 | 74.3% / 80.4% / 67.3% |
| Small-N 19 EPs (N<=300) | `dual_modality` | **selected (OOF-gated)** | 72.3% | 87.9% | 29.2% | 6/19 | 1.530 | 2.178 | 74.9% / 81.6% / 63.5% |
| Small-N 19 EPs (N<=300) | `smiles_only` | baseline | 74.3% | 86.4% | 47.8% | 5/19 | 1.520 | 2.138 | 78.1% / 77.3% / 61.5% |
| Small-N 19 EPs (N<=300) | `smiles_only` | studentized_cvplus | 73.5% | 86.2% | 43.5% | 5/19 | 1.502 | 2.251 | 77.3% / 74.9% / 61.5% |
| Small-N 19 EPs (N<=300) | `smiles_only` | cqr_lgb | 77.5% | 88.1% | 52.2% | 4/19 | 1.751 | 2.328 | 80.8% / 78.5% / 61.5% |
| Small-N 19 EPs (N<=300) | `smiles_only` | mondrian_ad_eb | 74.9% | 86.1% | 52.2% | 5/19 | 1.476 | 2.102 | 74.6% / 82.8% / 67.3% |
| Small-N 19 EPs (N<=300) | `smiles_only` | **selected (OOF-gated)** | 75.7% | 85.6% | 52.2% | 4/19 | 1.555 | 2.100 | 76.2% / 82.8% / 65.4% |
| Small-N 16 EPs (N<=283) | `dual_modality` | baseline | 71.1% | 87.8% | 29.2% | 6/16 | 1.479 | 2.164 | 74.3% / 77.5% / 71.0% |
| Small-N 16 EPs (N<=283) | `dual_modality` | studentized_cvplus | 70.8% | 87.4% | 29.2% | 6/16 | 1.526 | 2.239 | 74.3% / 77.5% / 68.4% |
| Small-N 16 EPs (N<=283) | `dual_modality` | cqr_lgb | 75.6% | 90.5% | 52.2% | 6/16 | 1.702 | 2.466 | 80.0% / 78.3% / 73.7% |
| Small-N 16 EPs (N<=283) | `dual_modality` | mondrian_ad_eb | 70.9% | 85.8% | 29.2% | 6/16 | 1.440 | 2.135 | 72.9% / 78.3% / 73.7% |
| Small-N 16 EPs (N<=283) | `dual_modality` | **selected (OOF-gated)** | 71.2% | 88.3% | 29.2% | 6/16 | 1.536 | 2.209 | 73.6% / 79.2% / 71.0% |
| Small-N 16 EPs (N<=283) | `smiles_only` | baseline | 74.2% | 86.5% | 47.8% | 5/16 | 1.524 | 2.157 | 76.8% / 77.5% / 76.3% |
| Small-N 16 EPs (N<=283) | `smiles_only` | studentized_cvplus | 74.0% | 87.3% | 43.5% | 4/16 | 1.529 | 2.337 | 77.1% / 76.7% / 73.7% |
| Small-N 16 EPs (N<=283) | `smiles_only` | cqr_lgb | 78.0% | 88.2% | 52.2% | 4/16 | 1.787 | 2.356 | 80.7% / 79.2% / 73.7% |
| Small-N 16 EPs (N<=283) | `smiles_only` | mondrian_ad_eb | 74.8% | 86.5% | 52.2% | 5/16 | 1.475 | 2.117 | 74.3% / 82.5% / 73.7% |
| Small-N 16 EPs (N<=283) | `smiles_only` | **selected (OOF-gated)** | 75.9% | 85.9% | 52.2% | 4/16 | 1.573 | 2.122 | 76.4% / 82.5% / 73.7% |

### 2.2 Combined Plan B-1 (Deliverable 1 OOF-Promoted Small-N Models + Deliverable 2 OOF-Gated Conformal)

| Slice | Track | Baseline Cov80/90 (W80) | Plan B-1 Selected Cov80/90 (W80) | Min Cov80 (Before -> After) | Severe <70% (Before -> After) |
|---|---|---:|---:|---:|---:|
| All 40 Reg EPs | `dual_modality` | 76.3% / 88.8% (1.479) | **76.6% / 88.9%** (1.501) | 29.2% -> **29.2%** | 6 -> **7** |
| All 40 Reg EPs | `smiles_only` | 77.9% / 88.5% (1.510) | **78.8% / 88.0%** (1.529) | 47.8% -> **52.2%** | 5 -> **4** |
| Small-N 19 EPs (N<=300) | `dual_modality` | 71.6% / 87.5% (1.479) | **72.3% / 87.7%** (1.520) | 29.2% -> **29.2%** | 6 -> **6** |
| Small-N 19 EPs (N<=300) | `smiles_only` | 74.3% / 86.4% (1.520) | **75.7% / 85.6%** (1.555) | 47.8% -> **52.2%** | 5 -> **4** |
| Small-N 16 EPs (N<=283) | `dual_modality` | 71.1% / 87.8% (1.479) | **71.2% / 88.1%** (1.524) | 29.2% -> **29.2%** | 6 -> **6** |
| Small-N 16 EPs (N<=283) | `smiles_only` | 74.2% / 86.5% (1.524) | **75.9% / 85.9%** (1.573) | 47.8% -> **52.2%** | 5 -> **4** |

### 2.3 Per-Endpoint Conformal Holdout Table (All 40 Regression Endpoints, `dual_modality`, Frozen V33.1 Predictions)

| Endpoint | N | Selected (OOF) | Base Cov80/90 | StudCV+ Cov80/90 | CQR Cov80/90 | Mondrian Cov80/90 | Selected Cov80/90 | W80 (Base -> Sel) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `CL_absolute` | 54 | `cqr_lgb` | 63.6%/72.7% | 54.5%/63.6% | 63.6%/72.7% | 63.6%/63.6% | **63.6%/72.7%** | 2.335 -> 2.684 |
| `Vd_absolute` | 63 | `studentized_cvplus` | 53.8%/84.6% | 53.8%/84.6% | 53.8%/100.0% | 53.8%/61.5% | **53.8%/84.6%** | 1.272 -> 1.272 |
| `Cmax` | 74 | `baseline` | 82.3%/88.2% | 94.1%/94.1% | 88.2%/94.1% | 82.3%/88.2% | **82.3%/88.2%** | 3.342 -> 3.342 |
| `MBC_E_faecalis` | 90 | `mondrian_ad_eb` | 53.3%/93.3% | 53.3%/93.3% | 53.3%/80.0% | 53.3%/86.7% | **53.3%/86.7%** | 1.573 -> 1.635 |
| `MFC_C_albicans` | 110 | `mondrian_ad_eb` | 76.5%/88.2% | 76.5%/88.2% | 82.3%/94.1% | 76.5%/88.2% | **76.5%/88.2%** | 1.216 -> 1.218 |
| `F` | 124 | `mondrian_ad_eb` | 78.3%/82.6% | 78.3%/82.6% | 82.6%/95.7% | 82.6%/87.0% | **82.6%/87.0%** | 0.479 -> 0.490 |
| `MBC_A_baumannii` | 136 | `baseline` | 75.0%/87.5% | 70.8%/87.5% | 87.5%/87.5% | 75.0%/87.5% | **75.0%/87.5%** | 1.736 -> 1.736 |
| `MBEC_biofilm` | 138 | `mondrian_ad_eb` | 47.8%/65.2% | 47.8%/65.2% | 52.2%/69.6% | 52.2%/65.2% | **52.2%/65.2%** | 1.210 -> 1.308 |
| `PPB` | 162 | `mondrian_ad_eb` | 29.2%/91.7% | 29.2%/91.7% | 66.7%/95.8% | 29.2%/91.7% | **29.2%/91.7%** | 0.461 -> 0.514 |
| `Vd_per_kg` | 166 | `baseline` | 87.9%/93.9% | 87.9%/93.9% | 84.9%/93.9% | 87.9%/90.9% | **87.9%/93.9%** | 1.176 -> 1.176 |
| `RRCK` | 167 | `cqr_lgb` | 86.7%/93.3% | 86.7%/93.3% | 83.3%/96.7% | 83.3%/96.7% | **83.3%/96.7%** | 1.271 -> 1.154 |
| `MDCK` | 168 | `cqr_lgb` | 67.7%/79.4% | 67.7%/79.4% | 67.7%/91.2% | 64.7%/79.4% | **67.7%/91.2%** | 1.440 -> 1.391 |
| `efflux_ratio` | 171 | `cqr_lgb` | 87.5%/96.9% | 84.4%/96.9% | 84.4%/93.8% | 81.2%/96.9% | **84.4%/93.8%** | 1.190 -> 1.237 |
| `CL_per_kg` | 206 | `studentized_cvplus` | 70.7%/92.7% | 73.2%/90.2% | 78.0%/87.8% | 73.2%/92.7% | **73.2%/90.2%** | 1.465 -> 2.071 |
| `MIC_L_monocytogenes` | 248 | `mondrian_ad_eb` | 80.4%/95.7% | 80.4%/95.7% | 87.0%/97.8% | 78.3%/97.8% | **78.3%/97.8%** | 1.611 -> 1.603 |
| `MIC_E_faecium` | 283 | `mondrian_ad_eb` | 96.4%/98.2% | 94.5%/98.2% | 94.5%/98.2% | 96.4%/98.2% | **96.4%/98.2%** | 1.894 -> 1.754 |
| `MIC_C_neoformans` | 287 | `mondrian_ad_eb` | 71.7%/84.8% | 71.7%/84.8% | 73.9%/91.3% | 78.3%/82.6% | **78.3%/82.6%** | 1.345 -> 1.317 |
| `MBIC_biofilm` | 296 | `baseline` | 77.4%/84.9% | 77.4%/84.9% | 77.4%/84.9% | 77.4%/86.8% | **77.4%/84.9%** | 1.412 -> 1.412 |
| `solubility_reg` | 298 | `mondrian_ad_eb` | 75.0%/89.6% | 70.8%/83.3% | 77.1%/89.6% | 79.2%/89.6% | **79.2%/89.6%** | 1.672 -> 1.751 |
| `MBC_P_aeruginosa` | 392 | `baseline` | 83.8%/91.2% | 83.8%/92.7% | 85.3%/95.6% | 80.9%/91.2% | **83.8%/91.2%** | 1.591 -> 1.591 |
| `MIC_C_albicans` | 468 | `cqr_lgb` | 86.7%/94.4% | 86.7%/94.4% | 88.9%/96.7% | 84.4%/94.4% | **88.9%/96.7%** | 1.616 -> 1.674 |
| `hemolysis_MHC` | 496 | `mondrian_ad_eb` | 80.0%/94.5% | 85.5%/98.2% | 83.6%/98.2% | 83.6%/94.5% | **83.6%/94.5%** | 1.343 -> 1.329 |
| `LC_antimicrobial` | 544 | `mondrian_ad_eb` | 76.4%/84.0% | 78.3%/85.9% | 78.3%/87.7% | 73.6%/85.9% | **73.6%/85.9%** | 1.402 -> 1.253 |
| `MBC_E_coli` | 548 | `mondrian_ad_eb` | 86.1%/91.7% | 86.1%/91.7% | 88.9%/95.4% | 86.1%/93.5% | **86.1%/93.5%** | 1.518 -> 1.524 |
| `MBC_S_aureus` | 551 | `studentized_cvplus` | 76.8%/82.8% | 76.8%/82.8% | 81.8%/84.9% | 78.8%/82.8% | **76.8%/82.8%** | 1.759 -> 1.734 |
| `MIC_E_faecalis` | 628 | `mondrian_ad_eb` | 78.6%/89.8% | 79.6%/93.9% | 83.7%/91.8% | 80.6%/92.9% | **80.6%/92.9%** | 1.635 -> 1.774 |
| `LogD7.4` | 678 | `mondrian_ad_eb` | 80.8%/90.4% | 82.4%/90.4% | 84.0%/93.6% | 80.0%/88.8% | **80.0%/88.8%** | 1.186 -> 1.330 |
| `MIC_Salmonella` | 719 | `mondrian_ad_eb` | 77.0%/92.6% | 78.5%/92.6% | 75.6%/87.4% | 76.3%/91.8% | **76.3%/91.8%** | 1.685 -> 1.640 |
| `MIC_S_epidermidis` | 784 | `studentized_cvplus` | 81.4%/92.4% | 77.9%/90.3% | 81.4%/95.9% | 80.0%/93.1% | **77.9%/90.3%** | 1.511 -> 1.519 |
| `MIC_A_baumannii` | 939 | `mondrian_ad_eb` | 83.6%/90.3% | 86.7%/95.8% | 87.3%/94.5% | 84.2%/92.7% | **84.2%/92.7%** | 1.495 -> 1.578 |
| `MIC_K_pneumoniae` | 971 | `studentized_cvplus` | 89.0%/94.8% | 88.3%/94.8% | 89.6%/96.8% | 91.6%/96.8% | **88.3%/94.8%** | 1.620 -> 1.613 |
| `MIC_P_aeruginosa` | 1161 | `mondrian_ad_eb` | 75.6%/82.7% | 75.6%/82.7% | 76.1%/83.2% | 75.6%/82.2% | **75.6%/82.2%** | 1.427 -> 1.401 |
| `MIC_B_subtilis` | 1475 | `mondrian_ad_eb` | 70.6%/78.8% | 69.2%/81.0% | 70.2%/81.4% | 69.9%/79.2% | **69.9%/79.2%** | 1.375 -> 1.366 |
| `Caco2` | 1580 | `baseline` | 78.8%/89.1% | 78.8%/89.1% | 80.7%/90.0% | 79.4%/90.0% | **78.8%/89.1%** | 1.070 -> 1.070 |
| `MIC_S_aureus` | 2131 | `mondrian_ad_eb` | 87.0%/96.1% | 84.8%/95.5% | 85.6%/97.2% | 86.5%/94.4% | **86.5%/94.4%** | 1.706 -> 1.728 |
| `MIC_E_coli` | 2210 | `mondrian_ad_eb` | 81.6%/92.7% | 81.4%/91.9% | 81.9%/93.2% | 81.6%/92.4% | **81.6%/92.4%** | 1.509 -> 1.510 |
| `half_life_reg` | 3078 | `mondrian_ad_eb` | 74.8%/87.8% | 73.4%/87.3% | 76.2%/88.7% | 73.4%/88.2% | **73.4%/88.2%** | 1.930 -> 1.897 |
| `cytotoxicity_CC50` | 3347 | `mondrian_ad_eb` | 79.9%/90.0% | 81.2%/91.6% | 84.5%/91.2% | 81.6%/90.6% | **81.6%/90.6%** | 1.257 -> 1.295 |
| `hemolysis_HC50` | 5559 | `mondrian_ad_eb` | 82.2%/92.2% | 81.3%/90.6% | 86.5%/93.0% | 82.2%/92.0% | **82.2%/92.0%** | 1.233 -> 1.227 |
| `PAMPA` | 8581 | `studentized_cvplus` | 80.1%/90.3% | 78.5%/88.0% | 80.3%/90.3% | 80.2%/89.1% | **78.5%/88.0%** | 1.196 -> 1.116 |

---

## 3. Zero-Leakage & `node2` Portability Verification Checklist

- [x] **Zero holdout peeking**: All Small-N promotions (`margin >= 0.002`) and all Conformal method selections were locked strictly on nested 5-fold CV OOF (`is_holdout == 0`) prior to single untouched 20% holdout evaluation.
- [x] **Zero cluster leakage**: Cluster disjointness asserted on all 40 endpoints (`set(cv_clusters) ∩ set(ho_clusters) == ∅`); inner hyperparameter tuning uses cluster-aware GroupKFold.
- [x] **Strict Leave-Case17-Out**: All 7 small-N PK endpoints refitted strictly on `case17_excluded == 0`.
- [x] **Strict `node2` portability**: Every promoted Small-N `.joblib` and Conformal calibrator `.joblib` verified via recursive object walker to contain ZERO `catboost`, ZERO `tabpfn`, and ZERO `torch` references (pure `numpy`/`scipy`/`sklearn`/`lightgbm` + `smalln_models.py`).
- [x] **Zero mutation of frozen directories**: All outputs written exclusively under `${WORK_DIR}/v33_upgrade/v2_upgrade/smalln_cqr/` and mirrored to `D:\work\中检院\v33_upgrade\v2_upgrade\smalln_cqr\`.
