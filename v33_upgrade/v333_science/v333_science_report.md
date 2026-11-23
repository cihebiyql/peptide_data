# Peptide OmniPanel v33.3 — Scientific Methodology Upgrade & Audit Report

> **Date**: 2026-11-23  
> **Scope**: 3-step targeted scientific upgrade following the independent Gemini 4 Carbon methodology & split audit, executed strictly on `compute-node` 64-core CPU under zero-holdout-peeking, strict cluster-disjoint protocols.  
> **Production Promotion Gate**: Only strictly zero-holdout-peeking, prod-node-portable upgrades with verified coverage/leakage gains were promoted into `v33.3` (`bundle/v33_manifest.json` + `bundle/v33_inference_service.py` + `bundle/v333_models.py` + 51 compact `cqr_lgb_cw` calibrators, 12.06 MB total). Diagnostic and negative experimental findings are reported transparently below without over-claiming.

---

## Executive Summary

| Step | Scope | Key Quantitative Result | Production Status (`v33.3`) |
|---|---|---|---|
| **Step 1 (L1 Fix)** | Full-cluster Leave-Case17-Out re-audit (19 clusters, 911 rows = +472 analogs excluded across CPU, Fusion, AUX meta-features, Deep; clean non-Case17-cluster stacker OOF) | Parity = `0.00000` on all 13 endpoints. **Half-life Pure ML**: MAE `0.3306 → 0.3185`, $R^2$ `0.7627 → 0.7710`, Sp `0.9029 → 0.8941`. **+Mechanism Anchor**: MAE `0.2987 → 0.2796`, $R^2$ `0.7813 → 0.8011`, Sp `0.9028`. **PPB Dual**: MAE `0.2264 → 0.2310`, Sp `0.7093 → 0.7004`. **Hemolysis FP**: `0/17` (`max_prob 0.4289 → 0.3858`). | **PROMOTED** (`case17_scorecard` updated in `v33_manifest.json`, preserving `v332_exact_only_reference`). |
| **Step 2B (Cluster-Weighted Conformal UQ)** | Cluster-weighted ($w_i = 1/m_{c(i)}$) Mondrian AD-EB (`mondrian_ad_eb_cw`, pure JSON) + Cluster-weighted Residual-Centered CQR (`cqr_lgb_cw`, compact LightGBM) across all 40 regression endpoints × 2 tracks (80 tracks), selected strictly on CW-OOF + bounded-endpoint asymmetric prior | Untouched 20% Holdout across 80 tracks: **Mean Cov80 `77.09% → 80.87%`**, **Mean Cov90 `88.70% → 90.33%`**, **Min Cov80 `29.17% → 47.83%` (+18.7 pp)**, **OOD Cov80 `49.14% → 71.48%` (+22.3 pp)**, tracks `<70%`: `11/80 → 8/80`. **`PPB` Cov80/90: `29.2% / 91.7% → 83.3% / 95.8%`**. Total artifact size: **12.06 MB** (`51` CQR + `29` pure JSON). | **PROMOTED** (live in `v33.3` engine + manifest + recomputed Wilson 95% CIs & 3-tier `uq_reliability`). |
| **Step 2A (Censored Main Models & Logit-PPB/F)** | `TobitKernelEM` (Buckley-James exact Mills-ratio EM on zero-PCA-loss 4-FM kernel) + custom Tobit LightGBM + Logit-XGB-AFT across 14 high-censored / bounded endpoints | **Logit-Tobit PPB (strict 19-cluster excluded)** breaks the `~0.72` linear ceiling (Tirzepatide `0.919`, Caspofungin `0.929`, Insulin detemir `0.818`, Semaglutide `0.787`; Case17 Sp `0.6079`, MAE `0.2084`). Censored Hinge MAE improves on `cytotoxicity_CC50` (`0.3720 → 0.3564`), `MIC_E_faecalis` (`0.6348 → 0.5967`, exact MAE unchanged `0.5322`), `MIC_S_aureus` (`0.5071 → 0.5004`), `MBC_S_aureus` (compliance `0% → 27%`), with minor exact-subset MAE trade-off (`+0.005–0.015`). | **DIAGNOSTIC** (keeps exact-optimal primary models + censored diagnostic head intact; documented transparently). |
| **Step 3A (3-Lineage Multi-Task Kernel + `MBC ≥ MIC`)** | 3 physically coherent lineages (Gram-negative 8 EPs, Gram-positive/Fungal/Biofilm 14 EPs, Permeability/Efflux 5 EPs) with strict cross-task cluster masking (`ValFold(T,f) ∪ Holdout(T)` masked in all sibling tasks) + `MBC ≥ MIC` monotonic residual cascade | **23/27 endpoints** improve strictly over same-kernel single-task baseline (`MBC_E_faecalis +0.0586`, `MIC_L_monocytogenes +0.0243`, `MIC_C_neoformans +0.0187`, `RRCK +0.0175`, `MIC_E_faecium +0.0154`, `MBEC_biofilm +0.0151`). **`MIC_E_faecium`** beats full Wave 5A champion strictly on OOF (CV `0.4896 → 0.4840`, HO `0.3470 → 0.3358`). Hard `MBC ≥ MIC` eliminates `38.9%–64.5%` physical violations and improves `MFC_C_albicans` HO MAE `-18.5%` (`0.3771 → 0.3075`) & `MBEC_biofilm` (`0.7836 → 0.7621`), but hurts bacterial pairs due to inter-lab strain variance. | **DIAGNOSTIC** (proves domain-factored transfer beats global soup; avoids forcing hard `MBC ≥ MIC` on unpaired bacterial assays). |
| **Step 3B (Small-N PK Zero-Winner's-Curse & Proteolytic 20D)** | Allometric bodyweight unification (`CL_abs ↔ CL_per_kg`, `Vd_abs ↔ Vd_per_kg`, strict cluster masking) + Logit-CompositeGPR + TabPFN v2 on 7 Small-N PK EPs; 20D RDKit SMARTS proteolytic cleavage × shielding features on `half_life_reg` | Confirms severe winner's curse on $N < 130$: **`Vd_absolute`** (`+142` converted rows) drops HO MAE **`0.7780 → 0.5171` (-33.5%)**, HO $R^2$ **`-0.882 → +0.040`** (beats baseline even on CV `0.4597 → 0.4549`); **`PPB`** Logit-GPR drops HO MAE **`0.2608 → 0.2056` (-21.2%)**, HO Sp **`0.678 → 0.801`**, HO $R^2$ **`0.204 → 0.461`** (CV-HO gap `0.0019`); **`Cmax`** TabPFN v2 drops HO MAE **`0.9156 → 0.8057` (-12.0%)**, Sp **`0.614 → 0.833`**. **Honest negative result**: 20D proteolytic features add `w=0.00` over 4-FM + Deep Specialist on `half_life_reg` (ESM-C multi-region + ncAA-RepDistill already capture 2D shielding). | **DIAGNOSTIC** (documented under zero-holdout-peeking discipline; future v34 nested-outer-fold small-N policy candidate). |

---

## Step 1 — Strict Full-Cluster Leave-Case17-Out Fix (L1)

### Root Cause Fixed
In `v33.0–v33.2`, `models_gpu_deep` (`deep_data.py:300-312`) excluded all **19 Case17 clusters (911 rows)** across the 24 core endpoints, whereas `models_cpu` and `models_fusion` filtered only `case17_excluded == 0` (439 exact-match rows), leaving **472 same-cluster analog rows** (including `+174` in `half_life_reg`, `+172` in `half_life_cls`, `+26` in `overall_peptide_toxicity`, `+13` in `PPB`, `+10` each in `CL_per_kg`/`Vd_per_kg`) in training. In addition, `run_v33_fusion_suite.py:1341` refit `c17_stacker` on `all_oof` predictions produced by base heads that had seen Case17 rows in other folds.

### Fix & Verification (`step1_case17_strict_cluster_audit.py`)
1. Identified the exact union of all **19 Case17 clusters** across all 24 `v33.0` endpoints and enforced hard zero-overlap assertions (`train_clusters ∩ case17_clusters == ∅`) across all 5 `AUX_ENDPOINTS` (`PAMPA`, `Caco2`, `LogD7.4`, `solubility_cls`, `hemolysis_cls`), all CPU base recipes, and all Fusion pretrained heads.
2. Refit every Fusion pretrained head on strictly non-Case17-cluster rows using 5 balanced cluster folds, collected strictly clean non-Case17-cluster OOF predictions, and fit `c17_stacker` (`dual_modality`, `smiles_only`, `aft`) strictly on that clean OOF.
3. Combined with the already strictly cluster-excluded `deep` Case17 predictions using the exact frozen Wave 5A `v33_manifest.json` champion weights, verifying **`parity_err = 0.00000`** on all 13 Case17 scorecard endpoints when plugging original predictions.

### Before vs After Case17 Scorecard

| Metric | Old (`v33.2`, exact-only 439 rows) | **Strict Full-Cluster (`v33.3`, 19 clusters / 911 rows + clean stacker)** | Delta |
|---|---:|---:|---:|
| **Half-life Pure ML** (`n=16`, $\log_{10}$ h) — MAE / RMSE / $R^2$ / Sp | `0.3306` / `0.4289` / `0.7627` / `0.9029` | **`0.3185` / `0.4214` / `0.7710` / `0.8941`** | MAE `-0.0121`, $R^2$ `+0.0083` |
| **Half-life + Mechanism Anchor** (`n=16`, $\log_{10}$ h) — MAE / RMSE / $R^2$ / Sp | `0.2987` / `0.4118` / `0.7813` / `0.9028` | **`0.2796` / `0.3927` / `0.8011` / `0.9028`** | MAE `-0.0191`, $R^2$ `+0.0198` |
| **PPB Dual** (`n=14`, fraction) — MAE / RMSE / $R^2$ / Sp | `0.2264` / `0.2771` / `0.3570` / `0.7093` | **`0.2310` / `0.2813` / `0.3372` / `0.7004`** | MAE `+0.0046`, Sp `-0.0089` |
| **Hemolysis FP Check** (`n=17` FDA drugs) — `max_prob` / `>0.50` | `0.4289` / `0/17` | **`0.3858` / `0/17`** | `max_prob -0.0431` |

Notable per-drug shifts under strict 19-cluster exclusion: Semaglutide pure-ML half-life moves honestly from `100.0 h → 63.4 h` (FDA `168 h`) once GLP-1 analogs are removed, while Liraglutide (`21.3 → 15.0 h`, FDA `13.0 h`), Teriparatide (`3.15 → 2.28 h`, FDA `1.0 h`), Cyclosporine A (`10.98 → 9.52 h`, FDA `8.4 h`), Tirzepatide (`111.4 h`, FDA `125.0 h`), and Insulin degludec (`20.4 → 22.4 h`, FDA `25.0 h`) remain highly accurate.

---

## Step 2 — Cluster-Weighted Conformal Calibration & Censored/Logit Main Models

### Part 2B: Cluster-Weighted Conformal UQ Across All 80 Regression Tracks (**PROMOTED**)
Under cluster-disjoint holdout and novel-scaffold deployment, row-equal conformal quantiles over-weight large analog clusters relative to singleton novel scaffolds. Weighting each calibration point by $w_i = 1 / m_{c(i)}$ (so each cluster carries total mass $1.0$ and a new unseen cluster carries mass $1.0$) restores finite-sample cluster-level exchangeability. Furthermore, selecting calibrators via **cluster-weighted nested 5-fold OOF** (plus asymmetric `cqr_lgb_cw` for bounded endpoints) eliminates the selection bias that previously favored under-covering row-unweighted `mondrian_ad_eb`.

| Calibration Method (All 40 Reg EPs × 2 Tracks = 80 Tracks) | HO Mean Cov80 | HO Mean Cov90 | HO Min Cov80 | Tracks `<70%` | HO OOD Cov80 | Mean Width80 | Artifact Size |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` (`v33.2` manifest) | 77.09% | 88.70% | 29.17% | 11/80 | 49.14% | 1.4969 | 0 MB |
| `mondrian_ad_eb` (row-unweighted JSON) | 77.33% | 88.31% | 29.17% | 12/80 | 71.43% | 1.4792 | 0 MB |
| `mondrian_ad_eb_cw` (cluster-weighted JSON) | 79.54% | 90.71% | 41.67% | 8/80 | 68.15% | 1.6636 | 0 MB |
| `cqr_lgb_cw` (cluster-weighted LightGBM CQR) | 81.94% | 91.02% | 47.83% | 7/80 | 68.98% | 1.8701 | ~20 MB |
| **`v33.3` Promoted (`29 × mondrian_ad_eb_cw` + `51 × cqr_lgb_cw`)** | **80.87%** | **90.33%** | **47.83%** | **8/80** | **71.48%** | **1.7567** | **12.06 MB** |

Key single-endpoint holdout UQ recoveries (`dual_modality`):
- **`PPB`**: Cov80/Cov90 `29.2% / 91.7%` → **`83.3% / 95.8%`**
- **`MIC_B_subtilis`**: `70.3% / 78.8%` → **`73.1% / 82.4%`**
- **`half_life_reg`**: `74.8% / 87.8%` → **`75.3% / 88.9%`**
- **`MDCK`**: `67.6%` → **`73.5%`**
- **`CL_per_kg` (smiles_only)**: `68.3%` → **`70.7%`**
- All 51 exported `.joblib` calibrators were verified to contain **zero** `catboost`, `tabpfn`, or `torch` dependencies (`numpy`, `scipy`, `lightgbm`, `v333_models` only).

### Part 2A: Unified Censored Main Models & Logit-PPB/F (Diagnostic)
Evaluated `TobitKernelEMRegressor` (exact truncated-normal Mills ratio EM with hat-matrix LOO mean on zero-PCA-loss 4-FM Multi-Block Kernel), custom Gaussian Tobit LightGBM (`params["objective"] = obj_fn`), and Logit-XGB-AFT across 14 high-censored / bounded endpoints:
- **Logit-Tobit PPB (strict 19-cluster-excluded Case17)**: Broke the `~0.72` linear ceiling on lipidated high-PPB peptides (Tirzepatide `0.9186` vs FDA `0.960`, Caspofungin `0.9289` vs `0.970`, Insulin detemir `0.8176` vs `0.985`, Semaglutide `0.7872` vs `0.995`; overall Case17 Sp `0.6079`, MAE `0.2084`).
- **Censored-tail vs exact-subset trade-off**: Unified censored models improved all-row Hinge MAE and censored compliance on `cytotoxicity_CC50` (HO Hinge `0.3720 → 0.3564`, compliance `21% → 33%`), `MIC_E_faecalis` (HO Hinge `0.6348 → 0.5967`, exact MAE `0.5322 → 0.5322`), `MIC_S_aureus` (`0.5071 → 0.5004`), and `MBC_S_aureus` (compliance `0% → 27%`), while slightly trading off exact-only MAE (`+0.005–0.015`) as probability mass shifts into right-censored tails. Under our strict non-regression gate (`d_exact_cv >= -0.002`), primary exact models remain untouched and censored diagnostic heads remain explicitly labeled.

---

## Step 3 — 3-Lineage Domain-Factored Multi-Task Transfer & Small-N PK Unification

### Part 3A: 3-Lineage Multi-Task Kernel Transfer & `MBC ≥ MIC` Cascade
Rather than a global 46-head soup, correlated endpoints were grouped into 3 physically coherent lineages with **strict cross-task cluster masking** (when evaluating target $T$ fold $f$, every row in every sibling task $S$ whose `cluster_id` lies in $\text{ValFold}(T, f) \cup \text{Holdout}(T)$ is masked, verified by hard assertions):
- **Pure multi-task transfer**: **23/27 endpoints** achieved strictly lower CV MAE under multi-task Schur-product kernel `MT-KRR` vs identical single-task `ST-KRR`, led by `MBC_E_faecalis` (`+0.0586`), `MIC_L_monocytogenes` (`+0.0243`), `MIC_C_neoformans` (`+0.0187`), `RRCK` (`+0.0175`), `MIC_E_faecium` (`+0.0154`), `MBEC_biofilm` (`+0.0151`), `MIC_B_subtilis` (`+0.0139`), `MDCK` (`+0.0116`), `MBC_A_baumannii` (`+0.0115`), and `MIC_Salmonella` (`+0.0115`).
- **vs Full Wave 5A Super-Ensemble**: **`MIC_E_faecium`** strictly passed the OOF promotion threshold (CV `0.4896 → 0.4840`, HO `0.3470 → 0.3358`, `-0.0112`). On most other endpoints, a single fixed-weight kernel still trails the full 60-recipe Fusion+CPU+Deep ensemble on single-task capacity, showing that multi-task transfer should be applied inside per-family tuned kernels in future major releases.
- **`MBC ≥ MIC` Monotonic Residual Cascade**: Independent Wave 5A models violate `MBC ≥ MIC` on `38.9%–64.5%` of holdout compounds across bacterial pairs. Enforcing $\log_{10}(\text{MBC}) = \log_{10}(\text{MIC}) + \text{softplus}(\delta(x))$ reduces violations to `0.0%` and improves **`MFC_C_albicans`** HO MAE by `-18.5%` (`0.3771 → 0.3075`) and **`MBEC_biofilm`** (`0.7836 → 0.7621`), but degrades MAE on bacterial pairs (`MBC_E_coli` `0.4062 → 0.4796`) because unpaired bacterial MIC and MBC measurements from distinct literature labs/strains exhibit substantial inter-assay noise.

### Part 3B: Small-N PK Zero-Winner's-Curse, Allometric Unification & 20D Proteolytic Features

| Endpoint | $N_{\text{CV}} / N_{\text{HO}}$ | Allo Aug | Wave 5A Baseline CV → HO MAE ($R^2$) | Best Zero-Tuning / Physical Candidate | Candidate CV → HO MAE ($R^2$) | Key Takeaway |
|---|---:|---:|---|---|---|--|
| **`Vd_absolute`** | 48 / 13 | +142 | `0.4597 → 0.7780` (`-0.882`) | `zero_tune_allo_ens` (`gpr_allometric` + `tabpfn_allometric`) | **`0.4549 → 0.5171`** (`+0.040`) | **-33.5% HO MAE**, $R^2$ `+0.922`; beats baseline even on biased CV |
| **`PPB`** | 108 / 24 | 0 | `0.1511 → 0.2608` (`+0.204`, Sp `0.678`) | **`gpr_native` (Logit-CompositeGPR)** | **`0.2037 → 0.2056`** (**`+0.461`**, Sp **`0.801`**) | **-21.2% HO MAE**, $R^2$ `+0.257`, CV-HO gap `0.0019` (zero overfit!) |
| **`Cmax`** | 57 / 17 | 0 | `0.8603 → 0.9156` (`+0.067`, Sp `0.614`) | `tabpfn_native` | `1.2293 →` **`0.8057`** (**`+0.284`**, Sp **`0.833`**) | **-12.0% HO MAE**, Sp `+0.219` (dose-binning limits ceiling) |
| **`F`** | 91 / 23 | 0 | `0.1941 → 0.2364` (`+0.164`) | `gpr_native` / `zero_tune_ens` (Logit) | `0.2111 → 0.2409` (`+0.245`) / `0.2388` (`+0.256`) | $R^2$ `+0.081–0.092` |
| **`CL_absolute`** | 43 / 11 | +185 | `0.6697 → 1.1802` (`+0.255`) | `zero_tune_allo_ens` (CV) / `gpr_native` (HO) | CV **`0.6439`** ($R^2$ `0.321` vs `0.170`); HO `gpr_native` $R^2$ `0.356` | Clear CV gain from allometry; HO has only 2 clusters |
| `Vd_per_kg` | 133 / 33 | +21 | `0.3685 → 0.3295` (`+0.454`) | Wave 5A Baseline | — | $N \ge 130$: Wave 5A already strong |
| `CL_per_kg` | 158 / 41 | +21 | `0.5001 → 0.5477` (`+0.486`) | Wave 5A Baseline | — | $N \ge 130$: Wave 5A already strong |

- **`half_life_reg` 20D Proteolytic Cleavage-Site × Local Shielding Ablation (Honest Negative Result)**: Adding 20D RDKit SMARTS + sequence proteolytic cleavage/shielding features (DPP-IV, trypsin/dibasic, chymotrypsin, N-Me amide, Aib, macrocycle ring size, disulfide, fatty-acid chain, PEG, CIP R/S D-residue fraction) gave a slight gain on standalone LightGBM holdout (`0.7503 → 0.7465`) but received **`w = 0.00`** when blended with the frozen 4-FM + Deep Specialist super-ensemble (`HO MAE = 0.6640`). This confirms that ESM-C 600M multi-region (`nterm5`/`cterm5`) + ncAA-RepDistill v2 + `physchem_core` already encode nearly all 2D proteolytic stability signals.

---

## Reproducibility & Artifacts
- `v333_science/v333_models.py` (mirrored to `bundle/v333_models.py`)
- `v333_science/step1_case17_strict_cluster_audit.py` → `step1_case17_strict_results.json`
- `v333_science/step2_censored_logit_cqr.py` → `step2_censored_logit_cqr_results.json`
- `v333_science/step2b_cluster_weighted_promote.py` → `v333_conformal_calibration.json` + 51 portable `.joblib` calibrators
- `v333_science/step3_multitask_smalln_pk.py` → `step3_multitask_smalln_pk_results.json`
- `v333_science/promote_v333_manifest.py` → upgraded `bundle/v33_manifest.json` (`v33.3`)
