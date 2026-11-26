# Peptide OmniPanel 1.3 — Peptide & Cyclic-Peptide ADMET, PK & Antimicrobial Platform

> **Live Platform (Public HTTPS, CPU-Only Self-Contained Production)**: <https://lmmd.ecust.edu.cn/peptide-omnipanel/> (OpenAPI `/docs`)  
> **Code License**: MIT (`LICENSE`) · **Trained Weights & Hosted Web Service**: Strictly Non-Commercial Academic & Preclinical Research Only (`internal_research_only`; see `DATA_LICENSES.md` & `NOTICE`).  
> **Data Redistribution Policy**: This public repository distributes **zero row-level third-party observation tables**. Only original source code, audited evaluation manifests, aggregate corpus statistics (`data/peptide_ml_cleaning_v1_statistics_v1/`), and factual NLM DailyMed / US FDA regulatory PK pointers are hosted here.

---

## Peptide OmniPanel 1.3 / Engine v33.6 Release (2026-11-26)

**Peptide OmniPanel 1.3 (`v33.6`)** spans **47 harmonized experimental endpoints** (**25 Tier-1** Core ADMET/PK/Safety/Lipophilicity including macrocycle `LogD_decadiene` + **22 Tier-2** ESKAPE/Clinical MIC, MBC, MFC, Biofilm MBIC/MBEC & LC) across **51,007 unique peptides** (`82,414` curated observations; `4,535` censored/interval bounds):

| Dimension | Audited V33.6 Value (from `v33_manifest.json`) |
|---|---|
| **Unified Endpoints** | **47** (**25 Tier-1** + **22 Tier-2**; 6 classification + 41 regression) |
| **CI-Aware Honest Evidence Grades** | **20 `quantitative`** (`n_ho ≥ 30`, `n_clusters ≥ 10`, `R² ≥ 0.15`, **`R²_ci95_lo > 0`**, `ρ ≥ 0.45`, `UQ ≠ low`) · **26 `rank_only`** (incl. `PPB`, `CL_per_kg`, `efflux_ratio`, `LogD_decadiene`) · **1 `exploratory_not_recommended`** (`Vd_absolute`, negative holdout correlation/R²) |
| **Multi-FM Complementarity & Significance** | **CPU vs 4-FM Fusion** (41 regression EPs, SD-normalized `MAE/SD(y)`): **Fusion `0.6349` vs CPU `0.6688`** (`32/41` wins, **paired Wilcoxon Holm-adjusted `p = 3.05 × 10⁻⁴`**). **4-FM Single-Model Friedman/Nemenyi** (`n=46` EPs): `χ² = 0.70, p = 0.872` (`CD = 0.692`), confirming complementarity across peptide modalities. |
| **Published Peptide-Specific Tools (Exact Same Frozen Holdouts, Holm-Adjusted)** | **`hemolysis_cls`**: beats **HemoPI2 v1.3 ESM2-t6** (*Protein Sci.* 2025) on all canonical (`AUC 0.8958 vs 0.8305`, `p_Holm=0.021`, `n=3,230`) & strictly zero-overlap novel holdout (`AUC 0.8867 vs 0.8317`, `p_Holm=0.044`, `n=2,334`).<br>**`PAMPA` (`n=1,715`)**: beats **Multi_CycGT SVM-Tanimoto** (*JCIM* 2024; `MAE 0.3738 vs 0.4223`, `p_Holm=0.021`), **CycPeptMP 2D-SVM** (`0.4132`), and **CycPeptMP RF-750** (`0.3926`) (*BiB* 2024), while matching internal Morgan+RDKit2D LGBM (`0.3728`).<br>**`half_life_reg` (`n=567`)**: outperforms **PLifePred-style AAC+DPC+PhysChem SVR** (`MAE 0.6640 vs 0.7618`, raw `p=0.010`, `p_Holm=0.070`). |
| **Production Super-Ensemble 2019–2023 FDA Blind Test (`n=11`)** | Direct Morgan-2048 `Tc < 0.40` + InChIKey14 + 3-mer sequence isolation across all 11 newly approved FDA peptides (incl. Zilucoplan, Vosoritide, Terlipressin): **Blind Half-Life `Sp = 0.8000` (Fisher 95% CI `[0.385, 0.946]`), `MAE = 0.4279 log₁₀ h` vs `0.6833` constant baseline (`-37.4%`)**; **Blind PPB (`n=8`) `Sp = 0.7306`, `MAE = 0.2967`**. |
| **Macrocycle Chameleonicity (`ΔLogD`)** | Within-series **LOCO** dual-phase validation (`n=27` macrocycles, 100% same-lab `Naylor LPE 2018 primary SI` octanol pairs, `5` clusters): **`Sp = 0.8993`** (cluster-bootstrap 95% CI `[0.842, 0.987]`), **`MAE = 0.4202`**, **`R² = 0.8287`**, **`AUROC(Δ ≥ 2.4) = 0.9506`**, 3-class tercile Acc `0.6667`. Strictly gated to cyclic macrocycles (non-macrocycles suppress tier & validation badges). |
| **2D Graph Surrogate Faithfulness (E-GAT)** | Audited across all 8 endpoints via **true graph atom & incident-edge removal** (`n=80` holdout molecules/EP + 1,000× bootstrap 95% CI): `overall_peptide_toxicity` (**Fid `ρ = 0.8695`**, **Top-20%/Rand deletion `4.68× [3.32, 6.50]`** → `faithful_surrogate`); remaining 7 endpoints transparently labeled `qualitative_saliency_heuristic` in API & UI. |
| **Non-Circular Case17 Clinical Benchmark** | **Primary Pure-ML (`n=16`)**: **`MAE = 0.3185 log₁₀ h`**, **`R² = 0.7710`**, **`ρ = 0.8941`**. **True Dynamic LODO SMARTS Overlay (`n=16`, exact production `detect_acylation_anchor`)**: **`MAE = 0.3082`**, **`R² = 0.7863`**, **`ρ = 0.8706`** (zero Fc/albumin fusion anchors; zero `Insulin degludec` self-anchor circularity; zero false positives on cosmetic `Pal-KTTKS`, `C16-KKKK-NH2`, free palmitic acid, or branched diacids). |
| **Factual Regulatory PK Reference Layer** | **59 marketed FDA-approved peptide drugs**, **428 factual numeric PK records**, **53 live-verified NLM DailyMed SPL `setid`s** (`dailymed.nlm.nih.gov`) & FDA Drugs@FDA links (`accessdata.fda.gov`), with zero third-party derivative tables or `ATLPD*` identifiers. |

---

## Quick Local Verification (Offline Manifest Mode)

```bash
V33_OFFLINE_ONLY=1 pytest v33_upgrade/web/tests/test_web_api.py -v
```

*(Large binary model weights `.joblib`/`.pt`/`.safetensors`, precomputed `.npz` embeddings, and upstream raw third-party datasets are intentionally excluded from this public repository; see `DATA_LICENSES.md` and `NOTICE` for upstream portals and terms.)*
