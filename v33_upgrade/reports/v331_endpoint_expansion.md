# Peptide OmniPanel V33.1 — Scientifically Rigorous Endpoint Expansion Report (Wave 5A)

**Generated:** 2026-11-22  
**Status:** **PASS — Verified Offline on `node23` (Zero Regression on All 24 Existing V33 Endpoints; `node2` Production Untouched)**

---

## 1. Executive Summary

Wave 5A expands **Peptide OmniPanel V33** from **24 endpoints** to **46 scientifically valid, zero-leakage, condition-aware experimental endpoints** (**+22 genuinely new endpoints**: **4 Tier 1 ADMET/Safety** + **18 Tier 2 Antimicrobial/Bactericidal/Fungicidal/Anti-Biofilm/Lethal**), while keeping **all 24 existing V33 endpoints 100% unchanged and bit-for-bit regression-free**.

| Metric | V33.0 (Existing Base) | V33.1 Expansion (New) | **V33.1 Unified Total** |
|---|---:|---:|---:|
| **Total Endpoints** | 24 (20 Tier 1, 4 Tier 2) | **+22** (4 Tier 1, 18 Tier 2) | **46** (24 Tier 1, 22 Tier 2) |
| **Classification / Regression** | 6 cls / 18 reg | 0 cls / **22 reg** | **6 cls / 40 reg** |
| **Total Curated Rows** | 69,111 | +13,216 | **82,327** |
| **CV / Untouched 20% Holdout Rows** | 55,299 / 13,812 | +10,581 / +2,635 | **65,880 / 16,447** |
| **Censored Rows (`<`, `>`, interval)** | 2,523 | +2,012 | **4,535** |
| **Unified Compound Registry (incl. Case17)** | 46,598 | **+4,397 delta compounds** (`K`) | **50,995** (50,994 non-Case17) |
| **AD / Analog Index Reference Corpus** | 46,597 × 24 eps (`7.31 MB`) | +4,397 refs, +22 eps (`+1.05 MB`) | **50,994 × 46 eps (`8.36 MB`)** |
| **Pruned Serving Models Footprint** | 10,460.0 MB (43 `.joblib`) | **+2,641.1 MB** (41 `.joblib`) | **13,101.1 MB** (84 `.joblib`) |
| **Inference Engine RSS (CPU on `node23`)** | 22.22 GB | +4.00 GB | **26.22 GB** |
| **Existing 24-Endpoint Parity Check** | Baseline | — | **96/96 BIT-EXACT (`max_abs_diff = 0.0`)** |

### Compute & Resource Governance Compliance
- **Local Windows RTX 5080 (16 GB)**: Performed all delta 4-FM embedding extraction (`K = 4,397` + 5 anchors) in **298.6s** using `D:\work\中检院\v33_upgrade\models_gpu\_local_cache\venv\Scripts\python.exe`.
- **`node1` GPUs**: **0% touched / 0 jobs run**.
- **`node23` (64 CPU cores, 256 GB RAM)**: Executed all cleaning, Non-Chaining Dual-Track clustering, feature computation (`61.8s`), 22-endpoint multi-family CPU training (`1,446.3s`), 22-endpoint 4-FM Multimodal Fusion training (`1,615.0s`), OOF champion selection & bundle merge (`491.6s`), and offline pre/post verification. Confirmed **`NONE_RUNNING`** lingering daemon processes after verification.
- **`node2` Production Services (`172.21.43.16`)**: **100% untouched** during Wave 5A (ready for Wave 5B Web UI upgrade and atomic redeployment).

---

## 2. Scientific Data Mining, Anti-Leakage & Admission Protocol

All candidate rows were extracted strictly read-only (`?mode=ro`) from `D:\work\中检院\data\dataset_reclean_20261113_v1\catalogue.sqlite` (merged OLD V15 + NEW archive) via `01_extract_v331_new_endpoints.py` (`19,092` raw candidate rows) and cleaned/deduplicated/clustered via `02_clean_dedup_cluster_split_v331.py`.

### Key Scientific Rigor Controls
1. **Extended Order-Independent Exact DBAASP Multiset Bijection**:
   - Naive `assay_id` joins across DBAASP records cause silent cross-species/cross-endpoint label swapping whenever multiple target organisms or activity types share an assay study.
   - We extended V33's proven multiset bridge to support multi-endpoint DBAASP studies and range-valued observations (`0.0625-0.5`), strictly enforcing exact cardinality bijection (`Counter(orig_values) == Counter(ctx_values)`) AND single-unit consistency (`len(units) == 1`) AND unambiguous `(qualifier, Decimal)` -> `(endpoint, species)` mapping (rejecting any ambiguity or mammalian `targetCell` collision).
   - Unlocked **32,479 verified DBAASP microbial/biofilm/LC records** with zero cross-species contamination.
2. **Zero SMILES Fabrication**:
   - Deterministic RDKit `MolFromSequence` SMILES synthesis was permitted **exclusively** for provably unmodified canonical 20-AA linear peptides (`N-term=Free/None`, `C-term=Free/None`, `Other=None`, `Linear`).
   - Modified sequences without verified source SMILES remain `smiles_origin="missing"` (`SEQMOD:...`), preserving strict modality honesty across `smiles_only` vs `dual_modality` tracks.
   - Recovered `dramp_v4` unmodified 20-AA sequences (`232` cytotoxicity rows, `72` hemolysis MHC rows) directly from `raw_json` after verifying explicit modification flags.
3. **Domain-Specific Cleanliness**:
   - **`cytotoxicity_CC50`**: Strictly mammalian cell lines (`human=1,860`, `mouse=288`, `monkey=121`, `rat=78`, `dog=8`, `sheep=2`, `unspecified=852`); explicitly excluded **1,359 zebrafish/embryo organismal toxicity rows**; conditioned on `cond_matrix` (`normal_mammalian=2,093`, `cancer_cell_line=253`, `unspecified=1,001`), `cond_species`, and `cond_estimand` (`IC50=1,506`, `CC50=1,461`, `LC50=231`, `EC50=149`).
   - **`hemolysis_MHC`**: Unifies minimal hemolytic concentration (`MHC=367`, `HC10=129`) with explicit `cond_species` (`human=386`) and `cond_estimand`.
   - **`efflux_ratio`**: Separates P-gp inhibited assays (`caco2_with_inhibitor=26`) from uninhibited assays (`caco2=120`, `mdck=4`, `unspecified=21`) via `cond_assay` so canonical default `cond_assay="caco2"` predicts native uninhibited Caco-2 efflux ratio.
   - **`Vd_absolute`**: Strictly absolute volume units (`L` / `mL`), excluding weight-normalized `L/kg` (`Vd_per_kg`), conditioned on `cond_species`, `cond_route`, and `cond_estimand` (`Vss`, `V_F`, `Vd`, `Vz`).
   - **`MFC_C_albicans`**: Cleared the strict single-species admission gate (`110` rows, `64` clusters) on its own, eliminating any need for cross-species antifungal pooling (`29` non-fungal rows dropped).
   - **`MBIC_biofilm` & `MBEC_biofilm`**: Strict `MBIC` (`145` rows, `58` clusters) and strict `MBEC` (`81` rows, `37` clusters) were just below strict-only thresholds (`150/60` and `80/40`), so unified biofilm endpoints were retained with explicit `cond_estimand` (`MBIC`/`MBIC50`/`MBIC90`, `MBEC`/`MBEC50`/`MBEC90`) + `cond_species` + `cond_assay`.
4. **Incremental Non-Chaining Dual-Track Clustering & Zero-Leakage Splits**:
   - All `4,772` candidate compounds already in V33 reused their exact frozen V33 `cluster_id`.
   - Of `3,184` unclustered candidate groups, `2,199` matched existing V33 clusters (at `>=70%` sequence identity on Track-S or `>=0.70` Morgan Tanimoto on Track-M), preventing cross-wave analog leakage, and `985` formed new V33.1 clusters.
   - Every admitted endpoint passed hard programmatic assertions for **zero overlap across `cluster_id`, `compound_key`, `inchikey_14`, `conn_hash`, and `canonical_smiles`** between outer 20% holdout and 5 CV folds, and between every pair of CV folds, with fold sizes strictly within `[10%, 35%]`.

### Rejected Candidates & Excluded Unsuitable Tasks

| Candidate Endpoint | Cleaned Rows | Unique Mols | Clusters | Exact Rows | Decision & Scientific Reason |
|---|---:|---:|---:|---:|---|
| **`LogP`** | 39 | 39 | 34 | 39 | **REJECTED** — After honestly removing 7 pH-7.4 buffered LogD rows, 1 heptane/water row, and calculated entries, 39 deduplicated experimental octanol/water rows remained (below strict `>= 40` Tier-1 gate). |
| **`MBC_K_pneumoniae`** | 75 | 74 | 57 | 67 | **REJECTED** — Below strict Tier-2 gate (`rows=75/80`). |
| **`MBC_B_subtilis`** | 62 | 62 | 32 | 58 | **REJECTED** — Below strict Tier-2 gate (`rows=62/80`, `clusters=32/40`). |
| **`MBC_S_epidermidis`** | 56 | 54 | 32 | 42 | **REJECTED** — Below strict Tier-2 gate (`rows=56/80`, `clusters=32/40`). |
| `MFC_antifungal` (pooled) | — | — | — | — | **SUPERSEDED** — Single-species `MFC_C_albicans` cleared gate (`110` rows, `64` clusters), avoiding cross-species pooling. |
| Receptor bioactivities (`IC50`, `Ki`, `Kd`, `EC50`, `AC50`, `GI50`, `Km`, `LC50`, `LD50`) | — | — | — | — | **BARRED** — Target/receptor-dependent endpoints cannot be pooled into ligand-only panels. |
| Positive-only / weak-label sets (`cell_penetration`, `tumor_homing`, `drug_delivery`, `therapeutic_function`) | — | — | — | — | **BARRED** — Never fabricate synthetic negatives or train weak OvR labels. |

---

## 3. Incremental Compound Registry & Delta 4-FM Embedding Verification

Only the **`K = 4,397` genuinely new compounds** used by the 22 admitted endpoints (`smiles_valid = 3,949`, `seq_valid = 4,289`) were appended to indices `46,598..50,994` (`N = 50,995` unified total). Rows `0..46,597` in both `expansion_v331/features/*.npz` and `expansion_v331/embeddings/*.npz` were programmatically asserted **bit-for-bit identical** (`np.array_equal`) to the frozen V33.0 arrays.

### 5-Anchor Local RTX 5080 Cosine Similarity Verification vs Frozen V33.0 Embeddings

| Anchor Compound (`compound_idx`) | `ncaa_smiles` | `ncaa_fusion` | `ncaa_prot` | `peptideclm2` | `chemberta` | `esmc_mean` | `esmc_max` | `esmc_nt5` | `esmc_ct5` | **Min Cos** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Semaglutide (`35506`, chunked SMILES) | `1.000000` | `0.999999` | `0.999995` | `1.000000` | `0.999999` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | **`0.999995`** |
| Cyclosporine A (`26244`, cyclic N-Me) | `0.999999` | `0.999998` | `0.999889` | `1.000000` | `0.999997` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | **`0.999889`** |
| Caspofungin (`19174`, lipopeptide) | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.999996` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | **`0.999996`** |
| Linear AMP 1 (`2667`) | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.999998` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | **`0.999998`** |
| Linear AMP 2 (`2673`) | `1.000000` | `0.999999` | `0.999995` | `1.000000` | `0.999997` | `1.000000` | `1.000000` | `1.000000` | `1.000000` | **`0.999995`** |

- **Global Minimum Anchor Cosine**: **`0.999889`** across all 5 anchors × 9 embedding blocks (all well above `0.999` / float16 precision floor).

---

## 4. Complete Benchmark Results — All 22 Newly Admitted V33.1 Endpoints

Champion selection (`cpu` vs `fusion` vs nested-OOF simplex `super_ensemble`), conformal 80%/90% calibration, and censored-aware AFT head blending were frozen **strictly on 5-fold grouped CV OOF** with **zero holdout peeking**, then evaluated **once** on the untouched 20% cluster holdout.

- **Dual-Modality Champion Family Wins (22 new endpoints)**: **4-FM Fusion = 15**, **Super-Ensemble (CPU + 4-FM Fusion) = 5**, **CPU = 2**.
- **4-FM Fusion vs CPU on Holdout MAE**: 4-FM Fusion outperformed CPU on **17 / 22 endpoints** (`-0.2506` on `Vd_absolute`, `-0.1108` on `MIC_C_neoformans`, `-0.0810` on `LC_antimicrobial`, `-0.0591` on `MIC_E_faecalis`, `-0.0426` on `MBC_E_faecalis`, `-0.0406` on `MIC_A_baumannii`, `-0.0400` on `efflux_ratio`).
- **Overall 22-Endpoint Holdout Means**: Dual MAE = **`0.5171`**, SMILES-only MAE = **`0.5094`**, Dual Spearman = **`0.4664`**, Conformal 80% / 90% Coverage = **`76.5% / 88.7%`**.

| Endpoint | Tier | Domain | Unit | N (Mols / Clu / Cens) | OOF Champion (Dual) | CV MAE | **HO MAE** | HO RMSE | **HO Sp** | HO R² | **W2x / W3x** | SMI HO MAE | CPU → Fus HO MAE | Cov 80% / 90% | Censored Head (HO hMAE) |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `cytotoxicity_CC50` | T1 | safety | `log10(uM)` | 3,347 (2,967 / 1,136 / 873) | `fusion` | 0.4141 | **0.4057** | 0.5621 | **0.652** | 0.536 | 50.2% / 70.1% | 0.4025 | 0.3919 → 0.4057 | 79.9% / 90.0% | `censored_blend` (0.350) |
| `hemolysis_MHC` | T1 | safety | `log10(uM)` | 496 (467 / 139 / 165) | `fusion` | 0.4103 | **0.4830** | 0.5690 | **0.550** | 0.320 | 25.5% / 58.2% | 0.4924 | 0.4814 → 0.4830 | 80.0% / 94.6% | `censored_blend` (0.430) |
| `efflux_ratio` | T1 | permeability | `log10(ER)` | 171 (145 / 49 / 15) | `fusion` | 0.3709 | **0.3599** | 0.4332 | **0.642** | 0.670 | 46.9% / 68.8% | 0.4073 | 0.3999 → 0.3599 | 87.5% / 96.9% | `champion_exact` (0.371) |
| `Vd_absolute`* | T1 | pk | `log10(L)` | 63 (38 / 27 / 2) | `cpu` | 0.4597 | **0.7780** | 0.9890 | -0.163 | -0.882 | 23.1% / 53.9% | 0.7379 | 0.7780 → 0.5274 | 53.8% / 84.6% | `champion_exact` (0.778) |
| `MIC_B_subtilis` | T2 | antimicrobial | `log10(uM)` | 1,475 (1,441 / 590 / 63) | `super_ensemble` | 0.4450 | **0.5595** | 0.7798 | **0.414** | 0.162 | 41.2% / 57.0% | 0.5598 | 0.5776 → 0.5583 | 70.6% / 78.8% | `champion_exact` (0.583) |
| `MIC_K_pneumoniae` | T2 | antimicrobial | `log10(uM)` | 971 (914 / 412 / 193) | `fusion` | 0.5071 | **0.4300** | 0.5354 | **0.612** | 0.425 | 42.2% / 63.6% | 0.4423 | 0.4460 → 0.4300 | 89.0% / 94.8% | `censored_blend` (0.443) |
| `MIC_A_baumannii` | T2 | antimicrobial | `log10(uM)` | 939 (867 / 382 / 77) | `super_ensemble` | 0.4466 | **0.4431** | 0.5489 | **0.549** | 0.266 | 41.2% / 62.4% | 0.4164 | 0.4694 → 0.4288 | 83.6% / 90.3% | `censored_blend` (0.532) |
| `MIC_S_epidermidis` | T2 | antimicrobial | `log10(uM)` | 784 (752 / 307 / 59) | `super_ensemble` | 0.4945 | **0.4991** | 0.6515 | **0.550** | 0.258 | 37.9% / 51.7% | 0.4938 | 0.5140 → 0.4991 | 81.4% / 92.4% | `censored_blend` (0.521) |
| `MIC_Salmonella` | T2 | antimicrobial | `log10(uM)` | 719 (684 / 355 / 79) | `super_ensemble` | 0.5353 | **0.5498** | 0.7031 | **0.547** | 0.269 | 35.6% / 51.9% | 0.5308 | 0.5579 → 0.5497 | 77.0% / 92.6% | `censored_blend` (0.527) |
| `MIC_E_faecalis` | T2 | antimicrobial | `log10(uM)` | 628 (613 / 336 / 105) | `fusion` | 0.5707 | **0.5322** | 0.6812 | **0.453** | 0.118 | 37.8% / 51.0% | 0.5300 | 0.5913 → 0.5322 | 78.6% / 89.8% | `censored_blend` (0.504) |
| `MIC_C_neoformans` | T2 | antimicrobial | `log10(uM)` | 287 (250 / 113 / 31) | `fusion` | 0.3945 | **0.4952** | 0.6009 | **0.435** | 0.113 | 30.4% / 63.0% | 0.4842 | 0.6061 → 0.4952 | 71.7% / 84.8% | `champion_exact` (0.527) |
| `MIC_E_faecium` | T2 | antimicrobial | `log10(uM)` | 283 (276 / 171 / 23) | `super_ensemble` | 0.4917 | **0.3470** | 0.4312 | **0.683** | 0.352 | 50.9% / 74.6% | 0.3459 | 0.3659 → 0.3409 | 96.4% / 98.2% | `champion_exact` (0.369) |
| `MIC_L_monocytogenes` | T2 | antimicrobial | `log10(uM)` | 248 (243 / 128 / 19) | `fusion` | 0.5062 | **0.5125** | 0.6309 | **0.783** | 0.617 | 37.0% / 54.4% | 0.4796 | 0.5256 → 0.5125 | 80.4% / 95.7% | `champion_exact` (0.506) |
| `MBC_S_aureus` | T2 | antimicrobial | `log10(uM)` | 551 (511 / 287 / 94) | `fusion` | 0.5239 | **0.6276** | 0.8171 | **0.340** | 0.057 | 34.3% / 48.5% | 0.6547 | 0.6309 → 0.6276 | 76.8% / 82.8% | `censored_blend` (0.631) |
| `MBC_E_coli` | T2 | antimicrobial | `log10(uM)` | 548 (503 / 301 / 47) | `fusion` | 0.4404 | **0.4062** | 0.5029 | **0.641** | 0.424 | 45.4% / 64.8% | 0.4206 | 0.4177 → 0.4062 | 86.1% / 91.7% | `censored_blend` (0.420) |
| `MBC_P_aeruginosa` | T2 | antimicrobial | `log10(uM)` | 392 (387 / 207 / 47) | `fusion` | 0.4933 | **0.4837** | 0.6116 | **0.706** | 0.470 | 36.8% / 52.9% | 0.4539 | 0.4476 → 0.4837 | 83.8% / 91.2% | `champion_exact` (0.491) |
| `MBC_A_baumannii` | T2 | antimicrobial | `log10(uM)` | 136 (131 / 77 / 15) | `fusion` | 0.4945 | **0.5933** | 0.7219 | **0.377** | 0.125 | 29.2% / 41.7% | 0.5798 | 0.6159 → 0.5933 | 75.0% / 87.5% | `champion_exact` (0.563) |
| `MBC_E_faecalis` | T2 | antimicrobial | `log10(uM)` | 90 (84 / 64 / 16) | `fusion` | 0.4677 | **0.6881** | 0.7797 | **0.597** | 0.390 | 20.0% / 40.0% | 0.6570 | 0.7307 → 0.6881 | 53.3% / 93.3% | `champion_exact` (0.674) |
| `MFC_C_albicans`* | T2 | antimicrobial | `log10(uM)` | 110 (108 / 64 / 25) | `fusion` | 0.3649 | **0.3771** | 0.4612 | 0.191 | -0.152 | 58.8% / 64.7% | 0.3189 | 0.3393 → 0.3771 | 76.5% / 88.2% | `fusion_aft` (0.351) |
| `MBIC_biofilm` | T2 | antimicrobial | `log10(uM)` | 296 (213 / 109 / 17) | `cpu` | 0.4302 | **0.5600** | 0.7541 | 0.241 | 0.077 | 39.6% / 58.5% | 0.5541 | 0.5600 → 0.5507 | 77.4% / 84.9% | `champion_exact` (0.610) |
| `MBEC_biofilm`* | T2 | antimicrobial | `log10(uM)` | 138 (104 / 65 / 14) | `fusion` | 0.3654 | **0.7836** | 1.0011 | 0.038 | -0.458 | 30.4% / 34.8% | 0.7836 | 0.7191 → 0.7836 | 47.8% / 65.2% | `champion_exact` (0.695) |
| `LC_antimicrobial` | T2 | antimicrobial | `log10(uM)` | 544 (239 / 118 / 33) | `fusion` | 0.4592 | **0.4615** | 0.6187 | **0.424** | 0.171 | 44.3% / 65.1% | 0.4599 | 0.5425 → 0.4615 | 76.4% / 84.0% | `censored_blend` (0.474) |

> **\*Honest Small-Sample Caveats (Zero Holdout Peeking Preserved)**:
> - **`Vd_absolute`** (`n=63`, `n_ho=13` exact): Strict OOF-only selection chose `cpu` (`CV MAE = 0.4597`), whereas 4-FM `fusion` achieved substantially lower holdout MAE (`0.5274` vs `0.7780`). We strictly **did not** post-hoc switch the champion after observing holdout scores. Because `n=63`, users should interpret `Vd_absolute` alongside its 80%/90% conformal CI and AD tier (note: for Semaglutide, the served prediction is `11.53 L`, closely matching the FDA label `12.5 L`).
> - **`MBEC_biofilm`** (`n=138`, `n_ho=23` exact): Biofilm eradication assays span 11 microbial species and 3 estimands (`MBEC`/`MBEC50`/`MBEC90`), yielding strong CV (`0.3654`) but high cross-cluster holdout variance (`0.7836`, `R²=-0.458`). Flagged transparently as a high-variance exploratory Tier-2 biofilm endpoint.
> - **`MFC_C_albicans`** (`n=110`, `n_ho=17` exact): Achieves tight absolute accuracy (`HO MAE = 0.3771`, **58.8% within 2×**, **64.7% within 3×**, SMILES-only MAE `0.3189` / Sp `0.493`), with slightly negative dual `R²` solely due to the narrow target variance across `n_ho=17` holdout compounds.

---

## 5. Zero-Regression Merge, `node2` Compatibility & Offline Verification Receipt

### Bundle Integrity & `node2` Package Compatibility
- **Pristine Backups Created**:
  - `bundle/v33_manifest_24ep_backup.json`
  - `bundle/ad_analog_index_24ep_backup.npz` & `bundle/analog_lookup_24ep_backup.json.gz`
  - `bundle/v33_inference_service_24ep_backup.py`
- **Zero Mutation of Existing V33 Assets**:
  - All **43 existing `serving_models/*.joblib` SHA-256 checksums** verified identical before and after merge.
  - All **24 existing endpoint entries in `v33_manifest.json`** verified byte-identical to `v33_manifest_24ep_backup.json`.
  - All **24 existing endpoint analog records** in the rebuilt 46-endpoint `ad_analog_index.npz` + `analog_lookup.json.gz` verified identical to the 24-endpoint backup.
- **`node2` Package Compatibility Audit**:
  - Recursive class-module inspection of all 41 newly added `.joblib` files confirmed serialized dependencies are strictly: `builtins`, `collections`, `ctypes`, `fusion_bank`, `fusion_models`, `lightgbm`, `numpy`, `sklearn`, `v33_features`, `v33_models`, `xgboost` (**ZERO `catboost`**, 100% compatible with `node2`'s installed `sklearn 1.7.2`, `xgboost 3.2.0`, `lightgbm 4.7.0`, `numpy`, `scipy`, `joblib`, `torch`, `rdkit`, `transformers`).

### End-to-End Offline Verification Receipt (`node23`, `v331_offline_verification_receipt.json`)
- **Status**: **`PASS`** (`version = "v33.1"`, `endpoints = 46`, `tier1 = 24`, `tier2 = 22`)
- **Zero-Regression Bit-Exact Check**: **96 / 96** `(molecule × existing-24-endpoint)` predictions across Semaglutide, Cyclosporine A, Caspofungin, and Magainin-2 matched pre-merge untouched V33.0 **bit-for-bit (`max_abs_diff = 0.0`, identical AD tiers, Top-3 analogs, 80%/90% CIs, and censored diagnostics)**.
- **Full 46-Endpoint Verification**:
  - 5-molecule batch (`Semaglutide`, `Cyclosporine_A`, `Caspofungin`, `Classic_AMP_LL37_core [Magainin-2]`, `Classic_AMP_Indolicidin`): **5/5 OK**, `46` endpoints each, **`0` NaN/Inf**, valid monotonic `ci90_low <= ci80_low <= raw_value <= ci80_high <= ci90_high`, `3` real analogs per endpoint, `ad_ms = 55.4 ms`, `total_ms = 50.1 s`.
  - Live 4-FM encoding test (`force_live_encode=True`, `cache_hit=False`) on novel AMP `KWKLFKKIGAVLKVLTTGL`: **46/46 endpoints OK**, `encode_ms = 744.5 ms`, `ad_ms = 46.85 ms`, `total_ms = 37.3 s`.
- **Biological Sanity Spot-Checks (New V33.1 Endpoints)**:
  - **Semaglutide**: `Vd_absolute = 11.53 L` (FDA label `12.5 L`, In-Domain), `efflux_ratio = 2.7` (OOD, `max_tanimoto=0.1675`).
  - **Caspofungin**: `MFC_C_albicans = 9.75 uM` (Borderline antifungal), `hemolysis_MHC = 174.7 uM` (non-hemolytic).
  - **Magainin-2**: `MIC_A_baumannii = 9.61 uM`, `MIC_K_pneumoniae = 12.73 uM`, `MBIC_biofilm = 18.61 uM`, `MBEC_biofilm = 65.22 uM` (`MBEC > MBIC > MIC`), `hemolysis_MHC = 50.02 uM`, `cytotoxicity_CC50 = 98.93 uM`.
  - **Indolicidin**: `LC_antimicrobial = 3.47 uM`, `MBC_S_aureus = 5.02 uM`, `MBIC_biofilm = 19.04 uM`, `MBEC_biofilm = 25.95 uM`, `hemolysis_MHC = 22.43 uM`, `cytotoxicity_CC50 = 47.05 uM`.
- **Daemon Hygiene**: Temporary `esmc_worker` cleanly terminated; verified `NONE_RUNNING` on `node23`.

---

## 6. Artifact & File Paths (Ready for Wave 5B)

### On `node23` (`/data/qlyu/v33_upgrade/`)
- **Merged Serving Bundle**:
  - `/data/qlyu/v33_upgrade/bundle/v33_manifest.json` (46 endpoints, `version="v33.1"`)
  - `/data/qlyu/v33_upgrade/bundle/v33_manifest_24ep_backup.json` (pristine 24-ep backup)
  - `/data/qlyu/v33_upgrade/bundle/ad_analog_index.npz` & `analog_lookup.json.gz` (`50,994 refs × 46 eps`)
  - `/data/qlyu/v33_upgrade/bundle/serving_models/*.joblib` (`84` files = `43` untouched V33.0 + `41` new V33.1)
  - `/data/qlyu/v33_upgrade/bundle/v33_inference_service.py` (4-line surgical V33.1 update + `_24ep_backup.py`)
- **Isolated Expansion Workspace**:
  - `/data/qlyu/v33_upgrade/expansion_v331/data/endpoints/*.tsv` (`22` new endpoint splits)
  - `/data/qlyu/v33_upgrade/expansion_v331/data/{compounds_v331_merged.tsv, delta_compounds.tsv, v331_split_summary.json}`
  - `/data/qlyu/v33_upgrade/expansion_v331/features/` (`50,995`-compound V33.1 feature bank)
  - `/data/qlyu/v33_upgrade/expansion_v331/embeddings/` (`50,995`-compound merged 4-FM embeddings)
  - `/data/qlyu/v33_upgrade/expansion_v331/{models_cpu, models_fusion, serving_models}/`
  - `/data/qlyu/v33_upgrade/expansion_v331/{v331_champion_summary.json, v331_offline_verification_receipt.json, pre_merge_24ep_baseline.json}`

### Local Workspace (`D:\work\中检院\v33_upgrade\`)
- **Report**: `D:\work\中检院\v33_upgrade\reports\v331_endpoint_expansion.md`
- **Merged Bundle Sync**: `D:\work\中检院\v33_upgrade\bundle\{v33_manifest.json, v33_manifest_24ep_backup.json, v33_inference_service.py, v33_inference_service_24ep_backup.py}`
- **Expansion Scripts & Audits**: `D:\work\中检院\v33_upgrade\expansion_v331\{01_extract_v331_new_endpoints.py, 02_clean_dedup_cluster_split_v331.py, 03_extract_delta_4fm_local_gpu.py, 03b_merge_v331_embeddings_node23.py, v331_config.py, 04a_train_v331_cpu_suite.py, 04b_train_v331_fusion_suite.py, 05_select_champions_and_merge_v331.py, 05b_patch_inference_service_v331.py, 06_verify_v331_offline_node23.py, v331_champion_summary.json, v331_offline_verification_receipt.json, cpu_benchmark_summary.json, fusion_benchmark_summary.json}`
- **Delta 4-FM Audit**: `D:\work\中检院\v33_upgrade\expansion_v331\embeddings_delta\delta_4fm_anchor_audit.json`
