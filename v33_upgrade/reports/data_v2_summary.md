# Peptide OmniPanel V33 — Data Cleaning 2.0 Summary Report

- **Generated**: 2026-11-21 CST
- **Pipeline Scope**: 24 harmonized ADMET / PK / Toxicity / Antimicrobial endpoints (20 Tier-1 ADMET/PK/Tox + 4 Tier-2 Species-Specific MIC), 46,597 unique V33 compounds (+1 Case17-only query index = 46,598 total feature-bank entries), 69,111 clean deduplicated observations.
- **Authoritative Remote Root (`node23`)**: `/data/qlyu/v33_upgrade/`
  - Frozen compound table (read-only): `/data/qlyu/v33_upgrade/data/compounds_v33.tsv` (`SHA-256: e669245fc2a89149b997c4452c62cc2af593abf043414a8b296b831b074a76a9`)
  - Final leak-free, fold-balanced endpoint tables (24 TSVs): `/data/qlyu/v33_upgrade/data/endpoints/*.tsv`
  - Frozen compact feature bank: `/data/qlyu/v33_upgrade/features/`
  - Audit artifacts (mirrored locally in [`D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs)):
    - [`split_summary.json`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/split_summary.json)
    - [`verification_report.json`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/verification_report.json)
    - [`clean_audit.json`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/clean_audit.json)
    - [`feature_manifest.json`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/feature_manifest.json)
    - [`case17_queries.tsv`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/case17_queries.tsv) & [`case17_overlap.tsv`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/case17_overlap.tsv)

---

## 1. Executive Highlights (V32 → V33 Data Cleaning 2.0)

1. **Massive High-Trust Data Recovery Without Dropping Condition Boundaries**:
   - Extracted **101,045** candidate observations from read-only [`catalogue.sqlite`](file:///D:/work/中检院/data/dataset_reclean_20261113_v1/catalogue.sqlite) (1,289,057 registered records) and deduplicated them per `(compound_key, endpoint, condition_signature)` into **69,111** clean modeling rows across **24 unified condition-aware endpoints** (vs. fragmented single-condition micro-tasks in V32).
   - **27,446 rows (39.7%)** were scientifically rescued from `review`, `censored_numeric`, or unlinked/unit-mismatch records using strict provenance rules, while **41,665 rows (60.3%)** come from native strict/binary baselines.
   - Headline gains over V32 (`peptide_modeling_multiseed20h_v1` / followup2 eligible tables):
     - **`hemolysis_HC50`**: `33` (+ `22` HC50 = `55`) → **5,559 rows** (**101×**)
     - **`half_life_cls`**: `147` (max task `67`) → **3,204 rows** (**21.8×**)
     - **`LogD7.4`**: `57` → **678 rows** (**11.9×**)
     - **`solubility_reg`**: `35` → **298 rows** (**8.5×**)
     - **`F` (bioavailability)**: `19` → **124 rows** (**6.5×**)
     - **`hemolysis_cls`**: `3,938` (max task `2,012`) → **20,506 rows** (**5.2×**)
     - **`Vd_per_kg`**: `0` (`Vdss_per_kg` `24–33`, typed follow-up `20`) → **166 rows** (**5.0×**)
     - **`cytotoxicity_cls`**: `0` (V32 binary deferred all-zero; `EC50`/`IC50` `50`) → **2,934 rows** (balanced `966 pos / 1,968 neg`)
     - **`CL_absolute`**: `0` (deferred in V32 main; ~27 raw) → **54 rows**
     - **`Cmax`**: `0` (deferred in V32 main; `36` in conditional follow-up) → **74 rows**
     - **`half_life_reg`**: `990` across 27 fragmented tasks (max task `375`) → **3,078 rows** (**3.1×** total, **8.2×** vs. largest V32 task)
     - **`MDCK`**: `55` → **168 rows** (**3.1×**); **`CL_per_kg`**: `82` → **206 rows** (**2.5×**); **`PPB`**: `71` → **162 rows** (**2.3×**); **`Caco2`**: `1,197` → **1,580 rows** (**1.32×**); **`PAMPA`**: `6,748` → **8,581 rows** (**1.27×**).
     - **4 Species-Specific MICs** (`E. coli` 2,210; `S. aureus` 2,131; `P. aeruginosa` 1,161; `C. albicans` 468): **5,970 rows** via the mathematically proven order-independent DBAASP multiset bridge + ChEMBL organism mapping (replacing 131–156 tiny V32 micro-tasks whose largest single task was only 64 rows).
2. **Native Censored & Interval Target Preservation**:
   - **2,523 non-exact rows** (`2,331` right-censored, `179` left-censored, `13` interval) are preserved with exact `[y_lower, y_upper]` physical bounds instead of discarding inequalities or fabricating point estimates.
3. **Non-Chaining Anti-Leakage Clustering & Balanced Splits (V2 Fix)**:
   - Replaced Wave-1's transitive Murcko/generic-ring + sequence Union-Find (which collapsed unrelated cyclic peptides into a `3,801`-compound mega-cluster and produced degenerate folds like `PAMPA=[1,2,495,2141,4637]` and `LogD7.4=[507,32,1,19,48]`) with **Stage (a) exact `canonical_identity_group`** + **Stage (b) non-chaining Dual-Track leader clustering** (`>=70%` CD-HIT-style sequence identity for linear canonical peptides; Taylor-Butina `Tanimoto >= 0.70` on `morgan_bits_1024.npz` for cyclic/short/modified peptides) + **stratified LPT size-balanced cluster bin-packing**.
   - Global max cluster size dropped **10.2×** (`3,801 → 373`), and **24/24 endpoints** pass strict fold-balance (`10%–35%` of CV rows; 21/24 endpoints within `18.6%–23.0%`!) with **ZERO** leakage (`cluster_id`, `compound_key`, `inchikey_14`, `conn_hash`, `canonical_smiles`).

---

## 2. Comprehensive V32 vs. V33 Dataset Comparison (All 24 Endpoints)

*V32 baseline counts are from [`D:/work/中检院/data/peptide_modeling_multiseed20h_v1/task_manifest.json`](file:///D:/work/中检院/data/peptide_modeling_multiseed20h_v1/task_manifest.json) and `followup2` [`endpoint_admission_summary.tsv`](file:///D:/work/中检院/data/peptide_modeling_multiseed20h_followup2_v1/endpoint_admission_summary.tsv) (with later single-endpoint follow-up cohorts noted in footnotes).*

| Endpoint | Tier | Task Type | V32 Eligible Rows (Max Single Task) | V33 Clean Rows | Gain vs. V32 | Native / Rescued | Unique Compounds | Unique Clusters | SMILES Valid | Censored / Interval | Case17 Overlap |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PAMPA** | Tier 1 | Regression (log10 cm/s) | 6,748 (6,748) | **8,581** | 1.27× | 6,689 / 1,892 | 8,059 | 627 | 100.0% | 9 (1R, 8L) | 2 |
| **Caco2** | Tier 1 | Regression (log10 cm/s) | 1,197 (1,197) | **1,580** | 1.32× | 1,146 / 434 | 1,452 | 274 | 100.0% | 13 (13L) | 3 |
| **MDCK** | Tier 1 | Regression (log10 cm/s) | 55 (55) | **168** | 3.05× | 60 / 108 | 151 | 58 | 100.0% | 16 (2R, 14L) | 5 |
| **RRCK** | Tier 1 | Regression (log10 cm/s) | 135 (135) | **167** | 1.24× | 125 / 42 | 165 | 33 | 100.0% | 0 | 2 |
| **BBB** | Tier 1 | Classification (weak bench) | 844 (844) | **844** | 1.00× | 844 / 0 | 844 | 643 | 100.0% | 0 | 2 |
| **LogD7.4** | Tier 1 | Regression (dimensionless) | 57 (57) | **678** | **11.89×** | 57 / 621 | 677 | 100 | 100.0% | 3 (3R) | 2 |
| **solubility_cls** | Tier 1 | Classification (sol ≥100µM) | 3,381 (1,334) | **3,722** | 1.10× | 3,489 / 233 | 1,567 | 1,304 | 99.65% | 0 | 1 |
| **solubility_reg** | Tier 1 | Regression (log10 µM) | 35 (35) | **298** | **8.51×** | 87 / 211 | 287 | 97 | 100.0% | 26 (21R, 5L) | 3 |
| **PPB** | Tier 1 | Regression (fraction 0–1) | 71 (54) | **162** | 2.28× | 87 / 75 | 108 | 68 | 100.0% | 30 (14R, 3L, 13I) | 25 |
| **half_life_reg** | Tier 1 | Regression (log10 h) | 990 (375) | **3,078** | **3.11×** | 1,891 / 1,187 | 2,020 | 839 | 93.89% | 177 (110R, 67L) | 133 |
| **half_life_cls** | Tier 1 | Classification (t1/2 ≥ 1h) | 147 (67) | **3,204** | **21.80×** | 1,982 / 1,222 | 2,082 | 837 | 92.57% | 0 | 117 |
| **CL_per_kg** | Tier 1 | Regression (log10 mL/min/kg) | 82 (54) | **206** | 2.51× | 95 / 111 | 152 | 85 | 100.0% | 7 (2R, 5L) | 29 |
| **CL_absolute** | Tier 1 | Regression (log10 L/h) | 0* (deferred) | **54** | New | 7 / 47 | 35 | 27 | 100.0% | 0 | 16 |
| **Vd_per_kg** | Tier 1 | Regression (log10 L/kg) | 24–33† (Vdss) | **166** | **5.03×** | 27 / 139 | 109 | 81 | 100.0% | 0 | 37 |
| **F** | Tier 1 | Regression (fraction 0–1) | 19 (19) | **124** | **6.53×** | 56 / 68 | 86 | 58 | 100.0% | 10 (2R, 8L) | 16 |
| **Cmax** | Tier 1 | Regression (log10 µM) | 0‡ (36 follow-up) | **74** | **2.06×** | 74 / 0 | 27 | 21 | 100.0% | 0 | 24 |
| **hemolysis_cls** | Tier 1 | Classification | 3,938 (2,012) | **20,506** | **5.21×** | 8,086 / 12,420 | 18,573 | 6,320 | 88.21% | 0 | 3 |
| **hemolysis_HC50** | Tier 1 | Regression (log10 µM) | 33–55§ (33) | **5,559** | **101.07×** | 105 / 5,454 | 5,215 | 1,337 | 100.0% | 1,375 (1,339R, 36L) | 2 |
| **cytotoxicity_cls** | Tier 1 | Classification | 0¶ (deferred) | **2,934** | New | 609 / 2,325 | 2,658 | 1,286 | 100.0% | 0 | 8 |
| **overall_peptide_toxicity** | Tier 1 | Classification (ToxinPred3) | 11,036 (11,036) | **11,036** | 1.00× | 11,036 / 0 | 11,036 | 7,261 | 100.0% | 0 | 2 |
| **MIC_E_coli** | Tier 2 | Regression (log10 µM) | ‖ (64 max task) | **2,210** | ‖ | 1,954 / 256 | 2,043 | 903 | 100.0% | 256 (249R, 7L) | 1 |
| **MIC_S_aureus** | Tier 2 | Regression (log10 µM) | ‖ (64 max task) | **2,131** | ‖ | 1,787 / 344 | 1,922 | 857 | 100.0% | 344 (342R, 2L) | 4 |
| **MIC_P_aeruginosa** | Tier 2 | Regression (log10 µM) | ‖ (64 max task) | **1,161** | ‖ | 960 / 201 | 1,086 | 490 | 100.0% | 201 (198R, 3L) | 1 |
| **MIC_C_albicans** | Tier 2 | Regression (log10 µM) | ‖ (64 max task) | **468** | ‖ | 412 / 56 | 460 | 250 | 100.0% | 56 (53R, 3L) | 1 |
| **TOTAL (24 Endpoints)** | 20 T1 + 4 T2 | 18 Reg + 6 Cls | ~31,452–31,750 | **69,111** | **~2.18×** | **41,665 / 27,446** | **46,597** | **17,123** | **96.15% rows** | **2,523 (3.65%)** | **439** |

*\*`CL_absolute` had 27 raw records and 0 eligible tasks in V32 main (`multiseed20h_v1`); a later targeted deployment (`cl_absolute_deployment_20261115_v3`) recovered a small cohort. †`Vd_per_kg` was deferred in V32 main while `Vdss_per_kg` had 24 (33 in followup2) and a typed volume follow-up had 20 rows. ‡`Cmax` had 0 eligible in V32 main and 36 rows / 8 groups in `conditional_cmax_followup_v1`. §V32 had `hemolysis_HC50=33` + `HC50=22` (38 in `rbc50_registered_followup`). ¶V32 `cytotoxicity` (891 rows) was all-zero and deferred; followup2 admitted 29 `EC50` + 21 `IC50` regression rows. ‖V32 `MIC` had 2,523–2,748 rows total across 131–156 fragmented strain/medium micro-tasks (largest single task: 64 rows), whereas V33 consolidates 5,970 species-harmonized rows across the 4 major pathogens.*

**Class Balance for the 6 Classification Endpoints**:
- `BBB`: 421 pos (49.9%) / 423 neg (50.1%)
- `cytotoxicity_cls`: 966 pos (32.9%) / 1,968 neg (67.1%)
- `half_life_cls`: 1,639 pos (51.2%) / 1,565 neg (48.8%)
- `hemolysis_cls`: 6,116 pos (29.8%) / 14,390 neg (70.2%)
- `overall_peptide_toxicity`: 5,518 pos (50.0%) / 5,518 neg (50.0%)
- `solubility_cls`: 2,318 pos (62.3%) / 1,404 neg (37.7%)

---

## 3. Rescue Provenance & Filtering Breakdown

### 3.1 Compound Identity & Representation Coverage (`46,597` V33 Compounds)
- **SMILES Origin**:
  - `source` (verified source SMILES, canonicalized by RDKit 2026.09.1): **27,108** (58.2%)
  - `rdkit_from_canonical_seq` (deterministic RDKit SMILES built **strictly** for provably unmodified canonical 20-AA linear peptides): **16,834** (36.1%)
  - `missing` (chemically modified peptides without source SMILES — **NEVER** fabricated into bare SMILES!): **2,655** (5.7%)
  - Overall SMILES coverage: **43,942 / 46,597** V33 compounds (**94.3%**), covering **66,443 / 69,111** endpoint rows (**96.15%**).
- **Sequence Origin (`parent_sequence`)**:
  - `source_bare_20aa`: **19,106** (41.0%)
  - `rdkit_backbone_extracted` (deterministic alpha-AA backbone walker): **15,567** (33.4%)
  - `source_parent_projection`: **10,177** (21.8%)
  - `none` (peptidomimetics / non-peptide macrocycles): **1,747** (3.7%)

### 3.2 Major Data Rescue Mechanisms (27,446 Rescued Deduplicated Rows = 39.7%)
1. **Verified RDKit-MW Mass-to-Molar Conversions (`5,098` rescued observations in Step 02; only 2 dropped out-of-range)**:
   - `hemolysis_cls` (`µg/mL` % hemolysis → `µM` threshold): **3,235**
   - `hemolysis_HC50` (`µg/mL` → `µM` via exact RDKit MW): **1,184** (`DBAASP_v4.0.1` 1,140; `Hemolytik2_2026` 14; `CHEMBL` 3 post-dedup)
   - `cytotoxicity_cls` (`µg/mL` CC50/IC50 & % viability → `µM` thresholds): **610** (460 CC50/IC50 + 150 %)
   - `solubility_reg` (`µg/mL` → `µM` via RDKit MW): **69** (62 post-dedup)
2. **Primary Literature & Curated Database Review Rescues**:
   - `PAMPA` (`+1,892`): `SweMacrocycleDB` (1,122), `Naylor LPE 2018` (522), non-sentinel `CycPeptMPDB` log10 (123), `Helicogenic macrocycles 2026` (28), `ChEMBL` (23), `Papp`/`permeability` mapped (36), `ACS` papers (27), censored (5).
   - `LogD7.4` (`+621`): `Naylor 2018 octanol/PBS pH 7.4` (516), dimensionless pH 7.4 unit rescues (54), verified `ChEMBL` / `Chloroalkene 2026` / `J. Med. Chem.` pH 7.4 rescues (51).
   - `Caco2` (`+434`) & `MDCK` (`+108`) & `RRCK` (`+42`): `SweMacrocycleDB`, `ChEMBL`, `Papp` mapped, non-sentinel `CycPeptMPDB`, `Pegasus MDCK` below-LOQ left-censored (12).
   - `hemolysis_HC50` (`+5,454`) & `hemolysis_cls` (`+12,420`): `DBAASP_v4.0.1`, `hemopi2_2025`, `Hemolytik2_2026` quantitative HC50 and strict unambiguous % hemolysis rules (`≤10% @ ≥50µM → safe`, `≥20% @ ≤50µM → hemolytic`, `HC50 ≤50µM → hemolytic`, `HC50 ≥100µM → safe`).
   - `cytotoxicity_cls` (`+2,325`): unambiguous CC50/IC50 (`≤25µM toxic`, `≥100µM safe`) and % viability cutoffs from `DBAASP_v4.0.1`, `DBAASP`, `ChEMBL`, `cyclicpepedia_pubchem`, `cybase`.
   - `half_life_reg` (`+1,187`) & `half_life_cls` (`+1,222`): `PEPlife2` semantic review (419), `Pepti-DrugPhAI` (264), censored inequalities (173), `PLifePred` seconds→hours (126), `approx (~)` relations (92+), `PepTherDia` / `openFDA` / `EMA SmPC` / `B3Pdb`.
   - `CL_absolute` (`+47`), `CL_per_kg` (`+111`), `Vd_per_kg` (`+139`), `F` (`+68`), `PPB` (`+75`): `Pepti-DrugPhAI`, `openFDA/DailyMed`, `EMA SmPC`, `PepTherDia` (incl. 13 range intervals), `ChEMBL`, `Helicogenic 2026`, `primary_pk_literature_v14`, `% bound / % unbound → fraction`, and `mL/min → L/h` conversions.
3. **Explicit Scientific Exclusions & Deduplication Conflict Drops**:
   - Step 01 hard exclusions: `hemolysis_cls|no_smiles_and_no_seq` (**2,058**), `cytotoxicity_cls|excluded_zebrafish_embryo` (**1,591** — non-cell-line organismal assay), `CL|excluded_renal_clearance` (**28**), `F|excluded_iv_route` (**13** — IV F≡100% tautology), `F|excluded_ambiguous_multi_value_sentence` (**7**), `half_life_reg|chembl_non_pk_enzyme_halflife` (**5**), `BBB` & `solubility_cls` missing structure+seq (**4**).
   - Step 02 deduplication conflicts dropped (incompatible replicate labels across identical compound + condition): **350** total (`hemolysis_cls` 157, `half_life_cls` 147, `cytotoxicity_cls` 15, `half_life_reg` 12, others ≤3).

---

## 4. Censored & Interval Target Statistics (`2,523` Rows = `3.65%`)

All non-exact observations carry explicit `censor_type ∈ {right, left, interval}` and finite physical bounds `[y_lower, y_upper]` ready for survival / Tobit / interval-censored loss heads:

| Endpoint | Exact | Right-Censored (`>`) | Left-Censored (`<`) | Interval (`[a, b]`) | Total Non-Exact | % Non-Exact | Finite Domain `[dmin, dmax]` |
|---|---:|---:|---:|---:|---:|---:|---|
| **hemolysis_HC50** | 4,184 | 1,339 | 36 | 0 | **1,375** | 24.73% | `[-3.0, 5.5]` log10 µM |
| **MIC_S_aureus** | 1,787 | 342 | 2 | 0 | **344** | 16.14% | `[-4.0, 5.5]`* log10 µM |
| **MIC_E_coli** | 1,954 | 249 | 7 | 0 | **256** | 11.58% | `[-4.0, 5.5]` log10 µM |
| **MIC_P_aeruginosa** | 960 | 198 | 3 | 0 | **201** | 17.31% | `[-4.0, 5.5]` log10 µM |
| **half_life_reg** | 2,901 | 110 | 67 | 0 | **177** | 5.75% | `[-4.0, 4.0]` log10 h |
| **MIC_C_albicans** | 412 | 53 | 3 | 0 | **56** | 11.97% | `[-4.0, 5.5]` log10 µM |
| **PPB** | 132 | 14 | 3 | 13 | **30** | 18.52% | `[0.0, 1.0]` fraction |
| **solubility_reg** | 272 | 21 | 5 | 0 | **26** | 8.72% | `[-4.0, 6.0]` log10 µM |
| **MDCK** | 152 | 2 | 14 | 0 | **16** | 9.52% | `[-10.0, -2.0]` log10 cm/s |
| **Caco2** | 1,567 | 0 | 13 | 0 | **13** | 0.82% | `[-10.0, -2.0]` log10 cm/s |
| **F** | 114 | 2 | 8 | 0 | **10** | 8.06% | `[0.0, 1.0]` fraction |
| **PAMPA** | 8,572 | 1 | 8 | 0 | **9** | 0.10% | `[-10.0, -2.0]` log10 cm/s |
| **CL_per_kg** | 199 | 2 | 5 | 0 | **7** | 3.40% | `[-4.0, 3.5]` log10 mL/min/kg |
| **LogD7.4** | 675 | 3 | 0 | 0 | **3** | 0.44% | `[-6.0, 10.0]` LogD |
| **Other 10 endpoints** | 45,229 | 0 | 0 | 0 | **0** | 0.00% | exact / binary |
| **TOTAL** | **66,588** | **2,331** | **179** | **13** | **2,523** | **3.65%** | 100% finite, `y_lower < y_upper` |

*\*2 exact ChEMBL `MIC_S_aureus` rows equal `6.000` (`10^6 µM`); all right-censored tails cap at `5.5`.*

---

## 5. Unified Non-Chaining Clustering & Split Architecture

To eliminate Wave-1's mega-cluster percolation bug (`CLU_000757` = 3,801 compounds across 18/21/22/30-membered macrocycles and even 96 linear peptides caused by Murcko/generic ring stripping chained transitively with sequence identity), [`03_cluster_and_split.py`](file:///D:/work/中检院/v33_upgrade/data_pipeline/03_cluster_and_split.py) implements a deterministic two-stage, non-chaining design:

1. **Stage (a) — Exact `canonical_identity_group`** (`46,597` compounds → `39,992` atomic identity units): Union-Find strictly on identical non-empty `inchikey_14`, `conn_hash`, or `canonical_smiles`. No atomic unit is ever split.
2. **Stage (b) — Non-Chaining Dual-Track Leader Clustering** (`39,992` units → **`17,123` clusters**, `max = 373`, `median = 1.0`):
   - **Track S (`30,374` units → `15,124` clusters)**: Linear sequence-defined peptides (`is_cyclic == 0` with verified canonical sequence or missing SMILES) clustered via CD-HIT-style length-sorted leader clustering at `>= 70%` global normalized edit identity. Avoids binary-Morgan bag-of-20AA collisions across unrelated linear peptides.
   - **Track M (`9,618` units → `1,999` clusters)**: Cyclic, macrocyclic, short, and chemically modified SMILES peptides clustered via exact Taylor-Butina leader clustering on `morgan_bits_1024.npz` at `Tanimoto >= 0.70` (zero transitive chaining; every member has `T >= 0.70` to its cluster centroid).
3. **Stratified LPT Size-Balanced Cluster Bin-Packing + 1-Move Refinement**:
   - **Outer Holdout (`~20%`)**: Cluster-level stratified 5-bin LPT partition (`18.0%–23.0%` across all 24 endpoints; `19.8%–20.4%` on 21/24 endpoints).
   - **Inner 5-Fold CV (`~80%`)**: Size-balanced stratified cluster bin-packing minimizing joint marginal variance of fold row counts and class/quantile strata counts. Achieves **24/24 PASS** on `[10%, 35%]` fold balance (exact `20.0%` balance on large endpoints like `PAMPA` `[1373×5]`, `hemolysis_cls` `[3281×5]`, `BBB` `[135×5]`, `MIC_S_aureus` `[341×5]`).

---

## 6. Frozen Compact Anti-`(p >> n)` Feature Bank Inventory

All feature artifacts reside on `node23:/data/qlyu/v33_upgrade/features/` (RDKit `2026.09.1`, NumPy `2.2.6`, computed in `3,423.67s`, covering **46,598** entries = 46,597 V33 compounds + 1 Case17-only query `Tirzepatide` at index `46597`, **43,943** `smiles_valid`, **44,773** `seq_valid`):

| File | Arrays / Contents | Shape | Dtype | Size | Verified SHA-256 (first 16 hex) |
|---|---|---|---|---:|---|
| `compound_index.tsv` | Unified `0..46597` index (`compound_key`, `smiles_valid`, `seq_valid`) | 46,598 rows | TSV | 3.04 MiB | `800e73ddf2306aa7` |
| `physchem_core.npz` | `X` (38 RDKit 2D/charge + 10 peptide modification flags + 35 sequence physchem/AAC/µH/pI/Boman) + valid masks | `(46598, 83)` | `float32` | 6.99 MiB | `a6981a31f01a8c20` |
| `fp_compact.npz` | `morgan_count_r2_1024` (chiral), `atompair_1024` (chiral), `maccs_167`, `torsion_512` (chiral) | `1024 / 1024 / 167 / 512` (2,727D) | `uint16` / `uint8` | 42.25 MiB | `aa9691e9b4b5e622` |
| `morgan_bits_1024.npz` | `morgan_bits_1024` (chiral binary) + `morgan_bits_packed` (fast AD Tanimoto lookup) | `(46598, 1024)` / `(46598, 128)` | `uint8` | 5.96 MiB | `442e8d3b6b8f49ed` |
| `feature_manifest.json` | Full feature names, Case17 indices (`0..46597`), versions, SHA-256 checksums | JSON | UTF-8 | 4.4 KiB | verified in Step 05 |

---

## 7. Case17 Clinical Peptide Benchmark Overlap Summary

All **17 clinical peptide drugs** in [`case17_queries.tsv`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/case17_queries.tsv) have `smiles_valid = 1`, verified RDKit modification flags, and exact indices in `features/compound_index.tsv`. Conservative multi-key matching (`exact_canonical_smiles` → `exact_inchikey` → `inchikey_connectivity_14` → `connectivity_smiles_hash` → `exact_parent_sequence` for `len ≥ 6`) flagged **439 rows** (`case17_excluded = 1`) in [`case17_overlap.tsv`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/case17_overlap.tsv) so downstream training can enforce strict zero-leakage Case17 leave-out evaluation:

- **By Match Tier**: `exact_canonical_smiles`: **256** (58.3%) | `exact_inchikey`: **75** (17.1%) | `exact_parent_sequence`: **75** (17.1%, conservative exclusion of bare/analog sequence matches) | `inchikey_connectivity_14`: **33** (7.5%).
- **By Drug (All 17 Represented)**:
  - `Exenatide` (92: `half_life_reg` 43, `half_life_cls` 39, `CL_absolute` 3, `Cmax` 2, `BBB`/`CL_per_kg`/`F`/`PPB`/`Vd_per_kg` 1 ea — 58 via conservative `exact_parent_sequence`), `Cyclosporine_A` (72 across 15 endpoints), `Daptomycin` (48 across 11 endpoints), `Liraglutide` (36), `Teriparatide` (31), `Semaglutide` (28), `Caspofungin` (26), `Oxytocin` (17), `Ziconotide` (16), `Octreotide` (15), `Leuprolide` (14), `Glucagon` (12), `Bivalirudin` (10), `Linaclotide` (8), `Tirzepatide` (6, all via `exact_inchikey`), `Insulin_degludec` (4), `Insulin_detemir` (4).
- **By Endpoint (Top Contributors)**: `half_life_reg` (133), `half_life_cls` (117), `Vd_per_kg` (37), `CL_per_kg` (29), `PPB` (25), `Cmax` (24), `CL_absolute` (16), `F` (16), `cytotoxicity_cls` (8), `MDCK` (5), `MIC_S_aureus` (4), all remaining endpoints ≤ 3 rows each.
