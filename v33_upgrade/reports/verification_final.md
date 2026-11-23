# Peptide OmniPanel V33 — Independent Final Verification & End-to-End Scientific/Production Audit

- **Audit Wave**: Wave 3C (Independent Skeptical Final Verification — Strict Read-Only)
- **Audit Date**: 2026-11-22 CST
- **Scope**: `internal_research_only`
- **Audited Environment**: Local Windows (`D:\work\中检院\v33_upgrade\`), `node23` (`10.0.0.3:8902` live V33 inference engine & authoritative bundle), `node1` (SSH tunnel + shared RTX 4090 GPU compliance), `node2` (`<APP_HOST>:7863` staged FastAPI+SPA platform & live V31/V32 production isolation)
- **Overall Staging Sign-Off**: **GO (APPROVED FOR STAGING — 6/6 GATES PASSED, ZERO BLOCKING DEFECTS)**

---

## Executive Gate Scorecard

| Gate | Audit Domain | Verdict | Key Independently Verified Evidence |
|---:|---|:---:|---|
| **Gate 1** | Data Integrity, Split Balance & Zero-Leakage | **PASS** | Exact match across `verification_report.json`, `embedding_manifest.json`, and `v33_manifest.json` (SHA-256 `433696c2…` identical across local/`node23`/`node2`): `46,597` compounds (`46,598` incl. Case17), `24` endpoints (`20+4`), `69,111` rows (`55,299` CV + `13,812` HO), `2,523` censored, `0` leakage, `24/24` balanced 5-fold splits. **46/46** bundle SHA-256 checksums, **110/110** deep `.pt` checkpoints, and **4/4** embedding `.npz` caches verified on disk; `0` NaN/Inf. |
| **Gate 2** | Scientific Benchmark Truthfulness & Leave-Case17-Out | **PASS** | Independently recomputed **all 24 endpoints × 2 splits × 2 tracks + 18 conformal coverages** from raw TSVs on `node23`: **0 mismatches** (`max_diff = 5e-6`, strictly `<< 1e-3`). **260/260** V32 reference fields verified against raw `endpoint_representatives.tsv` (`0` fabricated metrics). Leave-Case17-Out independently recomputed from `case17_champion.tsv`: `half_life_reg` pure-ML `MAE=0.3306 / Sp=0.9029 / R²=0.7627` (+ anchor `MAE=0.2987 / R²=0.7813`), `PPB` `MAE=0.2264 / Sp=0.7093`, hemolysis FP `0/17 > 0.50` (vs V32 `8/17 > 0.60`). |
| **Gate 3** | Live End-to-End Pipeline (`node2:7863 → node1:8902 → node23:8902`) | **PASS** | `/api/v1/health` OK (`v33_remote`, 24 EPs); `/api/v1/endpoints` **0 diffs** vs `v33_manifest.json`; all 6 `/api/v1/predict` test cases (incl. explicit C18-diacid SMILES, genuinely out-of-corpus cold unseen synthetic peptides, and HTTP 400 garbage rejection) + 3-molecule mixed CSV & FASTA `/api/v1/batch` (`3/3`, `0` failed, `0` NaN/Inf) verified live; `/` HTML uses subpath-safe relative assets. |
| **Gate 4** | Visual Quality Audit (All 5 Live V33 Screenshots) | **PASS** | All 5 live Playwright screenshots independently reviewed (Grades: `A`, `A-`, `A`, `A`, `A`). Single-row example chips, unclipped 6-axis radar labels, zero scientific-notation line wraps, authentic manifest numbers, prominent `internal_research_only` notices; 3 minor non-blocking UX polish notes logged. |
| **Gate 5** | Production Isolation & Zero-Disruption | **PASS** | V31 (`:7861`), V32 (`:7862`), V32 tunnel (`node1` + `node2:8901`), V32 backend (`node23:8901`), Nginx snippet (`/etc/nginx/snippets/peptide-omnipanel.location.conf`, mtime `2026-11-15`), and all 3 legacy data/model directories verified **100% intact, running, and unmodified**. `pytest` **11/11 PASSED** (`1.78 s`). |
| **Gate 6** | Resource Governance Compliance (`node1` GPU Hard Rule) | **PASS** | Live `node1` has **0** V33 GPU jobs (`v33_tunnel.sh` SSH tunnel only). Audited all `223` logged 2-sample checks in `gpu_compliance_log.jsonl`: `219` blocked (`0` GPUs allocated); `4` passed checks strictly respected `<50% VRAM AND <50% Util, max 2 GPUs` (allocated `1` GPU max, peak pre-launch `14.84%` VRAM / `32%` Util). 3/4 FMs + all deep training offloaded to idle local RTX 5080; all other waves & live serving ran on `node23` CPUs (`0` GPU). |

---

## Gate 1 — Data Integrity, Split Balance & Zero-Leakage Audit

**Verdict: PASS**

### 1.1 Cross-Artifact Count & Hash Reconciliation
Three independent manifest copies were hashed and cross-checked directly against [`verification_report.json`](file:///D:/work/中检院/v33_upgrade/data_pipeline/audit_outputs/verification_report.json) and `node23:<WORK_DIR>/v33_upgrade/embeddings/embedding_manifest.json`:

- **Manifest Identity (Bit-for-Bit Identical Across All 3 Hosts)**:
  - Local [`v33_manifest.json`](file:///D:/work/中检院/v33_upgrade/bundle/v33_manifest.json): `495,755` bytes, SHA-256 `433696c2bc0effacad81bf9cdeafd3a81339d8139791817d3a16b524461c5034`
  - `node23:<WORK_DIR>/v33_upgrade/bundle/v33_manifest.json`: `495,755` bytes, SHA-256 `433696c2bc0effacad81bf9cdeafd3a81339d8139791817d3a16b524461c5034`
  - `node2:/var/www/peptide-omnipanel/v33_next/v33_manifest.json`: `495,755` bytes, SHA-256 `433696c2bc0effacad81bf9cdeafd3a81339d8139791817d3a16b524461c5034`
- **Recursive NaN/Inf Audit**: `0` NaN or Inf floating-point values across the entire `v33_manifest.json` document tree.

| Quantity | `verification_report.json` | `embedding_manifest.json` | `v33_manifest.json` | Sum Over 24 Endpoints | Match |
|---|---:|---:|---:|---:|:---:|
| Unique V33 compounds | `46,597` | — | `46,597` | — | **EXACT** |
| Feature-bank compounds (incl. Case17) | `46,598` | `46,598` | `46,598` | — | **EXACT** |
| Valid SMILES (`smiles_valid == 1`) | `43,943` | `43,943` | — | — | **EXACT** |
| Valid Sequences (`seq_valid == 1`) | `44,773` | `44,773` | — | — | **EXACT** |
| Unified endpoints (Tier-1 + Tier-2) | `24` (`20 + 4`) | — | `24` (`20 + 4`) | `24` (`6` cls + `18` reg) | **EXACT** |
| Total endpoint rows | `69,111` | — | `69,111` | `69,111` | **EXACT** |
| CV rows (`is_holdout == 0`) | `55,299` | — | `55,299` | `55,299` | **EXACT** |
| Holdout rows (`is_holdout == 1`) | `13,812` | — | `13,812` | `13,812` | **EXACT** |
| Censored / interval rows | `2,523` | — | `2,523` | `2,523` | **EXACT** |
| Case17 overlap excluded rows | `439` | — | `439` (sum `n_case17_excluded`) | `439` | **EXACT** |
| Balanced 5-fold splits passed | `24 / 24` (`true`) | — | — | `24 / 24` | **EXACT** |
| Zero cluster/SMILES/InChIKey leakage | `24 / 24` (`true`) | — | — | `24 / 24` | **EXACT** |

Per-endpoint count verification between `verification_report.json` and `v33_manifest.json["endpoints"][ep]["counts"]` yielded **0 discrepancies** across all 24 endpoints.

### 1.2 On-Disk SHA-256 & Serving Artifact Integrity (`node23`)
Every artifact checksum recorded in `v33_manifest.json["sha256_checksums"]` and every embedding cache in `embedding_manifest.json` was recomputed directly from disk on `node23`:

- `ad_analog_index.npz` (`3,824,492` bytes): SHA-256 `021b7ac193b1df17…` — **MATCH**
- `analog_lookup.json.gz` (`3,480,980` bytes): SHA-256 `64661d268a12cc66…` — **MATCH**
- `v32_reference_snapshot.json` (`28,112` bytes): SHA-256 `51007d1acbb1a0d5…` — **MATCH**
- `serving_models/`: Exact file-set match (`43 / 43` pruned `.joblib` files, `10,459.97 MB`), **43 / 43 SHA-256 hashes MATCH** (`46 / 46` total bundle checksums verified, **`0` failures**).
- `deep_checkpoint_refs`: **110 / 110** `.pt` checkpoints verified present on disk and non-empty (`0` missing).
- Pretrained foundation-model embedding caches (`node23:<WORK_DIR>/v33_upgrade/embeddings/`):
  - `ncaa_repdistill_v2.npz` (`46,598` rows, `43,943` valid): SHA-256 **MATCH**, `zero_nan_inf = True`
  - `peptideclm2.npz` (`46,598` rows, `43,943` valid): SHA-256 **MATCH**, `zero_nan_inf = True`
  - `esmc_600m_multiregion.npz` (`46,598` rows, `44,773` valid): SHA-256 **MATCH**, `zero_nan_inf = True`
  - `chemberta_mtr.npz` (`46,598` rows, `43,943` valid): SHA-256 **MATCH**, `zero_nan_inf = True`

---

## Gate 2 — Scientific Benchmark Truthfulness & Leave-Case17-Out Audit

**Verdict: PASS**

### 2.1 Full 24-Endpoint Raw TSV Recomputation vs. `v33_manifest.json`
Rather than spot-checking 8 endpoints, we independently recomputed **all 24 endpoints** directly from raw prediction TSVs on `node23:<WORK_DIR>/v33_upgrade/bundle/champion_predictions/<ep>/{oof_champion.tsv, holdout_champion.tsv}` across both tracks (`dual_modality`, `smiles_only`) and all 18 regression Conformal 80% / 90% empirical coverages (`lo80/hi80/lo90/hi90`):

- **Dual-Modality (24 endpoints × OOF & Holdout + 18 Conformal coverages)**: **`max_diff = 0.000005`**, **0 mismatches** above `1e-3`.
- **SMILES-Only Track A (24 endpoints × OOF & Holdout)**: On the 4 endpoints with `<100%` valid SMILES (`half_life_cls` 92.57%, `half_life_reg` 93.89%, `hemolysis_cls` 88.21%, `solubility_cls` 99.65%), filtering `holdout_champion.tsv` and `oof_champion.tsv` to `smiles_valid == 1` via `compound_index.tsv` (`595`, `573`, `3,629`, and `743` holdout rows respectively) reproduces `v33_manifest.json["smiles_only"]` with **`max_diff = 0.000005` (`0` mismatches)**.
- **Summary MD Parity**: Every dual-modality holdout primary metric across all 24 endpoints in [`v33_champion_summary.md`](file:///D:/work/中检院/v33_upgrade/reports/v33_champion_summary.md) matches `v33_manifest.json` with **0 missing or discrepant values**.

#### Recomputed Raw TSV vs. `v33_manifest.json` Verification Table (All 24 Endpoints — Untouched 20% Holdout, Dual Track)

| Endpoint | Type | Holdout `n` (Exact) | Metric 1: Recomputed / Manifest | Metric 2: Recomputed / Manifest | Calib / Conf 80% & 90%: Recomputed / Manifest | Max `|Δ|` |
|---|---|---:|---:|---:|---:|---:|
| **BBB** | CLS | 169 | AUC `0.89874` / `0.89874` | PR `0.88976` / `0.88976` | Brier `0.12888`/`0.12888` · ECE `0.05751`/`0.05751` | `≤5e-6` |
| **cytotoxicity_cls** | CLS | 587 | AUC `0.93421` / `0.93421` | PR `0.89504` / `0.89504` | Brier `0.09426`/`0.09426` · ECE `0.03662`/`0.03662` | `≤5e-6` |
| **half_life_cls** | CLS | 641 | AUC `0.78175` / `0.78175` | PR `0.77416` / `0.77416` | Brier `0.19129`/`0.19129` · ECE `0.05006`/`0.05006` | `≤5e-6` |
| **hemolysis_cls** | CLS | 4,101 | AUC `0.89734` / `0.89734` | PR `0.80945` / `0.80945` | Brier `0.11263`/`0.11263` · ECE `0.01961`/`0.01961` | `≤5e-6` |
| **overall_peptide_toxicity** | CLS | 2,208 | AUC `0.95153` / `0.95153` | PR `0.95494` / `0.95494` | Brier `0.08916`/`0.08916` · ECE `0.03850`/`0.03850` | `≤5e-6` |
| **solubility_cls** | CLS | 745 | AUC `0.86649` / `0.86649` | PR `0.90668` / `0.90668` | Brier `0.13929`/`0.13929` · ECE `0.02383`/`0.02383` | `≤5e-6` |
| **CL_absolute** | REG | 11 (11) | MAE `1.18023` / `1.18023` | Sp `0.65754` / `0.65754` | Cov80 `0.6364`/`0.6364` · Cov90 `0.7273`/`0.7273` | `≤5e-6` |
| **CL_per_kg** | REG | 41 (41) | MAE `0.54773` / `0.54773` | Sp `0.83199` / `0.83199` | Cov80 `0.7073`/`0.7073` · Cov90 `0.9268`/`0.9268` | `≤5e-6` |
| **Caco2** | REG | 315 (311) | MAE `0.36331` / `0.36331` | Sp `0.85280` / `0.85280` | Cov80 `0.7878`/`0.7878` · Cov90 `0.8907`/`0.8907` | `≤5e-6` |
| **Cmax** | REG | 17 (17) | MAE `0.91561` / `0.91561` | Sp `0.61435` / `0.61435` | Cov80 `0.8235`/`0.8235` · Cov90 `0.8824`/`0.8824` | `≤5e-6` |
| **F** | REG | 25 (23) | MAE `0.23640` / `0.23640` | Sp `0.72754` / `0.72754` | Cov80 `0.7826`/`0.7826` · Cov90 `0.8261`/`0.8261` | `≤5e-6` |
| **LogD7.4** | REG | 125 (125) | MAE `0.45067` / `0.45067` | Sp `0.69853` / `0.69853` | Cov80 `0.8080`/`0.8080` · Cov90 `0.9040`/`0.9040` | `≤5e-6` |
| **MDCK** | REG | 34 (34) | MAE `0.54257` / `0.54257` | Sp `0.35888` / `0.35888` | Cov80 `0.6765`/`0.6765` · Cov90 `0.7941`/`0.7941` | `≤5e-6` |
| **MIC_C_albicans** | REG | 94 (90) | MAE `0.43867` / `0.43867` | Sp `0.71491` / `0.71491` | Cov80 `0.8667`/`0.8667` · Cov90 `0.9444`/`0.9444` | `≤5e-6` |
| **MIC_E_coli** | REG | 442 (397) | MAE `0.43226` / `0.43226` | Sp `0.66323` / `0.66323` | Cov80 `0.8161`/`0.8161` · Cov90 `0.9270`/`0.9270` | `≤5e-6` |
| **MIC_P_aeruginosa** | REG | 232 (197) | MAE `0.65717` / `0.65717` | Sp `0.13273` / `0.13273` | Cov80 `0.7563`/`0.7563` · Cov90 `0.8274`/`0.8274` | `≤5e-6` |
| **MIC_S_aureus** | REG | 426 (355) | MAE `0.47582` / `0.47582` | Sp `0.43452` / `0.43452` | Cov80 `0.8704`/`0.8704` · Cov90 `0.9606`/`0.9606` | `≤5e-6` |
| **PAMPA** | REG | 1,716 (1,715) | MAE `0.37375` / `0.37375` | Sp `0.78282` / `0.78282` | Cov80 `0.8006`/`0.8006` · Cov90 `0.9026`/`0.9026` | `≤5e-6` |
| **PPB** | REG | 32 (24) | MAE `0.26080` / `0.26080` | Sp `0.67828` / `0.67828` | Cov80 `0.2917`/`0.2917` · Cov90 `0.9167`/`0.9167` | `≤5e-6` |
| **RRCK** | REG | 30 (30) | MAE `0.26825` / `0.26825` | Sp `0.53370` / `0.53370` | Cov80 `0.8667`/`0.8667` · Cov90 `0.9333`/`0.9333` | `≤5e-6` |
| **Vd_per_kg** | REG | 33 (33) | MAE `0.32954` / `0.32954` | Sp `0.71409` / `0.71409` | Cov80 `0.8788`/`0.8788` · Cov90 `0.9394`/`0.9394` | `≤5e-6` |
| **half_life_reg** | REG | 616 (567) | MAE `0.66401` / `0.66401` | Sp `0.48470` / `0.48470` | Cov80 `0.7478`/`0.7478` · Cov90 `0.8783`/`0.8783` | `≤5e-6` |
| **hemolysis_HC50** | REG | 1,112 (820) | MAE `0.37839` / `0.37839` | Sp `0.67528` / `0.67528` | Cov80 `0.8220`/`0.8220` · Cov90 `0.9220`/`0.9220` | `≤5e-6` |
| **solubility_reg** | REG | 60 (48) | MAE `0.58074` / `0.58074` | Sp `0.66645` / `0.66645` | Cov80 `0.7500`/`0.7500` · Cov90 `0.8958`/`0.8958` | `≤5e-6` |

### 2.2 Aggregate Summaries, Ablation Wins & Zero Fabricated V32 Metrics
- **Overall Dual-Track Means**:
  - `6` Classification endpoints: Mean Holdout ROC-AUC recomputed `0.88834` (CPU `0.86725`, Fusion `0.88679`, Deep `0.86918`), PR-AUC `0.87167`, Brier `0.12592`, ECE `0.03769` — **exact match**.
  - `18` Regression endpoints: Mean Holdout MAE recomputed `0.50533` (CPU `0.55163`, Fusion `0.52464`, Deep `0.53701`), Spearman `0.62346` (CPU `0.58823`, Fusion `0.61745`, Deep `0.58594` vs manifest `0.58595`, benign `1e-5` rounding of 5-decimal inputs), Cov80/90 `0.76051 / 0.88883` (`10` medium/large: `0.80841 / 0.90633`) — **exact match**.
  - Champion mix counts: Dual `{super_ensemble: 13, fusion: 7, deep: 3, cpu: 1}`; SMILES-only `{fusion: 10, super_ensemble: 9, cpu: 3, deep: 2}` — **exact match**.
  - Single-FM holdout wins across 24 endpoints: `{esmc_600m: 9, ncaa_repdistill_v2: 9, peptideclm2: 4, chemberta_mtr: 2}` — **exact match**.
- **V32 Reference Integrity**:
  - `v33_manifest.json` vs. [`v32_reference_snapshot.json`](file:///D:/work/中检院/v33_upgrade/bundle/v32_reference_snapshot.json): **0 diffs** across all 24 endpoints.
  - `v32_reference_snapshot.json` vs. original V32 raw [`endpoint_representatives.tsv`](file:///D:/work/中检院/.publish/peptide_data_v31/omnigrid/configs/endpoint_representatives.tsv): **260 / 260 metric fields matched with 0 diffs**.
  - Newly added endpoints (`cytotoxicity_cls`, `MIC_S_aureus`, `MIC_P_aeruginosa`, `MIC_C_albicans`) are strictly `null` (`—` in UI); V32 CV-only endpoints (`CL_absolute`, `Cmax`) have `holdout_* = null` and are flagged `internal5_no_holdout` (`*` in UI). **Zero fabricated metrics.**

### 2.3 Strict Leave-Case17-Out Clinical Benchmark Recomputation
Independently recomputed from raw `node23:<WORK_DIR>/v33_upgrade/bundle/champion_predictions/{half_life_reg,PPB,hemolysis_cls}/case17_champion.tsv` (**0 per-drug discrepancies** vs `v33_manifest.json["case17_scorecard"]["drugs"]`):

| Clinical Benchmark Metric | `n` | Recomputed (Raw TSV) | `v33_manifest.json` | Match |
|---|---:|---|---|---|
| `half_life_reg` Pure-ML Route-Matched (`log10 h`) | 16 | MAE `0.3306`, RMSE `0.4289`, R² `0.7627`, Sp `0.9029`, r `0.8986` | MAE `0.3306`, RMSE `0.4289`, R² `0.7627`, Sp `0.9029`, r `0.8986` | **EXACT** |
| `half_life_reg` + Fatty-Acylation Anchor (`log10 h`) | 16 | MAE `0.2987`, RMSE `0.4118`, R² `0.7813`, Sp `0.9028`, r `0.9036` | MAE `0.2987`, RMSE `0.4118`, R² `0.7813`, Sp `0.9028`, r `0.9035` | **EXACT** (`1e-4` float rnd) |
| `PPB` Dual Champion | 14 | MAE `0.2264`, RMSE `0.2771`, R² `0.3570`, Sp `0.7093`, r `0.7545` | MAE `0.2264`, RMSE `0.2771`, R² `0.3570`, Sp `0.7093`, r `0.7545` | **EXACT** |
| `PPB` Censored Diagnostic Head | 14 | MAE `0.2253`, RMSE `0.3022`, R² `0.2353`, Sp `0.3524`, r `0.5748` | MAE `0.2253`, RMSE `0.3022`, R² `0.2353`, Sp `0.3524`, r `0.5748` | **EXACT** |
| FDA Hemolysis False Positives (`hemolysis_cls`) | 17 | Max `0.4289` (`Cyclosporine_A`), **`0/17 > 0.50`** | Max `0.4289`, **`0/17 > 0.50`** (raw V32 TSV verified **`8/17 > 0.60`**) | **EXACT** |

---

## Gate 3 — Live End-to-End Pipeline Audit (`node2:7863 → node1 :8902 → node23:8902`)

**Verdict: PASS**

Tested directly on `node2` against `http://127.0.0.1:7863` (backed live via `node1` reverse SSH tunnel `:8902` to `node23:8902`):

1. **`GET /api/v1/health`**: HTTP `200` (`10 ms`) — `ok: true`, `version: "33.0.0"`, `adapter_mode: "v33_remote"`, `v33_remote_healthy: true`, `endpoints_total: 24`, `tier1_endpoints: 20`, `tier2_endpoints: 4`, `rdkit_version: "2026.03.6"`, `scope: "internal_research_only"`.
2. **`GET /api/v1/endpoints`**: HTTP `200` (`40,710` bytes) — 1-to-1 programmatic audit across all 24 endpoints (`n_train`, `n_holdout`, `n_total`, `n_censored`, `weights`, `cv_score_v32`, `holdout_score_v32`, `cv_score_v33`, `holdout_score_v33`, `conformal_coverage_80/90`, `brier_holdout`, `ece_holdout`) and all top-level summaries (`dataset_summary`, `overall_summary`, `representation_ablation_summary`, `case17_scorecard`, `selection_protocol`) yielded **0 discrepancies** against `v33_manifest.json`.
3. **`GET /`**: HTTP `200` (`32,028` bytes) — clean HTML with relative subpath-safe asset links (`href="static/styles.css"`, `src="static/app.js"`, `href="docs"`, `href="redoc"`), compatible with both `/` and `/peptide-next/`.
4. **`POST /api/v1/predict` (All 6 Live Scenarios + Out-of-Corpus Unseen Verification)**:

| # | Live Test Case | HTTP | Track | Runtime (Cold / Cached) | Key Independently Verified Outputs | NaN/Inf |
|---|---|:-:|---|---:|---|---:|
| 1a | `Semaglutide` (name, `conditions.pk_context="human_sc"`) | `200` | `smiles_only` | `6.71 s` / `0.23 s` | `<svg>` present; `weekly_diacid` (`120–168 h`, Pt `144 h`); Pure-ML SC `51.68 h` (IV `21.57 h`); `24` EPs; `PAMPA` CI80 `1.13e-07 – 1.72e-06`, `Caco2` CI80 `3.92e-08 – 7.57e-07`; Top-3 analogs populated (`3/3`); `hemolysis_cls = 9.5%` (`< 25%`) | `0` |
| 1b | `Semaglutide` (explicit 647-char C18-diacid SMILES, `human_sc`) | `200` | `smiles_only` | `0.22 s` *(LRU hit)* | Identical outputs: `<svg>` present, `weekly_diacid`, `51.68 h`, non-zero `PAMPA`/`Caco2` CI80s, `3/3` analogs, `hemolysis_cls = 0.09472` | `0` |
| 2 | `Tirzepatide` (`conditions.pk_context="human_sc"`) | `200` | `smiles_only` | `6.98 s` / `0.51 s` | `weekly_diacid` (`120–168 h`); Pure-ML SC `69.42 h` (IV `31.35 h`); `PPB = 80.6%`; `hemolysis_cls = 17.2%` (`< 25%`); `24` EPs | `0` |
| 3 | `Cyclosporine A` | `200` | `smiles_only` | `6.50 s` / `0.02 s` | `<svg>` + RDKit highlights `Macrocycle Ring (33-atom)` & `N-Methylated Amide (×7)`; `PPB = 89.6%` (`> 75%`); `T½ = 8.64 h` (FDA `~8.4 h`) | `0` |
| 4 | `Classic AMP` (`HISEAVKIFGCKWKENWKWNSNC`, `include_tier2=true`) | `200` | `dual_modality` | `17.88 s` / `0.14 s` | RDKit seq→SMILES `<svg>` present; `24` EPs; all 4 Tier-2 MICs populated with CIs & Top-3 analogs: *E. coli* `20.31 µM [3.58–115.3]`, *S. aureus* `15.65 µM [2.19–111.6]`, *P. aeruginosa* `24.45 µM [4.73–126.4]`, *C. albicans* `52.58 µM [8.18–338.0]` | `0` |
| 5a | `Magainin-2` (`GIGKFLHSAKKFGKAFVGEIMNS`) | `200` | `dual_modality` | `18.51 s` / `0.06 s` | Live prediction succeeds cleanly (`24` EPs, `<svg>`); *audit note*: confirmed present in `compounds_v33.tsv` (`Tc=1.00` on hemolysis/MICs) | `0` |
| 5b | Brand-New Unseen Synthetic Peptides (`MQWVYHCGFPLKSRTDAENIQV` & `KWFVLYHCGTPNRSDAEQMIK`) | `200` | `dual_modality` | `19.97 s` & `27.46 s` *(true cold, `cache_hit=False`)* | Confirmed **absent** from `compounds_v33.tsv` (`max_tanimoto < 1.0` across all EPs); full 4-FM live encoding + 24-EP prediction + RDKit `<svg>` + Top-3 analogs succeeded cleanly (`KWF…`: `T½ 0.525 h`, `PPB 50.7%`, `Hemol 4.9%`, *E. coli* MIC `39.36 µM`) | `0` |
| 6 | Garbage Input (`"!!!not_a_peptide_123@@@"`) | **`400`** | — | `0.003 s` | Clean HTTP `400` validation error (`"Unable to parse input '!!!not_a_peptide_123@@@': not a recognized peptide drug name, canonical 20-AA sequence, or valid RDKit SMILES."`) | `0` |

5. **`POST /api/v1/batch`**:
   - 3-molecule mixed CSV (Drug name `Semaglutide` + Unseen sequence `MQWVYHCGFPLKSRTDAENIQV` + explicit `Cyclosporine A` SMILES): HTTP `200`, `total: 3`, `succeeded: 3`, `failed: 0`, `24` EPs each, `0` NaN/Inf.
   - 3-molecule FASTA (`ClassicAMP`, `Magainin2`, `UnseenSyn`): HTTP `200`, `total: 3`, `succeeded: 3`, `failed: 0`, `24` EPs each, `0` NaN/Inf.

*API Contract Observation (Non-Blocking)*: `PredictRequest` expects assay parameters nested inside `conditions: {"pk_context": "human_sc"}` (per `/openapi.json`). Sending a flat top-level `pk_context` key is silently ignored by Pydantic's default extra-field handling and defaults to `human_plasma` (IV). Callers integrating with the REST API must use the nested `conditions` object.

---

## Gate 4 — Visual Quality Audit of All 5 Live V33 Screenshots

**Verdict: PASS**

All 5 screenshots in [`D:\work\中检院\v33_upgrade\reports\screenshots\`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/) were directly inspected via multimodal image review:

| Screenshot | Grade | Independent Visual Inspection Findings |
|---|:---:|---|
| [`01_home_v33.png`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/01_home_v33.png) | **A** | Crisp light-theme academic layout; all 5 example chips (`Semaglutide`, `Tirzepatide`, `Cyclosporine A`, `Daptomycin`, `Classic AMP`) + `Advanced Options` align on **one single row**; KPI ribbon shows exact audited V33 numbers (`46,597`, `24 [20 + 4]`, `69,111 [2,523 Censored]`, `80%`); header badges display amber `Research Use Only` + green `v33_remote · 24 EPs`; footer displays `internal_research_only`. |
| [`02_semaglutide_v33.png`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/02_semaglutide_v33.png) | **A-** | High-res RDKit 2D structure with amber C18 fatty-diacid motif highlight; unclipped 6-axis radar (`Plasma Binding` & `Lipophilicity` well-margined); amber mechanism anchor banner (`120–168 h`, Pt `144 h`) paired transparently beside Pure-ML SC (`51.7 h`); zero `0.00e+00` CI errors (`PAMPA 1.13e-07 – 1.72e-06`, `Caco2 3.92e-08 – 7.57e-07`); expanded drawers accurately display `smiles_only` track champion weights, censored diagnostics, and Top-3 training analogs with `RIGHT` tags. *(Minor polish notes below.)* |
| [`03_amp_seq_v33.png`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/03_amp_seq_v33.png) | **A** | Clear `DUAL TRACK` & `Sequence · 23 AA` badges (`19.0 s · live`); RDKit canonical backbone 2D render; unclipped radar (confirmed `11.7 h` reflects `pk_context=human_sc` live SC prediction); Tier-2 tab cleanly presents all 4 reference-strain MICs with calibrated 80% CIs, distinct `MIC_E_coli` censored diagnostic (`20.08 uM` vs exact `20.31 uM`), Top-3 analogs, and `Selectivity & Strain Benchmark` (`HC50 / Best MIC = 6.8×`). |
| [`04_batch_v33.png`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/04_batch_v33.png) | **A** | Clean 7-molecule screening workflow (`Succeeded 7`, `Acylation Anchors 3`, `18.9 s`); **zero scientific-notation line wraps** (`120–168 h ★`, `5–13 h ★`, `4.41e-07`, `1.72e-07`, `2.48e-06` all single-line tabular numerals); low-risk hemolysis across all FDA peptides (`2.5%–29.6%`); unclipped 3-peptide radar overlay (`Semaglutide`, `Tirzepatide`, `Cyclosporine A`). |
| [`05_validation_v33.png`](file:///D:/work/中检院/v33_upgrade/reports/screenshots/05_validation_v33.png) | **A** | Prominent blue `Scientific Scope & Validation Disclosure (internal_research_only)` banner + `scope: internal_research_only` pill; 100% authentic manifest-hydrated KPIs, Strict Leave-Case17-Out Clinical Benchmark card, 4-FM Representation Dominance bars (`9/24, 9/24, 4/24, 2/24`), and all 24 endpoint rows including honest reporting of weak/small-n metrics (`PPB 29.2%` cov80, `MIC_P_aeruginosa 0.133` Sp, `—` for new endpoints). Zero placeholder text or fabricated numbers. |

*Non-Blocking Visual/UX Polish Suggestions (Future Iteration)*:
1. On `02_semaglutide_v33.png`, because `half_life_reg` OOF selection chose `champion_exact` (`weight=1.0`) for the censored head, the green banner reads `Censored AFT/Tobit Diagnostic: 51.68 h` (equal to the main estimate `51.68 h`); adding `(exact champion selected on OOF)` would clarify why the two values match.
2. On `05_validation_v33.png`, `CL_absolute` (`0.582*`) and `Cmax` (`0.738*`) correctly mark V32 CV-only (`internal5_no_holdout`) values with `*`; adding a one-line table footer defining `* = V32 5-fold CV only (no holdout)` would make the notation self-explanatory.
3. Unify minor ASCII `uM` strings in censored diagnostic banners with Greek `µM` used elsewhere in the UI.

---

## Gate 5 — Production Isolation & Zero-Disruption Audit

**Verdict: PASS**

| Production / Legacy Asset | Location | Status | Independent Verification Proof |
|---|---|:---:|---|
| Legacy V31 Web App | `node2:127.0.0.1:7861` | **INTACT** | PID `2362` running uninterrupted since `2026-11-04`; `GET /` → **HTTP `200`** (`86,290` bytes) |
| Live V32 Production Web App | `node2:127.0.0.1:7862` | **INTACT** | PID `2897002` running since `2026-11-19` (`/var/www/peptide-omnipanel/app` mtime `2026-09-23`); `GET /` → **HTTP `200`** (`59,759` bytes) |
| Live V32 Reverse Tunnel | `node1` → `node2:8901` | **INTACT** | `omnigrid_tunnel.sh` (PID `2526506/2526507`, running since `2026-11-15`); `GET http://127.0.0.1:8901/health` on `node2` → **HTTP `200`** (`models_cached: 152, endpoints: 55`) |
| Live V32 Inference Engine | `node23:0.0.0.0:8901` | **INTACT** | PID `1237413`; `GET http://127.0.0.1:8901/health` → **HTTP `200`** (`{"ok": true, "models_cached": 152, "entries": 57, "endpoints": 55}`) |
| Production Nginx Config | `node2:/etc/nginx/snippets/peptide-omnipanel.location.conf` | **INTACT** | `stat` mtime `2026-11-15 23:10:37 +0800` (`653` bytes); still routes `/peptide-omnipanel/` → `http://127.0.0.1:7862/` |
| Legacy Local Dataset | `D:\work\中检院\data\dataset_reclean_20261113_v1\` | **INTACT** | `3,150` entries, latest mtime `2026-11-13 18:50:50` |
| Legacy `node23` S3 Dir | `node23:<WORK_DIR>/s3/` | **INTACT** | Latest entry mtime `2026-11-19 14:56:05` |
| Legacy `node23` Modeling Dir | `node23:<WORK_DIR>/peptide_modeling_multiseed20h_v1/` | **INTACT** | Latest entry mtime `2026-11-15 16:17:25` |
| Web API Unit Test Suite | [`test_web_api.py`](file:///D:/work/中检院/v33_upgrade/web/tests/test_web_api.py) | **PASS** | `py -3.11 -m pytest -v` → **11/11 PASSED** (`1.78 s`, 0 failures) |

---

## Gate 6 — Resource Governance Compliance Audit (`node1` GPU Hard Rule)

**Verdict: PASS**

Hard rule audited: *Max 2 GPUs on `node1`, only after 2 consecutive `nvidia-smi` samples (5–10 s apart) confirm BOTH `< 50% VRAM (< 12,282 MiB)` AND `< 50% Util`.*

1. **Live `node1` Audit (`2026-11-22 07:39 CST`)**:
   - `nvidia-smi --query-compute-apps`: **0** `qlyu` V33 GPU compute processes on `node1` (all active compute processes belong to other users `zwdong`, `yxzhang`, `wxsong`, `jhu`, etc.).
   - `ps -ef | grep qlyu`: The ONLY V33 process on `node1` is the lightweight SSH reverse tunnel (`bash <WORK_DIR>/v33_tunnel.sh` + `ssh -N -R 127.0.0.1:8902:10.0.0.3:8902`, `0%` CPU, `0` GPU), running alongside the untouched V32 tunnel (`omnigrid_tunnel.sh`).
2. **Historical `embedding_manifest.json` & `gpu_compliance_log.jsonl` Audit (`223` Checks)**:
   - Enforced programmatically by [`gpu_gate.py`](file:///D:/work/中检院/v33_upgrade/models_gpu/gpu_gate.py):
     - **`219 / 223` checks** found no eligible GPU → **`selected_gpus = []`** (correctly refused to launch on `node1`).
     - **`4 / 223` checks** passed the strict 2-sample gate (`interval = 6.0 s`), every single one selecting **1 GPU** (well below `max_2_gpus`) with **both samples strictly `< 15%` VRAM and `≤ 32%` Util**:
       1. `preflight_check` (`21:47:32 → 21:47:38`): GPU 6 — `0.07%` VRAM (`18 MiB`), `0%` Util (both samples)
       2. `ncaa_repdistill_v2` (`21:48:44 → 21:48:51`): GPU 6 — `0.07%` VRAM (`18 MiB`), `0%` Util (both samples)
       3. `chemberta_mtr` (`21:54:21 → 21:54:27`): GPU 5 — `12.09%` VRAM (`2,971 MiB`), `20%` Util (both samples) → extracted lightweight 77M `chemberta_mtr.npz` in `315.65 s` on 1 GPU
       4. `ncaa_repdistill_v2` (`23:03:35 → 23:03:42`): GPU 5 — `14.84%` VRAM (`3,645 MiB`), `32% → 22%` Util
   - **Zero unauthorized `node1` GPU jobs were ever launched.**
3. **Hardware Offload Verification Across All Other Waves**:
   - Heavy foundation-model extractions (`ncaa_repdistill_v2`, `peptideclm2`, `esmc_600m_multiregion` verified via local `.meta.json` files) and all Wave-2D-2 deep multi-task + half-life specialist models (verified via [`gpu_deep_multitask_benchmark.md`](file:///D:/work/中检院/v33_upgrade/reports/gpu_deep_multitask_benchmark.md) §6) ran exclusively on the idle local Windows **RTX 5080**.
   - Wave-2C CPU models, Wave-2D-1 Multimodal Fusion, Wave-3A Champion Bundling, and the live V33 inference engine (`node23:8902` + `esmc_worker.py` on `127.0.0.1:18902`) executed 100% on `node23` CPUs (**0 GPU usage**).

---

## Honest Disclosure of Remaining Limitations & Scientific Boundaries

1. **Small-`n` Pharmacokinetic Endpoints**: `CL_absolute` (`n=54`, `11` HO rows across 2 holdout clusters, `cov80 = 63.6%`), `Cmax` (`n=74`, `17` HO rows), `F` (`n=124`, `25` HO rows), `PPB` (`n=162`, `32` HO rows / `24` exact, `cov80 = 29.2%`, `cov90 = 91.7%`), and `Vd_per_kg` (`n=166`, `33` HO rows) suffer from finite-sample cluster-shift variance. Predictions should be interpreted strictly for early-stage rank ordering alongside their 80%/90% conformal intervals and exact Tanimoto AD tiers — never for clinical/regulatory dose selection (`internal_research_only`).
2. **Hard Cluster Shift on `MIC_P_aeruginosa`**: Untouched holdout Spearman is **`0.133`** (`MAE = 0.657`, `cov80 = 75.6%`) despite grouped CV Spearman `0.603`, reflecting severe out-of-cluster shift on Gram-negative *P. aeruginosa* outer-membrane activity. Users must rely heavily on AD similarity and Top-3 analogs for this endpoint.
3. **Live Production AD Index vs. Leave-Case17-Out Distinction**: The Leave-Case17-Out scorecard (`MAE 0.3306 / Sp 0.9029`) comes from dedicated zero-leakage refits excluding all `439` overlap rows. The live production service (`node23:8902`) intentionally serves full-corpus 5-fold models and the full AD analog index, so FDA reference drugs like Semaglutide legitimately hit `Tc = 1.00` against their own curated record in live serving.
4. **CPU Inference Latency**: Live `dual_modality` cold inference takes **`~18–27 s`** on `node23` CPUs (vs **`~6.5–7.1 s`** for `smiles_only`) due to CPU ESM-C 600M encoding + multi-family ensemble inference. Repeat queries are served from the `node2` LRU+TTL cache in `< 0.3 s`.
5. **Pending `sudo` Step for Public HTTPS Staging (`/peptide-next/`)**: Staging is currently live and verified on `node2` loopback `127.0.0.1:7863`. Exposing `https://lmmd.ecust.edu.cn/peptide-next/` publicly requires a root/`sudo` operator to install `/var/www/peptide-omnipanel/v33_next/deploy/peptide-omnipanel-v33.service`, include `deploy/peptide-next.location.conf` in Nginx, and run `sudo nginx -t && sudo systemctl reload nginx` (exact commands and zero-downtime rollback documented in [`cutover_notes.md`](file:///D:/work/中检院/v33_upgrade/web/deploy/cutover_notes.md)).

---

## Final Staging Sign-Off

All **6 independent verification gates** (**Data & Zero-Leakage**, **Scientific Benchmark Truthfulness**, **Live E2E Pipeline**, **Visual Quality**, **Production Isolation**, and **GPU Resource Governance**) **PASSED** with zero blocking defects and zero production disruption.

**Recommendation: GO — Approved for staged `internal_research_only` rollout (`node2:7863` / `/peptide-next/`).**
