# Peptide OmniPanel V33 — Scientific Web Platform Architecture, Live Integration & Staging Audit Report

**Date**: 2026-11-22  
**Scope**: `internal_research_only`  
**Status**: Wave 3B Complete — Live `v33_remote` engine (`node23:8902` -> `node1` reverse tunnel -> `node2:7863`) staged safely with zero impact on live production (`:7861` / `:7862` / `:8901`), all 4 Wave-1 audit findings resolved, 11/11 pytest passing, and 5/5 live Playwright visual audit screenshots verified.

---

## 1. Executive Summary

Wave 3B replaced all legacy prototype/synthetic UI metrics with a 100% authoritative integration of the **Peptide OmniPanel V33 4-Foundation-Model Super-Ensemble** (`v33_manifest.json` + live `node23:8902` HTTP inference engine) across the FastAPI backend and zero-build ES-module SPA frontend:

- **Zero Fabricated Metrics (Wave-1 Audit #1 Fixed)**: Removed all hardcoded V32 corpus counts (`58,932`, `32`, `213,800`) and synthetic CV lifts from both `adapter.py` and `static/index.html`. Every landing KPI (`46,597` curated peptides, `24` tiered endpoints [`20 + 4`], `69,111` harmonized records [`2,523` censored]), every validation table entry (`Train/CV N`, `Holdout N`, `Censored`, `Winning Family & Weights`, `V32 Ref`, `V33 Grouped CV`, `V33 20% Holdout`, `80% Conformal Coverage` / `Holdout Brier & ECE`), the **Strict Leave-Case17-Out Clinical Benchmark** card, and the **4-Foundation-Model Representation Dominance** strip are dynamically hydrated from `v33_manifest.json`.
- **All 3 Remaining Wave-1 Visual/UX Findings + Subpath Safety Fixed**:
  - **Audit #2 (Radar Label Clipping)**: Widened single-molecule radar SVG `viewBox` to `396×256` (`cx=198, cy=128, R=74`) and batch overlay radar `viewBox` to `408×278` (`cx=204, cy=139, R=82`) in `static/charts.js`. Live Playwright bounding-box checks confirm `≥28 px` right margin on `Plasma Binding` and `≥44 px` left margin on `Lipophilicity` across both views.
  - **Audit #3 (Batch Scientific Notation Wrapping)**: Added `white-space: nowrap; font-variant-numeric: tabular-nums;` to `.data-table td.num`, `.data-table td.mono`, `.data-table td.nowrap`, and `.sortable-table td .ep-risk` and rebalanced `.batch-analytics-grid` to `8.3fr 3.7fr`. Values such as `120–168 h ★`, `4.41e-07`, and `1.72e-07` render cleanly on a single line.
  - **Audit #4 (Home Example Chips Wrapping)**: Widened `.predict-hero.hero-centered` to `1040px` (keeping `.query-bar-wrap` centered at `820px`) and tightened `.example-pill` padding/gaps so all 5 quick-example chips (`Semaglutide`, `Tirzepatide`, `Cyclosporine A`, `Daptomycin`, `Classic AMP`) and `Advanced Options` align on one balanced horizontal row at `1440×900` (`top = 331–332 px`, delta `1 px`).
  - **Subpath-Safe Mounting**: Switched `index.html` asset & OpenAPI links to relative paths (`static/styles.css`, `static/app.js`, `docs`, `redoc`), computed `API_BASE` dynamically in `static/app.js`, and added `V33_ROOT_PATH` support in `app.py` and `run_server.py`. The same build operates cleanly at both `/` (`http://127.0.0.1:7863/`) and behind Nginx at `/peptide-next/`.
- **Live Conformal CI Precision Catch During Visual Audit**: Visual inspection of the first live Semaglutide screenshot caught `0.00e+00 – 0.00e+00` on `cm/s` permeability endpoints (`PAMPA`, `Caco2`, `MDCK`, `RRCK`) caused by upstream `round(10**x, 4)` rounding `~1e-7` to zero in `ci80_linear`. `adapter.py` now computes exact 4-significant-figure linear intervals directly from `log10` `ci80_low`/`ci80_high` and `ci90_low`/`ci90_high`, guarded by a dedicated pytest regression assertion.

---

## 2. Unified 24-Endpoint V33 Taxonomy & Runtime Modes

### 2.1 Exact 24 Unified V33 Endpoints (`20 Tier-1 + 4 Tier-2`)

| Tier | Domain | Count | Endpoints |
| :--- | :--- | :---: | :--- |
| **Tier 1** | Pharmacokinetics & PK | 8 | `half_life_reg`, `half_life_cls`, `PPB`, `CL_per_kg`, `CL_absolute`, `Vd_per_kg`, `F`, `Cmax` |
| **Tier 1** | Membrane Permeability & Absorption | 5 | `PAMPA`, `Caco2`, `MDCK`, `RRCK`, `BBB` |
| **Tier 1** | Safety & Hemolysis | 4 | `hemolysis_cls`, `hemolysis_HC50`, `overall_peptide_toxicity`, `cytotoxicity_cls` |
| **Tier 1** | Physicochemical & In Vitro | 3 | `LogD7.4`, `solubility_reg`, `solubility_cls` |
| **Tier 2** | Reference-Strain Antimicrobial MICs | 4 | `MIC_E_coli`, `MIC_S_aureus`, `MIC_P_aeruginosa`, `MIC_C_albicans` |

### 2.2 Clean Adapter Modes (`web/adapter.py`)

| Mode | Trigger | Behavior |
| :--- | :--- | :--- |
| **`v33_remote`** *(Primary Production)* | `V33_REMOTE_URL` env var (or auto-detected `http://127.0.0.1:8902` when healthy) | Forwards `/predict` (`120s` timeout) and batched `/batch` (`300s` timeout) directly to live `node23:8902`. Preserves calibrated probabilities, 80%/90% Conformal intervals, censored AFT/Tobit diagnostics, exact Morgan Tanimoto AD & Top-3 training analogs, assay condition routing (`rbc_species`, `pk_route`, `pk_species`, `pk_matrix`), and fatty-acylation anchor. Local RDKit renders 2D motif-highlighted SVGs. Thread-safe in-memory LRU+TTL cache (`256` entries, `3600s` TTL) serves repeat queries in `<250 ms`. |
| **`v33_offline_fallback`** *(Offline Unit-Test Mode)* | `V33_OFFLINE_ONLY=1` or unreachable remote engine | Loads 100% real `v33_manifest.json` catalog, Case17 Leave-Case17-Out predictions, and conformal `q80/q90` quantiles; explicitly tags non-Case17 fallback estimates as `offline_fallback` and displays an amber fallback pill in the UI. Zero fabricated CV metrics. |

---

## 3. Safe Parallel Staging Topology (`node1` + `node2`)

```text
[Browser / Nginx Staging]
         │  (Ready-to-apply /peptide-next/ snippet)
         ▼
[node2: 127.0.0.1:7863]  FastAPI + Uvicorn + RDKit 2026.03.6 (/var/www/peptide-omnipanel/v33_next/)
         │  V33_REMOTE_URL=http://127.0.0.1:8902
         │  (Parallel to UNTOUCHED live V31 :7861, V32 :7862, V32 tunnel :8901)
         ▼
[node1 reverse tunnel]   /data/qlyu/v33_tunnel.sh  (-R 127.0.0.1:8902:10.0.0.3:8902)
         │  (Parallel to UNTOUCHED /data/qlyu/omnigrid_tunnel.sh on :8901)
         ▼
[node23: 10.0.0.3:8902]  Live V33 Super-Ensemble Engine (4 warm FMs, 24 endpoints, 0 GPU)
```

### Staging Isolation Verification
- **Live V31 (`node2:7861`)**: Verified `HTTP 200`, untouched.
- **Live V32 (`node2:7862`)**: Verified `HTTP 200`, untouched.
- **Live V32 Reverse Tunnel (`node2:8901` via `node1:/data/qlyu/omnigrid_tunnel.sh`)**: Verified `HTTP 200` (`models_cached: 152, endpoints: 55`), untouched.
- **New V33 Reverse Tunnel (`node2:8902` via `node1:/data/qlyu/v33_tunnel.sh`)**: Verified `HTTP 200` (`version: "v33.0", mode: "v33_live", endpoints: 24`).
- **New V33 Staged Web Platform (`node2:7863` under `/var/www/peptide-omnipanel/v33_next/`)**: Verified `HTTP 200` (`adapter_mode: "v33_remote", endpoints_total: 24, tier1_endpoints: 20, tier2_endpoints: 4, rdkit_version: "2026.03.6"`).
- **Ready-to-Apply Deployment Templates** (mirrored in both `D:\work\中检院\v33_upgrade\web\deploy\` and `node2:/var/www/peptide-omnipanel/v33_next/deploy/`):
  - `peptide-next.location.conf` — Nginx location snippet exposing `/peptide-next/ -> http://127.0.0.1:7863/`.
  - `peptide-omnipanel-v33.service` — `systemd` unit running as `mode2` with `V33_ROOT_PATH=/peptide-next`.
  - `cutover_notes.md` — Exact staging enablement, production cutover, and instant rollback commands for root/sudo.

---

## 4. Verification & Live Smoke Test Evidence

### 4.1 Local Unit Tests (`py -3.11 -m pytest`)
- Command: `py -3.11 -m pytest -v D:\work\中检院\v33_upgrade\web\tests\test_web_api.py`
- Result: **11/11 PASSED** (`1.83 s`), verifying:
  - Exact 24 unified endpoints (`20 Tier-1 + 4 Tier-2`) and real `v33_manifest.json` KPIs (`46,597`, `24`, `69,111`, `2,523`).
  - Zero fabricated metrics (exact equality with `v33_manifest.json`, `None` V32 ref for new endpoints, exact Case17 scorecard & 4-FM ablation wins).
  - Non-zero, scientifically formatted `PAMPA` 80% Conformal CI (`0.00e+00` regression guard).
  - RDKit 2D motif SVG highlighting (fatty diacid, disulfide, macrocycle, N-methylation), dual-input condition routing, invalid inputs returning clean `HTTP 400`, batch CSV/FASTA/multipart upload, and subpath-safe relative HTML asset references.

### 4.2 Live End-to-End Smoke Tests (`node2:7863 -> node23:8902`)

| Query | Track | Runtime (cold) | Pure-ML T½ | Mechanism Anchor | PPB | Hemolysis Risk (`hemolysis_cls`) | Hemolysis HC50 | *E. coli* MIC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Semaglutide** (`human_sc`) | `smiles_only` | `9.59 s` | `51.68 h` | `120–168 h ★` (C18 diacid) | `79.8%` | `9.5%` *(Low Risk)* | `49.97 µM` | `15.83 µM` |
| **Cyclosporine A** | `smiles_only` | `8.98 s` | `8.64 h` *(FDA 8.4 h)* | — | `89.6%` *(FDA ~90%)* | `29.6%` *(Moderate)* | `52.33 µM` | `26.74 µM` |
| **Classic AMP** (`HISEAVKIFGCKWKENWKWNSNC`) | `dual_modality` | `25.67 s` | `3.39 h` | — | `54.9%` | `12.2%` *(Low Risk)* | `106.1 µM` | `20.31 µM` |
| **Invalid Input (`123@@@`)** | — | `<5 ms` | — | — | — | **HTTP 400** (clean validation error) | — | — |

---

## 5. Live Playwright Visual Audit (`1440×900`, Chromium, Live V33 Engine)

All 5 screenshots were captured directly against the live staged `node2:7863 -> node23:8902` pipeline with **0 JS console errors** and **0 uncaught page errors**, and visually inspected via multimodal image review:

1. **Home Landing (`01_home_v33.png`)**: All 5 example chips + `Advanced Options` sit cleanly on one single row (`top = 331–332 px`); KPI ribbon displays real audited V33 numbers (`46,597`, `24 [20 + 4]`, `69,111 [2,523 Censored]`, `80%`); header shows green `v33_remote · 24 EPs`.
2. **Semaglutide Live Prediction (`02_semaglutide_v33.png`)**: Crisp RDKit 2D structure with amber C18 fatty-diacid highlighting; unclipped 6-axis radar (`Plasma Binding` right margin `29 px`); C18 fatty-diacid mechanism anchor banner paired beside pure-ML SC `51.7 h`; all 20 Tier-1 endpoints with exact non-zero Conformal CI bars (regression) and calibrated probability bars (classification); expanded `half_life_reg` and `hemolysis_HC50` drawers showing Champion family & weights, Censored AFT/Tobit diagnostic values, and real Top-3 nearest training analogs with censor tags (`RIGHT`).
3. **Classic AMP Sequence + Tier-2 MIC Panel (`03_amp_seq_v33.png`)**: Live `DUAL TRACK` inference on 23-AA sequence `HISEAVKIFGCKWKENWKWNSNC`; Tier-2 tab shows all 4 reference-strain MICs (`E. coli`, `S. aureus`, `P. aeruginosa`, `C. albicans`), expanded `MIC_E_coli` drawer with `Super-Ensemble (4-FM Fusion 54% · CPU GBDT 27% · Deep Multi-Task 19%)`, `Censored AFT/Tobit Diagnostic: 20.08 uM`, real Top-3 training analogs, and the paired **Selectivity & Strain Benchmark** (`HC50 / Best MIC = 6.8×`).
4. **Live Batch Screening (`04_batch_v33.png`)**: Live 7-peptide clinical & AMP batch completed via single remote `/batch` call (`18.3 s`), showing single-line non-wrapping scientific notation (`4.41e-07`, `1.72e-07`, `120–168 h ★`), real V33 low-risk hemolysis probabilities across all FDA peptides (`9.5%–29.6%` vs V32's false positives), and an unclipped 3-peptide radar overlay.
5. **Endpoints & Validation Transparency (`05_validation_v33.png`)**: Hydrated 100% from `v33_manifest.json` showing the 5 real top KPIs, **Strict Leave-Case17-Out Clinical Benchmark** (`Pure-ML T½ ρ = 0.903, R² = 0.763, MAE = 0.331`; `T½ + Anchor MAE = 0.299`; `PPB ρ = 0.709`; `0 / 17 FP` vs V32 `8 / 17`), **4-Foundation-Model Representation Dominance** (`ncAA-RepDistill v2: 9/24`, `ESM-C 600M: 9/24`, `PeptideCLM-2: 4/24`, `ChemBERTa-MTR: 2/24`), and all 24 audited endpoints with exact `Train/CV N`, `Holdout N`, `Censored`, `Winning Family & Weights`, `V32 Ref`, `V33 Grouped CV`, `V33 20% Holdout`, and `80% Cov. / Cal.` (`Brier · ECE` for classifiers).

### Artifact Paths
- [01_home_v33.png](file:///D:/work/中检院/v33_upgrade/reports/screenshots/01_home_v33.png)
- [02_semaglutide_v33.png](file:///D:/work/中检院/v33_upgrade/reports/screenshots/02_semaglutide_v33.png)
- [03_amp_seq_v33.png](file:///D:/work/中检院/v33_upgrade/reports/screenshots/03_amp_seq_v33.png)
- [04_batch_v33.png](file:///D:/work/中检院/v33_upgrade/reports/screenshots/04_batch_v33.png)
- [05_validation_v33.png](file:///D:/work/中检院/v33_upgrade/reports/screenshots/05_validation_v33.png)
