# Peptide OmniPanel V33.1 — Wave 5B Self-Contained Deployment on `node2` & External Public HTTPS Verification

**Date**: 2026-11-22  
**Release**: **Peptide OmniPanel V33.1** (**46 endpoints** = 24 Tier 1 + 22 Tier 2; **50,995** unified compounds / **50,994** unique curated compounds; **82,327** harmonized rows; **4,535** censored-aware records; **25** curated clinical & research examples incl. 17/17 Case17 + 4/4 FDA Echinocandins + 5 AMPs)  
**Target Host**: `node2` (`<APP_HOST>`, user `mode2`, 128 CPU cores, 251 GiB RAM, 879 GiB SSD)  
**Public URL**: `https://lmmd.ecust.edu.cn/peptide-omnipanel/`  
**Runtime Dependency on `node23` / `node1`**: **ZERO (100% self-contained on `node2`)**  
**GPU VRAM Used by V33.1 on `node2`**: **0 MiB** (`CUDA_VISIBLE_DEVICES=""`)  
**`sudo` Used**: **None** (`systemd` respawn via `mode2`-owned `peptide-omnigrid.service` uvicorn PID)

---

## 1. Portability Fix & Delta Deployment Summary

### 1.1 Unified Portable `v33_inference_service.py`
Wave 5A's `v33_inference_service.py` (copied back from `node23`) had reverted Wave 4's env-var portability (`BASE_DIR`, `ESMC_PYTHON`, `V33_ESMC_MODEL_DIR`, `V33_EP_WORKERS`, `_resolve_deep_ckpt_path`). We merged Wave 4's portability layer and Wave 5A's 46-endpoint V33.1 serving logic into a single portable `v33_inference_service.py`:
- Preserves `V33_BASE_DIR`, `V33_ESMC_PYTHON`, `V33_ESMC_MODEL_DIR`, `V33_EP_WORKERS=16`, `V33_ESMC_THREADS=24`, and `_resolve_deep_ckpt_path()`.
- Preserves V33.1 `PK_ENDPOINTS` (`+ Vd_absolute`), `HEMOL_ENDPOINTS` (`+ hemolysis_MHC`), dynamic `version` from `v33_manifest.json` across `/health`, `/predict`, and `/batch`, `expansion_wave == "v33.1"` censored-head activation, and dynamic endpoint count in startup logging.
- Identical SHA-256 verified on both local Windows (`D:\work\中检院\v33_upgrade\bundle\v33_inference_service.py`) and `node2` (`/var/www/peptide-omnipanel/v33_next/v33_runtime/bundle/v33_inference_service.py`):
  `cbe7ee9a23cf979a07c20d5ab9daa494a8d8893a12ea5ea4248bb11e65eaceff`.

### 1.2 Manifest Path Rewrite & Pre-Flight Audit
- Rewrote all 111 `<WORK_DIR>/v33_upgrade` prefixes in `node2`'s `v33_manifest.json` cleanly to `/var/www/peptide-omnipanel/v33_next/v33_runtime` (updated both `/var/www/peptide-omnipanel/v33_next/v33_runtime/bundle/v33_manifest.json` and `/var/www/peptide-omnipanel/v33_next/v33_manifest.json`).
- `grep -c "<WORK_DIR>"` returns **`0`** across `v33_runtime/bundle/v33_manifest.json`, `v33_next/v33_manifest.json`, `run_local_engine.sh`, and `/var/www/peptide-omnipanel/omnigrid/app.py`.
- Pre-flight check on `node2` verified **0 missing files** across all **84 `.joblib` bundles** (`42` CPU + `42` Fusion) and **110 `.pt` deep checkpoints**.

### 1.3 Delta Assets Synced onto `node2` (`19G` total under `/var/www/peptide-omnipanel/v33_next/`)
Existing Wave 4 assets (`pretrained/` 3.8 GB, `extra_site/` 238 MB, `models_cpu/`, `models_fusion/`, `models_gpu_deep/` 1.5 GB, `esmc_worker.py`, `v33_live_encoders.py`, `run_local_engine.sh`, `omnigrid/app.py`, and original 43 `.joblib` files) were preserved untouched; only the V33.1 delta was synced:

| Path on `node2` | Size | Contents / V33.1 Delta |
|---|---:|---|
| `v33_runtime/bundle/serving_models/` | `13 GB` | **84 `.joblib` bundles** (`43` original V33.0 + `41` new V33.1 models streamed via LAN `tar`) |
| `v33_runtime/bundle/` | `8.4 MB` (indices+code) | Updated `ad_analog_index.npz` (`4.17 MB`), `analog_lookup.json.gz` (`4.19 MB`), rewritten `v33_manifest.json` (`852 KB`), unified portable `v33_inference_service.py` |
| `v33_runtime/features/` | `64 MB` | Atomic-swapped **50,995-row** feature bank (`compound_index.tsv`, `fp_compact.npz`, `morgan_bits_1024.npz`, `physchem_core.npz`) |
| `v33_runtime/embeddings/` | `782 MB` | Atomic-swapped **50,995-row** embedding bank (`chemberta_mtr.npz`, `esmc_600m_multiregion.npz`, `ncaa_repdistill_v2.npz`, `peptideclm2.npz`) |
| `v33_runtime/data/` | `33 MB` | Updated `compounds_v33.tsv` (`50,994` rows from `compounds_v331_merged.tsv`, 100% aligned with `compound_index.tsv` + `case17_queries.tsv`) |
| `v33_next/web/` | `568 KB` | Upgraded V33.1 FastAPI backend + SPA frontend (`25` curated examples, `46` endpoints, Tier-2 sub-filters, `3` batch presets) |
| `v33_next/pretrained/` & `extra_site/` | `3.8 GB` + `238 MB` | Untouched Wave 4 local pretrained weights (ncAA-RepDistill v2, PeptideCLM-2, ChemBERTa-MTR, ESM-C 600M) & XGBoost/LightGBM overlay |

---

## 2. `node2` Runtime Health, Resource Footprint & Zero-`node23`/Zero-`node1` Proof

### 2.1 Engine Boot & Zero-`sudo` Service Respawn
- **V33.1 Inference Service (`127.0.0.1:8902`, PID `3490268`)**:
  - Cold startup on `node2` SSD: **`138.44 s`** (`esmc_worker` `7.0 s` [model load `5.17 s`], 50,995-compound bank `7.46 s` [`47,892` SMILES + `34,185` sequences cached], ncAA-RepDistill v2 `2.47 s`, PeptideCLM-2 `1.80 s`, ChemBERTa-MTR `0.14 s`, FastADAnalogEngine `50,994 refs × 46 endpoints` `0.38 s`, `84` `.joblib` + `110` `.pt` checkpoints `118.3 s`).
  - `/health`: `ok=true`, `version="v33.1"`, `mode="v33_live"`, `endpoints=46` (`24` Tier 1 + `22` Tier 2), `cpu_bundles=42`, `fusion_bundles=42`, `deep_endpoints=20`.
- **Standalone Pure-PyTorch ESM-C 600M Worker (`127.0.0.1:18902`, PID `3490282`)**:
  - `backend="standalone_pytorch"`, `dtype="float32"`, `ok=true`, live-encoded `2` unseen/non-cached sequences during verification.
- **FastAPI Web Platform (`127.0.0.1:7862`, PID `3491133`)**:
  - Zero-`sudo` `SIGKILL` of previous PID `3456260` triggered clean `systemd` respawn of `peptide-omnigrid.service` in **`11 s`** (`PID 3491133`).
  - `/api/v1/health`: `ok=true`, `version="33.1.0"`, `adapter_mode="v33_remote"`, `v33_remote_healthy=true`, `endpoints_total=46` (`24 + 22`), `n_unique_compounds=50994`, `n_total_rows=82327`, `n_censored_rows=4535`.

### 2.2 Host Ownership & Post-Workload Resource Audit

| Host | Live Runtime Audit |
|---|---|
| **`node2` (`<APP_HOST>`)** | • `127.0.0.1:8902` — PID `3490268` (`v33_inference_service.py`, RSS **`27,195.4 MB`** / `26.56 GiB` post-test; `26,151.5 MB` at boot)<br>• `127.0.0.1:18902` — PID `3490282` (`esmc_worker.py`, RSS **`2,579.8 MB`** / `2.52 GiB`)<br>• `127.0.0.1:7862` — PID `3491133` (`peptide-omnigrid.service`, RSS **`158.5 MB`**)<br>• System RAM available: **`185 GiB / 251 GiB`** free<br>• SSD free (`/var/www/peptide-omnipanel`): **`472 GiB / 879 GiB`** (`44%` used)<br>• GPU VRAM used by V33.1 PIDs (`3490268`, `3490282`, `3491133`): **`0 MiB`** (`nvidia-smi` confirmed zero V33 PIDs on any GPU) |
| **`node1`** | **`0`** V33 processes (`v33_inference|esmc_worker|v33_tunnel`), **`0`** `:8902`/`:18902` listeners |
| **`node23`** | **`0`** V33 processes (`v33_inference|esmc_worker|v33_tunnel`), **`0`** `:8902`/`:18902` listeners |

---

## 3. External Public HTTPS API Verification (`https://lmmd.ecust.edu.cn/peptide-omnipanel/`)

Executed end-to-end from local Windows against the public HTTPS endpoint:

| # | Endpoint / Case | HTTP | Latency | Verified Result |
|---|---|---:|---:|---|
| 1 | `GET /` | `200` | `200.0 ms` | Full V33.1 SPA HTML (`33,660 B`) |
| 2 | `GET /docs` | `200` | `130.8 ms` | Swagger OpenAPI UI (`1,060 B`) |
| 3 | `GET /api/v1/health` | `200` | `153.6 ms` | `ok=true`, `version="33.1.0"`, `adapter_mode="v33_remote"`, `46` EPs (`24+22`), `50,994` cpds, `82,327` rows, `4,535` censored |
| 4 | `GET /api/v1/endpoints` | `200` | `229.2 ms` | `46` endpoints (`24` T1 + `22` T2), all `22` new V33.1 EPs have finite CV & holdout scores, 0 NaN/Inf |
| 5 | `GET /api/v1/examples` | `200` | `152.0 ms` | `6` quick examples, `5` categories, `25` curated drugs (17 Case17 + 4 FDA Echinocandins + 5 AMPs), `3` presets (`demo_7`, `case17_17`, `echinocandin_6`) |
| 6 | `POST /api/v1/predict` — **Semaglutide** (`human_sc`) | `200` | `531.1 ms` *(10.4 s cold)* | `46/46` EPs, weekly diacid anchor `[120.0, 168.0] h`, `hemolysis_cls=9.5% (<50%)`, `efflux_ratio=2.7`, `cytotoxicity_CC50=38.09 µM`, `hemolysis_MHC=23.79 µM`, `Vd_absolute=13.06 L`, 2D SVG (`218 KB`), all CIs & Top-3 analogs finite |
| 7 | `POST /api/v1/predict` — **Micafungin** | `200` | `270.2 ms` | `46/46` EPs, macrocycle detected (`21`-ring), `MIC_C_albicans=10.98 µM`, `MFC_C_albicans=2.03 µM`, `PPB=86.8%`, `hemolysis_cls=5.7%`, 2D SVG (`75 KB`) |
| 8 | `POST /api/v1/predict` — **Anidulafungin** (`include_svg=false`) | `200` | `9,222.6 ms` | `46/46` EPs, `MIC_C_albicans=13.30 µM`, `MFC_C_albicans=4.27 µM`, `half_life_reg=28.64 h` |
| 9 | `POST /api/v1/predict` — **Rezafungin** (`include_svg=false`) | `200` | `8,466.9 ms` | `46/46` EPs, `MIC_C_albicans=12.37 µM`, `MFC_C_albicans=4.74 µM`, `half_life_reg=79.38 h` |
| 10 | `POST /api/v1/predict` — **Classic AMP** (`HISEAVKIFGCKWKENWKWNSNC`) | `200` | `27,880.9 ms` | `46/46` EPs, `22/22` Tier-2 finite (`MIC_E_coli=20.31 µM`, `MBC_E_coli=6.64 µM`, `MBIC=15.76 µM`, `MBEC=37.30 µM`, `LC=10.41 µM`) |
| 11 | `POST /api/v1/predict` — **Novel live-encoded** (`KWKLFKKIGAVLKVLTTGL`) | `200` | `25,646.3 ms` | `46/46` EPs, `cache_hit=false`, `dual_modality`, live encoding on `node2`: `smiles_live=570.4 ms` + `esmc_live=533.3 ms` = `1,103.9 ms`, `predict=24,260.0 ms`, `ad=13.1 ms` |
| 12 | `POST /api/v1/predict` — Invalid garbage input | `400` | `148.4 ms` | Clean HTTP `400` validation error |
| 13 | `POST /api/v1/batch` — **Case17 Clinical (17 drugs)** | `200` | `33,814.0 ms` | **`17/17` succeeded**, `0` failed, **`5/5` fatty-acylation anchors** (`Semaglutide`, `Tirzepatide`, `Liraglutide`, `Insulin detemir`, `Insulin degludec`), `46` EPs/drug |
| 14 | `POST /api/v1/batch` — **Echinocandins & Macrocycles (6 drugs)** | `200` | `361.2 ms` *(warm)* | **`6/6` succeeded**, `0` failed (`Caspofungin`, `Micafungin`, `Anidulafungin`, `Rezafungin`, `Cyclosporine A`, `Daptomycin`), `46` EPs/drug |

---

## 4. Headless Chromium Browser Verification (Playwright)

Verified interactively against `https://lmmd.ecust.edu.cn/peptide-omnipanel/` at `1440×960` (`device_scale_factor=1.5`):
- **Console errors**: **`0`**
- **Page / uncaught JS errors**: **`0`**
- **HTTP `>=400` network responses**: **`0`**
- **Status badge**: `v33_remote · 46 EPs` | KPI ribbon: `50,994` peptides, `46` endpoints (`24 + 22`), `82,327` records (`4,535` censored).

### Captured Full-Page Screenshots (Visually Inspected)
1. `D:\work\中检院\v33_upgrade\reports\screenshots\v331_01_examples_drawer.png` — Landing page + expanded **All 25 Examples** drawer across all 5 categories (`25` clickable drug/AMP chips, `C17` badges, 4/4 FDA Echinocandins).
2. `D:\work\中检院\v33_upgrade\reports\screenshots\v331_02_micafungin_tier1.png` — Live **Micafungin · 米卡芬净** prediction with RDKit 2D SVG (`21`-atom Macrocycle Ring highlight), FDA reference box (`14–17 h` vs predicted `14.71 h`), 6-axis radar, and 24-endpoint Tier-1 Core ADMET & PK 2×2 grid.
3. `D:\work\中检院\v33_upgrade\reports\screenshots\v331_03_tier2_eskape.png` — Tier-2 **Antimicrobial & Biofilm (`22`)** view with `4` sub-filter chips (`All (22)`, `MIC · ESKAPE (13)`, `MBC & Lethal (6)`, `Fungicidal & Biofilm (3)`), all `22` ESKAPE/MBC/LC/MFC/MBIC/MBEC rows, and Selectivity Index (`214×`) card.
4. `D:\work\中检院\v33_upgrade\reports\screenshots\v331_04_batch_case17.png` — **Batch** tab after running **Case17 Clinical (17 · 中检院案例)** preset (`17/17` succeeded in `14.4 s`, `5/5` acylation anchors badged, `0/17` hemolysis FP, 3-peptide multi-radar overlay).
5. `D:\work\中检院\v33_upgrade\reports\screenshots\v331_05_validation_46ep.png` — **Endpoints & Validation** tab hydrated live from `v33_manifest.json` showing all `46` endpoints, Leave-Case17-Out scorecard (`ρ = 0.903`, `MAE 0.299`, `0/17 FP`), and 4-FM dominance chart.
