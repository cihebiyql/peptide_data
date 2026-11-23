# Peptide OmniPanel V33 — Wave 4 Self-Contained Migration onto `node2`

**Date**: 2026-11-22  
**Target Host**: `node2` (`<APP_HOST>`, user `mode2`, 128 CPU cores, 251 GiB RAM, 474 GiB SSD free)  
**Public URL**: `https://lmmd.ecust.edu.cn/peptide-omnipanel/`  
**Runtime Dependency on `node23` / `node1`**: **ZERO (100% self-contained on `node2`)**  
**GPU VRAM Used by V33 on `node2`**: **0 MiB** (`CUDA_VISIBLE_DEVICES=""`)  
**`sudo` Used**: **None** (Nginx `/etc/nginx/`, V31 `:7861`, and `omnigrid_v32_backup` completely untouched)

---

## 1. Self-Contained `node2` Directory Layout (`17 GB` total)

Everything lives under `/var/www/peptide-omnipanel/v33_next/` (owned by `mode2:mode2`):

| Path on `node2` | Size | Contents |
|---|---:|---|
| `/var/www/peptide-omnipanel/v33_next/v33_runtime/` | `13 GB` | `bundle/` (43 pruned `.joblib` serving models, `v33_manifest.json` with 0 `<WORK_DIR>` strings, `ad_analog_index.npz`, `analog_lookup.json.gz`, `v32_reference_snapshot.json`, portable `v33_inference_service.py`, `v33_live_encoders.py`, `esmc_worker.py`, `verify_v33_bundle_and_service.py`), `features/` (59 MB), `embeddings/` (717 MB), `data/` (`compounds_v33.tsv`, `case17_queries.tsv`), `models_cpu/` + `models_fusion/` + `models_gpu_deep/` (Python class modules + 110 manifest-referenced `.pt` checkpoints) |
| `/var/www/peptide-omnipanel/v33_next/pretrained/` | `3.8 GB` | `ncaadistill/` (1.5 GB `best.pt`, config, vocab SHA-256 `7270675a...`, model modules), `peptideclm2/` (121 MB), `chemberta_mtr/` (14 MB, dereferenced from HF cache symlinks), `esmc_600m/` (2.2 GB `biohub-ESMC-600M@fcc9d36`) |
| `/var/www/peptide-omnipanel/v33_next/extra_site/` | `238 MB` | Isolated `xgboost==3.2.0` (+ `xgboost.libs/libgomp`) and `lightgbm==4.7.0` overlaid via `PYTHONPATH` so V31's `/var/www/peptide-omnipanel/app/.venv` remains untouched |
| `/var/www/peptide-omnipanel/v33_next/web/` | `524 KB` | FastAPI + static SPA with auto-reconnecting `adapter.py` and `app.py` |
| `/var/www/peptide-omnipanel/v33_next/run_local_engine.sh` | `1 KB` | Detached launcher for local `v33_inference_service.py` (`127.0.0.1:8902`, `esmc_worker` on `127.0.0.1:18902`, `EP_WORKERS=16`, `CUDA_VISIBLE_DEVICES=""`) |
| `/var/www/peptide-omnipanel/omnigrid/app.py` | `4 KB` | Self-healing entrypoint invoked by existing `systemd` unit `peptide-omnigrid.service` (`127.0.0.1:7862`): checks/auto-launches `run_local_engine.sh` with `fcntl.flock` idempotency across future `node2` reboots |

### Key Engineering Adaptations
1. **Standalone Pure-PyTorch ESM-C 600M (`esmc_worker.py`)**: Because `node2`'s `.venv` has `transformers 4.57.6` (which predates native `model_type="esmc"` in `transformers 5.x`) and `modeling_esmc_remote.py` requires `esm>=3.4.1`, `esmc_worker.py` now implements the exact 36-layer RoPE + QK-LayerNorm + SwiGLU `StandaloneEsmcModel` + `ESMC_VOCAB` tokenizer directly on `torch 2.6.0` + `safetensors` (`strict=True` weight loading, 0 missing/unexpected keys). Verified against cached `esmc_600m_multiregion.npz` with **cosine 0.99995–0.99998** across all 4 pooling regions (`mean`, `max`, `nterm5`, `cterm5`) including padded multi-length batches.
2. **100% Path Portability**: Updated `v33_inference_service.py`, `v33_live_encoders.py`, `esmc_worker.py`, `verify_v33_bundle_and_service.py`, `v33_config.py`, `fusion_config.py`, and `deep_data.py` to honor environment variables (`V33_BASE_DIR`, `V33_ESMC_PYTHON`, `V33_ESMC_MODEL_DIR`, `NCAA_REPO_DIR`, `V33_PEPTIDECLM2_DIR`, `V33_CHEMBERTA_DIR`, `V33_EP_WORKERS=16`). Rewrote `v33_manifest.json` on `node2` so `grep -c "<WORK_DIR>"` returns **`0`** across `v33_manifest.json`, `run_local_engine.sh`, and `omnigrid/app.py`.
3. **Self-Healing Reboot Recovery Without `sudo`**: `omnigrid/app.py` (started by `systemd` `peptide-omnigrid.service`) idempotently launches `run_local_engine.sh` detached if `127.0.0.1:8902` is not listening; `web/adapter.py` and `web/app.py` automatically call `refresh_remote_health()` on `/api/v1/health`, `/api/v1/endpoints`, `/api/v1/predict`, and `/api/v1/batch` so the web layer seamlessly switches to `v33_remote` as soon as the local engine finishes warming up.

---

## 2. Decommissioning & Runtime Ownership Audit

| Host | Verification Result |
|---|---|
| **`node1`** | **0** V33 tunnel processes (`v33_tunnel.sh` and `ssh -R 127.0.0.1:8902:10.0.0.3:8902` terminated; no V33 crontab). Legacy V32 `:8901` tunnel (`omnigrid_tunnel.sh`, PID `2526506`) left untouched as requested. |
| **`node23`** | **0** V33 processes (`v33_inference_service.py` and `esmc_worker.py` terminated; TCP `:8902` and `:18902` closed). Training archive `<WORK_DIR>/v33_upgrade/` left untouched read-only. |
| **`node2`** | All listening ports owned strictly by local `mode2` Python processes:<br>• `127.0.0.1:7862` — PID `3456260` (`peptide-omnigrid.service`, RSS `121 MB`)<br>• `127.0.0.1:8902` — PID `3455227` (`v33_inference_service.py`, RSS `23.2 GiB` / `23,755 MB`)<br>• `127.0.0.1:18902` — PID `3455323` (`esmc_worker.py`, RSS `2.51 GiB`)<br>• `127.0.0.1:7861` — PID `2362` (legacy V31 service, untouched)<br>• RAM available: **187 GiB / 251 GiB** free<br>• GPU VRAM: GPU 0 `15173 MiB` / GPU 1 `7050 MiB` (unchanged; **0 MiB** used by V33 PIDs) |

---

## 3. Local `node2` Parity & Benchmark Verification (`verify_v33_bundle_and_service.py`)

Executed directly on `node2` against `http://127.0.0.1:8902` (`ALL 6 VERIFICATION SUITES PASSED in 128.55s`, receipt saved on `node2:/var/www/peptide-omnipanel/v33_next/v33_runtime/bundle/verification_receipt.json`):

- **Cold Engine Boot Time (local SSD)**: **104.22 s** (`esmc_worker` 6.5 s, 46,598-compound bank 6.83 s, ncAA-RepDistill v2 2.48 s, PeptideCLM-2 1.68 s, ChemBERTa-MTR 0.10 s, FastADAnalogEngine 0.39 s, 43 `.joblib` + 110 `.pt` checkpoints 85.3 s).
- **Manifest & Integrity**: 24/24 endpoints, 0 NaN/Inf, all 43 serving `.joblib` + 3 index SHA-256 checksums verified.

### Cached vs. Live-Encoded Cosine Parity (`force_live_encode=True`)

| Test Case | Track | Cached (ms) | Live (ms) | `ncAA` min cos | `PeptideCLM-2` cos | `ChemBERTa` cos | `ESM-C 600M` (4-reg min) | Max 24-ep Pred Diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Semaglutide` | `smiles_only` | 3,415 | 4,601 | `0.999042` | `0.999998` | `0.999990` | N/A | `0.01071` |
| `Tirzepatide` | `smiles_only` | 3,315 | 6,689 | `0.996829` | `0.999999` | `0.999988` | N/A | `0.03587` |
| `Cyclosporine_A` | `smiles_only` | 3,404 | 3,763 | `0.999481` | `0.999998` | `0.999980` | N/A | `0.00950` |
| `Bank_AMP (MKTK...LEL)` | `dual_modality` | 10,737 | 12,209 | `0.998971` | `0.999999` | `0.999988` | `0.999952` | `0.01089` |
| `AMP (HISE...SNC)` | `dual_modality` | 10,785 | 11,974 | `1.000000` | `1.000000` | `1.000000` | `1.000000` | `0.00000` |

### Unseen Synthetic Peptides (True Out-of-Cache Live Encoding, `cache_hit=False`) & Batch

| Test Case | Track | Encode (ms) | Predict (ms) | AD (ms) | Total (ms) | Sample Outputs |
|---|---|---:|---:|---:|---:|---|
| `Unseen_AMP_KLLKLLKKLLKLLK` | `dual_modality` | 879.4 (SMILES 471.9 + ESM-C 407.3) | 11,058.2 | 6.1 | 11,959.3 | `T½=6.13h`, `PPB=0.559`, `Hemol=0.706`, `MIC_E_coli=6.64 µM` |
| `Unseen_AMP_Fresh_KLLKLLKKLLKLLKWW` | `dual_modality` | 820.4 (SMILES 560.4 + ESM-C 259.9) | 10,376.9 | 6.2 | 11,220.1 | `T½=3.78h`, `PPB=0.667`, `Hemol=0.953`, `MIC_E_coli=4.40 µM` |
| `Unseen_C18Diacid_NMe_Tetrapeptide` | `smiles_only` | 312.7 | 3,271.5 | 8.4 | 3,607.2 | `T½=2.83h` (`weekly_diacid` anchor `120–168h`), `PPB=0.941`, `Hemol=0.336` |
| Condition Routing (`Semaglutide`) | `smiles_only` | — | — | — | — | `IV=21.57h` vs `SC=51.68h` |
| `POST /batch` (5 diverse molecules) | mixed | 0.4 | 15,818.0 | 33.6 | 15,916.7 | `5/5 succeeded`, 0 NaN/Inf |

---

## 4. External Public HTTPS Verification (`https://lmmd.ecust.edu.cn/peptide-omnipanel/`)

Verified end-to-end from Windows workstation after `node1`/`node23` V33 teardown:

| Endpoint | HTTP | Latency | Verified Result |
|---|---:|---:|---|
| `GET /` | `200` | 234 ms | Full SPA HTML (`32,028 B`) |
| `GET /api/v1/health` | `200` | 144 ms | `adapter_mode="v33_remote"`, `v33_remote_url="http://127.0.0.1:8902"`, `v33_remote_healthy=true`, `endpoints_total=24`, `manifest_path="/var/www/peptide-omnipanel/v33_next/v33_runtime/bundle/v33_manifest.json"` |
| `GET /api/v1/endpoints` | `200` | 205 ms | `24` endpoints (`20` Tier-1 + `4` Tier-2), `adapter_mode="v33_remote"` |
| `GET /docs` | `200` | 159 ms | Swagger OpenAPI UI |
| `POST /api/v1/predict` (`Semaglutide`) | `200` | 5,375 ms | `24/24` `v33_remote` predictions, `weekly_diacid` anchor `[120, 168]h`, `T½=21.57h`, `PPB=79.8%`, `Hemol=9.5%`, 0 NaN/Inf |
| `POST /api/v1/predict` (`GIGKFLKKAKKFGKAFVKILKKAA`) | `200` | 17,165 ms | True cold unseen live encode (`smiles_live=978 ms`, `esmc_live=916 ms`, `ad=9.5 ms`), `dual_modality`, `T½=0.882h`, `MIC_E_coli=3.07 µM`, 0 NaN/Inf |
| `POST /api/v1/predict` (invalid string) | `400` | 137 ms | Clean 400 validation error |
| `POST /api/v1/batch` (3 molecules) | `200` | 19,742 ms | `succeeded=3/3`, `failed=0`, `adapter_mode="v33_remote"`, `24/24` endpoints each |
