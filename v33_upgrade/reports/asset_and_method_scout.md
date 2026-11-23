# Peptide OmniPanel V33 — Asset, Environment, Algorithm & GPU Scout Report

- **Audit Date / Time**: 2026-11-21 18:29–18:43 CST
- **Scope**: Read-only audit of `node1`, `node23`, `node2` (`<DEPLOY_USER>@<APP_HOST>`), and local Windows workspace `D:\work\中检院`. Zero existing files, models, configs, or services were modified.
- **Baseline Context**: V32 unified frozen feature bank (`D:\work\中检院\data\peptide_modeling_unified_features_20261114_v1`, fingerprint `ee232f37b6acb5c3`, 40,557 structures / 33,761 sequences / 16,940 dims across 13 independent methods: 10 RDKit 2D/graph families + AAC/physicochem + DPC + ESM2-650M mean-pool) and 8 tabular learners (`linear`, `elasticnet`, `random_forest`, `extra_trees`, `histgb`, `lightgbm`, `xgboost`, `svm`).

---

## Executive Summary & Actionable Decisions

1. **Live GPU Blocker on `node1` (HARD RULE CHECK)**: **0 / 8 RTX 4090 GPUs currently pass** the dual threshold (`VRAM < 50% [<12,282 MiB]` **AND** `Util < 50%`) across 4 snapshots (18:29–18:42 CST). GPUs 0/1/2 have <50% VRAM (3.7–7.4 GiB) but are pinned at **100% compute utilization** by `zwdong` (`esm2-3b`/`esm2-650m` indel scoring, running 14–29h); GPUs 3/4/5/6/7 all exceed 50% VRAM (12.5–22.5 GiB). **Wave-2 MUST run a live preflight gate before launching GPU jobs, and should prioritize CPU-capable work on `node23` (64 cores, 116 GiB RAM free, 236 GiB `/tmp` free) until 1–2 GPUs drop below both 50% thresholds.**
2. **`ncAA-RepDistill` Is Unblocked & Proven (< 15 min to fix)**:
   - Original training vocab `<WORK_DIR>/projects/AA_peptide/teacher/data/ncaa_adaptation_v2/vocab_smiles.txt` is **byte-identical** (SHA256 `7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92`, 37 tokens) to `<WORK_DIR>/ncaadistill/data/raw/adaptation/vocab_smiles.txt`.
   - Verified read-only on CPU: loading `best.pt` with `ema_state_dict` + `graph_attention` + `fusion_cfg` + `VocabManager(adaptation/vocab_smiles.txt)` achieves **24/24 (100%) exact top-1 MLM token recovery** on a masked peptide SMILES, mathematically proving vocab equivalence.
   - All 4 defects in `ncaa_admission.json` came solely from the legacy ad-hoc script `D:\work\中检院\scripts\node1_ncaa_extract.py` bypassing `<WORK_DIR>/ncaadistill/scripts/extract_phase3_features.py`.
3. **High-Value Pretrained Assets Found OUTSIDE HF Hub Stubs**:
   - `<WORK_DIR>/.cache/huggingface/hub/models--dptech--Uni-Mol2` and `models--tattabio--gLM2_650M` are **empty 4 KB stubs**, **HOWEVER full weights ARE on disk elsewhere on shared NFS `<WORK_DIR>`**:
     - **Uni-Mol2 (all 5 sizes, 84M–1.1B)**: `<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2/modelzoo/{84M,164M,310M,570M,1.1B}/checkpoint.pt` (322 MB to 4.3 GB) + editable `unimol_tools` in `cuda12.1`.
     - **gLM2-650M**: `<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/glm2/assets/tattabio-gLM2_650M@08754cba` (2.5 GB).
     - **GeminiMol (3D conformer-aware 2048-d GNN)**: `<WORK_DIR>/models/dd_model/GeminiMol/GeminiMol.pt` (3.7 GB).
     - **MoLFormer-XL-both-10pct**: `<WORK_DIR>/models/dd_model/MoLFormer-XL-both-10pct/` (187 MB safetensors).
     - **PeptideCLM-2 (`hybrid-small`, `mlm-small`, 121 MB each) & PeptideCLM-23M-all (167 MB)**: `<WORK_DIR>/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/`.
     - **ESM-C 600M (2.2 GB), ProtT5-XL-half (2.3 GB), ESM2-3B (11 GB), ESM2-650M (2.5 GB), ChemBERTa-77M-MTR/MLM (14 MB each), PPLM-650M (2.5 GB), ProtST-ESM2 (8.0 GB), PepLand (pre-trained GNN checkpoints)**.
4. **Recommended Python Binaries**:
   - **Tabular / Optuna / RDKit / PyTorch (`node1` & `node23` shared NFS)**: `/usr/bin/python3` (Python 3.10.12 + `<WORK_DIR>/.local/lib/python3.10/site-packages`: `rdkit 2026.09.1`, `lightgbm 4.7.0`, `xgboost 3.2.0`, `optuna 5.0.0`, `sklearn 1.7.2`, `torch 2.5.1+cu121`, `transformers 4.44.2`).
   - **PyG / Uni-Mol / FastAPI / Uvicorn / `fair-esm` (`node1` & `node23`)**: `<WORK_DIR>/anaconda3/envs/cuda12.1/bin/python` (Python 3.11.13: `torch 2.6.0+cu124`, `torch_geometric 2.6.1`, `unimol_tools`, `rdkit 2024.03.2`, `lightgbm 4.6.0`, `xgboost 3.1.2`, `transformers 4.48.1`, `esm 2.0.0`, `fastapi 0.128.0`, `uvicorn 0.40.0`).
   - **ESM-C 600M (`esm>=3`) / ESMFold2 (`node1`)**: `<WORK_DIR>/esmfold2-env312/bin/python` (Python 3.12: `torch 2.11.0+cu130`, `esm 3.4.2`, `rdkit 2026.9.1`, `transformers 5.22.0.dev0`).
   - **Local Windows Web Dev & Playwright Testing**: `py -3.11` (`<LOCAL_HOME>\AppData\Local\Programs\Python\Python311\python.exe`, Python 3.11.9: `rdkit 2026.09.1`, `fastapi 0.165.0`, `uvicorn 0.54.0`, `torch 2.11.0+cu128`, `lightgbm 4.7.0`, `xgboost 3.2.0`, `pytest 9.2.0`, `playwright` + Chromium in `%LOCALAPPDATA%\ms-playwright`). **Do NOT use `D:\work\中检院\.venv` (it is a Linux venv) or bare `python` (`hermes-agent\venv`, lacks `rdkit`/`pytest`).**
   - **Missing Everywhere on `node1`/`node23`**: `catboost`, `chemprop`, `tabpfn` (though `<WORK_DIR>/.cache/tabpfn/tabpfn-v2-regressor.ckpt` 44 MB is cached on disk).

---

## 1. Existing Peptide/Molecular Modeling Codebases & Pretrained Assets

### 1.1 Pretrained Foundation Model Weights on `node1` (Shared NFS `<WORK_DIR>`)

| Model / Asset | Verified On-Disk Path | Size | Status & Recommended Use for V33 |
|---|---|---:|---|
| **ncAA-RepDistill** (`phase2_repro_sweep24_e0_b0`) | `<WORK_DIR>/ncaadistill/artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt` | 1.6 GB | **READY (TOP PRIORITY)**. 13-layer SMILES+graph-distance Transformer distilled from ESM2-650M + GeminiMol. Produces `smiles_repr` (768d), `protein_proj` (1280d), `molecular_proj` (2048d), `fusion_hf` (2048d). Native ncAA/cyclic/modified peptide SMILES encoder. |
| **Uni-Mol2** (`84M / 164M / 310M / 570M / 1.1B`) | `<WORK_DIR>/models/dd_model/Uni-Mol/Uni-Mol2/modelzoo/{84M,164M,310M,570M,1.1B}/checkpoint.pt` | 322 MB – 4.3 GB | **READY** *(Note: HF hub `models--dptech--Uni-Mol2` is an empty 4 KB stub; use `dd_model` path + `cuda12.1` `unimol_tools`)*. Strong 3D/2D molecular encoder; `84M`/`164M` fast for cyclic/modified peptide ADME (PAMPA/Caco2/PPB/CL/Vd). |
| **GeminiMol** | `<WORK_DIR>/models/dd_model/GeminiMol/GeminiMol.pt` | 3.7 GB | **READY**. 2048-d conformer-space-profiled GNN (the exact teacher for ncAA-RepDistill Phase 2). Strong for permeability/PK/ADME of modified peptides. |
| **PeptideCLM-2** (`hybrid-small`, `mlm-small`) | `<WORK_DIR>/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/models/{peptideclm-2-hybrid-small,peptideclm-2-mlm-small}/` | 121 MB each | **READY (FAST, 31M params)**. Purpose-built peptide SMILES RoBERTa/ChemPepMTR (max 768 tokens, 512d). Verified chirality (D/L) & PTM sensitivity in `peptide-four-task-benchmark`. Very fast even on CPU. |
| **PeptideCLM-23M-all** (2025, SPE-fixed) | `<WORK_DIR>/projects/peptide-four-task-benchmark-20261022/resources/chemical-encoders-20261024-v1/peptideclm2025-assets-v1/model/pytorch_model.bin` + `methods/chemical-encoders-20261025-v3-peptideclm2025-spefix/` | 167 MB | **READY (FAST, 23M params, 768d)**. Includes audited `spefix_adapter.py` & `extract_cpu_spefix.py`; verified L/D-Ala (`Δ=1.45`) and Ser/pSer (`Δ=2.95`) sensitivity. |
| **MoLFormer-XL-both-10pct** | `<WORK_DIR>/models/dd_model/MoLFormer-XL-both-10pct/` (`model.safetensors`) | 187 MB | **READY (FAST, 768d)**. Linear-attention rotary SMILES transformer. Fast inference on long peptide SMILES. |
| **ChemBERTa-77M-MTR / MLM** | `<WORK_DIR>/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8adebea0cb59a8effa66b2020f204ca` & `<WORK_DIR>/models/dd_model/ChemBERTa-77M-{MTR,MLM}/` | 14 MB each | **READY (ULTRA-FAST, 384d)**. Runs on CPU or GPU in seconds; useful compact chemical baseline (truncated at 512 tokens). |
| **ESM-C 600M** | `<WORK_DIR>/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12` & `<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/esmc/assets/biohub-ESMC-600M@fcc9d36` | 2.2 GB | **READY (TOP SEQ PRIORITY)**. Next-gen ESM representation (1152d); strong complement to ESM2-650M (shown in `_fusionprop` with ~0.38 learned fusion weight alongside ESM2's ~0.62). Use `esmfold2-env312` (`esm 3.4.2`) or `transformers` loader. |
| **ESM2 family** (`8M / 35M / 150M / 650M / 3B`) | `<WORK_DIR>/.cache/huggingface/hub/models--facebook--esm2_*` (note: `15B` is an empty 4 KB stub) | 30 MB – 11 GB | **READY**. `650M` (2.5 GB, 1280d) is already frozen in V32; `3B` (11 GB, 2560d) available if GPU permits. |
| **ProtT5-XL-half (encoder)** | `<WORK_DIR>/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_half_uniref50-enc` *(note: `prot_t5_xl_uniref50` is empty 0 B)* | 2.3 GB | **READY (1024d)**. Historically strong orthogonal PLM to ESM2 for hemolysis, toxicity, AMP, and solubility. |
| **gLM2-650M** | `<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/glm2/assets/tattabio-gLM2_650M@08754cba` *(HF hub `models--tattabio--gLM2_*` are empty 4 KB stubs)* | 2.5 GB | **READY (1280d)**. Mixed genomic/protein LM; already wired in `D:\work\中检院\scripts\node1_all_embeddings.py`. |
| **PPLM-650M** | `<WORK_DIR>/projects/peptide-interaction-representation-20261025/methods/pplm/assets/pplm_t33_650M.pt` | 2.5 GB | **READY**. Peptide-adapted ESM2-650M backbone (`affinity_models.pkl` 13 GB also present for PPI). |
| **ProtST-ESM2** | `<WORK_DIR>/models/ProtST/protst_esm2.pth` | 8.0 GB | **READY**. Biomedical-text-aligned ESM2-650M; potentially strong for toxicity/functional endpoints. |
| **ESMFold2 / ESMFold2-Fast / ESMFold-v1** | `<WORK_DIR>/.cache/huggingface/hub/models--biohub--ESMFold2` (25 GB), `.../esmfold2/assets/biohub-ESMFold2-Fast@45fe8656` (25 GB), `models--facebook--esmfold_v1` (7.9 GB) | 7.9 – 25 GB | **READY (HEAVY)**. 3D structure prediction only; too slow for 40k-scale feature extraction or real-time web inference. Use only for selected showcase drugs / small endpoints if GPUs free up. |
| **PepLand** | `<WORK_DIR>/projects/AA_peptide/pepland/cpkt/{model,linear_pred_atoms,linear_pred_bonds,linear_pred_pharms}/data/model.pth` | ~10s MB | **READY**. Fragment/pharmacophore heterogeneous GNN pretrained specifically on canonical + non-canonical peptides. Requires `dgl`/legacy env audit before reuse. |
| **SaProt-650M-AF2 / all-MiniLM / ESM2-15B** | `<WORK_DIR>/.cache/huggingface/hub/models--*` | 4 KB | **EMPTY STUBS — NOT USABLE OFFLINE**. |
| **TabPFN v2 weights** | `<WORK_DIR>/.cache/tabpfn/tabpfn-v2-regressor.ckpt` | 44 MB | **WEIGHTS ONLY**. No `tabpfn` Python package installed in `<WORK_USER>` envs (would require `pip install tabpfn` in an isolated venv, high payoff for small-N endpoints N=20–500!). |

### 1.2 Survey of Existing Modeling Codebases (`node1` & Local Windows)

| Codebase | Path | Core Method / Architecture | Empirical Evidence & Concrete Reusability for V33 |
|---|---|---|---|
| **PepLife I5a-v2 & `peplife_next`** | `<WORK_DIR>/projects/PepLife/` (`src/peplife_i5a_attention_pool_model.py`, 15 frozen checkpoints) & `<WORK_DIR>/projects/peplife_next/` | **Frozen ESM2-650M residue-level embeddings + coarse positional modification features + fold-safe exact-modification token embeddings + learned Attention/Max pooling + global-modification & experimental-condition branches.** | **HIGHEST REUSABILITY FOR HALF-LIFE & CONDITION-AWARE ENDPOINTS**. Achieved Development OOF `R²=0.529`, `Pearson=0.728`, `MAE=0.515 log10 h` (1,314 samples / 507 sequence groups; `peplife_next` `l1beta` reached `R²=0.540`, `Pearson=0.735`, `Spearman=0.720`). Directly solves V32's biggest bottleneck on `half_life` and modified peptides where mean-pooling loses N/C-terminal & residue-level modification sites! Fast to train on CPU or 1 GPU. |
| **`_fusionprop`** | `<WORK_DIR>/projects/_fusionprop/` (`protein_feature_extractor/`, `protein_predictor/{solubility,thermostability,toxicity}/`) | **Learned weighted dual-PLM fusion (`ESM2-650M` + `ESM-C 600M`)** with per-sequence normalization + MLP head (`hidden_dim=256/768`, `dropout=0.5`). Includes clean `esm2_extractor.py`, `esmc_extractor.py`, `model_lifecycle.py` & 5-fold solubility + toxicity + thermostability checkpoints. | **HIGH REUSABILITY**. Learned fusion weights (`~0.62 ESM2 + ~0.38 ESMC` on solubility, `test R²=0.52–0.55`) validate that combining `ESM2` + `ESM-C` improves sequence endpoints without heavy compute. Reuse both the `ESMC` extractor pattern and gated/weighted multi-embedding fusion head. |
| **`ncAA-RepDistill`** | `<WORK_DIR>/ncaadistill/` | **13-layer RoPE SMILES Transformer + RDKit shortest-path graph-distance attention bias (`max_distance=8`, per-head) + dual-teacher distillation (`ESM2-650M` + `GeminiMol 2048d`) + residual pretrain fusion (`fusion_hf`) + EMA.** | **HIGHEST REUSABILITY FOR MODIFIED / CYCLIC PEPTIDES**. Single-input SMILES model encoding BOTH protein semantics (`protein_proj` 1280d) and 3D-aware chemical structure (`molecular_proj` 2048d, `fusion_hf` 2048d). Ready to fix & extract in <15 min (see §3). |
| **`kan-methods-20261019` & `kan-peptide-physical-benchmark-20261019`** | `<WORK_DIR>/projects/kan-methods-20261019/` & `kan-peptide-physical-benchmark-20261019/library/peptide-physical-benchmark-20261019/` | Audited implementations of B-spline KAN, FastKAN-RBF, ReLU-KAN, Wav-KAN, Chebyshev KAN, Fourier KAN, and **Nature KA-GNN (Fourier graph conv, 4 layers)**. Also SVD32 + `tanh(z/2)` compressed tabular pipeline. | **SELECTIVE REUSABILITY**. Key empirical finding: **KA-GNN beat MLP-SAGE substantially on BACE molecule classification (`0.8077 ± 0.0159` vs `0.7312 ± 0.0100` AUROC, 43.6k params, ~28s/run)**; **B-spline KAN beat strong MLP by 4.52% MAE on PROSPECT peptide iRT (`8.804` vs `9.221`, 95% CI `[-0.693, -0.120]`)** with 58% fewer params (5.2× CPU cost: ~25s CPU), while FastKAN-RBF underperformed. Worth testing **KA-GNN** as a fast GNN model family (43k params) and **B-spline KAN** on compact physicochemical/SVD-projected small-to-medium endpoints. |
| **`ncaa` (`AmpHGT`)** | `<WORK_DIR>/projects/ncaa/code/AmpHGT/` | Heterogeneous Graph Transformer (`PharmHGT_refactor.py`, `PepMAE.py`, `MacFrag.py`) over atom + pharmacophore/fragment graphs for ncAA AMPs. | **MEDIUM REUSABILITY**. Good GNN architecture reference for fragment+atom graphs, though requires DGL/custom graph preprocessing; KA-GNN or PyG AttentiveFP/GINE is cleaner and faster in `cuda12.1` (`torch_geometric 2.6.1`). |
| **`peptide-four-task-benchmark-20261022`** | `<WORK_DIR>/projects/peptide-four-task-benchmark-20261022/` (`methods/chemical-encoders-20261025-v3-peptideclm2025-spefix/`, `baselines/chemical-encoder-finetune-20261024-v1/`) | Production-audited extractors & full-finetune scripts for **PeptideCLM-2 (`hybrid-small`/`mlm-small`)** and **PeptideCLM-23M-all (SPE-fixed)** with L/D & PTM canary tests. | **HIGH REUSABILITY**. Ready-to-run `extract_cpu_spefix.py` and `ChemPepMTR.py` provide fast, stereochemistry- and PTM-aware peptide SMILES embeddings (512d / 768d) runnable even on `node23` CPU! |
| **`peptide-interaction-representation-20261025`** | `<WORK_DIR>/projects/peptide-interaction-representation-20261025/` (`analysis/pian/`, `analysis/augridge-arm2/`) | Tested PIAN (Perceiver + FiLM + SupCon) vs flat MLP & Ridge on peptide-protein affinity (`LOPO`), plus `/dev/shm` npy caching (`1.6–4.9 s/step -> 0.12–0.45 s/step`). | **CAUTIONARY / ARCHITECTURAL LESSON**. Complex PIAN components (FiLM conditioning, Perceiver, SupCon) *hurt* vs a simple flat MLP/Ridge (`0.740` vs `0.694` flat MLP vs `0.477` Ridge on LOPO)! Validates strong regularized linear / GBDT / compact neural heads + **use `/dev/shm` caching on `node1`/`node23` to eliminate NFS I/O bottlenecks**. |
| **`backbone_comparison_v4_20261017` & `molecular_memory_v3_20261017`** | `<WORK_DIR>/projects/backbone_comparison_v4_20261017/` & `molecular_memory_v3_20261017/` | Frozen Qwen2.5-1.5B / Qwen3-4B causal LM with hooked cross-attention `MolecularMemoryLM` reading ESM2-150M memory banks (`p/t/c/r` slots) + condition embeddings. | **LOW PRIORITY FOR V33**. Heavy LLM backbone (1.5B–4B) with high VRAM/latency, designed for conditional text+numeric reasoning rather than 55-endpoint ADMET screening. However, its **missing-condition embedding slot pattern** is a clean trick for assay-context conditioning. |
| **`AMP` & `BioTD`** | `<WORK_DIR>/projects/AMP/` (`all_sequences.csv` 12 MB + `MSA/`) & `<WORK_DIR>/projects/BioTD/cache_v3_short50_7128_20260508/` | `AMP`: curated AMP sequence corpus + MSAs. `BioTD`: precomputed ESM2-150M / graph / MSA caches for 7,128 short peptides (≤50 aa). | **DATA / CACHE REFERENCE ONLY**. No standalone new model architecture; avoid MSA-dependent models for fast web inference. |
| **Local `tmp_wheels/pepadmet`** | `D:\work\中检院\tmp_wheels\pepadmet\pepADMET-main\` (`utils/MY_GNN.py`, `calculate_descriptors.py`, `model/toxicity_early_stop.pth` 45.8 MB) | **Multi-task RGCN + FPN (Fingerprint/Descriptor Network) with task-specific atom-level attention (`WeightAndSum`)** for interpretable per-atom toxicity attribution + 2,133 descriptors (PyBioMed + modlAMP). | **MEDIUM-HIGH REUSABILITY (CONCEPT & INTERPRETABILITY)**. Legacy code targets Python 3.7 / DGL / PyBioMed (do not install legacy deps), **but its core dual-branch `GNN + Descriptor-FPN` + `WeightAndSum` per-atom attention readout is easy to port natively to PyG (`torch_geometric 2.6.1`)** for V33 multi-task ADMET + atom-level web UI highlighting! |
| **Local `D:\work\中检院\scripts\`** | `peptide_modeling_{features,expand_features,add_smiles_methods,unify_features,train,search}.py`, `extract_all_features.py`, `node1_all_embeddings.py`, `train_production_v2.py` | Complete audited V32 CPU feature extraction (13 independent methods), Optuna TPE grouped-CV search across 8 model families, plus `extract_all_features.py` (charge@pH7/7.4, pI, GRAVY, aliphatic/Boman index, α/β hydrophobic moments, modification counts) & `node1_all_embeddings.py` (ESM-C, ChemBERTa, gLM2, ESM2-3B). | **PRIMARY REUSABLE SCAFFOLD**. V33 should extend `peptide_modeling_train.py`/`search.py` rather than rewriting from scratch: add **OOF stacking/blending**, **PyTorch residual MLP / Attention-Pool (PepLife-style) / PyG GNN (KA-GNN / AttentiveFP)**, **multi-task heads** for correlated endpoints (PAMPA/Caco2/MDCK/RRCK; hemolysis/cytotoxicity/toxicity; CL/Vd/half_life), and **newly unlocked representations (`ncAA-RepDistill`, `ESM-C 600M`, `PeptideCLM-2/23M`, `MoLFormer-XL`, `Uni-Mol2`, `ProtT5-XL`)**. |

### 1.3 Ranked Concrete Recommendations for Wave-2 Modeling

1. **Immediate CPU-Safe Wins (Zero GPU Wait — Runnable on `node23` 64-core CPU right now)**:
   - **PeptideCLM-2 (`hybrid-small` 512d) & PeptideCLM-23M-all (SPE-fixed, 768d)** + **ChemBERTa-77M-MTR (384d)** + **MoLFormer-XL (768d)** extraction on `node23` CPU for modified/cyclic peptide SMILES.
   - **Physicochemical peptide sequence descriptors** from `extract_all_features.py` (pI, net charge @ pH 7.0/7.4, Eisenberg α/β hydrophobic moments, Boman/aliphatic index, D-AA/N-Me/cyclization counts) — fast (<10s) and missing from the current 28-dim `sequence` group!
   - **Out-of-Fold Multi-Representation Stacking + Multi-Seed Blending** across diverse families (GBDT + Ridge/ElasticNet + SVM-RBF + compact PyTorch Residual-MLP / B-spline KAN on SVD-compressed features) with proper probability calibration (isotonic/Platt) and split-conformal prediction intervals.
   - **Optional small-N breakthrough**: Install `tabpfn` in a clean private venv on `node23`/`node1` to leverage the already-cached `<WORK_DIR>/.cache/tabpfn/tabpfn-v2-regressor.ckpt` for small-N PK/permeability endpoints (`N=20–300`: Caco2, MDCK, RRCK, IV half-life, Vd, renal CL).
2. **Highest-Payoff GPU Wins (Once 1–2 `node1` GPUs Pass the <50%/<50% Gate)**:
   - **Fixed `ncAA-RepDistill` (`smiles_repr` 768d + `molecular_proj` 2048d + `protein_proj` 1280d + `fusion_hf` 2048d)** — strongest single representation upgrade for modified/cyclic peptides and ncAAs.
   - **ESM-C 600M (1152d)** & **ProtT5-XL-half (1024d)** — strongest orthogonal sequence PLM upgrades alongside ESM2-650M (validated by `_fusionprop`).
   - **PepLife I5a-v2 Attention-Pooling + Residue-Modification-Token Architecture** — directly upgrades `half_life` and sequence-based PK/stability tasks over mean-pooled ESM2.
   - **Uni-Mol2-84M/164M** or **GeminiMol (2048d)** — 3D-conformer-aware molecular embeddings for cyclic peptide permeability (PAMPA/Caco2/MDCK/RRCK) and PPB/CL/Vd.
   - **Multi-Task PyG GNN (KA-GNN / GINE + Descriptor FPN + `WeightAndSum` atom attention)** in `cuda12.1` (`torch_geometric 2.6.1`) — adds true end-to-end molecular graph diversity + web-ready atom-level explainability.

---

## 2. Python / Conda Environment Audit (`node1`, `node23`, `node2`, Local Windows)

> **Critical Infrastructure Note**: `<WORK_DIR>` (including `<WORK_DIR>/anaconda3`, `<WORK_DIR>/.local`, `<WORK_DIR>/software`, `<WORK_DIR>/esmfold2-env312`, `<WORK_DIR>/ncaa-venv`) is **shared NFS** (`<STORAGE_HOST>:/data`) mounted identically on both `node1` and `node23`. Cold imports of large packages (`torch`, `torch_geometric`, `sklearn`, `lightgbm`) over NFS take **8–30 seconds**; keep workers long-lived rather than spawning per-sample processes.

### 2.1 Verified Import Matrix — `node1` & `node23` (16 Target Packages)

| Package | `/usr/bin/python3` (+ `<WORK_DIR>/.local`, py3.10.12) | `<WORK_DIR>/anaconda3/envs/cuda12.1/bin/python` (py3.11.13) | `<WORK_DIR>/esmfold2-env312/bin/python` (py3.12) | `<WORK_DIR>/software/envs/pepbinding/bin/python` | `<WORK_DIR>/anaconda3/envs/protenix-v2/bin/python` | `<WORK_DIR>/ncaa-venv/bin/python` (py3.10.12) |
|---|---|---|---|---|---|---|
| `torch` | **2.5.1+cu121** (OK) | **2.6.0+cu124** (OK) | **2.11.0+cu130** | **2.4.1** | **2.7.1** | installed (`cu121`) |
| `torch_geometric` | — | **2.6.1** (OK) | — | **2.6.0** | — | — |
| `chemprop` | — | — | — | — | — | — |
| `rdkit` | **2026.09.1** (OK) | **2024.03.2** (OK) | **2026.9.1** | — *(in `pepbinding39`: 2025.9.2)* | **2025.9.3** | installed |
| `lightgbm` | **4.7.0** (OK) | **4.6.0** (OK) | — | — | — | — |
| `xgboost` | **3.2.0** (OK) | **3.1.2** (OK) | — | — | — | — |
| `catboost` | — | — | — | — | — | — |
| `optuna` | **5.0.0** (OK) | — | — | **5.0.0** | — | — |
| `scikit-learn` | **1.7.2** (OK) | **1.4.2** (OK) | **1.9.1** | **1.9.1** | **1.7.1** | installed |
| `tabpfn` | — | — | — | — | — | — |
| `esm` | — | **2.0.0** (`fair-esm` active; `esm 3.1.6` dist-info present) | **3.4.2** (`EvolutionaryScale`) | **2.0.0** (`fair-esm`) | **2.0.0** (`fair-esm`) | — |
| `transformers` | **4.44.2** (OK) | **4.48.1** (OK) | **5.22.0.dev0** | — | — | — |
| `unimol_tools` | — | **OK** (editable `${WORK_DIR}/dd_model/Uni-Mol/unimol_tools`) | — | — | — | — |
| `scipy` | **1.15.3** (OK) | **1.11.4** (OK) | **1.18.1** | **1.17.1** | **1.17.1** | installed |
| `fastapi` | — | **0.128.0** (OK) | — | — | — | — |
| `uvicorn` | — | **0.40.0** (OK) | — | — | — | — |

**Additional Notable Conda Envs on `node1`/`node23`**:
- `<WORK_DIR>/anaconda3/envs/web/bin/python`: `torch 2.6.0`, `esm 3.1.6`, `transformers 4.48.1`, `sklearn 1.6.1`, `scipy 1.15.2`.
- `<WORK_DIR>/anaconda3/envs/diffdock/bin/python`: `fastapi 0.128.0`, `uvicorn 0.39.0`, `torch 1.13.1+cu117`, `torch_geometric 2.2.0`, `rdkit`, `fair-esm 2.0.0`.
- `<WORK_DIR>/anaconda3/envs/boltz/bin/python`: `torch 2.6.0`, `rdkit 2024.3.2`, `sklearn 1.6.1`, `scipy 1.13.1`.
- `<WORK_DIR>/anaconda3/envs/esmfold2/bin/python`: `torch 2.15.0`, `transformers 5.22.1`.
- `<WORK_DIR>/software/micromamba-root/envs/torch-cu126-py311/bin/python`: `torch 2.9.1+cu126`, `transformers 4.57.3`, `sklearn 1.8.0`.
- **Confirmed Absent Across All ~45 `<WORK_USER>` Envs**: `catboost`, `chemprop`, `tabpfn`. *(Can be installed into a new isolated venv under `<WORK_DIR>/v33_venv` if Wave-2 wants CatBoost/TabPFN/Chemprop without touching existing envs.)*

---

## 3. `ncAA-RepDistill` Deep Audit vs. `ncaa_admission.json`

### 3.1 Exact Files & Checkpoint Inventory on `node1`
- **Repository**: `<WORK_DIR>/ncaadistill/`
- **Checkpoint**: `<WORK_DIR>/ncaadistill/artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt` (**1,602.73 MB**, epoch 29, global_step 13,020, `best_metric=0.014277`, `run_id=v8__phase2_sweep24_tradeoff__E0_B0__s42_e30__20260118_125500`).
  - Top-level keys: `['model_state_dict', 'optimizer_state_dict', 'global_step', 'epoch', 'config', 'best_metric', 'scheduler_state_dict', 'ema_state_dict']`.
  - Contains **234 EMA parameter tensors** (`ema_state_dict`, max L∞ diff vs `model_state_dict` on embedding = `0.01173`; only `smiles_encoder.rope.inv_freq` buffer is omitted from EMA and has `0.0` diff vs fresh initialization).
  - Architecture stored inside `ckpt['config']`:
    - `smiles_encoder`: `vocab_size=37`, `embed_dim=768`, `num_layers=13`, `num_heads=12`, `max_length=768`, `dropout=0.12`.
    - `projection`: `esm2_dim=1280`, `chemberta_dim=2048` (GeminiMol target).
    - `graph_attention`: `enabled=true`, `max_distance=8`, `bias_mode="per_head"` (`smiles_encoder.graph_bias_embed.weight` shape `(10, 12)`).
    - `fusion`: `enabled=true`, `mode="residual_distill"`, `target="geminimol"`, `hs_source="molecular_proj"`, `proj_dim=2048`.
    - `training.ema`: `enabled=true`, `decay=0.999`, `use_for_eval=true`.
- **Canonical Config**: `<WORK_DIR>/ncaadistill/config/phase2/phase2_repro_sweep24_e0_b0.yaml`
- **Official Extractors (Already Correct!)**:
  - Batch: `<WORK_DIR>/ncaadistill/scripts/extract_phase3_features.py` (supports `--use_ema`, `graph_attention`, `fusion_cfg`, `VocabManager`, all 5 feature heads: `smiles_repr` [768], `smiles_mean_pool` [768], `protein_proj` [1280], `molecular_proj` [2048], `fusion_hf` [2048], plus `--long_smiles_strategy chunk_mean|seq_segment_mean` for SMILES > 768 chars).
  - Single-molecule: `<WORK_DIR>/ncaadistill/scripts/extract_single_smiles_feature.py`.

### 3.2 Root Cause of Every Failure in `ncaa_admission.json` + Mathematical Proof

`D:\work\中检院\data\peptide_modeling_20261113_v1\ncaa_admission.json` rightly rejected the legacy cache (`<WORK_DIR>/ncaa_features_v32.npz`) because the old ad-hoc script `D:\work\中检院\scripts\node1_ncaa_extract.py` introduced **4 exact bugs**:

| Defect in `ncaa_admission.json` | Legacy Script (`node1_ncaa_extract.py`) Bug | Verified Ground-Truth Training Contract |
|---|---|---|
| **1. Vocab file & token ID mismatch** (`10/30` same IDs) | Loaded `data/raw/downstream/vocab_smiles.txt` (32 chars, no special tokens, SHA256 `c807d8a7...`) and prepended custom `{"<pad>":0,"<unk>":1,"<cls>":2,"<mask>":3}` | Training config points to `ncaa_adaptation_v2/vocab_smiles.txt`, which is **present on `node1`** at `<WORK_DIR>/projects/AA_peptide/teacher/data/ncaa_adaptation_v2/vocab_smiles.txt` and is **byte-identical** (SHA256 `7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92`, 37 lines: `0=[PAD], 1=[CLS], 2=[SEP], 3=[MASK], 4=[UNK], 5=# ... 36=s`) to `<WORK_DIR>/ncaadistill/data/raw/adaptation/vocab_smiles.txt`. `vocab_manager.py` is also byte-identical (`2fa54f4f...`). |
| **2. Missing graph attention** | Omitted `graph_attention` arg and `graph_distances` input, silently dropping `smiles_encoder.graph_bias_embed.weight` `(10, 12)` | Pass `graph_attention=cfg["student"]["graph_attention"]` and RDKit shortest-path `graph_distances` via `SingleInputDataset._get_graph_distances(smiles)`. |
| **3. Missing fusion module** | Omitted `fusion_cfg`, silently dropping `fusion_prot_to_target` | Pass `fusion_cfg=cfg["fusion"]` to enable `fusion_hf = molecular_proj + fusion_prot_to_target(protein_proj)` (2048d). |
| **4. Non-EMA weights** | Loaded `ckpt["model_state_dict"]` | Training set `ema.use_for_eval: true`; must load `ckpt["ema_state_dict"]` (0 missing trainable weights; only 4 unused Phase-2 aux heads `p4_task_head*` are unexpected). |

- **Empirical Proof Executed (Read-Only CPU Probe)**: Running `SingleInputDualTeacher` with `ema_state_dict` + `graph_attention` + `fusion_cfg` + `VocabManager(data/raw/adaptation/vocab_smiles.txt)` on Phe-Leu (`CC(C)CC(NC(=O)C(N)Cc1ccccc1)C(=O)O`) with single-position `[MASK]` probing achieved **24/24 (100.0%) exact top-1 MLM token recovery** (`C->C`, `(->(`, `)->)`, `N->N`, `=->=`, etc.). **Vocab equivalence is now 100% proven.**

### 3.3 Feasibility & Exact Fix Steps (< 15 Minutes Engineering)
- **Feasible in < 30 min?** **YES — ~10–15 minutes to write and verify** a clean standalone extractor (no edits to `<WORK_DIR>/ncaadistill/` required).
- **Exact Steps for Wave-2**:
  1. Write a new standalone extractor script (e.g., `v33_extract_ncaa_repdistill.py`) that adds `<WORK_DIR>/ncaadistill` to `sys.path`, reads `config/phase2/phase2_repro_sweep24_e0_b0.yaml`, instantiates `VocabManager("<WORK_DIR>/ncaadistill/data/raw/adaptation/vocab_smiles.txt")` (asserting SHA256 `7270675ae5db795089e44a1fa6f574b13694013ba05e315d52f28fa76d5f4b92`), instantiates `SingleInputDualTeacher` with `graph_attention` + `fusion_cfg`, and loads `ckpt["ema_state_dict"]` (asserting `missing == ["smiles_encoder.rope.inv_freq"]` only).
  2. Use `SingleInputDataset` (or its exact `VocabManager.encode(..., add_special_tokens=True)` + `_get_graph_distances`) to compute `smiles_input_ids`, `smiles_attention_mask`, `graph_distances`.
  3. **Optimization**: Pad dynamically to each batch's `max_len` (capped at 768) instead of fixed 768 (1 peptide at L=768 took 5.2s on CPU; dynamic length gives ~5–10× speedup on typical peptides, and on 1 RTX 4090 `batch=64` in bf16/fp16 extracts 40,557 SMILES in ~6–10 minutes). For SMILES > 766 chars (e.g. GLP-1/lipidated peptides), record an explicit `truncated` mask or apply `chunk_mean` pooling from `extract_phase3_features.py` (noting graph bias is disabled on split chunks).
  4. Extract and save all 4 complementary heads in one forward pass: `smiles_repr` (768d), `molecular_proj` (2048d), `protein_proj` (1280d), `fusion_hf` (2048d), plus a 24/24 MLM self-check receipt and input/vocab/checkpoint SHA256s to satisfy an updated `ncaa_admission_v2.json`.
  5. **Python binary**: `/usr/bin/python3` on `node1` (has `torch 2.5.1+cu121`, `rdkit 2026.09.1`, `yaml`, `pandas`, `numpy`).

---

## 4. Live GPU Audit on `node1` (`nvidia-smi`)

**HARD USER RULE**: Use **AT MOST 2 GPUs** on `node1`, and **ONLY** GPUs satisfying **BOTH**:
- `VRAM used < 50%` (`< 12,282 MiB / 24,564 MiB`), **AND**
- `GPU utilization < 50%`.

### 4.1 Multi-Timestamp Live Snapshot (2026-11-21, 8× NVIDIA GeForce RTX 4090, 24,564 MiB each, Driver 580.95.05 / CUDA 13.0)

| GPU | VRAM @ 18:29 | Util @ 18:29 | VRAM @ 18:36 | Util @ 18:36 | VRAM @ 18:38 | Util @ 18:38 | VRAM @ 18:42 | Util @ 18:42 | Rule Pass? | Active Owners & Processes (PID / User / Runtime / Job) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| **0** | 3,680–13,224 MiB | 100% | 3,680 MiB (15%) | 100% | 7,516 MiB (31%) | 100% | 3,684 MiB (15%) | 100% | **NO (Util 100%, VRAM spiky)** | `1257076` `zwdong` (1d 00h, `esm2-650m` indel, 2.5G); `2428805` `zfchen` (2h, `graphb3` jupyter, 0.5G); `2527768` `jhu` (1h, `tabpfn` worker, 0.7–10.2G spiky) |
| **1** | 7,371 MiB (30%) | 100% | 7,371 MiB (30%) | 100% | 7,371 MiB (30%) | 100% | 7,371 MiB (30%) | 100% | **NO (Util 100%)** | `850249` `zwdong` (1d 05h, `esm2-3b` shard 1/3, 7.3G) |
| **2** | 7,393 MiB (30%) | 100% | 7,393 MiB (30%) | 100% | 7,393 MiB (30%) | 100% | 7,393 MiB (30%) | 100% | **NO (Util 100%)** | `1824142` `zwdong` (13h 50m, `esm2-3b` shard 0/3, 6.9G); `1569406` `zfchen` (20h, `graphb3` jupyter, 0.5G) |
| **3** | 19,033 MiB (77%) | 88–96% | 19,039 MiB (77%) | 32% | 19,039 MiB (77%) | 5% | 19,039 MiB (77%) | 8% | **NO (VRAM 77%)** | `3033020` `jjfang` (2d 08h, `without_rc.py`, 18.4G); `2524207` `jjzhuang` (1h 13m, `cardiosimtox` 5-fold, 0.6G) |
| **4** | 22,519 MiB (92%) | 100% | 22,519 MiB (92%) | 99% | 22,519 MiB (92%) | 99% | 22,519 MiB (92%) | 100% | **NO (VRAM 92%, Util 100%)** | `3378383` `jjfang` (5d 08h, `without_distance.py`, 17.5G); `1238294` `zwdong` (1d 00h, `esm1v` indel, 5.0G) |
| **5** | 14,937 MiB (61%) | 88–90% | 14,937 MiB (61%) | 29% | 14,937 MiB (61%) | 93% | 14,937 MiB (61%) | 93% | **NO (VRAM 61%, Util 93%)** | `2433007` `yxwang` (2h 04m, `MOSAIC` fold 8, 12.0G); `2503437` `hyhu` (1h 27m, `DTIUQ2` train, 2.9G) |
| **6** | 12,523 MiB (51%) | 57–79% | 12,517 MiB (51%) | 42% | 12,517 MiB (51%) | 71% | 12,517 MiB (51%) | 72% | **NO (VRAM 51%, Util 72%)** | `2345642` `yxwang` (2h 59m, `MOSAIC` fold 7, 12.0G); `2534848` `jjzhuang` (1h 06m, `cardiosimtox`, 0.5G) |
| **7** | 7,371 MiB (30%) | 100% | 15,705 MiB (64%) | 100% | 15,705 MiB (64%) | 100% | 15,705 MiB (64%) | 100% | **NO (VRAM 64%, Util 100%)** | `850248` `zwdong` (1d 05h, `esm2-3b` shard 2/3, 7.3G) + new job started ~18:35 (+8.3G) |

- **Current Verdict**: **0 GPUs eligible right now.** Most likely first candidates to clear are **GPU 1 / GPU 2** (once `zwdong`'s `esm2-3b` shards finish) or **GPU 5 / GPU 6** (once `yxwang`'s `MOSAIC` fold 7/8 finishes — currently ~2–3h in).
- **Mandatory Wave-2 Preflight Gate**: Before any GPU job on `node1`, query `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`, require at least 2 consecutive samples (10s apart) where `memory.used < 12282` AND `utilization.gpu < 50`, select at most 2 indices, and bind strictly via `CUDA_VISIBLE_DEVICES=<idx>`.
- **Host CPU/RAM/Disk on `node1`**: 64 cores, 503 GiB RAM (377 GiB available), `/` has 78 GiB free (90% used — avoid dumping large caches to `/tmp` on `node1`), `/data` NFS has 2.7 TiB free / 73 TiB (97% used).

---

## 5. Runtime & Web Environment Check (`node23`, Local Windows, `node2`)

### 5.1 `node23` (CPU Compute + OmniGrid Backend)
- **Hardware & Disk**: 64 CPU cores, 125 GiB RAM (116 GiB available), **NO GPU**.
  - `/` & `/tmp` (`/dev/sda3`): **236 GiB free** / 549 GiB (55% used) — ample fast local scratch!
  - `<WORK_DIR>` (`<STORAGE_HOST>:/data` NFS): **2.7 TiB free** / 73 TiB (**97% used** — shared cluster volume; keep artifacts tidy).
- **Live Service (`http://127.0.0.1:8901/health`)**:
  - **HEALTHY — HTTP 200**: `{"ok": true, "models_cached": 152, "entries": 57, "endpoints": 55}` *(Note: `curl` is not installed on `node23`; probe via `/usr/bin/python3` `urllib.request`)*.
  - Running as `<WORK_USER>` PID `1052187` (`bash <WORK_DIR>/s3/omnigrid/run_service.sh` -> PID `1237413` `python3 <WORK_DIR>/s3/omnigrid/node23_omnigrid_service.py`).
  - **Important Architecture & Fragility Findings**:
    - Implemented using Python stdlib `ThreadingHTTPServer` (NOT FastAPI/Uvicorn), because `/usr/bin/python3` lacks `fastapi`/`uvicorn`.
    - Reads `<WORK_DIR>/s3/omnigrid/deploy_manifest.json` (11 MB), `engine/` (frozen feature extractors), `halflife_anchor.joblib`, AND **hardcodes `FEAT_MANIFEST` at `/tmp/peptide_modeling_work_multiseed20h_campaign_v1/round_000_seed_42/features/manifest.json`** (volatile `/tmp` dependency — V33 should store its manifest under `<WORK_DIR>/`!).
    - `service.log` shows occasional RDKit SMILES parse errors when raw Chinese drug names (e.g. `卡泊芬净`) are sent without name-to-SMILES resolution.
- **RDKit & FastAPI/Uvicorn Verification on `node23`**:
  - `/usr/bin/python3` (3.10.12): `rdkit 2026.09.1` smoke test PASSED (`MW=89.094`), `lightgbm 4.7.0`, `xgboost 3.2.0`, `optuna 5.0.0`, `sklearn 1.7.2`, `scipy 1.15.3`, `torch 2.5.1+cu121`, `transformers 4.44.2` all import cleanly. No `fastapi`/`uvicorn`.
  - `<WORK_DIR>/anaconda3/envs/cuda12.1/bin/python` (3.11.13): **`fastapi 0.128.0`, `uvicorn 0.40.0`, `rdkit 2024.03.2`, `lightgbm 4.6.0`, `xgboost 3.1.2`, `sklearn 1.4.2`, `scipy 1.11.4` ALL import cleanly on `node23`**. Note: if deploying FastAPI on `node23` serving models trained with `sklearn 1.7.2`/`rdkit 2026.09.1`, match the training sklearn/lightgbm versions (or install `fastapi`+`uvicorn` into a clean py3.10 venv inheriting `/usr/bin/python3` + user-site) to avoid pickle/version skew!

### 5.2 Local Windows (`D:\work\中检院`) — Dev & Playwright Testing
- **Node / npm**: `node v24.14.1` (`C:\Program Files\nodejs\node.exe`), `npm 11.11.0` (`npx playwright` is NOT installed globally/locally in Node; use Python Playwright).
- **Python Interpreters**:
  - **RECOMMENDED**: `py -3.11` -> `<LOCAL_HOME>\AppData\Local\Programs\Python\Python311\python.exe` (Python 3.11.9)
    - Verified installed: **`rdkit 2026.09.1`, `fastapi 0.165.0`, `uvicorn 0.54.0`, `starlette 1.10.0`, `jinja2 3.1.6`, `httpx 0.28.1`, `torch 2.11.0+cu128`, `lightgbm 4.7.0`, `xgboost 3.2.0`, `scikit-learn 1.9.0`, `scipy 1.17.1`, `pandas 2.3.3`, `numpy 2.4.6`, `pytest 9.2.0`, `playwright` (with Chromium builds `1208/1223/1234/1243` + headless shells already cached in `%LOCALAPPDATA%\ms-playwright`)**.
  - **DO NOT USE**:
    - `D:\work\中检院\.venv`: **Linux venv** (`home = /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin`, has `lib64/`, no Windows `Scripts\python.exe`).
    - Default `PATH` `python`: points to `<LOCAL_HOME>\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe` (has `fastapi 0.133.1`, `uvicorn 0.41.0`, `playwright`, but **missing `rdkit` and `pytest`**).
    - `uv` managed Pythons (`3.10.20`, `3.12.13`): bare stdlib only.

### 5.3 `node2` (`<DEPLOY_USER>@<APP_HOST>`, Read-Only Production Web Audit)
- **Host & Hardware**: `<APP_HOSTNAME>`, `/` has **490 GiB free** / 879 GiB (42% used).
  - **GPUs**: 2× RTX 4090 — GPU 0: `15,173 / 24,564 MiB` (0% util), GPU 1: `8,901 / 24,564 MiB` (0% util; serves ESM2-650M for OmniPanel/OmniGrid).
- **Directory Layout (`/var/www/peptide-omnipanel/`)**:
  - `app/` (`src/`, `pyproject.toml`, `requirements.txt`, `.venv/`)
  - `omnigrid/app.py` (38 KB, V32 Gradio frontend on `127.0.0.1:7862`, `cuda:1`)
  - `scripts/run_peptide_omnipanel_v31_web.py` + `run_server.sh` (V31 Gradio on `127.0.0.1:7861`, `cuda:1`)
  - `esm2_650m/`, `models/release_peptide18_a3_20260722/`, `run_exports/`, `run_history/`, `nginx/`, `wheels/`
- **Active Systemd Services**:
  - `peptide-omnipanel.service`: active running (`127.0.0.1:7861`, `cuda:1`, user `<DEPLOY_USER>`).
  - `peptide-omnigrid.service`: active running (`127.0.0.1:7862`, `cuda:1`, user `<DEPLOY_USER>`, runs `/var/www/peptide-omnipanel/app/.venv/bin/python /var/www/peptide-omnipanel/omnigrid/app.py`).
- **Installed Packages in `/var/www/peptide-omnipanel/app/.venv/bin/python` (Python 3.10.12)**:
  - **Web / ASGI (READY for V33 FastAPI!)**: `fastapi 0.141.1`, `uvicorn 0.53.0`, `starlette 0.52.1`, `pydantic 2.12.3`, `Jinja2 3.1.6`, `aiofiles 24.1.0`, `httpx 0.28.1`, `python-multipart 0.0.32`, `gradio 5.50.0`.
  - **ML / Chem / PLM**: `rdkit 2026.03.6`, `torch 2.6.0+cu124`, `transformers 4.57.6`, `lightgbm 4.6.0`, `scikit-learn 1.7.2`, `scipy 1.15.3`, `numpy 2.2.6`, `pandas 2.3.3`.
  - **Missing in `node2` venv**: `xgboost`, `catboost` *(so either run XGBoost/heavy multi-endpoint inference via the `node23` backend API or add `xgboost` wheel to `node2` venv at deploy time; `lightgbm`, `sklearn`, `rdkit`, `torch`, `transformers`, `fastapi`, `uvicorn` are already native on `node2`!)*.
- **Nginx Reverse Proxy Layout (Read-Only)**:
  - Main site `/etc/nginx/sites-enabled/lmmd.ecust.edu.cn` includes:
    - Line 372: `include /etc/nginx/snippets/peptide-omnipanel.location.conf;`
    - Line 373: `include /etc/nginx/snippets/peptide-omniv32.location.conf;`
  - `/etc/nginx/snippets/peptide-omnipanel.location.conf`: redirects `/peptide-omnipanel` -> `/peptide-omnipanel/` and reverse-proxies `location ^~ /peptide-omnipanel/` -> `http://127.0.0.1:7862/` (with WebSocket headers, `300s` timeouts, `client_max_body_size 10M`).
  - `/etc/nginx/snippets/peptide-omniv32.location.conf`: 301-redirects `/peptide-omniv32` and `/peptide-omniv32/` -> `/peptide-omnipanel/`.
  - **V33 Deployment Implication**: V33 FastAPI+SPA can be brought up cleanly on a fresh local port (e.g., `127.0.0.1:7863`) alongside V31 (`7861`) and V32 (`7862`) for side-by-side verification, and cut over atomically via a single `proxy_pass` update in `/etc/nginx/snippets/peptide-omnipanel.location.conf` (respecting `--root-path /peptide-omnipanel` and relative asset paths).
