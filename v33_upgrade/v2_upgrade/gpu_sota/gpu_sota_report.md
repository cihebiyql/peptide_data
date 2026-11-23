# Plan B-2: GPU SOTA Upgrades Report (Local RTX 5080 — Zero-Leakage & CPU-Portable)

> **Production Status (v33.2 Route A)**: **Upgrade 1 (`MultiTaskPeptideNet46`) is ARCHIVED RESEARCH-ONLY (`0 / 92` endpoint tracks served in production)** following independent audit of `unified_mt46` vs per-endpoint train-serve mismatch and pure-OOF regression (`+0.00180` dual HO MAE); all 46 production endpoints preserve strict OOF-selected Wave 5A weights (`0.0` diff). **Upgrade 2 (3D Chameleonic)** is archived research-only. **Upgrade 3 (Pure-PyTorch E-GAT)** is deployed solely as a **qualitative 2D graph surrogate visualization** (`0.35·y_add` residual branch), distinct from the 4-FM super-ensemble main predictions.

- **Generated:** `2026-11-23 01:41:08 +0800`
- **Hardware:** `Local Windows RTX 5080 (16,303 MiB VRAM, bf16) for Upgrades 1 & 3 + local CPU for Upgrade 2 3D conformer extraction + node23 64-core CPU (125 GB RAM) for Upgrade 2 11-endpoint benchmark under local memory guardrail (zero node1 GPU / node2 live interference)`
- **Global Zero-Leakage Audit:** `0` leaked clusters & `0` leaked compounds across `110` MT46 CV folds + `40` GNN CV folds + all 11 3D-Chameleonic CV/Leave-17-Out runs.

## Upgrade 1 — 46-Endpoint Unified `MultiTaskPeptideNet` for 22 V33.1 Expansion Endpoints

- **Data Scale:** `50,995` compounds, `46` endpoints (`82,327` rows); model parameters: `9,226,632`; wall-clock: `3514.8s`.
- **Strict Zero-Leakage Masking:** Per target endpoint $E$ & fold $k$, masked ALL auxiliary rows in all 45 other endpoints whose `cluster_id` or `compound_idx` belonged to `Holdout(E) U ValFold(E, k)` (`110/110` folds verified, `0` leaks).
- **Dual-Modality Aggregate (22 Endpoints, Nested OOF Simplex `[CPU, 4-FM Fusion, 46-EP MT]` vs Wave 5A Champion):**
  - OOF CV Mean MAE: `0.4585` → **`0.4532`** (Δ=`-0.0052`; MT46 adopted on **`11/22`** endpoints)
  - Untouched 20% Holdout Mean MAE: `0.5171` → **`0.5187`** (Δ=`+0.0016`; improved on `7`, unchanged `10`, regressed `5`)
  - Untouched Holdout Mean Spearman: `0.4664` → **`0.4651`** | Mean R²: `0.1968` → **`0.1927`**
- **SMILES-Only Aggregate (22 Endpoints):** Holdout Mean MAE `0.5093` → **`0.5082`** (MT46 adopted on `7/22` endpoints; Holdout Spearman `0.4866` → `0.4845`).

| Endpoint | n_CV / n_HO | Wave5A CV MAE | 3-Way CV MAE | Wave5A HO MAE | MT46 Solo HO MAE | 3-Way HO MAE | Δ HO MAE | HO Spearman (Base→New) | Frozen Weights (`dual`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LC_antimicrobial` | 435 / 109 | 0.4592 | 0.4550 | 0.4615 | 0.5150 | **0.4734** | +0.0119 | 0.424 → 0.442 | `cpu:0.20, fusion:0.48, mt46:0.32` |
| `MBC_A_baumannii` | 109 / 27 | 0.4945 | 0.4721 | 0.5933 | 0.6745 | **0.6140** | +0.0207 | 0.377 → 0.354 | `cpu:0.05, fusion:0.67, mt46:0.28` |
| `MBC_E_coli` | 438 / 110 | 0.4404 | 0.4404 | 0.4062 | 0.4901 | **0.4062** | +0.0000 | 0.641 → 0.641 | `fusion:1.00` |
| `MBC_E_faecalis` | 72 / 18 | 0.4677 | 0.4677 | 0.6881 | 0.6464 | **0.6881** | +0.0000 | 0.597 → 0.597 | `fusion:1.00` |
| `MBC_P_aeruginosa` | 314 / 78 | 0.4933 | 0.4933 | 0.4837 | 0.4994 | **0.4837** | +0.0000 | 0.706 → 0.706 | `fusion:1.00` |
| `MBC_S_aureus` | 441 / 110 | 0.5239 | 0.5239 | 0.6276 | 0.6367 | **0.6276** | +0.0000 | 0.340 → 0.340 | `fusion:1.00` |
| `MBEC_biofilm` | 110 / 28 | 0.3654 | 0.3612 | 0.7836 | 0.9426 | **0.8269** | +0.0433 | 0.038 → -0.129 | `cpu:0.09, fusion:0.56, mt46:0.35` |
| `MBIC_biofilm` | 238 / 58 | 0.4302 | 0.4302 | 0.5600 | 0.6152 | **0.5600** | +0.0000 | 0.241 → 0.241 | `cpu:1.00` |
| `MFC_C_albicans` | 88 / 22 | 0.3649 | 0.3649 | 0.3771 | 0.3415 | **0.3771** | +0.0000 | 0.191 → 0.191 | `fusion:1.00` |
| `MIC_A_baumannii` | 751 / 188 | 0.4466 | 0.4451 | 0.4431 | 0.4806 | **0.4424** | -0.0006 | 0.549 → 0.523 | `cpu:0.22, fusion:0.44, mt46:0.34` |
| `MIC_B_subtilis` | 1180 / 295 | 0.4450 | 0.4423 | 0.5595 | 0.5973 | **0.5570** | -0.0024 | 0.414 → 0.417 | `cpu:0.22, fusion:0.52, mt46:0.26` |
| `MIC_C_neoformans` | 237 / 50 | 0.3945 | 0.3945 | 0.4952 | 0.5125 | **0.4952** | +0.0000 | 0.435 → 0.435 | `fusion:1.00` |
| `MIC_E_faecalis` | 502 / 126 | 0.5707 | 0.5707 | 0.5322 | 0.5455 | **0.5322** | +0.0000 | 0.453 → 0.453 | `fusion:1.00` |
| `MIC_E_faecium` | 226 / 57 | 0.4917 | 0.4930 | 0.3470 | 0.4278 | **0.3409** | -0.0061 | 0.683 → 0.707 | `fusion:1.00` |
| `MIC_K_pneumoniae` | 777 / 194 | 0.5071 | 0.5071 | 0.4300 | 0.4667 | **0.4300** | +0.0000 | 0.612 → 0.612 | `fusion:1.00` |
| `MIC_L_monocytogenes` | 198 / 50 | 0.5062 | 0.4742 | 0.5125 | 0.5749 | **0.5266** | +0.0141 | 0.783 → 0.788 | `cpu:0.06, fusion:0.32, mt46:0.62` |
| `MIC_S_epidermidis` | 627 / 157 | 0.4945 | 0.4948 | 0.4991 | 0.5510 | **0.5000** | +0.0009 | 0.550 → 0.551 | `cpu:0.36, fusion:0.39, mt46:0.25` |
| `MIC_Salmonella` | 575 / 144 | 0.5353 | 0.5323 | 0.5498 | 0.5788 | **0.5370** | -0.0128 | 0.547 → 0.541 | `cpu:0.47, fusion:0.26, mt46:0.27` |
| `Vd_absolute` | 50 / 13 | 0.4597 | 0.4597 | 0.7780 | 0.8273 | **0.7780** | +0.0000 | -0.163 → -0.163 | `cpu:1.00` |
| `cytotoxicity_CC50` | 2678 / 669 | 0.4141 | 0.4064 | 0.4057 | 0.4411 | **0.3931** | -0.0126 | 0.652 → 0.677 | `cpu:0.36, fusion:0.36, mt46:0.28` |
| `efflux_ratio` | 138 / 33 | 0.3709 | 0.3572 | 0.3599 | 0.3541 | **0.3472** | -0.0126 | 0.642 → 0.741 | `cpu:0.30, fusion:0.32, mt46:0.39` |
| `hemolysis_MHC` | 397 / 99 | 0.4103 | 0.3844 | 0.4830 | 0.4958 | **0.4738** | -0.0092 | 0.550 → 0.567 | `fusion:0.61, mt46:0.39` |

## Upgrade 2 — 3D Conformer-Aware Chameleonic Descriptors (11 Tier-1 PK/Permeability Endpoints)

- **3D Chameleonic Extractor (`node2`-CPU-portable RDKit ETKDGv3 + `rdFreeSASA` + Geometric IMHB):** `10,450/11,002` valid (`94.98%`, `>250`-heavy-atom macrocycles cleanly masked with `valid_3d=0.0`); batch time `4505.61s`.
- **Live CPU Inference Latency (60-mol sample, 4 threads):** median **`708.72 ms/mol`**, mean `1360.12 ms/mol`, p90 `2823.2 ms/mol`.
- **Uni-Mol2-84M Extraction (`deferred_vram_ram_ceiling`):** Verified load & forward pass on local RTX 5080 (100-mol length-sorted bf16 probe: 40.3 mol/s, 2.6 GiB active alloc; first 1,500 cyclic peptides extracted cleanly), but variable-length O(B*L^2*128) 4D pair tensors fragmented PyTorch WDDM reserved VRAM to 15.8/16.3 GiB and 24.2 GB host RAM commit on 80-128 atom macrocycles. Deferred cleanly under strict local stability guardrails (<=9 GB VRAM, <=65% RAM), occupied node1 4090s, and node2-CPU-portability requirement.
- **Leave-17-Cases-Out (`case17_leaveout`):** Evaluated cleanly across all 11 Tier-1 endpoints (`10/17` drugs with valid 3D ETKDGv3 conformers; `7/17` `>250`-heavy-atom peptides/insulins such as Semaglutide, Tirzepatide, Liraglutide, Exenatide, Teriparatide, Insulin detemir/degludec cleanly masked with `valid_3d=0.0`, zero failures).
- **Controlled Direct 4-FM Fusion Ablation (11 Endpoints):** Dual HO Mean MAE `0.4260` → `0.4215` (improved on `6/11`); SMILES-only HO Mean MAE `0.4152` → `0.4081` (improved on `5/11`).
- **Nested Stacked 4-FM Fusion vs +3D Chameleonic (11 Endpoints):** Holdout Mean MAE `0.4016` → **`0.4005`** (improved on `3/11`; Mean Spearman `0.6582` → `0.6608`).

| Endpoint | 3D Cov | Direct Base→+3D HO MAE | Base 4-FM CV→+3D CV MAE | Base 4-FM HO MAE | +3D Cham HO MAE | Δ HO MAE | +3D+UniMol2 HO MAE | Base→+3D HO Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PAMPA` | 100.0% | 0.4337 → 0.4300 | 0.3804 → 0.3804 | 0.3732 | **0.3732** | +0.0000 | — | 0.778 → 0.778 |
| `Caco2` | 99.9% | 0.3557 → 0.3584 | 0.3723 → 0.3723 | 0.3663 | **0.3663** | +0.0000 | — | 0.861 → 0.861 |
| `MDCK` | 100.0% | 0.5354 → 0.5385 | 0.4530 → 0.4548 | 0.4989 | **0.4895** | -0.0093 | — | 0.455 → 0.473 |
| `RRCK` | 100.0% | 0.3316 → 0.3141 | 0.3785 → 0.3785 | 0.3200 | **0.3200** | +0.0000 | — | 0.548 → 0.548 |
| `efflux_ratio` | 100.0% | 0.3995 → 0.4079 | 0.3709 → 0.3709 | 0.3599 | **0.3599** | +0.0000 | — | 0.642 → 0.642 |
| `F` | 83.1% | 0.2500 → 0.2484 | 0.2071 → 0.2032 | 0.2390 | **0.2539** | +0.0149 | — | 0.548 → 0.519 |
| `PPB` | 83.3% | 0.2910 → 0.2619 | 0.1511 → 0.1511 | 0.2608 | **0.2608** | +0.0000 | — | 0.678 → 0.678 |
| `LogD7.4` | 100.0% | 0.4816 → 0.4689 | 0.4188 → 0.4148 | 0.4570 | **0.4623** | +0.0053 | — | 0.702 → 0.704 |
| `half_life_reg` | 66.3% | 0.7088 → 0.7149 | 0.5957 → 0.5957 | 0.6654 | **0.6654** | +0.0000 | — | 0.482 → 0.482 |
| `CL_per_kg` | 75.7% | 0.5633 → 0.5698 | 0.5001 → 0.4977 | 0.5477 | **0.5346** | -0.0131 | — | 0.832 → 0.817 |
| `Vd_per_kg` | 76.5% | 0.3356 → 0.3236 | 0.3685 → 0.3727 | 0.3295 | **0.3201** | -0.0094 | — | 0.714 → 0.766 |

## Upgrade 3 — Portable Pure-PyTorch Molecular E-GAT + Live Per-Atom Attribution (`gnn_atom_attribution.py`)

- **Architecture:** `Pure-PyTorch 4-Layer Edge-Conditioned Multi-Head GATv2 + Multi-Scale JK + Task WeightAndSum` (`1,800,557` params/fold, 100% pure PyTorch + RDKit + NumPy — AST-audited external deps: `['numpy', 'rdkit', 'torch']`, ZERO `torch_geometric` / `dgl`).
- **Compact Deployment Checkpoint:** `gnn_8ep_5fold_fp16.pt` = **`17.47 MB`** (< 25 MB budget, 5 folds in fp16, `0` leaked holdout/Case17 clusters).

### Strict Zero-Leakage 5-Fold CV & Untouched 20% Holdout (8 Flagship Web Endpoints, SMILES Graphs)

| Endpoint | Type | n_CV / n_HO | Per-EP CV Primary | Per-EP HO Primary | Per-EP HO Secondary | Unified 5-Fold HO Primary | Leaked Clusters |
|---|---|---:|---:|---:|---:|---:|---:|
| `PAMPA` | reg | 6865 / 1716 | MAE=0.4628 | **MAE=0.5048** | Sp=0.543, R2=0.265 | MAE=0.4977 (Sp=0.576) | 0 |
| `Caco2` | reg | 1265 / 315 | MAE=0.4810 | **MAE=0.4738** | Sp=0.733, R2=0.531 | MAE=0.4388 (Sp=0.782) | 0 |
| `hemolysis_cls` | cla | 14460 / 3629 | ROC=0.7942 | **ROC=0.8223** | PR=0.641, MCC=0.336 | ROC=0.7869 (PR=0.570) | 0 |
| `hemolysis_HC50` | reg | 4447 / 1112 | MAE=0.5223 | **MAE=0.4980** | Sp=0.434, R2=0.069 | MAE=0.5216 (Sp=0.441) | 0 |
| `overall_peptide_toxicity` | cla | 8828 / 2208 | ROC=0.8932 | **ROC=0.8980** | PR=0.907, MCC=0.645 | ROC=0.8893 (PR=0.896) | 0 |
| `cytotoxicity_cls` | cla | 2347 / 587 | ROC=0.7253 | **ROC=0.8340** | PR=0.703, MCC=0.353 | ROC=0.8066 (PR=0.633) | 0 |
| `cytotoxicity_CC50` | reg | 2394 / 605 | MAE=0.5290 | **MAE=0.5370** | Sp=0.404, R2=0.291 | MAE=0.5571 (Sp=0.392) | 0 |
| `half_life_reg` | reg | 2317 / 573 | MAE=0.7415 | **MAE=0.7852** | Sp=0.409, R2=0.165 | MAE=0.7704 (Sp=0.411) | 0 |

### Unit Test Verification (`gnn_atom_attribution.py` on Local CPU)

| Peptide | Input Mode | Heavy Atoms | Median CPU Latency | `< 200 ms` | Weights Finite `[0,1]` | PAMPA vs Hemolysis Weight Corr | Valid SVG Heatmap |
|---|---|---:|---:|---|---|---:|---|
| **Semaglutide** | `smiles` | 291 | **`92.68 ms`** | PASS | PASS | +0.206 | PASS (`half_life_reg`, 298.6 KB) |
| **Cyclosporine_A** | `smiles` | 85 | **`49.45 ms`** | PASS | PASS | +0.885 | PASS (`PAMPA`, 87.0 KB) |
| **Micafungin** | `smiles` | 86 | **`57.16 ms`** | PASS | PASS | +0.441 | PASS (`hemolysis_cls`, 94.3 KB) |
| **Magainin_2** | `sequence` | 174 | **`65.24 ms`** | PASS | PASS | +0.183 | PASS (`hemolysis_HC50`, 185.1 KB) |
