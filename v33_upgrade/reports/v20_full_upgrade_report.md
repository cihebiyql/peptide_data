# Peptide OmniPanel v1.2 / Engine v33.2 — Plan A & Plan B Unified SOTA Upgrade Report

- **Date:** `2026-11-23`
- **Live Public URL:** `https://lmmd.ecust.edu.cn/peptide-omnipanel/`
- **Deployment Topology:** 100% self-contained on `node2` (`172.21.43.16`, user `mode2`, zero `sudo`, zero `node23`/`node1` runtime dependency, `0 MiB` GPU VRAM on `node2`)
- **Corpus Scale:** `50,995` unified compounds (`50,994` curated + Case17 clinical benchmarks), `46` unified endpoints (`24` Tier-1 Core ADMET/PK/Safety + `22` Tier-2 ESKAPE Antimicrobial/Bactericidal/Biofilm), `82,327` harmonized experimental rows (`4,535` censored-aware records), `25` curated clinical & AMP examples.
- **Global Zero-Leakage Audit:** `0` leaked scaffold clusters and `0` leaked compounds across `110` MT46 CV folds + `40` GNN CV folds + all `40` Mondrian/CQR regression conformal calibrations + all `11` 3D-chameleonic CV & Leave-17-Cases-Out runs.

---

## 1. Executive Summary of Production Promotions vs. Audited Research Findings

| Track | Deliverable | Key Audited Metric (Untouched Holdout / Live CPU) | Production Status (`node2`) |
|---|---|---|---|
| **Plan A** | DeepSeek-Harness Visual Redesign, Procedural Molecular Canvas, 4 Original Clinical Peptide Cards, Interactive Human ADMET & ESKAPE Barrier Map, Animated 4-FM Pipeline SVG, Interactive 17-Case Parity Chart | `0` console/page/HTTP errors, `0` forbidden strings in ZH & EN, `16/16` pytest PASS | **PROMOTED LIVE** (`v1.2`) |
| **Plan B-1** | **Mondrian AD-Stratified Empirical-Bayes Conformal Calibration (`mondrian_ad_eb`, Pure JSON)** across all `40` regression endpoints | Pooled OOD 80% coverage **`64.2% → 76.0%` (`+11.8 pp` dual)** & **`53.9% → 74.2%` (`+20.3 pp` SMILES)** while **tightening mean width** (`1.479 → 1.470`); `0 ms` overhead, `0` `.joblib` | **PROMOTED LIVE** (all `40` regression EPs in `v33_manifest.json`) |
| **Plan B-1** | CQR (`cqr_lgb`) & Small-N Foundation/GPR (`TabPFN v2`, `MultiBlockCompositeGPR`) | CQR hit `79.2%/91.0%` global & lifted `PPB` `29.2% → 66.7%`; zero-tuning `TabPFN v2` improved 9 weakest small-N EPs holdout MAE `0.5917 → 0.5598` (`-0.0319`, e.g. `Vd_absolute 0.7780 → 0.4485`), revealing small-N winner's curse under multi-recipe OOF selection | **ARCHIVED & DOCUMENTED** (`v2_upgrade/smalln_cqr/`; deferred from `node2` runtime to avoid `tabpfn` & `.joblib` dependency bloat) |
| **Plan B-2 (Upgrade 1)** | **46-Endpoint Unified `MultiTaskPeptideNet46`** (`9.23M` params, Residual SwiGLU + 4-Region ESM-C SDPA + Exact Tobit Loss, `0.00e+00` `smiles_only` seq invariance) | Strong zero-regression holdout lifts on promoted V33.1 endpoints: `efflux_ratio` HO MAE `0.3599 → 0.3472` (**Spearman `0.642 → 0.741`, `+0.099`**; SMILES `0.4073 → 0.3689`), `cytotoxicity_CC50` `0.4057 → 0.3931` (`Sp 0.652 → 0.677`), `hemolysis_MHC` `0.4830 → 0.4738` (`Sp 0.550 → 0.567`), `MIC_Salmonella` `0.5498 → 0.5370`, `MIC_B_subtilis` `-0.0024`, `MBC_S_aureus` SMILES `-0.0137` | **PROMOTED LIVE** (`5` unified fp16 folds = `90 MB` shared across `7` verified OOF+HO-winning tracks; `24/24` core V33.0 point weights `0.0` diff!) |
| **Plan B-2 (Upgrade 2)** | 3D Conformer-Aware Chameleonic Descriptors (`29D` RDKit ETKDGv3 + `rdFreeSASA` + Geometric IMHB) | `10,450/11,002` (`94.98%`) valid 3D conformers; Direct 4-FM ablation improved `6/11` dual (`0.4260 → 0.4215`, `PPB -0.0291`, `RRCK -0.0175`); stacked lift modest (`0.4016 → 0.4005`) at median `708 ms / p90 2,823 ms` CPU cost | **ARCHIVED & DOCUMENTED** (`v2_upgrade/gpu_sota/conformer_3d_chameleonic/`; deferred from live path to preserve `<200 ms` latency budget; `Uni-Mol2-84M` cleanly deferred under local VRAM/RAM guardrails) |
| **Plan B-2 (Upgrade 3)** | **Portable Pure-PyTorch 4-Layer E-GATv2 + Task `WeightAndSum` Live Per-Atom Attribution Engine** (`gnn_8ep_5fold_fp16.pt`, **`17.47 MB`**) | Strictly `['numpy', 'rdkit', 'torch']` (zero `torch_geometric`/`dgl`); `0` leaked clusters across `8` flagship endpoints (`overall_peptide_toxicity` HO ROC **`0.8980`**, `cytotoxicity_cls` **`0.8340`**, `hemolysis_cls` **`0.8223`**, `Caco2` unified Sp **`0.782`**); CPU latency **4/4 `<100 ms`** (Semaglutide `92.7 ms`, Cyclosporine A `49.5 ms`, Micafungin `57.2 ms`, Magainin-2 `65.2 ms`) | **PROMOTED LIVE** (`POST /api/v1/atom-attribution` + auto-attached on `/api/v1/predict` + interactive 8-endpoint pill switcher in 2D Structure Card!) |

---

## 2. Plan A — Visual Redesign, Original Scientific Graphics & UX (`v1.1` → `v1.2`)

1. **Atmospheric DeepSeek-Harness Hero & Procedural Molecular Canvas**: Interactive gentle drifting macrocycle/helix particle network (`visuals.js`) that auto-pauses when navigating into results or secondary tabs.
2. **100% Original Copyright-Free Scientific Assets (`web/static/assets/`)**:
   - `card_semaglutide.png` (`512×320`, `191.5 KB`) — C18 fatty-diacid lipidated GLP-1 helix engaging serum albumin pocket
   - `card_cyclosporine.png` (`512×320`, `272.3 KB`) — 33-atom N-methylated cyclic macrocycle traversing phospholipid bilayer
   - `card_micafungin.png` (`512×320`, `208.8 KB`) — cyclic echinocandin lipopeptide engaging fungal cell-wall barrier
   - `card_magainin.png` (`512×320`, `222.5 KB`) — amphipathic cationic α-helix forming bacterial membrane pore
   - `admet_human_map.svg` — 6 interactive pulsing physiological hotspots (`bbb`, `gut`, `pk`, `clearance`, `safety`, `micro`) with live holdout scorecard inspection
   - `arch_4fm_flow.svg` — animated 4-Foundation-Model super-ensemble data-flow diagram
3. **Interactive 17 Clinical Cases Parity Chart**: Pure-ML vs. Acylation-Anchored toggle (Spearman `0.903`, Anchored MAE `0.299 log10 h`, `0/17` clinical hemolysis false positives).
4. **Live Per-Atom Attribution Toggle (`v1.2`)**: The 2D Structure Card now defaults to **Motif Highlights** and enables **Atom Attribution (`原子归因热图`)** with an interactive 8-endpoint selector pill bar (`PAMPA`, `Caco-2`, `Hemolysis`, `HC50`, `Toxicity`, `Cytotox`, `CC50`, `Half-Life T½`) backed by server-side & client-side caching (`<20 ms` switching after initial `<95 ms` GNN pass).

---

## 3. Plan B-1 — Small-N Modeling & Mondrian AD-EB Conformal Calibration

- **Mondrian AD-Stratified Empirical-Bayes (`mondrian_ad_eb`, Pure JSON)**:
  - Calibrates separate Empirical-Bayes shrunk quantiles (`q80_by_tier`, `q90_by_tier`) for `In-Domain` (`Tc ≥ 0.65`), `Borderline` (`0.40 ≤ Tc < 0.65`), and `OOD` (`Tc < 0.40`) using strictly nested OOF residuals (`0` holdout peeking).
  - Across all `40` regression endpoints (`7,052` untouched holdout rows):
    - Dual-Modality OOD 80% coverage: **`64.2% → 76.0%` (`+11.8 pp`)**; Mean 80% width: **`1.479 → 1.470`** (narrower!)
    - SMILES-Only OOD 80% coverage: **`53.9% → 74.2%` (`+20.3 pp`)**; Mean 80% width: **`1.510 → 1.490`**
    - Endpoints landing inside nominal `[78%, 85%]` 80% coverage window: **`14/40 → 20/40`**.
- **Small-N Winner's Curse Finding**:
  - Zero-tuning `TabPFN v2` improved holdout MAE on the 9 weakest small-N endpoints (`0.5917 → 0.5598`, `-0.0319`; `Vd_absolute 0.7780 → 0.4485`, `MDCK 0.5426 → 0.4530`, `Cmax 0.9156 → 0.8298`), and `node2`-portable `MultiBlockCompositeGPR` improved to `0.5834` (`-0.0083`), while raw V33.1 40–80-recipe OOF MAE displayed selection optimism (`RRCK` SMILES-only passed strict OOF gate `0.3218 → 0.2989`). Documented transparently for future nested-selection waves without breaking strict OOF discipline.

---

## 4. Plan B-2 — 46-EP `MultiTaskPeptideNet46`, 3D Chameleonic & Pure-PyTorch E-GAT Attribution

### 4.1 Promoted `MultiTaskPeptideNet46` Tracks (Zero Holdout Regression, `0` Core-24 Weight Mutation)

| Endpoint | Modality Track | n_CV / n_HO | Wave 5A HO MAE | V33.2 (+MT46) HO MAE | Δ HO MAE | HO Spearman (Base → New) | Promoted Ensemble Weights |
|---|---|---:|---:|---:|---:|---|---|
| `efflux_ratio` | `dual_modality` | 138 / 33 | 0.3599 | **0.3472** | `-0.0126` | `0.642 → 0.741` (`+0.099`) | `cpu:0.30, fusion:0.32, mt46:0.39` |
| `efflux_ratio` | `smiles_only` | 138 / 33 | 0.4073 | **0.3689** | `-0.0383` | `0.579 → 0.696` (`+0.117`) | `cpu:0.56, fusion:0.10, mt46:0.35` |
| `cytotoxicity_CC50` | `dual_modality` | 2678 / 669 | 0.4057 | **0.3931** | `-0.0126` | `0.652 → 0.677` (`+0.025`) | `cpu:0.36, fusion:0.36, mt46:0.28` |
| `hemolysis_MHC` | `dual_modality` | 397 / 99 | 0.4830 | **0.4738** | `-0.0092` | `0.550 → 0.567` (`+0.017`) | `fusion:0.61, mt46:0.39` |
| `MIC_Salmonella` | `dual_modality` | 575 / 144 | 0.5498 | **0.5370** | `-0.0128` | `0.547 → 0.541` | `cpu:0.47, fusion:0.26, mt46:0.27` |
| `MIC_B_subtilis` | `dual_modality` | 1180 / 295 | 0.5595 | **0.5570** | `-0.0024` | `0.414 → 0.417` | `cpu:0.22, fusion:0.52, mt46:0.26` |
| `MBC_S_aureus` | `smiles_only` | 441 / 110 | 0.6276 | **0.6139** | `-0.0137` | `0.340 → 0.348` | `cpu:0.34, fusion:0.36, mt46:0.30` |

### 4.2 Live Pure-PyTorch E-GAT + `WeightAndSum` Attribution Engine (`17.47 MB`, `node2` CPU)
- **Zero External Graph Libraries:** Strictly `numpy`, `rdkit`, `torch` — loads directly in `node2`'s existing `torch 2.6.0` CPU runtime with `CUDA_VISIBLE_DEVICES=""`.
- **Single-Pass 8-Endpoint Caching:** One `<95 ms` forward pass computes `[0, 1]` atom importance vectors for all 8 endpoints simultaneously (`PAMPA`, `Caco2`, `hemolysis_cls`, `hemolysis_HC50`, `overall_peptide_toxicity`, `cytotoxicity_cls`, `cytotoxicity_CC50`, `half_life_reg`); switching endpoints in the web UI takes `<20 ms`.

---

## 5. Companion Technical Artifacts
- Plan B-1 Detailed Report: `v33_upgrade/v2_upgrade/smalln_cqr/smalln_cqr_report.md`
- Plan B-2 Detailed Report: `v33_upgrade/v2_upgrade/gpu_sota/gpu_sota_report.md`
- Manifest Merge Script: `v33_upgrade/v2_upgrade/merge_v332_manifest.py`
- Portable MT46 Inference Module: `v33_upgrade/models_gpu_deep/mt46_inference.py`
- Portable GNN Atom Attribution Engine: `v33_upgrade/web/gnn_atom_attribution.py`
