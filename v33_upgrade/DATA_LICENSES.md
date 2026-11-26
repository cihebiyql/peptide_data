# Data & Pretrained Model License Ledger (`DATA_LICENSES.md`)

> **Code License**: MIT (original source code only — see `LICENSE`).  
> **Trained Model Weights & Hosted Web Service**: Strictly **Non-Commercial Academic & Early Preclinical Research Only** (`internal_research_only`), respecting the most restrictive non-commercial upstream terms among constituent corpora and foundation model encoders.  
> **Public Repository Redistribution Policy**: This public repository redistributes **zero row-level third-party observation tables**. Only original code, audited evaluation manifests, aggregate corpus statistics, and factual FDA/DailyMed numeric PK pointers are hosted here.

---

## 1. Upstream Experimental Datasets (Training-Only — NOT Redistributed in Public Repo)

| Dataset / Source | Endpoints Covered | Verified Upstream License / Terms | Redistributed in Public Repo? | Primary Citation / Portal |
| :--- | :--- | :--- | :---: | :--- |
| **CycPeptMPDB (v1.2)** | PAMPA, Caco-2, MDCK, RRCK, LogD7.4 | CC BY 4.0 (as declared on official CycPeptMPDB portal; underlying source copyrights retained by original papers) | **No** (training-only; aggregate stats only) | Li et al., *J. Chem. Inf. Model.* 2023; https://cycpeptmpdb.com/ |
| **SweMacrocycleDB** | PAMPA, Efflux Ratio | CC BY 4.0 (Figshare release) | **No** (training-only) | SweMacrocycleDB Figshare v1 (2024–2025) |
| **Naylor LPE 2018 / JMC 2024 Macrocycle Lipophilicity** | LogD7.4 (octanol/water), LogD_decadiene (1,9-decadiene/water) | Published literature factual measurements (JMC 2024 decadiene measurements ingested via ADMETatlas V1.0 structured table, CC-BY-NC 4.0 upstream curation; strictly non-commercial training-only) | **No** (training-only) | Naylor et al., *J. Med. Chem.* 2018; *J. Med. Chem.* 2024; Yi et al., *NAR* 2026 |
| **DBAASP v4.0.1** | Hemolysis (cls/HC50/MHC), Cytotoxicity, ESKAPE MIC/MBC/LC | Academic / Non-Commercial (DBAASP Terms of Use) | **No** (training-only) | Pirtskhalava et al., *Nucleic Acids Res.* 2021; https://dbaasp.org/ |
| **DRAMP / Hemolytik2 / HemoPI2** | Hemolysis, Cytotoxicity, AMP MIC | Academic / Non-Commercial (Raghava Lab / DRAMP academic terms; HemoPI2 GPL-3.0 code) | **No** (training-only) | Shi et al., *NAR* 2022; Rathore et al., *Protein Sci.* 2025 (HemoPI2) |
| **PEPlife2 / THPdb / PepTherDia** | Half-life (`half_life_reg/cls`), PPB, CL, Vd, F, Cmax | Academic / Non-Commercial (Raghava Lab / PepTherDia CC BY-NC 4.0) | **No** (training-only) | Mathur et al. (*PEPlife*); Usmani et al. (*THPdb*); D'Aloisio et al. (*PepTherDia*) |
| **B3Pdb** | Blood-Brain Barrier (`BBB`) | Academic / Non-Commercial (Raghava Lab academic terms) | **No** (training-only) | Kumar et al., *J. Chem. Inf. Model.* 2021 |
| **ChEMBL (v33–v35)** | Multi-endpoint PK / Toxicity / MIC | CC BY-SA 3.0 | **No** (training-only) | Zdrazil et al., *Nucleic Acids Res.* 2024 |
| **NLM DailyMed SPL & US FDA Drugs@FDA NDA/BLA Reviews** | Clinical & Regulatory PK Reference Panel (`clinical_evidence_index.json`) | Factual numeric PK measurements (non-copyrightable facts); FDA NDA/BLA reviews are US Federal Government works (17 U.S.C. § 105); DailyMed SPL contains manufacturer-submitted prescribing text (not reproduced; official `setid` URLs linked directly) | **Yes** (factual numeric PK table + verified DailyMed `setid` & FDA URLs only) | https://dailymed.nlm.nih.gov/ ; https://www.accessdata.fda.gov/ |
| **ADMETatlas V1.0** | Outbound portal interoperability link; non-commercial training-only source for JMC 2024 `LogD_decadiene` table (not redistributed) | CC BY-NC 4.0 | **No** (zero derivative mirroring; outbound link only) | Yi et al., *Nucleic Acids Res.* 2026, doi:10.1093/nar/gkag1147 |

---

## 2. Pretrained Foundation Models (Frozen Feature Encoders — Weights Not Redistributed)

| Model | Modality | Verified Upstream License | Source / Repository |
| :--- | :--- | :--- | :--- |
| **ESM-C 600M (`esmc_600m`)** | Canonical 20-AA Sequence (4-region pooling) | **EvolutionaryScale Cambrian Non-Commercial License** | https://github.com/evolutionaryscale/esm |
| **ncAA-RepDistill v2** | Covalent SMILES (cyclic / modified peptides) | Non-Commercial Academic Research (upstream distilled checkpoint; weights not redistributed) | Internal distilled checkpoint |
| **PeptideCLM-2** | Covalent Peptide SMILES | **MIT License** (official PeptideCLM / AaronFeller GitHub release) | https://github.com/AaronFeller/PeptideCLM |
| **ChemBERTa-MTR (`ChemBERTa-77M-MTR`)** | Small-molecule / Peptide SMILES | **MIT License** (DeepChem / Seyone Chithrananda HuggingFace release) | https://huggingface.co/DeepChem/ChemBERTa-77M-MTR |

---

## 3. External Published Benchmark Tools Evaluated (Non-Redistributed Evaluation Only)

| Tool | Endpoint Benchmarked | Upstream License | Source |
| :--- | :--- | :--- | :--- |
| **HemoPI2 (v1.3)** | `hemolysis_cls`, `hemolysis_HC50` | GPL-3.0 (PyPI `hemopi2==1.3`) | Rathore et al., *Protein Sci.* 2025; https://github.com/raghavagps/hemopi2 |
| **CycPeptMP** | `PAMPA` | MIT License | Li et al., *Brief. Bioinform.* 2024; https://github.com/akiyamalab/cycpeptmp |
| **Multi_CycGT** | `PAMPA` | GPL-3.0 License | Cao et al., *J. Chem. Inf. Model.* 2024; https://github.com/hongliangduan/Multi_CycGT |

---

## 4. Compliance Notes for Journal Data Availability Review
1. **Strict Separation of Code, Data, and Weights**: Original code is MIT-licensed. Because constituent training corpora (DBAASP, PEPlife2, THPdb, ADMETatlas-extracted JMC 2024 decadiene table) and ESM-C 600M carry non-commercial terms, trained model weights and hosted predictions are strictly limited to non-commercial academic research (`internal_research_only`).
2. **Public Repository Cleanup Policy**: All row-level third-party observation TSVs (previously under `data/peptide_ml_cleaning_v1/*.tsv`) and prior CC-BY-NC clinical index versions have been removed from the working tree and rewritten out of the `main` branch via `git-filter-repo`, with legacy non-`main` branches deleted from the remote repository. Note that GitHub may retain cached views of previously force-pushed orphan commits until cleared via GitHub Support; a fresh, single-commit archival repository/Zenodo release will be minted for final journal publication.
3. **DailyMed Clinical PK Provenance**: All 59 marketed FDA-approved peptide drugs in `clinical_evidence_index.json` contain strictly factual numeric PK values (initially cross-checked from public FDA label extractions including ADMETatlas V1.0, then stripped of all CC-BY-NC derivative fields, FAERS summaries, and `ATLPD*` identifiers, and verified against live NLM DailyMed v2 REST API `setid`s — 53/53 resolved, 87.3% exact numeric match in live SPL text).
4. **Live API Nearest-Analog Transparency**: The hosted web service's `nearest_analogs` panel displays up to 3 nearest training compounds (structure/observed endpoint value) solely for non-commercial academic applicability-domain inspection (`internal_research_only`).
