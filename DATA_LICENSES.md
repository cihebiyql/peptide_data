# Data & Pretrained Model License Ledger (`DATA_LICENSES.md`)

> **Code License**: MIT (see `LICENSE`).  
> **Trained Model Weights & Web Inference Scope**: Strictly **Non-Commercial Academic & Preclinical Research Only** (`internal_research_only`), respecting the most restrictive upstream non-commercial academic terms among constituent public corpora.  
> **Public Repository Redistribution Policy**: This repository distributes platform code, audited evaluation manifests, and US Federal Government public-domain regulatory PK references only. Raw third-party datasets with non-commercial or share-alike terms are NOT mirrored in this repository.

---

## 1. Upstream Experimental Datasets

| Dataset / Source | Endpoints Covered | Upstream License / Terms | Redistributed in Public Repo? | Primary Citation / URL |
| :--- | :--- | :--- | :---: | :--- |
| **CycPeptMPDB (v1.2)** | PAMPA, Caco-2, MDCK, RRCK, LogD7.4 | Academic / CC-BY-4.0 (verify latest portal terms) | No (training-only) | Li et al., *J. Chem. Inf. Model.* 2023; https://cycpeptmpdb.com/ |
| **SweMacrocycleDB** | PAMPA, Efflux Ratio | CC-BY 4.0 (Figshare) | No (training-only) | Figshare v1 (2024–2025) |
| **Naylor LPE 2018 / JMC 2024 Macrocycle Lipophilicity** | LogD7.4 (octanol/water), LogD_decadiene (1,9-decadiene/water) | Published literature factual measurements (ACS JMC SI) | No (training-only) | Naylor et al., *J. Med. Chem.* 2018; *J. Med. Chem.* 2024 |
| **DBAASP v4.0.1** | Hemolysis (cls/HC50/MHC), Cytotoxicity, ESKAPE MIC/MBC/LC | Academic / Non-Commercial | No (training-only) | Pirtskhalava et al., *Nucleic Acids Res.* 2021; https://dbaasp.org/ |
| **DRAMP / Hemolytik2 / HemoPI2** | Hemolysis, Cytotoxicity, AMP MIC | Academic / Non-Commercial | No (training-only) | Shi et al., *NAR* 2022; HemoPI2 (2025); Hemolytik2 (2026) |
| **PEPlife2 / THPdb / PepTherDia** | Half-life (`half_life_reg/cls`), PPB, CL, Vd, F, Cmax | Academic / Non-Commercial | No (training-only) | Mathur et al. (*PEPlife*); Usmani et al. (*THPdb*); D'Aloisio et al. (*PepTherDia*) |
| **B3Pdb** | Blood-Brain Barrier (`BBB`) | Academic / Non-Commercial | No (training-only) | Kumar et al., *J. Chem. Inf. Model.* 2021 |
| **ChEMBL (v33–v35)** | Multi-endpoint PK / Toxicity / MIC | CC-BY-SA 3.0 | No (training-only) | Zdrazil et al., *Nucleic Acids Res.* 2024 |
| **US FDA DailyMed SPL & Drugs@FDA NDA/BLA Reviews** | Clinical & Regulatory PK Reference Panel (`clinical_evidence_index.json`) | **US Federal Government Public Domain (17 U.S.C. § 105)** | **Yes (factual PK table + direct FDA/DailyMed URLs)** | https://dailymed.nlm.nih.gov/ ; https://www.accessdata.fda.gov/ |
| **ADMETatlas V1.0** | Outbound interoperability links only; factual JMC 2024 decadiene measurements audited independently from primary JMC SI | **CC-BY-NC 4.0** (upstream portal) | **No** (zero derivative mirroring; outbound link only) | Yi et al., *Nucleic Acids Res.* 2026, doi:10.1093/nar/gkag1147 |

---

## 2. Pretrained Foundation Models (Frozen Feature Encoders)

| Model | Modality | Upstream License | Source |
| :--- | :--- | :--- | :--- |
| **ESM-C 600M** | Canonical 20-AA Sequence (4-region pooling) | EvolutionaryScale Cambrian Open License (Non-Commercial) | EvolutionaryScale (`esmc_600m`) |
| **ncAA-RepDistill v2** | Covalent SMILES (cyclic / modified peptides) | Academic Research License | Upstream checkpoint |
| **PeptideCLM-2** | Covalent Peptide SMILES | Academic / Apache-2.0 | Upstream checkpoint |
| **ChemBERTa-MTR** | Small-molecule / Peptide SMILES | MIT / Academic | DeepChem / HuggingFace |

---

## 3. Compliance Notes for Journal Data Availability Review
1. **No CC-BY-NC Redistribution**: All CC-BY-NC derivative clinical summaries, FAERS counts, and clinical trial aggregations have been stripped from the public repository and REST API. The live Clinical & Regulatory PK panel serves exclusively US Federal Public-Domain FDA/DailyMed label facts with primary `dailymed.nlm.nih.gov` and `accessdata.fda.gov` URLs.
2. **Non-Commercial Research Scope**: Because several constituent academic corpora (DBAASP, PEPlife2, THPdb) and ESM-C 600M carry non-commercial academic terms, all trained weights and live web predictions are restricted strictly to non-commercial academic research (`internal_research_only`). Commercial deployment requires independent re-licensing of upstream data and encoders.
