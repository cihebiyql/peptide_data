# Data Dictionary (`DATA_DICTIONARY.md`)

> **Notice (v33.6)**: Raw row-level third-party TSVs previously located under `data/peptide_ml_cleaning_v1/` are NOT distributed in this public repository (see `DATA_LICENSES.md` and `NOTICE`). This dictionary documents the retained aggregate metadata files and the authoritative `v33_manifest.json` schema.

1. **`v33_upgrade/bundle/v33_manifest.json`** — Authoritative machine-readable audit ledger for all 47 endpoints (`dual_modality` and `smiles_only` CV/holdout metrics, 1,000× cluster-block bootstrap 95% CIs, CI-aware `evidence_grade`, conformal calibration coverage & `uq_reliability`, `egat_fidelity_audit`, `external_baselines_same_split`, `temporal_split_benchmarks`, `recent_fda_2019_2023_blind_set`, and `case17_scorecard.non_circular_separation`).
2. **`v33_upgrade/web/clinical_evidence_index.json`** — 59 marketed FDA-approved peptide drugs with 428 factual numeric PK records and 53 live-verified NLM DailyMed SPL `setid` links (`dailymed.nlm.nih.gov`) and Drugs@FDA links (`accessdata.fda.gov`).
3. **`data/peptide_ml_cleaning_v1_statistics_v1/`** — Aggregate corpus histograms, amino-acid composition, censoring distributions, and endpoint inventories (no row-level third-party records).
4. **`data/peptide_ml_cleaning_v1/{exclusion_summary.tsv,task_registry.tsv,manifest.json}`** — Aggregate cleaning counts and task semantic registry only.
