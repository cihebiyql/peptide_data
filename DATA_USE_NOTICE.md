# Data Use Notice (`DATA_USE_NOTICE.md`)

This public repository distributes **Peptide OmniPanel 1.3 (`v33.6`)** source code (MIT License), audited evaluation manifests, reproducible audit scripts, aggregate corpus statistics (`data/peptide_ml_cleaning_v1_statistics_v1/`), and factual NLM DailyMed / US FDA regulatory PK pointers (`v33_upgrade/web/clinical_evidence_index.json`).

- **No Raw Third-Party Observation Tables**: Row-level third-party observation tables (`data/peptide_ml_cleaning_v1/*.tsv`) are intentionally NOT redistributed in this repository. Researchers wishing to rebuild training corpora must obtain upstream datasets directly from their official portals under their respective licenses documented in [`DATA_LICENSES.md`](DATA_LICENSES.md).
- **Non-Commercial Academic Research Scope**: Trained model weights and hosted web service predictions are strictly restricted to non-commercial academic and early preclinical research (`internal_research_only`). See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for full legal terms.
