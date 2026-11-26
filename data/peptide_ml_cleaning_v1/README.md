# Aggregate Metadata Only (Raw Third-Party TSVs Excluded)

Row-level third-party observation tables previously generated during cleaning (`normalized_observations.tsv`, `strict_numeric.tsv`, `censored_observations.tsv`, `binary_evidence_catalog.tsv`, `condition_conflicts.tsv`, `exact_duplicates.tsv`, `literature_review_cleaned.tsv`, `positive_unlabeled.tsv`, `review_observations.tsv`) are intentionally excluded from this public repository and purged from git history to respect upstream third-party dataset licenses (see `DATA_LICENSES.md` and `NOTICE`).

Only aggregate metadata (`exclusion_summary.tsv`, `task_registry.tsv`, `manifest.json`) and aggregate corpus statistics in `data/peptide_ml_cleaning_v1_statistics_v1/` are distributed here.
