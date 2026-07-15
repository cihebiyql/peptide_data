# Peptide ML Cleaning v1 Statistics

This deterministic statistical release covers all 50 raw endpoints in the frozen V15 training-deduplicated table (215,568 rows) and all declared ML-clean endpoint families, including the zero-row Kp family.

- ML-clean representatives: 19,979
- Strict numeric: 11,591
- Binary evidence catalog: 3,930
- Positive-unlabeled: 920
- Censored: 230
- Literature review is reported separately: 1,560

Never pool `normalized_value` across task kinds or incompatible units. Use `strict_numeric_task_distribution.tsv` for model-facing numeric diagnostics. Endpoint-unit summaries are descriptive only. Sequence projections, selected model representations, and identity groups are reported separately.
