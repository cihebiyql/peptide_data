# Peptide ML Cleaning v1

This release implements cleaning steps 1-6 over the V15 training-deduplicated candidate table: semantic tasks, normalized observation deduplication, safe identity/model representations, unit normalization, censoring sidecars, and binary evidence lanes. It does not create splits and does not promote literature staging.

## Counts

- Input in-scope rows: 20,052
- Normalized representative rows: 19,979
- Strict numeric research candidates: 11,591
- Censored numeric sidecar: 230
- Binary evidence catalog: 3,930
- Positive-unlabeled: 920
- Identity/normalization/semantic review: 3,308
- Literature review retained: 1,560; direct training rows: 0

`strict_numeric.tsv` is the only point-regression candidate table. Binary rows remain physically separated by evidence lane. `task_registry.tsv` declares one semantic signature per task. Future split work must use identity and source groups; no split is fabricated here.
