# Documentation Index

This repository is a public-safe release subset. The current cross-project status is maintained in [`PROJECT_PROGRESS.md`](../PROJECT_PROGRESS.md); frozen dated reports remain unchanged as evidence snapshots.

## Current

- [`peptide_omnipanel_v31_latest_model_architecture_training_results_20260723.md`](peptide_omnipanel_v31_latest_model_architecture_training_results_20260723.md): current V31 architecture, input contract, training method, grouped-OOF results, and limitations.
- [`peptide_omnipanel_v31_peptide18_detailed_report_20260722.md`](peptide_omnipanel_v31_peptide18_detailed_report_20260722.md): frozen V31 a3 detailed report.
- [`peptide_omnipanel_v31_peptide18_documentation_manifest.json`](peptide_omnipanel_v31_peptide18_documentation_manifest.json): SHA-bound V31 report and evidence inventory.
- [`peptide_omnipanel_v31_peptide18_verification.json`](peptide_omnipanel_v31_peptide18_verification.json): engineering verification for 18 dynamic models; `validated_model_count=0`.
- [`peptide-cyclic-peptide-property-expansion-v15.md`](peptide-cyclic-peptide-property-expansion-v15.md): upstream peptide evidence collection, processing, licensing, and training gates.
- [`peptide-ml-cleaning-v1.md`](peptide-ml-cleaning-v1.md): ML-cleaning semantics, deduplication, numeric/binary/PU lanes, and review queues.

## Web Evidence

- [`peptide_omnipanel_v31_web_history_and_evidence_20260723.md`](peptide_omnipanel_v31_web_history_and_evidence_20260723.md): four-page UI and historical browser E2E evidence.
- [`peptide_omnipanel_v31_web_deployment_20260723.md`](peptide_omnipanel_v31_web_deployment_20260723.md): historical deployment snapshot; it does not imply that the service is currently online.
- [`peptide_omnipanel_v31_web_batch_chinese_upgrade_20260723.md`](peptide_omnipanel_v31_web_batch_chinese_upgrade_20260723.md): batch input and Chinese UI upgrade.

## Superseded Or Limited Evidence

- V28's 39-endpoint display mixed local models, transferred teachers, and transparent priors; V31 replaces it with 18 peptide-domain endpoints.
- V29 completed strict ESM2-8M held-out experiments, but no model passed the preregistered promotion gate.
- V30 failed the peptide-only scope audit because its cases included substantial small-molecule teacher transfer; those outputs are quarantined and not published here as current evidence.
- V31 a2 was invalidated by cell-penetration modality/source leakage; only a3 is current.
- `release`, `pass`, and 18 dynamic outputs must always be reported together with `internal_research_only` and `validated_model_count=0`.

## Maintenance Rules

1. Update current status only in `PROJECT_PROGRESS.md`.
2. Keep dated reports and machine-readable manifests frozen.
3. State whether a result is a data assertion, bundle check, development OOF, source-held-out test, or independent external validation.
4. Do not publish model weights, source-restricted raw rows, user sequences, browser histories, runtime state, or credentials.
