# V2.8 Research Panel Public Sync Notice

This commit publishes the **source code, endpoint/source registries, compact
provenance manifests, tests, and Chinese research documentation** for the
V2.8 39-endpoint peptide/cyclic-peptide research panel.

## Intentionally excluded from this public Git repository

- raw TDC/Dataverse downloads and normalized observation tables;
- the 25 local research model weights (`*.joblib`);
- generated prediction JSON and any development-only OOF/run outputs;
- the wider local workspace and caches.

Those assets remain `internal_research_only` because source-level license
clearance is incomplete and because the research heads are not validated for
peptide/cyclic-peptide deployment. Their immutable filenames, source URLs,
row counts, and SHA-256 values remain recorded in the included manifests and
research-coverage documents.

## Scope and safety contract

Every V2.8 research-panel output must be displayed with:

```text
research_only=true
low_confidence=true
status=predicted_low_evidence
```

No model is promoted to A/B or to an active service by this publication.  In
particular, the 17 TDC routes are small-molecule-to-peptide pseudo-label
transfer students, not peptide measurement models.  Four added endpoints
without honest sequence-supervised labels remain transparent E-tier priors.

## Rebuilding an internal research bundle

After confirming each source's terms and downloading the public raw inputs,
run the V2.8 collectors and trainer locally.  The resulting raw observations
and `*.joblib` weights must not be committed here unless a separate license,
security, and release review explicitly clears them.
