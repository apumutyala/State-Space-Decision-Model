# Train-only wording diversity evidence

- `registration.json`: frozen model/source/data hashes, exact six training
  templates per operation, assignment counts, and unchanged validation digest.
- `predictions.json.gz`: this condition's 1080 raw predictions, semantic
  targets, typed programs, kernel outcomes, checked renderings, and scores.
- `frozen_rows.json.gz`: all 600 modified training inputs and 360 exactly
  reused disjoint validation members, evidence, labels, and pair identities.
- `baseline_predictions.json.gz`: unchanged PR #81 predictions, reused for
  paired comparison without refitting the baseline.
- `receipt.json`, `baseline_receipt.json`, `paired_analysis.json`: model
  results, baseline reference, per-kind paired CIs and intervention results.
- `raw_artifact_manifest.json`: exact compressed/decompressed checksums/sizes.

Run this standard-library verifier from the repository root:

```text
python language_growth/receipts/typed_emission_diversity_20261006/verify_raw.py
```

It checks every archive hash, the reused validation-row digest, all 2160
baseline/current kernel outcomes and checked renderings, and per-kind results.
It performs no training, model inference, downloads, or paid work.

See [the receipt](../../typed_emission_ssm_diversity_receipt.md) for model/data
provenance and uncertainty. This is one frozen iterative synthetic-development
condition, not official holdout or parent-retention evidence.
