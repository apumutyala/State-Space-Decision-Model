# Paired-value supervision raw evidence

`registration.json` freezes the clarified six-template baseline, unchanged
model/config/base-world sampler, original training/validation row digests,
200 authored comparison swaps, finite per-kind/member exposure counts, and
static/random control definitions before fitting.

Four gzip archives preserve exact raw JSON bytes:

- `frozen_rows.json.gz`: unchanged 600 six-template base training rows, their
  200 additional swapped-value comparison members, and unchanged validation.
- `predictions.json.gz`: this condition's 1080 complete semantic predictions,
  typed programs, checker outcomes, checked renderings, and scoring flags.
- `baseline_predictions.json.gz`: saved six-template baseline predictions;
  the baseline was not refitted.
- `intervention_controls.json.gz`: 720 checked static-output/random-relation
  predictions on the same comparison worlds.

Checksums are in `raw_artifact_manifest.json`. Both receipts and
`paired_analysis.json` retain per-kind/paired CIs and intervention statistics.
Verify all archives, frozen row digests, authored swaps, all 2880 checker/render
outcomes and model per-kind totals with the standard-library command:

```text
python language_growth/receipts/paired_value_training_20261006/verify_raw.py
```

No training, model inference, downloads, or paid work occurs during verification.
See [the full receipt](../../paired_value_training_receipt.md). This is one
frozen iterative synthetic-development intervention, not official holdout,
parent retention, or evidence about other numeric heads or Track D parents.
