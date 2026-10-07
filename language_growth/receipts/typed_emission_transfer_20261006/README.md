# Disjoint-development transfer raw evidence

This directory preserves the frozen registration, result receipt, post-run
paired analysis, and unchanged raw JSON in two gzip archives:

- `predictions.json.gz`: all 1080 seed/task predictions, complete semantic
  targets, programs, kernel outcomes, checked renderings, and scoring flags.
- `frozen_rows.json.gz`: exact training and disjoint-validation inputs,
  permitted evidence, scorer-only reference programs, and pair identities.
- `raw_artifact_manifest.json`: compressed and decompressed checksums and sizes.

To verify hashes and independently replay every kernel check and checked
rendering from the committed evidence, run from the repository root:

```text
python language_growth/receipts/typed_emission_transfer_20261006/verify_raw.py
```

This uses the Python standard library and the repository's kernel/renderer.
It performs no model training, inference, downloads, or paid work.

The model source was copied unchanged from PR #80. This condition trains on
`development_train` and evaluates on `development_validation`, with disjoint
names, ranges, and templates, plus preauthored alternate read/explain wording.
It is iterative synthetic development, not an official holdout or a parent
retention test. See [the full receipt](../../typed_emission_ssm_transfer_receipt.md).
