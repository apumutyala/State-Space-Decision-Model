# Typed emission SSM raw evidence

`predictions.json.gz` contains the unchanged raw predictions for all three
fixed seeds: 1080 observations including complete predicted/gold semantic
targets, typed programs, kernel results, checked renderings, cluster identities,
and scoring flags. It preserves the original JSON bytes, including line
endings. Its decompressed SHA-256 equals `receipt.json`'s
`raw_predictions_sha256`:

`69392556a2f2f786e5126f5a8d0e02488882815b56966f5f3f191d52ed183e30`

`raw_artifact_manifest.json` records the compressed checksum, original checksum,
byte counts, seed count, and observation count. To verify from the repository
root using Python's standard library:

```python
from pathlib import Path
import gzip, hashlib, json

root = Path("language_growth/receipts/typed_emission_ssm_20261006")
receipt = json.loads((root / "receipt.json").read_text())
manifest = json.loads((root / "raw_artifact_manifest.json").read_text())
archive = (root / "predictions.json.gz").read_bytes()
assert hashlib.sha256(archive).hexdigest() == manifest["compressed_sha256"]
raw = gzip.decompress(archive)
assert hashlib.sha256(raw).hexdigest() == receipt["raw_predictions_sha256"]
assert len(raw) == manifest["decompressed_bytes"]
predictions = json.loads(raw)
rows = [row for seed_rows in predictions.values() for row in seed_rows]
assert len(rows) == manifest["observation_count"] == 1080
for metric in ("target_exact", "kernel_accepted", "whole_answer_correct"):
    assert sum(row[metric] for row in rows) == receipt["pooled"]["all"][metric]
```

See [the diagnostic receipt](../../typed_emission_ssm_receipt.md) for the model,
candidate construction, paired confidence intervals, data provenance, and
random-initialization/no-parent limitation. Adding this raw archive did not
rerun training or change results. Local checkpoints and frozen input rows
remain at the artifact path recorded in that receipt.
