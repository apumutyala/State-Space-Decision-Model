# Synthetic development data v2

`development_train` is the only split intended for supervised typed-choice
training. PR #17 may read `development_validation` for iterative free-running
development checks and may also inspect training examples for debugging. Both
splits are development data. Neither is an untouched audit or test set, and
results on either do not establish transfer beyond these synthetic domains.

The split domains use disjoint entity names, task templates and value ranges.
Comparison tasks vary which record is named first and whether price or warranty
months is compared. The manifest for each split records the exact sample hash,
field/relation/operand counts, declared domain and seed. Run `python
receipts/2026-10-06-L-development-data-v2/reproduce.py` from the repository
root to recreate both JSONL files and manifests. The source and dependency
commits used for this receipt are pinned in `receipt.json`.
`python receipts/2026-10-06-L-development-data-v2/check_adapter.py` verifies
one exact generated v0.1 request against Track D's merged CPU oracle; its
summary is `adapter-v01-check.json`.

The older `receipts/2026-10-06-L-development-data/` sample is retained only as
historical evidence: its request envelope was v0 and is rejected by the current
Track D v0.1 adapter. No model, GPU, teacher or paid service was used here.
