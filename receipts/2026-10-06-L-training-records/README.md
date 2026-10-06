# CPU language training-record compatibility

Reproduce by extracting `git archive <sha> <paths>` into a fresh directory, in the order in `receipt.json` (an empty paths list means the full source tree). Copy this receipt's `integration_check.py` into that directory. Run the two commands in the receipt there, with the recorded Python environment or one with the same Track D dependencies installed.

The integration uses the actual Track D `items_from_records`, `ByteEncoder` and synthetic-only guard. It reads only freshly generated development worlds. `raw.jsonl` includes every exported request and label, the development manifests and each resulting option-index mapping/token hash. Earlier gold slot choices are teacher-forced context; current labels remain separate. This is data compatibility, not a trained model, a free-running result or proof of benchmark disjointness for arbitrary imported data.

No training step, weight load, GPU or paid compute was used. Dependencies from unmerged PRs are explicit; do not launch training from this composition until the appropriate proofs are merged and the spend gate is acknowledged.
