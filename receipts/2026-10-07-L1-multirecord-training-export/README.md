# L1 training-export data-plumbing receipt

2026-10-07. Exact stacked source: PR #128 head
`4538e9d72dfc587221aba096b91ca60f1bca76d9`. Fixture source:
`bd31c9051732ddfa325b17f6ad986c2d08574f49` (PR #127).
Branch: `codex/l1-multirecord-training-export`. Local Windows CPU; cost $0.
No model, weights, optimizer, GPU, paid service, Decision Index, audit or heldout
data was accessed. Model training, L1 capability and free-running performance
are not claimed. Strong comparator, decision retention, adversarial soundness,
causal intervention and learned L0 gates remain required.

Only the committed seed-1707, mixed-format, one-world, 40-task
`l1_development_train` cohort is allowed. Its exact JSONL SHA-256 is
`2d455c7ad7e2a1116d026bb5801dec0c821de24602375ccecf66b2569011f007`;
manifest SHA-256 is
`c5d94f90e4c41daab59fedb6745ecac758bce48b2724acfa7ca13e1dc47e9e2d`.
The bridge rejects changed bytes even with a replacement manifest, verifies
every typed reference program against its host/catalog, and routes through the
existing `export_training_records` function's explicit L1 allowlist.

All 128 exported requests copy authored questions/options and permitted public
evidence. The only addition is the exact sequence of prior target slot IDs,
option IDs and their offered descriptions. Current targets remain separate
labels. Task/world IDs and source manifest/row/world/catalog hashes are metadata
outside requests. The output manifest preserves the complete source manifest.

Exact export command, run from this worktree:

```text
python -m language_growth.multirecord_training --out C:/Users/apumu/research/CHLabs/ssdm-data/l1-multirecord-training-export-20261007
```

The external local output is preserved at that path. To reproduce elsewhere,
use a fresh output directory; output bytes are deterministic. Output hashes:

- `training_records.jsonl` (632544 bytes, 128 rows):
  `bf251a42c177a84cf0bc0eaa03f1396821c59ed0a67efdc22cb74f07ec9b0796`
- `export-manifest.json` (4206 bytes):
  `2e1fe53b3cca6498c6ab9af00ac2b1f007d7637375cc525f1fce44358a561f0b`

Exact successful verification commands:

```text
python -X utf8 -m pytest language_growth/tests/test_multirecord_training.py language_growth/tests/test_training_records.py language_growth/tests/test_multirecord_eval.py language_growth/tests/test_multirecord_data.py language_growth/tests/test_kernel.py language_growth/tests/test_kernel_adversarial.py -q
python -X utf8 -m pytest -q tests/test_jobs_lint.py
git diff --check
```

Focused result: 46 tests, 47 subtests passed. Repository lint: 67 passed.
The focused tests verify all 40 program checks, every slot's exact prefix and
separate label, unchanged raw input, all lineage hashes, direct-call and file
pin enforcement, non-train split rejection and fail-closed checker behavior.
Artifact publication stages both files in a unique sibling directory before
renaming to the final path. Injected manifest-write failure leaves no final
path or staging debris; a successful retry publishes the complete pair and a
duplicate-output retry preserves the published bytes.
L0 output for seed17/count24/mixed is exactly equal to the exporter fetched
from the pinned #128 base (24 tasks/64 records); its canonical-list SHA-256 is
`93a221534d4ebbe7cea4b1c5dc9b8950a547d31ac4f68a7a0fea0fa8ac18e1c6`.
Two initial test invocations named nonexistent bridge test files and collected
no tests; the exact successful commands above replace those attempts.

`verification.json` records implementation and output hashes. Independent
adversarial review and Claude cross-review status are recorded on the PR.
No merge is authorized by this preparation.
