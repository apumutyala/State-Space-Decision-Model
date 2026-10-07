# L1r0 relation data receipt

Data-only CPU preparation from `agent-main@32d975159c64216193c47478e102d17c8d7e5658`.
The original preparation and `source-sha256.json` describe commit `563aacb`;
they are preserved unchanged after the source-verification follow-up.
No Torch, checkpoint/model load, inference, GPU, paid service, Index or heldout
access. Cost $0. There are no predictions or performance claims.

`preparation.json` records exact source-data, partition, normalized-content,
selected relation-row and target-sidecar hashes, plus every row's token count.
`source-sha256.json` pins the adapter, focused tests, plan and dependencies.
Fixture seed 1707; existing fixed world-assignment seed 1708; no new random seed.

Commands (repository root, local Python 3.12.13):

```
python -m unittest language_growth.tests.test_l1_relation_data -v
python -X utf8 -m pytest -q language_growth/tests/test_l1_relation_data.py tests/test_jobs_lint.py
python -m language_growth.l1_relation_data --out receipts/2026-10-07-L1r0-relation-data/preparation.json
```

Focused tests: 10 passed. Focused tests plus launch lint: 80 passed, 8 subtests
passed. This is a deterministic census, one run; timing and statistical
intervals are not capability measurements and are not reported.

| Partition | Complete worlds | All checked rows | Relation rows | Maximum prefix / call tokens |
|---|---:|---:|---:|---:|
| Fit | 24 | 3,072 | 576 | 1,317 / 1,321 |
| Reserved calibration | 8 | 1,024 | 192 | 1,309 / 1,313 |
| Teacher-forced conditional validation | 32 | 768 | 768 | 1,158 / 1,162 |

The validation prefix contains exactly operation/left/right authored choices
and descriptions; the relation target is only in a distinct evaluation
sidecar. Normalized state/questions contents have zero overlap across full
fit, full calibration and conditional validation. Label and evaluator-sidecar
mutation leave requests and encoded prompt identities unchanged.

The tokenizer census reads only the pinned vocabulary, identified by its
SHA-256, model ID and snapshot revision in the receipt. It uses the serving
`jrt-rwkv-g1-v1` template, ordered layout, segment-wise encoding and complete
continuations. Nothing was truncated or dropped. Per coordinator direction,
real Item conversion and exact runtime LayoutRow acceptance are deferred to
the guarded L1r1 runner: it must pass the active backend template and encoder,
convert all 3,072 pinned fit records before selecting the 576 relation Items,
and recheck row lengths before feature extraction. R0's dispatch test mocks
the Torch-dependent consumer and proves ordering only.

Review follow-up: verification now derives selected records and validation
targets from the independently pinned full-world/raw-source data instead of
trusting mutable manifest hashes. Thirteen focused CPU tests pass, including
paired data/hash edits and post-load source drift. The original source hashes,
preparation manifest and complete vocabulary-only census reproduced exactly;
no token or data result changed. LESSONS L-37 records the correction.
