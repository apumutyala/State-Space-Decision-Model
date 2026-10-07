# L1 multiworld training-distribution preparation

2026-10-07. This replaces the training API's one-world plumbing smoke cohort
with 32 complete worlds, while preparing 32 disjoint development-validation
worlds for the existing evaluator. It is synthetic data preparation, not
training, inference, learned L1 capability or generalization evidence. Local
Windows CPU; cost $0. No model/weights, optimizer, GPU, paid compute, Decision
Index, audit corpus or heldout data was accessed. Future training still needs
its separate prerequisites and contamination process; no suite data was read
to create these synthetic premises.

Branch: `codex/l1-training-world-distribution`, based exactly on PR #132 head
`427a3f3b20473669647832e87027d9e3df1c8e87`. The generator is the modified
`language_growth/multirecord_data.py`, SHA-256
`e3470d2b61ff2b31790b5ee70dac0528cc94fd8647325c24bd6ba3e1311da14f`,
Git blob `c87c835b5094ff2d8ae23fc50416f5c9f851d1c8`. The base commit is
ancestry only; it does not contain the new split identities. The commit
containing this receipt pins that generator blob and all four source files.
`verification.json` records generator provenance, source/export hashes,
observed distributions and overlap counts. Exported rows and their export
manifest retain the exact generator hash/blob and the base commit separately.

Both cohorts use seed 1707 and mixed choice formats. The new split names are
`l1_multiworld_development_train` and
`l1_multiworld_development_validation`. Their name pools, task templates and
numeric ranges retain the original two disjoint development domain specs;
new split identities enter the deterministic RNG and produce fresh worlds.
The original one-world data directory and training-export receipt are
unchanged. Its `l1_development_train` name is no longer training-allowlisted.
Only the new train sample and manifest bytes can pass L1 training export.
The new train split is excluded from the evaluation loader; only the new
validation split is added there. Validation rows, including relabelled rows
presented with the train manifest, fail closed in the training exporter.

| Source file | Bytes | SHA-256 |
| --- | ---: | --- |
| `l1_multiworld_development_train.jsonl` | 22,244,672 | `175cf7cc2d7c39a8c931a3a452149be7c141db712bde81a0ce7f606b0a27e9f4` |
| `l1_multiworld_development_train-manifest.json` | 97,133 | `eaa38a852dab840a51175a12a1a58a74539338f3ee119088a3442f054d0586e7` |
| `l1_multiworld_development_validation.jsonl` | 22,362,224 | `a25c938eba725611a74ba9ff9dac4b83ed4e28bb00a2bc50f9e76c1202c85339` |
| `l1_multiworld_development_validation-manifest.json` | 103,747 | `140b0f7f577f23fd98090c583c1fdb3d5adc2cf7d1a13f626700724f55657870` |

Each cohort has 32 unique world/catalog/context/row hashes, 1,280 distinct
tasks and 4,096 authored slots. Every world has four distinct entity records,
eight reads, eight explanations and 24 comparisons. Each field has 20 tasks
including 12 comparisons: all 12 ordered distinct-record pairs, arranged into
six unordered pairs with both orders. Per field there are two equality,
five less-than and five greater-than comparisons; each field's four values
contain exactly one equal pair and three distinct values. Every record-field
is read and explained. Each world has 20 production and 20 augmented tasks.
Every world has 40 operation, 16 fact, 24 left, 24 right and 24 relation slots.

Per cohort these totals are 256 reads, 256 explanations, 768 comparisons,
640 tasks per field, 384 comparisons per field, 128 equal/320 less-than/320
greater-than comparisons, and 640 tasks per choice format. Slot totals are
1,280 operation, 512 fact, and 768 each of left/right/relation. All eight
fact candidates and all three relations remain available in each applicable
request. The relation distribution is deliberately paired and symmetric;
it is not uniformly balanced across the three relations.

The train name pool has **six names**, each observed at all four record refs
across the worlds (24 distinct name/ref bindings). Its four record-price
positions each have 32 distinct values, and record-warranty positions have
17/18/19/20 distinct values. Validation also has six names and 24 bindings;
its record-price positions each have 32 distinct values and warranty
positions have 18/18/18/17. Train declared ranges are USD 300.01–450.00 and
61–84 months; observed ranges are USD 300.23–448.07 and 61–84 months.
Validation declared ranges are USD 450.01–600.00 and 85–108 months; observed
ranges are USD 450.88–595.13 and 85–108 months. Train/validation overlap is
zero for names, values by field, task IDs, world IDs and catalogs; templates
and numeric intervals are disjoint from one another and both L0 domains.
Record refs, fields, units and operation vocabulary are intentionally shared.

These 32 worlds provide changing evidence bindings with modest artifact size
(about 22 MB per source cohort). They remain a narrow six-name, fixed-template,
four-record development distribution. Task order is fixed across worlds and
is authored scaffolding; task index, task/world IDs, split identity and labels
are excluded from model requests. Prefixes contain exactly earlier target
slot IDs, option IDs and offered descriptions. The current label remains
separate. World/catalog/source hashes are metadata outside requests. This
does not establish unseen-name/template, adversarial, free-running, comparator,
retention or causal-intervention performance. Learned L0 gates remain required.

Commands run from the worktree (use fresh output paths when reproducing):

```text
python -X utf8 receipts/2026-10-07-L1-training-world-distribution/reproduce.py --out C:/Users/apumu/research/CHLabs/ssdm-data/l1-world-distribution-reproduced-20261007
python -X utf8 -m language_growth.multirecord_training --out C:/Users/apumu/research/CHLabs/ssdm-data/l1-training-world-distribution-20261007
python -X utf8 -m pytest language_growth/tests/test_multirecord_distribution.py language_growth/tests/test_multirecord_training.py language_growth/tests/test_training_records.py language_growth/tests/test_multirecord_eval.py language_growth/tests/test_multirecord_data.py language_growth/tests/test_kernel.py language_growth/tests/test_kernel_adversarial.py -q
python -X utf8 -m pytest -q tests/test_jobs_lint.py
git diff --check
```

The reproduction script compared all four regenerated files to the committed
source bytes. The external training export is 4,096 rows / 21,381,960 bytes,
SHA-256 `a7806d12672be658f66f75d57a0d53a888cb00b93074b183a0a7c5bceb52d489`.
Its 98,003-byte export manifest SHA-256 is
`a4367fa4f975a585a85176709e408ee2fa3b4f3ac6c069476482a3b1363041af`.
The complete source manifest preserves per-world IDs and hashes.

Focused CPU result: 53 passed, 54 subtests passed. Repository lint: 67 passed.
The initial focused run had one test-only tuple/list equality failure when
comparing an in-memory split spec to decoded JSON. Canonical JSON comparison
fixed the assertion; regeneration and file hashes did not change. Tests check
every train reference program, exact prefixes, current-label/identity metadata
separation, complete worlds, field/kind/relation support, changing bindings,
split rejection, direct-call/file pins, and export write-failure preservation.
An explicit comparison against the exporter loaded from exact #132 base
showed identical seed17/count24/mixed L0 output (24 tasks, 64 records), canonical
list SHA-256 `93a221534d4ebbe7cea4b1c5dc9b8950a547d31ac4f68a7a0fea0fa8ac18e1c6`.
The existing parity regression also retains that hash from #128.

Self-adversarial review checked the fixed task schedule, small name pool,
potential label/identity leakage, asymmetric relation counts, overlapping
record refs, source ancestry attribution, and validation relabelling bypass.
The listed boundaries are explicit, identity metadata stays outside requests,
and exact file/manifest pins reject changed or substituted cohorts. Independent
exact-head review and Track D/Claude cross-review remain pending; no PR or merge
is authorized by this receipt.
