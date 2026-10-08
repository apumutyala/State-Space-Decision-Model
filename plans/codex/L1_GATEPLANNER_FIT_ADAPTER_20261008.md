# L1 fit-cohort GatePlanner supervision adapter

2026-10-08. Local CPU, $0. This adapter replays the pinned L1 multi-record
training cohort through the current `GatePlanner`, replacing the legacy
hand-authored slot menus as a source of semantic-slot supervision. It does not
train a model or establish learned language capability.

The source JSONL and manifest are pinned by `L1_TRAINING_ALLOWLIST`; the
whole-world partition is rechecked against fit hash
`34909156035594729d7891d64c840efb8ed853dfe4fcb10a56f77b0fa8e91be0`,
calibration hash
`f58d51a51cdb6621d9b7ae86d348d67406488e13bb0c6839c174aaa5c47dce28`, and
partition manifest hash
`3ad9f962cc6896c4540a13633ea371a00d8537c46d061813b66188e43920b0ec`.
Only the 24 fit worlds enter GatePlanner replay. Calibration stays reserved;
the separate development-validation and language-heldout inputs are not read
by the adapter.

The export contains 960 complete tasks and 3,072 actual GatePlanner slot rows
from 24 worlds. Each world contributes 8 reads, 8 explanations and 24
comparisons. Reads and explanations cover all four records and both fields;
comparisons cover every ordered distinct-record pair for both fields, with
equal/greater/less counts of 2/5/5 per field and world. The planner creates
the offered option menus, and the current label is stored outside its request;
only earlier gold choices form each diagnostic teacher-forced prefix.

## Artifact

The current frozen local export is
`C:/Users/apumu/research/CHLabs/ssdm-data/agent-c-l1-gate-training-fit-20261008-v3`.
It contains 3,072 JSONL rows (16,579,126 bytes). Version 2 is preserved; v3
adds target-description frequencies to the manifest, while the JSONL and gzip
record bytes remain identical. SHA-256 values:

- `training_records.jsonl`: `02345dfb60c92742e137d96aa9e34576d04a77181dfc4aa0ac5d308730cb4ccd`
- `training_records.jsonl.gz`: `d767627b781596f5f1e2882f2ed2e0b69587c055b3522117d165332af2d859ef`
- `manifest.json`: `0c08fd0da13a175f6a66107e728ee27eff9a69c6d7ff115a072be18938f694c5`

Reproduce to a new output directory with:

```powershell
python -B -X utf8 -m language_growth.l1_gate_training --out C:/Users/apumu/research/CHLabs/ssdm-data/l1-gate-training-fit-reproduction --model gate-training --parent-revision training-only-v1
```

The focused tests passed (2 tests), including exact replay through GatePlanner,
fit/calibration disjointness, request/label separation, complete slot and
semantic coverage, deterministic artifact bytes, refusal to overwrite, and
compatibility with the development-only trainer record loader and whole-world
split. `py_compile` and `git diff --check` passed. No model, weights, optimizer,
GPU, paid service, Decision Index, development-validation input, or held-out
data was used. The records are supervision plumbing only; free-running
usefulness, parent decision retention, adversarial soundness and semantic
intervention gates remain open.

## Exact RWKV7 token census

After the export was frozen, a CPU-only census tokenized all 3,072 unique
record/slot rows with the pinned RWKV7-G1k-13.3B tokenizer revision
`ce3630b854cedbd673569f4600c6cf744bca9dba`, the trainer's exact
`RwkvSegmentEncoder`, ordered `heads.layout.build_row`, and template
`jrt-rwkv-g1-v1`. The fit-only receipt is
`C:/Users/apumu/research/CHLabs/ssdm-data/agent-c-l1-fit-token-census-rwkv7-20261008/summary.json`
(SHA-256 `a467706aca9b2365ecb3044ffd58a844d6ad35d713719e2165d03f5864ceeb13`).

Prompt lengths range from 1,092 to 1,406 tokens; the longest prompt plus
continuation is 1,410. All 3,072 rows exceed 1,024 tokens and fit under 2,048;
max options is 8 and all continuations are 4 tokens. In the checked-in
`m2_first_cell_rwkv7_d1_fast.json`, the ladder is 512/1,024/2,048 with K=16,
so the 2,048 rung is required to retain the complete fit cohort. Its
`min_train_items=5000` also exceeds this export's 3,072 rows. This data cannot
be inserted into that D1fast configuration unchanged; any later C training
run needs its own parent-pinned, reviewed row-count and T/K configuration.

The census loaded only this fit export and tokenizer/template/trainer code; it
did not load Torch, model weights, validation or held-out inputs. Its first
discovery search was broader than its assigned code-file allowlist; the
auditor reported that deviation, then restricted census reads to the fit
artifact and exact encoder/tokenizer sources.

## Fit-only target distribution

The v3 manifest pins the exact offered-description counts by slot under
`target_counts_by_slot` (SHA-256
`137789a613387eddc8edc6c55cc651091f1b80255a2e54298e74d0187344bd3c`). Across
the 24 fit worlds, operation targets are ReadFact 192, ExplainDerivation 192,
and Compare 576 (20/20/60%); comparison relations total equal 96, greater-than
240, and less-than 240 across both fields, or 48/120/120 per field. Fact targets
are split 192/192 across price and warranty; each operand side is 288/288.
Record positions 1–4 each receive 96 fact, 144 left-operand and 144
right-operand targets.

Entity-count differences are proportional to the number of fit worlds in
which each entity is present: conditional on presence, each entity gets two
fact targets per field and three targets per field on each comparison side.
This audit gives no evidence to reweight classes. Preserve the exhaustive
source task mix for the first model run; consider a bounded reweighting
ablation only if a measured development failure justifies it. Option IDs are
local to each slot and must not be pooled across slots.
