# Operation-only cached q_end diagnostic (prepared, unexecuted)

Hypothesis: the actual stock RWKV7 parent can retain its direct answer path while
a small CE-only q_end head corrects the initial operation decision. PR #71's
direct=1/1 versus free-running=0/1 (Assert instead of Compare) motivates this
specific bottleneck test. Folded/weak #81–83 and full-slot axis MSE do not answer
this question. This is reused-development screening, never L0 qualification.

Deliverable: freeze the corpus digest
`cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`,
24 training and 12 reused development worlds; retain only their initial
`slot-001-operation` rows (24/12). Requests must contain empty accepted and
proposed prefixes, and exactly the permitted audience state. Gold stays in
separate targets. Cache detached CPU FP32 q_end/e plus option masks, ks and
typed fields; key on row IDs, token IDs, spans, canonical option order,
template/layout and parent/encoder identity, never labels or task-kind metadata.

Use the existing qend_axes head with CE only (axis loss=0), seed 20261006,
AdamW lr=.001, batch=2, 96 updates/eight full passes. Three paired conditions:
unchanged stock-LM planner (primary baseline), the exact step-0 q_end checkpoint
(secondary architecture/random initialization control), and its fixed step-96
checkpoint (treatment). Step-0 and trained heads share initialization and
optimizer recipe. Train-minus-stock is primary; train-minus-step0 is secondary.
No selection or tuning. Compose the
head only at the initial operation protocol boundary (empty prefix and exact
slot shape); all other calls use unchanged HFReadoutBackend. Evaluate both
through the normal Decider/planner, checked kernel and renderer on all 12 dev
worlds. Retain failures, wrong-task answers and false relations in denominators.

Exit proof: synthetic CPU tests cover cache/live parity, label exclusion,
boundary/delegation, lineage/checkpoint, backbone immutability and launch guards.
Future execution must report train/dev operation accuracy, useful answers,
downstream slots, kernel acceptance, clustered paired CI, stock-parent direct
retention/equality and cost $0. Freeze source, parent and encoder identities and
reject drift. CPU/offline only, two threads, >=12 GiB start and >=3 GiB running
memory, <=1200 seconds, authenticated owned child and cleanup on timeout.

Decision for this implementation task: prepare and dry-run only now. Do not load
real weights, fit, infer, use GPU, access heldout/Index, spend, or open a PR.
The standing goal authorizes CPU work once tiny-model proofs, cross-review and
dependency merges are complete and a fresh >=12 GiB start guard passes. This
plan grants no paid/GPU authority. A future completed run is
eligible for review only with strong operation fit, strict useful-answer gain,
no direct-path regression, complete paired evidence and unchanged backbone.
Weak fit diagnoses head learnability; strong operation fit without downstream
gain diagnoses later stock slots. Neither outcome licenses capability claims.

Preparation command (no weights):
`python -m language_growth.rwkv_operation_cache --out <fresh-dir> --dry-run`

Future execution command (after tiny proofs, cross-review/dependency merges and fresh resource guard):
`python -m language_growth.rwkv_operation_cache --out <fresh-dir> --execute`
