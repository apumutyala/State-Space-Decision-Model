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
slot shape); all other calls use unchanged HFReadoutBackend. Evaluate all arms
through the normal Decider/planner, checked kernel and renderer on all 12 dev
worlds. Retain failures, wrong-task answers and false relations in denominators.

Exit proof: synthetic CPU tests cover cache/live parity, label exclusion,
boundary/delegation, lineage/checkpoint, backbone immutability and launch guards.
Future execution must report train/dev operation accuracy, useful answers,
downstream slots, kernel acceptance, clustered paired CI (seed 9203, 2000 draws),
aggregate and predeclared read/explain/compare CIs, read non-regression, stock-parent direct
retention/equality and cost $0. Freeze source, parent and encoder identities and
reject drift. Checkpoint lineage additionally binds a separate target hash;
feature identities and feature-cache metadata exclude that target hash.
CPU/offline only, two threads, >=6 GiB start and >=3 GiB running
memory, <=1200 seconds, authenticated owned child and cleanup on timeout.

The 6 GiB start floor follows #71's measured same-parent CPU run2: start
available memory 7,554,498,560 B, minimum 6,093,643,776 B, with a 6 GiB start
requirement. The additional detached operation cache is approximately 442,368 B;
the parent is shared and feature extraction retains the one-row path, with stock
readout capped at two rows and 2048 tokens. Repeated longer sequences, high-water
memory and unrelated host processes remain risks; keep the 3 GiB running guard,
two threads, owned child and 1200s cap. This is a measured start floor, not a
promise that the complete three-arm run fits or completes.

Decision for this implementation task: prepare and dry-run only now. Do not load
real weights, fit, infer, use GPU, access heldout/Index, spend, or open a PR.
The standing goal authorizes CPU work once tiny-model proofs, cross-review and
the #71/#85/#110 dependency gates are complete and a fresh >=6 GiB start guard passes. This
plan grants no paid/GPU authority. A future completed run is
eligible for review only with strong operation fit, strict useful-answer gain,
no direct-path regression, complete paired evidence and unchanged backbone.
Weak fit diagnoses head learnability; strong operation fit without downstream
gain diagnoses later stock slots. Neither outcome licenses capability claims.

Preparation command (no weights):
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_operation_cache --out <fresh-dir> --dry-run`

Use this exact existing interpreter for preparation, testing and execution.
Its Torch wheel is `2.8.0+cu128`; execution is CPU only with
`CUDA_VISIBLE_DEVICES=-1`, two CPU threads and CUDA availability required false.
The registration pins interpreter, Python and package versions strictly.

Future execution command (after tiny proofs, cross-review/dependency merges and fresh resource guard):
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_operation_cache --out <fresh-dir> --execute`
