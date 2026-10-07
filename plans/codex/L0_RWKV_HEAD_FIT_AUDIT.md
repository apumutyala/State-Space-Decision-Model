# Frozen RWKV head fit versus transfer diagnostic

Status: preparation only. A real checkpoint load requires a separately authorized
`--execute` invocation. Base `5499fe754bf2c3d9fc43a87e07890f5df6845fb8`;
prior experiment source `d8b3be06b9f4e8f142d86c082ce9e69818a41d8f`.

The prior recipe made 24 updates of two rows, covering 48 of 64 training slots
before completing an epoch. Useful validation rollouts fell 2/12 to 1/12.
Training loss declined substantially, but fixed training-set accuracy was never
measured. This prevents distinguishing an unfit mapping from transfer or rollout
conditioning failure. This diagnostic measures that missing evidence without
another optimizer update.

- Hypothesis: the preserved step-24 head has not strongly fit its supervised
  mapping. Evaluate fit before changing data, representation or training budget.
- Variant: restore the one existing final head checkpoint against its verified
  frozen parent; score the exact 64 training and 32 development-validation slot
  requests from the prior frozen corpus. One pass, one native backend scoring call per row.
  Earlier reference choices remain teacher-forced context. Current labels stay
  outside model requests and are consulted only after scoring.
- Primary: training-slot exact accuracy and mean unrounded cross-entropy (CE).
  Report both splits by task kind, semantic slot and wire question type, including
  kind/slot/type cells. No calibration fit or transformed probabilities. The
  diagnostic uses the native DecisionModelBackend's normalized log-probabilities.
- Failure accounting: each attempted row is flushed immediately. Exceptions,
  malformed outputs, unsupported context lengths and nonfinite scores count
  incorrect in all-row accuracy. Failed-row CE is undefined. Overall CE is null
  if any row fails; successful-only CE and coverage are separately labeled.
  Timeouts preserve partial rows and cannot claim a complete diagnostic.
- Stop: exactly 96 requests, no retry, optimizer, new data, rollout or checkpoint
  selection. Stop the owned worker at 900 seconds or below 3 GiB available RAM;
  launcher and worker require 8 GiB before runtime/model load. Two CPU threads,
  one inter-op thread, CUDA hidden, Hugging Face offline, context at most 2048.
  Authenticated inherited anonymous IPC is the only worker entry path.
- Preserve: source closure, original JSON hashes, canonical corpus/splits,
  checkpoint SHA, parent verification evidence, raw row requests/results,
  summaries, prior free-running receipt and both launcher outcome/worker errors.
  No snapshot Python is executed. Parent full state is checked against the prior
  recorded hash and again after evaluation. No new checkpoint is written.

Interpretation is descriptive, using a fixed 80% slot-accuracy threshold for
"strong fit"; this is an operational cutoff, not a statistical qualification.
Any row failure prevents fit/transfer classification. Below 80% training
accuracy means the mapping is not strongly fit: representation versus optimization
remains unresolved. At least 80% training but below 80% validation indicates a
fit/transfer gap on reused development worlds. Both at least 80%, alongside the
existing 1/12 rollout result, indicate that teacher-forced slot competence does
not translate into free-running competence; compounding earlier choices is a
candidate explanation, not a demonstrated causal mechanism. Always inspect
per-kind/slot cells because aggregate slots weight comparisons more heavily.

The corpus has train relation counts equal=3, greater=2, less=3, while all four
validation comparisons are less-than. These reused, tiny synthetic domains do
not qualify broad generalization. Uniform-choice accuracy on the 64 training
rows is 29.17%, and mean uniform CE is `(log(3)+log(4))/2 = 1.24245`.
They are reference arithmetic, not a trained control. This study cannot establish
pretraining benefit, adapted-head retention, a strong-comparator win, causal
mediation or an L0 gate. No held-out/Index data, network, GPU or spend is involved.

From `C:/Users/apumu/research/CHLabs/ssdm-codex-l0-head-fit-audit`, prepare with:

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I -X utf8 language_growth/rwkv_head_fit_audit.py --dry-run --out 'C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-fit-prepared-20261006'
```

Only after explicit run authorization, use a separate fresh directory:

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I -X utf8 language_growth/rwkv_head_fit_audit.py --execute --out 'C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-fit-run1-20261006'
```

Default preparation reads pinned JSON/source files and installed package metadata.
It does not import Torch or inspect/load a checkpoint or parent snapshot. The
committed registration pins the executable closure; preparation alone is not
execution authorization. Both output directories must be fresh.
