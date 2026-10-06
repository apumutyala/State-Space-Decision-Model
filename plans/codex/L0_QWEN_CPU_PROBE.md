# Frozen Qwen3.5-0.8B CPU development diagnostic

This is a bounded L0d2/L0e diagnostic under L0h. It does not qualify an
attention-free parent or satisfy the L0 learned capability gate.

- **Follow-up hypothesis:** The first diagnostic's wrong-field selection may
  repeat across comparison tasks, indicating a field-grounding/program-discovery
  failure. Test the same direct three-relation choice and checked free-running
  planner on two pinned tasks; retain failures and work in both denominators.
- **Deliverable:** One bounded, offline CPU script for two seeded
  `development_validation` comparison tasks (seed 271828, indices 2 and 5;
  warranty and price respectively). It uses the pinned local snapshot,
  the Track D `load_qwen35(device="cpu", dtype="float32")` readout, one
  `Decider`, and the PR #37 paired-comparison caller. It writes raw JSONL as
  soon as each stage finishes. The launcher checks memory before loading,
  hides CUDA, limits Torch threads to two and batch rows to two, and enforces
  one 300-second subprocess cap for the entire cohort, including model load.
  The child rechecks free RAM before Torch loads. Both pairs use the same
  loaded parent/revision and start each arm from the same initial evidence.
- **Preparation proof:** Static and synthetic tests verify task selection,
  runtime gates, no reference answer in provider inputs, stage flushing, and
  timeout/failure accounting. A failed package-file startup was traced to
  `language_growth/types.py` shadowing stdlib `types`; the root-level isolated
  launcher fixes it.
- **Completed diagnostic:** One offline CPU run loaded the pinned inherited
  hybrid Qwen checkpoint. On the single comparison, the direct relation choice
  was correct, but the planner selected price for both warranty operands and
  `greater_than`; the kernel rejected the false assertion as
  `unsupported_assertion`. Both arms shared the same initial-state hash. The
  receipt and complete raw trace are in
  `receipts/2026-10-06-L-qwen-cpu-development-diagnostic/`.
- **First diagnostic interpretation:** For this one case, the target is expressible and the
  realizer was not reached; the observed failure is program discovery/task
  grounding. A separate development cohort is needed to learn whether this is
  systematic, including per-slot accuracy and free-running task correctness.
  Do not infer a general capability rate or L0 pass from N=1.
- **Completed follow-up:** Both pinned comparisons completed within the cohort
  cap. Direct relation selection was correct on 1/2 tasks. The planner was
  task-correct on 0/2: for both tasks it selected the first record's price for
  both operands and `greater_than`, and the checker rejected each false
  assertion. Both arms used identical starting state per task. The preselected
  two-case result has no CI and does not establish a population rate. Full raw
  trace and receipt are in
  `receipts/2026-10-06-L-qwen-cpu-cohort2/`.
- **Preparation tests:** Sixteen tests cover exact task hashes, the N=2
  denominator and matched initial-state checks, both-arm provider failures,
  pre-provider failures, per-task raw identities, and a second-task runner
  exception preserving the completed first pair. An unattempted task remains
  in the pinned cohort manifest if the global cap stops execution; it cannot
  count as success or silently shrink the intended N=2 denominator. No N=2
  aggregate is emitted unless both task reports complete.

Dependency: PR #37 merged into `agent-main` at `57804e2`; this branch now
integrates the tracked paired caller. The script still fails with a clear
dependency error if that module is absent. No source overlay is needed.

This follow-up extends PR #52's current diagnostic head
`14256ba` (`codex/l0-cpu-probe-bootstrap`), including its first-run receipt;
PR #52 is a dependency, not an independent qualification result. The initial
task is deliberately repeated without modifying prompts or choices, and the
second task adds one bounded comparison. Their SHA256 pins are:

| Index | Task SHA256 |
| --- | --- |
| 2 | `e996f537fea463e480f691b36ed68b5f3a8aa1da014e798b706a672473029e0c` |
| 5 | `cc9d1d2418ef38891870e517dd1fcc75d01e238023e65dcc80e9de9aac783b0f` |

The diagnostic is inherited **HYBRID** pretrained Qwen with attention layers,
not the attention-free Track D target, a strong unrestricted comparator, a
parent-retention test, or Decision Index/held-out evidence.

The original package-file invocation failed during interpreter startup because
`language_growth/types.py` shadowed stdlib `types`. The repository-root entry
point adds only the repository root under `-I`, then imports the probe's `main`.
The owned child uses the same entry point and `-I`, with the existing offline CPU
environment. Preparation tests cover subprocess `--help` for both modes even
from the package directory with a conflicting `PYTHONPATH`, and assert the
recorded child command and offline environment. These checks do not import
Torch, load weights, or execute the diagnostic.

For the completed run, the exact command and output path were:
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe` runtime (known
`transformers` 5.17.0, `accelerate` 1.15.0, `torch` 2.8.0, `psutil` 7.2.2).
Run from the repository root:

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I l0_qwen_cpu_probe.py --raw 'C:/Users/apumu/research/CHLabs/ssdm-data/l0-qwen-cpu-probe-271828-run2/raw.jsonl'
```

The output directory must be fresh. The launcher exits before loading if
available RAM is below 8 GiB or the paired dependency is absent. During the
run it stops only its own child if available RAM falls below 3 GiB or elapsed
time reaches 300 seconds, including if launcher monitoring is interrupted.
Each request and response is flushed to `raw.jsonl`
before the next provider call, with task ID and task hash on every provider
event. Each singleton `task_report` is flushed before validation and before
starting the next task. The final `paired_report` includes both pairs and
checks the full denominator, exact task order and both initial-state hashes;
pre-provider failures retain their task with unknown state identity and cannot
be credited as paired correct. A stopped run keeps all partial evidence.

Prepared follow-up command (not executed; run from this worktree's root with
a fresh output directory):

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I l0_qwen_cpu_probe.py --raw 'C:/Users/apumu/research/CHLabs/ssdm-data/l0-qwen-cpu-probe-271828-cohort2-run1/raw.jsonl'
```

Exact preparation check (no model load, API calls or network):

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -m unittest language_growth.tests.test_l0_qwen_cpu_probe -v
```
