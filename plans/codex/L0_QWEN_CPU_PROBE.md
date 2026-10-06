# Frozen Qwen3.5-0.8B CPU development diagnostic

This is a preparation phase under L0d2/L0e/L0h. It does not qualify an
attention-free parent or satisfy the L0 learned capability gate.

- **Hypothesis:** The existing stock Qwen3.5-0.8B readout can make a direct
  three-relation choice and drive the checked free-running planner on the same
  generated development-validation evidence, with failures and work retained
  in both denominators.
- **Deliverable:** One bounded, offline CPU script for one seeded
  `development_validation` comparison task (seed 271828, index 2). It uses the pinned local snapshot,
  the Track D `load_qwen35(device="cpu", dtype="float32")` readout, one
  `Decider`, and the PR #37 paired-comparison caller. It writes raw JSONL as
  soon as each stage finishes. The launcher checks memory before loading,
  hides CUDA, limits Torch threads and batch rows, and enforces a 300-second
  subprocess cap. The child rechecks free RAM before Torch loads.
- **Exit proof for preparation:** Static and synthetic tests verify the task
  selection, runtime gates, no reference answer in provider inputs, stage
  flushing, and timeout/failure accounting without loading model weights.
  The exact run command and required environment are documented. A real-model
  result requires separate resource clearance and a raw receipt.
- **Decision unlocked:** If resource clearance and runtime preconditions pass,
  run this single CPU diagnostic and inspect direct versus planner failure
  modes before considering a larger or paid experiment.

Dependency: PR #37 (`codex/l0-paired-comparison`) is currently unmerged from
`origin/agent-main`. Keep this phase's commit distinct and rebase when that PR
lands. The script must fail with a clear dependency error until the paired
module is present; no module copy or speculative fork is included.

The diagnostic is inherited **HYBRID** pretrained Qwen with attention layers,
not the attention-free Track D target, a strong unrestricted comparator, a
parent-retention test, or Decision Index/held-out evidence. No model load or
run belongs to this preparation phase.

After PR #37 lands and the resource owner clears a run, use the existing
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe` runtime (known
`transformers` 5.17.0, `accelerate` 1.15.0, `torch` 2.8.0, `psutil` 7.2.2):

```powershell
& 'C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe' -I language_growth/l0_qwen_cpu_probe.py --raw 'C:/Users/apumu/research/CHLabs/ssdm-data/l0-qwen-cpu-probe-271828/raw.jsonl'
```

The output directory must be fresh. The launcher exits before loading if
available RAM is below 8 GiB or the paired dependency is absent. During the
run it stops only its own child if available RAM falls below 3 GiB or elapsed
time reaches 300 seconds, including if launcher monitoring is interrupted.
Each request and response is flushed to `raw.jsonl`
before the next provider call; a stopped run keeps partial evidence.
