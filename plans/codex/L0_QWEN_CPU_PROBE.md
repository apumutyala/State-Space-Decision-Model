# Frozen Qwen3.5-0.8B CPU development diagnostic

This is a bounded L0d2/L0e diagnostic under L0h. It does not qualify an
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
- **Decision unlocked:** For this one case, the target is expressible and the
  realizer was not reached; the observed failure is program discovery/task
  grounding. A separate development cohort is needed to learn whether this is
  systematic, including per-slot accuracy and free-running task correctness.
  Do not infer a general capability rate or L0 pass from N=1.

Dependency: PR #37 merged into `agent-main` at `57804e2`; this branch now
integrates the tracked paired caller. The script still fails with a clear
dependency error if that module is absent. No source overlay is needed.

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
before the next provider call; a stopped run keeps partial evidence.
