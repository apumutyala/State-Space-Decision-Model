# Prepared operation-only diagnostic

Status: prepared, not executed against the real parent. Cost: $0.

`registration.json` freezes source/software, the existing corpus digest and all
24 training / 12 reused development operation rows. `operation-records.json`
contains only those rows, with targets separated from requests. Feature cache
and model checkpoints do not exist in this receipt; future gated execution
creates them in a fresh output directory.

Conditions: unchanged stock-LM planner (primary baseline), exact step-0 q_end
head (secondary architecture/random-init control), fixed step-96 q_end head
(treatment). Both head conditions delegate every later slot to stock readout.
Training uses operation-only detached features and CE; no MSE/axis targets.

Verification: `7 passed in 9.52s` for
`python -m pytest language_growth/tests/test_operation_cache.py -q`, with
`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, and offline model-library flags. Tests use only tiny
random recurrent weights. Their 96 synthetic optimizer steps are fixture proof,
not a real diagnostic fit or capability measurement.

Default and explicit dry-run preparation both passed an audit hook rejecting
Torch/Transformers/Safetensors imports and checkpoint/snapshot file access.
No real checkpoint load, real fit/inference, GPU, heldout/Index, spend or PR.

Execution dependencies: tiny-model proofs, cross-review/dependency merges,
fresh >=12 GiB start and >=3 GiB runtime guards, owned CPU child, <=1200s.
The standing goal authorizes gated CPU work; this receipt grants no paid/GPU
authority. Review the plan at `plans/codex/L0_OPERATION_ONLY_CACHE.md`.
