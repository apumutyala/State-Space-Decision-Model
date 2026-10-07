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

Runtime correction: preparation, tests and future execution use exactly
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe`.
The registration pins Torch `2.8.0+cu128`, Python and all package versions;
CUDA is hidden and the synthetic suite asserts CUDA is unavailable.

Full synthetic verification before the start-floor-only update: **7 passed in 3.58s** using:
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest language_growth/tests/test_operation_cache.py -q`, with
`CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, and offline model-library flags. Tests use only tiny
random recurrent weights. Their 96 synthetic optimizer steps are fixture proof,
not a real diagnostic fit or capability measurement.

Review fixes pin bootstrap seed 9203 / 2000 draws, add all three predeclared
kind-specific paired CIs and read non-regression, record the full source base
commit, and bind ordered training-target hashes in checkpoint lineage. Target
hashes remain excluded from feature keys and feature-cache metadata. The tiny
restore test rejects a changed training target.

Default and explicit dry-run preparation both passed an audit hook rejecting
Torch/Transformers/Safetensors imports and checkpoint/snapshot file access.
No real checkpoint load, real fit/inference, GPU, heldout/Index, spend or PR.

After the start-floor-only update, focused preparation/no-load guard verification
passed: **2 passed, 5 deselected in 1.10s**, selecting
`default_and_dry_run or guards_and_owned_worker` from the same test file and
interpreter. Model/data/seed/statistics remain unchanged; no fixture fitting or
feature inference was repeated for this update.

Execution dependencies: tiny-model proofs, #71/#85/#110 cross-review/dependency
gates, fresh >=6 GiB start and >=3 GiB runtime guards, two threads, owned CPU
child, <=1200s.
The 6 GiB start floor follows #71 same-parent CPU run2: available memory was
7,554,498,560 B at start and 6,093,643,776 B at minimum, with a 6 GiB start
requirement. The operation cache adds approximately 442,368 B; the parent is
shared, feature extraction uses one row, and stock readout is capped at two
rows/2048 tokens. Repeated longer sequences, high-water memory and unrelated
host processes remain risks, so the 3 GiB running guard and owned-child limits
remain necessary. The start floor does not establish full-run completion.
The standing goal authorizes gated CPU work; this receipt grants no paid/GPU
authority. Review the plan at `plans/codex/L0_OPERATION_ONLY_CACHE.md`.
