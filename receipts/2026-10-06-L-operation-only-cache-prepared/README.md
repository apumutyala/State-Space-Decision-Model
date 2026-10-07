# Prepared operation-only diagnostic

Status: prepared, not executed against the real parent. Cost: $0.

Integration base: `018585548104dfbcb9d65a22214ec51405529711` on merged
`agent-main`, containing #71, #85 and #110. The original stacked branch is
preserved. Only the operation-only experiment and one Track L status note are
transplanted; unrelated head-learning and qend-ablation experiments are absent.

`registration.json` freezes current merged source/software and the unchanged
corpus digest `cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466`.
`corpus.json.gz` preserves the original compressed corpus byte-for-byte, with
24 training / 12 reused development worlds. `operation-records.json` contains
only their initial operation rows, with targets separated from requests.
Feature caches and model checkpoints do not exist in this receipt.

The runner owns only its required execution guards, owned-child supervisor,
request capture, attribution evaluation and full-parent digest helpers. They
are pinned through the runner's source hash; offline environment/event helpers
come from merged CPU-probe code. No unmerged experimental module is imported.

Conditions: unchanged stock-LM planner (primary baseline), exact step-0 q_end
head (secondary architecture/random-init control), fixed step-96 q_end head
(treatment). Both head conditions delegate every later slot to stock readout.
Training uses operation-only detached features and CE; no MSE/axis targets.
The seed, AdamW recipe, batch size, eight full passes, target lineage,
bootstrap seed 9203 / 2000 draws and predeclared per-kind comparisons are unchanged.

Preparation, focused synthetic checks and future execution use exactly
`C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe`.
Registration pins Torch `2.8.0+cu128`, Python and package versions strictly.
Checks use `CUDA_VISIBLE_DEVICES=-1`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
`OPENBLAS_NUM_THREADS=2`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`. The suite uses only tiny random recurrent fixtures;
96 synthetic optimizer steps are execution proof, not a real diagnostic fit.
Verification: **8 passed in 14.69s**, using the focused operation-cache test file
with pytest cache disabled. The additional synthetic supervisor test proves
timeout and low-memory cleanup without starting a worker. Default/dry-run
preparation rejects Torch/Transformers/Safetensors imports and checkpoint/snapshot
access. Exact command, environment and output are saved in `tests.txt`.

Execution still requires tiny-model proofs, review of this integrated diff and
fresh >=6 GiB start / >=3 GiB runtime guards, two threads, an authenticated
owned CPU child and a <=1200-second cap. The 6 GiB start floor follows #71
same-parent CPU run2: available memory was 7,554,498,560 B at start and
6,093,643,776 B at minimum, with a 6 GiB start requirement. The operation cache
adds approximately 442,368 B; the parent is shared, extraction uses one row,
and stock readout is capped at two rows/2048 tokens. Longer prompts, allocator
high-water memory and unrelated host processes remain risks; keep the runtime
guard and owned-child limits. This floor does not establish full-run completion.

No real checkpoint load, real fit/inference, GPU, heldout/Index data access,
spend or PR. The standing goal authorizes gated CPU work; this receipt grants
no paid/GPU authority. Plan: `plans/codex/L0_OPERATION_ONLY_CACHE.md`.
