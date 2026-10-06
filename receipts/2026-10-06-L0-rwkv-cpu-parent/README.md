# Cached RWKV7 CPU parent dependency

Exploratory Track D parent → Track L path, based on `origin/agent-main`
`4733f785e5284ea959d4d640fff884136487bd05`. No strong-parent capability claim.
This experiment reads only the cached checkpoint and one generated
`development_train` comparison. No official, held-out or Index data, network,
GPU, paid service, or closed teacher is involved. No snapshot Python is executed.

## Compatibility evidence

`static-inspection.json` records the complete safetensors header, all 399 learned
tensor mappings, original config, snapshot/config/vocabulary/weight hashes, and
the inspected local FLA 0.5.2 wheel and source hashes. The full stored checkpoint
has 191,034,624 parameters, although its model ID says 0.1B. FP32 weights occupy
764,138,496 bytes (0.712 GiB); the BF16 checkpoint payload is 382,069,248 bytes.

The FLA `RWKV7Attention` constructor prioritizes `head_dim` and calculates
`hidden_size // head_dim`: 12 heads here, despite `num_heads: 32` in the snapshot
config. Every `r_k` tensor is `[12, 64]`. The CPU config uses these effective
12 heads explicitly. Hybrid attention is absent, value width equals hidden
width, channel mixing is squared ReLU, and initial/layer/output normalization
matches. GroupNorm epsilon is `head_dim * 1e-5` in both implementations.

LoRA `Linear` weights are transposed into the vendored matrix parameters;
LoRA biases and channel coefficients are reshaped without changing values.
Projection, normalization, embedding and head names are explicitly translated.
FLA omits layer-zero's unused value gate; the vendored model registers three
unused buffers, which are filled with zero. Strict loading forbids all other
missing, extra, or differently shaped tensors. Vendored files remain unmodified
and their existing byte hashes are checked before loading.

The FLA recurrence's key/value state is the transpose of the vendored
value/key state. Both implement decay plus the same generalized delta update
with `a=-kk`, `b=kk*sigmoid(a_lora)` and scale 1. Gates, value-residual mixing,
token shift, key correction and output correction agree algebraically. The CPU
implementation uses FP32 and its chunked arithmetic order differs from FLA's
Triton fusion and the token-serial reference; bitwise kernel equivalence is
not claimed. A tiny two-layer, two-row independent transcription of the FLA
equations verifies conversion and full logits at lengths 1, 19, and 65 against
both vendored chunked and eager paths (`2e-5` absolute/relative tolerances).

## Guarded run

Use the existing runtime, with a fresh output directory:

```powershell
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -X utf8 -I language_growth/l0_rwkv_cpu_probe.py --raw C:/Users/apumu/research/CHLabs/ssdm-data/l0-rwkv-cpu-parent-20261006-run2/raw.jsonl
```

The launcher and child both require at least 6 GiB available before Torch starts.
Only the launcher's own child is killed at 300 seconds or below 3 GiB available,
checked every 0.5 seconds, including model loading. CPU threads are limited to 2
(1 inter-op); CUDA is disabled and Hugging Face offline flags are enforced before
interpreter startup. This is a model-specific floor justified by 0.712 GiB of
FP32 weights, no training, at most two unpadded 2048-token rows and last-position
vocabulary logits during scoring. Qwen's existing 8 GiB start guard is unchanged.

The actual-checkpoint CPU smoke must pass before the paired experiment runs.
It compares finite chunked/eager logits on a short fixed prompt, preserves the
full smoke tensors, and refuses a maximum absolute difference above 0.005.
One fixed synthetic training comparison (seed 271828, index 2, production choice
format, no repair) is then scored with identical initial evidence in direct and
free-running planner arms, identity calibration and no option masking. This is
free-running finite-choice planning, not teacher-forced planner slots or a
general language-generation benchmark.

The run directory preserves source/model/data hashes, exact task, parent
revision, raw provider requests/responses, paired report, smoke tensors, stdout,
stderr, elapsed time and minimum available RAM. `smoke.pt` is locally produced
tensor evidence, never loaded from the checkpoint; it is preserved outside Git.

Initial attempts were refused before child/weight load because only 5.38 / 5.37
GiB was available. `run1/raw.jsonl` preserves the second refusal with exact byte
counts. The CPU-smoke dependency is therefore still pending until resource
availability permits an actual bounded run. A passing smoke would establish
CPU execution, not parent capability or Track L success.
