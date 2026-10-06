# fla_rwkv7: fla `chunk_rwkv7` equivalence + throughput gate on RWKV7-G1k-13.3B (≤ $4.50). Spec only, NOT submitted

**Why.** The vendored "chunked" WKV is pure PyTorch: 3.6–4.3k tok/s prefill on the 13.3B, about 11.5% of H100 bf16 peak
(`jobs/PERF_AUDIT.md`, branch `claude/readout-fork`). At that rate an m2 factorial cell takes about 12 h, against the
5.75 h estimate, and the m2 child evals need ≥ ~12k tok/s. This job decides whether fla's `chunk_rwkv7` replaces it.

**What is registered.** `heads/rwkv7_wkv.py` adds `"fla"` and `"fla_safe_gate"` to the vendored
`RWKV7_WKV_FUNCTIONS` from our code. The vendored files are untouched and stay sha-pinned. Selection is by config:
- `select_wkv(model, name)`;
- `load_rwkv7_hf_backbone(..., wkv=)`, `load_rwkv7_hf(..., wkv=)`;
- the cell config key `rwkv7_wkv` (`train/run_cell.py`).

It is **off by default**: `None` keeps the checkpoint's `"chunked"`. fla names refuse until `GPU_VERIFIED` is set,
which happens only after this job returns ADOPT.

## fla API, checked against fla-core 0.5.2

The source was read from the wheel, sha256 `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761`.

- **Call.** `fla.ops.rwkv7.chunk_rwkv7(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False,
  cu_seqlens=None, cu_seqlens_cpu=None, safe_gate=False, chunk_size=None, ...)` calls `chunk_dplr_delta_rule(q=r, gk=w, ...)`.
- **Mapping.** `w = w_log`, `a = -kk`, `b = kk * a`, `scale = 1.0`. fla keeps the state as (key, value) and the
  vendored code as (value, key), so the state is transposed on the way in and on the way out. fla returns its final
  state in fp32, and it is cast back to the carried dtype. The derivation is in the module docstring.
- **Decision path.** fla takes `initial_state` of shape `[N, H, K, V]` and returns the final state when
  `output_final_state=True`. Its backward returns `dh0`, so gradients reach the prefix through a forked state.
  With `cu_seqlens` (B = 1), each segment starts from zeros and the wrapper returns the last segment's state, which is
  the vendored packed contract. `fused_recurrent_*` has no backward ("inference only"), so it is not used.
- **#640.** The RWKV-7 path goes through `generalized_delta_rule/dplr/*`. Its import graph never reaches the gated
  `chunk_bwd_dqkwg` in `fla/ops/common/chunk_o.py`, so **#640 does not affect the RWKV-7 backward**. The kernel phase
  re-checks this from the installed fla source (`kernel.json:p640`). `fla_guard.py` still runs at setup, as in every
  fla image.
- **fp32.** The DPLR wrapper warns that fp32 is unsupported "on some platforms". The model runs bf16, so fp32 calls
  fall back to "chunked" (`ALLOW_FP32 = False`) unless this job's fp32 cases pass.
- **The tokenwise trap.** `modeling_rwkv7` sets `tokenwise = wkv_implementation != "chunked"` at 3 sites. Tokenwise
  means one GEMV per token for every Linear, which would cancel the kernel's gain. `WkvName("fla")` answers
  `!= "chunked"` with False, so fla runs the same batched Linears as "chunked". A test pins those 3 sites.

## Image

This is the **jobs/rwkv7 image, unchanged**: `build_context.sh` copies its Dockerfile and rewrites only the CMD.

Correction to the brief: that image does **not** contain fla (`fla_guard` reports "fla not installed (rwkv7 / m2
images)"). The runner therefore installs `fla-core==0.5.2` at setup with `--no-deps --require-hashes`
(`requirements-fla.txt`). torch 2.13.0, triton 3.7.1, einops and transformers 5.17.0 are already pinned in the image.
Setup asserts all of these versions, then runs `fla_guard.py` (exit 5).

## Phases (`job_params.json`; caps sum to 31 = `max_duration_minutes`)

| Phase | Cap | Content |
|---|---|---|
| setup | 2 | pins, fla-core install, vendored-code self-check, fla #640 guard; the snapshot download starts in the background. job-4idg4: 2 min 05 s (not timeout-wrapped) |
| warmup | 13 | fla JIT compile + autotune, paid once and timed: both variants × {initial state, packed varlen} × {fwd, fwd+bwd}, bf16, T=2048 then T=1000. fla's autotune keys are block sizes, not T, so the first call's length fixes the config for every length: it is a real one (PR #59 N1), and 1000 covers the non-multiple-of-16 specialisation. `TRITON_CACHE_DIR` + fla's `cache_results` carry the results to every later phase process |
| kernel | 3 | synthetic inputs drawn from the model's distributions (decay law `-e^-0.5·σ`, unit `kk`, `a∈(0,1)`, fp32 state) at the real head shape 64×64. Forward at B=2, T=2048 and T=1000 (not a chunk multiple). Gradients of all 7 inputs at T=512, and at T=24, where d/d(initial state) is still non-zero (fla's `dh0`). bf16. Implementations: chunked, fla, fla_safe_gate vs eager. fp64 token-serial run as a diagnostic. #640 source check |
| fork | 2 | (1) state split, P=1500 + L=37, vs one-shot eager; (2) one prefix state `repeat_interleave`d to K=4 tails, with gradients into prefix and tails; (3) packed row `[0,100,101,1100,2048]`, including a 1-token segment, with gradients; (4) right padding with the model's keep transform leaves the state untouched |
| download_wait | 2 | wait for the snapshot, `sha256sum -c jobs/rwkv7/snapshot.sha256`, snapshot code == vendored (exit 10) |
| model | 4 | real weights, bf16. (1) Logits vs eager at T=256 (real text) for chunked, fla and fla_safe_gate. (2) Continuation: the state of a 1900-token prefix + 148 tokens vs one shot, and 2 deep-copied forks. (3) Real WKV inputs captured at layers 0/30/60 (T=2048 text, non-zero carried state), then forward and gradients there |
| throughput | 4 | real model, median of 3 after a warm-up. Prefill tok/s at **2k / 8k / 16k**. Training-shaped fwd+bwd at T=2048 (frozen weights, grad into the embeddings). WKV-only fwd and fwd+bwd times at each length (attribution). Any fla fallback voids the measurement |
| reserve | 1 | |
| fp32 (optional) | time left | the kernel + fork cases at fp32. Runs last, only if ≥ 90 s remain before `max_duration − reserve`, with a timeout of exactly the time left. It only sets `allow_fp32`. It is true only if `fp32.json` carries `complete: true`, which is written after the last case (PR #59 B1); a missing or partial file gives false and never changes the verdict or the exit code |

**job-jucxx (exit 7, warmup timeout at 3 min, $0.25 + one preempted attempt).** The warmup cap was an unmeasured guess. `warmup.json` measured the real cost: `state_fwd` T=2048 took 72.6 s and `packed_fwd` took 69.7 s on the first call, with the second call about 2 ms. So the cost is compile + autotune, about 70 s per specialisation, and the kernels themselves are fast. Two of the eight specialisations fit in 3 min, and the backward ones compile more kernels. **Fix:** warmup cap 3 → 13 min (expected 10), max duration 21 → 31, hard cap $3 → $4.50. If this recurs, persist the Triton cache across jobs; don't trim the specialisations, because the later phases need every one.

**job-4idg4 (exit 7, kernel timeout at 4 min, $0.30).**
- Setup and the guard passed; the snapshot download finished in about 1 s (warm `hf_cache`).
- The kernel phase finished every bf16 case (all pass: fla out ≤ 7.8e-3, grads ≤ 1.1e-2 vs eager) and fp32 chunked
  fwd T=2048 by +2 min 58 s.
- The first fp32 fla call started then (fla's fp32 warning is the last line of `kernel.log`) and was still running at
  +4 min 00 s, when it was killed.
- **Cause: first-call Triton JIT compile + autotune of fla's DPLR kernels.** Every new specialisation (dtype,
  `safe_gate`, initial state, varlen, backward) compiles and benchmarks every autotune config. The bf16 set took a
  large share of those 3 min; the fp32 set was a second full compile.
- Not the eager reference: about 25k small launches at T=2048, under a second. Not a hang: every bf16 kernel
  returned.
- **Fix:** a capped, timed warmup phase with one shared Triton cache across phases, bf16-only gated phases, and fp32
  moved to an optional tail.
- The schedule still sums to 21 min, so the worst case is unchanged.

## Pre-registered tolerances and decision rule (`fla_equivalence.py`; pinned by `tests/test_fla_rwkv7.py`)

The metric is `rel = ‖x − ref‖∞ / max(‖ref‖∞, 1e-6)` per tensor, plus the cosine similarity of the flattened tensors
(taken as 1 when both tensors are below 1e-6). The floor exists because the decay drives some gradients, such as
d/d(initial state) after 1024 tokens, to ~1e-45: those are zero, not a 100% error. The reference is the vendored
`rwkv7_eager` at the same activation dtype.

| dtype | out | final state | every gradient | min cosine |
|---|---|---|---|---|
| bf16 (decides) | 2e-2 | 2e-2 | 5e-2 | 0.999 |
| fp32 (only sets `ALLOW_FP32`) | 2e-3 | 2e-3 | 5e-3 | 0.99999 |

Model level (real weights, bf16):
- **Logits.** fla's logits rel vs eager must be ≤ max(2e-2, 1.5 × chunked's rel vs eager). The per-position cosine
  must be ≥ 0.999 and top-1 agreement ≥ 0.98.
- **Continuation and fork.** The continuation rel must be ≤ 2e-2, and the two forks must agree to rel ≤ 1e-5.

**Rule, per variant.**
- **Numerics** pass when every bf16 kernel, fork and captured case passes, and the model-level checks pass.
- **Readout speed** passes when fla/chunked prefill is **≥ 2.0× at every one of 2k, 8k and 16k**.
- **Training speed** passes when fwd+bwd at 2k is ≥ 2.0×.

**Verdicts.**

| Verdict | Condition |
|---|---|
| `ADOPT` | numerics and both speeds pass |
| `ADOPT_READOUT_ONLY` / `ADOPT_TRAINING_ONLY` | numerics and one speed pass |
| `REJECT_SPEED` | numerics pass, neither speed passes |
| `REJECT_NUMERICS` | any measured numerics failure |
| `INCONCLUSIVE` | a measurement is missing, crashed or fell back |

Plain `fla` is chosen unless `fla_safe_gate` also qualifies and is ≥ 1.10× faster on every ratio. Whether the chosen
variant reaches 12k tok/s at 2k is reported, but it is not part of the rule.

**On ADOPT**, in a follow-up commit:
1. set `GPU_VERIFIED = True`;
2. set `ALLOW_FP32` only if the fp32 cases passed;
3. set `rwkv7_wkv: "fla"` in the m2 cell configs.

On an `_ONLY` verdict, enable fla only on the path that passed.

## Platform call: SF Compute batch `submit_job` (coordinator, after merge; not done here)

```
open_mission("m2-fla-rwkv7-kernel")
jobs/fla_rwkv7/build_context.sh <sha> <dir>  -> create_context(<dir>.tar.zst) -> context_id
submit_job(chip="h100", chip_count=1, mission="m2-fla-rwkv7-kernel", context_id=<id>, dockerfile_path="Dockerfile",
  build_timeout_minutes=18, command="bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh", max_duration_minutes=31,
  max_restarts=0, resume="none", queue_ttl_minutes=1440, hf_cache=true,
  model="RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
  checks={"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/FLA_RWKV7_DONE"},
                                           {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
          "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                     "interval_s": 300, "grace_period_s": 600, "fail_threshold": 2, "on_fail": "kill"}})
```

## Cost (`python -m jobs.common.costs` convention: build billed, at least one restart assumed)

**Worst case: (18 + 2 × 31) × $0.0495 = $3.96 ≤ $4.50.** The expected run is about 22 min plus the optional fp32 tail, about $1.09–1.50. Before submitting:
1. Run `set_workspace_cap(MTD + 3)`.
2. Run `validate_job`; it must pass the cost-cap preflight without `force`.

**Risks.**
- The background download (24.7 GiB) must finish within warmup + kernel + fork + 2 min. `hf_cache=true` and the earlier
  rwkv7 job on the same pin make a warm cache likely. If it does not finish, the job exits 10 at no extra cost beyond
  the minutes used.
- fla's first call per specialisation compiles and autotunes; that cost now sits in `warmup` (timed, `warmup.json`).
  Every case also records `t_s`, its elapsed time in the phase, so a timeout can be attributed without guessing.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | a verdict was reached; any verdict counts |
| 5 | setup: pins, fla install, #640 guard |
| 6 | export failed |
| 7 | phase timeout |
| 9 | decision missing or invalid |
| 10 | snapshot or vendored-code mismatch, or the download failed |
| 30 | warmup (fla compile) crashed |
| 31–34 | the kernel / fork / model / throughput phase crashed |
| 1 | finalize failed |
