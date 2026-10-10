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
submit_job(chip="h100", chip_count=1, mission="m2-fla-rwkv7-kernel", idempotency_key="m2-fla-rwkv7-gate-<context_sha256[:12]>", context_id=<id>, dockerfile_path="Dockerfile",
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
1. Run `set_workspace_cap(MTD + 4)` (re-priced 2026-10-09 at $0.063/min; was MTD + 3; x 14/11, rounded up).
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

## RULE v2 (`decide_v2`; mode `truth_v2`)

### Rule text, verbatim (coordinator, committed before any truth_v2 run)

RULE v2 (pre-registered 2026-10-06, before job execution):
- Truth = fp64 token-serial WKV (existing `truth()` helper) on the SAME captured bf16-rounded real inputs. Capture layers {0, 15, 30, 45, 60} (fwd T=1024 and grad T=512, as now) from the same 2,048-token prompt as job-jn47u (same seed/ids construction).
- For each case and each tensor (out, final state, every grad): e(impl) = rel error vs truth (same rel metric, same ABS_FLOOR).
- NUMERICS_OK(fla) iff for every case & tensor: e(fla) <= 1.10 * max(e(chunked), e(eager_bf16)) + ABS_FLOOR, AND e(fla) <= 1.10 * e(chunked) + ABS_FLOOR on at least 90% of (case,tensor) pairs, AND all finite, AND the existing synthetic kernel/fork/model-level checks still pass (rerun them unchanged). Same for fla_safe_gate.
- TRAIN_SPEED: measure the m2 first training cell's ACTUAL training step (whatever WKV impl and config jobs/m2_first_cell/m2_first_cell_rwkv7.json / train/ use) with the current impl vs fla: same LoRA rank, batch, grad-accum, seq-length mix, checkpointing, 20 timed steps after warmup, median tok/s and peak GB. If the current impl is token-serial eager and too slow/OOM, record that as the measurement.
- VERDICT: ADOPT_FOR_TRAINING iff NUMERICS_OK and train ratio >= 2.0 and fla peak <= 76 GB; ADOPT_READOUT_ONLY iff NUMERICS_OK and prefill ratio >= 2 at all lengths but train ratio < 2; REJECT_NUMERICS iff NUMERICS_OK false; INCONCLUSIVE if anything missing. The old v1 rule is NOT edited; v2 is a new, separately named rule (`decide_v2`) and its constants are pinned by tests.

### Why v2 (job-jn47u receipt)

job-jn47u (SF batch, H100, code `4733f785`, 2026-10-06 20:13–20:30 UTC, exit 0, verdict `REJECT_NUMERICS`; artifact
sha256 `692f8986…96fc`, `decision.json` `c35d38f6…3030`, `captured.json` `c7546203…86c1`):
- Synthetic kernel (12 gated cases) and fork (12) all pass for chunked, fla and fla_safe_gate.
- Model level vs eager logits (T=256): fla rel 0.021, cos_min 0.99995, top-1 1.0; chunked rel 0.023; fla_safe_gate
  rel 0.018. Continuation rel: fla 0.0199 (pass), fla_safe_gate 0.0265 (> 0.02, fails).
- The verdict comes only from the captured real inputs at L0/30/60, where **every** implementation, including the
  production `chunked` path, is far from the bf16 token-serial eager reference, and fla and chunked agree with each
  other to 3 decimals:

| case | chunked out / state / grad | fla out / state / grad |
|---|---|---|
| L0 fwd T1024 | 0.110 / 0.370 / – | 0.112 / 0.370 / – |
| L0 grad T512 | 0.102 / 0.250 / 0.271 | 0.102 / 0.250 / 0.271 |
| L30 fwd T1024 | 0.033 / 0.101 / – | 0.033 / 0.101 / – |
| L30 grad T512 | 0.022 / 0.074 / 0.046 | 0.022 / 0.074 / 0.046 |
| L60 fwd T1024 | 0.060 / 0.225 / – | 0.063 / 0.225 / – |
| L60 grad T512 | 0.038 / 0.135 / 0.102 | 0.038 / 0.135 / 0.102 |

- Speed: prefill fla / chunked 3.09× (2k), 2.69× (8k), 2.96× (16k); fla 11.7–12.1k tok/s. Training-shaped fwd+bwd at
  T=2048 (grad into the embeddings): fla 4,561 tok/s at 51.1 GB peak, chunked **OOM** (80 GB).

So the v1 per-layer gate is inconsistent: it rejects the production path, because a bf16 token-serial run is not
ground truth on real activations (state magnitudes up to 76 at L60). v2 measures every implementation against fp64.

### How the rule is implemented (fixed in the rule commit, before any run; `fla_equivalence.py`)

- **Cases.** 5 layers × {fwd T=1024: out, state; grad T=512: out, state, 7 input gradients} = 10 cases, 55
  (case, tensor) pairs (`v2_required_cases`). The 90% clause needs ≥ 50 of 55. Capture is the jn47u construction
  (`_text_ids(model_dir, 2048)`; the first 1024 tokens build the carried state; the WKV inputs are captured during the
  next 1024 on the chunked path).
- **Truth.** `truth()` (fp64 `rwkv7_recurrent` on the bf16-rounded inputs, `.double()`) for out/state. Gradients:
  fp64 autograd through the same serial loop at T=512, with the same cotangents as `run_impl` (seeded `randn`, the
  same values cast to fp64). `eager_bf16` = `run_impl("eager")`, the v1 reference.
- **All finite.** A crash or a non-finite output/e of the fla variant fails NUMERICS_OK (v1 treats a crash the same
  way). A missing case, a crashed or non-finite chunked / eager_bf16 reference, or a non-finite truth makes the
  comparison undefined: INCONCLUSIVE.
- **Synthetic checks.** The kernel, fork and model phases run unchanged; v2 requires every bf16 kernel and fork
  case to pass `TOL`, the model logits to pass `model_passes` and the continuation/fork to pass, exactly as v1. The
  v1 captured-input cases (still produced by the unchanged model phase) are not part of v2. A fallback or a missing fla
  call in the model or throughput phase voids that phase (INCONCLUSIVE), as in v1.
- **Train speed.** The cell's WKV is `rwkv7_wkv: "chunked"` (the checkpoint's pure-PyTorch chunked path, not the
  token-serial eager loop), so the current impl is `chunked`. The step is `train.trainer.Trainer.train_step` with the
  cell's TrainConfig (fork head, cache fork, LoRA r=64, Clef loss, batch 1 × grad_accum 8, bf16 autocast, no
  gradient checkpointing) on rows with the shapes of the cell's first optimiser steps (`train_mix_v2.json`, below).
  3 untimed warmup steps (`V2_TRAIN_WARMUP`), then 20 timed steps. Per step: tok/s = prefix tokens of its 8 rows /
  synchronised wall-clock; median over the 20. Peak = `torch.cuda.max_memory_allocated()` / 1e9 over the process,
  as the cell's `peak()`. Each impl runs in its own process.
- ~~**OOM / too slow.** A timed step that runs out of memory counts 0 tok/s. The current impl's median is over the timed
  steps it reached before its phase deadline (≥ 5, `V2_CURRENT_MIN_STEPS`; fewer is INCONCLUSIVE). An fla variant
  must reach all 20; an fla OOM on any step fails the peak clause. Train ratio = median(fla) / median(chunked); +inf
  when chunked's median is 0 and fla's is not; 0 when fla's is 0. A fallback or zero fla calls voids the fla
  measurement (INCONCLUSIVE).~~ **Withdrawn by Amendment A1 below.**
- **Prefill ratio** = the unchanged throughput phase (fla / chunked at 2k, 8k, 16k).
- **Verdict order.** REJECT_NUMERICS if NUMERICS_OK is false; else INCONCLUSIVE if NUMERICS_OK, a prefill ratio, the
  train ratio or the fla peak is missing; else ADOPT_FOR_TRAINING / ADOPT_READOUT_ONLY as stated. The rule names no
  verdict for the remaining measured combinations (train ratio ≥ 2 but peak > 76 GB or an fla OOM; or train ratio < 2
  and prefill < 2 at some length): those are `NOT_ADOPTED`, reported and not adopted.
- **Variants.** Each of fla and fla_safe_gate gets its own verdict under the same rule; the top-level `verdict` is plain
  fla's. `decision_v2.json` carries both; `valid_decision_v2` is the finalize validator.

### Amendment A1 (2026-10-06, after independent review, before any v2 run)

Amendment A1 (2026-10-06, after independent review, before any v2 run): coordinator decisions, quoted verbatim:

> B1 — I withdraw the OOM-as-0-tok/s convention (it was written after jn47u showed chunked OOM and it alone could
> produce ADOPT). Binding train ratio = median fla tok/s / median chunked tok/s over the timed steps BOTH impls
> completed (matched rows). Require ≥ 10 matched steps, else INCONCLUSIVE. Chunked OOM counts (and which row lengths)
> are reported as a separate, non-binding capability finding `chunked_oom` ("chunked cannot keep T_max=2048 at the
> cell config"); likewise fla OOM still fails the peak clause. This also fixes the deadline-subset mismatch
> (non-blocking #2).
>
> B2 — Void the fla (and fla_safe_gate) train measurement → INCONCLUSIVE if any warmup or timed loss is nonfinite.
> Report step-0 loss rel-diff vs chunked (same rows/seed/init) as a diagnostic; additionally make it binding that
> |step-0 loss rel-diff| ≤ 0.02 when chunked's step 0 completed, else INCONCLUSIVE (this only ever moves toward
> INCONCLUSIVE). Also record the dtypes of r/kk/w reaching the kernel during training.
>
> Acknowledged as part of the rule: V2_CURRENT_MIN_STEPS (now superseded by the ≥10 matched-steps requirement),
> V2_TRAIN_WARMUP=3, NOT_ADOPTED, ratio≥2 & peak>76GB → NOT_ADOPTED.

As implemented (`train_ratio_v2`, `train_summary_v2`, `decide_v2`; constants `V2_MIN_MATCHED_STEPS = 10`,
`V2_STEP0_LOSS_REL_MAX = 0.02`, pinned by `test_rule_v2_constants_are_pinned`):
- **Matched steps.** A step is completed when it has a measured tok/s and neither ran out of memory nor was skipped by
  the deadline. The matched set is the timed steps (by index, hence the same rows) completed by both chunked and the
  fla variant. Train ratio = median(variant tok/s) / median(chunked tok/s) over exactly that set; fewer than 10 is
  INCONCLUSIVE. OOM steps never enter a median. `V2_CURRENT_MIN_STEPS` and `train_speed_v2` are removed.
- **chunked_oom.** `decision_v2.json["chunked_oom"]`: chunked's OOM step count and each OOM step's longest row (max T),
  with `binding: false` and the finding "chunked cannot keep T_max=2048 at the cell config" when the count is non-zero.
  It never changes a verdict.
- **fla OOM.** Any OOM step of the variant (warmup or timed) fails the peak clause, as before.
- **Non-finite loss.** Any completed warmup or timed step of the variant with a non-finite loss (or `finite: false`)
  voids its train measurement (ratio None → INCONCLUSIVE). Chunked's own non-finite losses do not void the variant.
- **Step-0 loss.** `step0_loss_rel_diff = |loss_v(0) − loss_chunked(0)| / |loss_chunked(0)|` (step 0 = the first warmup
  step: same rows, seed, head and LoRA init). It is binding only when chunked's step 0 completed with a finite loss:
  then a missing or non-finite variant step-0 loss, or a rel diff > 0.02, is INCONCLUSIVE. Otherwise it is reported
  as None.
- **Dtypes.** `train_v2_<impl>.json` records the dtypes of r / kk / w_log (and state) reaching the WKV during training
  (code change, separate commit). This is a diagnostic only.
- **Order unchanged.** REJECT_NUMERICS beats everything; the loss guards only ever produce INCONCLUSIVE.

## Mode `truth_v2` (`fla_job.sh truth_v2`, `job_params_truth_v2.json`). Spec only, NOT submitted

Same image, `build_context.sh` and setup as the v1 gate. The setup, warmup (13-min cap, reused unchanged:
~70 s per specialisation, job-jucxx; 341 s in job-jn47u), kernel, fork, snapshot and model phases, the background
snapshot download and the shared `TRITON_CACHE_DIR` all run exactly as in v1. No fp32 tail. Then:

| Phase | Cap | Expected | Content |
|---|---|---|---|
| setup / warmup / kernel / fork / download_wait / model / throughput | 3 / 13 / 2 / 2 / 2 / 3 / 3 | 2.3 / 5.7 / 0.5 / 0.5 / 1 / 1 / 1.5 | unchanged v1 phases; expected = job-jn47u's measured phase times (20:13–20:23 UTC) |
| truth_v2 | 3 | 1.5 | load, capture L0/15/30/45/60 (one 2,048-token pass, jn47u construction), then per layer: fp64 truth fwd T=1024, fp64 truth + autograd T=512, and eager_bf16 / chunked / fla / fla_safe_gate on the same inputs → `truth_v2.json` |
| train_chunked / train_fla / train_fla_safe_gate | 6 / 6 / 5 | 4 / 3 / 3 | one process each: load the backbone (`load_rwkv7_hf_backbone`, cache fork), select the WKV, build `DecisionModel` + `Trainer` with `train.run_cell.train_config(cell)`, 3 + 20 optimiser steps on the mix rows → `train_v2_<impl>.json`. In-process deadline = cap − 45 s: no step starts if fewer than 1.5 × the slowest step's seconds remain (recorded `skipped`) |
| decide_v2 | – | – | `decision_v2.json` (`valid_decision_v2`) |
| reserve | 1 | | |

Caps sum to 49 = `max_duration_minutes`; every cap is ≥ 1.3× its expected time (pinned by
`test_truth_v2_runner_and_params`).

**The m2 cell, as read.** `jobs/m2_first_cell/m2_first_cell_rwkv7.json` exists only on `claude/m2-first-cell`
(`3e97182` + uncommitted edits in that worktree, read-only here): `rwkv7_wkv: "chunked"`, head `fork`, LoRA r=64,
`loss_arm: clef`, batch 1 × grad_accum 8, `t_ladder` 512/1024/2048, `max_k` 16, seed 0, and `train_config` sets
`grad_checkpointing=False`, `fork_mode="cache"`, `bf16` on CUDA. `train/trainer.py`, `heads/*` and `train/losses.py`
are identical on `agent-main` and that branch, so the job's code runs the cell's step. (`warmup_steps` only shapes the
learning rate, which does not change a step's cost; agent-main's `train_config` predates it.)

**The row mix (`train_mix_v2.json`, `train_mix.py`).** The training rows are private and do not ship. The table holds
per row only integers and the question type: prefix length, continuation lengths, spans and qtype. It was built once,
locally and CPU-only, by the cell's own code path (`train.train_set.draw` → `cell_items` at cap 2,048 →
`Trainer._order(0)` = `randperm(n, seed·100003)`). It keeps the first (3 + 20) × 8 = 184 rows, which are the rows of
the cell's first 23 optimiser steps (assuming the ladder keeps T_max = 2,048). The job rebuilds rows of exactly those
shapes with token ids from a fixed generator; an RWKV-7 step's cost does not depend on the ids. The table's sha256 is
checked on load and recorded in every `train_v2_<impl>.json`.
Built 2026-10-06 (local CPU, ~25 min serial tokenisation) from cell config sha256 `01d59a97…ecc6` against the pinned
150k set: 21,348 kept rows (mean T 605, max 2,048, mean K 3.46). The 184 pinned rows: mean T 677, max 2,025, mean K
3.61, max K 10, 124,586 prefix tokens (5,417 per step, 3,489–8,535), 34 rows > 1,024 and 11 > 1,536 tokens; 7 of the
20 timed steps hold a row ≥ 1,883 tokens. Table sha256 `5e88fee3…433a`.

## Bottleneck check (`jobs/PERF_AUDIT.md` convention)

Where the time goes in mode truth_v2 (projections; measured phases from job-jn47u):

| Piece | Work | Estimate | Memory |
|---|---|---|---|
| fp64 truth fwd, T=1024, per layer | 1,024 serial steps × ~12 launches; per step 64 heads × (64×64 @ 64×64) ≈ 34 MFLOP fp64, negligible on H100 → launch-bound | ~12k launches × ~15 µs ≈ **0.2 s** | out + state, < 0.1 GB |
| fp64 truth + autograd, T=512, per layer | forward as above with the graph kept, backward ≈ 2× the launches | ≈ **0.5 s** | saved per step: previous state, `anti_value_key`, post-update state (each 1×64×64×64 fp64 = 2.1 MB) ≈ 6.3 MB × 512 ≈ **3.2 GB**, < 5 GB with transients |
| 5 layers of truth | | **≈ 3.5 s** | peak < 5 GB of 80 (model freed first) |
| eager_bf16 / chunked / fla × 10 cases | jn47u: 3 layers' captured cases (same impls incl. the token-serial bf16 eager) took 3.6 s | ≈ 6 s | as jn47u |
| model load + 2,048-token capture pass | jn47u model phase ~31 s | ≈ 40 s | 27 GB |

So the fp64 token-serial truth is NOT the bottleneck: ~4 s of a 3-min cap, launch-bound, ~3 GB.
`truth_v2.json` records `truth_s` and `truth_peak_gb` per case to confirm it.

The job's real time sinks are the unchanged fla warmup (5.7 min measured) and the three training-speed processes:

| Process | Per step (8 rows) | 23 steps + load | Basis |
|---|---|---|---|
| train_chunked | prefix 5,417 tok × 3 (fwd+bwd) / 3,760 tok/s ≈ 4.3 s + 8 fork calls (B=K, ≤ 16 tokens; ~0.12 s fwd → ~0.35 s fwd+bwd, launch-bound) ≈ 2.8 s → **≈ 7 s** | ≈ 2.7 + 1 ≈ **4 min** (cap 6) | jn47u/tkfaj prefill rates; PERF_AUDIT single-call fork |
| train_fla | 5,417 / 4,561 tok/s (jn47u fwd+bwd T=2048) ≈ 1.2 s + fork calls ≈ 2.8 s → **≈ 4 s** | ≈ 1.5 + 1 (+ compile headroom) ≈ **3 min** (cap 6) | jn47u train fla |
| train_fla_safe_gate | as fla; its kernels share fla's compile (jn47u warmup: 2.6 s vs 72.7 s) | ≈ **3 min** (cap 5) | jn47u warmup |

Pre-mortem from these numbers (not part of the rule): on rows that fit, the per-row fork call is launch-bound and fla
does not speed it, so the projected chunked/fla step ratio is only **~1.8×**, below 2. Under Amendment A1 the binding
ratio uses only steps BOTH impls completed, so chunked OOMs cannot raise it. chunked OOMed on the T=2,048 fwd+bwd in
jn47u (80 GB), and 7 of the 20 timed steps hold a row ≥ 1,883 tokens. If chunked OOMs on them, those steps leave the
matched set (13 remain, ≥ 10 needed). They are reported as the non-binding `chunked_oom` finding: the m2 cell's own
ladder could not keep T_max = 2,048 on chunked. With > 10 chunked OOM steps the verdict is INCONCLUSIVE.
No compile is expected in the train phases: jn47u showed T is not a specialisation (warmup T=1000 recompiled nothing)
and safe_gate reuses fla's kernels; the 3 untimed warmup steps absorb any that does occur.

## Cost (`python -m jobs.common.costs` convention)

`job_params_truth_v2.json`: H100 batch $0.0495/min, build 18 min billed, `max_restarts` 0 with an assumed restart floor
of 1 (2 attempts).

| Item | Minutes | $ |
|---|---|---|
| **Worst case** = (build 18 + 2 × 49) | 116 | **$5.74** (≤ hard cap $6.00) |
| One attempt at the cap (what `validate_job` reports) | 49 | $2.43 |
| Expected run (sum of `expected_minutes`, build excluded as in v1) | 24.0 | $1.19 |
| Expected incl. a ~5 min cached build | ~29 | ~$1.44 |

| Phase | Cap | Expected | Cap / expected |
|---|---|---|---|
| setup | 3 | 2.3 (jn47u 2:17) | 1.30 |
| warmup | 13 | 5.7 (jn47u 341 s) | 2.28 |
| kernel | 2 | 0.5 (jn47u 12 s) | 4.0 |
| fork | 2 | 0.5 (jn47u 17 s) | 4.0 |
| download_wait | 2 | 1 (jn47u ~24 s sha256) | 2.0 |
| model | 3 | 1 (jn47u ~31 s) | 3.0 |
| throughput | 3 | 1.5 (jn47u 81 s) | 2.0 |
| truth_v2 | 3 | 1.5 (projection, below) | 2.0 |
| train_chunked | 6 | 4 (projection, below) | 1.5 |
| train_fla | 6 | 3 (projection, below) | 2.0 |
| train_fla_safe_gate | 5 | 3 (projection, below) | 1.67 |
| reserve | 1 | | |

Before submitting: `set_workspace_cap(MTD + 8)` (re-priced 2026-10-09 at $0.063/min; was MTD + 6; cap now $7.64); `validate_job` must pass the cost-cap preflight without `force`.

## Exit codes (mode truth_v2)

| Code | Meaning |
|---|---|
| 0 | `decision_v2.json` reached; any v2 verdict counts (ADOPT_FOR_TRAINING … NOT_ADOPTED, INCONCLUSIVE) |
| 5 | setup (pins, fla install, #640 guard, job_params), or an unknown mode |
| 6 | export failed |
| 7 | phase timeout (setup … throughput, truth_v2) |
| 9 | decision_v2 missing or invalid |
| 10 | snapshot or vendored-code mismatch, or the download failed |
| 30–34 | warmup / kernel / fork / model / throughput crashed (as v1) |
| 35 | truth_v2 crashed |
| – | a train_v2 crash or timeout is **not** an exit code: its rc is in the gates, its measurement is missing or partial, and decide_v2 says INCONCLUSIVE (or uses the timed steps the current impl reached, per the rule) |
| 1 | finalize failed |

Marker `/output/FLA_RWKV7_TRUTH_V2_DONE` (finalize.py, only on exit 0).

## Platform call (truth_v2; coordinator, after merge; not done here)

```
jobs/fla_rwkv7/build_context.sh <sha> <dir>  -> create_context(<dir>.tar.zst) -> context_id
submit_job(chip="h100", chip_count=1, mission="m2-fla-rwkv7-kernel", idempotency_key="m2-fla-rwkv7-truth-v2-<context_sha256[:12]>", context_id=<id>, dockerfile_path="Dockerfile",
  build_timeout_minutes=18, command="bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh truth_v2", max_duration_minutes=49,
  max_restarts=0, resume="none", queue_ttl_minutes=1440, hf_cache=true,
  model="RWKV/RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba",
  checks={"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/FLA_RWKV7_TRUTH_V2_DONE"},
                                           {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
          "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                     "interval_s": 300, "grace_period_s": 600, "fail_threshold": 2, "on_fail": "kill"}})
```

## Re-pricing 2026-10-09: H100 batch $0.0495 -> $0.063/min

SF Compute raised the H100 list price $3.96 -> $5.04/h on 2026-10-08; jobs now bill $0.063 per GPU-minute (Mon-Thu and weekends; Fridays $0.05628) and the preflight quotes max_duration x $0.063. Evidence and history: `jobs/common/rates.py`. The dollar figures above this section are at the old $0.0495/min and stay as written (receipts and past decisions used them); the numbers below are the current ones (`python -m jobs.common.costs`). Where a hard cap was raised, it was raised by the same ratio as the rate (x 0.063/0.0495 = x 14/11, rounded up to the cent), so the cap still buys the same minutes.

| Params | Worst case (build + 2 attempts) | Hard cap | Expected | One attempt (quoted max_cost) |
|---|---|---|---|---|
| `job_params.json` | $5.04 (was $3.96) | $5.73 (was $4.50) | $1.39 (was $1.09) | $1.95 (was $1.53) |
| `job_params_truth_v2.json` | $7.31 (was $5.74) | $7.64 (was $6.00) | $1.51 (was $1.19) | $3.09 (was $2.43) |
