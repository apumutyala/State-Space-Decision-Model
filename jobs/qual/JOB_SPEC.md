# Rung-2 hardware qualification job (≤$20). Spec only, NOT submitted

**Scope (issue #3 item 5):** numerics, throughput, memory and artifact recovery on one H100. Nothing else is in this job:
- no conversion curve and no capability claim;
- no m0 / debias / steer phase (dropped to keep the job pure qualification; those get their own job);
- the 0.8B cost-of-attention-freedom curve (`configs/m1_pilot_0p8b_h100.json`) is a later, separate job with its own spend issue, after this receipt merges.

All numeric parameters live in `jobs/qual/job_params.json`. `tests/test_qual_job.py` checks the cost arithmetic and the schedule against them.

## Platform call: SF Compute batch `submit_job`

```
open_mission("m1-hw-qual-h100")
create_context(<jobs/qual/build_context.sh output .tar.gz>)  -> context_id   (immutable: sha256 printed by the builder)
submit_job(
  chip="h100", chip_count=1, mission="m1-hw-qual-h100",
  idempotency_key="m1-hw-qual-<m1_sha[:12]>",
  context_id=<id>, dockerfile_path="Dockerfile", build_timeout_minutes=20,
  command="bash /opt/ssdm/m1/jobs/qual/qual_job.sh",
  max_duration_minutes=180, max_restarts=0, resume="none", queue_ttl_minutes=1440,
  hf_cache=true, model="Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
  checks={"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/QUAL_PASS"},
                                           {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
          "health":  {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                      "interval_s": 600, "grace_period_s": 900, "fail_threshold": 2, "on_fail": "kill"}})
```

## Cost ceiling: build time included

| Item | Minutes | $ at the batch rate ($0.0495/min) |
|---|---|---|
| Image build (`build_timeout_minutes`), counted as billable to be safe | 20 | 0.99 |
| Run attempt (`max_duration_minutes`) | 180 | 8.91 |
| **Worst case with `max_restarts=0`** | 200 | **9.90** |
| Worst case if the platform floors restarts at 1 (build + 2 attempts) | 380 | **18.81** (≤ $20) |

- Docs say only the running attempt is billed. We count the build anyway, so the cap does not rely on that.
- `validate_job` (free; receipt `validate_job_2026-10-06.json`) with these parameters returned `max_cost_usd: 8.91` for the run. Both preflight checks passed:
  - cost-cap;
  - disk-model: 9B ≈ 18.0 GiB at the pinned revision vs a 100 GiB layer.
- That preflight was scoped to the org **balance** ($525), not to a $20 cap.

**REQUIRED before submit (both, recorded in the issue #3 launch comment):**
1. An org admin runs `set_workspace_cap(workspace, cap_usd = month-to-date spend + 20)`.
   - This is the hard ceiling: the platform cap-kills at it. It also covers preemption requeues, which do not consume `max_restarts`.
2. **Re-run `validate_job` after setting the cap.**
   - Its cost-cap preflight must then show `scope` = the workspace cap and `verdict: pass` for the $9.90 worst case.
   - If it BLOCKs, do not pass `force`. Fix the cap or the duration first.

Auto top-up stays off. `queue_ttl_minutes=1440` is the start deadline: the job expires unbilled if it has not started within 24 h. The $500 promo credit expires 2026-10-27.

## Required gates

Exit codes propagate. `QUAL_PASS` is written only by `jobs/qual/finalize.py`, and only when its final code is 0.

| Phase | Cap (job_params) | Gate | On failure |
|---|---|---|---|
| setup | 15 | Image build pins (a failed pip pin fails the build). In-job checks: context manifest, `nvidia-smi`, version pins asserted, CUDA available, `job_params.json` readable, pinned fetch of 0.8B / wikitext / MMLU | exit 5 |
| A numerics | 5 per test × 5 | `test_gdn` (fla Triton vs fp32 on H100, module vs HF on real weights), `test_init_shapes`, `test_progressive_resume`, `test_guard`, `test_qual_job` | exit 2 |
| B 0.8B | 20 | `convert.qual --model 0p8b`: stage 1, then one-layer stage 2 (5 min), numerics sanity, throughput and memory 1k–8k | 2 / 3 / 4 propagate; timeout → 7 |
| C 9B | 25 weight wait + 80 | `convert.qual --model 9b`: stage 1 100 steps @8×4096, one-layer stage 2 for 35 min (`clone_neighbor`), numerics sanity, throughput and memory 4k–32k | weights → 8; 2 / 3 / 4 propagate; timeout → 7 |
| D decision | — | `decide_9b.py` must exit 0 **and** `finalize.py` must re-validate `decision.json`: known verdict, numeric MFU and peak | exit 9 (no `QUAL_PASS`) |
| final | 8 reserve | `finalize.py`: any export with `ok:false` → 6; finalize crash → 1; a stale `QUAL_PASS` is deleted | — |

Schedule check: 15 + 25 + 20 + 25 + 80 + 8 = 173 ≤ 180.

The in-code stop conditions are in `align.Guard`:
- **Exit 3:** 9B steady tok/s after 50 steps is below 50% of the 25%-MFU prediction (stage 1 6,150 tok/s; stage 2 1,450 tok/s).
- **Exit 4:** peak memory above 76 GB.
- **Fail fast:** `vram_fraction` is 0.95, so an out-of-memory error stops the job instead of degrading it.

**Artifact recovery (qualified, not assumed):**
- After every phase the runner tars `out/` and `logs/` into `/output/phases/<phase>.tgz`.
- It re-reads each archive and sha256-hashes it twice.
- Each result is recorded `{phase, sha256, bytes, ok}` in `/output/MANIFEST.jsonl`, also published through `POST /v1/sample`.
- `$GMN_RESULT_PATH` carries `{ok, exit_code, runner_rc, gates, exports_verified, decision}`.
- After the job, the orchestrator downloads the artifact and re-verifies every sha256 locally. That match is the recovery proof.

## Pre-registered decision (`configs/qual_h100.json: decision_rule`)

| Measured 9B stage-2 result | Action |
|---|---|
| MFU ≥ 25% and peak ≤ 60 GB | Open the $40 one-layer-probe `spend` issue at the measured rate |
| MFU 12.5–25%, or peak 60–76 GB | Re-price to the lean plan first |
| MFU < 12.5%, peak > 76 GB, or exit 3/4 | Fold |

## Immutable references

**Code.** Built by `jobs/qual/build_context.sh <m1_sha>`.
- It refuses any SHA that is not an ancestor of `origin/agent-main`.
- It writes `CONTEXT_MANIFEST.json`: the SHA plus the sha256 of the config, Dockerfile, runner and `job_params.json`.
- It builds a reproducible tar and prints its sha256.
- **Code identity:** the launch SHA is the agent-main merge of the pinned-snapshot fix. It is recorded with the context, config and Dockerfile hashes in the issue #3 launch receipt. Both model dirs in `configs/qual_h100.json` name the exact pinned snapshot revisions (0.8B `2fc06364…`, 9B `c2022362…`), and `resolve_model_dir` refuses an ambiguous multi-snapshot parent.

**Image.** `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel@sha256:09c0260351b12b617408ff5711e08d88ccdbb9a21f7f6d7de67e39039b022ae1` (Ubuntu 24.04, system Python 3.12, `PIP_BREAK_SYSTEM_PACKAGES=1`, `FLA_TILELANG=0`). Pins:
- torch 2.13.0 (+cu126), triton 3.7.1 (asserted at build time by a `RUN python -c ...` line)
- transformers 5.17.0, flash-linear-attention 0.5.2, fla-core 0.5.2

Why not the original torch 2.8.0 / triton 3.4.0 image: SF job-jbw22 (first H100 run) stopped at the numerics gate
because fla 0.5.2 refuses the gated `chunk_bwd_dqkwg` backward on Hopper with 3.4.0 <= Triton < 3.7.1 (fla #640:
those Triton releases miscompile it). torch 2.13.0 is the first torch line whose wheel pins triton 3.7.1. Setup runs
`jobs/common/fla_guard.py`, which re-derives fla's #640 rule from fla's own flags, requires Triton >= 3.7.1 and
the Triton (not TileLang) backward, and runs one tiny gated fwd+bwd; any failure is exit 5 before phase A.
- einops 0.8.2, pyarrow 25.0.1, huggingface_hub 1.32.0, safetensors 0.8.0, accelerate 1.15.0, numpy 2.2.6

**Weights and data.** Pinned by revision in `scripts/sf_fetch.py`. Data file sha256s are recorded by `convert/data.py`.

| Repo | Revision |
|---|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Salesforce/wikitext | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| cais/mmlu | `c30699e8356da336a370243923dbaf21066bb9fe` |

The two models' `tokenizer.json` files are byte-identical.

## Phase-C re-run after the job-c4zkm 9B OOM (`QUAL_MODE=c_only`)

job-c4zkm ($0.74) PASSED phase A (all GPU tests, including test_gdn on torch 2.13 / triton 3.7.1) and phase B (0.8B). Phase C, the 9B, ran out of GPU memory at stage-1 step 1: 8×4096 tokens, peak 68.9 GB, against a 75.2 GB process cap.

**Root cause.** All 8 student GDN graphs were alive at once at the full 32,768-token batch: 1.29 MB per token of activations. That came on top of 17.9 GB of weights and 8.6 GB of fp32 GDN parameters plus Adam state. `convert/memmodel.py` reproduces the observed peaks:

| Run | Model | Observed |
|---|---|---|
| 9B stage 1 | 68.9 GB | 68.9 GB |
| 0.8B stage 1 | 11.7 GB | 11.8 GB |
| 0.8B stage 2 | 6.5 GB (upper bound) | 5.7 GB |

**Fix** (`convert/align.py`, `convert/model.py`, `convert/qual.py`; the CPU tests `tests/test_microbatch.py` and `tests/test_memmodel.py` cover each item):

- **Gradient accumulation.** Micro-batch 4×4096 accumulated to the same 8×4096 tokens per step. The losses decompose exactly, so the gradient equals the full-batch one.
  - Stage 1 uses per-sequence relative MSE.
  - Stage 2 uses the KL summed over tokens, divided by the full-batch token count.
- **Stage 1.** Each layer's student graph is back-propagated inside its hook, so one graph is alive instead of 8. The teacher runs under `no_grad` and stops after the last aligned layer.
- **Stage 2.**
  - The prefix before the first student layer is identical for teacher and student, so it runs once under `no_grad` and is shared.
  - The teacher suffix runs under `no_grad`.
  - Only the student suffix builds a graph, with activation checkpointing.
  - Only the converted layer's GDN requires grad; the other GDNs are frozen.
  - It converts the *first* attention layer (L3), the near-worst-case suffix.
- **Allocator and teacher copy.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (2.8 GB was reserved but unallocated at the OOM). The teacher and student already share one frozen copy of the model (`MixerSwap`).

**Expected** (`memmodel.plan`, FRAG 1.10 included):

| Stage | Peak | FLOP/token | tok/s at 25% MFU | Exit-3 floor (12.5% MFU) |
|---|---|---|---|---|
| Stage 1 | **31.4 GB** | 17.1 GF | 14.5k | 7.2k |
| Stage 2 | **33.0 GB** | 61.8 GF | 4.0k | 2.0k |

- Guards use `min_mfu: 0.125`, re-derived from each stage's FLOP/token.
- Decision MFU uses the FLOP/token recorded in the result, not the old 10N = 85 GF.
- Risk: the 0.8B ran at about 10% MFU in stage 1. If the 9B does not do better, the pre-registered exit 3 (fold) fires.

**Job:** the same `submit_job` with these differences:
- `env={"QUAL_MODE": "c_only"}`;
- `max_duration_minutes=140`, from `jobs/qual/job_params_c_only.json`: setup 15, CPU gates 5×2, weight wait 25, C 80, reserve 8.

The run skips `test_gdn` and phase B. It re-runs only the CPU gates that cover the changed code (`test_progressive_resume`, `test_guard`, `test_qual_job`, `test_microbatch`, `test_memmodel`), then phases C and D.

**Cost:**
- One attempt: **$6.93**.
- With the 20-min build counted: **$7.92**.
- Worst case with one platform restart: **$14.85**.
- Expected: about $3.5 (about 70 min).

The workspace-cap precondition and the `validate_job` re-run apply unchanged.

## Preconditions (all required)

1. #2/#6/#7/#8/#9 merged, and the SHA filled in.
2. A LEDGER planned row.
3. `set_workspace_cap` = MTD + $20, **then** `validate_job` re-run and passing against that cap.
4. Codex acknowledges on issue #3.
