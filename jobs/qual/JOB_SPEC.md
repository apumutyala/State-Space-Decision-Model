# Rung-2 hardware qualification job (≤$20). Spec only, NOT submitted

**Scope (issue #3 item 5):** numerics, throughput, memory and artifact recovery on one H100. Nothing else.
- There is no conversion curve and no capability claim.
- The 0.8B cost-of-attention-freedom curve is a later, separate job (`configs/m1_pilot_0p8b_h100.json`). It needs its own spend issue, filed after this receipt merges.

## Platform call: SF Compute batch `submit_job`, not an interactive node

```
open_mission("m1-hw-qual-h100")
create_context(<jobs/qual/build_context.sh output .tar.gz>)  -> context_id   (immutable: sha256 printed by the builder)
submit_job(
  chip="h100", chip_count=1, mission="m1-hw-qual-h100",
  idempotency_key="m1-hw-qual-<m1_sha[:12]>",
  context_id=<id>, dockerfile_path="Dockerfile",            # FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283…
  command="bash /opt/ssdm/m1/jobs/qual/qual_job.sh",
  max_duration_minutes=200, max_restarts=0, resume="none", queue_ttl_minutes=1440,
  hf_cache=true, model="Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
  checks={"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/QUAL_PASS"},
                                           {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
          "health":  {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                      "interval_s": 600, "grace_period_s": 900, "fail_threshold": 2, "on_fail": "kill"}})
```

## Cost ceiling

The batch rate is $0.0495/min ($2.97/h); the interactive rate ($0.066/min) does not apply.
- **`validate_job` with these parameters (free, nothing submitted) returned `max_cost_usd: 9.90`.** Preflight passed both checks:
  - cost-cap: worst case $9.90 against the remaining $525.00 balance;
  - disk-model: 9B ≈ 18.0 GiB at the pinned revision, against a 100 GiB writable layer.
  - Receipt: `validate_job_2026-10-06.json`.
- The bill covers only the running attempt (the image build and queueing are not billed). `max_restarts=0` means a provider fault fails the job rather than starting a second attempt.
  - If the platform floors `max_restarts` at 1, the worst case is 2 × $9.90 = $19.80. That still fits ≤$20.
- **Preemptions do not consume `max_restarts`.** That is why the preflight above was scoped to `balance`, not to a $20 cap.
  - Before submitting, an org admin runs `set_workspace_cap(workspace, cap_usd = month-to-date spend + 20)` and re-runs `validate_job`.
  - Preflight then BLOCKs any worst case above the remaining cap, and the platform cap-kills at it (`cap_kill`). This is the hard $20 ceiling.
  - No top-up: auto top-up stays off (`get_billing`).
- `queue_ttl_minutes=1440` is the start deadline: the job expires unbilled if it has not started within 24 h. The $500 promo credit expires 2026-10-27.

## Required gates (exit codes propagate; the success check needs `QUAL_PASS`, written only on exit 0)

| Phase | Cap | Gate | On failure |
|---|---|---|---|
| setup | — | Image build pins (pip fails the build, before billing). In-job checks: context manifest, `nvidia-smi`, version pins asserted, CUDA available, pinned fetch of 0.8B / wikitext / MMLU | exit 5 |
| A numerics | 10 min each | `tests/test_gdn` (fla Triton vs fp32 reference on H100, module vs HF on real weights), `test_init_shapes`, `test_progressive_resume`, `test_guard` | exit 2 |
| B 0.8B | 20 min | `convert.qual --model 0p8b`: stage 1 100 steps @16×1024, stage 2 one layer for 5 min, numerics sanity (finite, KL ≥ 0) on gate windows, prefill throughput and peak memory 1k–8k, stock vs one-layer-converted | 2 / 3 / 4 propagate; timeout → 7 |
| 0 optional | 15 min total, not gating | m0 leftovers: RWKV7-0.1B and Mamba-3-187M debias controls, PR #8 canonical steer smoke on 0.8B. Fixture regeneration is dropped (CPU fp32: a later cpu-8 job) | recorded only |
| C 9B | 40 min weight wait, 90 min run | `convert.qual --model 9b`: stage 1 100 steps @8×4096, stage 2 one layer for 45 min (`clone_neighbor`), numerics sanity, throughput and memory 4k–32k | weights missing → 8; 2 / 3 / 4 propagate; timeout → 7 |
| D | — | `scripts/decide_9b.py` writes `decision.json` | — |

The in-code stop conditions are in `align.Guard`:
- **Exit 3:** steady tok/s after 50 steps is below 50% of the 25%-MFU prediction: 6,150 tok/s for stage 1 or 1,450 tok/s for stage 2 on the 9B.
- **Exit 4:** peak memory above 76 GB.
- **Fail fast:** `vram_fraction` is 0.95, so an out-of-memory error stops the job instead of degrading it.

The worst-case schedule is about 15 + 40 + 20 + 15 + 90 + 5 min, under the 200 min cap.

**Artifact recovery (qualified, not assumed):**
- After every phase (setup, A, B, 0, C, final) the runner tars `out/` and `logs/` into `/output/phases/<phase>.tgz`.
- It then re-reads each archive (`tar -t`) and hashes it twice (sha256).
- Each result is appended to `/output/MANIFEST.jsonl` as `{phase, sha256, bytes, ok}`.
- Any `ok:false` makes the exit code 6, so the success check fails.
- `MANIFEST.jsonl` is also published live through `POST /v1/sample`.
- `$GMN_RESULT_PATH` carries `{ok, exit_code, gates, exports_verified, decision}`.
- After the job, the orchestrator downloads the artifact and re-verifies every sha256 in `MANIFEST.jsonl` locally. That match is the recovery proof.

## Pre-registered decision (`configs/qual_h100.json: decision_rule`, CPU-tested)

| Measured 9B stage-2 result | Action |
|---|---|
| MFU ≥ 25% and peak ≤ 60 GB | Open the $40 one-layer-probe `spend` issue at the measured rate |
| MFU 12.5–25%, or peak 60–76 GB | Re-price to the lean plan first |
| MFU < 12.5%, peak > 76 GB, or exit 3/4 | Fold: cache teacher hidden states or fuse the KL head first |

## Immutable references

**Code.** Built by `jobs/qual/build_context.sh <m1_sha> <decide_sha> <steer_sha>`.
- The script refuses any SHA that is not an ancestor of `origin/agent-main`.
- It writes `CONTEXT_MANIFEST.json`: the three SHAs plus the sha256 of `configs/qual_h100.json`, the Dockerfile and the runner.
- It builds a reproducible tar and prints its sha256.
- **SHAs:** TBD. Fill them in once #2, #6, #7, #8 and #9 merge. agent-main is currently `4b72a5d`, and none of the code merged yet. Until then this spec is not launchable.

**Image.** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6`. Pins:
- transformers 5.17.0, flash-linear-attention 0.5.2, fla-core 0.5.2
- einops 0.8.2, pyarrow 25.0.1, huggingface_hub 1.32.0, safetensors 0.8.0, accelerate 1.15.0, numpy 2.2.6
- mamba_ssm @ e9594ce

**Weights and data.** Pinned by revision in `scripts/sf_fetch.py`. Data file sha256s are recorded by `convert/data.py`.

| Repo | Revision |
|---|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Salesforce/wikitext | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| cais/mmlu | `c30699e8356da336a370243923dbaf21066bb9fe` |
| fla-hub/rwkv7-0.1B-g1 | `7450ee04eb9c66baa7161b7a5c0fe09e5b2817d8` |
| state-spaces/mamba3-siso-187m | `6792c27c00f3bb41506db1066dcd1c51bb0f4b02` |
| unsloth/Llama-3.1-8B-Instruct (tokenizer) | `4699cc75b550f9c6f3173fb80f4703b62d946aa5` |

The Qwen3.5-0.8B and 9B `tokenizer.json` files are byte-identical (sha256 prefix `5f9e4d4901a9…`). The token files are therefore shared between the two models.

## Preconditions

1. #2, #6, #7, #8 and #9 merged, and the SHAs above filled in.
2. A LEDGER planned row.
3. The workspace cap set to MTD + $20, and `validate_job` re-run against it.
4. Codex acknowledges on issue #3.
