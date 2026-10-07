# jobs/q9_feature_cache — frozen Qwen3.5-9B feature cache for the proxy head loop

**Status:** spec + runner + CPU dry run. Nothing has been submitted. Launch needs a merge into agent-main, an independent review, a LEDGER row and the cap raise (`jobs/LAUNCH_CHECKLIST.md`).

**Why.** PREMORTEM blocker #1: can an attention-free head on frozen Q9 states approach Clef-flash (55.8 on the DEV subset; stock Q9 readout 38.3)? This job runs the frozen 9B once and stores the features the proxy heads need. Every head, loss and recipe iteration after that is CPU-only (`train/proxy_loop.py`). The decision rule was pre-registered before this job existed: `train/proxy_rule.py` (commit `e9db899`, pre-data amendment A1 `06abf1f`), `jobs/q9_feature_cache/PREREG.md`.

## Parts (one job = one part = one artifact of at most 4.5 GiB)

The platform packages a job's whole output directory as ONE tar artifact, and `max_artifact_bytes` is 5 GiB (`list_limits`: "the object store's single-PUT bound; cannot be raised past it today"). The org has no storage connection for `output_to` (`list_connections` is empty). So the cache ships as six independent jobs, one per arm × {DEV, train}, each sized well under the bound:
- `cache.py` refuses (exit 16) before the model loads if the exact estimate exceeds `max_output_gib` 4.5.
- It refuses again on the bytes actually written.
- `report.py` succeeds only if every part file listed in a MANIFEST exists and re-hashes, and the whole output directory is ≤ 4.5 GiB. `result.ok` alone is not trusted.

The parts can run in parallel.

| Part (`Q9C_MODE`) | Arm / splits | Needs | Est. output | Arm projection / cap (min) | Worst case | Hard cap | Expected |
|---|---|---|---|---|---|---|---|
| `stock_dev` | stock / select + holdout | nothing (independent of full-8). **Ship first** | 2.87 GiB | 48.1 / 65 | **$14.85** | $20 | ~$3.66 |
| `stock_train` | stock / train (20,000 rows) | nothing. **Ship first** | 2.77 GiB | 63.3 / 85 | **$16.83** | $30 | ~$4.41 |
| `conv1_dev` | conv1 / select + holdout | `gdn_converted.pt` (sha256 `8a50c36c…`) | 2.87 GiB | 48.1 / 65 | **$14.85** | $30 | ~$3.66 |
| `conv1_train` | conv1 / train (first 5,000 rows) | `gdn_converted.pt` | 1.61 GiB | 27.0 / 40 | **$12.38** | $30 | ~$2.62 |
| `e5_dev` | e5 / select + holdout | nothing | 2.87 GiB | 47.7 / 65 | **$14.85** | $30 | ~$3.61 |
| `e5_train` | e5 / train (first 5,000 rows) | nothing | 1.61 GiB | 26.9 / 40 | **$12.38** | $30 | ~$2.57 |

- Worst case = `costs.worst_case_usd`: (build_timeout 60 + 2 attempts × max_duration) × $0.0495/min. Every hard cap is at least 1.2× its worst case; the conv1 and e5 parts use $30.
- Expected = the exact volume at the measured rates, plus a 15-min build. The stock arm costs ~$8.07 (two parts) and all six parts ~$20.5. Splitting into parts adds about $1 of setup and model load per extra job. That is the price of the 5 GiB bound.
- The workspace cap for all six in parallel is MTD + 87; for the two stock parts, MTD + 32 (`costs.concurrent_cap_usd`).

## What it stores (per arm; `jobs/q9_feature_cache/extract.py`)

Layout `ordered` (byte-identical to the m0 readout prompt), template `jrt-chatml-v1`, final post-norm hidden states. Tensors are bf16 unless noted.

Per question:

| Field | Shape | Content |
|---|---|---|
| `fork` | [K, D] | Fork feature after prefix + continuation_k. Equal to `HeadInput.fork` (tested) |
| `e` | [K, D] | Option-span mean. Equal to `HeadInput.e` |
| `opt_tok`, `opt_off` | int32 | Option span token ids and offsets. The lexical feature (mean lm_head row over them, Clef's lexical-prior input) is rebuilt on CPU from the part's `lex_vocab_<part>.pt`, the lm_head rows of every option token in the part. It used to be stored as a [K, D] tensor per option (review B1) |
| `q_pre`, `q_end` | [D] each | Equal to `HeadInput.q_pre` / `q_end` |
| `lm_lp` | [K], fp32 | Summed log-likelihood of `" " + option_id + <\|im_end\|>`. This is the stock readout's own option score (E4). `softmax(lm_lp)` is the stock readout, recomputable on HOLDOUT |
| `lm_ntok` | [K] | Continuation token counts |
| `key_class` | — | letter / option_n / digits / hash / semantic (`heads/option_keys.py`) |
| `label` | — | Training rows only |

Per request, for every DEV request and the first 5,000 training rows: `banks`, up to 32 equal-count chunk means of each distinct evidence span (the state, or a question's re-read instruction when the state is empty). These are the "richer per-position features" for the routing heads, compressed from [T, D] to [≤32, D].

Every shard (256 requests) is a weights-only `torch.save`. Its sha256 is recorded in `<arm>/<split>/MANIFEST.jsonl`, and the lexical table's in `<arm>/summary_<splits>.json`. `report.py` and `proxy_loop.load_split` re-hash every shard and table before use. Parts of one arm unpack side by side into one cache directory.

## Rows

| Split | Source | Rows (stock) |
|---|---|---|
| `train` | The m2 first-cell draw (PR #74 `train.train_set.draw`, brought here byte-identical). Details below | 20,000 rows / 17,917 records (conv1 / e5: the first 5,000) |
| `select` | The 2,008-row m1 DEV subset (`dev_subset_m1.json`) | 2,008 requests / 8,467 questions |
| `holdout` | The pre-registered HOLDOUT split (`dev_holdout.json`, disjoint by kit group) | 2,014 requests / 8,680 questions |

How the `train` draw is built:
1. The pointer-verified 150k set (`sample-150000-seed20261006`).
2. Cohort-v2 exclusion by id and by normalised content.
3. The 3 pinned D1-B exclusions, plus the drawn-record held-out scan.
4. Every record passes `SplitGuard.check` with the verified content-index guard (`jobs.m2_pilot.guard_ready`) **and** `harness.heldout.check_record` (heldout-draw-v1: no record may target one of the 9 frozen held-out benchmarks; exit 28). Both run on the ORIGINAL record, before tokenisation. The build context also runs `harness.heldout check` over the whole shipped set.
5. Rows are kept by the cell's own filter (`keep_item`: K ≤ 16, T ≤ 2,048), in draw order.

DEV rows are never labelled and never trained on. DEV questions are never truncated. A question longer than 32,768 tokens, or one that cannot be encoded, makes its request unanswerable, and that request scores as wrong (none exist in either split).

## Volume and price (measured rates; exact volume)

`volume_plan.json` was produced locally by `cache.py run --plan-only` on the real draw and the real DEV rows with the real Qwen3.5-9B tokenizer. That is the same code the job runs before its model loads.

| Arm | select pos / calls | holdout pos / calls | train pos / calls |
|---|---|---|---|
| stock | 6.14M / 11,818 | 6.39M / 12,109 | 13.14M / 38,817 |
| conv1 | 6.14M / 11,818 | 6.39M / 12,109 | 3.28M / 9,731 |
| e5 | 5.87M / 11,816 | 6.12M / 12,108 | 3.14M / 9,694 |

The rates come from job-5cmat on the same image and the same 9B revision:
- prefill: 22,144 tok/s, the minimum over 4k–32k;
- per call: 0.0595 s. That is (548.9 s stock-readout kit wall time − 6.23M fork positions / 22,144) / 4,501 calls.

Part projection = its extraction + 14.2 min of tokenisation/draw (the maximum measured locally for a whole arm; DEV-only parts tokenise less) + 0.7 min load. Every cap is ≥ 1.3× its projection, checked by `preflight.py` at setup and in `tests/test_q9_feature_cache_params.py`.

Inside each part, `cache.py` re-projects the exact volume at the same rates after tokenising. It exits 15 before loading the model if the projection exceeds 95% of the cap. `jobs.common.rate_guard run-weighted` then stops the part (exit 15) if the measured progress projects past the cap.

Memory: option copies per call are capped so that copies × cached tokens ≤ 2^19 (about 16 GiB of attention KV at 32 KiB per token). A 28k-token DEV row is forked 18 options at a time instead of 32.

The extractor runs a question's re-read once and forks only the continuations when copying it K times would cost more than 512 positions. Without that, the DEV splits were 24.8M positions per split; with it they are 6.1M, about the same as the readout's own fork volume.

Estimated output (`cache.estimate_bytes`, a bf16 upper bound from the same plan; GiB): select 1.36 and holdout 1.38 per arm; train 2.51 (stock) or 1.47 (conv1, e5); one lexical table per part, 0.13–0.26.

## Phases and exit codes (`cache_job.sh`)

1. **setup:** pins, fla #640 guard, static preflight (caps and the ≤ 4.5 GiB output bound; exit 5), DEV rows sha256, training-set and suite sha256 (`*_train` parts), SELECT/HOLDOUT split identity, gdn sha256 (`conv1_*`; exit 10), weights via `sf_fetch.py --stage q9`.
2. **guard** (`*_train` parts): content index; exit 30/31/32.
3. **heldout** (`*_train` parts): D1-B scan of the whole set; exit 33.
4. **arm_<arm>**, under the rate guard:
   - 2: numerics self-check v3 (`extract.selfcheck`, LESSONS L-28). It covers every row of the part's first 3 requests (the first two rows of each, which share the prefix) and options {0, K//2, K-1}, compared against an fp32 truth. The truth uses fp32 weights and the pure-torch reference GDN and conv kernels with TF32 off. That swap fails closed: all 4 kernels must be swapped, and the implementation that actually ran is recorded. Weights are restored bit-exactly afterwards.
     - **Logic:** the cached fork computed in fp32 must equal the plain fp32 forward: fork rel ≤ 1e-3, LM ≤ 1e-3 per continuation token. q_end / q_pre / e must equal the plain fp32 hidden states within 5e-3.
     - **Cache:** the production bf16 cached fork may be no further from the truth than 1.5× the plain bf16 forward, plus 0.005 (fork) and 0.02 per token (LM).
     - **fla fp32 pass:** the cached fork with fp32 weights and the production kernels (fla, TF32 inside), bound 1.5× the plain fla-fp32 forward + 5e-3. **On this first launch it is RECORD-ONLY (`fla_gate: false`)**, because fla's fp32 noise on the 9B is unmeasured. It becomes a gate once that receipt exists.
     - Every record carries its ratios and the applied bounds (`summary_*.json` `selfcheck` / `selfcheck_info`).
     - An OOM in the fp32 pass, or a kernel that cannot be swapped, exits 2 with a stated reason. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set and the CUDA cache is emptied between passes.
     - The v1 check compared two bf16 computations against unmeasured bounds (0.03 / 0.15). It rejected a correct path on the 9B (job-8vjve, job-b7d3n) and could not detect an off-by-one KV bug. Receipt: `receipts/2026-10-07-D-q9-cache-fork-selfcheck/`.
   - 10: data identity.
   - 15: too slow.
   - 16: output over 4.5 GiB (estimate before the model loads, or bytes written).
   - 27 / 28: guard / leak (including a frozen held-out family).
   - 7: timeout.
   - 12: other.
5. **report:** every part file listed in a MANIFEST exists and re-hashes, the lexical table matches its summary, request counts equal the plan, and the output directory is ≤ 4.5 GiB; exit 9.
6. **finalize:** writes `Q9_CACHE_DONE` only at exit 0 with every export verified.

## Verification at $0

- `tests/test_q9_feature_cache.py`:
  - Extractor == m2 `build_head_input` features and == plain recompute (fork, e, q_pre, q_end, LM score), in one-stage, two-stage and KV-capped modes.
  - The volume model equals the calls and positions actually made.
  - The lexical feature rebuilt from the table equals the lm_head mean.
  - Empty-state banks, E5 round trip, shard tamper detection, partial conversion.
- `tests/test_q9_feature_cache_shell.py` **executes `cache_job.sh`** end to end for two parts (`stock_dev`, `stock_train`; dry run, tiny Qwen3.5 stand-in). Every step runs: preflight, split check, DEV pins, rate guard, in-job preflight and size gate, self-check, shards, lexical table, report, finalize. It then:
  - merges the two parts' caches;
  - checks that a tampered shard fails the report;
  - trains every proxy head family on the merged cache and writes kit-format results.
- A local plan-only rehearsal on the real data (receipt `volume_plan.json`) exercised the real path except the GPU:
  - guard readiness: the content index reproduced the pinned `7cbe6c89…`;
  - `train_set.draw` with all its checks, plus `harness.heldout.check_record` on every drawn record;
  - tokenisation of 24k records and 4,022 DEV requests.

## Launch (coordinator)

1. Merge.
2. Per part: `python -m jobs.common.launch q9_feature_cache --mode <part> --sha <sha> --data $SSDM_DATA --out <dir>` (or `jobs/q9_feature_cache/build_context.sh <part> <sha> <dir>`). The `conv1_*` parts also need the art-2bybh `gdn_converted.pt` placed at `$SSDM_DATA/q9-feature-cache/`.
3. Submit `stock_dev` and `stock_train` first, in parallel. Submit the other four whenever the coordinator chooses.
4. Download each artifact into one cache directory. Then run on CPU:
   1. `python -m train.proxy_loop stock-check` (A1(d): |Δ| ≤ 0.3, else no-measurement);
   2. `recipes`;
   3. `select`;
   4. `holdout --dev-rows ... --clef ...` (single shot). It re-derives the choice from the SELECT receipts and checks their hashes, scores, and the code identity. It validates every input before creating anything, then binds the attempt in `holdout_attempt.json`. After a crash, re-running resumes the same attempt.
5. Commit `select_summary.json`, `holdout_attempt.json` and `verdict.json` to `stack/q9/proxy-head` (`receipts/<date>-D-q9-proxy-holdout/`) as soon as each is written. Those commits are the once-only record.

## Exact submit_job fields (one block per mode; generated from job_params by jobs/common/launch.py, checked field by field in tests/test_launch.py)

The `checks` block is the platform CheckSpec the coordinator submitted stock_dev / stock_train with (LESSONS L-25): success = `/output/Q9_CACHE_DONE` and `/output/MANIFEST.jsonl`; health = heartbeat `/output/heartbeat`, 600 s, grace 1800 s, threshold 2, kill.

Mode `conv1_dev` (`job_params_conv1_dev.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-conv1-dev-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "conv1_dev"
 },
 "max_duration_minutes": 120,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```

Mode `conv1_train` (`job_params_conv1_train.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-conv1-train-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "conv1_train"
 },
 "max_duration_minutes": 95,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```

Mode `e5_dev` (`job_params_e5_dev.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-e5-dev-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "e5_dev"
 },
 "max_duration_minutes": 120,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```

Mode `e5_train` (`job_params_e5_train.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-e5-train-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "e5_train"
 },
 "max_duration_minutes": 95,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```

Mode `stock_dev` (`job_params_stock_dev.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-stock-dev-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "stock_dev"
 },
 "max_duration_minutes": 120,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```

Mode `stock_train` (`job_params_stock_train.json`):

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9-feature-cache",
 "idempotency_key": "q9-feature-cache-stock-train-<context sha256[:12]>",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9_feature_cache/cache_job.sh",
 "env": {
  "Q9C_MODE": "stock_train"
 },
 "max_duration_minutes": 140,
 "max_restarts": 0,
 "resume": "none",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9_CACHE_DONE"
     },
     {
      "type": "file_exists",
      "path": "/output/MANIFEST.jsonl"
     }
    ]
   }
  },
  "health": {
   "check": {
    "type": "heartbeat",
    "path": "/output/heartbeat"
   },
   "interval_s": 600,
   "grace_period_s": 1800,
   "fail_threshold": 2,
   "on_fail": "kill"
  }
 }
}
```
