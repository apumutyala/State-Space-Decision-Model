# jobs/q9_feature_cache — frozen Qwen3.5-9B feature cache for the proxy head loop

**Status:** spec + runner + CPU dry run. Nothing has been submitted. Launch needs a merge into agent-main, an independent review, a LEDGER row and the cap raise (`jobs/LAUNCH_CHECKLIST.md` once merged).

**Why.** PREMORTEM blocker #1: can an attention-free head on frozen Q9 states approach Clef-flash (55.8 on the DEV subset; stock Q9 readout 38.3)? This job runs the frozen 9B once and stores the features the proxy heads need. Every head, loss and recipe iteration after that is CPU-only (`train/proxy_loop.py`). The decision rule was pre-registered before this job existed: `train/proxy_rule.py`, commit `e9db899`, `jobs/q9_feature_cache/PREREG.md`.

## Modes

The two modes are independent. Run them in parallel on two H100s.

| Mode | Arms | Needs | Worst case | Hard cap | Expected |
|---|---|---|---|---|---|
| `stock` (`job_params_stock.json`) | stock Q9 | nothing: independent of full-8. **Ship first** | **$21.29** | $25 | ~$6.0 |
| `extra` (`job_params_extra.json`) | `conv1` (layer 23 → GDN from art-2bybh) + `e5` (stock weights, short codes for opaque keys) | `gdn_converted.pt` (sha256 `8a50c36c…`) in the context | **$24.26** | $25 | ~$7.2 |

- Worst case = `costs.worst_case_usd`: (build_timeout 60 + 2 attempts × max_duration) × $0.0495/min.
- Expected = the volume below at the measured rates, plus a 15-min build.
- The ≲$5 target is met by neither mode alone. stock is ~$6.0 because 20k training rows cost ~39k model calls at the measured 0.0595 s per call. Dropping to 15k rows would save ~$0.6. We kept 20k as specified.

## What it stores (per arm; `jobs/q9_feature_cache/extract.py`)

Layout `ordered` (byte-identical to the m0 readout prompt), template `jrt-chatml-v1`, final post-norm hidden states. Tensors are bf16 unless noted.

Per question:

| Field | Shape | Content |
|---|---|---|
| `fork` | [K, D] | Fork feature after prefix + continuation_k. Equal to `HeadInput.fork` (tested) |
| `e` | [K, D] | Option-span mean. Equal to `HeadInput.e` |
| `lex` | [K, D] | Mean lm_head (output-embedding) row over the option span's tokens. Clef's lexical-prior input |
| `q_pre`, `q_end` | [D] each | Equal to `HeadInput.q_pre` / `q_end` |
| `lm_lp` | [K], fp32 | Summed log-likelihood of `" " + option_id + <\|im_end\|>`. This is the stock readout's own option score (E4). `softmax(lm_lp)` is the stock readout, recomputable on HOLDOUT |
| `lm_ntok` | [K] | Continuation token counts |
| `key_class` | — | letter / option_n / digits / hash / semantic (`heads/option_keys.py`) |
| `label` | — | Training rows only |

Per request, for every DEV request and the first 5,000 training rows: `banks`, up to 32 equal-count chunk means of each distinct evidence span (the state, or a question's re-read instruction when the state is empty). These are the "richer per-position features" for the routing heads, compressed from [T, D] to [≤32, D] so that 5k rows fit in an artifact.

Every shard (256 requests) is a weights-only `torch.save`. Its sha256 is recorded in `<arm>/<split>/MANIFEST.jsonl`. `report.py` and `proxy_loop.load_split` re-hash every shard before use.

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
4. Every record passes `SplitGuard.check` with the verified content-index guard (`jobs.m2_pilot.guard_ready`), on the ORIGINAL record, before tokenisation.
5. Rows are kept by the cell's own filter (`keep_item`: K ≤ 16, T ≤ 2,048), in draw order.

DEV rows are never labelled and never trained on. DEV questions are never truncated. A question longer than 32,768 tokens, or one that cannot be encoded, makes its request unanswerable, and that request scores as wrong (none exist in either split).

## Volume and price (measured rates; exact volume)

`volume_plan.json` was produced locally by `cache.py run --plan-only` on the real draw and the real DEV rows with the real Qwen3.5-9B tokenizer. That is the same code the job runs before its model loads.

| Arm | select pos / calls | holdout pos / calls | train pos / calls | extraction min | arm projection | arm cap |
|---|---|---|---|---|---|---|
| stock | 6.14M / 11,818 | 6.39M / 12,109 | 13.14M / 38,817 | 81.5 | 96.4 | 130 |
| conv1 | 6.14M / 11,818 | 6.39M / 12,109 | 3.28M / 9,731 | 45.3 | 60.2 | 80 |
| e5 | 5.87M / 11,816 | 6.12M / 12,108 | 3.14M / 9,694 | 44.7 | 59.6 | 80 |

The rates come from job-5cmat on the same image and the same 9B revision:
- prefill: 22,144 tok/s, the minimum over 4k–32k;
- per call: 0.0595 s. That is (548.9 s stock-readout kit wall time − 6.23M fork positions / 22,144) / 4,501 calls.

Arm projection = extraction + 14.2 min of tokenisation/draw (the maximum measured locally) + 0.7 min load. Every cap is ≥ 1.3× its projection, checked by `preflight.py` at setup and in `tests/test_q9_feature_cache_params.py`.

Inside each arm, `cache.py` re-projects the exact volume at the same rates after tokenising. It exits 15 before loading the model if the projection exceeds 95% of the cap. `jobs.common.rate_guard run-weighted` then stops the arm (exit 15) if the measured progress projects past the cap.

The extractor runs a question's re-read once and forks only the continuations when copying it K times would cost more than 512 positions. Without that, the DEV splits were 24.8M positions per split; with it they are 6.1M, about the same as the readout's own fork volume.

Estimated output: stock ≈ 6–7 GB, conv1 / e5 ≈ 4–5 GB each (`summary.json` `est_output_gb`, bf16 upper bound).

## Phases and exit codes (`cache_job.sh`)

1. **setup:** pins, fla #640 guard, static preflight (exit 5), DEV rows sha256, SELECT/HOLDOUT split identity, gdn sha256 (exit 10), weights via `sf_fetch.py --stage q9`.
2. **guard:** content index; exit 30/31/32.
3. **heldout:** D1-B scan of the whole set; exit 33.
4. **arm_<arm>**, under the rate guard:
   - 2: numerics self-check. The first 3 SELECT requests are recomputed with plain full-sequence forwards; fork rel diff must be ≤ 0.03 and LM score abs diff ≤ 0.15 in bf16.
   - 10: data identity.
   - 15: too slow.
   - 27 / 28: guard / leak.
   - 7: timeout.
   - 12: other.
5. **report:** every shard re-hashed; exit 9.
6. **finalize:** writes `Q9_CACHE_DONE` only at exit 0 with every export verified.

## Verification at $0

- `tests/test_q9_feature_cache.py`:
  - Extractor == m2 `build_head_input` features and == plain recompute (fork, e, q_pre, q_end, LM score), in both one-stage and two-stage modes.
  - The volume model equals the calls and positions actually made.
  - Empty-state banks, E5 round trip, shard tamper detection, partial conversion.
- `tests/test_q9_feature_cache_shell.py` **executes `cache_job.sh`** end to end (dry run, tiny Qwen3.5 stand-in). Every step runs: preflight, split check, DEV pins, rate guard, in-job preflight, self-check, shards, report, finalize. Then it trains every proxy head family on the cache the runner wrote and writes kit-format results.
- A local plan-only rehearsal on the real data (receipt `volume_plan.json`) exercised the real path except the GPU:
  - guard readiness: the content index reproduced the pinned `7cbe6c89…`;
  - `train_set.draw` with all its checks;
  - tokenisation of 24k records and 4,022 DEV requests.

## Launch (coordinator)

1. Merge.
2. `jobs/q9_feature_cache/build_context.sh stock <sha> <dir>`. For `extra`, also pass the art-2bybh `gdn_converted.pt`.
3. Run `create_context`, then `submit_job` with `chip h100`, `build_timeout_minutes 60`, `max_duration_minutes` from the params, `max_restarts 0`, `env {"Q9C_MODE": "<mode>"}`, `hf_cache true`, `mission q9-feature-cache`.
4. Download the artifact, verify `MANIFEST.jsonl`, then run `python -m train.proxy_loop stock-check` and then `recipes` (CPU).
