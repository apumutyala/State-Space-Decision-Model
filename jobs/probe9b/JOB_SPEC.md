# probe9b: one-layer Qwen3.5-9B conversion probe (<= $40). Spec only, NOT submitted or validated

**Runs only after** the rung-2 qualification receipt says `open-$40-one-layer-probe-spend-issue` (`PROBE_CONFIG=full`)
or `re-price-lean-plan` (`PROBE_CONFIG=lean`, re-priced first). Mapping: `jobs/DECISION_TREE.md`.

## What runs (`jobs/probe9b/probe_job.sh`)

| Phase | Cap (min) | Content | On failure |
|---|---|---|---|
| setup + weights | 10 + 20 | context manifest, `nvidia-smi`, pins (as qual) + kit import, DEV rows sha256, the pinned 2,008-row DEV subset materialised, `sf_fetch --stage probe9b` (wikitext, MMLU, 9B @ pinned revisions) | 5 / 8 / 10 |
| P | 140 | `convert.pipeline` with `configs/probe9b_h100.json`: stock probes; stage-1 alignment of all 8 attention layers (<= 400 steps, 25 min); **KL-importance ranking** (KL(teacher \|\| single-swap student) on the gate split); the least important layer converted with **MOHAWK-style stage-1 alignment + end-to-end KL** (stage 2, 60 min); **gate, `policy: enforce`** (dNLL <= 0.05, cumulative <= 0.15, hidden cos >= 0.95: a failing layer is rolled back); probes on the converted model; prefill throughput + peak memory 4k-32k. `DEADLINE_MINUTES=128` | 2 / 3 / 4 / 7 |
| I | 40 + 40 | Decision Index on the DEV subset through `harness`-compatible kit runs: **stock arm** and **converted arm**, both via `convert.readout` (same code path, only the layer modes differ), engine `jobs.common.readout_engine`. Measured stop conditions (`jobs/common/rate_guard.py`): **preflight** from the pipeline's measured stock prefill tok/s (subset proxy tokens x 2 / tok/s must fit 95% of the cap); **live**: after 8 min the rows written so far project the finish; past 95% of the cap -> stop | 12 / 7 / 15 |
| S | 10 | `jobs.common.dev_subset compare` (paired source-case cluster bootstrap, B=2000), `score`, `ece` per arm | 13 |
| D | - | `scripts/decide_probe9b.py`; `finalize.py` re-validates; `PROBE_DONE` only on exit 0 | 9 |

Schedule: 10 + 20 + 140 + 40 + 40 + 10 + reserve 10 = 270 = `max_duration_minutes`. Stop rules (in code, `align.Guard`):
exit 3 if steady tok/s after 50 steps < 50% of the 25%-MFU prediction (stage 1: 6,150; stage 2: 1,450; lean: half of
those); exit 4 if peak > 76 GB; `vram_fraction` 0.95 so an OOM fails fast. A rejected layer is **not** a failure: the
job exits 0 with the verdict `quality-fail` and skips the Index arms.

Measured on the converted model vs stock (all in `results.json` / `index/`):
- held-out perplexity: wikitext-2 test, ctx 1024 (64 windows) and 4096 (16 windows);
- MMLU-style MC subset: 5 per subject x 57 = 285 questions, 0-shot letter logit;
- MQAR: 8/32/64/128 pairs, 20 sequences each;
- S-NIAH: 1k, 2k, 4k, 8k, 16k (lean: to 8k), 3 depths x 8 = 24 needles per length;
- Decision Index stock readout, DEV-subset (2,008 rows, 1,899 groups, all 38 benchmarks; `jobs/common/dev_subset_m1.json`,
  ids sha256 `6e4cb8df...95c3` = sha256 of the sorted run_ids joined by newlines; built from the DEV manifest whose
  uncompressed text hashes to `26cd6bc0...964a` (the value in harness/manifests/slices.json) and whose .gz bytes
  hash to `1362c23e...073a`; both recorded, the uncompressed one checked on every load). **No TEST rows, no D1-B**: scoring loads `--slice dev` only.

## Pre-registered success rule (`configs/probe9b_h100.json: success_rule`)

| Check | Proceed requires |
|---|---|
| Gate (enforced) | layer accepted |
| Perplexity ratio converted / stock | <= 1.02 at ctx 1024 and 4096 |
| MMLU subset | drop <= 3.0 points |
| Index DEV subset, paired diff (converted - stock, 0-100) | point >= -1.0 and 95% CI lower bound >= -2.5 |
| MQAR | mean accuracy drop <= 0.10 |
| S-NIAH | mean drop <= 0.10 and every length's drop <= 0.25 |

Verdicts: `proceed-all-8` (all pass) / `retrieval-fail` (gate, quality and Index pass; MQAR or NIAH fails) /
`quality-fail` (gate rejected, or ppl, MMLU or Index fails). Missing numbers give `no-measurement` -> exit 9, never a
verdict. The rule is identical for the lean config (tested).

## `submit_job` fields (do not submit until the spend issue is approved)

Context: `bash jobs/common/build_context.sh probe9b <m1_sha> <dir>` (code from `git archive` of a SHA merged into
`agent-main`, the PRIVATE DEV rows, `CONTEXT_MANIFEST.json`, reproducible `tar.zst`); `create_context`, upload,
`finalize_context`; `open_mission("m1-probe9b-one-layer")`.

```json
{"chip": "h100", "chip_count": 1, "mission": "m1-probe9b-one-layer",
 "idempotency_key": "m1-probe9b-<full|lean>-<context_sha256[:12]>",
 "label": "probe9b: one-layer Q9-AF conversion probe (<full|lean>)",
 "context_id": "<from create_context>", "dockerfile_path": "Dockerfile", "build_timeout_minutes": 30,
 "command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "full"},
 "max_duration_minutes": 270, "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
 "hf_cache": true, "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a", "clock_lock": true,
 "checks": {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/PROBE_DONE"},
                                             {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
            "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                       "interval_s": 600, "grace_period_s": 1800, "fail_threshold": 2, "on_fail": "kill"}}}
```

A `depends_on` on the qual job is possible (`require: "result.decision.verdict == ..."`, `allow_different_context: true`)
but is **not** used: the probe needs a human-approved spend issue after the qual receipt anyway.

## Health check (`checks.health`) and heartbeat, field by field

- `type: heartbeat`, `path: /output/heartbeat`: the platform reads the file's freshness from outside the container.
  The runner never vouches for liveness itself; `jobs/clef_baseline/watchdog.sh` rewrites the file every 60 s **only
  while** the main process is alive **and** some file under the watched dirs (`$OUT`, `$CK`, `$HF_DIR`,
  `WATCH_EXTRA=$WORK`; `find -H` follows a symlinked HF cache root) changed in the last `STALL_S` = 1,200 s.
- `interval_s: 600`: the platform evaluates the heartbeat every 10 min.
- `fail_threshold: 2`: two consecutive stale evaluations trigger `on_fail`; one late write is tolerated.
- `grace_period_s: 1800`: no evaluation during the first 30 min of an attempt (image start, weight download and
  sha256, model load write few files).
- `on_fail: kill`: the attempt ends `self_terminated` (its own terminal state, billed to that point).
- Worst-case detection of a hung worker: `STALL_S` + 2 x `interval_s` = 20 + 20 = 40 min after its last file write
  (plus up to the 30-min grace if it hangs at start), instead of billing to `max_duration`. A dead main process stops
  the heartbeat at once.
- Every phase writes under a watched dir (kit: one results row per request; pipeline: `train_log.jsonl` every 25
  steps; `run_cell`: `train_log.jsonl` every step; probes and scoring: per-phase JSON). `tests/test_m1_next_jobs.py`
  checks statically that every path a runner writes resolves to a watched dir, and runs the watchdog with progress
  only under `WATCH_EXTRA`.

## Cost ceiling

| Item | Minutes | $ at $0.0495/min |
|---|---|---|
| Build (`build_timeout_minutes`, counted billable) | 30 | 1.49 |
| One attempt (`max_duration_minutes`; `validate_job` should report this) | 270 | 13.37 |
| **Worst case, build + 2 attempts** | 570 | **28.21** |
| Expected (build 15, setup+weights 15, P 120, I 70, S 3) | 223 | ~11.04 |

**Hard cap: `set_workspace_cap(MTD + 40)`**, then re-run `validate_job` and require `scope` = workspace cap, `verdict:
pass`. The readout rate is not assumed: the preflight and the live row-rate check stop a too-slow arm with exit 15
(a throughput finding with its receipt) before it reaches the cap. After a 15, raising I to 60 + 60
(`max_duration` 310) keeps the worst case at $32.18 < $40. If rwkv7 stage A runs at the same time (qual pass
branch), the workspace cap must cover both: MTD + `costs.concurrent_cap_usd(["probe9b", "rwkv7_readout"])` = MTD + 45.

## Exit codes

0 measured + decided + exports verified | 2 pipeline numerics/crash | 3 throughput stop | 4 memory stop | 5 setup |
6 export verification | 7 phase timeout | 8 9B weights unavailable | 9 decision missing/invalid | 10 DEV rows or
subset identity | 12 Index kit run | 13 subset scoring | 15 readout too slow by measurement | 1 finalize itself.

## Artifacts and export verification

- `/output/phases/<phase>.tgz` after every phase, re-read and double-hashed into `/output/MANIFEST.jsonl` (also
  POSTed as samples). The orchestrator downloads the output tar and re-verifies every sha256 locally.
- `/output/gdn_converted.pt` (+ `.sha256`): the converted layer's GDN weights only, the starting point of the full-8 job.
- `$GMN_RESULT_PATH`: `{ok, exit_code, runner_rc, gates, exports_verified, decision, verdict}`.
- Re-check locally on CPU: `python -X utf8 -m jobs.common.dev_subset compare --candidate conv/results.jsonl
  --baseline stock/results.jsonl` must reproduce `compare_conv_vs_stock.json` exactly, and
  `python scripts/decide_probe9b.py ...` must reproduce `decision.json`. Independent re-run by the other agent before
  any "proceed" claim (COORDINATION).

## Known limitations (stated, not hidden)

- Distillation data is wikitext-103 only. PLAN's recall levers (JRT layout, chunked state, 20-30% retrieval tokens)
  are **not** in the probe; they belong to the full-8 job. A `retrieval-fail` here is therefore a fail without the
  levers, which is why it maps to the pre-registered fallback rather than to a silent hybrid.
- The subset Index is a 2,008-row paired comparison for a go/no-go, not a board number.
- The stock arm uses `convert.model.load` (the same `ConvertibleQwen` path as the converted arm), not
  `hf_readout.load_qwen35`, so the pair differs only in the layer mode. Its absolute score may differ slightly from the
  m0 stock readout; the paired difference is what the rule uses.
- NIAH has 24 needles per length and MMLU 285 questions: the bounds are deliberately loose (one layer of eight).
- Stage-2 lr runs on the wall-clock-aware cosine of `align._cosine_lr`; tokens trained depend on the measured rate.
