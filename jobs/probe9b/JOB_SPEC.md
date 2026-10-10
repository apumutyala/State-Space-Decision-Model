# probe9b: one-layer Qwen3.5-9B GDN conversion probe (<= $40). Spec only, NOT submitted or validated

**Trigger.** Issue #68: job-4j3vu (m1 qual C-only, 9B, @3b4169a) exited 0 with verdict
`open-$40-one-layer-probe-spend-issue` (stage-2 MFU 0.341 >= 0.25, peak 31.9 GB <= 60 GB). So this runs
`PROBE_CONFIG=full`. The lean config stays for the `re-price-lean-plan` branch only. Mapping: `jobs/DECISION_TREE.md`.

**Question.** Can one of Qwen3.5-9B's 8 full-attention layers be replaced by a Gated DeltaNet (MOHAWK-style stage-1
alignment, then end-to-end KL distillation, accept/rollback gate) with quality kept well enough to justify converting
all 8 (the Q9-AF route)? It is the 9B analogue of the 0.8B one-layer pilot (job-c4zkm phase B: dNLL -0.0036,
KL 0.017, hidden cosine 0.991).

## Measured basis (job-4j3vu, artifact art-5ys6j, sha256 `6e758dbe...4f59`)

All numbers come from `out/qual_9b.json`, `logs/run.log`, `logs/qual_9b.log` and `logs/fetch_rest.log`. They are
recorded in `job_params.json: measured`. The run used the same image stack and the same 9B revision, with
micro_batch 4 x 4096 accumulated to 8 x 4096. Its stage 2 converted the **first** attention layer (layer 3), which
has the longest student suffix, so its stage-2 rate is the slowest that any probe layer can have.

| Quantity | Measured | Used for |
|---|---|---|
| stage-1 steady tok/s (all 8 GDNs aligned) | 18,952 (MFU 0.327); first step 92 s (fla autotune) | stage-1 minutes; stop floor |
| stage-2 steady tok/s (layer 3) | 5,461 (MFU 0.341) | stage-2 tokens per capped minute; stop floor |
| peak memory | 31.9 GB | `max_peak_gb` 76 leaves 2.4x |
| stock prefill, 4,096 tokens | 25,803 tok/s | probes, gate evals, Index preflight |
| model load / 9B fetch from hf_cache | 19.6 s / 28 s | setup and weights |
| build + start (submit to first runner line) | ~14 min | expected build minutes |
| 9B one-layer gate (layer 3, 8 windows, after 35 min stage 2) | dNLL -0.0005, KL 0.054, cos 0.980 | **disclosure only**, see the decision rule |

Not measured on the 9B (labelled as estimates in `preflight.py`): the probe-suite time, the gate-eval time, and the
Index per-call cost. The pipeline now measures the per-call cost (below).

## Fixes applied before shipping (`jobs/PERF_AUDIT.md` discipline)

1. **The committed probe would have OOMed.** `convert.pipeline` never passed `micro_batch` to stage 1 or stage 2.
   With `batch 8 x 4096` it would run the exact shape that OOMed at 68.9 GB on job-c4zkm. job-4j3vu measured the fix
   (micro_batch 4, 31.9 GB) only through `convert.qual`. Now the config has `"micro_batch": 4`, which the pipeline
   passes to both stages. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set as in `convert/qual.py`, and the
   dry run checks that the micro-batch reaches both stages.
2. **Stop floors re-derived from measurement.** The old floors were 50% of a *predicted* 25%-MFU rate (6,150 and
   1,450 tok/s). They are now 50% of the *measured* rates: stage 1 9,476 and stage 2 2,730.
3. **The teacher runs once on the gate windows.** The importance ranking (one `evaluate` per attention layer), the
   gate baseline and the gate each re-ran the frozen teacher's forward. `align.teacher_hidden` now caches the
   teacher's final hidden states on the CPU, so the GPU peak is unchanged. The result is identical (tested): 10
   teacher passes become 1.
4. **The 9B's own readout per-call cost is measured.** `probes.throughput.call_latency` runs in the pipeline's
   throughput phase on the `ConvertedStateOps` path that both Index arms use. The Index preflight now uses that
   number instead of the RWKV7-fitted 0.098 s/call. The preflight is conservative on both inputs:
   - prefill rate = the **minimum** measured over the throughput lengths (22,576 tok/s at 32k on job-4j3vu, against
     25,803 at 4k);
   - per-call cost = **max(small_call_s, pack_call_s)**, where `pack_call_s` also contains its own 4,096 positions.

   With the fitted 0.098 s the projection is 11.95 min against the 38-min line (95% of the 40-min cap): margin
   2.18x. At a plausible 9B `pack_call_s` of ~0.26 s it is 24.1 min (margin 58%). It breaks even at 0.445 s/call.
5. **Gate windows 16 -> 32** (131k tokens of wikitext-103 validation). The per-32k-token batch KL in job-4j3vu's
   stage 2 ranged from 0.045 to 0.085, so 65k tokens gives a noisy gate KL. The cost is about
   1 min.
6. **Phase timings are written to `results.json: timing`**, so the next spec is priced from receipts, not from
   estimates.

## What runs (`jobs/probe9b/probe_job.sh`) and the projected rate of each phase

The projections come from `python -m jobs.probe9b.preflight` (also run at setup: exit 5 if any margin breaks).
Margin is cap / projection - 1. The rule is >= 30% everywhere (`margin_min_frac` 0.3).

| Phase | Cap (min) | Projected (min) | Margin | Content | On failure |
|---|---|---|---|---|---|
| setup | 10 | 2.7 | 270% | context manifest, `nvidia-smi`, pins + kit import, fla #640 guard, **measured-rate preflight**, DEV rows sha256, the pinned 2,008-row DEV subset materialised | 5 / 10 |
| weights | 20 | 0.5 | >40x | `sf_fetch --stage probe9b` (wikitext, MMLU, 9B @ pinned revisions; hf_cache hit measured at 28 s; a miss is 19.3 GB) | 8 |
| P | 140 (deadline 128) | 82.8 (80.5 before the converted probes) | 69% (deadline 59%) | `convert.pipeline`: load 0.3, tokenise 40M wikitext-103 train tokens 2.8 (upper bound scaled from 20M), stock probes 1.6, **stage 1** 13.1 (400 x 32,768 tokens at 18,952 tok/s + 92 s), KL importance + gate on the cached teacher 2.8, **stage 2** 60.0 (wall-clock cap = 19.7M tokens at the measured 5,461 tok/s, more on a later layer), enforced gate, converted probes 1.6, prefill throughput + call latency 0.8 | 2 / 3 / 4 / 7 |
| I_stock | 40 | 12.3 | 209% (vs the 95% guard line) | Decision Index DEV subset, stock arm, prefix-fork readout: 6.23M positions / 22,576 tok/s (min measured) + 4,501 calls x call_s (0.098 until the pipeline's measurement replaces it) + model load. Preflight exit 15, live row-rate guard exit 15 | 12 / 7 / 15 |
| I_conv | 40 | 12.3 | 209% | same, converted arm (`converted=<layer>`, `gdn_ckpt=gdn_converted.pt`) | 12 / 7 / 15 |
| S | 10 | 1 | 900% | `jobs.common.dev_subset compare` (paired source-case cluster bootstrap, B=2000), `score`, `ece` per arm | 13 |
| D | - | - | - | `scripts/decide_probe9b.py`; `finalize.py` re-validates; `PROBE_DONE` only on exit 0 | 9 |
| reserve | 10 | | | | |

10 + 20 + 140 + 40 + 40 + 10 + reserve 10 = 270 = `max_duration_minutes`.

**Stop rules** (in code, `align.Guard`):
- exit 3 if the steady tok/s after 50 steps is below the floor: stage 1 9,476 (measured 18,952), stage 2 2,730
  (measured 5,461 on the slowest layer). Both floors leave a 50% margin;
- exit 4 if peak > 76 GB;
- `vram_fraction` 0.95, so an OOM fails fast.

A rejected layer is **not** a failure. The job exits 0 with the verdict `quality-fail` and skips the Index arms.

Measured on converted vs stock (`results.json`, `index/`):
- held-out perplexity: wikitext-2 test, ctx 1024 (64 windows) and 4096 (16 windows);
- MMLU-style MC subset: 285 questions, 0-shot letter logit;
- MQAR: 8/32/64/128 pairs;
- S-NIAH: 1k-16k, 24 needles per length;
- Decision Index DEV subset: 2,008 rows, 1,899 groups, 38 benchmarks; `jobs/common/dev_subset_m1.json`, ids sha256
  `6e4cb8df...95c3`, manifest hashes checked on every load.

**No TEST rows, no D1-B.** Scoring loads `--slice dev` only.

## Bottleneck analysis

| Phase | Share of the run (~133 min expected) | Dominant cost | Lever | Decision |
|---|---|---|---|---|
| **Stage 2** | **60 min, ~45%** | Per step: shared no-grad prefix, no-grad teacher suffix (~1/4 of step FLOPs), then the student suffix forward, its checkpoint re-forward and the activation backward. 61.8 GFLOP/token on layer 3, 34% MFU measured | (a) **Cache teacher outputs.** Rejected: one pass over fresh random windows uses each teacher output exactly once, so caching moves the FLOPs rather than removing them. Reusing stage 1's teacher pass would need ~107 GB of bf16 final hidden states for 13M tokens, or a top-k KL that changes the objective. (b) **Drop the checkpoint re-forward** (~25% of FLOPs). Not applied: unmeasured memory on the 9B, and an OOM wastes the whole spend. (c) **Batching.** Already 4 x 4,096 micro-batches x 2 accumulation = 32,768 tokens/step, measured | The stage is **wall-clock capped**, so a faster step buys distillation tokens, not dollars. job-4j3vu's trace shows the per-batch KL flat over 11.5M tokens (0.051 at step 0 -> 0.045 at step 349, range 0.045-0.085; gate 0.054), so more tokens are unlikely to change the verdict. The cap stays at the pre-registered 60 min |
| Stage 1 | 13 min, ~10% | Teacher forward to layer 31 + 8 GDN graphs (17.1 GFLOP/token, 33% MFU measured) + a 92 s first-step autotune | Fewer steps would degrade the KL-importance ranking, which needs all 8 layers aligned | Kept: 400 steps |
| Index arms | 2 x 12.3 min, ~18% | Per-call overhead (~7 of the ~12 min, borrowed from RWKV7) + 6.23M positions of prefill | The prefix-fork readout is already applied (239M -> 6.2M positions per arm, PERF_AUDIT). The 9B per-call cost is **now measured** before the arms run | Applied (measurement) |
| Importance + gate evals | 2.8 min | 8 + 2 `evaluate` calls over 131k tokens (forward + fp32 log-softmax over the 248k vocab) | **Teacher hidden cached once** (10 teacher passes -> 1) | Applied |
| Data / load / probes / throughput | ~7 min | CPU tokenisation of 40M tokens; B=1 probe loops (~500 calls) | Minor. Note: job-4j3vu's 9-min gap before stage 1 was the background fetch of an unrelated model (mamba3, 351 s), **not** tokenisation. The probe's fetch stage has no such model | Not applied (low leverage) |
| Build / setup / weights | 15 + 3 + 1 min | Image build (measured ~14 min to first runner line) | Same base digest as qual | - |

Biggest risk to the spend, now fixed: item 1 above (the OOM shape).

## Decision rule: BINDING = the 511e8dd pre-registration, unchanged (`configs/probe9b_h100.json: success_rule`)

The verdict is computed by `scripts/decide_probe9b.py` from `success_rule` only. That rule was committed at 511e8dd,
before any 9B data existed, and this spec does not change it (`tests/test_probe9b.py` checks it against
`git show 511e8dd:configs/probe9b_h100.json`):

| Check (converted 1 GDN + 7 attention minus stock, same code path, same items) | Proceed requires |
|---|---|
| Gate (enforced: step dNLL <= 0.05, cumulative <= 0.15, final-hidden cosine >= 0.95) | layer accepted |
| Perplexity ratio, wikitext-2 test, ctx 1024 and 4096 | <= 1.02 |
| MMLU subset drop | <= 3.0 points |
| Index DEV-subset paired diff (0-100) | point >= -1.0 and 95% CI lower bound >= -2.5 |
| MQAR mean accuracy drop | <= 0.10 |
| S-NIAH | mean drop <= 0.10 and every length's drop <= 0.25 |

**The enforced gate is 511e8dd's except `n_windows`.** `n_windows` went from 16 to 32. This change was made after
the job-4j3vu data. It is non-directional: it tightens the estimate and neither loosens nor tightens any threshold
(`tests/test_probe9b.py` checks every other gate field against 511e8dd).

**MMLU is noise-limited near its bound.** The subset has 285 items. At ~70% accuracy, one arm's SE is ~2.7 points,
and the paired difference's SE is roughly 1.5-2.7 points, depending on how correlated the two arms are. So an MMLU
fail within ~2 SE of the 3.0-point threshold (a drop of up to ~6-8 points) is reported in the decision write-up as
**noise-limited**. This does not change the binding verdict: the rule is applied as pre-registered.

| Verdict | When | Next |
|---|---|---|
| `proceed-all-8` (= convert-all-8) | every row passes | the ~$150 full-8 spend issue |
| `retrieval-fail` | gate, ppl, MMLU and Index pass; MQAR or S-NIAH fails | pre-registered fallback (PLAN): no full-8 spend, the cost-of-attention-freedom curve, RWKV7; ship-hybrid is decision-needed |
| `quality-fail` (= fold to RWKV7-only) | gate rejected the layer, or ppl / MMLU / Index fails | rwkv7 stage B |

Missing numbers give `no-measurement`, which is exit 9 and never a verdict. The rule is identical for the lean config.

### Post-hoc diagnostics (reported, NOT binding): `decision.json: diagnostic_bands_posthoc`

These bands were **written after seeing job-4j3vu's qualification gate numbers** for 9B layer 3 (8 windows, after
35 min of stage 2): dNLL -0.0005, **KL 0.054**, cos 0.980. They are therefore **not used for the verdict**. The note
`"written after seeing job-4j3vu qual gate numbers; not used for the verdict"` is written into every `decision.json`.
They never change the verdict and never authorise spend. `decide_probe9b.py` computes the verdict first and wraps
the diagnostics in try/except: if they fail, the error is recorded under `diagnostic_bands_posthoc.error`, and the
verdict and exit code are unchanged (tested).

| Diagnostic number | GREEN | AMBER | RED |
|---|---|---|---|
| Gate dNLL vs teacher | <= 0.019 | <= 0.05 | > 0.05 |
| Gate KL(teacher \|\| student) | <= 0.05 | <= 0.15 | > 0.15 |
| Gate final-hidden cosine | >= 0.97 | >= 0.95 | < 0.95 |
| ppl ratio | <= 1.02 | <= 1.05 | > 1.05 |
| MMLU drop | <= 3.0 | <= 6.0 | > 6.0 |
| Index point / CI low | >= -1.0 / >= -2.5 | >= -2.5 / >= -5.0 | below |

Where the diagnostic edges come from:
- **dNLL GREEN 0.019** = the 8-layer cumulative gate budget, 0.15 / 8.
- **KL and 1 - cos GREEN** = 3x the 0.8B pilot single-layer values: KL 0.017 and cos 0.991. **The pilot cos 0.991 is
  unverified.** It is coordinator-reported in HANDOFF.md and appears in neither LEDGER nor issue #3. The LEDGER has
  only dNLL -0.0036 and KL 0.017.
- **AMBER edges** = the enforced gate, or ~3x the GREEN edge.

The qualification's layer-3 numbers would land GREEN / AMBER (KL) / GREEN.

The diagnostic also reports the measured stage-2 KL gain:

> gain = (importance KL of the layer after stage 1 - gate KL after stage 2) / (importance KL after stage 1)

This is a reported number only. There is no re-probe path: `quality-fail` goes to rwkv7 stage B
(`jobs/DECISION_TREE.md`).

## `submit_job` fields (do not submit until the spend issue is approved)

Context:
1. `bash jobs/probe9b/build_context.sh <m1_sha> <dir>`. This wraps `jobs/common/build_context.sh probe9b`: code from
   `git archive` of a SHA merged into `agent-main`, the PRIVATE DEV rows, `CONTEXT_MANIFEST.json` (now including
   `preflight.py`, `decide_probe9b.py`, `convert/pipeline.py` and `convert/align.py`) and a reproducible `tar.zst`.
2. `create_context`, upload, `finalize_context`.
3. `open_mission("m1-probe9b-one-layer")`.

```json
{"chip": "h100", "chip_count": 1, "mission": "m1-probe9b-one-layer",
 "idempotency_key": "m1-probe9b-full-<context_sha256[:12]>",
 "label": "probe9b: one-layer Q9-AF conversion probe (full, job-4j3vu rates)",
 "context_id": "<from create_context>", "dockerfile_path": "Dockerfile", "build_timeout_minutes": 30,
 "command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "full"},
 "max_duration_minutes": 270, "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
 "hf_cache": true, "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a", "clock_lock": true,
 "checks": {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/PROBE_DONE"},
                                             {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
            "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                       "interval_s": 600, "grace_period_s": 1800, "fail_threshold": 2, "on_fail": "kill"}}}
```

Lean config (`python -m jobs.common.launch probe9b --mode lean`): the same block with `env {"PROBE_CONFIG": "lean"}`,
`idempotency_key "m1-probe9b-lean-<context_sha256[:12]>"` and label `probe9b: one-layer Q9-AF conversion probe (lean, job-4j3vu rates)`.

`depends_on` on the qual job is not used: the probe needs a human-approved spend issue after the qual receipt anyway.

## Health check (`checks.health`) and heartbeat

- `type: heartbeat`, `path: /output/heartbeat`. `jobs/clef_baseline/watchdog.sh` rewrites the file every 60 s
  **only while** the main process is alive **and** some file under `$OUT`, `$CK`, `$HF_DIR` or
  `WATCH_EXTRA=$WORK` changed in the last `STALL_S` = 1,200 s.
- Stage 2 writes `train_log.jsonl` every 25 steps (~150 s at the measured rate). The Index arms write one results
  row per request.
- `interval_s: 600`: the platform evaluates the heartbeat every 10 min.
- `fail_threshold: 2`: two consecutive stale evaluations trigger `on_fail`; one late write is tolerated.
- `grace_period_s: 1800`: no evaluation during the first 30 min of an attempt (image start, weight fetch, model load).
- `on_fail: kill`: the attempt ends `self_terminated`, billed to that point.
- A hung worker is detected at most `STALL_S` + 2 x `interval_s` = 40 min after its last file write. A dead main
  process stops the heartbeat at once.
- `tests/test_m1_next_jobs.py` checks statically that every path the runner writes is under a watched dir.

## Cost ceiling (`python -m jobs.common.costs`; build billed, >= 1 restart assumed)

| Item | Minutes | $ at $0.0495/min |
|---|---|---|
| Build (`build_timeout_minutes`, counted billable) | 30 | 1.49 |
| One attempt (`max_duration_minutes`; `validate_job` should report this) | 270 | $13.37 |
| **Worst case, build + 2 attempts** | 570 | **$28.21** (<= $40 hard cap) |
| Expected (build 15, setup 3, weights 1, P 86, I 13 + 13, S 2; each >= its measured-rate projection) | 133 | **$6.58** |

**Hard cap: `set_workspace_cap(MTD + 40)`**, then re-run `validate_job` and require `scope` = workspace cap and
`verdict: pass`. If rwkv7 stage A runs at the same time: MTD + `costs.concurrent_cap_usd(["probe9b",
"rwkv7_readout"])` = MTD + 45.

## Exit codes

0 measured + decided + exports verified | 1 finalize itself | 2 pipeline numerics/crash | 3 throughput stop |
4 memory stop | 5 setup (incl. measured-rate preflight) | 6 export verification | 7 phase timeout |
8 9B weights unavailable | 9 decision missing/invalid | 10 DEV rows or subset identity | 12 Index kit run |
13 subset scoring | 15 readout too slow by measurement

## Artifacts and export verification

- After every phase, `/output/phases/<phase>.tgz` (setup, P, I_stock, I_conv, S, final) is re-read and
  double-hashed into `/output/MANIFEST.jsonl`, and also POSTed as a sample.
- `/output/gdn_converted.pt` (+ `.sha256`): the converted GDN weights, the starting point of the full-8 job.
- `results.json: timing` (per-phase seconds), `stage1/progressive: mfu, flop_per_token, micro_batch`,
  `throughput.*.call_latency`. These are the receipts for the next spec.
- `$GMN_RESULT_PATH`: `{ok, exit_code, runner_rc, gates, exports_verified, decision, verdict}`.
- Re-check locally on CPU: `dev_subset compare` and `scripts/decide_probe9b.py` must reproduce
  `compare_conv_vs_stock.json` and `decision.json`. The other agent re-runs these independently before any "proceed"
  claim.

## Tests (CPU)

`tests/test_probe9b.py` checks:
- the cost line, the schedule, the measured block, the >= 30% margin for every phase, floor and the deadline (and
  that the preflight refuses tight caps);
- that the verdict comes from the 511e8dd rule only (unchanged by removing or perturbing the post-hoc diagnostic
  bands), every verdict path, the diagnostics' labelling, and the contamination boundary;
- the runner's preflight, exports and exit codes;
- **a CPU dry run of the real `convert.pipeline` on a tiny random Qwen3.5**, with the real probe config shrunk only in
  size: micro-batch reaches both stages, KL importance, one layer, enforced gate accept and reject/rollback, a single
  teacher pass, timings, call latency, resume, and the decision on the result.

`tests/test_m1_next_jobs.py` keeps the decision-tree, finalize and heartbeat checks.

## Known limitations (stated, not hidden)

- Distillation data is wikitext-103 train only (open, CC BY-SA). PLAN's recall levers belong to the full-8 job.
  A `retrieval-fail` here is therefore a fail without the levers.
- The probe suite's time, the gate-eval time and the 9B per-call cost are estimates in the projection. The first two
  have >= 69% P-margin; the third is measured in-run before it is used.
- The 0.8B pilot cosine (0.991) is unverified: coordinator-reported (HANDOFF.md), not in LEDGER or issue #3. Only the
  post-hoc diagnostic cosine edge depends on it; the binding rule does not.
- The subset Index is a 2,008-row paired go/no-go, not a board number. NIAH has 24 needles per length and MMLU has
  285 questions.

## Re-pricing 2026-10-09: H100 batch $0.0495 -> $0.063/min

SF Compute raised the H100 list price $3.96 -> $5.04/h on 2026-10-08; jobs now bill $0.063 per GPU-minute (Mon-Thu and weekends; Fridays $0.05628) and the preflight quotes max_duration x $0.063. Evidence and history: `jobs/common/rates.py`. The dollar figures above this section are at the old $0.0495/min and stay as written (receipts and past decisions used them); the numbers below are the current ones (`python -m jobs.common.costs`). Where a hard cap was raised, it was raised by the same ratio as the rate (x 0.063/0.0495 = x 14/11, rounded up to the cent), so the cap still buys the same minutes.

| Params | Worst case (build + 2 attempts) | Hard cap | Expected | One attempt (quoted max_cost) |
|---|---|---|---|---|
| `job_params.json` | $35.91 (was $28.21) | $40.00 (unchanged) | $8.38 (was $6.58) | $17.01 (was $13.37) |

Workspace cap for probe9b + rwkv7 stage A at $0.063/min: `costs.concurrent_cap_usd(["probe9b", "rwkv7_readout"])` = MTD + 57 (was MTD + 45).
