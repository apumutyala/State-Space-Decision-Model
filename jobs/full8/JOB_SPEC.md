# full8: all 8 attention layers of Qwen3.5-9B -> Gated DeltaNet (Q9-AF). Spec only, NOT submitted

**Trigger.** Issue #87. Job-5cmat (one-layer probe, issue #68) returned `proceed-all-8` under the 511e8dd rule.
`jobs/DECISION_TREE.md` branch A maps that verdict to this job: convert all 8 layers in KL-importance order, with
accept/rollback per layer.

**Status of this file.** The decision rule (`jobs/full8/rule.json`, `scripts/decide_full8.py`) was pre-registered
in its own commit, **be9eed8**, before the runner, the pricing or any run. That commit also held this file's
"Decision rule" section. Nothing in the rule has changed since. `tests/test_full8.py` checks:
- `rule.json` and `decide_full8.py` against `git show be9eed8:...`;
- this file's Decision rule section, byte for byte.

Everything after that section was added afterwards: pricing, runner, export, resume, dry run. Spec only: nothing is
submitted.

## Decision rule (BINDING, pre-registered: `jobs/full8/rule.json`, applied by `scripts/decide_full8.py` only)

**Where every constant comes from.** Every constant comes from text that existed before job-5cmat ran
(2026-10-06T21:33Z). No threshold was tuned on 5cmat's numbers. There are two sources:
- the probe's 511e8dd `success_rule` and enforced gate (`configs/probe9b_h100.json` @ 511e8dd);
- the full-8 row of `HANDOFF.md` / `AGENT_HANDOFF.md` @ caa8f55 (2026-10-06 17:59Z): "per-layer gate ΔNLL ≤ 0.05,
  cumulative ≤ 0.15, hidden cos ≥ 0.95; on retrieval failure: no further spend, report the cost-of-attention-freedom
  curve, `decision-needed` on shipping a hybrid".

**Per-layer gate.** This is the 511e8dd gate, unchanged. It is enforced inside the run by `convert.progressive`, in
KL-importance order. Each layer passes or fails:
- step ΔNLL vs the last accepted model <= 0.05;
- cumulative ΔNLL vs the teacher <= 0.15;
- final-hidden cosine >= 0.95;
- 32 gate windows, as shipped in PR #73 and run in 5cmat.

A failing layer is rolled back to attention. Its GDN is reset, the earlier GDNs return to their pre-step weights, and
the loop goes on to the next layer. No layer is retried.

**Gate audit.** Before any verdict, `decide_full8` re-derives every step from its recorded numbers. It checks:
- the 8 steps are the 8 attention layers, in ascending single-swap-KL order;
- pass/fail re-computes from the constants above;
- `accepted == passed`, since the policy is enforce;
- each baseline is the last accepted nll, or the teacher's if none was accepted;
- the converted list equals the accepted layers;
- the final probe ran on exactly that list.

Any mismatch gives `no-measurement`.

**Final model vs stock.** Same code paths as the probe, same items. The 511e8dd bounds apply, with one rescaled:

| Check | Bound | Source |
|---|---|---|
| ppl ratio, wikitext-2 test, ctx 1024 and 4096 | <= **1.1618** | Rescaled. See below |
| MMLU subset drop | <= 3.0 points | 511e8dd, unchanged |
| Index DEV-subset paired diff (2,008 rows, ids sha256 `6e4cb8df...`) | point >= -1.0 and 95% CI low >= -2.5 | 511e8dd, unchanged |
| MQAR mean accuracy drop | <= 0.10 | 511e8dd, unchanged |
| S-NIAH | mean drop <= 0.10 and every length's drop <= 0.25 | 511e8dd, unchanged |

**Why only ppl is rescaled.** The ppl bound is exp(0.15) = 1.16183, rounded down to 1.1618. Of these checks, only
ppl has an 8-layer budget in the plan text: the gate's cumulative ΔNLL <= 0.15. That budget bounds the same
quantity the held-out ppl ratio measures, NLL vs the teacher on wikitext. The 511e8dd one-layer bound of 1.02 is the
per-layer share of that budget: ln 1.02 = 0.0198 ≈ 0.15 / 8. Keeping 1.02 for 8 layers would make the plan's own
cumulative budget unreachable. MMLU, Index, MQAR and S-NIAH have no 8-layer scaling anywhere in the plan, so their
one-layer bounds apply unchanged. That makes them stricter for 8 layers than for one.

**Verdicts.** They are evaluated in this order. N is the number of accepted layers.

| Verdict | When | Next step |
|---|---|---|
| `stopped` / `no-measurement` | Any of: a pipeline stop (exit 3/4); fewer than 8 layers attempted (deadline or crash); a gate-audit mismatch; a missing number | Exit 9. Never a verdict and authorises nothing. The checkpoint and the exports keep every finished layer |
| `no-conversion` | N = 0 | Fold: the m2 backbone is RWKV7. The Index arms are skipped |
| `partial-conversion` | 1 <= N <= 7 | See "When the gate rejects layers" below |
| `quality-fail` | N = 8, and ppl, MMLU or Index is past its bound | Fold: the m2 backbone is RWKV7 (the DECISION_TREE quality-fail mapping). The Q9-AF weights are archived, not pinned |
| `retrieval-fail` | N = 8, quality passes, and MQAR or S-NIAH is past its bound | See below |
| `q9af-backbone` | N = 8 and every bound passes | **Q9-AF becomes the m2 decision-cell backbone.** Pin `gdn_converted.pt`'s sha256 as `q9af.converted_sha256` in `jobs/m2_pilot/m2_pilot_cell_q9af.json`, in its own reviewed PR. Then the m2 parent arm and pilot cell run with `PARENT=q9af` |

**What happens on `retrieval-fail`.** This is the PLAN's pre-registered fallback:
- attention is **not** added back;
- the cost-of-attention-freedom curve is reported, with 8, 7, ... 0 attention layers taken from this job's
  per-layer probes;
- Q9-AF stays the pure-recurrent entrant, as PLAN says;
- the m2 backbone (Q9-AF anyway, RWKV7, or a hybrid) is `decision-needed` for the user;
- there is no further conversion spend.

**When the gate rejects layers.** A model with 1-7 converted layers is a hybrid with 8 - N attention layers. It is
**not attention-free**, so it can never be the Q9-AF backbone: `heads.backbone.q9af_from_stock` refuses any
conversion that does not cover exactly the 8 attention layers. Such a model:
- is reported as `Q9-H<8-N>`, with the converted and kept layer lists;
- gets the same ppl / MMLU / Index / MQAR / S-NIAH numbers and the curve, non-binding for Q9-AF;
- has its weights exported as `gdn_partial.pt`, never as `gdn_converted.pt`.

The next step is the same whatever those numbers say:
- the m2 backbone defaults to RWKV7, which is attention-free;
- shipping the hybrid is `decision-needed` for the user (PLAN: "we do not quietly add attention");
- a rejected layer is not re-converted without a new spend issue and a new pre-registered rule.

If the cumulative budget runs out on a late layer, the result is `partial-conversion` too. The verdict does not
depend on why a layer was rejected.

**Reported, never binding** (`decision.json: report`, computed after the verdict):
- the cost-of-attention-freedom curve;
- per-layer gate numbers and stage-2 receipts (tokens, tok/s, MFU, peak);
- ECE per arm and prefill throughput / call latency, stock vs converted;
- an MMLU noise note. The subset has 285 items, so a fail within 2 SE of the 3.0-point bound is labelled
  noise-limited. The rule is still applied as written.

Job-5cmat's single-layer numbers (dNLL 0.0068, ppl ratio 1.007, Index +1.66) are context only. No bound depends on
them.

### Amendment A1 (2026-10-07, after independent review, before any run)

The independent review of PR #94 returned FIX-THEN-LAUNCH. The coordinator's decisions, quoted:

> Coordinator decisions for Amendment A1, made before any run.
> - **B1:** every configured niah_lengths value must be numeric in BOTH arms, and the same for every other number the
>   verdict reads. Missing or None means a "missing" decision and exit 9. Add `niah_length_drop` to REQUIRED and
>   valid_decision. Add tests for None, an empty dict, and a length missing from the stock arm.
> - **B2:** in full8, the final probe must never be deadline-gated, or on resume the probe must re-run for the last
>   accepted step if it has none. Do whichever is cleaner and test it: deadline passes after layer 8, kill, resume,
>   probe runs, verdict.
> - **B3:** add `"decision_rule": "scripts/decide_full8.py"` to job_params.json.
> - **MMLU noise:** don't loosen the bound. Remove the noise: score MMLU on the FULL MMLU test set (~14k items, both
>   arms, same code path), with a paired per-item bootstrap. The bound stays a point drop ≤ 3.0. Also report the
>   paired 95% CI.
> - **Honesty:** the JOB_SPEC must state that be9eed8 was committed about 25 min after 5cmat finished, with its
>   numbers known. The constants trace to pre-5cmat text, but the exp(0.15) ppl derivation was written knowing
>   5cmat's ppl.

**What A1 changes in the binding rule.** These changes touch only `rule.json`, `decide_full8.py`, its tests and this
section. B2, B3 and the MMLU measurement itself are code, in a separate commit.

1. **S-NIAH lengths (B1).** `rule.json: required_niah_lengths` is [1024, 2048, 4096, 8192, 16384]. That equals the
   recipe's `probes.niah_lengths`, unchanged.
   - Each `L<n>` must be numeric in both the stock and the final arm.
   - `niah_length_drop` is in `REQUIRED`, and `valid_decision` refuses a None or an empty per-length dict.
   - A hole gives `no-measurement` (exit 9). Before A1 it could become a `retrieval-fail` verdict.
   - Every other number the verdict reads was already REQUIRED from both arms.
2. **MMLU on the full test set.** The binding MMLU drop is now measured on the **full MMLU test set**:
   - cais/mmlu `all/test` @ c30699e8, **14,042 items**, instead of the 285-item subset;
   - both arms run in the same job on the same code path, as `results.json: mmlu_full`;
   - each arm reports per-item 0/1 correctness, the same `items_sha256`, and stock vs exactly-the-converted modes;
     anything less is missing.

   **The bound is unchanged:** point drop <= 3.0 points.
   - The paired per-item bootstrap 95% CI (B = 10,000, seed 0) is **reported, never binding**.
   - The 285-item subset stays in the per-layer curve as a report.
   - No bound was loosened. With 14,042 paired items the CI is about ±0.5 point, against about ±7 for the subset.
     A fail is therefore a measured drop, not noise.
3. **Disclosure (honesty).** be9eed8 (2026-10-06T23:40Z) was committed **about 25 min after job-5cmat finished**
   (21:33Z + 101 min), **with 5cmat's numbers known**.
   - Every constant is copied from pre-5cmat text: 511e8dd, and HANDOFF @ caa8f55.
   - But the exp(0.15) ppl derivation was written **knowing 5cmat's ppl** (ratio 1.007 on one layer).
   - The independent review accepted exp(0.15) on the merits:
     - the pre-5cmat full-8 text has no ppl bound, only the gate's cumulative 0.15;
     - exp(0.15) is stricter than 1.02^8 = 1.172;
     - with wikitext distillation, gate and evaluation, the bound is nearly redundant with the gate.
   - It also noted that this reading, rather than a strict 1.02, decides whether ppl can fail at all.

## Question and recipe

**Question.** Can all 8 attention layers of Qwen3.5-9B be replaced by Gated DeltaNets with the probe's recipe, giving
a fully attention-free 9B (Q9-AF) that keeps quality and retrieval within the pre-registered bounds above?

**Recipe.** `configs/full8_h100.json` is the probe's config, exactly as job-5cmat ran it, with **one** change:
`progressive.max_layers` goes from 1 to 8.

Everything else is the probe's, unchanged:
- seed 0, kernel fla;
- data: 40M wikitext-103 train tokens (open text only; no recall-lever mix, see the deviations below);
- stage 1: 400 steps of 8 x 4096 at lr 1e-3;
- KL importance (ascending single-swap KL after stage 1);
- per layer, stage 2: end-to-end KL with all converted GDNs training, lr 3e-4, `micro_batch` 4, a **60-minute
  wall-clock cap per layer**;
- the enforced gate on 32 windows;
- the probes: ppl, MMLU 285, MQAR, S-NIAH 1k-16k;
- the throughput lengths;
- the stop floors: stage 1 9,476 tok/s, stage 2 2,730 tok/s, 76 GB.

`tests/test_full8.py` checks that the config differs from the probe's only in `progressive`, `ckpt_dir` and the
removed rule blocks.

**No warm start: layer 23 is re-converted.** The pipeline has no clean way to warm-start layer 23 from 5cmat's
`gdn_converted.pt`:
- that file holds only layer 23's post-stage-2 weights. It does not hold the stage-1 weights of the other 7 layers
  (5cmat exported only the converted layer), nor layer 23's own stage-1 state, which the rollback needs;
- reusing it would mean fabricating a progressive record (gate numbers, baseline) from another process, against a
  stage 1 this job re-runs anyway;
- it would put new, untested code in the money path.

What it would save is one 61-minute layer (about $3). Re-converting also gives an independent re-measurement of the
layer-23 result: wins must reproduce. Stage 1 runs in either case.

The importance order is re-computed in-run. 5cmat measured 23, 31, 15, 11, 7, 19, 3, 27; GPU nondeterminism may
change it. The price does not depend on the order, because stage 2 is capped in minutes. The rule's gate audit
checks the recorded order.

## Measured basis (job-5cmat, art-2bybh, sha256 `dd491ade...f409`; plus job-4j3vu for layer 3)

All numbers are in `job_params.json: measured`. They come from `out/pipeline/results.json` (timing, stage1,
progressive.steps[0], throughput), `logs/run.log` and `out/index/rate_*.json`.

| Quantity | Measured | Used for |
|---|---|---|
| P phase, one layer | 79.2 min | Model check: the projection reproduces it within 3 min (tested) |
| load + tokenise 40M tokens | 49.5 s | Pre-progressive |
| stock probes | 109.0 s | Pre-progressive |
| stage 1 | 818.9 s (18,297 tok/s, MFU 0.316) | Pre-progressive; stage-1 floor |
| teacher cache + importance (8 evaluates) | 5.7 s + 61.4 s | Pre-progressive |
| progressive step (layer 23) | 3,677.6 s = stage 2 3,602.7 s + converted probes 57.4 s + evaluate/save 17.5 s | Per-layer cost |
| stage 2, layer 23 | 9,460 tok/s, MFU 0.343, 34.1M tokens, peak 31.4 GB | Rates, memory |
| stage 2, layer 3 (job-4j3vu) | 5,461 tok/s, MFU 0.341 | Slowest rate (longest suffix) |
| throughput phase | 33.6 s (the converted arm's call latency failed with a dtype error: fixed, see below) | Throughput |
| Index arms | stock 10.0 min, converted 9.0 min; in-run volume preflight projected 22.5 min (min prefill 22,144 tok/s, pack call 0.237 s) | I caps |
| setup / weights from hf_cache | 2.4 min / 20 s | Setup |

**Per-layer stage-2 rates.** These come from `convert.memmodel.flops_stage2` at the lower measured MFU (0.341). The
FLOP model reproduces both measured layers within 0.002 MFU. A later step trains more GDNs over a longer student
suffix, so it is slower:

| Step (5cmat order) | 23 | 31 | 15 | 11 | 7 | 19 | 3 | 27 |
|---|---|---|---|---|---|---|---|---|
| First student layer | 23 | 23 | 15 | 11 | 7 | 7 | 3 | 3 |
| GFLOP/token | 35.8 | 36.0 | 46.6 | 52.0 | 57.3 | 57.5 | 62.9 | 63.1 |
| Predicted tok/s | 9,411 | 9,362 | 7,239 | 6,490 | 5,881 | 5,862 | 5,361 | 5,345 |
| Tokens in 60 min | 33.9M | 33.7M | 26.1M | 23.4M | 21.2M | 21.1M | 19.3M | 19.2M |

That is 198M stage-2 tokens in total. The slowest step's predicted peak is 38.7 GB, against the 76 GB stop. The
stage-2 floor of 2,730 tok/s is 0.51 x the slowest predicted rate.

## What runs (`jobs/full8/full8_job.sh`) and the projected minutes of each phase

The projections come from `python -m jobs.full8.preflight`, which also runs at setup and exits 5 if any rule breaks.
The rule: every cap >= 1.3 x its projection. For the Index arms the usable cap is 95% (the rate guard's stop line).

| Phase | Cap | Projected | Margin | Content | On failure |
|---|---|---|---|---|---|
| setup | 10 | 2.4 | 317% | manifest, nvidia-smi, pins, fla #640 guard, **measured-rate preflight**, DEV rows sha256, the 2,008-row subset | 5 / 10 |
| weights | 20 | 0.3 | >40x | `sf_fetch --stage probe9b` (wikitext, MMLU, 9B @ pinned revisions) | 8 |
| P | 675 (deadline 585) | **514.8** | 31.1% | `convert.pipeline` into `$GMN_CHECKPOINT_DIR` (see below), then the full-MMLU arms (A1) | 2 / 3 / 4 / 7 |
| X | 15 | 3.0 (estimate) | 400% | `jobs.full8.export_gdn --mode final`: the accepted GDNs, verified | 6 |
| I_stock | 40 | 22.5 (measured 10.0) | 69% | Decision Index DEV subset, stock arm, prefix-fork readout; preflight and live row-rate guard | 12 / 7 / 15 |
| I_conv | 40 | 22.5 (measured 9.0) | 69% | same, converted arm, loading the exported file | 12 / 7 / 15 |
| S | 10 | 1.0 | 900% | `dev_subset compare` (paired cluster bootstrap), `score`, `ece` per arm | 13 |
| D | - | - | - | `scripts/decide_full8.py` (the pre-registered rule); `finalize.py` re-validates; `FULL8_DONE` only on exit 0 | 9 |
| reserve | 15 | | | | |

10 + 20 + 675 + 15 + 40 + 40 + 10 + reserve 15 = 825 = `max_duration_minutes`.

**What P is made of.**
- Pre-progressive work: 17.4 min. This is load + data 0.8, stock probes 1.8, stage 1 13.6, teacher cache + importance
  1.1.
- 8 layers x 61.35 min each:
  - the stage-2 cap: 60;
  - one step of overshoot at the slowest rate: 0.10;
  - evaluate + checkpoint save: 0.29;
  - converted probes: 0.96.
- Throughput, both arms with call latency: 1.1 min.
- Full MMLU, stock + final (A1): 5.5 min. Per arm that is 2.49M padded positions / 22,144 tok/s, plus 156 calls x
  0.237 s, plus 0.25 min of tokenisation. The volume was computed exactly with the pinned tokenizer.

**Deadline.** The pipeline starts no new layer after 585 min. Two conditions fix that number:
- in a healthy run the 8th layer starts at 446.9 min, and 1.3 x 446.9 = 581;
- a layer started just before the deadline still fits the P cap, including the post-progressive work (throughput
  and full MMLU, never deadline-gated): 585 + 1.3 x (61.35 + 6.6) = 673.3 <= 675.

If a layer is skipped at the deadline, the result is `no-measurement` (exit 9), never a verdict. The checkpoint keeps
every finished layer.

**What else is measured.** Everything below compares converted vs stock (`results.json`, `index/`), on the same code
paths as the probe:
- ppl on wikitext-2 test at ctx 1024 and 4096;
- MMLU subset (285);
- MQAR (8/32/64/128 pairs);
- S-NIAH (1k-16k, 24 needles per length);
- the Index DEV subset: 2,008 rows, ids sha256 `6e4cb8df...95c3`, manifest hashes checked;
- ECE per arm;
- prefill tok/s at 4k-32k and the readout call latency, both arms.

The probes also run after **every** accepted layer. That gives the cost-of-attention-freedom curve, 8 -> 0 attention
layers, as PLAN requires. **No TEST rows, no D1-B**: scoring loads `--slice dev` only.

## Checkpoint / resume (`resume: "checkpoint"`, `max_restarts` 1)

**What lives in the checkpoint dir.** The pipeline's `--out` (results.json, train_log) and `ckpt_dir`
(`${FULL8_CK}/gdn`: `stage1.pt`, `prog_<n>.pt`) are both under `$GMN_CHECKPOINT_DIR`.

**Write order and checkpoint declaration.**
1. After each step `convert.progressive` writes `prog_<n>.pt`, the accepted weights (a rejected layer is already
   reset).
2. Then it writes results.json.
3. Then `PHASE_HOOK` (`jobs/full8/phase_hook.sh`) POSTs `/v1/checkpoint` and logs to `/output/checkpoints.jsonl`.

The hook also fires after stage 1, importance and throughput. It never fails the run. The platform allows one
capture per 300 s, so the importance capture about a minute after stage 1 may return 429. That is harmless: importance
is recomputed in about 1 min.

**A restarted attempt** re-runs the runner from the top: setup, then the same pipeline command. `convert.pipeline`
finds results.json and skips:
- the stock probes and stage 1 (it loads `stage1.pt`);
- importance;
- every finished layer (it loads `prog_<n>.pt`).

It continues at layer n + 1. The Index arms always re-run from a fresh directory (a stale one is moved aside, never
appended to).

**Partial-work protection.** A preemption or a late failure never loses converted layers:
- every finished layer is in the checkpoint slot;
- on any non-zero exit the runner also salvages the finished layers to `/output/gdn_salvage_step<n>.pt`, with
  sha256;
- after P, the weights are exported **before** the Index arms run, so an Index or scoring failure leaves
  `gdn_converted.pt` in `/output`.

**Checkpoint size.** `save` writes all 8 attached GDNs, fp32, 2.16 GB per file, so the slot grows to ~19.4 GB
(stage1 + prog_1..8). The platform documents no size limit. `/output` holds at most ~4.3 GB (final + one salvage),
under the 5 GiB artifact cap.

## Shell-level CPU dry run (the runner itself, executed)

`tests/test_full8.py::test_shell_dry_run_*` runs `bash jobs/full8/full8_job.sh` end to end with `FULL8_DRYRUN=1`.

**What is replaced.** Only the GPU / kit / private-data / weight-fetch steps (`tests/full8_dryrun/`). The tiny Qwen3.5
stand-in keeps the **real 9B layout**: 32 layers with attention at 3, 7, ..., 31, so the 8-layer rule applies
unchanged. The seam is refused when `/opt/ssdm/m1` exists (the image), and the submit fields set no `FULL8_*`
variable (tested).

**What runs for real:**
- the preflight;
- the subset row filter and the DEV sha256 check;
- `convert.pipeline`: stage 1, importance, 8 enforced-gate steps with rollback, the checkpoint dir, `PHASE_HOOK`
  (a shimmed `curl` records the POSTs), resume;
- `export_gdn`, including the strict HF load;
- the readout plugin that loads the exported file for the converted Index arm;
- the rate guards;
- `decide_full8` on the pipeline's own results.json;
- `finalize.py`, the phase exports and the manifest.

**Scenarios:**
- **A, uninterrupted.** Exit 0, `q9af-backbone`, `FULL8_DONE`. `gdn_converted.pt` holds 8 layers, and its sha256
  matches what the converted Index arm loaded. `heads.backbone.q9af_from_stock` (the m2 consumer) loads it strictly.
  There are 8 progressive checkpoint POSTs.
- **B, preemption after layer k = 3.** The process is hard-killed (`os._exit(137)`) right after step 3 is persisted.
  The runner exits 2, salvages the 3 layers and has made 3 progressive POSTs. The **restarted attempt** gets a fresh
  `/output` and work disk but the same checkpoint dir. It exits 0 and does no stage 1. Stage 2 runs for exactly
  importance order [3:] (layers 1-3 are not re-trained). The first 3 progressive records are identical, and the final
  weights are **bit-identical** to A's.
- **Partial: the gate rejects layers 3 and 27.** `partial-conversion`, `Q9-H2`, `gdn_partial.pt` (no
  `gdn_converted.pt`), which `q9af_from_stock` refuses.

**Bugs the dry run found** (none of them is visible to a Python-level test):
1. A CRLF-printing `python` leaked `\r` into the exported file name read by `read` (fixed: stripped).
2. On Windows, rate_guard's child `python` resolved to the venv's base interpreter (fixed in the test environment
   only).
3. The 60-s watchdog sleep outlived the script (dry run: 1 s heartbeat).

## Bottleneck analysis (`jobs/PERF_AUDIT.md` discipline)

| Phase | Share of the expected run (563 min) | Dominant cost | Decision |
|---|---|---|---|
| **Stage 2** | **481 min, 85%** (93% of P) | Wall-clock capped at 60 min per layer, by pre-registration, so $ do not depend on speed. Per step: no-grad shared prefix + teacher suffix (~1/4 of FLOPs), student suffix forward + checkpoint re-forward + activation backward. MFU 0.34 measured on two layers | **Recipe kept** (pre-registered). Faster steps would buy tokens, not dollars. See "waste" |
| Stage 1 | 13.6 min, 2.4% | Teacher to layer 31 + 8 GDN graphs, 0.32 MFU | Kept. It must re-run anyway (no clean warm start) |
| Probes after each layer | 8 x 1.0 min | B=1 probe loops | Kept: they are the PLAN's cost-of-attention-freedom curve |
| Index arms | 19 min measured, 3.4% | Per-call overhead + 6.2M positions | Already prefix-forked (PERF_AUDIT). The stock arm repeats 5cmat's stock arm. Kept for same-job pairing, and it doubles as a reproduction check of that arm |
| Build / setup / weights / X / S | ~24 min | Image build (~14 min) | Same base digest as the probe |

**Waste, reported, not acted on.** The recipe is pre-registered.

1. **Stage 2 shows diminishing returns on layer 23 in 5cmat.**
   - The logged per-batch KL (single batches, so noisy) fell from 0.0155 (steps 0-150) to 0.0143 (150-500), then to
     0.0119 (500-800) and 0.0115 (800-1,040). That is a 25% drop by mid-run, then about 3% more over the last 260
     steps (~8.5M tokens).
   - The gate KL moved 0.0157 -> 0.0113 (27.7%) end to end. In job-4j3vu (layer 3, 35 min) it was flat.
   - If later layers behave like layer 23, roughly the last third of each 60-min cap (~20 min/layer, ~160 min,
     ~$8) buys little measurable KL.
   - The counter-argument: later layers train more GDNs and get 43% fewer tokens (19M vs 34M), and the gate is
     cumulative, so their marginal tokens may matter more.
   - We do not shorten the cap. That would be tuning the recipe on 5cmat.
2. **Checkpoint writes.** Each step saves all 8 GDNs (2.16 GB; ~19.4 GB in total) where only the converted layers
   plus the stage-1 states are needed. The cost is seconds of I/O, not $.
3. **Stage-2 teacher suffix.** It is recomputed each step (~25% of stage-2 FLOPs). Caching it is impossible on fresh
   windows (PERF_AUDIT probe analysis).
4. **Checkpoint re-forward.** It is ~25% of FLOPs. Dropping it is not applied: memory is unmeasured, and an OOM
   wastes the whole spend.

**Fixed while pricing (receipt bug).** 5cmat's converted-arm `call_latency` failed with "mat1 and mat2 ...
BFloat16 != float": the small call ran outside autocast with fp32 student GDNs. Now `probes.throughput.call_latency`
runs under `no_grad` with the same autocast for both calls. The dry run asserts that the converted arm's
`small_call_s` is measured.

## Cost ceiling (`python -m jobs.common.costs`; build billed, >= 1 restart assumed)

| Item | Minutes | $ at $0.0495/min |
|---|---|---|
| Build (`build_timeout_minutes`, counted billable) | 60 | 2.97 |
| One attempt (`max_duration_minutes`; what `validate_job` should report) | 825 | $40.84 |
| **Worst case: build + 2 full attempts** | 1,710 | **$84.65** (<= $85 hard cap) |
| Expected (build 15, setup 3, weights 1, P 518, X 3, I 11 + 10, S 2) | 563 | **$27.87** |

**Measured pricing is far below the ~$150 budget.** DECISION_TREE branch A budgeted ~$150 from the PR #9 FLOP model.
The measured 5cmat phases put the expected cost at **$27.87**. Even the costs.py worst case, **$84.65**, is 56% of
the budget. That worst case is conservative: a resumed attempt does not repeat finished layers, so a real second
attempt costs only the remaining work. Branch A now leaves $120.80 of System 1 for m2. A1's full-MMLU arms add
5.5 projected minutes (+$0.30 expected, +$0.50 worst case).

**Hard cap: `set_workspace_cap(MTD + 108.19)`** (re-priced 2026-10-09 at $0.063/min; was MTD + 85). MTD: read it live (`autor usage --json`) or from the newest usage receipt `costs.SPENT_USD` loads (it was $16.11 on 2026-10-07). Then re-run `validate_job` and require
scope = workspace cap and `verdict: pass`.

## `submit_job` fields (do not submit until #87 is approved and independently reviewed)

1. Context: `bash jobs/full8/build_context.sh <m1_sha> <dir>` (a SHA merged into `agent-main`), then
   `create_context`, upload, `finalize_context`.
2. `open_mission("m1-full8-q9af")`.

```json
{"chip": "h100", "chip_count": 1, "mission": "m1-full8-q9af",
 "idempotency_key": "m1-full8-<context_sha256[:12]>",
 "label": "full8: all 8 attention layers -> GDN (Q9-AF), rule be9eed8, 5cmat-priced",
 "context_id": "<from create_context>", "dockerfile_path": "Dockerfile", "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/full8/full8_job.sh", "env": {},
 "max_duration_minutes": 825, "max_restarts": 1, "resume": "checkpoint", "queue_ttl_minutes": 1440,
 "hf_cache": true, "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a", "clock_lock": true,
 "checks": {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/FULL8_DONE"},
                                             {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
            "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"},
                       "interval_s": 600, "grace_period_s": 1800, "fail_threshold": 2, "on_fail": "kill"}}}
```

**Heartbeat.** It is the probe's progress-gated watchdog: `STALL_S` 1,200 s, watching `$OUT`, `$CK`, `$HF_DIR` and
`$WORK`. Stage 2 appends to `$CK/pipeline/train_log.jsonl` every 25 steps, which is <= 153 s at the slowest
predicted step. A hung worker is killed at most 40 min after its last write.

## Exit codes

0 measured + decided + exports verified | 1 finalize itself | 2 pipeline numerics/crash (incl. a killed attempt) |
3 throughput stop | 4 memory stop | 5 setup (incl. measured-rate preflight) | 6 export verification | 7 phase timeout |
8 9B weights unavailable | 9 decision missing/invalid (incl. no-measurement) | 10 DEV rows or subset identity |
12 Index kit run | 13 subset scoring | 15 readout too slow by measurement

## Artifacts

- `/output/gdn_converted.pt` + `.sha256`: `{layer: GDN state_dict}`, fp32, about 2.16 GB. It exists only when all 8
  layers are accepted. Its sha256 is the `q9af.converted_sha256` pin for m2. Otherwise the file is
  `gdn_partial.pt`.
- `export.json` (in the X phase tgz) records: sha256, bytes, layers, the source `prog_<n>.pt`, the results.json
  sha256, `base_config_sha256` and the strict HF load per layer.
- `/output/decision.json` and `phases/{setup,P,X,I_stock,I_conv,S,final}.tgz`, re-read and double-hashed into
  `MANIFEST.jsonl`.
- `/output/checkpoints.jsonl`.
- `$GMN_RESULT_PATH`.

**Before any claim.** The other agent re-runs `dev_subset compare` and `scripts/decide_full8.py` on the exported
files on CPU, and both must reproduce.

## Deviations from the pre-registered plan (stated, not hidden)

1. **Recall levers.** PLAN m1 and the DECISION_TREE row both say "recall levers". Only one is in effect:
   - **JRT** re-read is in effect in the Index readout (`jrt-chatml-v1`), but not in the MQAR / S-NIAH probes.
   - **Retrieval-targeted distillation tokens (20-30%)** are not applied: the data is open wikitext-103 only, as in
     the probe, per the job instructions.
   - **OPRM chunked state inference** is not applied.

   So a `retrieval-fail` here is a fail **without** the distillation-mix lever. PLAN's fallback is worded "fails the
   retrieval gate *after* the recall levers". The rule still maps `retrieval-fail` to the PLAN fallback with
   `decision-needed`; the user may decide that a levers run is worth its own spend issue.
2. **DECISION_TREE says "starting from `gdn_converted.pt`".** Layer 23 is re-converted instead. See "No warm start"
   above for why.
3. **Cost-of-attention-freedom curve.** It comes from the per-layer probes (8 -> 0 attention layers). PLAN's
   zero-training SWA+sinks control is **not** produced: the probe recipe has `swa_windows: []`, and this job keeps
   that recipe.
4. **PLAN m1 also lists a "steer baseline on all parents".** It is not part of this job.
5. **The ppl bound is rescaled to exp(0.15) = 1.1618.** The derivation is in the rule section. It is the only
   rescaled bound.
6. **The budget is far below plan.** ~$150 was planned; $84.65 is the worst case and $27.87 the expected cost.

## Pre-mortem (PR #91) and m2-consumer checks

**PR #91 PREMORTEM.md findings:**

1. **The probe runner was single-layer.** `probe_job.sh` and `max_layers: 1` were built for one layer, per-layer
   checkpoints were never exported, and the deadline was undersized. This job has its own runner, `max_layers: 8`, a
   deadline of 585 min and a P cap of 675 min (8.6 h projected, 1.3x margins). Every layer is checkpointed in
   `$GMN_CHECKPOINT_DIR` with a platform checkpoint POST, and a failed attempt gets a salvage export.
2. **Hidden-cos error compounds against the 0.95 gate, so late layers may be rejected.** The handling is
   pre-registered as `partial-conversion` (be9eed8):
   - the result is reported as `Q9-H<k>`, exported as `gdn_partial.pt`, and is never Q9-AF;
   - m2 defaults to RWKV7, and the hybrid is `decision-needed`.

   Every step's `hidden_cos` / KL / dNLL is in `decision.json: report.steps`.
3. **Retrieval may collapse; no recall levers are built.** MQAR and S-NIAH run after **every** accepted layer, so
   the curve shows where retrieval goes:
   - `decision.json: report.curve` has mqar and niah per point;
   - `results.json: probe_conv<k>_kept<8-k>` has the per-length S-NIAH.

   See deviation 1.
4. **Block checkpointing (P3) is not built here.** The export records the block boundaries (`export.json: blocks`):
   - `q9af_from_stock` keeps HF `Qwen3_5DecoderLayer` blocks at `model.language_model.layers[i]` and swaps only the
     mixer (`linear_attn`);
   - so a GDN block activation checkpoint can wrap those 32 modules without touching the checkpoint format.

**m2 consumer.** Clef-flash = frozen Qwen3.5-9B + routing head + rank-256 LoRA, and the m2 cell trains the same
shape on the converted backbone. Three checks, each tested in dry-run A:
- **(a) Load path.** The exported file loads through `heads.backbone.q9af_from_stock` and passes `q9af_pin_check`
  on its sha256. The heads' cache-fork features on the converted backbone equal recompute (1e-5).
- **(b) What `export.json` records:**
  - `layer_types_after` (GDN vs attention for all 32 layers), `gdn_layers`, `attention_layers_kept`;
  - `gdn_state_shapes_per_sequence`: recurrent S [32, 128, 128] and conv tail [8192, 3] on the 9B. On the stand-in
    they are checked against a real `GDN.step` state.
- **(c) Pinning.** `gdn_converted.pt.sha256` and `export.json: sha256` are the value for
  `q9af.converted_sha256`. `base_config_sha256` is recorded as well.

**Incompatibilities found:** none blocking. Notes:
- the weights are fp32 (as gated) and m2 loads them into bf16 by `load_state_dict` casting;
- `q9af_from_stock` refuses a partial conversion, by design;
- `load_q9af_backbone` needs `base_config_sha256` pinned, and that value equals `export.json: base_config_sha256`.

## Amendment A1: code changes (separate commit after the rule commit 2cb8cbc)

- **Full MMLU arms (the A1 measurement).**
  - `probes.core.mmlu_full` uses the same `_mmlu_prompt` and letter-logit scoring as the subset probe, over all 14,042
    items.
  - Items are batched by padded length (right padding to 128, at most 16,384 positions per forward), so a batched
    forward scores each item exactly as a one-item forward would. Tested on CPU: the per-item correctness is equal.
  - `convert.pipeline` runs it after the 8th layer, for stock and then the final model, never gated by the deadline,
    and writes it as `results.json: mmlu_full` (per-item 0/1, `items_sha256`, modes).
- **B2: the final probe.** There are two independent guards:
  - `progressive.probe_after_deadline: true` in the config: an accepted layer's probes always run, and the deadline
    only stops new layers;
  - a resume repair in `convert.progressive.run_progressive`: if the last accepted step has no probes, they run on
    resume, before anything else.

  Both are tested, the first with a fake ops object. The repair is also tested in the shell dry run: the deadline
  passes during layer 8 with the flag off, the probe is skipped, the attempt is killed after step 8, and the restarted
  attempt repairs the probe with nothing re-trained, giving `q9af-backbone` and `FULL8_DONE`. The late-start budget
  already covered a post-deadline probe.
- **B3.** `job_params.json` now has `"decision_rule": "scripts/decide_full8.py"`. PR #93's lint
  (`jobs/common/lint.py` from `claude/flywheel-tooling`, run in a scratch worktree, not in this PR) passes:
  `python -m jobs.common.lint full8` gives `ok`.
- **Non-blocking #2: the converted arm's own preflight.** The converted Index arm is preflighted with **its own**
  measured prefill rate and call latency (`throughput.converted_<n>`): `index/preflight_conv.json`, gate
  `I:preflight_conv`. It falls back to the stock numbers only if the converted ones are missing.

**Re-pricing.**
- The full-MMLU volume, computed exactly with the pinned tokenizer, is 2,485,632 padded positions and 156 calls per
  arm.
- At 5cmat's measured 22,144 tok/s and 0.237 s/call, plus 0.25 min of tokenisation per arm, that is 5.5 min for both
  arms.
- So P = 514.8 min, against a cap of 675 (1.31x). The deadline stays 585: 585 + 1.3 x (61.35 + 6.6) = 673.3 <= 675.
- max_duration is 825.
- **Worst case $84.65**, single attempt $40.84, **expected $27.87** (563 min). The hard cap stays $85.

## Info only (not binding): what 5cmat's single-swap numbers project for the 8-layer gate

**Inputs.** This costs $0, using art-2bybh's `importance.per_layer`: the single-swap gate numbers after stage 1,
before stage 2, in 5cmat's KL order. The table also gives the running sums.

| Layer (order) | dNLL | 1 - cos | Σ dNLL | Σ (1 - cos) | Additive cos |
|---|---|---|---|---|---|
| 23 | 0.0109 | 0.0053 | 0.0109 | 0.0053 | 0.995 |
| 31 | 0.0179 | 0.0064 | 0.0287 | 0.0116 | 0.988 |
| 15 | 0.0077 | 0.0058 | 0.0364 | 0.0175 | 0.983 |
| 11 | 0.0088 | 0.0064 | 0.0452 | 0.0239 | 0.976 |
| 7 | 0.0013 | 0.0091 | 0.0465 | 0.0330 | 0.967 |
| 19 | 0.0136 | 0.0073 | 0.0601 | 0.0403 | 0.960 |
| 3 | 0.0051 | 0.0139 | 0.0652 | **0.0542** | **0.946 < 0.95** |
| 27 | 0.0281 | 0.0134 | 0.0934 | 0.0676 | 0.932 |

**Stage 2 on layer 23 in 5cmat.** It cut dNLL by a factor of 0.63 (0.0109 -> 0.0068). It did *not* improve the
cosine: 1 - cos went 0.0053 -> 0.0055.

**The projection.**
- **dNLL.** The cumulative dNLL projects to 0.058-0.093, inside the 0.15 budget under either scaling.
- **Final-hidden cosine.** Under the independent-error model (additive 1 - cos), the cumulative cosine crosses the
  0.95 gate at the 7th layer (layer 3, cos ≈ 0.946) and ends near 0.93. If the errors add coherently (angles add),
  it is far worse: cos ≈ 0.52.
- **Stage-2 scaling.** Applying the stage-2 scaling seen on layer 23 changes nothing, because stage 2 did not move
  the cosine.

**What this suggests.** The most likely pre-registered outcome is `partial-conversion` with layers 3 and 27 rejected
(`Q9-H2`), unless joint stage-2 training of later layers reduces the accumulated hidden-state error better than
layer 23's single-layer stage 2 did. This is an information-only estimate: single-swap numbers do not compose
exactly, and the rule is unchanged by it. It is the main risk to the `q9af-backbone` verdict and should be weighed
before the spend. A rejected layer is not retried; see the rule section.

## Re-pricing 2026-10-09: H100 batch $0.0495 -> $0.063/min

SF Compute raised the H100 list price $3.96 -> $5.04/h on 2026-10-08; jobs now bill $0.063 per GPU-minute (Mon-Thu and weekends; Fridays $0.05628) and the preflight quotes max_duration x $0.063. Evidence and history: `jobs/common/rates.py`. The dollar figures above this section are at the old $0.0495/min and stay as written (receipts and past decisions used them); the numbers below are the current ones (`python -m jobs.common.costs`). Where a hard cap was raised, it was raised by the same ratio as the rate (x 0.063/0.0495 = x 14/11, rounded up to the cent), so the cap still buys the same minutes.

| Params | Worst case (build + 2 attempts) | Hard cap | Expected | One attempt (quoted max_cost) |
|---|---|---|---|---|
| `job_params.json` | $107.73 (was $84.65) | $108.19 (was $85.00) | $35.47 (was $27.87) | $51.98 (was $40.84) |
