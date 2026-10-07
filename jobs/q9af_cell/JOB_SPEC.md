# q9af_cell: the first Q9-AF decision cell (Clef-flash recipe analogue, attention-free). DRAFT, NOT SUBMITTED

**Question.** Does the Clef-flash recipe (frozen backbone + LoRA + a trained decision head, LS-CE + Brier, synthetic
permuted decision data) move the attention-free Qwen3.5-9B (all 8 attention layers converted to Gated DeltaNet:
"Q9-AF", job-2xe5w) off its stock readout toward Clef-flash (55.78 on the 2,008-row DEV subset; 56.79 full DEV; 57.35
full DEV with the 51 over-length rows at 32k, job-ftpnf), and does LoRA rank 256 beat rank 64 (E9, techniques/10)?

**Decision rule.** `jobs/q9af_cell/rule.py`, pre-registered in its own commit **2d18afa** before any q9af_cell run and
before the export existed; constants pinned by `tests/test_q9af_cell_rule.py`. Binding edition 0.2.1 (0.3 reported).
Per variant: PROMOTE (G12 + G13 accepted, paired full-DEV gain CI > 0, index >= 50), CONTINUE (gain CI > 0, index >= 44),
FOLD (trained but gain CI upper < 3 or index < 40, or nothing else matched), ESCALATE (index >= 56.79 Clef-flash, or
G13 rejected with a positive capability gain: decision-needed), HYBRID_REPORT (a partial export: never promoted),
THROUGHPUT (run_cell exit 26), NO_MEASUREMENT. Pair: R256 iff paired r256 - r64 full-DEV CI lower > 0, else R64.
G12 / G13 and their margins are `train.m2_gates` / `train.promotion.MARGINS` (PR #102, Codex #36 contract): read
through the gates exit code, never re-derived. A gate REJECT is a measured outcome (job exit 0, verdict recorded).

**Amendment A1 (2026-10-07, after independent review, before any data)** (its own commit; rule.py, its tests and this
text only; nothing loosened):
- A G12 reject (rc 40) can never yield CONTINUE or PROMOTE (rc 40 / 42 -> FOLD unless an ESCALATE condition holds first).
- An underpowered G13 (rc 45) yields INCONCLUSIVE, never PROMOTE or CONTINUE (action: expand the frozen retention
  evaluation and re-gate the same checkpoint).
- A run that failed its training-pass rule cannot PROMOTE.
- Bootstrap B = 2000 and seed = 0 are pinned in the rule (`BOOTSTRAP_B`, `BOOTSTRAP_SEED`), not in summarize.py; a
  verdict computed under any other B / seed is NO_MEASUREMENT.
- Every verdict maps to an action (`rule.ACTIONS`, pinned by the tests).

## Modes (one runner, `jobs/q9af_cell/q9af_cell_job.sh`, env `Q9AFC_MODE`)

| Mode | What | depends on |
|---|---|---|
| `parent` | the Q9-AF PARENT ARM once: the stock decision readout of the converted model on cohort v3 (dev, steer_dev, steer_gate, G13 retention from the 150k pool), the FULL DEV slice frozen first (`jobs.common.dev_full`), cached as `parent_arm/` | the export pin |
| `cell_r64` / `cell_r256` | one cell each: training + child evals (G12 + G13 vs `parent_arm/` as comparator AND lineage anchor) + summary + verdict | `parent` (its `arm.json`, units and frozen-set sha256 pinned in the cell params) |

The parent arm is rank-independent (unit identity = cohort, K/T caps, template, layout, items; the two cell configs
differ only in `lora_rank` / `variant`, asserted by `tests/test_q9af_cell.py`), so one parent job serves both cells.

## Cell (`q9af_cell_r64.json`, `q9af_cell_r256.json`)

| Item | Value | Why |
|---|---|---|
| Parent | `Qwen/Qwen3.5-9B@c2022362` + the job-2xe5w export via `heads.backbone.load_q9af_backbone` (`q9af_from_stock`), pinned: `converted_sha256`, `export_json_sha256`, `base_config_sha256` | PENDING: `python -m jobs.q9af_cell.pin fill --export-json <export.json> --gdn <gdn_converted.pt>` when job-2xe5w lands, committed alone |
| Partial export | refused (exit 10) unless `q9af.hybrid: true`; then loaded with `allow_partial`, LoRA also on the kept attention layers, labelled `kind: q9af-hybrid`, `attention_free: false`, "NOT attention-free" in every report; rule verdict HYBRID_REPORT | PREMORTEM S1-F4 (late layers may roll back) |
| Head | `fork` (final-state fork MLP, cache fork) | best available attention-free head now; hook `head_from_proxy` for the proxy loop's winning mixer head (resolved into a new committed config, never at run time; a non-attention-free head is refused) |
| LoRA | r = 64 / r = 256 on the GDN (`in_proj_qkv`, `in_proj_z`, `out_proj`) and MLP projections of all 32 layers (159.4 M / 637.5 M params) | E9 single-factor pair; Clef-flash uses r = 256 |
| Loss | `clef` arm: LS-CE (0.1) + Brier, + RPS on score questions | the Clef-flash loss (RPS ablation is a proxy-loop question) |
| KD | WIRED (PR #116, b511643), OFF in both default cells (`teacher: null`; their TrainConfig hashes are unchanged: r64 `cea41d9610c90faf`, r256 `90d6e79a70c23165`, pinned by `tests/test_q9af_cell.py`). Variant `q9af_cell_r64_kd.json`: alpha 0.5, T 1 (proxy recipe R16), teacher = Clef-flash export art-ggrs8 (job-tnrwu, sha256 `a88b4fe0…`, `train.proxy_rule.TEACHER`); `train.run_cell.attach_teacher` loads it through `train.teacher.load_teacher` (identity, file sha256, contamination) and requires the cell's draw to BE the teacher's draw (`2bbbf180…`); m2_gates rebuilds the items the same way. **Committed, not launchable**: there is no `job_params_cell_r64_kd.json` (launch.py refuses the mode) until the proxy loop's KD recipes R16/R17 report | KD on the same draw the teacher scored |
| Data | PR #44 150k set (pointer `sample-150000-seed20261006`), 24,000 records by `sha256(seed:id)`; cohort **v3** (`498efa61…76da`) excluded by id, content and source item (`train.train_set.draw` + `cohort_v3.training_exclusion`); pinned D1-B exclusions (3 MMLU aux records); D1-B heldout gate over the whole set (exit 33); **frozen-family check** (`harness.heldout.check_record`, heldout-draw-v1) over every drawn record (exit 34) | GOAL integrity; INDEX_0.3 frozen new-domain proxy |
| Execution | `act_ckpt: true` (this PR), `row_batch: 1` | PREMORTEM blocker #4; row batching: below |
| Steps | 8 rows / optimiser step (batch 1 x accumulation 8), warmup 50, `max_steps` 2,800 (~1 epoch of the draw's ~21.8k kept rows), `min_steps` 1,000 (exit 26 if the MEASURED step time cannot reach it), `max_minutes` 200 (cumulative over restarts), T ladder 1,024 / 2,048 / 4,096, K <= 16, `eval_max_t` 4,096 | the 455 / 2,008 DEV rows above 2,048 tokens are now trainable (PREMORTEM S2-F4) |

**TODO before `cell_r64_kd` gets job_params** (not done now; PR #121 review): a dated, pre-registered rule
amendment for the single-factor KD question (paired child(kd) - child(r64) on full DEV, 0.2.1, its CI rule and the
action of each outcome), committed BEFORE the KD params and before any KD run; the KD cell's own pricing and
preflight; and the teacher data pin, already committed as `jobs/q9af_cell/kd_context.json` (context `data` entry
`teacher/teacher-clef-flash.jsonl` pinned by `pin_json` to the KD config's `teacher.teacher_sha256` + its
`manifest_files`), which the KD params must include (`tests/test_q9af_cell.py`).

## Activation checkpointing for Q9-AF (PREMORTEM blocker #4)

`heads/act_ckpt.py` now covers HF Qwen3.5-family decoders: each `Qwen3_5DecoderLayer` runs under
`torch.utils.checkpoint` (non-reentrant) with its cache slot moved OUT of the checkpointed region: the GDN conv tail
and recurrent state (or, for a kept attention layer, K/V) are read before the block, the block runs on a private
one-layer holder (functional updates, `_assign_not_copy` semantics; the 1-token decode path's in-place conv update
hits a clone), and the new state is written into the real cache once, after. Only block inputs are saved.

CPU evidence (`tests/test_q9af_act_ckpt.py`, transformers 5.17.0, fp32): one and two optimiser steps of the real
`Trainer` (cache-fork head, clef arm, LoRA incl. the converted layer) with and without checkpointing give a
**bit-identical loss and bit-identical LoRA / head gradients** on Q9-AF, a hybrid (one attention layer kept) and the
stock hybrid; also multi- and 1-token continuations (CPU torch fallback only: on the fla image the fork REFUSES
1-token continuations under autograd, since fla's single-token kernel has no backward; `jrt-chatml-v1` continuations
are always >= 2 tokens), the recompute fork / plain features, no_grad bypass, and
fail-closed refusal of unsupported cache kinds (offloaded, past-recording). Saved-for-backward slope on the stand-in
(d = 64): **112.8 -> 5.0 bytes per token per layer per d (22.5x smaller)**
(`receipts/2026-10-07-D-q9af-act-ckpt/cpu_saved_bytes.json`).

**Row batching (not done; why).** The batched-row cache fork (`fork_features_rows`) needs right-padding to be exact.
For the HF GDN it is not: the causal conv runs over padded positions and the gated decay keeps acting there, so a
padded row's state differs from its unpadded prefix. Left padding is exact only in exact arithmetic (chunk alignment
changes the rounding) and shifts RoPE in a hybrid. A varlen (`cu_seqlens`) path would be exact on fla but has no CPU
reference to test against. At 9B the per-row call is not launch-bound the way job-mden7's eager RWKV7 was (a 619-token
row is ~2.7e13 FLOPs per pass), so the cell trains one row per call; the price below charges a per-call floor anyway.

## Memory and speed projections (`python -m jobs.q9af_cell.price`; assumptions named in `price.ASSUMPTIONS`)

Measured inputs: job-5cmat (same image, same 9B revision): prefill 22,144 tok/s (min over 4k-32k), fitted per-call
0.0595 s, small call 0.0493 s, stage-1 MFU 0.316, model load 43 s. CPU act-ckpt ratio above. Assumed: training MFU 0.30,
mean row 619 tokens + K 3.4 x 4 continuation tokens (PR #74's 2,000-record sample; Qwen tokenizer assumed equal),
6N FLOPs/token with checkpointing (forward + recompute + input-gradient backward; frozen base) + LoRA terms, a per-call
floor of 3 passes x 2 calls x 0.0493 s per row, 0.97 MB/token/layer without checkpointing (PREMORTEM P4 anchor).

| | r64 | r256 |
|---|---|---|
| step (8 rows): compute + per-call floor | 0.74 + 2.37 = **3.11 s** | 0.84 + 2.37 = **3.21 s** |
| 2,800 steps | ~145 min | ~150 min |
| steps inside max_minutes 200 | 3,862 | 3,744 |
| peak at T = 4,096, K = 16, **with** act_ckpt | **25.9 GB** (weights 15.9, LoRA+Adam 2.6, states 2.2 incl. ~1.1 retained by the checkpoint closures until backward, activations 5.3) | **33.6 GB** (LoRA+Adam 10.2) |
| peak at T = 4,096 **without** act_ckpt | 146.7 GB (OOM) | 154.3 GB (OOM) |
| max T under the 76 GB stop, without / with act_ckpt (model) | 1,819 / > 4,096 | 1,572 / > 4,096 |

The ladder measures the real peak at each rung on the box (L-12: estimates have been off by 2x) and picks T_max; the
10-step check measures the real step time (exit 26 if 1,000 steps are out of reach in the time left).

Scoring (one arm, stock readout rates): G12 (dev + steer_dev) 17.9 min, G13 213.5 min = cohort v3 retention at the
cells' unit spec T <= 4,096 (**97.7 M positions / 86,452 calls = 159.3 min, 41,991 usable items**, measured with the
real Qwen3.5 template + tokenizer: `receipts/2026-10-07-D-q9af-cell-v3-retention-t4096/`; the earlier T <= 2,048
receipt priced 79.7 min, review B1) + steer_gate + the full DEV slice (38.8 M fork positions, every frozen row, no T
filter); the child adds the temperature fit + dev card (9.5 min) and re-tokenisation (15 min).

## Phases (caps >= 1.3 x projection: `jobs/q9af_cell/preflight.py`, run by the runner at setup, exit 5)

| Mode | Phase | Cap | Projected / expected |
|---|---|---|---|
| parent | setup / guard / parent_g12 / parent_g13 / report | 25 / 5 / 30 / 280 / 5 | 4 / 1 / 18.6 / 214.2 / 1 |
| cell_r* | setup / guard / heldout / frozen / cell / evals / report | 25 / 5 / 10 / 10 / 310 / 335 / 10 | 4 / 1 / 3 / 5 / 234.7 (180 expected) / 256.6 / 3 |

Schedules: parent 345 + reserve 10 = **355 min**; cell 705 + reserve 15 = **720 min** (= the 720-min org ceiling,
so no split is needed yet; any growth splits the cell at the cell / evals boundary into its own resumable job, the
checkpoint dir carrying over).
Resume: `resume: checkpoint`; run_cell resumes from `$GMN_CHECKPOINT_DIR/cell` (smoke / ladder skipped from the
mirrored `cell.json`, training from the last checkpoint, the 200-min budget reduced by `train_clock`).

## Cost (H100 batch $0.0495/min; worst case = (build 60 + 2 x max_duration) x rate, `jobs.common.costs`)

| Mode | max_duration | Worst case | Hard cap | Expected |
|---|---|---|---|---|
| parent | 355 | **$38.12** | $40 | ~$12.56 |
| cell_r64 | 720 | **$74.25** | $75 | ~$23.15 |
| cell_r256 | 720 | **$74.25** | $75 | ~$23.15 |
| campaign (parent, then both cells) | | $186.62 | | ~$58.86 |

Workspace cap: both cells submitted together after the parent: MTD + 149 (`costs.concurrent_cap_usd`); all three
chained with `depends_on`: MTD + 187.

## Exit codes

0 measured + verdict + exports verified (FOLD / ESCALATE / a gate REJECT included) | 5 setup / preflight | 6 export
verification | 7 phase timeout | 9 report invalid | 10 identity (data sha256, export pin unpinned or partial without
`hybrid`, parent arm pins) | 12 parent scoring | 20 run_cell unexpected | 21-29 run_cell stops | 30-32 guard |
33 held-out D1-B | 34 frozen-family hit | 35 config / hooks | 1 finalize.

## What the result shows

`q9af_cell_result.json` (compact, < 31 KiB, L-27) and `report.json`: child and parent Index on the full DEV slice and
on the 2,008-row subset (restricted to the same frozen rows), 0.2.1 (binding) and 0.3 public DEV without GSM8K
(reported), each with CI, areas, the paired child - parent difference, and the **in_domain / new_domain /
new_domain_frozen** columns (`harness.families`); G12 / G13 verdicts and failed criteria (anchor = the parent arm);
training (steps, T_max, ladder peaks and step times, loss, smoke); the export label; the Clef-flash references; the
rule verdict. `/output/adapters/adapters.pt` holds the trained LoRA + head weights (no optimiser state).
`python -m jobs.q9af_cell.summarize pair` gives the E9 verdict off-box from both cells' outputs.

## What it does not show

- One seed per rank; one head; the KD variant is committed but not launched (it waits for R16/R17).
- The full Index / TEST is never read; the full-DEV slice is a development slice.
- The child is scored at the stock readout's rates in the price; the head arm's own rate is measured only in-job.
- The dry run (`tests/test_q9af_cell_shell.py`) replaces the two GPU scoring phases by `jobs.q9af_cell.dry_gates`
  (synthetic rows, same file contracts); everything else in the runner runs as paid.

## Before submit

1. job-2xe5w lands: `python -m jobs.q9af_cell.pin fill ...` (both configs + `gdn.sha256`), committed alone; a partial
   export needs an explicit `--hybrid` and a decision-needed issue (a hybrid is not the goal model).
2. Independent fresh-context review; merge to `agent-main`.
3. Spend issue + LEDGER rows; `set_workspace_cap(MTD + 40)` for the parent, then pin its `arm.json` / units /
   `index-frozen.json` sha256 into `job_params_cell_r*.json` (`pins_for_cell_params` in its result) and commit.
4. `python -m jobs.common.launch q9af_cell --mode <mode> --sha <merged sha> --out <dir> --data $SSDM_DATA` with
   `Q9AF_GDN_PT` / `Q9AF_EXPORT_JSON` (and for the cells `Q9AF_PARENT_*`, `Q9AF_INDEX_FROZEN`) pointing at the artifacts.

## `submit_job` fields (do not submit until the steps above are done)

Mode `cell_r256`:

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9af-cell",
 "idempotency_key": "q9af-cell-r256-<context_sha256[:12]>",
 "label": "q9af_cell r256: Q9-AF LoRA r256 + fork head, clef arm, 150k draw, G12/G13 (rule 2d18afa)",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9af_cell/q9af_cell_job.sh",
 "env": {
  "Q9AFC_MODE": "cell_r256"
 },
 "max_duration_minutes": 720,
 "max_restarts": 0,
 "resume": "checkpoint",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9AF_CELL_DONE"
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

Mode `cell_r64`:

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9af-cell",
 "idempotency_key": "q9af-cell-r64-<context_sha256[:12]>",
 "label": "q9af_cell r64: Q9-AF LoRA r64 + fork head, clef arm, 150k draw, G12/G13 (rule 2d18afa)",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9af_cell/q9af_cell_job.sh",
 "env": {
  "Q9AFC_MODE": "cell_r64"
 },
 "max_duration_minutes": 720,
 "max_restarts": 0,
 "resume": "checkpoint",
 "queue_ttl_minutes": 1440,
 "hf_cache": true,
 "model": "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a",
 "checks": {
  "success": {
   "check": {
    "all_of": [
     {
      "type": "file_exists",
      "path": "/output/Q9AF_CELL_DONE"
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

Mode `parent`:

```json
{
 "chip": "h100",
 "chip_count": 1,
 "mission": "q9af-cell",
 "idempotency_key": "q9af-cell-parent-<context_sha256[:12]>",
 "label": "q9af_cell parent arm: Q9-AF stock readout, cohort v3 + full DEV (rule 2d18afa)",
 "context_id": "<from create_context>",
 "dockerfile_path": "Dockerfile",
 "build_timeout_minutes": 60,
 "command": "bash /opt/ssdm/m1/jobs/q9af_cell/q9af_cell_job.sh",
 "env": {
  "Q9AFC_MODE": "parent"
 },
 "max_duration_minutes": 355,
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
      "path": "/output/Q9AF_CELL_DONE"
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
