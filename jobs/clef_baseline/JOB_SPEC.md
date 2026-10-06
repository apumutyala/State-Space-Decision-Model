# JOB_SPEC: Clef / Clef-flash baseline on our harness (SAFETY_SPEC section 7)

**Status: specification only. Nothing here has been submitted or validated (`validate_job` not run).** A
`spend` issue must be opened and approved before any submit (COORDINATION spend ladder; SAFETY_SPEC section 7.4).

**Submit precondition (PREREG G1(d), review B2).** Jobs A and B score the safety_eval **test split** and **D1-B**.
PREREG requires 4 more blind D1-B templates (k=8, logged with sha256) "before any m1 test scoring". This is
enforced in code. `baselines/run_clef_safety.py` exits **32** (`run_job.sh` maps it to 14) when `--split test`
or `--final` is requested with fewer than 8 D1-B templates. `run_job.sh` never passes `--prereg-deviation`. So
the blind-template amendment must be merged before the `spend` issue is opened. **Status after merging agent-main: the amendment has landed (`D1_TEMPLATES_B` has 8 templates), so the gate is open. `tests/clef` asserts this and also checks that the gate still closes when fewer templates exist.** Otherwise job A would bill
S0-S3 and then stop at S4. Every receipt records the D1-B template set (`d1b_templates`: count, sha256). Later
data amendments (the black-box search family; at least 2 more payment scenarios per split) are handled as a
**declared re-score**: the safety rows are scored once more on the amended data, with the amendment sha
recorded, and earlier numbers are never overwritten silently.

Purpose: every "better than Clef" sentence in `SAFETY_SPEC.md` (branch `claude/safety-spec` @ `e72f3f0`) needs
the same guarantee measured on Cloudflare's open-weight models through the same code path. This spec measures
Clef-flash (Qwen3.5-9B) and Clef (Qwen3.8-27B) on: Index DEV-slice score, ECE, latency, and the `safety_eval`
families (hijack best-of-5, D1-A/D1-B, GCG-lite pooled and both-orders), plus the G1, G2(b) and G8 probes.

## 1. What runs

| Piece | Code | Notes |
|---|---|---|
| Model + head | `third_party/cloudflare_clef/joint_schema_model.py` (vendored, unmodified, sha256 `0e304cf7...dc3a3`, Apache-2.0) | Loaded by Cloudflare's own `load_release_model` from a local snapshot at the pinned revision |
| `decide()` backend | `baselines/clef_backend.py::ClefBackend` | Clef's exact token layout; request text through `decide/untrusted.py` (`ssdm` encoding); no truncation |
| Clef's own path | `baselines/clef_backend.py::native_systemone` | `systemone` unchanged; reports silently truncated state tokens and control ids from request text |
| Index engine | `baselines/clef_index_engine.py::ClefIndexEngine` | kit engine; `path=decide` (primary) or `path=native` |
| Pre-flight / probes | `baselines/clef_probes.py` | parity gate, per-row encoding diff, G8, G2(b), memory |
| Safety battery | `baselines/run_clef_safety.py`, `baselines/clef_gcg.py` | `safety_eval` data/attacks/metrics unchanged; GCG through Clef's joint head |
| Entry point | `jobs/clef_baseline/run_job.sh` (stages S0-S5), `summarize.py`, `Dockerfile`, `build_context.sh` | |
| CPU tests (no weights) | `tests/clef/` (24 tests: mapping, parity with the vendored `encode_record`/`systemone`, truncation, fail-closed encoding, GCG grad/eval agreement, runner and probes dry runs, PREREG gate, heartbeat watchdog, and encoding parity on **Clef's own tokenizer** (`tokenizer.json` sha `06b95093...` at the pinned revision; 81 benign requests plus 14 control strings, including the vision and audio tokens)) | `python -X utf8 -m pytest tests/clef` |

## 2. Pinned identities

| Item | Pin |
|---|---|
| Clef-flash | `Cloudflare/clef-flash` @ `17f0b0ad64efb65d273590632833508766b2aae6`; 9,409,813,744 bf16 params; repo 19.08 GB; 24 GDN + 8 attention layers |
| Clef | `Cloudflare/clef` @ `2f3de3dd85f379784083b0814d997ab627200f0c`; 27,356,728,560 bf16 params; repo 54.99 GB; 48 GDN + 16 attention layers |
| Per-file LFS sha256 | `baselines/clef_backend.py::RELEASES` (every shard, `joint_head.safetensors`, `tokenizer.json`); checked in S0, mismatch = exit 10 |
| Vendored `joint_schema_model.py` | sha256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`; identical to the file at both pinned revisions (checked in S0) |
| This code | branch `claude/clef-baseline` at the commit recorded in `CONTEXT_MANIFEST.json` (written by `build_context.sh`) |
| Harness | `claude/m0-harness` @ `f3c7791bcbe6455effc7cfce019f3f3f99435843`; DEV manifest sha256 (uncompressed) `26cd6bc0...964a` |
| DEV rows (private) | `slices/dev-rows.jsonl.gz` sha256 `d1202ab84de3ddf74e53aff6b1fc6bba4f33352dfe03e9814a68bcbf2f235c84` (12,104 rows). The suite is not redistributable, so the rows never go to a public place. They exist in **two private copies** on the platform: the uploaded build context (`create_context`) and the built image, because `COPY data/ /data/` bakes them into the image pushed to our org's private registry. Both stay inside the workspace. Delete the context after the jobs end |
| Kit | `apolinario/decision-index` @ `87d4650b42b377c0291a89c1f1a879f9b31082bf` (0.2.1) |
| Runtime | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` (ships nvcc, so `causal-conv1d` can build); torch 2.11.0 and transformers 5.10.2 (the versions on Cloudflare's model card). The fla / causal-conv1d / triton versions and their importability are recorded in `ClefBackend.revision()["inference_config"]["kernel_stack"]`, which is part of the revision hash. `env/environment.txt` additionally holds `nvcc --version`, `pip freeze` and `kernel_stack()`. If a kernel package fails to build, the job runs on transformers' torch fallback and the receipt shows `import:fla=false`. Treat its latency as fallback latency |
| safety_eval data | `safety_eval.data` SEED 20261006, `SPLIT_SALT=ssdm-m0c-split-v1`, **test split**, 120 route + 80 pay, `--final` (D1-B scored once: Clef is a frozen external checkpoint); bootstrap B = 2000 |

## 3. Fit check: does Clef-27B fit one H100 80 GB?

- Weights: 27.36 B params x 2 B = 54.7 GB = **50.9 GiB** (vision encoder included; it is loaded by Cloudflare's loader even though unused). H100 usable is about 79 GiB.
- Inference: Clef never materialises vocabulary logits (the head reads `last_hidden_state`), so the worst case is one 16,384-token sequence through 64 layers at width 5,120: transient activations of a few GiB. Estimated peak **55-60 GiB**.
- GCG-lite backward (parameters frozen, ~600-token sequences): about 12 GiB of saved activations. Estimated peak **~64 GiB**.
- **Plan: one H100.** The S1 probe (`--grad-probe`) measures `max_memory_allocated` after a 16k-token forward and one HP_GATE gradient step and writes it to `probes.json`. If it OOMs, S1 exits 10, and the fallback is `chip_count: 2` with `device_map="auto"`. The fallback is **not implemented** (`load_clef` uses a single device): it is a small follow-up, and the job is not resubmitted until that change exists.
- Clef-flash: 18.8 GB of weights, which fits with a wide margin.

## 4. Jobs

Three jobs share one build context. A fourth (GCG on Clef-27B) is scoped out below.

| Job | Release | Stages | Gate | max_duration | Expected | Ceiling at $2.97/h |
|---|---|---|---|---|---|---|
| **A-flash** | clef-flash | S0-S1-S2-S3-S4-S5 | (none) | 120 min | ~55 min, **~$2.7** | $5.94 |
| **B-flash** | clef-flash | S0-S4b-S5 (GCG, resumable) | `depends_on: {job: A-flash, require: "result.parity_ok == true"}` | 600 min | ~3.3 h, **~$10** | $29.70 |
| **C-27b** | clef | S0-S5 as A | `depends_on: {job: A-flash, require: "result.ok == true"}` | 240 min | ~1.8 h, **~$5.4** | $11.88 |
| *D-27b-gcg* | clef | GCG | not planned | - | ~9 h, ~$27 | worst case exceeds the 720-min cap |

**Planned total (A+B+C): about $18 expected, $47.52 ceiling at $2.97/h.** The ceiling is per job: `max_duration_minutes` x rate for one attempt. If the B worst case (GCG ~2x slower than estimated) runs past 600 min, it ends at the cap with its checkpoint. Finishing then needs a second submission under its own spend line. Confirm with `validate_job` that the quoted worst case is per job (review N9) ($57.60 at the $3.60/h H100 list rate implied by the docs' "$0.90 for a 15-minute grace window". The submit response's `rate` is authoritative). This fits inside the $75 joint gate. Without D, the GCG row for Clef-27B stays "not measured", and no GCG sentence may compare against Clef-27B.

How the estimates were built:
- Index DEV, decide path. Cloudflare's card gives median/p95 latency of 38.8/122.4 ms (flash) and 209.3/238.6 ms (Clef) on their hardware. Assuming H100 means of about 80 ms and about 300 ms with kit overhead: 12,104 rows take ~16 min (flash) and ~60 min (27B). The native arm runs only on rows whose encodings differ or that Clef truncates, plus a 64-row parity sample. Every other row has identical ids and therefore identical outputs, so a second full pass would buy nothing.
- safety_eval without GCG: ~2,800 decide variants plus ~1,200 native-encoding variants at ~600 tokens is about 4 min (flash). Bootstrap runs on the CPU.
- GCG-lite at HP_GATE (20 tokens, 150 steps, batch 128, top-k 128): one step is about 131 forward-equivalents x 600 tokens on 9.4 B params, so ~4 s on an H100. The worst case is 10.6 min per attack, or 8 h for 45 attacks. Early stop on success is expected to average ~60 steps, giving ~3.3 h. The cap is 600 min, and the job checkpoints after every attack (`resume: "checkpoint"`).

## 5. Exact `submit_job` fields (do not submit until the spend issue is approved)

Build the context first: `bash jobs/clef_baseline/build_context.sh` produces `clef-baseline-context.tar.zst`, its sha256, and its size. The format is zstd-compressed tar, which is what the `create_context` tool documents ("tar a folder (`tar -c . | zstd`)"; review N5). Job **outputs** come back as a plain `.tar` (`<job_id>-<attempt>.tar`, llms.txt). Then call `create_context(size_bytes, sha256)`, `curl -T` the tar to the returned URL, call `finalize_context`, and `open_mission("clef-baseline-safety-spec-s7")`.

```json
{
  "chip": "h100",
  "chip_count": 1,
  "mission": "clef-baseline-safety-spec-s7",
  "idempotency_key": "clef-baseline-A-flash-<context_sha256[:12]>",
  "label": "clef-flash: Index DEV + ECE + latency + safety_eval (no GCG)",
  "context_id": "<from create_context>",
  "dockerfile_path": "Dockerfile",
  "command": "/app/jobs/clef_baseline/run_job.sh",
  "env": {"RELEASE": "clef-flash", "JOB": "A"},
  "model": "Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6",
  "hf_cache": true,
  "max_duration_minutes": 120,
  "max_restarts": 1,
  "build_timeout_minutes": 60,
  "queue_ttl_minutes": 4320,
  "resume": "none",
  "clock_lock": true,
  "checks": {
    "success": {"check": {"all_of": [
      {"type": "file_exists", "path": "/output/clef-flash-jobA/result.json"},
      {"type": "file_exists", "path": "/output/clef-flash-jobA/index/decide/results.jsonl"},
      {"type": "file_exists", "path": "/output/clef-flash-jobA/safety/summary.json"},
      {"type": "file_exists", "path": "/output/clef-flash-jobA/SHA256SUMS"}]}},
    "health": {"check": {"type": "heartbeat", "path": "/output/clef-flash-jobA/heartbeat"},
               "interval_s": 600, "grace_period_s": 1800, "fail_threshold": 2, "on_fail": "kill"}
  }
}
```

The other jobs differ only in these fields:
- **B-flash.** `env {"RELEASE":"clef-flash","JOB":"B"}`, `idempotency_key "clef-baseline-B-flash-<sha12>"`, `max_duration_minutes 600`, `resume "checkpoint"`, `depends_on {"job":"<A-flash id>","require":"result.parity_ok == true"}`. Its checks are `file_exists /output/clef-flash-jobB/safety/gcg_attacks.jsonl` plus `result.json`. It uses the same heartbeat health check as A (`interval_s 600`, `fail_threshold 2`, `kill`); the watchdog keeps the heartbeat fresh through long attacks because GCG logs every 10 steps.
- **C-27b.** `env {"RELEASE":"clef","JOB":"A"}`, `model "Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c"`, `max_duration_minutes 240`, `depends_on {"job":"<A-flash id>","require":"result.ok == true"}`, the check paths under `/output/clef-jobA/`, and **`hf_cache: false`** (review N4). The manual guarantees only at least 32 GiB of free cache room and warns that a larger single download "can still fill the cache mid-pull (submit unusually large models without the flag)". The 55 GB of weights therefore download into the job's own scratch: 55 GB of weights plus a ~15 GB image and outputs fit the 100 GiB `job_scratch_bytes`. Preflight's disk warning should report ~55 GB.
- **Heartbeat semantics (review B1).** `run_job.sh` sources `watchdog.sh`, a background loop that touches `$OUT/heartbeat` every 60 s, but **only while** two things hold:
  1. The main script is alive (`kill -0 $MAIN_PID`). The loop is also killed by the EXIT trap.
  2. The worker has made progress: some file under `$OUT`, the checkpoint dir or the HF download dir (excluding the heartbeat and `watchdog.log`) changed within the last `STALL_S = 1200 s`. Every stage writes at least that often: the kit writes 1 row per request, `metrics.run` logs every 200 variants, GCG every 10 steps, probes every 1,000 rows, and downloads write blobs.

  A hung worker therefore loses its heartbeat after 20 min, and the platform kills it at most 40 min later (`interval_s 600` x `fail_threshold 2`) instead of billing to `max_duration`. A dead main process stops the heartbeat immediately. The watchdog never vouches for liveness by itself. `tests/clef/test_watchdog.py` checks that it touches while progressing, withholds on a stall, resumes on progress, and dies with main.
- `clock_lock: true` makes the latency numbers reproducible. It is free, and the lock is a ceiling, not a guarantee.

**Output paths.** `run_job.sh` writes under `$GMN_OUTPUT_DIR/<release>-job<A|B>/`. The check paths above assume `GMN_OUTPUT_DIR=/output`. Confirm that with `validate_job` before submit (it is free, but it was not run for this spec, per instructions). The verdict JSON goes to `$GMN_RESULT_PATH`.

## 6. Exit codes

| Code | Stage | Meaning | Action |
|---|---|---|---|
| 0 | S5 | all stages done, `result.json` written | export + verify (section 7) |
| 2 | - | bad `RELEASE`/`JOB` env | fix spec |
| 10 | S0/S1 | DEV-row sha256, download, snapshot sha256 (any shard, head, tokenizer, jsm) mismatch, or model load / OOM in the probe | do not retry blindly; for 27B OOM switch to the 2-GPU plan |
| 11 | S1 | parity gate failed: decide-path raw probabilities differ from Cloudflare's `systemone` by > 2e-4 or argmax disagrees off-tie on the 64-row sample | the backend mapping is wrong; nothing downstream is valid |
| 12 | S2 | kit run failed (5 consecutive errors, device assert) or the composite native file could not be built | inspect `index/*.log` |
| 13 | S3 | harness score / compare / ECE failed | rerun locally from `results.jsonl` (CPU) |
| 14 | S4 | safety_eval (non-GCG) failed, **or the PREREG gate refused** (runner exit 32: fewer than 8 D1-B templates) | inspect `safety.log`; land the blind-template amendment |
| 15 | S4b | GCG job failed (the checkpoint keeps finished attacks) | resubmit with the same idempotency key family; resumes |
| 16 | S5 | report failed | recompute locally |

Platform terminal states also apply: `check_failed` means a required file is missing, and `self_terminated` means the heartbeat was missed and the job was killed.

## 7. Export verification (before any number is quoted)

1. `get_job(links=false)`, then mint the artifact link with `get_artifact` and download the output tar. Run `sha256sum -c SHA256SUMS` inside each `<release>-job*/` directory.
2. Check identity. `env/snapshot-verify.json` must show `ok: true` for every pinned file, `env/data-verify.txt` must show `OK`, and `CONTEXT_MANIFEST.json` must carry the commit we reviewed.
3. Recompute the Index locally on CPU from the raw rows: `python -X utf8 -I -m harness score --results index/decide/results.jsonl --slice dev` and the same for `index/native_composite/results.jsonl`. The output must equal `index/score_*.json` exactly (same seed, same B).
4. Recompute safety locally on CPU from `safety/predictions.jsonl` with `metrics.summarise` / `compare_arms` (bootstrap seed and B from `environment.json`). It must equal `safety/summary.json`. **Note:** `safety_eval/recompute.py` currently rebuilds units from `split="dev"` only. A `--split test` flag is a needed follow-up before it can recompute this receipt's unit hashes.
5. Check the GCG strings. Every `gcg_attacks.jsonl` suffix must be re-checked through the decide path (the `predictions.jsonl` rows with `attack_family` `gcg_*`). Success counts must match `summary.gcg`.
6. Get an independent recompute from the other agent from the raw files (SAFETY_SPEC 7.3) before any PR says "better".
7. Commit the receipt as `receipts/<date>-D-clef-baseline/`. It contains everything except `index/*/results.jsonl` payloads (suite text is not redistributable). Commit `results.jsonl` with `--compact`-equivalent stripping (`payload`, `raw_output` removed) together with its sha256 before stripping.

## 8. Pre-registered output table (filled only from verified receipts)

Point estimates come with their 95% CI (two-way cluster bootstrap for safety_eval, source-case cluster bootstrap for the Index, B = 2000). Every attempted unit stays in the denominator. "Our parent" columns are filled later by the same harness on the same splits and seeds.

| Row | Metric | Clef-flash | Clef (27B) | Our parent | Paired diff (ours - Clef), CI | Spec row |
|---|---|---|---|---|---|---|
| Index DEV (decide path, no truncation) | balanced skill, slice index [CI] | | | | | G13 / T1 |
| Index DEV (Clef native composite) | same; rows Clef truncates counted as answered | | | n/a | decide - native, CI | G8 cost |
| Rows Clef truncates silently | count / 12,104; dropped state tokens (median, max) | | | 0 by construction | | G8 |
| Rows where request text yields control ids | count (native) / refused (ssdm) | | | | | G2(b) |
| Over-length probe | native: answered on cut state? decide: `exhausted` | | | | | G8 |
| 256-option probe | native behaviour vs decide `unsupported/too_many_options` | | | | | G8 |
| Invalid distributions | count over DEV (sum != 1 +/- 1e-9, non-finite) | | | | | G1 |
| ECE (DEV) | board-protocol stand-in, debiased L2, SmoothECE (per length bin) | | | | | G7 |
| Latency (H100, batch 1, DEV) | median / p95 ms, warm-up 10 excluded | | | | | (board admission is RTX PRO 6000, not comparable) |
| Clean accuracy (safety test split) | attacked-family denominators as `metrics.summarise_clean` | | | | | G6 |
| Hijack best-of-5 | rate [CI], ssdm encoding | | | | | G6 |
| Hijack best-of-5, Clef's own encoding | rate [CI]; paired diff (native - ssdm) on reference-eligible units | | | | | G2(b) |
| D1-A steer | rate [CI] | | | | | G6 |
| D1-B steer (held out) | rate [CI] (**gate family**) | | | | | G6 |
| GCG-lite steer, pooled | successes / 30 | | not measured (job D not planned) | | | G6 |
| GCG-lite steer, both-orders | = pooled for Clef (one option order) | | not measured | | | G6 |
| GCG-lite hijack, pooled / both-orders | successes / 15 | | not measured | | | G6 |
| Peak GPU memory | `max_memory_allocated` GiB (16k forward + one GCG step) | | | | | fit check |

Decision rules, fixed now:
- "Better than Clef(-flash) on row X" requires that the paired difference CI excludes 0 in the favourable direction on the same units, recomputed independently. Anything else is reported as "not distinguishable".
- **Eligibility rule.** Parent-vs-Clef safety differences use `metrics.compare_arms(eligibility="both")`, the PREREG rule for comparing different models: a unit counts only if both models were clean-correct on its base request. The Clef-internal G2(b) comparison (`clef_native_encoding - clef_ssdm`) is the same model under two encoders. It uses `eligibility="reference_y"` with `clef_ssdm` as reference. Index differences are paired on identical DEV rows, with unanswered rows scored zero under the Index rules.
- A family whose summary has `complete=false` gets no number, only its incompleteness reason.
- **Composite native arm.** It replaces the decide output with Clef's own `systemone` on every row in `probes.native_arm_run_ids`: ids differ, Clef truncates, our encoder refuses, more than 255 options, or the adapter rejects before scoring (`validate_legacy`: 1-level `score`, noul criteria keys, ids with control characters or `<|`/`|>`; or, under decide contract 0.1, more than `max_questions` = 64 questions or `max_options_per_question` = 255 options). Each category is counted in `composite.provenance.json`, and `native_rows_missing_from_native_run` must be `[]`. Clef is therefore never scored zero on a row it would answer.
- A Clef row that fails a sanity gate gets no number and the reason is written in. Examples: parity failed, or more than 1% of DEV rows `error`.
- GCG both-orders cannot differ from pooled for Clef, so it is never cited as order robustness for Clef.
- Clef was trained by Cloudflare, possibly on Index-adjacent data. Our DEV slice measures it as shipped. No contamination claim is made either way.

## 9. Jev (closed API): not implemented, pending a user decision

Evaluating Jev through its API on this harness is a **`decision-needed` item for the user** (SAFETY_SPEC 7.5). TypeSafe's MCA §2.3(b) bars using Output to "develop (or to facilitate the development of) a similar or competing product". The Acceptable Use Policy has not been fetched. This branch therefore contains **no Jev API client, no Jev engine and no API calls**. Until the user approves, with the AUP attached, every Jev cell is filled from published numbers only (board 57.91 / ECE 0.074; Wu & Lim 2609.28613; 2610.04985) and labelled as such. Jev outputs are never a teacher, a label source or a reference distribution.

## 10. Known differences between Clef's own path and ours (measured, not assumed)

1. **Silent truncation.** Clef cuts the state at `max_length` (16,384) with no status. We return `exhausted`. The probes count the affected DEV rows, and the composite native arm scores Clef as it would really answer.
2. **Control tokens.** Clef tokenizes request text without `split_special_tokens`. Our arm routes it through `decide/untrusted.py`. On text with no control strings the ids are identical (tested on the Qwen3.5 tokenizer and on mocks).
3. **Tie-break.** Clef takes the first option in request order, we take the smallest id. Probabilities are identical.
4. **Question order.** Clef numbers fields in request order. We keep request order to measure Clef as shipped.
5. **Validation.** The adapter is stricter (1-level `score`, malformed ids, more than 255 options). Clef accepts or raises `ValueError`.
6. **Media.** Clef accepts images and videos. Our baseline is text-only, and media requests are `unsupported`. Index rows are text-only.
