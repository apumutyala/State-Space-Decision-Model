# JOB_SPEC: Clef / Clef-flash baseline on our harness (SAFETY_SPEC section 7)

**Status (2026-10-06): job A (Clef-flash) ran as `job-sgihj` ($1.40) and stopped at exit 12 in the composite step
(root cause and fix: section 12; the safety battery S4 therefore never ran). Jobs B, C and the new Clef-27B Index
(section 4b) and teacher-export (section 11) jobs are specification only: not submitted, `validate_job` not run.** A
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
| Pre-flight / probes | `baselines/clef_probes.py` | parity gate, fast-vs-reference GDN kernel check (`kernel_path`), per-row encoding diff, G8, G2(b), memory |
| Image guard | `jobs/common/fla_guard.py` (shared with qual/probe9b/rwkv7) | fla #640 check + tiny gated fwd+bwd on the job GPU, before any download or GPU phase; exit 5 |
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
| Runtime | `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel@sha256:09c0260351b12b617408ff5711e08d88ccdbb9a21f7f6d7de67e39039b022ae1`, the same base as qual/probe9b/rwkv7. It carries torch 2.13.0, triton 3.7.1, CUDA 12.6 and nvcc. Pins: `transformers==5.10.2` (the model card's version), `flash-linear-attention==0.5.2`, `fla-core==0.5.2`, `causal-conv1d==1.7.0` (best effort; built from source), `FLA_TILELANG=0`, and tilelang not installed. A build-time assert checks the torch/CUDA/triton versions. `kernel_stack` (in the revision hash) records torch, triton, fla, causal-conv1d, `FLA_TILELANG` and whether tilelang is installed. `env/environment.txt` holds `nvcc --version`, `pip freeze` and `kernel_stack()`. Why the image changed: see section 2a |
| safety_eval data | `safety_eval.data` SEED 20261006, `SPLIT_SALT=ssdm-m0c-split-v1`, **test split**, 120 route + 80 pay, `--final` (D1-B scored once: Clef is a frozen external checkpoint); bootstrap B = 2000 |

## 2a. Image: fla #640 and the deviation from the model card

**Why.** The PR #41 review found that Clef was wrongly exempt from the fla #640 fix as "inference-only". In fact the S1 `--grad-probe` and the whole of job B (GCG-lite, `clef_gcg.py` `loss.backward()`) run backward through Qwen3.5's Gated-DeltaNet layers. transformers 5.10.2 routes those layers to fla's `chunk_gated_delta_rule` whenever fla imports, and that does not depend on causal-conv1d (`modeling_qwen3_5.py`: `self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule`). fla 0.5.2 refuses the gated `chunk_bwd_dqkwg` on Hopper with 3.4.0 <= Triton < 3.7.1. The old image (torch 2.11.0, so triton 3.6.0, with unpinned fla) would therefore have stopped at S1 on an H100, after the weights download.

**Fix chosen: move to the shared torch 2.13.0 / triton 3.7.1 base with `FLA_TILELANG=0`.** The alternative was to pin fla and send the gradient phases through a non-gated path, i.e. the torch reference GDN. It was rejected because it would measure GCG against a different kernel than the forward that serves the Index and the safety battery, and GCG at ~131 forwards per step would run several times slower on the reference path. The runner calls `jobs/common/fla_guard.py` right after S0's data check, before the weights download and before every GPU phase of jobs A and B. Exit 5 means the guard refused the image.

**Declared deviation from Cloudflare's card** ("Tested with torch 2.11 and transformers 5.10.2"): torch is 2.13.0, not 2.11. Everything else that defines the model stays as on the card:
- transformers is kept at **5.10.2**. The Qwen3.5 model code, `Qwen3_5ForConditionalGeneration` and `AutoProcessor` are therefore the card's. Its requirements (`torch>=2.4`, `huggingface-hub>=1.5,<2`, `safetensors>=0.4.3`) are met by the pins.
- `joint_schema_model.py` (vendored, unmodified) uses only stable `torch.nn` pieces: `LayerNorm`, `Linear`, `Embedding`, `MultiheadAttention(batch_first)`, `TransformerDecoderLayer(norm_first, batch_first, gelu)`, `functional.normalize`, `cosine_similarity` and `softmax`. It also uses `safetensors.load_file(strict=True)`. None of these is among the known torch 2.12/2.13 removals (e.g. `torch.cholesky`/`qr`). The CPU tests run Cloudflare's head and `ClefModel` on torch 2.14.1, newer than 2.13, and match Clef's own `systemone` (tolerance 1e-6 in fp32; about 5e-8 measured in the #19 review).
- The fla API that transformers 5.10.2 calls was read from both wheels. `chunk_gated_delta_rule(q, k, v, g=, beta=, initial_state=, output_final_state=, use_qk_l2norm_in_kernel=, cu_seqlens=)`, `fused_recurrent_gated_delta_rule` and `FusedRMSNormGated(hidden, eps, activation, device, dtype)` all exist with those parameters in fla-core 0.5.2. transformers requires fla >= 0.2.2.

**Not claimed: bitwise identity with Cloudflare's runs.** cuBLAS, SDPA and fla Triton numerics can differ at bf16 rounding level between torch versions. Cloudflare does not say whether it used fla at all. Two checks inside the image decide whether our stack scores Clef the same way:
1. The S1 **parity** gate compares the decide path with Clef's own `systemone`: max |dp| <= 2e-4.
2. The new S1 **`kernel_path`** check compares the fast fla/causal-conv1d GDN path with transformers' own torch reference (`clef_backend.reference_gdn_path`, which swaps `torch_chunk_gated_delta_rule` and the conv fallback into every GDN layer). It runs on up to 16 parity rows of at most 2,048 tokens. The bar is max |dp| <= 2e-2, with no argmax flip where the top-2 margin exceeds 2e-2. **Non-vacuous (PR #43 B1).** transformers sets `chunk_gated_delta_rule = fla_kernel or torch_chunk_gated_delta_rule` and falls back silently when fla is unavailable. On a stack where that happens, the old check "switched" 3 layers from the reference to the reference, measured max |dp| = 0 and reported `ok=True`. The check now passes only if three things hold:
   - (a) at least one GDN layer's kernel object is **not** already the torch reference (`gdn_layers_switched` > 0; `gdn_layers_found` and `gdn_layers_already_reference` are reported);
   - (b) a real difference was measured (`max_abs_dp > 0`; two different kernels never agree on every bf16 probability);
   - (c) the tolerance and argmax rules above hold.

   `fail_reasons` names every failed condition. A failure exits 11, and job B is gated on `result.s1_ok` (parity **and** `kernel_path`).

`FusedRMSNormGated` (an fla Triton norm; not the gated `chunk_bwd_dqkwg` kernel of #640) stays on in both arms of that check. Clef numbers are reported as "Clef weights under our pinned stack (`kernel_stack`)", next to the card's own numbers, never merged with them.

**Image build pins (review non-blocking items).**
- `ninja` is installed before the `causal-conv1d==1.7.0` source build (its build requirement; without it the build falls back to a serial setuptools compile).
- `build_timeout_minutes` is raised to 90 for that compile.
- The build-time assert now also checks `transformers==5.10.2`, imports `torchvision` (needed by the Qwen3.5 image/video processor classes that `AutoProcessor` resolves), and imports `transformers.AutoProcessor` / `Qwen3_5ForConditionalGeneration`. A missing processor dependency therefore fails the free build, not the paid S1 load.

**Latency deviation (H200 -> H100).** Cloudflare's card reports Index latency on **one H200** (median/p95: Clef-flash 38.8/122.4 ms, Clef 209.3/238.6 ms). The board's own admission measurement is on one RTX PRO 6000. Our latency row is on **one H100 80 GB SXM**, which has less HBM bandwidth than the H200 (3.35 vs 4.8 TB/s), with `clock_lock: true`, batch 1, our pinned `kernel_stack` and DEV rows. It is therefore comparable to neither number. It is reported as "H100, our stack", next to the card's H200 figures, and is never used for a latency "better than" claim against the card or the board.

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
| **B-flash** | clef-flash | S0-S4b-S5 (GCG, resumable) | `depends_on: {job: A-flash, require: "result.s1_ok == true"}` (parity **and** non-vacuous `kernel_path`) | 600 min | ~3.3 h, **~$10** | $29.70 |
| **C-27b** | clef | S0-S5 as A | `depends_on: {job: <I-subset-27b id>, require: "result.s1_ok == true"}` (was A-flash `result.ok`, which job-sgihj never wrote: C could not launch as specced). For the Index, C is superseded by I-subset/I-full (section 4b); C stays only for the 27B safety battery and is not proposed now | 240 min | ~1.8 h, **~$5.4** | $11.88 |
| **I-subset-27b** | clef | S0-S1-S2-S2b-S3-S5 on the 2,008-row m1 DEV subset (section 4b) | (none) | 58 min | **$2.82** | worst $8.71 |
| **I-full-27b** | clef | as I-subset on all 12,104 DEV rows | `depends_on: {job: <I-subset-27b id>, require: "result.s1_ok == true and result.ok == true"}` | 145 min | **$5.69** | worst $17.32 |
| *T-flash / T-27b* | clef-flash / clef | teacher export on the m2 TRAINING draw (section 11) | T-27b on I-subset `s1_ok` | 88 / 258 min | $3.56 / $9.26 | worst $11.68 / $28.51 |
| *D-27b-gcg* | clef | GCG | not planned | - | ~9 h, ~$27 | worst case exceeds the 720-min cap |

**Update 2026-10-06:** A ran (job-sgihj, $1.40). Joint-gate commitments now: B $29.70 + I-subset-27b $8.71 + I-full-27b $17.32 worst case (`costs.branch_ledger`). The original plan follows. **Planned total (A+B+C): about $18 expected, $47.52 ceiling at $2.97/h.** The ceiling is per job: `max_duration_minutes` x rate for one attempt. If the B worst case (GCG ~2x slower than estimated) runs past 600 min, it ends at the cap with its checkpoint. Finishing then needs a second submission under its own spend line. Confirm with `validate_job` that the quoted worst case is per job (review N9) ($57.60 at the $3.60/h H100 list rate implied by the docs' "$0.90 for a 15-minute grace window". The submit response's `rate` is authoritative). This fits inside the $75 joint gate. Without D, the GCG row for Clef-27B stays "not measured", and no GCG sentence may compare against Clef-27B.

How the estimates were built:
- Index DEV, decide path. Cloudflare's card gives median/p95 latency of 38.8/122.4 ms (flash) and 209.3/238.6 ms (Clef) on their hardware. Assuming H100 means of about 80 ms and about 300 ms with kit overhead: 12,104 rows take ~16 min (flash) and ~60 min (27B). The native arm runs only on rows whose encodings differ or that Clef truncates, plus a 64-row parity sample. Every other row has identical ids and therefore identical outputs, so a second full pass would buy nothing.
- safety_eval without GCG: ~2,800 decide variants plus ~1,200 native-encoding variants at ~600 tokens is about 4 min (flash). Bootstrap runs on the CPU.
- GCG-lite at HP_GATE (20 tokens, 150 steps, batch 128, top-k 128): one step is about 131 forward-equivalents x 600 tokens on 9.4 B params, so ~4 s on an H100. The worst case is 10.6 min per attack, or 8 h for 45 attacks. Early stop on success is expected to average ~60 steps, giving ~3.3 h. The cap is 600 min, and the job checkpoints after every attack (`resume: "checkpoint"`).

## 4b. Clef (27B) on the Index: I-subset-27b and I-full-27b (`JOB=I`, `ROWS=subset|full`)

Purpose: measure Clef, which has never been measured, through exactly the harness and readout that measured
Clef-flash in job-sgihj. Every Clef vs Clef-flash vs ours comparison is then **paired on identical DEV rows**. This is
a baseline measurement: nothing is tunable, and only the stop conditions below are pre-stated.

**DEV only.**
- The jobs load `slices/dev-rows.jsonl.gz` (12,104 rows) and score with `--slice dev` / `jobs.common.dev_subset`.
- `SSDM_UNSEAL_TEST` is never set, so the harness refuses TEST.
- The Index board scores all 150,317 rows; our CLI scores only dev/test and renormalises. A full-board run therefore
  includes the sealed TEST slice and is itself a TEST unseal. It needs explicit operator approval and is **not** part
  of these jobs.

**Model pin (verified 2026-10-06 against the Hugging Face API and model card).**

| | Clef | Clef-flash |
|---|---|---|
| Repo @ revision | `Cloudflare/clef` @ `2f3de3dd85f379784083b0814d997ab627200f0c` (= `main` on 2026-10-06; lastModified 2026-10-01) | `Cloudflare/clef-flash` @ `17f0b0ad64efb65d273590632833508766b2aae6` (= `main`) |
| License | **apache-2.0** (card metadata + `LICENSE` file) | apache-2.0 |
| Base | `Qwen/Qwen3.8-27B`, `base_model_relation: finetune` | `Qwen/Qwen3.5-9B` |
| Repo size | 54,989,894,057 B (12 shards) | 19,083,377,402 B (4 shards) |
| Text backbone | hidden 5,120; 64 layers (48 GDN + 16 gated full attention); MLP 17,408 | hidden 4,096; 32 layers (24 + 8); MLP 12,288 |
| Per-token matmul params (`price.py`) | 24.35 B | 6.92 B, so the FLOPs ratio is **3.52x** |

Two notes on the release:
- Cloudflare's blog describes both models as frozen Qwen backbones with a jointly trained routing head and rank-256
  LoRA adapters. The repos ship full merged weights plus `joint_head.safetensors`.
- The blog says "64k context", but the shipped `joint_schema_model.py` defaults `max_length` to 16,384. Both our decide
  path (`exhausted`) and the native arm (silent truncation) use 16,384. 51 DEV rows exceed it (job-sgihj). A 64k arm
  would be a separate, declared measurement.

**GPU shape: one H100 80 GB.** This is a scaled measurement, not an estimate from scratch.
- Weights: 27.36 B params x 2 B = 50.9 GiB, vision encoder included.
- job-sgihj measured the Clef-flash peak at **20.34 GiB** after a 16k-token forward **and** a GCG gradient step. That
  is 17.5 GiB of weights plus ~2.8 GiB transient.
- Scaling the transient by width x depth (1.25 x 2 = 2.5) gives ~7 GiB.
- So the estimated 27B peak is **~58 GiB of 79 GiB usable**.
- `JOB=I` runs no gradient probe, so its peak is lower still. No 2-GPU fallback is needed. S1 records
  `max_memory_allocated_gib` and exits 10 on OOM.

**Pricing (measured, `jobs/clef_baseline/price.py`).**
- **Clef-flash rate from job-sgihj.** 12,104 rows and 18.60 M input tokens took 18.86 min of summed per-row wall time.
  The per-row model is ms = 32.0 + 0.0399 x tokens; with an intercept, the fitted sum equals the measured sum.
  Latency: median 57.1 ms, p95 286.5 ms (H100, our stack).
- **Clef projection.** Clef runs at the FLOPs ratio 3.52x: **66.3 min full DEV, 11.1 min subset**, plus 1.6 min of load.
- **Stress case.** Cloudflare's card gives a second ratio: median 209.3/38.8 = 5.39x and p95 1.95x, on an H200. It is
  reported as the stress case (101.6 / 17.1 min).
- **Caps** are sized at >= 1.3x the FLOPs projection.

Two guards replace that assumption with a measurement before money goes into the long phase:
- **Preflight (S1 -> S2).** The 64 parity rows run through the same decide path on this box. Their measured time over
  the job-sgihj model's time for the same token counts gives the box's speed factor. Speed factor x volume gives the
  projected Index minutes. If that exceeds 95% of the cap, the job exits 18 before S2.
- **Live guard (S2).** `jobs.common.rate_guard run --exclude-startup` checks after 3 min (subset) or 8 min (full). The
  model-load seconds before the first row are not extrapolated. Along job-sgihj's row order, a linear projection made
  after 12% of the phase overshoots the final time by at most 1.32x (subset) or 1.28x (full). So the expected rate
  stays under 95% of the cap: 13 x 1.32 = 17.2 < 19, and 68 x 1.28 = 87 < 95.

| Job | Phases: expected / cap (min) | max_duration | Expected $ | One attempt $ | Worst $ (build 60 + 2 attempts) | Hard cap |
|---|---|---|---|---|---|---|
| I-subset-27b (`job_params_index_subset.json`) | build 20; setup 10/15; probes 8/11; index 13/20; native 3/4; score 2/3; report 1/2; reserve 3 | 58 | **$2.82** | $2.87 | **$8.71** | $9 |
| I-full-27b (`job_params_index_full.json`) | build 20; setup 10/15; probes 10/15; index 68/100; native 4/6; score 2/3; report 1/2; reserve 4 | 145 | **$5.69** | $7.18 | **$17.32** | $18 |

How the two jobs relate:
- The subset job is the canary. The full job `depends_on` its `result.s1_ok == true and result.ok == true`.
- If budget is short, run I-full alone. Its report also scores the 2,008-row subset from the same rows
  (`index_dev_decide_on_m1_subset`), as was done for Clef-flash (55.8 [53.0, 58.5]).
- Chained, the workspace cap is MTD + `costs.concurrent_cap_usd(["clef_index_subset", "clef_index_full"])` = MTD + $27.
- Setup is unmeasured for a 55 GB pull (`hf_cache: false`, review N4). Its cap is 1.5x a 10-min estimate, and a slower
  pull ends at exit 10/18 inside the setup cap.

**Readout (identical to Clef-flash, paired).** The report carries:
- `index_dev_decide`: slice index + cluster-bootstrap CI.
- `index_dev_native_composite`: Clef's own systemone on the native-arm rows.
- `compare_decide_vs_native`.
- `ece_decide`: board protocol, debiased L2, SmoothECE.
- `latency_decide`: median/p95 after 10 warm-up rows.
- The G1 audit and the probes: parity, kernel_path, encodings, G8, G2b.
- `compare_clef_vs_flash` (Clef - Clef-flash), when job-sgihj's decide results ship as
  `data/reference/clef_flash.decide.results.jsonl` (sha256 `463f21ff...d0aa`, via `build_context.sh index` with
  `REF_FLASH=`).

For reference, Clef-flash recomputed locally from job-sgihj's own files with the fixed reader:

| Clef-flash (job-sgihj, recomputed locally) | Value |
|---|---|
| DEV, decide path | 56.79 [55.40, 58.22] |
| DEV, native composite | 56.82 |
| 2,008-row subset | 55.78 [52.99, 58.52] |
| Board-protocol ECE | 0.0285 (n 5,615) |
| Overall SmoothECE | 0.0239 |

**Stop conditions (pre-stated; nothing tunable).**
- Any identity mismatch: exit 10.
- Parity > 2e-4 or kernel_path failure: exit 11. No Index number is reported.
- Preflight or live projection past 95% of a cap, or any phase timeout: exit 18.
- A truncated, malformed or partial results file: exit 17. Nothing is scored from it.
- More than 1% of rows `error`: no number; the reason is written in (section 8 sanity rule).
- No retries beyond the platform's one restart. A re-run needs its own spend line.

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
  "build_timeout_minutes": 90,
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
- **B-flash.** `env {"RELEASE":"clef-flash","JOB":"B"}`, `idempotency_key "clef-baseline-B-flash-<sha12>"`, `max_duration_minutes 600`, `resume "checkpoint"`, `depends_on {"job":"<A-flash id>","require":"result.s1_ok == true"}`. `s1_ok` is true only if the S1 parity gate **and** `kernel_path` passed. GCG backward through the GDN kernels never starts unless the fast path was shown to be in use and to match the reference. Its checks are `file_exists /output/clef-flash-jobB/safety/gcg_attacks.jsonl` plus `result.json`. It uses the same heartbeat health check as A (`interval_s 600`, `fail_threshold 2`, `kill`); the watchdog keeps the heartbeat fresh through long attacks because GCG logs every 10 steps.
- **C-27b.** `env {"RELEASE":"clef","JOB":"A"}`, `model "Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c"`, `max_duration_minutes 240`, `depends_on {"job":"<I-subset-27b id>","require":"result.s1_ok == true"}` (fixed 2026-10-06: the old dependency on A-flash `result.ok` can never be met after job-sgihj), the check paths under `/output/clef-jobA/`, and **`hf_cache: false`** (review N4). The manual guarantees only at least 32 GiB of free cache room and warns that a larger single download "can still fill the cache mid-pull (submit unusually large models without the flag)". The 55 GB of weights therefore download into the job's own scratch: 55 GB of weights plus a ~15 GB image and outputs fit the 100 GiB `job_scratch_bytes`. Preflight's disk warning should report ~55 GB.
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
| 5 | S0 | `jobs/common/fla_guard.py` refused the image (fla #640: triton < 3.7.1, TileLang selected, guard/fla rule disagreement, no CUDA, or the gated fwd+bwd smoke failed); runs before the weights download | fix the image; do not resubmit unchanged |
| 10 | S0/S1 | DEV-row sha256, download, snapshot sha256 (any shard, head, tokenizer, jsm) mismatch, or model load / OOM in the probe | do not retry blindly; for 27B OOM switch to the 2-GPU plan |
| 11 | S1 | parity gate failed (decide-path raw probabilities differ from Cloudflare's `systemone` by > 2e-4, or argmax disagrees off-tie, on the 64-row sample) **or** `kernel_path` failed (fast GDN kernels vs torch reference: max abs dp > 2e-2, an argmax flip at margin > 2e-2, or no GDN layer switched) | mapping or kernel numerics are wrong; nothing downstream is valid |
| 12 | S2 | kit run failed (5 consecutive errors, device assert) or the composite native file could not be built | inspect `index/*.log` |
| 13 | S3 | harness score / compare / ECE failed | rerun locally from `results.jsonl` (CPU) |
| 14 | S4 | safety_eval (non-GCG) failed, **or the PREREG gate refused** (runner exit 32: fewer than 8 D1-B templates) | inspect `safety.log`; land the blind-template amendment |
| 15 | S4b | GCG job failed (the checkpoint keeps finished attacks) | resubmit with the same idempotency key family; resumes |
| 16 | S5 | report failed | recompute locally |
| 17 | S3/S5 | a results file is truncated (no final newline), has an unparseable line or a duplicate `run_id`, or misses rows it must have (decide rows != probes' row count; native-arm rows missing; a slice/subset row with no result). Raised by `harness/jsonl.py` (scoring CLIs exit 3) and `summarize.py composite` (exit 3/4); nothing is scored or written from a partial file | inspect the named file and line; never re-score with `--allow-partial` for a receipt |
| 18 | S1/S2/T1 | too slow: the measured-rate preflight (S1 parity timing, `price.py preflight`) or the live rate guard (`rate_guard run --exclude-startup`) projected the phase past 95% of its cap, or a phase timeout hit | re-price from `env/preflight_index.json` / `env/rate_guard_*.json`; do not resubmit unchanged |
| 19 | T1 | teacher export failed (model load, decide error, row/selection mismatch) | inspect `teacher/`; the export resumes from its own file |
| 20 | T0 | contamination guard refused: a drawn record is an Index DEV/TEST row (id, payload hash or content hash), is in the frozen m2 cohort, carries a held-out D1-B fragment, or `guard_ready` could not verify the pinned content index | nothing was exported and no weights were downloaded; investigate the training set |

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

## 11. Teacher export (`JOB=T`): Clef / Clef-flash per-option probabilities on m2 TRAINING records

This export is for later distillation into the attention-free student. PLAN.md allows open-weight teachers, and both
Clef models are Apache-2.0. It is optional and is not on the joint-gate path.

**Records.**
- **Source.** The PR #44 150k m2 training set, named by `data/manifests/sample-150000-seed20261006.pointer.json`. This
  is the pointer `jobs/m2_first_cell` trains on.
- **Draw.** `jobs/clef_baseline/teacher_scope.json`: 24,000 records, seed 0, by `train.run_cell.split_ids` over the set
  minus the three pinned held-out D1-B exclusions. This is exactly the m2 first cell's training draw.
- **Wider draw.** `--n` (or `CLEF_TEACHER_N`) widens the draw to the 149,997-record pool, priced below.
- DEV, TEST and held-out rows are **never** exported.

**Contamination guard (`baselines/clef_teacher.py select`).** It reuses the train/ guard machinery and runs on the CPU
before the weight download and before any GPU phase. Any hit exits 20 and nothing is exported. Checks:
1. Pointer file hashes: uncompressed sha256 of the records, manifest and content map (exit 10 on mismatch).
2. Manifest ids == content-map ids == `n_records`.
3. `train.cohort` holds none of the set. This is the frozen m2 dev/calibration/retention cohort v2, matched by id
   **and** by normalized content.
4. For every drawn record:
   - its line matches the manifest sha256;
   - its content hash matches the content map;
   - `train.guard.SplitGuard.check` passes. That covers Index metadata, any id or payload hash in the harness DEV
     **or TEST** manifest, and a content-hash match against the pinned content index of every DEV + TEST row
     (24,150 hashes, rebuilt and verified on the box by `jobs.m2_pilot.guard_ready`; its exits 30-32 map to 20).
5. `train.heldout.scan` (held-out D1-B fragments, k=8 templates) finds zero hits in the draw.
6. `Cohort.assert_training_ids` and `assert_training_content` pass.

I ran this locally on 2026-10-06 against the real files, with the production guard and the pinned content index:

| Draw | Result | Time |
|---|---|---|
| 24,000 records | passes | 61 s |
| Whole 149,997 pool | passes | 172 s |

Both runs: cohort hits 0/0, guard refusals 0, held-out hits 0, template set `7a81287e...`.

**Export (`clef_teacher export`).**
- Each record goes through the same `decide()` path as the Index readout: ssdm encoding, no truncation. Labels are
  never read into the request.
- Output is one line per record in `teacher-<release>.jsonl`: `id`, `content_sha256`, `status`, `raw_probabilities`
  {question: {option: p}}, `sequence_tokens`, `wall_ms`.
- Lines are appended and flushed. The export resumes: a partial last line is cut and kept as `.truncated-tail`, and
  finished ids are skipped.
- `teacher-<release>.identity.json` records the model and backend revisions, the selection's drawn-ids sha256, status
  counts and the file's sha256.
- About 550 of the 24,000 records (~3.4k of the 150k) exceed 16,384 tokens. They come out `exhausted`, with no
  probabilities, and are counted.

**Pricing.** The job-sgihj per-row model is applied to the draw's measured volume: 24,000 records with 45.2 M tokens
(proxy x 1.381), and 279.6 M tokens for the 150k pool. Clef runs at 3.52x Clef-flash.

| Job | Phases: expected / cap (min) | max_duration | Expected $ | Worst $ | Hard cap |
|---|---|---|---|---|---|
| T-flash, 24k draw (`job_params_teacher_clef_flash.json`) | build 20; guard 3/6; setup 4/10; export 44/66; report 1/2; reserve 4 | 88 | **$3.56** | **$11.68** | $12 |
| T-27b, 24k draw (`job_params_teacher_clef.json`) | build 20; guard 3/6; setup 10/15; export 153/230; report 1/2; reserve 5 | 258 | **$9.26** | **$28.51** | $29 |
| T-flash, whole 149,997 pool (not proposed; `--n 149997`) | export 266.6 expected | ~430 | ~$14.70 | ~$47 | - |
| T-27b, whole pool | export ~937 min > the 720-min job cap | - | ~$48 | not feasible at batch 1 | - |

Batching (`collate_records`) is the obvious lever for the whole pool. It is unmeasured, so it is not priced. T-27b
`depends_on` I-subset-27b `s1_ok`, so the 27B decide path is first shown to match systemone on this stack.

## 12. job-sgihj composite crash: root cause and fix (LESSONS L-15)

I reproduced this from the job's own output: artifact `art-8y6d8`, sha256 `eab45f29...64cb`.

**Symptom.**
- `summarize.py composite` raised `JSONDecodeError: Unterminated string ... line 1 column 1399`.
- The runner mapped this to exit 12 at 17:33:23Z, after S2b.
- S3 scoring, **S4 safety** and S5 never ran. The Clef-flash numbers in LEDGER were scored off-box afterwards.

**Root cause: not a truncated file.**
- Both inputs were complete. `index/decide/results.jsonl` has 12,104 rows and a final newline;
  `native/results.jsonl` has 115 rows.
- Row `25:GPQA-Diamond:GPQA-Diamond:test:4` carries a raw **U+2028** (LINE SEPARATOR) inside a JSON string, because
  the kit writes `ensure_ascii=False`.
- `rows()` split the file with `str.splitlines()`. That method also breaks on U+2028/U+2029, U+0085, \v, \f and
  \x1c-\x1e.
- So 12,104 records became 12,105 "lines", and one record was cut mid-string. The crash came from reading the
  **decide** file, not the native one.

A second, opposite failure was latent in the scoring path. The kit's `load_results` stops silently at a partial last
line. A truncated or partial results file would therefore have scored its missing rows as unanswered, with no error.

**Fix: strict reader in `harness/jsonl.py`.** It is used by `summarize.py`, `harness score|compare|ece`,
`jobs.common.dev_subset` and the GCG resume.
- Lines split on `\n` only.
- A missing final newline raises `truncated_tail`.
- An unparseable line raises `malformed`, with path:line.
- A duplicate `run_id` raises `duplicate`.
- A slice or subset row with no result raises `incomplete`. `--allow-partial` opts out explicitly and is never used
  for a receipt.
- Every CLI exits **3** on these errors.

**Fix: fail-closed composite.**
- Decide rows must equal the probes' row count.
- Every native-arm id must be present.
- Nothing is written unless all checks pass (exit 3 or 4).

**Fix: runner.**
- `run_job.sh` maps exits 3 and 4 to **exit 17**.
- It now scores the decide arm *before* building the composite, so the primary number exists even if the composite
  is refused.

The same U+2028 hazard is removed from `train/cohort.py` and `train/guard.py` (the manifest readers).

**Tests.**
- `tests/clef/test_results_strict.py` covers the U+2028 reproduction, the other separators, truncated, malformed,
  duplicate and incomplete files, composite exits 3 and 4, and `drop_truncated_tail`.
- `tests/clef/test_clef_job_sh.py` executes `run_job.sh` end to end on CPU. The stand-in subset includes the GPQA
  U+2028 row and a native-arm row, and a fault-injected truncated decide file must end in exit 17.
