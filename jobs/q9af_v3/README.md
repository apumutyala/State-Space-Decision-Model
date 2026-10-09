# Q9-AF v3: pre-registered plan and gate

**Takeaway.** v3 converts the same 8 attention layers of Qwen3.5-9B to Gated DeltaNet as v2 and ends just as
attention-free. Four things change, all taken from 2025–26 distillation recipes:
1. **Per-layer gated conversion.** Layers switch two at a time, most sensitive first, with a keep-best checkpoint at
   each stage.
2. **Distillation on teacher-generated text.** Completions sampled from the stock open-weight Qwen3.5-9B, trained with
   a completion-only loss.
3. **A 32k long-context stage.**
4. **A short on-policy stage.** The student samples and the teacher scores every token (reverse KL).

v3 is promoted over v2 only if it beats v2 on the proxy600 Index under the same readout, with a CI lower bound above 0.
v3 runs in parallel with v2's verdict, as two chained jobs:
- **job A** (581 min): the data, stage 1, the probe and the 4 gated stages;
- **job B** (219 min): long context, on-policy, and the gate.

Committed before any v3 run or data. Code: `convert/v3.py`, `convert/distill.py`, `convert/gendata.py`. Rule:
`jobs/q9af_v3/rule.py` (`q9af-v3-rule-v1`, constants pinned by `tests/test_q9af_v3.py`). Config:
`configs/q9af_v3_h100.json`. Phases and pricing: `JOB_SPEC.md`.

## What changes vs v2 (`stack/q9af/v2-clean-rerun`, job-7ji99)

| Item | v2 | v3 |
|---|---|---|
| Layers | all 8 jointly from step 0 | **4 stages of 2**, in a probe-chosen order; each stage trains ALL layers converted so far; **keep-best** checkpoint per stage (held-out KL), restored before the next stage |
| Init / stage 1 | reuse_attn, 70 min MOHAWK | reuse_attn (pre-registered, no init_cmp), **40 min** MOHAWK, all 8 at once (teacher-driven, so order-free) |
| KD data | generic windows (full-sequence KL) | generic windows **+ 25 % of batches teacher-generated rows** (completion-only KL + 0.25 × CE on the teacher's tokens), each row read once |
| Long context | trained at 4,096 | **+45 min at 32,768 tokens** on 512 reserved 32k blocks of the same mix (never read in job A) |
| On-policy | none | **+40 min**: student rollouts (T = 1, ≤ 256 new tokens) on 12,000 held-apart prompts; per-token reverse KL to the teacher |
| Gate | held-out KL ≤ 0.25 + stock-readout proxy600 vs stock | **proxy600 under the generic v3w readout (PR #236), paired v3 − v2** (or v3 − stock); KL reported, never gating |
| Data / decontamination | v2 clean mix, `mix.decontam` full-eval-v1 | **identical** (same `convert.mixdata` build, sources, revisions, shares, held-out); `train.attack_guard` + `train.doc_overlap_guard` unchanged |

No TEST, held-out, cohort-v3 or Index row is ever trained on. Prompts are prefixes of documents from the decontaminated
mix. The only teacher is the open-weight stock Qwen3.5-9B: no Clef, Jev or API outputs.

## Stage plan (minutes are wall-clock caps; `align.Guard` stops: rate floor → exit 3, memory → exit 4)

| Job | Phase | Cap | What |
|---|---|---|---|
| A | setup + weights | 8 + 10 | pins, fla guard, preflight, shipped-data sha256, `sf_fetch --stage q9af_v2` |
| A | D | 30 | `convert.mixdata build`: v2's clean mix (280M tokens), full-eval-v1, window manifest |
| A | G | 85 | 32k block split (512 blocks → `lc.u32`), teacher generation (**75 min** or 16M completion tokens), on-policy prompts |
| A | P | 425 | stage 1 **40** → probe (8 single-layer evals) → gated stages, **360 min** total KD: stage k = 45 + 180 × its KL share |
| A | X | 8 | `stageA/gdn_converted.pt` (verified export) + the members job B pins |
| A | reserve | 15 | |
| B | setup + weights | 8 + 10 | + stage-A pin present, v3w readout identity `2e4907ee…`, proxy600 rows |
| B | F | 15 | stream + sha256-check the stage-A artifact (and v2's export if pinned) |
| B | P | 110 | 32k long context **45** → on-policy **40** → final held-out KL (4k / 32k / generated completions) + throughput |
| B | X | 8 | final `gdn_converted.pt` (same format v2 / q9af_cell consume) |
| B | L arms | 15 × 3 | proxy600, family `q9af-probe-v3w`: stock, v3, v2 (v2 only if pinned) |
| B | S | 8 | `jobs.common.dev_subset compare/score` (the frozen `harness/score.py` + bootstrap) |
| B | reserve | 15 | |

Total distillation: 40 + 360 + 45 + 40 = 485 min, against v2's 70 + 385 = 455.

## The per-layer order rule (`convert.distill.conversion_plan`, pre-registered)

1. After stage 1 (all 8 GDNs aligned layer-wise from reuse_attn), measure for each attention layer i its
   **sensitivity** s_i: the held-out KL(teacher ‖ model) with **only layer i** converted. These are the curve windows,
   8 per source (web / chat / math) × 4,096 tokens.
2. **Order = descending s_i** (most sensitive first; ties go to the lower layer index). **Groups = consecutive pairs**:
   4 stages, and stage k converts its 2 layers while training all layers converted so far.
3. **Stage minutes = 45 + 180 × S_k / ΣS**, where S_k is the summed s_i of the layers stage k adds.

Why most-sensitive first: full8 (job-2xe5w) converted its hardest layers last (L27, L3), with the fewest tokens, and
lost them. Here the hardest layers are converted while the rest of the network is still the exact teacher. They then
keep training in every later stage and get the largest stage budgets. The easiest layers, which barely move KL, come
last. The order is a deterministic function of the probe and is recorded in `results.json` (`plan`).

## Teacher-generated data (GenDistill-style, phase G)

- **Prompts.** 60,000 candidate prompts (chat 60 %, web 25 %, math 15 %) from documents outside the reserved 32k
  blocks.
  - chat: the conversation up to the first `<|im_start|>assistant\n<think>\n\n</think>` (the readout's format), ≤ 1,024
    tokens;
  - web / math: the first 256 tokens of a doc that is at least 320 tokens long.
- **Sampling.** The stock Qwen3.5-9B with all attention layers, T = 1.0, ≤ 512 new tokens, stopping at
  `<|im_end|>` / `<|endoftext|>`. Batch 192, one cached prefill, then one-token steps on the fp32 recurrent GDN kernel.
- **Size and time.** Capped at **75 min or 16M completion tokens**, whichever comes first. At the estimated
  ~3,000 completion tok/s (no receipt; phase G records the measured rate), that is **~13M completion tokens (~35k
  samples)**. Packed into 4,096-token rows that hold the prompt and the completion, it comes to ~6–7k rows (~26M tokens).
- **Held out.** 32 rows score the completion-only KL.
- **On-policy prompts.** A further 12,000 prompts from disjoint docs, saved for job B.
- **Loss on these rows.** Completion positions only: KL(teacher ‖ student) + 0.25 × CE on the teacher's sampled
  token. Prompt tokens carry no loss.

## THE GATE (binding; `jobs/q9af_v3/rule.py`, rule `q9af-v3-rule-v1`)

**v3 is PROMOTED over v2 iff the paired proxy600 Index difference (v3 − reference) has a CI lower bound > 0.
Otherwise the verdict is REPORT.**

- **Readout.** The generic v3w LM-logit readout (PR #236: `decide.v3w_generic.make_v3w("qwen3.5-9b")` +
  `V3WGenericBackend`, family `q9af-probe-v3w`, readout identity `2e4907ee…`). Every arm runs through the same code path
  in job B.
- **Scoring.** proxy600 (ids `7d42c398…`), edition 0.2.1, scored by the frozen `harness/score.py` through
  `jobs.common.dev_subset compare`. Bootstrap B = 2000, seed 0.
- **Reference.** v2's converted model (job-7ji99's `gdn_converted.pt`, an all-8 attention-free export) when
  `job_params_B.json` pins its artifact. Otherwise it is the stock Qwen3.5-9B under the same readout. The committed
  params fix the reference before the run (`rule.expected_reference`); the reference is never chosen afterwards.
- **Reported, not gating.**
  - v3 − stock, and v2 − stock when v2 is present;
  - the held-out KL: 4k mix, 32k, and teacher-generated completions, at entry and final;
  - every stage's KL curve, its best checkpoint and its tokens;
  - the probe sensitivities and the order.
- **NO_MEASUREMENT** (exit 9): the export is not the 8-layer attention-free final; a compare is not on proxy600 or not
  B 2000 / seed 0; the reference does not match the pre-registered one; the binding numbers are missing.

| Verdict | Action |
|---|---|
| PROMOTE | v3 replaces v2: pin v3's `gdn_converted.pt` for `jobs/q9af_cell` (`jobs.q9af_cell.pin fill`) in its own commit |
| REPORT | v3 is not promoted. Report the deltas and curves. The reference stays the Q9-AF input |

## Chaining A → B (sha256-pinned, nothing trusted from the artifact)
1. Job A ends with `/output/stageA/*` (12 members; `MEMBERS.sha256`).
2. Download that artifact tar, then run `python -m jobs.q9af_v3.artifact fill --params jobs/q9af_v3/job_params_B.json
   --which stageA --tar <tar> --artifact art-… --job job-…`. If job-7ji99 exported an attention-free model, do the
   same with `--which v2`.
3. Commit, push, and re-package B. Export `STAGEA_URL` (and `V2_URL`) in the submitting shell, and `sf_submit` fills
   the placeholders.
4. Job B streams each tar and checks the whole-tar sha256 and every member's sha256 before using it
   (`jobs.q9af_v3.artifact fetch`).
