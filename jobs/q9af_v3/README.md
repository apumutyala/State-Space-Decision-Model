# Q9-AF v3: pre-registered plan and gates (recall redesign)

**Takeaway.** v2 (job-7ji99) shows that a good held-out KL can still hide a broken model.

| v2 measurement | Result |
|---|---|
| KL | passed (0.097, hidden cos 0.951, wikitext ppl ×1.08) |
| proxy600 | 35.82 → 14.05 (Δ −21.77 [−25.87, −16.37]) |
| retrieval / knowledge / tools | 36.9 → 8.1 / 23.3 → 7.6 / 49.7 → 24.5 |
| MQAR / NIAH drop | 0.59 / 0.525 |
| distillation tokens | 196M |

So v3 attacks recall directly:
1. **More recall state.** The GDN replacing each attention layer gets **2× the key head dim** (dk 256 = the attention
   head dim). That gives an exact q/k init and a 2× state.
2. **A recall-heavy stream.** 40 % synthetic associative-recall / needle / quote rows, and 25 % teacher generations
   that include retrieval, tool and knowledge instructions on mix documents.
3. **Recall-gated conversion.** Layers are converted in a recall-sensitivity order with the hardest last. Every stage
   is gated on MQAR + NIAH + proxy600, with a pre-registered STOP rule.
4. **A recall-bound final gate.** proxy600 + recall decide, never KL.

A **cheap first job S (≤ $22.09)** answers the go/no-go question before any big spend: can any single attention
layer be converted with a small recall loss under this mix and capacity?

Committed before any v3 run or data.
- Code: `convert/v3.py`, `convert/recall.py`, `convert/distill.py`, `convert/gendata.py`, `convert/model.py`
  (`init_gdn(dk=)`, `attach_from_state`).
- Rules: `jobs/q9af_v3/rule.py` (`q9af-v3-rule-v1`, `q9af-v3-sens-v1`; constants pinned by `tests/test_q9af_v3.py`).
- Configs: `configs/q9af_v3_h100.json`, `configs/q9af_v3_sens_h100.json`.
- Phases and money: `JOB_SPEC.md`.

## The chain (each job ≤ 900 min, one H100, one attempt; dry-run only, nothing submitted)

| Job | Mode | Max min | Worst ($0.084/min + 60-min build) | Platform hold ($0.063/min) | Expected | Submit when |
|---|---|---|---|---|---|---|
| S | sensitivity table | 203 | $22.09 | $12.79 | $14.78 | now (the cheap first job) |
| A | recall-gated conversion | 863 | $77.53 | $54.37 | $65.10 | S says GO (operator call on PARTIAL) |
| B | 32k + on-policy + final gate | 219 | $23.44 | $13.80 | $13.19 | A ends STAGE_A_DONE (artifact pinned) |

## 1. Capacity: a widened GDN (`gdn.dk`, converted layers only)

| | v2 (config) | v3 |
|---|---|---|
| key heads × key dim × value heads × value dim | 16 × 128 × 32 × 128 | 16 × **256** × 32 × 128 |
| recall state per sequence per layer (fp32) | 2 MiB | **4 MiB** |
| q/k init from the attention projections | per-head SVD 256 → 128 (lossy) | **exact** (dk = the attention head dim) |
| params per converted layer | 67.2M | 84.0M (**+16.8M**: in_proj_qkv +4,096 rows, conv +16k taps) |
| total over 8 layers | | **+134M (+1.5 % of 9B; +0.27 GB bf16)** |

- Native GDN layers are unchanged.
- Exports record `gdn_dk` per layer and pass a strict load into HF `Qwen3_5GatedDeltaNet` with that key dim.
  `convert.readout` sizes the GDN from the weights.
- `jobs/q9af_cell` / `heads.backbone` assume the config's dk. They must read `gdn_dk` before using a widened export.
- The alternative of a bigger state through more heads (nk 32) would need a new value/key head mapping and loses the
  exact init, so dk is the first lever. Job S tests whether it is enough: its base-width control re-runs the 2 worst
  layers at dk 128.

## 2. The recall-heavy stream (`mix_fracs`; smooth weighted round robin, `convert.recall.MultiStream`)

| Source | Share | Loss | What |
|---|---|---|---|
| generic windows | 35 % | full-sequence KL(teacher ‖ student) | v2's clean mix, verbatim (full-eval-v1) |
| teacher-generated rows | 25 % | completion-only KL + 0.25 × CE | stock Qwen3.5-9B (open weights), T = 1. Prompts: chat 35 %, web 10 %, math 10 %, **retrieval 25 %, tool 10 %, knowledge 10 %** (fixed instructions on ≤ 600-token mix documents) |
| synthetic recall rows | 40 % | answer-only KL + 0.25 × CE | `convert.recall.RecallSynth` |

The synthetic recall tasks are:
- **kv:** 16–256 key → id records, then 32 lookups;
- **needle:** 1–4 memos at random depths of real text, then a question per memo;
- **quote:** "continue the quote" from 6 tokens of the text.

The data is decontaminated:
- Every token of text is the full-eval-v1 decontaminated mix.
- The instruction templates are checked against the shipped eval-hash index + attack templates at run time; phase G
  refuses to start if any is flagged.
- The recall **probe** (`probes.core` MQAR / NIAH, v2's exact settings) shares none of the training templates, key
  words, value ranges or filler, so the gate stays held out in format.
- No eval item is used anywhere.

## 3. Per-layer order and the STOP rule (job A; `conversion_plan(descending=False)`, `rule.stage_gate`)

1. Stage 1 aligns all 8 widened GDNs (40 min, MOHAWK, teacher-driven).
2. For each layer alone, measure the **recall damage** (MQAR drop + NIAH drop vs stock) and the held-out KL.
3. **Order = ascending damage** (ties: KL, then index). The recall-hardest layers convert **last**. Groups of 2 give
   4 stages; stage minutes = 60 + 280 × the damage share of the layers the stage adds (520 KD min in total). Every
   stage trains all converted layers, with a keep-best (held-out KL) checkpoint.
4. **After every stage**, the runner measures:
   - MQAR + NIAH;
   - proxy600 (v3w readout) of the converted-so-far model, paired vs stock.

   **CONTINUE** iff all three conditions hold:

   | Condition | Bound |
   |---|---|
   | the stage's own MQAR drop **and** NIAH drop (vs the previous stage) | **≤ X = 0.05** each |
   | the cumulative drops vs stock | **≤ 0.20** each (v2 lost 0.59 / 0.525) |
   | the proxy600 point vs stock | **≥ −10** (v2's ADVANCE bound) |

   Otherwise the verdict is **STOP**:
   - no further layer is converted and no further GPU time goes to this conversion;
   - the breaching layers are named and the hybrid is exported only as `gdn_salvage.pt`;
   - the record is STOPPED, and nothing chains to job B.

## 4. The cheap first job S (≤ $22.09 worst, ~$15 expected): does ANY single layer convert with a small recall loss?

**Steps.**
1. A 60M-token clean mix (same recipe).
2. Stock MQAR / NIAH.
3. Stage 1 of all 8 widened GDNs (15 min).
4. **Each layer alone:** 10 min KD on 50 % generic + 50 % synthetic recall, then MQAR + NIAH + KL.
5. **Capacity control:** the 2 worst layers again at v2's width (dk 128).

**Pre-registered rule** (`rule.sensitivity`): a layer **passes** iff its MQAR drop and its NIAH drop are each
**≤ 0.05**.

| Verdict | When | Action |
|---|---|---|
| GO | ≥ 6 of 8 pass | job A |
| PARTIAL | 1–5 pass | decision-needed: A's gate will likely stop before the failing layers, so fund A only for a hybrid or raise capacity first |
| NO_GO | 0 pass | no big spend at this capacity / mix; next lever: dk 512 / more heads, or another mixer |

The output is the full per-layer table: drops, KL, tokens, base vs wide.

## 5. The final gate (job B; `rule.decide`, binding)

**v3 is PROMOTED over v2 iff both hold:**
1. the paired proxy600 Index difference (v3 − reference) has a **CI lower bound > 0**;
2. v3's final **MQAR and NIAH drops vs stock are each ≤ 0.20**.

Otherwise the verdict is REPORT.

- **Readout.** The generic v3w LM-logit readout (PR #236, family `q9af-probe-v3w`, identity `2e4907ee…`) on every
  arm.
- **Scoring.** proxy600 (`7d42c398…`), edition 0.2.1, the frozen `harness/score.py` via `jobs.common.dev_subset
  compare`. Bootstrap B = 2000, seed 0.
- **Reference.** v2's export when `job_params_B.json` pins it, else stock. The committed params fix it
  (`rule.expected_reference`).
- **Reported, never gating.** KL (4k mix, 32k, generated completions), v3 − stock, v2 − stock, and every stage curve.
- **Before the gate, job B runs:** 32k long context (45 min, 512 reserved blocks never read in A) and on-policy
  reverse KL (40 min, 12,000 held-apart prompts).

## 6. Token budget: what one or eight H100s reach, and the ≥ 1B-token plan

| Setup | Rate | Tokens/hour | Basis |
|---|---|---|---|
| 1 H100, stage 1 (alignment) | 18,297 tok/s | 66M | measured (job-5cmat) |
| 1 H100, all-8 joint KD, first student layer 3 | 5,210 tok/s | 18.8M | measured (job-2xe5w) |
| 1 H100, KD with the first student at layer 23 | 9,460 tok/s | 34M | measured (job-2xe5w) |
| 1 H100, all-8 joint KD, widened GDN | ~5,000 tok/s | ~18M | estimate: +0.4 % FLOPs, 2× state work in 8 of 32 mixers |
| 8 H100 DDP | ~34,000 tok/s | ~120M | estimate at 0.85 scaling. **Not implemented** (`convert.v3` is single-GPU); no receipt |

**Plan to ≥ 1B tokens on 1 H100.** After S → GO:
- **A:** ~245M tokens (stage 1 44M + KD ~200M).
- **C1, C2:** continuation jobs of 880 min each: all-8 joint KD on the same stream, ~255M tokens each, ~$79 worst each.
- **C3:** if the curve still falls.
- **B:** ~25M + the gate.

That makes ~0.78B tokens with A + C1 + C2, or ~1.03B with C3. Worst case: $77.53 + 2 × $79 + $23.44 ≈ $259 (≈ $338
with C3). Expected ≈ $230.

Opal-scale 3B tokens on one H100 would be ~170 GPU-hours (~$860 at list): outside the $525 envelope. On 8 H100 it is
25 h wall-clock at the same dollars, so only 1B is in budget.

**Not built yet.** These are pre-conditions for C, deliberately gated on S and A:
- a `convert.v3 cont` mode (load the pinned artifact, `MultiStream.restore` the stream state, rebuild the mix
  deterministically in-job);
- generic-text headroom: the 263M-token short region covers ~1.0B total at 35 % generic with ~12 % re-read. Raise
  `mix.train_tokens` to 400M before C2.
