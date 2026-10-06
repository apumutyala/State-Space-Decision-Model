# Teacher plan for m2 soft labels (Track D)

Status: **plan only. $0 spent.** No teacher has been run. Every number below is an estimate and is labelled as one. A ≤$20 qualification run, behind a `spend` issue acknowledged by Codex, replaces the estimates before any capped run (COORDINATION spend ladder).

## 1. Which teachers are allowed

Only open-weight teachers. Jev (TypeSafe MCA §2.3(b)) and frontier-API terms forbid distillation from their outputs (`reviews/01`, `GOAL.md`).

| Teacher | Weights / licence | Interface | Knowledge cutoff |
|---|---|---|---|
| **Clef** (`Cloudflare/clef`, ~27B hybrid backbone + 128M joint schema head; 12 shards ≈ 55 GB) | Apache-2.0 | Native SystemOne. One prefill pass, then every option of every question scored jointly, so we get a full distribution per question. | **Unrecorded. Must be read from the card/blog before use.** The release is post-2026-09; the backbone pretraining cutoff is undisclosed. |
| **clef-flash** (`Cloudflare/clef-flash`) | Apache-2.0 | Same head family, smaller backbone (m0 parity target) | Same as above: record before use |
| **Qwen3.8-27B** (base of AutoJev / pplx-decider) | Apache-2.0 (verify on card) | Generic LM. Option scoring by letter log-probs (see §2) | Record from card before use |
| **Gemma-4** (size TBD; ≤27B so it fits one 96 GB card in bf16) | Gemma terms (**verify that distillation into a model with different terms is permitted, and propagate the use restrictions**) | Generic LM, option scoring | Record from card before use |

**ForecastBench rule (binding).** The ForecastBench rows resolved 2026-07-01 to 2026-09-18 (INDEX_NOTES §9.3).
- Any teacher whose knowledge cutoff is unknown or later than a question's forecast date must not label ForecastBench-format items whose events resolve after that cutoff.
- Teacher labels on *any* forecasting-style record are restricted to questions that resolved before 2026-07-01.
- Clef was released after the resolution window, so it is treated as possibly leaky for forecasting until its cutoff is documented.
- The cutoff of every teacher and of the student's parent goes into the m2 receipt.

## 2. How soft labels are generated (rented GPU)

1. **Inputs.** Records from `build.py` after the contamination guard. Teachers never see Index rows, and the student never sees teacher outputs on Index rows.
2. **Clef / clef-flash.**
   - Run the native SystemOne request (`state` + `questions`).
   - Keep the full per-option `probabilities` and also the pre-softmax logits (the head exposes one logit per allowed option).
   - Use prefill-only batching; no generation.
3. **Generic LMs (Qwen3.8-27B, Gemma-4).**
   - Render the record (the same renderer as `decide/prompt.py`) with options labelled A/B/… and read the next-token logits over the option letters.
   - Average over **K = 2–4 option permutations**, mapping the logits back to option ids. This removes letter/position bias (PriDe; techniques/06 §6.7).
   - Records with more than 26 options (CLINC 151, retrieval up to 255) use digit ids, or a two-stage shortlist. A shortlist is only allowed for teacher labelling, never on submitted runs.
4. **Calibrate first (temperature scaling on held-out labels).**
   - Hold out 10% of each source's train items as a teacher-calibration split (disjoint by source item from the distillation set; seeded hash split).
   - Fit one temperature T per teacher × family (and × option-count bin) by NLL on the gold labels.
   - Report ECE (debiased L2 and SmoothECE, techniques/06) before and after.
   - A teacher whose calibrated accuracy on a family is below the gold-label baseline is not used for that family.
5. **Targets.**
   - `p_target = α·onehot(gold) + (1−α)·softmax(z/T)` where gold exists (α tuned on DEV-proxy slices of *synthetic* schemas, never on Index DEV).
   - Where the teacher's argmax disagrees with gold, keep the one-hot target and log the disagreement rate per source; it doubles as a label-noise audit.
   - Fact-change pairs and injected records are scored separately. A teacher that flips under injection (steer) is not used as a target on injected records.
6. **Receipts.** Teacher id + revision sha, cutoff, dtype, batch size, tokens processed, $ billed, per-family T, ECE before/after, agreement with gold, and the sha256 of the soft-label file (stored outside git).

## 3. Cost estimate (SF Compute batch rate $2.97/h; one GPU per experiment)

**Assumptions (estimates, not measurements).**
- Mean record length is ~600 tokens. The sample build reports a word-length histogram (`build-summary.json`); tokens ≈ 1.3 × words. The 20k sample is mostly under 256 words, with a long tail from retrieval/tool records.
- 27B dense prefill costs about 2 × 27e9 = 54 GFLOP/token. At ~35% MFU on one H100-class / RTX PRO 6000 card (~350 bf16 TFLOP/s effective at best; likely lower on the PRO 6000), that is about 2,000–3,000 tokens/s.

| Job | Tokens | GPU-hours (est.) | $ at $2.97/h |
|---|---|---|---|
| Qualification: one teacher on 2k records, both calibration splits | ~1.2M | ~0.2 h + ~0.5 h setup/download | **≤ $5** (rung 2, ≤ $20 cap) |
| Clef (27B) on 200k records, single pass (joint head, no permutations) | ~120M | 11–17 h | **$33–50** |
| clef-flash on 200k records | ~120M | ~4–8 h (smaller backbone; measure) | **$12–24** |
| Qwen3.8-27B on 200k records × K=2 permutations | ~240M | 22–33 h | **$65–100** |
| Gemma-4 (≤27B) on 200k records × K=2 | ~240M | 22–33 h | **$65–100** |
| Temperature fits / ECE | CPU | 0 | $0 |

- A sensible first capped run is clef-flash + Qwen3.8-27B on ~200k records: **≈ $80–125**.
- All four teachers: **≈ $175–275**. That fits the $250 System 1 line only if the m1 conversion spend leaves room, so it needs a `spend` issue.
- Download time is billed: 55 GB of Clef weights is ~0.3 h at typical node bandwidth (estimate).
- Stop conditions:
  - throughput below 50% of the estimate after the qualification run → re-plan;
  - calibrated teacher accuracy below gold baseline on >50% of families → fold that teacher.

## 4. What this plan does not do

- No teacher outputs on Index rows (DEV or TEST), ever.
- No Jev, GLiDE, Liquid, Drex or frontier-API outputs.
- No soft labels for ForecastBench-format items from teachers with unknown cutoffs.
