# State-Space Decision Model: technical report (2026-10-12)

For the operator and potential funders. This report covers what the project built, what it measured, and what it did not achieve. Every number cites the branch and commit sha of its receipt. No new measurement was made for this report, and no money was spent. The numbers were re-read from the receipts named next to them, mainly the evidence lane's target-by-target report (claude/ev-evidence-report-v2 @4fa4ecf4, `docs/EVIDENCE_REPORT_V2.md`; body written at @e4d4913f) and the shared ledger (`results/ledger.tsv` on `state` @99e841e6).

## Summary first

- **The headline performance targets were not met.** No model we trained or served beats H2O-Lightning-4B v1.1 on the paired local JevBench Composite or Capability. Our best attention-free model (D19) loses to both Clef models on the full-DEV Decision Index. Every one of these comparisons has a 95% CI below zero, except our newest cell (JEV3-LM-S), which is flat against its own stock control and still below H2O.
- **The guarantees are the strongest result.** Exact reorder and relabel invariance is proven by construction, reviewed independently, and measured bitwise on two trained GPU models (0 flips on 565 perturbed pairs, max |Δp| = 0, while the stock control flips 28/21/12). Trusted-interval invariance (G15) holds on CPU under the real Qwen BPE tokenizer, 1000/1000 cases. Conformal coverage holds on our own trained stack. GDN state contraction is proven, and a contractive-by-construction RWKV-7 variant is built and CPU-tested.
- **One guarantee result is negative for everyone.** Under a cluster-robust test, no model certifies abstention (acting only when error is at most 5%) on DEV. That includes both Clef models and our own cells.
- **The main diagnosed blocker is transfer, not capacity.** Our leak-free training data raises held-out accuracy but does not move public JevBench accuracy. The slice-to-public transfer correlations are +0.15, −0.32 and −0.50.
- **Budget.** $785 was granted. The balance is $11.67 after the last cell (ledger row job-zbmhc, 2026-10-12T23:40Z). Section 6 lists what each open item costs.

---

## 1. Goal and benchmarks

**Objective** (operator, 2026-10-09; `state` BOARD.md). In order:
1. Beat the JevBench v1.6.1 open-weights leader on the official 4-axis Composite.
2. Then beat it on Capability.
3. Then beat MD1's reported numbers.
4. Also beat Clef-flash and Clef 27B on the Decision Index.

All of this on a state-space hybrid main line plus a strict attention-free research line, with structure that controls learning under stated guarantees. A later directive (2026-10-11 ~19:00Z, BOARD.md) made third-party models display-only. Since then, new cells are our own stacks only.

| Benchmark | Target | How we measure it |
|---|---|---|
| JevBench Composite (I/C/S/K: Intelligence, Calibration, Speed, Cost) | H2O-Lightning-4B v1.1: 72.5 on the board, **76.6 Composite A locally** | Paired Δ vs H2O on the 231 open public items, stratified bootstrap B = 1000, seed 16 (EV-PREREG-COMPOSITE-LOCAL). Scorecard: claude/ev-scorecard @e2fcd82b |
| JevBench Capability | Quyet-1.0-Large 81.7. It cannot be downloaded, so the target becomes H2O, **80.5 [72.6, 84.1] locally** | same |
| MD1 (reported only) | 83.5% accuracy, calibration 92.2, p50 85 ms, 1.3% flips | Only the flip rate is partly comparable (§4) |
| Decision Index 0.2.1, full DEV (12,104 rows) | Clef-flash **57.29**, Clef 27B **61.89** | Claim rule B-PREREG-CLAIM: paired CI > 0 vs each Clef (`state` PREREG.md) |

**Kit parity.** Our JevBench numbers count only because our local kit (jevkit) reproduces published rows. It reproduces three published open-weights systems on their author-published public-231 cards: H2O 205/205, decider-12b-v1 200/200, decider-12b v2 197/198. torchcast-12b scores 198 vs 203. That FAIL is explained: on 5 items the option letter is missing from the raw top-20 logprobs (claude/ev-kit-parity @5d674b2d, jobs zbtbj and r3uc4).

**Caveat on every JevBench number.** These are the 231 open items only. The board also uses 1,200 sealed items. On H2O the open-to-board Intelligence gap is −17.6 (77.6 locally vs 60.0 on the board), so no local number is ever compared with the board's 72.5. A board submission would be the only way to get a board-scale number, and the operator declined it (EVIDENCE_REPORT_V2 addendum).

---

## 2. Architecture and methodology

### 2.1 Main line: state-space hybrid (Qwen3.5-4B, GDN-hybrid)

- **Host.** Qwen3.5-4B has 32 layers: 24 Gated DeltaNet (GDN) recurrent layers and 8 full-attention layers (layers 3, 7, …, 31). We chose it after a size-cliff probe. Under A-PREREG-SIZECLIFF the ≤ 2B line was closed: the 2B/4B ratios were 0.78 on proxy and 0.68 on jevkit open-I, both below the 0.80 bar (PREREG.md).
- **Training.** LoRA training cells on leak-gated draws (jev1, jev2, jev3; §5).
- **Readouts:**
  - the H2O shim format;
  - a native head;
  - `lm_label`, which trains and scores at the same answer slot (JEV3-LM).
- **Cells in this report:**
  - JEV1-4B r2 (job-xt3ax);
  - JEV3-SYM seed 0 (job-5x4vr training, job-6zs79 eval-only screen);
  - JEV3-LM-S (job-zbmhc).
- **H2O-host arms (x7c42, q668q).** These train H2O-Lightning-4B itself on our data. They test whether our data helps, not whether our architecture does.

### 2.2 Research line: attention-free RWKV-7 (D19 / D20 / D21)

- **D19** (stack/rwkv7/d19-v3b-joint):
  - Recipe: RWKV7-G1k-13.3B, fully recurrent, with LoRA, a joint record head, the v3w readout and the bschema v3b data (253,046 rows).
  - proxy600: 48.61 [44.10, 53.08], paired +10.90 [5.61, 14.68] vs D1+v3w (job-fi5xf, independently rescored).
  - Full DEV: 49.49 (§4).
- **D20** (stack/rwkv7/d20-v4-chain):
  - Recipe: D19 on bschema v4, trained as a 3-link chain.
  - Link 1 (job-7d3u9) ran. Links 2–3 are **parked**, because they need ~$113 of holds.
- **D21** (stack/rwkv7/sym-contractive @c6cc6a5e, `docs/D21_DESIGN.md`):
  - Recipe: D20 plus two built-in guarantees:
    - (A) the sym-fork layout with a `joint_record_sym` head (DeepSets mixer), for invariance;
    - (B) `rwkv7_scalar_a`, a rate shared per head, for contraction.
  - Status: built and CPU-tested. With both flags off, the code is bit-identical to the parent. It has never been trained.

### 2.3 Controlling learning: four mechanisms

1. **Invariance by construction** (§3.1).
   - The sym-fork layout takes options out of the shared prefix, and the model never sees option ids. So the row the model reads is literally the same token sequence under any reorder or relabel.
   - For the served system, a canonicalised shim (JEV3-LM) sorts the options into a content-canonical order at inference. That gives the same property on a standard LM-label readout.
2. **νPI controller.** A constraint controller on training (stack/qwen35/nupi): CPU 13/13, OFF golden reproduced (jobs jxawy, urjpw). **Never trained with the controller on.**
3. **Consistency views.** Training on perturbed views of the same item (stack/qwen35/consistency @d517df9c, 13/13, OFF bit-identical, job-xejp8).
   - JEV3-SYM used them at a fixed λ 0.5, with no off-arm. So its result cannot be attributed to this mechanism in either direction.
4. **State stability.** Bounds on how fast the recurrent state forgets (§3.4).

### 2.4 Method discipline

- **Pre-registration.** Every claim-bearing cell was pre-registered before its result existed (`state` PREREG.md, append-only).
- **Selection.** Selection and calibration were fitted only on a held-out slice, and the slice predictions were written before any public prediction (the slice_first timestamps are in the ledger).
- **Multiple arms.** Holm correction across related arms.
- **Independent rescoring.** A separate evidence lane rescored every lead claim from the raw artifacts. All agree numerically, max |diff| ≤ 1.4e-14 (EVIDENCE_REPORT_V2 §7). Two interpretive claims were corrected.

---

## 3. Guarantees

This is the section with positive results. Each guarantee states its scope.

### 3.1 Exact reorder and relabel invariance: PROVEN, by construction

**Theorem.**
- **Statement.** If a head scores each option from (state, option content) only, with a permutation-equivariant normaliser, under conditions C1–C5, then its output is exactly invariant to option reorder and to handle relabel.
- **Typed version (2T).** It holds for nominal, ordinal/score and yes/no ("noul") items, provided the level value of a score item is carried in the option content, never only in the handle.
- **Receipt.** claude/ev-guarantee-proofs @a06343b6, `docs/guarantees/REORDER_INVARIANCE.md` §2T.
- **Review.** An adversarial review corrected the floating-point tolerance to γ_{2K} (REVIEW.md @54701347).

**Independent CPU check** (claude/ev-sym-check @2e465827, 3 rounds):
- sym-fork harness 11/11;
- 74/74 adversarial cases at tolerance 0;
- G16-sym 90 passed after a duplicate-leak fix.

Two leaks found by the reviewer were fixed by the lead before the final round.

**On trained GPU models:**

| Model | G1 (jevkit reverse / shuffle / relabel) | Score-item invariance op | Receipt |
|---|---|---|---|
| JEV3-SYM seed 0 (job-6zs79, ckpt 5cb7da42) | **0 flips / 565 pairs (213/213/139), 0 invalid, max\|Δp\| 0, rows bit-identical** | not run on this cell | claude/ev-6zs79-rescore @f93eccfe (independent rescore) |
| JEV3-LM-S (job-zbmhc, step 150, export 8e936008) | **0 / 565, bitwise PASS** | **bitwise PASS** (the stock model fails it, so the test has teeth) | ledger 2026-10-12T23:40Z; receipts stack/qwen35/jev3-lm @95a731f6. Lead-reported, **independent rescore pending** |
| Stock Qwen3.5-4B + shim (control, same files) | **FAILS: 28 / 21 / 12 flips, max\|Δp\| 0.84** | fails | f93eccfe |

**Scope and limits:**
- **jevkit's G1 never perturbs score items.** It returns n/a for score items under all three operators, and for noul items under relabel. So on score items, invariance rests on Theorem 2T plus the score-invariance op (claude/ev-score-inv @c47c0a46), not on G1.
- **Empty option descriptions break relabel invariance,** as declared: the id then becomes the content.
- **A small numeric residual.** A 2.86e-8 batch-vs-single difference can flip a decision only at near-exact ties.
- **The guarantee came at a Composite cost in its first instance:** JEV3-SYM is −36.66 vs its stock control (§4). JEV3-LM-S keeps the guarantee at −0.78 (n.s.) vs stock.

**What it buys.** Our reorder and relabel flips are 0 bitwise. This matches MD1's reported "0 reorder flips", and ours hold by construction. All four other systems we tested (both Clefs, D13, D19) sort options by id. That makes them reorder-invariant but not relabel-invariant: D19 flips 8.6% under relabel, and stock RWKV7 flips 22.3% (claude/ev-flip-rate @4a65dd18, run 2).

### 3.2 G15 trusted-interval invariance: CONFIRMED on CPU

- **What it is.** The injection-robust conformal set needs each option's trusted interval [lo, hi] to be unaffected by untrusted spans.
- **CPU result.** On stack/qwen35/g16-sym @84aa0910 (per-row P-stream default; CPU-tested in job-r2y6v), G15 holds with the byte encoder and with the **real Qwen3.5-4B BPE tokenizer: 1000/1000 fuzz cases, 0 mismatches**. This was independently confirmed in rounds 2–3 of ev-sym-check @2e465827, after a round-1 BPE leak was fixed.
- **The construction.** It is exact in simulation. Coverage under attack is **0.901** with the construction vs **0.559** for the naive set; the clean reference is 0.897 (claude/ev-guarantee-proofs @1c7c7a70, `INJECTION_ROBUST_CONFORMAL.md`).
- **Open:**
  - no trained G16 head exists yet;
  - G15 has not been checked on a serving GPU.

### 3.3 Conformal coverage on our stack: HOLDS. Abstention certificate: FALSIFIED

**Setup.**
- Split conformal: LAC, Mondrian per question type.
- LTT: ε 0.05, δ 0.10, Bonferroni over 101 λ.
- Data: the jev3 held-out slice (1,200 rows, ids 844233f2; FIT/LTT/TEST 376/406/418).
- Receipt: claude/ev-guar-jev3, pre-registered before any probability was read.

| Cell | TEST coverage @α 0.10 / 0.05 | Mean set size @0.10 | LTT certified act fraction |
|---|---|---|---|
| JEV3-SYM s0 (6zs79) | 0.888 / 0.935 | 1.62 | **0** in every arm and variant (@77b8be9c) |
| JEV3-LM-S (zbmhc), native_tfit | **0.916 / 0.955** | 1.61 | **0** in every arm and variant (@36141c14) |

**Why nothing certifies.**
- **JEV3-SYM.** Confident yes/no answers are wrong about 6% of the time, and 49% of yes/no answers sit in the hedge band.
- **JEV3-LM-S.** The pre-registered per-type temperature over-sharpens yes/no (T_noul = 0.26): the best λ gives 187 acts with 19 errors.
- **The raw and shimcal readouts are near misses.** Each gives about 75 acts with 1 error, while ~135 zero-error acts are needed.
- **The fix is model-side.** It needs roughly 2× the mass of confident correct yes/no answers. A diagnostic, not pre-registered: shimcal beats the tfit readout on NLL, −0.28 [−0.35, −0.21] (@36141c14).

### 3.4 State stability

Lead analysis: claude/lead-state-stability @836bc6f6 (`docs/STATE_STABILITY.md`). It applies the evidence review claude/ev-review-stability @e89b29cc.

| Claim | Verdict |
|---|---|
| **GDN T1:** ‖A_t‖₂ = α_t < 1 on every executed path (β ∈ (0,1), l2-normalised k) | **CONFIRMED** (wording fix) |
| **RWKV-7 T2(a):** the per-step spectrum lies in (−0.455, 1) | Exact arithmetic: **CONFIRMED**. **In floats: NEGATIVE.** exp(w_log) in bf16 can round w to 1.0 |
| **RWKV-7 is not contractive in general** | **Proved at the parameter-box level.** Two admissible steps alternate with ρ(A₂A₁)^{1/2} = 1.105, so no input-independent norm makes every step non-expansive |
| **T2(b):** per-step certificate ‖A_t‖₂ ≤ w_max + σ_a | **CONFIRMED.** With `a` shared per head (σ_a = 0), RWKV-7 is contractive for every input |
| **T5:** cascade forgetting bound through a recurrent stack | Correct, but applies to **no model we run today**: Qwen3.5 has attention at layer 3, and the shipped RWKV-7 has per-channel `a` |

**RWKV-7 contractive by construction.** This is D21's `rwkv7_scalar_a` (stack/rwkv7/sym-contractive @c6cc6a5e): 144 CPU tests pass. CPU probes (jobs 9atmw, ki6vi) show the per-channel `a` exceeds 1 on real steps, while the scalar-a variant stays ≤ w_max < 1. It is untrained.

**Limit.** These are forgetting bounds, not safety claims. They bound how long an injection stays in the state, not whether it gets in. For the hybrid, injection robustness has to come from G15.

### 3.5 The cluster-robust LTT finding: no model certifies on DEV, Clef included

On full-DEV predictions, the certified act fraction at α 0.10 changes with how dependence between decisions is handled.

![Guarantee progression](evidence_report_v2/guarantee_progression_v2.png)

| Model | Per-decision LTT | Per-area (Mondrian) | One decision per row | **Cluster-robust ratio (CRR, c = 8)** |
|---|---|---|---|---|
| Clef 27B | 81.1% | 61.6% | 59.3% | **0** |
| Clef-flash | 72.2% | 67.3% | 51.6% | **0** |
| D19 | 75.1% | 60.1% | 0 | **0** |
| D13 | 36.1% | 40.3% | 0 | 0 |

Receipts, by column: ev-guar-area @ca32b61f (columns 1–2); ev-guarantee-proofs @b4835e05 (column 3); @01738528 `GUARANTEES_CRR.md` (column 4).

**The test is valid.** In simulation with shared-fate clusters, CRR had 0 violations over 107 certifications, while the naive per-decision binomial had 16/120.

**Why every certificate vanishes:**
- The row-weighted acted error at the published λ really is above ε: 8.7% for D19, 5.1% for D13, 6.3% for Clef-flash, 6.5% for Clef 27B.
- At high λ the test lacks power: Bonferroni needs 6.92 nats, the Clefs reach ~6.5 and D19 reaches 4.3.

**Consequence.** Every earlier "D19 is more actionable than Clef-flash" verdict was **withdrawn**. The finding cuts against the competitor's certificates as much as ours. A fixed-sequence CRR on an unseen split is pre-registered (B-PREREG-GUAR-FSCRR), and its 10,365-row sample is built (ev-guar-fscrr @4a5046c0). It is unfunded (§6).

---

## 4. Results

All JevBench rows are paired, stratified bootstrap B = 1000, seed 16, on the public 231 items. "Primary" is each cell's pre-registered comparison.

![Composite deltas](evidence_report_v2/composite_delta_v2.png)

*The chart predates JEV3-LM-S; that row is in the table below.*

| Cell (job) | Comparison | Δ Composite A [95% CI] | Δ vs H2O | Verdict | Receipt |
|---|---|---|---|---|---|
| Stock Qwen3.5-4B + H2O shim (zbtbj) | vs H2O | **−9.4 [−23.1, −4.2]** (67.3 vs 76.6) | same | best of ours, CI < 0 | ev-scorecard @e2fcd82b |
| JEV1-4B r2 (xt3ax) | vs stock + shim | −4.07 [−25.64, +3.22] | −13.21 [−37.82, −7.59] | NOT SUCCESS | ev-j1-rescore @a8ca3e7e |
| JEV2-H2O (x7c42): H2O + jev2, 2,664 rows | vs in-job H2O control | −0.41 [−1.75, +1.39]; p 0.565 | — | FLAT | ev-x7c42-rescore @bf706854; stack/qwen35/jev2-h2o |
| JEV2-H2O-L (q668q): 17,959 rows | vs in-job H2O control | −0.97 [−3.38, +1.12]; p 0.810 | — | FLAT; Holm claims neither arm | ev-q668q-rescore @a100c3b9 |
| JEV3-SYM seed 0 (6zs79) | vs in-job stock + shim | **−36.66 [−47.97, −19.19]** | −46.80 [−59.36, −30.87] | NEGATIVE; single seed | ev-6zs79-rescore @f93eccfe; stack/qwen35/jev3-sym @f591d84e |
| JEV3-LM-S (zbmhc): native_tfit | vs non-canonical stock + shim | **−0.78 [−16.36, +3.47]** | −9.81 [−29.80, −4.64] | NOT CLAIMED (flat). Secondary native_shimcal +3.48 [−1.09, +12.56]. **Independent rescore pending** | ledger 2026-10-12T23:40Z; stack/qwen35/jev3-lm @95a731f6 |
| D19, RWKV7-13.3B + joint head (9j3uu) | vs H2O | −14.4 [−31.7, −10.6] | same | best trained cell, CI < 0 | ev-scorecard @e2fcd82b |

**Capability** (vs H2O, local 80.5):

| Cell | Δ Capability | Receipt |
|---|---|---|
| D19 | −17.2 [−26.6, −10.7] | ev-scorecard |
| stock 4B + shim | −17.7 [−26.5, −8.2] | ev-scorecard |
| JEV1-4B r2 | −22.56 [−31.95, −12.40] | ev-j1-rescore |
| x7c42 (vs in-job H2O) | −0.74 [−3.60, +3.33] (flat) | ev-x7c42-rescore |

**Decision Index, full DEV vs Clef:**

| System | Index [95% CI] | Δ vs Clef-flash 57.29 | Δ vs Clef 27B 61.89 | Receipt |
|---|---|---|---|---|
| D19 | 49.49 [47.77, 50.80] | −7.80 [−9.84, −6.18] | −12.40 [−14.29, −10.82] | ledger 2026-10-10T12:08Z (y6ste + uq5p5); reproduced exactly in ev-d19-cal-rescore @281a0bb9 |
| D19-cal (global T 0.806) | 49.44 [47.72, 50.77] | −7.85 | −12.45 | same; DEAD_ENDS |
| D13 | 43.21 [41.63, 44.26] | −14.08 | −18.68 | ledger 2026-10-10T04:45Z |

- **Where D19 loses** (accuracy vs Clef-flash): knowledge −9.1 [−10.6, −7.8], arts −3.9, tools −2.4. Against Clef 27B, retrieval also loses −9.0 (ev-guar-area @ca32b61f).
- **Where D13 wins.** On held-out attacks, D13 is more robust than Clef-flash: hijack 0.025 vs 0.115, steer 0.003 vs 0.089, Holm-significant (BOARD.md).

**Flip rate** (the only axis partly comparable with MD1; average over 8 jevkit perturbation operators):

![Flip rate](evidence_report_v2/flip_rate_v2.png)

| System | Avg flip % [95% CI] | reorder / shuffle / relabel | Receipt |
|---|---|---|---|
| MD1 (reported; its own items and operators) | 1.3 | 0 / 0 / ? | context only |
| H2O-Lightning-4B v1.1 | 2.44 [1.26, 3.75] | 2.8 / 2.3 / 4.3 | ev-flip-rate @4a65dd18, run 2 |
| JEV3-LM-S | 3.64 (pooled 4.14) | 0 / 0 / 0 bitwise | ledger zbmhc (lead-reported) |
| JEV3-SYM s0 | 4.14 [2.94, 5.35] | **0 / 0 / 0, bitwise** | ev-6zs79-rescore @f93eccfe |
| stock RWKV7-13.3B | 5.37 [3.95, 6.99] | 0* / 0* / 22.3 | ev-flip-rate run 2 |
| Qwen3.5-4B + shim | 7.55 [5.71, 9.67] | 12.2 / 9.4 / 8.6 | ev-flip-rate run 2 |
| D19 | 3-op average 2.88 [1.36, 4.44] | 0* / 0* / 8.6 | ev-flip-rate run 2 (claude/ev-flip-d19) |

\* Zero because the readout sorts options by id. That is a property of the readout, not robustness.

- **Against MD1.** NOT MET on flip rate. Our remaining flips come from paraphrase and formatting operators (whitespace 10.4%, option paraphrase 8.7% on JEV3-SYM), which the invariance guarantee does not cover.
- **Not comparable.** MD1's accuracy, calibration and latency use private items, a different formula and a different serving stack.
- **Not paired.** The JEV3 vs stock-shim flip comparison is not paired; the baseline comes from another job.

---

## 5. Negative results and gap attribution

### 5.1 Where the Composite gap comes from

Lead analysis: claude/lead-gap-attribution @1edcd524 (`docs/GAP_ATTRIBUTION_COMPOSITE.md`). It was reviewed in claude/ev-review-gap-attr @f2a04059: the review reproduced the analysis byte-identically and corrected three claims.

![Shapley split](evidence_report_v2/gap_shapley.png)

- **Intelligence is the largest axis.** By exact Shapley over I/C/S/K it is 50–67% of the gap: stock 6.28 of 9.38, JEV1 8.06 of 13.21, D19 7.26 of 14.39. P(I is the largest axis) ≥ 0.979.
- **Within I, yes/no ("noul") items are the biggest hole:**
  - stock 12.20 [5.97, 19.38];
  - D19 13.41;
  - JEV1 16.21.
- **The scorer weights hard score items heavily.** One hard score item is worth ~10.4 I points per full miss, and the 6 hard score items carry ~22% of I weight. This is a property of the scorer, known before any result.
- **Claims the review corrected:**
  - "six families hold 66–81% of the gap": retracted as post-hoc;
  - "C is second for D19": wrong;
  - "I and C close the gap": wrong for JEV1 and D19, because S and K cost them 1.09 and 5.3.

### 5.2 The transfer finding (the central negative)

![Slice vs public](evidence_report_v2/slice_vs_public.png)

| Run | Held-out slice gain (top-1) | Public Δ Composite A | Per-family transfer correlation |
|---|---|---|---|
| x7c42 | +4.44 pp [+3.47, +5.31] | −0.41 (flat) | +0.15 |
| q668q | +8.97 pp [+7.61, +10.19] | −0.97 (flat) | −0.32 |
| JEV3-SYM | +1.08 pp (n.s.) | −36.66 | −0.50 |

Receipts: the three rescores in §4.

- **Where the gains land.** Slice gains fall on families already saturated on public, where H2O is at 83–100%.
- **Where the public errors are.** 23 of the 26 public errors sit in temporal_numeric, judge_hard, long_policy and multi_hop, and those families see no slice gain.
- **More data made it worse.** Seven times more data doubled the slice gain and made the public result worse.
- **Human-labelled rubric data did not help either.** The jev3 data (HelpSteer/Aegis/PRM800K-style rows) also did not transfer (−0.50).
- **All draws were leak-free** (ev-jev2a-regate @688e8d7c, which includes jev3 provenance on all 7,160 rows). So the failure is not contamination.

### 5.3 Yes/no hedging is a readout effect, not a ranking effect

- **The hedging.** JEV3-SYM puts **64.9% (48/74)** of public yes/no answers in the 0.2–0.8 band, which jevkit scores as wrong. Stock and H2O put 0/74 there.
- **The ranking is intact.** noul top-1 rises +5.4 pp (ev-6zs79-rescore @f93eccfe §4).
- **Other losses are real ranking losses:**
  - score: top-1 −33.3 pp (0/6);
  - routing: −58.3 pp (0/7);
  - ordinal: −50 pp (0/6).
- **The fork head is a cause.** A state-branch check (`receipts/2026-10-12-sym-label-check/RESULT.md`) found the per-option fork head predicts only the extreme score levels: on the slice, 37.2% vs stock 48.8%, CI [−30.2, +7.0]. It scores each level alone, never next to the scale. This motivated the typed rendering (Theorem 2T) and the scored LM-label readout of JEV3-LM.

### 5.4 Fork head vs scored readout

On the same Qwen3.5-4B host and jev3 data, two readouts were compared:

| Cell | Readout | Δ Composite A vs its stock control |
|---|---|---|
| JEV3-SYM | sym-fork head | −36.66 |
| JEV3-LM-S | LM-label scored at the answer slot, canonical-order inference | **−0.78 (n.s.)** |

- **What carries over.** Both keep bitwise G1 invariance. JEV3-LM-S also passes the score-item op, and its slice floor is +4.5 pp [+1.9, +7.1].
- **The canonicalised stock model alone costs a lot.** canon_stock vs stock is **−22.29 [−33.49, −3.27]**, so training recovers most of what canonical ordering costs the untrained model.
- **Caveats:**
  - two different cells, each a single seed;
  - JEV3-LM-S has not been independently rescored yet;
  - this is not a controlled ablation.

### 5.5 The letter-prior finding

Source: claude/ev-canonical-prior @1ccdc676, `GOLD_LETTER_CANONICAL.md`.

- **The prior exists in train and slice.** Canonical option order gives train choice items gold = A on **35.0%** vs 28.0% chance (p = 2.5e-19). The slice has 38.5%.
- **It does not exist on public.** Public has 24.5% vs 23.0% (p = 0.78).
- **One family drives it: `ambiguous`.** There "decline" is never gold and sorts last.
- **Consequence.** A letter-shortcut model would gain on the slice and not on public. So choice gains on the slice are not evidence against a shortcut.
- **Mitigation.** JEV3-LM trains on shuffled orders. JEV3-LM-S showed no flag: A on the slice −4.1 pp, on public −0.7 pp (ledger zbmhc).

### 5.6 Other killed lines (`state` DEAD_ENDS.md)

| Line | Result |
|---|---|
| Q9-AF attention→GDN conversion of the 9B | Proxy −21.77, FOLD |
| KD from Clef-flash | No gain. The teacher follows injected instructions |
| D19 global temperature | Index 49.49 → 49.44, ECE worse |
| RAGTruth statement rephrase | +0.000 Index |
| fla WKV kernel for inference | Failed its equivalence gate: 98.74% < 99% agreement |
| Constant-LR training past one epoch | Collapse or overfit. Every long run now uses cosine decay + best-checkpoint selection + a collapse guard |
| JEV3-LM, first attempts (79fki, wm8b5) | **Infrastructure failures, not falsifications.** A truncation assert; then a timeout at 497/729 steps, with no weights exported. All three runner bugs were fixed CPU-only (b4da25e0) before JEV3-LM-S |

---

## 6. What's next, costs, and the funding ask

**Balance: $11.67 of $785** (ledger, job-zbmhc row). A job is blocked when its worst-case hold exceeds the balance, even if its expected cost fits.

| Item | What it tests | Cost | Source |
|---|---|---|---|
| **JEV3-LM-R** (branch stack/qwen35/jev3-lm-r, cut at 95a731f6, local only, no new commits yet) | Follow-up to JEV3-LM-S. PROGRESS 2026-10-12 names the next step: pre-register shimcal (no slice-fitted T_noul) as primary. shimcal was JEV3-LM-S's non-pre-registered secondary, at +3.48 [−1.09, +12.56] | Not sourced. Comparable: JEV3-LM-S billed $2.63 against a $13.73 hold; EVIDENCE_REPORT_V2 estimates JEV3-LM at ≈ $10–18 | ledger; EV2 addendum |
| **Independent rescore of JEV3-LM-S** | Required before any JEV3-LM-S number is cited as confirmed | $0 (CPU) | — |
| **D21** (scalar-a RWKV-7 + sym-fork) | The attention-free cell with both guarantees by construction | Queued behind D20 links 2–3: **~$113** of holds. D21's own cost is not sourced | BOARD #13; EV2 §5d |
| **jev-heldout** (branch claude/ev-jev-heldout, local only, still at the jev3-draw commit 5c963b25) | A new held-out set. Its scope is not yet committed; the need for it comes from the transfer finding (§5.2) | Not sourced | — |
| **sym-fast** (branch stack/qwen35/sym-fast, local only, still at the jev3-typed commit a21049a9) | Scope not yet committed. Context: the native readout's Speed cost (0.197 s vs 0.067 s for the shim; GAP_ATTRIBUTION C3) | Not sourced | — |
| FS-CRR on the unseen split (D19 + both Clefs) | Whether any model certifies abstention under a valid test | **~$35** (+ D13 ~$22). Also needs the operator directive checked, because it runs Clef | ledger job-9na9j; EV2 §5b |
| Trained G16 head + GPU G15 check | The injection-robust sets on a real model | ~$6 expected / **$21.67** hold | EV2 §5c |
| νPI ON/OFF pair (JEV3-SYM-NU) | Whether the controller helps | ~$12 / **$43.34** hold | EV2 §5e |
| JEV3-SYM seed 1 | Replicates a negative (low value) | ≤ $21.67 | EV2 §1.4 |

**Funding ask.** Itemised from the sourced worst-case holds:

| Tier | Contents | Amount |
|---|---|---|
| Minimum to close the open guarantee items | FS-CRR (~$35) + G16 hold ($21.67) | **~$57** |
| Controller test | νPI pair | **+$43.34** |
| Main-line next cell | JEV3-LM-R | **+~$18** (upper estimate) |
| Attention-free line | D20 links 2–3 before D21 | **+~$113** |

**Expectations.** The sourced items total about **$231** in worst-case holds; billed costs have run well below holds. None of these items is expected to close the Composite or Clef gaps by itself. §5.2 shows that more of our data has not transferred. So the cheap readout fix (JEV3-LM-R) and the guarantee items come before larger training runs.

---

### Numbers not sourced in this report
- Costs and scope for D21 itself, jev-heldout and sym-fast. Their branches (jev3-lm-r too) exist only locally, with no commits beyond their cut points.
- JEV3-LM-S numbers are the lead's ledger row and receipts @95a731f6 only. The evidence lane's independent rescore of that cell does not exist yet. Its guarantees run (@36141c14) does exist.
- JEV3-LM-S flip CIs are not in the ledger row.
