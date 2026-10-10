# Guarantees on the full-DEV predictions: D19, D13, Clef 27B, Clef-flash

Pre-registered as **B-PREREG-GUAR-FULLDEV** (state @2c683e74, pushed before any metric was computed). Code: `jevkit/guarantees.py` @ agent-main 7c131f1, unmodified. Driver: `b-landscape/guarantees_fulldev/guarantees_fulldev.py`. Numbers: `GUARANTEES_FULLDEV.json`.

## Takeaway

- **Clef 27B gives the most useful guarantees. D19 comes second, Clef-flash third, and D13 is far behind.** These are the guarantees we can certify: an error rate of at most 5% on the decisions the model acts on (ε = 0.05, δ = 0.10). The share of held-out decisions each model may act on under that certificate:
  - Clef 27B: **81.1%**
  - D19: **75.1%**
  - Clef-flash: **72.2%**
  - D13: **36.1%**
- **D19 vs Clef 27B:** D19 can act on 6.0 points fewer decisions, 95% CI [−8.6, −3.9]. Its 90% sets are also 0.08 labels larger [+0.07, +0.10]. Clef 27B is ahead on both measures, and that holds at α = 0.05 too.
- **D19 vs Clef-flash:** D19 can act on more decisions: +2.9 points [+0.2, +5.1] at α = 0.10, and +6.9 [+3.5, +9.7] at α = 0.05. Its sets are slightly larger at α = 0.10 (+0.04 [+0.02, +0.06]), and there is no separation in set size at α = 0.05.
  - **The act-fraction edge at α = 0.10 is fragile.** In the case-keyed sensitivity split, the CI becomes [−0.1, +6.4], which means no separation. The α = 0.05 edge survives that split.
- **The 13 → 19 gain is large:** +39 points of act fraction and −0.30 labels of set size at α = 0.10.
- **Every model meets its guarantee.**
  - Marginal held-out coverage is 0.899–0.903 at a target of 0.90, and 0.949–0.954 at a target of 0.95.
  - Held-out acted-on error is 3.7–4.3%, below ε = 5%.
- **The marginal guarantee hides area-level failure, and this affects every model.**
  - At α = 0.10, **arts** coverage is 0.69–0.79 and **knowledge** coverage is 0.74–0.83.
  - Language is over-covered at about 0.97, and it is 57% of decisions.
  - Fix: use Mondrian conformal per area instead of per type. This is a change to the procedure, not the model. It must be pre-registered before we use it.
- **Reorder invariance:** all four models get the same result whatever order the options are listed in. That is because they sort options by id before reading them, not because of a structural guarantee. None of them is invariant to renaming option ids, so this mechanism does not separate D19 from Clef. These runs contain no permuted-option predictions, so no flip rate is measured.
- **Injection-robust sets (CONTROL_RESEARCH mechanism 3):** not computable from these runs (section 4).

## Setup

**Inputs.** Each file's sha256 was checked at run time.

| Model | Results file | sha256 |
|---|---|---|
| D19 | `d19_fulldev_land/merged/fulldev/child.results.jsonl` | b6bab803…46a8 |
| D13 | `d13_fulldev_land/merged/fulldev/child.results.jsonl` | 7b42f4c3…b1dd |
| Clef 27B | `clef_zfexa_land/.../decide_32k_substituted/results.jsonl` (job-zfexa) | f5ad788b…b09e |
| Clef-flash | `clef_t5yg2_land/.../decide_32k_substituted/results.jsonl` (job-t5yg2) | 2cb2e961…49c5 |

- All four files cover all 12,104 DEV rows with status ok.
- Each file contains 49,444 decisions (46,414 choice and 3,030 noul), and every decision has a valid distribution.

**Gold.**
- Gold is `expected[key]` from the materialised 0.2.1 DEV rows, edition 0.2.1.
- The run_id order was asserted equal to `harness/manifests/dev-manifest.jsonl.gz`, which is the frozen loader's check.
- The rows were streamed rather than loaded through `_load_slice_021`, because of RAM.

**Decisions without gold are excluded: 14,725 of 49,444.**

| Benchmark | Decisions without gold | Why there is no gold |
|---|---|---|
| ToolRet (cat 2) | 9,459 | Graded by nDCG over documents, not by a per-field label |
| BRIGHT (cat 36) | 2,966 | Graded by nDCG over documents, not by a per-field label |
| PhishNChips (cat 56) | 2,300 | Signal fields with `expected = None` |

This leaves **34,719 scorable decisions in 8,473 source cases**.
- The pre-registration listed decision counts over all question keys (24,701 / 24,743). The shas of the row split are unchanged.
- Excluding these decisions is forced, because there is no conformal score without a label. It is still a deviation from the registered counts, and it is stated here.

**Split** (answer-independent, identical for every model).

| Step | Rule | Result |
|---|---|---|
| Rows into CAL and TEST | sha256("B-PREREG-GUAR-FULLDEV\|v1\|" + run_id) < 0.5 → CAL | 6,156 CAL rows (sha 3f3dc857…943f), 5,948 TEST rows (4e2ff6f5…3837) |
| Scorable decisions | Rows split as above | 17,234 CAL, 17,485 TEST |
| CAL into FIT and LTT | jevkit `split_calibration` (seed 0, ½) | Disjoint FIT and LTT halves; per-model shas are in the JSON |

**Method.**
- Score s = 1 − p_gold. Choice distributions go through jevkit `validate_distribution`, which renormalises the 4-dp rounding. Noul decisions use {yes: p, no: 1 − p}.
- Conformal is Mondrian per decision type, the jevkit default.
- LTT uses the rule "act iff |C| = 1 and p_max ≥ λ". It uses Bonferroni over 101 values of λ (1.00 down to 0.00), with ε = 0.05 and δ = 0.10.

**Bootstrap.**
- Pairwise CIs come from a case-cluster bootstrap over the TEST decisions (4,658 harness source cases; B = 2000, seed 0).
- q_hat and λ are held fixed, so the CIs are conditional on calibration. They therefore understate the total variability somewhat.

## 1. Conformal sets (TEST half; Mondrian per type)

| Model | α | Coverage | Mean set size | Singleton rate | Empty rate | q_hat (choice / noul) |
|---|---|---|---|---|---|---|
| **Clef 27B** | 0.10 | 0.903 | **1.017** | 0.934 | 0.024 | 0.609 / 0.612 |
| Clef-flash | 0.10 | 0.901 | 1.060 | 0.882 | 0.029 | 0.679 / 0.797 |
| D19 | 0.10 | 0.899 | 1.100 | 0.884 | 0.010 | 0.775 / 0.878 |
| D13 | 0.10 | 0.902 | 1.404 | 0.584 | 0.030 | 0.884 / 0.927 |
| **Clef 27B** | 0.05 | 0.949 | **1.231** | 0.765 | 0.012 | 0.861 / 0.794 |
| D19 | 0.05 | 0.949 | 1.338 | 0.728 | 0.002 | 0.920 / 0.910 |
| Clef-flash | 0.05 | 0.950 | 1.352 | 0.659 | 0.013 | 0.883 / 0.892 |
| D13 | 0.05 | 0.954 | 1.927 | 0.195 | 0.016 | 0.941 / 0.942 |

- **By type.** Choice decisions (n = 17,112) track the overall numbers. Noul decisions (n = 373) cover 0.887–0.938 at α = 0.10. The lowest is Clef 27B at 0.887, within noise of 0.90 at n = 373.

**By area, α = 0.10.** Each cell is coverage / mean set size (the marginal target is 0.90).

| Area (n) | D19 | D13 | Clef 27B | Clef-flash |
|---|---|---|---|---|
| arts (1,196) | **0.725** / 1.24 | 0.786 / 1.67 | **0.688** / 0.96 | **0.710** / 1.00 |
| knowledge (2,288) | **0.737** / 1.17 | 0.830 / 1.52 | **0.765** / 0.96 | **0.779** / 1.01 |
| language (9,985) | 0.974 / 1.05 | 0.967 / 1.40 | 0.968 / 1.03 | 0.971 / 1.08 |
| retrieval (1,943) | 0.797 / 1.20 | 0.722 / 1.27 | 0.840 / 1.07 | 0.784 / 1.08 |
| tools (2,073) | 0.913 / 1.10 | 0.902 / 1.28 | 0.923 / 1.01 | 0.915 / 1.04 |

- **The guarantee is marginal, as it should be, and it is met.** It is not area-conditional.
- **Language is the easy majority** (57% of decisions). Its low scores pull the pooled q_hat down, so arts, knowledge and retrieval get sets that are too small to cover their 90%.
- **All four models show this.** It is a property of pooling over heterogeneous areas, not a model defect.
- **Remedy:** Mondrian per 0.2.1 area, or per benchmark. This gives each area its own coverage of at least 1 − α, at the cost of larger sets in the hard areas.
  - It is a natural next pre-registration, and it costs $0 on CPU.

## 2. Learn-then-Test: certified act fraction at ε = 0.05, δ = 0.10

Act fractions and errors are on the TEST half. "LTT UCB" is the Clopper-Pearson bound on acted-on error, computed on the LTT half.

| Model | α | Certified λ | LTT act rate | LTT UCB | **TEST act fraction** | TEST acted-on error |
|---|---|---|---|---|---|---|
| **Clef 27B** | 0.10 | 0.78 | 0.806 | 0.0488 | **0.811** | 0.038 |
| D19 | 0.10 | 0.86 | 0.742 | 0.0491 | **0.751** | 0.043 |
| Clef-flash | 0.10 | 0.81 | 0.724 | 0.0497 | **0.722** | 0.037 |
| D13 | 0.10 | 0.92 | 0.350 | 0.0487 | **0.361** | 0.037 |
| **Clef 27B** | 0.05 | 0.00 | 0.763 | 0.0469 | **0.765** | 0.037 |
| D19 | 0.05 | 0.00 | 0.720 | 0.0486 | **0.728** | 0.042 |
| Clef-flash | 0.05 | 0.00 | 0.670 | 0.0472 | **0.659** | 0.035 |
| D13 | 0.05 | 0.95 | 0.064 | 0.0402 | **0.059** | 0.021 |

- **λ = 0.00 means "act on every singleton set".** At α = 0.05 the singleton sets are already accurate enough, so the conformal set does all the work.
- **All acted-on errors on held-out data are below 5%.** So the certificate is honoured out of sample.

## Pairwise comparisons (case-cluster bootstrap, 95% CI)

Primary split, keyed on run_id.

| Pair | α | Δ mean set size (lower = tighter) | Δ TEST act fraction | Verdict |
|---|---|---|---|---|
| D19 − Clef 27B | 0.10 | +0.082 [+0.068, +0.100] | −0.060 [−0.086, −0.039] | Clef 27B tighter and more actionable |
| D19 − Clef 27B | 0.05 | +0.107 [+0.073, +0.149] | −0.037 [−0.068, −0.012] | Clef 27B tighter and more actionable |
| D19 − Clef-flash | 0.10 | +0.040 [+0.024, +0.060] | **+0.029 [+0.002, +0.051]** | Clef-flash tighter; **D19 more actionable** |
| D19 − Clef-flash | 0.05 | −0.015 [−0.050, +0.029] | **+0.069 [+0.035, +0.097]** | Sizes tie; **D19 more actionable** |
| D13 − Clef 27B | 0.10 | +0.386 [+0.359, +0.418] | −0.449 [−0.468, −0.431] | Clef 27B |
| D13 − Clef-flash | 0.10 | +0.344 [+0.315, +0.377] | −0.361 [−0.382, −0.341] | Clef-flash |
| D19 − D13 | 0.10 | −0.304 [−0.329, −0.280] | +0.390 [+0.361, +0.416] | D19 |
| Clef 27B − Clef-flash | 0.10 | −0.042 [−0.049, −0.036] | +0.088 [+0.080, +0.097] | Clef 27B |

**Sensitivity: split keyed on the harness case id**, so no source case straddles CAL and TEST (6,116 TEST rows, 17,441 decisions, 4,276 clusters).
- **Coverage still meets its targets.** It is 0.891–0.905 at α = 0.10 and 0.944–0.953 at α = 0.05. D19 at α = 0.10 is 0.891, just above the 0.89 flag line.
- **Rankings are unchanged.** The TEST act fractions at α = 0.10 are:
  - Clef 27B: 0.817
  - D19: 0.747
  - Clef-flash: 0.712
  - D13: 0.275
- **One verdict changes.** D19 − Clef-flash act at α = 0.10 becomes +0.035 [−0.001, +0.064], which is **no separation**. At α = 0.05 it remains +0.068 [+0.023, +0.102], with D19 more actionable.
- **Every other verdict keeps its direction and significance.**

## 3. Reorder invariance

**Summary:** all four models are reorder-invariant, but by sorting their input rather than by a structural guarantee. None of them is invariant to relabeling option ids. So on CONTROL_RESEARCH mechanism 1 these runs do not separate D19 from Clef. The argument comes from reading the code, spot-checked by agent-b. Nothing was measured.

- **D13 and D19 (RWKV7-13.3B, `joint_record` cache-fork head).**
  - **Configuration.**
    - D13: `layout: ordered`, template `jrt-rwkv-g1-v1` (`stack/rwkv7/d13-joint`).
    - D19: `layout: g1v2`, template `jrt-rwkv-g1-v3w` (`claude/b-d19-fulldev`). "v3b" is the training draw, not a readout.
  - **Options are sorted before the prompt is built.** `decide/schema.py` `canonical_options` sorts choice options by id; noul is fixed as (true, false); score keeps level order.
  - **The prompt then lists them in that order.** The shared prefix contains an ordered option list (`heads/layout.py` `OPTION j: ...`). D19's `prompt_g1v3.py` assigns codes A, B, ... by sorted position.
  - **The head itself treats options symmetrically.**
    - Each option forks the recurrent state from the shared prefix and is scored on its own.
    - `JointRecordHead` pools options with a masked mean and max, has no option-position embedding, and takes a log-softmax over the option set.
  - **Result: the request bytes are identical under any reordering of `criteria`, so the distribution is identical.** It is exact if kernels are deterministic.
  - **This is canonicalisation, not equivariance.** The `symmetric` layout, which keeps the option list out of the prefix, was not used. Relabeling ids changes the tokens, and if the new ids sort differently it also changes each option's slot or code. So relabel flips are possible.
- **Clef 27B and Clef-flash (`baselines/clef_index_engine.py`, `decide` path over `ClefBackend`).**
  - **The same sorting happens twice.** `decide/schema.py` sorts, and Cloudflare's own `joint_schema_model.question_options` also sorts choice options by id (`third_party/cloudflare_clef/joint_schema_model.py`:55-56).
  - **Clef's JointSchemaHead is order-symmetric over options.**
    - Each option gets its own encoding and evidence lookup.
    - Options are pooled by a softmax-weighted sum (:408-414).
    - Options are scored with no index, followed by a softmax.
  - **The verdict is the same as for D13/D19.** Clef is invariant to option order by canonicalisation, not invariant to relabeling, and also NOT invariant to question order (`FIELD i` is numbered in request order).
- **Measured flips: not measured.**
  - No predictions under permuted or relabelled options exist for any of the four models on DEV. The search covered the repo and `C:\Users\apumu` by file name, so it is not exhaustive.
  - The d6-order-perm worktree holds a training config, not predictions.
- **What would produce the measurement.** `jevkit.guarantees.invariance_report` (all K! orders for K ≤ 4, plus 3 relabelings) on a DEV subset, for each backend.
  - Expected result: max|Δp| = 0 on reorders for all four (they are canonicalised).
  - Relabelings are where models could differ, and where only a `symmetric`-layout cell with neutral canonical ids (CONTROL_RESEARCH §1) could certify 0.

## 4. Injection-robust sets (CONTROL_RESEARCH mechanism 3): not computable from these runs

- **What is needed.** `robust_set` / `injection_robust_report` need, for every decision, the per-option trusted-only logit intervals `[lo, hi]` from G16: `logit_k ∈ [b_k − c‖a_k‖₁, b_k + c‖a_k‖₁]`.
- **What these runs contain.** None of them does that.
  - D13 and D19 are the plain joint-head readout without the G16 trust-factored head.
  - Clef has no trust factorisation at all.
- **The job that would produce the intervals** is a G16 cell scored on full DEV:
  - Train a D19-recipe cell with the G16 head (`controlsafe/g16.py`; the `c` sweep through `train.g16_card sweep`).
  - Then run `fulldev_eval` with an export that writes, per decision, `{id, type, label, intervals: {y: [lo, hi]}, clean_logits}` (the input format of `injection_robust_report`), plus `attacked_logits` from the m2 `placement: instructions` injection rows and the GCG letter-logit attacker.
  - The ordinary split-conformal q_hat would come from the same CAL half registered here.
- **Clef cannot produce these intervals by architecture.** So on this mechanism the comparison is "certified for us vs empirical only for Clef", once the G16 cell exists.
- **Before trained weights exist,** a random-judge G16 can validate the pipeline at $0 (CONTROL_RESEARCH §3).

## Caveats

- **Exchangeability.** Both guarantees assume deployment decisions are exchangeable with DEV. Neither survives distribution shift.
  - The primary split keys on run_id, so correlated decisions from one source case can sit in both CAL and TEST.
  - The case-keyed sensitivity removes that, and the conclusions hold, apart from the noted D19-vs-flash α = 0.10 act edge.
- **Coverage is marginal over decisions**, not per area (section 1).
- **Bootstrap CIs are conditional on the fitted q_hat and λ.**
- **About 30% of DEV decisions (ToolRet, BRIGHT, PhishNChips signals) have no per-decision gold** and are outside these guarantees.
- **These are DEV numbers (0.2.1), not held-out or TEST.** None of this is a board claim.
- **Environment.** The harness venv (`ssdm-wt/m0-harness/.venv`, which has the decision_index kit) was used instead of `m2-heads/.venv`, because the latter lacks the kit that `harness.cases` / `harness.slicing` import. jevkit was imported from the read-only agent-main checkout.
