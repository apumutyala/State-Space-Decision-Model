# Independent rescore: JEV1-4B r2 (job-xt3ax), evidence lane

**Bottom line: the PRIMARY test FAILS (not a success).** Δ Composite A (trained `native` − `stock_shim`) = **−4.07, 95% CI [−25.64, +3.22]**. The CI includes 0, and the point estimate is negative. This rescore **AGREES** with the in-job `screen.json` on all 67 compared numbers (max |diff| 0.0), plus the own-loop vs `jevkit.scoring.bootstrap` cross-check (exact). Against H2O-Lightning-4B, the trained arm is clearly **worse**: Δ Composite A −13.21 [−37.82, −7.59].

The training draw does help on some families and hurt on others, but there is no net gain. On the 8 families the jev1 draw COVERS, top-1 accuracy *falls* by 6.7 pp, driven by long_policy (−31.6 pp). On the 10 families it does NOT cover, top-1 is flat (+0.9 pp).

## Inputs (all independently fetched)
- Artifact art-mv9cr, `autor job get job-xt3ax`, downloaded by this lane to `C:\Users\apumu\art\job-xt3ax\dl\art.tar`. Its sha256 0b0707c6…b4a2 (506,060,800 B) **equals** the autor-reported sha. Only `phases/screen.tgz` and `phases/final.tgz` were extracted, and they were read with `python -I`.
- Preds sha256: native 4915fd96…a853, stock_shim a2a3d337…85b1, native_shimcal accbac06…0d3d. These are identical to the lead's copy in `jev1_4b_r2_land`, matched by prefix. The lead's extraction was **not** used.
- Checkpoint: cell.json train.steps **1121** (= `--expect-steps`, matching max_steps). The scored checkpoint is **best@1000** (`ckpt-step0001000.pt`, sha 824ab4d8…f552; dev-card nll 0.644, acc 0.74), the same as 27anq. Trainable safetensors best: step1000 sha ea672c8d….
- Code: `scripts/ev_rescore_jevkit.py` on claude/ev-j1-rescore. It is unchanged from 2f9d0de7 for the primary path and gains a new `--secondary-axes/--groups/--vs` option (EV-PREREG-SECONDARY-AXES). Frozen jevkit files are untouched.
- Command: `python -I -X utf8 scripts/ev_rescore_jevkit.py --art <x> --out rescore-xt3ax-full.json --baseline stock_shim --primary native --expect-steps 1121 --secondary-axes --groups "COVERED=intent+extraction+long_policy+policy+ordinal+adequacy+routing+routing_hard;UNCOVERED=multi_hop+judge_hard+temporal_numeric+fact+tool_selection+probability+trap+ambiguous+tradeoff+adversarial" --vs "h2o=<zbtbj final/out/cand/h2o-lightning-4b/h2o_shim/preds.jsonl>::<…/price.json>"`. Exit 0, no FLAG. A plain run without the new options (`rescore-xt3ax.json`) gives the identical primary.
- Integrity: every arm has 231/231 pinned ids, 0 dupes, all status ok, 0 choice answers out of criteria order, and the declared price class Qwen/Qwen3.5-4B equals the in-job one. Latency comes from the open standard+judge timed set.

## PRIMARY (A-PREREG-JEV1-4B), with paired B=1000 seed 16 and the same draws for every row
| axis | native | stock_shim | Δ | 95% CI | vs screen.json |
|---|---|---|---|---|---|
| **Composite A** | 63.43 | 67.50 | **−4.07** | [−25.64, +3.22] | AGREES (0.0) |
| Intelligence (open) | 51.18 | 57.05 | −5.87 | [−15.81, +5.11] | AGREES |
| Calibration | 64.69 | 70.21 | −5.52 | [−17.62, +5.18] | AGREES |
| Speed (measured, held fixed) | 85.25 | 89.64 | −4.39 | fixed | AGREES |
| Cost (same price class) | 61.23 | 61.23 | 0.00 | fixed | AGREES |
| Capability | 57.93 | 63.63 | −5.69 | [−15.76, +4.22] | AGREES |
| tokens / decision (mean) | 653.6 | 653.6 | 0.0 | (p50 157) | AGREES |
| top-1 (jevkit correctness) | 166/231 | 173/231 | −7 (−3.0 pp) | [−8.7, +3.0] pp; McNemar exact p 0.38 (20 vs 27 discordant) | n/a |
| mean NLL, choice (n 139) | 0.689 | 0.615 | +0.075 | [−0.100, +0.253] | n/a |

**PRIMARY verdict: NOT SUCCESS.** The in-job `primary.success` is also false, so this rescore agrees with it.

Secondary rows from B-PREREG-JEV1-4B (reported only, not used for the decision):
- The ΔI CI lower bound must be > 0. It is −15.81 → **fails**.
- "Composite rises": it falls by 4.07 → **fails**.
- So J1 would not be a KEEP under B's rule either. (B's rule specifies a case-cluster bootstrap with B=2000, seed 0. Here the stratified B=1000, seed 16 draws are used, so the CI is indicative only. The point estimates are rule-independent.)
- Proxy600 Decision Index, in-job only and not rescored here: child 36.36 [33.46, 41.61] vs the D1+v3w reference 37.71, Δ −1.35 [−5.73, +3.59].
- `native_shimcal` arm (stock shim's calibration readout over the trained weights): Δ A −2.66 [−12.14, +6.42], ΔI −0.41, ΔC −7.60. Also not a success.

## Why the lower bound is −25.64: the Intelligence floor
Composite A = harmonic mean(I, C, S, K) × min(1, I/50)² × min(1, S/50)² × min(1, K/50)². The trained arm's I is 51.18, only 1.2 points above the floor. In **41.9%** of resamples its I falls below 50 (stock: 9.0%), and each of those draws takes the (I/50)² penalty. For example, I = 41 multiplies the harmonic mean by 0.67.

Quantiles of the Composite-A bootstrap (same draws):

| quantile | 1 | 2.5 | 5 | 25 | 50 | 75 | 95 | 97.5 | 99 |
|---|---|---|---|---|---|---|---|---|---|
| native A | 35.55 | 38.24 | 42.26 | 55.81 | 62.69 | 64.71 | 67.33 | 67.95 | 68.71 |
| stock_shim A | 50.71 | 55.22 | 60.14 | 65.13 | 67.05 | 68.68 | 70.99 | 71.67 | 72.76 |
| **Δ A** | −28.92 | **−25.64** | −22.43 | −10.41 | −5.05 | −2.34 | +1.66 | **+3.22** | +7.24 |
| Δ A with the floor term removed (diagnostic) | −11.42 | −10.25 | −9.17 | −6.24 | −4.23 | −2.20 | +1.07 | +1.92 | +2.90 |
| Δ A, only draws where both I ≥ 50 (n 558) | −9.17 | −8.12 | −7.33 | −4.93 | −3.20 | −1.54 | +1.30 | +2.03 | +2.89 |
| Δ A, only draws where native I < 50 (n 419) | −29.79 | −28.68 | −26.25 | −17.43 | −11.88 | −7.94 | −1.98 | +1.30 | +5.28 |

The distribution is strongly left-skewed. The wide lower bound is the floor term working as designed, not a bug. Without the floor term the CI would be [−10.25, +1.92], which still includes 0, so the verdict does not depend on the floor. Note that the point estimate has no floor penalty, because both full-sample I values are ≥ 50.

## Per-family top-1, trained vs stock_shim (jevkit correctness)
| group | family | n | native | stock | Δ pp | native-only / stock-only |
|---|---|---|---|---|---|---|
| COVERED | intent | 24 | 91.7 | 91.7 | 0.0 | 1/1 |
| COVERED | extraction | 24 | 95.8 | 100.0 | −4.2 | 0/1 |
| COVERED | long_policy | 19 | 21.1 | 52.6 | **−31.6** | 2/8 (p 0.11) |
| COVERED | policy | 12 | 100.0 | 91.7 | +8.3 | 1/0 |
| COVERED | ordinal | 12 | 100.0 | 100.0 | 0.0 | 0/0 |
| COVERED | adequacy | 12 | 66.7 | 83.3 | −16.7 | 1/3 |
| COVERED | routing | 12 | 91.7 | 91.7 | 0.0 | 1/1 |
| COVERED | routing_hard | 5 | 100.0 | 100.0 | 0.0 | 0/0 |
| UNCOVERED | multi_hop | 18 | 72.2 | 55.6 | +16.7 | 4/1 |
| UNCOVERED | judge_hard | 17 | 52.9 | 58.8 | −5.9 | 1/2 |
| UNCOVERED | temporal_numeric | 15 | 6.7 | 13.3 | −6.7 | 1/2 |
| UNCOVERED | fact | 12 | 91.7 | 91.7 | 0.0 | 1/1 |
| UNCOVERED | tool_selection | 12 | 100.0 | 100.0 | 0.0 | 0/0 |
| UNCOVERED | probability | 10 | 50.0 | 40.0 | +10.0 | 2/1 |
| UNCOVERED | trap | 8 | 75.0 | 87.5 | −12.5 | 0/1 |
| UNCOVERED | ambiguous | 7 | 57.1 | 42.9 | +14.3 | 2/1 |
| UNCOVERED | tradeoff | 6 | 33.3 | 83.3 | −50.0 | 1/4 |
| UNCOVERED | adversarial | 6 | 100.0 | 66.7 | +33.3 | 2/0 |

Grouped deltas (paired bootstrap over the same draws, restricted to the group's items):

| group | n | top-1 native / stock | Δ top-1 pp [95% CI] | McNemar p | I on group (Δ [CI]) | C on group (Δ [CI]) |
|---|---|---|---|---|---|---|
| COVERED | 120 | 80.8 / 87.5 | **−6.67 [−13.93, +0.02]** | 0.12 (6 vs 14) | 42.6 vs 54.1: −11.49 [−31.21, −1.61] | 61.8 vs 73.9: −12.06 [−25.28, +7.42] |
| UNCOVERED | 111 | 62.2 / 61.3 | +0.90 [−8.49, +9.62] | 1.00 (14 vs 13) | 39.2 vs 41.1: −1.84 [−19.0, +16.6] | 54.1 vs 54.1: +0.06 [−18.0, +13.1] |

**Reconciling with the lead's "covered −5.2 pp / uncovered +4.3 pp": DIFFERENT DEFINITION, so I DISAGREE on the label, not on the data.**
- Under plain argmax (noul p_yes ≥ 0.5 → yes), I get covered −5.0 pp and uncovered +4.5 pp. That is within an item or so of the lead's numbers.
- jevkit's Intelligence correctness counts a noul answer with 0.2 < p < 0.8 as an abstention, which scores as wrong. The trained arm abstains on **10/74 noul items**; stock abstains on 0. That accounts for the whole gap (noul Δ +1.4 pp under argmax vs −6.8 pp under jevkit; total top-1 172 vs 166).
- The jevkit definition is the one that feeds I and the Composite, so it is the one to quote. The abstentions are themselves a finding: the trained arm has gone soft on noul.
- Using the jevkit definition is my choice of what to quote. Choice-type items show the same −2.2 pp under both definitions.

## Length vs jev1 priors: split by prompt length (> 2,048 vs ≤ 2,048 input tokens under the shim)
Token counts come from the preds' `usage.input_tokens`, which are identical in both arms. Script: `scripts/ev_rescore_length.py`, output `length-xt3ax.json`. Paired bootstrap of Δ top-1 within each cell: iid item resample, B 1000, seed 16.

| cell | n | median tok | top-1 native / stock | Δ pp [95% CI] | McNemar p | choice NLL native / stock (Δ) |
|---|---|---|---|---|---|---|
| > 2,048 (all) | 36 | 2571 | 16 / 20 | −11.1 [−30.6, +8.3] | 0.42 | 1.296 / 1.008 (+0.288) |
| ≤ 2,048 (all) | 195 | 147 | 150 / 153 | −1.5 [−7.7, +4.1] | 0.73 | 0.543 / 0.520 (+0.023) |
| **multi_hop > 2,048** (UNCOVERED) | 17 | 2337 | 12 / 10 | **+11.8** [−11.8, +35.3] | 0.63 | 0.880 / 0.931 (**−0.051**) |
| **long_policy > 2,048** (COVERED; all 19 are long) | 19 | 2657 | 4 / 10 | **−31.6** [−58.0, 0.0] | 0.11 | 1.817 / 1.105 (**+0.712**) |

- The only items over 2,048 tokens are long_policy (19) and multi_hop (17). The "> 2,048" row is therefore exactly these two families combined.
- Below 2,048 the arms are flat: −1.5 pp under jevkit correctness, and +1.5 pp under argmax, since all 10 noul abstentions fall in this cell.
- **Verdict: length is NOT the cause.** multi_hop is just as long, is not covered by jev1, and *improves* on both top-1 and NLL. Only long_policy drops, and it drops hard (6 items lost, NLL +0.71). That points to the jev1 draw's long_policy templates and priors teaching a wrong answer pattern for this family.
- Caveat: the per-family cells are small (19 and 17 items), and neither family's CI excludes 0 on its own. The opposite signs at the same length are what carries the conclusion.

## EV-PREREG-COMPOSITE-LOCAL: trained `native` vs H2O-Lightning-4B v1.1 (our run, job-zbtbj, same 231 ids, same draws)
- H2O preds sha 60a06842…53cc, own price.json (base_model_reference 0.03/0.15, Qwen/Qwen3.5-4B). The rescore reproduces H2O's in-job report: I 77.56, C 83.42, S 91.31, K 61.13, Capability 80.49, A 76.64.

| | native | H2O | Δ (native − H2O) | 95% CI |
|---|---|---|---|---|
| Composite A | 63.43 | 76.64 | **−13.21** | [−37.82, −7.59] |
| Intelligence | 51.18 | 77.56 | −26.38 | [−37.98, −13.19] |
| Calibration | 64.69 | 83.42 | −18.73 | [−28.20, −8.50] |
| Capability | 57.93 | 80.49 | **−22.56** | [−31.95, −12.40] |
| top-1 | 166 | 205 | −39 | McNemar p 2.8e−9 |
| choice NLL | 0.689 | 0.307 | +0.382 | [+0.246, +0.529] |

The CI excludes 0 **against** us, so the trained arm is significantly worse than H2O on both Composite A and Capability. The gap is the same size on COVERED families (−14.2 pp top-1) and UNCOVERED families (−19.8 pp top-1). This leads to a question for the lead lane, which is theirs to answer (it is not a claim): what does H2O's training data contain that the jev1 draw lacks?

Caveat: these are open public items only, and the board's sealed split may rank differently. On H2O, the open→board I gap was measured at −17.6. None of these numbers are comparable to the board's 72.5.

## Files (this directory)
`rescore-xt3ax-full.json` (all blocks), `rescore-xt3ax.json` (plain primary run), `diag-floor-xt3ax.json` (floor diagnostic + two top-1 definitions, from `scripts/ev_rescore_diag_floor.py`), `length-xt3ax.json` (length split, `scripts/ev_rescore_length.py`), `reports/*.rescore.jevkit.json` (frozen-CLI report per arm and for H2O), and `screen.in-job.json` / `cell.in-job.json` (the in-job files being diffed against).
