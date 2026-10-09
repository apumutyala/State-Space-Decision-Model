# D13 vs D1: which losses can a readout change fix? (2026-10-09, agent-b, CPU only)

**Takeaway.** D13 loses 3.48 Index points on proxy600 across 11 families. Two parameter-free readout rules recover
**0.37 pts** of that: BFCL "exactly one tool" (+0.21) and SATA "top-m by yes-mass" (+0.16). Both rules come from the
task format, not from fitted values.

- **Prior/threshold shifts:** an in-sample oracle bias looks large (+5.90 pts summed; +3.07 if capped at each family's
  loss). Under 2-fold cross-fitting it falls to **+0.39 pts**. On families of 9 to 21 items, these biases mostly fit noise.
- **When2Call (the biggest loser, -1.08):** mostly a capacity loss. D13's ability to recognise the tool-call option
  dropped (one-vs-rest AUROC 0.87 to 0.59).
- **RAGTruth:** D13 scores its rows in the wrong direction (AUROC 0.18). That points to a polarity misread of the noul
  statement. A flip would be worth +1.18 pts on the proxy, but it is diagnostic only until a non-DEV check confirms it.
- **Input cap (not a loss vs D1):** 6 retrieval rows fail in both arms because the RWKV readout inherits Clef's
  16,384-token input cap. Lifting it is worth about +0.44 pts realistically (+0.98 oracle).
- **Significance:** no per-family loss has a paired CI that excludes 0.
- **Calibration:** worth 0 Index points here. Every losing family is scored on the argmax, and temperature scaling does
  not change the argmax. D13's ECE is already better: 0.024 vs 0.048.

> **Label: DIAGNOSTIC.** Every "UB" below is measured on proxy600 (the DEV rows, using their gold labels). It is a
> ceiling, not a result, and must not appear in the ledger as an Index gain. A lever becomes claimable only when two
> things are true: its parameters come from non-DEV data (or it is a parameter-free format rule), and the frozen screen
> is re-run.

## Inputs (receipts)
- **Rows:** proxy600, 644 rows, ids sha `7d42c398…`, edition 0.2.1.
- **Scorer:** the frozen `harness.score.decompose` + `score.point`. The counterfactuals only rewrite the prediction
  dicts fed to it. No frozen file was touched.
- **D13 child:** `d13_land/dl/x/final/out/screen/child.results.jsonl`, sha256 `7611368e…` (art-cr9a2).
- **D1 v3w reference:** `d13_land/work/ref.proxy600.results.jsonl`, sha256 `257aa5d4…`. This is the proxy subset of
  art-zskmx units `941af29e…`.
- **Baseline check:** the script reproduces child 41.07 and ref 37.71.
- **Script:** `analysis/readout_levers_d13.py`. Full numbers: `analysis/readout_levers_d13_2026-10-09.json`.
- **Methods:**
  - **Oracle prior:** a coordinate-ascent log-prob bias per shared label (grid ±4.0, step 0.1), or a noul threshold
    (0.02 to 0.98), chosen to maximise that family's skill. It is fit on the proxy gold, so it is an in-sample ceiling.
  - **Cross-fit:** the same fit run on half of the groups and scored on the other half, over 5 seeds × 2 folds. This is
    still DEV data, but it is a less optimistic estimate.
  - **Capacity residual:** child with its own oracle bias, minus ref with its own oracle bias.
- **Points:** Index pts = 100 × effective weight × Δskill.

## Per-family table (Index points on proxy600)

| family (n) | loss vs D1 | cause | lever | UB pts (oracle / x-fit) | proposed fix (non-eval params only) |
|---|---|---|---|---|---|
| When2Call MCQ (21) | **-1.08** | (d) **capacity**, plus a prior shift. Gold-class AUROC: tool_call 0.59 (ref 0.87), cannot_answer 0.73 (ref 1.00). Mean p(gold) 0.45 vs 0.63. D13 over-predicts D/cannot_answer (10 predicted vs 7 gold); ref over-predicted B (14 vs 8). Residual after both arms get an oracle bias: -0.86 | prior bias (C +1.7, B +0.4); tool-call option format | 0.86 / **0.18** | (1) Render tool-call options as text, e.g. "Call `temperature`(location="Paris, France")", instead of a raw JSON blob in `criteria`. This is a `decide/` prompt change; test it in a GPU screen. (2) Fit the per-class bias on `ssdm-data/m2/sources/when2call/when2call_train_sft.jsonl` (15k, non-DEV), scored by the D13 readout. |
| Humicroedit (9) | -0.46 | (a) **position prior + noise**. AUROC: child 0.33, ref 0.39, so neither arm discriminates. Ref "won" by answering headline_2 on 9/9 items in a sample where 6/9 are headline_2. D13 answers headline_1 on 7/9 | headline_2 bias +1.6 | 0.46 / 0.17 | Use order-symmetrised scoring: average over both presentation orders with `decide/debias.description_swap`/`neutral_cyclic`. It needs no parameters, and the training prior is 50/50 (humor_pair 2247/2253 in the 150k draw). Expected legitimate gain is about 0. Do not chase ref's always-B luck. |
| RAGTruth (15) | -0.42 | (b?) **likely polarity misread**. D13 AUROC is **0.18** (about 2 SE below 0.5; ref 0.55). Ref's skill comes from noul values packed into 0.49 to 0.72, which predicts True 13/15 (near all-positive, and F1-on-positive rewards that) | threshold (oracle 0.22, which is just the degenerate all-positive point); polarity flip | threshold 0.60 / 0.15; **flip 1.18 (diag)** | (1) Check polarity on the **RAGTruth train split**: 15,090 non-DEV responses in `m2/sources/dl-2026-10-08/ragtruth/response.jsonl`, prevalence 0.445. If D13 is inverted there too, rephrase the noul statement as a question, or present the "supported" polarity and invert. (2) An F1-aware threshold t = F*/2 with π_train = 0.445 gives t = 0.31. That is worth **0** for D13, because its ordering is inverted. |
| GPQA-Diamond (14) | -0.27 | (d) capacity/noise. Oracle +1.09 does not survive cross-fit (-0.10) | — | 1.09 / **-0.10** | none |
| VAST (15) | -0.27 | (d) noise. D13 never predicts C (3 gold) | — | 0.45 / **-0.19** | none |
| Amazon-ESCI (19) | -0.26 | (a) mild prior. Accuracy is equal (11/19 both); conservative F1 penalises D13's E/I over-prediction | E -1.7, I -0.4 | 0.62 / 0.10 | Fit the bias on the ESCI public train split (not on disk; would need a download). Low priority. |
| BFCL tool-selection (25 cases) | -0.21 | (b) **format**. Two single-call cases each get one extra false "yes" | **exactly-one rule** | **0.21 rule** (oracle 0.21) | Parameter-free. In the BFCL categories simple, multiple, live_simple and live_multiple (one gold call by BFCL's definition; parallel* excluded), answer yes only for argmax p(yes). It recovers D13 to 0.948, equal to ref. It changes 0 of ref's fields. |
| SATA-Bench (15 cases) | -0.16 | (b) **multi-answer handling**. 7 of the 10 failed cases miss by exactly 1 field. Errors are balanced (9 missed-yes, 8 false-yes), so this is not a prior | **top-m by yes-mass rule** | **0.16 rule** (oracle 0.49 / x-fit 0.06) | Parameter-free. Per case, select the m options with the highest p(yes), where m = round(Σ p(yes)) and m ≥ 1. D13 goes 0.324 → 0.392 (= ref). The rule also lifts ref 0.392 → 0.460, so it is a general readout gain. The "at least one yes" rule never fires (0 flips). |
| PhishNChips (19) | -0.16 | (d) capacity + prior. D13 answers "legitimate" 17/19 (12/19 gold phishing). AUROC 0.62 vs ref 0.87 | phishing +2.0 | 1.12 / **0.02** | none legitimate (no non-DEV PhishNChips data on disk) |
| CLINC150+OOS (23) | -0.14 | (d) capacity/noise: 2 items. D13's distributions are flatter (mean top-p 0.06 vs 0.15) | — (opaque option ids, no shared prior) | 0 / — | none |
| BANKING77 (23) | -0.06 | (d) 1 item | — | 0 / — | none |
| Home-Appliance (12 cases) | 0 (0/12 all arms) | (d) capacity at case level. Field accuracy 0.71 over about 18 fields per case. Systematic "execute now" prior: all 8 non-execute golds are predicted execute. Target over-inclusion: 36 false-yes vs 2 missed | joint-consistency decoding; resolution/target priors | oracle on resolution+outcome = **0**; any single oracle = 0.28 (1 case) | Parameter-free consistency decode: a non-execute resolution implies "no device property changes". Ceiling is +0.28 on the proxy. Low priority. |
| *not a loss:* BRIGHT (5/16 queries) + ToolRet (1/21) | 0 (same rows fail in both arms) | (b) **format/budget**. `input_tokens_exceeded`: 16.8k to 28.8k tokens > the 16,384 `max_input_tokens` in `decide/adapter.py` (labelled "Clef encode_record default") | lift the cap for the RWKV backend | oracle **0.98**; realistic **≈0.44** (UB × D13's mean per-query value on its ok queries: 0.42 BRIGHT, 0.71 ToolRet) | Set the RWKV backend `max_input_tokens` to the backend's real limit (RWKV is recurrent, so none for the state). Alternatively, score long retrieval rows per document from one prefilled state. First confirm that the board protocol does not require 16,384. |

**Calibration (c), per family.** The table gives top-label ECE10 as D13 / ref:

| family | D13 ECE10 | ref ECE10 |
|---|---|---|
| When2Call | 0.33 | 0.22 |
| Humicroedit | 0.53 | 0.24 |
| RAGTruth | 0.37 | 0.10 |
| PhishNChips | 0.40 | 0.37 |
| BFCL | 0.05 | 0.03 |
| SATA | 0.06 | 0.04 |

D13 is worse on the small losing families but better overall (board 0.087 vs 0.144). None of these families is scored
on probabilities, so a temperature or feature-temperature fix moves the Index by 0. It matters only for board ECE
reporting, and the existing `decide/calibration.py` feature temperature already covers that.

## What to ship (non-eval parameters only)
1. **`decide/` post-processing: BFCL exactly-one and SATA top-m.** Both are parameter-free. Together they are worth
   +0.37 on the proxy (diagnostic estimate), and they apply to every arm. Before claiming, confirm on the full DEV
   screen; no fitting is needed.
2. **Lift the RWKV input cap** from 16,384. Worth about +0.44 realistic, ≤ +0.98. Confirm the board/kit has no
   16,384-token requirement first.
3. **RAGTruth polarity check** on the train split (GPU readout on about 300 train responses). If D13 is inverted there
   too, change the noul phrasing. The diagnostic ceiling is +1.18.
4. **When2Call tool-call option rendering**, plus a bias fit on `when2call_train_sft` (non-DEV). Cross-fit suggests
   about +0.2 at most from the prior. The rest is capacity and needs training data: when2call is on disk but is not in
   the 150k draw.
5. **Do not ship any per-family bias fit on proxy/DEV.** The in-sample +5.9 shrinks to +0.39 under cross-fit, and
   GPQA, VAST and PhishNChips go to ≤ 0.

Bottom line for the D13-vs-D1 question: about **0.4 of the 3.5 lost points** are readout-fixable with rules that need no
eval fitting. Up to about **1.2 more** is possible if the RAGTruth polarity hypothesis holds on non-DEV data. The rest is
capacity or sampling noise (When2Call tool-call recognition, PhishNChips, GPQA, VAST, CLINC/BANKING), and every one of
those intervals includes 0.
