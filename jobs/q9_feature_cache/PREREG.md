# PRE-REGISTRATION: frozen-Q9 proxy head loop (PREMORTEM blocker #1)

Committed 2026-10-06, **before any feature cache exists**. No GPU job has run and no cached feature has been seen. The binding rule is the code in `train/proxy_rule.py` (`RULE_VERSION = "proxy-rule-v1"`); its constants are pinned by `tests/test_proxy_rule.py`. Amendments are dated, committed before the run they affect, and never edit v1 in place (LESSONS L-07).

## Question

Can an **attention-free** head on frozen Qwen3.5-9B prefix states approach Clef-flash before we pay for the full-8 conversion (S1, $28–74) and the first Q9-AF cell?

- Clef-flash: 55.78 on the 2,008-row DEV subset. Stock Q9 readout: 38.27.
- Feature arms: stock Q9, Q9 with layer 23 converted to GDN (art-2bybh `gdn_converted.pt`), and stock Q9 with E5 short codes for opaque option keys.

## Go / no-go

| Verdict | Condition |
|---|---|
| **go** | The attention-free recipe chosen on SELECT scores **≥ 50.0** Index points on HOLDOUT |
| **escalate** | Below 50.0. Report the measured head ceiling to the user **before any full-8 spend** |
| **no-measurement** | No attention-free candidate, an incomplete split, or a scoring failure |

Always reported, never verdict-changing:
- the SELECT score of every recipe;
- the selection gap (SELECT − HOLDOUT);
- paired CIs vs Clef-flash and vs stock Q9 on both splits;
- IN-domain vs NEW-domain breakdown;
- the attention reference head's score (the "how much does attention buy" number).

## Selection-overfitting risk, and the second split

The DEV subset (SELECT, 2,008 rows) is used to **choose** among 14 recipes. The chosen recipe's SELECT score is therefore biased upward. With an index SE of about 1.6 points and many correlated recipes, the winner's curse is plausibly 1–2 points. That is the size of the margin around the 50 threshold.

So the verdict reads **HOLDOUT**:
- 2,014 DEV rows, disjoint from SELECT by kit group.
- Same stratified construction, seed 20261008, `jobs/q9_feature_cache/dev_holdout.json`, ids sha256 `8ae9c9fb…`.
- Rebuilt and checked by `python -m jobs.q9_feature_cache.splits check`.
- HOLDOUT is scored once, for the chosen recipe only, with seeds 0 (verdict), 1 and 2 (spread).

Residual risks:
- Both splits come from the same 0.2.1 public DEV slice. Index 0.3's private same-skill and new-domain tests are not represented. The NEW-domain column is our only proxy for them.
- Clef-flash per-row results come from full DEV, so they cover HOLDOUT. Stock Q9 on HOLDOUT is recomputed from the cache's option log-likelihoods, after we check that the recompute reproduces job-5cmat on SELECT.

## Domain split

Superseded by amendment A1(a), before any data: the families-v1 columns `in_domain`, `new_domain` and `new_domain_frozen` (below). v1's hand-defined list (1, 4, 5, 12, 21, 28, 29, 30, 57, 62) is withdrawn. It missed 9 same-family benchmarks, so its new-domain column would have been optimistic.

## Recipes (anchor-then-ablate)

| Id | Recipe |
|---|---|
| R00 | Clef-like ANCHOR: attention routing head on stock features, clef loss. **Reference only, never a candidate** |
| R01 | Full attention-free mixer |
| R02–R06 | One-factor ablations of R01: LM log-prob (E4), key class, evidence routing, field mixer, lexical prior |
| R07–R08 | R01 with loss CE, or LS-CE+Brier without RPS |
| R09 | R01 on E5 features |
| R10 | R01 on 1-layer-converted features |
| R11–R13 | m2 fork MLP head: clef loss, CE, converted features |

Every recipe gets the same steps budget and seed. There is no early stopping on DEV.

## Proxy → real transfer

Adopt the proxy ranking for the S2 cell only if all three hold:
1. At least **4** recipes have both a proxy score and a real head+LoRA cell score.
2. Kendall τ-a ≥ **0.6** over those recipes.
3. On the one real-cell calibration pair (the proxy's best and worst recipe), the real cells order the pair the same way the proxy does.

Otherwise the ranking is a prior only.

## Attention-free definition (disclosure)

A head is attention-free if it has no softmax-normalised, or otherwise normalised, weighting over sequence positions or over chunk summaries of positions.

Mixing over the fixed set of option and field vectors of one request is allowed and disclosed. It is O(Q·K), not O(T).

Family (c), the attention routing head, violates this definition by construction. It is labelled `attention_free = False` and is excluded by `CANDIDATE_FAMILIES`.

## Amendment A1 (2026-10-07, before any feature cache exists; no GPU job has run)

The coordinator requested this after the independent review of PR #101 (REQUEST-CHANGES). The request, quoted:

> Pre-data amendment A1 [...]: (a) Replace the hand-defined IN_DOMAIN_CATALOG_IDS with agent-main's harness/families.py families-v1 split. Report three columns: in-domain, new-domain, and new_domain_frozen (the 9 frozen benchmarks). (b) Equalise optimiser steps across arms, or else state that conv1/e5 comparisons use a matched-steps stock arm. (c) Pin the verdict to edition 0.2.1 and report 0.3 alongside. (d) Declare the post-job stock-check tolerance now: |Δ| ≤ 0.3 against job-5cmat's 38.27 on SELECT. (e) State that "attention-free" refers to the head; the stock features come from a backbone with attention. Keep the go threshold at 50.0 on HOLDOUT, with escalation to the user, unchanged.

| Item | Rule after A1 (`train/proxy_rule.py`, pinned in `tests/test_proxy_rule.py`) |
|---|---|
| (a) Domain columns | families-v1 (`harness/families.py`, sha256 `e0e4bd5f…`), evaluated in edition 0.2.1. `in_domain` (19 benchmarks: same_source + same_family): 1, 2, 3, 4, 5, 11, 12, 21, 25, 28, 29, 30, 36, 37, 39, 42, 45, 57, 62. `new_domain` (19, the 'none' tier): 9, 20, 22, 23, 31, 32, 33, 38, 40, 41, 43, 44, 48, 50, 56, 58, 59, 61, 64. `new_domain_frozen` (heldout-draw-v1, 9): 9, 20, 22, 31, 32, 33, 44, 58, 64 |
| (b) Training budget | Every recipe trains exactly 2,000 optimiser steps of 8 requests, on every arm (was 3 epochs). The arms still differ in data: stock has 20,000 rows, conv1/e5 the first 5,000 rows of the same draw. So R09/R10 (conv1/e5 mixer) are compared only with **R14** (R01 on the stock arm's first 5,000 rows), and R13 (conv1 fork) only with **R15** (R11 on those 5,000 rows). Matched data, matched steps |
| (c) Edition | The verdict, the SELECT choice and the threshold use edition **0.2.1**. Every score is also reported under 0.3's public index, scored without the 0.3 GSM8K twin rows, which the 0.2.1-row cache does not hold |
| (d) Stock check | Before any recipe trains, the stock readout is recomputed from the cached option log-likelihoods on SELECT. If it differs from job-5cmat's 38.27 by more than **0.3** Index points, the whole loop is `no-measurement` |
| (e) Scope | "Attention-free" refers to the **head**. The stock features come from Qwen3.5-9B with all 8 softmax-attention layers; conv1 converts 1 of 8. A "go" bounds what an attention-free head can reach on these features. It is not a Q9-AF measurement. The verdict record carries this `scope`, and the conv1 − stock gap is reported next to it |
| Unchanged | GO = 50.0 on HOLDOUT; below it, **escalate to the user**; SELECT/HOLDOUT; the transfer rule; HOLDOUT is single-shot. All candidate recipes, now R01–R15, must have a SELECT score before the choice, and the holdout runs once, for the chosen recipe only |

## Amendment A3 (2026-10-07, written AFTER the stock-check failed with Δ=+0.772; see receipts/2026-10-07-D-q9-proxy-stock-check; before any recipe, select or holdout data)

**Disclosure.** This amendment is written after a data-bearing step. The A1(d) stock-check ran on the cached SELECT split and FAILED: the cached stock readout scored 39.04 against job-5cmat's 38.27, Δ = +0.772 > 0.3. No recipe has been trained or scored, and no select_summary, HOLDOUT attempt or HOLDOUT score exists. A3 changes a validity check of the cache, not the go/no-go rule.

**Why the check failed** (decomposition recomputed by the independent review of PR #130; every variant rescored with the kit). All 2,008 SELECT ids match. job-5cmat did not answer 65 rows, which the kit scores as wrong; the cache answers all 65:

| Reference rows not answered | Cause | Δ with only these masked | Share of the gap |
|---|---|---|---|
| 7 `unsupported` (3 ToolRet + 4 BRIGHT) | longest model input 17,126–28,497 tokens, over the reference's `max_input_tokens` 16,384 (no truncation); the cache's cap is 32,768. An **input-limit policy difference, not a bug** | +0.105 | ≈ +0.67 |
| 58 `error` (POP909, backend_error) | believed to be the GPU OOM root-caused in #102; the fix is **not yet confirmed on GPU** (L-21 stays partial) | **+0.714 (would still FAIL)** | ≈ +0.06 |
| all 65 (A3) | | **+0.047** (paired vs job-5cmat +0.048, CI −0.79…+0.72) | |

Masking only the bug rows (`error`) was a plausible alternative, and it fails. The `unsupported` rows are masked as well because the reference has no answer on them: comparing an answer to a non-answer measures the serving protocol (the input cap), not feature fidelity, which is what A1(d) tests. The masking criterion is the reference's own row status, never the cache's score. The cache's answers on the 65 rows sit at chance level like the rows around them; they are not favourable (POP909 2/58; BRIGHT 3.1%, the same as its ok rows).

**Not validated by this check:** rows whose model input is 16,384–32,768 tokens. The reference cannot answer them, so no stock-check can. On SELECT these are exactly the 7 `unsupported` rows; on HOLDOUT there are 10. The HOLDOUT report states them (below).

| Item | Rule after A3 (`train/proxy_rule.py`, pinned in `tests/test_proxy_rule.py`) |
|---|---|
| (a) Rule | The stock-check compares the cached readout to the reference **only on rows the reference answered with status `ok`**. Any other status (`error`, `unsupported`, …) is masked from both sides: the cached results carry the reference's own row for that id, so it counts identically in the cached index and in 38.27. Δ therefore measures only the rows the reference answered. The tolerance is **unchanged: \|Δ\| ≤ 0.3** |
| (b) Pins | The reference is job-5cmat's stock results file, sha256 `646b6149…7beb`, holding exactly the SELECT ids (none missing, extra or duplicated). The masked set is pinned too: 65 ids, sha256 `d3465d44…6629` (listed in the receipt's `masked_ids.json`). Any other set fails closed |
| (c) Failure cap | The check **fails closed** if more than **5%** of SELECT rows are masked |
| (d) Masked rows must be valid | Every masked id's **cached** row must be status `ok` with well-formed answers (choice: finite, non-negative probabilities summing to 1 within 0.01, with the choice among them; noul: finite in [0, 1]). Otherwise the check **fails closed**, so masking cannot hide a cache crash, NaN or unanswerable row |
| (e) Reporting | `stock_check.json` reports the masked Δ (decisive), the unmasked Δ, the masked count, fraction, ids and their sha256, the counts per reference status and each group's contribution (unmasked point − point with only that group masked), any masked-row problems, and both paired comparisons with job-5cmat |
| Report only | Every receipt lists its rows whose cached model input exceeds 16,384 tokens. The HOLDOUT verdict reports how many there are and how the chosen head, the stock readout and Clef-flash scored them: status on those rows, index with and without them, and the head-vs-Clef-flash paired difference without them. Never part of the verdict |
| Unchanged | GO = 50.0 on HOLDOUT; below it, **escalate to the user**; STOCK_CHECK_TOL 0.3; the 38.27 reference; splits, editions, budget, seeds, recipes R01–R17, the teacher, the transfer rule; HOLDOUT is single-shot. A3 is added to `AMENDMENTS`, so receipts and select_summary record it |
