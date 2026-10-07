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

- **IN_DOMAIN** catalog ids: 1, 4, 5, 12, 21, 28, 29, 30, 57, 62. These are the benchmarks whose own train split, or a proxy named for them in `data/SOURCES.md`, is in the 150k mix.
- **NEW_DOMAIN:** the other 28 DEV benchmarks.
- Defined from harness catalog ids. The Index-0.3 migration branch's family definition was not on origin at commit time. If it lands, it is reported alongside this split.

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
