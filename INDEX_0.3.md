# Decision Index 0.3: verified facts, what changed for us, and the working retarget

Written 2026-10-06 (Track D), amended 2026-10-07. Everything below was fetched from primary sources on 2026-10-06 between 17:20 and 17:45 UTC. Hashes and URLs are in `receipts/2026-10-06-D-index03-rescore/README.md`. `GOAL.md` and `PLAN.md` are not edited. **The section 4 reading was adopted 2026-10-07 by the coordinator under the operator's delegation; the operator may override it.**

## 0. Summary

1. **0.3 is live on the board but not yet in the kit.**
   - The board's data files are 0.3: `data/v03.json`, `data/index.json` and `data/methodology.json`, with changelog date 2026-10-06.
   - The kit change is still an open, unmerged PR (#74). Kit `main` still says "The live board is Decision Index 0.2.1".
2. **Every PR #92 number checks out**: Clef 53.08 (rank 19), Clef-flash 47.61 (=31), Jev 60.11 (=2), Nimble-9B-v3 54.67, and the 20/50/30 weights.
3. **Three claims need correcting or qualifying:**
   - The private parts are *equated* before weighting. Each one is linearly mapped onto the public scale using 21 untuned stock models. So the 20/50/30 weights act on rescaled scores, not on the raw private scores.
   - GSM8K's flaw was that the gold was the *middle value* among the sorted options, not the centre *position*.
   - "No prompt tuning / one fixed rendering" is not new in 0.3. It was already a 0.2.1 rule.
4. **Paired CIs on 80% of the Full score are impossible from published data.**
   - The maintainers run the private parts themselves on the submitted model.
   - The private sets are never published, and only aggregate scores appear on the board.
   - The board publishes no per-row outputs for any entrant, public or private.
5. **Our $0 re-score.**
   - On the 0.3 public index (DEV, GSM8K excluded until its rebuilt rows are run), Clef-flash scores **56.06 [54.61, 57.59]** on full DEV (0.2.1: 56.79).
   - On the m1 subset, the 1-layer GDN conversion stays above stock: **+1.52 [+0.04, +3.02]**.
   - All three of our backbones remain 14–29 points below Clef-flash.
6. **New held-out-family split (families-v1).**
   - 18 of the 37 0.3 benchmarks have no training coverage in our 150k mix. They are the local "new-domain proxy".
   - Clef-flash's DEV in-domain minus proxy gap is +10.9 [6.9, 14.1]. Untrained stock Qwen3.5-9B's gap is even larger (+16.3), so the proxy benchmarks are also intrinsically harder.
   - The usable signal is therefore the *paired change vs the parent* on each tier, not the raw gap (section 5).

## 1. Verified / unverified table

V = verified verbatim in a primary source. V* = verified, but PR #92's wording needs a correction. U = not verifiable from public sources.

| # | Claim (PR #92) | Status | Primary source and what it says |
|---|---|---|---|
| 1 | 0.3 went live 2026-10-06 | **V** (board) / kit pending | `v03.json`: `"edition":"0.3"`, `generated_utc 2026-10-06T16:32:56Z`, sha256 `9f8a0a03…01af`. `methodology.json` changelog: `{"edition":"v0.3","date":"2026-10-06"}`. Space last modified 2026-10-06T17:39:36Z (commit `6ceab83`). Kit PR #74 "Edition 0.3" opened 16:11Z and is **open, not merged**. Kit `main` (`87d4650`) README still says 0.2.1 is live. |
| 2 | Ranking = 20% public + 50% private same-skill + 30% private new-domain | **V*** | `v03.json` `weights {public 0.2, same_skills 0.5, new_domains 0.3}`. **Missing from PR #92:** `formula.equating` maps each private part onto the public scale first: eq(x) = 23.5 + 16.54·(x − μ)/σ, with (μ, σ) = (24.27, 16.35) for same-skills and (23.68, 12.29) for new-domains, over 21 "bases" (untuned stock models). The kit PR README says the same in prose. The equated part enters *unclipped*; the board displays it floored at 0. `harness/full03.py` reproduces all 112 entrants' Full score to ≤0.015 and every tie rank (tested). |
| 2a | Ties | **V** (new) | "Models within 0.9 points of the first model in their group share a rank" (methodology changelog; `tie_band 0.9`). In 0.2.1 the tie band was 0.25 on the public index. |
| 3 | Clef 53.08, rank 19 | **V** | `v03.json`: pub 61.71, same 53.11, new 41.88, v 53.08, `rank_tie 19`, not tied. |
| 4 | Clef-flash 47.61, =31 | **V** | pub 56.15, same 44.01, new 42.50, v 47.61, `rank_tie 31`, tied. |
| 5 | Jev 60.11, =2 | **V** | pub 57.96, same 58.04, new 54.98, v 60.11, `rank_tie 2`, tied (`rank_v 3`; #1 is Perplexity Decider v1.1 at 62.75). |
| 6 | Nimble-9B-v3 54.67 | **V** | pub 57.19, same 52.34, new 49.05, v 54.67, rank 18. It is above Clef on Full by 1.59, outside the 0.9 tie band. |
| 7 | ForecastBench retired | **V** | `v03.json` `public_index_note` "GSM8K rebuilt, ForecastBench retired". PR #74 `index-0.3.json` drops 48 from Arts and empties `loss_rules`; `editions.py` has `retired=(48,)`. |
| 8 | GSM8K rebuilt, "gold was always the centre option" | **V*** | Issue #32 (open): the gold's *rank among the sorted option values* was fixed. In GSM8K-10 it was always the 4th–6th smallest ("pick the 5th smallest" alone scores 57%). In GSM8K-4 it was 2nd/3rd smallest in 82% of rows. The display position was shuffled. Rebuild (PR #74 README): "every wrong option is now the answer to a different GSM8K test problem of similar size". There are 2,638 new requests with run ids prefixed `v0.3-gsm8k:`, and the chance levels are unchanged. |
| 9 | WinoGrande moved to Knowledge | **V** | `index-0.3.json` and the live `index.json` categories: knowledge = [25, 30, 31, 32, 33, 43, 44, 45, 57, 58, **28**]. |
| 9a | Area weights | **V** (new detail) | Fixed at the 0.2.1 values (K 25.8494%, L 25.8494%, R 20.0229%, T 18.2783%, A 10%) with `rule: "fixed"`. They are no longer √n, even though K now has 11 benchmarks and L 9. |
| 10 | Latency cap 1,000 ms on median, mean and p80 | **V** (kit PR only) | PR #74 README: "median, mean or 80th-percentile latency is over 1,000 ms … are not added". The Space `methodology.json` latency section still describes median/p95 reporting and does not restate the cap. In 0.2.1 the cap was on the median only. The board shows `mean` for 0.3 entrants (Clef 245.5 ms, Clef-flash 149.8 ms). |
| 11 | House rules: no prompt tuning, one fixed rendering | **V*** (not new) | Both lines are in kit `main` 0.2.1 README §Rules: "No prompt tuning. The reference engines use one fixed rendering for every benchmark". They also appear in the board's `hard_rules`: "Each entrant runs with its own native inference helpers, prompts and readout rules as published". The rule binds *us* as "no per-benchmark prompts". "One fixed rendering" describes the reference engines. **No new house rule in 0.3.** |
| 12 | Who runs the private sets | **V** | PR #74 README: "the maintainers run the submitted model themselves on the two private parts … exactly as you did … weights on the Hub (public, or access granted to the maintainers on request), the inference code you used (or a `/v1/systemone` server they can start), and its settings". Submitters run and submit the public 0.3 suite; the run is re-scored on review. |
| 13 | Per-row outputs published? | **V (no)** | "The private sets are never published or shared, with submitters or anyone else, and they appear on the board only as each model's aggregate scores." The board's public data has only per-benchmark aggregates per entrant; we found no per-row outputs for any entrant. |
| 14 | Paired CI vs Clef from published data? | **V (no)** | Private: impossible (13). Public: only by running Clef / Clef-flash ourselves on the same public rows, as job-sgihj did for Clef-flash on DEV. Board CI: none. The tie band (0.9) is the board's only uncertainty statement: "a gap that small could come from which questions happened to be sampled". |
| 15 | Clef-flash "lab run" / ECE 0.0395 (Clef) / latency 102.1 / 53.4 ms | **V** (numbers) | Live `index.json`: Clef ECE 0.0395, median 102.1 ms; Clef-flash median 53.4 ms. `lab` is `false` for every v03 entry, so "lab run" is unclear wording, not a data field. |
| 16 | Private same-skill / new-domain *content* | **U** | Only "Private tests of equivalent competence for most public tasks, built from fresh data … plus a set of private decision tasks from new domains". Which tasks and domains, their sizes and their formats are not published. |
| 17 | Per-benchmark private scores | **U** (undocumented field) | Live `index.json` carries an undocumented per-benchmark `private_eq` (34 of 37 benchmarks; absent for 11, 23, 45) and `combined`. Chance-correcting `private_eq` and aggregating with the public weights lands within 0.7 of each entrant's same-skills score (Clef 52.43 vs 53.11; Clef-flash 43.58 vs 44.01; Nimble 51.85 vs 52.34). So it is *probably* a per-benchmark private same-skill score, but its definition is not published. Use it for direction only (section 3.3), never as a target. |

## 2. What changed vs 0.2.1 for our harness and GOAL

| Area | 0.2.1 | 0.3 | Harness action (this PR) |
|---|---|---|---|
| Headline | public index (38 benchmarks) | **Full** = 0.2 pub + 0.5 eq(same) + 0.3 eq(new) | `harness/full03.py` (arithmetic only; private parts come from the board) |
| Public panel | 38 | 37: −ForecastBench; GSM8K rows replaced; WinoGrande K | `config.EDITIONS`, `suite.spec(edition)`, `suite.in_edition`, vendored `harness/editions_data/index-0.3.json` (sha `158c3843…`) |
| Area weights | √n with Arts fixed at 10% | the same numbers, now `fixed` | from the spec; tested |
| GSM8K | 2,638 rows (center-value artifact) | 2,638 rebuilt `v0.3-gsm8k:` rows | **Rebuilt locally** (CPU, $0) with kit PR #74's `gsm8k_v3.build` from the pinned GSM8K test parquet (`openai/gsm8k@740312a`, sha256 `ee7b8da9…`): byte-identical to the board's file (uncompressed sha256 `75933489…`). **Slices amended** (`harness/gsm8k03.py`): each slice gets the rebuilt twins of its own GSM8K rows (run id `v0.3-gsm8k:` + the 0.2.1 id; same id, group, track and state), a pure ID mapping of the committed manifests. DEV 262 / TEST 262 rows, disjoint, no source overlap. 58 answer groups (shared *numeric* option sets) span both sides, because 0.3 distractors are other test problems' golds; no question text is shared. The TEST addendum is IDs and hashes only and is never materialised; TEST loads keep the unseal gate. `load_slice(name, edition="0.3")` includes the twins; `--omit-gsm8k03` leaves them out for old result files that never ran them. The section 3 re-score omits GSM8K for exactly that reason. |
| Ties | 0.25 on public | 0.9 on Full | `full03.ranks` |
| Latency gate | median ≤ 1 s | median, mean **and** p80 ≤ 1 s | none needed in the harness. The m3 serving-latency report must now give mean and p80. |
| Submission | public run + PR line | the same, plus model repo on the Hub, inference code or a `/v1/systemone` server, and settings, because the maintainers re-run privately | none. This is an outward-facing step and needs the user's explicit approval at the time. |
| ECE note | 32 benchmarks, "ToolRet, BRIGHT, RouterBench and ForecastBench excluded" | the same note without ForecastBench | unchanged. `board_ece` was already flagged not board-comparable. |

The default stays **0.2.1**, so every existing receipt reproduces the same numbers. The score/compare JSON now also carries an `"edition"` key, so the output bytes differ by that key only. `--edition 0.3` is opt-in on `python -m harness score|compare` and `python -m jobs.common.dev_subset score|compare`. Replicates carry their edition: `summarize_one`, `families.summarize` and `families.compare` take it from the replicates and raise on any disagreement, so there is no silent 0.2.1 default. Tests: `harness/tests/test_edition03.py`, `test_gsm8k03.py`, `test_heldout.py` (148 tests pass across `harness/tests` and `jobs`). They check:
- the 0.3 public index against the kit fixture for 11 entrants and the live board for Clef and Clef-flash;
- Full and tie ranks against all 112 live entrants;
- DEV-slice parity between the official kit path and our decomposition under 0.3;
- refusal to pair across editions.

**Effect on PLAN.md (not edited).** T1/T2/T3 are stated on 0.2.1 public. Under 0.3:
- Nimble (T1's bar) is 54.67;
- Jev (T2) is 60.11;
- Clef-flash (T2) is 47.61;
- Clef (T3) is 53.08.

Beating Clef is now a Full-score question, and Full is 80% private.

## 3. Re-score of existing per-row predictions ($0, CPU)

Receipt: `receipts/2026-10-06-D-index03-rescore/` (`rescore.py`, `rescore.json`). Paired cluster bootstrap B = 2,000, seed 0, source-case clusters. The 0.3 column is the **0.3 public index restricted to DEV, without GSM8K** (36 of 37 benchmarks; rebuilt rows not run). Sanity check: the 0.2.1 subset numbers reproduce the earlier receipts exactly (stock 38.27, Clef-flash 55.78).

### 3.1 Full DEV slice (12,104 rows in 0.2.1; 11,453 count in 0.3)

| System | 0.2.1 DEV index | 0.3 public DEV index (no GSM8K) | K / L / R / T / A (0.3) |
|---|---|---|---|
| Clef-flash (job-sgihj) | 56.79 [55.40, 58.22] | **56.06 [54.61, 57.59]** | 54.8 / 48.8 / 48.1 / 80.8 / 48.9 |

For orientation, the board's full-suite public 0.3 score for Clef-flash is 56.15. The DEV estimate sits within its CI, but it is a different row set and omits GSM8K.

### 3.2 m1 DEV subset (2,008 rows; 1,900 count in 0.3)

| System | 0.2.1 subset | 0.3 public subset | Δ vs Clef-flash, 0.3 (paired) |
|---|---|---|---|
| Clef-flash | 55.78 [52.99, 58.52] | 55.00 [52.04, 57.92] | — |
| Qwen3.5-9B stock (job-5cmat) | 38.27 [34.97, 41.40] | 38.96 [35.67, 42.11] | −16.04 [−19.59, −12.60] |
| Qwen3.5-9B, 1 attn layer → GDN (job-5cmat) | 39.93 [36.48, 42.90] | 40.48 [36.99, 43.50] | −14.52 [−18.23, −10.94] |
| RWKV7-G1k-13.3B stock (job-tkfaj) | 24.86 [22.44, 27.82] | 25.57 [23.02, 28.53] | −29.43 [−32.76, −25.48] |
| *1-GDN − stock (paired)* | +1.66 [+0.19, +3.18] | **+1.52 [+0.04, +3.02]** | |

Takeaways:
- The edition change moves our numbers by under 1 point. The ranking of our arms and their gaps to Clef-flash do not change.
- The 1-layer conversion's small gain survives 0.3, with the CI barely above zero. It is concentrated on the new-domain proxy: +2.59 [+0.53, +4.63] there, vs +0.37 [−1.91, +2.97] in-domain.

### 3.3 What the board says that DEV cannot

Per-benchmark public raw score vs the undocumented `private_eq` (live `index.json`; direction only, see item 17):

| Benchmark | Clef-flash pub raw / private_eq | Clef | Nimble-9B-v3 |
|---|---|---|---|
| Home appliances | 0.977 / 0.570 | 0.807 / 0.693 | 0.989 / 0.830 |
| WinoGrande | 0.976 / 0.751 | 0.935 / 0.698 | 0.864 / 0.731 |
| CRUXEval | 0.865 / 0.557 | 0.865 / 0.633 | 0.647 / 0.664 |
| MMLU-Pro | 0.652 / 0.350 | 0.660 / 0.470 | 0.601 / 0.485 |
| BANKING77 | 0.908 / 0.648 | 0.941 / 0.877 | 0.837 / 0.840 |
| GSM8K (rebuilt) | 0.467 / 0.572 | 0.599 / 0.662 | 0.590 / 0.681 |

If `private_eq` is what it appears to be, Clef-flash's largest public advantages (Home appliances, WinoGrande, CRUXEval) mostly do not carry over to the private same-skill tests. That fits PR #92's "format supervision" hypothesis but does not prove it.

## 4. Retarget: working interpretation (adopted 2026-10-07 by the coordinator under the operator's delegation; operator may override)

GOAL.md and PLAN.md are untouched. This is the working reading of GOAL.md's "the edition current at submission".

GOAL.md: "beats every Cloudflare Clef model … on the full Decision Index 0.2.1 (or the edition current at submission) … a paired confidence interval for candidate-minus-Clef that sits above zero". Under 0.3, 80% of the score is private and unpublished, so that sentence can only be met in parts.

**Proposed reading (three gates; a candidate must pass all three):**

1. **G-pub: public portion, paired.**
   - What: candidate − Clef and candidate − Clef-flash on the **0.3 public index**, paired cluster bootstrap, 95% CI above zero.
   - Rows: our own runs of both Clef models. Pre-registered on DEV; the final number on the untouched TEST slice once.
   - Ready now: `--edition 0.3` exists, and the Clef-flash DEV run exists. Still needed: Clef-27B DEV/TEST runs and the rebuilt GSM8K rows.
2. **G-proxy: private-part proxies, paired, local.**
   - What: the same paired CI must also sit above zero on the two local proxies below, reported separately.
   - Why: in-domain public wins are cheap and the board discounts them. Clef-flash loses 8.5 points from public to Full, Nimble loses 2.5.
   - **Same-skill proxy** = fresh items for the same skills that no public suite row and no training row comes from. Options, cheapest first:
     - (a) the **DEV "same_family" tier** (families-v1: 11 benchmarks whose task family, but not dataset, we train on);
     - (b) **re-generated items** from public generators for the generator-backed benchmarks (CLadder, MuSR, CRUXEval-style executed functions, GSM8K-style rebuilt options, Home-appliances-style device records), drawn fresh with a held-out seed. This needs a pinned generator receipt and our dedup guard.
   - **New-domain proxy** = the **families-v1 "none" tier** (18 benchmarks in 0.3 with zero coverage in our 150k mix). Better still: a frozen **held-out task-generator set** we never train on (for example 3–5 decision formats absent from both the Index and our data).
   - Every candidate reports in-domain and new-domain-proxy scores and their paired Δ vs the parent and vs Clef-flash.
3. **G-full: the board.**
   - What: after G-pub and G-proxy pass, and only with explicit user approval at that moment, submit to the board. The maintainers' Full score is the only measure of the private 80%.
   - Win condition on the board: Full above Clef (53.08) by more than the 0.9 tie band, i.e. **≥ 53.98**, and the same for Clef-flash (≥ 48.51).
   - Honesty clause: the board gives no CI and no per-row data, so the private part of the claim is "beats Clef on Full by more than the board's tie band". It is not "paired CI above zero". That must be stated exactly that way.

**What Full scores the gates imply** (from `full03`; S = private same-skills raw, new-domains assumed S − gap):

| Target | Public 50, gap 0 / 5 | Public 56, gap 0 / 5 | Public 60, gap 0 / 5 |
|---|---|---|---|
| > Clef by tie band (53.98) | S ≥ 51.7 / 53.9 | S ≥ 50.4 / 52.6 | S ≥ 49.5 / 51.7 |
| > Clef-flash by tie band (48.51) | S ≥ 45.7 / 47.9 | S ≥ 44.4 / 46.6 | S ≥ 43.5 / 45.7 |
| > Jev by tie band (61.01) | S ≥ 59.4 / 61.6 | S ≥ 58.1 / 60.3 | S ≥ 57.2 / 59.4 |

Each raw private point is worth about 0.51 Full points (same-skills) or 0.40 (new-domains). A public point is worth 0.20. The private same-skill score is the lever. Nimble shows the profile that wins: public 57.2, same 52.3, new 49.1. Clef-flash's own private profile (44.0 / 42.5) is what we must beat.

**Restated T1–T3 against the 0.3 board** (PLAN.md's tiers keep their comparators; numbers from `v03.json`; "beat" = Full above the comparator by more than the 0.9 tie band, plus G-pub and G-proxy passed locally):

| Tier | PLAN.md (0.2.1) | 0.3 restatement | Full needed |
|---|---|---|---|
| T1 | First attention-free entrant; beats the best non-Clef 9B open entrant (Nimble-9B-v3, 56.88); ECE in the board's top 3; steer rate reported | First attention-free entrant; beats the best 9B entrant on Full, which is still Nimble-9B-v3 (54.67, rank 18; the next 9B is JevK5-9B at 48.35); ECE and steer rate reported as before | ≥ 55.57 |
| T2 | Beats Jev (57.91) and Clef-flash (≈57.07) | Beats Clef-flash (47.61, =31) **and** Jev (60.11, =2) on Full | ≥ 48.51 and ≥ 61.01 |
| T3 | Beats Clef (61.98, pending) | Beats Clef (53.08, rank 19) on Full | ≥ 53.98 |

What the reorder means: on 0.3, Clef (53.08) sits *below* T1's Nimble bar (54.67), and Jev sits above all of them. So T1 implies T3 and T2's Clef-flash half. The hardest gate is now Jev, in T2. The tiers are restated literally and not reordered; reordering is a PLAN.md change and is left to the operator.

**Resolved by the coordinator (2026-10-07):**
- (i) The three-gate reading above is adopted.
- (ii) The held-out families are split by a seeded draw, not frozen wholesale (section 5.1).
- (iii) T1–T3 are restated as above.

Still open for the operator: whether to reorder the tiers in PLAN.md.

## 5. Held-out-family split (E1), as implemented

`harness/manifests/families-v1.json` and `harness/families.py`. Tiers are relative to *our* 150k training mix (`data/manifests/sample-150000-seed20261006.pointer.json`). For Clef-family models, whose training data is unknown, the split shows which benchmarks *we* can and cannot claim to generalise to. It says nothing about *their* contamination.

| Tier | 0.3 benchmarks |
|---|---|
| same_source (8) | GSM8K 30, WinoGrande 28, ANLI 12, HellaSwag 29, BANKING77 4, CLINC 5, When2Call 62, Humicroedit 21 |
| same_family (11) | MMLU-Pro 57, GPQA 25, HLE 45 (academic MCQ via mmlu_aux); ContractNLI 11, NLI4CT 42 (NLI); FinEntity 39 (sentiment); BRIGHT 36, ESCI 37 (relevance via NQ); BFCL 1, API-Bank 3, ToolRet 2 (function selection via Glaive/ToolACE) |
| **none = new-domain proxy (18)** | ChessBench 31, MuSR 32, SATA 33, CRUXEval 43, CLadder 44, BBH 58; ACOS 38, iSarcasmEval 40, VAST 41, RAGTruth 59; PhishNChips 56, HoVer 61; Home appliances 9; BPoMP 20, POP909 22, cfcolor 23, Habermas 50, New Yorker 64 (ForecastBench 48 in 0.2.1 only) |

Measured on DEV (0.3 public, no GSM8K):

| System | in-domain | new-domain proxy | in − proxy |
|---|---|---|---|
| Clef-flash (full DEV) | 60.47 [58.58, 62.02] | 49.56 [46.86, 53.05] | +10.90 [6.90, 14.06] |
| Q9 stock (subset) | 43.98 | 27.65 | +16.34 [8.77, 23.86] |
| Q9 1-GDN (subset) | 44.35 | 30.24 | +14.12 [7.03, 22.29] |
| RWKV7-13.3B (subset) | 28.14 | 18.20 | +9.93 [4.34, 16.24] |

How to read it:
- The untrained stock parent already shows a large in − proxy gap, so the proxy benchmarks are harder in themselves.
- A candidate's *generalisation* is therefore its paired gain over its own parent on the proxy, compared with its gain in-domain (`families.compare`).
- A training change that raises in-domain but not proxy is the 0.3 failure mode, and does not count.

### 5.1 Frozen vs generatable half (heldout-draw-v1, adopted 2026-10-07)

The 18 "none" benchmarks were split by a seeded draw, committed before any generator data existed (`harness/manifests/heldout-draw-v1.json`, sha256 `c3172003…`; reproduced by `harness.heldout.draw` and tested).
- Seed **20261007**.
- Procedure: key(n) = sha256(`heldout-draw-v1:{seed}:{n}`) for each catalog id n; sort ascending; the first 9 are frozen.

| Half | Benchmarks |
|---|---|
| **FROZEN** (new-domain proxy; never targeted by training data or generators) | Home appliances 9, BPoMP 20, POP909 22, ChessBench 31, MuSR 32, SATA-Bench 33, CLadder 44, BBH 58, New Yorker 64 |
| GENERATABLE (PR #92 E6 may add generated data) | cfcolor 23, ACOS 38, iSarcasmEval 40, VAST 41, CRUXEval 43, Habermas 50, PhishNChips 56, RAGTruth 59, HoVer 61 |

- The draw put CLadder and Home appliances in the frozen half and CRUXEval in the generatable half. That is binding for E6.
- `harness.heldout.check_record` / `check_generator` (and `python -m harness.heldout check FILE`) refuse any record or generator whose tags name a frozen benchmark, catalog id, family or alias. The check is fail-closed.
- The existing 150k build passes: 0 of 150,000 records refused (`receipts/2026-10-06-D-index03-rescore/heldout-check-150k.json`).
- Interface issue #97 (decision-needed for Codex) covers the new shared files under `harness/manifests/`.

Measured on DEV (0.3 public, no GSM8K):

| System | in-domain | frozen proxy | generatable half |
|---|---|---|---|
| Clef-flash (full DEV) | 60.47 [58.58, 62.02] | 68.71 [65.69, 71.14] | 45.06 [41.66, 49.62] |
| Q9 stock (subset) | 43.98 | 22.18 | 23.28 |
| Q9 1-GDN (subset) | 44.35 | 23.17 | 26.63 |
| RWKV7-13.3B (subset) | 28.14 | 7.93 | 19.05 |
| Q9 1-GDN − stock (paired) | +0.37 [−1.91, +2.97] | +0.99 [−1.55, +3.61] | +3.35 [−0.46, +7.28] |

Caveats on the frozen half:
- **It covers only three areas** (Knowledge: 31, 32, 33, 44, 58; Tools: 9; Arts: 20, 22, 64). It has no Language or Retrieval benchmark. The proxy index renormalises over those three areas.
- It contains Clef-flash's most format-inflated public benchmarks (Home appliances, CLadder, SATA). That is why Clef-flash scores *higher* on the frozen half (68.7) than in-domain, while every stock backbone scores far lower.
- Compare candidates against their own parent on this proxy, not against Clef-flash's level.
- The second, independent new-domain proxy is a frozen set of held-out task *generators* (3–5 decision formats absent from both the Index and our data). It is kept as planned work and is not built yet.

### 5.2 Caveats

- Tier assignment is a judgement made once, by task family, before any score was seen. The rationale is in the JSON. GPQA and HLE are the weakest "same_family" calls (mmlu_aux is far easier), and moving them to "none" is defensible.
- DEV proxy CIs are wide on the 2,008-row subset (±6). Use full DEV for decisions.
- The proxy is public and the board's new-domain set is private and unknown. Agreement between them is untested until we have a board submission.
