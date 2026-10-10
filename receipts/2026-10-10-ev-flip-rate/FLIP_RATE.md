# FLIP_RATE: the MD1 robustness axis, measured on our candidates (B-PREREG-FLIP)

Status: SKELETON (2026-10-10). Jobs submitted, numbers pending. Rule: state `PREREG.md` B-PREREG-FLIP (state 225f534b).
Receipts: branch `claude/ev-flip-rate`, `receipts/2026-10-10-ev-flip-rate/`.

## What MD1 reports
- 8 perturbation types per request. Average flip rate 1.3 %. 0 flips on option reorder / reverse / shuffle and on
  option-description paraphrase.
- Named families: state paraphrase, instruction paraphrase, option-description changes, reorder / shuffle, key changes,
  formatting noise (LANDSCAPE.md, MD1 section). The exact operators, item set and per-type numbers are not published.

## Our operators vs MD1's 8 types (jevkit/perturb.py, frozen)

| MD1 type | jevkit operator | Coverage |
|---|---|---|
| State paraphrase | `state_paraphrase` (contraction / connective table over every state string leaf) | yes; rule-based and shallow, applies to 104 / 231 items only |
| Instruction paraphrase | `instruction_paraphrase` (same table + one of 3 neutral task frames) | yes, 231 / 231 |
| Option-description paraphrase | `option_paraphrase` (same table over option / level texts + final period toggled) | yes, 231 / 231 |
| Option reverse | `option_reverse` | yes, 213 (score items n/a) |
| Option shuffle | `option_shuffle` (seeded; identity draw rotated) | yes, 213 |
| Option reorder (a separate MD1 type) | none distinct: reverse and shuffle are the two reorderings | **missing as a third reorder** (a rotation or a move-one would be the natural third) |
| Key change | `key_relabel` (keys renamed to opt_1..opt_k, seeded order; choice only) | yes, 139 |
| Formatting noise | `whitespace_noise` + `markup_noise` (two operators) | yes, split in two |

So jevkit has 8 operators that cover 7 of MD1's 8 named types. One MD1 type (a third reorder) has no distinct operator.
Formatting noise counts twice. An "8-type average" here is therefore a different mix from MD1's.

## Method (pre-registered)
- Items: the 231 public JevBench v1.2 items (pin b6b8fff7). Perturbations from `python -m jevkit perturb make --public
  --seeds 0` at agent-main 7c131f19: 1,593 rows, sha256 655d028a...79d7. One seed. EVAL-ONLY, nothing fitted.
- Flip = the argmax decision changes after mapping back through `label_map`. Invalid answers on either side are
  excluded and counted. Per-operator rate = flips / valid pairs. Headline = unweighted mean over operators.
- CI: bootstrap over the 231 base items, with each draw carrying all of its variants. B = 1000, seed 16, percentile
  95 %. Paired deltas use the same resamples (`flip_ci.py`, whose points must equal `jevkit perturb score` exactly).
- Systems and jobs:
  - H2O-Lightning-4B v1.1 (its shim) and stock Qwen3.5-4B + H2O shim: job `<pending>` (branch claude/ev-flip-hs). All
    8 operators.
  - Stock RWKV7-G1k-13.3B (stock v3w, parent arm) and D19 (art-p3gpi step 2800, untempered child): job `<pending>`
    (branch claude/ev-flip-d19). Stock gets all 8 operators. Budget rule: D19 gets only `option_shuffle` +
    `key_relabel` (352 rows), because its readout takes ~5.6 s per item.

## Results

| System | state | instr | opt-para | reverse | shuffle | relabel | whitespace | markup | avg (8) [95 % CI] | reorder flips |
|---|---|---|---|---|---|---|---|---|---|---|
| MD1 (reported, own items) | ? | ? | 0 | 0 | 0 | ? | ? | ? | 1.3 % | 0 |
| H2O-Lightning-4B v1.1 | | | | | | | | | | |
| Qwen3.5-4B + H2O shim | | | | | | | | | | |
| RWKV7-G1k-13.3B stock v3w | | | | | | | | | | |
| D19 (shuffle + relabel only) | n/a | n/a | n/a | n/a | | | n/a | n/a | 2-op avg only | |

Paired deltas vs Qwen3.5-4B + shim (the baseline that the sym-invariant head must beat): pending.

## Comparability caveats (MD1)
1. **Operators differ.** MD1's 8 operators are unpublished. Ours are jevkit's own, rule-based ones. Our state
   paraphrase is a contraction / connective rewrite that changes only 104 / 231 states. MD1's is presumably a model
   paraphrase, which is a harder perturbation. Our formatting noise is two operators. We have no third reorder type.
2. **Items differ.** MD1's item set is private. Ours is the 231 public JevBench items, which include long and
   multi-hop items that MD1 (32k context) may never see.
3. **Mix differs.** The average is unweighted over operators. Because the operator sets differ, the 1.3 % and our
   average are not the same estimand. Per-type comparisons are only meaningful for the types MD1 names with a number:
   reorder / reverse / shuffle = 0 and option-description paraphrase = 0.
4. **MD1 was not run.** We compare against its reported numbers only (hard rule 3).
5. **A hard zero is achievable by construction.** An option-order-invariant head gives exactly 0 reverse and shuffle
   flips, and a relabel-invariant one gives 0 key_relabel flips. A measured 0 there is a property of the architecture,
   not luck, and should be reported as such.
6. **One perturbation seed.** The CI covers item sampling, not operator randomness. Shuffle and relabel use seed 0
   only.
