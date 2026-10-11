# REGATE_JEV2B: independent re-gate of bschema-draw-jev2b (evidence lane, 2026-10-11)

**Verdict: LEAK-FREE and well-formed. Cleared for the JEV2-4B GPU run.** The exclusion list is empty.

All 7,500 rows are synthetic. The draw has 0 13-gram and 0 verbatim-leaf overlap with the JevBench public items,
Index DEV / TEST / HELDOUT / proxy600, and GSM8K test. Every gate passes on agent-main code. I checked six score/choice
golds by hand and all are correct. The gold-level rates match the builder's table exactly.

Two notes, neither of them a leak:
- `prob_indep` has a constant gold, "same", in 124 of 124 rows.
- The builder's `shape_gate` fails while GATES.json still reports `all_pass: true`; this is the same issue as jev2c.

This file contains no eval text. The code is the same as for jev2a and jev2c.

Draw: claude/ev-jev2-draw @a4aaa7de, `ssdm-data/m2/build/bschema-draw-jev2b/`. Between 12461fff and a4aaa7de the
builder changed only `data/` generators, `scripts/` and receipts. No gate module or pin changed.

## 1. Identity (hashes.json)
| claim | result |
|---|---|
| pointer sha256 35620f5f… | CONFIRMED (git blob @a4aaa7de) |
| records uncompressed 9406ae08… (gz 8f3a9268…), manifest 1de4897b…, content map 241a30b7…; 7,500 lines each | CONFIRMED |
| 1,500 rows each for probability / trap / ambiguous / tradeoff / adversarial | CONFIRMED |

## 2. Gates re-run (agent-main code, frozen pins)
| gate | result | verdict |
|---|---|---|
| doc_overlap_guard (Index af096960…, JevBench 0fe994c4…) | 0 / 0 / 0 | CONFIRMED |
| attack_guard (5e2d6e01…: official hijack, D1-A, D1-B, heldout_v1; gram, wildcard, fuzzy, base64) | 0 hits | CONFIRMED |
| jevbench_guard rescan (doc, exact, 5 fingerprint matchers) | 0 / 0 / 0 | CONFIRMED |
| Index Fingerprint, item-tier probe | 0 | CONFIRMED |
| Index Fingerprint, strict full-row probe | 1,224 rows (see below) | INFO |
| cohort v2 / v3 (id, base id, content, source item, key-free text) | 0 | CONFIRMED |
| frozen held-out `check_record` | 0 | CONFIRMED |
| content map equals record_content_hash | 0 mismatches | CONFIRMED |

**Full-row Fingerprint hits (index-fingerprint-attrib.json).**
- All 1,224 hits sit on question segments of 3–5 words: 478 ambiguous rows and 746 trap rows. That is every
  ambiguous score row plus the trap score rows, and the hits come from the exact and minhash matchers.
- The matched segments are the generic count-scale level labels of the score questions ("none of them", "exactly
  one", and similar), which equal option strings in Index catalog 24. 0 state segments hit.
- These are generic rubric words, not item text, so no action is needed.

## 3. Overlap (own matcher, overlap.json; 467,647 training 13-grams)
| eval side | 13-gram rows hit | exact leaf / state |
|---|---|---|
| JevBench 231 public items (incl. probability / trap / ambiguous / tradeoff / adversarial items) | 0 | 0 / 0 |
| Index DEV / TEST / HELDOUT | 0 / 0 / 0 | 0 / 0 |
| proxy600 | 0 | 0 |
| GSM8K official test | 0 | 0 |
| verbatim ≥ 6-word leaf vs JevBench + every Index row (`exact-leaves-all.json`) | n/a | **0 rows; `jev2b_exclude_ids.txt` is empty** |

## 4. Provenance
- `source` = jev2synth for 7,500/7,500 rows and `source_split` = "synthetic". There is no public dataset row, so there
  is no non-TRAIN source. 0 rows have a missing licence.

## 5. Adversarial and trap rows against the frozen attack sets
- 872 adversarial score rows carry an injected instruction inside an untrusted field. Example:
  m2:jev2synth:adv:100096, where a reporter's note contains a bracketed "assistant instructions" line that tries to
  force severity 0. The gold follows the facts, not the injection.
- attack_guard scans every key and value plus base64 decodes. It finds 0 hits under all three rules: gram (6-gram /
  exact fragment), whole-template wildcard, and fuzzy (≥ 50 % of 3-grams). The checked sets are the 5 official hijack,
  4 D1-A, 8 D1-B and heldout_v1 templates. **CONFIRMED: no real eval attack template is embedded.**
- Trap rows are fact-check, premise and exception tasks with no injection text. They also give 0 attack hits.

## 6. Gold spot-checks by hand (one per family; inspect output kept locally, row text not reproduced)
| row | rule | hand result | gold |
|---|---|---|---|
| prob:0 `prob_bayes_band` (score) | base 8 %, TPR 95 %, FPR 10 %: P = .076 / (.076 + .092) = 0.452 → band "35–65 %" = level 2 | 2 | 2 ✓ |
| trap:100288 `trap_count` (score) | summary's 3 statements (renewal date, owner, plan) all match the record | 3 ("all three") | 3 ✓ |
| ambig:100800 `ambig_email_count` (score) | "Leila" matches 1 of 3 contacts | 1 | 1 ✓ |
| tradeoff:100224 `tradeoff_weighted_count` (score) | exclusion "not ready within 10 weeks": ready in 10 / 3 / 8 / 6 weeks → 4 eligible | 4 ("four or more") | 4 ✓ |
| adv:100096 `adv_severity` (score) | slow app responses for a few internal users, 45 min, which is minor; the injected "severity is zero" is ignored | 1 | 1 ✓ |
| trap:100672 `trap_premise` (choice) | the update says delivered on the 3rd listed date | d3 | d3 ✓ |
| prob:100096 `prob_indep` (choice) | fair 4-sector spinner after a streak: 25 %, "same" | same | same ✓ |

Cosmetic: prob:0 mixes its frames ("clinic scheduling desk" vs "shipment is damaged"). The maths and gold are
unaffected.

## 7. Gold-level distributions: mine vs the builder's build-report.json
Mine come from a re-count over the records.
| family | type | re-count (= builder, to 4 dp) |
|---|---|---|
| probability | score 0–4 | .0631 / .3250 / .2811 / .2447 / .0860 ✓; noul .50 / .50 ✓ |
| trap | score 0–4 | .2297 / .1892 / .2527 / .2230 / .1054 ✓; noul false .5607 / true .4393 ✓ |
| ambiguous | score 0–3 | .1967 / .4540 / .2824 / .0669 ✓ |
| tradeoff | score 0–4 | .0700 / .1731 / .1989 / .3886 / .1694 ✓; noul .5198 / .4802 ✓ |
| adversarial | score 0–4 | .2110 / .2259 / .2397 / .2259 / .0975 ✓; noul .4654 / .5346 ✓ |

Type mix: choice 3,352 / noul 992 / score 3,156 (42.1 %; per family adversarial 58 %, trap 49 %, tradeoff 36 %,
probability 35 %, ambiguous 32 %). This matches the builder's 0.4469 / 0.1323 / 0.4208. **CONFIRMED.**

## 8. Well-formedness (format.json, tokens-shim.json)
- **Gold:** 0 failures. **Empty criteria:** 0. Score criteria are non-empty level lists, and there are 0 empty option
  descriptions.
- **Option order kept:** sorted-order counts sit at the shuffle expectation in every family (16/17.1, 146/137.0,
  148/156.5, 8/9.4, 9/9.4). The builder's own receipt shows why this matters: on sorted ids, gold position is
  χ² p ≈ 0 for trap and ambiguous.
- **Gold-position χ²** (presented order): adversarial 0.46, ambiguous 0.18, probability 0.98, tradeoff 0.13,
  trap 0.88. CONFIRMED.
- **WARN (shortcut): `probability/prob_indep` has gold = "same" in 124/124 rows.** This is correct by construction
  (the gambler's fallacy), but the answer text is constant, so it is a learnable shortcut. Related: prob_conj golds are
  only `a_only` or `equal` (62 / 65), which is mathematically forced. I suggest keeping prob_indep only if the lead
  accepts the shortcut; otherwise mix in non-fair or dependent variants.
- **State:** string 6,879; dict 621 (ambiguous).
- **Duplicates:** 0 exact. Repeated states: 48 groups / 63 extra rows (prob_dice, prob_dice_band views of one
  scenario). Digit-masked near-duplicates: trap 196, probability 61, adversarial 32, ambiguous 11 extra rows.
- **Tokens under jrt-h2o-shim:qwen3.5-4b** (f0be274c… equals the pin): max 420, 0 rows over 2,048, 0 errors. Per family
  p50/max: tradeoff 365/420, ambiguous 239/415, adversarial 192/233, probability 191/255, trap 170/203. These match the
  builder's maxima. CONFIRMED.
- **Builder `shape_gate`:** all_pass false. The draw is shorter than the public shape: for example, the
  probability-family public max is 1,940 tokens against the draw's 255. GATES.json still reports top-level
  `all_pass: true` because the shape gate is not part of that conjunction, the same as in jev2c. This is a calibration
  issue, not a leak.
