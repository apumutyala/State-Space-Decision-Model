# REGATE_JEV3: independent re-gate of bschema-draw-jev3 (train) + bschema-draw-jev3-slice (evidence lane, 2026-10-11)

**Verdict: LEAK-FREE. The exclusion list is empty. Cleared for the JEV3-SYM run.** Two caveats:
- **(P)** Train-split provenance is verified file-to-file only for Aegis 2.0 and HelpSteer2. For PRM800K, train
  provenance is verified through the MATH `unique_id` split prefix. The split files for HelpSteer3, AmbigNQ, TAT-QA,
  TimeQA, PrivacyQA, PRM800K phase2 and MATH were deleted after the build. Counts that could not be verified are given
  below.
- **(L)** Two label shortcuts (section 8).

**Exclusion list:** `jev3_exclude_ids.txt` is empty: a single LF, sha256
01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b.

This file contains no eval text. Ref: claude/ev-jev3-draw @5c963b25 (builder f34e8413).

Gate code: agent-main @2c6f5d61 and 5c963b25 have byte-identical gate modules and pins. The one difference is
`jevkit/items.py::write_items` (agent-main dropped `sort_keys`), which no gate uses. The re-gate ran from this
branch's 7c131f19 copies, which are byte-identical to 5c963b25.

## 1. Identity (`jev3-checks.json`)
| artefact | claimed | recomputed |
|---|---|---|
| train pointer file | ce70500d… | ce70500d… ✓ |
| train records / manifest / content / lens (uncompressed) | 368fd715 / 4350f8a8 / 9f77efe7 / e13d707c | identical ✓ |
| train ids sha (sorted, "\n"-joined, no trailing newline) | feb681f3… | feb681f3… ✓; 5,960 unique ids |
| slice pointer file | 4d1cbba6… | 4d1cbba6… ✓ |
| slice records / manifest / content / lens | 1f184bb9 / 8d874e0a / 85e7e1f3 / 0070a940 | identical ✓ |
| slice ids sha | 844233f2… | 844233f2… ✓; 1,200 unique ids |
| families | train 1,000 × 5 + long_policy 960; slice 200 × 6 | ✓ |

## 2. Gates, agent-main code, run on BOTH draws (`train/`, `slice/`)
| gate | train | slice |
|---|---|---|
| doc_overlap_guard (Index af096960…, JevBench 0fe994c4…) | 0 / 0 / 0 | 0 / 0 / 0 |
| attack_guard (5e2d6e01…) | 0 | 0 |
| jevbench_guard rescan (doc, exact, 5 fingerprint matchers) | 0 / 0 / 0 | 0 / 0 / 0 |
| Index Fingerprint index-fp-v3, item tier **and strict full-row** | 0 and **0** | 0 and **0** |
| cohort v2 / v3 (id, base id, content, source item, key-free text) | 0 | 0 |
| frozen held-out `check_record` | 0 | 0 |
| content map equals record_content_hash | 0 mismatches | 0 mismatches |

## 3. Overlap (own matcher): 13-gram, verbatim leaf, and an extra line-level rule
| eval side | 13-gram rows (train / slice) | exact leaf ≥ 6 words / whole state |
|---|---|---|
| JevBench 231 public items | 0 / 0 | 0 / 0 |
| Index DEV / TEST / HELDOUT | 0 / 0 (all three) | 0 / 0 |
| proxy600 | 0 / 0 | 0 / 0 |
| GSM8K official test | 0 / 0 | 0 / 0 |

**Extra line-level rule (`line-exact.json`, `scripts/regate_line_exact.py`).** This catches short copied questions,
such as AmbigNQ / NQ-style questions under 6 words, which the leaf rule misses. The rule: any normalised sentence or
line of ≥ 4 words that exactly equals one on the eval side.
- 140 rows hit. **Every hit is a generic phrase of 4–7 words**, for example a conversational stock phrase, a MATH
  answer-format sentence, or a variable-name question.
- Eval-side frequency runs up to 5,500 rows for the most common phrase. None of these is an item question, passage or
  answer.
- I inspected every matched line locally (`linewhy`, not reproduced here). The hits are not leaks and are not
  excluded.

## 4. Provenance (`jev3-checks.json` → `prov`), per row
| source | train / slice rows | check | result |
|---|---|---|---|
| Aegis 2.0 (`dl-2026-10-10/aegis2/train.json`) | 1,000 / 200 | `row_sha256` = sha256(canon(source row)); gold recomputed from the source's human label | **1,200/1,200 sha match; 1,200/1,200 gold match**. Label sources: prompt labels human 891 + 173, response labels human 109 + 27 ✓ |
| HelpSteer2 (`dl-2026-10-10/helpsteer2/train.jsonl.gz`) | 430 / 80 | row sha; gold = (attribute rating ≥ 3, rating 2 skipped) | **510/510 sha; 510/510 gold** ✓ |
| HelpSteer3 feedback/train | 570 / 120 | official file streamed, never stored; only the builder's `hs3_cache` is on disk | row sha and gold match the cache (690/690). **690 rows not verifiable against the official file** |
| PRM800K phase2_train | 1,000 / 200 | MATH `unique_id` split prefix; problem text = builder cache | **1,200/1,200 `train/…` unique ids (0 `test/…`)**; problem text equals the cache for 1,200/1,200. The MATH test / MATH-500 text cross-check could not run, because the hendrycks_math parquets were deleted (see note) |
| AmbigNQ light train | 1,000 / 200 | zip deleted | **1,200 not verifiable** (`prov.split` = train on all) |
| TAT-QA train | 730 / 143 | file deleted | **873 not verifiable** |
| TimeQA human_annotated_train | 270 / 57 | file deleted | **327 not verifiable** |
| PrivacyQA policy_train_data | 960 / 200 | file deleted | **1,160 not verifiable** |

Not verifiable against an official train file, per source and in total:

| source | rows not verifiable |
|---|---|
| HelpSteer3 | 690 |
| AmbigNQ | 1,200 |
| TAT-QA | 873 |
| TimeQA | 327 |
| PrivacyQA | 1,160 |
| **total** | **4,250 of 7,160** |

**PRM800K note.** PRM800K's own train split folds in 4,500 MATH *test* problems, but they keep their original
`test/<subject>/<n>.json` unique_id. A `train/` prefix on every row therefore means every problem comes from the MATH
TRAIN split, and none from MATH test or MATH-500 (a subset of MATH test). That is an id-level proof. The text-level
re-check against the official parquets (hendrycks_math @21a56338; the 7 test files total about 1.9 MB) needs a
download, which I did not do.

**Licences:** present and consistent with prov on every row of both draws (0 missing).

## 5. Slice against train: disjoint (`disjoint`)
- 0 shared values for each of: id, source item (base id), group, content hash, normalised state, and long_policy
  policy document (PrivacyQA DocID).
- No normalised state is shared by any two rows across train + slice.

## 6. Cross-draw: jev3 against jev1 and jev2 (`cross_draw`)
- **0 hits** on each of id, source item, content hash and normalised state. The draws checked are bschema-draw-jev1,
  jev2a-v2, jev2b and jev2c-v2 (the mix), and the superseded jev2a and jev2c.
- The source-item check is meaningful: jev1 holds 7,118 aegis2 and 4,827 helpsteer2 base ids, and jev3 shares none of
  them.

## 7. Gold hand-checks (`scripts/regate_jev3_spotcheck.py`; 10 rows, one or more per family)
| row | check | gold |
|---|---|---|
| prm800k train/algebra/1131 `prm_step` (judge_hard) | Asks about step 4 ("midpoint x = (x_A + x_B)/2"), which is correct. The human PRM rating of step 4 is +1; step 5 is rated −1 and is truly wrong, since the sum is preserved, not halved | true ✓ |
| prm800k train/algebra/1012 `prm_all` | Midpoint (3, −3), sum 0. Human ratings all +1, finish "solution" | true ✓ |
| helpsteer2:10122 `hs2_bar_helpfulness` (adequacy) | Human helpfulness rating 3 against a bar of ≥ 3 | true ✓ |
| aegis2:0067c2d5… `aegis_message` (policy) | Human prompt_label safe, so not flagged | false ✓ |
| aegis2:022e7cad… `aegis_reply` (policy) | Human response_label safe | false ✓ |
| ambignq:-1007388… `ambignq_single` (ambiguous) | Unambiguous question with a single answer (Greg Sestero) | answer ✓ |
| ambignq:-1091680… `ambignq_multi` (ambiguous) | "Head of state in a republican state" has multiple readings | ask ✓ |
| tatqa:000abf67… `tatqa_choice` (temporal_numeric) | The table gives the 2019 grant date as 30 Nov 2018 | 30 Nov 2018 ✓ |
| timeqa:004f64e761-1 `timeqa_yn` (temporal_numeric) | In 1990 he was at Emory (B.A. 1994); George Mason J.D. was 2001 | false ✓ |
| privacyqa:006abb9710 `pqa_answerable` / `pqa_count` (long_policy) | Question about app permissions; the excerpt and the listed sentences cover cookies and analytics, not app permissions | false / 0 ✓ |

## 8. Well-formedness (`format.json`, `tokens-shim.json`)
- **Gold:** 0 failures. **Empty criteria:** 0. **Empty option descriptions:** 0.
- **Option order kept:** sorted-order counts sit at the shuffle expectation.
- **Gold-position χ²** (train / slice): ambiguous 0.14 / 0.89, long_policy 0.41 / 0.34, temporal 0.70 / 0.81.
- **Duplicates:** 0 exact, 0 repeated states, 0 digit-masked near-duplicates.
- **Tokens under jrt-h2o-shim:qwen3.5-4b** (equals the pin; 0 errors):
  - long_policy: train 2,141–4,091, slice 2,143–3,954. Every row is inside its [2,000, 4,096] window.
  - Other families: max 2,532 (temporal), with 15 train rows and 2 slice rows over 2,048.
  - The cell needs max_len 4,096.
- **(L) Label shortcuts. Not leaks; for the lead's attention.**
  - `policy/aegis_reply`: the gold is **false in all rows (109 train, 27 slice)**. Every sampled reply is human-labelled
    safe, so "is the reply unsafe?" always has the answer no.
  - `adequacy/hs2_bar_coherence`: 26 true / 1 false.
  - `ambiguous`: the label follows the AmbigNQ annotation by construction (single → answer, multi → ask, 500 / 500).
    This is the task itself, and it is balanced.
