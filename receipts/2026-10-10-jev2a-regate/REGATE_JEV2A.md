# REGATE_JEV2A: independent re-gate of bschema-draw-jev2a (evidence lane, 2026-10-10)

**Verdict: LEAK-FREE and well-formed. Cleared for GPU use.** Every gate the builder claimed is CONFIRMED. I re-ran them from
origin/agent-main @7c131f19 and added my own checks. Three WARN items are listed below. None of them is an eval-item
leak: each is a public tool-catalog string or a named entity that the frozen rules exempt on purpose. This file contains
no eval text, only counts, hashes and TRAINING row ids.

Draw: claude/ev-jev2-draw @77065b1c, `ssdm-data/m2/build/bschema-draw-jev2a/`. Code used: `scripts/regate_jev2a.py`
(my own streaming matcher, plus format, cohort, fingerprint and provenance checks), `scripts/regate_fp_attrib.py`,
`scripts/regate_jev2a_tplcount.py`. The receipts (JSON) are in this directory and listed in SHA256SUMS.

## 1. Identity (hashes.json)
| claim | result |
|---|---|
| pointer `data/manifests/bschema-draw-jev2a.pointer.json` sha256 53416094… | CONFIRMED (git blob @77065b1c) |
| records uncompressed 393d0f0f… (gz 9da7e35f…), 30,000 lines | CONFIRMED |
| manifest uncompressed 4e89bd50…, content map b0070572… (30,000 each) | CONFIRMED |
| 30,000 rows, 6,000 each of multi_hop / judge_hard / temporal_numeric / fact / tool_selection; unique ids | CONFIRMED |

## 2. Gate modules: agent-main against the builder branch
I compared the git blobs at origin/agent-main and origin/claude/ev-jev2-draw for these files: train/jevbench_guard.py,
doc_overlap_guard.py, attack_guard.py, cohort.py and guard.py; data/dedup.py, text.py and config.py; harness/heldout.py;
jevkit/items.py; the JevBench public pin; the jevbench guard pin; safety_eval/attacks.py; and heldout_v1_templates.json.
**All are byte-identical, so there is no difference to report.** The builder branch changes only data/SOURCES.md and
otherwise adds new files. Its Checker (data/b_schema.py, b1_check.py) exists only on that branch, so I did NOT reuse it.
Cohort, held-out and fingerprint checks below call the agent-main modules directly.

## 3. Gates re-run (agent-main code, frozen pins)
| gate | builder | re-run | verdict |
|---|---|---|---|
| doc_overlap_guard scan (Index index af096960…) | 0 | n_fail 0; jevbench block 0 doc / 0 exact (index 0fe994c4…) | CONFIRMED |
| attack_guard scan (templates 5e2d6e01…, incl. D1-B + heldout_v1) | 0 | n_hit 0 | CONFIRMED |
| jevbench_guard rescan (doc + exact + all 5 fingerprint matchers) | 0 | 0 / 0 / 0 over 30,000 checked | CONFIRMED |
| Index Fingerprint (index-fp-v3, arrays b2961e35…, assert_production ok), item-tier probe as the builder's gate | 0 | 0 | CONFIRMED |
| cohort v2 (be6efe2b…) and v3 (498efa61…): id, base id, normalized content, source item (`aug.b1_view.base_id`), key-free text | 0 | 0 on all four keys, both cohorts | CONFIRMED (*) |
| frozen held-out (`harness.heldout.check_record`, frozen families 9/20/22/31/32/33/44/58/64) | 0 | 0 refused | CONFIRMED |
| content map equals `train.cohort.record_content_hash` per row | n/a | 0 mismatches | CONFIRMED |

(*) Adversarial note: `train.cohort.record_item_keys` strips only `~inj:`, not `~view:`. The source-item check is
therefore valid only on the base id, so I checked it on `aug.b1_view.base_id`. The check catches real cases: cohort v3
holds out 3,124 boolq, 1,994 gsm8k and 2,469 glaive_fc source items, and the draw touches none of them. The
builder's check stage had already dropped 2,335 cohort-item rows.

## 4. Independent overlap (own matcher, overlap.json)
The matcher is mine: NFKC plus casefold, `\w+` words, word 13-grams per leaf, and exact matches on normalized leaves of
at least 6 words or on the whole state.
| eval side | 13-gram: train rows hit | exact leaf / state | verdict |
|---|---|---|---|
| 231 jevkit public items (b6b8fff7) | **0** | 0 states; 97 glaive rows carry 1 of 3 tool-catalog descriptions | CONFIRMED, WARN-1 |
| GSM8K official test (local parquet ee7b8da9…, 1,319) | **0** | 0 | CONFIRMED |
| Index DEV (12,366 rows) | 18 glaive rows | 1 musique row (document title) | CONFIRMED, WARN-2 |
| Index TEST (12,423) | 20 glaive rows | 0 | CONFIRMED, WARN-2 |
| Index HELDOUT/other suite rows (128,608) | 9 glaive rows | 2 musique rows (document titles) | CONFIRMED, WARN-2 |
| proxy600 (644 ids) | **0** | 0 | CONFIRMED |

- **WARN-1.** 97 tool_selection/glaive rows contain one of three short tool-catalog descriptions (6–8 words). Each of
  the three appears verbatim as an option in 6 JevBench easy tool_selection items. In 58 of those rows it is the gold
  option. Examples: m2:glaive_fc:10034, :11771, :25341. jevbench_guard v2 defines a segment shared by 2 or more items as
  a template, so this is not an item copy (the user request 13-grams match nothing). It is a skill-level overlap with
  the easy tool_selection format. I see no action needed. If the lead wants zero exposure, dropping these rows costs
  97 / 6,000. The ids are in `jevbench-template-leaves.json`.
- **WARN-2.** Every Index 13-gram hit is a glaive tool description of 13–15 words, such as a BMI-category, tip-split
  or mortgage tool. It matches the candidate-tool documents in the instructions of 7 ToolRet-retrieval eval rows.
  `allowlist-v1.json` entry `toolret-public-tool-corpus` exempts that pool by design. No ToolRet query (S:query) and no
  other Index family matched. The "≥ 90 % coverage" figures only reflect that this description is the row's one leaf
  long enough to form a 13-gram. The 3 exact-leaf hits are MuSiQue document titles (a place and two film titles) that
  equal option strings in HLE-text-MC (DEV) and BBH movie_recommendation (HELDOUT). These are named entities, not items.
  The training ids are in `overlap-hit-train-ids/`.
- **Full-row Index Fingerprint (stricter than the gate, index-fingerprint-attrib.json).** Probing state plus questions
  hits 1,221 rows. 1,005 of them are question-only hits on short rubric, option or number strings (1–12 words, mostly
  minhash and edit). 216 hit a state segment: 210 of these are 1–7-word MuSiQue titles, and 6 are gsm8k judge rows
  matched by contain or minhash. Those 6 gsm8k rows have 0 13-gram overlap with GSM8K test and the Index. The builder's
  item-tier gate is the defined one, and it gives 0. Logged as WARN-3, information only.

## 5. Split provenance (provenance.json; source files sha-match the builder's sources.sha256)
| source | rows | check | verdict |
|---|---|---|---|
| MuSiQue-Ans v1.0 **train** (83a75b1e…) | 1,856 | 1,856/1,856 ids present in the train file; every row's documents ⊆ that train record's paragraphs | CONFIRMED |
| GSM8K **train** (ea82612e…) | 2,635 | 2,635/2,635 questions equal the train row at source_id; 942/942 choice golds equal the source `####` answer; 0 overlap with the official test set | CONFIRMED |
| BoolQ **train** (4f028e99…) | 1,453 | 1,453/1,453 passage + question equal the train row | CONFIRMED |
| Glaive FC v2 (train-only dataset, 7e7e32f0…) | 2,997 | 2,997/2,997 user texts found in the source row | CONFIRMED |
| jev2synth (own generators) | 21,059 | source_split "synthetic" | n/a |
| licence field on every row | 30,000 | 0 missing | CONFIRMED |

SKIPPED: the BoolQ validation and MuSiQue dev cross-check by hash. Neither file is on disk, and I did not download them.
The ids and texts above trace every row to the train file, and no MuSiQue id outside the train file occurs.

## 6. Well-formedness (format.json, tokens-shim.json)
- **Gold:** 0 failures. Choice gold is always in the options with unique text, noul labels are always true/false with
  {true,false} keys, and score labels are in range.
- **Option order kept** (keys not re-sorted): 54/6,000, 185/4,403 and 135/5,053 rows are already in sorted order,
  against 50.0 / 175.4 / 137.9 expected under a uniform shuffle. CONFIRMED.
- **Gold-position χ²** on the presented order (scipy): tool_selection p 0.774, temporal_numeric p 0.675, multi_hop
  p 0.470. This matches the builder's 0.77 / 0.67 / 0.47. CONFIRMED.
- **Empty criteria: 1,100 rows.** These are the score questions with no option descriptions: multi_hop ref_chase
  `count` (402) and temporal `days_late` (698). There are 0 empty noul criteria and 0 empty option descriptions. The
  audit says empty criteria hurt, so the lead should decide whether to keep these 1,100 rows (3.7 %).
- **Type mix:** fact noul 6,000; judge_hard noul 6,000; tool_selection choice 6,000; temporal_numeric choice 4,403 /
  noul 899 / score 698; multi_hop choice 5,053 / noul 545 / score 402.
- **State type:** dict 17,058 (judge 6,000, multi_hop 6,000, temporal 5,058) and plain prose 12,942 (fact 6,000, tool
  6,000, temporal 942).
- **Labels:** gsm judge corruption matches its label in every row. final: false 440; step: gsm_final true 168 (the final
  answer is still right), gsm_steps false 234; clean: true 851. boolq true:false is 882:571.
- **Duplicates:** 0 exact duplicate rows by normalized full text. Repeated **states** within a family: tool_selection
  626 groups / 2,418 extra rows, of which 2,232 rows are glaive. Glaive v2 repeats user requests under new ids with
  different option sets. temporal_numeric has 208 groups / 249 extra rows. With digits masked (generator templates),
  the near-duplicate extra rows are temporal 2,719, judge 1,105 and fact 302. This is a diversity issue, not a leak.
- **Tokens under jrt-h2o-shim:qwen3.5-4b** (identity f0be274c…, equals the pin). The tool is the trainer's row builder
  via the builder's tokcheck, re-run on the final records: max 1,789, 0 rows over 2,048, 0 refused rows. Per family
  p50/max: multi_hop 980/1,789; temporal 212/333; judge 153/547; tool 147/214; fact 122/496. CONFIRMED.

## 7. Notes for the lead
1. The draw is clean. WARN-1 to WARN-3 are by-design template or entity overlaps, and I recommend no change.
2. Optional quality changes: drop or regenerate the 1,100 empty-criteria score rows, and deduplicate glaive by request
   text (about 2.2k rows).
3. Family task shapes were calibrated on the public items' FORMAT only (option count, type, state shape; data/jev2.py
   docstring), never their text. I note this so the lead is aware; no text overlap exists (section 4).

## 8. Conservative exclusion list (coordinator, 2026-10-10)
`jev2a_exclude_ids.txt` holds 105 ids, sha256 aeb8027aaaebdb6fad2b24f8b5bee9d37c4e176fcc4e18ad0d366acf1b579705 (sorted,
LF, trailing newline). It lists EVERY row that has a normalised leaf of at least 6 words verbatim equal to a leaf of a
JevBench public item or an Index suite row (`scripts/regate_exact_leaves.py`, report `exact-leaves-all.json`):
97 tool_selection/glaive rows (WARN-1; the 3 descriptions are each in 6 JevBench easy tool_selection items) and
8 multi_hop/musique rows. 6 of the musique rows share one place title that equals an HLE-text-MC (DEV) option, and 2
have film titles that equal BBH movie_recommendation (HELDOUT) options. Section 4 counted 3 musique rows because that
pass kept only the first row per leaf; this pass counts every row. The draw keeps 29,895 rows after exclusion.
