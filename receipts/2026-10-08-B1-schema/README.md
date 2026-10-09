# B-1 schema-permutation synthetic draw (bschema-draw300k-v1)

## What the Clef post says about its data
The Cloudflare Clef post (https://blog.cloudflare.com/clef-decision-models/, read in full 2026-10-08) says only that
Clef trained on internal synthetic datasets that vary field orders, prompts and schema structures. It gives no record
structure, scale, option counts, label source or mixing. The training method it describes (LS-CE + Brier, then RLCD
with partial credit for adjacent ordinal choices and a reward for fully correct records) implies multi-field records
with ordinal fields, but the post does not describe them as data. So the recipe here is our reading: label-preserving
permutations of field order, prompts and schema structure over allowed train material.

## Scope
Single-record permutations (task items 2b-2e). Multi-question, cross-field and case-exact generation (2a) is Agent C's
work (`stack/rwkv7/c-multirecord-synth`, coordinator re-scope 2026-10-08). C's draw plugs in as pool P3:
`candidates --extra-pool <dir> --extra-pointer <pointer> --extra-quota N`. P3 records are verified against their
pointer, passed through unchanged under the same checks, and capped at 8 rows per source item. The 300k draw was built
without P3, because C's pool had not landed.

## Build
- Builder: `data/b_schema.py`. Candidates were built at c881984 (SLACK 1.35, seed 20261009). The select stage and the
  pointer are at cb1bdba; only a guard against a partial token check changed between the two commits.
- Token check: `scripts/b_schema_tokcheck.py`, run in 6 shards with the trainer's own code from
  claude/b-train-v3w-template @9c3e435 (`train.data.items_from_records`, layout g1v2) and the pinned RWKV7-G1k tokenizer.
  Every candidate was rendered under `jrt-rwkv-g1-v2f` AND `jrt-rwkv-g1-v3w`. A row is kept only if both templates
  render it at 2,048 tokens or fewer and every continuation is 16 tokens or fewer. Results: 365,155 candidates,
  0 render errors, 14,931 dropped for length.
- Commands:
  ```
  python -X utf8 -I -c "...data.b_schema.main([...])" candidates --n 300000
  python -X utf8 -I scripts/b_schema_tokcheck.py --trainer-root <9c3e435> --tokdir <G1k> --records <work>/candidates.jsonl.gz --out <work>/lens-$i.jsonl --shard $i --nshards 6
  python -X utf8 -I -c "...data.b_schema.main([...])" select --n 300000
  ```
- Pools:
  - P1: b1-draw100k-v3 (100,000 rows; its pointer is verified).
  - P2: rows of the 150k set and the B1 supplement whose source item is new. 48,018 of these passed the B1 Filter checks.
  - Not used from P2: 7,596 `mixed` multi-item rows.
  - No downloads.
- Files (outside git): `ssdm-data/m2/build/bschema-draw300k-v1/`, pointer `data/manifests/bschema-draw300k-v1.pointer.json`.
  - Uncompressed sha256: records 9f4e27d3..., manifest 4a0604c6..., content map 0ef13137... (full values in the pointer).
  - Raw gz sha256: records ad821710..., manifest 70701f81..., content 6a6d50a3...
  - Candidates gz sha256: 6bd2ef60...
- 1M option: `--n 1000000` is scripted but was not built. At this run's rates (10 min to generate and 25 min to
  token-check 365k rows), it would take about 1.5 h. It would also push reuse to 7-8 variants per item, because only
  about 148k source items are eligible.

## Transformations: rows touched (300k draw)
| Operation | Rows |
|---|---|
| Instruction frame | 288,216 |
| Instruction template swap | 80,352 |
| Option-id restyle | 244,100 |
| Question-id rename / reorder | 178,385 |
| State nest | 67,927 |
| State key rename | 67,817 |
| State as text | 19,541 |
| State as ordered list | 19,275 |
| State string wrap / header | 11,679 |
| K shrink | 54,256 |
| K expand (intent) | 11,901 |
| noul to choice | 5,161 |
| choice to noul | 9,228 |
| Score scale reversal | 3,110 |

Option-id styles are close to uniform: about 30k rows each for letters, lowercase letters, digits (0-based), digits
(1-based), option_k, opaque, neutral words and kept.

Option counts per question:

| K | Questions |
|---|---|
| 2 | 328,795 |
| 3-8 | 95,995 |
| 9-16 | 35,633 |
| 17-40 | 24,411 |
| >40 | 2,442 |

Question types: 454,263 choice, 22,113 noul, 10,900 score. Records with one question: 273,177.

## Per-source counts
Targets are the v3 draw's mix times 3, with short pools spilling into the others.

| Source | Rows |
|---|---|
| glaive_fc | 83,198 |
| clinc150 | 30,315 |
| banking77 | 18,167 |
| mmlu_aux | 17,902 |
| winogrande | 14,986 |
| gsm8k | 12,818 |
| toolace | 12,515 |
| boolq | 12,245 |
| anli | 12,157 |
| hellaswag | 12,103 |
| when2call | 11,716 |
| CRUXEval-synth | 7,355 |
| BFCL-synth | 7,007 |
| sst5 | 6,128 |
| factchange | 6,048 |
| humicroedit s1 / s2 | 4,594 / 4,538 |
| hover | 4,292 |
| ragtruth | 4,292 |
| mnli | 3,985 |
| vast | 3,985 |
| acos | 3,066 |
| nli4ct | 2,620 |
| API-Bank | 1,849 |
| habermas | 1,838 |
| nq | 281 |

## Token lengths
| Template | p50 | p95 | p99 | max |
|---|---|---|---|---|
| v2f | 454 | 1,528 | 1,709 | 2,048 |
| v3w | 468 | 1,600 | 1,954 | 2,048 |

## Checks (all pass)
- `exclusion-check.json`: `data.b1_check --max-item-reuse 8` PASSES over all 300,000 rows with 0 failures. It checks:
  - cohort v2 and v3 by id, item, content and key-free text;
  - Index DEV/TEST rows by content, key-free text and item text;
  - SplitGuard;
  - `check_record` per row and `check_generator` per (source, family);
  - `assert_draw_disjoint` v2/v3;
  - D1-B (0 hits);
  - no near-duplicates across items.

  Context-tier overlaps (tool schemas, background passages) are reported but not failed, as in B1.
- `generator-specs.json`: `check_generator` passes for the b_schema_synth spec and all 42 (source, family) specs. The
  frozen catalogs [9, 20, 22, 31, 32, 33, 44, 58, 64] are never targeted.
- `trainer-draw.json`: the trainer at 9c3e435 loads the draw with `train.train_set.draw` (cell d3_v3w_fast, pointer
  swapped, `heldout_excluded` empty, n 24,000) and `heldout_gate` over all 300k rows. Both PASS, with no sidecar.
- `binary-label-rates.json`: detection tasks keep their natural base rates.

  | Task | Base draw | This draw |
  |---|---|---|
  | ACOS yes | 1.37% | 1.36% |
  | CLINC member yes | 5.33% | 5.35% |
  | BANKING member yes | 6.3% | 6.4% |
  | BFCL yes | 29.6% | 29.6% |
  | RAGTruth true | 42.1% | 43.3% |

  For K 2-4 (n of 31k or more each), the gold's canonical position is within 2% of uniform. Large K is noisier
  (relative deviation up to 0.27 at K 15, n 1,974), partly from the "kept" id style.

## Diversity and leakage audit (`audit.json`)
- Source items: 122,831. Max variants per item: 8 (cap 8; reached only by fact-change pairs, which share one item).
  Histogram: 1: 27,190, 2: 31,099, 3: 49,581, 4: 14,085, 6-8: 876.
- 90.9% of rows share their source item with at least one other row.
- Exact key-free-text collisions: 0 across items and 74,542 within items. Within-item collisions are expected, because
  re-styling ids and reordering fields keep the same text.
- MinHash near-duplicates (Jaccard 0.8 or more, 10% sample): 1.8% of rows have a near-duplicate from another item. For
  the base 100k v3 draw (30% sample) the figure is 4.7%. Both are dominated by shared intent and tool catalogs.

# bschema-draw206k-v2 (Agent A's mix change + the v1 review + item-level mix + attack-template exclusion)

**Takeaway.** 206,544 records = **280,585 trained items** (one per question, as `train.run_cell.cell_items`). The draw is
sized by unique source items, not by a 300k row target. The review said "add NEW items instead of more variants", so
every key has at most 2 variants per item (3 for keys with 10k or more items). Every remaining guard-family item is
used.

- Every check passes. Agent A's `train/attack_guard.py` had not landed when this was built.
- Binary items: 42.8% yes (no-share 57.2%, cap 60%). v1 was 18.2% yes.
- Membership / verify views: 4.1% of items. v1 had 29%.
- glaive: 10.3% of items.

## Build
- Pools:
  - P1: b1-draw100k-v3.
  - P2: the 150k set + supplement.
  - P4: `b1-pool-v2x` (pointer `data/manifests/b1-pool-v2x.pointer.json`, 84,183 rows). It is `data.b1 draw`, no
    backfill, with:
    - every remaining ACOS (4,391), NLI4CT (855), VAST (1,835 posts) and Habermas (1,019) item;
    - HoVer 12k, RAGTruth 8.5k, MNLI 15k;
    - CLINC / BANKING member views 12.5k, ANLI-claim views 6k;
    - from `b1-families-x` (`scripts/b1_families_more.py`, unchanged b1-families @a4bc718 builder): BFCL-synthetic 9,490,
      CRUXEval-synthetic 11,999 (executed answers) and API-Bank train 604. API-Bank has only 604 usable rows: the other
      1,680 train dialogues render over 2,048 tokens.
  - P3: Agent C's `c-multirecord-synth-100k-v1` (pointer in C's worktree, records sha 45b4fa45...). The first 20,000
    rows were appended with the same checks (`extra` stage). 17,031 of those fit in 2,048 tokens and 7,000 were drawn,
    giving 17.5% of items.
- Builder: `data/b_schema.py --mix v2` (`MIX_V2` weights read as ITEM shares, `plan_items_v2`), at 34693a9.
  - Candidates: 339,306 (+20,000 C).
  - Dropped: 10,026 over 2,048 tokens; 46,521 by the per-item variant cap.
  - Every candidate was token-checked under v2f and v3w with the trainer at 9c3e435.
- Review fixes, each with tests in `tests/test_b_schema.py`:
  - choice to noul only for yes/no, true/false and correct/incorrect pairs. v1's 1/0 and Y/N pairs made fake true/false
    questions.
  - Option texts that name ids ("A and C") or aggregate options block the id restyle.
  - Digit ids are zero-padded when K > 10.
  - Options that duplicate the gold's text are deduplicated (1,579 rows).
  - State keys named in instructions keep the state unchanged.
- Item-level constraints, applied to what availability actually allows:
  - membership / verify views at most 5% of items combined;
  - binary no-share at most 60%, by scaling down no-heavy binary keys, never below floors for ACOS / BFCL / MNLI /
    NLI4CT;
  - glaive at most its weight share.
- Attack templates (`data/attack_filter.py`, operator rule): injected rows (`aug.injection` / `~inj:`) are dropped. So
  is any 6-gram, or exact 4-5-word fragment, of D1-A, D1-B, the 5 official hijack templates, the 6 training-only hijack
  templates, and heldout_v1 (21 templates; @f273618 copy sha 25df1dcf...).
  - Draw: 0 hits.
  - v1: 0 hits.
  - b1-draw100k-v3: 6,776 injected rows, which is what the filter removes from any base:
    - d1a: authority 1,153, urgency 1,153, system_note 1,117, fake_correction 1,094;
    - training hijack: operator_override 392, disregard_task 387, data_close 386, predecided 385, admin_header 359,
      grader_note 350.
- Files (outside git): `ssdm-data/m2/build/bschema-draw206k-v2/`.
  - Pointer: `data/manifests/bschema-draw206k-v2.pointer.json` (records 09c0fef7..., manifest 8cc03c36..., content
    2eee8e6c...).
  - Raw gz: records 45bdb602..., manifest b210d538..., content 01c7e9fe...

## Checks (all PASS)
- `exclusion-check.json`: b1_check, max item reuse 8. Observed max is 3 per item.
- `generator-specs.json`: 38 specs, including C's family.
- `trainer-draw.json`: `train_set.draw` at n 24k plus `heldout_gate` over all 206,544 rows (0 D1-B hits).
- `attack-filter.json`: 0 hits.
- `audit.json`:
  - 139,100 source items; 59% of rows share an item;
  - exact key-free duplicates across items: 0;
  - MinHash cross-item near-duplicates: 3.5% of sampled rows.
- Tokens: v2f p50 450 / p95 1,472 / max 2,048; v3w p50 455 / p95 1,588 / max 2,048.

## Binary balance (`binary-balance.json`, `binary-balance-index-dev.json`)
- Index DEV binary questions are 8.7% yes, dominated by ACOS (31% of Index binary items, 0.5% yes) and ToolRet /
  BRIGHT per-candidate questions. Training stays at 42.8% yes overall, with each family at its natural rate.
- Open conflict: ACOS is 98.6% "no", so the 60% cap holds it at 2.5% of items (704 records) despite the review's
  "raise ACOS". Raising ACOS needs a looser no-share cap (`MAX_NO_SHARE`) or a higher `FLOOR_V2`.

# bschema-draw230k-v2c (CURRENT v2; supersedes bschema-draw206k-v2 and bschema-draw230k-v2b)

**Takeaway.** 229,824 records = **322,594 trained items**. The draw fits in 2,048 tokens under all three
presentations:
- v1: `jrt-rwkv-g1-v1`, ordered layout, D1's training presentation;
- v2f and v3w: layout g1v2;
- all measured with the real G1k tokenizer and the trainer's own renderers at 9c3e435.

Every check passes.

## What changed (coordinator, 2026-10-08)
- The aggregate 60%-"no" cap is gone (`MAX_NO_SHARE = None`). Index DEV binary questions are 91.3% "no", so the cap
  had pushed training toward "yes". Each family keeps its natural base rate.
- Membership / verify views stay capped at 5% of items: they are 4.1%.
- ACOS weight 8k -> 11k: **ACOS is 4.0% of items** (1,314 records, 1,314 unique items, 12.9k items).
- Binary items are 38.9% yes overall (v2: 42.8%; v1: 18.2%; Index DEV: 8.7%).
- Rows must also be at most 2,048 tokens, with continuations of 16 tokens or fewer, under the v1 TRAINING template
  (`scripts/b_schema_tokcheck.py --set v1`).
  - In v2b, 795 rows (0.35%) were over 2,048 under v1 (max 2,233): C multirecord 682, BFCL 58, ToolACE multi 30, other 25.
  - Across all candidates, 8,350 were over (1,842 of them after the other filters).
  - In v2c those rows are replaced. Max is 2,048 under all three templates, and no row has a continuation over 16.

## Token lengths (v2c)
| Template | p50 | p95 | p99 | max |
|---|---|---|---|---|
| v1 | 461 | 1,618 | 1,857 | 2,048 |
| v2f | 438 | 1,466 | 1,705 | 2,048 |
| v3w | 438 | 1,570 | 1,913 | 2,048 |

## Item shares (v2c)
| Family | Share of items |
|---|---|
| C multirecord | 15.2% |
| CRUXEval | 11.2% |
| glaive | 10.5% |
| BFCL | 8.7% |
| HoVer | 6.3% |
| MNLI | 5.4% |
| RAGTruth | 5.3% |
| ACOS | 4.0% |
| clinc intent | 3.8% |
| ToolACE multi | 3.1% |
| banking intent | 2.9% |
| clinc membership | 2.6% |
| MMLU mcq | 2.2% |
| When2Call | 2.1% |
| GSM8K | 2.0% |
| ToolACE select | 1.6% |
| BoolQ | 1.6% |
| VAST | 1.1% |
| banking membership | 1.1% |
| ANLI | 1.8% |
| Habermas | 0.6% |
| NLI4CT | 0.5% |
| MMLU verify | 0.4% |
| API-Bank | 0.4% |

Rows and unique items per family are in `build-report.json` under `by_key`. Diversity: 156,849 source items; at most
3 rows per item.

## Pins
- Pointer: `data/manifests/bschema-draw230k-v2c.pointer.json`, built at 918a237.
  - Uncompressed sha256: records 0d4a89c429a3be7ae8eae1db74835b7e5fcc8b6d640aee7b0efec9c52e82be04, manifest 62e5a635...,
    content 14a72879...
  - Raw gz: records af52af79..., manifest 7724a97f..., content 5bb2b8d8...

## Checks (all PASS)
- b1_check (reuse cap 8): 0 failures.
- check_generator: 38 specs.
- attack_filter: 0 hits.
- trainer at 9c3e435: `train_set.draw` at n 24k and `heldout_gate` over all 229,824 rows, 0 D1-B hits.
- Audit:
  - exact key-free duplicates across items: 0;
  - MinHash cross-item near-duplicates: 3.1% of sampled rows.
- Agent A's hard gate `train.attack_guard scan` (claude/attack-guard @ca9d482, templates sha 93086366...) passes on all
  229,824 rows: 0 hits, exit 0 (`attack-guard-ca9d482.json`).

# bschema-draw230k-v2d (2026-10-09): v2c minus every eval-document overlap

v2d removes rows from v2c and adds nothing. It is built by `scripts/b_schema_subtract.py`. Record, manifest and
content lines are byte copies of v2c, and v2c's pins are verified before anything is copied. Decisions are the
coordinator's (2026-10-09): eval documents count as "test" (we can't train on test), and dropping is cheaper than arguing.

## Removed: 1,878 rows (229,824 -> 227,946 rows; 322,594 -> 320,716 items)
- The whole NLI4CT family: 1,710 rows. This includes the 18 NLI4CT doc_check hits.
- The other 164 v2c `data/doc_check.py` hits (`../bschema-draw230k-v2c/doc-check.json`):
  - glaive_fc 99 and when2call 60. These are BFCL tool schemas, plus 8 HoVer and 5 ESCI hits.
  - mmlu_aux 2, sst5 2, hellaswag 1 (HoVer/ESCI).
- 4 rows that `train.doc_overlap_guard` (PR #235, claude/b-doc-overlap-gate @4849e36) failed on the first v2d cut,
  all under rule R: hellaswag 1, mmlu_aux 1 (RouterBench frame), winogrande 2 (BRIGHT pool).
  - These rows only appear once v2c's NLI4CT rows are gone, because the per-draw boilerplate and template dfs shift.
  - The ids are in `doc-overlap-guard-4849e36-try1.json`.
- Removed counts per list and the sorted-id sha256 of each list are in `subtract-report.json`.

## Gates (all PASS on the final 227,946 rows)
- `doc-overlap-guard-4849e36.json`: `train.doc_overlap_guard scan` at 4849e36 is the final gate.
  - Result: 0 fails, exit 0.
  - The PR's allowlist applies (context corpora only). Rows it saved are counted under `allowlist_saved`.
- `doc-check.json`: `data/doc_check.py`, with the allowlist narrowed to the single pair (anli, HoVer).
  - Result: 0 non-allowlisted hits.
  - The 1 allowlisted row is the ANLI -> HoVer Wikipedia row.
- `exclusion-check.json`: b1_check, max item reuse 8, 0 failures. Log: `ssdm-data/m2/build/bschema-draw230k-v2d-check.log`.
- `generator-specs.json`: check_generator passes 37 specs. That is v2c's 38 minus nli4ct.
- `attack-filter.json`: 0 hits.
- `attack-guard-ca9d482.json`: `train.attack_guard scan` gives 0 hits.
- `trainer-draw.json`: trainer 9c3e435, `train_set.draw` at n 24k, plus `heldout_gate` over all rows. 0 hits.
- `token-check.json`: re-tokenized on v2d itself, 0 errors.
  - The max is 2,048 under v1, v2f and v3w.
  - Continuations: at most 1 token under g1v2 and at most 14 under v1.
- `audit.json`: 0 key-free collisions across items; MinHash cross-item near-duplicates 2.6% of sampled rows.

## Pins
- Pointer: `data/manifests/bschema-draw230k-v2d.pointer.json`, sha256 1adf0c82...
  - The pointer records builder commit e58e3b9, the parent HEAD. The subtract script itself lands in the commit that
    adds this receipt.
- Uncompressed sha256:
  - records b5149e12e089d0eddde9a0fc0fa801ce9530766c07a0db17a00817a372a3bde3
  - manifest e403fd52b39b048f2631effea649a6e0d0f0402a80f7f8eee7d53454f500baa7
  - content 5f661509ab99cabaf88d2006198764d6e9981b8a95f756908fba53e0c17b2f2e
- Raw gz:
  - records fd79eaaa605c83da1a58a55eb482030adbfee973de7981e2f44840586f5e08a6
  - manifest 1f9514eb...
  - content 38b5b715...
- The C multirecord rows are still v2c's `synthetic_c_multirecord_v1`, 7,000 rows. C's all-K admitted pool has not
  been swapped in.

## v2d at the adopted policy
`doc-overlap-guard-1f64368.json`: `train.doc_overlap_guard` at origin/claude/b-doc-overlap-gate @1f64368 (index
2c821e29..., allowlist 8a0bf416...) reports 0 fails over all 227,946 rows. v2d is unchanged.

# bschema-draw230k-v2e (2026-10-09): v2d with C's all-K pool. READY (v2f/v3w; under v1, 7,942 C rows exceed 2,048 and are dropped by the trainer)
v2e is v2d with its 7,000 `synthetic_c_multirecord_v1` rows replaced by all 13,575 rows of C's
`c-multirecord-13575-v2-allk-admitted` pool (pointer sha256 531faa15...). Built with `scripts/b_schema_swap_pool.py`.
- Every pool row passed the P3 Checker against the kept v2d rows: 13,575 of 13,575 kept.
- Size: 234,521 rows / 366,741 items. C's share of items is 25.9%.

Gates:
- PASS:
  - doc_overlap_guard @1f64368: 0 fails.
  - doc_check: 0 non-allowlisted hits (1 allowlisted).
  - attack_filter: 0 hits.
  - attack_guard @ca9d482: 0 hits.
  - b1_check: 0 failures.
  - check_generator: 37 specs.
  - Trainer 9c3e435: draw at 24k, and heldout_gate with 0 hits.
  - Tokens: max 2,048 under v2f and v3w.
- Under v1 (not verified; coordinator decision 2026-10-09: current cells train under v3w): `token-check.json` shows 7,942 C rows over 2,048 under the v1 TRAINING template (max 2,309). The trainer drops (does not cut) rows over T.
  - Every over-length row is from the C pool.
  - C proved v2f/v3w only.
  - Those rows are exactly the pool's high-K worlds: option count 53-77. The rows that fit under v1 span 5-62.
- Pins:
  - pointer `data/manifests/bschema-draw230k-v2e.pointer.json`, sha256 94290d18872d41a6b0a06cfbe40780f5f4ff9160dc1c87a90904eed09b033314
    - It carries `templates_verified` = [jrt-rwkv-g1-v2f, jrt-rwkv-g1-v3w].
    - It carries `v1_overlength_ids` = `v1-overlength-ids.txt` (7,942 ids, sha256 00783161...).
    - `trainer-draw.json` was re-run against this pointer. The first pointer (7bcbaca5...) is superseded.
  - uncompressed records 9967766bf5fdd292f2e5d34bff9e6eda79a027b299ee6cf55c11ec94dada4081
  - manifest a4ec3875...
  - content 3789c913...

# bschema-draw-v3 (2026-10-09): v2e plus data-lever rows. READY for v2f/v3w
Built from v2e (C's all-K pool kept) plus 20,015 lever rows.
- Levers come from train splits on disk: `data/b_schema_levers.py`.
- Each lever row keeps <= 2,048 tokens under v2f and v3w: `scripts/b_schema_levers_finalize.py`.
- The merge applies the P3 checks: `scripts/b_schema_swap_pool.py --keep-aug`.
- No target comes from DEV or eval rows. Every target's file, split and computed value is in `target_provenance.json`,
  which is also inlined in the pointer.

Size: 254,535 rows / 386,755 items.

Removed from v2e:
- 1 anli row, which failed the doc guard at 6d3a4c1 (the allowlist no longer applies to rows of the eval family's own
  source).

Lever rows added:

| Lever | Rows | K | Notes |
|---|---|---|---|
| API-Bank long catalogs | 2,273 | 17-32 | Largest K that fits 2,048 with one-line descriptions. The lv1 train pool has 1,767 APIs. 145 rows over 2,048 and 285 with K_fit < 17 were dropped. |
| MMLU aux 10-option | 6,969 | 10 | Gold unchanged. Distractors are non-gold options of rare-word-nearest aux questions, never equal to an own option. |
| BoolQ {yes, no} choice | 6,246 | 2 | Every train item not held out by a cohort. 62.1% yes, the train rate. |
| BANKING77 77-option | 4,527 | 77 | Digit ids only; others do not fit. The draw's K77 share is 2.215%, at or above D16's 2.204% (2,089 / 94,799). |
| CLINC 151-option | 0 | 151 | None fit under v3w (min 3,174 tokens). CLINC out-of-scope stays at its train share. |

Gates (all PASS):
- doc_overlap_guard @6d3a4c1 (index af096960, allowlist 556296cc): 0 fails.
  - The first cut failed 4 rows: the anli row above and 3 BANKING77 K77 rows near-identical to test utterances. All 4
    were dropped.
- doc_check: 0 non-allowlisted hits.
- attack_filter: 0 hits.
- attack_guard @ca9d482: 0 hits.
- b1_check: 0 failures.
- check_generator: 37 specs.
- Trainer 9c3e435: draw at 24k, and heldout_gate with 0 hits.
- Tokens: max 2,048 under v2f and v3w. Under v1 the same 7,942 C rows as v2e are over.

Shares vs v2e: `shares-vs-v2e.json`.
- Binary yes share: K=2 yes/no-id choice 20.2% -> 24.4%; including noul, 37.9% -> 38.9%.

# bschema-draw-v3b (2026-10-09): v3 with name-only API-Bank catalogs and the review fixes. READY for v2f/v3w
This replaces v3. The independent review of v3 failed on B1 and B2; both are fixed here.
- Base: v2e minus the anli doc-guard row, plus lever rows regenerated with the fixes:

| Lever | Generated | Index fingerprint drop | Over 2,048 | Other drops | In draw |
|---|---|---|---|---|---|
| API-Bank name-only, K 40-60 | 2,738 | 454 | 490 | P3 7 | 1,787 |
| MMLU K10 | 7,019 | 912 | 0 | P3 5; doc guard 1 | 6,101 |
| BoolQ yes/no | 6,303 | 142 | 0 | P3 2 | 6,159 |
| BANKING77 K77 | 6,315 | 947 | 13 | trimmed to 4,496; P3 17 | 4,479 |

- Draw K77 share: 2.209%, at or above D16's 2.204%. Draw size: 253,046 rows / 385,266 items.
- The finalize stage now runs the item-tier Index fingerprint (`data.dedup.Fingerprint`, production v3; exact, ngram13,
  minhash, edit and contain) on every lever row, so the B1 leak cannot recur.
- `fingerprint-recheck.json` re-runs it on the final draw: 0 hits. None of the 1,014 ids the review flagged are present.

Review fixes:
- B2: all 10 MMLU options are shuffled before ids are assigned. Gold-position chi2 = 3.9 (df 9, p 0.92).
- N1: no borrowing from NOT/EXCEPT/false-stem questions, and no K10 view of a negated-stem target (667 skipped).
- N2: options are borrowed from other passages first.
- N3: API-Bank synonym and case-variant distractors are removed by name word-set.
- N4: half the BoolQ rows use letter ids in random order (letter half: 1,541 / 1,554).
- N5: post-check counts are in `target_provenance.json` and the pointer.

API-Bank eval-catalog check:
- Each built catalog's API-name set was compared with all 508 Index API-Bank option sets (DEV 228, TEST 228,
  held-out 52; all K=53).
- Max Jaccard 0.030; 0 refused at >= 0.8.

Gates (all PASS):
- doc_overlap_guard @6d3a4c1: 0 fails.
- doc_check: 0 non-allowlisted hits.
- attack_filter: 0 hits.
- attack_guard @ca9d482: 0 hits.
- b1_check: 0 failures.
- check_generator: 37 specs.
- Trainer 9c3e435: draw at 24k, and heldout_gate with 0 hits.
- Tokens: max 2,048 under v2f and v3w. Under v1 the same 7,942 C rows are over.
