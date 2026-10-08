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
