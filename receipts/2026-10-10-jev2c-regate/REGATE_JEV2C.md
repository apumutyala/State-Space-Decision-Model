# REGATE_JEV2C: independent re-gate of bschema-draw-jev2c (evidence lane, 2026-10-10)

**Verdict: LEAK-FREE. Every row is synthetic, with no non-TRAIN source and no ContractNLI row in the final draw.** All
1,500 long_policy rows have 0 13-gram and 0 verbatim-leaf overlap with the JevBench public items, Index DEV / TEST /
HELDOUT / proxy600, and GSM8K test.
Two caveats for the lead, neither of them a leak:
- **(A)** 1,448 of 1,500 rows exceed 2,048 shim tokens (max 3,347). This draw needs a ≥ 3.4k training context.
- **(B)** The builder's own `shape_gate` FAILS (length_p90 check), even though GATES.json reports top-level
  `all_pass: true`.

This file contains no eval text. The code is the same as for jev2a (`scripts/regate_jev2a.py`,
`scripts/regate_exact_leaves.py`, agent-main gate modules, byte-identical to the builder branch).

Draw: claude/ev-jev2-draw @12461fff, `ssdm-data/m2/build/bschema-draw-jev2c/`.

## 1. Identity (hashes.json)
| claim | result |
|---|---|
| pointer `data/manifests/bschema-draw-jev2c.pointer.json` sha256 b9f82ab1… | CONFIRMED (git blob @12461fff) |
| records uncompressed 0a297a58… (gz 79afafec…), manifest aefdc2e6…, content map d2e9d344…; 1,500 lines each | CONFIRMED |
| 1,500 long_policy rows | CONFIRMED |

## 2. Gates re-run (agent-main code, frozen pins)
| gate | result | verdict |
|---|---|---|
| doc_overlap_guard (Index af096960…, JevBench 0fe994c4…) | 0 / 0 / 0 | CONFIRMED |
| attack_guard (5e2d6e01…) | 0 | CONFIRMED |
| jevbench_guard rescan (doc, exact, 5 fingerprint matchers) | 0 / 0 / 0 | CONFIRMED |
| Index Fingerprint index-fp-v3 (assert_production ok) | item-tier 0 **and full-row 0** | CONFIRMED |
| cohort v2 / v3 (id, base id, content, source item, key-free text) | 0 | CONFIRMED |
| frozen held-out `check_record` | 0 | CONFIRMED |
| content map equals record_content_hash | 0 mismatches | CONFIRMED |

## 3. Policy-text overlap (own matcher, overlap.json; 2,692,447 training 13-grams)
| eval side | 13-gram rows hit | exact leaf / state | verdict |
|---|---|---|---|
| JevBench 231 public items (incl. the 19 long_policy items) | 0 | 0 / 0 | CONFIRMED |
| Index DEV / TEST / HELDOUT (12,366 / 12,423 / 128,608), incl. the ContractNLI test track | 0 / 0 / 0 | 0 / 0 | CONFIRMED |
| proxy600 (644) | 0 | 0 | CONFIRMED |
| GSM8K official test | 0 | 0 | CONFIRMED |
| verbatim ≥ 6-word leaf check (JevBench + every Index row; `exact-leaves-all.json`) | n/a | 0 rows | CONFIRMED; `jev2c_exclude_ids.txt` is empty |

## 4. Source and split provenance
- Final draw: `source` = jev2synth for 1,500/1,500 rows, `source_split` "synthetic", licence "own generator (repo)".
  0 rows have a missing licence.
- **ContractNLI:** the generator offered 300 ContractNLI rows (`subset == "train"` only, HF mirror kiddothe2b,
  zip ab05047d…). The check stage dropped all 303 Index-Fingerprint hits (GATES.json: n_pool 2,103 → kept 1,800,
  `index_fingerprint` 303). My scan of the final draw finds 0 contractnli rows. **No ContractNLI text of any split
  reaches training.** The mirror's subset labels therefore do not need verifying for this draw. They WOULD need
  verifying against the official split files before any later draw keeps ContractNLI rows: the Index scores the
  ContractNLI test split, and these rows were Fingerprint hits.
- The 1,800 → 1,500 trim is the token window plus the family target.

## 5. Well-formedness (format.json, tokens-shim.json)
- **Gold:** 0 failures. **Option order** is kept (33/778 rows already in sorted order, 33.9 expected). **Gold-position
  χ²:** p 0.848 (builder 0.848). CONFIRMED.
- **Types:** choice 778 (K 3/4/5/6 = 122/282/193/181), noul 647, score 75. **State:** plain string for 1,500/1,500.
- **Empty criteria: 75 rows** (the score questions). 0 empty noul criteria, and there are 24 distinct noul criteria pairs.
- **Duplicates:** 0 exact, 0 repeated states, 0 digit-masked near-duplicates.
- **Gold spot-check** (m2:jev2synth:long_handbook:0, access domain): the case is an employee on the incident response
  team, untrained, asking for confidential data. Rule 6 (incident override, "whatever the other rules say") applies,
  so the correct answer is grant. The gold is grant. CORRECT.
- **Tokens under jrt-h2o-shim:qwen3.5-4b** (identity f0be274c… equals the pin; trainer row builder via tokcheck):
  min 2,001, p50 2,642, p90 3,159, max 3,347, 0 errors, 0 rows over 4,096, **1,448 rows over 2,048**. These match the
  builder's figures. Caveat (A): a 2,048-token cell would truncate these rows.
- **Caveat (B):** the builder's `shape_gate.json` gives `all_pass: false`. The draw's token p90 is 3,158, against the
  public long_policy shape p90 of 3,606. GATES.json still says `all_pass: true` because the shape gate is not part of
  that conjunction. This is a calibration-quality issue, not a leak. Note that the shape gate compares against eval-item
  SHAPE stats: `public-shape-fingerprint.json` holds counts and token quantiles only, and I checked it contains no
  string over 4 words except its `what` description.
