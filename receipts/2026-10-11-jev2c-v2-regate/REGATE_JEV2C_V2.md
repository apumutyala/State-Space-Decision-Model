# REGATE_JEV2C_V2: independent re-gate of bschema-draw-jev2c-v2 (evidence lane, 2026-10-11)

**Verdict: LEAK-FREE and well-formed. Cleared.** The exclusion list is empty. All 1,500 long_policy rows are synthetic.
They have 0 13-gram and 0 verbatim-leaf overlap with the JevBench public items, Index DEV / TEST / HELDOUT / proxy600,
and GSM8K test. Every gate passes on agent-main code, and 3/3 hand-checked golds (including 1 score row) are correct.

Draw: claude/ev-jev2-draw @1be15168, `ssdm-data/m2/build/bschema-draw-jev2c-v2/`. The procedure, code and gate modules
are the same as for REGATE_JEV2C. Between a4aaa7de and 1be15168 only `scripts/jev2_mix.py` and the receipts changed.
No gate code or pin changed.

## 1. Identity (hashes.json)
| claim | result |
|---|---|
| pointer `data/manifests/bschema-draw-jev2c-v2.pointer.json` sha256 86960d90… | CONFIRMED (git blob @1be15168) |
| records uncompressed 0bdb1ac6… (gz 42976c8f…), manifest 75b7665e…, content map c2d40e76…; 1,500 lines each | CONFIRMED |

## 2. Gates (agent-main code, frozen pins)
| gate | result | verdict |
|---|---|---|
| doc_overlap_guard (Index af096960…, JevBench 0fe994c4…) | 0 / 0 / 0 | CONFIRMED |
| attack_guard (5e2d6e01…) | 0 | CONFIRMED |
| jevbench_guard rescan (doc, exact, 5 fingerprint matchers) | 0 / 0 / 0 | CONFIRMED |
| Index Fingerprint index-fp-v3 (assert_production ok) | item-tier 0 and **strict full-row 0** | CONFIRMED |
| cohort v2 / v3 (id, base id, content, source item, key-free text) | 0 | CONFIRMED |
| frozen held-out `check_record` | 0 | CONFIRMED |
| content map equals record_content_hash | 0 mismatches | CONFIRMED |

## 3. Policy-text overlap (own matcher, overlap.json; 2,996,938 training 13-grams)
| eval side | 13-gram rows hit | exact leaf / state |
|---|---|---|
| JevBench 231 public items (incl. the 19 long_policy items) | 0 | 0 / 0 |
| Index DEV / TEST / HELDOUT (12,366 / 12,423 / 128,608) | 0 / 0 / 0 | 0 / 0 |
| proxy600 (644) | 0 | 0 |
| GSM8K official test (1,319) | 0 | 0 |
| verbatim ≥ 6-word leaf vs JevBench + every Index row | n/a | **0 rows → `jev2c-v2_exclude_ids.txt` empty** |

## 4. Provenance
`source` = jev2synth on 1,500/1,500 rows, `source_split` "synthetic", licence "own generator (repo)", 0 missing.
There is no ContractNLI row and no other public-dataset row, so there is no non-TRAIN source. The builder's check stage
dropped 3 Index-Fingerprint hits (pool 1,903 → 1,900).

## 5. Gold hand-checks (`scripts/regate_jev2c_spotcheck.py`; synthetic rows)
| row | rules applied | hand result | gold |
|---|---|---|---|
| long_handbook:0, access, choice | Confidential data requested; untrained (2.4 would refuse); manager approval given. 2.6: an employee on the incident response team gets the requested access "whatever the other rules say" | grant | grant ✓ |
| long_handbook:1001, leave, noul | 2 days of emergency leave with 2 days' notice. 1.6: emergency leave of up to two days is always granted, unaffected by other clauses | true | true ✓ |
| long_handbook:1011, expense, **score** | Trip ended day 11, so the old $75 meal limit applies (amendment from day 17). Filed 5 days later (≤ 30). Total $215 ≤ $800, so no pre-approval needed. Meals $55 ≤ 75, hotel $120 ≤ 220, taxi at cost, all receipts attached | whole amount paid → level 2 | 2 ✓ |

## 6. Well-formedness (format.json, tokens-shim.json)
- **Gold:** 0 failures. **Empty criteria:** 0. **Empty option descriptions:** 0.
- **Types:** choice 551 (K 3/4/5/6 = 78/200/135/138), noul 655, score 294 (19.6 %). **State:** plain string, 1,500/1,500.
- **Option order kept:** 20/551 rows already sorted, against 22.7 expected. **Gold-position χ²:** p 0.334 (builder 0.334).
  CONFIRMED.
- **Duplicates:** 0 exact, 0 repeated states, 0 digit-masked near-duplicates.
- **Tokens under jrt-h2o-shim:qwen3.5-4b** (f0be274c… equals the pin): min 2,007, p50 2,957, p90 3,652, max 3,981,
  0 rows over 4,096, 0 errors. This matches the builder's 2,007 / 3,650 / 3,981, up to the quantile convention.
  **1,497 of 1,500 rows exceed 2,048 tokens**, so the cell needs max_len ≥ 4,096. jev2_mix.py's default `--max-len
  4096` covers this.
- **Builder shape gate:** passes now (all_pass true; p90 3,650 against the public 3,606), which fixes jev2c caveat (B).
