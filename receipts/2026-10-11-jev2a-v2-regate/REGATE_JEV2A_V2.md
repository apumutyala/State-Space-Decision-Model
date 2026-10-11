# REGATE_JEV2A_V2: independent re-gate of bschema-draw-jev2a-v2 (evidence lane, 2026-10-11)

**Verdict: LEAK-FREE after a 23-row exclusion, and well-formed.** One caveat: train-split provenance for 104 MuSiQue
rows could not be checked, because the MuSiQue train file is no longer on disk (section 4).

- **Exclusion list:** `jev2a-v2_exclude_ids.txt` holds 23 ids, sha256 c0c175f50a7886b53bc647e022bb00705747deadf535a2708fef19c32ab9c2ad.
- **Hand-checked golds:** I did not hand-check golds in this round. The jev2a hand checks cover the same generators.
- **No eval text** appears in this file.

Draw: claude/ev-jev2-draw @282a4867, built from 1be15168, `ssdm-data/m2/build/bschema-draw-jev2a-v2/`. The code is
the same as for jev2a, plus `scripts/regate_musique_transitive.py`. No gate module or pin has changed.

## 1. Identity
| claim | result |
|---|---|
| pointer `bschema-draw-jev2a-v2.pointer.json` sha256 eba70d2c… | CONFIRMED (git blob @282a4867) |
| records uncompressed 00a1d722… (gz 7bcb7489…), manifest ce92cf8a…, content map 35c5cb2e…; 7,500 lines each | CONFIRMED |
| 1,500 rows each for multi_hop / judge_hard / temporal_numeric / fact / tool_selection; unique ids | CONFIRMED |

## 2. Gates (agent-main code, frozen pins)
| gate | result | verdict |
|---|---|---|
| doc_overlap_guard / attack_guard / jevbench_guard rescan | 0 / 0 / 0 | CONFIRMED |
| Index Fingerprint, item-tier probe | 0 | CONFIRMED |
| Index Fingerprint, strict full-row probe | 264 rows; state-segment hits: 27 short MuSiQue title segments and 2 gsm8k judge rows (contain) | INFO, same pattern as jev2a |
| cohort v2 / v3 (id, base id, content, source item, key-free text) | 0 | CONFIRMED |
| frozen held-out | 0 | CONFIRMED |
| content map equals record_content_hash | 0 mismatches | CONFIRMED |

## 3. Overlap (own matcher) and the verbatim-leaf rule
| eval side | 13-gram train rows | exact leaf rows |
|---|---|---|
| JevBench 231 | 0 | 22 (glaive tool-catalog descriptions; see below) |
| Index DEV / TEST / HELDOUT | 5 / 1 / 4 glaive rows (5 distinct), all on ToolRet-retrieval candidate-tool descriptions (allowlist `toolret-public-tool-corpus`) | 0 / 0 / 1 (MuSiQue title, BBH movie_recommendation option) |
| proxy600 | 0 | 0 |
| GSM8K official test | 0 | 0 |

**Exclusion list (`exact-leaves-all.json`):** 23 rows.
- 22 tool_selection/glaive rows carry one of the JevBench easy tool_selection catalog descriptions. In 12 of them that
  description is the gold option.
- 1 MuSiQue row (m2:musique:2hop__45324_576533) has a document title that equals a BBH option.
- The HLE "St. John's" title rows from jev2a are not in this draw.
- The 5 ToolRet-description rows are NOT in the list. They are allowlisted by design and do not meet the verbatim-leaf
  rule. If the lead wants them excluded as well, the ids are m2:glaive_fc:{10634, 4893, 5923, 6959, 8152}~view:jev2v2-tool.

## 4. Provenance
- **GSM8K train:** 622/622 rows match their train-parquet row; 253/253 choice golds equal the source answer.
- **BoolQ train:** 386/386 rows match. **Glaive:** 726/726 rows match.
- **Licences** are present on all rows. 5,400 rows are synthetic.
- **MuSiQue (366 rows):** the train file (`raw/jev2/musique/musique_ans_v1.0_train.jsonl`, sha 83a75b1e…) is no
  longer on disk, so I checked by transitivity (`musique-transitive.json`).
  - 262/366 rows share their base id with a jev2a row that I verified against that train file. Their documents are a
    subset of the verified documents, or appear verbatim in the prose state.
  - **104 rows have a base id that jev2a did not cover, so their split is UNVERIFIED**
    (`jev2a-v2_musique_provenance_unverified_ids.txt`, sha256 5cebf60b…).
  - All 366 carry a `2hop/3hop/4hop__` id and `source_split` "train", but MuSiQue dev ids look the same, so this is
    not proof.
  - There is still no leak signal: these rows have 0 13-gram and 0 exact overlap with every Index split, including
    HELDOUT.
  - To close this, restore the train file from the builder's recorded revision (sha 83a75b1e…) and re-run
    `regate_jev2a.py provenance`, or exclude the 104 rows conservatively.

## 5. Well-formedness
- **Gold:** 0 failures. **Empty criteria:** 0. **Empty option descriptions:** 0. **Option order kept:** sorted counts
  13/12.5, 35/40.6, 34/31.0.
- **Gold-position χ²:** tool_selection 0.93, temporal 0.58, multi_hop 0.86.
- **Types:** temporal choice 1,068 / noul 274 / score 158; multi_hop choice 1,216 / noul 168 / score 116. The other
  three families are single-type, as in jev2a.
- **State:** multi_hop 676 dict / 824 prose; temporal 431 dict / 1,069 prose.
- **gsm judge:** corruption is consistent with the label in every row.
- **Duplicates:** 0 exact. Repeated-request groups: glaive 461 rows. Digit-masked near-duplicates: temporal 459,
  judge 192 and fact 23 extra rows.
- **Tokens (jrt-h2o-shim:qwen3.5-4b, equals the pin):** max 2,581 (multi_hop p50 1,875), 0 rows over 4,096, and
  **350 multi_hop rows over 2,048**. The pointer's token window is [0, 4,096], so the cell needs max_len 4,096. The
  mix default covers this. 0 errors.
