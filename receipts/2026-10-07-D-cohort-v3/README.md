# Receipt: m2 development cohort v3 (rebuilt after the PR #105 review: text twins and passage near-duplicates excluded)

**Result.**
- `train/cohorts/m2-dev-cohort-v3.jsonl.gz` is frozen. The uncompressed sha256 is
  **`498efa617c931a5ea073bc5c62ce94158270888f64b9bbdb027982cde43876da`**, pinned in `tests/test_cohort_v3.py`.
  - The first build was `0c0309b9…`. It is superseded by review B1 and the coordinator's stricter rule.
- Every cohort v2 entry is unchanged and in order.
- The G13 `retention` role has **39,872 records**: the 1,670 PR #16 records plus 38,202 records of the PR #44 150k set.
- Measured usable items (K ≤ 16, T ≤ 2,048, the real template and tokenizer):
  - **36,696 on RWKV7-G1k** and **36,691 on Qwen3.5 / Q9**;
  - this is ≥ `REQUIRED_N` 33,300 × 1.08 = 35,964 on both parents.
- Margins are unchanged.
- Billed $0; everything ran on CPU locally.

## What changed in the rebuild (review FIX-THEN-MERGE)

**B1: a key-free text union key.** `train.cohort.record_text_keys` is the review normalizer:
- NFKC, casefold, punctuation stripped, dict keys and option ids dropped;
- `t:` = state + instructions + sorted option texts;
- `s:` = state + sorted option texts, when longer than 40 chars.

The key is applied in three places:
- **Union key in `select_retention`.** A text twin of any reserve record, or of any PR #16 record, makes its group ineligible.
- **Stored per new entry** as `text_keys`.
- **Mirrored in the guards.** `Cohort.assert_draw_disjoint(..., records=)` and `training_exclusion(..., text_keys=)` both check it. `train_set.draw` and `clef_teacher` now pass the drawn records.

All 196 review pairs (`review-b1-pairs.json`) are out of v3. That includes `m2:glaive_fc:100743` and `m2:factchange:renewal:1124:v0`.

**Passage near-duplicates (coordinator decision).** A candidate group is ineligible when a member shares ≥ 50 % of its
sampled word 13-gram STATE shingles with ONE record. That record can be:
- among the first 60,000 of the seed-0 draw order; or
- in the PR #16 20k sample (its records file is read, and every line is verified against the committed manifest).

Sampling is consistent, keeping hashes ≡ 0 mod 8. 18,567 candidates were flagged: 13,036 against the reserve and 5,531 against PR #16. By source:

| Source | Flagged |
|---|---:|
| glaive_fc | 4,595 |
| mmlu_aux | 3,530 |
| anli | 1,874 |
| when2call | 1,680 |
| synthetic_factchange | 1,296 |
| boolq | 1,011 |
| toolace | 932 |
| others | 3,649 |

Shingles are taken over the state only. A first trial over state + options flagged 35,930, because shared label sets (banking77/clinc150 intents) and tool catalogs dominated. That is a shared template, not a shared passage.

**Non-blocking review items fixed:**
- **(2)** `training_exclusion(pool=...)` raises `PoolMismatch` when a pinned 150k build lacks any v3 record.
- **(3)** The cohort test floor is `REQUIRED_N` (33,300), plus the 1.08 headroom.
- **(6)** `Cohort(path)` always verifies a committed cohort against its pin. Any other path needs `expected_sha256` or `allow_unpinned=True`.

The selection was replenished with the same construction (whole groups, quotas, seed 20261006), and `n_new` was raised to 38,200.

| Ineligible (of 150,000) | Records |
|---|---:|
| `draw_reserve`: groups touching the first 60,000 of the seed-0 order, now also via text twins | 69,542 |
| `pr16_sample`: source item, content or key-free text shared with any PR #16 record | 26,744 |
| `shingle_reserve` / `shingle_pr16` (group-level) | 14,843 / 6,109 |
| `injected` | 13,168 |
| `heldout_excluded` | 3 |
| guard hits (SplitGuard + content index, frozen-9, D1-A/D1-B) | 0 |
| cohort v2 | 0 |

There were 53,077 eligible records in 52,929 groups. 38,202 were selected.

## Commands

```sh
python -X utf8 -m train.cohort_v3 build --train-dir <ssdm-data>/m2/build/sample-150000-seed20261006   # ~6 min
python -X utf8 receipts/2026-10-07-D-cohort-v3/measure.py train/cohorts/m2-dev-cohort-v3.jsonl.gz \
    receipts/2026-10-07-D-cohort-v3/measure.json --workers 20                                     # ~15 min
COHORT_V3_REBUILD=1 python -X utf8 -m pytest -q tests/test_cohort_v3.py -k rebuild               # pytest-rebuild.txt
python -X utf8 -m train.margin_sim > receipts/2026-10-07-D-cohort-v3/margin_sim.json
```

## Disjointness evidence

All of the following are recorded in the header and re-checked by the build (fail-closed) and by the tests:

| Against | How | Result |
|---|---|---|
| m2 training draws: first cell 24k, q9_feature_cache, teacher | Nothing from groups touching the 60k seed-0 reserve (id, content, item, key-free text), and no ≥ 50 % state-shingle overlap with any reserve record. The rebuild test checks that the 24k and 60k seed-0 draws are byte-identical with and without v3, and that `training_exclusion(150k, pool=…)` == exactly the 38,202 v3 ids | 0 |
| PR #16 pilot draws | id / content / item / key-free text / passage vs all 20,000 PR #16 records | 0 |
| Index DEV/TEST | SplitGuard (manifests + content index `7cbe6c89…`) over all 82,710 candidates | 0 |
| D1-A / D1-B, frozen 9 families | injected records never eligible; template scans; `check_record` | 0 |

## Composition delta (`measure.json` `composition_retention_role`; v3 first build → rebuild)

| | first build (0c0309b9) | rebuild (498efa61) |
|---|---:|---:|
| records (v2 + new) | 1,670 + 38,002 = 39,672 | 1,670 + 38,202 = 39,872 |
| usable items, RWKV7 / Q9 | 36,005 / 35,992 | **36,696 / 36,691** |
| knowledge / language / retrieval / tools / arts (item parts) | 11,493 / 11,401 / 10,525 / 7,632 / 2,834 | 13,067 / 11,862 / 11,440 / 5,539 / 2,853 |
| area share K / L / R / T / A | 26.2 / 26.0 / 24.0 / 17.4 / 6.5 % | 29.2 / 26.5 / 25.6 / 12.4 / 6.4 % |
| choice / noul / score questions | 46,918 / 2,588 / 3,659 | 50,635 / 3,036 / 3,399 |
| length <64 / 64-255 / 256-1023 / 1024-4095 / ≥4096 words | 6,263 / 15,005 / 14,882 / 2,090 / 1,432 | 6,775 / 13,249 / 15,276 / 2,701 / 1,871 |

New records by source, first build → rebuild:

| Source | First build | Rebuild |
|---|---:|---:|
| mmlu_aux | 8,922 | 10,075 |
| glaive_fc | 4,291 | 2,469 |
| nq | 2,503 | 3,310 |
| hellaswag | 2,140 | 2,815 |
| mixed | 2,397 | 2,760 |
| boolq | 2,133 | 2,556 |
| winogrande | 1,575 | 2,053 |
| clinc150 | 2,152 | 1,760 |
| toolace | 1,373 | 1,741 |
| humicroedit_s2 | 1,266 | 1,642 |
| gsm8k | 1,139 | 1,512 |
| banking77 | 1,595 | 1,315 |
| humicroedit_s1 | 1,568 | 1,211 |
| anli | 2,370 | 1,183 |
| when2call | 1,266 | 877 |
| sst5 | 944 | 805 |
| synthetic_factchange | 368 | 118 |

**Tools drops** from 17.4 % to 12.4 % of item parts. Tool-catalog states and the synthetic fact-change generator repeat across records, so the stricter rule removes them.

## Per-arm cost of the retention term (`measure.json` `cost`; exact fork plan × measured rates)

| Backbone | Positions | Calls | Min / arm | $ / arm |
|---|---:|---:|---:|---:|
| RWKV7-13.3B, job-tkfaj (3,757 tok/s, 0.117 s/call) | 22,877,944 | 63,741 | 225.8 | 11.18 |
| Q9 / Qwen3.5-9B, job-5cmat (22,144 tok/s, 0.0595 s/call) | 22,106,052 | 63,583 | 79.7 | 3.94 |

- **Conservative bound** (pack-call latency per call): RWKV7 738.9 min / $36.58; Q9 267.7 min / $13.25.
- **`VOLUME_TOKENS` upper bound:** RWKV7 158.9 M (704.9 min / $34.89); Q9 162.1 M (122.0 min / $6.04).

## margin_sim (`margin_sim.json`; margins unchanged)

- Retention `EVAL_SIZES` = 36,691.
- Both retention criteria pass with a simulated null pass rate of 1.0 at that size, and `SHORTFALLS` is empty.
- The 1/sqrt(n) re-estimate is ~7.5k (accuracy) and ~7.9k (ECE); the `REQUIRED_N` pin (33,300) is kept.

## job-8vjve (q9_feature_cache stock_train @ 06e9733; `job-8vjve-exclusions.json`, `job_8vjve_exclusions.py`)

The job's rows were recomputed with the Qwen3.5 template + tokenizer and `keep_item`, in draw order:
- **20,000 rows from 17,917 records.** The last record is at draw position 22,524. This matches `volume_plan.json`.
- **The job's rows do overlap the records the rebuild removed.** Of the 10,056 records dropped from the first v3 build, 5,820 have a twin or passage neighbour in the job's rows: 2,253 key-free text twins and 3,567 passage-only.
- **3,251 distinct job records are involved.** They are listed by draw position in the JSON; the first few are at positions 7, 14, 20, 21 and 22.
- These are training-cache records. v3 now excludes their twins, so the job needs no change.
- **The rebuilt v3 overlaps the job's rows 0 by id and 0 by key-free text.** By construction it also has no passage overlap with any reserve record.

The earlier draw check (`q9-cache-draw-check.json`) still holds: the 24k draw is `2bbbf180…`.

## Limits

- **Jobs still pin v2.** Moving to v3 means four things: `pool_dirs`, `VOLUME_TOKENS`, re-sized caps and a re-cached parent arm.
- **Power inputs** are `margin_sim` assumptions until the first real G13 run.
- **Off-set draws.** Text-twin exclusion BEFORE a draw needs the set's text keys. Within the 150k set every twin of a v3 record is inside v3 (whole groups), so id exclusion covers it there. For any other set, the post-draw check `assert_draw_disjoint(records=…)` refuses a twin.
