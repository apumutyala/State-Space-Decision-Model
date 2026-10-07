# Receipt: m2 development cohort v3 (retention role expanded to 39,672 records / ~36k items)

**Result.**
- `train/cohorts/m2-dev-cohort-v3.jsonl.gz` is frozen. The uncompressed sha256 is
  **`0c0309b9ec96dec8907e5c421925924d70a4df0f87f9a3dae3dbd2e5701cc32f`**, pinned in `tests/test_cohort_v3.py`.
- Every cohort v2 entry is unchanged and in order.
- The G13 `retention` role grows from 1,670 to **39,672 records**: the 1,670 PR #16 records plus 38,002 records of the
  PR #44 150k set.
- Measured usable items are **36,005** (RWKV7-G1k) and **35,992** (Qwen3.5 / Q9). These items meet the pre-registered
  unit spec K ≤ 16, T ≤ 2,048, with the real template and tokenizer.
- This clears the ≥ 32,000 target, and so the ~30.7k items PR #102's `margin_sim` puts on retention ECE, plus ~9.3k
  for accuracy.
- Margins are unchanged.
- Billed $0. Everything ran on CPU locally.

## Commands

```sh
python -X utf8 -m train.cohort_v3 build --train-dir <ssdm-data>/m2/build/sample-150000-seed20261006   # 165-300 s
python -X utf8 receipts/2026-10-07-D-cohort-v3/measure.py train/cohorts/m2-dev-cohort-v3.jsonl.gz \
    receipts/2026-10-07-D-cohort-v3/measure.json --workers 20                                     # ~15 min, CPU
COHORT_V3_REBUILD=1 python -X utf8 -m pytest -q tests/test_cohort_v3.py   # 13 passed (rebuild byte-identical) -> pytest-rebuild.txt
```

Environment: `uv run --with pytest --with torch --with numpy --with pyarrow --with transformers==5.17.0` on the CPU
index, Python 3.12.13, Windows.

## Construction

The v2 conventions are followed:
- whole groups, assigned by stratified quotas;
- largest-remainder quotas;
- group order sha256(seed|version|role|key);
- seed 20261006;
- canonical JSONL, gzip mtime 0.

Groups are the connected components of "same normalized content" OR "same source item" (`record_item_keys`). This
covers:
- injection suffixes;
- multi-record parts;
- both variants of a fact-change pair.

All 368 fact-change records in the new set are whole pairs.

Strata are source × word-length bin (5 bins). Sources nest in areas.

| Ineligible (of 150,000) | Records | Why |
|---|---:|---|
| `draw_reserve` | 60,734 | groups with any member among the first 60,000 of the seed-0 draw order (`train.run_cell.split_ids`) |
| `pr16_sample` | 17,910 | the group shares a source item or content with ANY PR #16 record (all m2 pilot draws come from PR #16) |
| `injected` | 12,294 | D1-A verbatim and training-only hijack injections (`~inj:` ids) |
| `heldout_excluded` | 3 | the pinned D1-B fragment records (PR #74 / teacher scope) |
| guard hits (SplitGuard, frozen families, D1-A/D1-B scan) | 0 | `guard_excluded == {}` |
| cohort v2 (id / content / item) | 0 | the 150k build already excluded v2 |

Eligible: 72,204 records in 71,856 groups. Selected: 38,002, against a target of 38,000; whole groups overshoot by 2.

## Disjointness evidence (all in the cohort header, and re-checked by tests)

- **m2 training draws, used and planned.** These are the first cell (24,000, seed 0), the q9_feature_cache rows of
  that draw, and the Clef teacher export.
  - Every new id has a `split_ids` key above the reserve cutoff `669844ed…19bb`.
  - The reserve prefix contains the pinned draw, `2bbbf180…556b`, which the build recomputes.
  - With the 150k files present, the rebuild test checks that `training_exclusion(set)` equals exactly the 38,002 v3
    pool ids, so no twin or item is left in the set.
  - The same test checks that the seed-0 draws of 24,000 and of 60,000 are byte-identical with and without the v3
    exclusion.
- **PR #16 draws.** These are the m2 pilot's 3,000 draw and the rwkv7 pilot's 2,000 + 300. No new record shares a
  source item or content with any of the 20,000 PR #16 records. Before this rule was added, a trial build overlapped
  87 records of the m2 pilot draw and 38 of the legacy draw (negative evidence, scratch).
- **Index DEV/TEST.** This guard is `train.guard.SplitGuard` over the harness DEV + TEST manifests:
  - DEV manifest `26cd6bc0…`, TEST manifest `8de5aaed…`;
  - the pinned content index `7cbe6c89…`, n = 24,150, via `data.build.content_guard`.
  - It ran over every one of the 82,710 candidates and produced 0 hits.
- **D1-A / D1-B.** `train.heldout.scan` ran over all candidates with the D1-B set (`7a81287e…`) and the D1-A set:
  0 hits. Injected records are never eligible. The steer roles are unchanged.
- **Frozen held-out families.** `harness.heldout.check_record` (heldout-draw-v1 `c3172003…`, 9 families) gave 0
  `FrozenFamilyError`.
- **Within the cohort.** No id, content hash or source item is shared between the new records and v2's
  dev / calibration / retention.

## Composition of the retention role (`measure.json` `composition_retention_role`)

| | v2 (PR #16) | new (m2-150k) | all |
|---|---:|---:|---:|
| records | 1,670 | 38,002 | 39,672 |
| usable items, RWKV7 / Q9 | 1,491 / 1,490 | 34,514 / 34,502 | **36,005 / 35,992** |
| question types choice / noul / score | 1,936 / 155 / 157 | 44,982 / 2,433 / 3,502 | 46,918 / 2,588 / 3,659 |
| length <64 / 64-255 / 256-1023 / 1024-4095 / ≥4096 words | 303 / 580 / 597 / 114 / 76 | 5,960 / 14,425 / 14,285 / 1,976 / 1,356 | 6,263 / 15,005 / 14,882 / 2,090 / 1,432 |

Items by area (multi parts counted per part):

| Area | Item parts | Share |
|---|---:|---:|
| knowledge | 11,493 | 26.2 % |
| language | 11,401 | 26.0 % |
| retrieval | 10,525 | 24.0 % |
| tools | 7,632 | 17.4 % |
| arts | 2,834 | 6.5 % |

Arts has no v2 records: the PR #16 sample had no arts source. New records by source are in the cohort header
(`composition_new`). mmlu_aux is the largest source at 8,922; 17 sources in total.

## Per-arm evaluation cost of the retention term (`measure.json` `cost`)

The volume is exact: the decide prefix-fork plan over every record with at least one usable item. The rates are
measured.

| Backbone | Positions | Calls | Minutes / arm | $ / arm |
|---|---:|---:|---:|---:|
| RWKV7-G1k-13.3B, job-tkfaj (3,757 tok/s, 0.117 s/call) | 21,182,992 | 64,054 | 218.9 | 10.83 |
| Q9 / Qwen3.5-9B, job-5cmat (22,144 tok/s, 0.0595 s/call) | 20,514,242 | 63,930 | 78.8 | 3.90 |

- **Conservative bound** (pack-call latency on every call): RWKV7 734.5 min / $36.36; Q9 267.9 min / $13.26.
- **`VOLUME_TOKENS` upper-bound convention:** RWKV7 129.6 M tokens (575 min / $28.46); Q9 132.2 M tokens (99.5 min / $4.93).
  - This method reproduces v2's 5,724,231 exactly.
  - It over-counts, because records with no usable item are never forwarded.

## Limits

- The jobs still pin v2. `score_parent.VOLUME_TOKENS` and the job caps are v2-sized. Switching a job to v3 requires
  three things:
  - `pool_dirs={"m2-150k": dir}` in `Cohort.load_role_records`;
  - re-caching the parent arm;
  - re-pricing the caps from the table above.

  This is left for after PR #102 and PR #74, which own those files.
- PR #74's `train.train_set.draw` refuses any set that contains a cohort-held-out record. Once a cell pins v3, it must
  subtract `train.cohort_v3.training_exclusion(...)` from its pool and call `assert_draw_disjoint`. Both are already
  done in `baselines.clef_teacher`. The 24k draw is unchanged by this.
- The power inputs are still the `margin_sim` assumptions: synthetic ECE arms and discordance 0.10. The first real G13
  run re-sizes from measured values.

## Power re-check on top of PR #102 (`margin_sim.json`)

`python -X utf8 -m train.margin_sim` was run with `EVAL_SIZES` retention = 35,992 measured items, the smaller of the
two tokenizers. Both retention criteria now meet the 0.8 floor, with a simulated null pass rate of 1.0 at that size.
`SHORTFALLS` is empty.

The pinned `REQUIRED_N` is 9,300 for accuracy and 33,300 for ECE; it is the gate's exit-45 floor. 35,992 clears both,
so the floor is met with ~8 % headroom over the ECE pin. The 1/sqrt(n) re-estimate from the simulated half-width at
35,992 is ~7.5k (accuracy) and ~7.9k (ECE). That suggests the 33,300 ECE pin is conservative. The pin is left unchanged.

Tests: `pytest.txt` holds the task command plus test_margin_sim / test_promotion_margins / test_m2_gates.
`pytest-rebuild.txt` holds the opt-in byte-identical rebuild.
