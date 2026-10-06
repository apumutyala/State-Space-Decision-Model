# Receipt: m2 training set scaled to 150,000 records (Track D, m2-data-150k)

**Result.** 150,000 SystemOne training records, built on CPU from 15 pinned public train splits plus synthetic fact-change pairs. The PR #16 pipeline ran with its guarantees unchanged, and three additive guards were layered on top.
- 169,671 records were checked against all 155,390 rows of the Decision Index 0.2.1 suite. 19,671 of them (11.6%) were dropped for item-tier overlap.
- 4,102 candidate records were skipped before the guard because they belong to the frozen dev cohort v2: 3,250 by id, 710 by source item, 142 by content.
- An independent re-check from disk found 0 failures. A second full build reproduced the same record sha256.
- The PR #16 20k build, rebuilt with this code, is byte-identical (`76f568df…`; `regress-20k.json`).
- Billed $0. Nothing ran on a GPU. The build is a single process at Idle priority and takes 914 s.

## Identity

| What | Value |
|---|---|
| Records (outside git) | `ssdm-data/m2/build/sample-150000-seed20261006/records.jsonl.gz` (305 MB gz), uncompressed sha256 **`82fb395d889c473d9fbc4877b51c792438e94b53cddd54e4cf825d4603baa166`** |
| Manifest (id, line sha256, source, family) | `…/sample-150000-seed20261006.manifest.jsonl.gz` (7.0 MB gz), uncompressed sha256 `0be33669fa077d2a2abe29ecd159aea1a27c7cc9daaf910b584755410da4d961` |
| Content map (id → normalized content hash, `train.cohort` format) | `…/sample-150000-seed20261006.content.jsonl.gz` (6.7 MB gz), uncompressed sha256 `d4ffd9073e0490e409a757508b2821f8108a3a9e36b2531ec3c97a94b2bc0898` |
| Seed | `20261006` |
| Fingerprint | `index-fp-v3.npz` (`ssdm-dedup-v3`), the same production identity as PR #16. The build fails closed on any other fingerprint. |
| Content-index guard | `train.guard.SplitGuard` over the harness DEV/TEST manifests (`26cd6bc0…`, `8de5aaed…`) plus `index-content-hashes-v1.txt.gz`, sha256 `7cbe6c89…2d8d` with n=24,150. Refused unless equal to `jobs/m2_pilot/guard_pin.json`. |
| Cohort excluded | `m2-dev-cohort-v2`, sha256 `be6efe2b…acf2`: 3,505 held-out ids, 3,410 content groups, 3,864 source items |
| Sources | `data/manifests/sources.lock.json`, with 3 new entries |
| Environment | Windows CPU, Python 3.12.10 (`python -X utf8 -I`), numpy 2.5.3, pyarrow 25.0.1 |

The manifest and content map are **not in git** (14 MB of hashes). They are deterministic: the rebuild reproduces them. `data/manifests/sample-150000-seed20261006.pointer.json` pins their hashes.

## Commands

```sh
python -X utf8 -I -m ssdm_data fetch glaive_fc humicroedit_s1 humicroedit_s2
python -X utf8 -I -m ssdm_data build --n 150000 --mix 150k --receipt receipts/2026-10-06-D-m2-data-150k   # 914 s -> build.log
python -X utf8 -I receipts/2026-10-06-D-m2-data-150k/recheck.py <ssdm-data>/m2/build/sample-150000-seed20261006 > recheck.json
python -X utf8 -I -m pytest data/tests tests/test_m2_blockers.py     # 88 passed, 12 skipped -> pytest.txt
```

The commands ran from the worktree through a launcher, because the shared venv's editable `ssdm_data` points at the m2-data worktree. `--mix 150k` needs the repo root to be importable, since it uses `train.cohort` and `train.guard`.

## Guards (what changed vs PR #16)

The following PR #16 guards are unchanged:
- the five item-tier matchers (exact, ngram13, minhash, edit, contain) over the full 0.2.1 suite;
- the D1-B held-out 6-gram guard (`assert_no_heldout`), which checked all 169,671 records;
- the injection families (D1-A verbatim plus the training-only hijack templates);
- fact-change pairs with a shared permutation;
- the fail-closed fingerprint.

New, additive, and switched on only by `--mix 150k` (with the defaults, the 20k build is byte-identical):
1. **Cohort-v2 exclusion (before the guard).** A record is skipped if any of the following holds. The third check is stricter than `Cohort.assert_training_*`:
   - its id is held out;
   - its normalized content equals a held-out record's;
   - **any source item it renders belongs to a held-out record**: the injection suffix is stripped, a multi record is split into its parts, and both variants of a fact-change pair are covered.
   - Why the third check matters: in the 20k set an item can sit inside a held-out *multi* record. At 150k the same item comes back as a *single* record, with a new id and new content. 710 records were caught this way.
2. **Content-index matcher.** `SplitGuard.check` runs as a sixth item-tier matcher, `item_content_index`. It fired 0 times.
3. **Fact-change twin dedup.** The generator's value space is small: 4,000 pairs gave only 3,771 distinct variant contents out of 8,000. A pair is skipped if either variant's content already occurred. 554 twin pairs were skipped; 1,500 distinct pairs were kept.

Other changes:
- Records are streamed to disk instead of held in memory.
- **Bug found and fixed during the build.** `train.cohort.normalize_content` casefolds dict keys. When two keys collide under casefold (two ToolACE tool schemas), the hash depends on key order. The first build hashed in-memory records, which put 2 content-map mismatches in the re-check. The build now hashes the canonical (sorted-key) form, the same form the cohort's content map reads back from disk. After the rebuild there are 0 mismatches. The quirk itself is in `train/cohort.py`, which is left unchanged; flagged for its owner.

## Per-source counts and drop rates (`dedup-log.json`, `build-summary.json`)

| Bucket | Area | Kept | Cohort-skipped (pre-guard) | Guard-checked | Dropped | % | edit / contain fired | Kept w/ context overlap |
|---|---|---:|---|---:|---:|---:|---|---:|
| mmlu_aux | K | 33,000 | 380 (id 316 / item 64) | 35,086 | 2,086 | 5.9 | 619 / 85 | 147 |
| gsm8k | K | 5,500 | 324 (id 246 / item 78) | 6,562 | 1,062 | 16.2 | 188 / 198 | 30 |
| anli | L | 10,000 | 357 (id 315 / item 42) | 13,597 | 3,597 | 26.5 | 188 / 145 | 62 |
| hellaswag | L | 9,000 | 345 (id 280 / item 65) | 9,320 | 320 | 3.4 | 183 / 100 | 555 |
| winogrande | L | 7,000 | 319 (id 246 / item 73) | 7,058 | 58 | 0.8 | 0 / 4 | 35 |
| sst5 | L | 4,000 | 255 (id 175 / item 80) | 4,002 | 2 | 0.0 | 0 / 2 | 23 |
| nq | R | 10,000 | 315 (id) | 10,188 | 188 | 1.8 | 13 / 39 | 5,397 |
| banking77 | R | 7,000 | 311 (id 246 / item 65) | 7,996 | 996 | 12.5 | 811 / 166 | 38 |
| clinc150 | R | 9,000 | 298 (id 246 / item 52) | 9,934 | 934 | 9.4 | 835 / 83 | 9,000 |
| boolq | R | 8,000 | 279 (id 211 / item 68) | 8,127 | 127 | 1.6 | 5 / 25 | 35 |
| **glaive_fc** (new) | T | 15,000 | 0 | 15,443 | 443 | 2.9 | 243 / 195 | 9,627 |
| toolace | T | 6,000 | 281 (id) | 6,207 | 207 | 3.3 | 93 / 104 | 4,777 |
| when2call | T | 5,500 | 210 (id) | 13,315 | **7,815** | **58.7** | 1,772 / 852 | 3,878 |
| fact-change (synthetic) | T | 3,000 | 428 (id 163 / item 123 / content 142) | 3,000 | 0 | 0.0 | 0 / 0 | 0 |
| **humicroedit_s1** (new) | A | 5,000 | 0 | 5,007 | 7 | 0.1 | 4 / 0 | 22 |
| **humicroedit_s2** (new) | A | 4,500 | 0 | 4,505 | 5 | 0.1 | 4 / 0 | 21 |
| multi (merged) | mixed | 8,500 | 0 | 10,324 | 1,824 | 17.7 | 638 / 165 | 2,873 |
| **Total** | | **150,000** | **4,102** | **169,671** | **19,671** | **11.6** | 5,596 / 2,163 | **36,520** |

Item matchers fired (a record can fire more than one): exact 7,698 · ngram13 13,501 · minhash 10,686 · edit 5,596 · contain 2,163 · content_index 0.

## Area balance vs the Index area weights

Items are counted by area, with each part of a multi record counted in its source's area.

| Area | Items | Share | Index weight | Sources |
|---|---:|---:|---:|---|
| Knowledge | 42,107 | 25.9% | 25.8 | MMLU aux_train, GSM8K |
| Language | 42,361 | 26.0% | 25.8 | ANLI, HellaSwag, WinoGrande, SST-5 |
| Retrieval | 39,230 | 24.1% | 20.0 | NQ, BANKING77, CLINC150, BoolQ |
| Tools | 29,500 | 18.1% | 18.3 | Glaive FC v2, ToolACE, When2Call, fact-change |
| Arts | 9,500 | 5.8% | 10.0 | Humicroedit s1/s2 |

Area assignments that are judgement calls:
- BoolQ counts as retrieval (passage-evidence verification, like HoVer).
- Fact-change pairs count as tools.

Knowledge and tools sit at their Index weights, and retrieval is above its weight, as asked. Arts is 4.2 points under its weight. The reasons:
- Humicroedit is the only fetchable text-only arts train source. 15.8k items are usable, and the build uses 9.5k of them.
- The New Yorker matching train split is 548 MB of parquet because it includes the images. That is over the fetch cap.
- The other arts benchmarks (BPoMP, POP909, cfcolor, ForecastBench, Habermas) have no fetched train data.

Language is slightly over its weight because multi records draw from 4 language sources out of 9.

## Composition

- **Question types:** 173,695 choice / 9,349 noul / 12,286 score.
- **Shapes:** single 135,201; per-tool yes/no 1,804; per-candidate yes/no 1,495; multi 8,500.
- **Injection (labels unchanged):** D1-A verbatim 8,049; training-only hijack 4,066.
- **Padded:** 11,063.
- **Length (words):** <64: 23,273 · 64–255: 57,628 · 256–1,023: 56,063 · 1,024–4,095: 7,579 · ≥4,096: 5,457.
- **Within-set content twins:** 1 group of 2 records. They are not held out; this is a source duplicate.

## Independent re-check (`recheck.json`)

The re-check reads the records back from disk and checks, for each of the 150,000:
- Its line sha256 equals the manifest entry.
- It passes `validate`.
- The D1-B 6-gram guard passes.
- `SplitGuard` (the content index) passes.
- Its recomputed normalized content equals the content map.
- It has no cohort id, content or item.

It also runs `Cohort.assert_training_ids` and `assert_training_content`, and both pass. **0 failures.** Ids are unique.

## New sources (pinned; `data/SOURCES.md` updated)

- **Glaive FC v2.** `glaiveai/glaive-function-calling-v2`, Apache-2.0.
  - The single 271 MB JSON is over `MAX_BYTES`, so the Hub's parquet conversion of the same main-branch commit is pinned instead: `refs/convert/parquet` @ `8862406`, sha256 `7e7e32f0…`, equal to the LFS oid.
  - Record: catalog plus first user turn → which function to call now, or `no_tool`. Labels: 40,518 `no_tool` / 37,493 call, out of 78,011 usable rows.
- **Humicroedit.** `tasksource/humicroedit` @ `f5a16e6`, train files only; licence unknown on HF (research release).
  - s1: headline edit → 4-level funniness score.
  - s2: two edits of the same headline → which is funnier. Ties are skipped.
  - Original headlines are item-tier, so a shared headline drops the record.

## Limits and flags

- **Licences:**
  - ANLI is CC-BY-NC.
  - SST-5, NQ-pairs and Humicroedit state no licence on HF.
  - Flag before any commercial use.
- **When2Call:** 58.7% of the items drawn overlap the test MCQ. The PR #16 rate was 57%.
- **Capacity:** BANKING77, GSM8K, When2Call and ToolACE are near their post-guard capacity. A larger build needs new sources for them, not bigger quotas.
- **Downstream wiring:** `train/` still reads the PR #16 manifest (`train.cohort.PR16_MANIFEST`, `train.data`). Pointing the training run at this set means passing the 150k manifest path to `load_m2_records`. The cohort stays the PR #16 v2 cohort, and this set is disjoint from it.
