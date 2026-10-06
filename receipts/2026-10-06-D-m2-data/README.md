# Receipt: m2-data sample build and contamination guard (Track D)

**Result.** 20,000 SystemOne training records were built end to end on CPU from 12 pinned public train splits plus synthetic fact-change pairs.
- 22,815 records were checked against every row of the official Decision Index 0.2.1 suite (155,390 rows: all 44 catalogs, DEV and TEST, HLE included).
- 2,815 (12.3%) were dropped for item-level exact or near-duplicate overlap.
- The independent re-check reproduced the record-file hash and every logged count.
- Billed $0. Nothing ran on a GPU.

## Identity

- Branch `claude/m2-data`, stacked on `claude/m0-harness` @ `b2edd86`. The code commit is this PR's head.
- Hardware: local Windows CPU only. Python 3.12.10, run as `python -X utf8 -I`, with numpy 2.5.3 and pyarrow 25.0.1. Seed `20261006`.
- **Suite snapshot.** The official row files were copied to `ssdm-data/m2/suite-snapshot/` and their hashes checked before the build. `suite-0.2` was re-staged by another process while this work ran (HLE added), so the guard builds from a frozen copy.
  - `selected-rows.jsonl` uncompressed sha256: `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5` (official, HLE included)
  - `added-rows.jsonl`: `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89` (official)
- **Fingerprint** `ssdm-data/m2/index-fp-v2.npz`, sha256 `e44882d871162ef82babda0eee1d67a508b18fcd124ffffb9e02b88356272e19` (`ssdm-dedup-v2`):
  - 871,416 indexed segments;
  - 10,942,455 13-grams;
  - 811,688 MinHash signatures;
  - boilerplate removed: 2,072 segments and 22,289 13-grams that occur in more than 100 rows.
- **Sources:** `data/manifests/sources.lock.json`, sha256 `a053f776…a9d3f`, which pins a file sha256 per source.
- **Records** (outside git): `ssdm-data/m2/build/sample-20000-seed20261006/records.jsonl.gz`, uncompressed sha256 `b29ae109cb2af695b5d6e4bcf2f0842b67f9d36c7cb7700e7b46a6a90d1d89f7`.
- **Manifest** (committed): `data/manifests/sample-20000-seed20261006.manifest.jsonl.gz` (record id + per-line sha256), file sha256 `5ccf438f…5f45`.

## Commands

```sh
python -X utf8 -I -m ssdm_data fetch                     # 12 sources -> ssdm-data/m2/sources/, sha256 locked
python -X utf8 -I -m ssdm_data fingerprint               # full-suite fingerprint (208 s)
python -X utf8 -I -m ssdm_data build --n 20000 --receipt receipts/2026-10-06-D-m2-data   # 107 s -> build.log
python -X utf8 -I receipts/2026-10-06-D-m2-data/recheck.py receipts/2026-10-06-D-m2-data # -> recheck.json
python -X utf8 -I -m pytest -v data/tests                # -> pytest.txt (42 passed)
```

## Dedup log (the required receipt: `dedup-log.json`)

Rule: a record is dropped if any matcher fires on its **item-tier** text against any Index row. Item-tier text is the utterance, premise/hypothesis, question, MCQ options or gold passage, as declared by each loader. Matches found only in the rest of the payload (shared label sets, tool catalogs, retrieval negatives, padding, instructions, injections) carry no Index label (INDEX_NOTES §7). They are logged and kept.

| Source | Checked | Dropped | Drop % | Main Index catalogs hit (item tier) | Kept with context overlap |
|---|---:|---:|---:|---|---:|
| ANLI | 2,383 | 583 | 24.5 | ANLI 12 (shared premises), HoVer 61 | 16 |
| BANKING77 | 1,453 | 53 | 3.6 | BANKING77 4 (near-dups, MinHash) | 7 |
| BoolQ | 1,205 | 5 | 0.4 | BRIGHT 36, HoVer 61 | 4 |
| CLINC150 | 1,423 | 23 | 1.6 | CLINC 5, SGD 10 | 1,400 (label names ~ short Index strings) |
| GSM8K | 1,658 | 258 | 15.6 | RouterBench 6 (228), BRIGHT 36 (25) | 6 |
| HellaSwag | 1,629 | 29 | 1.8 | HellaSwag 29, ANLI 12 | 110 |
| MMLU aux_train | 1,904 | 104 | 5.5 | MMLU 24, ARC 26/27, RouterBench 6, MMLU-Pro 57 | 9 |
| multi (merged) | 1,396 | 196 | 14.0 | ANLI 12, RouterBench 6 | 369 |
| NQ (retrieval) | 1,831 | 31 | 1.7 | HoVer 61 | 888 (Wikipedia negatives vs HoVer/BRIGHT/ANLI) |
| SST-5 | 1,000 | 0 | 0.0 | — | 3 |
| fact-change (synthetic) | 1,200 | 0 | 0.0 | — | 0 |
| ToolACE | 1,637 | 37 | 2.3 | ToolRet 2 | 1,124 (tool schemas vs ToolRet/BFCL) |
| When2Call | 2,683 | 1,483 | 55.3 | **When2Call 62 (1,420 by 13-gram)** | 813 |
| WinoGrande | 1,413 | 13 | 0.9 | SATA 33, MMLU 24 | 6 |
| **Total** | **22,815** | **2,815** | **12.3** | | **4,755** |

Matcher counts over all records: item_exact 1,184; item_ngram13 2,514; item_minhash 1,659 (one record can fire several).

## Sample composition (`build-summary.json`)

- **Question types:** 25,281 choice, 1,791 noul, 1,583 score.
- **Option counts per choice question:** 2: 9,534 · 3–4: 9,116 · 5–10: 1,764 · 11–64: 728 · 65–151: 3,857 · 152–255: 282.
- **Option-id styles:** digits 3,159 · letters 3,069 · opaque 3,140 · option_k 3,133 · semantic 2,131. Semantic ids are never used where they would reveal the gold. Option order is permuted on every record.
- **Placement:** item in state 12,433, in instructions 5,167. State shape: dict 10,392 · string 2,041 · empty 5,167.
- **Shapes:**
  - single 16,832;
  - multi-question 1,200 (2–4 items each);
  - per-candidate yes/no retrieval 284;
  - per-tool yes/no 484.
- **Length (words):**
  - <64: 3,478;
  - 64–255: 7,144;
  - 256–1,023: 7,225;
  - 1,024–4,095: 1,211;
  - ≥4,096: 942.
  - 1,435 records are padded with unrelated passages.
- **Injection (labels unchanged): 1,569 records.**
  - D1-A verbatim (provenance `944fa02`, blob `1c9c1067`): 1,083.
  - Training-only hijack templates: 486.
  - The held-out guard (D1-B plus the eval hijack 6-gram hashes) passed on all 22,815 checked records.
- **Fact-change:** 600 pairs (1,200 records) over 6 families. Labels differ within every pair, and only the untrusted fact field differs (tested).

## Independent re-check (`recheck.json`)

- A rebuild with the same seed gives the same record sha256 (deterministic).
- The written file re-reads with:
  - 20,000 records;
  - 0 validation failures;
  - 0 held-out leaks;
  - 0 manifest mismatches.
- The whole-payload overlap re-computed from disk is 4,755, the same as the log.

## Defects found and fixed during the build (logged per the fold rules)

1. **Normalisation dropped non-Latin text.** v1 used `[^0-9a-z]`, so Arabic iSarcasmEval and Chinese RouterBench text vanished from the guard. Fixed with Unicode-aware `[\W_]+`; fingerprint rebuilt as v2.
2. **HellaSwag `ind` is not unique within train.** This gave 12 duplicate record ids, which the re-check caught. The id is now the row index, and the build asserts unique ids.
3. **The first guard version dropped every CLINC record** (0/1,400 kept), because short label names ("transactions", "what is your name") MinHash-matched short Index strings. That led to the item/context split above. CLINC label names are now context overlap.
4. **The suite file was rewritten mid-build by another process.** The fingerprint now builds from a hash-verified snapshot and fails if the files change during the build.

## Limits

- **Not dropped:** overlap that does not carry the label is kept, and it is logged and counted:
  - tool-schema text shared with ToolRet/BFCL;
  - Wikipedia passages shared with HoVer/BRIGHT;
  - label names.
  - A stricter policy is a one-line change (drop on any full-tier hit). Expected drop rates under it would be roughly ToolACE 70%, NQ 50%, When2Call 85%.
- **Not covered by the guard:** paraphrase-level overlap below MinHash J 0.75 or without a shared 13-gram. Only the shuffle test and the held-in/held-out gap at m3 can bound that.
- **Heuristic label:** When2Call SFT labels (ask vs decline) come from the source responses. The distractors are rule-drawn. There are no tool-call targets in that split.
- No teacher labels exist yet (TEACHERS.md is a plan).
