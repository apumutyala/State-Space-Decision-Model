# Receipt: m2-data sample build and contamination guard (Track D), revision 2

**Result.** 20,000 SystemOne training records were built end to end on CPU from 12 pinned public train splits plus synthetic fact-change pairs.
- 23,399 records were checked against every row of the official Decision Index 0.2.1 suite (155,390 rows: all 44 catalogs, DEV and TEST, HLE included).
- 3,399 (14.5%) were dropped for item-tier overlap.
- The independent re-check reproduced the record-file hash and every logged count.
- Billed $0. Nothing ran on a GPU.

**Revision 2** follows the PR #16 reviews:
- The guard gains two short-item matchers (`edit`, `contain`; `data/shortedit.py`). One-word-substitution recall on short DEV items rises from 15–33% to 95–97% (BANKING77/CLINC).
- Fact-change pairs now share one option permutation.
- Tool question ids are collision-free.
- The build refuses a non-production fingerprint.
- Pair logging is fixed, and the WinoGrande/NQ licences are corrected.

Revision-1 numbers are superseded. They stay in git history at `e05869f`.

## What the guard claims (scope)

**Claim.** Detected-overlap control for each record's declared **item-tier** text (utterance, premise/hypothesis, question, MCQ options, gold passage) under five matchers. A record is dropped if any matcher fires against any Index row:

| Matcher | Rule | Reaches |
|---|---|---|
| exact | normalised segment equality (NFKC, casefold, Unicode non-word runs collapsed; ≥12 chars; lines and "Label:"-stripped lines too) | exact and reformatted duplicates |
| ngram13 | any shared word 13-gram | edits and truncations of items ≥ ~13 words |
| minhash | byte-5-gram Jaccard ≥ 0.75 (64 perms; effective threshold ~0.8) | **~1-character edits** of short items, not one-word edits |
| **edit** (new) | token Levenshtein d ≤ 2 **and** d/max(len) ≤ 0.2, against all 536,344 non-boilerplate Index segments of 4–30 tokens; exact search (deletion variants), verified | one-word edits for 5–9 tokens, ≤2 from 10 tokens |
| **contain** (new) | an Index segment of 6–30 tokens occurs as a contiguous token run inside a training segment ≤ 80 tokens | item plus an appended sentence |

**Not claimed.**
- That no Index text reaches the training payload. Context-tier overlap (shared label names, tool schemas, Wikipedia passages used as retrieval negatives or padding) carries no Index label (INDEX_NOTES §7) and is kept and logged: 4,942 records.
- Coverage of the following, none of which any matcher reaches:
  - truncations of short items;
  - edits beyond the bound (e.g. 2 words in a 5–9-token item);
  - items under 5 tokens beyond exact match;
  - paraphrase;
  - translation (all sources are English).

## Evasion test (DEV only; `evasion_test.py` → `evasion.json`)

The review's protocol, re-implemented:
- 40 seeded DEV rows per catalog, 13 catalogs.
- Per row, the shortest and longest indexed single-line item texts.
- Each text mutated and run through the guard.
- TEST is never opened.

Recall is shown as v2 matchers → v3 matchers, for the **short** probe:

| Catalog (median tokens) | 1-word sub | 2-word sub | appended sentence | first-half truncation |
|---|---|---|---|---|
| BANKING77 (10) | 0.33 → **0.95** | 0.05 → 0.50 | 0.20 → **0.93** | 0.03 → 0.05 |
| CLINC150 (8) | 0.23 → **0.97** | 0.03 → 0.38 | 0.20 → **0.88** | 0.03 → 0.07 |
| When2Call question (7) | 0.28 → **0.82** | 0.07 → 0.15 | 0.05 → **0.78** | 0.10 → 0.10 |
| ANLI hypothesis (9) | 0.28 → **0.97** | 0.07 → 0.40 | 0.17 → **0.85** | 0.05 → 0.05 |
| MMLU-Pro short (7) | 0.35 → 0.75 | 0.30 → 0.40 | 0.35 → 0.65 | 0.30 → 0.30 |
| WinoGrande (18) | 0.90 → **1.00** | 0.33 → **1.00** | 1.00 → 1.00 | 0.03 → 0.03 |
| HellaSwag short (22) | 0.80 → 0.97 | 0.53 → 0.82 | 0.80 → 0.93 | 0.33 → 0.33 |
| MuSR short (8) | 0.50 → 0.95 | 0.12 → 0.38 | 0.33 → 0.90 | 0.05 → 0.05 |
| BBH short (9) | 0.53 → 0.85 | 0.38 → 0.45 | 0.47 → 0.75 | 0.38 → 0.40 |
| RAGTruth short (9) | 0.47 → 0.97 | 0.30 → 0.42 | 0.38 → 0.93 | 0.30 → 0.30 |
| GPQA short (9) | 0.53 → 0.82 | 0.20 → 0.45 | 0.30 → 0.75 | 0.28 → 0.28 |
| ContractNLI short (4) | 0.15 → 0.20 | 0.03 → 0.05 | 0.05 → 0.05 | 0.15 → 0.15 |

- Exact and reformatted duplicates are caught 100% in every catalog.
- Long probes (GSM8K, MuSR, RAGTruth, ContractNLI, ANLI premises; ≥30 tokens) are ≥0.95 on substitutions and appends under both versions. The remaining long-probe gaps are truncation for When2Call, MMLU-Pro, BBH, HellaSwag and GPQA (0.53–0.85).
- The remaining short-probe gaps follow from the specified bound, which is the stricter of ≤2 edits and ≤0.2 normalised:
  - two-word edits of 5–9-token items;
  - <5-token spans (ContractNLI);
  - truncations.

## Identity

- Branch `claude/m2-data`, stacked on `claude/m0-harness` @ `b2edd86`. The code is this PR's head.
- Local Windows CPU, Python 3.12.10 run as `python -X utf8 -I`, with numpy 2.5.3 and pyarrow 25.0.1. Seed `20261006`.
- **Suite snapshot** `ssdm-data/m2/suite-snapshot/`, hash-verified as the official 0.2.1 row files. `suite-0.2` was re-staged by another process during the work, so the guard builds from a frozen copy.
  - `selected-rows.jsonl`: `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`
  - `added-rows.jsonl`: `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89`
- **Fingerprint** `ssdm-data/m2/index-fp-v3.npz` (`ssdm-dedup-v3`), sha256 `592dd7bc9a921147e22f383fa9b805356aa801f3223b443e8100a154c4c1818f`. Contents:
  - 536,344 short items;
  - plus the v2 arrays: segments, 13-grams and MinHash, with >100-row boilerplate removed.
  - `build_sample` now fails closed unless the fingerprint's version, suite hashes, row count and HLE coverage match the production identity. Synthetic fingerprints are allowed only in unit tests, never with `--receipt`.
  - **Array identity (rev 2.1, Track L finding).** The guard also hashes the arrays it actually queries: name, dtype, shape and bytes of all 9 arrays, independent of npz zip timestamps. It refuses unless that hash equals both the claimed `meta.arrays_sha256` and the pinned `PRODUCTION_ARRAYS_SHA256` = `b2961e355e27308d730141f5bd7e99c1a90d29c7db01c44abe3e0fcf9538ee28`. An independent fingerprint rebuild from the snapshot reproduces this hash. Regression tests cover forged metadata, a forged self-hash, and one flipped array element on the real fingerprint.
- **Sources:** `data/manifests/sources.lock.json`, sha256 `5c2dcbee…bd0d`.
- **Records** (outside git): `ssdm-data/m2/build/sample-20000-seed20261006/records.jsonl.gz`, uncompressed sha256 `76f568df57d25bf5626ef26390ed82f0bb078da26d836fc5146c25699aaefb23`.
- **Manifest** (committed): `data/manifests/sample-20000-seed20261006.manifest.jsonl.gz`, sha256 `a355ad0f94da889d15ef0f8541194e0c136211dd37e9dd9ec2a6795fc2e5231c`.

## Commands

```sh
python -X utf8 -I -m ssdm_data fetch                     # 12 sources, sha256 locked
python -X utf8 -I -m ssdm_data fingerprint               # 169 s
python -X utf8 -I -m ssdm_data build --n 20000 --receipt receipts/2026-10-06-D-m2-data     # 208 s -> build.log
python -X utf8 -I receipts/2026-10-06-D-m2-data/recheck.py receipts/2026-10-06-D-m2-data   # -> recheck.json
python -X utf8 -I receipts/2026-10-06-D-m2-data/evasion_test.py <ssdm-data>/slices/dev-rows.jsonl.gz > evasion.json
python -X utf8 -I -m pytest -v data/tests                # -> pytest.txt
```

## Dedup log (`dedup-log.json`, the required receipt)

| Source | Checked | Dropped (v3) | % | Of which edit / contain fired | Rev-1 dropped / checked | Main Index catalogs (item tier) | Kept with context overlap |
|---|---:|---:|---:|---|---|---|---:|
| ANLI | 2,389 | 589 | 24.7 | 36 / 27 | 583 / 2,383 | ANLI 12 (shared premises) | 17 |
| BANKING77 | 1,616 | **216** | 13.4 | 174 / 42 | 53 / 1,453 | BANKING77 4 | 9 |
| BoolQ | 1,208 | 8 | 0.7 | 2 / 1 | 5 / 1,205 | HoVer 61, BRIGHT 36 | 4 |
| CLINC150 | 1,525 | **125** | 8.2 | 116 / 8 | 23 / 1,423 | CLINC 5, SGD 10 | 1,400 |
| GSM8K | 1,660 | 260 | 15.7 | 46 / 47 | 258 / 1,658 | RouterBench 6, BRIGHT 36 | 6 |
| HellaSwag | 1,665 | 65 | 3.9 | 38 / 19 | 29 / 1,629 | HellaSwag 29 | 110 |
| MMLU aux_train | 1,915 | 115 | 6.0 | 33 / 7 | 104 / 1,904 | MMLU 24, ARC 26/27 | 11 |
| multi (merged) | 1,497 | 297 | 19.8 | 122 / 35 | 196 / 1,396 | ANLI 12, RouterBench 6, BANKING77 4 | 328 |
| NQ (retrieval) | 1,839 | 39 | 2.1 | 3 / 6 | 31 / 1,831 | HoVer 61 | 964 |
| SST-5 | 1,001 | 1 | 0.1 | 0 / 1 | 0 / 1,000 | HoVer 61 | 4 |
| fact-change | 1,200 | 0 | 0.0 | 0 / 0 | 0 / 1,200 | — | 0 |
| ToolACE | 1,653 | 53 | 3.2 | 18 / 26 | 37 / 1,637 | ToolRet 2 | 1,247 |
| When2Call | 2,818 | **1,618** | 57.4 | 369 / 152 | 1,483 / 2,683 | When2Call 62 | 836 |
| WinoGrande | 1,413 | 13 | 0.9 | 0 / 0 | 13 / 1,413 | SATA 33, MMLU 24 | 6 |
| **Total** | **23,399** | **3,399** | **14.5** | 957 / 371 | 2,815 / 22,815 | | **4,942** |

**Over-dropping (measured, by hand).** I hand-checked 24 BANKING77/CLINC records dropped by `edit`/`contain` only.
- About half are true near-duplicates: "where will i get my card" ↔ "when will i get my card"; "what is the date tomorrow" ↔ "what is the date of tomorrow".
- About half are templated siblings that differ in a content word, sometimes the label-bearing one: "can i get a refund" ↔ "can i get a mastercard"; "what s the timezone for brasilia" ↔ "britain".
- One `contain` hit joined a BANKING77 training question to an unrelated CLINC test item through a shared run of 6+ tokens.
- The bound is deliberately conservative. The cost is ~10% more of BANKING77/CLINC dropped; no item is let through.

## Sample composition (`build-summary.json`)

- **Question types:** 25,239 choice / 1,788 noul / 1,599 score.
- **Options per choice question:**
  - 2: 9,553;
  - 3–4: 9,127;
  - 5–10: 1,771;
  - 11–64: 730;
  - 65–151: 3,777;
  - 152–255: 281.
- **Id styles:** digits 3,162 · letters 3,074 · opaque 3,122 · option_k 3,130 · semantic 2,143.
- **Placement:** state 12,454 / instructions 5,146.
- **Shapes:** single 16,831; multi-question 1,200; per-candidate yes/no 284; per-tool yes/no 485.
- **Length (words):** <64: 3,497 · 64–255: 7,142 · 256–1,023: 7,230 · 1,024–4,095: 1,190 · ≥4,096: 941. 1,443 records are padded.
- **Injection (labels unchanged): 1,558 records.**
  - D1-A verbatim: 1,066.
  - Training-only hijack: 492.
  - The held-out 6-gram guard passed on all 23,399 checked records. It covers the **k=8** D1-B set (rev 2.1: plus the 4 blind v0.4 templates of PR #22, 47 hashes recomputed from `a44a35e` and equal to PR #22's list; 96 forbidden 6-grams in total), and the record file is unchanged (`76f568df…`).
- **Fact-change:** 600 pairs over 6 families. Each pair now carries identical questions (same permutation and ids) and differs only in the untrusted fact and the label. This is asserted in the build and tested through the build path.

## Independent re-check (`recheck.json`)

- A rebuild with the same seed gives the same record sha256.
- The written file re-reads with:
  - 20,000 records;
  - 0 validation failures;
  - 0 held-out leaks;
  - 0 manifest mismatches.
- Whole-payload overlap recomputed from disk is 4,942, the same as the log.

## Defects found and fixed

1. **Rev 1: normalisation dropped non-Latin text.** Fixed with Unicode `[\W_]+` (fingerprint v2).
2. **Rev 1: duplicate HellaSwag ids** (`ind` is not unique). The id is now the row index, and the build asserts unique ids.
3. **Rev 1: every CLINC record was dropped** on short label names. This led to the item/context tiers.
4. **Rev 1: the suite was rewritten mid-build.** The guard now builds from a hash-verified snapshot.
5. **Rev 2 (review): short-item one-word edits evaded the guard.** Added `edit` and `contain`, which were exhaustively cross-checked against brute force on random sequences.
6. **Rev 2 (Codex review): fact-change variants were confounded.** The two variants got different option permutations from a shared advancing RNG. Each pair now uses one fresh RNG.
7. **Rev 2 (Codex review): `tool_{name}`[:60] question ids could collide.** Ids are now positional, with a cardinality assertion.
8. **Rev 2 (Codex review): the builder accepted any fingerprint.** It now fails closed.
9. **Rev 2: a half-dropped fact-change pair logged its clean partner as "kept".** Pairs are now decided first and partners are logged as `dropped_pair_partner`; there were 0 in this sample.
10. **Rev 2: licences.**
    - WinoGrande: Apache-2.0 per its GitHub LICENSE; the HF card states none.
    - NQ: the HF card states none; upstream is CC-BY-SA-3.0.
    - When2Call `train_pref` is documented as not fetched.

## Limits

- **Coverage gaps:** see "Not claimed" above.
- **When2Call labels:** the SFT labels (ask vs decline) come from the source responses, and the distractors are rule-drawn. That split has no tool-call targets.
- **Licences to flag:**
  - ANLI is CC-BY-NC-4.0;
  - SST-5 and the NQ card state no licence;
  - flag these before any commercial use.
- No teacher labels yet (TEACHERS.md is a plan).
