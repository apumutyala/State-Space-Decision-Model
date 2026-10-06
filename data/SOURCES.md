# m2 training sources (Track D, phase m2-data)

Written 2026-10-06. This file lists where m2 training records may legitimately come from. No Decision Index row is ever a training input, DEV and TEST alike. Every built record also goes through the contamination guard (`dedup.py`, receipt `receipts/2026-10-06-D-m2-data/dedup-log.json`) before it can be used.

Rules (from `harness/INDEX_NOTES.md` §7 and `COORDINATION.md`):

- **Allowed:** public train and dev splits of the upstream sources. **Not allowed:** test items and Index rows.
- Maintainers rebuild declared training sets and count exact duplicates of test items as wrong (lev, Jet, Lumma-Fev). So the guard drops near-duplicates as well as exact ones, and checks every 0.2.1 row: all 44 catalogs, display-only ones and HLE included.
- Declare every training set at submission. This file plus `manifests/sources.lock.json` is that declaration.
- Teachers are open-weight only (`TEACHERS.md`).

Legend:
- **Fetched** means pinned in `sources.py`, sha256 in `manifests/sources.lock.json`, and used in the sample. Its size is the exact count our loader read.
- Every other size and license comes from the source card. Treat them as **to verify at fetch**.
- "None" means no train split exists that is disjoint from what the Index samples.

## 1. The 38 Index benchmarks

| # | Benchmark (area) | Index uses | Legitimate train source | Size | License | Notes / traps |
|---|---|---|---|---|---|---|
| 25 | GPQA Diamond (K) | all 198 Diamond | **None** | — | CC-BY-4.0 | GPQA main and extended are **supersets of Diamond**, so they are not train data. |
| 30 | GSM8K (K) | test 1,319 (4- and 10-choice) | `openai/gsm8k` main/train, **fetched** | 7,473 | MIT | RouterBench prompts embed GSM8K problems. The guard dropped 15.7% of GSM8K train items drawn (mostly RouterBench, some BRIGHT). That is conservative: §7 says label-free overlap is not counted. |
| 31 | ChessBench (K) | DeepMind action-value test | `searchless_chess` train action-value data | very large (GB scale) | Apache-2.0 | Dedup by FEN before use. Positions repeat across splits. |
| 32 | MuSR (K) | all 756 | **None** | — | CC-BY-4.0 | The generator code is public. New instances need a teacher (open-weight only). |
| 33 | SATA-Bench (K) | all 1,650 | **None** known | — | see repo | Synthetic multi-answer MCQ from our own sources covers the format. |
| 43 | CRUXEval (K) | 570 of 800 | **None** | — | MIT | We can generate unlimited *new* Python functions and execute them to get exact labels at $0 (to do). |
| 44 | CLadder (K) | 5,000 of 10,112 | **None** (the full release is the pool) | — | MIT | The repo's generator can make new graphs/queries. Dedup against all 10,112. |
| 45 | HLE (K) | 513 text-only MC | **None** | — | MIT (gated) | Covered by the guard: the snapshot includes HLE. |
| 57 | MMLU-Pro (K) | test 12,032 | `validation` (70, few-shot) only. Proxy: `cais/mmlu` auxiliary_train, **fetched** | 99,842 | MIT | MMLU-Pro reuses MMLU *test* questions, so MMLU test is never train. aux_train comes from ARC/OBQA/RACE/MCTest train splits. |
| 58 | BBH (K) | 5,507 | **None** | — | MIT | **Trap:** BIG-Bench "train" splits (e.g. `tasksource/bigbench`) of the 23 BBH tasks contain BBH test items. Excluded. |
| 11 | ContractNLI (L) | test 123 contracts | official train (423 docs) and dev (61) | ~7k doc×hypothesis pairs | CC-BY-4.0 | The 17 hypotheses are shared across splits (label set). Contracts must not overlap; the guard checks. |
| 12 | ANLI (L) | test r1–r3 | `facebook/anli` train r1–r3, **fetched** | 162,865 | **CC-BY-NC-4.0** | Premises recur across splits. The guard dropped 24.7% of ANLI train items drawn (premise shared with Index rows). Non-commercial licence; flag before any product use. |
| 28 | WinoGrande (L) | xl validation 1,267 | `allenai/winogrande` winogrande_xl train, **fetched** | 40,398 | Apache-2.0 per the `allenai/winogrande` GitHub LICENSE; the HF card states none | Twin sentences: the `edit` matcher (≤1 token edit for 5–9 tokens, ≤2 from 10) catches one-word twins; MinHash alone did not. |
| 29 | HellaSwag (L) | validation 10,042 | `Rowan/hellaswag` train, **fetched** | 39,905 | MIT | Same ActivityNet/WikiHow sources as validation. Endings are item-tier for the guard. |
| 38 | ACOS (L) | test reviews (400-review subset scored) | Laptop-ACOS and Restaurant-ACOS train/dev | ~4.4k train sentences | see repo (verify) | Dedup against the full 1,399 test reviews, not only the subset. The guard does this. |
| 39 | FinEntity (L) | all 979 docs | **None** | — | ODC-BY (verify) | |
| 40 | iSarcasmEval (L) | 5 test tasks | official train (En ~3.5k, Ar ~3.1k) | ~6.6k | research (verify) | Multilingual: Unicode-aware normalisation in the guard (fix found while testing: v1 dropped non-Latin text). |
| 41 | VAST (L) | test 3,006 | official train | ~13.5k | research (verify) | |
| 42 | NLI4CT (L) | 2024 test 5,500 | 2023/2024 train and dev | ~1.9k | research (verify) | Trial evidence is shared across splits. The statement is the item. |
| 59 | RAGTruth (L) | 2,700 rows | official train split | ~15k responses | MIT | Check that response IDs don't overlap the Index rows' source responses. |
| 4 | BANKING77 (R) | test 3,080 | PolyAI train.csv, **fetched** | 10,003 | CC-BY-4.0 | Jet v6.2 / Lumma-Fev were penalised for exact test duplicates. The guard drops exact, one-word-edit (`edit`) and appended-text (`contain`) duplicates: 13.4% of the BANKING77 draw, some of them templated siblings (receipt). |
| 5 | CLINC150+OOS (R) | test + oos_test 5,500 | `data_full.json` train + oos_train only, **fetched** | 15,100 | CC-BY-3.0 | The file contains all splits. The loader reads only `train`/`oos_train` (tested). |
| 36 | BRIGHT (R) | 220 answerable queries | **None** (test-only benchmark) | — | CC-BY-4.0 | Its document corpus overlaps StackExchange/Wikipedia text. Logged as context overlap. |
| 37 | Amazon ESCI (R) | 5,000 test pairs | `shopping_queries_dataset` split=train | ~1.4M pairs | Apache-2.0 | Queries recur across splits with different products. Dedup on (query, product). |
| 56 | PhishNChips (R) | 2,000 rows | **None** known | — | see card | Proxy: public phishing corpora (licences vary). Not used yet. |
| 61 | HoVer (R) | 4,000 rows | official train | ~18k claims | CC-BY-SA-4.0 | Built from HotpotQA/Wikipedia. Wikipedia passages (e.g. NQ) hit HoVer evidence as context overlap (logged). |
| 1 | BFCL (T) | 1,694 cases | **None** | — | Apache-2.0 | Proxies: ToolACE (**fetched**), Glaive FC v2 (Apache-2.0, **fetched** for the 150k build), xLAM-60k (CC-BY-4.0, gated). |
| 2 | ToolRet (T) | 685 answerable queries | ToolRet training release (verify ID/licence) | — | verify | ToolRet's tool corpus overlaps RapidAPI tool docs in When2Call/ToolACE. Logged as context overlap (tool schemas carry no ToolRet label). |
| 3 | API-Bank (T) | 508 decision points | API-Bank training dialogues (DAMO-ConvAI) | ~6k | MIT | Same 53-tool catalog: catalog text is context, dialogues are items. |
| 9 | Home appliances (T) | 88 test rows | published dev file (80 cases) | 80 | see kit | The Index already dropped 48 test rows identical to dev. Dev is allowed. |
| 62 | When2Call (T) | test MCQ 3,652 | `nvidia/When2Call` train_sft, **fetched**; train_pref (9,000) exists but is **not fetched or used** | 15,000 | CC-BY-4.0 | **57% of train items drawn shared item text with the test MCQ** (mostly 13-grams of responses and questions) and were dropped. The SFT split has no tool-call targets (refuse/ask only). |
| 20 | BPoMP (A) | 5,000 rows over 811 cases | **None** known | — | see Zenodo | |
| 21 | Humicroedit (A) | subtask-2 test 2,628 | SemEval-2020 T7 **train** splits via `tasksource/humicroedit` @ `f5a16e6` (subtask-1 train, subtask-2 train), **fetched** for the 150k build | 9,652 (s1) / 6,178 usable of 9,381 (s2; ties and malformed rows skipped) | HF card: unknown; SemEval research release (verify) | Headlines recur across subtasks. The clean original headline is item-tier (as well as the edited text), so a train record sharing a headline with a test row is dropped, not only exact edit matches. |
| 22 | POP909-CL (A) | 2,000 beats over 907 songs | **None** disjoint (909 songs total) | — | MIT (verify) | Symbolic chord data from other corpora is the only option. |
| 23 | cfcolor (A) | 5,000 held-out pairs | original train-rating history | — | see release | The Index state already carries user history. Training on other users' ratings is allowed. Dedup on palettes. |
| 48 | ForecastBench (A) | 10,139 forecasts resolving 2026-07-01..09-18 | question sets resolved **before 2026-07-01** | — | CC-BY-SA-4.0 (verify) | **Leakage:** any text or teacher with knowledge after the forecast date leaks outcomes (see TEACHERS.md cutoffs). |
| 50 | Habermas Machine (A) | evaluation cohorts | training cohorts of the released data | — | CC-BY-4.0 (verify) | Questions recur across cohorts. The guard drops shared question text. |
| 64 | New Yorker (A) | matching test 528 | `jmhessel/newyorker_caption_contest` matching train | ~2.3k | CC-BY-4.0 | Contests (cartoons) are shared across task configs. Dedup on contest ID as well as text. |

Display-only rows (MMLU, ARC-E/C, SimpleBench, RouterBench, SGD) are in the guard too. **RouterBench is never a train source**, because it embeds other benchmarks' items.

## 2. General decision-format sources (no Index eval rows)

| Source | Format → record shape | Size | License | Status |
|---|---|---|---|---|
| SST-5 (`SetFit/sst5` train) | ordinal sentiment → `score` (5 levels, scale sometimes reversed) | 8,544 | unspecified on HF; SST research release | **fetched** |
| BoolQ (`google/boolq` train) | passage + yes/no question → `noul` | 9,427 | CC-BY-SA-3.0 | **fetched** |
| Natural Questions pairs (`sentence-transformers/natural-questions`) | query + 1 gold + up to 254 negatives → 8–255-option `choice`, or per-candidate yes/no | 100,231 | HF card states none; upstream Natural Questions is CC-BY-SA-3.0 | **fetched** |
| ToolACE (`Team-ACE/ToolACE`) | tool catalog + dialogue → single-tool `choice` (+ none) or per-tool yes/no | 11,300 dialogues | Apache-2.0 | **fetched** |
| MNLI / SNLI | NLI `choice` | 393k / 550k | mixed (OANC, CC-BY-SA) / CC-BY-SA-4.0 | not fetched |
| CommonsenseQA, OpenBookQA, PIQA, SciQ | MCQ | 9.7k / 5k / 16k / 11.7k | MIT / Apache-2.0 / AFL-3.0 / CC-BY-NC-3.0 | not fetched |
| AG News, DBpedia-14, TweetEval, GoEmotions | classification (4–28 labels) | 120k / 560k / ~50k / 43k | research / CC-BY-SA / mixed / Apache-2.0 | not fetched |
| MS MARCO passage | retrieval up to 255 candidates | 500k queries | **non-commercial** | not fetched (licence) |
| Glaive FC v2 (`glaiveai/glaive-function-calling-v2`, Hub parquet conversion `refs/convert/parquet` @ `8862406` of main `e7f4b64`; the 271 MB JSON exceeds the fetch cap) | catalog + first user turn → which function to call now, or `no_tool` (answer / clarify / decline) | 112,960 rows; 78,011 usable (rows with no published function skipped) | Apache-2.0 | **fetched** (150k build) |
| xLAM-60k | tool selection | 60k | CC-BY-4.0 (gated) | not fetched |
| HotpotQA, FEVER | evidence → verdict | 90k / 145k | CC-BY-SA-4.0 / CC-BY-SA-3.0 | not fetched. **Overlaps HoVer (built from HotpotQA)**; the guard must run. |
| SGD train (`dstc8-schema-guided-dialogue` train) | dialogue intent `choice` | ~16k dialogues | CC-BY-SA-4.0 | not fetched. SGD is display-only (test in guard). |

## 3. Synthetic sources (this repo)

- **Fact-change pairs** (`injection.py`; families: delivery, spend_limit, schedule, stock, renewal, rating). The two records differ only in one untrusted fact and have different labels (tested). The wording is our own; no safety_eval scenario is reused.
- **Injection augmentation.** D1 family A (verbatim, provenance pinned) plus six training-only hijack templates. Family B and the five eval hijack templates are excluded by a 6-gram hash guard.
- **To do:** CRUXEval-style executed functions and CLadder/MuSR-style generators, with exact labels and no teacher.

## 4. Known gaps

- Licences marked *verify* have to be checked at fetch time, before a source enters a paid run.
- Two fetched sources are not permissive:
  - ANLI is CC-BY-NC-4.0;
  - SST-5's licence is unspecified on HF;
  - the NQ pairs card states no licence (upstream NQ is CC-BY-SA-3.0);
  - WinoGrande's HF card states none (its GitHub LICENSE is Apache-2.0).
  - They are fine for research, but need a `decision-needed` call before any commercial release.
- Benchmarks with no train split (GPQA, MuSR, SATA, CRUXEval, CLadder, HLE, BBH, BRIGHT, FinEntity, BFCL, BPoMP, POP909, PhishNChips) depend on proxy formats, synthetic generators or teachers. That is where m2 attribution should expect gaps.
