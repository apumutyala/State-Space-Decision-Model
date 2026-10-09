# 6-gram train/test contamination audit (2026-10-09)

## Takeaway

- **No Index DEV or TEST item was copied into any training draw.** This holds for every draw checked: the 150k pool v1/v2, D1-24k, the Clef-flash teacher export, B1-45k, B1-100k v1/v2/v3, b1-families (and -x), bschema-300k-v1, and the bschema-v2 candidate pool. No eval question, claim, statement, utterance, review or answer text appears verbatim in training. No eval row has both its state and its question covered.
- **One finding that needs a decision: NLI4CT-2024 premise text.** The `nli4ct` training records are in **B1-100k-v3 (855 records), bschema-300k-v1 (2,620) and the bschema-v2 candidates (2,284)**. They contain the clinical-trial-report (CTR) sections that also serve as premises in Index NLI4CT rows:
  - 40 DEV and 72 TEST rows have at least 90% of their premise 6-grams present in training.
  - 84 DEV and 72 TEST rows reach at least 50%.
  - The statements being judged are **not** copied: 0 rows reach 90% coverage of non-template statement grams. 5 DEV and 9 TEST statements reach at least 50%, because they are related statements about the same trial.
  - The cause is the dataset's design: NLI4CT reuses each CTR across its train and test splits. Under a strict reading of "we can't train on test" (eval *context* counts), this is real overlap in those three draws. The fix is simple (see "Recommendation").
- Every other non-boilerplate hit falls into one of three groups. None is an item copy:
  - shared public text, mostly Wikipedia paragraphs inside NQ passages and injected filler that also appear as HoVer evidence, ANLI premises or an Amazon-ESCI product description;
  - generic tool-schema phrasing;
  - per-dataset instruction templates.
- This matches the prior result: D1 ∩ subset = 0 by id and content, and the 186 "stricter-check" hits were template text. The 6-gram scan finds no item overlap in D1 or the teacher export.

## Method

- **Eval side** (read-only, hashed in memory, no eval text written anywhere): 24,789 rows.
  - DEV: the full 12,104-row DEV slice (`ssdm-data/slices/dev-rows.jsonl.gz`, which contains the 2,008 subset) plus 262 rebuilt GSM8K DEV twins.
  - TEST: 12,161 + 262 rows, selected from the local suite files (`suite-0.2/selected-rows`, `added-rows`, `suite-0.3-gsm8k`) by the `run_id`s in `harness/manifests/test-manifest*.jsonl.gz`. All 12,423 were found.
  - The TEST unseal gate (`harness.cli.load_slice`) was not invoked. This report gives TEST rows by id and number only.
- **Normalisation:** lowercase; every non-alphanumeric character becomes a space; collapse whitespace; split on whitespace. 6-grams never cross a field leaf. Each 6-gram is a 64-bit hash (blake2b per token, polynomial combine, splitmix finaliser).
- **Field types:**
  - S = state (every string leaf);
  - Q = question instructions;
  - O = option/criteria texts.
  - Each eval gram carries its field-type bits. Each hit records train field → eval field.
  - There are 1,963,595 distinct eval 6-grams.
- **Boilerplate exclusion:** 6-grams that occur in more than 1% of eval rows (1,364 such grams) **and** in more than 1% of the draw's training rows. This is computed per draw; the union is listed in the appendix.
- **Per training row:**
  - the non-boilerplate hits;
  - the longest verbatim matched span;
  - attribution to the eval row with the highest covered fraction, using a postings index that skips grams found in more than 500 eval rows.
- **Checks on top of the 6-gram scan:**
  - (a) token-level re-verification of every "strong" pair (eval-row coverage of at least 0.5), leaf by leaf;
  - (b) exact match of every normalised eval state or question leaf (5 tokens or more, non-template) against every training leaf;
  - (c) per-family check: union the 6-grams of each same-named training source (for example `banking77`, `when2call`, `ragtruth`, `acos`, `nli4ct`) across all draws, then measure each DEV/TEST row's state coverage and non-template question coverage.
- Script: `analysis/ngram_test_audit.py`. It contains `build-eval` and `scan`; checks (a) to (c) are small scratch drivers over the same hashing functions. CPU only, streamed, one draw per process.

## Per-draw results

Column definitions:
- **Raw-hit rows:** training rows with any eval 6-gram.
- **Non-boiler rows:** training rows with at least one non-boilerplate 6-gram.
- **≥25-tok span:** training rows containing a verbatim run of 25 or more tokens.
- **Cov ≥0.5:** training rows that cover at least half of some eval row's 6-grams.

| Draw | Rows | Raw-hit rows | Boiler grams excl. | Non-boiler rows (any overlap) | ≥25-tok span | Cov ≥0.5 | Real item overlap | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 150k pool v1 (`sample-150000-seed20261006`) | 150,000 | 68,225 | 14 | 52,943 | 864 | 265 | 0 | CLEAN |
| 150k pool v2 (`…-v2`) | 150,000 | 67,724 | 14 | 52,406 | 866 | 266 | 0 | CLEAN |
| D1 24k (`sample-24000-seed0-intent16-v1`, drawn ids sha `2bbbf180…`) | 24,000 | 10,706 | 13 | 8,736 | 141 | 47 | 0 | CLEAN |
| Clef-flash teacher export (24,000/24,000 ids resolved) | 24,000 | 10,940 | 13 | 8,736 | 141 | 47 | 0 | CLEAN |
| B1 45k (`b1-d4-draw45k`) | 45,000 | 28,375 | 92 | 18,026 | 10 | 5 | 0 | CLEAN |
| B1 100k v1 | 100,000 | 54,732 | 63 | 42,874 | 630 | 13 | 0 | CLEAN |
| B1 100k v2 | 100,000 | 55,419 | 67 | 40,867 | 626 | 12 | 0 | CLEAN |
| B1 100k v3 | 100,000 | 59,198 | 86 | 42,981 | 3,316 | 17 | 0 items; **NLI4CT premises** | **PREMISE OVERLAP** |
| b1-families | 7,084 | 7,084 | 83 | 3,734 | 0 | 0 | 0 | CLEAN |
| b1-families-x | 24,284 | 24,284 | 83 | 8,995 | 0 | 0 | 0 | CLEAN |
| bschema-300k-v1 | 300,000 | 169,370 | 122 | 129,066 | 7,114 | 68 | 0 items; **NLI4CT premises** | **PREMISE OVERLAP** |
| bschema-v2 (not built; scanned the 359,306-row candidate pool, a superset) | 359,306 | 233,029 | 138 | 135,933 | 20,204 | 63 | 0 items; **NLI4CT premises** | **PREMISE OVERLAP** |

The large non-boiler counts come mostly from common English 6-grams such as "he is best known for his". The span and coverage columns isolate the cases that matter. Every row with coverage ≥0.5 was re-verified:

| Eval family hit (cov ≥0.5) | What is shared (verified leaf by leaf) | Class |
|---|---|---|
| HoVer (DEV and TEST) | `state.evidence[].text` = Wikipedia paragraphs that also sit in NQ passages or injected filler. The claim is never covered. | shared public text |
| Amazon-ESCI TEST `…:1636442` | `state.product.description` = the Wikipedia lead of *Beauty and the Beast (1991)*, repeated in about 135 training rows | shared public text |
| ANLI (r1/r2) | Wikipedia lead sentence in the premise. The hypothesis is not covered. | shared public text |
| MMLU-Pro TEST `57:7437` | a 6-token economics phrase found in an NQ passage; 6 of 16 tokens of 3 options | shared public text (common phrase) |
| NLI4CT-2024 | CTR sections (`primary_trial`/`secondary_trial` text). The statement is not copied. | **shared source document** |

### Top-50 hits per draw (ranked by eval-row coverage, then verbatim span)

| Draw | Shared public text (Wikipedia) | Tool-schema / assistant phrasing | NLI4CT CTR premise | Other |
|---|---:|---:|---:|---|
| pool v1, pool v2, D1, teacher | 50 | 0 | 0 | 0 |
| B1-45k | 11 | 37 | 0 | 2 short Banking77 near-paraphrases ("entered the wrong pin too many times"; best eval coverage 0.23) |
| B1-100k v1 / v2 | 25 / 22 | 25 / 28 | 0 | 0 |
| B1-100k v3 | 13 | 24 | 13 | 0 |
| b1-families / -x | 0 | 50 / 49 | 0 | 1 CRUXEval numeric-list option (`3 0 4 0 5 0`) |
| bschema-300k-v1 | 31 | 0 | 19 | 0 |
| bschema-v2 candidates | 33 | 0 | 17 | 0 |

There are no real item overlaps among any draw's top 50. The most frequent shared n-grams are Wikipedia biography phrases ("is an american actress she is", "of the national football league nfl") and generic tool-argument descriptions ("the unique identifier of the order").

### Same-dataset checks (union of all draws; per DEV/TEST row)

| Training source → Index family | Training rows | DEV state ≥0.9 | TEST state ≥0.9 | DEV/TEST question ≥0.9 (non-template) | Note |
|---|---:|---:|---:|---:|---|
| nli4ct → NLI4CT-2024 | 5,759 | **40** (84 ≥0.5) | **72** (72 ≥0.5) | 0 / 0 | CTR premises shared by dataset design |
| acos → ACOS | 15,984 | 0 | 0 (7 ≥0.5) | 262 / 267 | per-category instruction template only. Exact review match: 0 |
| ragtruth → RAGTruth | 26,022 | 0 | 0 | 0 / 0 | spans of 25+ tokens are RAGTruth prompt templates |
| habermas → Habermas | 5,189 | 0 | 0 | 0 / 0 | shared policy-question wording |
| when2call → When2Call | 46,491 | 0 | 0 | 0 / 0 | generic tool-parameter descriptions |
| b1fam_bfcl_synth / glaive_fc / toolace → BFCL | 54k / 261k / 50k | 0 | 0 | 4 / 6 (synth) | shared tool names in "Candidate tool: X". The conversation is not covered. |
| b1fam_apibank → API-Bank | 10,465 | 0 | 0 | 0 / 0 | API-Bank instruction template + catalog phrasing |
| banking77, clinc150, anli, hellaswag, winogrande, vast, gsm8k, hover, cruxeval-synth | 25k–119k each | 0 | 0 | 0 / 0 | |

Exact-leaf check (b), across all draws, by matched eval leaf type:
- ACOS per-category instruction text;
- BFCL and When2Call tool parameter descriptions (for example "Maximum number of search results to return.");
- NLI4CT CTR sections;
- one API-Bank DEV user turn: "The quick brown fox jumps over the lazy dog".

No eval question, claim, statement or utterance matched.

## NLI4CT detail and recommendation

- **Affected rows (premise coverage ≥0.9):**
  - DEV (40, id prefix after `NLI4CT-2024:test:`): 06bf2ef1 06e6eb2b 0d8cdde2 0f90b547 116c9dea 2c519a1e 2df3c784 32cc173a 344ac04d 3657b15f 36cbd8dd 3d35c61e 4246dc90 47c91759 487046b0 51c9a1a4 5acd7311 5c9037a6 6c12493c 80b58d76 84e7c52d 856ab26e 85f5396f 892e63ba 92fe4c4d a52b9758 a60074c0 a62aed40 ab3fc92f b6084ccd be6130d9 cafc343b ccce8199 cf935264 d303ce9d ea0e2918 ece744da ef0cf72d f9e4988c fb41d62d
  - TEST (72): 9c74310c 907ae179 6fbc7c06 679f8ab4 5d9fff0a 333c097e ad1c4967 0d49e948 fb1605b0 cd8d5a6a 79515622 be3df611 a723722c 77890255 5255cd06 c6ca9f93 0dc19d61 40fceb75 34a95632 b93384b1 7caa5dd0 7bad3ae8 f4d3c3ad a3ba6a16 f02e41c3 1a80bab2 f995d440 0c03c804 fdc63af9 58726c08 c48550ef a8932849 ae5ec938 4b5b9e75 5ba46a49 151bb6d0 d595c94d 43a81620 b1cb1388 ba87189b d20c94ae 251e979f d14fe83a 3c164518 1b314250 eef90c1d 429199a7 a434b916 b4fd327d 070ba2da ab848536 840eeabe a141432b f102f343 e25b66fb 4499b316 dd7bc677 711fdd40 da24c082 3693ad36 baf9254a 55af1301 b5976fd8 282cb769 91e178f6 a68e6309 4f5fb0bf 693c9e39 27180597 5f45e043 a7e702b9 13d5fd15
- Some premise overlap appears in 204 DEV and 276 TEST NLI4CT rows; most of it is short shared CTR boilerplate. The overlap is identical across B1-100k-v3, bschema-300k-v1 and the v2 candidates, because they draw the same `nli4ct` train records.
- **Recommendation:** before any of those three draws is used, drop the `nli4ct` records whose CTR sections cover at least 50% of an Index DEV/TEST NLI4CT premise, or drop `nli4ct` entirely (0.9–2.9% of each draw). Then re-run `scan` on the rebuilt draw. Draws without `nli4ct` (pool v1/v2, D1, teacher, B1-45k, B1-100k v1/v2, b1-families) need no change.

## Limitations

- Scope:
  - The eval set is DEV ∪ TEST (24,789 rows).
  - The other ~96k public-suite rows that are in neither slice were not hashed, because RAM on this machine is tight. Re-running `build-eval` with those rows added would cover them.
  - bschema-v2 is not built yet; its candidate pool was scanned instead.
- The two coverage thresholds (0.5 for a strong hit, 0.9 for full coverage) are heuristics.
- Exact-leaf matching needs at least 5 tokens. Short eval states (under 6 tokens) have no 6-grams and are covered only by checks (b) and (c).
- The 1% boilerplate rule excludes some label-set grams, for example the Banking77 class names "balance not updated after bank transfer". Option *labels* are shared by construction.

## Appendix: boilerplate 6-grams excluded (union over draws; 178)

Grouped:
- **Banking77/CLINC label set:** "balance not updated after bank transfer / cheque or cash deposit", "top up by bank transfer charge", "top up by cash or cheque", "wrong exchange rate for cash withdrawal", "out of scope none of the", "classify the (banking) intent of this user request".
- **BFCL instruction:** "given the complete user conversation and the published function schemas in state mark each candidate tool yes if it should be invoked to answer the request or no otherwise this evaluates tool name selection only do not produce arguments or call ordering candidate tool" (all 6-grams of this sentence).
- **CRUXEval instruction:** "choose the correct output of f called with the supplied input arguments candidates are python literal values".
- **API-Bank instruction:** "select the single api tool to invoke for the dialogue prefix in state use the published catalog in state do not infer arguments or use the current future api result", "api bank current api tool selection".
- **RAGTruth options:** "the response contains content that is not supported by the context in the prompt", "all content of the response is supported by the context".
- **HoVer:** "is the claim supported by the evidence", "the evidence does not support the claim".
- **ANLI:** "classify the relationship between the premise and hypothesis".
- **VAST:** "determine the stance of the post toward the topic".
- **ACOS:** "given the review below decide whether it expresses at least one aspect category sentiment pair … anywhere in the review answer yes only when that exact category and sentiment pair is present answer no otherwise multiple opposite sentiments are separate valid questions", plus category/sentiment variants.
- **Misc:** "which option correctly fills the blank", "choose the numeric answer to the problem", "apologies but i m unable to", date-format phrases ("in the format yyyy mm dd", "y m d h m s").

Per-draw boilerplate counts are in the table above.
