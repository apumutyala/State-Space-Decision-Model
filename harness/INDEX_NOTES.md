# Decision Index 0.2.1: what the harness must respect

Sources (all pinned; read in full for this note):
- Kit: `github.com/apolinario/decision-index` @ `87d4650b42b377c0291a89c1f1a879f9b31082bf` ("Version 0.2.1", 2026-09-27; MIT). README, `docs/format.md`, `docs/suite.md`, `decision_index/editions.py`, `scoring/{index,index02,report,added,metrics}.py`, `suite/{io,sample,download}.py`, `suite/build/{acquire,freeze,release_v2,rebuild}.py`, `tests/test_index021.py`.
- Board data: HF Space `multimodalart/jev-decision-index` @ `cdbd1cab3b6eb1811ebd82d2686ad563a4d0fc0d`: `data/methodology-v0.2.1.json` (sha256 `3724d15e…20ce8`), `data/index-v0.2.1.json` (sha256 `37526dd3…20ce8`, generated 2026-09-28T00:39Z).
- The kit has no `methodology.json`; the methodology lives in the Space.

Quotes below are verbatim from those files unless marked *inferred*.

## 1. Suite and row format

- Two row files: `selected-rows.jsonl.gz` (124,971 lines; uncompressed sha256 `b2b56d6f…92d5`) and `added-rows.jsonl.gz` (30,419 lines, the seven benchmarks added in 0.2; `7429f3c9…f89a`). Plus `excluded-questions.json` (442 run ids, `331df32d…5e81`) applied at scoring time, and read-time subsets (ACOS 400 reviews; 0.2.1: ToolRet 685 and BRIGHT 220 answerable queries, Home appliances 88 rows). Totals: 120,340 requests, **119,898 scoreable** + 30,419 added.
- The suite is **not redistributed** ("several sources don't allow republishing"); `suite rebuild` reproduces it byte-identically from pinned sources (~7 GB download, 17 GB working). The Hub copy `multimodalart/decision-index-suite-0.2` is private.
- One JSON object per request:

| Key | Meaning |
|---|---|
| `id`, `family`, `split` | source case id, benchmark family, upstream split |
| `state` | context: string or JSON object; may be empty |
| `questions` | `{key: {"type": "choice", "instructions", "criteria": {option_key: description}}}` with 2–255 options, or `{"type": "noul"}` (yes/no probability; RAGTruth and PhishNChips signal questions) |
| `expected` | gold per question key (`null` for unjudged retrieval candidates and PhishNChips signal questions) |
| `metadata` / `provenance` | group ids, subject/song/user ids for macro metrics; never sent to an engine |
| `scoring` | custom scorer payloads: `retrieval_ranking` (qrels, `field_to_document`), `routing_outcome`, `chess_reference_value` / `human_group_rank` (accepted ties), `forecast_probability`; added benchmarks carry `unscored_fields` |
| `_evaluation` | `run_id`, `catalog_id`, `dataset`, `group_id` (linked case), `track`, `payload_sha256` (hash of `{"model": "jev-1.13.0", "state", "questions"}`), `proxy_tokens` (= len(JSON payload)//4), `benchmark_origin` |

- "An engine receives exactly `state` and `questions`." Response per question: `{"type": "choice", "choice": key, "probabilities": {every option: p}}` (sum within 0.01 of 1) or `{"type": "noul", "noul": p_yes}`; `validate` rejects anything else.

## 2. The 38 counted benchmarks, areas and weights

| Area (weight) | Benchmarks (★ gold = 1.2, others 1.0, normalised inside the area) |
|---|---|
| Knowledge & Reasoning (25.85%) | GPQA Diamond★ (25), GSM8K (30), ChessBench (31), MuSR (32), SATA-Bench (33), CRUXEval (43), CLadder (44), HLE★ (45), MMLU-Pro★ (57), BBH★ (58) |
| Language Understanding (25.85%) | ContractNLI (11), ANLI★ (12), WinoGrande★ (28), HellaSwag★ (29), ACOS (38), FinEntity (39), iSarcasmEval (40, headline track A-English), VAST (41), NLI4CT (42), RAGTruth (59) |
| Retrieval & Classification (20.02%) | BANKING77★ (4), CLINC150+OOS★ (5), BRIGHT★ (36), Amazon ESCI (37), PhishNChips (56), HoVer (61) |
| Tools & Automation (18.28%) | BFCL★ (1), ToolRet (2), API-Bank★ (3), Home appliances (9), When2Call (62) |
| Arts & Human Taste (10%, fixed) | BPoMP (20), Humicroedit (21), POP909 (22), cfcolor (23), ForecastBench★ (48), Habermas (50), New Yorker (64) |

"Arts & Human Taste is fixed at 10%; the other four areas share the rest in proportion to the square root of their benchmark count." Shown but **not counted**: MMLU, ARC-Easy, ARC-Challenge, SimpleBench, RouterBench ("the prompt reveals that GPT-4 is the best route"), SGD ("build bug, pending re-run"). The six interactive environments stay out for everyone. Ties: scores within 0.25 points share a rank.

## 3. Scoring and chance correction (`scoring/index02.py`)

1. **Coverage first.** "Unsupported, errored, abstained and pending requests score zero before chance correction." For the 17 benchmarks scored at track level (`track_scored`: 1, 2, 11, 20–23, 25, 30, 31, 36, 37, 40, 41, 43, 44, 50) a linked case whose requests did not all succeed scores zero inside the metric. For the others `raw = native score on complete cases × answered / requests` (the native score is rounded to 4 decimals first).
2. **Chance correction.** `skill = clip((raw − chance) / (1 − chance), 0, 1)`. Chance: mean of 1/options; per track (GSM8K 4- and 10-choice averaged); per query expected random nDCG@10 (ToolRet 0.1341, BRIGHT 0.116 on answerable queries); F1 of random labels for F1 benchmarks (Monte Carlo, fixed label universe, `chance-baselines.json`); whole-case chance for BFCL/SATA/Home appliances; RAGTruth = F1 of always answering "hallucinated" (0.5177); ACOS per-review F1 of answering yes to every pair (0.031). **ForecastBench** instead: `clip((0.25 − Brier) / 0.25) × coverage`.
3. Metrics per benchmark: accuracy; macro-F1 (BANKING77, CLINC, ANLI, FinEntity, NLI4CT); *conservative* macro-F1 with missing predictions in the denominator (ContractNLI, ESCI, VAST, iSarcasmEval A); nDCG@10 (ToolRet, BRIGHT); case-exact (BFCL, SATA, Home appliances); per-review F1 (ACOS); song/user/variant macro accuracy (POP909, cfcolor, BPoMP); accepted-tie accuracy (ChessBench, Habermas); F1 on the hallucinated class (RAGTruth); `verdict` field only (PhishNChips).
4. **Index** = 100 × Σ_area w_area × (gold-weighted mean of skills). `raw_index` is the same on raw. The board says "Point estimates only, no uncertainty intervals yet", although methodology declares "paired hierarchical cluster bootstrap through complete index formula", 2000 replicates, 95%. `harness.bootstrap` implements that declared method.
5. Parity: the kit reproduces every 0.2.1 entrant; `harness/tests/test_aggregate.py` checks our weights against 13 live entrants (Jev 57.89 … Lumma-Fev-0.1B 1.78) to ±0.01.

## 4. Calibration (ECE) protocol

Verbatim (`index-v0.2.1.json` `suite.calibration_note`): "32 benchmarks with a per-field right or wrong answer (ToolRet, BRIGHT, RouterBench and ForecastBench excluded), each weighted equally; a 1-in-6 sample of requests; confidence is the probability placed on the chosen option. Rows an entrant trained on count as wrong here too." Board entries report `acc, conf, ece, brier, over95, n` (72,594 fields for every entrant), 10 equal-width bins (`rel`), and per-area ECE.

- **Not published:** the rule that picks the 1-in-6 sample, and the list of the 32 benchmarks. A natural guess (the 37 base benchmarks minus the four exclusions and SimpleBench) has 216,932 scored fields in the 0.2.1 scope (with HLE), so one sixth is ~36k fields, half the board's n = 72,594. Even every scored field of the whole 0.2.1 scope minus the four exclusions gives only 42k per sixth. **So the board's ECE sample is drawn from a wider scope than the scored 0.2.1 rows.** The most likely candidate is the full 0.1 ACOS (5,479 reviews; ACOS alone has 91k fields in its 400-review subset). This is unresolved; ask the maintainers before claiming a board-comparable ECE. `harness.calibration.board_ece` uses a hash-of-run-id 1-in-6 stand-in, benchmark-equal weights and 10 bins. Its numbers match the board's definition but not its sample.
- Board ECE is the plain binned L1 estimator, which is biased upward on small or sharp samples; the harness additionally reports debiased L2 ECE (Kumar et al. 2019) and SmoothECE (Błasiok & Nakkiran 2023) per request-length bin.

## 5. Latency protocol

"latency-v1": 750 timed rows + 10 untimed warm-up, systematic sample (seed 20260926) over the live scope sorted by benchmark then payload length; "one process, one request at a time, no concurrency, on exactly one RTX PRO 6000 (96 GB); no multi-GPU entrants"; "fastest inference path the model or its code supports (CUDA graphs, fused/fla kernels, flash attention, compiled paths, vLLM/SGLang, prefix/KV reuse, batching questions inside a row)"; timing = "end-to-end wall time of one row (request in, all answers out)", CUDA-synchronised, load excluded; median and p95 reported. **Board admission:** "Models whose median latency is over 1,000 ms per request on one RTX PRO 6000 … are not added to the board." Maintainers measure it themselves "on a private held-out sample", and use the same sample "to validate submitted runs by comparing answers". Latency in your own `scores.json` "is a guide, not that measurement."

## 6. Abstention, refusals and capacity limits

- Hard rules: no truncation (a request that does not fit is `unsupported` = wrong); no option pruning/label filtering (every option gets a probability); no prompt tuning; no labels in the adapter; no correctness-based retry (transient errors retry ≤2); "Native abstentions stand" but **score zero**: "Unsupported, refused and errored requests score zero against the full denominator. Nobody climbs by skipping the hard cases."
- Real cost examples (`answer_gaps`): Verdict lost 59% of rows (46.6% self-abstention, 12.2% option-count cap of 24); Solomon 22.5% from an 8-option cap; Tev1 12.3% from a 24-label cap (CLINC has 151 options, BANKING77 77, POP909 129, API-Bank 53). Linked cases (ToolRet/BRIGHT chunks, ACOS reviews, BFCL) count only when every linked request succeeded.

## 7. What counts as training contamination

- "Treat the suite as evaluation data …; do not train on it and do not republish it." Training on **test items** is scored as wrong for those items (benchmark still computed over the full set); a confirmed leak gets the entry removed (reflex-4B: LoRA trained on 800 MMLU-Pro test items, 13.7-point held-in gap).
- **Train/dev splits of the public sources are allowed**, but the maintainers rebuild declared training sets and count *exact duplicates of test items that sit in public train splits* as wrong (lev: 45 items; Jet v6.2: 5 BANKING77/CLINC duplicates; Lumma-Fev: 3,077 BANKING77 items found in the author's own datasets though undeclared). Home appliances dropped 48 test rows identical to dev rows.
- Overlap that does not carry the label (e.g. GSM8K questions inside RouterBench prompts) is not counted.
- Detection: rebuilding declared training data, held-in vs held-out gaps, and a shuffle test (options reordered on MMLU, GPQA, ARC-C, HellaSwag, MMLU-Pro; flag = drop ≥5 points with CI excluding zero) plus a 1,695-row perturbation control set.

## 8. Submission process

1. Run the full suite (`pipeline` or `hf-job`); "Runs must be complete (`scores.json` says `"complete": true`) and untouched: the results file is re-scored on review." Complete = every scoreable row of the edition (119,898 + 30,419).
2. Upload the run directory to a Hub dataset (`runs/<name>/scores.json` must exist).
3. Open a PR adding one line to `submissions/README.md`: model name, results dataset link, engine/commit, hardware; declare capacity limits. Board runs use the author's own inference code (or the `http` engine for a `/v1/systemone` server).
4. Maintainers re-score, measure latency on their private sample, compare answers, and may run contamination checks. "A model appears once, at the configuration that stands."
5. Before the full run, a compatibility pass: "Two requests per benchmark/subtrack form the 86-request compatibility pass."

## 9. Implications for PLAN.md (Track D)

1. **T1's ECE criterion needs rewording.** The board ranks plain 10-bin L1 ECE on an undisclosed sample from a wider scope (§4), not debiased ECE on ≥4k rows. Suggested wording: "board-protocol ECE below the current #3 (0.0178), plus debiased ECE and SmoothECE on DEV, reported side by side". Until the maintainers disclose the sample, any board-comparable ECE claim is provisional.
2. **Unanswered = wrong, abstention included.** An `ASK`/`ABSTAIN` option, an option-count cap, or a context cap all score zero. The D core must emit a full distribution for 2–255 options (CLINC 151, POP909 129, BANKING77 77, API-Bank 53) and accept the longest rows: 1.9% of DEV rows are ≥8,192 proxy tokens, mostly ToolRet/BRIGHT/BFCL. This is consistent with the interface's "do not add ASK to an external benchmark without reporting its scoring effect", which should be tightened to "never on the Index".
3. **ForecastBench is a gold benchmark scored by Brier.** Calibrated probabilities move the score directly, not only ECE. Its events resolved 2026-07-01 to 2026-09-18, so any backbone or teacher trained on data after July 2026 can leak outcomes. Record the knowledge cutoff of every parent and teacher.
4. **Contamination gate.** Train-split data is allowed, but maintainers count exact duplicates of test items found in public train splits as wrong (lev, Jet, Lumma). Every training/distillation set needs an auditable dedup log against the suite's `state`/question text, not only against benchmark names. They also run a shuffle test (options reordered).
5. **Latency is an admission gate (<1,000 ms median, one RTX PRO 6000, one request at a time), measured by the maintainers on a private sample.** The m3 serving path must be the fastest path the code supports.
6. **HLE is gated** (see §10). The harness runs without it. m3 and any full-suite parity claim need an HF token with the HLE terms accepted.
7. Per-benchmark noise matters for m2 decisions. On DEV (11,125 clusters) a paired comparison between two systems ~5 points apart has SE ≈ 0.90 index points, so MDE ≈ 2.5 points (α 0.05, power 0.8). Doubling the slice brings MDE to ≈1.8 and quadrupling to ≈1.3. Gaps under ~2.5 points between m2 arms cannot be resolved on DEV alone.
8. The board itself publishes "point estimates only", and ties within 0.25 points share a rank. T1/T2 margins (56.88, 57.91, ≈57.07) are within ~1 point of each other, below what one full run can separate without the paired bootstrap.

## 10. Local build status (2026-10-06)

- Built with `suite rebuild` in WSL (`bi-ubuntu`, Python 3.12.3, C.UTF-8) from the pinned kit, everything except HLE (gated; no HF login on this machine). Script and log: `receipts/2026-10-06-D-m0-harness/`.
- `added-rows.jsonl`: **byte-identical** to the official file (sha256 `7429f3c9…f89a`).
- Base rows cannot match the official hash without HLE, but every count matches the published numbers with HLE's 513 rows removed: 0.1 rows 131,909 = 132,422 − 513; cases 117,251 = 117,764 − 513; fields 774,689 = 775,202 − 513; 0.2 cut 124,458 = 124,971 − 513 (ToolRet 1,328, BRIGHT 637 as documented). In the 0.2.1 scope, 119,397 scoreable base rows + HLE's published 501 = **119,898**, exactly the official count. Exclusions and all five read-time subset files match their pinned hashes.
- Our base file: uncompressed sha256 `6787a1d5…41c7`.
- **Two kit portability bugs found on Windows** (no effect on Linux/the board):
  - A Windows clone with `core.autocrlf=true` changes the packaged subset JSON hashes. Fix: clone with `core.autocrlf=false`.
  - The kit decodes its subset lists with the locale encoding. Under cp1252, two BRIGHT run ids containing U+2019 are garbled and silently dropped (218 instead of 220 BRIGHT rows). The harness refuses to import unless Python runs in UTF-8 mode (`python -X utf8 -I`). Worth an upstream issue/PR.
