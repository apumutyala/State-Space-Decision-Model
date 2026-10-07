# TRAIN-only order-consistency census

The pinned first-cell draw supports a low-K learning pilot: at the current T=2048 cap, 21,348 base items survive and 11,059 have eligible reversed views, including 10,458 moved-gold pairs. K=16 contributes no pair at T=2048; the retained pair domain is K<=10. This is data feasibility, not a learned result, model qualification or launch authorization.

## Scope and provenance

- Owned branch: `codex/order-consistency-census`; receipt base: `6ab62f5b8f43f9f916900fb1963d37a36657dda4`.
- Original measurement checkout: `b4be283af9086d929a3498152fcfe48543876a10`. Relevant training source files were byte-identical to `agent-main@3dc591e14b5e9a2db225bf3394be53305c06fe5a` and receipt base 6ab62f5 (git diff --quiet on the nine recorded source paths returned 0).
- The order objective is integrated through squash 4b59a735f002dd7d58d3f8947c76fe8278b7f708, which includes stacked PR #160. Non-ancestry of the original implementation commit does not mean absence of the squashed code.
- Measurement: Windows, AMD Ryzen AI 9 HX 375, Python 3.12.13, CPU only, CUDA hidden, one tokenizer process with one numerical thread. No model/checkpoint construction, load, forward, training, GPU or paid job. Billed cost $0.
- Tokenization/counting runtimes: original 189.984 s; saved-script reproduction 229.469 s, excluding draw verification. All other aggregate fields reproduce exactly after normalizing local manifest-path prefixes and the excluded-ID redaction. Two CPU censuses; this runtime spread is not a model-performance result.
- No item text, labels or per-item identifiers are exported. JSON contains aggregate source/cardinality counts and ID-set hashes only.
- The original inline source hash is 60b088d86c0ad7e9b943570d4d9bfa3da4172cd1721fc164c51d1abc595c202c. Saved census.py preserves measurement logic; it adds repository import-path setup, source-pin assertions, and output-only redaction of the verifier's three inherited excluded IDs into a count/hash. Its hash is in aggregate.json. The original source/command are preserved in local session log C:/Users/apumu/.codex/sessions/2026/10/07/rollout-2026-10-07T12-29-58-01a117d8-0f5e-76d1-89c2-6b9e2f89c4c1.jsonl, line 198.

## Data exclusion and exact path

census.py calls the integrated train.train_set.draw with the unchanged jobs/m2_first_cell/m2_first_cell_rwkv7.json: 24,000 training records, seed 0, max_k=16, ordered layout and jrt-rwkv-g1-v1. The three pointer-pinned 150k files are re-hashed uncompressed; all 150,000 source lines are manifest-verified. The loader scans/verifies the backing file but retains and renders only the selected training draw.

The eligible pool is 111,795 after three pinned D1-B exclusions and 38,202 v3 exclusions. The draw has zero v2 held-out overlap by ID/content, zero v3 overlap by ID/content/source-item/key-free text, and zero D1-B scan hits across 24,000 selected texts. Calibration, dev, retention and TEST records are not rendered, tokenized or scored. Protected cohort metadata and the DEV/TEST ID/hash manifests are used solely for exclusion; the existing job-uhxb5 hash-only content index is reused without rebuilding or opening suite rows. Cohort, data, guard, config, encoder and source hashes are in aggregate.json.

Base and eligible view rows use the production parent_encoder, items_from_records, build_row, reversed_record and keep_item paths. A bounded 4096-segment LRU cache returns exactly the same token lists and changes no tokenizer/trust policy. Every eligible view is guard-checked, and option-ID domain plus gold/permutation alignment is asserted. No experiment selection or coefficient is fitted.

## Results

| T | Base retained | Paired retained | Pairs moving gold |
|---:|---:|---:|---:|
| 512 | 11,657 | 6,091 | 5,597 |
| 1024 | 18,314 | 9,919 | 9,348 |
| 2048 (current maximum) | 21,348 | 11,059 | 10,458 |
| 4096 (diagnostic only) | 23,755 | 11,555 | 10,930 |

The 24,000 records contain 31,269 question items: 27,839 choice, 1,462 noul and 1,968 score. There are 26,944 K<=16 items and 4,325 K>16 items; 11,643 K<=16 items pass view eligibility before T filtering, and 15,301 do not. One K<=16 base continuation exceeds the 16-token continuation limit.

At T=2048, paired K counts are: K2 2792; K3 1760; K4 5967; K5 64; K6 72; K7 53; K8 90; K10 261; K16 0. At T4096, K16 has 80 pairs. Full K/source/rung counts are in aggregate.json; omitted sparse counters mean zero.

Eligible base/view lengths are identical throughout this draw. Every rung has zero base-kept/view-dropped eligible pairs. Before T filtering, eligible length quantiles (floor((n-1)*p), no interpolation) are p50 493, p80 887, p95 2070 and p99 3890; min 200, max 10304. T2048's 11,059 additional views sum to 6,227,245 real tokens per full traversal. These are not padded positions, fork continuation work, timing, memory or GPU-throughput measurements. T4096 is not added to the registered training ladder by this census.

## 600-step feasibility and budget limits

At T2048 the paired share is 51.8034%, and 94.5655% of retained pairs move gold. A 600-step run with eight base observations per update consumes 4,800 observations; uniform first-epoch sampling would include about 2,486.57 paired and 2,351.43 moved-gold observations in expectation. No actual trainer batch order or model was executed. The 5,000-item minimum is cleared at every current rung; no minimum number of paired examples is invented here.

The existing job-tvuat baseline median 8.247520288998203 s/step projects 600 baseline steps to 82.4752 minutes and $4.0818 training-runtime cost at that receipt's $0.049491/GPU-minute rate. Treatment time and memory are unmeasured: its extra graph and changed grouping require hardware qualification. A 190-minute training ceiling requires a mean of at most 19 s/step before overhead; preserve the existing 76 GB/rate stops and price the complete job separately.

Documented planning arithmetic only: approximately $57 prior spend plus two $70 whole-job caps is $197, leaving $53 of the $250 D allocation. Adding the approximate $50 G13 plan leaves only $3. Neither prior spend nor pending reservations were queried live, and the G13 estimate is not a guaranteed maximum. The existing whole-job formula is (60+2*637)*0.0495=$66.033/job, but it has not been qualified for this new arm. This receipt establishes data support, not that a concrete launch and all downstream safety work fit the remaining allocation. No spend is authorized.

## Reproduction and verification

Run from the owned worktree root; the data, tokenizer and prior hash-only guard paths are pinned inside census.py. Nothing is downloaded and the script writes no output files.

```powershell
Set-Location C:/Users/apumu/research/CHLabs/ssdm-wt/order-consistency-census
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -B -X utf8 receipts/2026-10-07-D-order-consistency-census/census.py
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -B -X utf8 -m pytest -q -p no:cacheprovider tests/test_jobs_lint.py
git diff --check
```

The final CENSUS_RESULT line is the aggregate report. Compare it to aggregate.json's census object, allowing wall_s and local manifest-path prefixes to differ; cached_segments are diagnostic cache counters. Moving the original inline source to a file initially failed to import train; explicit repository-root sys.path setup fixed it before the successful reproduction, and no measurement ran in that failed attempt.

Repository jobs-lint on receipt base 6ab62f5: 69 passed, one existing failure in tests/test_jobs_lint.py::test_coordination_has_no_superseded_spend_rule (3.58 s). That test requires the old literal phrase "requested but not blocking", which the takeover COORDINATION.md no longer contains. Neither coordination nor tests are changed by this receipt. This failure is recorded rather than reported as a passing repository check.

Limits: syntactic eligibility is not general semantic equivalence; source distribution is uneven; current coverage does not address high-K sensitivity; no safety/calibration/abstention/control/interpretability floor or primary model metric is measured. This receipt does not register a new training arm or change the existing default-off objective.
