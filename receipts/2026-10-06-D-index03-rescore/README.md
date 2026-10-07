# Receipt: Decision Index 0.3 verification, edition-flagged harness, $0 re-score, held-out-family split (Track D)

Billed $0. CPU only. No GPU, no cloud job, nothing submitted anywhere. The TEST slice was not opened (only `load_slice("dev")`). Backs `INDEX_0.3.md`.

## Primary sources fetched 2026-10-06 (17:20-17:45 UTC)

| What | URL | Identity |
|---|---|---|
| Board Full scores | https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/v03.json | sha256 `9f8a0a0327b800c33219042fbed6db597557819608c7254dc37fb183e13b01af`, `generated_utc 2026-10-06T16:32:56Z` (vendored: `harness/tests/fixtures/v03.json`) |
| Board 0.3 public index, per benchmark | …/resolve/main/data/index.json | sha256 `62b016ec9eb4f5bbd5bc67a2d1b0fad62372b5ed674264fae7dca72601e0d4dc`, `generated_utc 2026-10-06T17:39:21Z` (not vendored, 1.5 MB) |
| Board methodology (rules, changelog v0.3) | …/resolve/main/data/methodology.json | sha256 `d68c323f4d8d87d2efc4a1218751b5005629116e7ee2a858d53bac68f9418294` |
| Board 0.2.1 (unchanged) | …/resolve/main/data/index-v0.2.1.json | sha256 `37526dd3…0d50` (= the pin already in `harness/config.py`) |
| Space revision | https://huggingface.co/api/spaces/multimodalart/jev-decision-index | sha `6ceab8392133b34f880d6f87f66ba068f068d2af`, lastModified 2026-10-06T17:39:36Z |
| Kit PR "Edition 0.3" | https://github.com/apolinario/decision-index/pull/74 | **open, not merged**; head `62d2f51de34a2de64906345b6bc3e98e27ff55c7`; `decision_index/data/index-0.3.json` sha256 `158c3843…7167` (vendored: `harness/editions_data/index-0.3.json`); `tests/fixtures/board-0.3.json` sha256 `062466c3…83b5` (vendored: `harness/tests/fixtures/board-0.3.json`) |
| Kit main | https://github.com/apolinario/decision-index (`87d4650b`, "Version 0.2.1") | README still "The live board is Decision Index 0.2.1"; §Rules already has "No prompt tuning … one fixed rendering" |
| GSM8K artifact | https://github.com/apolinario/decision-index/issues/32 | open; gold's rank among sorted values fixed (GSM8K-10 4th-6th smallest; "5th smallest" alone 57%) |

Cross-checks run on those files (all in `harness/tests/test_edition03.py` or reproducible from the commands below):
- the kit fixture's 10 non-Jev entrants equal the live `index.json` per-benchmark skills (max diff 0.0) and `public_skill`;
- `full03.full` reproduces all 112 `v` to <= 0.015 and every `rank_tie`/`tied`;
- the undocumented per-benchmark `private_eq` aggregates (chance-corrected, public weights, 34 benchmarks) to within 0.7 of `same_skills` (Clef 52.43 vs 53.11, Clef-flash 43.58 vs 44.01, Nimble 51.85 vs 52.34, Kev-27B 54.98 vs 55.51, Perplexity 60.51 vs 61.10) -- indicative only.

## Re-score

```sh
PYTHONPATH=<ssdm-data>/decision-index uv run --with numpy --with pyarrow python -X utf8 \
    receipts/2026-10-06-D-index03-rescore/rescore.py <scratchpad> > receipts/2026-10-06-D-index03-rescore/rescore.json
```

Inputs (outside git; `rescore.json` `inputs` has paths, sha256 and row counts; 0 missing rows):

| System | Job | results.jsonl sha256 (16) | Rows |
|---|---|---|---|
| Clef-flash `decide` | job-sgihj | `463f21ff492fc792` | 12,104 (full DEV) |
| Qwen3.5-9B stock | job-5cmat | `646b61490a37b3aa` | 2,008 (m1 subset `6e4cb8df…95c3`) |
| Qwen3.5-9B 1 attn → GDN | job-5cmat | `021a19259eff7879` | 2,008 |
| RWKV7-G1k-13.3B stock | job-tkfaj | `82b027ed20b2f623` | 2,008 |

B = 2,000, seed 0, source-case clusters; all systems in a block share resamples (paired). Reproduction check: 0.2.1 subset stock 38.27 and Clef-flash 55.78 equal the earlier receipts (`jobs.common.dev_subset score`; PR #92 receipt). Tables: `INDEX_0.3.md` §3 and §5. Runtime ~10 s.

## Tests

```sh
PYTHONPATH=<ssdm-data>/decision-index uv run --with pytest --with torch --with numpy --with pyarrow --with transformers==5.17.0 \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match python -X utf8 -m pytest -q harness/tests jobs
```
Result in `pytest.txt` (148 passed after the 2026-10-07 amendments: edition-enforcing replicates, heldout-draw-v1, 0.3 GSM8K addenda).
