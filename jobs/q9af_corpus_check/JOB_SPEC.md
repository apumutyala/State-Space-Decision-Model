# q9af_corpus_check: is job-w3c2e's distillation corpus free of eval material?

**Why.** job-w3c2e (jobs/q9af_v2: Qwen3.5-9B with all 8 attention layers converted to GDN by end-to-end distillation)
built its 280M-token generic mix in-job (`q9af_v2_job.sh` phase D, `convert/mixdata.py`: fineweb-edu 55% /
ultrachat 30% / finemath 15%, every window read at most once, deterministic) with a **DEV-only** 13-gram filter
(`dev13-v1`). Operator rule: we can't train on test, and that covers TEST, held-out Index rows, the frozen catalogs
[9, 20, 22, 31, 32, 33, 44, 58, 64] and the eval attack templates. w3c2e's result is **provisional** until this
census says what its corpus contained.

**What w3c2e exports.** Only `mix.json` (per-source token / doc / drop counts, file sha256s, `train_sha256`,
`heldout_sha256`), inside `/output/phases/D.tgz` (export D sha256 `efd3be06...`, job log 2026-10-09T07:27:44Z). No
document or window ids. The job is still running, so no artifact can be fetched yet; the check therefore rebuilds.

## What the job does (cpu-8: 8 physical cores, 32 GiB, no GPU)
1. **setup**: env pins, the measured-rate preflight (`jobs/q9af_corpus_check/preflight.py`, exit 5), sha256 of the
   shipped DEV rows and every eval-index file (`data.sha256`, exit 10).
2. **fetch**: `scripts/sf_fetch.py --stage q9af_corpus_check`: the three parquet sources at w3c2e's pinned revisions
   and the 9B **tokenizer files only** (`*.json`, `*.txt`, `*.jinja`; never the safetensors). `hf_cache` mounted.
3. **C**: `python -m jobs.q9af_corpus_check.corpus_check run` (one streaming pass):
   - `convert.mixdata.build` with w3c2e's config (`configs/q9af_v2_h100.json`, `mix.decontam` = `dev13-v1`, seed 0,
     the same DEV rows) rewrites train.u32; its `on_doc` hook hands every WRITTEN document (id
     `stream/source/f<file>/rg<row group>/r<row>`, token offset, length) to 6 spawn workers as it is written.
   - Gates: the scanned documents tile train.u32 (sum of lengths = train tokens, per-source kept docs = mix.json; exit
     15); the rebuilt mix.json equals w3c2e's once `w3c2e_mix_pin.json` holds it (exit 16; until then
     `corpus_check compare` does it offline). `train_sha256` equal = byte-identical training stream.
   - Classes, per document (reported separately; counts, tokens, windows, per source, up to 25 example ids):
     - **policy**: `train.doc_overlap_guard` rule v1 over the whole draw (`scan_hits`: R / E / U, boilerplate,
       reviewed allowlist) vs the pinned eval-hash index (DEV + TEST + held-out, `index_sha256` `2c821e29...`), classed
       by the best eval unit: TEST / HELDOUT / DEV, plus FROZEN when its family is a frozen catalog.
     - **contact**: any distinctive eval 13-gram (non-allowlisted unit, < 100 eval rows: the full-eval-v1 rule), by
       every eval row it occurs in. Informational, and exactly what `full-eval-v1` drops in future builds.
     - **attack**: `train.attack_guard` (every eval attack template fragment, base64 payloads decoded).
     - frozen generator check: `harness.heldout.check_generator` / `check_record` on the mix's generator spec and
       source-file metadata (the kit is in the image).
   - Windows: train.u32 is consumed as non-overlapping 4096-token windows in one permutation (`convert.joint`:
     `WindowStream(seed = cfg.seed + 7)`); each flagged document lists its windows and their permutation ranks.
     After w3c2e lands, `corpus_check resolve --windows-used <stage2 windows_used_after>` counts the windows it
     actually trained on.
   - Output (ids and counts only, never eval or TEST text): `report.json`, `flagged.jsonl.gz`, `mix.json`.

**Verdict.** No decision rule (`decision_rule: null`): the census reports `CLEAN` (no policy fail in any class and no
attack hit in the training stream) or `CONTAMINATED` with the counts; the operator decides what w3c2e's result is
worth. `/output/Q9CC_DONE` only when the report is valid (finalize validator `corpus_check:valid_report`).

## Caps and cost
| phase | cap (min) | projected (min) | basis |
|---|---|---|---|
| setup | 10 | 2.0 | data sha256 over ~290 MB + preflight |
| fetch | 20 | 2.6 | ~3.0 GB parquet at 20 MB/s (w3c2e fetched + built the mix in 4.87 min) |
| C | 60 | 9.8 | 280.3M tokens at 0.8M tok/s (laptop 2.40M tok/s on 24 threads, scaled to 8 cores) + ~224M words at 6 workers x 0.45M words/s x 0.7 + 2 min fixed |
| reserve | 10 | | |

max_duration 100 min, build_timeout 60, one attempt, cpu-8 at $0.009/min ($0.54 per 60 min, `autor job validate`
2026-10-09): **worst case $1.44** (hard cap $2.00), expected ~30 min wall incl. a ~15 min image build (the q9af_v2
layers, likely cached) = **~$0.27**.

## Launch (do not submit from an agent; the operator submits)
    python -m jobs.common.launch q9af_corpus_check --sha <pushed sha> --out C:\Users\apumu\q9af_corpus_pkg --data C:\Users\apumu\research\CHLabs\ssdm-data
    python -I jobs/common/sf_submit.py C:\Users\apumu\q9af_corpus_pkg --dry
After it lands: `corpus_check compare --rebuilt report-dir/mix.json --expected <w3c2e out/mix.json>` and
`corpus_check resolve --flagged report-dir/flagged.jsonl.gz --windows-used <N>`.

## The future-proofing patch (convert/mixdata.py)
`mix.decontam` selects the filter. `dev13-v1` = w3c2e's (recorded in `configs/q9af_v2_h100.json`, so a rebuild of
w3c2e stays byte-identical, and its mix.json keeps w3c2e's shape). Any config without the key gets `full-eval-v1`:
dev13-v1 + drop every doc sharing a distinctive word 13-gram with the pinned eval-hash index (DEV + TEST + held-out,
frozen families included) + drop every doc containing an eval attack template fragment; drops counted per reason;
mix.json records the filter, the index sha256 and the attack template sha256.
