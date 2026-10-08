# ExactSum fit export: bounded tokenizer-only shape sample

The deterministic sample contains 256 rows from 256 tasks distributed across
the 52,224-row / 5,184-task fit export. **0/256 sampled rows pass the pilot's
T <= 2,048 cap**; 76/256 pass K <= 16. These are sample results, not an exact
full-corpus retention fraction. No model behavior was measured.

`receipt.json` records the executed command, exact indices, hashes, environment,
source hashes, sample quantiles and timing. The standalone utility is
[`scripts/profile_exact_sum_fit_shapes.py`](../../scripts/profile_exact_sum_fit_shapes.py).
It emits JSON to stdout and does not write files. Only derived results and the
utility are committed; input data and tokenizer files remain outside this tree.

## Identity and scope

- Worktree base: `a6b0755a3b81de7948edd71425e663a52fe9557d`.
- Original probe loader: `c8601527c107313b833f4bb922474150522f47bf`.
  `heads/layout.py`, `decide/prompt.py`, `decide/schema.py` and the vendored
  tokenizer are unchanged between those heads.
- Export receipt: `../2026-10-07-L1-exact-sum-training-records-draft/receipt.json`
  in the external `l1-exact-sum-training-records-draft` worktree, with dataset
  manifest `dataset/manifest.json`. Its scope is
  `local_fit_only_draft_v0.2_and_D_parent_pending`; it declares 24 fit worlds and
  excludes calibration, development validation, evaluator expectations/status/
  selectors, adversaries, held-out data and authored programs.
- Compressed SHA-256:
  `8e65294a156164170a88904579ec40203c81fd41b9086c03f40a80e1ad9d3d99`.
- Decompressed SHA-256:
  `8b12acb2aa602ca0e0a9eb5e5bd463d8bb86774e1a76304e8aa1d9b4721e5c0b`;
  705,850,419 bytes. Both hashes were recomputed before tokenization.
- Tokenizer: `RWKV/RWKV7-G1k-13.3B-20260930` at
  `ce3630b854cedbd673569f4600c6cf744bca9dba`.
- `tokenizer.json` SHA-256:
  `1c0bc8e5f355de60a4587a1e9abb9aa4ef65508ff59641344e7f9f10f76feacf`.
- `tokenizer_config.json` SHA-256:
  `2b5e075c2c16780061ae272c9f0b52ff620ddd8e87fdf7fa82fc425661667329`.

The full first pass counts nonblank rows and verifies the decompressed hash.
The second pass counts unique task IDs and collects only positions
`floor(i * (N - 1) / 255)` for `i = 0..255`, with zero-based nonblank positions
and `N = 52,224`. Only those 256 rows are tokenized. The index-list SHA-256 is
`944b9060d6c173d9cedbb0168c59802110e2122416baa318625d11a0b76130ba`
over decimal indices separated and terminated by LF.

## Exact command

Run from the worktree root with the existing Python environment:

```powershell
py -3.12 -B -X utf8 scripts/profile_exact_sum_fit_shapes.py --records C:/Users/apumu/research/CHLabs/ssdm-wt/l1-exact-sum-training-records-draft/receipts/2026-10-07-L1-exact-sum-training-records-draft/dataset/training_records.jsonl.gz --tokenizer-dir C:/Users/apumu/research/CHLabs/ssdm-data/tokenizers/RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba --expected-gzip-sha256 8e65294a156164170a88904579ec40203c81fd41b9086c03f40a80e1ad9d3d99 --expected-jsonl-sha256 8b12acb2aa602ca0e0a9eb5e5bd463d8bb86774e1a76304e8aa1d9b4721e5c0b --max-seconds 180
```

Paths are explicit and may be replaced with local copies having the same hashes.
The utility requires the existing `transformers`, `tokenizers` and `psutil`
packages; it installs nothing. It uses `local_files_only=True`, offline flags,
one worker, one logical CPU, below-normal Windows process priority, a bounded
1,024-entry segment cache, and a 180-second watchdog covering imports and both
input passes. It stops early if an input pass threatens the cap. Failure emits
JSON and a nonzero exit status; watchdog termination returns 124.

## Reproduced sample result

Exact `heads.layout.build_row` accounting uses `ordered` layout and
`jrt-rwkv-g1-v1`, including the segmented schema, state and question re-read.
T is `len(row.ids)`; K is `row.n_options`. Quantiles use nearest rank
`max(0, ceil(q * n) - 1)`, including the sample endpoints.

| Sample metric | Min | p50 | p90 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| K | 2 | 18 | 18 | 18 | 18 | 36 |
| T | 3,280 | 3,949 | 4,251 | 4,306 | 4,466 | 4,467 |

K counts: `2:30, 4:23, 15:23, 18:179, 36:1`. K <= 16 passes 76/256
(29.6875%); T <= 2,048 passes 0/256. The joint filter, including continuation
length <= 16, passes 0/256 (0%). Maximum sampled continuation length is 4.

The committed utility reproduced the earlier ephemeral uniform-position
probe's index hash, K counts, K/T quantiles and all cap counts exactly.
The earlier ordered-prefix probe is not used for these results.

Runtime was 10.469 seconds total: 0.938 seconds for count/hash, 1.766 seconds
for collection and 2.453 seconds for sample tokenization. Final RSS was 275 MiB.
This is tokenizer/data-processing wall time, not model throughput.

Environment: Windows 11 `10.0.26200-SP0`; AMD Ryzen AI 9 HX 375 w/ Radeon 890M,
12 physical cores / 24 logical CPUs, affinity `[0]`; CPython 3.12.10
(`tags/v3.12.10:0cc8128`, MSC v.1943 64 bit); `transformers` 5.14.1,
`tokenizers` 0.22.2, `psutil` 7.2.2. The exact interpreter and version string
are in `receipt.json`.

Fail-closed check: rerunning with an all-zero expected gzip SHA returned exit 1
in 0.046 seconds, with `phase=verify_compressed_input_and_tokenizer` and
`error=gzip_sha256_mismatch`, before tokenizer loading.

## Limits

Uniform record positions are deterministic and are not independent random
sampling or task/world-stratified sampling. Sample extrema and quantiles do not
establish full-corpus extrema or quantiles. Fit-only provenance comes from the
hash-pinned existing export receipt; no held-out or Index inputs were opened.
The utility does not independently regenerate or validate the authored labels.

No model was constructed, no weights were opened, and no forward/backward,
GPU or network calls occurred. Importing the existing `heads` package imports
Torch indirectly, but the profiler performs no Torch operations. This receipt
does not qualify capability, calibration, safety, training, parent approval,
spending, or any held-out result. Source hashes capture the exact code run;
`source_head` records the base before this utility and receipt were committed.
