# Cost and latency: RWKV7-13.3B vs Clef 27B and Clef-flash (2026-10-08)

**Takeaway.** On matched conditions, RWKV is **slower and more expensive per Index row than both Clefs**. The
trained cell D1 is 8.1x slower than Clef 27B and 18.9x slower than Clef-flash. The stock RWKV readouts are 2.7-3.0x
slower than Clef 27B and 6.3-6.9x slower than Clef-flash. **The data supports no 2x cost/latency claim against either
Clef, at any Index.** D1's 2.78x slowdown against its own parent is a serving-path problem: per-question
backbone calls and repeated tokenisation, not the model. The exact fixes (branch `stack/rwkv7/cost-latency`) are
projected to bring D1 to about parent speed (~3x faster). Even then RWKV would still be ~2.7x slower than Clef 27B.

Matched conditions: 1x H100 80GB HBM3 (SXM) for every arm, batch 1, one request at a time, `clock_lock: true` in every
job's submit block, the same 2,008-row Index DEV subset, and the **2,000 run_ids that are `ok` in every arm**. The
metric is the kit's per-row `total_wall_ms` around one `decide` request. Clef is shown on its FAST (fla, production)
GDN path: job-d2u54 for Clef 27B, and job-sgihj for Clef-flash (sgihj's pinned decide file, from ftpnf's
`decide_32k_substituted`, with the 51 substituted rows excluded). The torch-reference runs are listed only to show
how big that confound is. Reproduce with `python -I -X utf8 analysis/cost_latency.py`. The script reads each landed
artifact's tar members directly and extracts nothing.

| arm (2,000 common rows) | GDN path | Index (subset) | median ms | p95 ms | rows/min | $/1k rows | proxy tok/s | time vs Clef 27B fast | time vs Clef-flash fast |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clef 27B (d2u54) | fast | 62.14 | 100 | 782 | 266.9 | 0.185 | 4989 | 1.00x | 2.32x |
| Clef 27B (zfexa) | reference | 62.09 | 288 | 1742 | 119.6 | 0.414 | 2236 | 2.23x | 5.19x |
| Clef-flash (sgihj) | fast | 55.78 | 57 | 287 | 620.5 | 0.080 | 11600 | 0.43x | 1.00x |
| Clef-flash (t5yg2) | reference | 55.55 | 154 | 681 | 254.7 | 0.194 | 4760 | 1.05x | 2.44x |
| RWKV7 stock g1-v2 (zhyvg) | chunked WKV | 27.31 | 352 | 1625 | 98.7 | 0.502 | 1845 | 2.70x | 6.29x |
| RWKV7 stock v3w (fgu2c) | chunked WKV | 31.07 | 292 | 2662 | 89.5 | 0.553 | 1673 | 2.98x | 6.93x |
| RWKV7 parent g1-v1 (c7pq2 seed) | chunked_lean | 24.85 | 272 | 2909 | 91.3 | 0.542 | 1707 | 2.92x | 6.80x |
| **D1 trained cell (c7pq2)** | chunked_lean | **35.12** | 546 | 11913 | 32.8 | 1.507 | 614 | **8.13x** | **18.90x** |
| D1 + exact fixes (projected) | chunked_lean | 35.12 (same outputs) | | | ~97 | ~0.51 | ~1820 | ~2.7x | ~6.4x |

$/1k rows is billed H100 time at $0.0495/min. Index per $ per 1k rows: Clef-flash 699, Clef 27B 335, stock RWKV 46-56,
D1 23.

**Confounds that remain.**
1. The Clef numbers in zfexa and t5yg2 are on the slow torch reference GDN, which costs 2.2-2.4x. The fair Clef speed is
   the fast-path row (d2u54 / sgihj). Both fast-path runs exist on this subset, so no Clef re-measure is needed for the
   subset. A full-DEV fast-path Clef 27B run does not exist (jxevr stopped at S1).
2. Cloudflare's own card latency is H200 (Clef-flash 38.8 / 122.4 ms, Clef 209 ms median). Our H100 medians are
   57 / 100 ms. Do not mix the two.
3. Prompt lengths differ. Clef reads a request in one sequence of its own tokens (e.g. 4,313 for the 17-question
   ContractNLI row). The RWKV readout reads 5,841 World-tokenizer tokens there (shared prefix + per-option fork tails).
   `proxy_tokens` is the tokenizer-free numerator used for tok/s. Model-token rates are not comparable across families.
4. Batch and prefix sharing: every arm runs batch 1 sequentially. All arms share the prefix within a request (Clef
   natively, RWKV by state fork). None shares across requests.
5. The parent-arm timing was seeded from job-uhxb5, a different box of the same class. D1's timing comes from job-c7pq2.

**Why RWKV is slow even stock.** The RWKV readout runs at 1.7-1.8k proxy tok/s. Clef 27B, with 24.4B matmul params per
token against RWKV's 12.7B, runs at 5.0k. RWKV therefore does about half the FLOPs per token, yet takes 2.7x longer.
That is roughly 4x lower efficiency per FLOP. The cause is the pure-PyTorch chunked WKV (3.6-4.3k tok/s prefill, about
11.5% of H100 bf16 peak, jobs/fla_rwkv7/JOB_SPEC.md). fla `chunk_rwkv7` measured ~3x faster prefill but was rejected
on numerics (job-jn47u, job-mden7). That kernel is the biggest lever for every RWKV arm, and it is gated on numerics,
not on speed.

## Why D1 is 2.78x slower than its parent (ranked, with evidence)

A per-row least-squares fit (R^2 0.979) over the 2,000 rows explains child wall time. Its inputs are the backbone calls,
tokens and tokenisation volume recomputed per row from the real DEV rows and tokenizer.

| # | cause | share of D1 wall | evidence |
|---|---|---:|---|
| 1 | **One backbone call per question.** `heads/prefix_features.shared_prefix_features` makes 1 prefix + 1 per distinct suffix length + 1 cache-fork call per question: 7.7 calls/row on average, 64.5 on rows with 20+ questions. The stock readout packs every (question, option) tail into ~2 calls (`decide.backends.state_fork`). A 61-layer pure-PyTorch call is launch-bound at ~121 ms. | **~51 %** | 121 ms/call fitted. child/parent ratio grows with questions: 1.72x (1 question), 2.91x (2-4), 4.20x (5-19), 3.71x (20+). Per-question cost 307 ms (child) vs 36 ms (parent) |
| 2 | **Prompt tokenised 2xQ times.** `DecisionModelBackend._rows` rebuilds every question's full prompt (state included) in both `plan()` and `score()` with no memo. The stock readout memoises segments (`HFReadoutBackend._enc_memo`). | **~27 %** (laptop CPU timing) | 107k tok/s on CPU, 126 ms per row per pass, 5.9 s per pass for the 64-question row. 46% of wall on rows with 20+ questions. With a memo: 46 s in total, against 989 s |
| 3 | Real token work (prefix + suffix, continuations) | 18 % | 0.11 ms/token. Not a slowdown cause, since the parent reads the same tokens |
| 4 | **Unmerged LoRA** (r=64 on 366 Linears). Each call casts A and B from fp32 to bf16 and adds 2 matmuls, a multiply and an add per Linear: ~2.2k extra kernel launches per call. FLOPs +2-3%. | inside #1's per-call cost (not separable here) | code: `train/lora.py` `LoRALinear.forward` |
| - | Not causes: temperature fit (once per job, applied post hoc), lineage checks (once at load), dtype (bf16 in both arms), WKV path (`chunked_lean` in both arms), activation checkpointing (bypassed under no_grad). | 0 | code read: `train/m2_gates.index_predict`, `fulldev_eval.load_child`, `heads/act_ckpt` |

## Fixes (implemented, opt-in, training untouched): `DecisionModelBackend(..., inference="fast")`

| fix | exact? | CPU check | expected speedup on D1 |
|---|---|---|---|
| `memo_encode`: per-request segment memo | identical ids | `tests/test_infer_opt.py` | 1.35x alone |
| `packed_fork`: prefix once + every (q, option) tail in packed right-padded calls (64 rows / 8,192 positions max) | float rounding only | fp32, tiny RWKV7 + Q9-AF, every head: packed = grouped = plain at 1e-5; 2 calls instead of 1+G+Q | 1.55x alone |
| `merge_lora`: W + s·B·A folded in fp32, stored bf16 | fp32: 1e-5. bf16: rounds once, so the GPU A/B must check it | fp32 test; backend revision hashed before the merge | +10-25% (est., unmeasured) |
| memo + packed together | | | **~3.0x (1.83 s → ~0.62 s per row; about parent speed)** |

Default off: the previous revision id and path are unchanged. Tests: `tests/test_infer_opt.py` (21 passed),
`tests/test_latency_ab.py` (3 passed), `tests/test_perf_paths.py` (31 passed), `tests/test_m2_fulldev_eval.py`
(9 passed).

## GPU A/B package (draft, NOT submitted, NOT packaged)

Mode `latency_ab` (`jobs/m2_first_cell/job_params_latency_ab.json`). It loads the D1 checkpoint once and scores the
250 pre-declared smoke rows twice: `base` first, then `fast`. `fulldev_eval ab-compare` then checks status, per-question
decision flips, max |dp| and wall time row by row. Expected cost is ~66 run min = **$3.27** billed, ~$4.0 with the build;
worst case $16.04. To package, run `python -m jobs.common.launch m2_first_cell --mode latency_ab --sha <sha> --out <dir>
--data <ssdm-data>`, then put the art-bm3e6 URL into `submit.json` env `CKPT_URL`. Packaging was refused by the
permission system in this session, so it is left to the operator.

## What 2x would take

D1 with the fixes is ~2.7x slower than Clef 27B. Beating Clef 27B 2x on latency needs another ~5.5x. Three levers
remain: a numerically accepted fast WKV kernel (~3x prefill), fewer read tokens (packing all questions into one
sequence, as Clef does), and CUDA graphs for the launch-bound calls. Beating Clef-flash 2x would need ~13x and is not
realistic for a 13.3B model. On capability, D1 is at 35.12 against 55.78 and 62.14.
