# Receipt: q9 proxy loop stock-check (A1(d)) — FAIL, loop halted

**Verdict: stock-check FAILED (|Δ| = 0.772 > STOCK_CHECK_TOL 0.3).** Per PREREG A1(d) the proxy loop is
no-measurement; `recipes`, `select` and `holdout` were NOT run (the code also refuses them: `require_stock_check`).

## Inputs
| input | identity |
|---|---|
| code | agent-main 401bc9a (worktree q9fcl-run, detached) |
| stock_train cache | art-mtqne, sha256 443954ff…1b00; train MANIFEST b642bf7d… (70 shards) |
| stock_dev cache | art-7x7dp (job-sxv3u), sha256 bc2d58d6de89c38471c094b860bf1354fcc8213235477b7ad3ef8fd1b458529f; select MANIFEST b94b7224…, holdout MANIFEST 018dc5a7… |
| job-5cmat stock results | sha256 646b61490a37b3aa8887a57bb1a6797f8254c0136a63a0c3cf18d00f90497beb |
| dev rows | ssdm-data/slices/dev-rows.jsonl.gz |

Command (`run_proxy_loop.sh`, kit = ssdm-data/decision-index):

    run_proxy_loop.sh stock-check --dev-rows .../dev-rows.jsonl.gz --stock <job-5cmat stock results.jsonl>

Exit 3, 48 s wall. Output `stock_check.json`, results `stock_from_cache_select.jsonl.gz`
(uncompressed sha256 12ffc634…0da7).

## Result
| | SELECT index |
|---|---|
| cached stock readout | 39.04 (CI 35.70–42.00) |
| job-5cmat reference | 38.27 |
| Δ | **+0.772** (paired CI −0.20 … +1.65, p≤0 0.055) |

## Diagnosis (post-hoc, not part of the rule)
`diag_rowdiff.py`: 2,008 ids identical. job-5cmat has **58 `error` (backend_error, e.g. POP909 chord rows) + 7
`unsupported`** rows that score zero; the cached readout answers all 65. On the 8,185 questions both answered, the
choice differs on 51 (0.6%; bf16 noise, median max|Δp| 2e-4).

`diag_masked.py`: replacing those 65 rows with job-5cmat's failed rows verbatim gives **38.317, Δ = +0.047**
(paired vs 5cmat: +0.048, CI −0.79 … +0.72). The whole gap is job-5cmat's serving failures, not a feature/readout
mismatch. Whether to amend the reference (e.g. compare on the jointly-answered rows) is a decision for the
coordinator/user; this receipt changes nothing in the rule.
