# Receipt: Clef-flash vs stock Qwen3.5-9B gap attribution and a $0 readout-fusion test (Track D, recipe-research)

Billed $0. CPU only. Backs `techniques/10-clef-recipe-and-gap-attribution.md` sections 4 and 7.

## Inputs (all already measured by earlier jobs; nothing re-run on a GPU)

| Input | Job / LEDGER row (branch `claude/ledger-2026-10-06`) | sha256 (first 16 hex) |
|---|---|---|
| Clef-flash `decide` path, full DEV, 12,104 rows | job-sgihj (Clef-flash baseline A, $1.40) | `results.jsonl` 463f21ff492fc792 |
| Qwen3.5-9B stock readout, 2,008-row DEV subset | job-5cmat (probe9b, stock arm) | `results.jsonl` 646b61490a37b3aa |
| Qwen3.5-9B with 1 attention layer -> GDN, same subset | job-5cmat (conv arm) | (`score_conv.json` in the job export) |
| RWKV7-13.3B stock state-fork readout, same subset | job-tkfaj | (`score.json` in the job export) |
| Index 0.3 board data | HF Space `multimodalart/jev-decision-index` `data/v03.json`, `generated_utc 2026-10-06T16:32:56Z` | 9f8a0a0327b800c3 |

Subset = `jobs/common/dev_subset_m1.json` (ids sha256 `6e4cb8df…95c3`). DEV rows are private and are not in this directory. The fused result files carry request payloads and are kept outside git.

## Commands

```sh
python -I gap.py <scratch>                      > gap_table.txt          # per-benchmark gap, Index-weighted
python -I -X utf8 paired.py <scratch> dev-rows.jsonl.gz > paired_by_idclass.txt   # question-level accuracy by option-id class
python -I -X utf8 status.py stock/results.jsonl > stock_status.txt       # ok / error / unsupported per benchmark
python -I fuse.py <scratch> <w> pool fused_<w>.jsonl                     # log-linear pool p ∝ p_clef^w · p_stock^(1-w)
PYTHONPATH=<ssdm-data>/decision-index python -X utf8 -m jobs.common.dev_subset score   --results fused_<w>.jsonl --B 500
PYTHONPATH=<ssdm-data>/decision-index python -X utf8 -m jobs.common.dev_subset compare --candidate fused_<w>.jsonl --baseline fused_1.0.jsonl --B 1000
```

`dev_subset score` on the untouched stock file reproduces the job's 38.2687 exactly (check run before the fusion runs).

## Results

| Arm | Subset Index [95% CI] | K / L / R / T / A |
|---|---|---|
| w = 0.0 (stock; Clef fills the 58 POP909 + 7 over-16k rows the stock arm failed) | 38.37 [35.4, 41.4] | 25.0 / 47.4 / 40.2 / 49.9 / 24.9 |
| w = 0.5 | 56.65 [53.4, 59.4] | 48.0 / 64.4 / 49.8 / 70.5 / 47.3 |
| w = 0.7 | 57.04 [54.1, 60.2] | 47.8 / 59.8 / 50.9 / 77.0 / 49.6 |
| w = 1.0 (Clef-flash) | 55.78 [52.9, 58.6] | 47.2 / 56.2 / 45.9 / 82.4 / 48.0 |

Paired difference vs Clef-flash (cluster bootstrap, B = 1000): w = 0.7 **+1.27 [−0.38, +2.83]**; w = 0.5 +0.87 [−1.99, +3.34]. Tools falls (−5.4 / −11.8), language and retrieval rise.

**Caveats.** w was picked from three values on the same rows, so +1.27 is in-sample and optimistic. The CI includes zero. This is a diagnostic of complementarity, not a candidate. Fusing two systems that each use attention says nothing about attention-freedom.
