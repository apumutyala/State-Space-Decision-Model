# RWKV7-0.1B q_end operation-only diagnostic

This is a **reused-development diagnostic**, not an L0 qualification. It tested
whether a small CE-only q_end head can improve the planner's initial operation
selection while leaving the later stock readout intact. The model was the
cached open-weight `fla-hub/rwkv7-0.1B-g1` snapshot pinned in `registration.json`.
Execution was local CPU-only, with CUDA hidden, two threads, a 6 GiB start
guard, 3 GiB runtime guard and 1,200-second cap. Cost was $0.

Source was `agent-main` base `018585548104dfbcb9d65a22214ec51405529711` plus
the frozen operation-only runner/source registration at run source `94d440f`.
The exact command and environment were:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m language_growth.rwkv_operation_cache --out receipts/2026-10-07-L-operation-only-run1 --execute
```

The frozen corpus had 24 operation-only training rows and 12 reused-development
rows (four each for read, explain and compare). The three arms were stock,
step-0 head and 96-step CE-trained head. No Index or held-out items were used.

| Result | Stock | Step 0 | Trained |
|---|---:|---:|---:|
| Live dev operation choice | 4/12 | 4/12 | 5/12 |
| Useful answers | 1/12 | 2/12 | 1/12 |
| Kernel accepted | 12/12 | 12/12 | 11/12 |
| Cached operation fit (train) | n/a | 8/24 | 24/24 |
| Cached operation fit (dev) | n/a | 4/12 | 5/12 |

Trained-minus-stock useful-answer difference was **0 [0, 0]**; acceptance was
**−1/12 [−0.25, 0]**. Trained-minus-step-0 useful-answer difference was
**−1/12 [−0.25, 0]**. These are paired percentile-bootstrap intervals over
12 generated development tasks (seed 9203, 2,000 draws), not Decision Index
scores. Read/explain/compare kind CIs and the full table are in
`receipt.json` and `independent-remeasurement.json`.

The model fit the 24 training operation labels perfectly but did not improve
useful dev answers. One trained Compare path selected the correct operands but
the unchanged stock relation slot chose `greater_than` for 204.75 versus
213.23, so the kernel rejected it. The four direct bookends were exactly equal
before and after, but all were wrong: the four references were `less_than`,
while stock predicted `greater_than`. This small, label-skewed direct subset
does not establish broad parent retention or capability.

An independent no-Torch replayer matched all 74 saved requests/responses,
regenerated every planner/kernel/renderer outcome, recomputed all paired and
per-kind intervals, and found zero saved-versus-recomputed differences. It
used the recorded `direct_before`/`direct_after` arrays; the run did not save
direct-call envelopes in `raw.jsonl`, so request identity and direct accuracy
outside the four comparison cases cannot be independently verified. Cached
CE is an aggregate of saved per-row CE, not independently recomputed from
score vectors. The audit script and structured report are
`analysis/independent_operation_only_recompute.py` and
`independent-remeasurement.json`.

The pre-registered gate required strong train fit, strict useful-answer gain,
read non-regression and complete paired evidence. Strong train fit passed;
the useful-answer gain did not. **Fold this exact 96-step CE-only q_end recipe**;
do not promote it or infer L0 competence. The result is recorded in
`CANDIDATES.md` and `LEDGER.md`.
