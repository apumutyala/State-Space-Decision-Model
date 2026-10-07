# PR #151 CPU reconciliation and corrected handoff

The negative calibration result reproduces exactly. Probability averaging remains unmeasured. The evidence does not establish that averaging is ineffective, or isolate label preference from position preference. Keep the calibration correction folded; do not add or enable a serving flag on this evidence.

2026-10-07; local Windows CPU; no model/checkpoint loads, paid work, or GPU calls; $0. One deterministic re-score (B=2000, seed=0), plus a separate raw-outcome reconciliation. This is a receipt audit, not a new model candidate or a qualification run.

## Verified repository and session state

- Existing PR: https://github.com/apumutyala/State-Space-Decision-Model/pull/151, OPEN against `agent-main`, head `stack/rwkv7/order-debias` at `478aec24f19656243ec8e347bf1563acb188c5b8`; no comments or reviews when checked. Original worktree `C:/Users/apumu/research/CHLabs/ssdm-wt/order-debias` was clean and was not edited.
- The diff adds only the original four receipt files. It changes no scoring code, cell config, or serving flag. There is no `ORDER_DEBIAS.md` proposal. Nothing from PR #151 has been merged.
- This follow-up is isolated on `codex/order-debias-handoff-20261007`, based on fetched `origin/agent-main` at `ee0adb70029fc80c65ee6072c307873034052362`.
- Actual Claude worker log: `C:/Users/apumu/.claude/projects/C--Users-apumu-research-CHLabs/167d8677-872b-45ed-b144-905f7c6dd3bc/subagents/agent-a48befd72e8106751.jsonl`. Its first entry is the coordinator's delegated brief, not a direct human message. Lines 126, 143, 151, 154 and 157 record the CPU command, input hashes, commit/push, PR creation and final handback. The parent log at the same root, `167d8677-872b-45ed-b144-905f7c6dd3bc.jsonl`, line 10916 contains the human request to swarm, parallelize and pipeline. The handback at parent line 11007 explicitly identifies itself as model output with no user authority.

## Reproduced result

`recomputed.json` is byte-identical to the saved `analysis.json` (SHA256 `347eca08ba18c1c6ae8891c4c1d2cdb535f92777a2f70e38fb72de4250d492d5`). `reconcile.py` also recomputes `position_bias.json`, verifies the exact diagnostic row/question selection and stored outcome schema, and checks the interpretation counterexamples below.

| Variant | Index (0.2.1 DEV subset) | Paired delta [95% CI] | Harness board-protocol ECE |
|---|---:|---:|---:|
| Original forward | 24.8460 [22.3463, 27.7075] | 0 | 0.082056 |
| 5-fold group cross-fit | 21.7471 | -3.0989 [-5.4649, -0.1585] | 0.106592 |
| Leave-one-benchmark-out | 22.7018 | -2.1442 [-4.5700, +0.8464] | 0.110962 |
| Probability/log-prob order average | unavailable | unavailable | unavailable |

All 2,008 rows remain in the Index denominator: 2,000 `ok`, eight `exhausted`. The calibration-only correction is diagnostic cross-fitting on DEV inputs, not a deployable train-fitted prior. Its paired CI does not refit the prior within each bootstrap replicate, so it is conditional on the already computed cross-fitted predictions. The `board_protocol` ECE is this harness's stand-in; `harness/INDEX_NOTES.md:52` explicitly says its sample/benchmark set are not board-comparable.

## Interpretation corrections

1. **Small net endpoint excess does not mean order invariance.** At k=2, first-in-both-orders occurs on 25/387 questions and last-in-both on 31/387; these partly cancel in the reported -0.009 statistic. The mapped choice changes on 56/387 questions. At k=4, it changes on 14/38. The original forward first-choice rate 0.691 versus gold-first 0.669 is consistent with label/gold imbalance, but it does not prove the causal attribution "mostly label prior."

2. **For k>2, the reported `position_effect` is endpoint concentration relative to uniform, not isolated position sensitivity.** An invariant model choosing one fixed original endpoint in both orders has `(1+0)/2 - 1/k`, which is +0.4870 at k=77 despite zero choice changes. The existing high-k data do show order/relabel sensitivity: 21/24 changes and six first-in-both-orders. Position, numeric-prefix preference, continuation-token changes and prompt context are not separated. Also, the 5-16 bucket has +0.148 on 16 questions; "real primacy only k>16" is too categorical. The >16 set includes ChessBench-legal-move as well as the other listed tasks; `reconciliation.json` records all datasets by bucket.

3. **The binary both/either numbers need a failure convention.** The 387 binary questions include one with null choices in both conditions. Treat it as wrong and return no ensemble answer if either required pass fails. Under valid normalized binary probabilities and consistent argmax/tie handling, the observed both/either accuracies bracket a two-pass average; they are not an observed averaging result. No probability data exist to audit the averaging or boundary ties. They say nothing directly about the full Index, which also includes nonbinary questions and benchmark-specific metrics.

4. **The existing diagnostic changes IDs as well as order.** `jobs/m2_first_cell/readout_diag.py:118` relabels `id` to `NN-id`; `decide/prompt.py:173` also uses that changed ID in the scored continuation. Both diagnostic conditions are relabelled; neither equals the original scored forward condition. One new reversed/relabelled full pass plus the existing original forward would measure a mixed-ID ensemble, not a matched relabelled forward/reverse average. A symmetric relabelled pair requires two new full passes with their probability maps saved. The current diagnostic only relabels gold-eligible choice questions; a serving experiment must transform all applicable choice questions without consulting `expected`.

5. **The stage timer was mistaken for forward latency.** `train/m2_gates.py:340-353` times both the 2,008 Index requests and 320 extra diagnostic requests. `units-index.json.seconds=1531.344` therefore includes those diagnostics. Summing the scored requests' own `total_wall_ms` gives 1,315.683 seconds (21.928 minutes), not 1,531 seconds. Their median/mean/p80 are **270.303 / 655.221 / 603.756 ms**, with failures retained; the 2,000 successful requests give 272.115 / 657.071 / 603.815 ms. These are local exported H100 timings, not the board's private RTX PRO 6000 latency qualification. Doubling even these measured times is only a planning assumption; no two-pass latency has been measured.

6. **The $1.26 figure is stage arithmetic, not a safe launch quote.** At the recorded $0.0495/H100-minute rate, 1,531.344 seconds is $1.2634 including diagnostics, while the forward request-wall sum is $1.0854. A new relabelled pass has extra tokens; setup, build, finalize, retries and a measured headroom/cap are absent from that shorthand. Do not set a spend cap from it.

## Concrete next decision and matched design

The cheapest useful action now is to correct the scope of PR #151 and retain its reproduced negative calibration result. Do not promote or enable order averaging. The operator can then choose whether its remaining diagnostic value warrants a separate capped measurement after a runner/protocol PR is reviewed and merged.

If pursued, pin before measurement: the same 2,008-row DEV manifest, model/revision/template, identity calibration, fixed relabelled-forward and relabelled-reverse maps over every choice question, complete per-option probability export mapped back to original IDs, and a failure-as-wrong convention over the full denominator. Transform requests without `expected`. Run both new full conditions with the same numeric-ID pattern; keep the original-ID baseline as a third, already measured condition. Compare each single relabelled condition, arithmetic probability averaging and normalized geometric/log-prob averaging with the original baseline using the existing harness (B=2000, seed=0), ECE and per-area scores. Time the complete two-pass request. This measures a predeclared two-condition inference method; it still does not isolate pure position from numeric-ID preference. That causal question needs a presentation-order control that preserves scored IDs, or balanced neutral-ID controls.

No runner or scoring flag was implemented, no job was staged or submitted, and no new spend was authorized by this handoff. All launch, review and ownership rules remain applicable.

## Inputs, preservation and commands

The compressed subset hash was verified from the original scratch file. Durable copies outside git now live in `C:/Users/apumu/research/CHLabs/review-artifacts/order-debias-handoff-20261007/`:

- `units-index.json`: SHA256 `f450d000912f5e479eee9943a57ebdb145660396d1fda73fd21c3d7140e57001`, 2,870,250 bytes. Original remains at `C:/Users/apumu/m2fc_uhxb5_out/x_final/out/parent_arm/units-index.json`.
- `index-rows.jsonl.gz`: SHA256 `406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2`, 1,599,061 bytes. Original remains in the Claude scratchpad's `m2fc/` directory.

Full input/source/output hashes are in `reconciliation.json`. These contain DEV data only; no TEST rows were materialized. Interpreter: `C:/Users/apumu/research/CHLabs/ssdm-wt/m0-harness/.venv/Scripts/python.exe` (NumPy 2.5.3, installed `decision_index`, pinned harness kit `87d4650b42b377c0291a89c1f1a879f9b31082bf`). Reproduce sequentially in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$orderPython='C:/Users/apumu/research/CHLabs/ssdm-wt/m0-harness/.venv/Scripts/python.exe'
$orderOriginal='C:/Users/apumu/research/CHLabs/ssdm-wt/order-debias'
$orderEvidence='C:/Users/apumu/research/CHLabs/review-artifacts/order-debias-handoff-20261007'
$orderReceipt='C:/Users/apumu/research/CHLabs/ssdm-wt/order-debias-handoff/receipts/2026-10-07-D-order-debias-handoff'
& $orderPython -X utf8 -I "$orderOriginal/receipts/2026-10-07-D-order-debias/analysis.py" $orderOriginal $orderEvidence "$orderEvidence/index-rows.jsonl.gz" "$orderReceipt/recomputed.json"
& $orderPython -X utf8 -I "$orderReceipt/reconcile.py" $orderOriginal "$orderEvidence/units-index.json" "$orderEvidence/index-rows.jsonl.gz" "$orderReceipt/recomputed.json" "$orderReceipt/reconciliation.json"
```

Both commands passed. `python -X utf8 -m pytest -q tests/test_jobs_lint.py` passed (70 tests, 3.25 s); `git diff --check` passed. The follow-up does not alter serving/training behavior; its executable checks are the exact receipt recomputation and raw diagnostic/schema assertions, rather than model tests.
