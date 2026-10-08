# CANDIDATES — every measured variant (negative results included)

One row per measured variant. A row needs a branch (`stack/<family>/<variant>` from 2026-10-06; earlier rows name
the `claude/*` branch they ran from) and a receipt (job id, artifact, or `receipts/` path). Promotion rule and the
current best: `CHAMPION.md`. Enforced by `tests/test_jobs_lint.py::test_champion_and_candidates_ledgers`.

DEV subset = the 2,008-row m1 subset (`jobs/common/dev_subset_m1.json`); CI = 95% bootstrap as reported by the job.

| Date | Family | Hypothesis | Branch | Sha | Receipt | Score (DEV subset) + CI | Verdict | Kept/folded |
|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | Clef-flash (reference) | open-weight baseline to beat | `claude/clef-baseline` | see issue #3 | job-sgihj, LEDGER.md | 55.8 [53.0, 58.5] (full DEV 56.8 [55.4, 58.2]) | reference measured (composite crashed on a truncated native file; scored locally) | kept (reference) |
| 2026-10-06 | Qwen3.5-9B stock | parent before conversion | `claude/q9-one-layer-probe` | `24f164e` | job-5cmat, art-2bybh, issue #68 | 38.3 [35.0, 41.4]; ECE 0.148 | parent baseline | kept (reference) |
| 2026-10-06 | Q9-AF | converting 1 of 8 attention layers (L23) to GDN costs ≤ 1 pt Index | `claude/q9-one-layer-probe` | `24f164e` | job-5cmat, art-2bybh, issue #68 | 39.9 [36.5, 42.9]; paired +1.66 [+0.19, +3.18]; ECE 0.112 | `proceed-all-8` | kept → `stack/q9af/full8` |
| 2026-10-06 | RWKV7-G1k-13.3B stock | attention-free fallback parent readout | `claude/readout-fork` | see issue #3 | job-tkfaj, LEDGER.md | 24.9 [22.4, 27.8] | readout-measured (−31 pts vs Clef-flash) | kept as fallback parent |
| 2026-10-06 | RWKV7 + fla kernel | fla `chunk_rwkv7` matches eager numerics and is faster | `claude/fla-rwkv7-kernel` | see PR #77 | job-jn47u, job-mden7 | n/a (kernel gate) | REJECT_NUMERICS (fla 2–3x chunked's grad error vs fp64) | folded |
| 2026-10-07 | Q9-AF (family q9af, variant full8) | all 8 attention layers of Qwen3.5-9B -> GDN in KL order with the enforced 511e8dd gate gives an attention-free 9B within the pre-registered bounds (jobs/full8/rule.json, be9eed8 + Amendment A1 2cb8cbc) | `stack/q9af/full8` | rule `2cb8cbc` (be9eed8 + A1); code `53b4cf7` | job-2xe5w, art-ipnxz, receipts/2026-10-07-D-full8; PR #94, issue #87 | ran: 6 of 8 layers converted (L3, L27 rejected on hidden_cos) -> see the Q9-H2 row below (non-binding, not attention-free) | `partial-conversion` (job-2xe5w) | folded under rule v1 (pending decision #140) |
| 2026-10-07 | Q9-H2 (family q9af, variant full8 result; HYBRID, NOT attention-free) | full8 run of the row above: the enforced gate accepted 6 of 8 layers (23, 15, 31, 11, 7, 19) and rejected L3 (hidden_cos 0.9487) and L27 (0.9259; step dNLL 0.0419 passed) vs the 0.95 floor, so 2 attention layers kept | `stack/q9af/full8-result` | rule be9eed8 + A1 (full8-rule-v1-A1); code 53b4cf7 | job-2xe5w, art-ipnxz (sha 1cc3c09c…), receipts/2026-10-07-D-full8 | non-binding: Index diff vs stock −26.0 [−28.9, −22.2]; full MMLU drop 12.2 pts [11.4, 13.0]; ppl ratio 1.028 (1k) / 1.040 (4k); MQAR/NIAH pass; KV 8,192 vs 32,768 B/token; $26.34 | `partial-conversion` (not attention-free: cannot satisfy GOAL; Index and MMLU bounds also failed) | folded for Q9-AF under rule v1; ship-hybrid is decision-needed  |
| 2026-10-06 | rwkv7 | first trained cell: a decision head (cache fork, LoRA r64, Clef loss arm) trained 190 min on the 150k set moves RWKV7-13.3B off its stock readout toward Clef-flash (execution config: job-tvuat ADOPT `ckpt_rows_lean`, receipts/2026-10-07-D-cell-step-profile-tvuat) | `claude/m2-first-cell` | PR #74 | job-uhxb5, art-sq6mp; jobs/m2_first_cell; CPU dry run receipts/2026-10-06-D-m2-first-cell-dryrun | n/a (no cell score: parent_g12 and parent_index passed; cell gate rc 1 ~3 min into the cell phase) | FAILED (exit 20; $3.69); root cause under diagnosis on `stack/rwkv7/first-cell-fix` | pending (fix in progress) |

## Registered, not yet measured: proxy head loop on frozen Qwen3.5-9B features (PREMORTEM blocker #1)

Pre-registered in `train/proxy_rule.py` (proxy-rule-v1, commit `e9db899`, plus pre-data amendment A1 `06abf1f`) on branch `stack/q9/proxy-head`. They move into the table above, one row each with its `train.proxy_loop` receipt (results sha256, head sha256, scores with paired CIs vs Clef-flash and stock Q9, families-v1 columns, 0.2.1 and 0.3), once the `jobs/q9_feature_cache` parts have run. No cloud job has been submitted.

- **proxy R00: attn_ref (REFERENCE, not attention-free)**: Clef-like anchor: 2 cross-attention routing layers over the 32-chunk evidence bank + 2 joint self-attention layers over all options of the request + lexical prior, on frozen stock-Q9 features, clef loss. Upper bound on what routing buys; never a candidate
- **proxy R01: mixer (attention-free)**: full attention-free mixer: gated linear-recurrence evidence routing + DeepSets option mixing + field mixer + lexical prior + gated LM-logprob pool (E4) + key class, clef loss, reaches >= 50 on HOLDOUT
- **proxy R02: mixer**: R01 minus the LM answer log-prob (E4): the stock readout's own signal is worth > 0 (techniques/10 §5: stock beats Clef-flash on 6 benchmarks)
- **proxy R03: mixer**: R01 minus key-class embedding: the opaque-key tax (letter 36 vs 55, option_N 51 vs 77, hash 68 vs 99) is partly learnable from the key class
- **proxy R04: mixer**: R01 minus evidence routing: option-conditioned recurrence over the evidence adds over final-state forks
- **proxy R05: mixer**: R01 minus field mixing: joint multi-field decoding helps case-exact slices (Home appliances, SATA, BFCL)
- **proxy R06: mixer**: R01 minus lexical prior
- **proxy R07: mixer**: R01 with plain CE instead of LS-CE+Brier+RPS
- **proxy R08: mixer**: R01 with LS-CE+Brier, no RPS
- **proxy R09: mixer**: R01 on E5 features (opaque option keys rendered as short codes A, B, ...): the rendering closes part of the opaque-key gap
- **proxy R10: mixer**: R01 on 1-layer-converted (L23 GDN) features: head-level conversion gap ~0 (stock-readout gap was +1.66)
- **proxy R11: fork (m2 head)**: m2 FinalStateFork head on cached fork features, clef loss: the current head's frozen-backbone ceiling
- **proxy R12: fork (m2 head)**: R11 with CE
- **proxy R13: fork (m2 head)**: R11 on 1-layer-converted features
- **proxy R14: mixer (A1 control)**: R01 on the stock arm's first 5,000 training rows, 2,000 steps: the matched-data, matched-steps control for R09 / R10
- **proxy R15: fork (A1 control)**: R11 on the stock arm's first 5,000 training rows: the matched control for R13
