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

### Proxy head loop on frozen Qwen3.5-9B features (PREMORTEM blocker #1)

Pre-registered rule `train/proxy_rule.py` (proxy-rule-v1, commit `e9db899`, before any data). Scored with the real harness scorer on SELECT (the 2,008-row DEV subset; used for selection) and, for the chosen attention-free recipe only, on the disjoint 2,014-row HOLDOUT split (verdict). No row below is measured yet: the feature cache (`jobs/q9_feature_cache`, modes stock + extra) has not run (no cloud job submitted). Receipts will be `train.proxy_loop` `receipt.json` files (results sha256, head params, train curve, scores, paired CIs vs Clef-flash and stock Q9, IN/NEW-domain split).

| Date | Family | Hypothesis | Branch | Sha | Receipt | Score (DEV subset) + CI | Verdict | Kept/folded |
|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | proxy R00: attn_ref (REFERENCE, not attention-free) | Clef-like anchor: 2 cross-attention routing layers over the 32-chunk evidence bank + 2 joint self-attention layers over all options of the request + lexical prior, on frozen stock-Q9 features, clef loss. Upper bound on what routing buys; never a candidate | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R01: mixer (attention-free) | full attention-free mixer: gated linear-recurrence evidence routing + DeepSets option mixing + field mixer + lexical prior + gated LM-logprob pool (E4) + key class, clef loss, reaches >= 50 on HOLDOUT | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R02: mixer | R01 minus the LM answer log-prob (E4): the stock readout's own signal is worth > 0 (techniques/10 §5: stock beats Clef-flash on 6 benchmarks) | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R03: mixer | R01 minus key-class embedding: the opaque-key tax (letter 36 vs 55, option_N 51 vs 77, hash 68 vs 99) is partly learnable from the key class | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R04: mixer | R01 minus evidence routing: option-conditioned recurrence over the evidence adds over final-state forks | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R05: mixer | R01 minus field mixing: joint multi-field decoding helps case-exact slices (Home appliances, SATA, BFCL) | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R06: mixer | R01 minus lexical prior | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R07: mixer | R01 with plain CE instead of LS-CE+Brier+RPS | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R08: mixer | R01 with LS-CE+Brier, no RPS | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R09: mixer | R01 on E5 features (opaque option keys rendered as short codes A, B, ...): the rendering closes part of the opaque-key gap | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R10: mixer | R01 on 1-layer-converted (L23 GDN) features: head-level conversion gap ~0 (stock-readout gap was +1.66) | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R11: fork (m2 head) | m2 FinalStateFork head on cached fork features, clef loss: the current head's frozen-backbone ceiling | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R12: fork (m2 head) | R11 with CE | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
| 2026-10-06 | proxy R13: fork (m2 head) | R11 on 1-layer-converted features | `stack/q9/proxy-head` | `e9db899` (rule) | pending: jobs/q9_feature_cache not run | not measured | registered (pre-registered recipe) | pending |
