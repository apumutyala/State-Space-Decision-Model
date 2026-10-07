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
| 2026-10-06 | RWKV7-0.1B + SnapshotFork head | 24 CE head-only steps improve free-running read/explain/compare on disjoint worlds | `stack/rwkv7/l0-head-only` | `d8b3be0` | `receipts/2026-10-06-L-rwkv-head-learning-run1/`; fit/transfer: `receipts/2026-10-06-L-rwkv-head-fit-run1/` | Not Index DEV; synthetic useful answers 2/12→1/12, paired Δ−8.33pp [−33.33,+16.67]; head fit 22/64 train, 10/32 dev | no observed gain; slot mapping not strongly fit | folded after 24/96-step checks; retain as baseline, no promotion |
| 2026-10-06 | RWKV7-0.1B + same SnapshotFork head | Exposure-only ablation: 96 fixed CE updates cover all 64 slots three times; all architecture/data/seed/loss settings unchanged | `stack/rwkv7/head-ce-96` | `ce1dcfb` executed | `receipts/2026-10-06-L-rwkv-head-ce-96-partial-run1/` | Not Index; partial: useful 2/12→1/12, no gain vs step 24; train 23/64, dev 10/32 with one missing row counted failed | memory guard at 839.938s; 96 steps complete but only 95/96 final teacher rows; full dev CE null | partial result; fold this fixed CE-only line after two exposure checks; no promotion |
