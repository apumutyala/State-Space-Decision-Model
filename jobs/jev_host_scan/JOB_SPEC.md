# jobs/jev_host_scan: JevBench host scan (EVAL-ONLY measurement)

**Question.** Which small open-weight model should host our decision system for the new objective: beat the JevBench
v1.6.1 official Composite (harmonic mean of Intelligence, Calibration, Speed, Cost), where the leader H2O-Lightning-4B
v1.1 scores 72.5 (I 60.0 / C 90.0 / S 92.6 / K 60.3)?

**What runs.** One H100. Candidates in `candidates.json` order (priority), one model load at a time; each readout arm
answers the 231 public JevBench v1.2 items (pinned b6b8fff7, `jevkit/data/jevbench_public_pin.json`) one request at a
time, and `python -m jevkit score` turns each arm's predictions into the board's report: open-split Intelligence
(chance-corrected, tier-weighted), Calibration, raw and adjusted (x2 + 0.15 s) p50 / p95 latency on the 72 standard
items, input tokens per decision, Cost under the candidate's declared price class, Composite A/B/C and Capability with
bootstrap CIs (B = 1000, seed 16).

| # | candidate (pinned revision) | readout arms | price class (USD / 1M in) |
|---|---|---|---|
| 1 | h2oai/h2o-lightning-4b @ v1.1 193ad740 (parity anchor) | h2o_shim | Qwen3.5-4B reference 0.03 |
| 2 | Qwen/Qwen3.5-4B @ 851bf6e8 | m0, v3w, h2o_shim | Qwen3.5-4B reference 0.03 |
| 3 | Qwen/Qwen3.5-2B @ 15852e8c | m0, v3w | reference 0.02 |
| 4 | LiquidAI/d1-3B @ 051bcc46 | d1 (its own system_one) | d1 list 0.04 (estimate) |
| 5 | LiquidAI/LFM2.5-2.6B @ 654f9463 | v3w | size class 0.03 |
| 6 | google/gemma-4-E4B-it @ ee0ef602 | v3w | reference 0.02 |
| 7 | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de | v3w | size class 0.03 |
| 8 | RWKV/RWKV7-2.9B-20260805 (G1i) @ aae64511 | v3w | size class 0.03 |
| 9 | RWKV/RWKV7-G1k-13.3B-20260930 @ ce3630b8 (old reference) | v3w | size class 0.05 |

Readouts (`host_scan.py`): `h2o_shim` = H2O's v1.1 shim vendored unmodified (prompt, labels, T 0.8, yes/no floor 0.801),
one transformers forward instead of vLLM, lm head in fp32 per its config; `m0` = jrt-chatml-v1 option log-likelihood
without the prefix fork; `v3w` = `make_v3w(backbone)` (the model's own chat template, single-token code table) scored by
one full forward per question (`decide/backends/v3w_nofork.py`; no fork, job-gen87); `d1` = d1-3B's documented
`system_one`, its code vendored unmodified. Nothing is fitted on the items. Every downloaded file is sha256-checked
against `snapshots/<id>.sha256`; vendored code against its pins.

**Parity anchor.** H2O-Lightning-4B v1.1's model card states 205 / 231 top-1 correct (easy 48, standard 71, hard 86) on
these same public items, with four named near-tie misses. `decision.json` reports our count against it (tolerance 3).
Its published board cells are a different item draw (the v1.6.1 open set P300 + sealed), so only ordering is comparable.

**Caps.** Per candidate: download cap, then the run under `jobs.common.rate_guard` with the candidate's cap (exit 15 on
a measured rate past the cap, 124 at the cap); partial predictions are scored and exported per candidate. A candidate
whose caps no longer fit the scan phase is skipped. Phases: setup 10, scan 220, summarize 5, reserve 5 = 240 min.

**B-PREREG-FLIP mode (branch claude/ev-flip-hs).** Candidates trimmed to H2O-Lightning-4B v1.1 and stock Qwen3.5-4B,
both under the H2O shim; each answers the 231 public items and then the 1,593 pinned jevkit perturbations (`jevkit
perturb make --public --seeds 0`, sha256 655d028a...79d7, shipped as /data/jevbench-perturb/seed0-655d028a/). Base
predictions stay in `<arm>/preds.jsonl` (the jevkit report is unchanged); perturbed ones go to `<arm>/preds_pert.jsonl`,
and `jevkit perturb score` writes `<arm>/flip.json`. Phases: setup 10, scan 46, summarize 5, reserve 5 = 66 min.

**Cost.** Hold = 66 min x $0.063 = **$4.16**. Worst case (costs.worst_case_usd, build + 2 attempts at the billed rate):
**$9.50** (hard cap $10). Expected ~34 min, ~$1.70.

**Verdict.** None (measurement only). Exit codes: see `scan_job.sh`.
