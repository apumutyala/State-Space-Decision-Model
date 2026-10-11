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

**Cost.** Hold = 240 min x $0.063 = **$15.12**. Worst case (costs.worst_case_usd, build + 2 attempts at the billed rate):
**$26.73** (hard cap $27). Expected ~104 min, ~$5.15.

**Verdict.** None (measurement only). Exit codes: see `scan_job.sh`.

## Mode parity (`job_params_parity.json`, `candidates_parity.json`): the KIT PARITY gate

Pre-registered as EV-PREREG-KIT-PARITY (state PREREG.md). Three published JevBench v1.6.1 open-weights rows, each through
its own documented readout, on the same 231 public items, against the public-231 result their authors published for the
same pinned weights (tolerance 3 top-1 items, the H2O anchor rule; `decision.json` `reference_checks`):

| candidate (pinned revision) | readout | published public-231 |
|---|---|---|
| torchcast-ai/torchcast-decision-12b @ 49107b55 | `torchcast_shim` + `torchcast_shim_double_bos` (third_party/torchcast_shim, vendored unmodified; only the vLLM call replaced) | 203 (48/70/85), per-item results in its v1.0.0 repo |
| Mapika/decider-12b @ 8ac1efa7 (v2) | decider-ai 1.8.0 `Decider.system_one`, eager (wheel pinned by sha256, `--no-deps`) | 198 (48/71/79) |
| Mapika/decider-12b @ 09e9e338 (v1) | same | 200 (48/71/81) |

The board cells themselves (P300 open + 1,200 sealed) cannot be reproduced from the 231 public items (H2O: local I 77.6
vs board 60.0, but 205/231 = its card). **Cost.** Hold 120 min x $0.063 = $7.56. Worst case (costs.worst_case_usd):
**$14.85** (hard cap $15). Expected ~60 min, ~$3.

## Mode h2o_eval (`job_params_h2o_eval.json`, runner `h2o_eval.sh`, code `h2o_eval.py`): EV-PREREG-H2O-HOST eval-only arms

Pre-registered on state PREREG.md (EV-PREREG-H2O-HOST, with EV-PREREG-COMPOSITE-LOCAL, EV-PREREG-SECONDARY-AXES and
EV-PREREG-AREA-TEMP/a). **Weights:** h2oai/h2o-lightning-4b at the lead-pinned revision **0940cbd7** (Apache-2.0); the
job fails closed (exit 10) unless the snapshot commit, the HF-resolved sha and every loaded file (LFS sha256 / git blob
id from the HF API at that revision) match; LICENSE (and NOTICE / README when present; there is no NOTICE file at this
revision) are copied into the artifact. This revision is NOT the v1.1 tag the host scan / kit parity ran (193ad740): new
weights (multimodal Qwen3_5ForConditionalGeneration checkpoint, text path read), a new shim and per-type temperatures
(choice 0.75 / noul 0.8 / score 0.65). The control therefore runs the revision's OWN shim from the pinned snapshot.

| arm | what | role |
|---|---|---|
| `h2o` | the revision's shim (prompt, labels, per-type T, yes/no floor 0.801), one transformers forward, fp32 head | CONTROL |
| `a_T` | scalar T_J fitted by NLL on the held-out jev2a slice, floor kept | (a) contender |
| `a_Ttype` / `a_Tfam` | one T per question type / per jev2a family (>= 200 decisions, else T_J), slice-fitted | (a) secondary |
| `b_native` / `b_shimcal` | raw reading (T 1, no floor) / screen.shim_calibrate semantics on it | (b); on H2O b_shimcal == control algebraically |
| `c_symfork` | sym-fork rows (jrt-h2o-shim format) scored by LM continuation likelihood, no head | (c) G1 probe, NOT a contender |

**Held-out slice:** bschema-draw-jev2a (TRAIN draw, pointer sha256 53416094...), 200 ids per family by smallest
sha256(salt + id), 1,000 ids, ids sha256 **204e9bfe...2a9c** (committed: `h2o_slice_ids.txt`; re-derived in-job from the
pinned records and asserted). Never the public items. **Report:** per arm the jevkit axes / Composite A / Capability;
paired vs the control on the SAME bootstrap draws (B = 1000, seed 16): Δ Composite A 95% CI (primary per contender),
Δ I / C CIs, Δ S / K, Δ top-1 with exact McNemar p, Δ mean NLL on choice items, fraction of resamples with I < 50,
Composite-A quantiles; a per-TYPE breakdown (choice 139 / noul 74 / score 18); the G1 flip report (option_reverse,
option_shuffle, key_relabel, seed 0) for c_symfork and the control. Every report carries the sealed-split caveat.

**Mode h2o_eval_v11** (`job_params_h2o_eval_v11.json`, same runner via `JEV_SCAN_PARAMS`): the identical arms on the
v1.1 tag revision **193ad740** (the revision job-zbtbj / kit parity / the scorecard ran; 4.2058 B text params, no
vision tower, scalar shim T 0.8), pinned by `snapshots/h2o-lightning-4b.sha256` + LICENSE / README blob ids. Which
revision governs the prereg is the lead / evidence decision; both packages are built from the same sha.

**Caps.** setup 14 (download cap 10) + run 30 (rate-guarded) + summarize 8 + reserve 8 = 60 min. **Cost.** Hold
60 x $0.063 = $3.78. Worst case (costs.worst_case_usd): **$11.34** (hard cap $12). Expected ~33 min, ~$2.08.
