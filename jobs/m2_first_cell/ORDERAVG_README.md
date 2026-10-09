# Order-averaging screen: `orderavg_d13_v3w` (pre-registration)

Written and committed before the job runs. Nothing below may change after results are seen.

## Hypothesis
The readout analysis found position and label priors in Humicroedit, When2Call (D over-predicted), ESCI and
PhishNChips. At test time, score each multiple-choice question under several option orders and average the per-option
probabilities. If those priors are real, this cancels them and raises D13's Index under v3w (D13 = job-hjwpq art-cr9a2,
best step 1600, joint_record head; proxy600 screen 41.07 [37.57, 45.22]). The cost is up to N times the inference, so
the result also moves the cost axis against Clef.

## What runs (one job, three arms, shared forward passes)
- **Rows:** the pre-declared proxy600 rows (`jobs/common/dev_proxy600.json`), all 644 of them. There is no subsampling;
  the 322-row hash half was not needed to stay inside the hold.
- **Readout:** v3w (`jrt-rwkv-g1-v3w` / layout `g1v2`), untempered (identity calibration), exactly as the 41.07 screen
  scored it.
- **Arms** (`decide.debias.OrderAveragedBackend`, id `ssdm.order_avg.v1`):
  - **N=1:** the plain prompt. This is the control and must reproduce about 41.07 within noise. Its rows are
    byte-identical to the base commit's on all 3,730 proxy600 question rows (CPU check with the byte encoder: sha256 81afd5ea... at both 1fa91e7 and this branch) and in the
    golden test (`tests/test_order_avg.py`).
  - **N=2:** identity plus the reversed option order.
  - **N=4:** identity, reversed, and two cyclic shifts. Shifts s in 1..k-2 are ranked by
    `sha256("ssdm.order_avg.v1/cyclic" | qid | option ids | s)`, then s = k-1. Duplicates are skipped, so k = 2 gives
    2 orders and k >= 4 puts 4 different options first.
- **What gets reordered:** only `choice` questions with at least 2 options. A presentation order moves where options
  are LISTED (schema, re-read, code key) and which code letter each option gets. `noul` and `score` questions are
  never reordered and keep the N=1 logits bit for bit.
- **How orders are combined:** the per-option probability is the arithmetic mean over a question's distinct orders.
  Nothing is fitted, on DEV or anywhere else.
- **Shared passes:** each row runs N=4's passes once. N=2 and N=1 reuse passes 0-1 and pass 0 from a per-row cache.
- **Cost per arm:** the measured walls of that arm's own passes, plus its own adapter overhead (what the arm alone
  would cost), and the input tokens summed over its passes. The expected inference cost multiplier vs N=1 on proxy600
  is about 1.99x tokens for N=2 and about 2.84x for N=4, not 2x and 4x, because k = 2 questions have only 2 orders
  (`python -m jobs.m2_first_cell.orderavg passes`). The job reports the measured wall-clock and token multiplier per
  arm, and those measured values are the cost-axis numbers.

## Decision rule (pre-registered; constants in `jobs/m2_first_cell/orderavg.py`, pinned by tests)
- **Pairing:** paired difference per arm, `(N=2) - (N=1)` and `(N=4) - (N=1)`, on the identical 644 proxy600 rows.
  The scorer is frozen `harness.score.decompose` + `harness.bootstrap.compare` (case-cluster bootstrap, B = 2000,
  seed 0, 95 % CI).
- **ADOPT_CANDIDATE:** N=2 or N=4 has a paired 95 % CI lower bound > 0. If both qualify, the cheaper arm (N=2) is the
  candidate unless N=4's point gain exceeds N=2's upper bound.
- **REJECT:** otherwise. The result is logged in DEAD_ENDS with the job id.
- **Confirmation:** an ADOPT_CANDIDATE must be confirmed on DEV-minus-proxy600 (the full-DEV rows outside the proxy,
  same arms, same rule) before the readout changes for any claim run. The proxy result alone never changes the
  champion readout.
- **Sanity gate:** if N=1's Index is not within the 41.07 screen's CI [37.57, 45.22], the job is treated as broken.
  No verdict is drawn from it (verdict INVALID_CONTROL); investigate first.

## Cost
- **Hold:** `max_duration_minutes` 190, quoted at 190 x $0.063 = **$11.97**.
- **Expected:** 119 scoring min. That is 1,481,526 v1 tokens x 0.9004 (v3w/v1) x 2.844 passes x 1.2513 s/1k (the
  screen's measured rate) = 79 min, plus 40 fixed min (load + lineage + the plan pass over every pass). The whole run
  is about 156 min, which is about **$7.72** billed at $0.0495/min (about $9.8 at the quoted rate).
- **Worst case:** (60 + 2 x 190) x 0.0495 = $21.78.
