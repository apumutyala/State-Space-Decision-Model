# SIM: review capacity is part of the safety contract

## Proposition and boundary

Fail-closed deferral can preserve action containment while denying legitimate service. The useful question is not whether overload exists, but whether admission and scheduling improve legitimate deadline completion at the **same four reviewers and bounded queue storage**, and which authenticated identity boundary that requires.

**Source boundary.** `HostKernel` at base `df6a3f5` owns one FIFO per instance and binds one authenticated principal; there is no implemented cross-instance global queue, reviewer service model, or approval path. The multi-principal service below is a **proposed deployment abstraction (SIM)**, not a measured multi-tenant vulnerability in today's kernel. A concrete single-instance trace separately checks that review-inducing inputs under that instance's principal consume its queue and make later legitimate deferrals `held`. Gate-allowed authorized actions remain executable. No kernel production code is changed.

## Prior work (opened 2026-10-16)

- Luo et al., USENIX Security 2026, [Autonomy Comes with Costs](https://www.usenix.org/conference/usenixsecurity26/presentation/luo): resource lifecycle abuse and agent denial of service already exist as a research topic. Our target is pre-execution review admission, not a new general DoS claim.
- Najt et al., 2026, [SLEIGHT-Bench](https://alignment.anthropic.com/2026/sleight-bench/): false positives can make human review infeasible. This motivates an availability outcome distinct from monitor discrimination.
- Terekhov et al., 2025 / revised March 2026, [Adaptive Attacks on Trusted Monitors](https://arxiv.org/html/2510.09462v2), §2: human auditing has a finite budget and is assumed perfect in their model. We likewise expose the perfect-review assumption rather than presenting simulated zeros as measured human safety.
- This implements [portfolio B3](../proposition_scan/README.md#rank-3--b3-oversight-queue-availability-frontier) and the missing service-model caveat in [the proposition's safety table](../../PROPOSITION.md#4-safety-implementations).

## Frozen method (committed before confirmation)

### Events and assumptions

Each cell has **100 independent schedules**, each with 1,000 legitimate proposals (100,000 legitimate arrivals per cell, not 100,000 independent replications). Given that count, arrival times are independent uniforms in an observation interval: a Poisson process conditional on its count. Eight equally likely benign principals have deferral probabilities `(0.6, 0.6, 0.6, 0.6, 1, 1, 1, 1)`; their mean is 0.8. Attack proposals defer with probability 1. Frozen request-level routing coins, times, service requirements and deadlines are reused across policies. No real model is run or routing accuracy claimed.

Time is in mean review-service units. Four non-preemptive reviewers each have mean service 1. The interval length is `1000 * 0.8 / (4 * rho)`, so `rho` is benign offered **review** utilization, before attack. Service is deterministic (CV 0), exponential (CV 1), or lognormal with mean 1 and CV 2. Both classes draw the same distribution independently of routing. All proposals expire eight time units after arrival. Waiting expiry removes the item; an in-service expiry withholds execution but does not reclaim its reviewer until service finishes. Completion at the deadline counts; events tied in time are ordered service completion, expiry, arrival. Held items are never enqueued or retried. The run drains all service events; there is no end-of-window censoring or steady-state claim.

Review is a **perfect synthetic oracle**: benign cases may execute only after approval before expiry; harmful cases are always rejected. Gate-allowed cases in the generated population are benign. Thus zero harmful effects is a simulator containment invariant under these assumptions, not an estimate of an actual gate or reviewer error rate.

### Equal-resource policies and controls

All four candidate policies have four reviewers and the same maximum **waiting** storage `K`; in-service items do not occupy queue slots. Admission precedes dispatch, matching the kernel even at capacity zero.

| Policy | Admission | Dispatch |
|---|---|---|
| `fifo` | Hold on global queue full | FIFO |
| `fair` | Same global hold-on-full | Round robin over nonempty principal queues; FIFO within principal |
| `cap` | Global bound plus at most `max(1, K // 8)` waiting items per principal | FIFO |
| `reserved` | Reserve 75% of waiting slots against unentitled arrivals; entitled work can use the whole queue | Three reviewers serve entitled work only; one shared reviewer serves FIFO; no preemption |

Reservation does not know request harmfulness. It uses a hypothetical **host-authenticated entitlement**: the eight benign principals have it, attackers do not. It can waste reserved service when entitled work is absent; unused reviewers are not additional resources. Authentication arms are one attacker principal versus 32 freely split attacker principals, with identical arrivals and no extra acquisition charge. Sybils cannot forge the reservation entitlement. A prespecified sensitivity at `rho=.75, CV=1, K=16, sustained` gives all 32 Sybils that entitlement to expose this boundary.

Controls are no attack, zero-time infinite-service review oracle (not resource-matched), always-hold-all (completion zero), and **UNSAFE fail-open**: the FIFO resource path, but overflow and deadline expiry execute rather than hold. The last is a contrast, never a candidate recommendation. Its statistics can be derived exactly from FIFO because release does not change the review workload; in-service review continues after expiry.

### Grid, attack costs and inference

Full grid: `rho = .25, .50, .75, .95, 1.10`; CV `0, 1, 2`; `K = 16, 64`; sustained versus burst arrivals; one authenticated attacker versus 32 Sybils; attack request counts per benign request `r = 0, .5, 1, 2, 4`. This is 600 cells, with shared no-attack controls and shared FIFO/reserved results where identity labels cannot alter their behavior. Sustained attack times are uniform in the same interval. Burst times concentrate the identical count into the first 20% of each of ten equal subintervals (random schedule-wide cyclic phase). The tested attack family knows the policy but does not adapt online. Taking the cheaper of these two schedules is a restricted-family cost, not a globally optimal attacker.

Primary outcome: legitimate completed-before-deadline fraction, denominator all 1,000 legitimate arrivals, including holds. Also report harmful executed effects, p95 queue wait **among legitimate reviews that start** (conditional; cannot stand alone), starvation (fraction of legitimate review-required arrivals never starting review before expiry, including immediate admission holds), benign admission holds, and reviewer work. Always-hold-all starvation is 1; infinite-service starvation is 0. Undefined wait quantiles are null, not zero.

Attack cost to halve service is the first **tested** `r` for which mean completion is at most half the same policy's paired no-attack completion. Use the minimum over the two tested timing families; no interpolation or monotonicity assumption. If absent, report `>4 tested` (right-censored, not proof against all attacks). Costs are request counts, exclude identity-acquisition cost, and also report required number of identities.

Use 4,000 paired schedule bootstrap resamples, seeded separately from schedules. Report ordinary 95% intervals, and Bonferroni 98.333% intervals for the three candidate-minus-FIFO contrasts **within a prespecified cell**; no simultaneous claim across the grid. The anchor cell is `.75, CV=1, K=16, sustained, r=4`, in each identity arm. Material improvement requires adjusted lower bound above +5 percentage points and zero harmful effects. Report no-attack paired changes; a stress-cell gain is not universal Pareto dominance. For cost intervals resample the whole schedule's rate/timing curve together, including no attack; preserve censoring. Check empirical paired SD against B3's planning assumption 0.10, without changing sample size or stopping on favorable results.

A small development run uses disjoint seeds for correctness/runtime checks, not parameter selection. Confirmation seeds and all thresholds are frozen here. Per-schedule data and complete cell summaries remain under `$SSDM_PRIVATE/research-review-queue/`; only compact aggregates enter git. Each run uses CPU, `nice -n 10`, a no-network sandbox, <2 GB memory and a <60-minute deadline. Spend is $0.
