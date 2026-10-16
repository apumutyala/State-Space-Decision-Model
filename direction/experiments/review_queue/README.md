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

Each reported estimate averages schedule-level metrics. In particular, a p95 entry is the **mean of 100 within-schedule p95 waits**, with a bootstrap interval for that mean; it is not the pooled 95th percentile. Review work is mean service units spent per 1,000 benign arrivals, including attacker reviews and work continuing past expiry.

Attack cost to halve service is the first **tested** `r` for which mean completion is at most half the same policy's paired no-attack completion. Use the minimum over the two tested timing families; no interpolation or monotonicity assumption. If absent, report `>4 tested` (right-censored, not proof against all attacks). Costs are request counts, exclude identity-acquisition cost, and also report required number of identities.

Use 4,000 paired schedule bootstrap resamples, seeded separately from schedules. Report ordinary 95% intervals, and Bonferroni 98.333% intervals for the three candidate-minus-FIFO contrasts **within a prespecified cell**; no simultaneous claim across the grid. The anchor cell is `.75, CV=1, K=16, sustained, r=4`, in each identity arm. Material improvement requires adjusted lower bound above +5 percentage points and zero harmful effects. Report no-attack paired changes; a stress-cell gain is not universal Pareto dominance. For cost intervals resample the whole schedule's rate/timing curve together, including no attack; preserve censoring. Check empirical paired SD against B3's planning assumption 0.10, without changing sample size or stopping on favorable results.

A small development run uses disjoint seeds for correctness/runtime checks, not parameter selection. Confirmation seeds and all thresholds are frozen here. Per-schedule data and complete cell summaries remain under `$SSDM_PRIVATE/research-review-queue/`; only compact aggregates enter git. Each run uses CPU, `nice -n 10`, a no-network sandbox, <2 GB memory and a <60-minute deadline. Spend is $0.

## Experiment actually run and results

**[MEAS] Execution record.** All 600 cells completed, with 100 schedules per cell and the prespecified common random numbers. There are 1,500 independent underlying seed blocks (five utilizations × three service laws × 100); storage, timing, rate and identity comparisons share their blocks. Do not count reused controls or individual arrivals as independent replications. The three confirmation jobs took 376.3, 398.6 and 399.2 seconds wall time, with peak RSS 63.9, 65.3 and 65.8 MiB; aggregation took 7.1 seconds. All used `nice -n 10`, CPU-only, `bwrap --unshare-net`, 2-GiB address-space and 3,500-CPU-second caps. Paid spend: **$0**.

**[MEAS] Kernel admission validation.** The real `HostKernel` and the simulator's FIFO admission primitive matched **22/22 outcomes** over capacities 0–3, including an attack-filled same-principal queue, legitimate `held/review_queue_full`, gate-allowed authorized execution while full, and drain-then-defer. Receipt counts/chains and queue lengths matched; each trace had exactly one executor call, the explicitly gate-allowed benign case. Gate outcomes were synthetic/forced: this checks the consequence of deferral, not whether a real injection induces it. The development reusable-code test invocation reported **17 passed**; the integrator should rerun current tests as described below.

### Equal-resource frontier and its identity boundary

**[SIM] Prespecified anchor:** benign review utilization .75, exponential service, 16 waiting slots, sustained four attacker requests per legitimate request. Completion intervals are ordinary paired-schedule-bootstrap 95%; difference intervals are **98.333%**, correcting the three comparisons against FIFO within each identity cell. Differences are percentage points (pp).

| Policy / attacker identities | Legitimate completion before deadline | Difference from FIFO, adjusted interval |
|---|---:|---:|
| FIFO / either identity arm | 37.34% [37.02, 37.65] | reference |
| Fair / one authenticated attacker | 37.86% [37.54, 38.17] | +0.51 pp [0.15, 0.87] |
| Admission cap / one authenticated attacker | **88.73% [88.47, 88.99]** | **+51.39 pp [51.00, 51.77]** |
| Reserved / either arm, entitlement protected | **95.92% [95.60, 96.23]** | **+58.58 pp [58.14, 59.00]** |
| Fair / 32 Sybils | 37.28% [36.95, 37.60] | −0.07 pp [−0.42, 0.28] |
| Admission cap / 32 Sybils | 37.47% [37.12, 37.81] | +0.13 pp [−0.24, 0.50] |

All four fail-closed policies had **zero harmful executed effects in every cell**. That is conditional containment under the perfect-review/all-harm-defers model, not a statistical estimate of real semantic safety. Both cap and reservation pass the anchor's +5-pp material-gain rule against one attacker; only reservation passes against 32 Sybils. Reservation's repeated result across identity arms is expected label invariance, not an extra independent replication.

**No-attack cost matters.** At the same anchor without attack, FIFO and reservation both complete 99.84% [99.77, 99.89]. Cap completes 98.11% [97.96, 98.25], a paired **−1.72 pp [−1.84, −1.61]** (ordinary 95%) loss. Thus cap improves a stress operating point, not universal Pareto dominance. Reservation exactly matches no-attack FIFO throughout this grid because all no-attack work has the entitlement and can use all four reviewers. Its advantage is conditional on a deployable protected class.

**Reservation fails when the boundary fails.** Giving all 32 Sybils the protected entitlement reduces reservation to FIFO: 37.34% [37.02, 37.65], paired difference **0.00 pp [0.00, 0.00]** in the same anchor. A real same-principal compromised workflow has precisely the attribution problem this optimistic separation avoids. Neither a principal name nor the gate's claim that a request is benign supplies the missing boundary.

### Wait, starvation, work and the unsafe contrast

**[SIM] One-attacker anchor; intervals below are ordinary 95%.** Work is mean total service units per 1,000 benign arrivals; full work/hold intervals are in `results.json`.

| Policy | Mean schedule p95 queue wait | Review starvation | Benign admission holds | Review work |
|---|---:|---:|---:|---:|
| FIFO | 5.69 [5.64, 5.76] | 77.39% [77.10, 77.69] | 62.14% | 1,086.2 |
| Fair | 1.30 [1.28, 1.32] | 77.39% [77.08, 77.70] | 62.14% | 1,085.6 |
| Cap | 3.47 [3.41, 3.53] | 13.71% [13.41, 14.00] | 11.01% | 1,076.3 |
| Reserved | 4.06 [3.99, 4.14] | 4.54% [4.21, 4.88] | 3.65% | 1,015.1 |

Fairness makes admitted benign jobs start sooner but barely changes the large number denied admission: its attractive conditional wait statistic is not an availability rescue. Under Sybils, cap starvation is 77.17% [76.87, 77.47], essentially FIFO again.

The infinite-service oracle completes 100%, with zero wait, starvation or harmful effects; always-hold completes 0%, with starvation 100% and undefined p95 wait. **UNSAFE fail-open** completes 100% but executes **312,087 / 400,000 harmful proposals** in this anchor (78.02%), versus zero for FIFO. It is not a candidate frontier improvement. These controls' degenerate completion intervals follow their definitions, not certainty about a real system.

### Attacker cost to halve completion

**[SIM] Same `.75 / CV1 / K16` configuration, taking the cheaper of the prespecified burst and sustained families.** Numbers are attacker requests per benign request, with paired schedule-bootstrap 95% intervals for the first tested crossing.

| Policy | One attacker identity | 32 free Sybil identities |
|---|---:|---:|
| FIFO | 2 [2, 4] | 2 [2, 4] |
| Fair | 4 [4, 4] | 2 [2, 2] |
| Cap | >4 tested [>4, >4] | 2 [2, 4] |
| Reserved, protected entitlement | >4 tested [>4, >4] | >4 tested [>4, >4] |

For FIFO, two means 2,000 attack proposals per 1,000 benign proposals. Its completion at `r=2` is 49.86% [49.46, 50.25], very near half its no-attack baseline; that explains the [2, 4] uncertainty. Fair's coarse-grid crossing at four is **not evidence of a precisely doubled attack cost**. Censored intervals mean no crossing in the tested grid, not a lower bound against every possible attack or rate. Account acquisition costs are excluded.

### Sensitivity and verdict

**[SIM] Service and timing do not reverse the protected-policy gain at the anchor.** Across CV 0/1/2, sustained-attack completion is 38.29/37.34/37.00% for FIFO, 90.92/88.73/85.24% for cap, and 97.51/95.92/92.83% for reservation. For cap, the smallest adjusted lower gain bound in these three cells is +47.54 pp; for reservation it is +54.95 pp. With exponential service and burst timing, FIFO is 82.10% [81.81, 82.38], cap 95.49% [95.27, 95.70], and reservation 99.28% [99.14, 99.40]. Sustained load is much more damaging than this burst family at the same proposal cost.

**Storage is not automatically service.** Raising the anchor's waiting storage to 64 lowers FIFO completion to 24.26% [23.96, 24.56] and reservation to 91.12% [90.19, 92.06], because queued work can outlive its deadline. Fair scheduling does help there: 61.68% [61.37, 61.99], +37.42 pp [37.04, 37.80] adjusted against FIFO. We therefore reject “fair queueing never helps,” while retaining the narrower finding that scheduling alone does not protect saturated admission.

Across utilization .25/.50/.75/.95/1.10 at `CV1/K16/r4/sustained`, reservation completion is 99.98/99.90/95.92/85.87/78.53%: isolation does not eliminate the protected workload's own capacity limit. Across the equal-weight attacked grid, cap's paired gain is +18.66 pp [18.59, 18.72] with one attacker but **−1.17 pp [−1.19, −1.15]** with Sybils (ordinary 95%); this is a descriptive fixed-grid average, not a deployment mixture or a simultaneous claim for every cell. Per-cell candidate paired SDs are at most .0659 versus B3's assumed .10; the five-schedule development anchor SDs were .027–.060. Sample size was not changed.

**[INFERENCE] Worth pursuing as a service/identity contract, not as a novel queue algorithm or an unconditional mitigation.** Trusted reservation improves the equal-resource safety/availability frontier here without a no-attack loss; simple caps buy a large but Sybil-fragile stress improvement and sacrifice some clean throughput. The critical falsifiable deployment constraint is whether the host can identify a genuinely protected class independently of attacker-controlled inputs. Today's one-principal-per-instance kernel does not provide that separation by itself.

## Reproduction

Method freeze: `72fc51a7`; simulator/confirmation freeze: `f02f12db`. Python 3.12.13 and NumPy 2.5.4 from the existing CPU environment; no package was installed. `sim.py` is the reusable event engine and concrete kernel trace, `run.py` the fixed experiment/report driver, and `keel/tests/test_review_queue_sim.py` contains reusable-code tests only. The main integrator should run that file and the existing host tests together.

From this worktree, after `source ~/ssdm/env.sh`, the confirmation job for each `CV` in `0 1 2` is:

```bash
timeout 3550s nice -n 10 prlimit --as=2147483648 --cpu=3500 \
  bwrap --unshare-net --ro-bind / / --bind "$PWD" "$PWD" \
  --bind "$SSDM_PRIVATE/research-review-queue" "$SSDM_PRIVATE/research-review-queue" \
  --dev /dev --proc /proc --tmpfs /tmp --chdir "$PWD" \
  env PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m direction.experiments.review_queue.run --cv "$CV"
```

Create the private output directory first. The same sandbox prefix with `--cv 1 --development` makes a disjoint five-schedule, 100-benign-arrival development run; with `--summarize` it builds compact `results.json` plus private complete cell/cost reports. Confirmation consumes no development output or fitted parameters; reporting reads its manifest only for the development record. The initial development run checked runtime and the admission trace; before the frozen confirmation, the burst phase was changed from a deterministic replicate-index phase to the prespecified random schedule-wide phase. No rate, threshold, service distribution, policy or sample size was selected from outcomes. Consequently a fresh development manifest need not byte-match that historical pilot.

Private artifacts: `confirmation-cv{0,1,2}.npz` (schedule metrics, not independent individual-request observations), corresponding manifests with hashes/resources, `all_cells.jsonl`, and `all_costs.json`. Raw synthetic requests are reproducible from frozen seeds and not committed. `results.json` retains the prespecified anchor, service/utilization/storage panels and identity-breach sensitivity, not every row of the sweep.

## Concrete kernel change proposal — design only

Do **not** turn a full queue or expired review into execution. Keep I1–I5 and the existing executor boundary. A future service integration should:

1. Inject one explicitly scoped `ReviewService` into authenticated host instances rather than accidentally treating each instance's queue as a global service. Its fixed configuration carries total pending capacity, reviewer count, per-security-domain waiting cap, reserved entitlement capacity and policy version. Queue entries need immutable host-bound domain/entitlement, arrival time, expiry, request fingerprint and original receipt ID; none may come from proposal text, tool outputs or model-asserted principal names.
2. Under the existing serialize/receipt/enqueue critical section, reserve admission atomically. Classify `held` reasons separately (`review_queue_full`, `principal_cap`, `reserved_capacity`, `expired`); a failed receipt must roll back the reservation, not enqueue or execute. Cap by the authenticated security domain, aggregating accounts belonging to one tenant; otherwise the Sybil arm applies.
3. Schedule only admitted work. Draining remains **not approval**. A reviewer verdict must bind the exact immutable request/proposal, be authenticated, and pass fresh grant/argument/deadline checks before a separate authorized dispatch with a linked write-ahead receipt. Expiry is a terminal non-executing outcome; a late reviewer verdict cannot resurrect it. Specify idempotency and crash recovery before implementation rather than inferring them from a hash chain.
4. Record admission, review start, review finish, expiry and authenticated domain so real service distributions and queue delays can replace these assumptions. Extend the bounded model check and CPU concurrency tests to joint admission, service starts, expiry, reservation rollback and late verdicts. Test global storage and per-domain accounting under shared workers, not just independent instance locks.

This does not solve the current **same-principal** attack: injected tool outputs and legitimate proposals in one host instance share the same authenticated domain. Fairness/caps cannot distinguish them, and a reservation assigned to that domain is available to both. An additional trusted task/source admission boundary or independently scarce entitlement is a deployment prerequisite, not a label the gate can safely invent.

## What this cannot show; next step and cost

All service benefits are SIM, conditional on fixed synthetic routing, homogeneous mean service across classes, perfect reviewers, non-preemptive service, eight-unit deadlines, no retries, and this finite attack family. There is no measurement of human throughput, a real prompt-injection success rate, safe review approvals, or production availability. A harmful action that the gate does not defer can execute; the simulator explicitly exposes that case in a reusable-code test. Authentication alone does not prove that accounts are scarce or that an entitled account is benign. Long service tails and expiry can change which storage size helps, so larger queues are not assumed better.

The next zero-spend step is a **trusted-identity and service-trace feasibility pilot**, not training: establish which principals/tasks the adapter can actually separate, replay existing frozen gate outcomes, and obtain real review-duration/deadline records before choosing a policy. Planning cost: $0 paid compute, about one engineering day plus 1–2 reviewer hours (an estimate, not observed work). If no trusted boundary separates abusive inputs from protected work, reject per-principal isolation as a remedy and retain the quantified availability limitation; do not pay for a model run to evade that prerequisite.

