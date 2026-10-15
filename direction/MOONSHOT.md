# Keel — a moonshot for aligned decision models and workflows (draft, 2026-10-15)

*Working name: a keel keeps a boat on course against the wind.* Builds on `DIRECTION_MEMO.md`, `GRANTS.md`, and the
evidence in `experiments/` and `research/`.

## 1. The moonshot

**Build the open judgment layer for AI workflows.** These are decision models that cannot be talked off course by
the content they judge, that know when to hand a decision to a human, and that prove both with numbers anyone can
reproduce. The point is to make it safe to give AI agents real authority.

## 2. Why this, why now

- **Decision models are becoming the control plane of AI workflows.** They route requests, approve tool calls, flag
  and moderate content, triage tickets, and monitor agents. Jev, Clef, Microsoft-Decision-1 (MD1), OpenAI Decisions
  and H2O-Lightning all launched between September and October 2026. Frontier labs already run language-model
  monitors over their internal agents.
- **The content a judge reads is often written by the party being judged:** an agent's own output, a customer's
  message, a web page, a tool result. A judgment layer is only as aligned as it is unmanipulable.
- **Today's decision models are easy to manipulate.**
  - Published attacks redirect 61–73% of initially correct decisions across four decision systems.
  - Our probe (`experiments/steering_probe/`): one planted sentence steers 43–63% of correct one-pass decisions, and
    92–97% when the model thinks first, with high confidence.
  - Across 30 models, steerability rises with accuracy (Spearman +0.55).
  - No vendor publishes steering numbers or offers guarantees.
- **The scientific core is the capability–manipulability coupling.** The models that read evidence more carefully
  also obey planted instructions more readily. Breaking that coupling — judgment without obedience — is the moonshot.

## 3. What "done" looks like (12-month targets)

| # | Target | Bar | Today (our probe, stock 4B) |
|---|---|---|---|
| 1 | **Unswayable** | adaptive steer success ≤ 5% (95% upper ≤ 8%) under a preregistered attack budget: static, transfer, probability-guided and white-box tiers | 63% one pass / 97% thinking, static notes |
| 2 | **Still listening** | evidence retention ≥ 90%, and trusted-operator compliance ≥ 95%, so robustness never comes from ignoring the input | not yet measured; channel probe landing |
| 3 | **Capable** | clean decision quality within 2 points of the best open decision model of its size on a sealed set; stretch: half of the thinking gain recovered in one pass | 54 vs 78 local I (H2O) |
| 4 | **Knows when to defer** | cluster-valid certified act-rate ≥ 60% at ≤ 5% acted-on error on unseen data; unfloored, honest probabilities | 0% (nothing certifies) |
| 5 | **Consistent** | reorder/relabel invariance exactly 0 by construction; 8-operator flip rate ≤ 1.3% | 0 bitwise / 3.6–4.1% |
| 6 | **Cheap and open** | one forward pass, ≤ 50 ms p50 on one GPU, open 4B hybrid weights | ≈ 30 ms |
| 7 | **The yardstick** | the Decision Alignment Card benchmark and harness reported by ≥ 3 outside groups or vendors | — |

**Stretch:** long-horizon monitoring. A state-space or hybrid monitor streams a whole agent trajectory of 1M tokens or
more, with measured poisoning resistance.

## 4. The bets

**A. Judgment, not obedience.** Train decision models to treat content as evidence, never as instruction.

- **Data.** Triplets in which the same sentence appears in three roles:
  1. an injection inside untrusted content, which must be ignored;
  2. legitimate evidence that must change the decision;
  3. a trusted operator instruction, which must be followed.

  This is an instruction hierarchy for typed decisions. Open-weight attacker models generate fresh attacks each
  round, and a held-out attacker family is reserved for evaluation.
- **Objective.** Maximise clean accuracy subject to three constraints: steer success ≤ ε, evidence retention ≥ ρ,
  trusted compliance ≥ κ. The constraints are held by the already-built νPI (PID-Lagrangian) controller. This is the
  project's control-theory idea, used for what it is good at: holding constraints during training.
- **Deliberation.** "Think, but don't obey." Distil reasoning traces that name and discount planted instructions,
  because raw thinking follows them 92–97% of the time today.

**B. Channels by construction.** Typed, trust-labelled decision requests: delimited untrusted fields, provenance,
and a trusted operator channel. The bounded-influence head is used only where the trusted fields carry the decision.
Certificates are claimed only after the trusted-only information-gap test shows they are not vacuous.

**C. Deferral as a first-class output.**
- Calibrated act/defer with cluster-valid risk control.
- Escalation on two separate signals: uncertainty, and a manipulation-attempt detector. A separate detector is
  needed because confident attacks evade confidence gates.

**D. Measurement as a product.** The Decision Alignment Card:
- adaptive attack tiers, evidence retention, trusted compliance and capability matching;
- cluster bootstrap, preregistration and independent rescoring;
- an open harness, and incumbents measured on identical units once permitted.

**E. A long-horizon state-space monitor (bet).** Stream long trajectories with constant memory, test HiSPA-style state
poisoning, and use trusted state snapshots for cheap erase-and-check. This is the one place the original state-space
identity could matter. It is an add-on with a kill rule, not the headline.

## 5. Stepping stones (each worth publishing if the moonshot stalls)

| When | Stone | Output |
|---|---|---|
| now | Steering + channel probes | note: "Thinking makes decision models obey planted instructions" |
| month 1–2 | Decision Alignment Card v1 | open benchmark + harness + measurements of open decision models |
| month 2–4 | Keel-4B v0 | first constraint-trained decision model; paired results against its own base and incumbents |
| month 4–6 | Workflow SDK | trust-labelled typed requests, act/defer contracts, System One API compatibility, ControlArena adapter |
| month 6–12 | Keel v1 + external red team | adaptive-attack results, audit, long-horizon monitor bet |

## 6. Phases, gates, kill rules

| Phase | Work | Gate to continue | Kill / pivot |
|---|---|---|---|
| 0 (now → 2 weeks) | Measure the problem: steering, channel discrimination, evidence retention, trusted compliance on stock 4B/9B; Card v0 | The problem survives trust-labelled prompting (if a prompt fixes it at no cost, raise the bar to adaptive attacks) | — |
| 1 (weeks 2–8) | Triplet data + νPI-constrained training pilot on Qwen3.5-4B (two seeds) | ≥ 50% relative cut in steer success on held-out attacks at ≤ 2 points clean cost, evidence retention ≥ 90% | robustness only by ignoring evidence |
| 2 (months 2–6) | Adaptive tiers, deferral, workflow SDK, Card release | ≤ 10% steer success at tier 3 (probability-guided) | defence broken at tier 1–2 with no fix |
| 3 (months 6–12) | Scale, external red team, long-horizon monitor bet | targets in §3 | publish what holds |

## 7. How we work

- **One lane.** The chief of staff runs a single research lane with worker agents. Every change is committed. Each
  run gets a run card (hypothesis, kill rule, cost cap, data hashes). A decision log is kept weekly.
- **Compute.** Per-second H100s on Modal: evaluation runs $1–5 and minutes long; short training cells $3–10.
- **Evidence tiers.** Exploratory runs are light. Confirmatory runs need preregistration, sealed splits, two seeds
  and independent rescoring. Negative results are published.
- **Open by default:** code, benchmark and weights, subject to the owner's approval.

## 8. Funding the moonshot

| Phase | Need | Source |
|---|---|---|
| 0–1 | ≈ $200 compute, agent time | tranche from the owners |
| 1–2 | $50–150k people + compute | Transformative AI Fund, BlueDot, Manifund, Foresight (Oct 31), SFF Speculation; API-credit programmes for comparators |
| 2–3 | institution-scale | only after the Phase-1 gate; some routes (e.g. Coefficient Tailwind) need a nonprofit or public-benefit structure |

## 9. Decisions needed

1. Adopt Keel as the north star, replacing "JevBench Composite first"?
2. The owner's permission to measure incumbents (Jev/Clef/MD1/H2O/Quyet) on the Card and to publish code and
   benchmark.
3. Tranche 1 (≈ $200) for Phases 0–1.
4. Access to `ssdm-data` (training draws, sealed sets, checkpoints), or a rebuild.
5. Who fronts grant applications (founders must author them).

## 10. Risks

| Risk | Mitigation |
|---|---|
| The attacker moves second: adaptive attacks break most published defences | claims capped at measured attack budgets; tiered preregistered attacks; external red team; publish breaks |
| Robustness by ignoring the input | evidence retention and trusted compliance are mandatory, co-equal metrics |
| Capability externality | judgment robustness first; no capability recipes released beyond what the safety claim needs |
| Being scooped | publish stepping stones early (probe note, Card v1) |
| Small team | agents + strict scope; one lane; kill rules |
