# Next propositions: zero-cost research round (2026-10-16)

- **Scope:** twelve propositions, each tested with a real experiment on one workstation's CPU, plus a literature
  portfolio of fourteen more ([`proposition_scan`](experiments/proposition_scan/README.md)).
- **Cost:** no model calls, no Modal, $0 of paid compute.
- **Verification:** every result below was re-run from its committed command before this summary, and matched up
  to timing fields.
- **Detail:** each folder's README gives the method, prior work with checked links, uncertainty and limits.
- **Labels:** SIM marks a simulation; everything else is a measurement on real artifacts.

## Scoreboard

| # | Proposition | Experiment | Headline (95% CI) | Verdict | Next step (cost) |
|---|---|---|---|---|---|
| 1 | A trusted planner that sees only the task text can scope agent authority | [`planner_feasibility`](experiments/planner_feasibility/README.md), 97 AgentDojo tasks, offline | Recovers 60.9% (56.6–65.2) of ORACLE-denied attack calls at 94.6% (91.4–97.3) ground-truth utility; 19.6% of tasks need write targets that appear only in tool outputs | **Pursue** as the first restriction layer | Identifier-resolution study ($0); matched 216-episode agent pilot (≈ $4.70, cap $8) |
| 2 | Audit datasets for artifacts before paying to score on them | [`artifact_audit`](experiments/artifact_audit/README.md) | Monitor: style-only static features reach AUROC 0.772 (0.698–0.837); all 41 features 0.951. JevBench: longest option +6.8 pp over chance | **Adopt** as a gate before paid scoring | Done for the monitor set; see row 8 |
| 3 | Statistical resolution as a deliverable | [`eval_noise`](experiments/eval_noise/README.md), [`gate_cert_v0/power.py`](experiments/gate_cert_v0/power.py) | Paired MDD on public-231 is 8.5–19 Intelligence points, so the board's top three are tied. AgentDojo 7/48 → 2/48 has p = 0.079. The v0 certificate design certifies about 9% of the time | **Adopt** for claims and run sizing | Certification v1 (≈ $1); AgentDojo confirmation at about 120 attacked episodes per arm (≈ $15) |
| 4 | A cheap detector trained on synthetic injections gives the gate a second signal | [`detector_generalization`](experiments/detector_generalization/README.md) | Perfect synthetic dev; held out: T1 AUROC 0.748, T3/T3L 0.692, AgentDojo 0.971. Frozen thresholds flagged **every** held-out benign input | **No-go** as built | Template-held-out rebuild ($0) |
| 5 | Feedback control (νPI) beats fixed penalties at holding a training-time constraint | [`dev_controllability`](experiments/dev_controllability/README.md), toy, 8 paired seeds | Decision flips: none 12.9 ± 0.9%, νPI 6.6 ± 0.7%, fixed weight 8 3.4 ± 0.8%. νPI met its monitor target while held-out reliance stayed 0.037 | **No advantage** on this toy | Monitor-refresh replication ($0) before any LLM organism (≈ $24–40) |
| 6 | A readout bias fitted on synthetic items fixes the one-pass yes prior | [`transfer_calibration`](experiments/transfer_calibration/README.md) | Student yes/no −5.4 pp (−15.9 to +2.6); the bias flips sign between synthetic and public items | **Drop** | — |
| 7 | A smaller variance prior rescues the betting certificate | [`gate_cert_v0/power.py`](experiments/gate_cert_v0/power.py) | 72.7% vs 73.3% certification rate | **Drop**; design and sample size are the levers | — |
| 8 | Presentation normalisation repairs the monitor corpus | [`monitor_repair`](experiments/monitor_repair/README.md), 715 programs after dedup | Style-only AUROC 0.772 → 0.768 (paired −0.004, −0.044 to +0.034). Identifier features alone reach 0.748. Static minimal repairs were plausible for 19/20 seeded backdoors | **Normalisation fails**; pursue matched repairs (behaviour × presentation, portfolio B2) | Sandbox validation of the repairs plus independent label review ($0) |
| 9 | Executable evidence worlds give exact labels for retention, injection and policy in one world (portfolio B1) | [`evidence_worlds`](experiments/evidence_worlds/README.md), 200 worlds, 2 independent oracles | 0 oracle disagreements (Wilson 0–1.9%). Heuristics: ignore-evidence 50% and follow-last-directive 75% accuracy, but 0/200 joint success. Binary answers collide on 50% of intended-distinct contrasts | **Pursue** with joint success as the metric; fix the collisions | Elicitation pilot, 800 one-pass decisions (≈ $3.95) plus a human wording audit |
| 10 | Fail-closed review can be flooded; equal-resource admission policies restore service (portfolio B3) | [`review_queue`](experiments/review_queue/README.md), SIM, 600 cells × 100 schedules; admission validated 22/22 against the real kernel | Sustained attack: FIFO completes 37.3% of legitimate work; per-principal caps 88.7%, falling to 37.5% with 32 Sybils; protected reservation 95.9% (+58.6 pp). No fail-closed policy executed a harmful action. FIFO halves at 2 attack requests per benign request | **Pursue** as an identity and service contract | Trusted-identity service-trace pilot (≈ 1 engineering day, $0) |
| 11 | Open-source RLVR answer verifiers accept wrong answers (portfolio C4, unassociated) | [`verifier_adequacy`](experiments/verifier_adequacy/README.md), 10 endpoints, held-out mutation families | Reasoning Gym `number_sorting` gives full credit to sorted lists with every value off by up to 1 (64/64 at ½, 62/64 at 1). Its default scorer pays partial credit for hedged answers. `math-verify` and GSM8K accept `485%` for 485 | **Useful pre-RL gate**; weak standalone novelty | Small RL run testing whether a policy learns the exploit (≈ $3.95) |
| 12 | Randomised audits give valid acted-risk certificates when labels arrive only on reviewed cases (portfolio A3) | [`selective_labels`](experiments/selective_labels/README.md), SIM, 96 cells × 2,000 worlds | IPW null rejection ≤ 1.8% (nominal 5%); naive review-only certification falsely certifies in 96.7–100% of null worlds. At 1% true risk, a 20% audit rate and 1,000 clusters certify 85.7%; audit rates ≤ 5% never certify | **Pursue** the sampling contract | About 384 acted audits plus 1,080 routine reviews (≈ 122 labelling hours, $0 compute) |

## What the round says as a whole

- **Signals fitted to a fixed proxy did not transfer** (rows 4–6, 8). The detector was perfect on synthetic dev and
  useless held out. The readout bias flipped sign between distributions. The controller satisfied its monitor while
  the behaviour persisted. Normalising layout left the identifier style that gives labels away.
  - This strengthens the program's central choice: rest the safety case on structure, plus sealed or held-out
    evaluation, not on a learned signal tuned on synthetic data.
- **Identity is the hinge of the structural layer** (rows 1, 10, 12).
  - Planner scoping fails exactly where a write target appears only in tool output.
  - Review reservation works only if the host can tell a protected principal from Sybils.
  - Acted-risk certificates need audit sampling the attacker cannot steer.
  - Each is a concrete, testable contract for Keel's host rather than a model property.
- **Measure resolution and data validity before spending** (rows 2, 3, 8, 11). Two claims were corrected in place:
  - "thinking beats H2O" is unresolved;
  - the AgentDojo 3–4× is a point estimate.

  The monitor corpus would have produced misleading scores, and the RLVR verifiers would pay for wrong answers.
- **Exact labels are cheap when the world is executable** (row 9). The remaining work is making intervention
  outcomes distinct, not getting labels right.

## Untested propositions from the literature portfolio

The scan ranked fourteen propositions by value × probability of a publishable or fundable result / cost. Ranks 1–4
and 7 are tested above (B1, B2 partly, B3, A3, C4). Still untested, each with a free CPU first experiment described in
the portfolio:

| Rank | ID | Proposition | First paid step |
|---|---|---|---|
| 5 | A1 | Receipt-to-effect fault conformance | ≈ $3.95 |
| 6 | B4 | Budget-normalised attack ranking | ≈ $3.95 |
| 8 | A4 | Monotone-risk resampling protocols | ≈ $5.93 |
| 9 | A5 | Discernment distillation with opposite response constraints | ≈ $7.90 |
| 10 | C1 | Rare-mechanism-preserving reference-model curation | ≈ $7.90 |
| 11 | A6 | Trust-bounded coherent one-pass decisions | ≈ $5.93 |
| 12 | C2 | Cross-domain controllability and collateral effects | ≈ $11.85 |
| 13 | C3 | Minimal hybrid memory under state revisions | ≈ $15.80 |
| 14 | A2 | Cumulative authority conformance | ≈ $7.90 |

## Order of work

Free work first. Paid steps run only with an owner-approved budget (`keel/runtime/budget.py`).

| Order | Step | Cost |
|---|---|---|
| 1 | Matched monitor repairs: sandbox validation and label review (row 8) | $0 |
| 2 | Trusted-identity contract: planner identifier resolution plus the review-reservation service trace (rows 1, 10) | $0 |
| 3 | Evidence worlds with non-colliding outcomes (row 9) | $0 |
| 4 | Certification v1 sized by `power.py` (row 3) | ≈ $1 |
| 5 | Evidence-world elicitation pilot (row 9) | ≈ $3.95 |
| 6 | Planner agent pilot (row 1) | ≈ $4.70, cap $8 |
| 7 | AgentDojo confirmation at about 120 attacked episodes per arm (row 3) | ≈ $15 |
| 8 | Resume Student v1, and the Monitor Card only on a validated matched corpus | ≈ $8–14 / ≈ $4–8 |

**Funding lead (verified):** Foresight AI for Science & Safety Nodes, Coordination and accountability track. The
deadline is **2026-10-31, 23:59 PDT**; typical grants are $30–100k; outputs must be open source; in-person Node
participation is prioritised. Rows 9, 10 and 12 fit its listed "distributed oversight, agent monitoring, and
verification projects".
