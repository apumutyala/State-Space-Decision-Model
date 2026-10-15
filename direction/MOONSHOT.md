# Keel — a moonshot for aligned decision models and workflows (v2, 2026-10-15)

*A keel keeps a boat on course against the wind.* v2 folds in the channel probe (`experiments/channel_probe/`) and the
independent review (`research/keel-review.md`). Git history holds v1.

## 1. The moonshot

**Build the open, auditable judgment protocol for AI workflows.** It has three parts:

- **A reference judge:** a small open decision model that lets *authorized evidence* influence its decisions and
  resists *unauthorized authority transfer* from the content it reads.
- **A narrow verified host component:** only the host assigns authority, untrusted content can never expand the
  operator-authorized action set, and "defer" means no side effect.
- **Replayable receipts:** every act/defer decision can be replayed under a declared threat model.

**Stretch result:** at least **10× fewer executed unauthorized actions** than the strongest preregistered,
same-budget defended baseline, at matched useful task completion and human-review load, in **two distinct
workflows**. **Adoption bar:** three independent groups reproduce and use the protocol.

Empirical robustness, structural guarantees and statistical certificates are separate claims with separate
assumptions.

## 2. Why this, why now — and what we measured

- **Decision models are becoming the control plane of AI workflows.** They route, approve tool calls, flag,
  moderate, triage and monitor agents; Jev, Clef, MD1, OpenAI Decisions and H2O-Lightning launched in Sep–Oct 2026.
  The content a judge reads is often written by the party being judged.
- **Our pilots (static attacks, stock Qwen3.5 4B/9B, public decision items):**

| Condition | Injection steers | Authority claim steers | Robust accuracy | Trusted operator note followed |
|---|---|---|---|---|
| one pass, no trust policy | 42–63% | 53–72% | 17–28% | — |
| thinking, no trust policy | 95–98% | 86–94% | 2–3% | — |
| one pass, trust policy + delimited record | 6–35% | 31–44% | 35–52% | 14–30% |
| **thinking, trust policy + delimited record** | **2.5–6.6%** | 10–25% | **68–83%** | 36–88% |

- **Reading, scoped to these pilots:**
  - Without a trust boundary, thinking makes these models obey a planted note.
  - With an explicit trust policy, thinking becomes discernment, at no clean-accuracy cost.
  - One-pass models cannot use the policy: they follow an authority claim planted in the record more often than the
    same sentence from the trusted operator channel.
  - So the gap Keel closes is **fast judgment that is as channel-aware as slow, policy-guided judgment**, plus a
    workflow that never lets content grant itself authority.
- **What this does not show:** a general law that capability causes manipulability (the 30-model correlation is
  descriptive and selection-confounded). Nor robustness to adaptive attacks, legitimate-evidence retention (no
  counterfactual yet), or agent-workflow outcomes.

## 3. Targets (scoped; comparators, budgets, denominators and populations frozen in preregistration)

| # | Target | Bar | Status |
|---|---|---|---|
| 1 | **Unauthorized-authority resistance** | adaptive steer success ≤ 5% (95% upper ≤ 8%) under a stated attack-access/budget curve and at least one defense-aware held-out attack family; a stretch, not a commitment | static pilots only |
| 2 | **Authorized evidence still counts** | evidence-counterfactual retention ≥ 90% (validated labels); authorized operator *policy* followed ≥ 95%; factual conflicts between operator and evidence → defer, scored separately | not measured |
| 3 | **Capable** | clean quality within 2 points of the best open decision model of its size on a sealed set; stretch: beat H2O's one-pass quality, which needs ≈ 78% of the thinking gain | gap to H2O ≈ 23 I |
| 4 | **Honest deferral** | cluster-valid certificate on the *complete* gate (judge + detector + filters), stated loss, population and δ; full risk–coverage and review-load curves; ≥ 60% certified act-rate as a stretch | shipping code is not cluster-valid yet |
| 5 | **Consistent** | reorder/relabel exactly 0 under stated kernel/tie assumptions; 8-operator flips ≤ 1.3% on our published operators | 0 bitwise / 3.6–4.1% |
| 6 | **Deployable** | full-pipeline cold and warm latency and memory published at fixed lengths; one pass by default; open weights | warm-cache estimates only |
| 7 | **Adopted** | the protocol reproduced and used by three independent groups across two workflows | — |

## 4. The bets

**A. Distill discernment.** The teacher is the model thinking under an explicit trust policy; the student is the
one-pass judge.

- **Data:** validated triplets. The same text appears as (i) an unauthorized instruction inside untrusted content,
  (ii) legitimate evidence carried by untrusted content, which must change the decision, and (iii) an authorized
  operator policy. Evidence counterfactuals are label-checked independently.
- **Training:** constrained by νPI only if it beats fixed penalties.
- **Causal design before any scale-up:**
  - arms: triplet data × {fixed penalty, νPI} × a matched RAISED-style control;
  - same parent, budget and confirmation units; two seeds.
- **Novelty is the measured joint frontier, not hierarchy training itself.** RAISED (Oct 2026) and IH-Challenge
  (2026) already cover parts of this ground.

**B. A narrow verified host component.**
- Provenance propagation; authority assigned only by the authenticated host; "defer ⇒ no side effect".
- An explicit trusted computing base; reuses `decide/provenance.py`.
- The claim is that untrusted content cannot expand the authorized action set, not that every permitted action is
  correct. It builds on CaMeL and Fides design lessons.

**C. Manipulation-aware deferral.**
- Escalation on uncertainty *and* on a separate manipulation signal, because confident attacks evade confidence
  gates.
- Certified on the complete gate with cluster-valid splitting: merge `claude/ev-guarantee-proofs` first.
- Review-queue saturation blocks or holds risky actions; it never auto-approves them.

**D. The Decision Alignment Card as a protocol, not a leaderboard.**
- Paired units, adaptive and defense-aware tiers, evidence-retention controls, matched-budget baselines and
  versioned receipts.
- Confirmation sets are replaced after release.
- It assumes attackers train against everything public.

**E. Long-horizon state-space monitors (conditional).** One matched 128K cost–recall–poisoning experiment, hybrid vs
transformer monitor. It proceeds only if recall holds and poisoning does not collapse it; otherwise the negative is
published and E stops. No constant-total-memory claim is made for attention-containing hybrids.

## 5. Stepping stones

| When | Stone | Output |
|---|---|---|
| now | Steering + channel pilots | note: one-pass judges cannot use trust policies that deliberative judges can (scoped) |
| week 1–2 | Protocol sketch + partner check | two-page contract + one replayable example shown to three prospective evaluators; build only what changes a real deployment decision |
| month 1–2 | Card v1 + adaptive baseline | frozen task semantics, trust-labelled prompted baseline, attack budget curves |
| month 2–4 | Reference judge v0 | distilled one-pass judge with causal ablations; paper |
| month 4–6 | Host component + receipts | verified authority/defer enforcement on one workflow; ControlArena-compatible adapter |
| month 6–12 | Two workflows, external red team, adoption | the 10× stretch tested at matched budget; independent reproductions |

## 6. Phases and gates

| Phase | Work | Gate | Kill / pivot |
|---|---|---|---|
| 0 (≤ 2 weeks) | Merge the Keel base branch (`BRANCHES.md`); evidence-counterfactual set; adaptive attack plan; protocol sketch to three evaluators | baselines and attack budgets frozen; at least one evaluator would use the contract | nobody would use it → narrow to Card + judge |
| 1 (weeks 2–8) | Discernment distillation with causal ablations | defense-aware held-out frontier improves over the trust-prompted and RAISED-style baselines at ≤ 2 points clean cost and ≥ 90% evidence retention | gains only via ignoring evidence, or νPI adds nothing (drop νPI, keep the rest) |
| 2 (months 2–6) | Host component, complete-gate certification, Card v1 release, workflow one | ≤ 10% steer success at the probability-guided tier; valid complete-gate certificate | defense broken at the first two tiers with no fix |
| 3 (months 6–12) | Workflow two, external red team, adoption, conditional E | §1 stretch and adoption bar | publish what holds |

## 7. How we work

One research lane run by the chief of staff with worker agents. Every change is committed and pushed. Each run gets a
run card (hypothesis, kill rule, cost cap, data hashes). Exploratory runs stay light. Confirmatory runs need
preregistration, sealed splits, two seeds and independent rescoring. Negative results are published. Compute is
per-second H100s on Modal.

## 8. Funding

| Phase | Need | Source |
|---|---|---|
| 0–1 | capped $200 exploratory compute, only after data access, baselines, adaptive plan and an all-in estimate are documented; people, annotation and red-teaming budgeted separately | owners |
| 1–2 | $80–150k people + compute + external attack round | safety grants (Transformative AI Fund, BlueDot, Manifund, Foresight, SFF Speculation); API credits |
| 2–3 | institution-scale | after the Phase-1 gate; some routes need a nonprofit/PBC structure |

## 9. Decisions needed

1. Adopt Keel v2 as the north star (instead of JevBench Composite first).
2. The owner's permission to measure incumbents on common units and to publish protocol, code and benchmark.
3. The capped $200 exploratory tranche, under the condition in §8.
4. Access to `ssdm-data`, or a rebuild with a new seal.
5. A named human lead who owns commitments and authors grant applications.

## 10. Risks and safeguards

| Risk | Safeguard |
|---|---|
| The attacker moves second | budget-scoped claims; defense-aware held-out attacks; external red team; published breaks |
| Robustness by ignoring evidence | evidence counterfactuals and operator-policy compliance as co-equal metrics |
| Multi-judge "consensus" without independence | evaluate homogeneous vs heterogeneous judges at equal cost; measure joint failures |
| Actor–judge collusion, forged reasoning | separate tests for shared-family pairs, identity cues and role-forged reasoning; teacher, attacker and adjudicator never the same authority |
| Provenance laundering, unreplayable decisions | only the authenticated host assigns authority; derived evidence keeps labels; full receipts |
| Attackers training against the public Card | held-out task families, mechanisms and an external attacker; versioned confirmation sets |
| Label artefacts | attacks must preserve the task label unless scored as evidence updates; blind adjudication |
| Shift, rare harms, repeated attempts | per-role and episode-level risk; revalidation after changes; no lifetime guarantees |
| Review exhaustion | escalation curves at 2/10/40% review budgets; saturation blocks |
| Capability externality | judgment robustness first; no capability recipes released beyond the safety claim |
