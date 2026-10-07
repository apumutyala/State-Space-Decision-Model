# L1 whole-world trainer consumer

## L1a — make the frozen split consumable without losing world identity

- **Hypothesis:** the pinned L1 fit, calibration and free-running validation cohorts can enter the existing Track D record consumer without row-level split leakage, label leakage, or loss of world-cluster identity.
- **Deliverable:** a strict loader for the exact PR #134 fit/calibration partition and PR #133 validation cohort; an explicit fit-only conversion to the existing `Item` type; an external `item_id -> world_id` sidecar for clustered metrics; and a reproducible receipt/test proving complete, disjoint membership and source hashes.
- **Exit proof:** reject changed source bytes/manifests; prove 24 fit, 8 calibration and 32 validation worlds with complete records and zero overlap; prove the trainer sees no world/label/manifest metadata in requests; convert fit records only after the whole-world split and recover each converted item's world from its sidecar; focused CPU tests plus repository lint pass at the exact commit.
- **Decision unlocked:** whether the next L1 experiment may begin CPU-only trainer smoke on the fit cohort, with calibration kept out of optimizer updates and the separate validation tasks reserved for free-running evaluation. This phase is data plumbing only; it is not an L1 capability or calibrated-performance result.

## L1b — train a decision-derived semantic-slot model

- **Hypothesis:** recurrent adaptation on 24 complete synthetic worlds can learn operation/fact/ordered-operand/relation selection beyond the current frozen/random baselines without changing the checked kernel or renderer.
- **Deliverable:** a frozen recipe and variant checkpoint trained only on fit worlds; a separately derived calibration artifact from the 8 calibration worlds; free-running evaluation on the 32 validation worlds; parent decision-retention and relation-soundness results; paired world-cluster intervals against an unrestricted open-weight comparator and matched parent baseline.
- **Exit proof:** independent remeasurement from pinned raw predictions; exact source/config/checkpoint/data hashes; paired CIs clustered by base world; no Index or held-out access; all accepted outputs pass the checker/renderer; adversarial wrong-question, negation, stale fact, unit, authority injection and splice fixtures remain in the scored denominator; causal intervention remains separately gated.
- **Decision unlocked:** advance, revise the semantic-slot representation/training objective, or fold this L1 recipe. A synthetic-cohort result cannot establish general language capability or L1 qualification.

## L1c — causal mediation and domain growth

- **Hypothesis:** a registered recurrent semantic intervention changes downstream checked programs in the predicted direction more than norm-matched random and output-only controls, and the learned module can transfer to a newly registered domain without weakening parent competence.
- **Deliverable:** identity-bound capture/edit/replay protocol, pre-registered interventions, exact traces, and a separate new-domain module/evaluation cohort.
- **Exit proof:** paired source-world CIs against both controls, intervention replay and countermand tests, unchanged parent decision checks, complete adversarial soundness and calibration evidence.
- **Decision unlocked:** whether this semantic module supports the next L-level gate or must be folded.

No spend, parent load, model inference, Index or held-out access is part of L1a.
