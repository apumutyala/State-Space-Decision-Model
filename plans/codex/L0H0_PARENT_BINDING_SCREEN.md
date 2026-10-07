# L0h0 — frozen-parent fact-binding screen

Status: planned, not run. This is a $0 CPU diagnostic inside L0; it cannot qualify L0 or authorize paid training. It implements the phase row added in the conditional-pointer diagnosis PR.

## Hypothesis

The frozen exploratory RWKV7-0.1B parent sometimes selects the wrong record because its fact/operand options say only `record_ref` and `field`, while the user names an entity. Adding the already permitted entity name to each candidate may make the name-to-record relation local to the choice, improving checked read/explain answers. This is an input-representation hypothesis, not an internal-causality claim.

## Conditions and data

- **Control:** current `Fact <record_ref> field <field>` candidate descriptions.
- **Treatment:** `Fact <entity_ref> (record <record_ref>) field <field>` for each fact/operand. The entity name is copied only from the exact `audience_state(...).records` used to enumerate candidates.
- Preserve the complete candidate set, option IDs, order, field and reference bindings, operation choices, relation choices, host policy, evidence values, call budgets and checker/renderer. The model must still choose each option; no resolver or parser may substitute a choice. The compiled `ReadFact` node contains only `record_ref` and `field`.
- Generate **16 synthetic development worlds**: eight read and eight explain, balanced over price/warranty and requested record positions. Each world has a consistent name-renamed/record-order partner, giving 32 paired input renderings per arm. Use fresh names, ranges and independently authored wording not present in the old train/validation scaffolds. Synthetic labels/programs stay evaluator-only.
- Include boundary cases with duplicate visible names, an audience-hidden private record, and an instruction-shaped entity name. All supplied choices remain present. Entity names and descriptions continue through the existing untrusted-data prompt encoding.
- Parent: exact pinned `fla-hub/rwkv7-0.1B-g1` CPU conversion. No training, Index/held-out access, GPU, network or paid resource.

## Measures and exit proof

Record raw provider calls, full request fingerprints/options, programs, checked outputs, rendered text, operation and fact-slot attribution, per-task budgets, direct baseline decision responses on unchanged requests, and separate full-parent tensor hashes before/after. Compare arms with paired world-cluster 95% CIs. A missing/failed call remains in the denominator. Verify the direct path is identical and no unsupported output is emitted; accepted wrong-question answers are still failures.

The direct check here is limited to operation-slot response equality on identical inputs and a full-state hash before/after; it does not establish Track D decision competence retention on its native workload. The only decision this screen may unlock is a larger independent development cohort. Continue to that cohort only if the named arm adds at least two paired worlds whose both members are useful, loses no previously successful paired world, improves exact fact binding, has no operation regression, and the paired useful-answer CI lower bound is above zero. Otherwise fold this exact candidate-description change. A CI crossing zero or fewer complete worlds is inconclusive and does not advance. This does not establish useful capability against a strong unrestricted comparator, parent competence retention on the full decision task distribution, causal semantic fidelity, or an L0 gate.

## Runtime guard and reproducibility

Before loading weights, verify at least 6 GiB available RAM. Run with CUDA hidden, two threads, offline Hugging Face mode, and an owned-child timeout of 1,200 seconds; stop at 3 GiB available RAM. Save the exact source SHA, parent/tokenizer/config hashes, dataset/template hashes, both arm orders, all raw predictions, CI seed and command. The parent’s pinned state hash must match before and after. A code PR with synthetic boundary tests and independent review is required before executing this screen.
