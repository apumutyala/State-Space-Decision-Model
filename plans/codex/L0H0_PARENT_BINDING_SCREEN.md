# L0h0 — frozen-parent fact-binding screen

Status: run1 incomplete, stopped safely at `memory_guard` after 74.063 seconds; 12 requests/11 responses in direct-before and zero efficacy worlds. No efficacy result. Preserved receipt: `receipts/2026-10-07-L-fact-name-screen-run1/`. A follow-up to PR #120 prepares a narrower measurement protocol and higher admission reserve; no retry has run. This is a $0 CPU diagnostic inside L0; it cannot qualify L0 or authorize paid training.

## Hypothesis

The frozen exploratory RWKV7-0.1B parent sometimes selects the wrong record because its fact/operand options say only `record_ref` and `field`, while the user names an entity. Adding the already permitted entity name to each candidate may make the name-to-record relation local to the choice, improving checked read/explain answers. This is an input-representation hypothesis, not an internal-causality claim.

## Conditions and data

- **Control:** current `Fact <record_ref> field <field>` candidate descriptions.
- **Treatment:** `Fact <entity_ref> (record <record_ref>) field <field>` for each fact/operand. The entity name is copied only from the exact `audience_state(...).records` used to enumerate candidates.
- Preserve the complete candidate set, option IDs, order, field and reference bindings, operation choices, relation choices, host policy, evidence values, call budgets and checker/renderer. The model must still choose each option; no resolver or parser may substitute a choice. The compiled `ReadFact` node contains only `record_ref` and `field`.
- Generate **16 synthetic development worlds**: eight read and eight explain, balanced over price/warranty and requested record positions. Each world has a consistent name-renamed/record-order partner, giving 32 paired input renderings per arm. Use fresh names, ranges and independently authored wording not present in the old train/validation scaffolds. Synthetic labels/programs stay evaluator-only.
- Duplicate visible names, an audience-hidden private record, and an instruction-shaped entity name are mandatory separate deterministic/scripted safety fixtures, outside the 16-world efficacy cohort and its primary estimate. Ambiguous or hostile names change the question distribution and can confound the binding metric. These fixtures check distinct record bindings, audience filtering and untrusted-data encoding; all permitted choices remain present and the model-efficacy denominator remains the fixed 32 ordinary-name renderings per arm.
- Parent: exact pinned `fla-hub/rwkv7-0.1B-g1` CPU conversion. No training, Index/held-out access, GPU, network or paid resource.

## Measures and exit proof

Record raw provider calls, full request fingerprints/options, programs, checked outputs, rendered text, operation and fact-slot attribution, per-task budgets, direct baseline decision responses on unchanged requests, and separate full-parent tensor hashes before/after. Compare arms with paired world-cluster 95% CIs. A missing/failed call remains in the denominator. Verify the direct path is identical and no unsupported output is emitted; accepted wrong-question answers are still failures.

After run1, remove only the 64 redundant pre/post direct inference bookends. Use the already-paired initial-operation request and decision equality for **all 32 members across arms**, removing request IDs and volatile latency only; compare answers, raw/served probabilities, calibration and status exactly and require `ok` responses. Retain the pre/post full-parent state hash. All 16 worlds, 32 members per arm, primary paired estimates, mandatory boundary-fixture coverage and statistical gates are unchanged. This amended check does not establish repeated temporal response equality or Track D decision competence retention on its native workload. The only decision this screen may unlock is a larger independent development cohort. Continue to that cohort only if the named arm adds at least two paired worlds whose both members are useful, loses no previously successful paired world, improves exact fact binding, has no operation regression, and the paired useful-answer CI lower bound is above zero. Otherwise fold this exact candidate-description change. A CI crossing zero or fewer complete worlds is inconclusive and does not advance. This does not establish useful capability against a strong unrestricted comparator, parent competence retention on the full decision task distribution, causal semantic fidelity, or an L0 gate.

## Runtime guard and reproducibility

Before importing Torch or loading weights, verify at least **9 GiB available RAM**, a conservative operational reserve after run1, not a model-only RAM estimate. Run with CUDA hidden, two threads, offline Hugging Face mode, and an owned-child timeout of 1,200 seconds; stop at 3 GiB available RAM. Save the exact source SHA, parent/tokenizer/config hashes, dataset/template hashes, both arm orders, all raw predictions, CI seed and command. The parent’s pinned state hash must match before and after. A code PR with synthetic boundary tests and independent review is required before executing this screen.

Run1 began at 7,920,824,320 available bytes and stopped at 3,202,723,840 bytes; available-memory telemetry does not isolate model RSS. The unchanged 192-request/384-forward protocol projected 1,255.5 seconds before longer fact prompts/final checks, above the cap; reducing max_batch_rows to 1 also projected over cap. The amended nominal 128-request/256-forward protocol projects 837.0 seconds by proportional arithmetic only, with prompt and final-check uncertainty still binding. No runtime completion is claimed. Every terminal abort must save an explicit `incomplete` receipt with stop reason and preserved hashes, without metrics or eligibility. The new registration is in `receipts/2026-10-07-L-fact-name-screen-retry-prepared/`; original registrations/raw files are preserved unchanged.

The mandatory existing boundary tests in `language_growth/tests/test_fact_candidate_style.py` are:

- `test_duplicate_visible_names_stay_distinct_record_bound_candidates`
- `test_private_record_never_enters_entity_named_candidate_options`
- `test_injection_shaped_name_remains_untrusted_and_program_contains_only_refs`

The required focused validation includes these scripted fixtures, the complete paired-data checks, and the runner's synthetic/no-load checks:

```powershell
& C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_rwkv_fact_name_screen.py language_growth/tests/test_fact_binding_screen.py language_growth/tests/test_fact_candidate_style.py
```
