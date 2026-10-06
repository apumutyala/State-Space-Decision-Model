# A 525 dollar pilot for decision first SSMs

6 October 2026. This is a proposed use of the user's reported SF Compute credits. Account balance, credit expiry, availability and account-specific prices have not been queried. No compute was started. This supplements [the architecture plan](PLAN.md) with a small first campaign; the long-term competitive target remains unchanged.

The useful objective is to test one complete piece of the proposed lineage: **a learned SSM makes typed decisions, exposes editable semantic state, and grows a checked language module from those decisions without losing useful decision capability**. This budget can support a substantive pilot. It does not establish that a frontier competitor or a generally aligned language model can be completed for this amount.

## Compute envelope

The [public rate card](https://autoresearch.sfcompute.com/) lists one H100 at $3.96/hour interactively and $2.97/GPU-hour for batch jobs. At the batch rate, $525 represents about 176.8 GPU-hours before other charges. Reserve $60; the remaining $465 represents about 156.6 batch GPU-hours. Calendar discounts may improve this, but the plan does not depend on them. The live job quote governs spending; illustrative prices elsewhere in the docs differ from the front-page rate card.

Use one GPU per job. Several independent seeds can overlap if available, but an eight-GPU node does not give eight times the work for the same price. Prefer short interactive qualification followed by batch training.

SF credits cover services on that account, not external Astra/Fable/Jev API charges or human label payments. Use public labelled data, independently reviewed examples, and locally served open reference models for this pilot. Preserve external-model comparison as an uncompleted gate unless access is separately available.

## Budget by evidence gate

| Stage | Cap | Approximate batch hours at $2.97 | Deliverable |
| --- | ---: | ---: | --- |
| 0 Runtime and throughput qualification | $15 | Use at most 3.8 interactive hours instead | Working pure-SSM forward/backward, serialization and measured tokens/sec |
| 1 Data pilot and open reference models | $60 | 20.2 | Frozen development/audit tasks, reference predictions, evidence that language learning is needed |
| 2 Matched decision-core experiment | $210 | 70.7 | Three model conditions across three seeds, with capability and intervention results |
| 3 One targeted repair or confirmation | $90 | 30.3 | A causal test of the largest measured failure, or replication on a new domain |
| 4 Language grown from the winning core | $90 | 30.3 | One checked semantic-program and language extension, plus decision-retention evaluation |
| Reserve | $60 | Unallocated | Failures, artifact export, final checks, storage and any extra confirmation |
| Total ceiling | $525 | | |

These are maximum allocations, not estimates that every run will consume them. Unlock each stage only after inspecting the preceding results. A failed decision model does not earn $90 of language polishing. The first commitment should be stages 0 and 1, at most $75.

## The first task

Use a small document/catalog assistant with natural-language questions, authoritative records, and explicit policy. It chooses an answer or operation, tracks a corrected fact across several turns, and can explain its decision. Include product specifications, deadlines, prices/units, eligibility conditions, and missing or contradictory evidence. Public datasets and data licenses are selected and pinned before running.

Construct two clearly separated evaluation strata:

1. **Trusted structured evidence, natural-language request.** Record lookup, comparison and permitted effects can be checked exactly. Misinterpreting the user's question can still produce a valid but irrelevant answer; score relevance and complete task success separately.
2. **Natural-language evidence requiring learned extraction.** Independently reviewed labels measure semantic perception. An extracted belief is not promoted to authoritative truth merely because the model attaches a source span. Its correctness remains an empirical endpoint.

Freeze 1,000–2,000 independent held-out cases, with longer multi-turn episodes as a separately scored group. Keep paraphrase/template families and entities disjoint. Initial training target: 20,000–50,000 varied examples plus an appropriately licensed representation-learning corpus. These are construction targets, not data already present. Inspect a sample and all disputed labels locally before GPU rental.

Include paired cases: changing a necessary fact changes the correct answer; changing a permission changes the permissible effect; adding an instruction inside evidence changes neither authority nor the intended task. Test wrong permitted answers, retractions, stale facts, unit mismatches and unsupported relations. A rule-based parser and an oracle semantic interpreter are required controls. If the simple parser solves all natural-language cases, improve the task before training.

## Stage zero

Qualify one production pure-SSM implementation. Prefer Mamba-3 if its pinned implementation works promptly on the H100; use Mamba-2 for this first pilot if Mamba-3 compatibility would consume the qualification allowance. Preserve the same proposed semantic interfaces. Do not turn this into a kernel-porting project.

Measure a short training run at the actual sequence lengths and batch sizes. Check finite gradients, CPU/reference agreement where practical, chunked versus full processing, recurrent-state reset, and checkpoint resume. Set the next runs' token budget from observed throughput and memory. A model's FLOP estimate alone is not the runtime budget.

## Stage one

Run Clef-flash and one pure pretrained SSM reference on a small frozen task cohort, with a separate development cohort for adaptation. Add larger Clef only if the memory/runtime smoke test fits this stage. All local references receive identical evidence and choices. These results are capability anchors, not a full Decision Index result.

A small pretrained SSM with a direct decision head tests whether the task is tractable with inherited language competence. It is explicitly a reference, not the from-initialization controlled model. If all baselines fail, inspect labels and task construction before attributing the result to architecture.

## Stage two

Train a 50M–100M pure SSM from initialization in three conditions:

| Condition | Purpose |
| --- | --- |
| A Direct decision head | Measures capability without the semantic bottleneck |
| B Explicit semantic interface | Tests whether typed evidence/state and program decisions preserve capability |
| C Same interface plus causal intervention training | Tests whether the intended semantic edits become reliable and resistant to nuisance instructions |

All conditions get the same evidence, training example pairs, token budget, action support and evaluation wrapper. B and C have the proposed authority/source interfaces from initialization. C adds the intervention objective; it does not get extra hidden task answers. Match total and active capacity as closely as possible and report any mismatch. A receives an equally strong ordinary input/policy wrapper, so mask validity alone cannot win the comparison.

Plan nine fits, three seeds per condition. Start with a short development run to catch failures, then lock the configuration. Target 100M–300M training tokens per fit only if the measured runtime fits a seven-hour cap; nine seven-hour batch fits cost about $187.11, leaving $22.89 within this stage. Every matched fit uses the same fixed token count; time caps must not silently create unequal-training comparisons. These are narrow learning experiments, not fully pretrained general models.

Measure useful decision accuracy, complete trajectories, necessary-fact pair accuracy, calibration and coverage, semantic-edit effects and collateral damage. Report both raw predictions and checked deployed decisions. Rejections are not correct answers. Choose any operating thresholds on development data, then audit without retuning.

## Stage three

Spend on one diagnosed weakness. Oracle evidence, oracle previous state and oracle programs distinguish perception, memory, program expressivity and decision errors. Examples of useful repairs are question-conditioned scanning, richer entity binding, explicit RETRACT/CONFLICT updates, or increasing semantic capacity. Preserve the shared recurrence and control contracts while testing the change.

If B/C loses capability, repair that loss before claiming a successful controlled core. If it preserves capability but interventions do not behave as intended, repair semantic fidelity. If both work, use the stage for a fresh-domain confirmation. Do not spread the budget across PID calibration, LLC estimation, JEPA, RL and several uncertainty systems.

## Stage four

Extend the actual surviving checkpoint with a minimal program language: Assert, Compare, Explain and Ask. First render it through checked templates. Then allow the model to choose among several grammar productions with the same denotation. This tests genuine modular growth from decisions into language without an unchecked string channel.

Score full-answer relevance, proof/record support, identity, units, temporal scope, negation, and useful answer coverage. Test semantic fidelity on held-out compositions, and rerun the original decision tasks to detect forgetting. Have oracle programs drive the same realizer to isolate expression failures. Learning a short program from a request is part of the model experiment; the checker must not secretly solve the entire task for it.

The result can support a scoped claim about verified realization and permitted effects. It cannot prove arbitrary source text true or establish universal absence of hallucinations/misalignment. A useful result is that the larger output interface preserves competence and extends the scope of checked meaning.

## Running economically on SF Compute

The [service documentation](https://autoresearch.sfcompute.com/docs) supports job preflight, bounded runtime, checkpoint capture, output export and customer-selected experiments. Its managed researcher incurs separate intelligence charges. Promotional credits normally expire after three weeks; sponsored grants can have different terms. Verify this account's expiry and automatic-top-up status before scheduling.

Recommended execution policy: plain batch jobs selected here, one GPU each, explicit runtime ceilings, no automatic retries without a retry allowance, and no credit purchases. Prepare source and data locally. Save resumable checkpoints and verify committed capture; download final weights, predictions, configs and data hashes. Stop interactive nodes immediately after qualification. Keep all costs, including setup and failures, inside the stage ledger. Export only the minimal project bundle, not the CHLabs directory or session logs.

No SF Compute tool connection or autor CLI was available in this chat's initial inspection. Authentication is an execution prerequisite, not a blocker to planning or preparing the local experiment.

## What would make the credits worthwhile

A strong result is one reproducible checkpoint lineage that learns useful decisions, preserves capability through a semantic interface, responds predictably to semantic edits, and produces a richer checked answer without an opaque bypass. A well-localized failure is also valuable: it tells us whether to improve representation capacity, data, state updates or program expressivity before paying for scale.

The campaign does not need a leaderboard victory to validate a prerequisite, but it must not describe that prerequisite as victory over Clef, Jev, Astra or Fable. The broader benchmark remains a later acceptance gate.
