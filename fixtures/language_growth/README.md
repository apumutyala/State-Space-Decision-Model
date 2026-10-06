# Language-growth pilot fixtures

These 21 compact cases make the proposed [v0.1 boundary](../../interfaces/DECISION_LANGUAGE_V0.md) reviewable alongside the checker. They are synthetic design fixtures authored in the Codex lane, **not empirical model results, independently authored evaluation labels, or an audit set**. The catalog's accepted premises are trusted only within these toy worlds; they make no real-world product claims.

`manifest.json` explicitly lists every case, stable case/world IDs, and decision-question IDs. `catalog.json` contains nine immutable records, four snapshot worlds, host policies and a small proposed operation registry. Each case contains the question, host-owned inputs, a model proposal, expected outcomes and the reason for each outcome. `unresolved` identifies assumptions still requiring interface agreement. Cases sharing an evidence world must stay together in any future split; these cases are not twenty independent observations.

Every case also contains `decision_only`: finite choices and an evaluator-only target description for the identical question, snapshot and policy. This allows later same-task decision retention comparisons. Never supply target IDs, descriptions, programs or expectations as model input. LG15's contaminated provider input is ineligible for inference or retention scoring until fresh public state is established. LG19's direct decision can succeed within one call even though its proposed program fails; the program failure does not determine the direct-answer label. ASK options belong to this pilot domain and do not change any external benchmark.

Expected outcomes have separate layers:

- `provider_precondition` checks whether the input/state boundary is eligible to run. LG15 must stop here: a public answer selected using private-conditioned state cannot pass merely because its final references are public.
- `kernel_status` concerns evidence-relative support and permission. LG14 and LG17 are deliberately accepted but task-wrong; the host supplies no formal relevance obligation.
- `task_outcome` and `useful_complete_answer` concern answering the user's question. An appropriate ASK is not a complete substantive answer. Report asks, avoidable asks, answerability, provider exclusions and denominators separately when these fixtures become tests.

The cases cover supported read/comparison, false comparison, record splicing, negation, causal overclaim, stale context/policy, currency/time mismatch, conflict, belief escalation, source injection, irrelevant supported output, private-state selection, necessary/avoidable ASK, one bounded repair, exhausted repair budget, and unknown operations and arbitrary text in separate cases.

Numbers are integers or exact decimal strings. LG02 compares `199.90 USD > 149.90 USD`; LG03/LG13 compare integer month values. No currency conversion, implicit temporal conversion, generic negation or new arithmetic operation is licensed. Ordinary comparison arithmetic can use Python `Decimal`; binary floating-point approximations are unnecessary.

Programs use `ssdm.semantic_program.v0.1`: `ReadFact` returns a subject-bound Fact, `Assert.proposition` references its node, and `Compare` consumes explicit `{node, project: "object"}` projections. `Conjoin.speech_acts` and `Ask.obligation_ref/references` follow the agreed fixture conventions; evidence storage remains local packaging. Revoked policy uses `stale_policy`. LG04 catches a literal proposition splice at the input-shape boundary; it does not establish that a future richer binding language is semantically sound. LG20 isolates an unknown operation and LG21 isolates a text assertion using a registered operation. LG13 also exercises the renderer boundary; the packaging validator alone does not test prompt-injection resistance.

LG18 retains its rejected first proposal and permits one repair; final success must not erase the planning error or extra call. LG19 forbids another inference after the one-call budget is spent. Their counts illustrate complete-proposal calls, not measured model calls or an assumed number of typed-choice slots. No latency or cost is measured.

Run the standard-library packaging check from the repository root:

```text
python fixtures/language_growth/validate.py
```

The check verifies JSON shape, unique IDs, manifest coverage, exact numeric representation and declared reference integrity. It deliberately **does not evaluate programs, validate expected truth, enforce audience isolation or certify semantics**. Run `python -m language_growth.evaluate --fixtures fixtures/language_growth --out <new-directory>` for observed checker/provider/renderer results. That report keeps authored task expectations separate; it is not a learned accuracy or coverage measurement.

Code and packaging tests live under fixtures/. Run packaging unit tests with `python -m unittest discover -s fixtures/language_growth -p test_fixture_pack.py -v`. Historical receipts retain their original command paths.
