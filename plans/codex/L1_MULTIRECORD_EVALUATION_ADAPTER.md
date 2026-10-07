# L1 fixture-to-evaluator preparation

2026-10-07. This small $0 CPU adapter is stacked on PR #127's exact
`bd31c9051732ddfa325b17f6ad986c2d08574f49` head. It prepares the committed
synthetic four-record fixtures for the existing free-running development
evaluator; it does not claim L1, learning, causal fidelity or independent labels.

`load_multirecord_evaluation(split)` accepts only the two pinned
`l1_development_*` cohorts. Independent source constants pin both JSONL and
manifest bytes. The loader verifies the manifest, task/world/template identity,
initial audience-visible evidence and checker-valid reference programs before
returning any tasks. A relocated fixture directory must have the same bytes.

Each result retains one initial state containing task wording and public
evidence. All authored slot questions, choice IDs, later requests and targets
are discarded. The existing runner extracts wording and constructs its own
requests, with choice history determined by provider responses. The legacy
`training_only` key holds just kind, host, catalog and reference program for
scoring compatibility; it is evaluator metadata, not training permission.
The L1 split labels remain intact, so the existing L0 training exporter refuses
the results. Empty-prefix authored slot requests are never exported as
compositional or teacher-forced training inputs. No shared evaluator, trainer,
interface, fixture or PR #127 source changes are needed.

Exit proof: both committed cohorts run through the existing planner/scorer with
a scripted evidence oracle; label/program poisoning cannot change model inputs;
later authored requests are ignored; wrong-but-valid, false-relation, reversed
comparison and provider-error controls exercise scoring and denominators. Hash,
identity and evidence tampering fail closed. These are infrastructure controls,
not model measurements. Receipt: `receipts/2026-10-07-L1-multirecord-eval-adapter/`.

Learned L0 prerequisites and future causal interventions, random semantic
controls, adversarial splicing/negation/stale-fact/unit-confusion/injected-authority/
wrong-question gates, strong comparators and parent retention remain separate
requirements. This preparation does not replace them or exact-calculation v0.2
review. No weights, model execution, GPU, paid service, Index or held-out data
are accessed. Independent internal review and Claude cross-review are requested
on the stacked PR; no merge is authorized.
