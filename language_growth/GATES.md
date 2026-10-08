# L0–L4 gate runner

`python -m language_growth.gates --parent <checkpoint>` reports missing evidence
without loading weights or opening heldout data. Exit 2 means unqualified.
`--check-public-fixtures` runs only the authored public boundary pack.

The suites follow the landed modules: L0 read/compare; L1 multirecord and exact
difference; L2 conjunction, registered hypothesis and proposed workflow; L3
licensed coordination/quantitative contrast; L4 historical electricity knowledge.
Every operation, argument, workflow step and production is chosen by the actual
pinned parent provider. False candidates remain offered; checking never chooses
the answer. Errors and incomplete answers stay in denominators.

Prepare separate model inputs and evaluator references, including joint source
injection, revoked-policy and private-state cases:

```text
python -m language_growth.gates --parent <checkpoint> --prepare-split development_validation --worlds 32 --out <fresh-suite-root>
```

Splits are `development_train`, `development_validation` and explicit
`language_heldout`. Heldout preparation records development content/world
inventories; evaluation rejects overlap. It never generates heldout tasks during
evaluation. Authored/checker-validated label provenance remains visible and is
not called an independent label audit. The electricity table is one shared
source-world; repeating its questions cannot meet the 20-world CI floor.

Development execution uses `--run-development --out <fresh-results>` plus the
[qualified parent](QUALIFIED_PARENT.md) arguments: `--parent-report`,
`--parent-report-sha256`, `--parent-qualification`,
`--parent-qualification-sha256`, and `--parent-snapshot` (or a strict
`--parent-loader module:callable`). Stage 1 qualification remains necessary.
No new review, approval or prose preregistration document is required.

The open-weight comparator exporter is `python -m
language_growth.gate_comparator --help`. It runs the selected local snapshot with
unrestricted greedy generation and preserves raw responses, identity and file
hashes. It gets the same permitted evidence and host definitions from the first
call. Both arms use the same final semantic answer schema. Numeric formatting,
inverse comparisons and claim order are normalized; unsupported additions and
malformed JSON fail. Transport failures/missing raw responses block final
comparison eligibility. This measures bounded semantic usefulness, not general
prose fluency. Pass its export as `--comparator-observations`.

Final execution uses `--run-evaluation --suite-dir <prepared-root>` with the same
parent pins, `--comparator-observations`, `--dev-authorization`,
`--dev-parent-predictions`, `--dev-candidate-predictions`, `--dev-scoring-rows`,
`--intervention-adapter language_growth.gate_interventions:run_native_interventions`
and `--intervention-registration`. Evidence paths may contain `{level}`.
Select one level with `--level L0`; overall `qualified` requires all five levels.

Before final execution, pass the existing artifact locations to
`--prepare-evidence --out <fresh-pin-directory>`. This writes
`evidence-manifest.json` with exact suite, comparator, DEV and intervention
artifact/source hashes, selected open-weight model/revision/source and the
primary metric. Use `--evidence-manifest <file>
--evidence-manifest-sha256 <hash>` during execution. This is machine-readable
artifact binding, not a human approval. An optional
`--comparator-min-accuracy` objective is recorded only when explicitly supplied;
no universal comparator accuracy floor is invented.

The final gates require paired 95% CI lower(candidate minus selected strong
comparator) >= 0; no regression in the unchanged frozen Decision Index formula
on complete authorized identical DEV exports; all joint/adversarial cases
correct; and semantic interventions with paired lower bounds > 0 against sham,
norm-matched random and discovery-fitted output-only controls. All confidence
intervals preserve source-world clusters. Invalid neural instrumentation,
missing fitted controls or incomplete native captures cannot pass, even when
raw differences look positive. Discovery and evaluation worlds must be disjoint.
The direction registration is executable frozen data, not an approval document.

For a direction discovered at selected semantic slots, use the v2 native
registration and explicit `target_slots`; see [learning prerequisites](LEARNING_PREREQUISITES.md).
The evaluator independently checks fixed slot applicability, unchanged captures
outside that scope and at least one valid targeted capture in every arm. All
tasks remain in paired denominators. A relation-only K=3 fit therefore need not
invent controls for unedited operation/fact slots. It supplies no evidence of
control over semantic slots outside its registered scope.

DEV exports contain only separately hash-authorized scorer metadata/predictions;
no original dataset, TEST, D1-B or retention holdout loader is called. The
initial language runner keeps the qualified parent frozen and uses that actual
identity for both sides. It is not adaptation-retention proof for a different
checkpoint; that requires supplying the actual independently qualified baseline
identity to `decision_index_retention` alongside the loaded candidate.

Completed baseline raw traces are saved to `<out>/<level>/candidate.json` before
neural interventions, and blocked/partial reports preserve completed evidence.
Native final-hidden edits are a narrow semantic-axis experiment, not proof of
recurrent-memory mediation or full internal interpretation. No global
no-hallucination claim follows from these bounded tasks.

Tiny public CPU tests:

```text
python -m unittest language_growth.tests.test_language_gates language_growth.tests.test_gate_evidence language_growth.tests.test_gate_comparator -v
```
