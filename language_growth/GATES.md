# L0–L4 gate entry point

`python -m language_growth.gates --parent <checkpoint>` reports explicit missing
measurements and exits 2. It loads no weights and opens no heldout data. Add
`--check-public-fixtures` for the cheap authored boundary replay; success there
is boundary implementation evidence only. Tiny CPU checks:

```text
python -m unittest language_growth.tests.test_language_gates language_growth.tests.test_gate_comparator -v
```

The suites follow the landed modules: L0 read/compare; L1 multirecord and exact
difference; L2 conjunction, registered hypothesis and proposed workflow; L3
licensed coordination/quantitative contrast; L4 historical electricity knowledge.
Synthetic worlds randomize record bindings and candidate order. Their labels
are checker-validated authored data, not an independent audit. The historical
electricity table has one source-world, regardless of the requested world count;
its current question set cannot pass the 20-world uncertainty floor.

Prepare a fresh suite without executing a model:

```text
python -m language_growth.gates --parent <checkpoint> --prepare-split development_validation --worlds 32 --out <fresh-suite-root>
```

Preparation supports `development_train`, `development_validation` and explicit
`language_heldout`. Never generate/read a heldout suite while iterating or fit a
model to its evaluator file. Each level has separate `inputs.jsonl`,
`evaluator.jsonl` and a file-hash manifest; no reference programs or labels reach
either model. Heldout preparation does not evaluate it and is not qualification.

Actual development execution additionally requires `--run-development --out
<fresh-results>` and the qualified-parent arguments from
[QUALIFIED_PARENT.md](QUALIFIED_PARENT.md): `--parent-report`,
`--parent-report-sha256`, `--parent-qualification`,
`--parent-qualification-sha256`, and `--parent-snapshot` (or a strict
`--parent-loader module:callable`). The qualified loader reconstructs the exact
native decision head and serving pins. The gate planner uses that provider for
every operation/argument/production choice. It never checks candidates to hide
false alternatives. Parent errors and checker failures remain in denominators.

Run `python -m language_growth.gate_comparator --help` for the local-snapshot,
open-weight comparator exporter. Its greedy decoding is unrestricted: no
grammar/checker/tool/repair or gold program. Both arms receive the same permitted
evidence and host definitions, including false workflow preconditions. Candidate
outputs and raw comparator JSON are scored against the same final semantic
answer contract. Numeric formatting, inverse comparisons and claim order are
normalized; unsupported extras, malformed answers and errors fail. This measures
bounded semantic usefulness, not unrestricted prose fluency. Pass its raw export
with `--comparator-observations`; metadata alone never establishes strength.

The development-only `--dev-authorization`, `--dev-authorization-sha256`,
`--dev-parent-predictions`, `--dev-candidate-predictions` path consumes only
separately pinned DEV prediction exports and computes decision accuracy from
probabilities. Its v1 authorization pins checkpoint and request identities,
option sets/labels/clusters and both export hashes. It never reads original
Decision Index rows, TEST, D1-B or retention cohorts. This accuracy diagnostic
cannot substitute for the frozen Index formula in final qualification.

Final execution is separate and must be explicitly requested with
`--run-evaluation --suite-dir <prepared-root>`. It also requires all parent pins,
`--review <review.json> --review-sha256 <hash>`, raw `--comparator-observations`,
`--comparator-strength`, v2 full-DEV authorization/predictions and
`--dev-scoring-rows`, plus `--intervention-adapter
language_growth.gate_interventions:run_native_interventions` and
`--intervention-registration`. Evidence path arguments may contain `{level}`
when evaluating multiple levels. Omit any artifact and the command fails closed.
`--level L0` executes one level; overall `qualified` can become true only when
all five requested levels are evaluated and pass.

The independently pinned host review has schema `ssdm.language_gate_review.v1`,
`review_id`, `reviewer`, `candidate_checkpoint_sha256`,
`independent_language_holdout: true`, `labels_independently_reviewed: true`, and
`levels` containing exactly the requested names. Each level pins:

```text
manifest_sha256
label_audit_sha256
joint_fixture_audit_sha256
decision_authorization_sha256
comparator_artifact_sha256
comparator_identity_sha256
comparator_strength_evidence_sha256
intervention_registration_sha256
intervention_adapter_source_sha256
```

This is a reviewed host attestation, not reviewer authentication. The manifest
binds the two files and exact task/world counts, `training_allowed:false` and
`split:language_heldout`. Evaluator rows must explicitly divide into `stratum`
`usefulness` and `joint_adversarial`. Rejection labels require
`expected_behavior:reject`, an allowed `expected_status` and exact bounded
`expected_reason`; a provider crash/exhaustion cannot satisfy them. The generated
packs intentionally lack independently reviewed joint labels and cannot pass
this path unchanged.

The comparator-strength artifact has schema
`ssdm.language_comparator_strength.v1`, the reviewed `identity_sha256`,
`open_weight:true`, `unrestricted_generation:true`,
`strong_same_task_comparator:true`, and the supporting `evidence_sha256`.
The review binds its file hash. This assertion never replaces raw same-task
scoring. Final decision preservation uses the unchanged frozen Index scorer and
paired bootstrap on exact, authorized DEV scoring/prediction exports.
The initial runner keeps the qualified parent frozen and binds that actual
loaded identity to both sides of the DEV export authorization. It is not an
adaptation-retention proof for a different checkpoint. Such a run must supply
the independently qualified baseline identity to `decision_index_retention`
alongside the actual loaded candidate; silently relabeling the baseline is
rejected.

The source-pinned intervention adapter accepts `(tasks, parent, preregistration)`
and returns raw `PlannerResult` lists for `semantic`, `sham`, `random` and
`output_only`, plus per-task native capture/registration/unchanged-weight
provenance. The registration binds the parent revision, discovery-only worlds,
native final-hidden capture boundary, semantic direction and fitted output-only
controls from the same discovery pairs. Discovery and evaluation worlds must
be disjoint. The evaluator rescans every program/released output and computes
paired world-cluster 95% CIs against every control; each lower bound must exceed
zero. Native final-hidden edits are not a recurrent-memory or full-mechanism
interpretation claim. Missing directions, candidate-count controls or capture
evidence cannot pass.

Final usefulness needs the paired 95% lower bound of candidate minus strong
comparator to be at least zero; parent decision preservation uses the frozen
Index metric; reviewed joint/adversarial cases must all match. There is no
coverage escape: useful complete answers and failures share the denominator.
Reports preserve raw traces and limitations. No broad no-hallucination claim
follows from any bounded suite.
