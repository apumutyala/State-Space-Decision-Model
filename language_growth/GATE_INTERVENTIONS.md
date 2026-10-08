# Native fork hidden-axis intervention callback

`run_native_interventions(tasks, parent, preregistration)` in
`gate_interventions.py` runs semantic, sham, random and output-only arms through
the same `gate_planner.plan_gate_task` and pinned native `DecisionModelBackend`.
It returns `arms: {arm: list[PlannerResult]}` in original task order and per-task
`provenance`. The outer gate owns Stage 1 parent admission and evidence/file
bindings. This callback accepts the low-level `NativeParent` as well as
the admitted `QualifiedParent` for synthetic integration tests.

The exact intervention boundary is `native_fork_prehead.v1`: the actual native
backbone's continuation features `HeadInput.fork[1,K,D]`, immediately before the
unchanged trained `FinalStateFork` layer norm and MLP. Capture identity is
returned by `native_capture_identity(parent)` and includes the exact parent
revision, head, hidden dimension, layout, encoder and prompt template. A scoped
head hook edits copies of these features. It neither replaces the trained head
with the LM readout nor changes candidate sets or checker behavior. Experimental
diagnostics identify the arm and registration; the pinned parent revision and
calibration remain the underlying model identity. Capture identity also binds
the exact served configuration and full calibration-payload hash, preserving
the existing calibration content/disjointness proof; no arm refits calibration.

Scope is exactly `final_hidden_semantic_axis_only`. It is not recurrent-memory
editing, a countermand experiment, or proof of overall model interpretation.
The outer gate binds the scope and direction/control evidence through the
machine-readable registration. This adapter introduces no requirement for a
new prose preregistration document or additional user approval.

The four arms are:

- `semantic`: add the registered signed `D`-vector times the fixed positive
  scale to every candidate's final hidden feature.
- `sham`: pass through the identical hook/scorer/planner with zero edit.
- `random`: one seeded random `D`-vector for the entire experiment, normalized
  to the semantic direction's L2 norm, then applied with the same scale/support.
- `output_only`: leave hidden features intact and add a registered fitted
  candidate-count-specific `K`-vector to native head logits, normalized to the
  hidden edit's total L2 norm. These are fitted logit bias controls, not an
  unrestricted state-conditioned readout. The semantic and output controls must
  share exactly the same discovery worlds and discovery-pair hash. Their
  fitting/source evidence is hash pinned; metadata alone cannot prove the
  declared fitting procedure.

The callback receives no labels or target IDs. Its task inputs are the gate's
label-free task shape; extra evaluator fields are rejected. Discovery worlds
must be disjoint from every supplied evaluation world. Every encountered
candidate count must have a fitted output-only direction. Missing controls,
failed edits and unsupported provider calls remain failed denominator entries;
they never become unedited successes. Every successful provider call must
traverse exactly one full-domain native head capture. Realized edit norms are
checked within relative tolerance `1e-4`, with actual norms recorded.
The callback also returns global and per-task `instrumentation_valid` flags.
The outer gate must refuse intervention qualification when any flag is false;
failed control instrumentation cannot establish semantic superiority. Failures
remain in the report and denominator even when this eligibility condition fails.

The frozen machine registration JSON has exactly:

```text
schema: ssdm.language_intervention_preregistration.v1
parent_revision: exact served parent revision
capture_boundary: native_fork_prehead.v1
scope: final_hidden_semantic_axis_only
capture_identity_sha256: gate_suites.digest(native_capture_identity(parent))
discovery_world_ids: nonempty unique list of train-only discovery worlds
discovery_pairs_sha256: pinned source-pair identity
directions: {path: absolute local artifact path, sha256: full file SHA-256}
scale: fixed finite positive number
random_seed: integer in [0, 2**63)
output_control: fitted_candidate_count_logit_direction.v1
registration_sha256: registration_sha256(the full registration object)
```

The registration digest excludes its own hash field and uses sorted compact
JSON, ASCII escaping, finite numbers, UTF-8/ASCII bytes and SHA-256. The outer
gate separately pins the full registration file SHA and source/fitting evidence
manifest. Keep the direction artifact immutable throughout loading.

The direction file is a restricted PyTorch weights-only artifact, at most
16 MiB, with exactly these keys:

```text
schema: ssdm.native_fork_direction.v1
parent_revision, capture_identity_sha256, capture_boundary,
discovery_world_ids, discovery_pairs_sha256: identical to registration
source_split: discovery_train_only
output_only_fit_evidence_sha256: pinned fitting evidence identity
direction: finite nonzero CPU float32 tensor of shape [D]
output_only_directions: {"K": finite nonzero CPU float32 tensor of shape [K]}
```

Candidate counts are canonical decimal strings from `1` to `255`. For counts
above one, output directions must change relative logits. No random-logit or
unfitted fallback is available. Use the original canonical candidate order when
fitting these controls. Direction discovery/fitting remains actual future model
work; no trained directions or efficacy are supplied here.

Full parent `state_dict` bytes, including backbone, LoRA, head, typed output and
persistent buffers, are streamed once before the entire matched run and once
after; unchanged hashes are required and reused in per-task provenance. Per-call
records bind exact requests, request fingerprints, candidate IDs, hidden/logit
capture hashes, shapes, dtype and actual edit norms. Hooks have one owner and
are removed on all exit paths. The outer gate pins the executing source file;
recorded traces alone do not establish source or fitting provenance.

Tiny fake-state CPU tests exercise the actual `DecisionModel`/`FinalStateFork`,
native adapter, calibration and ordinary free-running L0 planner. They establish
instrumentation and failure behavior only. The real qualified checkpoint,
discovery-fitted direction/control artifact with source/fitting evidence, and an
authorized full-model run remain external prerequisites. No neural semantic success or
L0–L4 qualification is established by these tests.
