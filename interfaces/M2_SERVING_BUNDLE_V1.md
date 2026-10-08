# M2 serving export v1

Interface issue: [#185](https://github.com/apumutyala/State-Space-Decision-Model/issues/185).
This additive artifact contract does not
change `ssdm.decide.v0.1`, training, checkpoint promotion or serving behavior.

The M2 gate report adds `identities.serving_bundle`, with schema
`ssdm.m2.serving_bundle.v1`. Existing report fields remain available.

- `artifact_sha256`: the existing checkpoint SHA-256, also retained in
  `revision.checkpoint.sha256`. Consumers must verify the checkpoint bytes before
  strict reconstruction; this metadata is not artifact authentication.
- `revision`: the complete calibrated `Decider.revision()` object, including
  backbone/encoder/layout/template/checkpoint identity and the temperature ID.
- `calibration`: the complete serving `FeatureTemperature.to_dict()` payload,
  including calibration **content hashes**, disjointness and its proof. Rebuild
  it with `FeatureTemperature.from_dict()`, then call `assert_servable()` and
  compare its identity with `revision.calibration`. Identity-only temperature
  summaries are insufficient to reconstruct fitted serving calibration.
- `runtime`: exact `fork_mode`, `shared_prefix`, `backend_max_input_tokens`,
  `enable_experimental_provenance` and `enable_v0_1_deltas`. A null backend token
  limit means no additional backend cap: adapter defaults and caller request
  limits still apply. It must not be replaced with a guessed hard context cap.

The exporter reads existing metadata only. It neither loads weights nor scores a
request, and the report's existing promotion verdict still governs use of the
checkpoint. The first-cell report remains non-promotable.

Track L must reconstruct the **same** frozen backbone, LoRA, head and typed-output
weights through a strict reviewed D loader, preserve the runtime values, bind the
complete served revision, and use its existing checked planner/kernel/renderer.
The export does not provide that loader or qualify the parent. It creates no
parallel model, unchecked text path, calibration generalization claim or language
capability result. PR #79's pending native-parent contract must reconcile its
positive-only context cap and `checkpoint.artifact_sha256` convention with these
existing M2 runtime/checkpoint fields before claiming direct compatibility.
