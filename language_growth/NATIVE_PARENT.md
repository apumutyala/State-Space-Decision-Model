# Native M2 parent contract

This successor to PR #79 consumes the merged [M2 serving bundle v1](../interfaces/M2_SERVING_BUNDLE_V1.md)
from interface issue #185 / PR #186. The low-level bridge binds reconstructed
models; [qualified parent admission](QUALIFIED_PARENT.md) now supplies strict
Stage 1 admission and native RWKV7 reconstruction above this bridge. Neither
module supplies a qualified artifact or a language result.

```python
pin = NativeParentPin.from_m2_report(reviewed_gate_report)
parent = load_native_parent(local_artifact, reviewed_strict_d_loader, pin=pin)
```

The caller must independently pin/review the report and honor its existing
promotion and safety verdicts. Internal metadata consistency is not artifact
authentication or permission to serve an unqualified checkpoint. In particular,
the first-cell report remains non-promotable.

`from_m2_report` requires `identities.serving_bundle.schema` to be
`ssdm.m2.serving_bundle.v1`. It consumes the complete calibration payload through
`FeatureTemperature.from_dict().assert_servable()`, checks its canonical JSON
identity against `revision.calibration`, and retains its exact parameters,
content hashes and proof. Identity-only calibration summaries are rejected.
Other report checkpoint/served-revision fields, when present, must agree.

`artifact_sha256` must equal `revision.checkpoint.sha256`, the verified local
file hash and the reconstructed backend's `checkpoint.sha256`. The old optional
`checkpoint.artifact_sha256` alias is accepted only when it agrees with `sha256`;
it cannot replace the canonical field. The file hash is checked before and after
the loader call. The caller must keep the resolved artifact immutable throughout
loading and treat the bound backend as inference-only.

Runtime pins are exactly `fork_mode`, `shared_prefix`,
`backend_max_input_tokens`, `enable_experimental_provenance` and
`enable_v0_1_deltas`. `backend_max_input_tokens=None` means no additional backend
cap. The bridge preserves it and does not infer a context limit from training
length, adapter defaults or the checkpoint architecture. Caller/request limits
still apply. Full served revision, calibration payload, runtime and inexpensive
live backend configuration are rechecked before planner/provider use. Native
tuple and JSON-list identity representations compare through canonical JSON.

The loader remains an explicit trusted callable. It must reject missing, extra
or mismatched head/typed/LoRA state, validate frozen-backbone lineage and encoder
identity, and return `LoadedNativeParent` with the exported calibration and runtime
values. Exceptions propagate; there is no stock model, random-state, default
temperature or permissive-load fallback.

The built-in `rwkv7_m2_loader` restores the complete saved head/typed/LoRA
state independently of `Trainer.load`, so it does not depend on open PR #144.
Track D must still supply the exact qualified artifact/report, reviewed Stage 1
attestation and original snapshot/runtime pins. Full-model load and score parity
remain unmeasured; metadata, tiny-tensor and scripted planner tests do not
establish those results.
