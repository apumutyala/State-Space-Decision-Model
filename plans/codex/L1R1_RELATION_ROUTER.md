# L1r1 relation-only routing seam

## Hypothesis

A separate protocol-only wrapper can serve a learned relation head for exactly one `slot-004-relation` choice after the three prior operation/left/right proposals, while every other request reaches the stock backend unchanged.

## Implementation and bounds

- Add `language_growth/relation_router.py`; leave `OperationOnlyBackend` and its folded runner unchanged.
- Require the public Track L serialization shape, an empty accepted program, and exactly three proposal objects with slots operation, left, right in that order. Inspect shape and string types only; never task wording, option descriptions, expected answers or evaluator metadata.
- Route `plan` and `score` through the same predicate, preserving the request and deadline callback. Reject state handles/history and extra state fields by falling through to stock.
- Match each supplied envelope context/policy/registry reference and audience to serialized state. The outer Decider checks envelope keys/schema/audience but does not establish these bindings; absent optional bindings remain supported.
- Bind the composite revision to the complete stock/head revisions and a versioned routing contract. Pin revisions at construction and refuse revision drift. Match the head's native-parent provenance model ID (or ordinary model ID for a generic backend) against the stock model ID; the native head's outer model ID is its internal backbone spec name.
- Keep the outer Decider responsible for envelope validation and requested composite-revision pinning; verify that mismatched revision requests never invoke either scorer.

## Exit proof

Fake CPU backends cover exact interception, unchanged fallback for missing/extra/reordered/wrong prefixes, malformed/multi-question/nonrelation/forbidden state, text/label independence, incompatible parent IDs, revision drift, defensive revision copies and unchanged parent/request data. Run only focused tests with no Torch import, checkpoint/model load, GPU, Index/heldout access or spending. Commit only the plan, new wrapper and its tests as Apuroop; do not push.

## Result

`python -m pytest language_growth/tests/test_relation_router.py -q`: **71 passed**. The same focused suite passed with an import blocker for Torch/Transformers/Safetensors and an audit hook rejecting checkpoint/snapshot access. The folded operation cache and runner match `32d9751` unchanged. This proves the protocol seam with fake backends only; native cache/live parity, training and real-parent diagnostics belong to their separate owners/phases.

Review follow-up: the routing contract is now v2 and requires equality for every supplied envelope/state binding. **92 tests passed** under the same no-runtime/checkpoint guards, including malformed binding fallback and a Decider regression demonstrating that mismatched context reaches stock unchanged.
