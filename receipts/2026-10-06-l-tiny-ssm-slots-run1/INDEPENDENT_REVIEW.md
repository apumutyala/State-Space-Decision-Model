# Independent receipt audit

Verdict: **approve within the documented one-seed development scope**.
The coordinator assigned an independent reviewer who did not implement or train
this model. The completed audit covered frozen receipt commit
`19d6a846cd746f50c25d4540f675e6ecf91370de`; it recomputed result claims from raw
predictions and verified the recorded artifact hashes, without model reruns or
training. This note records the audit outcome relayed by the coordinator.

Verified headline outcomes: initial/trained teacher-forced validation slots
254/800 and 470/800; training diagnostic 9,078/13,332; answer-meaning free-running
scores 56/300 and 119/300; exact-reference program scores 34/300 and 119/300.
The initial difference is 22 checked read explanations accepted by the semantic
scorer. Both arms accept all 300 programs under the kernel and answer 0/100
comparison tasks correctly. Acceptance does not establish answering the request.

Hash proof: all eleven hashes listed in `receipt.json/artifacts_sha256` matched
their preserved artifact bytes; the implementation agent additionally checked
those hashes against committed Git blobs with zero mismatches. Scoped
`.gitattributes` preserves measured bytes. Training pool records hash:
`eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c`;
validation tasks hash:
`d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f`;
unique sampled training rows hash:
`ba796ace8523d8b9fd482b3381fe42319ba56d2778ca325a42b333c479e803b3`.

Nonblocking reporting correction: null kernel reasons were displayed as
`{"None": 0}`. The code and receipt summary now omit null reasons and count
statuses directly from the authoritative raw planner observations. No measured
predictions, weights or artifact hashes changed; no rerun occurred.

No multi-seed, held-out/generalization, safety, retention, or learned L0 gate is
established. This audit does not authorize paid scaling or merge to main.
Claude cross-review is requested separately through the PR.
