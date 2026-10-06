# Public-output boundary handoff

The supported public entry point returns either freshly released text or exactly `{"status":"unavailable"}`. It rejects nonpublic host audiences before calling the provider, never trusts `PlannerResult.text`, and recomputes/releases the program before emitting text. It does not serialize internal reasons, programs, traces, probabilities or exception details.

Independent Track L review found that the initial wrapper also accepted private-audience hosts. That was repaired with a public precondition and a zero-provider-call regression. Other tests cover actual scripted planning, malformed/forged planner results, error equivalence, and revocation before check or before release. No learned model was used.

This is a callable boundary with trusted host authority/evidence, not a network service, authentication system, or timing-side-channel guarantee. Shutdown/cancellation and process failure are outside ordinary exception projection. The application must not expose the diagnostic planner API as its public response. The current renderer still receives host/catalog for release validation; a future ReleasedMeaning type can tighten that separation.

Kernel/profile0.1.1 and fixtures are in this branch's committed base (`2dc3e17`). Renderer dependency source is `0375b7b` (#11, now merged); planner/bridge dependency source is `dc1a3fe` (#13). Their exact hashes are pinned by the receipt. Borrowed source files are not included as duplicate implementation changes. Merge required dependencies before this PR. No paid or learned gate is unlocked.
