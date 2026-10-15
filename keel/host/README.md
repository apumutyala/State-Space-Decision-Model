# Keel host enforcement and receipts

This slice implements a CPU-only host authority boundary, bounded review queue,
write-ahead decision receipts, deterministic prompt replay, and an executable
bounded safety/refinement checker. It does not call a model, certify a gate,
approve human reviews, or implement downstream action semantics.

## API and integration

- `Authority(policy, role_actions, principal_roles)` snapshots trusted configuration.
  For principal `p`, its authorized set is **policy.authorized_actions ∩ the union
  of grants for p's host-assigned roles**. Unknown roles/principals grant nothing.
  Neither policy prose, model answers, citations nor segment text adds grants.
- `HostKernel(authority, principal=..., executor=..., receipt_log=...,
  review_capacity=..., model_id=..., model_revision=..., gate_version=...,
  calibration_id=...)` binds an authenticated principal. A proposal cannot select
  another principal merely by naming one. Identifiers are required caller-supplied
  strings; a calibration ID is a reference, not proof of calibration.
- Construct segments through `kernel.inputs.untrusted(...)`, `.operator_note(...)`,
  `.metadata(...)` and `.derive(...)`. Only authenticated host code may call the
  trusted constructors. They snapshot input content; the registry rejects foreign
  `Segment` instances and subsequently modified contents. Derivation calls
  `keel.protocol.derive`; mixed/untrusted/no-parent derivations cannot gain trust.
  Include all parents in the request. For an all-trusted derivation, explicitly
  choose `kind="metadata"` or `kind="operator_note"`.
- `kernel.request(id, segments, questions)` supplies the pinned policy. Do not
  deserialize an attacker-provided policy/role/trust map into trusted host inputs.
  The host must authenticate policy, questions, source labels and its principal.
  The segment registry is session-local and retains issued segments for that session.
- `kernel.process(request, ActionProposal(action, args, proposer, cited_segment_ids),
  DecisionResult(...), prompts=None, attack_tag=None)` returns
  `Outcome(status, reason, receipt_id)`. Status is `executed`, `deferred`, `held`
  or `denied`. Invalid policy/principal/citations/result binding/probabilities are
  denied. Any defer reason vetoes acting even if the result's `act` flag is true.
  Deferral reserves queue capacity atomically; saturation holds. Authorized acts
  that do not require review can still dispatch while the review queue is full.
- `review_queue.pop()` only removes a pending review. It is not an approval and
  never calls the executor. Human review needs a separately authenticated new
  decision; this package does not silently replay pending actions.

The gate integration boundary is the existing `keel.protocol.DecisionResult`.
No action-to-answer semantic mapping is assumed: a workflow adapter must decide
which proposal the answers justify. This kernel checks authority, not that mapping
or the safety of action arguments. The executor receives a snapshot proposal.

## Receipt semantics and replay

`Receipt.capture(...)` snapshots the request fingerprint, prompt manifest, pinned
model/revision, rendering/trust versions, caller's gate/calibration identifiers,
segment trust/kind/source/content hashes and parent IDs, optional attack tag,
per-question probabilities/decisions, outcome/reasons, UTC time, proposal, and
host-computed authorization set. Outcomes in the receipt are `act/defer/hold/deny`.
Content hashes use canonical JSON (including JSON quotes for string content);
prompt hashes use the exact UTF-8 prompt bytes.

`ReceiptLog(path).append(receipt)` seals `receipt_id = SHA256(canonical payload
excluding receipt_id)` and sets `prev_hash` to the previous receipt ID (all-zero
hash at genesis). JSONL is strictly canonical. Writers use a POSIX advisory lock,
append, fsync, and atomically replace/fsync a separate `.checkpoint` count/tip.
`verify()` checks every line and the checkpoint. It detects edits, internal or
suffix deletions, whole-log truncation, reordering and noncanonical formatting.
It refuses to append to an inconsistent log. `verify(expected=checkpoint)` also
checks an externally pinned tip. Independently protect the checkpoint: anyone
able to rewrite both log and tip can forge a fresh history. These are hashes,
not signatures. `ReceiptLog()` without a path is the same JSONL/hash implementation
in memory, with **no durability**; only use it when durable audit is unnecessary.

A receipt is durably appended **before** enqueue/dispatch. Its `act` outcome means
**authorized dispatch intent**, not downstream completion. An executor exception
raises `ExecutionError` with the already recorded receipt ID; partial effects may
have occurred. It is never mislabeled as defer/deny and is never automatically
retried. Process termination between receipt and dispatch can leave an unexecuted
intent. Termination between log and checkpoint fsync fails closed on reopening;
there is no automatic repair or crash-atomic transaction with an external action.
Audit I/O failure prevents dispatch; if it occurs after a partial append, the
uncertain storage state requires operator recovery. No exception path is treated
as permission to act.

`replay_check(receipt, request)` re-renders and verifies fingerprint, trust map,
version identifiers and **every listed prompt**, not probabilities or model
outputs. Default `prompts=None` attests canonical trust-mode one-pass rendering
for every question. For any other inference path, supply the complete actual
manifest with `PromptUse.from_rendered(qid, rendered)`. For thinking, include the
opening prompt and each `PromptUse.from_rendered(..., reasoning=..., truncated=...)`
answer-slot prompt; closed prompts without their opening are rejected. Include
repeated samples when used. Reasoning is retained for byte-exact replay and may
be sensitive; protect these local logs. The host adapter must truthfully supply
all prompts and model metadata; the log cannot observe undisclosed remote calls.

See root [`PROTOCOL.md`](../../PROTOCOL.md) for a runnable replay example.

## Exact invariants and trusted computing base

1. **I1:** every executor invocation's action belongs to the trusted authorized
   set. Changing only untrusted content cannot change that set. It may change the
   chosen authorized action or cause deferral.
2. **I2:** a returned `deferred`, `denied` (also `held`) outcome invokes no executor.
3. **I3:** an authorized proposal requiring review when the queue is full is
   `held`, never auto-approved; queued count never exceeds capacity.
4. **I4:** accepted trust labels originate in this host's constructors, not wire
   data or content; untrusted derivation lineage cannot be upgraded.
5. **I5:** each returned outcome has exactly one appended receipt; an executor
   exception also has exactly one write-ahead receipt. This assumes serializable,
   renderable inputs and working audit storage. Pre-admission errors, audit I/O
   failure, and process crashes do not guarantee a returned outcome or atomic
   persistence, but cannot bypass the receipt-before-dispatch ordering.

The trusted computing base is: authenticated workflow adapter/configuration and
principal binding; `Authority`, `HostInputs`, `HostKernel`, `ReviewQueue`; the
protocol type/derive/canonicalization and renderer code used for receipts;
`receipts.py`; Python/stdlib and the uncompromised host process/OS/filesystem;
the protected checkpoint; and the executor's faithful mapping of action names to
side effects. Model outputs, untrusted segments and claimed roles inside them
are **not** authority. The gate is trusted only for its own act/defer decision,
not for grants; statistical gate validity is outside these structural claims.
The existing `decide.provenance` uses legacy JSON-pointer spans; this package
reuses the frozen Keel segment/derive interface instead of creating a competing
span representation.

These guarantees assume correct host labeling and principal authentication. They
do not guarantee semantic correctness, argument-level permissions, model
robustness, evidence quality, exactly-once external effects, or safety after host
compromise. Direct calls to the executor outside this kernel are outside scope.

## Verification and measured bounds

`python -X utf8 -m keel.host.model_check` reports:

```text
CheckReport(configurations=48, states=632, transitions=16920, max_depth=4, concrete_transitions=16920)
```

The exhaustive abstract checker covers two actions `{a,b}`, two roles
`reader->{a}` / `writer->{b}`, every subset of policy actions and assigned roles,
queue capacities **0..2**, and sequences of length **0..4**. Forty events combine
action, act/defer, five provenance patterns (untrusted, trusted, mixed, transitive
untrusted derivation, foreign forged trusted segment), and benign/injection text.
It visits **632 reachable quotient states** and **16,920 transitions** across
48 configurations. Each concrete transition is compared against the independent
abstract specification, with real host construction, in-memory JSONL receipts,
and a side-effecting executor. The checker compares authorization before/after
untrusted-content changes, provenance, calls, queue counts, outcomes and receipt
counts. Every concrete state is rebuilt from a representative reaching history.

The quotient retains depth, queue occupancy, executor-call count and receipt
count. Histories differing only in past payloads are equivalent for this kernel:
it does not consult past queue item values, executor results or receipt contents
to choose the next outcome (the chain is still verified). This is a bounded
refinement check under that abstraction, **not** an unbounded formal proof or
exhaustive enumeration of arbitrary strings/Python objects. Queue drains,
concurrency, malformed inputs, disk tampering and executor/storage exceptions
are outside the abstract space and covered by separate concrete tests.

Five deliberate broken kernels are each rejected with the corresponding invariant:
unauthorized dispatch (I1), dispatch after deferral (I2), overflow approval (I3),
trust laundering (I4), duplicate receipts (I5). Hypothesis adds 50 deterministic
stateful examples of up to 12 proposals with arbitrary untrusted text, action
sets, gate outcomes and queue drains. Disk tests cover concurrent independent
writers, tampering, deletion/reordering, checkpoint loss and external anchors;
replay tests cover multiple questions, both render modes, option ordering,
Unicode separators and thinking/truncation.

Acceptance command (Python 3.12, CPU only):

```sh
/home/alp/ssdm-venv/bin/python -X utf8 -m pytest -q keel/tests
```

Verified 2026-10-15: **70 passed in 18.15s**, including the original protocol
tests, with no warnings. Dependencies are the repo's existing pytest/hypothesis
environment; no packages were added.
GPU budget: **$0; no paid or GPU runs**. No dataset text or model outputs were used
or committed.
