# Keel decision-authority protocol v1

Keel separates **evidence**, **judgment**, and **host authority**. A model can
recommend an action or defer; it cannot grant permission. Integrators authenticate
the principal and policy, construct trust-labelled inputs, obtain a judgment,
then pass the proposal through `keel.host.HostKernel`. Never dispatch outside that
boundary if relying on its guarantees.

## Request and judgment

`DecisionRequest` contains `request_id`, trusted `policy`, `segments`, and
`questions`. Its wire format includes `protocol="keel-protocol/1"`.
`to_json()/from_json()` round-trip; `fingerprint()` is SHA256 of canonical JSON.

- `OperatorPolicy`: `policy_id`, `text`, `version`, `authorized_actions`.
- `Segment`: unique `segment_id`, JSON `content`, `trust`, `kind`, `source`,
  `parent_ids`. Only `operator_note` and `metadata` may be trusted, and must be.
  `record`, `tool_output`, `user_message`, and `retrieved` are untrusted evidence.
- `Question`: unique `qid`, `type` (`choice`, `noul`, `score`), `instructions`,
  and `criteria`. Question definitions are host-authored, not authority supplied
  by retrieved text.
- `DecisionResult`: request ID, per-question `Answer(qid, probs, decision)`,
  `act`, `defer_reasons`, optional signals, and eventual `receipt_id`.

Use `kernel.inputs.untrusted`, `.operator_note`, `.metadata`, and `.derive`;
only authenticated host code may access trusted constructors. Derived segments
reuse `keel.protocol.derive`: any untrusted parent (or no parents) implies
untrusted output. Keep lineage parents in the request. The kernel rejects
unregistered segment objects and modified contents; wire labels are not credentials.

`Authority(policy, role_actions, principal_roles, argument_clauses=(), argument_defaults=None)` computes each
principal's allowed set as policy actions intersected with its host-configured role grants. An action with
`ArgumentClause`s runs only with arguments matching at least one clause (fixed values, finite sets, numeric upper
bounds; omitted arguments filled from defaults). An action without clauses has unconstrained arguments.
The kernel binds an authenticated principal independently of the proposal's
`proposer`. `ActionProposal` contains `action`, JSON-object `args`, `proposer`,
and `cited_segment_ids`. Citations are evidence references, not permission.

## Rendering and receipts

Current versions are `keel-render/1` and `trust-policy/1`.
`render(request, qid, mode="trust", think=False, canonical=True)` places trusted
policy/notes in the system turn and escaped untrusted records in the user turn.
Canonical option ordering uses option contents. `mode="base"` is the deliberately
undefended comparison, not an authority mechanism. For thinking, `close_think`
builds the second answer-slot prompt from reasoning and the truncation flag.

`Receipt` (`keel-receipt/1`) records:

- `receipt_id` (SHA256 of the canonical receipt excluding that field), `prev_hash`,
  request fingerprint, and UTC timestamp;
- every supplied prompt's SHA256 and replay settings (`qid`, mode, thinking,
  canonical ordering, stage, plus reasoning/truncation for answer-slot prompts);
- model ID/revision, render/trust versions, caller-supplied gate version and
  calibration ID, and optional attack tag;
- segment-ID trust map: trust, kind, source, canonical-JSON content SHA256, parents;
- per-question probabilities/decisions, `act/defer/hold/deny` outcome and reasons,
  decision-result request ID, proposal and host-computed authorization set.

`ReceiptLog(path).append` writes canonical JSONL and advances a separately
protected `.checkpoint` count/tip. `verify()` checks hashes, order, canonical bytes
and that checkpoint, detecting edits and deletions **including suffix deletion**.
Protect the checkpoint independently or pin `LogCheckpoint` externally and call
`verify(expected=anchor)`: a hash chain alone cannot detect tail removal or a
complete rewrite of both log and checkpoint. This is not a digital signature.
Disk writers lock and fsync; `ReceiptLog()` without a path is ephemeral.

The host appends exactly one write-ahead receipt before enqueue/dispatch. An
`act` receipt means **authorized dispatch intent**, not proof of completion.
Executor failure raises `ExecutionError(receipt_id)`; effects may be partial.
A crash may leave an intent without execution. An interrupted log/checkpoint
update fails closed, with no automatic repair or action retry.

## Five invariants and exact scope

| Invariant | Guarantee |
|---|---|
| I1 | Every executor invocation's action is in the trusted authorized set, and its arguments match the action's trusted argument clauses when it has any. Changing only untrusted content cannot expand or otherwise change the set or the clauses; it may change which permitted action or in-clause arguments are proposed, or whether to defer. |
| I2 | Returned defer/deny/hold outcomes never call the executor. |
| I3 | A proposal requiring review at a full queue is held, never auto-approved; occupancy never exceeds capacity. Independently authorized acts need not enter that queue. |
| I4 | Accepted labels originate in host constructors; untrusted kinds and untrusted derivation lineage cannot gain trust. |
| I5 | Each returned outcome, and each executor exception after authorization, has exactly one receipt. |

Assumptions: correct host labelling, authenticated policy/principal/configuration,
uncompromised host process/OS, faithful executor dispatch, serializable/renderable
inputs, truthful inference metadata, and functioning audit storage. Malformed
pre-admission inputs or audit failures raise without a returned outcome; storage
failures/crashes do not promise atomic persistence, but cannot bypass the
receipt-before-dispatch ordering. Draining a review queue is not approval.

**Not guaranteed:** semantic correctness of permitted actions, safety of arguments a clause leaves free, model
robustness, model-output reproducibility, statistical validity of a named calibration, exactly-once external effects,
or security after host compromise. The bounded checker covers 632 reachable abstract quotient states,
25,380 concrete-refinement transitions (including an argument-clause dimension) and sequences up to length four;
this is not an unbounded formal proof. Full bounds, trusted computing base and tests are
in [`keel/host/README.md`](keel/host/README.md).

## Replay example (local, no model call)

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from keel.host import ActionProposal, Authority, HostKernel
from keel.protocol import Answer, DecisionResult, OperatorPolicy, Question
from keel.protocol.receipts import ReceiptLog, replay_check

with TemporaryDirectory() as directory:
    log = ReceiptLog(Path(directory) / "decisions.jsonl")
    calls = []
    policy = OperatorPolicy("routing", authorized_actions=("route",))
    host = HostKernel(
        Authority(policy, {"router": ("route",)}, {"service": ("router",)}),
        principal="service", executor=calls.append, receipt_log=log,
        review_capacity=2, model_id="example-judge", model_revision="example-rev",
        gate_version="example-gate/1", calibration_id="uncertified-example")
    request = host.request("r1", [host.inputs.untrusted("mail", "Parcel late.")],
        [Question("q", "choice", "Which team?", {"shipping": "Late parcels"})])
    result = DecisionResult("r1", {"q": Answer("q", {"shipping": 1.0}, "shipping")}, True)
    outcome = host.process(request,
        ActionProposal("route", {"team": "shipping"}, "service", ("mail",)), result)
    anchor = log.verify()
    receipt = log.receipts()[-1]
    assert log.verify(expected=anchor).count == 1
    assert outcome.status == "executed" and len(calls) == 1
    assert replay_check(receipt, request)
```

The synthetic result above illustrates the boundary, not model quality. In real
inference, pass every actual `PromptUse` to `process(..., prompts=manifest)`;
include thinking openings and all sampled answer-slot prompts. Omitting the
manifest asserts canonical trust-mode one-pass prompts for every question.
Replay re-renders those prompts and matches hashes/fingerprint; it does not rerun
the model or verify probabilities. Retain the original request securely; receipts
contain hashes rather than request text (thinking reasoning is retained).
