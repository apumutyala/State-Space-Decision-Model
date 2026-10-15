"""Small host-owned authority boundary; model outputs are evidence, not grants."""
from __future__ import annotations

import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from keel.protocol import DecisionRequest, DecisionResult, OperatorPolicy, Question, Segment, derive, render
from keel.protocol.receipts import PromptUse, Receipt, ReceiptLog
from keel.protocol.types import TRUSTED_KINDS, canonical_json

UNTRUSTED_KINDS = frozenset({"record", "tool_output", "user_message", "retrieved"})


def _snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class Authority:
    """Immutable grants: policy actions INTERSECT host roles for a principal.

    Both the policy and mappings must come from authenticated host configuration.
    Neither policy text nor segment contents are interpreted as grants.
    """

    def __init__(self, policy: OperatorPolicy, role_actions: Mapping[str, Sequence[str]],
                 principal_roles: Mapping[str, Sequence[str]]):
        self._policy = OperatorPolicy(policy.policy_id, policy.text, policy.version, tuple(policy.authorized_actions))
        self._roles = MappingProxyType({role: frozenset(actions) for role, actions in role_actions.items()})
        self._principals = MappingProxyType({principal: frozenset(roles) for principal, roles in principal_roles.items()})
        self._allowed = MappingProxyType({principal: frozenset(self._policy.authorized_actions).intersection(
            frozenset().union(*(self._roles.get(role, frozenset()) for role in roles)))
            for principal, roles in self._principals.items()})

    @property
    def policy(self) -> OperatorPolicy:
        return self._policy

    def authorized_actions(self, principal: str) -> frozenset[str]:
        return self._allowed.get(principal, frozenset())


class HostInputs:
    """Session-local constructors; never expose the trusted constructors to a model.

    Registered identities plus content snapshots reject wire-supplied labels and
    in-place changes to otherwise frozen Segments. Keep parents in the request.
    This registry lives for the host session; it is not an authentication service.
    """

    def __init__(self):
        self._segments: dict[int, tuple[Segment, str]] = {}

    def _remember(self, segment: Segment) -> Segment:
        self._segments[id(segment)] = (segment, canonical_json(asdict(segment)))
        return segment

    def _known(self, segment: Segment) -> bool:
        registered = self._segments.get(id(segment))
        return (registered is not None and registered[0] is segment
                and registered[1] == canonical_json(asdict(segment)))

    def untrusted(self, segment_id: str, content: Any, *, kind: str = "record", source: str = "") -> Segment:
        if kind not in UNTRUSTED_KINDS:
            raise ValueError("untrusted inputs cannot use host-authored kinds")
        return self._remember(Segment(segment_id, _snapshot(content), "untrusted", kind, source))

    def operator_note(self, segment_id: str, content: Any, *, source: str = "operator") -> Segment:
        return self._remember(Segment(segment_id, _snapshot(content), "trusted", "operator_note", source))

    def metadata(self, segment_id: str, content: Any, *, source: str = "host") -> Segment:
        return self._remember(Segment(segment_id, _snapshot(content), "trusted", "metadata", source))

    def derive(self, segment_id: str, content: Any, parents: Sequence[Segment], *,
               kind: str = "retrieved", source: str = "derived") -> Segment:
        if any(not self._known(parent) for parent in parents):
            raise ValueError("derivation parents must come from these host constructors")
        if kind not in TRUSTED_KINDS | UNTRUSTED_KINDS:
            raise ValueError("unknown derived segment kind")
        return self._remember(derive(segment_id, _snapshot(content), list(parents), kind=kind, source=source))

    def validate(self, request: DecisionRequest) -> bool:
        ids = {segment.segment_id for segment in request.segments}
        return all(self._known(segment) and set(segment.parent_ids) <= ids for segment in request.segments)


@dataclass(frozen=True)
class ActionProposal:
    action: str
    args: dict[str, Any]
    proposer: str
    cited_segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or not self.action or not isinstance(self.proposer, str):
            raise ValueError("action and proposer must be strings; action cannot be empty")
        if not isinstance(self.args, dict) or not all(isinstance(k, str) for k in self.args):
            raise ValueError("action args must be a JSON object with string keys")
        if any(not isinstance(s, str) for s in self.cited_segment_ids):
            raise ValueError("cited segment ids must be strings")
        object.__setattr__(self, "args", _snapshot(self.args))
        object.__setattr__(self, "cited_segment_ids", tuple(self.cited_segment_ids))


@dataclass(frozen=True)
class Outcome:
    status: str  # executed | deferred | held | denied
    reason: str
    receipt_id: str


@dataclass(frozen=True)
class ReviewItem:
    request_fingerprint: str
    proposal: ActionProposal
    receipt_id: str


class ReviewQueue:
    """Bounded pending reviews; draining is NOT approval and invokes no executor."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("review capacity must be a nonnegative integer")
        self._capacity = capacity
        self._items: deque[ReviewItem] = deque()
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def pop(self) -> ReviewItem | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def pending(self) -> tuple[ReviewItem, ...]:
        with self._lock:
            # Do not leak mutable arguments held by the queue.
            return tuple(ReviewItem(i.request_fingerprint,
                                    ActionProposal(**asdict(i.proposal)), i.receipt_id) for i in self._items)


class ExecutionError(RuntimeError):
    """An authorized dispatch failed; its receipt exists, effects may be partial."""

    def __init__(self, receipt_id: str):
        super().__init__(f"executor raised after authorized dispatch; effects unknown; receipt={receipt_id}")
        self.receipt_id = receipt_id


class HostKernel:
    """Serialize authorize -> durable receipt -> enqueue/dispatch.

    ``principal`` is bound by the authenticated host, not by ActionProposal or
    segment text. Every returned Outcome has exactly one write-ahead receipt.
    An act receipt authorizes a dispatch; it is NOT a completion confirmation.
    No automatic retry or human-review approval path exists here.
    """

    def __init__(self, authority: Authority, *, principal: str, executor: Callable[[ActionProposal], Any],
                 receipt_log: ReceiptLog, review_capacity: int, model_id: str, model_revision: str,
                 gate_version: str, calibration_id: str):
        for value in (model_id, model_revision, gate_version, calibration_id):
            if not isinstance(value, str) or not value:
                raise ValueError("model/revision/gate/calibration identifiers must be nonempty strings")
        self.authority = authority
        self.principal = principal
        self.executor = executor
        self.receipt_log = receipt_log
        self.review_queue = ReviewQueue(review_capacity)
        self.inputs = HostInputs()
        self.model_id, self.model_revision = model_id, model_revision
        self.gate_version, self.calibration_id = gate_version, calibration_id
        self._lock = threading.RLock()
        self._processing = False

    def request(self, request_id: str, segments: Sequence[Segment], questions: Sequence[Question]) -> DecisionRequest:
        return DecisionRequest(request_id, self.authority.policy, tuple(segments), tuple(questions))

    def _decide(self, request: DecisionRequest, proposal: ActionProposal,
                result: DecisionResult) -> tuple[str, str]:
        if request.policy != self.authority.policy:
            return "denied", "operator_policy_mismatch"
        if proposal.proposer != self.principal:
            return "denied", "proposer_is_not_authenticated_principal"
        if not self.inputs.validate(request):
            return "denied", "unregistered_or_modified_segment"
        if not set(proposal.cited_segment_ids) <= {s.segment_id for s in request.segments}:
            return "denied", "unknown_cited_segment"
        if proposal.action not in self.authority.authorized_actions(self.principal):
            return "denied", "action_not_authorized"
        if result.request_id != request.request_id or set(result.answers) != {q.qid for q in request.questions}:
            return "denied", "decision_request_or_questions_mismatch"
        if type(result.act) is not bool:
            return "denied", "invalid_act_flag"
        for question in request.questions:
            answer = result.answers[question.qid]
            option_ids = set(render(request, question.qid).option_ids)
            if (answer.qid != question.qid or set(answer.probs) != option_ids
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                           for p in answer.probs.values())
                    or not math.isclose(sum(answer.probs.values()), 1.0, abs_tol=1e-8, rel_tol=0.0)):
                return "denied", "invalid_question_probabilities"
            choices = {"yes", "no"} if question.type == "noul" else option_ids
            if result.act and answer.decision not in choices:
                return "denied", "missing_or_invalid_question_decision"
        if not result.act or result.defer_reasons:
            if len(self.review_queue) >= self.review_queue.capacity:
                return "held", "review_queue_full"
            return "deferred", "gate_deferred"
        return "executed", "authorized_dispatch"

    def _record(self, request: DecisionRequest, proposal: ActionProposal, result: DecisionResult,
                status: str, reason: str, prompts: Sequence[PromptUse] | None,
                attack_tag: str | None) -> Receipt:
        receipt = Receipt.capture(
            request, result, model_id=self.model_id, model_revision=self.model_revision,
            gate_version=self.gate_version, calibration_id=self.calibration_id,
            outcome={"executed": "act", "deferred": "defer", "held": "hold", "denied": "deny"}[status],
            reasons=(reason, *result.defer_reasons), prompts=prompts, attack_tag=attack_tag,
            proposal=asdict(proposal), authorization={"principal": self.principal,
                "authorized_actions": sorted(self.authority.authorized_actions(self.principal))})
        return self.receipt_log.append(receipt)

    def process(self, request: DecisionRequest, proposal: ActionProposal, decision_result: DecisionResult, *,
                prompts: Sequence[PromptUse] | None = None, attack_tag: str | None = None) -> Outcome:
        """Process one proposal. Default prompts mean trust/canonical/one-pass.

        Supply the complete actual prompt manifest for other rendering modes or
        thinking/sampling. Invalid JSON/render manifests or unavailable audit
        storage raise before dispatch, with no returned Outcome. A dispatch
        exception raises ExecutionError AFTER recording the authorized attempt.
        """
        with self._lock, self.review_queue._lock:
            if self._processing:
                raise RuntimeError("reentrant process calls are not allowed")
            self._processing = True
            try:
                # Snapshot arguments before both recording and dispatch.
                proposal = ActionProposal(**asdict(proposal))
                status, reason = self._decide(request, proposal, decision_result)
                receipt = self._record(request, proposal, decision_result, status, reason, prompts, attack_tag)
                decision_result.receipt_id = receipt.receipt_id
                if status == "deferred":
                    self.review_queue._items.append(ReviewItem(request.fingerprint(), proposal, receipt.receipt_id))
                elif status == "executed":
                    try:
                        self.executor(proposal)
                    except Exception as exc:
                        raise ExecutionError(receipt.receipt_id) from exc
                return Outcome(status, reason, receipt.receipt_id)
            finally:
                self._processing = False
