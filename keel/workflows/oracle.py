"""ORACLE task grants from clean AgentDojo ground truth, never a deployable planner.

The grants are enforced by the host kernel: tool names through ``Authority`` and, for the argument arm, argument
values through host ``ArgumentClause`` objects built here. The adapter itself never decides a grant.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from agentdojo.functions_runtime import FunctionCall

from keel.host import ArgumentClause
from keel.protocol import Answer, DecisionResult, OperatorPolicy, Question, render
from keel.protocol.receipts import PromptUse

from .gate_element import KeelGateElement, _has_nested_call, build_request

ORACLE_ARMS = ("oracle_tool_authority", "oracle_arg_authority")
ORACLE_QUESTION = Question(
    "q", "choice", "Record the deterministic ORACLE policy result; no model is queried.",
    {"allow": "The adapter does not veto; the kernel checks tool-name and argument authority.",
     "hold": "The deterministic adapter policy vetoes this call."},
)

@dataclass(frozen=True)
class ArgumentPolicy:
    """Disjunction of ground-truth calls, each a conjunction of fixed arguments.

    Omitted arguments remain free. Only the benchmark's literal body placeholders
    '<summary>' and '...' in Slack send-message calls are free. Every other supplied
    value is constrained, including dates, subjects, read limits and literal bodies.
    Calls are not consumed: neither ordering nor multiplicity is constrained.
    Immutable JSON snapshots prevent later mutation of ground-truth objects/grants.
    """

    VERSION: ClassVar[str] = "oracle-ground-truth-arguments/2"
    _clauses: tuple[str, ...]
    _defaults: str = "{}"

    @classmethod
    def from_calls(cls, calls: Sequence[FunctionCall], *, tool_defaults: Mapping | None = None) -> "ArgumentPolicy":
        clauses = []
        for call in calls:
            if _has_nested_call(call.args):
                raise ValueError("nested ground-truth calls are unsupported")
            args = call.model_dump(mode="json")["args"]
            free = [key for key, value in args.items()
                    if call.function in ("send_direct_message", "send_channel_message")
                    and key == "body" and isinstance(value, str) and value in ("<summary>", "...")]
            clause = {"tool": call.function, "fixed": {k: v for k, v in args.items() if k not in free},
                      "free_placeholders": free}
            clauses.append(json.dumps(clause, ensure_ascii=False, allow_nan=False, sort_keys=True))
        defaults = json.dumps(tool_defaults or {}, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return cls(tuple(clauses), defaults)

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(sorted({json.loads(clause)["tool"] for clause in self._clauses}))

    @property
    def defaults(self) -> dict:
        return json.loads(self._defaults)

    def clauses(self) -> tuple[ArgumentClause, ...]:
        """Host argument clauses: one per ground-truth call, each fixing that call's supplied arguments."""
        return tuple(ArgumentClause(c["tool"], fixed=c["fixed"]) for c in map(json.loads, self._clauses))

    def allows(self, tool: str, args: Mapping) -> bool:
        """The kernel's own matching (``ArgumentClause.allows``) for a granted tool, after filling tool defaults."""
        effective = {**self.defaults.get(tool, {}), **args}
        return any(clause.allows(effective) for clause in self.clauses() if clause.action == tool)

    def manifest(self) -> dict:
        return {"version": self.VERSION, "source": "clean AgentDojo user-task ground_truth()",
                "clauses": [json.loads(clause) for clause in self._clauses],
                "missing_fixed_arguments_use_tool_defaults": json.loads(self._defaults),
                "omitted_arguments": "unconstrained", "order_and_multiplicity": "unconstrained",
                "enforced_by": "keel.host.Authority argument clauses"}

def task_argument_policy(suite, task) -> ArgumentPolicy:
    """Build before attack injection, on an isolated task-initialized clean state."""
    clean = task.init_environment(suite.load_and_inject_default_environment({}))
    calls = task.ground_truth(clean)
    names = {call.function for call in calls}
    defaults = {tool.name: {name: field.default for name, field in tool.parameters.model_fields.items()
                           if not field.is_required()}
                for tool in suite.tools if tool.name in names}
    return ArgumentPolicy.from_calls(calls, tool_defaults=defaults)


class OracleAuthorityElement(KeelGateElement):
    """Reuse kernel-only dispatch/receipts, replacing judgment with an ORACLE rule."""

    def __init__(self, user_task, receipt_log, episode_id, *, arm, argument_policy, executor=None):
        if arm not in ORACLE_ARMS:
            raise ValueError(f"unknown ORACLE arm {arm}")
        self.arm = arm
        self.argument_policy = argument_policy
        self.grant = ({"version": "oracle-ground-truth-tools/1", "tools": list(argument_policy.tools),
                       "source": "clean AgentDojo user-task ground_truth()"}
                      if arm == "oracle_tool_authority" else argument_policy.manifest())
        policy = OperatorPolicy(
            f"{arm}/1", "ORACLE: a perfect trusted planner supplies task-scoped ground-truth grants. "
            "No judge is used. Tool outputs cannot expand these grants. The render is an audit-only record.")
        argument_arm = arm == "oracle_arg_authority"
        super().__init__(user_task, None, receipt_log, episode_id,
                         authorized_actions=argument_policy.tools, model_id="none:oracle-policy",
                         model_revision=self.grant["version"], executor=executor, policy=policy,
                         gate_version=f"{arm}/1", calibration_id="deterministic-policy-not-probabilities",
                         argument_clauses=argument_policy.clauses() if argument_arm else (),
                         argument_defaults=argument_policy.defaults if argument_arm else None)

    def _request(self, history, call):
        request = build_request(self.kernel, self.user_task, history, call,
                                f"{self.episode_id}:{len(self.outcomes)}")
        grant = self.kernel.inputs.metadata("oracle-grant", self.grant, source="trusted_oracle_planner")
        return self.kernel.request(request.request_id, (*request.segments, grant), (ORACLE_QUESTION,))

    def _evaluate(self, request, call):
        # Tool names and arguments outside the grant are denied by HostKernel itself, independent of this
        # adapter's act flag. No pretend judge is called.
        reasons = ["nested_calls_not_supported"] if _has_nested_call(call.args) else []
        act = not reasons
        probs = {"allow": float(act), "hold": float(not act)}
        result = DecisionResult(request.request_id, {"q": Answer("q", probs, "allow" if act else "hold")},
                                act=act, defer_reasons=reasons)
        # Receipt v1 requires a render manifest for every question. This is an
        # audit-only serialization, explicitly marked no-model in the receipt.
        return result, (PromptUse.from_rendered("q", render(request, mode="trust")),)
