"""Backend protocol.

A backend turns a validated request into one logit per option per question.
Probabilities, calibration, policy conditioning, wire conversion, status codes
and limits are the adapter's job, so every backend shares one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Plan:
    """What a backend would do for a request, computed before any inference."""

    inference_calls: int
    max_call_input_tokens: int  # longest single model input (row) in tokens
    total_input_tokens: int  # tokens across all rows that would be forwarded
    per_question: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: prompt tokens per label (template / instruction / data), summed over question rows;
    #: None when the backend does not tokenize (oracle, replay)
    prompt_token_labels: dict[str, int] | None = None
    #: True only for backends that consume a per-token trust mask (ISE/ASIDE)
    provenance_applied: bool = False


@dataclass
class ScoreResult:
    logits: dict[str, dict[str, float]]  # qid -> option_id -> logit
    inference_calls: int
    total_input_tokens: int
    extras: dict[str, Any] = field(default_factory=dict)  # backend diagnostics
    attribution: dict[str, list[dict[str, Any]]] | None = None


class DeadlineExceeded(Exception):
    pass


class CallBudgetExceeded(Exception):
    """Raised by the adapter callback when a backend starts more inference calls than
    ``max_inference_calls`` allows (e.g. its plan under-reported the calls it needs)."""


class BackendUnsupported(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Backend(Protocol):
    name: str

    def revision(self) -> dict[str, Any]:
        """Exact model / adapter / tokenizer / inference-config identity."""

    def plan(self, request: dict[str, Any]) -> Plan:
        ...

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        """Must call ``check_deadline()`` between inference calls."""
