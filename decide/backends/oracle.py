"""OracleBackend: deterministic, content-addressed logits for tests.

Logits depend only on (question id, option id, option description, state), never
on the order options were listed, so it is permutation-equivariant by
construction. ``labels`` forces a chosen option per question (test oracle only;
the real adapter never receives evaluation labels).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from ..schema import canonical_options, render
from .base import BackendUnsupported, Plan, ScoreResult


def _hash_logit(*parts: str) -> float:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "big") / 2**64) * 6.0 - 3.0


class OracleBackend:
    name = "oracle"

    def __init__(
        self,
        labels: dict[str, str] | None = None,
        margin: float = 4.0,
        tokens_per_char: float = 0.25,
        emit_attribution: bool = False,
        fail_with: tuple[str, str] | None = None,
        sleep_per_call_s: float = 0.0,
    ) -> None:
        self.labels = labels or {}
        self.margin = margin
        self.tokens_per_char = tokens_per_char
        self.emit_attribution = emit_attribution
        self.fail_with = fail_with
        self.sleep_per_call_s = sleep_per_call_s

    def revision(self) -> dict[str, Any]:
        return {
            "revision_id": f"oracle:v1;margin={self.margin}",
            "backend": self.name,
            "model_id": None,
            "note": "test oracle; not a model",
        }

    def _tokens(self, request: dict[str, Any]) -> int:
        size = len(render(request["state"])) + sum(len(render(q)) for q in request["questions"].values())
        return max(1, int(size * self.tokens_per_char))

    def plan(self, request: dict[str, Any]) -> Plan:
        if self.fail_with:
            raise BackendUnsupported(*self.fail_with)
        n = self._tokens(request)
        k = len(request["questions"])
        return Plan(inference_calls=k, max_call_input_tokens=n, total_input_tokens=n * k)

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import time

        state = render(request["state"])
        logits: dict[str, dict[str, float]] = {}
        attribution: dict[str, list[dict[str, Any]]] = {}
        for qid, q in request["questions"].items():
            if self.sleep_per_call_s:
                time.sleep(self.sleep_per_call_s)
            check_deadline()
            row = {}
            for opt, desc in canonical_options(q):
                row[opt] = _hash_logit(str(qid), opt, render(desc), state)
                if self.labels.get(qid) == opt:
                    row[opt] += self.margin
            logits[qid] = row
            if self.emit_attribution:
                attribution[qid] = [
                    {"evidence_ref": "state", "span": [0, len(state)], "option_id": o, "contribution": v}
                    for o, v in row.items()
                ]
        n = self._tokens(request)
        return ScoreResult(
            logits=logits,
            inference_calls=len(logits),
            total_input_tokens=n * len(logits),
            extras={"oracle": True},
            attribution=attribution if self.emit_attribution else None,
        )
