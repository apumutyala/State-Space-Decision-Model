"""ReplayBackend: serves recorded per-option logits from a fixture.

Fixture layout (``fixtures/replay/v0/``):

* ``manifest.json``: ``revision`` (the recording backend's full revision dict),
  ``calibration``, ``enable_v0_1_deltas``, generator command and hashes.
* ``recordings.jsonl``: one line per scored request:
  ``{"fingerprint", "logits": {qid: {option_id: logit}}, "plan": {...},
  "inference_calls", "total_input_tokens"}``.

Lookup is by :func:`decide.schema.request_fingerprint` (state + questions,
option/question order-insensitive). A miss is ``unsupported/replay_miss``;
there is no fallback to any model. Replayed logits go through the same adapter
code path as live inference, so probabilities, wire answers and status codes are
recomputed, not copied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..schema import request_fingerprint
from .base import BackendUnsupported, Plan, ScoreResult


class ReplayBackend:
    name = "replay"

    def __init__(self, fixture_dir: str | Path) -> None:
        self.fixture_dir = Path(fixture_dir)
        self.manifest = json.loads((self.fixture_dir / "manifest.json").read_text(encoding="utf-8"))
        self.recordings: dict[str, dict[str, Any]] = {}
        with open(self.fixture_dir / "recordings.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self.recordings[rec["fingerprint"]] = rec

    def revision(self) -> dict[str, Any]:
        # Replay reports the recorded parent's revision so parent_revision pinning
        # behaves exactly as against the live parent; the replay itself is noted.
        rev = dict(self.manifest["revision"])
        rev["served_by"] = "replay:" + self.manifest.get("fixture_id", "unknown")
        return rev

    def _lookup(self, request: dict[str, Any]) -> dict[str, Any]:
        fp = request_fingerprint(request)
        rec = self.recordings.get(fp)
        if rec is None:
            raise BackendUnsupported("replay_miss", f"no recording for fingerprint {fp[:16]}; no fallback")
        return rec

    def plan(self, request: dict[str, Any]) -> Plan:
        rec = self._lookup(request)
        p = rec["plan"]
        return Plan(
            inference_calls=p["inference_calls"],
            max_call_input_tokens=p["max_call_input_tokens"],
            total_input_tokens=p["total_input_tokens"],
        )

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        rec = self._lookup(request)
        if rec.get("logits") is None:
            raise BackendUnsupported(
                "replay_miss", "recording has a plan but no logits (it was rejected by limits when recorded)"
            )
        check_deadline()
        return ScoreResult(
            logits=rec["logits"],
            inference_calls=rec["inference_calls"],
            total_input_tokens=rec["total_input_tokens"],
            extras={"replayed": True, "recorded_extras": rec.get("extras", {})},
        )


VOLATILE_DIAGNOSTICS = ("latency_ms", "backend", "model_revision")


def normalize_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that legitimately differ between a live run and its replay
    (wall-clock latency, backend call timings, the replay's ``served_by`` note).
    Everything else, including revision_id, probabilities and wire answers,
    must match exactly."""
    out = json.loads(json.dumps(resp))
    diag = out.get("ssdm", {}).get("diagnostics", {})
    for k in VOLATILE_DIAGNOSTICS:
        diag.pop(k, None)
    return out
