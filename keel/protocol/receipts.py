"""Canonical, hash-chained decision receipts with a separately protected tip.

The tip is essential: a hash chain alone cannot detect removal of its suffix.
Disk logs use a sibling ``.checkpoint`` file. Protect it independently (or pin a
``LogCheckpoint`` externally); neither a chain nor this file is a signature.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import fcntl

from .render import RENDER_VERSION, TRUST_POLICY_VERSION, Rendered, close_think, render
from .types import DecisionRequest, DecisionResult, sha256_json

GENESIS = "0" * 64
RECEIPT_VERSION = "keel-receipt/1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReceiptIntegrityError(ValueError):
    """The log, its protected checkpoint, or a replay input disagrees."""


@dataclass(frozen=True)
class PromptUse:
    """One actual model input; repeated samples may have repeated entries.

    A thinking call records its opening prompt AND every answer-slot prompt.
    The latter retains reasoning so replay can reconstruct its exact bytes.
    """

    qid: str
    sha256: str
    mode: str = "trust"
    think: bool = False
    canonical: bool = True
    stage: Literal["render", "close_think"] = "render"
    reasoning: str | None = None
    truncated: bool = False

    @classmethod
    def from_rendered(cls, qid: str, rendered: Rendered, *, reasoning: str | None = None,
                      truncated: bool = False) -> "PromptUse":
        if rendered.version != RENDER_VERSION:
            raise ValueError("unsupported rendering version")
        prompt = rendered.prompt if reasoning is None else close_think(rendered, reasoning, truncated)
        return cls(qid, _hash(prompt), rendered.mode, rendered.think, rendered.canonical,
                   "render" if reasoning is None else "close_think", reasoning, truncated)

    def replay(self, request: DecisionRequest) -> str:
        if self.stage not in ("render", "close_think"):
            raise ValueError("unknown prompt stage")
        rendered = render(request, self.qid, mode=self.mode, think=self.think, canonical=self.canonical)
        if self.stage == "close_think":
            if self.reasoning is None:
                raise ValueError("answer-slot replay requires reasoning")
            return close_think(rendered, self.reasoning, self.truncated)
        if self.reasoning is not None or self.truncated:
            raise ValueError("opening prompts cannot contain reasoning or truncation")
        return rendered.prompt


def trust_map(request: DecisionRequest) -> dict[str, dict[str, Any]]:
    return {s.segment_id: {"trust": s.trust, "kind": s.kind, "source": s.source,
                           "content_sha256": sha256_json(s.content), "parent_ids": list(s.parent_ids)}
            for s in request.segments}


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    prev_hash: str
    request_fingerprint: str
    prompts: tuple[PromptUse, ...]
    model_id: str
    model_revision: str
    render_version: str
    trust_policy_version: str
    gate_version: str
    calibration_id: str
    trust_map: dict[str, dict[str, Any]]
    attack_tag: str | None
    answers: dict[str, dict[str, Any]]
    outcome: Literal["act", "defer", "hold", "deny"]
    reasons: tuple[str, ...]
    timestamp: str
    decision_request_id: str
    proposal: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    version: str = RECEIPT_VERSION

    @classmethod
    def capture(cls, request: DecisionRequest, result: DecisionResult, *, model_id: str,
                model_revision: str, gate_version: str, calibration_id: str,
                outcome: Literal["act", "defer", "hold", "deny"], reasons: Sequence[str],
                prompts: Sequence[PromptUse] | None = None, attack_tag: str | None = None,
                proposal: dict[str, Any] | None = None,
                authorization: dict[str, Any] | None = None) -> "Receipt":
        for name, value in (("model_id", model_id), ("model_revision", model_revision),
                            ("gate_version", gate_version), ("calibration_id", calibration_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty caller-supplied string")
        if outcome not in ("act", "defer", "hold", "deny"):
            raise ValueError("invalid receipt outcome")
        uses = tuple(prompts) if prompts is not None else tuple(
            PromptUse.from_rendered(q.qid, render(request, q.qid)) for q in request.questions)
        receipt = cls("", GENESIS, request.fingerprint(), uses, model_id, model_revision,
                      RENDER_VERSION, TRUST_POLICY_VERSION, gate_version, calibration_id,
                      trust_map(request), attack_tag,
                      {qid: {"qid": a.qid, "probs": dict(a.probs), "decision": a.decision}
                       for qid, a in result.answers.items()}, outcome, tuple(reasons),
                      datetime.now(timezone.utc).isoformat(), result.request_id, proposal, authorization)
        # Snapshot mutable result/argument dictionaries; reject non-JSON and nonfinite inputs.
        receipt = cls.from_json(json.loads(_json(receipt.to_json())))
        if not replay_check(receipt, request):
            raise ValueError("prompt manifest does not replay against the request")
        return receipt

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Receipt":
        data = dict(value)
        data["prompts"] = tuple(PromptUse(**p) for p in data["prompts"])
        data["reasons"] = tuple(data["reasons"])
        return cls(**data)

    def digest(self) -> str:
        payload = self.to_json()
        del payload["receipt_id"]
        return _hash(_json(payload))


def replay_check(receipt: Receipt, request: DecisionRequest) -> bool:
    """Check input/prompt identity, NOT reproduce probabilities or prove correctness."""
    try:
        if (receipt.version != RECEIPT_VERSION or receipt.render_version != RENDER_VERSION
                or receipt.trust_policy_version != TRUST_POLICY_VERSION
                or receipt.request_fingerprint != request.fingerprint()
                or receipt.trust_map != trust_map(request)
                or {p.qid for p in receipt.prompts} != {q.qid for q in request.questions}):
            return False
        if receipt.receipt_id and receipt.receipt_id != receipt.digest():
            return False
        openings = {(p.qid, p.mode, p.think, p.canonical) for p in receipt.prompts if p.stage == "render"}
        if any(p.stage == "close_think" and (p.qid, p.mode, p.think, p.canonical) not in openings
               for p in receipt.prompts):
            return False
        return all(_hash(p.replay(request)) == p.sha256 for p in receipt.prompts)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


@dataclass(frozen=True)
class LogCheckpoint:
    count: int = 0
    head_hash: str = GENESIS


class ReceiptLog:
    """Append-only JSONL; each append verifies the existing chain and protected tip.

    ``path=None`` is an ephemeral JSONL log (the same chain and verification),
    useful for short-lived hosts and bounded checking, NOT durable auditing.
    Disk writers serialize with flock and fsync both data and checkpoint. A
    crash between those writes fails closed; there is no silent tip recovery.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._memory = ""
        self._tip = LogCheckpoint()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._locked():
                if not self.path.exists():
                    # Never bless a deleted log using a surviving checkpoint.
                    if self._checkpoint_path.exists():
                        raise ReceiptIntegrityError("log missing but checkpoint exists")
                    with self.path.open("x", encoding="utf-8") as stream:
                        stream.flush()
                        os.fsync(stream.fileno())
                    self._store_tip(LogCheckpoint())
                self._verify_locked()

    @property
    def _checkpoint_path(self) -> Path:
        assert self.path is not None
        return self.path.with_name(self.path.name + ".checkpoint")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            if self.path is None:
                yield
            else:
                with self.path.with_name(self.path.name + ".lock").open("a") as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _text(self) -> str:
        if self.path is None:
            return self._memory
        return self.path.read_bytes().decode("utf-8")

    def _load_tip(self) -> LogCheckpoint:
        if self.path is None:
            return self._tip
        value = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        tip = LogCheckpoint(**value)
        if type(tip.count) is not int or tip.count < 0 or not isinstance(tip.head_hash, str):
            raise ReceiptIntegrityError("invalid checkpoint")
        return tip

    def _store_tip(self, tip: LogCheckpoint) -> None:
        if self.path is None:
            self._tip = tip
            return
        temporary = self._checkpoint_path.with_name(self._checkpoint_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(asdict(tip)) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self._checkpoint_path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _verify_locked(self, expected: LogCheckpoint | None = None) -> tuple[LogCheckpoint, list[Receipt]]:
        try:
            text, tip = self._text(), self._load_tip()
            previous, receipts = GENESIS, []
            if text and not text.endswith("\n"):
                raise ReceiptIntegrityError("truncated receipt line")
            for line in text.split("\n")[:-1]:
                data = json.loads(line)
                if line != _json(data):
                    raise ReceiptIntegrityError("noncanonical receipt line")
                receipt = Receipt.from_json(data)
                timestamp = datetime.fromisoformat(receipt.timestamp)
                if (receipt.version != RECEIPT_VERSION or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
                        or receipt.prev_hash != previous or receipt.receipt_id != receipt.digest()):
                    raise ReceiptIntegrityError("receipt hash, order, version or UTC timestamp mismatch")
                receipts.append(receipt)
                previous = receipt.receipt_id
            actual = LogCheckpoint(len(receipts), previous)
            if actual != tip or (expected is not None and actual != expected):
                raise ReceiptIntegrityError("log does not match protected checkpoint (including its length)")
            return actual, receipts
        except (OSError, UnicodeError, TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, ReceiptIntegrityError):
                raise
            raise ReceiptIntegrityError(f"unreadable receipt log/checkpoint: {exc}") from exc

    def verify(self, expected: LogCheckpoint | None = None) -> LogCheckpoint:
        """Raise on edits/removals/reordering; optionally compare an external tip."""
        with self._locked():
            return self._verify_locked(expected)[0]

    def receipts(self) -> tuple[Receipt, ...]:
        with self._locked():
            return tuple(self._verify_locked()[1])

    def append(self, receipt: Receipt) -> Receipt:
        with self._locked():
            tip, _ = self._verify_locked()
            sealed = replace(receipt, prev_hash=tip.head_hash, receipt_id="")
            sealed = replace(sealed, receipt_id=sealed.digest())
            line = _json(sealed.to_json()) + "\n"
            # Check schema/timestamp before committing bytes, even for direct callers.
            if sealed.version != RECEIPT_VERSION or datetime.fromisoformat(sealed.timestamp).utcoffset() != timezone.utc.utcoffset(None):
                raise ValueError("receipt requires the current schema and a UTC timestamp")
            if self.path is None:
                self._memory += line
            else:
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            self._store_tip(LogCheckpoint(tip.count + 1, sealed.receipt_id))
            return Receipt.from_json(json.loads(line))
