"""Experimental RWKV-7 option scorer bound to the exact checked-layout request.

This backend exposes only ordinary option log scores to the existing Decider. It never constructs programs or text;
the planner, semantic kernel and renderer remain the only path to released answers. CPU/fake-cache contract work only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable

from decide.backends.base import BackendUnsupported, Plan, ScoreResult
from decide.schema import canonical_options
from heads.layout import LAYOUT_VERSION, TEMPLATES, build_row


HARD_CONTEXT_TOKENS = 2048


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _row_identity(request: dict[str, Any], row, *, encoder_identity: dict[str, Any],
                  template_id: str, template_identity: dict[str, Any]) -> str:
    # Complete D request binds authority/context/policy as well as the prompt tokens. Only an internally built row
    # reaches this function; callers cannot provide their own row or split point.
    row_fields = ("qid", "qtype", "option_ids", "ids", "state_span", "field_span", "option_spans",
                  "option_seqs", "continuations", "layout", "meta")
    return _sha({"request": request, "row": {key: deepcopy(getattr(row, key)) for key in row_fields},
                 "layout_version": LAYOUT_VERSION, "template_id": template_id,
                 "template_identity": template_identity, "encoder_identity": encoder_identity})


class RWKV7LayoutCausalBackend:
    """Score one D choice request with a fixed registered state edit or a sham replay.

    The caller supplies a wrapper whose direction and allowed contexts were registered before evaluation. The exact
    state boundary is re-derived from ``build_row`` on each request, with empty-state fallback refused.
    """
    name = "rwkv7_layout_bound_causal_experiment"

    def __init__(self, *, intervention, encoder: Callable, encoder_identity: dict[str, Any],
                 template_id: str, parent_identity: dict[str, Any], vocab_size: int,
                 edit_id: str | None = None, edit_sign: int | None = None,
                 max_context_tokens: int = HARD_CONTEXT_TOKENS):
        if (not callable(encoder) or not callable(getattr(encoder, "identity", None))
                or not isinstance(encoder_identity, dict) or not encoder_identity):
            raise ValueError("identity-pinned callable encoder required")
        if encoder.identity() != encoder_identity:
            raise ValueError("encoder identity differs from its callable identity")
        if template_id not in TEMPLATES or intervention._runtime_identity.get("template_id") != template_id:
            raise ValueError("template identity differs from the registered intervention")
        if not isinstance(parent_identity, dict) or not parent_identity:
            raise ValueError("parent identity required")
        if (intervention._runtime_identity.get("parent_identity") != parent_identity
                or intervention._runtime_identity.get("tokenizer_identity") != encoder_identity):
            raise ValueError("parent or tokenizer identity differs from the registered intervention")
        self.template_identity = TEMPLATES[template_id].identity()
        expected_capture_identity = {"layout_version": LAYOUT_VERSION,
                                     "template_identity": self.template_identity}
        if intervention._runtime_identity.get("capture_identity") != expected_capture_identity:
            raise ValueError("layout/template identity differs from the registered intervention")
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("positive vocabulary size required")
        if (edit_id is None) != (edit_sign is None) or (edit_sign is not None and
                (type(edit_sign) is not int or edit_sign not in (-1, 1))):
            raise ValueError("use sham mode (no edit) or a registered edit with sign -1/+1")
        registration = None if edit_id is None else intervention._registrations.get(edit_id)
        if edit_id is not None and registration is None:
            raise ValueError("edit_id must be registered before backend construction")
        if type(max_context_tokens) is not int or not 1 <= max_context_tokens <= HARD_CONTEXT_TOKENS:
            raise ValueError("max_context_tokens must be within the hard 2048-token limit")
        self.intervention = intervention
        self.encoder = encoder
        self.encoder_identity = deepcopy(encoder_identity)
        self.template_id = template_id
        self.parent_identity = deepcopy(parent_identity)
        self.vocab_size = vocab_size
        self.edit_id, self.edit_sign = edit_id, edit_sign
        self.registration_sha256 = None if registration is None else registration.direction_sha256
        self.max_context_tokens = max_context_tokens

    def revision(self) -> dict[str, Any]:
        identity = {"backend": self.name, "runtime_identity_sha256": self.intervention.runtime_identity_sha256,
                    "parent_identity": self.parent_identity, "encoder_identity": self.encoder_identity,
                    "template_id": self.template_id, "template_identity": self.template_identity,
                    "layout_version": LAYOUT_VERSION,
                    "vocab_size": self.vocab_size, "max_context_tokens": self.max_context_tokens,
                    "edit_id": self.edit_id, "edit_sign": self.edit_sign,
                    "registration_sha256": self.registration_sha256}
        return {"model_id": str(self.parent_identity.get("model_id", "rwkv7")),
                "revision_id": f"{self.name}@{_sha(identity)}", **identity}

    def _row(self, request: dict[str, Any]):
        try:
            self.intervention.verify_runtime_identity()
            if self.encoder.identity() != self.encoder_identity:
                raise ValueError("encoder identity changed")
            if TEMPLATES[self.template_id].identity() != self.template_identity:
                raise ValueError("template identity changed")
            if self.edit_id is not None:
                self.intervention.verify_registered_edit(self.edit_id, self.registration_sha256)
        except (KeyError, TypeError, ValueError):
            raise BackendUnsupported("identity_mismatch", "registered model, encoder, template or edit changed") from None
        questions = request.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise BackendUnsupported("unsupported_layout", "one choice question is required")
        qid, question = next(iter(questions.items()))
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise BackendUnsupported("unsupported_layout", "choice questions only")
        try:
            row = build_row(request, qid, self.encoder, template=self.template_id, layout="ordered",
                            empty_state="error")
        except (KeyError, TypeError, ValueError):
            raise BackendUnsupported("unsupported_layout", "request cannot form the pinned ordered layout") from None
        if row.meta.get("evidence") != "state":
            raise BackendUnsupported("unsupported_layout", "evidence must be nonempty request state")
        start, end = row.state_span
        if (type(start) is not int or type(end) is not int or not (0 < start < end < len(row.ids))
                or not row.continuations or len(row.continuations) != len(row.option_ids)
                or row.option_ids != [oid for oid, _ in canonical_options(question)]):
            raise BackendUnsupported("unsupported_layout", "malformed evidence span or option mapping")
        context_ref = request.get("ssdm", {}).get("context_ref")
        if context_ref not in self.intervention._context_refs:
            raise BackendUnsupported("unsupported_context", "context is outside the registered experiment set")
        identity = _row_identity(request, row, encoder_identity=self.encoder_identity, template_id=self.template_id,
                                 template_identity=self.template_identity)
        full_lengths = [len(row.ids) + len(cont) for cont in row.continuations]
        if (any(type(token) is not int or token < 0 or token >= self.vocab_size for token in row.ids)
                or any(not cont or any(type(token) is not int or token < 0 or token >= self.vocab_size for token in cont)
                       for cont in row.continuations)):
            raise BackendUnsupported("unsupported_layout", "invalid prompt or continuation token IDs")
        if max(full_lengths) > self.max_context_tokens:
            raise BackendUnsupported("context_exceeded", "complete prompt and continuation exceeds context limit")
        return qid, context_ref, row, identity, max(full_lengths)

    def plan(self, request: dict[str, Any]) -> Plan:
        qid, _context_ref, row, _identity, max_full = self._row(request)
        prefix = row.state_span[1]
        suffix_len = len(row.ids) - prefix
        branches = [suffix_len + len(cont) - 1 for cont in row.continuations]
        # As in HFReadoutBackend, this API field reports the full context conditioned on for a row (prompt plus
        # continuation), even though this recurrent backend physically forwards the prefix and each branch apart.
        # The independent max_context_tokens check in _row also enforces RWKV's effective trained context.
        return Plan(inference_calls=1 + len(branches), max_call_input_tokens=max_full,
                    total_input_tokens=prefix + sum(branches),
                    per_question={qid: {"prompt_tokens": len(row.ids), "options": len(row.option_ids)}},
                    prompt_token_labels=None, provenance_applied=False)

    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        import torch

        qid, context_ref, row, row_sha256, max_full = self._row(request)
        prefix_end = row.state_span[1]
        prefix_ids = row.ids[:prefix_end]
        suffix_ids = row.ids[prefix_end:]
        # Decider's callback also accounts an inference call. Invoke it exactly once before each actual model call.
        check_deadline()
        capture = self.intervention.capture(prefix_ids, context_ref=context_ref)
        if capture.prefix_sha256 != _sha(prefix_ids):
            raise BackendUnsupported("backend_error", "capture prefix is not the exact layout prefix")
        source = (self.intervention.apply_registered_edit(capture, edit_id=self.edit_id, sign=self.edit_sign)
                  if self.edit_id is not None else capture)
        scores = {}
        total_tokens = len(prefix_ids)
        for option_id, continuation in zip(row.option_ids, row.continuations):
            branch_ids = suffix_ids + continuation[:-1]
            if not branch_ids:
                raise BackendUnsupported("unsupported_layout", "question suffix and scoring prefix must be nonempty")
            check_deadline()
            replay = self.intervention.replay(source, branch_ids)
            logits = replay.logits
            if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1
                    or logits.shape[1] != len(branch_ids) or logits.shape[2] != self.vocab_size):
                raise BackendUnsupported("backend_error", "replay logits do not match the bound continuation")
            positions = torch.arange(len(continuation), dtype=torch.long) + len(suffix_ids) - 1
            targets = torch.tensor(continuation, dtype=torch.long)
            selected = logits[0].index_select(0, positions).float()
            log_probs = torch.log_softmax(selected, dim=-1)
            score = float(log_probs.gather(1, targets[:, None]).sum().item())
            if not torch.isfinite(torch.tensor(score)):
                raise BackendUnsupported("backend_error", "non-finite continuation score")
            scores[option_id] = score
            total_tokens += len(branch_ids)
        if set(scores) != set(row.option_ids):
            raise BackendUnsupported("backend_error", "scorer did not return the full option domain")
        return ScoreResult(
            logits={qid: scores}, inference_calls=1 + len(row.option_ids), total_input_tokens=total_tokens,
            extras={"layout_version": LAYOUT_VERSION, "layout_row_sha256": row_sha256,
                    "context_ref": context_ref, "state_boundary": "ordered_layout_state_end",
                    "intervention": "sham" if self.edit_id is None else self.edit_id,
                    "registration_sha256": self.registration_sha256,
                    "intervention_sign": self.edit_sign, "max_complete_context_tokens": max_full},
        )
