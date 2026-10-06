"""``decide`` backend over a trained ``DecisionModel`` (frozen backbone + LoRA + head).

The m0 path scores options with the stock LM readout (``decide.backends.hf_readout``); a trained m2
checkpoint is a different scorer and needs its own backend so the eval card, ``safety_eval`` (via
``safety_eval.decide_bridge.DecideBridge``) and the G4/G5 wrappers all drive it through the one
production adapter (validation, canonical option order, policy conditioning, status codes).

Logits are the head's typed-output log-probabilities, one row per question, built with the same
``heads.layout.build_row`` (template, layout, encoder) the model was trained on. Calibration stays the
adapter's job (``Decider(calibration=...)``); this backend never sees evaluation labels.

* Untrusted text that can only encode to control tokens keeps the encoder's own status code
  (``decide.untrusted.ControlTokenError.code``, as ``hf_readout`` does), not a generic one.
* Non-finite logits raise (a model fault is never served as a decision).
* ``revision()`` is computed once per backend (it hashes the frozen weights); build a new backend per
  checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import torch

from decide import versions
from decide.backends.base import BackendUnsupported, Plan, ScoreResult
from decide.prompt import TEMPLATES, Template
from decide.untrusted import ControlTokenError

from .layout import EncodeFn, LayoutRow, build_row
from .model import DecisionModel


class DecisionModelBackend:
    name = "decision_head"

    def __init__(self, model: DecisionModel, encode: EncodeFn, template: str | Template = "jrt-plain-v1",
                 layout: str = "ordered", checkpoint: dict[str, Any] | None = None) -> None:
        self.model = model
        self.encode = encode
        self.template = TEMPLATES[template] if isinstance(template, str) else template
        self.layout = layout
        self.checkpoint = checkpoint or {}
        self._revision: dict[str, Any] | None = None

    def revision(self) -> dict[str, Any]:
        if self._revision is not None:
            return self._revision
        from train.lineage import backbone_identity  # local import: train depends on heads, not vice versa

        enc = getattr(self.encode, "identity", None)
        ident = {
            "backend": self.name,
            "head": type(self.model.head).__name__,
            "backbone": backbone_identity(self.model.backbone),
            "encoder": enc() if callable(enc) else {"kind": "unidentified"},
            "prompt_template": self.template.identity(),
            "layout": self.layout,
            "checkpoint": self.checkpoint,
            "adapter_version": versions.ADAPTER_VERSION,
        }
        h = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
        self._revision = {"revision_id": f"{self.name}:{ident['head']}@{h[:12]}",
                          "model_id": ident["backbone"]["spec"], **ident}
        return self._revision

    def _rows(self, request: dict[str, Any]) -> list[LayoutRow]:
        try:
            return [build_row(request, qid, self.encode, template=self.template, layout=self.layout)
                    for qid in request["questions"]]
        except ControlTokenError as exc:  # before ValueError: it subclasses ValueError
            raise BackendUnsupported(exc.code, str(exc)) from exc
        except ValueError as exc:
            raise BackendUnsupported("tokenization_failed", str(exc)) from exc

    def plan(self, request: dict[str, Any]) -> Plan:
        rows = self._rows(request)
        lens = [len(r.ids) for r in rows]
        return Plan(inference_calls=len(rows), max_call_input_tokens=max(lens), total_input_tokens=sum(lens))

    @torch.no_grad()
    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        rows = self._rows(request)
        was_training = self.model.training
        self.model.eval()
        logits: dict[str, dict[str, float]] = {}
        try:
            for row in rows:
                check_deadline()
                lp = self.model([row]).log_probs[0].float().cpu()
                if not torch.isfinite(lp[: row.n_options]).all():
                    raise FloatingPointError(f"{row.qid}: non-finite head log-probabilities {lp.tolist()}")
                logits[row.qid] = {oid: float(lp[k]) for k, oid in enumerate(row.option_ids)}
        finally:
            self.model.train(was_training)
        return ScoreResult(logits=logits, inference_calls=len(rows), total_input_tokens=sum(len(r.ids) for r in rows),
                           extras={"head": type(self.model.head).__name__})
