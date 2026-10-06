"""``decide`` backend over a trained ``DecisionModel`` (frozen backbone + LoRA + head).

The m0 path scores options with the stock LM readout (``decide.backends.hf_readout``); a trained m2
checkpoint is a different scorer and needs its own backend so the eval card, ``safety_eval`` (via
``safety_eval.decide_bridge.DecideBridge``) and the G4/G5 wrappers all drive it through the one
production adapter (validation, canonical option order, policy conditioning, status codes).

Logits are the head's typed-output log-probabilities, one row per question, built with the same
``heads.layout.build_row`` (template, layout, encoder) the model was trained on. Calibration stays the
adapter's job (``Decider(calibration=...)``); this backend never sees evaluation labels. It reports each row's
``evidence_tokens`` (``LayoutRow.evidence_len``, the feature the fitted ``FeatureTemperature`` was trained on) in
``ScoreResult.features``, so the served temperature sees exactly the training-time feature.

* Untrusted text that can only encode to control tokens keeps the encoder's own status code
  (``decide.untrusted.ControlTokenError.code``, as ``hf_readout`` does), not a generic one.
* Non-finite logits raise (a model fault is never served as a decision).
* ``revision()`` is computed once per backend (it hashes the frozen weights); build a new backend per
  checkpoint.
* G16 head (``controlsafe.g16.G16DecisionHead``): the host allowed set (``ssdm.policy_restriction`` and/or
  ``policy.allowed``) is the structural veto INSIDE the decision function. Vetoed options get the finite logit
  ``min(allowed) - G16_VETO_GAP`` (the adapter requires finite logits; their softmax mass is exactly 0 in float64),
  and every question's certificate fields (margin, radii, independence, radius-covers-box, E, vetoed ids; and, with
  ``g16_certify_k``, the full ``controlsafe.g16.certify`` G4-over-E record) are returned in
  ``diagnostics.backend["g16"][qid]``.
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


G16_VETO_GAP = 1000.0


class DecisionModelBackend:
    name = "decision_head"

    def __init__(self, model: DecisionModel, encode: EncodeFn, template: str | Template = "jrt-plain-v1",
                 layout: str = "ordered", checkpoint: dict[str, Any] | None = None, shared_prefix: bool = True,
                 g16_certify_k: int | None = None) -> None:
        self.model = model
        self.g16_certify_k = g16_certify_k
        self.shared_prefix = shared_prefix
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
            rows = [build_row(request, qid, self.encode, template=self.template, layout=self.layout)
                    for qid in request["questions"]]
            if self._g16():
                from controlsafe.g16 import attach_host_allowed

                for r in rows:
                    attach_host_allowed(r, request)
            return rows
        except ControlTokenError as exc:  # before ValueError: it subclasses ValueError
            raise BackendUnsupported(exc.code, str(exc)) from exc
        except ValueError as exc:
            raise BackendUnsupported("tokenization_failed", str(exc)) from exc

    def plan(self, request: dict[str, Any]) -> Plan:
        rows = self._rows(request)
        lens = [len(r.ids) for r in rows]
        return Plan(inference_calls=1 if self._one_pass() else len(rows), max_call_input_tokens=max(lens),
                    total_input_tokens=sum(lens))

    def _g16(self) -> bool:
        return getattr(self.model.head, "name", None) == "g16" and getattr(self.model, "row_head", False)

    def _one_pass(self) -> bool:
        return not getattr(self.model.backbone, "wants_trust", False)

    @torch.no_grad()
    def score(self, request: dict[str, Any], check_deadline: Callable[[], None]) -> ScoreResult:
        rows = self._rows(request)
        was_training = self.model.training
        self.model.eval()
        logits: dict[str, dict[str, float]] = {}
        g16diag: dict[str, dict[str, Any]] = {}
        extra_calls = 0  # G4-over-E origin recomputes (g16_certify_k): reported, so the adapter's call cap sees them
        features = {r.qid: {"evidence_tokens": r.evidence_len, "n_options": r.n_options,
                            "evidence_span": r.meta.get("evidence")} for r in rows}
        try:
            if self._one_pass():
                # all questions of the request in one pass, reading the shared schema + state prefix once
                # (heads.prefix_features; was one full-prompt forward per question, B=1)
                check_deadline()
                lps = list(self.model(rows, shared_prefix=self.shared_prefix).log_probs.float().cpu())
            else:
                # controlsafe trust-gated backbone: per-row calls exactly as before (its protected stream is bitwise
                # stable only for a fixed host batch shape, and it needs per-token trust labels on every call)
                lps = []
                for row in rows:
                    check_deadline()
                    lps.append(self.model([row]).log_probs[0].float().cpu())
                    if self._g16():
                        g16diag[row.qid] = self._g16_certificate(request, row, check_deadline)
                        extra_calls += g16diag[row.qid].get("g4_over_E", {}).get("n_origins_evaluated") or 0
            for row, lp in zip(rows, lps):
                lp = lp[: row.n_options]
                if self._g16():
                    veto = ~torch.tensor(g16diag[row.qid]["allowed_mask"], dtype=torch.bool)
                    ok = lp[~veto]
                    if not (torch.isfinite(ok).all() and bool((lp[veto] == float("-inf")).all())):
                        raise FloatingPointError(f"{row.qid}: non-finite G16 log-probabilities {lp.tolist()}")
                    lp = torch.where(veto, ok.min() - G16_VETO_GAP, lp)
                if not torch.isfinite(lp).all():
                    raise FloatingPointError(f"{row.qid}: non-finite head log-probabilities {lp.tolist()}")
                logits[row.qid] = {oid: float(lp[k]) for k, oid in enumerate(row.option_ids)}
        finally:
            self.model.train(was_training)
        extras: dict[str, Any] = {"head": type(self.model.head).__name__}
        if self._g16():
            extras["g16"] = g16diag
        return ScoreResult(logits=logits, inference_calls=(1 if self._one_pass() else len(rows)) + extra_calls,
                           total_input_tokens=sum(len(r.ids) for r in rows), extras=extras, features=features)

    def _g16_certificate(self, request: dict[str, Any], row: LayoutRow,
                         check_deadline: Callable[[], None] | None = None) -> dict[str, Any]:
        """Certificate fields of the decision just computed (no recompute), plus the full G4-over-E record with
        ``g16_certify_k`` (each origin's E recomputed; P reused exactly)."""
        from controlsafe import g16 as G

        head = self.model.head
        cert = head.certificate(0)
        cert["decision"] = row.option_ids[cert["decision"]]
        cert["vetoed"] = [o for o, m in zip(row.option_ids, cert["allowed_mask"]) if not m]
        if self.g16_certify_k is not None:
            if check_deadline is not None:
                check_deadline()
            rec = G.certify(head, request, row.qid, self.g16_certify_k,
                            lambda r: build_row(r, row.qid, self.encode, template=self.template, layout=self.layout))
            cert["g4_over_E"] = {k: rec.get(k) for k in ("certified", "reason", "k", "n_origins_evaluated",
                                                         "p_cache_misses", "d_tokens", "changed", "decision")}
        return cert
