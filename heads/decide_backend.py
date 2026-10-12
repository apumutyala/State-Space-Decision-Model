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
* G16 head (``controlsafe.g16.G16DecisionHead``): the logits returned are the UNVETOED ``b + a.E`` (finite, every
  supplied option), so ``raw`` / ``served`` are the model's distribution and ``served_config.policy_masking:
  none`` stays true. The veto source is ONLY the validated envelope ``ssdm.policy_restriction``
  (``controlsafe.g16.envelope_allowed_ids``; a top-level ``request["policy"]`` is ignored). The adapter turns it
  into ``policy_conditioned_probabilities`` (stage ``present``); G16's vetoed decision is exactly that stage's
  argmax: ``softmax`` of ``b + a.E`` over the allowed set (``-inf`` elsewhere) IS the served distribution
  conditioned on the allowed set (at T = 1 the distributions are equal; with a served temperature the argmax is).
  So for G16, executing the policy-conditioned stage == executing the vetoed decision, and the served distribution
  is never replaced (G2(d)). Every question's certificate (computed WITH the veto: margin, radii, independence,
  radius-covers-box, E, ``unvetoed_logits``, ``vetoed``, ``veto_source``, ``decision_stage``; with
  ``g16_certify_k`` the full ``controlsafe.g16.certify`` G4-over-E record) is in ``diagnostics.backend["g16"][qid]``.
  The learned P-gate (``p_gate=True``) is a model-internal mask that would make logits non-finite: refused here.
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
from .symmetric import SYM_LAYOUTS, SYM_VERSION, row_input_tokens, sym_tie_info
from .model import DecisionModel


def _template_identity(template: Template) -> dict[str, str]:
    from decide.prompt import G1V2_ID

    from decide.prompt_g1v3 import G1V2F_ID, G1V3W_ID
    from decide.v3w_generic import is_generic_template, spec_for_template

    from decide.shim_format import is_shim_template
    from decide.shim_format import spec_for_template as shim_spec

    if is_shim_template(template):  # the trainable H2O shim format: the bound spec's identity
        return shim_spec(template.id).identity()
    if is_generic_template(template):  # generic v3w: the bound spec's identity (the parent readout's prompt identity)
        return spec_for_template(template.id).identity()
    if template.id == G1V3W_ID:  # v3w: the R3 words readout's identity (decide.backends.rwkv_g1_v3 mode "words")
        from decide.backends.rwkv_g1_v3 import renderer_identity as v3_identity

        return v3_identity("words")
    if template.id == G1V2F_ID:  # v2f: the fixed-instruction v2 renderer's identity (decide.backends.rwkv_g1_v3)
        from decide.backends.rwkv_g1_v3 import renderer_identity as v3_identity

        return v3_identity("v2f")
    if template.id == G1V2_ID:  # D3 / R1: the C1 v2 renderer identity (head, tail, system, bos, codebook, canary)
        from decide.backends.rwkv_g1_v2 import renderer_identity

        return renderer_identity()
    return template.identity()


class DecisionModelBackend:
    name = "decision_head"

    def __init__(self, model: DecisionModel, encode: EncodeFn, template: str | Template = "jrt-plain-v1",
                 layout: str = "ordered", checkpoint: dict[str, Any] | None = None, shared_prefix: bool = True,
                 g16_certify_k: int | None = None) -> None:
        self.model = model
        if getattr(model.head, "name", None) == "g16" and getattr(model.head, "gate", None) is not None:
            raise ValueError("decide backend: a G16 head with the learned P-gate is not servable (non-finite logits)")
        self.g16_certify_k = g16_certify_k
        self.shared_prefix = shared_prefix
        self.encode = encode
        # (request key, rows) of the last request: decide.adapter calls plan() then score() on the SAME request, and
        # build_row is a pure function of (request, qid, encoder, template, layout), so score reuses plan's rows
        # instead of tokenizing every question twice (JEV1-4B r2: the native readout's per-decision overhead)
        self._row_cache: tuple[str, list[LayoutRow]] | None = None
        from .layout import resolve_template

        self.template = resolve_template(template)
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
            "prompt_template": _template_identity(self.template),
            "layout": self.layout,
            **({"sym_version": SYM_VERSION} if self.layout in SYM_LAYOUTS else {}),  # the sym row renderer's version
            "checkpoint": self.checkpoint,
            "adapter_version": versions.ADAPTER_VERSION,
        }
        h = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
        self._revision = {"revision_id": f"{self.name}:{ident['head']}@{h[:12]}",
                          "model_id": ident["backbone"]["spec"], **ident}
        return self._revision

    @staticmethod
    def _request_key(request: dict[str, Any]) -> str:
        """Content key of a request (canonical JSON): a mutated or different request never hits the row cache."""
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, default=repr)
                              .encode("utf-8")).hexdigest()

    def _rows(self, request: dict[str, Any]) -> list[LayoutRow]:
        key = self._request_key(request)
        if self._row_cache is not None and self._row_cache[0] == key:
            return self._row_cache[1]
        try:
            # G16: rows carry NO host allowed set (unvetoed logits); the veto is the adapter's conditioned stage
            rows = [build_row(request, qid, self.encode, template=self.template, layout=self.layout)
                    for qid in request["questions"]]
        except ControlTokenError as exc:  # before ValueError: it subclasses ValueError
            raise BackendUnsupported(exc.code, str(exc)) from exc
        except ValueError as exc:
            raise BackendUnsupported("tokenization_failed", str(exc)) from exc
        self._row_cache = (key, rows)
        return rows

    def plan(self, request: dict[str, Any]) -> Plan:
        rows = self._rows(request)
        lens = [row_input_tokens(r) for r in rows]  # sym layouts also feed their option continuations
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
                            "evidence_span": r.meta.get("evidence"),
                            **({"sym": sym_tie_info(r)} if sym_tie_info(r) is not None else {})} for r in rows}
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
                if not torch.isfinite(lp).all():
                    raise FloatingPointError(f"{row.qid}: non-finite head log-probabilities {lp.tolist()}")
                logits[row.qid] = {oid: float(lp[k]) for k, oid in enumerate(row.option_ids)}
                canon = (row.meta or {}).get("canon")
                if canon and canon.get("tie_sets"):  # JEV3-LM: duplicates of one content share their probability
                    from decide.canon_shim import equalize_ties

                    logits[row.qid] = equalize_ties(logits[row.qid], canon["tie_sets"])
        finally:
            self.model.train(was_training)
        extras: dict[str, Any] = {"head": type(self.model.head).__name__}
        if self._g16():
            extras["g16"] = g16diag
        return ScoreResult(logits=logits, inference_calls=(1 if self._one_pass() else len(rows)) + extra_calls,
                           total_input_tokens=sum(row_input_tokens(r) for r in rows), extras=extras, features=features)

    def _g16_certificate(self, request: dict[str, Any], row: LayoutRow,
                         check_deadline: Callable[[], None] | None = None) -> dict[str, Any]:
        """Certificate fields of the decision just computed (no recompute), plus the full G4-over-E record with
        ``g16_certify_k`` (each origin's E recomputed; P reused exactly)."""
        from controlsafe import g16 as G

        head = self.model.head
        ok = G.envelope_allowed_ids(request, row.qid)
        cert = head.certificate(0, allowed=G.host_allowed_mask(request, row.qid, row.option_ids,
                                                                G.envelope_allowed_ids))
        cert["decision"] = row.option_ids[cert["decision"]]
        cert["vetoed"] = [o for o, m in zip(row.option_ids, cert["allowed_mask"]) if not m]
        cert["veto_source"] = "ssdm.policy_restriction" if ok is not None else None
        cert["decision_stage"] = "policy_conditioned_probabilities" if ok is not None else "served_probabilities"
        if self.g16_certify_k is not None:
            if check_deadline is not None:
                check_deadline()
            rec = G.certify(head, request, row.qid, self.g16_certify_k,
                            lambda r: build_row(r, row.qid, self.encode, template=self.template, layout=self.layout),
                            allowed_ids=G.envelope_allowed_ids)
            cert["g4_over_E"] = {k: rec.get(k) for k in ("certified", "reason", "k", "n_origins_evaluated",
                                                         "p_cache_misses", "d_tokens", "changed", "decision")}
        return cert
