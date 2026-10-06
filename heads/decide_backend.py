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
* ``revision()`` binds the actual head, typed-output and LoRA state with an exact digest and the supported
  head's behavior-changing configuration, alongside the existing sampled frozen-backbone identity. The model,
  encoder and checkpoint metadata must remain immutable after the first call; build a new backend after training
  or loading a different checkpoint.
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
from .model import DecisionModel


def _trained_state_identity(model: DecisionModel) -> dict[str, Any]:
    """Exact digest of the state learned on top of the separately identified frozen backbone.

    Read state rather than ``requires_grad``: inference commonly freezes every parameter. Hashing LoRA's
    scaling and dropout configuration also distinguishes equal tensors with different adapter behavior.
    """
    from train.lora import LoRALinear  # local import: train depends on heads

    digest = hashlib.sha256()

    def add(label: str, value: Any) -> None:
        encoded = json.dumps([label, value], sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        add("tensor", {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        raw = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    for component in ("head", "typed"):
        for name, tensor in sorted(getattr(model, component).state_dict().items()):
            add_tensor(f"{component}.{name}", tensor)

    lora_modules = [(name, module) for name, module in model.backbone.named_modules()
                    if isinstance(module, LoRALinear)]
    for name, module in sorted(lora_modules):
        dropout = module.dropout.p if isinstance(module.dropout, torch.nn.Dropout) else 0.0
        add("lora_config", {"name": name, "rank": module.rank,
                            "scaling": module.scaling, "dropout": dropout})
        add_tensor(f"backbone.{name}.lora_A", module.lora_A)
        add_tensor(f"backbone.{name}.lora_B", module.lora_B)

    return {"algorithm": "ssdm.trained_state_sha256.exact.v1", "sha256": digest.hexdigest(),
            "head_tensors": len(model.head.state_dict()), "typed_tensors": len(model.typed.state_dict()),
            "lora_modules": len(lora_modules)}


def _head_configuration(head: torch.nn.Module) -> dict[str, Any]:
    """Explicit inference settings for the supported factory heads; unknown heads fail closed."""
    from .heads import (ClefCrossAttentionControl, FinalStateFork, HeadB, IntegratorBank,
                        SnapshotFork)

    if type(head) is IntegratorBank:
        return {"kind": head.name, "channels": head.C, "t_chunk": head.t_chunk}
    if type(head) is FinalStateFork:
        return {"kind": head.name, "hidden": head.mlp[0].out_features, "norm_eps": head.norm.eps}
    if type(head) is SnapshotFork:
        return {"kind": head.name, "channels": head.inc.C, "t_chunk": head.inc.t_chunk,
                "max_snapshots": head.S, "aggregator": head.agg}
    if type(head) is HeadB:
        return {"kind": head.name, "d_state": head.d, "t_chunk": head.t_chunk}
    if type(head) is ClefCrossAttentionControl:
        return {"kind": head.name, "width": head.width,
                "routing_layers": len(head.routing), "decoder_layers": len(head.decoder.layers),
                "routing_heads": [layer.attn.num_heads for layer in head.routing],
                "decoder_heads": [layer.self_attn.num_heads for layer in head.decoder.layers],
                "routing_feedforward": [layer.ff[0].out_features for layer in head.routing],
                "decoder_feedforward": [layer.linear1.out_features for layer in head.decoder.layers]}
    raise ValueError(f"no revision configuration contract for head {type(head).__name__}")


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
            "head_configuration": _head_configuration(self.model.head),
            "backbone": backbone_identity(self.model.backbone),
            "trained_state": _trained_state_identity(self.model),
            "pad_id": self.model.pad_id,
            "fork_mode": self.model.backbone.fork_mode,
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
            # G16: rows carry NO host allowed set (unvetoed logits); the veto is the adapter's conditioned stage
            return [build_row(request, qid, self.encode, template=self.template, layout=self.layout)
                    for qid in request["questions"]]
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
