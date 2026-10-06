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
* ``revision()`` binds the actual head, typed-output and LoRA state with an exact digest and the supported
  head's behavior-changing configuration, alongside the existing sampled frozen-backbone identity. The model,
  encoder and checkpoint metadata must remain immutable after the first call; build a new backend after training
  or loading a different checkpoint.
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
