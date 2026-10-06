"""CPU prototype: a native RWKV-7 parent feeding Track D's learned head.

The caller owns checkpoint verification and strict loading. This module never
loads weights, imports snapshot code, or accesses a service. It consumes the
merged vendored RWKV classes, takes ownership of their base module, and freezes
it in place. The LM vocabulary head is excluded from the decision model.

Use the existing Trainer with lora_rank=0 for head-only growth. A newly created
head is random; construction or a finite forward does not qualify capability.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import torch

from heads.backbone import BackboneSpec, HiddenStateBackbone
from heads.decide_backend import DecisionModelBackend
from heads.layout import EncodeFn
from heads.model import DecisionModel


def rwkv_cpu_parent_encoder(plugin) -> EncodeFn:
    """Bind PR #71's verified CPU plugin encoder to D's encoder identity API.

    Call only after the strict CPU loader succeeds. Its ``files`` pins come
    from verified snapshot bytes and include the local implementation digest.
    The wrapper captures that exact callable and copies the identity; edits to
    the plugin or returned identity cannot silently select another encoder.
    This records loader provenance, not a new artifact verification path.
    """
    from decide.backends.hf_readout import ModelPlugin

    if not isinstance(plugin, ModelPlugin) or (plugin.family, plugin.device, plugin.dtype) != (
            "rwkv7", "cpu", "float32"):
        raise ValueError("requires the verified native RWKV CPU loader plugin")
    if (plugin.untrusted_encoder_id != "rwkv-world-greedy (no control tokens)"
            or plugin.special_ids or not callable(plugin.encode)):
        raise ValueError("requires the loader's no-control-token World encoder")
    if (not isinstance(plugin.model_id, str) or not plugin.model_id
            or not isinstance(plugin.snapshot, str) or not plugin.snapshot):
        raise ValueError("encoder source must identify model and snapshot")
    pins = {role: plugin.files.get(role) for role in ("weights", "config", "tokenizer", "code")}
    if any(not isinstance(pin, str) or len(pin) != 64
           or any(c not in "0123456789abcdef" for c in pin) for pin in pins.values()):
        raise ValueError("encoder source requires complete SHA256 loader pins")
    identity = {"kind": "rwkv-world-greedy", "binding": "track-l-rwkv-cpu-plugin-encoder-v1",
                "model_id": plugin.model_id, "snapshot": plugin.snapshot,
                "files_sha256": pins, "untrusted_encoding": plugin.untrusted_encoder_id,
                "special_ids": []}
    parent_encode = plugin.encode

    def encode(text: str, trusted: bool = True) -> list[int]:
        return parent_encode(text, trusted)

    encode.identity = lambda: deepcopy(identity)  # type: ignore[attr-defined]
    return encode


def rwkv_decision_backend(parent, encode: EncodeFn, *, source_identity: dict[str, Any],
                          head: str = "snapshot", head_kwargs: dict | None = None,
                          seed: int = 0, template: str = "jrt-rwkv-g1-v1",
                          layout: str = "ordered", fork_mode: str = "cache") -> DecisionModelBackend:
    """Attach an attention-free decision head to an already loaded CPU FP32 parent.

    ``source_identity`` is caller-supplied provenance, not authentication. It
    must identify model_id and snapshot (or snapshot_revision). The encoder's
    identity must name the same model and snapshot. Shared files_sha256 roles
    must agree when supplied. These are provenance consistency checks; the
    caller still owns verification of actual checkpoint/tokenizer bytes.
    A returned backend serves the ordinary D adapter and L planner interfaces.
    """
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM, Rwkv7Model

    if not isinstance(parent, (Rwkv7Model, Rwkv7ForCausalLM)):
        raise TypeError("requires the merged vendored native RWKV-7 model")
    if head not in ("integrator", "snapshot", "head_b", "fork"):
        raise ValueError("prototype requires an attention-free decision head")
    if (not isinstance(source_identity, dict) or not isinstance(source_identity.get("model_id"), str)
            or not source_identity["model_id"]):
        raise ValueError("source_identity must identify model_id")
    # Freeze an independent JSON value so later caller edits cannot rewrite lineage.
    identity = json.loads(json.dumps(source_identity, sort_keys=True, allow_nan=False))
    encoder_identity = getattr(encode, "identity", None)
    encoder_info = encoder_identity() if callable(encoder_identity) else None
    if not isinstance(encoder_info, dict) or not encoder_info:
        raise ValueError("encoder must expose a nonempty identity()")
    snapshot = identity.get("snapshot", identity.get("snapshot_revision"))
    if not isinstance(snapshot, str) or not snapshot:
        raise ValueError("source_identity must identify snapshot or snapshot_revision")
    if "snapshot" in identity and "snapshot_revision" in identity and identity["snapshot_revision"] != snapshot:
        raise ValueError("source_identity snapshot aliases disagree")
    if encoder_info.get("model_id") != identity["model_id"]:
        raise ValueError("encoder model_id does not match source parent identity")
    if encoder_info.get("snapshot") != snapshot:
        raise ValueError("encoder snapshot does not match source parent identity")
    source_files = identity.get("files_sha256", {})
    encoder_files = encoder_info.get("files_sha256", {})
    if not isinstance(source_files, dict) or not isinstance(encoder_files, dict):
        raise ValueError("files_sha256 identity pins must be objects")
    for role in source_files.keys() & encoder_files.keys():
        if source_files[role] != encoder_files[role]:
            raise ValueError(f"encoder {role} pin does not match source parent identity")
    for value in list(parent.parameters()) + list(parent.buffers()):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("prototype requires an already loaded CPU FP32 parent")
    base = parent.rwkv7 if isinstance(parent, Rwkv7ForCausalLM) else parent
    c = base.config
    d, n = c.hidden_size, c.num_hidden_layers
    resident = sum(p.numel() for p in base.parameters())
    nonembedding = resident - base.emb.weight.numel()
    spec = BackboneSpec(
        name=f"track-l-native-rwkv7-{d}x{n}", family="rwkv7", hf_id=identity["model_id"],
        params=nonembedding, params_memory=resident, n_layers=n, d_model=d,
        lora_groups=(("att", r"att\.(receptance|key|value|output)$", n * 4),
                     ("ffn", r"ffn\.(key|value)$", n * 2)),
        capture="hidden_states", attention_free=True,
        notes="Native CPU parent bridge prototype; capability unqualified.",
    )
    backbone = HiddenStateBackbone(base, spec=spec, pad_id=0, fork_mode=fork_mode)
    backbone.provenance = {"track_l_parent": identity, "scope": "unqualified native parent bridge"}
    # Validate construction before changing the caller's trainability flags.
    model = DecisionModel(backbone, head, deepcopy(head_kwargs), pad_id=0, seed=seed)
    backend = DecisionModelBackend(model, encode, template=template, layout=layout)
    parent.eval().requires_grad_(False)
    return backend
