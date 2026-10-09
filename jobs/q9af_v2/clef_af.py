"""Clef-flash's OFFICIAL joint-head readout over a converted (attention-free) backbone: arm-pair A of jobs/q9af_v2 mode
clefflash (README_clefflash.md).

    python -m decision_index run --engine jobs.q9af_v2.clef_af:ClefAFIndexEngine --option model_dir=<clef-flash snapshot>
        --option release=clef-flash --option path=decide --option gdn_path=reference
        --option gdn_ckpt=<gdn_converted.pt> --option converted=3,7,11,15,19,23,27,31 --rows ... --out ...

``ClefAFIndexEngine`` IS ``baselines.clef_index_engine.ClefIndexEngine`` (Cloudflare's loader, the pinned snapshot
check, the ssdm encoding, ``decide`` over ``ClefBackend``, Clef's joint schema head unchanged, gdn_path=reference on
every native GDN), with one change after loading: each full-attention layer's ``self_attn`` of the merged backbone is
replaced by the converted GDN of the same layer (``StudentMixer``; the attention module is dropped, so its weights
cannot run). The stock arm runs the unmodified ``ClefIndexEngine`` with the same options, so the paired difference is
the backbone conversion alone. The joint head cross-attends over every sequence position (EvidenceRoutingLayer,
TransformerDecoderLayer over ``memory``): the converted SYSTEM is therefore attention-free in its backbone only; the
fully attention-free system is arm-pair B (``q9af-probe-v3w``, the LM-logit v3w readout).

Converted-layer numerics, as the conversion's own readout (convert.readout): fp32 GDN parameters, bf16 autocast on CUDA;
kernel ``chunk`` (convert.gdn's pure-PyTorch chunked form, the torch reference) when gdn_path=reference, else ``fla``.
"""
from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class StudentMixer(nn.Module):
    """A converted GDN in an HF Qwen3.5 decoder layer's ``self_attn`` slot (no cache: Clef runs use_cache=False)."""

    def __init__(self, gdn: nn.Module, layer_idx: int, autocast_dtype=None):
        super().__init__()
        self.gdn = gdn
        self.layer_idx = layer_idx
        self.autocast_dtype = autocast_dtype

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
        if past_key_values is not None:
            raise NotImplementedError("converted Clef backbone: cached decoding is not supported (Clef uses no cache)")
        dev = hidden_states.device.type
        ctx = (torch.autocast(device_type=dev, dtype=self.autocast_dtype) if self.autocast_dtype is not None
               else contextlib.nullcontext())
        with ctx:
            out = self.gdn(hidden_states)
        return out.to(hidden_states.dtype), None


def text_model(clef_model):
    """The Qwen3.5 text decoder inside a ClefModel (the module ClefModel.forward runs for text-only batches)."""
    lm = clef_model.language_model
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    tm = base.model
    return tm.language_model if hasattr(tm, "language_model") else tm


def swap_converted(clef_model, gdn_ckpt: str | Path, converted, kernel: str = "chunk") -> dict[str, Any]:
    """Replace the attention layers ``converted`` of a loaded ClefModel by the GDNs in ``gdn_ckpt`` (strict load).
    Returns the provenance. Refuses anything but ALL attention layers (the diagnostic is the attention-free backbone)."""
    from convert.gdn import GDN
    from convert.readout import parse_layers

    layers = parse_layers(converted)
    tm = text_model(clef_model)
    cfg = tm.config
    attn = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    if layers != attn or attn != ATTENTION_LAYERS:
        raise ValueError(f"converted {layers} must be all attention layers {attn} (expected {ATTENTION_LAYERS})")
    sd = torch.load(gdn_ckpt, map_location="cpu", weights_only=True)
    missing = [i for i in layers if i not in sd]
    if missing:
        raise ValueError(f"{gdn_ckpt}: no weights for layers {missing}")
    ref = next(tm.parameters())
    autocast = torch.bfloat16 if ref.device.type == "cuda" and ref.dtype == torch.bfloat16 else None
    for i in layers:
        layer = tm.layers[i]
        if not hasattr(layer, "self_attn"):
            raise ValueError(f"layer {i} has no self_attn")
        gdn = GDN.from_config(cfg, kernel=kernel)
        gdn.load_state_dict(sd[i], strict=True)
        gdn = gdn.to(device=ref.device, dtype=torch.float32).eval().requires_grad_(False)
        layer.self_attn = StudentMixer(gdn, i, autocast)  # the attention module is released with this assignment
    n_attn_left = sum(1 for m in tm.modules() if type(m).__name__.endswith("Attention")
                      and not type(m).__name__.endswith("GatedDeltaNet"))
    if n_attn_left:
        raise RuntimeError(f"{n_attn_left} attention modules remain in the backbone after the swap")
    return {"converted_layers": layers, "gdn_ckpt_sha256": _sha256(gdn_ckpt), "converted_kernel": kernel,
            "converted_param_dtype": "float32", "autocast": str(autocast) if autocast else None,
            "attention_free_backbone": True, "attention_free_system": False,
            "joint_head_note": "Clef's joint head cross-attends over all positions (not attention-free)"}


try:  # the Clef engine needs the kit + Clef's stack; CPU tests import swap_converted without them
    from baselines.clef_index_engine import ClefIndexEngine
except Exception:  # noqa: BLE001
    ClefIndexEngine = None


if ClefIndexEngine is not None:
    class ClefAFIndexEngine(ClefIndexEngine):
        name = "clef-af"

        def __init__(self, **options: Any):
            super().__init__(**options)
            kernel = "chunk" if self.gdn_path == "reference" else "fla"
            info = swap_converted(self.model, options["gdn_ckpt"], options.get("converted", ""), kernel=kernel)
            rev = self.backend._revision  # Decider reads backend.revision() per response: two arms, two ids
            rev["revision_id"] += f";gdn={info['gdn_ckpt_sha256'][:12]};conv={'-'.join(map(str, info['converted_layers']))}"
            rev["notes"] = {**rev.get("notes", {}), "attention_free": False, "attention_free_backbone": True,
                            "converted": info}
            self.provenance.update({"converted": info, "backend_revision": rev["revision_id"]})
