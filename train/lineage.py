"""Resume lineage: what a checkpoint was trained on, pinned at save and validated at load.

A checkpoint stores only trainable weights (LoRA + head), so resuming it against a different
frozen backbone, tokenizer, prompt layout or training set would silently produce a run that
never existed. ``lineage(...)`` records:

* ``backbone``  spec name, model class, a hash of the model config, ``name_or_path`` and HF
                commit hash when present, and a digest of the FROZEN base weights (LoRA tensors
                excluded; LoRA-wrapped names normalised). ``full=True`` hashes every byte;
                the default hashes names, shapes, dtypes and a strided sample of 3 x 4096
                values per tensor (seconds on a 9B model; catches a different checkpoint, not a
                single flipped weight).
* ``tokenizer`` the encoder identity stamped on every row by ``items_from_records``.
* ``layout``    layout kind(s), template id(s), ``decide.prompt.BODY_VERSION`` and ``LAYOUT_VERSION``.
* ``data``      sha256 over the train items in list order (id, label, token ids, option ids),
                plus the data-order function version. The order itself is a pure function of
                (seed, epoch, len(items)), so equal hashes imply the same batches.

CUDA RNG state is saved and restored by the trainer when CUDA is available.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import torch

from decide.prompt import BODY_VERSION
from heads.layout import LAYOUT_VERSION

ORDER_FN = "randperm(seed*100003+epoch); batches = consecutive bs-slices; partial tail dropped"


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _config_dict(model) -> dict:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
    elif hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        d = asdict(cfg)
    else:
        d = dict(vars(cfg))
    return {k: v for k, v in d.items() if not k.startswith("_") and k not in ("transformers_version",)}


def weights_digest(module: torch.nn.Module, full: bool = False, sample: int = 4096) -> str:
    h = hashlib.sha256()
    named = []
    for name, p in module.named_parameters():
        if name.endswith(("lora_A", "lora_B")):
            continue
        named.append((name.replace(".base.", "."), p))
    for name, p in sorted(named, key=lambda x: x[0]):
        h.update(f"{name}|{tuple(p.shape)}|{p.dtype}".encode())
        if p.device.type == "meta":
            h.update(b"meta")
            continue
        flat = p.detach().reshape(-1)
        if not full and flat.numel() > 3 * sample:
            stride = max(1, flat.numel() // sample)
            flat = torch.cat([flat[:sample], flat[::stride][:sample], flat[-sample:]])
        h.update(flat.float().cpu().numpy().tobytes())
    return h.hexdigest()


def backbone_identity(backbone, full: bool = False) -> dict[str, Any]:
    m = backbone.model
    cfg = getattr(m, "config", None)
    return {
        "spec": getattr(backbone.spec, "name", None),
        "class": type(m).__name__,
        "config_sha256": _h(_config_dict(m)),
        "name_or_path": getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None),
        "commit_hash": getattr(cfg, "_commit_hash", None),
        "capture": backbone.capture,
        "layer": backbone.layer,
        "weights_digest": weights_digest(m, full=full),
        "weights_digest_mode": "full" if full else "sampled",
        **({"provenance": backbone.provenance} if getattr(backbone, "provenance", None) else {}),
        **({"controlsafe": backbone.controlsafe} if getattr(backbone, "controlsafe", None) else {}),
    }


def items_identity(items: Sequence) -> dict[str, Any]:
    h = hashlib.sha256()
    for it in items:
        r = it.row
        value = [it.id, it.label, r.ids, r.option_ids, r.continuations, r.option_seqs, r.qtype]
        axes = getattr(it, "axis_target", None)
        if axes is not None:
            value.append({"axis_target": list(axes)})  # opt-in sidecar identity; legacy digests unchanged
        h.update(_h(value).encode())
    return {"n": len(items), "sha256": h.hexdigest(), "order_fn": ORDER_FN}


def layout_identity(items: Sequence) -> dict[str, Any]:
    kinds = sorted({(it.row.layout, it.row.meta.get("template")) for it in items})
    return {"layouts": kinds, "body_version": BODY_VERSION, "layout_version": LAYOUT_VERSION}


def tokenizer_identity(items: Sequence) -> list:
    ids = {json.dumps(it.row.meta.get("encoder"), sort_keys=True, default=str) for it in items}
    return sorted(ids)


def lineage(backbone, items: Sequence, full_weights: bool = False) -> dict[str, Any]:
    return {
        "backbone": backbone_identity(backbone, full=full_weights),
        "tokenizer": tokenizer_identity(items),
        "layout": layout_identity(items),
        "data": items_identity(items),
    }


def diff(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Top-level lineage keys (and backbone sub-keys) that differ."""
    out = []
    for k in sorted(set(saved) | set(current)):
        a, b = saved.get(k), current.get(k)
        if k == "backbone" and isinstance(a, dict) and isinstance(b, dict):
            out += [f"backbone.{kk}" for kk in sorted(set(a) | set(b)) if a.get(kk) != b.get(kk)]
        elif json.dumps(a, sort_keys=True, default=str) != json.dumps(b, sort_keys=True, default=str):
            out.append(k)
    return out
