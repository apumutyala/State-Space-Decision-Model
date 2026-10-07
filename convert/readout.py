"""Decision readout (``decide`` HF-readout plugin) over a partly converted Qwen3.5: the probe9b job's Index arm.

``load_converted_plugin(model_dir, gdn_ckpt, converted)`` loads the stock checkpoint through ``convert.model.load``,
attaches the student GDNs saved by ``convert.pipeline`` (``<ckpt_dir>/prog_<n>.pt`` or ``stage1.pt``), and runs the
layers in ``converted`` as GDN (``student``) and every other attention layer as stock attention (``teacher``).
``converted=[]`` is the stock model through the very same code path, which is the paired baseline arm.

The GDN checkpoint sha256 and the converted-layer list are part of the plugin's ``files``/``notes``, so they enter the
backend revision hash: two arms can never share a revision id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def parse_layers(spec: str | list | None) -> list[int]:
    if spec is None or spec == [] or str(spec).strip().lower() in ("", "none", "stock"):
        return []
    if isinstance(spec, list):
        return sorted(int(x) for x in spec)
    return sorted(int(x) for x in str(spec).replace(";", ",").split(",") if x.strip())


def load_converted_plugin(model_dir: str, gdn_ckpt: str | None, converted, device: str = "cuda",
                          dtype: str = "bfloat16", model_id: str | None = None):
    import torch

    from convert.model import ConvertedStateOps, attach_gdn, init_gdn, load, resolve_model_dir
    from decide.backends.hf_readout import ModelPlugin, _hash_files, _pkg_versions, _snapshot_of
    from decide.backends.state_fork import KV_STATE_TOKEN_BUDGET
    from decide.untrusted import UntrustedEncoder

    layers = parse_layers(converted)
    model, tok, mdir = load(model_dir, device=device, dtype=getattr(torch, dtype))
    if layers and not gdn_ckpt:
        raise ValueError("converted layers given but no GDN checkpoint")
    unknown = set(layers) - set(model.attn_layers)
    if unknown:
        raise ValueError(f"layers {sorted(unknown)} are not attention layers {model.attn_layers}")
    files = {"weights": _hash_files(Path(mdir), ["*.safetensors"]), "tokenizer": _hash_files(Path(mdir), ["tokenizer.json"]),
             "config": _hash_files(Path(mdir), ["config.json"])}
    if layers:
        sd = torch.load(gdn_ckpt, map_location=device)
        missing = [i for i in layers if i not in sd]
        if missing:
            raise ValueError(f"GDN checkpoint {gdn_ckpt} has no weights for layers {missing}")
        for i in layers:
            attach_gdn(model, i, init_gdn(model, i, "random", 0), train_dtype=torch.float32)
            model.swaps[i].gdn.load_state_dict(sd[i])
            model.swaps[i].gdn.requires_grad_(False)
        files["gdn_ckpt"] = _sha(gdn_ckpt)
    model.set_modes({i: "student" for i in layers})
    model.eval()
    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:
        return tok(text, add_special_tokens=False).input_ids if trusted else untrusted.encode(text)

    @torch.no_grad()
    def forward(ids, n_last: int):
        with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16):
            h = model.hidden(ids)
        return model.lm_head(h[:, -n_last:]).float()

    cfg = json.loads((Path(resolve_model_dir(mdir)) / "config.json").read_text(encoding="utf-8"))
    text_cfg = cfg.get("text_config", cfg)
    tag = "stock" if not layers else "conv" + "-".join(map(str, layers))
    return ModelPlugin(
        family="qwen3.5",
        model_id=model_id or f"{cfg.get('_name_or_path') or Path(mdir).name}+{tag}",
        snapshot=_snapshot_of(Path(mdir)),
        encode=encode,
        forward=forward,
        special_ids=untrusted.control_ids,
        files=files,
        max_context=text_cfg.get("max_position_embeddings"),
        dtype=dtype,
        device=device,
        kernel_stack=_pkg_versions("flash-linear-attention", "fla-core", "causal-conv1d", "triton"),
        state_ops=ConvertedStateOps(model, torch.bfloat16 if str(device).startswith("cuda") else None),
        # kept attention layers hold a KV cache that the fork copies once per packed row (job-5cmat POP909-CL OOM)
        fork_state_token_budget=KV_STATE_TOKEN_BUDGET if [i for i in model.attn_layers if i not in layers] else None,
        notes={"model_class": "convert.model.ConvertibleQwen", "converted_layers": layers,
               "attention_layers_kept": [i for i in model.attn_layers if i not in layers],
               "attention_free": False, "path": "convert.readout (same code path for both arms)"},
    )
