"""Export a Q9-AF v3 GDN checkpoint (stage A or the final job-B model): jobs/q9af_v2/export.py's verified format
(``{layer: GDN state_dict}``, fp32, sha256 on two reads, tensor re-load equality, sidecar ``.sha256``), extended to the
v3 WIDENED key dim: every layer is strict-loaded into the HF ``Qwen3_5GatedDeltaNet`` of the snapshot's config with
``linear_key_head_dim`` set to the dk its weights imply (``gdn_dk``, recorded per layer; 128 = the config = v2's shape).

    python -m jobs.q9af_v3.export --src <ckpt>.pt --model-dir <9B snapshot> --out-dir <dir> --meta-out <json> \
        [--mode final|salvage] [--stage A|B|S] [--layers 3,7,...]

``--layers`` exports a subset (a STOPPED job A's hybrid, always as ``--mode salvage``). Consumers: convert.readout
(``attach_from_state`` sizes the GDN from the weights); jobs.q9af_cell / heads.backbone assume the config's dk and must
read ``gdn_dk`` first (a widened export is NOT drop-in for them).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from jobs.full8.export_gdn import layer_map, sha256, text_config


def _dk(cfg, s) -> int:
    key_dim = (s["in_proj_qkv.weight"].shape[0] - cfg.linear_num_value_heads * cfg.linear_value_head_dim) // 2
    return key_dim // cfg.linear_num_key_heads


def strict_check(sd: dict, model_dir) -> dict:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = text_config(model_dir)
    out = {}
    for i, s in sd.items():
        c = copy.deepcopy(cfg)
        c.linear_key_head_dim = _dk(cfg, s)
        with torch.device("meta"):
            m = Qwen3_5GatedDeltaNet(c, int(i))
        m.load_state_dict(s, strict=True, assign=True)
        if cfg.layer_types[int(i)] != "full_attention":
            raise ValueError(f"layer {i} is not an attention layer of this snapshot")
        out[str(i)] = {"layer_type": cfg.layer_types[int(i)], "gdn_dk": c.linear_key_head_dim}
    return out


def export(src, model_dir, out_dir, mode="final", layers=None) -> dict:
    import torch

    from convert.model import resolve_model_dir

    c = text_config(model_dir)
    attn = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = {int(i): v for i, v in sd.items()}
    conv = sorted(layers) if layers else sorted(sd)
    if not set(conv) <= set(sd) or not set(conv) <= set(attn):
        raise ValueError(f"layers {conv}: checkpoint has {sorted(sd)}, attention layers {attn}")
    if mode == "final" and conv != attn:
        raise ValueError(f"a final export must cover all attention layers {attn} (got {conv})")
    name = "gdn_converted.pt" if mode == "final" else "gdn_salvage.pt"
    p = Path(out_dir) / name
    tmp = p.with_suffix(".pt.tmp")
    torch.save({i: sd[i] for i in conv}, tmp)
    os.replace(tmp, p)
    h1, h2 = sha256(p), sha256(p)
    if h1 != h2:
        raise RuntimeError("sha256 differs between two reads")
    back = torch.load(p, map_location="cpu", weights_only=True)
    if sorted(back) != conv or any(sorted(back[i]) != sorted(sd[i]) or any(not torch.equal(back[i][k], sd[i][k])
                                                                           for k in sd[i]) for i in conv):
        raise RuntimeError("re-loaded export differs from the source checkpoint")
    strict = strict_check(back, model_dir)
    meta = {"mode": mode, "converted": conv, "attention_free": mode == "final" and conv == attn, "file": name,
            "sha256": h1, "bytes": p.stat().st_size, "source": Path(src).name, "source_sha256": sha256(src),
            "hf_strict_load": strict, "gdn_dk": {k: v["gdn_dk"] for k, v in strict.items()},
            "widened": any(v["gdn_dk"] != c.linear_key_head_dim for v in strict.values()),
            "dtypes": sorted({str(t.dtype) for s in back.values() for t in s.values()}),
            "base_config_sha256": sha256(Path(resolve_model_dir(model_dir)) / "config.json"), **layer_map(model_dir, conv)}
    (Path(out_dir) / (name + ".sha256")).write_text(f"{h1}  {name}\n")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_v3.export")
    ap.add_argument("--src", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--mode", choices=["final", "salvage"], default="final")
    ap.add_argument("--stage", choices=["A", "B", "S"], default="B")
    ap.add_argument("--layers", default=None, help="comma-separated subset (salvage of a STOPPED hybrid)")
    a = ap.parse_args(argv)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",") if x.strip()] if a.layers else None
    try:
        meta = export(a.src, a.model_dir, a.out_dir, a.mode, layers)
        meta.update({"ok": True, "producer": f"jobs/q9af_v3 stage {a.stage} (convert.v3)", "stage": a.stage})
    except Exception as e:  # noqa: BLE001
        meta = {"mode": a.mode, "ok": False, "stage": a.stage, "error": repr(e)[:500]}
    Path(a.meta_out).write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("hf_strict_load", "layer_types_after")}))
    return 0 if meta["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())
