"""Export the converted GDN weights of the full-8 run (phase X) or salvage the finished layers of a failed attempt.

    python -m jobs.full8.export_gdn --mode final|salvage --results <ck>/pipeline/results.json --ckpt-dir <ck>/gdn \
        --model-dir <9B snapshot> --rule jobs/full8/rule.json --out-dir /output --meta-out <work>/out/export.json

The source is the checkpoint the pipeline wrote after its LAST recorded step (``prog_<n>.pt``, n = steps in
results.json), which holds the weights the gate accepted (rejected layers already reset). Only the accepted layers are
exported, as ``{layer: GDN state_dict}`` (fp32, exactly as gated): the format ``heads.backbone.q9af_from_stock`` and
``convert.readout`` load.

Names (never ambiguous for the m2 consumer, which refuses anything but the full set anyway):
  final, all 8 attention layers accepted -> gdn_converted.pt   (the Q9-AF artifact; its sha256 is the m2 pin)
  final, 1-7 layers                      -> gdn_partial.pt     (a hybrid Q9-H<8-N>; never a Q9-AF input)
  salvage (a failed / stopped attempt)   -> gdn_salvage_step<n>.pt (what finished before the failure)
  no accepted layer                      -> nothing written (export.json says so)

Verification (exit 6 on any failure): written via a temp file + os.replace; sha256 computed on two independent reads;
re-loaded and compared tensor by tensor with the source checkpoint; every layer strict-loaded into the HF
``Qwen3_5GatedDeltaNet`` built from the snapshot's own text config (meta device: keys and shapes, no memory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def text_config(model_dir):
    from transformers import AutoConfig

    from convert.model import resolve_model_dir

    cfg = AutoConfig.from_pretrained(resolve_model_dir(model_dir))
    return getattr(cfg, "text_config", cfg)


def hf_strict_check(sd: dict, model_dir) -> dict:
    """Every exported layer must load strictly into the HF GDN the m2 consumer builds (q9af_from_stock)."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = text_config(model_dir)
    out = {}
    for i, s in sd.items():
        with torch.device("meta"):
            m = Qwen3_5GatedDeltaNet(cfg, int(i))
        m.load_state_dict(s, strict=True, assign=True)
        out[str(i)] = cfg.layer_types[int(i)]
    if any(t != "full_attention" for t in out.values()):
        raise ValueError(f"exported layers are not attention layers of this snapshot: {out}")
    return out


def layer_map(model_dir, conv) -> dict:
    """What the m2 consumer (heads.backbone.load_q9af_backbone -> q9af_from_stock, cache-fork readout / heads) needs to
    pin and to size the prefix-state fork: the mixer of every layer after conversion (GDN vs attention) and the
    per-sequence decode state of one GDN layer (recurrent S and the causal-conv tail; identical for native and
    converted layers, since the converted GDN is the HF Qwen3_5GatedDeltaNet shape)."""
    c = text_config(model_dir)
    types = ["linear_attention" if (t == "linear_attention" or i in conv) else "full_attention"
             for i, t in enumerate(c.layer_types)]
    conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
    return {"layer_types_after": types,
            "attention_layers_kept": [i for i, t in enumerate(types) if t == "full_attention"],
            "gdn_layers": [i for i, t in enumerate(types) if t == "linear_attention"],
            "gdn_state_shapes_per_sequence": {
                "recurrent_S": [c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim],
                "conv_tail": [conv_dim, c.linear_conv_kernel_dim - 1]},
            "consumer": "heads.backbone.load_q9af_backbone (q9af_from_stock: strict load, needs all attention layers)",
            # block boundaries for a later GDN block activation checkpoint (PREMORTEM P3): q9af_from_stock keeps the HF
            # decoder blocks; only each attention block's mixer is swapped, so blocks = text_model.layers[0..n-1]
            "blocks": {"n": len(types), "module_path": "model.language_model.layers[i]",
                       "block_class": "Qwen3_5DecoderLayer (input_layernorm -> linear_attn GDN -> post_attention_layernorm -> mlp)",
                       "mixer_attr": "linear_attn"}}


def export(mode, results, ckpt_dir, model_dir, rule, out_dir, meta_out) -> dict:
    import torch

    R = json.loads(Path(results).read_text())
    attn = list(json.loads(Path(rule).read_text(encoding="utf-8"))["attention_layers"])
    prog = R.get("progressive") or {}
    n = len(prog.get("steps") or [])
    conv = sorted(int(i) for i in prog.get("converted") or [])
    meta = {"mode": mode, "n_steps": n, "converted": conv, "attention_free": conv == attn, "file": None,
            "results_sha256": sha256(results)}
    if not conv:
        meta["note"] = "no accepted layer: nothing exported"
        return meta
    src = Path(ckpt_dir) / f"prog_{n}.pt"
    full = torch.load(src, map_location="cpu", weights_only=True)
    sd = {i: full[i] for i in conv}
    if mode == "salvage":
        name = f"gdn_salvage_step{n}.pt"
    else:
        name = "gdn_converted.pt" if conv == attn else "gdn_partial.pt"
    p = Path(out_dir) / name
    tmp = p.with_suffix(".pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, p)
    h1, h2 = sha256(p), sha256(p)
    if h1 != h2:
        raise RuntimeError("sha256 differs between two reads")
    back = torch.load(p, map_location="cpu", weights_only=True)
    if sorted(back) != conv or any(sorted(back[i]) != sorted(sd[i]) or any(not torch.equal(back[i][k], sd[i][k])
                                                                           for k in sd[i]) for i in conv):
        raise RuntimeError("re-loaded export differs from the source checkpoint")
    meta.update(file=name, sha256=h1, bytes=p.stat().st_size, source=src.name,
                hf_strict_load=hf_strict_check(back, model_dir),
                dtypes=sorted({str(t.dtype) for s in back.values() for t in s.values()}),
                **layer_map(model_dir, conv))
    try:
        from convert.model import resolve_model_dir

        meta["base_config_sha256"] = sha256(Path(resolve_model_dir(model_dir)) / "config.json")
    except Exception as e:  # noqa: BLE001  (lineage convenience; the m2 pin re-checks it)
        meta["base_config_sha256_error"] = repr(e)[:200]
    (Path(out_dir) / (name + ".sha256")).write_text(f"{h1}  {name}\n")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.full8.export_gdn")
    ap.add_argument("--mode", choices=["final", "salvage"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-out", required=True)
    a = ap.parse_args(argv)
    try:
        meta = export(a.mode, a.results, a.ckpt_dir, a.model_dir, a.rule, a.out_dir, a.meta_out)
        meta["ok"] = True
    except Exception as e:  # noqa: BLE001
        meta = {"mode": a.mode, "ok": False, "error": repr(e)[:500]}
    Path(a.meta_out).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0 if meta["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())
