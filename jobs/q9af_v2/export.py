"""Phase X of jobs/q9af_v2: export the 8 converted GDN layers as ``gdn_converted.pt`` (the Q9-AF artifact) + export.json.

    python -m jobs.q9af_v2.export --src <ckpt_dir>/final.pt --model-dir <9B snapshot> --out-dir /output \
        --meta-out <work>/out/export.json [--mode final|salvage]

Same file format and export.json fields as jobs/full8/export_gdn.py (``{layer: GDN state_dict}``, fp32), so the existing
consumers take it unchanged: ``jobs.q9af_cell.pin fill`` / ``hooks.check_export`` (needs ok, mode final, sha256,
attention_free, file, converted, attention_layers_kept, base_config_sha256) and ``heads.backbone.q9af_from_stock``.
Verification (exit 6 on any failure): temp file + os.replace, sha256 on two reads, tensor-by-tensor re-load equality,
every layer strict-loaded into the HF ``Qwen3_5GatedDeltaNet`` of the snapshot's own config (meta device).
A salvage export (a stage-2 checkpoint of a failed attempt) is named ``gdn_salvage.pt`` and is never a Q9-AF input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from jobs.full8.export_gdn import hf_strict_check, layer_map, sha256, text_config


def export(src, model_dir, out_dir, mode="final") -> dict:
    import torch

    from convert.model import resolve_model_dir

    c = text_config(model_dir)
    attn = [i for i, t in enumerate(c.layer_types) if t == "full_attention"]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    conv = sorted(int(i) for i in sd)
    if conv != attn:
        raise ValueError(f"checkpoint layers {conv} != the snapshot's attention layers {attn}")
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
    meta = {"mode": mode, "converted": conv, "attention_free": mode == "final" and conv == attn, "file": name,
            "sha256": h1, "bytes": p.stat().st_size, "source": Path(src).name, "source_sha256": sha256(src),
            "hf_strict_load": hf_strict_check(back, model_dir),
            "dtypes": sorted({str(t.dtype) for s in back.values() for t in s.values()}),
            "base_config_sha256": sha256(Path(resolve_model_dir(model_dir)) / "config.json"),
            "producer": "jobs/q9af_v2 (convert.joint)", **layer_map(model_dir, conv)}
    (Path(out_dir) / (name + ".sha256")).write_text(f"{h1}  {name}\n")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_v2.export")
    ap.add_argument("--src", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-out", required=True)
    ap.add_argument("--mode", choices=["final", "salvage"], default="final")
    a = ap.parse_args(argv)
    try:
        meta = export(a.src, a.model_dir, a.out_dir, a.mode)
        meta["ok"] = True
    except Exception as e:  # noqa: BLE001
        meta = {"mode": a.mode, "ok": False, "error": repr(e)[:500]}
    Path(a.meta_out).write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0 if meta["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())
