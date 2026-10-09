"""INFERENCE export of a Trainer checkpoint: the trainable tensors (LoRA + head) only, as bf16 safetensors + a JSON
sidecar. No optimizer / scheduler / RNG state, so the artifact stays under the store's single-PUT cap (job-igcu4: a
LoRA r256 ``.pt`` with Adam state was ~14 GB and the whole upload failed). The full ``.pt`` stays in the platform
checkpoint dir ($GMN_CHECKPOINT_DIR), never in $GMN_OUTPUT_DIR.

    python -X utf8 -m train.ckpt_export inference --ckpt <ck>/cell/ckpt-step0001439.pt --out-dir <out>/ckpt

writes, in ``--out-dir``::

    trainable-step0001439.safetensors          trainable tensors (bf16 by default); ONE metadata key "ssdm" = sorted
                                               JSON {format, config_hash, step, source_ckpt_sha256, dtype}
                                               (one key: safetensors writes a metadata map in random order)
    trainable-step0001439.json                 sidecar: sha256 of the safetensors + everything Trainer.load checks
                                               (config_hash, lineage) or restores (step, epoch, cursor, history)
    trainable-step0001439.safetensors.sha256   ``sha256sum`` line
    latest.json                                {"path": "trainable-step0001439.safetensors", "format": ..., ...}

``Trainer.load`` accepts this export (eval only: ``Trainer.resume`` refuses it, there is no optimizer state), so
``train.m2_gates`` / ``jobs.m2_first_cell.fulldev_eval`` / ``screen`` restore it exactly like a ``.pt``, and
``jobs.m2_first_cell.checkpoint_loader`` installs it from an artifact. bf16 storage: the cell runs under bf16
autocast, so every LoRA / head matmul already sees bf16 operands; the dtype is recorded in both files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

FORMAT = "ssdm-trainable-v1"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
NAME_RE = re.compile(r"trainable-step[0-9]+\.safetensors")
SIDECAR_KEYS = ("step", "epoch", "cursor", "config", "config_hash", "lineage", "init_seed", "history", "backbone",
                "lora_modules", "guard")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every OS (`sha256sum -c`, stable sha256s)
        f.write(text)


def is_inference(path: str | Path) -> bool:
    return str(path).endswith(".safetensors")


def sidecar_path(st_path: str | Path) -> Path:
    p = Path(st_path)
    return p.with_name(p.name[: -len(".safetensors")] + ".json")


def _jsonable(x: Any) -> Any:
    """JSON-safe copy (tuples -> lists; tensors / arrays / other objects -> str): the sidecar never pickles."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def export_inference(ckpt: str | Path, out_dir: str | Path, dtype: str = "bf16") -> dict[str, Any]:
    """Write the inference export of ``ckpt`` (a Trainer ``.pt``, or an existing export, which is copied) into
    ``out_dir``; returns the sidecar record."""
    from safetensors.torch import save_file

    ckpt, out = Path(ckpt), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if is_inference(ckpt):  # eval-only modes installed an export: re-publish it unchanged
        side = json.loads(sidecar_path(ckpt).read_text(encoding="utf-8"))
        for f in (ckpt, sidecar_path(ckpt)):
            if f.parent.resolve() != out.resolve():
                shutil.copyfile(f, out / f.name)
        name = ckpt.name
    else:
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} not in {sorted(DTYPES)}")
        # the job's own checkpoint (written by Trainer.save in this run), loaded as Trainer.load does; mmap keeps a
        # 14 GB r256 file out of RAM, only the trainable tensors are materialised
        st = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
        tr = st["trainable"]
        if not isinstance(tr, dict) or not tr or not all(isinstance(v, torch.Tensor) for v in tr.values()):
            raise ValueError(f"{ckpt}: no trainable tensor dict")
        tensors = {k: v.detach().to(DTYPES[dtype]).contiguous() for k, v in tr.items()}
        step = int(st["step"])
        name = f"trainable-step{step:07d}.safetensors"
        src_sha = sha256_file(ckpt)
        tmp = out / (name + ".tmp")
        meta = {"format": FORMAT, "config_hash": str(st["config_hash"]), "step": step, "source_ckpt_sha256": src_sha,
                "dtype": dtype}
        save_file(tensors, str(tmp), metadata={"ssdm": json.dumps(meta, sort_keys=True)})
        os.replace(tmp, out / name)
        side = {"format": FORMAT, "file": name, "sha256": sha256_file(out / name), "dtype": dtype,
                "source_ckpt": ckpt.name, "source_ckpt_sha256": src_sha, "n_tensors": len(tensors),
                "n_params": sum(t.numel() for t in tensors.values()),
                **{k: _jsonable(st.get(k)) for k in SIDECAR_KEYS}}
        _write(sidecar_path(out / name), json.dumps(side, indent=1))
        del st, tr, tensors
    _write(out / (name + ".sha256"), f"{side['sha256']}  {name}\n")
    _write(out / "latest.json", json.dumps({"path": name, "step": side["step"], "config_hash": side["config_hash"],
                                            "format": FORMAT, "dtype": side["dtype"]}))
    return side


def read_inference(path: str | Path, verify: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """(tensors, sidecar) of an export; ``ValueError`` if the sidecar is missing / does not match the file."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path)
    side_p = sidecar_path(path)
    if not path.is_file() or not side_p.is_file():
        raise ValueError(f"inference export {path} or its sidecar {side_p.name} is missing")
    side = json.loads(side_p.read_text(encoding="utf-8"))
    if side.get("format") != FORMAT or side.get("file") != path.name:
        raise ValueError(f"{side_p.name}: format {side.get('format')!r} / file {side.get('file')!r} != {FORMAT} / {path.name}")
    if verify and sha256_file(path) != side.get("sha256"):
        raise ValueError(f"{path.name} sha256 != its sidecar's")
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads((f.metadata() or {}).get("ssdm") or "{}")
    if meta.get("config_hash") != side.get("config_hash") or meta.get("step") != side.get("step"):
        raise ValueError(f"{path.name}: safetensors metadata disagrees with the sidecar")
    return load_file(str(path)), side


def read_trainable(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Trainable tensors + identity (config_hash, step, config) of EITHER format: a Trainer ``.pt`` (untrusted data:
    ``weights_only=True``, numpy RNG arrays allow-listed) or an inference export. For D14 ``train.rlcd export``."""
    if is_inference(path):
        tr, side = read_inference(path)
        return tr, {"config_hash": str(side["config_hash"]), "step": int(side["step"]), "config": side.get("config")}
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    with torch.serialization.safe_globals([core.multiarray._reconstruct, np.ndarray, np.dtype,
                                           type(np.dtype(np.uint32))]):
        st = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    tr = st["trainable"]
    if not isinstance(tr, dict) or not tr or not all(isinstance(v, torch.Tensor) for v in tr.values()):
        raise ValueError("checkpoint has no trainable tensor dict")
    return {k: v.contiguous() for k, v in tr.items()}, {"config_hash": str(st["config_hash"]), "step": int(st["step"]),
                                                        "config": st.get("config")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.ckpt_export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inference")
    i.add_argument("--ckpt", required=True)
    i.add_argument("--out-dir", required=True)
    i.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    a = ap.parse_args(argv)
    side = export_inference(a.ckpt, a.out_dir, a.dtype)
    print(json.dumps({k: side[k] for k in ("file", "sha256", "dtype", "step", "n_tensors", "n_params",
                                           "source_ckpt_sha256") if k in side}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
