"""RLCD second stage (mode d14_rlcd): start the cell from a SAVED cell checkpoint's trainable tensors (LoRA + head).

The init ships as data: a safetensors file holding ONLY the parent run's ``trainable`` tensors (no pickle reaches the
GPU job), sha256-pinned by the cell config's ``init_from`` block, which also records the provenance (job, artifact,
artifact sha256, .pt member + sha256, source config hash and step).

    # once, on the laptop (the .pt is untrusted data: torch.load(weights_only=True), numpy RNG arrays allow-listed)
    python -X utf8 -m train.rlcd export --ckpt <artifact>/ckpt/ckpt-step0001439.pt --out <data>/m2/init/.../trainable.safetensors \
        --artifact art-bm3e6 --job job-c7pq2

In the job (``train.run_cell --init-file``): ``load_init`` checks the file's sha256 and metadata against ``init_from``,
``apply_init`` copies every tensor into the identically named trainable parameter (names, shapes and count must match
exactly), then ``Trainer.set_reference`` freezes those tensors as the KL reference p_ref. A resumed attempt installs the
init first (so p_ref is the init, not the resumed weights) and then loads its own checkpoint over it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

META_KEYS = ("source_ckpt_sha256", "config_hash", "step", "artifact", "job")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _safe_globals():
    import numpy as np

    core = getattr(np, "_core", None) or np.core
    return [core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]


def read_trainable(ckpt: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """``trainable`` tensors + identity of a Trainer checkpoint, loaded with ``weights_only=True`` (tensors, plain
    containers and numpy arrays only; never arbitrary pickled objects). An INFERENCE export
    (``train.ckpt_export``: ``trainable-stepN.safetensors`` + sidecar, what first-cell runs upload since job-igcu4) is
    read through ``train.ckpt_export.read_trainable`` (sha256-checked against its sidecar; no pickle at all)."""
    if str(ckpt).endswith(".safetensors"):
        from train.ckpt_export import read_trainable as read_export

        return read_export(ckpt)
    with torch.serialization.safe_globals(_safe_globals()):
        st = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    tr = st["trainable"]
    if not isinstance(tr, dict) or not tr or not all(isinstance(v, torch.Tensor) for v in tr.values()):
        raise ValueError("checkpoint has no trainable tensor dict")
    return {k: v.contiguous() for k, v in tr.items()}, {"config_hash": str(st["config_hash"]), "step": int(st["step"]),
                                                        "config": st.get("config")}


def export(ckpt: str | Path, out: str | Path, artifact: str, job: str) -> dict:
    from safetensors.torch import save_file

    tr, ident = read_trainable(ckpt)
    meta = {"source_ckpt_sha256": sha256_file(ckpt), "config_hash": ident["config_hash"], "step": str(ident["step"]),
            "artifact": artifact, "job": job}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tr, str(out), metadata=meta)
    return {**meta, "sha256": sha256_file(out), "n_tensors": len(tr), "n_params": sum(t.numel() for t in tr.values())}


def load_init(path: str | Path, block: dict) -> dict[str, torch.Tensor]:
    """The pinned init tensors (``ValueError`` on any sha256 / metadata mismatch with the cell's ``init_from``)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if not Path(path).is_file():
        raise ValueError(f"init file {path} missing")
    got = sha256_file(path)
    if got != block["sha256"]:
        raise ValueError(f"init file sha256 {got} != init_from.sha256 {block['sha256']}")
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    for k in META_KEYS:
        want = block.get("ckpt_sha256" if k == "source_ckpt_sha256" else k)
        if want is not None and str(meta.get(k)) != str(want):
            raise ValueError(f"init metadata {k} {meta.get(k)!r} != init_from {want!r}")
    return load_file(str(path))


def apply_init(model: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> dict:
    """Copy ``tensors`` into the model's trainable parameters: the name sets and shapes must be identical."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    miss, extra = sorted(set(params) - set(tensors)), sorted(set(tensors) - set(params))
    if miss or extra:
        raise ValueError(f"init tensors != trainable parameters: missing {miss[:5]} extra {extra[:5]}")
    bad = [n for n, t in tensors.items() if tuple(t.shape) != tuple(params[n].shape)]
    if bad:
        raise ValueError(f"init tensor shapes differ: {bad[:5]}")
    with torch.no_grad():
        for n, t in tensors.items():
            params[n].copy_(t.to(params[n].device, params[n].dtype))
    return {"n_tensors": len(tensors), "n_params": sum(t.numel() for t in tensors.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.rlcd")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--artifact", required=True)
    e.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(export(a.ckpt, a.out, a.artifact, a.job), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
