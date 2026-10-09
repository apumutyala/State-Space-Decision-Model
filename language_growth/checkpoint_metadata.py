"""Export a native Trainer checkpoint header without reconstructing a model.

Only config/lineage and trainable tensor names, shapes and dtypes are emitted.
This is a handoff artifact, never Stage 1 admission, numerical validation or a
serving bundle. The caller supplies the independently known checkpoint hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .qualified_parent import _digest, canonical_sha256, file_sha256
from .rwkv7_m2_loader import _load_checkpoint

FIELDS = ("step", "config", "config_hash", "lineage", "init_seed", "backbone", "lora_modules")


def checkpoint_metadata(checkpoint: str | Path, *, sha256: str) -> dict:
    """Read the restricted mmap header; do not materialize optimizer or weights."""
    import torch

    path = Path(checkpoint).resolve(strict=True)
    before = path.stat()
    if file_sha256(path) != _digest(sha256):
        raise ValueError("checkpoint hash differs from the handoff pin")
    state = _load_checkpoint(path)
    if not isinstance(state, dict) or not set(FIELDS).issubset(state):
        raise ValueError("incomplete native Trainer checkpoint metadata")
    if (type(state["step"]) is not int or state["step"] < 0
            or type(state["init_seed"]) is not int
            or not isinstance(state["config"], dict) or not isinstance(state["lineage"], dict)
            or not isinstance(state["config_hash"], str) or not state["config_hash"]
            or not isinstance(state["backbone"], str) or not state["backbone"]
            or not isinstance(state["lora_modules"], list)
            or any(not isinstance(name, str) or not name for name in state["lora_modules"])
            or state["lora_modules"] != sorted(set(state["lora_modules"]))):
        raise ValueError("invalid native Trainer checkpoint metadata")
    # An inert JSON round trip rejects unexpected tensor/object metadata and
    # nonfinite scalars. Never serialize state['optimizer'], history or RNG.
    metadata = json.loads(json.dumps({name: state[name] for name in FIELDS}, allow_nan=False))
    weights = state.get("trainable")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("checkpoint has no trainable tensor inventory")
    inventory = {}
    for name, tensor in sorted(weights.items()):
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError("invalid checkpoint trainable tensor inventory")
        inventory[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed while reading its header")
    return {"schema": "ssdm.native_parent_checkpoint_metadata.v1",
            "checkpoint_sha256": sha256, "checkpoint_bytes": before.st_size,
            "checkpoint_metadata": metadata, "metadata_sha256": canonical_sha256(metadata),
            "trainable_inventory": inventory, "model_reconstructed": False,
            "tensor_values_validated": False, "stage1_qualified": False,
            "limitation": "Header identity only; serving/calibration/runtime and Stage 1 evidence remain separate. "
                          "Keep the checkpoint immutable; strict loading rechecks bytes and tensor values."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; preserve prior handoffs")
    result = checkpoint_metadata(args.checkpoint, sha256=args.sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checkpoint_sha256": args.sha256, "metadata_sha256": result["metadata_sha256"],
                      "trainable_tensors": len(result["trainable_inventory"]),
                      "stage1_qualified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
