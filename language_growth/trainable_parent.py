"""Strict Stage 1 RWKV7 reconstruction for a future language training cell.

The inference path intentionally freezes the qualified parent. This helper
reuses the same reviewed qualification, checkpoint, snapshot, WKV, lineage and
trainable-tensor checks, but returns the restored model before the inference
wrapper freezes it. It loads no training data and starts no training.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True)
class TrainableQualifiedParent:
    """Exact, still-trainable model plus its pinned Stage 1 initialization identity."""

    model: Any
    encoder: Any
    config: Any
    lora_modules: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    identity: dict


def split_guard_fingerprint(guard: Any) -> str:
    """Bind the complete in-memory SplitGuard sets and their source identities."""
    from train.guard import SplitGuard

    if not isinstance(guard, SplitGuard):
        raise ValueError("guard fingerprint requires a SplitGuard")
    if any(not isinstance(value, str) for collection in (guard.ids, guard.hashes, guard.content_hashes)
           for value in collection):
        raise ValueError("SplitGuard sets must contain strings only")
    identity = guard.identity()
    if any(not _is_sha256(value) for value in guard.hashes | guard.content_hashes):
        raise ValueError("SplitGuard protected hashes must be lowercase SHA-256 values")
    payload = {
        "schema": "ssdm.agent_c.split_guard_pin.v1",
        "manifests": sorted(guard.manifests.items()),
        "ids": sorted(guard.ids),
        "hashes": sorted(guard.hashes),
        "content_hashes": sorted(guard.content_hashes),
        "content_index": identity.get("content_index"),
        "note": guard.note,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_qualified_trainable_parent(
    checkpoint: str | Path,
    *,
    report: str | Path,
    report_sha256: str,
    qualification: str | Path,
    qualification_sha256: str,
    snapshot: str | Path,
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> TrainableQualifiedParent:
    """Validate Stage 1 before touching weights, then return strict trainable reconstruction.

    No fallback to stock weights is allowed. The caller remains responsible for
    the fit-only row gate, resources, authorization, and recording this parent
    identity alongside any child checkpoint.
    """
    # Keep qualification before checkpoint/snapshot resolution and before
    # importing the loader that reconstructs any weights.
    from .qualified_parent import file_sha256, inspect_qualified_parent

    spec = inspect_qualified_parent(report=report, report_sha256=report_sha256,
                                    qualification=qualification,
                                    qualification_sha256=qualification_sha256)
    from .native_parent import validate_loaded_native_parent
    from .rwkv7_m2_loader import native_rwkv7_loader, validate_reconstruction_metadata

    config = validate_reconstruction_metadata(spec.qualification, spec.pin)
    path = Path(checkpoint).resolve(strict=True)
    if file_sha256(path) != spec.pin.artifact_sha256:
        raise ValueError("qualified Stage 1 checkpoint hash mismatch")
    load = native_rwkv7_loader(snapshot, spec.qualification, spec.pin, device=device, dtype=dtype)
    loaded = load(path, artifact_sha256=spec.pin.artifact_sha256)
    validate_loaded_native_parent(loaded, spec.pin)
    backend = loaded.backend
    model = backend.model
    trainable_names = tuple(sorted(name for name, parameter in model.named_parameters()
                                   if parameter.requires_grad))
    if not trainable_names:
        raise ValueError("qualified checkpoint restored no trainable head or adapter parameters")
    lora_modules = tuple(spec.qualification["lora_modules"])
    if tuple(sorted(set(lora_modules))) != lora_modules:
        raise ValueError("qualified LoRA module inventory is not canonical")
    return TrainableQualifiedParent(model=model, encoder=backend.encode, config=config,
                                    lora_modules=lora_modules,
                                    trainable_parameter_names=trainable_names,
                                    identity=spec.identity())


def build_prepared_language_trainer(parent: TrainableQualifiedParent, *, config: Any,
                                    train_items: list, diagnostic_items: list,
                                    guard: Any, expected_guard_sha256: str):
    """Bind fit-only items to the exact restored model without reinjecting LoRA.

    The child may change operational budgets, but its TrainConfig hash and
    execution-strategy fields must match the qualified parent. This keeps the
    reconstructed module graph and prepared forward path identical. The
    supplied guard and world split remain the caller's responsibility. The
    expected guard fingerprint must come from a reviewed external pin, not be
    computed from an untrusted guard in the same invocation.
    """
    if guard is None or getattr(config, "require_guard", None) is not True:
        raise ValueError("fit-only language training requires an explicit guarded TrainConfig")
    if not train_items or not diagnostic_items:
        raise ValueError("fit-only language training requires nonempty train and diagnostic rows")
    from train.guard import SplitGuard

    if not isinstance(guard, SplitGuard):
        raise ValueError("fit-only language training requires a SplitGuard")
    guard_identity = guard.identity()
    manifests = guard_identity.get("manifests", {})
    note = guard_identity.get("note")
    if (not isinstance(note, str) or note.startswith("synthetic-only")
            or not isinstance(manifests, dict)
            or any(not isinstance(path, str) for path in manifests)
            or not any("dev-manifest" in path for path in manifests)
            or not any("test-manifest" in path for path in manifests)
            or any(not _is_sha256(sha) for sha in manifests.values())
            or guard_identity.get("n_ids", 0) < 1 or guard_identity.get("n_hashes", 0) < 1):
        raise ValueError("SplitGuard must bind both DEV/TEST manifests and their identifiers/hashes")
    content_index = guard_identity.get("content_index")
    if (not isinstance(content_index, dict) or type(content_index.get("n")) is not int
            or content_index["n"] < 1 or not _is_sha256(content_index.get("sha256"))):
        raise ValueError("SplitGuard must include a pinned nonempty content index")
    if (not _is_sha256(expected_guard_sha256)
            or split_guard_fingerprint(guard) != expected_guard_sha256):
        raise ValueError("SplitGuard contents differ from the externally pinned guard fingerprint")
    if config.hash() != parent.config.hash():
        raise ValueError("child TrainConfig must preserve the qualified parent's training identity")
    for field in parent.config.EXECUTION:
        if getattr(config, field) != getattr(parent.config, field):
            raise ValueError(f"child execution setting {field} differs from the prepared parent")
    live_trainables = tuple(sorted(name for name, parameter in parent.model.named_parameters()
                                   if parameter.requires_grad))
    if live_trainables != parent.trainable_parameter_names:
        raise ValueError("prepared parent trainable parameter inventory changed")

    from train.trainer import Trainer

    trainer = Trainer(parent.model, config, train_items, diagnostic_items,
                      prepared=True, guard=guard)
    # Trainer intentionally has no way to infer prepared modules. Preserve the
    # strict loader's inventory in child checkpoints for the next reconstruction.
    trainer.lora_modules = list(parent.lora_modules)
    return trainer
