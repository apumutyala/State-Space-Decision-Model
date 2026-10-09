"""Strict Stage 1 RWKV7 reconstruction for a future language training cell.

The inference path intentionally freezes the qualified parent. This helper
reuses the same reviewed qualification, checkpoint, snapshot, WKV, lineage and
trainable-tensor checks, but returns the restored model before the inference
wrapper freezes it. It loads no training data and starts no training.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib


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


@dataclass(frozen=True)
class PinnedSplitGuard:
    """Manifest-derived hash guard loaded from an independently pinned artifact."""

    guard: Any
    artifact_sha256: str
    fingerprint_sha256: str


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


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate guard artifact member {key!r}")
        result[key] = value
    return result


def load_pinned_split_guard(path: str | Path, *, artifact_sha256: str,
                            expected_fingerprint_sha256: str) -> PinnedSplitGuard:
    """Load hash-only DEV/TEST guard metadata without reading either manifest or row text.

    The artifact schema carries manifest identities and the complete ID/payload/
    content-hash sets. Its byte pin and the separately supplied fingerprint
    must both match before the resulting guard can reach the Trainer.
    """
    if not _is_sha256(artifact_sha256) or not _is_sha256(expected_fingerprint_sha256):
        raise ValueError("guard artifact and fingerprint pins must be lowercase SHA-256 values")
    raw = Path(path).resolve(strict=True).read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact_sha256:
        raise ValueError("SplitGuard artifact byte hash mismatch")
    try:
        payload_bytes = gzip.decompress(raw) if str(path).endswith(".gz") else raw
        payload = json.loads(payload_bytes, object_pairs_hook=_json_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (OSError, EOFError, UnicodeDecodeError, ValueError, zlib.error) as exc:
        raise ValueError("SplitGuard artifact is malformed") from exc
    required = {"schema", "manifests", "ids", "hashes", "content_hashes", "content_index", "note"}
    if not isinstance(payload, dict) or set(payload) != required or payload["schema"] != "ssdm.agent_c.split_guard_pin.v1":
        raise ValueError("unsupported SplitGuard artifact schema")
    manifests = payload["manifests"]
    if (not isinstance(manifests, dict) or not any("dev-manifest" in str(p) for p in manifests)
            or not any("test-manifest" in str(p) for p in manifests)
            or any(not isinstance(p, str) or not _is_sha256(sha) for p, sha in manifests.items())):
        raise ValueError("SplitGuard artifact must pin DEV and TEST manifests")
    for name in ("ids", "hashes", "content_hashes"):
        values = payload[name]
        if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"SplitGuard {name} must be sorted unique strings")
        if not values:
            raise ValueError(f"SplitGuard {name} must be nonempty")
    if any(not _is_sha256(value) for name in ("hashes", "content_hashes") for value in payload[name]):
        raise ValueError("SplitGuard payload/content hashes must be lowercase SHA-256 values")
    content_index = payload["content_index"]
    canonical_content_hashes = ("\n".join(payload["content_hashes"]) + "\n").encode("ascii")
    if (not isinstance(content_index, dict) or set(content_index) != {"path", "sha256", "n"}
            or not isinstance(content_index["path"], str) or not content_index["path"]
            or not _is_sha256(content_index["sha256"])
            or type(content_index["n"]) is not int or content_index["n"] != len(payload["content_hashes"])
            or content_index["n"] < 1
            or hashlib.sha256(canonical_content_hashes).hexdigest() != content_index["sha256"]):
        raise ValueError("SplitGuard content index identity is invalid")
    if not isinstance(payload["note"], str) or not payload["note"] or payload["note"].startswith("synthetic-only"):
        raise ValueError("SplitGuard artifact must have non-synthetic provenance")
    from train.guard import SplitGuard

    guard = SplitGuard(ids=set(payload["ids"]), hashes=set(payload["hashes"]),
                       manifests=manifests, note=payload["note"],
                       content_hashes=set(payload["content_hashes"]), content_index=content_index)
    if split_guard_fingerprint(guard) != expected_fingerprint_sha256:
        raise ValueError("SplitGuard contents differ from the independent fingerprint pin")
    return PinnedSplitGuard(guard, artifact_sha256, expected_fingerprint_sha256)


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
                                    pinned_guard: PinnedSplitGuard):
    """Bind fit-only items to the exact restored model without reinjecting LoRA.

    The child may change operational budgets, but its TrainConfig hash and
    execution-strategy fields must match the qualified parent. This keeps the
    reconstructed module graph and prepared forward path identical. The
    supplied guard and world split remain the caller's responsibility. The
    expected guard fingerprint must come from a reviewed external pin, not be
    computed from an untrusted guard in the same invocation.
    """
    if not isinstance(pinned_guard, PinnedSplitGuard) or getattr(config, "require_guard", None) is not True:
        raise ValueError("fit-only language training requires an explicit guarded TrainConfig")
    guard = pinned_guard.guard
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
    if (not _is_sha256(pinned_guard.artifact_sha256) or not _is_sha256(pinned_guard.fingerprint_sha256)
            or split_guard_fingerprint(guard) != pinned_guard.fingerprint_sha256):
        raise ValueError("SplitGuard contents differ from the pinned guard artifact/fingerprint")
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
