"""Experimental-only, identity-bound RWKV-7 state capture/edit/replay.

This module returns logits for a controlled research replay; it does not mint
provider handles, construct semantic programs, score answers, or bypass the
ordinary planner/kernel/renderer. It is CPU-only and not part of serving.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from typing import Any


class InterventionError(ValueError):
    """Fail-closed state-intervention precondition error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tensor_bytes(tensor) -> bytes:
    import torch

    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise InterventionError("state directions and slots must be floating tensors")
    value = tensor.detach().contiguous().cpu()
    if not torch.isfinite(value).all():
        raise InterventionError("non-finite state tensor")
    return value.view(torch.uint8).numpy().tobytes()


def _state_slots(state) -> dict[tuple[int, int], Any]:
    import torch

    def ordered_slots(value, name):
        if isinstance(value, dict):
            if set(value) != {0, 1, 2}:
                raise InterventionError(f"RWKV cache {name} must have exactly slots 0, 1, 2")
            return [value[i] for i in range(3)]
        if isinstance(value, list) and len(value) == 3:
            return value
        raise InterventionError(f"RWKV cache {name} must use three keyed slots")

    layers = getattr(state, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise InterventionError("expected nonempty RWKV cache layers")
    result = {}
    for layer_idx, layer in enumerate(layers):
        slots = ordered_slots(getattr(layer, "recurrent_states", None), "recurrent_states")
        initialized = ordered_slots(getattr(layer, "is_recurrent_states_initialized", None),
                                    "is_recurrent_states_initialized")
        for slot_idx, tensor in enumerate(slots):
            if not initialized[slot_idx] or not isinstance(tensor, torch.Tensor):
                raise InterventionError("all RWKV-7 recurrent slots must be initialized")
            if tensor.device.type != "cpu" or not tensor.is_floating_point():
                raise InterventionError("this protocol supports CPU floating recurrent state only")
            result[(layer_idx, slot_idx)] = tensor
    return result


def _state_manifest(state) -> tuple[dict[str, Any], str]:
    config = getattr(state, "config", None)
    if isinstance(config, dict):
        config_value = config
    elif callable(getattr(config, "to_dict", None)):
        config_value = config.to_dict()
    else:
        raise InterventionError("serializable RWKV cache config required")
    manifest = {
        "$cache": {
            "type": f"{type(state).__module__}.{type(state).__qualname__}",
            "config_sha256": _sha(_canonical(config_value)),
        }
    }
    layer_metadata = {}
    for layer_idx, layer in enumerate(state.layers):
        flags = {}
        for name in ("is_recurrent_states_initialized", "has_previous_state"):
            value = getattr(layer, name, None)
            if isinstance(value, dict) and set(value) == {0, 1, 2}:
                value = [value[i] for i in range(3)]
            if not isinstance(value, list) or len(value) != 3 or any(type(x) is not bool for x in value):
                raise InterventionError(f"RWKV cache layer metadata {name} is missing or malformed")
            flags[name] = list(value)
        layer_metadata[str(layer_idx)] = {
            "type": f"{type(layer).__module__}.{type(layer).__qualname__}", **flags,
        }
    manifest["$layers"] = layer_metadata
    state_metadata = {}
    for name in ("seen", "_seen_tokens", "max_batch_size", "max_cache_len"):
        if hasattr(state, name):
            value = getattr(state, name)
            if value is not None and type(value) not in (int, bool, str):
                raise InterventionError(f"unsupported RWKV cache metadata: {name}")
            state_metadata[name] = value
    manifest["$state_metadata"] = state_metadata
    for (layer_idx, slot_idx), tensor in sorted(_state_slots(state).items()):
        manifest[f"layer/{layer_idx}/recurrent/{slot_idx}"] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": _sha(_tensor_bytes(tensor)),
        }
    return manifest, _sha(_canonical(manifest))


def _model_weights_sha256(model) -> str:
    import torch

    state_dict = model.state_dict() if callable(getattr(model, "state_dict", None)) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise InterventionError("model state_dict required to bind immutable parent weights")
    manifest = {}
    for name, tensor in sorted(state_dict.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise InterventionError("CPU tensors required to verify immutable parent weights")
        value = tensor.detach().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise InterventionError("non-finite parent weight")
        manifest[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": _sha(value.reshape(-1).view(torch.uint8).numpy().tobytes())}
    return _sha(_canonical(manifest))


def _model_implementation_identity(model) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if isinstance(config, dict):
        config_value = config
    elif callable(getattr(config, "to_dict", None)):
        config_value = config.to_dict()
    else:
        raise InterventionError("serializable parent model config required")
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "config_sha256": _sha(_canonical(config_value)),
    }


def _direction_digest(tensors: dict[tuple[int, int], Any]) -> str:
    manifest = {}
    for (layer_idx, slot_idx), tensor in sorted(tensors.items()):
        manifest[f"layer/{layer_idx}/recurrent/{slot_idx}"] = {
            "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": _sha(_tensor_bytes(tensor)),
        }
    return _sha(_canonical(manifest))


def _registration_digest(edit_id: str, runtime_identity_sha256: str,
                        discovery_cohort_sha256: str, validation_cohort_sha256: str,
                        scale: float, tensors: dict[tuple[int, int], Any]) -> str:
    return _sha(_canonical({
        "edit_id": edit_id,
        "runtime_identity_sha256": runtime_identity_sha256,
        "discovery_cohort_sha256": discovery_cohort_sha256,
        "validation_cohort_sha256": validation_cohort_sha256,
        "scale": scale,
        "direction_sha256": _direction_digest(tensors),
    }))


def _capture_digest(runtime_identity_sha256: str, prefix_sha256: str, prefix_length: int,
                    context_ref: str, state_sha256: str) -> str:
    return _sha(_canonical({"runtime_identity_sha256": runtime_identity_sha256,
                            "prefix_sha256": prefix_sha256, "prefix_length": prefix_length,
                            "context_ref": context_ref, "state_sha256": state_sha256}))


@dataclass(frozen=True)
class RegisteredStateDirection:
    edit_id: str
    runtime_identity_sha256: str
    discovery_cohort_sha256: str
    validation_cohort_sha256: str
    scale: float
    tensors: dict[tuple[int, int], Any]
    direction_sha256: str

    @classmethod
    def freeze(cls, *, edit_id: str, runtime_identity_sha256: str,
               discovery_cohort_sha256: str, validation_cohort_sha256: str,
               scale: float, tensors: dict[tuple[int, int], Any]) -> "RegisteredStateDirection":
        import torch

        if (not isinstance(edit_id, str) or not edit_id
                or any(not isinstance(x, str) or len(x) != 64 for x in
                       (runtime_identity_sha256, discovery_cohort_sha256, validation_cohort_sha256))
                or not math.isfinite(scale) or scale <= 0 or not tensors):
            raise InterventionError("invalid pre-registered direction identity or scale")
        copied = {}
        for path, tensor in tensors.items():
            if (not isinstance(path, tuple) or len(path) != 2
                    or any(type(i) is not int or i < 0 for i in path)):
                raise InterventionError("direction paths must be (layer, recurrent_slot) integer pairs")
            if path[1] not in (0, 1, 2):
                raise InterventionError("only RWKV-7's three recurrent slots are editable")
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or not tensor.is_floating_point() or not torch.isfinite(tensor).all()
                    or torch.linalg.vector_norm(tensor.float()).item() == 0):
                raise InterventionError("each registered direction must be finite, nonzero CPU floats")
            copied[path] = tensor.detach().clone().contiguous()
        return cls(edit_id, runtime_identity_sha256, discovery_cohort_sha256,
                   validation_cohort_sha256, float(scale), copied,
                   _registration_digest(edit_id, runtime_identity_sha256, discovery_cohort_sha256,
                                        validation_cohort_sha256, float(scale), copied))


@dataclass(frozen=True)
class StateCapture:
    runtime_identity_sha256: str
    prefix_sha256: str
    prefix_length: int
    context_ref: str
    _state: Any
    state_manifest: dict[str, Any]
    state_sha256: str
    capture_sha256: str


@dataclass(frozen=True)
class EditedStateCapture:
    original: StateCapture
    edit_id: str
    sign: int
    _state: Any
    state_manifest: dict[str, Any]
    state_sha256: str
    registration_sha256: str


@dataclass(frozen=True)
class Replay:
    logits: Any
    final_state_sha256: str


class RWKV7SemanticStateIntervention:
    """Experimental RWKV-7 wrapper; registration and runtime identity are fixed at construction."""

    def __init__(self, *, state_ops, parent_identity: dict[str, Any], tokenizer_identity: dict[str, Any],
                 implementation_sha256: str, backend_identity: str, template_id: str,
                 context_ref: str, registrations: list[RegisteredStateDirection]):
        from decide.backends.state_fork import RWKV7StateOps

        if type(state_ops) is not RWKV7StateOps:
            raise InterventionError("only the registered RWKV7StateOps backend is supported")
        if not isinstance(parent_identity, dict) or not parent_identity:
            raise InterventionError("parent identity is required")
        if not isinstance(tokenizer_identity, dict) or not tokenizer_identity:
            raise InterventionError("tokenizer identity is required")
        if (not isinstance(implementation_sha256, str) or len(implementation_sha256) != 64
                or not isinstance(backend_identity, str) or not backend_identity
                or not isinstance(template_id, str) or not template_id
                or not isinstance(context_ref, str) or not context_ref):
            raise InterventionError("implementation/backend/template/context identity required")
        self._state_ops = state_ops
        self._state_ops_id = id(state_ops)
        self._model_object_id = id(state_ops.model)
        self._model_impl_identity = _model_implementation_identity(state_ops.model)
        self._runtime_identity = {
            "parent_identity": copy.deepcopy(parent_identity),
            "tokenizer_identity": copy.deepcopy(tokenizer_identity),
            "implementation_sha256": implementation_sha256,
            "backend_identity": backend_identity,
            "template_id": template_id,
            "context_ref": context_ref,
            "capture_boundary": "caller_supplied_exact_prefix_layout_not_verified",
            "parent_weights_sha256": _model_weights_sha256(state_ops.model),
            "model_implementation": self._model_impl_identity,
        }
        self.runtime_identity_sha256 = _sha(_canonical(self._runtime_identity))
        self._registrations = {}
        self._sealed = False
        for registration in registrations:
            self.register(registration)

    @property
    def state_ops(self):
        return self._state_ops

    def verify_parent_unchanged(self) -> None:
        if (self._state_ops is None or id(self._state_ops) != self._state_ops_id
                or id(self._state_ops.model) != self._model_object_id
                or _model_implementation_identity(self._state_ops.model) != self._model_impl_identity
                or _model_weights_sha256(self._state_ops.model)
                != self._runtime_identity["parent_weights_sha256"]):
            raise InterventionError("parent weights or state-operation identity changed")

    def register(self, registration: RegisteredStateDirection) -> None:
        if self._sealed:
            raise InterventionError("edit registrations are sealed after the first capture")
        if (not isinstance(registration, RegisteredStateDirection)
                or not registration.edit_id or not math.isfinite(registration.scale) or registration.scale <= 0
                or registration.runtime_identity_sha256 != self.runtime_identity_sha256
                or not registration.tensors
                or _registration_digest(registration.edit_id, registration.runtime_identity_sha256,
                    registration.discovery_cohort_sha256, registration.validation_cohort_sha256,
                    registration.scale, registration.tensors) != registration.direction_sha256
                or registration.edit_id in self._registrations):
            raise InterventionError("direction registration identity, bytes or ID mismatch")
        self._registrations[registration.edit_id] = registration

    def capture(self, prefix_ids: list[int], *, context_ref: str) -> StateCapture:
        import torch
        from decide.backends.state_fork import _deepcopy_cache

        if self._runtime_identity_hash() != self.runtime_identity_sha256:
            raise InterventionError("registered runtime identity changed")
        self.verify_parent_unchanged()
        if (not isinstance(prefix_ids, list) or not prefix_ids
                or any(type(token) is not int or token < 0 for token in prefix_ids)):
            raise InterventionError("nonempty integer prefix token IDs required")
        if context_ref != self._runtime_identity["context_ref"]:
            raise InterventionError("capture context does not match the registered runtime")
        self._sealed = True
        ids = torch.tensor([prefix_ids], dtype=torch.long, device="cpu")
        with torch.inference_mode():
            _logits, state = self.state_ops.prefill(ids)
        private = _deepcopy_cache(state, [getattr(state, "config", None)])
        manifest, state_sha = _state_manifest(private)
        self.verify_parent_unchanged()
        prefix_sha = _sha(_canonical(prefix_ids))
        capture_sha = _capture_digest(self.runtime_identity_sha256, prefix_sha, len(prefix_ids),
                                      context_ref, state_sha)
        return StateCapture(self.runtime_identity_sha256, prefix_sha, len(prefix_ids),
                            context_ref, private, manifest, state_sha, capture_sha)

    def apply_registered_edit(self, capture: StateCapture, *, edit_id: str, sign: int) -> EditedStateCapture:
        import torch
        from decide.backends.state_fork import _deepcopy_cache

        self._check_capture(capture)
        if type(sign) is not int or sign not in (-1, 1):
            raise InterventionError("registered intervention sign must be -1 or +1")
        registration = self._registrations.get(edit_id)
        if (registration is None or registration.runtime_identity_sha256 != self.runtime_identity_sha256
                or _registration_digest(registration.edit_id, registration.runtime_identity_sha256,
                    registration.discovery_cohort_sha256, registration.validation_cohort_sha256,
                    registration.scale, registration.tensors) != registration.direction_sha256):
            raise InterventionError("unknown or mutated registered edit")
        slots = _state_slots(capture._state)
        edited = _deepcopy_cache(capture._state, [getattr(capture._state, "config", None)])
        edited_slots = _state_slots(edited)
        for path, direction in registration.tensors.items():
            original = slots.get(path)
            target = edited_slots.get(path)
            if (original is None or target is None or tuple(original.shape) != tuple(direction.shape)
                    or original.dtype != direction.dtype):
                raise InterventionError("registered edit path/shape/dtype does not match captured state")
            updated = target + (direction * (sign * registration.scale))
            if not torch.isfinite(updated).all():
                raise InterventionError("state edit produced non-finite values")
            edited.layers[path[0]].recurrent_states[path[1]] = updated
        manifest, state_sha = _state_manifest(edited)
        self._check_capture(capture)  # applying an edit must not mutate the registered original
        return EditedStateCapture(capture, edit_id, sign, edited, manifest, state_sha,
                                  registration.direction_sha256)

    def replay(self, capture: StateCapture | EditedStateCapture, suffix_ids: list[int]) -> Replay:
        import torch
        from decide.backends.state_fork import _deepcopy_cache

        original = capture.original if isinstance(capture, EditedStateCapture) else capture
        self._check_capture(original)
        self.verify_parent_unchanged()
        if isinstance(capture, EditedStateCapture):
            registration = self._registrations.get(capture.edit_id)
            if (registration is None or type(capture.sign) is not int or capture.sign not in (-1, 1)
                    or capture.registration_sha256 != registration.direction_sha256):
                raise InterventionError("edited capture is not bound to a registered intervention")
            if _registration_digest(registration.edit_id, registration.runtime_identity_sha256,
                    registration.discovery_cohort_sha256, registration.validation_cohort_sha256,
                    registration.scale, registration.tensors) != registration.direction_sha256:
                raise InterventionError("registered intervention bytes or metadata changed")
            expected = _deepcopy_cache(original._state, [getattr(original._state, "config", None)])
            expected_slots = _state_slots(expected)
            for path, direction in registration.tensors.items():
                target = expected_slots.get(path)
                if (target is None or tuple(target.shape) != tuple(direction.shape)
                        or target.dtype != direction.dtype):
                    raise InterventionError("registered edit path/shape/dtype does not match captured state")
                expected.layers[path[0]].recurrent_states[path[1]] = target + (direction * (capture.sign * registration.scale))
            _expected_manifest, expected_sha = _state_manifest(expected)
            if expected_sha != capture.state_sha256:
                raise InterventionError("edited capture does not match the registered direction")
        manifest, state_sha = _state_manifest(capture._state)
        if manifest != capture.state_manifest or state_sha != capture.state_sha256:
            raise InterventionError("replay source state changed after capture/edit")
        if (not isinstance(suffix_ids, list) or not suffix_ids
                or any(type(token) is not int or token < 0 for token in suffix_ids)):
            raise InterventionError("nonempty integer suffix token IDs required")
        state = capture._state
        with torch.inference_mode():
            _logits, final_state = self.state_ops.extend(
                state, torch.tensor([suffix_ids], dtype=torch.long, device="cpu"), len(suffix_ids))
        final_manifest, final_sha = _state_manifest(final_state)
        after_manifest, after_sha = _state_manifest(capture._state)
        if after_manifest != manifest or after_sha != state_sha:
            raise InterventionError("replay mutated its captured source state")
        self.verify_parent_unchanged()
        return Replay(_logits, final_sha)

    def _runtime_identity_hash(self) -> str:
        return _sha(_canonical(self._runtime_identity))

    def _check_capture(self, capture: StateCapture) -> None:
        if (not isinstance(capture, StateCapture)
                or self._runtime_identity_hash() != self.runtime_identity_sha256
                or capture.runtime_identity_sha256 != self.runtime_identity_sha256
                or capture.context_ref != self._runtime_identity["context_ref"]
                or type(capture.prefix_length) is not int or capture.prefix_length < 1
                or not isinstance(capture.prefix_sha256, str) or len(capture.prefix_sha256) != 64):
            raise InterventionError("capture lineage mismatch")
        manifest, state_sha = _state_manifest(capture._state)
        capture_sha = _capture_digest(capture.runtime_identity_sha256, capture.prefix_sha256,
                                      capture.prefix_length, capture.context_ref, capture.state_sha256)
        if capture_sha != capture.capture_sha256:
            raise InterventionError("capture lineage mismatch: prefix or identity changed")
        if manifest != capture.state_manifest or state_sha != capture.state_sha256:
            raise InterventionError("captured recurrent state changed after registration")
