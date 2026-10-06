"""Bind a strictly reconstructed Track D parent to the existing L planner.

This is a loader contract, not a loader for PR #74's unmerged artifact format.
The supplied loader owns architecture/tokenizer reconstruction and strict state
loading. L verifies the checkpoint bytes and the independently supplied serving
pins; it never substitutes stock weights or invents calibration defaults.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from decide.adapter import Decider
from decide.calibration import FeatureTemperature, OptionCountTemperature
from heads.decide_backend import DecisionModelBackend

from .planner import PlannerResult, plan_checked_answer

POLICY_MASKING = "none (served covers every supplied option)"


@dataclass(frozen=True)
class LoadedNativeParent:
    """D loader result after exact key/shape and backbone-lineage validation.

    ``backend.checkpoint['artifact_sha256']`` must name the supplied file. The
    backend owns the reconstructed DecisionModel, encoder, layout and template.
    Its head/typed/LoRA state must all come from that artifact, with no missing
    keys or random replacements. Frozen backbone weights must match its lineage.
    ``calibration`` must be explicitly loaded/selected, even for identity serving.
    The object is inference-only after binding; never mutate its weights in place.
    """

    backend: DecisionModelBackend
    calibration: FeatureTemperature | OptionCountTemperature
    max_input_tokens: int


class NativeParentLoader(Protocol):
    def __call__(self, artifact: Path, *, artifact_sha256: str) -> LoadedNativeParent:
        """Strictly reconstruct the exact local artifact, or raise; no fallback."""
        ...


@dataclass(frozen=True)
class NativeParentPin:
    """Independent export/review pins, including the served (calibrated) revision.

    ``runtime`` has exactly fork_mode, shared_prefix and max_input_tokens. The
    merged backend revision omits these settings, so L binds them separately.
    ``revision`` is the entire ``Decider.revision()`` object, not just its id.
    """

    artifact_sha256: str
    revision: dict[str, Any]
    calibration: dict[str, Any]
    runtime: dict[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime(loaded: LoadedNativeParent) -> dict[str, Any]:
    return {"fork_mode": loaded.backend.model.backbone.fork_mode,
            "shared_prefix": loaded.backend.shared_prefix,
            "max_input_tokens": loaded.max_input_tokens}


class NativeParent:
    """Pinned inference provider plus the planner arguments derived from its pins."""

    def __init__(self, loaded: LoadedNativeParent, pin: NativeParentPin) -> None:
        self._loaded = loaded
        self._pin = deepcopy(pin)
        self._decider = Decider(loaded.backend, calibration=loaded.calibration,
                                backend_max_input_tokens=loaded.max_input_tokens)
        self._check_pins()
        loaded.backend.model.eval()
        loaded.backend.model.requires_grad_(False)

    def _check_pins(self) -> None:
        backend = self._loaded.backend
        if (_json(self._decider.revision()) != _json(self._pin.revision)
                or _json(self._decider.calibration.identity()) != _json(self._pin.calibration)
                or _json(_runtime(self._loaded)) != _json(self._pin.runtime)
                or self._decider.backend_max_input_tokens != self._pin.runtime["max_input_tokens"]):
            raise ValueError("native parent serving pins do not match loader result")
        # revision() is cached by the backend. Check the small live configuration
        # too; do not rehash a full backbone on each planner slot.
        revision = self._pin.revision
        if (backend.layout != revision["layout"]
                or _json(backend.template.identity()) != _json(revision["prompt_template"])
                or _json(backend.encode.identity()) != _json(revision["encoder"])
                or _json(backend.checkpoint) != _json(revision["checkpoint"])):
            raise ValueError("native parent backend configuration changed")

    def planner_kwargs(self) -> dict[str, Any]:
        self._check_pins()
        return {"model": self._pin.revision["model_id"],
                "parent_revision": self._pin.revision["revision_id"],
                "expected_served_config": {"calibration": deepcopy(self._pin.calibration),
                                           "policy_masking": POLICY_MASKING}}

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        self._check_pins()
        return self._decider.decide(request)

    def plan(self, task_input: str, host: dict[str, Any], catalog: dict[str, Any], *,
             allow_repair: bool = False) -> PlannerResult:
        return plan_checked_answer(task_input, host, catalog, self,
                                   allow_repair=allow_repair, **self.planner_kwargs())


def load_native_parent(artifact: str | Path, loader: NativeParentLoader, *,
                       pin: NativeParentPin) -> NativeParent:
    """Verify bytes before loading, then bind exact serving pins before inference.

    A path selects only the checkpoint; backbone location and dtype/device policy
    belong to the caller-supplied D loader. This function does not fetch weights.
    """
    if (not isinstance(pin.artifact_sha256, str) or len(pin.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in pin.artifact_sha256)):
        raise ValueError("native parent artifact pin must be a sha256")
    if (set(pin.runtime) != {"fork_mode", "shared_prefix", "max_input_tokens"}
            or pin.runtime["fork_mode"] not in ("cache", "recompute")
            or type(pin.runtime["shared_prefix"]) is not bool
            or type(pin.runtime["max_input_tokens"]) is not int
            or pin.runtime["max_input_tokens"] <= 0):
        raise ValueError("native parent runtime pin is invalid")
    path = Path(artifact).resolve(strict=True)
    if _sha256(path) != pin.artifact_sha256:
        raise ValueError("native parent artifact hash mismatch")
    loaded = loader(path, artifact_sha256=pin.artifact_sha256)
    if not isinstance(loaded, LoadedNativeParent) or not isinstance(loaded.backend, DecisionModelBackend):
        raise TypeError("native parent loader must return a LoadedNativeParent with DecisionModelBackend")
    if not isinstance(loaded.calibration, (FeatureTemperature, OptionCountTemperature)):
        raise TypeError("native parent loader must supply explicit supported calibration")
    if (loaded.backend.checkpoint.get("artifact_sha256") != pin.artifact_sha256
            or _sha256(path) != pin.artifact_sha256):
        raise ValueError("native parent loader did not bind the verified artifact")
    if not callable(getattr(loaded.backend.encode, "identity", None)):
        raise ValueError("native parent encoder must provide an identity")
    return NativeParent(loaded, pin)
