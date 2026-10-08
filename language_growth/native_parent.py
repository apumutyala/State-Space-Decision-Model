"""Bind a strictly reconstructed Track D parent to the existing L planner.

This consumes the merged M2 serving bundle v1, not a checkpoint reconstruction
implementation. PR #144's strict Trainer loading is still an open dependency
for a Trainer-based D loader; no default or permissive loader is supplied here.
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
from decide.calibration import FeatureTemperature
from heads.decide_backend import DecisionModelBackend

from .planner import PlannerResult, plan_checked_answer

POLICY_MASKING = "none (served covers every supplied option)"
SERVING_BUNDLE_SCHEMA = "ssdm.m2.serving_bundle.v1"
RUNTIME_KEYS = {"fork_mode", "shared_prefix", "backend_max_input_tokens",
                "enable_experimental_provenance", "enable_v0_1_deltas"}


@dataclass(frozen=True)
class LoadedNativeParent:
    """D loader result after exact key/shape and backbone-lineage validation.

    ``backend.checkpoint['sha256']`` must name the supplied file. Any legacy
    ``artifact_sha256`` alias must agree. The
    backend owns the reconstructed DecisionModel, encoder, layout and template.
    Its head/typed/LoRA state must all come from that artifact, with no missing
    keys or random replacements. Frozen backbone weights must match its lineage.
    ``calibration`` must be reconstructed from the full serving export payload.
    The object is inference-only after binding; never mutate its weights in place.
    """

    backend: DecisionModelBackend
    calibration: FeatureTemperature
    backend_max_input_tokens: int | None
    enable_experimental_provenance: bool = False
    enable_v0_1_deltas: bool = False


class NativeParentLoader(Protocol):
    def __call__(self, artifact: Path, *, artifact_sha256: str) -> LoadedNativeParent:
        """Strictly reconstruct the exact local artifact, or raise; no fallback.

        The caller must keep the resolved artifact immutable for the entire
        loader call. The loader may stream from the path but must not modify it.
        """
        ...


@dataclass(frozen=True)
class NativeParentPin:
    """Independent export/review pins, including the served (calibrated) revision.

    ``runtime`` carries all five M2 v1 settings, preserving a null backend cap.
    ``calibration`` is the full payload, not the abbreviated identity summary.
    ``revision`` is the entire ``Decider.revision()`` object, not just its id.
    """

    artifact_sha256: str
    revision: dict[str, Any]
    calibration: dict[str, Any]
    runtime: dict[str, Any]

    @classmethod
    def from_m2_report(cls, report: dict[str, Any]) -> "NativeParentPin":
        """Read identities.serving_bundle, validating before any artifact access.

        This checks internal consistency, not authenticity or promotion. The
        caller must pin/review the report and apply its existing decision gates.
        """
        try:
            identities = report["identities"]
            bundle = identities["serving_bundle"]
            if (not isinstance(bundle, dict)
                    or set(bundle) != {"schema", "artifact_sha256", "revision", "calibration", "runtime"}
                    or bundle["schema"] != SERVING_BUNDLE_SCHEMA):
                raise ValueError("unsupported M2 serving bundle")
            pin = cls(bundle["artifact_sha256"], bundle["revision"], bundle["calibration"], bundle["runtime"])
            _validate_pin(pin)
            # The actual report retains these older identity fields. Reject
            # disagreement rather than picking one representation silently.
            if ("checkpoint" in identities
                    and _json(identities["checkpoint"]) != _json(pin.revision["checkpoint"])):
                raise ValueError("M2 report checkpoint disagrees with serving bundle")
            if ("child" in identities
                    and _json(identities["child"]["served"]) != _json(pin.revision)):
                raise ValueError("M2 report served revision disagrees with serving bundle")
            return cls(**json.loads(_json(pin.__dict__)))
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed M2 serving bundle report") from exc


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_digest(checkpoint: dict[str, Any]) -> str:
    if not isinstance(checkpoint, dict):
        raise ValueError("native parent checkpoint identity is malformed")
    digest = checkpoint.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or ("artifact_sha256" in checkpoint and checkpoint["artifact_sha256"] != digest)):
        raise ValueError("native parent checkpoint sha256 is missing or inconsistent")
    return digest


def _validate_pin(pin: NativeParentPin) -> FeatureTemperature:
    if (not isinstance(pin.artifact_sha256, str) or len(pin.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in pin.artifact_sha256)):
        raise ValueError("native parent artifact pin must be a sha256")
    if not all(isinstance(value, dict) for value in (pin.runtime, pin.revision, pin.calibration)):
        raise ValueError("native parent serving pins are malformed")
    runtime = pin.runtime
    if (set(runtime) != RUNTIME_KEYS or runtime["fork_mode"] not in ("cache", "recompute")
            or any(type(runtime[key]) is not bool for key in (
                "shared_prefix", "enable_experimental_provenance", "enable_v0_1_deltas"))
            or (runtime["backend_max_input_tokens"] is not None and (
                type(runtime["backend_max_input_tokens"]) is not int
                or runtime["backend_max_input_tokens"] <= 0))):
        raise ValueError("native parent runtime pin is invalid")
    calibration = FeatureTemperature.from_dict(pin.calibration)
    calibration.assert_servable()
    if _checkpoint_digest(pin.revision["checkpoint"]) != pin.artifact_sha256:
        raise ValueError("native parent artifact hash mismatch with checkpoint pin")
    if (_json(calibration.to_dict()) != _json(pin.calibration)
            or _json(calibration.identity()) != _json(pin.revision["calibration"])):
        raise ValueError("native parent serving pins are internally inconsistent")
    return calibration


def _runtime(loaded: LoadedNativeParent) -> dict[str, Any]:
    return {"fork_mode": loaded.backend.model.backbone.fork_mode,
            "shared_prefix": loaded.backend.shared_prefix,
            "backend_max_input_tokens": loaded.backend_max_input_tokens,
            "enable_experimental_provenance": loaded.enable_experimental_provenance,
            "enable_v0_1_deltas": loaded.enable_v0_1_deltas}


class NativeParent:
    """Pinned inference provider plus the planner arguments derived from its pins."""

    def __init__(self, loaded: LoadedNativeParent, pin: NativeParentPin) -> None:
        self._loaded = loaded
        self._pin = deepcopy(pin)
        self._decider = Decider(loaded.backend, calibration=loaded.calibration,
                                backend_max_input_tokens=loaded.backend_max_input_tokens,
                                enable_experimental_provenance=loaded.enable_experimental_provenance,
                                enable_v0_1_deltas=loaded.enable_v0_1_deltas)
        self._check_pins()
        loaded.backend.model.eval()
        loaded.backend.model.requires_grad_(False)

    def _check_pins(self) -> None:
        backend = self._loaded.backend
        if (_json(self._decider.revision()) != _json(self._pin.revision)
                or _json(self._decider.calibration.to_dict()) != _json(self._pin.calibration)
                or _json(_runtime(self._loaded)) != _json(self._pin.runtime)
                or self._decider.backend_max_input_tokens != self._pin.runtime["backend_max_input_tokens"]
                or self._decider.enable_experimental_provenance != self._pin.runtime["enable_experimental_provenance"]
                or self._decider.enable_v0_1_deltas != self._pin.runtime["enable_v0_1_deltas"]):
            raise ValueError("native parent serving pins do not match loader result")
        # revision() is cached by the backend. Check the live configuration,
        # capture point and layer too; do not rehash full backbone weights on
        # each planner slot.
        revision = self._pin.revision
        from train.lineage import backbone_config_identity

        backbone_config = backbone_config_identity(backend.model.backbone)
        pinned_backbone = revision["backbone"]
        if (backend.layout != revision["layout"]
                or _json(backend.template.identity()) != _json(revision["prompt_template"])
                or _json(backend.encode.identity()) != _json(revision["encoder"])
                or _json(backend.checkpoint) != _json(revision["checkpoint"])
                or _json(backbone_config) != _json(
                    {key: pinned_backbone.get(key) for key in backbone_config})):
            raise ValueError("native parent backend configuration changed")

    def planner_kwargs(self) -> dict[str, Any]:
        self._check_pins()
        return {"model": self._pin.revision["model_id"],
                "parent_revision": self._pin.revision["revision_id"],
                "expected_served_config": {"calibration": json.loads(_json(self._decider.calibration.identity())),
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
    belong to the caller-supplied D loader. The caller must keep the resolved
    checkpoint path immutable until the loader returns. This function does not
    fetch weights.
    """
    # Validate the full exported calibration/runtime contract before invoking
    # the caller's strict reconstruction callable. Never fall back to defaults.
    pin = deepcopy(pin)
    _validate_pin(pin)
    path = Path(artifact).resolve(strict=True)
    if _sha256(path) != pin.artifact_sha256:
        raise ValueError("native parent artifact hash mismatch")
    loaded = loader(path, artifact_sha256=pin.artifact_sha256)
    if not isinstance(loaded, LoadedNativeParent) or not isinstance(loaded.backend, DecisionModelBackend):
        raise TypeError("native parent loader must return a LoadedNativeParent with DecisionModelBackend")
    if not isinstance(loaded.calibration, FeatureTemperature):
        raise TypeError("native parent loader must supply explicit supported calibration")
    if (_checkpoint_digest(loaded.backend.checkpoint) != pin.artifact_sha256
            or _sha256(path) != pin.artifact_sha256):
        raise ValueError("native parent loader did not bind the verified artifact")
    if not callable(getattr(loaded.backend.encode, "identity", None)):
        raise ValueError("native parent encoder must provide an identity")
    return NativeParent(loaded, pin)
