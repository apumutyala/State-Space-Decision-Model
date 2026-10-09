"""Strict, explicitly unqualified admission for a fitted language child.

The Stage 1 report qualifies only the immutable initialization checkpoint. A
fit child has a different checkpoint and training lineage, so it must be bound
to a byte-pinned fit receipt and reconstructed from the qualified parent's
frozen backbone plus the child's exact trainable state. This module never
promotes the child or transfers Stage 1 floors/calibration evidence to it.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FIT_RECEIPT_SCHEMA = "ssdm.agent_c.l1_fit_receipt.v1"
LANGUAGE_CHILD_SCHEMA = "ssdm.language_child.v1"


class LanguageChildError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hex(value: Any, size: int) -> bool:
    return (isinstance(value, str) and len(value) == size
            and all(c in "0123456789abcdef" for c in value))


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LanguageChildError("duplicate fit receipt JSON member")
        result[key] = value
    return result


def _exact(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise LanguageChildError(f"unsupported {name} schema")


def _same(left: Any, right: Any) -> bool:
    return _canonical(left) == _canonical(right)


def read_fit_receipt(path: str | Path, expected_sha256: str) -> dict:
    if not _hex(expected_sha256, 64):
        raise LanguageChildError("independent fit receipt byte pin required")
    raw = Path(path).resolve(strict=True).read_bytes()
    if _sha(raw) != expected_sha256:
        raise LanguageChildError("fit receipt byte pin mismatch")
    try:
        receipt = json.loads(raw, object_pairs_hook=_pairs,
                             parse_constant=lambda _: (_ for _ in ()).throw(
                                 LanguageChildError("nonfinite fit receipt JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LanguageChildError("malformed fit receipt JSON") from exc
    if not isinstance(receipt, dict):
        raise LanguageChildError("fit receipt must be an object")
    return receipt


def validate_fit_child_receipt(receipt: dict, *, receipt_sha256: str,
                               child_checkpoint: str | Path,
                               stage1_spec: Any) -> dict:
    """Check the child receipt and checkpoint bytes before loading model weights."""
    keys = {
        "schema", "status", "plan_sha256", "order_sha256", "source_commit", "ordered_by", "order_ref",
        "stage1_parent", "guard", "recipe", "resources", "spend", "split", "item_sidecar_sha256",
        "training_rows", "diagnostic_rows", "language_qualified", "promotion_eligible", "checkpoints",
        "config", "config_hash", "lora_modules", "trainable_parameter_names", "diagnostic", "steps",
        "lineage", "seconds",
    }
    _exact(receipt, keys, "fit receipt")
    from .l1_fit_cell import PLAN_SHA256

    if (receipt["schema"] != FIT_RECEIPT_SCHEMA or receipt["status"] != "completed_fit_only"
            or receipt["plan_sha256"] != PLAN_SHA256 or not _hex(receipt_sha256, 64)
            or not _hex(receipt["order_sha256"], 64) or not _hex(receipt["source_commit"], 40)
            or receipt["ordered_by"] != "agent-a" or not isinstance(receipt["order_ref"], str)
            or not receipt["order_ref"].strip()):
        raise LanguageChildError("fit receipt is incomplete or not a completed authorized L1 fit")
    if (receipt["language_qualified"] is not False or receipt["promotion_eligible"] is not False
            or not _same(receipt["stage1_parent"], stage1_spec.identity())):
        raise LanguageChildError("fit receipt does not preserve the exact unqualified Stage 1 ancestry")
    parent = stage1_spec.qualification
    operational = {"ckpt_dir", "ckpt_every", "eval_every", "max_steps", "require_guard"}
    child_config = receipt["config"]
    parent_config = parent["checkpoint_config"]
    child_core = ({key: value for key, value in child_config.items() if key not in operational}
                  if isinstance(child_config, dict) else None)
    parent_core = ({key: value for key, value in parent_config.items() if key not in operational}
                   if isinstance(parent_config, dict) else None)
    if (receipt["config_hash"] != parent["checkpoint_config_hash"]
            or child_core is None or parent_core is None or not _same(child_core, parent_core)
            or receipt["lora_modules"] != parent["lora_modules"]):
        raise LanguageChildError("fit child configuration or LoRA inventory differs from Stage 1 parent")
    if (type(receipt["training_rows"]) is not int or receipt["training_rows"] != 3600
            or type(receipt["diagnostic_rows"]) is not int or receipt["diagnostic_rows"] != 1200
            or not _hex(receipt["item_sidecar_sha256"], 64)):
        raise LanguageChildError("fit receipt has the wrong pinned fit-only row counts")
    recipe = receipt["recipe"]
    _exact(recipe, {"config_hash", "max_steps", "ckpt_every", "diagnostic_batch_size", "max_seconds"}, "recipe")
    if (recipe["config_hash"] != receipt["config_hash"]
            or type(recipe["max_steps"]) is not int or recipe["max_steps"] < 1
            or type(recipe["ckpt_every"]) is not int or recipe["ckpt_every"] < 0
            or type(recipe["diagnostic_batch_size"]) is not int or recipe["diagnostic_batch_size"] < 1
            or type(recipe["max_seconds"]) not in (int, float)
            or not math.isfinite(recipe["max_seconds"]) or recipe["max_seconds"] <= 0
            or type(receipt["steps"]) is not int or receipt["steps"] != recipe["max_steps"]):
        raise LanguageChildError("fit receipt step/config recipe mismatch")
    guard = receipt["guard"]
    _exact(guard, {"artifact_sha256", "fingerprint_sha256"}, "guard pins")
    if any(not _hex(value, 64) for value in guard.values()):
        raise LanguageChildError("fit receipt guard pins are malformed")
    spend = receipt["spend"]
    _exact(spend, {"authorized", "merged_proof_commit", "acknowledged_issue", "max_cost_usd"}, "spend order")
    if (spend["authorized"] is not True or not _hex(spend["merged_proof_commit"], 40)
            or not isinstance(spend["acknowledged_issue"], str) or not spend["acknowledged_issue"].strip()
            or type(spend["max_cost_usd"]) not in (int, float)
            or not math.isfinite(spend["max_cost_usd"]) or spend["max_cost_usd"] <= 0):
        raise LanguageChildError("fit receipt lacks its acknowledged A spend authorization")
    resource = receipt["resources"]
    _exact(resource, {"host", "device", "dtype", "isolated", "max_peak_gb", "min_free_gb"}, "resource order")
    if (not isinstance(resource["host"], str) or not resource["host"] or resource["isolated"] is not True
            or not isinstance(resource["device"], str)
            or not (resource["device"] == "cuda" or resource["device"].startswith("cuda:")
                    and resource["device"][5:].isdigit())
            or resource["dtype"] not in ("float32", "bfloat16")):
        raise LanguageChildError("fit receipt resource order is malformed")
    diagnostic = receipt["diagnostic"]
    if (not isinstance(diagnostic, dict)
            or diagnostic.get("scope") != "teacher-forced development_train fit worlds only"
            or not isinstance(diagnostic.get("metrics"), dict)):
        raise LanguageChildError("fit receipt diagnostic scope is invalid")

    lineage = receipt["lineage"]
    parent_lineage = parent["checkpoint_lineage"]
    if (not isinstance(lineage, dict) or set(lineage) != {"backbone", "tokenizer", "layout", "data"}
            or any(not _same(lineage[key], parent_lineage[key]) for key in ("backbone", "tokenizer", "layout"))
            or not isinstance(lineage["data"], dict) or set(lineage["data"]) != {"n", "sha256", "order_fn"}
            or type(lineage["data"]["n"]) is not int or lineage["data"]["n"] != receipt["training_rows"]
            or not _hex(lineage["data"]["sha256"], 64)
            or not isinstance(lineage["data"]["order_fn"], str) or not lineage["data"]["order_fn"]):
        raise LanguageChildError("child lineage does not preserve its Stage 1 backbone/tokenizer/layout")
    names = receipt["trainable_parameter_names"]
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
            or names != sorted(set(names))):
        raise LanguageChildError("child trainable parameter inventory is malformed")

    checkpoints = receipt["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise LanguageChildError("fit receipt contains no child checkpoints")
    previous_step = -1
    for entry in checkpoints:
        _exact(entry, {"path", "sha256", "step"}, "checkpoint receipt")
        if (not isinstance(entry["path"], str) or not entry["path"] or not _hex(entry["sha256"], 64)
                or type(entry["step"]) is not int or entry["step"] <= previous_step):
            raise LanguageChildError("fit receipt checkpoint order is malformed")
        previous_step = entry["step"]
    final = checkpoints[-1]
    path = Path(child_checkpoint).resolve(strict=True)
    actual_sha256 = _file_sha(path)
    if (final["step"] != receipt["steps"] or actual_sha256 != final["sha256"]
            or len({entry["sha256"] for entry in checkpoints}) != len(checkpoints)):
        raise LanguageChildError("candidate checkpoint does not match the final receipt checkpoint")
    return {"receipt": deepcopy(receipt), "receipt_sha256": receipt_sha256,
            "checkpoint_path": path, "checkpoint_sha256": actual_sha256,
            "step": final["step"], "lineage": deepcopy(lineage)}


def validate_child_checkpoint_state(state: dict, candidate: dict, stage1_spec: Any) -> None:
    """Validate checkpoint headers before loading the base model or copying tensors."""
    expected_keys = {"step", "epoch", "cursor", "config", "config_hash", "trainable", "optimizer",
                     "scheduler", "rng", "lineage", "init_seed", "history", "backbone", "lora_modules", "guard"}
    _exact(state, expected_keys, "Trainer child checkpoint")
    receipt = candidate["receipt"]
    q = stage1_spec.qualification
    if (type(state["step"]) is not int or state["step"] != candidate["step"]
            or not _same(state["config"], receipt["config"])
            or state["config_hash"] != receipt["config_hash"]
            or not _same(state["lineage"], receipt["lineage"])
            or state["lora_modules"] != q["lora_modules"]
            or state["init_seed"] != receipt["config"].get("seed")
            or state["backbone"] != "rwkv7-13.3b"):
        raise LanguageChildError("child checkpoint header differs from its receipt/Stage 1 parent")
    weights, names = state["trainable"], receipt["trainable_parameter_names"]
    if not isinstance(weights, dict) or sorted(weights) != names:
        raise LanguageChildError("child checkpoint trainable keys differ from the receipt")


class LanguageChild:
    """Inference provider with a native child pin and separate Stage 1 ancestry."""

    def __init__(self, native_parent: Any, *, stage1_identity: dict, candidate: dict):
        self._native = native_parent
        self._stage1_identity = deepcopy(stage1_identity)
        self._candidate = deepcopy(candidate)
        serving = native_parent.planner_kwargs()
        revision = serving["parent_revision"]
        calibration = serving["expected_served_config"]["calibration"]
        self._identity = {
            "schema": LANGUAGE_CHILD_SCHEMA,
            "artifact_sha256": candidate["checkpoint_sha256"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
            "revision_id": revision,
            "candidate_revision": revision,
            "fit_receipt_sha256": candidate["receipt_sha256"],
            "fit_plan_sha256": candidate["receipt"]["plan_sha256"],
            "fit_source_commit": candidate["receipt"]["source_commit"],
            "fit_step": candidate["step"],
            "lineage_sha256": _sha(_canonical(candidate["lineage"])),
            "stage1_parent": deepcopy(stage1_identity),
            "calibration": {"identity": deepcopy(calibration),
                            "source": "inherited_parent_configuration",
                            "child_calibration_measured": False},
            "language_qualified": False,
            "stage1_qualification_inherited": False,
            "decision_retention_measurement_required": True,
        }

    def identity(self) -> dict:
        serving = self._native.planner_kwargs()
        if (serving["parent_revision"] != self._identity["revision_id"]
                or not _same(serving["expected_served_config"]["calibration"],
                             self._identity["calibration"]["identity"])):
            raise LanguageChildError("live child revision/calibration differs from its fit receipt identity")
        return deepcopy(self._identity)

    def planner_kwargs(self) -> dict:
        self.identity()
        return self._native.planner_kwargs()

    def decide(self, request: dict) -> dict:
        self.identity()
        return self._native.decide(request)

    def plan(self, task_input: str, host: dict, catalog: dict, *, allow_repair: bool = False):
        self.identity()
        return self._native.plan(task_input, host, catalog, allow_repair=allow_repair)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Runtime:
    """Production loaders are deferred until after receipt and header validation."""

    def inspect_parent(self, **kwargs):
        from .qualified_parent import inspect_qualified_parent
        return inspect_qualified_parent(**kwargs)

    def checkpoint_state(self, path):
        from .rwkv7_m2_loader import _load_checkpoint
        return _load_checkpoint(Path(path))

    def trainable_parent(self, checkpoint, *, report, report_sha256, qualification,
                         qualification_sha256, snapshot, device, dtype):
        from .trainable_parent import load_qualified_trainable_parent
        return load_qualified_trainable_parent(checkpoint, report=report, report_sha256=report_sha256,
            qualification=qualification, qualification_sha256=qualification_sha256,
            snapshot=snapshot, device=device, dtype=dtype)

    def restore_child(self, parent, state, candidate):
        import torch
        params = {name: parameter for name, parameter in parent.model.named_parameters() if parameter.requires_grad}
        if tuple(sorted(params)) != tuple(parent.trainable_parameter_names):
            raise LanguageChildError("loaded Stage 1 trainable inventory changed")
        values = state["trainable"]
        if not isinstance(values, dict) or set(values) != set(params):
            raise LanguageChildError("child checkpoint trainable key set differs from loaded model")
        for name, parameter in params.items():
            value = values[name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not bool(torch.isfinite(value).all())):
                raise LanguageChildError(f"child trainable tensor mismatch: {name}")
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(values[name])

    def native_parent(self, parent, state, candidate, stage1_spec):
        from decide.calibration import FeatureTemperature
        from decide.adapter import Decider
        from heads.decide_backend import DecisionModelBackend
        from .native_parent import LoadedNativeParent, NativeParent, NativeParentPin

        pin = stage1_spec.pin
        checkpoint = {"path": candidate["checkpoint_path"].name,
                      "step": candidate["step"],
                      "sha256": candidate["checkpoint_sha256"],
                      "artifact_sha256": candidate["checkpoint_sha256"],
                      "parent_artifact_sha256": pin.artifact_sha256,
                      "fit_receipt_sha256": candidate["receipt_sha256"]}
        backend = DecisionModelBackend(parent.model, parent.encoder,
            template=pin.revision["prompt_template"]["id"], layout=parent.config.layout,
            checkpoint=checkpoint, shared_prefix=pin.runtime["shared_prefix"])
        calibration = FeatureTemperature.from_dict(pin.calibration)
        loaded = LoadedNativeParent(backend, calibration,
            backend_max_input_tokens=pin.runtime["backend_max_input_tokens"],
            enable_experimental_provenance=pin.runtime["enable_experimental_provenance"],
            enable_v0_1_deltas=pin.runtime["enable_v0_1_deltas"])
        child_decider = Decider(backend, calibration=calibration,
            backend_max_input_tokens=loaded.backend_max_input_tokens,
            enable_experimental_provenance=loaded.enable_experimental_provenance,
            enable_v0_1_deltas=loaded.enable_v0_1_deltas)
        child_pin = NativeParentPin(candidate["checkpoint_sha256"], child_decider.revision(),
                                    calibration.to_dict(), {
                                        "fork_mode": backend.model.backbone.fork_mode,
                                        "shared_prefix": backend.shared_prefix,
                                        "backend_max_input_tokens": loaded.backend_max_input_tokens,
                                        "enable_experimental_provenance": loaded.enable_experimental_provenance,
                                        "enable_v0_1_deltas": loaded.enable_v0_1_deltas,
                                    })
        return NativeParent(loaded, child_pin)


def load_language_child(child_checkpoint: str | Path, *, stage1_checkpoint: str | Path,
                        receipt: str | Path, receipt_sha256: str,
                        report: str | Path, report_sha256: str,
                        qualification: str | Path, qualification_sha256: str,
                        snapshot: str | Path, device: str = "cpu", dtype: str = "bfloat16",
                        runtime: Any | None = None) -> LanguageChild:
    """Admit a receipt-pinned fit child, validate its header, then restore its trainables."""
    runtime = runtime or Runtime()
    stage1 = runtime.inspect_parent(report=report, report_sha256=report_sha256,
                                    qualification=qualification,
                                    qualification_sha256=qualification_sha256)
    fit_receipt = read_fit_receipt(receipt, receipt_sha256)
    candidate = validate_fit_child_receipt(fit_receipt, receipt_sha256=receipt_sha256,
                                           child_checkpoint=child_checkpoint, stage1_spec=stage1)
    state = runtime.checkpoint_state(candidate["checkpoint_path"])
    if not isinstance(state, dict):
        raise LanguageChildError("child checkpoint must contain a Trainer state object")
    validate_child_checkpoint_state(state, candidate, stage1)

    parent = runtime.trainable_parent(stage1_checkpoint, report=report, report_sha256=report_sha256,
        qualification=qualification, qualification_sha256=qualification_sha256, snapshot=snapshot,
        device=device, dtype=dtype)
    if (not _same(parent.identity, stage1.identity())
            or tuple(parent.trainable_parameter_names) != tuple(fit_receipt["trainable_parameter_names"])):
        raise LanguageChildError("loaded Stage 1 model differs from fit receipt ancestry/inventory")
    runtime.restore_child(parent, state, candidate)
    native_parent = runtime.native_parent(parent, state, candidate, stage1)
    return LanguageChild(native_parent, stage1_identity=stage1.identity(), candidate=candidate)
