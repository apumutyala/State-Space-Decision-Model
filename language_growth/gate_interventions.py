"""Four registered intervention arms through the unchanged native fork scorer.

Edits are at final continuation hidden features, before the trained fork head.
This is a final-hidden-axis experiment, not recurrent-memory editing or proof
of overall interpretability. Direction discovery/fitting and Stage 1 admission
belong to pinned fitting/source inputs; no evaluator labels enter here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from typing import Any

from .native_parent import NativeParent
from .planner import PlannerResult
from .qualified_parent import QualifiedParent, file_sha256

SCHEMA = "ssdm.language_intervention_preregistration.v1"
DIRECTION_SCHEMA = "ssdm.native_fork_direction.v1"
SCHEMA_V2 = "ssdm.language_intervention_preregistration.v2"
DIRECTION_SCHEMA_V2 = "ssdm.native_fork_direction.v2"
BOUNDARY = "native_fork_prehead.v1"
SCOPE = "final_hidden_semantic_axis_only"
OUTPUT_CONTROL = "fitted_candidate_count_logit_direction.v1"
ARMS = ("semantic", "sham", "random", "output_only")
NORM_TOLERANCE = 1e-4
_OWNER = threading.Lock()
_TASK_KEYS = {"task_id", "world_id", "level", "split", "task_input", "host", "catalog",
              "planner_profile", "production", "answer_format", "energy_module"}
TARGET_SLOTS = frozenset({"semantic-operation", "fact", "left-operand", "right-operand", "comparison-relation",
                         "first-fact", "second-fact", "assumption", "workflow", "plan-step", "licensed-production",
                         "energy-operation", "energy-record"})


def question_slot(question_id: str) -> str | None:
    """Read only the fixed GatePlanner slot suffix; never inspect model/task text."""
    match = re.fullmatch(r"slot-([0-9]{3})-([a-z][a-z-]*)", question_id) if isinstance(question_id, str) else None
    return match[2] if match and int(match[1]) > 0 and match[2] in TARGET_SLOTS else None


def validate_target_slots(value) -> list[str]:
    if (not isinstance(value, list) or not value or any(not isinstance(slot, str) or slot not in TARGET_SLOTS for slot in value)
            or len(value) != len(set(value))):
        raise ValueError("nonempty unique explicit GatePlanner target slots required")
    return list(value)


def capture_is_targeted(question_id: str, target_slots: list[str] | None) -> bool:
    if target_slots is None:
        return True  # exact legacy all-slot behavior, including legacy planner IDs
    slot = question_slot(question_id)
    if slot is None:
        raise ValueError("targeted intervention requires a known GatePlanner question slot")
    return slot in target_slots


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def registration_sha256(preregistration: dict) -> str:
    """Canonical registration identity, excluding its self-referential hash field."""
    return _sha({key: value for key, value in preregistration.items() if key != "registration_sha256"})


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("lowercase SHA-256 pin required")
    return value


def _tensor_sha(value) -> str:
    import torch
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    flat = value.detach().contiguous().reshape(-1).view(torch.uint8)
    for start in range(0, flat.numel(), 1024 * 1024):
        h.update(flat[start:start + 1024 * 1024].cpu().numpy().tobytes())
    return h.hexdigest()


def _parent_state_sha(model) -> str:
    # Once before/after the whole matched experiment, never per slot. Include
    # backbone, LoRA, trained head, typed output and persistent buffers.
    return _sha({name: _tensor_sha(value) for name, value in sorted(model.state_dict().items())})


def _native(parent) -> NativeParent:
    from heads.decide_backend import DecisionModelBackend
    from heads.heads import FinalStateFork
    from heads.model import DecisionModel
    native = parent._parent if isinstance(parent, QualifiedParent) else parent
    if not isinstance(native, NativeParent):
        raise ValueError("a pinned native decision parent is required")
    native.planner_kwargs()  # recheck existing native revision/runtime/config pins
    backend = native._loaded.backend
    model = backend.model
    if (type(backend) is not DecisionModelBackend or type(model) is not DecisionModel
            or type(model.head) is not FinalStateFork or model.backbone.fork_mode != "cache"
            or model.backbone.capture != "hidden_states" or getattr(model.backbone, "wants_trust", False)
            or model.training or any(p.requires_grad for p in model.parameters())):
        raise ValueError("only the frozen native cache-fork DecisionModel scorer is supported")
    return native


def native_capture_identity(parent) -> dict:
    """Discovery and evaluation must register the same exact native capture point."""
    native = _native(parent)
    backend = native._loaded.backend
    return {"capture_boundary": BOUNDARY, "scope": SCOPE, "feature": "HeadInput.fork",
            "shape_contract": "one_question_all_candidates_by_hidden_dimension",
            "parent_revision": parent.planner_kwargs()["parent_revision"],
            "served_config": parent.planner_kwargs()["expected_served_config"],
            "calibration_payload_sha256": _sha(native._pin.calibration),
            "layout": backend.layout, "prompt_template": backend.template.identity(),
            "encoder": backend.encode.identity(), "head": type(backend.model.head).__name__,
            "hidden_dimension": backend.model.backbone.d_model}


def _vector(value, size):
    import torch
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.layout != torch.strided or value.shape != (size,)
            or not bool(torch.isfinite(value).all()) or float(value.double().norm()) <= 0):
        raise ValueError("registered directions must be finite nonzero CPU float32 vectors of the pinned shape")
    return value.detach().clone().contiguous()


def _registration(preregistration, parent, tasks):
    import torch
    p = deepcopy(preregistration)
    keys = {"schema", "parent_revision", "registration_sha256", "capture_identity_sha256", "capture_boundary",
            "scope", "discovery_world_ids", "discovery_pairs_sha256", "directions", "scale", "random_seed", "output_control"}
    targeted = isinstance(p, dict) and p.get("schema") == SCHEMA_V2
    if targeted:
        keys.add("target_slots")
    if not isinstance(p, dict) or set(p) != keys:
        raise ValueError("incomplete native intervention preregistration")
    if p["registration_sha256"] != registration_sha256(p):
        raise ValueError("intervention registration identity mismatch")
    capture = native_capture_identity(parent)
    if targeted:
        validate_target_slots(p["target_slots"])
    if (p["schema"] not in (SCHEMA, SCHEMA_V2) or p["parent_revision"] != capture["parent_revision"]
            or p["capture_boundary"] != BOUNDARY or p["scope"] != SCOPE
            or p["capture_identity_sha256"] != _sha(capture) or p["output_control"] != OUTPUT_CONTROL):
        raise ValueError("intervention registration does not match the native capture point/control")
    worlds = p["discovery_world_ids"]
    if (not isinstance(worlds, list) or not worlds or any(not isinstance(w, str) or not w for w in worlds)
            or len(set(worlds)) != len(worlds) or set(worlds) & {task["world_id"] for task in tasks}):
        raise ValueError("discovery worlds must be unique and disjoint from evaluation worlds")
    _digest(p["discovery_pairs_sha256"])
    if (type(p["scale"]) not in (int, float) or not math.isfinite(p["scale"]) or p["scale"] <= 0
            or type(p["random_seed"]) is not int or not 0 <= p["random_seed"] < 2**63):
        raise ValueError("finite positive scale and registered random seed required")
    art = p["directions"]
    if not isinstance(art, dict) or set(art) != {"path", "sha256"}:
        raise ValueError("independently pinned direction artifact required")
    path = Path(art["path"])
    if not path.is_absolute():
        raise ValueError("direction artifact path must be absolute")
    path = path.resolve(strict=True)
    if path.stat().st_size > 16 * 1024 * 1024 or file_sha256(path) != _digest(art["sha256"]):
        raise ValueError("direction artifact size/hash mismatch")
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected_keys = {"schema", "parent_revision", "capture_identity_sha256", "capture_boundary", "source_split",
                     "discovery_world_ids", "discovery_pairs_sha256", "output_only_fit_evidence_sha256",
                     "direction", "output_only_directions"}
    if targeted:
        expected_keys.add("target_slots")
    if (not isinstance(data, dict) or set(data) != expected_keys
            or data["schema"] != (DIRECTION_SCHEMA_V2 if targeted else DIRECTION_SCHEMA)):
        raise ValueError("incomplete native direction artifact")
    if targeted and data["target_slots"] != p["target_slots"]:
        raise ValueError("direction applicability differs from preregistration")
    if (data["source_split"] != "discovery_train_only"
            or any(data[key] != p[key] for key in ("parent_revision", "capture_identity_sha256", "capture_boundary",
                                                  "discovery_world_ids", "discovery_pairs_sha256"))):
        raise ValueError("direction/control discovery lineage differs from preregistration")
    _digest(data["output_only_fit_evidence_sha256"])
    semantic = _vector(data["direction"], capture["hidden_dimension"])
    outputs = data["output_only_directions"]
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("fitted output-only control directions are required")
    checked_outputs = {}
    for key, value in outputs.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key or not 1 <= int(key) <= 255:
            raise ValueError("output control keys must be canonical candidate counts")
        checked_outputs[int(key)] = _vector(value, int(key))
        if int(key) > 1 and float(checked_outputs[int(key)].double().std()) == 0:
            raise ValueError("output-only control must change relative logits")
    # One random hidden axis for the entire run; no labels, task-specific
    # target IDs, or data-dependent resampling enter its construction.
    generator = torch.Generator(device="cpu").manual_seed(p["random_seed"])
    random = torch.randn(semantic.shape, dtype=torch.float64, generator=generator)
    random = (random * (semantic.double().norm() / random.norm())).float()
    if file_sha256(path) != art["sha256"]:
        raise ValueError("direction artifact changed during loading")
    return p, capture, semantic, random, checked_outputs, data["output_only_fit_evidence_sha256"]


class _ArmProvider:
    def __init__(self, parent, arm, registration, capture_identity, semantic, random, outputs):
        self.parent, self.arm, self.registration = parent, arm, registration
        self.capture_identity = capture_identity
        self.semantic, self.random, self.outputs = semantic, random, outputs
        self.records = []
        self.current = None
        self.failed = False

    def planner_kwargs(self):
        return self.parent.planner_kwargs()

    def decide(self, request):
        from decide.schema import canonical_options
        questions = request.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError("native intervention supports one complete choice question per planner call")
        qid, question = next(iter(questions.items()))
        if question.get("type") != "choice":
            raise ValueError("native intervention requires choice questions")
        options = [oid for oid, _ in canonical_options(question)]
        self.current = {"question_id": qid, "option_ids": options, "request_sha256": _sha(request),
                        "head_calls": 0, "capture": None, "status": "attempted", "targeted": False}
        self.records.append(self.current)
        try:
            self.current["targeted"] = capture_is_targeted(qid, self.registration.get("target_slots"))
            response = self.parent.decide(request)
            envelope = response.get("ssdm", {})
            self.current["status"] = envelope.get("status", "missing_status")
            if self.failed:
                raise ValueError("native intervention capture/edit failed")
            if envelope.get("status") == "ok" and self.current["head_calls"] != 1:
                raise ValueError("successful provider call did not traverse exactly one native head capture")
            diagnostics = envelope.get("diagnostics", {})
            self.current["request_fingerprint"] = diagnostics.get("request_fingerprint")
            response = deepcopy(response)
            response.setdefault("ssdm", {}).setdefault("diagnostics", {})["native_intervention"] = {
                "arm": self.arm, "registration_sha256": self.registration["registration_sha256"],
                "capture_identity_sha256": self.registration["capture_identity_sha256"],
                "capture_boundary": BOUNDARY, "scope": SCOPE}
            return response
        except Exception:
            self.failed = True
            raise
        finally:
            self.current = None

    def pre_hook(self, _head, args):
        import torch
        try:
            from heads.features import HeadInput
            if self.current is None or not args or not isinstance(args[0], HeadInput):
                raise ValueError("native hidden edit reached an unbound head call")
            x = args[0]
            k, d = len(self.current["option_ids"]), self.semantic.numel()
            if (x.fork is None or tuple(x.fork.shape) != (1, k, d) or x.ks != [k]
                    or not bool(x.option_mask.all()) or not bool(torch.isfinite(x.fork).all())):
                raise ValueError("native capture does not cover the complete registered candidate domain")
            targeted = self.current["targeted"]
            if targeted and k not in self.outputs:
                raise ValueError("discovery artifact lacks a fitted output-only direction for this candidate count")
            original = x.fork
            wanted_norm = float(self.semantic.double().norm()) * float(self.registration["scale"]) * math.sqrt(k) if targeted else 0.0
            edited = original
            if targeted and self.arm in {"semantic", "random"}:
                direction = self.semantic if self.arm == "semantic" else self.random
                edited = original + direction.to(device=original.device, dtype=original.dtype).view(1, 1, d) * self.registration["scale"]
            actual_norm = float((edited - original).double().norm())
            if targeted and self.arm in {"semantic", "random"} and not math.isclose(actual_norm, wanted_norm, rel_tol=NORM_TOLERANCE, abs_tol=1e-12):
                raise ValueError("realized hidden edit is not norm matched")
            self.current["head_calls"] += 1
            self.current["capture"] = {"hidden_before_sha256": _tensor_sha(original),
                                       "hidden_after_sha256": _tensor_sha(edited),
                                       "shape": list(original.shape), "dtype": str(original.dtype),
                                       "hidden_delta_l2": actual_norm, "matched_delta_l2": wanted_norm,
                                       "output_delta_l2": 0.0}
            return (replace(x, fork=edited), *args[1:])
        except Exception:
            self.failed = True
            raise

    def post_hook(self, _head, _args, result):
        import torch
        try:
            if self.current is None or self.current["capture"] is None:
                raise ValueError("native output control has no bound hidden capture")
            capture = self.current["capture"]
            before, after = result.logits, result.logits
            if tuple(before.shape) != (1, len(self.current["option_ids"])) or not bool(torch.isfinite(before).all()):
                raise ValueError("native head logits do not cover the complete candidate domain")
            if self.current["targeted"] and self.arm == "output_only":
                direction = self.outputs[before.shape[1]].double()
                delta = direction * (capture["matched_delta_l2"] / direction.norm())
                after = before + delta.to(device=before.device, dtype=before.dtype).unsqueeze(0)
                actual = float((after - before).double().norm())
                if not math.isclose(actual, capture["matched_delta_l2"], rel_tol=NORM_TOLERANCE, abs_tol=1e-12):
                    raise ValueError("realized output-only edit is not norm matched")
                capture["output_delta_l2"] = actual
            capture.update(logits_before_sha256=_tensor_sha(before), logits_after_sha256=_tensor_sha(after))
            return replace(result, logits=after)
        except Exception:
            self.failed = True
            raise


def run_native_interventions(tasks: list[dict], parent, preregistration: dict) -> dict:
    """Callback for gate_evidence: identical free-running planner, four native arms.

    The gate runner supplies Stage 1 admission and pins the machine registration
    and discovery artifacts before evaluation. Failed/unsupported tasks stay in all
    arm denominators; missing controls never become an unedited success.
    """
    from .gate_planner import plan_gate_task
    import torch
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("nonempty gate task list required")
    for task in tasks:
        if (not isinstance(task, dict) or set(task) - _TASK_KEYS
                or any(not isinstance(task.get(key), str) or not task[key] for key in ("task_id", "world_id", "task_input"))):
            raise ValueError("only label-free gate task inputs are accepted")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("gate task IDs must be unique")
    native = _native(parent)
    model = native._loaded.backend.model
    head = model.head
    if head._forward_pre_hooks or head._forward_hooks:
        raise ValueError("native head already has an instrumentation owner")
    p, capture_identity, semantic, random, outputs, fit_sha = _registration(preregistration, parent, tasks)
    if not _OWNER.acquire(blocking=False):
        raise ValueError("another native intervention experiment owns the instrumentation")
    handles = []
    arms = {arm: [] for arm in ARMS}
    traces = {arm: [] for arm in ARMS}
    try:
        before = _parent_state_sha(model)
        for arm in ARMS:
            provider = _ArmProvider(parent, arm, p, capture_identity, semantic, random, outputs)
            handles = [head.register_forward_pre_hook(provider.pre_hook), head.register_forward_hook(provider.post_hook)]
            try:
                for task in tasks:
                    start = len(provider.records)
                    provider.failed = False
                    with torch.inference_mode():
                        result = plan_gate_task(deepcopy(task), provider)
                    if provider.failed:
                        # D intentionally masks backend exceptions; preserve the
                        # denominator while distinguishing invalid instrumentation.
                        result = replace(result, status="runner_error", text=None,
                                         reason="native_intervention_failed")
                    arms[arm].append(result)
                    traces[arm].append(deepcopy(provider.records[start:]))
            finally:
                for handle in handles:
                    handle.remove()
                handles = []
        parent.planner_kwargs()
        after = _parent_state_sha(model)
        if before != after:
            raise ValueError("native parent bytes changed during the intervention experiment")
        provenance = [{"task_id": task["task_id"], "input_sha256": _sha(task), "parent_revision": p["parent_revision"],
                       "registration_sha256": p["registration_sha256"], "capture_identity_sha256": _sha(capture_identity),
                       "capture_boundary": BOUNDARY, "scope": SCOPE, "arms": list(ARMS),
                       "state_unchanged_sha256_before": before, "state_unchanged_sha256_after": after,
                       "unchanged_scope": "full_parent_state_dict", "direction_artifact_sha256": p["directions"]["sha256"],
                       "semantic_direction_sha256": _tensor_sha(semantic), "random_direction_sha256": _tensor_sha(random),
                       "output_only_fit_evidence_sha256": fit_sha,
                       "instrumentation_valid": all(
                           arms[arm][i].status != "runner_error" and bool(traces[arm][i])
                           and all(record["status"] == "ok" and record["head_calls"] == 1 and record["capture"] is not None
                                   for record in traces[arm][i]) for arm in ARMS),
                       "captures": {arm: traces[arm][i] for arm in ARMS}}
                      for i, task in enumerate(tasks)]
        if p["schema"] == SCHEMA_V2:
            for row in provenance:
                row["target_slots"] = list(p["target_slots"])
        result = {"arms": arms, "provenance": provenance,
                  "instrumentation_valid": all(row["instrumentation_valid"] for row in provenance)}
        if p["schema"] == SCHEMA:
            return result
        targeted_counts = {arm: sum(record["targeted"] and record["head_calls"] == 1
                                   and record["status"] == "ok" and record["capture"] is not None
                                   for task_trace in traces[arm] for record in task_trace) for arm in ARMS}
        targeted_cohort_valid = all(count > 0 for count in targeted_counts.values())
        result.update(targeted_capture_counts=targeted_counts, targeted_cohort_valid=targeted_cohort_valid,
                      instrumentation_valid=result["instrumentation_valid"] and targeted_cohort_valid)
        return result
    finally:
        for handle in handles:
            handle.remove()
        _OWNER.release()
