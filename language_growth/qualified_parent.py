"""Fail-closed Stage 1 admission for the native M2 -> L0 bridge.

The independently pinned qualification is a reviewed host attestation, not a
model-authored certificate. This module checks its contract and byte bindings;
it does not rerun Stage 1, authenticate the reviewer, or turn a ledger keep into
qualification. No weights are fetched and preflight never reconstructs a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .native_parent import NativeParent, NativeParentLoader, NativeParentPin, load_native_parent

SCHEMA = "ssdm.stage1.qualification.v1"
FLOORS = {"safety", "calibration", "abstention", "control", "interpretability"}
COMPARATORS = {"clef-27b", "clef-flash"}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256 pin")
    return value


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON member")
        obj[key] = value
    return obj


def read_pinned_json(path: str | Path, digest: str) -> dict:
    raw = Path(path).resolve(strict=True).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _digest(digest):
        raise ValueError("JSON artifact hash mismatch")
    value = json.loads(raw, object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must be an object")
    return value


def _exact(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"incomplete or unsupported {name}")


@dataclass(frozen=True)
class QualifiedParentSpec:
    """Validated immutable-by-convention host metadata; no loaded model."""
    pin: NativeParentPin
    qualification: dict
    report_sha256: str
    qualification_sha256: str

    def identity(self) -> dict:
        q = self.qualification
        return deepcopy({"schema": SCHEMA, "artifact_sha256": self.pin.artifact_sha256,
                         "report_sha256": self.report_sha256,
                         "qualification_sha256": self.qualification_sha256,
                         "serving_bundle_sha256": q["serving_bundle_sha256"],
                         "parent_revision": self.pin.revision["revision_id"],
                         "checkpoint_config_hash": q["checkpoint_config_hash"],
                         "checkpoint_lineage_sha256": canonical_sha256(q["checkpoint_lineage"]),
                         "stage1": q["stage1"], "language_qualified": False})


def inspect_qualified_parent(*, report: str | Path, report_sha256: str,
                             qualification: str | Path, qualification_sha256: str) -> QualifiedParentSpec:
    """Validate pinned Stage 1 admission metadata before touching checkpoint/weights."""
    rep = read_pinned_json(report, report_sha256)
    q = read_pinned_json(qualification, qualification_sha256)
    _exact(q, {"schema", "verdict", "model_line", "artifact_sha256", "report_sha256",
               "serving_bundle_sha256", "checkpoint_config", "checkpoint_config_hash",
               "checkpoint_lineage", "lora_modules", "snapshot_files_sha256", "rwkv7_wkv", "stage1"}, "qualification")
    if q["schema"] != SCHEMA or q["verdict"] != "qualified" or q["model_line"] != "rwkv7-13.3b":
        raise ValueError("Stage 1 parent is not qualified on the selected RWKV7 line")
    pin = NativeParentPin.from_m2_report(rep)
    if (pin.revision.get("model_id") != "rwkv7-13.3b"
            or q["artifact_sha256"] != pin.artifact_sha256 or q["report_sha256"] != report_sha256
            or q["serving_bundle_sha256"] != canonical_sha256(rep["identities"]["serving_bundle"])):
        raise ValueError("qualification does not bind the exact M2 serving artifact/report")
    # Local M2 keep/promotion is necessary but is not the Clef/safety Stage 1 gate.
    promotion = rep.get("promotion", {})
    if (rep.get("schema") != "ssdm.m2.gates.v1" or rep.get("exit_code") != 0
            or promotion.get("promotable") is not True or promotion.get("verdict") != "accept"
            or set(promotion.get("stages", [])) != {"g12", "g13"}
            or promotion.get("margins_pending_user_approval") is not False
            or promotion.get("underpowered") or promotion.get("failed")
            or any(rep.get("verdicts", {}).get(s, {}).get("accept") is not True for s in ("g12", "g13"))):
        raise ValueError("M2 report is incomplete or non-promotable")
    stage = q["stage1"]
    _exact(stage, {"evaluation_scope", "primary_metric", "preregistration_sha256",
                   "matched_evaluation_sha256", "paired_ci95", "floors",
                   "independent_rescore_sha256", "merged_commit"}, "Stage 1 evidence")
    if stage["evaluation_scope"] != "full_dev" or stage["primary_metric"] != "decision_index":
        raise ValueError("Stage 1 requires matched full-DEV decision Index qualification")
    for key in ("preregistration_sha256", "matched_evaluation_sha256", "independent_rescore_sha256"):
        _digest(stage[key])
    commit = stage["merged_commit"]
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Stage 1 evidence must identify its merged source commit")
    _exact(stage["paired_ci95"], COMPARATORS, "Clef comparisons")
    for ci in stage["paired_ci95"].values():
        _exact(ci, {"lower", "upper", "confidence", "paired", "matched"}, "paired CI")
        if (ci["paired"] is not True or ci["matched"] is not True
                or type(ci["confidence"]) not in (int, float) or ci["confidence"] != .95
                or any(type(ci[k]) not in (int, float) or not math.isfinite(ci[k]) for k in ("lower", "upper"))
                or not 0 < ci["lower"] <= ci["upper"]):
            raise ValueError("Stage 1 paired 95% CI must be above zero against every Clef model")
    _exact(stage["floors"], FLOORS, "Stage 1 floors")
    for floor in stage["floors"].values():
        _exact(floor, {"passed", "evidence_sha256"}, "floor evidence")
        if floor["passed"] is not True:
            raise ValueError("Stage 1 safety/control floor did not pass")
        _digest(floor["evidence_sha256"])
    # Parsing the config is cheap and establishes that no unsupported settings
    # can disappear into a reconstruction default.
    from .rwkv7_m2_loader import validate_reconstruction_metadata
    validate_reconstruction_metadata(q, pin)
    return QualifiedParentSpec(pin, deepcopy(q), report_sha256, qualification_sha256)


class QualifiedParent:
    def __init__(self, parent: NativeParent, spec: QualifiedParentSpec):
        self._parent, self._spec = parent, deepcopy(spec)

    def identity(self) -> dict:
        self._parent.planner_kwargs()
        return self._spec.identity()

    def planner_kwargs(self) -> dict:
        return self._parent.planner_kwargs()

    def decide(self, request: dict) -> dict:
        return self._parent.decide(request)

    def plan(self, task_input: str, host: dict, catalog: dict, *, allow_repair: bool = False):
        return self._parent.plan(task_input, host, catalog, allow_repair=allow_repair)


def load_qualified_parent(checkpoint: str | Path, *, report: str | Path, report_sha256: str,
                          qualification: str | Path, qualification_sha256: str,
                          loader: NativeParentLoader | None = None, snapshot: str | Path | None = None,
                          device: str = "cpu", dtype: str = "bfloat16") -> QualifiedParent:
    spec = inspect_qualified_parent(report=report, report_sha256=report_sha256,
                                    qualification=qualification, qualification_sha256=qualification_sha256)
    if loader is None:
        if snapshot is None:
            raise ValueError("strict native reconstruction requires a local pinned snapshot")
        from .rwkv7_m2_loader import native_rwkv7_loader
        loader = native_rwkv7_loader(snapshot, spec.qualification, spec.pin, device=device, dtype=dtype)
    return QualifiedParent(load_native_parent(checkpoint, loader, pin=spec.pin), spec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", required=True)
    p.add_argument("--parent-report", required=True)
    p.add_argument("--parent-report-sha256", required=True)
    p.add_argument("--parent-qualification", required=True)
    p.add_argument("--parent-qualification-sha256", required=True)
    p.add_argument("--parent-snapshot")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    p.add_argument("--task", help="JSON with task_input, host, catalog; omit for metadata-only preflight")
    a = p.parse_args(argv)
    kwargs = dict(report=a.parent_report, report_sha256=a.parent_report_sha256,
                  qualification=a.parent_qualification, qualification_sha256=a.parent_qualification_sha256)
    if not a.task:
        spec = inspect_qualified_parent(**kwargs)
        print(json.dumps({"status": "metadata_validated", "model_loaded": False, **spec.identity()}, sort_keys=True))
    else:
        parent = load_qualified_parent(a.parent, **kwargs, snapshot=a.parent_snapshot, device=a.device, dtype=a.dtype)
        task = json.loads(Path(a.task).read_text(encoding="utf-8"), object_pairs_hook=_unique)
        from .public_output import public_answer
        print(json.dumps(public_answer(task["task_input"], task["host"], task["catalog"], parent,
                                       **parent.planner_kwargs()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
