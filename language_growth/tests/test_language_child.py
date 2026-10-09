from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from language_growth.gate_suites import build_suite
from language_growth.gates import run_tasks
from language_growth.gates import main as gates_main
from language_growth.gate_decision_retention import decision_identity
from language_growth.language_child import (
    FIT_RECEIPT_SCHEMA, LanguageChildError, load_language_child,
)
from language_growth.tests.test_language_gates import FakeParent
from language_growth.tests.test_planner import EXPECTED_CONFIG, REVISION, ScriptedProvider


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FakeCalibration:
    def identity(self):
        return deepcopy(EXPECTED_CONFIG["calibration"])


class FakeDecider(ScriptedProvider):
    def __init__(self, selectors):
        super().__init__(selectors)
        self.calibration = FakeCalibration()

    def revision(self):
        return {"model_id": "scripted", "revision_id": REVISION}


class FakeNativeParent:
    def __init__(self, decider, checkpoint_sha):
        self.decider = decider
        self.checkpoint_sha = checkpoint_sha

    def identity(self):
        return {"artifact_sha256": self.checkpoint_sha, "revision_id": REVISION}

    def planner_kwargs(self):
        return {"model": "scripted", "parent_revision": REVISION,
                "expected_served_config": deepcopy(EXPECTED_CONFIG)}

    def decide(self, request):
        return self.decider.decide(request)

    def plan(self, task_input, host, catalog, *, allow_repair=False):
        from language_growth.planner import plan_checked_answer
        return plan_checked_answer(task_input, host, catalog, self,
                                   allow_repair=allow_repair, **self.planner_kwargs())


class FakeStage1Spec:
    def __init__(self):
        self.qualification = {
            "checkpoint_config": {"seed": 17, "layout": "ordered", "max_steps": 50,
                                   "ckpt_dir": "parent-checkpoints", "ckpt_every": 10,
                                   "eval_every": 10, "require_guard": True},
            "checkpoint_config_hash": "f" * 16,
            "checkpoint_lineage": {
                "backbone": {"spec": "rwkv7-13.3b", "weights_digest": "1" * 64},
                "tokenizer": ["pinned-tokenizer"],
                "layout": {"layouts": [["ordered", "jrt-rwkv-g1-v1"]]},
                "data": {"n": 100, "sha256": "2" * 64, "order_fn": "parent-fit"},
            },
            "lora_modules": [],
        }
        self.pin = type("Pin", (), {"artifact_sha256": "a" * 64})()

    def identity(self):
        return {"schema": "ssdm.stage1.qualification.v1",
                "artifact_sha256": self.pin.artifact_sha256,
                "report_sha256": "b" * 64, "qualification_sha256": "c" * 64,
                "parent_revision": "qualified-parent-revision",
                "language_qualified": False}


def receipt_for(spec, child_sha):
    lineage = deepcopy(spec.qualification["checkpoint_lineage"])
    lineage["data"] = {"n": 3600, "sha256": "3" * 64, "order_fn": "train.v1"}
    return {
        "schema": FIT_RECEIPT_SCHEMA, "status": "completed_fit_only",
        "plan_sha256": "9f420d057a7cb3a00703793ac1f1f1a65cf5ae7ea5d9a6d8d701db985101fb8f",
        "order_sha256": "4" * 64, "source_commit": "5" * 40,
        "ordered_by": "agent-a", "order_ref": "A-order-1",
        "stage1_parent": spec.identity(),
        "guard": {"artifact_sha256": "6" * 64, "fingerprint_sha256": "7" * 64},
        "recipe": {"config_hash": "f" * 16, "max_steps": 2, "ckpt_every": 0,
                   "diagnostic_batch_size": 1, "max_seconds": 30},
        "resources": {"host": "fit-host", "device": "cuda:0", "dtype": "bfloat16",
                      "isolated": True, "max_peak_gb": 80, "min_free_gb": 40},
        "spend": {"authorized": True, "merged_proof_commit": "8" * 40,
                  "acknowledged_issue": "issue-123", "max_cost_usd": 2},
        "split": {"training_rows": 3600, "diagnostic_rows": 1200},
        "item_sidecar_sha256": "9" * 64,
        "training_rows": 3600, "diagnostic_rows": 1200,
        "language_qualified": False, "promotion_eligible": False,
        "checkpoints": [{"path": "ckpt-step0000002.pt", "sha256": child_sha, "step": 2}],
        "config": {**deepcopy(spec.qualification["checkpoint_config"]), "max_steps": 2,
                   "ckpt_dir": "child-checkpoints", "ckpt_every": 0, "eval_every": 0},
        "config_hash": "f" * 16,
        "lora_modules": [], "trainable_parameter_names": ["head.weight"],
        "diagnostic": {"scope": "teacher-forced development_train fit worlds only", "metrics": {"loss": 0.5}},
        "steps": 2, "lineage": lineage, "seconds": 1.0,
    }


def checkpoint_state(spec, receipt):
    return {
        "step": 2, "epoch": 0, "cursor": 2,
        "config": deepcopy(receipt["config"]), "config_hash": "f" * 16,
        "trainable": {"head.weight": object()}, "optimizer": {}, "scheduler": {}, "rng": {},
        "lineage": deepcopy(receipt["lineage"]), "init_seed": 17, "history": [],
        "backbone": "rwkv7-13.3b", "lora_modules": [], "guard": {"n_ids": 1},
    }


class FakeRuntime:
    def __init__(self, spec, state):
        self.spec, self.state = spec, state
        self.checkpoint_reads = 0
        self.parent_loads = 0
        self.restores = 0
        self.decider_value = None

    def inspect_parent(self, **kwargs):
        return self.spec

    def checkpoint_state(self, path):
        self.checkpoint_reads += 1
        return deepcopy(self.state)

    def trainable_parent(self, *args, **kwargs):
        self.parent_loads += 1
        return type("Parent", (), {"identity": self.spec.identity(),
                                   "trainable_parameter_names": ("head.weight",)})()

    def restore_child(self, parent, state, candidate):
        self.restores += 1

    def native_parent(self, parent, state, candidate, stage1):
        return FakeNativeParent(self.decider_value, candidate["checkpoint_sha256"])


class MiniTensor:
    layout = "strided"

    def __init__(self, shape=(2,), dtype="float32", finite=True, value=0):
        self.shape, self.dtype, self.finite, self.value = shape, dtype, finite, value


class MiniParameter(MiniTensor):
    requires_grad = True

    def copy_(self, value):
        self.value = value.value


class MiniModel:
    def __init__(self):
        self.parameters = {"a": MiniParameter(value=1), "b": MiniParameter(value=2)}

    def named_parameters(self):
        return list(self.parameters.items())


class MiniFinite:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


class LanguageChildTests(unittest.TestCase):
    def _files(self, root, *, receipt_mutate=None, state_mutate=None, child_bytes=b"child-checkpoint"):
        spec = FakeStage1Spec()
        child = Path(root) / "child.pt"
        child.write_bytes(child_bytes)
        receipt = receipt_for(spec, sha(child_bytes))
        if receipt_mutate:
            receipt_mutate(receipt)
        receipt_path = Path(root) / "receipt.json"
        raw = canonical(receipt)
        receipt_path.write_bytes(raw)
        state = checkpoint_state(spec, receipt)
        if state_mutate:
            state_mutate(state)
        return spec, child, receipt_path, sha(raw), state

    def test_receipt_to_distinct_candidate_runs_free_running_gate_path(self):
        tasks, refs = build_suite("L0", "development_validation", worlds=2, seed=23)
        task, ref = tasks[0], refs[0]
        target = next(n for n in ref["program"]["nodes"] if n["op"] == "ReadFact")
        selectors = ["ReadFact", f"{target['record_ref']}) field {target['field']}"]
        decider = FakeDecider(selectors)
        with TemporaryDirectory() as directory:
            spec, child, receipt, receipt_sha, state = self._files(directory)
            runtime = FakeRuntime(spec, state)
            runtime.decider_value = decider
            candidate = load_language_child(child, stage1_checkpoint=Path(directory) / "parent.pt",
                receipt=receipt, receipt_sha256=receipt_sha, report=Path("report"), report_sha256="b" * 64,
                qualification=Path("qualification"), qualification_sha256="c" * 64,
                snapshot=Path("snapshot"), runtime=runtime)
            identity = candidate.identity()
            self.assertEqual(decision_identity(identity), {
                "artifact_sha256": sha(child.read_bytes()), "revision_id": REVISION})
            self.assertNotEqual(identity["candidate_checkpoint_sha256"], identity["stage1_parent"]["artifact_sha256"])
            self.assertEqual(identity["candidate_revision"], REVISION)
            self.assertEqual(identity["stage1_parent"], spec.identity())
            self.assertFalse(identity["language_qualified"])
            self.assertFalse(identity["stage1_qualification_inherited"])
            self.assertFalse(identity["calibration"]["child_calibration_measured"])
            rows = run_tasks([task], [ref], candidate)
            self.assertTrue(rows[0]["correct"], rows[0])
            self.assertEqual((runtime.checkpoint_reads, runtime.parent_loads, runtime.restores), (1, 1, 1))

    def test_receipt_parent_mismatch_refuses_before_checkpoint_or_parent_loader(self):
        with TemporaryDirectory() as directory:
            spec, child, receipt, receipt_sha, state = self._files(
                directory, receipt_mutate=lambda r: r["stage1_parent"].update(artifact_sha256="0" * 64))
            runtime = FakeRuntime(spec, state)
            runtime.decider_value = FakeDecider([])
            with self.assertRaisesRegex(LanguageChildError, "ancestry"):
                load_language_child(child, stage1_checkpoint=Path(directory) / "parent.pt",
                    receipt=receipt, receipt_sha256=receipt_sha, report=Path("report"), report_sha256="b" * 64,
                    qualification=Path("qualification"), qualification_sha256="c" * 64,
                    snapshot=Path("snapshot"), runtime=runtime)
            self.assertEqual((runtime.checkpoint_reads, runtime.parent_loads), (0, 0))

    def test_candidate_file_hash_mismatch_refuses_before_checkpoint_or_parent_loader(self):
        with TemporaryDirectory() as directory:
            spec, child, receipt, receipt_sha, state = self._files(directory)
            child.write_bytes(b"different bytes")
            runtime = FakeRuntime(spec, state)
            with self.assertRaisesRegex(LanguageChildError, "checkpoint"):
                load_language_child(child, stage1_checkpoint=Path(directory) / "parent.pt",
                    receipt=receipt, receipt_sha256=receipt_sha, report=Path("report"), report_sha256="b" * 64,
                    qualification=Path("qualification"), qualification_sha256="c" * 64,
                    snapshot=Path("snapshot"), runtime=runtime)
            self.assertEqual((runtime.checkpoint_reads, runtime.parent_loads), (0, 0))

    def test_child_step_lineage_and_config_mismatches_refuse_before_parent_loader(self):
        mutations = (
            lambda state: state.update(step=1),
            lambda state: state["lineage"]["data"].update(sha256="0" * 64),
            lambda state: state.update(config_hash="0" * 16),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as directory:
                spec, child, receipt, receipt_sha, state = self._files(directory, state_mutate=mutate)
                runtime = FakeRuntime(spec, state)
                with self.assertRaises(LanguageChildError):
                    load_language_child(child, stage1_checkpoint=Path(directory) / "parent.pt",
                        receipt=receipt, receipt_sha256=receipt_sha, report=Path("report"), report_sha256="b" * 64,
                        qualification=Path("qualification"), qualification_sha256="c" * 64,
                        snapshot=Path("snapshot"), runtime=runtime)
                self.assertEqual(runtime.parent_loads, 0)

    def test_trainable_restore_preflights_all_tensor_keys_shapes_dtypes_and_finiteness(self):
        fake_torch = SimpleNamespace(Tensor=MiniTensor, strided="strided",
            isfinite=lambda value: MiniFinite(value.finite),
            no_grad=lambda: __import__("contextlib").nullcontext())
        parent = type("Parent", (), {"model": MiniModel(), "trainable_parameter_names": ("a", "b")})()
        runtime = __import__("language_growth.language_child", fromlist=["Runtime"]).Runtime()

        valid = {"trainable": {"a": MiniTensor(value=10), "b": MiniTensor(value=20)}}
        with patch.dict(sys.modules, {"torch": fake_torch}):
            runtime.restore_child(parent, valid, {})
        self.assertEqual([parent.model.parameters[name].value for name in ("a", "b")], [10, 20])

        invalid = (
            {"a": MiniTensor(value=30)},
            {"a": MiniTensor(value=30), "b": MiniTensor(value=40), "extra": MiniTensor()},
            {"a": MiniTensor(value=30), "b": MiniTensor(shape=(3,), value=40)},
            {"a": MiniTensor(value=30), "b": MiniTensor(dtype="float64", value=40)},
            {"a": MiniTensor(value=30), "b": MiniTensor(finite=False, value=40)},
        )
        for weights in invalid:
            before = [parent.model.parameters[name].value for name in ("a", "b")]
            with self.subTest(weights=weights), patch.dict(sys.modules, {"torch": fake_torch}), \
                 self.assertRaises(LanguageChildError):
                runtime.restore_child(parent, {"trainable": weights}, {})
            self.assertEqual([parent.model.parameters[name].value for name in ("a", "b")], before)

    def test_gate_cli_requires_complete_child_receipt_and_rejects_custom_loader(self):
        base = ["--parent", "child.pt", "--run-development", "--out", "fresh-output",
                "--parent-report", "report.json", "--parent-report-sha256", "1" * 64,
                "--parent-qualification", "qualification.json", "--parent-qualification-sha256", "2" * 64,
                "--parent-snapshot", "snapshot"]
        cases = (
            base + ["--fit-receipt", "receipt.json"],
            base + ["--stage1-parent-checkpoint", "parent.pt", "--fit-receipt", "receipt.json",
                    "--fit-receipt-sha256", "3" * 64, "--parent-loader", "custom:loader"],
        )
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gates_main(args), 2)


if __name__ == "__main__":
    unittest.main()
