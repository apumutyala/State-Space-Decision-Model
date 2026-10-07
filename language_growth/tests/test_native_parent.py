"""Tiny CPU cache-fork + LoRA contract fixture; no released D weights/capability."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class NativeParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from decide.calibration import FeatureTemperature, content_set_sha256, disjointness_proof_sha256
        from heads.layout import ByteEncoder, build_row

        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.folder = tempfile.TemporaryDirectory(prefix="native-parent-cpu-fixture-")
        cls.path = Path(cls.folder.name) / "fixture.pt"
        model = cls._model()
        request = {"model": "tiny", "state": "Synthetic development fixture.",
                   "questions": {"q": {"type": "choice", "criteria": {"a": "one", "b": "two"}}}}
        row = build_row(request, "q", ByteEncoder())
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss = -model([row]).log_probs[0, 1]
        loss.backward()
        optimizer.step()
        cls.trained_state = deepcopy(model.state_dict())
        assert any(not torch.equal(value, cls.trained_state[name]) for name, value in before.items()
                   if name.startswith("head."))
        assert any(not torch.equal(value, cls.trained_state[name]) for name, value in before.items()
                   if name.endswith("lora_B"))
        calib = (hashlib.sha256(b"synthetic fixture calibration").hexdigest(),)
        fitted_on = content_set_sha256(calib)
        disjointness = {"eval": content_set_sha256([hashlib.sha256(b"synthetic fixture eval").hexdigest()])}
        cls.calibration = FeatureTemperature(w=(0.5, 0.03, 0.1), fitted_on=fitted_on,
            calib_content=calib, disjointness=tuple(disjointness.items()),
            proof_sha256=disjointness_proof_sha256(fitted_on, disjointness))
        cls._save(cls.path, cls.trained_state)
        cls.pin = cls._pin(cls.path)

    @classmethod
    def tearDownClass(cls):
        import torch
        cls.folder.cleanup()
        torch.set_num_threads(cls.old_threads)

    @staticmethod
    def _model():
        import torch
        from heads import DecisionModel
        from heads.backbone import tiny_backbone
        from train.lora import apply_lora
        backbone = tiny_backbone(seed=17, hidden=8, layers=1, fork_mode="cache")
        apply_lora(backbone.model, backbone.spec.lora_groups, rank=2,
                   generator=torch.Generator().manual_seed(17))
        return DecisionModel(backbone, "fork", seed=17)

    @classmethod
    def _save(cls, path, state):
        import torch
        torch.save({"fixture_format": "tiny-cache-fork-lora-v1", "state": state,
                    "calibration": cls.calibration.to_dict()}, path)

    @classmethod
    def _loader(cls, path, *, artifact_sha256):
        """Fixture only. This format is deliberately unrelated to PR #74."""
        import torch
        from decide.calibration import FeatureTemperature
        from heads import ByteEncoder
        from heads.decide_backend import DecisionModelBackend
        from language_growth.native_parent import LoadedNativeParent
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (set(payload) != {"fixture_format", "state", "calibration"}
                or payload["fixture_format"] != "tiny-cache-fork-lora-v1"):
            raise ValueError("wrong CPU fixture format")
        model = cls._model()
        model.load_state_dict(payload["state"], strict=True)
        backend = DecisionModelBackend(model, ByteEncoder(), layout="ordered",
            template="jrt-plain-v1", checkpoint={"artifact_sha256": artifact_sha256,
            "scope": "synthetic CPU contract fixture"}, shared_prefix=True)
        return LoadedNativeParent(backend, FeatureTemperature.from_dict(payload["calibration"]), 32768)

    @classmethod
    def _pin(cls, path):
        from decide.adapter import Decider
        from language_growth.native_parent import NativeParentPin
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = cls._loader(path, artifact_sha256=sha)
        decider = Decider(loaded.backend, calibration=loaded.calibration)
        return NativeParentPin(sha, deepcopy(decider.revision()), loaded.calibration.identity(),
            {"fork_mode": "cache", "shared_prefix": True, "max_input_tokens": 32768})

    def test_exact_cpu_reload_calibration_and_free_running_planner(self):
        import torch
        from heads.layout import build_row
        from language_growth.development_data import generate_development_tasks
        from language_growth.native_parent import load_native_parent
        from language_growth.planner import plan_checked_answer
        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        parent = load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        self.assertTrue(all(p.device.type == "cpu" and not p.requires_grad
                            for p in loaded.backend.model.parameters()))
        self.assertFalse(loaded.backend.model.training)
        self.assertTrue(all(torch.equal(value, loaded.backend.model.state_dict()[name])
                            for name, value in self.trained_state.items()))
        self.assertIn("+T:", parent.planner_kwargs()["parent_revision"])
        example = generate_development_tasks(19, 1, choice_format="production",
                                              split="development_validation")[0]
        calls = []
        class Recording:
            def decide(inner, request):
                response = parent.decide(request)
                calls.append((deepcopy(request), response))
                return response
        training = example["training_only"]
        task = json.loads(example["model_inputs"][0]["state"])["task_input"]
        result = plan_checked_answer(task, training["host"], training["catalog"],
                                     Recording(), **parent.planner_kwargs())
        self.assertIn(result.status, ("accepted", "kernel_rejected"))
        self.assertGreater(result.provider_calls, 0)
        self.assertEqual(result.provider_calls, result.forward_calls)
        self.assertTrue(any("kernel_status" in step for step in result.trace))
        for request, response in calls:
            env = response["ssdm"]
            self.assertEqual(env["status"], "ok")
            self.assertEqual(env["diagnostics"]["revision_id"], self.pin.revision["revision_id"])
            qid = next(iter(request["questions"]))
            row = build_row(request, qid, loaded.backend.encode)
            expected = loaded.calibration.temperature_for(row.n_options, {"evidence_tokens": row.evidence_len})
            self.assertAlmostEqual(env["served_config"]["temperature_per_question"][qid], expected)
            self.assertNotEqual(env["raw_probabilities"], env["served_probabilities"])
            state = json.loads(request["state"])
            self.assertNotIn("training_only", state)
            self.assertNotIn("expected", state)
        direct = parent.plan(task, training["host"], training["catalog"])
        self.assertEqual(direct.program, result.program)
        class JsonTransport:
            def decide(inner, request):
                return json.loads(json.dumps(parent.decide(request), allow_nan=False))
        transported = plan_checked_answer(task, training["host"], training["catalog"],
                                          JsonTransport(), **parent.planner_kwargs())
        self.assertEqual(transported.status, result.status)
        self.assertEqual(transported.program, result.program)

    def test_hash_mismatch_refuses_before_loader(self):
        from language_growth.native_parent import load_native_parent
        calls = []
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            load_native_parent(self.path, lambda *a, **k: calls.append(a),
                               pin=replace(self.pin, artifact_sha256="0" * 64))
        self.assertEqual(calls, [])

    def test_revision_calibration_and_runtime_pins_are_independent(self):
        from language_growth.native_parent import load_native_parent
        variants = []
        for key, value in (("layout", "symmetric"), ("encoder", {"kind": "wrong"}),
                           ("prompt_template", {"id": "wrong"}), ("model_id", "stock")):
            revision = deepcopy(self.pin.revision)
            revision[key] = value
            variants.append(replace(self.pin, revision=revision))
        variants.append(replace(self.pin, calibration={"kind": "wrong"}))
        for key, value in (("fork_mode", "recompute"), ("shared_prefix", False), ("max_input_tokens", 123)):
            runtime = deepcopy(self.pin.runtime)
            runtime[key] = value
            variants.append(replace(self.pin, runtime=runtime))
        for pin in variants:
            with self.subTest(pin=pin), self.assertRaisesRegex(ValueError, "serving pins"):
                load_native_parent(self.path, self._loader, pin=pin)

    def test_head_and_lora_artifacts_change_identity_on_same_backbone(self):
        from language_growth.native_parent import load_native_parent
        for suffix in ("lora_B", "head.mlp.0.weight"):
            state = deepcopy(self.trained_state)
            name = next(name for name in state if name.endswith(suffix))
            state[name].add_(0.25)
            path = Path(self.folder.name) / ("lora.pt" if suffix == "lora_B" else "head.pt")
            self._save(path, state)
            pin = self._pin(path)
            self.assertEqual(pin.revision["backbone"], self.pin.revision["backbone"])
            self.assertNotEqual(pin.revision["revision_id"], self.pin.revision["revision_id"])
            load_native_parent(path, self._loader, pin=pin)
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                load_native_parent(path, self._loader, pin=self.pin)

    def test_strict_loader_refuses_missing_trainable_key(self):
        from language_growth.native_parent import load_native_parent
        state = deepcopy(self.trained_state)
        del state[next(name for name in state if name.endswith("lora_B"))]
        path = Path(self.folder.name) / "incomplete.pt"
        self._save(path, state)
        pin = replace(self.pin, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            load_native_parent(path, self._loader, pin=pin)

    def test_fitted_calibration_requires_its_proof_before_inference(self):
        from decide.calibration import CalibrationNotServable
        from language_growth.native_parent import load_native_parent
        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        loaded = replace(loaded, calibration=replace(loaded.calibration, proof_sha256=None))
        with self.assertRaises(CalibrationNotServable):
            load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)

    def test_refuses_unbound_artifact_and_live_config_drift(self):
        from language_growth.native_parent import load_native_parent
        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        loaded.backend.checkpoint["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "did not bind"):
            load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        parent = load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        kwargs = parent.planner_kwargs()
        kwargs["expected_served_config"]["calibration"]["id"] = "changed returned copy"
        self.assertEqual(parent.planner_kwargs()["expected_served_config"]["calibration"], self.pin.calibration)
        loaded.backend.layout = "symmetric"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            parent.planner_kwargs()

    def test_live_backbone_config_and_capture_drift_reject_before_use(self):
        from types import SimpleNamespace
        from heads.rwkv7_wkv import select_wkv
        from language_growth.native_parent import load_native_parent

        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        parent = load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        backbone = loaded.backend.model.backbone
        # The tiny model does not otherwise use WKV; this exercises the same
        # public config switch that changes the RWKV-7 inference path.
        backbone.model.config = SimpleNamespace(**vars(backbone.model.config),
                                                wkv_implementation="tokenwise")
        select_wkv(backbone.model, "chunked")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            parent.planner_kwargs()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            parent.decide({})

        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        parent = load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        loaded.backend.model.backbone.capture = "changed-capture"
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            parent.planner_kwargs()

        loaded = self._loader(self.path, artifact_sha256=self.pin.artifact_sha256)
        parent = load_native_parent(self.path, lambda *a, **k: loaded, pin=self.pin)
        loaded.backend.model.backbone.layer = 0
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            parent.planner_kwargs()


if __name__ == "__main__":
    unittest.main()
