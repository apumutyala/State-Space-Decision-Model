"""CPU-light contract tests for the Stage 1 trainable-parent entry point."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

from language_growth.trainable_parent import (
    TrainableQualifiedParent,
    build_prepared_language_trainer,
    load_pinned_split_guard,
    load_qualified_trainable_parent,
    split_guard_fingerprint,
)


class FakeParameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


class FakeModel:
    def named_parameters(self):
        return iter([("backbone.weight", FakeParameter(False)), ("head.weight", FakeParameter(True)),
                     ("backbone.block.lora_A", FakeParameter(True))])


class FakeConfig:
    EXECUTION = ("row_batch", "act_ckpt", "compile_blocks")

    def __init__(self, config_hash="same", *, row_batch=8, act_ckpt=True, compile_blocks=True):
        self._config_hash = config_hash
        self.row_batch, self.act_ckpt, self.compile_blocks = row_batch, act_ckpt, compile_blocks
        self.require_guard = True

    def hash(self):
        return self._config_hash


class PreparedTrainer:
    def __init__(self, model, config, train_items, diagnostic_items, *, prepared, guard):
        self.model, self.config = model, config
        self.train_items, self.diagnostic_items = train_items, diagnostic_items
        self.prepared, self.guard = prepared, guard
        self.lora_modules = []


class TrainableParentTests(unittest.TestCase):
    def test_stage1_refusal_happens_before_checkpoint_resolution_or_loader(self):
        qualified = ModuleType("language_growth.qualified_parent")
        qualified.inspect_qualified_parent = lambda **kwargs: (_ for _ in ()).throw(
            ValueError("Stage 1 not qualified"))
        qualified.file_sha256 = lambda path: "0" * 64
        with patch.dict(sys.modules, {"language_growth.qualified_parent": qualified}):
            with self.assertRaisesRegex(ValueError, "not qualified"):
                load_qualified_trainable_parent(
                    "missing-checkpoint.pt", report="missing-report.json", report_sha256="0" * 64,
                    qualification="missing-qualification.json", qualification_sha256="1" * 64,
                    snapshot="missing-snapshot")
        self.assertNotIn("torch", sys.modules)

    def test_returns_only_pinned_restored_trainables(self):
        content = b"synthetic test checkpoint bytes"
        digest = hashlib.sha256(content).hexdigest()
        revision = {"layout": "ordered", "prompt_template": {"id": "jrt-rwkv-g1-v1"},
                    "encoder": {"kind": "rwkv7-fixture"}}
        spec = SimpleNamespace(pin=SimpleNamespace(artifact_sha256=digest, revision=revision),
                               qualification={"lora_modules": ["backbone.block"]},
                               identity=lambda: {"stage1": "qualified-fixture", "artifact_sha256": digest})
        backend = SimpleNamespace(
            checkpoint={"sha256": digest}, layout="ordered",
            template=SimpleNamespace(identity=lambda: {"id": "jrt-rwkv-g1-v1"}),
            encode=SimpleNamespace(identity=lambda: {"kind": "rwkv7-fixture"}),
            model=FakeModel())
        loaded = SimpleNamespace(backend=backend, calibration=None)
        config = object()
        qualified = ModuleType("language_growth.qualified_parent")
        qualified.inspect_qualified_parent = lambda **kwargs: spec
        qualified.file_sha256 = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        loader_module = ModuleType("language_growth.rwkv7_m2_loader")
        loader_module.validate_reconstruction_metadata = lambda qualification, pin: config
        loader_calls = []
        pin_validation_calls = []
        native_module = ModuleType("language_growth.native_parent")
        native_module.validate_loaded_native_parent = lambda parent, pin: pin_validation_calls.append(
            (parent, pin))

        def make_loader(*args, **kwargs):
            loader_calls.append((args, kwargs))
            return lambda path, *, artifact_sha256: loaded

        loader_module.native_rwkv7_loader = make_loader
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parent.pt"
            checkpoint.write_bytes(content)
            with patch.dict(sys.modules, {"language_growth.qualified_parent": qualified,
                                          "language_growth.native_parent": native_module,
                                          "language_growth.rwkv7_m2_loader": loader_module}):
                parent = load_qualified_trainable_parent(
                    checkpoint, report="report.json", report_sha256="2" * 64,
                    qualification="qualification.json", qualification_sha256="3" * 64,
                    snapshot="snapshot", device="cuda:0", dtype="bfloat16")
        self.assertEqual(loader_calls, [(("snapshot", spec.qualification, spec.pin),
                                        {"device": "cuda:0", "dtype": "bfloat16"})])
        self.assertEqual(pin_validation_calls, [(loaded, spec.pin)])
        self.assertIs(parent.model, backend.model)
        self.assertIs(parent.encoder, backend.encode)
        self.assertIs(parent.config, config)
        self.assertEqual(parent.lora_modules, ("backbone.block",))
        self.assertEqual(parent.trainable_parameter_names,
                         ("backbone.block.lora_A", "head.weight"))
        self.assertEqual(parent.identity["artifact_sha256"], digest)
        self.assertNotIn("torch", sys.modules)

    def test_prepared_trainer_preserves_parent_modules_and_requires_same_execution(self):
        model = FakeModel()
        config = FakeConfig()
        parent = TrainableQualifiedParent(
            model=model, encoder=object(), config=config, lora_modules=("backbone.block",),
            trainable_parameter_names=("backbone.block.lora_A", "head.weight"), identity={"stage1": "pin"})
        trainer_module = ModuleType("train.trainer")
        trainer_module.Trainer = PreparedTrainer
        from train.guard import SplitGuard

        guard_payload = {"schema": "ssdm.agent_c.split_guard_pin.v1",
                         "manifests": {"/manifests/dev-manifest.jsonl.gz": "a" * 64,
                                       "/manifests/test-manifest.jsonl.gz": "b" * 64},
                         "ids": ["dev-id", "test-id"],
                         "hashes": ["c" * 64, "d" * 64],
                         "content_hashes": ["e" * 64],
                         "content_index": {"path": "content-index.gz",
                                           "sha256": hashlib.sha256(("e" * 64 + "\n").encode("ascii")).hexdigest(),
                                           "n": 1},
                         "note": "A/B precomputed hash-only DEV/TEST guard v1"}
        raw_guard = json.dumps(guard_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with TemporaryDirectory() as guard_dir:
            guard_path = Path(guard_dir) / "split-guard.json"
            guard_path.write_bytes(raw_guard)
            artifact_sha = hashlib.sha256(raw_guard).hexdigest()
            from train.guard import SplitGuard
            fixture_guard = SplitGuard(ids=set(guard_payload["ids"]), hashes=set(guard_payload["hashes"]),
                                       manifests=guard_payload["manifests"], note=guard_payload["note"],
                                       content_hashes=set(guard_payload["content_hashes"]),
                                       content_index=guard_payload["content_index"])
            guard_sha = split_guard_fingerprint(fixture_guard)
            pinned_guard = load_pinned_split_guard(
                guard_path, artifact_sha256=artifact_sha, expected_fingerprint_sha256=guard_sha)
            bad_payload = dict(guard_payload)
            bad_payload["content_index"] = {**guard_payload["content_index"], "sha256": "f" * 64}
            bad_raw = json.dumps(bad_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            bad_path = Path(guard_dir) / "bad-split-guard.json"
            bad_path.write_bytes(bad_raw)
            with self.assertRaisesRegex(ValueError, "content index identity"):
                load_pinned_split_guard(bad_path, artifact_sha256=hashlib.sha256(bad_raw).hexdigest(),
                                        expected_fingerprint_sha256=guard_sha)
        with patch.dict(sys.modules, {"train.trainer": trainer_module}):
            trainer = build_prepared_language_trainer(
                parent, config=FakeConfig(), train_items=["train"], diagnostic_items=["diagnostic"],
                pinned_guard=pinned_guard)
            self.assertTrue(trainer.prepared)
            self.assertIs(trainer.model, model)
            self.assertIs(trainer.guard, pinned_guard.guard)
            self.assertEqual(trainer.lora_modules, ["backbone.block"])
            with self.assertRaisesRegex(ValueError, "execution setting row_batch"):
                build_prepared_language_trainer(
                    parent, config=FakeConfig(row_batch=1), train_items=["train"],
                    diagnostic_items=["diagnostic"], pinned_guard=pinned_guard)
            with self.assertRaisesRegex(ValueError, "DEV/TEST manifests"):
                build_prepared_language_trainer(
                    parent, config=FakeConfig(), train_items=["train"],
                    diagnostic_items=["diagnostic"],
                    pinned_guard=type(pinned_guard)(SplitGuard.synthetic_only("test-only"),
                                                    artifact_sha, guard_sha))
        with self.assertRaisesRegex(ValueError, "TrainConfig"):
            build_prepared_language_trainer(
                parent, config=FakeConfig("different"), train_items=["train"],
                diagnostic_items=["diagnostic"], pinned_guard=pinned_guard)
        altered = SplitGuard(ids=set(pinned_guard.guard.ids) | {"extra-dev-id"},
                             hashes=set(pinned_guard.guard.hashes),
                             manifests=dict(pinned_guard.guard.manifests),
                             note=pinned_guard.guard.note,
                             content_hashes=set(pinned_guard.guard.content_hashes),
                             content_index=dict(pinned_guard.guard.content_index))
        tampered_guard = type(pinned_guard)(altered, artifact_sha, guard_sha)
        with self.assertRaisesRegex(ValueError, "pinned guard"):
            build_prepared_language_trainer(
                parent, config=FakeConfig(), train_items=["train"], diagnostic_items=["diagnostic"],
                pinned_guard=tampered_guard)


if __name__ == "__main__":
    unittest.main()
