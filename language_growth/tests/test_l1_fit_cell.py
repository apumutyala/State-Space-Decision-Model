"""Synthetic admission/preparation tests; no Torch, model, optimizer, or training."""
from dataclasses import dataclass
import builtins
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import ClassVar
import unittest
from unittest.mock import patch

from language_growth.l1_fit_cell import (
    FitCellError, FitInputs, ORDER_SCHEMA, PLAN_SHA256, SOURCES, ValidatedFit,
    _child_config, canonical, execute_fit, main, prepare_items, sha, validate_fit,
)
from language_growth.language_child import validate_child_checkpoint_state, validate_fit_child_receipt


class Encoder:
    def identity(self):
        return {"kind": "synthetic-test-encoder", "vocab_sha256": "1" * 64}


class Guard:
    def __init__(self):
        self.checked = 0

    def check(self, record):
        self.checked += 1
        if record.get("forbidden"):
            raise FitCellError("forbidden synthetic test row")


def items_factory(records, encoder, **kwargs):
    result = []
    for r in records:
        qid = next(iter(r["request"]["questions"]))
        row = SimpleNamespace(qid=qid, ids=[1, 2, 3], n_options=2,
                              continuations=[[4], [5]], evidence_len=1)
        result.append(SimpleNamespace(id=f"{r['id']}/{qid}", row=row, label=0))
    return result


def fixture():
    records, census, sources = {}, {}, {}
    for source, per_world in zip(SOURCES, (128, 72)):
        worlds = [f"{source}-world-{w:02d}" for w in range(24)]
        train, diag = worlds[:18], worlds[18:]
        rows, tokens = [], []
        for world in worlds:
            for index in range(per_world):
                task = f"{world}-task-{index}"
                qid = "slot-001-semantic-operation"
                rows.append({"id": task, "group_id": world,
                             "request": {"questions": {qid: {"criteria": {"a": "A", "b": "B"}}}},
                             "labels": {qid: "a"},
                             "metadata": {"world_id": world, "slot": "semantic-operation"}})
                tokens.append({"record_line": len(rows), "record_id": task, "slot": qid,
                               "prompt_tokens": 3, "options": 2, "continuation_tokens": [1, 1],
                               "max_continuation_tokens": 1,
                               "longest_prompt_plus_continuation_tokens": 4, "state_tokens": 1})
        records[source], census[source] = rows, tokens
        sources[source] = {"source": {"training_worlds": train, "diagnostic_worlds": diag,
                                      "training_rows": 18 * per_world, "diagnostic_rows": 6 * per_world}}
    split = {"training_worlds": sum((s["source"]["training_worlds"] for s in sources.values()), []),
             "diagnostic_worlds": sum((s["source"]["diagnostic_worlds"] for s in sources.values()), []),
             "training_rows": 3600, "diagnostic_rows": 1200}
    summaries = {s: {"encoder_identity": Encoder().identity(),
                      "tokenizer_sha256": {"tokenizer.json": "2" * 64}} for s in SOURCES}
    return ValidatedFit({"sources": sources, "split": split, "training_authorized": False},
                        records, census, summaries)


@dataclass
class Loss:
    kd: bool = False
    order_consistency_weight: float = 0


@dataclass
class Config:
    EXECUTION: ClassVar = ("row_batch",)
    loss: object = None
    axis_loss_weight: float = 0
    layout: str = "ordered"
    row_batch: int = 4
    lr_head: float = .001
    max_steps: int = 50
    ckpt_every: int = 10
    ckpt_dir: str = "parent-directory"
    eval_every: int = 5
    require_guard: bool = True

    def __post_init__(self):
        if self.loss is None:
            self.loss = Loss()

    def hash(self):
        return "f" * 16

    def to_json(self):
        return {"max_steps": self.max_steps, "row_batch": self.row_batch,
                "lr_head": self.lr_head, "ckpt_dir": self.ckpt_dir,
                "eval_every": self.eval_every, "require_guard": self.require_guard}


def order():
    return {"schema": ORDER_SCHEMA, "ordered_by": "agent-a", "order_ref": "synthetic-test-A-order",
            "plan_sha256": PLAN_SHA256, "source_commit": "a" * 40,
            "parent": {"report_sha256": "b" * 64, "qualification_sha256": "c" * 64,
                       "checkpoint_sha256": "d" * 64},
            "guard": {"artifact_sha256": "e" * 64, "fingerprint_sha256": "0" * 64},
            "recipe": {"config_hash": "f" * 16, "max_steps": 2, "ckpt_every": 0,
                       "diagnostic_batch_size": 1, "max_seconds": 30},
            "resources": {"host": "isolated-test-host", "device": "cuda:0", "dtype": "bfloat16",
                          "isolated": True, "max_peak_gb": 80, "min_free_gb": 40},
            "spend": {"authorized": True, "merged_proof_commit": "1" * 40,
                      "acknowledged_issue": "synthetic-test-issue", "max_cost_usd": 1}}


class FakeTrainer:
    """Records calls only; its step method does no model computation or training."""
    def __init__(self, config):
        self.config, self.step = config, 0
        self.lineage = {
            "backbone": {"spec": "rwkv7-13.3b", "weights_digest": "1" * 64,
                         "weights_digest_mode": "sampled"},
            "tokenizer": ["fake-tokenizer"],
            "layout": {"layouts": [["ordered", "jrt-rwkv-g1-v1"]],
                       "body_version": "test", "layout_version": "test"},
            "data": {"n": 3600, "sha256": "9" * 64, "order_fn": "train.v1"},
        }

    def train_step(self):
        self.step += 1
        return .5

    def save(self):
        directory = Path(self.config.ckpt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fake-child-{self.step}.bin"
        path.write_bytes(b"fake child receipt fixture; contains no tensors")
        return path


class FakeRuntime:
    def __init__(self):
        self.calls, self.config_value = [], Config()
        self.guard_value = Guard()
        self.identity = {"artifact_sha256": "d" * 64, "parent_revision": "exact-parent",
                         "report_sha256": "b" * 64, "qualification_sha256": "c" * 64}

    def proof_is_ancestor(self, proof, source):
        return True

    def guard(self, *args, **kwargs):
        self.calls.append("guard")
        return SimpleNamespace(guard=self.guard_value)

    def inspect(self, **kwargs):
        self.calls.append("inspect")
        return SimpleNamespace(identity=lambda: dict(self.identity),
                               pin=SimpleNamespace(revision={"encoder": Encoder().identity(),
                                   "prompt_template": {"id": "jrt-rwkv-g1-v1"}}))

    def config(self, spec):
        return self.config_value

    def encoder(self, *args):
        self.calls.append("encoder")
        return Encoder()

    def resources(self, resource):
        self.calls.append("resources")

    def load(self, *args, **kwargs):
        self.calls.append("load")
        return SimpleNamespace(identity=dict(self.identity), encoder=Encoder(), model=object(),
                               lora_modules=("block.adapter",), trainable_parameter_names=("head.weight",),
                               config=self.config_value)

    def trainer(self, parent, **kwargs):
        self.calls.append("trainer")
        assert len(kwargs["train_items"]) == 3600
        assert len(kwargs["diagnostic_items"]) == 1200
        assert kwargs["config"].hash() == parent.config.hash()
        return FakeTrainer(kwargs["config"])

    def peak(self):
        return 1

    def finite(self, parent):
        pass

    def diagnostics(self, model, items, batch_size):
        assert len(items) == 1200
        return {"accuracy": .5}


class FitCellTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()
        self.runtime = FakeRuntime()

    def execute(self, directory, payload=None, **kwargs):
        path = Path(directory) / "order.json"
        raw = canonical(order() if payload is None else payload)
        path.write_bytes(raw)
        return execute_fit(self.data, order_path=path, order_sha256=sha(raw),
                           source_commit="a" * 40, host="isolated-test-host", report=Path("report"),
                           qualification=Path("qualification"), checkpoint=Path("checkpoint"),
                           snapshot=Path("snapshot"), guard_path=Path("guard"), tokenizer=Path("tokenizer"),
                           out=Path(directory) / "run", runtime=self.runtime,
                           items_factory=kwargs.get("items_factory", items_factory))

    def test_exact_rows_world_split_and_sidecars(self):
        guard = Guard()
        tr, di, sidecar = prepare_items(self.data, Encoder(), guard, items_factory=items_factory)
        self.assertEqual((len(tr), len(di), len(sidecar), guard.checked), (3600, 1200, 4800, 4800))
        self.assertFalse({sidecar[i.id]["world"] for i in tr} & {sidecar[i.id]["world"] for i in di})
        self.assertEqual(len({sidecar[i.id]["world"] for i in tr}), 36)

    def test_census_mismatch_refuses_before_weights(self):
        self.data.census[SOURCES[1]][0]["prompt_tokens"] = 2049
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "census/Item"):
            self.execute(d)
        self.assertNotIn("load", self.runtime.calls)
        self.assertNotIn("resources", self.runtime.calls)

    def test_no_dropped_rows_and_invalid_world_partition(self):
        with self.assertRaisesRegex(FitCellError, "dropped"):
            prepare_items(self.data, Encoder(), Guard(),
                          items_factory=lambda *a, **k: items_factory(*a, **k)[:-1])
        self.data.plan["sources"][SOURCES[0]]["source"]["diagnostic_worlds"][0] = "phantom"
        with self.assertRaisesRegex(FitCellError, "partition"):
            prepare_items(self.data, Encoder(), Guard(), items_factory=items_factory)

    def test_invalid_order_and_guard_refuse_before_parent_metadata(self):
        payload = order()
        payload["resources"]["isolated"] = False
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "resource"):
            self.execute(d, payload)
        self.assertEqual(self.runtime.calls, [])
        self.data.records[SOURCES[0]][0]["forbidden"] = True
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "forbidden"):
            self.execute(d)
        self.assertEqual(self.runtime.calls, ["guard"])

    def test_wrong_issuer_and_unmerged_proof_refuse_before_runtime(self):
        payload = order()
        payload["ordered_by"] = "someone-else"
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "binding"):
            self.execute(d, payload)
        self.runtime.proof_is_ancestor = lambda proof, source: False
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "ancestor"):
            self.execute(d)
        self.assertEqual(self.runtime.calls, [])

    def test_incompatible_parent_config_refuses_before_weights(self):
        self.runtime.config_value.loss.kd = True
        with TemporaryDirectory() as d, self.assertRaisesRegex(FitCellError, "incompatible"):
            self.execute(d)
        self.assertNotIn("encoder", self.runtime.calls)
        self.assertNotIn("load", self.runtime.calls)

    def test_child_config_preserves_parent(self):
        original = Config()
        cfg = _child_config(original, order()["recipe"], Path("new"))
        self.assertEqual((cfg.row_batch, cfg.lr_head, cfg.hash()),
                         (original.row_batch, original.lr_head, original.hash()))
        self.assertEqual(original.max_steps, 50)
        self.assertEqual((cfg.max_steps, cfg.eval_every, cfg.require_guard), (2, 0, True))

    def test_fake_execution_receipts_child_parent_binding_without_torch(self):
        original_import = builtins.__import__
        def reject_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Torch import forbidden")
            return original_import(name, *args, **kwargs)
        with TemporaryDirectory() as d, patch("builtins.__import__", reject_torch):
            receipt = self.execute(d)
            saved = json.loads((Path(d) / "run" / "receipt.json").read_text())
            self.assertEqual(receipt, saved)
            self.assertEqual(receipt["stage1_parent"], self.runtime.identity)
            self.assertEqual(receipt["lora_modules"], ["block.adapter"])
            self.assertEqual(receipt["item_sidecar_sha256"],
                             sha((Path(d) / "run" / "item_sidecar.json").read_bytes()))
            child = receipt["checkpoints"][0]
            self.assertEqual(child["sha256"], sha(Path(child["path"]).read_bytes()))
            self.assertEqual(receipt["status"], "completed_fit_only")
            self.assertFalse(receipt["promotion_eligible"])
            self.assertFalse(receipt["language_qualified"])
            receipt_bytes = (Path(d) / "run" / "receipt.json").read_bytes()
            stage1_spec = SimpleNamespace(
                qualification={"checkpoint_config": receipt["config"],
                               "checkpoint_config_hash": receipt["config_hash"],
                               "checkpoint_lineage": receipt["lineage"],
                               "lora_modules": receipt["lora_modules"]},
                identity=lambda: dict(receipt["stage1_parent"]))
            candidate = validate_fit_child_receipt(receipt, receipt_sha256=sha(receipt_bytes),
                child_checkpoint=child["path"], stage1_spec=stage1_spec)
            state = {"step": child["step"], "epoch": 1, "cursor": 0,
                     "config": receipt["config"], "config_hash": receipt["config_hash"],
                     "trainable": {name: object() for name in receipt["trainable_parameter_names"]},
                     "optimizer": {}, "scheduler": {}, "rng": {}, "lineage": receipt["lineage"],
                     "init_seed": receipt["config"].get("seed"), "history": [],
                     "backbone": "rwkv7-13.3b", "lora_modules": receipt["lora_modules"], "guard": {}}
            validate_child_checkpoint_state(state, candidate, stage1_spec)
        self.assertLess(self.runtime.calls.index("encoder"), self.runtime.calls.index("load"))

    def test_default_and_missing_execution_gates_never_construct_runtime(self):
        args = []
        for name in ("base-fit", "base-census", "base-plan", "supplement", "supplement-census", "combined-plan"):
            args.extend(["--" + name, "synthetic-unused-path"])
        for execute in ([], ["--execute"]):
            output = io.StringIO()
            with patch("language_growth.l1_fit_cell.validate_fit", return_value=self.data), \
                 patch("language_growth.l1_fit_cell.Runtime", side_effect=AssertionError("runtime reached")), \
                 contextlib.redirect_stdout(output):
                code = main(args + execute)
            result = json.loads(output.getvalue())
            self.assertEqual(code, 2 if execute else 0)
            self.assertEqual(result["status"], "data_ready_execution_blocked")
            self.assertFalse(result["training_authorized"])

    def test_actual_pinned_fit_artifacts_validate_without_order_guard_or_torch(self):
        # Read-only integration proof when the frozen local fit bundle is installed.
        root = Path(__file__).resolve().parents[4] / "ssdm-data"
        combined = root / "agent-c-l1-combined-difference-preflight-20261008-v2"
        if not combined.is_file():
            self.skipTest("frozen local fit artifacts are not installed")
        inputs = FitInputs(root / "agent-c-l1-gate-training-fit-20261008-v3",
                           root / "agent-c-l1-fit-token-census-rwkv7-20261008",
                           root / "agent-c-l1-fit-cell-preflight-20261008-v6" / "plan.json",
                           root / "agent-c-l1-exactdifference-fit-supplement-20261008-v2",
                           root / "agent-c-l1-exactdifference-fit-supplement-20261008-v2", combined)
        original_import = builtins.__import__
        def reject_model_import(name, *args, **kwargs):
            if (name == "torch" or name.startswith("torch.")
                    or name in ("language_growth.native_parent", "language_growth.qualified_parent",
                                "language_growth.rwkv7_m2_loader", "train.trainer", "heads.model")):
                raise AssertionError(f"model import forbidden: {name}")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", reject_model_import):
            data = validate_fit(inputs)
            args = []
            for name in ("base-fit", "base-census", "base-plan", "supplement", "supplement-census", "combined-plan"):
                args.extend(["--" + name, str(getattr(inputs, name.replace("-", "_")))])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "data_ready_execution_blocked")
        self.assertEqual([len(data.records[s]) for s in SOURCES], [3072, 1728])
        self.assertEqual([len(data.census[s]) for s in SOURCES], [3072, 1728])
        self.assertEqual(data.plan["split"]["training_rows"], 3600)
        self.assertEqual(data.plan["split"]["diagnostic_rows"], 1200)
        self.assertFalse(data.plan["training_authorized"])


if __name__ == "__main__":
    unittest.main()
