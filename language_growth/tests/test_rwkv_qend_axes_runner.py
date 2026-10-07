"""Frozen paired recipes, default no-load preparation and checked outputs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from language_growth import rwkv_qend_axes as runner


def test_default_dry_runs_equal_both_materialized_committed_pins(tmp_path):
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','transformers','safetensors'):
            raise AssertionError('runtime imported during default preparation')
sys.meta_path.insert(0,Reject())
def access(event,args):
    if event=='open':
        path=str(args[0]).lower().replace('\\','/')
        if path.endswith(('.pt','.safetensors')) or '/snapshots/' in path:
            raise AssertionError('checkpoint accessed during preparation')
sys.addaudithook(access)
from language_growth.rwkv_qend_axes import main
sys.argv=['runner','--arm',sys.argv[1],'--out',sys.argv[2]]
assert main()==0
assert 'torch' not in sys.modules
'''
    pins = []
    for arm in runner.ARMS:
        pin = runner.PREPARED / f"registration-{arm}.json"
        assert pin.is_file(), "both immutable pins must be materialized before review/run"
        out = tmp_path / arm
        subprocess.run([sys.executable, "-c", code, arm, str(out)], cwd=runner.ROOT, check=True)
        registration = json.loads((out / "registration.json").read_text())
        assert registration == json.loads(pin.read_text())
        pins.append(registration)
    assert pins[0]["config"] == pins[1]["config"]
    assert pins[0]["axis_targets_sha256"] == pins[1]["axis_targets_sha256"]
    assert pins[0]["source_sha256"] == pins[1]["source_sha256"]
    assert [p["axis_loss_weight"] for p in pins] == [0, 1]


def test_pin_drift_and_12gib_start_guard_fail_closed(monkeypatch, tmp_path):
    import psutil
    pin = json.loads((runner.PREPARED / "registration-control.json").read_text())
    runner.verify_frozen(pin)
    for key in ("axis_loss_weight", "config", "source_sha256", "axis_targets_sha256", "decision_rule"):
        changed = deepcopy(pin)
        changed[key] = "drift"
        with pytest.raises(ValueError, match="preregistration drift"):
            runner.verify_frozen(changed)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("RAM", (), {"available": 11 * 1024**3})())
    monkeypatch.setattr(runner.multiprocessing, "get_context", lambda *args: pytest.fail("spawned below12 GiB"))
    assert runner.launch(tmp_path / "refused", "control") == 125
    event = json.loads((tmp_path / "refused/raw.jsonl").read_text())
    assert event["required_bytes"] == 12 * 1024**3
    assert event["available_bytes"] == 11 * 1024**3
    assert runner.CONFIG["min_running_bytes"] == runner.guard.CONFIG["min_running_bytes"] == 3 * 1024**3
    assert runner.CONFIG["timeout_seconds"] == runner.guard.CONFIG["timeout_seconds"] == 1200


def backend():
    from heads.backbone import tiny_hf_backbone
    from language_growth.rwkv_decision_head import rwkv_decision_backend
    class Encoder:
        def __call__(self, text, trusted=True):
            return [1+sum(word.encode()) % 255 for word in text.split()] or [1]
        def identity(self):
            return {"model_id": "tiny-qend-test", "snapshot": "random71"}
    parent = tiny_hf_backbone("rwkv7", seed=71, hidden=16).model
    return rwkv_decision_backend(parent, Encoder(), source_identity={"model_id": "tiny-qend-test", "snapshot": "random71"},
                                 head="qend_axes", seed=runner.CONFIG["seed"])


def test_matched_initialization_and_exact_one_pass_without_real_fit():
    torch.set_num_threads(2)
    data = runner.corpus()
    control = runner.make_trainer(backend(), data, "control")
    axes = runner.make_trainer(backend(), data, "axes")
    assert runner.head_digest(control.model) == runner.head_digest(axes.model)
    a = [[item.id for item in control._next_batch()] for _ in range(32)]
    b = [[item.id for item in axes._next_batch()] for _ in range(32)]
    assert a == b
    assert len({identity for batch in a for identity in batch}) == 64
    assert (control.epoch, control.cursor, control.step) == (0, 64, 0)
    assert control.cfg.axis_loss_weight == 0 and axes.cfg.axis_loss_weight == 1
    assert all("axis_target" not in item.row.meta for item in control.train_items)


def test_clamp_random_norm_and_output_only_equivalence():
    torch.manual_seed(3)
    port = {"z": torch.randn(1,6).tanh(), "basis": torch.randn(1,4,6), "bias": torch.randn(1,4)}
    before = port["z"].clone()
    sweep = runner.port_sweep(port, 9321)
    assert len(sweep) == 18
    for condition in sweep:
        assert condition["clamp_norm"] == pytest.approx(condition["random_norm"], abs=1e-6)
        assert condition["affine_replay_max_abs_error"] < 1e-5
        assert condition["clamp_index"] == condition["output_only_index"]
        assert condition["output_only_axes_unchanged"] is True
    torch.testing.assert_close(port["z"], before, atol=0, rtol=0)


def test_no_unchecked_output_even_when_a_typed_choice_is_false(tmp_path):
    from decide.adapter import Decider
    from decide.backends.oracle import OracleBackend
    example = next(e for e in runner.corpus()["development_validation"] if e["training_only"]["kind"] == "compare")
    labels = {next(iter(request["questions"])): target["option_id"]
              for request, target in zip(example["model_inputs"], example["training_only"]["targets"])}
    relation_request = example["model_inputs"][-1]
    qid, question = next(iter(relation_request["questions"].items()))
    labels[qid] = next(key for key in question["criteria"] if key != labels[qid])
    oracle = OracleBackend(labels, margin=20)
    report = runner.checked_answers(oracle, Decider(oracle), [example], tmp_path)
    assert report[0]["status"] == "kernel_rejected"
    assert report[0]["text"] is None and report[0]["task_correct"] is False


def test_paired_screening_requires_fit_gain_and_no_read_regression(tmp_path):
    examples = runner.corpus()["development_validation"]
    common = {"status": "complete", "config": runner.CONFIG, "source_sha256": {}, "corpus_sha256": "same",
              "axis_targets_sha256": "same", "initial_head_sha256": "same", "full_parent_state_sha256": "same",
              "expected_training_order_sha256": "same"}
    rows = [{"task_id": e["task_id"], "task_correct": False, "kernel_accepted": True} for e in examples]
    control = {**common, "arm": "control", "axis_loss_weight": 0, "after": deepcopy(rows)}
    axes = {**common, "arm": "axes", "axis_loss_weight": 1, "after": deepcopy(rows), "fit": {"summary": [
        {"split": "development_train", "kind": "all", "slot": "all", "qtype": "all", "n_failed": 0, "accuracy_all": 52/64}]}}
    control_dir, axes_dir = tmp_path / "control", tmp_path / "axes"
    control_dir.mkdir(); axes_dir.mkdir()
    def report(name):
        (control_dir / "receipt.json").write_text(json.dumps(control))
        (axes_dir / "receipt.json").write_text(json.dumps(axes))
        return runner.compare_arms(control_dir, axes_dir, tmp_path / name)
    axes["after"][0]["task_correct"] = True
    assert report("good.json")["eligible_for_review"] is True
    axes["fit"]["summary"][0]["accuracy_all"] = 51/64
    assert report("notfit.json")["eligible_for_review"] is False
    axes["fit"]["summary"][0]["accuracy_all"] = 52/64
    control["after"][0]["task_correct"] = True
    axes["after"][0]["task_correct"] = False
    axes["after"][1]["task_correct"] = axes["after"][2]["task_correct"] = True
    assert report("read-regression.json")["eligible_for_review"] is False
    axes["initial_head_sha256"] = "different"
    with pytest.raises(ValueError, match="unmatched"):
        report("bad.json")
