"""Prepared-run correctness only: tiny random CPU RWKV, no real weight load."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from language_growth import rwkv_head_learning as probe


def test_default_dry_run_never_imports_torch_or_opens_checkpoint(tmp_path):
    code = """
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'safetensors', 'transformers'):
            raise AssertionError('weight/runtime import during dry run')
sys.meta_path.insert(0, Reject())
from language_growth.rwkv_head_learning import main
sys.argv = ['rwkv_head_learning', '--dry-run', '--out', sys.argv[1]]
assert main() == 0
"""
    out = tmp_path / "dry"
    subprocess.run([sys.executable, "-c", code, str(out)], cwd=probe.ROOT, check=True)
    registration = json.loads((out / "registration.json").read_text())
    assert registration["status"] == "prepared_not_run"
    assert set(p.name for p in out.iterdir()) == {"corpus.json", "registration.json"}
    with pytest.raises(FileExistsError):
        probe.preparation(out)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    reused = subprocess.run([sys.executable, "-m", "language_growth.rwkv_head_learning",
                             "--dry-run", "--out", str(out)], cwd=probe.ROOT)
    assert reused.returncode == 1
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_corpus_reproducible_stratified_and_disjoint():
    first = probe.corpus()
    assert probe.digest(first) == probe.digest(probe.corpus())
    names = []
    for split, examples in first.items():
        names.append({r["entity_ref"] for e in examples for r in e["training_only"]["catalog"]["records"]})
        assert {e["development_split"] for e in examples} == {split}
        assert all(e["development_only"] for e in examples)
        assert set(sum(e["training_only"]["kind"] == kind for e in examples)
                   for kind in ("read", "explain", "compare")) == {len(examples) // 3}
    assert not names[0] & names[1]


def test_current_training_label_is_separate_from_model_request():
    from language_growth.training_records import export_training_records
    example = probe.corpus()["development_train"][0]
    changed = deepcopy(example)
    question = next(iter(changed["model_inputs"][-1]["questions"].values()))
    target = changed["training_only"]["targets"][-1]
    target["option_id"] = next(k for k in question["criteria"] if k != target["option_id"])
    original_records = export_training_records([example])
    changed_records = export_training_records([changed])
    assert [r["request"] for r in original_records] == [r["request"] for r in changed_records]
    assert original_records[-1]["labels"] != changed_records[-1]["labels"]


def test_internal_child_cannot_bypass_launcher(monkeypatch, tmp_path):
    from language_growth.l0_qwen_cpu_probe import _offline_env
    env = _offline_env()
    # A same-user caller can spoof this old marker, but there is no --child CLI.
    env["SSDM_HEAD_LEARNING_PARENT_PID"] = str(os.getpid())
    output = tmp_path / "spoofed"
    result = subprocess.run([sys.executable, "-m", "language_growth.rwkv_head_learning",
                             "--child", "--out", str(output)], env=env, cwd=probe.ROOT)
    assert result.returncode == 2
    assert not output.exists()
    monkeypatch.setattr(probe.multiprocessing, "parent_process", lambda: None)
    with pytest.raises(RuntimeError, match="owned launcher"):
        probe.authorize_worker(None, os.getpid(), b"x" * 32)


def test_environment_refuses_gpu_online_or_thread_drift(monkeypatch):
    from language_growth.l0_qwen_cpu_probe import _offline_env
    for name, value in _offline_env().items():
        monkeypatch.setenv(name, value)
    probe.assert_environment()
    for name in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "OMP_NUM_THREADS"):
        old = os.environ[name]
        monkeypatch.setenv(name, "wrong")
        with pytest.raises(RuntimeError, match="offline CPU"):
            probe.assert_environment()
        monkeypatch.setenv(name, old)


def test_committed_preregistration_rejects_recipe_data_and_source_drift(monkeypatch, tmp_path):
    registration = probe.preparation(tmp_path / "prepared")
    path = tmp_path / "frozen.json"
    probe.write(path, registration)
    monkeypatch.setattr(probe, "FROZEN_REGISTRATION", path)
    probe.verify_frozen(registration)
    for key in ("config", "dependencies", "expected_parent", "corpus_sha256", "split_sha256", "source_sha256", "software"):
        changed = deepcopy(registration)
        changed[key] = "drift"
        with pytest.raises(ValueError, match="preregistration drift"):
            probe.verify_frozen(changed)
    for name in ("train/losses.py", "heads/layout.py", "heads/backbone.py", "heads/prefix_features.py",
                 "heads/decide_backend.py", "decide/adapter.py", "decide/prompt.py",
                 "language_growth/parent_bridge.py", "language_growth/paired_comparison.py"):
        assert name in registration["source_sha256"]


class Child:
    def __init__(self, terminal=None):
        self.terminal, self.killed, self.waited = terminal, 0, 0
    def poll(self):
        return self.terminal
    def kill(self):
        self.killed += 1
        self.terminal = -9
    def wait(self):
        self.waited += 1
        return self.terminal


@pytest.mark.parametrize("reason,available,clock,code", [
    ("memory_guard", 1, lambda: 0, 125),
    ("timeout", 10 * 1024**3, iter([0, 901]).__next__, 124),
])
def test_stop_paths_kill_only_owned_child(tmp_path, reason, available, clock, code):
    child, unrelated = Child(), Child()
    raw = tmp_path / "raw.jsonl"
    assert probe.supervise(child, raw, available=lambda: available, now=clock) == code
    assert (child.killed, child.waited) == (1, 1)
    child = Child()
    def fail_initial():
        raise RuntimeError("first RAM observation failed")
    with pytest.raises(RuntimeError):
        probe.supervise(child, tmp_path / "raw", available=fail_initial)
    assert (child.killed, child.waited) == (1, 1)
    assert unrelated.killed == 0
    assert json.loads(raw.read_text())["reason"] == reason


def test_monitor_preserves_terminal_status_and_cleans_up_on_error(tmp_path):
    child = Child(7)
    assert probe.supervise(child, tmp_path / "raw", available=lambda: 10**12) == 7
    assert child.killed == 0
    child = Child()
    def fail(_):
        raise RuntimeError("monitor failed")
    with pytest.raises(RuntimeError):
        probe.supervise(child, tmp_path / "raw", available=lambda: 10**12, sleep=fail)
    assert (child.killed, child.waited) == (1, 1)


def test_low_start_ram_never_spawns_or_loads(monkeypatch, tmp_path):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("RAM", (), {"available": 1})())
    monkeypatch.setattr(probe.multiprocessing, "get_context", lambda *a, **kw: pytest.fail("spawned under RAM floor"))
    assert probe.launch(tmp_path / "refused") == 125
    assert json.loads((tmp_path / "refused/raw.jsonl").read_text())["event"] == "launcher_refused"


def test_bootstrap_pairing_and_failures_remain_denominator():
    result = probe.bootstrap([False, False, True], [True, False, True])
    assert result["n_worlds"] == 3
    assert result["estimate"] == 1 / 3
    assert result == probe.bootstrap([False, False, True], [True, False, True])
    with pytest.raises(ValueError):
        probe.bootstrap([], [])


def test_capture_logs_provider_exception_with_phase_and_request_id(tmp_path):
    class Fails:
        def decide(self, request):
            raise RuntimeError("provider exploded")
    raw = tmp_path / "raw.jsonl"
    request = {"ssdm": {"request_id": "request-123"}, "questions": {"slot-001-operation": {}}}
    with pytest.raises(RuntimeError, match="exploded"):
        probe.Capture(Fails(), raw, "before").decide(request)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert rows[-1]["event"] == "provider_exception"
    assert rows[-1]["phase"] == "before"
    assert rows[-1]["request_id"] == "request-123"


def test_full_parent_digest_covers_buffers_and_independent_lm_head():
    import torch
    from torch import nn
    parent = nn.Module()
    parent.backbone = nn.Linear(2, 2)
    parent.head = nn.Linear(2, 3)
    parent.register_buffer("state", torch.zeros(2))
    original = probe.state_digest(parent)
    with torch.no_grad():
        parent.state[0] = 1
    assert probe.state_digest(parent) != original
    with torch.no_grad():
        parent.state.zero_()
        parent.head.weight[0, 0] += 1
    assert probe.state_digest(parent) != original


def _authenticated_test_worker(connection, pid, secret, output):
    probe.authorize_worker(connection, pid, secret)
    Path(output).write_text("authenticated", encoding="utf-8")


def test_internal_pipe_handshake_works_only_for_matching_secret(tmp_path):
    import secrets
    for matches in (True, False):
        ctx = probe.multiprocessing.get_context("spawn")
        receive, send = ctx.Pipe(duplex=False)
        secret = secrets.token_bytes(32)
        output = tmp_path / str(matches)
        process = ctx.Process(target=_authenticated_test_worker,
                              args=(receive, os.getpid(), secret, str(output)))
        process.start()
        receive.close()
        try:
            send.send_bytes(secret if matches else b"x" * 32)
            send.close()
            process.join(10)
            assert not process.is_alive()
            assert (process.exitcode == 0) == matches
            assert output.exists() == matches
        finally:
            send.close()
            if process.is_alive():
                process.kill()
                process.join()


def tiny_backend():
    from heads.backbone import tiny_hf_backbone
    from language_growth.rwkv_decision_head import rwkv_decision_backend
    class Encoder:
        def __call__(self, text, trusted=True):
            return [1 + sum(word.encode()) % 255 for word in text.split()] or [1]
        def identity(self):
            return {"model_id": "tiny-rwkv-test", "snapshot": "seed-20261006", "kind": "test-word-hash"}
    parent = tiny_hf_backbone("rwkv7", seed=probe.CONFIG["seed"], hidden=16).model
    return rwkv_decision_backend(parent, Encoder(), seed=probe.CONFIG["seed"],
        head_kwargs=probe.CONFIG["head_kwargs"], source_identity={"model_id": "tiny-rwkv-test", "snapshot": "seed-20261006"})


def test_tiny_rwkv_fit_freezes_backbone_and_strict_reload(tmp_path):
    import torch
    torch.set_num_threads(2)
    examples = probe.corpus()["development_train"][:3]
    backend = tiny_backend()
    trainer = probe.trainer_for(backend, examples, steps=1)
    frozen = {n: p.clone() for n, p in backend.model.backbone.named_parameters()}
    before = trainer.trainable_state()
    assert torch.isfinite(torch.tensor(trainer.train_step()))
    assert any(not torch.equal(v, trainer.trainable_state()[n]) for n, v in before.items())
    assert all(torch.equal(v, dict(backend.model.backbone.named_parameters())[n]) for n, v in frozen.items())
    checkpoint = trainer.save(tmp_path / "checkpoint")
    restored = probe.trainer_for(tiny_backend(), examples, steps=1)
    probe.strict_restore(restored, checkpoint)
    assert all(torch.equal(v, restored.trainable_state()[n]) for n, v in trainer.trainable_state().items())
    for mutate in ("keys", "config", "lineage"):
        state = torch.load(checkpoint, weights_only=False)
        if mutate == "keys":
            state["trainable"].pop(next(iter(state["trainable"])))
        elif mutate == "config":
            state["config_hash"] = "wrong"
        else:
            state["lineage"] = {}
        path = tmp_path / (mutate + ".pt")
        torch.save(state, path)
        with pytest.raises(ValueError):
            probe.strict_restore(restored, path)


def test_evaluation_never_supplies_labels_or_gold_prefixes(tmp_path):
    from decide.adapter import Decider
    backend = tiny_backend()
    example = probe.corpus()["development_validation"][0]
    changed = deepcopy(example)
    changed["training_only"]["targets"][0]["option_id"] = "poisoned-evaluator-only"
    # Replace scoring function, then corrupt every evaluator-only target/program.
    import language_growth.free_running_dev as evaluator
    original = evaluator.score_development_task
    evaluator.score_development_task = lambda *args: False
    try:
        changed["training_only"]["program"] = {"poison": "must never reach provider"}
        calls = []
        class Provider:
            calibration = Decider(backend).calibration
            def decide(self, request):
                calls.append(deepcopy(request))
                return Decider(backend).decide(request)
        # Keep the post-call reference attribution valid while poisoning gold program.
        changed["training_only"]["targets"] = deepcopy(example["training_only"]["targets"])
        report = probe.evaluate(backend, Provider(), [example], tmp_path / "a.jsonl", "a")
        assert report[0]["slot_attribution"]
        assert all(row["selected"] is not None and row["selected_description"] is not None
                   for row in report[0]["slot_attribution"])
        first = calls[:]
        calls.clear()
        probe.evaluate(backend, Provider(), [changed], tmp_path / "b.jsonl", "b")
        def semantic(request):
            return request["state"], request["questions"]
        assert [semantic(r) for r in calls] == [semantic(r) for r in first]
        assert calls
        state = json.loads(calls[0]["state"])
        assert state["proposed_slot_choices"] == []
        assert "poison" not in json.dumps(calls)
    finally:
        evaluator.score_development_task = original
