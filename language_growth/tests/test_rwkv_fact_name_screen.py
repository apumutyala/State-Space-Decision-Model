"""Synthetic CPU/preflight proofs only; never load a checkpoint or real parent."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from language_growth import rwkv_fact_name_screen as runner
from language_growth.tests.test_planner import EXPECTED_CONFIG, REVISION, ScriptedProvider


def scripted_screen(examples, *, wrong_first_entity_fact=False):
    selectors = []
    groups = {}
    for e in examples:
        groups.setdefault(e["pair_id"], []).append(e)
    for world, (_, members) in enumerate(sorted(groups.items())):
        styles = runner.CONFIG["styles"] if world % 2 == 0 else list(reversed(runner.CONFIG["styles"]))
        for style in styles:
            for e in members:
                tr = e["training_only"]
                selectors.append("Assert one selected fact" if tr["kind"] == "read" else
                                 "Explain a checked assertion of one selected fact")
                ref = tr["target_record_ref"]
                if wrong_first_entity_fact and world == 0 and style == "entity_name" and e["pair_member"] == 0:
                    ref = "binding-screen-record-2"
                field = tr["field"]
                record = next(r for r in tr["catalog"]["records"] if r["record_ref"] == ref)
                selectors.append(f"Fact {ref} field {field}" if style == "record_ref" else
                                 f"Fact {record['entity_ref']} (record {ref}) field {field}")
    return ScriptedProvider(selectors)


def evaluate(tmp_path, *, wrong=False):
    examples = runner.screen_data()
    provider = scripted_screen(examples, wrong_first_entity_fact=wrong)
    rows, order, equal = runner.evaluate_screen(examples, provider, model="scripted", revision=REVISION,
        served_config=EXPECTED_CONFIG, raw=tmp_path / "raw.jsonl")
    return examples, provider, rows, order, equal


def test_full_screen_alternates_styles_retains_all_members_and_separates_gold(tmp_path):
    examples, provider, rows, order, equal = evaluate(tmp_path)
    assert len(order) == 64 and len(provider.requests) == 128
    assert equal
    assert [r["style"] for r in order[:8]] == ["record_ref"] * 2 + ["entity_name"] * 4 + ["record_ref"] * 2
    assert {s: len(v) for s, v in rows.items()} == {"record_ref": 32, "entity_name": 32}
    assert all(r["task_correct"] and r["kernel_accepted"] and r["fact_correct"] for v in rows.values() for r in v)
    assert all("training_only" not in json.dumps(r) and "target_record_ref" not in json.dumps(r) for r in provider.requests)
    summary = runner.paired_summary(rows, examples)
    assert summary["all"]["task_correct"]["n_worlds"] == 16
    assert summary["read"]["task_correct"]["n_worlds"] == 8
    assert summary["explain"]["task_correct"]["n_members"] == 16
    assert summary["all"]["task_correct"]["estimate"] == 0
    inputs, targets = runner.separated_data(examples)
    assert len(inputs) == len(targets) == 32
    assert "training_only" not in json.dumps(inputs) and "target_record_ref" not in json.dumps(inputs)
    assert all("program" in t and "target_record_ref" in t for t in targets)


def test_valid_wrong_fact_remains_failure_and_fractional_world_delta_is_preserved(tmp_path):
    examples, _, rows, _, equal = evaluate(tmp_path, wrong=True)
    assert equal
    row = rows["entity_name"][0]
    assert row["kernel_accepted"] and not row["task_correct"]
    assert row["operation_correct"] and row["field_correct"] and not row["record_correct"]
    summary = runner.paired_summary(rows, examples)
    assert summary["all"]["task_correct"]["estimate"] == -1 / 32
    assert summary["all"]["both_members_task_correct"]["estimate"] == -1 / 16
    assert summary["read"]["task_correct"]["estimate"] == -1 / 16
    with pytest.raises(ValueError, match="complete"):
        runner.paired_summary({**rows, "entity_name": rows["entity_name"][:-1]}, examples)
    with pytest.raises(ValueError, match="complete"):
        runner.evaluate_screen(examples[:-1], None, model="scripted", revision=REVISION,
            served_config=EXPECTED_CONFIG, raw=tmp_path / "missing.jsonl")


def test_provider_failure_is_logged_and_kept_in_denominator(tmp_path):
    class Failing:
        def decide(self, request): raise RuntimeError("scripted provider error")
    examples = runner.screen_data()
    rows, order, equal = runner.evaluate_screen(examples, Failing(), model="scripted", revision=REVISION,
        served_config=EXPECTED_CONFIG, raw=tmp_path / "raw.jsonl")
    assert len(order) == 64 and all(not r["task_correct"] for v in rows.values() for r in v)
    assert not equal
    assert "provider_exception" in (tmp_path / "raw.jsonl").read_text()
    assert runner.paired_summary(rows, examples)["all"]["kernel_accepted"]["baseline"] == 0


def test_paired_operation_decision_fields_exclude_latency_but_detect_drift(tmp_path):
    _, provider, _, _, equal = evaluate(tmp_path)
    assert equal and len(provider.requests) == 128  # no 64-request direct bookends
    changed = deepcopy(provider.requests[0])
    response = ScriptedProvider(["Assert one selected fact"]).decide(changed)
    projected = deepcopy(runner.decision_fields(response))
    response["ssdm"]["diagnostics"]["latency_ms"] = 1000
    assert runner.decision_fields(response) == projected
    response["answers"]["slot-001-operation"]["choice"] = "op002"
    assert runner.decision_fields(response) != projected


@pytest.mark.parametrize("drift", ["answer", "probability", "calibration", "status", "request"])
def test_paired_operation_equality_covers_every_member_and_exact_decisions(monkeypatch, tmp_path, drift):
    from language_growth import free_running_dev
    original = free_running_dev.run_development_tasks
    calls = 0
    def run(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 64:  # last member: completeness must include more than the first pair
            first = args[1].calls[0]
            if drift == "request":
                first["request"]["state"] += " "
            elif drift == "answer":
                first["response"]["answers"]["slot-001-operation"]["choice"] = "op001"
            else:
                key = {"probability": "raw_probabilities", "calibration": "served_config", "status": "status"}[drift]
                first["response"]["ssdm"][key] = "changed"
        return result
    monkeypatch.setattr(free_running_dev, "run_development_tasks", run)
    assert not evaluate(tmp_path)[-1]


def test_preflight_and_default_do_not_import_runtime_or_access_snapshot(tmp_path):
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported in no-load preflight')
sys.meta_path.insert(0, Reject())
def audit(event, args):
    if event == 'open':
        path = str(args[0]).lower().replace('\\', '/')
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path:
            raise AssertionError('snapshot/checkpoint accessed in preflight')
sys.addaudithook(audit)
from language_growth.rwkv_fact_name_screen import main
sys.argv = ['runner', '--out', sys.argv[1]] + sys.argv[2:]
assert main() == 0
assert 'torch' not in sys.modules
'''
    for mode in ([], ["--dry-run"]):
        out = tmp_path / ("dry" if mode else "default")
        subprocess.run([sys.executable, "-c", code, str(out), *mode], cwd=runner.ROOT, check=True)
        assert json.loads((out / "registration.json").read_text())["status"] == "prepared_not_run"
        assert len(json.loads((out / "screen-inputs.json").read_text())) == 32


def test_identical_operation_errors_in_both_arms_do_not_pass_equality_or_gate(monkeypatch, tmp_path):
    from language_growth import free_running_dev
    original = free_running_dev.run_development_tasks
    def errors(*args, **kwargs):
        result = original(*args, **kwargs)
        # Both arms retain identical recorded initial responses, but not `ok`.
        args[1].calls[0]["response"]["ssdm"]["status"] = "error"
        return result
    monkeypatch.setattr(free_running_dev, "run_development_tasks", errors)
    examples, _, rows, _, equal = evaluate(tmp_path)
    assert not equal
    gate = runner.continuation_gate(rows, examples, runner.paired_summary(rows, examples),
                                    frozen_parent=True, operation_equal=equal)
    assert not gate["gates"]["arm_operation_equal"] and not gate["eligible_for_review"]


def test_resource_and_worker_guards_refuse_before_load(monkeypatch, tmp_path):
    import psutil
    monkeypatch.setattr(runner, "preparation", lambda out: out.mkdir())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=9 * 1024**3 - 1, total=16 * 1024**3))
    assert runner.launch(tmp_path / "refused") == 125
    refused = json.loads((tmp_path / "refused" / "receipt.json").read_text())
    assert refused["status"] == "incomplete" and refused["stop_reason"] == "start_memory_guard"
    assert "eligible_for_review" not in refused and "summary" not in refused
    monkeypatch.setattr(runner, "assert_environment", lambda: None)
    with pytest.raises(MemoryError, match="start reserve"):
        runner.run(tmp_path)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1200.)
    with pytest.raises(TimeoutError): runner.budget_check(0., 20 * 1024**3)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.)
    with pytest.raises(MemoryError): runner.budget_check(0., 3 * 1024**3 - 1)
    with pytest.raises(Exception): runner.worker(tmp_path, None, -1, b"invalid")


def test_9gib_admission_refuses_before_runtime_import_or_checkpoint_access(tmp_path):
    code = r'''
import sys
from types import SimpleNamespace
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported before admission refusal')
sys.meta_path.insert(0, Reject())
def audit(event, args):
    if event == 'open':
        path = str(args[0]).lower().replace('\\', '/')
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path:
            raise AssertionError('snapshot/checkpoint opened before admission refusal')
sys.addaudithook(audit)
import psutil
psutil.virtual_memory = lambda: SimpleNamespace(available=9 * 1024**3 - 1, total=16 * 1024**3)
from pathlib import Path
from language_growth import rwkv_fact_name_screen as runner
assert runner.CONFIG['min_start_bytes'] == 9 * 1024**3
assert runner.launch(Path(sys.argv[1])) == 125
runner.assert_environment = lambda: None
try:
    runner.run(Path(sys.argv[1]))
except MemoryError:
    pass
else:
    raise AssertionError('worker admission did not refuse')
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "refused")], cwd=runner.ROOT, check=True)


@pytest.mark.parametrize("stage", ["spawn", "partial_spawn", "handshake", "supervisor"])
def test_launcher_exceptions_preserve_incomplete_receipt_and_owned_cleanup_without_runtime(tmp_path, stage):
    code = r'''
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported by failing launcher')
sys.meta_path.insert(0, Reject())
def audit(event, args):
    if event == 'open':
        path = str(args[0]).lower().replace('\\', '/')
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path:
            raise AssertionError('checkpoint accessed by failing launcher')
sys.addaudithook(audit)
import psutil
psutil.virtual_memory = lambda: SimpleNamespace(available=12 * 1024**3, total=16 * 1024**3)
from language_growth import rwkv_fact_name_screen as runner
stage = sys.argv[2]
class Pipe:
    closes = 0
    def close(self): self.closes += 1
    def send_bytes(self, secret):
        if stage == 'handshake': raise BrokenPipeError('simulated handshake failure')
class Child:
    pid = exitcode = None
    kills = joins = 0
    def start(self):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        if stage == 'spawn': raise OSError('simulated spawn failure')
        self.pid = 12345
        if stage == 'partial_spawn': raise OSError('simulated partial spawn failure')
    def kill(self):
        assert self.pid is not None
        self.kills += 1
        self.exitcode = -9
    def join(self):
        assert self.pid is not None
        self.joins += 1
receive, send, child = Pipe(), Pipe(), Child()
class Context:
    def Pipe(self, **kwargs): return receive, send
    def Process(self, **kwargs): return child
runner.multiprocessing.get_context = lambda mode: Context()
def supervise(*args, **kwargs):
    assert stage == 'supervisor'
    raise RuntimeError('simulated supervisor failure')
runner.supervise = supervise
old = dict(os.environ)
out = Path(sys.argv[1])
assert runner.launch(out) == 1
assert dict(os.environ) == old
assert receive.closes and send.closes
assert child.kills == child.joins == (0 if stage == 'spawn' else 1)
receipt = json.loads((out / 'receipt.json').read_text())
assert receipt['status'] == 'incomplete'
assert receipt['stop_reason'] == ('spawn_error' if stage == 'partial_spawn' else stage + '_error')
assert receipt['exception_type'] == {'spawn':'OSError', 'partial_spawn':'OSError', 'handshake':'BrokenPipeError', 'supervisor':'RuntimeError'}[stage]
assert 'simulated' in receipt['exception_message']
assert not any(k in receipt for k in ('summary', 'metrics', 'eligible', 'eligible_for_review'))
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path / stage), stage], cwd=runner.ROOT, check=True)


@pytest.mark.parametrize("reason,code", [("memory_guard", 125), ("timeout", 124), ("worker_failed", 1)])
def test_terminal_abort_records_incomplete_without_success_fields(tmp_path, reason, code):
    event = {"event": "launcher_stopped" if code != 1 else "launcher_exit",
             "elapsed_seconds": 74.063, "minimum_available_bytes": 3202723840}
    if code != 1:
        event["reason"] = reason
    (tmp_path / "launcher.jsonl").write_text(json.dumps(event) + "\n")
    (tmp_path / "raw.jsonl").write_text('{"event":"request"}\n')
    # Even a stale complete receipt cannot survive a nonzero terminal exit.
    runner.write(tmp_path / "receipt.json", {"status": "complete", "eligible_for_review": True})
    assert runner.finalize_launch(tmp_path, code) == code
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["stop_reason"] == reason
    assert receipt["cost_usd"] == 0 and receipt["exit_code"] == code
    assert receipt["artifact_sha256"]["raw.jsonl"] == runner.sha(tmp_path / "raw.jsonl")
    assert not any(k in receipt for k in ("summary", "metrics", "eligible", "eligible_for_review"))


def test_supervisor_kills_only_owned_child_on_timeout_or_low_memory(tmp_path):
    class Child:
        def __init__(self): self.kills = self.waits = 0
        def poll(self): return -9 if self.kills else None
        def kill(self): self.kills += 1
        def wait(self): self.waits += 1
    for ram, times, code in ((6 * 1024**3, [0., 1200.], 124), (3 * 1024**3 - 1, [0., 0.], 125)):
        child, clock = Child(), iter(times)
        assert runner.supervise(child, tmp_path / f"{code}.jsonl", available=lambda: ram,
            now=lambda: next(clock), sleep=lambda _: pytest.fail("guard must stop")) == code
        assert child.kills == child.waits == 1


def test_full_state_digest_covers_lm_head_parameters_and_buffers():
    import torch
    module = torch.nn.Module()
    module.head = torch.nn.Linear(2, 2)
    module.register_buffer("buffer", torch.zeros(2))
    original = runner.state_digest(module)
    with torch.no_grad(): module.head.weight[0, 0] += 1
    head_changed = runner.state_digest(module)
    assert original != head_changed
    module.buffer[0] = 1
    assert runner.state_digest(module) != head_changed


def test_continuation_gate_requires_world_gains_without_losses_fact_gain_operation_and_positive_ci():
    examples = runner.screen_data()
    rows = {s: [{"task_id": e["task_id"], "pair_id": e["pair_id"], "pair_member": e["pair_member"],
        "task_correct": False, "kernel_accepted": True, "operation_correct": True,
        "fact_correct": False, "record_correct": False, "field_correct": True} for e in examples]
        for s in runner.CONFIG["styles"]}
    for row in rows["entity_name"][:8]:
        row.update(task_correct=True, fact_correct=True, record_correct=True)
    def gate(value):
        return runner.continuation_gate(value, examples, runner.paired_summary(value, examples),
            frozen_parent=True, operation_equal=True)
    passed = gate(rows)
    assert passed["eligible_for_review"] and passed["both_member_world_gains"] == 4
    short = deepcopy(rows)
    for row in short["entity_name"][4:8]: row["task_correct"] = False
    inconclusive = gate(short)
    assert inconclusive["both_member_world_gains"] == 2
    assert not inconclusive["gates"]["useful_answer_ci_lower_positive"]
    assert not inconclusive["eligible_for_review"]
    lost = deepcopy(rows)
    for row in lost["record_ref"][-2:]: row["task_correct"] = True
    assert gate(lost)["both_member_world_losses"] == 1
    assert not gate(lost)["eligible_for_review"]
    operation = deepcopy(rows)
    operation["entity_name"][0]["operation_correct"] = False
    assert not gate(operation)["gates"]["no_operation_regression"]
    fact = deepcopy(rows)
    for row in fact["entity_name"]: row["fact_correct"] = False
    assert not gate(fact)["gates"]["exact_fact_binding_improved"]
    summary = runner.paired_summary(rows, examples)
    assert not runner.continuation_gate(rows, examples, summary,
        frozen_parent=True, operation_equal=False)["eligible_for_review"]
