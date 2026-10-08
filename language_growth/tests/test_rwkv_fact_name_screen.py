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


class FiniteStockBackend:
    """Protocol-only fake: returns fixed scores without tensors or model inference."""
    name = "finite-stock-test"

    def __init__(self):
        self.calls = 0
        self.requests = []
        self.failure = None
        self.identity = {"revision_id": "finite-stock-test:v1", "model_id": "scripted"}

    def revision(self): return deepcopy(self.identity)

    def plan(self, request):
        from decide.backends.base import Plan
        return Plan(inference_calls=2, max_call_input_tokens=8, total_input_tokens=30)

    def score(self, request, deadline):
        from decide.backends.base import ScoreResult
        self.calls += 1
        self.requests.append(deepcopy(request))
        deadline()
        if self.failure == "exception": raise RuntimeError("scripted score failure")
        deadline()
        logits = {qid: {oid: -float(i) for i, oid in enumerate(sorted(q["criteria"]))}
                  for qid, q in request["questions"].items()}
        if self.failure == "nonfinite":
            next(iter(logits.values()))[next(iter(next(iter(logits.values()))))] = float("nan")
        return ScoreResult(logits, 2, 30, extras={"call_latency_ms": [1., 2.]})


def memo_provider():
    from decide.adapter import Decider
    stock = FiniteStockBackend()
    memo = runner.InitialOperationScoreMemo(stock)
    decider = Decider(memo)
    return stock, memo, runner.SuccessfulMemoProvider(decider, memo), decider


def initial_request(provider):
    class Capture:
        def decide(self, request):
            self.request = request
            raise RuntimeError("capture only")
    from language_growth.planner import plan_checked_answer
    example = runner.screen_data()[0]
    capture = Capture()
    plan_checked_answer(json.loads(example["model_inputs"][0]["state"])["task_input"],
        example["training_only"]["host"], example["training_only"]["catalog"], capture,
        model="scripted", parent_revision=provider.revision()["revision_id"],
        expected_served_config=EXPECTED_CONFIG)
    return capture.request


def test_exact_initial_operation_hit_reuses_scores_through_adapter_with_fresh_identity_and_zero_work():
    stock, memo, provider, _ = memo_provider()
    request = initial_request(provider)
    before = provider.decide(request)
    changed_id = deepcopy(request)
    changed_id["ssdm"]["request_id"] = "second-logical-call"
    after = provider.decide(changed_id)
    assert before["ssdm"]["status"] == after["ssdm"]["status"] == "ok"
    assert runner.decision_fields(before) == runner.decision_fields(after)
    assert after["ssdm"]["request_id"] == "second-logical-call"
    assert stock.calls == 1 and memo.hits == 1 and len(memo.entries) == 1
    assert before["ssdm"]["diagnostics"]["inference_calls"] == 2
    assert after["ssdm"]["diagnostics"]["inference_calls"] == 0
    assert after["usage"]["input_tokens"] == 0
    assert after["ssdm"]["diagnostics"]["input_counts"]["max_call_input_tokens"] == 8
    replay = after["ssdm"]["diagnostics"]["backend"]
    assert replay["call_latency_ms"] == []
    assert replay["operation_score_memo"]["source_request_id"] == request["ssdm"]["request_id"]
    # Neither the caller nor a returned result can corrupt the saved exact scores.
    before["ssdm"]["raw_probabilities"].clear()
    after["ssdm"]["raw_probabilities"].clear()
    assert provider.decide(changed_id)["ssdm"]["raw_probabilities"]


def test_every_non_request_id_field_changes_exact_cache_identity_and_key_order_does_not():
    stock, memo, provider, _ = memo_provider()
    request = initial_request(provider)
    key = memo.key(request)
    assert memo.key(dict(reversed(list(request.items())))) == key
    def leaves(value, path=()):
        if isinstance(value, dict):
            for name, child in value.items(): yield from leaves(child, path + (name,))
        else: yield path, value
    paths = list(leaves(request))
    assert len(paths) > 12
    for path, value in paths:
        if path == ("ssdm", "request_id"): continue
        changed, node = deepcopy(request), None
        node = changed
        for name in path[:-1]: node = node[name]
        node[path[-1]] = value + " " if isinstance(value, str) else value + 1
        assert memo.key(changed) != key, path
    changed = deepcopy(request)
    changed["ssdm"]["limits"]["deadline_ms"] = 10000
    assert memo.key(changed) != key  # additional envelope fields are part of the key too
    provider.decide(request)
    changed = deepcopy(request)
    changed["state"] += " "  # semantically equivalent JSON is still an exact-input miss
    provider.decide(changed)
    assert stock.calls == 2 and memo.hits == 0


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "adapter_rejection"])
def test_failed_operation_is_never_cached_and_later_success_scores_live(failure):
    stock, memo, provider, _ = memo_provider()
    request = initial_request(provider)
    if failure == "adapter_rejection":
        original = provider.decider._success
        provider.decider._success = lambda *args: {"ssdm": {"status": "error"}}
    else:
        stock.failure = failure
    assert provider.decide(request)["ssdm"]["status"] != "ok"
    assert not memo.entries and memo.pending is None
    if failure == "adapter_rejection": provider.decider._success = original
    stock.failure = None
    assert provider.decide(request)["ssdm"]["status"] == "ok"
    assert stock.calls == 2 and len(memo.entries) == 1 and memo.hits == 0


def test_cache_is_bounded_to_32_and_other_slots_always_score_live():
    stock, memo, provider, _ = memo_provider()
    request = initial_request(provider)
    for index in range(33):
        changed = deepcopy(request)
        state = json.loads(changed["state"])
        state["task_input"] += f" no-load-cache-case-{index}"
        changed["state"] = json.dumps(state)
        assert provider.decide(changed)["ssdm"]["status"] == "ok"
    assert len(memo.entries) == 32
    provider.decide(changed)
    assert stock.calls == 34 and memo.hits == 0  # no eviction or hidden 33rd entry
    other = deepcopy(request)
    other["questions"]["slot-002-fact"] = other["questions"].pop("slot-001-operation")
    assert memo.key(other) is None
    provider.decide(other)
    provider.decide(other)
    assert stock.calls == 36 and len(memo.entries) == 32
    stock.identity["revision_id"] += ":drift"
    with pytest.raises(ValueError, match="revision drift"): memo.key(request)


def test_cached_screen_keeps_all_logical_calls_worlds_arms_failures_and_raw_traces(tmp_path):
    stock, memo, provider, decider = memo_provider()
    examples = runner.screen_data()
    rows, order, equal = runner.evaluate_screen(examples, provider, model="scripted",
        revision=provider.revision()["revision_id"], served_config={
            "calibration": decider.calibration.identity(),
            "policy_masking": "none (served covers every supplied option)"}, raw=tmp_path / "raw.jsonl")
    assert equal and len(order) == 64
    assert {s: len(values) for s, values in rows.items()} == {"record_ref": 32, "entity_name": 32}
    assert stock.calls == 96 and memo.hits == 32 and len(memo.entries) == 32
    assert sum(r["provider_calls"] for values in rows.values() for r in values) == 128
    assert sum(r["forward_calls"] for values in rows.values() for r in values) == 192
    events = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    requests = [e for e in events if e["event"] == "request"]
    responses = [e for e in events if e["event"] == "response"]
    assert len(requests) == len(responses) == 128
    assert len({e["request"]["ssdm"]["request_id"] for e in requests}) == 128
    assert sum(e["response"]["ssdm"]["diagnostics"]["inference_calls"] == 0 for e in responses) == 32
    # Fixed fake chooses read even on explain tasks: valid wrong-task outcomes stay failures.
    assert any(not r["task_correct"] for values in rows.values() for r in values)
    summary = runner.paired_summary(rows, examples)
    assert summary["all"]["task_correct"]["n_worlds"] == 16
    assert summary["all"]["task_correct"]["n_members"] == 32
    assert "weaker than two independent forwards" in runner.OPERATION_EQUALITY_EVIDENCE


def test_cached_full_cohort_proof_never_imports_runtime_or_opens_checkpoints(tmp_path):
    code = r'''
import sys
from pathlib import Path
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported during cache protocol proof')
sys.meta_path.insert(0, Reject())
def audit(event, args):
    if event == 'open':
        path = str(args[0]).lower().replace('\\', '/')
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path:
            raise AssertionError('checkpoint accessed during cache protocol proof')
sys.addaudithook(audit)
from language_growth.tests.test_rwkv_fact_name_screen import test_cached_screen_keeps_all_logical_calls_worlds_arms_failures_and_raw_traces
test_cached_screen_keeps_all_logical_calls_worlds_arms_failures_and_raw_traces(Path(sys.argv[1]))
assert not any(name in sys.modules for name in ('torch', 'transformers', 'safetensors'))
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], cwd=runner.ROOT, check=True)


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


@pytest.mark.parametrize("stage", ["spawn", "partial_spawn", "handshake", "supervisor", "interrupt"])
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
    if stage == 'interrupt': raise KeyboardInterrupt('simulated launcher interruption')
    assert stage == 'supervisor'
    raise RuntimeError('simulated supervisor failure')
runner.supervise = supervise
old = dict(os.environ)
out = Path(sys.argv[1])
assert runner.launch(out) == (130 if stage == 'interrupt' else 1)
assert dict(os.environ) == old
assert receive.closes and send.closes
assert child.kills == child.joins == (0 if stage == 'spawn' else 1)
receipt = json.loads((out / 'receipt.json').read_text())
assert receipt['status'] == 'incomplete'
assert receipt['stop_reason'] == {'partial_spawn':'spawn_error', 'interrupt':'launcher_interrupted'}.get(stage, stage + '_error')
assert receipt['exit_code'] == (130 if stage == 'interrupt' else 1)
assert receipt['exception_type'] == {'spawn':'OSError', 'partial_spawn':'OSError', 'handshake':'BrokenPipeError', 'supervisor':'RuntimeError', 'interrupt':'KeyboardInterrupt'}[stage]
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
