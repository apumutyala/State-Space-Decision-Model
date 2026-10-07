"""Small random CPU fixtures only; never load the real parent checkpoint."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from language_growth import operation_cache as cache
from language_growth import rwkv_operation_cache as runner


def request():
    from language_growth.parent_bridge import SERIALIZATION_REF
    return {"state": json.dumps({"schema": SERIALIZATION_REF, "task_input": "arbitrary words",
                                 "accepted_partial_program": [], "proposed_slot_choices": []}),
            "questions": {"slot-001-operation": {"type": "choice", "criteria": {"x": "a", "y": "b"}}}}


def tiny():
    import torch
    from heads import ByteEncoder
    from heads.backbone import tiny_backbone
    from heads.model import DecisionModel
    from heads.layout import build_row
    model = DecisionModel(tiny_backbone(hidden=8, layers=1, seed=31), "qend_axes", seed=20261006)
    model.backbone.requires_grad_(False)
    encode = ByteEncoder()
    row = build_row(request(), "slot-001-operation", encode)
    item = SimpleNamespace(id="random-world/slot-001-operation", row=row, label=0, axis_target=None)
    identity = {"parent": {"model": "random-tiny"}, "encoder": encode.identity(), "template": {"name": "jrt-plain-v1"}}
    return model, item, identity


def test_default_and_dry_run_no_torch_checkpoint_or_snapshot_access(tmp_path):
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch','transformers','safetensors'):
            raise AssertionError('runtime imported during preparation')
sys.meta_path.insert(0,Reject())
def access(event,args):
    if event=='open':
        path=str(args[0]).lower().replace('\\','/')
        if path.endswith(('.pt','.safetensors')) or '/snapshots/' in path:
            raise AssertionError('checkpoint accessed during preparation')
sys.addaudithook(access)
from language_growth.rwkv_operation_cache import main
sys.argv=['runner','--out',sys.argv[1]]+sys.argv[2:]
assert main()==0
assert 'torch' not in sys.modules
'''
    for mode in ([], ["--dry-run"]):
        out = tmp_path/("dry" if mode else "default")
        subprocess.run([sys.executable, "-c", code, str(out), *mode], cwd=runner.ROOT, check=True)
        assert json.loads((out/"registration.json").read_text()) == json.loads((runner.PREPARED/"registration.json").read_text())


def test_frozen_records_only_operation_and_reject_prefix_or_gold():
    data = runner.corpus()
    records = cache.operation_records(data)
    assert {s: len(r) for s, r in records.items()} == {"development_train": 24, "development_validation": 12}
    assert all(cache.initial_operation(r["request"]) for rows in records.values() for r in rows)
    broken = deepcopy(data)
    first = broken["development_train"][0]["model_inputs"][0]
    state = json.loads(first["state"])
    state["gold"] = "op001"
    first["state"] = json.dumps(state, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError): cache.operation_records(broken)
    broken = deepcopy(data)
    first = broken["development_train"][0]["model_inputs"][0]
    state = json.loads(first["state"])
    state["accepted_partial_program"] = [{"gold": "op001"}]
    first["state"] = json.dumps(state)
    with pytest.raises(ValueError): cache.operation_records(broken)


def test_exact_boundary_and_all_later_calls_delegate():
    calls = []
    class Stub:
        def __init__(self, name): self.name = name
        def revision(self): return {"model_id": "tiny", "revision_id": self.name}
        def plan(self, req): calls.append((self.name,"plan")); return self.name
        def score(self, req, check): check(); calls.append((self.name,"score")); return self.name
    learned, stock = Stub("learned"), Stub("stock")
    backend = cache.OperationOnlyBackend(learned, stock)
    original = request()
    assert backend.plan(original) == backend.score(original, lambda: None) == "learned"
    wording = deepcopy(original)
    wording["questions"]["slot-001-operation"]["criteria"] = {"p": "unknown", "q": "utterance"}
    state = json.loads(wording["state"]); state["task_input"] = "new task"
    wording["state"] = json.dumps(state)
    assert backend.route(wording) is learned
    mutations = []
    for qid in ("slot-002-fact", "slot-003-relation", "slot-002-operation", "slot-001-direct-comparison-relation"):
        req = deepcopy(original); req["questions"] = {qid: req["questions"]["slot-001-operation"]}; mutations.append(req)
    for field in ("accepted_partial_program", "proposed_slot_choices"):
        req = deepcopy(original); state = json.loads(req["state"]); state[field] = [{"x":"y"}]; req["state"] = json.dumps(state); mutations.append(req)
    req = deepcopy(original); req["questions"]["other"] = {}; mutations.append(req)
    for req in mutations:
        assert backend.plan(req) == backend.score(req, lambda: None) == "stock"
    assert backend.revision()["stock"] == stock.revision()


def test_cache_live_parity_no_label_or_arbitrary_meta_in_identity():
    import torch
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert not torch.cuda.is_available(), "registered CPU runtime requires CUDA unavailable"
    assert torch.__version__ == runner.registration()["software"]["torch"]
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        model, item, identity = tiny()
        entry = cache.capture(model, item, **identity)
        with torch.no_grad():
            torch.testing.assert_close(cache.cached_forward(model, [entry]), model([item.row]).log_probs)
        item.label = 1; item.axis_target = (99.,)*6; item.row.meta["gold"] = "secret"
        repeated = cache.capture(model, item, **identity)
        assert entry["key"] == repeated["key"]
        assert "secret" not in json.dumps(entry["identity"])
        assert set(entry) == {"key","identity","q_end","e","option_mask","ks","qtypes"}
        assert all(not entry[k].requires_grad and entry[k].dtype == torch.float32 for k in ("q_end","e"))
        tampered = dict(entry, label=1)
        with pytest.raises(ValueError, match="allowlisted"): cache.validate_entry(tampered)
        changed = deepcopy(identity); changed["parent"]["model"] = "different"
        assert cache.capture(model, item, **changed)["key"] != entry["key"]
    finally: torch.set_num_threads(old)


def test_checkpoint_lineage_exact_96_updates_and_frozen_backbone(tmp_path):
    import torch
    from language_growth.rwkv_head_learning import state_digest
    old = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        model, item, identity = tiny()
        entry = cache.capture(model, item, **identity)
        entries = []
        for i in range(24):
            copied = deepcopy(entry); copied["identity"]["row_id"] = f"tiny-{i}"
            copied["key"] = cache.digest(copied["identity"]); entries.append(copied)
        lin = runner.lineage(entries, identity["parent"], identity["encoder"], identity["template"], [0]*24)
        opt = torch.optim.AdamW(model.head_parameters(), lr=.001, weight_decay=0.)
        before = state_digest(model.backbone)
        runner.save_checkpoint(tmp_path/"initial.pt", model, opt, 0, lin)
        initial = runner.train_state(model)
        history, orders = runner.fit_cached(model, entries, [0]*24, opt)
        assert len(history) == 96 and len(orders) == 8
        assert all(sorted(order) == sorted(e["key"] for e in entries) for order in orders)
        assert state_digest(model.backbone) == before
        assert all(p.grad is None for p in model.backbone.parameters())
        runner.save_checkpoint(tmp_path/"trained.pt", model, opt, 96, lin)
        trained = runner.train_state(model)
        runner.restore_checkpoint(tmp_path/"initial.pt", model, opt, lin, 0)
        assert all(torch.equal(initial[k], runner.train_state(model)[k]) for k in initial)
        runner.restore_checkpoint(tmp_path/"trained.pt", model, opt, lin, 96)
        assert all(torch.equal(trained[k], runner.train_state(model)[k]) for k in trained)
        with pytest.raises(ValueError, match="lineage"):
            runner.restore_checkpoint(tmp_path/"trained.pt", model, opt, {**lin,"parent":{}}, 96)
        with pytest.raises(ValueError, match="lineage"):
            runner.restore_checkpoint(tmp_path/"trained.pt", model, opt, lin, 0)
        changed_targets = runner.lineage(entries, identity["parent"], identity["encoder"], identity["template"], [1]+[0]*23)
        with pytest.raises(ValueError, match="lineage"):
            runner.restore_checkpoint(tmp_path/"trained.pt", model, opt, changed_targets, 96)
    finally: torch.set_num_threads(old)


def test_guards_and_owned_worker_refuse_before_loading(monkeypatch, tmp_path):
    import psutil
    from language_growth import rwkv_head_learning as guard
    monkeypatch.setattr(runner, "preparation", lambda out: out.mkdir())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=12*1024**3-1))
    assert runner.launch(tmp_path/"refused") == 125
    monkeypatch.setattr(guard, "assert_environment", lambda: None)
    with pytest.raises(MemoryError, match="start guard"): runner.run(tmp_path)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 1200.)
    with pytest.raises(TimeoutError): runner.budget_check(0., 20*1024**3)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.)
    with pytest.raises(MemoryError): runner.budget_check(0., 3*1024**3-1)
    with pytest.raises(Exception): runner.worker(tmp_path, None, -1, b"invalid")


def test_pairing_keeps_all_failures_and_refuses_missing_worlds():
    rows = [{"task_id": str(i), "task_correct": False, "kernel_accepted": False} for i in range(12)]
    trained = deepcopy(rows); trained[0].update(task_correct=True, kernel_accepted=True)
    assert runner.paired(trained, rows)["task_correct"]["estimate"] == pytest.approx(1/12)
    assert runner.paired(trained, rows)["task_correct"]["seed"] == 9203
    with pytest.raises(ValueError): runner.paired(trained[:-1], rows)
    examples = [{"task_id": str(i), "training_only": {"kind": ("read", "explain", "compare")[i//4]}} for i in range(12)]
    grouped = runner.paired_by_kind(trained, rows, examples)
    assert grouped["read"]["task_correct"]["estimate"] == .25
    assert grouped["compare"]["task_correct"]["estimate"] == 0
    assert all(grouped[k][m]["n_worlds"] == 4 for k in grouped for m in grouped[k])
