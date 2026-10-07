"""Synthetic/mock evaluation and preparation proofs; no real weights opened."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from language_growth import rwkv_head_fit_audit as audit


def data():
    return json.loads((audit.PRIOR / "corpus.json").read_text())


def test_default_preparation_blocks_runtime_and_checkpoint_access(tmp_path):
    code = r'''
import sys, pathlib
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported in preparation')
sys.meta_path.insert(0, Reject())
def access(event, args):
    if event == 'open':
        path = str(args[0]).lower()
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path.replace('\\','/'):
            raise AssertionError('checkpoint or snapshot accessed during preparation')
sys.addaudithook(access)
from language_growth.rwkv_head_fit_audit import main
sys.argv = ['audit', '--out', sys.argv[1]]
assert main() == 0
assert 'torch' not in sys.modules
'''
    out = tmp_path / "prepared"
    subprocess.run([sys.executable, "-c", code, str(out)], cwd=audit.ROOT, check=True)
    registration = json.loads((out / "registration.json").read_text())
    assert registration["status"] == "prepared_not_run"
    assert registration["checkpoint_sha256"] == audit.CHECKPOINT_SHA256
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    result = subprocess.run([sys.executable, "-m", "language_growth.rwkv_head_fit_audit",
                             "--dry-run", "--out", str(out)], cwd=audit.ROOT)
    assert result.returncode == 1
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_only_existing_pinned_corpus_and_exact_96_rows(monkeypatch, tmp_path):
    rows = audit.rows_for(data())
    assert len(rows) == 96
    assert sum(r["split"] == "development_train" for r in rows) == 64
    assert sum(r["split"] == "development_validation" for r in rows) == 32
    assert {r["qtype"] for r in rows} == {"choice"}
    assert len({(r["task_id"], r["qid"]) for r in rows}) == 96
    assert {r["slot"] for r in rows} == {"operation", "fact", "left", "right", "relation"}
    changed = data()
    changed["development_train"].pop()
    with pytest.raises(ValueError, match="exactly 64"):
        audit.rows_for(changed)
    hashes = dict(audit.INPUT_HASHES)
    monkeypatch.setitem(audit.INPUT_HASHES, "corpus.json", "0" * 64)
    with pytest.raises(ValueError, match="original input drift"):
        audit.preparation(tmp_path / "drift")
    assert hashes["corpus.json"] != audit.INPUT_HASHES["corpus.json"]


def test_preregistration_pins_source_inputs_checkpoint_recipe_and_software(monkeypatch, tmp_path):
    reg = audit.preparation(tmp_path / "prepared")
    frozen = tmp_path / "frozen.json"
    audit.shared.write(frozen, reg)
    monkeypatch.setattr(audit, "FROZEN_REGISTRATION", frozen)
    audit.verify_frozen(reg)
    for key in ("config", "input_sha256", "source_sha256", "checkpoint_sha256", "software", "corpus_sha256"):
        changed = deepcopy(reg)
        changed[key] = "drift"
        with pytest.raises(ValueError, match="preregistration drift"):
            audit.verify_frozen(changed)
    for path in ("heads/decide_backend.py", "heads/typed.py", "heads/layout.py", "train/lineage.py",
                 "language_growth/rwkv_head_learning.py", "language_growth/rwkv7_cpu_parent.py", audit.PLAN):
        assert path in reg["source_sha256"]


class FakeBackend:
    def __init__(self, rows, failures=None):
        self.rows, self.failures, self.calls = rows, failures or {}, []
    def plan(self, request):
        return SimpleNamespace(max_call_input_tokens=100)
    def score(self, request, deadline):
        index = len(self.calls)
        self.calls.append(deepcopy(request))
        failure = self.failures.get(index)
        if failure == "exception":
            raise RuntimeError("synthetic provider failure")
        row = self.rows[index]
        criteria = request["questions"][row["qid"]]["criteria"]
        k = len(criteria)
        logits = {key: math.log(.7 if key == row["label"] else .3 / (k - 1)) for key in criteria}
        if failure == "nan":
            logits[row["label"]] = float("nan")
        if failure == "missing":
            logits.pop(row["label"])
        if failure == "unnormalized":
            logits = {key: value + 5 for key, value in logits.items()}
        return SimpleNamespace(logits={row["qid"]: logits}, inference_calls=1, total_input_tokens=100)


def test_single_pass_exact_ce_no_label_inputs_and_failure_denominators(tmp_path):
    rows = audit.rows_for(data())
    backend = FakeBackend(rows, {1: "exception", 2: "nan", 3: "missing", 4: "unnormalized"})
    raw = tmp_path / "raw.jsonl"
    results = audit.evaluate_rows(rows, backend, raw)
    assert len(results) == len(backend.calls) == 96
    assert [r["status"] for r in results[:5]] == ["ok"] + ["failed"] * 4
    assert results[0]["ce"] == pytest.approx(-math.log(.7))
    assert backend.calls == [r["request"] for r in rows]
    assert all("label" not in r for r in backend.calls)
    summary = audit.aggregate(results)
    train = next(r for r in summary if (r["split"], r["kind"], r["slot"], r["qtype"]) ==
                 ("development_train", "all", "all", "all"))
    assert (train["n"], train["n_ok"], train["correct"]) == (64, 60, 60)
    assert train["accuracy_all"] == 60 / 64
    assert train["ce"] is None
    assert train["ce_on_successful_rows"] == pytest.approx(-math.log(.7))
    lines = [json.loads(line) for line in raw.read_text().splitlines()]
    assert sum(r["event"] == "request" for r in lines) == 96
    assert sum(r["event"] == "row_result" for r in lines) == 96
    assert any("nan" in r.get("score_repr", "") for r in lines)


def test_teacher_forcing_is_explicit_current_label_not_in_request():
    original = data()
    changed = deepcopy(original)
    last = changed["development_train"][-1]
    q = next(iter(last["model_inputs"][-1]["questions"].values()))
    old = last["training_only"]["targets"][-1]["option_id"]
    last["training_only"]["targets"][-1]["option_id"] = next(k for k in q["criteria"] if k != old)
    a, b = audit.rows_for(original), audit.rows_for(changed)
    assert [r["request"] for r in a] == [r["request"] for r in b]
    assert a[63]["label"] != b[63]["label"]
    assert json.loads(a[63]["request"]["state"])["proposed_slot_choices"]


def test_ties_are_canonical_and_oversized_row_never_scores(tmp_path):
    rows = audit.rows_for(data())[:1]
    backend = FakeBackend(rows)
    backend.plan = lambda request: SimpleNamespace(max_call_input_tokens=2049)
    result = audit.evaluate_rows(rows, backend, tmp_path / "too-long")[0]
    assert result["status"] == "failed" and not backend.calls
    qid = rows[0]["qid"]
    options = rows[0]["request"]["questions"][qid]["criteria"]
    backend.plan = lambda request: SimpleNamespace(max_call_input_tokens=100)
    backend.score = lambda *args: SimpleNamespace(logits={qid: {k: -math.log(len(options)) for k in reversed(options)}},
                                                 inference_calls=1, total_input_tokens=100)
    result = audit.evaluate_rows(rows, backend, tmp_path / "ties")[0]
    assert result["selected"] == min(options)


@pytest.mark.parametrize("train,dev,fail,diagnosis", [
    (.5, .5, 0, "mapping_not_strongly_fit_representation_vs_optimization_unresolved"),
    (.8, .7, 0, "fit_transfer_gap_on_reused_development_worlds"),
    (.8, .8, 0, "strong_teacher_forced_slots_do_not_establish_free_running_capability"),
    (.9, .9, 1, "execution_failures_no_fit_transfer_classification"),
])
def test_interpretation_rules_are_fixed_and_prior_rollouts_not_rerun(train, dev, fail, diagnosis):
    rows = [{"split": split, "kind": "all", "slot": "all", "qtype": "all",
             "accuracy_all": accuracy, "n_failed": fail} for split, accuracy in
            (("development_train", train), ("development_validation", dev))]
    prior = {"before": [{"task_correct": True, "kernel_accepted": True}],
             "after": [{"task_correct": False, "kernel_accepted": True}]}
    result = audit.interpretation(rows, prior)
    assert result["diagnosis"] == diagnosis
    assert result["prior_free_running"]["before"]["useful"] == 1


def test_low_ram_and_unauthenticated_cli_never_launch(monkeypatch, tmp_path):
    import psutil
    monkeypatch.setattr(audit, "verify_frozen", lambda reg: None)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    monkeypatch.setattr(audit.multiprocessing, "get_context", lambda *args: pytest.fail("spawned under RAM floor"))
    assert audit.launch(tmp_path / "refused") == 125
    result = subprocess.run([sys.executable, "-m", "language_growth.rwkv_head_fit_audit", "--child",
                             "--out", str(tmp_path / "child")], cwd=audit.ROOT)
    assert result.returncode == 2
    assert not (tmp_path / "child").exists()
    monkeypatch.setattr(audit.shared, "authorize_worker", lambda *args: (_ for _ in ()).throw(RuntimeError("unauthorized")))
    monkeypatch.setattr(audit, "_run", lambda *args: pytest.fail("unauthenticated execution"))
    with pytest.raises(RuntimeError, match="unauthorized"):
        audit.child(tmp_path, None, os.getpid(), b"x" * 32)
    assert not (tmp_path / "stdout.txt").exists()


def test_synthetic_restore_has_no_optimizer_and_strict_lineage(monkeypatch, tmp_path):
    import torch
    from language_growth.tests.test_rwkv_head_learning import tiny_backend
    from train.data import items_from_records
    from train.lineage import lineage
    from language_growth.training_records import export_training_records
    torch.set_num_threads(2)
    backend = tiny_backend()
    items = items_from_records(export_training_records(data()["development_train"][:3]), backend.encode,
                               template=backend.template, layout=backend.layout)
    config = {"seed": audit.shared.CONFIG["seed"]}
    trainable = {n: p.detach().clone() for n, p in backend.model.named_parameters() if n.startswith(("head.", "typed."))}
    saved = {"step": 24, "config": config, "init_seed": config["seed"], "lora_modules": [],
             "lineage": lineage(backend.model.backbone, items), "trainable": trainable}
    checkpoint = tmp_path / "synthetic.pt"
    torch.save(saved, checkpoint)
    monkeypatch.setattr(audit, "CHECKPOINT_SHA256", audit.sha(checkpoint))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kw: pytest.fail("optimizer constructed"))
    frozen = audit.shared.state_digest(backend.model.backbone.model)
    with torch.no_grad():
        for n, p in backend.model.named_parameters():
            if n in trainable:
                p.add_(.1)
    audit.restore_head(backend.model, checkpoint, items, config)
    assert all(torch.equal(p, trainable[n]) for n, p in backend.model.named_parameters() if n in trainable)
    assert not any(p.requires_grad for p in backend.model.parameters())
    assert frozen == audit.shared.state_digest(backend.model.backbone.model)
    for issue in ("keys", "lineage", "step", "nonfinite"):
        bad = deepcopy(saved)
        if issue == "keys":
            bad["trainable"].pop(next(iter(trainable)))
        elif issue == "lineage":
            bad["lineage"] = {}
        elif issue == "step":
            bad["step"] = 23
        else:
            bad["trainable"][next(iter(trainable))].fill_(float("nan"))
        torch.save(bad, checkpoint)
        monkeypatch.setattr(audit, "CHECKPOINT_SHA256", audit.sha(checkpoint))
        with pytest.raises(ValueError):
            audit.restore_head(backend.model, checkpoint, items, config)
