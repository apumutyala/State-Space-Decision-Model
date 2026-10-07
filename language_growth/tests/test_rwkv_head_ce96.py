"""One exposure ablation: sampler and evaluation-only proofs, no real weights."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from language_growth import rwkv_head_learning as runner
from language_growth import rwkv_head_ce96_evaluation as reporting


def test_recipe_changes_only_step_count_and_keeps_exact_corpus():
    prior = reporting.references()["old_recipe"]
    old = deepcopy(prior["config"])
    old["steps"] = 96
    old["timeout_seconds"] = 1200  # operational guard only; absent from TrainConfig
    assert runner.CONFIG == old
    data = runner.corpus()
    assert runner.digest(data) == prior["corpus_sha256"]
    contract = reporting.exposure_contract(data, runner.CONFIG)
    assert contract["slot_exposures"] == 192
    assert contract["complete_passes"] == 3
    assert contract["dropped_tail"] == 0
    for change in ({"batch_size": 3}, {"steps": 97}, {"lr_head": .002}, {"timeout_seconds": 900}):
        with pytest.raises(ValueError):
            reporting.exposure_contract(data, {**runner.CONFIG, **change})


def test_actual_sampler_replays_three_passes_and_same_initial_24_batches():
    import torch
    from language_growth.tests.test_rwkv_head_learning import tiny_backend
    torch.set_num_threads(2)
    data = runner.corpus()["development_train"]
    longer = runner.trainer_for(tiny_backend(), data)
    reporting.training_config_parity(longer.cfg.to_json())
    prior = runner.trainer_for(tiny_backend(), data, steps=24)
    all_ids = {item.id for item in longer.train_items}
    assert len(all_ids) == 64
    batches = [[item.id for item in longer._next_batch()] for _ in range(96)]
    first = [[item.id for item in prior._next_batch()] for _ in range(24)]
    assert batches[:24] == first
    for start in (0, 32, 64):
        pass_ids = [item for batch in batches[start:start + 32] for item in batch]
        assert len(pass_ids) == 64 and set(pass_ids) == all_ids
        assert set(Counter(pass_ids).values()) == {1}
    assert set(Counter(item for batch in batches for item in batch).values()) == {3}
    assert (longer.epoch, longer.cursor) == (2, 64)
    assert longer.step == prior.step == 0  # sampler-only proof; no training


def test_preparation_has_no_runtime_or_checkpoint_access(tmp_path, monkeypatch):
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('runtime imported during preparation')
sys.meta_path.insert(0, Reject())
def access(event, args):
    if event == 'open':
        path = str(args[0]).lower().replace('\\','/')
        if path.endswith(('.pt', '.safetensors')) or '/snapshots/' in path:
            raise AssertionError('real checkpoint access during preparation')
sys.addaudithook(access)
from language_growth.rwkv_head_learning import main
sys.argv = ['runner', '--dry-run', '--out', sys.argv[1]]
assert main() == 0
assert 'torch' not in sys.modules
'''
    out = tmp_path / "prepared"
    subprocess.run([sys.executable, "-c", code, str(out)], cwd=runner.ROOT, check=True)
    reg = json.loads((out / "registration.json").read_text())
    assert reg["variant_base"] == "bb2eadd798b5d5ae317524767b37115e9a8df168"
    assert reg["exposure"]["complete_passes"] == 3
    assert reg["reference_sha256"] == reporting.references()["sha256"]
    assert reg["status"] == "prepared_not_run"
    monkeypatch.setattr(runner, "FROZEN_REGISTRATION", out / "registration.json")
    runner.verify_frozen(reg)
    for key in ("variant_base", "training_delta", "operational_delta", "exposure", "reference_sha256"):
        changed = deepcopy(reg)
        changed[key] = "drift"
        with pytest.raises(ValueError, match="preregistration drift"):
            runner.verify_frozen(changed)


def test_measured_operational_budget_and_exact_watchdog_boundary():
    budget = reporting.operational_budget()
    assert budget["predicted_total_seconds"] == 1013.67
    assert budget["headroom_seconds"] == 186.33
    assert budget["timeout_seconds"] == [900, 1200]
    assert budget["scientific_recipe_change"] is False
    assert runner.CONFIG["timeout_seconds"] == 1200
    assert runner.stop_reason(1199.999, 20 * 1024**3) is None
    assert runner.stop_reason(1200, 20 * 1024**3) == ("timeout", 124)
    assert runner.stop_reason(0, 1) == ("memory_guard", 125)


def test_teacher_forced_scoring_is_optimizer_free_and_keeps_failed_rows(monkeypatch, tmp_path):
    import train.trainer
    from language_growth.rwkv_head_fit_audit import rows_for
    from language_growth.tests.test_rwkv_head_fit_audit import FakeBackend
    def forbidden(*args, **kwargs):
        pytest.fail("optimizer/trainer constructed during post-evaluation")
    monkeypatch.setattr(train.trainer, "Trainer", forbidden)
    data = runner.corpus()
    rows = rows_for(data)
    backend = FakeBackend(rows, {0: "exception"})
    result = reporting.teacher_forced(data, backend, tmp_path, "final")
    assert len(result["rows"]) == len(backend.calls) == 96
    assert result["rows"][0]["status"] == "failed"
    assert result["rows"][0]["ce"] is None
    assert backend.calls == [r["request"] for r in rows]
    assert all("label" not in r for r in backend.calls)


def test_world_cluster_pairing_and_undefined_failed_ce():
    rows = deepcopy(reporting.references()["fit_rows"])
    comparison = reporting.paired_slots(rows, rows, "correct", "development_train")
    assert (comparison["n_slots"], comparison["n_worlds"]) == (64, 24)
    assert comparison["estimate"] == 0 and comparison["ci95"] == [0, 0]
    assert reporting.paired_slots(rows, rows, "ce", "development_validation")["ci95"] == [0, 0]
    changed = deepcopy(rows)
    changed[0]["ce"] = None
    incomplete = reporting.paired_slots(changed, rows, "ce", "development_train")
    assert incomplete["failed_pairs"] == 1
    assert incomplete["estimate"] is None and incomplete["ci95"] is None
    with pytest.raises(ValueError, match="pair exactly"):
        reporting.paired_slots(rows[:-1], rows, "correct", "development_validation")


def test_comparisons_include_both_prior_checkpoints_and_exact_initial_parity():
    prior = reporting.references()
    before, after = deepcopy(prior["learning"]["before"]), deepcopy(prior["learning"]["after"])
    reporting.initial_parity(before, prior["learning"]["before"])
    changed = deepcopy(before)
    changed[0]["task_correct"] = not changed[0]["task_correct"]
    with pytest.raises(ValueError, match="step-0"):
        reporting.initial_parity(changed, prior["learning"]["before"])
    teacher = {"rows": prior["fit_rows"]}
    report = reporting.comparison_report(before, after, teacher, teacher, prior)
    assert set(report["free_running"]) == set(report["teacher_forced"]) == {"step0", "step24"}
    assert report["free_running"]["step24"]["task_correct"]["estimate"] == 0
    assert report["free_running"]["step24"]["kernel_accepted"]["estimate"] == 0
