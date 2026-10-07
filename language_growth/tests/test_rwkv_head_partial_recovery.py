"""Recover a stopped 95/96-row fixture without a model import, call or retry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from language_growth import rwkv_head_ce96_evaluation as recovery
from language_growth import rwkv_head_learning as runner


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def fixture_output(out):
    out.mkdir()
    write(out / "registration.json", json.loads(runner.FROZEN_REGISTRATION.read_text()))
    write(out / "corpus.json", runner.corpus())
    prior = recovery.references()
    for phase in ("before", "after"):
        write(out / f"{phase}.json", prior["learning"][phase])
    steps = [{"event": "train_step", "step": i, "loss": 1.0} for i in range(1, 97)]
    steps.append({"event": "launcher_stopped", "reason": "memory_guard",
                  "elapsed_seconds": 839.938, "minimum_available_bytes": 3215503360})
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in steps) + "\n")
    for phase, rows in (("teacher-before", prior["fit_rows"]), ("teacher-after", prior["fit_rows"][:-1])):
        (out / f"{phase}-raw.jsonl").write_text("\n".join(
            json.dumps({"event": "row_result", **row}) for row in rows) + "\n")
    (out / "checkpoint-after").mkdir()
    (out / "checkpoint-after/ckpt-step0000096.pt").write_bytes(b"opaque checkpoint; hash only")


def test_95_of_96_recovery_blocks_model_imports_preserves_files_and_counts_missing_failure(tmp_path):
    out = tmp_path / "stopped"
    fixture_output(out)
    before = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in out.rglob("*") if p.is_file()}
    code = r'''
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise AssertionError('model runtime imported in partial recovery')
sys.meta_path.insert(0, Reject())
from language_growth.rwkv_head_ce96_evaluation import recover_partial
recover_partial(sys.argv[1])
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code, str(out)], cwd=runner.ROOT, check=True)
    result = json.loads((out / "partial_receipt.json").read_text())
    assert result["status"] == "partial_watchdog_stop" and result["clean_success"] is False
    assert result["flushed_train_steps"] == 96 and result["all_training_steps_flushed"] is True
    assert result["stop"]["exit_code"] == 125
    assert result["stop"]["elapsed_seconds"] == 839.938
    final = result["teacher_forced_after"]
    assert final["flushed_rows"] == 95 and len(final["rows"]) == 96
    assert len(final["missing_rows"]) == 1
    total = lambda split: next(r for r in final["summary"] if
        (r["split"], r["kind"], r["slot"], r["qtype"]) == (split, "all", "all", "all"))
    assert total("development_train")["n_failed"] == 0
    assert total("development_train")["ce"] is not None
    assert total("development_validation")["n_failed"] == 1
    assert total("development_validation")["ce"] is None
    assert total("development_validation")["ce_coverage"] == 31 / 32
    assert len(result["before"]) == len(result["after"]) == 12
    assert set(result["prior_free_running"]) == {"step0", "step24"}
    assert result["recovery_checkpoint_deserialized"] is False
    assert result["recovery_model_loaded_or_run"] is False
    assert all(hashlib.sha256((out / path).read_bytes()).hexdigest() == sha for path, sha in before.items())
    assert {r["path"]: r["sha256"] for r in result["preserved_external_artifacts"]} == before
    with pytest.raises(FileExistsError):
        recovery.recover_partial(out)


def test_duplicate_identity_rejected_and_truncated_tail_audited(tmp_path):
    out = tmp_path / "stopped"
    fixture_output(out)
    path = out / "teacher-after-raw.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="duplicate"):
        recovery.recover_partial(out)
    assert not (out / "partial_receipt.json").exists()
    path.write_bytes(original + b'{"event":')
    result = recovery.recover_partial(out)
    assert result["teacher_forced_after"]["flushed_rows"] == 95
    assert result["teacher_forced_after"]["incomplete_json_tail"]["bytes"] == 9


def test_parent_recovers_only_after_watchdog_stop_and_preserves_stop_code(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(recovery, "recover_partial", lambda out: calls.append(out))
    assert runner.recover_stopped_run(tmp_path, 0) == 0
    assert not calls
    for code in (124, 125):
        assert runner.recover_stopped_run(tmp_path, code) == code
    assert calls == [tmp_path, tmp_path]
    raw = tmp_path / "raw.jsonl"
    raw.write_bytes(b"preserved watchdog stop\n")
    def fails(out):
        raise RuntimeError("recovery fixture error")
    monkeypatch.setattr(recovery, "recover_partial", fails)
    assert runner.recover_stopped_run(tmp_path, 125) == 125
    assert raw.read_bytes() == b"preserved watchdog stop\n"
    assert json.loads((tmp_path / "partial_recovery_error.json").read_text())["exception_type"] == "RuntimeError"
