"""A-PREREG-JEV3-SYM (draft) package checks, CPU / stdlib-light: both seeds' cells, the budget, the gates, the slice
ordering and the slice/train state-disjointness (builder + in-job)."""
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ROOT / "jobs/m2_first_cell"


def cell(m):
    return json.loads((M / f"m2_first_cell_rwkv7_{m}.json").read_text(encoding="utf-8"))


def params(m):
    return json.loads((M / f"job_params_{m}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("seed", [0, 1])
def test_jev3_sym_cells(seed):
    from jobs.common import costs
    c, p = cell(f"jev3_sym_s{seed}"), params(f"jev3_sym_s{seed}")
    assert c["seed"] == seed and c["layout"] == "sym-fork" and c["head"] == "fork"
    assert c["qwen35"]["repo"] == "Qwen/Qwen3.5-4B"  # our stack: stock backbone, not a third-party fine-tune
    cons = c["consistency"]
    assert cons["view_frac"] == 0.25 and cons["families"] == ["format", "instruction", "option"]
    assert cons["controller"] is None and cons["lambda"] == 0.5
    assert c["label_smoothing_noul"] == 0.0 and max(c["t_ladder"]) == c["eval_max_t"] == 4096
    assert c["assert_no_truncation"] and c["data_pins"]["status"] == "final"  # re-gate cleared (empty list)
    assert c["min_train_items"] == c["n_train_records"] == 5824 and c["max_steps"] == 728 and c["min_steps"] == 110
    jk = p["jevkit"]
    assert jk["slice_first"] and jk["slice_floor"] and jk["stock_shim"] and jk["paired_primary"] == "native"
    assert jk["g1_keep_gate"] and jk["g1_max_dp"] == 0.0 and jk["flip_ops_all"]
    assert jk["slice_state_disjoint"]["train_records"].startswith(p["train_dir"] + "/")
    assert costs.worst_case_usd(p) <= p["hard_cap_usd"] <= 22.0
    assert p["max_duration_minutes"] == sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"]
    assert f"{costs.worst_case_usd(p):.2f}" in (M / "JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"job_params_jev3_sym_s{seed}.json" in (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")


def test_artifacts_landing_and_g1_per_arm(tmp_path):
    p = params("jev3_sym_s0")
    assert p["jevkit"]["g1_per_arm"] is True
    job = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert job.index("landing.py") < job.index("export_phase final")  # landing.json lands in the final export
    import subprocess
    out = tmp_path / "out"
    (out / "screen" / "jevkit").mkdir(parents=True)
    (out / "screen" / "jevkit" / "slice_first.json").write_text("{}", encoding="utf-8")
    import os
    import time as _t
    (out / "screen" / "jevkit" / "native.preds.jsonl").write_text("", encoding="utf-8")
    os.utime(out / "screen" / "jevkit" / "slice_first.json", (_t.time() - 60, _t.time() - 60))
    man = tmp_path / "MANIFEST.jsonl"
    man.write_text('{"phase": "cell", "t": "2026-10-11T00:00:00Z"}\n', encoding="utf-8")
    subprocess.run([sys.executable, str(M / "landing.py"), str(out), str(man), '{"cell": 0}', str(out / "landing.json")],
                   check=True)
    rec = json.loads((out / "landing.json").read_text(encoding="utf-8"))
    assert rec["phases"][0]["phase"] == "cell" and rec["gates"] == {"cell": 0}
    assert rec["slice_first_before_public_preds"] is True and all("sha256" in f for f in rec["files"])


def test_ids_sha_both_conventions(tmp_path, capsys):
    import hashlib
    from scripts.verify_ids_sha import main as verify
    _gz(tmp_path / "r.gz", [{"id": "b"}, {"id": "a"}])
    no_nl, nl = hashlib.sha256(b"a\nb").hexdigest(), hashlib.sha256(b"a\nb\n").hexdigest()
    assert verify(["--records", str(tmp_path / "r.gz"), "--expect", nl]) == 0
    assert verify(["--records", str(tmp_path / "r.gz"), "--expect", no_nl]) == 0
    assert verify(["--records", str(tmp_path / "r.gz"), "--expect", "0" * 64]) == 1
    pytest.importorskip("torch")  # train.run_cell imports torch
    from train.run_cell import check_ids_pin
    assert check_ids_pin(["b", "a"], nl, "t")["convention"] == "trailing_nl"
    with pytest.raises(ValueError):
        check_ids_pin(["b", "a"], "1" * 64, "t")
    assert "no_trailing_nl=" + no_nl in capsys.readouterr().out  # both logged before the refusal


def test_exclude_files_pinned_and_fail_closed(tmp_path, monkeypatch):
    import hashlib
    import train.train_set as TS
    c = cell("jev3_sym_s0")
    man = set()
    with gzip.open(ROOT / "receipts/2026-10-11-jev3-sampler/jev3_train_exclude_subs.txt.gz", "rt") if False else \
            open(ROOT / "receipts/2026-10-11-jev3-sampler/jev3_train_exclude_subs.txt", encoding="utf-8") as f:
        man = {ln.strip() for ln in f if ln.strip()}
    ids, rec = TS.exclude_files(c, man | {"other"})
    assert len(ids) == 136 and [r["n"] for r in rec] == [136, 0]
    assert c["n_train_records"] == 5960 - 136
    monkeypatch.setattr(TS, "ROOT", tmp_path)
    (tmp_path / "x.txt").write_bytes(b"a\nb\n")
    good = hashlib.sha256(b"a\nb\n").hexdigest()
    for spec, err in (({"file": "x.txt", "sha256": "0" * 64, "n": 2}, "sha256"),
                      ({"file": "x.txt", "sha256": good, "n": 3}, "n"),
                      ({"file": "x.txt", "sha256": good, "n": 2}, "not in the set")):
        with pytest.raises(TS.TrainSetMismatch, match=err):
            TS.exclude_files({"train_set": {"exclude_files": [spec]}}, {"a"})


def test_slice_first_runs_before_any_screen_row():
    src = (M / "screen.py").read_text(encoding="utf-8")
    i = src.index("def score(a)")
    assert src.index("slice_first(jk0", i) < src.index("G.index_predict", i)


def _gz(p, rows):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_in_job_slice_state_disjoint(tmp_path):
    from jobs.m2_first_cell.screen import slice_state_disjoint
    _gz(tmp_path / "s.gz", [{"state": "a"}, {"state": {"k": 1}}])
    _gz(tmp_path / "t.gz", [{"state": "b"}])
    assert slice_state_disjoint(tmp_path / "s.gz", tmp_path / "t.gz")["shared"] == 0
    _gz(tmp_path / "t2.gz", [{"state": {"k": 1}}])
    with pytest.raises(ValueError):
        slice_state_disjoint(tmp_path / "s.gz", tmp_path / "t2.gz")


def test_builder_slice_skips_training_states(tmp_path):
    from scripts.jev12_balanced_draw import main
    d = tmp_path / "jev1"
    d.mkdir()
    rows = []
    for i in range(40):  # every state appears twice: a slice row could otherwise share a train row's state
        rows.append({"id": f"jev1:intent:{i}", "family": "jev1_intent", "source": "t", "state": {"k": f"s{i // 2}"},
                     "questions": {"q": {"type": "noul", "instructions": f"q{i}"}}, "labels": {"q": "true"},
                     "aug": {"jev1": {"family8": "intent"}}})
    _gz(d / "records.jsonl.gz", rows)
    assert main(["--src", str(d), "--train-per-family", "10", "--calib-per-family", "5", "--score-share", "0.25",
                 "--name", "x", "--out", str(tmp_path / "o")]) == 0
    rep = json.loads((tmp_path / "o/x/draw-report.json").read_text(encoding="utf-8"))
    assert rep["calib_state_disjoint"] is True
    from jobs.m2_first_cell.screen import slice_state_disjoint
    assert slice_state_disjoint(tmp_path / "o/x-calib/records.jsonl.gz", tmp_path / "o/x/records.jsonl.gz")["shared"] == 0
