"""job-s6kz7 class of bug, for the unsubmitted hijack_eval draft: a saved-checkpoint params file without "eval_only"
is parsed as a TRAINING mode (setup prices parent_g12, which these params do not cap). The draft now declares
"eval_only": "hijack" (the value the hijack runner lineage expects), and the runner's mode parse refuses any params
with a checkpoint block but no eval_only. CPU only: runs the runner's own mode-parse snippet on the real params."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "jobs/m2_first_cell"
MODE = "hijack_eval"


def _p():
    return json.loads((M / f"job_params_{MODE}.json").read_text(encoding="utf-8"))


def _mode_env_script() -> str:
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    start = sh.index("MODE_ENV=$(python -I - ")
    body = sh[sh.index("<<'EOF'\n", start) + len("<<'EOF'\n"):]
    return body[:body.index("\nEOF\n")]


def _mode_env(params: dict, tmp_path):
    f = tmp_path / f"job_params_{MODE}.json"
    f.write_text(json.dumps(params), encoding="utf-8")
    return subprocess.run([sys.executable, "-I", "-", MODE, params["cell_config"], str(f), ""], cwd=ROOT,
                          input=_mode_env_script(), capture_output=True, text=True, timeout=120)


def test_hijack_draft_declares_eval_only():
    p = _p()
    assert p["eval_only"] == "hijack" and "checkpoint" in p and p["mode"] == MODE
    assert "parent_g12" not in p["phase_caps_minutes"]


def test_checkpoint_params_without_eval_only_are_refused(tmp_path):
    p = dict(_p())
    del p["eval_only"]  # the shape that ran job-s6kz7 as a training mode
    r = _mode_env(p, tmp_path)
    assert r.returncode != 0 and "no eval_only" in (r.stdout + r.stderr)


def test_hijack_draft_never_parses_as_a_training_mode(tmp_path):
    r = _mode_env(_p(), tmp_path)
    out = r.stdout + r.stderr
    if r.returncode == 0:  # a runner that supports eval_only hijack
        assert r.stdout.split()[2] == "hijack"
    else:  # this runner has no hijack branch: a clean exit-5 refusal at mode parse, never the training preflight
        assert "eval_only 'hijack'" in out


def test_every_checkpoint_params_declares_eval_only():
    for f in sorted(M.glob("job_params*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        if "checkpoint" in p:
            assert p.get("eval_only", "none") != "none", f.name


def test_missing_cap_is_exit_5_not_15():
    sh = (M / "first_cell_job.sh").read_text(encoding="utf-8")
    assert 'CAPM=$(cap "$3" 2> /dev/null) || fail 5' in sh and '--cap-min "$CAPM"' in sh
