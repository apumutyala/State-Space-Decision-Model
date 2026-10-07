"""jobs/common/launch.py on a tiny fixture job in a throwaway git repo (CPU, no network, no platform call), plus the
legacy submit/context tables cross-checked against the real JOB_SPECs and a real-tree context build.

    python -X utf8 -m pytest -q tests/test_launch.py
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from jobs.common import launch, lint

zstandard = pytest.importorskip("zstandard")
ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)
pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="needs git and bash")

COSTS = '''PARAMS = {"tiny": "jobs/tiny/job_params.json"}
def attempts(p):
    return 1 + max(p["max_restarts"], p.get("assumed_restart_floor", 0))
def worst_case_usd(p):
    return round((p["build_timeout_minutes"] + attempts(p) * p["max_duration_minutes"]) * p["batch_usd_per_min"], 2)
'''
RUNNER = "#!/usr/bin/env bash\nset -euo pipefail\necho preflight ok\necho ran tiny\n"
TEST = '''import shutil, subprocess
from pathlib import Path
from jobs.tiny import rule
ROOT = Path(__file__).resolve().parents[1]
def test_runner_executes():
    r = subprocess.run([shutil.which("bash"), str(ROOT / "jobs/tiny/tiny_job.sh")], capture_output=True, text=True)
    assert r.returncode == 0 and "ran tiny" in r.stdout
def test_rule_pinned():
    assert rule.MIN_GAIN == 0.5
'''


def params(**kw) -> dict:
    p = {"job": "tiny", "chip": "h100", "chip_count": 1, "max_duration_minutes": 10, "build_timeout_minutes": 60,
         "max_restarts": 0, "assumed_restart_floor": 1, "queue_ttl_minutes": 1440, "resume": "none",
         "hf_cache": True, "batch_usd_per_min": 0.0495, "hard_cap_usd": 5.0, "reserve_minutes": 1,
         "phase_caps_minutes": {"setup": 4, "run": 5}, "model": "org/tiny@abc123", "mission": "tiny-mission",
         "decision_rule": "jobs/tiny/rule.py",
         "submit": {"command": "bash /opt/ssdm/m1/jobs/tiny/tiny_job.sh", "key": "tiny",
                    "checks": launch._checks("TINY_DONE", 300, 600)},
         "context": {"dockerfile": "jobs/tiny/Dockerfile",
                     "data": [{"src": "d/rows.txt", "dest": "rows.txt", "pin": "jobs/tiny/data.sha256"}]}}
    p.update(kw)
    return p


def g(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def make_repo(tmp: Path, p: dict | None = None, runner: str = RUNNER, test: str = TEST, attrs: bool = True,
              extra: dict[str, bytes] | None = None) -> tuple[Path, Path, str]:
    repo, data = tmp / "repo", tmp / "data"
    rows = b"row1\nrow2\n"
    (data / "d").mkdir(parents=True)
    (data / "d/rows.txt").write_bytes(rows)
    files = {
        ".gitattributes": ("*.sh text eol=lf\n*.py text eol=lf\n*.json text eol=lf\n"
                           + ("*.sha256 text eol=lf\n" if attrs else "") + "Dockerfile text eol=lf\n").encode(),
        "jobs/__init__.py": b"", "jobs/common/__init__.py": b"", "jobs/common/costs.py": COSTS.encode(),
        "jobs/DECISION_TREE.md": b"| tiny | `jobs/tiny/` | $3.96 |\n",
        "jobs/tiny/__init__.py": b"",
        "jobs/tiny/JOB_SPEC.md": b"# tiny\nWorst case (60 + 2 x 10) x 0.0495 = $3.96. verdict: PASS/FAIL by rule.py\n",
        "jobs/tiny/job_params.json": json.dumps(p or params(), indent=1).encode(),
        "jobs/tiny/tiny_job.sh": runner.encode(),
        "jobs/tiny/rule.py": b"MIN_GAIN = 0.5\n",
        "jobs/tiny/Dockerfile": b"FROM scratch\nCOPY code/m1 /opt/ssdm/m1\nCOPY data/ /data/\n",
        "jobs/tiny/data.sha256": f"{hashlib.sha256(rows).hexdigest()}  /data/rows.txt\n".encode(),
        "tests/test_tiny.py": test.encode(),
        **(extra or {}),
    }
    for rel, b in files.items():
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
    g(repo.parent, "init", "-q", str(repo))
    g(repo, "add", "-A")
    g(repo, "update-index", "--chmod=+x", "jobs/tiny/tiny_job.sh")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture")
    sha = g(repo, "rev-parse", "HEAD")
    g(repo, "update-ref", "refs/remotes/origin/agent-main", sha)
    return repo, data, sha


def run(repo, data, sha, out, **kw):
    return launch.launch("tiny", sha, out, root=repo, fetch=False, data_dir=data, level=3, **kw)


def test_launch_end_to_end_is_deterministic(tmp_path):
    repo, data, sha = make_repo(tmp_path)
    r1 = run(repo, data, sha, tmp_path / "o1")
    r2 = run(repo, data, sha, tmp_path / "o2", run_job_tests=False)
    assert r1["context"]["sha256"] == r2["context"]["sha256"]
    assert r1["tests"]["files"] == ["tests/test_tiny.py"] and "passed" in r1["tests"]["summary"]
    blob = Path(r1["context"]["file"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == r1["context"]["sha256"] and len(blob) == r1["context"]["size_bytes"]
    with tarfile.open(fileobj=io.BytesIO(zstandard.ZstdDecompressor().decompressobj().decompress(blob))) as t:
        ms = t.getmembers()
        names = [m.name for m in ms]
        assert names == sorted(names)
        assert all(m.mtime == launch.MTIME and m.uid == 0 and m.gid == 0 and m.uname == "" for m in ms)
        mode = {m.name: m.mode for m in ms}
        assert mode["code/m1/jobs/tiny/tiny_job.sh"] == 0o755 and mode["code/m1/jobs/tiny/rule.py"] == 0o644
        assert t.extractfile("data/rows.txt").read() == b"row1\nrow2\n"
        man = json.loads(t.extractfile("CONTEXT_MANIFEST.json").read())
        assert not any(n.startswith("code/m1/.pytest_cache") or "__pycache__" in n for n in names)
    assert man["m1_sha"] == sha and "jobs/tiny/tiny_job.sh" in man["files_sha256"]
    s = json.loads((tmp_path / "o1/submit.json").read_text(encoding="utf-8"))
    sub = s["calls"][-1]
    assert sub["tool"] == "submit_job"
    a = sub["args"]
    assert a["idempotency_key"] == f"tiny-{r1['context']['sha256'][:12]}"
    assert (a["build_timeout_minutes"], a["max_duration_minutes"], a["max_restarts"]) == (60, 10, 0)
    assert a["command"] == "bash /opt/ssdm/m1/jobs/tiny/tiny_job.sh" and a["model"] == "org/tiny@abc123"
    assert a["mission"] == "tiny-mission" and a["checks"]["success"] and a["checks"]["health"]["on_fail"] == "kill"
    assert s["cost"]["worst_case_usd"] == 3.96 and s["cost"]["set_workspace_cap_increment_usd"] == 4
    cc = s["calls"][1]
    assert cc["tool"] == "create_context" and cc["args"]["sha256"] == r1["context"]["sha256"]


def test_refuses_reused_out_dir(tmp_path):
    repo, data, sha = make_repo(tmp_path)
    run(repo, data, sha, tmp_path / "o", run_job_tests=False)
    with pytest.raises(launch.LaunchError, match="fresh --out"):
        run(repo, data, sha, tmp_path / "o", run_job_tests=False)


def test_refuses_unmerged_sha(tmp_path):
    repo, data, sha = make_repo(tmp_path)
    (repo / "x.txt").write_text("x")
    g(repo, "add", "x.txt")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "unmerged")
    with pytest.raises(launch.LaunchError, match="not merged"):
        run(repo, data, g(repo, "rev-parse", "HEAD"), tmp_path / "o")


@pytest.mark.parametrize("case,match", [
    ("build_timeout", "build_timeout"),                       # L-09, new jobs cannot be waived
    ("backslash_n", "no_backslash_n"),                        # L-10
    ("no_shell_test", "shell_test"),                          # L-10
    ("crlf_sha256", "CR in shipped"),                         # L-14
    ("data_pin", "sha256"),                                   # pinned data
    ("failing_test", "job tests failed"),
])
def test_refusals(tmp_path, case, match):
    kw = {}
    if case == "build_timeout":
        kw["p"] = params(build_timeout_minutes=18)
    elif case == "backslash_n":
        kw["runner"] = RUNNER + f"python -m x {BS}n    --flag 1\n"
    elif case == "no_shell_test":
        kw["test"] = "from jobs.tiny import rule\ndef test_rule_pinned():\n    assert rule.MIN_GAIN == 0.5\n"
    elif case == "crlf_sha256":
        kw["attrs"] = False
        kw["extra"] = {"jobs/tiny/extra.sha256": b"abc  /data/x\r\n"}
    elif case == "failing_test":
        kw["test"] = TEST + "def test_fails():\n    assert False\n"
    repo, data, sha = make_repo(tmp_path, **kw)
    if case == "data_pin":
        (data / "d/rows.txt").write_bytes(b"tampered\n")
    with pytest.raises(launch.LaunchError, match=match):
        run(repo, data, sha, tmp_path / "o")


def test_cli_refusal_exit_code(tmp_path, capsys):
    repo, data, sha = make_repo(tmp_path, p=params(build_timeout_minutes=18))
    rc = launch.main(["tiny", "--sha", sha, "--out", str(tmp_path / "o"), "--repo", str(repo), "--data", str(data),
                      "--no-fetch", "--level", "3"])
    assert rc == 1 and "REFUSED" in capsys.readouterr().err


# ----------------------------------------------------------------------------- legacy tables vs the real specs
@pytest.mark.parametrize("rel", lint.params_files(ROOT))
def test_legacy_tables_cover_every_real_params_file(rel):
    job, mode = lint.job_of(rel), lint.mode_of(rel)
    p = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    spec = p.get("context") or launch.LEGACY_CONTEXT[job]
    if "unsupported" in spec:
        return
    sub = p.get("submit") or launch.LEGACY_SUBMIT[(job, mode)]
    runner = sub["command"].split()[1].removeprefix("/opt/ssdm/m1/")
    assert (ROOT / runner).is_file() and runner == lint.runner_of(ROOT, job, p)
    spec_text = (ROOT / f"jobs/{job}/JOB_SPEC.md").read_text(encoding="utf-8") + p.get("_comment", "")
    assert sub["command"] in spec_text, f"{rel}: command not transcribed from JOB_SPEC"
    marker = sub["checks"]["success"]["check"]["all_of"][0]["path"].removeprefix("/output/")
    assert marker in spec_text or marker in (ROOT / runner).read_text(encoding="utf-8") or "RWKV7_" in marker
    a = launch.submit_args(job, mode, p, sub, "0" * 64)
    assert a["build_timeout_minutes"] == p["build_timeout_minutes"] and a["max_duration_minutes"] == p["max_duration_minutes"]
    assert (ROOT / spec["dockerfile"]).is_file()


def test_real_tree_context_builds_without_cr_and_deterministically():
    """The cell_step_profile context from origin/agent-main's tree as the archive sees it (no data, no tests)."""
    if not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    files = launch.archive(ROOT, sha)
    rel = launch.params_rel(files, "cell_step_profile", None)
    ctx = launch.build_context(files, "cell_step_profile", sha, launch.LEGACY_CONTEXT["cell_step_profile"], rel, None)
    assert b"jobs/cell_step_profile/cell_step_job.sh" in ctx["Dockerfile"][0]
    assert launch.cr_violations(ctx, ROOT, sha) == []
    assert launch.deterministic_tar(ctx) == launch.deterministic_tar(dict(reversed(list(ctx.items()))))
