"""jobs/common/launch.py on a tiny fixture job in a throwaway git repo (CPU, no network, no platform call), plus the
legacy submit/context tables cross-checked against the real JOB_SPECs and a real-tree context build.

    python -X utf8 -m pytest -q tests/test_launch.py
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from jobs.common import launch, lint

zstandard = pytest.importorskip("zstandard")
ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)


def bash_executable() -> str | None:
    """Prefer Git for Windows Bash; WindowsApps\\bash.exe is only a WSL shim."""
    git_exe = shutil.which("git")
    if git_exe:
        candidate = Path(git_exe).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")


pytestmark = pytest.mark.skipif(shutil.which("git") is None or bash_executable() is None,
                                reason="needs git and bash")

COSTS = '''PARAMS = {"tiny": "jobs/tiny/job_params.json", "parent": "jobs/tiny/parent_params.json"}
def load(name):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2] / PARAMS[name]).read_text())
def attempts(p):
    return 1 + max(p["max_restarts"], p.get("assumed_restart_floor", 0))
def worst_case_usd(p):
    return round((p["build_timeout_minutes"] + attempts(p) * p["max_duration_minutes"]) * p["batch_usd_per_min"], 2)
def concurrent_cap_usd(names):
    import math
    return math.ceil(sum(worst_case_usd(load(n)) for n in names))
'''
RUNNER = ('#!/usr/bin/env bash\nset -euo pipefail\n'
          'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
          'bash "$ROOT/preflight.sh"\necho ran tiny\n')
TEST = '''import os, shutil, subprocess, sys
from pathlib import Path
from jobs.tiny import rule
ROOT = Path(__file__).resolve().parents[1]
def bash_executable():
    git_exe = shutil.which("git")
    if git_exe:
        candidate = Path(git_exe).resolve().parents[1] / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("bash")
def test_runner_executes():
    env = {**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    r = subprocess.run([bash_executable(), str(ROOT / "jobs/tiny/tiny_job.sh")], capture_output=True, text=True,
                       cwd=ROOT, env=env)
    assert r.returncode == 0 and "ran tiny" in r.stdout and "preflight ok" in r.stdout, r.stdout + r.stderr
def test_rule_pinned():
    assert rule.MIN_GAIN == 0.5
'''


def params(**kw) -> dict:
    p = {"job": "tiny", "chip": "h100", "chip_count": 1, "max_duration_minutes": 10, "build_timeout_minutes": 60,
         "max_restarts": 0, "assumed_restart_floor": 1, "queue_ttl_minutes": 1440, "resume": "none",
         "hf_cache": True, "batch_usd_per_min": 0.0495, "hard_cap_usd": 5.0, "reserve_minutes": 1,
         "phase_caps_minutes": {"setup": 4, "run": 5}, "model": "org/tiny@abc123", "mission": "tiny-mission",
         "decision_rule": "jobs/tiny/rule.py",
         "submit": {"command": "bash /opt/ssdm/m1/jobs/tiny/tiny_job.sh", "env": {}, "key": "tiny",
                    "checks": launch._checks("TINY_DONE", 300, 600)},
         "context": {"dockerfile": "jobs/tiny/Dockerfile",
                     "data": [{"src": "d/rows.txt", "dest": "rows.txt", "pin": "jobs/tiny/data.sha256"}]}}
    p.update(kw)
    return {k: v for k, v in p.items() if v is not DROP}


DROP = object()


def g(repo: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", *a], capture_output=True, text=True,
                          check=True).stdout.strip()


def make_repo(tmp: Path, p: dict | None = None, runner: str = RUNNER, test: str = TEST, attrs: bool = True,
              extra: dict[str, bytes] | None = None, symlink: bool = False,
              spec: bytes = b"# tiny\nWorst case (60 + 2 x 10) x 0.0495 = $3.96. verdict: PASS/FAIL by rule.py\n"
              ) -> tuple[Path, Path, str]:
    repo, data = tmp / "repo", tmp / "data"
    rows = b"row1\nrow2\n"
    (data / "d").mkdir(parents=True)
    (data / "d/rows.txt").write_bytes(rows)
    files = {
        ".gitattributes": ("*.sh text eol=lf\n*.json text eol=lf\n"
                           + ("*.sha256 text eol=lf\n*.py text eol=lf\n" if attrs else "")
                           + "Dockerfile text eol=lf\n").encode(),
        "jobs/__init__.py": b"", "jobs/common/__init__.py": b"", "jobs/common/costs.py": COSTS.encode(),
        "jobs/common/lint.py": (ROOT / "jobs/common/lint.py").read_bytes(),
        "jobs/tiny/preflight.py": b"print('preflight ok')\n",
        "jobs/DECISION_TREE.md": b"| tiny | `jobs/tiny/` | $3.96 |\n",
        "jobs/tiny/__init__.py": b"",
        "jobs/tiny/preflight.sh": b"#!/usr/bin/env bash\nset -euo pipefail\necho preflight ok\n",
        "jobs/tiny/JOB_SPEC.md": spec,
        "jobs/tiny/job_params.json": json.dumps(p or params(), indent=1).encode(),
        "jobs/tiny/parent_params.json": json.dumps({"build_timeout_minutes": 60, "max_duration_minutes": 30,
            "max_restarts": 0, "assumed_restart_floor": 1, "batch_usd_per_min": 0.0495}).encode(),
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
    if symlink:
        blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input="tiny_job.sh",
                              capture_output=True, text=True, check=True).stdout.strip()
        g(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},jobs/tiny/link.sh")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture")
    sha = g(repo, "rev-parse", "HEAD")
    g(repo, "update-ref", "refs/remotes/origin/agent-main", sha)
    return repo, data, sha


def run(repo, data, sha, out, **kw):
    kw.setdefault("review_url", "https://github.com/example/ssdm/pull/1")
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
    assert s["source"] == {"kind": "agent-main", "branch": "agent-main"}
    assert s["review_url"] == "https://github.com/example/ssdm/pull/1"
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


def test_candidate_branch_launches_exact_pushed_stack_tip_without_agent_main_merge(tmp_path):
    repo, data, _ = make_repo(tmp_path)
    g(repo, "checkout", "-qb", "stack/tiny/variant")
    (repo / "candidate.txt").write_text("candidate")
    g(repo, "add", "candidate.txt")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "candidate")
    candidate_sha = g(repo, "rev-parse", "HEAD")
    g(repo, "update-ref", "refs/remotes/origin/stack/tiny/variant", candidate_sha)
    result = run(repo, data, candidate_sha, tmp_path / "candidate", candidate_branch="stack/tiny/variant",
                 run_job_tests=False)
    assert result["sha"] == candidate_sha
    assert result["source"] == {"kind": "candidate_branch", "branch": "stack/tiny/variant"}
    assert result["review_url"] == "https://github.com/example/ssdm/pull/1"


@pytest.mark.parametrize("branch", ["codex/feature", "stack/tiny", "stack/../bad/variant"])
def test_candidate_branch_requires_stack_family_variant(branch, tmp_path):
    repo, data, sha = make_repo(tmp_path)
    with pytest.raises(launch.LaunchError, match="stack/<family>/<variant>"):
        run(repo, data, sha, tmp_path / "candidate", candidate_branch=branch, run_job_tests=False)


def test_candidate_branch_requires_exact_remote_tip(tmp_path):
    repo, data, _ = make_repo(tmp_path)
    g(repo, "checkout", "-qb", "stack/tiny/variant")
    (repo / "candidate.txt").write_text("candidate")
    g(repo, "add", "candidate.txt")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "candidate")
    old_sha = g(repo, "rev-parse", "HEAD")
    (repo / "candidate.txt").write_text("moved tip")
    g(repo, "add", "candidate.txt")
    g(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "moved candidate")
    tip = g(repo, "rev-parse", "HEAD")
    g(repo, "update-ref", "refs/remotes/origin/stack/tiny/variant", tip)
    with pytest.raises(launch.LaunchError, match="exact pushed tip"):
        run(repo, data, old_sha, tmp_path / "candidate", candidate_branch="stack/tiny/variant", run_job_tests=False)


def test_refuses_missing_review_url(tmp_path):
    repo, data, sha = make_repo(tmp_path)
    with pytest.raises(launch.LaunchError, match="independent review URL"):
        run(repo, data, sha, tmp_path / "o", review_url=None, run_job_tests=False)


def test_chained_job_requires_combined_workspace_reservations(tmp_path):
    repo, data, sha = make_repo(tmp_path, p=params(submit={**params()["submit"],
        "depends_on": {"job": "parent-id", "require": "result.ok == true"}}))
    with pytest.raises(launch.LaunchError, match="workspace_cap_jobs"):
        run(repo, data, sha, tmp_path / "o", run_job_tests=False)


def test_chained_job_cap_includes_parent_and_child_worst_cases(tmp_path):
    p = params(workspace_cap_jobs=["parent", "tiny"], workspace_cap_parent_jobs=["parent"],
        submit={**params()["submit"],
        "depends_on": {"job": "parent-id", "require": "result.ok == true"}})
    repo, data, sha = make_repo(tmp_path, p=p)
    result = run(repo, data, sha, tmp_path / "o", run_job_tests=False)
    assert result["cost"]["workspace_cap_jobs"] == ["parent", "tiny"]
    assert result["cost"]["set_workspace_cap_increment_usd"] == 10


def test_chained_job_cannot_list_only_its_own_cost(tmp_path):
    p = params(workspace_cap_jobs=["tiny"], workspace_cap_parent_jobs=["parent"],
        submit={**params()["submit"],
        "depends_on": {"job": "parent-id", "require": "result.ok == true"}})
    repo, data, sha = make_repo(tmp_path, p=p)
    with pytest.raises(launch.LaunchError, match="must include all workspace_cap_parent_jobs"):
        run(repo, data, sha, tmp_path / "o", run_job_tests=False)


def test_chained_job_requires_parent_mapping(tmp_path):
    p = params(workspace_cap_jobs=["parent", "tiny"], submit={**params()["submit"],
        "depends_on": {"job": "parent-id", "require": "result.ok == true"}})
    repo, data, sha = make_repo(tmp_path, p=p)
    with pytest.raises(launch.LaunchError, match="workspace_cap_parent_jobs"):
        run(repo, data, sha, tmp_path / "o", run_job_tests=False)


def test_active_reservation_jobs_are_added_to_workspace_cap(tmp_path):
    p = params(workspace_cap_jobs=["parent", "tiny"], workspace_cap_parent_jobs=["parent"],
        submit={**params()["submit"],
        "depends_on": {"job": "parent-id", "require": "result.ok == true"}})
    repo, data, sha = make_repo(tmp_path, p=p)
    result = run(repo, data, sha, tmp_path / "o", run_job_tests=False, active_reservation_jobs=["parent"])
    assert result["cost"]["workspace_cap_jobs"] == ["parent", "tiny", "parent"]
    assert result["cost"]["set_workspace_cap_increment_usd"] == 16


@pytest.mark.parametrize("case,match", [
    ("build_timeout", "build_timeout"),                       # L-09, new jobs cannot be waived
    ("backslash_n", "no_backslash_n"),                        # L-10
    ("no_shell_test", "shell_test"),                          # L-10
    ("crlf_sha256", "CR in shipped"),                         # L-14
    ("crlf_py", "CR in shipped"),                             # L-14 (.py too)
    ("data_pin", "sha256"),                                   # pinned data
    ("failing_test", "job tests failed"),
    ("preflight_comment", "rate_guard"),                      # PR #93 B2: a comment is not a guard
    ("preflight_echo", "rate_guard"),                         # ... nor an echo of the command
    ("no_decision_rule", "decision_rule"),                    # PR #93 B3: PASS/FAIL spec without the word verdict
    ("null_rule_no_reason", "no_verdict_reason"),             # null needs a reason
    ("hf_cache_missing", "no defaults"),                      # PR #93 B1: no silent hf_cache=false
    ("symlink", "non-regular"),                               # symlinks refused loudly
    ("unknown_mode", "unknown --mode"),
])
def test_refusals(tmp_path, case, match):
    kw, mode = {}, None
    if case == "preflight_comment":
        kw["runner"] = "#!/usr/bin/env bash\nset -euo pipefail\n# TODO preflight\necho ran tiny\n"
    elif case == "preflight_echo":
        kw["runner"] = "#!/usr/bin/env bash\nset -euo pipefail\necho 'python -m jobs.common.rate_guard'\necho ran tiny\n"
    elif case == "no_decision_rule":
        kw["p"] = params(decision_rule=DROP)
        kw["spec"] = b"# tiny\nWorst case $3.96. Decision: PASS if gain >= 0.5 else FAIL\n"
    elif case == "null_rule_no_reason":
        kw["p"] = params(decision_rule=None)
    elif case == "hf_cache_missing":
        kw["p"] = params(hf_cache=DROP)
    elif case == "symlink":
        kw["symlink"] = True
    elif case == "unknown_mode":
        mode = "typo"
    elif case == "crlf_py":
        kw["attrs"] = False
        kw["extra"] = {"jobs/tiny/extra.py": b"x = 1\r\n"}
    elif case == "build_timeout":
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
        run(repo, data, sha, tmp_path / "o", mode=mode)


def test_null_decision_rule_with_reason_launches(tmp_path):
    repo, data, sha = make_repo(tmp_path, p=params(decision_rule=None, no_verdict_reason="throughput probe only"))
    r = run(repo, data, sha, tmp_path / "o", run_job_tests=False)
    assert r["calls"][-1]["args"]["hf_cache"] is True


def test_lint_runs_from_the_launched_sha_not_the_caller(tmp_path, monkeypatch):
    """A looser caller-side lint must not apply: the sha's own jobs/common/lint.py decides."""
    repo, data, sha = make_repo(tmp_path, p=params(build_timeout_minutes=18))
    monkeypatch.setattr(lint, "MIN_BUILD_TIMEOUT", 1)
    with pytest.raises(launch.LaunchError, match="build_timeout"):
        run(repo, data, sha, tmp_path / "o", run_job_tests=False)


def test_cli_refusal_exit_code(tmp_path, capsys):
    repo, data, sha = make_repo(tmp_path, p=params(build_timeout_minutes=18))
    rc = launch.main(["tiny", "--sha", sha, "--out", str(tmp_path / "o"), "--repo", str(repo), "--data", str(data),
                      "--review-url", "https://github.com/example/ssdm/pull/1", "--no-fetch", "--level", "3"])
    assert rc == 1 and "REFUSED" in capsys.readouterr().err


# ----------------------------------------------------------------------------- legacy tables vs the real specs
def spec_blocks(job: str) -> list[dict]:
    """Every submit_job block in jobs/<job>/JOB_SPEC.md, in document order: ```json {...}``` blocks with a "chip"
    key and python-style submit_job(...) calls (bare <placeholders> quoted, true/false mapped)."""
    text = (ROOT / f"jobs/{job}/JOB_SPEC.md").read_text(encoding="utf-8")
    found = []
    for m in re.finditer(r"```json\n(.*?)```", text, re.S):
        if '"chip"' in m.group(1):
            found.append((m.start(), json.loads(m.group(1))))
    for m in re.finditer(r"submit_job\(", text):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        body = re.sub(r"=(<[^>]*>)", r'="\1"', text[m.end():i - 1])
        body = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", body))
        call = ast.parse("f(" + body + ")", mode="eval").body
        found.append((m.start(), {k.arg: ast.literal_eval(k.value) for k in call.keywords}))
    return [b for _, b in sorted(found, key=lambda t: t[0])]


#: (job, mode) -> (block index, overrides stated in the JOB_SPEC prose, evidence strings that prose must contain)
SPEC_BLOCK = {
    ("cell_step_profile", None): (0, {}, []),
    ("fla_rwkv7", None): (0, {}, []),
    ("fla_rwkv7", "truth_v2"): (1, {}, []),
    ("probe9b", None): (0, {}, []),
    ("probe9b", "lean"): (0, {"env": {"PROBE_CONFIG": "lean"}, "idempotency_key": "m1-probe9b-lean-<sha12>",
                          "label": "probe9b: one-layer Q9-AF conversion probe (lean, job-4j3vu rates)"},
                      ['env {"PROBE_CONFIG": "lean"}', 'idempotency_key "m1-probe9b-lean-']),
    ("qual", None): (0, {}, []),
    ("qual", "c_only"): (0, {"env": {"QUAL_MODE": "c_only"}, "max_duration_minutes": 140,
                          "idempotency_key": "m1-hw-qual-c-only-<sha12>"},
                         ['env={"QUAL_MODE": "c_only"}', "max_duration_minutes=140", 'idempotency_key="m1-hw-qual-c-only-']),
    ("rwkv7", "readout"): (0, {}, []),
    ("rwkv7", "lora"): (0, {"env": {"JOB": "lora"}, "max_duration_minutes": 180, "max_restarts": 1,
                            "resume": "checkpoint", "queue_ttl_minutes": 2880, "label": None,
                            "idempotency_key": "m1-rwkv7-lora-<sha12>",
                            "depends_on": {"job": "<stage A id>", "require": "result.ok == true"},
                            "_marker": "RWKV7_LORA_DONE"},
                        ['idempotency_key "m1-rwkv7-lora-', 'env {"JOB": "lora"}', "max_duration_minutes 180", "max_restarts 1", 'resume "checkpoint"',
                         "queue_ttl_minutes 2880", "/output/RWKV7_LORA_DONE"]),
}


@pytest.mark.parametrize("rel", lint.params_files(ROOT))
def test_submit_args_equal_the_job_spec_field_by_field(rel):
    """PR #93 B1: every field launch.py emits equals the JOB_SPEC's submit block (and the block has no field launch.py
    drops). Only label (cosmetic), context_id and the idempotency suffix (always the context sha256) are exempt."""
    job, mode = lint.job_of(rel), lint.mode_of(rel)
    p = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    if "unsupported" in (p.get("context") or launch.LEGACY_CONTEXT[job]):
        return
    # this file's own mode, plus legacy modes that run on the base params (probe9b lean: no job_params_lean.json)
    keys = [k for k in launch.LEGACY_SUBMIT if k[0] == job and (
        k[1] == mode or (mode is None and k[1] and not (ROOT / f"jobs/{job}/job_params_{k[1]}.json").exists()))]
    assert keys, f"{rel}: no LEGACY_SUBMIT entry"
    text = (ROOT / f"jobs/{job}/JOB_SPEC.md").read_text(encoding="utf-8")
    for key in keys:
        idx, over, evidence = SPEC_BLOCK[key]
        for e in evidence:
            assert e in text, f"{key}: override evidence {e!r} not in JOB_SPEC"
        want = dict(spec_blocks(job)[idx])
        over = dict(over)
        marker = over.pop("_marker", None)
        want.update(over)
        if marker:
            want["checks"] = json.loads(json.dumps(want["checks"]).replace(
                want["checks"]["success"]["check"]["all_of"][0]["path"], f"/output/{marker}"))
        got = launch.submit_args(job, key[1], p, launch.LEGACY_SUBMIT[key], "c" * 64)
        assert want.pop("idempotency_key").split("<")[0] == got["idempotency_key"][:-12], key
        want.pop("context_id", None)
        label = want.pop("label", None)
        if label:
            assert got["label"] == label, key
        for k, v in want.items():
            assert got.get(k) == v, f"{key}: {k}: launch {got.get(k)!r} != JOB_SPEC {v!r}"
        extra = set(got) - set(want) - {"label", "idempotency_key", "context_id"}
        assert extra <= {"env"} and (got["env"] == {} or "env" in want), f"{key}: fields not in JOB_SPEC: {extra}"


def test_spec_parser_reads_both_block_styles():
    assert spec_blocks("qual")[0]["hf_cache"] is True  # job-4j3vu ran with hf_cache true
    assert spec_blocks("fla_rwkv7")[1]["command"].endswith("truth_v2")
    assert spec_blocks("rwkv7")[0]["env"] == {"JOB": "readout"}


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
