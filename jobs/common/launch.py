"""One command from a merged SHA to the exact submit_job arguments (CPU only; never submits anything).

    python -m jobs.common.launch <job> [--mode M] --sha <merged sha> --out <dir> [--data <dir>] [--allow-waived]

Steps (each refuses with a non-zero exit; jobs/LAUNCH_CHECKLIST.md):
  1. `git fetch origin agent-main`; the SHA must be an ancestor of origin/agent-main.
  2. `git -c core.autocrlf=false archive <sha>` -> code/m1 (in memory, plus a staged copy under <out>/stage).
  3. jobs.common.lint on the staged tree (waived failures only with --allow-waived, recorded in submit.json).
  4. The job's CPU tests on the staged tree (every tests/ file naming jobs/<job>/ or jobs.<job>, + test_jobs_lint).
  5. Context = Dockerfile (+ CMD rewrite) + code/m1 + pinned data (sha256 checked against the pin file in the
     archive) + CONTEXT_MANIFEST.json, as a deterministic tar (sorted, mtime 2026-01-01, uid/gid 0, modes 644/755)
     compressed with Python zstandard (no zstd binary). Shipped *.sh/*.sha256/*.json/Dockerfile must have no CR.
  6. <out>/<name>.tar.zst, <out>/CONTEXT_MANIFEST.json and <out>/submit.json: create_context / upload /
     finalize_context / open_mission / submit_job arguments (idempotency key from the context sha256).

The context and submit fields come from job_params ("submit" and "context" blocks) for new jobs, or from the tables
below, transcribed from each legacy JOB_SPEC's platform call and build_context.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from jobs.common import lint

MTIME = 1767225600  # 2026-01-01T00:00:00Z, as `tar --mtime=2026-01-01` in the build_context.sh scripts
LF_SHIPPED = (".sh", ".sha256", ".json")
_HB = {"type": "heartbeat", "path": "/output/heartbeat"}


def _checks(marker: str, interval: int = 600, grace: int = 1800) -> dict:
    return {"success": {"check": {"all_of": [{"type": "file_exists", "path": f"/output/{marker}"},
                                             {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
            "health": {"check": dict(_HB), "interval_s": interval, "grace_period_s": grace, "fail_threshold": 2,
                       "on_fail": "kill"}}


_DEV = {"src": "slices/dev-rows.jsonl.gz", "dest": "slices/dev-rows.jsonl.gz", "pin": "jobs/clef_baseline/data.sha256"}
_REC = {"src": "m2/build/sample-20000-seed20261006/records.jsonl.gz", "dest": "m2/records.jsonl.gz",
        "pin": "jobs/rwkv7/records.sha256"}
_RWKV7_IMAGE = {"dockerfile": "jobs/rwkv7/Dockerfile"}

#: (job, mode) -> submit fields not in job_params. Transcribed from the JOB_SPEC "submit_job" blocks (2026-10-06).
LEGACY_SUBMIT: dict[tuple[str, str | None], dict] = {
    ("cell_step_profile", None): {"command": "bash /opt/ssdm/m1/jobs/cell_step_profile/cell_step_job.sh",
                                  "key": "m2-cell-step-profile", "checks": _checks("CELL_STEP_PROFILE_DONE", 300, 600)},
    ("fla_rwkv7", None): {"command": "bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh", "key": "m2-fla-rwkv7-gate",
                          "checks": _checks("FLA_RWKV7_DONE", 300, 600)},
    ("fla_rwkv7", "truth_v2"): {"command": "bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh truth_v2",
                                "key": "m2-fla-rwkv7-truth-v2", "checks": _checks("FLA_RWKV7_TRUTH_V2_DONE", 300, 600)},
    ("probe9b", None): {"command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "full"},
                        "key": "m1-probe9b-full", "clock_lock": True, "checks": _checks("PROBE_DONE")},
    ("probe9b", "lean"): {"command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "lean"},
                          "key": "m1-probe9b-lean", "clock_lock": True, "checks": _checks("PROBE_DONE")},
    ("qual", None): {"command": "bash /opt/ssdm/m1/jobs/qual/qual_job.sh", "key": "m1-hw-qual", "mission": "m1-hw-qual-h100",
                     "checks": _checks("QUAL_PASS", 600, 900)},
    ("qual", "c_only"): {"command": "bash /opt/ssdm/m1/jobs/qual/qual_job.sh", "env": {"QUAL_MODE": "c_only"},
                         "key": "m1-hw-qual-c-only", "mission": "m1-hw-qual-h100", "checks": _checks("QUAL_PASS", 600, 900)},
    ("rwkv7", "readout"): {"command": "bash /opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh", "env": {"JOB": "readout"},
                           "key": "m1-rwkv7-readout", "clock_lock": True, "checks": _checks("RWKV7_READOUT_DONE")},
    ("rwkv7", "lora"): {"command": "bash /opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh", "env": {"JOB": "lora"},
                        "key": "m1-rwkv7-lora", "clock_lock": True, "checks": _checks("RWKV7_LORA_DONE"),
                        "depends_on": {"job": "<stage A job id>", "require": "result.ok == true"}},
}

#: job -> context recipe, transcribed from jobs/<job>/build_context.sh and jobs/common/build_context.sh.
LEGACY_CONTEXT: dict[str, dict] = {
    "cell_step_profile": {**_RWKV7_IMAGE, "cmd_rewrite": ["jobs/rwkv7/rwkv7_job.sh",
                                                          "jobs/cell_step_profile/cell_step_job.sh"], "data": []},
    "fla_rwkv7": {**_RWKV7_IMAGE, "cmd_rewrite": ["jobs/rwkv7/rwkv7_job.sh", "jobs/fla_rwkv7/fla_job.sh"], "data": []},
    "probe9b": {"dockerfile": "jobs/probe9b/Dockerfile", "data": [_DEV]},
    "rwkv7": {**_RWKV7_IMAGE, "data": [_DEV, _REC]},
    "qual": {"dockerfile": "jobs/qual/Dockerfile", "data": []},
    "m2_pilot": {"unsupported": "needs the parent-arm cache + guard_ready dry run: use jobs/m2_pilot/build_context.sh"},
    "m2_parent_arm": {"unsupported": "runs guard_ready on the shipped suite: use jobs/m2_parent_arm/build_context.sh"},
}


class LaunchError(RuntimeError):
    pass


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    r = subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=false", *args], capture_output=True)
    if r.returncode:
        raise LaunchError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode().strip()


def resolve_merged(root: Path, sha: str, fetch: bool = True) -> str:
    if fetch:
        git(root, "fetch", "-q", "origin", "agent-main")
    full = git(root, "rev-parse", "--verify", f"{sha}^{{commit}}")
    r = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", full, "origin/agent-main"])
    if r.returncode != 0:
        raise LaunchError(f"{sha} is not merged into origin/agent-main (merge the PR first; launch only merged code)")
    return full


def archive(root: Path, sha: str) -> dict[str, tuple[bytes, int]]:
    """`git -c core.autocrlf=false archive <sha>` as {path: (bytes, mode)}; modes normalised to 644/755."""
    raw = git(root, "archive", sha, binary=True)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = (t.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
    return out


def params_rel(files: dict, job: str, mode: str | None) -> str:
    for rel in ([f"jobs/{job}/job_params_{mode}.json"] if mode else []) + [f"jobs/{job}/job_params.json"]:
        if rel in files:
            return rel
    raise LaunchError(f"no job_params for {job} mode {mode} at this sha")


def stage(files: dict, d: Path) -> None:
    for rel, (b, _) in files.items():
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)


def run_lint(code: Path, rel: str, allow_waived: bool) -> dict:
    res = lint.lint_params(rel, code)
    bad, waived = lint.failures(rel, res)
    if bad:
        raise LaunchError(f"lint failed for {rel}: {json.dumps(bad, indent=1)}")
    if waived and not allow_waived:
        raise LaunchError(f"lint has WAIVED failures for {rel} (fix them, or pass --allow-waived and say why in the "
                          f"launch receipt): {json.dumps(waived, indent=1)}")
    hits = lint.backslash_n_hits(code)
    if hits:
        raise LaunchError(f"literal backslash-n continuation in shipped shell: {hits}")
    return {"waived": waived}


def job_tests(code: Path, job: str) -> list[str]:
    out = []
    for p in sorted((code / "tests").rglob("test_*.py")):
        s = p.read_text(encoding="utf-8", errors="replace")
        if f"jobs/{job}/" in s or f"jobs.{job}" in s or p.name == "test_jobs_lint.py":
            out.append(p.relative_to(code).as_posix())
    return out


def run_tests(code: Path, job: str) -> dict:
    tests = job_tests(code, job)
    if not tests:
        raise LaunchError(f"no CPU tests reference jobs/{job}/")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
    cmd = [sys.executable, "-X", "utf8", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    r = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    if r.returncode:
        raise LaunchError(f"job tests failed ({' '.join(tests)}):\n" + "\n".join(tail))
    return {"files": tests, "summary": tail[-1] if tail else ""}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pins(text: str) -> dict[str, str]:
    """`<sha>  <path>` lines -> {path relative to /data/: sha}."""
    out = {}
    for ln in text.splitlines():
        if ln.strip():
            h, p = ln.split(None, 1)
            out[p.strip().removeprefix("/data/").lstrip("*")] = h
    return out


def build_context(files: dict, job: str, sha: str, spec: dict, params_path: str, data_dir: Path | None) -> dict:
    """{path in context: (bytes, mode)}: Dockerfile, code/m1/**, data/**, CONTEXT_MANIFEST.json."""
    if "unsupported" in spec:
        raise LaunchError(f"{job}: {spec['unsupported']}")
    ctx = {f"code/m1/{p}": v for p, v in files.items()}
    df = files[spec["dockerfile"]][0].decode()
    if spec.get("cmd_rewrite"):
        old, new = spec["cmd_rewrite"]
        df = df.replace(old, new)
        if new not in df:
            raise LaunchError("Dockerfile CMD rewrite failed")
    ctx["Dockerfile"] = (df.encode(), 0o644)
    for item in spec.get("data", []):
        if data_dir is None:
            raise LaunchError(f"{job} ships data ({item['dest']}): pass --data <SSDM_DATA dir>")
        b = (data_dir / item["src"]).read_bytes()
        want = pins(files[item["pin"]][0].decode()).get(item["dest"])
        if want != sha256(b):
            raise LaunchError(f"data {item['dest']}: sha256 {sha256(b)} != pin {want} ({item['pin']})")
        ctx[f"data/{item['dest']}"] = (b, 0o644)
    if "COPY data/" in df and not any(p.startswith("data/") for p in ctx):
        ctx["data/.keep"] = (b"", 0o644)
    man = {"job": job, "m1_sha": sha, "builder": "jobs.common.launch", "params": params_path,
           "dockerfile_sha256": sha256(ctx["Dockerfile"][0]),
           "files_sha256": {p: sha256(files[p][0]) for p in sorted({params_path, *spec.get("manifest_files", [])})},
           "data_sha256": {p: sha256(b) for p, (b, _) in sorted(ctx.items()) if p.startswith("data/")}}
    ctx["CONTEXT_MANIFEST.json"] = (json.dumps(man, indent=1).encode() + b"\n", 0o644)
    return ctx


def cr_violations(ctx: dict, root: Path, sha: str) -> list[str]:
    """Shipped text files that must be LF but carry CR (byte-exact `-text` paths in the archive are exempt)."""
    cand = [p for p, (b, _) in ctx.items() if (p.endswith(LF_SHIPPED) or p.endswith("Dockerfile")) and b"\r" in b
            and not p.removeprefix("code/m1/").startswith(lint.CR_EXEMPT_PREFIXES)]
    if not cand:
        return []
    code = [p.removeprefix("code/m1/") for p in cand if p.startswith("code/m1/")]
    unset = set()
    if code:
        out = git(root, "check-attr", "--source", sha, "text", "--", *code)
        unset = {"code/m1/" + ln.split(": text:")[0] for ln in out.splitlines() if ln.endswith("unset")}
    return [p for p in cand if p not in unset]


def deterministic_tar(ctx: dict) -> bytes:
    dirs = set()
    for p in ctx:
        parts = p.split("/")[:-1]
        dirs.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as t:
        for name in sorted(dirs | set(ctx)):
            ti = tarfile.TarInfo(name)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = MTIME, 0, 0, "", ""
            if name in ctx:
                b, mode = ctx[name]
                ti.size, ti.mode = len(b), mode
                t.addfile(ti, io.BytesIO(b))
            else:
                ti.type, ti.mode = tarfile.DIRTYPE, 0o755
                t.addfile(ti)
    return buf.getvalue()


def compress(b: bytes, level: int) -> bytes:
    import zstandard

    return zstandard.ZstdCompressor(level=level).compress(b)


def submit_args(job: str, mode: str | None, p: dict, sub: dict, ctx_sha: str) -> dict:
    a = {"chip": p["chip"], "chip_count": p.get("chip_count", 1), "mission": p.get("mission") or sub.get("mission"),
         "idempotency_key": f"{sub['key']}-{ctx_sha[:12]}",
         "label": sub.get("label", f"{job}{'/' + mode if mode else ''} @ ctx {ctx_sha[:12]}"),
         "context_id": "<from create_context>", "dockerfile_path": "Dockerfile",
         "build_timeout_minutes": p["build_timeout_minutes"], "command": sub["command"], "env": sub.get("env", {}),
         "max_duration_minutes": p["max_duration_minutes"], "max_restarts": p["max_restarts"],
         "resume": p.get("resume", "none"), "queue_ttl_minutes": p.get("queue_ttl_minutes", 1440),
         "hf_cache": p.get("hf_cache", False), "checks": sub["checks"]}
    if p.get("model"):
        a["model"] = p["model"]
    for k in ("clock_lock", "depends_on"):
        if k in sub:
            a[k] = sub[k]
    if not a["mission"]:
        raise LaunchError("no mission (job_params 'mission' or submit.mission)")
    return a


def launch(job: str, sha: str, out: Path, mode: str | None = None, root: Path | None = None, fetch: bool = True,
           data_dir: Path | None = None, allow_waived: bool = False, run_job_tests: bool = True, level: int = 19) -> dict:
    root = Path(root or git(Path.cwd(), "rev-parse", "--show-toplevel"))
    full = resolve_merged(root, sha, fetch)
    files = archive(root, full)
    rel = params_rel(files, job, mode)
    p = json.loads(files[rel][0])
    out.mkdir(parents=True, exist_ok=True)
    code = out / "stage" / full[:12]
    if code.exists():
        raise LaunchError(f"{code} exists: use a fresh --out so stale files cannot be linted or tested")
    stage(files, code)
    lint_res = run_lint(code, rel, allow_waived)
    tests = run_tests(code, job) if run_job_tests else {"skipped": True}
    sub = p.get("submit") or LEGACY_SUBMIT.get((job, mode))
    spec = p.get("context") or LEGACY_CONTEXT.get(job)
    if not sub or not spec:
        raise LaunchError(f"{job}/{mode}: job_params needs 'submit' {{command, key, checks}} and 'context' blocks")
    runner = lint.runner_of(code, job, p)
    spec = {**spec, "manifest_files": [*spec.get("manifest_files", []), *([runner] if runner else [])]}
    ctx = build_context(files, job, full, spec, rel, data_dir)
    bad = cr_violations(ctx, root, full)
    if bad:
        raise LaunchError(f"CR in shipped LF files (fix .gitattributes / the blob): {bad}")
    blob = compress(deterministic_tar(ctx), level)
    ctx_sha, size = sha256(blob), len(blob)
    name = f"{job}{'-' + mode if mode else ''}-{full[:12]}"
    (out / f"{name}.tar.zst").write_bytes(blob)
    (out / "CONTEXT_MANIFEST.json").write_bytes(ctx["CONTEXT_MANIFEST.json"][0])
    costs = lint.load_costs(code)
    worst = costs.worst_case_usd(p)
    receipt = {
        "job": job, "mode": mode, "sha": full, "params": rel,
        "context": {"file": str(out / f"{name}.tar.zst"), "sha256": ctx_sha, "size_bytes": size},
        "cost": {"worst_case_usd": worst, "hard_cap_usd": p.get("hard_cap_usd"),
                 "set_workspace_cap_increment_usd": math.ceil(worst)},
        "lint": lint_res, "tests": tests,
        "calls": [
            {"tool": "open_mission", "args": {"name": p.get("mission") or sub.get("mission")}},
            {"tool": "create_context", "args": {"sha256": ctx_sha, "size_bytes": size}},
            {"tool": "upload", "args": {"file": str(out / f"{name}.tar.zst"), "how": "curl -T <file> '<put url>'"}},
            {"tool": "finalize_context", "args": {"context_id": "<from create_context>"}},
            {"tool": "validate_job", "args": "submit_job args minus context_id/idempotency_key/mission; must pass "
                                             "the cost-cap preflight without force"},
            {"tool": "submit_job", "args": submit_args(job, mode, p, sub, ctx_sha)}],
    }
    (out / "submit.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.common.launch", description=__doc__.split("\n")[0])
    ap.add_argument("job")
    ap.add_argument("--mode")
    ap.add_argument("--sha", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("SSDM_DATA",
                                                                     Path.home() / "research/CHLabs/ssdm-data")))
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--no-fetch", action="store_true", help="tests only: trust the local origin/agent-main ref")
    ap.add_argument("--allow-waived", action="store_true")
    ap.add_argument("--level", type=int, default=19)
    a = ap.parse_args(argv)
    try:
        r = launch(a.job, a.sha, a.out, a.mode, a.repo, not a.no_fetch, a.data, a.allow_waived, True, a.level)
    except LaunchError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    c = r["context"]
    print(f"{c['sha256']}  {c['file']}\nsize_bytes {c['size_bytes']}\nsubmit.json -> {a.out / 'submit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
