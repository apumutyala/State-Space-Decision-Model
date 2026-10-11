"""Package the cpu_tests job: git archives of the four shas + this directory's job files -> <out>/cpu_tests.tar.zst +
submit.json (the jobs/common/sf_submit.py format). Usage: uv run --no-project --with zstandard python build.py <repo> <out>
"""
import hashlib, io, json, pathlib, subprocess, sys, tarfile

import zstandard

import os
TREES = {"nupi": os.environ.get("NUPI_SHA", "c3f3a3f9"), "base": "5e159437", "g16": "2d913b3e", "sym": "42388c2c"}
SUITES = os.environ.get("SUITES", "nupi g16 sym")
SKIP = ("receipts/", "docs/nupi_toy_traces.svg")  # not imported by any of the suites (third_party/ is: rwkv7_hf)
KEEP = ("docs/nupi_toy_traces.json",)
JOB_FILES = ("run.sh", "golden.py", "patch_golden.py")
MTIME = 1767225600
HERE = pathlib.Path(__file__).resolve().parent


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "core.autocrlf=false", *a], check=True, capture_output=True).stdout


def main():
    repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ctx, shas = {}, {}
    for name, sha in TREES.items():
        full = git(repo, "rev-parse", f"{sha}^{{commit}}").decode().strip()
        shas[name] = full
        with tarfile.open(fileobj=io.BytesIO(git(repo, "archive", full))) as t:
            for m in t.getmembers():
                if m.isfile() and (not m.name.startswith(SKIP) or m.name in KEEP):
                    ctx[f"trees/{name}/{m.name}"] = (t.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
    for f in JOB_FILES:
        b = (HERE / f).read_bytes().replace(b"\r\n", b"\n")
        ctx[f"job/{f}"] = (b, 0o755 if f.endswith(".sh") else 0o644)
    ctx["Dockerfile"] = ((HERE / "Dockerfile").read_bytes().replace(b"\r\n", b"\n"), 0o644)
    man = {"job": "cpu_tests", "trees": shas, "job_files_sha256": {f: hashlib.sha256(ctx[f"job/{f}"][0]).hexdigest()
                                                                 for f in JOB_FILES}}
    ctx["CONTEXT_MANIFEST.json"] = (json.dumps(man, indent=1).encode() + b"\n", 0o644)
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
    blob = zstandard.ZstdCompressor(level=19).compress(buf.getvalue())
    sha = hashlib.sha256(blob).hexdigest()
    f = out / "cpu_tests.tar.zst"
    f.write_bytes(blob)
    (out / "CONTEXT_MANIFEST.json").write_bytes(ctx["CONTEXT_MANIFEST.json"][0])
    checks = {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/CPU_TESTS_DONE"},
                                               {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
              "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"}, "interval_s": 300,
                         "grace_period_s": 900, "fail_threshold": 2, "on_fail": "kill"}}
    sub = {"chip": "cpu-8", "mission": "ssdm-cpu-tests", "idempotency_key": f"ssdm-cpu-tests-{sha[:12]}",
           "label": f"cpu_tests: suites [{SUITES}] nupi {TREES['nupi']} (+5e159437 golden), g16-sym 2d913b3e, sym-invariant 42388c2c",
           "dockerfile_path": "Dockerfile", "build_timeout_minutes": 20, "command": "bash /opt/job/run.sh", "env": {"SUITES": SUITES},
           "max_duration_minutes": 45, "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
           "hf_cache": False, "checks": checks}
    receipt = {"job": "cpu_tests", "trees": shas, "context": {"file": str(f), "sha256": sha, "size_bytes": len(blob)},
               "cost": {"worst_case_usd": 0.41, "basis": "validate_job cpu-8: $0.27 per 30 min -> 45 min"},
               "calls": [{"tool": "open_mission", "args": {"name": "ssdm-cpu-tests"}},
                         {"tool": "submit_job", "args": sub}]}
    (out / "submit.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(sha, len(blob), len(ctx), "files")


main()
