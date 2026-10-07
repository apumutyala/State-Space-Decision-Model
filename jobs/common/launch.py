"""One command from a merged SHA to the exact submit_job arguments (CPU only; never submits anything).

    python -m jobs.common.launch <job> [--mode M] --sha <merged sha> --out <dir> [--data <dir>] [--allow-waived]

Steps (each refuses with a non-zero exit; jobs/LAUNCH_CHECKLIST.md):
  1. `git fetch origin agent-main`; the SHA must be an ancestor of origin/agent-main.
  2. `git -c core.autocrlf=false archive <sha>` -> code/m1 (in memory, plus a staged copy under <out>/stage).
  3. The sha's own jobs/common/lint.py on the staged tree (waived failures only with --allow-waived, recorded in submit.json).
  4. The job's CPU tests on the staged tree (every tests/ file naming jobs/<job>/ or jobs.<job>, + test_jobs_lint).
  5. Context = Dockerfile (+ CMD rewrite) + code/m1 + pinned data (sha256 checked against the pin file in the
     archive) + CONTEXT_MANIFEST.json, as a deterministic tar (sorted, mtime 2026-01-01, uid/gid 0, modes 644/755)
     compressed with Python zstandard (no zstd binary). Shipped *.sh/*.sha256/*.json/*.py/Dockerfile must have no CR.
  6. <out>/<name>.tar.zst, <out>/CONTEXT_MANIFEST.json and <out>/submit.json: create_context / upload /
     finalize_context / open_mission / submit_job arguments (idempotency key from the context sha256).

The context and submit fields come from job_params ("submit" and "context" blocks) for new jobs, or from the tables
below, transcribed from each legacy JOB_SPEC's platform call and build_context.sh. CONTEXT_RECIPES supplies the context
for jobs whose params defer to a build_context.sh (Clef: Dockerfile rewrite, literal / JSON-key data pins, REF_FLASH).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import importlib.util

from jobs.common import lint  # only for the caller-side helpers; every check on a sha runs the sha's own lint.py

MTIME = 1767225600  # 2026-01-01T00:00:00Z, as `tar --mtime=2026-01-01` in the build_context.sh scripts
LF_SHIPPED = (".sh", ".sha256", ".json", ".py")
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
    ("cell_step_profile", None): {"command": "bash /opt/ssdm/m1/jobs/cell_step_profile/cell_step_job.sh", "env": {},
                                  "key": "m2-cell-step-profile", "clock_lock": True,
                                  "checks": _checks("CELL_STEP_PROFILE_DONE", 300, 600)},
    ("fla_rwkv7", None): {"command": "bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh", "env": {}, "key": "m2-fla-rwkv7-gate",
                          "checks": _checks("FLA_RWKV7_DONE", 300, 600)},
    ("fla_rwkv7", "truth_v2"): {"command": "bash /opt/ssdm/m1/jobs/fla_rwkv7/fla_job.sh truth_v2", "env": {},
                                "key": "m2-fla-rwkv7-truth-v2", "checks": _checks("FLA_RWKV7_TRUTH_V2_DONE", 300, 600)},
    ("probe9b", None): {"command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "full"},
                        "key": "m1-probe9b-full", "clock_lock": True, "checks": _checks("PROBE_DONE"),
                        "label": "probe9b: one-layer Q9-AF conversion probe (full, job-4j3vu rates)"},
    ("probe9b", "lean"): {"command": "bash /opt/ssdm/m1/jobs/probe9b/probe_job.sh", "env": {"PROBE_CONFIG": "lean"},
                          "key": "m1-probe9b-lean", "clock_lock": True, "checks": _checks("PROBE_DONE"),
                          "label": "probe9b: one-layer Q9-AF conversion probe (lean, job-4j3vu rates)"},
    ("qual", None): {"command": "bash /opt/ssdm/m1/jobs/qual/qual_job.sh", "env": {}, "key": "m1-hw-qual",
                     "mission": "m1-hw-qual-h100", "chip_count": 1, "resume": "none", "hf_cache": True,
                     "checks": _checks("QUAL_PASS", 600, 900)},
    ("qual", "c_only"): {"command": "bash /opt/ssdm/m1/jobs/qual/qual_job.sh", "env": {"QUAL_MODE": "c_only"},
                         "key": "m1-hw-qual-c-only", "mission": "m1-hw-qual-h100", "chip_count": 1, "resume": "none",
                         "hf_cache": True, "checks": _checks("QUAL_PASS", 600, 900)},
    ("rwkv7", "readout"): {"command": "bash /opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh", "env": {"JOB": "readout"},
                           "key": "m1-rwkv7-readout", "clock_lock": True, "checks": _checks("RWKV7_READOUT_DONE"),
                           "label": "rwkv7 stage A: readout + steer smoke"},
    ("rwkv7", "lora"): {"command": "bash /opt/ssdm/m1/jobs/rwkv7/rwkv7_job.sh", "env": {"JOB": "lora"},
                        "key": "m1-rwkv7-lora", "clock_lock": True, "checks": _checks("RWKV7_LORA_DONE"),
                        "label": "rwkv7 stage B: LoRA pilot (cache fork)",
                        "depends_on": {"job": "<stage A id>", "require": "result.ok == true"}},
}
#: submit_job fields that must be stated explicitly (job_params, else the submit block); no silent defaults (PR #93 B1).
REQUIRED_FIELDS = ("chip", "chip_count", "mission", "build_timeout_minutes", "max_duration_minutes", "max_restarts",
                   "resume", "queue_ttl_minutes", "hf_cache", "model")
REQUIRED_SUBMIT = ("command", "env", "key", "checks")

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

# ----------------------------------------------------------------------------- Clef baseline (jobs/clef_baseline)
#: The Clef image copies app/ -> /app (not code/m1) and reads /app/CONTEXT_MANIFEST.json, so its Dockerfile is rewritten
#: to take the launcher's layout. Data pins that are not `<sha>  <path>` files in the archive:
#:   pin_sha256  literal sha256 of the shipped bytes (job-sgihj's decide results, art-8y6d8);
#:   pin_json    [archive file, key, ...]: the sha256 the committed pointer / guard pin records, of the UNCOMPRESSED
#:               bytes when "gunzip" is true (the 150k pointer, jobs/m2_pilot/guard_pin.json).
#: src_env names an environment variable that may point at the source file outside --data (REF_FLASH).
_CLEF_IMAGE = {"dockerfile": "jobs/clef_baseline/Dockerfile",
               "rewrites": [["COPY app/ /app/",
                             "COPY code/m1/ /app/\nCOPY CONTEXT_MANIFEST.json /app/CONTEXT_MANIFEST.json"]]}
_CLEF_REF = {"src": "reference/clef_flash.decide.results.jsonl", "src_env": "REF_FLASH",
             "dest": "reference/clef_flash.decide.results.jsonl",
             "pin_sha256": "463f21ff492fc792da4d40ecc501cffab0498d2458715f8effd3270614cbd0aa"}
_PTR = "data/manifests/sample-150000-seed20261006.pointer.json"
_B150 = "m2/build/sample-150000-seed20261006/"
_GPIN = "jobs/m2_pilot/guard_pin.json"
_CLEF_TEACHER_DATA = [
    {"src": _B150 + "records.jsonl.gz", "dest": "m2-150k/records.jsonl.gz", "gunzip": True,
     "pin_json": [_PTR, "records_uncompressed_sha256"]},
    {"src": _B150 + "sample-150000-seed20261006.manifest.jsonl.gz",
     "dest": "m2-150k/sample-150000-seed20261006.manifest.jsonl.gz", "gunzip": True,
     "pin_json": [_PTR, "manifest_uncompressed_sha256"]},
    {"src": _B150 + "sample-150000-seed20261006.content.jsonl.gz",
     "dest": "m2-150k/sample-150000-seed20261006.content.jsonl.gz", "gunzip": True,
     "pin_json": [_PTR, "content_map_uncompressed_sha256"]},
    {"src": "suite-0.2/selected-rows.jsonl.gz", "dest": "suite-0.2/selected-rows.jsonl.gz", "gunzip": True,
     "pin_json": [_GPIN, "suite_files_sha256_uncompressed", "selected-rows.jsonl.gz"]},
    {"src": "suite-0.2/added-rows.jsonl.gz", "dest": "suite-0.2/added-rows.jsonl.gz", "gunzip": True,
     "pin_json": [_GPIN, "suite_files_sha256_uncompressed", "added-rows.jsonl.gz"]},
    {"src": "suite-0.2/excluded-questions.json", "dest": "suite-0.2/excluded-questions.json",
     "pin_json": [_GPIN, "suite_exclusions_sha256"]},
]
_CLEF_INDEX = {**_CLEF_IMAGE, "data": [_DEV, _CLEF_REF],
               "manifest_files": ["jobs/clef_baseline/overlength_rows.json", "jobs/common/dev_subset_m1.json"]}
_CLEF_TEACHER = {**_CLEF_IMAGE, "data": _CLEF_TEACHER_DATA, "manifest_files": ["jobs/clef_baseline/teacher_scope.json",
                                                                              _PTR, _GPIN]}
#: (job, mode) -> context recipe. Takes precedence over a job_params "context" block that says "unsupported" (the
#: Clef params merged in #98 point at build_context.sh, which needs a zstd binary and a hand-built submit.json).
CONTEXT_RECIPES: dict[tuple[str, str | None], dict] = {
    ("clef_baseline", "index_subset"): _CLEF_INDEX,
    ("clef_baseline", "index_full"): _CLEF_INDEX,
    ("clef_baseline", "overlength_flash"): _CLEF_INDEX,
    ("clef_baseline", "teacher_clef_flash"): _CLEF_TEACHER,
    ("clef_baseline", "teacher_clef"): _CLEF_TEACHER,
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
    odd = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        for m in t.getmembers():
            if m.isfile():
                out[m.name] = (t.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
            elif not m.isdir() and m.name != "pax_global_header":
                odd.append(f"{m.name} ({'symlink -> ' + m.linkname if m.issym() else 'type ' + repr(m.type)})")
    if odd:
        raise LaunchError(f"the archive has non-regular members (symlinks are not shipped silently): {odd}")
    return out


def params_rel(files: dict, job: str, mode: str | None) -> str:
    """job_params_<mode>.json for a mode; job_params.json without one. A mode with no params file is accepted only
    when a legacy submit entry defines it on the base params (probe9b lean); anything else is a typo and refused."""
    if mode:
        rel = f"jobs/{job}/job_params_{mode}.json"
        if rel in files:
            return rel
        if (job, mode) not in LEGACY_SUBMIT:
            raise LaunchError(f"unknown --mode {mode!r} for {job}: no {rel} at this sha")
    rel = f"jobs/{job}/job_params.json"
    if rel not in files:
        raise LaunchError(f"no {rel} at this sha" + (f" (pass --mode)" if not mode else ""))
    return rel


def load_lint(code: Path):
    """The lint module of the sha being launched (not the caller's checkout)."""
    f = code / "jobs/common/lint.py"
    if not f.is_file():
        raise LaunchError("this sha predates jobs/common/lint.py: launch a sha that contains it")
    spec = importlib.util.spec_from_file_location(f"_sha_lint_{abs(hash(str(code)))}", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def stage(files: dict, d: Path) -> None:
    for rel, (b, _) in files.items():
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)


def run_lint(code: Path, rel: str, allow_waived: bool, sha_lint=None) -> dict:
    sha_lint = sha_lint or load_lint(code)
    res = sha_lint.lint_params(rel, code)
    bad, waived = sha_lint.failures(rel, res)
    if bad:
        raise LaunchError(f"lint failed for {rel}: {json.dumps(bad, indent=1)}")
    if waived and not allow_waived:
        raise LaunchError(f"lint has WAIVED failures for {rel} (fix them, or pass --allow-waived and say why in the "
                          f"launch receipt): {json.dumps(waived, indent=1)}")
    hits = sha_lint.backslash_n_hits(code)
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


def data_pin(files: dict, item: dict) -> str | None:
    """The sha256 a data item must have: from a `<sha>  <path>` pin file, a literal, or a key of a committed JSON."""
    if item.get("pin"):
        return pins(files[item["pin"]][0].decode()).get(item["dest"])
    if item.get("pin_sha256"):
        return item["pin_sha256"]
    if item.get("pin_json"):
        rel, *keys = item["pin_json"]
        v = json.loads(files[rel][0])
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        return v if isinstance(v, str) else None
    raise LaunchError(f"data {item['dest']}: no pin")


def build_context(files: dict, job: str, sha: str, spec: dict, params_path: str, data_dir: Path | None) -> dict:
    """{path in context: (bytes, mode)}: Dockerfile, code/m1/**, data/**, CONTEXT_MANIFEST.json."""
    if "unsupported" in spec:
        raise LaunchError(f"{job}: {spec['unsupported']}")
    ctx = {f"code/m1/{p}": v for p, v in files.items()}
    df = files[spec["dockerfile"]][0].decode()
    for old, new in ([spec["cmd_rewrite"]] if spec.get("cmd_rewrite") else []) + list(spec.get("rewrites", [])):
        if old not in df:
            raise LaunchError(f"Dockerfile rewrite failed: {old!r} not in {spec['dockerfile']}")
        df = df.replace(old, new)
        if new not in df:
            raise LaunchError("Dockerfile CMD rewrite failed")
    ctx["Dockerfile"] = (df.encode(), 0o644)
    for item in spec.get("data", []):
        if data_dir is None:
            raise LaunchError(f"{job} ships data ({item['dest']}): pass --data <SSDM_DATA dir>")
        src = Path(os.environ[item["src_env"]]) if item.get("src_env") and os.environ.get(item["src_env"]) \
            else data_dir / item["src"]
        if not src.is_file():
            raise LaunchError(f"data {item['dest']}: source {src} missing" +
                              (f" (set {item['src_env']})" if item.get("src_env") else ""))
        b = src.read_bytes()
        want, got, what = data_pin(files, item), sha256(gzip.decompress(b) if item.get("gunzip") else b), \
            item.get("pin") or item.get("pin_json") or "pin_sha256"
        if not want or want != got:
            raise LaunchError(f"data {item['dest']}: sha256 {got} != pin {want} ({what})")
        ctx[f"data/{item['dest']}"] = (b, 0o644)
    if "COPY data/" in df and not any(p.startswith("data/") for p in ctx):
        ctx["data/.keep"] = (b"", 0o644)
    man = {"job": job, "m1_sha": sha, "branch_commit": sha, "builder": "jobs.common.launch", "params": params_path,
           "dockerfile_sha256": sha256(ctx["Dockerfile"][0]),
           "files_sha256": {p: sha256(files[p][0]) for p in sorted({params_path, *spec.get("manifest_files", [])})},
           "data_sha256": {p: sha256(b) for p, (b, _) in sorted(ctx.items()) if p.startswith("data/")}}
    ctx["CONTEXT_MANIFEST.json"] = (json.dumps(man, indent=1).encode() + b"\n", 0o644)
    return ctx


def cr_violations(ctx: dict, root: Path, sha: str, exempt: tuple[str, ...] = lint.CR_EXEMPT_PREFIXES) -> list[str]:
    """Shipped text files that must be LF but carry CR (byte-exact `-text` paths in the archive are exempt)."""
    cand = [p for p, (b, _) in ctx.items() if (p.endswith(LF_SHIPPED) or p.endswith("Dockerfile")) and b"\r" in b
            and not p.removeprefix("code/m1/").startswith(exempt)]
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
    """Every platform field is read from job_params, else from the submit block; a missing one is an error, never a
    default (PR #93 B1: a defaulted hf_cache=false would have re-downloaded 19 GB inside the capped run)."""
    missing = [k for k in REQUIRED_FIELDS if k not in p and k not in sub] + [k for k in REQUIRED_SUBMIT if k not in sub]
    if missing:
        raise LaunchError(f"{job}/{mode}: submit fields not stated (no defaults): {missing}")
    field = {k: (p[k] if k in p else sub[k]) for k in REQUIRED_FIELDS}
    if not field["mission"]:
        raise LaunchError(f"{job}/{mode}: empty mission")
    a = {"chip": field["chip"], "chip_count": field["chip_count"], "mission": field["mission"],
         "idempotency_key": f"{sub['key']}-{ctx_sha[:12]}",
         "label": sub.get("label", f"{job}{'/' + mode if mode else ''} @ ctx {ctx_sha[:12]}"),
         "context_id": "<from create_context>", "dockerfile_path": "Dockerfile",
         "build_timeout_minutes": field["build_timeout_minutes"], "command": sub["command"], "env": sub["env"],
         "max_duration_minutes": field["max_duration_minutes"], "max_restarts": field["max_restarts"],
         "resume": field["resume"], "queue_ttl_minutes": field["queue_ttl_minutes"], "hf_cache": field["hf_cache"],
         "model": field["model"], "checks": sub["checks"]}
    for k in ("clock_lock", "depends_on"):
        if k in sub:
            a[k] = sub[k]
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
    sha_lint = load_lint(code)
    lint_res = run_lint(code, rel, allow_waived, sha_lint)
    tests = run_tests(code, job) if run_job_tests else {"skipped": True}
    sub = p.get("submit") or LEGACY_SUBMIT.get((job, mode))
    spec = p.get("context") or LEGACY_CONTEXT.get(job)
    if (not spec or "unsupported" in spec) and (job, mode) in CONTEXT_RECIPES:
        spec = CONTEXT_RECIPES[(job, mode)]
    if not sub or not spec:
        raise LaunchError(f"{job}/{mode}: job_params needs 'submit' {{command, key, checks}} and 'context' blocks")
    runner = sha_lint.runner_of(code, job, p)
    spec = {**spec, "manifest_files": [*spec.get("manifest_files", []), *([runner] if runner else [])]}
    ctx = build_context(files, job, full, spec, rel, data_dir)
    bad = cr_violations(ctx, root, full, sha_lint.CR_EXEMPT_PREFIXES)
    if bad:
        raise LaunchError(f"CR in shipped LF files (fix .gitattributes / the blob): {bad}")
    blob = compress(deterministic_tar(ctx), level)
    ctx_sha, size = sha256(blob), len(blob)
    name = f"{job}{'-' + mode if mode else ''}-{full[:12]}"
    (out / f"{name}.tar.zst").write_bytes(blob)
    (out / "CONTEXT_MANIFEST.json").write_bytes(ctx["CONTEXT_MANIFEST.json"][0])
    costs = sha_lint.load_costs(code)
    worst = costs.worst_case_usd(p)
    receipt = {
        "job": job, "mode": mode, "sha": full, "params": rel,
        "context": {"file": str(out / f"{name}.tar.zst"), "sha256": ctx_sha, "size_bytes": size},
        "cost": {"worst_case_usd": worst, "hard_cap_usd": p.get("hard_cap_usd"),
                 "set_workspace_cap_increment_usd": math.ceil(worst)},
        "lint": lint_res, "tests": tests,
        "calls": [
            {"tool": "open_mission", "args": {"name": p["mission"] if "mission" in p else sub["mission"]}},
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
