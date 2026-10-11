"""Package the cpu_tests job: git archives of the four shas + this directory's job files -> <out>/cpu_tests.tar.zst +
submit.json (the jobs/common/sf_submit.py format). Usage: uv run --no-project --with zstandard python build.py <repo> <out>
"""
import hashlib, io, json, pathlib, subprocess, sys, tarfile

import zstandard

import os
# env overrides: NUPI_SHA / G16_SHA / SYM_SHA pick the trees, EV_SHA the evidence lane's adversarial suites
# (tests/guarantees/test_*_adversarial.py, overlaid on g16 + sym), SUITES which suites run
TREES = {"nupi": os.environ.get("NUPI_SHA", "c3f3a3f9"), "base": "5e159437",
         "g16": os.environ.get("G16_SHA", "2d913b3e"), "sym": os.environ.get("SYM_SHA", "42388c2c"),
         "ev": os.environ.get("EV_SHA", "1351ed9f"),
         # stack/qwen35/jev2-4b (A-PREREG-JEV2-4B): row-cache bit-identity, LS per qtype, noul band gate, short-budget
         # scheduler (suite "jev2"; JEV2_SHA overrides)
         "jev2": os.environ.get("JEV2_SHA", "a5b4cb33"),
         # stack/qwen35/jev2-h2o (A-PREREG-JEV2-H2O): host pins, lm_label head == the shim readout, tiny host
         # forward/backward (suite "jev2h"; JEV2H_SHA overrides)
         "jev2h": os.environ.get("JEV2H_SHA", "63277ebd")}
# BASELINE="g16old=2d913b3e symold=42388c2c": extra trees for the "baseline" suite (same tests on the pre-fix shas)
TREES.update(dict(x.split("=", 1) for x in os.environ.get("BASELINE", "").split()))
SUITES = os.environ.get("SUITES", "nupi g16 sym")
#: tokenizer files only (no weights) for the real-tokenizer tests: <dir name> under SSDM_TOKENIZERS=/opt/tokenizers
TOK_ROOT = pathlib.Path(os.environ.get("TOK_ROOT", r"C:\Users\apumu\research\CHLabs\ssdm-data\tokenizers"))
TOKENIZERS = ("Qwen--Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",)
TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json",
             "chat_template.jinja", "special_tokens_map.json", "generation_config.json")
SKIP = ("receipts/", "docs/nupi_toy_traces.svg")  # not imported by any of the suites (third_party/ is: rwkv7_hf)
KEEP = ("docs/nupi_toy_traces.json",)
EV_ONLY = ("tests/guarantees/test_sym_fork_adversarial.py", "tests/guarantees/test_g16_sym_adversarial.py")
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
                if name == "ev" and m.name not in EV_ONLY:
                    continue
                if m.isfile() and (not m.name.startswith(SKIP) or m.name in KEEP):
                    ctx[f"trees/{name}/{m.name}"] = (t.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
    toks = {}
    for d in TOKENIZERS:
        for fn in TOK_FILES:
            p = TOK_ROOT / d / fn
            if p.is_file():
                ctx[f"tokenizers/{d}/{fn}"] = (p.read_bytes(), 0o644)
                toks[f"{d}/{fn}"] = hashlib.sha256(ctx[f"tokenizers/{d}/{fn}"][0]).hexdigest()
        assert f"{d}/tokenizer.json" in toks, f"missing {TOK_ROOT / d / 'tokenizer.json'}"
    for f in JOB_FILES:
        b = (HERE / f).read_bytes().replace(b"\r\n", b"\n")
        ctx[f"job/{f}"] = (b, 0o755 if f.endswith(".sh") else 0o644)
    ctx["Dockerfile"] = ((HERE / "Dockerfile").read_bytes().replace(b"\r\n", b"\n"), 0o644)
    man = {"job": "cpu_tests", "trees": shas, "suites": SUITES, "tokenizer_files_sha256": toks,
           "job_files_sha256": {f: hashlib.sha256(ctx[f"job/{f}"][0]).hexdigest() for f in JOB_FILES}}
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
    blob = zstandard.ZstdCompressor(level=int(os.environ.get("ZSTD_LEVEL", "9"))).compress(buf.getvalue())  # 9: low RAM
    sha = hashlib.sha256(blob).hexdigest()
    f = out / "cpu_tests.tar.zst"
    f.write_bytes(blob)
    (out / "CONTEXT_MANIFEST.json").write_bytes(ctx["CONTEXT_MANIFEST.json"][0])
    checks = {"success": {"check": {"all_of": [{"type": "file_exists", "path": "/output/CPU_TESTS_DONE"},
                                               {"type": "file_exists", "path": "/output/MANIFEST.jsonl"}]}},
              "health": {"check": {"type": "heartbeat", "path": "/output/heartbeat"}, "interval_s": 300,
                         "grace_period_s": 900, "fail_threshold": 2, "on_fail": "kill"}}
    sub = {"chip": "cpu-8", "mission": "ssdm-cpu-tests", "idempotency_key": f"ssdm-cpu-tests-{sha[:12]}",
           "label": f"cpu_tests: suites [{SUITES}] nupi {TREES['nupi']} (+5e159437 golden), g16-sym {TREES['g16']}, "
                    f"sym-invariant {TREES['sym']}, ev {TREES['ev']}, jev2 {TREES['jev2']}, jev2h {TREES['jev2h']}",
           "dockerfile_path": "Dockerfile", "build_timeout_minutes": 20, "command": "bash /opt/job/run.sh", "env": {"SUITES": SUITES},
           "max_duration_minutes": int(os.environ.get("MAX_MIN", "75")), "max_restarts": 0, "resume": "none", "queue_ttl_minutes": 1440,
           "hf_cache": False, "checks": checks}
    receipt = {"job": "cpu_tests", "trees": shas, "context": {"file": str(f), "sha256": sha, "size_bytes": len(blob)},
               "cost": {"worst_case_usd": round(0.27 * int(os.environ.get("MAX_MIN", "75")) / 30, 2),
                        "basis": "validate_job cpu-8: $0.27 per 30 min x max_duration_minutes"},
               "calls": [{"tool": "open_mission", "args": {"name": "ssdm-cpu-tests"}},
                         {"tool": "submit_job", "args": sub}]}
    (out / "submit.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(sha, len(blob), len(ctx), "files")


main()
