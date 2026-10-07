"""EXECUTES jobs/m2_first_cell/first_cell_job.sh itself under bash on CPU, end to end, on a tiny random RWKV7.

tests/test_m2_first_cell.py's dry run calls the Python drivers directly and never runs the shell runner, so a broken
runner line (de49b2f: a literal backslash-n in the reprice command -> exit 5 right after ~2 h of the paid parent arm)
passed it. Here bash runs the real script: every phase, every argument it builds, every exit-code mapping, the export
manifest and finalize. Only what a CPU laptop cannot provide is stood in, from OUTSIDE the script:
  * paths: the script's documented M2FC_* overrides (code = this checkout, a data dir of hard links to the private
    local inputs, the kit clone, a scratch dir, the dry run's cell config) + GMN_* + HF_HUB_CACHE / HF_HUB_OFFLINE
    (an HF cache holding the tiny model under the pinned repo + revision, so the real snapshot_download line runs);
  * PATH stubs: nvidia-smi; sha256sum (the snapshot pin lists the 13.3B shards, so ``-c .../snapshot.sha256`` checks
    a manifest of the tiny snapshot's real hashes instead, and data.sha256's absolute ``/data/`` paths are mapped to
    the test data dir; every other call is the real sha256sum);
  * PYTHONPATH: the pinned kit clone (the image pip-installs it) and tests/shell_stubs/sitecustomize.py (the CPU dry
    run's shim, --device cpu, the env check's CUDA answer, the DEV subset / cohort cut, rate-guard time compression,
    and the two negative-test slow-box injections).
Skipped without a POSIX bash (Git Bash on Windows), torch, or the private local inputs (like the dry run).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs/m2_first_cell/first_cell_job.sh"
STUBS = ROOT / "tests/shell_stubs"
REPO_ID, REV = "RWKV/RWKV7-G1k-13.3B-20260930", "ce3630b854cedbd673569f4600c6cf744bca9dba"
PHASES = ["setup", "guard", "heldout", "parent_g12", "parent_index", "reprice", "cell", "evals", "report"]
EXPORTS = ["setup", "guard", "heldout", "parent_g12", "parent_index", "cell", "evals", "final"]
#: run.log lines (lib.sh ``say``) that open each phase, in the runner's order
SAYS = {"setup": "] setup", "guard": "] phase guard", "heldout": "] phase heldout", "parent_g12": "] phase parent_g12",
        "parent_index": "] phase parent_index", "reprice": "] re-price the child evals", "cell": "] phase cell",
        "evals": "] phase evals", "report": "] phase report", "finalize": "] EXIT "}

pytestmark = pytest.mark.shell_runner


def _bash() -> str | None:
    """A POSIX bash with coreutils (Git Bash on Windows; never WSL's System32 / WindowsApps launcher)."""
    cands = []
    w = shutil.which("bash")
    # Git for Windows' bin\bash.exe is a launcher that puts /usr/bin FIRST on PATH, which shadows this test's PATH stubs
    # (nvidia-smi, sha256sum) with the real tools; usr\bin\bash.exe keeps the caller's PATH order
    if w and not (os.name == "nt" and re.search(r"system32|windowsapps|git[\\/]bin[\\/]bash", w, re.I)):
        cands.append(w)
    if os.name == "nt":
        cands += [r"C:\Program Files\Git\usr\bin\bash.exe"]
    for c in cands:
        if Path(c).exists():
            r = subprocess.run([c, "-c", "command -v sha256sum tar timeout stat"], capture_output=True, text=True)
            if r.returncode == 0:
                return c
    return None


def _posix(p) -> str:
    """Path as bash sees it (Git Bash: C:\\x\\y -> /c/x/y; elsewhere unchanged)."""
    s = str(p)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    return f"/{m.group(1).lower()}/{m.group(2)}".replace("\\", "/") if (os.name == "nt" and m) else s


def _link_tree(src: Path, dst: Path, names=None):
    dst.mkdir(parents=True, exist_ok=True)
    for f in (names or [p.name for p in src.iterdir() if p.is_file()]):
        try:
            os.link(src / f, dst / f)
        except OSError:
            shutil.copy2(src / f, dst / f)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    bash = _bash()
    if not bash:
        pytest.skip("no POSIX bash with coreutils (Git Bash on Windows)")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("huggingface_hub")
    from scripts import m2_first_cell_dryrun as D

    if D.missing_inputs():
        pytest.skip(f"local private inputs absent: {D.missing_inputs()}")
    from scripts.m2_e2e_dryrun import ensure_kit

    try:
        kit = Path(ensure_kit()["kit_path"]).parent
    except ImportError:
        pytest.skip("Decision Index kit not importable")
    t = tmp_path_factory.mktemp("m2fc-shell")
    data = t / "data"  # the build context's data/ layout (jobs/m2_first_cell/build_context.sh)
    _link_tree(D.RECORDS.parent, data / "m2", ["records.jsonl.gz"])
    _link_tree(D.TRAIN_DIR, data / "m2-150k", ["records.jsonl.gz", "sample-150000-seed20261006.manifest.jsonl.gz",
                                              "sample-150000-seed20261006.content.jsonl.gz"])
    _link_tree(D.SUITE, data / "suite-0.2", ["selected-rows.jsonl.gz", "added-rows.jsonl.gz", "excluded-questions.json"])
    _link_tree(D.DEV_ROWS.parent, data / "slices", ["dev-rows.jsonl.gz"])
    (t / "CONTEXT_MANIFEST.json").write_text(json.dumps({"job": "m2_first_cell", "test": "shell runner, CPU"}))
    hf = t / "hf"
    store = hf / ("models--" + REPO_ID.replace("/", "--"))
    snap = D.tiny_snapshot(store / "snapshots" / REV)
    try:  # huggingface_hub >= 1.33 resolves a commit revision through its cached tree listing even offline
        from huggingface_hub._tree_cache import TreeCacheEntry, write_tree_cache
    except ImportError:  # older releases return the cached snapshot folder of a commit revision directly
        pass
    else:
        write_tree_cache(str(store), REV, {p.name: TreeCacheEntry(size=p.stat().st_size, blob_id=hashlib.sha1(
            b"blob %d\0" % p.stat().st_size + p.read_bytes()).hexdigest()) for p in snap.iterdir() if p.is_file()})
    (t / "tiny-snapshot.sha256").write_text("".join(  # LF: a CRLF manifest fails sha256sum -c (see the LF test)
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(snap.iterdir()) if p.is_file()),
        newline="\n")
    cell = t / "cell_dryrun.json"  # the dry run's config: the job's cell + its CPU-time OVERRIDES
    cell.write_text(json.dumps({**json.loads(D.CELL.read_text(encoding="utf-8")), **D.OVERRIDES}, indent=1))
    real = subprocess.run([bash, "-c", "command -v sha256sum"], capture_output=True, text=True).stdout.strip()
    bin_ = t / "bin"
    bin_.mkdir()
    (bin_ / "nvidia-smi").write_text("#!/usr/bin/env bash\necho 'nvidia-smi: CPU test stand-in (no GPU)'\n",
                                     newline="\n")
    (bin_ / "sha256sum").write_text(
        "#!/usr/bin/env bash\n"
        "# test stand-in: the snapshot pin -> the tiny snapshot's manifest; data.sha256's /data/ -> the test data dir\n"
        f"REAL='{real}'\n"
        'if [ "$#" -eq 2 ] && [ "$1" = "-c" ]; then\n'
        '  case "$2" in */jobs/rwkv7/snapshot.sha256) exec "$REAL" -c "$M2FC_STUB_SNAPSHOT_SHA256" ;; esac\n'
        '  if grep -q "  /data/" "$2"; then\n'
        '    m=$(mktemp); sed "s#  /data/#  $M2FC_DATA/#" "$2" > "$m"; "$REAL" -c "$m"; r=$?; rm -f "$m"; exit $r\n'
        "  fi\nfi\n"
        'exec "$REAL" "$@"\n', newline="\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("M2FC_", "GMN_", "DRYRUN_"))}
    sep = ";" if os.name == "nt" else ":"
    env.update({
        "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1", "TOK_S": "3757", "CALL_S": "0.117",
        "PATH": str(bin_) + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": sep.join([str(STUBS), str(kit)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
        "M2FC_STUB": "1", "M2FC_STUB_SNAPSHOT_SHA256": _posix(t / "tiny-snapshot.sha256"), "M2FC_STUB_POLL_S": "1",
        "M2FC_CODE": _posix(ROOT), "M2FC_CONTEXT_MANIFEST": _posix(t / "CONTEXT_MANIFEST.json"),
        "M2FC_KIT_DIR": _posix(kit), "M2FC_DATA": _posix(data), "M2FC_CELL": _posix(cell),
        "HF_HUB_CACHE": _posix(hf), "HF_HUB_OFFLINE": "1", "GMN_METADATA_URL": "http://127.0.0.1:9",
        "DRYRUN_ROLE_N": "12", "DRYRUN_INDEX_N": "24",
    })
    return {"bash": bash, "t": t, "env": env}


def run_job(ctx, name: str, **extra) -> dict:
    d = ctx["t"] / name
    env = {**ctx["env"], "M2FC_WORK": _posix(d / "work"), "GMN_OUTPUT_DIR": _posix(d / "output"), **extra}
    r = subprocess.run([ctx["bash"], _posix(JOB)], env=env, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=3600)
    out, work = d / "output", d / "work"
    log = (work / "logs/run.log").read_text(encoding="utf-8", errors="replace")
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines() if x.strip()]
    res = json.loads((out / "result.json").read_text())
    return {"rc": r.returncode, "stdout": r.stdout, "log": log, "manifest": man, "result": res, "out": out,
            "work": work, "tail": log[-6000:]}


def _opened(log: str) -> list[str]:
    """Phases in the order run.log opened them."""
    pos = {k: log.find(v) for k, v in SAYS.items()}
    return [k for k, p in sorted(pos.items(), key=lambda kv: kv[1]) if p >= 0]


def _exports_verified(run) -> list[str]:
    for m in run["manifest"]:
        f = run["out"] / m["file"]
        assert m["ok"] is True and f.exists() and m["bytes"] == f.stat().st_size > 0, m
        assert hashlib.sha256(f.read_bytes()).hexdigest() == m["sha256"], m
    return [m["phase"] for m in run["manifest"]]


@pytest.fixture(scope="module")
def full_run(ctx):
    return run_job(ctx, "full")


def test_runner_executes_every_phase_end_to_end(full_run):
    r = full_run
    assert r["rc"] == 0, r["tail"]
    assert _opened(r["log"]) == PHASES + ["finalize"], r["tail"]
    assert "EXIT 0 (runner rc 0)" in r["log"] and "FAIL(" not in r["log"]
    assert "TEST OVERRIDES SET (never in a paid run): M2FC_CELL M2FC_CODE" in r["log"]
    assert _exports_verified(r) == EXPORTS
    res = r["result"]
    assert res["ok"] is True and res["exit_code"] == 0 and res["runner_rc"] == 0 and res["exports_verified"]
    assert list(res["gates"]) == PHASES and all(v == 0 for k, v in res["gates"].items() if k != "evals")
    assert res["gates"]["evals"] in (0, 40)  # a G12 reject is a measured outcome (exit 0)
    from jobs.m2_first_cell.summarize import valid_result

    dec = json.loads((r["work"] / "out/first_cell_result.json").read_text(encoding="utf-8"))
    assert valid_result(dec) and res["decision"] == dec and res["verdict"] == "measured"
    assert json.loads((r["out"] / "M2_FIRST_CELL_DONE").read_text()) == {"exit_code": 0}


def test_runner_setup_ran_every_identity_check(full_run):
    o = full_run["work"] / "out"
    env = json.loads((o / "env.json").read_text())
    assert env["wkv"] == "chunked_lean" and env["transformers"] == "5.17.0" and env["fla_gpu_verified"] is False
    for f in ("records-verify.txt", "snapshot-verify.txt", "data-verify.txt"):
        t = (o / f).read_text()
        assert ": OK" in t and "FAILED" not in t, (f, t)
    assert "m2/records.jsonl.gz: OK" in (o / "records-verify.txt").read_text()
    assert "slices/dev-rows.jsonl.gz: OK" in (o / "data-verify.txt").read_text()
    assert json.loads((o / "cohort-verify.json").read_text()) == {"rebuild_matches_committed": True}
    assert json.loads((o / "fla_guard.json").read_text())
    for p in ("parent_g12", "parent_index", "evals"):
        assert json.loads((o / f"preflight-{p}.json").read_text())["total_minutes"] > 0
    rp = json.loads((o / "reprice.json").read_text())
    assert rp["ok"] is True and rp["factor"] is not None  # measured from this run's parent rate reports
    assert json.loads((o / "heldout.json").read_text())["pass"]


def test_runner_exports_the_checkpoint(full_run):
    ck = full_run["out"] / "ckpt"
    latest = json.loads((ck / "latest.json").read_text())["path"]
    sha = (ck / f"{latest}.sha256").read_text().split()
    assert sha[0] == hashlib.sha256((ck / latest).read_bytes()).hexdigest() and sha[1].lstrip("*") == latest


def _stopped_before_training(r, code: int, fail_line: str, gate: str, exports: list[str]):
    assert r["rc"] == code, r["tail"]
    assert f"FAIL({code}): {fail_line}" in r["log"], r["tail"]
    assert "] phase cell" not in r["log"] and "] phase evals" not in r["log"]
    assert not (r["work"] / "out/cell.json").exists()
    assert _exports_verified(r) == exports
    res = r["result"]
    assert res["ok"] is False and res["exit_code"] == code and res["gates"][gate] == code and "cell" not in res["gates"]
    assert not (r["out"] / "M2_FIRST_CELL_DONE").exists()


def test_slow_box_at_parent_g12_exits_15_before_training(ctx):
    r = run_job(ctx, "slow-parent", M2FC_STUB_SLOW_PARENT="g12", M2FC_STUB_CHECK_AFTER_MIN="0.01",
                M2FC_STUB_POLL_S="0.2")
    _stopped_before_training(r, 15, "parent g12: live rate past its cap", "parent_g12",
                             ["setup", "guard", "heldout", "parent_g12", "final"])
    rep = json.loads((r["work"] / "out/parent_arm/rate-parent-g12.json").read_text())
    assert rep["exit"] == 15 and "> cap 85" in rep["stopped"]
    assert "parent_index" not in r["result"]["gates"]


def test_reprice_failure_exits_15_before_training(ctx):
    # the box's measured parent speed projects the child evals (~1.4x the parent volume + 8 fixed min) past a 5-min cap
    r = run_job(ctx, "reprice-fail", M2FC_STUB_EVALS_CAP="5")
    _stopped_before_training(r, 15, "child evals re-priced past their cap", "reprice",
                             ["setup", "guard", "heldout", "parent_g12", "parent_index", "final"])
    rp = json.loads((r["work"] / "out/reprice.json").read_text())
    assert rp["ok"] is False and rp["cap_min"] == 5 and rp["child_projection"]["total_minutes"] > 0.95 * 5
    assert r["result"]["gates"]["parent_index"] == 0


def test_overrides_default_to_the_paid_run_paths():
    """The M2FC_* overrides only relocate paths; unset, the runner is byte-for-byte the paid-run layout."""
    s = JOB.read_text(encoding="utf-8")
    for var, default in (("M2FC_CODE", "/opt/ssdm/m1"), ("M2FC_CONTEXT_MANIFEST", "/opt/ssdm/CONTEXT_MANIFEST.json"),
                         ("M2FC_KIT_DIR", "/opt/decision-index"), ("M2FC_DATA", "/data"),
                         ("M2FC_WORK", "/tmp/m2-first-cell"),
                         ("M2FC_CELL", "jobs/m2_first_cell/m2_first_cell_rwkv7.json")):
        assert f"${{{var}:-{default}}}" in s, var
        assert f"#   {var} " in s, var  # documented in the header
    body = s.split("set -Euo pipefail", 1)[1]
    assert not re.search(r"(?<![\w$])/(data|opt)/", re.sub(r"\$\{M2FC_\w+:-[^}]*\}", "", body))  # no stray hard paths
    assert set(re.findall(r"M2FC_\w+", body)) == {"M2FC_CODE", "M2FC_CONTEXT_MANIFEST", "M2FC_KIT_DIR", "M2FC_DATA",
                                                  "M2FC_WORK", "M2FC_CELL"}
    assert "M2FC_STUB" not in s  # the stubs live in the test, never in the runner


def test_sha256_manifests_are_lf_in_checkout_and_archive():
    """Shell-run finding: on a Windows checkout (core.autocrlf=true) the sha256sum -c manifests were CRLF, so
    ``sha256sum -c jobs/rwkv7/records.sha256`` looked for 'm2/records.jsonl.gz\r' (exit 10 at setup), and
    build_context.sh's ``git archive`` would ship them CRLF to the box. .gitattributes pins *.sha256 to LF."""
    git = shutil.which("git")
    if not git or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    files = subprocess.run([git, "ls-files", "*.sha256"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert {"jobs/rwkv7/records.sha256", "jobs/rwkv7/snapshot.sha256", "jobs/clef_baseline/data.sha256"} <= set(files)
    for f in files:
        attr = subprocess.run([git, "check-attr", "eol", "--", f], cwd=ROOT, capture_output=True, text=True).stdout
        assert attr.strip().endswith("eol: lf"), attr
        assert b"\r" not in (ROOT / f).read_bytes(), f


def test_restart_reuses_the_parent_arm_and_resumes_the_cell(ctx, full_run):
    """A platform restart (resume: checkpoint) keeps $CK: both parent stages come from the sha-verified cache, the
    re-price uses their kept rate reports, the cell resumes from its checkpoint, and the run still finalizes 0."""
    ck = ctx["t"] / "restart-ck"
    shutil.copytree(full_run["work"] / "ck", ck)
    r = run_job(ctx, "restart", GMN_CHECKPOINT_DIR=_posix(ck))
    assert r["rc"] == 0, r["tail"]
    assert "parent g12: cached (restart)" in r["log"] and "parent index: cached (restart)" in r["log"]
    assert "] phase parent_g12" not in r["log"] and "] phase parent_index" not in r["log"]
    assert json.loads((r["work"] / "out/reprice.json").read_text())["factor"] is not None
    assert _exports_verified(r) == ["setup", "guard", "heldout", "cell", "evals", "final"]
    assert r["result"]["exit_code"] == 0 and list(r["result"]["gates"]) == PHASES
    assert (r["out"] / "M2_FIRST_CELL_DONE").exists()
