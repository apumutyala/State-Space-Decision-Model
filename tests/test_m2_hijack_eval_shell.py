"""EXECUTES jobs/m2_first_cell/first_cell_job.sh hijack_eval (BOARD H1, eval-only) under bash on CPU, end to end: the
tiny checkpoint the base runner trains in tests/test_m2_first_cell_shell.py (fixture ``full_run``) is packed as a job
artifact (./ckpt/<pt> + ./ckpt/latest.json, the layout of art-bm3e6), served by file:// as $CKPT_URL, and pinned in a
copy of job_params_hijack_eval.json (M2FC_PARAMS). Every phase, exit-code mapping, export and finalize runs for real;
the stand-ins are the base shell test's (tiny model, DRYRUN_ROLE_N cohort cut, PATH stubs)."""
import hashlib
import json
import subprocess
import tarfile

import pytest

from tests.test_m2_first_cell_shell import JOB, ROOT, _exports_verified, _opened, _posix, ctx, full_run  # noqa: F401

pytestmark = pytest.mark.shell_runner
SAYS = {"setup": "] setup", "guard": "] phase guard", "checkpoint": "] phase checkpoint",
        "parent_hijack": "] phase parent_hijack", "evals": "] phase evals", "report": "] phase report",
        "finalize": "] EXIT "}


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pack(ctx, full_run, name: str, tamper: bool = False):
    ck = full_run["work"] / "ck" / "cell"
    latest = json.loads((ck / "latest.json").read_text())
    pt = ck / latest["path"]
    d = ctx["t"] / name
    d.mkdir()
    tar = d / "artifact.tar"
    with tarfile.open(tar, "w") as t:
        t.add(pt, arcname=f"./ckpt/{pt.name}")
        t.add(ck / "latest.json", arcname="./ckpt/latest.json")
    cell = full_run["work"] / "out" / "cell.json"
    data_cell = ctx["t"] / "data" / "checkpoint" / "cell.json"
    data_cell.parent.mkdir(parents=True, exist_ok=True)
    data_cell.write_bytes(cell.read_bytes())
    p = json.loads((ROOT / "jobs/m2_first_cell/job_params_hijack_eval.json").read_text(encoding="utf-8"))
    p["checkpoint"] = {**p["checkpoint"], "artifact": "art-shell", "job": "job-shell", "artifact_sha256": _sha(tar),
                       "member": f"./ckpt/{pt.name}", "ckpt_sha256": "0" * 64 if tamper else _sha(pt),
                       "latest_json_sha256": _sha(ck / "latest.json"), "cell_json_sha256": _sha(data_cell)}
    params = d / "job_params_hijack_eval.json"
    params.write_text(json.dumps(p, indent=1), encoding="utf-8", newline="\n")
    return tar, params


def _run(ctx, full_run, name: str, tamper: bool = False):
    tar, params = _pack(ctx, full_run, name, tamper)
    d = ctx["t"] / name
    env = {**ctx["env"], "M2FC_WORK": _posix(d / "work"), "GMN_OUTPUT_DIR": _posix(d / "output"),
           "M2FC_PARAMS": _posix(params), "CKPT_URL": tar.resolve().as_uri()}
    r = subprocess.run([ctx["bash"], _posix(JOB), "hijack_eval"], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    work, out = d / "work", d / "output"
    log = (work / "logs/run.log").read_text(encoding="utf-8", errors="replace")
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines() if x.strip()]
    return {"rc": r.returncode, "log": log, "manifest": man, "result": json.loads((out / "result.json").read_text()),
            "out": out, "work": work, "tail": log[-6000:]}


def test_hijack_eval_runs_every_phase_on_a_saved_checkpoint(ctx, full_run):
    assert full_run["rc"] == 0, full_run["tail"]
    r = _run(ctx, full_run, "hijack")
    assert r["rc"] == 0, r["tail"]
    pos = {k: r["log"].find(v) for k, v in SAYS.items()}
    assert all(p >= 0 for p in pos.values()) and list(pos) == sorted(pos, key=pos.get), r["tail"]
    for absent in ("] phase heldout", "] phase cell", "] re-price", "] phase parent_g12"):
        assert absent not in r["log"], absent
    assert _exports_verified(r) == ["setup", "guard", "checkpoint", "parent_hijack", "evals", "final"]
    res = r["result"]
    assert res["ok"] is True and res["exit_code"] == 0 and res["gates"]["evals"] == 46
    from jobs.m2_first_cell.hijack_eval import valid_result

    dec = json.loads((r["work"] / "out/hijack_eval_result.json").read_text(encoding="utf-8"))
    assert valid_result(dec) and res["decision"] == dec and dec["checkpoint"]["artifact"] == "art-shell"
    assert dec["hijack"]["n_units"] > 0 and dec["units"]["by_family"]["hijack"] == 5 * dec["units"]["n_requests"]
    assert (r["out"] / "M2_FIRST_CELL_DONE").exists()
    assert len((r["out"] / "result.json").read_bytes()) < 32 * 1024  # the platform's result limit


def test_hijack_eval_refuses_a_checkpoint_that_fails_its_pin(ctx, full_run):
    r = _run(ctx, full_run, "hijack-tamper", tamper=True)
    assert r["rc"] == 10 and "FAIL(10): checkpoint" in r["log"], r["tail"]
    assert "] phase parent_hijack" not in r["log"] and r["result"]["ok"] is False
