"""TEST-ONLY interpreter hook for tests/test_m2_first_cell_shell.py, which EXECUTES jobs/m2_first_cell/first_cell_job.sh
under bash on CPU. That test puts this directory on PYTHONPATH (and sets M2FC_STUB=1), so every ``python`` the runner
starts -- including the drivers jobs.common.rate_guard launches -- loads this file first. Inert without M2FC_STUB=1;
never on a job image's path. It only stands in for what a CPU laptop cannot provide, per entry point:

* the drivers that load the model (dev_subset rows, guard_ready, score_parent, run_cell, m2_gates, summarize, the
  import / snapshot-code checks, finalize's validator): scripts/first_cell_dryrun_shim.py is applied -- the pinned kit
  clone on sys.path, the tiny snapshot accepted by verify_snapshot_code, DRYRUN_ROLE_N cohort roles, LoRA group counts
  scaled to the tiny model -- exactly the CPU dry run's substitutions;
* ``--device cpu`` is appended to the three drivers whose default is cuda (score_parent, run_cell, m2_gates);
* the setup env check (``python -`` with no arguments): torch.cuda.is_available / get_device_name answer as on a GPU box;
* ``dev_subset rows``: with DRYRUN_INDEX_N the written subset is cut to about that many rows (whole groups; CPU time),
  as the dry run does (scripts.m2_first_cell_dryrun.index_subset);
* ``rate_guard run-weighted``: ``--poll-s`` from M2FC_STUB_POLL_S and, if set, ``--check-after-min`` from
  M2FC_STUB_CHECK_AFTER_MIN (time compression; the guard's projection rule itself is unchanged); on Windows the
  wrapped command's bare ``python`` becomes sys.executable (CreateProcess would find the venv's base interpreter);
* negative tests: M2FC_STUB_SLOW_PARENT=<stage> makes that score_parent stage a slow box (each scored unit weighs
  1e6 x its planned share of the total, plus 0.05 s); M2FC_STUB_EVALS_CAP=<min> hands jobs.m2_first_cell.reprice a
  copy of job_params with that evals cap (the box's measured speed no longer fits the child evals).
"""
import os
import sys

if os.environ.get("M2FC_STUB") == "1":
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    _DEVICE = {"jobs.m2_parent_arm.score_parent", "train.run_cell", "train.m2_gates"}
    _SHIM = _DEVICE | {"jobs.common.dev_subset", "jobs.m2_pilot.guard_ready", "jobs.m2_first_cell.summarize"}

    def _split(argv):
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in ("-m", "-c"):
                return a, argv[i + 1], argv[i + 2:]
            if a == "-":
                return "-", None, argv[i + 1:]
            if a in ("-X", "-W"):
                i += 2
                continue
            if a.startswith("-"):
                i += 1
                continue
            return "script", a, argv[i + 1:]
        return None, None, []

    def _shim():
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        import scripts.first_cell_dryrun_shim  # noqa: F401  (applies the dry run's substitutions on import)

    _kind, _what, _rest = _split(list(getattr(sys, "orig_argv", []))[1:])
    if _kind == "-m" and _what in _SHIM:
        _shim()
    elif _kind == "-c" and any(m in _what for m in ("third_party", "import train.", "jobs.m2_first_cell")):
        _shim()
    elif _kind == "script" and _what.replace("\\", "/").endswith("jobs/common/finalize.py"):
        _shim()
    elif _kind == "-" and not _rest:  # the setup env check
        _shim()
        import torch

        torch.cuda.is_available = lambda: True
        torch.cuda.get_device_name = lambda *a, **k: "cpu (tests/shell_stubs stand-in)"

    if _kind == "-m" and _what in _DEVICE and _rest[:1] != ["estimate"] and "--device" not in _rest:
        sys.argv += ["--device", "cpu"]

    if _kind == "-m" and _what == "jobs.common.dev_subset" and _rest[:1] == ["rows"] and os.environ.get("DRYRUN_INDEX_N"):
        import atexit

        def _cut(out=_rest[_rest.index("--out") + 1], n=int(os.environ["DRYRUN_INDEX_N"])):
            if os.path.exists(out):
                from scripts.m2_first_cell_dryrun import index_subset

                index_subset(Path(out), Path(out), n)

        atexit.register(_cut)

    if _kind == "-m" and _what == "jobs.common.rate_guard" and _rest[:1] == ["run-weighted"]:
        extra = ["--poll-s", os.environ.get("M2FC_STUB_POLL_S", "1")]
        if os.environ.get("M2FC_STUB_CHECK_AFTER_MIN"):
            extra += ["--check-after-min", os.environ["M2FC_STUB_CHECK_AFTER_MIN"]]
        j = sys.argv.index("--")
        sys.argv[j:j] = extra
        if os.name == "nt" and sys.argv[j + len(extra) + 1:j + len(extra) + 2] == ["python"]:
            # Windows: a venv's python.exe re-launches the base interpreter, whose directory CreateProcess searches
            # first for a bare "python" -> the base Python without this env's packages. Linux resolves it on PATH.
            sys.argv[j + len(extra) + 1] = sys.executable

    _slow = os.environ.get("M2FC_STUB_SLOW_PARENT")
    if (_slow and _kind == "-m" and _what == "jobs.m2_parent_arm.score_parent" and "--stage" in _rest
            and _rest[_rest.index("--stage") + 1] == _slow):
        import time

        from train import m2_gates as _G

        _plan, _wrap = _G.Progress.plan, _G.Progress.wrap
        _G.Progress.plan = lambda self, weights: _plan(self, [w * 1_000_000 for w in weights])

        def _slow_wrap(self, fn):
            g = _wrap(self, fn)

            def call(req):
                time.sleep(0.05)
                return g(req)

            if hasattr(g, "decide_many"):
                def many(reqs):
                    time.sleep(0.05 * len(reqs))
                    return g.decide_many(reqs)
                call.decide_many = many
            return call

        _G.Progress.wrap = _slow_wrap

    if os.environ.get("M2FC_STUB_EVALS_CAP") and _kind == "-m" and _what == "jobs.m2_first_cell.reprice":
        import json
        import tempfile

        k = sys.argv.index("--params") + 1
        p = json.loads(Path(sys.argv[k]).read_text(encoding="utf-8"))
        p["phase_caps_minutes"]["evals"] = float(os.environ["M2FC_STUB_EVALS_CAP"])
        f = Path(tempfile.mkdtemp()) / "job_params.json"
        f.write_text(json.dumps(p), encoding="utf-8")
        sys.argv[k] = str(f)
