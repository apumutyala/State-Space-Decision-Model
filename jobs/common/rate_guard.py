"""Measured stop conditions for the Index readout phases (replaces the unmeasured "~1 s/row" assumption).

Two checks, both pre-registered here and used by jobs/probe9b and jobs/rwkv7:

  preflight  before the kit run, from a MEASURED prefill rate (rwkv7: probe.json; probe9b: the pipeline's stock
             throughput at 4,096): projected minutes = subset proxy tokens x READ_FACTOR / tok_s / 60. If the
             projection already exceeds the phase cap, exit 15 without starting (a lower bound: decode-free option
             scoring still re-reads the question, so READ_FACTOR = 2 under-counts rather than over-counts).
  run        wraps the kit command; after CHECK_AFTER_MIN minutes (and >= MIN_ROWS rows) it projects the finish time
             from the rows the kit has actually written (one results.jsonl line per request). If the projection is past
             the cap, the run is stopped with exit 15 instead of timing out at the cap (exit 7) with nothing to score.
             The child's own exit code is returned otherwise; the wall-clock cap still gives 124.

python -m jobs.common.rate_guard preflight --tok-s X --subset S --cap-min M
python -m jobs.common.rate_guard run --results R --total N --cap-min M -- <command...>
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

EXIT_TOO_SLOW = 15
READ_FACTOR = 2.0
CHECK_AFTER_MIN = 8.0
MIN_ROWS = 20
MARGIN = 0.95  # stop if the projected finish is past 95% of the cap


def subset_proxy_tokens(subset_path: str) -> int:
    from jobs.common import dev_subset

    ids = set(dev_subset.load_subset(subset_path)["run_ids"])
    return sum(e["proxy_tokens"] for e in dev_subset.read_manifest() if e["run_id"] in ids)


def preflight(tok_s: float, proxy_tokens: int, cap_min: float, read_factor: float = READ_FACTOR) -> dict:
    proj = proxy_tokens * read_factor / max(tok_s, 1e-9) / 60
    return {"tok_s": tok_s, "proxy_tokens": proxy_tokens, "read_factor": read_factor, "projected_min": proj,
            "cap_min": cap_min, "ok": proj <= MARGIN * cap_min}


def preflight_volume(volume: dict, tok_s: float, call_s: float, cap_min: float, path: str = "fork") -> dict:
    """Realistic preflight (PERF_AUDIT): the subset's EXACT positions and calls under the scoring path that will run
    (jobs/common/readout_cost.py volume, real prompt builder + tokenizer) at the MEASURED prefill rate and per-call
    latency. The proxy preflight above under-counted job-mhc3h 40x: it priced 2.37M proxy tokens x 2 against a
    recompute path that forwards 236.7M (every question x option re-reads the prompt)."""
    from jobs.common.readout_cost import project as _proj

    p = _proj(volume["totals"], tok_s, call_s, path)
    return {**p, "family": volume.get("family"), "rows": volume.get("rows"), "cap_min": cap_min,
            "ok": p["projected_min"] <= MARGIN * cap_min, "method": "volume"}


def project(rows_done: int, elapsed_s: float, total: int, startup_s: float = 0.0) -> float:
    """Projected total minutes from the measured row rate. ``startup_s`` (opt-in, ``--exclude-startup``): seconds
    before the first row was written (model load, warm-up). They are counted once, not extrapolated per row, so a
    phase whose load is a large share of the first check is not stopped for its load time."""
    return (startup_s + (elapsed_s - startup_s) / max(rows_done, 1) * total) / 60


def _rows(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run(cmd: list[str], results: Path, total: int, cap_min: float, check_after_min: float = CHECK_AFTER_MIN,
        poll_s: float = 15.0, report: Path | None = None, exclude_startup: bool = False) -> int:
    t0 = time.time()
    child = subprocess.Popen(cmd, start_new_session=(os.name != "nt"))
    rec = {"cmd": cmd, "total": total, "cap_min": cap_min, "checks": [], "exclude_startup": exclude_startup}
    startup_s = None

    def stop(code, why):
        rec.update(stopped=why, exit=code)
        try:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            child.wait(timeout=60)
        except Exception:  # noqa: BLE001
            child.kill()
        return code

    code = None
    while code is None:
        rc = child.poll()
        el = time.time() - t0
        if rc is not None:
            code = rc
            rec["exit"] = rc
            break
        if el > 60 * cap_min:
            code = stop(124, "wall-clock cap")
            break
        n = _rows(results)
        if startup_s is None and n > 0:
            startup_s = el
            rec["startup_s"] = round(el, 1)
        if el >= 60 * check_after_min and n >= MIN_ROWS:
            proj = project(n, el, total, startup_s if exclude_startup and startup_s else 0.0)
            rec["checks"].append({"elapsed_min": el / 60, "rows": n, "projected_min": proj})
            if proj > MARGIN * cap_min:
                code = stop(EXIT_TOO_SLOW, f"measured {n} rows in {el / 60:.1f} min -> {proj:.0f} min > cap {cap_min}")
                break
        time.sleep(poll_s)
    rec["rows_done"] = _rows(results)
    rec["elapsed_min"] = (time.time() - t0) / 60
    if report:
        report.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("exit", "stopped", "rows_done", "elapsed_min")}), flush=True)
    return code


def _weight_done(path: Path) -> tuple[int, float]:
    n, w = 0, 0.0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    w += float(json.loads(line)["w"])
                    n += 1
                except (ValueError, KeyError):
                    continue  # a line being written
    except FileNotFoundError:
        pass
    return n, w


def project_weighted(start_s: float, now_s: float, done_w: float, total_w: float) -> float:
    """Projected finish in minutes since process start: scoring started at ``start_s`` (when the total file
    appeared, i.e. after model load / temperature fit) and proceeds at the measured weight rate."""
    return (start_s + (now_s - start_s) * total_w / max(done_w, 1e-9)) / 60


def run_weighted(cmd: list[str], progress: Path, cap_min: float, check_after_min: float = CHECK_AFTER_MIN,
                 poll_s: float = 15.0, report: Path | None = None, min_units: int = MIN_ROWS) -> int:
    """Like ``run`` for the m2 scoring drivers (``train.m2_gates.Progress``): units are heterogeneous (decision records,
    ~600-token steer units, ~3k-token Index rows), so the projection uses their weights (request chars + a per-call
    constant) instead of a row count. ``<progress>.total.json`` gives the total weight; the clock for the projection
    starts when it appears. Checked after ``check_after_min`` minutes of scoring and >= ``min_units`` units."""
    total_file = Path(str(progress) + ".total.json")
    for p in (progress, total_file):  # never project from a previous attempt's files
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    t0 = time.time()
    child = subprocess.Popen(cmd, start_new_session=(os.name != "nt"))
    rec = {"cmd": cmd, "cap_min": cap_min, "checks": [], "mode": "weighted"}
    start = total = None

    def stop(code, why):
        rec.update(stopped=why, exit=code)
        try:
            if os.name != "nt":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            child.wait(timeout=60)
        except Exception:  # noqa: BLE001
            child.kill()
        return code

    code = None
    while code is None:
        rc = child.poll()
        el = time.time() - t0
        if rc is not None:
            code = rc
            rec["exit"] = rc
            break
        if el > 60 * cap_min:
            code = stop(124, "wall-clock cap")
            break
        if total is None and total_file.exists():
            try:
                total = float(json.loads(total_file.read_text())["total_weight"])
                start = el
                rec["scoring_started_min"] = el / 60
            except (ValueError, KeyError):
                total = None
        if total is not None and el - start >= 60 * check_after_min:
            n, w = _weight_done(progress)
            if n >= min_units:
                proj = project_weighted(start, el, w, total)
                rec["checks"].append({"elapsed_min": el / 60, "units": n, "done_frac": w / total, "projected_min": proj})
                if proj > MARGIN * cap_min:
                    code = stop(EXIT_TOO_SLOW, f"{n} units ({100 * w / total:.1f}% of the weight) in "
                                               f"{(el - start) / 60:.1f} min of scoring -> {proj:.0f} min > cap {cap_min}")
                    break
        time.sleep(poll_s)
    rec["units_done"], rec["weight_done"] = _weight_done(progress)
    rec["elapsed_min"] = (time.time() - t0) / 60
    if report:
        report.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("exit", "stopped", "units_done", "elapsed_min")}), flush=True)
    return code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tail = []
    if "--" in argv:
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="rate_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--tok-s", type=float, required=True)
    p.add_argument("--subset", default=None, help="proxy-token lower bound (legacy); prefer --volume")
    p.add_argument("--volume", default=None, help="jobs/common/readout_volume_<family>.json")
    p.add_argument("--call-s", type=float, default=None, help="MEASURED per-call latency (probe small_call_s)")
    p.add_argument("--path", default="fork", choices=["fork", "recompute", "per_question"])
    p.add_argument("--cap-min", type=float, required=True)
    p.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("--results", required=True)
    r.add_argument("--total", type=int, required=True)
    r.add_argument("--cap-min", type=float, required=True)
    r.add_argument("--check-after-min", type=float, default=CHECK_AFTER_MIN)
    r.add_argument("--poll-s", type=float, default=15.0)
    r.add_argument("--report", default=None)
    r.add_argument("--exclude-startup", action="store_true",
                   help="count the time before the first row once instead of extrapolating it (model load)")
    w = sub.add_parser("run-weighted")
    w.add_argument("--progress", required=True, help="train.m2_gates.Progress file (+ .total.json)")
    w.add_argument("--cap-min", type=float, required=True)
    w.add_argument("--check-after-min", type=float, default=CHECK_AFTER_MIN)
    w.add_argument("--poll-s", type=float, default=15.0)
    w.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run-weighted":
        if not tail:
            ap.error("run-weighted needs a command after --")
        return run_weighted(tail, Path(a.progress), a.cap_min, a.check_after_min, a.poll_s,
                            Path(a.report) if a.report else None)
    if a.cmd == "preflight":
        if a.volume:
            if a.call_s is None:
                ap.error("--volume needs --call-s")
            res = preflight_volume(json.loads(Path(a.volume).read_text()), a.tok_s, a.call_s, a.cap_min, a.path)
        elif a.subset:
            res = preflight(a.tok_s, subset_proxy_tokens(a.subset), a.cap_min)
        else:
            ap.error("preflight needs --volume (or the legacy --subset)")
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return 0 if res["ok"] else EXIT_TOO_SLOW
    if not tail:
        ap.error("run needs a command after --")
    return run(tail, Path(a.results), a.total, a.cap_min, a.check_after_min, a.poll_s,
               Path(a.report) if a.report else None, a.exclude_startup)


if __name__ == "__main__":
    sys.exit(main())
