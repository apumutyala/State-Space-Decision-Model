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


def project(rows_done: int, elapsed_s: float, total: int) -> float:
    """Projected total minutes from the measured row rate."""
    return elapsed_s / max(rows_done, 1) * total / 60


def _rows(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def run(cmd: list[str], results: Path, total: int, cap_min: float, check_after_min: float = CHECK_AFTER_MIN,
        poll_s: float = 15.0, report: Path | None = None) -> int:
    t0 = time.time()
    child = subprocess.Popen(cmd, start_new_session=(os.name != "nt"))
    rec = {"cmd": cmd, "total": total, "cap_min": cap_min, "checks": []}

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
        if el >= 60 * check_after_min and n >= MIN_ROWS:
            proj = project(n, el, total)
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
    p.add_argument("--subset", required=True)
    p.add_argument("--cap-min", type=float, required=True)
    p.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("--results", required=True)
    r.add_argument("--total", type=int, required=True)
    r.add_argument("--cap-min", type=float, required=True)
    r.add_argument("--check-after-min", type=float, default=CHECK_AFTER_MIN)
    r.add_argument("--poll-s", type=float, default=15.0)
    r.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "preflight":
        res = preflight(a.tok_s, subset_proxy_tokens(a.subset), a.cap_min)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res))
        return 0 if res["ok"] else EXIT_TOO_SLOW
    if not tail:
        ap.error("run needs a command after --")
    return run(tail, Path(a.results), a.total, a.cap_min, a.check_after_min, a.poll_s,
               Path(a.report) if a.report else None)


if __name__ == "__main__":
    sys.exit(main())
