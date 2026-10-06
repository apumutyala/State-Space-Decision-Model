"""CPU tests for the paid-run stop conditions (align.Guard) and the pre-registered 9B decision rule.

Run: python -m tests.test_guard
"""
from __future__ import annotations

import json
import sys
import time

from convert.align import Guard, StopCondition
from scripts.decide_9b import decide

CFG = json.load(open("configs/qual_h100.json"))


def test_throughput_stop():
    g = Guard({"min_tok_s": 1e9, "check_after_steps": 20}, "t")
    try:
        for s in range(30):
            time.sleep(0.002)  # ~5e5 tok/s << 1e9
            g.step(s, (s + 1) * 1000)
        return False
    except StopCondition as e:
        return e.code == 3


def test_no_stop_when_fast():
    g = Guard({"min_tok_s": 1.0, "check_after_steps": 20}, "t")
    for s in range(30):
        g.step(s, (s + 1) * 1000)
    return g.rate is not None and g.rate > 1.0


def test_wall_clock():
    g = Guard({"max_minutes": 1e-4}, "t")
    time.sleep(0.02)
    return g.step(0, 1) is True and g.time_fraction() > 1


def test_rules():
    mk = lambda tok_s, peak, stopped=None: {"stage1": {"peak_gb": peak}, "progressive": {"steps": [  # noqa: E731
        {"steady_tok_s": tok_s, "peak_gb": peak, "flop_per_token": 8.5e10}]}, **({"stopped": stopped} if stopped else {})}
    f = 8.5e10 / 989e12
    cases = [
        (mk(0.30 / f, 50), "open-$40-one-layer-probe-spend-issue"),
        (mk(0.30 / f, 70), "re-price-lean-plan"),
        (mk(0.20 / f, 50), "re-price-lean-plan"),
        (mk(0.10 / f, 50), "fold"),
        (mk(0.30 / f, 80), "fold"),
        (mk(0.30 / f, 50, {"exit_code": 3, "tok_s": 100}), "fold"),
        ({"stage1": {}}, None),
    ]
    ok = True
    for R, want in cases:
        v = decide(R, CFG)["verdict"]
        ok &= (v == want) if want else v.startswith("no-measurement")
    return ok


def main():
    res = {f.__name__: f() for f in (test_throughput_stop, test_no_stop_when_fast, test_wall_clock, test_rules)}
    res["PASS"] = all(res.values())
    print(json.dumps(res, indent=1))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
