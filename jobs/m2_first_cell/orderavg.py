"""Test-time ORDER AVERAGING screen of a saved m2 checkpoint (``fulldev_eval score --order-avg 1,2,4``).

Hypothesis (agent B, 2026-10-09): scoring every choice question under several option presentation orders and
averaging the per-option probabilities (``decide.debias.OrderAveragedBackend``) removes position / label priors seen in
the readout analysis (Humicroedit, When2Call "D" over-predicted, ESCI, PhishNChips) and raises D13's Index under v3w.

One job, three arms from the SAME forward passes: per row the N = 4 arm runs passes 0..3 (identity, reversed, two
seeded cyclic shifts; fewer when a row's questions have fewer distinct orders, e.g. all k = 2), then N = 2 and N = 1
re-use passes 0..1 / 0 from a per-row cache (``decide.debias.OrderCache``). Pass 0 is the plain v3w prompt through the
plain backend (N = 1 = the control, must reproduce the 41.07 proxy600 screen within noise). Each arm's COST is the sum
of the measured walls of its own passes (+ its own adapter overhead), i.e. what that arm alone would take; its input
tokens are the sum over its passes. Results: ``child.results.jsonl`` = N = 1 (the existing report / validator read it
unchanged), ``child.oa2.results.jsonl``, ``child.oa4.results.jsonl``.

Report (``orderavg_section``): per arm the proxy600 Index (frozen ``harness.score.decompose`` + ``harness.bootstrap``,
as ``fulldev_eval.subset_score``), the paired difference vs N = 1 (``harness.bootstrap.compare`` on identical rows,
B = 2000, seed 0), wall and tokens per arm and the cost multiplier vs N = 1. Pre-registered rule (README):
ADOPT_CANDIDATE if N = 2 or N = 4 minus N = 1 has a paired 95 % CI lower bound > 0 on proxy600, then confirm on
DEV-minus-proxy600; else REJECT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ARMS = (4, 2, 1)  # scoring order per row: the largest arm runs every pass, the others read the cache
CANDIDATE_ARMS = (2, 4)  # each compared with N = 1 (the control)
CI_LO_THRESHOLD = 0.0  # adopt-candidate iff the paired 95 % CI lower bound of (N - N=1) is > this
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
CONTROL_CI = (37.57, 45.22)  # the 41.07 proxy600 screen's 95 % CI: N = 1 outside it -> INVALID_CONTROL
RULE = ("ADOPT_CANDIDATE if (N=2 - N=1) or (N=4 - N=1) has a paired 95% bootstrap CI lower bound > 0 on proxy600 "
        "(harness.bootstrap.compare, B=2000, seed 0, identical rows); then confirm on DEV-minus-proxy600 before any "
        "adoption. Else REJECT. Pre-registered in jobs/m2_first_cell/ORDERAVG_README.md before the job ran.")


def parse_ns(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    ns = tuple(sorted({int(x) for x in str(text).split(",") if x.strip()}, reverse=True))
    if ns != ARMS:
        raise ValueError(f"--order-avg {text!r}: the pre-registered arm set is {','.join(map(str, sorted(ARMS)))}")
    return ns


def arm_file(arm: str, n: int, kind: str) -> str:
    """``kind`` partial | results. N = 1 keeps the plain names (the control the existing report scores)."""
    return f"{arm}.{kind}.jsonl" if n == 1 else f"{arm}.oa{n}.{kind}.jsonl"


class OrderAvgSet:
    """The three deciders (identity calibration, as the untempered screen arm) over one base backend + one cache."""

    def __init__(self, decider, ns: tuple[int, ...] = ARMS) -> None:
        from decide import Decider
        from decide.debias import OrderAveragedBackend, OrderCache

        self.cache = OrderCache()
        self.ns = ns
        self.backends = {n: OrderAveragedBackend(decider.backend, n, self.cache) for n in ns}
        self.deciders = {n: Decider(self.backends[n], calibration=decider.calibration,
                                    enable_v0_1_deltas=getattr(decider, "enable_v0_1_deltas", False),
                                    backend_max_input_tokens=getattr(decider, "backend_max_input_tokens", None))
                         for n in ns}

    def identity(self) -> dict[str, Any]:
        from decide.debias import ORDER_AVG_ID, ORDER_AVG_SEED

        return {"id": ORDER_AVG_ID, "seed": ORDER_AVG_SEED, "arms": list(self.ns), "combine": "mean_probability",
                "revisions": {str(n): self.deciders[n].revision()["revision_id"] for n in self.ns}}

    def score_row(self, row: dict, decide_for, index_predict) -> dict[int, dict]:
        """{n: result row} for one DEV row. ``decide_for(n, decide)`` wraps a decide (limits, progress, beat)."""
        rid = row["_evaluation"]["run_id"]
        self.cache.clear()
        out = {}
        for n in self.ns:
            b = self.backends[n]
            b.last_pass_ms, b.last_fresh_ms, b.last_input_tokens = [], 0.0, 0
            t = time.perf_counter()
            res = index_predict(decide_for(n, self.deciders[n].decide), [row])[rid]
            shared = (time.perf_counter() - t) * 1000
            res = dict(res)
            # this arm alone = its adapter wall minus the passes it ran here + every pass it needs (cached or fresh)
            res["oa_shared_wall_ms"] = shared
            res["oa_pass_ms"] = list(b.last_pass_ms)
            res["oa_input_tokens"] = b.last_input_tokens
            res["total_wall_ms"] = float(res.get("total_wall_ms") or shared) - b.last_fresh_ms + sum(b.last_pass_ms)
            res["oa_n"] = n
            out[n] = res
        self.cache.clear()
        return out


def pass_ratio(rows: list[dict], lengths: dict[str, int], n: int = max(ARMS)) -> dict[str, Any]:
    """CPU pricing input: sum over rows of (passes of the N = n arm x committed v1 row tokens) / sum of row tokens.
    Depends only on the rows' question types / option counts (``decide.debias.presentation_passes``)."""
    from decide.debias import presentation_passes

    tot = base = n_pass = 0
    for r in rows:
        k = len(presentation_passes(r, n))
        t = lengths[r["_evaluation"]["run_id"]]
        tot, base, n_pass = tot + k * t, base + t, n_pass + k
    return {"n": n, "rows": len(rows), "passes": n_pass, "ratio": round(tot / base, 4) if base else None}


def check_rows(rows: list[dict], n: int = max(ARMS)) -> dict[str, Any]:
    """CPU: every pass of every row builds under the v3w layout (``heads.layout.build_row`` with the test byte
    encoder: same prompt text / spans logic, not the RWKV tokenizer) and puts each option's code / span where its
    presentation says. Returns counts; raises on the first failure."""
    from decide.debias import presentation_passes
    from heads.layout import ByteEncoder, build_row

    enc = ByteEncoder()
    n_rows = n_built = 0
    for r in rows:
        req = {"model": "x", "state": r["state"], "questions": r["questions"]}
        for p in presentation_passes(req, n)[1:]:
            for qid in req["questions"]:
                row = build_row(req, qid, enc, template="jrt-rwkv-g1-v3w", layout="g1v2", presentation=p)
                if len(row.option_ids) != len(req["questions"][qid]["criteria"]) and req["questions"][qid]["type"] == "choice":
                    raise ValueError(f"{r['_evaluation']['run_id']}/{qid}: option axis changed")
                n_built += 1
        n_rows += 1
    return {"rows": n_rows, "variant_rows_built": n_built}


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.orderavg")
    ap.add_argument("cmd", choices=("passes", "check"))
    ap.add_argument("--rows", required=True, help="DEV rows (data/slices/dev-rows.jsonl.gz)")
    ap.add_argument("--select", default="proxy600")
    a = ap.parse_args(argv)
    import gzip
    from jobs.m2_first_cell.fulldev_eval import LENGTHS, selected_ids

    keep = selected_ids(a.select)
    with gzip.open(a.rows, "rt", encoding="utf-8") as fh:
        rows = [r for r in map(json.loads, fh) if keep is None or r["_evaluation"]["run_id"] in keep]
    if a.cmd == "passes":
        out = pass_ratio(rows, json.loads(LENGTHS.read_text(encoding="utf-8"))["lengths"])
    else:
        out = check_rows(rows)
    print(json.dumps(out))
    sys.stdout.flush()
    return 0


def _sum(rows: list[dict], key: str) -> float:
    return float(sum(float(r.get(key) or 0) for r in rows))


def orderavg_section(d: Path, keep: set[str] | None, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Per-arm Index + paired diff vs N = 1 + cost, from the arm result files in ``d`` (proxy600 rows ``keep``)."""
    from jobs.m2_first_cell.fulldev_eval import subset_compare, subset_score

    base = d / arm_file("child", 1, "results")
    arms: dict[str, Any] = {}
    rows_by = {}
    for n in sorted(ARMS):
        p = d / arm_file("child", n, "results")
        if not p.is_file():
            raise SystemExit(f"order averaging: no {p.name}")
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows_by[n] = rows
        sc = subset_score(p, keep, B=B, seed=seed)
        ent = {"index": sc["slice_index"]["index"], "rows": sc["rows"],
               "status": {s: sum(1 for r in rows if r.get("status") == s) for s in sorted({r.get("status") for r in rows})},
               "wall_s": round(_sum(rows, "total_wall_ms") / 1000, 1),
               "passes": int(sum(len(r.get("oa_pass_ms") or []) for r in rows)),
               "input_tokens": int(_sum(rows, "oa_input_tokens"))}
        if n != 1:
            cmp = subset_compare(p, base, keep, B=B, seed=seed)
            (d / f"compare_oa{n}_vs_n1.json").write_text(json.dumps(cmp, indent=1), encoding="utf-8")
            ent["minus_n1"] = cmp["diff"]["index"]
        arms[f"n{n}"] = ent
    w1, t1, p1 = arms["n1"]["wall_s"], arms["n1"]["input_tokens"], arms["n1"]["passes"]
    for n in sorted(ARMS):
        a = arms[f"n{n}"]
        a["cost_multiplier_vs_n1"] = {"wall": round(a["wall_s"] / w1, 3) if w1 else None,
                                      "input_tokens": round(a["input_tokens"] / t1, 3) if t1 else None,
                                      "passes": round(a["passes"] / p1, 3) if p1 else None}
    c1 = (arms["n1"]["index"] or {}).get("point")
    control_ok = isinstance(c1, (int, float)) and CONTROL_CI[0] <= c1 <= CONTROL_CI[1]
    win = [n for n in CANDIDATE_ARMS if (arms[f"n{n}"].get("minus_n1") or {}).get("lo", float("-inf")) > CI_LO_THRESHOLD]
    verdict = "INVALID_CONTROL" if not control_ok else ("ADOPT_CANDIDATE" if win else "REJECT")
    return {"arms": arms, "rule": RULE, "verdict": verdict, "control_in_screen_ci": control_ok,
            "winning_arms": [f"n{n}" for n in win],
            "job_wall_note": "one job ran every pass once (N=4's); N=2 / N=1 re-used passes 0..1 / 0 from the per-row "
                             "cache, so each arm's wall_s = the measured walls of its own passes + its own adapter "
                             "overhead (what that arm alone costs), and the job's scoring wall ~= wall_s of n4"}


if __name__ == "__main__":
    raise SystemExit(main())
