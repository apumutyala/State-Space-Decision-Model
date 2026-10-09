"""jobs/backbone_screen: decision.json from the screen's own outputs (measurement only; no pass/fail threshold).

    python -m jobs.backbone_screen.summarize <job_params.json> <out dir> <decision.json>
    python -m jobs.backbone_screen.summarize check-arm-a <job_params.json> <index/score.json>   (two-arm modes)

Reads index/score.json (proxy600 slice index), index/ece.json, index/compare_stock_v3w.json (paired bootstrap vs the
stock RWKV7-G1k v3w readout on the same 644 rows), index/selfcheck.json (fork == recompute canary) and
index/rate.json, plus the per-row statuses of index/stock/results.jsonl. ``valid_screen`` is finalize.py's validator.

Two-arm modes (params ``screen.arm_b``, MODE qrwkv6_32b_v3w / qwen36_a3b): arm a is the stock screen above and must
reproduce ``screen.expect_arm_a_index`` within ``expect_arm_a_tol`` when one is given (``check_arm_a``; the runner exits
16 before arm b otherwise; without one, e.g. a first measurement, only the 644 rows are checked); arm b's outputs (index/score_b.json, ece_b.json, compare_b_stock_v3w.json, compare_b_vs_a.json,
selfcheck_b.json, rate_b.json, v3w/results.jsonl) go to decision.json ``arm_b`` with the paired diff b - a [CI].
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

SCHEMA = "backbone-screen/v1"
#: stock RWKV7-G1k-13.3B v3w readout on the same proxy600 rows (art-ivu28 v3w.results.jsonl, rescored locally with
#: jobs.common.dev_subset score --subset jobs/common/dev_proxy600.json; 31.07 on the 2,008-row subset)
STOCK_V3W_PROXY600 = 27.32


def _row_status(path: Path) -> dict:
    status = collections.Counter()
    for line in path.open(encoding="utf-8"):
        status[json.loads(line).get("status", "?")] += 1
    return dict(status)


def check_arm_a(params: dict, score: dict) -> dict:
    s = params["screen"]
    got = score["slice_index"]["index"]["point"]
    if s.get("expect_arm_a_index") is None:  # first measurement of this backbone (MODE qwen36_a3b): rows only
        return {"index": got, "expected": None, "tol": None, "source": None, "rows": score["rows"],
                "note": "no arm-a reproduction target (first measurement of this checkpoint); rows checked only",
                "ok": bool(score["rows"] == 644)}
    want, tol = s["expect_arm_a_index"], s.get("expect_arm_a_tol", 0.01)
    return {"index": got, "expected": want, "tol": tol, "source": s.get("expect_arm_a_source"), "rows": score["rows"],
            "ok": bool(abs(got - want) <= tol and score["rows"] == 644)}


def _arm_b(params: dict, ix: Path) -> dict:
    def j(name):
        return json.loads((ix / name).read_text(encoding="utf-8"))

    score, sc = j("score_b.json"), j("selfcheck_b.json")
    vs_a, vs_stock = j("compare_b_vs_a.json"), j("compare_b_stock_v3w.json")
    rate = j("rate_b.json") if (ix / "rate_b.json").exists() else {}
    return {"family": params["screen"]["arm_b"]["family"], "backbone_id": params["screen"]["arm_b"]["backbone_id"],
            "v3w_sha256": params["screen"]["arm_b"]["v3w_sha256"], "rows": score["rows"],
            "index": score["slice_index"]["index"], "areas": score["slice_index"].get("areas"),
            "diff_vs_arm_a": vs_a.get("diff", {}).get("index"), "mde_vs_arm_a": vs_a.get("mde"),
            "vs_stock_rwkv7_v3w": {"baseline_proxy600": STOCK_V3W_PROXY600, "diff": vs_stock.get("diff", {}).get("index"),
                                   "mde": vs_stock.get("mde")},
            "ece": {k: v for k, v in j("ece_b.json").items() if not isinstance(v, (dict, list))},
            "row_status": _row_status(ix / "v3w" / "results.jsonl"),
            "selfcheck": {k: sc.get(k) for k in ("ok", "max_abs_delta", "same_argmax", "options", "tol")},
            "rate": {k: rate.get(k) for k in ("rows_done", "elapsed_min", "startup_s", "exit")},
            "arm_a_check": j("arm_a_check.json")}


def build(params: dict, out: Path) -> dict:
    ix = out / "index"
    score = json.loads((ix / "score.json").read_text(encoding="utf-8"))
    comp = json.loads((ix / "compare_stock_v3w.json").read_text(encoding="utf-8"))
    ece = json.loads((ix / "ece.json").read_text(encoding="utf-8"))
    selfcheck = json.loads((ix / "selfcheck.json").read_text(encoding="utf-8"))
    rate = json.loads((ix / "rate.json").read_text(encoding="utf-8")) if (ix / "rate.json").exists() else {}
    s = params["screen"]
    res = ix / "stock" / ("results.scored.jsonl" if s.get("primary_subset") else "results.jsonl")
    want = set(_ids(s["primary_subset"])) if s.get("primary_subset") else None
    status = collections.Counter()
    for line in res.open(encoding="utf-8"):
        r = json.loads(line)
        if want is None or r.get("run_id") in want:
            status[r.get("status", "?")] += 1
    index = score["slice_index"]["index"]
    return {
        "schema": SCHEMA, "verdict": "measured", "mode": params["mode"], "repo": s["repo"], "revision": s["revision"],
        "family": s["family"], "rows": score["rows"], "subset_ids_sha256": score["subset_ids_sha256"],
        **({"one_load_arms": True} if s.get("one_load_arms") else {}),
        **({"readout_a": s["readout_a"]} if s.get("readout_a") else {}),
        "index": index, "areas": score["slice_index"].get("areas"),
        "vs_stock_rwkv7_v3w": {"baseline": "art-ivu28 v3w (job-fgu2c)", "baseline_proxy600": STOCK_V3W_PROXY600,
                               "diff": comp.get("diff", {}).get("index"), "mde": comp.get("mde")},
        "ece": {k: v for k, v in ece.items() if not isinstance(v, (dict, list))} or ece,
        "row_status": dict(status),
        "selfcheck": {k: selfcheck.get(k) for k in ("ok", "max_abs_delta", "same_argmax", "options", "tol")},
        "rate": {k: rate.get(k) for k in ("rows_done", "elapsed_min", "startup_s", "exit")},
        **({"arm_b": _arm_b(params, ix)} if s.get("arm_b") else {}),
        **(_primary(params, ix) if s.get("primary_subset") else {}),
    }


def _ids(rel: str) -> list[str]:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / rel).read_text(encoding="utf-8"))["run_ids"]


def _primary(params: dict, ix: Path) -> dict:
    """primary_subset mode (qwen36_a3b): the headline numbers above are on the primary subset; ``full`` adds all
    proxy600 rows when they finished inside the index cap."""
    s = params["screen"]
    parts = json.loads((ix / "parts.json").read_text(encoding="utf-8")) if (ix / "parts.json").exists() else {}
    out = {"subset": s["primary_subset"], "expected_rows": len(_ids(s["primary_subset"])), "parts": parts}
    if (ix / "score_full.json").exists():
        sf = json.loads((ix / "score_full.json").read_text(encoding="utf-8"))
        cf = json.loads((ix / "compare_full_stock_v3w.json").read_text(encoding="utf-8"))
        ef = json.loads((ix / "ece_full.json").read_text(encoding="utf-8"))
        out["full"] = {"rows": sf["rows"], "subset_ids_sha256": sf["subset_ids_sha256"],
                       "index": sf["slice_index"]["index"], "areas": sf["slice_index"].get("areas"),
                       "vs_stock_rwkv7_v3w": {"baseline_proxy600": STOCK_V3W_PROXY600,
                                              "diff": cf.get("diff", {}).get("index"), "mde": cf.get("mde")},
                       "ece": {k: v for k, v in ef.items() if not isinstance(v, (dict, list))}}
    return out


def valid_screen(d: dict) -> bool:
    try:
        ok = (d["schema"] == SCHEMA and d["verdict"] == "measured" and isinstance(d["index"]["point"], (int, float))
              and d["selfcheck"]["ok"] is True and d["rows"] == d.get("expected_rows", 644))
        if ok and "arm_b" in d:
            b = d["arm_b"]
            ok = (isinstance(b["index"]["point"], (int, float)) and b["selfcheck"]["ok"] is True and b["rows"] == 644
                  and b["arm_a_check"]["ok"] is True and isinstance(b["diff_vs_arm_a"]["point"], (int, float)))
        return ok
    except (KeyError, TypeError):
        return False


def main(argv: list[str]) -> int:
    if argv and argv[0] == "check-arm-a":
        rep = check_arm_a(json.loads(Path(argv[1]).read_text(encoding="utf-8")),
                          json.loads(Path(argv[2]).read_text(encoding="utf-8")))
        print(json.dumps(rep))
        return 0 if rep["ok"] else 16
    params = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    d = build(params, Path(argv[1]))
    Path(argv[2]).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({"index": d["index"], "vs_stock_v3w": d["vs_stock_rwkv7_v3w"]["diff"],
                      **({"arm_b_index": d["arm_b"]["index"], "b_minus_a": d["arm_b"]["diff_vs_arm_a"]} if "arm_b" in d else {})}))
    return 0 if valid_screen(d) else 9


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
