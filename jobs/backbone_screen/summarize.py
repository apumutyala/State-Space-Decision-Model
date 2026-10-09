"""jobs/backbone_screen: decision.json from the screen's own outputs (measurement only; no pass/fail threshold).

    python -m jobs.backbone_screen.summarize <job_params.json> <out dir> <decision.json>

Reads index/score.json (proxy600 slice index), index/ece.json, index/compare_stock_v3w.json (paired bootstrap vs the
stock RWKV7-G1k v3w readout on the same 644 rows), index/selfcheck.json (fork == recompute canary) and
index/rate.json, plus the per-row statuses of index/stock/results.jsonl. ``valid_screen`` is finalize.py's validator.
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


def build(params: dict, out: Path) -> dict:
    ix = out / "index"
    score = json.loads((ix / "score.json").read_text(encoding="utf-8"))
    comp = json.loads((ix / "compare_stock_v3w.json").read_text(encoding="utf-8"))
    ece = json.loads((ix / "ece.json").read_text(encoding="utf-8"))
    selfcheck = json.loads((ix / "selfcheck.json").read_text(encoding="utf-8"))
    rate = json.loads((ix / "rate.json").read_text(encoding="utf-8")) if (ix / "rate.json").exists() else {}
    status = collections.Counter()
    for line in (ix / "stock" / "results.jsonl").open(encoding="utf-8"):
        status[json.loads(line).get("status", "?")] += 1
    s = params["screen"]
    index = score["slice_index"]["index"]
    return {
        "schema": SCHEMA, "verdict": "measured", "mode": params["mode"], "repo": s["repo"], "revision": s["revision"],
        "family": s["family"], "rows": score["rows"], "subset_ids_sha256": score["subset_ids_sha256"],
        "index": index, "areas": score["slice_index"].get("areas"),
        "vs_stock_rwkv7_v3w": {"baseline": "art-ivu28 v3w (job-fgu2c)", "baseline_proxy600": STOCK_V3W_PROXY600,
                               "diff": comp.get("diff", {}).get("index"), "mde": comp.get("mde")},
        "ece": {k: v for k, v in ece.items() if not isinstance(v, (dict, list))} or ece,
        "row_status": dict(status),
        "selfcheck": {k: selfcheck.get(k) for k in ("ok", "max_abs_delta", "same_argmax", "options", "tol")},
        "rate": {k: rate.get(k) for k in ("rows_done", "elapsed_min", "startup_s", "exit")},
    }


def valid_screen(d: dict) -> bool:
    try:
        return (d["schema"] == SCHEMA and d["verdict"] == "measured" and isinstance(d["index"]["point"], (int, float))
                and d["selfcheck"]["ok"] is True and d["rows"] == 644)
    except (KeyError, TypeError):
        return False


def main(argv: list[str]) -> int:
    params = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    d = build(params, Path(argv[1]))
    Path(argv[2]).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({"index": d["index"], "vs_stock_v3w": d["vs_stock_rwkv7_v3w"]["diff"]}))
    return 0 if valid_screen(d) else 9


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
