"""Per-row cost profile of a kit readout run (results.jsonl): questions, options, input tokens vs wall time.

python -X utf8 scripts/profile_readout.py <results.jsonl> [--json out.json]

Prints one line per row and a least-squares fit of wall seconds on (input tokens, sum over questions of options x
row tokens): the second term is what an option-by-option recompute costs, the first what a shared-prefix fork costs.
"""
from __future__ import annotations

import argparse
import ast
import json

import numpy as np


def _obj(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:  # noqa: BLE001  the kit stores repr() strings
        return ast.literal_eval(x)


def profile(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        qs = r["payload"]["questions"]
        k = [len(q.get("criteria") or {}) if q["type"] == "choice" else 2 for q in qs.values()]
        raw = _obj(r.get("raw_output") or "{}")
        toks = raw.get("sequence_tokens")
        out.append({"run_id": r["run_id"], "n_questions": len(qs), "options": k, "sum_options": sum(k),
                    "max_options": max(k), "input_tokens": toks, "proxy_tokens": r["proxy_tokens"],
                    "wall_s": r["model_request_wall_ms"] / 1000})
    return out


def fit(rows):
    rows = [r for r in rows if r["input_tokens"]]
    X = np.array([[r["input_tokens"], r["input_tokens"] * r["sum_options"] / max(r["n_questions"], 1), 1.0]
                  for r in rows])
    y = np.array([r["wall_s"] for r in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return {"s_per_input_token": coef[0], "s_per_option_token": coef[1], "s_per_row": coef[2], "r2": r2,
            "effective_tok_s_input": sum(r["input_tokens"] for r in rows) / y.sum(),
            "effective_tok_s_if_K_x_prompt": sum(r["input_tokens"] * r["sum_options"] / max(r["n_questions"], 1)
                                                  for r in rows) / y.sum()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = profile(a.results)
    for r in rows:
        print(f"{r['wall_s']:8.1f}s  q={r['n_questions']:3d}  K={r['options']!s:40.40}  in={r['input_tokens']}  "
              f"proxy={r['proxy_tokens']}  {r['run_id'][:50]}")
    f = fit(rows)
    print(json.dumps(f, indent=1))
    if a.json:
        json.dump({"rows": rows, "fit": f}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
