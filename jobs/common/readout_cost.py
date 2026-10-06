"""Readout cost model: exact token / call volumes of the DEV subset under each scoring path, calibrated per-token and
per-call costs, and the projected minutes the preflight uses.

Volumes are computed with the real prompt builder and tokenizer (no model): ``HFReadoutBackend.plan`` is pure
tokenisation. Three paths:
  recompute     today's HF readout: every (question, option) forwards prompt_q + continuation_o
  per_question  one forward per question row (decision-head backend ``heads.decide_backend``): sum_q |prompt_q|
  fork          decide.backends.state_fork: shared prefix once + question suffixes + option continuations

Time model per row: ``tokens / tok_s + calls * call_s``. ``tok_s`` and ``call_s`` are fitted on the measured rows of a
kit run (``scripts/profile_readout.py`` gives the wall times; this module recomputes the same rows' plans).

    python -X utf8 -m jobs.common.readout_cost volume --family rwkv7|qwen3.5 --rows <dev rows .jsonl.gz> --out <json>
    python -X utf8 -m jobs.common.readout_cost fit --family rwkv7 --results <results.jsonl> --out <json>
    python -X utf8 -m jobs.common.readout_cost project --volume <json> --tok-s X --call-s Y [--path fork]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {"rwkv7": "jrt-rwkv-g1-v1", "qwen3.5": "jrt-chatml-v1"}
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")


def encoder(family: str, tok_dir: str | None = None):
    """Tokenizer only. RWKV7-G1k uses the RWKV World vocab (ids identical to the 13.3B tokenizer.json, checked);
    Qwen3.5-0.8B and -9B tokenizer.json are byte-identical (jobs/qual/JOB_SPEC.md)."""
    if family == "rwkv7":
        from decide.rwkv_world_tokenizer import RWKVWorldTokenizer

        d = tok_dir or str(next((DATA / "hf" / "models--fla-hub--rwkv7-0.1B-g1" / "snapshots").glob("*")))
        tok = RWKVWorldTokenizer(Path(d) / "rwkv_vocab_v20230424.txt")
        return lambda text, trusted=True: tok.encode(text)
    from transformers import AutoTokenizer

    d = tok_dir or str(next((DATA / "hf" / "models--Qwen--Qwen3.5-0.8B" / "snapshots").glob("*")))
    tok = AutoTokenizer.from_pretrained(d)
    return lambda text, trusted=True: tok(text, add_special_tokens=False).input_ids


class _NoState:  # lets plan() take the fork path without a model
    pass


def backends(encode, template: str):
    from decide.backends.hf_readout import HFReadoutBackend, ModelPlugin

    plug = ModelPlugin("cost", "cost", None, encode, None, device="cpu", dtype="float32", state_ops=_NoState())
    rec = HFReadoutBackend(plug, template, max_batch_rows=16, fork=False)
    fk = HFReadoutBackend(plug, template, max_batch_rows=16, fork=True)
    fk._enc_memo = rec._enc_memo  # one tokenisation per segment
    return rec, fk


def row_volumes(request: dict, rec, fk) -> dict:
    """tokens = real tokens through the model; positions = tokens incl. the right padding of packed fork tails
    (what the GPU actually computes; the time model uses positions)."""
    from decide.backends.state_fork import fork_plan

    rows = rec._rows(request)
    p_rec, p_fk = rec.plan(request), fk.plan(request)
    prompts = {r["qid"]: len(r["prompt"]) for r in rows}
    if fk._use_fork(rows):
        _, calls = fork_plan(rows, fk.fork_max_rows, fk.fork_token_budget)
        pos = sum(c.tokens if c.kind == "prefix" else c.width * len(c.tails) for c in calls)
    else:
        pos = p_fk.total_input_tokens
    # recompute: rows of one question share a chunk only when their continuations have equal length (no padding)
    return {"recompute": {"tokens": p_rec.total_input_tokens, "calls": p_rec.inference_calls,
                          "positions": p_rec.total_input_tokens},
            "per_question": {"tokens": sum(prompts.values()) + sum(len(r["cont"]) for r in rows),
                             "calls": len(prompts),
                             "positions": sum(prompts.values()) + sum(len(r["cont"]) for r in rows)},
            "fork": {"tokens": p_fk.total_input_tokens, "calls": p_fk.inference_calls, "positions": pos},
            "n_questions": len(prompts), "n_options": len(rows), "max_prompt": max(prompts.values())}


def _requests(rows_path: str):
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield r["_evaluation"]["run_id"], {"model": "cost", "state": r["state"], "questions": r["questions"]}


def volume(family: str, rows_path: str, ids: set[str] | None = None, tok_dir: str | None = None) -> dict:
    rec, fk = backends(encoder(family, tok_dir), TEMPLATES[family])
    per = {}
    for rid, req in _requests(rows_path):
        if ids is not None and rid not in ids:
            continue
        rec._enc_memo.clear()
        try:
            per[rid] = row_volumes(req, rec, fk)
        except Exception as e:  # noqa: BLE001  rows the adapter refuses are scored zero, not forwarded
            per[rid] = {"refused": type(e).__name__}
    ok = [v for v in per.values() if "refused" not in v]
    tot = {p: {k: int(sum(v[p][k] for v in ok)) for k in ("tokens", "calls", "positions")}
           for p in ("recompute", "per_question", "fork")}
    return {"family": family, "template": TEMPLATES[family], "rows": len(per), "refused": len(per) - len(ok),
            "totals": tot, "per_row": per}


def fit(per_row: dict, results_path: str, path: str = "recompute") -> dict:
    """Least squares of measured wall seconds on (tokens, calls) of the same rows' plans."""
    X, y = [], []
    for line in open(results_path, encoding="utf-8"):
        r = json.loads(line)
        v = per_row.get(r["run_id"])
        if r.get("status") != "ok" or not v or "refused" in v:
            continue
        X.append([v[path]["tokens"], v[path]["calls"]])
        y.append(r["model_request_wall_ms"] / 1000)
    X, y = np.array(X, float), np.array(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    return {"rows": len(y), "s_per_token": float(coef[0]), "tok_s": float(1 / coef[0]) if coef[0] > 0 else None,
            "call_s": float(coef[1]), "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "max_rel_err": float(np.max(np.abs(pred - y) / y))}


#: per-call overhead fitted on job-mhc3h (RWKV7-13.3B, H100, 25 rows, r2 0.99999); used for a model whose own
#: small-call latency has not been measured (probe9b's 9B readout) -- the live row-rate guard stays authoritative
CALL_S_FITTED_RWKV7 = 0.098
TOK_S_FITTED_RWKV7 = 5360.0


def project(totals: dict, tok_s: float, call_s: float, path: str = "fork") -> dict:
    t = totals[path]
    minutes = (t.get("positions", t["tokens"]) / tok_s + t["calls"] * call_s) / 60
    return {"path": path, "tokens": t["tokens"], "positions": t.get("positions"), "calls": t["calls"],
            "tok_s": tok_s, "call_s": call_s, "projected_min": minutes}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="readout_cost")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volume")
    v.add_argument("--family", choices=sorted(TEMPLATES), required=True)
    v.add_argument("--rows", required=True)
    v.add_argument("--subset", default=str(ROOT / "jobs" / "common" / "dev_subset_m1.json"))
    v.add_argument("--tok-dir", default=None)
    v.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--volume", required=True)
    f.add_argument("--results", required=True)
    f.add_argument("--path", default="recompute")
    p = sub.add_parser("project")
    p.add_argument("--volume", required=True)
    p.add_argument("--tok-s", type=float, required=True)
    p.add_argument("--call-s", type=float, required=True)
    p.add_argument("--path", default="fork")
    a = ap.parse_args(argv)
    if a.cmd == "volume":
        ids = set(json.loads(Path(a.subset).read_text())["run_ids"])
        out = volume(a.family, a.rows, ids, a.tok_dir)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({k: out[k] for k in ("family", "rows", "refused", "totals")}, indent=1))
        return 0
    vol = json.loads(Path(a.volume).read_text())
    if a.cmd == "fit":
        print(json.dumps(fit(vol["per_row"], a.results, a.path), indent=1))
        return 0
    print(json.dumps(project(vol["totals"], a.tok_s, a.call_s, a.path), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
