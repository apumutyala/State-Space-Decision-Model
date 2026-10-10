"""KIT PARITY landing (EV-PREREG-KIT-PARITY): our end-to-end predictions vs an author's published per-item public-231 run.

    python -X utf8 -m analysis.kit_parity_compare --ref results.jsonl --pred cand/<id>/<arm>/preds.jsonl [--pred ...] \
        --ref-preds-out ref_preds.jsonl --out compare.json

* ``--ref``: a JevBench CLI results.jsonl (one line per item: task_id, correct, probs {label: p}, usage.prompt_tokens).
* Per ``--pred`` arm: input-token identity (our usage.input_tokens == the author's usage.prompt_tokens), top-1 agreement
  (jevkit's top-1 rule on both sides), argmax-label agreement, |dP| on the author's top label, per-item TVD.
* ``--ref-preds-out``: the author's per-item distributions converted to jevkit prediction records (noul -> P(yes)), so
  ``python -m jevkit score --public --pred ref_preds.jsonl`` scores the author's own outputs with the same kit.
CPU only, light (no model).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def dist_of(item, answer) -> dict[str, float] | None:
    if not isinstance(answer, dict):
        return None
    if item.type == "noul":
        p = answer.get("noul")
        return None if p is None else {"yes": float(p), "no": 1.0 - float(p)}
    pr = answer.get("probabilities")
    return {str(k): float(v) for k, v in pr.items()} if isinstance(pr, dict) else None


def ref_to_pred(item, r: dict) -> dict:
    probs = {str(k): float(v) for k, v in (r.get("probs") or {}).items()}
    if item.type == "noul":
        ans = {"type": "noul", "noul": probs.get("yes")} if probs else None
    else:
        ans = {"type": item.type, "probabilities": probs} if probs else None
    ok = r.get("status") == "ok" and ans is not None
    u = r.get("usage") or {}
    return {"id": item.id, "answer": ans, "status": "ok" if ok else "error", "error": r.get("error"),
            "usage": {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens")},
            "latency_s": r.get("latency_s"), "model": r.get("model"), "tier": item.tier}


def compare(items, ref: dict[str, dict], recs: list[dict]) -> dict:
    from jobs.jev_host_scan.host_scan import top1

    by = {r["id"]: r for r in recs}
    ours = top1(recs, items)
    theirs = top1([ref_to_pred(it, ref[it.id]) for it in items if it.id in ref], items)
    ow, tw = set(ours["wrong"]), set(theirs["wrong"])
    tok_same = tok_n = 0
    lab_same, dtop, tvd, deltas = 0, [], [], {}
    for it in items:
        r, rr = by.get(it.id), ref.get(it.id)
        if r is None or rr is None:
            continue
        rn = (rr.get("usage") or {}).get("prompt_tokens")
        on = (r.get("usage") or {}).get("input_tokens")
        if rn is not None and on is not None:
            tok_n += 1
            tok_same += int(rn == on)
            deltas[on - rn] = deltas.get(on - rn, 0) + 1
        od = dist_of(it, r.get("answer") if r.get("status") == "ok" else None)
        td = {str(k): float(v) for k, v in (rr.get("probs") or {}).items()}
        if od and td:
            ot, tt = max(od, key=od.get), max(td, key=td.get)
            lab_same += int(ot == tt)
            dtop.append(abs(od.get(tt, 0.0) - td[tt]))
            keys = set(od) | set(td)
            tvd.append(0.5 * sum(abs(od.get(k, 0.0) - td.get(k, 0.0)) for k in keys))
    return {"ours_top1": {"correct": ours["correct"], "by_tier": {k: v[0] for k, v in ours["by_tier"].items()}},
            "ref_top1_by_jevkit_rule": {"correct": theirs["correct"],
                                         "by_tier": {k: v[0] for k, v in theirs["by_tier"].items()}},
            "ref_correct_field": sum(1 for x in ref.values() if x.get("correct")),
            "top1_agreement": len(items) - len(ow ^ tw), "wrong_only_ours": sorted(ow - tw),
            "wrong_only_ref": sorted(tw - ow),
            "input_tokens_equal": f"{tok_same}/{tok_n}", "input_token_delta_hist": deltas,
            "argmax_label_agreement": f"{lab_same}/{len(dtop)}",
            "abs_dp_ref_top": {"median": statistics.median(dtop) if dtop else None, "max": max(dtop) if dtop else None},
            "tvd": {"median": statistics.median(tvd) if tvd else None, "p90": sorted(tvd)[int(0.9 * len(tvd))] if tvd else None,
                    "max": max(tvd) if tvd else None}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--ref-preds-out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from jevkit.items import load_public

    items = load_public()
    ref = {r["task_id"]: r for r in load_jsonl(a.ref)}
    missing = [it.id for it in items if it.id not in ref]
    out = {"ref": a.ref, "ref_items": len(ref), "items": len(items), "ref_missing": missing, "arms": {}}
    if a.ref_preds_out:
        with open(a.ref_preds_out, "w", encoding="utf-8", newline="\n") as fh:
            for it in items:
                if it.id in ref:
                    fh.write(json.dumps(ref_to_pred(it, ref[it.id]), ensure_ascii=False) + "\n")
    for p in a.pred:
        out["arms"][p] = compare(items, ref, load_jsonl(p))
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("ours_top1", "top1_agreement", "input_tokens_equal",
                                               "argmax_label_agreement", "tvd")} for k, v in out["arms"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
