"""SHAPE gate (JEV1_AUDIT.md fix 7): does each covered family's eval request SHAPE lie inside the draw's support?

Schema statistics only, never content. Per family the gate compares: the share of plain-string states, shim-token
quantiles, the question-type set, choice option counts K, and whether noul criteria are filled.

  public  build the public-items shape fingerprint (statistics only; item text is never written):
            python -X utf8 -I scripts/jev2_shape_gate.py public --shim-repo <sizecliff checkout> --out <fp.json>
  check   compare one draw (records + its shim lens file) with the fingerprint:
            python -X utf8 -I scripts/jev2_shape_gate.py check --fp <fp.json> --records <records.jsonl.gz>
                --lens <lens.jsonl> --out <shape-gate.json>

Rules, applied to each family present in both the draw and the fingerprint:
  types     every public question type occurs in the draw family
  K         every public choice K occurs in the draw family (choice rows)
  criteria  public noul criteria are always filled -> >= 90% of the draw's noul rows have filled criteria
  string    public states are >= 50% plain strings -> >= 40% of the draw's states are plain strings
  length    draw p90 shim tokens >= public p90 (the audit's support rule); the max is reported beside it
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def q(xs, f):
    xs = sorted(xs)
    return xs[int(f * (len(xs) - 1))] if xs else None


def stats(rows):
    """rows: (state, question, t_shim) -> schema statistics."""
    n = len(rows)
    types = collections.Counter(qq["type"] for _, qq, _ in rows)
    ks = collections.Counter(len(qq.get("criteria") or {}) for _, qq, _ in rows if qq["type"] == "choice")
    nl = [qq for _, qq, _ in rows if qq["type"] == "noul"]
    toks = [t for _, _, t in rows if t is not None]
    return {"n": n, "string_state_share": round(sum(isinstance(s, str) for s, _, _ in rows) / n, 4) if n else None,
            "types": dict(types), "choice_K": {str(k): v for k, v in sorted(ks.items())},
            "noul_criteria_filled_share": round(sum(bool(x.get("criteria")) for x in nl) / len(nl), 4) if nl else None,
            "tokens": {"p10": q(toks, 0.1), "p50": q(toks, 0.5), "p90": q(toks, 0.9), "max": max(toks) if toks else None}}


def cmd_public(a) -> int:
    sys.path.insert(0, str(Path(a.shim_repo).resolve()))
    from decide.shim_format import bind_template
    from jevkit.items import load_public
    spec = bind_template("jrt-h2o-shim:qwen3.5-4b")
    by = collections.defaultdict(list)
    for it in load_public():
        t = len(spec.row({"state": it.state, "questions": {"q": it.question}}, "q").ids) + 1
        by[it.family].append((it.state, it.question, t))
    fp = {"what": "JevBench v1.2 public items (eval-only): per-family SHAPE statistics only (no text)",
          "template": spec.identity(), "families": {f: stats(v) for f, v in sorted(by.items())}}
    Path(a.out).write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({f: v["n"] for f, v in fp["families"].items()}))
    return 0


def cmd_check(a) -> int:
    fp = json.loads(Path(a.fp).read_text(encoding="utf-8"))["families"]
    lens = {}
    for line in open(a.lens, encoding="utf-8"):
        d = json.loads(line)
        if "id" in d and "t_shim" in d:
            lens[d["id"]] = d["t_shim"]
    by = collections.defaultdict(list)
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            by[r["aug"]["jev1"]["family8"]].append((r["state"], r["questions"]["q"], lens.get(r["id"])))
    out, allok = {}, True
    for f, rows in sorted(by.items()):
        s = stats(rows)
        p = fp.get(f)
        if p is None:
            out[f] = {"draw": s, "public": None, "checks": {}, "pass": None}
            continue
        ch = {"types": set(p["types"]) <= set(s["types"]),
              "K": set(p["choice_K"]) <= set(s["choice_K"]),
              "criteria": p["noul_criteria_filled_share"] is None or p["noul_criteria_filled_share"] < 1 or
              (s["noul_criteria_filled_share"] or 0) >= 0.9,
              "string": (p["string_state_share"] or 0) < 0.5 or (s["string_state_share"] or 0) >= 0.4,
              "length_p90": (s["tokens"]["p90"] or 0) >= (p["tokens"]["p90"] or 0)}
        ok = all(ch.values())
        allok &= ok
        out[f] = {"draw": s, "public": p, "checks": ch, "pass": ok}
    rep = {"records": a.records, "fingerprint": a.fp, "families": out, "all_pass": allok}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({f: (v["pass"], {k: x for k, x in v["checks"].items() if not x}) for f, v in out.items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("public")
    p.add_argument("--shim-repo", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fp", required=True)
    c.add_argument("--records", required=True)
    c.add_argument("--lens", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"public": cmd_public, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
