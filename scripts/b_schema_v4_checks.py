"""bschema-draw-v4 per-family checks on the FINAL draw (new-family rows only, plus shares vs the parent).

  gold_once      every choice question: the gold id is an option and its normalised text differs from every other
                 option's text (the gold appears exactly once); noul labels in {true, false}
  gold_position  chi2 of the gold's canonical (sorted-id) position per family / question / K: all rows, the
                 schema-view half (ids re-styled: the gate, expected uniform) and the Index-presentation half (semantic
                 ids, so position = label: reported with the label rate, not gated)
  fingerprint    production data.dedup.Fingerprint (exact / ngram13 / minhash / edit / contain) on each new row's ITEM
                 fields (scripts/b_schema_levers_finalize.item_probe): must be 0 hits
  generator      harness.heldout.check_generator on each new family's spec (data.b_schema_levers_v4.GEN_SPECS)
  tokens         new rows' lengths from the lever token runs (lens-*.jsonl g1v2 / lensv1-*.jsonl v1)
  shares         item shares of every source/family in the draw vs the parent (bschema-draw-v3b)

    python -X utf8 -I scripts/b_schema_v4_checks.py --draw <records.jsonl.gz> --parent <v3b records.jsonl.gz>
        --lens-dir <work dir> --out <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
NEW = ("esci_relevance", "nli4ct_trial", "phish_email")


def p_chi2(x: float, k: int) -> float:  # Wilson-Hilferty upper tail
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def chi(d: dict) -> dict:
    x = df = n = 0
    for K, c in d.items():
        m = sum(c.values())
        if m < 5 * K:
            continue
        e = m / K
        x += sum((c.get(i, 0) - e) ** 2 / e for i in range(K))
        df += K - 1
        n += m
    return {"n": n, "chi2": round(x, 1), "df": df, "chi2_per_df": round(x / df, 2) if df else None,
            "p": round(p_chi2(x, df), 6) if df else None}


def _norm(s) -> str:
    return re.sub(r"\W+", " ", str(s).casefold()).strip()


def _jsonl(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def shares(p) -> tuple[int, int, dict]:
    rows = items = 0
    c = collections.Counter()
    for r in _jsonl(p):
        rows += 1
        items += len(r["questions"])
        c[f"{r['source']}/{r['family']}"] += len(r["questions"])
    return rows, items, c


def main(argv=None) -> int:
    from decide.schema import canonical_options
    from data import config
    from data.dedup import Fingerprint
    from data.b_schema_levers_v4 import GEN_SPECS
    from b_schema_levers_finalize import item_probe
    from train.cohort_v3 import _ensure_kit

    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--lens-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _ensure_kit()
    from harness.heldout import check_generator

    fp = Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    lens = {}
    for f in glob.glob(str(Path(a.lens_dir) / "lens-*.jsonl")) + glob.glob(str(Path(a.lens_dir) / "lensv1-*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            lens.setdefault(d["id"], {}).update(d)
    pos = {k: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for k in ("all", "view", "index")}
    labels = collections.defaultdict(collections.Counter)
    gold_once_fail, noul_fail, fp_hits = [], [], collections.Counter()
    n_new = collections.Counter()
    toks = collections.defaultdict(lambda: collections.defaultdict(list))
    items_new = collections.Counter()
    for r in _jsonl(a.draw):
        lv = ((r.get("aug") or {}).get("lever") or {}).get("lever")
        if lv not in NEW:
            continue
        n_new[lv] += 1
        items_new[lv] += len(r["questions"])
        view = bool(r["aug"]["lever"].get("view"))
        labels[lv][r["aug"]["lever"]["label"]] += 1
        hit = fp.check(*item_probe(r))
        for m, v in hit.items():
            if v:
                fp_hits[f"{lv}:{m}"] += 1
        d = lens.get(r["id"], {})
        for k in ("t_v2f", "t_v3w", "t_v1"):
            if k in d:
                toks[lv][k].append(d[k])
        for qid, q in r["questions"].items():
            g = str(r["labels"][qid])
            if q["type"] == "noul":
                if g not in ("true", "false"):
                    noul_fail.append(r["id"])
                continue
            ids = [o for o, _ in canonical_options(q)]
            texts = {o: _norm(t if t not in (None, "") else o) for o, t in q["criteria"].items()}
            if g not in ids or sum(1 for o in ids if texts[o] == texts[g]) != 1:
                gold_once_fail.append(r["id"])
                continue
            K, i = len(ids), ids.index(g)
            for half in ("all", "view" if view else "index"):
                pos[half][lv][K][i] += 1
    gp = {half: {lv: chi(d) for lv, d in sorted(v.items())} for half, v in pos.items()}
    gen = {}
    for lv, spec in GEN_SPECS.items():
        check_generator(spec)
        gen[lv] = spec

    def q(xs):
        xs = sorted(xs)
        if not xs:
            return None
        return {"n": len(xs), "p50": xs[len(xs) // 2], "p99": xs[int(0.99 * (len(xs) - 1))], "max": xs[-1],
                "n_over_2048": sum(1 for x in xs if x > 2048)}

    p_rows, p_items, p_sh = shares(a.parent)
    d_rows, d_items, d_sh = shares(a.draw)
    sh = {k: {"v3b": round(p_sh.get(k, 0) / p_items, 4), "v4": round(d_sh.get(k, 0) / d_items, 4),
              "v3b_items": p_sh.get(k, 0), "v4_items": d_sh.get(k, 0)}
          for k in sorted(set(p_sh) | set(d_sh), key=lambda k: -d_sh.get(k, 0))}
    out = {"draw": a.draw, "rows_by_family": dict(n_new), "items_by_family": dict(items_new),
           "label_counts": {k: dict(v) for k, v in labels.items()},
           "label_rates": {k: {x: round(y / sum(v.values()), 4) for x, y in v.items()} for k, v in labels.items()},
           "gold_once": {"n_fail": len(gold_once_fail), "fail_ids": gold_once_fail[:20],
                         "noul_label_fail": len(noul_fail)},
           "gold_position_chi2": gp,
           "gold_position_gate": {lv: (gp["view"].get(lv, {}).get("p") or 0) >= 0.001 for lv in NEW},
           "fingerprint_item_tier": {"version": fp.meta.get("version"), "npz_sha256": fp.meta.get("npz_sha256"),
                                     "hits": dict(fp_hits)},
           "generator_specs": gen,
           "tokens": {lv: {k: q(v) for k, v in d.items()} for lv, d in toks.items()},
           "rows": {"v3b": p_rows, "v4": d_rows}, "items": {"v3b": p_items, "v4": d_items}, "item_shares": sh}
    ok = (not gold_once_fail and not noul_fail and not fp_hits and all(out["gold_position_gate"].values())
          and all((t.get(k) or {}).get("n_over_2048", 0) == 0 for t in out["tokens"].values() for k in ("t_v2f", "t_v3w")))
    out["pass"] = ok
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("rows_by_family", "label_rates", "gold_once", "gold_position_gate",
                                          "fingerprint_item_tier", "tokens", "pass")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
