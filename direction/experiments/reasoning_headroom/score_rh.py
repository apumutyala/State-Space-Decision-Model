"""Score the reasoning-headroom probe with the frozen jevkit CLI (EVAL-ONLY public items).

Usage: python score_rh.py <results_dir>   (run from the repo root with SSDM_DATA set)
Writes <results_dir>/<model>.<arm>.<variant>.preds.jsonl and .report.json; prints a summary table.
Variants: shim = the H2O shim post-processing (softmax(lp/0.8), yes/no floor 0.801); raw = softmax(lp), no floor.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import subprocess
import sys

# board v1.6.1 frozen 25 Sep references, (USD/M input, USD/M output): deepinfra Qwen3.5 0.8B/2B/4B, qwen/qwen3.5-9b
PRICE = {"Qwen3.5-2B": (0.02, 0.10), "Qwen3.5-4B": (0.03, 0.15), "Qwen3.5-9B": (0.10, 0.15),
         "h2o-lightning-4b": (0.03, 0.15)}


def probs(lp, t):
    v = [x / t for x in lp]
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def answer(typ, names, lp, t, floor):
    p = probs(lp, t)
    if typ == "noul":
        py = p[0]
        if floor and max(py, 1 - py) < floor:
            py = floor if py >= 0.5 else 1 - floor
        return {"type": "noul", "noul": py}
    if typ == "score":
        return {"type": "score", "score": sum(i * x for i, x in enumerate(p)),
                "probabilities": {str(i): x for i, x in enumerate(p)}}
    k = max(range(len(p)), key=p.__getitem__)
    return {"type": "choice", "choice": names[k], "probabilities": dict(zip(names, p))}


def main(d: str) -> None:
    d = pathlib.Path(d)
    items = {json.loads(l)["id"]: json.loads(l) for l in open("/tmp/ssdm-scratch/cos/public231.jsonl", encoding="utf-8")}
    summary = []
    for f in sorted(d.glob("*.json")):
        if f.name.count(".") > 2 or f.name.startswith("_"):
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        mname = res["model"].split("/")[-1]
        for arm, a in res["arms"].items():
            lat = res.get("latency", {}).get(arm, {})
            for var, (t, floor) in {"shim": (0.8, 0.801), "raw": (1.0, 0.0)}.items():
                pp = d / f"{mname}.{arm}.{var}.preds.jsonl"
                with open(pp, "w", encoding="utf-8") as w:
                    for r in a["items"]:
                        w.write(json.dumps({"id": r["id"], "status": "ok",
                                            "answer": answer(r["type"], r["names"], r["logprobs"], t, floor),
                                            "latency_s": lat.get(r["id"]),
                                            "usage": {"input_tokens": r["input_tokens"],
                                                      "output_tokens": r["output_tokens"]}}) + "\n")
                pin, pout = PRICE[mname]
                price = (f"kind=base_model_reference,in={pin},out={pout},source=ssdm-scorecard-class,"
                         f"as_of=2026-10-14")
                rp = d / f"{mname}.{arm}.{var}.report.json"
                cmd = [sys.executable, "-X", "utf8", "-m", "jevkit", "score", "--public", "--pred", str(pp),
                       "--price", price, "--out", str(rp)]
                subprocess.run(cmd, check=True, capture_output=True)
                rep = json.loads(rp.read_text(encoding="utf-8"))
                summary.append((mname, arm, var, rep))
    print(f"{'model':12s} {'arm':8s} {'var':5s} {'I':>6s} {'C':>6s} {'S':>6s} {'K':>6s} {'CompA':>6s} {'tok_in':>7s} {'tok_out':>7s} {'p50raw':>7s}")
    for mname, arm, var, rep in summary:
        ax = rep.get("axes") or {}
        sp = rep.get("speed") or {}
        co = rep.get("cost") or {}
        sc = rep.get("scores") or {}
        res = json.loads((d / f"{mname}.json").read_text(encoding="utf-8"))
        its = res["arms"][arm]["items"]
        tin = sum(r["input_tokens"] for r in its) / len(its)
        tout = sum(r["output_tokens"] for r in its) / len(its)
        def f(x):
            return f"{x:6.1f}" if isinstance(x, (int, float)) else f"{str(x)[:6]:>6s}"
        print(f"{mname:12s} {arm:8s} {var:5s} {f(ax.get('intelligence'))} {f(ax.get('calibration'))} "
              f"{f(ax.get('speed'))} {f(ax.get('cost'))} {f(sc.get('A'))} {tin:7.0f} {tout:7.0f} "
              f"{f(sp.get('p50_s_raw'))}")
    # paired analysis per model: think vs nothink (shim variant), same items, same bootstrap draws
    sys.path.insert(0, ".")
    from jevkit import items as IT
    from jevkit import scoring as SC

    pub = IT.load_public()
    fam = {k: v["family"] for k, v in items.items()}
    for mname in sorted({s[0] for s in summary}):
        reps = {(a, v): r for m, a, v, r in summary if m == mname}
        tabs, sc = {}, {}
        for arm in ("nothink", "think"):
            preds = [json.loads(l) for l in open(d / f"{mname}.{arm}.shim.preds.jsonl", encoding="utf-8")]
            tabs[arm] = SC.build_table(pub, {p["id"]: p["answer"] for p in preds})
            ax = reps[(arm, "shim")]["axes"]
            sc[arm] = (ax.get("speed"), ax.get("cost"))
        bs = SC.bootstrap({"think": tabs["think"], "nothink": tabs["nothink"]}, sc)
        pair = bs["_pairs"]["think - nothink"]
        print(f"\n{mname}: think - nothink Composite A diff 95% CI {pair['composite_A_diff_ci95']}, "
              f"P(think better) {pair['p_first_better']:.3f}; I CI think {bs['think']['I']} nothink {bs['nothink']['I']}")
        t0, t1 = tabs["nothink"], tabs["think"]
        by = collections.defaultdict(lambda: [0, 0, 0, 0])   # n, nothink correct, think correct, net flips
        for i, iid in enumerate(t0.ids):
            for key in (f"tier:{SC.spec.TIERS[t0.tier[i]]}", f"fam:{fam[iid]}"):
                b = by[key]
                b[0] += 1
                b[1] += int(t0.correct[i])
                b[2] += int(t1.correct[i])
                b[3] += int(t1.correct[i] and not t0.correct[i]) - int(t0.correct[i] and not t1.correct[i])
        for key in sorted(by, key=lambda k: (not k.startswith("tier"), k)):
            n, c0, c1, net = by[key]
            print(f"  {key:24s} n={n:3d} nothink {c0:3d} think {c1:3d} net {net:+d}")


if __name__ == "__main__":
    main(sys.argv[1])
