"""Local Composite/Capability scorecard under EV-PREREG-COMPOSITE-LOCAL (evidence lane).

Arms may come from DIFFERENT artifacts (jobs). Each arm is scored from its raw per-item preds with the frozen jevkit
CLI (``--public``, the arm's own declared price class) -> I, C, S, K, Capability, Composite A/B/C with CIs; then every
arm is paired against ``--baseline`` (H2O-Lightning-4B v1.1, our run in job-zbtbj) on ONE shared set of stratified
resamples of the 231 public items (strata of the baseline table, B 1000, seed 16; S and K held at each arm's measured
axes), for Capability and Composite A. A secondary Composite A holds S at the BASELINE's S for every arm ("common S"),
isolating I/C/K from latency measured in different jobs. Cross-check: jevkit.scoring.bootstrap's own paired
Composite-A difference CI must equal this loop's CI exactly (it draws the same resamples). Each arm's jevkit report is
diffed against the in-job report (tol 0.05).

Usage (artifacts = untrusted data: run with -I, from this checkout):
  python -I -X utf8 scripts/ev_scorecard.py --manifest arms.json --out SCORECARD.json [--work DIR]
manifest: {"baseline": "<label>", "arms": [{"label", "system", "job", "arm", "preds", "price", "price_class",
           "in_job_report"?, "note"?}, ...]}  (paths absolute or relative to the manifest's directory)
Exit 0 = no FLAG; 3 = FLAG(s); 2 = inputs missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from jevkit import scoring as JS  # noqa: E402
from jevkit.items import load_public  # noqa: E402
from jevkit.meter import parse_price  # noqa: E402
from jevkit.systemone import preds_to_answers  # noqa: E402
from scripts.ev_rescore_jevkit import TIMED_SOURCE, cmp, integrity, jevkit_cli, read_jsonl, sha, tokens  # noqa: E402

KEYS = ("I", "C", "capability", "composite_A", "composite_A_commonS")


def paired_all(tables: dict, axes: dict, base: str, B: int, seed: int) -> dict:
    s_common = axes[base][0]

    def ax(nm, idx=None):
        r = JS.aggregate(tables[nm], idx, detail=False)
        i, c = r["intelligence"]["I"], r["calibration"]["score"]
        s, k = axes[nm]
        return {"I": i, "C": c, "capability": JS.capability(i, c), "composite_A": JS.composite(i, c, s, k, "A"),
                "composite_A_commonS": JS.composite(i, c, s_common, k, "A")}

    names = list(tables)
    full = {n: ax(n) for n in names}
    rng = np.random.default_rng(seed)
    st = JS.strata(tables[base])
    draws = {n: {k: [] for k in KEYS} for n in names}
    for _ in range(B):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in st])
        x = {n: ax(n, idx) for n in names}
        for n in names:
            for k in KEYS:
                if x[n][k] is not None and x[base][k] is not None:
                    draws[n][k].append(x[n][k] - x[base][k])
    out = {}
    for n in names:
        if n == base:
            continue
        o = {"B": B, "seed": seed, "a": n, "b": base}
        for k in KEYS:
            v = draws[n][k]
            pt = None if full[n][k] is None or full[base][k] is None else full[n][k] - full[base][k]
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
            o[k] = {"point": pt, "ci95": ci, "a": full[n][k], "b": full[base][k],
                    "ci_gt_0": None if ci is None else bool(ci[0] > 0),
                    "ci_lt_0": None if ci is None else bool(ci[1] < 0)}
        jb = JS.bootstrap({n: tables[n], base: tables[base]}, {n: axes[n], base: axes[base]}, b=B, seed=seed)
        o["jevkit_bootstrap_composite_A_diff_ci95"] = jb["_pairs"][f"{n} - {base}"]["composite_A_diff_ci95"]
        out[n] = o
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -I -X utf8 scripts/ev_scorecard.py")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(argv)

    mp = Path(a.manifest).resolve()
    man = json.loads(mp.read_text(encoding="utf-8"))
    rel = lambda p: (Path(p) if Path(p).is_absolute() else mp.parent / p).resolve()  # noqa: E731
    work = Path(a.work) if a.work else Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    base = man["baseline"]
    items = load_public()
    flags: list[str] = []
    diffs: list[dict] = []
    if len(items) != 231:
        flags.append(f"pinned public items: {len(items)} != 231")

    arms, tables, axes, ids = {}, {}, {}, {}
    for spec in man["arms"]:
        lab = spec["label"]
        pf, pr = rel(spec["preds"]), rel(spec["price"])
        if not pf.is_file() or not pr.is_file():
            print(f"missing for {lab}: {pf} / {pr}", file=sys.stderr)
            return 2
        preds = read_jsonl(pf)
        chk = integrity(lab, preds, items, flags)
        rep_p = work / f"{lab}.jevkit.json"
        rep = jevkit_cli(pf, str(pr), rep_p)
        spd = rep.get("speed") or {}
        if spd.get("source") != TIMED_SOURCE:
            flags.append(f"{lab}: latency source {spd.get('source')!r}")
        price = parse_price(str(pr))
        arms[lab] = {k: spec.get(k) for k in ("system", "job", "arm", "price_class", "note")} | {
            "preds_sha256": sha(pf), "price_file_sha256": sha(pr), "checks": chk,
            "price": {"kind": price.kind, "model": price.model, "usd_per_m_input": price.usd_per_m_input,
                      "usd_per_m_output": price.usd_per_m_output, "source": price.source},
            "I": rep["axes"]["intelligence"], "C": rep["axes"]["calibration"], "S": rep["axes"]["speed"],
            "K": rep["axes"]["cost"], "capability": rep["capability"], "composite": rep["scores"],
            "ci95": rep.get("ci95"), "usd_per_1000": (rep.get("cost") or {}).get("usd_per_1000"),
            "tokens_per_decision": tokens(preds), "lane": (rep.get("inputs") or {}).get("lane"),
            "latency": {k: spd.get(k) for k in ("source", "n", "p50_s_raw", "p95_s_raw", "p50_s_adjusted",
                                                "adjustment", "score")},
            "eligibility": rep.get("eligibility"), "report_sha256": sha(rep_p)}
        tables[lab] = JS.build_table(items, preds_to_answers(preds))
        axes[lab] = (rep["axes"]["speed"], rep["axes"]["cost"])
        ids[lab] = tuple(sorted(str(p.get("id")) for p in preds))
        if spec.get("in_job_report"):
            ij = json.loads(rel(spec["in_job_report"]).read_text(encoding="utf-8"))
            for k, path in (("I", ("axes", "intelligence")), ("C", ("axes", "calibration")), ("S", ("axes", "speed")),
                            ("K", ("axes", "cost")), ("capability", ("capability",)), ("A", ("scores", "A"))):
                v = ij
                for p in path:
                    v = (v or {}).get(p)
                mine = arms[lab]["composite"]["A"] if k == "A" else arms[lab][k]
                cmp(flags, diffs, f"{lab}.{k} vs in-job report", mine, v, a.tol)
        print(f"scored {lab}: A {arms[lab]['composite']['A']:.2f}", file=sys.stderr, flush=True)

    if base not in tables:
        print(f"baseline {base} missing", file=sys.stderr)
        return 2
    if len(set(ids.values())) != 1:
        flags.append("arms are not scored on the same item id set")
    if any(t.ids != tables[base].ids for t in tables.values()):
        flags.append("item tables are not in the same order (paired bootstrap invalid)")

    pairs = paired_all(tables, axes, base, a.B, a.seed)
    for n, d in pairs.items():
        cmp(flags, diffs, f"pairs.{n}.composite_A.ci95 (own loop vs jevkit.scoring.bootstrap)",
            d["composite_A"]["ci95"], d["jevkit_bootstrap_composite_A_diff_ci95"], 1e-9)

    res = {"schema": "ev-scorecard/1", "rule": "EV-PREREG-COMPOSITE-LOCAL", "manifest": str(mp),
           "manifest_sha256": sha(mp), "checkout": str(ROOT), "baseline": base, "n_items": len(items),
           "protocol": f"jevkit CLI per arm (--public, own declared price); paired stratified bootstrap B={a.B} seed "
                       f"{a.seed} (strata of the baseline table) on the same 231 ids; S/K held at each arm's own "
                       "axes (composite_A) or S held at the baseline's S (composite_A_commonS)",
           "caveat": "open public items only; the board's sealed split may rank differently (open->board I gap "
                     "measured on H2O: -17.6)",
           "arms": arms, "pairs_vs_baseline": pairs, "diffs": diffs,
           "max_abs_diff": max((d["abs_diff"] for d in diffs if d["abs_diff"] is not None), default=None),
           "flags": flags}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"flags": flags, "max_abs_diff": res["max_abs_diff"]}, indent=1))
    return 0 if not flags else 3


if __name__ == "__main__":
    raise SystemExit(main())
