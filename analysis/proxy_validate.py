"""Validate the pre-declared DEV proxy designs (jobs/common/dev_proxy.py) against the 2,008-row m1 subset.

Inputs: a JSON {system name: results path} (``.jsonl`` with run_id per line, or a units ``.json`` holding
``index_results``), every file treated as untrusted data (run with ``python -X utf8 -I``). Every system is decomposed
ONCE on the 2,008 rows with the frozen scorer (``harness.score.decompose``, case clusters); a proxy's Index is the
same value function on the summed statistics of its (whole) cases, i.e. exactly what ``dev_subset score --subset
<proxy>`` returns (checked against a direct decompose of the proxy rows for every system, below).

Per (design, size): the declared seed (``dev_proxy.SEED``) plus ``--seeds`` alternative seeds (design-level spread,
NOT a search: no seed is ever picked). Metrics vs the full subset: Spearman / Kendall of the Index over systems, mean
and max |dIndex|, and on system PAIRS whose full-subset gap is 1-5 points (the cell-screening regime): sign agreement,
mean |proxy gap - full gap|, the proxy's paired 95% CI (frozen ``harness.bootstrap`` draws) covering the full gap, and
the paired CI resolution (excludes 0) on the proxy vs the full subset; the proxy's own Index CI width.
Leave-one-system-out: for each held-out system h, the design is chosen at each size by the mean over seeds of the
pair metric (mean |gap error|, pairs NOT involving h) and scored on h alone (|dIndex| of h, the pairs involving h).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


def read_results(path: str, ids: set[str]) -> dict:
    p = Path(path)
    out = {}
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d["index_results"].items():
            if k in ids:
                out[k] = dict(v, run_id=k)
    else:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("run_id") in ids:
                        if r["run_id"] in out:
                            raise SystemExit(f"{path}: duplicate {r['run_id']}")
                        out[r["run_id"]] = r
    if set(out) != ids:
        raise SystemExit(f"{path}: {len(ids - set(out))} subset rows missing")
    return out


def _ranks(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):  # average ranks for ties
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def rankcorr(a, b):
    """(Spearman rho, Kendall tau-b) without scipy."""
    ra, rb = _ranks(a), _ranks(b)
    sp = float(np.corrcoef(ra, rb)[0, 1])
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    n0 = len(i)
    ta, tb = np.sum(da == 0), np.sum(db == 0)
    kt = float(np.sum(da * db) / np.sqrt((n0 - ta) * (n0 - tb)))
    return sp, kt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--sizes", default="400,600,800")
    ap.add_argument("--seeds", type=int, default=100, help="alternative seeds per design (spread only)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--pair-lo", type=float, default=1.0)
    ap.add_argument("--pair-hi", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from harness import bootstrap as HB
    from harness import score as S
    from jobs.common import dev_proxy as P

    t0 = time.time()
    rows = P.subset_rows()
    ids = {r["_evaluation"]["run_id"] for r in rows}
    cases = P.case_table(rows)
    weights = S.weights(edition="0.2.1")
    systems = json.loads(Path(a.systems).read_text(encoding="utf-8"))
    names = sorted(systems)
    benches = {}
    for s in names:
        res = read_results(systems[s], ids)
        benches[s] = S.decompose(rows, res, edition="0.2.1")
        print(f"decomposed {s} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    bn = sorted(benches[names[0]])
    for s in names:
        assert sorted(benches[s]) == bn and all(benches[s][n].clusters == benches[names[0]][n].clusters for n in bn)
    clusters = {n: benches[names[0]][n].clusters for n in bn}
    case_rows = {(n, c): len(v["rows"]) for (n, c), v in cases.items()}
    full = {s: S.point(benches[s])["index"] for s in names}

    def idx_from_masks(masks):  # masks: {n: (K, n_clusters) 0/1} -> {system: (K,) Index}
        out = {}
        for s in names:
            sk = {n: benches[s][n].value(masks[n] @ benches[s][n].S)[1] for n in bn}
            out[s] = S.index_from_skills(sk, edition="0.2.1")
        return out

    def masks_for(idsets):
        K = len(idsets)
        m = {n: np.zeros((K, len(clusters[n]))) for n in bn}
        pos = {n: {c: i for i, c in enumerate(clusters[n])} for n in bn}
        rid_case = {}
        for (n, c), v in cases.items():
            for r in v["rows"]:
                rid_case[r] = (n, c)
        for k, sel in enumerate(idsets):
            for r in sel:
                n, c = rid_case[r]
                m[n][k, pos[n][c]] = 1.0
        return m

    pairs = [(x, y) for x, y in itertools.combinations(names, 2) if a.pair_lo <= abs(full[x] - full[y]) <= a.pair_hi]
    seeds = [P.SEED] + [f"alt-{i}" for i in range(a.seeds)]
    sizes = [int(x) for x in a.sizes.split(",")]
    report = {"systems": {s: {"path": systems[s], "full_index": full[s]} for s in names},
              "n_pairs_1_5": len(pairs), "pairs": [[x, y, full[x] - full[y]] for x, y in pairs],
              "declared_seed": P.SEED, "alt_seeds": a.seeds, "designs": {}}
    per = {}  # (design, size) -> {"idx": {s: (K,)}, "ids": [...]}
    for size in sizes:
        for d in P.DESIGNS:
            idsets = [P.build_proxy(cases, weights, size, d, sd) for sd in seeds]
            idx = idx_from_masks(masks_for(idsets))
            per[(d, size)] = {"idx": idx, "ids": idsets}
            M = np.array([idx[s] for s in names])  # (systems, K)
            F = np.array([full[s] for s in names])
            sp, kt = zip(*[rankcorr(M[:, k], F) for k in range(len(seeds))])
            dabs = np.abs(M - F[:, None])
            gerr = np.array([[abs((idx[x][k] - idx[y][k]) - (full[x] - full[y])) for k in range(len(seeds))]
                             for x, y in pairs]) if pairs else np.zeros((0, len(seeds)))
            sign = np.array([[np.sign(idx[x][k] - idx[y][k]) == np.sign(full[x] - full[y]) for k in range(len(seeds))]
                             for x, y in pairs]) if pairs else np.zeros((0, len(seeds)))
            nrows = [len(x) for x in idsets]

            def dist(v):
                v = np.asarray(v, dtype=float)
                return {"declared": float(v[0]), "median": float(np.median(v[1:])), "p10": float(np.quantile(v[1:], .1)),
                        "p90": float(np.quantile(v[1:], .9))}

            report["designs"][f"{d}@{size}"] = {
                "rows": dist(nrows), "spearman": dist(sp), "kendall": dist(kt),
                "mean_abs_dindex": dist(dabs.mean(axis=0)), "max_abs_dindex": dist(dabs.max(axis=0)),
                "pair_sign_agree": dist(sign.mean(axis=0)), "pair_mean_abs_gap_err": dist(gerr.mean(axis=0)),
                "pair_abs_gap_err_pooled_alt_seeds": ({q: float(np.quantile(gerr[:, 1:], float(q)))
                                                       for q in ("0.5", "0.8", "0.9", "0.95")} if pairs else None),
                "declared_per_system_dindex": {s: float(M[i, 0] - F[i]) for i, s in enumerate(names)}}
            print(f"{d}@{size}: rows {nrows[0]} sp {sp[0]:.3f} |d| {dabs.mean(axis=0)[0]:.2f} "
                  f"sign {sign.mean(axis=0)[0] if len(pairs) else float('nan'):.2f} ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)

    # leave-one-system-out design choice per size (pair metric over the other systems' pairs, mean over all seeds)
    loso = {}
    for size in sizes:
        rows_out = []
        for h in names:
            other = [(x, y) for x, y in pairs if h not in (x, y)]
            mine = [(x, y) for x, y in pairs if h in (x, y)]

            def crit(d):
                idx = per[(d, size)]["idx"]
                return float(np.mean([abs((idx[x] - idx[y]) - (full[x] - full[y])).mean() for x, y in other]))

            best = min(P.DESIGNS, key=crit)
            idx = per[(best, size)]["idx"]
            rows_out.append({"held_out": h, "chosen": best,
                             "abs_dindex_declared": float(abs(idx[h][0] - full[h])),
                             "abs_dindex_mean_seeds": float(np.mean(np.abs(idx[h][1:] - full[h]))),
                             "pair_sign_agree_declared": (float(np.mean([np.sign(idx[x][0] - idx[y][0]) == np.sign(full[x] - full[y])
                                                                         for x, y in mine])) if mine else None),
                             "pair_abs_gap_err_declared": (float(np.mean([abs((idx[x][0] - idx[y][0]) - (full[x] - full[y]))
                                                                          for x, y in mine])) if mine else None)})
        ch = [r["chosen"] for r in rows_out]
        loso[str(size)] = {"chosen_counts": {d: ch.count(d) for d in P.DESIGNS},
                           "heldout_mean_abs_dindex_declared": float(np.mean([r["abs_dindex_declared"] for r in rows_out])),
                           "heldout_mean_abs_dindex_seeds": float(np.mean([r["abs_dindex_mean_seeds"] for r in rows_out])),
                           "heldout_pair_sign_agree": float(np.nanmean([r["pair_sign_agree_declared"] for r in rows_out
                                                                         if r["pair_sign_agree_declared"] is not None])),
                           "heldout_pair_abs_gap_err": float(np.nanmean([r["pair_abs_gap_err_declared"] for r in rows_out
                                                                          if r["pair_abs_gap_err_declared"] is not None])),
                           "per_system": rows_out}
    report["loso"] = loso

    # bootstrap (frozen harness.bootstrap draws) for the declared seed of every (design, size) + the full subset
    def boot(sel_ids):
        bl = []
        for s in names:
            if sel_ids is None:
                bl.append(benches[s])
                continue
            m = masks_for([sel_ids])
            sub = {}
            for n in bn:
                keep = m[n][0] > 0
                src = benches[s][n]
                b = S.Bench(n, [c for c, k in zip(clusters[n], keep) if k], src.S[keep],
                            src.spec if src.spec is not None else src.fn, src.kind, "0.2.1")
                sub[n] = b
            bl.append(sub)
        reps = HB.replicate_skills(bl, B=a.B, seed=0)
        idx = {s: S.index_from_skills({n: v[1] for n, v in reps[i].items()}, edition="0.2.1") for i, s in enumerate(names)}
        width = {s: float(np.quantile(idx[s][1:], .975) - np.quantile(idx[s][1:], .025)) for s in names}
        pr = {}
        for x, y in pairs:
            dd = idx[x] - idx[y]
            lo, hi = np.quantile(dd[1:], [.025, .975])
            pr[f"{x}|{y}"] = {"point": float(dd[0]), "lo": float(lo), "hi": float(hi), "full": full[x] - full[y],
                              "covers_full": bool(lo <= full[x] - full[y] <= hi), "excludes_zero": bool(lo > 0 or hi < 0)}
        return {"median_ci_width": float(np.median(list(width.values()))), "ci_width": width, "pairs": pr,
                "pair_ci_covers_full": float(np.mean([v["covers_full"] for v in pr.values()])) if pr else None,
                "pair_ci_excludes_zero": float(np.mean([v["excludes_zero"] for v in pr.values()])) if pr else None,
                "median_pair_ci_width": float(np.median([v["hi"] - v["lo"] for v in pr.values()])) if pr else None}

    report["bootstrap"] = {"B": a.B, "full_2008": boot(None)}
    for (d, size), v in per.items():
        report["bootstrap"][f"{d}@{size}"] = boot(v["ids"][0])
        print(f"boot {d}@{size} ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    # exactness: proxy Index via case masks == a direct decompose of the proxy rows (frozen scorer), every system
    chk = {}
    for s in names:
        res = read_results(systems[s], ids)
        for (d, size), v in per.items():
            if size != 600:
                continue
            sel = set(v["ids"][0])
            prow = [r for r in rows if r["_evaluation"]["run_id"] in sel]
            direct = S.point(S.decompose(prow, {k: res[k] for k in sel}, edition="0.2.1"))["index"]
            chk[f"{d}@600|{s}"] = abs(direct - float(v["idx"][s][0]))
    report["exactness_max_abs_err"] = max(chk.values()) if chk else None
    report["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("n_pairs_1_5", "exactness_max_abs_err", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
