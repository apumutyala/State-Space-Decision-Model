"""B-PREREG-GUAR-FULLDEV: split conformal + LTT (jevkit/guarantees.py, unmodified) on four full-DEV prediction files.

    python -I guarantees_fulldev.py <agent-main checkout> <out.json>

Streams every file (RAM is tight). One model at a time; only per-TEST-decision summaries are kept for the bootstrap.
"""
import collections
import gzip
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, sys.argv[1])
from harness import cases, config, slicing  # noqa: E402
from harness.suite import area_of  # noqa: E402
from jevkit import guarantees as G  # noqa: E402
from jevkit.answers import validate_distribution  # noqa: E402

H = r"C:\Users\apumu"
MODELS = {
    "D19": (H + r"\d19_fulldev_land\merged\fulldev\child.results.jsonl", "b6bab8034458b1a6b45da300f3a253141ed91b6fdb66eba9b5370c5ff8d846a8"),
    "D13": (H + r"\d13_fulldev_land\merged\fulldev\child.results.jsonl", "7b42f4c34c4d80d997a2d349b15cbcc8560571ea3f4a4e17b32d24a7360bb1dd"),
    "Clef27B": (H + r"\clef_zfexa_land\x\clef-index-full\index\decide_32k_substituted\results.jsonl", "f5ad788be191f7737dfadd1594cab7b02a428d847b0cfd9478660df8032ab09e"),
    "ClefFlash": (H + r"\clef_t5yg2_land\dl\clef-flash-index-full\index\decide_32k_substituted\results.jsonl", "2cb2e96114694b589a35c392099b2f67dc5ae0970e9939f8500f0bedf86649c5"),
}
NOGOLD = collections.Counter()
SALT = "B-PREREG-GUAR-FULLDEV|v1"
ALPHAS = (0.10, 0.05)
EPS, DELTA = 0.05, 0.10
B, SEED = 2000, 0
PAIRS = [("D19", "Clef27B"), ("D19", "ClefFlash"), ("D13", "Clef27B"), ("D13", "ClefFlash"), ("D19", "D13"),
         ("Clef27B", "ClefFlash")]


def u(key):
    return int(hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gold():
    """Stream DEV rows: gold per decision, catalog per run, case id per run (harness.cases.case_ids logic)."""
    man = [m["run_id"] for m in slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")]
    gold, cat, grp, ids = {}, {}, {}, []
    uf, groups = cases._UF(), set()
    with gzip.open(config.SLICE_ROWS_DIR / "dev-rows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            e = r["_evaluation"]
            rid = e["run_id"]
            ids.append(rid)
            cat[rid] = e["catalog_id"]
            g = ("G", e["catalog_id"], str(e["group_id"]))
            grp[rid] = (e["catalog_id"], str(e["group_id"]))
            groups.add(g)
            for k in cases.link_keys(r):
                uf.union(g, ("K",) + k)
            for key, q in r["questions"].items():
                exp = r["expected"].get(key)
                if exp is None:
                    NOGOLD[q["type"]] += 1
                    continue
                if q["type"] == "noul":
                    assert isinstance(exp, bool), (rid, key, exp)
                    gold[f"{rid}#{key}"] = ("noul", "yes" if exp else "no", ["yes", "no"])
                elif q["type"] == "choice":
                    gold[f"{rid}#{key}"] = ("choice", str(exp), [str(c) for c in q["criteria"]])
                else:
                    raise SystemExit(f"unexpected question type {q['type']}")
    assert ids == man, "materialised DEV rows do not match the manifest"
    members = collections.defaultdict(list)
    for g in groups:
        members[uf.find(g)].append(g)
    case_of_group = {}
    for comp in members.values():
        cid = f"{comp[0][1]}:{min(m[2] for m in comp)}"
        for m in comp:
            case_of_group[(m[1], m[2])] = cid
    case = {rid: case_of_group[grp[rid]] for rid in ids}
    return ids, gold, cat, case


def split(ids, case, mode):
    key = (lambda rid: rid) if mode == "run_id" else (lambda rid: "case|" + case[rid])
    return {rid: (u(key(rid)) < 0.5) for rid in ids}  # True = calibration


def load_model(path, gold):
    """{decision_id: Scored} for valid distributions; counts of invalid."""
    out, bad = {}, collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = r["run_id"]
            ans = ((r.get("response") or {}).get("answers") or {}) if r.get("status") == "ok" else {}
            del r
            for key, a in ans.items():
                did = f"{rid}#{key}"
                if did not in gold:
                    bad["not_in_dev"] += 1
                    continue
                t, y, labels = gold[did]
                try:
                    if t == "noul":
                        p = float(a["noul"])
                        if not 0.0 <= p <= 1.0:
                            raise ValueError
                        probs = {"yes": p, "no": 1.0 - p}
                    else:
                        probs, _strict, _renorm = validate_distribution(a["probabilities"], labels)
                except Exception:
                    bad["invalid"] += 1
                    continue
                out[did] = G.Scored(did, t, probs, y)
    bad["missing"] = len(set(gold) - set(out))
    return out, dict(bad)


def area_stats(test, fit, cat, amap):
    by = collections.defaultdict(list)
    for r in test:
        by[amap.get(cat[r.id.split("#", 1)[0]], "non-index")].append(r)
    return {a: G.evaluate_sets(v, fit)["all"] for a, v in sorted(by.items())}


def run_split(mode, ids, gold, cat, case, amap):
    is_cal = split(ids, case, mode)
    res, per_test = {}, {}
    test_ids = sorted(d for d in gold if not is_cal[d.split("#", 1)[0]])
    for name, (path, want) in MODELS.items():
        t0 = time.time()
        sc, bad = load_model(path, gold)
        cal = [sc[d] for d in sorted(sc) if is_cal[d.split("#", 1)[0]]]
        test = [sc[d] for d in test_ids if d in sc]
        m = {"invalid_or_missing": bad, "n_cal": len(cal), "n_test": len(test), "alpha": {}}
        arrays = {}
        for alpha in ALPHAS:
            rep = G.conformal_report(cal, test, alpha=alpha, epsilon=EPS, delta=DELTA, mondrian=True,
                                     method="bonferroni", split_seed=0, fit_frac=0.5)
            fit_half, _ltt_half, _ = G.split_calibration(cal, 0, 0.5)
            fit = G.fit_conformal(fit_half, alpha, True)
            rep["sets_per_area"] = area_stats(test, fit, cat, amap)
            lam = rep["ltt"]["lambda"]
            size, act, cov, err = [], [], [], []
            for r in test:
                s = G.prediction_set(r.probs, G._qhat_for(fit, r))
                size.append(len(s))
                cov.append(r.label in s)
                ok, y = G._acts(r, fit, lam) if lam is not None else (False, None)
                act.append(ok)
                err.append(ok and y != r.label)
            arrays[alpha] = {"size": np.array(size, float), "act": np.array(act, float), "cov": np.array(cov, float),
                             "err": np.array(err, float)}
            m["alpha"][str(alpha)] = rep
        per_test[name] = ({r.id for r in test}, [r.id for r in test], arrays)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        del sc, cal, test
        print(f"[{mode}] {name} done in {m['seconds']}s", file=sys.stderr, flush=True)
    # every model must cover the same TEST decisions for paired bootstrap
    common = set.intersection(*(v[0] for v in per_test.values()))
    order = sorted(common)
    clusters = sorted({case[d.split("#", 1)[0]] for d in order})
    cidx = {c: i for i, c in enumerate(clusters)}
    ci_of = np.array([cidx[case[d.split("#", 1)[0]]] for d in order])
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=B).astype(np.float32)

    def csum(x):
        return np.bincount(ci_of, weights=x, minlength=len(clusters))

    n_c = csum(np.ones(len(order)))
    n_b = W @ n_c

    def stat(name, alpha, field):
        ids_list, arr = per_test[name][1], per_test[name][2][alpha][field]
        pos = {d: i for i, d in enumerate(ids_list)}
        x = arr[[pos[d] for d in order]]
        return float(x.mean()), (W @ csum(x)) / n_b

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    boot = {"B": B, "seed": SEED, "n_clusters": len(clusters), "n_decisions": len(order), "per_model": {}, "pairs": {}}
    for alpha in ALPHAS:
        a = str(alpha)
        for name in MODELS:
            for field in ("size", "act", "cov"):
                p, bs = stat(name, alpha, field)
                boot["per_model"].setdefault(name, {}).setdefault(a, {})[field] = {"point": p, "ci95": ci(bs)}
        for x, y in PAIRS:
            row = {}
            for field, better in (("size", "lower"), ("act", "higher")):
                px, bx = stat(x, alpha, field)
                py, by = stat(y, alpha, field)
                lo, hi = ci(bx - by)
                if field == "size":
                    verdict = f"{x} tighter" if hi < 0 else (f"{y} tighter" if lo > 0 else "no separation")
                else:
                    verdict = f"{x} more actionable" if lo > 0 else (f"{y} more actionable" if hi < 0 else "no separation")
                row[field] = {"diff": px - py, "ci95": [lo, hi], "verdict": verdict}
            boot["pairs"].setdefault(f"{x}-{y}", {})[a] = row
    split_rec = {"mode": mode, "salt": SALT, "cal_rows": sum(is_cal.values()), "test_rows": len(ids) - sum(is_cal.values()),
                 "cal_run_ids_sha256": G.ids_sha256(r for r in ids if is_cal[r]),
                 "test_run_ids_sha256": G.ids_sha256(r for r in ids if not is_cal[r]),
                 "test_decision_ids_sha256": G.ids_sha256(test_ids), "n_test_decisions": len(test_ids)}
    return {"split": split_rec, "models": res, "bootstrap": boot}


def main():
    out_path = sys.argv[2]
    shas = {}
    for name, (p, want) in MODELS.items():
        got = sha_file(p)
        assert got == want, (name, got)
        shas[name] = got
    ids, gold, cat, case = load_gold()
    amap = area_of("0.2.1")
    print(f"gold: {len(gold)} decisions, {len(set(case.values()))} cases", file=sys.stderr, flush=True)
    out = {"prereg": "B-PREREG-GUAR-FULLDEV (state 2c683e74)", "agent_main": "7c131f1", "edition": "0.2.1",
           "epsilon": EPS, "delta": DELTA, "alphas": ALPHAS, "input_sha256": shas,
           "n_rows": len(ids), "n_decisions": len(gold), "n_cases": len(set(case.values())),
           "no_gold_decisions_excluded": dict(NOGOLD), "guarantee": G.GUARANTEE}
    for mode in ("run_id", "case"):
        out[mode] = run_split(mode, ids, gold, cat, case, amap)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
