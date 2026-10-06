"""Official point estimates on any row subset, and a cluster-additive decomposition of them.

`official()` runs the unmodified kit scorer (`decision_index.scoring`) on a slice.

`decompose()` re-expresses every index benchmark's 0.2.1 value as a function of per-cluster
sufficient statistics (cluster = linked case group, i.e. one record family). Summing the
statistics over clusters with weight 1 reproduces the official value (checked in the tests);
summing with multinomial bootstrap weights gives a cluster-bootstrap replicate in one
matrix product, which is what `harness.bootstrap` does. Every branch below names the kit
code path it mirrors; any change to the kit at a new commit must be re-checked by
`tests/test_scorer_parity.py`.
"""

import collections

import numpy as np

from decision_index.scoring import added as added_mod
from decision_index.scoring import index02
from decision_index.scoring.index import chance_baselines, score_panel
from decision_index.scoring.metrics import random_ndcg_baseline, score_query, semantic_label
from decision_index.scoring.report import benchmark_summary

from harness.suite import RowSuite, spec

HEADLINE_40 = "iSarcasmEval-A-En"
FIELD_ACC = {3, 28, 29, 32, 45}
MACRO_F1 = {4, 5, 12, 39, 42}
CASE_EXACT = {9, 33}
CONSERVATIVE = {11, 37, 41}


# ----------------------------------------------------------------------------- official


def official(rows, results):
    """Unrounded per-benchmark values and index exactly as `index02.index_entry` builds them."""
    s = spec()
    suite = RowSuite(rows)
    added_ids = {int(n) for n in s["added"]}
    base, extra = [], collections.defaultdict(list)
    for r in rows:
        n = r["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(r)
    metrics = {int(n): (m["name"], m["key"]) for n, m in s.get("metrics", {}).items()}
    summary = benchmark_summary(suite, results, "harness", None, rows=base, metrics=metrics)
    added_reports = {n: added_mod.report(n, rs, results) for n, rs in sorted(extra.items())}
    scored = score_panel(suite, results)
    native = {b["catalog_id"]: b for b in summary["benchmarks"]}
    native.update(added_reports)
    values = {}
    for a in s["areas"]:
        for n in a["benchmarks"]:
            values[n] = index02.benchmark_value(n, s, scored.get(n), native.get(n))
    scores, areas = index02.aggregate(values, s)
    entry = index02.index_entry(suite, results, summary, added_reports, s)
    return {"values": values, "scores": scores, "areas": areas, "index_entry": entry}


# ----------------------------------------------------------------------------- helpers


def _res(results, row):
    return results.get(row["_evaluation"]["run_id"])


def _ok(results, row):
    r = _res(results, row)
    return r is not None and r["status"] == "ok"


def _answers(results, row):
    return results[row["_evaluation"]["run_id"]]["response"]["answers"]


def _pred(q, a):
    return a["choice"] if q["type"] == "choice" else a["noul"] >= 0.5


def _round4(x):
    return np.round(x, 4)


def _clip(x):
    return np.clip(x, 0.0, 1.0)


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


class Bench:
    """Cluster statistics S (n_clusters x d) and a vectorised value function."""

    def __init__(self, n, clusters, stats, fn, kind):
        self.n = n
        self.clusters = clusters
        self.S = np.asarray(stats, dtype=np.float64).reshape(len(clusters), -1)
        self.fn = fn
        self.kind = kind

    def value(self, T=None):
        """T: (B, d) summed statistics; default = plain sum (the point estimate)."""
        if T is None:
            T = self.S.sum(axis=0, keepdims=True)
        return self.fn(T)


def _groups(rows):
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(str(r["_evaluation"]["group_id"]), []).append(r)
    return g


# ----------------------------------------------------------------------------- track-scored (index.static_score)


def _track_rows(n, rows):
    tracks = collections.OrderedDict()
    for r in rows:
        t = r["_evaluation"]["track"] if n in (30, 40, 6) else "all"
        if n == 40 and t != HEADLINE_40:
            continue  # only the headline track enters the 0.2.1 index (index.headline)
        tracks.setdefault(t, []).append(r)
    return tracks


def _track_part(n, track, rr, groups, good, results, baselines):
    """Return (column_builder(gid)->vector, fn(T)->(raw, base)) for one track."""
    gids_in_track = list(collections.OrderedDict.fromkeys(str(r["_evaluation"]["group_id"]) for r in rr))
    by_gid = collections.defaultdict(list)
    for r in rr:
        by_gid[str(r["_evaluation"]["group_id"])].append(r)

    if n in (2, 36) or n == 1:
        # per-query nDCG@10 (2, 36) or whole-case exactness (1); one value per group.
        def col(gid):
            if gid not in by_gid:
                return np.zeros(4)
            rows_g = groups[gid]
            ok = good[gid]
            if n in (2, 36):
                s = by_gid[gid][0]["scoring"]
                probs = {}
                if ok:
                    for row in rows_g:
                        a = _answers(results, row)
                        probs.update({doc: a[field]["probabilities"]["yes"] for field, doc in row["scoring"]["field_to_document"].items()})
                v = score_query(s, probs)["ndcg_at_10"] if ok else 0.0
                b = random_ndcg_baseline(s)
            else:
                hits, chance = [], 1.0
                for row in rows_g:
                    a = _answers(results, row) if ok else {}
                    for key, q in row["questions"].items():
                        hits.append(ok and _pred(q, a[key]) == row["expected"][key])
                        chance /= len(q["criteria"]) if q["type"] == "choice" else 2
                v, b = float(all(hits)), chance
            return np.array([v, b, 1.0, float(ok)])

        def fn(T):
            return _safe_div(T[:, 0], T[:, 2]), _safe_div(T[:, 1], T[:, 2]), _safe_div(T[:, 3], T[:, 2])

        return col, fn, 4

    if n in CONSERVATIVE or n == 40:
        classes = set()
        for r in rr:
            for key, q in r["questions"].items():
                classes.update(q["criteria"] if n == 40 else (semantic_label(q, k) for k in q["criteria"]))
        classes = ["yes"] if n == 40 else sorted(classes)
        idx = {k: i for i, k in enumerate(classes)}
        K = len(classes)
        base = baselines[str(n)]["tracks"][track]["value"] if n == 40 else baselines[str(n)]["primary"]["value"]

        def col(gid):
            v = np.zeros(3 * K + 3)
            for r in by_gid.get(gid, []):
                ok = good[gid]
                a = _answers(results, r) if ok else {}
                for key, q in r["questions"].items():
                    gold = r["expected"][key]
                    p = _pred(q, a[key]) if ok else None
                    if n != 40:
                        gold = semantic_label(q, gold)
                        p = semantic_label(q, p) if ok else None
                    if p is None:
                        v[3 * K] += 1  # missing
                    for k, i in idx.items():
                        v[3 * i] += gold == k and p == k
                        v[3 * i + 1] += gold != k and p == k
                        v[3 * i + 2] += gold == k and p is not None and p != k
            if gid in by_gid:
                v[3 * K + 1] += 1
                v[3 * K + 2] += float(good[gid])
            return v

        def fn(T):
            tp, fp, fn_, miss = T[:, 0 : 3 * K : 3], T[:, 1 : 3 * K : 3], T[:, 2 : 3 * K : 3], T[:, 3 * K : 3 * K + 1]
            d = 2 * tp + fp + fn_ + miss
            raw = _safe_div(2 * tp, d).mean(axis=1)
            return raw, np.full(T.shape[0], base), _safe_div(T[:, 3 * K + 2], T[:, 3 * K + 1])

        return col, fn, 3 * K + 3

    # macro / plain accuracy with per-field chance (accepted ties for 31, 50)
    def key_of(r):
        if n == 20:
            return r["metadata"]["variant"]
        if n == 22:
            return r["metadata"]["song_id"]
        if n == 23:
            return r["metadata"]["user_id"]
        return "all"

    keys = sorted({str(key_of(r)) for r in rr})
    kidx = {k: i for i, k in enumerate(keys)}
    K = len(keys)

    def col(gid):
        v = np.zeros(3 * K + 2)
        ok = good[gid]
        for r in by_gid.get(gid, []):
            a = _answers(results, r) if ok else {}
            i = kidx[str(key_of(r))]
            for key, q in r["questions"].items():
                gold = r["expected"][key]
                p = _pred(q, a[key]) if ok else None
                accepted = r.get("scoring", {}).get("accepted", [gold]) if n in (31, 50) else [gold]
                v[3 * i] += float(ok and p in accepted)
                v[3 * i + 1] += len(accepted) / (len(q["criteria"]) if q["type"] == "choice" else 2)
                v[3 * i + 2] += 1
        if gid in by_gid:
            v[3 * K] += 1
            v[3 * K + 1] += float(ok)
        return v

    def fn(T):
        sv, sc, cnt = T[:, 0 : 3 * K : 3], T[:, 1 : 3 * K : 3], T[:, 2 : 3 * K : 3]
        present = cnt > 0
        npres = np.maximum(present.sum(axis=1), 1)
        raw = (_safe_div(sv, cnt) * present).sum(axis=1) / npres
        base = (_safe_div(sc, cnt) * present).sum(axis=1) / npres
        return raw, base, _safe_div(T[:, 3 * K + 1], T[:, 3 * K])

    return col, fn, 3 * K + 2


def _track_bench(n, rows, results, baselines):
    groups = _groups(rows)
    good = {g: all(_ok(results, r) for r in rs) for g, rs in groups.items()}
    parts = [_track_part(n, t, rr, groups, good, results, baselines) for t, rr in _track_rows(n, rows).items()]
    clusters = list(groups)
    widths = [w for _, _, w in parts]
    offsets = np.cumsum([0] + widths)
    S = np.array([np.concatenate([col(g) for col, _, _ in parts]) for g in clusters])

    def fn(T):
        raws, skills = [], []
        for (col, f, w), o in zip(parts, offsets):
            raw, base, _ = f(T[:, o : o + w])
            raws.append(raw)
            skills.append(_clip(_safe_div(raw - base, 1 - base)))
        raw = np.mean(raws, axis=0)
        skill = np.mean(skills, axis=0)
        return _round4(raw), _round4(skill)

    return Bench(n, clusters, S, fn, "track")


# ----------------------------------------------------------------------------- native-metric benchmarks (report.score / added.report)


def _display_bench(n, rows, results, chance):
    groups = _groups(rows)
    clusters = list(groups)
    labels = []
    if n in MACRO_F1:
        lab = set()
        for r in rows:
            for key, q in r["questions"].items():
                lab.update(semantic_label(q, k) for k in q["criteria"])
        labels = sorted(lab)
    lidx = {k: i for i, k in enumerate(labels)}
    L = len(labels)

    def col(gid):
        rs = groups[gid]
        complete = all(_ok(results, r) for r in rs)
        ok_rows = sum(_ok(results, r) for r in rs)
        tail = [ok_rows, len(rs)]
        if n in MACRO_F1:
            v = np.zeros(3 * L)
            if complete:
                for r in rs:
                    a = _answers(results, r)
                    for key, q in r["questions"].items():
                        g, p = semantic_label(q, r["expected"][key]), semantic_label(q, _pred(q, a[key]))
                        for k in {g, p}:
                            i = lidx[k]
                            v[3 * i] += g == k and p == k
                            v[3 * i + 1] += g != k and p == k
                            v[3 * i + 2] += g == k and p != k
            return np.concatenate([v, tail])
        if n in FIELD_ACC:
            hits = fields = 0
            if complete:
                for r in rs:
                    a = _answers(results, r)
                    for key, q in r["questions"].items():
                        hits += _pred(q, a[key]) == r["expected"][key]
                        fields += 1
            return np.array([hits, fields] + tail, dtype=float)
        if n in CASE_EXACT:
            hit = 0.0
            if complete:
                hit = float(all(_pred(r["questions"][q], _answers(results, r)[q]) == g for r in rs for q, g in r["expected"].items()))
            return np.array([hit, float(complete)] + tail)
        if n == 38:
            f1 = 0.0
            if complete:
                gold_yes, pred_yes = set(), set()
                for r in rs:
                    a = _answers(results, r)
                    for q, g in r["expected"].items():
                        if g in ("yes", True):
                            gold_yes.add((r["_evaluation"]["run_id"], q))
                        if _pred(r["questions"][q], a[q]) in ("yes", True):
                            pred_yes.add((r["_evaluation"]["run_id"], q))
                d = len(gold_yes) + len(pred_yes)
                f1 = 2 * len(gold_yes & pred_yes) / d if d else 1.0
            return np.array([f1, float(complete)] + tail)
        if n == 48:
            brier = cnt = 0.0
            if complete:
                for r in rs:
                    p = _answers(results, r)["answer"]["probabilities"]["yes"]
                    y = int(r["expected"]["answer"] == "yes")
                    brier += (p - y) ** 2
                    cnt += 1
            return np.array([brier, cnt] + tail)
        raise KeyError(n)

    S = np.array([col(g) for g in clusters])

    def fn(T):
        ok, req = T[:, -2], T[:, -1]
        cov = _safe_div(ok, req)
        if n in MACRO_F1:
            tp, fp, fn_ = T[:, 0 : 3 * L : 3], T[:, 1 : 3 * L : 3], T[:, 2 : 3 * L : 3]
            present = (tp + fp + fn_) > 0
            f1 = _safe_div(2 * tp, 2 * tp + fp + fn_)
            npres = present.sum(axis=1)
            score = _safe_div((f1 * present).sum(axis=1), npres)
            has = npres > 0
        else:
            score = _safe_div(T[:, 0], T[:, 1])
            has = T[:, 1] > 0
        if n == 48:
            raw = np.where(has, _clip((0.25 - _round4(score)) / 0.25) * cov, 0.0)
            return raw, raw
        raw = np.where(has, _round4(score), 0.0) * cov
        return raw, _clip(_safe_div(raw - chance, 1 - chance))

    return Bench(n, clusters, S, fn, "display")


def _added_bench(n, rows, results, chance):
    groups = _groups(rows)
    clusters = list(groups)
    f1 = n in added_mod.F1_POSITIVE
    positive = added_mod.F1_POSITIVE.get(n)

    def col(gid):
        rs = groups[gid]
        v = np.zeros(4 if not f1 else 6)  # [correct | tp, fp, fn], n_fields, ok_rows, rows
        for r in rs:
            if not _ok(results, r):
                continue
            a = _answers(results, r)
            for key in added_mod.scored_keys(n, r):
                x = added_mod.field(r["questions"][key], r["expected"][key], a[key])
                if f1:
                    v[0] += x["pred"] == positive and x["gold"] == positive
                    v[1] += x["pred"] == positive and x["gold"] != positive
                    v[2] += x["pred"] != positive and x["gold"] == positive
                    v[-3] += 1
                else:
                    v[0] += bool(x["correct"])
                    v[-3] += 1
        v[-2] = sum(_ok(results, r) for r in rs)
        v[-1] = len(rs)
        return v

    S = np.array([col(g) for g in clusters])

    def fn(T):
        cov = _safe_div(T[:, -2], T[:, -1])
        nf = T[:, -3]
        if f1:
            tp, fp, fn_ = T[:, 0], T[:, 1], T[:, 2]
            score = _safe_div(2 * tp, 2 * tp + fp + fn_)
        else:
            score = _safe_div(T[:, 0], nf)
        score = np.where(nf > 0, score, 0.0)
        raw = _round4(score * cov)
        return raw, _round4(_clip(_safe_div(raw - chance, 1 - chance)))

    return Bench(n, clusters, S, fn, "added")


def decompose(rows, results):
    """{catalog_id: Bench} for every index benchmark present in `rows`."""
    s = spec()
    baselines = chance_baselines()
    by = collections.defaultdict(list)
    for r in rows:
        by[r["_evaluation"]["catalog_id"]].append(r)
    out = {}
    for a in s["areas"]:
        for n in a["benchmarks"]:
            if n not in by:
                continue
            rs = by[n]
            if n in s["track_scored"]:
                out[n] = _track_bench(n, rs, results, baselines)
            elif str(n) in s["added"]:
                out[n] = _added_bench(n, rs, results, s["chance"][str(n)]["chance"])
            else:
                chance = s["chance"][str(n)]["chance"] if str(n) in s["chance"] else 0.0
                out[n] = _display_bench(n, rs, results, chance)
    return out


# ----------------------------------------------------------------------------- aggregation


def weights(present=None, renormalize=True):
    """Effective weight of each index benchmark in the 0.2.1 index.

    renormalize=False reproduces the board (a missing benchmark keeps its weight and scores 0);
    renormalize=True spreads a missing benchmark's weight over its area (dev-slice index)."""
    s = spec()
    aw = index02.area_weights(s)
    out = {}
    for a in s["areas"]:
        ids = a["benchmarks"]
        if renormalize and present is not None:
            ids = [n for n in ids if n in present]
        bw = index02.bench_weights(ids, s) if ids else {}
        for n, w in bw.items():
            out[n] = aw[a["id"]] * w
    return out


def index_from_skills(skills, renormalize=True):
    """skills: {n: array(B)} -> array(B) index on the 0-100 scale."""
    w = weights(set(skills), renormalize)
    total = None
    for n, wn in w.items():
        term = wn * np.asarray(skills.get(n, 0.0))
        total = term if total is None else total + term
    return 100 * total


def point(benches, renormalize=True):
    vals = {n: b.value() for n, b in benches.items()}
    skills = {n: v[1] for n, v in vals.items()}
    return {
        "per_benchmark": {n: {"raw": float(v[0][0]), "skill": float(v[1][0])} for n, v in vals.items()},
        "index": float(index_from_skills(skills, renormalize)[0]),
    }
