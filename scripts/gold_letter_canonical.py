"""Gold-letter distribution under CANONICAL option order (JEV3-LM shim risk diagnostic; $0, CPU, streaming).

Canonical key (stated; the lead's own implementation was not yet pushed on stack/qwen35/jev3-lm when this ran):
    key(desc) = " ".join(unicodedata.normalize("NFC", str(desc)).split()).casefold()
    order    = sorted by (key(desc), sha256(NFC(desc) utf-8))      # content hash breaks key ties
    tie_set  = options whose (key, hash) are identical (exact duplicates after NFC); gold in a tie_set is counted
               separately (letter ambiguous) and excluded from the letter chi-square.
    letters  = A, B, C, ... in that order.
noul: the two options are the criteria descriptions of true / false (sorted the same way).
score: options in level order (letter = level index), so the gold letter IS the gold level.

Also computed: the PRESENTED order (train/slice: the criteria dict order = seeded per-row shuffle; public: the item's
criteria dict order, plus the item's `labels` list order) and the decide.schema.canonical_options order
(choice sorted by option id) for comparison.

Public JevBench items: FORMAT-LEVEL statistics only (gold position). Eval-only; never used for training or selection.

Usage: python -X utf8 gold_letter_canonical.py --data <ssdm-data> --repo <checkout with jevkit> --out <json>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ABSTAIN_RE = re.compile(r"\b(none|other|unknown|cannot|can't|can ?not|escalat\w*|declin\w*|refus\w*|neither|"
                        r"not possible|no listed|no final|abstain)\b")


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s))


def ckey(desc) -> str:
    return " ".join(nfc(desc).split()).casefold()


def chash(desc) -> str:
    return hashlib.sha256(nfc(desc).encode("utf-8")).hexdigest()


def canonical(opts: list[tuple[str, object]]) -> tuple[list[str], dict[str, int]]:
    """opts: [(option_id, description)] -> (ids in canonical order, {id: tie-set size})."""
    keyed = [((ckey(d), chash(d)), oid) for oid, d in opts]
    keyed.sort(key=lambda t: (t[0], t[1]))
    sizes = Counter(k for k, _ in keyed)
    return [oid for _, oid in keyed], {oid: sizes[k] for k, oid in keyed}


# --------------------------------------------------------------------------------------- chi-square survival fn
def _gser(a, x):
    s = term = 1.0 / a
    n = a
    for _ in range(10000):
        n += 1
        term *= x / n
        s += term
        if abs(term) < abs(s) * 1e-15:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x):
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        dl = d * c
        h *= dl
        if abs(dl - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    return 1.0 - _gser(a, xx) if xx < a + 1 else _gcf(a, xx)


def chi_uniform_by_k(rows):
    """rows: [(K, letter_index)]. chi-square vs uniform over K, stratified by K and summed (df summed)."""
    byk = defaultdict(Counter)
    for k, i in rows:
        byk[k][i] += 1
    chi = df = 0.0
    strata = {}
    for k, cnt in sorted(byk.items()):
        n = sum(cnt.values())
        e = n / k
        c = sum((cnt.get(j, 0) - e) ** 2 / e for j in range(k))
        strata[k] = {"n": n, "counts": [cnt.get(j, 0) for j in range(k)], "chi2": round(c, 3), "df": k - 1,
                     "p": chi2_sf(c, k - 1), "min_expected": round(e, 2)}
        chi += c
        df += k - 1
    return {"chi2": round(chi, 3), "df": int(df), "p": chi2_sf(chi, df), "by_k": strata}


# --------------------------------------------------------------------------------------------- per-item analysis
def options_of(q):
    crit = q.get("criteria")
    if q["type"] == "score":
        return [(str(i), d) for i, d in enumerate(crit)]
    return [(str(k), v) for k, v in crit.items()]


def analyse(qtype, opts, gold, alt_orders):
    """gold: the gold option id (noul: 'true'/'false'; score: level str). Returns a record."""
    rec = {"type": qtype, "K": len(opts)}
    ids = [o for o, _ in opts]
    if qtype == "score":
        rec["can"] = int(gold)
        rec["tie"] = 1
    else:
        order, ties = canonical(opts)
        rec["can"] = order.index(gold)
        rec["tie"] = ties[gold]
        rec["any_tie"] = max(ties.values()) > 1
        desc = dict(opts)
        first = order[0]
        rec["first_abstain"] = bool(ABSTAIN_RE.search(ckey(desc[first])) or ABSTAIN_RE.search(first.lower()))
        rec["has_abstain"] = any(ABSTAIN_RE.search(ckey(d)) or ABSTAIN_RE.search(o.lower()) for o, d in opts)
        rec["gold_abstain"] = bool(ABSTAIN_RE.search(ckey(desc[gold])) or ABSTAIN_RE.search(gold.lower()))
        rec["true_letter"] = order.index("true") if qtype == "noul" else None
    for name, seq in alt_orders.items():
        if seq is not None:
            rec[name] = list(seq).index(gold)
    rec["pres_default"] = ids.index(gold)
    return rec


def iter_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            fam = r["aug"]["jev1"]["family8"]
            for qid, q in r["questions"].items():
                gold = str(r["labels"][qid])
                opts = options_of(q)
                alt = {"pres": [o for o, _ in opts]}
                if q["type"] == "choice":
                    alt["idsort"] = sorted(o for o, _ in opts)
                yield fam, analyse(q["type"], opts, gold, alt)


def iter_public(repo):
    sys.path.insert(0, str(repo))
    from jevkit.items import load_public  # noqa: E402
    items = load_public()
    for it in items:
        q = it.question
        opts = options_of(q)
        if it.type == "noul":
            gold = {"yes": "true", "no": "false"}[str(it.expected)]
            lab = [{"yes": "true", "no": "false"}[x] for x in it.labels]
        else:
            gold = str(it.expected)
            lab = list(it.labels)
        alt = {"pres": [o for o, _ in opts], "pres_labels": lab}
        if it.type == "choice":
            alt["idsort"] = sorted(o for o, _ in opts)
        yield it.family, analyse(it.type, opts, gold, alt)


def summarise(recs, order_field="can"):
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["type"]].append(r)
    out = {}
    for t, rs in sorted(by_type.items()):
        clean = [r for r in rs if r.get("tie", 1) == 1 or order_field != "can"]
        rows = [(r["K"], r[order_field]) for r in clean if order_field in r]
        if not rows:
            continue
        n = len(rows)
        letters = Counter(i for _, i in rows)
        maxk = max(k for k, _ in rows)
        exp = [sum(1.0 / k for k, _ in rows if k > j) for j in range(maxk)]
        best_const = max(letters.values()) / n
        s = {
            "n": len(rs), "n_scored": n, "n_gold_in_tie_set": len(rs) - len(clean) if order_field == "can" else 0,
            "K_dist": dict(sorted(Counter(k for k, _ in rows).items())),
            "letter_share": {LET[j]: round(letters.get(j, 0) / n, 4) for j in range(maxk)},
            "uniform_share": {LET[j]: round(exp[j] / n, 4) for j in range(maxk)},
            "share_A": round(letters.get(0, 0) / n, 4), "chance_A": round(exp[0] / n, 4),
            "best_constant_letter": LET[max(letters, key=letters.get)], "best_constant_acc": round(best_const, 4),
            "chance_acc": round(sum(1.0 / k for k, _ in rows) / n, 4),
            "chi2_vs_uniform": chi_uniform_by_k(rows),
        }
        if order_field == "can" and t != "score":
            s["items_with_any_tie_set"] = sum(1 for r in rs if r.get("any_tie"))
            s["abstain"] = {
                "has_abstain_option": sum(r["has_abstain"] for r in rs),
                "abstain_sorts_first": sum(r["first_abstain"] for r in rs),
                "abstain_first_and_gold": sum(r["first_abstain"] and r["gold_abstain"] and r["can"] == 0 for r in rs),
                "gold_is_abstain": sum(r["gold_abstain"] for r in rs),
            }
            if t == "noul":
                s["true_option_letter"] = {LET[k]: v for k, v in sorted(Counter(r["true_letter"] for r in rs).items())}
                s["gold_true_share"] = round(sum(1 for r in rs if (r["can"] == r["true_letter"])) / len(rs), 4)
        out[t] = s
    return out


def run(name, it):
    by_fam = defaultdict(list)
    for fam, r in it:
        by_fam[fam].append(r)
    allr = [r for rs in by_fam.values() for r in rs]
    fields = ["can", "pres", "pres_labels", "idsort"]
    res = {"n_questions": len(allr)}
    for f in fields:
        if any(f in r for r in allr):
            res[f] = {"ALL": summarise(allr, f), "by_family": {fam: summarise(rs, f) for fam, rs in sorted(by_fam.items())}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\apumu\research\CHLabs\ssdm-data")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = Path(a.data) / "m2" / "build"
    res = {
        "key": "NFC -> whitespace-collapse -> casefold of the option description; ties by sha256(NFC desc); "
               "score = level order",
        "order_fields": {"can": "canonical (this key)", "pres": "presented criteria dict order",
                         "pres_labels": "public only: item.labels list order",
                         "idsort": "choice sorted by option id (decide.schema.canonical_options)"},
        "abstain_regex": ABSTAIN_RE.pattern,
        "train": run("train", iter_records(b / "bschema-draw-jev3" / "records.jsonl.gz")),
        "slice": run("slice", iter_records(b / "bschema-draw-jev3-slice" / "records.jsonl.gz")),
        "public": run("public", iter_public(a.repo)),
    }
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
