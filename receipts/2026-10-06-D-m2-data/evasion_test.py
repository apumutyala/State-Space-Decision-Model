"""Guard recall under small edits, on DEV rows only (the TEST slice is never opened).

Reproduces the PR #16 review's evasion test: 40 DEV rows per catalog (13 catalogs), seeded. For each row two
probes are taken among its single-line item texts that the fingerprint indexes (>= 4 tokens): the shortest
("short", e.g. an utterance or an ANLI hypothesis) and the longest ("long"). Each probe is mutated and run
through ``Fingerprint.check``; a probe counts as caught if any matcher fires. Reported per mutation, for the
v2 matcher set (exact, ngram13, minhash) and for v3 (+ edit, contain).

Usage: python -X utf8 -I evasion_test.py <dev-rows.jsonl.gz> > evasion.json
"""
import collections
import gzip
import json
import random
import sys

from ssdm_data import config, dedup as D, text as T

CATALOGS = {4: "BANKING77", 5: "CLINC150", 62: "When2Call", 57: "MMLU-Pro", 12: "ANLI", 28: "WinoGrande",
            30: "GSM8K", 29: "HellaSwag", 32: "MuSR", 58: "BBH", 59: "RAGTruth", 11: "ContractNLI", 25: "GPQA"}
PER_CAT = 40
VOCAB = ["banana", "quietly", "purple", "seventeen", "harbor", "violin", "gravel", "tomorrow", "lantern", "maple"]
V2 = ("exact", "ngram13", "minhash")
V3 = V2 + ("edit", "contain")


def probes(row, fp):
    out = []
    for s in T.raw_segments(row["state"], row["questions"]):
        for line in s.split("\n"):
            m = T._LABEL_PREFIX.match(line)
            line = line[m.end():] if m else line
            n = T.normalize(line)
            ws = n.split()
            if len(ws) < 4 or len(n) < T.MIN_SEGMENT_CHARS:
                continue
            if fp._lookup(fp.seg_hash, D.np.array([T.h64(n)], dtype=D.np.uint64))[0] >= 0:
                out.append(n)
    if not out:
        return {}
    out = sorted(set(out), key=lambda x: (len(x.split()), x))
    return {"short": out[0], "long": out[-1]}


def mutate(text, kind, rng):
    ws = text.split()
    if kind == "control":
        return "Question: " + text.upper() + "  ?!"
    if kind in ("sub1", "sub2"):
        k = 1 if kind == "sub1" else 2
        for p in rng.sample(range(len(ws)), min(k, len(ws))):
            ws[p] = rng.choice([v for v in VOCAB if v != ws[p]])
        return " ".join(ws)
    if kind == "append":
        return text + ". Please answer as soon as you can."
    if kind == "trunc_half":
        return " ".join(ws[: max(1, len(ws) // 2)])
    raise ValueError(kind)


def main(path):
    fp = D.Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    by = collections.defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["_evaluation"]["catalog_id"]
            if c in CATALOGS:
                by[c].append(r)
    res = {}
    for c, name in CATALOGS.items():
        rows = sorted(by.get(c, []), key=lambda r: r["_evaluation"]["run_id"])
        random.Random(f"evasion|{c}").shuffle(rows)
        tally = collections.Counter()
        lens = collections.defaultdict(list)
        n = 0
        for r in rows:
            pr = probes(r, fp)
            if not pr:
                continue
            n += 1
            for which, text in pr.items():
                lens[which].append(len(text.split()))
                for kind in ("control", "sub1", "sub2", "append", "trunc_half"):
                    rng = random.Random(f"{r['_evaluation']['run_id']}|{which}|{kind}")
                    h = fp.check(mutate(text, kind, rng), {})
                    tally[(which, kind, "v2")] += any(h[k] for k in V2)
                    tally[(which, kind, "v3")] += any(h[k] for k in V3)
            if n >= PER_CAT:
                break
        res[name] = {"catalog": c, "n": n,
                     "median_tokens": {w: sorted(v)[len(v) // 2] for w, v in lens.items()},
                     "recall": {f"{w}:{k}:{v}": round(tally[(w, k, v)] / n, 3)
                                for w in ("short", "long") for k in ("control", "sub1", "sub2", "append", "trunc_half")
                                for v in ("v2", "v3")} if n else {}}
    print(json.dumps({"fingerprint": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "rows", "short_items")},
                      "dev_rows": path, "per_catalog": res}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
