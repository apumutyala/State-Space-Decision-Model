"""Anonymous receipt packet: per-cluster sufficient statistics + official aggregate references.

Lets a reviewer recompute every per-benchmark value, the index and the paired bootstrap
without seeing any DEV item. The packet holds, per index benchmark:
- `spec`: the plain-data value-function spec (`score.value_fn`), i.e. column layout, number
  of label/key blocks, chance level and track baseline (all public numbers);
- per system, the cluster x column statistics matrix in sparse triplets.
And per system, the official kit scorer's per-benchmark raw/skill and `balanced_skill`.

Not included: item text, options, gold labels, row/run/group/case ids, label or key names,
or the order of anything. Clusters are randomly shuffled and the per-label / per-key column
blocks are randomly permuted, per benchmark, with a seed drawn from os.urandom that is not
recorded. One permutation is shared by all systems, so systems can still be paired.

Residual information (by construction, needed for recomputation): per-cluster counts
(rows, fields, ok rows) and, for each system, per-cluster correctness counts in permuted
label columns. Someone who already holds the DEV rows could try to align clusters by
their size profile; nobody without the rows can recover items, gold labels or ids.
"""

import gzip
import json
import os

import numpy as np

from harness import bootstrap, config, score

FORMAT = "ssdm-anon-packet-v1"


def _block_perm(spec, width, rng):
    """Column permutation that shuffles whole per-label / per-key blocks of 3 columns."""
    perm = list(range(width))
    if spec["kind"] == "display" and spec["metric"] == "macro_f1":
        blocks = [(0, spec["L"])]
    elif spec["kind"] == "track":
        blocks, o = [], 0
        for ps in spec["parts"]:
            if ps["type"] in ("cf1", "macro"):
                blocks.append((o, ps["K"]))
            o += ps["width"]
    else:
        blocks = []
    for start, k in blocks:
        order = rng.permutation(k)
        for new, old in enumerate(order):
            for j in range(3):
                perm[start + 3 * new + j] = start + 3 * old + j
    return np.array(perm)


def _sparse(S):
    r, c = np.nonzero(S)
    return {"shape": list(S.shape), "rows": r.tolist(), "cols": c.tolist(), "values": [float(v) for v in S[r, c]]}


def _dense(sp):
    S = np.zeros(sp["shape"])
    S[np.array(sp["rows"], dtype=int), np.array(sp["cols"], dtype=int)] = np.array(sp["values"], dtype=float)
    return S


def build(rows, systems, official_refs=True):
    """systems: {name: results}. Returns the packet dict (no secrets, no ids)."""
    rng = np.random.default_rng(int.from_bytes(os.urandom(16), "little"))  # never recorded
    decomposed = {name: score.decompose(rows, res) for name, res in systems.items()}
    first = next(iter(decomposed.values()))
    benches = {}
    for n in sorted(first):
        spec = first[n].spec
        k, d = first[n].S.shape
        cperm = rng.permutation(k)
        colperm = _block_perm(spec, d, rng)
        sysmats = {}
        for name, dec in decomposed.items():
            b = dec[n]
            if b.clusters != first[n].clusters or b.spec != spec:
                raise ValueError(f"benchmark {n}: systems disagree on clusters/spec")
            sysmats[name] = _sparse(b.S[cperm][:, colperm])
        benches[str(n)] = {"spec": spec, "n_clusters": int(k), "width": int(d), "systems": sysmats}
    out = {
        "format": FORMAT,
        "edition": config.EDITION,
        "kit_commit": config.KIT_COMMIT,
        "slice": "dev",
        "weights": {str(n): w for n, w in score.weights().items()},
        "benchmarks": benches,
    }
    if official_refs:
        refs = {}
        for name, res in systems.items():
            off = score.official(rows, res)
            refs[name] = {
                "balanced_skill": float(off["scores"]["balanced_skill"]),
                "per_benchmark": {str(n): {"raw": float(v["raw"]), "skill": float(v["skill"])} for n, v in off["values"].items()},
            }
        out["official_reference"] = refs
    return out


def save(packet, path):
    with gzip.GzipFile(path, "wb", mtime=0) as g:
        g.write((json.dumps(packet, separators=(",", ":"), sort_keys=True) + "\n").encode())


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        packet = json.load(f)
    if packet.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} packet")
    return packet


def benches(packet, system):
    """{catalog_id: score.Bench} for one system, clusters labelled 0..k-1."""
    out = {}
    for n, b in packet["benchmarks"].items():
        S = _dense(b["systems"][system])
        out[int(n)] = score.Bench(int(n), list(range(b["n_clusters"])), S, b["spec"])
    return out


def recompute(packet, pairs=(), B=2000, seed=0):
    """Point values + index per system (checked against the official reference), and paired
    bootstrap CIs/MDE for the requested (candidate, baseline) pairs."""
    from harness import mde

    systems = sorted(next(iter(packet["benchmarks"].values()))["systems"])
    out = {"systems": {}, "pairs": {}}
    for name in systems:
        pt = score.point(benches(packet, name), renormalize=False)
        entry = {"index": pt["index"], "per_benchmark": {str(n): v for n, v in pt["per_benchmark"].items()}}
        ref = packet.get("official_reference", {}).get(name)
        if ref:
            entry["max_abs_skill_gap_vs_official"] = max(abs(v["skill"] - ref["per_benchmark"][n]["skill"]) for n, v in entry["per_benchmark"].items())
            entry["max_abs_raw_gap_vs_official"] = max(abs(v["raw"] - ref["per_benchmark"][n]["raw"]) for n, v in entry["per_benchmark"].items())
            entry["index_gap_vs_official"] = abs(entry["index"] - ref["balanced_skill"])
        out["systems"][name] = entry
    for cand, base in pairs:
        r = bootstrap.compare(benches(packet, cand), benches(packet, base), B=B, seed=seed)
        n = sum(b["n_clusters"] for b in packet["benchmarks"].values())
        out["pairs"][f"{cand} - {base}"] = {"diff": r["diff"]["index"], "mde": mde.report(r["diff"]["index"]["se"], n)}
    return out
