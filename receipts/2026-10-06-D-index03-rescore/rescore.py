"""$0 CPU re-score of existing per-row predictions under the Decision Index 0.3 public index (vs 0.2.1).

    PYTHONPATH=<ssdm-data>/decision-index python -X utf8 receipts/2026-10-06-D-index03-rescore/rescore.py <scratch> > rescore.json

<scratch> holds the job exports (outside git; they carry request payloads):
  clef-art1/x/clef-flash-jobA/index/decide/results.jsonl   job-sgihj, Clef-flash, full DEV (12,104 rows)
  5cmat/s/out/index/stock/results.jsonl                    job-5cmat, Qwen3.5-9B stock readout, m1 subset (2,008)
  5cmat/s/out/index/conv/results.jsonl                     job-5cmat, Qwen3.5-9B 1 attn layer -> GDN, m1 subset
  rwkv7-art2/x/phases/final/out/index/stock/results.jsonl  job-tkfaj, RWKV7-G1k-13.3B stock readout, m1 subset
Only DEV rows are read (harness.cli.load_slice("dev")); TEST is never touched. Output: aggregates only.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness import bootstrap, families, score  # noqa: E402
from harness.cli import load_results, load_slice  # noqa: E402
from jobs.common.dev_subset import load_subset  # noqa: E402

B, SEED = 2000, 0
S = Path(sys.argv[1])
FILES = {
    "clef-flash": S / "clef-art1/x/clef-flash-jobA/index/decide/results.jsonl",
    "q9-stock": S / "5cmat/s/out/index/stock/results.jsonl",
    "q9-1gdn": S / "5cmat/s/out/index/conv/results.jsonl",
    "rwkv7-13b": S / "rwkv7-art2/x/phases/final/out/index/stock/results.jsonl",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ci(x):
    lo, hi = np.quantile(x[1:], [0.025, 0.975])
    return {"point": round(float(x[0]), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}


def block(rows, systems, edition):
    """Score every system on the same rows under one edition, sharing bootstrap resamples."""
    names = list(systems)
    benches = [score.decompose(rows, systems[k], edition=edition) for k in names]
    reps = bootstrap.replicate_skills(benches, B=B, seed=SEED)
    out = {"edition": edition, "benchmarks": sorted(benches[0]), "systems": {}, "paired_vs_clef_flash": {}}
    skills = {k: {n: v[1] for n, v in rep.items()} for k, rep in zip(names, reps)}
    idx = {k: score.index_from_skills(skills[k], True, edition) for k in names}
    areas = {k: bootstrap.area_skills(skills[k], True, edition) for k in names}
    for k, rep in zip(names, reps):
        fam = families.summarize(rep, edition)
        out["systems"][k] = {"index": ci(idx[k]), "areas": {a: ci(v) for a, v in areas[k].items()},
                             "families": {g: ci_from(fam[g]) for g in families.GROUPS} | {"in_minus_proxy": ci_from(fam["in_minus_proxy"])},
                             "benchmarks": {str(n): round(100 * float(v[1][0]), 2) for n, v in sorted(rep.items())}}
    base = names.index("clef-flash") if "clef-flash" in names else None
    pairs = [(k, "clef-flash") for k in names if k != "clef-flash"] if base is not None else []
    if "q9-1gdn" in names and "q9-stock" in names:
        pairs.append(("q9-1gdn", "q9-stock"))
    for a, b in pairs:
        d = idx[a] - idx[b]
        fam = families.compare(reps[names.index(a)], reps[names.index(b)], edition)
        tgt = out["paired_vs_clef_flash"] if b == "clef-flash" else out.setdefault("paired_other", {})
        tgt[f"{a} - {b}"] = {"index": ci(d) | {"p_le_zero": round(float(np.mean(d[1:] <= 0)), 4)},
                             "areas": {x: ci(areas[a][x] - areas[b][x]) for x in areas[a]},
                             "families": {g: ci_from(fam[g]) for g in families.GROUPS}}
    return out


def ci_from(c):
    return {k: round(c[k], 3) for k in ("point", "lo", "hi")}


def main():
    dev = load_slice("dev")
    sub = load_subset()
    ids = set(sub["run_ids"])
    sub_rows = [r for r in dev if r["_evaluation"]["run_id"] in ids]
    assert len(sub_rows) == len(ids) == 2008
    res = {k: load_results(p) for k, p in FILES.items()}
    out = {"B": B, "seed": SEED, "families": families.load()["id"],
           "inputs": {k: {"path": p.relative_to(S).as_posix(), "sha256": sha(p), "rows": len(res[k])} for k, p in FILES.items()},
           "dev_rows": len(dev), "subset": {"ids_sha256": sub["ids_sha256"], "rows": len(sub_rows)}}
    for k, r in res.items():
        need = dev if k == "clef-flash" else sub_rows
        out["inputs"][k]["missing_rows"] = sum(x["_evaluation"]["run_id"] not in r for x in need)
    out["full_dev"] = {e: block(dev, {"clef-flash": res["clef-flash"]}, e) for e in ("0.2.1", "0.3")}
    out["subset"]["scores"] = {e: block(sub_rows, res, e) for e in ("0.2.1", "0.3")}
    for e in ("0.2.1", "0.3"):
        out["subset"]["scores"][e]["rows_scored"] = len(sub_rows) if e == "0.2.1" else sum(1 for r in sub_rows if r["_evaluation"]["catalog_id"] not in (30, 48))
        out["full_dev"][e]["rows_scored"] = len(dev) if e == "0.2.1" else sum(1 for r in dev if r["_evaluation"]["catalog_id"] not in (30, 48))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
