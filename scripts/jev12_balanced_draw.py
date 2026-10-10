"""JEV2-4B training + calibration draws: family-balanced, from one or more reviewed draw dirs (jev1, jev2, ...).

Each source dir is a finished, gated draw (``records.jsonl.gz`` + ``<build>.manifest.jsonl.gz`` +
``<build>.content.jsonl.gz``, as scripts/jev1_build.py ``final`` writes it). Per JevBench family (``aug.<draw>.family8``
or ``aug.<draw>.family``) the rows are ranked by sha256("<seed>:<id>") (label-blind, deterministic); the first
``--train-per-family`` go to the TRAIN draw and the next ``--calib-per-family`` to the CALIB draw (post-hoc temperature
fit only; never trained on). A family with fewer rows than train + calib fails closed unless ``--allow-short`` (then
it contributes what it has, train first, and the report says so). Content disjointness train vs calib is checked on the
source content maps (record_content_hash), so a content twin under another id cannot straddle the two.

Two passes, O(families x (train + calib)) memory: pass 1 ranks ids per family (heaps), pass 2 streams the lines out.
Writes <out>/<name>/ and <out>/<name>-calib/ in the jev1 three-file format + a pointer JSON + draw-report.json.

    python -X utf8 -I scripts/jev12_balanced_draw.py --src <ssdm-data>/m2/build/bschema-draw-jev1 \
        --src <ssdm-data>/m2/build/bschema-draw-jev2 --train-per-family 1000 --calib-per-family 60 \
        --name bschema-draw-jev12-bal1000 --out <ssdm-data>/m2/build
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import heapq
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _lines(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def _gz(p):
    return io.TextIOWrapper(gzip.GzipFile(p, "wb", mtime=0), encoding="utf-8", newline="\n")


def family_of(r: dict) -> str | None:
    for v in (r.get("aug") or {}).values():
        if isinstance(v, dict) and (v.get("family8") or v.get("family")):
            return v.get("family8") or v.get("family")
    return None


def rank(seed: int, rid: str) -> str:
    return hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()


def plan(srcs: list[Path], n_train: int, n_calib: int, seed: int) -> tuple[dict, dict]:
    """Pass 1: per family the (n_train + n_calib) smallest-rank ids; returns ({id: (family, role)}, family counts)."""
    keep = n_train + n_calib
    heaps: dict[str, list] = collections.defaultdict(list)
    seen: set[str] = set()
    counts = collections.Counter()
    for d in srcs:
        for line in _lines(d / "records.jsonl.gz"):
            r = json.loads(line)
            rid, fam = r["id"], family_of(r)
            if fam is None:
                raise SystemExit(f"{d}: {rid} has no aug.<draw>.family8 / family")
            if rid in seen:
                raise SystemExit(f"duplicate id across sources: {rid}")
            seen.add(rid)
            counts[fam] += 1
            item = (-int(rank(seed, rid)[:16], 16), rid)  # max-heap of the smallest ranks
            h = heaps[fam]
            if len(h) < keep:
                heapq.heappush(h, item)
            elif item > h[0]:
                heapq.heapreplace(h, item)
    roles = {}
    for fam, h in heaps.items():
        ids = [rid for _, rid in sorted(h, key=lambda x: -x[0])]  # ascending rank
        for k, rid in enumerate(ids):
            roles[rid] = (fam, "train" if k < n_train else "calib")
    return roles, dict(counts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--train-per-family", type=int, required=True)
    ap.add_argument("--calib-per-family", type=int, default=200)  # EV-PREREG-AREA-TEMP addendum: >= 200 per family
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-short", action="store_true")
    a = ap.parse_args(argv)
    from train.cohort import record_content_hash

    srcs = [Path(s) for s in a.src]
    roles, counts = plan(srcs, a.train_per_family, a.calib_per_family, a.seed)
    short = {f: n for f, n in counts.items() if n < a.train_per_family + a.calib_per_family}
    if short and not a.allow_short:
        raise SystemExit(f"families below train + calib: {short} (pass --allow-short to accept)")
    out = Path(a.out)
    dirs = {"train": out / a.name, "calib": out / f"{a.name}-calib"}
    for d in dirs.values():
        if (d / "records.jsonl.gz").exists():
            raise SystemExit(f"refusing to overwrite {d}")
        d.mkdir(parents=True, exist_ok=True)
    fh = {role: (_gz(d / "records.jsonl.gz"), _gz(d / f"{d.name}.manifest.jsonl.gz"), _gz(d / f"{d.name}.content.jsonl.gz"))
          for role, d in dirs.items()}
    hs = {role: [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()] for role in dirs}
    per = {role: collections.Counter() for role in dirs}
    content = {role: set() for role in dirs}
    ids = {role: [] for role in dirs}
    try:
        for d in srcs:
            for line in _lines(d / "records.jsonl.gz"):
                r = json.loads(line)
                got = roles.get(r["id"])
                if got is None:
                    continue
                fam, role = got
                ch = record_content_hash(r)
                content[role].add(ch)
                parts = (line + "\n",
                         canonical({"family": r["family"], "id": r["id"],
                                    "sha256": hashlib.sha256(line.encode()).hexdigest(), "source": r["source"]}) + "\n",
                         canonical({"content_sha256": ch, "id": r["id"]}) + "\n")
                for f, h, p in zip(fh[role], hs[role], parts):
                    f.write(p)
                    h.update(p.encode())
                per[role][fam] += 1
                ids[role].append(r["id"])
    finally:
        for fs in fh.values():
            for f in fs:
                f.close()
    clash = content["train"] & content["calib"]
    if clash:
        raise SystemExit(f"{len(clash)} content hash(es) in both train and calib (rebuild with another seed)")
    rep = {"name": a.name, "seed": a.seed, "sources": [str(s) for s in srcs], "source_family_rows": counts,
           "train_per_family": a.train_per_family, "calib_per_family": a.calib_per_family, "short_families": short,
           "train_rows": dict(per["train"]), "calib_rows": dict(per["calib"]),
           "n_train": sum(per["train"].values()), "n_calib": sum(per["calib"].values()), "content_disjoint": True,
           # sha256 of the sorted record ids, newline-joined (== jobs.m2_first_cell.screen.slice_ids_sha256)
           "train_ids_sha256": hashlib.sha256("\n".join(sorted(ids["train"])).encode()).hexdigest(),
           "calib_ids_sha256": hashlib.sha256("\n".join(sorted(ids["calib"])).encode()).hexdigest(),
           "rule": "per family, ascending sha256('<seed>:<id>'): first train_per_family -> train, next calib_per_family "
                   "-> calib (label-blind)"}
    for role, d in dirs.items():
        h = hs[role]
        ptr = {"build": f"{d.name} (scripts/jev12_balanced_draw.py over {', '.join(s.name for s in srcs)})",
               "location": f"ssdm-data/m2/build/{d.name}/ (outside git; deterministic)",
               "records_uncompressed_sha256": h[0].hexdigest(),
               "manifest": f"{d.name}.manifest.jsonl.gz", "manifest_uncompressed_sha256": h[1].hexdigest(),
               "content_map": f"{d.name}.content.jsonl.gz", "content_map_uncompressed_sha256": h[2].hexdigest(),
               "n_records": sum(per[role].values()), "role": role,
               "ids_sha256": hashlib.sha256("\n".join(sorted(ids[role])).encode()).hexdigest(),
               "schema": "ssdm-train-v0, one question per row (qid 'q'), JevBench request shape",
               "excludes_cohort": "inherited from every source draw's own gates (each source pointer's excludes_cohort)",
               "builder": {"module": "scripts/jev12_balanced_draw.py", "seed": a.seed},
               "templates_verified": ["jrt-h2o-shim:qwen3.5-4b"]}
        (d / "pointer.json").write_text(json.dumps(ptr, indent=1) + "\n", encoding="utf-8")
        rep[f"{role}_pointer"] = ptr
    (dirs["train"] / "draw-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_train", "n_calib", "train_rows", "short_families")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
