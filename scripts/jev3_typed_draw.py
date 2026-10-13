"""JEV3-TYPED training draw: the jev3 train set (verbatim) + routing / intent / severity (the ordinal score scales) rows
from the gated jev1 build, so no public JevBench family TYPE is untrained (lead 2026-10-12).

  python -X utf8 -I scripts/jev3_typed_draw.py --data <ssdm-data> [--per-family 600] [--seed 20261012] \
      [--out-name bschema-draw-jev3-typed]

Rules (each asserted; the build fails closed):
  * jev3 rows: every row of bschema-draw-jev3 (its pointer's three uncompressed sha256 checked), byte-identical lines.
  * added rows: jev1 build rows whose ``aug.*.family8`` is routing / intent / severity, ranked by sha256("<seed>:<id>"),
    the first --per-family per family that pass: not a score row with all-empty level rubrics; id not in the jev2-mix
    exclusion lists (c0c175f5..., 5cebf60b...; sha256-checked); and DISJOINT from the jev3 selection slice, the jev3
    train rows, the public JevBench items and every other added row by id, (source, source_id), content hash and
    normalised state (canonical JSON; and whitespace-collapsed casefolded text).
  * outputs: records.jsonl.gz, <name>.manifest.jsonl.gz, <name>.content.jsonl.gz (gzip mtime 0: reproducible) and
    draw-report.json with every count; prints the pointer fields (uncompressed sha256s, ids sha256, families).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAMILIES = ("routing", "intent", "severity")
JEV2_EXCLUDES = (("receipts/2026-10-11-jev2-final-regate/mix_exclude_ids.txt",
                  "c0c175f50a7886b53bc647e022bb00705747deadf535a2708fef19c32ab9c2ad", 23),
                 ("receipts/2026-10-11-jev2a-v2-regate/jev2a-v2_musique_provenance_unverified_ids.txt",
                  "5cebf60bfa489d462a7e7d9b82e152a4968f30e1dc7ea50a2155e491a1ef37ab", 104))


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lines(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def unc_sha(p: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_gz(p: Path, rows: list[str]) -> str:
    data = "".join(r + "\n" for r in rows).encode("utf-8")
    with open(p, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as g:
        g.write(data)
    return hashlib.sha256(data).hexdigest()


def family_of(r: dict) -> str | None:  # == scripts/jev12_balanced_draw.family_of
    for v in (r.get("aug") or {}).values():
        if isinstance(v, dict) and (v.get("family8") or v.get("family")):
            return v.get("family8") or v.get("family")
    return None


def empty_levels(q: dict) -> bool:  # == scripts/jev12_balanced_draw._empty_levels
    crit = q.get("criteria")
    vals = list(crit.values()) if isinstance(crit, dict) else list(crit or [])
    return q.get("type") == "score" and not any(str(v).strip() for v in vals if v is not None)


def state_keys(state) -> tuple[str, str]:
    """(canonical-JSON key == screen._state_key, normalised-text key)."""
    a = hashlib.sha256(canonical(state).encode()).hexdigest()[:24]
    txt = state if isinstance(state, str) else canonical(state)
    b = hashlib.sha256(re.sub(r"\s+", " ", txt).strip().casefold().encode()).hexdigest()[:24]
    return a, b


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-family", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out-name", default="bschema-draw-jev3-typed")
    ap.add_argument("--out-dir", default=None, help="default: <data>/m2/build/<out-name>")
    ap.add_argument("--public-dir", default=None, help="the pinned public JevBench files (default: jevkit's)")
    a = ap.parse_args(argv)
    data = Path(a.data)
    b3 = data / "m2/build/bschema-draw-jev3"
    sl = data / "m2/build/bschema-draw-jev3-slice"
    b1 = data / "m2/build/bschema-draw-jev1"
    p3 = json.loads((REPO / "data/manifests/bschema-draw-jev3.pointer.json").read_text(encoding="utf-8"))
    for f, k in (("records.jsonl.gz", "records_uncompressed_sha256"), (p3["manifest"], "manifest_uncompressed_sha256"),
                 (p3["content_map"], "content_map_uncompressed_sha256")):
        got = unc_sha(b3 / f)
        assert got == p3[k], f"jev3 {f}: {got} != pointer {p3[k]}"
    p1 = json.loads((REPO / "data/manifests/bschema-draw-jev1.pointer.json").read_text(encoding="utf-8"))
    assert unc_sha(b1 / "records.jsonl.gz") == p1["records_uncompressed_sha256"], "jev1 records != its pointer"
    excl = set()
    for f, sha, n in JEV2_EXCLUDES:
        b = (REPO / f).read_bytes()
        assert hashlib.sha256(b).hexdigest() == sha, f"{f} sha256 != {sha}"
        ids = {x.strip() for x in b.decode("utf-8").splitlines() if x.strip()}
        assert len(ids) == n, f"{f}: {len(ids)} ids != {n}"
        excl |= ids
    # ---- what the added rows must be disjoint from
    taken_id, taken_src, taken_c, taken_s = set(), set(), set(), set()

    def take(r: dict, content: str | None):
        taken_id.add(r["id"])
        if r.get("source") is not None and r.get("source_id") is not None:
            taken_src.add((str(r["source"]), str(r["source_id"])))
        if content:
            taken_c.add(content)
        taken_s.update(state_keys(r.get("state")))

    c3 = {json.loads(x)["id"]: json.loads(x)["content_sha256"] for x in lines(b3 / p3["content_map"])}
    j3_lines = list(lines(b3 / "records.jsonl.gz"))
    for x in j3_lines:
        r = json.loads(x)
        take(r, c3.get(r["id"]))
    ps = json.loads((REPO / "data/manifests/bschema-draw-jev3-slice.pointer.json").read_text(encoding="utf-8"))
    cs = {json.loads(x)["id"]: json.loads(x)["content_sha256"] for x in lines(sl / ps["content_map"])}
    n_slice = 0
    for x in lines(sl / "records.jsonl.gz"):
        r = json.loads(x)
        take(r, cs.get(r["id"]))
        n_slice += 1
    from jevkit.items import load_public

    pub = load_public(a.public_dir)
    for it in pub:  # public JevBench items: eval-only; their states must never appear in training
        taken_s.update(state_keys(it.state))
    # ---- candidates
    m1 = {}
    for x in lines(b1 / "bschema-draw-jev1.manifest.jsonl.gz"):
        m1[json.loads(x)["id"]] = x
    cm1 = {}
    for x in lines(b1 / "bschema-draw-jev1.content.jsonl.gz"):
        cm1[json.loads(x)["id"]] = x
    cand = {f: [] for f in FAMILIES}
    for x in lines(b1 / "records.jsonl.gz"):
        r = json.loads(x)
        f = family_of(r)
        if f in cand:
            cand[f].append((hashlib.sha256(f"{a.seed}:{r['id']}".encode()).hexdigest(), r["id"], x))
    drop = Counter()
    added: list[tuple[str, str]] = []
    per = Counter()
    for f in FAMILIES:
        for _, rid, x in sorted(cand[f]):
            if per[f] >= a.per_family:
                break
            r = json.loads(x)
            q = next(iter(r["questions"].values()))
            content = json.loads(cm1[rid])["content_sha256"] if rid in cm1 else None
            sk = state_keys(r.get("state"))
            why = ("empty_levels" if empty_levels(q) else "jev2_exclusion" if rid in excl
                   else "no_manifest" if rid not in m1 or rid not in cm1
                   else "id" if rid in taken_id
                   else "source_item" if (str(r.get("source")), str(r.get("source_id"))) in taken_src
                   else "content" if content in taken_c
                   else "state" if (sk[0] in taken_s or sk[1] in taken_s) else None)
            if why:
                drop[f"{f}:{why}"] += 1
                continue
            take(r, content)
            added.append((rid, x))
            per[f] += 1
    short = {f: per[f] for f in FAMILIES if per[f] < a.per_family}
    assert not short, f"not enough disjoint rows: {short}"
    out = Path(a.out_dir) if a.out_dir else data / "m2/build" / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    m3 = list(lines(b3 / p3["manifest"]))
    cm3 = list(lines(b3 / p3["content_map"]))
    rec = j3_lines + [x for _, x in added]
    man = m3 + [m1[rid] for rid, _ in added]
    con = cm3 + [cm1[rid] for rid, _ in added]
    ids = [json.loads(x)["id"] for x in rec]
    assert len(set(ids)) == len(ids) and {json.loads(x)["id"] for x in man} == set(ids) == {json.loads(x)["id"] for x in con}
    shas = {"records_uncompressed_sha256": write_gz(out / "records.jsonl.gz", rec),
            "manifest_uncompressed_sha256": write_gz(out / f"{a.out_name}.manifest.jsonl.gz", man),
            "content_map_uncompressed_sha256": write_gz(out / f"{a.out_name}.content.jsonl.gz", con)}
    fams = Counter()
    for x in rec:
        r = json.loads(x)
        fams[family_of(r) or (r.get("family") or "?")] += 1
    rep = {"name": a.out_name, "seed": a.seed, "per_family": a.per_family, "n_records": len(rec),
           "n_jev3": len(j3_lines), "n_added": len(added), "added_per_family": dict(per),
           "added_ids_sha256": ids_sha([rid for rid, _ in added]), "ids_sha256": ids_sha(ids),
           "dropped": dict(sorted(drop.items())), "slice_rows_checked": n_slice, "public_items_checked": len(pub),
           "jev2_exclusions": [{"file": f, "sha256": s, "n": n} for f, s, n in JEV2_EXCLUDES],
           "disjoint_rule": "added rows vs jev3 slice + jev3 train + public items + each other: 0 shared ids, "
                            "(source, source_id), content hashes, canonical / normalised states",
           "families": dict(sorted(fams.items())), **shas}
    (out / "draw-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
