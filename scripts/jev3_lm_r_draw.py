"""JEV3-LM-R training draw (A-PREREG-JEV3-LM-R): the JEV3-TYPED draw (bschema-draw-jev3-typed = jev3 verbatim + 600
each of jev1 routing / intent / severity, scripts/jev3_typed_draw.py @ff06d4c0, built by job-stbvu) with the ADDED
choice rows' option counts subsampled to the public JevBench K of their family.

  python -X utf8 -I scripts/jev3_lm_r_draw.py --data <ssdm-data> [--out-name bschema-draw-jev3-lmr]

Rules (each asserted; the build fails closed):
  * input = <data>/m2/build/bschema-draw-jev3-typed, its three uncompressed sha256s == the typed pointer
    (data/manifests/bschema-draw-jev3-typed.pointer.json), its ids sha == the pointer's.
  * jev3 rows (no ``aug.*.family8`` in routing / intent / severity): byte-identical lines, manifest and content lines.
  * added rows: a CHOICE row whose K exceeds the public K of its family (max K over the pinned public items of that
    JevBench family: routing 6 (routing_hard 5 is covered), intent 5; computed from jevkit's pinned public files and
    asserted) keeps the gold option + the K_pub - 1 distractors ranked first by sha256("jev3-lm-r-k:<id>:<key>"), in
    their original relative order (training rows are re-shuffled per row by g1v2-canon-shuffle anyway). Rows with
    K <= K_pub, yes/no rows and score rows (severity, K 4-5 levels; a scale is not subsampled) are unchanged.
  * changed rows are re-serialised compactly (== the build's own serialisation, asserted on every unchanged row);
    their manifest sha256 (of the line) and content_sha256 (train.cohort.record_content_hash) are recomputed; every
    unchanged row's content hash is re-derived and must equal the typed content map.
  * outputs: records.jsonl.gz, <name>.manifest.jsonl.gz, <name>.content.jsonl.gz (gzip mtime 0: reproducible),
    draw-report.json (every count, the K histograms before / after); prints the pointer fields.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FAMILIES = ("routing", "intent", "severity")
#: the public JevBench families whose K each added family is subsampled to (max K over them)
PUBLIC_FAMILIES = {"routing": ("routing", "routing_hard"), "intent": ("intent",)}
EXPECT_PUBLIC_K = {"routing": 6, "intent": 5}
TYPED_POINTER = "data/manifests/bschema-draw-jev3-typed.pointer.json"
SALT = "jev3-lm-r-k"


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


def compact(o) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def canonical(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def family_of(r: dict) -> str | None:  # == scripts/jev3_typed_draw.family_of
    for v in (r.get("aug") or {}).values():
        if isinstance(v, dict) and (v.get("family8") or v.get("family")):
            return v.get("family8") or v.get("family")
    return None


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def public_k(public_dir: str | None = None) -> dict[str, int]:
    from jevkit.items import load_public

    ks: dict[str, int] = {}
    for it in load_public(public_dir):
        for fam, pubs in PUBLIC_FAMILIES.items():
            if it.family in pubs and it.question.get("type") == "choice":
                ks[fam] = max(ks.get(fam, 0), len(it.question.get("criteria") or {}))
    return ks


def subsample(r: dict, k_pub: int) -> dict | None:
    """The row with its choice options cut to k_pub (gold kept), or None when unchanged."""
    q = r["questions"]["q"]
    crit = q.get("criteria")
    if q.get("type") != "choice" or not isinstance(crit, dict) or len(crit) <= k_pub:
        return None
    gold = r["labels"]["q"]
    if gold not in crit:
        raise ValueError(f"{r['id']}: gold {gold!r} is not an option key")
    rank = sorted((k for k in crit if k != gold),
                  key=lambda k: hashlib.sha256(f"{SALT}:{r['id']}:{k}".encode()).hexdigest())
    keep = {gold, *rank[:k_pub - 1]}
    out = json.loads(json.dumps(r))
    out["questions"]["q"]["criteria"] = {k: v for k, v in crit.items() if k in keep}
    out["aug"]["jev3_lm_r_k"] = {"k_from": len(crit), "k_to": k_pub, "rule": f"gold + sha256('{SALT}:<id>:<key>') first"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-name", default="bschema-draw-jev3-lmr")
    ap.add_argument("--out-dir", default=None, help="default: <data>/m2/build/<out-name>")
    ap.add_argument("--public-dir", default=None)
    a = ap.parse_args(argv)
    from train.cohort import record_content_hash

    data = Path(a.data)
    src = data / "m2/build/bschema-draw-jev3-typed"
    pt = json.loads((REPO / TYPED_POINTER).read_text(encoding="utf-8"))
    for f, k in (("records.jsonl.gz", "records_uncompressed_sha256"), (pt["manifest"], "manifest_uncompressed_sha256"),
                 (pt["content_map"], "content_map_uncompressed_sha256")):
        got = unc_sha(src / f)
        assert got == pt[k], f"typed {f}: {got} != pointer {pt[k]}"
    kp = public_k(a.public_dir)
    assert kp == EXPECT_PUBLIC_K, f"public K {kp} != {EXPECT_PUBLIC_K}"
    man_in = {json.loads(x)["id"]: x for x in lines(src / pt["manifest"])}
    con_in = {json.loads(x)["id"]: x for x in lines(src / pt["content_map"])}
    rec_out, man_out, con_out, ids = [], [], [], []
    k_before, k_after, changed = Counter(), Counter(), Counter()
    for x in lines(src / "records.jsonl.gz"):
        r = json.loads(x)
        assert compact(r) == x, f"{r['id']}: the build's serialisation is not compact JSON"
        rid = r["id"]
        ids.append(rid)
        f = family_of(r)
        new = subsample(r, kp[f]) if f in kp else None
        if f in FAMILIES:
            q = r["questions"]["q"]
            k_before[f"{f}:{q['type']}:{len(q.get('criteria') or {})}"] += 1
        if new is None:
            want = json.loads(con_in[rid])["content_sha256"]
            assert record_content_hash(r) == want, f"{rid}: content hash != typed content map"
            rec_out.append(x)
            man_out.append(man_in[rid])
            con_out.append(con_in[rid])
            if f in FAMILIES:
                k_after[f"{f}:{q['type']}:{len(q.get('criteria') or {})}"] += 1
            continue
        line = compact(new)
        m = json.loads(man_in[rid])
        m["sha256"] = hashlib.sha256(line.encode("utf-8")).hexdigest()
        rec_out.append(line)
        man_out.append(canonical(m))
        con_out.append(canonical({"content_sha256": record_content_hash(new), "id": rid}))
        changed[f] += 1
        q = new["questions"]["q"]
        k_after[f"{f}:{q['type']}:{len(q['criteria'])}"] += 1
    assert ids_sha(ids) == pt["ids_sha256"] and len(ids) == pt["n_records"] == len(set(ids))
    assert set(man_in) == set(con_in) == set(ids)
    out = Path(a.out_dir) if a.out_dir else data / "m2/build" / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    shas = {"records_uncompressed_sha256": write_gz(out / "records.jsonl.gz", rec_out),
            "manifest_uncompressed_sha256": write_gz(out / f"{a.out_name}.manifest.jsonl.gz", man_out),
            "content_map_uncompressed_sha256": write_gz(out / f"{a.out_name}.content.jsonl.gz", con_out)}
    gz = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.gz"))}
    rep = {"name": a.out_name, "source": {"pointer": TYPED_POINTER, "ids_sha256": pt["ids_sha256"]},
           "public_k": kp, "salt": SALT, "n_records": len(ids), "ids_sha256": ids_sha(ids),
           "n_changed": sum(changed.values()), "changed_per_family": dict(changed),
           "k_before": dict(sorted(k_before.items())), "k_after": dict(sorted(k_after.items())),
           "gz_sha256": gz, **shas}
    (out / "draw-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
