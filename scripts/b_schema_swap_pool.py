"""Derive a draw from a pointer-pinned b_schema draw by REPLACING one source's rows with a pointer-pinned extra pool.

Drops every parent row whose ``source`` is in ``--drop-source`` (and any id in ``--drop-ids`` files, same formats as
scripts/b_schema_subtract.py), then adds the rows of ``--add-pool`` (Agent C's train_set dir; its pointer.json must hash
to ``--add-pointer-sha256``, and its three files must match the pointer's uncompressed sha256s). Every pool row passes
the same ``data.b_schema.Checker`` as the select stage's P3 rows (structure, D1 templates, cohort v2/v3 id / item /
content / text, exact-content duplicates and key-free near-duplicates across items against the kept parent rows,
SplitGuard, frozen families, Index content / text / item text), and is written unchanged except for
``aug.schema_synth = {"ops": ["extra_pool_passthrough"], "pool": "P3"}`` (canonical JSON), as ``append_extra`` does.
Parent lines are byte copies; the output is the id-sorted merge (the draw convention), streamed. ``--drop-pool-ids``
removes listed pool ids (e.g. rows a later gate refuses). Writes the draw dir, the pointer and ``swap-report.json``.

    python -X utf8 scripts/b_schema_swap_pool.py --parent-pointer data/manifests/bschema-draw230k-v2d.pointer.json
        --build bschema-draw230k-v2e --drop-source synthetic_c_multirecord_v1 --add-pool <dir> --add-pointer-sha256 <sha>
        --receipt <dir>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import heapq
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _lines(p: Path):
    with gzip.open(p, "rt", encoding="utf-8", newline="\n") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def main(argv=None) -> int:
    from data import config
    from data.b_schema import Checker, canonical, gz_text, h64, root_items, sha
    from data.b1 import dedup_keys, exclusion_sets
    from train.cohort import record_content_hash
    from train.train_set import read_pointer, verify_files
    from b_schema_subtract import _ids_from

    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pointer", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--drop-source", action="append", default=[])
    ap.add_argument("--drop-ids", action="append", default=[])
    ap.add_argument("--add-pool", required=True)
    ap.add_argument("--add-pointer-sha256", required=True)
    ap.add_argument("--drop-pool-ids", action="append", default=[])
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--keep-aug", action="store_true",
                    help="keep the pool rows' own aug.schema_synth (lever rows carry their view ops)")
    a = ap.parse_args(argv)

    pp = Path(a.parent_pointer)
    ptr = read_pointer(pp)
    parent = ptr["build"].split()[0]
    pdir = config.BUILD_DIR / parent
    pfiles = verify_files(pdir, ptr)
    pool = Path(a.add_pool)
    cptr = read_pointer(pool / "pointer.json")
    if cptr["_pointer_sha256"] != a.add_pointer_sha256:
        raise SystemExit(f"pool pointer sha {cptr['_pointer_sha256']} != {a.add_pointer_sha256}")
    cfiles = verify_files(pool, cptr)
    drop_ids = set()
    for f in a.drop_ids:
        drop_ids |= set(_ids_from(Path(f)))
    drop_pool = set()
    for f in a.drop_pool_ids:
        drop_pool |= set(_ids_from(Path(f)))
    srcs = set(a.drop_source)

    # parent: kept ids, seed the checker with every kept row (content + key-free text ownership)
    chk = Checker(exclusion_sets())
    keep = set()
    removed = collections.Counter()
    n_parent = 0
    for line in _lines(pdir / "records.jsonl.gz"):
        r = json.loads(line)
        n_parent += 1
        if r["source"] in srcs or r["id"] in drop_ids:
            removed[r["source"]] += 1
            continue
        keep.add(r["id"])
        ih = h64(sorted(root_items(r))[0])
        chk.content.add(h64(record_content_hash(r)))
        for k in dedup_keys(r):
            chk.text_owner.setdefault(h64(k), ih)
    # pool rows through the P3 checks
    added = []  # (id, canonical line)
    n_pool = 0
    dropped_listed = 0
    for line in _lines(pool / "records.jsonl.gz"):
        r = json.loads(line)
        n_pool += 1
        if r["id"] in drop_pool:
            dropped_listed += 1
            continue
        if r["id"] in keep:
            raise SystemExit(f"pool id {r['id']} collides with a kept parent id")
        if chk.accept(r, "P3", h64(sorted(root_items(r))[0])):
            if a.keep_aug:
                r["aug"]["schema_synth"] = {**r["aug"].get("schema_synth", {}), "pool": "P3"}
            else:
                r.setdefault("aug", {})["schema_synth"] = {"ops": ["extra_pool_passthrough"], "pool": "P3"}
            added.append((r["id"], canonical(r)))
    added.sort()
    add_src = collections.Counter(json.loads(x)["source"] for _, x in added)

    def parent_stream():
        last = None
        for line in _lines(pdir / "records.jsonl.gz"):
            rid = json.loads(line)["id"]
            if last is not None and rid <= last:
                raise SystemExit("parent records are not id-sorted")
            last = rid
            if rid in keep:
                yield rid, line

    out = config.BUILD_DIR / a.build
    if (out / "records.jsonl.gz").exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    rec_h, man_h, con_h = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n = 0
    n_items = 0
    by_src = collections.Counter()
    items_src = collections.Counter()
    with gz_text(out / "records.jsonl.gz") as fr, gz_text(out / f"{a.build}.manifest.jsonl.gz") as fm, \
            gz_text(out / f"{a.build}.content.jsonl.gz") as fc:
        last = None
        for rid, line in heapq.merge(parent_stream(), iter(added)):
            if last is not None and rid <= last:
                raise SystemExit(f"duplicate / unsorted id {rid}")
            last = rid
            r = json.loads(line)
            fr.write(line + "\n")
            rec_h.update((line + "\n").encode("utf-8"))
            m = canonical({"family": r["family"], "id": r["id"], "sha256": sha(line), "source": r["source"]}) + "\n"
            fm.write(m)
            man_h.update(m.encode("utf-8"))
            c = canonical({"content_sha256": record_content_hash(r), "id": r["id"]}) + "\n"
            fc.write(c)
            con_h.update(c.encode("utf-8"))
            n += 1
            by_src[r["source"]] += 1
            items_src[r["source"]] += len(r["questions"])
            n_items += len(r["questions"])
    shas = {"records": rec_h.hexdigest(), "manifest": man_h.hexdigest(), "content": con_h.hexdigest()}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    raw = json.loads(pp.read_text(encoding="utf-8"))
    npt = dict(raw)
    npt.update({"build": f"{a.build} (scripts/b_schema_swap_pool.py: {parent} with {sorted(srcs)} replaced by "
                         f"{cptr.get('build', '').split()[0]}; P3 checks)",
                "location": f"ssdm-data/m2/build/{a.build}/ (outside git; deterministic, a rerun reproduces it)",
                "records_uncompressed_sha256": shas["records"], "manifest": f"{a.build}.manifest.jsonl.gz",
                "manifest_uncompressed_sha256": shas["manifest"], "content_map": f"{a.build}.content.jsonl.gz",
                "content_map_uncompressed_sha256": shas["content"], "n_records": n,
                "derived_from": {"pointer": f"data/manifests/{pp.name}", "pointer_sha256": ptr["_pointer_sha256"],
                                 "records_uncompressed_sha256": ptr["records_uncompressed_sha256"],
                                 "n_records": n_parent},
                "extra_pool": {"dir": str(pool).replace("\\", "/").split("ssdm-data/")[-1],
                               "pointer_sha256": cptr["_pointer_sha256"], "build": cptr.get("build"),
                               "files_uncompressed_sha256": cfiles, "builder": cptr.get("builder")},
                "builder": {"module": "scripts/b_schema_swap_pool.py", "commit": commit,
                            "parent_builder": raw.get("builder")},
                "receipt": f"receipts/2026-10-08-B1-schema/{a.build}/"})
    npt.pop("_pointer_sha256", None)
    npt.pop("_pointer_path", None)
    npp = config.MANIFEST_DIR / f"{a.build}.pointer.json"
    npp.write_bytes((json.dumps(npt, indent=1) + "\n").encode("utf-8"))
    rep = {"parent": parent, "parent_pointer_sha256": ptr["_pointer_sha256"], "parent_files": pfiles,
           "n_parent": n_parent, "removed_from_parent": dict(removed.most_common()),
           "pool": {"pointer_sha256": cptr["_pointer_sha256"], "files": cfiles, "n_rows": n_pool,
                    "n_dropped_by_list": dropped_listed, "n_added": len(added), "added_by_source": dict(add_src),
                    "checker": {k: dict(v) for k, v in chk.reasons.items()}, "hit_examples": chk.hit_examples},
           "n_records": n, "n_items": n_items, "rows_by_source": dict(by_src.most_common()),
           "items_by_source": dict(items_src.most_common()),
           "item_share_by_source": {k: round(v / n_items, 4) for k, v in items_src.most_common()},
           "uncompressed_sha256": shas, "pointer": str(npp),
           "pointer_sha256": hashlib.sha256(npp.read_bytes()).hexdigest(), "commit": commit}
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "swap-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_parent", "removed_from_parent", "n_records", "n_items",
                                          "uncompressed_sha256")} | {"pool": {k: rep["pool"][k] for k in (
        "n_rows", "n_added", "checker")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
