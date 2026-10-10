"""Derive a draw from a pointer-pinned b_schema draw by SUBTRACTION only (no new rows, no re-selection).

Drops every row whose ``source`` is in ``--drop-source`` and every id listed in ``--drop-ids`` files (JSON receipts:
a ``fails`` list of {"id"} (data.doc_check) or a ``fail_ids`` list (train.doc_overlap_guard scan), or plain text, one
id per line). Record / manifest / content lines are copied byte-for-byte from the parent, in the parent's order, and
written with the deterministic gzip writer (mtime 0). The parent's uncompressed sha256s are verified against its pointer
first. Writes the draw dir, the pointer (same keys as data.b_schema.write_pointer + ``derived_from``) and a
``subtract-report.json`` receipt.

    python -X utf8 -I scripts/b_schema_subtract.py --parent-pointer data/manifests/bschema-draw230k-v2c.pointer.json
        --build bschema-draw230k-v2d --drop-source nli4ct --drop-ids <doc-check.json> [--drop-ids ...] --receipt <dir>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _ids_from(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8")
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        return [x.strip() for x in txt.splitlines() if x.strip()]
    if isinstance(d, list):
        return [x if isinstance(x, str) else x["id"] for x in d]
    if "fail_ids" in d:
        return list(d["fail_ids"])
    if "fails" in d:
        if d.get("n_fail") is not None and d["n_fail"] != len(d["fails"]):
            raise SystemExit(f"{path}: fails list truncated ({len(d['fails'])} of {d['n_fail']})")
        return [x["id"] for x in d["fails"]]
    raise SystemExit(f"{path}: no fail_ids / fails list")


def _lines(p: Path):
    with gzip.open(p, "rt", encoding="utf-8", newline="\n") as fh:
        for line in fh:
            if line.strip():
                yield line.rstrip("\n")


def main(argv=None) -> int:
    from data import config
    from data.b_schema import gz_text

    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pointer", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--drop-source", action="append", default=[])
    ap.add_argument("--drop-ids", action="append", default=[])
    ap.add_argument("--receipt", required=True)
    a = ap.parse_args(argv)

    pp = Path(a.parent_pointer)
    ptr = json.loads(pp.read_text(encoding="utf-8"))
    parent = ptr["build"].split()[0]
    pdir = config.BUILD_DIR / parent
    files = {"records": (pdir / "records.jsonl.gz", "records.jsonl.gz", ptr["records_uncompressed_sha256"]),
             "manifest": (pdir / ptr["manifest"], f"{a.build}.manifest.jsonl.gz", ptr["manifest_uncompressed_sha256"]),
             "content": (pdir / ptr["content_map"], f"{a.build}.content.jsonl.gz",
                         ptr["content_map_uncompressed_sha256"])}
    drop_ids: dict[str, list[str]] = {}
    for f in a.drop_ids:
        drop_ids[f] = _ids_from(Path(f))
    want = set().union(*drop_ids.values()) if drop_ids else set()
    srcs = set(a.drop_source)

    # pass 1: verify the parent and decide the kept ids from the records
    h = hashlib.sha256()
    keep, n_parent = set(), 0
    removed_src, removed_ids_src = collections.Counter(), collections.Counter()
    seen_ids = set()
    for line in _lines(files["records"][0]):
        h.update((line + "\n").encode("utf-8"))
        r = json.loads(line)
        n_parent += 1
        seen_ids.add(r["id"])
        if r["source"] in srcs:
            removed_src[r["source"]] += 1
        elif r["id"] in want:
            removed_ids_src[r["source"]] += 1
        else:
            keep.add(r["id"])
    if h.hexdigest() != files["records"][2]:
        raise SystemExit(f"parent records sha {h.hexdigest()} != pointer {files['records'][2]}")
    missing = sorted(want - seen_ids)
    if missing:
        raise SystemExit(f"{len(missing)} drop ids not in the parent, e.g. {missing[:3]}")

    out = config.BUILD_DIR / a.build
    if (out / "records.jsonl.gz").exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    shas = {}
    for key, (src, name, pin) in files.items():
        hp, ho, n = hashlib.sha256(), hashlib.sha256(), 0
        with gz_text(out / name) as fo:
            for line in _lines(src):
                hp.update((line + "\n").encode("utf-8"))
                if json.loads(line)["id"] in keep:
                    fo.write(line + "\n")
                    ho.update((line + "\n").encode("utf-8"))
                    n += 1
        if hp.hexdigest() != pin:
            raise SystemExit(f"parent {key} sha {hp.hexdigest()} != pointer {pin}")
        if n != len(keep):
            raise SystemExit(f"{key}: wrote {n} lines, kept {len(keep)} ids")
        shas[key] = ho.hexdigest()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    npt = dict(ptr)
    npt.update({"build": f"{a.build} (scripts/b_schema_subtract.py: {parent} minus dropped rows; no new rows)",
                "location": f"ssdm-data/m2/build/{a.build}/ (outside git; deterministic, a rerun reproduces it)",
                "records_uncompressed_sha256": shas["records"], "manifest": files["manifest"][1],
                "manifest_uncompressed_sha256": shas["manifest"], "content_map": files["content"][1],
                "content_map_uncompressed_sha256": shas["content"], "n_records": len(keep),
                "derived_from": {"pointer": pp.as_posix().split("ssdm-wt/")[-1].split("/", 1)[-1] if "ssdm-wt/" in pp.as_posix() else pp.as_posix(),
                                 "pointer_sha256": hashlib.sha256(pp.read_bytes()).hexdigest(),
                                 "records_uncompressed_sha256": ptr["records_uncompressed_sha256"],
                                 "n_records": n_parent},
                "builder": {"module": "scripts/b_schema_subtract.py", "commit": commit,
                            "parent_builder": ptr.get("builder")},
                "receipt": f"receipts/2026-10-08-B1-schema/{a.build}/"})
    npp = config.MANIFEST_DIR / f"{a.build}.pointer.json"
    npp.write_bytes((json.dumps(npt, indent=1) + "\n").encode("utf-8"))
    rep = {"parent": parent, "parent_pointer_sha256": npt["derived_from"]["pointer_sha256"], "n_parent": n_parent,
           "n_records": len(keep), "n_removed": n_parent - len(keep),
           "removed_by_source_rule": {"sources": sorted(srcs), "rows": dict(removed_src.most_common())},
           "removed_by_id_lists": {"rows_by_source": dict(removed_ids_src.most_common()),
                                   "lists": {Path(f).name: {"n_ids": len(v), "sha256_sorted_ids": hashlib.sha256(
                                       "\n".join(sorted(v)).encode("utf-8")).hexdigest()} for f, v in drop_ids.items()},
                                   "n_ids_union": len(want),
                                   "n_ids_already_removed_by_source": sum(1 for _ in want) - sum(removed_ids_src.values())},
           "uncompressed_sha256": shas, "pointer": str(npp), "commit": commit}
    rd = Path(a.receipt)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "subtract-report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
