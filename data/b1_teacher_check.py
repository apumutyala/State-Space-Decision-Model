"""Exclusion check of a KD teacher export keyed to the 150k pool (BOARD B1 add-on, requested by Agent A).

Each export row's id is resolved to its 150k record (pointer-verified files; the row's ``content_sha256`` must equal
``train.cohort.record_content_hash`` of that record), then every record is checked exactly as ``data.b1_check`` checks
a draw: ``harness.heldout.check_record`` per row and ``check_generator`` per (source, family); cohort v2 / v3
``assert_draw_disjoint`` (id, content, source item, key-free text); SplitGuard (DEV/TEST manifests + content index);
Index content / key-free text / item-text overlap; ``train.heldout`` D1-B fragments. Read-only on the export.

    python -X utf8 -I -m data.b1_teacher_check --export <dir with teacher-*.jsonl> [--out report.json]
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    import argparse
    from data import b1, b1_check as C
    from data.build import content_guard, item_keys
    from train.cohort import Cohort, record_content_hash, record_text_keys
    from train.cohort_v3 import _ensure_kit, cohort_v3
    from train.guard import ForbiddenRow
    from train.heldout import record_text, scan
    from train.train_set import read_pointer, verify_files

    ap = argparse.ArgumentParser(prog="data.b1_teacher_check")
    ap.add_argument("--export", required=True)
    ap.add_argument("--data", default=r"C:\Users\apumu\research\CHLabs\ssdm-data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    exp = Path(a.export)
    files = sorted(exp.glob("teacher-*.jsonl"))
    shas = {p.name: sha_file(p) for p in sorted(exp.iterdir()) if p.is_file()}
    rows = {}
    for p in files:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    e = json.loads(line)
                    rows[e["id"]] = e
    ptr = read_pointer(b1.PARENT_POINTER)
    pdir = Path(a.data) / "m2" / "build" / b1.PARENT
    verify_files(pdir, ptr)
    recs = {}
    with gzip.open(pdir / "records.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in rows:
                recs[r["id"]] = r
    fails: dict[str, list] = collections.defaultdict(list)
    missing = sorted(set(rows) - set(recs))
    if missing:
        fails["id_not_in_150k"] = missing
    _ensure_kit()
    from harness.heldout import check_generator, check_record
    v2, v3 = Cohort(), cohort_v3()
    guard, _ = content_guard()
    idx = C.build_index(Path(a.data))
    templates = C.instruction_templates()
    content = {}
    for rid, r in recs.items():
        ch = record_content_hash(r)
        content[rid] = ch
        if rows[rid].get("content_sha256") != ch:
            fails["content_sha256_mismatch"].append(rid)
        try:
            check_record(r)
        except Exception:
            fails["frozen_family"].append(rid)
        try:
            guard.check(r)
        except ForbiddenRow:
            fails["splitguard"].append(rid)
        if ch in idx["content"]:
            fails["index_content"].append(rid)
        if record_text_keys(r) & idx["text"]:
            fails["index_text"].append(rid)
        for hit, path, s, ctx in C.item_hits(r, idx, templates):
            if not ctx:
                fails[f"index_item_{hit}"].append((rid, path, s[:80]))
    gens = sorted({(r["source"], r["family"]) for r in recs.values()})
    for src, fam in gens:
        try:
            check_generator({"name": src, "task_family": fam, "targets": [src, fam]})
        except Exception as e:
            fails["frozen_generator"].append(f"{src}/{fam}: {e}")
    ids = sorted(recs)
    for name, co in (("v2", v2), ("v3", v3)):
        try:
            co.assert_draw_disjoint(ids, content, records=[recs[i] for i in ids])
        except Exception as e:
            fails[f"assert_draw_disjoint_{name}"].append(str(e)[:300])
        held_items = set().union(*(item_keys(i) for i in co.held_out()))
        bad = [i for i in ids if item_keys(i) & held_items]
        if bad:
            fails[f"cohort_{name}_source_item"].extend(bad)
    d1b = scan((i, record_text(recs[i])) for i in ids)
    if d1b["hits"]:
        fails["d1b_heldout"] = d1b["hits"]
    status = collections.Counter(e.get("status") for e in rows.values())
    rep = {"export": str(exp), "files_sha256": shas, "n_rows": len(rows), "n_resolved": len(recs),
           "status_counts": dict(status), "generators_checked": [f"{s}/{f}" for s, f in gens],
           "index_rows_indexed": idx["n_rows"], "d1b_templates_sha256": d1b["templates_sha256"],
           "cohorts": {"v2": v2.identity()["sha256"], "v3": v3.identity()["sha256"]},
           "failures": {k: {"n": len(v), "examples": v[:10],
                            "ids": sorted({x[0] if isinstance(x, tuple) else str(x) for x in v})} for k, v in fails.items()},
           "failures_by_source": {k: dict(collections.Counter((x[0] if isinstance(x, tuple) else str(x)).split(":")[1]
                                                              for x in v if ":" in str(x)))
                                  for k, v in fails.items()},
           "pass": not fails}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "n_resolved", "pass", "failures", "files_sha256")}, indent=1,
                     ensure_ascii=False))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
