"""External m2 training set named by a committed pointer (PR #44: ``sample-150000-seed20261006``), verified and drawn
for one cell, with the frozen development cohort (v2) excluded by id AND by normalized content.

The record file, its manifest and its content map live outside git; the pointer
(``data/manifests/<build>.pointer.json``) pins the sha256 of each one's UNCOMPRESSED text. A cell config selects the set
with ``"train_set": {"pointer": "<repo-relative pointer path>"}`` and the job passes ``--train-dir <dir>`` holding the
three files under the names the pointer gives (records: ``records.jsonl.gz``).

What ``draw`` checks before any record reaches a trainer (any failure raises ``TrainSetMismatch`` / ``CohortLeak``):
  1. the three files' uncompressed sha256 equal the pointer;
  2. content-map ids == manifest ids, and the manifest has ``n_records`` ids;
  3. NO record of the set is held out by the cohort (by id or by normalized content). The PR #44 build excluded the cohort
     (and every source item of a held-out record) before its guard, so the expected count is 0; a non-zero count means the
     files are not the reviewed build, and the cell refuses instead of silently filtering;
  4. the draw: ``n_train_records`` ids in sha256(seed:id) order (``train.run_cell.split_ids``);
  5. every line read is verified against the manifest (``train.data.load_m2_records``), every drawn record's content is
     re-hashed and must equal the content map, and ``Cohort.assert_training_ids`` / ``assert_training_content`` pass.
  6. cohort v3 (``train.cohort_v3``: the G13 retention role expanded FROM this set, sha256 0c0309b9...) is removed from
     the pool before the draw, by id, normalized content and source item (``training_exclusion``), whatever cohort the
     cell pins, and the drawn ids are re-checked (``assert_draw_disjoint``). v3 holds nothing among the first 60,000 of
     the seed-0 order, so every seed-0 draw of up to 60,000 records (the first cell's 24,000, q9_feature_cache) is
     unchanged. Records v3 holds are the only cohort-held-out records allowed in the set (check 3).
Evaluation records of v2 roles never come from this set: they stay in the PR #16 sample the cohort was built from.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RECORDS_NAME = "records.jsonl.gz"


class TrainSetMismatch(ValueError):
    """The training-set files are not the ones the pointer pins (or contradict its build receipt)."""


def read_pointer(pointer: str | Path) -> dict[str, Any]:
    p = Path(pointer)
    p = p if p.is_absolute() else ROOT / p
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("records_uncompressed_sha256", "manifest", "manifest_uncompressed_sha256", "content_map",
              "content_map_uncompressed_sha256", "n_records"):
        if k not in d:
            raise TrainSetMismatch(f"pointer {p.name} lacks {k}")
    d["_pointer_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    d["_pointer_path"] = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
    return d


def uncompressed_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def files(train_dir: str | Path, ptr: dict) -> dict[str, Path]:
    d = Path(train_dir)
    return {"records": d / RECORDS_NAME, "manifest": d / ptr["manifest"], "content_map": d / ptr["content_map"]}


def verify_files(train_dir: str | Path, ptr: dict) -> dict[str, str]:
    want = {"records": ptr["records_uncompressed_sha256"], "manifest": ptr["manifest_uncompressed_sha256"],
            "content_map": ptr["content_map_uncompressed_sha256"]}
    got = {}
    for k, p in files(train_dir, ptr).items():
        if not p.exists():
            raise TrainSetMismatch(f"training-set {k} file missing: {p}")
        got[k] = uncompressed_sha256(p)
        if got[k] != want[k]:
            raise TrainSetMismatch(f"training-set {k} uncompressed sha256 {got[k]} != pointer {want[k]}")
    return got


def draw(cfg: dict, train_dir: str | Path, cohort=None, verify: bool = True) -> tuple[dict, list[str], dict]:
    """(drawn records as ``train.run_cell.to_train_record`` dicts keyed by id, training ids in draw order, identity)."""
    from train.cohort import Cohort, read_content_map, record_content_hash
    from train.data import load_m2_records, read_manifest
    from train.run_cell import split_ids, to_train_record

    ptr = read_pointer(cfg["train_set"]["pointer"])
    fs = files(train_dir, ptr)
    sha = verify_files(train_dir, ptr) if verify else {}
    co = cohort if cohort is not None else Cohort(expected_sha256=cfg["cohort_sha256"])
    man = read_manifest(fs["manifest"])
    content = read_content_map(fs["content_map"])
    if set(content) != set(man):
        raise TrainSetMismatch("content-map ids != manifest ids")
    if len(man) != int(ptr["n_records"]):
        raise TrainSetMismatch(f"manifest has {len(man)} ids, pointer says {ptr['n_records']}")
    from train.cohort_v3 import POINTERS, POOL_150K, cohort_v3, training_exclusion

    co3 = cohort_v3()
    pinned_150k = ptr["_pointer_path"] == POINTERS[POOL_150K].relative_to(ROOT).as_posix()
    v3x = training_exclusion(man, content, co3, pool=POOL_150K if pinned_150k else None)
    held, held_c = co.held_out(), co.held_out_content()
    by_id = sorted(i for i in man if i in held and i not in v3x["ids"])
    by_content = sorted(i for i in man if content[i] in held_c and i not in v3x["ids"])
    if by_id or by_content:
        raise TrainSetMismatch(f"training set contains cohort-held-out records ({len(by_id)} by id, {len(by_content)} "
                               f"by content, e.g. {(by_id + by_content)[:3]}): not the reviewed cohort-excluded build")
    excluded = heldout_excluded(cfg)
    unknown = sorted(excluded - set(man))
    if unknown:
        raise TrainSetMismatch(f"heldout_excluded ids not in the set: {unknown[:3]}")
    pool = sorted(set(man) - excluded - v3x["ids"])
    n = int(cfg["n_train_records"])
    if n > len(pool):
        raise TrainSetMismatch(f"n_train_records {n} > pool {len(pool)}")
    tr, _ = split_ids(pool, n, 0, cfg["seed"])
    want = set(tr)
    recs, ident = load_m2_records(fs["records"], man, keep=lambda r: r["id"] in want)
    if ident["lines_read"] != len(man):
        raise TrainSetMismatch(f"records file has {ident['lines_read']} lines, manifest {len(man)}")
    got = {r["id"]: r for r in recs}
    missing = sorted(want - set(got))
    if missing:
        raise TrainSetMismatch(f"{len(missing)} drawn record(s) missing from the records file, e.g. {missing[:3]}")
    drift = [i for i in tr if record_content_hash(got[i]) != content[i]]
    if drift:
        raise TrainSetMismatch(f"{len(drift)} drawn record(s) differ in content from the content map, e.g. {drift[:3]}")
    co.assert_training_ids(tr)
    co.assert_training_content(content[i] for i in tr)
    co3.assert_draw_disjoint(tr, content, records=[got[i] for i in tr])
    from train.heldout import record_text, scan

    rep = scan((i, record_text(got[i])) for i in tr)
    if rep["hits"]:
        raise TrainSetMismatch(f"{len(rep['hits'])} drawn record(s) carry a held-out D1-B fragment, e.g. {rep['hits'][:2]}")
    identity = {"train_set": ptr.get("build"), "pointer": ptr["_pointer_path"], "pointer_sha256": ptr["_pointer_sha256"],
                "files_uncompressed_sha256": sha or "not re-verified (verify=False)", "n_pool": len(pool),
                "heldout_excluded": sorted(excluded),
                "drawn_heldout_scan": {"n_texts": rep["n_texts"], "hits": 0, "templates_sha256": rep["templates_sha256"]},
                "held_out_in_set": {"by_id": 0, "by_content": 0}, "n_drawn": len(tr), "draw_seed": cfg["seed"],
                "drawn_ids_sha256": hashlib.sha256("\n".join(tr).encode()).hexdigest(),
                "lines_manifest_verified": ident["lines_read"], "cohort": co.identity(),
                "cohort_v3_excluded": {"n": v3x["n"], "rule": v3x["rule"], "cohort": v3x["cohort"], "drawn_overlap": 0}}
    return {i: to_train_record(got[i]) for i in tr}, tr, identity


def heldout_excluded(cfg: dict) -> set[str]:
    """Pre-registered record ids excluded from the draw because ``train.heldout`` finds a held-out D1-B fragment in
    them (``cfg["train_set"]["heldout_excluded"]["ids"]``)."""
    return set(((cfg.get("train_set") or {}).get("heldout_excluded") or {}).get("ids") or [])


def heldout_gate(cfg: dict, train_dir: str | Path, out: str | Path | None = None) -> int:
    """The job's held-out phase over the WHOLE training set: ``train.heldout`` hits must be exactly the pre-registered
    exclusions (same template set). 0 = pass; 3 = any other hit, a missing pinned hit, or a template-set change."""
    from train.heldout import scan_records

    ptr = read_pointer(cfg["train_set"]["pointer"])
    rep = scan_records(files(train_dir, ptr)["records"])
    pin = (cfg["train_set"].get("heldout_excluded") or {})
    hit_ids = {h["id"] for h in rep["hits"]}
    rep["pinned_exclusions"] = sorted(heldout_excluded(cfg))
    rep["pinned_templates_sha256"] = pin.get("templates_sha256")
    rep["pass"] = (hit_ids == heldout_excluded(cfg)
                   and (not hit_ids or pin.get("templates_sha256") == rep["templates_sha256"]))
    if out:
        Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_texts", "pass", "templates_sha256", "pinned_exclusions")}
                     | {"hits": sorted(hit_ids)}))
    return 0 if rep["pass"] else 3


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -X utf8 -m train.train_set")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "heldout"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True, help="cell config with a train_set block")
        s.add_argument("--train-dir", required=True)
        s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if a.cmd == "heldout":
        return heldout_gate(cfg, a.train_dir, a.out)
    try:
        got = verify_files(a.train_dir, read_pointer(cfg["train_set"]["pointer"]))
    except TrainSetMismatch as e:
        print(f"train set: {e}", file=sys.stderr)
        return 10
    text = json.dumps({"files_uncompressed_sha256": got, "pointer": cfg["train_set"]["pointer"]}, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
