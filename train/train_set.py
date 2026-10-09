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
  7. ATTACK GUARD (``train.attack_guard``; operator rule: no training on safety test material). The whole set is scanned
     for every EVAL attack template (official hijack, D1-A steer, D1-B steer, B's heldout_v1 battery; 6-gram or exact
     fragment match after normalisation). Any hit refuses the set (``AttackGuardRefusal``) unless the cell pins the
     cleaned version (``train_set.attack_clean`` = ``train.attack_guard.clean_spec``: rule, templates sha256, count and
     sha256 of the excluded ids) and the hits equal it exactly; then those records leave the pool before the draw.
     The drawn records are re-scanned and must have zero hits. The heldout gate (job exit 33) runs the same check.
  8. DOC-OVERLAP GUARD (``train.doc_overlap_guard``; operator rule: no training on eval DOCUMENTS). The whole set is
     scanned against the 13-gram hash index of every Index eval row's document text (DEV + TEST + held-out). Any
     failing row refuses the set (``DocOverlapRefusal``, job exit 34) unless the cell pins ``train_set.doc_clean``
     (``train.doc_overlap_guard.clean_spec``) and the fails equal it exactly; then those rows leave the pool before the
     draw. The heldout gate returns 4 (job exit 34) when only this check fails.
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


class AttackGuardRefusal(TrainSetMismatch):
    """The training set carries an eval attack template outside a pinned ``attack_clean`` spec."""


from train.doc_overlap_guard import DocOverlapError as _DocOverlapError  # noqa: E402


class DocOverlapRefusal(TrainSetMismatch, _DocOverlapError):
    """The training set carries Index eval document text outside a pinned ``doc_clean`` spec (job exit 34)."""


def doc_clean_pin(cfg: dict) -> dict | None:
    return (cfg.get("train_set") or {}).get("doc_clean") or cfg.get("doc_clean")


def doc_exclusion(pin: dict | None, records_path: str | Path, pre_excluded: set[str] = frozenset()) -> tuple[set[str], dict]:
    """Scan the WHOLE set with ``train.doc_overlap_guard``; return (ids to exclude, summary). Raises
    ``DocOverlapRefusal`` unless the fails are zero or exactly the pinned spec. ``pre_excluded`` ids (pinned D1-B
    exclusions) leave the pool anyway and are not part of the pin. Shared by ``draw`` and the KD teacher ``select``
    so both see the same pool."""
    from train.attack_guard import ids_sha256
    from train.doc_overlap_guard import DocOverlapError, check_pinned, scan_file, summary as dsummary

    try:
        rep = scan_file(records_path)
    except DocOverlapRefusal:
        raise
    except DocOverlapError as e:  # missing eval data / index pin mismatch: fail closed
        raise DocOverlapRefusal(f"doc-overlap guard cannot run: {e}") from e
    pre = set(pre_excluded) & set(rep["fail_ids"])
    if pre:
        rep["fail_ids"] = [i for i in rep["fail_ids"] if i not in pre]
        rep["n_fail"], rep["fail_ids_sha256"] = len(rep["fail_ids"]), ids_sha256(rep["fail_ids"])
    ok, why = check_pinned(rep, pin)
    out = dsummary(rep) | {"rule": rep["rule"], "index_npz_sha256": rep["index_npz_sha256"],
                           "allowlist_sha256": rep["allowlist_sha256"], "excluded_ids_sha256": rep["fail_ids_sha256"],
                           "n_excluded": rep["n_fail"], "allowlist_saved": rep["allowlist_saved"],
                           "n_heldout_excluded_overlap": len(pre), "check": why}
    if not ok:
        raise DocOverlapRefusal(f"doc-overlap guard: {why} (per source {rep['per_source_fail']})")
    return set(rep["fail_ids"]), out


def attack_clean_pin(cfg: dict) -> dict | None:
    return (cfg.get("train_set") or {}).get("attack_clean")


def attack_exclusion(cfg: dict, records_path: str | Path) -> tuple[set[str], dict]:
    """Scan the whole set; return (ids to exclude, scan summary). Raises ``AttackGuardRefusal`` unless the hits are
    zero or exactly the pinned clean spec."""
    from train.attack_guard import check_pinned, ids_sha256, scan_file

    rep = scan_file(records_path)
    pre = heldout_excluded(cfg) & set(rep["hit_ids"])  # pinned D1-B exclusions already leave the pool
    if pre:
        rep["hit_ids"] = [i for i in rep["hit_ids"] if i not in pre]
        rep["n_hit"], rep["hit_ids_sha256"] = len(rep["hit_ids"]), ids_sha256(rep["hit_ids"])
    ok, why = check_pinned(rep, attack_clean_pin(cfg))
    summary = {"rule": rep["rule"], "templates_sha256": rep["templates_sha256"], "n_records": rep["n_records"],
               "n_excluded": rep["n_hit"], "excluded_ids_sha256": rep["hit_ids_sha256"], "per_set": rep["per_set"],
               "per_template": rep["per_template"], "n_heldout_excluded_overlap": len(pre),
               "per_counts_include_heldout_excluded": True, "check": why}
    if not ok:
        raise AttackGuardRefusal(f"attack guard: {why} (per set {rep['per_set']})")
    return set(rep["hit_ids"]), summary


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
    attack_ids, attack = attack_exclusion(cfg, fs["records"])
    doc_ids, doc = doc_exclusion(doc_clean_pin(cfg), fs["records"], excluded)
    pool = sorted(set(man) - excluded - v3x["ids"] - attack_ids - doc_ids)
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
    from train.attack_guard import record_hits

    drawn_attack = [i for i in tr if record_hits(got[i])]
    if drawn_attack:
        raise AttackGuardRefusal(f"{len(drawn_attack)} drawn record(s) carry an eval attack template, e.g. {drawn_attack[:3]}")
    attack["drawn_hits"] = 0
    if want & doc_ids:  # structural: the pool already excludes them
        raise DocOverlapRefusal("drawn records carry Index eval document text")
    doc["drawn_fails"] = 0
    identity = {"train_set": ptr.get("build"), "pointer": ptr["_pointer_path"], "pointer_sha256": ptr["_pointer_sha256"],
                "files_uncompressed_sha256": sha or "not re-verified (verify=False)", "n_pool": len(pool),
                "heldout_excluded": sorted(excluded),
                "drawn_heldout_scan": {"n_texts": rep["n_texts"], "hits": 0, "templates_sha256": rep["templates_sha256"]},
                "held_out_in_set": {"by_id": 0, "by_content": 0}, "n_drawn": len(tr), "draw_seed": cfg["seed"],
                "drawn_ids_sha256": hashlib.sha256("\n".join(tr).encode()).hexdigest(),
                "lines_manifest_verified": ident["lines_read"], "cohort": co.identity(),
                "cohort_v3_excluded": {"n": v3x["n"], "rule": v3x["rule"], "cohort": v3x["cohort"], "drawn_overlap": 0},
                "attack_guard": attack, "doc_overlap_guard": doc}
    return {i: to_train_record(got[i]) for i in tr}, tr, identity


def attack_clean_spec(cfg: dict, records_path: str | Path) -> dict:
    """The ``train_set.attack_clean`` block to pin for this cell: every attack-guard hit outside the cell's pinned D1-B
    exclusions (``python -X utf8 -m train.attack_guard clean-spec --records R --config CELL``)."""
    from train.attack_guard import clean_spec, ids_sha256, scan_file

    rep = scan_file(records_path)
    pre = heldout_excluded(cfg)
    rep["hit_ids"] = [i for i in rep["hit_ids"] if i not in pre]
    rep["n_hit"], rep["hit_ids_sha256"] = len(rep["hit_ids"]), ids_sha256(rep["hit_ids"])
    return clean_spec(rep) | {"hit_ids": rep["hit_ids"]}


def heldout_excluded(cfg: dict) -> set[str]:
    """Pre-registered record ids excluded from the draw because ``train.heldout`` finds a held-out D1-B fragment in
    them (``cfg["train_set"]["heldout_excluded"]["ids"]``)."""
    return set(((cfg.get("train_set") or {}).get("heldout_excluded") or {}).get("ids") or [])


def heldout_gate(cfg: dict, train_dir: str | Path, out: str | Path | None = None) -> int:
    """The job's held-out phase over the WHOLE training set: ``train.heldout`` hits must be exactly the pre-registered
    exclusions (same template set), AND ``train.attack_guard`` hits (every eval attack template) must be zero or exactly
    the pinned ``attack_clean`` spec, AND ``train.doc_overlap_guard`` fails must be zero or exactly the pinned ``doc_clean``
    spec. 0 = pass; 3 = any other D1-B / attack hit, a missing pinned hit, or a template-set change; 4 = only the
    doc-overlap guard refused (job exit 34)."""
    from train.attack_guard import check_pinned, scan_file
    from train.heldout import scan_records

    ptr = read_pointer(cfg["train_set"]["pointer"])
    rep = scan_records(files(train_dir, ptr)["records"])
    pin = (cfg["train_set"].get("heldout_excluded") or {})
    hit_ids = {h["id"] for h in rep["hits"]}
    rep["pinned_exclusions"] = sorted(heldout_excluded(cfg))
    rep["pinned_templates_sha256"] = pin.get("templates_sha256")
    d1b_pass = (hit_ids == heldout_excluded(cfg)
                and (not hit_ids or pin.get("templates_sha256") == rep["templates_sha256"]))
    try:
        _, asum = attack_exclusion(cfg, files(train_dir, ptr)["records"])
        a_ok, a_why = True, asum["check"]
    except AttackGuardRefusal as e:
        asum, a_ok, a_why = {}, False, str(e)
    rep["attack_guard"] = asum | {"pass": a_ok, "check": a_why}
    try:
        _, dsum = doc_exclusion(doc_clean_pin(cfg), files(train_dir, ptr)["records"], heldout_excluded(cfg))
        d_ok, d_why = True, dsum["check"]
    except DocOverlapRefusal as e:
        dsum, d_ok, d_why = {}, False, str(e)
    rep["doc_overlap_guard"] = dsum | {"pass": d_ok, "check": d_why}
    rep["pass"] = d1b_pass and a_ok and d_ok
    if out:
        Path(out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_texts", "pass", "templates_sha256", "pinned_exclusions")}
                     | {"hits": sorted(hit_ids), "attack_guard": {"pass": a_ok, "check": a_why},
                        "doc_overlap_guard": {"pass": d_ok, "check": d_why}}))
    if rep["pass"]:
        return 0
    return 3 if not (d1b_pass and a_ok) else 4  # 4: only the doc-overlap guard refused (job exit 34)


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
