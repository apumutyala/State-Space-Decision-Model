"""Independent re-check of the 150k build, from disk only (not from the build's in-memory state).

    python -X utf8 -I <launcher> receipts/2026-10-06-D-m2-data-150k/recheck.py <build dir> > recheck.json

Re-reads records.jsonl.gz and checks, for every line: its sha256 against the manifest; schema validation
(``augment.validate``); the held-out D1-B 6-gram guard; the content-index guard (``train.guard.SplitGuard`` + pinned
index); and cohort-v2 disjointness by id, by normalized content (recomputed, and equal to the content map) and by
source item (``build.item_keys``), plus ``Cohort.assert_training_ids`` / ``assert_training_content``. Recounts
records per source / bucket / Index area. Reports the sha256 of the uncompressed records, manifest and content map.
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

from ssdm_data import augment as A, injection as I
from ssdm_data.build import AREA, INDEX_AREA_WEIGHTS, Exclusion, content_guard, item_keys
from train.cohort import Cohort, normalized_content_hash
from train.guard import ForbiddenRow


def sha_gz(p: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main(build_dir: str) -> dict:
    d = Path(build_dir)
    n = int(d.name.split("-")[1])
    seed = d.name.split("seed")[1]
    man_p = d / f"sample-{n}-seed{seed}.manifest.jsonl.gz"
    con_p = d / f"sample-{n}-seed{seed}.content.jsonl.gz"
    man = {}
    with gzip.open(man_p, "rt", encoding="utf-8") as f:
        for x in f:
            if x.strip():
                e = json.loads(x)
                man[e["id"]] = e
    cmap = {}
    with gzip.open(con_p, "rt", encoding="utf-8") as f:
        for x in f:
            if x.strip():
                e = json.loads(x)
                cmap[e["id"]] = e["content_sha256"]
    ex = Exclusion.from_cohort()
    guard, gid = content_guard()
    cohort = Cohort()
    c = collections.Counter()
    by_src, by_bucket, areas = collections.Counter(), collections.Counter(), collections.Counter()
    content_seen = collections.Counter()
    ids, bad = [], collections.defaultdict(list)
    with gzip.open(d / "records.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rec = json.loads(line)
            rid = rec["id"]
            ids.append(rid)
            c["lines"] += 1
            if man.get(rid, {}).get("sha256") != hashlib.sha256(line.encode()).hexdigest():
                bad["manifest_mismatch"].append(rid)
            try:
                A.validate(rec)
            except AssertionError:
                bad["validation"].append(rid)
            try:
                I.assert_no_heldout(rec["state"], rec["questions"], rid)
            except Exception:  # noqa: BLE001
                bad["d1b_heldout_6gram"].append(rid)
            try:
                guard.check(rec)
            except ForbiddenRow:
                bad["content_index"].append(rid)
            ch = normalized_content_hash(rec["state"], rec["questions"])
            content_seen[ch] += 1
            if cmap.get(rid) != ch:
                bad["content_map_mismatch"].append(rid)
            if rid in ex.ids:
                bad["cohort_id"].append(rid)
            if ch in ex.contents:
                bad["cohort_content"].append(rid)
            if item_keys(rid) & ex.items:
                bad["cohort_item"].append(rid)
            by_src[rec["source"]] += 1
            by_bucket["multi" if rec["family"] == "multi" else rec["source"]] += 1
            parts = rec["aug"]["parts"] if rec["family"] == "multi" else [rid]
            for p in parts:
                areas[AREA.get(p.split(":")[1] if rec["family"] == "multi" else rec["source"], "unmapped")] += 1
    cohort.assert_training_ids(ids)
    cohort.assert_training_content(content_seen)
    tot = sum(areas.values())
    return {
        "build_dir": str(d), "n_records": c["lines"], "n_manifest": len(man), "n_content_map": len(cmap),
        "unique_ids": len(set(ids)) == len(ids),
        "failures": {k: {"n": len(v), "examples": v[:5]} for k, v in bad.items()},
        "failures_total": sum(len(v) for v in bad.values()),
        "cohort_assert_training_ids": "passed", "cohort_assert_training_content": "passed",
        "cohort": ex.identity, "content_guard": gid,
        "records_uncompressed_sha256": sha_gz(d / "records.jsonl.gz"),
        "manifest_uncompressed_sha256": sha_gz(man_p), "content_map_uncompressed_sha256": sha_gz(con_p),
        "by_source": dict(sorted(by_src.items())), "by_bucket": dict(sorted(by_bucket.items())),
        "area_items": dict(sorted(areas.items())),
        "area_share_pct": {a: round(100 * v / tot, 2) for a, v in sorted(areas.items())},
        "index_area_weight_pct": INDEX_AREA_WEIGHTS,
        "within_set_content_twins": {"groups": sum(1 for v in content_seen.values() if v > 1),
                                     "records": sum(v for v in content_seen.values() if v > 1)},
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
