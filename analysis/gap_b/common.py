"""Shared loaders for the B gap breakdown (CPU only). Prediction files are data (json.loads), never executed.
The frozen scorer (harness.score) is only imported and called."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = r"C:\Users\apumu\research\CHLabs\ssdm-data\decision-index"
for p in (str(ROOT), KIT):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import score  # noqa: E402  (frozen; only called)

SUBSET = ROOT / "jobs" / "common" / "dev_subset_m1.json"
DEV_ROWS = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\slices\dev-rows.jsonl.gz")
FROZEN = {9, 20, 22, 31, 32, 33, 44, 58, 64}
U = Path(r"C:\Users\apumu")
R3 = U / "r3_fgu2c_land" / "work2" / "x" / "out" / "index"
SOURCES = {
    "d1": U / "rescore_c7pq2" / "dl" / "x_final" / "out" / "units-child-index.json",
    "d1_parent": U / "rescore_c7pq2" / "dl" / "x_final" / "out" / "parent_arm" / "units-index.json",
    "clef": U / "research" / "CHLabs" / "review-artifacts" / "rwkv7-baseline-preserve-20261007" / "clef-flash-job-sgihj-results.jsonl",
}
for v in ("v1", "v2", "v2f", "v3", "v3c", "v3p", "v3r", "v3w"):
    SOURCES[v] = R3 / v / "results.jsonl"


def load_results(p):
    p = Path(p)
    out = {}
    if p.suffix == ".json":
        d = json.load(open(p, encoding="utf-8"))["index_results"]
        for rid, r in d.items():
            r = dict(r)
            r.setdefault("run_id", rid)
            out[rid] = r
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["run_id"]] = r
    return out


def load_rows():
    ids = set(json.load(open(SUBSET, encoding="utf-8"))["run_ids"])
    rows = []
    with gzip.open(DEV_ROWS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["_evaluation"]["run_id"] in ids:
                rows.append(r)
    assert len(rows) == len(ids), len(rows)
    return rows


def index_of(rows, res):
    return score.point(score.decompose(rows, res))
