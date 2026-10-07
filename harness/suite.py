"""Loading the 0.2.1 suite through the official kit, plus row subsets."""

import collections
import hashlib
import json
from functools import lru_cache

from decision_index import editions
from decision_index.scoring import index02
from decision_index.suite.io import Suite, dumps, sha256_file

from harness import config


@lru_cache(maxsize=None)
def _spec(edition):
    if edition == "0.3":
        raw = config.INDEX_03_SPEC.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != config.INDEX_03_SPEC_SHA256:
            raise RuntimeError(f"vendored 0.3 index spec hash {got} != pinned {config.INDEX_03_SPEC_SHA256}")
        return json.loads(raw.decode("utf-8"))
    return index02.spec(edition)


def spec(edition=None):
    """Index spec (areas, weights, gold, chance, track_scored, ...) for an edition (default config.EDITION)."""
    return _spec(config.edition_id(edition))


def index_ids(edition=None):
    return [n for a in spec(edition)["areas"] for n in a["benchmarks"]]


def area_of(edition=None):
    return {n: a["id"] for a in spec(edition)["areas"] for n in a["benchmarks"]}


def in_edition(row, edition=None):
    """Does a suite row count in `edition`'s public index?

    0.3 (kit PR #74 `editions.in_edition`): ForecastBench (48) is retired and GSM8K (30) counts only
    through its rebuilt rows (run_id prefix `v0.3-gsm8k:`); the 0.2.1 GSM8K rows do not exist in 0.3."""
    e = config.edition_id(edition)
    ev = row["_evaluation"]
    n = ev["catalog_id"]
    if n in config.RETIRED[e]:
        return False
    prefix = config.REPLACED[e].get(n)
    if prefix is not None and not str(ev["run_id"]).startswith(prefix):
        return False
    return n in set(index_ids(e))


def edition_rows(rows, edition=None):
    """Filter rows (e.g. a 0.2.1 DEV slice) down to the ones that count in `edition`."""
    return [r for r in rows if in_edition(r, edition)]


def load_suite(directory=None):
    return Suite(directory or config.SUITE_DIR, config.EDITION)


def row_sha256(row):
    """Hash of the whole frozen row (incl. gold), canonical kit serialisation."""
    return hashlib.sha256(dumps(row).encode()).hexdigest()


def verify(suite):
    """Official hash check (non-strict) plus our per-catalog row counts."""
    report = suite.verify(strict=False)
    counts = collections.Counter(r["_evaluation"]["catalog_id"] for r in suite.rows(apply_exclusions=True))
    report["scoreable_by_catalog"] = dict(sorted(counts.items()))
    report["missing_index_benchmarks"] = [n for n in index_ids() if n not in counts]
    return report


def index_rows(suite):
    """Scoreable 0.2.1 rows of the 38 index benchmarks (exclusions and read-time subsets applied)."""
    ids = set(index_ids())
    for r in suite.rows(apply_exclusions=True):
        if r["_evaluation"]["catalog_id"] in ids:
            yield r


class RowSuite:
    """Duck-types `decision_index.suite.io.Suite` over an explicit list of rows.

    The official scorer only calls `.rows(apply_exclusions=...)` and reads `.edition`,
    so a slice can be scored by the unmodified kit functions.
    """

    def __init__(self, rows, edition=config.EDITION):
        self._rows = list(rows)
        self.edition = editions.get(edition)
        self.in_edition = editions.in_edition(self.edition["id"])

    def rows(self, apply_exclusions=False):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


def read_rows(path):
    from decision_index.suite.io import read_jsonl

    return list(read_jsonl(path))


def write_rows(path, rows):
    import gzip

    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(dumps(r) + "\n")


def file_hashes(suite):
    out = {}
    for p in suite.row_paths:
        out[p.name] = {"sha256_gz": sha256_file(p), "sha256_uncompressed": sha256_file(p, gunzip=True)}
    return out


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
