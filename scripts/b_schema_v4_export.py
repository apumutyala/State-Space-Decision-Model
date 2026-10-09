"""bschema-draw-v4: export the Amazon ESCI TRAIN shard to compact candidate rows (needs pyarrow; data only).

Reads ``SOURCES_DIR/dl-2026-10-09/esci/train-00000-of-00011.parquet`` (tasksource/esci @8113b17, the official ESCI
``train`` split joined with the product catalogue; sha256 checked against data/manifests/sources.lock.json) and the
Index ESCI rows (suite snapshot: every Amazon-ESCI row, i.e. the 5,000 released-test pairs the Index uses). Keeps
US-locale pairs whose product_id and normalised query are NOT in any Index ESCI row, orders them by
sha256(seed|example_id) and writes the first ``--n`` as ``esci.jsonl.gz`` (fields the Index presentation needs only).
Natural label rates: no label is selected for.

    python -X utf8 -I scripts/b_schema_v4_export.py --src <esci dir> --suite <suite-snapshot dir> --lock <lock json>
        --n 9000 --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SEED = 20261009
FILE = "train-00000-of-00011.parquet"
LOCK_KEY = "esci/" + FILE
FIELDS = (("product_title", "title"), ("product_description", "description"),
          ("product_bullet_point", "bullet_point"), ("product_brand", "brand"), ("product_color", "color"))
LABEL = {"Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I"}
MAX_STATE_CHARS = 4500


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(s)).casefold()))


def main(argv=None) -> int:
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    p = Path(a.src) / FILE
    want = json.loads(Path(a.lock).read_text(encoding="utf-8"))[LOCK_KEY]["sha256"]
    if _sha(p) != want:
        raise SystemExit("esci shard sha drift")
    pids, queries, n_index = set(), set(), 0
    for f in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        with gzip.open(Path(a.suite) / f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if "Amazon-ESCI" not in line[:300]:
                    continue
                r = json.loads(line)
                if r.get("family") != "Amazon-ESCI":
                    continue
                n_index += 1
                pids.add(str(r["metadata"]["product_id"]))
                queries.add(_norm(r["state"]["search_query"]))
    stats = collections.Counter()
    keyed = []
    seen_ex = set()  # the mirror's example/product join repeats some example_ids: the first occurrence is kept
    cols = ["example_id", "query", "query_id", "product_id", "product_locale", "esci_label"] + [c for c, _ in FIELDS]
    pf = pq.ParquetFile(p)
    for b in pf.iter_batches(batch_size=20000, columns=cols):
        for r in b.to_pylist():
            stats["rows"] += 1
            if r["example_id"] in seen_ex:
                stats["dup_example_id"] += 1
                continue
            seen_ex.add(r["example_id"])
            if r["product_locale"] != "us":
                stats["not_us"] += 1
                continue
            if str(r["product_id"]) in pids:
                stats["index_product"] += 1
                continue
            if _norm(r["query"]) in queries:
                stats["index_query"] += 1
                continue
            prod = {k: str(r[c]) for c, k in FIELDS if r[c] not in (None, "") and str(r[c]).strip()}
            if "title" not in prod:
                stats["no_title"] += 1
                continue
            row = {"example_id": int(r["example_id"]), "query": r["query"], "query_id": int(r["query_id"]),
                   "product_id": str(r["product_id"]), "gold": LABEL[r["esci_label"]], "product": prod}
            if len(json.dumps({"search_query": r["query"], "product": prod}, ensure_ascii=False)) > MAX_STATE_CHARS:
                stats["state_over_chars"] += 1
                continue
            stats["eligible"] += 1
            stats[f"eligible_{row['gold']}"] += 1
            keyed.append((hashlib.sha256(f"{SEED}|esci|{row['example_id']}".encode()).hexdigest(), row))
        keyed.sort(key=lambda x: x[0])
        del keyed[a.n:]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted((r for _, r in keyed), key=lambda r: r["example_id"])
    with gzip.GzipFile(out / "esci.jsonl.gz", "wb", mtime=0) as fh:
        fh.write(("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n").encode("utf-8"))
    man = {"input": {"file": f"dl-2026-10-09/esci/{FILE}", "sha256": want, "repo": "tasksource/esci",
                     "revision": "8113b17a5d4099e20243282c926f1bc1a08a4d13", "split": "train"},
           "index_esci_rows": n_index, "index_product_ids": len(pids), "index_queries": len(queries),
           "rule": f"first occurrence per example_id; US locale; product_id and normalised query not in any Index ESCI row; state <= {MAX_STATE_CHARS} "
                   f"chars; first {a.n} by sha256(seed|example_id) (seed {SEED}); natural label rates",
           "stats": dict(stats), "n_out": len(rows),
           "out_label_counts": dict(collections.Counter(r["gold"] for r in rows)),
           "esci.jsonl.gz_sha256": _sha(out / "esci.jsonl.gz")}
    (out / "export-manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
