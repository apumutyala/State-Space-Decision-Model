"""Training-layout row lengths of the 2,008-row Index DEV subset (CPU, no model), committed as
``jobs/m2_first_cell/index_lengths_rwkv7.json`` (run ids and token counts only; no row content).

    python -X utf8 -m jobs.m2_first_cell.index_lengths build --rows <dev-subset-rows.jsonl.gz> --tokenizer-dir <dir>

Per row: the LONGEST question row the trained head reads, ``len(heads.layout.build_row(...).ids)`` under the cell's
template and layout with the RWKV7-G1k tokenizer (the exact length ``train.run_cell.keep_item`` compares with T). The
report splits the child - parent Index difference at T = 2,048 (the longest training length) by kit GROUP (a group's
length is its longest row), so the split never cuts a group. Built locally once ($0); the job only reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "index_lengths_rwkv7.json"
CELL = HERE / "m2_first_cell_rwkv7.json"
SPLIT_T = 2048


class MemoEncoder:
    """Segment -> ids cache over an encoder (a row's state is the same segment for each of its questions)."""

    def __init__(self, enc):
        self.enc, self.memo = enc, {}

    def identity(self):
        return self.enc.identity()

    def __call__(self, text, trusted=True):
        k = (text, trusted)
        if k not in self.memo:
            self.memo[k] = self.enc(text, trusted=trusted)
        return self.memo[k]


def row_length(row: dict, enc, template: str, layout: str) -> int | None:
    from heads.layout import build_row

    req = {"model": "ssdm-m2", "state": row["state"], "questions": row["questions"]}
    try:
        return max(len(build_row(req, q, enc, template=template, layout=layout).ids) for q in row["questions"])
    except ValueError:  # ControlTokenError etc.: the row cannot be laid out; the head refuses it too
        return None


def build(rows_path: str, tokenizer_dir: str) -> dict:
    from harness.suite import read_rows
    from jobs.common.dev_subset import load_subset
    from train.run_cell import parent_encoder

    cfg = json.loads(CELL.read_text(encoding="utf-8"))
    sub = load_subset()
    rows = read_rows(rows_path)
    if sorted(r["_evaluation"]["run_id"] for r in rows) != sorted(sub["run_ids"]):
        raise SystemExit("rows are not exactly the pre-registered subset")
    enc = MemoEncoder(parent_encoder("rwkv7", tokenizer_dir))
    lengths = {}
    for r in rows:
        enc.memo.clear()
        lengths[r["_evaluation"]["run_id"]] = row_length(r, enc, cfg["template"], cfg["layout"])
    tok = json.dumps(enc.identity(), sort_keys=True)
    return {"_comment": __doc__.split("\n\n")[0], "subset_ids_sha256": sub["ids_sha256"], "template": cfg["template"],
            "layout": cfg["layout"], "encoder_sha256": hashlib.sha256(tok.encode()).hexdigest(), "split_t": SPLIT_T,
            "n_rows": len(lengths), "n_over_split": sum(1 for v in lengths.values() if v is not None and v > SPLIT_T),
            "n_unmeasurable": sum(v is None for v in lengths.values()), "lengths": dict(sorted(lengths.items()))}


def load(path: Path = OUT) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = json.loads(CELL.read_text(encoding="utf-8"))
    if d["template"] != cfg["template"] or d["layout"] != cfg["layout"]:
        raise ValueError("index lengths were built for another template / layout")
    return d


def group_bins(rows: list[dict], lengths: dict[str, int | None], split_t: int = SPLIT_T) -> dict[str, list[dict]]:
    """Rows by kit group length (the group's longest row): ``<=T`` / ``>T`` / ``unmeasurable``."""
    glen: dict = {}
    for r in rows:
        e = r["_evaluation"]
        k = (e["catalog_id"], str(e["group_id"]))
        L = lengths.get(e["run_id"])
        glen[k] = None if (L is None or glen.get(k, 0) is None) else max(glen.get(k, 0), L)
    out: dict[str, list[dict]] = {f"<={split_t}": [], f">{split_t}": [], "unmeasurable": []}
    for r in rows:
        e = r["_evaluation"]
        L = glen[(e["catalog_id"], str(e["group_id"]))]
        out["unmeasurable" if L is None else (f"<={split_t}" if L <= split_t else f">{split_t}")].append(r)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.index_lengths")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rows", required=True)
    b.add_argument("--tokenizer-dir", required=True)
    b.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    d = build(a.rows, a.tokenizer_dir)
    Path(a.out).write_text(json.dumps(d, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: d[k] for k in ("n_rows", "n_over_split", "n_unmeasurable", "subset_ids_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
