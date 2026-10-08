"""Gold-option position histogram of m2 training records, in record order vs canonical (prompt) order.

Record lines are written key-sorted (canonical JSON), so the criteria order read back from a record IS canonical
order: the build-time shuffle survives only through the relabelled ids. The PR #16 builder shuffles choice options and relabels their ids at BUILD time (data.augment._choice_question).
Every RWKV7 training template (jrt-rwkv-g1-v1 and jrt-rwkv-g1-v2) presents options in ``canonical_options`` order
(choice: sorted by id), so the gold position the model sees is the gold's canonical index, the same for v1 and v2.
This counts, per option-count bucket (and per id style), where the gold sits:

  * record    : index of the gold in the record's criteria order (the build-time shuffle order)
  * canonical : index of the gold in canonical_options order (what v1 AND v2 render and score)

Usage: python -X utf8 scripts/gold_position_hist.py <records.jsonl.gz> [--limit N] [--out hist.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decide.schema import canonical_options  # noqa: E402


def bucket(k: int) -> str:
    return str(k) if k <= 4 else ("5-16" if k <= 16 else ">16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hist: dict[str, dict[str, Counter]] = defaultdict(lambda: {"record": Counter(), "canonical": Counter()})
    by_style: dict[str, dict[str, Counter]] = defaultdict(lambda: {"record": Counter(), "canonical": Counter()})
    n_rec = 0
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rec += 1
            styles = rec.get("aug", {}).get("id_styles") or {}
            for qid, gold in rec["labels"].items():
                q = rec["questions"][qid]
                gold = str(gold).lower() if q["type"] == "noul" else str(gold)
                canon = [o for o, _ in canonical_options(q)]
                record = canon if q["type"] != "choice" else [str(o) for o in q["criteria"]]
                k = len(canon)
                if k > 4:
                    # k > 4 pools several k: positions binned to fifths of k (0 = first fifth ... 4 = last): uniform ~20 %
                    rpos, cpos = (record.index(gold) * 5) // k, (canon.index(gold) * 5) // k
                else:
                    rpos, cpos = record.index(gold), canon.index(gold)
                b = f"{q['type']}:{bucket(k)}"
                hist[b]["record"][rpos] += 1
                hist[b]["canonical"][cpos] += 1
                if q["type"] == "choice":
                    st = styles.get(qid) or rec.get("aug", {}).get("id_style", "?")
                    by_style[f"{st}:{bucket(k)}"]["record"][rpos] += 1
                    by_style[f"{st}:{bucket(k)}"]["canonical"][cpos] += 1
            if a.limit and n_rec >= a.limit:
                break

    def table(h):
        out = {}
        for b in sorted(h):
            n = sum(h[b]["record"].values())
            out[b] = {"n": n, **{view: {str(p): round(c / n, 4) for p, c in sorted(h[b][view].items())}
                                 for view in ("record", "canonical")}}
        return out

    res = {"records": a.records, "n_records": n_rec, "by_type_k": table(hist), "choice_by_id_style_k": table(by_style),
           "note": "k>4 positions are fifths of k; v1 and v2 both render canonical order, so canonical = v1 = v2"}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
