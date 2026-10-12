"""Stream PRM800K phase2_train.jsonl (456 MB, never written to disk) and cache ONE compact human-labelled sample per
MATH-TRAIN problem for the jev3 judge_hard family.

    python -X utf8 -I scripts/jev3_prm_stream.py --splits <math_splits_train.jsonl> --out <prm_cache.jsonl.gz>

- problems: PRM800K `math_splits/train.jsonl` rows whose unique_id starts with "train/" (MATH TRAIN); the 4.5k MATH-test
  problems PRM800K trains on (unique_id "test/...") and the 500 MATH-500 problems are never used.
- samples: not quality-control, not initial-screening; finish_reason in {solution, found_error}; the human rating of each
  pre-generated step is taken from the label step whose completion text equals that pre-generated step (labelling
  follows the saved solution until the first -1); a sample whose labels diverge from the saved steps before the end
  (and before a -1) is skipped.
- one sample per problem: the smallest sha256("jev3|prm|<problem uid>|<line no>") (label-blind, deterministic).
The cache keeps the line number and the sha256 of the raw line, so every row traces back to the pinned file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.request

URL = ("https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/"
       "prm800k/data/phase2_train.jsonl")
POINTER = ("https://raw.githubusercontent.com/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/"
           "prm800k/data/phase2_train.jsonl")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    uid = {}
    for line in open(a.splits, encoding="utf-8"):
        d = json.loads(line)
        if d["unique_id"].startswith("train/"):
            uid[d["problem"].strip()] = d["unique_id"]
    with urllib.request.urlopen(POINTER, timeout=60) as r:
        pointer = r.read().decode("utf-8", "replace")
    best: dict[str, tuple[str, dict]] = {}
    stats = {"lines": 0, "not_math_train": 0, "qc_or_screen": 0, "finish_other": 0, "diverged": 0, "eligible": 0}
    h_all = hashlib.sha256()
    req = urllib.request.Request(URL, headers={"User-Agent": "ssdm-jev3"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        for ln, raw in enumerate(io.TextIOWrapper(resp, encoding="utf-8", newline="\n")):
            h_all.update(raw.encode("utf-8"))
            stats["lines"] += 1
            if not raw.strip():
                continue
            d = json.loads(raw)
            q, lab = d["question"], d["label"]
            u = uid.get(q["problem"].strip())
            if u is None:
                stats["not_math_train"] += 1
                continue
            if d.get("is_quality_control_question") or d.get("is_initial_screening_question"):
                stats["qc_or_screen"] += 1
                continue
            fin = lab.get("finish_reason")
            if fin not in ("solution", "found_error"):
                stats["finish_other"] += 1
                continue
            pre = q.get("pre_generated_steps") or []
            ratings = []
            for i, st in enumerate(lab["steps"]):
                if i >= len(pre):
                    break
                m = [c for c in st["completions"] if c["text"] == pre[i]]
                if not m or m[0]["rating"] is None:
                    break
                ratings.append(int(m[0]["rating"]))
                if ratings[-1] == -1:
                    break
            ok = (fin == "solution" and len(ratings) == len(pre) and -1 not in ratings) or \
                 (fin == "found_error" and ratings and ratings[-1] == -1)
            if not ok:
                stats["diverged"] += 1
                continue
            stats["eligible"] += 1
            key = hashlib.sha256(f"jev3|prm|{u}|{ln}".encode()).hexdigest()
            if u in best and best[u][0] <= key:
                continue
            best[u] = (key, {"line": ln, "line_sha256": hashlib.sha256(raw.rstrip("\n").encode()).hexdigest(),
                             "unique_id": u, "problem": q["problem"], "steps": pre,
                             "answer": q.get("pre_generated_answer"), "ratings": ratings, "finish_reason": fin})
            if ln % 5000 == 0:
                print(json.dumps({"line": ln, **stats, "problems": len(best)}), flush=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as fo:
        meta = {"_meta": {"url": URL, "lfs_pointer": pointer, "stream_sha256": h_all.hexdigest(), "stats": stats,
                          "problems": len(best), "splits_file": a.splits,
                          "splits_sha256": hashlib.sha256(open(a.splits, "rb").read()).hexdigest()}}
        fo.write(json.dumps(meta) + "\n")
        for u in sorted(best):
            fo.write(json.dumps(best[u][1], ensure_ascii=False) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
