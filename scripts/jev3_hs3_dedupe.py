"""HelpSteer3 prompt dedupe (coordinator 2026-10-11): drop every jev3 HelpSteer3 row whose prompt (any user turn of its
context, whitespace/case/punctuation-normalised) equals a HelpSteer2 prompt turn (train + preference files; this covers
every prompt jev1 used and jev3's own HelpSteer2 rows), or whose context is shared with an earlier HelpSteer3 row of
another context hash. Writes the drop ids (consumed by jev3_gates_summary.py dropids via drop-hs3.txt).

    python -X utf8 -I scripts/jev3_hs3_dedupe.py --records <checked.jsonl.gz> --out <drop-hs3.txt> --report <json>
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data import jev3  # noqa: E402
from data.jev1 import norm  # noqa: E402


def turns_of(prompt: str) -> list[str]:
    parts = re.split(r"<extra_id_1>(?:User|Assistant)\n?", prompt)
    return [norm(p) for p in parts if norm(p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    hs2 = set()
    for l in gzip.open(jev3.DL1 / "helpsteer2" / "train.jsonl.gz", "rt", encoding="utf-8"):
        hs2.update(turns_of(json.loads(l)["prompt"]))
    for l in gzip.open(jev3.DL1 / "helpsteer2" / "preference.jsonl.gz", "rt", encoding="utf-8"):
        hs2.update(turns_of(json.loads(l)["prompt"]))
    cache = {}
    for l in gzip.open(jev3.RAW / "helpsteer3" / "hs3_cache.jsonl.gz", "rt", encoding="utf-8"):
        d = json.loads(l)
        if "_meta" not in d:
            cache[(d["line"], d["k"])] = d
    drop, n = [], 0
    for l in gzip.open(a.records, "rt", encoding="utf-8"):
        if '"helpsteer3"' not in l:
            continue
        r = json.loads(l)
        if r["source"] != "helpsteer3":
            continue
        n += 1
        m = re.match(r"feedback/train\.jsonl\.gz line (\d+) response([12])$", r["aug"]["jev1"]["prov"]["row_key"])
        d = cache[(int(m.group(1)), int(m.group(2)))]
        user = [norm(t.get("content") or "") for t in d["context"] if t.get("role") == "user"]
        if any(u and u in hs2 for u in user):
            drop.append(r["id"])
    Path(a.out).write_text("\n".join(sorted(drop)) + ("\n" if drop else ""), encoding="utf-8")
    rep = {"hs3_rows": n, "dropped_prompt_in_helpsteer2": len(drop), "helpsteer2_prompt_turns": len(hs2),
           "rule": "any user turn of the HelpSteer3 context equals (jev1.norm) a HelpSteer2 train/preference prompt turn"}
    Path(a.report).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
