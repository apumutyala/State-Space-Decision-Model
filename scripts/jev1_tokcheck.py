"""Token lengths of every row under the 4B host's TRAIN format: the H2O-Lightning shim template
``jrt-h2o-shim:qwen3.5-4b`` (decide/shim_format.py on stack/qwen35/sizecliff, PREREG sizecliff Amendment 1), bound
to the pinned Qwen3.5-4B tokenizer (spec identity checked against ``PINNED_SPEC_SHA256``, fail-closed).

``ShimSpec.row`` is the trainer's own row builder (chat template, shim system text, "record: ... / question: ... /
options:", the "Answer:" prefill, one label token); a row it refuses (control token, layout) is reported as an error.
The length is ``len(ids) + 1`` (prompt + the one answer token).

    python -X utf8 -I scripts/jev1_tokcheck.py --shim-repo <sizecliff checkout> --records <pool.jsonl.gz>
        --out <lens.jsonl> [--shard i/n]
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

TEMPLATE = "jrt-h2o-shim:qwen3.5-4b"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shim-repo", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.shim_repo).resolve()))
    from decide.shim_format import PINNED_SPEC_SHA256, bind_template

    spec = bind_template(TEMPLATE)
    ident = spec.identity()
    si, sn = map(int, a.shard.split("/"))
    n = 0
    with gzip.open(a.records, "rt", encoding="utf-8") as fh, open(a.out, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(json.dumps({"_identity": ident, "_pinned": PINNED_SPEC_SHA256[TEMPLATE]}) + "\n")
        for i, line in enumerate(fh):
            if i % sn != si:
                continue
            r = json.loads(line)
            req = {"state": r["state"], "questions": r["questions"]}
            try:
                row = spec.row(req, "q")
                d = {"id": r["id"], "t_shim": len(row.ids) + 1}
            except Exception as e:  # noqa: BLE001 -- reported, the row is dropped downstream
                d = {"id": r["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
            fo.write(json.dumps(d) + "\n")
            n += 1
    print(json.dumps({"n": n, "identity": ident}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
