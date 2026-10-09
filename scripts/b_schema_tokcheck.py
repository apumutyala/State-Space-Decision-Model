"""Token-length check of ssdm-train-v0 records under the D3 cell renderers (CPU, tokenizer only).

Renders every (record, question) row with the TRAINER's own code (``train.data.items_from_records`` on
``train.run_cell.to_train_record``, layout ``g1v2``) under templates ``jrt-rwkv-g1-v2f`` AND ``jrt-rwkv-g1-v3w`` with the
pinned RWKV7-G1k tokenizer (``train.run_cell.parent_encoder("rwkv7", <tokdir>)``), and writes one line per record:
``{"id", "t_v2f", "t_v3w", "k", "cont"}`` (max row tokens per template, max options, max continuation tokens) or
``{"id", "error"}``. The trainer code comes from ``--trainer-root`` (a checkout of claude/b-train-v3w-template @9c3e435
or a descendant), imported first on sys.path, so the lengths are exactly what that trainer sees.

    set CUDA_VISIBLE_DEVICES= & python -X utf8 -I scripts/b_schema_tokcheck.py --trainer-root <checkout> \
        --tokdir <RWKV7-G1k snapshot> --records <candidates.jsonl.gz> --out <lens.jsonl> [--shard i --nshards n]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TEMPLATES = (("t_v2f", "jrt-rwkv-g1-v2f", "g1v2"), ("t_v3w", "jrt-rwkv-g1-v3w", "g1v2"))
#: --set v1: D1's v1 TRAINING presentation (template jrt-rwkv-g1-v1, layout "ordered"; multi-token JSON option ids)
TEMPLATES_V1 = (("t_v1", "jrt-rwkv-g1-v1", "ordered"),)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-root", required=True)
    ap.add_argument("--tokdir", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--set", choices=("g1v2", "v1"), default="g1v2", help="g1v2: v2f + v3w (default); v1: jrt-rwkv-g1-v1 ordered")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.trainer_root)
    from train.data import items_from_records
    from train.run_cell import parent_encoder, to_train_record

    enc = parent_encoder("rwkv7", a.tokdir)
    n = 0
    with gzip.open(a.records, "rt", encoding="utf-8") as fh, open(a.out, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec["id"]
            if zlib.crc32(rid.encode("utf-8")) % a.nshards != a.shard:
                continue
            res = {"id": rid}
            try:
                tr = to_train_record(rec)
                ck = "cont" if a.set == "g1v2" else "cont_v1"
                for key, tpl, layout in (TEMPLATES if a.set == "g1v2" else TEMPLATES_V1):
                    its = items_from_records([tr], enc, layout=layout, template=tpl)
                    res[key] = max(len(x.row.ids) for x in its)
                    res["k"] = max(x.row.n_options for x in its)
                    res[ck] = max(res.get(ck, 0), max(max(len(c) for c in x.row.continuations) for x in its))
            except Exception as e:  # recorded, never silently dropped: the selector refuses error rows
                res = {"id": rid, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            out.write(json.dumps(res) + "\n")
            n += 1
    print(json.dumps({"shard": a.shard, "rows": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
