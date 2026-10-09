"""Finalize the bschema-draw-v3 lever rows: keep rows <= 2,048 tokens under v2f AND v3w (trainer 9c3e435 token check,
``lens-*.jsonl``), trim BANKING77 K77 rows to the D16-share count, and write a pointer-pinned pool dir
(records / manifest / content, ``pointer.json``, ``target_provenance.json``) for ``scripts/b_schema_swap_pool.py``.

    python -X utf8 -I scripts/b_schema_levers_finalize.py --work <dir with levers.jsonl.gz + lens-*.jsonl>
        --parent-n <parent rows> --out <pool dir>
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
T_MAX = 2048


def main(argv=None) -> int:
    from data.b_schema import canonical, gz_text, sha
    from train.cohort import record_content_hash

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--parent-n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refuse-ids", default=None, help="ids refused by scripts/b_schema_eval_catalog_overlap.py")
    ap.add_argument("--k77-margin", type=int, default=0,
                    help="extra K77 rows kept for rows the P3 checker / doc gate drop later (final share >= D16's)")
    a = ap.parse_args(argv)
    w = Path(a.work)
    lens = {}
    for f in sorted(glob.glob(str(w / "lens-*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            lens[d["id"]] = d
    prov = json.loads((w / "target_provenance.json").read_text(encoding="utf-8"))
    refused = set(Path(a.refuse_ids).read_text(encoding="utf-8").split()) if a.refuse_ids else set()
    rows, k77 = [], []
    stats = collections.defaultdict(collections.Counter)
    kdist = collections.defaultdict(collections.Counter)
    with gzip.open(w / "levers.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            r = json.loads(line)
            lv = r["aug"]["lever"]["lever"]
            d = lens.get(r["id"])
            if d is None or "error" in d:
                raise SystemExit(f"{r['id']}: no token length ({d})")
            if r["id"] in refused:
                stats[lv]["eval_catalog_overlap_refused"] += 1
                continue
            if max(d["t_v2f"], d["t_v3w"]) > T_MAX or d.get("cont", 0) > 16:
                stats[lv]["over_2048"] += 1
                continue
            stats[lv]["fit"] += 1
            if lv == "banking_k77":
                k77.append((r, line))
            else:
                rows.append((r, line))
                kdist[lv][len(next(iter(r["questions"].values())).get("criteria") or {})] += 1
    if "banking_k77" in prov:
        c = prov["banking_k77"]["count_rule"]
        s, have = c["k77_row_share"], c["parent_k77_rows"]
        n_final_wo = a.parent_n + len(rows)
        need = max(0, math.ceil((s * n_final_wo - have) / (1 - s))) + a.k77_margin
        if need > len(k77):
            raise SystemExit(f"need {need} K77 rows, only {len(k77)} fit")
        rows += k77[:need]
        stats["banking_k77"]["kept_after_trim"] = need
        kdist["banking_k77"][77] = need
        c.update({"rows_without_new_k77": n_final_wo, "k77_rows_kept": need, "k77_margin": a.k77_margin,
                  "final_k77_share": round((have + need) / (n_final_wo + need), 5)})
    for lv, st in stats.items():
        prov.setdefault(lv, {})["token_check"] = dict(st)
        prov[lv]["K_distribution"] = dict(sorted(kdist[lv].items()))
    rows.sort(key=lambda x: x[0]["id"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_h, man_h, con_h = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    name = out.name
    with gz_text(out / "records.jsonl.gz") as fr, gz_text(out / f"{name}.manifest.jsonl.gz") as fm, \
            gz_text(out / f"{name}.content.jsonl.gz") as fc:
        for r, line in rows:
            fr.write(line + "\n")
            rec_h.update((line + "\n").encode("utf-8"))
            m = canonical({"family": r["family"], "id": r["id"], "sha256": sha(line), "source": r["source"]}) + "\n"
            fm.write(m)
            man_h.update(m.encode("utf-8"))
            cc = canonical({"content_sha256": record_content_hash(r), "id": r["id"]}) + "\n"
            fc.write(cc)
            con_h.update(cc.encode("utf-8"))
    (out / "target_provenance.json").write_bytes((json.dumps(prov, indent=1) + "\n").encode("utf-8"))
    ptr = {"build": f"{name} (data/b_schema_levers.py + scripts/b_schema_levers_finalize.py)", "n_records": len(rows),
           "records_uncompressed_sha256": rec_h.hexdigest(), "manifest": f"{name}.manifest.jsonl.gz",
           "manifest_uncompressed_sha256": man_h.hexdigest(), "content_map": f"{name}.content.jsonl.gz",
           "content_map_uncompressed_sha256": con_h.hexdigest(),
           "target_provenance_sha256": hashlib.sha256((out / "target_provenance.json").read_bytes()).hexdigest(),
           "templates_verified": ["jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"]}
    (out / "pointer.json").write_bytes((json.dumps(ptr, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"n": len(rows), "stats": {k: dict(v) for k, v in stats.items()},
                      "K": {k: dict(sorted(v.items())) for k, v in kdist.items()}, "pointer_sha256":
                      hashlib.sha256((out / "pointer.json").read_bytes()).hexdigest()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
