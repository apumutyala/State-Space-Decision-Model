"""Row-level gates re-run on the FINAL bschema-draw-jev1 records (independent of the build's own check stage).

  cohort        m2-dev-cohort-v2 held-out roles + cohort v3 (G13 retention, 498efa61...): id, source item, normalised
                content, key-free text overlaps (data.b1.exclusion_sets) -> must be 0
  index_fp      production data.dedup.Fingerprint on each row's item-tier texts -> 0 hits
  frozen        harness.heldout.check_files (the 9 frozen benchmarks never named) + check_generator on every
                data.jev1.GEN_SPECS entry -> 0 refused
  d1b           train.heldout.scan_records (held-out D1-B steer templates) -> 0 hits
  gold          gold-once on every row; noul labels in {true, false}; score labels in range
  shape         every row: ssdm-train-v0, one question 'q', type in choice / noul / score, jevkit Item-valid
                (labels '0'..'K-1' for score, criteria keys for choice, no answer key in the state)

    python -X utf8 -I scripts/jev1_gates.py --records <records.jsonl.gz> --out <gates-rows.json>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None) -> int:
    from data import config, jev1
    from data.b1 import dedup_keys, exclusion_sets
    from data.b_schema import root_items
    from data.dedup import Fingerprint
    from jev1_build import gold_once, item_probe
    from jevkit.items import Item
    from train.cohort import record_content_hash, record_text_keys
    from train.cohort_v3 import _ensure_kit
    from train import heldout as D1

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generator", default="data.jev1", help="module whose GEN_SPECS are checked (data.jev2 for jev2)")
    a = ap.parse_args(argv)
    import importlib
    gen_mod = importlib.import_module(a.generator)
    _ensure_kit()
    from harness import heldout as H

    ex = exclusion_sets()
    fp = Fingerprint.load(config.FINGERPRINT)
    fp.assert_production()
    c = collections.Counter()
    bad = collections.defaultdict(list)
    n = 0
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n += 1
            rid = r["id"]
            if rid in ex["ids"]:
                c["cohort_id"] += 1
                bad["cohort_id"].append(rid)
            if root_items(r) & ex["items"]:
                c["cohort_item"] += 1
                bad["cohort_item"].append(rid)
            if record_content_hash(r) in ex["content"]:
                c["cohort_content"] += 1
                bad["cohort_content"].append(rid)
            if record_text_keys(r) & ex["text"]:
                c["cohort_text"] += 1
                bad["cohort_text"].append(rid)
            if any(fp.check(*item_probe(r)).values()):
                c["index_fp"] += 1
                bad["index_fp"].append(rid)
            if not gold_once(r):
                c["gold_once"] += 1
                bad["gold_once"].append(rid)
            q = r["questions"].get("q")
            ok = r.get("schema") == "ssdm-train-v0" and list(r["questions"]) == ["q"] and q["type"] in (
                "choice", "noul", "score")
            if ok:
                try:
                    lab = r["labels"]["q"]
                    if q["type"] == "noul":
                        labels, exp = ["no", "yes"], ("yes" if lab == "true" else "no")
                    elif q["type"] == "score":
                        labels, exp = [str(i) for i in range(len(q["criteria"]))], int(lab)
                    else:
                        labels, exp = list(q["criteria"]), lab
                    Item(id=rid, family=r["family"], state=r["state"], question=q, labels=labels, expected=exp,
                         split="train").validate()
                except Exception:  # noqa: BLE001
                    ok = False
            if not ok:
                c["shape"] += 1
                bad["shape"].append(rid)
            _ = dedup_keys
    frozen = H.check_files([a.records])
    gens = {}
    for k, spec in gen_mod.GEN_SPECS.items():
        H.check_generator(spec)
        gens[k] = "ok"
    d1 = D1.scan_records(a.records)
    d1["n_hits"] = len(d1.get("hits") or [])
    rep = {"records": a.records, "generator": a.generator, "n": n, "fails": dict(c),
           "fail_ids": {k: v[:50] for k, v in bad.items()},
           "cohort_identity": ex["identity"],
           "index_fingerprint": {k: fp.meta.get(k) for k in ("version", "npz_sha256", "arrays_sha256", "rows")},
           "harness_heldout": frozen, "generators": gens,
           "d1b_heldout": {k: v for k, v in d1.items() if k not in ("hits_examples",)}}
    rep["pass"] = (not c and not frozen["refused"] and not (d1.get("n_hits") or d1.get("hits")))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "fails", "pass")}, indent=1))
    print(json.dumps({"harness_heldout_refused": frozen["refused"], "d1b": {k: v for k, v in d1.items()
                                                                            if not isinstance(v, (list, dict))}}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
