"""Data levers 2026-10-09, step 2 (training side, CPU only): per-source shape of the D16 and D17 training draws.

D16 = b1-draw100k-v2 minus the 5,201 `aug.injection.family == "d1a"` rows (`train_set.attack_clean`; the 14-id doc_clean
pin is not applied here, <0.02%). D17 = bschema-draw230k-v2e (whole pool). Records are read as JSON data only.
Per (source, family): records, scored items (questions), item share, K histogram, gold POSITION rate (index of the gold
key in the criteria order; options are shown in that order), semantic label rates for named label sets, noul true rate,
instruction length (chars; ~4 chars/token).

    python -X utf8 -I analysis/data_levers_train.py OUT.json
"""
import collections
import gzip
import json
import statistics
import sys
from pathlib import Path

BUILD = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build")
DRAWS = {"D16": (BUILD / "b1-draw100k-v2" / "records.jsonl.gz", True),
         "D17": (BUILD / "bschema-draw230k-v2e" / "records.jsonl.gz", False)}
GENERIC_IDS = {"A", "B", "C", "D", "E", "F", "a", "b", "c", "d"}


def generic(keys):
    return all(k in GENERIC_IDS or k.isdigit() or k.startswith("option_") for k in keys)


out = {}
for draw, (p, attack_clean) in DRAWS.items():
    S = {}
    n_rec = n_items = 0
    for line in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(line)
        inj = (r.get("aug") or {}).get("injection") or {}
        if attack_clean and inj.get("family") == "d1a":
            continue
        n_rec += 1
        s = S.setdefault(f"{r['source']}/{r['family']}", {"records": 0, "items": 0, "qtype": collections.Counter(),
                         "K": collections.Counter(), "pos": collections.Counter(), "sem": collections.Counter(),
                         "noul": collections.Counter(), "chars": [], "split": collections.Counter()})
        s["records"] += 1
        s["split"][str(r.get("source_split"))] += 1
        for key, q in r["questions"].items():
            lab = r["labels"].get(key)
            if lab is None:
                continue
            n_items += 1
            s["items"] += 1
            t = q.get("type")
            s["qtype"][t] += 1
            if len(s["chars"]) < 4000:
                s["chars"].append(len(q.get("instructions") or ""))
            if t == "choice":
                keys = list((q.get("criteria") or {}).keys())
                K = len(keys)
                s["K"][K] += 1
                if str(lab) in keys:
                    i = keys.index(str(lab))
                    s["pos"][f"K{K}:p{i}" if K <= 4 else f"K{K}:{'first' if i == 0 else 'last' if i == K - 1 else 'mid'}"] += 1
                if not generic(keys):
                    s["sem"][str(lab)] += 1
            elif t == "noul":
                s["noul"]["true" if lab in (True, 1, 1.0, "true") else "false"] += 1
    fam = {}
    for k, s in S.items():
        ch = s["chars"]
        fam[k] = dict(records=s["records"], items=s["items"], item_share=round(s["items"] / n_items, 5),
                      qtype=dict(s["qtype"]), K=dict(sorted(s["K"].items())), maxK=max(s["K"]) if s["K"] else 0,
                      pos=dict(sorted(s["pos"].items())), sem=dict(s["sem"].most_common(12)), noul=dict(s["noul"]),
                      chars_median=int(statistics.median(ch)) if ch else 0,
                      chars_p95=int(sorted(ch)[int(0.95 * (len(ch) - 1))]) if ch else 0,
                      frac_over_8k_chars=round(sum(c > 8000 for c in ch) / len(ch), 4) if ch else 0,
                      split=dict(s["split"]))
    out[draw] = {"records": n_rec, "items": n_items, "families": fam}
    print(draw, n_rec, n_items)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
