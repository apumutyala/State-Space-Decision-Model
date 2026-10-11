"""train == serve on a few bschema-draw-jev1 rows under jrt-h2o-shim:qwen3.5-4b, with the option-order check
(A-PREREG-JEV1-4B: the loader must NOT re-sort option keys). Light: one streaming pass keeps <= N rows per family.

For each picked record (``train.run_cell.to_train_record``, as the cell trains it) and its question:
  train   = ``train.data.items_from_records`` with the cell's template / layout (the rows run_cell trains on)
  serve   = the vendored H2O shim's own prompt (``Contract.question_prompt``) on the record's request, tokenized
            whole-string by transformers' AutoTokenizer (= vLLM /tokenize), the label of option k = labels[k] in the
            REQUEST's criteria order, its token = the shim's probe token of " <label>"
and requires: ids equal; every option's continuation == its shim label token; the prompt lists the options in the
record's criteria order with labels A, B, C, ...; the gold continuation == the label at the gold's position in that
order; T <= 2,048.

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_jev1_train_serve.py --per-family 5 --out report.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("SSDM_DATA", r"C:\Users\apumu\research\CHLabs\ssdm-data"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(DATA / "m2/build/bschema-draw-jev1/records.jsonl.gz"))
    ap.add_argument("--cell", default=str(ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_jev1_q35_4b.json"))
    ap.add_argument("--per-family", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    import third_party.h2o_lightning_shim as H
    from decide.shim_format import bind_template
    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS
    from heads.layout import hf_segment_encoder
    from train.data import items_from_records
    from train.run_cell import to_train_record

    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    tdir = DEFAULT_TOKENIZERS / BACKBONES["qwen3.5-4b"].tokenizer_dir
    spec = bind_template(cfg["template"], tdir)
    tok = AutoTokenizer.from_pretrained(str(tdir))
    enc = hf_segment_encoder(tok)
    mod, c = H.load_contract()
    picks: dict = defaultdict(list)
    unsorted_seen = Counter()
    with gzip.open(a.records, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fam = r.get("family", "?")
            crit = (r["questions"]["q"].get("criteria") or {})
            unsorted = isinstance(crit, dict) and list(crit) != sorted(crit)
            unsorted_seen[fam] += unsorted
            # prefer rows whose option keys are NOT in sorted order (the case a re-sort would change)
            if len(picks[fam]) < a.per_family and (unsorted or unsorted_seen[fam] == 0):
                picks[fam].append(r)
    rows, mism = 0, []
    for fam, recs in sorted(picks.items()):
        for raw in recs:
            rec = to_train_record(raw)
            for it in items_from_records([rec], enc, layout=cfg["layout"], template=cfg["template"]):
                rows += 1
                q = rec["request"]["questions"]["q"]
                text, names, _, labels = c.question_prompt(mod.render_state(rec["request"]["state"], c.state_compact),
                                                          q, "q")
                want_ids = tok(text, add_special_tokens=False).input_ids
                lab_of = dict(zip(names, labels))
                errs = []
                if list(it.row.ids) != want_ids:
                    errs.append("ids != the shim prompt")
                if [list(x) for x in it.row.continuations] != [[spec.label_ids[lab_of[o]]] for o in it.row.option_ids]:
                    errs.append("continuations != the shim label tokens")
                if q["type"] == "choice":
                    order = list(q["criteria"])
                    if names != order:
                        errs.append("shim option order != the record's criteria order")
                    if labels != c.labels[: len(order)]:
                        errs.append("labels are not A, B, C, ... in order")
                    pos = [text.index(f"\n{lab}) {n}: ") for lab, n in zip(labels, names)]
                    if pos != sorted(pos):
                        errs.append("the prompt does not list the options in the record's order")
                gold = it.row.option_ids[it.label]
                if it.row.continuations[it.label] != [spec.label_ids[lab_of[gold]]]:
                    errs.append("gold continuation != the label of the gold's position")
                if len(it.row.ids) > 2048:
                    errs.append(f"T {len(it.row.ids)} > 2048")
                if errs:
                    mism.append({"id": it.id, "errors": errs})
    rep = {"template": cfg["template"], "spec_identity": spec.identity(), "rows": rows,
           "families": {k: len(v) for k, v in picks.items()},
           "unsorted_option_rows_seen": dict(unsorted_seen),
           "unsorted_rows_checked": sum(1 for v in picks.values() for r in v
                                        if isinstance(r["questions"]["q"].get("criteria"), dict)
                                        and list(r["questions"]["q"]["criteria"]) != sorted(r["questions"]["q"]["criteria"])),
           "mismatches": len(mism), "mismatch_examples": mism[:5]}
    rep["ok"] = rows > 0 and not mism and rep["unsorted_rows_checked"] > 0
    s = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes((s + "\n").encode("utf-8"))
    print(s)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
